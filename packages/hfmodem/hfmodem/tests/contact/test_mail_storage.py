# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import nullcontext
from pathlib import Path

import pytest

from hfmodem.contact import mail_storage


@pytest.mark.parametrize('platform', ['nt', 'posix'])
def test_directory_sync_uses_native_support(monkeypatch, tmp_path, platform):
    calls = Mock()
    calls.open.return_value = 42
    monkeypatch.setattr(mail_storage, 'os', SimpleNamespace(
        name=platform, O_RDONLY=0, open=calls.open, fsync=calls.fsync, close=calls.close))
    mail_storage.sync_directory(tmp_path)
    if platform == 'nt':
        assert calls.mock_calls == []
    else:
        from unittest.mock import call
        assert calls.mock_calls == [call.open(tmp_path, 0), call.fsync(42), call.close(42)]


def test_directory_sync_failure_closes_the_descriptor(monkeypatch, tmp_path):
    calls = Mock()
    calls.open.return_value = 42
    calls.fsync.side_effect = OSError('directory sync failed')
    monkeypatch.setattr(mail_storage, 'os', SimpleNamespace(
        name='posix', O_RDONLY=0, open=calls.open, fsync=calls.fsync, close=calls.close))
    with pytest.raises(OSError, match='directory sync failed'):
        mail_storage.sync_directory(tmp_path)
    calls.close.assert_called_once_with(42)


def test_windows_mailbox_orphan_recovery_syncs_without_changing_record(tmp_path, monkeypatch):
    from hfmodem.contact import varac
    box = varac.Mailbox(tmp_path)
    record = box.store('in', 'W9SSJ', 'K1ABC', varac.VMail('café', '73'))
    path = tmp_path / (record['id'] + '.json')
    before = path.read_bytes()
    (tmp_path / 'index.jsonl').unlink()
    original_open, fsync = Path.open, varac.os.fsync
    sources, synced = [], []

    def opened(source, mode='r', *args, **kwargs):
        stream = original_open(source, mode, *args, **kwargs)
        if source == path and 'b' in mode:
            sources.append(stream)
        return stream

    def sync(fd):
        for stream in sources:
            if not stream.closed and stream.fileno() == fd:
                assert stream.writable()
                synced.append(fd)
        fsync(fd)

    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(varac, 'os', SimpleNamespace(name='nt', fsync=sync))
    assert varac.Mailbox(tmp_path).read(1) == record
    assert len(synced) == 1 and all(stream.closed for stream in sources)
    assert path.read_bytes() == before


@pytest.mark.parametrize('store', ['mailbox', 'collection'])
@pytest.mark.parametrize('failure', [None, 'fsync', 'replace'])
def test_atomic_mail_writes_close_before_replace_and_failure_cleanup(tmp_path, monkeypatch, store, failure):
    from hfmodem.contact import mail_collection, varac
    module = varac if store == 'mailbox' else mail_collection
    opened = {}
    create, replace, unlink = module.tempfile.NamedTemporaryFile, module.os.replace, Path.unlink
    def temporary(*args, **kwargs):
        stream = create(*args, **kwargs)
        opened[Path(stream.name)] = stream
        return stream
    def move(source, destination):
        assert opened[Path(source)].closed
        if failure == 'replace':
            raise OSError('replace failed')
        return replace(source, destination)
    def remove(path, *args, **kwargs):
        if path in opened:
            assert opened[path].closed
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(module.tempfile, 'NamedTemporaryFile', temporary)
    monkeypatch.setattr(module.os, 'replace', move)
    monkeypatch.setattr(Path, 'unlink', remove)
    if failure == 'fsync':
        fsync = module.os.fsync

        def fail_temporary(fd):
            if any(not stream.closed and stream.fileno() == fd for stream in opened.values()):
                raise OSError('fsync failed')
            return fsync(fd)
        monkeypatch.setattr(module.os, 'fsync', fail_temporary)
    with pytest.raises(OSError, match=f'{failure} failed') if failure else nullcontext():
        if store == 'mailbox':
            varac.Mailbox(tmp_path).store('in', 'W9SSJ', 'K1ABC', varac.VMail('café', '73'))
        else:
            mail_collection.MailCollection(SimpleNamespace(mailbox=SimpleNamespace(root=tmp_path)))._save([])
    assert opened and all(stream.closed and not path.exists() for path, stream in opened.items())
    assert len(list(tmp_path.glob('*.json'))) == int(failure is None)


def test_mail_stores_roundtrip_utf8_with_a_non_utf8_locale(tmp_path, monkeypatch):
    import json
    from hfmodem.contact import mail_collection, mail_parking, varac
    original_open = Path.open
    def locale_open(path, mode='r', buffering=-1, encoding=None, *args, **kwargs):
        if 'b' not in mode and encoding in (None, 'locale'):
            encoding = 'cp1252'
        return original_open(path, mode, buffering, encoding, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', locale_open)
    box = varac.Mailbox(tmp_path)
    row = box.store('in', 'W9SSJ', 'K1ABC', varac.VMail('café', 'café\r\n73'))
    assert varac.Mailbox(tmp_path).read(1) == row
    parking = mail_parking.ParkedStore(tmp_path)
    try:
        row, duplicate = parking.accept(owner='W9SSJ', sender='K1ABC', recipient='W9SSJ-1',
            via='K1ABC', subject='café', body='café\r\n73', composed='2026-10-28 12:00:00')
        assert not duplicate
    finally:
        parking.close()
    restarted = mail_parking.ParkedStore(tmp_path)
    try:
        assert restarted.available and restarted.get(row['id']) == row
    finally:
        restarted.close()
    collection = mail_collection.MailCollection(SimpleNamespace(mailbox=box))
    notices = [dict(source='K1ABC', destination='W9SSJ', utc='café', dial_hz=14105000)]
    collection.path.write_text(json.dumps(notices, ensure_ascii=False), encoding='utf-8')
    assert collection._load() == notices
    collection._save(notices)
    assert collection._load() == notices
