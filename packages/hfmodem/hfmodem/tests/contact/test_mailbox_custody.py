# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
from pathlib import Path

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk, answered


def deposit(box, subject='Subject'):
    return box.store('in', 'W9SSJ', 'K1ABC', v.VMail(subject, 'café\r\n73'),
                     composed_utc='2026-10-28 12:00:00')


@pytest.mark.parametrize('boundary', [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('after', [False, True])
def test_fsync_failure_never_returns_custody_and_recovers_committed_record(
        tmp_path, monkeypatch, boundary, after):
    box = v.Mailbox(tmp_path)
    fsync = v.os.fsync
    calls = 0

    def fail(fd):
        nonlocal calls
        calls += 1
        if calls == boundary and not after:
            raise OSError('disk full before fsync')
        fsync(fd)
        if calls == boundary and after:
            raise OSError('disk full after fsync')

    with monkeypatch.context() as patch:
        patch.setattr(v.os, 'fsync', fail)
        with pytest.raises(OSError, match='disk full'):
            deposit(box)
    restarted = v.Mailbox(tmp_path)
    assert len(restarted.entries()) == (0 if boundary <= 3 else 1)
    if boundary > 3:
        assert restarted.read(1)['body'] == 'café\r\n73'
        assert restarted.read(1)['composed_utc'] == '2026-10-28 12:00:00'
    assert not list(tmp_path.glob('.vmail-*'))


@pytest.mark.parametrize('after', [False, True])
def test_replace_failure_recovers_only_complete_record(tmp_path, monkeypatch, after):
    box = v.Mailbox(tmp_path)
    replace = v.os.replace

    def fail(source, destination):
        if after:
            replace(source, destination)
        raise OSError('replace failed')

    with monkeypatch.context() as patch:
        patch.setattr(v.os, 'replace', fail)
        with pytest.raises(OSError, match='replace failed'):
            deposit(box)
    assert len(v.Mailbox(tmp_path).entries()) == int(after)
    assert not list(tmp_path.glob('.vmail-*'))


@pytest.mark.parametrize('mode', ['before', 'partial', 'after'])
def test_index_failure_exposes_orphan_to_duplicate_lookup(tmp_path, monkeypatch, mode):
    box = v.Mailbox(tmp_path)
    append = box._append_index

    def fail(records):
        if mode == 'partial':
            (tmp_path / 'index.jsonl').write_bytes(b'{"id": "unfinished')
        elif mode == 'after':
            append(records)
        raise OSError('index failed')

    with monkeypatch.context() as patch:
        patch.setattr(box, '_append_index', fail)
        with pytest.raises(OSError, match='index failed'):
            deposit(box)
    assert len(box.entries()) == 1
    record = box.read(1)
    assert record['from'] == 'K1ABC' and record['to'] == 'W9SSJ'
    restarted = v.Mailbox(tmp_path)
    assert restarted.read(1) == record
    assert len((tmp_path / 'index.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('tail', [b'{"id": "unfinished', b'\xff\xfe', b''])
def test_recovery_preserves_existing_numbers_and_appends_orphans(tmp_path, tail):
    box = v.Mailbox(tmp_path)
    first = deposit(box, 'first')
    first_line = (tmp_path / 'index.jsonl').read_bytes()
    second = deposit(box, 'second')
    (tmp_path / 'index.jsonl').write_bytes(first_line + tail)
    restarted = v.Mailbox(tmp_path)
    assert [row['id'] for row in restarted.entries()] == [first['id'], second['id']]
    assert restarted.read(1)['subject'] == 'first'
    assert restarted.read(2)['subject'] == 'second'
    assert len(restarted.entries()) == 2


def test_complete_final_index_line_without_newline_is_preserved(tmp_path):
    box = v.Mailbox(tmp_path)
    first = deposit(box)
    path = tmp_path / 'index.jsonl'
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    second = deposit(v.Mailbox(tmp_path), 'second')
    assert [row['id'] for row in box.entries()] == [first['id'], second['id']]


def test_legacy_index_and_record_are_readable_without_renumbering(tmp_path):
    old = dict(id='20261028T010203123456Z-K1ABC', utc='2026-10-28T01:02:03.123+00:00',
               direction='in', **{'from': 'K1ABC', 'to': 'W9SSJ'}, via='K1ABC',
               subject='legacy', body='hello')
    (tmp_path / (old['id'] + '.json')).write_text(json.dumps(old))
    (tmp_path / 'index.jsonl').write_text(json.dumps(v.Mailbox._entry(old)) + '\n')
    box = v.Mailbox(tmp_path)
    deposit(box, 'new')
    assert box.read(1) == old
    assert box.read(2)['subject'] == 'new'


def test_corrupt_complete_index_line_is_preserved(tmp_path):
    path = tmp_path / 'index.jsonl'
    path.write_bytes(b'{bad}\n')
    with pytest.raises(ValueError, match='invalid mailbox index'):
        v.Mailbox(tmp_path).entries()
    assert path.read_bytes() == b'{bad}\n'


def test_mailbox_recovery_ignores_other_stores_and_uncommitted_temporaries(tmp_path):
    for filename in ('parked-mail.json', 'relay-notices.json', '.vmail-interrupted'):
        (tmp_path / filename).write_text('{bad')
    assert v.Mailbox(tmp_path).entries() == []


def test_successful_custody_is_readable_after_restart(tmp_path):
    record = deposit(v.Mailbox(tmp_path))
    assert v.Mailbox(tmp_path).read(1) == record


@pytest.mark.parametrize('boundary', [
    'file_fsync_before', 'file_fsync_after', 'directory_fsync_before', 'directory_fsync_after',
    'index_fsync_before', 'index_fsync_after', 'final_fsync_before', 'final_fsync_after',
    'replace_before', 'replace_after', 'index_before', 'index_partial', 'index_after',
])
def test_inbox_fault_never_sends_smr_and_restart_retry_keeps_one_record(
        tmp_path, monkeypatch, boundary):
    d = answered(mailbox=tmp_path)
    message = '<SM><TME:2026-10-28 12:00:00><FRM:K1ABC><SBJ:Custody><MSG:café\r\n73>'
    fsync, replace, append = v.os.fsync, v.os.replace, d.mailbox._append_index
    calls = 0
    failure_at = dict(file_fsync=3, directory_fsync=4, index_fsync=5, final_fsync=6)

    def fail_fsync(fd):
        nonlocal calls
        calls += 1
        target = failure_at.get(boundary.rsplit('_', 1)[0]) == calls
        if target and boundary.endswith('_before'):
            raise OSError('inbox fsync failure')
        fsync(fd)
        if target and boundary.endswith('_after'):
            raise OSError('inbox fsync failure')

    def fail_replace(source, destination):
        if boundary == 'replace_before':
            raise OSError('inbox replace failure')
        replace(source, destination)
        if boundary == 'replace_after':
            raise OSError('inbox replace failure')

    def fail_index(records):
        if boundary == 'index_partial':
            (tmp_path / 'index.jsonl').write_bytes(b'{"id":"interrupted')
        elif boundary == 'index_after':
            append(records)
        raise OSError('inbox append failure')

    with monkeypatch.context() as patch:
        patch.setattr(v.os, 'fsync', fail_fsync)
        patch.setattr(v.os, 'replace', fail_replace)
        if boundary in ('index_before', 'index_partial', 'index_after'):
            patch.setattr(d.mailbox, '_append_index', fail_index)
        d.receive(message)
    assert v.frame_message('<SMR>') not in d.host.data.sent
    assert v.frame_message('<SMF>') in d.host.data.sent
    assert d.host.client.connected
    d.session.parking.store.close()
    d.host.client.close()
    restarted = answered(mailbox=tmp_path)
    try:
        restarted.receive(message)
        assert restarted.host.data.sent.count(v.frame_message('<SMR>')) == 1
        assert len(restarted.mailbox.entries()) == 1
        saved = restarted.mailbox.read(1)
        assert saved['from'] == 'K1ABC' and saved['body'] == 'café\r\n73'
        assert saved['composed_utc'] == '2026-10-28 12:00:00'
        restarted.receive(message)
        assert len(restarted.mailbox.entries()) == 1
        assert restarted.host.data.sent.count(v.frame_message('<SMR>')) == 2
    finally:
        restarted.session.parking.store.close()
        restarted.host.client.close()


@pytest.mark.parametrize('boundary', [
    'file_fsync_before', 'file_fsync_after', 'replace_before', 'replace_after',
    'directory_fsync_before', 'directory_fsync_after',
])
def test_smr_outcome_disk_failure_reserves_link_and_restart_recovers_authoritative_state(
        tmp_path, monkeypatch, boundary):
    from hfmodem.contact import mail_parking as p
    d = Desk(mailbox=tmp_path)
    for line in ('/vmail park N0XYZ Outcome custody', 'payload', '.'):
        d.session.command(line)
    d.flow.dial = v.slot_frequency(20, 3)
    d.hear('CONNECTED N0XYZ W9SSJ 500')
    assert d.session.parking.store.get(1)['state'] == 'in_flight'
    fsync, replace = p.os.fsync, p.os.replace
    calls = 0

    def fail_fsync(fd):
        nonlocal calls
        calls += 1
        target = (calls == 1 and boundary.startswith('file_fsync') or
                  calls == 2 and boundary.startswith('directory_fsync'))
        if target and boundary.endswith('_before'):
            raise OSError('outcome fsync failure')
        fsync(fd)
        if target and boundary.endswith('_after'):
            raise OSError('outcome fsync failure')

    def fail_replace(source, destination):
        if boundary == 'replace_before':
            raise OSError('outcome replace failure')
        replace(source, destination)
        if boundary == 'replace_after':
            raise OSError('outcome replace failure')

    with monkeypatch.context() as patch:
        patch.setattr(p.os, 'fsync', fail_fsync)
        patch.setattr(p.os, 'replace', fail_replace)
        d.receive('<SMR>')
    assert d.session.mail_ack.state == 'unresolved'
    assert d.session.mail_ack.busy and not d.session.parking.store.available
    before = list(d.host.data.sent)
    d.session.parking.start_delivery()
    d.session.command('/vmail send Cannot pass unresolved owner')
    assert d.host.data.sent == before
    d.session.parking.store.close()
    d.host.client.close()
    restarted = Desk(mailbox=tmp_path)
    committed = boundary in ('replace_after', 'directory_fsync_before', 'directory_fsync_after')
    try:
        row = restarted.session.parking.store.get(1)
        assert row['state'] == ('delivered' if committed else 'uncertain')
        restarted.flow.dial = v.slot_frequency(20, 3)
        restarted.hear('CONNECTED N0XYZ W9SSJ 500')
        sent = [raw for raw in restarted.host.data.sent if b'<SM>' in raw]
        assert len(sent) == int(not committed)
        if not committed:
            restarted.receive('<SMR>')
        row = restarted.session.parking.store.get(1)
        assert row['state'] == 'delivered' and row['attempts'] == (1 if committed else 2)
        assert json.loads(restarted.session.parking.store.path.read_text())['records'] == [row]
    finally:
        restarted.session.parking.store.close()
        restarted.host.client.close()


def test_new_mailbox_directories_sync_their_naming_parents(tmp_path, monkeypatch):
    from hfmodem.contact import mail_storage
    from hfmodem.contact.mail_parking import ParkedStore
    synced = []
    real_sync = mail_storage.sync_directory
    def sync(path):
        synced.append(Path(path))
        real_sync(path)
    monkeypatch.setattr(mail_storage, 'sync_directory', sync)
    root = tmp_path / 'new-parent' / 'mail'
    store = ParkedStore(root)
    try:
        assert store.available
        assert tmp_path in synced and root.parent in synced and root in synced
    finally:
        store.close()


def test_parent_directory_sync_failure_cannot_accept_custody(tmp_path, monkeypatch):
    from hfmodem.contact import mail_storage
    from hfmodem.contact.mail_parking import ParkedStore
    real_sync = mail_storage.sync_directory
    def sync(path):
        if Path(path) == tmp_path:
            raise OSError('parent directory fsync failed')
        real_sync(path)
    monkeypatch.setattr(mail_storage, 'sync_directory', sync)
    root = tmp_path / 'new-mail'
    store = ParkedStore(root)
    assert not store.available
    with pytest.raises(OSError):
        store.accept(owner='W9SSJ', sender='K1ABC', recipient='K2ABC', via='K1ABC',
                     subject='not acknowledged', body='retained by depositor')
    assert not (root / 'parked-mail.json').exists()
