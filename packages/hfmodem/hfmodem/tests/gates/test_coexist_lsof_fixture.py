# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A retained lsof fixture never republishes closed Linux descriptors."""
from pathlib import Path

import pytest

from hfmodem.tests.coexist import observer
from hfmodem.tests.coexist.fixtures.live_openers import linux_openers


def process(proc, *, state='S', start='10', uids=(501,) * 4):
    directory = proc / '41'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'stat').write_text(f'41 (fixture process) {state} ' + '0 ' * 18 + f'{start} 0\n')
    (directory / 'status').write_text('Uid:\t' + '\t'.join(map(str, uids)) + '\n')
    return directory


@pytest.fixture
def captured(tmp_path):
    proc = tmp_path / 'proc'
    directory = process(proc)
    (directory / 'fd').mkdir()
    (directory / 'fdinfo').mkdir()
    target = tmp_path / 'claim'
    target.touch()
    (directory / 'fd/9').symlink_to(target)
    (directory / 'fdinfo/9').write_text('pos:\t27\n')
    info = target.stat()
    rows = {'41': [dict(fd=9, path=str(target), dev=info.st_dev, ino=info.st_ino, pos=0)]}
    participants = {'41': dict(start='10', uids=[501] * 4)}
    return proc, directory, participants, rows


def test_retained_live_descriptor_uses_current_position(captured):
    proc, _, participants, rows = captured
    actual = linux_openers(participants, rows, proc)
    assert actual['41'][0]['pos'] == 27
    assert {key: actual['41'][0][key] for key in ('fd', 'path', 'dev', 'ino')} == {
        key: rows['41'][0][key] for key in ('fd', 'path', 'dev', 'ino')}


@pytest.mark.parametrize('case', ['gone', 'zombie', 'closed'])
def test_confirmed_closed_resources_are_omitted(captured, monkeypatch, case):
    proc, directory, participants, rows = captured
    if case == 'gone':
        (directory / 'stat').unlink()
    elif case == 'zombie':
        process(proc, state='Z')
        monkeypatch.setattr(observer, 'linux_descriptors',
                            lambda *args: pytest.fail('terminal process descriptors inspected'))
    else:
        (directory / 'fd/9').unlink()
    assert not any(linux_openers(participants, rows, proc).values())


@pytest.mark.parametrize('field', ['fd', 'path', 'dev', 'ino'])
def test_replaced_descriptor_is_not_the_captured_opener(captured, field):
    proc, _, participants, rows = captured
    rows['41'][0][field] = 'other' if field == 'path' else rows['41'][0][field] + 1
    assert linux_openers(participants, rows, proc) == {'41': []}


@pytest.mark.parametrize('change', ['start', 'uids'])
def test_process_identity_drift_remains_fatal(captured, change):
    proc, _, participants, rows = captured
    process(proc, **({'start': '11'} if change == 'start' else {'uids': (502,) * 4}))
    with pytest.raises(observer.ObservationError, match='reused|credentials changed'):
        linux_openers(participants, rows, proc)


@pytest.mark.parametrize('failure', ['permission', 'missing_fds', 'malformed', 'start_after', 'uids_after'])
def test_unresolved_live_discovery_remains_fatal(captured, monkeypatch, failure):
    proc, directory, participants, rows = captured
    if failure == 'permission':
        read = Path.read_text

        def denied(path, *args, **kwargs):
            if path == directory / 'fdinfo/9':
                raise PermissionError('live descriptor denied')
            return read(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', denied)
        expected = observer.ObservationError
    elif failure == 'missing_fds':
        (directory / 'fd/9').unlink()
        (directory / 'fd').rmdir()
        expected = observer.ObservationError
    elif failure == 'malformed':
        (directory / 'fdinfo/9').write_text('not fdinfo\n')
        expected = observer.ObservationError
    else:
        original = observer.linux_descriptors

        def changed(pid, root):
            result = original(pid, root)
            process(proc, **({'start': '11'} if failure == 'start_after' else {'uids': (502,) * 4}))
            return result

        monkeypatch.setattr(observer, 'linux_descriptors', changed)
        expected = observer.ObservationError
    with pytest.raises(expected):
        linux_openers(participants, rows, proc)


@pytest.mark.parametrize('resolves', [True, False])
def test_denied_exiting_fixture_requires_confirmation_not_just_exit_flag(captured, monkeypatch, resolves):
    proc, directory, participants, rows = captured
    stat_path = directory / 'stat'
    text = stat_path.read_text()
    prefix, _, tail = text.rpartition(')')
    fields = tail.split()
    fields[0], fields[6] = 'R', '4194316'
    stat_path.write_text(prefix + ') ' + ' '.join(fields) + '\n')
    read = Path.read_text
    denied_reads = [None]

    def metadata(path, *args, **kwargs):
        text = read(path, *args, **kwargs)
        if path == stat_path and denied_reads[0] is not None:
            index = denied_reads[0]
            denied_reads[0] += 1
            if resolves and index >= 1:
                return text.replace(') R ', ') Z ')
        return text

    def denied(pid, root):
        denied_reads[0] = 0
        raise PermissionError('exiting fixture access')

    monkeypatch.setattr(Path, 'read_text', metadata)
    monkeypatch.setattr(observer, 'linux_descriptors', denied)
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    if resolves:
        assert linux_openers(participants, rows, proc) == {}
        assert denied_reads[0] == 3  # first R, then UID-bracketed Z/Z
    else:
        with pytest.raises(observer.ObservationError, match='live descriptor discovery unavailable'):
            linux_openers(participants, rows, proc)
        assert denied_reads[0] == 65  # initial R plus32 bounded stat/UID/stat checks
