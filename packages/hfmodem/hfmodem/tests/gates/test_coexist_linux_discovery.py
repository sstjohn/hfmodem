# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Deterministic /proc namespace and denial checks; no live process access."""
import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.tests.coexist import observer


def make_proc(root, *, start=10, uids=None, run='run', state='S'):
    directory = root / '41'
    (directory / 'fd').mkdir(parents=True)
    (directory / 'fdinfo').mkdir()
    (directory / 'stat').write_text(f'41 (test process) {state} ' + '0 ' * 18 + f'{start} 0\n')
    uids = (os.getuid(),) * 4 if uids is None else uids
    (directory / 'status').write_text('Uid:\t' + '\t'.join(map(str, uids)) + '\n')
    (directory / 'environ').write_bytes(f'HFMODEM_COEXIST_RUN={run}\0'.encode())
    (directory / 'cmdline').write_bytes(b'python\0child\0')
    (root / 'locks').write_text('')
    return directory


def foreign_uid():
    return 0 if os.getuid() != 0 else 1


def deny_environment(monkeypatch, directory, *, owner=None):
    read, stat = Path.read_bytes, Path.stat
    attempted = []

    def denied(path):
        if path == directory / 'environ':
            attempted.append(path)
            raise PermissionError(errno.EACCES, 'synthetic proc denial', str(path))
        return read(path)

    def ownership(path, *args, **kwargs):
        if path == directory / 'environ':
            value = foreign_uid() if owner is None else owner() if callable(owner) else owner
            return SimpleNamespace(st_uid=value)
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_bytes', denied)
    monkeypatch.setattr(Path, 'stat', ownership)
    return attempted


def test_foreign_process_credentials_are_independent_of_directory_owner(tmp_path, monkeypatch):
    directory = make_proc(tmp_path, uids=(foreign_uid(),) * 4)
    # The fake directory belongs to us; the process credentials do not.
    assert directory.stat().st_uid == os.getuid()
    attempted = deny_environment(monkeypatch, directory)
    assert observer.linux_participants('run', tmp_path) == {}
    assert attempted == []


def test_pre_run_unknown_foreign_owned_environment_can_be_excluded(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    attempted = deny_environment(monkeypatch, directory)
    assert observer.linux_participants('run', tmp_path, started_after=20) == {}
    assert attempted == [directory / 'environ']


@pytest.mark.parametrize('case', ['at_bound', 'after_bound', 'no_bound', 'same_owner', 'known'])
def test_denied_in_scope_or_unclassified_process_remains_fatal(tmp_path, monkeypatch, case):
    start = 20 if case == 'at_bound' else 21 if case == 'after_bound' else 10
    directory = make_proc(tmp_path, start=start)
    deny_environment(monkeypatch, directory, owner=os.getuid() if case == 'same_owner' else None)
    known = {41: {'start': str(start), 'uids': (os.getuid(),) * 4}} if case == 'known' else None
    with pytest.raises(observer.ObservationError, match='discovery denied'):
        observer.linux_participants('run', tmp_path,
                                   started_after=None if case == 'no_bound' else 20, known=known)


@pytest.mark.parametrize('change', ['start', 'credentials'])
def test_known_identity_drift_is_not_foreign_nonparticipant_evidence(tmp_path, change):
    make_proc(tmp_path)
    known = {41: {'start': '9' if change == 'start' else '10',
                  'uids': (foreign_uid(),) * 4 if change == 'credentials' else (os.getuid(),) * 4}}
    with pytest.raises(observer.ObservationError, match='known participant identity changed'):
        observer.linux_participants('run', tmp_path, started_after=20, known=known)


@pytest.mark.parametrize('change', ['start', 'credentials', 'environment_owner'])
def test_exclusion_requires_stable_identity_and_environment_owner(tmp_path, monkeypatch, change):
    directory = make_proc(tmp_path)
    owners = iter((foreign_uid(), os.getuid()))
    deny_environment(monkeypatch, directory,
                     owner=(lambda: next(owners)) if change == 'environment_owner' else None)
    read = Path.read_text
    counts = {}

    def changed(path, *args, **kwargs):
        counts[path.name] = counts.get(path.name, 0) + 1
        text = read(path, *args, **kwargs)
        if change == 'start' and path.name == 'stat' and counts[path.name] > 1:
            return text.replace('10 0\n', '11 0\n')
        if change == 'credentials' and path.name == 'status' and counts[path.name] > 1:
            return 'Uid:\t' + '\t'.join([str(foreign_uid())] * 4) + '\n'
        return text

    monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(observer.ObservationError, match='reused|credentials changed|owner changed'):
        observer.linux_participants('run', tmp_path, started_after=20)


def test_preexisting_readable_marker_is_found_and_never_later_exempted(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    live = observer.LinuxObserver('run', proc=tmp_path, started_after=20)
    assert set(live.capture().participants) == {41}
    deny_environment(monkeypatch, directory)
    with pytest.raises(observer.ObservationError, match='discovery denied'):
        live.capture()


@pytest.mark.parametrize('known', [False, True])
def test_denied_zombie_environment_requires_terminal_identity_not_run_age(tmp_path, monkeypatch, known):
    directory = make_proc(tmp_path, state='Z')
    deny_environment(monkeypatch, directory, owner=os.getuid())
    history = {41: {'start': '10', 'uids': (os.getuid(),) * 4}} if known else {}
    # Newer than the bound and owned by us: neither old age nor foreign
    # ownership can grant this exemption. Only the proved terminal state can.
    assert observer.linux_participants('run', tmp_path, started_after=5, known=history) == {}
    assert bool(history) is known


def test_denied_environment_of_disappeared_identity_is_not_a_live_denial(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    read = Path.read_bytes

    def disappeared(path):
        if path == directory / 'environ':
            (directory / 'stat').unlink()
            raise PermissionError(errno.EACCES, 'process exited', str(path))
        return read(path)

    monkeypatch.setattr(Path, 'read_bytes', disappeared)
    assert observer.linux_participants('run', tmp_path) == {}


@pytest.mark.parametrize('known', [False, True])
@pytest.mark.parametrize('change', ['birth', 'uids', 'state', 'malformed'])
def test_zombie_exemption_rejects_identity_or_state_drift(tmp_path, monkeypatch, known, change):
    directory = make_proc(tmp_path, state='Z')
    deny_environment(monkeypatch, directory)
    read = Path.read_text
    calls = {}

    def drift(path, *args, **kwargs):
        calls[path.name] = calls.get(path.name, 0) + 1
        text = read(path, *args, **kwargs)
        if path.name == 'stat' and calls[path.name] == 2:
            if change == 'birth':
                return text.replace('10 0\n', '11 0\n')
            if change == 'malformed':
                return '41 (truncated) Z\n'
        if change == 'uids' and path.name == 'status' and calls[path.name] == 2:
            return 'Uid:\t' + '\t'.join([str(foreign_uid())] * 4) + '\n'
        if change == 'state' and path.name == 'stat' and calls[path.name] == 3:
            return text.replace(') Z ', ') S ')
        return text

    monkeypatch.setattr(Path, 'read_text', drift)
    history = {41: {'start': '10', 'uids': (os.getuid(),) * 4}} if known else {}
    with pytest.raises(observer.ObservationError, match='reused|credentials changed|state changed|start identity'):
        observer.linux_participants('run', tmp_path, started_after=5, known=history)


def test_descriptor_denial_of_live_participant_remains_fatal(tmp_path, monkeypatch):
    make_proc(tmp_path)

    def denied(pid, proc):
        raise PermissionError(errno.EACCES, 'live descriptor denial')

    monkeypatch.setattr(observer, 'linux_descriptors', denied)
    live = observer.LinuxObserver('run', proc=tmp_path)
    with pytest.raises(PermissionError, match='live descriptor denial'):
        live.capture()
    assert live._known[41]['start'] == '10'


@pytest.mark.parametrize('terminal', ['zombie', 'gone'])
def test_terminal_descriptor_denial_omits_sample_but_retains_known_identity(tmp_path, monkeypatch, terminal):
    directory = make_proc(tmp_path)

    def denied(pid, proc):
        path = directory / 'stat'
        if terminal == 'gone':
            path.unlink()
        else:
            path.write_text(path.read_text().replace(') S ', ') Z '))
        raise PermissionError(errno.EACCES, 'terminal descriptor denial')

    monkeypatch.setattr(observer, 'linux_descriptors', denied)
    live = observer.LinuxObserver('run', proc=tmp_path)
    snapshot = live.capture()
    assert snapshot.participants == snapshot.descriptors == {}
    assert live._known[41] == {'start': '10', 'argv': 'python child ', 'uids': (os.getuid(),) * 4}
    (directory / 'stat').write_text('41 (new process) S ' + '0 ' * 18 + '11 0\n')
    with pytest.raises(observer.ObservationError, match='known participant identity changed'):
        live.capture()


@pytest.mark.parametrize('change', ['birth', 'uids'])
def test_terminal_descriptor_denial_cannot_hide_identity_drift(tmp_path, monkeypatch, change):
    directory = make_proc(tmp_path)

    def denied(pid, proc):
        path = directory / 'stat'
        text = path.read_text().replace(') S ', ') Z ')
        path.write_text(text.replace('10 0\n', '11 0\n') if change == 'birth' else text)
        if change == 'uids':
            (directory / 'status').write_text('Uid:\t' + '\t'.join([str(foreign_uid())] * 4) + '\n')
        raise PermissionError(errno.EACCES, 'terminal identity drift')

    monkeypatch.setattr(observer, 'linux_descriptors', denied)
    with pytest.raises(observer.ObservationError, match='reused|credentials changed'):
        observer.LinuxObserver('run', proc=tmp_path).capture()


@pytest.mark.parametrize('status', ['', 'Uid: 1 2 3\n', 'Uid: x 2 3 4\n', 'Uid: 1 2 3 4\nUid: 1 2 3 4\n'])
def test_unclassifiable_credentials_fail_closed(tmp_path, status):
    directory = make_proc(tmp_path)
    (directory / 'status').write_text(status)
    with pytest.raises(observer.ObservationError, match='UID identity'):
        observer.linux_participants('run', tmp_path, started_after=20)


@pytest.mark.parametrize('start', ['invalid', ''])
def test_unclassifiable_start_fails_closed(tmp_path, start):
    directory = make_proc(tmp_path)
    (directory / 'stat').write_text(f'41 (test) S {start}\n')
    with pytest.raises(observer.ObservationError, match='start identity'):
        observer.linux_participants('run', tmp_path, started_after=20)


def test_process_stat_must_match_the_sampled_pid(tmp_path):
    directory = make_proc(tmp_path)
    path = directory / 'stat'
    path.write_text(path.read_text().replace('41 (', '42 (', 1))
    with pytest.raises(observer.ObservationError, match='PID identity'):
        observer.linux_participants('run', tmp_path, started_after=20)


@pytest.mark.parametrize('bound', [-1, True, '20', 20.0])
def test_invalid_run_bound_fails_before_discovery(tmp_path, bound):
    with pytest.raises(observer.ObservationError, match='run start bound'):
        observer.linux_participants('run', tmp_path, started_after=bound)


@pytest.mark.parametrize('known', [{41: '10'}, {41: {'start': 'x', 'uids': [1] * 4}},
                                   {41: {'start': '10', 'uids': [True] * 4}},
                                   {41: {'start': '10', 'uids': [1] * 3}}])
def test_invalid_known_identity_fails_before_discovery(tmp_path, known):
    with pytest.raises(observer.ObservationError, match='known participant identity'):
        observer.linux_participants('run', tmp_path, known=known)


def test_boot_time_bound_uses_kernel_ticks_and_conservative_floor(monkeypatch):
    monkeypatch.setattr(observer.time, 'CLOCK_BOOTTIME', 987, raising=False)
    monkeypatch.setattr(observer.time, 'clock_gettime_ns', lambda clock: 7_999_999_999 if clock == 987 else None)
    monkeypatch.setattr(observer.os, 'sysconf', lambda name: 100 if name == 'SC_CLK_TCK' else None)
    assert observer.linux_run_start_floor() == 798


def exit_stat(*, pid=41, start='10', state='R', flags='4194316'):
    fields = [state, *(['0'] * 18), start, '0']
    fields[6] = flags
    return f'{pid} (test process) ' + ' '.join(fields) + '\n'


def terminal_sequence(monkeypatch, directory, sequence):
    read = Path.read_text
    calls = {'stat': 0, 'status': 0}

    def scripted(path, *args, **kwargs):
        if path == directory / 'stat':
            index = calls['stat']
            calls['stat'] += 1
            value = sequence[min(index, len(sequence) - 1)]
            if isinstance(value, Exception):
                raise value
            return value
        if path == directory / 'status':
            calls['status'] += 1
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', scripted)
    return calls


@pytest.mark.parametrize('terminal', ['zombie', 'dead', 'zombie_to_dead', 'gone'])
def test_denied_exiting_identity_requires_confirmed_terminal_resolution(tmp_path, monkeypatch, terminal):
    directory = make_proc(tmp_path)
    states = {'zombie': ('Z', 'Z'), 'dead': ('X', 'X'), 'zombie_to_dead': ('Z', 'X')}
    end = ([exit_stat(state=s, flags='0') for s in states[terminal]] if terminal in states
           else [FileNotFoundError()])
    terminal_sequence(monkeypatch, directory, [exit_stat()] * 3 + end)
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    budget = observer._ExitConfirmationBudget()
    assert observer._linux_terminal(directory, '10', (os.getuid(),) * 4,
                                    denied=PermissionError('exiting'), budget=budget)
    assert budget.checks == 2


@pytest.mark.parametrize('flags', ['0', '8', '4194312'])
def test_live_denial_without_exit_flag_does_not_poll(tmp_path, monkeypatch, flags):
    directory = make_proc(tmp_path)
    calls = terminal_sequence(monkeypatch, directory, [exit_stat(flags=flags)])
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: pytest.fail('live denial polled'))
    budget = observer._ExitConfirmationBudget()
    denied = PermissionError('live')
    assert not observer._linux_terminal(directory, '10', (os.getuid(),) * 4, denied=denied, budget=budget)
    assert calls == {'stat': 1, 'status': 0}
    assert budget.checks == 0
    assert any(f'flags={flags!r}' in note for note in denied.__notes__)


def test_exit_flag_without_denied_access_does_not_start_confirmation(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    terminal_sequence(monkeypatch, directory, [exit_stat()])
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: pytest.fail('no access was denied'))
    assert not observer._linux_terminal(directory, '10', (os.getuid(),) * 4)


def test_persistent_exit_flag_never_establishes_absence_and_has_check_limit(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    calls = terminal_sequence(monkeypatch, directory, [exit_stat()])
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    budget = observer._ExitConfirmationBudget()
    denied = PermissionError('persistent')
    assert not observer._linux_terminal(directory, '10', (os.getuid(),) * 4, denied=denied, budget=budget)
    assert budget.checks == 32
    assert calls == {'stat': 65, 'status': 32}
    assert any('budget exhausted' in note and "state='R'" in note for note in denied.__notes__)


def test_exit_confirmation_identity_deadline_does_not_reset(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    terminal_sequence(monkeypatch, directory, [exit_stat()])
    ticks = iter([0, observer.IDENTITY_DEADLINE_NS])
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: next(ticks, observer.IDENTITY_DEADLINE_NS))
    budget = observer._ExitConfirmationBudget()
    for _ in range(2):
        assert not observer._linux_terminal(directory, '10', (os.getuid(),) * 4,
                                            denied=PermissionError('exiting'), budget=budget)
    assert budget.checks == 1


def test_capture_exit_confirmation_deadline_cannot_reset_for_new_pids(monkeypatch):
    now = [0]
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: now[0])
    budget = observer._ExitConfirmationBudget()
    for index in range(7):
        now[0] = index * (observer.CAPTURE_DEADLINE_NS // 7)
        assert budget.allow(Path(str(index + 1)), '10', (501,) * 4)
    now[0] = observer.CAPTURE_DEADLINE_NS
    assert not budget.allow(Path('99'), '10', (501,) * 4)
    assert budget.checks == 7


def test_capture_exit_confirmation_check_limit_cannot_multiply_per_pid(monkeypatch):
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    budget = observer._ExitConfirmationBudget()
    for pid in range(1, 5):
        assert all(budget.allow(Path(str(pid)), '10', (501,) * 4) for _ in range(32))
        assert not budget.allow(Path(str(pid)), '10', (501,) * 4)
    assert budget.checks == 128
    assert not budget.allow(Path('99'), '10', (501,) * 4)


@pytest.mark.parametrize('change', ['birth', 'pid', 'uids', 'flags', 'malformed', 'zombie_to_live'])
def test_exit_confirmation_identity_and_metadata_changes_remain_fatal(tmp_path, monkeypatch, change):
    directory = make_proc(tmp_path)
    if change == 'birth':
        after = exit_stat(start='11')
    elif change == 'pid':
        after = exit_stat(pid=42)
    elif change == 'flags':
        after = exit_stat(flags='not-flags')
    elif change == 'malformed':
        after = '41 (truncated) R\n'
    else:
        after = exit_stat(state='Z') if change == 'zombie_to_live' else exit_stat()
    if change == 'uids':
        (directory / 'status').write_text('Uid:\t' + '\t'.join([str(foreign_uid())] * 4) + '\n')
    terminal_sequence(monkeypatch, directory, [exit_stat(), after, exit_stat()])
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    with pytest.raises(observer.ObservationError, match='reused|PID identity|credentials|flags|start identity|state changed'):
        observer._linux_terminal(directory, '10', (os.getuid(),) * 4, denied=PermissionError('exiting'))


def test_exit_flag_clearing_is_fatal_without_further_polling(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    calls = terminal_sequence(monkeypatch, directory, [exit_stat(), exit_stat(flags='0')])
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    budget = observer._ExitConfirmationBudget()
    assert not observer._linux_terminal(directory, '10', (os.getuid(),) * 4,
                                        denied=PermissionError('exiting'), budget=budget)
    assert budget.checks == 1 and calls['stat'] == 3


@pytest.mark.parametrize('known', [False, True])
def test_discovery_confirms_exiting_identity_without_weakening_live_denials(tmp_path, monkeypatch, known):
    directory = make_proc(tmp_path)
    terminal_sequence(monkeypatch, directory, [exit_stat()] * 2 + [exit_stat(state='Z')] * 2)
    deny_environment(monkeypatch, directory, owner=os.getuid())
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    history = {41: {'start': '10', 'uids': (os.getuid(),) * 4}} if known else {}
    assert observer.linux_participants('run', tmp_path, started_after=5, known=history) == {}
    assert bool(history) is known


@pytest.mark.parametrize('known', [False, True])
def test_discovery_persistent_exiting_identity_denial_remains_fatal(tmp_path, monkeypatch, known):
    directory = make_proc(tmp_path)
    terminal_sequence(monkeypatch, directory, [exit_stat()])
    deny_environment(monkeypatch, directory)
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    history = {41: {'start': '10', 'uids': (os.getuid(),) * 4}} if known else {}
    with pytest.raises(observer.ObservationError, match='discovery denied'):
        observer.linux_participants('run', tmp_path, started_after=5, known=history)


def test_descriptor_exit_confirmation_omits_only_terminal_identity_and_keeps_history(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    terminal_sequence(monkeypatch, directory, [exit_stat()] * 3 + [exit_stat(state='Z')] * 2)
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)

    def denied(pid, proc):
        raise PermissionError('exiting descriptor access')

    monkeypatch.setattr(observer, 'linux_descriptors', denied)
    live = observer.LinuxObserver('run', proc=tmp_path)
    snapshot = live.capture()
    assert snapshot.participants == snapshot.descriptors == {}
    assert live._known[41]['start'] == '10'


def test_exiting_identity_gets_scheduler_turn_before_confirmation_budget_is_spent(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    path = directory / 'stat'
    path.write_text(exit_stat())
    now = [0]
    sleeps = []
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: now[0])

    def scheduled(delay):
        sleeps.append(delay)
        now[0] += round(delay * 1_000_000_000)
        path.write_text(exit_stat(state='Z'))

    monkeypatch.setattr(observer.time, 'sleep', scheduled)
    budget = observer._ExitConfirmationBudget()
    assert observer._linux_terminal(directory, '10', (os.getuid(),) * 4,
                                    denied=PermissionError('exiting'), budget=budget)
    assert sleeps == [observer.YIELD_S]
    assert budget.checks == 2  # R/UID/R, scheduler turn, then stable Z/UID/Z


@pytest.mark.parametrize('oversleep', [False, True])
def test_exit_confirmation_yield_is_clipped_and_expiry_never_establishes_absence(tmp_path, monkeypatch, oversleep):
    directory = make_proc(tmp_path)
    read = Path.read_text
    now = [0]
    reads = [0]
    sleeps = []

    def metadata(path, *args, **kwargs):
        if path == directory / 'stat':
            reads[0] += 1
            if reads[0] == 3:
                now[0] = observer.IDENTITY_DEADLINE_NS - 500_000  # 0.5 ms of the identity budget left
            return exit_stat(state='Z' if sleeps else 'R')
        return read(path, *args, **kwargs)

    def scheduled(delay):
        sleeps.append(delay)
        now[0] += 100_000_000 if oversleep else round(delay * 1_000_000_000)

    monkeypatch.setattr(Path, 'read_text', metadata)
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: now[0])
    monkeypatch.setattr(observer.time, 'sleep', scheduled)
    budget = observer._ExitConfirmationBudget()
    assert not observer._linux_terminal(directory, '10', (os.getuid(),) * 4,
                                        denied=PermissionError('exiting'), budget=budget)
    assert sleeps == [.0005]
    assert reads[0] == 3 and budget.checks == 1  # no read of Z after deadline/oversleep


def test_identity_drift_after_scheduler_turn_is_still_fatal(tmp_path, monkeypatch):
    directory = make_proc(tmp_path)
    path = directory / 'stat'
    path.write_text(exit_stat())
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: 0)
    monkeypatch.setattr(observer.time, 'sleep', lambda delay: path.write_text(exit_stat(state='Z', start='11')))
    with pytest.raises(observer.ObservationError, match='pid reused'):
        observer._linux_terminal(directory, '10', (os.getuid(),) * 4, denied=PermissionError('exiting'))


@pytest.mark.parametrize('states', [('Z', 'Z'), ('Z', 'X'), ('X', 'X'), ('X', 'Z')])
@pytest.mark.parametrize('denied', [False, True])
def test_kernel_terminal_states_are_uid_bracketed_without_waiting(tmp_path, monkeypatch, states, denied):
    directory = make_proc(tmp_path)
    calls = terminal_sequence(monkeypatch, directory, [exit_stat(state=s, flags='0') for s in states])
    monkeypatch.setattr(observer.time, 'monotonic_ns', lambda: pytest.fail('terminal proof started a wait'))
    monkeypatch.setattr(observer.time, 'sleep', lambda delay: pytest.fail('terminal proof slept'))
    budget = observer._ExitConfirmationBudget()
    assert observer._linux_terminal(directory, '10', (os.getuid(),) * 4,
                                    denied=PermissionError('terminal') if denied else None, budget=budget)
    assert calls == {'stat': 2, 'status': 1}
    assert budget.checks == 0


@pytest.mark.parametrize('terminal', ['Z', 'X'])
@pytest.mark.parametrize('other', ['R', 'S', '?'])
def test_terminal_to_live_or_unknown_drift_is_fatal_and_preserves_both_raw_states(
        tmp_path, monkeypatch, terminal, other):
    directory = make_proc(tmp_path)
    first, second = exit_stat(state=terminal), exit_stat(state=other)
    terminal_sequence(monkeypatch, directory, [first, second])
    with pytest.raises(observer.ObservationError, match='state changed') as caught:
        observer._linux_terminal(directory, '10', (os.getuid(),) * 4, denied=PermissionError('terminal'))
    notes = '\n'.join(caught.value.__notes__)
    assert 'first_stat=' + repr(first) in notes
    assert 'last_stat=' + repr(second) in notes


@pytest.mark.parametrize('change', ['birth', 'pid', 'uids', 'malformed'])
def test_dead_state_never_hides_identity_or_credential_drift(tmp_path, monkeypatch, change):
    directory = make_proc(tmp_path)
    second = {'birth': exit_stat(state='X', start='11'), 'pid': exit_stat(state='X', pid=42),
              'uids': exit_stat(state='X'), 'malformed': '41 (truncated) X\n'}[change]
    terminal_sequence(monkeypatch, directory, [exit_stat(state='X'), second])
    if change == 'uids':
        (directory / 'status').write_text('Uid:\t' + '\t'.join([str(foreign_uid())] * 4) + '\n')
    with pytest.raises(observer.ObservationError, match='reused|PID identity|credentials|start identity'):
        observer._linux_terminal(directory, '10', (os.getuid(),) * 4, denied=PermissionError('terminal'))
