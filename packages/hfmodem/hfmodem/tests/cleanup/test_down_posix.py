# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from hfmodem.core import cat, claim, down, down_posix, native, ptt
from hfmodem.station import client

from .test_down import Immediate

pytestmark = pytest.mark.posix_only('POSIX down inventory, flock and process identities')

START = 'Sat_Oct_17_12:00:00_2026'
BOOT = '12345678-1234-5678-9abc-123456789abc'


@pytest.fixture
def cleanup(tmp_path, monkeypatch, capsys):
    state = tmp_path / 'state'
    state.mkdir()
    journal = tmp_path / 'journal'
    journal.touch()
    station = tmp_path / 'station.toml'
    scenario = {}
    queried = set()
    clock = NS(now=0.)
    monkeypatch.setenv('HFMODEM_STATE_DIR', str(state))
    monkeypatch.setenv('HFMODEM_STATION', str(station))
    monkeypatch.setenv('CAT_PORT', str(tmp_path / 'cat'))
    monkeypatch.setenv('PTT_PORT', str(tmp_path / 'ptt'))
    for name in ('cat', 'ptt'):
        (tmp_path / name).touch()
    monkeypatch.delenv('HFMODEM_RIG', raising=False)
    monkeypatch.delenv('RIG_CLAIMED', raising=False)
    monkeypatch.setattr(down, 'ThreadPoolExecutor', Immediate)
    monkeypatch.setattr(down.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(down.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    monkeypatch.setattr(claim, 'audio_nodes', lambda *a, **kw: ())

    def log(value):
        with journal.open('a') as stream:
            stream.write(value + '\n')

    def rows():
        result = {pid: row for pid, row in scenario.get('rows', {}).items()
                  if pid not in scenario.get('gone', [])}
        result[os.getpid()] = [1, 'python cleanup-unit-test']
        return result

    def started(pid):
        # ps lstart is boot time plus ticks; a wall-clock step moves it from a given table on.
        stepped = scenario.get('clock_shift') or scenario.get('tables', 0) >= scenario.get('clock_step', float('inf'))
        return ('Sun Oct 18 12:00:00 2026' if pid in scenario.get('reused', []) else
                'Sat Oct 17 11:59:59 2026' if stepped else START.replace('_', ' '))

    def run(argv, **kwargs):
        assert argv[0] in {'ps', 'lsof'}, argv
        if argv[0] == 'ps':
            if '-axo' in argv:
                scenario['tables'] = scenario.get('tables', 0) + 1
                output = ''.join(f'{pid} {parent} {os.getuid()} {started(pid)} {command}\n'
                                 for pid, (parent, command) in rows().items())
                if scenario.get('ps_always_fails') or scenario['tables'] in scenario.get('ps_failures', []):
                    # Even well-formed stdout containing our root is not a full inventory.
                    output = f'{os.getpid()} 1 {os.getuid()} {START.replace("_", " ")} python\n'
                    error = scenario.get('ps_stderr', '')
                    if kwargs.get('check'):
                        raise down_posix.subprocess.CalledProcessError(1, argv, output, error)
                    return NS(returncode=1, stdout=output, stderr=error)
            else:
                pid = int(argv[argv.index('-p') + 1])
                output = started(pid) if 'lstart=' in argv else rows()[pid][1]
        elif '-t' in argv:
            port = next(word.split(':')[1] for word in argv if word.startswith('-iTCP:'))
            output = '\n'.join(str(pid) for pid in scenario.get('listeners', {}).get(port, []) if pid in rows())
        else:
            output = ''
            for raw in argv[argv.index('--') + 1:]:
                path = Path(raw)
                queried.add(str(path))
                if path == claim.mutex_path():
                    selected = scenario.get('mutex_holders', [])
                elif path == claim.claim_path():
                    selected = scenario.get('holders', [])
                    if scenario.get('tables') == scenario.get('late_table'):
                        # A helper that starts holding after this table and is gone by the next.
                        output += f'p505\ncpython\nf9\no0t{path.read_text().split()[4]}\n'
                else:
                    selected = scenario.get('devices', [])
                for item in selected:
                    pid = item if isinstance(item, int) else item[0]
                    if pid not in rows():
                        continue
                    if isinstance(item, list):
                        offset = item[1]
                    elif path in (claim.claim_path(), claim.mutex_path()):
                        offset = int(path.read_text().split()[2 if path == claim.mutex_path() else 4])
                    else:
                        offset = 0
                    output += f'p{pid}\nc{rows()[pid][1].split()[0]}\nf9\no0t{offset}\n'
        return NS(returncode=0, stdout=output, stderr='')

    monkeypatch.setattr(down_posix.subprocess, 'run', run)
    monkeypatch.setattr(down_posix.subprocess, 'Popen', lambda *a, **kw: pytest.fail('spawned a process'))
    openers = claim.openers
    def discover(paths, **kwargs):
        kwargs['platform'] = 'darwin'
        return openers(paths, **kwargs)
    monkeypatch.setattr(claim, 'openers', discover)
    # Unless a scenario sets `linux`, it models darwin, where `ps -o lstart=` (faked above) is the
    # identity; on a Linux host the real process_started would read /proc for pids that do not exist.
    process_started = claim.process_started
    monkeypatch.setattr(claim, 'process_started',
                        lambda pid, *, deadline=None: claim.ps_started(pid, deadline=deadline))
    flock = claim.fcntl.flock

    def lock(fd, operation):
        if operation == claim.fcntl.LOCK_EX | claim.fcntl.LOCK_NB:
            info = os.fstat(fd)
            if info.st_ino == claim.claim_path().stat().st_ino:
                holders = [item if isinstance(item, int) else item[0] for item in scenario.get('holders', [])]
                if scenario.get('held') and not scenario.get('free') and (101 in rows() or any(pid in rows() for pid in holders)):
                    raise BlockingIOError()
            elif scenario.get('mutex_busy'):
                raise BlockingIOError()
        return flock(fd, operation)

    monkeypatch.setattr(claim.fcntl, 'flock', lock)

    def terminate(child):
        log(f'TERM {child.pid}')
        if not scenario.get('resist'):
            scenario.setdefault('gone', []).append(child.pid)
        for pid, row in scenario.pop('arrive', {}).items():
            scenario.setdefault('rows', {})[pid] = row
            scenario.setdefault('devices', []).append(pid)
            if scenario.get('arrive_holds'):
                scenario.setdefault('holders', []).append(pid)

    def kill(child):
        log(f'KILL {child.pid}')
        if not scenario.get('survive'):
            scenario.setdefault('gone', []).append(child.pid)

    monkeypatch.setattr(native.NativeChild, 'alive', lambda child: child.pid in rows())
    monkeypatch.setattr(native.NativeChild, 'terminate', terminate)
    monkeypatch.setattr(native.NativeChild, 'kill', kill)

    def service_state(self, **kwargs):
        if scenario.get('station_unresponsive'):
            raise client.ClientError('old station')
        return {'service': {'state': scenario.get('station_state', 'observing')}}

    def release(self, **kwargs):
        log('station release')
        if scenario.pop('transitioning', False):
            raise client.CommandFailed({'reason': 'transitioning'})
        if not scenario.get('release_kept'):
            scenario.update(station_state='observing', holders=[], devices=[], free=True)

    monkeypatch.setattr(client.Client, 'state', service_state)
    monkeypatch.setattr(client.Client, 'release', release)

    class Ptt:
        released_low = True
        def __init__(self, port):
            assert port == os.environ['PTT_PORT']
        def open(self):
            log('proof open')
        def assert_(self, state):
            assert state is False
        def sense(self):
            return {'low': False, 'high': True, 'unreadable': None}[scenario.get('line', 'low')]
        def release(self):
            log('proof close')

    class Cat:
        def __init__(self, port, profile):
            assert claim.serial_port(port) == claim.serial_port(os.environ['CAT_PORT'])
        def _run(self):
            pass
        def open(self):
            log('CAT serial open')
        def freq(self):
            if scenario.get('cat_fail'):
                raise claim.ClaimError('CAT readback failed')
            return 7100000
        def close(self):
            log('CAT serial close')

    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def sendall(self, payload):
            assert payload == b'f\n'
            log('CAT TCP readback')
        def settimeout(self, timeout):
            pass
        def recv(self, size):
            return b'7100000\n'

    monkeypatch.setattr(ptt, 'RtsPtt', Ptt)
    monkeypatch.setattr(cat, 'YaesuCat', Cat)
    def connect(endpoint, timeout):
        assert endpoint == ('127.0.0.1', 4532)
        return Connection()
    monkeypatch.setattr(down.socket, 'create_connection', connect)
    replace = os.replace
    def publish(source, target):
        assert Path(target) == state / 'cleanup.json'
        if scenario.get('evidence_fail'):
            raise OSError('evidence failed')
        log('evidence')
        return replace(source, target)
    monkeypatch.setattr(down.os, 'replace', publish)

    def cleanup(body=None, *, backend='none', inherited=False, forged=False):
        body = dict(body or {})
        for name in ('rows', 'arrive'):
            if name in body:
                body[name] = {int(pid): row for pid, row in body[name].items()}
        station.write_text('[rig]\ncat = ' + json.dumps(backend) + '\n')
        claim.bootstrap()
        with monkeypatch.context() as context:
            if body.get('linux'):
                context.setattr(claim.sys, 'platform', 'linux')
                context.setattr(claim, 'process_started', process_started)
                read_text = Path.read_text
                def read(path, *args, **kwargs):
                    if str(path) == '/proc/sys/kernel/random/boot_id':
                        return BOOT + '\n'
                    if str(path).startswith('/proc/') and path.name == 'stat':
                        pid = int(path.parent.name)
                        ticks = pid * 100 + int(pid in scenario.get('reused', []))
                        return f'{pid} (fake process)) S ' + '0 ' * 18 + str(ticks)
                    return read_text(path, *args, **kwargs)
                context.setattr(Path, 'read_text', read)
            descriptor = None
            if inherited or forged:
                descriptor = claim.acquire('shell', 'test')
                if forged:
                    reopened = os.open(claim.claim_path(), os.O_RDWR)
                    context.setattr(claim, 'received', lambda: claim.inherited(reopened))
                else:
                    context.setattr(claim, 'received', lambda: claim.inherited(descriptor))
                context.setenv('RIG_CLAIMED', str(claim.claim_path()))
            scenario.update(body)
            if body.get('mutex_busy'):
                claim.mutex_path().write_text(f'303 {START} 123\n')
            if body.get('held'):
                birth = f'linux:{BOOT}:10100' if body.get('linux') else START
                claim.claim_path().write_text(f'101 {birth} shell test 123 fake owner\n')
            try:
                capsys.readouterr()
                result = down.run(None, client)
                output = capsys.readouterr()
            finally:
                if forged:
                    os.close(reopened)
                    claim.release(descriptor)
            return NS(returncode=result, stdout=output.out, stderr=output.err)

    cleanup.state, cleanup.journal, cleanup.queried = state, journal, queried
    cleanup.clock = clock
    return cleanup


def test_a_claim_holder_younger_than_the_inventory_table_is_looked_up_again(cleanup):
    # Table 1 is down's issue snapshot; the inventory takes table 2.
    result = cleanup({'late_table': 2}, inherited=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CLEANUP COMPLETE' in result.stdout
    assert 'TERM' not in cleanup.journal.read_text()


@pytest.mark.parametrize('failure', [1, 2])
@pytest.mark.parametrize('error', ['', 'ps: process disappeared\n'])
def test_failed_ps_inventory_is_retaken_before_signalling(cleanup, failure, error):
    result = cleanup({'ps_failures': [failure], 'ps_stderr': error,
                      'rows': {'202': [1, 'hfmodem.shrike.onair']}})
    assert result.returncode == 0, result.stdout + result.stderr
    assert cleanup.journal.read_text().startswith('TERM 202\n')
    assert 'CLEANUP COMPLETE' in result.stdout


def test_failed_ps_stdout_cannot_hide_an_unknown_holder(cleanup):
    result = cleanup({'ps_failures': [2], 'rows': {'303': [1, 'unrelated']}, 'devices': [303]})
    assert result.returncode == 1
    assert 'unidentified device/claim holder' in result.stdout
    assert not cleanup.journal.read_text()


def test_ps_inventory_that_never_settles_is_bounded_and_fail_closed(cleanup):
    result = cleanup({'ps_always_fails': True})
    assert result.returncode == 1
    assert 'process inventory unavailable' in result.stdout
    assert 0 < cleanup.clock.now <= 45
    assert not cleanup.journal.read_text()
    assert 'CLEANUP COMPLETE' not in result.stdout


@pytest.mark.parametrize('late_result', [False, True])
def test_holder_discovery_cannot_settle_after_inventory_deadline(cleanup, monkeypatch, late_result):
    holders = claim.holders
    def slow(descriptor, **kwargs):
        assert kwargs['deadline'] > cleanup.clock.now
        if late_result:
            cleanup.clock.now = kwargs['deadline'] + 1
            return []
        def listing(paths):
            cleanup.clock.now = kwargs['deadline'] + 1
            return ''
        return holders(descriptor, lsof=listing, **kwargs)
    monkeypatch.setattr(claim, 'holders', slow)
    result = cleanup(inherited=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'deadline expired' in result.stdout
    assert 'CLEANUP COMPLETE' not in result.stdout
    assert not cleanup.journal.read_text()


def test_down_success_and_publication(cleanup):
    result = cleanup(backend='yaesu')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PTT LINE low\nCAT ok\nCLEANUP COMPLETE' in result.stdout
    log = cleanup.journal.read_text()
    assert log.index('proof close') < log.index('CAT serial open') < log.index('evidence')
    assert (cleanup.state / 'rig.claim').read_text() == ''
    assert (cleanup.state / 'rig.claim.cleanup').read_text() == ''
    record = json.loads((cleanup.state / 'cleanup.json').read_text())
    assert (record['line'], record['cat']) == ('low', 'ok')


@pytest.mark.parametrize('cat_port,ptt_port,line', [('', 'ptt', 'low'), ('cat', '', 'none'),
                                                ('', '', 'none'), (' ', ' ', 'none')])
def test_blank_ports_are_no_ports(cleanup, monkeypatch, tmp_path, cat_port, ptt_port, line):
    for name, value in (('CAT_PORT', cat_port), ('PTT_PORT', ptt_port)):
        monkeypatch.setenv(name, str(tmp_path / value) if value.strip() else value)
    result = cleanup(backend='yaesu')
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'PTT LINE {line}\nCAT ok\nCLEANUP COMPLETE' in result.stdout
    queried = {path for path in cleanup.queried if not path.startswith(str(cleanup.state))}
    assert queried == {str(tmp_path / value) for value in (cat_port, ptt_port) if value.strip()}


def test_inherited_cleanup_preserves_epoch(cleanup):
    result = cleanup(inherited=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ' shell test ' in (cleanup.state / 'rig.claim').read_text()
    assert (cleanup.state / 'rig.claim.cleanup').read_text() == ''
    assert 'CLEANUP COMPLETE' in result.stdout


def test_forged_inheritance_refuses_without_proof(cleanup):
    result = cleanup(forged=True)
    assert result.returncode != 0
    assert 'does not possess the holder nonce' in result.stdout
    assert not cleanup.journal.read_text()
    assert not (cleanup.state / 'cleanup.json').exists()


def test_mutex_refusal_never_signals_or_proves(cleanup):
    result = cleanup({'rows': {'303': [1, 'prior cleanup']}, 'mutex_holders': [303],
                      'mutex_busy': True})
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'cleanup in progress' in result.stdout
    assert 'CLEANUP COMPLETE' not in result.stdout
    assert not cleanup.journal.read_text()


@pytest.mark.parametrize('reused', [False, True])
def test_linux_down_publisher_identity_after_clock_shift(cleanup, reused):
    result = cleanup({'linux': True, 'clock_shift': True, 'held': True,
                      'rows': {'101': [1, 'unrelated']}, 'holders': [101],
                      'reused': [101] if reused else []})
    assert result.returncode == (1 if reused else 7), result.stdout + result.stderr
    assert ('publisher pid reused' in result.stdout) == reused
    if not reused:
        assert 'interrupt it and run down again' in result.stdout
    assert not cleanup.journal.read_text()


@pytest.mark.parametrize('body, evidence', [({'line': 'high'}, 'PTT LINE high'),
                                           ({'line': 'unreadable'}, 'PTT LINE unreadable'),
                                           ({'cat_fail': True}, 'CAT failed'),
                                           ({'evidence_fail': True}, 'CAT ok')])
def test_partial_cleanup_never_publishes_completion(cleanup, body, evidence):
    result = cleanup(body, backend='yaesu')
    assert result.returncode != 0
    assert evidence in result.stdout, result.stdout + result.stderr
    assert 'NOT CONFIRMED' in result.stdout
    assert 'CLEANUP COMPLETE' not in result.stdout


def test_cat_none_and_no_ptt(cleanup, monkeypatch):
    monkeypatch.setenv('PTT_PORT', '')
    result = cleanup()
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PTT LINE none\nCAT none\nCLEANUP COMPLETE' in result.stdout
    assert 'rigctl' not in cleanup.journal.read_text()


def test_live_root_refuses_entire_inventory(cleanup):
    result = cleanup({'held': True, 'rows': {'101': [1, 'console fly'],
                                         '202': [101, 'hfmodem.shrike.onair']},
                    'holders': [101, 202], 'devices': [202]})
    assert result.returncode == 7, result.stdout + result.stderr
    assert 'holds the claim; interrupt it and run down again' in result.stdout
    assert 'TERM' not in cleanup.journal.read_text()
    assert 'proof' not in cleanup.journal.read_text()


def test_unknown_device_blocks_all_signals(cleanup):
    result = cleanup({'rows': {'202': [1, 'hfmodem.shrike.onair'], '303': [1, 'unrelated']},
                    'devices': [202, 303]})
    assert result.returncode == 1
    assert 'unidentified' in result.stdout
    assert cleanup.journal.read_text() == ''


def test_legacy_target_verified_then_inventory_repeated(cleanup):
    result = cleanup({'rows': {'202': [1, 'hfmodem.shrike.onair']}, 'devices': [202]})
    assert result.returncode == 0, result.stdout + result.stderr
    log = cleanup.journal.read_text()
    assert log.index('TERM 202') < log.index('proof open')
    assert 'KILL' not in log


def test_unknown_station_is_never_signalled(cleanup):
    result = cleanup({'rows': {'202': [1, 'hfmodem station old.toml']},
                    'listeners': {'8737': [202]}, 'devices': [202], 'station_unresponsive': True})
    assert result.returncode == 1
    assert 'station not released' in result.stdout
    assert 'TERM' not in cleanup.journal.read_text()


def test_station_release_retries_transitioning(cleanup):
    result = cleanup({'rows': {'202': [1, 'hfmodem station station.toml']},
                    'listeners': {'8737': [202]}, 'station_state': 'owning',
                    'transitioning': True, 'devices': [202]})
    assert result.returncode == 0, result.stdout + result.stderr
    assert cleanup.journal.read_text().count('station release') == 2
    assert 'TERM' not in cleanup.journal.read_text()


def test_delegated_daemon_not_signalled_or_serial_queried(cleanup):
    result = cleanup({'rows': {'303': [1, 'rigctld -m 1036']},
                    'listeners': {'4532': [303]}}, backend='rigctld')
    assert result.returncode == 0, result.stdout + result.stderr
    log = cleanup.journal.read_text()
    assert 'TERM' not in log
    assert 'CAT TCP readback' in log and 'CAT serial' not in log
    assert 'delegated rigctld pid 303' in result.stdout


@pytest.mark.parametrize('survive', [False, True])
def test_checked_escalation_requires_termination_before_proof(cleanup, survive):
    result = cleanup({'rows': {'202': [1, 'hfmodem.lanner.qso']},
                    'devices': [202], 'resist': True, 'survive': survive})
    log = cleanup.journal.read_text()
    assert log.startswith('TERM 202\nKILL 202\n')
    assert ('proof open' in log) != survive
    assert result.returncode == int(survive), result.stdout + result.stderr


@pytest.mark.parametrize('step', [2, 3], ids=['before-signals', 'after-signals'])
def test_a_linux_clock_step_during_down_keeps_its_issued_targets(cleanup, step):
    """Table 1 is issue(), table 2 signals 202, and the wall clock steps from `step` on."""
    result = cleanup({'linux': True, 'clock_step': step, 'resist': True,
                      'rows': {'202': [1, 'hfmodem.lanner.qso']}, 'devices': [202]})
    assert 'started after this down was issued' not in result.stdout
    assert cleanup.journal.read_text().startswith('TERM 202\nKILL 202\n')
    assert result.returncode == 0, result.stdout + result.stderr


def test_station_descendant_matching_legacy_pattern_is_protected(cleanup):
    result = cleanup({'rows': {'101': [1, 'hfmodem station fixture.toml'],
                             '202': [101, 'hfmodem.lanner.qso']},
                    'listeners': {'8737': [101]}, 'devices': [202]})
    assert result.returncode == 1
    assert 'observing station still holds' in result.stdout
    assert not cleanup.journal.read_text()


def test_observing_station_still_holding_after_its_release_is_refused_at_once(cleanup):
    result = cleanup({'held': True, 'rows': {'101': [1, 'hfmodem station fixture.toml']},
                    'listeners': {'8737': [101]}, 'holders': [101], 'release_kept': True})
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'observing station still holds a device or claim' in result.stdout
    assert cleanup.journal.read_text() == 'station release\n'


def test_unidentified_delegated_listener_blocks_cleanup(cleanup):
    result = cleanup({'rows': {'303': [1, 'unrelated listener']},
                    'listeners': {'4532': [303]}}, backend='rigctld')
    assert result.returncode == 1
    assert 'delegated CAT listener unidentified' in result.stdout
    assert not cleanup.journal.read_text()


def test_reopened_claim_descriptor_is_not_a_legacy_target(cleanup):
    result = cleanup({'held': True, 'free': True, 'rows': {'303': [1, 'unrelated']},
                    'holders': [[303, 0]]})
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TERM' not in cleanup.journal.read_text()


def test_unmatched_lock_settles_to_deadline_without_signalling_or_proof(cleanup, monkeypatch):
    attempts = []
    def occupied(*args, **kwargs):
        attempts.append(args)
        raise claim.ClaimBusy('rig claimed: opaque lock holder')
    monkeypatch.setattr(claim, 'acquire', occupied)
    result = cleanup()
    assert result.returncode == 1
    assert 'NOT CONFIRMED: cleanup inventory did not settle within 45 s: rig claimed:' in result.stdout
    assert 'CLEANUP COMPLETE' not in result.stdout and 'PTT LINE' not in result.stdout
    assert len(attempts) > 1 and not cleanup.journal.read_text()


@pytest.mark.parametrize('model,stopped', [(0, True), (1, False)])
def test_generic_daemon_selection_uses_the_configured_profile(cleanup, monkeypatch, model, stopped):
    monkeypatch.setenv('HFMODEM_RIG', 'generic')
    result = cleanup({'rows': {'303': [1, f'rigctld -m {model} -P NONE']}})
    assert result.returncode == 0, result.stdout + result.stderr
    assert ('TERM 303' in cleanup.journal.read_text()) == stopped
    assert 'CLEANUP COMPLETE' in result.stdout


def test_dead_root_one_shot_is_reaped_by_descriptor_possession(cleanup):
    result = cleanup({'held': True, 'rows': {'202': [1, 'rigctl -P NONE f']},
                    'holders': [202], 'devices': [202]})
    assert result.returncode == 0, result.stdout + result.stderr
    log = cleanup.journal.read_text()
    assert log.index('TERM 202') < log.index('proof open')


def test_reused_publisher_pid_refuses_without_signalling(cleanup):
    result = cleanup({'held': True, 'rows': {'101': [1, 'unrelated']},
                    'holders': [101], 'reused': [101]})
    assert result.returncode == 1
    assert 'publisher pid reused' in result.stdout
    assert not cleanup.journal.read_text()


@pytest.mark.parametrize('resist', [False, True])
@pytest.mark.parametrize('parent,holds', [(1, False), (1, True), (202, False), (202, True)])
def test_new_opener_after_snapshot_is_not_killed_or_proved_over(cleanup, parent, holds, resist):
    result = cleanup({'rows': {'202': [1, 'hfmodem.lanner.qso']}, 'devices': [202], 'held': holds,
                    'arrive': {'303': [parent, 'rigctl -P NONE f']}, 'arrive_holds': holds, 'resist': resist})
    if parent == 202 and holds:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'CLEANUP COMPLETE' in result.stdout
        assert cleanup.journal.read_text().startswith('TERM 202\nTERM 303\n')
    else:
        assert result.returncode == 1
        assert 'started after this down was issued' in result.stdout
        assert cleanup.journal.read_text() == 'TERM 202\n'


def test_cleanup_holder_tree_never_receives_sigkill(cleanup):
    result = cleanup({'rows': {'202': [1, 'hfmodem.lanner.qso']}, 'devices': [202],
                    'mutex_holders': [202], 'resist': True})
    assert result.returncode == 1
    assert 'no escalation' in result.stdout
    assert cleanup.journal.read_text() == 'TERM 202\n'


ORIGINAL_REPO = '/home/op/src/hfmodem'


RECORDER_COMMANDS = [
    ORIGINAL_REPO + '/.venv/bin/python ' + ORIGINAL_REPO
    + '/operator-data/working/safecomlink/survey-continued-20261024T202433Z-provenance/capture.py '
    + '--config ' + ORIGINAL_REPO + '/operator-data/working/safecomlink/'
    + 'survey-continued-20261024T202433Z-provenance/config.json --out '
    + ORIGINAL_REPO + '/operator-data/working/safecomlink/survey-continued-20261024T202433Z '
    + '--hours 4 --segment-seconds 600',
    ORIGINAL_REPO + '/.venv/bin/python ' + ORIGINAL_REPO
    + '/operator-data/working/safecomlink/survey-recontinued-20261024T211519Z-provenance/'
    + 'detached_recorder.py 2026-10-25T00:24:33.119404+00:00 '
    + ORIGINAL_REPO + '/operator-data/working/safecomlink/survey-recontinued-20261024T211519Z',
    ORIGINAL_REPO + '/.venv/bin/python ' + ORIGINAL_REPO + '/working/analyze.py',
]


@pytest.mark.parametrize('command', RECORDER_COMMANDS)
def test_unrelated_repo_orphan_survives_down_alongside_real_legacy_modem(cleanup, command):
    result = cleanup({'rows': {'202': [1, 'hfmodem.shrike.onair'],
                            '682': [1, command]}, 'devices': [202]})
    assert result.returncode == 0, result.stdout + result.stderr
    log = cleanup.journal.read_text()
    assert 'TERM 202' in log
    assert 'TERM 682' not in log and 'KILL 682' not in log
    assert 'CLEANUP COMPLETE' in result.stdout


@pytest.mark.parametrize('command', RECORDER_COMMANDS[:2])
def test_repo_orphan_with_unidentified_device_still_blocks_without_signal(cleanup, command):
    result = cleanup({'rows': {'682': [1, command]}, 'devices': [682]})
    assert result.returncode != 0
    assert 'unidentified device/claim holder' in result.stdout
    assert 'TERM' not in cleanup.journal.read_text()
    assert 'CLEANUP COMPLETE' not in result.stdout


@pytest.mark.parametrize('command', [
    'python -m hfmodem.besra.host.run_server', 'python -m hfmodem.lanner.qso',
    'python -m hfmodem.sabir.onair', '/tmp/run_server.py',
    '/tmp/kestrel_connect.py', '/tmp/kiwi_witness.py', 'python -m creance.cli',
    'rigctld -m 1036',
])
def test_named_legacy_radio_orphan_still_reaped(cleanup, command):
    result = cleanup({'rows': {'202': [1, command]}})
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TERM 202' in cleanup.journal.read_text()



@pytest.mark.parametrize('ended', [False, True])
@pytest.mark.parametrize('changed', ['none', 'parent', 'child', 'nonce'])
def test_cleanup_inheritor_requires_matching_parent_child_and_nonce(ended, changed):
    backend = down_posix.Posix(None, (), None)
    backend.lineage[303] = ('child', 202, 'parent')
    (backend.ended if ended else backend.stopping)[202] = (
        'reused-parent' if changed == 'parent' else 'parent')
    identities = {303: 'reused-child' if changed == 'child' else 'child'}
    matched = set() if changed == 'nonce' else {303}
    assert backend.inheritor(303, matched, identities) is (changed == 'none')


LISTING = ('1 0 0 Sat Oct 17 12:00:00 2026 init\n'
           '2 1 0 Sat Oct 17 12:00:00 2026 between\n'
           '3 2 0 Sat Oct 17 12:00:00 2026 below\n'
           '4 1 0 Sat Oct 17 12:00:00 2026 ps -axo pid=,ppid=,uid=,lstart=,command=\n')


@pytest.fixture
def linux_rows(monkeypatch):
    gone = set()
    def started(pid):
        if pid in gone:
            raise claim.ClaimError(f'cannot identify pid {pid}')
        return f'linux:boot:{pid}'
    monkeypatch.setattr(down_posix.sys, 'platform', 'linux')
    monkeypatch.setattr(claim, 'process_started', started)
    monkeypatch.setattr(claim, '_exited', lambda pid: pid in gone)
    return gone


def test_a_linux_table_drops_a_childless_row_that_exits_while_identified(linux_rows):
    # The listing's own ps has always exited by the time its rows are identified.
    linux_rows.add(4)
    rows = down_posix.Posix.rows(None, LISTING, float('inf'))
    assert sorted(rows) == [1, 2, 3] and rows[3] == (2, 0, 'below', 'linux:boot:3')


def test_a_linux_table_whose_parent_row_exits_while_identified_is_retaken(linux_rows):
    linux_rows.update({2, 4})
    assert down_posix.Posix.rows(None, LISTING, float('inf')) is None


def test_a_linux_table_refuses_a_row_it_cannot_identify_while_it_lives(linux_rows, monkeypatch):
    linux_rows.add(2)
    monkeypatch.setattr(claim, '_exited', lambda pid: False)
    with pytest.raises(claim.ClaimError, match='cannot identify pid 2'):
        down_posix.Posix.rows(None, LISTING, float('inf'))


@pytest.mark.parametrize('reads', [0, 4], ids=['before-the-first-read', 'after-the-last-read'])
def test_a_linux_table_is_refused_once_its_deadline_passes(linux_rows, monkeypatch, reads):
    clock = iter(range(100))
    monkeypatch.setattr(down_posix.time, 'monotonic', lambda: next(clock))
    with pytest.raises(claim.ClaimError, match='deadline expired'):
        down_posix.Posix.rows(None, LISTING, reads)


def test_a_process_table_cut_by_its_timeout_is_a_discovery_failure(monkeypatch):
    """The settle window's last inventory retries a ClaimError with the final pass's
    bound; a bare TimeoutExpired escaped it in place of the refusal naming the holder."""
    def run(argv, **kwargs):
        raise down_posix.subprocess.TimeoutExpired(argv, kwargs['timeout'])
    monkeypatch.setattr(down_posix.subprocess, 'run', run)
    with pytest.raises(claim.ClaimError, match='process inventory unavailable'):
        down_posix.Posix.table(None, time.monotonic() + 1)
