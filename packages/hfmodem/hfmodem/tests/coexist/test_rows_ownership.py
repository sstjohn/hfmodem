# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
import shlex
import signal
import socket
import textwrap

import pytest

from .conftest import TREE
from .oracle import OracleFailure, assert_claim_possession, validate
from .test_rows_admission import (LINE_CASES, METADATA, binary, command, completed, finish, held,
                                  launch, line_low, observe, proof_line, shell_argv)

__all__ = ['proof_line']


STATION = '''
from hfmodem.core import config
from hfmodem.station.process import Station
cfg = config.StationConfig(
    station=config.Station(mycall='K1ABC', transmit=False),
    rig=config.Rig(cat='none'),
    audio=config.Audio(input='Coexist fake audio', output='Coexist fake audio'), protocols={},
    resolved_profile={'cat_port': os.environ['CAT_PORT']})
station = Station(cfg)
'''


CONSOLE = '''
import errno
from pathlib import Path
from types import SimpleNamespace
from consolelib import claim as console_claim, station as selection
root = Path(os.environ['HFMODEM_STATE_DIR'])
def absent(self, *args, **kwargs):
    raise ConnectionRefusedError(errno.ECONNREFUSED, 'test has no station service')
console_claim.station_client.Client._json = absent
'''


def station_file(harness, *, cat='none', port=None):
    if port is None:
        with socket.socket() as listener:
            try:
                listener.bind(('127.0.0.1', 0))
            except PermissionError:
                pytest.skip('sandbox blocks loopback bind; conductor reruns outside')
            port = listener.getsockname()[1]
    path = harness.root / 'station.toml'
    path.write_text(f'''schema = 1
[station]
port = {port}
mycall = "K1ABC"
control = "local"
regulatory = "unregulated"
because = "coexist fake devices"
transmit = false
[rig]
model = "ft891"
cat = "{cat}"
[audio]
input = "Coexist fake audio"
output = "Coexist fake audio"
''')
    return path


def shell_setup(harness):
    inventory = harness.root / 'inventory.json'
    if not inventory.exists():
        inventory.write_text('{}')
    binary(harness, 'ps', '''
        import ctypes, json, os, subprocess, sys
        from pathlib import Path
        rows = json.loads((Path(os.environ['HFMODEM_STATE_DIR']) / 'inventory.json').read_text())
        def started(pid):
            if os.environ.get('ROW_FAKE_METADATA') == '1':
                return 'Sat Oct 17 12:00:00 2026'
            result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'lstart='],
                                    capture_output=True, text=True, check=True)
            return result.stdout.strip()
        def state(pid):
            if os.environ.get('ROW_FAKE_METADATA') == '1':
                info = (ctypes.c_uint32 * 34)()
                lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
                size = lib.proc_pidinfo(pid, 3, 0, info, ctypes.sizeof(info))
                return 'Z' if size == ctypes.sizeof(info) and info[1] == 5 else 'S'
            result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat='],
                                    capture_output=True, text=True, check=True)
            return result.stdout.strip()
        root = int(os.environ.get('ROW_SHELL_PID', os.getppid()))
        if '-axo' in sys.argv:
            rows[str(root)] = {'command': 'bash fake-row', 'parent': 1}
            rows[str(os.getppid())] = {'command': 'python helper', 'parent': root}
            for pid, row in rows.items():
                try:
                    os.kill(int(pid), 0)
                except ProcessLookupError:
                    continue
                try:
                    if state(pid).startswith('Z'):
                        continue
                    birth = started(pid)
                except subprocess.CalledProcessError:
                    # A row can disappear between the two real ps queries.
                    # Unknown/live failures must still fail the whole fixture.
                    try:
                        os.kill(int(pid), 0)
                    except ProcessLookupError:
                        continue
                    raise
                print(pid, row.get('parent', 1), os.getuid(),
                      birth, row['command'])
        else:
            pid = int(sys.argv[sys.argv.index('-p') + 1])
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                sys.exit(1)
            print(started(pid) if 'lstart=' in sys.argv
                  else state(pid) if 'stat=' in sys.argv else rows.get(str(pid), {}).get('command', 'coexist-participant'))
    ''')
    binary(harness, 'python', '''
        import os, sys
        if sys.argv[1:3] == ['-m', 'hfmodem.core.rigs']:
            print('generic\\n1\\n38400\\nUSB\\nCoexist fake audio\\n'
                  + os.environ['CAT_PORT'] + '\\n' + os.environ['PTT_PORT']
                  + '\\nCoexist fake audio')
        else:
            os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
    ''')
    binary(harness, 'lsof', '''
        import json, os, subprocess, sys
        from pathlib import Path
        rows = json.loads((Path(os.environ['HFMODEM_STATE_DIR']) / 'inventory.json').read_text())
        for pid, row in rows.items():
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                continue
            status = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='],
                                    capture_output=True, text=True)
            if status.returncode or status.stdout.strip().startswith('Z'):
                continue
            if '-t' in sys.argv:
                if any('-iTCP:' + str(port) in sys.argv for port in row.get('listeners', [])):
                    print(pid)
            else:
                for descriptor in row.get('descriptors', []):
                    if descriptor.get('path') in sys.argv:
                        print('p' + pid + '\\nc' + row['command'].split()[0]
                              + '\\nf' + str(descriptor['fd']) + '\\no0t' + str(descriptor.get('pos', 0)))
    ''')
    binary(harness, 'rigctl', '''
        import sys
        from hfmodem.core import claim, coexist
        claim.inherited(9)
        coexist.event('fake_rigctl', argv=sys.argv)
        print('ptt_type=None' if '--get-conf=ptt_type' in sys.argv else '7100000')
    ''')
    binary(harness, 'rigctld', 'raise SystemExit("unexpected daemon launch")')
    binary(harness, 'timeout', '''
        import os, sys
        args = sys.argv[1:]
        if args[0] == '-k':
            args = args[2:]
        os.execvp(args[1], args[1:])
    ''')
    harness.env['HFMODEM_PYTHON'] = str(harness.root / 'bin/python')
    return ('set -euo pipefail\nexport ROW_SHELL_PID=$$\n'
            f'source {shlex.quote(str(TREE / "tools/lib/ports.sh"))}\n'
            f'source {shlex.quote(str(TREE / "tools/lib/rig.sh"))}\n')


def inventory(harness, commands):
    snapshot = harness.observer.capture('shell-inventory')
    rows = {str(pid): dict(command=command, descriptors=snapshot.descriptors.get(pid, []))
            for pid, command in commands.items()}
    (harness.root / 'inventory.json').write_text(json.dumps(rows))


def outsider(harness, source):
    inner = METADATA + textwrap.dedent(source)
    return launch(harness, f'''
        import subprocess, sys
        child = subprocess.Popen([sys.executable, '-c', {inner!r}], stdin=subprocess.PIPE)
        coexist.event('native_spawn', child_pid=child.pid,
                      child_started=claim.process_started(child.pid), argv=['coexist-outsider'])
        child.stdin.close()
        assert child.wait(timeout=30) in (0, -15, -9)
    ''')


def down(harness, setup, *, line=''):
    path = harness.root / 'station.toml'
    if not path.exists():
        path.write_text('[station]\nport = 42371\n[rig]\ncat = "none"\n')
    env = dict(harness.env, HFMODEM_STATION=str(path))
    return command(harness, ['/bin/bash', '-c', setup + '\nPTT_PORT=' + shlex.quote(line)
                             + '\nstation_down'], env=env)


def test_cat_none_claims(harness):
    observe(harness)
    setup = shell_setup(harness)
    harness.pin('held')
    owner = launch(harness, STATION + '''
station.own(receive=True)
coexist.checkpoint('held')
station.release()
''')
    harness.reached('held', quiescent=True)
    assert any(e['event'] == 'device_open' and e['path'] == str(harness.audio)
               for e in harness.events())
    harness.release('held')
    completed(owner)
    cleanup = down(harness, setup)
    out, _ = completed(cleanup)
    assert 'PTT LINE none\nCAT none\nCLEANUP COMPLETE' in out
    finish(harness)


def test_shell_metadata_matches_observer(harness):
    observe(harness)
    shell_setup(harness)
    harness.pin('held')
    owner = launch(harness, STATION + '''
station.own(receive=True)
coexist.checkpoint('held')
station.release()
''')
    harness.reached('held', quiescent=True)
    harness.release('held')
    completed(owner)
    finish(harness)


CONSOLE_ENTRIES = {
    '_fly': '''
        from consolelib import arm
        arm.qualified = lambda mode, plan, qualify: (plan, {})
        arm._fly(SimpleNamespace(name='coexist'), dict(directory=str(root / 'flight')))
    ''',
    'monitor': '''
        import console
        from consolelib import monitor
        monitor.report(console.monitor_parser().parse_args([]))
    ''',
    'rtty-qso': '''
        from consolelib import rtty
        rtty.fly(dict(directory=str(root / 'rtty')))
    ''',
    'sabir-fly': '''
        from consolelib import sabir
        sabir.execute(dict(directory=str(root / 'sabir')))
    ''',
    'hflink': '''
        from consolelib import hflink
        from hfmodem.core import config
        cfg = config.StationConfig(station=config.Station(mycall='K1ABC'),
            rig=config.Rig(ptt=config.Ptt(port=os.environ['PTT_PORT'])),
            audio=config.Audio(input='Coexist fake audio', output='Coexist fake audio', tx_drive=.04),
            protocols={})
        hflink.configuration = lambda args, family: (cfg, [7100000])
        hflink.plan = lambda *args: dict(callsign='K1ABC', rig='generic',
            regulatory='coexist', dial_khz=[7100], sounding='off', pilot_khz=[])
        hflink.run(['station', '--config', str(root / 'station.toml'),
                    '--log-dir', str(root / 'hflink'), '--freq', '7100', '--no-sound', '--execute'])
    ''',
    'voice-fly': '''
        import sys
        from dataclasses import replace
        from consolelib import voice
        from hfmodem.core import config
        cfg = config.load(root / 'station.toml')
        voice.settings = lambda args: replace(cfg,
            station=replace(cfg.station, transmit=True),
            voice=replace(cfg.voice, language_model=str(root / 'fake-model')))
        sys.stdin = SimpleNamespace(isatty=lambda: True)
        voice.run(['fly', '--execute', '--waveform', 'cw', '--dial', '7.1',
                   '--log', str(root / 'voice')])
    ''',
}


@pytest.mark.noline
@pytest.mark.parametrize('entry', CONSOLE_ENTRIES)
def test_console_prepares_under_claim_and_monitor_hflink_rtty_sabir_claim(harness, entry):
    observe(harness)
    path = station_file(harness, port=42371)
    harness.env['HFMODEM_STATION'] = str(path)
    harness.pin('station-held')
    owner = launch(harness, STATION + '''
station.own(receive=True)
coexist.checkpoint('station-held')
station.release()
''')
    event = harness.reached('station-held', quiescent=True)
    body = textwrap.indent(textwrap.dedent(CONSOLE_ENTRIES[entry]).strip(), '    ')
    contender = launch(harness, CONSOLE + f'''
def forbidden(*args, **kwargs):
    raise AssertionError('console prepared before admission')
selection.prepare_radio = forbidden
selection.check_rig = forbidden
try:
{body}
except claim.ClaimBusy as exc:
    assert 'station' in str(exc) and str({event['pid']}) in str(exc), str(exc)
else:
    raise AssertionError('console bypassed station claim')
''')
    try:
        completed(contender)
    finally:
        harness.release('station-held')
        completed(owner)
    evidence = finish(harness)
    assert all(e['pid'] == event['pid'] for e in evidence.events if e['event'] == 'device_open')
    assert not any(e['event'] == 'native_spawn' for e in evidence.events)


@pytest.mark.noline
@pytest.mark.parametrize('mode', ['pactor', 'vara', 'ardop'])
@pytest.mark.parametrize('blocker', ['station', 'cleanup'])
def test_assessment_under_claim_refused(harness, mode, blocker):
    observe(harness)
    path = station_file(harness, port=42371)
    harness.env['HFMODEM_STATION'] = str(path)
    harness.pin('assessment-blocked')
    owner = launch(harness, (STATION + '''
station.own(receive=True)
coexist.checkpoint('assessment-blocked')
station.release()
''' if blocker == 'station' else '''
mutex = claim.mutex(timeout=0)
coexist.checkpoint('assessment-blocked')
claim.mutex_release(mutex)
'''))
    event = harness.reached('assessment-blocked', quiescent=True)
    child = launch(harness, CONSOLE + f'''
from consolelib import {mode} as assessor
try:
    assessor.assess(None, dict(call='K1XYZ', dial_hz=7100000, bw_hz=500), 1)
except claim.ClaimBusy as exc:
    assert {'station' if blocker == 'station' else 'cleanup in progress'!r} in str(exc), str(exc)
else:
    raise AssertionError('assessment admitted over blocker')
''')
    try:
        completed(child)
    finally:
        harness.release('assessment-blocked')
        completed(owner)
    evidence = finish(harness)
    assert all(e['pid'] == event['pid'] for e in evidence.events if e['event'] == 'device_open')
    assert not any(e['event'] == 'native_spawn' for e in evidence.events)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('mode', ['pactor', 'vara', 'ardop'])
def test_assessment_under_claim_between_campaign_attempts(harness, request, with_line, mode):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    station_file(harness)
    setup = shell_setup(harness)
    harness.env['TMPDIR'] = str(harness.root)
    if line:
        harness.env['PTT_PORT'] = line[0]
    harness.pin('audio_before_open')
    bandwidth = shlex.quote({'pactor': '', 'vara': '2300', 'ardop': '500'}[mode])
    script = (setup + '\nclaim_rig campaign\n'
              'claim_twin inherited 9\n'
              'REPO="$HFMODEM_STATE_DIR"\n'
              'rc=0\n'
              f'sense_channel 7100000 {mode} {bandwidth} 0 || rc=$?\n'
              '[ "$rc" = 1 ]\n'
              'claim_twin inherited 9\n'
              'trap - EXIT INT TERM HUP\n'
              'claim_close\nrm -f "$CLAIM_CHILDREN"\n')
    campaign = command(harness, ['/bin/bash', '-c', script])
    event = harness.reached('audio_before_open', timeout=10, quiescent=True)
    path = harness.root / 'rig.claim'
    nonce = int(path.read_text().split()[4])
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'],
                            claim_path=path, nonce=nonce)
    assert path.read_text().split()[3] == 'campaign'
    competitor = launch(harness, '''
        try:
            claim.mutex(timeout=0)
        except claim.ClaimBusy as exc:
            assert 'admission in progress' in str(exc)
        else:
            raise AssertionError('cleanup overtook campaign assessment')
    ''')
    completed(competitor)
    harness.release('audio_before_open')
    out, _ = completed(campaign)
    assert 'not live enough to judge' in out
    evidence = finish(harness)
    assert sum(e['event'] == 'claim_acquired' for e in evidence.events) == 1
    assert any(e['event'] == 'device_open' and e['pid'] == event['pid'] for e in evidence.events)
    if line:
        assert line_low(line)


@pytest.mark.noline
@pytest.mark.parametrize('kind', ['flight', 'monitor', 'hflink', 'rtty', 'voice', 'sabir', 'campaign'])
def test_console_session_refuses_fly_naming_holder(harness, kind):
    observe(harness)
    path = station_file(harness)
    harness.pin('session-held')
    owner = launch(harness, STATION + f'''
from hfmodem.station.service import Service
service = Service({str(path)!r}, station=station)
station.service = service
service.bind()
try:
    station.own(receive=True)
    session = station.sessions.start(kind={kind!r}, lane='fake', attended=False,
        directory=os.environ['HFMODEM_STATE_DIR'], client='coexist-session-holder')
    coexist.event('session_holder', session_id=session.id)
    coexist.checkpoint('session-held')
finally:
    station.release()
    service.close()
''')
    event = harness.reached('session-held', quiescent=True)
    session = next(e['session_id'] for e in harness.events() if e['event'] == 'session_holder')
    contender = launch(harness, f'''
        from hfmodem.station.client import Client, Claimed
        client = Client({str(path)!r})
        assert client.state()['owner']['id'] == {session!r}
        try:
            client.command('fly', wait=True)
        except Claimed as exc:
            assert {session!r} in str(exc) and 'coexist-session-holder' in str(exc), str(exc)
        else:
            raise AssertionError('fly displaced the session holder')
        assert client.state()['owner']['id'] == {session!r}
    ''')
    try:
        completed(contender)
    finally:
        harness.release('session-held')
        completed(owner)
    evidence = finish(harness)
    assert all(e['pid'] == event['pid'] for e in evidence.events if e['event'] == 'device_open')


def console_flight(harness, setup, *, line='', station_present=False, competing=False):
    from .test_rows_cleanup import _console_source

    modem = METADATA + '''
import sounddevice as sd
admission = claim.inherited(9, launch=True)
with sd.InputStream() as stream:
    admission.opened()
    coexist.checkpoint('console-modem-open')
claim.close(9)
'''
    source = _console_source(harness, 'preparation', modem=modem)
    source = 'import subprocess, sys\nfrom pathlib import Path\n' + source
    source = source.replace('console_claim.release = lambda *a, **kw: None', '''
release = console_claim.release
def releasing(*args, **kwargs):
    result = release(*args, **kwargs)
    coexist.checkpoint('console-release-gap')
    return result
console_claim.release = releasing
''')
    if not station_present:
        source = CONSOLE + source
    if competing:
        source = source.replace('except KeyboardInterrupt:\n    pass', '''
except claim.ClaimBusy as exc:
    assert 'station' in str(exc), str(exc)
    coexist.event('console_lost_claim')
else:
    raise AssertionError('console displaced the competing station')
''')
    env = dict(harness.env, RIG_CLAIMED=str(harness.root / 'rig.claim'))
    argv = shell_argv(['/bin/bash', '-c', setup + '\nPTT_PORT=' + shlex.quote(line) + '\nstation_down'], env)
    (harness.root / 'fake-down').write_text('#!/bin/bash\n' + shlex.join(argv) + '\n')
    return source


@pytest.mark.parametrize('with_line', LINE_CASES)
def test_console_prepares_under_claim_without_station(harness, request, with_line):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    path = station_file(harness)
    harness.env['HFMODEM_STATION'] = str(path)
    if line:
        harness.env['PTT_PORT'] = line[0]
    setup = shell_setup(harness)
    for point in ('console_point', 'console-modem-open'):
        harness.pin(point)
    console = launch(harness, console_flight(harness, setup, line=line[0] if line else ''))
    prepared = harness.reached('console_point', quiescent=True)
    claim_path = harness.root / 'rig.claim'
    nonce = int(claim_path.read_text().split()[4])
    assert_claim_possession(harness.observer.snapshots[-1], pid=prepared['pid'],
                            claim_path=claim_path, nonce=nonce)
    assert not any(e['event'] == 'device_open' for e in harness.events())
    harness.release('console_point')
    child = harness.reached('console-modem-open', quiescent=True)
    assert_claim_possession(harness.observer.snapshots[-1], pid=child['pid'],
                            claim_path=claim_path, nonce=nonce)
    harness.release('console-modem-open')
    out, _ = completed(console)
    assert 'cleanup NOT CONFIRMED' not in out
    evidence = finish(harness)
    assert {e['pid'] for e in evidence.events if e['event'] == 'device_open'} >= {child['pid']}
    if line:
        assert line_low(line)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('winner', ['legacy', 'own'])
def test_console_release_then_fly_legacy(harness, request, with_line, winner):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    path = station_file(harness)
    harness.env['HFMODEM_STATION'] = str(path)
    if line:
        harness.env['PTT_PORT'] = line[0]
    setup = shell_setup(harness)
    for point in ('station-held', 'console-release-gap', 'console_point', 'console-modem-open'):
        harness.pin(point)
    owner = launch(harness, STATION + f'''
from hfmodem.station.service import Service
service = Service({str(path)!r}, station=station)
station.service = service
service.bind()
try:
    station.own(receive=True)
    coexist.checkpoint('station-held')
finally:
    station.release()
    service.close()
''')
    station = harness.reached('station-held', quiescent=True)
    console = launch(harness, console_flight(harness, setup, line=line[0] if line else '',
                                            station_present=True, competing=winner == 'own'))
    harness.reached('console-release-gap', quiescent=True)
    assert not any(d.get('path') == str(harness.audio)
                   for d in harness.observer.snapshots[-1].descriptors.get(station['pid'], []))
    if winner == 'legacy':
        harness.release('console-release-gap')
        harness.reached('console_point', quiescent=True)
    contender = launch(harness, f'''
        from hfmodem.station.client import Client, Claimed
        client = Client({str(path)!r})
        try:
            client.own(receive=True)
        except Claimed as exc:
            assert {winner == 'legacy'!r} and 'rig claimed' in str(exc), str(exc)
        else:
            assert {winner == 'own'!r}, 'station displaced console'
    ''')
    completed(contender)
    harness.release('console-release-gap')
    if winner == 'legacy':
        harness.release('console_point')
        opened = harness.reached('console-modem-open', quiescent=True)
        assert_claim_possession(harness.observer.snapshots[-1], pid=opened['pid'],
            claim_path=harness.root / 'rig.claim',
            nonce=int((harness.root / 'rig.claim').read_text().split()[4]))
        harness.release('console-modem-open')
    completed(console)
    harness.release('station-held')
    completed(owner)
    evidence = finish(harness)
    assert bool([e for e in evidence.events if e['event'] == 'console_lost_claim']) == (winner == 'own')
    if line:
        assert line_low(line)


@pytest.mark.noline
@pytest.mark.parametrize('case', ['old-code', 'detached', 'missing-line', 'partial-line'])
def test_old_station_reported(harness, fake_yaesu, case):
    observe(harness, [fake_yaesu.port])
    path = station_file(harness, cat='yaesu')
    harness.env['HFMODEM_STATION'] = str(path)
    harness.env['CAT_PORT'] = fake_yaesu.port
    setup = shell_setup(harness)
    harness.pin('old-station-held')
    detached = METADATA + textwrap.dedent('''
        import sys
        sys.stdin.buffer.read()
        os.setsid()
        fd = os.open(os.environ['CAT_PORT'], os.O_RDWR | os.O_NOCTTY)
        coexist.checkpoint('old-station-held')
        os.close(fd)
    ''')
    owner = launch(harness, f'''
        if {case!r} == 'detached':
            import subprocess, sys
            child = subprocess.Popen([sys.executable, '-c', {detached!r}], stdin=subprocess.PIPE)
            os._exit(0)
        fd = os.open(os.environ['CAT_PORT'], os.O_RDWR | os.O_NOCTTY)
        path = claim.claim_path()
        if {case!r} == 'missing-line':
            path.write_bytes(b'')
        elif {case!r} == 'partial-line':
            path.write_bytes(b'123 partial')
        coexist.checkpoint('old-station-held')
        os.close(fd)
    ''')
    event = harness.reached('old-station-held', quiescent=True)
    pid = event['pid']
    inventory(harness, {pid: 'hfmodem station old-code'})
    snapshot = harness.observer.snapshots[-1]
    assert any(fd.get('path') == fake_yaesu.port for fd in snapshot.descriptors[pid])
    contender = launch(harness, STATION.replace("cat='none'", "cat='yaesu'") + f'''
try:
    station.own(receive=True)
except claim.ClaimBusy as exc:
    assert str({pid}) in str(exc) and 'held by' in str(exc), str(exc)
else:
    raise AssertionError('own opened over old station')
assert station.state == 'observing'
''')
    try:
        completed(contender)
        out, err = completed(down(harness, setup), 1)
        assert f'station not released, held by pid {pid}' in err, (out, err)
        assert 'SIGTERM' not in out and 'SIGKILL' not in out
        assert 'CLEANUP COMPLETE' not in out and 'PTT LINE' not in out
        os.kill(pid, 0)
    finally:
        harness.release('old-station-held')
        completed(owner)
    evidence = harness.collect()
    with pytest.raises(OracleFailure, match='unlogged (sampled holder|traced opener)'):
        validate(evidence)
    assert not evidence.lost
    assert not any(e['event'] == 'device_open' for e in evidence.events)


@pytest.mark.noline
@pytest.mark.parametrize('locked', [False, True], ids=['free-lock', 'held-lock'])
def test_dead_root_discovery_unlocked_opener(harness, locked):
    observe(harness)
    station_file(harness)
    setup = shell_setup(harness)
    harness.pin('stale-unlocked-opener')
    root = launch(harness, '''
        owned = claim.acquire('legacy', 'dead-publisher')
        claim.close(owned)
    ''')
    completed(root)
    published = (harness.root / 'rig.claim').read_bytes()
    opener = launch(harness, f'''
        import fcntl
        fd = os.open(claim.claim_path(), os.O_RDWR)
        if {locked!r}:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coexist.checkpoint('stale-unlocked-opener')
        os.close(fd)
    ''')
    event = harness.reached('stale-unlocked-opener', quiescent=True)
    snapshot = harness.observer.snapshots[-1]
    assert any(fd.get('path') == str(harness.root / 'rig.claim') and fd.get('pos') == 0
               for fd in snapshot.descriptors[event['pid']])
    inventory(harness, {event['pid']: 'unrelated-file-reader'})
    try:
        out, err = completed(down(harness, setup), 2 if locked else 0)
        assert 'SIGTERM' not in out and 'SIGKILL' not in out
        assert ('CLEANUP COMPLETE' in out) is (not locked), (out, err)
        if locked:
            assert 'rig claimed:' in err
            assert (harness.root / 'rig.claim').read_bytes() == published
        os.kill(event['pid'], 0)
    finally:
        harness.release('stale-unlocked-opener')
        completed(opener)
    finish(harness)


@pytest.mark.noline
def test_generic_rig_down_kills_operator_rigctld(harness, fake_yaesu):
    observe(harness, [fake_yaesu.port])
    station_file(harness)
    setup = shell_setup(harness)
    harness.pin('operator-daemon-held')
    daemon = outsider(harness, f'''
        import signal, sys
        sys.stdin.buffer.read()
        signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
        fd = os.open({fake_yaesu.port!r}, os.O_RDWR | os.O_NOCTTY)
        coexist.event('device_open', path={fake_yaesu.port!r}, fd=fd)
        try:
            coexist.checkpoint('operator-daemon-held')
        finally:
            os.close(fd)
            coexist.event('device_close', path={fake_yaesu.port!r}, fd=fd)
    ''')
    event = harness.reached('operator-daemon-held', quiescent=True)
    inventory(harness, {event['pid']: f'rigctld -m 1 -r {fake_yaesu.port} -t 42399 -P NONE'})
    try:
        out, err = completed(down(harness, setup))
        assert 'CLEANUP COMPLETE' in out, err
        assert str(event['pid']) in out and 'SIGTERM' in out
        completed(daemon)
    finally:
        harness.release('operator-daemon-held')
    evidence = finish(harness)
    closed = next(e for e in evidence.events if e['event'] == 'device_close')
    acquired = next(e for e in evidence.events if e['event'] == 'claim_acquired')
    assert closed['monotonic_ns'] < acquired['monotonic_ns']


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('different_cwd', [False, True], ids=['absolute-file', 'same_port_and_file_resolved'])
def test_rigctld_backend_delegation(harness, request, fake_yaesu, with_line, different_cwd):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [fake_yaesu.port, *([line[0]] if line else [])])
    path = station_file(harness, cat='rigctld')
    setup = shell_setup(harness)
    harness.env['CAT_PORT'] = fake_yaesu.port
    for point in ('delegate-held', 'delegation-owned', 'delegate-released', 'delegation-finished'):
        harness.pin(point)
    daemon = outsider(harness, f'''
        import sys
        from hfmodem.tests.core.fakerig import FakeRigctld
        sys.stdin.buffer.read()
        fake = FakeRigctld(ptt_type='None')
        fd = os.open({fake_yaesu.port!r}, os.O_RDWR | os.O_NOCTTY)
        coexist.event('device_open', path={fake_yaesu.port!r}, fd=fd)
        coexist.event('delegated_listener', port=fake.port)
        try:
            coexist.checkpoint('delegate-held')
        finally:
            fake.close()
            os.close(fd)
            coexist.event('device_close', path={fake_yaesu.port!r}, fd=fd)
    ''')
    daemon_pid = harness.reached('delegate-held', quiescent=True)['pid']
    port = next(e['port'] for e in harness.events() if e['event'] == 'delegated_listener')
    path.write_text(path.read_text().replace('[audio]', f'host = "127.0.0.1"\nport = {port}\n[audio]'))
    owner = launch(harness, f'''
        from hfmodem.core import config
        from hfmodem.station.process import Station
        from hfmodem.station.service import Service
        station = Station(config.load({str(path)!r}), station_file={str(path)!r})
        service = Service({str(path)!r}, station=station)
        station.service = service
        service.bind()
        release = station.release
        def released(**kwargs):
            result = release(**kwargs)
            coexist.checkpoint('delegate-released')
            return result
        station.release = released
        try:
            station.own(receive=True)
            if {with_line!r}:
                station._line = claim.line_factory(os.environ['ROW_PTT'])()
            coexist.checkpoint('delegation-owned')
            coexist.checkpoint('delegation-finished')
        finally:
            release()
            service.close()
    ''')
    station_pid = harness.reached('delegation-owned', quiescent=True)['pid']

    def refresh():
        inventory(harness, {daemon_pid: f'rigctld -m 1 -r {fake_yaesu.port} -t {port} -P NONE',
                            station_pid: f'hfmodem station {path}'})
        target = harness.root / 'inventory.json'
        rows = json.loads(target.read_text())
        rows[str(daemon_pid)]['listeners'] = [port]
        target.write_text(json.dumps(rows))

    refresh()
    cwd = harness.root / 'another-cwd' if different_cwd else harness.root
    cwd.mkdir(exist_ok=True)
    selected = os.path.relpath(path, cwd) if different_cwd else str(path)
    cleanup = command(harness, ['/bin/bash', '-c', setup + '\nPTT_PORT='
                                + shlex.quote(line[0] if line else '') + '\nstation_down'],
                      env=dict(harness.env, HFMODEM_STATION=selected), cwd=cwd)
    try:
        harness.reached('delegate-released', timeout=10, quiescent=True)
        refresh()
        harness.release('delegate-released')
        out, err = completed(cleanup)
        assert f'delegated rigctld pid {daemon_pid}' in out
        assert 'CAT ok\nCLEANUP COMPLETE' in out, err
        assert 'SIGTERM' not in out and 'SIGKILL' not in out
        os.kill(daemon_pid, 0)
        commands = [e['argv'] for e in harness.events() if e['event'] == 'fake_rigctl']
        assert any(argv[argv.index('-m') + 1] == '2'
                   and argv[argv.index('-r') + 1] == f'127.0.0.1:{port}' for argv in commands)
    finally:
        for point in ('delegate-held', 'delegation-owned', 'delegate-released', 'delegation-finished'):
            harness.release(point)
    completed(owner)
    completed(daemon)
    evidence = finish(harness)
    assert all(e['pid'] == daemon_pid for e in evidence.events
               if e['event'] == 'device_open' and e['path'] == fake_yaesu.port)
    if line:
        assert line_low(line)


@pytest.mark.proof
@pytest.mark.parametrize('entry', ['collector', 'hfcapture', 'onair-listen', 'vara-attempt', 'ardop-attempt'])
def test_daemon_ptt_none(harness, proof_line, fake_yaesu, entry):
    observe(harness, [proof_line[0], fake_yaesu.port])
    path = station_file(harness)
    setup = shell_setup(harness)
    harness.env.update(CAT_PORT=fake_yaesu.port, PTT_PORT=proof_line[0],
                       HFMODEM_STATION=str(path), TMPDIR=str(harness.root))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    harness.env['ROW_DAEMON_PORT'] = str(port)
    for point in ('daemon-ready', 'daemon-get-conf', 'daemon-checked'):
        harness.pin(point)
    binary(harness, 'rigctld', '''
        import os, signal, socket, sys
        from hfmodem.core import claim, coexist
        from hfmodem.tests.core.fakerig import FakeRigctld
        claim.inherited(9)
        assert '--set-conf=ptt_type=None' in sys.argv, sys.argv
        device = sys.argv[sys.argv.index('-r') + 1]
        assert device == os.environ['CAT_PORT']
        bind = socket.socket.bind
        socket.socket.bind = lambda self, address: bind(self, ('127.0.0.1', int(os.environ['ROW_DAEMON_PORT'])))
        daemon = FakeRigctld(ptt_type='None')
        fd = os.open(device, os.O_RDWR | os.O_NOCTTY)
        coexist.event('device_open', path=device, fd=fd)
        coexist.event('daemon_argv', argv=sys.argv)
        def stop(*args):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        try:
            coexist.checkpoint('daemon-ready')
            signal.pause()
        finally:
            daemon.close()
            os.close(fd)
            coexist.event('device_close', path=device, fd=fd)
    ''')
    binary(harness, 'rigctl', '''
        import os, socket, sys
        from hfmodem.core import claim, coexist
        claim.inherited(9)
        assert '--get-conf=ptt_type' in sys.argv, sys.argv
        coexist.checkpoint('daemon-get-conf')
        with socket.create_connection(('127.0.0.1', int(os.environ['ROW_DAEMON_PORT'])), timeout=2) as conn:
            conn.sendall(b'+\\\\get_conf ptt_type\\n')
            reply = conn.makefile('rb')
            lines = []
            while not (line := reply.readline()).startswith(b'RPRT '):
                assert line
                lines.append(line)
            assert line == b'RPRT 0\\n'
            print(b''.join(lines).decode(), end='')
    ''')
    if entry == 'collector':
        body = f'''
import socket
from types import SimpleNamespace
import corpus_watch
bind = socket.socket.bind
socket.socket.bind = lambda self, address: bind(self, ('127.0.0.1', {port}) if address[1] == 4537 else address)
request = corpus_watch.request
corpus_watch.request = lambda address, command: request(('127.0.0.1', {port}), command)
args = SimpleNamespace(cat_port=os.environ['CAT_PORT'], device='Coexist fake audio',
                       rigctld=None, model=1, baud=38400)
with corpus_watch.receiver(args, Path(os.environ['HFMODEM_STATE_DIR'])):
    coexist.checkpoint('daemon-checked')
'''
    elif entry == 'hfcapture':
        body = f'''
import hfcapture
from hfcapture import close_native, stop_native
hfcapture.RIGCTLD = ('127.0.0.1', {port})
owned = claim.acquire('tool', 'hfcapture-daemon')
children = []
with (Path(os.environ['HFMODEM_STATE_DIR']) / 'daemon.log').open('w') as log:
    child = hfcapture.start_rigctld(1, os.environ['CAT_PORT'], 38400,
                                  claim_fd=owned, log=log, children=children)
    coexist.checkpoint('daemon-checked')
    stop_native(child)
    for _, readers in children:
        for reader in readers:
            reader.join(2)
            assert not reader.is_alive()
    close_native(child, stopped=True)
claim.release(owned)
'''
    else:
        if entry == 'onair-listen':
            argv = ['/bin/bash', str(TREE / 'tools/onair.sh'), 'listen', '7101500', '1']
        else:
            attempts = shlex.quote(str(TREE / 'tools/lib/attempts.sh'))
            script = (setup + f'\nsource {attempts}\nOUT="$HFMODEM_STATE_DIR"\n'
                      'claim_rig campaign\n'
                      'tune() { return 0; }\n'
                      'gate_rx_unattended() { return 1; }\n'
                      'gate_unattended() { return 1; }\n'
                      + ('vara_try K1XYZ 7101500' if entry == 'vara-attempt'
                         else 'ardop_try K1XYZ 7101500 500'))
            argv = ['/bin/bash', '-c', script]
        argv = shell_argv(argv, dict(harness.env, RIG_CLAIMED=str(harness.root / 'rig.claim')))
        body = f'''
import subprocess
owned = claim.acquire('campaign', 'daemon-constructor')
os.dup2(owned.fd, 9)
if owned.fd != 9:
    claim.close(owned)
child = subprocess.Popen({argv!r}, pass_fds=(9,))
assert child.wait(timeout=30) == 1
claim.release(9)
'''
    parent = launch(harness, 'from pathlib import Path\n' + textwrap.dedent(body))
    event = harness.reached('daemon-ready', timeout=10, quiescent=True)
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'],
        claim_path=harness.root / 'rig.claim',
        nonce=int((harness.root / 'rig.claim').read_text().split()[4]))
    shell_entry = entry not in ('collector', 'hfcapture')
    if shell_entry:
        inventory(harness, {event['pid']: f'rigctld -m 1 -r {fake_yaesu.port} -P NONE'})
        harness.reached('daemon-get-conf', quiescent=True)
        assert parent.poll() is None
    else:
        harness.reached('daemon-checked', quiescent=True)
    check = launch(harness, f'''
        import socket
        from types import SimpleNamespace
        from hfmodem.core.rig import Cat, Rig
        cat = Cat('127.0.0.1', {port})
        try:
            assert Rig._check_ptt_owner(SimpleNamespace(cat=cat), [])
        finally:
            cat.close()
        with socket.create_connection(('127.0.0.1', {port}), timeout=2) as connection:
            connection.sendall(b'T 1\\n')
            assert connection.makefile('rb').readline() == b'RPRT -11\\n'
    ''')
    try:
        completed(check)
        assert line_low(proof_line)
    finally:
        for point in ('daemon-ready', 'daemon-get-conf', 'daemon-checked'):
            harness.release(point)
    completed(parent)
    evidence = finish(harness)
    commands = [e['argv'] for e in evidence.events if e['event'] == 'daemon_argv']
    assert len(commands) == 1 and '--set-conf=ptt_type=None' in commands[0]
    assert line_low(proof_line)


@pytest.mark.parametrize('case', ['between-open-and-lock', 'legacy', 'receive-only'])
@pytest.mark.parametrize('target', ['claim', 'mutex'])
def test_claim_inode_check(harness, case, target):
    observe(harness)
    if case == 'between-open-and-lock':
        point = 'before_claim_lock' if target == 'claim' else 'after_mutex'
        harness.pin(point)
        child = launch(harness, '''
            try:
                owned = claim.acquire('legacy', 'inode-race')
            except claim.ClaimError as exc:
                assert 'descriptor does not name' in str(exc)
            else:
                claim.release(owned)
        ''')
        harness.reached(point, quiescent=True)
        path = harness.root / ('rig.claim' if target == 'claim' else 'rig.claim.cleanup')
        path.rename(path.with_suffix('.old'))
        path.touch()
        harness.release(point)
        completed(child)
        evidence = finish(harness)
        assert any(e['event'] == 'claim_acquired' for e in evidence.events) == (target == 'mutex')
        return
    harness.pin('inode-held')
    harness.pin('inode-checked')
    setup = (STATION + 'station.own(receive=True)\nowned = station.claim\n'
             if case != 'legacy' else "owned = claim.acquire('legacy', 'inode-residue')\n")
    child = launch(harness, setup + textwrap.dedent(f'''
        coexist.checkpoint('inode-held')
        if {case!r} != 'legacy':
            station.check_claim(now=station._last_inode_check + 60)
            assert station.state == {'observing' if target == 'claim' else 'owning-receive'!r}
            if {target!r} == 'claim':
                assert 'descriptor does not name' in station.reason
            station.release()
        else:
            assert owned.offset() > 0
            claim.close(owned)
        coexist.checkpoint('inode-checked')
    '''))
    harness.reached('inode-held', quiescent=True)
    path = harness.root / ('rig.claim' if target == 'claim' else 'rig.claim.cleanup')
    path.rename(path.with_suffix('.old'))
    path.touch()
    if target == 'claim':
        successor = launch(harness, '''
            owned = claim.acquire('legacy', 'new-inode-residue')
            claim.release(owned)
        ''')
        completed(successor)
    harness.release('inode-held')
    harness.reached('inode-checked', quiescent=True)
    harness.release('inode-checked')
    completed(child)
    finish(harness)


@pytest.mark.parametrize('point', ['after_mutex', 'after_claim_flock', 'before_holder_line',
                                  'after_holder_line', 'after_device_check', 'before_release',
                                  'after_release'])
@pytest.mark.parametrize('failure', ['raise', 'kill'])
def test_cat_none_claims_faults(harness, point, failure):
    observe(harness)
    env = dict(harness.env, HFMODEM_COEXIST_FAULT=f'{point}:{failure}')
    inner = METADATA + STATION + '\nstation.own(receive=True)\nstation.release()\n'
    expected = -9 if failure == 'kill' else 1
    child = launch(harness, f'''
        import subprocess, sys
        process = subprocess.run([sys.executable, '-c', {inner!r}])
        assert process.returncode == {expected}
    ''', env=env)
    completed(child)
    evidence = finish(harness)
    assert any(e['event'] == 'fault' and e['point'] == point for e in evidence.events)


def test_own_refused_while_legacy_holds(harness):
    observe(harness)
    owner = held(harness)
    child = launch(harness, STATION + '''
try:
    station.own(receive=True)
except claim.ClaimBusy as exc:
    assert 'rig claimed:' in str(exc) and 'legacy' in str(exc)
else:
    raise AssertionError('station admitted over holder')
assert station.state == 'observing'
''')
    completed(child)
    harness.release('held')
    completed(owner)
    evidence = finish(harness)
    assert not any(e['event'] == 'device_open' for e in evidence.events)


def test_release_reads_line_low_receive_only(harness):
    observe(harness)
    harness.pin('station-held')
    child = launch(harness, STATION + '''
station.own(receive=True)
coexist.checkpoint('station-held')
assert station.release()['state'] == 'observing'
assert station.claim is None
assert station.audio is None
assert station._poll_thread is None
''')
    event = harness.reached('station-held', quiescent=True)
    path = harness.root / 'rig.claim'
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                            nonce=int(path.read_text().split()[4]))
    harness.release('station-held')
    completed(child)
    evidence = finish(harness)
    opened = [e for e in evidence.events if e['event'] == 'device_open']
    closed = [e for e in evidence.events if e['event'] == 'device_close']
    released = [e for e in evidence.events if e['event'] == 'claim_released']
    assert opened and len(opened) == len(closed)
    assert max(e['monotonic_ns'] for e in closed) < released[-1]['monotonic_ns']


TOOL_OPENERS = {
    'onair_session': """
        tool.pick_targets = lambda *a: [('K1ABC', 7103500, 1, 'VARA')]
        sys.argv = ['onair_session', '--mycall', 'K1ABC', '--band', '40', '--rigctld',
                    'fake.invalid:1234', '--device', 'Coexist fake audio', '--no-record']
        tool.session()
    """,
    'rigaudio': """
        sys.argv = ['rigaudio', '--in', 'Coexist fake audio', '--out', 'Coexist fake audio']
        tool.main()
    """,
    'rx_levels': "tool.DeviceSource('Coexist fake audio').capture(1, 'test')",
    'corpus_watch': """
        args = tool.parser().parse_args(['--execute', '--cat-port', os.environ['CAT_PORT']])
        with tool.receiver(args, root):
            raise AssertionError('collector opened')
    """,
    'hfcapture': "tool.cmd_check(SimpleNamespace(device='Coexist fake audio', seconds=1))",
    'txwitness': "tool.record.__wrapped__.__wrapped__(1, str(root / 'tap.wav'), 'Coexist fake audio')",
    'vara_monitor': "next(tool.device_source('Coexist fake audio'))",
    'hfmodem.sabir.monitor': "next(tool.device_chunks('Coexist fake audio'))",
}


def _assert_unfenced_tool(harness, tool_name, body):
    observe(harness)
    harness.pin('station-held')
    owner = launch(harness, STATION + '''
station.own(receive=True)
coexist.checkpoint('station-held')
station.release()
''')
    harness.reached('station-held', quiescent=True)
    source = '''
import importlib, sys
from pathlib import Path
from types import SimpleNamespace
root = Path(os.environ['HFMODEM_STATE_DIR'])
tool = importlib.import_module(TOOL)
try:
BODY
except claim.ClaimBusy as exc:
    assert 'station' in str(exc), str(exc)
else:
    raise AssertionError('tool bypassed station claim')
'''.replace('TOOL', repr(tool_name)).replace(
        'BODY', textwrap.indent(textwrap.dedent(body).strip(), '    '))
    contender = launch(harness, source)
    try:
        completed(contender)
    finally:
        harness.release('station-held')
        completed(owner)
    evidence = finish(harness)
    owner_pid = next(e['pid'] for e in evidence.events if e['event'] == 'claim_acquired')
    assert all(e['pid'] == owner_pid for e in evidence.events if e['event'] == 'device_open')
    assert not any(e['event'] == 'native_spawn' for e in evidence.events)


@pytest.mark.parametrize('tool_name', TOOL_OPENERS)
def test_unfenced_opener_cannot_exist_tools(harness, tool_name):
    _assert_unfenced_tool(harness, tool_name, TOOL_OPENERS[tool_name])


@pytest.mark.noline
@pytest.mark.parametrize('holder', ['station', 'launcher'])
@pytest.mark.parametrize('verb', ['--arm', '--preflight'])
def test_unfenced_opener_cannot_exist_rig_cli(harness, verb, holder):
    observe(harness)
    path = station_file(harness, port=42371)
    harness.pin('station-held')
    owner = launch(harness, STATION + '''
station.own(receive=True)
coexist.checkpoint('station-held')
station.release()
''' if holder == 'station' else '''
owned = claim.acquire('legacy', 'onair-pactor')
coexist.checkpoint('station-held')
claim.release(owned)
''')
    event = harness.reached('station-held', quiescent=True)
    contender = launch(harness, f'''
        import contextlib
        import io
        from dataclasses import replace
        from hfmodem import cli
        from hfmodem.core import config, ptt
        from hfmodem.station import preflight, process
        cfg = config.load({str(path)!r})
        cfg = replace(cfg, station=replace(cfg.station, transmit=True),
                      rig=replace(cfg.rig, ptt=config.Ptt(port=os.environ['PTT_PORT'])))
        cli._config = lambda path: cfg
        def forbidden(*args, **kwargs):
            raise AssertionError('rig CLI reached hardware before acquiring the claim')
        process.cat_driver = ptt.RtsPtt = preflight.run = forbidden
        refusal = io.StringIO()
        with contextlib.redirect_stderr(refusal):
            rc = cli.main(['rig', {str(path)!r}, {verb!r}])
        assert rc == 5, f'rig CLI admitted over station: {{rc}}'
        assert {'station' if holder == 'station' else 'onair-pactor'!r} in refusal.getvalue()
        assert str({event['pid']}) in refusal.getvalue()
    ''')
    try:
        completed(contender)
    finally:
        harness.release('station-held')
        completed(owner)
    evidence = finish(harness)
    assert all(e['pid'] == event['pid'] for e in evidence.events if e['event'] == 'device_open')


@pytest.mark.noline
@pytest.mark.parametrize('failed', [False, True], ids=['closed', 'close-raised'])
def test_rig_cli_claim_covers_teardown(harness, failed):
    observe(harness)
    path = station_file(harness, port=42371)
    harness.pin('rig-teardown')
    owner = launch(harness, f'''
        from dataclasses import replace
        import sounddevice
        from hfmodem import cli
        from hfmodem.core import config, ptt, rig
        from hfmodem.station import process
        cfg = config.load({str(path)!r})
        cfg = replace(cfg, station=replace(cfg.station, transmit=True),
                      rig=replace(cfg.rig, ptt=config.Ptt(port=os.environ['PTT_PORT'])))
        cli._config = lambda path: cfg
        class Line:
            def __init__(self, port):
                assert port == os.environ['PTT_PORT']
            def open(self):
                self.stream = sounddevice.RawInputStream()
            def release(self):
                coexist.checkpoint('rig-teardown')
                self.stream.close()
                if {failed!r}:
                    raise RuntimeError('injected close failure')
        class FakeRig:
            def __init__(self, **kwargs):
                self.ptt = kwargs['ptt']
            def arm(self, **kwargs):
                return 'fake armed'
            def close(self):
                self.ptt.release()
        process.cat_driver = lambda cfg: object()
        rig.Rig = FakeRig
        ptt.RtsPtt = Line
        try:
            assert cli.main(['rig', {str(path)!r}, '--arm']) == 0
        except RuntimeError as exc:
            assert {failed!r} and str(exc) == 'injected close failure'
        else:
            assert not {failed!r}
    ''')
    event = harness.reached('rig-teardown', quiescent=True)
    path = harness.root / 'rig.claim'
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                            nonce=int(path.read_text().split()[4]))
    contender = launch(harness, '''
        try:
            claim.acquire('legacy', 'during-rig-teardown')
        except claim.ClaimBusy:
            pass
        else:
            raise AssertionError('rig CLI relinquished the claim before closing')
    ''')
    try:
        completed(contender)
    finally:
        harness.release('rig-teardown')
        completed(owner)
    evidence = finish(harness)
    closed = next(e for e in evidence.events if e['event'] == 'device_close')
    ending = next(e for e in evidence.events
                  if e['event'] == ('claim_closed' if failed else 'claim_released'))
    assert closed['monotonic_ns'] < ending['monotonic_ns']
    assert bool(path.read_bytes()) is failed


@pytest.mark.parametrize('target', ['claim', 'mutex', 'state-directory'])
def test_namespace_bootstrap(harness, target):
    observe(harness)
    namespace = harness.root / 'fresh-namespace'
    if target != 'state-directory':
        namespace.mkdir()
        if target == 'claim':
            (namespace / 'rig.claim.cleanup').touch()
        else:
            (namespace / 'rig.claim').touch()
    env = dict(harness.env, HFMODEM_STATE_DIR=str(namespace))
    child = launch(harness, '''
        claim.bootstrap()
        mutex = claim.mutex(timeout=0)
        owned = claim.acquire('cleanup', 'bootstrap', cleanup=mutex)
        assert claim.claim_path().is_file() and claim.mutex_path().is_file()
        claim.release(owned)
        claim.mutex_release(mutex)
        owned = claim.acquire('station', 'own-after-bootstrap')
        claim.release(owned)
    ''', env=env)
    completed(child)
    finish(harness)


@pytest.mark.proof
@pytest.mark.parametrize('case', ['first-key', 'later-key', 'failed-unkey', 'unreadable-unkey',
                                'overtaken-by-stop', 'partial-initialization', 'no-line-profile'])
def test_p1_line_keyer(harness, proof_line, case):
    observe(harness, [proof_line[0]])
    binary(harness, 'rigctl', 'raise SystemExit("unexpected CAT child")')
    harness.pin('keyer-checked')
    child = launch(harness, f'''
        import threading
        from hfmodem.shrike.ota import Rig
        from hfmodem.core.ptt import PttError
        owned = claim.acquire('legacy', 'p1-line-keyer')
        rig = Rig(1, os.environ['CAT_PORT'], 38400,
                  ptt_port=None if {case!r} == 'no-line-profile' else os.environ['ROW_PTT'])
        if {case!r} == 'no-line-profile':
            assert rig._keyer is None
            assert 'transmit unavailable' in rig.key_failure()
        else:
            keyer = rig._keyer
            line = keyer.transport._ptt
            sense = line.sense
            drive = line.assert_
            try:
                if {case!r} == 'partial-initialization':
                    line.sense = lambda: None
                    try:
                        keyer.arm()
                    except PttError:
                        pass
                    else:
                        raise AssertionError('unreadable arm admitted')
                    assert keyer.must_retire and not keyer.key(True)
                else:
                    keyer.arm()
                    assert sense() is False
                    if {case!r} == 'overtaken-by-stop':
                        entered, resume = threading.Event(), threading.Event()
                        original = keyer.transport.drive
                        def drive_paused(on):
                            result = original(on)
                            if on:
                                entered.set()
                                assert resume.wait(5)
                            return result
                        keyer.transport.drive = drive_paused
                        results = []
                        worker = threading.Thread(target=lambda: results.append(keyer.key(True)))
                        worker.start()
                        assert entered.wait(5)
                        rig._down.set()
                        resume.set()
                        worker.join(5)
                        assert not worker.is_alive() and results == [False]
                        assert keyer.must_retire and rig.key_failure()
                    else:
                        assert keyer.key(True)
                        assert rig.key_failure() is None
                        if {case!r} == 'later-key':
                            assert keyer.unkey() and keyer.key(True)
                        if {case!r} in ('failed-unkey', 'unreadable-unkey'):
                            line.sense = lambda: True if {case!r} == 'failed-unkey' else None
                            assert not keyer.unkey()
                            assert keyer.must_retire and rig.retired
                            assert not keyer.key(True)
                            assert rig.key_failure()
                        else:
                            assert keyer.unkey()
            finally:
                line.sense, line.assert_ = sense, drive
                keyer.hand_back()
        coexist.checkpoint('keyer-checked')
        claim.release(owned)
    ''')
    harness.reached('keyer-checked', quiescent=True)
    assert line_low(proof_line)
    harness.release('keyer-checked')
    completed(child)
    evidence = finish(harness)
    assert not any(e['event'] == 'native_spawn' for e in evidence.events)


@pytest.mark.proof
@pytest.mark.parametrize('case', ['low', 'stuck-high', 'unreadable'])
def test_release_reads_line_low(harness, proof_line, case):
    observe(harness, [proof_line[0]])
    harness.pin('release-checked')
    child = launch(harness, STATION + textwrap.dedent(f'''
        station.own(receive=True)
        line = claim.line_factory(os.environ['ROW_PTT'])()
        station._line = line
        if {case!r} == 'stuck-high':
            line.assert_(True)
            line._clear = lambda fd, bits: None
        elif {case!r} == 'unreadable':
            line._sense_fd = lambda *args: None
        try:
            station.release()
        except RuntimeError:
            assert {case!r} != 'low'
        assert station.state == {'observing' if case == 'low' else 'failed'!r}
        if {case!r} != 'low':
            assert station.claim is not None
        coexist.checkpoint('release-checked')
        if station.claim is not None:
            claim.line_proof(claim.line_factory(os.environ['ROW_PTT']))
            claim.close(station.claim)
    '''))
    event = harness.reached('release-checked', quiescent=True)
    if case != 'low':
        path = harness.root / 'rig.claim'
        assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                                nonce=int(path.read_text().split()[4]))
    assert line_low(proof_line) == (case != 'stuck-high')
    harness.release('release-checked')
    completed(child)
    assert line_low(proof_line)
    finish(harness)


def station_cleanup(harness, proof_line, *, detached=False):
    observe(harness, [proof_line[0]])
    setup = shell_setup(harness)
    for point in ('station-serving', 'station-released', 'station-finished'):
        harness.pin(point)
    path = harness.root / 'station.toml'
    source = STATION + textwrap.dedent(f'''
        import json
        from pathlib import Path
        from hfmodem.station.service import Service
        if {detached!r}:
            os.setsid()
        path = Path({str(path)!r})
        service = Service(path, station=station, port=0)
        station.service = service
        service.bind()
        path.write_text('[station]\\nport = ' + str(service.port) + '\\n[rig]\\ncat = "none"\\n')
        release = station.release
        def checked_release(**kwargs):
            result = release(**kwargs)
            coexist.checkpoint('station-released')
            return result
        station.release = checked_release
        try:
            station.own(receive=True)
            station._line = claim.line_factory(os.environ['ROW_PTT'])()
            coexist.checkpoint('station-serving')
            coexist.checkpoint('station-finished')
        finally:
            release()
            service.close()
    ''')
    owner = launch(harness, source)
    event = harness.reached('station-serving', quiescent=True)
    pid = event['pid']
    inventory(harness, {pid: f'hfmodem station {path}'})
    cleanup = down(harness, setup, line=proof_line[0])
    try:
        harness.reached('station-released', quiescent=True)
        inventory(harness, {pid: f'hfmodem station {path}'})
        assert line_low(proof_line)
        harness.release('station-released')
        out, _ = completed(cleanup)
        assert 'CLEANUP COMPLETE' in out and 'PTT LINE low' in out
        assert 'SIGTERM' not in out and 'SIGKILL' not in out
        reown = launch(harness, f'''
            from hfmodem.station.client import Client
            client = Client({str(path)!r})
            client.own(receive=True)
            assert client.state()['service']['state'] == 'owning-receive'
            client.release()
        ''')
        completed(reown)
    finally:
        for point in ('station-serving', 'station-released', 'station-finished'):
            harness.release(point)
    completed(owner)
    finish(harness)


@pytest.mark.proof
def test_detached_station_survives_down(harness, proof_line):
    station_cleanup(harness, proof_line, detached=True)


@pytest.mark.proof
def test_down_idle_station(harness, proof_line):
    station_cleanup(harness, proof_line)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('failed', [False, True], ids=['own-completes', 'own-rolls-back'])
def test_external_down_during_own(harness, request, with_line, failed):
    line = request.getfixturevalue('proof_line') if with_line else ('', None)
    observe(harness, [line[0]] if line[0] else [])
    path = station_file(harness)
    setup = shell_setup(harness)
    points = ('own-in-progress', 'transition-refused', 'station-released', 'station-finished')
    for point in points:
        harness.pin(point)
    owner = launch(harness, STATION + textwrap.dedent(f'''
        import threading
        from hfmodem.station.service import Service
        service = Service({str(path)!r}, station=station)
        station.service = service
        service.bind()
        opened = station.open_audio
        def opening():
            coexist.checkpoint('own-in-progress')
            if {failed!r}:
                raise RuntimeError('injected own failure')
            opened()
        station.open_audio = opening
        refused = service._refused
        def rejecting(command, exc):
            result = refused(command, exc)
            if command['kind'] == 'release' and exc.reason == 'transitioning':
                coexist.checkpoint('transition-refused')
            return result
        service._refused = rejecting
        release = station.release
        def releasing(**kwargs):
            result = release(**kwargs)
            coexist.checkpoint('station-released')
            return result
        station.release = releasing
        errors = []
        def own():
            try:
                station.own(receive=True)
            except RuntimeError as exc:
                errors.append(str(exc))
        worker = threading.Thread(target=own)
        worker.start()
        try:
            coexist.checkpoint('station-finished')
            worker.join(10)
            assert not worker.is_alive()
            assert errors == {['injected own failure'] if failed else []!r}, errors
        finally:
            release()
            service.close()
    '''))
    event = harness.reached('own-in-progress', quiescent=True)
    inventory(harness, {event['pid']: f'hfmodem station {path}'})
    cleanup = down(harness, setup, line=line[0])
    try:
        harness.reached('transition-refused', quiescent=True)
        assert cleanup.poll() is None
        harness.release('own-in-progress')
        harness.release('transition-refused')
        if not failed:
            harness.reached('station-released', timeout=15, quiescent=True)
            inventory(harness, {event['pid']: f'hfmodem station {path}'})
            harness.release('station-released')
        out, _ = completed(cleanup)
        assert 'CLEANUP COMPLETE' in out
        assert 'SIGTERM' not in out and 'SIGKILL' not in out
        if with_line:
            assert line_low(line)
    finally:
        for point in points:
            harness.release(point)
    completed(owner)
    evidence = finish(harness)
    assert any(e['event'] == 'claim_released' for e in evidence.events)


@pytest.mark.noline
def test_external_down_leaves_a_controlled_observing_station_observing(harness):
    observe(harness)
    path = station_file(harness)
    setup = shell_setup(harness)
    harness.pin('controlled')
    owner = launch(harness, STATION + f'''
from hfmodem.station.client import Client
from hfmodem.station.service import Service
service = Service({str(path)!r}, station=station)
station.service = service
service.bind()
try:
    Client({str(path)!r}).command('control', wait=True)
    coexist.checkpoint('controlled')
finally:
    service.close()
''')
    event = harness.reached('controlled', quiescent=True)
    inventory(harness, {event['pid']: f'hfmodem station {path}'})
    try:
        out, err = completed(down(harness, setup))
        assert 'CLEANUP COMPLETE' in out, (out, err)
    finally:
        harness.release('controlled')
    completed(owner)
    finish(harness)


def cli_detach(harness, inherited, line=''):
    observe(harness, [line] if line else ())
    path = station_file(harness)
    setup = shell_setup(harness)
    binary(harness, 'hfmodem', '''
        from hfmodem.cli import main
        from hfmodem.core import coexist
        from hfmodem.station.process import Station
        release = Station.release
        def released(self, **kwargs):
            result = release(self, **kwargs)
            coexist.checkpoint('cli-released')
            return result
        Station.release = released
        raise SystemExit(main())
    ''')
    for point in ('cli-started', 'cli-owned', 'cli-stop'):
        harness.pin(point)
    guardian = launch(harness, f'''
        from hfmodem.station.client import attach, Claimed
        owned = claim.acquire('legacy', 'attach-parent') if {inherited!r} else None
        client = attach({str(path)!r}, data_root=os.environ['HFMODEM_DATA_DIR'])
        try:
            coexist.checkpoint('cli-started')
            if owned is not None:
                try:
                    client.own(receive=True)
                except Claimed as exc:
                    assert 'attach-parent' in str(exc)
                else:
                    raise AssertionError('attach inherited the parent claim')
                claim.release(owned)
                owned = None
            client.own(receive=True)
            coexist.checkpoint('cli-owned')
            coexist.checkpoint('cli-stop')
            if not {inherited!r}:
                assert client.state()['service']['state'] == 'observing'
                client.own(receive=True)
            client.release()
        finally:
            client.command('shutdown', wait=True)
            if owned is not None:
                claim.release(owned)
    ''')
    try:
        harness.reached('cli-started', quiescent=True)
        snapshot = harness.observer.snapshots[-1]
        services = [pid for pid, row in snapshot.participants.items()
                    if ' station ' in row.get('argv', '') and '--detach' in row['argv']
                    and pid != guardian.pid]
        assert len(services) == 1, snapshot.participants
        pid = services[0]
        assert not any(fd.get('path') in (str(harness.root / 'rig.claim'),
                                         str(harness.root / 'rig.claim.cleanup'))
                       for fd in snapshot.descriptors.get(pid, []))
        harness.release('cli-started')
        harness.reached('cli-owned', quiescent=True)
        inventory(harness, {pid: f'hfmodem station {path} --detach'})
        if not inherited:
            harness.pin('cli-released')
            cleanup = down(harness, setup, line=line)
            harness.reached('cli-released', quiescent=True)
            inventory(harness, {pid: f'hfmodem station {path} --detach'})
            harness.release('cli-released')
            out, err = completed(cleanup)
            assert f'PTT LINE {"low" if line else "none"}\nCAT none\nCLEANUP COMPLETE' in out, err
            os.kill(pid, 0)
        harness.release('cli-owned')
        harness.release('cli-stop')
        completed(guardian)
    finally:
        for point in ('cli-started', 'cli-owned', 'cli-stop', 'cli-released'):
            harness.release(point)
    evidence = finish(harness)
    assert sum(e['event'] == 'claim_acquired' and e['pid'] == pid for e in evidence.events) == (
        1 if inherited else 2)


@pytest.mark.noline
@pytest.mark.parametrize('inherited', [False, True], ids=['detached-down', 'attach-no-descriptors'])
def test_detached_station_survives_down_cli_and_inherited_descriptor_proven_attach(harness, inherited):
    cli_detach(harness, inherited)


@pytest.mark.proof
def test_detached_station_survives_down_full_cli_proof(harness, proof_line):
    cli_detach(harness, False, proof_line[0])
    assert line_low(proof_line)


@pytest.mark.parametrize('with_line', LINE_CASES)
def test_release_reads_line_low_foreground_cli(harness, request, with_line):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    path = station_file(harness)
    harness.env['HFMODEM_COEXIST_INSTRUMENT'] = '1'
    harness.pin('foreground-ready')
    harness.pin('foreground-owned')
    owner = launch(harness, f'''
        from hfmodem import cli
        from hfmodem.station.process import Station
        run, open_rig = Station.run, Station.open_rig
        def instrumented_run(self, **kwargs):
            return run(self, on_ready=lambda station: coexist.checkpoint('foreground-ready'), **kwargs)
        def instrumented_open(self, **kwargs):
            result = open_rig(self, **kwargs)
            if {with_line!r}:
                self._line = claim.line_factory(os.environ['ROW_PTT'])()
            return result
        Station.run, Station.open_rig = instrumented_run, instrumented_open
        raise SystemExit(cli.main(['station', {str(path)!r}]))
    ''')
    event = harness.reached('foreground-ready', quiescent=True)
    harness.release('foreground-ready')
    client = launch(harness, f'''
        from hfmodem.station.client import Client
        client = Client({str(path)!r})
        client.own(receive=True)
        coexist.checkpoint('foreground-owned')
    ''')
    harness.reached('foreground-owned', quiescent=True)
    claim_path = harness.root / 'rig.claim'
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=claim_path,
                            nonce=int(claim_path.read_text().split()[4]))
    os.kill(event['pid'], signal.SIGINT)
    harness.release('foreground-owned')
    completed(client)
    completed(owner, 130)
    evidence = finish(harness)
    closed = [e for e in evidence.events if e['event'] == 'device_close']
    released = next(e for e in evidence.events if e['event'] == 'claim_released')
    assert closed and max(e['monotonic_ns'] for e in closed) < released['monotonic_ns']
    if line:
        assert line_low(line)


@pytest.mark.parametrize('verb', ['power', 'pactor'])
def test_power_verb_claims_and_legacy_launcher_during_down(harness, fake_yaesu, verb):
    observe(harness)
    shell_setup(harness)
    harness.pin('blocker-held')
    owner = launch(harness, ('''
        owned = claim.acquire('station', 'own')
        coexist.checkpoint('blocker-held')
        claim.release(owned)
    ''' if verb == 'power' else '''
        mutex = claim.mutex(timeout=0)
        coexist.checkpoint('blocker-held')
        claim.mutex_release(mutex)
    '''))
    harness.reached('blocker-held', quiescent=True)
    env = dict(harness.env, MYCALL='K1ABC', HFMODEM_CAPTURE_DIR=str(harness.root / 'capture'))
    for name in ('CAT_PORT', 'PTT_PORT'):
        alias = harness.root / name.lower()
        alias.symlink_to(fake_yaesu.port)
        env[name] = str(alias)
    contender = command(harness, ['/bin/bash', str(TREE / 'tools/onair.sh'), verb,
                                   *(['K1XYZ', '7101500'] if verb == 'pactor' else [])], env=env)
    try:
        out, err = completed(contender, 1)
        sandbox_pipe = verb == 'pactor' and '/dev/fd/' in err and 'Operation not permitted' in err
        if not sandbox_pipe:
            assert ('rig claimed:' if verb == 'power' else 'cleanup in progress') in out + err
    finally:
        harness.release('blocker-held')
        completed(owner)
    evidence = finish(harness)
    assert not any(e['event'] in ('device_open', 'native_spawn', 'fake_rigctl') for e in evidence.events)
    if sandbox_pipe:
        pytest.skip('sandbox blocks launcher /dev/fd process substitution; conductor reruns outside')


def test_two_station_files_one_account(harness):
    observe(harness)
    paths = [harness.root / name for name in ('first.toml', 'second.toml')]
    for path, port in zip(paths, (42371, 42372)):
        path.write_text(f'[station]\nport = {port}\n[rig]\ncat = "none"\n')
    harness.pin('first-station-held')
    owner = launch(harness, STATION + f'''
station.station_file = {str(paths[0])!r}
station.own(receive=True)
coexist.checkpoint('first-station-held')
station.release()
''', env=dict(harness.env, HFMODEM_STATION=str(paths[0])))
    harness.reached('first-station-held', quiescent=True)
    contender = launch(harness, STATION + f'''
station.station_file = {str(paths[1])!r}
try:
    station.own(receive=True)
except claim.ClaimBusy as exc:
    assert 'station' in str(exc)
else:
    raise AssertionError('a second station file created a second claim')
''', env=dict(harness.env, HFMODEM_STATION=str(paths[1]), AUDIO_IN='different'))
    try:
        completed(contender)
    finally:
        harness.release('first-station-held')
        completed(owner)
    evidence = finish(harness)
    assert sum(e['event'] == 'claim_acquired' for e in evidence.events) == 1


def test_alias_ports_one_claim(harness, fake_yaesu):
    observe(harness, [fake_yaesu.port])
    alias = harness.root / 'cat-alias'
    alias.symlink_to(fake_yaesu.port)
    shell_setup(harness)
    owner = held(harness, f'''
owned = claim.acquire('legacy', 'node', devices=claim.Devices(cat_port={fake_yaesu.port!r}))
''')
    contender = launch(harness, f'''
        try:
            claim.acquire('legacy', 'alias', devices=claim.Devices(cat_port={str(alias)!r}))
        except claim.ClaimBusy as exc:
            assert 'rig claimed:' in str(exc)
        else:
            raise AssertionError('alias created a second claim')
        assert claim.serial_nodes({str(alias)!r}) == claim.serial_nodes({fake_yaesu.port!r})
    ''')
    try:
        completed(contender)
    finally:
        harness.release('held')
        completed(owner)
    finish(harness)


@pytest.mark.parametrize('root_kind', ['console', 'tool', 'shell', 'modem'])
@pytest.mark.parametrize('child_kind', ['modem', 'one-shot', 'interactive-cat', 'tune-only-daemon'])
def test_external_down_against_live_session_classification(harness, root_kind, child_kind):
    observe(harness)
    setup = shell_setup(harness)
    harness.pin('live-session-held')
    inner = METADATA + '''
import sys
claim.inherited(int(sys.argv[1]))
coexist.checkpoint('live-session-held')
'''
    owner = launch(harness, f'''
        import subprocess, sys
        owned = claim.acquire({root_kind!r}, 'live-session')
        child = subprocess.Popen([sys.executable, '-c', {inner!r}, str(owned.fd)],
                                 pass_fds=(owned.fd,), start_new_session=True)
        assert child.wait(timeout=15) == 0
        claim.release(owned)
    ''')
    child_event = harness.reached('live-session-held', quiescent=True)
    owner_pid = next(e['pid'] for e in harness.events() if e['event'] == 'claim_acquired')
    child_command = {'modem': 'hfmodem.shrike.onair', 'one-shot': 'rigctl -P NONE f',
                     'interactive-cat': 'rigctl -P NONE', 'tune-only-daemon': 'rigctld -m 1 -P NONE'}
    inventory(harness, {owner_pid: root_kind, child_event['pid']: child_command[child_kind]})
    cleanup = down(harness, setup)
    try:
        out, _ = completed(cleanup, 7)
        assert f'pid {owner_pid}' in out and 'holds the claim; interrupt it and run down again' in out
        assert 'SIGTERM' not in out and 'SIGKILL' not in out
        assert 'PTT LINE' not in out and 'CLEANUP COMPLETE' not in out
        assert owner.poll() is None
        os.kill(child_event['pid'], 0)
    finally:
        harness.release('live-session-held')
        completed(owner)
    finish(harness)


@pytest.mark.parametrize('entry', ['preflight', 'arm'])
def test_unconfirmed_unkey_refuses_next_key_preflight(harness, entry):
    observe(harness)
    child = launch(harness, f'''
        from contextlib import nullcontext
        from types import SimpleNamespace
        from hfmodem.core.regulatory import Control, Unregulated
        from hfmodem.core.rig import Rig, RigError
        from hfmodem.station.preflight import Preflight
        from hfmodem.tests.core.fakerig import FakePtt
        class Cat:
            cache_free = True
            reads_ptt = False
            def open(self):
                pass
            def close(self):
                pass
            def identify(self):
                return 'coexist fake'
            def freq(self):
                return 7100000
            def ptt_state(self, **kwargs):
                return False
            def sole_ptt_owner(self, notes):
                return True
            def hold(self, *args):
                return nullcontext(True)
        class Line(FakePtt):
            unknown = False
            keys = 0
            def assert_(self, on):
                super().assert_(on)
                if on:
                    self.keys += 1
                elif self.keys:
                    self.unknown = True
            def sense(self):
                return None if self.unknown else super().sense()
        owned = claim.acquire('legacy', 'preflight-unconfirmed')
        line = Line()
        rig = Rig(model='generic', cat=Cat(), ptt=line,
                  profile=Unregulated(because='coexist fake devices'),
                  control=Control.LOCAL, transmit=True)
        check = Preflight.__new__(Preflight)
        check.cancelled = lambda: False
        check.cancel_event = SimpleNamespace(wait=lambda timeout: False)
        try:
            rig.arm(prove_ptt=False)
            assert rig.line_state == 'low'
            key = (lambda: check._key(rig)) if {entry!r} == 'preflight' else rig.arm
            key()
            assert line.keys == 1 and line.sense() is None
            try:
                key()
            except RigError:
                pass
            assert line.keys == 1, 'preflight commanded another key after unreadable unkey'
        finally:
            line.unknown = False
            rig.close()
            claim.release(owned)
    ''')
    try:
        completed(child)
    finally:
        finish(harness)


@pytest.mark.parametrize('overlap', [False, True], ids=['subsequent-key', 'pending-drive'])
def test_unconfirmed_unkey_refuses_next_key_regression(harness, overlap):
    observe(harness)
    child = launch(harness, f'''
        import json
        import threading
        from contextlib import nullcontext
        from hfmodem.core.regulatory import Control, Unregulated, centred
        from hfmodem.core.rig import Rig, RigError
        from hfmodem.tests.core.fakerig import FakePtt

        class Cat:
            cache_free = True
            def open(self):
                pass
            def close(self):
                pass
            def identify(self):
                return 'coexist fake'
            def freq(self):
                return 7100000
            def ptt_state(self, **kwargs):
                return False
            def hold(self, *args):
                return nullcontext(True)
            def clock(self):
                return 0
            def protect(self, *args):
                return object()
            def unprotect(self, protection):
                pass

        entered = threading.Event()
        resume = threading.Event()
        class Line(FakePtt):
            pause = False
            unknown = False
            def assert_(self, on):
                if on and self.pause:
                    entered.set()
                    assert resume.wait(5), 'drive was not released'
                super().assert_(on)
            def sense(self):
                return None if self.unknown else super().sense()

        owned = claim.acquire('legacy', 'unconfirmed-unkey')
        line = Line()
        rig = Rig(model='generic', cat=Cat(), ptt=line,
                  profile=Unregulated(because='coexist fake devices'),
                  control=Control.LOCAL, transmit=True)
        results = []
        failures = []
        def key():
            try:
                rig.key(centred(7100000, 500), why='coexist', duration_s=10)
                results.append('keyed')
            except RigError as exc:
                results.append(str(exc))
            except BaseException as exc:
                failures.append(repr(exc))
        try:
            rig.arm(prove_ptt=False)
            assert rig.line_state == 'low'
            if {overlap!r}:
                line.pause = True
                worker = threading.Thread(target=key)
                worker.start()
                assert entered.wait(5), 'key did not reach drive'
                line.unknown = True
                rig.unkey()
                assert rig.line_state == 'unconfirmed'
                resume.set()
                worker.join(5)
                assert not worker.is_alive()
            else:
                key()
                results.clear()
                line.unknown = True
                rig.unkey()
                assert rig.line_state == 'unconfirmed'
                key()
            print(json.dumps(dict(results=results, failures=failures,
                                  line=line.line, keyed=rig.keyed,
                                  watchdog=rig._timer is not None,
                                  drives=line.calls)), flush=True)
        finally:
            resume.set()
            line.unknown = False
            rig.close()
            claim.release(owned)
    ''')
    out, _ = completed(child)
    evidence = finish(harness)
    assert any(e['event'] == 'claim_acquired' for e in evidence.events)
    result = json.loads(out.splitlines()[-1])
    assert not result['failures'], result
    assert result['line'] is False, result
    assert result['keyed'] is False, result
    assert result['watchdog'] is False, result
    assert len(result['results']) == 1 and 'unconfirmed' in result['results'][0], result
