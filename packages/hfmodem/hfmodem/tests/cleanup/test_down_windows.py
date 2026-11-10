# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from concurrent.futures import ThreadPoolExecutor
import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import wave

import pytest

from hfmodem.core import claim, down, winclaim, winproc
from hfmodem.tests import hermetic
from hfmodem.tests.core.fakeyaesu import SerialFakeYaesu

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows cleanup protocol')

PRELUDE = '''
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from hfmodem.core import claim, down, posix, winclaim, winproc
'''
CHILD = '''
a = claim.received(launch=True)
a.opened()
ready = Path(sys.argv[1])
'''


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    pytest.fail('condition timed out')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch, isolated_claim_state, station_kept_out):
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(hermetic.GUARD), *filter(None, sys.path)]))
    monkeypatch.setenv('__PYVENV_LAUNCHER__', sys.executable)
    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setenv('HFMODEM_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('CAT_PORT', 'none')
    monkeypatch.setenv('PTT_PORT', 'COM41')
    for name in ('HFMODEM_COEXIST_LOG', 'HFMODEM_COEXIST_FAULT', 'HFMODEM_COEXIST_BARRIERS',
                 'HFMODEM_STATION', 'HFMODEM_RIG', 'RIG_CLAIMED'):
        monkeypatch.delenv(name, raising=False)
    claim.bootstrap()
    assert claim.state_dir() == tmp_path / 'state'
    return claim.state_dir()


@pytest.fixture
def com_pair():
    from hfmodem.tests.hermetic import allow_virtual_serial
    serial = pytest.importorskip('serial')
    for port in ('COM41', 'COM42'):
        try:
            target = winclaim.serial_target(port)
        except claim.ClaimError:
            pytest.skip(f'com0com {port} is absent')
        if not any(name in target.casefold() for name in ('cnc', 'com0com')):
            pytest.skip(f'{port} is not com0com')
    allow_virtual_serial('COM41', 'COM42')
    return serial


@pytest.fixture
def station_file(tmp_path):
    path = tmp_path / 'station.toml'
    path.write_text(f'''schema = 1
[station]
port = {free_port()}
mycall = "K1ABC"
control = "local"
regulatory = "unregulated"
because = "com0com cleanup test"
transmit = false
[rig]
model = "ft891"
cat = "none"
[rig.ptt]
port = "COM41"
''')
    return path


def end(entry):
    deadline = time.monotonic() + winproc.KILL_WAIT_S
    error = None
    if not entry.wait():
        try:
            entry.terminate()
        except winproc.ProcessAccessError as exc:
            error = exc
    assert entry.wait(max(0, deadline - time.monotonic())), str(error or f'pid {entry.pid} did not exit')


class Processes:
    def __init__(self, state):
        self.state = state
        self.entries, self.tracked = [], []

    def popen(self, *args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        self.tracked.append(process)
        handle = winclaim.duplicate(int(process._handle))
        try:
            process.entry = winproc._pin(handle, process.pid, winproc.END)
        except BaseException:
            winclaim.close_handle(handle)
            raise
        self.entries.append(process.entry)
        return process

    def collect(self, lineage):
        roots = {entry.identity for entry in self.entries}
        errors = []
        rows, _ = winproc.pinned()
        try:
            pinned = {entry.pid: entry for entry in rows}
            pinned.update((entry.pid, entry) for entry in self.entries)
            lineage.update(pinned.values())
            identities = set().union(*(lineage.descendants(root) for root in roots)) - roots
            assert claim.state_dir() == self.state
            for _, record in winproc.launch_records():
                child = record['child']
                identities.add((child['pid'], child['started']))
            for pid, started in identities - roots:
                try:
                    entry = winproc.open(pid, winproc.END)
                    if entry is not None:
                        if entry.started == started:
                            self.entries.append(entry)
                        else:
                            entry.close()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise ExceptionGroup('cannot pin test children', errors)
        finally:
            for entry in rows:
                entry.close()

    def close(self):
        errors, ended, lineage = [], set(), winproc.Lineage()
        try:
            while True:
                pending = [entry for entry in self.entries if entry.identity not in ended]
                for entry in reversed(pending):
                    ended.add(entry.identity)
                    try:
                        end(entry)
                    except Exception as exc:
                        errors.append(exc)
                try:
                    self.collect(lineage)
                except Exception as exc:
                    errors.append(exc)
                if all(entry.identity in ended for entry in self.entries):
                    break
            for process in self.tracked:
                try:
                    process.wait(timeout=5)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise ExceptionGroup('test process teardown failed', errors)
        finally:
            for entry in self.entries:
                entry.close()
            self.entries.clear()
            self.tracked.clear()


@pytest.fixture(autouse=True)
def processes(tmp_path, namespace):
    owned, streams = Processes(namespace), []
    def start(code, *args, env=None):
        log = (tmp_path / f'child-{len(streams)}.log').open('w+b')
        streams.append(log)
        return owned.popen([sys._base_executable, '-c', PRELUDE + code, *map(str, args)],
            env=dict(os.environ, **(env or {})), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    start.entries, start.popen = owned.entries, owned.popen
    yield start
    try:
        owned.close()
    finally:
        for stream in streams:
            stream.flush(); stream.seek(0)
            diagnostic = stream.read(65536).decode('utf-8', errors='replace')
            if diagnostic:
                print(diagnostic)
            stream.close()


@pytest.fixture
def survivor(processes, tmp_path):
    count = [0]
    def start(body=None, *, role='python', grace=2, fault=None, kill_launcher=True):
        count[0] += 1
        ready = tmp_path / f'ready-{count[0]}'
        result = tmp_path / f'launch-{count[0]}'
        body = body or '''
def stop(signum, frame):
    ready.with_suffix('.stopped').touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
ready.touch()
while True: time.sleep(.02)
'''
        script = (CHILD if role != 'recorder' else 'winproc.install_channels()\nready = Path(sys.argv[1])\n') + body
        launcher = processes('''
d = claim.acquire('launcher', 'test') if sys.argv[4] != 'recorder' else None
a = claim.inherited(d, launch=True) if d is not None else None
try:
    child = claim.spawn(a, [sys.executable, '-c', sys.argv[1], sys.argv[2]], role=sys.argv[4],
        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr)
finally:
    if a is not None: a.failed()
Path(sys.argv[3]).write_text(json.dumps({'pid': child.child.pid, 'started': child.child.started}))
time.sleep(180)
''', PRELUDE + script, ready, result, role,
            env={'HFMODEM_COEXIST_FAULT': fault} if fault else None)
        if fault:
            wait_for(lambda: launcher.poll() is not None)
            records, _ = winproc.live_launches()
            record, = [record for record in records if record['role'] == role]
            identity = record['child']
        else:
            wait_for(result.exists)
            identity = json.loads(result.read_text())
            wait_for(ready.exists)
        entry = winproc.open(identity['pid'], winproc.END)
        assert entry is not None and entry.started == identity['started']
        processes.entries.append(entry)
        if grace != 2:
            record = winproc.read_launch(entry)
            winproc.advance_launch(entry, record['phase'], grace=grace)
        if kill_launcher and launcher.poll() is None:
            end(launcher.entry)
            launcher.wait(timeout=5)
        return NS(entry=entry, launcher=launcher, ready=ready)
    return start


@pytest.fixture
def invoke(processes):
    def run(path, *, setup='', timeout=110, env=None):
        code = PRELUDE + setup + '\nfrom hfmodem import cli\nsys.exit(cli.main(["down", sys.argv[1]]))'
        process = processes.popen([sys._base_executable, '-c', code, str(path)],
            env=dict(os.environ, **(env or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            output, _ = process.communicate(timeout=timeout)
            return process.returncode, output
        finally:
            end(process.entry)
            process.wait(timeout=5)
            process.stdout.close()
    return run


def confirmed(result):
    code, text = result
    assert code == 0, text
    assert 'PTT LINE low' in text and 'CAT none' in text and 'CLEANUP COMPLETE' in text


@pytest.mark.wall_clock
@pytest.mark.parametrize('launcher_exits', [False, True])
def test_process_teardown_reaps_record_only_and_unreported_children(namespace, tmp_path, launcher_exits):
    owned, observers = Processes(namespace), []
    ready = tmp_path / 'children.json'
    script = '''
reader = subprocess.Popen([sys._base_executable, '-c', sys.argv[2], sys.argv[1] + '.reader'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
child = winproc.python_suspended(['-c', 'import time; time.sleep(120)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with winproc.open(child.pid) as entry:
    winproc.create_launch(entry, role='python')
while not Path(sys.argv[1] + '.reader').exists(): time.sleep(.01)
Path(sys.argv[1]).write_text(json.dumps([reader.pid, child.pid]))
if sys.argv[3] == 'True': sys.exit(0)
time.sleep(120)
'''
    reader = PRELUDE + '''
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
Path(sys.argv[1]).touch(); time.sleep(120)
'''
    try:
        launcher = owned.popen([sys._base_executable, '-c', PRELUDE + script, str(ready), reader, str(launcher_exits)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        wait_for(ready.exists)
        if launcher_exits:
            assert launcher.wait(timeout=5) == 0
        observers = [winproc.open(pid) for pid in [launcher.pid, *json.loads(ready.read_text())]]
        owned.close()
        assert all(entry.wait() for entry in observers)
    finally:
        owned.close()
        for entry in observers:
            entry.close()


@pytest.mark.wall_clock
def test_d_t1_nothing_running(com_pair, station_file, invoke):
    confirmed(invoke(station_file))
    evidence = json.loads((claim.state_dir() / 'cleanup.json').read_text())
    assert set(evidence) == {'pid', 'at', 'line', 'cat'}
    assert evidence['line'] == 'low' and evidence['cat'] == 'none'


@pytest.mark.wall_clock
def test_d_t1b_native_cat(com_pair, station_file, monkeypatch, invoke):
    monkeypatch.setenv('CAT_PORT', 'COM41')
    monkeypatch.delenv('PTT_PORT')
    station_file.write_text(station_file.read_text().replace('cat = "none"', 'cat = "yaesu"')
                            .replace('port = "COM41"', 'port = ""'))
    fake = SerialFakeYaesu('COM42')
    try:
        code, text = invoke(station_file)
        assert code == 0, text
        assert 'PTT LINE none' in text and 'CAT ok' in text and 'CLEANUP COMPLETE' in text
        assert 'FA;' in fake.log and not any(command.startswith('TX1') for command in fake.log)
        assert fake.error is None
    finally:
        fake.close()


@pytest.mark.wall_clock
@pytest.mark.parametrize('mode', ['handler', 'event', 'blocked-handler'])
def test_d_t2_survivor(com_pair, station_file, survivor, mode, invoke):
    body = None if mode == 'handler' else '''
def stop(signum, frame):
    ready.with_suffix('.stopped').touch()
    threading.Event().wait()
signal.signal(signal.SIGTERM, stop)
ready.touch()
''' + ('threading.Event().wait()' if mode == 'event' else 'while True: time.sleep(.02)')
    child = survivor(body)
    confirmed(invoke(station_file))
    assert child.entry.wait()
    if mode != 'event':
        assert child.ready.with_suffix('.stopped').exists()


@pytest.mark.wall_clock
def test_d_t3_deadman_clear_precedes_close(com_pair, station_file, survivor, invoke):
    from hfmodem.core import serialline
    child = survivor('''
from hfmodem.core import windeadman
from hfmodem.core.ptt import RtsPtt
line = RtsPtt('COM41'); line.open(); line.assert_(True)
original_exit = windeadman.os._exit
def held_exit(code):
    posix._windows_deadman._word.value = 0
    ready.with_suffix('.cleared').touch()
    while not ready.with_suffix('.release').exists(): time.sleep(.005)
    original_exit(code)
windeadman.os._exit = held_exit
signal.signal(signal.SIGTERM, lambda *args: threading.Event().wait())
ready.touch()
threading.Event().wait()
''')
    sense = serialline.open_port(com_pair.Serial, 'COM42', baudrate=9600, rts=False, dtr=False, write_timeout=.25)
    try:
        wait_for(lambda: sense.cts)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(invoke, station_file)
            try:
                wait_for(child.ready.with_suffix('.cleared').exists)
                assert not child.entry.wait()
                wait_for(lambda: not sense.cts)
                assert not child.entry.wait()
            finally:
                child.ready.with_suffix('.release').touch()
            confirmed(future.result(timeout=20))
    finally:
        child.ready.with_suffix('.release').touch()
        sense.close()


@pytest.mark.wall_clock
@pytest.mark.parametrize('point,phase', [('launch_after_push_claim', 'created'),
                                        ('launch_after_resuming', 'resuming')])
def test_d_t4_recovery(com_pair, station_file, survivor, point, phase, invoke):
    child = survivor(fault=point + ':kill')
    code, text = invoke(station_file)
    if phase == 'created':
        assert code == 0, text
        assert child.entry.wait() and not child.ready.exists()
        assert 'recovered never-run child' in text
        confirmed(invoke(station_file))
    else:
        assert code == 2 and 'launched and never acknowledged' in text
        assert f'Stop-Process -Id {child.entry.pid}' in text
        assert not child.entry.wait() and not child.ready.exists()


@pytest.mark.wall_clock
def test_d_t4_recovery_fresh_read(com_pair, station_file, processes, tmp_path, invoke):
    barriers = tmp_path / 'barriers'; barriers.mkdir()
    (barriers / 'launch_after_push_claim').touch()
    ready = tmp_path / 'ready'
    launcher = processes('''
d = claim.acquire('launcher', 'race'); a = claim.inherited(d, launch=True)
claim.spawn(a, [sys.executable, '-c', sys.argv[1], sys.argv[2]], role='python',
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
a.failed()
''', PRELUDE + 'Path(sys.argv[1]).touch(); time.sleep(120)', ready,
        env={'HFMODEM_COEXIST_BARRIERS': str(barriers)})
    wait_for((barriers / 'launch_after_push_claim.reached').exists)
    record, = winproc.live_launches()[0]
    child = winproc.open(record['child']['pid'], winproc.END); processes.entries.append(child)
    (barriers / 'down_recovery_read').touch()
    with ThreadPoolExecutor() as pool:
        future = pool.submit(invoke, station_file, env={'HFMODEM_COEXIST_BARRIERS': str(barriers)})
        try:
            wait_for((barriers / 'down_recovery_read.reached').exists)
            (barriers / 'launch_after_push_claim.release').touch()
            assert launcher.wait(timeout=10) == 0
            wait_for(ready.exists)
        finally:
            (barriers / 'down_recovery_read.release').touch()
            (barriers / 'launch_after_push_claim.release').touch()
        code, text = future.result(timeout=40)
    assert code == 2 and 'never acknowledged' in text
    assert not child.wait() and 'recovered never-run child' not in text


@pytest.mark.wall_clock
def test_d_t4_recorder_finalizes(com_pair, station_file, survivor, invoke):
    child = survivor('''
import wave
recording = ready.with_suffix('.wav')
stream = wave.open(str(recording), 'wb'); stream.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
stream.writeframesraw(b'\\0\\0' * 80)
def stop(*args):
    stream.close(); sys.exit(0)
signal.signal(signal.SIGTERM, stop)
ready.touch()
while True: time.sleep(.01)
''', role='recorder')
    confirmed(invoke(station_file))
    assert child.entry.wait()
    with wave.open(str(child.ready.with_suffix('.wav'))) as stream:
        assert stream.getnframes() == 80 and stream.readframes(80) == b'\0\0' * 80


@pytest.mark.wall_clock
@pytest.mark.parametrize('mimic', [False, True])
def test_d_t5_unknown_com_is_not_ended(com_pair, station_file, processes, tmp_path, mimic, invoke):
    ready = tmp_path / 'foreign'
    foreign = processes('''
import serial
port = serial.Serial(baudrate=9600, write_timeout=.25)
port.port, port.rts, port.dtr = 'COM41', False, False
port.open()
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready, 'hfmodem.shrike.onair' if mimic else 'foreign')
    wait_for(ready.exists)
    code, text = invoke(station_file)
    assert code == 1 and 'COM41 is open in another process' in text
    assert 'CLEANUP COMPLETE' not in text and foreign.poll() is None


@pytest.mark.wall_clock
@pytest.mark.parametrize('release_at', ['during-walk', 'waiting'])
def test_d_t5_plain_claim_reader_waited(com_pair, station_file, processes, tmp_path, invoke, release_at):
    ready, release, seen = tmp_path / 'reader', tmp_path / 'release', tmp_path / 'seen'
    foreign = processes('''
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
Path(sys.argv[1]).touch()
while not Path(sys.argv[2]).exists(): time.sleep(.01)
winclaim.close_handle(h)
''', ready, release)
    wait_for(ready.exists)
    setup = f'''
seen = Path({str(seen)!r})
reader_pid = {foreign.pid}
'''
    if release_at == 'during-walk':
        setup += '''
walk = winproc.device_holders
def observed(entries, devices, **kwargs):
    for reader in entries:
        if reader.pid == reader_pid:
            seen.touch()
            assert reader.wait(5)
    return walk(entries, devices, **kwargs)
winproc.device_holders = observed
'''
    else:
        setup += '''
inventory = down.Windows.inventory
def observed(self, deadline):
    result = inventory(self, deadline)
    if result.readers:
        seen.touch()
        assert self.entries[reader_pid].wait(5)
    return result
down.Windows.inventory = observed
'''
    with ThreadPoolExecutor() as pool:
        future = pool.submit(invoke, station_file, setup=setup)
        try:
            wait_for(lambda: seen.exists() or future.done())
            assert seen.exists(), future.result()
            assert not future.done() and foreign.poll() is None
        finally:
            release.touch()
        code, text = future.result(timeout=20)
    confirmed((code, text))
    assert f'unidentified claim holder pid {foreign.pid}' in text


@pytest.mark.wall_clock
def test_d_t5_reader_reopens_at_final_acquisition(com_pair, station_file, processes, tmp_path, invoke):
    ready = tmp_path / 'reader'
    reader = processes('''
ready = Path(sys.argv[1])
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
ready.touch()
while not ready.with_suffix('.close').exists(): time.sleep(.01)
winclaim.close_handle(h)
ready.with_suffix('.closed').touch()
while not ready.with_suffix('.reopen').exists(): time.sleep(.01)
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
ready.with_suffix('.reopened').touch()
while not ready.with_suffix('.release').exists(): time.sleep(.01)
winclaim.close_handle(h)
''', ready)
    wait_for(ready.exists)
    setup = f'''
ready = Path({str(ready)!r})
reader_pid = {reader.pid}
def reached(suffix):
    until = time.monotonic() + 10
    while not ready.with_suffix(suffix).exists():
        assert time.monotonic() < until, suffix
        time.sleep(.01)
inventory = down.Windows.inventory
def observed(self, deadline):
    result = inventory(self, deadline)
    if result.readers:
        assert reader_pid in self.entries
        if ready.with_suffix('.reopened').exists():
            ready.with_suffix('.reclassified').touch()
            ready.with_suffix('.release').touch()
            assert self.entries[reader_pid].wait(5)
        else:
            ready.with_suffix('.close').touch()
            reached('.closed')
    return result
down.Windows.inventory = observed
acquire = claim.acquire
attempts = 0
gate = None
def reopened(*args, **kwargs):
    global attempts, gate
    attempts += 1
    claim.verify_cleanup(kwargs['cleanup'])
    if attempts == 1:
        gate = kwargs['cleanup']
        ready.with_suffix('.reopen').touch()
        reached('.reopened')
        try:
            acquire(*args, **kwargs)
        except claim.ClaimBusy:
            ready.with_suffix('.busy').touch()
            raise
        raise AssertionError('reader did not block acquisition')
    assert attempts == 2 and kwargs['cleanup'] is gate
    assert ready.with_suffix('.reclassified').exists()
    return acquire(*args, **kwargs)
claim.acquire = reopened
'''
    confirmed(invoke(station_file, setup=setup))
    assert ready.with_suffix('.busy').exists() and ready.with_suffix('.reclassified').exists()
    assert reader.wait(timeout=5) == 0


@pytest.mark.wall_clock
def test_d_t6_gate_refuses_target_launch(com_pair, station_file, survivor, invoke):
    child = survivor('''
def stop(*args):
    try:
        a = claim.inherited(a_claim, launch=True)
    except claim.ClaimBusy as exc:
        ready.with_suffix('.refused').write_text(str(exc)); sys.exit(0)
    raise AssertionError('admission allowed during cleanup')
a_claim = a.claim
signal.signal(signal.SIGTERM, stop)
ready.touch()
while True: time.sleep(.01)
''')
    confirmed(invoke(station_file))
    assert 'cleanup in progress' in child.ready.with_suffix('.refused').read_text()


@pytest.mark.wall_clock
def test_d_t6_late_foreign_holder_is_refused(com_pair, station_file, processes, tmp_path, invoke):
    barriers = tmp_path / 'barriers'; barriers.mkdir(); (barriers / 'down_issued').touch()
    ready = tmp_path / 'foreign'
    with ThreadPoolExecutor() as pool:
        future = pool.submit(invoke, station_file, env={'HFMODEM_COEXIST_BARRIERS': str(barriers)})
        try:
            wait_for((barriers / 'down_issued.reached').exists)
            process = processes('''
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready)
            wait_for(ready.exists)
        finally:
            (barriers / 'down_issued.release').touch()
        code, text = future.result(timeout=20)
    assert code == 1 and 'started after this down was issued' in text
    assert process.poll() is None


@pytest.mark.wall_clock
@pytest.mark.parametrize('station_exits', [False, True])
def test_d_t7_station_release_preserves_subtree(com_pair, station_file, processes, tmp_path, invoke, station_exits):
    ready = tmp_path / 'station-ready'
    station = processes('''
from hfmodem.core import config
from hfmodem.station.process import Station
from hfmodem.station.service import Service
from hfmodem.tests.station.test_process import FakeRig, FakeAudio
cfg = config.load(sys.argv[1]); st = Station(cfg, station_file=sys.argv[1])
st.preflight_reference = lambda: 1
st.open_audio = lambda: setattr(st, 'audio', FakeAudio([]))
st.open_rig = lambda **kw: setattr(st, 'rig', FakeRig([], receive=True))
st.build_arbiter = lambda: None
st.build_lanes = lambda: None
st._tick = lambda: st._poll_stop.wait()
service = Service(sys.argv[1], station=st); st.service = service; service.bind()
st.own(receive=True)
child = subprocess.Popen([sys._base_executable, '-c', 'import time; time.sleep(120)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with winproc.open(child.pid) as entry:
    winproc.create_launch(entry, role='recorder'); winproc.advance_launch(entry, 'acknowledged')
Path(sys.argv[2]).write_text(str(child.pid))
st.run(detach=True)
''', station_file, ready)
    wait_for(ready.exists)
    child = winproc.open(int(ready.read_text()), winproc.END); processes.entries.append(child)
    if station_exits:
        seen, gone = tmp_path / 'first-round', tmp_path / 'station-gone'
        setup = f'''
inventory = down.Windows.inventory
calls = 0
def observed(self, deadline):
    global calls
    result = inventory(self, deadline)
    calls += 1
    if calls == 1:
        Path({str(seen)!r}).touch()
        until = time.monotonic() + 10
        while not Path({str(gone)!r}).exists():
            assert time.monotonic() < until, 'station did not exit'
            time.sleep(.01)
        result.releasing = True
    return result
down.Windows.inventory = observed
request = winproc.request_stop
def requested(entry):
    assert entry.pid != {child.pid}, 'station child received stop request'
    return request(entry)
winproc.request_stop = requested
terminate = winproc.Pinned.terminate
def terminated(entry):
    assert entry.pid != {child.pid}, 'station child received termination'
    return terminate(entry)
winproc.Pinned.terminate = terminated
'''
        with ThreadPoolExecutor() as pool:
            future = pool.submit(invoke, station_file, setup=setup)
            try:
                wait_for(lambda: seen.exists() or future.done())
                assert seen.exists(), future.result()
                end(station.entry)
                assert station.wait(timeout=5) is not None
            finally:
                gone.touch()
            confirmed(future.result(timeout=20))
    else:
        confirmed(invoke(station_file))
        from hfmodem.station.client import Client
        assert Client(station_file).state()['service']['state'] == 'observing'
        assert station.poll() is None
    assert not child.wait()


@pytest.mark.wall_clock
def test_d_t8_inherited_production_handoff(com_pair, station_file, processes, tmp_path):
    result = tmp_path / 'inherited-result'
    console = processes('''
d = claim.acquire('console', 'lease')
a = claim.inherited(d, launch=True)
child = claim.spawn(a, [sys.executable, '-c', sys.argv[1]], role='python',
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
a.failed()
while winproc.read_launch(child.child)['phase'] != 'acknowledged': time.sleep(.01)
script = """
from hfmodem.core import claim, down, winclaim
from hfmodem.station import client
original = winclaim.open_role
def checked(path, role):
    assert role != 'holder', 'inherited cleanup opened holder role'
    return original(path, role)
winclaim.open_role = checked
import sys
sys.exit(down.run(sys.argv[1], client))
"""
cleanup = claim.spawn(d, [sys.executable, '-c', script, sys.argv[2]], role='cleanup',
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = cleanup.process.communicate(timeout=90)
assert cleanup.process.returncode == 0, (out, err)
assert b'CLEANUP COMPLETE' in out and child.child.wait(5)
assert claim.inherited(d) == d
Path(sys.argv[3]).write_text(out.decode())
while True: time.sleep(.02)
''', PRELUDE + '''
a = claim.received(launch=True); a.opened()
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
while True: time.sleep(.01)
''', station_file, result)
    wait_for(result.exists, timeout=90)
    assert console.poll() is None
    assert 'CLEANUP COMPLETE' in result.read_text()
    with pytest.raises(claim.ClaimBusy):
        claim.acquire('test', 'still-held')


@pytest.mark.wall_clock
def test_d_t9_same_shell_keeps_ancestor(com_pair, station_file, survivor, invoke):
    child = survivor()
    # pytest is the common shell ancestor of the dead launcher and this down.
    confirmed(invoke(station_file))
    assert child.entry.wait()
    with winproc.open(os.getpid()) as shell:
        assert not shell.wait()


@pytest.mark.wall_clock
def test_d_t10_admission_reader_cleanup_distinction(com_pair, station_file, processes, tmp_path, invoke):
    ready = tmp_path / 'gate-ready'
    process = processes('''
fd = claim._shared()
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready)
    wait_for(ready.exists)
    reader_ready = tmp_path / 'gate-reader-ready'
    reader = processes('''
h = winclaim.open_handle(claim.mutex_path(), 1 | 0x80, 1)
Path(sys.argv[1]).touch(); time.sleep(120)
''', reader_ready)
    wait_for(reader_ready.exists)
    code, text = invoke(station_file)
    assert code == 2 and 'admission in progress' in text and str(process.pid) in text
    assert 'python' in text.casefold() and process.poll() is None
    assert str(reader.pid) in text and reader.poll() is None
    end(process.entry); process.wait(timeout=5)
    end(reader.entry); reader.wait(timeout=5)
    ready.unlink()
    cleanup = processes('''
gate = claim.mutex(timeout=0)
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready)
    wait_for(ready.exists)
    code, text = invoke(station_file)
    assert code == 2 and f'cleanup in progress (pid {cleanup.pid}' in text
    assert cleanup.poll() is None


@pytest.mark.wall_clock
def test_d_t10_live_publisher_only_standalone(com_pair, station_file, processes, tmp_path, invoke):
    ready = tmp_path / 'publisher'
    publisher = processes("d = claim.acquire('console', 'lease'); Path(sys.argv[1]).touch(); time.sleep(120)", ready)
    wait_for(ready.exists)
    code, text = invoke(station_file)
    assert code == 7 and f'pid {publisher.pid}' in text and 'interrupt it and run down again' in text
    assert publisher.poll() is None


@pytest.mark.wall_clock
@pytest.mark.parametrize('gap', ['using', 'holder', 'record', 'negative'])
def test_d_t11_relevant_inventory_gaps_refuse(com_pair, station_file, processes, tmp_path, gap, invoke):
    ready = tmp_path / 'holder'
    holder = processes('''
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready)
    wait_for(ready.exists)
    if gap == 'record':
        with winproc.open(holder.pid) as entry:
            winproc.create_launch(entry, role='recorder'); winproc.advance_launch(entry, 'acknowledged')
    setup = f'holder_pid = {holder.pid}\n'
    if gap == 'using':
        setup += "winproc.using = lambda *args: (_ for _ in ()).throw(winproc.InventoryUnavailable('FileProcessIdsUsingFileInformation', 0xc0000022))"
    elif gap in ('holder', 'record'):
        setup += '''
opened = winproc.open
def denied(pid, *args, **kwargs):
    if pid == holder_pid: raise winproc.ProcessAccessError(pid, winproc.IDENTIFY)
    return opened(pid, *args, **kwargs)
winproc.open = denied
'''
    else:
        setup += '''
walk = winproc.device_holders
def incomplete(entries, devices, **kwargs):
    result = walk(entries, devices, **kwargs)
    from dataclasses import replace
    walks = dict(result.entries)
    walks[holder_pid] = winproc.DeviceWalk(message='cannot inspect pid ' + str(holder_pid))
    return replace(result, entries=walks)
winproc.device_holders = incomplete
'''
    code, text = invoke(station_file, setup=setup)
    assert code == 1 and 'NOT CONFIRMED:' in text and 'CLEANUP COMPLETE' not in text
    assert ('FileProcessIdsUsingFileInformation' in text) if gap == 'using' else str(holder.pid) in text
    assert holder.poll() is None


@pytest.mark.wall_clock
@pytest.mark.parametrize('case', ['child', 'parent'])
def test_d_t11_recycled_snapshot_id_never_authorizes(com_pair, station_file, processes, survivor, tmp_path, case, invoke):
    body = None
    if case == 'parent':
        body = '''
code = """
import serial
s = serial.Serial(baudrate=9600, write_timeout=.25)
s.port, s.rts, s.dtr = 'COM41', False, False
s.open()
Path(sys.argv[1]).touch(); time.sleep(120)
"""
p = winproc.python_suspended(['-c', sys.argv[2] + code, str(ready.with_suffix('.child'))],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with winproc.open(p.pid, winproc.END | winproc.INSPECT) as entry:
    winproc.push_handle(entry, winclaim.handle(a.claim.fd)); winproc.resume(entry)
while not ready.with_suffix('.child').exists(): time.sleep(.01)
ready.write_text(str(p.pid))
while True: time.sleep(.01)
'''.replace('sys.argv[2] + code', repr(PRELUDE) + ' + code')
    original = survivor(body)
    descendant = None
    if case == 'parent':
        descendant = winproc.open(int(original.ready.read_text()), winproc.END)
        processes.entries.append(descendant)
    ready = tmp_path / 'replacement'
    replacement = processes('Path(sys.argv[1]).touch(); time.sleep(120)', ready)
    wait_for(ready.exists)
    setup = f'''
from dataclasses import replace
census, opened = winproc._census, winproc.open
swapped = False
def recycle_before_pinning():
    global swapped
    pids = census()
    if not swapped:
        assert {original.entry.pid} in pids
        with opened({original.entry.pid}, winproc.END) as old:
            old.terminate(); assert old.wait(5)
        swapped = True
    return pids
def open_recycled(pid, rights=winproc.IDENTIFY):
    if swapped and pid == {original.entry.pid}:
        return replace(opened({replacement.pid}, rights), pid=pid)
    return opened(pid, rights)
winproc._census, winproc.open = recycle_before_pinning, open_recycled
'''
    if descendant is not None:
        setup += f'''
update = winproc.Lineage.update
def checked_lineage(self, entries):
    entries = list(entries)
    result = update(self, entries)
    rows = {{entry.pid: entry for entry in entries}}
    assert rows[{original.entry.pid}].identity not in self.ancestors(rows[{descendant.pid}])
    return result
winproc.Lineage.update = checked_lineage
'''
    code, text = invoke(station_file, setup=setup)
    if case == 'child':
        confirmed((code, text))
    else:
        assert code == 1 and f'COM41 is open in pid {descendant.pid}' in text
        assert not descendant.wait()
    assert original.entry.wait()
    assert replacement.poll() is None


@pytest.mark.wall_clock
@pytest.mark.parametrize('test', ['test_d_t2_survivor[handler]', 'test_d_t2_survivor[event]',
                                 'test_d_t2_survivor[blocked-handler]',
                                 'test_d_t5_unknown_com_is_not_ended[False]',
                                 'test_d_t5_unknown_com_is_not_ended[True]',
                                 'test_d_t5_plain_claim_reader_waited',
                                 'test_d_t10_admission_reader_cleanup_distinction'])
def test_d_t12_filtered_token(request, tmp_path, test):
    from hfmodem.tests.core.win_token import filtered_test
    target = NS(config=request.config,
                node=NS(path=Path(__file__), nodeid=str(Path(__file__)) + '::' + test))
    filtered_test(target, tmp_path / 'filtered')


@pytest.mark.wall_clock
def test_d_t13_full_fifty_second_grace(com_pair, station_file, survivor, invoke):
    child = survivor('''
winproc.declare_grace(50)
def stop(*args):
    ready.with_suffix('.stopping').write_text(str(time.monotonic()))
    until = time.monotonic() + 48.5
    while time.monotonic() < until: time.sleep(.01)
    ready.with_suffix('.finished').touch(); sys.exit(0)
signal.signal(signal.SIGTERM, stop)
ready.touch()
while True: time.sleep(.01)
''', grace=50)
    start = time.monotonic()
    confirmed(invoke(station_file))
    assert child.ready.with_suffix('.finished').exists()
    assert 48 <= time.monotonic() - start < 65
    assert child.entry.wait()


@pytest.mark.wall_clock
def test_d_t13_later_round_concurrent(com_pair, station_file, survivor, invoke):
    first = survivor('''
winproc.declare_grace(8)
def stopping(*args):
    ready.with_suffix('.stopping').write_text(str(time.monotonic()))
    threading.Event().wait()
signal.signal(signal.SIGTERM, stopping)
ready.touch()
while True: time.sleep(.01)
''', grace=8)
    second = survivor('''
winproc.declare_grace(8)
def stopping(*args):
    ready.with_suffix('.stopping').write_text(str(time.monotonic()))
    threading.Event().wait()
signal.signal(signal.SIGTERM, stopping)
ready.touch()
while True: time.sleep(.01)
''', role='recorder', grace=8)
    setup = f'''
original = down.Windows.inventory
calls = 0
def delayed(self, deadline):
    global calls
    calls += 1
    offer = self.offer
    if calls == 1:
        self.offer = lambda entry: offer(entry) if entry.pid != {second.entry.pid} else None
    try:
        result = original(self, deadline)
    finally:
        self.offer = offer
    if calls == 1:
        result.targets = [entry for entry in result.targets if entry.pid != {second.entry.pid}]
    return result
down.Windows.inventory = delayed
'''
    start = time.monotonic()
    confirmed(invoke(station_file, setup=setup))
    assert first.entry.wait() and second.entry.wait()
    first_at = float(first.ready.with_suffix('.stopping').read_text())
    second_at = float(second.ready.with_suffix('.stopping').read_text())
    # The stop timestamps prove concurrency; the elapsed bound limits completion.
    assert 0 <= second_at - first_at < 4
    assert time.monotonic() - start < 20


@pytest.mark.wall_clock
def test_d_t13_unresolved_is_final(com_pair, station_file, survivor, tmp_path, invoke):
    child = survivor()
    report = tmp_path / 'requests.json'
    setup = f'''
requests = []; terms = []
request = winproc.request_stop
def requested(entry):
    if entry.pid == {child.entry.pid}:
        requests.append(entry.pid)
        return 2
    return request(entry)
winproc.request_stop = requested
original_stop = down.Windows.stop
def stop(self, entry):
    stopper = original_stop(self, entry)
    if entry.pid == {child.entry.pid}:
        stopper.entry.wait = lambda timeout=0: False
        def terminate(): terms.append(entry.pid)
        stopper.entry.terminate = terminate
    return stopper
down.Windows.stop = stop
import atexit
atexit.register(lambda: Path({str(report)!r}).write_text(json.dumps([requests, terms])))
'''
    code, text = invoke(station_file, setup=setup)
    assert code == 1 and 'termination unresolved' in text and 'CLEANUP COMPLETE' not in text
    assert json.loads(report.read_text()) == [[child.entry.pid], [child.entry.pid]]
    assert text.count('-- stop requested') == 1 and text.count('-- terminated') == 1
    assert not child.entry.wait()


@pytest.mark.wall_clock
def test_n10_down_nt_ratchet(com_pair, station_file, survivor, invoke):
    child = survivor()
    setup = '''
os.kill = lambda *args: (_ for _ in ()).throw(AssertionError('os.kill reached on Windows'))
if hasattr(signal, 'SIGKILL'): del signal.SIGKILL
'''
    confirmed(invoke(station_file, setup=setup))
    assert child.entry.wait()
    source = ast.parse(Path(down.__file__).read_text())
    for node in ast.walk(source):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in {('os', 'kill'), ('signal', 'SIGKILL')}


@pytest.mark.wall_clock
def test_d_t4_live_launcher_created_child_untouched(com_pair, station_file, processes, tmp_path, invoke):
    barriers = tmp_path / 'barriers'; barriers.mkdir(); (barriers / 'launch_after_push_claim').touch()
    launcher = processes('''
d = claim.acquire('launcher', 'still-creating'); a = claim.inherited(d, launch=True)
claim.spawn(a, [sys.executable, '-c', 'import time; time.sleep(120)'], role='python')
''', env={'HFMODEM_COEXIST_BARRIERS': str(barriers)})
    wait_for((barriers / 'launch_after_push_claim.reached').exists)
    record, = winproc.live_launches()[0]
    entry = winproc.open(record['child']['pid'], winproc.END); processes.entries.append(entry)
    try:
        # Shorten only the admission wait; recovery still goes through the real flow.
        code, text = invoke(station_file, setup='''
mutex = claim.mutex
claim.mutex = lambda **kwargs: mutex(**dict(kwargs, timeout=.2))
''')
        assert code == 2 and 'admission in progress' in text
        assert not entry.wait() and launcher.poll() is None
        assert winproc.read_launch(entry)['phase'] == 'created'
    finally:
        (barriers / 'launch_after_push_claim.release').touch()


@pytest.mark.wall_clock
def test_d_t5_known_candidate_com_attribution(com_pair, station_file, processes, tmp_path, invoke):
    ready = tmp_path / 'known-foreign'
    foreign = processes('''
import serial
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
port = serial.Serial(baudrate=9600, write_timeout=.25)
port.port, port.rts, port.dtr = 'COM41', False, False
port.open()
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready)
    wait_for(ready.exists)
    code, text = invoke(station_file)
    assert code == 1 and f'COM41 is open in pid {foreign.pid} (python.exe)' in text
    assert foreign.poll() is None


@pytest.mark.wall_clock
def test_d_t5_reader_settle_expiry_names_holder(com_pair, station_file, processes, tmp_path, invoke):
    ready = tmp_path / 'reader'
    reader = processes('''
h = winclaim.open_handle(claim.claim_path(), 1 | 0x80, 1)
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready)
    wait_for(ready.exists)
    started = time.monotonic()
    code, text = invoke(station_file)
    assert code == 1 and f'unidentified claim holder pid {reader.pid}' in text
    assert 'did not settle within 45 s' in text and reader.poll() is None
    assert '-- waiting for plain reader' in text
    assert time.monotonic() - started >= down.SETTLE_S


@pytest.mark.wall_clock
def test_d_t5_descendant_port_holder_after_root_exit(com_pair, station_file, survivor, processes, invoke):
    child = survivor('''
script = """
import serial
from hfmodem.core import serialline, winproc
s = serialline.open_port(serial.Serial, 'COM41', baudrate=9600, rts=False, dtr=False, write_timeout=.25)
winproc.install_channels()
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
Path(sys.argv[1]).touch()
while True: time.sleep(.01)
"""
p = subprocess.Popen([sys._base_executable, '-c', sys.argv[2] + script, str(ready.with_suffix('.descendant'))],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not ready.with_suffix('.descendant').exists(): time.sleep(.01)
ready.write_text(str(p.pid))
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
while True: time.sleep(.01)
'''.replace('sys.argv[2] + script', repr(PRELUDE) + ' + script'))
    entry = winproc.open(int(child.ready.read_text()), winproc.END); processes.entries.append(entry)
    confirmed(invoke(station_file))
    assert child.entry.wait() and entry.wait()


@pytest.mark.wall_clock
@pytest.mark.parametrize('phase', ['created', 'resuming', 'unresolved'])
def test_d1_abandoned_inspector_recovery(com_pair, station_file, processes, tmp_path, phase, invoke):
    ready = tmp_path / 'inspector-launcher'
    launcher = processes('''
p = winproc.python_suspended(['-c', 'import time; time.sleep(120)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with winproc.open(p.pid, winproc.END) as entry:
    winproc.create_launch(entry, role='inspector')
    if sys.argv[2] != 'created': winproc.advance_launch(entry, sys.argv[2])
    Path(sys.argv[1]).write_text(str(p.pid))
''', ready, phase)
    wait_for(ready.exists)
    assert launcher.wait(timeout=5) == 0
    entry = winproc.open(int(ready.read_text()), winproc.END); processes.entries.append(entry)
    code, text = invoke(station_file)
    assert code == 0, text
    assert entry.wait(5)


@pytest.mark.wall_clock
def test_d1_live_unresolved_inspector_refuses(com_pair, station_file, processes, tmp_path, invoke):
    ready = tmp_path / 'unresolved'
    inspector = processes('''
with winproc.open(os.getpid()) as entry:
    winproc.create_launch(entry, role='inspector'); winproc.advance_launch(entry, 'unresolved')
Path(sys.argv[1]).touch(); time.sleep(120)
''', ready)
    wait_for(ready.exists)
    code, text = invoke(station_file)
    assert code == 1 and f'inspection unavailable: pid {inspector.pid}' in text
    assert inspector.poll() is None


@pytest.mark.wall_clock
def test_d1_created_recovery_pins_launcher_before_child(com_pair, station_file, survivor, invoke):
    child = survivor(fault='launch_after_push_claim:kill')
    setup = f'''
opened = winproc.open
launcher_checked = False
def ordered(pid, rights=winproc.IDENTIFY):
    global launcher_checked
    if pid == {child.entry.pid}:
        assert launcher_checked, 'child pinned before launcher exit was confirmed'
    entry = opened(pid, rights)
    if pid == {child.launcher.pid}:
        assert entry is None or entry.wait()
        launcher_checked = True
    return entry
winproc.open = ordered
'''
    confirmed(invoke(station_file, setup=setup))
    assert child.entry.wait()


@pytest.mark.wall_clock
@pytest.mark.parametrize('attempts', [1, 8], ids=['once', 'stress'])
def test_d1_created_recovery_concurrent_is_idempotent(com_pair, station_file, survivor, invoke, attempts):
    with ThreadPoolExecutor(max_workers=2) as pool:
        for attempt in range(attempts):
            child = survivor(fault='launch_after_push_claim:kill')
            results = list(pool.map(invoke, [station_file, station_file]))
            assert any(code == 0 for code, _ in results), (attempt, results)
            assert all(code in (0, 2) for code, _ in results), (attempt, results)
            assert child.entry.wait() and not child.ready.exists()
            confirmed(invoke(station_file))


@pytest.mark.wall_clock
def test_d_t13_no_channel_reports_then_terminates_once(com_pair, station_file, processes, tmp_path, invoke):
    ready = tmp_path / 'no-channel'
    process = processes("Path(sys.argv[1]).touch(); time.sleep(120)", ready)
    wait_for(ready.exists)
    with winproc.open(process.pid) as entry:
        winproc.create_launch(entry, role='one-shot')
        winproc.advance_launch(entry, 'acknowledged')
    code, text = invoke(station_file)
    confirmed((code, text))
    assert f'pid {process.pid} (python.exe) -- no stop channel' in text
    assert text.count(f'pid {process.pid} -- terminated') == 1
    assert process.poll() is not None
