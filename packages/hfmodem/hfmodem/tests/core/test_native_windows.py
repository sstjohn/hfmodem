# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hfmodem.core import claim, native, winclaim, winproc
from .fakeyaesu import SerialFakeYaesu

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows native launch protocol")

# CD:W13 is one atomic test: a skip of any prerequisite cannot qualify a subset.
WINDOWS_NATIVE_GATE = "test_n6_cd_w13_real_executables"
N6_CASES = {"rigctl", "rigctld", "one-shot", "missing-device", "ffmpeg-file", "ffmpeg-pipe",
            "rigctld-before", "rigctld-after", "ffmpeg-before", "ffmpeg-after"}
PRELUDE = """
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from hfmodem.core import claim, native, winclaim, winproc
"""


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, sys.path)))
    monkeypatch.setenv("__PYVENV_LAUNCHER__", sys.executable)
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_STATE_DIR", str(tmp_path / "state"))
    for key in ("HFMODEM_COEXIST_LOG", "HFMODEM_COEXIST_FAULT", "HFMODEM_COEXIST_BARRIERS",
                "RIG_CLAIMED"):
        monkeypatch.delenv(key, raising=False)


def command(code, *args):
    return [sys.executable, "-c", PRELUDE + code, *map(str, args)]


def audio_command(code, *args, output="pipe:1"):
    return command(code, *args, "-f", "dshow", "-i", "test", output)


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    pytest.fail("condition timed out")


def records(role=None):
    rows = [winproc._read_json(path) for path in
            (claim.state_dir() / "rig.claim.launch.d").glob("*.json")]
    return [row for row in rows if row is not None and (role is None or row["role"] == role)]


def native_record():
    return next(iter(records("native")), None)


def cleanup_available(*, timeout=0):
    mutex = claim.mutex(timeout=timeout)
    claim.mutex_release(mutex)


def claim_free():
    handle = winclaim._open(claim.claim_path(), "holder")
    winclaim.close_handle(handle)


def claim_held():
    with pytest.raises(OSError) as failure:
        winclaim._open(claim.claim_path(), "holder")
    assert failure.value.winerror == 32


def raw_gate_free():
    handle = winclaim._open(claim.mutex_path(), "cleanup")
    winclaim.close_handle(handle)


@contextmanager
def acquired():
    descriptor = claim.acquire("test", "native-windows")
    original = winclaim.handle(descriptor.fd)
    try:
        yield descriptor
    finally:
        entry = winclaim.entries().get(descriptor.fd)
        if entry is not None and entry.handle == original:
            claim.close(descriptor)


@pytest.fixture
def owned():
    with acquired() as descriptor:
        yield descriptor


@pytest.fixture
def children():
    tracked = []
    yield tracked
    for child in reversed(tracked):
        try:
            child.kill()
            assert child.wait_gone(5)
            if child._process is not None:
                child._process.wait(timeout=5)
        finally:
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


@pytest.fixture
def launchers():
    tracked = []
    def start(code, *args, env=None):
        process = subprocess.Popen([sys._base_executable, *command(code, *args)[1:]],
            env=dict(os.environ, **(env or {})), close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tracked.append(process)
        return process
    yield start
    for process in reversed(tracked):
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stderr.close()


def recorded_child(record, children):
    identity = record["child"]
    child = native.NativeChild(identity["pid"], identity["started"], None, None, None)
    children.append(child)
    return child


def supervisor_setup(monkeypatch, setup):
    spawn = claim.spawn
    def configured(descriptor, argv, *, role, **options):
        if role == "supervisor":
            argv = command(setup + "\nsys.exit(native._supervise(json.loads(sys.argv[1])))", argv[-1])
        return spawn(descriptor, argv, role=role, **options)
    monkeypatch.setattr(claim, "spawn", configured)


@pytest.fixture
def com_pair():
    from hfmodem.tests.hermetic import allow_virtual_serial
    serial = pytest.importorskip("serial")
    for port in ("COM41", "COM42"):
        try:
            target = winclaim.serial_target(port)
        except claim.ClaimError:
            pytest.skip(f"com0com {port} is absent")
        if not any(name in target.casefold() for name in ("cnc", "com0com")):
            pytest.skip(f"{port} is not a com0com device")
    allow_virtual_serial('COM41', 'COM42')
    return serial


@pytest.mark.parametrize("failure", ["record", "push", "resuming", "resume"])
def test_n1_suspension_gates(owned, children, tmp_path, monkeypatch, failure):
    marker = tmp_path / "ran"
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise claim.ClaimError("injected setup failure")
        if failure == "resuming":
            advance = winproc.advance_launch
            def advanced(entry, phase, **fields):
                if phase == "resuming":
                    fail()
                return advance(entry, phase, **fields)
            patch.setattr(winproc, "advance_launch", advanced)
        else:
            patch.setattr(winproc, {"record": "create_launch", "push": "push_handle",
                                   "resume": "resume"}[failure], fail)
        with pytest.raises(native.NativeSpawnError, match="injected setup failure") as exc:
            native.spawn_native(command("Path(sys.argv[1]).touch()", marker), claim_fd=owned)
        child = exc.value.child
        children.append(child)
        assert not child.alive()
        assert child._process.wait(timeout=0) != 0
    assert not marker.exists()
    cleanup_available()
    claim.close(owned)
    claim_free()


@pytest.mark.wall_clock
def test_one_shot_holds_admission_and_wait_records_exit(owned, children):
    child = native.spawn_native(command("""
record = winproc.read_launch()
assert record['role'] == 'one-shot' and record['phase'] == 'resuming'
assert not os.get_handle_inheritable(record['handles']['claim'])
print('ready', flush=True)
sys.stdin.buffer.read(1)
"""), claim_fd=owned)
    children.append(child)
    assert child.stdout.readline().rstrip(b"\r\n") == b"ready"
    assert winproc.read_launch(child._win_entry)["phase"] == "resuming"
    with pytest.raises(claim.ClaimBusy, match=f"admission in progress.*{child.pid}"):
        claim.mutex(timeout=0)
    out, err = child._process.communicate(b"x", timeout=5)
    assert (out, err) == (b"", b"")
    assert winproc.read_launch(child._win_entry)["phase"] == "exited"
    cleanup_available()


@pytest.mark.wall_clock
def test_n2_no_descendant_claim_or_admission(owned, children, tmp_path):
    marker = tmp_path / "grandchild"
    grandchild_code = "Path(sys.argv[1]).touch(); time.sleep(60)"
    child = native.spawn_native(command("""
record = winproc.read_launch()
assert not os.get_handle_inheritable(record['handles']['claim'])
p = subprocess.Popen([sys._base_executable, '-c', sys.argv[1], sys.argv[2]], close_fds=False,
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid, flush=True)
time.sleep(60)
""", PRELUDE + grandchild_code, marker), claim_fd=owned)
    children.append(child)
    grandchild = winproc.open(int(child.stdout.readline()), winproc.END)
    try:
        wait_for(marker.exists)
        claim.close(owned)
        claim_held()
        child.kill()
        assert child.wait_gone(5)
        assert not grandchild.wait()
        claim_free()
        cleanup_available()
    finally:
        grandchild.terminate()
        assert grandchild.wait(5)
        grandchild.close()


BOUNDARIES = [
    ("native_after_creation", None, False),
    ("native_after_record", "created", False),
    ("native_after_push_claim", "created", True),
    ("native_after_resuming", "resuming", True),
    ("native_after_resume", "resuming", True),
    ("native_after_acknowledgement", "acknowledged", True),
    ("native_after_reply", "acknowledged", True),
]

BOUNDARY_AUDIO = """
try:
    os.write(1,b'audio')
except OSError:
    pass
Path(sys.argv[1]).touch()
time.sleep(60)
"""


@pytest.mark.wall_clock
def test_boundary_audio_survives_closed_stdout(tmp_path):
    marker = tmp_path / "ready"
    reader, writer = os.pipe()
    os.close(reader)
    try:
        process = subprocess.Popen(audio_command(BOUNDARY_AUDIO, marker),
            stdout=writer, stderr=subprocess.PIPE)
    finally:
        os.close(writer)
    try:
        wait_for(marker.exists)
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)
        process.stderr.close()


def killed_boundary(launchers, children, tmp_path, point, phase, pushed, *, long_lived):
    marker, log, result = [tmp_path / name for name in ("ran", "events", "result")]
    argv = audio_command(BOUNDARY_AUDIO, marker)
    if not long_lived:
        argv = command("Path(sys.argv[1]).touch(); time.sleep(60)", marker)
    caller = launchers("""
d = claim.acquire('caller', 'boundary')
try:
    try:
        child = native.spawn_native(json.loads(sys.argv[1]), claim_fd=d,
                                    long_lived=sys.argv[2] == 'True')
        data = {'pid': child.pid}
    except native.NativeSpawnError as exc:
        data = {'error': str(exc)}
    Path(sys.argv[3]).write_text(json.dumps(data))
finally:
    claim.close(d)
""", json.dumps(argv), long_lived, result,
        env={"HFMODEM_COEXIST_FAULT": point + ":kill", "HFMODEM_COEXIST_LOG": str(log)})
    assert caller.wait(timeout=15) == (0 if long_lived else 1), caller.stderr.read()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    spawned, = [event for event in events if event["event"] == "native_spawn"]
    child = native.NativeChild(spawned["child_pid"], spawned["child_started"], None, None, None)
    children.append(child)
    assert child.alive()
    record = winproc.read_launch(child._win_entry)
    if phase is None:
        assert record is None
    else:
        assert record["phase"] == phase
        assert record["role"] == ("native" if long_lived else "one-shot")
        assert set(record["handles"]) == ({"claim"} if pushed else set())
        assert record["launcher"]["pid"] == spawned["pid"]
    raw_gate_free()
    if phase in ("created", "resuming"):
        with pytest.raises(claim.ClaimBusy, match=f"admission in progress.*{child.pid}"):
            claim.mutex(timeout=0)
    else:
        cleanup_available()
    claim_held() if pushed else claim_free()
    if point in ("native_after_creation", "native_after_record", "native_after_push_claim",
                 "native_after_resuming"):
        assert not marker.exists()
    else:
        wait_for(marker.exists)
    if long_lived:
        reply = json.loads(result.read_text())
        assert ("pid" in reply) == (point == "native_after_reply")
    child.kill()
    assert child.wait_gone(5)
    cleanup_available()
    claim_free()
    assert winproc.read_launch(child._win_entry) is None
    assert not Path(str(log) + ".lost").exists()


@pytest.mark.wall_clock
def test_n3_killed_supervisor_keeps_unacknowledged_admission(launchers, children, tmp_path):
    killed_boundary(launchers, children, tmp_path, "native_after_resume", "resuming", True,
                    long_lived=True)


@pytest.mark.wall_clock
@pytest.mark.parametrize("point,phase,pushed", BOUNDARIES)
def test_n4_supervisor_failure_table(launchers, children, tmp_path, point, phase, pushed):
    killed_boundary(launchers, children, tmp_path, point, phase, pushed, long_lived=True)


@pytest.mark.wall_clock
@pytest.mark.parametrize("point,phase,pushed", BOUNDARIES[:5])
def test_n4_one_shot_caller_failure_table(launchers, children, tmp_path, point, phase, pushed):
    killed_boundary(launchers, children, tmp_path, point, phase, pushed, long_lived=False)


@pytest.mark.wall_clock
@pytest.mark.parametrize("acknowledged,unresolved", [(False, False), (True, False),
                                                   (False, True), (True, True)])
def test_n5_caller_death_ends_child_or_preserves_last_phase(
        launchers, children, tmp_path, acknowledged, unresolved):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    point = "native_after_acknowledgement" if acknowledged else "native_before_acknowledgement"
    (barriers / point).touch()
    setup = "native._end = lambda child: False" if unresolved else "pass"
    caller = launchers("""
spawn = claim.spawn
def configured(d, argv, *, role, **options):
    script = sys.argv[1] + '\\nsys.exit(native._supervise(json.loads(sys.argv[1])))'
    return spawn(d, [sys.executable, '-c', script, argv[-1]], role=role, **options)
claim.spawn = configured
d = claim.acquire('caller', 'reply')
native.spawn_native(json.loads(sys.argv[2]), claim_fd=d)
""", PRELUDE + setup, json.dumps(audio_command("os.write(1,b'audio'); time.sleep(60)")),
        env={"HFMODEM_COEXIST_BARRIERS": str(barriers)})
    supervisor = None
    try:
        wait_for((barriers / (point + ".reached")).exists)
        record = native_record()
        child = recorded_child(record, children)
        supervisor = winproc.open(record["launcher"]["pid"], winproc.END)
        caller.kill()
        caller.wait(timeout=5)
        (barriers / (point + ".release")).touch()
        assert supervisor.wait(8)
        record = winproc.read_launch(child._win_entry)
        assert record["phase"] == ("acknowledged" if unresolved else "exited")
        assert child.alive() == unresolved
        if unresolved:
            claim_held()
            cleanup_available()
            child.kill()
            assert child.wait_gone(5)
        cleanup_available()
        claim_free()
    finally:
        (barriers / (point + ".release")).touch()
        if supervisor is not None:
            if not supervisor.wait():
                supervisor.terminate()
                assert supervisor.wait(5)
            supervisor.close()


@pytest.mark.wall_clock
def test_supervisor_exit_is_waited_after_reply(owned, children, tmp_path, monkeypatch):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    point = "native_after_reply"
    (barriers / point).touch()
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(barriers))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native,
            audio_command("os.write(1,b'audio'); time.sleep(60)"), claim_fd=owned)
        try:
            wait_for((barriers / (point + ".reached")).exists)
            with pytest.raises(TimeoutError):
                future.result(timeout=.1)
        finally:
            (barriers / (point + ".release")).touch()
            child = future.result(timeout=10)
            children.append(child)
    assert child.alive()
    supervisor, = records("supervisor")
    assert supervisor["phase"] == "exited"
    cleanup_available()


@pytest.mark.wall_clock
def test_n7_audio_first_byte_stderr_and_pushed_stdin(owned, children):
    payload = bytes(range(256)) * 512
    child = native.spawn_native(audio_command("""
os.write(2,b'capture diagnostic\\n')
os.write(1,bytes(range(256))*512)
print(sys.stdin.readline().strip(), flush=True)
time.sleep(60)
"""), claim_fd=owned)
    children.append(child)
    received = bytearray()
    while len(received) < len(payload):
        chunk = child.stdout.read(len(payload) - len(received))
        assert chunk
        received.extend(chunk)
    assert received == payload
    assert child.stderr.readline() == b"capture diagnostic\n"
    child.stdin.write(b"hello\n")
    assert child.stdout.readline().rstrip(b"\r\n") == b"hello"
    assert child.alive()
    assert winproc.read_launch(child._win_entry)["phase"] == "acknowledged"
    assert all(row["phase"] == "exited" for row in records("supervisor"))
    cleanup_available()
    assert child.alive()
    claim_held()


@pytest.mark.wall_clock
def test_child_exit_before_acknowledgement_fails_launch(owned, children, tmp_path, monkeypatch):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    point = "native_before_acknowledgement"
    (barriers / point).touch()
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(barriers))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native,
            audio_command("os.write(1,b'audio'); time.sleep(60)"), claim_fd=owned)
        try:
            wait_for((barriers / (point + ".reached")).exists)
            child = recorded_child(native_record(), children)
            child.kill()
            assert child.wait_gone(5)
            (barriers / (point + ".release")).touch()
            with pytest.raises(native.NativeSpawnError, match="exited before acknowledgement") as exc:
                future.result(timeout=5)
            children.append(exc.value.child)
            assert winproc.read_launch(child._win_entry)["phase"] == "exited"
            cleanup_available()
        finally:
            (barriers / (point + ".release")).touch()


def test_audio_detection_and_kernel_device_resolution(monkeypatch):
    assert native._audio([r"C:\bin\ffmpeg.exe", "-i", "test", "pipe:1"])
    assert native._audio(["capture.exe", "-f", "dshow", "-i", "test", "pipe:1"])
    assert not native._audio(["rigctl.exe", "-m", "1036", "-r", "COM41"])
    monkeypatch.setattr(winclaim, "serial_target", lambda port: r"\Device\test" if port == "COM41"
                        else pytest.fail("wrong port"))
    assert native._observation(["rigctl.exe", "-r", "COM41"], None) == ((r"\Device\test",), None, False, None)


@pytest.mark.parametrize("initial", [b"", b"W" * 78, b"stale" * 100])
def test_file_output_requires_growth_beyond_first_header(tmp_path, initial):
    output = tmp_path / "capture.wav"
    output.write_bytes(initial)
    observed = {}
    assert not native._output_ready(None, output, observed)
    output.write_bytes(b"W" * 78)
    assert not native._output_ready(None, output, observed)
    assert not native._output_ready(None, output, observed)
    with output.open("ab") as stream:
        stream.write(b"frames")
    assert native._output_ready(None, output, observed)


@pytest.mark.parametrize("signalled", [True, False])
def test_force_waits_after_terminate_access_denied(monkeypatch, signalled):
    clock, requests, waits, exits = [100], [], [], []
    error = winproc.ProcessAccessError(123, winproc.END)
    def terminate():
        requests.append(clock[0])
        clock[0] += .25
        raise error
    def wait(timeout):
        waits.append(timeout)
        clock[0] += timeout
        return signalled
    child = native.NativeChild.__new__(native.NativeChild)
    child._win_entry = SimpleNamespace(terminate=terminate, wait=wait, close=lambda: None)
    monkeypatch.setattr(child, "_record_exit", lambda: exits.append(clock[0]))
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    result = child._force("TerminateProcess")
    assert result.status == ("terminated" if signalled else "unresolved")
    assert (str(error) in result.detail) == (not signalled)
    assert bool(result) == signalled
    assert requests == [100]
    assert waits == [native.KILL_WAIT_S - .25]
    assert exits == ([105] if signalled else [])


def test_stop_channel_precedes_owned_stdin(monkeypatch):
    entry = SimpleNamespace(wait=lambda: True)
    monkeypatch.setattr(winproc, "request_stop", lambda target: 15)
    monkeypatch.setattr(winproc, "bounded_write", lambda *a, **k: pytest.fail("q before stop channel"))
    stopper = winproc.Stopper(entry, stdin=object())
    assert stopper.grace == 15
    assert stopper.deadline == stopper.requested_at + 16
    assert stopper.wait() == winproc.StopResult("stopped", "stop channel")


@pytest.mark.wall_clock
def test_n8_twelve_second_deadline(owned, children, monkeypatch):
    monkeypatch.setattr(native, "_observation", lambda *args: ((r"\Device\test",), None, False))
    started = time.monotonic()
    with pytest.raises(native.NativeSpawnError, match="never opened COM41") as exc:
        native.spawn_native(command("time.sleep(60)"), claim_fd=owned, long_lived=True, device="COM41")
    assert 12 <= time.monotonic() - started < 16
    children.append(exc.value.child)
    assert not exc.value.child.alive()
    cleanup_available()


@pytest.mark.wall_clock
@pytest.mark.parametrize("ending", ["deadline", "interrupt"])
def test_observation_bound_and_interrupt_during_inspection(
        owned, children, tmp_path, monkeypatch, ending):
    monkeypatch.setattr(native, "_observation", lambda *args: ((r"\Device\test",), None, False))
    supervisor_setup(monkeypatch, "native.OPEN_TIMEOUT_S=3")
    barriers, log = tmp_path / "barriers", tmp_path / "events"
    barriers.mkdir()
    for point in ("inspector_before_read", "inspector_after_walk"):
        (barriers / point).touch()
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(barriers))
    monkeypatch.setenv("HFMODEM_COEXIST_LOG", str(log))
    inspector = None
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native, command("time.sleep(60)"), claim_fd=owned,
                             long_lived=True, device="COM41")
        try:
            wait_for((barriers / "inspector_before_read.reached").exists, timeout=2.5)
            wait_for((barriers / "inspector_after_walk.reached").exists, timeout=2.5)
            record = native_record()
            inspector = winproc.open(records("inspector")[0]["child"]["pid"], winproc.END)
            assert not inspector.wait()
            events = [json.loads(line) for line in log.read_text().splitlines()]
            observing, = [event for event in events if event["event"] == "native_observing"]
            if ending == "interrupt":
                limit = time.monotonic() + .15
                with winproc.open(record["launcher"]["pid"]) as supervisor:
                    assert winproc.request_interrupt(supervisor) is not None
            else:
                limit = observing["deadline"] + .15
            reason = "interrupted" if ending == "interrupt" else "never opened COM41"
            with pytest.raises(native.NativeSpawnError, match=reason) as exc:
                future.result(timeout=5)
            children.append(exc.value.child)
            assert not exc.value.child.alive()
            events = [json.loads(line) for line in log.read_text().splitlines()]
            stops = [event for event in events if event["event"] == "native_stop"]
            assert len({event["child_pid"] for event in stops}) == len(stops)
            stopped, = [event for event in stops if event["child_pid"] == exc.value.child.pid]
            assert stopped["monotonic_ns"] / 1e9 <= limit
            assert not inspector.wait(), "inspector reclamation must not delay ending the native child"
            cleanup_available()
        finally:
            for point in ("inspector_before_read", "inspector_after_walk"):
                (barriers / (point + ".release")).touch()
            if inspector is not None:
                inspector.terminate()
                assert inspector.wait(5)
                inspector.close()


@pytest.mark.wall_clock
@pytest.mark.parametrize("outcome", ["busy-found", "blocked", "unavailable", "cannot-inspect"])
def test_observation_uses_only_positive_evidence(owned, children, monkeypatch, outcome):
    monkeypatch.setattr(native, "_observation", lambda *args: ((r"\Device\test",), None, False))
    setup = fr"""
native.OPEN_TIMEOUT_S=.3
attempts = 0
def walk(entries, devices, *, deadline, cancelled):
    global attempts
    assert callable(cancelled) and deadline > time.monotonic()
    assert devices == {{'COM41': r'\Device\test'}}
    attempts += 1
    pid = entries[0].pid
    if {outcome!r} == 'busy-found':
        if attempts == 1:
            return winproc.DeviceHolders({{}}, 'busy', 'inspection busy')
        result = winproc.DeviceWalk(frozenset(['COM41']), {{4: 'blocked', 8: 'found'}})
    elif {outcome!r} == 'unavailable':
        return winproc.DeviceHolders({{}}, 'unavailable', 'inspection unavailable: forced')
    elif {outcome!r} == 'cannot-inspect':
        result = winproc.DeviceWalk(message=f'cannot inspect pid {{pid}} (python.exe)')
    else:
        result = winproc.DeviceWalk(handles={{4: 'blocked'}})
    return winproc.DeviceHolders({{pid: result}}, 'incomplete')
winproc.device_holders = walk
"""
    supervisor_setup(monkeypatch, setup)
    if outcome == "busy-found":
        child = native.spawn_native(command("time.sleep(60)"), claim_fd=owned,
                                     long_lived=True, device="COM41")
    else:
        reason = {"blocked": "never opened COM41", "unavailable": "inspection unavailable",
                  "cannot-inspect": "cannot inspect pid.*python.exe"}[outcome]
        with pytest.raises(native.NativeSpawnError, match=reason) as exc:
            native.spawn_native(command("time.sleep(60)"), claim_fd=owned,
                                long_lived=True, device="COM41")
        child = exc.value.child
        assert not child.alive()
    children.append(child)
    cleanup_available()


@pytest.mark.wall_clock
def test_status_reader_unconfirmed_completion_is_retained(owned, tmp_path, monkeypatch):
    finish = winproc._finish_read
    release, finished = threading.Event(), threading.Event()
    readers = []
    pair = winproc.pipe_pair
    def captured_pair():
        reader, writer = pair()
        readers.append(reader)
        return reader, writer
    def delayed(*args):
        result = finish(*args)
        release.wait(10)
        finished.set()
        return result
    monkeypatch.setattr(winproc, "pipe_pair", captured_pair)
    monkeypatch.setattr(winproc, "_finish_read", delayed)
    start = len(winproc._unresolved_io)
    try:
        with pytest.raises(native.NativeSpawnError, match="status read unresolved"):
            native.spawn_native(audio_command("time.sleep(60)"), claim_fd=owned)
        reader, = readers
        assert reader._pending is not None
        assert not reader.close()
        assert winproc._api().GetFileType(reader.handle) == 3
        assert len(winproc._unresolved_io) > start
        wait_for(lambda: not winproc.live_launches()[0])
    finally:
        release.set()
        assert finished.wait(2)
        for pending in winproc._unresolved_io[start:]:
            if len(pending) == 3:
                pending[2].join(2)
                assert not pending[2].is_alive()
        for reader in readers:
            if reader._pending is not None:
                _, _, event = reader._pending
                reader._pending = None
                winclaim.close_handle(event)
            assert reader.close()
        del winproc._unresolved_io[start:]


@pytest.fixture
def dummy_rigctld_argv():
    hamlib = os.environ.get("HAMLIB_BIN")
    configured = Path(hamlib) / "rigctld.exe" if hamlib else None
    executable = (str(configured) if configured is not None and configured.is_file()
                  else shutil.which("rigctld.exe"))
    if executable is None:
        pytest.skip("portless native regression requires rigctld.exe")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return [executable, "-P", "NONE", "-m", "1", "--set-conf=ptt_type=None",
            "-T", "127.0.0.1", "-t", str(port)]


@pytest.mark.wall_clock
def test_portless_dummy_rigctld_acknowledged_by_listener(owned, children, dummy_rigctld_argv):
    child = native.spawn_native(dummy_rigctld_argv, claim_fd=owned)
    children.append(child)
    assert child.alive()
    record = winproc.read_launch(child._win_entry)
    assert record["role"] == "native" and record["phase"] == "acknowledged"
    assert native._listener_ready(("127.0.0.1", int(dummy_rigctld_argv[9])),
        deadline=time.monotonic() + 1, cancelled=lambda: False)
    cleanup_available()


@pytest.mark.wall_clock
def test_listening_rigctld_without_configured_device_is_refused(
        owned, children, dummy_rigctld_argv, com_pair):
    argv = [*dummy_rigctld_argv, "-r", "COM41"]
    endpoint = ("127.0.0.1", int(argv[9]))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(native.spawn_native, argv, claim_fd=owned)
        try:
            wait_for(lambda: native._listener_ready(endpoint,
                deadline=time.monotonic() + .2, cancelled=lambda: False))
        finally:
            with pytest.raises(native.NativeSpawnError, match="never opened COM41") as failure:
                future.result(timeout=20)
            children.append(failure.value.child)
    assert not failure.value.child.alive()
    cleanup_available()


@pytest.fixture
def real_executables(com_pair):
    hamlib = os.environ.get("HAMLIB_BIN")
    paths = {}
    for name in ("rigctl", "rigctld", "ffmpeg"):
        configured = Path(hamlib) / (name + ".exe") if hamlib and name != "ffmpeg" else None
        paths[name] = (str(configured) if configured is not None and configured.is_file() else
                       shutil.which(name + ".exe"))
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        pytest.skip("CD:W13 gate requires " + ", ".join(missing))
    for name in ("rigctl", "rigctld"):
        version = subprocess.run([paths[name], "--version"], capture_output=True, text=True,
                                 close_fds=True, timeout=5)
        assert "4.7.2" in version.stdout + version.stderr, "CD:W13 requires Hamlib 4.7.2"
    return paths


def rig_command(executable, *, daemon=False):
    argv = [executable, "-m", "1036", "-r", "COM41", "-s", "38400"]
    if daemon:
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            number = port.getsockname()[1]
        argv += ["-T", "127.0.0.1", "-t", str(number)]
    return argv


def ffmpeg_command(executable, output):
    return [executable, "-y", "-nostats", "-loglevel", "error", "-re", "-f", "lavfi",
            "-i", "sine=frequency=1000:sample_rate=8000", "-c:a", "pcm_s16le",
            "-flush_packets", "1", "-f", "s16le", str(output)]


def real_supervisor_kill(argv, kind, stage, owned, children, tmp_path, monkeypatch):
    barriers = tmp_path / f"{kind}-{stage}"
    barriers.mkdir()
    point = "native_before_acknowledgement" if stage == "before" else "native_after_acknowledgement"
    (barriers / point).touch()
    with monkeypatch.context() as patch:
        patch.setenv("HFMODEM_COEXIST_BARRIERS", str(barriers))
        with ThreadPoolExecutor() as pool:
            future = pool.submit(native.spawn_native, argv, claim_fd=owned,
                                 long_lived=True, device="COM41" if kind == "rigctld" else None)
            try:
                wait_for((barriers / (point + ".reached")).exists, timeout=15)
                record = wait_for(lambda: next((r for r in records("native")
                    if r["phase"] == ("resuming" if stage == "before" else "acknowledged")), None))
                child = recorded_child(record, children)
                with winproc.open(record["launcher"]["pid"], winproc.END) as supervisor:
                    supervisor.terminate()
                    assert supervisor.wait(5)
                with pytest.raises(native.NativeSpawnError, match="without acknowledgement"):
                    future.result(timeout=5)
                claim.close(owned)
                assert child.alive()
                claim_held()
                raw_gate_free()
                if stage == "before":
                    with pytest.raises(claim.ClaimBusy, match=f"admission in progress.*{child.pid}"):
                        claim.mutex(timeout=0)
                else:
                    cleanup_available()
                assert child.alive()
                claim_held()
                child.kill()
                assert child.wait_gone(5)
                cleanup_available()
                claim_free()
            finally:
                (barriers / (point + ".release")).touch()


@pytest.mark.wall_clock
def test_n6_cd_w13_real_executables(real_executables, com_pair, children, tmp_path,
                                   monkeypatch, record_property):
    executables = real_executables
    passed = set()
    peer = SerialFakeYaesu("COM42", "ft891")
    try:
        for name in ("rigctl", "rigctld"):
            evidence = tmp_path / (name + "-walk.json")
            with monkeypatch.context() as patch:
                supervisor_setup(patch, f"""
walk = winproc.device_holders
def observed(entries, devices, **options):
    result = walk(entries, devices, **options)
    for pid, found in result.entries.items():
        if 'COM41' in found.devices:
            Path({str(evidence)!r}).write_text(json.dumps(dict(pid=pid, handles=found.handles,
                devices=list(found.devices))))
    return result
winproc.device_holders = observed
""")
                with acquired() as descriptor:
                    child = native.spawn_native(rig_command(executables[name], daemon=name == "rigctld"),
                        claim_fd=descriptor, long_lived=True, device="COM41")
                    children.append(child)
                    assert child.alive()
                    found = json.loads(evidence.read_text())
                    assert found["pid"] == child.pid and found["devices"] == ["COM41"]
                    assert "found" in found["handles"].values()
                    assert "other" in found["handles"].values()
                    assert winproc.read_launch(child._win_entry)["phase"] == "acknowledged"
                    cleanup_available()
                    if name == "rigctl":
                        child.stdin.write(b"f\n")
                        response = bytearray()
                        wait_for(lambda: winproc.pipe_available(child.stdout.fileno()))
                        deadline = time.monotonic() + 8
                        while b"7100000" not in response and time.monotonic() < deadline:
                            count = winproc.pipe_available(child.stdout.fileno())
                            if count:
                                response.extend(child.stdout.read(count))
                            else:
                                time.sleep(.01)
                        assert b"7100000" in response, response
                        found = winproc.device_holders([child._win_entry], {"COM41"},
                                                       deadline=time.monotonic() + 5)
                        walk = found.entries[child.pid]
                        assert walk.devices == {"COM41"}
                        assert "other" in walk.handles.values()
                    child.kill()
                    assert child.wait_gone(5)
                    passed.add(name)
        with acquired() as descriptor:
            child = native.spawn_native(rig_command(executables["rigctl"]) + ["f"], claim_fd=descriptor)
            children.append(child)
            out, err = child._process.communicate(timeout=10)
            assert child._process.returncode == 0, err
            assert b"7100000" in out
            assert winproc.read_launch(child._win_entry)["phase"] == "exited"
            passed.add("one-shot")
        with acquired() as descriptor:
            held = com_pair.Serial()
            held.port, held.rts, held.dtr = "COM41", False, False
            held.open()
            try:
                with pytest.raises(native.NativeSpawnError, match="exited before acknowledgement") as exc:
                    native.spawn_native(rig_command(executables["rigctl"]), claim_fd=descriptor,
                                        long_lived=True, device="COM41")
                children.append(exc.value.child)
                assert not exc.value.child.alive()
                assert "exit code 2" in str(exc.value)
                cleanup_available()
                passed.add("missing-device")
            finally:
                held.close()
        for sink in ("file", "pipe"):
            output = tmp_path / "sine.raw" if sink == "file" else "pipe:1"
            with acquired() as descriptor:
                child = native.spawn_native(ffmpeg_command(executables["ffmpeg"], output), claim_fd=descriptor)
                children.append(child)
                assert child.alive()
                assert winproc.read_launch(child._win_entry)["phase"] == "acknowledged"
                if sink == "file":
                    assert output.stat().st_size > 0
                else:
                    assert winproc.pipe_available(child.stdout.fileno()) > 0
                    assert child.stdout.read(2) == b"\x00\x00"
                cleanup_available()
                assert native._end(child, grace=.1)
                passed.add("ffmpeg-" + sink)
        for kind in ("rigctld", "ffmpeg"):
            for stage in ("before", "after"):
                cleanup_available()
                with acquired() as descriptor:
                    argv = (rig_command(executables[kind], daemon=True) if kind == "rigctld" else
                            ffmpeg_command(executables[kind], tmp_path / f"{stage}.raw"))
                    real_supervisor_kill(argv, kind, stage, descriptor, children, tmp_path, monkeypatch)
                passed.add(kind + "-" + stage)
        assert peer.error is None
        assert passed == N6_CASES
        record_property("windows_native_gate", "CD:W13: " + ", ".join(sorted(passed)))
    finally:
        peer.close()


@pytest.mark.wall_clock
@pytest.mark.parametrize("role", ["python", "supervisor", "native", "one-shot", "cleanup",
                                  "inspector", "recorder"])
def test_n10_every_role_claim_is_noninheritable(owned, children, role):
    code = """
record = winproc.read_launch()
if record['role'] == 'recorder':
    assert 'claim' not in record['handles']
else:
    assert not os.get_handle_inheritable(record['handles']['claim'])
print('checked', flush=True)
sys.stdin.buffer.read(1)
"""
    if role == "native":
        child = native.spawn_native(audio_command(code.replace("print('checked', flush=True)",
                        "os.write(1,b'checked\\n')")), claim_fd=owned)
        children.append(child)
        assert child.stdout.readline() == b"checked\n"
        assert winproc.read_launch(child._win_entry)["role"] == role
    elif role == "one-shot":
        child = native.spawn_native(command(code), claim_fd=owned)
        children.append(child)
        assert child.stdout.readline().rstrip(b"\r\n") == b"checked"
    else:
        admission = claim.inherited(owned, launch=True)
        launch = None
        try:
            launch = claim.spawn(admission, command(code), role=role, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process = launch.process
            assert process.stdout.readline().rstrip(b"\r\n") == b"checked"
            process.communicate(b"x", timeout=5)
            assert process.returncode == 0
        finally:
            admission.failed()
            if launch is not None:
                if launch.process.poll() is None:
                    launch.process.kill()
                    launch.process.wait(timeout=5)
                for stream in (launch.process.stdin, launch.process.stdout, launch.process.stderr):
                    stream.close()
                launch.child.close()


@pytest.mark.wall_clock
def test_n10_native_nt_ratchet(owned, children, monkeypatch):
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("os.kill reached on nt"))
    monkeypatch.delattr(signal, "SIGKILL", raising=False)
    monkeypatch.setattr(native, "fcntl", SimpleNamespace())
    monkeypatch.setattr(native, "termios", SimpleNamespace())
    monkeypatch.setattr(native, "select", SimpleNamespace())
    monkeypatch.setattr(native, "_process_state", lambda *args: pytest.fail("ps reached on nt"))
    for operation in ("interrupt", "terminate", "kill", "end"):
        child = native.spawn_native(command("print('ready',flush=True); sys.stdin.readline()"),
                                    claim_fd=owned)
        children.append(child)
        assert child.stdout.readline().rstrip(b"\r\n") == b"ready"
        assert child.alive() and not child.wait_gone(0)
        assert native._end(child, grace=.1) if operation == "end" else getattr(child, operation)()
        assert child.wait_gone(5)
    supervisor_setup(monkeypatch, """
os.kill = lambda *args: (_ for _ in ()).throw(AssertionError('os.kill on nt'))
if hasattr(signal, 'SIGKILL'): del signal.SIGKILL
native.select = native.fcntl = native.termios = object()
native._process_state = lambda *args: (_ for _ in ()).throw(AssertionError('ps on nt'))
""")
    child = native.spawn_native(audio_command("os.write(1,b'audio'); sys.stdin.readline()"), claim_fd=owned)
    children.append(child)
    assert child.stdout.read(5) == b"audio"
    assert native._end(child, grace=.1)
    cleanup_available()


@pytest.mark.wall_clock
def test_recorder_channel_finalizes_production_witness_after_receiver_goes_silent(
        owned, children, tmp_path):
    import struct
    import wave
    from consolelib import claim as console_claim

    root = Path(__file__).resolve().parents[5]
    output = tmp_path / 'witness.wav'
    sidecar = output.with_suffix('.json')
    inventory = tmp_path / 'kiwi.csv'
    log = tmp_path / 'witness.log'
    silent, finished = threading.Event(), threading.Event()
    errors = []
    # Every block has a brief loud burst above the quiet background, so the
    # production placement probe can establish that this receiver hears audio.
    samples = struct.pack('>h', 30) * 960 + struct.pack('>h', 10000) * 240
    def frame(payload):
        return b'\x82\x7e' + struct.pack('>H', len(payload)) + payload

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        listener.settimeout(20)
        port = listener.getsockname()[1]
        inventory.write_text('hostname,port,name_location,lat_lon,free_channels\n'
                             f'127.0.0.1,{port},Local fake,"41.3,-87.6",1\n')
        def serve():
            try:
                with listener.accept()[0] as connection:
                    connection.settimeout(10)
                    request = bytearray()
                    while b'\r\n\r\n' not in request:
                        request.extend(connection.recv(4096))
                    connection.sendall(b'HTTP/1.1 101 Switching Protocols\r\n'
                                       b'Upgrade: websocket\r\n\r\n')
                    connection.sendall(frame(b'MSG\x00sample_rate=12000'))
                    connection.settimeout(.1)
                    deadline = time.monotonic() + 20
                    while not sidecar.exists() and not finished.is_set():
                        if time.monotonic() >= deadline:
                            raise TimeoutError('witness never placed')
                        connection.sendall(frame(b'SND' + bytes(7) + samples))
                        # Drain commands; the stop test concerns a quiet inbound
                        # stream. Outbound pressure is covered in-process.
                        try:
                            if not connection.recv(4096):
                                raise EOFError('witness exited before placement')
                        except socket.timeout:
                            pass
                    silent.set()
                    while not finished.is_set():
                        try:
                            if not connection.recv(4096):
                                break
                        except socket.timeout:
                            pass
            except BaseException as exc:
                if not finished.is_set():
                    errors.append(exc)
                    silent.set()
        receiver = threading.Thread(target=serve, daemon=True)
        receiver.start()
        process = None
        try:
            with log.open('wb') as diagnostics:
                process = console_claim.spawn(owned,
                    [sys.executable, str(root / 'tools/kiwi_witness.py'), '--dial', '7100',
                     '--seconds', '720', '--grid', 'EN54ID', '--near', '0', '--far', '1000',
                     '--candidates', str(inventory), '--out', str(output)],
                    role='recorder', stdout=diagnostics, stderr=subprocess.STDOUT)
            children.append(process._claim_child)
            assert silent.wait(25), log.read_text(errors='replace')
            assert not errors and sidecar.exists(), (errors, log.read_text(errors='replace'))
            assert process.poll() is None
            def initial_metadata():
                try:
                    return json.loads(sidecar.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    return None
            before = wait_for(initial_metadata)
            assert not before['complete'] and before['seconds'] > 0
            record = winproc.read_launch(process._claim_child._win_entry)
            assert record['role'] == 'recorder' and record['grace'] == 30
            console_claim.end_child(process)
            assert process.wait(timeout=5) == 0, log.read_text(errors='replace')
            facts = json.loads(sidecar.read_text())
            assert facts['complete'] and facts['seconds'] >= before['seconds']
            with wave.open(str(output), 'rb') as audio:
                assert audio.getnchannels() == 1 and audio.getframerate() == 12000
                assert audio.getnframes() == round(facts['seconds'] * 12000)
                assert len(audio.readframes(audio.getnframes())) == audio.getnframes() * 2
        finally:
            finished.set()
            if process is not None and process.poll() is None:
                console_claim.end_child(process)
            receiver.join(2)
        assert not receiver.is_alive() and not errors
