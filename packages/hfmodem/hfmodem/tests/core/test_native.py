# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hfmodem.core import claim, native, posix, winclaim, winproc
from hfmodem.tests import hermetic

fcntl = posix.module("fcntl")


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch, request):
    if (os.name == "nt" and request.node.originalname !=
            "test_windows_shell_and_cleanup_refuse_before_open_or_spawn"):
        pytest.skip("native launch tests require POSIX")
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_COEXIST_RUN", "native-unit")
    monkeypatch.setenv("HFMODEM_STATE_DIR", str(tmp_path / "state"))
    for key in ("HFMODEM_COEXIST_FAULT", "HFMODEM_COEXIST_LOG", "HFMODEM_COEXIST_BARRIERS"):
        monkeypatch.delenv(key, raising=False)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    ps = fakebin / "ps"
    ps.write_text(f"#!{sys.executable}\nimport os, sys\n"
                  "pid = int(sys.argv[sys.argv.index('-p') + 1])\n"
                  "try: os.kill(pid, 0)\n"
                  "except ProcessLookupError: sys.exit(1)\n"
                  "print(('fake-' + str(pid)) if 'lstart=' in sys.argv else "
                  "'S' if 'stat=' in sys.argv else 'fake-native')\n")
    ps.chmod(0o700)
    monkeypatch.setenv("PATH", str(fakebin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(claim, "audio_nodes", lambda *a, **k: ())
    monkeypatch.setattr(claim, "_lsof", lambda paths: "")
    monkeypatch.setattr(native, "REAP_GRACE_S", 0.1)


@pytest.fixture
def owned():
    descriptor = claim.acquire("test", "native")
    yield descriptor
    claim.release(descriptor)


@pytest.fixture
def children():
    tracked = []
    yield tracked
    for child in tracked:
        try:
            native._end(child, grace=0.1)
        finally:
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


def command(code):
    return [sys.executable, "-c", code]


@pytest.mark.wall_clock
def test_interrupt_delivers_sigint_on_posix(owned, children):
    child = native.spawn_native(command(
        "import signal,sys,time; "
        "signal.signal(signal.SIGINT, lambda *args: sys.exit(42)); "
        "print('ready', flush=True); time.sleep(30)"), claim_fd=owned)
    children.append(child)
    assert child.stdout.readline() == b"ready\n"
    assert child.interrupt()
    assert child.wait_gone(5)
    assert child._process.wait(timeout=5) == 42


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition timed out")


def mutex_available():
    fd = os.open(claim.mutex_path(), os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    finally:
        os.close(fd)


def records():
    return [path for path in claim.state_dir().glob("rig.claim.native.d/*")
            if not path.name.endswith(".tmp")]


def record_child(path):
    data = json.loads(path.read_text())
    return native.NativeChild(data["child_pid"], data["child_started"], None, None, None)


def test_one_shot_inherits_claim_and_own_admission(owned, children, tmp_path):
    inherited = os.open(tmp_path / "unrelated", os.O_RDWR | os.O_CREAT, 0o600)
    os.set_inheritable(inherited, True)
    try:
        child = native.spawn_native(command(
            "import os,sys; "
            "print(os.lseek(9,0,os.SEEK_CUR),flush=True); "
            f"print(any(os.path.exists('/proc/self/fd/' + str(n)) and "
            f"os.path.realpath('/proc/self/fd/' + str(n)) == {str(tmp_path / 'unrelated')!r} "
            "for n in range(3,128)),flush=True); sys.stdin.buffer.read(1)"), claim_fd=owned)
        children.append(child)
        assert int(child.stdout.readline()) == owned.offset()
        assert child.stdout.readline() == b"False\n"
        assert child.alive()
        assert not mutex_available()
        child.stdin.write(b"x")
        child.stdin.flush()
        assert child.wait_gone(5)
        assert mutex_available()
    finally:
        os.close(inherited)


def test_refuses_before_spawn_during_cleanup(owned, monkeypatch):
    mutex = claim.mutex(timeout=0)
    monkeypatch.setattr(native, "_helper", lambda *a, **k: pytest.fail("spawned during cleanup"))
    try:
        with pytest.raises(claim.ClaimBusy, match="cleanup in progress"):
            native.spawn_native(command("pass"), claim_fd=owned)
    finally:
        claim.mutex_release(mutex)


def test_reopened_claim_never_admits(owned):
    fd = os.open(claim.claim_path(), os.O_RDWR)
    try:
        with pytest.raises(claim.ClaimError, match="nonce"):
            native.spawn_native(command("pass"), claim_fd=fd)
        assert mutex_available()
    finally:
        os.close(fd)


def test_cleanup_one_shot_inherits_exclusive_description(owned, children):
    mutex = claim.mutex(timeout=0)
    try:
        child = native.spawn_native(command(
            "import os,sys; print(os.lseek(8,0,os.SEEK_CUR),flush=True); sys.stdin.read(1)"),
            claim_fd=owned, cleanup_fd=mutex)
        children.append(child)
        assert int(child.stdout.readline()) == claim.ClaimDescriptor(mutex).offset()
        os.close(mutex)
        mutex = None
        assert not mutex_available()
        assert native._end(child, grace=0.1)
        assert mutex_available()
    finally:
        if mutex is not None:
            claim.mutex_release(mutex)


def test_exec_failure_releases_admission(owned):
    argv = ["/nonexistent-native-unit-command"]
    hermetic.guard.admit(argv, os.environ)
    with pytest.raises(native.NativeSpawnError, match="No such file") as exc:
        native.spawn_native(argv, claim_fd=owned)
    child = exc.value.child
    assert not child.alive()
    for pipe in (child.stdin, child.stdout, child.stderr):
        pipe.close()
    assert mutex_available()


def test_identity_failure_prevents_exec(owned, monkeypatch, tmp_path):
    started = claim.process_started
    monkeypatch.setattr(claim, "process_started", lambda pid: started(pid) if pid == os.getpid()
                        else (_ for _ in ()).throw(claim.ClaimError("identity unavailable")))
    output = tmp_path / "opened"
    with pytest.raises(claim.ClaimError, match="identity unavailable"):
        native.spawn_native(command(f"open({str(output)!r},'w').close()"), claim_fd=owned)
    assert not output.exists()
    assert mutex_available()


def supervise_observer(monkeypatch, *, timeout=1.0, observe=False, output_observed=None):
    helper = native._helper
    def launch(verb, config, **kwargs):
        if verb != "--supervise":
            return helper(verb, config, **kwargs)
        setup = ("import json,sys; from hfmodem.core import native; "
                 f"native.OPEN_TIMEOUT_S={timeout}; native.REAP_GRACE_S=.1; ")
        if observe:
            setup += "native._opened=lambda pid,nodes: bool(nodes) and __import__('pathlib').Path(nodes[0]).exists(); "
        if output_observed is not None:
            setup += (
                "\noutput_ready = native._output_ready\n"
                "def output_probe(fd, output, observed):\n"
                "    ready = output_ready(fd, output, observed)\n"
                "    if observed.get('file', (0, 0, 0))[2] == 78:\n"
                f"        __import__('pathlib').Path({str(output_observed)!r}).touch()\n"
                "    return ready\n"
                "native._output_ready = output_probe\n")
        return subprocess.Popen([sys.executable, "-c", setup +
                                 "sys.exit(native._supervise(json.loads(sys.argv[1])))",
                                 json.dumps(config)], close_fds=True, **kwargs)
    monkeypatch.setattr(native, "_helper", launch)


def audio_command(tmp_path, code, output="pipe:1", backend="pulse"):
    script = tmp_path / ("capture-" + str(time.monotonic_ns()))
    script.write_text(f"#!{sys.executable}\n" + code + "\n")
    script.chmod(0o700)
    argv = [str(script), "-f", backend, "-i", "fake", str(output)]
    hermetic.guard.admit(argv, os.environ)
    return argv


def test_audio_first_byte_is_untouched_and_stderr_preserved(owned, children, tmp_path):
    payload = bytes(range(256)) * 512
    argv = audio_command(tmp_path,
        "import os,time; os.write(2,b'capture diagnostic\\n'); "
        "os.write(1,bytes(range(256))*512); time.sleep(20)")
    child = native.spawn_native(argv, claim_fd=owned, device="server")
    children.append(child)
    received = bytearray()
    while len(received) < len(payload):
        received.extend(child.stdout.read(len(payload) - len(received)))
    assert received == payload
    assert child.stderr.readline() == b"capture diagnostic\n"
    assert child.alive()
    assert mutex_available()
    assert records() == []


def test_file_audio_waits_beyond_header(owned, children, tmp_path, monkeypatch):
    observed = tmp_path / "observed"
    supervise_observer(monkeypatch, timeout=5, output_observed=observed)
    output = tmp_path / "capture.wav"
    release = tmp_path / "release"
    argv = audio_command(tmp_path,
        f"import os,time; f=open({str(output)!r},'wb',buffering=0); f.write(b'W'*78)\n"
        f"while not os.path.exists({str(release)!r}): time.sleep(.01)\n"
        "f.write(b'frames'); time.sleep(20)", output)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native, argv, claim_fd=owned)
        wait_for(observed.exists)
        assert not future.done()
        assert not mutex_available()
        release.touch()
        child = future.result(timeout=8)
        children.append(child)
    assert mutex_available()
    assert output.read_bytes() == b"W" * 78 + b"frames"


def test_file_output_requires_growth_after_first_observation(tmp_path):
    output = tmp_path / "capture.wav"
    observed = {}
    assert not native._output_ready(None, output, observed)
    output.write_bytes(b"W" * 78)
    assert not native._output_ready(None, output, observed)
    # Neither an unchanged header nor metadata changes prove captured frames.
    assert not native._output_ready(None, output, observed)
    os.utime(output, ns=(1, 1))
    assert not native._output_ready(None, output, observed)
    with output.open("ab") as stream:
        stream.write(b"frames")
    assert native._output_ready(None, output, observed)


def test_a_header_written_after_an_empty_first_look_is_not_audio(tmp_path):
    # The order the 10-27 Linux qualification traced: created, seen empty, header written.
    output = tmp_path / "silent.wav"
    output.touch()
    observed = {}
    assert not native._output_ready(None, output, observed)
    output.write_bytes(b"W" * 78)
    assert not native._output_ready(None, output, observed)
    assert not native._output_ready(None, output, observed)
    with output.open("ab") as stream:
        stream.write(b"frames")
    assert native._output_ready(None, output, observed)


@pytest.mark.parametrize("change", ["truncate", "replace", "remove"])
def test_file_output_restarts_baseline_on_new_capture(tmp_path, change):
    output = tmp_path / "capture.wav"
    output.write_bytes(b"stale" * 100)
    observed = {}
    assert not native._output_ready(None, output, observed)
    if change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"W" * 78)
        replacement.replace(output)
    else:
        if change == "remove":
            output.unlink()
            assert not native._output_ready(None, output, observed)
        output.write_bytes(b"W" * 78)
    assert not native._output_ready(None, output, observed)
    assert not native._output_ready(None, output, observed)
    with output.open("ab") as stream:
        stream.write(b"frames")
    assert native._output_ready(None, output, observed)


@pytest.mark.parametrize("mode", ["silent", "header", "stale", "exited"])
def test_unacknowledged_audio_is_ended(owned, tmp_path, monkeypatch, mode):
    supervise_observer(monkeypatch, timeout=0.35)
    output = tmp_path / "capture.wav"
    if mode == "stale":
        output.write_bytes(b"stale" * 100)
    code = "import time,sys; sys.stderr.write('no frames\\n'); sys.stderr.flush(); "
    if mode == "header":
        code += f"open({str(output)!r},'wb').write(b'W'*78); "
    code += "time.sleep(20)" if mode != "exited" else "sys.exit(3)"
    argv = audio_command(tmp_path, code, output if mode in ("header", "stale") else "pipe:1")
    with pytest.raises(native.NativeSpawnError) as exc:
        native.spawn_native(argv, claim_fd=owned)
    child = exc.value.child
    try:
        assert not child.alive()
        assert child.stderr.read() == b"no frames\n"
        assert mutex_available()
        assert records() == []
    finally:
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()


def test_node_acknowledgement_needs_no_output(owned, children, tmp_path, monkeypatch):
    supervise_observer(monkeypatch, observe=True)
    node = tmp_path / "opened-node"
    monkeypatch.setattr(native, "_observation", lambda *a: ((str(node),), None, True))
    child = native.spawn_native(audio_command(tmp_path,
        f"import time; f=open({str(node)!r},'w'); time.sleep(20)", backend="alsa"), claim_fd=owned)
    children.append(child)
    assert child.alive()
    assert mutex_available()
    assert not __import__('select').select([child.stdout], [], [], 0)[0]


def test_cat_node_acknowledgement_and_interactive_pipe(owned, children, tmp_path, monkeypatch):
    supervise_observer(monkeypatch, observe=True)
    node = tmp_path / "fake-cat"
    child = native.spawn_native(command(
        f"import sys; f=open({str(node)!r},'w'); "
        "print(sys.stdin.readline().strip(),flush=True)"), claim_fd=owned,
        long_lived=True, device=str(node))
    children.append(child)
    assert child.alive()
    assert mutex_available()
    child.stdin.write(b"hello\n")
    assert child.stdout.readline() == b"hello\n"
    assert child.wait_gone(5)
    with pytest.raises(BrokenPipeError):
        child.stdin.write(b"again\n")


def test_supervisor_killed_before_ack_retains_child_admission(owned, children, tmp_path, monkeypatch):
    supervise_observer(monkeypatch, timeout=5)
    argv = audio_command(tmp_path, "import time; time.sleep(20)")
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native, argv, claim_fd=owned)
        path = wait_for(lambda: next((p for p in records() if 'child_pid' in p.read_text()), None))
        child = record_child(path)
        children.append(child)
        os.kill(json.loads(path.read_text())["pid"], signal.SIGKILL)
        with pytest.raises(native.NativeSpawnError, match="without acknowledgement"):
            future.result(timeout=8)
    assert child.alive()
    assert not mutex_available()
    assert path.exists()
    assert native._end(child, grace=0.1)
    assert mutex_available()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, posix.SIGHUP])
def test_supervisor_traps_and_verifies_child_termination(owned, tmp_path, monkeypatch, sig):
    supervise_observer(monkeypatch, timeout=5)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native,
            audio_command(tmp_path, "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"),
            claim_fd=owned)
        path = wait_for(lambda: next((p for p in records() if 'child_pid' in p.read_text()), None))
        child = record_child(path)
        os.kill(json.loads(path.read_text())["pid"], sig)
        with pytest.raises(native.NativeSpawnError, match="interrupted") as exc:
            future.result(timeout=8)
    try:
        assert not child.alive()
        assert mutex_available()
        assert records() == []
    finally:
        for pipe in (exc.value.child.stdin, exc.value.child.stdout, exc.value.child.stderr):
            pipe.close()


def test_two_stage_records_and_concurrent_supervisors(owned, children, tmp_path, monkeypatch):
    supervise_observer(monkeypatch, timeout=5)
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    (barriers / "native_before_spawn").touch()
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(barriers))
    with ThreadPoolExecutor() as pool:
        first = pool.submit(native.spawn_native,
                           audio_command(tmp_path, "import time; time.sleep(20)"), claim_fd=owned)
        second = pool.submit(native.spawn_native,
                            audio_command(tmp_path, "import os,time; os.write(1,b'x'); time.sleep(20)"), claim_fd=owned)
        wait_for(lambda: len(records()) == 2)
        for path in records():
            data = json.loads(path.read_text())
            assert data["run"] == "native-unit"
            assert data["argv"]
            assert "child_pid" not in data
        (barriers / "native_before_spawn.release").touch()
        children.append(second.result(timeout=8))
        path = wait_for(lambda: records()[0] if len(records()) == 1 else None)
        wait_for(lambda: 'child_pid' in path.read_text())
        stranded = record_child(path)
        children.append(stranded)
        os.kill(json.loads(path.read_text())["pid"], signal.SIGKILL)
        with pytest.raises(native.NativeSpawnError):
            first.result(timeout=8)
    assert not mutex_available()
    assert path.exists()


def test_partial_and_stale_records_cannot_authorize(owned):
    directory = claim.state_dir() / "rig.claim.native.d"
    directory.mkdir()
    (directory / "partial").write_text('{"pid":')
    (directory / "stale").write_text('{"pid":99999999,"started":"stale"}')
    mutex = claim.mutex(timeout=0)
    try:
        with pytest.raises(claim.ClaimBusy):
            native.spawn_native(command("pass"), claim_fd=owned)
    finally:
        claim.mutex_release(mutex)


def test_pid_reuse_and_unresolved_identity_never_signal(monkeypatch):
    child = native.NativeChild(123, "old", None, None, None)
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(claim, "process_started", lambda pid: "new")
    assert not child.alive()
    assert not child.terminate()
    assert not child.kill()
    assert set(sent) == {0}
    monkeypatch.setattr(claim, "process_started", lambda pid: (_ for _ in ()).throw(claim.ClaimError("no identity")))
    with pytest.raises(claim.ClaimError, match="unresolved"):
        child.terminate()
    assert set(sent) == {0}


def test_alive_caches_identity_but_keeps_checking_state(monkeypatch):
    child = native.NativeChild(123, "old", None, None, None)
    inspected = []
    now = [0.0]
    state = ["S"]
    monkeypatch.setattr(os, "kill", lambda *args: None)
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(claim, "process_started", lambda pid: inspected.append(pid) or "old")
    monkeypatch.setattr(native, "_process_state", lambda pid: state[0])
    assert child.alive()
    now[0] += native.POLL_S
    assert child.alive()
    assert inspected == [123]
    state[0] = "Z"
    assert not child.alive()
    state[0] = "S"
    now[0] = native.IDENTITY_CACHE_S
    assert child.alive()
    assert inspected == [123, 123]
    another = native.NativeChild(123, "old", None, None, None)
    assert another.alive()
    assert inspected == [123, 123, 123]


@pytest.mark.parametrize("method", ["terminate", "kill"])
@pytest.mark.parametrize("identity", ["new", None])
def test_signals_refresh_cached_identity(monkeypatch, method, identity):
    child = native.NativeChild(123, "old", None, None, None)
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(claim, "process_started", lambda pid: "old")
    monkeypatch.setattr(native, "_process_state", lambda pid: "S")
    assert child.alive()

    def changed(pid):
        if identity is None:
            raise claim.ClaimError("no identity")
        return identity

    monkeypatch.setattr(claim, "process_started", changed)
    if identity is None:
        with pytest.raises(claim.ClaimError, match="unresolved"):
            getattr(child, method)()
        with pytest.raises(claim.ClaimError, match="unresolved"):
            child.alive()
    else:
        assert not getattr(child, method)()
        assert not child.alive()
    assert set(sent) == {0}


def test_termination_is_outcome_judged(monkeypatch):
    attempts = []
    child = SimpleNamespace(terminate=lambda: attempts.append("TERM"),
                            kill=lambda: attempts.append("KILL"), wait_gone=lambda timeout: False)
    assert not native._end(child, grace=0)
    assert attempts == ["TERM", "KILL"]


def test_darwin_descriptor_observation_abi():
    assert ctypes.sizeof(native._ProcFD) == 8
    assert ctypes.sizeof(native._VnodePath) == 1200
    assert native._VnodePath.path.offset == 176
    class Lib:
        def proc_pidinfo(self, pid, flavor, arg, entries, size):
            assert (pid, flavor, arg) == (12, 1, 0)
            if entries is None:
                return 24
            for entry, fd, kind in zip(entries, (3, 4, 5), (1, 2, 1)):
                entry.fd, entry.kind = fd, kind
            return 24

        def proc_pidfdinfo(self, pid, fd, flavor, info, size):
            assert flavor == 2 and fd != 4
            if fd == 5:
                return 0
            info._obj.path = b"/dev/tty.fake"
            return size
    assert native._darwin_paths(12, Lib()) == ("/dev/tty.fake",)


def test_both_macos_nodes_and_linux_audio_resolution(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    nodes, output, audio, listener = native._observation(["native", "-r", "/dev/cu.fake"], None)
    assert nodes == ("/dev/cu.fake", "/dev/tty.fake")
    monkeypatch.setattr(native, "_darwin_paths", lambda pid: ("/dev/tty.fake",))
    assert native._opened(12, nodes)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(claim, "audio_nodes", lambda endpoint, direction: ("/dev/snd/pcmC2D0c",))
    assert native._observation(["ffmpeg", "-i", "hw:2", "pipe:1"], "hw:2") == (
        ("/dev/snd/pcmC2D0c",), None, True, None)


@pytest.mark.parametrize("blank", ["", " "])
def test_blank_long_lived_device_is_no_device(monkeypatch, blank):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert native._observation(["rigctld", "-r", "/dev/cu.fake"], blank)[0] == (
        "/dev/cu.fake", "/dev/tty.fake")
    for argv in (["rigctld", "-r", blank, "-s", "38400"], ["rigctld", f"--rig-file={blank}"]):
        with pytest.raises(claim.ClaimError, match="requires a device"):
            native._observation(argv, blank)


@pytest.mark.parametrize("program", ["rigctld", "rigctld.exe"])
@pytest.mark.parametrize("host,port", [("127.0.0.1", 4532), ("127.0.0.1", 14532)])
def test_portless_observation_is_explicit_in_argv(monkeypatch, program, host, port):
    monkeypatch.setattr(claim, "serial_nodes", lambda *a: pytest.fail("portless device walk"))
    argv = [program, "-P", "NONE", "-m", "1", "--set-conf=ptt_type=None", "-T", host, "-t", str(port)]
    assert native._long_lived(argv)
    assert native._observation(argv, None) == ((), None, False, (host, port))


@pytest.mark.parametrize("options,device,port", [
    (["-r", "/dev/real"], None, "/dev/real"),
    (["-r/dev/real"], None, "/dev/real"),
    (["--rig-file=/dev/real"], None, "/dev/real"),
    (["--rig-file", "/dev/real"], None, "/dev/real"),
    ([], "/dev/real", "/dev/real"),
    (["-r", "/dev/real"], "/dev/other", "/dev/real"),
])
def test_real_port_observation_cannot_select_listener(monkeypatch, options, device, port):
    monkeypatch.setattr(claim, "serial_nodes", lambda value: (value,))
    observation = native._observation(["rigctld", "-m1", *options], device)
    assert observation.nodes == (port,)
    assert observation.listener is None


@pytest.mark.parametrize("argv", [
    ["rigctld"], ["rigctld", "-m1036"], ["rigctl", "-m1"],
    ["rigctld", "-m1", "-r", ""], ["rigctld", "-m1", "--rig-file="],
    ["rigctld", "-m1", "-pCOM41"], ["rigctld", "-m1", "--ptt-file=COM41"],
])
def test_missing_device_is_not_guessed_to_be_portless(argv):
    with pytest.raises(claim.ClaimError, match="requires a device"):
        native._observation(argv, None)


@pytest.mark.parametrize("platform", ["posix", "windows"])
def test_portless_daemon_routes_to_supervisor(monkeypatch, platform):
    released = []
    admission = SimpleNamespace(fd=0, claim=object(), failed=lambda: released.append(True))
    monkeypatch.setattr(claim, "inherited", lambda *a, **kw: admission)
    monkeypatch.setattr(claim, "_fd", lambda fd: 0)
    monkeypatch.setattr(native, "_start", lambda *a, **kw: pytest.fail("one-shot path"))
    monkeypatch.setattr(native, "_start_windows", lambda *a, **kw: pytest.fail("one-shot path"))
    class SupervisorReached(Exception):
        pass
    def supervise(config):
        assert config["listener"] == ("127.0.0.1", 14532)
        assert config["nodes"] == () and not config["audio"]
        raise SupervisorReached
    if platform == "posix":
        def helper(verb, config, **kwargs):
            assert verb == "--supervise"
            supervise(config)
        monkeypatch.setattr(native, "_helper", helper)
    else:
        monkeypatch.setattr(native, "os", SimpleNamespace(**(vars(os) | {
            "name": "nt", "O_BINARY": 0, "O_NOINHERIT": 0})))
        monkeypatch.setattr(winclaim, "_crt", lambda handle, *a, **kw: handle)
        monkeypatch.setattr(winproc, "pipe_pair", lambda: (
            SimpleNamespace(close=lambda: True), os.open(os.devnull, os.O_WRONLY)))
        def spawn(admission, argv, **kwargs):
            assert kwargs["role"] == "supervisor" and argv[-2] == "--supervise"
            config = json.loads(argv[-1])
            config["nodes"] = tuple(config["nodes"])
            config["listener"] = tuple(config["listener"])
            assert config["port"] is None
            supervise(config)
        monkeypatch.setattr(claim, "spawn", spawn)
    with pytest.raises(SupervisorReached):
        native.spawn_native(["rigctld", "-P", "NONE", "-m", "1", "--set-conf=ptt_type=None", "-T", "127.0.0.1", "-t", "14532"],
                            claim_fd=object(), long_lived=False)
    assert released == [True]


@pytest.mark.parametrize("reply,expected", [
    ([b"No", b"ne\n"], True), ([b"ptt_type=None\n"], True),
    ([b"RIG\n"], True), ([b"ptt_type=RTS\n"], True),
    ([b"ready\n"], False), ([b"RPRT\n"], False),
    ([b"RPRT -1\n"], False), ([b""], False), ([b"None", b""], False),
    ([b"x" * 1024], False),
])
def test_listener_requires_configuration_reply(monkeypatch, reply, expected):
    calls, pending = [], list(reply)
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            calls.append("closed")
        def settimeout(self, timeout):
            assert 0 < timeout <= native.POLL_S
        def connect(self, endpoint):
            calls.append(endpoint)
        def send(self, data):
            calls.append(data[:3])
            return min(3, len(data))
        def recv(self, size):
            return pending.pop(0)
    monkeypatch.setattr(socket, "socket", lambda *a: Connection())
    assert native._listener_ready(("127.0.0.2", 14532),
        deadline=time.monotonic() + 1, cancelled=lambda: False) is expected
    assert calls[0] == ("127.0.0.2", 14532)
    assert b"".join(calls[1:-1]) == b"\\get_conf ptt_type\n"
    assert calls[-1] == "closed"


@pytest.mark.parametrize("ending", ["deadline", "interrupt", "late-reply"])
def test_listener_deadline_and_interrupt_bound_receive(monkeypatch, ending):
    now, interrupted, closed = [100.0], [False], []
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: now[0]))
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
        def settimeout(self, timeout):
            assert 0 < timeout <= native.POLL_S
        def connect(self, endpoint):
            pass
        def send(self, data):
            return len(data)
        def recv(self, size):
            if ending == "interrupt":
                interrupted[0] = True
            else:
                now[0] = 112.0
            if ending == "late-reply":
                return b"None\n"
            raise socket.timeout()
    monkeypatch.setattr(socket, "socket", lambda *a: Connection())
    if ending == "interrupt":
        with pytest.raises(native.NativeSpawnError, match="interrupted"):
            native._listener_ready(("127.0.0.1", 4532), deadline=112,
                                   cancelled=lambda: interrupted[0])
    else:
        assert not native._listener_ready(("127.0.0.1", 4532), deadline=112,
                                         cancelled=lambda: interrupted[0])
    assert closed == [True]


@pytest.mark.parametrize("platform", ["posix", "windows"])
@pytest.mark.parametrize("ending", ["ready", "deadline", "interrupt", "exit"])
def test_portless_observers_share_deadline_and_interrupt(monkeypatch, platform, ending):
    now, running, interrupted = [100.0], [True], [False]
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda t: None))
    cancelled = lambda: interrupted[0]
    child = SimpleNamespace(pid=1234, alive=lambda: running[0], _process=None)
    config = dict(argv=["rigctld", "-m1"], nodes=(), audio=False,
                  listener=("127.0.0.2", 14532))
    def probe(endpoint, *, deadline, cancelled):
        assert endpoint == config["listener"] and deadline == 112
        assert not cancelled()
        if ending == "deadline":
            now[0] = deadline
        elif ending == "interrupt":
            interrupted[0] = True
        elif ending == "exit":
            running[0] = False
        return True
    monkeypatch.setattr(native, "_listener_ready", probe)
    monkeypatch.setattr(native, "_listener_owned", lambda *a, **kw: True)
    monkeypatch.setattr(native, "_opened", lambda *a: pytest.fail("portless handle walk"))
    monkeypatch.setattr(winproc, "device_holders", lambda *a, **kw: pytest.fail("portless handle walk"))
    observe = native._observe_windows if platform == "windows" else native._observe_posix
    if ending == "ready":
        observe(child, config, cancelled)
    else:
        reason = {"deadline": "timed out|never answered", "interrupt": "interrupted",
                  "exit": "exited before acknowledgement"}[ending]
        with pytest.raises(native.NativeSpawnError, match=reason):
            observe(child, config, cancelled)


@pytest.mark.parametrize("platform", ["posix", "windows"])
def test_real_port_observers_refuse_listener_fallback(monkeypatch, platform):
    now = [100.0]
    monkeypatch.setattr(native, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda t: now.__setitem__(0, now[0] + t)))
    monkeypatch.setattr(native, "_listener_ready", lambda *a, **kw: pytest.fail("listener fallback"))
    monkeypatch.setattr(native, "_opened", lambda *a: False)
    child = SimpleNamespace(pid=1234, alive=lambda: True, _process=None, _win_entry=object())
    def walk(entries, devices, *, deadline, cancelled):
        assert deadline == 112 and not cancelled()
        return winproc.DeviceHolders({child.pid: winproc.DeviceWalk()}, "complete")
    monkeypatch.setattr(winproc, "device_holders", walk)
    config = dict(argv=["rigctld", "-m1", "-rCOM41"], nodes=("device",),
                  port="COM41", audio=False, listener=None)
    observe = native._observe_windows if platform == "windows" else native._observe_posix
    with pytest.raises(native.NativeSpawnError, match="timed out|never opened COM41"):
        observe(child, config, lambda: False)


@pytest.fixture
def dummy_rigctld_argv():
    executable = shutil.which("rigctld")
    if executable is None:
        pytest.skip("portless native regression requires rigctld")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return [executable, "-P", "NONE", "-m", "1", "--set-conf=ptt_type=None",
            "-T", "127.0.0.1", "-t", str(port)]


@pytest.mark.wall_clock
def test_portless_dummy_rigctld_acknowledged_by_listener(owned, children, dummy_rigctld_argv):
    # No long_lived flag: the native layer must recognize this daemon itself.
    child = native.spawn_native(dummy_rigctld_argv, claim_fd=owned)
    children.append(child)
    assert child.alive()
    assert native._listener_ready(("127.0.0.1", int(dummy_rigctld_argv[9])),
        deadline=time.monotonic() + 1, cancelled=lambda: False)
    assert records() == []
    assert mutex_available()


@pytest.mark.wall_clock
def test_listening_rigctld_without_configured_device_is_refused(
        owned, children, dummy_rigctld_argv, tmp_path):
    port = tmp_path / "unopened-port"
    port.touch()
    # Dummy Hamlib listens but ignores -r; configuring a port still requires its open.
    argv = [*dummy_rigctld_argv, "-r", str(port)]
    endpoint = ("127.0.0.1", int(argv[9]))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(native.spawn_native, argv, claim_fd=owned)
        try:
            wait_for(lambda: native._listener_ready(endpoint,
                deadline=time.monotonic() + .2, cancelled=lambda: False))
        finally:
            with pytest.raises(native.NativeSpawnError, match="timed out after 12 s") as failure:
                future.result(timeout=20)
            children.append(failure.value.child)
    assert not failure.value.child.alive()
    assert mutex_available()


def test_linux_descriptor_observation(tmp_path, monkeypatch):
    (tmp_path / "3").symlink_to(tmp_path / "node")
    (tmp_path / "4").symlink_to(tmp_path / "other")
    original = Path.iterdir
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "iterdir", lambda path: original(tmp_path))
    assert native._opened(12, [str(tmp_path / "node")])
    assert not native._opened(12, [str(tmp_path / "absent")])


def real_descriptor_observation(tmp_path):
    path = (tmp_path / "node").resolve()
    path.touch()
    # The child acknowledges each stage over stdout and holds the file open
    # until stdin closes. No observer, /proc, or libproc calls are mocked.
    with subprocess.Popen(command(
        "import sys; print('ready',flush=True); sys.stdin.buffer.read(1); "
        f"f=open({str(path)!r},'rb'); print('opened',flush=True); "
        "sys.stdin.buffer.read()"), stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
        try:
            assert __import__('select').select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline() == b"ready\n"
            assert not native._opened(child.pid, [str(path)])
            child.stdin.write(b"x")
            child.stdin.flush()
            assert __import__('select').select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline() == b"opened\n"
            assert native._opened(child.pid, [str(path)])
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc descriptor observation requires Linux")
def test_linux_real_descriptor_observation(tmp_path):
    real_descriptor_observation(tmp_path)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc descriptor observation requires macOS")
def test_darwin_real_descriptor_observation(tmp_path):
    real_descriptor_observation(tmp_path)


@pytest.mark.parametrize("timeout", [None, 0, -1, 31, float("inf"), float("nan")])
def test_shell_requires_bounded_one_shot_timeout(owned, timeout):
    args = SimpleNamespace(timeout=timeout, argv=command("pass"), fd=owned.fd,
                           cleanup=None, long_lived=False, device=None)
    with pytest.raises(claim.ClaimError, match="timeout"):
        native.shell_spawn(args)


@pytest.mark.parametrize("verb", ["spawn", "claim_spawn"])
def test_shell_one_shot_exit_and_output(owned, verb, capfd):
    result = claim.main([verb, "--fd", str(owned.fd), "--timeout", "3", "--",
                         *command("import sys; print('native output'); sys.exit(3)")])
    assert result == 3
    assert "native output" in capfd.readouterr().out
    assert mutex_available()


def test_shell_timeout_escalates_and_verifies(owned):
    result = claim.main(["spawn", "--fd", str(owned.fd), "--timeout", ".3", "--",
                         *command("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)")])
    assert result == 124
    assert mutex_available()


def test_one_shot_caller_death_before_open(owned, children, tmp_path):
    pidfile = tmp_path / "child-pid"
    release = tmp_path / "release"
    opened = tmp_path / "opened"
    argv = command(
        f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid()))\n"
        f"while not os.path.exists({str(release)!r}): time.sleep(.01)\n"
        f"f=open({str(opened)!r},'w'); time.sleep(20)")
    launcher = subprocess.Popen(command(
        "import os,time; from hfmodem.core.native import spawn_native; "
        f"spawn_native({argv!r},claim_fd={owned.fd}); time.sleep(20)"),
        pass_fds=(owned.fd,))
    try:
        wait_for(lambda: pidfile.exists() and pidfile.read_text())
        pid = int(pidfile.read_text())
        child = native.NativeChild(pid, claim.process_started(pid), None, None, None)
        children.append(child)
        launcher.kill()
        launcher.wait(timeout=5)
        assert not mutex_available()
        assert not opened.exists()
        release.touch()
        wait_for(opened.exists)
        assert child.alive()
        assert not mutex_available()
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=5)


@pytest.mark.parametrize("point, locked", [
    ("native_before_child_publication", True),
    ("native_after_acknowledgement", False),
])
def test_supervisor_killed_at_publication_and_after_ack(owned, children, tmp_path, monkeypatch, point, locked):
    supervise_observer(monkeypatch, timeout=5)
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    (barriers / point).touch()
    log = tmp_path / "events"
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(barriers))
    monkeypatch.setenv("HFMODEM_COEXIST_LOG", str(log))
    code = "import time; time.sleep(20)" if locked else (
        "import os,time; os.write(1,b'audio'); time.sleep(20)")
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native,
            audio_command(tmp_path, code), claim_fd=owned)
        wait_for((barriers / (point + ".reached")).exists)
        events = [json.loads(line) for line in log.read_text().splitlines()]
        spawned = next(event for event in events if event["event"] == "native_spawn")
        child = native.NativeChild(spawned["child_pid"], spawned["child_started"], None, None, None)
        children.append(child)
        if locked:
            assert "child_pid" not in json.loads(records()[0].read_text())
        else:
            assert records() == []
        os.kill(spawned["pid"], signal.SIGKILL)
        with pytest.raises(native.NativeSpawnError, match="without acknowledgement"):
            future.result(timeout=8)
    assert child.alive()
    assert mutex_available() is not locked


def test_failed_termination_never_unlocks_or_removes_diagnostics(owned, monkeypatch):
    retained = []
    child = SimpleNamespace(pid=12345, started="fake-12345", alive=lambda: True)
    def start(argv, claim_fd, hold_fd, cleanup, streams):
        retained.append(os.dup(hold_fd))
        return child
    monkeypatch.setattr(native, "_start", start)
    monkeypatch.setattr(native, "_opened", lambda *a: False)
    monkeypatch.setattr(native, "_end", lambda child: False)
    monkeypatch.setattr(native, "OPEN_TIMEOUT_S", 0)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    read, write = os.pipe()
    try:
        config = dict(claim=owned.fd, argv=["fake-native"], streams=(None,) * 3,
                      status=write, nodes=["/fake"], audio=False)
        assert native._supervise(config) == 1
        assert not mutex_available()
        assert len(records()) == 1
        result = json.loads(os.read(read, 16384))
        assert "termination unresolved" in result["error"]
        assert result["pid"] == child.pid
    finally:
        os.close(read)
        os.close(write)
        for fd in retained:
            os.close(fd)
    assert mutex_available()


def test_shell_long_lived_stdio_and_bookkeeping(owned, children, tmp_path, monkeypatch, capfd):
    supervise_observer(monkeypatch, observe=True)
    node = tmp_path / "fake-cat"
    result = claim.main(["spawn", "--fd", str(owned.fd), "--long-lived", "--device", str(node), "--",
                        *command(f"import os,time; f=open({str(node)!r},'w'); "
                                 "os.write(2,b'native stderr\\n'); time.sleep(20)")])
    captured = capfd.readouterr()
    line = next(line for line in captured.out.splitlines() if line.startswith("CLAIM NATIVE "))
    _, _, pid, started = line.split()
    child = native.NativeChild(int(pid), started, None, None, None)
    children.append(child)
    assert result == 0
    assert "native stderr" in captured.err
    assert child.alive()
    assert mutex_available()


def test_native_fd_inventory_excludes_launchers_mutex(owned, children, tmp_path):
    launch = claim.inherited(owned, launch=True)
    os.set_inheritable(launch.fd, True)
    sentinel = tmp_path / "sentinel"
    fd = os.open(sentinel, os.O_RDWR | os.O_CREAT, 0o600)
    os.set_inheritable(fd, True)
    probe = (
        "import os,json,fcntl,sys; rows=[]\n"
        "for fd in range(3,256):\n"
        " try:\n"
        "  st=os.fstat(fd); rows.append([fd,st.st_dev,st.st_ino,os.lseek(fd,0,os.SEEK_CUR)])\n"
        " except OSError: pass\n"
        "print(json.dumps(rows),flush=True); sys.stdin.read(1)"
    )
    try:
        child = native.spawn_native(command(probe), claim_fd=owned)
        children.append(child)
        rows = json.loads(child.stdout.readline())
        inode = claim.mutex_path().stat()
        mutex_rows = [r for r in rows if r[1:3] == [inode.st_dev, inode.st_ino]]
        assert len(mutex_rows) == 1
        assert mutex_rows[0][3] == 0
        assert not any(r[2] == sentinel.stat().st_ino and r[1] == sentinel.stat().st_dev for r in rows)
        launch.failed()
        assert not mutex_available()
    finally:
        launch.failed()
        os.close(fd)


def test_supervisor_killed_during_failed_termination(owned, children, tmp_path, monkeypatch):
    supervise_observer(monkeypatch, timeout=.3)
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    (barriers / "native_after_term").touch()
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(barriers))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(native.spawn_native,
            audio_command(tmp_path, "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"),
            claim_fd=owned)
        wait_for((barriers / "native_after_term.reached").exists)
        path = records()[0]
        child = record_child(path)
        children.append(child)
        os.kill(json.loads(path.read_text())["pid"], signal.SIGKILL)
        with pytest.raises(native.NativeSpawnError, match="without acknowledgement"):
            future.result(timeout=8)
    (barriers / "native_after_term.release").touch()
    assert child.alive()
    assert not mutex_available()
    assert path.exists()


def test_silent_capture_hits_twelve_second_open_bound(owned, tmp_path):
    before = time.monotonic()
    with pytest.raises(native.NativeSpawnError, match="timed out after 12 s") as exc:
        native.spawn_native(audio_command(tmp_path, "import time; time.sleep(25)"), claim_fd=owned)
    try:
        assert 12 <= time.monotonic() - before < 22
        assert not exc.value.child.alive()
        assert mutex_available()
        assert records() == []
    finally:
        for pipe in (exc.value.child.stdin, exc.value.child.stdout, exc.value.child.stderr):
            pipe.close()


@pytest.mark.parametrize("entry", ["spawn_native", "_spawn_native", "shell_spawn"])
def test_windows_shell_and_cleanup_refuse_before_open_or_spawn(monkeypatch, entry):
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(native, "_start", lambda *a, **k: pytest.fail("native spawn"))
    monkeypatch.setattr(claim, "inherited", lambda *a, **k: pytest.fail("claim admission"))
    monkeypatch.setattr(claim, "verify_cleanup", lambda *a, **k: pytest.fail("cleanup admission"))
    with pytest.raises(claim.ClaimError, match="native launch from the shell is not supported on Windows"):
        if entry == "shell_spawn":
            native.shell_spawn(None)
        else:
            getattr(native, entry)([], claim_fd=-1, cleanup_fd=8)


@pytest.mark.parametrize("reply", [None, {"error": "original launch failure"}, {}])
def test_windows_supervisor_failure_uses_pinned_entry(monkeypatch, reply):
    monkeypatch.setattr(native, "os", SimpleNamespace(**(vars(os) | {
        "name": "nt", "O_BINARY": 0, "O_NOINHERIT": 0})))
    monkeypatch.setattr(native, "_observation", lambda *args: ((), None, True))
    released, stopped, closed = [], [], []
    admission = SimpleNamespace(claim=object(), failed=lambda: released.append(True))
    entry = winproc.Pinned(101, 1234, "win:1", "python.exe")
    process = SimpleNamespace(_handle=100, poll=lambda: None)
    monkeypatch.setattr(claim, "inherited", lambda *a, **k: admission)
    monkeypatch.setattr(claim, "spawn", lambda *a, **k: claim.Launch(process, entry))
    monkeypatch.setattr(winclaim, "_crt", lambda handle, *a, **k: handle)
    monkeypatch.setattr(winclaim, "duplicate", lambda handle: handle + 100)
    monkeypatch.setattr(winclaim, "close_handle", closed.append)
    monkeypatch.setattr(winproc, "_wait", lambda handle, timeout: timeout > 0)
    updated = []
    monkeypatch.setattr(winproc, "_update_launch", lambda entry, phase: updated.append(phase))
    monkeypatch.setattr(winproc, "_pin", lambda *a: pytest.fail("re-pinned the supervisor"))
    def denied(pid, rights):
        assert pid == 4321
        raise winproc.ProcessAccessError(pid, rights)
    monkeypatch.setattr(winproc, "open", denied)
    if reply is None:
        response = winproc.IOResult("eof")
    else:
        response = winproc.IOResult("data", (json.dumps(dict(reply, pid=4321, started="win:2"))
                                             + "\n").encode())
    reader = SimpleNamespace(read=lambda *a, **k: response, close=lambda: True)
    monkeypatch.setattr(winproc, "pipe_pair", lambda: (reader, os.open(os.devnull, os.O_WRONLY)))
    monkeypatch.setattr(native, "_end", lambda child: stopped.append(child) or True)
    try:
        with pytest.raises(native.NativeSpawnError) as failure:
            native._spawn_windows(["capture"], claim_fd=object(), cleanup_fd=None,
                                  long_lived=True, device=None, streams=None, own_session=False)
        message = str(failure.value)
        if reply is None:
            assert message == "native supervisor exited without acknowledgement"
        else:
            assert "cannot identify native child pid 4321" in message
            assert "access denied" in message
            if "error" in reply:
                assert message.startswith(reply["error"] + "; ")
        supervisor, = stopped
        assert supervisor._process is process
        assert supervisor._win_entry.identity == (1234, "win:1")
        assert supervisor._win_entry.handle == 201
        assert entry.handle is None
        assert updated == ["exited"]
        assert closed == [101]
        assert released == [True]
    finally:
        for child in stopped:
            child._win_entry.close()


@pytest.mark.parametrize("gone", [False, True])
@pytest.mark.parametrize("reply_error", [False, True])
@pytest.mark.parametrize("stop_result", [False, True])
def test_windows_supervisor_waits_before_releasing_admission(monkeypatch, gone, reply_error,
                                                           stop_result):
    monkeypatch.setattr(native, "os", SimpleNamespace(**(vars(os) | {
        "name": "nt", "O_BINARY": 0, "O_NOINHERIT": 0})))
    monkeypatch.setattr(native, "_observation", lambda *args: ((), None, True))
    events, children = [], []
    admission = SimpleNamespace(claim=object(), failed=lambda: events.append("released"))
    def wait(timeout=0):
        events.append(("wait", timeout))
        return gone if timeout else False
    entry = SimpleNamespace(pid=1234, started="win:1", wait=wait,
                            close=lambda: events.append("closed"))
    process = SimpleNamespace(poll=lambda: events.append("polled"))
    monkeypatch.setattr(claim, "inherited", lambda *a, **k: admission)
    monkeypatch.setattr(claim, "spawn", lambda *a, **k: claim.Launch(process, entry))
    monkeypatch.setattr(winclaim, "_crt", lambda handle, *a, **k: handle)
    monkeypatch.setattr(winproc, "_update_launch",
                        lambda child, phase: events.append((child, phase)))
    reply = dict(pid=4321, started="win:2")
    if reply_error:
        reply["error"] = "original launch failure"
    reader = SimpleNamespace(
        read=lambda *a, **k: winproc.IOResult("data", (json.dumps(reply) + "\n").encode()),
        close=lambda: True)
    monkeypatch.setattr(winproc, "pipe_pair", lambda: (reader, os.open(os.devnull, os.O_WRONLY)))
    def child(pid, started, stdin, stdout, stderr, *args, **kwargs):
        result = SimpleNamespace(pid=pid, started=started, stdin=stdin, stdout=stdout, stderr=stderr)
        children.append(result)
        return result
    monkeypatch.setattr(native, "NativeChild", child)
    monkeypatch.setattr(native, "_end", lambda child: events.append("ended") or stop_result)
    try:
        if gone and not reply_error:
            result = native._spawn_windows(["capture"], claim_fd=object(), cleanup_fd=None,
                long_lived=True, device=None, streams=None, own_session=False)
            assert result is children[0]
        else:
            with pytest.raises(native.NativeSpawnError) as failure:
                native._spawn_windows(["capture"], claim_fd=object(), cleanup_fd=None,
                    long_lived=True, device=None, streams=None, own_session=False)
            assert failure.value.child is children[0]
            if reply_error:
                assert str(failure.value).startswith("original launch failure")
                if gone:
                    assert str(failure.value) == "original launch failure"
            if not gone:
                assert "supervisor did not exit within 5s; termination unresolved" in str(failure.value)
        expected = ([("wait", 0), "ended"] if reply_error else [])
        expected += [("wait", native.KILL_WAIT_S)]
        if gone:
            expected += ["polled", (entry, "exited")]
        expected += ["closed", "released"]
        assert events == expected
    finally:
        for result in children:
            for stream in (result.stdin, result.stdout, result.stderr):
                if stream is not None:
                    stream.close()


def test_windows_pinned_child_duplicate_failure_keeps_borrowed_entry(monkeypatch):
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt"))
    entry = winproc.Pinned(101, 1234, "win:1", "python.exe")
    def fail(handle):
        raise claim.ClaimError("duplicate failed")
    monkeypatch.setattr(winclaim, "duplicate", fail)
    monkeypatch.setattr(winclaim, "close_handle", lambda *a: pytest.fail("closed borrowed entry"))
    with pytest.raises(claim.ClaimError, match="duplicate failed"):
        native.NativeChild(entry.pid, entry.started, None, None, None, _win_entry=entry)
    assert entry.handle == 101


@pytest.mark.parametrize("outcome", ["unavailable", "message", "walk", "found", "raised", "audio"])
def test_windows_observation_exit_during_inspection_precedes_walk_result(monkeypatch, outcome):
    running = [True]
    child = SimpleNamespace(pid=1234, _win_entry=object(), alive=lambda: running[0],
                            _process=SimpleNamespace(poll=lambda: 23))
    def inspect(*args, **kwargs):
        running[0] = False
        if outcome in ("raised", "audio"):
            raise claim.ClaimError("inspection failed")
        if outcome == "unavailable":
            return winproc.DeviceHolders({}, "unavailable", "inspection unavailable")
        if outcome == "message":
            return winproc.DeviceHolders({}, "incomplete", "inspection failed")
        walk = (winproc.DeviceWalk(message="cannot inspect pid 1234 (rigctl.exe)")
                if outcome == "walk" else winproc.DeviceWalk(frozenset({"COM41"})))
        return winproc.DeviceHolders({child.pid: walk}, "incomplete")
    monkeypatch.setattr(winproc, "device_holders", inspect)
    monkeypatch.setattr(native, "_output_ready", inspect)
    config = dict(argv=["rigctl.exe"], audio=outcome == "audio", port="COM41",
                  nodes=[r"\Device\test"], output_fd=None, output=None)
    with pytest.raises(native.NativeSpawnError,
                       match=r"exited before acknowledgement \(exit code 23\)") as failure:
        native._observe_windows(child, config, lambda: False)
    assert failure.value.child is child
