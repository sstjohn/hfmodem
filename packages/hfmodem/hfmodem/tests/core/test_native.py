# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from concurrent.futures import ThreadPoolExecutor
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hfmodem.core import claim, native


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch):
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
    with pytest.raises(native.NativeSpawnError, match="No such file") as exc:
        native.spawn_native(["/nonexistent-native-unit-command"], claim_fd=owned)
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
    return [str(script), "-f", backend, "-i", "fake", str(output)]


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


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
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
    nodes, output, audio = native._observation(["native", "-r", "/dev/cu.fake"], None)
    assert nodes == ("/dev/cu.fake", "/dev/tty.fake")
    monkeypatch.setattr(native, "_darwin_paths", lambda pid: ("/dev/tty.fake",))
    assert native._opened(12, nodes)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(claim, "audio_nodes", lambda endpoint, direction: ("/dev/snd/pcmC2D0c",))
    assert native._observation(["ffmpeg", "-i", "hw:2", "pipe:1"], "hw:2") == (
        ("/dev/snd/pcmC2D0c",), None, True)


@pytest.mark.parametrize("blank", ["", " "])
def test_blank_long_lived_device_is_no_device(monkeypatch, blank):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert native._observation(["rigctld", "-r", "/dev/cu.fake"], blank)[0] == (
        "/dev/cu.fake", "/dev/tty.fake")
    for argv in (["rigctld", "-r", blank, "-s", "38400"], ["rigctld", f"--rig-file={blank}"]):
        with pytest.raises(claim.ClaimError, match="requires a device"):
            native._observation(argv, blank)


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
