# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
import fcntl
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hfmodem.core import claim, coexist, rxreadiness

_audio_nodes = claim.audio_nodes
_real_lsof = claim._lsof


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch):
    real_path = os.environ["PATH"]
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_STATE_DIR", str(tmp_path / "state"))
    for key in ("HFMODEM_COEXIST_FAULT", "HFMODEM_COEXIST_LOG", "HFMODEM_COEXIST_BARRIERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(claim, "audio_nodes", lambda *a, **k: ())
    monkeypatch.setattr(claim, "_lsof", lambda paths: "")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    ps = fakebin / "ps"
    ps.write_text(f"#!{sys.executable}\nimport sys\n"
                  "print('Sat Oct 17 12:00:00 2026' if 'lstart=' in sys.argv else 'fake-python')\n")
    ps.chmod(0o700)
    monkeypatch.setenv("PATH", str(fakebin) + os.pathsep + os.environ["PATH"])
    (fakebin / "sounddevice.py").write_text(
        "default = type('Defaults', (), {'device': (0, 1)})()\n"
        "def query_devices(*args):\n    return {'name': 'fake server'}\n")
    monkeypatch.setenv("PYTHONPATH", str(fakebin) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return real_path


def acquire(**kwargs):
    return claim.acquire("test", "unit", **kwargs)


def assert_free():
    descriptor = acquire()
    claim.release(descriptor)


def real_ps_output(monkeypatch, real_path):
    monkeypatch.setenv("PATH", real_path)
    if shutil.which("ps") is None:
        pytest.skip("real ps unavailable: executable not found")
    try:
        result = subprocess.run(["ps", "-p", str(os.getpid()), "-o", "lstart="],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"real ps unavailable: {exc}")
    if result.returncode or not result.stdout.strip():
        pytest.skip(f"real ps unavailable: exit {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def test_process_started_real_ps(monkeypatch, namespace):
    output = real_ps_output(monkeypatch, namespace)
    assert claim.process_started(os.getpid()) == "_".join(output.split())


@pytest.mark.parametrize("flag", ["password", "passwd", "secret", "token",
                                 "mail-password", "mail-passwd", "mail-secret", "mail-token"])
@pytest.mark.parametrize("separator", ["=", " ", "\t"])
def test_holder_command_preserves_arguments_and_redacts_secrets(monkeypatch, flag, separator):
    run = subprocess.run
    second = "second"
    def fake_run(argv, **kwargs):
        if argv[-1] == "lstart=":
            return run(argv, **kwargs)
        assert argv == ["ps", "-p", str(os.getpid()), "-o", "command="]
        return SimpleNamespace(returncode=0, stdout=(
            f"python -m hfmodem station --{flag}{separator}sensitive "
            f"--token={second} --operation monitor\n"))
    monkeypatch.setattr(subprocess, "run", fake_run)
    descriptor = acquire()
    try:
        expected_separator = "=" if separator == "=" else " "
        assert claim.holder(descriptor).command == (
            f"python -m hfmodem station --{flag}{expected_separator}<redacted> "
            "--token=<redacted> --operation monitor")
        assert b"sensitive" not in descriptor.pread() and second.encode() not in descriptor.pread()
    finally:
        claim.release(descriptor)


@pytest.mark.parametrize("status", [0, 1, 2])
@pytest.mark.parametrize("stderr", ["", "lsof: permission denied"])
def test_lsof_options_timeout_and_errors(monkeypatch, tmp_path, status, stderr):
    path = tmp_path / "claim"
    path.touch()

    def run(argv, **kwargs):
        assert argv == ["lsof", "-w", "+c", "0", "-nP", "-o", "-Fpcfo", "--", str(path)]
        assert kwargs == dict(capture_output=True, text=True, timeout=20)
        return SimpleNamespace(returncode=status, stdout="inventory", stderr=stderr)
    monkeypatch.setattr(subprocess, "run", run)
    if status in (0, 1) and not stderr:
        assert _real_lsof([path]) == "inventory"
    else:
        with pytest.raises(claim.ClaimError, match="discovery failed"):
            _real_lsof([path])


def test_holders_real_lsof_child_offset_match(monkeypatch, namespace):
    real_ps_output(monkeypatch, namespace)
    if shutil.which("lsof") is None:
        pytest.skip("real lsof unavailable: executable not found")
    descriptor = acquire()
    try:
        with subprocess.Popen([sys.executable, "-c", """
import os, sys
from hfmodem.core import claim
descriptor = claim.inherited(int(sys.argv[1]))
reopened = os.open(claim.claim_path(), os.O_RDWR)
print(reopened, flush=True)
sys.stdin.read()
os.close(reopened)
claim.close(descriptor)
""", str(descriptor.fd)], pass_fds=(descriptor.fd,), stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            try:
                assert select.select([process.stdout], [], [], 10)[0], "child readiness timed out"
                ready = process.stdout.readline()
                assert ready.strip().isdigit(), "child failed before opening claim"
                reopened = int(ready)
                try:
                    inventory = _real_lsof([claim.claim_path()])
                except claim.ClaimError as exc:
                    pytest.skip(f"real lsof unavailable: {exc}")
                if not inventory.strip():
                    pytest.skip("real lsof unavailable: cannot observe known open claim (sandbox restriction)")
                rows = claim.holders(descriptor, lsof=_real_lsof)
                child_rows = [row for row in rows if row.pid == process.pid]
                assert [(row.fd, row.offset) for row in child_rows] == [
                    (descriptor.fd, claim.holder(descriptor).nonce)]
                assert all(row.fd != reopened for row in child_rows)
            finally:
                try:
                    _, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    raise
            assert process.returncode == 0, stderr
    finally:
        claim.release(descriptor)


def test_operator_namespace_comes_from_the_password_database(monkeypatch, tmp_path):
    for key in ("HFMODEM_COEXIST_HARNESS", "HFMODEM_STATE_DIR", "PYTEST_CURRENT_TEST", "HERMETIC_WORK"):
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path / "wrong-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "wrong-xdg"))
    monkeypatch.setattr(claim.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path)))
    assert claim.claim_path() == tmp_path / ".local/state/hfmodem/rig.claim"
    paths = claim.bootstrap()
    assert all(p.is_file() for p in paths)
    inodes = [p.stat().st_ino for p in paths]
    assert claim.bootstrap() == paths
    assert [p.stat().st_ino for p in paths] == inodes


@pytest.mark.parametrize("kept", ["HFMODEM_COEXIST_HARNESS", "HFMODEM_STATE_DIR", "PYTEST_CURRENT_TEST"])
def test_one_test_marker_alone_refuses_the_operator_namespace(monkeypatch, tmp_path, kept):
    for key in {"HFMODEM_COEXIST_HARNESS", "HFMODEM_STATE_DIR", "PYTEST_CURRENT_TEST"} - {kept}:
        monkeypatch.delenv(key)
    monkeypatch.setattr(claim.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path)))
    for resolve in (claim.state_dir, claim.claim_path, claim.bootstrap):
        with pytest.raises(claim.ClaimError, match="refusing the operator's state directory"):
            resolve()
    with pytest.raises(claim.ClaimError, match="refusing"):
        claim.acquire("test", "guard", devices=claim.Devices(cat="none"))
    assert not (tmp_path / ".local").exists()


@pytest.mark.parametrize("nonce", [1, claim.MAX_NONCE])
def test_holder_line_published_under_lock(monkeypatch, nonce):
    monkeypatch.setattr(claim.secrets, "randbelow", lambda maximum: nonce - 1)
    seen = []
    def checkpoint(point):
        if point not in ("before_holder_line", "after_holder_line"):
            return
        fd = os.open(claim.claim_path(), os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen.append(os.pread(fd, 16384, 0))
        finally:
            os.close(fd)
    monkeypatch.setattr(coexist, "checkpoint", checkpoint)
    descriptor = acquire()
    try:
        assert seen[0] == b""
        published = claim.holder(descriptor)
        assert published.pid == os.getpid()
        assert published.nonce == descriptor.offset() == nonce
        assert published.kind == "test" and published.operation == "unit"
        assert descriptor.fstat().st_size == len(seen[1]) < nonce + 16384
        with pytest.raises(claim.ClaimBusy, match="rig claimed"):
            acquire()
        assert descriptor.pread() == seen[1]
    finally:
        claim.release(descriptor)


@pytest.mark.parametrize("point", ["after_mutex", "after_claim_flock", "before_holder_line",
                                   "after_holder_line", "after_device_check", "during_line_proof"])
def test_acquisition_failure_unwinds(point, monkeypatch):
    monkeypatch.setenv("HFMODEM_COEXIST_FAULT", point)
    with pytest.raises(RuntimeError, match=point):
        acquire(prove_line=True, ptt_factory=lambda: FakePtt())
    if claim.claim_path().exists():
        assert claim.claim_path().read_bytes() == b""
    monkeypatch.delenv("HFMODEM_COEXIST_FAULT")
    assert_free()


@pytest.mark.parametrize("failure", ["raise", "wrong_return", "short_write"])
def test_nonce_seek_and_publication_failure_unwind(monkeypatch, failure):
    with monkeypatch.context() as patch:
        original = os.lseek
        def seek(fd, offset, how):
            if how == os.SEEK_SET:
                if failure == "raise":
                    raise OSError("seek refused")
                return offset - 1
            return original(fd, offset, how)
        if failure == "short_write":
            patch.setattr(os, "pwrite", lambda *args: 1)
        else:
            patch.setattr(os, "lseek", seek)
        with pytest.raises((OSError, claim.ClaimError)):
            acquire()
    assert claim.claim_path().read_bytes() == b""
    assert_free()


def test_inherited_descriptor_proven_without_locking(monkeypatch):
    descriptor = acquire()
    duplicate = claim.ClaimDescriptor(os.dup(descriptor.fd))
    reopened = os.open(claim.claim_path(), os.O_RDWR)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(fcntl, "flock", lambda *args: pytest.fail("inheritance attempted flock"))
            assert claim.inherited(duplicate).fd == duplicate.fd
            with pytest.raises(claim.ClaimError, match="nonce"):
                claim.inherited(reopened)
        claim.close(duplicate)
        assert claim.inherited(descriptor).offset() == claim.holder(descriptor).nonce
        claim.release(descriptor)
        with pytest.raises(claim.ClaimError, match="holder unknown"):
            claim.inherited(reopened)
    finally:
        os.close(reopened)


@pytest.mark.parametrize("fd", [None, -1, 0, 1, 2, True, "9", 999999])
def test_missing_or_forged_claim_refused(fd, monkeypatch):
    monkeypatch.setenv("RIG_CLAIMED", "yes")
    with pytest.raises(claim.ClaimError):
        claim.inherited(fd)


@pytest.mark.parametrize("data", [b"", b"123 start kind op 4 cmd", b"123 start kind op 0 cmd\n",
                                   b"123 start kind op 1099511627777 cmd\n", b"partial\n",
                                   b"1 start kind op 4 cmd\nextra\n", b"\xff\n"])
def test_free_stale_or_partial_line_never_inherited(data):
    claim.bootstrap()
    claim.claim_path().write_bytes(data)
    fd = os.open(claim.claim_path(), os.O_RDWR)
    try:
        with pytest.raises(claim.ClaimError):
            claim.inherited(fd)
    finally:
        os.close(fd)
    assert_free()


def test_partial_line_retried(monkeypatch):
    descriptor = acquire()
    original = os.pread
    calls = []
    def pread(fd, size, offset):
        calls.append(fd)
        return b"partial" if len(calls) == 1 else original(fd, size, offset)
    with monkeypatch.context() as patch:
        patch.setattr(os, "pread", pread)
        assert claim.inherited(descriptor)
        assert len(calls) == 2
    claim.release(descriptor)


def test_claim_inode_replaced_between_open_and_lock(monkeypatch):
    def replace(point):
        if point == "before_claim_lock":
            claim.claim_path().unlink()
            claim.claim_path().touch()
    with monkeypatch.context() as patch:
        patch.setattr(coexist, "checkpoint", replace)
        with pytest.raises(claim.ClaimError, match="does not name"):
            acquire()
    assert_free()


def test_claim_inode_replaced_after_acquisition():
    descriptor = acquire()
    claim.claim_path().unlink()
    claim.claim_path().touch()
    with pytest.raises(claim.ClaimError, match="does not name"):
        claim.inherited(descriptor)
    successor = acquire()
    successor_line = successor.pread()
    claim.release(descriptor)
    assert successor.pread() == successor_line
    claim.release(successor)


def test_missing_claim_file_refuses_a_stray_descriptor(tmp_path):
    stray = os.open(tmp_path / "stray", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        for launch in (False, True):
            with pytest.raises(claim.ClaimError, match="does not name"):
                claim.inherited(stray, launch=launch)
    finally:
        os.close(stray)
    descriptor = acquire()
    claim.claim_path().unlink()
    with pytest.raises(claim.ClaimError, match="does not name"):
        claim.inherited(descriptor)
    os.close(descriptor.fd)


def test_wrong_uid_and_nonregular_refused(monkeypatch, tmp_path):
    descriptor = acquire()
    actual = descriptor.fstat()
    with monkeypatch.context() as patch:
        patch.setattr(os, "fstat", lambda fd: SimpleNamespace(
            st_mode=actual.st_mode, st_uid=actual.st_uid + 1,
            st_dev=actual.st_dev, st_ino=actual.st_ino))
        with pytest.raises(claim.ClaimError):
            claim.inherited(descriptor)
    directory = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(claim.ClaimError):
            claim.inherited(directory)
    finally:
        os.close(directory)
    claim.release(descriptor)


def test_parent_release_invalidates_retained_reference():
    descriptor = acquire()
    duplicate = os.dup(descriptor.fd)
    claim.release(descriptor)
    try:
        with pytest.raises(claim.ClaimError, match="holder unknown"):
            claim.inherited(duplicate)
        with pytest.raises(claim.ClaimBusy):
            acquire()
    finally:
        claim.close(duplicate)
    assert_free()


def child(code, *args, pass_fds=()):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)],
                          pass_fds=pass_fds, capture_output=True, text=True, timeout=10)


def test_child_cannot_release_and_close_preserves_parent():
    descriptor = acquire()
    result = child("""
import sys
from hfmodem.core import claim
fd = int(sys.argv[1])
claim.inherited(fd)
try:
    claim.release(fd)
except claim.ClaimError:
    claim.close(fd)
else:
    raise AssertionError('child released parent')
""", descriptor.fd, pass_fds=(descriptor.fd,))
    assert result.returncode == 0, result.stderr
    assert claim.inherited(descriptor)
    claim.release(descriptor)


def test_dead_root_inheritance_and_reopened_race(tmp_path):
    claim.bootstrap()
    script = """
import os, sys
from hfmodem.core import claim, coexist
claim.audio_nodes = lambda *a, **k: ()
root = claim.acquire('test', 'dead-root')
pid = os.fork()
if pid:
    os._exit(0)
while os.getppid() == claim.holder(root).pid:
    __import__('time').sleep(.01)
claim.inherited(root)
fd = os.open(claim.claim_path(), os.O_RDWR)
try:
    claim.inherited(fd)
except claim.ClaimError:
    pass
else:
    raise AssertionError('reopened descriptor admitted')
os.close(fd)
claim.close(root)
print('dead-root verified', flush=True)
"""
    result = child(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "dead-root verified\n"
    assert_free()


@pytest.mark.parametrize("fate", ["close", "release", "failed_acquisition"])
def test_reopened_descriptor_refuses_if_holder_changes_during_validation(monkeypatch, fate):
    descriptor = acquire()
    reopened = os.open(claim.claim_path(), os.O_RDWR)
    def checkpoint(point):
        if point == "inherited_after_inode":
            if fate == "release":
                claim.release(descriptor)
            else:
                if fate == "failed_acquisition":
                    os.ftruncate(descriptor.fd, 0)
                claim.close(descriptor)
    with monkeypatch.context() as patch:
        patch.setattr(coexist, "checkpoint", checkpoint)
        with pytest.raises(claim.ClaimError):
            claim.inherited(reopened)
    os.close(reopened)
    assert_free()


def test_cleanup_acquires_under_own_mutex():
    mutex = claim.mutex(timeout=0)
    try:
        line = os.pread(mutex, 16384, 0)
        assert claim.verify_cleanup(mutex) == mutex
        descriptor = claim.acquire("cleanup", "down", cleanup=mutex, prove_line=True,
                                   ptt_factory=FakePtt)
        assert os.pread(mutex, 16384, 0) == line
        claim.release(descriptor)
        with pytest.raises(claim.ClaimBusy, match="cleanup"):
            acquire()
        fresh = os.open(claim.mutex_path(), os.O_RDWR)
        try:
            with pytest.raises(claim.ClaimError, match="another description"):
                claim.verify_cleanup(fresh)
        finally:
            os.close(fresh)
    finally:
        claim.mutex_release(mutex)
    assert_free()


def test_mutex_release_truncates_under_lock_and_closes(monkeypatch):
    mutex = claim.mutex(timeout=0)
    duplicate = os.dup(mutex)
    nonce = claim.holder(mutex, cleanup=True).nonce
    events = []
    def event(name, fd, value):
        assert name == "cleanup_released" and value == nonce
        assert os.pread(fd, 16384, 0) == b""
        probe = os.open(claim.mutex_path(), os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)
        events.append(name)
    with monkeypatch.context() as patch:
        patch.setattr(claim, "_event", event)
        claim.mutex_release(mutex)
    try:
        assert events == ["cleanup_released"]
        with pytest.raises(OSError):
            os.fstat(mutex)
        with pytest.raises(claim.ClaimError, match="holder unknown"):
            claim.holder(duplicate, cleanup=True, attempts=1)
        with pytest.raises(claim.ClaimBusy):
            acquire()
    finally:
        os.close(duplicate)
    shared = claim._shared()
    try:
        assert os.pread(shared, 16384, 0) == b""
        with pytest.raises(claim.ClaimBusy, match="admission in progress"):
            claim.mutex(timeout=0, **discovery("p42\ncadmission\nf7\no0t0\n"))
    finally:
        os.close(shared)
    assert_free()


def test_mutex_release_refuses_nonowner_and_reopened_descriptor(monkeypatch):
    mutex = claim.mutex(timeout=0)
    line = os.pread(mutex, 16384, 0)
    reopened = os.open(claim.mutex_path(), os.O_RDWR)
    try:
        with pytest.raises(claim.ClaimError, match="another description"):
            claim.mutex_release(reopened)
        with pytest.raises(claim.ClaimError, match="only the acquirer"):
            claim.mutex_release(mutex, pid=os.getpid() + 1)
        with monkeypatch.context() as patch:
            patch.setattr(claim, "process_started", lambda pid: "reused-pid")
            with pytest.raises(claim.ClaimError, match="only the acquirer"):
                claim.mutex_release(mutex)
        result = child("""
import sys
from hfmodem.core import claim
try:
    claim.mutex_release(int(sys.argv[1]))
except claim.ClaimError:
    pass
else:
    raise AssertionError('child released cleanup mutex')
""", mutex, pass_fds=(mutex,))
        assert result.returncode == 0, result.stderr
        assert os.pread(mutex, 16384, 0) == line
        assert claim.verify_cleanup(mutex) == mutex
    finally:
        os.close(reopened)
        claim.mutex_release(mutex)


def test_cleanup_requires_existing_exclusive_authority():
    claim.bootstrap()
    mutex = os.open(claim.mutex_path(), os.O_RDWR)
    try:
        for shared in (False, True):
            if shared:
                fcntl.flock(mutex, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with pytest.raises(claim.ClaimError, match="not held exclusively"):
                claim.acquire("cleanup", "down", cleanup=mutex)
    finally:
        os.close(mutex)
    assert_free()


def test_cleanup_failure_does_not_drop_mutex(monkeypatch):
    mutex = claim.mutex(timeout=0)
    monkeypatch.setenv("HFMODEM_COEXIST_FAULT", "after_holder_line")
    with pytest.raises(RuntimeError):
        claim.acquire("cleanup", "down", cleanup=mutex)
    monkeypatch.delenv("HFMODEM_COEXIST_FAULT")
    assert claim.verify_cleanup(mutex) == mutex
    with pytest.raises(claim.ClaimBusy):
        acquire()
    os.close(mutex)
    assert_free()


def test_launch_admission_holds_until_opened_or_failed(capsys):
    descriptor = acquire()
    for method in ("opened", "failed"):
        admission = claim.inherited(descriptor, launch=True)
        with pytest.raises(claim.ClaimBusy, match="admission in progress"):
            claim.mutex(timeout=0)
        getattr(admission, method)()
        admission.failed()
        mutex = claim.mutex(timeout=0)
        with pytest.raises(claim.ClaimBusy, match="cleanup"):
            claim.inherited(descriptor, launch=True)
        assert claim.inherited(descriptor)
        os.close(mutex)
    assert capsys.readouterr().out == "CLAIM ADMITTED\n"
    claim.release(descriptor)


def test_failed_inheritance_drops_admission():
    with pytest.raises(claim.ClaimError):
        claim.inherited(None, launch=True)
    mutex = claim.mutex(timeout=0)
    os.close(mutex)


@pytest.mark.parametrize("value, expected", [("0t1099511627776", claim.MAX_NONCE),
                                            ("0x10000000000", claim.MAX_NONCE), ("1", 1)])
def test_descriptor_offsets(value, expected):
    assert claim.parse_offset(value) == expected
    assert claim.fdinfo_offset(f"pos:\t{expected}\nflags:\t0100002\n") == expected


def discovery(text, *, platform="darwin", offsets=None):
    def fdinfo(pid, fd):
        if (pid, fd) not in (offsets or {}):
            raise OSError("fake fdinfo unavailable")
        return f"pos:\t{offsets[(pid, fd)]}\n"
    return dict(lsof=lambda paths: text, platform=platform, started=lambda pid: f"start-{pid}",
                fdinfo=fdinfo)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_mutex_holder_identified_including_dead_publisher_child(platform):
    mutex = claim.mutex(timeout=0)
    nonce = claim.holder(mutex, cleanup=True).nonce
    text = f"p101\ncchild\nf8\no0x{nonce:x}\np102\nccontender\nf8\no0t0\n"
    options = discovery(text, platform=platform, offsets={(101, 8): nonce, (102, 8): 0})
    try:
        rows = claim.holders(mutex, cleanup=True, **options)
        assert [r.pid for r in rows] == [101]
        assert rows[0].started == "start-101"
        with pytest.raises(claim.ClaimBusy, match=r"cleanup in progress \(pids 101\)"):
            claim.mutex(timeout=30, **options)
    finally:
        os.close(mutex)


def test_stale_cleanup_line_not_an_admission_identity():
    mutex = claim.mutex(timeout=0)
    os.close(mutex)
    shared = claim._shared()
    try:
        options = discovery("p42\ncadmission\nf7\no0t0\n")
        with pytest.raises(claim.ClaimBusy, match=r"admission in progress \(pids 42\)"):
            claim.mutex(timeout=0, **options)
    finally:
        os.close(shared)


def test_discovery_reused_pid_and_unknown_fail_closed():
    starts = iter(("old", "new"))
    with pytest.raises(claim.ClaimError, match="changed"):
        claim.openers([claim.claim_path()], lsof=lambda paths: "p42\nf7\no0t9\n",
                      started=lambda pid: next(starts), platform="darwin")
    with pytest.raises(claim.ClaimError, match="unavailable"):
        claim.openers([], **discovery("p42\nf7\no0t9\n", platform="linux", offsets={}))


@pytest.fixture
def exited_pid():
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    return process.pid


def exiting(pid, lookups, rest, inherited=""):
    calls = {}
    def started(p):
        calls[p] = calls.get(p, 0) + 1
        if p == pid and calls[p] > lookups:
            raise claim.ClaimError(f"cannot identify pid {p}")
        return f"start-{p}"
    def lsof(paths):
        return (f"p{pid}\ncbrief\nf7\no0t5\n" if calls.get(pid, 0) <= lookups else inherited) + rest
    return dict(lsof=lsof, started=started)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("lookups", [0, 1])
def test_a_holder_that_exits_during_discovery_is_not_a_holder(exited_pid, platform, lookups):
    live = os.getpid()
    options = discovery("", platform=platform, offsets={(exited_pid, 7): 5, (live, 7): 5})
    options.update(exiting(exited_pid, lookups, f"p{live}\nclive\nf7\no0t5\n"))
    rows = claim.openers([claim.claim_path()], **options)
    assert [(r.pid, r.started) for r in rows] == [(live, f"start-{live}")]


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("lookups", [0, 1])
def test_a_child_left_holding_an_exited_holders_line_refuses(exited_pid, platform, lookups):
    child = os.getpid()
    options = discovery("", platform=platform, offsets={(exited_pid, 7): 5, (child, 7): 5})
    options.update(exiting(exited_pid, lookups, "", f"p{child}\ncdaemon\nf7\no0t5\n"))
    with pytest.raises(claim.ClaimError, match="changed"):
        claim.openers([claim.claim_path()], **options)


def test_a_live_holder_that_cannot_be_identified_refuses():
    live = os.getpid()
    def started(pid):
        raise claim.ClaimError(f"cannot identify pid {pid}")
    with pytest.raises(claim.ClaimError, match=f"cannot identify pid {live}"):
        claim.openers([], lsof=lambda paths: f"p{live}\nf7\no0t5\n", started=started,
                      platform="darwin")


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_a_pid_signalling_cannot_reach_is_not_exited(monkeypatch, platform):
    def kill(pid, signal):
        raise PermissionError(1, "Operation not permitted")
    def started(pid):
        raise claim.ClaimError(f"cannot identify pid {pid}")
    monkeypatch.setattr(claim.os, "kill", kill)
    with pytest.raises(claim.ClaimError, match="cannot identify pid 4242"):
        claim.openers([], lsof=lambda paths: "p4242\nf7\no0t5\n", started=started,
                      platform=platform)


def test_an_exited_pid_back_in_the_rescan_refuses(exited_pid):
    def started(pid):
        raise claim.ClaimError(f"cannot identify pid {pid}")
    with pytest.raises(claim.ClaimError, match="changed"):
        claim.openers([], lsof=lambda paths: f"p{exited_pid}\nf7\no0t5\n", started=started,
                      platform="darwin")


class FakePtt:
    def __init__(self, sensed=False):
        self.sensed = sensed
        self.calls = []
        self.released_low = True

    def assert_(self, value):
        self.calls.append(("assert", value))

    def sense(self):
        self.calls.append("sense")
        return self.sensed

    def release(self):
        self.calls.append("release")


@pytest.mark.parametrize("sensed, verdict", [(False, "low"), (True, "high"), (None, "unreadable")])
def test_line_proof_under_claim_and_admission(sensed, verdict, capsys):
    ptt = FakePtt(sensed)
    def factory():
        with pytest.raises(claim.ClaimBusy):
            acquire()
        with pytest.raises(claim.ClaimBusy):
            claim.mutex(timeout=0)
        return ptt
    if sensed is False:
        descriptor = acquire(prove_line=True, ptt_factory=factory)
        claim.release(descriptor)
    else:
        with pytest.raises(claim.ClaimError, match="NOT CONFIRMED"):
            acquire(prove_line=True, ptt_factory=factory)
        assert claim.claim_path().read_bytes() == b""
    assert ptt.calls == [("assert", False), "sense", "release"]
    assert capsys.readouterr().out == f"PTT LINE {verdict}\n"
    assert_free()


def test_proof_exception_still_closes_line(monkeypatch, capsys):
    ptt = FakePtt()
    monkeypatch.setattr(ptt, "sense", lambda: (_ for _ in ()).throw(OSError("readback")))
    with pytest.raises(OSError):
        acquire(prove_line=True, ptt_factory=lambda: ptt)
    assert ptt.calls[-1] == "release"
    assert capsys.readouterr().out == "PTT LINE unreadable\n"
    assert_free()


def test_device_check_precedes_proof_and_blocks_unknown_opener():
    called = []
    with pytest.raises(claim.ClaimBusy, match="held by pid 42"):
        acquire(prove_line=True, ptt_factory=lambda: called.append(True),
                devices=claim.Devices(ptt_port="/dev/cu.fake"),
                **discovery("p42\ncunknown\nf4\no0t0\n"))
    assert not called
    assert claim.claim_path().read_bytes() == b""
    assert_free()


def test_serial_aliases_and_rigctld_delegation(tmp_path):
    target = tmp_path / "ttyUSB0"
    target.touch()
    alias = tmp_path / "by-id"
    alias.symlink_to(target)
    assert claim.serial_nodes(alias, platform="linux") == (str(target),)
    assert claim.serial_nodes("/dev/cu.fake", platform="darwin") == ("/dev/cu.fake", "/dev/tty.fake")
    assert claim.serial_nodes("/dev/tty.fake", platform="darwin") == ("/dev/cu.fake", "/dev/tty.fake")
    calls = []
    options = dict(lsof=lambda paths: calls.append(paths) or "", platform="darwin")
    verdicts = claim.check_devices(claim.Devices(cat_port="/dev/cu.cat", ptt_port="/dev/cu.ptt",
                                                cat="rigctld"), **options)
    assert calls == [("/dev/cu.ptt", "/dev/tty.ptt")]
    assert verdicts[0].verdict == "delegated"


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_blank_port_is_no_port(blank):
    devices = claim.Devices(cat_port=blank, ptt_port=blank)
    assert devices.cat_port is None and devices.ptt_port is None
    assert claim.serial_nodes(blank) == ()
    assert claim.line_factory(blank) is None
    verdicts = claim.check_devices(devices, lsof=lambda paths: pytest.fail(f"lsof {paths}"))
    assert [v.endpoint for v in verdicts] == ["capture", "playback"]


def test_serial_check_without_nodes_never_runs_discovery(monkeypatch):
    monkeypatch.setattr(claim, "serial_nodes", lambda *a, **k: ())
    verdicts = claim.check_devices(claim.Devices(cat_port="/dev/cu.cat", ptt_port="/dev/cu.ptt"),
                                   lsof=lambda paths: pytest.fail(f"lsof {paths}"))
    assert [(v.endpoint, v.verdict, v.nodes) for v in verdicts[:2]] == [
        ("cat", "clear", ()), ("ptt", "clear", ())]


@pytest.mark.parametrize("cat_port, ptt_port", [("", ""), ("", " "), (" ", "")])
def test_shell_twin_blank_ports_never_resolve_the_working_directory(tmp_path, cat_port, ptt_port):
    if shutil.which("lsof") is None:
        pytest.skip("real lsof unavailable: executable not found")
    claim.bootstrap()
    fd = os.open(claim.claim_path(), os.O_RDWR)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "hfmodem.core.claim", "acquire", str(fd), "--kind", "cleanup",
             "--operation", "down", "--prove-line", "--cat-port", cat_port, "--ptt-port", ptt_port],
            cwd=tmp_path, pass_fds=(fd,), capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "PTT LINE none\nCLAIM ACQUIRED\n"
        assert twin("release", fd, pass_fds=(fd,)).stdout == "CLAIM RELEASED\n"
    finally:
        os.close(fd)
    assert_free()


def test_shell_twin_still_checks_a_configured_port(tmp_path):
    if shutil.which("lsof") is None:
        pytest.skip("real lsof unavailable: executable not found")
    port = tmp_path / "ptt"
    port.touch()
    claim.bootstrap()
    fd = os.open(claim.claim_path(), os.O_RDWR)
    try:
        with open(port, "rb"):
            if not _real_lsof([port]).strip():
                pytest.skip("real lsof unavailable: cannot observe a known open port (sandbox restriction)")
            result = twin("acquire", fd, "--kind", "shell", "--operation", "test", "--cat", "none",
                          "--ptt-port", port, pass_fds=(fd,))
        assert result.returncode == 2, result.stdout + result.stderr
        assert f"ptt held by pid {os.getpid()}" in result.stderr
        assert os.pread(fd, 16384, 0) == b""
    finally:
        os.close(fd)
    assert_free()


@pytest.mark.parametrize("command, refused", [("pipewire", False), ("pulseaudio", False),
                                            ("wireplumber", False), ("jackd", False),
                                            ("/usr/bin/wireplumber", False), ("jackd-fake", True),
                                            ("coreaudiod", False), ("recorder", True),
                                            ("pipewire-fake", True)])
def test_audio_server_exemption(monkeypatch, command, refused):
    monkeypatch.setattr(claim, "audio_nodes", lambda endpoint, direction, **kw:
                        ("/dev/snd/pcmC1D0c",) if direction == "capture" else ())
    options = discovery(f"p42\nc{command}\nf5\no0t0\n")
    if refused:
        with pytest.raises(claim.ClaimBusy, match="capture held by pid 42"):
            claim.check_devices(**options)
    else:
        verdicts = claim.check_devices(**options)
        assert verdicts[0].verdict == "cooperative" and verdicts[0].pids == (42,)
        assert verdicts[1].verdict == "cooperative" and not verdicts[1].nodes


def test_audio_resolution_defaults_different_endpoints_and_cooperative(monkeypatch):
    monkeypatch.setattr(claim, "audio_nodes", _audio_nodes)
    calls = []
    sd = SimpleNamespace(default=SimpleNamespace(device=(3, 7)),
                         query_devices=lambda device, kind: calls.append((device, kind)) or
                         {"name": f"Fake USB (hw:{device},0)"})
    assert claim.audio_nodes(None, "capture", sounddevice=sd, platform="linux") == ("/dev/snd/pcmC3D0c",)
    assert claim.audio_nodes("default", "playback", sounddevice=sd, platform="linux") == ("/dev/snd/pcmC7D0p",)
    assert calls == [(3, "input"), (7, "output")]
    assert claim.audio_nodes("plughw:2,1", "capture", platform="linux") == ("/dev/snd/pcmC2D1c",)
    assert claim.audio_nodes("/dev/snd/pcmC1D0p", "playback", platform="linux") == ("/dev/snd/pcmC1D0p",)
    assert claim.audio_nodes(3, "capture", platform="darwin") == ()
    sd.query_devices = lambda *args: {"name": "default server"}
    assert claim.audio_nodes(None, "capture", sounddevice=sd, platform="linux") == ()


def test_rxreadiness_uses_possession_and_preserves_record(monkeypatch):
    descriptor = acquire()
    fd = os.open(claim.claim_path(), os.O_RDWR)
    with monkeypatch.context() as patch:
        patch.setattr(fcntl, "flock", lambda *args: pytest.fail("RX verifier locked"))
        info = descriptor.fstat()
        assert rxreadiness.verify_inherited_claim("fake-cat", descriptor.fd) == {
            "device": info.st_dev, "inode": info.st_ino, "cat": "fake-cat"}
        with pytest.raises(rxreadiness.ReceiverReadbackError):
            rxreadiness.verify_inherited_claim("fake-cat", fd)
    os.close(fd)
    claim.release(descriptor)


def twin(*args, pass_fds=()):
    return subprocess.run([sys.executable, "-m", "hfmodem.core.claim", *map(str, args)],
                          pass_fds=pass_fds, capture_output=True, text=True, timeout=10)


def test_shell_twin_status_and_parent_identity():
    result = twin("path")
    assert result.returncode == 0 and result.stdout.strip() == str(claim.claim_path())
    fd = os.open(claim.claim_path(), os.O_RDWR)
    result = twin("acquire", fd, "--kind", "shell", "--operation", "test", pass_fds=(fd,))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "CLAIM ACQUIRED\n"
    assert claim.holder(fd).pid == os.getpid()
    assert claim.inherited(fd)
    for verb, status in (("inherited", "CLAIM INHERITED"), ("close", "CLAIM CLOSED")):
        result = twin(verb, fd, pass_fds=(fd,))
        assert result.returncode == 0 and result.stdout.strip() == status
        assert claim.inherited(fd)
    result = twin("release", fd, pass_fds=(fd,))
    assert result.returncode == 0 and result.stdout == "CLAIM RELEASED\n"
    assert os.pread(fd, 16384, 0) == b""
    os.close(fd)
    assert_free()


def test_shell_twin_mutex_cleanup_and_launch():
    claim.bootstrap()
    mutex = os.open(claim.mutex_path(), os.O_RDWR)
    fd = os.open(claim.claim_path(), os.O_RDWR)
    result = twin("mutex", mutex, "--timeout", 0, pass_fds=(mutex,))
    assert result.returncode == 0 and result.stdout == "CLAIM MUTEX\n"
    assert claim.holder(mutex, cleanup=True).pid == os.getpid()
    result = twin("acquire", fd, "--kind", "cleanup", "--operation", "down",
                  "--cleanup", mutex, "--prove-line", pass_fds=(mutex, fd))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "PTT LINE none\nCLAIM ACQUIRED\n"
    result = twin("mutex_release", mutex, pass_fds=(mutex,))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "CLAIM MUTEX RELEASED\n"
    assert os.pread(mutex, 16384, 0) == b""
    # A helper closes only its reference; the shell still holds fd 8.
    with pytest.raises(claim.ClaimBusy):
        claim.mutex(timeout=0)
    os.close(mutex)
    admission = os.open(claim.mutex_path(), os.O_RDWR)
    result = twin("inherited", fd, "--launch", admission, pass_fds=(fd, admission))
    assert result.returncode == 0 and result.stdout == "CLAIM INHERITED\n"
    with pytest.raises(claim.ClaimBusy):
        claim.mutex(timeout=0)
    os.close(admission)
    claim.release(fd)
    result = twin("inherited", fd)
    assert result.returncode != 0 and result.stderr.startswith("CLAIM REFUSED")


def test_claim_descriptor_surface_and_no_unlock_ratchet():
    assert {name for name in vars(claim.ClaimDescriptor) if not name.startswith("_")} == {
        "pread", "fstat", "offset"}
    tree = ast.parse(Path(claim.__file__).read_text())
    assert not any(isinstance(node, ast.Attribute) and node.attr == "LOCK_UN" for node in ast.walk(tree))
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                assert node.func.attr not in ("read", "write")
                if node.func.attr == "lseek":
                    assert function.name in ("_publish", "offset")


@pytest.mark.parametrize("flags", [os.O_RDONLY, os.O_WRONLY, os.O_RDWR | os.O_APPEND])
def test_acquisition_refuses_wrong_open_mode(flags):
    claim.bootstrap()
    claim.claim_path().write_text("stale diagnostic\n")
    fd = os.open(claim.claim_path(), flags)
    with pytest.raises(claim.ClaimError, match="O_RDWR"):
        acquire(fd=fd)
    assert claim.claim_path().read_text() == "stale diagnostic\n"
    with pytest.raises(OSError):
        os.fstat(fd)
    assert_free()


def test_attach_style_child_inherits_nothing():
    descriptor = acquire()
    result = child("""
import sys
from hfmodem.core import claim
try:
    claim.inherited(int(sys.argv[1]))
except (claim.ClaimError, OSError):
    pass
else:
    raise AssertionError('attach inherited authority')
""", descriptor.fd)
    assert result.returncode == 0, result.stderr
    claim.release(descriptor)


def test_claim_discovery_excludes_unlocked_reopener():
    descriptor = acquire()
    nonce = descriptor.offset()
    rows = claim.holders(descriptor, **discovery(
        f"p42\ncinheritor\nf9\no0t{nonce}\np43\ncunlocked\nf9\no0t0\n"))
    assert [row.pid for row in rows] == [42]
    claim.release(descriptor)


def test_mutex_replacement_refuses_cleanup_authority():
    mutex = claim.mutex(timeout=0)
    claim.mutex_path().unlink()
    claim.mutex_path().touch()
    try:
        with pytest.raises(claim.ClaimError, match="does not name"):
            claim.acquire("cleanup", "down", cleanup=mutex)
    finally:
        os.close(mutex)


def test_cleanup_unknown_discovery_refuses_at_once():
    mutex = claim.mutex(timeout=0)
    def unavailable(paths):
        raise OSError("fake lsof unavailable")
    try:
        with pytest.raises(claim.ClaimBusy, match="cleanup in progress .*holder unknown"):
            claim.mutex(timeout=30, lsof=unavailable)
    finally:
        os.close(mutex)


def test_publication_event_failure_unwinds(monkeypatch):
    def fail(name, **fields):
        if name == "holder_published":
            raise OSError("evidence unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(coexist, "event", fail)
        with pytest.raises(OSError, match="evidence unavailable"):
            acquire()
    assert claim.claim_path().read_bytes() == b""
    assert_free()


def test_line_factory_opens_the_configured_line(monkeypatch):
    from hfmodem.core import ptt
    opened = []
    class Line:
        def __init__(self, port):
            self.port = port
        def open(self):
            opened.append(self.port)
    monkeypatch.setattr(ptt, "RtsPtt", Line)
    assert claim.line_factory(None) is None
    factory = claim.line_factory("/dev/cu.fake")
    assert opened == [] and factory().port == "/dev/cu.fake" and opened == ["/dev/cu.fake"]


def test_missing_configured_proof_factory_refuses():
    with pytest.raises(claim.ClaimError, match="proof factory"):
        acquire(prove_line=True, devices=claim.Devices(ptt_port="/dev/cu.fake"))
    assert_free()


def test_unconfirmed_release_refuses_even_after_low_readback():
    ptt = FakePtt()
    ptt.released_low = False
    with pytest.raises(claim.ClaimError, match="NOT CONFIRMED"):
        acquire(prove_line=True, ptt_factory=lambda: ptt)
    assert_free()


@pytest.mark.parametrize("text", ["garbage", "p42\nfmem\n", "p42\nf7\nounknown\n"])
def test_malformed_discovery_is_not_an_empty_inventory(text):
    with pytest.raises(claim.ClaimError, match="discovery unavailable"):
        claim.check_devices(claim.Devices(ptt_port="/dev/cu.fake"), **discovery(text))


def test_an_absent_port_has_no_opener(tmp_path):
    verdicts = claim.check_devices(claim.Devices(cat_port=tmp_path / "unplugged-cat",
                                                 ptt_port=tmp_path / "unplugged-ptt"), lsof=_real_lsof)
    assert [(v.endpoint, v.verdict) for v in verdicts[:2]] == [("cat", "clear"), ("ptt", "clear")]


def test_a_present_port_held_open_is_refused(tmp_path):
    node = tmp_path / "plugged-ptt"
    node.touch()
    holder = subprocess.Popen(["bash", "-c", 'exec 9<"$0"; echo ready; exec sleep 60', str(node)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline() == "ready\n"
        with pytest.raises(claim.ClaimBusy, match=f"ptt held by pid {holder.pid}"):
            claim.check_devices(claim.Devices(cat_port=tmp_path / "unplugged-cat", ptt_port=node),
                                lsof=_real_lsof)
    finally:
        holder.kill()
        holder.wait()
