# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
from contextlib import contextmanager
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import textwrap
import time
from types import SimpleNamespace

import pytest

from hfmodem.core import claim, coexist, native, winclaim, winproc
from hfmodem.tests import hermetic

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows share-mode claim")

PYTHON = getattr(sys, "_base_executable", None) or sys.executable

PRELUDE = """
import ctypes as C, json, os, sys, time
PYTHON = getattr(sys, "_base_executable", None) or sys.executable
from hfmodem.core import claim, winclaim, winproc
"""


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", guarded_path())
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_STATE_DIR", str(tmp_path / "state"))
    for key in ("HFMODEM_COEXIST_FAULT", "HFMODEM_COEXIST_LOG", "HFMODEM_COEXIST_BARRIERS",
                "RIG_CLAIMED"):
        monkeypatch.delenv(key, raising=False)
    assert claim._adopted is None
    monkeypatch.setattr(winclaim, "_received_loaded", False)
    monkeypatch.setattr(winclaim, "_received_record", None)
    monkeypatch.setattr(winclaim, "_received_admission", None)
    monkeypatch.setattr(winclaim, "_extras", {})
    yield
    leaked = winclaim.entries()
    try:
        assert claim._adopted is None
        assert leaked == {}
    finally:
        for fd in leaked:
            claim.close(fd)


def guarded_path():
    """This interpreter's path behind the guard's startup, which sys.path holds
    only once something in this process has imported it from there."""
    return os.pathsep.join([str(hermetic.GUARD), *filter(None, sys.path)])


def set_token(patch, value):
    patch.setattr(winclaim, "_received_loaded", True)
    patch.setattr(winclaim, "_received_record", {"handles": {"claim": value}})


def spawn_claim(descriptor, argv, *, role="python", **options):
    admission = (claim.inherited(descriptor, launch=True)
                 if role in ("python", "supervisor", "one-shot") else None)
    try:
        return claim.spawn(admission or descriptor, argv, role=role, **options)
    finally:
        if admission is not None:
            admission.failed()


def acquire(**kwargs):
    return claim.acquire("test", "windows", **kwargs)


def assert_free():
    descriptor = acquire()
    claim.release(descriptor)


@contextmanager
def held():
    descriptor = acquire()
    try:
        yield descriptor
    finally:
        claim.release(descriptor)


@contextmanager
def raw(path, role=None, *, access=None, share=None):
    h = (winclaim._open(path, role) if role is not None else
         winclaim.open_handle(path, access, share))
    try:
        yield h
    finally:
        winclaim.close_handle(h)


class Child:
    def __init__(self, process):
        self.process = process
        self.lines = queue.Queue()
        def read():
            for line in process.stdout:
                self.lines.put(line.rstrip("\n"))
            self.lines.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def line(self, timeout=30):
        value = self.lines.get(timeout=timeout)
        if value is None:
            self.process.wait(timeout=10)
            pytest.fail(f"child exited {self.process.returncode}: {self.process.stderr.read()}")
        return value

    def send(self, value="go"):
        self.process.stdin.write(value + "\n")
        self.process.stdin.flush()

    def wait(self, code=0):
        assert self.process.wait(timeout=30) == code, self.process.stderr.read()
        self.reader.join(timeout=5)


@pytest.fixture
def children():
    running = []
    def start(code, *args, inherited=None, env=None, role="python", handles=None, own_session=False):
        child_env = dict(os.environ) if env is None else dict(env)
        child_env["PYTHONPATH"] = guarded_path()
        options = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=child_env)
        argv = [PYTHON, "-c", PRELUDE + code, *map(str, args)]
        launch = (spawn_claim(inherited, argv, role=role, handles=handles, own_session=own_session,
                              **options) if inherited is not None or role == "recorder" else None)
        process = subprocess.Popen(argv, **options) if launch is None else launch.process
        child = Child(process)
        child.launch = launch
        running.append(child)
        return child
    yield start
    for child in reversed(running):
        if child.process.poll() is None:
            child.process.kill()
        child.process.wait(timeout=30)
        child.reader.join(timeout=5)
        if child.launch is not None:
            child.launch.child.close()
        for stream in (child.process.stdin, child.process.stdout, child.process.stderr):
            stream.close()


@pytest.mark.parametrize("nonce", [1, claim.MAX_NONCE])
def test_publication_under_reservation(monkeypatch, nonce):
    monkeypatch.setattr(claim.secrets, "randbelow", lambda limit: nonce - 1)
    seen = []
    def checkpoint(point):
        if point in ("before_holder_line", "after_holder_line"):
            with pytest.raises(OSError) as failure:
                winclaim._open(claim.claim_path(), "holder")
            assert failure.value.winerror == 32
            seen.append(winclaim.read_record(claim.claim_path()))
    monkeypatch.setattr(coexist, "checkpoint", checkpoint)
    with held() as descriptor:
        assert seen[0] == b""
        assert seen[1] == descriptor.pread()
        published = claim.holder(descriptor)
        assert (published.pid, published.kind, published.operation) == (os.getpid(), "test", "windows")
        assert published.nonce == descriptor.offset() == nonce
        assert not os.get_handle_inheritable(winclaim.handle(descriptor.fd))


def test_short_publication_unwinds(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(winclaim, "_write", lambda *args: 1)
        with pytest.raises(claim.ClaimError, match="short holder line write"):
            acquire()
    assert claim.claim_path().read_bytes() == b""
    assert winclaim.entries() == {}
    assert_free()


def test_inherited_proves_without_acquisition(monkeypatch):
    with held() as descriptor:
        duplicate = os.dup(descriptor.fd)
        original = winclaim._open
        def no_reservations(path, role):
            assert role not in ("holder", "cleanup", "admission")
            return original(path, role)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(winclaim, "_open", no_reservations)
                assert claim.inherited(duplicate).fd == duplicate
            with pytest.raises(OSError) as failure:
                winclaim._open(claim.claim_path(), "holder")
            assert failure.value.winerror == 32
            with raw(claim.claim_path(), access=winclaim.COMMON_ACCESS | 1, share=3) as h:
                reopened = winclaim._crt(winclaim.duplicate(h), os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
            try:
                with pytest.raises(claim.ClaimError, match="read and write"):
                    claim.inherited(reopened)
            finally:
                claim.close(reopened)
        finally:
            os.close(duplicate)


@pytest.mark.parametrize("data", [b"", b"123 start kind op 4 cmd", b"123 start kind op 0 cmd\n",
    b"123 start kind op 1099511627777 cmd\n", b"partial\n", b"1 start kind op 4 cmd\nextra\n",
    b"\xff\n", b"x" * 16385])
def test_malformed_records(data):
    claim.bootstrap()
    claim.claim_path().write_bytes(data)
    fd = winclaim.open_role(claim.claim_path(), "holder")
    try:
        with pytest.raises(claim.ClaimError, match="incomplete holder line"):
            claim.holder(fd)
        # Match the nonce used by fixtures with otherwise plausible syntax.
        os.lseek(fd, 4, os.SEEK_SET)
        with pytest.raises(claim.ClaimError, match="incomplete holder line"):
            claim.inherited(fd)
    finally:
        claim.close(fd)
    assert_free()


def test_stale_record_and_released_reference():
    with held() as descriptor:
        duplicate = os.dup(descriptor.fd)
        nonce = descriptor.offset()
    try:
        assert winclaim.offset(duplicate) == nonce
        with pytest.raises(claim.ClaimError, match="incomplete holder line"):
            claim.inherited(duplicate)
    finally:
        os.close(duplicate)
    claim.claim_path().write_bytes(b"123 start kind op 4 cmd\n")
    # A stale line and a matching pointer give a reopened reader no authority.
    # The admission role carries READ_CONTROL, so the owner check passes and
    # the access predicate is what refuses.
    reader = winclaim.open_role(claim.claim_path(), "admission")
    try:
        os.lseek(reader, 4, os.SEEK_SET)
        assert winclaim.offset(reader) == claim.holder(reader).nonce
        with pytest.raises(claim.ClaimError, match="read and write"):
            claim.inherited(reader)
    finally:
        claim.close(reader)
    fd = winclaim.open_role(claim.claim_path(), "holder")
    try:
        assert claim.holder(fd).nonce == 4
        with pytest.raises(claim.ClaimError, match="does not possess"):
            claim.inherited(fd)
    finally:
        claim.close(fd)
    assert_free()


def test_partial_record_retries_without_moving_pointer(monkeypatch):
    with held() as descriptor:
        original = winclaim.read_record
        calls = []
        def read(*args, **kwargs):
            calls.append(1)
            return b"partial" if len(calls) < 3 else original(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(winclaim, "read_record", read)
            assert claim.inherited(descriptor) == descriptor
        assert len(calls) == 3
        assert descriptor.offset() == claim.holder(descriptor).nonce


@pytest.mark.parametrize("cleanup", [False, True])
def test_held_files_cannot_be_replaced(tmp_path, cleanup):
    fd = claim.mutex(timeout=0) if cleanup else acquire().fd
    path = claim.mutex_path() if cleanup else claim.claim_path()
    replacement = tmp_path / "replacement"
    replacement.touch()
    try:
        for action, errors in ((lambda: path.unlink(), {32}),
                               (lambda: os.replace(replacement, path), {5, 32})):
            with pytest.raises(PermissionError) as failure:
                action()
            assert failure.value.winerror in errors
        assert claim.verify_cleanup(fd) == fd if cleanup else claim.check_inode(fd)
    finally:
        claim.mutex_release(fd) if cleanup else claim.release(fd)


def test_unheld_replacement_and_stray_file(tmp_path):
    claim.bootstrap()
    # A delete-sharing handle permits replacement so the identity check is exercised.
    h = winclaim.open_handle(claim.claim_path(), winclaim.COMMON_ACCESS | 3, 7)
    fd = winclaim._crt(h, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
    try:
        claim.claim_path().unlink()
        claim.claim_path().touch()
        with pytest.raises(claim.ClaimError, match="does not name"):
            claim.check_inode(fd)
    finally:
        claim.close(fd)
    stray = os.open(tmp_path / "stray", os.O_CREAT | os.O_RDWR)
    try:
        with pytest.raises(claim.ClaimError, match="does not name"):
            claim.inherited(stray)
    finally:
        os.close(stray)
    directory = winclaim._api().CreateFileW(str(tmp_path), winclaim.READ_CONTROL | 0x80, 7,
                                           None, 3, 0x02000000, None)
    assert directory != winclaim.INVALID_HANDLE_VALUE
    try:
        assert not winclaim.owned_regular(directory)
    finally:
        winclaim.close_handle(directory)


@pytest.mark.parametrize("matching_class", [1, 4, None])
def test_owner_accepts_token_user_or_default_owner(monkeypatch, matching_class):
    with held() as descriptor:
        classes = []
        original = winclaim._token_sid
        def token_sid(token, kind):
            classes.append(kind)
            return original(token, kind)
        with monkeypatch.context() as patch:
            patch.setattr(winclaim, "_token_sid", token_sid)
            patch.setattr(winclaim._api(), "EqualSid",
                          lambda owner, sid: classes[-1] == matching_class)
            assert winclaim.owned_regular(winclaim.handle(descriptor.fd)) == (matching_class is not None)
        assert classes == ([1] if matching_class == 1 else [1, 4])


def test_real_owner_and_foreign_sid_routing(monkeypatch):
    with held() as descriptor:
        assert winclaim.owned_regular(winclaim.handle(descriptor.fd))
        assert claim.inherited(descriptor) == descriptor
        with monkeypatch.context() as patch:
            patch.setattr(winclaim, "owned_regular", lambda h: False)
            with pytest.raises(claim.ClaimError, match="does not name"):
                claim.inherited(descriptor)
            with pytest.raises(claim.ClaimError, match="acquired description"):
                claim.release(descriptor)
    hosts = Path(os.environ["SystemRoot"]) / "System32/drivers/etc/hosts"
    with raw(hosts, access=winclaim.READ_CONTROL | 0x80, share=7) as h:
        assert not winclaim.owned_regular(h)


@pytest.mark.parametrize("flags", [os.O_RDONLY, os.O_WRONLY, os.O_RDWR | os.O_APPEND, os.O_RDWR])
def test_acquire_refuses_foreign_descriptor(flags):
    claim.bootstrap()
    claim.claim_path().write_text("stale diagnostic\n")
    fd = os.open(claim.claim_path(), flags)
    with pytest.raises(claim.ClaimError, match="its own share-mode open on Windows"):
        acquire(fd=fd)
    with pytest.raises(OSError):
        os.fstat(fd)
    assert claim.claim_path().read_text() == "stale diagnostic\n"
    assert_free()


@pytest.mark.parametrize("access", [winclaim.COMMON_ACCESS | 1, winclaim.COMMON_ACCESS | 2,
                                     winclaim.COMMON_ACCESS | 3])
def test_inherited_reduced_access(access):
    with held() as descriptor:
        h = winclaim.duplicate(winclaim.handle(descriptor.fd), access=access)
        fd = winclaim._crt(h, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
        try:
            assert winclaim.offset(fd) == descriptor.offset()
            if access & 3 == 3:
                assert claim.inherited(fd).fd == fd
            else:
                with pytest.raises(claim.ClaimError, match="read and write data access"):
                    claim.inherited(fd)
        finally:
            claim.close(fd)


@pytest.mark.parametrize("access, message", [(4, "append-only"), (1, "read and write"),
                                              (2, "read and write"), (3, None)])
def test_access_predicates(tmp_path, access, message):
    path = tmp_path / "access"
    path.touch()
    h = winclaim.open_handle(path, access | winclaim.COMMON_ACCESS, 3)
    fd = winclaim._crt(h, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
    try:
        if message:
            with pytest.raises(claim.ClaimError, match=message):
                winclaim.check_access(fd)
        else:
            winclaim.check_access(fd)
    finally:
        claim.close(fd)


def test_backend_conversion_never_uses_append(monkeypatch, tmp_path):
    convert = winclaim._api().msvcrt.open_osfhandle
    flags_seen = []
    def checked(h, flags):
        flags_seen.append(flags)
        assert not flags & os.O_APPEND
        return convert(h, flags)
    monkeypatch.setattr(winclaim._api().msvcrt, "open_osfhandle", checked)
    monkeypatch.setenv("HFMODEM_COEXIST_LOG", str(tmp_path / "events.jsonl"))
    with held() as descriptor:
        mutex = claim.mutex(timeout=0)
        claim.mutex_release(mutex)
        admission = claim.inherited(descriptor, launch=True)
        admission.failed()
        token = winclaim.duplicate(winclaim.handle(descriptor.fd), inheritable=False)
        set_token(monkeypatch, token)
        adopted = claim.received()
        claim.close(adopted)
    assert len(flags_seen) >= 5
    assert {flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR) for flags in flags_seen} == {
        os.O_RDONLY, os.O_WRONLY, os.O_RDWR,
    }


def test_external_crt_append_duplicate_is_accepted():
    with held() as descriptor:
        h = winclaim.duplicate(winclaim.handle(descriptor.fd))
        fd = winclaim._api().msvcrt.open_osfhandle(h, os.O_RDWR | os.O_APPEND | os.O_BINARY)
        try:
            assert claim.inherited(fd).fd == fd
        finally:
            os.close(fd)


def test_cleanup_share_table_and_readers():
    claim.bootstrap()
    path = claim.mutex_path()
    for role in ("admission", "cleanup"):
        with raw(path, "reader"):
            with raw(path, role):
                with raw(path, "reader"):
                    for contender in ("admission", "cleanup"):
                        if role == contender == "admission":
                            with raw(path, contender):
                                pass
                        else:
                            with pytest.raises(OSError) as failure:
                                winclaim._open(path, contender)
                            assert failure.value.winerror == 32


def test_cleanup_requires_backend_authority():
    claim.bootstrap()
    path = claim.mutex_path()
    path.write_bytes(b"123 stale 45\n")
    plain = os.open(path, os.O_RDWR)
    fixtures = [plain]
    fixtures += [winclaim._crt(winclaim.open_handle(path, winclaim.COMMON_ACCESS | 3, 3),
                              os.O_RDWR | os.O_BINARY | os.O_NOINHERIT) for _ in range(2)]
    def refuses(fd, expected):
        before = winclaim.read_record(path)
        registry = winclaim.entries()
        with pytest.raises(claim.ClaimError, match=expected):
            claim.verify_cleanup(fd)
        with pytest.raises(claim.ClaimError, match=expected):
            acquire(cleanup=fd)
        with pytest.raises(claim.ClaimError, match=expected):
            claim.mutex_close(fd)
        os.fstat(fd)
        assert winclaim.read_record(path) == before
        assert winclaim.entries() == registry
        with pytest.raises(OSError) as failure:
            winclaim._open(path, "cleanup")
        assert failure.value.winerror == 32
    try:
        for fd in fixtures:
            refuses(fd, "another description")
    finally:
        for fd in fixtures:
            claim._close(fd)
    admission = claim._shared()
    try:
        refuses(admission, "another description")
    finally:
        claim.Admission(claim.ClaimDescriptor(-1), admission).failed()
    mutex = claim.mutex(timeout=0)
    duplicate = os.dup(mutex)
    try:
        refuses(duplicate, "another description")
        assert winclaim.entries()[mutex].role == "cleanup"
        assert claim.verify_cleanup(mutex) == mutex
    finally:
        os.close(duplicate)
        claim.mutex_close(mutex)
    assert mutex not in winclaim.entries()
    mutex = claim.mutex(timeout=0)
    descriptor = acquire(cleanup=mutex)
    claim.release(descriptor)
    claim.mutex_release(mutex)


@pytest.mark.parametrize("deterministic", [False, True])
def test_cleanup_numeric_reuse_refuses_at_pointer(monkeypatch, deterministic):
    mutex = claim.mutex(timeout=0)
    previous = winclaim.handle(mutex)
    before = winclaim.read_record(claim.mutex_path())
    os.close(mutex)
    fd = os.open(claim.mutex_path(), os.O_RDWR)
    try:
        if fd != mutex or (not deterministic and winclaim.handle(fd) != previous):
            pytest.skip("Windows did not reuse both numeric slots")
        original = winclaim.handle
        with monkeypatch.context() as patch:
            if deterministic:
                patch.setattr(winclaim, "handle", lambda n: previous if n == fd else original(n))
                snapshot = winclaim.entries()
                patch.setattr(winclaim, "entries", lambda: snapshot)
                patch.setattr(winclaim, "handle", original)
            registry = dict(winclaim._registry)
            with pytest.raises(claim.ClaimError, match="another description"):
                claim.verify_cleanup(fd)
            with pytest.raises(claim.ClaimError, match="another description"):
                claim.mutex_close(fd)
            os.fstat(fd)
            assert winclaim._registry == registry
            assert winclaim.read_record(claim.mutex_path()) == before
    finally:
        claim._close(fd)
        winclaim.forget(mutex)
    mutex = claim.mutex(timeout=0)
    claim.mutex_close(mutex)
    assert mutex not in winclaim.entries()


def test_w1_exclusivity_across_processes(children):
    child = children('''
d = claim.acquire("child", "holding")
print("held", flush=True)
input()
claim.release(d)
''')
    assert child.line() == "held"
    with pytest.raises(claim.ClaimBusy) as failure:
        acquire()
    text = str(failure.value)
    assert str(child.process.pid) in text and "child" in text and "holding" in text
    assert "rig.claim open in pid" in text
    assert "publisher exited" not in text
    child.send()
    child.wait()
    assert_free()


@pytest.mark.wall_clock
def test_w2_terminate_releases_immediately(children, tmp_path, record_property):
    timings = []
    for _ in range(20):
        child = children('''
d = claim.acquire("child", "terminate")
print("held", flush=True)
time.sleep(60)
''')
        assert child.line() == "held"
        started = time.monotonic()
        child.process.kill()
        child.process.wait(timeout=10)
        descriptor = acquire()
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        claim.release(descriptor)
        assert elapsed < .250
    evidence = dict(median=statistics.median(timings), maximum=max(timings), seconds=timings)
    (tmp_path / "kill-to-free.json").write_text(json.dumps(evidence))
    record_property("kill_to_free_median_s", evidence["median"])
    record_property("kill_to_free_maximum_s", evidence["maximum"])


@pytest.mark.parametrize("ending, code", [("raise RuntimeError('unhandled')", 1),
    ("sys.exit(3)", 3), ("os._exit(1)", 1)])
def test_w3_every_exit_releases(children, ending, code):
    child = children('''
d = claim.acquire("child", "exit")
print("held", flush=True)
input()
''' + ending)
    assert child.line() == "held"
    child.send()
    child.wait(code)
    assert_free()


@pytest.mark.wall_clock
@pytest.mark.parametrize("route", ["python", "timer_queue"])
def test_w3_phase1_deadman_releases(children, route):
    if not Path(claim.__file__).with_name("windeadman.py").exists():
        pytest.skip("phase-1 windeadman is absent")
    child = children('''
from hfmodem.core import windeadman
from hfmodem.core.rig import arm_deadman
d = claim.acquire("child", "deadman")
print("held", flush=True)
input()
try:
''' + textwrap.indent("arm_deadman(.1)" if route == "python" else
       """
from ctypes import wintypes as W
api = C.WinDLL('kernel32', use_last_error=True)
api.GetCurrentProcess.argtypes, api.GetCurrentProcess.restype = [], W.HANDLE
api.TerminateProcess.argtypes, api.TerminateProcess.restype = [W.HANDLE, W.UINT], W.BOOL
api.CreateTimerQueueTimer.argtypes = [C.POINTER(W.HANDLE), W.HANDLE, C.c_void_p,
                                     C.c_void_p, W.DWORD, W.DWORD, W.ULONG]
api.CreateTimerQueueTimer.restype = W.BOOL
timer = W.HANDLE()
assert api.CreateTimerQueueTimer(C.byref(timer), None, C.cast(api.TerminateProcess, C.c_void_p),
                                 api.GetCurrentProcess(), 100, 0, 8)
""", "    ") + '''
except BaseException:
    os._exit(97)
print("armed", flush=True)
time.sleep(10)
os._exit(98)
''')
    assert child.line() == "held"
    child.send()
    assert child.line() == "armed"
    child.process.wait(timeout=10)
    # 97 is a failed setup and 98 a deadman that never fired. The raw timer
    # passes the callback's BOOLEAN as TerminateProcess's UINT exit code, so
    # only its low byte (TRUE) is defined.
    code = child.process.returncode
    assert (code if route == "python" else code & 0xff) == 1
    assert_free()


@pytest.mark.wall_clock
def test_w4_inheritance_survives_root(children, tmp_path):
    ready, finish, result = [tmp_path / name for name in ("ready", "finish", "result")]
    code = '''
d = claim.received()
h = claim.holder(d)
assert d.offset() == h.nonce
try:
    claim.release(d)
except claim.ClaimError as exc:
    assert 'only the acquirer' in str(exc)
else:
    raise AssertionError('child released parent')
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()))
while not Path(sys.argv[2]).exists():
    time.sleep(.01)
assert claim.inherited(d).offset() == h.nonce
try:
    winclaim._open(claim.claim_path(), 'holder')
except OSError as exc:
    assert exc.winerror == 32
else:
    raise AssertionError('reopen wrote under holder')
claim.close(d)
Path(sys.argv[3]).touch()
'''
    root = children('''
d = claim.acquire("root", "survivor")
admission = claim.inherited(d, launch=True)
launch = claim.spawn(admission, [PYTHON, '-c', sys.argv[1], *sys.argv[2:]], role='python',
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
admission.failed()
print(launch.process.pid, flush=True)
time.sleep(60)
'''.replace('d = claim.acquire', 'import subprocess\nd = claim.acquire'),
                    PRELUDE + code, ready, finish, result)
    survivor = int(root.line())
    try:
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert int(ready.read_text()) == survivor
        root.process.kill()
        root.process.wait(timeout=10)
        with pytest.raises(claim.ClaimBusy, match="publisher exited; claim remains busy; current holder unknown"):
            acquire()
        finish.touch()
        deadline = time.monotonic() + 20
        while not result.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert_free()
    finally:
        finish.touch()
        # A failed assertion must not leave the deliberately surviving child behind.
        h = winclaim._api().OpenProcess(0x1 | winclaim.SYNCHRONIZE, False, survivor)
        if h:
            try:
                if winclaim._api().WaitForSingleObject(h, 5000) == winclaim.WAIT_TIMEOUT:
                    subprocess.run(["taskkill", "/PID", str(survivor), "/F"], capture_output=True, timeout=10)
            finally:
                winclaim.close_handle(h)


PROBE = '''
from hfmodem.core import claim
import os, sys
winproc.read_launch = lambda: {'handles': {'claim': sys.argv[1]}}
try:
    claim.received()
except (claim.ClaimError, OSError):
    print('refused', flush=True)
else:
    raise AssertionError('unintended inheritance')
'''


@pytest.mark.parametrize("route", ["popen", "system", "multiprocessing"])
def test_w5_no_unrequested_inheritance(children, tmp_path, route):
    with held() as descriptor:
        value = winclaim.handle(descriptor.fd)
        assert not os.get_handle_inheritable(value)
        if route == "popen":
            child = children(PROBE, value)
            assert child.line() == "refused"
            child.wait()
        elif route == "system":
            script = tmp_path / "probe.py"
            script.write_text(PRELUDE + PROBE)
            command = '"' + subprocess.list2cmdline([PYTHON, str(script), str(value)]) + '"'
            hermetic.guard.admit_system(command, os.environ)
            assert os.system(command) == 0
        else:
            module = tmp_path / "spawn_probe.py"
            module.write_text('''
import multiprocessing, os, sys
from hfmodem.core import claim, winclaim, winproc
def probe(value):
    winproc.read_launch = lambda: {'handles': {'claim': value}}
    try:
        claim.received()
    except (claim.ClaimError, OSError):
        return
    raise AssertionError('multiprocessing inherited claim')
if __name__ == '__main__':
    d = claim.received()
    try:
        assert claim.inherited(d) == d
        value = winclaim.handle(d.fd)
        assert not os.get_handle_inheritable(value)
        p = multiprocessing.get_context('spawn').Process(target=probe, args=(value,))
        p.start()
        p.join(15)
        assert p.exitcode == 0
    finally:
        claim.close(d)
''')
            launch = spawn_claim(descriptor, [PYTHON, str(module)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = launch.process.communicate(timeout=30)
                assert launch.process.returncode == 0, stderr
            finally:
                if launch.process.poll() is None:
                    launch.process.kill()
                    launch.process.wait(timeout=10)
                launch.child.close()


def test_w5_adopted_child_does_not_leak_to_grandchild(children, tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(PRELUDE + PROBE)
    with held() as descriptor:
        child = children('''
import subprocess
import hermetic_guard
d = claim.received()
assert not os.get_handle_inheritable(winclaim.handle(d.fd))
command = '"' + subprocess.list2cmdline([PYTHON, sys.argv[1], str(winclaim.handle(d.fd))]) + '"'
hermetic_guard.admit_system(command, os.environ)
assert os.system(command) == 0
claim.close(d)
print('done', flush=True)
''', probe, inherited=descriptor)
        assert child.line() == "refused"
        assert child.line() == "done"
        child.wait()


def test_w6_pointer_round_trip(children):
    with held() as descriptor:
        nonce = descriptor.offset()
        with raw(claim.claim_path(), "reader") as reader:
            reader_fd = winclaim._api().msvcrt.open_osfhandle(winclaim.duplicate(reader), os.O_RDONLY | os.O_BINARY)
        try:
            reader_position = len(os.read(reader_fd, 7))
            child = children('''
d = claim.received()
print('ready', flush=True)
input()
nonce = int(sys.argv[1])
assert d.offset() == nonce + 1
os.lseek(d.fd, nonce + 2, os.SEEK_SET)
print('moved', flush=True)
input()
assert d.offset() == nonce
with_read = winclaim._crt(winclaim._open(claim.claim_path(), 'reader'), os.O_RDONLY | os.O_BINARY)
assert winclaim.offset(with_read) == 0
try:
    claim.inherited(with_read)
except claim.ClaimError:
    pass
else:
    raise AssertionError('reopen possessed claim')
claim.close(with_read)
claim.close(d)
''', nonce, inherited=descriptor)
            assert child.line() == "ready"
            os.lseek(descriptor.fd, nonce + 1, os.SEEK_SET)
            child.send()
            assert child.line() == "moved"
            assert descriptor.offset() == nonce + 2
            assert os.lseek(reader_fd, 0, os.SEEK_CUR) == reader_position
            os.lseek(descriptor.fd, nonce, os.SEEK_SET)
            child.send()
            child.wait()
            assert descriptor.offset() == nonce
        finally:
            os.close(reader_fd)
            os.lseek(descriptor.fd, nonce, os.SEEK_SET)


@pytest.mark.wall_clock
def test_w7_real_pid_reuse(children):
    seen = {}
    for _ in range(5000):
        child = children('''
print(claim.process_started(os.getpid()), flush=True)
input()
''')
        pid, started = child.process.pid, child.line()
        if pid in seen:
            try:
                assert started != seen[pid]
                assert not claim.process_matches(pid, seen[pid])
                with held() as descriptor:
                    nonce = descriptor.offset()
                    line = f'{pid} {seen[pid]} old reused {nonce} cmd\n'.encode()
                    winclaim.publish(descriptor.fd, line, nonce)
                    with pytest.raises(claim.ClaimError, match="only the acquirer"):
                        claim.release(descriptor, pid=pid)
                    claim._publish(descriptor.fd, kind="test", operation="restored")
            finally:
                child.send()
                child.wait()
            return
        seen[pid] = started
        child.send()
        child.wait()
        # Release our process-object reference so this pid can actually be reused.
        child.process._handle.Close()
        for stream in (child.process.stdin, child.process.stdout, child.process.stderr):
            stream.close()
    pytest.skip("no pid reused in 5000 processes")


@pytest.mark.wall_clock
def test_w8_readers_never_block_or_mislead(children, tmp_path):
    claim.bootstrap()
    writer_file = tmp_path / "published.json"
    readers = []
    for i in range(3):
        output = tmp_path / f"reader-{i}.json"
        readers.append((children('''
from pathlib import Path
seen = set()
original_read = winclaim.read_record
def read(*args, **kwargs):
    try:
        return original_read(*args, **kwargs)
    except claim.ClaimError as exc:
        assert 'WinError 32' not in str(exc) and 'WinError 33' not in str(exc)
        raise
winclaim.read_record = read
print('ready', flush=True)
end = float(input())
while time.monotonic() < end:
    try:
        h = claim.holder(None)
    except claim.ClaimError as exc:
        assert 'holder unknown' in str(exc)
    else:
        assert h.kind == 'writer' and h.operation == 'stress'
        seen.add(h.nonce)
Path(sys.argv[1]).write_text(json.dumps(sorted(seen)))
''', output), output))
    writer = children('''
from pathlib import Path
published = set()
print('ready', flush=True)
end = float(input())
while time.monotonic() < end:
    d = claim.acquire('writer', 'stress')
    published.add(claim.holder(d).nonce)
    claim.release(d)
Path(sys.argv[1]).write_text(json.dumps(sorted(published)))
''', writer_file)
    for child in [writer, *(r[0] for r in readers)]:
        assert child.line() == "ready"
    end = time.monotonic() + 10
    for child in [writer, *(r[0] for r in readers)]:
        child.send(str(end))
    writer.wait()
    published = set(json.loads(writer_file.read_text()))
    assert published
    for child, output in readers:
        child.wait()
        observed = set(json.loads(output.read_text()))
        assert observed and observed <= published
    assert_free()


def test_w9_foreign_openers_and_stale_record(children):
    claim.bootstrap()
    child = children('''
h = winclaim.open_handle(claim.claim_path(), winclaim.FILE_READ_DATA, 0)
print('held', flush=True)
input()
winclaim.close_handle(h)
''')
    assert child.line() == "held"
    with pytest.raises(claim.ClaimBusy, match="holder unknown") as failure:
        acquire()
    assert "publisher exited" not in str(failure.value)
    child.send()
    child.wait()
    assert_free()
    publisher = children('''
d = claim.acquire('dead', 'publisher')
print('held', flush=True)
time.sleep(60)
''')
    assert publisher.line() == "held"
    publisher.process.kill()
    publisher.process.wait(timeout=10)
    foreign = children('''
h = winclaim.open_handle(claim.claim_path(), winclaim.FILE_READ_DATA, winclaim.FILE_SHARE_READ)
print('held', flush=True)
input()
winclaim.close_handle(h)
''')
    assert foreign.line() == "held"
    with pytest.raises(claim.ClaimBusy, match="publisher exited; claim remains busy; current holder unknown") as failure:
        acquire()
    assert str(publisher.process.pid) in str(failure.value)
    assert str(foreign.process.pid) in str(failure.value)
    foreign.send()
    foreign.wait()
    assert_free()


def test_restart_manager_uncapped_and_failure_is_nonfatal(monkeypatch):
    with held():
        with monkeypatch.context() as patch:
            patch.setattr(winclaim, "restart_manager", lambda path: [(i, f"app{i}") for i in range(1, 513)])
            with pytest.raises(claim.ClaimBusy) as failure:
                acquire()
            for i in range(1, 513):
                assert f"pid {i} app{i}" in str(failure.value)
        with monkeypatch.context() as patch:
            patch.setattr(winclaim, "restart_manager", lambda path: (_ for _ in ()).throw(claim.ClaimError("RM failed")))
            with pytest.raises(claim.ClaimBusy, match="list unavailable"):
                acquire()


@pytest.mark.parametrize("failure", ["register", "list", None])
def test_restart_manager_session_always_ends(monkeypatch, failure):
    api = winclaim._rm_api()
    ended = []
    def start(session, flags, key):
        C.cast(session, C.POINTER(W.DWORD)).contents.value = 42
        return 0
    def listing(session, needed, count, data, reasons):
        if failure == "list":
            return 5
        if data is None:
            C.cast(needed, C.POINTER(W.UINT)).contents.value = 300
            return 234
        C.cast(count, C.POINTER(W.UINT)).contents.value = 300
        for i in range(300):
            data[i].Process.dwProcessId = i + 1
            data[i].strAppName = f"app{i}"
        return 0
    monkeypatch.setattr(api, "RmStartSession", start)
    monkeypatch.setattr(api, "RmRegisterResources", lambda *args: 5 if failure == "register" else 0)
    monkeypatch.setattr(api, "RmGetList", listing)
    monkeypatch.setattr(api, "RmEndSession", lambda session: ended.append(session.value) or 0)
    if failure:
        with pytest.raises(claim.ClaimError):
            winclaim.restart_manager(claim.claim_path())
    else:
        assert len(winclaim.restart_manager(claim.claim_path())) == 300
    assert ended == [42]


def test_w10_real_event_log_and_admission_lifecycle(monkeypatch, tmp_path):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("HFMODEM_COEXIST_LOG", str(log))
    with held() as descriptor:
        mutex = claim.mutex(timeout=0)
        claim.mutex_release(mutex)
        with pytest.raises(claim.ClaimError):
            claim.inherited(None, launch=True)
        admission = claim.inherited(descriptor, launch=True)
        admission.failed()
        admission.failed()
        mutex = claim.mutex(timeout=0)
        claim.mutex_close(mutex)
        assert mutex not in winclaim.entries()
    with monkeypatch.context() as patch:
        patch.setenv("HFMODEM_COEXIST_FAULT", "after_holder_line")
        with pytest.raises(RuntimeError, match="after_holder_line"):
            acquire()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    for name in ("claim_acquired", "admission_acquired", "admission_closed", "claim_released"):
        matches = [row for row in records if row["event"] == name]
        assert matches
        assert all("dev" in row and "ino" in row for row in matches)
    assert any(row["event"] == "claim_closed" and row["failed"] and "dev" in row and "ino" in row
               for row in records)
    original = winclaim.open_role
    observed = []
    probe = tmp_path / "log_probe.py"
    probe.write_text("""
import msvcrt, os, sys
try:
    fd = msvcrt.open_osfhandle(int(sys.argv[1]), os.O_WRONLY)
    os.write(fd, b'bad')
except OSError:
    pass
else:
    raise AssertionError('log inherited')
""")
    def observe(path, role):
        fd = original(path, role)
        if role == "event":
            observed.append(winclaim.handle(fd))
            assert not os.get_handle_inheritable(observed[-1])
            command = '"' + subprocess.list2cmdline([PYTHON, str(probe), str(observed[-1])]) + '"'
            hermetic.guard.admit_system(command, os.environ)
            assert os.system(command) == 0
        return fd
    with monkeypatch.context() as patch:
        patch.setattr(winclaim, "open_role", observe)
        coexist.event("log_inheritance_probe")
    assert observed
    assert all(json.loads(line) for line in log.read_text().splitlines())
    assert winclaim.entries() == {}


def test_w10_closed_standard_slots(children, tmp_path):
    result = tmp_path / "closed-slots.json"
    child = children('''
from pathlib import Path
for fd in range(3):
    os.close(fd)
d = claim.acquire('test', 'closed-stdio')
assert d.fd >= 3
m = claim.mutex(timeout=0)
assert m >= 3
claim.mutex_close(m)
a = claim.inherited(d, launch=True)
assert a.fd >= 3
a.failed()
claim.release(d)
for fd in range(3):
    try:
        os.fstat(fd)
    except OSError:
        pass
    else:
        raise AssertionError('standard slot leaked')
assert winclaim.entries() == {}
Path(sys.argv[1]).write_text('ok')
''', result)
    child.wait()
    assert result.read_text() == "ok"
    assert_free()


@pytest.mark.parametrize("role", ["holder", "admission", "cleanup", "adopted"])
def test_w10_concurrent_redirected_child_cannot_retain_conversion(children, tmp_path, role):
    result = tmp_path / "conversion-overlap.txt"
    child = children('''
from pathlib import Path
from unittest.mock import patch
import subprocess, threading
claim.bootstrap()
role = sys.argv[1]
path = claim.claim_path() if role in ('holder', 'adopted') else claim.mutex_path()
h = winclaim._open(path, 'holder' if role == 'adopted' else role)
os.close(0)
ready, finish = threading.Event(), threading.Event()
converted, errors = [], []
convert = winclaim._api().msvcrt.open_osfhandle
def paused(*args):
    fd = convert(*args)
    if args[0] == h:
        # The old implementation has installed the reservation as stdin here.
        ready.set()
        if not finish.wait(30):
            errors.append(TimeoutError('redirected spawn did not finish'))
    return fd
def worker():
    try:
        mode = os.O_RDONLY if role == 'admission' else os.O_RDWR
        converted.append(winclaim._crt(h, mode | os.O_BINARY | os.O_NOINHERIT, role=role))
    except BaseException as exc:
        errors.append(exc)
started = Path(sys.argv[2] + '.started')
done = Path(sys.argv[2] + '.done')
process = None
with patch.object(winclaim._api().msvcrt, 'open_osfhandle', paused):
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        try:
            assert ready.wait(30)
            # Leave stdin unspecified: Popen copies the process standard handle
            # when stdout is redirected, even with close_fds=True.
            process = subprocess.Popen([PYTHON, '-c',
                "from pathlib import Path; import sys, time; "
                "Path(sys.argv[1]).write_text('ready'); "
                "exec('while not Path(sys.argv[2]).exists(): time.sleep(.01)')",
                str(started), str(done)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                close_fds=True)
            deadline = time.monotonic() + 20
            while not started.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            assert started.exists() and process.poll() is None
        finally:
            finish.set()
            thread.join(30)
        assert not thread.is_alive() and not errors, errors
        assert len(converted) == 1
        claim.close(converted.pop())
        # Admission shares with admission; an exclusive cleanup probe proves
        # that neither the claim nor any gate reservation reached the child.
        probe = winclaim._open(path, 'holder' if role in ('holder', 'adopted') else 'cleanup')
        winclaim.close_handle(probe)
        assert process.poll() is None
        assert winclaim.entries() == {}
    finally:
        for fd in converted:
            claim.close(fd)
        done.write_text('done')
        if process is not None:
            try:
                assert process.wait(timeout=20) == 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=20)
                process.stdout.close()
                process.stderr.close()
Path(sys.argv[2]).write_text('ok')
''', role, result)
    child.wait()
    assert result.read_text() == "ok"
    assert_free()


@pytest.mark.parametrize("failure", ["convert", "reserve_first", "reserve_second", "reserve_third",
                                     "inherit", "register"])
def test_w10_crt_failure_consumes_every_resource(children, tmp_path, failure):
    result = tmp_path / "crt-failure.json"
    child = children('''
from pathlib import Path
from unittest.mock import patch
claim.bootstrap()
h = winclaim._open(claim.claim_path(), 'holder')
api = winclaim._api()
real_close, real_handle_close = os.close, api.CloseHandle
closed, raw_closed, opened = [], [], []
def close(fd):
    closed.append(fd)
    real_close(fd)
def handle_close(value):
    raw_closed.append(value)
    return real_handle_close(value)
def fail(*args):
    raise OSError('injected conversion')
failure = sys.argv[1]
for fd in range(3):
    real_close(fd)
real_open = os.open
def reserve(*args):
    if len(opened) == {'reserve_first': 0, 'reserve_second': 1, 'reserve_third': 2}[failure]:
        raise OSError('injected reservation')
    fd = real_open(*args)
    opened.append(fd)
    assert not os.get_inheritable(fd)
    return fd
real_set_role = winclaim.set_role
def register(*args, **kwargs):
    real_set_role(*args, **kwargs)
    raise OSError('injected registration')
where = ((os, 'open', reserve) if failure.startswith('reserve_') else
         (api.msvcrt, 'open_osfhandle', fail) if failure == 'convert' else
         (os, 'set_inheritable', fail) if failure == 'inherit' else
         (winclaim, 'set_role', register))
with patch.object(os, 'close', close), patch.object(api, 'CloseHandle', handle_close):
    with patch.object(*where):
        try:
            winclaim._crt(h, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
        except OSError:
            pass
        else:
            raise AssertionError('conversion did not fail')
assert len(closed) == len(set(closed))
assert raw_closed == ([] if failure in ('inherit', 'register') else [h])
assert len(closed) == {'convert': 3, 'reserve_first': 0, 'reserve_second': 1,
                       'reserve_third': 2, 'inherit': 4, 'register': 4}[failure]
assert winclaim.entries() == {}
for fd in range(3):
    try:
        os.fstat(fd)
    except OSError:
        pass
    else:
        raise AssertionError('standard slot leaked')
probe = winclaim._open(claim.claim_path(), 'holder')
winclaim.close_handle(probe)
Path(sys.argv[2]).write_text('ok')
''', failure, result)
    child.wait()
    assert result.read_text() == "ok"
    assert_free()


def test_w10_registry_reconciles_closed_and_reused_values(monkeypatch):
    descriptor = acquire()
    value, nonce = winclaim.handle(descriptor.fd), descriptor.offset()
    os.close(descriptor.fd)
    h = winclaim._open(claim.claim_path(), "holder")
    try:
        original = winclaim.handle
        with monkeypatch.context() as patch:
            patch.setattr(winclaim, "handle", lambda fd: value + 4 if fd == descriptor.fd else original(fd))
            assert descriptor.fd not in winclaim.entries()
        assert not winclaim.managed(h)
        assert winclaim._api().SetFilePointerEx(h, nonce, None, 0)
        set_token(monkeypatch, h)
        adopted = claim.received()
        h = None
        claim.close(adopted)
    finally:
        if h is not None:
            winclaim.close_handle(h)
        winclaim.forget(descriptor.fd)
    assert_free()


def test_w10_real_handle_value_reuse_allows_new_adoption(monkeypatch):
    descriptor = acquire()
    value = winclaim.handle(descriptor.fd)
    nonce = descriptor.offset()
    os.close(descriptor.fd)
    h = winclaim._open(claim.claim_path(), "holder")
    if h != value:
        winclaim.close_handle(h)
        winclaim.forget(descriptor.fd)
        pytest.skip("Windows did not reuse handle value")
    assert winclaim._api().SetFilePointerEx(h, nonce, None, 0)
    try:
        set_token(monkeypatch, h)
        adopted = claim.received()
        assert claim.inherited(adopted) == adopted
        claim.close(adopted)
        h = None
    finally:
        if h is not None:
            winclaim.close_handle(h)
        winclaim.forget(descriptor.fd)


def test_w11_adopt_transfers_and_revalidates(children):
    with held() as descriptor:
        child = children('''
original = winproc.read_launch()['handles']['claim']
d = claim.received()
assert 'claim' not in winclaim.received_record()['handles']
try:
    os.get_handle_inheritable(original)
except OSError:
    pass
else:
    raise AssertionError('original not closed')
assert not os.get_handle_inheritable(winclaim.handle(d.fd))
assert claim.received().fd == d.fd
copy = os.dup(d.fd)
os.close(copy)
assert claim.received().fd == d.fd
claim.close(d)
assert claim._adopted is None
assert winclaim.entries() == {}
try:
    claim.received()
except claim.ClaimError as exc:
    assert 'missing' in str(exc)
else:
    raise AssertionError('repeated closed adoption')
print('ok', flush=True)
''', inherited=descriptor)
        assert child.line() == "ok"
        child.wait()


@pytest.mark.parametrize("deterministic", [False, True])
def test_w11_cached_adoption_refuses_successor_fd_reuse(monkeypatch, deterministic):
    original = acquire()
    token = winclaim.duplicate(winclaim.handle(original.fd), inheritable=False)
    set_token(monkeypatch, token)
    adopted = claim.received()
    adopted_fd = adopted.fd
    claim.release(original)
    os.close(adopted_fd)
    # Reserve lower CRT slots so the successor deterministically occupies the cached fd.
    fillers = []
    while True:
        fd = os.open(os.devnull, os.O_RDONLY)
        fillers.append(fd)
        if fd == adopted_fd:
            break
        assert fd < adopted_fd
    successor = acquire()
    os.close(fillers.pop())
    reused = winclaim._crt(winclaim.duplicate(winclaim.handle(successor.fd)),
                           os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
    try:
        if reused != adopted_fd:
            if not deterministic:
                claim._adopted = None
                pytest.skip("CRT did not reuse adopted fd")
            os.dup2(successor.fd, adopted_fd)
            claim.close(reused)
            reused = adopted_fd
            winclaim.set_role(reused, "holder")
        if deterministic:
            # Even numeric reconciliation and the old role cannot bypass holder identity.
            winclaim.set_role(reused, "adopted")
        with pytest.raises(claim.ClaimError, match="adopted claim descriptor"):
            claim.received()
        assert claim._adopted is None
        assert claim.inherited(reused).fd == reused
        assert claim.inherited(successor).fd == successor.fd
    finally:
        claim.close(reused)
        claim.release(successor)
        for fd in fillers:
            os.close(fd)
        winclaim.forget(adopted_fd)
        claim._adopted = None


@pytest.mark.parametrize("managed", ["stdin", "acquired", "adopted"])
def test_w11_managed_tokens_are_refused_without_closing(monkeypatch, managed):
    with held() as descriptor:
        adopted = None
        if managed == "stdin":
            value = winclaim.handle(0)
        elif managed == "acquired":
            value = winclaim.handle(descriptor.fd)
        else:
            value = winclaim.duplicate(winclaim.handle(descriptor.fd), inheritable=False)
            set_token(monkeypatch, value)
            adopted = claim.received()
            value = winclaim.handle(adopted.fd)
        previous_inheritance = os.get_handle_inheritable(value)
        try:
            set_token(monkeypatch, value)
            with pytest.raises(claim.ClaimError, match="already managed"):
                claim.received()
            assert os.get_handle_inheritable(value) == previous_inheritance
            assert claim.inherited(descriptor) == descriptor
            if adopted is not None:
                assert claim.inherited(adopted) == adopted
        finally:
            if adopted is not None:
                claim.close(adopted)


def test_w11_claim_redirected_stdin_is_preserved(children):
    claim.bootstrap()
    claim.claim_path().write_bytes(b"stdin content")
    child = children('''
fd = os.open(claim.claim_path(), os.O_RDONLY)
os.dup2(fd, 0)
os.close(fd)
value = winclaim.handle(0)
winproc.read_launch = lambda: {'handles': {'claim': value}}
try:
    claim.received()
except claim.ClaimError as exc:
    assert 'already managed' in str(exc)
else:
    raise AssertionError('stdin adopted')
assert os.read(0, 13) == b'stdin content'
print('ok', flush=True)
''')
    assert child.line() == "ok"
    child.wait()


@pytest.mark.parametrize("token", ["", "0", "-1", "abc", "123 4", "999999999"])
def test_w11_invalid_token_consumed(monkeypatch, token):
    set_token(monkeypatch, token)
    with pytest.raises(claim.ClaimError):
        claim.received()
    assert "claim" not in winclaim.received_record()["handles"]
    with pytest.raises(claim.ClaimError, match="missing"):
        claim.received()


def test_w11_not_claim_and_probe_error_preserve_original(monkeypatch, tmp_path):
    claim.bootstrap()
    other = tmp_path / "other"
    other.touch()
    for path, probe_error in ((other, False), (claim.claim_path(), True)):
        h = winclaim.open_handle(path, winclaim.COMMON_ACCESS | 3, 3)
        try:
            with monkeypatch.context() as patch:
                set_token(patch, h)
                if probe_error:
                    patch.setattr(winclaim, "names", lambda *args: (_ for _ in ()).throw(claim.ClaimError("probe failed")))
                with pytest.raises(claim.ClaimError):
                    claim.received()
            assert winclaim._api().GetFileType(h) == 1
            with pytest.raises(claim.ClaimError, match="missing"):
                claim.received()
        finally:
            winclaim.close_handle(h)


def test_w11_duplicate_failure_closes_nothing(monkeypatch):
    with held() as descriptor:
        h = winclaim.duplicate(winclaim.handle(descriptor.fd), inheritable=False)
        try:
            set_token(monkeypatch, h)
            def fail(*args):
                C.set_last_error(6)
                return False
            with monkeypatch.context() as patch:
                patch.setattr(winclaim._api(), "DuplicateHandle", fail)
                with pytest.raises(claim.ClaimError, match="DuplicateHandle"):
                    claim.received()
            assert not os.get_handle_inheritable(h)
            assert claim.inherited(descriptor) == descriptor
        finally:
            winclaim.close_handle(h)


def test_w11_stale_record_closes_classified_original(monkeypatch):
    descriptor = acquire()
    h = winclaim.duplicate(winclaim.handle(descriptor.fd), inheritable=False)
    claim.release(descriptor)
    set_token(monkeypatch, h)
    with pytest.raises(claim.ClaimError, match="holder unknown"):
        claim.received()
    with pytest.raises(OSError):
        os.get_handle_inheritable(h)
    assert winclaim.entries() == {}
    assert_free()


@pytest.mark.parametrize("failure", ["convert", "reserve", "validate", "classify"])
def test_w11_adoption_unwinds_copy_and_original(children, tmp_path, failure):
    result = tmp_path / "adopt-failure.json"
    with held() as descriptor:
        child = children('''
from pathlib import Path
from unittest.mock import patch
api = winclaim._api()
h = winproc.read_launch()['handles']['claim']
winclaim.received_record()
failure = sys.argv[1]
real_close, real_handle_close = os.close, api.CloseHandle
closed, raw_closed, copies = [], [], []
real_duplicate = winclaim.duplicate
def duplicate(*args, **kwargs):
    value = real_duplicate(*args, **kwargs)
    copies.append(value)
    return value
def close(fd):
    closed.append(fd)
    real_close(fd)
def handle_close(value):
    raw_closed.append(value)
    return real_handle_close(value)
def fail(*args, **kwargs):
    raise claim.ClaimError('injected adoption')
if failure == 'reserve':
    # One vacant slot exercises reservation without losing child diagnostics.
    real_close(0)
real_open = os.open
reservations = []
def reserve(*args):
    fd = real_open(*args)
    reservations.append(fd)
    assert not os.get_inheritable(fd)
    return fd
# Fail conversion after the vacant stdin slot has been reserved.
where = {'convert': (api.msvcrt, 'open_osfhandle'), 'reserve': (api.msvcrt, 'open_osfhandle'),
         'validate': (claim, 'inherited'), 'classify': (winclaim, 'names')}[failure]
with patch.object(os, 'close', close), patch.object(api, 'CloseHandle', handle_close), patch.object(winclaim, 'duplicate', duplicate):
    with patch.object(os, 'open', reserve if failure == 'reserve' else real_open), patch.object(*where, fail):
        try:
            claim.received()
        except claim.ClaimError:
            pass
        else:
            raise AssertionError('adoption did not fail')
assert winclaim.entries() == {}
assert len(closed) == len(set(closed))
assert raw_closed.count(h) == (0 if failure == 'classify' else 1)
assert len(copies) == 1
assert raw_closed.count(copies[0]) == (1 if failure in ('convert', 'reserve', 'classify') else 0)
# Classification also opens and closes an identity probe; the ownership
# assertions above count only the original and its adoption copy.
if failure == 'classify':
    assert len(raw_closed) == 1 and not closed
    assert not os.get_handle_inheritable(h)
else:
    try:
        os.get_handle_inheritable(h)
    except OSError:
        pass
    else:
        raise AssertionError('original leaked')
    assert len(closed) == (0 if failure == 'convert' else 1)
    if failure == 'reserve':
        assert closed == reservations == [0]
try:
    claim.received()
except claim.ClaimError as exc:
    assert 'missing' in str(exc)
else:
    raise AssertionError('failed token retained')
Path(sys.argv[2]).write_text('ok')
''', failure, result, inherited=descriptor)
        child.wait()
        assert result.read_text() == "ok"
    assert_free()


@pytest.mark.wall_clock
@pytest.mark.parametrize("release_admission", [False, True])
def test_w12_stale_cleanup_record_does_not_classify_admission(children, release_admission):
    mutex = claim.mutex(timeout=0)
    claim.mutex_close(mutex)
    child = children('''
fd = claim._shared()
print('admission', flush=True)
input()
claim.Admission(claim.ClaimDescriptor(-1), fd).failed()
''')
    assert child.line() == "admission"
    if release_admission:
        timer = threading.Timer(.15, child.send)
        timer.start()
        try:
            mutex = claim.mutex(timeout=2)
            claim.mutex_release(mutex)
        finally:
            timer.join()
        child.wait()
    else:
        started = time.monotonic()
        with pytest.raises(claim.ClaimBusy, match="admission in progress"):
            claim.mutex(timeout=.2)
        assert time.monotonic() - started >= .15
        child.send()
        child.wait()


@pytest.mark.wall_clock
def test_w12_live_cleanup_refuses_at_once(children):
    child = children('''
fd = claim.mutex(timeout=0)
print('cleanup', flush=True)
input()
claim.mutex_release(fd)
''')
    assert child.line() == "cleanup"
    started = time.monotonic()
    with pytest.raises(claim.ClaimBusy, match=f"cleanup in progress.*{child.process.pid}"):
        claim.mutex(timeout=2)
    assert time.monotonic() - started < 1
    with pytest.raises(claim.ClaimBusy, match="cleanup in progress"):
        acquire()
    child.send()
    child.wait()


def test_process_identity_liveness_and_retained_object(children):
    current = claim.process_started(os.getpid())
    assert current.startswith("win:") and current[4:].isdecimal()
    assert current == claim.process_started(os.getpid())
    child = children('''
print(claim.process_started(os.getpid()), flush=True)
input()
''')
    started = child.line()
    assert started != current
    assert claim.process_started(child.process.pid) == started
    assert winclaim.process_info(child.process.pid) == (started, False)
    child.send()
    child.wait()
    assert winclaim.process_info(child.process.pid) == (started, True)
    assert claim.process_matches(child.process.pid, started)


@pytest.mark.parametrize("error, exited", [(87, True), (5, False), (6, False)])
def test_openprocess_failure_contract(monkeypatch, error, exited):
    def fail(*args):
        C.set_last_error(error)
        return None
    monkeypatch.setattr(winclaim._api(), "OpenProcess", fail)
    if exited:
        assert winclaim.process_info(4242) == (None, True)
        assert claim._exited(4242)
    else:
        with pytest.raises(claim.ClaimError, match="cannot identify pid 4242"):
            claim._exited(4242)


def test_wait_failed_is_an_error(monkeypatch):
    def fail(*args):
        C.set_last_error(6)
        return winclaim.WAIT_FAILED
    monkeypatch.setattr(winclaim._api(), "WaitForSingleObject", fail)
    with pytest.raises(claim.ClaimError, match="cannot wait"):
        claim._exited(os.getpid())


def test_process_queries_do_not_leak_handles():
    api = winclaim._api()
    def count():
        result = W.DWORD()
        assert api.GetProcessHandleCount(api.GetCurrentProcess(), C.byref(result))
        return result.value
    claim.process_started(os.getpid())
    before = count()
    for _ in range(1000):
        claim.process_started(os.getpid())
        assert not claim._exited(os.getpid())
    assert count() == before


def test_real_protected_process_is_not_exited():
    argv = ["tasklist", "/FI", "IMAGENAME eq csrss.exe", "/FO", "CSV", "/NH"]
    hermetic.guard.admit(argv, os.environ)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    import csv
    rows = list(csv.reader(result.stdout.splitlines()))
    pids = [int(row[1]) for row in rows if len(row) > 1 and row[0].casefold() == "csrss.exe"]
    if not pids:
        pytest.skip("csrss.exe pid unavailable")
    try:
        winclaim.process_info(pids[0])
    except claim.ClaimError as exc:
        assert "cannot identify" in str(exc)
    else:
        pytest.skip("this account can query csrss.exe")


def test_retained_dead_cleanup_publisher_not_attributed(children, monkeypatch):
    child = children('''
fd = claim.mutex(timeout=0)
print(claim.process_started(os.getpid()), flush=True)
time.sleep(60)
''')
    started = child.line()
    child.process.kill()
    child.process.wait(timeout=10)
    calls = []
    read = winclaim.read_record
    info = winclaim.process_info
    def recorded_read(*args, **kwargs):
        data = read(*args, **kwargs)
        calls.append(("read", data))
        return data
    def recorded_info(pid):
        result = info(pid)
        calls.append(("process", pid, result))
        return result
    with raw(claim.mutex_path(), "cleanup"):
        with monkeypatch.context() as patch:
            patch.setattr(winclaim, "read_record", recorded_read)
            patch.setattr(winclaim, "process_info", recorded_info)
            with pytest.raises(claim.ClaimBusy, match="cleanup in progress.*holder unknown") as failure:
                claim.mutex(timeout=0)
            assert str(child.process.pid) not in str(failure.value)
        assert any(row[0] == "read" and str(child.process.pid).encode() in row[1] for row in calls)
        assert ("process", child.process.pid, (started, True)) in calls


def test_unreadable_gate_does_not_probe_publisher(monkeypatch):
    claim.bootstrap()
    with raw(claim.mutex_path(), access=1, share=0):
        monkeypatch.setattr(winclaim, "process_info", lambda pid: pytest.fail("unreadable publisher queried"))
        with pytest.raises(claim.ClaimBusy, match="cleanup in progress.*holder unknown"):
            claim.mutex(timeout=0)
        with pytest.raises(claim.ClaimBusy, match="cleanup in progress.*holder unknown"):
            acquire()


def test_serial_verdict_is_exclusive_without_discovery(monkeypatch):
    monkeypatch.setattr(claim, "openers", lambda *a, **k: pytest.fail("Windows device discovery"))
    verdicts = claim.check_devices(claim.Devices(cat_port="com41", ptt_port=r"\\.\COM42",
                                                capture="Windows endpoint, MME"))
    assert [(v.endpoint, v.verdict, v.nodes) for v in verdicts] == [
        ("cat", "exclusive", (r"\\.\COM41",)), ("ptt", "exclusive", (r"\\.\COM42",)),
        ("capture", "cooperative", ()), ("playback", "cooperative", ())]
    delegated = claim.check_devices(claim.Devices(cat="rigctld", cat_port="COM41"))
    assert delegated[0].verdict == "delegated"


@pytest.mark.parametrize("inspectable", [True, False])
def test_foreign_com41_open_is_refused_under_claim(children, monkeypatch, inspectable):
    from hfmodem.tests.hermetic import allow_virtual_serial
    from hfmodem.core.ptt import PttError
    try:
        target = winclaim.serial_target("COM41")
    except claim.ClaimError:
        pytest.skip("com0com COM41 is absent")
    if "cnc" not in target.casefold() and "com0com" not in target.casefold():
        pytest.skip("COM41 is not a com0com device")
    allow_virtual_serial('COM41')
    child = children('''
import serial
port = serial.Serial(baudrate=38400, write_timeout=.25)
port.port, port.rts, port.dtr = 'COM41', False, False
port.open()
print('held', flush=True)
input()
port.close()
''')
    assert child.line() == "held"
    assert claim.check_devices(claim.Devices(ptt_port="COM41"))[0].verdict == "exclusive"
    expected = f"COM41 is open in pid {child.process.pid} ({Path(PYTHON).name})"
    if not inspectable:
        real_open = winproc.open
        def denied(pid, rights=winproc.IDENTIFY):
            if pid == child.process.pid and rights & 0x0040:
                raise winproc.ProcessAccessError(pid, rights)
            return real_open(pid, rights)
        monkeypatch.setattr(winproc, "open", denied)
        expected = "COM41 is open in another process"
    with pytest.raises(PttError) as failure:
        acquire(devices=claim.Devices(ptt_port="COM41"), prove_line=True,
                ptt_factory=claim.line_factory("COM41"))
    assert str(failure.value).endswith(expected)
    assert_free()
    assert child.process.poll() is None
    child.send()
    child.wait()


@pytest.mark.parametrize("verb, args", [("mutex", ["9"]), ("mutex_release", ["9"]),
    ("acquire", ["9", "--kind", "test", "--operation", "refuse"]),
    ("inherited", ["9"]), ("close", ["9"]), ("release", ["9"]),
    ("spawn", ["--", "never.exe"]), ("claim_spawn", ["--", "never.exe"])])
def test_w13_cli_fd_and_native_verbs_refuse(verb, args, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("spawn on Windows"))
    monkeypatch.setattr(winclaim, "open_role", lambda *a, **k: pytest.fail("open before refusal"))
    assert claim.main([verb, *args]) == 1
    assert "Windows" in capsys.readouterr().err
    assert not claim.claim_path().exists()


def test_path_and_bootstrap_work(capsys):
    for verb in ("path", "bootstrap"):
        assert claim.main([verb]) == 0
        assert str(claim.claim_path()) in capsys.readouterr().out
    with held():
        assert claim.main(["path"]) == 0


def test_no_windows_kill_or_claim_pointer_io(monkeypatch):
    tree = ast.parse(Path(winclaim.__file__).read_text())
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    assert node.func.attr not in ("kill", "read", "write")
                    if node.func.attr == "lseek":
                        assert function.name in ("publish", "offset")
                if node.func.attr in ("ReadFile", "SetFilePointerEx"):
                    assert function.name == "read_record"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_write":
                assert function.name == "publish"
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("os.kill is fatal on Windows"))
    monkeypatch.setattr(claim, "fcntl", SimpleNamespace())
    monkeypatch.setattr(claim, "pwd", SimpleNamespace())
    with held() as descriptor:
        assert claim.inherited(descriptor) == descriptor
        assert not claim._exited(os.getpid())
        with pytest.raises(claim.ClaimBusy):
            acquire()
        mutex = claim.mutex(timeout=0)
        assert claim.verify_cleanup(mutex) == mutex
        claim.mutex_release(mutex)
    assert not any(isinstance(node, ast.Attribute) and node.attr in
                   ("CreateJobObjectW", "AssignProcessToJobObject", "SetInformationJobObject")
                   for node in ast.walk(tree))


@pytest.mark.parametrize("point", ["after_claim_flock", "before_holder_line", "after_holder_line",
                                   "after_device_check", "during_line_proof"])
def test_failed_cleanup_acquisition_preserves_mutex(monkeypatch, point):
    mutex = claim.mutex(timeout=0)
    before = winclaim.read_record(claim.mutex_path())
    try:
        with monkeypatch.context() as patch:
            patch.setenv("HFMODEM_COEXIST_FAULT", point)
            with pytest.raises(RuntimeError, match=point):
                acquire(cleanup=mutex, prove_line=True,
                        ptt_factory=lambda: SimpleNamespace(assert_=lambda value: None,
                            sense=lambda: False, release=lambda: None))
        assert claim.claim_path().read_bytes() == b""
        assert winclaim.read_record(claim.mutex_path()) == before
        assert claim.verify_cleanup(mutex) == mutex
        assert set(winclaim.entries()) == {mutex}
        with pytest.raises(claim.ClaimBusy, match="cleanup in progress"):
            acquire()
    finally:
        claim.mutex_release(mutex)
    assert_free()


@pytest.mark.parametrize("point", ["before_holder_line", "after_holder_line", "after_mutex"])
def test_cleanup_publication_failure_unwinds(monkeypatch, point):
    with monkeypatch.context() as patch:
        patch.setenv("HFMODEM_COEXIST_FAULT", point)
        with pytest.raises(RuntimeError, match=point):
            claim.mutex(timeout=0)
    assert claim.mutex_path().read_bytes() == b""
    assert winclaim.entries() == {}
    mutex = claim.mutex(timeout=0)
    claim.mutex_release(mutex)


def test_cleanup_release_invalidates_retained_dup():
    mutex = claim.mutex(timeout=0)
    copy = os.dup(mutex)
    try:
        claim.mutex_release(mutex)
        assert mutex not in winclaim.entries()
        assert winclaim.read_record(claim.mutex_path()) == b""
        with pytest.raises(claim.ClaimError, match="another description"):
            claim.verify_cleanup(copy)
        with pytest.raises(claim.ClaimBusy, match="cleanup in progress.*holder unknown"):
            claim.mutex(timeout=0)
    finally:
        os.close(copy)
    mutex = claim.mutex(timeout=0)
    claim.mutex_close(mutex)


@pytest.mark.parametrize("cleanup", [False, True])
def test_release_refuses_changed_process_identity(monkeypatch, cleanup):
    fd = claim.mutex(timeout=0) if cleanup else acquire().fd
    path = claim.mutex_path() if cleanup else claim.claim_path()
    before = winclaim.read_record(path)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(claim, "process_started", lambda pid: "win:reused")
            with pytest.raises(claim.ClaimError, match="only the acquirer"):
                claim.mutex_release(fd) if cleanup else claim.release(fd)
        assert winclaim.read_record(path) == before
        assert fd in winclaim.entries()
    finally:
        claim.mutex_release(fd) if cleanup else claim.release(fd)


def test_unrestricted_spawns_absent_from_product_paths():
    root = Path(claim.__file__).resolve().parents[3]
    for source in root.rglob("*.py"):
        if "tests" in source.parts or "__pycache__" in source.parts:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                assert node.func.attr not in ("system", "popen")
                assert not node.func.attr.startswith("spawn")
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                for keyword in node.keywords:
                    if keyword.arg == "close_fds":
                        assert (isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                                or ast.unparse(keyword.value) == "os.name == 'nt'"), (source, node.lineno)


@pytest.mark.parametrize("entry", ["spawn_native", "_spawn_native", "shell_spawn"])
def test_shell_and_cleanup_entries_refuse_before_resources(monkeypatch, entry):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Windows native spawn"))
    monkeypatch.setattr(winclaim, "open_role", lambda *a, **k: pytest.fail("Windows native open"))
    with pytest.raises(claim.ClaimError, match="native launch from the shell is not supported on Windows"):
        if entry == "shell_spawn":
            native.shell_spawn(None)
        else:
            getattr(native, entry)([], claim_fd=-1, cleanup_fd=8)


def test_windows_command_redaction(monkeypatch):
    password, token = "one", "two"
    monkeypatch.setattr(winclaim, "command",
                        lambda: f'python.exe --password={password} --mail-token {token} keep')
    assert claim._command(os.getpid()) == 'python.exe --password=<redacted> --mail-token <redacted> keep'


@pytest.mark.parametrize("role", ["holder", "admission", "ordinary", "duplicate", "closed"])
def test_mutex_close_rejects_unauthorized_descriptors(role):
    claim.bootstrap()
    mutex = None
    if role == "holder":
        fd = acquire().fd
    elif role == "admission":
        fd = claim._shared()
    elif role == "duplicate":
        mutex = claim.mutex(timeout=0)
        fd = os.dup(mutex)
    else:
        fd = os.open(claim.mutex_path(), os.O_RDWR)
        if role == "closed":
            os.close(fd)
    before = winclaim.entries()
    try:
        with pytest.raises(claim.ClaimError):
            claim.mutex_close(fd)
        assert winclaim.entries() == before
        if role != "closed":
            os.fstat(fd)
        if mutex is not None:
            assert claim.verify_cleanup(mutex) == mutex
    finally:
        if role != "closed":
            claim.close(fd)
        if mutex is not None:
            claim.mutex_close(mutex)
    assert_free()


def test_mutex_close_validates_and_retires_under_lock(monkeypatch):
    mutex = claim.mutex(timeout=0)
    verify, close = winclaim.verify_cleanup, os.close
    calls = []
    def checked(fd):
        assert winclaim.lock._is_owned()
        calls.append("verify")
        return verify(fd)
    def closed(fd):
        if fd == mutex:
            assert winclaim.lock._is_owned()
            assert fd not in winclaim.entries()
            calls.append("close")
        return close(fd)
    with monkeypatch.context() as patch:
        patch.setattr(winclaim, "verify_cleanup", checked)
        patch.setattr(os, "close", closed)
        claim.mutex_close(mutex)
    assert calls == ["verify", "close"]
    with pytest.raises(claim.ClaimError):
        claim.mutex_close(mutex)
    assert winclaim.entries() == {}


def await_condition(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    pytest.fail('condition did not become true')


def cleanup_available():
    mutex = claim.mutex(timeout=0)
    claim.mutex_release(mutex)


def test_spawn_received_extra_round_trip_and_channels(children):
    read, write = os.pipe()
    try:
        os.write(write, b'owner')
        with held() as descriptor:
            child = children('''
record = winproc.read_launch()
assert record['phase'] == 'resuming'
assert record['child']['pid'] == os.getpid()
raw_claim, raw_owner = record['handles']['claim'], record['handles']['owner']
assert not os.get_handle_inheritable(raw_claim)
assert not os.get_handle_inheritable(raw_owner)
reads = []
read_record = winproc.read_launch
def counted(*args, **kwargs):
    reads.append(args)
    return read_record(*args, **kwargs)
winproc.read_launch = counted
d = claim.received()
assert claim.received() == d
owner = claim.extra('owner')
assert claim.extra('owner') == owner
assert os.read(owner, 5) == b'owner'
assert not os.get_inheritable(owner)
# Channel publication reads too; only the no-argument token read happens once.
assert reads.count(()) == 1
assert winproc._channels is not None
assert winproc._target_grace(winproc._channels.entry) == 2
for kind in ('stop', 'int'):
    event = winproc._api().OpenEventW(2, False, winproc.event_name(kind, winproc._channels.entry))
    assert event
    winclaim.close_handle(event)
assert winproc.read_launch()['grace'] == 2
claim.close(owner)
try:
    claim.extra('owner')
except claim.ClaimError:
    pass
else:
    raise AssertionError('closed extra replayed')
claim.close(d)
try:
    claim.received()
except claim.ClaimError:
    pass
else:
    raise AssertionError('closed claim replayed')
print('ok', flush=True)
''', inherited=descriptor, handles={'owner': read})
            assert child.line() == 'ok'
            child.wait()
    finally:
        os.close(read)
        os.close(write)


@pytest.mark.parametrize('role', ['python', 'supervisor', 'one-shot', 'cleanup', 'inspector', 'recorder'])
@pytest.mark.parametrize('own_session', [False, True])
def test_every_role_keeps_claim_noninheritable(children, monkeypatch, role, own_session):
    suspended, seen = winproc.python_suspended, []
    def checked(args, **kwargs):
        seen.append(kwargs['creationflags'])
        if role == 'recorder':
            assert 'RIG_CLAIMED' not in kwargs['env']
        else:
            assert kwargs['env']['RIG_CLAIMED'] == str(claim.claim_path())
        return suspended(args, **kwargs)
    monkeypatch.setattr(winproc, 'python_suspended', checked)
    with held() as descriptor:
        child = children('''
record = winproc.read_launch()
if record['role'] == 'recorder':
    assert 'claim' not in record['handles']
    try:
        claim.received()
    except claim.ClaimError:
        pass
    else:
        raise AssertionError('recorder received a claim')
else:
    assert not os.get_handle_inheritable(record['handles']['claim'])
    d = claim.received()
    assert not os.get_handle_inheritable(winclaim.handle(d.fd))
    assert claim.received() == d
    claim.close(d)
print('ok', flush=True)
''', inherited=descriptor, role=role, own_session=own_session)
        assert child.line() == 'ok'
        child.wait()
    expected = subprocess.CREATE_NEW_PROCESS_GROUP if own_session or role == 'supervisor' else 0
    if role == 'supervisor':
        expected |= subprocess.CREATE_NO_WINDOW
    assert seen == [expected]


@pytest.mark.parametrize('role', ['python', 'supervisor', 'one-shot'])
def test_spawn_refuses_without_live_share_admission(role):
    with held() as descriptor:
        with pytest.raises(claim.ClaimError, match='live share-mode admission'):
            claim.spawn(descriptor, [sys.executable, '-c', 'raise AssertionError()'], role=role)
        for method in ('failed', 'close'):
            admission = claim.inherited(descriptor, launch=True)
            if method == 'failed':
                admission.failed()
            else:
                os.close(admission.fd)
            with pytest.raises(claim.ClaimError, match='live share-mode admission'):
                claim.spawn(admission, [sys.executable, '-c', 'raise AssertionError()'], role=role)
    assert not (claim.state_dir() / 'rig.claim.launch.d').exists()


@pytest.mark.parametrize('role', ['python', 'supervisor', 'one-shot'])
@pytest.mark.parametrize('deterministic', [False, True])
def test_spawn_refuses_stale_admission_reused_by_gate_reader(monkeypatch, role, deterministic):
    with held() as descriptor:
        admission = claim.inherited(descriptor, launch=True)
        previous = winclaim.entries()[admission.fd]
        os.close(admission.fd)
        reader = os.open(claim.mutex_path(), os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
        mutex = None
        try:
            if deterministic:
                # Model both reused numbers without depending on the handle allocator.
                if reader != admission.fd:
                    os.dup2(reader, admission.fd, inheritable=False)
                    os.close(reader)
                    reader = admission.fd
                winclaim._registry[reader] = winclaim.Entry(winclaim.handle(reader), previous.role)
            elif reader != admission.fd or winclaim.handle(reader) != previous.handle:
                pytest.skip('Windows did not reuse both numeric slots')
            assert winclaim.entries()[reader].role == 'admission'
            claim._check_inode(reader, claim.mutex_path())
            mutex = claim.mutex(timeout=0)
            monkeypatch.setattr(winproc, 'python_suspended',
                                lambda *a, **k: pytest.fail('created a process during cleanup'))
            with pytest.raises(claim.ClaimBusy, match='cleanup in progress'):
                claim.spawn(admission, [sys.executable, '-c', 'raise AssertionError()'], role=role)
            assert claim.verify_cleanup(mutex) == mutex
            assert not (claim.state_dir() / 'rig.claim.launch.d').exists()
        finally:
            if mutex is not None:
                claim.mutex_close(mutex)
            claim.close(reader)
            winclaim.forget(admission.fd)


@pytest.mark.wall_clock
def test_python_grandchild_inherits_no_claim(children, tmp_path):
    ready = tmp_path / 'grandchild'
    descriptor = acquire()
    grandchild = None
    try:
        child = children('''
import subprocess
from pathlib import Path
d = claim.received()
script = ''' + repr(PRELUDE + PROBE + "\nfrom pathlib import Path\nPath(sys.argv[2]).touch()\ntime.sleep(60)\n") + '''
p = subprocess.Popen([PYTHON, '-c', script, str(winclaim.handle(d.fd)), sys.argv[1]],
                     close_fds=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
print(p.pid, flush=True)
time.sleep(60)
''', ready, inherited=descriptor)
        grandchild = winproc.open(int(child.line()), winproc.END)
        assert grandchild is not None
        await_condition(ready.exists)
        assert not grandchild.wait()
        claim.close(descriptor)
        descriptor = None
        with pytest.raises(claim.ClaimBusy):
            acquire()
        child.process.kill()
        child.process.wait(timeout=10)
        assert_free()
        cleanup_available()
        assert not grandchild.wait()
    finally:
        if descriptor is not None:
            claim.close(descriptor)
        if grandchild is not None:
            grandchild.terminate()
            assert grandchild.wait(5)
            grandchild.close()


@pytest.mark.wall_clock
def test_acknowledgement_wins_while_launcher_paused_after_resume(children, tmp_path, monkeypatch):
    barriers = tmp_path / 'barriers'
    barriers.mkdir()
    (barriers / 'launch_after_resume').touch()
    with held() as descriptor:
        monkeypatch.setenv('HFMODEM_COEXIST_BARRIERS', str(barriers))
        launches, failures = [], []
        def launch():
            try:
                launches.append(spawn_claim(descriptor, [sys.executable, '-c', PRELUDE + '''
a = claim.received(launch=True)
a.opened()
input()
'''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            except BaseException as exc:
                failures.append(exc)
        thread = threading.Thread(target=launch)
        thread.start()
        process = entry = None
        try:
            await_condition(lambda: (barriers / 'launch_after_resume.reached').exists())
            def acknowledged():
                records, _ = winproc.live_launches()
                return next((r for r in records if r['phase'] == 'acknowledged'), None)
            record = await_condition(acknowledged)
            entry = winproc.open(record['child']['pid'])
            path = winproc._path('launch', entry)
            before = path.read_bytes()
            (barriers / 'launch_after_resume.release').touch()
            thread.join(10)
            assert not thread.is_alive() and not failures
            launch, = launches
            process = launch.process
            assert path.read_bytes() == before
            assert process.stdout.readline() == 'CLAIM ADMITTED\n'
            cleanup_available()
            process.stdin.write('go\n')
            process.stdin.flush()
            assert process.wait(timeout=10) == 0
            assert winproc.read_launch(launch.child)['phase'] == 'acknowledged'
            winproc.advance_launch(launch.child, 'exited')
            assert winproc.read_launch(launch.child)['phase'] == 'exited'
        finally:
            (barriers / 'launch_after_resume.release').touch()
            thread.join(10)
            for launch in launches:
                process = launch.process
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                launch.child.close()
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
            if entry is not None:
                entry.close()


@pytest.mark.parametrize('ending', ['acknowledged', 'failed', 'terminated'])
def test_record_admission_lifecycle_and_signalled_pruning(children, ending):
    with held() as descriptor:
        child = children('''
a = claim.received(launch=True)
assert a.fd is None
assert claim.received(launch=True) is a
print('resuming', flush=True)
input()
if sys.argv[1] == 'acknowledged':
    a.opened()
else:
    a.failed()
    a.failed()
    assert winproc.read_launch()['phase'] == 'resuming'
    print('failed', flush=True)
input()
''', ending, inherited=descriptor)
        assert child.line() == 'resuming'
        entry = child.launch.child
        with pytest.raises(claim.ClaimBusy, match=f'admission in progress.*{entry.pid}.*python'):
            claim.mutex(timeout=0)
        if ending != 'terminated':
            child.send()
            assert child.line() == ('CLAIM ADMITTED' if ending == 'acknowledged' else 'failed')
            if ending == 'acknowledged':
                cleanup_available()
            else:
                with pytest.raises(claim.ClaimBusy, match=f'admission in progress.*{entry.pid}'):
                    claim.mutex(timeout=0)
        child.process.kill()
        child.process.wait(timeout=10)
        assert int(child.process._handle) and entry.wait()
        path = winproc._path('launch', entry)
        assert path.exists()
        cleanup_available()
        assert not path.exists()


@pytest.mark.parametrize('role', ['cleanup', 'inspector', 'recorder'])
def test_nonadmitting_roles_launch_under_cleanup_and_never_admit(children, role):
    with held() as descriptor:
        mutex = claim.mutex(timeout=0)
        try:
            child = children("print('ready', flush=True); input()", inherited=descriptor, role=role)
            assert child.line() == 'ready'
            assert winproc.read_launch(child.launch.child)['phase'] == 'resuming'
            assert claim.verify_cleanup(mutex) == mutex
        finally:
            claim.mutex_release(mutex)
        cleanup_available()
        child.send()
        child.wait()


@pytest.mark.wall_clock
@pytest.mark.parametrize('point, phase, pushed', [
    ('launch_after_creation', None, set()),
    ('launch_after_record', 'created', set()),
    ('launch_after_push_claim', 'created', {'claim'}),
    ('launch_after_push_owner', 'created', {'claim', 'owner'}),
    ('launch_after_push_status', 'created', {'claim', 'owner', 'status'}),
    ('launch_after_resuming', 'resuming', {'claim', 'owner', 'status'}),
    ('launch_after_resume', 'resuming', {'claim', 'owner', 'status'}),
])
def test_killed_launcher_preserves_failure_table(children, tmp_path, point, phase, pushed):
    log, marker = tmp_path / 'events.jsonl', tmp_path / 'ran'
    launcher = children('''
import subprocess
r, w = os.pipe()
d = claim.acquire('launcher', 'kill-checkpoint')
a = claim.inherited(d, launch=True)
claim.spawn(a, [sys.executable, '-c',
    "from pathlib import Path; import sys,time; Path(sys.argv[1]).touch(); time.sleep(60)",
    sys.argv[1]], role='python', handles={'owner': r, 'status': w},
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
raise AssertionError('checkpoint did not kill')
''', marker, env=dict(os.environ, HFMODEM_COEXIST_LOG=str(log),
                      HFMODEM_COEXIST_FAULT=point + ':kill'))
    entry = None
    try:
        assert launcher.process.wait(timeout=15) == 1
        events = [json.loads(line) for line in log.read_text().splitlines()]
        created, = [event for event in events if event['event'] == 'launch_created']
        entry = winproc.open(created['child_pid'], winproc.END)
        assert entry is not None and not entry.wait()
        assert entry.started == created['child_started']
        record = winproc.read_launch(entry)
        if phase is None:
            assert record is None
            cleanup_available()
        else:
            assert record['phase'] == phase
            assert set(record['handles']) == pushed
            assert record['launcher']['pid'] == launcher.process.pid
            assert winclaim.process_info(launcher.process.pid)[1]
            with pytest.raises(claim.ClaimBusy, match=f'admission in progress.*{entry.pid}'):
                claim.mutex(timeout=0)
        if point == 'launch_after_resume':
            await_condition(marker.exists)
        else:
            assert not marker.exists()
        if 'claim' in pushed:
            with pytest.raises(claim.ClaimBusy):
                acquire()
        else:
            assert_free()
        entry.terminate()
        assert entry.wait(5)
        cleanup_available()
        assert winproc.read_launch(entry) is None
        assert_free()
    finally:
        if entry is not None:
            if not entry.wait():
                entry.terminate()
                assert entry.wait(5)
            entry.close()


@pytest.mark.parametrize('failure', ['record', 'push', 'resuming', 'resume'])
def test_spawn_failure_ends_never_run_child(monkeypatch, tmp_path, failure):
    marker = tmp_path / 'ran'
    processes = []
    suspended = winproc.python_suspended
    def created(*args, **kwargs):
        process = suspended(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(winproc, 'python_suspended', created)
    with held() as descriptor:
        admission = claim.inherited(descriptor, launch=True)
        try:
            with monkeypatch.context() as patch:
                def fail(*args, **kwargs):
                    raise claim.ClaimError('injected launch failure')
                if failure == 'resuming':
                    advance = winproc.advance_launch
                    def advanced(entry, phase, **fields):
                        if phase == 'resuming':
                            fail()
                        return advance(entry, phase, **fields)
                    patch.setattr(winproc, 'advance_launch', advanced)
                else:
                    patch.setattr(winproc, {'record': 'create_launch', 'push': 'push_handle',
                                           'resume': 'resume'}[failure], fail)
                with pytest.raises(claim.ClaimError, match='injected launch failure'):
                    claim.spawn(admission, [sys.executable, '-c',
                        'from pathlib import Path; import sys; Path(sys.argv[1]).touch()',
                        str(marker)], role='python')
        finally:
            admission.failed()
        process, = processes
        assert process.poll() is not None
        assert not marker.exists()
        cleanup_available()
        assert claim.inherited(descriptor) == descriptor
    assert_free()


@pytest.mark.parametrize('condition', ['unreadable', 'malformed', 'unidentified'])
def test_uncertain_launch_records_admit(children, monkeypatch, condition):
    child = children("print('ready', flush=True); input()")
    assert child.line() == 'ready'
    with winproc.open(child.process.pid) as entry:
        winproc.create_launch(entry, role='python')
        path = winproc._path('launch', entry)
        if condition == 'malformed':
            path.write_text('{')
        with monkeypatch.context() as patch:
            if condition == 'unreadable':
                read = winproc._read_json
                def unreadable(candidate):
                    if candidate == path:
                        raise PermissionError('record denied')
                    return read(candidate)
                patch.setattr(winproc, '_read_json', unreadable)
            elif condition == 'unidentified':
                opened = winproc.open
                def denied(pid, *args):
                    if pid == entry.pid:
                        raise winproc.ProcessAccessError(pid, winproc.IDENTIFY)
                    return opened(pid, *args)
                patch.setattr(winproc, 'open', denied)
            with pytest.raises(claim.ClaimBusy, match='admission in progress'):
                claim.mutex(timeout=0)
            assert path.exists()
            with raw(claim.mutex_path(), 'cleanup'):
                pass
        path.unlink()
        cleanup_available()
    child.send()
    child.wait()


@pytest.mark.wall_clock
@pytest.mark.parametrize('acknowledge', [False, True])
def test_record_admission_retries_until_timeout_or_acknowledgement(children, acknowledge):
    with held() as descriptor:
        child = children('''
a = claim.received(launch=True)
print('ready', flush=True)
input()
a.opened()
input()
''', inherited=descriptor)
        assert child.line() == 'ready'
        timer = threading.Timer(.15, child.send) if acknowledge else None
        if timer is not None:
            timer.start()
        start = time.monotonic()
        try:
            if acknowledge:
                mutex = claim.mutex(timeout=3)
                claim.mutex_release(mutex)
                assert child.line() == 'CLAIM ADMITTED'
            else:
                with pytest.raises(claim.ClaimBusy, match=f'admission in progress.*{child.process.pid}'):
                    claim.mutex(timeout=.2)
            assert time.monotonic() - start >= .1
        finally:
            if timer is not None:
                timer.join(5)


def test_record_admission_bridges_nested_python_launch(children):
    with held() as descriptor:
        child = children('''
import subprocess
a = claim.received(launch=True)
assert a.fd is None
shared, bridges = claim._shared, []
def checked_shared(*args):
    fd = shared(*args)
    bridges.append(fd)
    assert winclaim.entries()[fd].role == 'admission'
    return fd
claim._shared = checked_shared
launch = claim.spawn(a.claim, [sys.executable, '-c',
    'from hfmodem.core import claim; a=claim.received(launch=True); a.opened()'],
    role='python', stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
try:
    assert bridges and len(bridges) == 1
    assert bridges[0] not in winclaim.entries()
    out, err = launch.process.communicate(timeout=10)
    assert launch.process.returncode == 0, err
    winproc.advance_launch(launch.child, 'exited')
finally:
    if launch.process.poll() is None:
        launch.process.kill()
        launch.process.wait(timeout=5)
    launch.child.close()
a.opened()
print('nested', flush=True)
input()
''', inherited=descriptor)
        assert child.line() == 'CLAIM ADMITTED'
        assert child.line() == 'nested'
        cleanup_available()
        child.send()
        child.wait()


def test_received_never_replays_a_closed_raw_value(children):
    with held() as descriptor:
        child = children('''
record = winproc.read_launch()
original = record['handles']['claim']
d = claim.received()
try:
    os.get_handle_inheritable(original)
except OSError:
    pass
else:
    raise AssertionError('original raw handle remained open')
replacement = winclaim.duplicate(winclaim.handle(d.fd))
# The record may now name a different live claim copy at the freed raw slot.
# Deterministically republish a valid token too: received must never read it.
winproc.advance_launch(winproc._channels.entry, 'resuming', handles={'claim': replacement})
claim.close(d)
read = winproc.read_launch
def no_replay(*args, **kwargs):
    if not args:
        raise AssertionError('launch record read twice')
    return read(*args, **kwargs)
winproc.read_launch = no_replay
try:
    try:
        claim.received()
    except claim.ClaimError as exc:
        assert 'missing' in str(exc)
    else:
        raise AssertionError('raw token replayed')
    assert not os.get_handle_inheritable(replacement)
finally:
    winclaim.close_handle(replacement)
print('ok', flush=True)
''', inherited=descriptor)
        assert child.line() == 'ok'
        child.wait()


def test_w5_failed_launch_preserves_noninheritable_claim(monkeypatch, tmp_path):
    copies = []
    duplicate = winclaim.duplicate
    def duplicated(*args, **kwargs):
        value = duplicate(*args, **kwargs)
        copies.append(value)
        return value
    with held() as descriptor:
        admission = claim.inherited(descriptor, launch=True)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(winclaim, 'duplicate', duplicated)
                patch.setattr(subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(OSError('spawn failed')))
                with pytest.raises(OSError, match='spawn failed'):
                    claim.spawn(admission, [sys.executable, '-c', 'pass'], role='python')
            assert copies == []
            assert not os.get_handle_inheritable(winclaim.handle(descriptor.fd))
            probe = tmp_path / 'probe.py'
            probe.write_text(PRELUDE + PROBE)
            command = '"' + subprocess.list2cmdline([PYTHON, str(probe), str(winclaim.handle(descriptor.fd))]) + '"'
            hermetic.guard.admit_system(command, os.environ)
            assert os.system(command) == 0
            assert claim.inherited(descriptor) == descriptor
        finally:
            admission.failed()


@pytest.mark.parametrize('fails', [False, True])
def test_launcher_preserves_supplied_startupinfo(monkeypatch, fails):
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 7
    startup.hStdInput, startup.hStdOutput, startup.hStdError = 11, 12, 13
    harmless = os.open(os.devnull, os.O_RDWR)
    os.set_inheritable(harmless, True)
    handles = [winclaim.handle(harmless)]
    attributes = {'handle_list': handles}
    startup.lpAttributeList = attributes
    try:
        with held() as descriptor:
            for _ in range(2):
                with monkeypatch.context() as patch:
                    if fails:
                        def fail(*args, **kwargs):
                            raise OSError('launch failed')
                        patch.setattr(subprocess, 'Popen', fail)
                        with pytest.raises(OSError, match='launch failed'):
                            spawn_claim(descriptor, [sys.executable, '-c', 'pass'], startupinfo=startup)
                    else:
                        launch = spawn_claim(descriptor, [sys.executable, '-c', 'pass'],
                                             startupinfo=startup, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        try:
                            out, err = launch.process.communicate(timeout=10)
                            assert launch.process.returncode == 0, err
                        finally:
                            if launch.process.poll() is None:
                                launch.process.kill()
                                launch.process.wait(timeout=5)
                            launch.child.close()
                assert startup.dwFlags == subprocess.STARTF_USESHOWWINDOW
                assert startup.wShowWindow == 7
                assert (startup.hStdInput, startup.hStdOutput, startup.hStdError) == (11, 12, 13)
                assert startup.lpAttributeList is attributes
                assert attributes['handle_list'] is handles
                assert handles == [winclaim.handle(harmless)]
                assert not os.get_handle_inheritable(winclaim.handle(descriptor.fd))

    finally:
        os.close(harmless)


@pytest.mark.wall_clock
def test_flac_does_not_inherit_during_claim_push(monkeypatch):
    from hfmodem.core import wav
    ready, finish = threading.Event(), threading.Event()
    errors, claims = [], []
    push, run = winproc.push_handle, subprocess.run
    def paused_push(*args):
        value = push(*args)
        ready.set()
        assert finish.wait(20)
        return value
    with held() as descriptor:
        source = winclaim.handle(descriptor.fd)
        def probe_run(argv, **kwargs):
            return run([PYTHON, '-c', PRELUDE + PROBE, str(source)], timeout=15, **kwargs)
        def launch():
            result = None
            try:
                result = spawn_claim(descriptor, [sys.executable, '-c', 'pass'])
                assert result.process.wait(timeout=10) == 0
            except BaseException as exc:
                errors.append(exc)
            finally:
                if result is not None:
                    if result.process.poll() is None:
                        result.process.kill()
                        result.process.wait(timeout=5)
                    result.child.close()
        with monkeypatch.context() as patch:
            patch.setattr(winproc, 'push_handle', paused_push)
            patch.setattr(wav, '_flac_exe', lambda: PYTHON)
            patch.setattr(subprocess, 'run', probe_run)
            thread = threading.Thread(target=launch)
            thread.start()
            try:
                assert ready.wait(10)
                api = winproc._api()
                for value, index in winproc.process_handles(api.GetCurrentProcess()):
                    if api.CompareObjectHandles(source, value):
                        claims.append(value)
                        assert not os.get_handle_inheritable(value)
                assert source in claims
                wav._flac('--version')
            finally:
                finish.set()
                thread.join(30)
                assert not thread.is_alive()
            assert errors == []
        assert claim.inherited(descriptor) == descriptor
    assert_free()


def test_recording_control_reaches_one_child_and_ends_at_eof(children):
    from hfmodem.core import recording
    control_read, control_write = os.pipe()
    out_read, out_write = os.pipe()
    env, options = recording.control_launch(
        dict(os.environ, HFMODEM_RECORDING_CONTROL_OUT_FDS=str(out_write)), read=control_read)
    assert env['HFMODEM_RECORDING_CONTROL_FD'] == 'recording-control'
    assert env['HFMODEM_RECORDING_CONTROL_OUT_FDS'] == 'recording-out-0'
    assert 'pass_fds' not in options
    kernel = C.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = W.HANDLE
    grandchild = None
    try:
        child = children('''
import msvcrt, subprocess
from hfmodem.core import recording
control, (out,) = recording.control_fd(), recording.control_fds()
assert recording.control_fd() == control and recording.control_fds() == (out,)
for fd in (control, out):
    assert not os.get_inheritable(fd)
    assert not os.get_handle_inheritable(msvcrt.get_osfhandle(fd))
# Inherits every inheritable handle: the control ends must not be among them.
grandchild = subprocess.Popen([PYTHON, '-c', 'import time; time.sleep(60)'], close_fds=False,
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(grandchild.pid, flush=True)
buffer = b''
while True:
    data = os.read(control, 4096)
    if not data:
        break
    buffer += data
    while b'\\n' in buffer:
        line, buffer = buffer.split(b'\\n', 1)
        os.write(out, json.dumps(dict(echo=json.loads(line)['op'])).encode() + b'\\n')
os.write(out, b'"eof"\\n')
''', env=env, role='recorder', **options)
        os.close(control_read)
        control_read = None
        # Terminate, query and wait access: teardown must see the grandchild gone.
        grandchild = kernel.OpenProcess(0x101001, False, int(child.line()))
        assert grandchild
        for op in ('begin', 'end'):
            os.write(control_write, (json.dumps(dict(op=op)) + '\n').encode())
        os.close(control_write)
        control_write = None
        os.close(out_write)
        out_write = None
        received, data = b'', None
        os.set_blocking(out_read, False)
        deadline = time.monotonic() + 30
        while data != b'':
            assert time.monotonic() < deadline, f'no EOF within 30 s; read {received!r}'
            try:
                data = os.read(out_read, 4096)
            except BlockingIOError:
                time.sleep(.02)
                continue
            received += data
        code = W.DWORD()
        assert kernel.GetExitCodeProcess(W.HANDLE(grandchild), C.byref(code))
        # EOF while the grandchild still runs proves it holds no control end.
        assert code.value == 259
        assert [json.loads(line) for line in received.splitlines()] == [
            {'echo': 'begin'}, {'echo': 'end'}, 'eof']
        child.wait()
    finally:
        try:
            if grandchild:
                kernel.TerminateProcess(W.HANDLE(grandchild), 1)
                try:
                    assert winclaim._api().WaitForSingleObject(grandchild, 10000) == 0
                finally:
                    winclaim.close_handle(grandchild)
        finally:
            for fd in (control_read, control_write, out_read, out_write):
                if fd is not None:
                    os.close(fd)


def test_recording_receiver_ends_a_contact_when_its_writer_is_terminated(children):
    from hfmodem.core import recording
    control_read, out_write = os.pipe()
    writer_env, writer_options = recording.control_launch(
        dict(os.environ, HFMODEM_RECORDING_CONTROL_OUT_FDS=str(out_write)))
    reader_env, reader_options = recording.control_launch(dict(os.environ), read=control_read)
    try:
        reader = children('''
from types import SimpleNamespace
from hfmodem.core import recording
session = SimpleNamespace(lanes={}, report=print,
    begin=lambda cid, positions, **kw: print('begin', cid, flush=True),
    end=lambda cid, positions, keep, reason, **kw: print('end', cid, keep, reason, flush=True))
receiver = recording.ChildControlReceiver(session, recording.control_fd())
while not receiver._closed:
    receiver.drain()
    time.sleep(.02)
''', env=reader_env, role='recorder', **reader_options)
        writer = children('''
from hfmodem.core import recording
(out,) = recording.control_fds()
os.write(out, json.dumps(dict(op='begin', contact_id='c1', positions={})).encode() + b'\\n')
print('sent', flush=True)
time.sleep(120)
''', env=writer_env, role='recorder', **writer_options)
        for fd in (control_read, out_write):
            os.close(fd)
        control_read = out_write = None
        assert writer.line() == 'sent'
        assert reader.line() == 'begin c1'
        writer.process.kill()
        writer.process.wait(timeout=30)
        assert reader.line() == 'end c1 True child control interrupted'
        reader.wait()
    finally:
        for fd in (control_read, out_write):
            if fd is not None:
                os.close(fd)
