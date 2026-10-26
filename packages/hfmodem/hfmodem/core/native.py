# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import array
import ctypes
from dataclasses import dataclass, field
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import termios
import threading
import time

from . import claim, coexist

OPEN_TIMEOUT_S = 12.0
REAP_GRACE_S = 12.0
KILL_WAIT_S = 5.0
POLL_S = 0.05
IDENTITY_CACHE_S = 1.0


class NativeSpawnError(claim.ClaimError):
    def __init__(self, message, child=None):
        super().__init__(message)
        self.child = child


def _process_state(pid):
    if sys.platform == "linux":
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return "gone"
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                            capture_output=True, text=True, timeout=5)
    if result.returncode or not result.stdout.strip():
        raise claim.ClaimError(f"cannot inspect pid {pid}")
    return result.stdout.strip()


@dataclass
class NativeChild:
    pid: int
    started: str
    stdin: object
    stdout: object
    stderr: object
    _process: object = field(default=None, repr=False, compare=False)
    _identity_checked_at: float = field(default=-math.inf, init=False, repr=False, compare=False)

    def alive(self):
        return self._alive()

    def _alive(self, *, refresh_identity=False):
        # Our own child's pid is ours until we reap it, so poll() answers without
        # the ps calls an unwind's deadman would otherwise have to wait out.
        if self._process is not None:
            return self._process.poll() is None
        try:
            os.kill(self.pid, 0)
            now = time.monotonic()
            # Polls may conservatively report a reused PID alive for one second.
            # Signals always revalidate; a failed refresh must discard the cache.
            if refresh_identity or now - self._identity_checked_at >= IDENTITY_CACHE_S:
                self._identity_checked_at = -math.inf
                if claim.process_started(self.pid) != self.started:
                    return False
                self._identity_checked_at = now
            state = _process_state(self.pid)
            return state != "gone" and not state.startswith("Z")
        except ProcessLookupError:
            return False
        except (OSError, claim.ClaimError, subprocess.TimeoutExpired) as exc:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                return False
            raise claim.ClaimError(f"termination unresolved for pid {self.pid}: {exc}") from exc

    def _signal(self, sig):
        if not self._alive(refresh_identity=True):
            return False
        try:
            os.kill(self.pid, sig)
            return True
        except ProcessLookupError:
            return False

    def terminate(self):
        return self._signal(signal.SIGTERM)

    def kill(self):
        return self._signal(signal.SIGKILL)

    def wait_gone(self, timeout):
        deadline = time.monotonic() + timeout
        while self.alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(POLL_S, remaining))
        return True


def _end(child, grace=None):
    grace = REAP_GRACE_S if grace is None else grace
    child.terminate()
    coexist.checkpoint("native_after_term")
    if child.wait_gone(grace):
        return True
    child.kill()
    return child.wait_gone(KILL_WAIT_S)


class _Descriptors:
    def __init__(self):
        self.fds = set()

    def __enter__(self):
        return self

    def add(self, fd):
        self.fds.add(fd)
        return fd

    def close(self, fd):
        self.fds.remove(fd)
        os.close(fd)

    def __exit__(self, *exc):
        for fd in self.fds:
            os.close(fd)


def _dup(fd, stack):
    copied = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 10)
    return stack.add(copied)


def _pipe(stack):
    read, write = os.pipe()
    try:
        return _dup(read, stack), _dup(write, stack)
    finally:
        os.close(read)
        os.close(write)


def _helper(verb, config, **kwargs):
    return subprocess.Popen([sys.executable, "-m", "hfmodem.core.native", verb,
                             json.dumps(config)], close_fds=True, **kwargs)


def _start(argv, claim_fd, hold_fd, cleanup, streams, own_session=False):
    with _Descriptors() as stack:
        inherited = _dup(claim_fd, stack)
        hold = _dup(hold_fd, stack)
        gate_read, gate_write = _pipe(stack)
        error_read, error_write = _pipe(stack)
        config = dict(argv=argv, claim=inherited, hold=hold, cleanup=cleanup,
                      gate=gate_read, error=error_write)
        process = _helper("--exec", config, pass_fds=(inherited, hold, gate_read, error_write),
                          stdin=streams[0], stdout=streams[1], stderr=streams[2],
                          start_new_session=own_session)
        child = None
        try:
            started = claim.process_started(process.pid)
            child = NativeChild(process.pid, started, process.stdin, process.stdout,
                                process.stderr, process)
            coexist.event("native_spawn", child_pid=child.pid, child_started=started, argv=argv)
            os.write(gate_write, b"1")
            # Only the exec trampoline owns the writer while we wait for exec's EOF.
            stack.close(error_write)
        except BaseException:
            # The unreleased gate prevents any device open when identification fails.
            process.kill()
            process.wait(timeout=KILL_WAIT_S)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            raise
        finally:
            for fd in (inherited, hold, gate_read, gate_write):
                stack.close(fd)
        try:
            if not select.select([error_read], [], [], OPEN_TIMEOUT_S)[0]:
                gone = _end(child)
                raise NativeSpawnError("native exec timed out" + (
                    "" if gone else "; termination unresolved"), child)
            error = os.read(error_read, 16384)
            if error:
                process.wait(timeout=KILL_WAIT_S)
                raise NativeSpawnError(error.decode(errors="replace"), child)
            return child
        except NativeSpawnError:
            raise
        except BaseException as exc:
            raise NativeSpawnError(f"native exec unresolved: {exc}", child) from exc
        finally:
            stack.close(error_read)


def _exec(config):
    error = config["error"]
    os.set_inheritable(error, False)
    try:
        if os.read(config["gate"], 1) != b"1":
            return 1
        os.close(config["gate"])
        os.dup2(config["claim"], 9)
        os.close(config["claim"])
        if config["cleanup"]:
            os.dup2(config["hold"], 8)
            os.close(config["hold"])
        else:
            os.set_inheritable(config["hold"], True)
        os.execvpe(config["argv"][0], config["argv"], os.environ)
    except BaseException as exc:
        os.write(error, str(exc).encode()[:16384])
        return 1


class _ProcFD(ctypes.Structure):
    _fields_ = [("fd", ctypes.c_int32), ("kind", ctypes.c_uint32)]


class _VnodePath(ctypes.Structure):
    # sys/proc_info.h: proc_fileinfo (24), vnode_info (152), MAXPATHLEN (1024).
    _fields_ = [("info", ctypes.c_uint64 * 22), ("path", ctypes.c_char * 1024)]


def _darwin_paths(pid, lib=None):
    if lib is None:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib.proc_pidinfo.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                    ctypes.c_void_p, ctypes.c_int)
        lib.proc_pidinfo.restype = ctypes.c_int
        lib.proc_pidfdinfo.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_int)
        lib.proc_pidfdinfo.restype = ctypes.c_int
    size = lib.proc_pidinfo(pid, 1, 0, None, 0)
    if size <= 0:
        return ()
    entries = (_ProcFD * (size // ctypes.sizeof(_ProcFD) + 32))()
    count = lib.proc_pidinfo(pid, 1, 0, entries, ctypes.sizeof(entries))
    paths = []
    for entry in entries[:max(0, count) // ctypes.sizeof(_ProcFD)]:
        if entry.kind != 1:
            continue
        info = _VnodePath()
        if lib.proc_pidfdinfo(pid, entry.fd, 2, ctypes.byref(info),
                              ctypes.sizeof(info)) == ctypes.sizeof(info):
            paths.append(os.fsdecode(info.path))
    return tuple(paths)


def _opened(pid, nodes):
    if not nodes:
        return False
    try:
        if sys.platform == "darwin":
            paths = _darwin_paths(pid)
        else:
            paths = []
            for fd in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    paths.append(os.readlink(fd))
                except OSError:
                    continue
        return bool(set(nodes).intersection(os.path.realpath(p) for p in paths))
    except OSError:
        return False


def _audio(argv):
    return Path(argv[0]).name == "ffmpeg" or any(
        argv[i:i + 2] in (["-f", "alsa"], ["-f", "pulse"], ["-f", "avfoundation"])
        for i in range(len(argv) - 1))


def _observation(argv, device):
    audio = _audio(argv)
    if audio:
        endpoint = device
        if endpoint is None and "-i" in argv:
            endpoint = argv[argv.index("-i") + 1]
        # Server endpoints have no directly owned node; do not initialise PortAudio.
        nodes = claim.audio_nodes(endpoint, "capture") if (
            sys.platform == "linux" and isinstance(endpoint, str)
            and (endpoint.startswith("/dev/snd/") or re.fullmatch(r"(?:plug)?hw:\d+(?:,\d+)?", endpoint))
        ) else ()
        output = argv[-1]
        return nodes, None if output in ("-", "pipe:", "pipe:1") else output, True
    device = claim.serial_port(device)
    if device is None:
        for i, arg in enumerate(argv[:-1]):
            if arg in ("-r", "--rig-file"):
                device = argv[i + 1]
                break
        if device is None:
            device = next((a.split("=", 1)[1] for a in argv
                           if a.startswith("--rig-file=")), None)
    if claim.serial_port(device) is None:
        raise claim.ClaimError("long-lived native CAT child requires a device")
    return claim.serial_nodes(device), None, False


def _output_ready(fd, output, observed):
    if output is not None:
        try:
            info = Path(output).stat()
            current = (info.st_dev, info.st_ino, info.st_size)
            initial = observed.get("file")
            if initial is None or current[:2] != initial[:2] or current[2] < initial[2]:
                observed["file"] = current
                return False
            return current[2] > initial[2]
        except FileNotFoundError:
            observed.pop("file", None)
            return False
        except OSError:
            return False
    count = array.array("i", [0])
    fcntl.ioctl(fd, termios.FIONREAD, count, True)
    return count[0] > 0


def _record(path, data):
    temporary = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reply(fd, data):
    payload = (json.dumps(data) + "\n").encode()
    while payload:
        payload = payload[os.write(fd, payload):]


def _supervise(config):
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    admission = child = record = None
    completed = False
    try:
        admission = claim.inherited(config["claim"], launch=True)
        started = claim.process_started(os.getpid())
        directory = claim.claim_path().with_name("rig.claim.native.d")
        directory.mkdir(mode=0o700, exist_ok=True)
        record = directory / f"{os.getpid()}-{started}"
        data = dict(pid=os.getpid(), started=started, argv=config["argv"],
                    run=os.environ.get("HFMODEM_COEXIST_RUN"))
        _record(record, data)
        coexist.event("native_supervisor", started=started, argv=config["argv"])
        coexist.checkpoint("native_before_spawn")
        if interrupted:
            raise NativeSpawnError("native supervisor interrupted")
        child = _start(config["argv"], config["claim"], admission.fd, False,
                       config["streams"])
        coexist.checkpoint("native_before_child_publication")
        data.update(child_pid=child.pid, child_started=child.started)
        _record(record, data)
        coexist.checkpoint("native_after_child_publication")
        deadline = time.monotonic() + OPEN_TIMEOUT_S
        observed = {}
        while True:
            if interrupted:
                raise NativeSpawnError("native supervisor interrupted")
            if not child.alive():
                raise NativeSpawnError("native child exited before acknowledgement")
            if _opened(child.pid, config["nodes"]) or (
                    config["audio"] and _output_ready(config["output_fd"], config["output"],
                                                       observed)):
                break
            if time.monotonic() >= deadline:
                raise NativeSpawnError("native open timed out after 12 s")
            time.sleep(POLL_S)
        coexist.checkpoint("native_before_acknowledgement")
        if interrupted:
            raise NativeSpawnError("native supervisor interrupted")
        fcntl.flock(admission.fd, fcntl.LOCK_UN)
        completed = True
        record.unlink(missing_ok=True)
        coexist.event("native_admitted", child_pid=child.pid, child_started=child.started)
        coexist.checkpoint("native_after_acknowledgement")
        if interrupted:
            raise NativeSpawnError("native supervisor interrupted")
        _reply(config["status"], dict(pid=child.pid, started=child.started))
        return 0
    except BaseException as exc:
        if child is None and isinstance(exc, NativeSpawnError):
            child = exc.child
        if child is not None:
            try:
                completed = _end(child)
            except Exception:
                completed = False
        else:
            completed = True
        if completed and record is not None:
            record.unlink(missing_ok=True)
        message = str(exc)
        if not completed:
            message += "; termination unresolved"
        try:
            _reply(config["status"], dict(error=message, pid=child.pid if child else None,
                                          started=child.started if child else None))
        except OSError:
            pass
        return 1
    finally:
        if admission is not None:
            admission.failed()


def spawn_native(argv, *, claim_fd, cleanup_fd=None, long_lived=False, device=None,
                 own_session=False):
    """`own_session` puts a one-shot in a session of its own, out of reach of the
    terminal's Ctrl-C; a long-lived child always has one, under its supervisor."""
    return _spawn_native(argv, claim_fd=claim_fd, cleanup_fd=cleanup_fd,
                         long_lived=long_lived, device=device, own_session=own_session)


def _spawn_native(argv, *, claim_fd, cleanup_fd=None, long_lived=False, device=None,
                  streams=None, own_session=False):
    argv = [os.fspath(arg) for arg in argv]
    if not argv:
        raise ValueError("empty native command")
    long_lived = long_lived or _audio(argv)
    if cleanup_fd is not None and long_lived:
        raise claim.ClaimError("cleanup native children must be one-shots")
    observation = _observation(argv, device) if long_lived else None
    admission = None
    try:
        if cleanup_fd is None:
            admission = claim.inherited(claim_fd, launch=True)
            hold_fd = admission.fd
        else:
            hold_fd = claim.verify_cleanup(cleanup_fd)
            claim.inherited(claim_fd)
        claim_fd = claim._fd(claim_fd)
        if not long_lived:
            return _start(argv, claim_fd, hold_fd, cleanup_fd is not None,
                          streams or (subprocess.PIPE,) * 3, own_session)
        with _Descriptors() as stack:
            inherited = _dup(claim_fd, stack)
            status_read, status_write = _pipe(stack)
            stdin_read, stdin_write = _pipe(stack)
            stdout_read, stdout_write = _pipe(stack)
            stderr_read, stderr_write = _pipe(stack)
            nodes, output, audio = observation
            child_streams = (stdin_read, stdout_write, stderr_write)
            if streams is not None:
                if audio and output is None:
                    raise claim.ClaimError("shell capture requires file output; use spawn_native for pipes")
                child_streams = tuple(_dup(i if stream is None else stream, stack)
                                      for i, stream in enumerate(streams))
            config = dict(argv=argv, claim=inherited, status=status_write,
                          streams=child_streams,
                          output_fd=stdout_read, output=output, nodes=nodes, audio=audio)
            process = _helper("--supervise", config, start_new_session=True,
                              pass_fds=(inherited, status_write, *child_streams, stdout_read),
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
            # These copies must close before EOF can report a dead supervisor.
            stack.close(status_write)
            stack.close(stdin_read)
            stack.close(stdout_write)
            stack.close(stderr_write)
            result = bytearray()
            deadline = time.monotonic() + OPEN_TIMEOUT_S + REAP_GRACE_S + KILL_WAIT_S + 15
            while b"\n" not in result:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.terminate()
                    raise NativeSpawnError("native supervisor response timed out; termination unresolved")
                ready, _, _ = select.select([status_read], [], [], min(remaining, POLL_S))
                if ready:
                    chunk = os.read(status_read, 16384)
                    if not chunk:
                        process.wait(timeout=KILL_WAIT_S)
                        raise NativeSpawnError("native supervisor exited without acknowledgement")
                    result.extend(chunk)
            reply = json.loads(result)
            process.wait(timeout=KILL_WAIT_S)
            child = None
            if reply.get("pid") is not None:
                pipes = (None, None, None) if streams is not None else [
                    os.fdopen(os.dup(fd), mode, buffering=0) for fd, mode in
                    ((stdin_write, "wb"), (stdout_read, "rb"), (stderr_read, "rb"))]
                child = NativeChild(reply["pid"], reply["started"], *pipes)
            if "error" in reply:
                raise NativeSpawnError(reply["error"], child)
            return child
    finally:
        if admission is not None:
            admission.failed()


def shell_spawn(args):
    timeout = args.timeout
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0
                                or timeout > claim.ADMISSION_TIMEOUT):
        raise claim.ClaimError("native timeout must be in (0, 30] seconds")
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise claim.ClaimError("empty native command")
    if not args.long_lived and not _audio(argv) and timeout is None:
        raise claim.ClaimError("native one-shot requires --timeout")
    child = _spawn_native(argv, claim_fd=args.fd, cleanup_fd=args.cleanup,
                          long_lived=args.long_lived, device=args.device,
                          streams=(None, None, None))
    if args.long_lived or _audio(argv):
        print(f"CLAIM NATIVE {child.pid} {child.started}", flush=True)
        return 0
    stopped = threading.Event()
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[sig] = signal.signal(sig, lambda signum, frame: stopped.set())
    try:
        deadline = time.monotonic() + timeout
        while not stopped.is_set() and child.alive():
            if time.monotonic() >= deadline:
                break
            stopped.wait(POLL_S)
        if child.alive():
            if not _end(child, grace=5.0):
                raise NativeSpawnError(f"termination unresolved for pid {child.pid}", child)
            return 124
        code = child._process.wait(timeout=KILL_WAIT_S)
        return code if code >= 0 else 128 - code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    operation, configuration = sys.argv[1:]
    raise SystemExit((_exec if operation == "--exec" else _supervise)(json.loads(configuration)))
