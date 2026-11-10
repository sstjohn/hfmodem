# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import array
import ctypes
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import NamedTuple

from . import claim, coexist, posix

fcntl = posix.module("fcntl")
termios = posix.module("termios")

OPEN_TIMEOUT_S = 12.0
REAP_GRACE_S = 12.0
KILL_WAIT_S = 5.0
POLL_S = 0.05
IDENTITY_CACHE_S = 1.0
WINDOWS_SHELL_UNSUPPORTED = (
    "native launch from the shell is not supported on Windows; "
    "the console and `hfmodem down` launch in-process")


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
    _win_entry: object = field(default=None, repr=False, compare=False, kw_only=True)

    def __post_init__(self):
        if os.name == "nt":
            from . import winproc, winclaim
            if self._win_entry is not None:
                entry, self._win_entry = self._win_entry, None
                self._win_entry = replace(
                    entry, handle=winclaim.duplicate(entry.handle))
            elif self._process is not None:
                handle = winclaim.duplicate(int(self._process._handle))
                try:
                    self._win_entry = winproc._pin(handle, self.pid)
                except BaseException:
                    winclaim.close_handle(handle)
                    raise
            else:
                self._win_entry = winproc.open(self.pid, winproc.END)
            if self._win_entry is not None and self._win_entry.started != self.started:
                self._win_entry.close()
                self._win_entry = None

    def __del__(self):
        entry = getattr(self, "_win_entry", None)
        if entry is not None:
            try:
                entry.close()
            except Exception:
                pass

    def alive(self):
        return self._alive()

    def _alive(self, *, refresh_identity=False):
        if os.name == "nt":
            gone = self._win_entry is None or self._win_entry.wait()
            if gone:
                self._record_exit()
            return not gone
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
                if not claim.process_matches(self.pid, self.started):
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
        if os.name == "nt":
            raise claim.ClaimError("Windows children use stop and interrupt channels")
        if not self._alive(refresh_identity=True):
            return False
        try:
            os.kill(self.pid, sig)
            return True
        except ProcessLookupError:
            return False

    def terminate(self):
        if os.name == "nt":
            return self._stop(REAP_GRACE_S)
        return self._signal(signal.SIGTERM)

    def interrupt(self):
        if os.name == "nt":
            from . import winproc
            if not self.alive():
                return winproc.StopResult("stopped", "already exited")
            if winproc.request_interrupt(self._win_entry) is not None:
                return winproc.StopResult("requested", "interrupt channel")
            return self._force("no interrupt channel; TerminateProcess")
        return self._signal(signal.SIGINT)

    def kill(self):
        if os.name == "nt":
            from . import winproc
            if not self.alive():
                return winproc.StopResult("stopped", "already exited")
            return self._force("TerminateProcess")
        return self._signal(signal.SIGKILL)

    def _force(self, detail):
        from . import winproc
        deadline = time.monotonic() + KILL_WAIT_S
        error = None
        try:
            self._win_entry.terminate()
        except winproc.ProcessAccessError as exc:
            error = exc
        gone = self._win_entry.wait(max(0, deadline - time.monotonic()))
        result = winproc.StopResult(
            "terminated" if gone else "unresolved",
            f"{detail}; {error}" if not gone and error is not None else detail)
        if result:
            self._record_exit()
        return result

    def _record_exit(self):
        if self._process is not None and isinstance(self._process, _WindowsPopen):
            self._process.poll()

    def _stop(self, grace):
        from . import winproc
        if not self.alive():
            return winproc.StopResult("stopped", "already exited")
        stdin = self.stdin if not getattr(self.stdin, "closed", False) else None
        stopper = winproc.Stopper(self._win_entry, stdin=stdin, reap_grace=grace)
        coexist.event("native_stop", child_pid=self.pid, detail=stopper.channel)
        if stopper.grace is None:
            return self._force("no stop channel; TerminateProcess")
        winproc._inspect_checkpoint("native_after_term")
        result = stopper.wait()
        if result:
            self._record_exit()
        return result

    def wait_gone(self, timeout):
        if os.name == "nt":
            gone = self._win_entry is None or self._win_entry.wait(timeout)
            if gone:
                self._record_exit()
            return gone
        deadline = time.monotonic() + timeout
        while self.alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(POLL_S, remaining))
        return True


def _end(child, grace=None):
    grace = REAP_GRACE_S if grace is None else grace
    if os.name == "nt":
        return bool(child._stop(grace))
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
    if os.name == "nt" and (Path(argv[0]).name.casefold() == "ffmpeg.exe" or
                            any(argv[i:i + 2] == ["-f", "dshow"]
                                for i in range(len(argv) - 1))):
        return True
    return Path(argv[0]).name == "ffmpeg" or any(
        argv[i:i + 2] in (["-f", "alsa"], ["-f", "pulse"], ["-f", "avfoundation"])
        for i in range(len(argv) - 1))


def _option(argv, short, long, default=None):
    value = default
    for i, arg in enumerate(argv[1:], 1):
        if arg in (short, long):
            if i + 1 == len(argv):
                raise claim.ClaimError(f"{arg} requires a value")
            value = argv[i + 1]
        elif arg.startswith(long + "="):
            value = arg.split("=", 1)[1]
        elif arg.startswith(short) and not arg.startswith("--"):
            value = arg[len(short):]
    return value


def _rigctld(argv):
    return Path(argv[0]).name.casefold() in ("rigctld", "rigctld.exe")


def _long_lived(argv, requested=False):
    return requested or _audio(argv) or _rigctld(argv)


def _serial_device(argv, device):
    # Observe the port the command actually opens, including attached short options.
    device = _option(argv, "-r", "--rig-file", claim.serial_port(device))
    if claim.serial_port(device) is None:
        raise claim.ClaimError("long-lived native CAT child requires a device")
    return device


class _Observation(NamedTuple):
    nodes: tuple
    output: str | None
    audio: bool
    listener: tuple | None = None


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
        return _Observation(nodes, None if output in ("-", "pipe:", "pipe:1") else output, True)
    # Listener evidence is reserved for the exact argv form our launcher emits.
    if (_rigctld(argv) and len(argv) == 10
            and argv[1:7] == ["-P", "NONE", "-m", "1",
                              "--set-conf=ptt_type=None", "-T"]
            and argv[7] == "127.0.0.1" and argv[8] == "-t"
            and claim.serial_port(device) is None):
        host = argv[7]
        try:
            port = int(argv[9])
            if not 0 < port < 65536 or str(port) != argv[9]:
                raise ValueError("invalid TCP port")
        except ValueError as exc:
            raise claim.ClaimError(f"invalid rigctld listen endpoint: {host}") from exc
        return _Observation((), None, False, (host, port))
    device = _serial_device(argv, device)
    if os.name == "nt":
        from . import winclaim
        return _Observation((winclaim.serial_target(device),), None, False)
    return _Observation(claim.serial_nodes(device), None, False)


def _listener_owned(child, endpoint, *, deadline, cancelled):
    if cancelled():
        raise NativeSpawnError("native supervisor interrupted", child)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    host, port = endpoint
    if os.name == "nt":
        from .down import listeners
        owners = listeners(port, host, deadline=deadline, cancelled=cancelled)
    else:
        from .down_posix import listeners
        try:
            owners = listeners(port, timeout=min(.5, remaining))
        except subprocess.TimeoutExpired:
            return False
    return (not cancelled() and time.monotonic() < deadline
            and owners == {child.pid} and child.alive())


def _child_listener_ready(child, endpoint, *, deadline, cancelled):
    return (_listener_owned(child, endpoint, deadline=deadline, cancelled=cancelled)
            and _listener_ready(endpoint, deadline=deadline, cancelled=cancelled)
            and _listener_owned(child, endpoint, deadline=deadline, cancelled=cancelled))


def _listener_ready(endpoint, *, deadline, cancelled):
    """A connect alone is not evidence; require a Hamlib configuration answer.

    Every socket operation shares the observation deadline and polls interrupt
    state, including a listener that accepts but never finishes its reply.
    """
    def budget():
        if cancelled():
            raise NativeSpawnError("native supervisor interrupted")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native observation deadline")
        return min(POLL_S, remaining)

    host, port = endpoint
    try:
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET,
                           socket.SOCK_STREAM) as connection:
            connection.settimeout(budget())
            connection.connect((host, port))
            pending = b"\\get_conf ptt_type\n"
            while pending:
                connection.settimeout(budget())
                try:
                    sent = connection.send(pending)
                except socket.timeout:
                    continue
                if not sent:
                    return False
                pending = pending[sent:]
            response = bytearray()
            while b"\n" not in response and len(response) < 1024:
                connection.settimeout(budget())
                try:
                    data = connection.recv(1024 - len(response))
                except socket.timeout:
                    continue
                if not data:
                    return False
                response.extend(data)
            budget()
            answer = bytes(response).split(b"\n", 1)[0].strip()
            return b"\n" in response and re.fullmatch(
                rb"(?:ptt_type=)?(?:None|RIG|RIGMICDATA|DTR|RTS|PARALLEL|CM108|GPIO|GPION)",
                answer) is not None
    except OSError:
        return False


def _output_ready(fd, output, observed):
    if output is not None:
        try:
            info = Path(output).stat()
            current = (info.st_dev, info.st_ino, info.st_size)
            initial = observed.get("file")
            # An empty file is no baseline: ffmpeg creates the output before writing the
            # header, and a header is not captured audio.
            if (initial is None or initial[2] == 0 or current[:2] != initial[:2]
                    or current[2] < initial[2]):
                observed["file"] = current
                return False
            return current[2] > initial[2]
        except FileNotFoundError:
            observed.pop("file", None)
            return False
        except OSError:
            return False
    if os.name == "nt":
        from . import winproc
        return winproc.pipe_available(fd) > 0
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


def _observe_posix(child, config, interrupted):
    deadline = time.monotonic() + OPEN_TIMEOUT_S
    observed = {}
    while True:
        if interrupted():
            raise NativeSpawnError("native supervisor interrupted", child)
        if not child.alive():
            raise NativeSpawnError("native child exited before acknowledgement", child)
        if time.monotonic() >= deadline:
            raise NativeSpawnError("native open timed out after 12 s", child)
        if config.get("listener") is not None:
            ready = _child_listener_ready(child, config["listener"], deadline=deadline, cancelled=interrupted)
        else:
            ready = _opened(child.pid, config["nodes"]) or (
                config["audio"] and _output_ready(config["output_fd"], config["output"], observed))
        if ready:
            if interrupted() or time.monotonic() >= deadline:
                continue
            if not child.alive():
                raise NativeSpawnError("native child exited before acknowledgement", child)
            return
        time.sleep(min(POLL_S, max(0, deadline - time.monotonic())))


def _supervise(config):
    if os.name == "nt":
        return _supervise_windows(config)
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, posix.SIGHUP):
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
        _observe_posix(child, config, lambda: bool(interrupted))
        coexist.checkpoint("native_before_acknowledgement")
        if interrupted:
            raise NativeSpawnError("native supervisor interrupted")
        if not child.alive():
            raise NativeSpawnError("native child exited before acknowledgement", child)
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


if os.name == "nt":
    class _WindowsPopen(subprocess.Popen):
        _launch_record = None

        def _exited(self):
            if self.returncode is not None and self._launch_record is not None:
                from . import winproc
                with winproc._launch_lock:
                    if self._launch_record is not None:
                        winproc._update_launch(self._launch_record, "exited")
                        self._launch_record = None

        def wait(self, timeout=None):
            code = super().wait(timeout)
            self._exited()
            return code

        def poll(self):
            code = super().poll()
            self._exited()
            return code


def _start_windows(argv, claim_fd, streams, *, role, device=None, own_session=False):
    from . import winclaim, winproc
    from .winconsole import child_streams
    streams = child_streams(streams)
    checkpoint = winproc._inspect_checkpoint
    python = os.path.normcase(os.path.abspath(argv[0])) in {
        os.path.normcase(os.path.abspath(path)) for path in (sys.executable, sys._base_executable)}
    flags = winproc.creation_flags(role, python=python, own_session=own_session)
    environment = dict(os.environ)
    if python:
        argv = [sys._base_executable, *argv[1:]]
        environment["__PYVENV_LAUNCHER__"] = sys.executable
    child = process = None
    admission = claim.inherited(claim_fd, launch=True)
    try:
        process = _WindowsPopen(argv, close_fds=True, creationflags=4 | flags,
                                env=environment, stdin=streams[0], stdout=streams[1],
                                stderr=streams[2])
        started, _ = winclaim.process_info(process.pid, handle=int(process._handle))
        child = NativeChild(process.pid, started, process.stdin, process.stdout,
                            process.stderr, process)
        entry = child._win_entry
        process._launch_record = entry
        coexist.event("native_spawn", child_pid=child.pid, child_started=started, argv=argv)
        checkpoint("native_after_creation")
        winproc.create_launch(entry, role=role, device=device, grace=REAP_GRACE_S)
        checkpoint("native_after_record")
        pushed = {"claim": winproc.push_handle(entry, winclaim.handle(claim._fd(claim_fd)))}
        winproc.advance_launch(entry, "created", handles=pushed)
        checkpoint("native_after_push_claim")
        winproc.advance_launch(entry, "resuming")
        checkpoint("native_after_resuming")
        winproc.resume(entry)
        checkpoint("native_after_resume")
        return child
    except BaseException as exc:
        gone = process is None
        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=KILL_WAIT_S)
                gone = True
            except Exception:
                gone = False
            if child is None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        raise NativeSpawnError(str(exc) + ("" if gone else "; termination unresolved"),
                               child) from exc
    finally:
        admission.failed()


def _check_windows_child(child):
    if not child.alive():
        code = child._process.poll() if child._process is not None else None
        detail = "" if code is None else f" (exit code {code})"
        raise NativeSpawnError("native child exited before acknowledgement" + detail, child)


def _observe_windows(child, config, interrupted):
    from . import winproc
    deadline = time.monotonic() + OPEN_TIMEOUT_S
    coexist.event("native_observing", child_pid=child.pid, deadline=deadline)
    observed = {}
    while True:
        if interrupted():
            raise NativeSpawnError("native supervisor interrupted", child)
        _check_windows_child(child)
        if time.monotonic() >= deadline:
            program = Path(config["argv"][0]).name
            reason = (f"{program} never produced audio" if config["audio"] else
                      f"{program} never answered on {config['listener']}"
                      if config.get("listener") is not None else
                      f"{program} never opened {config['port']}")
            raise NativeSpawnError(reason, child)
        ready = False
        try:
            if config["audio"]:
                ready = _output_ready(config["output_fd"], config["output"], observed)
            elif config.get("listener") is not None:
                ready = _child_listener_ready(child, config["listener"], deadline=deadline, cancelled=interrupted)
            else:
                result = winproc.device_holders(
                    [child._win_entry], {config["port"]: config["nodes"][0]},
                    deadline=deadline, cancelled=interrupted)
        except (claim.ClaimError, OSError):
            _check_windows_child(child)
            raise
        if not config["audio"] and config.get("listener") is None:
            if interrupted() or time.monotonic() >= deadline:
                continue
            _check_windows_child(child)
            if result.status == "unavailable":
                raise NativeSpawnError(result.message, child)
            if result.message and result.status != "busy":
                raise NativeSpawnError(f"inspection unavailable: {result.message}", child)
            walk = result.entries.get(child.pid)
            if walk is not None and walk.message:
                raise NativeSpawnError(walk.message, child)
            ready = walk is not None and config["port"] in walk.devices
        if ready:
            if interrupted() or time.monotonic() >= deadline:
                continue
            _check_windows_child(child)
            return
        time.sleep(min(POLL_S, max(0, deadline - time.monotonic())))


def _supervise_windows(config):
    from . import winproc
    interrupted = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, posix.SIGHUP):
        signal.signal(sig, lambda *args: interrupted.set())
    posix.on_stop(interrupted.set)
    winproc.declare_grace(REAP_GRACE_S + KILL_WAIT_S + 1)
    admission = child = status = None
    replied = False
    try:
        admission = claim.received(launch=True)
        status = claim.extra("status")
        streams = tuple(claim.extra(name) for name in ("stdin", "stdout", "stderr"))
        config["output_fd"] = claim.extra("output")
        coexist.event("native_supervisor", started=admission.record.started, argv=config["argv"])
        winproc._inspect_checkpoint("native_before_spawn")
        if interrupted.is_set():
            raise NativeSpawnError("native supervisor interrupted")
        child = _start_windows(config["argv"], admission.claim, streams, role="native",
                               device=config.get("device"))
        _observe_windows(child, config, interrupted.is_set)
        winproc._inspect_checkpoint("native_before_acknowledgement")
        if interrupted.is_set():
            raise NativeSpawnError("native supervisor interrupted", child)
        _check_windows_child(child)
        winproc.advance_launch(child._win_entry, "acknowledged")
        winproc.advance_launch(admission.record, "acknowledged")
        admission.failed()
        coexist.event("native_admitted", child_pid=child.pid, child_started=child.started)
        winproc._inspect_checkpoint("native_after_acknowledgement")
        if interrupted.is_set():
            raise NativeSpawnError("native supervisor interrupted", child)
        _reply(status, dict(pid=child.pid, started=child.started))
        replied = True
        winproc._inspect_checkpoint("native_after_reply")
        return 0
    except BaseException as exc:
        if replied:
            return 0
        if child is None and isinstance(exc, NativeSpawnError):
            child = exc.child
        gone = child is None
        if child is not None:
            try:
                gone = _end(child)
                if gone:
                    child._process.wait(timeout=0)
            except Exception:
                gone = False
        message = str(exc) + ("" if gone else "; termination unresolved")
        if status is not None:
            try:
                _reply(status, dict(error=message, pid=child.pid if child else None,
                                   started=child.started if child else None))
            except OSError:
                pass
        return 1
    finally:
        if admission is not None:
            admission.failed()


def _spawn_windows(argv, *, claim_fd, cleanup_fd, long_lived, device, streams, own_session):
    from . import winclaim, winproc
    if cleanup_fd is not None:
        raise claim.ClaimError(WINDOWS_SHELL_UNSUPPORTED)
    argv = list(map(os.fspath, argv))
    if not argv:
        raise ValueError("empty native command")
    long_lived = _long_lived(argv, long_lived)
    observation = _observation(argv, device) if long_lived else None
    admission = claim.inherited(claim_fd, launch=True)
    launch = reader = child = None
    try:
        if not long_lived:
            return _start_windows(argv, admission.claim, streams or (subprocess.PIPE,) * 3,
                                  role="one-shot", device=device, own_session=own_session)
        with _Descriptors() as stack:
            reader, writer = winproc.pipe_pair()
            status = stack.add(winclaim._crt(writer, os.O_WRONLY | os.O_BINARY | os.O_NOINHERIT,
                                             role="native-status"))
            stdin_read, stdin_write = map(stack.add, os.pipe())
            stdout_read, stdout_write = map(stack.add, os.pipe())
            stderr_read, stderr_write = map(stack.add, os.pipe())
            nodes, output, audio, listener = _Observation(*observation)
            child_streams = (stdin_read, stdout_write, stderr_write)
            if streams is not None:
                if audio and output is None:
                    raise claim.ClaimError("shell capture requires file output; use spawn_native for pipes")
                child_streams = tuple(stack.add(os.dup(i if stream is None else stream))
                                      for i, stream in enumerate(streams))
            config = dict(argv=argv, nodes=nodes, output=output, audio=audio, device=device,
                          listener=listener,
                          port=None if audio or listener is not None else _serial_device(argv, device))
            extras = dict(zip(("stdin", "stdout", "stderr"), child_streams))
            extras.update(status=status, output=stdout_read)
            launch = claim.spawn(admission, [sys.executable, "-m", "hfmodem.core.native",
                                            "--supervise", json.dumps(config)],
                                 role="supervisor", handles=extras, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for fd in set((status, stdin_read, stdout_write, stderr_write, *child_streams)):
                stack.close(fd)
            result = bytearray()
            deadline = time.monotonic() + OPEN_TIMEOUT_S + REAP_GRACE_S + KILL_WAIT_S + 15
            while b"\n" not in result:
                if time.monotonic() >= deadline:
                    raise NativeSpawnError("native supervisor response timed out; termination unresolved")
                chunk = reader.read(16384, deadline=min(deadline, time.monotonic() + POLL_S))
                if not chunk.confirmed:
                    raise NativeSpawnError("native supervisor status read unresolved; termination unresolved")
                if chunk.status == "deadline":
                    continue
                if chunk.status == "eof":
                    raise NativeSpawnError("native supervisor exited without acknowledgement")
                if chunk.status != "data":
                    raise NativeSpawnError(f"native supervisor status read {chunk.status}")
                result.extend(chunk.data)
                if len(result) > 16384:
                    raise NativeSpawnError("native supervisor response too large")
            reply = json.loads(result)
            child = None
            if reply.get("pid") is not None:
                pipes = (None,) * 3 if streams is not None else [
                    os.fdopen(os.dup(fd), mode, buffering=0) for fd, mode in
                    ((stdin_write, "wb"), (stdout_read, "rb"), (stderr_read, "rb"))]
                try:
                    child = NativeChild(reply["pid"], reply["started"], *pipes)
                except (claim.ClaimError, OSError) as exc:
                    for stream in pipes:
                        if stream is not None:
                            stream.close()
                    message = f"cannot identify native child pid {reply['pid']}: {exc}"
                    if "error" in reply:
                        message = f"{reply['error']}; {message}"
                    raise NativeSpawnError(message) from exc
            if "error" in reply:
                raise NativeSpawnError(reply["error"], child)
            return child
    except BaseException:
        if launch is not None and not launch.child.wait():
            try:
                supervisor = NativeChild(launch.child.pid, launch.child.started, None, None, None,
                                         launch.process, _win_entry=launch.child)
                _end(supervisor)
            except Exception:
                pass
        raise
    finally:
        failure = sys.exc_info()[1]
        retired = reader is None or reader.close()
        supervisor_gone = True
        try:
            if launch is not None:
                try:
                    supervisor_gone = launch.child.wait(KILL_WAIT_S)
                    if supervisor_gone:
                        launch.process.poll()
                        winproc._update_launch(launch.child, "exited")
                finally:
                    launch.child.close()
        finally:
            admission.failed()
        if not retired:
            raise NativeSpawnError("native supervisor status read unresolved; termination unresolved")
        if not supervisor_gone:
            message = f"native supervisor did not exit within {KILL_WAIT_S:g}s; termination unresolved"
            if failure is not None:
                message = f"{failure}; {message}"
            raise NativeSpawnError(message, getattr(failure, "child", child)) from failure


def spawn_native(argv, *, claim_fd, cleanup_fd=None, long_lived=False, device=None,
                 own_session=False):
    """`own_session` puts a one-shot in a session of its own, out of reach of the
    terminal's Ctrl-C; a long-lived child always has one, under its supervisor."""
    return _spawn_native(argv, claim_fd=claim_fd, cleanup_fd=cleanup_fd,
                         long_lived=long_lived, device=device, own_session=own_session)


def _spawn_native(argv, *, claim_fd, cleanup_fd=None, long_lived=False, device=None,
                  streams=None, own_session=False):
    if os.name == "nt":
        return _spawn_windows(argv, claim_fd=claim_fd, cleanup_fd=cleanup_fd,
                              long_lived=long_lived, device=device,
                              streams=streams, own_session=own_session)
    argv = [os.fspath(arg) for arg in argv]
    if not argv:
        raise ValueError("empty native command")
    long_lived = _long_lived(argv, long_lived)
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
            nodes, output, audio, listener = _Observation(*observation)
            child_streams = (stdin_read, stdout_write, stderr_write)
            if streams is not None:
                if audio and output is None:
                    raise claim.ClaimError("shell capture requires file output; use spawn_native for pipes")
                child_streams = tuple(_dup(i if stream is None else stream, stack)
                                      for i, stream in enumerate(streams))
            config = dict(argv=argv, claim=inherited, status=status_write,
                          streams=child_streams,
                          output_fd=stdout_read, output=output, nodes=nodes, audio=audio,
                          listener=listener)
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
    if os.name == "nt":
        raise claim.ClaimError(WINDOWS_SHELL_UNSUPPORTED)
    timeout = args.timeout
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0
                                or timeout > claim.ADMISSION_TIMEOUT):
        raise claim.ClaimError("native timeout must be in (0, 30] seconds")
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise claim.ClaimError("empty native command")
    if not _long_lived(argv, args.long_lived) and timeout is None:
        raise claim.ClaimError("native one-shot requires --timeout")
    child = _spawn_native(argv, claim_fd=args.fd, cleanup_fd=args.cleanup,
                          long_lived=args.long_lived, device=args.device,
                          streams=(None, None, None))
    if _long_lived(argv, args.long_lived):
        print(f"CLAIM NATIVE {child.pid} {child.started}", flush=True)
        return 0
    stopped = threading.Event()
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT, posix.SIGHUP):
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
