# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import pwd
import re
import secrets
import stat
import subprocess
import sys
import time

from . import coexist

MAX_NONCE = 2**40
ADMISSION_TIMEOUT = 30.0
_LINE_LIMIT = 16384


class ClaimError(RuntimeError):
    pass


class ClaimBusy(ClaimError):
    pass


def state_dir():
    chosen = os.environ.get("HFMODEM_STATE_DIR")
    if coexist.enabled() and chosen:
        return Path(chosen)
    if coexist.test_run():
        raise ClaimError("test run without both HFMODEM_COEXIST_HARNESS and "
                         "HFMODEM_STATE_DIR; refusing the operator's state directory")
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local/state/hfmodem"


def claim_path():
    return state_dir() / "rig.claim"


def mutex_path():
    return state_dir() / "rig.claim.cleanup"


def _open(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)


def bootstrap():
    for path in (claim_path(), mutex_path()):
        fd = _open(path)
        try:
            _check_inode(fd, path)
        finally:
            os.close(fd)
    return claim_path(), mutex_path()


def _fd(value):
    fd = value.fd if isinstance(value, ClaimDescriptor) else value
    if type(fd) is not int or fd < 3:
        raise ClaimError("missing inherited station claim descriptor")
    return fd


@dataclass(frozen=True)
class ClaimDescriptor:
    fd: int

    def pread(self, size=_LINE_LIMIT, offset=0):
        return os.pread(self.fd, size, offset)

    def fstat(self):
        return os.fstat(self.fd)

    def offset(self):
        return os.lseek(self.fd, 0, os.SEEK_CUR)


def _check_inode(fd, path):
    try:
        owned, named = os.fstat(fd), path.stat()
    except OSError as exc:
        raise ClaimError(f"descriptor does not name {path}: {exc.strerror}") from None
    if (not stat.S_ISREG(owned.st_mode) or owned.st_uid != os.getuid()
            or (owned.st_dev, owned.st_ino) != (named.st_dev, named.st_ino)):
        raise ClaimError(f"descriptor does not name {path}")
    return owned


def check_inode(descriptor):
    return _check_inode(_fd(descriptor), claim_path())


def _linux_process_started(pid, proc=Path("/proc")):
    try:
        text = (proc / str(pid) / "stat").read_text(errors="surrogateescape")
        opening, closing = text.find("("), text.rfind(")")
        fields = text[closing + 1:].split()
        if (opening < 1 or closing < opening or text[:opening].strip() != str(pid)
                or len(fields) <= 19 or not re.fullmatch(r"[0-9]+", fields[19])):
            raise ValueError
        boot = (proc / "sys/kernel/random/boot_id").read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot):
            raise ValueError
        return f"linux:{boot}:{fields[19]}"
    except (OSError, ValueError) as exc:
        raise ClaimError(f"cannot identify pid {pid}: {exc}") from exc


def ps_started(pid, lstart=None):
    """`ps` start time: what process inventories record and compare among themselves.
    On Linux it moves with clock slew, so a claim holder uses process_started instead."""
    if lstart is None:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise ClaimError(f"cannot identify pid {pid}")
        lstart = result.stdout
    value = "_".join(lstart.split())
    if not value:
        raise ClaimError(f"cannot identify pid {pid}")
    return value


def process_started(pid):
    if sys.platform.startswith("linux"):
        return _linux_process_started(pid)
    return ps_started(pid)


def process_matches(pid, started):
    current = process_started(pid)
    if started == current:
        return True
    # Existing Linux holders can still carry the pre-upgrade ps timestamp.
    if current.startswith("linux:") and not started.startswith("linux:"):
        return started == ps_started(pid)
    return False


def _command(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                            capture_output=True, text=True, timeout=5)
    if result.returncode or not result.stdout.strip():
        raise ClaimError(f"cannot identify pid {pid}")
    # Keep the same flag-shaped redaction as tools/lib/ports.sh.
    redacted = re.sub(r"(--(?:mail-)?(?:password|passwd|secret|token)(?:=|\s+))\S+",
                      r"\1<redacted>", result.stdout)
    return " ".join(redacted.split())


@dataclass(frozen=True)
class Holder:
    pid: int
    started: str
    nonce: int
    kind: str = ""
    operation: str = ""
    command: str = ""


def _parse_line(data, *, cleanup=False):
    try:
        if not data.endswith(b"\n") or data.count(b"\n") != 1:
            raise ValueError
        parts = data.decode().rstrip("\n").split(" ", 5)
        if cleanup:
            pid, started, nonce = parts
            kind = operation = command = ""
        else:
            pid, started, kind, operation, nonce, command = parts
        if not all(parts) or not 1 <= int(nonce) <= MAX_NONCE or int(pid) <= 0:
            raise ValueError
        return Holder(int(pid), started, int(nonce), kind, operation, command)
    except (ValueError, UnicodeError):
        raise ClaimError("holder unknown: incomplete holder line") from None


def holder(descriptor, *, cleanup=False, attempts=3):
    fd = _fd(descriptor)
    for attempt in range(attempts):
        try:
            return _parse_line(os.pread(fd, _LINE_LIMIT, 0), cleanup=cleanup)
        except ClaimError:
            if attempt + 1 == attempts:
                raise
            coexist.event("holder_retry", fd=fd)
            time.sleep(0.01)


def _identity(fd, nonce=None):
    info = os.fstat(fd)
    return dict(start=process_started(os.getpid()), fd=fd, dev=info.st_dev,
                ino=info.st_ino, nonce=nonce)


def _event(name, fd, nonce=None, **fields):
    if coexist.enabled():
        coexist.event(name, **_identity(fd, nonce), **fields)


def _publish(fd, *, kind=None, operation=None, pid=None, cleanup=False):
    pid = os.getpid() if pid is None else pid
    started = process_started(pid)
    nonce = secrets.randbelow(MAX_NONCE) + 1
    if cleanup:
        line = f"{pid} {started} {nonce}\n"
    else:
        if not all(isinstance(s, str) and s and not any(c.isspace() for c in s)
                   for s in (kind, operation)):
            raise ClaimError("kind and operation must be nonempty words")
        line = f"{pid} {started} {kind} {operation} {nonce} {_command(pid)}\n"
    data = line.encode()
    if len(data) >= _LINE_LIMIT:
        raise ClaimError("holder line too long")
    coexist.checkpoint("before_holder_line")
    os.ftruncate(fd, 0)
    if os.pwrite(fd, data, 0) != len(data):
        raise ClaimError("short holder line write")
    if os.lseek(fd, nonce, os.SEEK_SET) != nonce:
        raise ClaimError("holder nonce seek failed")
    _event("holder_published", fd, nonce, cleanup=cleanup)
    coexist.checkpoint("after_holder_line")
    return nonce


def parse_offset(value):
    if value.startswith("0t"):
        return int(value[2:], 10)
    if value.startswith("0x"):
        return int(value[2:], 16)
    if value.isdecimal():
        return int(value, 10)
    raise ValueError("unrecognised descriptor offset")


def fdinfo_offset(text):
    for line in text.splitlines():
        if line.startswith("pos:"):
            return parse_offset(line.split(":", 1)[1].strip())
    raise ValueError("fdinfo has no pos")


@dataclass(frozen=True)
class Opener:
    pid: int
    fd: int
    offset: int | None
    command: str
    started: str = ""


def parse_lsof(text):
    rows = []
    pid = fd = offset = None
    command = ""

    def append():
        if pid is not None and fd is not None:
            rows.append(Opener(pid, fd, offset, command))

    for line in text.replace("\0", "\n").splitlines():
        key, value = line[:1], line[1:]
        if key == "p":
            append()
            pid, fd, offset, command = int(value), None, None, ""
        elif key == "c":
            command = value
        elif key == "f":
            append()
            match = re.fullmatch(r"(\d+)[a-zA-Z]*", value)
            if pid is None or match is None:
                raise ValueError("unresolved lsof descriptor")
            fd, offset = int(match[1]), None
        elif key == "o":
            offset = parse_offset(value)
        else:
            raise ValueError("unrecognised lsof field")
    append()
    return rows


def _lsof(paths):
    paths = [path for path in map(str, paths) if os.path.exists(path)]
    if not paths:
        return ""
    try:
        result = subprocess.run(["lsof", "-w", "+c", "0", "-nP", "-o", "-Fpcfo", "--", *paths],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClaimError(f"device discovery unavailable: {exc}") from exc
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise ClaimError(f"device discovery failed: {result.stderr.strip()}")
    return result.stdout


def _exited(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def openers(paths, *, lsof=None, fdinfo=None, started=None, platform=None):
    lsof, started = lsof or _lsof, started or process_started
    platform = platform or sys.platform
    if fdinfo is None:
        fdinfo = lambda pid, fd: Path(f"/proc/{pid}/fdinfo/{fd}").read_text()

    def identity(pid):
        try:
            return started(pid)
        except ClaimError:
            if _exited(pid):
                return None
            raise

    result = []
    try:
        rows = parse_lsof(lsof(paths))
        identities = {pid: identity(pid) for pid in {row.pid for row in rows}}
        dropped = None in identities.values()
        if rows and not platform.startswith("linux"):
            rows = parse_lsof(lsof(paths))
        else:
            rows = [row for row in rows if identities[row.pid] is not None]
        for row in rows:
            if identities.get(row.pid) is None:
                raise ClaimError("openers changed during discovery; retry")
            before = identities[row.pid]
            offset = row.offset
            if platform.startswith("linux"):
                offset = fdinfo_offset(fdinfo(row.pid, row.fd))
            after = identity(row.pid)
            if after is None:
                dropped = True
                continue
            if before != after:
                raise ClaimError(f"pid {row.pid} changed during discovery")
            result.append(Opener(row.pid, row.fd, offset, row.command, before))
        if dropped and {row.pid for row in parse_lsof(lsof(paths))} - {row.pid for row in result}:
            raise ClaimError("openers changed during discovery; retry")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise ClaimError(f"holder discovery unavailable: {exc}") from exc
    return result


def holders(descriptor, *, cleanup=False, **discovery):
    fd = _fd(descriptor)
    path = mutex_path() if cleanup else claim_path()
    _check_inode(fd, path)
    published = holder(fd, cleanup=cleanup)
    return [row for row in openers([path], **discovery) if row.offset == published.nonce]


def _mutex_busy(fd, **discovery):
    try:
        rows = openers([mutex_path()], **discovery)
        try:
            nonce = holder(fd, cleanup=True).nonce
        except ClaimError:
            nonce = None
        exclusive = [row for row in rows if nonce is not None and row.offset == nonce]
        if exclusive:
            return True, "cleanup in progress (pids " + ", ".join(
                str(pid) for pid in sorted({row.pid for row in exclusive})) + ")"
        if any(row.offset not in (0, None) for row in rows):
            return True, "cleanup in progress (holder unknown)"
        admission = [row for row in rows if row.offset == 0
                     and (row.pid, row.fd) != (os.getpid(), fd)]
        names = ", ".join(str(pid) for pid in sorted({row.pid for row in admission}))
        return False, f"admission in progress ({'pids ' + names if names else 'holder unknown'})"
    except ClaimError:
        return True, "cleanup in progress (holder unknown)"


def _shared(fd=None):
    owned = fd is None
    fd = _open(mutex_path()) if owned else _fd(fd)
    try:
        _check_inode(fd, mutex_path())
        if ClaimDescriptor(fd).offset() != 0:
            raise ClaimError("admission requires a fresh mutex descriptor")
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            _, message = _mutex_busy(fd)
            if message.startswith("admission"):
                message = "cleanup in progress (holder unknown)"
            raise ClaimBusy(message) from None
        _check_inode(fd, mutex_path())
        _event("admission_acquired", fd, 0)
        return fd
    except BaseException:
        os.close(fd)
        raise


def mutex(fd=None, *, timeout=ADMISSION_TIMEOUT, pid=None, **discovery):
    fd = _open(mutex_path()) if fd is None else _fd(fd)
    locked = False
    try:
        _check_inode(fd, mutex_path())
        if ClaimDescriptor(fd).offset() != 0:
            raise ClaimError("cleanup requires a fresh mutex descriptor")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                exclusive, message = _mutex_busy(fd, **discovery)
                if exclusive or time.monotonic() >= deadline:
                    raise ClaimBusy(message) from None
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        _check_inode(fd, mutex_path())
        nonce = _publish(fd, pid=pid, cleanup=True)
        _event("cleanup_acquired", fd, nonce)
        coexist.checkpoint("after_mutex")
        return fd
    except BaseException:
        try:
            if locked:
                os.ftruncate(fd, 0)
                _event("cleanup_closed", fd, ClaimDescriptor(fd).offset(), failed=True)
        finally:
            os.close(fd)
        raise


def verify_cleanup(fd):
    fd = _fd(fd)
    _check_inode(fd, mutex_path())
    probe = _open(mutex_path())
    try:
        _check_inode(probe, mutex_path())
        try:
            fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ClaimError("cleanup mutex is not held exclusively")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ClaimError("cleanup mutex belongs to another description") from None
    finally:
        os.close(probe)
    return fd


@dataclass
class Admission:
    claim: ClaimDescriptor
    fd: int | None

    def opened(self):
        coexist.checkpoint("before_acknowledgement")
        self.failed()
        print("CLAIM ADMITTED", flush=True)
        coexist.checkpoint("after_acknowledgement")

    def failed(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            try:
                _event("admission_closed", fd, 0)
            finally:
                os.close(fd)


def inherited(fd, *, launch=False):
    admission_fd = None
    try:
        if launch is not False:
            admission_fd = _shared(None if launch is True else launch)
        fd = _fd(fd)
        descriptor = ClaimDescriptor(fd)
        check_inode(descriptor)
        coexist.checkpoint("inherited_after_inode")
        published = holder(descriptor)
        offset = descriptor.offset()
        if offset != published.nonce:
            raise ClaimError("claim descriptor does not possess the holder nonce")
        _event("inherited", fd, published.nonce, offset=offset, verdict=True)
        coexist.checkpoint("after_launch_validation" if launch is not False else "after_inherited")
        return Admission(descriptor, admission_fd) if launch is not False else descriptor
    except BaseException:
        if admission_fd is not None:
            Admission(ClaimDescriptor(-1), admission_fd).failed()
        coexist.event("inherited", verdict=False)
        raise


@dataclass(frozen=True)
class Devices:
    cat_port: str | None = None
    ptt_port: str | None = None
    capture: str | int | None = None
    playback: str | int | None = None
    cat: str = "serial"

    def __post_init__(self):
        object.__setattr__(self, "cat_port", serial_port(self.cat_port))
        object.__setattr__(self, "ptt_port", serial_port(self.ptt_port))


@dataclass(frozen=True)
class DeviceVerdict:
    endpoint: str
    verdict: str
    nodes: tuple[str, ...] = ()
    pids: tuple[int, ...] = ()


def serial_port(value):
    return value if value is not None and str(value).strip() else None


def serial_nodes(path, *, platform=None):
    if serial_port(path) is None:
        return ()
    path = os.path.realpath(path)
    if (platform or sys.platform) == "darwin":
        if path.startswith("/dev/cu."):
            return (path, path.replace("/dev/cu.", "/dev/tty.", 1))
        if path.startswith("/dev/tty."):
            return (path.replace("/dev/tty.", "/dev/cu.", 1), path)
    return (path,)


def audio_nodes(endpoint, direction, *, sounddevice=None, platform=None):
    if (platform or sys.platform) != "linux":
        return ()
    if isinstance(endpoint, str) and endpoint.startswith("/dev/snd/"):
        return (os.path.realpath(endpoint),)
    name = endpoint
    if not isinstance(name, str) or not re.search(r"(?:plug)?hw:\d+(?:,\d+)?", name):
        if sounddevice is None:
            import sounddevice
        if endpoint is None or endpoint == -1 or endpoint == "default":
            endpoint = sounddevice.default.device[0 if direction == "capture" else 1]
        name = sounddevice.query_devices(endpoint, "input" if direction == "capture" else "output")["name"]
    match = re.search(r"(?:plug)?hw:(\d+)(?:,(\d+))?(?![\w,])", str(name))
    if not match:
        return ()
    card, device = match[1], match[2] or "0"
    suffix = "c" if direction == "capture" else "p"
    return (f"/dev/snd/pcmC{card}D{device}{suffix}",)


def check_devices(devices=None, *, sounddevice=None, platform=None, **discovery):
    devices = devices or Devices()
    platform = platform or sys.platform
    verdicts = []
    for label, endpoint in (("cat", devices.cat_port), ("ptt", devices.ptt_port)):
        if label == "cat" and devices.cat in ("rigctld", "none"):
            verdicts.append(DeviceVerdict(label, "delegated" if devices.cat == "rigctld" else "none"))
            continue
        if endpoint is None:
            continue
        nodes = serial_nodes(endpoint, platform=platform)
        rows = openers(nodes, platform=platform, **discovery) if nodes else []
        if rows:
            raise ClaimBusy(f"{label} held by pid " + ", ".join(str(r.pid) for r in rows))
        verdicts.append(DeviceVerdict(label, "clear", nodes))
    for direction in ("capture", "playback"):
        endpoint = getattr(devices, direction)
        try:
            nodes = audio_nodes(endpoint, direction, sounddevice=sounddevice, platform=platform)
        except Exception:
            nodes = ()
        rows = openers(nodes, platform=platform, **discovery) if nodes else []
        servers = {"pipewire", "pulseaudio", "coreaudiod", "wireplumber", "jackd"}
        direct = [r for r in rows if Path(r.command).name not in servers]
        if direct:
            raise ClaimBusy(f"{direction} held by pid " + ", ".join(str(r.pid) for r in direct))
        verdicts.append(DeviceVerdict(direction, "cooperative", nodes, tuple(r.pid for r in rows)))
    for verdict in verdicts:
        coexist.event("device_check", endpoint=verdict.endpoint, verdict=verdict.verdict,
                      nodes=verdict.nodes, pids=verdict.pids)
    return tuple(verdicts)


def line_proof(ptt_factory=None):
    if ptt_factory is None:
        print("PTT LINE none", flush=True)
        return "none"
    ptt = None
    verdict = "unreadable"
    try:
        coexist.checkpoint("during_line_proof")
        ptt = ptt_factory()
        ptt.assert_(False)
        sensed = ptt.sense()
        verdict = "low" if sensed is False else "high" if sensed is True else "unreadable"
        coexist.event("line_readback", verdict=verdict)
    finally:
        try:
            if ptt is not None:
                ptt.release()
        finally:
            print(f"PTT LINE {verdict}", flush=True)
    if verdict != "low" or getattr(ptt, "released_low", True) is not True:
        raise ClaimError(f"NOT CONFIRMED: PTT LINE {verdict}")
    return verdict


def line_factory(port):
    if serial_port(port) is None:
        return None

    def open_line():
        from .ptt import RtsPtt
        ptt = RtsPtt(port)
        ptt.open()
        return ptt
    return open_line


def acquire(kind, operation, *, prove_line=False, cleanup=None, devices=None,
            ptt_factory=None, fd=None, pid=None, **device_options):
    admission_fd = claim_fd = None
    locked = False
    try:
        if cleanup is None:
            admission_fd = _shared()
        else:
            verify_cleanup(cleanup)
        coexist.checkpoint("after_mutex")
        claim_fd = _open(claim_path()) if fd is None else _fd(fd)
        flags = fcntl.fcntl(claim_fd, fcntl.F_GETFL)
        if flags & os.O_ACCMODE != os.O_RDWR or flags & os.O_APPEND:
            raise ClaimError("claim requires O_RDWR without O_APPEND")
        if ClaimDescriptor(claim_fd).offset() != 0:
            raise ClaimError("acquisition requires a fresh claim descriptor")
        coexist.checkpoint("before_claim_lock")
        try:
            fcntl.flock(claim_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                description = str(holder(claim_fd))
            except ClaimError:
                description = "holder unknown"
            raise ClaimBusy(f"rig claimed: {description}") from None
        locked = True
        _check_inode(claim_fd, claim_path())
        _event("claim_acquired", claim_fd)
        coexist.checkpoint("after_claim_flock")
        _publish(claim_fd, kind=kind, operation=operation, pid=pid)
        check_devices(devices, **device_options)
        coexist.checkpoint("after_device_check")
        if prove_line:
            if devices is not None and devices.ptt_port and ptt_factory is None:
                raise ClaimError("configured PTT line requires a proof factory")
            line_proof(ptt_factory)
        if admission_fd is not None:
            completed, admission_fd = admission_fd, None
            Admission(ClaimDescriptor(claim_fd), completed).failed()
        return ClaimDescriptor(claim_fd)
    except BaseException:
        if claim_fd is not None:
            try:
                if locked:
                    os.ftruncate(claim_fd, 0)
                    _event("claim_closed", claim_fd, ClaimDescriptor(claim_fd).offset(), failed=True)
            finally:
                os.close(claim_fd)
        raise
    finally:
        if admission_fd is not None:
            Admission(ClaimDescriptor(-1), admission_fd).failed()


def close(descriptor):
    fd = _fd(descriptor)
    try:
        _event("claim_closed", fd, ClaimDescriptor(fd).offset())
    finally:
        os.close(fd)


def release(descriptor, *, pid=None):
    _release(descriptor, pid=pid)


def mutex_release(descriptor, *, pid=None):
    """End cleanup after descendant teardown; shell callers then close fd 8."""
    verify_cleanup(descriptor)
    _release(descriptor, pid=pid, cleanup=True)


def _release(descriptor, *, pid=None, cleanup=False):
    descriptor = ClaimDescriptor(_fd(descriptor))
    published = holder(descriptor, cleanup=cleanup)
    info = descriptor.fstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or descriptor.offset() != published.nonce):
        raise ClaimError("release requires the acquired description")
    pid = os.getpid() if pid is None else pid
    if published.pid != pid or not process_matches(pid, published.started):
        raise ClaimError("only the acquirer may release the " + ("mutex" if cleanup else "claim"))
    coexist.checkpoint("before_release")
    os.ftruncate(descriptor.fd, 0)
    try:
        _event("cleanup_released" if cleanup else "claim_released", descriptor.fd, published.nonce)
    finally:
        os.close(descriptor.fd)
    coexist.checkpoint("after_release")


def main(argv=None):
    parser = argparse.ArgumentParser()
    verbs = parser.add_subparsers(dest="verb", required=True)
    paths = verbs.add_parser("path")
    paths.add_argument("--mutex", action="store_true")
    verbs.add_parser("bootstrap")
    spawn = verbs.add_parser("spawn", aliases=["claim_spawn"])
    spawn.add_argument("--fd", type=int, default=9)
    spawn.add_argument("--cleanup", type=int)
    spawn.add_argument("--long-lived", action="store_true")
    spawn.add_argument("--device")
    spawn.add_argument("--timeout", type=float)
    spawn.add_argument("argv", nargs=argparse.REMAINDER)
    for verb in ("mutex", "mutex_release", "acquire", "inherited", "close", "release"):
        sub = verbs.add_parser(verb)
        sub.add_argument("fd", type=int)
        if verb == "mutex":
            sub.add_argument("--timeout", type=float, default=ADMISSION_TIMEOUT)
        elif verb == "acquire":
            sub.add_argument("--kind", required=True)
            sub.add_argument("--operation", required=True)
            sub.add_argument("--cleanup", type=int)
            sub.add_argument("--prove-line", action="store_true")
            sub.add_argument("--cat", choices=("serial", "rigctld", "none"), default="serial")
            sub.add_argument("--cat-port")
            sub.add_argument("--ptt-port")
            sub.add_argument("--capture")
            sub.add_argument("--playback")
        elif verb == "inherited":
            sub.add_argument("--launch", type=int)
    args = parser.parse_args(argv)
    try:
        if args.verb in ("spawn", "claim_spawn"):
            from .native import shell_spawn
            return shell_spawn(args)
        if args.verb in ("path", "bootstrap"):
            paths = bootstrap()
            print(paths[1] if args.verb == "path" and args.mutex else paths[0])
            return 0
        if args.verb == "mutex":
            mutex(args.fd, timeout=args.timeout, pid=os.getppid())
            print("CLAIM MUTEX", flush=True)
        elif args.verb == "mutex_release":
            mutex_release(args.fd, pid=os.getppid())
            print("CLAIM MUTEX RELEASED", flush=True)
        elif args.verb == "acquire":
            factory = line_factory(args.ptt_port) if args.prove_line else None
            acquire(args.kind, args.operation, fd=args.fd, pid=os.getppid(),
                    cleanup=args.cleanup, prove_line=args.prove_line, ptt_factory=factory,
                    devices=Devices(args.cat_port, args.ptt_port, args.capture, args.playback, args.cat))
            print("CLAIM ACQUIRED", flush=True)
        elif args.verb == "inherited":
            inherited(args.fd, launch=False if args.launch is None else args.launch)
            print("CLAIM INHERITED", flush=True)
        elif args.verb == "close":
            close(args.fd)
            print("CLAIM CLOSED", flush=True)
        else:
            release(args.fd, pid=os.getppid())
            print("CLAIM RELEASED", flush=True)
        return 0
    except Exception as exc:
        print(f"CLAIM REFUSED {exc}", file=sys.stderr, flush=True)
        return 2 if isinstance(exc, ClaimBusy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
