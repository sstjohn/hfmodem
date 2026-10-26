# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The tripwire a test's Python carries: no radio's device, no modem port, no card.

Opening a device node other than the few in TEST_DEVICES, which no radio is,
or the explicitly requested, verified tty0tty software pair in Linux
qualification's coexist phase,
or anything under the operator's own state directory,
or connecting to rigctld (4532), VARA (8300, 8301) or ARDOP (8515), is a TRIP: refused before it happens and written to
HERMETIC_WORK/trips, which fails the test. Paths are resolved first, so a
symlink, a `..` or a relative name after a chdir lands where the kernel would
send it. A claim file locked through `fcntl.flock` is written to
HERMETIC_WORK/claims, and `sweep` clears it when the test ends -- but only a
claim named for the run's HERMETIC_TAG. Any other is somebody else's, perhaps
the station's: it is left where it is and the test TRIPs.

`sitecustomize` arms this in every child a test starts, including one that has
dropped every other mark of a test run; hermetic.py arms it in the pytest process
itself, around each test not marked `hardware`.
"""
import glob
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys

TEST_DEVICES = re.compile(r"/dev/(null|zero|urandom|tty|ptmx|ttys\d+|pts/\d+|std(in|out|err)|fd/\d+)")
PORTS = {4532, 8300, 8301, 8515}
CLAIM = re.compile(r"/(private/)?tmp/hfmodem-rig-[^/]*\.claim$")
STATE = os.path.realpath(os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".local/state/hfmodem"))
armed = False
_installed = False
_tty0tty = {}
TTY0TTY_PAIR = ("/dev/tnt0", "/dev/tnt1")
TTY0TTY_RECEIPT = "/var/lib/hfmodem-ci/provisioning.json"
TTY0TTY_COMMIT = "e6334d0cbd5a7956571cd6c01181e3992732af8e"


def _text(path):
    with open(path) as fh:
        return fh.read()


def _trusted(path, kind):
    info = os.lstat(path)
    return kind(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022


def _tty0tty_node(path, major):
    info = os.lstat(path)
    minor = TTY0TTY_PAIR.index(path)
    if (not stat.S_ISCHR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid() or info.st_nlink != 1
            or (os.major(info.st_rdev), os.minor(info.st_rdev)) != (major, minor)
            or _text(f"/sys/class/tty/tnt{minor}/dev").strip() != f"{major}:{minor}"
            or os.path.realpath(f"/sys/dev/char/{major}:{minor}")
            != f"/sys/devices/virtual/tty/tnt{minor}"):
        raise ValueError("not the provisioned tty0tty node")
    return (info.st_dev, info.st_ino, info.st_rdev, info.st_uid, info.st_mode,
            info.st_ctime_ns)


def _validated_tty0tty():
    """A software-pair exception only in the owned Linux coexist execution.

    The root-managed receipt, kernel module and device namespace are trusted;
    participant environments or a matching device name alone grant nothing.
    No device is opened while validating this provenance.
    """
    if os.environ.get("HFMODEM_COEXIST_LINE_PROOF") != "1":
        return {}
    if sys.platform != "linux" or os.environ.get("HFMODEM_QUALIFICATION_PHASE") != "coexist":
        return {}
    try:
        system = os.uname()
        if system.nodename.split(".")[0] not in ("hfmodem-ci", "lima-hfmodem-ci"):
            return {}
        if not all(_trusted(path, stat.S_ISDIR) for path in
                   ("/", "/dev", "/var", "/var/lib", "/var/lib/hfmodem-ci")):
            return {}
        if not _trusted(TTY0TTY_RECEIPT, stat.S_ISREG):
            return {}
        receipt = json.loads(_text(TTY0TTY_RECEIPT))
        module = receipt["tty0tty"]
        if (receipt["schema_version"] != 1 or receipt["vm"] != "hfmodem-ci"
                or receipt["kernel"] != system.release
                or module["upstream"] != "https://github.com/freemed/tty0tty.git"
                or module["commit"] != TTY0TTY_COMMIT
                or module["pair"] != list(TTY0TTY_PAIR)
                or module["owner_uid"] != os.getuid()
                or not re.fullmatch(r"[0-9A-F]+", module["srcversion"])
                or _text("/sys/module/tty0tty/srcversion").strip() != module["srcversion"]):
            return {}
        drivers = [row.split() for row in _text("/proc/tty/drivers").splitlines()
                   if row.split()[:1] == ["tty0tty"]]
        if (len(drivers) != 1 or len(drivers[0]) != 5
                or drivers[0][1] != "/dev/tnt" or drivers[0][3:] != ["0-7", "serial"]):
            return {}
        major = int(drivers[0][2])
        if major <= 0:
            return {}
        return dict(nodes={path: _tty0tty_node(path, major) for path in TTY0TTY_PAIR},
                    major=major, srcversion=module["srcversion"])
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _tty0tty_open(event, original, path):
    if os.environ.get("HFMODEM_COEXIST_LINE_PROOF") != "1":
        return False
    if (event != "open" or original != path or path not in _tty0tty.get("nodes", {})
            or os.environ.get("HFMODEM_QUALIFICATION_PHASE") != "coexist"):
        return False
    try:
        # /dev is root-managed, so participants cannot replace an entry between
        # this audit check and open. Concurrent privileged namespace/module
        # changes are outside this Python tripwire's trust boundary.
        return (_trusted("/dev", stat.S_ISDIR)
                and _text("/sys/module/tty0tty/srcversion").strip() == _tty0tty["srcversion"]
                and _tty0tty_node(path, _tty0tty["major"]) == _tty0tty["nodes"][path])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _record(name, line):
    work = os.environ.get("HERMETIC_WORK")
    if work:
        with open(os.path.join(work, name), "a") as fh:
            fh.write(line + "\n")


def trip(what):
    _record("trips", f"python {what}")
    print(f"!! HERMETIC TRIP: python {what}", file=sys.stderr)
    raise PermissionError(f"HERMETIC TRIP: {what}")


def _fd_path(fd):
    import fcntl
    if hasattr(fcntl, "F_GETPATH"):
        return os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0"))
    return os.readlink(f"/proc/self/fd/{fd}")


def guard(event, args):
    if not armed:
        return
    if event in ("open", "os.mkdir") and isinstance(args[0], (str, bytes, os.PathLike)):
        original = os.fsdecode(args[0])
        path = os.path.realpath(original)
        if (path.startswith("/dev/") and not TEST_DEVICES.fullmatch(path)
                and not _tty0tty_open(event, original, path)):
            trip(f"opened {path}")
        if path == STATE or path.startswith(STATE + os.sep):
            trip(f"reached the operator's state directory at {path}")
    elif event == "socket.connect" and isinstance(args[1], tuple) and args[1][1:2] and args[1][1] in PORTS:
        trip(f"connected to {args[1][0]}:{args[1][1]}")
    elif event == "fcntl.flock":
        try:
            path = _fd_path(args[0])
        except OSError:
            return
        if CLAIM.search(path):
            _record("claims", path)


def arm():
    global armed, _installed, _tty0tty
    _tty0tty = _validated_tty0tty()
    if not _installed:
        sys.addaudithook(guard)
        _installed = True
    armed = True


def disarm():
    global armed, _tty0tty
    armed = False
    _tty0tty = {}


def _held(path):
    lsof = shutil.which("lsof", path=os.environ.get("HERMETIC_PATH") or os.environ.get("PATH"))
    return bool(lsof and subprocess.run([lsof, "-t", path], capture_output=True, text=True).stdout.split())


def sweep(work, tag):
    """Clear the claims this run took: every recorded one named for `tag`, and,
    for a claim a shell or another non-Python child took, every one the tag
    names. A recorded claim without the tag is a TRIP and stays put."""
    try:
        recorded = {os.path.realpath(p) for p in open(os.path.join(work, "claims")).read().split()}
    except FileNotFoundError:
        recorded = set()
    named = {os.path.realpath(p) for p in glob.glob(f"/tmp/hfmodem-rig-*{tag}*.claim")}
    for path in sorted(recorded | named):
        if tag not in os.path.basename(path):
            _record("trips", f"claim {path} is not this run's -- left in place")
            continue
        if f"no-cat.{tag}" in path:
            _record("trips", f"claim {path} taken by a leg that holds one already")
        if os.path.exists(path) and not _held(path):
            os.unlink(path)


if __name__ == "__main__" and sys.argv[1:2] == ["sweep"]:
    sweep(os.environ["HERMETIC_WORK"], os.environ["HERMETIC_TAG"])
