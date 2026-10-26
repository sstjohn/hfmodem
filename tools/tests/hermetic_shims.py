# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""What hermetic.sh puts on PATH in front of the tools that reach the station.

Each of rigctl, rigctld, nc, lsof, ps, pgrep, pkill and killall runs this with
its own name. A call scoped to the test -- its own processes, its own files under
WORK, its own free port -- goes through; a listing is cut down to the test's own
processes; anything else is a TRIP, written to WORK/trips and failing the test,
and is never passed on. `own`, `kill` and `reap` are hermetic.sh's bookkeeping.

The test's own processes are its descendants, the pids and process groups it
recorded, each checked against the start time it had when recorded so that a pid
the system has since handed to something else is never taken for one of ours,
and whatever still carries this run's HERMETIC_PID and HERMETIC_WORK in its
environment: the native supervisor's children are reparented the moment it
exits. The pid alone would adopt an earlier run's orphan once the pid comes
round again; no other run has had this WORK.
"""
import ctypes
import os
import re
import shutil
import signal
import subprocess
import sys

WORK = os.environ["HERMETIC_WORK"]
ROOT_PID = int(os.environ["HERMETIC_PID"])
PORT = os.environ["HERMETIC_PORT"]
STATION_PORT = os.environ["HERMETIC_STATION_PORT"]
#: The ports a launcher looks for listeners on, and the test's own stand-ins for them.
STAND_INS = {"4532": PORT, "8737": STATION_PORT}
TAG = os.environ["HERMETIC_TAG"]
OWN = os.path.join(WORK, "own")
REAL_PATH = os.pathsep.join(p for p in os.environ["HERMETIC_PATH"].split(os.pathsep)
                            if p and os.path.realpath(p) != os.path.realpath(os.path.join(WORK, "shim")))


def real(name):
    found = shutil.which(name, path=REAL_PATH)
    if not found:
        sys.exit(f"!! hermetic: no real {name} on PATH")
    return found


def trip(what):
    line = f"{sys.argv[1]} {what}"
    with open(os.path.join(WORK, "trips"), "a") as fh:
        fh.write(line + "\n")
    print(f"!! HERMETIC TRIP: {line}", file=sys.stderr)
    sys.exit(97)


def table():
    """pid -> (ppid, pgid, lstart) for every process, from one listing."""
    out = subprocess.run([real("ps"), "-Ao", "pid=,ppid=,pgid=,lstart="],
                         capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        pid, ppid, pgid, *start = line.split()
        rows[int(pid)] = (int(ppid), int(pgid), " ".join(start))
    return rows


def recorded():
    pids, groups = {}, {}
    try:
        lines = open(OWN).read().splitlines()
    except FileNotFoundError:
        return pids, groups
    for line in lines:
        kind, pid, start = line.split(" ", 2)
        (groups if kind == "group" else pids)[int(pid)] = start
    return pids, groups


def environment(pid):
    """A process's environment as NUL-separated words, never its command line."""
    if os.path.isdir("/proc/self"):
        with open(f"/proc/{pid}/environ", "rb") as fh:
            return fh.read().split(b"\0")
    # KERN_PROCARGS2: argc, the exec path and its padding, argv, then the environment.
    size = ctypes.c_size_t(len(_ARGS))
    if _libc.sysctl((ctypes.c_int * 3)(1, 49, pid), 3, _ARGS, ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), f"no arguments for pid {pid}")
    raw = _ARGS.raw[:size.value]
    argc, rest = int.from_bytes(raw[:4], sys.byteorder), raw[4:]
    start = rest.index(b"\0")
    while start < len(rest) and rest[start] == 0:
        start += 1
    return rest[start:].split(b"\0")[argc:]


if sys.platform == "darwin":
    _libc = ctypes.CDLL(None, use_errno=True)
    _ARGS = ctypes.create_string_buffer(1 << 20)


def marked(pids):
    """Those of `pids` whose environment still carries this run's mark."""
    mark = {f"HERMETIC_PID={ROOT_PID}".encode(), f"HERMETIC_WORK={WORK}".encode()}
    found = set()
    for pid in pids:
        try:
            if mark <= set(environment(pid)):
                found.add(pid)
        except (OSError, ValueError):
            pass
    return found


def ours(rows=None):
    """The test's own live processes."""
    rows = table() if rows is None else rows
    pids, groups = recorded()
    mine = marked(rows)
    for pid, (ppid, pgid, start) in rows.items():
        if pids.get(pid) == start:
            mine.add(pid)
            continue
        leader = rows.get(pgid)
        if pgid in groups and (leader is None or leader[2] == groups[pgid]):
            mine.add(pid)
            continue
        seen, up = set(), pid
        while up in rows and up not in seen and up > 1:
            if up == ROOT_PID:
                mine.add(pid)
                break
            seen.add(up)
            up = rows[up][0]
    return mine


def under_work(path):
    path = os.path.abspath(path)
    return (path.startswith(os.path.realpath(WORK) + os.sep) or path.startswith(WORK + os.sep)
            or os.path.basename(path).startswith("hfmodem-rig-") and TAG in path)


def keep_pids(text, mine, column=0):
    kept = []
    for line in text.splitlines():
        words = line.split()
        if len(words) > column and words[column].isdigit():
            if int(words[column]) in mine:
                kept.append(line)
        elif words:
            kept.append(line)
    return kept


def emit(lines):
    if lines:
        print("\n".join(lines))
    return 0 if any(l.split() and l.split()[0].isdigit() for l in lines) else 1


def ps(args):
    rows = table()
    mine = ours(rows)
    if "-p" in args:
        i = args.index("-p")
        wanted = [int(p) for p in re.split(r"[ ,]+", args[i + 1].strip()) if p.isdigit()]
        keep = [str(p) for p in wanted if p in mine]
        if not keep:
            return 1
        return subprocess.run([real("ps"), *args[:i + 1], ",".join(keep), *args[i + 2:]]).returncode
    # -o, or a cluster ending in it (-Ao, -axo): the next word is the field list.
    at = next((i for i, a in enumerate(args) if re.fullmatch(r"-[A-Za-z]*o", a)), None)
    fields = args[at + 1] if at is not None and at + 1 < len(args) else ""
    if not fields.startswith("pid"):
        trip(f"a process listing without pid first, which cannot be scoped: {' '.join(args)}")
    out = subprocess.run([real("ps"), *args], capture_output=True, text=True).stdout
    return emit(keep_pids(out, mine))


def pgrep(args):
    out = subprocess.run([real("pgrep"), *args], capture_output=True, text=True).stdout
    return emit(keep_pids(out, ours()))


def pkill(args):
    sig = signal.SIGTERM
    rest = list(args)
    if rest and re.fullmatch(r"-(\d+|[A-Za-z]{2,})", rest[0]):
        sig = _signal(rest.pop(0))
    rows = table()
    mine = ours(rows)
    if rest[:1] == ["-P"]:
        parent = int(rest[1])
        if parent not in mine:
            trip(f"children of a process not the test's: {' '.join(args)}")
        targets = [pid for pid, row in rows.items() if row[0] == parent and pid in mine]
    elif rest[:1] == ["-f"] and WORK in rest[1]:
        out = subprocess.run([real("pgrep"), "-f", rest[1]], capture_output=True, text=True).stdout
        targets = [int(p) for p in out.split() if int(p) in mine]
    else:
        trip(f"a kill by pattern that is not scoped to WORK: {' '.join(args)}")
    targets = [pid for pid in targets if pid != os.getpid()]
    for pid in targets:
        os.kill(pid, sig)
    return 0 if targets else 1


#: lsof options that take the next word as their argument.
LSOF_ARG = {"+c", "-c", "-p", "-u", "-g", "-d", "+d", "-D", "+D", "-s", "-T", "-x"}


def keep_records(text, mine):
    """An `-F` listing cut to the test's processes: a `p` line opens a process's
    record and every line up to the next `p` belongs to it."""
    kept, keep = [], False
    for line in text.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            keep = int(line[1:]) in mine
        if keep:
            kept.append(line)
    return kept


def lsof(args):
    rewritten, files, sockets, fields = [], [], False, None
    words = iter(args)
    for a in words:
        if a == "--":
            rewritten.append(a)
            files.extend(words)
            rewritten.extend(files)
            break
        if a.startswith("-i"):
            sockets = True
            for port, stand_in in STAND_INS.items():
                a = a.replace(f":{port}", f":{stand_in}")
            if not re.search(rf":({PORT}|{STATION_PORT})\b", a):
                trip(f"a socket that is not the test's port: {' '.join(args)}")
        elif a.startswith("-F"):
            fields = a[2:]
        elif a in LSOF_ARG:
            rewritten.append(a)
            a = next(words, "")
        elif not a.startswith(("-", "+")):
            files.append(a)
        rewritten.append(a)
    if "-t" not in args and (fields is None or "p" not in fields):
        trip(f"a listing that names no pids: {' '.join(args)}")
    if not files and not sockets:
        trip(f"an unscoped listing: {' '.join(args)}")
    for f in files:
        if not under_work(f):
            trip(f"a file outside WORK: {f}")
    done = subprocess.run([real("lsof"), *rewritten], capture_output=True, text=True)
    sys.stderr.write(done.stderr)
    if "-t" in args:
        return emit(keep_pids(done.stdout, ours()))
    kept = keep_records(done.stdout, ours())
    if kept:
        print("\n".join(kept))
    return 0 if kept else 1


def nc(args):
    if PORT not in args or "4532" in args:
        trip(f"a connection that is not to the test's port: {' '.join(args)}")
    os.execv(real("nc"), ["nc", *args])


def rigctl(args):
    serial = args[args.index("-r") + 1] if "-r" in args else ""
    if not under_work(serial):
        trip(f"a radio that is not a file under WORK: {' '.join(args)}")
    return 1


def own(args):
    kind = "group" if args[:1] == ["--group"] else "pid"
    rows = table()
    with open(OWN, "a") as fh:
        for pid in args[1:] if kind == "group" else args:
            if int(pid) in rows:
                fh.write(f"{kind} {pid} {rows[int(pid)][2]}\n")
    return 0


def _signal(word):
    name = word[1:]
    if name.isdigit():
        return int(name)
    try:
        return signal.Signals["SIG" + name.upper().removeprefix("SIG")]
    except KeyError:
        trip(f"not a signal this shim knows: {word}")


def kill(args):
    sig = signal.SIGTERM
    if args and args[0].startswith("-"):
        sig, args = _signal(args[0]), args[1:]
    mine = ours()
    for pid in args:
        if pid.isdigit() and int(pid) in mine:
            try:
                os.kill(int(pid), sig)
            except ProcessLookupError:
                pass
    return 0


def reap(args):
    rows = table()
    mine = ours(rows) - {ROOT_PID}
    ancestors, up = set(), os.getpid()
    while up in rows and up > 1:
        ancestors.add(up)
        up = rows[up][0]
    for pid in mine - ancestors:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return 0


TOOLS = dict(ps=ps, pgrep=pgrep, pkill=pkill, lsof=lsof, nc=nc, rigctl=rigctl,
             rigctld=lambda a: trip(" ".join(a) or "(no arguments)"),
             killall=lambda a: trip(" ".join(a)), own=own, kill=kill, reap=reap)

if __name__ == "__main__":
    sys.exit(TOOLS[sys.argv[1]](sys.argv[2:]))
