# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import ast
import ctypes
from dataclasses import dataclass, field
import fcntl
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import termios
import threading
import time


MACOS_TRACE_GAP = ('macOS has no unprivileged syscall tracer; short native one-shots '
                   'require pinned barriers and the matching Linux trace')
MACOS_PTY_GAP = ('macOS ptys permit a non-root reopen despite TIOCEXCL '
                 '(XNU bsd/kern/tty_dev.c); exclusion requires the real-adapter hw subset')


class ObservationError(RuntimeError):
    pass


def clock_pair(point, participant_monotonic_ns=None):
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return dict(point=point, time_ns=wall, monotonic_ns=(before + after) // 2,
                uncertainty_ns=after - before, participant_monotonic_ns=participant_monotonic_ns)


def proc_started(stat, expected_pid=None):
    opening, closing = stat.find('('), stat.rfind(')')
    pid = stat[:opening].strip()
    if opening < 1 or closing < opening or not pid.isdecimal() or int(pid) <= 0:
        raise ObservationError('invalid process PID identity')
    if expected_pid is not None and int(pid) != expected_pid:
        raise ObservationError('invalid process PID identity')
    fields = stat[closing + 2:].split()
    if len(fields) <= 19 or not fields[19].isdecimal():
        raise ObservationError('invalid process start identity')
    return fields[19]


def linux_run_start_floor():
    """A conservative /proc start-time bound, captured before the run UUID.

    Linux starttime and CLOCK_BOOTTIME include suspend time. Leave a full user
    clock tick of margin, so a process born on the boundary is never excluded.
    """
    frequency = os.sysconf('SC_CLK_TCK')
    now = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    if type(frequency) is not int or frequency <= 0 or type(now) is not int or now < 0:
        raise ObservationError('invalid process run clock')
    return max(0, now * frequency // 1_000_000_000 - 1)


def _linux_uids(directory):
    rows = [row.split()[1:] for row in (directory / 'status').read_text().splitlines()
            if row.startswith('Uid:')]
    if len(rows) != 1 or len(rows[0]) != 4 or not all(value.isdecimal() for value in rows[0]):
        raise ObservationError(f'invalid process UID identity: {directory.name}')
    return tuple(map(int, rows[0]))


def _linux_identity_unchanged(directory, start, uids):
    if proc_started((directory / 'stat').read_text(), int(directory.name)) != start:
        raise ObservationError(f'pid reused during participant discovery: {directory.name}')
    if _linux_uids(directory) != uids:
        raise ObservationError(f'credentials changed during participant discovery: {directory.name}')


class _ExitConfirmationBudget:
    """Bound additional stat/UID/stat confirmations, shared by one capture.

    Initial metadata reads and immediate terminal proofs do not spend this budget.
    Only a denied PF_EXITING identity spends these additional-check budgets.
    Deadlines start once and never reset across calls for the same identity.
    """
    def __init__(self):
        self.deadline_ns = None
        self.checks = 0
        self.identities = {}

    def allow(self, directory, start, uids):
        now = time.monotonic_ns()
        if self.deadline_ns is None:
            self.deadline_ns = now + 25_000_000
        key = (int(directory.name), start, tuple(uids))
        identity = self.identities.setdefault(key, dict(deadline_ns=now + 5_000_000, checks=0))
        if (now >= self.deadline_ns or now >= identity['deadline_ns']
                or self.checks >= 128 or identity['checks'] >= 32):
            return False
        self.checks += 1
        identity['checks'] += 1
        return True

    def yield_to_exit(self, directory, start, uids):
        identity = self.identities[(int(directory.name), start, tuple(uids))]
        if self.checks >= 128 or identity['checks'] >= 32:
            return
        remaining = min(self.deadline_ns, identity['deadline_ns']) - time.monotonic_ns()
        if remaining > 0:
            # Let strace/the exiting task run instead of spending every check
            # in one tight loop. allow() rechecks deadlines before any new read.
            time.sleep(min(.001, remaining / 1_000_000_000))


def _linux_terminal(directory, start, uids, denied=None, budget=None):
    """After denied proc data, prove that this identity has no live resources.

    Linux terminal tasks retain readable stat/status but can deny environ/fd access.
    Z/X states, bracketed around the UID check, are terminal observations;
    a denial alone says nothing about whether the process is still live. After
    denial, PF_EXITING can authorize bounded confirmation of Z/X/disappearance;
    neither that flag nor an exhausted budget establishes terminal absence.
    """
    first_stat = None
    previous_stat = None
    last_stat = None
    # Linux 6.8 sched.h maps EXIT_ZOMBIE to Z and EXIT_DEAD to X.
    # wait_task_zombie() changes Z to EXIT_DEAD before release_task().
    terminal_states = ('Z', 'X')

    def metadata():
        nonlocal first_stat, previous_stat, last_stat
        text = (directory / 'stat').read_text()
        previous_stat, last_stat = last_stat, text
        if first_stat is None:
            first_stat = text
        if proc_started(text, int(directory.name)) != start:
            raise ObservationError(f'pid reused during terminal verification: {directory.name}')
        return text.rpartition(')')[2].split()

    def exiting(fields):
        # include/linux/sched.h in the installed Linux 6.8 headers defines
        # PF_EXITING as 0x00000004 ("Getting shut down"). This is not a proof
        # of closed resources: Z/X/disappearance must still be observed.
        if not fields[6].isascii() or not fields[6].isdecimal():
            raise ObservationError(f'invalid process flags identity: {directory.name}')
        return bool(int(fields[6]) & 0x00000004)

    def refused(reason):
        if denied is not None:
            fields = first_stat.rpartition(')')[2].split()
            denied.add_note(
                f'nonterminal proc denial: pid={directory.name}, state={fields[0]!r}, '
                f'start={fields[19]!r}, flags={fields[6]!r}, expected_uids={uids!r}, '
                f'raw_stat={first_stat!r}, previous_stat={previous_stat!r}, '
                f'last_stat={last_stat!r}, reason={reason}')
        return False

    try:
        fields = metadata()
        if fields[0] in terminal_states:
            if _linux_uids(directory) != uids:
                raise ObservationError(f'credentials changed during terminal verification: {directory.name}')
            if metadata()[0] not in terminal_states:
                raise ObservationError(f'state changed during terminal verification: {directory.name}')
            return True
        if denied is None or not exiting(fields):
            return refused('no denied exiting identity')
        budget = _ExitConfirmationBudget() if budget is None else budget
        while budget.allow(directory, start, uids):
            before = metadata()
            if _linux_uids(directory) != uids:
                raise ObservationError(f'credentials changed during terminal verification: {directory.name}')
            after = metadata()
            if before[0] in terminal_states:
                if after[0] not in terminal_states:
                    raise ObservationError(f'state changed during terminal verification: {directory.name}')
                return True
            if not exiting(before) or (after[0] not in terminal_states and not exiting(after)):
                return refused('exit flag no longer present')
            budget.yield_to_exit(directory, start, uids)
        return refused(f'terminal confirmation budget exhausted after {budget.checks} shared checks')
    except (FileNotFoundError, ProcessLookupError):
        return True
    except (ObservationError, PermissionError) as exc:
        exc.add_note(f'terminal verification metadata: first_stat={first_stat!r}, '
                     f'previous_stat={previous_stat!r}, last_stat={last_stat!r}, '
                     f'expected_uids={uids!r}')
        raise


def linux_participants(run, proc=Path('/proc'), *, started_after=None, known=None, budget=None):
    """Find exact run markers, including unlogged children outside our group.

    A trusted bound means the UUID was created here after that boot-tick floor
    and participants were launched as new descendants; pre-existing process
    adoption/exec is unsupported. Only a never-observed, older process with a
    foreign-owned unreadable environment can then be proved outside the run.
    Without that bound, denied same-UID discovery remains an error.
    """
    if started_after is not None and (type(started_after) is not int or started_after < 0):
        raise ObservationError('invalid participant run start bound')
    known = {} if known is None else known
    if not isinstance(known, dict) or any(
            type(pid) is not int or pid <= 0 or not isinstance(info, dict)
            or not isinstance(info.get('start'), str) or not info['start'].isdecimal()
            or not isinstance(info.get('uids'), (tuple, list)) or len(info['uids']) != 4
            or any(type(uid) is not int or uid < 0 for uid in info['uids'])
            for pid, info in known.items()):
        raise ObservationError('invalid known participant identity')
    found = {}
    budget = _ExitConfirmationBudget() if budget is None else budget
    uid = os.getuid()
    marker = f'HFMODEM_COEXIST_RUN={run}'.encode()
    for directory in proc.iterdir():
        if not directory.name.isdecimal():
            continue
        pid = int(directory.name)
        try:
            before = proc_started((directory / 'stat').read_text(), pid)
            uids = _linux_uids(directory)
            previous = known.get(pid)
            if previous is not None and (previous['start'] != before or tuple(previous['uids']) != uids):
                raise ObservationError(f'known participant identity changed: {pid}')
            if uid not in uids:
                if previous is not None:
                    raise ObservationError(f'known participant credentials are foreign: {pid}')
                _linux_identity_unchanged(directory, before, uids)
                continue
            environment_path = directory / 'environ'
            try:
                environment = environment_path.read_bytes().split(b'\0')
            except PermissionError as exc:
                if _linux_terminal(directory, before, uids, denied=exc, budget=budget):
                    continue
                exc.add_note(f'participant scope: start_floor={started_after!r}, '
                             f'observed_start={before!r}, observed_uids={uids!r}, '
                             f'known_identity={None if previous is None else (previous["start"], tuple(previous["uids"]))!r}')
                if previous is not None or started_after is None or int(before) >= started_after:
                    raise
                owner = environment_path.stat().st_uid
                if owner == uid:
                    raise
                _linux_identity_unchanged(directory, before, uids)
                if environment_path.stat().st_uid != owner:
                    raise ObservationError(f'environment owner changed during participant discovery: {pid}')
                continue
            if marker not in environment:
                _linux_identity_unchanged(directory, before, uids)
                continue
            argv = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            _linux_identity_unchanged(directory, before, uids)
            found[pid] = dict(start=before, argv=argv, uids=uids)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise ObservationError(f'participant discovery denied: {directory}') from exc
    return found


def macos_participants(run, runner=subprocess.run):
    result = runner(['ps', '-E', '-axo', 'pid=,lstart=,command='],
                    capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise ObservationError(f'ps participant discovery failed: {result.stderr}')
    found = {}
    marker = re.compile(r'(?:^|\s)HFMODEM_COEXIST_RUN=' + re.escape(run) + r'(?:\s|$)')
    for line in result.stdout.splitlines():
        parts = line.split(None, 6)
        if marker.search(line):
            if len(parts) != 7 or not parts[0].isdecimal():
                raise ObservationError('malformed run-marked ps row')
            found[int(parts[0])] = dict(start=' '.join(parts[1:6]),
                                       event_start='_'.join(parts[1:6]), argv=parts[6])
    return found


def fdinfo(text):
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(':')
        if key == 'pos':
            fields['pos'] = int(value.strip())
        elif key == 'lock':
            fields.setdefault('locks', []).append(value.strip())
    if 'pos' not in fields:
        raise ObservationError('fdinfo lacks pos')
    fields.setdefault('locks', [])
    return fields


def linux_descriptors(pid, proc=Path('/proc')):
    result = []
    for entry in (proc / str(pid) / 'fd').iterdir():
        try:
            target = os.readlink(entry)
            info = entry.stat()
            details = fdinfo((proc / str(pid) / 'fdinfo' / entry.name).read_text())
            result.append(dict(fd=int(entry.name), path=target, dev=info.st_dev,
                               ino=info.st_ino, **details))
        except FileNotFoundError:
            continue
    return result


def parse_lsof(text):
    result = {}
    pid = None
    current = None
    for line in text.splitlines():
        if not line:
            continue
        tag, value = line[0], line[1:]
        if tag == 'p':
            pid = int(value)
            result.setdefault(pid, [])
            current = None
        elif tag == 'f':
            current = None
            if value.isdecimal() and pid is not None:
                current = dict(fd=int(value))
                result[pid].append(current)
        elif current is not None:
            if tag == 'n':
                current['path'] = value
            elif tag == 'i':
                current['ino'] = int(value)
            elif tag == 'D':
                current['dev'] = int(value, 16)
            elif tag == 'o':
                if not value.startswith(('0t', '0x')):
                    raise ObservationError('unknown lsof offset format')
                current['pos'] = int(value[2:], 10 if value.startswith('0t') else 16)
    return result


def probe_claim(path, *, quiescent):
    if not quiescent:
        raise ObservationError('claim probe requires a quiescent barrier')
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    finally:
        os.close(fd)


@dataclass
class Snapshot:
    clock: dict
    participants: dict
    descriptors: dict
    locks: str | None = None
    claim_free: bool | None = None


class _PollingObserver:
    def start(self):
        self._stop = threading.Event()
        self._errors = []

        def sample():
            while not self._stop.wait(self.interval):
                try:
                    self.capture()
                except BaseException as exc:
                    self._errors.append(exc)
                    return

        self.capture()
        self._thread = threading.Thread(target=sample, name='coexist-observer', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(6)
        if self._thread.is_alive():
            raise ObservationError('observer did not stop')
        if self._errors:
            raise ObservationError('observer sampling failed') from self._errors[0]


class LinuxObserver(_PollingObserver):
    interval = .1

    def __init__(self, run, *, proc=Path('/proc'), started_after=None):
        self.run = run
        self.proc = Path(proc)
        self.started_after = started_after
        self._known = {}
        self._capture_lock = threading.Lock()
        self.snapshots = []

    def capture(self, point='sample', participant_monotonic_ns=None):
        with self._capture_lock:
            budget = _ExitConfirmationBudget()
            participants = linux_participants(self.run, self.proc,
                                             started_after=self.started_after, known=self._known, budget=budget)
            self._known.update(participants)
            descriptors = {}
            for pid, info in participants.items():
                try:
                    descriptors[pid] = linux_descriptors(pid, self.proc)
                    if proc_started((self.proc / str(pid) / 'stat').read_text(), pid) != info['start']:
                        raise ObservationError(f'pid reused during descriptor sampling: {pid}')
                except FileNotFoundError:
                    descriptors.pop(pid, None)
                except PermissionError as exc:
                    if not _linux_terminal(self.proc / str(pid), info['start'], info['uids'], denied=exc, budget=budget):
                        raise
                    descriptors.pop(pid, None)
            # Terminal identities remain in _known, so a reused PID or later
            # denied live identity cannot inherit a zombie exemption.
            participants = {pid: info for pid, info in participants.items() if pid in descriptors}
            snapshot = Snapshot(clock_pair(point, participant_monotonic_ns), participants,
                                descriptors, (self.proc / 'locks').read_text())
            self.snapshots.append(snapshot)
            return snapshot

    barrier = capture


class MacOSObserver(_PollingObserver):
    interval = 1.0
    trace_gap = MACOS_TRACE_GAP

    def __init__(self, run, *, runner=subprocess.run):
        self.run = run
        self.runner = runner
        self.snapshots = []

    def capture(self, point='sample', participant_monotonic_ns=None):
        participants = macos_participants(self.run, self.runner)
        descriptors = {}
        if participants:
            result = self.runner(['lsof', '-nP', '-o', '-a', '-p',
                                  ','.join(map(str, sorted(participants))), '-FpcfDino'],
                                 capture_output=True, text=True, timeout=5)
            if result.returncode not in (0, 1) or result.stderr.strip():
                raise ObservationError(f'lsof failed: {result.stderr}')
            descriptors = parse_lsof(result.stdout)
            after = macos_participants(self.run, self.runner)
            for pid, before in participants.items():
                if pid in after and before['start'] != after[pid]['start']:
                    raise ObservationError(f'pid reused during lsof: {pid}')
                if pid in after and pid not in descriptors:
                    raise ObservationError(f'lsof omitted live participant: {pid}')
        snapshot = Snapshot(clock_pair(point, participant_monotonic_ns), participants, descriptors)
        self.snapshots.append(snapshot)
        return snapshot

    def barrier(self, point, *, claim_path=None, quiescent=False, participant_monotonic_ns=None):
        snapshot = self.capture(point, participant_monotonic_ns)
        if claim_path is not None:
            snapshot.claim_free = probe_claim(claim_path, quiescent=quiescent)
        return snapshot


IN_OPEN = 0x20
IN_CLOSE = 0x18
IN_Q_OVERFLOW = 0x4000


class Inotify:
    def __init__(self, paths):
        if sys.platform != 'linux':
            raise ObservationError('inotify requires Linux')
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        self.watches = {}
        self.events = []
        try:
            for path in paths:
                wd = libc.inotify_add_watch(self.fd, os.fsencode(path), IN_OPEN | IN_CLOSE)
                if wd < 0:
                    raise OSError(ctypes.get_errno(), f'inotify_add_watch: {path}')
                self.watches[wd] = str(path)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def decode(raw):
        events = []
        offset = 0
        while offset < len(raw):
            if len(raw) - offset < 16:
                raise ObservationError('truncated inotify record')
            wd, mask, cookie, size = struct.unpack_from('iIII', raw, offset)
            offset += 16 + size
            if offset > len(raw):
                raise ObservationError('truncated inotify name')
            events.append(dict(wd=wd, mask=mask, cookie=cookie))
        return events

    def drain(self):
        while True:
            try:
                raw = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            for event in self.decode(raw):
                self.events.append(dict(event, path=self.watches.get(event['wd']),
                                        monotonic_ns=time.monotonic_ns()))
        return list(self.events)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class LineSampler:
    label = 'non-HUPCL'

    def __init__(self, sense):
        self.sense = sense
        self.samples = []
        self.errors = []
        self._stop = threading.Event()

    def start(self):
        def sample():
            deadline = time.monotonic()
            while not self._stop.is_set():
                try:
                    before = time.monotonic_ns()
                    level = self.sense()
                    after = time.monotonic_ns()
                    self.samples.append((before, after, level))
                except BaseException as exc:
                    self.errors.append(exc)
                    return
                deadline += .001
                self._stop.wait(max(0, deadline - time.monotonic()))

        self._thread = threading.Thread(target=sample, name='coexist-line-1khz', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(2)
        if self._thread.is_alive() or self.errors:
            raise ObservationError('line sampler failed')
        return self.samples


def modem_bits(fd):
    return struct.unpack('i', fcntl.ioctl(fd, termios.TIOCMGET, struct.pack('i', 0)))[0]


def set_modem_bits(fd, bits):
    fcntl.ioctl(fd, termios.TIOCMSET, struct.pack('i', bits))


@dataclass
class Trace:
    records: list = field(default_factory=list)
    pids: set = field(default_factory=set)
    exited: set = field(default_factory=set)


def parse_trace(text):
    trace = Trace()
    pending = {}
    if not text or not text.endswith('\n'):
        raise ObservationError('empty or truncated trace')
    for line in text.splitlines():
        match = re.fullmatch(r'(?:\[pid\s+(\d+)\]|(\d+))\s+(\d+\.\d+)\s+(.*)', line)
        if not match:
            raise ObservationError(f'unattributed or malformed trace record: {line[:120]}')
        pid = int(match[1] or match[2])
        stamp = round(float(match[3]) * 1e9)
        body = match[4]
        trace.pids.add(pid)
        if body.startswith('+++ exited with ') or body.startswith('+++ killed by '):
            trace.exited.add(pid)
            pending.pop(pid, None)
            continue
        if body.startswith('--- '):
            continue
        if body.endswith('<unfinished ...>'):
            if pid in pending:
                raise ObservationError('nested unfinished syscall')
            pending[pid] = (stamp, body[:-len('<unfinished ...>')])
            continue
        if body.startswith('<... '):
            resumed = re.match(r'<\.\.\. (\w+) resumed>(.*)', body)
            if not resumed or pid not in pending:
                raise ObservationError('unmatched resumed syscall')
            stamp, prefix = pending.pop(pid)
            if not prefix.startswith(resumed[1] + '('):
                raise ObservationError('resumed syscall mismatch')
            body = prefix + resumed[2]
        call = re.fullmatch(r'(\w+)\((.*)\)\s+=\s+(.*)', body)
        if not call:
            raise ObservationError(f'incomplete syscall: {body[:120]}')
        name, args, result = call.groups()
        numeric = re.match(r'(-?\d+)', result)
        value = int(numeric[1]) if numeric else None
        record = dict(pid=pid, time_ns=stamp, syscall=name, args=args, result=result, value=value)
        if name in ('open', 'openat', 'openat2', 'creat') and value is not None and value >= 0:
            annotated = re.match(r'\d+<([^>]+)>', result)
            quoted = re.search(r'"(?:[^"\\]|\\.)*"', args)
            if annotated and re.fullmatch(r'pipe:\[[0-9]+\]', annotated[1]):
                # Opening a /dev/fd alias to a process-substitution pipe has a
                # kernel pipe identity, not a filesystem path, under strace -yy.
                record['fd_target'] = annotated[1]
            elif annotated:
                record['path'] = annotated[1]
            elif quoted:
                record['path'] = ast.literal_eval(quoted[0])
            else:
                raise ObservationError('successful open lacks path')
            if 'path' in record and not os.path.isabs(record['path']):
                raise ObservationError('relative open requires strace -yy descriptor paths')
        if name in ('clone', 'clone3', 'fork', 'vfork') and value is not None and value > 0:
            trace.pids.add(value)
        trace.records.append(record)
    if pending:
        raise ObservationError('trace ends inside syscall')
    return trace


class TracedProcess:
    def __init__(self, argv, *, directory, env, pass_fds=(), tracer='strace', stdin=None, cwd=None):
        if sys.platform != 'linux':
            raise ObservationError(MACOS_TRACE_GAP)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.path = directory / 'trace'
        self._out = (directory / 'stdout').open('wb')
        self._err = (directory / 'stderr').open('wb')
        self.argv = [tracer, '-f', '-ttt', '-yy', '-s', '4096', '--seccomp-bpf',
                     '-e', 'trace=%process,%desc,%file,flock,ioctl',
                     '-o', str(self.path), '--', *map(str, argv)]
        try:
            self.process = subprocess.Popen(self.argv, env=env, pass_fds=pass_fds, cwd=cwd,
                                            stdin=stdin, stdout=self._out, stderr=self._err)
        except BaseException:
            self._out.close()
            self._err.close()
            raise

    def finish(self, timeout=10):
        try:
            code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ObservationError('trace tree did not finish') from exc
        finally:
            self._out.close()
            self._err.close()
        if code < 0:
            raise ObservationError('tracer exited by signal')
        diagnostic = '\n'.join(line for line in Path(self._err.name).read_text().splitlines()
                               if line.startswith('strace:'))
        if any(word in diagnostic for word in ('ptrace(', 'detached', 'seccomp-bpf is not enabled',
                                               'seccomp-bpf is not supported', 'invalid system call')):
            raise ObservationError('tracer failed: ' + diagnostic[-1000:])
        trace = parse_trace(self.path.read_text())
        if trace.pids != trace.exited:
            raise ObservationError(f'trace ends before its tree: {trace.pids - trace.exited}')
        return trace
