# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import codecs
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import glob
import io
import math
import os
from pathlib import Path
from queue import Empty
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

from hfmodem.core import claim as hardware, down, ftdx10, hamlib, posix
from hfmodem.core.band import AMATEUR_HF_BANDS
from hfmodem.core.native import KILL_WAIT_S, OPEN_TIMEOUT_S, REAP_GRACE_S, spawn_native
from hfmodem.core.recording import control_launch
from hfmodem.core.rigs import CAT_BACKENDS, mode_passband, refuse_device_in_test, resolve_profile
from hfmodem.station import client

from . import launch_io
from .operator import data_root, mode_root
from .launch_io import ENCODING, ERRORS, Reader

ROOT = Path(__file__).resolve().parents[2]
PORTED = frozenset({'check', 'power', 'listen', 'down'})
MAIL_FLIGHTS = frozenset({'vara', 'vara-mail', 'ardop', 'ardop-mail', 'rpr-mail'})
LAST_FLIGHTS = frozenset({'sabir', 'vara-chat', 'vara-txcal', 'rtty', 'pounce'})
AVAILABLE = PORTED | {'pactor', 'pactor-answer', 'preflight'} | MAIL_FLIGHTS | LAST_FLIGHTS
RECORDING_VERBS = ('pactor', 'vara', 'ardop', 'rpr-mail', 'sabir', 'rtty', 'listen', 'pounce')

CHILD_STOP_S = 45.0
READER_CLOSE_S = 1.5
DAEMON_STOP_S = 5.0
DOWN_WAIT_S = 90.0
WITNESS_ALIGN_S = 120.0
CLAIM_CLOSE_S = 1.0
READOUT_S = 15.0
ACKNOWLEDGE_S = 30.0
CAT_QUERY_S = 3.0
NATIVE_COMMAND_S = 10.0
DAEMON_READY_S = 12.0
DAEMON_POLL_S = .1
TUNE_QUERY_S = 5.0
TUNE_SETTLE_S = 1.0
# A flight can unwind its worker and witness, sweep children twice (possession
# and lease), and reap down. Native stopping includes its published grace,
# escalation and the two subsequent waits in stop_native.
DAEMON_UNWIND_S = REAP_GRACE_S + 1 + 2 * KILL_WAIT_S + 2 * DAEMON_STOP_S
UNWIND_GRACE_S = (5 * CHILD_STOP_S + 3 * READER_CLOSE_S + DAEMON_UNWIND_S
                  + hardware.ADMISSION_TIMEOUT + DOWN_WAIT_S + WITNESS_ALIGN_S + 3 * CLAIM_CLOSE_S
                  + READOUT_S + CHILD_STOP_S + READER_CLOSE_S)


def pounce_stage_bounds():
    from hfmodem.core.busy import POUNCE_WINDOW_S, WINDOW_S
    native = (hardware.ADMISSION_TIMEOUT + OPEN_TIMEOUT_S + NATIVE_COMMAND_S
              + REAP_GRACE_S + 2 * KILL_WAIT_S)
    tcp_tune = TUNE_QUERY_S * (len(ftdx10.session_batch(0).splitlines()) + len(ftdx10.READBACK.splitlines()))
    return dict(launch=hardware.ADMISSION_TIMEOUT, cleanup=down.SETTLE_S,
                daemon=hardware.ADMISSION_TIMEOUT + OPEN_TIMEOUT_S + DAEMON_READY_S + CAT_QUERY_S + DAEMON_POLL_S,
                power=2 * (CAT_QUERY_S + native),
                tuning=CAT_QUERY_S + max(2 * CAT_QUERY_S + TUNE_QUERY_S + 3 * native,
                                        tcp_tune + TUNE_SETTLE_S),
                sense=ACKNOWLEDGE_S + max(WINDOW_S, 2 * POUNCE_WINDOW_S),
                receive_level=ACKNOWLEDGE_S + 2, modem_start=ACKNOWLEDGE_S,
                capture_start=hardware.ADMISSION_TIMEOUT)


def pounce_acknowledge(wait):
    return math.ceil(sum(pounce_stage_bounds().values()) + int(wait))


@contextmanager
def daemon_cleanup_bound():
    started = time.monotonic()
    deadline = started + down.SETTLE_S
    previous = signal.getitimer(signal.ITIMER_REAL)
    handler = signal.getsignal(signal.SIGALRM)
    replace = not previous[0] or previous[0] > down.SETTLE_S
    def expired(*_):
        raise hardware.ClaimError('daemon cleanup deadline expired')
    if replace:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    try:
        yield deadline
        if time.monotonic() >= deadline:
            expired()
    finally:
        if replace:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            if previous[0]:
                signal.setitimer(signal.ITIMER_REAL,
                                 max(.001, previous[0] - (time.monotonic() - started)), previous[1])


@contextmanager
def unwind_bound(seconds):
    if os.name != 'nt':
        yield
        return
    previous, started = posix.itimer_left(), time.monotonic()
    posix.set_itimer(min(previous, seconds) if previous else seconds)
    try:
        yield
    finally:
        if previous:
            posix.set_itimer(max(.001, previous - (time.monotonic() - started)))
        else:
            posix.cancel_itimer()


class DaemonReader(Reader):
    def _run(self):
        decoder = codecs.getincrementaldecoder(ENCODING)(errors=ERRORS)
        try:
            for block in self._blocks():
                self.output.write(decoder.decode(block))
                self.output.flush()
            self.output.write(decoder.decode(b'', final=True))
            self.output.flush()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()

    def __init__(self, stream, output):
        self.output = output
        super().__init__(stream)


@contextmanager
def environment(values):
    previous = dict(os.environ)
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def positional(label, value, *, integer=True):
    if value.startswith('-'):
        raise Failure(f"<{label}> was given the flag '{value}'. Put {label} in before the flags -- "
                      f'otherwise it is read as {label} and every flag after it lands one place to the left.')
    return int(value) if integer else value


def key_arguments(words):
    arguments, force, flagged = [], False, False
    for word in words:
        if word == '--force':
            force = flagged = True
        elif word == 'force':
            raise Failure("say --force -- a bare 'force' here used to be swallowed silently")
        else:
            if word.startswith('-'):
                flagged = True
            if not flagged:
                raise Failure(f"'{word}' is one positional more than this verb takes, and nothing here knows "
                              'what to do with it -- the usage at the top of onair.sh says what each verb reads, '
                              'and its flags come after')
            arguments.append(word)
    return arguments, force


def rpr_arguments(words):
    result, iterator = [], iter(words)
    values = {'--mail-send', '--mail-to', '--mail-subject', '--mail-sid', '--mail-out', '--mail-timeout'}
    for word in iterator:
        flag, separator, _ = word.partition('=')
        if flag in {'--mycall', '--idcall'}:
            if not separator:
                next(iterator, None)
        elif word in {'--cwid-optional', '--mail-fetch'} or flag in values:
            result.append(word)
            if flag in values and not separator:
                value = next(iterator, None)
                if value is None:
                    raise Failure('mail flag needs a value')
                result.append(value)
        else:
            raise Failure(f'unknown rpr-mail flag {word}')
    return result


class Failure(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


class Interrupted(KeyboardInterrupt):
    def __init__(self, signum):
        self.code = 128 + signum


class MailControl:
    def __init__(self, seconds):
        self.seconds, self.code = seconds, 0
        self.process = self.deadline = None
        self.sent = 0

    def attach(self, process):
        self.process = process

    def stop(self, signum, frame):
        self.code = self.code or 128 + signum
        self.sent += 1
        if self.deadline is None:
            self.deadline = time.monotonic() + self.seconds
        self.send()

    def send(self):
        if self.process is not None and self.sent:
            count, self.sent = self.sent, 0
            try:
                self.process.stdin.write(b'stop\n' * count)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def poll(self):
        self.send()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise Interrupted(self.code - 128)

    @contextmanager
    def signals(self):
        previous = {sig: signal.signal(sig, self.stop)
                    for sig in (signal.SIGINT, signal.SIGTERM, posix.SIGHUP)}
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


@contextmanager
def signals():
    def stop(signum, frame):
        raise Interrupted(signum)
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, posix.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def say(message):
    print(f'\n\033[1m== {message}\033[0m', flush=True)


@contextmanager
def phase(name):
    path = os.environ.get('HFMODEM_LAUNCH_LOG')
    if not path:
        yield
        return
    started = time.monotonic()
    def record(event):
        try:
            with open(path, 'a', encoding=ENCODING, errors=ERRORS) as log:
                log.write(f'{datetime.now().astimezone().isoformat(timespec="milliseconds")} '
                          f'pid={os.getpid()} mono={time.monotonic():.3f} {name} {event}\n')
        except OSError as exc:
            # Diagnostic failures must not prevent hardware cleanup.
            print(f'launcher phase log {path}: {exc}', file=sys.stderr, flush=True)
    record('BEGIN')
    try:
        yield
    except BaseException as exc:
        record(f'ERROR after {time.monotonic() - started:.3f}s: {exc!r}')
        raise
    else:
        record(f'END after {time.monotonic() - started:.3f}s')


def arguments(argv):
    words, log = [], None
    iterator = iter(argv)
    for word in iterator:
        flag, separator, value = word.partition('=')
        if flag in {'--log', '--mail-password', '--mail-password-file',
                    '--capture-id', '--capture-root', '--capture-parent-id', '--capture-parent-kind'}:
            if not separator:
                try:
                    value = next(iterator)
                except StopIteration:
                    raise Failure(f'{flag} needs a value') from None
            if flag == '--log':
                log = value
            elif flag.startswith('--mail-password'):
                os.environ['WINLINK_PASSWORD'] = (Path(value).read_text().rstrip('\n')
                    if flag.endswith('-file') else value)
        else:
            words.append(word)
    return words, log


def recording_selection(argv):
    """Take the run-wide recording selection out of argv, wherever it stands, as onair.sh does."""
    words, explicit, record, run_dir, no_record = [], None, None, None, False
    iterator = iter(argv)
    for word in iterator:
        flag, separator, value = word.partition('=')
        if flag in {'--recording', '--record', '--run-dir'}:
            if not separator:
                try:
                    value = next(iterator)
                except StopIteration:
                    raise Failure(f'{flag} needs a value', 2) from None
            if flag == '--recording':
                explicit = value
            elif flag == '--record':
                record = value
            else:
                run_dir = value
        elif word == '--no-record':
            no_record = True
        else:
            words.append(word)
    return words, dict(explicit=explicit, record=record, no_record=no_record), record or run_dir


def recording_line(verb, selection):
    """The run's one recording line, carrying the off warning when one is owed.

    One line on one stream: a warning on stderr beside the selection on stdout
    ran together mid-line in a merged transcript."""
    if not verb.startswith(RECORDING_VERBS):
        return None
    from hfmodem.core.recording import resolve_recording
    warned = []
    try:
        selected = resolve_recording(selection['explicit'], record=selection['record'],
                                     no_record=selection['no_record'], report=warned.append)
    except ValueError as exc:
        raise Failure(str(exc), 2) from None
    return f'Recording: {selected.policy} ({selected.source})' + ''.join(
        f" -- {warning.removeprefix('Recording off: ')}" for warning in warned)


class Transcript:
    def __init__(self, stream, output):
        self.stream, self.output = stream, output

    def write(self, text):
        self.output.write(text)
        self.output.flush()
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()
        self.output.flush()

    def isatty(self):
        return self.stream.isatty()


@contextmanager
def transcript(name):
    if not name:
        yield
        return
    root = Path(os.environ.get('HFMODEM_DATA_DIR', ROOT / 'operator-data'))
    path = root / 'working' / datetime.now().strftime('onair-%m%d-%H%M') / f'{name}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = sys.stdout, sys.stderr
    with path.open('a', encoding=ENCODING, errors=ERRORS) as output:
        sys.stdout, sys.stderr = Transcript(stdout, output), Transcript(stderr, output)
        try:
            print(f'  transcript: {path}', flush=True)
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def tree_line():
    if os.environ.get('ONAIR_TREE'):
        return
    def git(*words):
        return subprocess.check_output(['git', '-C', str(ROOT), *words],
                                       stderr=subprocess.DEVNULL, text=True, encoding=ENCODING,
                                       errors=ERRORS, timeout=2).rstrip('\n')
    try:
        head, branch = git('rev-parse', '--short', 'HEAD'), git('rev-parse', '--abbrev-ref', 'HEAD')
        dirty = ', '.join(line[3:] for line in git('status', '--porcelain', '--untracked-files=no').splitlines())
        line = f'  tree: {head} ({branch}) ' + (f'dirty — {dirty}' if dirty else 'clean')
    except (OSError, subprocess.SubprocessError):
        line = '  tree: unknown'
    os.environ['ONAIR_TREE'] = line
    print(line, flush=True)


class Launcher:
    @phase('configuration')
    def __init__(self):
        from . import claim
        self.claim = claim
        from hfmodem.core import config
        path = os.environ.get('HFMODEM_STATION')
        self.cfg = config.load(path) if path else None
        self.name, self.profile = ((self.cfg.selected_rig, self.cfg.resolved_profile)
                                  if self.cfg else resolve_profile()[:2])
        ports = [os.environ.get(env) or (self.profile.get(field, '') if self.name != 'ft891' or
                 self.cfg and self.cfg.resolution_sources.get(field) != 'rig-profile' else '')
                 for field, env in (('cat_port', 'CAT_PORT'), ('ptt_port', 'PTT_PORT'))]
        self.cat_port, self.ptt_port = ports
        refuse_device_in_test(*ports)
        self.audio_in = self.cfg.audio.input if self.cfg else self.profile.get('audio_device', '')
        self.audio_out = self.cfg.audio.output if self.cfg else self.profile.get('audio_device', '')
        os.environ.update(HFMODEM_RIG=self.name, AUDIO_IN=self.audio_in, AUDIO_OUT=self.audio_out)
        executable = hamlib.executable_path('rigctl', required=False, prefer_local=True)
        if executable:
            os.environ['PATH'] = str(Path(executable).parent) + os.pathsep + os.environ.get('PATH', '')
        self.descriptor = self.admission = self.daemon = None
        self.daemon_readers = []
        self.admitted = self.cleaned = False
        self.attempted = False
        self.after_down = None
        self.destination = None

    @property
    def cat_backend(self):
        return CAT_BACKENDS[self.cfg.rig.cat if self.cfg else self.profile.get('cat', 'rigctld')]

    def require_cat(self, *, transmit=False):
        if transmit and self.cfg and self.cfg.rig.cat == 'none':
            raise Failure('transmitting requires a real CAT connection; station CAT is none')
        if self.cat_port or (not transmit and self.cfg and self.cfg.rig.cat == 'none'):
            return
        how = ('list_ports' if os.name == 'nt' else
               'ls /dev/cu.*' if sys.platform == 'darwin' else 'ls /dev/serial/by-id/*')
        nodes = sorted(glob.glob('/dev/cu.usbserial-*') + glob.glob('/dev/cu.usbmodem*')
                       if sys.platform == 'darwin' else glob.glob('/dev/serial/by-id/*'))
        attached = '\n'.join('     ' + node for node in nodes) or '     (nothing — is the interface plugged in?)'
        raise Failure(f'''no serial ports configured — CAT_PORT is not set. That is a station setting,
   not a fault: device names differ on every machine, so these launchers ship
   with none rather than with one that is right here and wrong at the next rig.

     CAT_PORT   the rig's CAT interface — rigctld and rigctl own this one
     PTT_PORT   the line whose RTS keys the rig. Leave it unset on a
                two-interface adapter, where it derives from CAT_PORT

   Find yours with '{how}' — attached right now:
{attached}

   Then, in your shell profile so it outlives this session:
     export CAT_PORT=<the CAT interface above>''')

    def require_hamlib(self):
        if not all(shutil.which(name) for name in ('rigctl', 'rigctld')):
            raise Failure('hamlib is not on PATH, and nothing here can tune or key a radio without it.\n'
                          f'   rigctl {shutil.which("rigctl") or "not found"}, '
                          f'rigctld {shutil.which("rigctld") or "not found"}\n\n'
                          '     brew install hamlib          # or your package manager\n\n'
                          '   For a build of your own, name its bin directory and the launchers will put\n'
                          '   it on PATH for everything they start:\n\n'
                          f'     export HAMLIB_BIN={os.environ.get("HAMLIB_BIN", Path.home() / "src/radio/hamlib/_install/bin")}\n\n'
                          '   Check with: command -v rigctl rigctld')

    @contextmanager
    def lease(self, verb):
        # The console's lease offers a station release; external launchers must
        # instead refuse a held claim, as bash does.
        saved = None
        previous = os.environ.get('RIG_CLAIMED')
        if os.name != 'nt':
            import fcntl
            try:
                saved = fcntl.fcntl(9, fcntl.F_DUPFD_CLOEXEC, 10)
            except OSError as exc:
                if exc.errno != 9:
                    raise
        descriptor = None
        try:
            devices = hardware.Devices(self.cat_port, self.ptt_port, self.audio_in, self.audio_out,
                CAT_BACKENDS[self.cfg.rig.cat if self.cfg else self.profile.get('cat', 'rigctld')])
            with phase('claim acquisition'):
                descriptor = hardware.acquire('shell', verb if verb in {'power', 'pounce'} else 'launch', devices=devices)
            if os.name != 'nt' and descriptor.fd != 9:
                os.dup2(descriptor.fd, 9, inheritable=False)
                hardware.close(descriptor)
                descriptor = hardware.ClaimDescriptor(9)
            os.environ['RIG_CLAIMED'] = str(hardware.claim_path())
            with self.claim.stopping():
                yield descriptor
        finally:
            with self.claim.uninterrupted():
                try:
                    if descriptor is not None:
                        try:
                            with unwind_bound(CHILD_STOP_S):
                                self.claim.end_children()
                        except BaseException:
                            hardware.close(descriptor)
                            raise
                        else:
                            with unwind_bound(CLAIM_CLOSE_S):
                                (hardware.release if self.cleaned else hardware.close)(descriptor)
                            print('CLAIM RELEASED' if self.cleaned else 'CLAIM CLOSED', flush=True)
                finally:
                    if previous is None:
                        os.environ.pop('RIG_CLAIMED', None)
                    else:
                        os.environ['RIG_CLAIMED'] = previous
                    if saved is not None:
                        os.dup2(saved, 9, inheritable=False)
                        os.close(saved)

    @contextmanager
    def possession(self, verb, *, parent=False):
        if os.name == 'nt':
            from hfmodem.core import winproc
            winproc.declare_grace(UNWIND_GRACE_S)
        inherited = bool(os.environ.get('RIG_CLAIMED'))
        with (self.claim.stopping() if inherited else self.lease(verb)) as owned:
            with phase('claim inheritance/admission setup'):
                self.descriptor = hardware.received() if inherited else owned
                print('CLAIM INHERITED' if inherited else 'CLAIM ACQUIRED', flush=True)
                self.admission = hardware.received(launch=True) if inherited else hardware.inherited(owned, launch=True)
            backend = self.cat_backend
            try:
                with signals():
                    yield
            finally:
                with self.claim.uninterrupted(), phase('launcher cleanup'):
                    # The child sweep includes the native TCP delegate.
                    # Down inventories and ends workers before its CAT proof;
                    # sweep the remaining children after that delegate closes.
                    defer_children = backend == 'rigctld'
                    try:
                        self.admission.failed()
                        if not defer_children and (not inherited or os.name == 'nt'):
                            with phase('end children'), unwind_bound(CHILD_STOP_S):
                                self.claim.end_children()
                        # Bash preserves the configured TCP delegate for down's
                        # CAT proof. Serial CAT instead needs the port released.
                        if backend != 'rigctld':
                            self.close_daemon()
                        if self.admitted or parent and not inherited and self.attempted:
                            from . import cleanup
                            with phase('cleanup down'), unwind_bound(hardware.ADMISSION_TIMEOUT + DOWN_WAIT_S):
                                code, output = cleanup.run(self.descriptor,
                                    [sys.executable, str(ROOT / 'tools/onair.py'), 'down'],
                                    DOWN_WAIT_S, encoding=ENCODING, errors=ERRORS)
                            print(output, end='', flush=True)
                            if code != 0:
                                raise Failure('cleanup failed')
                            self.cleaned = True
                            if self.after_down is not None:
                                try:
                                    self.worker('readout', internal=True, timeout=READOUT_S,
                                        command=[sys.executable, str(ROOT / 'tools/onair.py'),
                                                 '--readout', str(self.after_down), self.audio_in])
                                except (Failure, hardware.ClaimError, OSError, subprocess.SubprocessError):
                                    print(f'  readout abandoned after {READOUT_S:g}s', flush=True)
                        else:
                            print('CLAIM NOT ADMITTED launcher initialization or termination failed', flush=True)
                    finally:
                        try:
                            if backend == 'rigctld':
                                self.close_daemon()
                        except BaseException:
                            self.cleaned = False
                            raise
                        finally:
                            try:
                                if defer_children and (not inherited or os.name == 'nt'):
                                    with phase('end children'), unwind_bound(CHILD_STOP_S):
                                        self.claim.end_children()
                            except BaseException:
                                self.cleaned = False
                                raise
                            finally:
                                if inherited:
                                    with unwind_bound(CLAIM_CLOSE_S):
                                        hardware.close(self.descriptor)
                                    print('CLAIM CLOSED', flush=True)

    def close_daemon(self):
        if self.daemon is None:
            return
        from hfcapture import stop_native, close_native
        try:
            with phase('stop rigctld'), unwind_bound(DAEMON_UNWIND_S):
                stop_native(self.daemon, timeout=DAEMON_STOP_S)
        finally:
            for reader in self.daemon_readers:
                with unwind_bound(READER_CLOSE_S):
                    if not reader.close():
                        raise Failure('rigctld output reader did not stop')
            if not self.daemon.alive():
                close_native(self.daemon, stopped=True)
                self.daemon = None

    def pending_line(self, stage):
        admission = self.admission
        return f'CLAIM PENDING {stage}' if admission is not None and admission.active else ''

    def pending(self, stage):
        # Each preparation stage renews an external admission wait, as bash's claim_pending does.
        line = self.pending_line(stage)
        if line:
            print(line, flush=True)

    @phase('launcher admission')
    def admit(self):
        self.admission.opened()
        self.admitted = True

    @phase('hardware preflight')
    def require_hardware(self, *, transmit=False):
        self.require_cat(transmit=transmit)
        disabled = (self.cfg and self.cfg.resolved_profile.get('ptt_port') == ''
                    and self.cfg.resolution_sources.get('ptt_port') == 'station-toml')
        if not self.ptt_port and not disabled:
            from hfmodem.core.ptt import derive_ptt_port
            try:
                self.ptt_port = derive_ptt_port(self.cat_port)
            except ValueError as exc:
                raise Failure(str(exc) + '\nset PTT_PORT or [rig.profiles.<rig>] ptt_port to the line that keys this rig, and run again') from exc
            os.environ['PTT_PORT'] = self.ptt_port
        refuse_device_in_test(self.cat_port, self.ptt_port)
        for label, port, message in (
            ('CAT', self.cat_port, 'is the rig on and cabled?'),
            ('PTT', self.ptt_port, 'set PTT_PORT if the key is not on the second interface of the CAT adapter')):
            if port and os.name != 'nt' and not Path(port).exists():
                raise Failure(f'{label} port {port} missing — {message}')

    @phase('codecs')
    def codecs(self):
        import sounddevice as sd
        from hfmodem.core.devices import find_device
        print(f'  CAT {self.cat_port} / PTT {self.ptt_port} present', flush=True)
        hardware.inherited(self.descriptor)
        self.require_hamlib()
        print(f'  rigctl {shutil.which("rigctl")}', flush=True)
        try:
            for kind, name in (('in', self.audio_in), ('out', self.audio_out)):
                index = find_device(name, kind, required=True)
                print(f"  codec {kind:<3} {index}: {sd.query_devices(index)['name']}", flush=True)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr, flush=True)
            raise Failure(f"the modems cannot bind '{self.audio_in}' — the line above says why") from None

    def tcp(self, commands, timeout=CAT_QUERY_S):
        requests = [line for line in commands.splitlines() if line and line != 'q']
        extended = bool(requests) and all(line.startswith('+') for line in requests)
        framed = os.name == 'nt' and (extended or len(requests) == 1)
        # Do not pipeline q on Windows: rigctld closes its socket immediately
        # and Winsock can abort the connection before we have read the reply.
        # Read the known reply boundary, then let our socket close send EOF.
        payload = ''.join(line + '\n' for line in requests) if framed else commands
        deadline = time.monotonic() + timeout
        reply, completed = bytearray(), 0
        try:
            with phase(f'rigctld TCP {requests!r}'), socket.create_connection(('127.0.0.1', 4532), timeout=timeout) as connection:
                connection.settimeout(max(.001, deadline - time.monotonic()))
                connection.sendall(payload.encode())
                while True:
                    connection.settimeout(max(.001, deadline - time.monotonic()))
                    block = connection.recv(4096)
                    if not block:
                        break
                    reply.extend(block)
                    # Each extended command has its own RPRT terminator. Raw
                    # Yaesu setters have no serial reply, so Hamlib can spend a
                    # full serial timeout on EACH one. Budget per command, not
                    # per batch; arbitrary bytes do not extend the deadline.
                    statuses = len(re.findall(r'^RPRT -?\d+\r?\n',
                        ftdx10.normal(reply.decode(errors='replace')), re.M))
                    if extended and statuses > completed:
                        completed = statuses
                        deadline = time.monotonic() + timeout
                    # Windows rigctld may reset, or leave the socket open,
                    # after the final reply. Its framing proves completion;
                    # EOF and the optional closing q's RPRT add no evidence.
                    if framed and (completed >= len(requests) if extended
                                   else reply.endswith((b'\n', b'\0'))):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('reply deadline expired')
        except OSError as exc:
            raise type(exc)(f'rigctld TCP {requests!r}: {exc}; '
                            f'completed {completed}/{len(requests)}, '
                            f'partial reply {bytes(reply)!r}') from exc
        return ftdx10.normal(reply.decode(errors='replace'))

    def native(self, *words):
        from hfcapture import native_command
        argv = ['rigctl', '-P', 'NONE', '-m', str(self.profile['model']),
                '-r', self.cat_port, '-s', str(self.profile.get('baud', 38400))]
        argv += list(map(str, words))
        return native_command(argv, timeout=NATIVE_COMMAND_S, descriptor=self.descriptor)

    def query(self, commands, pattern, *fallback):
        try:
            reply = self.tcp(commands).splitlines()[0]
        except (OSError, IndexError):
            reply = ''
        if not re.fullmatch(pattern, reply):
            try:
                reply = self.native(*fallback).stdout.split('\n', 1)[0].strip()
            except (hardware.ClaimError, OSError, subprocess.SubprocessError):
                reply = ''
        return reply

    def power(self, dial):
        self.pending('daemon')
        if self.cfg and self.cfg.rig.cat == 'none':
            print('  TX power: CAT none; power setting unavailable', flush=True)
            return
        level = self.query('\\get_level RFPOWER\nq\n', r'[0-9.]+', 'l', 'RFPOWER')
        if not re.fullmatch(r'[0-9.]+', level):
            print('  TX power: CAT did not answer; power setting unavailable', flush=True)
            return
        watts = self.query(f'\\power2mW {level} {dial} PKTUSB\nq\n', r'[0-9]+', '2', level, dial, 'PKTUSB')
        if watts.isdigit():
            value = int(watts)
            print(f'  TX power: RFPOWER {level} -> {value // 1000}.{value % 1000 // 100} W setting (not measured RF output)', flush=True)
        else:
            print(f'  TX power: RFPOWER {level} (setting; RF output unmeasured)', flush=True)

    @phase('rigctld startup')
    def start_daemon(self, *, dummy=False):
        if not dummy:
            self.require_cat(transmit=True)
        # Inventory before spawning preserves the old daemon's CAT handoff.
        self.stop_daemons()
        argv = [hamlib.rigctld_path(), '-P', 'NONE', '-m', '1' if dummy else str(self.profile['model'])]
        if not dummy:
            argv += ['-r', self.cat_port, '-s', str(self.profile.get('baud', 38400))]
        argv += ['--set-conf=ptt_type=None']
        if dummy:
            argv += ['-T', '127.0.0.1', '-t', '4532']
        with phase('rigctld spawn/handle acknowledgement'):
            self.daemon = spawn_native(argv, claim_fd=self.descriptor,
                long_lived=not dummy, device=None if dummy else self.cat_port)
        self.daemon_readers = [DaemonReader(stream, sys.stderr)
                               for stream in (self.daemon.stdout, self.daemon.stderr) if stream is not None]
        with phase('rigctld ptt_type confirmation'):
            deadline = time.monotonic() + DAEMON_READY_S
            while self.daemon.alive() and time.monotonic() < deadline:
                try:
                    reply = self.tcp('\\get_conf ptt_type\nq\n', min(CAT_QUERY_S, deadline - time.monotonic()))
                    if 'ptt_type=None' in reply or reply.strip() == 'None':
                        return
                except OSError:
                    pass
                time.sleep(DAEMON_POLL_S)
            raise Failure('rigctld did not confirm ptt_type=None within 12 s')

    @phase('tune and verify')
    def tune(self, dial):
        if self.name == 'ftdx10':
            reply = self.tcp(ftdx10.session_batch(dial), TUNE_QUERY_S)
            got = ftdx10.session_frequency(reply, daemon=True)
        else:
            self.tcp(f'+\\set_freq {dial}\nq\n')
            reply = self.tcp('+\\get_freq\nq\n', TUNE_QUERY_S)
            match = re.search(r'^Frequency: (\d+)$', reply, re.M)
            got = match[1] if match else ''
        from hfmodem.core.band import QSY_TOLERANCE_HZ
        if not got.isdigit():
            raise Failure('DIAL UNVERIFIED -- the rig reported no frequency, so the\n'
                          '   channel this would key on is unknown. Ask the operator what the display reads.')
        if abs(int(got) - dial) > QSY_TOLERANCE_HZ:
            raise Failure(f'DIAL MISMATCH -- asked {dial}, rig reads {got} (over {QSY_TOLERANCE_HZ} Hz). The rig is\n'
                          '   not where this would transmit. Ask the operator what the display reads.')
        print(f'  dial {got} verified', flush=True)
        if self.name == 'ftdx10':
            ftdx10.session_report(reply, daemon=True)
            time.sleep(TUNE_SETTLE_S)
            try:
                readback = self.tcp(ftdx10.READBACK, TUNE_QUERY_S)
            except OSError as exc:
                ftdx10.readback_report(str(exc), rc=1, daemon=True)
            else:
                ftdx10.readback_report(readback, daemon=True)
        elif self.name == 'ft891':
            try:
                reply = self.tcp('+\\set_level PREAMP 0\n+\\get_level PREAMP\nq\n')
            except OSError as exc:
                print(f'  !! IPO unverified (rigctl: {exc})', flush=True)
                return
            part = ftdx10.daemon_part(reply, 'set_level: PREAMP 0', 'get_level: PREAMP')
            error = ftdx10.reply_failure(part, daemon=True)
            readings = re.findall(r'^(-?\d+)$', part, re.M)
            if error or not readings:
                print(f'  !! IPO unverified (rigctl: {error or "no PREAMP reading (exit 0)"})', flush=True)
            else:
                print('  IPO verified' if readings[0] == '0' else f'  !! PREAMP reads {readings[0]} (not IPO)', flush=True)

    def worker(self, verb, *, internal=False, seconds=None, command=None, ready=None, control=None,
               handles=None, timeout=None, service=None, acknowledge=ACKNOWLEDGE_S):
        admission = hardware.inherited(self.descriptor, launch=True)
        process = reader = None
        admitted = False
        if command and not internal and verb != 'pounce':
            self.pending('modem')
        deadline = time.monotonic() + acknowledge
        expires = time.monotonic() + timeout if timeout is not None else None
        end = None
        try:
            with phase(f'{verb} worker spawn/admission acknowledgement'):
                options = {'stdin': subprocess.PIPE} if control else {}
                env, passed = control_launch(os.environ)
                passed['handles'] = {**(handles or {}), **passed.get('handles', {})}
                process = self.claim.spawn(self.descriptor, command or [sys.executable, '-u', str(ROOT / 'tools/onair.py'),
                    '--worker', verb, self.audio_in], role='python', env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    **options, **passed)
                self.attempted = True
            if control:
                control.attach(process)
            reader = Reader(process.stdout, lines=True)
            def pump():
                # Bound each drain so a talkative modem cannot starve stopping.
                for _ in range(128):
                    try:
                        print(reader.queue.get_nowait(), end='', flush=True)
                    except Empty:
                        break
                if reader.error:
                    raise reader.error
            while True:
                if service:
                    service()
                try:
                    line = reader.queue.get(timeout=.05)
                    if line.rstrip() == 'CLAIM ADMITTED' and not admitted:
                        admitted = True
                        admission.failed()
                        admission = None
                        if not internal:
                            self.admit()
                        if seconds is not None:
                            end = time.monotonic() + seconds
                        if ready:
                            ready(process, pump)
                    else:
                        print(line, end='', flush=True)
                except Empty:
                    pass
                if control:
                    control.poll()
                if reader.error:
                    raise reader.error
                if expires is not None and time.monotonic() >= expires:
                    raise Failure(f'{verb} deadline expired')
                if not admitted and time.monotonic() >= deadline:
                    raise Failure('CLAIM NOT ADMITTED child exited or acknowledgement timed out')
                if end is not None and time.monotonic() >= end and process.poll() is None:
                    with unwind_bound(CHILD_STOP_S):
                        self.claim.end_child(process)
                    end = None
                if process.poll() is not None and reader.eof.is_set() and reader.queue.empty():
                    break
                if process.poll() is not None and not reader.eof.is_set():
                    if end is None or end > time.monotonic() + 2:
                        end = time.monotonic() + 2
                    elif time.monotonic() >= end:
                        raise Failure('CLAIM NOT ADMITTED child output did not close')
            if control and control.code:
                raise Interrupted(control.code - 128)
            if not admitted:
                code = 128 - process.returncode if command and process.returncode < 0 else process.returncode
                raise Failure('CLAIM NOT ADMITTED child exited or acknowledgement timed out', code or 1)
            return 128 - process.returncode if command and process.returncode < 0 else process.returncode
        finally:
            try:
                if process is not None:
                    with unwind_bound(CHILD_STOP_S):
                        self.claim.end_child(process)
            finally:
                if control and process is not None:
                    process.stdin.close()
                if admission is not None:
                    admission.failed()
                if reader is not None:
                    with unwind_bound(READER_CLOSE_S):
                        if not reader.close():
                            raise Failure('launcher output reader did not stop')
                    process.stdout.close()

    def stop_daemons(self):
        self.pending('daemon')
        if os.name == 'nt':
            return
        from hfmodem.core.down_posix import Posix, load_settings, within
        from hfmodem.core.native import NativeChild
        with daemon_cleanup_bound() as deadline:
            settings, audio = load_settings()
            inventory = Posix(settings, audio, client)
            rows = inventory.table()
            matched = {r.pid for r in hardware.holders(self.descriptor, deadline=deadline)}
            for pid in rows:
                if time.monotonic() >= deadline:
                    raise hardware.ClaimError('daemon cleanup deadline expired')
                if pid != os.getpid() and within(pid, os.getpid(), rows) and 'rigctld' in rows[pid][2] and pid in matched:
                    self.claim.end_child(NativeChild(pid, rows[pid][3], None, None, None))

    def cat_free(self):
        self.pending('cat-free')
        if not self.cat_port or self.cfg and self.cfg.rig.cat == 'none':
            return
        if os.name == 'nt':
            import serial
            from hfmodem.core import serialline
            port = serialline.open_port(serial.Serial, self.cat_port, baudrate=self.profile.get('baud', 38400),
                                        rts=False, dtr=False, write_timeout=.25)
            port.close()
            return
        held = subprocess.run(['lsof', self.cat_port], capture_output=True, text=True,
                              encoding=ENCODING, errors=ERRORS, timeout=5)
        if held.returncode == 0 and held.stdout.strip():
            for line in held.stdout.splitlines():
                print('  ' + line, flush=True)
            raise Failure(f'{self.cat_port} is held, and shrike opens it directly -- it would read no\n'
                          '   frequency back and refuse the run (QSY UNVERIFIED). Stop the holder above,\n'
                          '   or run tools/onair.sh down, and call again.')

    def tune_channel(self, dial):
        if self.cfg and self.cfg.rig.cat == 'none':
            print('  CAT none; dial unverified', flush=True)
            return
        try:
            with socket.create_connection(('127.0.0.1', 4532), timeout=CAT_QUERY_S):
                pass
        except OSError:
            self.tune_native(dial)
        else:
            self.tune(dial)

    def tune_native(self, dial):
        if self.name == 'ftdx10':
            commands = ['-o', 'V', 'VFOA', 'F', 'VFOA', dial, 'f', 'VFOA', 'w', 'MD0C;',
                        'L', 'VFOA', 'AGC', 0, 'l', 'VFOA', 'AGC',
                        'L', 'VFOA', 'PREAMP', 0, 'l', 'VFOA', 'PREAMP']
            for command in ftdx10.filter_commands():
                commands.extend(['w', command])
            result = self.native(*commands, 'v')
            reply = ftdx10.normal(result.stdout + result.stderr)
            got = ftdx10.session_frequency(reply, result.returncode)
        else:
            self.native('F', dial)
            result = self.native('f')
            got = result.stdout.split('\n', 1)[0].strip()
        from hfmodem.core.band import QSY_TOLERANCE_HZ
        if not got.isdigit():
            raise Failure('DIAL UNVERIFIED -- the rig reported no frequency, so the\n'
                          '   channel this would key on is unknown. Ask the operator what the display reads.')
        if abs(int(got) - dial) > QSY_TOLERANCE_HZ:
            raise Failure(f'DIAL MISMATCH -- asked {dial}, rig reads {got} (over {QSY_TOLERANCE_HZ} Hz). The rig is\n'
                          '   not where this would transmit. Ask the operator what the display reads.')
        print(f'  dial {got} verified', flush=True)
        if self.name == 'ftdx10':
            ftdx10.session_report(reply, result.returncode)
            time.sleep(TUNE_SETTLE_S)
            try:
                result = self.native('w', 'MD0;', 'w', 'SH0;', 'w', 'RF0;', 'w', 'IS0;')
            except (hardware.ClaimError, OSError, subprocess.SubprocessError) as exc:
                ftdx10.readback_report(str(exc), rc=1)
            else:
                ftdx10.readback_report(ftdx10.normal(result.stdout + result.stderr), result.returncode)
        elif self.name == 'ft891':
            try:
                result = self.native('L', 'PREAMP', 0, 'l', 'PREAMP')
            except (hardware.ClaimError, OSError, subprocess.SubprocessError) as exc:
                print(f'  !! IPO unverified (rigctl: {exc})', flush=True)
                return
            reply = result.stdout + result.stderr
            readings = re.findall(r'^(-?\d+)$', reply, re.M)
            error = ftdx10.reply_failure(reply, result.returncode)
            if error or not readings:
                print(f'  !! IPO unverified (rigctl: {error or "no PREAMP reading (exit 0)"})', flush=True)
            else:
                print('  IPO verified' if readings[0] == '0' else f'  !! PREAMP reads {readings[0]} (not IPO)', flush=True)

    def gate_channel(self, dial, bandwidth, words, *, modem='pactor'):
        from . import sense
        self.pending('gate')
        arguments, force = key_arguments(words)
        wait = 0 if force else int(os.environ.get('POUNCE_WAIT') or '0')
        self.pending('power')
        print(f'  before the gate: setup {time.monotonic() - self.started:.0f}s '
              f'({time.monotonic() - self.started:.0f}s since launch)', flush=True)
        self.tune_channel(dial)
        if modem == 'vara':
            self.gate_rx_level(force)
            print(f'  occupancy check delegated to ready VARA caller (wait budget {wait}s)', flush=True)
            return [*(['--force'] if force else []), *arguments,
                    '--pounce-wait', str(wait), '--listen-first', '8']
        with self.capture() as opened:
            rc = sense.sense_channel(dial, modem, bandwidth, wait, device=self.audio_in, opened=opened,
                environ=dict(os.environ, CLAIM_PENDING_LINE=self.pending_line('sense')))
        if rc == 2:
            raise Failure(f"no '{self.audio_in}' input — this station cannot hear the channel, so it may not call on it")
        if rc == 3:
            raise Failure('', 3)
        if rc == 5:
            ident, kind, at = (sense.ROOT / 'logs/sense/host-free').read_text().split()
            print(f'  host {ident} free on {dial} — {kind}, calling now', flush=True)
            os.environ.update(HOST_FREE_IDENT=ident, HOST_FREE_AT=at)
        elif rc:
            if not force:
                raise Failure(f'channel in use, and the {wait}s wait budget is spent — listen first, '
                              'or pass --force to override', 4)
            print('  --force: calling anyway over an occupant', flush=True)
        self.gate_rx_level(force)
        return arguments

    def gate_rx_level(self, force):
        from . import sense
        self.pending('rx-level')
        with self.capture() as opened:
            rc = sense.rx_level(device=self.audio_in, opened=opened)
        if rc == 2:
            raise Failure(f"no audio from '{self.audio_in}' -- this station cannot hear the channel it is\n"
                          '   about to key on, and --force is not an answer to that')
        if rc:
            if not force:
                raise Failure('the receive level cannot carry this arm -- set the\n'
                              '   gain the line above names and call again, or pass --force to key anyway')
            print('  --force: calling anyway on a receiver at that level', flush=True)

    @contextmanager
    def capture(self):
        with self.claim.launch(self.descriptor) as admission:
            def opened():
                # Setup captures release only their own admission, as claim_worker does.
                with redirect_stdout(io.StringIO()):
                    admission.opened()
            try:
                yield opened
            finally:
                if admission.active:
                    print('CLAIM NOT ADMITTED capture open failed', flush=True)

    def mail_flight(self, words, witness=False):
        from .launch_capture import LegacyEvidence
        self.started = time.monotonic()
        evidence = LegacyEvidence()
        verb = words[0]
        vara, rpr = verb.startswith('vara'), verb == 'rpr-mail'
        mail = verb.endswith('-mail')
        if len(words) < 2 or not words[1]:
            raise Failure('gateway callsign')
        if mail and (len(words) < 3 or not words[2]):
            raise Failure('channel centre Hz')
        gateway = words[1]
        default_centre = '7103800' if vara else '7103500'
        centre = positional('centre', (words[2] or default_centre) if len(words) > 2 else default_centre)
        bandwidth, at = ('2300' if vara else '500'), 3
        if verb == 'vara-mail':
            if len(words) > at and not words[at].startswith('-'):
                bandwidth = words[at]
                if bandwidth not in {'500', '2300', '2750'}:
                    raise Failure(f"BW '{bandwidth}' is not one this station calls -- 500, 2300 or 2750")
                at += 1
        elif not rpr:
            if len(words) > at:
                bandwidth = words[at] or bandwidth
            positional('bandwidth', bandwidth, integer=False)
            at += 1
        dial = centre - 1500
        modem = 'vara' if vara else 'rpr' if rpr else 'ardop'
        title = 'RPR' if rpr else 'VARA' if vara else 'ARDOP'
        self.require_hardware(transmit=True)
        if os.name == 'nt' and not self.ptt_port:
            raise Failure(f'{verb} requires a dedicated PTT port; CAT cannot key this flight')
        with self.possession(verb):
            self.codecs()
            say(f'{title}{" mail" if mail else ""} {os.environ["MYCALL"]} '
                f'{"<->" if mail else "->"} {gateway}, centre {centre} (dial {dial}), BW{bandwidth}')
            if not (self.cat_backend == 'rigctld' and os.environ.get('ONAIR_CAT_DELEGATED') == '1'
                    and os.environ.get('RIG_CLAIMED')):
                self.start_daemon()
            self.power(dial)
            extra = self.gate_channel(dial, bandwidth, words[at:], modem=modem)
            if verb == 'ardop' and extra:
                raise Failure("ardop takes <gw> [centre] [bw] [--force] and "
                              f"nothing after -- '{extra[0]}' would go nowhere")
            if rpr:
                evidence.session.mkdir(parents=True, exist_ok=True)
                evidence.session = Path(tempfile.mkdtemp(prefix='rpr-', dir=evidence.session))
            self.pending('witness')
            with evidence.witness(self, dial, gateway, enabled=witness):
                if vara:
                    expected = {'ft891': 'FT-891', 'ftdx10': 'FTDX-10', 'g90': 'G90', 'x6100': 'X6100'}[self.name]
                    data = Path(os.environ.get('HFMODEM_DATA_DIR', ROOT / 'operator-data'))
                    command = [sys.executable, str(ROOT / 'tools/kestrel_connect.py'),
                        '--gateway', gateway, '--mycall', os.environ['MYCALL'], '--bw', bandwidth,
                        '--tx-device', self.audio_out, '--rx-device', self.audio_in,
                        '--rigctld', '127.0.0.1:4532', '--freq', f'{dial // 1000}.{dial % 1000:03d}',
                        '--arm', '--expect-model', expected, '--ptt-device', self.ptt_port, '--line-ptt',
                        *extra, '--run-dir', str(self.destination or evidence.session),
                        '--gateways', str(data / 'cache/winlink-vara-gateways.csv')]
                    rc = self.worker(verb, command=command)
                    label = 'kestrel_connect.py'
                elif rpr:
                    rc = self.rpr_mail(gateway, centre, evidence, extra)
                    label = 'RPR mail'
                else:
                    command = [sys.executable, '-m', 'hfmodem.besra.host.run_server',
                        '--radio', self.name, '--rigctld', '127.0.0.1:4532',
                        '--audio-in', self.audio_in, '--audio-out', self.audio_out,
                        '--channel', str(centre), '--bandwidth', bandwidth,
                        '--ptt-device', self.ptt_port, '--line-ptt',
                        '--run-dir', str(self.destination or evidence.session)]
                    if mail:
                        command += ['--mycall', os.environ['MYCALL'], '--call', gateway, '--transmit', *extra]
                        rc = self.worker(verb, command=command)
                        label = 'besra run_server'
                    else:
                        rc = self.ardop_call(command, gateway, bandwidth)
                        label = 'ardop_call.py'
                if rc == 2:
                    raise Failure(f'{label} refused its arguments (exit 2) -- NOTHING WAS TRANSMITTED. That is a\n'
                                  '   launcher bug, not a band result; do not record it as an attempt.')
                cat_ready = True
                if vara and self.cat_port and self.cat_backend == 'serial':
                    try:
                        self.close_daemon()
                        self.stop_daemons()
                    except (Failure, hardware.ClaimError, OSError, subprocess.SubprocessError):
                        print('  VARA CAT cleanup failed -- skipping the contended power query', flush=True)
                        cat_ready, rc = False, rc or 1
                if cat_ready:
                    self.power(dial)
                self.prove_unkeyed()
                return rc

    def ardop_call(self, command, gateway, bandwidth):
        result = []
        def call(modem, pump):
            for _ in range(40):
                pump()
                try:
                    with socket.create_connection(('127.0.0.1', 8515), timeout=.25):
                        break
                except OSError:
                    time.sleep(.25)
            else:
                raise Failure('besra did not serve on 8515 within 10 s')
            rc = self.worker('ardop client', internal=True, command=[sys.executable, '-u',
                str(ROOT / 'tools/ardop_call.py'), gateway, '90', '--bandwidth', bandwidth,
                '--launcher'], service=pump)
            result.append(rc)
            with unwind_bound(CHILD_STOP_S):
                if self.claim.end_child(modem) is False:
                    raise Failure('besra did not stop')
        self.worker('ardop', command=command, ready=call)
        time.sleep(2)
        return result[0]

    def rpr_mail(self, gateway, centre, evidence, extra):
        import argparse
        import rpr_mail
        mail = rpr_arguments(extra)
        parser = argparse.ArgumentParser()
        rpr_mail.mail_flags(parser)
        args = parser.parse_args(mail)
        path = rpr_mail.prepare(os.environ.get('HFMODEM_STATION', str(ROOT / 'station.toml')),
            centre, evidence.session, send=args.mail_send, to=args.mail_to, subject=args.mail_subject)
        print(f'  captures: {evidence.session} (station.toml, capture-*/stream, tx_events.jsonl, mail/)', flush=True)
        bound = rpr_mail.stop_bound(path, cwid_optional=args.cwid_optional)
        if os.name == 'nt':
            from hfmodem.core import winproc
            winproc.declare_grace(UNWIND_GRACE_S + bound)
        control = MailControl(bound)
        with control.signals():
            rc = self.worker('rpr-mail', command=[sys.executable, '-u', str(ROOT / 'tools/rpr_mail.py'),
                'run', str(path), gateway, '--launcher-control', *mail], control=control)
        streams = sorted(evidence.session.glob('capture-*/stream.json'))
        if streams:
            evidence.witness_stream = streams[-1].with_suffix('.wav')
        return rc

    def flight(self, words, evidence=None, witness=False):
        self.started = time.monotonic()
        verb = words[0]
        if verb == 'preflight':
            self.require_hardware(transmit=True)
            with self.possession(verb):
                self.codecs()
                if self.cat_backend == 'rigctld':
                    self.start_daemon()
                else:
                    # Native preflight owns CAT itself, exclusively on Windows.
                    self.stop_daemons()
                say('preflight (keys unmodulated; no RF)')
                rc = self.worker('preflight', command=[sys.executable, '-m', 'hfmodem', 'rig',
                    os.environ.get('HFMODEM_STATION', str(ROOT / 'station.toml')), '--preflight'])
                proof = self.prove_unkeyed()
                return rc or proof
        answer = verb == 'pactor-answer'
        if len(words) < 2:
            raise Failure('centre frequency' if answer else 'gateway callsign')
        centre = positional('centre', words[1] if answer else words[2] if len(words) > 2 else '7101500')
        seconds = words[2] if len(words) > 2 else '900'
        if answer:
            positional('seconds', seconds)
        dial, extra = centre - 1500, words[3:]
        self.require_hardware(transmit=True)
        with self.possession(verb):
            self.codecs()
            if answer:
                say(f'PACTOR-1 answer as {os.environ["MYCALL"]}, centre {centre} (dial {dial}), {seconds}s')
            else:
                say(f'PACTOR-1 {os.environ["MYCALL"]} -> {words[1]}, centre {centre} (dial {dial})')
            try:
                self.stop_daemons()
            except (hardware.ClaimError, OSError, subprocess.SubprocessError) as exc:
                print(f'NOT CONFIRMED: {exc}', file=sys.stderr, flush=True)
            self.power(dial)
            if not answer:
                os.environ['DXCALL'] = words[1]
                os.environ.pop('HOST_FREE_IDENT', None)
                os.environ.pop('HOST_FREE_AT', None)
                bandwidth = '1' if any(a in {'--pactor1-only', '--pactor2-only', '--p2-mail'} for a in extra) else ''
                extra = self.gate_channel(dial, bandwidth, extra)
            self.cat_free()
            command = [sys.executable, '-m', 'hfmodem.shrike.onair', '--transmit',
                '--serial', self.cat_port, '--ptt-port', self.ptt_port,
                '--audio-out', self.audio_out, '--audio-in', self.audio_in,
                '--mycall', os.environ['MYCALL']]
            if not answer:
                command += ['--dxcall', words[1]]
            command += ['--dial', str(dial)]
            if answer:
                command += ['--answer', seconds]
            else:
                hold = '0' if any(a in {'--mail-fetch', '--mail-send'} or a.startswith('--mail-send=') for a in extra) else '12'
                command += ['--retries', '20', '--max-cycles', '48', '--hold', hold]
            command += ['--outdir', str(evidence.session), *evidence.modem_args, *extra]
            self.pending('witness')
            with evidence.witness(self, dial, words[1] if not answer else None, enabled=witness):
                try:
                    rc = self.worker('pactor', command=command)
                finally:
                    print(f'  captures: {evidence.session}', flush=True)
                if rc == 2:
                    raise Failure('shrike onair refused its arguments (exit 2) -- NOTHING WAS TRANSMITTED. That is a\n'
                                  '   launcher bug, not a band result; do not record it as an attempt.')
                self.power(dial)
                self.prove_unkeyed()
                return rc

    def sabir(self, words, witness=False):
        from hfmodem.core import config
        from .launch_capture import LegacyEvidence
        if len(words) < 2 or not words[1]:
            raise Failure('authorized clear channel centre Hz required')
        centre = int(words[1])
        dial = centre - 1500
        self.started = time.monotonic()
        self.require_hardware(transmit=True)
        if os.name == 'nt' and not self.ptt_port:
            raise Failure('sabir requires a dedicated PTT port; CAT cannot key this flight')
        with self.possession('sabir'):
            self.codecs()
            say(f'sabir one-way transmission on centre {centre} (dial {dial}) — no reply expected')
            self.start_daemon()
            self.power(dial)
            extra = self.gate_channel(dial, '', words[2:], modem='sabir')
            _, force = key_arguments(words[2:])
            grid = os.environ.get('MYGRID') or config.grid(path=os.environ.get('HFMODEM_STATION'))
            grid_args = ['--grid', grid] if grid else []
            effective_grid, beacon, pending = grid, 'presence', None
            for word in extra:
                if pending:
                    if pending == '--grid':
                        effective_grid = word
                    else:
                        beacon = word
                    pending = None
                elif word in {'--grid', '--beacon'}:
                    pending = word
                elif word.startswith('--grid='):
                    effective_grid = word.split('=', 1)[1]
                elif word.startswith('--beacon='):
                    beacon = word.split('=', 1)[1]
            fallback = []
            if not effective_grid and beacon == 'wspr':
                print('warning: no grid, so the wspr beacon is sent as the presence beacon', file=sys.stderr, flush=True)
                fallback = ['--beacon', 'presence']
            evidence = LegacyEvidence()
            self.pending('witness')
            with evidence.witness(self, dial, None, enabled=witness):
                evidence.session.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-m', 'hfmodem.sabir.onair', '--transmit',
                    '--radio', self.name, '--rigctld', '127.0.0.1:4532',
                    '--ptt-device', self.ptt_port, '--line-ptt', '--channel', str(centre),
                    '--audio-in', self.audio_in, '--audio-out', self.audio_out,
                    '--mycall', os.environ['MYCALL'], *grid_args,
                    '--run-dir', str(self.destination or evidence.session),
                    *(['--force'] if force else []), *extra, *fallback]
                try:
                    rc = self.worker('sabir', command=command)
                finally:
                    print(f'  captures: {evidence.session}', flush=True)
                self.power(dial)
                self.prove_unkeyed()
                return rc

    def terminal_flight(self, words):
        verb = words[0]
        if len(words) < 2 or not words[1]:
            raise Failure('the calling frequency in Hz' if verb == 'vara-chat'
                          else 'a dial in Hz, or - to leave the rig where it is')
        self.require_hardware(transmit=True)
        if os.name == 'nt' and not self.ptt_port:
            raise Failure(f'{verb} requires a dedicated PTT port; CAT cannot key this flight')
        with self.possession(verb):
            self.codecs()
            if verb == 'rtty':
                say(f'RTTY terminal as {os.environ["MYCALL"]} -- nothing transmits until you key')
            self.start_daemon()
            warning = []
            if verb == 'rtty':
                mode = self.profile.get('mode', 'PKTUSB')
                try:
                    self.tcp(f'+\\set_mode {mode} {mode_passband(self.profile, 3000)}\nq\n')
                except OSError:
                    pass
                got = self.daemon_field('get_mode', 'Mode')
                if got != mode:
                    print(f'\n!! the rig reads mode {got or "unread"}, not {mode}: '
                          f'mark RF and the log assume {mode}', file=sys.stderr, flush=True)
                    warning = ['--warn', f'rig mode {got or "unread"}, not {mode}']
            dial = words[1]
            if dial == '-' and verb != 'vara-chat':
                dial = self.daemon_field('get_freq', 'Frequency')
                if verb == 'rtty':
                    if not dial.isdigit():
                        raise Failure('DIAL UNREAD -- the rig reported no frequency. Name the dial.')
                    print(f'  dial {dial}, as the rig sits', flush=True)
            dial = positional('dial', dial)
            if verb != 'rtty' or words[1] != '-':
                self.tune(dial)
            self.power(dial)
            handles = {}
            if verb in {'rtty', 'vara-chat'}:
                owner = (hardware.extra('owner', optional=True) if os.name == 'nt' else
                         int(os.environ['HFMODEM_OWNER_FD']) if os.environ.get('HFMODEM_OWNER_FD') else None)
                if owner is not None:
                    handles['owner'] = owner
            # The console names these; a verb typed at a prompt gets the console's own defaults.
            given = {word.split('=', 1)[0] for word in words[2:]}
            if verb == 'rtty':
                adif = [] if '--adif' in given else ['--adif', str(data_root() / 'working' / 'qso-log.adi')]
                command = [sys.executable, '-m', 'hfmodem.lanner.qso', '--dial', str(dial),
                    '--rig', self.name, '--ptt-device', self.ptt_port,
                    '--audio-in', self.audio_in, '--audio-out', self.audio_out,
                    '--mycall', os.environ['MYCALL'], '--launcher', str(os.getpid()), *adif, *warning, *words[2:]]
                label = 'lanner qso'
            else:
                expected = {'ft891': 'FT-891', 'ftdx10': 'FTDX-10', 'g90': 'G90', 'x6100': 'X6100'}[self.name]
                script = 'kestrel_host.py' if verb == 'vara-chat' else 'vara_txcal.py'
                defaults, destination = [], []
                if verb == 'vara-txcal':
                    from .launch_capture import LegacyEvidence
                    # A --session among the console's own arguments comes later and wins.
                    directory = LegacyEvidence().session
                    directory.mkdir(parents=True, exist_ok=True)
                    defaults = ['--session', str(directory)]
                elif self.destination:
                    destination = ['--session', str(self.destination)]
                if verb == 'vara-chat':
                    if not destination and '--session' not in given:
                        directory = mode_root('vara-chat') / 'sessions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                        directory.mkdir(parents=True)
                        defaults = ['--session', str(directory)]
                    band = next((m for lo, hi, m in AMATEUR_HF_BANDS if lo * 1e6 <= dial <= hi * 1e6), None)
                    if '--band' not in given and band is not None:
                        defaults += ['--band', str(band), '--dial', str(dial)]
                command = [sys.executable, str(ROOT / 'tools' / script),
                    *(['--chat'] if verb == 'vara-chat' else []),
                    '--mycall', os.environ['MYCALL'], '--expect-model', expected,
                    '--ptt-device', self.ptt_port, '--tx-device', self.audio_out,
                    '--rx-device', self.audio_in, *defaults, *words[2:], *destination]
                label = 'kestrel_host.py --chat' if verb == 'vara-chat' else script
            self.after_down = dial
            rc = self.worker(verb, command=command, handles=handles)
            self.after_down = None
            if rc == 2:
                raise Failure(f'{label} refused its arguments (exit 2) -- NOTHING WAS TRANSMITTED. That is a\n'
                              '   launcher bug, not a band result; do not record it as an attempt.')
            self.power(dial)
            self.prove_unkeyed()
            return rc

    def daemon_field(self, command, field):
        try:
            reply = self.tcp(f'+\\{command}\nq\n', 5)
        except OSError:
            return ''
        match = re.search(rf'^{field}: (.*)$', reply, re.M)
        return match[1] if match else ''

    def pounce(self, words, evidence=None, witness=False):
        if len(words) < 4:
            raise Failure(('vara-mail|ardop-mail|pactor', 'gateway callsign', 'channel centre Hz')[max(0, len(words) - 1)])
        centre = int(words[3])
        self.attempted = False
        with self.possession('pounce', parent=True):
            say(f'waiting for {centre // 1000}.{centre % 1000 // 100} kHz to clear, then {words[1]} -> {words[2]}')
            parent_args, handles, values = [], {}, {'POUNCE_WAIT': str(int(os.environ.get('POUNCE_WAIT') or '900'))}
            if evidence is not None:
                parent_args = ['--capture-parent-id', evidence.capture.capture_id,
                               '--capture-parent-kind', 'launch-group']
                handles['capture-group'] = evidence.children.fileno()
                values.update(HFMODEM_CAPTURE_GROUP_FD=str(evidence.children.fileno()),
                              HFMODEM_CAPTURE_GROUP_ROOT=str(evidence.session),
                              HFMODEM_CAPTURE_GROUP_ID=evidence.capture.capture_id)
            command = [sys.executable, str(ROOT / 'tools/onair.py'),
                       *(['--witness'] if witness else []), *words[1:]]
            if words[1] in MAIL_FLIGHTS and self.cat_backend == 'rigctld':
                self.require_hardware(transmit=True)
                self.start_daemon()
                values['ONAIR_CAT_DELEGATED'] = '1'
            acknowledge = pounce_acknowledge(values['POUNCE_WAIT'])
            with environment(values):
                for attempt in range(1, 6):
                    destination = []
                    if self.destination and evidence is not None:
                        destination = ['--run-dir', str(self.destination / f'{evidence.capture.capture_id}-attempt-{attempt}')]
                    elif self.destination:
                        destination = ['--run-dir', str(self.destination)]
                    try:
                        rc = self.worker('pounce', command=[*command, *destination, *parent_args],
                                         handles=handles, acknowledge=acknowledge)
                    except Failure as exc:
                        rc = exc.code
                        if str(exc):
                            print(str(exc), flush=True)
                    if rc == 3:
                        raise Failure('channel never cleared')
                    if rc != 4:
                        return rc
                    print('  the channel was taken again before we keyed — back to waiting', flush=True)
            raise Failure('gave up after five attempts')

    def prove_unkeyed(self):
        try:
            return self.worker('proof', internal=True)
        except Failure as exc:
            print(str(exc), flush=True)
            return exc.code

    @phase('launcher run')
    def run(self, words):
        verb = words[0]
        if verb == 'check':
            say('hardware')
        if verb in {'check', 'listen'}:
            self.require_hardware()
        with self.possession(verb):
            if verb == 'power':
                self.require_cat()
                self.require_hamlib()
                hardware.inherited(self.descriptor)
                self.power(int(words[1] if len(words) > 1 else 7101500) - 1500)
                self.admit()
            else:
                self.codecs()
                if verb == 'check':
                    say("the host's sound cards — '>' and '<' mark the system defaults, not ours")
                    from hfmodem.cli import main
                    output = io.StringIO()
                    with redirect_stdout(output):
                        code = main(['devices'])
                    for line in output.getvalue().splitlines():
                        print('  ' + line, flush=True)
                    if code:
                        return code
                    say('receiver')
                    with phase('receiver proof'):
                        rc = self.worker('proof', internal=True)
                    if rc:
                        return rc
                    self.admit()
                    say('ready — nothing above has transmitted')
                else:
                    if len(words) < 2:
                        raise Failure('centre frequency in Hz, e.g. 7101500')
                    centre, seconds = int(words[1]), float(words[2] if len(words) > 2 else 300)
                    shown = words[2] if len(words) > 2 else '300'
                    cat = not self.cfg or self.cfg.rig.cat != 'none'
                    if cat:
                        self.start_daemon()
                    say(f'listening on centre {centre} (dial {centre - 1500}) for {shown}s — NO TX')
                    if cat:
                        self.tcp(f'+\\set_mode PKTUSB {mode_passband(self.profile, 3000)}\nq\n', 5)
                        self.tune(centre - 1500)
                    else:
                        print('  CAT none; dial unverified', flush=True)
                    with phase('receive monitor'):
                        self.worker('monitor', seconds=seconds)
                    with phase('receiver proof'):
                        return self.worker('proof', internal=True)
        return 0


def worker(verb, device):
    if verb == 'monitor':
        from creance.cli import main
        with phase('monitor CLI/source'):
            return main(['monitor', '--device', device])
    import numpy as np
    import sounddevice as sd
    from hfmodem.core.devices import find_device
    from hfmodem.core.levels import NO_AUDIO_CAUSE, RX_SILENT_RMS
    admission = hardware.received(launch=True)
    try:
        try:
            index = find_device(device, 'in', required=True)
        except SystemExit as exc:
            print(f"  {exc}\n  no '{device}' input — CANNOT PROVE PTT IS DOWN", flush=True)
            return 1
        with phase('receiver audio open/read'), sd.InputStream(samplerate=48000, channels=1, device=index) as stream:
            admission.opened()
            audio, overflow = stream.read(int(.7 * 48000))
        rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
        if rms > .01:
            print(f'  receiver rms {rms:.5f} -> LIVE, not transmitting', flush=True)
        elif rms > RX_SILENT_RMS:
            print(f'  receiver rms {rms:.5f} -> QUIET: the codec is delivering audio and '
                  'the band is not. Check the radio\'s AF gain, its VFO and its mode '
                  'before suspecting the keying line. This reading is of the room; the '
                  "one that measures the line itself is the PTT verdict 'down' prints.", flush=True)
        else:
            print(f'  receiver rms {rms:.5f} -> NO AUDIO: {NO_AUDIO_CAUSE}', flush=True)
        return 0 if rms > .01 else 1
    finally:
        if admission.active:
            admission.failed()
            print('CLAIM NOT ADMITTED capture open failed', flush=True)


def main(argv=None):
    launch_io.initialize()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (['--help'], ['-h']):
        header = (ROOT / 'tools' / 'onair.sh').read_text(encoding='utf-8').splitlines()[1:]
        for line in header:
            if not line.startswith('#'):
                break
            print(line)
        return 0
    if argv[:1] == ['down']:
        return run_main(argv)
    if argv[:1] == ['--worker']:
        return worker(*argv[1:])
    if argv[:1] == ['--readout']:
        launcher = Launcher()
        launcher.descriptor = hardware.received()
        try:
            launcher.power(int(argv[1]))
            return worker('proof', argv[2])
        finally:
            hardware.close(launcher.descriptor)
    try:
        argv, selection, destination = recording_selection(argv)
        parsed, _ = arguments(argv[1:] if argv[:1] == ['--capture-prepared'] else argv)
        parsed = [word for word in parsed if word != '--witness']
        recording = recording_line(parsed[0] if parsed else '', selection)
    except (Failure, OSError) as exc:
        print(f'\n!! {exc}', file=sys.stderr, flush=True)
        return exc.code if isinstance(exc, Failure) else 1
    # Said even when the run stops before run_main would print it.
    pending = [recording] if recording else []
    try:
        return dispatch(argv, parsed, pending, destination)
    finally:
        if pending:
            print(pending.pop(), flush=True)


def dispatch(argv, parsed, recording, destination):
    if parsed and parsed[0] in MAIL_FLIGHTS | {'sabir', 'vara-chat', 'vara-txcal', 'rtty'}:
        from hfmodem.core import config, identification
        try:
            mycall = config.mycall(path=os.environ.get('HFMODEM_STATION'))
            if not mycall and parsed[0] != 'rtty':
                raise Failure('set MYCALL or [station] mycall before calling')
            if parsed[0] not in {'sabir', 'vara-txcal', 'rtty'}:
                effective = identification.options(parsed[1:], mycall=mycall)
                identity = identification.resolve(effective.mycall, effective.idcall)
                os.environ.update(MYCALL=identity.digital_call, IDCALL=identity.cw_call)
            else:
                os.environ['MYCALL'] = mycall
        except ValueError as exc:
            print(f'identification: {exc}', file=sys.stderr, flush=True)
            return 2
        except Failure as exc:
            print(f'\n!! {exc}', file=sys.stderr, flush=True)
            return exc.code
    if parsed[:1] in (['pactor'], ['pactor-answer']) or parsed[:2] == ['pounce', 'pactor']:
        from hfmodem.core import config, identification
        from .launch_capture import Evidence
        try:
            if parsed[0] != 'pounce':
                mycall = config.mycall(path=os.environ.get('HFMODEM_STATION'))
                if not mycall:
                    raise Failure('set MYCALL or [station] mycall before calling')
                effective = identification.options(parsed[1:], mycall=mycall)
                identity = identification.resolve(effective.mycall, effective.idcall)
                os.environ.update(MYCALL=identity.digital_call, IDCALL=identity.cw_call)
        except ValueError as exc:
            print(f'identification: {exc}', file=sys.stderr, flush=True)
            return 2
        except Failure as exc:
            print(f'\n!! {exc}', file=sys.stderr, flush=True)
            return exc.code
        evidence, code = None, 1
        try:
            if destination and parsed[0] != 'pounce':
                # Bash appends the destination's --outdir before the capture is validated,
                # so it wins over an operator --outdir.
                argv = [*argv, '--outdir', destination]
            evidence = Evidence([*argv, '--run-dir', destination] if destination else argv)
            with evidence.transcript():
                code = run_main(recording_selection(evidence.words)[0], evidence=evidence,
                                recording=recording, destination=destination)
            return code
        except (OSError, ValueError) as exc:
            print(f'\n!! {exc}', file=sys.stderr, flush=True)
            return 1
        finally:
            if evidence is not None:
                evidence.finish(code)
    return run_main(argv, recording=recording, destination=destination)


def run_main(argv, *, evidence=None, recording=None, destination=None):
    # Bash dispatches cleanup before stripping globals. Trailing arguments are
    # ignored, including log/password flags; a prefixed flag misses this gate.
    try:
        words, log = (list(argv), None) if argv[:1] == ['down'] else arguments(argv)
    except (Failure, OSError) as exc:
        print(f'\n!! {exc}', file=sys.stderr, flush=True)
        return 1
    with transcript(log):
        try:
            if argv[:1] == ['down']:
                return down.run(None, client)
            witness = '--witness' in words
            words = [word for word in words if word != '--witness']
            if not words or words[0] not in AVAILABLE:
                raise Failure(f"{words[0] if words else 'launcher'} is not yet ported to the Python launcher", 2)
            if recording:
                print(recording.pop(), flush=True)
            tree_line()
            if witness and words[0] not in {'pactor', 'sabir', 'pounce', *MAIL_FLIGHTS}:
                raise Failure('--witness records a third-party receiver alongside a transmission, '
                              f"and '{words[0]}' does not key")
            wait = os.environ.get('POUNCE_WAIT', '0')
            if wait and not re.fullmatch(r'[0-9]+', wait):
                raise Failure(f"POUNCE_WAIT is whole seconds, and '{wait}' is not")
            if words[0] == 'down':
                # There is no down case in bash's later verb dispatch: it
                # prints usage successfully, without acquiring the hardware.
                print('Usage: tools/onair.py VERB [ARGS] [--log NAME]\n'
                      '  down                       # stop everything, prove PTT is low')
                return 0
            launcher = Launcher()
            launcher.destination = Path(destination) if destination else None
            if words[0] == 'sabir':
                return launcher.sabir(words, witness)
            if words[0] == 'pounce':
                return launcher.pounce(words, evidence, witness)
            if words[0] in {'rtty', 'vara-chat', 'vara-txcal'}:
                return launcher.terminal_flight(words)
            if words[0] in MAIL_FLIGHTS:
                return launcher.mail_flight(words, witness)
            if words[0] in {'pactor', 'pactor-answer', 'preflight'}:
                return launcher.flight(words, evidence, witness)
            return launcher.run(words)
        except Interrupted as exc:
            return exc.code
        except KeyboardInterrupt:
            return 130
        except (Failure, hardware.ClaimError, OSError, ValueError, subprocess.SubprocessError) as exc:
            if str(exc):
                print(f'\n!! {exc}', file=sys.stderr, flush=True)
            return exc.code if isinstance(exc, Failure) else 1


if __name__ == '__main__':
    raise SystemExit(main())
