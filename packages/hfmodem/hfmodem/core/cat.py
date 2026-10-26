# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import math
import os
import re
import select
import socket
import termios
import threading
import time
from typing import ContextManager, Protocol

from .operations import OperationGate, leased
from .rig import RigError
from .rigs import RIGS
from .rxreadiness import RadioMode

EXCHANGE_S = 1.0
WRITE_S = 0.25
EMERGENCY_S = 0.4
SAMPLE_RATE = 48000


class CatDriver(Protocol):
    operations: OperationGate
    keyed_probe: Callable[[], bool]
    clock: Callable[[], float]
    reads_ptt: bool
    cache_free: bool
    model: int

    def open(self) -> None: ...
    def close(self) -> None: ...
    def hold(self, deadline_index=None) -> ContextManager[bool]: ...
    def protect(self, start_index: int, end_index: int | None) -> object: ...
    def unprotect(self, token: object) -> None: ...
    def identify(self) -> str: ...
    def sole_ptt_owner(self, notes) -> bool: ...
    def freq(self) -> int: ...
    def set_freq(self, hz: int) -> None: ...
    def mode(self) -> RadioMode: ...
    def set_mode(self, mode: str, width_hz: int) -> bool: ...
    def receiver_filters(self) -> tuple[RadioMode, int | None]: ...
    def ptt_state(self, *, _arming=False) -> bool | None: ...
    def set_power(self, watts: float) -> None: ...
    def read(self, command: str, prefix: str) -> str: ...
    def write(self, command: str) -> None: ...
    def emergency_unkey(self) -> bool | None: ...  # PTT state: False confirms down.


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RigError('CAT deadline expired')
    return remaining


def _row(profile):
    return dict(RIGS[profile] if isinstance(profile, str) else profile)


class _Driver:
    def __init__(self, profile):
        self.profile = _row(profile)
        self.model = self.profile['model']
        self.operations = OperationGate()
        self.keyed_probe = lambda: False
        self.clock = lambda: time.monotonic() * SAMPLE_RATE
        self.reads_ptt = self.cache_free = True
        self.poisoned = False
        self._condition = threading.Condition(threading.RLock())
        self._active_deadline = None
        self._holder = None
        self._depth = 0
        self._intervals = {}

    @property
    def source(self):
        return 'poisoned' if self.poisoned else 'live'

    def protect(self, start_index, end_index):
        if end_index is not None and end_index < start_index:
            raise ValueError('protected interval ends before it starts')
        token = object()
        with self._condition:
            self._intervals[token] = (start_index, end_index)
        return token

    def unprotect(self, token):
        with self._condition:
            self._intervals.pop(token, None)

    @contextmanager
    def hold(self, deadline_index=None):
        owner = threading.get_ident()
        admitted = False
        with self.operations.lease():
            with self._condition:
                while True:
                    now = self.clock()
                    if deadline_index is not None and now >= deadline_index:
                        break
                    if self._holder == owner:
                        self._depth += 1
                        admitted = True
                        break
                    if self._holder is None and self._active_deadline is None:
                        self._holder, self._depth = owner, 1
                        admitted = True
                        break
                    if deadline_index is not None:
                        budget = (deadline_index - now) / SAMPLE_RATE
                        if (self._holder is not None or
                                self._active_deadline - time.monotonic() > budget):
                            break
                        self._condition.wait(budget)
                    else:
                        self._condition.wait()
            try:
                yield admitted
            finally:
                if admitted:
                    with self._condition:
                        self._depth -= 1
                        if not self._depth:
                            self._holder = None
                            self._condition.notify_all()

    def _check(self, deadline, arming):
        with self._condition:
            if self.poisoned:
                raise RigError('CAT transport poisoned')
            remaining = _remaining(deadline)
            if arming:
                return
            now = self.clock()
            self._intervals = {token: (start, end)
                               for token, (start, end) in self._intervals.items()
                               if end is None or now < end}
            if self.keyed_probe() or any(
                    now + remaining * SAMPLE_RATE > start
                    for start, end in self._intervals.values()):
                raise RigError('protected interval')

    @contextmanager
    def _executing(self, deadline, arming=False, owner=None):
        owner = threading.get_ident() if owner is None else owner
        with self._condition:
            while self._active_deadline is not None or self._holder not in (None, owner):
                self._check(deadline, arming)
                self._condition.wait(min(_remaining(deadline), 0.01))
            self._check(deadline, arming)
            self._active_deadline = deadline
        try:
            yield
        finally:
            with self._condition:
                self._active_deadline = None
                self._condition.notify_all()

    def _power(self, watts):
        maximum = self.profile.get('max_watts')
        if maximum is None:
            raise RigError('power scaling unknown for this rig')
        if (not isinstance(watts, (float, int)) or isinstance(watts, bool)
                or not math.isfinite(watts)
                or not self.profile.get('min_watts', 0) <= watts <= maximum):
            raise RigError('power outside rig range')
        return maximum


@dataclass
class _Request:
    command: str
    prefix: str | None
    deadline: float
    read_after: str | None = None
    owner: int = field(default_factory=threading.get_ident)
    arming: bool = False
    sent: bool = False
    setter_refused: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: str | None = None
    error: Exception | None = None
    _finish_lock: threading.Lock = field(default_factory=threading.Lock)

    def finish(self, result=None, error=None):
        with self._finish_lock:
            if not self.done.is_set():
                self.result, self.error = result, error
                self.done.set()


class SerialTransport(_Driver):
    def __init__(self, port, profile='ft891', *, serial_factory=None):
        super().__init__(profile)
        self.port = port
        self._serial_factory = serial_factory
        self._serial = None
        self._thread = None
        self._queue = deque()
        self._pending = None
        self._emergency = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._open_error = None
        self._runtime_error = None
        self._owed = deque()
        self._buf = b''
        self.uncertain = self.wedged = False

    @property
    def source(self):
        return 'poisoned' if self.poisoned else 'uncertain' if self.uncertain else 'live'

    @leased
    def open(self):
        with self._condition:
            if self._thread is not None:
                if self._thread.is_alive() and self._ready.is_set() and not self._open_error:
                    return
                raise RigError('CAT transport must be closed before reopening')
            if self.profile.get('ptt_port') and os.path.realpath(self.port) == os.path.realpath(self.profile['ptt_port']):
                raise RigError('CAT must not open the PTT port')
            for key in ('cat_rts', 'cat_dtr'):
                if not isinstance(self.profile.get(key), bool):
                    raise RigError(f'{key} must be a profile boolean')
            self._thread = threading.Thread(target=self._run, name='cat-serial', daemon=True)
            self._thread.start()
        if not self._ready.wait(EXCHANGE_S):
            self._stop.set()
            self.poisoned = True
            raise RigError('CAT open deadline expired')
        if self._open_error:
            raise RigError(f'CAT open failed: {self._open_error}')

    def close(self):
        self._stop.set()
        self._poison('CAT closed')
        thread = self._thread
        if thread is not None:
            thread.join(EXCHANGE_S)
            if thread.is_alive():
                raise RigError('CAT transport did not close')
        with self._condition:
            self._thread = None
            self._stop.clear()
            self._ready.clear()
            self._open_error = None
            self._runtime_error = None
            self._buf = b''
            self._owed.clear()
            self._emergency = None
            self._intervals.clear()
            self.poisoned = self.uncertain = self.wedged = False

    def _poison(self, why):
        with self._condition:
            self.poisoned = True
            error = RigError(why)
            if self._pending:
                self._pending.finish(error=error)
            while self._queue:
                self._queue.popleft().finish(error=error)
            self._condition.notify_all()

    def exchange(self, command, prefix=None, *, timeout=EXCHANGE_S, _arming=False):
        prefixes = {'AI', 'ID', 'FA', 'FB', 'MD0', 'MD1', 'SH0', 'NA0',
                    'RF0', 'PC', 'TX', 'VS', 'IS0', 'RT', 'XT', 'FT',
                    'GT0', 'PA0', 'RA0', 'RG0', 'NR0', 'BC0', 'NB0', 'CO00'}
        prefixes.update('EX' + item for item in self.profile['readback']['menu_items'])
        if prefix not in prefixes or command != prefix + ';':
            raise RigError('expected a known CAT query; use YaesuCat.write for setters')
        return self._exchange(command, prefix, timeout=timeout, _arming=_arming)

    @leased
    def _exchange(self, command, prefix=None, *, timeout=EXCHANGE_S, _arming=False,
                  _read_after=None):
        if (not command.isascii() or not command.endswith(';') or ';' in command[:-1]
                or any(c in command for c in '\r\n\0')):
            raise RigError('expected one semicolon-terminated CAT command')
        if prefix is not None and not re.fullmatch(r'[A-Z][A-Z0-9]*', prefix):
            raise RigError('invalid CAT reply prefix')
        if _read_after is not None and (prefix is None or _read_after != prefix + ';'):
            raise RigError('invalid CAT setter readback')
        request = _Request(command, prefix, time.monotonic() + min(timeout, EXCHANGE_S),
                           read_after=_read_after, arming=_arming)
        with self._condition:
            self._check(request.deadline, _arming)
            if self._serial is None or self._stop.is_set():
                raise RigError('CAT is not open')
            self._queue.append(request)
            self._condition.notify_all()
        request.done.wait(max(0, request.deadline - time.monotonic()))
        with self._condition:
            if not request.done.is_set():
                if request.sent and request.prefix:
                    self.uncertain = True
                raise RigError('refusal after setter; CAT exchange timed out'
                               if request.setter_refused else 'CAT exchange timed out')
        if request.error:
            raise request.error
        return request.result

    def emergency_unkey(self):
        request = _Request('TX0;TX;', 'TX', time.monotonic() + EMERGENCY_S)
        with self._condition:
            self._poison('CAT emergency unkey')
            if self._serial is None or self.wedged or self._stop.is_set() or self._emergency:
                return None
            self._emergency = request
            self._condition.notify_all()
        request.done.wait(max(0, request.deadline - time.monotonic()))
        return {'TX0;': False, 'TX1;': True, 'TX2;': True}.get(request.result)

    def _write(self, command, deadline):
        raw = command.encode('ascii')
        deadline = min(deadline, time.monotonic() + WRITE_S)
        try:
            while raw:
                _remaining(deadline)
                _, writable, _ = select.select([], [self._serial.fileno()], [], min(0.01, _remaining(deadline)))
                if not writable:
                    continue
                try:
                    count = os.write(self._serial.fileno(), raw)
                except BlockingIOError:
                    continue
                if count <= 0:
                    raise RigError('CAT write made no progress')
                raw = raw[count:]
        except (OSError, RigError) as exc:
            self.wedged = True
            self._poison('CAT write wedged')
            raise RigError(f'CAT write failed: {exc}') from exc

    def _packets(self, wait):
        readable, _, _ = select.select([self._serial.fileno()], [], [], max(0, wait))
        if not readable:
            return []
        chunk = os.read(self._serial.fileno(), 4096)
        if not chunk:
            self._poison('CAT serial disconnected')
            raise RigError('CAT serial disconnected')
        self._buf += chunk
        if len(self._buf) > 65536:
            self._poison('CAT reply exceeds limit')
            raise RigError('CAT reply exceeds limit')
        parts = self._buf.split(b';')
        self._buf = parts.pop()
        try:
            return [(part + b';').decode('ascii') for part in parts]
        except UnicodeError as exc:
            self._poison('non-ASCII CAT reply')
            raise RigError('non-ASCII CAT reply') from exc

    def _consume(self, packets, expected=None, sentinel=False):
        result = None
        for reply in packets:
            if sentinel and reply.startswith(expected):
                pattern = {'ID': r'ID\d{4};', 'VS': r'VS[01];', 'FA': r'FA\d{9};'}[expected]
                if not re.fullmatch(pattern, reply):
                    self._poison('invalid resynchronization sentinel')
                    raise RigError('invalid resynchronization sentinel')
                self._owed.clear()
                self.uncertain = False
                result = reply
                sentinel = False
            elif not self._owed:
                self._poison('unsolicited or ambiguous CAT reply')
                raise RigError('unsolicited or ambiguous CAT reply')
            elif reply == '?;' or reply.startswith(self._owed[0]):
                prefix = self._owed.popleft()
                if not self._owed and prefix == expected:
                    result = reply
            else:
                self._poison('CAT parser desync')
                raise RigError(f'CAT parser desync: expected {self._owed[0]}, got {reply}')
        if not self._owed and self._buf:
            self._poison('unsolicited partial CAT reply')
            raise RigError('unsolicited partial CAT reply')
        return result

    def _reply(self, expected, deadline, *, sentinel=False, emergency=False,
               setter=None):
        after_setter = setter is not None
        refusal = False
        while True:
            if self._stop.is_set() or (self.poisoned and not emergency):
                raise RigError('CAT transport poisoned')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.uncertain = True
                raise RigError('refusal after setter; CAT reply timed out' if refusal
                               else 'CAT reply timed out')
            packets = self._packets(min(0.01, remaining))
            with self._condition:
                if self._stop.is_set() or (self.poisoned and not emergency):
                    raise RigError('CAT transport poisoned')
                if after_setter and packets:
                    # A successful setter is silent. Its optional refusal precedes
                    # the matching query's reply on this FIFO link. Drain that
                    # readback before returning the error to the setter's caller.
                    after_setter = False
                    if packets[0] == '?;':
                        # A readback '?;' is indistinguishable: report setter
                        # refusal and fail safe to uncertain when readback expires.
                        refusal = True
                        setter.setter_refused = True
                        packets = packets[1:]
                result = self._consume(packets, expected, sentinel)
            if result is not None:
                if time.monotonic() > deadline:
                    self.uncertain = True
                    raise RigError('CAT reply timed out')
                if refusal:
                    raise RigError('refusal after setter')
                if result == '?;':
                    raise RigError('radio refused CAT query')
                return result

    def _resync(self, deadline):
        prefix = next((p for p in ('ID', 'VS', 'FA') if not any(
            owed.startswith(p) or p.startswith(owed) for owed in self._owed)), None)
        if prefix is None:
            raise RigError('CAT uncertain: no distinct resynchronization sentinel')
        self._write(prefix + ';', deadline)
        self._owed.append(prefix)
        self._reply(prefix, deadline, sentinel=True)

    def _run(self):
        try:
            if self._serial_factory is None:
                import serial
                factory = serial.Serial
            else:
                factory = self._serial_factory
            ser = factory(port=None, baudrate=self.profile['baud'], bytesize=8,
                          parity='N', stopbits=1, timeout=0, write_timeout=WRITE_S)
            self._serial = ser
            ser.rts, ser.dtr = self.profile['cat_rts'], self.profile['cat_dtr']
            ser.port = self.port
            ser.open()
            fcntl.ioctl(ser.fileno(), termios.TIOCEXCL)
            os.set_blocking(ser.fileno(), False)
            self._write('AI0;', time.monotonic() + EXCHANGE_S)
            self._ready.set()
            while not self._stop.is_set():
                with self._condition:
                    emergency, self._emergency = self._emergency, None
                    request = None if emergency or not self._queue else self._queue.popleft()
                    self._pending = request
                if emergency:
                    try:
                        if self.wedged or self.uncertain or self._owed or self._buf:
                            emergency.finish()
                        else:
                            _remaining(emergency.deadline)
                            self._write(emergency.command, emergency.deadline)
                            self._owed.append('TX')
                            emergency.finish(self._reply('TX', emergency.deadline, emergency=True))
                    except (RigError, OSError):
                        emergency.finish()
                elif request:
                    try:
                        with self._executing(request.deadline, request.arming, request.owner):
                            if self.uncertain:
                                self._resync(request.deadline)
                            if request.read_after:
                                # Guard and setter/readback share this request's
                                # deadline and cannot interleave with other calls.
                                self._write('VS;', request.deadline)
                                request.sent = True
                                self._owed.append('VS')
                                vfo = self._reply('VS', request.deadline)
                                if vfo == 'VS1;':
                                    raise RigError('VFO-B selected; CAT writes VFO-A only')
                                if vfo != 'VS0;':
                                    raise RigError('unknown VFO')
                            self._check(request.deadline, request.arming)
                            self._write(request.command, request.deadline)
                            request.sent = True
                            result = None
                            if request.prefix:
                                self._owed.append(request.prefix)
                                if request.read_after:
                                    self._write(request.read_after, request.deadline)
                                result = self._reply(request.prefix, request.deadline,
                                                     setter=request if request.read_after else None)
                            request.finish(result)
                    except (RigError, OSError) as exc:
                        request.finish(error=RigError(str(exc)))
                        if isinstance(exc, OSError) or 'desync' in str(exc) or 'ambiguous' in str(exc):
                            self._poison(str(exc))
                    finally:
                        with self._condition:
                            self._pending = None
                else:
                    self._consume(self._packets(0.01))
        except (OSError, RigError, ValueError, ImportError) as exc:
            if self._ready.is_set():
                self._runtime_error = exc
            else:
                self._open_error = exc
            self._poison(str(exc))
        finally:
            self._ready.set()
            if self._serial is not None:
                self._serial.close()
                self._serial = None


class YaesuCat(SerialTransport):
    def __init__(self, port, profile='ft891', *, serial_factory=None):
        super().__init__(port, profile, serial_factory=serial_factory)
        self.freq_retry_at = 0.0
        self._freq_backoff = 0.3

    def read(self, command, prefix):
        reply = self.exchange(command, prefix)
        if len(reply) <= len(prefix) + 1:
            raise RigError('empty CAT readback')
        return reply[len(prefix):-1]

    def write(self, command):
        # Only known read forms can safely fence a setter's optional '?;'.
        # VFO-B forms (FB, MD1, SH1, ...) are deliberately absent.
        prefixes = {'FA', 'MD0', 'SH0', 'NA0', 'RF0', 'PC',
                    'IS0', 'RT', 'XT', 'GT0', 'PA0', 'RA0',
                    'RG0', 'NR0', 'BC0', 'NB0', 'CO00'}
        prefixes.update('EX' + item for item in self.profile['readback']['menu_items'])
        prefix = next((p for p in sorted(prefixes, key=len, reverse=True)
                       if command.startswith(p) and len(command) > len(p) + 1), None)
        if prefix is None:
            raise RigError('unsupported CAT setter; CAT writes VFO-A only')
        self._exchange(command, prefix, _read_after=prefix + ';')

    def identify(self):
        if self.read('ID;', 'ID') != self.profile['id']:
            raise RigError('unexpected Yaesu CAT identity')
        return self.profile['name']

    def sole_ptt_owner(self, notes):
        return True

    def freq(self):
        with self._condition:
            self._check(time.monotonic() + 0.3, False)
            if time.monotonic() < self.freq_retry_at:
                raise RigError('CAT frequency retry backoff')
        try:
            reply = self.exchange('FA;', 'FA', timeout=0.3)
            if not re.fullmatch(r'FA\d{9};', reply) or int(reply[2:-1]) <= 0:
                raise RigError('invalid frequency readback')
        except RigError:
            with self._condition:
                self._freq_backoff = min(10.0, self._freq_backoff * 2)
                self.freq_retry_at = time.monotonic() + self._freq_backoff
            raise
        with self._condition:
            self.freq_retry_at = 0.0
            self._freq_backoff = 0.3
        return int(reply[2:-1])

    def set_freq(self, hz):
        if isinstance(hz, bool) or not isinstance(hz, int) or not 0 < hz < 10**9:
            raise RigError('invalid frequency')
        self.write(f'FA{hz:09d};')
        with self._condition:
            self.freq_retry_at = 0.0
            self._freq_backoff = 0.3

    def mode(self):
        vfo = self.exchange('VS;', 'VS')
        if vfo not in ('VS0;', 'VS1;'):
            raise RigError('unknown VFO')
        prefix = 'MD' + vfo[2]
        mode = self.exchange(prefix + ';', prefix)
        code = mode[len(prefix):-1]
        names = self.profile['readback']['mode_codes']
        if code not in names:
            raise RigError('unknown mode code')
        name = {'DATA-U': 'PKTUSB', 'DATA-L': 'PKTLSB'}.get(names[code], names[code])
        if code not in self.profile['width_modes']:
            raise RigError('no width table for this mode')
        width = self.exchange('SH0;', 'SH0')
        narrow = self.exchange('NA0;', 'NA0')
        match = re.fullmatch('SH0' + self.profile['width_prefix'] + r'(\d{2});', width)
        if not match or narrow not in ('NA00;', 'NA01;'):
            raise RigError('unknown width or narrow readback')
        index = int(match[1])
        widths = self.profile['data_widths']
        if not 1 <= index <= len(widths):
            raise RigError('width index outside rig table')
        hz = widths[index - 1]
        if narrow == 'NA01;':
            hz = min(hz, 500)
        return RadioMode(name, hz, vfo + mode + width + narrow)

    def set_mode(self, mode, width_hz=0):
        name = {'PKTUSB': 'DATA-U', 'PKTLSB': 'DATA-L'}.get(mode, mode)
        codes = self.profile['readback']['mode_codes']
        code = next((key for key, value in codes.items() if value == name), None)
        if code is None or (width_hz and (code not in self.profile['width_modes'] or
                                        width_hz not in self.profile['data_widths'])):
            raise RigError('mode or width outside rig table')
        self.write(f'MD0{code};')
        if width_hz:
            index = self.profile['data_widths'].index(width_hz) + 1
            self.write('NA00;')
            self.write(f"SH0{self.profile['width_prefix']}{index:02d};")
        return True

    def receiver_filters(self):
        mode = self.mode()
        roofs = self.profile.get('roofing_codes')
        if not roofs:
            return mode, None
        reply = self.read('RF0;', 'RF0')
        if reply not in roofs:
            raise RigError('roofing code outside rig table')
        return mode, roofs[reply]

    def ptt_state(self, *, _arming=False):
        try:
            result = self.exchange('TX;', 'TX', _arming=_arming)
        except RigError:
            self.reads_ptt = False
            return None
        state = {'TX0;': False, 'TX1;': True, 'TX2;': True}.get(result)
        self.reads_ptt = state is not None
        return state

    def set_power(self, watts):
        self._power(watts)
        if watts != int(watts):
            raise RigError('Yaesu power requires whole watts')
        self.write(f'PC{int(watts):03d};')


class _Refused(RigError):
    pass


class RigctldCat(_Driver):
    def __init__(self, host='127.0.0.1', port=4532, *, profile='generic', max_watts=None):
        super().__init__(profile)
        if max_watts is not None:
            if isinstance(max_watts, bool) or not math.isfinite(max_watts) or max_watts <= 0:
                raise ValueError('max_watts must be positive and finite')
            self.profile['max_watts'] = max_watts
        self.host, self.port = host, port
        self._sock = None
        self._buf = b''

    def _connect(self, deadline):
        return socket.create_connection((self.host, self.port), _remaining(deadline))

    def _wire(self, sock, line, deadline, *, persistent=False):
        sock.settimeout(_remaining(deadline))
        sock.sendall(('+' + line + '\n').encode('ascii'))
        buf = self._buf if persistent else b''
        while True:
            match = re.search(rb'(?:^|\n)(?:Reply: )?RPRT (-?\d+)\r?\n', buf)
            if match:
                if persistent:
                    self._buf = buf[match.end():]
                data = buf[:match.end()].decode('ascii')
                _remaining(deadline)
                if int(match[1]) != 0:
                    raise _Refused(f'rigctld refused {line!r}: RPRT {int(match[1])}')
                # Plain setters also return RPRT 0. Only an extended-response
                # header proves that the daemon honoured our '+' framing.
                if not re.match(r'[A-Za-z_][A-Za-z0-9_]*:[^\r\n]*\r?\n', data):
                    raise RigError('rigctld reply lacks extended-response header')
                return data.strip()
            if len(buf) > 1024 * 1024:
                raise RigError('rigctld reply exceeds limit')
            sock.settimeout(_remaining(deadline))
            chunk = sock.recv(65536)
            if not chunk:
                raise RigError('rigctld closed the connection')
            buf += chunk.replace(b'\0', b'\n')

    def _open(self, deadline):
        if self._sock is not None:
            return
        self._sock = self._connect(deadline)
        self._buf = b''
        self.cache_free = True
        try:
            self._wire(self._sock, '\\set_cache 0', deadline, persistent=True)
        except _Refused:
            self.cache_free = False
            self._wire(self._sock, 'f', deadline, persistent=True)

    @leased
    def open(self):
        deadline = time.monotonic() + EXCHANGE_S
        with self._condition:
            if self._sock is None:
                self.poisoned = False
        with self._executing(deadline):
            try:
                self._open(deadline)
            except (OSError, RigError, UnicodeError) as exc:
                self._disconnect()
                raise RigError(f'no rigctld at {self.host}:{self.port}: {exc}. Start rigctld first. It owns the CAT port.') from exc

    def _disconnect(self):
        sock, self._sock = self._sock, None
        self._buf = b''
        if sock:
            sock.close()

    def close(self):
        with self._condition:
            self.poisoned = True
            self._intervals.clear()
            self._condition.notify_all()
        self._disconnect()

    @leased
    def command(self, line, *, _arming=False):
        if not line or any(c in line for c in '\n\r\0'):
            raise RigError('expected one rigctld command')
        deadline = time.monotonic() + EXCHANGE_S
        disposable = line.startswith('\\get_conf ')
        setup_complete = False
        with self._executing(deadline, _arming):
            try:
                self._open(deadline)
                setup_complete = True
                if disposable:
                    # Hamlib may close this connection after answering get_conf.
                    with self._connect(deadline) as sock:
                        try:
                            self._wire(sock, '\\set_cache 0', deadline)
                        except _Refused:
                            self._wire(sock, 'f', deadline)
                        result = self._wire(sock, line, deadline)
                else:
                    result = self._wire(self._sock, line, deadline, persistent=True)
                if self.poisoned:
                    raise RigError('CAT transport poisoned')
                return result
            except _Refused:
                raise
            except (OSError, RigError, UnicodeError) as exc:
                if not disposable or not setup_complete:
                    self._disconnect()
                raise RigError(f'CAT command {line!r} failed: {exc}') from exc

    @staticmethod
    def _field(data, name):
        values = [line.partition(':')[2].strip() for line in data.splitlines()
                  if line.partition(':')[0].strip() == name]
        if len(values) != 1:
            raise RigError(f'missing or ambiguous {name} field')
        return values[0]

    def identify(self):
        return self._field(self.command('\\dump_caps'), 'Model name')

    def get_conf(self, name, *, _arming=False):
        if not name or any(c.isspace() for c in name):
            raise ValueError('configuration token must be one word')
        return self.command('\\get_conf ' + name, _arming=_arming)

    def sole_ptt_owner(self, notes):
        try:
            data = self.get_conf('ptt_type', _arming=True)
        except RigError as exc:
            notes.append(f'rigctld PTT ownership is unverified: {exc}')
            return False
        values = re.findall(r'(?m)^ptt_type=([^\r\n]*)$', data)
        # Only an explicit None proves ownership; empty is unverified too.
        if len(values) != 1 or not values[0].strip():
            notes.append('rigctld did not name its PTT owner')
            return False
        if values[0].strip().lower() == 'none':
            return True
        raise RigError(f'rigctld is configured to key the radio itself (ptt_type={values[0]!r})')

    def freq(self):
        value = self._field(self.command('f'), 'Frequency')
        try:
            hz = float(value)
            if not math.isfinite(hz) or hz <= 0 or hz != int(hz):
                raise ValueError
            return int(hz)
        except ValueError as exc:
            raise RigError('invalid frequency readback') from exc

    def set_freq(self, hz):
        if isinstance(hz, bool) or not isinstance(hz, int) or hz <= 0:
            raise RigError('invalid frequency')
        self.command(f'F {hz}')

    def mode(self):
        reply = self.command('m')
        name, width = self._field(reply, 'Mode'), self._field(reply, 'Passband')
        if not name or not width.isdecimal():
            raise RigError('invalid mode readback')
        return RadioMode(name, int(width), reply)

    def set_mode(self, mode, width_hz=0):
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', mode) or not isinstance(width_hz, int)
                or isinstance(width_hz, bool) or width_hz < 0):
            raise RigError('invalid mode or width')
        try:
            self.command(f'M {mode} {width_hz}')
        except _Refused:
            return False
        return True

    def receiver_filters(self):
        mode = self.mode()
        roofs = self.profile.get('roofing_codes')
        if not roofs:
            return mode, None
        reply = self.read('RF0;', 'RF0')
        if reply not in roofs:
            raise RigError('roofing code outside rig table')
        return mode, roofs[reply]

    def ptt_state(self, *, _arming=False):
        try:
            reply = self.command('t', _arming=_arming)
            state = {'0': False, '1': True}.get(self._field(reply, 'PTT'))
        except RigError:
            state = None
        self.reads_ptt = state is not None
        return state

    def set_power(self, watts):
        maximum = self._power(watts)
        self.command(f'L RFPOWER {watts / maximum:g}')

    def read(self, command, prefix):
        data = self.command('\\send_raw ; ' + command)
        body = '\n'.join(line.removeprefix('Reply: ') for line in data.splitlines()
                         if not line.startswith(('send_cmd:', 'send_raw:', 'RPRT ')))
        replies = re.findall(r'[^;\s]+;', body)
        if (len(replies) != 1 or not replies[0].startswith(prefix)
                or len(replies[0]) <= len(prefix) + 1):
            raise RigError('missing or ambiguous raw CAT reply')
        return replies[0][len(prefix):-1]

    def write(self, command):
        self.command('\\send_raw 0 ' + command)

    def emergency_unkey(self):
        with self._condition:
            self.poisoned = True
            self._condition.notify_all()
        deadline = time.monotonic() + EMERGENCY_S
        try:
            with self._connect(deadline) as sock:
                cache_free = True
                try:
                    self._wire(sock, '\\set_cache 0', deadline)
                except _Refused:
                    cache_free = False
                self._wire(sock, 'T 0', deadline)
                if not cache_free:
                    return None
                reply = self._wire(sock, 't', deadline)
                return {'0': False, '1': True}.get(self._field(reply, 'PTT'))
        except (OSError, RigError, UnicodeError):
            return None
