# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The X6100 base board, keyed directly while the GUI is stopped.

The base takes its control registers over I2C (address 0x72 on /dev/i2c-0) in
the control library's framing: a big-endian register address, then the 32-bit
value. Register 12 carries the five TX request bits; digital modes key with
`MODEM`, which takes TX audio from the codec. A write is accepted or refused by
the bus. Nothing reads back, so an accepted write is evidence that the command
reached the base and not that RF came up or went away.

What the base reports comes in its flow packets on /dev/ttyS1, about every
35 ms. One process may read that port, and here it is the guard
(integration/guifree-timing/guard.c), which forwards each packet's TX flag,
power and SWR over the socket `GuardLink` reads. The guard also unkeys on its
own when this process dies, hangs, or exceeds a limit, so `Rig`'s watchdog is
not the only thing standing between a bug and a stuck carrier.

Nothing here tunes the radio or sets its power. Frequency, mode and power are
the GUI's, set before it steps aside; the base keeps them.
"""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import functools
import math
import os
import socket
import stat
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

from hfmodem.core.operations import OperationGate
from hfmodem.core.ptt import PttError
from hfmodem.core.rig import RigError

I2C_DEVICE = "/dev/i2c-0"
I2C_ADDRESS = 0x72
I2C_RDWR = 0x0707
REGISTER = 12

SPLIT = 0x00002
VOICE_RECORD = 0x00008
SWR_SCAN = 0x00010
ATU_ENABLED = 0x01000
ATU_TUNE = 0x02000
MODEM = 0x04000
CALIBRATION = 0x08000
POWER_OFF = 0x10000
MIC_PTT = 0x40000
TX_BITS = MIC_PTT | MODEM | ATU_TUNE | SWR_SCAN | CALIBRATION
KNOWN_BITS = TX_BITS | POWER_OFF | SPLIT | VOICE_RECORD | ATU_ENABLED

#: The library's write: three tries, 2 ms apart.
TRIES = 3
RETRY_S = 0.002

#: A report older than this says nothing about the transmitter now.
FRESH_S = 0.150
EMERGENCY_S = 0.400
PROOF_S = 0.150

TELEMETRY = struct.Struct("<4sIqBBBBBBHI")
HELLO = struct.Struct("<4s128sIIIII")
HEARTBEAT = struct.Struct("<4sqB3xI")

#: Telemetry flag bits and the guard's trip codes.
FLAG_TX, FLAG_ATU, FLAG_TRIPPED = 1, 2, 4


def frame(word: int) -> bytes:
    """Register 12's six bytes on the wire, as the control library sends them."""
    return struct.pack(">H", REGISTER * 4) + struct.pack("<I", word)


def receive_word(word: int) -> int:
    """`word` checked as a receive state: known bits only, nothing that keys or powers off."""
    if not 0 <= word <= 0xFFFFFFFF or word & ~KNOWN_BITS:
        raise ValueError(f"register 12 word {word:#x} has bits the control library does not name")
    if word & (TX_BITS | POWER_OFF):
        raise ValueError(f"register 12 receive word {word:#x} keys or powers off the radio")
    return word


class _Msg(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16),
                ("len", ctypes.c_uint16), ("buf", ctypes.c_void_p)]


class _Transfer(ctypes.Structure):
    _fields_ = [("msgs", ctypes.POINTER(_Msg)), ("nmsgs", ctypes.c_uint32)]


class I2cBus:
    """One I2C_RDWR write per call, as the GUI and the recovery helper make it.

    A UNIX datagram socket in place of the device is the simulated base; every
    other path is the kernel's i2c-dev.
    """

    def __init__(self, path: str = I2C_DEVICE) -> None:
        self.path = path
        if stat.S_ISSOCK(os.stat(path).st_mode):
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._sock.connect(path)
            self._fd = None
        else:
            self._sock = None
            self._fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)

    def write(self, data: bytes) -> bool:
        if self._sock is not None:
            return self._sock.send(data) == len(data)
        buf = ctypes.create_string_buffer(data, len(data))
        msg = _Msg(I2C_ADDRESS, 0, len(data), ctypes.addressof(buf))
        transfer = _Transfer(ctypes.pointer(msg), 1)
        try:
            return fcntl.ioctl(self._fd, I2C_RDWR, transfer) == 1
        except OSError:
            return False

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
        elif self._fd is not None:
            os.close(self._fd)
            self._fd = None


@dataclass(frozen=True)
class Write:
    """One register 12 write: monotonic ns before and after the bus call."""
    begin_ns: int
    end_ns: int
    word: int
    accepted: bool


class Register12:
    """Register 12 between a fixed receive word and that word plus `MODEM`."""

    def __init__(self, rx_word: int, bus) -> None:
        self.rx_word = receive_word(rx_word)
        self.bus = bus
        self.writes: deque[Write] = deque(maxlen=4096)
        self._lock = threading.Lock()

    def write(self, on: bool) -> bool:
        word = self.rx_word | MODEM if on else self.rx_word
        with self._lock:
            begin = time.monotonic_ns()
            for attempt in range(TRIES):
                if attempt:
                    time.sleep(RETRY_S)
                if self.bus.write(frame(word)):
                    ok = True
                    break
            else:
                ok = False
            self.writes.append(Write(begin, time.monotonic_ns(), word, ok))
        return ok


class X6100Ptt:
    """`core.ptt.Ptt` over register 12. `sense` reports the last write the bus accepted."""

    def __init__(self, register: Register12) -> None:
        self.register = register
        self.port = f"{getattr(register.bus, 'path', 'bus')} reg {REGISTER}"
        #: Stays None: an accepted write is not RF off, so a panic goes on to the
        #: CAT leg, which waits for the base to report TX down.
        self.released_low: bool | None = None
        self._state: bool | None = None
        self._released = False

    def open(self) -> None:
        """Write the receive word, so the line reads low before `Rig.arm`."""
        self.assert_(False)

    def assert_(self, on: bool) -> None:
        if on and self._released:
            raise PttError("the keying line was released")
        if not self.register.write(on):
            self._state = None
            raise PttError(f"the base refused the register {REGISTER} write")
        self._state = on

    def sense(self) -> bool | None:
        return self._state

    def release(self) -> None:
        # Latched first: a key racing a panic must not write MODEM after the unkey.
        self._released = True
        try:
            down = self.register.write(False)
        except Exception:
            down = False
        self._state = False if down else None


@dataclass(frozen=True)
class Report:
    """One flow packet as the guard saw it."""
    seq: int
    mono_ns: int
    tx: bool
    atu: bool
    tripped: bool
    power_w: float
    swr: float
    alc: float
    trip: int
    flow_fmt: int
    lo_freq: int


class GuardLink:
    """The measuring side of the guard's socket: reports in, heartbeats out."""

    def __init__(self, sock: socket.socket, *, heartbeat_s: float | None = None) -> None:
        sock.set_inheritable(False)
        self.sock = sock
        kind, version, rx_word, max_key_ms, heartbeat_ms, band_lo, band_hi = HELLO.unpack(
            self._recv(HELLO.size))
        if kind != b"XGH1":
            raise RigError(f"the guard did not say hello: {kind!r}")
        self.base_version = version.split(b"\0", 1)[0].decode("ascii", "replace")
        self.rx_word, self.max_key_ms = rx_word, max_key_ms
        self.guard_heartbeat_ms = heartbeat_ms
        self.band = (band_lo, band_hi) if band_hi else None
        self.reports: deque[Report] = deque(maxlen=1 << 16)
        #: The keyed interval the caller has asked for, in ms; 0 while none is due.
        #: The guard unkeys a transmission nobody declared.
        self.expect_ms = 0
        self.failure: BaseException | None = None
        self._latest: Report | None = None
        self._changed = threading.Condition()
        self._stop = threading.Event()
        self._beat_lock = threading.Lock()
        # Four beats inside the guard's timeout, and never slower than 100 ms.
        self._heartbeat_s = heartbeat_s or min(0.1, heartbeat_ms / 4000)
        self._threads = [threading.Thread(target=self._read, name="guard-read", daemon=True),
                         threading.Thread(target=self._beat, name="guard-beat", daemon=True)]
        for thread in self._threads:
            thread.start()

    @classmethod
    def from_environment(cls, **kw) -> "GuardLink":
        # Popped: the socket is not inherited, so a child must not go looking for it.
        fd = os.environ.pop("GUIFREE_GUARD_FD", None)
        if not fd:
            raise RigError("the X6100 onboard rig runs as the guard's child (run.sh), which "
                           "hands it the register 12 receive word; GUIFREE_GUARD_FD is unset")
        return cls(socket.socket(fileno=int(fd)), **kw)

    def _recv(self, size: int) -> bytes:
        data = self.sock.recv(256)
        if len(data) != size:
            raise RigError(f"guard message of {len(data)} bytes, expected {size}")
        return data

    def _read(self) -> None:
        try:
            while not self._stop.is_set():
                data = self.sock.recv(256)
                if not data:
                    raise RigError("the guard closed its socket")
                (kind, seq, mono_ns, flags, power, swr, alc, trip, fmt, _,
                 lo_freq) = TELEMETRY.unpack(data)
                if kind != b"XGT1":
                    raise RigError(f"unexpected guard message {kind!r}")
                report = Report(seq, mono_ns, bool(flags & FLAG_TX), bool(flags & FLAG_ATU),
                                bool(flags & FLAG_TRIPPED), power * 0.1, swr * 0.1, alc * 0.1,
                                trip, fmt, lo_freq)
                self.reports.append(report)
                with self._changed:
                    self._latest = report
                    self._changed.notify_all()
        except BaseException as exc:
            if not self._stop.is_set():
                self.failure = exc
            with self._changed:
                self._changed.notify_all()

    def _beat(self) -> None:
        while not self._stop.wait(self._heartbeat_s):
            try:
                self.heartbeat()
            except OSError as exc:
                self.failure = exc
                return

    def heartbeat(self) -> None:
        with self._beat_lock:
            ms = self.expect_ms
            self.sock.send(HEARTBEAT.pack(b"XGB1", time.monotonic_ns(), int(ms > 0), ms))

    def declare(self, ms: int) -> None:
        """Set the keyed interval and tell the guard now, so no beat carries a stale one."""
        with self._beat_lock:
            self.expect_ms = ms
        self.heartbeat()

    def tx_run_ms(self) -> int:
        """How long the base has reported TX without a break; 0 when it reports none now."""
        reports = list(self.reports)
        if not reports or not reports[-1].tx or self.tx() is not True:
            return 0
        start = reports[-1].mono_ns
        for report in reversed(reports):
            if not report.tx:
                break
            start = report.mono_ns
        return math.ceil((time.monotonic_ns() - start) / 1e6)

    @property
    def latest(self) -> Report | None:
        return self._latest

    def tx(self, *, fresh_s: float = FRESH_S) -> bool | None:
        report = self._latest
        if self.failure is not None or report is None:
            return None
        if time.monotonic_ns() - report.mono_ns > fresh_s * 1e9:
            return None
        return report.tx

    def wait_for(self, tx: bool, timeout_s: float) -> bool:
        """True once a report fresher than the call says `tx`."""
        start = time.monotonic_ns()
        deadline = time.monotonic() + timeout_s
        with self._changed:
            while True:
                report = self._latest
                if report is not None and report.mono_ns >= start and report.tx == tx:
                    return True
                left = deadline - time.monotonic()
                if left <= 0 or self.failure is not None:
                    return False
                self._changed.wait(left)

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()


class X6100Cat:
    """The part of `core.cat.CatDriver` that `Rig` needs, from the base and its guard.

    The dial is the operator's statement of where the GUI left the radio; the base
    cannot be asked.
    """

    model = 0
    reads_ptt = True
    cache_free = True

    def __init__(self, link: GuardLink, register: Register12, *, dial_hz: int) -> None:
        # `register` should be on its own bus handle: it is the panic path's
        # second leg, behind the keying line's.
        self.link, self.register, self.dial_hz = link, register, int(dial_hz)
        self.operations = OperationGate()
        self.keyed_probe = lambda: False
        self.clock = lambda: time.monotonic() * 48000
        #: Called once, off the main thread, when the guard link fails. The owner
        #: sets its own shutdown; unowned, the base is unkeyed and the process ends.
        self.lost = _unkey_and_exit(register)
        self._closed: threading.Event | None = None

    def open(self) -> None:
        if self.link.failure is not None:
            raise RigError(f"guard link failed: {self.link.failure}")
        if self._closed is None:
            self._closed = threading.Event()
            threading.Thread(target=self._watch_link, args=(self._closed,),
                             name="guard-watch", daemon=True).start()

    def _watch_link(self, closed: threading.Event) -> None:
        # The guard is the independent unkey; without it this process must not go on.
        while not closed.wait(0.05):
            if self.link.failure is not None:
                self.lost(f"guard link lost: {self.link.failure}")
                return

    def close(self) -> None:
        # Not joined: `lost` may be the owner's shutdown, which never returns. The
        # link stays up; it is the process's, and closing it tells the guard its
        # child has gone.
        if self._closed is not None:
            self._closed.set()
            self._closed = None

    @contextlib.contextmanager
    def hold(self, deadline_index=None):
        yield True

    def protect(self, start_index, end_index):
        """Declare the key to the guard before the line goes up; it unkeys an undeclared one."""
        ms = (self.link.max_key_ms if end_index is None
              else max(1, math.ceil((end_index - start_index) / 48)))
        # The guard times a declaration from the start of the base's TX run. A key
        # that follows the last before the base reports TX down is in that run.
        ms = min(ms + self.link.tx_run_ms(), self.link.max_key_ms)
        try:
            self.link.declare(ms)
        except OSError as exc:
            self.link.expect_ms = 0
            raise RigError(f"the guard did not take the key's declaration: {exc}") from None
        return ms

    def unprotect(self, token) -> None:
        with contextlib.suppress(OSError):
            self.link.declare(0)

    def identify(self) -> str:
        return f"Xiegu X6100 base {self.link.base_version}".strip()

    def sole_ptt_owner(self, notes) -> bool:
        notes.append("the guard owns the base's flow port and found no x6100_gui running")
        return True

    def freq(self) -> int:
        return self.dial_hz

    def set_freq(self, hz: int) -> None:
        raise RigError("the X6100 is tuned in its GUI before the GUI steps aside")

    def set_mode(self, mode: str, width_hz: int = 0) -> bool:
        raise RigError("the X6100's mode is set in its GUI before the GUI steps aside")

    def set_power(self, watts: float) -> None:
        raise RigError("the X6100's power is set in its GUI before the GUI steps aside")

    def keying_line(self) -> X6100Ptt:
        """Register 12 on a bus handle of its own, with the guard's receive word."""
        return X6100Ptt(Register12(self.link.rx_word, I2cBus(self.register.bus.path)))

    def ptt_state(self, *, _arming=False) -> bool | None:
        # The PTT proof asks 50 ms after keying; the base reports every 35 ms or so,
        # some time after the write, so wait for a report that can answer.
        if _arming and self.keyed_probe():
            return True if self.link.wait_for(True, PROOF_S) else self.link.tx()
        return self.link.tx()

    def emergency_unkey(self) -> bool | None:
        self.register.write(False)
        return False if self.link.wait_for(False, EMERGENCY_S) else None


def _unkey_and_exit(register: Register12):
    def lost(why: str) -> None:
        try:
            os.write(2, f"!! {why}; unkeying and exiting\n".encode())
            register.write(False)
        finally:
            os._exit(70)
    return lost


@functools.cache
def guard_link() -> GuardLink:
    """The process's one link to the guard that started it."""
    return GuardLink.from_environment()


def onboard(dial_hz: int, *, i2c: str | None = None) -> X6100Cat:
    """CAT for the onboard rig; `X6100Cat.keying_line` adds the line when transmitting.

    The receive word is the guard's: the operator confirms it from the running
    radio for `run.sh`, and nothing here supplies one.
    """
    link = guard_link()
    return X6100Cat(link, Register12(link.rx_word, I2cBus(i2c or I2C_DEVICE)), dial_hz=dial_hz)
