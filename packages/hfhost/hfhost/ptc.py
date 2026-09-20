# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Client for the SCS PTC serial dialect: terminal mode and CRC hostmode.

This is the third host dialect in the tree, and the only one that is not TCP.
Winlink reaches PACTOR through a serial port speaking WA8DED hostmode inside
SCS's CRC framing, so a modem that wants to be a drop-in PACTOR replacement
emulates a PTC — and a bench that wants to grade it has to speak the client half.

Sources, both public: the SCS PTC-IIIusb manual chapter 10 (10.5 extended
hostmode, 10.6 status channel, 10.9 CRC hostmode) and the WA8DED hostmode user's
guide. Cross-checked against harenber/ptc-go, the PACTOR driver Pat uses, which
is the closest thing to an executable specification for what a real client sends.

Framing, outside in::

    CRC layer   AA AA | stuffed( packet + crc16_lo + crc16_hi )
                stuffing inserts 00 after every AA *following* the header
    packet      host->modem:  channel, flags, len-1, data
                modem->host:  channel, code, [len-1,] data

A frame's length is only knowable by parsing it, so the decoder is
direction-aware. That is a property of the dialect, not a shortcut here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import NamedTuple

HEADER = b"\xaa\xaa"
REQUEST = b"\xaa\xaa\xaa\x55"          # modem asks for a repeat after a bad CRC
MAX_DATA = 256                          # manual 10.9.3
NACK_TIMEOUT = 0.25                     # manual 10.9.3, a minimum

# modem -> host response codes (manual 10.6 / WA8DED guide)
OK, MSG, FAIL, LINK = 0, 1, 2, 3
MON_HEADER, MON_HEADER_INFO, MON_INFO, DATA = 4, 5, 6, 7

_STRING_CODES = (MSG, FAIL, LINK, MON_HEADER, MON_HEADER_INFO)
_COUNTED_CODES = (MON_INFO, DATA)


class PtcError(Exception):
    pass


def crc16(data: bytes) -> int:
    """CRC-16/X-25, as AX.25 uses it: reflected 0x1021, init 0xFFFF, xorout
    0xFFFF. The manual's worked example (04 01 01 71 71 -> lo 213, hi 153) is
    written in decimal, the same notation it uses for the #170 #170 header, so
    the bytes are 04 01 01 47 47 and the CRC is 0x99D5; see tests."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def stuff(body: bytes) -> bytes:
    """Insert 00 after every AA, so no AA AA can appear inside a frame."""
    out = bytearray()
    for b in body:
        out.append(b)
        if b == 0xAA:
            out.append(0x00)
    return bytes(out)


def unstuff(body: bytes) -> bytes:
    out = bytearray()
    skip = False
    for b in body:
        if skip:
            skip = False
            if b == 0x00:
                continue            # the stuffed byte; drop it
        out.append(b)
        skip = b == 0xAA
    return bytes(out)


@dataclass(frozen=True, slots=True)
class Packet:
    """One host -> modem packet."""
    channel: int
    is_command: bool
    data: bytes = b""
    counter: int = 0
    ignore_counter: bool = False

    def flags(self) -> int:
        f = 1 if self.is_command else 0
        if self.ignore_counter:
            f |= 0x40
        if self.counter:
            f |= 0x80
        return f

    def encode(self) -> bytes:
        # The length byte is len-1 (10.6, 10.9.2), so an empty body has no
        # encoding at all: zero would claim one byte and send none, and the modem
        # then waits for a byte that never comes.
        if not 1 <= len(self.data) <= MAX_DATA:
            raise PtcError(f"packet body must be 1..{MAX_DATA} bytes,"
                           f" got {len(self.data)}")
        body = bytes([self.channel, self.flags(), len(self.data) - 1])
        body += self.data
        crc = crc16(body)
        return HEADER + stuff(body + bytes([crc & 0xFF, crc >> 8]))


@dataclass(frozen=True, slots=True)
class Response:
    """One modem -> host packet."""
    channel: int
    code: int
    data: bytes = b""

    @property
    def text(self) -> str:
        return self.data.decode("latin-1", "replace").rstrip("\x00")


def _split_response(body: bytes) -> Response | None:
    """Parse one modem->host packet, or None if more bytes are needed. The
    body shape depends on the response code, which is why this cannot be a
    length-prefixed reader."""
    if len(body) < 2:
        return None
    channel, code = body[0], body[1]
    if code == OK:
        return Response(channel, code)
    if code in _STRING_CODES:
        end = body.find(b"\x00", 2)
        if end < 0:
            return None
        return Response(channel, code, body[2:end])
    if code in _COUNTED_CODES:
        if len(body) < 3:
            return None
        want = body[2] + 1
        if len(body) < 3 + want:
            return None
        return Response(channel, code, body[3:3 + want])
    raise PtcError(f"unknown response code {code}")


class Deframer:
    """CRC-hostmode frames out of a byte stream.

    A frame is only complete once its packet parses *and* its CRC checks, and
    the packet's length is not known in advance — so this accumulates and
    retries rather than reading a length. A frame whose CRC fails reads the same
    as one that has not fully arrived, and neither yields a Response; the
    dialect's repeat request is the modem's to send.

    THEY ONLY READ THE SAME UNTIL MORE ARRIVES. A header whose body is corrupt
    never completes, so without a resync it sits at offset 0 for the life of the
    link: `find(HEADER)` returns it every time, the candidate scan fails every
    time, and every good frame behind it is unreachable. Corrupting any of the
    10 non-header byte positions of a real frame wedged it permanently, the
    buffer grew without bound (51 KB after 200 x 256 B, 102 KB after 400), and
    the scan is quadratic in what it holds — 3.43 s for the first 50 KB against
    14.63 s for the second.

    `stuff` is what makes the recovery sound rather than a guess: it inserts 00
    after every AA, so AA AA cannot occur inside a frame. A second header in the
    buffer is therefore a real frame start, and a scan that failed with one
    behind it has failed for good — drop to it. With no second header the frame
    may simply be short, so waiting is right until `MAX_FRAME`, past which no
    further byte can complete this one either. `resyncs` counts both, because a
    link losing frames should be visible rather than merely quiet."""

    #: Nothing longer can be one frame: 2 header, 3 packet header, `MAX_DATA`
    #: body, 2 CRC, each byte of which may be stuffed to two on the wire.
    MAX_FRAME = 2 + 2 * (3 + MAX_DATA + 2)

    def __init__(self) -> None:
        self._buf = bytearray()
        self.resyncs = 0
        self.requests = 0
        self.crc_errors = 0

    @property
    def nacks(self) -> int:
        """Modem reactions that ask the host to repeat: its request packet,
        and a reply of ours that arrived corrupt (10.9.3)."""
        return self.requests + self.crc_errors

    @property
    def partial(self) -> bool:
        """A header is in hand and its frame has not completed."""
        return self._buf.find(HEADER) >= 0

    def feed(self, chunk: bytes) -> list[Response]:
        self._buf += chunk
        out: list[Response] = []
        while True:
            start = self._buf.find(HEADER)
            if start < 0:
                if len(self._buf) > 4096:
                    del self._buf[:-1]      # no header in sight; do not grow
                return out
            del self._buf[:start]
            # After the header every literal AA is stuffed to AA 00, so AA AA AA
            # can only be the request packet's third byte (10.9.2, 10.9.5).
            if self._buf.startswith(REQUEST):
                del self._buf[:len(REQUEST)]
                self.requests += 1
                continue
            if REQUEST.startswith(bytes(self._buf)):
                return out
            rest = bytes(self._buf[2:])
            if not rest:
                return out
            raw = unstuff(rest)
            if len(raw) < 4:
                return out
            resp, bad_crc = None, False
            for body_len in range(2, len(raw) - 1):
                body, crc_bytes = raw[:body_len], raw[body_len:body_len + 2]
                if len(crc_bytes) < 2:
                    break
                try:
                    parsed = _split_response(body)
                except PtcError:
                    continue
                if parsed is None or len(body) != _encoded_len(body):
                    continue
                want = crc_bytes[0] | (crc_bytes[1] << 8)
                if crc16(body) == want:
                    resp = parsed
                    consumed = _restuffed_len(body + crc_bytes)
                    del self._buf[:2 + consumed]
                    break
                bad_crc = True
            if resp is None:
                nxt = self._buf.find(HEADER, 2)
                if nxt < 0 and len(self._buf) <= self.MAX_FRAME:
                    return out              # still short of a frame; wait
                del self._buf[:nxt if nxt > 0 else 2]
                self.resyncs += 1
                if bad_crc:
                    self.crc_errors += 1
                continue
            out.append(resp)


def _encoded_len(body: bytes) -> int:
    """How long this packet claims to be, so a candidate split is only accepted
    when it consumes exactly the packet and nothing more."""
    code = body[1]
    if code == OK:
        return 2
    if code in _STRING_CODES:
        end = body.find(b"\x00", 2)
        return end + 1 if end >= 0 else -1
    if code in _COUNTED_CODES:
        return 3 + body[2] + 1 if len(body) >= 3 else -1
    return -1


def _restuffed_len(raw: bytes) -> int:
    return len(raw) + raw.count(0xAA)


class LinkStatus(NamedTuple):
    """What `L` answers. The manual documents the command and its channel
    parameter but not the fields; the names are the WA8DED guide's."""
    link_events: int
    rx_frames: int
    unsent: int
    unacked: int
    retries: int
    state: int

    @classmethod
    def parse(cls, text: str) -> LinkStatus:
        try:
            values = [int(field) for field in text.split()]
        except ValueError:
            raise PtcError(f"L answered {text!r}") from None
        if len(values) != len(cls._fields):
            raise PtcError(f"L answered {len(values)} fields, want {len(cls._fields)}")
        return cls(*values)


class PtcClient:
    """A host attached to a PTC (or something emulating one) over a byte stream.

    Deliberately transport-agnostic: it takes any object with read/write, so a
    pty, a real serial port and a socketpair are all the same to it. Terminal
    mode and hostmode are separate methods because they are separate layers of
    the dialect, and a client that conflates them cannot recover a modem that
    was left in the wrong one.

    In hostmode this is the MASTER of manual 10.9.3: every action it takes draws
    exactly one reaction, which it waits for before taking another. The last
    packet stays buffered, the packet counter in bit 7 inverts on each new one,
    and bit 6 — which tells the modem to execute a packet whatever its counter —
    is set on the first packet after a hostmode start and on no other, so a
    repeat is recognised as a repeat rather than run twice.
    """

    def __init__(self, stream, *, channel: int = 4, name: str = "ptc") -> None:
        self.stream = stream
        self.channel = channel
        self.name = name
        self.deframer = Deframer()
        self.repeats = 0
        self._lock = threading.RLock()
        self._hostmode = False
        self._counter = 0
        self._buffered = b""
        self._first = True

    # -- terminal mode -----------------------------------------------------

    def command(self, text: str, timeout: float = 2.0) -> str:
        """Send a terminal-mode command and read whatever comes back before the
        stream goes quiet. Terminal mode has no framing, so quiet is the only
        available end marker."""
        with self._lock:
            self.stream.write(text.encode("ascii") + b"\r")
            self._flush()
            return self._read_quiet(timeout).decode("latin-1", "replace")

    def enter_hostmode(self, timeout: float = 2.0) -> None:
        """`JHOST4`, then start the host protocol from scratch: the modem's
        request bit is undefined across the switch, so our first packet carries
        bit 6 and nothing from before it can be repeated (10.9.6)."""
        self.command("JHOST4", timeout=timeout)
        with self._lock:
            self.deframer = Deframer()
            self._counter, self._buffered, self._first = 0, b"", True
            self._hostmode = True

    def leave_hostmode(self, timeout: float = NACK_TIMEOUT) -> None:
        try:
            self.exchange(Packet(0, True, b"JHOST0"), timeout=timeout)
        finally:
            self._hostmode = False

    @property
    def in_hostmode(self) -> bool:
        return self._hostmode

    # -- hostmode ----------------------------------------------------------

    def send(self, packet: Packet) -> None:
        """Stamp the master's flags, buffer the frame against a repeat, and put
        it on the wire (10.9.3)."""
        with self._lock:
            self._counter ^= 1
            frame = replace(packet, counter=self._counter,
                            ignore_counter=self._first).encode()
            self._first = False
            self._buffered = frame
            self._write(frame)

    def repeat(self) -> None:
        """Retransmit the buffered packet, request bit unchanged (10.9.3)."""
        with self._lock:
            if not self._buffered:
                raise PtcError("no packet to repeat")
            self.repeats += 1
            self._write(self._buffered)

    def exchange(self, packet: Packet, *, timeout: float = NACK_TIMEOUT,
                 retries: int = 3) -> Response:
        """One host action and the one modem reaction it draws (10.9.2).

        Silence past `timeout`, a request packet, or a reply that fails its CRC
        are all NACK conditions, and all draw the same reaction: send the
        buffered packet again.
        """
        with self._lock:
            self.send(packet)
            left = retries
            while (resp := self._reaction(timeout)) is None:
                if not left:
                    raise PtcError(f"no reaction to {packet.data[:24]!r}"
                                   f" after {retries} repeats")
                left -= 1
                self.repeat()
            return resp

    def cmd(self, text: str, channel: int | None = None) -> None:
        ch = self.channel if channel is None else channel
        self.send(Packet(ch, True, text.encode("ascii")))

    def ask(self, text: str, channel: int | None = None, **kw) -> Response:
        """A hostmode command and its answer."""
        ch = self.channel if channel is None else channel
        return self.exchange(Packet(ch, True, text.encode("ascii")), **kw)

    def write_data(self, blob: bytes) -> None:
        """Bulk fill, unacknowledged: for a stream with nothing answering on it.
        Traffic against a live modem goes through `exchange`, which is what the
        master protocol's one-action-one-reaction rule asks for."""
        for i in range(0, len(blob), MAX_DATA):
            self.send(Packet(self.channel, False, blob[i:i + MAX_DATA]))

    def poll(self, timeout: float = 0.2) -> list[Response]:
        return self.deframer.feed(self._read_quiet(timeout))

    def status(self, channel: int | None = None, **kw) -> LinkStatus:
        """`L <channel>` -- the link status of one channel (10.4.8).

        The manual gives `L` a channel parameter, 0..31, and says nothing about
        the channel the packet itself is addressed to; sending it on the channel
        it asks about satisfies either reading.
        """
        ch = self.channel if channel is None else channel
        resp = self.ask(f"L {ch}", channel=ch, **kw)
        if resp.code != MSG:
            raise PtcError(f"L answered code {resp.code}: {resp.text!r}")
        return LinkStatus.parse(resp.text)

    # -- transport ---------------------------------------------------------

    def _reaction(self, timeout: float) -> Response | None:
        """Wait out one modem reaction, or None for a NACK condition (10.9.3)."""
        nacks = self.deframer.nacks
        deadline = time.monotonic() + timeout
        while True:
            chunk = self._read_some()
            if chunk:
                # Exactly one reaction follows one action (10.9.2), so a second
                # response here would be the modem breaking the protocol.
                for resp in self.deframer.feed(chunk):
                    return resp
                if self.deframer.nacks != nacks:
                    return None
                # "Timeout watchdog is stopped as soon as a packet-header is
                # received" (10.9.3): a frame that has begun gets its own window.
                if self.deframer.partial:
                    deadline = time.monotonic() + timeout
            elif time.monotonic() >= deadline:
                return None
            else:
                time.sleep(0.001)

    def _write(self, frame: bytes) -> None:
        self.stream.write(frame)
        self._flush()

    def _flush(self) -> None:
        flush = getattr(self.stream, "flush", None)
        if flush is not None:
            flush()

    def _read_quiet(self, timeout: float, quiet: float = 0.05) -> bytes:
        """Read until the stream stops producing for `quiet`, or `timeout`."""
        end = time.monotonic() + timeout
        out = bytearray()
        last = time.monotonic()
        while time.monotonic() < end:
            chunk = self._read_some()
            if chunk:
                out += chunk
                last = time.monotonic()
            elif out and time.monotonic() - last >= quiet:
                break
            else:
                time.sleep(0.005)
        return bytes(out)

    def _read_some(self) -> bytes:
        try:
            return self.stream.read(4096) or b""
        except (BlockingIOError, InterruptedError):
            return b""
        except OSError:
            return b""

    def close(self) -> None:
        try:
            self.stream.close()
        except OSError:
            pass
