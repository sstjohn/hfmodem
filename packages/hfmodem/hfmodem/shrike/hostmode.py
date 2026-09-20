# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""WA8DED hostmode and the SCS CRC-hostmode framing that wraps it.

This is the wire between a host program (Winlink Express, Airmail, Pat/ptc-go,
GP) and an SCS PTC. It is a documented protocol, not a reverse-engineered one:
everything here is from the SCS PTC-IIIusb manual v4.1 chapter 10 (10.5 extended
hostmode, 10.6 status channel, 10.9 CRC hostmode) and the WA8DED host mode
user's guide, cross-checked against a working client (harenber/ptc-go, the
PACTOR driver Pat uses).

Framing, from the outside in:

  CRC layer (10.9)    #170 #170 | stuffed( WA8DED packet + CRC16 lo,hi )
                      Stuffing inserts #0 after every #170 *after* the header.
                      The CRC is CCITT-CRC16 as used by AX.25 -- CRC-16/X-25 --
                      over the unstuffed packet, appended low byte first. The
                      manual's worked example (bytes 04 01 01 71 71 -> low 213,
                      high 153) reproduces exactly with shrike.coding.crc16.
                      A modem that sees a bad CRC answers with the four-byte
                      request packet #170#170#170#85 to ask for a repeat.
                      JHOST1 is the same hostmode without this layer (10.4.6).

  WA8DED packet       host -> modem:  channel, flags, len-1, data
                      modem -> host:  channel, code, [len-1,] data
                      flags bit0 = command (1) or data (0), bit6 = ignore the
                      packet counter, bit7 = the master's 1-bit packet counter.
                      The response code decides the modem->host body shape:
                      0 nothing, 1-5 a NUL-terminated string, 6-7 a counted
                      block. So a frame's length is only knowable by parsing it,
                      which is why the decoder is direction-aware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from .coding import crc16

HEADER = b"\xaa\xaa"
REQUEST = b"\xaa\xaa\xaa\x55"

# modem -> host response codes [manual 10.6 / WA8DED guide]
OK = 0           # success, nothing follows
MSG = 1          # success, NUL-terminated message
FAIL = 2         # failure, NUL-terminated message
LINK = 3         # link status event, NUL-terminated
MON_HEADER = 4   # monitor header, no info field
MON_HEADER_INFO = 5
MON_INFO = 6     # counted
DATA = 7         # connected information, counted

MAX_DATA = 256   # "maximum useable data length ... must not exceed 256" [10.9.3]


@dataclass(frozen=True)
class Packet:
    """One host -> modem WA8DED packet."""
    channel: int
    is_command: bool
    counter: int
    ignore_counter: bool
    data: bytes

    @property
    def text(self) -> str:
        return self.data.decode("latin-1")


@dataclass(frozen=True)
class Response:
    """One modem -> host WA8DED packet."""
    channel: int
    code: int
    data: bytes


def wrap(payload: bytes) -> bytes:
    """CRC-hostmode envelope: header, then stuffed payload+CRC."""
    crc = crc16(payload)
    return HEADER + (payload + bytes((crc & 0xFF, crc >> 8))).replace(b"\xaa", b"\xaa\x00")


def unwrap(frame: bytes) -> bytes:
    """The bare WA8DED packet inside a CRC-hostmode envelope.

    Plain hostmode (JHOST1) carries the same packet with no envelope at all, so
    a modem serving it sends what this returns.
    """
    return frame[len(HEADER):].replace(b"\xaa\x00", b"\xaa")[:-2]


def command(channel: int, text: str, *, counter: int = 0,
            ignore_counter: bool = False) -> bytes:
    return _to_modem(channel, 1 | (0x40 if ignore_counter else 0) | (counter << 7),
                     text.encode("latin-1"))


def data(channel: int, blob: bytes, *, counter: int = 0,
         ignore_counter: bool = False) -> bytes:
    return _to_modem(channel, (0x40 if ignore_counter else 0) | (counter << 7), blob)


def _to_modem(channel: int, flags: int, blob: bytes) -> bytes:
    if not 1 <= len(blob) <= MAX_DATA:
        raise ValueError(f"packet body must be 1..{MAX_DATA} bytes, got {len(blob)}")
    return wrap(bytes((channel, flags, len(blob) - 1)) + blob)


def reply(channel: int, code: int, body: bytes = b"") -> bytes:
    """Build a modem -> host packet, shaped by its code."""
    if code == OK:
        return wrap(bytes((channel, code)))
    if code in (MON_INFO, DATA):
        if not 1 <= len(body) <= MAX_DATA:
            raise ValueError(f"counted body must be 1..{MAX_DATA} bytes, got {len(body)}")
        return wrap(bytes((channel, code, len(body) - 1)) + body)
    # A NUL ends a string body, so one inside it would truncate the frame and the
    # CRC would then fail at the master; over-length does the same to the packet
    # counter's 256. Neither may abort the reply: 10.9.2 gives the modem exactly
    # one reaction per master action, and a raise here would send nothing at all.
    return wrap(bytes((channel, code)) + body.replace(b"\x00", b"")[:MAX_DATA] + b"\x00")


def reply_text(channel: int, code: int, text: str) -> bytes:
    return reply(channel, code, text.encode("latin-1"))


# --------------------------------------------------------------------------- #
# Decoding

CRC_ERROR = "crc"
REQUEST_EVENT = "request"


class Decoder:
    """Byte-stream decoder for one direction of the CRC hostmode.

    ``role="modem"`` parses host -> modem packets, ``role="master"`` parses the
    modem's replies. Yields ``(Packet|Response)``, or the ``CRC_ERROR`` /
    ``REQUEST_EVENT`` sentinels.

    ``crc=False`` is plain WA8DED hostmode (JHOST1, 10.4.6): the same packets
    with no header, no stuffing and no CRC, so every byte is packet.
    """

    def __init__(self, role: str = "modem", *, crc: bool = True):
        if role not in ("modem", "master"):
            raise ValueError("role must be 'modem' or 'master'")
        self.role = role
        self.crc = crc
        self._buf = bytearray()
        self._in_packet = False
        self._prev_aa = False
        self._pending_aa = False

    def feed(self, chunk: bytes) -> Iterator:
        for b in chunk:
            yield from self._byte(b)

    def _byte(self, b: int):
        if not self.crc:
            self._buf.append(b)
            total = self._packet_len()
            if total is not None and len(self._buf) >= total:
                frame, self._buf[:] = bytes(self._buf[:total]), self._buf[total:]
                yield self._parse(frame)
            return

        if not self._in_packet:
            if self._prev_aa and b == 0xAA:
                self._start()
            else:
                self._prev_aa = b == 0xAA
            return

        if self._pending_aa:
            self._pending_aa = False
            if b == 0x00:
                self._buf.append(0xAA)
            elif b == 0xAA:
                self._start()          # #170#170 always restarts a packet [10.9.5]
                return
            elif b == 0x55 and not self._buf:
                self._reset()
                yield REQUEST_EVENT
                return
            else:
                self._reset()          # stuffing error -> resynchronise [10.9.5]
                return
        elif b == 0xAA:
            self._pending_aa = True
            return
        else:
            self._buf.append(b)

        total = self._packet_len()
        if total is not None and len(self._buf) >= total + 2:
            frame, self._buf[:] = bytes(self._buf[:total + 2]), self._buf[total + 2:]
            self._reset()
            body, got = frame[:-2], int.from_bytes(frame[-2:], "little")
            yield self._parse(body) if crc16(body) == got else CRC_ERROR

    def _start(self) -> None:
        self._in_packet, self._prev_aa, self._pending_aa = True, False, False
        self._buf.clear()

    def _reset(self) -> None:
        self._in_packet, self._prev_aa, self._pending_aa = False, False, False
        self._buf.clear()

    def _packet_len(self) -> Optional[int]:
        """Length of the WA8DED packet in the buffer, or None if not yet known."""
        buf = self._buf
        if self.role == "modem":
            return 3 + buf[2] + 1 if len(buf) >= 3 else None
        if len(buf) < 2:
            return None
        code = buf[1]
        if code == OK:
            return 2
        if code in (MON_INFO, DATA):
            return 3 + buf[2] + 1 if len(buf) >= 3 else None
        nul = buf.find(0, 2)
        return nul + 1 if nul >= 0 else None

    def _parse(self, body: bytes):
        if self.role == "master":
            return Response(body[0], body[1], body[3:] if body[1] in (MON_INFO, DATA)
                            else body[2:-1])
        flags = body[1]
        return Packet(channel=body[0], is_command=bool(flags & 1),
                      counter=(flags >> 7) & 1, ignore_counter=bool(flags & 0x40),
                      data=body[3:])


# --------------------------------------------------------------------------- #
# The modem half of the host/modem packet protocol [manual 10.9.4]

class Modem:
    """Drives ``handler(Packet) -> response bytes`` under the CRC-hostmode rules.

    The host is the master and every action of its gets exactly one reaction.
    A bad CRC draws the request packet; a repeated packet counter means the host
    never saw our last reply, so we repeat it verbatim and throw the packet away.

    One deliberate departure from 10.9.4: a repeated counter only counts as a
    repeat if the packet bytes are *identical* too. ptc-go writes data packets
    fire-and-forget without advancing its counter, and a strict reading would
    silently drop them; a genuine retransmission is byte-identical, so this rule
    is a strict superset of the documented one.

    ``crc=False`` serves plain hostmode (JHOST1): the same packets and the same
    rules, without the CRC envelope -- and so without the request packet, which
    only a CRC error can raise.
    """

    def __init__(self, handler: Callable[[Packet], bytes], *, crc: bool = True):
        self.handler = handler
        self.reset(crc=crc)

    def reset(self, *, crc: bool = True) -> None:
        """Start (or restart) hostmode.

        10.9.6: at a hostmode start the modem's REQUEST bit is undefined, so
        either counter value in the master's first packet is an ACK-condition.
        Without this a client that left with JHOST0 and came back is answered
        from the previous session's buffer, and its first command is never run.
        """
        self.crc = crc
        self._decoder = Decoder("modem", crc=crc)
        self._last: Optional[tuple[int, bytes]] = None   # counter, raw packet body
        self._buffered = b""

    def feed(self, chunk: bytes) -> bytes:
        out = bytearray()
        for event in self._decoder.feed(chunk):
            if event == CRC_ERROR:
                out += REQUEST
            elif event == REQUEST_EVENT:
                out += self._buffered          # master asks for a repeat
            else:
                reply_frame = self._accept(event)
                out += reply_frame if self.crc else unwrap(reply_frame)
        return bytes(out)

    def _accept(self, pkt: Packet) -> bytes:
        raw = bytes((pkt.channel, int(pkt.is_command))) + pkt.data
        if not pkt.ignore_counter and self._last == (pkt.counter, raw):
            return self._buffered
        self._last = (pkt.counter, raw)
        self._buffered = self.handler(pkt)
        return self._buffered


def _selftest() -> None:
    """The framing layer against the manual's own examples and its own inverse.

    Collected by `tests/shrike/test_hostmode.py`; the module name still runs it
    directly. It lived under ``__main__`` alone, which pytest never collects --
    the same blind spot that let arq's loopback regress unseen.
    """
    # The manual's own CRC example [10.9.7]: bytes 04 01 01 71 71 -> low 213, high 153.
    c = crc16(bytes([4, 1, 1, 71, 71]))
    assert (c & 0xFF, c >> 8) == (213, 153), hex(c)
    print(f"manual CRC example: low={c & 0xFF} high={c >> 8}  OK")

    # Round trip through both decoders, including a stuffed #170 in the payload.
    payload = b"\xaa" * 3 + b"hello"
    modem = Decoder("modem")
    (pkt,) = list(modem.feed(data(31, payload, counter=1)))
    assert pkt == Packet(31, False, 1, False, payload), pkt
    print(f"host->modem round trip, {len(payload)} bytes with stuffing  OK")

    host = Decoder("master")
    frames = reply(31, OK) + reply_text(31, LINK, "(31) CONNECTED to N0DX") + \
        reply(31, DATA, b"\xaa\xaa payload")
    got = list(host.feed(frames))
    assert got == [Response(31, OK, b""),
                   Response(31, LINK, b"(31) CONNECTED to N0DX"),
                   Response(31, DATA, b"\xaa\xaa payload")], got
    print("modem->host round trip, all three body shapes  OK")

    # Corruption draws a request packet; the request draws the buffered repeat.
    tnc = Modem(lambda p: reply_text(p.channel, MSG, "ack"))
    frame = bytearray(command(31, "L", counter=0))
    good = tnc.feed(bytes(frame))
    frame[-1] ^= 0xFF
    assert tnc.feed(bytes(frame)) == REQUEST
    assert tnc.feed(REQUEST) == good
    print("CRC error -> request packet, request -> repeat  OK")

    # Byte-at-a-time feeding must give the identical result (a real serial port).
    d = Decoder("modem")
    stream = command(31, "C N0DX") + data(31, b"xyz", counter=1)
    assert [e for b in stream for e in d.feed(bytes([b]))] == \
        list(Decoder("modem").feed(stream))
    print("byte-at-a-time == whole-buffer decode  OK")

    print("ALL PASS")


if __name__ == "__main__":
    _selftest()
