# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CWP — the creance wire protocol, riding the modem data channel only.

Frame: b"CRN1" | type u8 | length u16 BE (<=4096) | payload |
crc32(type + length + payload) u32 BE. Control bodies are JSON; unknown keys
are tolerated (forward compatibility). A major break bumps the magic.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field

MAGIC = b"CRN1"
MAX_PAYLOAD = 4096
V = 1

HELLO = 0x01
HELLO_ACK = 0x02
DATA = 0x10
END = 0x11
REPORT = 0x12

TYPE_NAMES = {HELLO: "HELLO", HELLO_ACK: "HELLO_ACK", DATA: "DATA",
              END: "END", REPORT: "REPORT"}

_HEADER = struct.Struct(">BH")          # type, length (after the magic)
_CRC = struct.Struct(">I")
_HDR_LEN = len(MAGIC) + _HEADER.size    # 7
_TRAILER_LEN = _CRC.size                # 4


def pack(ftype: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload {len(payload)} exceeds {MAX_PAYLOAD}")
    body = _HEADER.pack(ftype, len(payload)) + payload
    return MAGIC + body + _CRC.pack(zlib.crc32(body))


@dataclass(frozen=True, slots=True)
class Frame:
    type: int
    payload: bytes

    @property
    def name(self) -> str:
        return TYPE_NAMES.get(self.type, f"0x{self.type:02x}")


class NotCwp(Exception):
    """The stream never was CWP: bad magic before any frame parsed.
    .buffered holds every byte consumed so far — the sink fallback counts it."""

    def __init__(self, buffered: bytes) -> None:
        super().__init__(f"not a CWP stream ({len(buffered)} bytes buffered)")
        self.buffered = buffered


class Desync(Exception):
    """CWP framing broke mid-stream. The data channel is ARQ-reliable, so this
    is a real defect: record .reason/.evidence and degrade to sink — no resync.
    .frames holds any frames parsed earlier in the same feed() call."""

    def __init__(self, reason: str, evidence: bytes) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence
        self.frames: list[Frame] = []


class Deframer:
    """Feed arbitrary byte chunks; get back complete frames.

    Incomplete tails wait for more bytes. Bad magic before the first frame
    raises NotCwp; any later framing/CRC damage raises Desync. After either,
    the deframer is dead and re-raises on every subsequent feed.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._frames = 0
        self._dead: Exception | None = None

    def feed(self, chunk: bytes) -> list[Frame]:
        if self._dead is not None:
            raise self._dead
        self._buf += chunk
        frames: list[Frame] = []
        try:
            while (frame := self._next()) is not None:
                frames.append(frame)
        except Desync as exc:
            exc.frames = frames
            self._dead = exc
            raise
        except NotCwp as exc:
            self._dead = exc
            raise
        return frames

    def _next(self) -> Frame | None:
        buf = self._buf
        head = bytes(buf[:len(MAGIC)])
        if head != MAGIC[:len(head)]:
            if self._frames == 0:
                raise NotCwp(bytes(buf))
            raise Desync("bad magic", bytes(buf[:16]))
        if len(buf) < _HDR_LEN:
            return None
        ftype, length = _HEADER.unpack_from(buf, len(MAGIC))
        if length > MAX_PAYLOAD:
            raise Desync(f"length {length} exceeds {MAX_PAYLOAD}", bytes(buf[:_HDR_LEN]))
        total = _HDR_LEN + length + _TRAILER_LEN
        if len(buf) < total:
            return None
        body = bytes(buf[len(MAGIC):_HDR_LEN + length])
        (crc,) = _CRC.unpack_from(buf, _HDR_LEN + length)
        if crc != zlib.crc32(body):
            raise Desync("crc mismatch", bytes(buf[:min(total, 32)]))
        del buf[:total]
        self._frames += 1
        return Frame(ftype, body[_HEADER.size:])


# -- control frames -----------------------------------------------------------

def _json_body(frame: Frame, expected_type: int) -> dict:
    if frame.type != expected_type:
        raise ValueError(f"expected {TYPE_NAMES[expected_type]}, got {frame.name}")
    try:
        obj = json.loads(frame.payload)
    except ValueError:
        raise Desync(f"bad JSON in {frame.name}", frame.payload[:64]) from None
    if not isinstance(obj, dict):
        raise Desync(f"non-object JSON in {frame.name}", frame.payload[:64])
    return obj


def _pack_json(ftype: int, obj: dict) -> bytes:
    return pack(ftype, json.dumps(obj, separators=(",", ":")).encode())


@dataclass(frozen=True, slots=True)
class Hello:
    sid: str
    call: str
    scenario: str
    params: dict = field(default_factory=dict)
    v: int = V

    def pack(self) -> bytes:
        return _pack_json(HELLO, {"v": self.v, "sid": self.sid, "call": self.call,
                                  "scenario": self.scenario, "params": self.params})

    @classmethod
    def parse(cls, frame: Frame) -> "Hello":
        obj = _json_body(frame, HELLO)
        return cls(sid=str(obj.get("sid", "")), call=str(obj.get("call", "")),
                   scenario=str(obj.get("scenario", "")),
                   params=dict(obj.get("params") or {}), v=int(obj.get("v", V)))


@dataclass(frozen=True, slots=True)
class HelloAck:
    accept: bool
    caps: dict = field(default_factory=dict)
    v: int = V

    def pack(self) -> bytes:
        return _pack_json(HELLO_ACK, {"v": self.v, "accept": self.accept,
                                      "caps": self.caps})

    @classmethod
    def parse(cls, frame: Frame) -> "HelloAck":
        obj = _json_body(frame, HELLO_ACK)
        return cls(accept=bool(obj.get("accept")),
                   caps=dict(obj.get("caps") or {}), v=int(obj.get("v", V)))


@dataclass(frozen=True, slots=True)
class End:
    sha256: str
    bytes: int
    dur_s: float

    def pack(self) -> bytes:
        return _pack_json(END, {"sha256": self.sha256, "bytes": self.bytes,
                                "dur_s": self.dur_s})

    @classmethod
    def parse(cls, frame: Frame) -> "End":
        obj = _json_body(frame, END)
        return cls(sha256=str(obj.get("sha256", "")), bytes=int(obj.get("bytes", 0)),
                   dur_s=float(obj.get("dur_s", 0.0)))


@dataclass(frozen=True, slots=True)
class Report:
    """Far-end measurements: durations and counters only — no clock sync."""

    sid: str
    bytes: int
    sha256: str
    dur_s: float

    def pack(self) -> bytes:
        return _pack_json(REPORT, {"sid": self.sid, "bytes": self.bytes,
                                   "sha256": self.sha256, "dur_s": self.dur_s})

    @classmethod
    def parse(cls, frame: Frame) -> "Report":
        obj = _json_body(frame, REPORT)
        return cls(sid=str(obj.get("sid", "")), bytes=int(obj.get("bytes", 0)),
                   sha256=str(obj.get("sha256", "")), dur_s=float(obj.get("dur_s", 0.0)))
