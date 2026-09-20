# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARQ frame (de)serialisation over the 43-byte proven data payload.

The *waveform* and the 43-byte payload / marker / CRC container are the on-air
ones (spec 03/04, kestrel rx/tx). The field layout in THIS module — the control
message types, the marker bit assignment, the length-prefixed data framing — is
**kestrel's own** design for the kestrel<->kestrel link, and is not the VARA HF
control-frame format: those control bytes ride a separate CONTROL waveform the
spec does not describe (spec 05 §5.7). So these frames link two kestrels; a
VARA HF peer's connect/ACK exchange is not what this module encodes.

Marker byte (the CRC-protected 44th frame byte, `FrameResult.marker`):
    * DATA frame:    bit7 = last-block-of-over flag, bits0..5 = sequence (mod 64).
                     (range 0x00..0xBF)
    * CONTROL frame: fixed sentinel ``CTRL_MARKER`` (0xF0) — disjoint from DATA.

Control payload[0] = control type; remaining bytes are per-type fields.
"""
from __future__ import annotations

from dataclasses import dataclass

CTRL_MARKER = 0xF0
SEQ_MOD = 64
LAST_OF_OVER = 0x80

# Control types (ours)
CR = 0x01   # connect-request           (initiator -> responder)
CA = 0x02   # connect-answer / response (responder -> initiator)
CF = 0x03   # connect-confirm / final   (initiator -> responder)
ACK = 0x10  # data ACK (acked seq)      (IRS -> ISS)
NAK = 0x11  # data NAK (expected seq)   (IRS -> ISS)
DR = 0x20   # disconnect-request
DC = 0x21   # disconnect-answer
DF = 0x22   # disconnect-final
KA = 0x30   # keepalive

_CTRL_NAMES = {CR: "CR", CA: "CA", CF: "CF", ACK: "ACK", NAK: "NAK",
               DR: "DR", DC: "DC", DF: "DF", KA: "KA"}

PAYLOAD = 43


# --------------------------------------------------------------------------- #
# DATA frames
def data_marker(seq: int, last_of_over: bool) -> int:
    return (seq % SEQ_MOD) | (LAST_OF_OVER if last_of_over else 0)

def is_control_marker(marker: int) -> bool:
    return marker == CTRL_MARKER

def data_seq(marker: int) -> int:
    return marker & (SEQ_MOD - 1)

def data_is_last(marker: int) -> bool:
    return bool(marker & LAST_OF_OVER)


def encode_data(block: bytes, seq: int, last_of_over: bool):
    """(data block, seq, last-of-over) -> (block payload, marker).

    Block size is bandwidth-dependent (43 B at BW500, 89 B at BW2300); the caller
    sizes and pads the block, so this only attaches the marker."""
    return block, data_marker(seq, last_of_over)


def build_frame_bytes(payload43: bytes, marker: int) -> bytes:
    """(payload43, marker) -> 46-byte proven frame (adds CRC-16/GENIBUS)."""
    from . import phy
    return phy.build_frame(payload43, marker)


# --------------------------------------------------------------------------- #
# CONTROL frames
@dataclass
class Control:
    ctype: int
    seq: int = 0            # ACK: acked seq; NAK: expected seq
    src: str = ""
    dst: str = ""
    bw: str = ""

    @property
    def name(self) -> str:
        return _CTRL_NAMES.get(self.ctype, f"?{self.ctype:02x}")


def encode_control(c: Control, size: int = PAYLOAD):
    """Control message -> (payload, marker=CTRL_MARKER). ``size`` = bandwidth block."""
    body = bytearray(size)
    body[0] = c.ctype & 0xFF
    body[1] = c.seq & 0xFF
    src = c.src.encode("ascii", "replace")[:10]
    dst = c.dst.encode("ascii", "replace")[:10]
    bw = c.bw.encode("ascii", "replace")[:5]
    body[2] = len(src)
    body[3] = len(dst)
    body[4] = len(bw)
    off = 5
    for blob in (src, dst, bw):
        body[off:off + len(blob)] = blob
        off += len(blob)
    return bytes(body), CTRL_MARKER


def decode_control(payload43: bytes) -> Control:
    b = payload43
    ctype, seq = b[0], b[1]
    ls, ld, lb = b[2], b[3], b[4]
    off = 5
    src = b[off:off + ls].decode("ascii", "replace"); off += ls
    dst = b[off:off + ld].decode("ascii", "replace"); off += ld
    bw = b[off:off + lb].decode("ascii", "replace"); off += lb
    return Control(ctype=ctype, seq=seq, src=src, dst=dst, bw=bw)


# --------------------------------------------------------------------------- #
# Decoded-frame classification (from a proven FrameResult)
@dataclass
class RxFrame:
    crc_ok: bool
    is_control: bool
    marker: int
    payload: bytes                 # 43 bytes
    control: Control | None = None

    @property
    def seq(self) -> int:
        return data_seq(self.marker)

    @property
    def last_of_over(self) -> bool:
        return data_is_last(self.marker)


def classify(frame_result) -> RxFrame:
    """Proven ``FrameResult`` -> :class:`RxFrame` (control decoded if applicable)."""
    marker = frame_result.marker
    payload = frame_result.payload
    ctrl = is_control_marker(marker)
    control = decode_control(payload) if (ctrl and frame_result.crc_ok) else None
    return RxFrame(crc_ok=frame_result.crc_ok, is_control=ctrl,
                   marker=marker, payload=payload, control=control)


# --------------------------------------------------------------------------- #
# Length-prefixed data framing (ours) — reassembly across 43-byte blocks.
#
# VARA signals total message length out-of-band (rx docstring); kestrel<->kestrel
# instead prefixes each host `transmit()` blob with a 4-byte big-endian length,
# concatenates, and chops into 43-byte blocks (last one zero-padded). The
# receiver parses length prefixes to deliver exactly the bytes written, ignoring
# trailing pad. This is our own design choice.
LEN_PREFIX = 4

def frame_outbound(blob: bytes) -> bytes:
    return len(blob).to_bytes(LEN_PREFIX, "big") + blob


class Reassembler:
    """Parses the length-prefixed block stream back into host deliveries."""

    def __init__(self):
        self._buf = bytearray()

    def feed(self, block43: bytes) -> list:
        """Add one decoded 43-byte data block; return list of completed blobs."""
        self._buf += block43
        out = []
        while len(self._buf) >= LEN_PREFIX:
            n = int.from_bytes(self._buf[:LEN_PREFIX], "big")
            if n == 0:
                # zero-length prefix == trailing pad from the final block; a real
                # empty write is indistinguishable and harmless — consume + stop.
                del self._buf[:LEN_PREFIX]
                if not any(self._buf):
                    self._buf.clear()
                    break
                continue
            if len(self._buf) < LEN_PREFIX + n:
                break
            blob = bytes(self._buf[LEN_PREFIX:LEN_PREFIX + n])
            out.append(blob)
            del self._buf[:LEN_PREFIX + n]
        return out
