# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ARDOP frame catalog and byte-domain frame assembly.

This is the seam between the byte world and the sample world: it turns a frame
type + payload into the exact byte layout the modulator renders, and reads that
layout back on receive. No DSP here — see `besra.phy`.

The catalog (`FRAMES`) and the per-carrier block geometry are the code-authoritative
tables from `docs/protocols/ardop/11-WAVEFORM.md` §2.3, §3, §4.2, §5.2. Every data-frame carrier
block is ``count(1) ‖ data(k) ‖ CRC16(2) ‖ RS(r)``; ConReq/ID/Ping instead carry
``12 data ‖ 4 RS`` with no CRC (RS replaces it).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .. import crc
from ..fec import rs


class Mod(Enum):
    FSK4 = "4FSK"
    PSK4 = "4PSK"
    PSK8 = "8PSK"
    QAM16 = "16QAM"


@dataclass(frozen=True, slots=True)
class FrameDef:
    """One frame type. For control frames ``k``/``r`` describe the fixed payload;
    ``carriers``/``mod``/``baud`` describe the waveform. ``forces_session`` marks
    the frames that pin Session ID to 0xFF (unconnected/FEC/ConReq/Ping)."""
    type: int
    name: str
    carriers: int
    mod: Mod
    baud: int
    k: int                    # data bytes per carrier (control: total payload bytes)
    r: int                    # RS parity bytes per carrier
    has_crc: bool = True      # ConReq/ID/Ping carry RS instead of a CRC
    forces_session: bool = False

    @property
    def net_payload(self) -> int:
        return self.k * self.carriers


# -- the data-frame ladder (spec §4.2, code-authoritative) -------------------
# (type_even, name, carriers, mod, baud, k, r) — the odd twin is type_even+1.

_DATA = [
    (0x48, "4FSK.200.50S", 1, Mod.FSK4, 50, 16, 4),
    (0x42, "4PSK.200.100S", 1, Mod.PSK4, 100, 16, 8),
    (0x40, "4PSK.200.100", 1, Mod.PSK4, 100, 64, 32),
    (0x44, "8PSK.200.100", 1, Mod.PSK8, 100, 108, 36),
    (0x46, "16QAM.200.100", 1, Mod.QAM16, 100, 128, 64),
    (0x4C, "4FSK.500.100S", 1, Mod.FSK4, 100, 32, 8),
    (0x4A, "4FSK.500.100", 1, Mod.FSK4, 100, 64, 16),
    (0x50, "4PSK.500.100", 2, Mod.PSK4, 100, 64, 32),
    (0x52, "8PSK.500.100", 2, Mod.PSK8, 100, 108, 36),
    (0x54, "16QAM.500.100", 2, Mod.QAM16, 100, 128, 64),
    (0x60, "4PSK.1000.100", 4, Mod.PSK4, 100, 64, 32),
    (0x62, "8PSK.1000.100", 4, Mod.PSK8, 100, 108, 36),
    (0x64, "16QAM.1000.100", 4, Mod.QAM16, 100, 128, 64),
    (0x70, "4PSK.2000.100", 8, Mod.PSK4, 100, 64, 32),
    (0x72, "8PSK.2000.100", 8, Mod.PSK8, 100, 108, 36),
    (0x74, "16QAM.2000.100", 8, Mod.QAM16, 100, 128, 64),
    (0x7A, "4FSK.2000.600", 1, Mod.FSK4, 600, 600, 150),
    (0x7C, "4FSK.2000.600S", 1, Mod.FSK4, 600, 200, 50),
]

# -- control / connect / ID frames (spec §2.3, §3) — all 4FSK 50 baud, 1 car --
# (type, name, payload_bytes, rs, has_crc, forces_session)

_CONTROL = [
    (0x23, "BREAK", 0, 0, False, False),
    (0x24, "IDLE", 0, 0, False, False),
    (0x29, "DISC", 0, 0, False, False),
    (0x2C, "END", 0, 0, False, False),
    (0x2D, "ConRejBusy", 0, 0, False, False),
    (0x2E, "ConRejBW", 0, 0, False, False),
    (0x30, "IDFrame", 12, 4, False, True),
    (0x31, "ConReq200M", 12, 4, False, True),
    (0x32, "ConReq500M", 12, 4, False, True),
    (0x33, "ConReq1000M", 12, 4, False, True),
    (0x34, "ConReq2000M", 12, 4, False, True),
    (0x35, "ConReq200F", 12, 4, False, True),
    (0x36, "ConReq500F", 12, 4, False, True),
    (0x37, "ConReq1000F", 12, 4, False, True),
    (0x38, "ConReq2000F", 12, 4, False, True),
    # ConAck/PingAck carry the real (pending) session so the peer can verify it;
    # only the unconnected ConReq/Ping/ID force 0xFF on the wire (spec §2.1).
    (0x39, "ConAck200", 3, 0, False, False),
    (0x3A, "ConAck500", 3, 0, False, False),
    (0x3B, "ConAck1000", 3, 0, False, False),
    (0x3C, "ConAck2000", 3, 0, False, False),
    (0x3D, "PingAck", 3, 0, False, False),
    (0x3E, "Ping", 12, 4, False, True),
]


def _build_catalog() -> dict[int, FrameDef]:
    cat: dict[int, FrameDef] = {}
    for even, name, cars, mod, baud, k, r in _DATA:
        cat[even] = FrameDef(even, f"{name}.E", cars, mod, baud, k, r)
        cat[even + 1] = FrameDef(even + 1, f"{name}.O", cars, mod, baud, k, r)
    for t, name, k, r, has_crc, forces in _CONTROL:
        cat[t] = FrameDef(t, name, 1, Mod.FSK4, 50, k, r,
                          has_crc=has_crc, forces_session=forces)
    # DATANAK 0x00–0x1F and DATAACK 0xE0–0xFF: 32-code quality blocks, 0 payload.
    for t in range(0x00, 0x20):
        cat[t] = FrameDef(t, "DATANAK", 1, Mod.FSK4, 50, 0, 0)
    for t in range(0xE0, 0x100):
        cat[t] = FrameDef(t, "DATAACK", 1, Mod.FSK4, 50, 0, 0)
    return cat


FRAMES: dict[int, FrameDef] = _build_catalog()


# -- frame-type header (spec §2.1–2.2) ---------------------------------------

def header_symbols(frame_type: int, session_id: int) -> list[int]:
    """The 10 4FSK symbols (2-bit each) of the frame-type header:
    ``[type as 4 symbols][parity][type⊕session as 4 symbols][parity]``, where
    both parity symbols are ``crc.type_parity(frame_type)`` (from the raw type)."""
    parity = crc.type_parity(frame_type)
    xored = frame_type ^ (session_id & 0xFF)
    return _byte_symbols(frame_type) + [parity] + _byte_symbols(xored) + [parity]


def _byte_symbols(b: int) -> list[int]:
    """A byte as four 2-bit symbols, MSB pair first."""
    return [(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3]


def decode_header(symbols: list[int]) -> tuple[int, int] | None:
    """Recover ``(frame_type, session_id)`` from 10 received header symbols, or
    None if the type fails its parity. Both header parity symbols carry the same
    value (from the raw type), so both must agree — using the redundancy the
    header provides rather than trusting a single copy."""
    type_byte = _symbols_byte(symbols[0:4])
    parity = crc.type_parity(type_byte)
    if parity != symbols[4] or parity != symbols[9]:
        return None
    session = type_byte ^ _symbols_byte(symbols[5:9])
    return type_byte, session


def _symbols_byte(symbols: list[int]) -> int:
    b = 0
    for s in symbols:
        b = (b << 2) | (s & 3)
    return b


# -- data-frame byte assembly (spec §4.2, §5.2, §6) --------------------------

def carrier_block(data: bytes, k: int, r: int, frame_type: int,
                  with_crc: bool = True) -> bytes:
    """One carrier's on-air block. With CRC (data frames):
    ``count(1) ‖ data(k, zero-filled) ‖ CRC16(2, frame-type-bound) ‖ RS(r)``.
    Without CRC (ConReq/ID/Ping): ``data(k) ‖ RS(r)`` — RS covers the raw data."""
    if with_crc:
        payload = data[:k].ljust(k, b"\x00")
        body = bytes([len(data[:k])]) + payload
        body = crc.append_crc16_frametype(body, frame_type)
    else:
        body = data[:k].ljust(k, b"\x00")
    return body + rs.rs_parity(body, r)


#: 600-baud FM full frames carry three sequential sub-packets, each with its own
#: count/data/CRC/RS, rather than one oversized RS block (spec §4.2). The short
#: 600-baud frame (0x7C/0x7D) stays a single block.
_SUBPACKET_600 = {0x7A, 0x7B}
_SUB_K, _SUB_R = 200, 50


def build_data_frame(frame_type: int, payload: bytes,
                     session_id: int) -> tuple[list[int], list[bytes]]:
    """Byte-domain assembly of a data frame: the header symbols and the list of
    on-air blocks — one per carrier, or three sub-packets for a 600-baud full
    frame (whose k=600/r=150 would otherwise exceed RS(255))."""
    fd = FRAMES[frame_type]
    header = header_symbols(frame_type, session_id)
    if frame_type in _SUBPACKET_600:
        blocks = [carrier_block(payload[i * _SUB_K:(i + 1) * _SUB_K],
                                _SUB_K, _SUB_R, frame_type, with_crc=fd.has_crc)
                  for i in range(3)]
    else:
        blocks = [carrier_block(payload[c * fd.k:(c + 1) * fd.k], fd.k, fd.r,
                                frame_type, with_crc=fd.has_crc)
                  for c in range(fd.carriers)]
    return header, blocks
