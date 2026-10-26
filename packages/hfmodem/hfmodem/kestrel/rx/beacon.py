# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The narrow KISS broadcast frame VarAC's advanced beacon rides.

Measured off air (179 frames from 26 stations, 2026-10-23) and on the stock bench: a 32-symbol MFSK header on the CQ grid,
then one BW500 record-1 burst (host level 2) whose 22-byte payload is a type
byte, twenty 7-bit characters, pad, VARA's short-over trailer and the caller
CRC byte. Stock hands the twenty characters to its KISS port as a command-2
frame, and keys the same frame for one; what the characters say is VarAC's
business (contact.varac).
"""
from dataclasses import dataclass

import numpy as np

from . import varahf500 as _rx

#: Header tones on the CQ grid (48000/2048 Hz), as stock keyed them on the bench; the
#: strongest station on tape, KG9B, sits one bin (23 Hz) above them.
HEADER = np.array([74, 70, 76, 66, 70, 72, 76, 72, 72, 72, 62, 56, 54, 60, 58, 56, 58, 50,
                   72, 64, 66, 60, 68, 62, 68, 56, 58, 68, 64, 52, 72, 50], dtype=int)
LEVEL = 2
RECORD = _rx.INDEX_RECORDS[LEVEL]
#: The record's four training columns: the same bins from every station and from stock.
TRAINING = [32, 34, 31, 29]
TYPE = 0x21
CHARS = 20
TRAILER = 0x14


def pack7(text: str) -> bytes:
    """Twenty 7-bit characters behind the type byte, padded as VARA pads: with
    the complement of the last bit."""
    if len(text) > CHARS or not text.isascii() or not text.isprintable():
        raise ValueError('a beacon carries up to twenty printable ASCII characters')
    bits = f'{TYPE:08b}' + ''.join(f'{ord(c):07b}' for c in text.ljust(CHARS))
    bits += ('1' if bits[-1] == '0' else '0') * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, 'big')


def unpack7(payload: bytes) -> str:
    bits = ''.join(f'{b:08b}' for b in payload[1:1 + (CHARS * 7 + 7) // 8])
    return ''.join(chr(int(bits[i:i + 7], 2)) for i in range(0, CHARS * 7, 7))


def build_payload(text: str, caller_crc: int = 0) -> bytes:
    body = pack7(text) + bytes([TRAILER, caller_crc & 0xff])
    return body.ljust(RECORD.payload, b'\0')


@dataclass(frozen=True)
class BeaconResult:
    text: str
    sample: int = 0
    sn: float | None = 0.0


def parse_payload(payload: bytes, *, sample=0, sn=0.0) -> BeaconResult | None:
    if len(payload) < RECORD.payload or payload[0] != TYPE or payload[19] != TRAILER:
        return None
    text = unpack7(payload)
    if not text.isprintable() or not text.strip():
        return None
    return BeaconResult(text.rstrip(), sample, None if sn is None else float(sn))


def is_broadcast_frame(frame) -> bool:
    """An identified advanced-beacon or direct UI packet, not linked DATA.

    Keep generic physical decoding intact: callers use this at session admission.
    The bytes alone cannot distinguish deliberately identical application DATA
    from this broadcast record; the measured type, packing and field constrain
    the exclusion, rather than merely matching a payload prefix.
    """
    from .broadcast import is_broadcast_frame as is_ui_broadcast

    if is_ui_broadcast(frame):
        return True
    raw = bytes(frame.frame_bytes)
    if (not frame.crc_ok or frame.level != LEVEL or len(raw) != RECORD.payload + 3
            or raw[RECORD.payload] not in (0x00, 0x40, 0x80)):
        return False
    payload = raw[:RECORD.payload]
    parsed = parse_payload(payload)
    return parsed is not None and build_payload(parsed.text, payload[20]) == payload
