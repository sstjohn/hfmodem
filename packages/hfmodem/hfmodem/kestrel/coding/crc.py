# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The frame check used everywhere in kestrel: CRC-16/GENIBUS [spec 03 §3.4].

Every framed byte string in the modem — BW500 and BW2300 data frames, the
handshake callsign hash, the link-setup body — is protected by this one CRC.
``GENIBUS`` names the parameter set in the usual Rocksoft terms so a generic
implementation can be pointed at the same numbers and checked against
``crc16_genibus`` (``tests/kestrel/test_crc.py``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CRCSpec:
    """Rocksoft parameter record: what a generic CRC engine needs to be this one."""
    name: str
    width: int
    poly: int
    init: int
    refin: bool
    refout: bool
    xorout: int
    check: int          # CRC of b"123456789", the catalogue's identifying vector


GENIBUS = CRCSpec("CRC-16/GENIBUS", 16, 0x1021, 0xFFFF, False, False, 0xFFFF, 0xD64E)


def crc16_genibus(data: bytes) -> int:
    """CRC-16/GENIBUS over ``data``, MSB-first with no reflection."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc ^ 0xFFFF
