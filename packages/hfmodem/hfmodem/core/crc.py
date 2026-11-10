# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CRC-16/0x1021 variants shared by modem frame codecs."""

from __future__ import annotations

import numpy as np

CRC_VARIANTS: dict[str, dict] = {
    "ccitt-false": dict(init=0xFFFF, refin=False, refout=False, xorout=0x0000),
    "kermit":      dict(init=0x0000, refin=True,  refout=True,  xorout=0x0000),
    "x25":         dict(init=0xFFFF, refin=True,  refout=True,  xorout=0xFFFF),
}


def _reflect(x: int, width: int) -> int:
    r = 0
    for _ in range(width):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def reflected_crc_ccitt_table() -> list[int]:
    """The 256-entry reflected CRC-16/CCITT table (poly reversed = 0x8408).

    Used for the CRC and data whitening. First entries: 0x0000, 0x1189, 0x2312, ...
    """
    tbl = []
    for b in range(256):
        crc = b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
        tbl.append(crc & 0xFFFF)
    return tbl


_CRC_TABLE_LSB = reflected_crc_ccitt_table()


def _msb_crc_ccitt_table() -> list[int]:
    """The un-reflected companion to the above, for the ccitt-false variant."""
    tbl = []
    for b in range(256):
        crc = b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 \
                else (crc << 1) & 0xFFFF
        tbl.append(crc)
    return tbl


_CRC_TABLE_MSB = _msb_crc_ccitt_table()


def crc16(data: bytes, variant: str = "x25") -> int:
    """CRC-16 with polynomial 0x1021, in the named variant.

    Default is CRC-16/X-25: init 0xFFFF, reflected in and out, xorout 0xFFFF. The
    variant is pinned by its residue, 0xF0B8 -- the value a correct message plus
    its own CRC leaves behind -- which separates it from the other 0x1021
    variants, so a receiver's verify step doubles as identification of the
    variant.
    """
    if variant not in CRC_VARIANTS:
        raise ValueError(f"unknown CRC variant {variant!r}; "
                         f"choose from {sorted(CRC_VARIANTS)}")
    p = CRC_VARIANTS[variant]
    # Byte-at-a-time, which is how PACTOR receivers do it -- the reflected
    # variants run on `reflected_crc_ccitt_table` above. The bitwise form this
    # replaces was 8x the work, and the P1 packet scan CRCs tens of thousands of
    # candidate frames per capture.
    if p["refin"]:                          # refin and refout always agree here
        reg = _reflect(p["init"], 16)
        for byte in data:
            reg = (reg >> 8) ^ _CRC_TABLE_LSB[(reg ^ byte) & 0xFF]
    else:
        reg = p["init"]
        for byte in data:
            reg = ((reg << 8) & 0xFFFF) ^ _CRC_TABLE_MSB[((reg >> 8) ^ byte) & 0xFF]
    return reg ^ p["xorout"]


def crc16_rows(data: np.ndarray, variant: str = "x25") -> np.ndarray:
    """`crc16` of every row of a (k, n) uint8 array, as a (k,) uint32 array.

    The register recurrence is sequential in the byte but independent across rows,
    so a scan of k candidate frames is n table lookups wide rather than k*n
    lookups long. The PACTOR-1 packet search gates tens of thousands of candidate
    alignments per capture, which is a call per frame the narrow way.
    """
    if variant not in CRC_VARIANTS:
        raise ValueError(f"unknown CRC variant {variant!r}; "
                         f"choose from {sorted(CRC_VARIANTS)}")
    p = CRC_VARIANTS[variant]
    cols = np.asarray(data, np.uint32).T
    if p["refin"]:
        reg = np.full(cols.shape[1], _reflect(p["init"], 16), np.uint32)
        for col in cols:
            reg = (reg >> 8) ^ _TABLE_LSB[(reg ^ col) & 0xFF]
    else:
        reg = np.full(cols.shape[1], p["init"], np.uint32)
        for col in cols:
            reg = ((reg << 8) & 0xFFFF) ^ _TABLE_MSB[((reg >> 8) ^ col) & 0xFF]
    return reg ^ p["xorout"]


_TABLE_LSB = np.array(_CRC_TABLE_LSB, np.uint32)


_TABLE_MSB = np.array(_CRC_TABLE_MSB, np.uint32)
