# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP's integrity primitives: the nonstandard frame CRC-16, the session-ID
CRC-8, and the frame-type parity symbol.

All three are reimplemented from their functional description in
`docs/protocols/ardop/11-WAVEFORM.md` §2.2, §6 and validated byte-exact against the reference
vectors (`tests/besra/test_primitives.py`). The CRC-16 is deliberately *not* the
standard CRC-16/CCITT: despite the source's `x^16+x^12+x^5+1` comment the
polynomial constant is `0x8810` and the data bit is injected into the register
LSB *inside* the shift-and-divide step, which no table-standard CRC reproduces.
"""

from __future__ import annotations


def crc16(data: bytes) -> int:
    """The ARDOP frame CRC-16: poly constant 0x8810, init 0xFFFF, MSB-first per
    byte, data bit shifted into the register LSB before the polynomial XOR. No
    reflection, no final XOR. The MSB test is on the register *before* the shift."""
    reg = 0xFFFF
    for byte in data:
        mask = 0x80
        for _ in range(8):
            bit = 1 if (byte & mask) else 0
            mask >>= 1
            msb_set = reg & 0x8000
            reg = ((reg << 1) | bit) & 0xFFFF
            if msb_set:
                reg ^= 0x8810
    return reg


def crc8(data: bytes) -> int:
    """The session-ID CRC-8: poly constant 0xC6, init 0xFF, MSB-first, the same
    LSB-injection ordering as `crc16`."""
    reg = 0xFF
    for byte in data:
        for i in range(7, -1, -1):
            bit = (byte >> i) & 1
            msb_set = reg & 0x80
            reg = ((reg << 1) | bit) & 0xFF
            if msb_set:
                reg ^= 0xC6
    return reg


def session_id(caller: str, target: str) -> int:
    """8-bit session ID = CRC-8 of the ASCII concatenation ``caller+target``
    (canonical CALL / CALL-SSID forms); a result of 0xFF is remapped to 0x00
    (0xFF is reserved for unconnected / FEC / ConReq / Ping frames)."""
    sid = crc8((caller + target).encode("ascii"))
    return 0 if sid == 0xFF else sid


def type_parity(frame_type: int) -> int:
    """The 2-bit 4FSK parity symbol for a frame-type byte: 1 XOR the four 2-bit
    symbols of the byte (MSB pair first). Both header parity symbols carry this
    same value, computed only from the raw frame type."""
    parity = 1
    mask = 0xC0
    for k in range(4):
        parity ^= (frame_type & mask) >> (2 * (3 - k))
        mask >>= 2
    return parity & 0x3


def append_crc16_frametype(data: bytes, frame_type: int) -> bytes:
    """Append the frame CRC-16 the way ARDOP stores it: high byte verbatim, then
    low byte XORed with the frame-type byte (binding the check to its frame)."""
    crc = crc16(data)
    return data + bytes([crc >> 8, (crc & 0xFF) ^ frame_type])


def check_crc16_frametype(block: bytes, frame_type: int) -> bool:
    """True iff the last two bytes of ``block`` are a valid frame-type-bound
    CRC-16 over the preceding bytes."""
    crc = crc16(block[:-2])
    return (crc >> 8) == block[-2] and ((crc & 0xFF) ^ frame_type) == block[-1]
