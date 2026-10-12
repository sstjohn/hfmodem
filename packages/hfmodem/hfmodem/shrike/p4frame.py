# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P4 net packet framing, before channel coding and after channel decoding.

Status is E/S1S0/M2M1M0/C1C0 (SCS P4 section 5), not P3's independent
BK/QRT/long bits. Whitening facts: docs/protocols/pactor/pactor4-framing.md.
This module neither infers header meanings nor delivers source-coded user data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .coding import crc16


class ProtocolStatus(IntEnum):
    NONE = 0
    CLOSE = 1
    CHANGEOVER = 2
    SPEED_UP = 3  # No recorded instance establishes this meaning yet.


@dataclass(frozen=True)
class Status:
    counter: int
    source_mode: int = 0
    protocol: ProtocolStatus = ProtocolStatus.NONE
    extended: bool = False

    def __post_init__(self):
        for name, value, maximum in (("counter", self.counter, 3),
                                     ("source_mode", self.source_mode, 7),
                                     ("protocol", self.protocol, 3)):
            if not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 0..{maximum}")
        if not isinstance(self.extended, bool):
            raise ValueError("extended must be a boolean")

    @property
    def byte(self) -> int:
        return (int(self.extended) << 7 | int(self.protocol) << 5
                | self.source_mode << 2 | self.counter)

    @classmethod
    def from_byte(cls, value: int) -> Status:
        if not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError("status must be a byte")
        return cls(value & 3, (value >> 2) & 7,
                   ProtocolStatus((value >> 5) & 3), bool(value & 128))


def whiten(data: bytes) -> bytes:
    """Packet-reset synchronous XOR, covering payload, status and CRC.

    The same operation dewhitens. The flush byte/channel-code tails are outside
    this transform. No input-dependent state or session-specific seed is used.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    register = 0x00B770CD
    result = bytearray()
    for value in data:
        result.append(value ^ (register & 255))
        for _ in range(8):
            register = (register >> 1) ^ (0x48000000 if register & 1 else 0)
    return bytes(result)


def encode_field(data: bytes, status: Status) -> bytes:
    """Frame an already source-encoded, exactly sized user field.

    Does not guess padding or encode extended-status prefixes. The caller owns
    the field length and source encoding; channel coding follows this function.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not isinstance(status, Status):
        raise TypeError("status must be Status")
    field = data + bytes([status.byte])
    return whiten(field + crc16(field).to_bytes(2, "little"))


@dataclass(frozen=True)
class Frame:
    data: bytes
    status: Status
    crc: int


def decode_field(field: bytes, *, data_bytes: int) -> Frame | None:
    """Validate a complete decoded field; return None on bad length or CRC.

    Data is the raw net field, including any extended-status prefix and source
    framing. A CRC pass alone is not permission to acknowledge or deliver it.
    """
    if not isinstance(field, bytes):
        raise TypeError("field must be bytes")
    if not isinstance(data_bytes, int) or data_bytes < 0:
        raise ValueError("data_bytes must be a nonnegative integer")
    if len(field) != data_bytes + 3:
        return None
    packet = whiten(field)
    received = int.from_bytes(packet[-2:], "little")
    if crc16(packet[:-2]) != received:
        return None
    return Frame(packet[:-3], Status.from_byte(packet[-3]), received)
