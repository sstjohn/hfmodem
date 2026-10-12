# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""AX.25 addresses and frames; public frame values exclude the FCS."""
from __future__ import annotations

from dataclasses import dataclass

from hfmodem.core.crc import crc16

CALLSIGN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "


def address(call: str, *, last: bool = False, flag: bool = False) -> bytes:
    name, sep, suffix = call.partition("-")
    if not 1 <= len(name) <= 6 or any(c not in CALLSIGN_CHARS[:-1] for c in name):
        raise ValueError("callsign must contain one to six uppercase letters or digits")
    try:
        ssid = int(suffix) if sep else 0
    except ValueError:
        raise ValueError("invalid SSID") from None
    if not 0 <= ssid <= 15:
        raise ValueError("SSID must be 0 to 15")
    return bytes(ord(c) << 1 for c in name.ljust(6)) + bytes([
        0x60 | ssid << 1 | int(flag) << 7 | int(last)])


def validate_address(raw: bytes) -> None:
    if len(raw) != 7:
        raise ValueError("an address must have seven bytes")
    if any(b & 1 or chr(b >> 1) not in CALLSIGN_CHARS for b in raw[:6]):
        raise ValueError("invalid shifted callsign character")


@dataclass(frozen=True)
class Frame:
    addresses: tuple[bytes, ...]
    control: int
    pid: int | None
    info: bytes

    def encode(self) -> bytes:
        needs_pid = self.control & 1 == 0 or self.control & 0xEF == 3
        if needs_pid != (self.pid is not None):
            raise ValueError("PID presence does not match control")
        raw = b"".join(self.addresses) + bytes([self.control])
        if self.pid is not None:
            raw += bytes([self.pid])
        raw += self.info
        parse(raw)
        return raw


def parse(raw: bytes) -> Frame:
    addresses = []
    offset = 0
    while True:
        a = raw[offset:offset + 7]
        validate_address(a)
        addresses.append(a)
        offset += 7
        if len(addresses) > 10:
            raise ValueError("a frame needs two to ten addresses")
        if a[6] & 1:
            break
    if len(addresses) < 2:
        raise ValueError("a frame needs two to ten addresses")
    if offset >= len(raw):
        raise ValueError("missing control byte")
    control = raw[offset]
    offset += 1
    pid = None
    if control & 1 == 0 or control & 0xEF == 3:
        if offset >= len(raw):
            raise ValueError("missing PID")
        pid = raw[offset]
        offset += 1
    return Frame(tuple(addresses), control, pid, raw[offset:])


def ui_frame(src: str, dest: str, info: bytes, via: tuple[str, ...] = ()) -> bytes:
    calls = (dest, src, *via)
    return Frame(tuple(address(c, last=i == len(calls) - 1, flag=i == 0)
                       for i, c in enumerate(calls)), 3, 0xF0, info).encode()


def with_fcs(raw: bytes) -> bytes:
    parse(raw)
    return raw + crc16(raw).to_bytes(2, "little")


def check_fcs(wire: bytes) -> bytes:
    if len(wire) < 2 or crc16(wire[:-2]) != int.from_bytes(wire[-2:], "little"):
        raise ValueError("AX.25 FCS failure")
    parse(wire[:-2])
    return wire[:-2]
