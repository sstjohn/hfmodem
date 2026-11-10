# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class UIAddress:
    base: str
    ssid: int = 0
    flags: int = 0x60

    @property
    def call(self):
        return self.base + (f'-{self.ssid}' if self.ssid else '')

    @property
    def h(self):
        return self.flags >> 7

    def ax25(self, width):
        if len(self.base) > width or not 0 <= self.ssid <= 15:
            raise ValueError('UI address width or SSID out of range')
        return bytes(ord(c) << 1 for c in self.base.ljust(width)) + bytes([
            self.flags & 0xE1 | self.ssid << 1])


@dataclass(frozen=True)
class UIFrame:
    command: int
    addresses: tuple[UIAddress, ...]
    control: int
    pid: int
    info: bytes
    sample: int = 0
    sn: float | None = 0.0

    @classmethod
    def from_ax25(cls, raw, command=0):
        if command not in (0, 1):
            raise ValueError('UI uses KISS command 0 or 1')
        raw = bytes(raw)
        width = 6 + command
        addresses, at = [], 0
        while True:
            field = raw[at:at + width + 1]
            if len(field) != width + 1:
                raise ValueError('incomplete UI address')
            if any(c & 1 for c in field[:-1]):
                raise ValueError('UI address bytes must be shifted ASCII')
            base = bytes(c >> 1 for c in field[:-1]).decode('ascii').rstrip(' ')
            addresses.append(UIAddress(base, field[-1] >> 1 & 15, field[-1] & 0xE1))
            at += width + 1
            if field[-1] & 1:
                break
        if len(addresses) < 2 or len(raw) < at + 2:
            raise ValueError('UI requires destination, source, control and PID')
        return cls(command, tuple(addresses), raw[at], raw[at + 1], raw[at + 2:])

    @property
    def source(self):
        return self.addresses[1].call

    @property
    def destination(self):
        return self.addresses[0].call

    @property
    def path(self):
        return self.addresses[2:]

    @property
    def text(self):
        return self.info.decode('utf-8')

    def ax25(self):
        if self.command not in (0, 1):
            raise ValueError('UI uses KISS command 0 or 1')
        return (b''.join(a.ax25(6 + self.command) for a in self.addresses)
                + bytes([self.control, self.pid]) + self.info)

    def encode(self):
        return self.ax25()

    def normalized(self):
        if len(self.addresses) < 2 or not self.addresses[1].base.strip():
            raise ValueError('UI has no source identifier')
        if len(self.path) > 8:
            raise ValueError('UI has more than eight digipeaters')
        self.ax25()
        for address in self.addresses:
            if not re.fullmatch(r'[A-Z0-9]+', address.base.upper()):
                raise ValueError('UI address base must contain only A-Z or 0-9; '
                                 'VARA HF RF truncates unencodable identifiers')
        return replace(self, addresses=tuple(
            replace(a, base=a.base.upper()) for a in self.addresses))
