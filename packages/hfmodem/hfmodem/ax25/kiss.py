# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Incremental KISS framing, with bounded recovery at FEND."""
from __future__ import annotations

FEND, FESC, TFEND, TFESC = 0xC0, 0xDB, 0xDC, 0xDD
MAX_FRAME = 2048


def encode(payload: bytes, command: int = 0, port: int = 0) -> bytes:
    if not 0 <= command <= 15 or not 0 <= port <= 15:
        raise ValueError("KISS command and port must be four-bit values")
    if len(payload) > MAX_FRAME:
        raise ValueError("KISS frame exceeds 2048 bytes")
    wire = bytes([port << 4 | command]) + payload
    return bytes([FEND]) + wire.replace(bytes([FESC]), bytes([FESC, TFESC])).replace(
        bytes([FEND]), bytes([FESC, TFEND])) + bytes([FEND])


class Parser:
    def __init__(self) -> None:
        self._data = bytearray()
        self._active = False
        self._escape = False
        self.dropped = 0

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        """Return (port, command, payload), including commands for the host to apply."""
        out = []
        for b in data:
            if b == FEND:
                if self._active and self._data and not self._escape:
                    head = self._data[0]
                    out.append((head >> 4, head & 15, bytes(self._data[1:])))
                elif self._escape:
                    self.dropped += 1
                self._data.clear()
                self._active, self._escape = True, False
                continue
            if not self._active:
                continue
            if self._escape:
                self._escape = False
                if b not in (TFEND, TFESC):
                    self._active = False
                    self._data.clear()
                    self.dropped += 1
                    continue
                b = FEND if b == TFEND else FESC
            elif b == FESC:
                self._escape = True
                continue
            self._data.append(b)
            if len(self._data) > MAX_FRAME + 1:
                self._active = False
                self._data.clear()
                self.dropped += 1
        return out
