# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Propagation report service 1, independent of the selected waveform.

Receiver measurements belong in the reception event, not in this transmitter
record. Power is stated transmitter power, not a calibrated EIRP claim.
"""
from dataclasses import dataclass
import re
import struct

SERVICE = 1
_HEADER = struct.Struct(">BQQhB")


@dataclass(frozen=True)
class PropagationReport:
    utc_ms: int
    frequency_hz: int
    power_cdbm: int
    grid: str

    def pack(self):
        grid = self.grid.upper()
        if not re.fullmatch(r"[A-R]{2}[0-9]{2}(?:[A-X]{2}(?:[0-9]{2})?)?", grid):
            raise ValueError("invalid Maidenhead locator")
        return _HEADER.pack(1, self.utc_ms, self.frequency_hz, self.power_cdbm, len(grid)) + grid.encode("ascii")

    @classmethod
    def unpack(cls, raw):
        if len(raw) < _HEADER.size:
            raise ValueError("truncated propagation report")
        version, t, freq, power, n = _HEADER.unpack_from(raw)
        if version != 1 or len(raw) != _HEADER.size + n:
            raise ValueError("unsupported propagation report")
        obj = cls(t, freq, power, raw[_HEADER.size:].decode("ascii"))
        if obj.pack() != raw:
            raise ValueError("noncanonical propagation report")
        return obj


def beacon_power_status(power_dbm: int) -> int:
    """Compact propagation-v1 application profile for the existing status byte.

    Decode this interpretation only when propagation-v1 is selected by the
    application; other applications may interpret the status byte differently.
    """
    if not isinstance(power_dbm, int) or not -30 <= power_dbm <= 60:
        raise ValueError("compact power must be an integer from -30 to +60 dBm")
    return 0x80 | (power_dbm + 30)


def beacon_status_power(status: int) -> int:
    if not 0x80 <= status <= 0xDA:
        raise ValueError("not a propagation-v1 power status")
    return (status & 0x7F) - 30
