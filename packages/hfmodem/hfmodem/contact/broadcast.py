# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Direct AX.25 UI envelopes for VARA KISS commands 0 and 1.

EA5HVK's February 2024 interface specifies six-character standard addresses for
command 0 and seven-character VarAC addresses for command 1. Values exclude
AX.25 FCS. These are unlinked messages, never VarAC application-stream tags.
"""

from dataclasses import dataclass
import re

MAX_INFO_BYTES = 77  # Qualified application limit, not a claim about all KISS.


def _split(call, width):
    call = str(call).upper()
    match = re.fullmatch(r"(.+)-([0-9]{1,2})", call)
    base, ssid = (match[1], int(match[2])) if match else (call, 0)
    if not 1 <= len(base) <= width or not re.fullmatch(r"[A-Z0-9-]+", base):
        raise ValueError(
            f"UI address needs one to {width} uppercase letters, digits or hyphens"
        )
    if not 0 <= ssid <= 15:
        raise ValueError("UI SSID must be 0 to 15")
    return base, ssid


def _address(call, width, *, last=False, command=False):
    base, ssid = _split(call, width)
    return bytes(ord(c) << 1 for c in base.ljust(width)) + bytes(
        [0x60 | ssid << 1 | int(last) | (0x80 if command else 0)]
    )


def encode_ui(source: str, destination: str, info: bytes, *, command=1) -> bytes:
    """Encode a direct UI envelope; PHY admission is a separate operation.

    Internal hyphens are representable in the host envelope. Stock's measured
    RF encoder truncates X-W9SSJ to X; this function does not silently perform
    that lossy conversion or assert it is interoperable over the air.
    """
    if command not in (0, 1):
        raise ValueError("UI uses KISS command 0 or 1")
    if not isinstance(info, bytes) or not 1 <= len(info) <= MAX_INFO_BYTES:
        raise ValueError(f"broadcast needs 1 to {MAX_INFO_BYTES} information bytes")
    width = 6 + command
    return (
        _address(destination, width, command=True)
        + _address(source, width, last=True)
        + b"\x03\xf0"
        + info
    )


@dataclass(frozen=True)
class Broadcast:
    command: int
    source: str
    destination: str
    info: bytes
    sample: int = 0
    sn: float | None = 0.0

    @property
    def text(self):
        """Strict UTF-8; callers must label opaque bytes rather than invent text."""
        return self.info.decode("utf-8")

    def encode(self):
        return encode_ui(self.source, self.destination, self.info, command=self.command)


def decode_ui(raw: bytes, *, command=1, sample=0, sn=0.0) -> Broadcast:
    """Validate a direct UI frame; no digipeaters, FCS or application execution."""
    if command not in (0, 1):
        raise ValueError("UI uses KISS command 0 or 1")
    width = 6 + command
    size = width + 1
    if not 2 * size + 3 <= len(raw) <= 2 * size + 2 + MAX_INFO_BYTES:
        raise ValueError(
            "UI frame length is outside the qualified direct-message bounds"
        )
    calls = []
    for i in range(2):
        address = raw[i * size : (i + 1) * size]
        if any(v & 1 for v in address[:width]):
            raise ValueError("UI callsign bytes must be shifted ASCII")
        base = bytes(v >> 1 for v in address[:width]).decode("ascii").rstrip(" ")
        flags = address[-1]
        if flags & 0x60 != 0x60 or flags & 1 != int(i == 1):
            raise ValueError("UI requires two direct addresses with valid SSID flags")
        _split(base, width)  # Validate characters without rewriting the base.
        ssid = flags >> 1 & 15
        calls.append(base + (f"-{ssid}" if ssid else ""))
    if raw[2 * size : 2 * size + 2] != b"\x03\xf0":
        raise ValueError("broadcast requires unnumbered UI control and no-layer3 PID")
    return Broadcast(
        command, calls[1], calls[0], bytes(raw[2 * size + 2 :]), int(sample),
        None if sn is None else float(sn)
    )
