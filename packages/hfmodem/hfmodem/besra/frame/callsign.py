# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Callsigns and grid squares in the 6-byte fields of ConReq, Ping and IDFrame.

Each field is eight characters of six bits apiece, 48 bits sent big-endian. The
characters are ASCII space through underscore, each sent as its code less 0x20;
lowercase letters are sent as their capitals and anything else as a space. The
layout is ARDOP's (`docs/protocols/ardop/11-WAVEFORM.md` §8.2) and
`tests/besra/test_callsign.py` holds it to wire vectors.

A station's field is its callsign padded with spaces to seven characters, then
one character for the SSID. That character is the SSID's value read in base 36
and indexed into ``0-9``, ``A-Z``, then ``:;<=>?``. Base 36 is the interoperable
reading, and it has a consequence worth knowing: ``-15`` reads as 1·36 + 5 = 41
and is sent as ``?``. A callsign with no SSID is sent with ``0`` in that place,
not a space; peers hash the field into the session ID, so the difference matters.
"""

from __future__ import annotations

PACKED6_SIZE = 6
PACKED6_MAX = 8

CALLSIGN_MIN = 2
CALLSIGN_MAX = 7

#: What a callsign can be made of. Packed6's alphabet is ASCII 32-63, so an
#: RS-corrected block of noise decodes to a field that is the right length and
#: holds no space, and until this check existed that was the whole test -- a
#: ConReq or IDFrame sighting was worth what 4 RS parity bytes over 12 are worth
#: and no more. Over the 81-recording replay corpus, 64 named-callsign sightings:
#: this rejects 2, `ConReq1000M FC<C155-5 > ;_0Q5@@` and `IDFrame *?V"*JJ-E`,
#: and no legitimate sighting.
CALLSIGN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")

_BITS = 6
_FIRST = 0x20                    # space, sent as 0
_SSID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:;<=>?"


def _code(ch: str) -> int:
    if "a" <= ch <= "z":
        ch = ch.upper()
    code = ord(ch) - _FIRST
    return code if 0 <= code < 1 << _BITS else 0


def packed6_encode(s: str) -> bytes:
    """Pack a string (padded with spaces / truncated to 8 chars) into 6 bytes."""
    value = 0
    for ch in s[:PACKED6_MAX].ljust(PACKED6_MAX):
        value = value << _BITS | _code(ch)
    return value.to_bytes(PACKED6_SIZE, "big")


def packed6_decode(b: bytes) -> str:
    """Unpack 6 bytes to the 8-character (space-padded) SIXBIT string."""
    if len(b) != PACKED6_SIZE:
        raise ValueError(f"Packed6 is {PACKED6_SIZE} bytes, got {len(b)}")
    value = int.from_bytes(b, "big")
    mask = (1 << _BITS) - 1
    return "".join(chr(_FIRST + (value >> _BITS * i & mask))
                   for i in reversed(range(PACKED6_MAX)))


def _ssid_pack(ssid: str) -> str:
    """The SSID's one character: its text read in base 36, so "7" is 7, "A" is 10
    and "15" is 41."""
    try:
        n = int(ssid.strip(), 36)
    except ValueError as exc:
        raise ValueError(f"invalid SSID {ssid!r}") from exc
    if not 0 <= n < len(_SSID_CHARS):
        raise ValueError(f"SSID {ssid!r} out of range")
    return _SSID_CHARS[n]


def _ssid_unpack(char: str) -> str:
    """The SSID text a packed character stands for: "" for the default ``0``, a
    letter as itself, and ``1`` through ``?`` as their distance from ``0``."""
    if char == "0":
        return ""
    if "1" <= char <= "?":
        return str(ord(char) - ord("0"))
    if "A" <= char <= "Z":
        return char
    raise ValueError(f"invalid SSID byte {char!r}")


def pack_callsign(call_ssid: str) -> bytes:
    """Pack ``CALL`` or ``CALL-SSID`` into its 6-byte StationId wire form."""
    call, _, ssid = call_ssid.partition("-")
    call = call.upper()
    if not CALLSIGN_MIN <= len(call) <= CALLSIGN_MAX or not set(call) <= CALLSIGN_CHARS:
        raise ValueError(f"invalid callsign {call!r}")
    return packed6_encode(call.ljust(CALLSIGN_MAX) + _ssid_pack(ssid or "0"))


def unpack_callsign(b: bytes) -> str:
    """Recover the canonical ``CALL`` / ``CALL-SSID`` string from 6 wire bytes."""
    field = packed6_decode(b)
    call = field[:CALLSIGN_MAX].rstrip(" ")
    if not CALLSIGN_MIN <= len(call) <= CALLSIGN_MAX or not set(call) <= CALLSIGN_CHARS:
        raise ValueError(f"invalid packed callsign {call!r}")
    ssid = _ssid_unpack(field[CALLSIGN_MAX])
    return f"{call}-{ssid}" if ssid else call


def pack_grid(grid: str) -> bytes:
    """Pack a Maidenhead grid square (2/4/6/8 chars) into 6 bytes for IDFRAME."""
    grid = grid.strip()
    if len(grid) not in (2, 4, 6, 8):
        raise ValueError(f"grid must be 2/4/6/8 chars, got {grid!r}")
    return packed6_encode(grid)


def unpack_grid(b: bytes) -> str:
    """Recover the grid square from 6 wire bytes (uppercase, spaces trimmed)."""
    return packed6_decode(b).rstrip(" ")
