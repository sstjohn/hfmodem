"""ARDOP callsign and grid-square encoding: DEC SIXBIT "Packed6" plus the
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

StationId (callsign+SSID) and Locator (Maidenhead grid) fields that ride the
12-byte payload of ConReq, PING and IDFRAME.

Reimplemented from `docs/protocols/ardop/11-WAVEFORM.md` §8.2 and the reference
`Packed6.c` / `StationId.c` / `Locator.c`, validated against M0LTE's C# port
(`tests/besra/test_callsign.py`).

Packed6 packs an 8-character field into 6 bytes: two 4-char halves, each 4 chars
of 6 bits packed big-endian into 3 bytes (first char in the most-significant 6
bits). The alphabet is ASCII 32 (space) through 63 (underscore) stored as
value-32; lowercase folds to uppercase; anything else becomes space.

A StationId renders ``%-7.7s%c`` -- the callsign left-justified in 7 characters
followed by one SSID byte -- before packing. The SSID byte comes from a base-36
read of the SSID text (`stationid_ssid_pack`, StationId.c:189): 0-9 -> '0'-'9',
10-35 -> 'A'-'Z', 36-41 -> ':'..'?'. Note the base-36 quirk that makes ``-15``
read as 41 and pack to '?'. A callsign with no ``-SSID`` inherits the SSID
string "0" that `stationid_init` seeds (StationId.c:233), so it packs the byte
'0' -- not a blank -- which the ConReq/ID wire bytes depend on.

The parts of this module ``NOTICE`` names as ardopcf's are under that project's
MIT licence, Copyright (c) 2014-2024 Rick Muething, John Wiseman, Peter LaRue;
the copyright and permission notice it requires ship in ``NOTICE``.
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


def _compress_four(chars: str) -> bytes:
    pack = 0
    for ch in chars:
        b = ord(ch)
        if 0x20 <= b <= 0x5F:  # space .. underscore, verbatim
            v = b - 0x20
        elif 0x61 <= b <= 0x7A:  # a..z fold to uppercase
            v = b - (0x61 - 0x41) - 0x20
        else:
            v = 0  # unrepresentable -> space
        pack = (pack << 6) | (v & 0x3F)
    return bytes((pack >> (8 * (2 - i))) & 0xFF for i in range(3))


def _decompress_three(triple: bytes) -> str:
    unpack = 0
    for byte in triple:
        unpack = (unpack << 8) | byte
    return "".join(chr(((unpack >> (6 * (3 - i))) & 0x3F) + 0x20) for i in range(4))


def packed6_encode(s: str) -> bytes:
    """Pack a string (padded with spaces / truncated to 8 chars) into 6 bytes."""
    work = s[:PACKED6_MAX].ljust(PACKED6_MAX)
    return _compress_four(work[:4]) + _compress_four(work[4:])


def packed6_decode(b: bytes) -> str:
    """Unpack 6 bytes to the 8-character (space-padded) SIXBIT string."""
    if len(b) != PACKED6_SIZE:
        raise ValueError(f"Packed6 is {PACKED6_SIZE} bytes, got {len(b)}")
    return _decompress_three(b[:3]) + _decompress_three(b[3:])


def _ssid_pack(ssid: str) -> str:
    """One SSID byte from its text form, base-36 as in `stationid_ssid_pack`.

    strtol(base=36) reads "15" as 41 (1*36+5), so numeric SSIDs 10-15 land in the
    36-41 window that maps to ':'..'?'. Single letters A-Z read as 10-35 -> 'A'-'Z'.
    """
    try:
        n = int(ssid.strip(), 36)
    except ValueError as exc:
        raise ValueError(f"invalid SSID {ssid!r}") from exc
    if 0 <= n <= 9:
        return chr(ord("0") + n)
    if 10 <= n <= 35:
        return chr(ord("A") + n - 10)
    if 36 <= n <= 41:
        return chr(ord(":") + n - 36)
    raise ValueError(f"SSID {ssid!r} out of range")


def _ssid_unpack(byte: str) -> str:
    """SSID text from its packed byte, inverse of `_ssid_pack`. Returns "" for the
    default SSID (byte '0') so the canonical form drops the suffix."""
    if "0" <= byte <= "?":
        n = ord(byte) - ord("0")
        return "" if n == 0 else str(n)
    if "A" <= byte <= "Z":
        return byte
    raise ValueError(f"invalid SSID byte {byte!r}")


def pack_callsign(call_ssid: str) -> bytes:
    """Pack ``CALL`` or ``CALL-SSID`` into its 6-byte StationId wire form."""
    call, _, ssid = call_ssid.partition("-")
    call = call.upper()
    if not ssid:
        ssid = "0"  # stationid_init seeds ssid "0" -> packed byte '0' (StationId.c:233)
    if not CALLSIGN_MIN <= len(call) <= CALLSIGN_MAX or not set(call) <= CALLSIGN_CHARS:
        raise ValueError(f"invalid callsign {call!r}")
    return packed6_encode(f"{call[:CALLSIGN_MAX]:<7}{_ssid_pack(ssid)}")


def unpack_callsign(b: bytes) -> str:
    """Recover the canonical ``CALL`` / ``CALL-SSID`` string from 6 wire bytes."""
    work = packed6_decode(b)
    call = work[:CALLSIGN_MAX].rstrip(" ")
    if not CALLSIGN_MIN <= len(call) <= CALLSIGN_MAX:
        raise ValueError(f"invalid packed callsign {call!r}")
    if not set(call) <= CALLSIGN_CHARS:
        raise ValueError(f"invalid packed callsign {call!r}")
    ssid = _ssid_unpack(work[CALLSIGN_MAX])
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
