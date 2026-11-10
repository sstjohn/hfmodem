# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Selcall calls as characters (CCIR 493-4 land mobile; the UN open 6-digit
selcall).

**Characters** (ITU-R M.493 section 1): ten bits, seven of information sent
least significant first, then the number of 0 ("B") bits among those seven
as three bits, most significant first. 0 to 99 carry two decimal digits;
100 to 127 are service symbols.

**A call** is a dot pattern, a phasing sequence — six pairs of 125 (the DX
position) and 109 down to 104 (the RX position) — and then its characters,
each sent twice: once in the DX position and again four characters later
in the RX position. Before the first repeats reach the RX position it
carries the format character. This is the land-mobile transmit layout,
not the M.493 repeated-format layout. Reception accepts both; M.493
reception remains unverified against a foreign implementation (SELCALL.md).

**The content**, in the DX position: the format (120 for a call to
stations; 125 for a beacon call), the called address as digit pairs (two
for four digits, three for six), the category (100, routine), the caller's
own address the same way, what the call carries, and the end character
three times: 117 asks for an acknowledgment. A called address ending in
zeros reaches more than one station: X000 all, XX00 a group, XXX0 a
sub-group (XXX000, XXXX00 and XXXXX0 in six digits).

What each kind carries follows the UN open 6-digit specification's
diagrams (as HFLINK publishes them) and a capture of a real call printed
with them; the 4-digit calls carry the same with two-character addresses,
as the 4-digit layout HFLINK publishes shows for the selective call. Where
the diagram and the capture disagree — the format character of a 6-digit
call (126 drawn, 120 sent) — we send what was sent and read either.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Service symbols.
ROUTINE = 100
TELCALL = 102
STATUS = 103
HANGUP = (104, 105)
PAGE = 116
ACK_RQ = 117                  # end: an acknowledgment asked for
EMERGENCY = 119
INDIVIDUAL = 120
GPS_SEND = 121
ACK_BQ = 122                  # end: this is an acknowledgment; also emergency with GPS
BEACON = 125
PHASE_DX = 125
PHASE_RX = (109, 108, 107, 106, 105, 104)
NULL = 126
EOS = 127

KINDS = ("selcall", "beacon", "hangup", "telcall", "page", "gps", "gps_request",
         "status_request", "emergency", "ack")


def char_bits(v: int) -> list[int]:
    """A character's ten bits in the order sent."""
    if not 0 <= v <= 127:
        raise ValueError(f"a character is 0 to 127, not {v}")
    info = [(v >> i) & 1 for i in range(7)]
    zeros = 7 - sum(info)
    return info + [(zeros >> 2) & 1, (zeros >> 1) & 1, zeros & 1]


def char_value(bits) -> tuple[int, bool]:
    """Ten bits (as sent) -> (value, whether the check bits agree)."""
    b = [int(x) for x in bits]
    v = sum(bit << i for i, bit in enumerate(b[:7]))
    return v, (b[7] << 2 | b[8] << 1 | b[9]) == 7 - sum(b[:7])


def address_chars(addr: str) -> list[int]:
    """"1234" -> [12, 34]; six digits -> three characters."""
    if not (addr.isdigit() and len(addr) in (4, 6)):
        raise ValueError(f"a selcall address is 4 or 6 digits, not {addr!r}")
    return [int(addr[i:i + 2]) for i in range(0, len(addr), 2)]


def chars_address(cs) -> str:
    return "".join(f"{c:02d}" for c in cs)


def reaches(called: str, me: str) -> bool:
    """Whether a call to ``called`` is for station ``me``: the same address,
    or one whose trailing zeros make it an all, group or sub-group call
    that ``me`` falls in; all zeros is every station (the UN specification's
    emergency call is to 000000)."""
    if len(called) != len(me):
        return False
    stem = called.rstrip("0")
    if len(stem) == len(called):
        return called == me
    return me.startswith(stem)


@dataclass
class Call:
    kind: str
    to: str                                   # the called address
    frm: str                                  # the caller's address
    #: What the call carries: the telephone number, the page text, the
    #: position (lat, lon in degrees) and status byte, or nothing.
    number: str = ""
    text: str = ""
    position: tuple | None = None
    status: int = 0
    end: int = ACK_RQ
    fmt: int = INDIVIDUAL
    #: Receive only: characters that could not be read, and whether a
    #: parity character (telcall, GPS) disagreed.
    erasures: int = 0
    parity_ok: bool = True
    message: list = field(default_factory=list)


# -- the message of each kind -------------------------------------------------

def _xor(cs) -> int:
    v = 0
    for c in cs:
        v ^= c
    return v


def _position_chars(lat: float, lon: float) -> tuple[list[int], int]:
    """(D1..D8, the hemisphere bits of the status byte): latitude and
    longitude each as eight digits, degrees (three) minutes (two) and
    thousandths of a minute (three), paired."""
    def digits(v: float) -> str:
        a = abs(v)
        deg = int(a)
        thou = int(round((a - deg) * 60 * 1000))
        if thou >= 60000:
            deg, thou = deg + 1, thou - 60000
        return f"{deg:03d}{thou // 1000:02d}{thou % 1000:03d}"
    s = digits(lat) + digits(lon)
    hemi = (1 if lon >= 0 else 0) << 1 | (1 if lat >= 0 else 0)
    return [int(s[i:i + 2]) for i in range(0, 16, 2)], hemi


def _position_of(ds, sb) -> tuple[float, float]:
    s = "".join(f"{d:02d}" for d in ds)
    def value(p: str) -> float:
        return int(p[:3]) + (int(p[3:5]) + int(p[5:8]) / 1000) / 60
    lat, lon = value(s[:8]), value(s[8:])
    return (lat if sb & 1 else -lat), (lon if sb & 2 else -lon)


def encode(c: Call) -> list[int]:
    """The characters of a call in the DX position, format to end."""
    if c.kind not in KINDS:
        raise ValueError(f"no such call {c.kind!r}")
    to, frm = address_chars(c.to), address_chars(c.frm)
    if len(to) != len(frm):
        raise ValueError("the called and calling addresses are both 4 or both 6 digits")
    fmt, end = (BEACON if c.kind == "beacon" else c.fmt), c.end
    msg: list[int] = []
    if c.kind == "hangup":
        msg = list(HANGUP)
    elif c.kind == "telcall":
        n = c.number
        if not (n.isdigit() and 1 <= len(n) <= 16):
            raise ValueError("a telephone number is 1 to 16 digits")
        ts = [int(n.ljust(16, "0")[i:i + 2]) for i in range(0, 16, 2)]
        msg = [TELCALL, len(n)] + ts
        msg.append(_xor(msg[1:]))
    elif c.kind == "page":
        t = c.text.upper() if c.text else ""
        if not 1 <= len(t) <= 32 or any(not 32 <= ord(ch) < 127 for ch in t):
            raise ValueError("a page is 1 to 32 printable characters")
        msg = [PAGE] + [ord(ch) for ch in t]
    elif c.kind in ("gps", "emergency") and c.position is not None:
        ds, hemi = _position_chars(*c.position)
        sb = (c.status & ~3) | hemi
        msg = [GPS_SEND if c.kind == "gps" else ACK_BQ, sb] + ds
        msg.append(_xor(msg[1:]))                # the status byte and D1..D8
    elif c.kind == "gps":
        raise ValueError("a GPS call carries a position")
    elif c.kind == "emergency":
        msg, end = [EMERGENCY], EMERGENCY
    elif c.kind == "gps_request":
        msg = [BEACON, BEACON]
    elif c.kind == "status_request":
        msg = [STATUS, STATUS]
    elif c.kind == "ack":
        end = ACK_BQ
    return [fmt] + to + [ROUTINE] + frm + msg + [end] * 3


def air_chars(dx: list[int]) -> list[int]:
    """Phasing, then the DX and RX positions alternately."""
    out = []
    for rx in PHASE_RX:
        out += [PHASE_DX, rx]
    rxs = [dx[0], dx[0]] + dx[:-2]
    for d, r in zip(dx, rxs):
        out += [d, r]
    return out


ENDS = (ACK_RQ, ACK_BQ, EMERGENCY, EOS)


def decode(dx: list[int | None]) -> Call | None:
    """A call from its DX characters (None where unread); None if the
    characters do not make one."""
    if len(dx) < 8 or dx[0] is None:
        return None
    if len(dx) > 1 and dx[0] == dx[1]:
        dx = dx[:1] + dx[2:]
    fmt = dx[0]
    if fmt not in (INDIVIDUAL, NULL, BEACON):
        return None
    # The category marks where the called address ends.
    try:
        cat = next(i for i in range(1, 5) if dx[i] is not None and dx[i] >= 100)
    except StopIteration:
        return None
    n = cat - 1
    if n not in (2, 3) or dx[cat] != ROUTINE or len(dx) < cat + 1 + n + 3:
        return None
    to, frm = dx[1:cat], dx[cat + 1:cat + 1 + n]
    if any(c is None or c > 99 for c in to + frm):
        return None
    rest = list(dx[cat + 1 + n:])
    # The end: the last character, repeated; up to three of it go.
    while rest and rest[-1] is None:
        rest.pop()
    if not rest or rest[-1] not in ENDS:
        return None
    end = rest[-1]
    k = 0
    while rest and rest[-1] == end and k < 3:
        rest.pop()
        k += 1
    call = Call("selcall", chars_address(to), chars_address(frm), end=end, fmt=fmt,
                message=rest, erasures=sum(c is None for c in dx))
    m = rest
    if fmt == BEACON:
        call.kind = "beacon"
    elif end == ACK_BQ and not m:
        call.kind = "ack"
    elif m == list(HANGUP):
        call.kind = "hangup"
    elif m[:1] == [TELCALL] and len(m) >= 11 and None not in m:
        call.kind = "telcall"
        call.number = "".join(f"{t:02d}" for t in m[2:10])[:m[1]]
        call.parity_ok = _xor(m[1:10]) == m[10]
    elif m[:1] == [PAGE]:
        call.kind = "page"
        call.text = "".join("?" if ch is None else chr(ch) for ch in m[1:]).rstrip()
    elif m[:1] in ([GPS_SEND], [ACK_BQ]) and len(m) >= 11 and None not in m:
        call.kind = "gps" if m[0] == GPS_SEND else "emergency"
        call.status = m[1]
        call.position = _position_of(m[2:10], m[1])
        # The capture's parity covers the status byte; the legend says D1..D8.
        call.parity_ok = m[10] in (_xor(m[1:10]), _xor(m[2:10]))
    elif m == [EMERGENCY] or end == EMERGENCY:
        call.kind = "emergency"
    elif m == [BEACON, BEACON]:
        call.kind = "gps_request"
    elif m == [STATUS, STATUS]:
        call.kind = "status_request"
    elif m:
        call.kind = "unknown"
    return call
