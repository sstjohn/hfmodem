# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""D_PDUs: the frames the data transfer sublayer puts on the air (annex C.3).

Every D_PDU is the 16-bit Maury-Styles sync (0xEB90, least significant bit
first, so 0x90 then 0xEB), a header, and the header's 16-bit CRC; the types
that carry data follow it with a segment of a C_PDU (up to 1023 bytes) and
that segment's 32-bit CRC.

The header (C.3.2): the type (4 bits) and the engineering orderwire (12
bits), the end of transmission (EOT, in half seconds), a byte of the address
field's size (3 bits, m) and the header's (5 bits, h: everything but the
address, the CRC included), the addresses (destination then source, m/2
bytes each, so in half-byte steps), then what the type needs.

Both CRCs are computed a bit at a time, least significant bit of each byte
first, in a register shifted right from zero, the polynomial read backwards
(C.3.2.8, C.3.2.11); each is sent least significant byte first. The
annex's worked examples are in the test suite.

Where a figure's field would not read unambiguously off the text, the bits
were placed by the figure's columns: in the NON-ARQ header the bit the text
calls DELIVER IN ORDER is the one its figure marks reserved (bit 3), between
the C_PDU ID's top four bits and GROUP ADDRESS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SYNC = b"\x90\xeb"

# Types (table C-2).
DATA_ONLY = 0
ACK_ONLY = 1
DATA_ACK = 2
RESET_WIN_RESYNC = 3
EXP_DATA_ONLY = 4
EXP_ACK_ONLY = 5
MANAGEMENT = 6
NON_ARQ = 7
EXP_NON_ARQ = 8
WARNING = 15

NAMES = {DATA_ONLY: "DATA-ONLY", ACK_ONLY: "ACK-ONLY", DATA_ACK: "DATA-ACK",
         RESET_WIN_RESYNC: "RESET/WIN-RESYNC", EXP_DATA_ONLY: "EXP-DATA-ONLY",
         EXP_ACK_ONLY: "EXP-ACK-ONLY", MANAGEMENT: "MANAGEMENT", NON_ARQ: "NON-ARQ DATA",
         EXP_NON_ARQ: "EXP-NON-ARQ DATA", WARNING: "WARNING"}

#: The types that carry a segment of a C_PDU.
DATA_TYPES = frozenset({DATA_ONLY, DATA_ACK, EXP_DATA_ONLY, NON_ARQ, EXP_NON_ARQ})

MAX_SEGMENT = 1023
#: Addresses are 28 bits at most (3.5 bytes each, C.3.2.6).
MAX_ADDRESS = (1 << 28) - 1

# WARNING reasons (table C-3).
WARN_UNRECOGNISED = 0
WARN_UNCONNECTED = 1


# --------------------------------------------------------------------------
# CRCs
# --------------------------------------------------------------------------

def _crc_table(poly_reversed: int) -> list[int]:
    out = []
    for b in range(256):
        c = b
        for _ in range(8):
            c = (c >> 1) ^ poly_reversed if c & 1 else c >> 1
        out.append(c)
    return out


#: x^16 + x^15 + x^12 + x^11 + x^8 + x^6 + x^3 + 1, reversed.
_T16 = _crc_table(0x9299)
#: x^32 + x^27 + x^25 + x^23 + x^21 + x^18 + x^17 + x^16 + x^13 + x^10 + x^8
#: + x^7 + x^6 + x^3 + x^2 + x + 1, reversed.
_T32 = _crc_table(0xF3A4E550)


def crc16(data: bytes, crc: int = 0) -> int:
    for b in data:
        crc = (crc >> 8) ^ _T16[(crc ^ b) & 0xFF]
    return crc


def crc32(data: bytes, crc: int = 0) -> int:
    for b in data:
        crc = (crc >> 8) ^ _T32[(crc ^ b) & 0xFF]
    return crc


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

def address(text: str | int) -> int:
    """"w.x.y.z" (w 0-15, the rest 0-255; leading parts may be left out)
    or an integer -> the 28-bit address (C.3.2.6)."""
    if isinstance(text, int):
        if not 0 <= text <= MAX_ADDRESS:
            raise ValueError(f"an address is 0 to {MAX_ADDRESS}, not {text}")
        return text
    parts = [int(p) for p in str(text).strip().split(".")]
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"an address is w.x.y.z, not {text!r}")
    parts = [0] * (4 - len(parts)) + parts
    w, x, y, z = parts
    if not (0 <= w <= 15 and all(0 <= v <= 255 for v in (x, y, z))):
        raise ValueError(f"an address is w.x.y.z with w 0-15 and the rest 0-255: {text!r}")
    return (w << 24) | (x << 16) | (y << 8) | z


def address_text(a: int) -> str:
    return f"{a >> 24}.{(a >> 16) & 255}.{(a >> 8) & 255}.{a & 255}"


def _address_field(dest: int, src: int) -> tuple[int, bytes]:
    """(m, the address field): each address in m half-bytes, the fewest
    that hold both."""
    m = max(1, (max(dest, src).bit_length() + 3) // 4)
    bits = (dest << (4 * m)) | src
    return m, bits.to_bytes(m, "big")


def _split_addresses(raw: bytes) -> tuple[int, int]:
    m = len(raw)
    bits = int.from_bytes(raw, "big")
    return bits >> (4 * m), bits & ((1 << (4 * m)) - 1)


# --------------------------------------------------------------------------
# The D_PDU
# --------------------------------------------------------------------------

@dataclass
class DPDU:
    type: int
    dest: int
    src: int
    eow: int = 0                    # 12 bits (C.3.2.2)
    eot: int = 0                    # half seconds left in the transmission
    # DATA-ONLY, DATA-ACK, EXP-DATA-ONLY
    start: bool = False
    end: bool = False
    in_order: bool = False          # also NON-ARQ
    drop: bool = False
    tx_uwe: bool = False
    tx_lwe: bool = False
    seq: int = 0                    # transmit frame sequence number
    cpdu_id: int = 0                # EXP-DATA-ONLY (mod 16), NON-ARQ (mod 4096)
    # ACK-ONLY, DATA-ACK, EXP-ACK-ONLY
    rx_lwe: int = 0
    acks: bytes = b""               # the selective-ACK bitmap, as sent
    # RESET/WIN-RESYNC
    full_reset: bool = False
    reset_tx_win_rqst: bool = False
    reset_win_cmnd: bool = False
    reset_ack: bool = False
    new_rx_lwe: int = 0
    reset_id: int = 0
    # MANAGEMENT
    valid: bool = False
    ack: bool = False
    frame_id: int = 0
    extended: bytes = b""
    # NON-ARQ, EXP-NON-ARQ
    group: bool = False
    cpdu_size: int = 0
    offset: int = 0
    window: int = 0                 # C_PDU reception window, half seconds
    # WARNING
    rx_type: int = 0
    reason: int = 0
    # The segment of a C_PDU, and on receipt whether its CRC checked.
    data: bytes = b""
    data_ok: bool = True
    #: On receipt, bytes of data in error were kept (non-ARQ with errors).
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return NAMES.get(self.type, f"type {self.type}")


def _specific(p: DPDU) -> bytes:
    t, n = p.type, len(p.data)
    if n > MAX_SEGMENT:
        raise ValueError(f"a D_PDU carries at most {MAX_SEGMENT} bytes, not {n}")
    if t in (DATA_ONLY, DATA_ACK):
        flags = (p.start << 7) | (p.end << 6) | (p.in_order << 5) | (p.drop << 4) \
            | (p.tx_uwe << 3) | (p.tx_lwe << 2) | (n >> 8)
        out = bytes([flags, n & 0xFF, p.seq & 0xFF])
        if t == DATA_ACK:
            out += bytes([p.rx_lwe & 0xFF]) + p.acks
        return out
    if t in (ACK_ONLY, EXP_ACK_ONLY):
        return bytes([p.rx_lwe & 0xFF]) + p.acks
    if t == RESET_WIN_RESYNC:
        flags = (p.full_reset << 3) | (p.reset_tx_win_rqst << 2) | (p.reset_win_cmnd << 1) \
            | p.reset_ack
        return bytes([flags, p.new_rx_lwe & 0xFF, p.reset_id & 0xFF])
    if t == EXP_DATA_ONLY:
        flags = (p.start << 7) | (p.end << 6) | ((p.cpdu_id & 0xF) << 2) | (n >> 8)
        return bytes([flags, n & 0xFF, p.seq & 0xFF])
    if t == MANAGEMENT:
        if len(p.extended) > 23:
            raise ValueError("an extended management message is at most 23 bytes")
        flags = (bool(p.extended) << 2) | (p.valid << 1) | p.ack
        return bytes([flags, p.frame_id & 0xFF]) + p.extended
    if t in (NON_ARQ, EXP_NON_ARQ):
        cid = p.cpdu_id & 0xFFF
        b0 = ((cid >> 8) << 4) | (p.in_order << 3) | (p.group << 2) | (n >> 8)
        return bytes([b0, n & 0xFF, cid & 0xFF]) + p.cpdu_size.to_bytes(2, "big") \
            + p.offset.to_bytes(2, "big") + min(p.window, 0xFFFF).to_bytes(2, "big")
    if t == WARNING:
        return bytes([((p.rx_type & 0xF) << 4) | (p.reason & 0xF)])
    raise ValueError(f"no D_PDU type {t}")


def encode(p: DPDU) -> bytes:
    """A D_PDU -> its bytes, sync to the last CRC, in the order sent."""
    if not (0 <= p.dest <= MAX_ADDRESS and 0 <= p.src <= MAX_ADDRESS):
        raise ValueError("an address is 28 bits at most")
    m, addr = _address_field(p.dest, p.src)
    spec = _specific(p)
    h = 4 + len(spec) + 2
    if h > 31:
        raise ValueError(f"a D_PDU header is at most 31 bytes, not {h}")
    head = bytes([((p.type & 0xF) << 4) | ((p.eow >> 8) & 0xF), p.eow & 0xFF,
                  min(max(p.eot, 0), 255), (m << 5) | h]) + addr + spec
    out = SYNC + head + crc16(head).to_bytes(2, "little")
    if p.type in DATA_TYPES:
        out += p.data + crc32(p.data).to_bytes(4, "little")
    return out


def size(p: DPDU) -> int:
    """How many bytes ``encode(p)`` is, without encoding it."""
    m = max(1, (max(p.dest, p.src).bit_length() + 3) // 4)
    n = 2 + 4 + m + len(_specific(p)) + 2
    if p.type in DATA_TYPES:
        n += len(p.data) + 4
    return n


class HeaderError(ValueError):
    pass


def _parse_specific(p: DPDU, spec: bytes) -> int:
    """Fill in ``p`` from its type-specific bytes; -> the segment size that
    follows the header (0 for the types that carry none)."""
    t = p.type
    need = {DATA_ONLY: 3, DATA_ACK: 4, ACK_ONLY: 1, EXP_ACK_ONLY: 1, RESET_WIN_RESYNC: 3,
            EXP_DATA_ONLY: 3, MANAGEMENT: 2, NON_ARQ: 9, EXP_NON_ARQ: 9, WARNING: 1}.get(t)
    if need is None:
        return 0
    if len(spec) < need:
        raise HeaderError(f"{NAMES[t]} header too short")
    if t in (DATA_ONLY, DATA_ACK):
        f = spec[0]
        p.start, p.end, p.in_order, p.drop = bool(f & 0x80), bool(f & 0x40), bool(f & 0x20), \
            bool(f & 0x10)
        p.tx_uwe, p.tx_lwe = bool(f & 0x08), bool(f & 0x04)
        p.seq = spec[2]
        if t == DATA_ACK:
            p.rx_lwe, p.acks = spec[3], bytes(spec[4:])
        return ((f & 3) << 8) | spec[1]
    if t in (ACK_ONLY, EXP_ACK_ONLY):
        p.rx_lwe, p.acks = spec[0], bytes(spec[1:])
        return 0
    if t == RESET_WIN_RESYNC:
        f = spec[0]
        p.full_reset, p.reset_tx_win_rqst = bool(f & 8), bool(f & 4)
        p.reset_win_cmnd, p.reset_ack = bool(f & 2), bool(f & 1)
        p.new_rx_lwe, p.reset_id = spec[1], spec[2]
        return 0
    if t == EXP_DATA_ONLY:
        f = spec[0]
        p.start, p.end, p.cpdu_id = bool(f & 0x80), bool(f & 0x40), (f >> 2) & 0xF
        p.seq = spec[2]
        return ((f & 3) << 8) | spec[1]
    if t == MANAGEMENT:
        f = spec[0]
        p.valid, p.ack = bool(f & 2), bool(f & 1)
        p.frame_id = spec[1]
        p.extended = bytes(spec[2:]) if f & 4 else b""
        return 0
    if t in (NON_ARQ, EXP_NON_ARQ):
        b0 = spec[0]
        p.in_order, p.group = bool(b0 & 8), bool(b0 & 4)
        p.cpdu_id = ((b0 >> 4) << 8) | spec[2]
        p.cpdu_size = int.from_bytes(spec[3:5], "big")
        p.offset = int.from_bytes(spec[5:7], "big")
        p.window = int.from_bytes(spec[7:9], "big")
        return ((b0 & 3) << 8) | spec[1]
    p.rx_type, p.reason = spec[0] >> 4, spec[0] & 0xF
    return 0


def decode(raw: bytes) -> DPDU:
    """One whole D_PDU (as ``encode`` makes it) -> the D_PDU. A header
    whose CRC fails raises; a segment whose CRC fails comes back with
    ``data_ok`` false (C.3.2.8: a good header is used regardless)."""
    p, n = _decode_at(raw, 0)
    if p is None:
        raise HeaderError("incomplete D_PDU")
    return p


def _decode_at(buf: bytes, at: int) -> tuple[DPDU | None, int]:
    """The D_PDU whose sync is at ``at``: (it, its length), (None, 0) if the
    buffer ends first; raises HeaderError on a bad header."""
    if len(buf) < at + 6:
        return None, 0
    if buf[at:at + 2] != SYNC:
        raise HeaderError("no sync")
    b3 = buf[at + 5]
    m, h = b3 >> 5, b3 & 0x1F
    if m == 0 or h < 6:
        raise HeaderError("impossible header size")
    end_head = at + 2 + h + m
    if len(buf) < end_head:
        return None, 0
    head = buf[at + 2:end_head - 2]
    if crc16(head) != int.from_bytes(buf[end_head - 2:end_head], "little"):
        raise HeaderError("header CRC")
    t = head[0] >> 4
    dest, src = _split_addresses(head[4:4 + m])
    p = DPDU(t, dest, src, eow=((head[0] & 0xF) << 8) | head[1], eot=head[2])
    n = _parse_specific(p, head[4 + m:])
    if t not in DATA_TYPES:
        return p, end_head - at
    if len(buf) < end_head + n + 4:
        return None, 0
    p.data = bytes(buf[end_head:end_head + n])
    p.data_ok = crc32(p.data) == int.from_bytes(buf[end_head + n:end_head + n + 4], "little")
    return p, end_head + n + 4 - at


class Parser:
    """Bytes off the air in, D_PDUs out: each found by its sync, taken if
    its header's CRC checks, and passed over (one byte on) if not."""

    #: A header is at most 2 + 31 + 7 bytes, a segment 1023 + 4: past this
    #: without a whole D_PDU, what is held is noise.
    _HOLD = 2 + 31 + 7 + MAX_SEGMENT + 4

    def __init__(self) -> None:
        self.buf = b""
        self.bad_headers = 0

    def push(self, data: bytes) -> list[DPDU]:
        self.buf += data
        out = []
        at = 0
        while True:
            i = self.buf.find(SYNC, at)
            if i < 0:
                at = max(at, len(self.buf) - 1)
                break
            try:
                p, n = _decode_at(self.buf, i)
            except HeaderError:
                self.bad_headers += 1
                at = i + 1
                continue
            if p is None:
                at = i
                break
            out.append(p)
            at = i + n
        self.buf = self.buf[at:]
        if len(self.buf) > self._HOLD:
            self.buf = self.buf[-self._HOLD:]
        return out

    def flush(self) -> None:
        """The transmission ended: nothing held can be finished."""
        self.buf = b""


# --------------------------------------------------------------------------
# Selective acknowledgements (C.3.4)
# --------------------------------------------------------------------------

def ack_bitmap(rx_lwe: int, rx_uwe: int, received) -> bytes:
    """The SELECTIVE ACKS field: a bit for each sequence number from
    RX LWE + 1 up to RX UWE (modulo 256), least significant bit first; none
    when the window holds nothing past its lower edge."""
    span = (rx_uwe - rx_lwe) & 0xFF
    if span == 0 or span >= 128:
        return b""
    bits = 0
    for i in range(span):
        if (rx_lwe + 1 + i) & 0xFF in received:
            bits |= 1 << i
    return bits.to_bytes((span + 7) // 8, "little")


def acked(rx_lwe: int, acks: bytes) -> list[int]:
    """The sequence numbers a SELECTIVE ACKS field says arrived."""
    out = []
    for i in range(8 * len(acks)):
        if acks[i // 8] >> (i % 8) & 1:
            out.append((rx_lwe + 1 + i) & 0xFF)
    return out


# --------------------------------------------------------------------------
# EOW messages (C.5)
# --------------------------------------------------------------------------

EOW_DRC_REQUEST = 1
EOW_DRC_RESPONSE = 2
EOW_UNRECOGNISED = 3
EOW_CAPABILITY = 4

#: Table C-5: data-rate codes.
DRC_RATES = {75: 0, 150: 1, 300: 2, 600: 3, 1200: 4, 2400: 5, 3200: 6, 3600: 7, 4800: 8,
             6400: 9, 7200: 10, 9600: 11}
DRC_RATE_OF = {v: k for k, v in DRC_RATES.items()}
#: Table C-6.
INTERLEAVE = {"none": 0, "short": 1, "long": 2}
INTERLEAVE_OF = {v: k for k, v in INTERLEAVE.items()}
# Table C-7 (DRC response) and C-8 (reason).
RESP_ACCEPT, RESP_REFUSE, RESP_CANCEL, RESP_CONFIRM = 0, 1, 2, 3
REASON_NONE, REASON_TX_RX_SAME, REASON_NO_RATE, REASON_NO_INTERLEAVE, REASON_NEITHER, \
    REASON_LOCAL = 0, 1, 2, 3, 4, 5


def eow_drc_request(rate: int, interleave: str, other: int) -> int:
    """Type 1: the rate the sender will (or advises the other to) use;
    ``other`` per table C-7 (0/1 a request, 2/3 advisory)."""
    return (EOW_DRC_REQUEST << 8) | (DRC_RATES[rate] << 4) | (INTERLEAVE[interleave] << 2) \
        | (other & 3)


def eow_drc_response(response: int, reason: int = REASON_NONE) -> int:
    return (EOW_DRC_RESPONSE << 8) | ((response & 7) << 5) | (reason & 0x1F)


def eow_fields(eow: int) -> tuple[int, int]:
    """(type, contents)."""
    return eow >> 8, eow & 0xFF
