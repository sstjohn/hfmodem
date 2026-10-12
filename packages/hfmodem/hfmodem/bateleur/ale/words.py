# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ALE words, addresses and frames: MIL-STD-188-141B A.5.2.2 to A.5.2.5.

A word is 24 bits: a 3-bit preamble (W1-W3, table A-VIII) and a 21-bit data
field, usually three 7-bit characters (figure A-12). On the air it is 49 bits
(figure A-10): the two Golay halves interleaved A1 B1 ... A24 B24, B's check
bits inverted, and a stuff bit; and every word is sent three times running.

A frame is a calling cycle (a scanning call of the first address word, then
the whole address twice), an optional message section (CMD words), and a
conclusion (TIS or TWAS and the sender's whole address) — A.5.2.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import golay

# Table A-VIII.
DATA, THRU, TO, TWAS, FROM, TIS, CMD, REP = range(8)
PREAMBLE_NAMES = ("DATA", "THRU", "TO", "TWAS", "FROM", "TIS", "CMD", "REP")
ROUTING = (THRU, TO, TWAS, FROM, TIS)

#: A.5.2.4.2: A-Z, 0-9, and the utility "@" and wildcard "?".
BASIC38 = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@?")
#: A.5.7.2.1: every character whose two top bits are 01 or 10 — space to "_".
EXPANDED64 = frozenset(chr(c) for c in range(0x20, 0x60))

WORD_BITS = 49                       # on the air, per single word
TRIPLE_BITS = 3 * WORD_BITS          # 147: one redundant word, 49 tones


def bits_int(bits) -> int:
    bits = np.asarray(bits, dtype=np.uint8)
    return int.from_bytes(np.packbits(bits).tobytes(), "big") >> (-len(bits) % 8)


def word(preamble: int, text: str = "", *, bits: int | None = None) -> int:
    """A 24-bit word from a preamble and three characters, or 21 raw bits."""
    if bits is None:
        if len(text) != 3:
            raise ValueError(f"a word holds three characters, not {text!r}")
        bits = 0
        for ch in text:
            c = ord(ch)
            if c > 0x7F:
                raise ValueError(f"{ch!r} is not 7-bit ASCII")
            bits = (bits << 7) | c
    return ((preamble & 7) << 21) | (bits & 0x1FFFFF)


def preamble_of(w: int) -> int:
    return (w >> 21) & 7


def chars_of(w: int) -> str:
    return "".join(chr((w >> s) & 0x7F) for s in (14, 7, 0))


def describe(w: int) -> str:
    return f"{PREAMBLE_NAMES[preamble_of(w)]} {chars_of(w)!r}"


# --------------------------------------------------------------------------
# On the air: A.5.2.2.3
# --------------------------------------------------------------------------

def to_air(w: int) -> np.ndarray:
    """A word's 49 transmitted bits, first sent first."""
    a = golay.encode((w >> 12) & 0xFFF)
    b = golay.encode(w & 0xFFF) ^ 0xFFF              # G13-G24 inverted
    out = np.zeros(WORD_BITS, dtype=np.uint8)
    for i in range(24):
        out[2 * i] = (a >> (23 - i)) & 1
        out[2 * i + 1] = (b >> (23 - i)) & 1
    return out                                        # out[48], S49, stays 0


#: Where each of the 48 received bits goes: A (24, MSB first), then B.
_DEINTERLEAVE = np.concatenate([np.arange(0, 48, 2), np.arange(1, 48, 2)])
#: Soft values are positive for 0; B's check bits were inverted on the air.
_UNINVERT = np.concatenate([np.ones(36), -np.ones(12)])


def from_air_soft(llr49: np.ndarray):
    """49 soft values (positive meaning 0) -> (word, errors A, errors B,
    score), errors -1 where a half did not decode."""
    v = np.asarray(llr49[:48], dtype=np.float64)[_DEINTERLEAVE] * _UNINVERT
    da, ea, sa = golay.decode_soft(v[:24])
    db, eb, sb = golay.decode_soft(v[24:])
    return (da << 12) | db, ea, eb, min(sa, sb)


def from_air_hard(llr49: np.ndarray) -> int:
    """The word as its data bits read, uncorrected: for a DTM data word the
    Golay decoders could not decode, which the block's CRC then judges."""
    v = np.asarray(llr49[:48], dtype=np.float64)[_DEINTERLEAVE]
    bits = (v < 0).astype(int)
    da = bits_int(bits[:12])
    db = bits_int(bits[24:36])
    return (da << 12) | db


# --------------------------------------------------------------------------
# Addresses: A.5.2.4
# --------------------------------------------------------------------------

def check_address(addr: str) -> str:
    """An individual or net address as the standard allows one: 1 to 15
    characters of A-Z and 0-9 (A.5.2.4.4)."""
    a = addr.strip().upper()
    if not 1 <= len(a) <= 15:
        raise ValueError(f"an ALE address is 1 to 15 characters, not {len(a)} ({addr!r})")
    bad = [c for c in a if c not in BASIC38 or c in "@"]
    if bad:
        raise ValueError(f"{addr!r}: {''.join(bad)!r} not allowed in an address "
                         f"(A-Z, 0-9, and ? as a wildcard)")
    return a


def address_words(addr: str, first: int) -> list[int]:
    """The whole address: ``first`` preamble, then DATA, REP, DATA, REP; the
    last word stuffed with "@" (A.5.2.4.3)."""
    a = addr
    if len(a) % 3:
        a += "@" * (3 - len(a) % 3)
    out = []
    for i in range(0, len(a), 3):
        pre = first if i == 0 else (DATA if (i // 3) % 2 else REP)
        out.append(word(pre, a[i:i + 3]))
    return out


def address_of(texts: list[str]) -> str:
    """Words' characters back to an address, stuffing removed. The special
    addresses (AllCall, AnyCall, null, in-link: all start with "@" or "?@")
    are kept whole: their "@"s are the address."""
    a = "".join(texts)
    if a[:1] == "@" or a[:2] == "?@":
        return a
    return a.rstrip("@")


ALLCALL, ANYCALL, NULL = "@?@", "@@?", "@@@"


def is_allcall(a: str) -> bool:
    """Global ("@?@") or selective ("@A@") AllCall (A.5.2.4.7)."""
    return len(a) == 3 and a[0] == "@" and a[2] == "@" and a[1] != "@"


def is_anycall(a: str) -> bool:
    """Global ("@@?"), selective ("@@A") or double selective ("@AB") AnyCall
    (A.5.2.4.8)."""
    return len(a) == 3 and a[0] == "@" and a[2] != "@"


def with_preamble(w: int, preamble: int) -> int:
    return (w & 0x1FFFFF) | ((preamble & 7) << 21)


# --------------------------------------------------------------------------
# Messages: AMD (A.5.7.2) and LQA (A.5.4.2)
# --------------------------------------------------------------------------

def amd_words(text: str) -> list[int]:
    """CMD with the first three characters, then DATA, REP, ...; the last
    word padded with spaces. Lowercase is sent as uppercase; anything else
    outside the expanded 64 set is refused."""
    t = text.upper()
    bad = sorted({c for c in t if c not in EXPANDED64})
    if bad:
        raise ValueError(f"AMD carries the expanded 64 set only: {''.join(bad)!r} is not in it")
    if not t:
        raise ValueError("an empty AMD message")
    if len(t) > 90:
        raise ValueError(f"an AMD message is at most 90 characters (A.5.7.2.3), not {len(t)}")
    if len(t) % 3:
        t += " " * (3 - len(t) % 3)
    out = []
    for i in range(0, len(t), 3):
        pre = CMD if i == 0 else (DATA if (i // 3) % 2 else REP)
        out.append(word(pre, t[i:i + 3]))
    return out


def lqa_word(ber: int = 31, sinad: int = 31, mp: int = 7, request: bool = False) -> int:
    """CMD LQA: "a" in C1, KA1 (W11) asks for a report back, MP (W12-14),
    SINAD (W15-19), BER (W20-24)."""
    bits = (ord("a") << 14) | (int(request) << 13) | ((mp & 7) << 10) \
        | ((sinad & 31) << 5) | (ber & 31)
    return word(CMD, bits=bits)


def ber_code(average_votes: float | None) -> int:
    """Table A-XIII: the average of non-unanimous votes, rounded, 30 for 30 or
    more, 31 for no value."""
    if average_votes is None:
        return 31
    return int(min(30, max(0, round(average_votes))))


def sinad_code(db: float | None) -> int:
    """A.5.4.2.2: whole dB, 0 for 0 dB or less, 30 at most, 31 for none."""
    if db is None:
        return 31
    return int(min(30, max(0, round(db))))


def _cmd(c1: str, c2: str, c3: str | int) -> int:
    third = ord(c3) if isinstance(c3, str) else c3 & 0x7F
    return word(CMD, bits=(ord(c1) << 14) | (ord(c2) << 7) | third)


def version_query(components: int = 0b111, formats: int = 0b0001) -> int:
    """A.5.6.6.1, figure A-43: "v", "s", then KVC3-1 (hardware, operating
    firmware, management firmware) and KVF4-1 (reserved, DBM, DTM, AMD)."""
    return _cmd("v", "s", ((components & 7) << 4) | (formats & 15))


def capabilities_query() -> int:
    """A.5.6.6.2.1, figure A-44: "v", "c", "q"."""
    return _cmd("v", "c", "q")


def capabilities_report(*, scan_rate: int, channels: int, tune_100ms: int,
                        protocols: int = 0, lp_time: int = 0, alqa: int = 0,
                        orderwire: int = 0, scheduling: int = 0) -> list[int]:
    """A.5.6.6.2.2. The text says five DATA words; figure A-45 draws four,
    and the four are what is sent: scan rate (5 bits), channels scanned (8)
    and tune time in 100 ms (8); LP time (6), ALE protocols (7, VAP7 "all
    calls" to VAP1) and ALQA (8); orderwire (8) and 13 reserved; scheduling
    (21). The second and fourth go as REP, as the text asks."""
    clip = lambda v, n: int(min(max(v, 0), (1 << n) - 1))
    d = [
        (clip(scan_rate, 5) << 16) | (clip(channels, 8) << 8) | clip(tune_100ms, 8),
        (clip(lp_time, 6) << 15) | (clip(protocols, 7) << 8) | clip(alqa, 8),
        clip(orderwire, 8) << 13,
        clip(scheduling, 21),
    ]
    return [_cmd("v", "c", "r")] + [word(DATA if i % 2 == 0 else REP, bits=v)
                                    for i, v in enumerate(d)]


#: VAP bits (table A-XXVIII), MSB first: all calls, any calls, AMD, DTM, DBM,
#: DTM capable, DBM capable.
VAP_ALLCALL, VAP_ANYCALL, VAP_AMD = 1 << 6, 1 << 5, 1 << 4


#: Table A-XVI: the CMD functions' first characters, besides the expanded 64
#: set that is AMD's.
CMD_FUNCTIONS = frozenset("`abcdfmnprtvxyz{|~")


def is_known_cmd(w: int) -> bool:
    """A CMD word whose first character names a function table A-XVI lists."""
    return preamble_of(w) == CMD and (is_amd(w) or chars_of(w)[0] in CMD_FUNCTIONS)


def is_amd(w: int) -> bool:
    return all(c in EXPANDED64 for c in chars_of(w))


# --------------------------------------------------------------------------
# Time exchange (A.5.6.4.3 to A.5.6.4.6)
# --------------------------------------------------------------------------

#: Table A-XXII: time quality code -> uncertainty window, seconds (total).
TIME_QUALITY_S = (0.0, 0.020, 0.100, 0.500, 2.0, 10.0, 60.0, float("inf"))


def time_quality(window_s: float) -> int:
    """The code for a total uncertainty window (A.5.6.4.6)."""
    return next(q for q, w in enumerate(TIME_QUALITY_S) if window_s <= w and (q or not window_s))


def time_is(quality: int, seconds: int, ticks: int) -> int:
    """Figure A-38: CMD "~", time quality (3 bits), seconds (6), 40 ms ticks
    (5) — the time at the start of the word after this one."""
    return (CMD << 21) | (ord("~") << 14) | ((quality & 7) << 11) | ((seconds & 0x3F) << 5) \
        | (ticks & 0x1F)


#: A.5.6.4.3.3: a Time Request is a Time Is with quality 7, seconds all
#: ones, ticks 30; A.5.6.4.3.2: a NULL Time Is has ticks all ones too.
TIME_REQUEST = time_is(7, 0x3F, 30)
TIME_NULL = time_is(7, 0x3F, 0x1F)


def is_time(w: int) -> bool:
    return preamble_of(w) == CMD and (w >> 14) & 0x7F == ord("~")


def coarse_time(month: int, day: int, minute: int) -> int:
    """Figure A-39: DATA, a zero, month (4 bits), day (5), minutes since
    midnight (11) — as of the start of this word."""
    return (DATA << 21) | ((month & 15) << 16) | ((day & 31) << 11) | (minute & 0x7FF)


def authenticator(words, previous: int | None = None) -> int:
    """A.5.6.4.5: every word of the time exchange message before the
    authentication word exclusive-or'd, and with the authentication word of
    the message it answers, if any (all 24 bits, preamble included); the
    21 least significant bits, as a REP word."""
    x = 0
    for w in words:
        x ^= w & 0xFFFFFF
    if previous is not None:
        x ^= previous & 0xFFFFFF
    return word(REP, bits=x & 0x1FFFFF)


# --------------------------------------------------------------------------
# CRC (A.5.6.1) and DTM (A.5.7.3)
# --------------------------------------------------------------------------

def crc16(bits) -> int:
    """FED-STD-1003's 16-bit frame check sequence over a bit sequence, first
    bit first: X^16 + X^12 + X^5 + 1, the register preset to ones and the
    remainder complemented. Bit 15 of the result is X15, the first sent."""
    r = 0xFFFF
    for b in bits:
        fb = ((r >> 15) & 1) ^ (int(b) & 1)
        r = (r << 1) & 0xFFFF
        if fb:
            r ^= 0x1021
    return r ^ 0xFFFF


def crc_word(fcs: int) -> int:
    """CMD CRC (table A-XVII): "x" to "{", the FCS's top two bits being the
    first character's last two, X15 in W9 through X0 in W24."""
    return (CMD << 21) | (0b11110 << 16) | (fcs & 0xFFFF)


def is_crc(w: int) -> bool:
    return preamble_of(w) == CMD and (w >> 16) & 0x1F == 0b11110


#: Table A-XXXII's DTM data codes.
DTM_NULL, DTM_ARQ = 0, 352
DTM_EXTENDED_MAX = 351                 # words
DTM_BASIC_MAX = 31                     # words, 651 bits


def dtm_word(dc: int, *, arq: bool = False, ascii: bool = False, kd2: int = 0,
             kd1: int = 0) -> int:
    """CMD DTM (table A-XXXIII): "d", the control bits KD4 (ARQ; for a DTM
    ARQ, NAK), KD3 (ASCII; for a DTM ARQ, pause), KD2 (message), KD1
    (sequence), and the 10-bit data code."""
    kd = (int(arq) << 3) | (int(ascii) << 2) | ((kd2 & 1) << 1) | (kd1 & 1)
    return (CMD << 21) | (ord("d") << 14) | (kd << 10) | (dc & 0x3FF)


def is_dtm(w: int) -> bool:
    return preamble_of(w) == CMD and (w >> 14) & 0x7F == ord("d")


def dtm_fields(w: int) -> dict:
    """A CMD DTM word's kind, control bits, and for a data block its length
    in words and in data bits."""
    kd, dc = (w >> 10) & 15, w & 0x3FF
    out = {"kd4": kd >> 3, "kd3": (kd >> 2) & 1, "kd2": (kd >> 1) & 1, "kd1": kd & 1,
           "dc": dc, "words": 0, "bits": 0}
    if dc == DTM_NULL:
        out["kind"] = "null"
    elif dc <= DTM_EXTENDED_MAX:
        out.update(kind="extended", words=dc, bits=21 * dc)
    elif dc == DTM_ARQ:
        out["kind"] = "arq"
    else:
        m, p = dc >> 5, dc & 31
        if p == 0 or m < 11:
            out["kind"] = "reserved"           # 32m, and ARQ's neighbours
        else:
            out.update(kind="basic", words=p, bits=21 * (p - 1) + m - 10)
    return out


def _bits_of(data: bytes | str, ascii: bool) -> list[int]:
    if ascii:
        text = data if isinstance(data, str) else data.decode("ascii")
        if any(ord(c) > 0x7F for c in text):
            raise ValueError("DTM ASCII is 7-bit")
        return [(ord(c) >> (6 - i)) & 1 for c in text for i in range(7)]
    raw = data.encode() if isinstance(data, str) else bytes(data)
    return [(b >> (7 - i)) & 1 for b in raw for i in range(8)]


def _data_words(bits: list[int]) -> list[int]:
    """21 bits a word, DATA and REP alternating (figure A-47)."""
    return [word(DATA if k % 2 == 0 else REP,
                 bits=bits_int(bits[21 * k:21 * k + 21]))
            for k in range(len(bits) // 21)]


def dtm_blocks(data: bytes | str, *, ascii: bool | None = None, arq: bool = False,
               kd2: int = 0, kd1: int = 0, max_words: int = DTM_EXTENDED_MAX) -> list[list[int]]:
    """A message as DTM data blocks, each CMD DTM, its data words and CMD CRC.
    EXTENDED blocks as large as fill exactly (at most ``max_words``), and a
    BASIC block for the rest (A.5.7.3). The message always ends with a BASIC
    block, so a receiver knows it has the whole of it. KD2 is the message's,
    KD1 alternates block to block starting at ``kd1``. ``ascii`` defaults to
    whether ``data`` is a str: 7-bit characters, stuffed with DEL; bytes go
    as binary bits, stuffed with ones. The CRC covers these 21-bit fields,
    excluding command and preamble bits. This reading is unverified against
    a foreign implementation and does not reproduce figure A-48; see ALE.md."""
    if not 1 <= max_words <= DTM_EXTENDED_MAX:
        raise ValueError("DTM max_words must be 1 to 351")
    ascii = isinstance(data, str) if ascii is None else ascii
    bits = _bits_of(data, ascii)
    if not bits:
        raise ValueError("a DTM carries at least one bit")
    blocks, seq = [], kd1
    while bits:
        if len(bits) > 21 * min(DTM_BASIC_MAX, max_words):
            n = min(max_words, len(bits) // 21)
            if 21 * n == len(bits):
                n -= 1                           # leave the end to a BASIC block
            if n < 1:
                n = 1
            chunk, bits = bits[:21 * n], bits[21 * n:]
            dc = n
        else:
            chunk, bits = bits, []
            p = (len(chunk) + 20) // 21
            m = len(chunk) - 21 * (p - 1) + 10
            dc = 32 * m + p
            chunk = chunk + [1] * (21 * p - len(chunk))    # DEL, or ones
        blocks.append([dtm_word(dc, arq=arq, ascii=ascii, kd2=kd2, kd1=seq)]
                      + _data_words(chunk) + [crc_word(crc16(chunk))])
        seq ^= 1
    return blocks


def dtm_arq(*, nak: bool = False, pause: bool = False, kd2: int = 0, kd1: int = 0) -> int:
    """CMD DTM ARQ: the block it answers named by its KD2 and KD1."""
    return dtm_word(DTM_ARQ, arq=nak, ascii=pause, kd2=kd2, kd1=kd1)


def dtm_null(*, arq: bool = True, ascii: bool = False, kd2: int = 0, kd1: int = 0) -> int:
    """CMD DTM NULL: an interrogation (with ARQ), a break, or the end of the
    DTM exchange."""
    return dtm_word(DTM_NULL, arq=arq, ascii=ascii, kd2=kd2, kd1=kd1)


@dataclass
class Dtm:
    """One CMD DTM read from a frame, and for a data block its contents."""

    kind: str                            # basic, extended, null, arq, reserved
    kd4: int
    kd3: int
    kd2: int
    kd1: int
    words: list[int] = field(default_factory=list)     # the data words as read
    damaged: list[int] = field(default_factory=list)   # their indices read badly
    crc: int | None = None                             # the FCS sent
    bits: int = 0

    @property
    def ascii(self) -> bool:
        return bool(self.kd3)

    @property
    def size(self) -> int:
        return len(self.words)

    def arq_word(self, *, nak: bool = False, pause: bool = False) -> int:
        return dtm_arq(nak=nak, pause=pause, kd2=self.kd2, kd1=self.kd1)

    @property
    def data_bits(self) -> list[int]:
        out = [(w >> (20 - i)) & 1 for w in self.words for i in range(21)]
        return out[:self.bits]

    @property
    def ok(self) -> bool:
        if self.crc is None or len(self.words) * 21 < self.bits:
            return False
        full = [(w >> (20 - i)) & 1 for w in self.words for i in range(21)]
        return crc16(full) == self.crc

    @property
    def data(self) -> bytes | str:
        b = self.data_bits
        if self.ascii:
            return "".join(chr(bits_int(b[i:i + 7]))
                           for i in range(0, len(b) - 6, 7))
        return bytes(bits_int(b[i:i + 8]) for i in range(0, len(b) - 7, 8))

    def combined(self, other: "Dtm") -> "Dtm":
        """Figure A-48's overlay: word by word, a word read well in either."""
        ws = [o if (i in self.damaged and i not in other.damaged) else w
              for i, (w, o) in enumerate(zip(self.words, other.words))]
        bad = [i for i in self.damaged if i in other.damaged]
        crc = self.crc if self.crc is not None else other.crc
        return Dtm(self.kind, self.kd4, self.kd3, self.kd2, self.kd1, ws, bad, crc, self.bits)


# --------------------------------------------------------------------------
# DBM (A.5.7.4)
# --------------------------------------------------------------------------

#: Table A-XXXIV's block codes.
DBM_NULL, DBM_ARQ = 0, 1021
DBM_EXTENDED_MAX = 445                  # x 49 interleaver depth
DBM_BASIC = 448                         # + the data bits, 0 to 572
DBM_BASIC_BITS = 12 * 49 - 16           # 572


def dbm_word(bc: int, *, arq: bool = False, ascii: bool = False, kb2: int = 0,
             kb1: int = 0) -> int:
    """CMD DBM (table A-XXXV): "b", KB4 (ARQ; for a DBM ARQ, NAK), KB3 (ASCII;
    for a DBM ARQ, pause), KB2 (message), KB1 (sequence), the block code."""
    kb = (int(arq) << 3) | (int(ascii) << 2) | ((kb2 & 1) << 1) | (kb1 & 1)
    return (CMD << 21) | (ord("b") << 14) | (kb << 10) | (bc & 0x3FF)


def is_dbm(w: int) -> bool:
    return preamble_of(w) == CMD and (w >> 14) & 0x7F == ord("b")


def dbm_fields(w: int) -> dict:
    """A CMD DBM word's kind, control bits (named as DTM's, kd4 to kd1), and
    for a data block its interleaver depth and data bits."""
    kb, bc = (w >> 10) & 15, w & 0x3FF
    out = {"kd4": kb >> 3, "kd3": (kb >> 2) & 1, "kd2": (kb >> 1) & 1, "kd1": kb & 1,
           "bc": bc, "depth": 0, "bits": 0}
    if bc == DBM_NULL:
        out["kind"] = "null"
    elif bc <= DBM_EXTENDED_MAX:
        out.update(kind="extended", depth=49 * bc, bits=12 * 49 * bc - 16)
    elif DBM_BASIC <= bc <= DBM_BASIC + DBM_BASIC_BITS:
        out.update(kind="basic", depth=49, bits=bc - DBM_BASIC)
    elif bc == DBM_ARQ:
        out["kind"] = "arq"
    else:
        out["kind"] = "reserved"
    return out


def dbm_interleave(field_bits) -> np.ndarray:
    """Figure A-50: the data field's 12-bit groups Golay encoded into rows
    (data, then check bits), the matrix read out column by column. The
    depth is the number of rows."""
    f = np.asarray(field_bits, dtype=np.int64).reshape(-1, 12)
    data = f @ (1 << np.arange(11, -1, -1))
    rows = np.array([golay.encode(int(d)) for d in data], dtype=np.int64)
    matrix = (rows[:, None] >> np.arange(23, -1, -1)) & 1          # depth x 24
    return matrix.T.reshape(-1).astype(np.uint8)


def dbm_deinterleave(llr, depth: int) -> tuple[list[int], list[int], np.ndarray]:
    """The block's soft bits as sent (positive meaning 0) -> (the data
    field's bits, the rows that did not decode, the rows' soft values)."""
    rows = np.asarray(llr, dtype=np.float64).reshape(24, depth).T
    bits: list[int] = []
    bad = []
    for r, v in enumerate(rows):
        d, e, _ = golay.decode_soft(v)
        if e < 0:
            bad.append(r)
        bits += [(d >> (11 - i)) & 1 for i in range(12)]
    return bits, bad, rows


def dbm_blocks(data: bytes | str, *, ascii: bool | None = None, arq: bool = False,
               kb2: int = 0, kb1: int = 0, max_depth: int = 49 * DBM_EXTENDED_MAX) -> list[list]:
    """A message as DBM data blocks, each [CMD DBM, the interleaved bits]:
    EXTENDED blocks as deep as the message fills (at most ``max_depth``
    rows), then a BASIC block with the rest, so that the message always ends
    with one. ASCII characters are not split between blocks."""
    ascii = isinstance(data, str) if ascii is None else ascii
    bits = _bits_of(data, ascii)
    if not bits:
        raise ValueError("a DBM carries at least one bit")
    unit = 7 if ascii else 1
    blocks, seq = [], kb1
    while bits:
        if len(bits) > DBM_BASIC_BITS:
            n = min(max_depth // 49, (len(bits) + 16) // 588)
            room = (12 * 49 * n - 16) // unit * unit
            if room >= len(bits):
                n -= 1
                room = (12 * 49 * n - 16) // unit * unit
            if n < 1:
                n, room = 1, DBM_BASIC_BITS // unit * unit
            chunk, bits = bits[:room], bits[room:]
            bc, depth = n, 49 * n
        else:
            chunk, bits = bits, []
            bc, depth = DBM_BASIC + len(chunk), 49
        m = 12 * depth
        fld = chunk + [1] * (m - 16 - len(chunk))
        fcs = crc16(fld)
        fld += [(fcs >> (15 - i)) & 1 for i in range(16)]
        blocks.append([dbm_word(bc, arq=arq, ascii=ascii, kb2=kb2, kb1=seq),
                       dbm_interleave(fld)])
        seq ^= 1
    return blocks


def dbm_arq(*, nak: bool = False, pause: bool = False, kb2: int = 0, kb1: int = 0) -> int:
    return dbm_word(DBM_ARQ, arq=nak, ascii=pause, kb2=kb2, kb1=kb1)



@dataclass
class Dbm:
    """One CMD DBM read from a frame, and for a data block what the receiver
    made of it: its rows' soft values, kept so that a later copy can be
    added to them."""

    kind: str
    kd4: int
    kd3: int
    kd2: int
    kd1: int
    depth: int = 0
    bits: int = 0
    rows: np.ndarray | None = None       # depth x 24 soft values
    data_field: list[int] = field(default_factory=list)
    damaged: list[int] = field(default_factory=list)   # rows that did not decode

    @property
    def ascii(self) -> bool:
        return bool(self.kd3)

    @property
    def size(self) -> int:
        return self.depth

    @property
    def ok(self) -> bool:
        m = 12 * self.depth
        if not self.depth or len(self.data_field) != m:
            return False
        fcs = bits_int(self.data_field[m - 16:])
        return crc16(self.data_field[:m - 16]) == fcs

    @property
    def data_bits(self) -> list[int]:
        b = self.data_field[:self.bits]
        return b[:len(b) // 7 * 7] if self.ascii else b

    @property
    def data(self) -> bytes | str:
        b = self.data_bits
        if self.ascii:
            return "".join(chr(bits_int(b[i:i + 7])) for i in range(0, len(b), 7))
        return bytes(bits_int(b[i:i + 8]) for i in range(0, len(b) - 7, 8))

    def decode(self, llr) -> "Dbm":
        self.data_field, self.damaged, self.rows = dbm_deinterleave(llr, self.depth)
        return self

    def combined(self, other: "Dbm") -> "Dbm":
        """Two copies' soft values added, and decoded again: a row faded in
        one is read from the other, and a row weak in both from the two."""
        if self.rows is None or other.rows is None or self.rows.shape != other.rows.shape:
            return other
        out = Dbm(self.kind, self.kd4, self.kd3, self.kd2, self.kd1, self.depth, self.bits)
        return out.decode((self.rows + other.rows).T.reshape(-1))

    def arq_word(self, *, nak: bool = False, pause: bool = False) -> int:
        return dbm_arq(nak=nak, pause=pause, kb2=self.kd2, kb1=self.kd1)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def leading_call(called: list[str]) -> list[int]:
    """The whole addresses, twice (A.5.5.4.3.2): TO starts each, except that a
    TO that would follow a different TO word is sent as REP — a word that
    changes changes its preamble (A.5.2.3.4.2), while a one-word address
    repeated, "TO JOE, TO JOE", is the same word twice."""
    out: list[int] = []
    for a in called + called:
        ws = address_words(a, TO)
        if out and preamble_of(out[-1]) == TO and out[-1] != ws[0]:
            ws[0] = with_preamble(ws[0], REP)
        out += ws
    return out


def scanning_call(called: list[str], n: int) -> list[int]:
    """``n`` words of first address words (A.5.2.5.1). One address (or a net)
    repeats its first word in TO; a group rotates its members' different
    first words, at most five, alternating THRU and REP (A.5.5.4.3.1).
    This group reading is unverified against a foreign implementation;
    ALE.md describes the required group-call recording."""
    if n <= 0:
        return []
    firsts = list(dict.fromkeys(address_words(a, TO)[0] for a in called))
    if len(called) == 1:
        return [firsts[0]] * n
    return [with_preamble(firsts[k % len(firsts)], THRU if k % 2 == 0 else REP)
            for k in range(n)]


def call(called: str | list[str], caller: str, *, scan_words: int = 0,
         message: list[int] = (), terminator: int = TIS) -> list[int]:
    """A call, response or acknowledgment — they are the same frame (figures
    A-29 to A-31) — to one address (individual or net) or several (a group,
    A.5.5.4.3). ``scan_words`` of first words go ahead of the leading call to
    capture scanning receivers: 2 per channel they scan (A.5.5.3.1). The
    leading call is the whole addresses twice."""
    group = [called] if isinstance(called, str) else list(called)
    if len(group) > 1:
        words = sum(len(address_words(a, TO)) for a in group)
        firsts = {address_words(a, TO)[0] for a in group}
        if words > 12 or len(firsts) > 5:
            raise ValueError("a group call is at most 12 address words with at most 5 "
                             "different first words (A.5.5.4.3)")
    return scanning_call(group, scan_words) + leading_call(group) + list(message) \
        + address_words(caller, terminator)


def sound(caller: str, *, repeats: int = 2, terminator: int = TIS) -> list[int]:
    """A sound: the sender's whole address, over and over (A.5.3)."""
    return address_words(caller, terminator) * repeats


@dataclass
class Frame:
    """What a received frame said."""

    to: list[str] = field(default_factory=list)       # addresses called
    caller: str = ""                                   # from the conclusion
    terminator: int = -1                               # TIS or TWAS; -1 if none
    amd: list[str] = field(default_factory=list)
    lqa: list[dict] = field(default_factory=list)
    other: list[int] = field(default_factory=list)     # CMDs not understood
    version_query: tuple[int, int] | None = None       # (KVC, KVF) if asked
    capabilities_query: bool = False
    capabilities: dict | None = None                   # a report received
    do_not_respond: bool = False
    dtm: list[Dtm] = field(default_factory=list)      # DTM words and blocks
    dbm: list[Dbm] = field(default_factory=list)      # DBM words (the receiver fills blocks in)
    crc_ok: bool | None = None                         # a CMD CRC outside DTM, checked
    #: A time exchange (A.5.6.4.3): the command, coarse time and
    #: authentication words as received, and their fields.
    time: dict | None = None
    scanning: bool = False                             # had a scanning call
    words: int = 0

    @property
    def sound(self) -> bool:
        return not self.to and self.terminator >= 0

    @property
    def complete(self) -> bool:
        return self.terminator >= 0 and bool(self.caller)


def _calling_cycle(cc: list[int]) -> tuple[list[str], bool]:
    """The calling cycle's words -> (addresses called, in order; whether it had
    a scanning call). A REP straight after a TO starts another address (a
    group's leading call); after a DATA it continues the address; after THRU
    it is another first word of a group's scanning call."""
    inst: list[list[str]] = []
    scan: list[str] = []
    cur: list[str] | None = None
    prev = ""
    for w in cc:
        p, t = preamble_of(w), chars_of(w)
        if p == THRU or (p == REP and prev in ("THRU", "SCAN")):
            scan.append(t)
            prev = "THRU" if p == THRU else "SCAN"
        elif p == TO or (p == REP and prev == "TO"):
            if cur is not None:
                inst.append(cur)
            cur = [t]
            prev = "TO" if p == TO else "NEW"
        elif p == DATA and cur is not None and prev in ("TO", "NEW", "EXT"):
            cur.append(t)
            prev = "DATA"
        elif p == REP and cur is not None and prev == "DATA":
            cur.append(t)
            prev = "EXT"
        else:
            prev = "?"                          # invalid here: skipped
    if cur is not None:
        inst.append(cur)
    addrs = [address_of(x) for x in inst]
    # The leading call is the whole list twice: find it from the end.
    for k in range(1, len(addrs) // 2 + 1):
        if addrs[-2 * k:-k] == addrs[-k:]:
            return addrs[-k:], bool(scan) or len(addrs) > 2 * k
    # A word missed: every address once, a scanning call's first-word-only
    # instances dropped where the whole address was read.
    out = []
    for a in addrs:
        if a not in out and not any(b != a and b.startswith(a) and len(a) == 3 for b in addrs):
            out.append(a)
    return out, bool(scan) or len(addrs) > len(out)


def parse(ws: list[int], damaged: set[int] | None = None) -> Frame:
    """A received frame's words (one per redundant word, in order) -> what it
    says. Words outside A.5.2.5's valid sequences are dropped, not trusted.
    ``damaged``: indices of words the receiver could not decode, which only a
    DTM data block carries (its CRC judges them)."""
    damaged = damaged or set()
    f = Frame(words=len(ws))
    # The calling cycle runs to the first FROM, CMD, TIS or TWAS.
    start = next((i for i, w in enumerate(ws)
                  if preamble_of(w) in (FROM, CMD, TIS, TWAS)), len(ws))
    f.to, f.scanning = _calling_cycle(ws[:start]) if start else ([], False)
    amd: list[str] | None = None
    skip_data = False
    last_pre = -1
    skip_to = start
    checked_from = start                       # a CMD CRC checks the words since
    for i in range(start, len(ws)):
        if i < skip_to:
            continue
        w = ws[i]
        p, t = preamble_of(w), chars_of(w)
        if p in (DATA, REP):
            if skip_data or (p == last_pre and p in (DATA, REP)):
                pass
            elif amd is not None:
                amd.append(t)
        elif p == FROM:
            pass                                   # quick-ID: the conclusion names the sender
        elif p == CMD:
            skip_data = False
            if amd is not None:
                f.amd.append("".join(amd).rstrip())
                amd = None
            c1, c2, c3 = ((w >> 14) & 0x7F, (w >> 7) & 0x7F, w & 0x7F)
            if is_dtm(w):
                d = dtm_fields(w)
                blk = Dtm(d["kind"], d["kd4"], d["kd3"], d["kd2"], d["kd1"], bits=d["bits"])
                if d["words"]:
                    n = d["words"]
                    blk.words = [x & 0x1FFFFF for x in ws[i + 1:i + 1 + n]]
                    blk.damaged = [k for k in range(n) if i + 1 + k in damaged]
                    j = i + 1 + n
                    if j < len(ws) and is_crc(ws[j]) and j not in damaged:
                        blk.crc = ws[j] & 0xFFFF
                        j += 1
                    skip_to = checked_from = j
                f.dtm.append(blk)
                last_pre = CMD
                continue
            if is_dbm(w):
                d = dbm_fields(w)
                f.dbm.append(Dbm(d["kind"], d["kd4"], d["kd3"], d["kd2"], d["kd1"],
                                 d["depth"], d["bits"]))
                checked_from = i + 1
                last_pre = CMD
                continue
            if is_time(w):
                q, sec, tk = (w >> 11) & 7, (w >> 5) & 0x3F, w & 0x1F
                t = {"cmd": w, "quality": q, "seconds": sec, "ticks": tk,
                     "request": w == TIME_REQUEST, "null": w == TIME_NULL,
                     "coarse": None, "auth": None, "index": i}
                if i + 2 < len(ws) and preamble_of(ws[i + 1]) == DATA \
                        and preamble_of(ws[i + 2]) == REP:
                    c = ws[i + 1]
                    t["coarse"] = {"word": c, "month": (c >> 16) & 15, "day": (c >> 11) & 31,
                                   "minute": c & 0x7FF}
                    t["auth"] = ws[i + 2]
                    skip_to = i + 3
                f.time = t
                last_pre = CMD
                continue
            if is_crc(w):
                bits = [(x >> (23 - k)) & 1 for x in ws[checked_from:i] for k in range(24)]
                f.crc_ok = crc16(bits) == (w & 0xFFFF) and (f.crc_ok is not False)
                checked_from = i + 1
                last_pre = CMD
                continue
            if is_amd(w):
                amd = [t]
            elif (chr(c1), chr(c2)) == ("v", "s"):
                f.version_query = ((c3 >> 4) & 7, c3 & 15)
            elif (chr(c1), chr(c2), chr(c3)) == ("v", "c", "q"):
                f.capabilities_query = True
            elif (chr(c1), chr(c2), chr(c3)) == ("v", "c", "r"):
                d = [w2 & 0x1FFFFF for w2 in ws[i + 1:i + 5]
                     if preamble_of(w2) in (DATA, REP)]
                if len(d) == 4:
                    f.capabilities = {
                        "scan_rate": d[0] >> 16, "channels": (d[0] >> 8) & 0xFF,
                        "tune_100ms": d[0] & 0xFF, "lp_time": d[1] >> 15,
                        "protocols": (d[1] >> 8) & 0x7F, "alqa": d[1] & 0xFF,
                        "orderwire": (d[2] >> 13) & 0xFF, "scheduling": d[3]}
                skip_data = True
            elif (chr(c1), chr(c2)) == ("t", "x"):
                f.do_not_respond = True
            elif chr(c1) == "a":
                d = w & 0x3FFF
                f.lqa.append({"request": bool((d >> 13) & 1), "mp": (d >> 10) & 7,
                              "sinad": (d >> 5) & 31, "ber": d & 31})
            else:
                f.other.append(w)
        elif p in (TIS, TWAS):
            if amd is not None:
                f.amd.append("".join(amd).rstrip())
                amd = None
            f.terminator = p
            cur = [t]
            # The address runs until anything else — including the
            # terminator again, which is a sound starting over.
            for w2 in ws[i + 1:]:
                if preamble_of(w2) not in (DATA, REP) or len(cur) == 5:
                    break
                cur.append(chars_of(w2))
            f.caller = address_of(cur)
            break
        last_pre = p
    return f
