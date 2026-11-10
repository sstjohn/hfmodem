# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 39-tone parallel mode (MIL-STD-188-110B appendix B).

39 data tones 56.25 Hz apart, 675 to 2812.5 Hz, each keyed in quadrature
differential phase (QDPSK) once a 22.5 ms signal element, and a 40th,
unmodulated, at 393.75 Hz for the Doppler correction; 75 to 2400 bit/s,
Reed-Solomon coded, block interleaved, with in-band diversity below 1200.

**The text, and what a real modem settled.** Written from appendix B. Where
the text leaves a choice open, a published recording of a 39-tone modem
(a test message, 600 bit/s) decided it — used as a partner to test
against, not in the tree:

- each tone's phase is reckoned from the onset of its own signal element:
  the phase change of table B-V is between the phases the tones have at
  the starts of consecutive elements, each measured from that element's
  start (the extended preamble's part three says as much);
- the framing register of figure B-5 is read with its labels: each bit is
  the exclusive-or of the ones 2, 3, 5 and 9 before it;
- of a dibit, the earlier bit is the odd bit location of table B-IV;
- a Reed-Solomon symbol's first bit sent is its most significant, and a
  codeword's data symbols are its highest-degree coefficients.

**The receiver** finds part one's tones, takes the timing from which tones
each element holds, centres it in the guard (``settle``), and puts the data's
start where the framing opens it (``align``): senders lay out the preamble's
parts differently. The carrier offset comes from part one; the sender's
clock from the whole message (``drift``), or for gross errors from part one
(``clock_error``). (7,3) words are decoded by weighing every code word
against the soft bits, (14,10) words by erasing their least reliable
nibbles. See docs/protocols/bateleur/TONE39.md.

**Messages.** A byte message is sent in the asynchronous mode (B.5.7) with
ten-bit characters (a start bit, eight data bits least significant first,
a stop bit; start SPACE, stop MARK), ended by EOM characters (all
MARK); the receiver drops null characters and stops at the EOM.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, firwin

from . import rs16
from .spec import FS

T = 216                                   # samples a 22.5 ms signal element
GUARD = 45                                # the start of an element is not read
W = T - GUARD                             # the window read: 17.8125 ms, 56.14035 Hz resolution
DOPPLER_HZ = 393.75
TONE_HZ = 675.0 + 56.25 * np.arange(39)

#: Table B-I.
DEGREES = {75: (1, 4, 12, 36), 150: (1, 9, 25, 81), 300: (1, 17, 47, 153),
           600: (1, 33, 99, 297), 1200: (1, 63, 189, 567),
           2400: (1, 9, 18, 27, 36, 72, 144, 288)}

#: Table B-III (synchronous): (rate, degree) -> (super blocks between
#: framing insertions, framing sequence bits).
FRAMING_SYNC = {
    (75, 1): (567, 252), (75, 4): (234, 416), (75, 12): (75, 400), (75, 36): (16, 256),
    (150, 1): (576, 256), (150, 9): (100, 400), (150, 25): (36, 400), (150, 81): (8, 288),
    (300, 1): (567, 252), (300, 17): (54, 408), (300, 47): (18, 376), (300, 153): (4, 272),
    (600, 1): (567, 252), (600, 33): (30, 440), (600, 99): (10, 440), (600, 297): (2, 264),
    (1200, 1): (567, 252), (1200, 63): (14, 392), (1200, 189): (6, 504),
    (1200, 567): (1, 252),
    (2400, 1): (144, 256), (2400, 9): (16, 256), (2400, 18): (12, 384), (2400, 27): (9, 432),
    (2400, 36): (7, 448), (2400, 72): (3, 384), (2400, 144): (1, 256), (2400, 288): (1, 512),
}

#: Tables B-IX to B-XI, the asynchronous mode: "rate, character length,
#: degree, super blocks between framing insertions, fill bits, framing bits".
#: (Table B-XI prints 587 for the 1200 bit/s, 7-bit, least interleaved row;
#: its bits encoded, 6804, are 567 code words' worth.)
_ASYNC_ROWS = """
    75 7 1 567 0 252 | 75 7 5 189 0 420 | 75 7 12 84 0 448 | 75 7 35 18 0 280
    75 8 1 576 0 256 | 75 8 4 234 0 416 | 75 8 12 75 0 400 | 75 8 36 16 0 256
    75 9 1 567 0 252 | 75 9 4 252 0 448 | 75 9 12 84 0 448 | 75 9 36 16 0 256
    75 10 1 585 0 260 | 75 10 4 242 6 416 | 75 10 12 75 0 400 | 75 10 35 18 0 280
    75 11 1 594 0 264 | 75 11 4 260 6 448 | 75 11 11 99 0 484 | 75 11 33 18 0 264
    75 12 1 567 0 252 | 75 12 4 261 0 464 | 75 12 12 84 0 448 | 75 12 36 16 0 256
    150 7 1 567 0 252 | 150 7 9 112 0 448 | 150 7 27 35 0 420 | 150 7 81 7 0 252
    150 8 1 576 0 256 | 150 8 9 100 0 400 | 150 8 25 36 0 400 | 150 8 81 8 0 288
    150 9 1 567 0 252 | 150 9 9 112 0 448 | 150 9 25 38 6 408 | 150 9 81 7 0 252
    150 10 1 585 0 260 | 150 10 9 110 0 440 | 150 10 25 36 0 400 | 150 10 75 9 0 300
    150 11 1 594 0 264 | 150 11 9 110 0 440 | 150 11 27 33 0 396 | 150 11 77 9 0 308
    150 12 1 567 0 252 | 150 12 9 110 0 440 | 150 12 27 33 0 396 | 150 12 81 7 0 252
    300 7 1 567 0 252 | 300 7 15 63 0 420 | 300 7 49 18 0 392 | 300 7 145 5 6 308
    300 8 1 576 0 256 | 300 8 17 54 0 408 | 300 8 47 18 0 376 | 300 8 153 4 0 272
    300 9 1 567 0 252 | 300 9 17 54 0 408 | 300 9 47 18 0 376 | 300 9 153 4 0 272
    300 10 1 585 0 260 | 300 10 17 49 6 356 | 300 10 45 22 0 440 | 300 10 153 5 0 340
    300 11 1 594 0 264 | 300 11 19 43 3 356 | 300 11 45 22 0 440 | 300 11 161 4 6 272
    300 12 1 567 0 252 | 300 12 17 54 0 408 | 300 12 49 18 0 392 | 300 12 153 4 0 272
    600 7 1 567 0 252 | 600 7 35 27 0 420 | 600 7 105 9 0 420 | 600 7 315 2 0 280
    600 8 1 576 0 256 | 600 8 33 30 0 440 | 600 8 99 10 0 440 | 600 8 297 2 0 264
    600 9 1 567 0 252 | 600 9 33 30 0 440 | 600 9 99 10 0 440 | 600 9 297 2 0 264
    600 10 1 585 0 260 | 600 10 33 30 0 440 | 600 10 99 10 0 440 | 600 10 315 2 0 280
    600 11 1 594 0 264 | 600 11 33 30 0 440 | 600 11 99 10 0 440 | 600 11 297 2 0 264
    600 12 1 567 0 252 | 600 12 33 30 0 440 | 600 12 99 10 0 440 | 600 12 297 2 0 264
    1200 7 1 567 0 252 | 1200 7 63 15 0 420 | 1200 7 189 6 0 504 | 1200 7 567 1 0 252
    1200 8 1 576 0 256 | 1200 8 63 14 0 392 | 1200 8 189 6 0 504 | 1200 8 576 1 0 256
    1200 9 1 567 0 252 | 1200 9 63 15 0 420 | 1200 9 189 6 0 504 | 1200 9 567 1 0 252
    1200 10 1 585 0 260 | 1200 10 63 15 0 420 | 1200 10 195 6 0 520 | 1200 10 585 1 0 260
    1200 11 1 594 0 264 | 1200 11 65 16 6 448 | 1200 11 203 5 3 444 | 1200 11 619 1 3 268
    1200 12 1 567 0 252 | 1200 12 63 15 0 420 | 1200 12 189 6 0 504 | 1200 12 567 1 0 252
    2400 7 1 145 4 252 | 2400 7 36 7 0 448 | 2400 7 73 3 3 385 | 2400 7 282 1 3 497
    2400 8 1 144 0 256 | 2400 8 36 7 0 448 | 2400 8 72 3 0 384 | 2400 8 288 1 0 512
    2400 9 1 144 0 256 | 2400 9 36 7 0 448 | 2400 9 72 3 0 384 | 2400 9 288 1 0 512
    2400 10 1 144 0 256 | 2400 10 36 7 0 448 | 2400 10 72 3 0 384 | 2400 10 288 1 0 512
    2400 11 1 151 1 267 | 2400 11 33 9 0 528 | 2400 11 71 3 6 370 | 2400 11 297 1 0 528
    2400 12 1 144 0 256 | 2400 12 36 7 0 448 | 2400 12 72 3 0 384 | 2400 12 288 1 0 512
"""

FRAMING_ASYNC = {}
for _row in _ASYNC_ROWS.replace("|", "\n").split("\n"):
    if _row.strip():
        _r, _c, _d, _sb, _fill, _seq = (int(v) for v in _row.split())
        FRAMING_ASYNC[(_r, _c, _d)] = (_sb, _fill, _seq)

#: Table B-VIII: for each character length, the data bits it can carry
#: (after the start bit; any parity and stop bits follow them).
CHAR_DATA_BITS = {7: (5,), 8: (5, 6), 9: (5, 6, 7), 10: (6, 7, 8), 11: (7, 8), 12: (8,)}

#: Table B-II: (frequency, amplitude, initial phase in degrees) for each
#: preamble part.
PART1 = ((787.5, 3, 0.0), (1462.5, 3, 103.7), (2137.5, 3, 103.7), (2812.5, 3, 0.0))
PART2 = ((1125.0, 4, 0.0), (1800.0, 4, 90.0), (2475.0, 4, 0.0))
PART3_PHASE = np.radians([
    0.0, 5.6, 19.7, 42.2, 73.1, 115.3, 165.9, 225.0, 295.3, 14.1, 101.3, 199.7, 303.8,
    59.1, 185.6, 317.8, 101.3, 253.1, 56.3, 225.0, 45.0, 236.3, 73.1, 281.3, 137.8, 5.6,
    239.1, 123.8, 19.7, 281.3, 194.1, 115.3, 45.0, 345.9, 295.3, 253.1, 222.2, 199.7,
    185.6])
DOPPLER_AMP = 2.0                         # 6 dB over a data tone

#: Table B-V: (earlier bit, later bit) -> phase change.
_DPHI = {(0, 1): 45.0, (0, 0): 135.0, (1, 0): 225.0, (1, 1): 315.0}


@dataclass(frozen=True)
class Mode39:
    rate: int                             # bit/s
    degree: int                           # interleaving degree: code words a super block
    diversity: str = "tf"                 # below 1200: "tf" time/frequency, "f" frequency
    extended: bool = False                # the extended preamble (B.5.4.2)
    #: The asynchronous mode's character length (7-12), or 0 for the
    #: synchronous mode; and how many of its bits are data.
    char: int = 10
    data_bits: int = 8
    #: The framing bits, when not the table's: as a sender working from
    #: another edition's tables sends them, the receiver measuring them.
    #: Its framing is then looked for, not scheduled.
    seq: int = 0

    def __post_init__(self):
        if self.rate not in DEGREES:
            raise ValueError(f"39-tone rates are {sorted(DEGREES)}, not {self.rate}")
        if self.char and self.char not in CHAR_DATA_BITS:
            raise ValueError("a character is 7 to 12 bits, or 0 for the synchronous mode")
        if self.char and self.data_bits not in CHAR_DATA_BITS[self.char]:
            raise ValueError(f"a {self.char}-bit character carries "
                             f"{CHAR_DATA_BITS[self.char]} data bits")
        keys = FRAMING_ASYNC if self.char else FRAMING_SYNC
        key = (self.rate, self.char, self.degree) if self.char else (self.rate, self.degree)
        if key not in keys and not (self.seq and 1 <= self.degree <= max(
                k[-1] for k in list(FRAMING_SYNC) + list(FRAMING_ASYNC) if k[0] == self.rate)):
            opts = sorted(k[-1] for k in keys if k[0] == self.rate and
                          (not self.char or k[1] == self.char))
            raise ValueError(f"at {self.rate} bit/s the interleaving degree is one of {opts}")
        if self.diversity not in ("tf", "f"):
            raise ValueError("diversity is tf (time/frequency) or f (frequency)")

    @property
    def name(self) -> str:
        c = f"A{self.char}.{self.data_bits}" if self.char else "S"
        return f"{self.rate}/{self.degree}/{c}" + ("" if self.diversity == "tf" else "F") \
            + ("X" if self.extended else "") + (f":{self.seq}" if self.seq else "")

    @property
    def n(self) -> int:                    # code word symbols
        return 14 if self.rate == 2400 else 7

    @property
    def k(self) -> int:
        return 10 if self.rate == 2400 else 3

    @property
    def superblock_bits(self) -> int:       # coded
        return self.degree * self.n * 4

    @property
    def superblock_data(self) -> int:       # before coding
        return self.degree * self.k * 4

    @property
    def framing(self) -> tuple[int, int, int]:
        """(super blocks between insertions, fill bits, framing bits); no
        super blocks (0) when the framing is measured, not tabulated."""
        if self.seq:
            return 0, 0, self.seq
        if self.char:
            return FRAMING_ASYNC[(self.rate, self.char, self.degree)]
        sb, seq = FRAMING_SYNC[(self.rate, self.degree)]
        return sb, 0, seq

    @property
    def d(self) -> int:                     # diversity order
        return max(1, 1200 // self.rate)

    @property
    def word_bits(self) -> int:              # new coded bits an element
        return 78 if self.rate == 2400 else 64 // self.d


def mode(name: str | Mode39) -> Mode39:
    """"600/99" (ten-bit characters of eight data bits), "2400/288/A9.7"
    (nine-bit characters of seven), "600/99/S" (synchronous); "F" after it
    for frequency-only diversity, "X" for the extended preamble; ":<n>"
    after all an n-bit framing sequence not the table's; "<rate>L" the
    most interleaving, "<rate>Z" none."""
    if isinstance(name, Mode39):
        return name
    s = name.upper().replace(" ", "")
    seq = 0
    if ":" in s:
        s, n = s.split(":")
        seq = int(n)
    ext = s.endswith("X") or s.endswith("XF") or s.endswith("FX")
    fd = "F" in s.split("/")[-1] if "/" in s else s.rstrip("X").endswith("F")
    s = s.rstrip("XF")
    parts = s.split("/")
    char, data = 10, 8
    if len(parts) == 3:
        c = parts[2]
        if c == "S":
            char, data = 0, 8
        elif c.startswith("A"):
            char, data = (int(v) for v in c[1:].split("."))
        else:
            raise ValueError(f"no 39-tone mode {name!r}")
        parts = parts[:2]
    if len(parts) == 2:
        rate, deg = int(parts[0]), int(parts[1])
    else:
        core = parts[0]
        rate = int(core[:-1])
        keys = [k for k in (FRAMING_ASYNC if char else FRAMING_SYNC)
                if k[0] == rate and (not char or k[1] == char)]
        degs = sorted(k[-1] for k in keys)
        if not degs:
            raise ValueError(f"no 39-tone mode {name!r}")
        if core[-1] == "L":
            deg = degs[-1]
        elif core[-1] == "Z":
            deg = 1
        else:
            raise ValueError(f"no 39-tone mode {name!r}")
    return Mode39(rate, deg, "f" if fd else "tf", ext, char, data, seq)


# --------------------------------------------------------------------------
# The framing sequence (B.5.4.3, figure B-5)
# --------------------------------------------------------------------------

def _mseq() -> np.ndarray:
    """One period, ending with the SPACE that follows nine MARKs."""
    s = [1] * 9
    out = []
    for _ in range(1100):
        b = s[-2] ^ s[-3] ^ s[-5] ^ s[-9]
        s.append(b)
        out.append(b)
    out = np.array(out, dtype=np.uint8)
    for i in range(520, 1100):
        if out[i] == 0 and out[i - 9:i].all():
            return out[i - 510:i + 1]
    raise AssertionError("no run of nine")


_MSEQ = _mseq()


def framing_length(stream: np.ndarray, probe: int = 64) -> int:
    """How long the framing sequence opening ``stream`` (soft bits) is,
    measured, not looked up: where its first bits sit in the sequence says
    how far it is to the end, the SPACE after nine MARKs. 0 if it does not
    open with the sequence."""
    if stream.size < probe:
        return 0
    hard = (stream < 0).astype(np.uint8)
    tile = np.tile(_MSEQ, 3)
    p = _MSEQ.size
    agree = [int(np.sum(tile[o:o + probe] == hard[:probe])) for o in range(p)]
    o = int(np.argmax(agree))
    if agree[o] < 0.85 * probe:
        return 0
    n = p - o
    if n < 200:                                   # every table's is 252 or more
        n += p
    if n > stream.size or np.mean(hard[:n] == framing_bits(n)) < 0.8:
        return 0
    return n


def framing_bits(n: int) -> np.ndarray:
    """The ``n`` framing bits: the sequence's last ``n`` before its end."""
    reps = -(-n // _MSEQ.size) + 1
    return np.tile(_MSEQ, reps)[-n:]


# --------------------------------------------------------------------------
# Coding and interleaving (B.5.2, B.5.3, figures B-2 and B-3)
# --------------------------------------------------------------------------

def _nibbles(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 4).astype(np.int64)
    return b[:, 0] * 8 + b[:, 1] * 4 + b[:, 2] * 2 + b[:, 3]


def _bits(nibbles) -> np.ndarray:
    v = np.asarray(nibbles, dtype=np.int64)
    return np.stack([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1], axis=1).ravel()


def _load_order(m: Mode39) -> np.ndarray:
    """For each data nibble of a super block in input order, its (row,
    code word): code words are filled two at a time, a byte a row, the low
    nibble in the first; an odd last one alone."""
    D, k = m.degree, m.k
    out = []
    for p in range(0, D - D % 2, 2):
        for s in range(k):
            out += [(s, p), (s, p + 1)]
    if D % 2:
        out += [(s, D - 1) for s in range(k)]
    return np.array(out)


def encode_superblock(m: Mode39, data_bits: np.ndarray) -> np.ndarray:
    """``m.superblock_data`` bits -> the super block's coded bits, in the
    order sent (rows in turn, each row across the code words)."""
    D, n, k = m.degree, m.n, m.k
    mat = np.zeros((n, D), dtype=np.int64)
    order = _load_order(m)
    mat[order[:, 0], order[:, 1]] = _nibbles(data_bits)
    for c in range(D):
        mat[:, c] = rs16.encode(list(mat[:k, c]))[: n]
    return _bits(mat.ravel())


_CODEBOOK: np.ndarray | None = None


def _codebook() -> np.ndarray:
    """Every (7,3) code word as 28 signs (+1 for a 0 bit): 4096 of them,
    few enough to hold each received word against all."""
    global _CODEBOOK
    if _CODEBOOK is None:
        words = [rs16.encode([a, b, c])[:7] for a in range(16) for b in range(16)
                 for c in range(16)]
        _CODEBOOK = 1.0 - 2.0 * _bits(np.array(words).ravel()).reshape(4096, 28)
    return _CODEBOOK


def _chase(cw: list[int], rel: np.ndarray, llr: np.ndarray) -> list[int] | None:
    """The (14,10) word nearest the soft bits among those the hard decoder
    finds with erasures of the least reliable nibbles: two at a time
    (and one error), four at a time (and none)."""
    order = [int(i) for i in np.argsort(rel)[:6]]
    tries = [()] + [e for e in itertools.combinations(order, 2)] + \
        [e for e in itertools.combinations(order, 4)]
    best, best_v = None, -np.inf
    for er in tries:
        got = rs16.decode(cw, erasures=er)
        if got is None:
            continue
        v = float(llr @ (1.0 - 2.0 * _bits(got[0])))
        if v > best_v:
            best, best_v = got[0], v
    return best


def decode_superblock(m: Mode39, llr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Soft coded bits (positive: 0) -> (data bits, code words corrected,
    code words beyond what deciding each bit first could correct).

    A (7,3) word is the code word nearest the soft bits, all 4096 of them
    weighed. A (14,10) word that does not check is searched for among
    erasures of its least reliable nibbles (a nibble as sure as its least
    sure bit), the nearest found taken."""
    D, n = m.degree, m.n
    hard = (llr < 0).astype(np.int64)
    nib = _nibbles(hard).reshape(n, D)
    soft = llr.reshape(n, D, 4).transpose(1, 0, 2).reshape(D, n * 4)
    fixed = failed = 0
    if n == 7:
        best = np.argmax(soft @ _codebook().T, axis=1)
        for c in range(D):
            b = int(best[c])
            word = rs16.encode([b >> 8, (b >> 4) & 15, b & 15])[:7]
            dist = int(np.sum(np.array(word) != nib[:, c]))
            if dist:
                fixed += 1
                if dist > 2:
                    failed += 1
            nib[:, c] = word
    else:
        rel = np.abs(soft).reshape(D, n, 4).min(axis=2)
        for c in range(D):
            cw = [int(v) for v in nib[:, c]]
            got = rs16.decode(cw)
            if got is not None and not got[1]:
                continue
            word = _chase(cw, rel[c], soft[c])
            if word is None:
                failed += 1
                continue
            fixed += 1
            if got is None and sum(a != b for a, b in zip(word, cw)) > 2:
                failed += 1
            nib[:, c] = word
    order = _load_order(m)
    return _bits(nib[order[:, 0], order[:, 1]]), fixed, failed


# --------------------------------------------------------------------------
# Characters (B.5.7)
# --------------------------------------------------------------------------

def chars(data: bytes, m: "Mode39", order: str = "lsb") -> np.ndarray:
    """Bytes -> characters of ``m.char`` bits: a start bit, ``m.data_bits``
    data bits least significant first, then source parity when it fits in
    the input byte, and MARK stop bits. Start is SPACE; absent parity is 0."""
    b = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little").reshape(-1, 8)
    b = b[:, :m.data_bits]
    if order == "msb":
        b = b[:, ::-1]
    z0 = np.zeros((b.shape[0], 1), dtype=np.uint8)
    rest = np.ones((b.shape[0], m.char - 1 - m.data_bits), dtype=np.uint8)
    if rest.shape[1] > 1:
        source = np.frombuffer(data, dtype=np.uint8)
        rest[:, 0] = (source >> m.data_bits) & 1
    return np.concatenate([z0, b, rest], axis=1).ravel()


def null_char(m: "Mode39") -> np.ndarray:
    """Data bits SPACE, the rest MARK."""
    c = np.ones(m.char, dtype=np.uint8)
    c[1:1 + m.data_bits] = 0
    return c


# --------------------------------------------------------------------------
# The coded stream: framing, super blocks, fill
# --------------------------------------------------------------------------

def coded_stream(m: Mode39, data: bytes, order: str = "lsb") -> np.ndarray:
    """A message -> every coded bit sent after the preamble."""
    sbs, fill, seq = m.framing
    # Unscheduled framing is sent once, ahead of everything.
    per_interval = sbs * m.superblock_data if sbs else 1 << 62   # source + fill
    src = chars(data, m, order) if m.char else \
        np.unpackbits(np.frombuffer(data, dtype=np.uint8),
                      bitorder="little" if order == "lsb" else "big")
    out = []
    at = 0
    first = True
    while first or at < src.size:
        first = False
        room = per_interval - fill
        take = src[at:at + room]
        at += take.size
        if at >= src.size and m.char:
            # EOM characters (all MARK): at least ten, then to the end of
            # the super block.
            used = fill + take.size
            end = -(-(used + 10 * m.char) // m.superblock_data) * m.superblock_data
            n_eom = max(10, (end - used) // m.char)
            tail = np.ones(n_eom * m.char, dtype=np.uint8)
            take = np.concatenate([take, tail])
            used = fill + take.size
            pad = (-used) % m.superblock_data
            take = np.concatenate([take, np.ones(pad, dtype=np.uint8)])
        bits = np.concatenate([np.zeros(fill, dtype=np.uint8), take])
        if not m.char:
            pad = (-bits.size) % m.superblock_data
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        out.append(framing_bits(seq))
        for i in range(0, bits.size, m.superblock_data):
            out.append(encode_superblock(m, bits[i:i + m.superblock_data]))
    return np.concatenate(out).astype(np.uint8)


# --------------------------------------------------------------------------
# Tones: which coded bits go where (tables B-IV, B-VI, B-VII)
# --------------------------------------------------------------------------

def _groups(m: Mode39) -> list[tuple[range, int]]:
    """Below 1200 bit/s: (tones, how many elements back its word is) for
    each group of tones 1-32."""
    d = m.d
    per = 32 // d
    return [(range(g * per, (g + 1) * per),
             g * (16 // d) if m.diversity == "tf" else 0) for g in range(d)]


def element_dibits(m: Mode39, words: np.ndarray) -> np.ndarray:
    """New coded bits for each element (rows of ``m.word_bits``) -> the
    (earlier, later) bits of each of the 39 tones, each element."""
    n_el = words.shape[0]
    if m.rate == 2400:
        return words.reshape(n_el, 39, 2)
    out = np.zeros((n_el, 39, 2), dtype=np.uint8)
    if m.rate == 1200:
        out[:, :32] = words.reshape(n_el, 32, 2)
    else:
        for tones, back in _groups(m):
            src = np.zeros_like(words)
            if back < n_el:
                src[back:] = words[:n_el - back]
            out[:, tones.start:tones.stop] = src.reshape(n_el, -1, 2)
    out[:, 32:39] = out[:, 0:7]
    return out


def _words(m: Mode39, stream: np.ndarray) -> np.ndarray:
    """The coded stream in element-sized words; with time/frequency
    diversity, elements added until each word's last copy is sent."""
    w = m.word_bits
    n = -(-stream.size // w)
    if m.rate < 1200 and m.diversity == "tf":
        n += (m.d - 1) * (16 // m.d)
    pad = np.zeros(n * w - stream.size, dtype=np.uint8)
    return np.concatenate([stream, pad]).reshape(n, w)


# --------------------------------------------------------------------------
# The modulator (B.5.4, B.5.5)
# --------------------------------------------------------------------------

def _element(freqs, amps, phases, t_on: int) -> np.ndarray:
    t = np.arange(T)
    return np.sum(np.asarray(amps)[:, None] *
                  np.cos(2 * np.pi * np.asarray(freqs)[:, None] * t[None, :] / FS
                         + np.asarray(phases)[:, None]), axis=0)


def preamble(m: Mode39) -> tuple[np.ndarray, np.ndarray]:
    """(audio, the data tones' phases at part three's onset)."""
    n1, n2, n3 = (58, 27, 12) if m.extended else (14, 8, 1)
    out = []
    t = np.arange(n1 * T)
    f1 = np.array([p[0] for p in PART1])
    out.append(np.sum([a * np.cos(2 * np.pi * f * t / FS + np.radians(ph))
                       for f, a, ph in PART1], axis=0))
    f2 = [p[0] for p in PART2]
    a2 = [p[1] for p in PART2]
    ph2 = np.radians([p[2] for p in PART2])
    for i in range(n2):
        out.append(_element(f2, a2, ph2 + np.pi * i, 0))
    del f1
    freqs = np.r_[DOPPLER_HZ, TONE_HZ]
    amps = np.r_[DOPPLER_AMP, np.ones(39)]
    for i in range(n3):
        dop = 2 * np.pi * DOPPLER_HZ * i * T / FS         # the Doppler tone runs on
        out.append(_element(freqs, amps, np.r_[dop, PART3_PHASE], 0))
    return np.concatenate(out), PART3_PHASE.copy()


def coded_length(m: Mode39, n_bytes: int) -> int:
    """How many coded bits ``coded_stream`` sends for ``n_bytes``, without
    coding them: its arithmetic, and nothing else."""
    sbs, fill, seq = m.framing
    per_interval = sbs * m.superblock_data if sbs else 1 << 62
    n_src = n_bytes * (m.char or 8)
    total = at = 0
    first = True
    while first or at < n_src:
        first = False
        take = min(per_interval - fill, n_src - at)
        at += take
        used = fill + take
        if at >= n_src and m.char:
            end = -(-(used + 10 * m.char) // m.superblock_data) * m.superblock_data
            used += max(10, (end - used) // m.char) * m.char
        used += (-used) % m.superblock_data
        total += seq + used // m.superblock_data * m.superblock_bits
    return total


def element_count(m: Mode39 | str, n_bytes: int) -> int:
    """The signal elements a message of ``n_bytes`` keys for, the preamble's
    among them."""
    m = mode(m)
    n = -(-coded_length(m, n_bytes) // m.word_bits)
    if m.rate < 1200 and m.diversity == "tf":
        n += (m.d - 1) * (16 // m.d)
    return n + (97 if m.extended else 23)


_BANDPASS = firwin(401, [340, 2950], fs=FS, pass_zero=False, window=("kaiser", 8.0))


def transmit(data: bytes, m: Mode39 | str = "600L", *,
             order: str = "lsb") -> np.ndarray:
    """A message -> real audio at ``spec.FS``, peak 1.0. ``order`` is
    how bytes become bits: least significant first (the asynchronous
    convention, and B.5.7's for characters) or most."""
    m = mode(m)
    pre, phase = preamble(m)
    di = element_dibits(m, _words(m, coded_stream(m, data, order)))
    dphi = np.radians(np.array([[_DPHI[(int(e), int(l))] for e, l in row] for row in di]))
    n3 = 12 if m.extended else 1
    els = []
    for i in range(dphi.shape[0]):
        phase = phase + dphi[i]
        dop = 2 * np.pi * DOPPLER_HZ * (n3 + i) * T / FS
        els.append(_element(np.r_[DOPPLER_HZ, TONE_HZ], np.r_[DOPPLER_AMP, np.ones(39)],
                            np.r_[dop, phase], 0))
    x = np.concatenate([pre] + els)
    # Short bursts have wider keyed skirts than long random payloads.
    x = fftconvolve(x, _BANDPASS)
    return x / np.max(np.abs(x))


# --------------------------------------------------------------------------
# The receiver
# --------------------------------------------------------------------------

@dataclass
class Reception:
    mode: Mode39
    data: bytes
    eom: bool
    start_sample: int                 # part one's onset
    end_sample: int
    freq_hz: float
    corrected: int                    # code words with symbols corrected
    failed: int                       # code words beyond deciding bit by bit
    clock_ppm: float = 0.0            # the sender's clock against ours
    snr_db: float = float("nan")      # in 3 kHz, over the elements decoded


_P1_HZ = np.array([p[0] for p in PART1])
#: How far off the carrier may be and still be found.
MAX_OFFSET_HZ = 80.0


def _tone_corr(x: np.ndarray, start: int, freqs, df: float) -> np.ndarray:
    """Each tone's complex amplitude over the element starting at
    ``start``, reckoned from that start."""
    seg = x[start + GUARD:start + T]
    t = np.arange(GUARD, T)
    f = np.asarray(freqs) + df
    return np.exp(-2j * np.pi * f[:, None] * t[None, :] / FS) @ seg


_P2_HZ = np.array([p[0] for p in PART2])
#: Part three's (and the data's) tones but those parts one and two use
#: too, so that each part's tones say it is that part and no other.
_P3_HZ = np.array([f for f in np.r_[DOPPLER_HZ, TONE_HZ]
                   if not np.isclose(f, np.r_[_P1_HZ, _P2_HZ]).any()])


def _pattern(extended: bool, n_data: int = 4) -> list[np.ndarray]:
    """Which tones each element from part one's onset holds: part one's
    four, part two's three, then (part three and the data) the rest.
    Part one's and two's are data tones as well, left out of the last."""
    n1, n2, n3 = (58, 27, 12) if extended else (14, 8, 1)
    return [_P1_HZ] * n1 + [_P2_HZ] * n2 + [_P3_HZ] * (n3 + n_data)


def _preamble_match(x: np.ndarray, at: int, df: float, pattern: list[np.ndarray]) -> float:
    """How well the elements from ``at`` hold the tones they should, 0
    (as noise does) to 1. Energy, not phase: across a channel of more than
    one path the tones' phases part company, their energies do not."""
    k = len(pattern)
    if at < 0 or at + k * T > x.size:
        return 0.0
    seg = x[at:at + k * T].reshape(k, T)[:, GUARD:]
    t = np.arange(W)
    e_all = np.sum(seg ** 2, axis=1) + 1e-12
    score = 0.0
    for sel in (_P1_HZ, _P2_HZ, _P3_HZ):
        rows = [i for i, p in enumerate(pattern) if p is sel]
        if not rows:
            continue
        ref = np.exp(-2j * np.pi * (sel[:, None] + df) * t[None, :] / FS)
        share = np.sum(np.abs(seg[rows] @ ref.T) ** 2, axis=1) * 2 / W / e_all[rows]
        q = min(sel.size * 2 / W, 0.9)                # what noise would give
        score += float(np.sum((share - q) / (1 - q)))
    return score / k


#: Part one's share of the energy, averaged over half of it, to be taken
#: for part one: noise gives a few hundredths, data under a tenth.
PART1_SHARE = 0.25
#: The whole preamble's match (``_preamble_match``) to be taken for one.
MATCH = 0.3


def find_preamble(x: np.ndarray, start: int = 0, *, extended: bool = False,
                  threshold: float = PART1_SHARE) -> tuple[int, float] | None:
    """(the sample part three begins at, the carrier offset) of the first
    preamble at or after ``start``, or None.

    Part one's four tones are found by their share of the energy; the
    offset from their phase advance over it; then the timing, from
    which tones each element holds: the edges between the parts, part
    three's forty tones following part two's three. To within the guard;
    ``settle`` then centres it."""
    n1, n2 = (58, 27) if extended else (14, 8)
    n_pre = n1 + n2 + (12 if extended else 1)
    hop = T // 4
    win = T
    t = np.arange(win)
    grid = np.arange(-MAX_OFFSET_HZ, MAX_OFFSET_HZ + 1, 8.0)
    # The four tones at each offset on the grid.
    refs = np.exp(-2j * np.pi * (_P1_HZ[None, :, None] + grid[:, None, None])
                  * t[None, None, :] / FS)                         # (G, 4, win)
    need = n1 * T // hop // 2
    pos = start
    while pos + n_pre * T < x.size:
        seg_end = min(x.size - win, pos + 4 * FS)
        starts = np.arange(pos, seg_end, hop)
        if starts.size <= need:
            return None
        frames = np.stack([x[s:s + win] for s in starts])          # (F, win)
        e_tone = np.sum(np.abs(np.einsum("gkw,fw->fgk", refs, frames)) ** 2, axis=2) * 2 / win
        e_all = np.sum(frames ** 2, axis=1)[:, None] + 1e-12
        share_g = e_tone / e_all                                   # (F, G)
        share = share_g.max(axis=1)
        # Half of part one, on average: fades take single frames under.
        avg = np.convolve(share, np.ones(need) / need, mode="valid")
        hits = np.flatnonzero(avg >= threshold)
        if hits.size == 0:
            if seg_end >= x.size - win:
                return None
            pos = seg_end - n_pre * T
            continue
        i = int(hits[0])
        # Its onset: the first frame up to half its level, the level where
        # the window lies wholly in it.
        level = float(avg[i:i + need + 1].max())
        i = i + int(np.argmax(share[i:i + 2 * need] >= level / 2))
        s0 = int(starts[i])
        g0 = float(grid[int(np.argmax(share_g[i:i + need].sum(axis=0)))])
        # The offset, finely: part one's tones advance 2 pi f T each element.
        ref = np.exp(-2j * np.pi * (_P1_HZ[:, None] + g0) * t[None, :] / FS)
        df = g0
        if s0 + 6 * T < x.size:
            a = np.stack([ref @ x[s0 + i * T:s0 + i * T + win] for i in range(1, 6)])
            adv = np.angle(np.sum(a[1:] * np.conj(a[:-1]), axis=0))
            ideal = 2 * np.pi * (_P1_HZ + g0) * T / FS
            df = g0 + float(np.median(np.angle(np.exp(1j * (adv - ideal)))
                                      * FS / (2 * np.pi * T)))
        # The timing: the whole preamble, around where part one began.
        pattern = _pattern(extended)
        best = (0.0, s0)
        for c in range(s0 - 2 * T, s0 + 4 * T, 6):
            v = _preamble_match(x, c, df, pattern)
            if v > best[0]:
                best = (v, c)
        if best[0] < MATCH:
            pos = s0 + n1 * T
            continue
        return best[1] + (n1 + n2) * T, df
    return None


def find_any_preamble(x, start=0):
    """The first normal or extended preamble: (part three, offset, extended)."""
    found = []
    for extended in (False, True):
        got = find_preamble(x, start, extended=extended)
        if got is not None:
            found.append((*got, extended))
    if not found:
        return None
    return max(found, key=lambda g: _preamble_match(
        x, g[0] - (85 if g[2] else 22) * T, g[1], _pattern(g[2])))


#: Clock errors past this are put right from part one, by resampling;
#: smaller ones from the data (``drift``). Fading moves part one's tones
#: about enough to read as several hundred.
CLOCK_PPM = 1500.0


def clock_error(x: np.ndarray, part3: int, df: float, *, extended: bool = False) -> float:
    """The sending sound card's clock against ours, in parts per million
    (positive: its tones come out high), from how part one's four tones,
    spread over 2 kHz, are each offset: a carrier offset moves them all
    alike, a clock error in proportion to their frequency."""
    n1, n2 = (58, 27) if extended else (14, 8)
    s0 = part3 - (n1 + n2) * T
    if s0 < 0:
        return 0.0
    t = np.arange(W)
    ref = np.exp(-2j * np.pi * (_P1_HZ[:, None] + df) * t[None, :] / FS)
    a = np.stack([ref @ x[s0 + i * T + GUARD:s0 + i * T + T] for i in range(1, n1 - 1)])
    adv = np.angle(np.sum(a[1:] * np.conj(a[:-1]), axis=0))
    ideal = 2 * np.pi * (_P1_HZ + df) * T / FS
    off = np.angle(np.exp(1j * (adv - ideal))) * FS / (2 * np.pi * T)
    slope = np.polyfit(_P1_HZ, off, 1)[0]
    return float(slope * 1e6)


def _reclock(x: np.ndarray, ppm: float) -> np.ndarray:
    """``x`` as our clock would have taken it had the sender's been right."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    r = Fraction(1 + ppm * 1e-6).limit_denominator(4000)
    return resample_poly(x, r.numerator, r.denominator)


def _quality(x: np.ndarray, first: int, df: float, n: int) -> float:
    """How cleanly the changes over ``n`` elements from ``first`` fall on
    the four QDPSK points (1 for perfectly), each weighed by its strength:
    in a fade or a notch a tone says little."""
    if first < 0 or first + (n + 1) * T > x.size:
        return 0.0
    z = np.stack([_tone_corr(x, first + i * T, TONE_HZ, df) for i in range(n + 1)])
    w = z[1:] * np.conj(z[:-1]) * np.exp(-2j * np.pi * df * T / FS)
    a = np.abs(w) + 1e-12
    u = w / a * np.exp(-1j * np.pi / 4)
    return float(np.sum(np.abs(np.sum(u ** 4 * a, axis=1))) / np.sum(a))


def settle(x: np.ndarray, part3: int, df: float, *, extended: bool = False,
           n: int = 40) -> int:
    """``part3`` moved to the middle of the timings the data reads well at.

    Part three's phases fix the timing to a sample, but on whichever path
    was strongest then; a later path's element must not spill into the
    window. The changes from element to element do not care where in the
    guard the window starts, so how cleanly they read is flat across the
    timings every path allows, falling away either side: take the middle,
    between where it has fallen halfway."""
    first = part3 + (12 if extended else 1) * T
    offs = np.arange(-T // 2, T // 2 + 1, 4)
    q = np.array([_quality(x, first + o, df, n) for o in offs])
    if q.max() - np.median(q) < 0.08:
        # Too weak to show: the preamble's timing, already within the guard,
        # will serve.
        return part3
    top = int(np.argmax(q))
    half = (q.max() + q.min()) / 2
    lo = top
    while lo > 0 and q[lo - 1] >= half:
        lo -= 1
    hi = top
    while hi < q.size - 1 and q[hi + 1] >= half:
        hi += 1
    return part3 + int(round((offs[lo] + offs[hi]) / 2))


def _mseq_fit(soft: np.ndarray, probe: int = 96) -> tuple[float, int]:
    """How well the first ``probe`` bits fit somewhere in the framing
    sequence (0.5 for chance, 1 for perfectly), and how many framing bits
    that says there are: from where they fit to the sequence's end."""
    if soft.size < probe:
        return 0.0, 0
    hard = (soft[:probe] < 0).astype(np.uint8)
    tile = np.tile(_MSEQ, 2)
    windows = np.lib.stride_tricks.sliding_window_view(tile, probe)[:_MSEQ.size]
    agree = np.mean(windows == hard, axis=1)
    o = int(np.argmax(agree))
    n = _MSEQ.size - o
    return float(agree[o]), n + _MSEQ.size if n < 200 else n


def _table_seqs(rate: int) -> set[int]:
    return {v[-1] for k, v in list(FRAMING_SYNC.items()) + list(FRAMING_ASYNC.items())
            if k[0] == rate}


def align(x: np.ndarray, part3: int, df: float, *, m: Mode39 | None = None,
          extended: bool = False, reach: int = 2) -> int:
    """``part3`` moved by whole elements to where the data after it opens
    with the framing sequence. Senders differ in how they lay out the
    preamble's parts (one more element here, one fewer there), and the
    energy of the parts can mislead by an element; the framing that opens
    the data cannot. Started an element late, the data still fits (the
    framing runs for several elements), but its framing then reads short:
    the start whose framing is as long as a table says is the one. For a
    sender working from another edition's tables, the parts are trusted
    unless the data fits clearly better elsewhere."""
    n3 = 12 if extended else 1
    cands = [m] if m is not None else all_modes(extended)
    kinds = {}
    for c in cands:
        kinds.setdefault((c.rate, c.diversity), c)
    fits = []
    for k in range(-reach, reach + 1):
        p = part3 + k * T
        best = (0.0, False)
        if p - n3 * T >= 0:
            for c in kinds.values():
                n_el = -(-96 // c.word_bits) + (16 if c.diversity == "tf" else 0)
                if p + (n3 + n_el + 1) * T > x.size:
                    continue
                v, n = _mseq_fit(_combine(c, demodulate(x, p, df, n_el, c)).ravel())
                tabled = n == c.framing[2] if m is not None and not m.seq else \
                    n in _table_seqs(c.rate)
                best = max(best, (v, tabled))
        fits.append((best[0], best[1], p))
    top = max(v for v, _, _ in fits)
    if top < 0.75:
        return part3
    tabled = [p for v, t, p in fits if t and v >= 0.8]
    if tabled:
        return min(tabled, key=lambda p: abs(p - part3))
    if fits[reach][0] >= top - 0.03:
        return part3
    return min((p for v, _, p in fits if v >= top - 0.03), key=lambda p: abs(p - part3))


#: Where nothing is sent: the four places for a tone between the Doppler
#: tone and the first data tone. Orthogonal to every tone over the window,
#: so what is there is noise.
QUIET_HZ = DOPPLER_HZ + 56.25 * np.arange(1, 5)


def read_elements(x: np.ndarray, at: np.ndarray, df: float, freqs=None) -> np.ndarray:
    """Every tone's complex amplitude (the Doppler tone first; or those of
    ``freqs``) over the elements starting at the (fractional) samples
    ``at``: (elements, 40). Each is read from the nearest sample, and its
    phases put right for the fraction left, so that each is reckoned from
    its own start."""
    at = np.asarray(at, dtype=float)
    s = np.rint(at).astype(np.int64)
    f = (np.r_[DOPPLER_HZ, TONE_HZ] if freqs is None else np.asarray(freqs)) + df
    t = np.arange(GUARD, T)
    z = x[s[:, None] + t[None, :]] @ np.exp(-2j * np.pi * t[:, None] * f[None, :] / FS)
    return z * np.exp(-2j * np.pi * (s - at)[:, None] * f[None, :] / FS)


def _read_tones(x: np.ndarray, part3: int, df: float, n_el: int, m: Mode39,
                ppm: float = 0.0, freqs=None) -> np.ndarray:
    """``read_elements`` over part three's last element and ``n_el`` data
    elements after it, fewer if the audio ends. With the sender's clock
    ``ppm`` off, its elements are ``T / (1 + ppm)`` long, and each is read
    where it falls."""
    n3 = 12 if m.extended else 1
    step = T / (1 + ppm * 1e-6)
    at = part3 + (n3 - 1) * T + np.arange(n_el + 1) * step
    at = at[np.rint(at) + T <= x.size]
    if at.size == 0 or at[0] < 0:
        return np.zeros((0, 40), dtype=complex)
    return read_elements(x, at, df, freqs)


def _changes(z: np.ndarray, df: float) -> np.ndarray:
    """Each data tone's change from element to element. Each element is
    reckoned from its own start, so an offset turns every change by 2 pi df
    T: known, and past the fourth power's reach, so taken off here."""
    return z[1:, 1:] * np.conj(z[:-1, 1:]) * np.exp(-2j * np.pi * df * T / FS)


def drift(x: np.ndarray, part3: int, df: float, n_el: int, m: Mode39,
          ppm: float = 0.0) -> float:
    """The sender's clock against ours, in ppm, left over after ``ppm``,
    from the whole message. Read on our clock, a sender's element runs
    early or late by the same few samples more each element, which turns
    each tone's changes by an amount in proportion to its frequency: the
    fourth power takes out the data, and the slope across the tones is the
    clock. Fading and noise average out over the message, as they do not
    over part one."""
    z = _read_tones(x, part3, df, n_el, m, ppm)
    q = _read_tones(x, part3, df, n_el, m, ppm, QUIET_HZ)
    return drift_of(z[:extent(z, q)], df)


def extent(z: np.ndarray, quiet: np.ndarray) -> int:
    """How many of the elements read the transmission is still on, from
    ``read_elements`` of the forty tones and of ``QUIET_HZ`` over the same
    elements: while it is on, the data tones stand well over the empty
    places. Medians, so that a single tone (the Morse identification after
    a message, a carrier) is not taken for it."""
    n = z.shape[0]
    if n < 32:
        return n
    sig = np.median(np.abs(z[:, 1:]) ** 2, axis=1)
    noise = np.median(np.abs(quiet) ** 2, axis=1)
    k = np.ones(32) / 32
    on = np.convolve(sig, k, mode="valid") > 1.5 * np.convolve(noise, k, mode="valid")
    off = np.flatnonzero(~on[16:])
    return n if off.size == 0 else int(off[0]) + 16 + 16


def drift_of(z: np.ndarray, df: float) -> float:
    """``drift`` of elements already read (``read_elements``)."""
    w = _changes(z, df)
    if w.shape[0] < 16:
        return 0.0
    q = np.sum((w * np.exp(-1j * np.pi / 4)) ** 4 / (np.abs(w) ** 3 + 1e-12), axis=0)
    rho = np.angle(q) / 4
    (slope, _), cov = np.polyfit(TONE_HZ, rho, 1, w=np.abs(q) + 1e-12, cov=True)
    k = FS / (2 * np.pi * T) * 1e6
    d, se = float(slope * k), float(np.sqrt(max(cov[0, 0], 0.0)) * k)
    # Only what the tones say clearly: a short message at low SNR through
    # fading can show a slope of several hundred that is not there. And
    # past what a sound card does, past what this reads unaliased, it is
    # not a clock (part one is what finds those).
    return d if abs(d) >= 3 * se and abs(d) <= CLOCK_PPM else 0.0


def snr_db(tones: np.ndarray, quiet: np.ndarray) -> float:
    """The SNR in 3 kHz from ``read_elements`` of the forty tones and of
    ``QUIET_HZ`` over the same elements."""
    p_n = float(np.mean(np.abs(quiet) ** 2))
    p_s = float(np.mean(np.sum(np.abs(tones) ** 2, axis=1))) - tones.shape[1] * p_n
    bins_3k = 3000 / (FS / W)
    return 10 * np.log10(max(p_s, 1e-12) / max(p_n * bins_3k, 1e-12))


def soft_dibits(z: np.ndarray, df: float) -> tuple[np.ndarray, np.ndarray]:
    """``read_elements`` of consecutive elements -> the soft (earlier,
    later) bits of each data tone for all but the first (positive: 0), and
    each element's phase-noise variance."""
    w = _changes(z, df)
    if w.shape[0] == 0:
        return np.zeros((0, 39, 2)), np.zeros(0)
    # Every change is 45 + 90k degrees: what the fourth power leaves is
    # the common rotation an error in the offset left behind.
    rot = np.angle(np.sum((w * np.exp(-1j * np.pi / 4)) ** 4 / (np.abs(w) ** 3 + 1e-12),
                          axis=1)) / 4
    w = w * np.exp(-1j * rot)[:, None]
    # Normalised by the noise the tones' own spread shows.
    a = np.abs(z[:, 1:])
    amp = np.sqrt(a[1:] * a[:-1]) + 1e-12
    u = w / amp
    dev = np.abs(np.angle((u * np.exp(-1j * np.pi / 4)) ** 4)) / 4
    sig2 = np.maximum(np.mean(dev ** 2, axis=1), 1e-3)
    g = amp / np.mean(amp, axis=1, keepdims=True) / sig2[:, None]
    out = np.zeros((w.shape[0], 39, 2))
    # Table B-V: +45 (0,1), +135 (0,0), +225 (1,0), +315 (1,1).
    out[:, :, 0] = np.imag(u) * g           # earlier: 1 when the change is below the axis
    out[:, :, 1] = -np.real(u) * g          # later: 1 when it is to the right
    return out, sig2


def demodulate(x: np.ndarray, part3: int, df: float, n_el: int, m: Mode39,
               ppm: float = 0.0):
    """Soft (earlier, later) bits of each tone for up to ``n_el`` data
    elements (positive: 0)."""
    return soft_dibits(_read_tones(x, part3, df, n_el, m, ppm), df)[0]


def _combine(m: Mode39, soft: np.ndarray) -> np.ndarray:
    """Per-tone soft dibits -> soft words, each element's new word, the
    copies (tones 33-39, and the diversity groups) added in."""
    n_el = soft.shape[0]
    if m.rate == 2400:
        return soft.reshape(n_el, 78)
    s = soft.copy()
    s[:, 0:7] += soft[:, 32:39]
    if m.rate == 1200:
        return s[:, :32].reshape(n_el, 64)
    out = np.zeros((n_el, m.word_bits))
    for tones, back in _groups(m):
        part = s[:, tones.start:tones.stop].reshape(n_el, -1)
        if back:
            out[:n_el - back] += part[back:]
        else:
            out += part
    return out


def all_modes(extended: bool = False) -> list[Mode39]:
    """Every mode, a character length standing for all its data widths."""
    out = []
    for key in FRAMING_ASYNC:
        rate, char, deg = key
        for div in (("tf", "f") if rate < 1200 else ("tf",)):
            out.append(Mode39(rate, deg, div, extended, char, CHAR_DATA_BITS[char][-1]))
    for rate, deg in FRAMING_SYNC:
        for div in (("tf", "f") if rate < 1200 else ("tf",)):
            out.append(Mode39(rate, deg, div, extended, 0))
    return out


def detect_elements(extended: bool = False) -> int:
    """The data elements ``detect`` needs to have seen to weigh every
    mode: each one's framing and first super block, and the copies that
    diversity sends later."""
    need = 0
    for m in all_modes(extended):
        n = -(-(m.framing[2] + m.superblock_bits) // m.word_bits)
        if m.rate < 1200 and m.diversity == "tf":
            n += (m.d - 1) * (16 // m.d)
        need = max(need, n)
    return need + 2


def _text_score(data: bytes) -> float:
    """How much like text: printable ASCII, CR and LF."""
    if not data:
        return 0.0
    return sum(32 <= b < 127 or b in (9, 10, 13) for b in data) / len(data)


def _formed(bits: np.ndarray, m: Mode39) -> float:
    """The share of characters framed as B.5.7 frames them (start, parity
    and stop bits SPACE), or as a sender that leaves them as they were on
    the line does (start SPACE, stop MARK); nulls and EOM aside, 1 for
    nothing but idle."""
    n = bits.size - bits.size % m.char
    c = bits[:n].reshape(-1, m.char)
    if not c.size:
        return 0.0
    null = null_char(m)
    idle = np.all(c == null, axis=1) | np.all(c == 1, axis=1)
    c = c[~idle]
    if not c.size:
        return 1.0 if idle.any() else 0.0
    rest = np.delete(c, np.s_[1:1 + m.data_bits], axis=1)
    spec = float(np.mean(~rest.any(axis=1)))
    line = float(np.mean((c[:, 0] == 0) & (c[:, -1] == 1)))
    return max(spec, line)


def detect(x: np.ndarray, part3: int, df: float, *, extended: bool = False,
           candidates: list[Mode39] | None = None, ppm: float = 0.0) -> Mode39 | None:
    """The mode, which the preamble does not carry.

    The framing sequence opens the data: its length, and the tones it
    arrives on, say the rate and diversity and narrow the degree and the
    character length. The first super block's code words then say the
    degree (the one whose words check), and among the character formats
    left, the one whose characters read most like text is taken, ten-bit
    characters of eight data bits the default."""
    cands = candidates or all_modes(extended)
    n3 = 12 if extended else 1
    n_all = (x.size - part3) // T - n3
    if n_all <= 0:
        return None
    soft_cache: dict = {}

    def soft_for(m):
        key = (m.rate, m.diversity)
        if key not in soft_cache:
            soft_cache[key] = _combine(m, demodulate(x, part3, df, n_all, m, ppm)).ravel()
        return soft_cache[key]

    scored = []
    for m in cands:
        stream = soft_for(m)
        seq = m.framing[2]
        if stream.size < seq:
            continue
        scored.append((float(np.mean((stream[:seq] < 0) == framing_bits(seq))), m))
    if not scored:
        return None
    top = max(a for a, _ in scored)
    if top < 0.8:
        # No tabulated length: a sender working from another edition's
        # tables. Measure the framing on each rate and diversity, and try
        # every degree the rate has behind it.
        seen = set()
        for m in cands:
            if (m.rate, m.diversity) in seen:
                continue
            seen.add((m.rate, m.diversity))
            n = framing_length(soft_for(m))
            if not n:
                continue
            degs = sorted({k[-1] for k in list(FRAMING_ASYNC) + list(FRAMING_SYNC)
                           if k[0] == m.rate})
            for deg in degs:
                for char in (0, *CHAR_DATA_BITS):
                    data = CHAR_DATA_BITS[char][-1] if char else 8
                    scored.append((1.0, Mode39(m.rate, deg, m.diversity, extended, char,
                                               data, seq=n)))
        top = max(a for a, _ in scored)
        if top < 0.8:
            return None
    # The degree: the most code words that check, before correcting.
    by_deg = {}
    for agree, m in scored:
        if agree < top - 0.05:
            continue
        stream = soft_for(m)
        seq = m.framing[2]
        if stream.size < seq + m.superblock_bits:
            continue
        k = (m.rate, m.diversity, m.degree, seq)
        if k not in by_deg:
            hard = (stream[seq:seq + m.superblock_bits] < 0).astype(np.int64)
            nib = _nibbles(hard).reshape(m.n, m.degree)
            corrected = sum(rs16.decode(list(nib[:, c])) is not None for c in range(m.degree))
            by_deg[k] = corrected / m.degree
    if not by_deg:
        return None
    best_k = max(by_deg, key=lambda k: by_deg[k])
    rate, div, deg, seq = best_k
    # The character format: of those left, the one that reads as text.
    fmts = [m for _, m in scored if (m.rate, m.diversity, m.degree, m.framing[2]) == best_k]
    best, best_score = None, -1.0
    for m in fmts:
        widths = CHAR_DATA_BITS.get(m.char, (8,)) if m.char else (8,)
        for w in widths:
            mm = Mode39(m.rate, m.degree, m.diversity, m.extended, m.char, w, m.seq) \
                if m.char else m
            raw: list = []
            data = _decode_stream(soft_for(mm), mm, limit=2, raw=raw)[0]
            # Characters that are framed right say asynchronous; text
            # then says which widths, ten bits of eight the default.
            if mm.char:
                formed = _formed(raw[0], mm)
                shape = 1.0 if formed >= 0.9 else -1.0
            else:
                shape = 0.5
            score = shape + _text_score(data) + \
                (0.01 if (mm.char, mm.data_bits) == (10, 8) else 0)
            if score > best_score:
                best, best_score = mm, score
    return best


class Decoder:
    """Incremental framing, RS super blocks and asynchronous characters."""

    def __init__(self, m: Mode39, order: str = "lsb") -> None:
        self.mode, self.order = m, order
        self.bits = np.zeros(0)
        self.held = np.zeros(0, dtype=np.uint8)
        self.framing, self.j = True, 0
        self.used = self.corrected = self.failed = self.blocks = 0
        self.eom = False

    def push(self, soft, *, limit=None, raw=None) -> bytes:
        m = self.mode
        sbs, fill, seq = m.framing
        self.bits = np.concatenate([self.bits, soft])
        data = bytearray()
        while not self.eom:
            b = self.bits
            if limit is not None and self.blocks >= limit:
                break
            if self.framing:
                if b.size < seq:
                    break
                self.bits = b[seq:]
                self.used += seq
                self.framing, self.j = False, 0
                continue
            if not sbs and self.j:
                n = min(seq, b.size)
                if n >= min(32, m.superblock_bits) and \
                        np.mean((b[:n] < 0) == framing_bits(seq)[:n]) > 0.8:
                    if b.size < seq:
                        break
                    self.framing = True
                    continue
            if b.size < m.superblock_bits:
                break
            out, fixed, failed = decode_superblock(m, b[:m.superblock_bits])
            self.bits = b[m.superblock_bits:]
            self.used += m.superblock_bits
            self.corrected += fixed
            self.failed += failed
            self.blocks += 1
            if self.j == 0:
                out = out[fill:]
            self.j += 1
            self.framing = bool(sbs and self.j == sbs)
            if raw is not None:
                raw.append(out)
            data.extend(self._release(out))
        return bytes(data)

    def _release(self, bits) -> bytes:
        m = self.mode
        self.held = np.concatenate([self.held, bits.astype(np.uint8)])
        width = m.char or 8
        n = self.held.size - self.held.size % width
        if not m.char:
            out = np.packbits(self.held[:n], bitorder="little" if self.order == "lsb"
                             else "big").tobytes()
            self.held = self.held[n:]
            return out
        # One MARK character waits for the next: together they are EOM.
        if n and self.held[n - width:n].all():
            if n < 2 * width or not self.held[n - 2 * width:n - width].all():
                n -= width
        out, self.eom = _unchar(self.held[:n], m, self.order)
        self.held = self.held[n:]
        return out


def _decode_stream(stream: np.ndarray, m: Mode39, order: str = "lsb", limit: int | None = None,
                   raw: list | None = None):
    decoder = Decoder(m, order)
    decoded = [] if raw is not None else None
    data = decoder.push(stream, limit=limit, raw=decoded)
    if raw is not None:
        raw.append(np.concatenate(decoded) if decoded else np.zeros(0, dtype=np.uint8))
    return data, decoder.eom, decoder.used, decoder.corrected, decoder.failed


def receive(audio: np.ndarray, m: Mode39 | str | None = None, *, start: int = 0,
            order: str = "lsb", extended: bool = False) -> Reception | None:
    """The first 39-tone transmission in ``audio`` (real, at ``spec.FS``)
    at or after ``start``, decoded in mode ``m``, or (``None``) in the mode
    ``detect`` finds."""
    x = np.asarray(audio, dtype=np.float64)
    from .stream import Receiver39, receive_whole
    got = find_any_preamble(x, start)
    if got is None:
        return None
    part3, df, ext = got
    ppm, back = clock_error(x, part3, df, extended=ext), int
    if abs(ppm) > CLOCK_PPM:
        y = _reclock(x[start:], ppm)
        again = find_preamble(y, 0, extended=ext)
        n3 = 12 if ext else 1
        if again is not None and _quality(y, again[0] + n3 * T, again[1], 24) > \
                _quality(x, part3 + n3 * T, df, 24):
            x = y
            origin = start
            back = lambda n: origin + round(n / (1 + ppm * 1e-6))
            start = 0
        else:
            ppm = 0.0
    else:
        ppm = 0.0
    rx = Receiver39(m, bit_order=order)
    got = receive_whole(x, rx, start=start, parallel=True)
    if got is None:
        return None
    first, data, end = got
    return Reception(rx.last_mode, data, end.complete, back(first.at), back(end.at),
                     first.freq_hz, rx.last_corrected, rx.last_failed,
                     (1 + ppm * 1e-6) * (1 + rx.last_ppm * 1e-6) * 1e6 - 1e6, end.snr_db)


# --------------------------------------------------------------------------
# Late entry: joining a transmission under way
# --------------------------------------------------------------------------

def _late_offset(x: np.ndarray, start: int) -> float | None:
    """The carrier offset from the Doppler tone, a line 6 dB over any data
    tone in a spectrum of a second or two of the data; None if there is no
    such line where it would be, with the data tones beside it."""
    n = 1 << 14
    seg = x[start:start + n]
    if seg.size < n:
        return None
    s = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    near = np.abs(f - DOPPLER_HZ) <= MAX_OFFSET_HZ
    k = int(np.argmax(np.where(near, s, 0)))
    df = float(f[k] - DOPPLER_HZ)
    # Against its own neighbourhood, which below the first data tone is
    # all but empty: a median over the band would be the data tones' own.
    floor = np.median(s[near]) + 1e-20
    tones = np.abs(f[:, None] - (TONE_HZ[None, :] + df)) < 10
    if s[k] < 30 * floor or np.mean(s[tones.any(axis=1)]) < 3 * floor:
        return None
    return df


def _late_timing(x: np.ndarray, start: int, df: float, n_el: int = 64) -> int:
    """Where the elements lie, from the data alone: a window across an
    element's edge mixes two elements' phases, so the changes it reads sit
    less tightly on the constellation (45 degrees plus a multiple of 90,
    which the fourth power sends to one place). The windows that lie
    wholly inside an element are a run ``GUARD`` long; the middle of the
    best fitting run, round the element."""
    offs = np.arange(0, T, 3)
    coh = []
    for o in offs:
        z = read_elements(x, start + o + np.arange(n_el) * T, df)
        w = _changes(z, df)
        q = np.sum((w * np.exp(-1j * np.pi / 4)) ** 4 / (np.abs(w) ** 3 + 1e-12))
        coh.append(float(np.abs(q) / (np.sum(np.abs(w)) + 1e-12)))
    coh = np.array(coh)
    d = (offs[None, :] - offs[:, None] + T // 2) % T - T // 2
    inside = np.abs(d) <= GUARD // 2 - 4
    fit = [coh[m].mean() - coh[~m].mean() for m in inside]
    return start + int(offs[int(np.argmax(fit))])


def late_entry(x: np.ndarray, m: Mode39 | str, *, start: int = 0,
               order: str = "lsb") -> Reception | None:
    """A transmission in mode ``m`` joined after its preamble: from the
    first framing insertion (B.5.4.3) at or after ``start``.

    The framing sequence is put back into the data every few super blocks
    (table B-III), and from each insertion the data runs on exactly as it
    did from the first, so a receiver that missed the preamble can start at
    the next one. The offset comes from the Doppler tone, the element
    timing from the data itself (``_late_timing``); the framing is looked
    for at every bit of the stream, since in some asynchronous modes an
    insertion begins inside an element. How long that takes is the mode's
    insertion interval: seconds at the higher rates and degrees, minutes at
    75 bit/s with the least interleaving."""
    m = mode(m)
    x = np.asarray(x, dtype=np.float64)
    df = _late_offset(x, start)
    if df is None:
        return None
    at = _late_timing(x, start, df)
    # The element before the first read stands for part three: the changes
    # are what carry the data, so any element serves as their reference.
    n_el = (x.size - at) // T - 1                   # every whole element after the first
    if n_el < 32:
        return None
    stream = _combine(m, soft_dibits(read_elements(x, at + np.arange(n_el + 1) * T, df),
                                     df)[0]).ravel()
    seq = m.framing[2]
    want = 1.0 - 2.0 * framing_bits(seq)             # soft sign of each framing bit
    if stream.size < seq:
        return None
    sign = np.sign(stream)
    score = np.correlate(sign, want, mode="valid") / seq
    hits = np.flatnonzero(score >= 0.8)
    if hits.size == 0:
        return None
    b = int(hits[0])
    b += int(np.argmax(score[b:b + 8]))              # the best of a run of near hits
    data, eom, used, fixed, failed = _decode_stream(stream[b:], m, order)
    el0 = at + (b // m.word_bits) * T
    end = at + (-(-(b + used) // m.word_bits) + 1) * T
    z = read_elements(x, at + np.arange(min(n_el, (end - at) // T)) * T, df)
    q = read_elements(x, at + np.arange(min(n_el, (end - at) // T)) * T, df, QUIET_HZ)
    return Reception(m, data, eom, el0, min(end, x.size), df, fixed, failed,
                     0.0, snr_db(z, q))


def _unchar(bits: np.ndarray, m: Mode39, order: str = "lsb") -> tuple[bytes, bool]:
    """Characters -> bytes (their data bits): nulls dropped, stopping at the
    EOM (two EOM characters running)."""
    n = bits.size - bits.size % m.char
    c = bits[:n].reshape(-1, m.char)
    null = null_char(m)
    out = bytearray()
    for i, ch in enumerate(c):
        if ch.all() and i + 1 < len(c) and c[i + 1].all():
            return bytes(out), True
        if np.array_equal(ch, null):
            continue
        d = ch[1:1 + m.data_bits]
        if order == "msb":
            d = d[::-1]
        out.append(int(sum(int(b) << j for j, b in enumerate(d))))
    return bytes(out), False
