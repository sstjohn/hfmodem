# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Generate VARA HF receiver tables and define measured waveform constants.

`tests/gates/test_tablegen.py` checks the generators against reference vectors.

The table definitions are grouped by method:

  * **Computed from the VB6 `Rnd` stream** (`draw`/`permutation`/`whitener`): the
    whitener, the four interleaver arrays, and the base tables that are just
    draws -- `alloc_col3` and `map2_col3` continue the record-3 stream into the
    gap before record 4 and land exactly on record 4's seed, which is the proof
    they are the same stream and not a lucky reseed, and `bw500_alloc_col2` /
    `bw500_map2_col2` do the same in BW500's record-2 gap. One of those streams
    serves two bandwidths: BW2750's base level takes the *same 395 draws* as
    BW2300's and quantises each to 20 bins instead of 16, so `alloc_col3` is one
    function with two combs and there is no second table to know. The OFDM
    scramble continues the whitener's seed; the OFDM gap cells and tone pilots
    are the rest of their records' gaps, each closing on the next record's seed.
  * **Closed form, no PRNG**: `map1_col3`, `clsparm_gray`, the reference-column
    layout.
  * **Reconstructed to the bit** -- `constellation_lut`, matching VARA's own float
    arithmetic down to its 15-digit `Pi` and the x87 trig its `Sin`/`Cos` compile
    to.
  * **Measured** -- `grid480`, recovered by known-plaintext inversion of our own
    loopback recordings, and `base_bins_col2` with its two neighbours
    `base_bins_col1` and `base_bins_col0`, the whole free ladder below the base
    level, read off recorded VARA overs, and the tone family's grid rows 2-9. All
    are integers read off the air rather than formulae. Nothing derives them, so
    they are written out here as literals and labeled measured; what cannot be
    computed is at least readable.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import cache

import numpy as np


# --------------------------------------------------------------------------- #
# The Visual Basic 6 runtime PRNG, in exact integer arithmetic.
# --------------------------------------------------------------------------- #
# `Rnd` in msvbvm60.dll is a 24-bit LCG:
#     s <- (s*0x43FD43FD + 0xC39EC3) mod 2**24        (full period)
#     Rnd  = Single(s / 2**24)                         -- IEEE binary32
#     draw = Int(Rnd * n)                              -- 0 .. n-1
# The subtlety is the last line. `Rnd * n` is a binary32 multiply: `s / 2**24` is
# exact (a power-of-two divide of a 24-bit integer), but the product rounds to 24
# significant bits round-half-to-even before `Int` floors it. Truncating the exact
# product instead -- `(s*n) >> 24` -- disagrees with the runtime about once per
# 100000 draws, and each disagreement shifts every later value in the column,
# desynchronising 9 of the 68 interleaver columns. The witness is turbo column 7,
# n=2104, s=5613669: shift-and-truncate gives 703, the binary32 multiply 704.
_MULT, _INCR, _MOD = 0x43FD43FD, 0xC39EC3, 1 << 24


def step(s: int) -> int:
    """One LCG advance."""
    return (s * _MULT + _INCR) & (_MOD - 1)


def draw(s: int, n: int) -> int:
    """`Int(Rnd * n)` for state `s`: the binary32 multiply, done in integers.

    `float32(s*n / 2**24) = round24(s*n) / 2**24`, since dividing by a power of
    two only shifts the exponent; so the floored draw is `round24(s*n) >> 24`."""
    p = s * n
    width = p.bit_length()
    if width > 24:
        shift = width - 24
        low = p & ((1 << shift) - 1)
        p >>= shift
        half = 1 << (shift - 1)
        if low > half or (low == half and p & 1):      # round half to even
            p += 1
        p <<= shift
    return p >> 24


def permutation(n: int, seed: int) -> tuple[np.ndarray, int]:
    """Rejection-sampled permutation of 0..n-1, and the state left after it.

    Draw `Int(Rnd * n)` repeatedly, keeping the first appearance of each value. A
    record's stage-2 (turbo) permutation continues the stream from exactly the
    state stage 1 ends on, with no gap."""
    out = np.empty(n, dtype="<i4")
    seen = np.zeros(n, dtype=bool)
    s, k = seed, 0
    a, c = _permutation_jumps()
    while k < n:
        states = (a * np.uint64(s & (_MOD - 1)) + c) & np.uint64(_MOD - 1)
        values = (states.astype(np.float32) * np.float32(n)
                  * np.float32(2.0**-24)).astype(np.int64)
        _, first = np.unique(values, return_index=True)
        take = np.sort(first[~seen[values[first]]])[:n - k]
        out[k:k + len(take)] = values[take]
        seen[values[take]] = True
        k += len(take)
        s = int(states[take[-1] if k == n else -1])
    return out, s


@cache
def _permutation_jumps() -> tuple[np.ndarray, np.ndarray]:
    a = np.empty(8192, dtype=np.uint64)
    c = np.empty(8192, dtype=np.uint64)
    # Affine jumps advance any starting state by j + 1 draws.
    mult, incr = 1, 0
    for j in range(len(a)):
        mult, incr = (mult * _MULT) % _MOD, (incr * _MULT + _INCR) % _MOD
        a[j], c[j] = mult, incr
    return a, c


def whitener(nbits: int = 150000, seed: int = 29318) -> np.ndarray:
    """The additive pre-FEC PN whitener, one bit per byte: `bit = s >> 23`.

    The same 150000-bit sequence whitens BW500 (over its 368 input bits) and
    BW2300 (per coded block); both consume a prefix far shorter than the whole."""
    s = seed
    out = np.empty(nbits, dtype=np.uint8)
    for i in range(nbits):
        s = step(s)
        out[i] = s >> 23
    return out


# FFT bins 0..61 cover every OFDM carrier: BW2300 8..56, BW500 27..37, BW2750 3..61.
_S_BINS, _S_COLS = 62, 409


@cache
def scramble() -> np.ndarray:
    """The OFDM cell scramble `S[bin, column] = Int(Rnd*129)`, (62, 409).

    Drawn row-major from the whitener's own seed: bin k column c is draw
    `1 + 409*k + c`. One table serves every bandwidth and level; a cell goes on
    air as `value * exp(-j*2*pi*S/128)`, which is why the 2*pi/128 lattice shows
    through at every OFDM level."""
    s, out = 29318, np.empty(_S_BINS * _S_COLS, dtype="<i4")
    for i in range(len(out)):
        s = step(s)
        out[i] = draw(s, 129)
    out = out.reshape(_S_BINS, _S_COLS)
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------- #
# Interleavers.
# --------------------------------------------------------------------------- #
# Each record is (stage-1 seed, stage-1 length, stage-2 length). Stage 2 continues
# stage 1's stream. Column c of an array holds record c's permutation.
BW500 = ((12600710,  300,   96), ( 7910701,  612,  200), (14955604,  606,  297),
         ( 6437280,  748,  368), ( 9007905, 1500,  744), (11166657, 2172, 1080),
         ( 8962378, 3168, 1578), (10365048, 3168, 2104), (13750713, 3168, 2524),
         (14737177, 4752, 3160), ( 6898907, 4752, 3792), ( 5141288, 5720, 4755),
         ( 5015752, 7150, 5708), (14390024, 1724,  856), ( 7743493,  204,   64),
         (11842661, 2145, 1704), ( 6808906, 2145, 1704))

BW2300 = ((12600710,   300,    96), ( 7910701,   612,   200), (14955604,   816,   402),
          ( 9947396,  1484,   736), (15597635,  2172,  1080), (12826936,  2832,  1410),
          (11990104,  4164,  2076), ( 4346103,  5520,  2754), ( 3400245,  6860,  3424),
          (15908083, 14406,  7197), ( 8142078, 14406,  9596), ( 3308162, 14406, 11512),
          ( 5060790, 21609, 14398), ( 4014496, 21609, 17276), (15400067, 26068, 21710),
          ( 2656535, 32585, 26056), ( 8801727,  2145,  1704))

# BW2750 is BW2300's chain up to record 9's seed, which at BW2750 opens level 10
# (13 tones) and runs on: level 10's two permutations and its 39 pilot draws,
# then per level 11-17 both permutations and a 59-draw gap (one cell per
# carrier). Every seed from 11548315 on was found on tape and then hit exactly by
# the chain.
BW2750 = BW2300[:9] + (
          (15908083,  8944,  4466), (11548315, 17346,  8667), (12091152, 17346, 11556),
          ( 8109825, 17346, 13864), (15712339, 26019, 17338), (  367472, 26019, 20804),
          ( 2608703, 31388, 26145), (10769080, 39235, 31376))

#: Where a chain's last record's gap lands: no level uses it.
_CHAIN_END = {"bw2300": None, "bw500": None, "bw2750": 4061299}

# Full buffer heights each stage is built into: this many rows by 17 columns, read
# out row-major (column c = record c, strided by 17).
_ROWS = {1: 32585, 2: 26056}

# BW2300 uses the full array; BW500 uses a fixed-length prefix in int32 units.
# BW500 permutations reach row 7149 at most. The remaining entries retain the
# BW2300 values and are outside the BW500 receiver's active range.
_BW500_DUMP = {1: 225000, 2: 150000}


@cache
def _columns(spec: tuple) -> tuple:
    """Per record, the (stage-1, stage-2) permutation pair, stage 2 continuing."""
    out = []
    for seed, n1, n2 in spec:
        p1, s = permutation(n1, seed)
        p2, _ = permutation(n2, s)
        out.append((p1, p2))
    return tuple(out)


def interleaver(bandwidth: str, stage: int) -> np.ndarray:
    """Generate a flat int32 interleaver array, including its inactive tail.

    Build the BW2300 permutations into a 32585x17 (stage 1) or 26056x17
    (stage 2) array. For BW500, replace each column's active prefix with its
    shorter permutation and retain BW2300 values in the remaining rows. BW2750
    has no reference file to match; its array is the same 17-column layout,
    as tall as its longest permutation (39235 at level 17's stage 1), zero below
    each column's end.
    """
    j = stage - 1
    if bandwidth == "bw2750":
        cols = [c[j] for c in _columns(BW2750)]
        grid = np.zeros((max(map(len, cols)), 17), dtype="<i4")
        for c, p in enumerate(cols):
            grid[:len(p), c] = p
        return grid.reshape(-1)
    rows = _ROWS[stage]
    grid = np.zeros((rows, 17), dtype="<i4")
    for c, cols in enumerate(_columns(BW2300)):
        p = cols[j]
        grid[:len(p), c] = p
    if bandwidth == "bw2300":
        return grid.reshape(-1)
    if bandwidth != "bw500":
        raise ValueError(bandwidth)
    for c, cols in enumerate(_columns(BW500)):
        p = cols[j]
        grid[:len(p), c] = p
    return grid.reshape(-1)[:_BW500_DUMP[stage]]


# --------------------------------------------------------------------------- #
# BW2300 base (record-3) bin-placement tables.
# --------------------------------------------------------------------------- #
# Reference columns are deterministic, no PRNG: pilot column k sits at this index.
_REF_POS = np.array([(279 * k + 263) // 17 for k in range(24)])


@cache
def _base_stream_gap() -> int:
    """LCG state at the record-3 -> record-4 gap: after record 3's two permutations.

    Continuing the master interleaver stream to here, the base tables that follow
    are the next 419 draws, and they close exactly on record 4's own seed."""
    _, s = permutation(1484, 9947396)          # BW2300 record-3 stage 1
    _, s = permutation(736, s)                 # record-3 stage 2
    return s


def alloc_col3(span: int = 16, first_bin: int = 9) -> np.ndarray:
    """Per-column bin-allocation offset, `first_bin + Int(Rnd*span)`, 395 draws
    from the gap.

    One stream, two combs. BW2300 quantises each draw to 16 bins from bin 9;
    BW2750 quantises the same 395 draws to 20 bins from bin 6, and that is the
    only field of the base record the wider bandwidth changes. Measured on the
    2026-07-21 BW2750 link-setup over: all 395 columns land on `Int(Rnd*20)`,
    the 24 reference columns pin 24/24 with no plaintext at all, and the same
    table reads seven off-air gateway overs CRC-clean.

    (`9 + (state >> 20)` is `9 + draw(s, 16)`: `s*16` carries no bits below the
    rounding point, so round-half-even never fires at span 16.)
    """
    s = _base_stream_gap()
    out = np.zeros(789, dtype="<i4")
    for i in range(395):
        s = step(s)
        out[2 * i] = first_bin + draw(s, span)
    return out


def map2_col3() -> np.ndarray:
    """Reference-cell class at each of the 24 pilot columns: `Int(Rnd*8)`, stored 2x.

    These 24 draws follow the 395 `alloc` draws in the same stream; the state lands
    on 15597635 after them -- record 4's stage-1 seed -- which is what fixes this
    as a continuation rather than a coincidence."""
    s = _base_stream_gap()
    for _ in range(395):
        s = step(s)
    out = np.zeros(789, dtype="<i4")
    for k in range(24):
        s = step(s)
        out[2 * _REF_POS[k]] = 2 * draw(s, 8)
    if s != 15597635:
        raise ValueError("base stream did not close on record-4 seed")
    return out


def map1_col3() -> np.ndarray:
    """Reference gate: a 1 at each of the 24 pilot columns, 0 elsewhere. No PRNG."""
    out = np.zeros(789, dtype="<i4")
    out[2 * _REF_POS] = 1
    return out


# --------------------------------------------------------------------------- #
# BW2300 record-2 (host BITRATE(3)) bin-placement tables.
# --------------------------------------------------------------------------- #
# Same 24 reference columns as record 3, on 228 emission columns instead of 395.
# The closed form is fitted to the 23 positions three recorded overs agree on --
# each one a column where every over lights the same bin however its payload
# differs -- and it is the only (a, b, d) form of `_REF_POS`'s shape that hits all
# of them. The 24th, column 216, is the one the arithmetic then has to supply.
_REF_POS2 = np.array([(104 * k + 93) // 11 for k in range(24)])


#: The 228 measured base bins, one per record-2 column, in column order.
_REC2_BINS = (
     34,  35,  40,  39,  36,  31,  24,  47,  44,  47,  42,  45,  38,  43,  18,  17,
     48,  37,  36,  25,  20,  39,  40,  19,  36,  17,  40,  21,  24,  17,  32,  25,
     44,  27,  36,  19,  26,  17,  44,  39,  38,  17,  24,  17,  34,  21,  32,  25,
     44,  27,  38,  23,  48,  31,  24,  17,  30,  23,  30,  37,  26,  17,  48,  21,
     36,  31,  18,  19,  36,  33,  36,  23,  42,  37,  22,  47,  32,  39,  30,  19,
     26,  45,  18,  27,  22,  45,  18,  41,  48,  43,  24,  17,  26,  23,  34,  43,
     24,  23,  36,  41,  20,  29,  38,  45,  24,  31,  28,  35,  24,  43,  38,  17,
     38,  29,  34,  35,  48,  25,  20,  31,  32,  31,  46,  23,  20,  33,  32,  43,
     20,  17,  18,  47,  46,  45,  22,  39,  32,  27,  26,  17,  38,  37,  42,  31,
     48,  43,  40,  17,  18,  45,  36,  39,  30,  25,  18,  25,  42,  33,  44,  45,
     38,  17,  18,  21,  42,  23,  28,  39,  44,  45,  18,  29,  24,  35,  48,  19,
     46,  45,  46,  39,  38,  17,  26,  19,  24,  39,  26,  21,  30,  39,  44,  45,
     26,  17,  24,  47,  38,  39,  26,  37,  40,  27,  24,  35,  24,  45,  26,  47,
     18,  27,  38,  35,  44,  47,  34,  23,  20,  33,  28,  47,  18,  29,  36,  17,
     30,  29,  22,  33,
)


def base_bins_col2() -> np.ndarray:
    """Per-column base bin for record 2's 228 columns -- MEASURED, not generated.

    A data column's lit bin is ``((base + 2*gray - 17) % 32) + 17``; a reference
    column's is ``base`` outright, so one table serves both roles.

    Recovered by known-plaintext inversion of three recorded record-2 overs (two
    DATA overs of the 2026-08-14 two-sided bench session, one link-setup off the
    2026-07 debug bench), which agree on every column they share and reproduce all
    three overs bin-for-bin.

    Record 3's `alloc_col3` is 395 draws of `9 + Int(Rnd*16)` from its own
    interleaver gap. Record 2's gap holds 252 draws, and 228 allocations + 24
    reference classes is the same arithmetic -- but `17 + Int(Rnd*32)` off that
    gap lands within -1..+3 of the measured bin on 203 of 204 data columns: near
    enough to be the same table, not near enough to be it. What displaces it is
    not a function of the draw, of the column or of the cell value, and no
    `Int(Rnd*n)` stream anywhere in the 2**24 orbit reproduces it. So the table
    stands as read off the air rather than fitted to a formula that does not
    close.
    """
    return np.array(_REC2_BINS, dtype="<i4")


def map1_col2() -> np.ndarray:
    """Reference gate for record 2: a 1 at each of its 24 pilot columns."""
    out = np.zeros(228, dtype="<i4")
    out[_REF_POS2] = 1
    return out


# --------------------------------------------------------------------------- #
# BW2300 record-1 and record-0 (host BITRATE(2) / BITRATE(1)) bin-placement tables.
# --------------------------------------------------------------------------- #
# Record 1 emits on record 2's 228 columns and takes record 2's reference layout
# outright, so `map1_col2` gates both and there is no second gate to know. Record
# 0's 24 sit on 124 columns, in the same closed form one denominator down. Both
# read off the 2026-09-04 ladder tapes: the columns every over of a different
# payload agrees on, over eleven overs at each record.
_REF_POS0 = np.array([(41 * k + 33) // 8 for k in range(24)])


#: The 228 measured base bins, one per record-1 column, in column order.
_REC1_BINS = (
      20,  17,  24,  27,  22,  47,  24,  41,  48,  19,  42,  39,  44,  31,  40,  43,
      36,  21,  42,  25,  42,  43,  28,  23,  44,  33,  18,  17,  48,  43,  26,  23,
      18,  33,  38,  37,  34,  45,  42,  29,  22,  39,  46,  33,  40,  29,  22,  45,
      24,  37,  20,  45,  42,  33,  22,  31,  34,  47,  30,  27,  18,  25,  38,  17,
      24,  39,  38,  33,  20,  29,  40,  17,  18,  35,  40,  35,  24,  39,  32,  21,
      30,  39,  46,  33,  46,  31,  34,  35,  24,  33,  30,  45,  42,  27,  48,  23,
      24,  21,  46,  17,  28,  43,  22,  33,  40,  35,  42,  45,  28,  33,  34,  23,
      48,  23,  40,  27,  40,  47,  32,  31,  28,  29,  42,  37,  18,  43,  26,  33,
      30,  41,  48,  29,  48,  29,  30,  27,  30,  31,  46,  31,  22,  23,  24,  37,
      18,  45,  24,  25,  30,  23,  28,  19,  28,  29,  18,  39,  36,  29,  44,  33,
      20,  47,  36,  45,  42,  31,  34,  19,  40,  25,  28,  25,  44,  19,  42,  43,
      28,  31,  20,  29,  44,  35,  18,  19,  34,  25,  18,  19,  36,  27,  44,  25,
      42,  31,  28,  27,  42,  29,  42,  17,  48,  45,  42,  31,  18,  29,  24,  39,
      42,  45,  18,  39,  48,  27,  48,  47,  38,  31,  28,  43,  40,  43,  28,  45,
      44,  33,  18,  33,
)

#: The 124 measured base bins, one per record-0 column, in column order.
_REC0_BINS = (
      66,  61,  48,  49,  66,  55,  34,  75,  58,  95,  78,  75,  72,  91,  66,  93,
      88,  49,  64,  75,  88,  37,  76,  51,  90,  53,  60,  85,  34,  47,  80,  57,
      42,  59,  92,  59,  86,  45,  44,  33,  74,  57,  66,  85,  64,  71,  74,  91,
      84,  95,  44,  59,  38,  85,  52,  37,  66,  45,  56,  45,  50,  41,  60,  75,
      50,  47,  44,  45,  64,  51,  94,  41,  84,  45,  82,  35,  84,  83,  72,  33,
      62,  79,  80,  81,  42,  47,  52,  51,  92,  37,  74,  35,  94,  33,  94,  77,
      68,  39,  48,  49,  64,  73,  54,  41,  66,  79,  48,  67,  72,  89,  66,  71,
      60,  79,  68,  47,  40,  51,  52,  79,  70,  53,  40,  61,
)


def base_bins_col1() -> np.ndarray:
    """Per-column base bin for record 1's 228 columns -- MEASURED, not generated.

    Same law as record 2 one gray bit down: a data column's lit bin is
    ``((base + 2*gray - 17) % 32) + 17`` with ``gray`` under 8 rather than 16, and
    a reference column's is ``base`` outright.

    Recovered by known-plaintext inversion of the 2026-09-04 ladder tapes. With
    only 8 of the comb's 32 offsets reachable from an allocation, the columns
    themselves say what their base is: every over admits 8 bins and the true one
    is in all of them, which is how the reference layout was found before any
    plaintext was guessed. The frames then fell out of the counter payload the
    responder was pushing -- 22 counter bytes, one control byte, CRC-16/GENIBUS --
    and every over gives the same 228 numbers.

    `17 + Int(Rnd*32)` off record 1's own interleaver gap lands within -1..+3 of
    these, the same near miss record 2 has: `{-1: 3, 0: 56, +1: 83, +2: 50, +3: 12}`
    over the 204 data columns against record 2's `{-1: 9, 0: 57, +1: 77, +2: 48,
    +3: 13}`. Near enough to be the same stream, not near enough to be the table.
    """
    return np.array(_REC1_BINS, dtype="<i4")


def base_bins_col0() -> np.ndarray:
    """Per-column base bin for record 0's 124 columns -- MEASURED, not generated.

    Record 1's law on the wider comb: ``((base + 2*gray - 33) % 64) + 33``, 8
    values in 64 bins, so an allocation reaches a quarter of the comb and the
    columns are even more nearly self-determining than record 1's.
    """
    return np.array(_REC0_BINS, dtype="<i4")


def map1_col0() -> np.ndarray:
    """Reference gate for record 0: a 1 at each of its 24 pilot columns."""
    out = np.zeros(124, dtype="<i4")
    out[_REF_POS0] = 1
    return out


def clsparm_gray() -> np.ndarray:
    """Nested 2-D reflected-Gray bit maps for bpc 2, 3, 4 at offsets 0, 16, 32."""
    g = lambda i: i ^ (i >> 1)
    seq = [g(i >> 2) * 4 + (g(i & 3) ^ (3 * (g(i >> 2) & 1))) for i in range(16)]
    out = np.zeros(48, dtype="<i4")
    for bpc in (2, 3, 4):
        out[(bpc - 2) * 16:(bpc - 2) * 16 + 2 ** bpc] = seq[:2 ** bpc]
    return out


# --------------------------------------------------------------------------- #
# The shared 64-entry constellation LUT: cell = LUT[pos + 6*coded].
# --------------------------------------------------------------------------- #
# The Gray-decode permutation, identical to the clsparm_gray rows.
_CLSPARM = {1: [0, 1], 2: [0, 1, 3, 2], 3: [0, 1, 3, 2, 7, 6, 4, 5],
            4: [0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10]}
_SQRT2 = np.sqrt(2.0)

# VARA's own two pi's: the 15-significant-digit source literal (7 ulp short of the
# true double), and the 66-bit constant the x87 FSIN/FCOS reduce their argument
# against. Reproducing both is what puts the near-quadrant residues of the PSK
# columns on the exact float32 the LUT ships.
_PI = float("3.14159265358979")
_PI66 = Fraction(0xC90FDAA22168C234C, 16 ** 17) * 4


def _cis(a: float) -> complex:
    n = round(a / (math.pi / 2))
    r = float(Fraction(a) - _PI66 * n / 2)
    c, s = math.cos(r), math.sin(r)
    return (complex(c, s), complex(-s, c), complex(-c, -s), complex(s, -c))[n % 4]


def _psk(m: int, c: int) -> complex:
    return _cis(2 * _PI * _CLSPARM[m.bit_length() - 1][c] / m)


def _square_qam(bits_per_axis: int, corner: float, c: int) -> complex:
    m = 1 << bits_per_axis
    lvl = np.arange(-(m - 1), m, 2) * (corner / ((m - 1) * _SQRT2))
    g = _CLSPARM[bits_per_axis]
    return lvl[g[c >> bits_per_axis]] + 1j * lvl[g[c & (m - 1)]]


def _cross32(corner: float, c: int) -> complex:
    """32-QAM cross: the {+-1,+-3,+-5}^2 grid minus its four corners.

    `coded = 16*sx + 8*sy + 4*my + mx`; mx=2 reads through the Gray row to |I| = 7u,
    outside the grid, so that column folds onto the |Q| = 5u arm with its two rows
    reversed. All 32 labels are observed: high speed capable BW2300 level-16 and BW500
    level-13 overs decode byte-exact through this column indexed by the raw bit
    group, the 22 once inferred from sign symmetry included."""
    sx, sy, my, mx = (c >> 4) & 1, (c >> 3) & 1, (c >> 2) & 1, c & 3
    ix, iy = _CLSPARM[2][mx], my
    if ix == 3:
        ix, iy = 1 - my, 2
    u = corner / (5 * _SQRT2)
    return (1 - 2 * sx) * (2 * ix + 1) * u + 1j * (1 - 2 * sy) * (2 * iy + 1) * u


def _column(pos: int, c: int) -> complex:
    if pos < 3:
        return _psk(2 << pos, c)
    if pos == 3:
        return _square_qam(2, 1.1, c)
    if pos == 4:
        return _cross32(1.5, c)
    return _square_qam(3, 1.6, c)


def constellation_lut() -> np.ndarray:
    """The 64-entry complex LUT, bandwidth-independent.

    Six constellation columns interleaved with period 6. The 64 entries are a
    truncated prefix of that raster -- column `pos` reaches only `(63-pos)//6` --
    and the zeros are where a short column (pos 0-2) has run out, not
    constellation nulls."""
    out = np.zeros(64, dtype=np.complex128)
    for i in range(64):
        pos, c = i % 6, i // 6
        if c < (2, 4, 8, 16, 32, 64)[pos]:
            out[i] = _column(pos, c)
    return out


def to_c64(z: np.ndarray) -> bytes:
    """A complex table in the on-disk `.c64` layout: interleaved little-endian f4."""
    g = np.empty(2 * len(z), dtype="<f4")
    g[0::2], g[1::2] = z.real, z.imag
    return g.tobytes()


# --------------------------------------------------------------------------- #
# BW500 gear-down ladder: records 2, 1, 0 (host BITRATE(3), (2), (1)) bin tables.
# --------------------------------------------------------------------------- #
# Below BW500's base level the waveform is index modulation, not the level-4
# sub-band DBPSK one gear slower: one lit bin per column, three bits a column,
# 24 reference columns whose bin is fixed by a class. Record 2 lights one of
# eleven bins of a 1024-sample column over 226 columns; record 1 the same comb
# over 228 columns at turbo rate 1/3; record 0 one of 21 bins of a 2048-sample
# column -- the same band at half the spacing -- over 124 columns, its data
# offsets at stride 2. Same shape as BW2300 records 2, 1, 0, on this bandwidth's
# own stream: per record, ``(columns, first bin, span, reference layout)``. The
# reference columns are the closed form ``(a*k + b) // d``, k in 0..23, fitted to
# the columns recorded overs light payload-independently -- the smallest
# denominator that hits all 24 (records 1 and 0 take BW2300's own record-2 and
# record-0 forms outright).
_BW500_INDEX = {2: (226, 27, 11, (75, 67, 8)),
                1: (228, 27, 11, (104, 93, 11)),
                0: (124, 54, 21, (41, 33, 8))}
_BW500_NREF = 24

#: Record 0's 124 base bins, in column order -- MEASURED, not generated.
_BW500_REC0_BINS = (
     65,  62,  59,  58,  65,  60,  54,  68,  63,  72,  69,  68,  67,  72,  63,  74,
     73,  58,  65,  64,  73,  54,  69,  60,  73,  60,  63,  70,  55,  54,  69,  62,
     57,  62,  71,  62,  71,  58,  57,  54,  67,  62,  65,  70,  65,  62,  67,  72,
     71,  74,  54,  62,  55,  70,  61,  72,  65,  58,  61,  58,  59,  56,  63,  68,
     59,  56,  57,  58,  65,  60,  73,  56,  71,  58,  71,  72,  71,  70,  67,  54,
     63,  66,  69,  70,  57,  58,  55,  60,  73,  54,  67,  72,  54,  54,  54,  68,
     63,  56,  59,  58,  65,  64,  61,  56,  65,  68,  59,  64,  67,  72,  65,  64,
     63,  68,  65,  58,  55,  60,  61,  68,  67,  60,  54,  62,
)


_SPECS = {"bw2300": BW2300, "bw500": BW500, "bw2750": BW2750}


@cache
def _gap(bw: str, rec: int) -> int:
    """LCG state at the gap after a record's two permutations."""
    seed, n1, n2 = _SPECS[bw][rec]
    _, s = permutation(n1, seed)
    _, s = permutation(n2, s)
    return s


def _bw500_gap(rec: int) -> int:
    return _gap("bw500", rec)


def _next_seed(bw: str, rec: int) -> int:
    spec = _SPECS[bw]
    return spec[rec + 1][0] if rec + 1 < len(spec) else _CHAIN_END[bw]


def _closes(bw: str, rec: int, s: int) -> None:
    if s != _next_seed(bw, rec):
        raise ValueError(f"{bw} record-{rec} gap did not close on record {rec + 1}'s seed")


def bw500_ref_cols(rec: int) -> np.ndarray:
    """The 24 reference columns of a BW500 record-``rec`` frame. No PRNG.

    Record 2's columns 73 and 74 both satisfy every constraint two overs can
    raise (their ``map2`` class is 0, so a pilot there reads its own ``alloc``, and
    the data symbol the other column carries reads the same in both overs); 74 is
    the one the simpler form gives, and either choice decodes both overs to the
    same bytes."""
    a, b, d = _BW500_INDEX[rec][3]
    return np.array([(a * k + b) // d for k in range(_BW500_NREF)])


def bw500_alloc(rec: int) -> np.ndarray:
    """Per-column base bin of a BW500 record: ``first + Int(Rnd*span)`` for as many
    columns as the record has, drawn from its own interleaver gap -- except record
    0, which is read off the tape.

    Records 2 and 1 are exact: the gap of each holds precisely ``columns + 24``
    draws (250 and 252 steps to the next record's seed), and the table decodes
    every recorded over of the record byte-exact. Record 0's gap is the same
    arithmetic (124 + 24 = 148 steps), but ``54 + Int(Rnd*21)`` lands within
    -1..+2 of the measured bin on every one of its 124 columns (``{-1: 16, 0: 47,
    +1: 45, +2: 16}``) and on none of them reliably -- the near miss BW2300's
    records 2, 1 and 0 show on their wide combs (:func:`base_bins_col2`), and no
    ``Int(Rnd*A + B)`` reaches more than 61 of the 124. So the 124 stand as
    measured: known-plaintext inversion of the two 2026-09-11 stock BW500
    resends of the 225-byte gateway greeting (the level-1 over carries its first
    nine bytes), both arms giving the same 124 numbers, 29 data and 6 reference
    columns wrapping the 21-bin comb, which is what fixes the modulus at 21."""
    ncol, first, span, _ = _BW500_INDEX[rec]
    if rec == 0:
        return np.array(_BW500_REC0_BINS, dtype="<i4")
    s = _bw500_gap(rec)
    out = np.empty(ncol, dtype="<i4")
    for i in range(ncol):
        s = step(s)
        out[i] = first + draw(s, span)
    return out


def bw500_map2(rec: int) -> np.ndarray:
    """Reference-cell class at each of a BW500 record's 24 pilot columns:
    ``Int(Rnd*8)``, the 24 draws that follow the record's ``columns`` allocation
    draws in the same stream.

    The state lands on the next record's stage-1 seed after them -- 6437280,
    14955604, 7910701 for records 2, 1, 0 -- which is what fixes each pair as a
    continuation rather than a lucky reseed, and the column count as 226 / 228 /
    124 rather than any other split of the gap."""
    ncol = _BW500_INDEX[rec][0]
    s = _bw500_gap(rec)
    for _ in range(ncol):
        s = step(s)
    out = np.empty(_BW500_NREF, dtype="<i4")
    for k in range(_BW500_NREF):
        s = step(s)
        out[k] = draw(s, 8)
    if s != BW500[rec + 1][0]:
        raise ValueError(f"BW500 record-{rec} stream did not close on record {rec + 1}'s seed")
    return out


# --------------------------------------------------------------------------- #
# Gap draws: what a record's Rnd gap holds after its two permutations.
# --------------------------------------------------------------------------- #
# OFDM records, per bandwidth: (records, first bin, carriers, cells per carrier,
# QAM records). A QAM carrier's gap cell is the +1 right after pilot row
# `Int(Rnd*19)`; a PSK carrier's is a +1 repeat at symbol `7 + Int(Rnd*142)`.
_OFDM = {"bw2300": (range(9, 16), 8, 49, 1, (14, 15)),
         "bw500": (range(6, 13), 27, 11, 4, (11, 12)),
         "bw2750": (range(10, 17), 3, 59, 1, (15, 16))}


@cache
def gap_cells(bw: str, rec: int) -> np.ndarray:
    """An OFDM record's gap cells, (carriers, cells per carrier), in draw order.

    Bin-major from the record's gap; a draw repeating one of its carrier's earlier
    values is drawn again (BW500's four per carrier; BW2300 and BW2750 have one,
    so never). The draws close exactly on the next record's seed -- 49 at BW2300,
    59 at BW2750, 44-49 at BW500 -- which is what fixes the gap as these cells and
    nothing else."""
    recs, _, ncar, per, qam = _OFDM[bw]
    if rec not in recs:
        raise ValueError(f"{bw} record {rec} is not OFDM")
    base, span = (0, 19) if rec in qam else (7, 142)
    s = _gap(bw, rec)
    out = np.empty((ncar, per), dtype="<i4")
    for c in range(ncar):
        k = 0
        while k < per:
            s = step(s)
            v = base + draw(s, span)
            if v not in out[c, :k]:
                out[c, k] = v
                k += 1
    _closes(bw, rec, s)
    out.flags.writeable = False
    return out


# Tone records: (bands, pilots per band, columns per band after the reference).
# BW2750 levels 5-9 are BW2300 records 4-8 outright; its level 10 is record 9.
_TONE = {("bw500", 3): (2, 18, 392), ("bw500", 4): (2, 18, 393), ("bw500", 5): (3, 12, 374),
         ("bw2750", 9): (13, 3, 347)} | {
    (bw, rec): row
    for rec, row in {4: (3, 12, 374), 5: (4, 9, 363), 6: (6, 6, 353),
                     7: (8, 5, 350), 8: (10, 4, 347)}.items()
    for bw in ("bw2300", "bw2750")}


@cache
def tone_pilots(bw: str, rec: int) -> tuple[tuple[int, ...], ...]:
    """A tone record's pilot columns per band, sorted, counted from the reference.

    Band after band, `5 + Int(Rnd*(ncols-6))` until the band has its count of
    distinct values, from the record's gap; the draws close on the next record's
    seed. A pilot column repeats the previous cell. BW500 records 3-5 (levels
    4-6), BW2300 records 4-8 (levels 5-9) and BW2750 records 4-9 (levels 5-10)
    are the one law."""
    nb, npil, ncols = _TONE[bw, rec]
    s, out = _gap(bw, rec), []
    for _ in range(nb):
        got: list[int] = []
        while len(got) < npil:
            s = step(s)
            v = 5 + draw(s, ncols - 6)
            if v not in got:
                got.append(v)
        out.append(tuple(sorted(got)))
    _closes(bw, rec, s)
    return tuple(out)


# --------------------------------------------------------------------------- #
# usedmap: which cells the BW500 L4 receiver skips as pilots (record 3, off=3).
# --------------------------------------------------------------------------- #
_USEDMAP_LEN = 900000
_STRIDE, _OFF = 17, 3


def usedmap_plane() -> np.ndarray:
    """The record-3 usage plane: a 1 at each pilot cell of the off=3 stride.

    This reproduces only the stride the off=3 receiver reads (`17*(128*co+s)+3`);
    the shipped file interleaves 16 other records' planes on the other strides,
    which this receiver never consults and which are out of scope here."""
    out = np.zeros(_USEDMAP_LEN, dtype=np.uint8)
    for s, cols in enumerate(tone_pilots("bw500", 3)):
        for co in cols:
            out[_STRIDE * (128 * co + s) + _OFF] = 1
    return out


# --------------------------------------------------------------------------- #
# grid480: the per-cell differential-phase reference, recovered from recordings.
# --------------------------------------------------------------------------- #
_GRID480_LEN = 10000


#: The 786 recovered cells as `(index, k)`: the phase reference at grid480 index
#: `index` is the `k`-th point of the 128-point lattice. Sorted by index; the gaps
#: are the cells a SHORT burst never emits.
_GRID480_CELLS = (
    (  0,  78), (  1,  83), (  2,  81), (  3,  38), (  4,  79), (  5,  22),
    (  6,  91), (  7,  52), (  8,  72), (  9, 100), ( 10, 103), ( 11,  82),
    ( 12,  71), ( 13, 125), ( 14, 123), ( 15,  26), ( 16,   6), ( 17,  24),
    ( 18,  57), ( 19,  72), ( 20,  90), ( 21,  36), ( 22,  33), ( 23,  13),
    ( 24,  54), ( 25,  41), ( 26,  98), ( 27,   1), ( 28,  94), ( 29,  74),
    ( 30,  88), ( 31,  52), ( 32,  40), ( 33, 104), ( 34,  40), ( 35,  96),
    ( 36, 108), ( 37, 101), ( 38, 121), ( 39,  60), ( 40,  12), ( 41,   1),
    ( 42,  73), ( 43, 112), ( 44,  30), ( 45, 102), ( 46, 113), ( 47,  33),
    ( 48,  54), ( 49,  60), ( 50,  51), ( 51,  41), ( 52,  80), ( 53, 121),
    ( 54, 127), ( 55, 102), ( 56,  91), ( 57,  55), ( 58,  69), ( 59,  43),
    ( 60,  97), ( 61,  65), ( 62, 116), ( 63,  87), ( 64,  82), ( 65,  44),
    ( 66,  17), ( 67,   0), ( 68, 113), ( 69,  31), ( 70, 115), ( 71,  77),
    ( 72,  86), ( 73,  49), ( 74,  27), ( 75,  95), ( 76,  33), ( 77, 108),
    ( 78,  49), ( 79, 106), ( 80,  64), ( 81, 114), ( 82,  25), ( 83, 100),
    ( 84,  87), ( 85,  58), ( 86,  52), ( 87,  88), ( 88,  79), ( 89, 120),
    ( 90,  50), ( 91,  88), ( 92, 122), ( 93,  23), ( 94,  20), ( 95,  14),
    ( 96,  97), ( 97, 100), ( 98,  65), ( 99, 111), (100,  47), (101,  73),
    (102,  63), (103,  11), (104,  56), (105,  51), (106,  17), (107,   3),
    (108,  36), (109,  18), (110,  47), (111,  24), (112,  56), (113, 125),
    (114,  31), (115,  94), (116,  67), (117, 110), (118,  51), (119,  10),
    (120,  37), (121,  46), (122,  93), (123, 109), (124, 124), (125,  80),
    (126,  45), (127, 126), (128, 104), (129,  83), (130, 108), (131,  12),
    (132, 126), (133, 103), (134,  80), (135,  25), (136,  72), (137,  79),
    (138,  94), (139,  65), (140, 123), (141,  94), (142,  95), (143,  81),
    (144,  50), (145,  32), (146, 123), (147, 122), (148, 104), (149,  22),
    (150, 115), (151,  30), (152, 116), (153,  30), (154, 122), (155,  77),
    (156,  59), (157, 111), (158,  50), (159,  72), (160, 123), (161,  60),
    (162,  68), (163,  63), (164,  46), (165, 125), (166,  25), (167,  29),
    (168,  25), (169,  70), (170,  55), (171, 125), (172, 121), (173,  82),
    (174,  53), (175,  55), (176,  67), (177,  32), (178,  95), (179,  27),
    (180,  25), (181,  57), (182, 100), (183,  58), (184,  77), (185,  13),
    (186,  60), (187,  93), (188,  13), (189, 121), (190,  23), (191,  19),
    (192,  46), (193, 102), (194,   0), (195, 107), (196,  21), (197,  13),
    (198,  16), (199,  58), (200,  64), (201,  93), (202,  79), (203,  25),
    (204, 115), (205, 115), (206,  37), (207, 112), (208, 127), (209,  67),
    (210, 127), (211,  40), (212,  43), (213,  90), (214, 126), (215,   0),
    (216,  70), (217,  62), (218,  87), (219,   0), (220,  69), (221, 113),
    (222,  80), (223,   2), (224,  22), (225,  17), (226,  82), (227, 114),
    (228,   7), (229,  30), (230,  40), (231,  19), (232,  80), (233,  89),
    (234,  91), (235, 124), (236,  61), (237,  68), (238,  36), (239,  30),
    (240, 120), (241,  74), (242,  15), (243,   3), (244, 115), (245, 124),
    (246,  49), (247,  22), (248, 114), (249, 118), (250,   0), (251,  25),
    (252,  54), (253,  92), (254,  81), (255,  57), (256,  70), (257,   4),
    (258, 111), (259,  57), (260,  88), (261,  51), (262,  86), (263,  78),
    (264,  94), (265, 125), (266,  13), (267,   6), (268,  43), (269,  74),
    (270,  36), (271, 103), (272,  67), (273, 123), (274,  69), (275,  16),
    (276,  66), (277,  36), (278, 116), (279,  30), (280, 102), (281, 119),
    (282, 107), (283,  15), (284,  53), (285,  62), (286,  13), (287,  92),
    (288,  82), (289,   3), (290,   9), (291,  37), (292,  92), (293,  79),
    (294,  11), (295,  13), (296, 124), (297,  12), (298,  28), (299,  32),
    (300,  16), (301,   8), (302,  26), (303, 109), (304, 118), (305,  23),
    (306,  83), (307,  52), (308, 111), (309,  85), (310,  58), (311,  62),
    (312,  52), (313,   5), (314,  75), (315,  69), (316,  22), (317,  26),
    (318, 123), (319,  14), (320,  79), (321,  81), (322,  87), (323,  26),
    (324, 103), (325, 119), (326,  62), (327, 118), (328,  64), (329,  76),
    (330,  59), (331,  42), (332,  66), (333,   3), (334, 124), (335,  85),
    (336,  33), (337, 103), (338, 109), (339,  83), (340,  77), (341,  20),
    (342, 120), (343,  24), (344, 114), (345, 104), (346,  84), (347,  29),
    (348,  45), (349, 118), (350,  11), (351, 117), (352,  80), (353,  49),
    (354,  10), (355, 124), (356,  76), (357,  46), (358,  99), (359,  42),
    (360,  61), (361,   0), (362,  29), (363,   9), (364,  18), (365,  64),
    (366,  57), (367,  67), (368,  92), (369,  42), (370,  72), (371,  80),
    (372,  46), (373, 101), (374,  31), (375,  81), (376,  51), (377,  95),
    (378,  59), (379, 125), (380,  78), (381,  83), (382,  52), (383,  51),
    (384, 105), (385, 106), (386,  90), (387,  44), (388,  96), (389, 122),
    (390, 108), (391,  80), (392,   7), (471,  72), (472,  76), (473,  55),
    (474, 106), (475, 110), (476,  46), (477,  60), (478, 119), (479,  18),
    (480, 119), (481,  92), (482,  78), (483,  11), (484,  51), (485,  85),
    (486, 124), (487,  90), (488, 101), (489,   9), (490,  32), (491,  64),
    (492,  76), (493,  33), (494,  15), (495, 117), (496,  81), (497,  75),
    (498,   0), (499, 112), (500,  15), (501,  33), (502, 114), (503,  99),
    (504, 124), (505,  38), (506,  30), (507,  71), (508,  52), (509, 111),
    (510,  22), (511,  16), (512, 101), (513,  90), (514, 125), (515, 127),
    (516,  84), (517, 126), (518,  96), (519,  17), (520,  12), (521,  31),
    (522,  49), (523,  23), (524,  41), (525,  93), (526, 100), (527, 100),
    (528,  58), (529, 105), (530,  33), (531,  29), (532,  82), (533, 116),
    (534,  42), (535,  49), (536,  50), (537,  36), (538,  71), (539, 102),
    (540,  14), (541,  68), (542, 109), (543,  11), (544, 125), (545, 121),
    (546, 109), (547,  80), (548, 126), (549,  42), (550,  88), (551,  47),
    (552,  37), (553,   6), (554, 112), (555,   5), (556, 101), (557,  29),
    (558,  72), (559,  66), (560, 120), (561,  40), (562, 126), (563,  58),
    (564, 105), (565, 108), (566,  15), (567,  22), (568,  65), (569,  53),
    (570,  89), (571, 108), (572,  80), (573,   0), (574,  45), (575, 102),
    (576,  19), (577,  58), (578,  18), (579,  33), (580,  69), (581,  31),
    (582,  23), (583,  21), (584,   1), (585,  65), (586,  80), (587,  45),
    (588,  18), (589,  38), (590, 114), (591, 115), (592,  93), (593,  13),
    (594,   9), (595, 106), (596, 102), (597,  78), (598,  86), (599, 120),
    (600, 127), (601,  92), (602,  70), (603, 126), (604,  14), (605, 104),
    (606,  12), (607,   5), (608, 126), (609,  35), (610,  97), (611,  23),
    (612,  62), (613,  32), (614,  82), (615,  41), (616, 117), (617,  85),
    (618,  71), (619,  50), (620,  71), (621,  61), (622,  20), (623,  90),
    (624,  64), (625,  28), (626,  55), (627,  80), (628,  95), (629,  95),
    (630,  50), (631,  56), (632,  62), (633,  29), (634,   2), (635,  63),
    (636,  94), (637,  63), (638,  62), (639,  78), (640,  14), (641,  56),
    (642,  76), (643,  87), (644, 124), (645, 114), (646,  61), (647,  79),
    (648,  85), (649,  70), (650,  72), (651,  57), (652,  21), (653,  37),
    (654,  99), (655,  92), (656, 116), (657,  89), (658, 125), (659,  17),
    (660, 103), (661,  98), (662,  88), (663,  60), (664,  83), (665, 126),
    (666,   3), (667,  85), (668,  81), (669,  73), (670, 119), (671,  33),
    (672,  26), (673, 125), (674,  72), (675,   3), (676,  19), (677,  87),
    (678,  10), (679, 106), (680, 116), (681,  22), (682,  70), (683, 103),
    (684,  18), (685,  34), (686,  17), (687,  92), (688,  76), (689,  70),
    (690,  77), (691,  82), (692,  18), (693,  26), (694, 124), (695, 103),
    (696,  17), (697,   0), (698,  61), (699,  99), (700, 124), (701,  74),
    (702, 106), (703, 103), (704, 117), (705, 118), (706,  74), (707,  65),
    (708,  23), (709,  20), (710, 113), (711,  95), (712,  39), (713,  77),
    (714,  55), (715,  94), (716,  81), (717, 121), (718,  85), (719,  62),
    (720,  29), (721, 106), (722,  28), (723,  53), (724, 118), (725,  78),
    (726,  80), (727,  29), (728,  78), (729,  42), (730,  34), (731,  12),
    (732, 116), (733,   6), (734,  75), (735,   0), (736, 114), (737,  37),
    (738,  68), (739,  51), (740,  29), (741, 111), (742,  33), (743,  16),
    (744,  66), (745, 107), (746,  13), (747,  67), (748, 103), (749, 108),
    (750,  98), (751, 103), (752,  59), (753,  71), (754,  95), (755,  95),
    (756,  35), (757,  63), (758,  10), (759,  67), (760,  93), (761, 127),
    (762,  39), (763, 118), (764,  76), (765,  66), (766,  79), (767,  81),
    (768, 102), (769,  16), (770,  43), (771, 126), (772,  57), (773,  40),
    (774,  82), (775, 100), (776,  24), (777, 119), (778,  62), (779,  57),
    (780, 102), (781,  64), (782, 106), (783,  58), (784, 123), (785,  99),
    (786,   8), (787, 116), (788,  48), (789,  50), (790,  95), (791,  61),
    (792,  16), (793,   1), (794,  65), (795,  67), (796,  23), (797,  64),
    (798,  41), (799,  63), (800,  36), (801,  60), (802, 116), (803,  70),
    (804, 126), (805,   6), (806,  54), (807,  60), (808, 127), (809, 114),
    (810,  61), (811, 103), (812, 100), (813,  58), (814,  35), (815,  27),
    (816,  46), (817,  63), (818,  12), (819,  24), (820,  49), (821, 108),
    (822,   0), (823, 110), (824,  62), (825,  55), (826,  98), (827,  24),
    (828, 109), (829,  70), (830,  14), (831,  45), (832,   1), (833,  43),
    (834,  17), (835,  46), (836, 127), (837,  76), (838,   2), (839, 127),
    (840,  73), (841,  99), (842, 127), (843, 109), (844, 116), (845,  28),
    (846,  65), (847, 112), (848,  42), (849,  97), (850,  96), (851, 111),
    (852,  57), (853,  47), (854,  35), (855,  58), (856,  58), (857,  36),
    (858,   0), (859,  25), (860,  93), (861,  23), (862,  50), (863,   0),
)


@cache
def _grid480_k() -> np.ndarray:
    return np.array(_GRID480_CELLS, dtype=np.int64)


def grid480() -> np.ndarray:
    """The de-rotation reference, `exp(j*2*pi*k/128)` at each recovered cell.

    A BW500 SHORT burst emits only its data body, so known-plaintext inversion of
    our own loopback recordings recovers 786 of the 939 reference cells the
    receiver fetches -- every recovered value landing on the 128-point lattice,
    which is why the deliverable is 786 small integer pairs, written out above.
    The other 153 and the unread tail past them were never emitted by a SHORT
    burst and are not derivable from this corpus; they are left zero, which costs
    nothing: the 939 are 470 columns per sub-band, headroom for the blind c0 scan,
    and the 786 are exactly the 393 consecutive columns one alignment consults."""
    out = np.zeros(_GRID480_LEN, dtype=np.complex128)
    idx, k = _grid480_k().T
    out[idx] = np.exp(2j * np.pi * k / 128)
    return out


# --------------------------------------------------------------------------- #
# The tone family's de-rotation rows.
# --------------------------------------------------------------------------- #
# Band b of a tone level is de-rotated by row b at its column from the reference
# symbol, at every bandwidth. Rows 0 and 1 are grid480 at 0.. and 471.., each one
# column longer where BW500 level 5 (402 symbols) reads past it. Rows 2-12 are
# MEASURED, read off high speed capable overs by known-plaintext inversion: row 2 at
# BW2300 level 5, row 3 at level 6, rows 4-9 at level 9, the tails of rows 4-7 at
# level 8, the last three columns of rows 4 and 5 at BW2750 level 7 (three
# overs agreeing within 0.25 unit), rows 10-12 at BW2750 level 10. Each is as
# long as the longest level that reached it cleanly. Where rows 2-12 sit in
# VARA's own grid file, and whether a generator makes them, is unknown.
_GRID480_TAIL = {0: 109, 1: 82}

#: Lattice integers k (phase 2*pi*k/128) per column, rows 2-12.
_GRID_ROWS = {
    2: (
        110,  19,  83,  57, 115, 124,  79,  66,   4,  98, 118, 110,  63,  17,  96,  91,
         18, 106,   1,  24,  23,  69,  80,  40,  13,  15,  18,  37, 114, 118,  68, 125,
         59,  60,  37,  93,  53,  43,  98,  39, 118,  38,  91,  33, 123,  89,  17,  45,
        103,  13,  51,  47,  82,  27,  62,  15,  76, 123,  81,  91, 117, 120,  18,   8,
         52,   3,  59,  87,  20,  34,  63,  80,  63,   3,  24, 108,  18,  49,  50,  83,
         97,  98, 114,  61,  68, 108,  95, 122,  29,  82, 115,  10,  44, 126, 122, 106,
         75,  14,  40, 108,  70,  68,  57,  63,  85,  28,  66,  17,  56, 125,  21,  80,
        115,  11,  82,   8,  33,  24,   8, 110, 116,  54,   9, 119,  77, 107,  79,  36,
        118, 127,   0,  94, 124, 117,  38, 120,  42,  20, 109,  86,  32,  34,  15,  40,
         18,  46, 103,  84,  30,   6,  13, 111,  69,  76,  44, 101,   9,  33, 100,  60,
         39, 123,  88, 115, 109,  25,  89,   0,  51,  17,  43, 125, 127,   3, 117,  97,
         51, 109,  70,  99, 111,  26,  66, 127,   4,  56, 108,  24,  23,   5,  13,  54,
         84,  39,  67, 110,  74,  21,  36,  92, 101,  54,  17,  77,   8,   4,  24, 125,
         73, 109,   1, 122,  70,  56, 122,  37,  37,  30,  94,  84,  41, 124,  32,  22,
        112,  37,  85,  18,  73,  77,  95,  12,  51,  38,  52,   6, 113,  10,  79,   0,
         72,  82,  47,  94,  90,  66,  10, 100,  28,  34,  23,  94, 116, 107, 112,  95,
        111,  25,  34,  37,  30,  36,  90,  25,  16,   6,  40, 112, 107, 121, 108, 113,
         35,  64,  98,  82,  93,  30,  71,  62,  94, 104,  42, 119,  43,  51,  79,  22,
         69,  41,  63, 107, 124, 126, 108,   3, 114, 122, 127,  77,  43,  52,  68,  80,
         81,  91,  45,  25,   3,  48,   2,  58,  93,  19,  43, 104,   2,  57,  20,  62,
        102, 120,  63,  38,  24,  64, 103,  82,  76,  39,  79,  96, 103,  31,  43,  34,
         66,  71,  38, 121,   1,  91,  18,  91,  13,  73,  46, 112,  47,  98,  18,  93,
         69,  42,  54,  28,  38,  77,  32,   5,  17,  47,  43, 111,  81,  30, 117, 109,
        109,  39,  97,  54,  11,   3,  74,
    ),
    3: (
         56,  48,   6,  98, 100, 109,  33,  53,  33,  64,  53, 118,  13,  12,  62,  52,
        105,  91,  70,  90,  83,  25,  50,  68,  46, 108,  55,  90,  26,  49, 104, 105,
         36,  17, 110,   3, 124,  66,  83,  10,  92,  90, 123,  96,  13,  78,  50,  45,
          1,   8,  90,  69,  29,  39,  74,  58,  22,  81,  47,   2, 117,  50,  46,  63,
         58,  24,   8,  44,  25,  38, 121,  39,  98,  54,  30,  88,   4,  76,  17,   2,
         37,  53,  20,  97, 110,  62, 101,  66, 107,  16,  53,  32,   9,  42,  49, 121,
         60, 118,  58,  49,  63,  13,  53,  30, 119,   7,   0,  51, 117, 121,   0,  72,
         23,  19,  85,   5,  75,  82,  43, 110, 112,  95,  45,  10,  89,  11,  22,  47,
        111,  97, 100, 103, 115, 107,  44,   5,  93,   1,   6,  67, 101,  76,  36,  17,
         29,  88,   3,   3,  56,  86,  61,  84, 105, 114, 121,  21, 106,  75,   0, 114,
         24,  13, 110,  35,  57,  52,   2, 111,  90,  89,  19,  91,  88,  54,  32, 115,
        122,  56,   2,  49,  56,  65,  64,   8,  26, 125, 127,  19,  64,  91,  21,  87,
        122,  46,  59,  55,  20,  92,  91, 112,  50,  67,  11,  83,  82,  44,  23,   2,
        114, 102,  53,  96,  80,   4,  87,  28,  72,  51,  35,  88,  94,  51, 120, 116,
         89, 119,  45, 119,   9,  87,  90,  29,  38, 114,  82, 124,  85,  34,  46,  81,
         73,  23,   0, 102,   2,  20,  41,  38,  54,  72,  72,  58,  73,  69,  77,  90,
        121,  28,  42,  57,  28,  26,  38,   9, 125,  27,  78,  43, 104,  11,   9, 115,
         68, 126,  72,  24,  85, 101,  86,  58,  39, 113,  82,  11,   4,   4,  56,  31,
         31,  82,  21,  80,  81,  23,  95,  73, 123, 113,  96,  52,  14,  93,  76, 126,
         37,  80, 111,  73,  77, 107,   5, 106,  97,  97,  38,  32,  21, 102,  91,  85,
         14,  77,  82,  14,  42,  67,  44, 111,  98,  41, 110, 106,  74,  12,  27,   6,
         50,  64,  91,  76, 106,  27,  88,  76,  39,  78,  39,  77, 125,  93,  94,  19,
          9,  64,  71,  75,  46,  18,  48,  17, 120, 118,  91,  34,
    ),
    4: (
         72, 115, 104,  22, 122,  67, 105, 108,  18,  76, 115, 117,  33,  72,  43, 110,
        114,  56,  81,  65,  89,  12,  90,  46,  23, 101,  49,  34, 106, 116,   9,  45,
         52, 126,  91,  10,  66, 125, 126,  92,  61, 100, 100, 117,  45,   0, 113,  83,
         40, 118, 118,  40,  25, 109,  77,  19,  45,  75,  12, 113,  90,  19, 107,  76,
          7, 113,  51, 108,  96,  86,  92,  17,  52,  55,  72,  60,   3, 118,  29,  37,
         47,  95,  61,  78,  58,  79,  97,  50,  58, 107,  85,  26,  93,  18,  77,  12,
         23,  82, 100,  96, 104,  15,  53, 113,  76,  71,  18, 110,  58, 110,  97,  76,
         92, 120,  23,  89,  80, 113,  77, 108,  18,  76, 126,  63,  11,  53,  97,  82,
         55,  35,  99,  10, 111, 110,  63,  96,  88,  25,  56,  47,  99,  45, 116,  42,
          3,  67, 121, 106, 108,  29,  67,  40,   8, 111, 124,   4, 117,  64,  94,   0,
         58, 104,  38,  14,   9,  54,  46,  65,  45,  48,  44,  35,  20, 118,   8,  35,
        119,  68,  88,  41,  39,  18, 119,  72, 113,  91,  68,  15,  50, 119, 119, 125,
        117,  38,  30,  15,  73,  43,  53, 120,  31,  17,  99, 109,  96,  13,  82,  27,
         14, 127,  38,  55,  17,  23,  30,  48,  33,  18,  73,   3,  83,  31,  94,  43,
         60,  96,  64,  51,  66,  17,   9, 106,   2,  64,  82,  51,  92,  54,   5, 119,
         26, 122,  89,  57,  63, 111, 107,  69, 108,  25, 126,   3, 109, 122,  74, 110,
         98,  28, 127,  29,  10,  44,  93, 122,  42,  64, 107,  24,  26,  53,  86,  27,
        112,  56,  99,  83,  70,  95,  20, 105,  38, 115,  89,  53,  21,  79, 109,  43,
         61,  92,  82, 111,  52,  65, 102,  13, 106,  21,  36,  63,  48,  79, 119, 105,
         99,  60,  58,  41,  57,  42,  75,   2,  34,  38,  77,  59,  94,  94, 123,  69,
         31,  37,  42,  81,  85,  18,  88,   6,  23,  69, 111,  78,  50,  70,  31,  72,
         74,  10,  80,  94,  44,  19,  68, 116,  53,  52,  77,  59,  92, 110,  41,  32,
         93, 122,
    ),
    5: (
        117,  11,  96, 117, 108, 126,  29,  30, 113,   5,  95, 104,   1,  53,  86,  14,
         11,  66,  74,   1, 104,  45,  30,  93, 104,  16, 116,   8,   5,   0, 108, 102,
         55,  32,   9,  57, 100,  70,  47,  23, 114,  71,  77,   4, 123,  93,  21,  45,
         29,  56,  99,  70, 108,  34,  64, 124,   5,  76,   3,  19,  64,  54,  17,  34,
          2,  60,  65, 110,  44,  59,  96,  89,  53,  99,  77,  12,  80,  23,  64,  36,
         74, 126,  45,  22, 115,  45,  64,  30,  54, 109,   4, 101,  72,  39,  36,  48,
        122, 113,  78,  69,  43,  48,  48,  89,  94, 109,  33,  52, 100,  52,  94, 102,
         53,  38, 110,  34,  98,  33,  34,  57,  31,   6,  60,  49,   0,  39,  44,   4,
         66,  83, 115, 116,  67, 121,  37,  12,  16, 122,  15,  49,  26,  11, 113, 103,
          5,  71, 104,  32,  28,  83, 108,  63, 100,  42, 113,  44,  76,  10,  44,  16,
          0, 118, 117,  46,  88, 106,  68, 100, 113,  29,  90,  55,  86, 111, 126,  27,
         90, 115,  97,  68,   6,  23,  31,  38,  41, 112,  99,  10,  18,  35,  39,  72,
         86, 111,  22,  41,  79,  86,  15,  86,  35, 105,  70, 121, 125,  50,   9, 118,
         90,  61,  28,  66,   2,  64,  66,  97,  19, 106,  90, 126,  51,  71,  33,  34,
        106,  81,  28,  23,  12,  18,  10,  86,  75, 117,  23,  44, 114,  43,  21, 108,
         41,  59,  93,  77, 119,  35,  87, 100,  67,  44,  23,  60,  21,  74,  32,  18,
         94,  48,  73,  44, 113, 114,  57,  89,  14,  82,  18,   4,  25,  45,  38, 113,
        108,   1, 104,  26,  70,  22,  99,  34,  96,  74,  95, 121,  27,   0,  38,  10,
         87,  32,  99,  29, 100,  73,  34,  81,  14,  23, 121,  51,  87,  13,  65, 121,
         69,  33,   2,  92,  85, 104, 105,  18,  12,  89, 119, 104,  41,  60,  56, 127,
        122,  54,  44,  29,  72, 109,  71,  51, 112,  84,  18, 112,  16,  28, 107, 120,
         89,  49, 109,  71,   8, 113, 110,  36, 109, 108,  33,  62,  36,  84,  90, 100,
        105,   5,
    ),
    6: (
         90,  92,  79,  71,  63,  41,  11,  64, 111, 114,  81,  55,  75,  53,  32,  22,
        100, 107,  34,  76,  51,  75,  48,  48, 120,  72,   2,   9,  41,  25, 114,  40,
         28, 116, 111,  47,  49,  88, 126, 115,  53,   4, 108, 120,  94,   7,  16,  27,
         90, 117,  61,  84,  22,  51,  70,  12,  16, 112,  41,  49,  85,  49,  94,  72,
         20,   9,  57,  56,  99,  98, 123,  41,  15, 123, 118,   8,  30, 111,  94,  38,
         97, 109,  44, 125,  52,   9,  46,  15,  88,  47,  43, 108, 104,  49,  50,  78,
        119, 126,   2, 100,  74, 110,  19,  40,  49,  59,  68, 109,   1,  15,  29, 122,
         44,  55,  68,  76,   2, 115, 126, 124,   1, 107,  91,  61,  87,  61,   3,   1,
        121,  55,  30,  55,  90,  30,  94,  51,  78,  41,  41,  43, 122,  15, 100,  90,
        113,  56,  89,  68, 114,  19,  72,  22,  63,   5,  11,  40,  22, 122, 100,  33,
         76,  24,  97,  54,   8,  23, 111,  14,  26,  41, 123,  69, 124,  16,  68,  12,
         97,  84,  63, 107, 122,  14,  34,  35,  71,  99,  18,  51,  25,   9, 105, 117,
         38,   7,  30, 101,  70, 126,  88, 126,  75,  30,  12,  65, 125,  55,  83,  83,
         50, 110,  73,  67,  11,   9,  32, 102,  76, 104,  67,  96,  84,  16,  38,  64,
         59, 102,  42,  71,   9, 118,  46,  69,  18, 109,  49,  32, 112,  40,  36, 116,
         23, 107,  77,  80,  29,  38,  85,  35,   4, 121, 114,  51,  60,  52,  47,  70,
         62,  24,  90,  97,  70,  33,   6,  41,  37, 120,  62, 105,  72,   8,  75,  49,
         70,  45,  28,  23,  33,  10,  85,  30,  31, 121, 114,  50,  67,  23,  25,  76,
        101,   5,   0,  54, 111,  38, 115, 109,  56,  37,   8,  30, 123, 121,  93,  79,
        113,  26,  13,  28,  12,  90,  52,  28,  86,  76,  24,  95,  94,  95, 120,  20,
         97,  11, 113,  76, 122,  40,   9,  85, 127,  88,  96,  68, 124,  31, 109,  19,
         76,  21, 115,  99,  83,  55,   7,  96,  89,  87,  56,  62,   2,  46,  94,
    ),
    7: (
         65,  76,  17,  48,  69,   2,  82, 103,  77,   3,  32,  40,  41,  42,  20,  43,
         30, 118,  91, 118,   1,  37,  99,  68,   5, 109,  46,  22,  20,  89,  71,  86,
         81,  63,  79,  39,  66,  96,  51,  74,   9,  26, 125,  44,  74,  70,  68,  12,
         45,  55,  65,  71,   2,  74,   0,  69, 122,  39,  48, 126,  24,  27,  24,  37,
         36,  11,  38, 119,  87,  26,  37,  30,  25,  85,  12,  61,  99,  27, 110,  31,
         67,  10,  18, 122, 123,  36,  32,  94,  69,   1, 117,  60,  53,  45,  20,  19,
         60,  35,  57,  51,  70,  97,  12,  50,   0,  41, 111, 111,  56,  90, 117,  58,
         97,  99, 112,  36,  49, 102,  37, 101, 106, 110,  32, 103,  50,  70,  31, 113,
         32, 122,  43, 111,  85, 107,  74,  95,  67,  12,  75,  86,  13,  57,  61,  49,
         11,  54,   0,  87, 105,  69, 118,  43, 108,  97,  86,  17,  81,  25,  30,  21,
         82,   6,  35,  63,  70, 108,  69, 118, 101, 110,  68,   7,  38, 107,  43,  88,
         70, 119, 102,  41, 102, 117, 117,   8,  80,  80,  59,  77,  86,  88, 115,  83,
         87,  57,  64,  56,  96,  23,  97,  75, 108,  70, 127,  21,  72,  37,  35,   2,
         22,  27,  70,  46, 108,  40,   6,  76,  26,  45, 115,  49,   5,  57,   5, 127,
         50,   9,  43, 109, 100,  33,   1,  51,  89,   7,  30,  21,  53,  27,   7,   5,
        126,  20,  68, 121,  63,  22,  14,  77,  76, 111,  35,  80,  34, 112,  53, 109,
        105, 105,   6, 115,  24,  61,  10,   3,  50,  37,  69,  25,  51,   0,  61,  50,
          3,  87,   4,  30,  36, 111, 112,  95, 109,   7, 108,  63, 113,  48, 106, 112,
        126,  76, 116,  92,  61,  32,  94,  13, 124,  20,  27,  54,   5,   9,  40,  96,
         40,  88,  41,  49,  93, 104,  17,  86,   0, 105, 117,  17,  54,  46,   6,  93,
        117,  39,  26,  62,  28, 125,  99,  37,  56,  71,  64,   2, 111, 105,  45,  98,
        115,  11,  89,  70,  48,  54,  83,   6,   8,  11,  96,  90,  54,  30, 109,
    ),
    8: (
         61,  14,  60,  71,   2, 110,  46,  93,  24,  65,  23,  23,  43, 103,  35,  25,
         84,   0,  57,  86,  63, 112, 115,  38, 119,   1,  62, 108,  76,  36,  13, 119,
         99,  84,  56, 104,  94,  24, 113,  75,  35,  36, 102,  69,  71,  27, 115, 106,
         29, 115, 126,  41,  88, 117, 118, 100,  93,  96, 102,  33,  39,  43,  51,  86,
         82, 100, 117, 101,  72,  57, 113,  37,  10,  17,  53,  25,  22,  85,  75,  64,
        118,  81,  32,   9, 105,  51, 112,  97,  43,  45, 103, 107,  94,  60,  92,  76,
         28,   3,  40,  33,  21,  84,  33,  15,  97,  77,  53, 116, 107,  29,  34,  63,
        117,  96,  81,  90,  69,  45,  83,  64, 113,  43, 116,  49,  70,  89, 124, 126,
         84, 115,   4,   1,  22, 110, 117,  44,  62,  29,  28,  61,  28, 115, 105,  12,
         45,  99,  70,   1,  65, 104,  21,  38,  71,  30,  66,  84,  52,   9,   7,  14,
         17, 122,  63, 103,  34,  11,  97,  34,  50,  67,  30,  85, 123,  93,  21,  72,
         51,  31,  51,  97,  50, 103,  40,  53,  63,  73,   4,  59, 122,  78, 113,  33,
        100,  91,  10,  58,  11,  47,  88,  19,  81,   6, 109,  34,  92,  94,  26,  27,
         25,  86,  74,  94, 105,  48,   2,  17, 109, 114, 111,  71, 108,  46,  47,  90,
         99,  90,  27,  91,  38,  92,  80,  36,  51,  39,  64,   7,  21, 124, 108,  37,
        117,  78,  66,  93,  61,  68,  21,  95, 101,  92,  55,  94,  96,  43,  51,  95,
         33,  60,  38,  49,  70,  25,  61, 121, 103, 107,  73, 104, 120,  57, 127,  13,
         90,  73,  77,  65,   0,  43,  86,  68,  60,  76,  15,  99,  41,  72, 113,  82,
         51,  66,  94,  30,  65, 104,  17,  54,   5,  23,  62,  43,  90,  26,  71, 117,
         92,  14,  32, 112,   7, 100,  59,  98,   5,   5,  44,  59,  26,  13,  96,  89,
         43,  52, 120,   5, 106,  78,  65, 127,  30, 110,  79,  49,  16,  34,  57,   3,
         16,  94, 109,  74,  38, 119,  52,  95,  83,  74,  64,  73,
    ),
    9: (
         80,  97,  25,   3, 125, 127, 101,  78,  65, 117, 104,  57,  64,  84,  84, 124,
        125,  30,   7,  66,  40,  82, 118, 123,  11,  15,  35, 100,  38, 102,  74,  17,
         89,  61,  28,  54,  56,  93,  38, 118, 126,  28,  81,  97,  32,  38, 126,  17,
        125, 103,  87,  95,  49,  18,  30,  64,   4,  86,  65,  62,   2,  22, 108,  16,
         41, 102,  89,  92,  57,   6,  39,   2,  43,  21, 100,  41,  65,  89,  41,  72,
         54,  37,  98, 106,  19, 105,  39,   3,  30,  87,  75, 114, 113,  49, 112,  12,
         35,  47,  83, 104,  37,  14,  41,  42,  43,  53,  38,   5,  69,   0,  23,  85,
         11,  47,  44,  85, 115, 105,  85, 123,  62, 105,  70, 116,  20,  74,  24,  57,
         43, 110,  14,  79,  31,   9, 114,  30, 100,  78,  26, 106,  82,  48,  11, 110,
         97,  88,  58,  21, 114,  37, 106,  90,  71,  94,  53,  55,  18, 117,  42,  74,
         38, 118,  15,   5,  78,  12,  69,  20,  54,  52,  70,  74,   9, 123,  76,  68,
         26, 115,  15,  31,  55,  51,  43,  84,  29,  11,  29,  49,  13,  70, 105, 115,
        119,  25,  90, 127,  82,  43, 102,  64,   9,  61,  43,  26,  18, 119,  26,  12,
         28,  85,  48, 102, 101,  38,  93,  30,  37,  68,   1,  85,  43,  82,  25, 105,
          3,  46, 115,  48,  82, 120,  24,  85,  63,  59,  80,  79,  14,  56,  61, 124,
         75,  82,  33,  93,  31,  18,  66, 108,  67,  93, 104,  23,  14,  44, 126,  95,
         46,   8,  93,  11, 113,   8,  34,   7,  61,   2,  55,  92,  34,  81, 118,  71,
          9,  61, 102, 121,  10,  12,  32, 112,  90,  41,  82, 107,  92, 105,  93,   9,
         91, 122,  29,   6,  82, 113,  69,  11, 102, 105, 102,  52, 114,  86,  10,  31,
         61, 108,   2,  47,  75,  40,  58,  97,  77, 124,  68,  69,  55,  27, 120,  29,
        110,  90,  54,  19,  26,  68,  71,  21,  29,  64,  94,  77,  33,  91,  63,  58,
         73,  48, 121, 114,   4, 120,  84, 113,   2,  45,  68,  25,
    ),
    10: (
         81,  86,  18,  53,  37,  71,  30,  23,   9,  23,  91, 122,  46,  92,  11,   2,
        112,  42,   6,  15, 117,  56,  35,   1,  98,  64,  67,  29,  56, 101, 125,   8,
         75,  81, 101, 108,  98,  91, 120,  63, 106,  98,  36,  28,  85,  59, 114,  76,
         17,  65,  84, 103,  35,  13,  11,  16,  96,  69,  49,  96,  77,   9, 100,  97,
         15,  14,  20,  54,  12,  74,  99,  84,  35,  73,  60, 112,   7,  28, 110, 119,
         54, 108,   7,  51,  22,  47,  46,  14,  49, 118,  56, 119,   7,  97,  75,  92,
         18,  49,  48,  84,  86, 120,  52,  92,  39, 111,  55,  59, 117, 100,  84,   3,
         82,  79,  50,  53,   5,   1,  95,   0,  70, 121, 107,  38,  24,  82,   6, 124,
         70,  93,  78,  12, 118,  72,  61,  89, 106, 120,  41,  64,  85, 120, 121, 118,
         91,  14, 105,  51,  34, 104, 116, 105,  19, 112,  96,  49,  53,  67, 104,  68,
         32,  56,   3,  33,  30,  30,  81,  80,  96,   7,  38,  67,  91,  60,  47,  84,
         68,  77,  65, 111,  33,  71,  64,  78,  14,   1,  44,  92,  17,  19,  71,  57,
         20, 100,  98,  87,   2,  94,  72,  69, 125,  66,  67,   7,  60,  19,  74,  33,
          0,  48,  75,  48,  55,   2,  22,  52,  42,  78,  95, 108,  96,  41, 116,  93,
         23,   6,   0, 114,  86, 110, 115,  59,  53,  75,  19,  82,  40,  97,  32,  95,
          5,  91,  32,  55,  24, 126,  74,  28,  91, 125,  16,  36,  88, 104,  71,  56,
         27,  67, 111,  55,  79,  48,  39,  55, 124,  72,  42, 103,  86,  11,   3,  22,
         71, 113,  80,  73, 118,  30,  51,  12,  21,  48,  82,  72,  43,  54,  16,  74,
         21,  61,  70, 104,  34,   7,  56,  60,  70,  90,  29,  11, 123, 116,  72,  71,
         55,  24, 113,  43,   8,  71,  33,   8,  76,  12,  59,  28,  13, 115,  37,  26,
        122,  25,  22,  74,   0,  89, 121,  79,   7,  41, 127,   1,  72,   0,  67, 117,
         75, 114,  22,  32,   0,  91, 106,  19, 116,  54,  91,  99,
    ),
    11: (
         50, 107,  17,  70, 109, 125,  50, 126,  20,  64, 105, 124,  84, 110,  46,  90,
         72,   5,  52,  16, 117,  61, 116,  56, 107,  41,  95,   1,  13, 109,  56,  48,
        111,  20, 125,  82,  87,  18, 126, 127,  99,  89,  13,  72,  70,  20,  80,  65,
         46,   0,  54,  41,  70, 106,   6,  54,  84,  68,  14, 125, 127,  97, 125, 127,
         47,  80,  78,  78,  18,  84,  95,  95,  55,   2,  30, 106,  89, 116,  97, 124,
         59,  41,  80,  28,  42, 127,   0,  28,  36,  72,  28, 120,  21,  10,  66, 106,
         56,  83, 105,  15,  33,  57, 123,  48,  85, 111,   4,  51,  19,   3,   5,  29,
         49,  52, 101,  62,  37, 112,   6, 127,  32, 108, 112,  70,  82,  92,  44,   7,
         78,  82,  49, 102,   9,  51, 116,   9,   0,  85,  30, 123, 117,  53,  95,  58,
         86,  84,  90,  97,  59,  49,  59, 114,  11,  98, 108,  59,  59,  75,  96, 105,
         51,   1,  11,  41,  77,  56, 112, 123, 124, 104,  84,  19,   6, 127,  21, 102,
        109,  62,  18,  90, 112,  54,  69,   9,  41,  94, 116,  42,  84,  76,   1,  36,
         44,  19,  90,  45, 115,  59,  17,  27,  13,  91,   9,  78,  71,  82,  73,  53,
         57,  37, 113, 125,  70, 116,  70,  88,  62,  21, 109, 105,  32,  81,  49,  77,
        126,  60,   2,  68, 125,  47, 124, 127, 119,  98,  30,   4,  63,  38,  31,  60,
          1,  62,  91,  29,  68,  93, 100, 116,  15,  58,  58,  74,  36,  79,  24, 119,
        108,  49, 104,  67, 111,  11,  85,  56, 126,  50, 124,   6, 112, 108,  62,  15,
          6,  61,  53,  16, 109, 102,  65, 112,  94,   2,  63,  33, 100,  59,  87,  53,
         57,  38, 111, 126,  78,  63,  63,  92, 103,   0,   3,  40,  95,  26,  71,  63,
         14,  85,  98,  40,  67,   4,   3,  96, 113,  32,  97,  68,  99,   8,  19,  30,
         43,  78, 124,  72,  29,  66,  96, 126, 116, 125,  25,  64,  16,  34,  95, 100,
         91,  58,  71,  58,  77,  40,  76,  88,  11,  73,   4,   6,
    ),
    12: (
         96,  50, 102,  71,  66,  46,  48,  84, 121,   4,  81,  25,  75,  79,  74,  67,
         55, 126,  17,  19, 120,  60, 111,  85,  62, 109, 124, 106, 102,  85,  55,   0,
        118,  87,  63,  94,  24,  88,  46,  95,  75,  63,  21, 121,  47,  26,  18, 121,
        105,  93,  13,  78,  50,  74,  19, 127, 126, 105,  73,  60,  72,  52,  18,  12,
        122,  81,  60,  39, 122, 123,  99,  98,  80, 126, 105,  68, 116,  93,  14,  51,
         53,  19,  41,  77,  67, 114,   8,  85,  93,  45,  99,  72,  22, 109, 123,  12,
         66,  35,  82,  69, 126,  89,   5,  68,  93,  69,  69,  28,  48,  93, 114,  83,
        109,  33,  87,  51,  64, 121,   2,  59,  46,  58,  61,  53, 100,  70,  40, 100,
         32,  79, 118,  90,  56,  57,  69, 102,  70,  26,  26,  38, 117,  95, 115,  62,
        102,  13,  13,  35,  59,  34,  55,  19,  68,  22,  74,  38,  70,   2,  74, 122,
        105,  91,  27,  14,  59,  92,  90,  46,  95, 127,  92,   6,  89,  38,  69,  86,
        115,  91,  62,  66,  73,  49,  88,  68, 116,  69,   0,  62,  81, 101,  24,  48,
        112,  75,  53,   2,  27,   7, 118, 127, 124, 121,   2,  96, 112, 118,  27,   1,
        106,  15,  93,  52,   0, 105,  28,  47,  18,  74,  81,  34,  25,  51,  68, 105,
         11,  36,  56,  91, 107, 127,  78,  62, 112,  11, 125,  88,  76,  20,  41,  35,
         30,  44,  96,  29, 113,  17,  53,  58, 113,  43,  51, 118,  48,  49,   2,  12,
         12, 103,  23,  61,  67,   7,  21,  77,  16,  59,  66,  18,   4,  66,  36,  31,
         97,  55,  59,  33,  50, 106,  86,  71, 103, 105,  25,  94,  44,  33,   7, 122,
         72,  26,  71,  75,  50,  98,   0,  19,  48,  10,  69,  50,  41,  69,  64,  89,
         11,  51,  95,  99,  85,  58,  55,  62,  70,  11, 118, 127,  31,  72,   6,  26,
         19,  66,  62,  44,  79,  86,  66, 114,  36, 126, 122,  91,  80,  98,  48,  55,
        108,  37,  67,   9, 113,  67,   8, 126,  99,  19,  64, 123,
    ),
}


_GRID_ROW_STRIDE = 471


def grid_row(b: int) -> np.ndarray:
    """Row b of the tone family's de-rotation grid, as lattice integers by column."""
    if b < 2:
        k = _grid480_k()
        lo = b * _GRID_ROW_STRIDE
        row = k[(k[:, 0] >= lo) & (k[:, 0] < lo + _GRID_ROW_STRIDE), 1]
        return np.append(row, _GRID480_TAIL[b])
    return np.array(_GRID_ROWS[b], dtype=np.int64)
