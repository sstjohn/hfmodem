# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where shrike's constant tables come from: computed here, at the point of use.

The fixed tables `p3frame`/`pactor2`/`rx` need are built by this module every run.
Nothing is stored beside the code and read back, so there is no second copy to
drift. What holds the values is the conformance suites -- a table that came out
wrong decodes nothing -- and `tests/gates/test_tablegen.py` pins the shapes and
dtypes those suites index into. The generation methods are:

  * **Closed form**: the header symbol raster is one affine expression in the
    symbol index.
  * **Signed Hadamard-32**: the 16 acquisition markers are QPSK words whose 32
    bipolar chips are rows of a Hadamard-32 under a fixed column permutation. Both
    the P2 marker codes and the P3 `pre800` acquisition preamble are those same 32
    words -- `pre800` is the P2 codes with the two band halves swapped and scaled.
  * **SCS's published interleaver recurrence** (PACTOR-2 Annex II / PACTOR-4
    §11.6): eight permutations are that walk or a row/column product of two walks.
"""
from __future__ import annotations

from functools import cache
import math
from numbers import Integral

import numpy as np


# --------------------------------------------------------------------------- #
# Header symbol raster.
# --------------------------------------------------------------------------- #
@cache
def hdr_raster_src() -> np.ndarray:
    """Source symbol index for each of the 288 header cells: an affine map.

    `52 + 563*(3 - i%4) + i//4` -- four interleaved columns 563 apart, each
    counting up by one, which is a 4-deep block interleave written as a formula."""
    out = np.array([52 + 563 * (3 - i % 4) + i // 4 for i in range(288)], dtype=np.int64)
    out.setflags(write=False)
    return out


# --------------------------------------------------------------------------- #
# Acquisition markers: 16 QPSK words as signed Hadamard-32 rows.
# --------------------------------------------------------------------------- #
# Column permutation (the labeling points in F2^5) and, per word, its Walsh index
# and sign. bits[j] = popcount(a & PTS[j]) xor csign; chip = (-1)^bits, packed
# real/imag into 16 QPSK values of +-1+-1j. Real-orthogonal: the 16x16 Gram is
# 32*I + i*K with K integer antisymmetric, so the real parts are exactly orthogonal
# -- what a differential correlator needs.
_PTS = (0, 16, 8, 4, 2, 30, 18, 1, 21, 5, 7, 26, 3, 29, 31, 12,
        20, 25, 23, 6, 27, 11, 17, 14, 13, 9, 10, 15, 19, 22, 28, 24)
_MARKERS = ((8, 1), (2, 0), (9, 1), (11, 1), (18, 0), (16, 1), (27, 0), (25, 0),
            (28, 1), (30, 0), (21, 1), (23, 1), (14, 1), (12, 0), (7, 1), (5, 1))


def _marker_words() -> np.ndarray:
    """The 16 QPSK marker words, as a 16x16 complex array (+-1+-1j chips)."""
    rows = []
    for a, csign in _MARKERS:
        bits = np.array([((a & p).bit_count() + csign) % 2 for p in _PTS])
        x = 1 - 2 * bits
        rows.append(x[0::2] + 1j * x[1::2])
    return np.array(rows, dtype=np.complex64)


@cache
def marker_codes() -> np.ndarray:
    """P2 marker codes, `(2, 16, 8)` complex64: the two 8-chip halves of each word."""
    w = _marker_words()
    out = np.stack([w[:, :8], w[:, 8:]])
    out.setflags(write=False)
    return out


@cache
def pre800() -> np.ndarray:
    """P3 acquisition preamble, `(512,)` int64: the same words, +-800 real/imag taps.

    64 eight-tap groups; groups (2m, 2m+1) are the real and imaginary taps of the
    m-th complex 8-chip reference. The 32 references are the marker words with the
    band halves swapped -- references 0..15 are `codes[1]`, 16..31 are `codes[0]`."""
    codes = marker_codes()
    ref = np.concatenate([codes[1], codes[0]])          # 32 x 8, halves swapped
    taps = np.empty((64, 8))
    taps[0::2], taps[1::2] = ref.real * 800, ref.imag * 800
    out = taps.reshape(512).astype(np.int64)
    out.setflags(write=False)
    return out


# --------------------------------------------------------------------------- #
# PACTOR-III interleavers derivable from SCS's published recurrence.
# --------------------------------------------------------------------------- #
def _walk(n: int, m: int) -> np.ndarray:
    """SCS's interleave recurrence, read under `P >= N` (the in-range branch).

    S=1, P=0; OUT[i]=P; P+=M; if P>=N then P=S, S+=1. The other reading of the
    published pseudocode gives out-of-range indices wherever the two differ, so
    this is the only one that yields a bijection for every table."""
    out = np.empty(n, dtype=np.int64)
    s, p = 1, 0
    for i in range(n):
        out[i] = p
        p += m
        if p >= n:
            p, s = s, s + 1
    return out


def _grid(rows: int, cols: int, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """`t[rows*k + j] = cols*row[j] + col[k]`: a two-dimensional walk product."""
    return (cols * np.asarray(row)[None, :] + np.asarray(col)[:, None]).ravel()


def _cycfix(n: int, m: int) -> np.ndarray:
    """`k -> m*k mod (n-1)` with the last index fixed (the one column-rule table)."""
    out = np.empty(n, dtype=np.int64)
    out[:n - 1] = (m * np.arange(n - 1)) % (n - 1)
    out[n - 1] = n - 1
    return out


# The 8 tables the public recurrence reaches, keyed by their shipped npz name.
INTERLEAVERS = {
    "0xa4868_174":  lambda: _walk(174, 13),
    "0xb7510_580":  lambda: _walk(580, 24),
    "0xb2ef0_696":  lambda: _walk(696, 26),
    "0xb7f08_348":  lambda: _walk(348, 29),
    "0xb7998_696":  lambda: _walk(696, 29),
    "0xbaf20_1056": lambda: _grid(24, 44, _walk(24, 5), _walk(44, 22)),
    "0xb81c0_4968": lambda: _grid(24, 207, _walk(24, 5), _walk(207, 23)),
    "0xba890_840":  lambda: _grid(20, 42, _walk(20, 3), _cycfix(42, 18)),
}


# --------------------------------------------------------------------------- #
# Normal PACTOR-IV LONG symbol order and label-ordered constellations.
# These generators are independently checked; reader integration is separate.
# --------------------------------------------------------------------------- #
P4_POINT_SCALE = 4096


@cache
def p4_symbol_permutation() -> np.ndarray:
    """Direct field-order gather for all 4968 normal-LONG symbols, immutable intp.

    This is the existing 24-by-207 walk product, not its inverse. Bytes backing
    prevents a cached caller from restoring write permission.
    """
    values = _grid(24, 207, _walk(24, 5), _walk(207, 23)).astype(np.intp)
    return np.frombuffer(values.tobytes(), dtype=np.intp)


def p4_constellation_points(speedlevel) -> np.ndarray:
    """SL6--10 label-order signed axes: (2**(SL-5), 2), little-endian int16.

    Divide each axis by P4_POINT_SCALE for complex points. Validation precedes
    caching so equal nonintegral/bool cache keys cannot evade the contract.
    """
    if (isinstance(speedlevel, (bool, np.bool_))
            or not isinstance(speedlevel, Integral) or not 6 <= speedlevel <= 10):
        raise ValueError("speedlevel must be an integer from 6 through 10")
    return _p4_constellation_points(int(speedlevel))


@cache
def _p4_constellation_points(speedlevel: int) -> np.ndarray:
    width = speedlevel - 5
    rows = []
    if speedlevel == 10:
        # Equal inner-grid spacing and distance to the fixed 30-degree outer
        # point: 14*a**2 - 6*(cos(pi/6)+sin(pi/6))*a + 1 = 0.
        # The smaller root keeps the inner square inside the unit circle.
        c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
        a = (3 * (c + s) - math.sqrt(9 * (c + s) ** 2 - 14)) / 14
        primitives = (a, 3 * a, math.sqrt(1 - a * a), c, s)
        # One uniform three-decimal pre-rounding reproduces the reference.
        # It is an inferred compatibility rule, not a historical quantizer claim.
        lo, mid, outer, co, si = (round(round(v, 3) * P4_POINT_SCALE)
                                  for v in primitives)
    for label in range(1 << width):
        bits = [(label >> k) & 1 for k in range(width)]
        if speedlevel <= 8:
            gray, phase = label, 0
            while gray:
                phase ^= gray
                gray >>= 1
            angle = 2 * math.pi * phase / (1 << width)
            x = round(P4_POINT_SCALE * math.cos(angle))
            y = round(P4_POINT_SCALE * math.sin(angle))
        elif speedlevel == 9:
            quadrant = (label >> 2) ^ ((label >> 2) >> 1)
            if bits[0] and bits[1]:
                x = y = round(P4_POINT_SCALE * math.sin(math.pi / 12))
            else:
                step = 1 + (-1 if quadrant % 2 else 1) * (bits[0] - bits[1])
                angle = step * math.pi / 6
                x = round(P4_POINT_SCALE * math.cos(angle))
                y = round(P4_POINT_SCALE * math.sin(angle))
            for _ in range(quadrant):
                x, y = -y, x
        else:
            if bits[2]:
                x, y = (lo if bits[1] else mid), (lo if bits[0] else mid)
            else:
                x, y = (outer, lo) if bits[0] else (co, si)
                if bits[1]:
                    x, y = y, x
            x *= -1 if bits[3] else 1
            y *= -1 if bits[4] else 1
        rows.append((x, y))
    raw = np.asarray(rows, dtype="<i2").tobytes()
    return np.frombuffer(raw, dtype="<i2").reshape(1 << width, 2)


# --------------------------------------------------------------------------- #
# The two DSP kernels: the symbol pulse and the acquisition front end.
# --------------------------------------------------------------------------- #
SPS = 8
"""Samples per symbol both kernels are written at: 800 Hz against 100 Bd."""

ROLLOFF = 2 / 3
"""[SCS-P4] s11.8 prints a 32-coefficient, 8-per-symbol, unit-sum pulse and calls
it a quasi-RRC "optimized concerning spectral side lobes and orthogonality". A
true RRC at 2/3 tracks that published pulse to 3.3e-3; the same fit against
s10.1's 129-tap 1800 Bd filter returns 0.330 where that section prints 0.33."""


def _rrc(n: int, beta: float) -> np.ndarray:
    """Root-raised cosine, `n` taps at `SPS` per symbol, symmetric, unit sum."""
    t = (np.arange(n) - (n - 1) / 2) / SPS
    with np.errstate(divide="ignore", invalid="ignore"):
        h = ((np.sin(np.pi * t * (1 - beta))
              + 4 * beta * t * np.cos(np.pi * t * (1 + beta)))
             / (np.pi * t * (1 - (4 * beta * t) ** 2)))
    h[np.isclose(t, 0.0)] = 1 - beta + 4 * beta / np.pi
    edge = np.isclose(np.abs(t), 1 / (4 * beta))
    h[edge] = beta / np.sqrt(2) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return h / h.sum()


@cache
def symbol_pulse() -> np.ndarray:
    """The 31-tap transmit and matched-filter pulse: `ROLLOFF`, `SPS` per symbol.

    The roll-off is the spec's, and the air agrees with the spec: swept against
    the six packet headers of a real PACTOR-III session, the constant-header fit
    peaks at 2/3 -- 0.8510 worst and 0.8759 mean, falling away on both sides and
    down to 0.8412 by 0.4. A pulse optimised instead for s11.8's stated criteria,
    least out-of-band energy under exact orthogonality, reads that same session
    at 0.8481: better in theory and measurably worse on what SCS transmitters
    actually put on the band, which is what decides it.
    """
    return _rrc(31, ROLLOFF)


@cache
def acq_window() -> np.ndarray:
    """The 32-deep acquisition ring's window: a unit-roll-off RRC on 27 taps.

    Not `symbol_pulse`. The front end runs at the pulse's own eight samples per
    symbol, which makes it tempting to reuse, but the window is a wider kernel
    with its own signature: at beta = 1 the numerator collapses to
    `4t*cos(2*pi*t)`, whose zeros land at three quarters and five quarters of a
    symbol -- samples 6 and 10 either side of the peak. The 27 taps sit at
    indices 0..26 of the ring, peak at 13, so the transform's
    newest-sample-first blocks meet the pulse centred rather than five late.
    """
    w = np.zeros(32)
    w[:27] = _rrc(27, 1.0)
    return w


@cache
def acq_lowpass() -> np.ndarray:
    """The front end's decimation low-pass: 96 taps, 400 Hz at 9600, Kaiser 6.5."""
    from scipy.signal import firwin
    return firwin(96, 400.0, fs=9600, window=("kaiser", 6.5))
