# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The extended Golay (24, 12) code of MIL-STD-188-141B A.5.2.2.2.

The parity matrix is figure A-6's; with it the code has minimum distance 8,
and the text's own example (figure A-8: data 110100010101, check bits
010101100110) is reproduced by ``tests/bateleur/ale/test_coding.py``.

Decoding is by syndrome table, correcting up to three errors in a half word
and detecting four, which is the 3/4 mode of A.5.2.6.3. ``decode_soft`` adds
the received bits' reliabilities: it tries the syndrome decoder on the word
with each combination of its least reliable bits flipped, and keeps the
codeword the received values agree with most.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

#: Figure A-6, the parity part P of G = [I12 | P]: row i is the check bits
#: data bit i (MSB first) contributes.
_P_ROWS = (
    "101011100011", "111110010010", "110100101011", "110001110110",
    "110011011001", "011001101101", "001100110111", "101101111000",
    "010110111100", "001011011110", "101110001101", "010111000111",
)
_P = [int(r, 2) for r in _P_ROWS]


def check_bits(data: int) -> int:
    """The 12 check bits (G1 as the MSB) for 12 data bits (W1 as the MSB)."""
    out = 0
    for i in range(12):
        if (data >> (11 - i)) & 1:
            out ^= _P[i]
    return out


_CHECK = np.array([check_bits(d) for d in range(4096)], dtype=np.int32)


def encode(data: int) -> int:
    """12 data bits -> the 24-bit codeword, data in the top half."""
    return ((data & 0xFFF) << 12) | int(_CHECK[data & 0xFFF])


def _syndrome(word: int) -> int:
    return int(_CHECK[(word >> 12) & 0xFFF]) ^ (word & 0xFFF)


def _build_table() -> tuple[np.ndarray, np.ndarray]:
    """syndrome -> (error pattern, weight) for every pattern of weight <= 3;
    weight-4 patterns share syndromes and are left marked uncorrectable."""
    pattern = np.zeros(4096, dtype=np.int32)
    weight = np.full(4096, 99, dtype=np.int8)
    weight[0] = 0
    for w in (1, 2, 3):
        for pos in combinations(range(24), w):
            e = 0
            for p in pos:
                e |= 1 << p
            s = _syndrome(e)
            if weight[s] > w:
                weight[s], pattern[s] = w, e
    return pattern, weight


_PATTERN, _WEIGHT = _build_table()


def decode(word: int) -> tuple[int, int]:
    """A received 24-bit word -> (12 data bits, errors corrected), with the
    error count -1 where the word is uncorrectable (four or more)."""
    s = _syndrome(word)
    w = int(_WEIGHT[s])
    if w > 3:
        return (word >> 12) & 0xFFF, -1
    return ((word ^ int(_PATTERN[s])) >> 12) & 0xFFF, w


_FLIPS = 4                                   # least reliable bits tried
#: Every combination of the ``_FLIPS`` weakest bits, as rows of 0/1.
_COMBOS = (np.arange(1 << _FLIPS)[:, None] >> np.arange(_FLIPS)) & 1
_SHIFTS = np.arange(23, -1, -1)
_WEIGHTS = (1 << _SHIFTS).astype(np.int64)


def decode_soft(llr: np.ndarray) -> tuple[int, int, float]:
    """24 soft values, MSB first, positive meaning 0 -> (data, errors, score).

    ``errors`` counts the hard-decision bits the chosen codeword disagrees
    with (-1 if no candidate decoded); ``score`` is the received values'
    correlation with that codeword, normalised to 1 for a perfect match.
    """
    llr = np.asarray(llr, dtype=np.float64)
    hard_bits = (llr < 0).astype(np.int64)
    hard = int(hard_bits @ _WEIGHTS)
    weakest = np.argsort(np.abs(llr))[:_FLIPS]
    cand = hard ^ (_COMBOS @ _WEIGHTS[weakest])
    syn = _CHECK[(cand >> 12) & 0xFFF] ^ (cand & 0xFFF)
    ok = _WEIGHT[syn] <= 3
    if not np.any(ok):
        return (hard >> 12) & 0xFFF, -1, -1.0
    cw = np.unique(cand[ok] ^ _PATTERN[syn[ok]])
    bits = (cw[:, None] >> _SHIFTS) & 1
    total = float(np.sum(np.abs(llr))) or 1.0
    score = ((1 - 2 * bits) @ llr) / total
    j = int(np.argmax(score))
    return int(cw[j] >> 12) & 0xFFF, int(np.count_nonzero(bits[j] != hard_bits)), float(score[j])
