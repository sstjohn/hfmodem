# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The two interleavers: 188-110's block matrix and 4285's 32-row delay line.

Both are written as whole-array permutations rather than as the shift registers
the standards draw, because a transmission is rendered and decoded whole here,
and a permutation can be checked for being one.

**188-110** (110B 5.3.2.3.4-5). Coded bits load down the columns of the
matrix, the row stepping by 9 modulo 40 so that a column fills after 40 loads;
they are fetched starting at row 0, column 0, the row stepping by 1 and the
column back by 17 each fetch, and when the row wraps "the column number is
reset to be one larger than the value it had when the row number was last
zero". At 75 bit/s both steps are 7.

**4285** (Annex E 3-4). A convolutional interleaver of 32 rows: bit slot
``o`` of each 32-slot group goes out in position ``c = 9 o mod 32`` of its
group, delayed by ``c * J`` groups; the receiver delays position ``c`` by
``(31 - c) * J`` so every slot sees the same ``31 J``. Groups are aligned to
the first data symbol of a frame, which is where the transmission's first
group starts. The lines start empty (zeros), so the transmitter has to push
``31 J`` groups of padding through behind the message.
"""
from __future__ import annotations

import numpy as np



# --------------------------------------------------------------------------
# 188-110
# --------------------------------------------------------------------------

def block110_order(rows: int, cols: int, row_step: int = 9,
                   col_step: int = 17) -> np.ndarray:
    """``order[j]`` = the load index of the ``j``-th bit fetched. The steps are
    9 and 17 for the 40-row matrices and 7 and 7 at 75 bit/s."""
    k = np.arange(rows * cols)
    load_row = (row_step * (k % rows)) % rows
    load_col = k // rows
    cell_to_load = np.empty((rows, cols), dtype=np.int64)
    cell_to_load[load_row, load_col] = k
    p = np.repeat(np.arange(cols), rows)          # fetch pass
    r = np.tile(np.arange(rows), cols)
    c = (p - col_step * r) % cols
    return cell_to_load[r, c]


def interleave110(bits: np.ndarray, rows: int, cols: int, row_step: int = 9,
                  col_step: int = 17) -> np.ndarray:
    order = block110_order(rows, cols, row_step, col_step)
    b = np.asarray(bits).reshape(-1, rows * cols)
    return b[:, order].ravel()


def deinterleave110(values: np.ndarray, rows: int, cols: int, row_step: int = 9,
                    col_step: int = 17) -> np.ndarray:
    order = block110_order(rows, cols, row_step, col_step)
    v = np.asarray(values).reshape(-1, rows * cols)
    out = np.empty_like(v)
    out[:, order] = v
    return out.ravel()


# --------------------------------------------------------------------------
# 4285
# --------------------------------------------------------------------------

_ROWS = 32
_POS = (9 * np.arange(_ROWS)) % _ROWS            # input slot o -> output position


def interleave4285(slots: np.ndarray, increment: int) -> np.ndarray:
    """Groups of 32 in, the same number of groups out; the lines start at 0."""
    s = np.asarray(slots).reshape(-1, _ROWS)
    g = s.shape[0]
    out = np.zeros_like(s)
    for o in range(_ROWS):
        c = _POS[o]
        d = c * increment
        if d < g:
            out[d:, c] = s[: g - d, o]
    return out.ravel()


def deinterleave4285(slots: np.ndarray, increment: int) -> np.ndarray:
    """The inverse, non-causally: ``31 J`` groups fewer come out than went in,
    and the first one out is the transmitter's first group in."""
    s = np.asarray(slots).reshape(-1, _ROWS)
    n = s.shape[0] - 31 * increment
    if n <= 0:
        return np.zeros(0, dtype=s.dtype)
    out = np.empty((n, _ROWS), dtype=s.dtype)
    for o in range(_ROWS):
        c = _POS[o]
        d = c * increment
        out[:, o] = s[d:d + n, c]
    return out.ravel()
