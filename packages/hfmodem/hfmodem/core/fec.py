# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Convolutional coding, soft Viterbi and puncturing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

@dataclass(frozen=True)
class ConvCode:
    """Rate-1/2 feed-forward convolutional code with explicit generators."""
    constraint_length: int
    generators: tuple[int, int]

    @property
    def n_states(self) -> int:
        return 1 << (self.constraint_length - 1)

    def _taps(self) -> tuple[np.ndarray, np.ndarray]:
        K = self.constraint_length
        out = []
        for g in self.generators:
            out.append(np.array([(g >> (K - 1 - i)) & 1 for i in range(K)],
                                dtype=np.uint8))
        return out[0], out[1]

    def encode(self, bits: np.ndarray, terminate: bool = True) -> np.ndarray:
        """Encode, MSB-first, optionally flushing the register with K-1 zeros."""
        K = self.constraint_length
        g0, g1 = self._taps()
        bits = np.asarray(bits, dtype=np.uint8)
        if terminate:
            bits = np.concatenate([bits, np.zeros(K - 1, dtype=np.uint8)])
        # shift register, newest bit first
        reg = np.zeros(K, dtype=np.uint8)
        out = np.empty(bits.size * 2, dtype=np.uint8)
        for i, b in enumerate(bits):
            reg[1:] = reg[:-1]
            reg[0] = b
            out[2 * i] = np.bitwise_xor.reduce(reg & g0)
            out[2 * i + 1] = np.bitwise_xor.reduce(reg & g1)
        return out


@lru_cache(maxsize=None)
def _trellis(code: ConvCode) -> tuple[np.ndarray, ...]:
    """Branch amplitudes and predecessors for `code` -- a pure function of the code.

    State convention (must match encode()): the state holds the K-1 PREVIOUS input
    bits, with bit 0 = most recent. Feeding input b therefore yields
        next = ((state << 1) | b) & mask
    and so, going backwards, the input that produced a given state is simply
    `state & 1` -- which is what makes viterbi_decode's traceback unambiguous.

    Cached because the timing search calls the decoder once per candidate symbol
    position and this was rebuilt every time -- 512 ufunc reduces per call at K=9,
    which cost more than the trellis sweep it was setting up for.
    """
    K = code.constraint_length
    g0, g1 = code._taps()
    states = np.arange(code.n_states)
    out = np.zeros((code.n_states, 2, 2), dtype=np.int8)
    for s in range(code.n_states):
        for b in (0, 1):
            reg = np.empty(K, dtype=np.uint8)
            reg[0] = b                                  # newest bit
            for i in range(1, K):
                reg[i] = (s >> (i - 1)) & 1             # older bits
            out[s, b, 0] = np.bitwise_xor.reduce(reg & g0)
            out[s, b, 1] = np.bitwise_xor.reduce(reg & g1)

    # +1 for coded bit 0, -1 for coded bit 1: the branch metric is a correlation,
    # which we MAXIMISE. Passing 0.0 for a punctured bit contributes nothing --
    # exactly the "exactly intermediate value" the spec asks for.
    amp = np.where(out == 0, 1.0, -1.0)

    # For every next-state, its two predecessors and the input bit that got there.
    b_of = (states & 1).astype(np.int64)
    pred0 = (states >> 1).astype(np.int64)
    pred1 = pred0 | (1 << (K - 2))
    return (amp[pred0, b_of, 0], amp[pred0, b_of, 1],
            amp[pred1, b_of, 0], amp[pred1, b_of, 1], pred0, pred1)


def viterbi_decode(soft: np.ndarray, code: ConvCode, terminated: bool = True,
                   pinned: bool = False) -> np.ndarray:
    """Soft Viterbi: positive values mean zero; zero values are erasures.

    Terminated removes K-1 tail bits. Pinned fixes both boundary states
    at zero; otherwise both boundaries are free.
    """
    K = code.constraint_length
    n_states = code.n_states
    soft = np.asarray(soft, dtype=np.float64)
    if soft.size % 2:
        raise ValueError("expected an even number of coded soft values")
    n_steps = soft.size // 2
    a0_0, a0_1, a1_0, a1_1, pred0, pred1 = _trellis(code)

    if pinned:
        metric = np.full(n_states, -1e12)
        metric[0] = 0.0
    else:
        metric = np.zeros(n_states)
    back = np.zeros((n_steps, n_states), dtype=np.int32)

    for t in range(n_steps):
        r0, r1 = soft[2 * t], soft[2 * t + 1]
        cand0 = metric[pred0] + (a0_0 * r0 + a0_1 * r1)
        cand1 = metric[pred1] + (a1_0 * r0 + a1_1 * r1)
        take1 = cand1 > cand0
        metric = np.where(take1, cand1, cand0)
        back[t] = np.where(take1, pred1, pred0)

    state = 0 if pinned else int(np.argmax(metric))
    bits = np.zeros(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        bits[t] = state & 1          # the input bit is the low bit of the state
        state = int(back[t, state])
    if terminated:
        bits = bits[: n_steps - (K - 1)]
    return bits


@dataclass(frozen=True)
class Puncture:
    """A puncture pattern over the rate-1/2 output stream.

    `mask` is a 2 x period array of 0/1; 1 = transmit, 0 = delete, row 0 for the
    G0 output and row 1 for G1, with the period counted in trellis steps. Rate 1/2
    is the all-ones mask, which makes the stage a no-op rather than a branch at
    every caller that does not puncture.
    """
    mask: tuple[tuple[int, ...], ...]

    @property
    def period(self) -> int:
        return len(self.mask[0])

    @property
    def rate(self) -> tuple[int, int]:
        kept = sum(sum(row) for row in self.mask)
        return (self.period, kept)

    def apply(self, coded: np.ndarray) -> np.ndarray:
        m = np.array(self.mask, dtype=bool)          # shape (2, period)
        pairs = coded.reshape(-1, 2).T               # shape (2, n)
        n = pairs.shape[1]
        keep = np.tile(m, (1, int(np.ceil(n / self.period))))[:, :n]
        return pairs.T[keep.T]                       # column-major, time-ordered

    def depuncture(self, soft: np.ndarray, n_pairs: int) -> np.ndarray:
        """Reinsert punctured positions as 0.0 (the neutral value)."""
        m = np.array(self.mask, dtype=bool)
        keep = np.tile(m, (1, int(np.ceil(n_pairs / self.period))))[:, :n_pairs]
        full = np.zeros((2, n_pairs), dtype=np.float64)
        full.T[keep.T] = soft
        return full.T.ravel()
