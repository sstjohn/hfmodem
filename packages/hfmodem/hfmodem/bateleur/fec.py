# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The K=7 rate-1/2 convolutional code, and a soft-decision Viterbi for it.

Both standards use the same code (``spec.CONV_MASKS``), run continuously across
the whole transmission from a zero register and never tail-terminated — the
flush is data-level, zeros after the message — so the decoder pins the start and
leaves the end free.

This is a copy in kind of ``shrike.coding``'s decoder and not an import of it:
protocols do not import each other here (``tests/gates/test_import_direction``),
and the two have nothing in common but the algorithm.

Soft convention: one value per coded bit, positive for 0 and negative for 1,
scaled as a log-likelihood ratio; 0.0 is an erasure, which is how a punctured
bit is passed in.
"""
from __future__ import annotations

import numpy as np

from .spec import CONV_K, CONV_MASKS

_NSTATES = 1 << (CONV_K - 1)


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


def encode(bits, state: int = 0) -> tuple[np.ndarray, int]:
    """Coded bits T1, T2 per input bit, and the register to continue from."""
    b = np.asarray(bits, dtype=np.int64).ravel()
    out = np.empty(2 * b.size, dtype=np.uint8)
    m1, m2 = CONV_MASKS
    reg = state
    for i, bit in enumerate(b):
        reg = ((reg << 1) | int(bit)) & 0x7F
        out[2 * i] = _parity(reg & m1)
        out[2 * i + 1] = _parity(reg & m2)
    return out, reg & (_NSTATES - 1)


def _trellis():
    """For each next state s = ((prev << 1) | bit) & 63: its two predecessors
    and the +/-1 amplitudes of T1 and T2 on each branch."""
    m1, m2 = CONV_MASKS
    pred = np.zeros((_NSTATES, 2), dtype=np.int64)
    amp = np.zeros((_NSTATES, 2, 2))
    for s in range(_NSTATES):
        for j in range(2):
            p = (s >> 1) | (j << 5)          # the bit that fell off the end
            reg = (p << 1) | (s & 1)
            pred[s, j] = p
            amp[s, j, 0] = 1 - 2 * _parity(reg & m1)
            amp[s, j, 1] = 1 - 2 * _parity(reg & m2)
    return pred, amp


_PRED, _AMP = _trellis()


def viterbi(soft, *, free_start: bool = False) -> np.ndarray:
    """Decode a stream that started from the zero state (or, ``free_start``,
    any state), end left free."""
    soft = np.asarray(soft, dtype=np.float64).ravel()
    decoder = ViterbiStream()
    decoder.depth = soft.size // 2
    if free_start:
        decoder._metric[:] = 0.0
    decoder.push(soft)
    return decoder.finish()


def encode_tailbiting(bits) -> np.ndarray:
    """MIL-STD-188-110B C.5.3.2.2's full tail-biting: the register preloaded
    with the first six bits, which then go in again at the end. The output is
    twice the input, its pairs belonging to input bits 6, 7, ... n-1, 0 ... 5."""
    b = np.asarray(bits, dtype=np.int64).ravel()
    state = 0
    for bit in b[:CONV_K - 1]:
        state = ((state << 1) | int(bit)) & (_NSTATES - 1)
    out, _ = encode(np.roll(b, -(CONV_K - 1)), state)
    return out


def viterbi_tailbiting(soft, wrap: int = 48) -> np.ndarray:
    """Decode a tail-biting block: the trellis run round the circle, ``wrap``
    steps of it both sides of the block, so the survivors there have settled
    into the true state, then the middle kept. Returns the input bits in
    their original order (`encode_tailbiting` starts at bit 6)."""
    soft = np.asarray(soft, dtype=np.float64).ravel()
    n = soft.size // 2
    wrap = min(wrap, n)
    ext = np.concatenate([soft[2 * (n - wrap):], soft, soft[:2 * wrap]])
    steps = viterbi(ext, free_start=True)[wrap: wrap + n]
    return np.roll(steps, CONV_K - 1)


class ViterbiStream:
    """The same decoder fed a piece at a time, committing each bit once every
    survivor agrees on it — in practice once it is ``depth`` steps old, which
    at five constraint lengths and more costs nothing measurable."""

    def __init__(self, depth: int = 96) -> None:
        self.depth = depth
        self._metric = np.full(_NSTATES, -1e30)
        self._metric[0] = 0.0
        self._back: list[np.ndarray] = []
        self._pending = np.zeros(0)             # an odd soft value waits for its pair

    def push(self, soft) -> np.ndarray:
        s = np.concatenate([self._pending, np.asarray(soft, dtype=np.float64).ravel()])
        n = s.size // 2
        self._pending = s[2 * n:]
        metric = self._metric
        p0, p1 = _PRED[:, 0], _PRED[:, 1]
        a00, a01 = _AMP[:, 0, 0], _AMP[:, 0, 1]
        a10, a11 = _AMP[:, 1, 0], _AMP[:, 1, 1]
        for t in range(n):
            r0, r1 = s[2 * t], s[2 * t + 1]
            c0 = metric[p0] + a00 * r0 + a01 * r1
            c1 = metric[p1] + a10 * r0 + a11 * r1
            take = c1 > c0
            metric = np.where(take, c1, c0)
            self._back.append(take.astype(np.uint8))
        metric -= metric.max()
        self._metric = metric
        ready = len(self._back) - self.depth
        if ready <= 0:
            return np.zeros(0, dtype=np.uint8)
        bits = self._trace()[:ready]
        del self._back[:ready]
        return bits

    def pending(self) -> np.ndarray:
        """Tentative tail, for checking flush after an already committed EOM."""
        return self._trace()

    def finish(self) -> np.ndarray:
        """Everything still held, traced back from the best state."""
        bits = self._trace()
        self._back.clear()
        return bits

    def _trace(self) -> np.ndarray:
        state = int(np.argmax(self._metric))
        bits = np.empty(len(self._back), dtype=np.uint8)
        for t in range(len(self._back) - 1, -1, -1):
            bits[t] = state & 1
            state = int(_PRED[state, self._back[t][state]])
        return bits
