# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CA-TBCC: tail-biting convolutional code + CRC-aided list Viterbi.

The floor/control code. At n ~ 128-512 a tail-biting convolutional code
decoded by a CRC-screened list Viterbi sits ~0.5-1 dB closer to the
finite-blocklength bound than short binary LDPC -- the hardest dB in the
system to buy anywhere else. Every component is decades-old prior art:
tail-biting termination (Ma & Wolf 1986), list Viterbi (Seshadri & Sundberg
1994), wrap-around decoding of tail-biting trellises (WAVA, Shao et al.
2003), and CRC screening of the list.

The code is rate-1/2, K = 12 (memory 11), generators (4335, 5723) octal --
the published optimum-free-distance pair (Larsen 1973), dfree = 15. Octal
convention: the MSB of each generator taps the current input bit.

The decoder makes ``wraps - 1`` plain wrap-around Viterbi passes to converge
per-state tail-biting boundary metrics, then one list pass over the block
itself, keeping a survivor list per state -- so candidate paths differ
anywhere in the block, which is what makes the list worth screening.
Candidates are screened by a caller-supplied byte-level predicate (here: the
frame CRC every sabir block already carries), so the CRC pays for list
selection as well as error detection.

LLR convention: LLR > 0 <=> bit 0 (matches ``fec.ldpc`` and the demappers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

GEN12 = (0o4335, 0o5723)


@dataclass(frozen=True)
class ConvCode:
    """Rate-1/2 feed-forward convolutional code, any constraint length."""

    constraint_length: int = 12
    generators: tuple[int, int] = GEN12

    @property
    def n_states(self) -> int:
        return 1 << (self.constraint_length - 1)

    def taps(self) -> np.ndarray:
        """(2, K) 0/1; column t taps the input delayed by t symbols."""
        K = self.constraint_length
        return np.array([[(g >> (K - 1 - t)) & 1 for t in range(K)]
                         for g in self.generators], dtype=np.int64)

    def encode_tb(self, bits: np.ndarray) -> np.ndarray:
        """Tail-biting encode: k input bits -> 2k coded bits, interleaved
        c0[0], c1[0], c0[1], ... The shift register starts loaded with the
        block's last K-1 bits, so the trellis path is circular."""
        b = np.asarray(bits, dtype=np.int64).ravel()
        K = self.constraint_length
        if b.size < K:
            raise ValueError(f"tail-biting block must be >= K = {K} bits")
        g = self.taps()
        delayed = np.stack([np.roll(b, t) for t in range(K)])
        return ((g @ delayed) % 2).T.ravel()

    def free_distance(self, max_steps: int = 200) -> int:
        """Min weight over paths that leave and re-merge with the zero state."""
        w0, w1, p0, p1 = self._trellis_weights()
        INF = 1 << 30
        dist = np.full(self.n_states, INF, dtype=np.int64)
        dist[1] = w0[1]                    # the single leave-0 branch (p0[1] = 0)
        best = INF
        for _ in range(max_steps):
            nd = np.minimum(dist[p0] + w0, dist[p1] + w1)
            best = min(best, int(nd[0]))
            nd[0] = INF
            if best < INF and (nd >= dist).all():
                break
            dist = np.minimum(dist, nd)
        return best

    def _trellis(self):
        """Per next-state: predecessors p0/p1 and their 2-bit branch outputs."""
        K = self.constraint_length
        S = self.n_states
        states = np.arange(S)
        b = states & 1                       # input bit that enters this state
        p0 = states >> 1
        p1 = p0 | (1 << (K - 2))
        g = self.taps()                      # column 0 = current input

        def sym(pred):
            pbits = (pred[:, None] >> np.arange(K - 1)) & 1   # delay 1..K-1
            o = (b[:, None] * g[:, 0] + pbits @ g[:, 1:].T) % 2
            return 2 * o[:, 0] + o[:, 1]

        return p0, p1, sym(p0), sym(p1)

    def _trellis_weights(self):
        p0, p1, s0, s1 = self._trellis()
        w = np.array([0, 1, 1, 2])
        return w[s0], w[s1], p0, p1


class CATBCC:
    """Tail-biting ConvCode over byte blocks, list-Viterbi + CRC screening.

    ``decode``'s ``check`` receives candidate blocks as bytes, best metric
    first, and the first block it accepts wins; with no ``check`` the best
    path is returned unscreened.
    """

    def __init__(self, code: ConvCode | None = None, list_size: int = 24,
                 wraps: int = 3, state_list: int = 4):
        self.code = code or ConvCode()
        self.list_size = list_size
        self.wraps = wraps
        self.state_list = state_list
        self._p0, self._p1, self._s0, self._s1 = self.code._trellis()

    def encode(self, block: bytes) -> np.ndarray:
        bits = np.unpackbits(np.frombuffer(block, dtype=np.uint8))
        return self.code.encode_tb(bits)

    def decode(self, llr: np.ndarray, n_bytes: int,
               check: Callable[[bytes], bool] | None = None
               ) -> tuple[bytes | None, dict]:
        k = 8 * n_bytes
        llr = np.asarray(llr, dtype=np.float64).ravel()
        if llr.size != 2 * k:
            raise ValueError(f"expected {2 * k} LLRs, got {llr.size}")
        cands = self._list_viterbi(self._branch_metrics(llr), k)
        seen: list[bytes] = []
        for rank, bits in enumerate(cands):
            block = np.packbits(bits).tobytes()
            if block in seen:
                continue
            seen.append(block)
            if check is None or check(block):
                return block, {"ok": True if check else None, "rank": rank,
                               "tried": len(seen)}
        return None, {"ok": False, "rank": -1, "tried": len(seen)}

    def _branch_metrics(self, llr: np.ndarray) -> np.ndarray:
        """Correlation branch metric for each output symbol 2*c0 + c1, for
        every stage at once -- (k, 4).

        The trellis recursion is sequential, so the k (or 3k, wrapped) steps
        stay a Python loop; building their metrics one 4-vector at a time
        inside it was pure allocation overhead, and the wrap made it three
        allocations for every distinct stage.
        """
        l0, l1 = llr[0::2], llr[1::2]
        return 0.5 * np.stack([l0 + l1, l0 - l1, -l0 + l1, -l0 - l1], axis=1)

    def _list_viterbi(self, bm_tab: np.ndarray, k: int) -> np.ndarray:
        """Wrap-around warm-up + one k-step list pass over the block;
        returns candidate decisions (n_candidates, k), best metric first."""
        S = self.code.n_states
        L = self.state_list
        p0, p1, s0, s1 = self._p0, self._p1, self._s0, self._s1
        half = 1 << (self.code.constraint_length - 2)

        boundary = np.zeros(S)
        for t in range((self.wraps - 1) * k):
            bm = bm_tab[t % k]                      # the wrap repeats the block
            boundary = np.maximum(boundary[p0] + bm[s0],
                                  boundary[p1] + bm[s1])

        metric = np.full((S, L), -np.inf)
        metric[:, 0] = boundary
        back = np.empty((k, S, L), dtype=np.uint8)
        for t in range(k):
            bm = bm_tab[t]
            if L == 1:
                # one survivor per state is a two-way choice, so the general
                # partition below is all overhead: 0 keeps p0, 1 takes p1,
                # which is exactly what the traceback reads out of `back`.
                a = metric[p0, 0] + bm[s0]
                b = metric[p1, 0] + bm[s1]
                take1 = b > a
                metric = np.where(take1, b, a)[:, None]
                back[t] = take1[:, None]
                continue
            cand = np.concatenate(
                [metric[p0] + bm[s0][:, None], metric[p1] + bm[s1][:, None]],
                axis=1)
            sel = np.argpartition(-cand, L - 1, axis=1)[:, :L]
            metric = np.take_along_axis(cand, sel, axis=1)
            back[t] = sel

        flat = metric.ravel()
        n_cand = min(self.list_size, flat.size)
        top = np.argpartition(-flat, n_cand - 1)[:n_cand]
        top = top[np.argsort(-flat[top])]
        state, slot = top // L, top % L

        bits = np.empty((n_cand, k), dtype=np.uint8)
        for t in range(k - 1, -1, -1):
            bits[:, t] = state & 1
            sel = back[t, state, slot]
            from_p1 = sel >= L
            slot = sel - L * from_p1
            state = (state >> 1) | (half * from_p1)
        return bits
