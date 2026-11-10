# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""QC-LDPC code ladder: our own codes, encoder, and min-sum decoder.

Each code is a quasi-cyclic lifting (Z = 64 for the ladder proper; ``z`` is a
parameter so short comparison codes can reuse the structure) of an mb x nb
base matrix, giving (n, k) = (nb*Z, (nb-mb)*Z). The information columns carry
the code's character and the parity part is the classic dual-diagonal
accumulator structure with one weight-3 anchor column, so encoding is a short
shift-and-XOR recursion. The rungs are deliberately *independent* codes, not a
protograph mother code extended into a nested rate family.

The base matrices are fixed data, transcribed from SPEC.md §7.1.5, which is
normative for them (§7.1.4 records the construction search that once produced
them). Everything else here is decades-old open technique: QC lifting,
dual-diagonal (Richardson-Urbanke) encoding, and normalized min-sum decoding.

Circulant convention: the shift-s block is P_s with (P_s v)[r] = v[(r - s) mod Z],
i.e. P_s v == np.roll(v, s).

LLR convention: LLR > 0 <=> bit 0 (matches ``Constellation.llr``).
"""

import numpy as np

Z = 64


def _grid(text: str) -> np.ndarray:
    """A SPEC.md §7.1.5 shift grid: entries are circulant shifts, ``·`` the
    zero block (stored as -1)."""
    return np.array([[-1 if tok == "·" else int(tok) for tok in line.split()]
                     for line in text.splitlines() if line.strip()],
                    dtype=np.int16)


# Base matrices, verbatim from SPEC.md §7.1.5 — data, not the output of any
# generator; the spec is normative and test_kat pins each grid to its
# SHA-256.
CODES = {
    # (1536, 512)  rate 1/3
    "r13": _grid("""
          ·   ·  54   ·   ·   ·   ·  18  57   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
          4   ·   ·   ·   ·  42   ·   9   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
          ·   8   ·   ·  62  27   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
          ·   ·   ·  52  34   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
          ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
          3   ·  18   ·   2   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·   ·
          ·   ·   ·   ·   ·  51   0  62   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·   ·
          ·   ·  50   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·   ·
          ·  52   2   ·   ·   ·   6   ·   0   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·   ·
          ·   6   ·  58  25   ·   ·  21   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·   ·
          ·  51   ·  32   ·   ·  51   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·   ·
          ·   ·   ·  24   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·   ·
         46  24  46   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·   ·
         22   ·   ·   ·  59  39   ·   8   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·
         38   ·   ·   ·   ·   ·  18   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0   0
          ·   ·   ·   6   ·  50  12   ·  57   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   0
    """),
    # (1024, 512)  rate 1/2
    "r12": _grid("""
          ·  48   ·   ·   ·  12  20   ·  38   0   ·   ·   ·   ·   ·   ·
          ·   ·   9   ·   ·  37   ·   1   ·   0   0   ·   ·   ·   ·   ·
          9   ·  30   ·   ·   ·  40   ·   ·   ·   0   0   ·   ·   ·   ·
          5  46   ·   ·   ·   ·   ·  14   ·   ·   ·   0   0   ·   ·   ·
         32   ·   ·   5   ·   ·   ·  58   0   ·   ·   ·   0   0   ·   ·
          ·  50  23   ·   5   ·   ·   ·   ·   ·   ·   ·   ·   0   0   ·
          ·   ·   ·  39  50  49   ·   ·   ·   ·   ·   ·   ·   ·   0   0
          ·   ·   ·  18  59   ·  34   ·  38   ·   ·   ·   ·   ·   ·   0
    """),
    # (1024, 768)  rate 3/4
    "r34": _grid("""
         39  44  27  32  20   ·   8   3  35   5   ·  41  37   0   ·   ·
          ·  30  59  42   ·  15   ·   ·   1  42  58  17   ·   0   0   ·
          6  13   ·   ·  39  28   1  24  55  51  12  53   0   ·   0   0
         57   ·   0  48  13  55  41   6   ·   ·  43   ·  37   ·   ·   0
    """),
    # (1536, 1280) rate 5/6
    "r56": _grid("""
         28   9  25  51  62  49  42   ·   ·  58  56   ·  45  28  19  34  39  14  36   ·  52   0   ·   ·
         25   3  17  13   4   ·   ·  33  23  12   ·  42  47  42   9   ·  12  61   ·  50   ·   0   0   ·
         37   ·   ·   ·  11   4   0  20  51  36  27  16  28  27  24  15   ·   ·  43  49   0   ·   0   0
          ·   8  39  28   ·  38  57   2  48   ·  35  49   ·   ·   ·   2   4  49   5  56  52   ·   ·   0
    """),
}


class QCLDPC:
    def __init__(self, rate: str = "r12", norm: float = 0.8, z: int = Z):
        self.rate = rate
        self.norm = norm
        self.z = z
        B = CODES[rate]
        mb, nb = B.shape
        self.mb, self.nb, self.kb = mb, nb, nb - mb
        self.n, self.k, self.m = nb * z, self.kb * z, mb * z
        self._anchor_mid = mb // 2
        # shifts reduce mod z so the short comparison lifts (z < Z) reuse the
        # same grid; at Z the shifts are already in range
        self.entries = [(i, j, int(s) % z)
                        for (i, j), s in np.ndenumerate(B) if s >= 0]
        self._s0 = next(s for r, c, s in self.entries
                        if c == self.kb and r == 0)
        self._build_adjacency()

    def _build_adjacency(self):
        """Padded per-check adjacency: adj[m, w] = variable index, -1 = pad."""
        rows: list[list[int]] = [[] for _ in range(self.m)]
        lanes = np.arange(self.z)
        for bi, bj, s in self.entries:
            v = bj * self.z + (lanes - s) % self.z
            for r, c in zip(bi * self.z + lanes, v):
                rows[r].append(int(c))
        w = max(len(r) for r in rows)
        self.adj = np.full((self.m, w), -1, dtype=np.int64)
        for i, r in enumerate(rows):
            self.adj[i, : len(r)] = r
        self.mask = self.adj >= 0
        self.adj_pad = np.where(self.mask, self.adj, self.n)  # n = scratch col

    # -- encode ---------------------------------------------------------------
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        mbits = np.asarray(info_bits, dtype=np.uint8).reshape(self.kb, self.z)
        u = np.zeros((self.mb, self.z), dtype=np.uint8)
        for bi, bj, s in self.entries:
            if bj < self.kb:
                u[bi] ^= np.roll(mbits[bj], s)
        p = np.zeros((self.mb, self.z), dtype=np.uint8)
        p[0] = np.bitwise_xor.reduce(u, axis=0)          # anchor: shifts cancel
        p[1] = u[0] ^ np.roll(p[0], self._s0)
        for i in range(1, self.mb - 1):
            p[i + 1] = u[i] ^ p[i]
            if i == self._anchor_mid:
                p[i + 1] ^= p[0]
        return np.concatenate([mbits.ravel(), p.ravel()])

    def syndrome_ok(self, bits: np.ndarray) -> np.ndarray:
        """Per-codeword parity check; bits shape (..., n)."""
        b = np.asarray(bits, dtype=np.int64)
        padded = np.concatenate(
            [b, np.zeros(b.shape[:-1] + (1,), dtype=np.int64)], axis=-1)
        chk = padded[..., self.adj_pad].sum(axis=-1) % 2
        return (chk == 0).all(axis=-1)

    # -- decode ---------------------------------------------------------------
    def decode(self, llr: np.ndarray, max_iter: int = 50):
        """Normalized min-sum. ``llr`` shape (B, n) or (n,).

        Returns ``(hard_bits (B, n), ok (B,), iters (B,))``.
        """
        llr = np.atleast_2d(np.asarray(llr, dtype=np.float64))
        B = llr.shape[0]
        adj, mask = self.adj_pad, self.mask
        bidx = np.arange(B)[:, None, None]
        r = np.zeros((B,) + adj.shape)
        hard = np.zeros((B, self.n), dtype=np.int64)
        ok = np.zeros(B, dtype=bool)
        iters = np.full(B, max_iter, dtype=np.int64)

        for it in range(max_iter):
            # posterior per variable; column n is scratch for the pad entries
            post = np.zeros((B, self.n + 1))
            np.add.at(post, (bidx, adj), r)
            post[:, : self.n] += llr

            # extrinsic v->c messages (pad entries are masked out below)
            q = post[bidx, adj] - r
            absq = np.where(mask, np.abs(q), np.inf)
            sgn = np.where(mask & (q < 0), -1.0, 1.0)

            sprod = sgn.prod(axis=-1, keepdims=True)
            am = absq.argmin(axis=-1, keepdims=True)
            min1 = np.take_along_axis(absq, am, axis=-1)
            absq2 = absq.copy()
            np.put_along_axis(absq2, am, np.inf, axis=-1)
            min2 = absq2.min(axis=-1, keepdims=True)

            mag = np.where(np.arange(adj.shape[1]) == am, min2, min1)
            r = self.norm * sprod * sgn * np.where(mask, mag, 0.0)

            cur = (post[:, : self.n] < 0).astype(np.int64)
            newly = self.syndrome_ok(cur) & ~ok
            hard[newly] = cur[newly]
            iters[newly] = it + 1
            ok |= newly
            if ok.all():
                break
        hard[~ok] = cur[~ok]
        return hard, ok, iters
