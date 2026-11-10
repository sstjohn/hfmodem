# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Fast in-band control: control blocks on a short coherent OFDM burst.

Session blocks occupy 44 bytes; connectionless blocks occupy 22. One session
block takes 0.711 s and a piggybacked ACK plus DATA header takes 1.191 s.
CA-TBCC rate 1/2, PN9 whitening and a 16-row interleaver precede QPSK mapping
on the workhorse profile. Each block retains its own CRC16, also used for
TBCC candidate screening.

The FSM selects the tier with hysteresis and falls back to the robust floor
on loss. The receiver tries both tiers and the supported block lengths.
"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.fec.tbcc import CATBCC
from hfmodem.sabir.frame.codec import crc_ok, pn9
from hfmodem.sabir.phy.modem import CP, GUARD_HEAD, GUARD_TAIL, N_FFT, WINDOW, Phy
from hfmodem.sabir.phy.preamble import SEG_LEN

from .wire import BLOCK_BYTES


class FastControl:
    """Control blocks <-> a short coherent OFDM burst on the given Phy
    (the workhorse gear: QPSK, 24 carriers, the band every station has)."""

    def __init__(self, phy: Phy, block_bytes=BLOCK_BYTES):
        self.phy = phy
        self.block_bytes = block_bytes
        # One survivor per state bounds work on the frequent control path.
        # The more sensitive floor decoder keeps four; fast controls fall back
        # on loss rather than increasing this decoder's list depth.
        self.tb = CATBCC(state_list=1)

    # -- shape ---------------------------------------------------------------
    def _coded_bits(self, n_blocks: int) -> int:
        return 2 * 8 * self.block_bytes * n_blocks

    def n_syms(self, n_blocks: int) -> int:
        return self.phy.n_symbols_for(self._coded_bits(n_blocks))

    def n_samples(self, n_blocks: int) -> int:
        return (GUARD_HEAD + 2 * SEG_LEN + self.n_syms(n_blocks) * (N_FFT + CP)
                + WINDOW + GUARD_TAIL)

    def _perm(self, n: int) -> np.ndarray:
        return np.arange(n).reshape(16, n // 16).T.ravel()

    # -- transmit ------------------------------------------------------------
    def transmit(self, blocks: bytes) -> np.ndarray:
        coded = self.tb.encode(blocks)
        n = coded.size
        return self.phy.transmit((coded ^ pn9(n))[self._perm(n)])

    # -- receive -------------------------------------------------------------
    def receive(self, samples: np.ndarray, n_blocks: int,
                dd: int = 1) -> tuple[bytes | None, int]:
        """Samples -> (the blocks, or None; the sample the burst started at).

        The caller needs the second half of that whenever what it holds is a
        segmenter's bracket rather than a transmitter's own array: the burst
        sits behind a pre-roll, and whatever follows the header is that much
        further along.
        """
        n = self._coded_bits(n_blocks)
        try:
            _, llr, res = self.phy.receive(samples,
                                           n_symbols=self.n_syms(n_blocks),
                                           dd=dd)
        except ValueError:
            return None, 0
        if llr.size < n:
            return None, 0
        deint = np.empty(n)
        deint[self._perm(n)] = llr[:n]
        deint *= 1 - 2 * pn9(n)                       # de-whiten = sign flip

        def check(buf: bytes) -> bool:
            return all(crc_ok(buf[i : i + self.block_bytes])
                       for i in range(0, len(buf), self.block_bytes))

        block, _ = self.tb.decode(deint, self.block_bytes * n_blocks, check)
        # the detector reports the OFDM grid, biased early by design; the
        # preamble pair and the head guard sit ahead of it
        start = (res.frame_start + self.phy.detector.early_bias
                 - GUARD_HEAD - 2 * SEG_LEN)
        return block, max(0, start)
