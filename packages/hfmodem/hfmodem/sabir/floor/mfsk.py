# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Noncoherent 4-FSK for narrow controls and strongly varying channels.

Energy detection avoids the coherent OFDM receiver's pilot-tracking
requirement. This is the protocol's robust control and narrow DATA waveform.

Numerology: 1024-sample symbols at 48 kHz (21.33 ms, 46.875 baud) on the
46.875 Hz grid. Data rides 4 tones spaced 93.75 Hz, Gray
mapped, 2 bits/symbol. Sync is a 7x7 Costas array (Costas 1984, prior art)
on the 46.875 Hz single-unit grid -- same band as the data tones -- one block
every <=56 data symbols, supporting burst detection, timing and per-block CFO
tracking. The seven grid positions span 281.25 Hz between outer tones;
spectral skirts and receiver filtering must also be considered.

Coding: CA-TBCC (``fec.tbcc``) over CRC-terminated byte blocks, PN9-whitened
and block-interleaved; a gear's ``repeat`` R > 1 tiles the coded stream R
times and the receiver sums LLRs. Time separation provides diversity when the
channel varies between copies. The active waveform has constant envelope and
continuous phase, with symbol-edge frequency ramps for spectral containment
and amplitude ramps at burst boundaries.

The floor carries session and connectionless controls from ``arq.wire``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from hfmodem.sabir.fec.tbcc import CATBCC
from hfmodem.sabir.frame import pn9
from hfmodem.sabir.frame.codec import crc_ok
from hfmodem.sabir.phy.modem import FS, GUARD_HEAD, GUARD_TAIL
from hfmodem.sabir.phy.rate import CENTER_HZ

SYM = 1024
GRID = FS / SYM                     # 46.875 Hz
BASE_HZ = CENTER_HZ - 3 * GRID                 # tone 0 at 1359.375 Hz; COSTAS tops out
                                    # at unit 6, so the band is 1359-1641
COSTAS = (3, 1, 4, 0, 6, 5, 2)      # 7x7 Costas array, single-unit grid
DATA_UNITS = (0, 2, 4, 6)           # 4 tones, 93.75 Hz apart
NOISE_UNITS = (1, 3, 5)             # between-tone bins: the noise reference
GRAY = np.array([0, 1, 3, 2])       # 2-bit value -> tone index
SEG = 56                            # max data symbols between Costas blocks

HOP = 256                           # sync spectrogram hop
FINE = 4                            # fine bins per grid unit (11.72 Hz)
DMAX = 8                            # coarse CFO search: +/-8 fine = 93.75 Hz

# The seven unit tones of the Costas alphabet, and the symbol time base. Both
# are fixed for the life of the process; `_costas_metric` multiplies a tone by
# one f_off phase instead of building each reference from scratch.
_T = np.arange(SYM) / FS
_TONES = np.exp(-2j * np.pi * np.outer(BASE_HZ + np.arange(7) * GRID, _T))
EDGE = 128                          # burst amplitude ramp, samples


@dataclass(frozen=True)
class FloorGear:
    name: str
    repeat: int


FLOOR_GEARS = {
    "floor": FloorGear("mfsk4-tbcc-x1", 1),
    "floor2": FloorGear("mfsk4-tbcc-x2", 2),
    "floor4": FloorGear("mfsk4-tbcc-x4", 4),
}


def _plan(n_data: int, seg: int = SEG):
    """Burst symbol plan: Costas blocks bracketing <=seg-symbol data runs.

    Returns (cos_pos, cos_unit, data_pos) as symbol-index arrays.
    """
    n_seg = ceil(n_data / seg)
    base, extra = divmod(n_data, n_seg)
    sizes = [base + (i < extra) for i in range(n_seg)]
    cos_pos, data_pos = [], []
    at = 0
    for size in sizes:
        cos_pos.append(at)
        at += 7
        data_pos.extend(range(at, at + size))
        at += size
    cos_pos.append(at)
    cp = np.repeat(np.array(cos_pos), 7) + np.tile(np.arange(7), len(cos_pos))
    cu = np.tile(np.array(COSTAS), len(cos_pos))
    return cp, cu, np.array(data_pos)


class FloorModem:
    def __init__(self, gear: FloorGear = FLOOR_GEARS["floor"],
                 threshold: float = 1.5):
        self.gear = gear
        self.tb = CATBCC()
        self.threshold = threshold

    # -- shape ------------------------------------------------------------
    def n_symbols(self, n_bytes: int) -> int:
        n_data = 8 * n_bytes * self.gear.repeat
        cp, _, dp = _plan(n_data)
        return int(max(cp[-1], dp[-1])) + 1

    def duration_s(self, n_bytes: int) -> float:
        return (GUARD_HEAD + self.n_symbols(n_bytes) * SYM + GUARD_TAIL) / FS

    def _perm(self, n: int) -> np.ndarray:
        return np.arange(n).reshape(16, n // 16).T.ravel()

    # -- transmit ----------------------------------------------------------
    def transmit(self, block: bytes) -> np.ndarray:
        """CRC-terminated block -> complex analytic burst, unit envelope."""
        coded = self.tb.encode(block)
        return self.transmit_coded(coded)

    def transmit_coded(self, coded: np.ndarray) -> np.ndarray:
        """Tail-biting coded bits, before whitening, for selective HARQ."""
        coded = np.asarray(coded, dtype=np.int64)
        if coded.ndim != 1 or not coded.size or coded.size % 16:
            raise ValueError("floor codeword must contain a whole number of bytes")
        n = coded.size
        coded = (coded ^ pn9(n))[self._perm(n)]
        stream = np.tile(coded, self.gear.repeat)
        tones = GRAY[2 * stream[0::2] + stream[1::2]]
        cp, cu, dp = _plan(tones.size)
        units = np.empty(self.n_symbols(n // 16), dtype=np.int64)
        units[cp] = cu
        units[dp] = 2 * tones
        f = np.repeat(BASE_HZ + units * GRID, SYM)
        # ramp the frequency steps over ~64 samples: phase stays continuous
        # and the discontinuous-frequency sidelobes drop well below the
        # 500 Hz occupied-bandwidth line
        f = np.convolve(f, np.ones(64) / 64, mode="same")
        wave = np.exp(2j * np.pi * np.cumsum(f) / FS)
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(EDGE) + 0.5) / EDGE))
        wave[:EDGE] *= ramp
        wave[-EDGE:] *= ramp[::-1]
        return np.concatenate([np.zeros(GUARD_HEAD, dtype=np.complex128),
                               wave,
                               np.zeros(GUARD_TAIL, dtype=np.complex128)])

    # -- receive -----------------------------------------------------------
    def receive(self, samples: np.ndarray, n_bytes: int,
                check=crc_ok) -> tuple[bytes | None, dict]:
        """Raw analytic samples -> (block bytes or None, stats)."""
        llr, stats = self.demod(samples, n_bytes)
        if llr is None:
            return None, stats
        block, dec = self.tb.decode(llr, n_bytes, check)
        return block, stats | dec

    def demod(self, samples: np.ndarray, n_bytes: int):
        """Unwhitened coded LLRs, suitable for Chase combining across overs."""
        x = np.asarray(samples, dtype=np.complex128)
        n_data = 8 * n_bytes * self.gear.repeat
        cp, cu, dp = _plan(n_data)
        n_sym = self.n_symbols(n_bytes)
        det = self._sync(x, cp, cu, n_sym)
        if det is None:
            return None, {"sync": None}
        s0, cfo_hz, drift, metric = det
        if not (0 <= s0 < x.size):        # a false lock on noise can land out of range
            return None, {"sync": None}
        t = np.arange(x.size - s0) / FS
        y = x[s0:] * np.exp(-2j * np.pi * (cfo_hz * t + 0.5 * drift * t**2))
        if y.size < n_sym * SYM:
            return None, {"sync": None}
        E = self._energies(y[: n_sym * SYM].reshape(n_sym, SYM))
        Ed = E[dp]
        n0 = max(np.median(Ed[:, list(NOISE_UNITS)]) / np.log(2), 1e-12)
        e = Ed[:, list(DATA_UNITS)][:, GRAY]
        # e[:, v] = energy of the tone carrying the 2-bit value v. Copies of
        # a repeat gear carry identical tones, so square-law combine energies
        # *before* the LLR nonlinearity (classical noncoherent diversity).
        R = self.gear.repeat
        e = e.reshape(R, -1, 4).sum(axis=0)
        # exact tone log-likelihood is linear in energy with slope
        # (gamma/(1+gamma))/n0; per-bit LLR by log-sum-exp over the pair
        gamma = max(np.mean(e.sum(axis=1)) / (R * n0) - 4.0, 0.05)
        s = gamma / (1.0 + gamma) / n0 * e
        llr = np.empty((e.shape[0], 2))
        llr[:, 0] = np.logaddexp(s[:, 0], s[:, 1]) - np.logaddexp(s[:, 2], s[:, 3])
        llr[:, 1] = np.logaddexp(s[:, 0], s[:, 2]) - np.logaddexp(s[:, 1], s[:, 3])
        n = 16 * n_bytes
        deint = np.empty(n)
        deint[self._perm(n)] = llr.ravel()
        deint *= 1 - 2 * pn9(n)                       # de-whiten = sign flip
        return deint, {"cfo_hz": cfo_hz, "drift_hz_s": drift, "start": s0,
                       "metric": metric, "noise": n0}

    def _energies(self, syms: np.ndarray) -> np.ndarray:
        """Per-symbol tone energies on the unit grid: (n_sym, 7)."""
        t = np.arange(SYM) / FS
        C = np.exp(-2j * np.pi * np.outer(BASE_HZ + np.arange(7) * GRID, t))
        return np.abs(syms @ C.T / SYM) ** 2

    def _costas_metric(self, x, s0, cp, cu, f_off, blocks=None):
        """Noncoherent Costas correlation at sample timing s0, CFO f_off.

        The reference factors: exp(-2pi j (BASE + u*GRID + f_off) t) is the
        unit-`u` tone times a phase that depends only on f_off. The tone half
        takes one of seven values and is precomputed once in ``_TONES``; the
        f_off half is one exponential per call rather than one per Costas
        block. What is left is a gather and a row-wise dot, so the per-block
        Python loop goes away -- it ran ~630 times per acquisition.
        """
        idx = np.arange(cp.size) if blocks is None else np.fromiter(
            blocks, dtype=np.intp)
        lo = s0 + cp[idx] * SYM
        keep = (lo >= 0) & (lo + SYM <= x.size)
        if not keep.any():
            return 0.0
        lo, u = lo[keep], cu[idx][keep]
        ref = _TONES[u] * np.exp(-2j * np.pi * f_off * _T)      # (n, SYM)
        seg = x[lo[:, None] + np.arange(SYM)]                   # (n, SYM)
        corr = np.einsum("ij,ij->i", seg, ref)
        return float(np.sum(np.abs(corr) ** 2)) / SYM**2

    def _sync(self, x, cp, cu, n_sym):
        """Coarse spectrogram search + fine timing + per-block CFO/drift fit.

        Returns (start_sample, cfo_hz, drift_hz_s, metric) or None.
        """
        if x.size < n_sym * SYM:
            return None
        n_hops = (x.size - SYM) // HOP + 1
        view = np.lib.stride_tricks.as_strided(
            x, (n_hops, SYM), (x.itemsize * HOP, x.itemsize))
        q = np.arange(-DMAX, 6 * FINE + DMAX + 1)     # fine bins across band
        t = np.arange(SYM) / FS
        W = np.exp(-2j * np.pi * np.outer(BASE_HZ + q * GRID / FINE, t))
        E = np.abs(W @ view.T) ** 2 / SYM**2          # (n_q, n_hops)

        t_max = n_hops - (n_sym - 1) * (SYM // HOP)
        if t_max <= 0:
            return None
        M = np.zeros((2 * DMAX + 1, t_max))
        for i in range(cp.size):
            rows = DMAX + FINE * cu[i] + np.arange(-DMAX, DMAX + 1)
            off = cp[i] * (SYM // HOP)
            M += E[rows, off : off + t_max]
        peak = np.unravel_index(np.argmax(M), M.shape)
        if not np.isfinite(M[peak]) or M[peak] <= 0:
            return None
        metric = float(M[peak] / max(float(np.median(M)), 1e-300))
        if metric < self.threshold:
            return None
        d_hz = (peak[0] - DMAX) * GRID / FINE
        s0 = peak[1] * HOP

        # fine timing: direct correlation on a 32-sample comb around s0
        offs = np.arange(-4, 5) * 32
        mt = [self._costas_metric(x, s0 + o, cp, cu, d_hz) for o in offs]
        s0 += offs[int(np.argmax(mt))]

        # per-Costas-block CFO -> weighted linear drift fit
        n_blocks = cp.size // 7
        times, freqs, wts = [], [], []
        dgrid = np.arange(-4, 4.5) * GRID / 8         # +/-23 Hz, 5.86 Hz step
        for bb in range(n_blocks):
            blocks = range(7 * bb, 7 * bb + 7)
            mb = np.array([self._costas_metric(x, s0, cp, cu, d_hz + dd,
                                               blocks) for dd in dgrid])
            p = int(np.argmax(mb))
            df = dgrid[p]
            if 0 < p < mb.size - 1:
                den = mb[p - 1] - 2 * mb[p] + mb[p + 1]
                if den < 0:
                    df += 0.5 * (mb[p - 1] - mb[p + 1]) / den * (GRID / 8)
            times.append((cp[7 * bb] + 3.5) * SYM / FS)
            freqs.append(d_hz + df)
            wts.append(mb[p])
        tt, ff, ww = map(np.asarray, (times, freqs, wts))
        A = np.stack([np.ones_like(tt), tt], axis=1) * np.sqrt(ww)[:, None]
        c0, c1 = np.linalg.lstsq(A, ff * np.sqrt(ww), rcond=None)[0]
        c1 = float(np.clip(c1, -6.0, 6.0))
        return s0, float(c0), c1, metric
