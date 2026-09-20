# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Impulsive HF noise: Poisson-arriving broadband bursts.

Real HF bands are not Gaussian: atmospherics (sferics), ignition and line
noise arrive as sub-millisecond broadband crashes tens of dB above the
thermal floor -- a Middleton Class-A-like mixture the AWGN sims omit. Each
event is a slab of white complex Gaussian noise ``imp_db`` above the
stream's own mean power, with log-uniform duration; arrivals are Poisson.
A 0.1 ms crash at +25 dB dumps ~2x a whole OFDM symbol's energy across
every bin of that FFT, so unmitigated QAM cells under it are garbage with
confident LLRs -- exactly what the pre-FFT blanker exists to remove.
"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.phy.modem import FS


def add_impulse_noise(x: np.ndarray, rng: np.random.Generator,
                      rate_hz: float = 20.0, imp_db: float = 25.0,
                      dur_ms: tuple[float, float] = (0.05, 0.3),
                      fs: float = FS) -> np.ndarray:
    x = np.array(x, dtype=np.complex128)
    p = np.mean(np.abs(x) ** 2)
    sigma = np.sqrt(p * 10 ** (imp_db / 10) / 2)
    lo, hi = dur_ms
    for _ in range(rng.poisson(rate_hz * x.size / fs)):
        d = max(1, int(np.exp(rng.uniform(np.log(lo), np.log(hi)))
                       * 1e-3 * fs))
        s = int(rng.integers(0, max(1, x.size - d)))
        x[s : s + d] += sigma * (rng.standard_normal(d)
                                 + 1j * rng.standard_normal(d))
    return x
