# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""``bateleur.sim``'s channels for the ALE modem's 8 kHz audio.

The SNR is set on the transmission alone, in 3 kHz, as table A-II states it;
Half a second of matching noise precedes the signal, and one second follows.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from .. import sim
from ..spec import FS as SIM_FS
from .fsk import FS


def air(audio: np.ndarray, *, snr_db: float, preset: str = "awgn", freq_hz: float = 0.0,
        seed: int = 0) -> np.ndarray:
    up = resample_poly(np.asarray(audio, dtype=np.float64), SIM_FS // 1600, FS // 1600)
    y = sim.channel(up, snr_db=snr_db, preset=preset, freq_hz=freq_hz, seed=seed)
    # The same noise density as the signal part: measure it from the channel.
    p_sig = np.mean(up ** 2)
    n0 = p_sig / 10 ** (snr_db / 10) / 3000.0
    rng = np.random.default_rng(seed + 7919)
    pad = rng.normal(0, np.sqrt(n0 * SIM_FS / 2), int(1.5 * SIM_FS))
    k = int(0.5 * SIM_FS)
    y = np.concatenate([pad[:k], y, pad[k:]])
    return resample_poly(y, FS // 1600, SIM_FS // 1600)
