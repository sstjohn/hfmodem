# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Acquisition result and sample-wise frequency correction."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    frame_start: int                 # first OFDM symbol, including its CP
    coarse_cfo_hz: float = 0.0
    metric: float = 0.0
    preamble_len: int = 0


def remove_cfo(x: np.ndarray, cfo_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Derotate an analytic sample stream by -cfo_hz."""
    x = np.asarray(x, dtype=np.complex128)
    if cfo_hz == 0:
        return x
    return x * np.exp(-2j * np.pi * cfo_hz * np.arange(x.size) / sample_rate_hz)
