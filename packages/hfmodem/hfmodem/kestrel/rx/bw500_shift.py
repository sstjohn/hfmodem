# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Causal peer-grid correction for BW500's base records."""
import numpy as np
from scipy.signal import firwin, lfilter

from .varahf500 import FS

DELAY = 128
_T = np.arange(2 * DELAY + 1) - DELAY
_ANALYTIC = 2 * firwin(len(_T), 800, fs=FS) * np.exp(2j * np.pi * 1500 * _T / FS)
# The uncorrected L4 read remains clean at +/-10 Hz; VARA HF peers are within
# a few Hz. Keep their recorded waveform and turnaround exactly as received.
MIN_HZ = 10.


def correction(hz):
    return hz if abs(hz) > MIN_HZ else 0.


class PeerShift:
    def __init__(self):
        self.hz = 0.
        self.phase = 0.
        self.state = np.zeros(len(_ANALYTIC) - 1, complex)

    def feed(self, x):
        x = np.asarray(x, float)
        if not len(x):
            return x
        z, self.state = lfilter(_ANALYTIC, [1.], x, zi=self.state)
        step = -2 * np.pi * self.hz / FS
        phase = self.phase + step * np.arange(len(x))
        self.phase = (self.phase + step * len(x)) % (2 * np.pi)
        return (z * np.exp(1j * phase)).real


def retune(x, hz):
    hz = correction(hz)
    if not hz:
        return x
    shift = PeerShift()
    shift.hz = hz
    # A bracket is already whole; remove the causal filter's delay there.
    return shift.feed(np.pad(x, (0, DELAY)))[DELAY:]
