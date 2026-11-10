# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One track of a window, cut out and moved to where a modem listens.

Every decoder here expects its signal at a nominal centre and keeps expecting
it, so a per-track instance hears its track through this: a complex band-pass
around the track's centre, shifted to the nominal centre, real part taken. The
filter's delay is taken back out, so sample ``i`` out is sample ``i`` in and a
runner's own sample count stays the stream's clock.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, firwin

from hfmodem.monitor import FS

#: Where every monitor runner moves its track to: VARA's and ARDOP's centre, the
#: PACTOR-1 pair's and PACTOR-2 carriers' midpoint, and the PACTOR-4 chirp's
#: packet-long mean (its carriers average 1040 Hz at the header, 1957 at the end).
NOMINAL_HZ = 1500.0
TAPS = 1025
#: Kaiser beta 8 over TAPS: about 80 dB of stopband and a 235 Hz transition,
#: which GUARD_HZ keeps outside what was asked for.
BETA = 8.0
GUARD_HZ = 250.0
#: How close the passband may come to 0 Hz on either side of the shift; nearer,
#: its mirror folds back in when the real part is taken.
EDGE_HZ = 60.0


def tuned(centre_hz: float | None, bw_hz: float) -> "Channel | None":
    """A runner's ``--centre`` and ``--bw``: the channel that moves the track
    home, or None for a runner hearing the whole band unshifted."""
    return None if centre_hz is None else Channel(centre_hz, NOMINAL_HZ, bw_hz / 2)


class Channel:
    """Push audio at ``fs``; get the band ``centre_hz`` +/- ``half_hz`` back,
    moved to ``to_hz``, delay-compensated."""

    def __init__(self, centre_hz: float, to_hz: float, half_hz: float,
                 fs: int = FS) -> None:
        self.centre_hz, self.to_hz, self.fs = centre_hz, to_hz, fs
        self.half_hz = max(0.0, min(half_hz, centre_hz - EDGE_HZ, to_hz - EDGE_HZ,
                                    fs / 2 - centre_hz - EDGE_HZ,
                                    fs / 2 - to_hz - EDGE_HZ))
        cut = min(self.half_hz + GUARD_HZ / 2, fs / 2 - 1)
        n = np.arange(TAPS) - (TAPS - 1) // 2
        self._h = firwin(TAPS, cut, window=("kaiser", BETA), fs=fs) \
            * np.exp(2j * np.pi * centre_hz * n / fs)
        self._tail = np.zeros(TAPS - 1, complex)
        self._delay = self._skip = (TAPS - 1) // 2
        self._n = 0
        self._step = 2j * np.pi * (to_hz - centre_hz) / fs

    def offset(self, hz: float | None) -> float | None:
        """A decoder's offset from ``to_hz``, as an offset from where the
        track actually sits."""
        return None if hz is None else round(hz + self.centre_hz - self.to_hz, 1)

    def push(self, x: np.ndarray) -> np.ndarray:
        if not len(x):
            return np.zeros(0)
        y = fftconvolve(np.asarray(x, float), self._h)
        y[:TAPS - 1] += self._tail
        self._tail = y[len(x):].copy()
        return self._out(y[:len(x)])

    def flush(self) -> np.ndarray:
        """The last ``TAPS // 2`` samples, which the delay was holding."""
        y, self._tail = self._tail[:self._delay], np.zeros(TAPS - 1, complex)
        return self._out(y)

    def _out(self, y: np.ndarray) -> np.ndarray:
        k = self._n + np.arange(len(y))
        self._n += len(y)
        y = 2.0 * (y * np.exp(self._step * k)).real
        cut = min(self._skip, len(y))
        self._skip -= cut
        return y[cut:]
