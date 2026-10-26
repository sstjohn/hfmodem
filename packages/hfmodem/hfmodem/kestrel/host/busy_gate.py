# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock's BUSY at BW500: a signal in the BW500 passband, and nothing else.

Measured on the 2026-10-23 bench: stock B goes BUSY
for a CQ or a beacon keyed at the centre or 400 Hz off it, and for a 1500 Hz carrier. It
stays clear for the same frames in the VarAC slots 750 Hz either side, for a 2250 Hz
carrier, and for broadband noise stepped +3, +6 and +10 dB. The 10-23 22:25Z 40 m session
played into it was BUSY 21.7 % of 480 s, where our broadband gate said 39 %.

So the level judged is the BW500 passband against the quietest tenth of the bins in the
flanks either side. Band noise lifts both and changes nothing; slot traffic fills at most
500 Hz of each flank and leaves the quiet bins the reference is taken from. The ratio is
compared with a quarter-quantile of itself over the last 30 s, which describes the
receiver's own passband shape. On that session the gate agrees with stock 93 % of the
time (the broadband gate 69 %); what it does not copy is that stock also stays clear for
500 Hz of noise in its passband, which this reads as a signal.
"""
from collections import deque

import numpy as np

from ...core.occupied import occupied_hz

FS = 48000
FRAME = 2048
BAND = occupied_hz('vara', '500')
FLANKS = ((300.0, 950.0), (2000.0, 2700.0))
ENTER_DB = 3.0
AVERAGE = 4
FLOOR_S = 30.0


class BandGate:
    def __init__(self):
        f = np.fft.rfftfreq(FRAME, 1 / FS)
        self._band = (f >= BAND[0]) & (f <= BAND[1])
        self._flank = np.logical_or.reduce([(f >= a) & (f <= b) for a, b in FLANKS])
        self._window = np.hanning(FRAME)
        self._history = deque(maxlen=int(FLOOR_S * FS / FRAME))
        self.restart()
        self.ratio = self.floor = None

    def restart(self):
        """Forget the audio in flight (our own transmission), keep the floor."""
        self._recent = deque(maxlen=AVERAGE)
        self._pending = np.zeros(0)
        self.active = False

    def push(self, samples):
        x = np.concatenate((self._pending, np.asarray(samples, float)))
        n = len(x) // FRAME * FRAME
        self._pending = x[n:]
        for frame in x[:n].reshape(-1, FRAME):
            self._recent.append(np.abs(np.fft.rfft(frame * self._window)) ** 2)
            power = np.mean(self._recent, axis=0)
            self.ratio = power[self._band].mean() / max(np.quantile(power[self._flank], .1), 1e-30)
            self._history.append(self.ratio)
            self.floor = float(np.quantile(self._history, .25))
            self.active = len(self._history) >= 2 * AVERAGE and self.ratio > self.floor * 10 ** (ENTER_DB / 10)
        return self.active

    def detail(self):
        if self.ratio is None:
            return 'band gate unprimed'
        return f'band {10 * np.log10(self.ratio):.1f} dB over flanks, floor {10 * np.log10(self.floor):.1f} dB'
