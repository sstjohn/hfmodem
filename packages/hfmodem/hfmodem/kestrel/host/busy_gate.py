# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA's BUSY at BW500: a signal in the BW500 passband, and nothing else.

Observed 2026-10-23: a VARA HF station shows BUSY
for a CQ or a beacon keyed at the centre or 400 Hz off it, and for a 1500 Hz carrier. It
stays clear for the same frames in the VarAC slots 750 Hz either side, for a 2250 Hz
carrier, and for broadband noise at +3, +6 and +10 dB. Over the 10-23 22:25Z 40 m session
it was BUSY 21.7 % of 480 s, where our broadband gate said 39 %.

So the level judged is the BW500 passband against the quietest tenth of the bins in the
flanks either side. Band noise lifts both and changes nothing; slot traffic fills at most
500 Hz of each flank and leaves the quiet bins the reference is taken from. The ratio is
compared with a quarter-quantile of itself over the last 30 s. On that session the gate
agrees with VARA's BUSY 93 % of the time (the broadband gate 69 %); what it does not copy is
that VARA also stays clear for 500 Hz of noise in its passband, which this reads as a signal.

Band noise lifts both only where the receiver passes it. The FT-891 into a dummy load on
2026-11-06 measured -3 dB at 2161-2995 Hz and -10 dB at 1555: the lower flank was the
codec's own floor, the band was the filter's skirt, and every wander in the receiver's
noise moved the band and not its reference -- BUSY 13-22 dB "over flanks" on nothing,
on and off for minutes. So each bin is first divided by its own noise, a low quantile
of its last 30 s, and only bins within PASS_DB of the passband's top are judged at all:
what is left is the same flat noise whatever the filter's shape, and a signal stands
over it by its own SNR. A band the receiver does not pass is not judged.
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
#: A bin's noise is this quantile of its averaged power over FLOOR_S: a signal has to
#: fill the band nine tenths of the time before it is taken for noise.
NOISE_QUANTILE = .1
#: Bins whose noise sits further than this under the loudest tenth are the filter's skirt or
#: the codec's floor, where the receiver's noise is not what is heard.
PASS_DB = 10.0
#: Fewer bins than this left in the band or the flanks and nothing is judged.
MIN_BINS = 6


class BandGate:
    def __init__(self):
        f = np.fft.rfftfreq(FRAME, 1 / FS)
        band = (f >= BAND[0]) & (f <= BAND[1])
        flank = np.logical_or.reduce([(f >= a) & (f <= b) for a, b in FLANKS])
        self._bins = band | flank
        self._band, self._flank = band[self._bins], flank[self._bins]
        self._window = np.hanning(FRAME)
        self._history = deque(maxlen=int(FLOOR_S * FS / FRAME))
        self._spectra = deque(maxlen=int(FLOOR_S * FS / FRAME / AVERAGE))
        self._noise = None
        self._frames = 0
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
            self._recent.append(np.abs(np.fft.rfft(frame * self._window))[self._bins] ** 2)
            power = np.mean(self._recent, axis=0)
            if not power.any():
                # Digital silence is no receiver at all, not a quiet band; it would
                # set a noise of zero and a floor of zero under every frame after it.
                self.active = False
                continue
            self._frames += 1
            if self._frames % AVERAGE == 0:
                self._spectra.append(power)
                self._noise = np.maximum(np.quantile(self._spectra, NOISE_QUANTILE, axis=0), 1e-30)
            self.ratio = self._judge(power)
            if self.ratio is None:
                self.active = False
                continue
            self._history.append(self.ratio)
            self.floor = float(np.quantile(self._history, .25))
            self.active = len(self._history) >= 2 * AVERAGE and self.ratio > self.floor * 10 ** (ENTER_DB / 10)
        return self.active

    def _judge(self, power):
        if self._noise is None:
            return None
        heard = self._noise >= max(np.quantile(self._noise, .9), 1e-20) * 10 ** (-PASS_DB / 10)
        band, flank = self._band & heard, self._flank & heard
        if band.sum() < MIN_BINS or flank.sum() < MIN_BINS:
            return None
        white = power / self._noise
        return white[band].mean() / max(np.quantile(white[flank], .1), 1e-30)

    def detail(self):
        if self.ratio is None:
            return 'band gate unprimed' if self._noise is None else 'band not heard through the receiver'
        db = lambda v: 10 * np.log10(max(v, 1e-30))
        return f'band {db(self.ratio):.1f} dB over flanks, floor {db(self.floor):.1f} dB'
