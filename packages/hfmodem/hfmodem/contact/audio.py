# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Audio framing shared by speech and CW contacts."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from scipy.signal import resample_poly

RATE = 48000
BLOCK = 1536


def resample(audio, source, target):
    divisor = math.gcd(source, target)
    return resample_poly(audio, target // divisor, source // divisor).astype(np.float32)


@dataclass
class Segment:
    audio: np.ndarray
    self_copy: bool
    started: float
    ended: float
    speech_probability: float


class Segmenter:
    """Keep pre-roll, stop on silence; any overlap with our audio is self copy."""
    def __init__(self, threshold=.5, silence_s=.8, preroll_s=.32, max_over_s=25):
        self.threshold = threshold
        self.silence_blocks = max(1, math.ceil(silence_s * RATE / BLOCK))
        self.max_blocks = max(1, math.ceil(max_over_s * RATE / BLOCK))
        self.preroll = deque(maxlen=max(1, math.ceil(preroll_s * RATE / BLOCK)))
        self.active, self.quiet = [], 0
        self.peak_probability = 0.

    def feed(self, audio, probability, self_copy, at):
        item = (np.asarray(audio).copy(), bool(self_copy), at)
        if not self.active:
            if probability < self.threshold:
                self.preroll.append(item)
                return None
            self.active = list(self.preroll)
            self.preroll.clear()
        self.active.append(item)
        self.peak_probability = max(self.peak_probability, probability)
        self.quiet = self.quiet + 1 if probability < self.threshold else 0
        if self.quiet >= self.silence_blocks or len(self.active) >= self.max_blocks:
            return self.finish()
        return None

    def finish(self):
        if not self.active:
            return None
        items, self.active = self.active, []
        segment = Segment(np.concatenate([a for a, _, _ in items]),
                          any(own for _, own, _ in items), items[0][2],
                          items[-1][2] + BLOCK / RATE, self.peak_probability)
        self.quiet, self.peak_probability = 0, 0.
        return segment
