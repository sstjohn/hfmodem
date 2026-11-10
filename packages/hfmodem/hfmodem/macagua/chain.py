# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""SSB audio: stateful receive filtering, and bounded transmit conditioning."""
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt
from hfmodem.core.levels import at_drive

from hfmodem.contact.audio import (RATE as RATE, BLOCK as BLOCK, resample as resample,
                                  Segment as Segment, Segmenter as Segmenter)


def compress(audio, threshold_db, ratio):
    """Feed-forward soft-knee compression with a smooth 5 ms gain envelope.

    A draft is buffered, so each block's detector can see all its peaks. Gain
    moves with 5 ms attack and 100 ms release, interpolated at sample rate.
    Final peak normalization provides makeup gain without increasing drive.
    """
    step = round(RATE * .005)
    blocks = (len(audio) + step - 1) // step
    detector = np.max(np.abs(np.pad(audio, (0, blocks * step - len(audio))).reshape(blocks, step)), axis=1)
    above = 20 * np.log10(np.maximum(detector, 1e-12)) - threshold_db
    knee = 6.
    reduction = np.where(above >= knee / 2, (1 - 1 / ratio) * above,
                         np.where(above > -knee / 2,
                                  (1 - 1 / ratio) * (above + knee / 2) ** 2 / (2 * knee), 0.))
    gain_db = -np.maximum(reduction, 0)
    envelope = np.empty(blocks)
    previous = gain_db[0]
    for i, target in enumerate(gain_db):
        alpha = np.exp(-.005 / (.005 if target < previous else .100))
        previous = target + alpha * (previous - target)
        envelope[i] = previous
    gain = np.interp(np.arange(len(audio)), np.arange(blocks) * step, envelope)
    return audio * 10 ** (gain / 20)


def transmit(audio, rate, drive, *, compression='soft'):
    if compression not in ('off', 'soft', 'medium', 'strong'):
        raise ValueError('compression must be off, soft, medium or strong')
    audio = resample(np.asarray(audio, dtype=np.float32), rate, RATE)
    if audio.size < 64 or not np.isfinite(audio).all():
        raise ValueError('synthesizer produced empty or nonfinite speech')
    filters = butter(6, [300, 2700], 'bandpass', fs=RATE, output='sos')
    audio = sosfiltfilt(filters, audio)
    peak = np.max(np.abs(audio))
    if peak <= 1e-8:
        raise ValueError('synthesizer produced silence')
    if compression == 'soft':
        audio = np.tanh(audio / peak * 1.7)
    elif compression != 'off':
        threshold, ratio = (-18., 3.) if compression == 'medium' else (-24., 4.)
        audio = compress(audio / peak, threshold, ratio)
        # Remove modulation products before the final peak ceiling.
        audio = sosfiltfilt(filters, audio)
    fade = min(480, len(audio) // 2)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return at_drive(audio, drive).astype(np.float32)


class ReceiveChain:
    def __init__(self):
        self.sos = butter(5, [250, 2800], 'bandpass', fs=RATE, output='sos')
        self.state = np.zeros((len(self.sos), 2))

    def block(self, audio):
        filtered, self.state = sosfilt(self.sos, audio, zi=self.state)
        return resample(filtered, RATE, 16000)


