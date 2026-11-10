# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Robust Packet pulses, tones, matched filters and card-rate decimation."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve, firwin, hilbert, lfilter, upfirdn

from hfmodem.core.rates import CARD_RATE_HZ

from .frame import PREAMBLE

RATE = 9600
SPS = 192
CENTRE = 1500.0
TONES = np.arange(8) * 60 - 210
RAMP_SYMBOLS = 7
ROTATION = np.exp(0.25j * np.pi)
PREAMBLE_SIGNS = 1 - 2 * np.unpackbits(np.frombuffer(PREAMBLE, np.uint8),
                                      bitorder="little").astype(float).reshape(8, 8)


def rrc(times: np.ndarray, alpha: float = 0.33) -> np.ndarray:
    t = np.asarray(times, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (np.sin(np.pi * (1 - alpha) * t) +
                 4 * alpha * t * np.cos(np.pi * (1 + alpha) * t)) / (
                     np.pi * t * (1 - 16 * alpha ** 2 * t ** 2))
    value[t == 0] = 1 - alpha + 4 * alpha / np.pi
    edge = np.isclose(np.abs(t), 1 / (4 * alpha))
    value[edge] = alpha / np.sqrt(2) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
        (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
    return value


@lru_cache(maxsize=1)
def tx_pulse() -> np.ndarray:
    h = rrc((np.arange(119) - 59) / 24)
    return h * 24 / h.sum()


@lru_cache(maxsize=1)
def rx_pulse() -> np.ndarray:
    h = rrc(np.arange(-480, 481) / SPS)
    return h / h.sum()


def interpolate(samples: np.ndarray, factor: int, taps: np.ndarray) -> np.ndarray:
    out = upfirdn(taps, samples, up=factor)
    return out[:len(samples) * factor]


def crest_limit(samples: np.ndarray) -> np.ndarray:
    radius = 3.59 * np.sqrt(np.sum(tx_pulse() ** 2) / 24) / 4
    amplitude = np.abs(samples)
    excess = samples * np.maximum(0, 1 - radius / np.maximum(amplitude, 1e-30))
    kernel = firwin(31, 230, fs=1200, window=("kaiser", 4.5)) * 0.986
    delayed = np.zeros_like(samples)
    delayed[15:] = samples[:-15]
    return delayed - 2.4 * lfilter(kernel, [1], excess)


def composite(steps: np.ndarray, level: int, limiter: bool = True) -> np.ndarray:
    if level not in (0, 1):
        raise ValueError("level must be 0 or 1")
    phasors = np.exp(np.cumsum(steps, axis=0) * (1j * np.pi / 8))
    phasors = np.pad(phasors, ((0, RAMP_SYMBOLS), (0, 0)))
    t = np.arange(len(phasors) * 24) / 1200
    base = np.zeros(t.size, complex)
    for c, frequency in enumerate(TONES):
        base += interpolate(phasors[:, c], 24, tx_pulse()) * np.exp(2j * np.pi * frequency * t)
    base *= (1.0, 0.8)[level] / 4
    return crest_limit(base) if limiter else base


def audio(base: np.ndarray) -> np.ndarray:
    native = interpolate(base, 8, firwin(128, 600, fs=RATE) * 8)
    return 2 * (native * np.exp(2j * np.pi * CENTRE * np.arange(len(native)) / RATE)).real


def matched(samples: np.ndarray, centre: float, inverted: bool = False) -> np.ndarray:
    analytic = hilbert(samples)
    time = np.arange(len(samples)) / RATE
    base = analytic * np.exp(-2j * np.pi * centre * time)
    if inverted:
        base = base.conj()
    return matched_base(base)


def matched_base(base: np.ndarray) -> np.ndarray:
    time = np.arange(len(base)) / RATE
    return np.array([fftconvolve(base * np.exp(-2j * np.pi * f * time), rx_pulse(), "same")
                     for f in TONES])


def differentials(filtered: np.ndarray) -> np.ndarray:
    d = filtered[:, SPS:] * filtered[:, :-SPS].conj()
    return d / np.maximum(np.abs(d), 1e-30)


def preamble_correlation(filtered: np.ndarray) -> np.ndarray:
    d = differentials(filtered)
    length = d.shape[1] - 7 * SPS
    if length <= 0:
        return np.empty(0, complex)
    score = np.zeros(length, complex)
    for i, signs in enumerate(PREAMBLE_SIGNS):
        score += signs @ d[:, i * SPS:i * SPS + length]
    return score * ROTATION / 64


def band_centre(samples: np.ndarray) -> float:
    if len(samples) < SPS:
        raise ValueError("too little audio to locate the band")
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    width = max(1, round(480 * len(samples) / RATE))
    summed = np.concatenate(([0.0], np.cumsum(power)))
    index = int(np.argmax(summed[width:] - summed[:-width]))
    return (index + width / 2) * RATE / len(samples)


class Decimator:
    """Causal 48 kHz float input; output n is labelled with card index 5n + D."""
    TAPS = firwin(101, 4000, fs=CARD_RATE_HZ)
    DELAY = (len(TAPS) - 1) // 2

    def __init__(self, start: int = 0):
        self.reset(start)

    def reset(self, start: int) -> None:
        self.next_card = start
        self._history = np.zeros(len(self.TAPS) - 1)

    def feed(self, start: int, samples: np.ndarray) -> tuple[int, np.ndarray]:
        if start != self.next_card:
            raise ValueError("discontinuous card stream; reset required")
        samples = np.asarray(samples, float)
        if samples.ndim != 1:
            raise ValueError("mono audio required")
        first = (self.DELAY - start) % 5
        count = max(0, (len(samples) - first + 4) // 5)
        combined = np.concatenate((self._history, samples))
        offset = len(self._history) + first
        phase = -offset % 5
        filtered = upfirdn(self.TAPS, np.pad(combined, (phase, 0)), down=5)
        begin = (offset + phase) // 5
        self._history = combined[-len(self._history):].copy()
        self.next_card += len(samples)
        return (start + first - self.DELAY) // 5, filtered[begin:begin + count]
