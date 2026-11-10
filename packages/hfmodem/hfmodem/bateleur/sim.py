# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A channel to test against: Watterson-style fading paths, offset, noise.

Each path is the analytic signal delayed and multiplied by its own complex
Gaussian tap process, whose spectrum is Gaussian with the given two-sided
Doppler spread (ITU-R F.1487's model), or by a constant where the spread is
zero; the sum takes the real part, then a carrier offset, a sample-clock
error and white noise scaled to an SNR in a 3 kHz bandwidth (``inf``: none).
The CCIR/ITU presets are the ones the standards' own performance tables are
quoted against; ``paths`` gives any other: the swept Doppler and multipath
spreads of STANAG 4285 annex B's tests 6 and 7 (``two_paths``), or a
Ricean channel (a fixed path and a fading one, each at its own power).

Kept here because protocols do not import each other's channel implementations.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert

from .spec import FS

PRESETS = {
    "awgn": (),
    "good": ((0.0, 0.1), (0.5e-3, 0.1)),        # (delay s, Doppler spread Hz)
    "moderate": ((0.0, 0.5), (1.0e-3, 0.5)),
    "poor": ((0.0, 1.0), (2.0e-3, 1.0)),
    # MIL-STD-188-110B table XX's other channels, named by what they are.
    "2ms-5hz": ((0.0, 5.0), (2.0e-3, 5.0)),
    "5ms-1hz": ((0.0, 1.0), (5.0e-3, 1.0)),
    "5ms-5hz": ((0.0, 5.0), (5.0e-3, 5.0)),
    "2ms-0.5hz": ((0.0, 0.5), (2.0e-3, 0.5)),
    # MIL-STD-188-141B table A-II's, for ALE: CCIR good and poor, modified
    # (A.4.2.3) so that a fixed delay does not null the same tones throughout.
    "ale-good": ((0.0, 0.1), (0.52e-3, 0.1)),
    "ale-poor": ((0.0, 1.0), (2.2e-3, 1.0)),
    # One Rayleigh path, 1 Hz (STANAG 4285 annex B test 3, "flat fading").
    "flat": ((0.0, 1.0),),
}


def two_paths(delay_ms: float, spread_hz: float) -> tuple:
    """Two independent equal-power Rayleigh paths ``delay_ms`` apart, each
    spread ``spread_hz``: the channel every standard's table sweeps."""
    return ((0.0, spread_hz), (delay_ms * 1e-3, spread_hz))


def _fading(n: int, spread_hz: float, rng) -> np.ndarray:
    """Unit-power complex Gaussian process with a Gaussian Doppler spectrum."""
    if spread_hz <= 0:
        return np.ones(n, dtype=np.complex128)
    # Resolve even a slow spread independently of burst length, and keep the
    # FFT period longer than twice the excerpt to avoid a repeating channel.
    low_fs = max(100.0, 20 * spread_hz)
    need = int(np.ceil(n / FS * low_fs)) + 2
    m = next_fast_len(max(2 * need, int(np.ceil(20 * low_fs / spread_hz))))
    w = rng.normal(size=m) + 1j * rng.normal(size=m)
    sigma = spread_hz / 2                    # the spread is 2 sigma
    f = np.fft.fftfreq(m, 1 / low_fs)
    shape = np.exp(-f ** 2 / (2 * sigma ** 2))
    g = np.fft.ifft(np.fft.fft(w) * np.sqrt(shape))
    # E|w|^2 = 2; ensemble normalization preserves deep fades in short bursts.
    g *= np.sqrt(m / (2 * shape.sum()))
    t_low = np.arange(m) / low_fs
    t = np.arange(n) / FS
    return CubicSpline(t_low, g)(t)


def sweep(n: int, *, rate_hz_s: float, limit_hz: float) -> np.ndarray:
    """A carrier offset, per sample, swept between -``limit_hz`` and
    +``limit_hz`` at ``rate_hz_s`` as a triangle, from 0 upwards
    (MIL-STD-188-110B C.6.4's Doppler sweep)."""
    t = np.arange(n) / FS
    period = 4 * limit_hz / rate_hz_s
    ph = (t / period + 0.25) % 1.0                  # 0.25: start at 0 going up
    return limit_hz * (4 * np.abs(ph - 0.5) - 1) * -1


def channel(audio: np.ndarray, *, snr_db: float = 30.0, preset: str = "awgn",
            freq_hz: float | np.ndarray = 0.0, clock_ppm: float = 0.0,
            seed: int = 0, paths=None) -> np.ndarray:
    """``freq_hz`` may be an array, the offset at each sample (`sweep`).

    ``paths``, in place of ``preset``: (delay s, Doppler spread Hz) or
    (delay s, spread Hz, power dB) for each path. A spread of 0 is a fixed
    path. Without powers the paths share the power equally; with them, the
    total is normalised to 1. ``snr_db`` may be ``inf``."""
    rng = np.random.default_rng(seed)
    x = np.asarray(audio, dtype=np.float64)
    a = hilbert(x)
    paths = PRESETS[preset] if paths is None else tuple(paths)
    if paths:
        gains = np.array([10 ** (p[2] / 20) if len(p) > 2 else 1.0 for p in paths])
        gains = gains / np.sqrt(np.sum(gains ** 2))
        y = np.zeros_like(a)
        for (delay, spread, *_), g in zip(paths, gains):
            d = int(round(delay * FS))
            shifted = np.concatenate([np.zeros(d, complex), a[: a.size - d]])
            y += shifted * _fading(a.size, spread, rng) * g
        a = y
    if np.ndim(freq_hz):
        f = np.resize(np.asarray(freq_hz, dtype=np.float64), a.size)
        a = a * np.exp(2j * np.pi * np.cumsum(f) / FS)
    elif freq_hz:
        a = a * np.exp(2j * np.pi * freq_hz * np.arange(a.size) / FS)
    out = np.real(a)
    if clock_ppm:
        # The far end's clock runs fast by ``clock_ppm``: we see its samples
        # at slightly wider spacing.
        t = np.arange(0, out.size - 1, 1 + clock_ppm * 1e-6)
        out = CubicSpline(np.arange(out.size), out)(t)
    # The signal's power while it is on: the silence a test pads a burst
    # with is not signal, and averaged in it would set the noise that much
    # lower than the SNR asked for.
    on = np.flatnonzero(np.abs(x) > 1e-9 * (np.max(np.abs(x)) + 1e-300))
    p_sig = np.mean(x[on[0]:on[-1] + 1] ** 2) if on.size else 0.0
    # Noise power in 3 kHz of a white process sampled at FS.
    n0 = p_sig / 10 ** (snr_db / 10) / 3000.0
    noise = rng.normal(0, np.sqrt(n0 * FS / 2), out.size)
    return out + noise
