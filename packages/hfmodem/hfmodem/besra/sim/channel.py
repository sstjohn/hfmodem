# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""HF channel models for the bench: clean, AWGN, and a two-path Watterson fader.

Pure functions on int16 audio at 12 kHz — no coupling to the modem. Used by the
two-modem loopback and the noise-knee sweeps. SNR is defined in the ARDOP
convention: signal power measured over the active (non-silent) samples, noise
added at the requested dB below it across the whole block.
"""

from __future__ import annotations

import numpy as np

from ..dsp.templates import SAMPLE_RATE


def _rng(seed: int | None) -> np.random.Generator:
    # np.random.default_rng(None) seeds from the OS; tests pass an explicit seed.
    return np.random.default_rng(seed)


def add_awgn(samples: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add white Gaussian noise at ``snr_db`` relative to the signal's active
    power (mean square over samples above 1% of full scale)."""
    x = samples.astype(np.float64)
    active = x[np.abs(x) > 0.01 * 32767]
    sig_pow = float(np.mean(active ** 2)) if active.size else float(np.mean(x ** 2))
    if sig_pow <= 0:
        return samples.copy()
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = _rng(seed).normal(0.0, np.sqrt(noise_pow), size=x.shape)
    return _clip16(x + noise)


def watterson(samples: np.ndarray, snr_db: float, *, spread_hz: float = 1.0,
              delay_ms: float = 2.0, seed: int | None = None) -> np.ndarray:
    """A two-path Watterson fade (CCIR 520): two independent Rayleigh paths, the
    second delayed by ``delay_ms`` and each faded with a Gaussian Doppler of
    ``spread_hz``, then AWGN at ``snr_db``. The canonical HF test channel."""
    x = samples.astype(np.float64)
    rng = _rng(seed)
    delay = int(round(delay_ms * 1e-3 * SAMPLE_RATE))

    faded = _fade(x, spread_hz, rng) + _fade(_shift(x, delay), spread_hz, rng)
    faded *= np.sqrt(0.5)                        # equal-power paths → unit mean gain
    out = add_awgn(_clip16(faded), snr_db, seed=int(rng.integers(1 << 31)))
    return out


def _fade(x: np.ndarray, spread_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply by one Rayleigh path: complex Gaussian gain low-passed to a
    ``spread_hz`` Doppler bandwidth (the envelope the real part sees)."""
    n = x.size
    g = rng.normal(size=n) + 1j * rng.normal(size=n)
    g = _lowpass(g, spread_hz)
    g /= np.sqrt(np.mean(np.abs(g) ** 2))        # normalise to unit average power
    return x * np.abs(g)


def _lowpass(g: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Brick-wall low-pass in the frequency domain to the Doppler bandwidth."""
    n = g.size
    freqs = np.fft.fftfreq(n, d=1.0 / SAMPLE_RATE)
    G = np.fft.fft(g)
    G[np.abs(freqs) > max(cutoff_hz, 1.0 / n)] = 0
    return np.fft.ifft(G)


def _shift(x: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return x
    out = np.zeros_like(x)
    out[delay:] = x[:-delay]
    return out


def _clip16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -32768, 32767).astype("<i2")
