# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Baseline channel models for the OFDM engine.

AWGN with the noise scaled to a *specified* Eb/N0, so BER can be checked
against textbook curves -- which is what makes the rest of the measurements
trustworthy. Because the OFDM engine uses orthonormal transforms and
unit-energy constellations, adding complex Gaussian noise of variance
``sigma^2`` per time sample yields exactly ``sigma^2`` noise variance per
recovered subcarrier, so the mapping from Eb/N0 to sigma is
transform-independent and exact.

Single-impairment helpers for carrier and sample-clock offset are included and
off by default: they exercise one axis of the synchroniser at a time. Fading is
not modelled here -- ``sim/watterson.py`` carries the ITU-R F.1487 profiles.
"""

from __future__ import annotations

import numpy as np


def awgn(
    x: np.ndarray,
    ebn0_db: float,
    bits_per_symbol: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add complex AWGN to baseband samples ``x`` for a target Eb/N0 (dB).

    ``bits_per_symbol`` is the *constellation* order (e.g. 2 for QPSK). With
    unit-energy constellation points and orthonormal OFDM transforms, the
    per-subcarrier symbol energy Es = 1, so N0 = 1 / (k * Eb/N0) and the
    per-sample complex-noise variance equals N0.
    """
    rng = rng or np.random.default_rng()
    ebn0 = 10 ** (ebn0_db / 10)
    n0 = 1.0 / (bits_per_symbol * ebn0)          # noise variance per complex dim
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + noise


def apply_cfo(x: np.ndarray, cfo_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Constant carrier-frequency offset. Off by default.

    ``sync.py`` estimates and corrects this from the preamble and the pilots;
    applying it here in isolation is how that estimator gets exercised.
    """
    n = np.arange(x.size)
    return x * np.exp(2j * np.pi * cfo_hz * n / sample_rate_hz)


def apply_sco(x: np.ndarray, ppm: float) -> np.ndarray:
    """Marked seam: sampling-clock offset (SCO). Off by default.

    Models the receiver's ADC clock running ``ppm`` parts-per-million away from
    the transmitter's DAC clock: the RX re-samples the same continuous waveform
    on a stretched/compressed time grid. A positive ``ppm`` means the RX clock is
    *faster* (more samples per second), so the recovered stream is slightly
    longer and the OFDM symbol boundary drifts a fraction of a sample per symbol.

    This is the impairment the sync layer's residual-timing / SCO tracker must
    follow over a long frame. Implemented by band-limited-friendly linear
    interpolation, which is exact enough for heavily oversampled OFDM bursts
    (occupied BW << sample rate).
    """
    x = np.asarray(x)
    n = x.size
    scale = 1.0 + ppm * 1e-6                 # RX sample spacing in TX samples
    t = np.arange(0, n - 1, 1.0 / scale)     # RX sample instants, in TX samples
    grid = np.arange(n)
    xr = np.interp(t, grid, x.real)
    xi = np.interp(t, grid, x.imag)
    return xr + 1j * xi


def delay_fractional(x: np.ndarray, delta: float) -> np.ndarray:
    """Delay ``x`` by ``delta`` samples (may be fractional) via an FFT phase ramp.

    A test/sync helper for injecting or correcting sub-sample timing error.
    The shift is circular, which is harmless for the guarded bursts used here.
    Positive ``delta`` delays the signal (moves energy to later samples).
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    f = np.fft.fftfreq(n)
    return np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * f * delta))
