# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Watterson HF channel model (ITU-R F.1487 Gaussian-scatter).

A tapped delay line: each tap is an independent complex-Gaussian (Rayleigh)
process whose power spectrum is Gaussian with standard deviation sigma_f and
optional centre (Doppler) shift. Per F.1487 the quoted "Doppler spread" is the
two-sided 2*sigma width, so sigma_f = spread/2. Taps are synthesised by
Gaussian-FIR filtering white complex noise at a low tap rate (the processes
are sub-Hz wide) and interpolating up to the signal rate; the filter kernel
has unit energy so each tap has unit average power, and the tap sum is scaled
by 1/sqrt(n_taps) so E[|y|^2] = E[|x|^2] and downstream SNR accounting is
preserved on ensemble average.

Standard profiles (delay / Doppler-spread), mid-latitude Good through the
high-latitude (polar) rows:
"""

from __future__ import annotations

import numpy as np

# name -> (tap delays in ms, Doppler spread 2*sigma in Hz)
PROFILES = {
    "good":            ((0.0, 0.5), 0.1),
    "moderate":        ((0.0, 1.0), 0.5),
    "poor":            ((0.0, 2.0), 1.0),
    "subpolar":        ((0.0, 2.0), 5.0),
    "nvis":            ((0.0, 7.0), 1.0),
    "polar":           ((0.0, 3.0), 10.0),
    "polar_disturbed": ((0.0, 7.0), 30.0),
}


def tap_process(n: int, fs: float, spread_hz: float, shift_hz: float = 0.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """One Rayleigh tap: ``n`` samples at ``fs`` of a unit-power complex
    Gaussian process with Gaussian Doppler PSD (2-sigma width ``spread_hz``,
    centred on ``shift_hz``)."""
    rng = rng or np.random.default_rng()
    sigma_f = spread_hz / 2.0
    if sigma_f <= 0:
        h = np.full(n, np.exp(2j * np.pi * rng.uniform()))
    else:
        # |G(f)|^2 Gaussian with sigma_f  =>  time kernel sigma_t = 1/(2*sqrt(2)*pi*sigma_f)
        fs_tap = 64.0 * sigma_f
        sigma_t = 1.0 / (2.0 * np.sqrt(2.0) * np.pi * sigma_f)
        half = int(np.ceil(4 * sigma_t * fs_tap))
        t = np.arange(-half, half + 1) / fs_tap
        g = np.exp(-t**2 / (2 * sigma_t**2))
        g /= np.sqrt(np.sum(g**2))            # unit energy -> unit output power
        n_tap = int(np.ceil(n / fs * fs_tap)) + 2
        wn = (rng.standard_normal(n_tap + g.size - 1)
              + 1j * rng.standard_normal(n_tap + g.size - 1)) / np.sqrt(2)
        low = np.convolve(wn, g, mode="valid")
        t_out = np.arange(n) / fs * fs_tap
        h = np.interp(t_out, np.arange(n_tap), low.real) \
            + 1j * np.interp(t_out, np.arange(n_tap), low.imag)
    if shift_hz:
        h = h * np.exp(2j * np.pi * shift_hz * np.arange(n) / fs)
    return h


class Watterson:
    """Apply a Watterson profile to a complex baseband/analytic stream."""

    def __init__(self, profile: str, fs: float,
                 rng: np.random.Generator | None = None,
                 shift_hz: float = 0.0):
        self.delays_ms, self.spread_hz = PROFILES[profile]
        self.fs = fs
        self.shift_hz = shift_hz
        self.rng = rng or np.random.default_rng()
        self.taps: list[np.ndarray] = []      # realisations of the last call

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        y = np.zeros_like(x)
        self.taps = []
        for d_ms in self.delays_ms:
            d = int(round(d_ms * 1e-3 * self.fs))
            h = tap_process(x.size, self.fs, self.spread_hz, self.shift_hz,
                            self.rng)
            self.taps.append(h)
            xd = np.concatenate([np.zeros(d, dtype=np.complex128), x[: x.size - d]])
            y += h * xd
        return y / np.sqrt(len(self.delays_ms))


class ContinuousWatterson:
    """A fixed fading realization indexed by absolute virtual time.

    Unlike Watterson.__call__, fading does not restart between packets or
    during ACK/turnaround gaps. Low-rate taps bound memory; interpolation is
    the same model used by tap_process. Use a separate instance per direction
    when testing independent forward/reverse paths.
    """
    def __init__(self, profile, fs, *, duration_s=3600, seed=0):
        self.delays_ms, spread = PROFILES[profile]
        self.fs, self.duration_s = fs, duration_s
        self.rate = max(4.0, 32 * spread)
        rng = np.random.default_rng(seed)
        n = int(np.ceil(duration_s * self.rate)) + 2
        self.taps = [tap_process(n, self.rate, spread, rng=rng) for _ in self.delays_ms]

    def __call__(self, x, t):
        if t < 0 or t + len(x) / self.fs > self.duration_s:
            raise ValueError("continuous channel trace exhausted")
        at = (t + np.arange(len(x)) / self.fs) * self.rate
        y = np.zeros(len(x), dtype=complex)
        for delay, tap in zip(self.delays_ms, self.taps):
            h = np.interp(at, np.arange(len(tap)), tap.real) + 1j * np.interp(at, np.arange(len(tap)), tap.imag)
            d = round(delay * self.fs / 1000)
            if d:
                y[d:] += h[d:] * x[:-d]
            else:
                y += h * x
        return y / np.sqrt(len(self.taps))
