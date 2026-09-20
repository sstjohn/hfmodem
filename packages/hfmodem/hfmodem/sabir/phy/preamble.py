# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Paired conjugate-root Zadoff-Chu preamble: burst detection, timing, and CFO.

A single ZC/chirp correlator has a delay-Doppler ambiguity: a carrier offset
of f shifts the correlation peak in time by -f * K samples (K = N_zc D^2 / fs
for root u = 1), so timing and CFO cannot be separated. Transmitting two
segments with conjugate roots (u and -u) breaks the ambiguity -- their peaks
shift in *opposite* directions, so the sum of the peak positions gives timing
and the difference gives CFO. The detector searches beyond one carrier spacing;
successful acquisition still depends on SNR, drift and multipath.

Numerology: 128 centred-chirp chips at 1200 chips/s (40x upsampled to 48 kHz),
mixed to the 1500 Hz band centre. Its nominal 1200 Hz chip bandwidth is
shared by every OFDM gear; interpolation and burst edges add spectral skirts. Segment A carries root +1, segment B its conjugate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from hfmodem.sabir.dsp.sync import Detection

from .rate import CENTER_HZ, FS

N_ZC = 128
CHIP_RATE = 1200.0
UPSAMPLE = 40
SEG_LEN = N_ZC * UPSAMPLE                    # samples per segment at 48 kHz
assert CHIP_RATE * UPSAMPLE == FS      # the chip geometry lands on the rate
# peak shift per Hz of CFO for root u=1 (derived from the chirp's
# delay<->frequency equivalence): K = N_zc * D^2 / fs
K_SAMPLES_PER_HZ = N_ZC * UPSAMPLE**2 / FS


def _chips() -> np.ndarray:
    n = np.arange(N_ZC)
    # centred chirp: instantaneous frequency spans +/- chip-rate/2, no aliasing
    return np.exp(1j * np.pi * (n - N_ZC / 2) ** 2 / N_ZC)


def preamble_waveform() -> np.ndarray:
    """Unit-envelope chips, interpolated and mixed to the audio center."""
    z = _chips()
    full = resample_poly(np.concatenate([z, np.conj(z)]), UPSAMPLE, 1)
    t = np.arange(full.size) / FS
    return full * np.exp(2j * np.pi * CENTER_HZ * t)


def _peak_interp(mag: np.ndarray, p: int) -> float:
    if 0 < p < mag.size - 1:
        a, b, c = mag[p - 1], mag[p], mag[p + 1]
        denom = a - 2 * b + c
        if denom < 0:
            return p + 0.5 * (a - c) / denom
    return float(p)


class ZCPreambleDetector:
    """Burst timing and carrier offset from the conjugate chirp pair."""

    def __init__(self, max_cfo_hz: float = 90.0, threshold: float = 5.0,
                 early_bias: int = 64):
        wave = preamble_waveform()
        self.ref_a = wave[:SEG_LEN]
        self.ref_b = wave[SEG_LEN:]
        self.max_shift = int(np.ceil(K_SAMPLES_PER_HZ * max_cfo_hz))
        self.threshold = threshold
        # Start the FFT window this many samples early: any start inside the
        # (window-taper-reduced) cyclic prefix is ISI-free and shows up only as
        # a phase ramp the pilot equaliser absorbs, whereas a late start eats
        # the next symbol. Biasing early converts timing noise into margin.
        self.early_bias = early_bias

    def detect(self, samples: np.ndarray) -> Optional[Detection]:
        r = np.asarray(samples, dtype=np.complex128)
        if r.size < 2 * SEG_LEN:
            return None
        ca = np.abs(fftconvolve(r, np.conj(self.ref_a[::-1]), mode="valid"))
        cb = np.abs(fftconvolve(r, np.conj(self.ref_b[::-1]), mode="valid"))
        pa = int(np.argmax(ca))
        if not np.isfinite(ca[pa]) or ca[pa] <= 0:
            return None
        # A deeply faded preamble in front of an unfaded body can fail the
        # single-segment gate (the median is body-sidelobe level), so a
        # marginal peak is still accepted when the conjugate-pair
        # cross-correlation below confirms it. Pair prominence is an additional
        # detection gate; frame CRCs still decide whether data is accepted.
        sure = ca[pa] >= self.threshold * np.median(ca)
        # Segment B sits SEG_LEN later, shifted +K*CFO while A shifts -K*CFO.
        # On a multipath channel each envelope has one peak per tap, and the
        # taps fade independently between the two segments -- picking each
        # segment's argmax can pair peaks from *different* taps, which reads
        # as a large phantom CFO (a 2 ms echo maps to ~11 Hz). Correlating a
        # window of envelope A against envelope B instead scores each
        # candidate lag by the alignment of the whole multipath profile, so
        # the true CFO lag (which aligns every tap at once) wins.
        span = 2 * self.max_shift
        m = span + 4 * UPSAMPLE          # window: CFO walk + delay spread
        a_lo, a_hi = max(0, pa - m), min(ca.size, pa + m + 1)
        wa = ca[a_lo:a_hi]
        b_lo = a_lo + SEG_LEN - span
        b_hi = a_hi + SEG_LEN + span
        if b_lo < 0 or b_hi > cb.size:
            return None
        xc = np.correlate(cb[b_lo:b_hi], wa, mode="valid")
        med = np.median(xc)
        prominence = (xc.max() - med) / (med - xc.min() + 1e-12)
        if not sure and prominence < 2.5:
            return None
        lag = int(np.argmax(xc))
        flag = _peak_interp(xc, lag) - span      # = (fb - fa) - SEG_LEN
        fa = _peak_interp(ca, pa)
        tau = fa + flag / 2
        cfo = flag / (2 * K_SAMPLES_PER_HZ)
        return Detection(
            frame_start=int(round(tau)) + 2 * SEG_LEN - self.early_bias,
            coarse_cfo_hz=float(cfo),
            metric=float(ca[pa] / max(float(np.median(ca)), 1e-300)),
            preamble_len=2 * SEG_LEN,
        )
