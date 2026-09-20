# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline P4 robust symbol observations; no header, bit labels or payload decoder.

SCS PACTOR-4 protocol §§6.2,10 supply the symbol geometry.
Input to recovery is complex baseband at 28,800 Hz. The caller supplies a
reference boundary hypothesis and a bounded CFO grid; header length and meaning
are deliberately outside this API. SF16 supports SL2/3, SF8 supports SL4, but
spreading factor alone cannot identify speed or distinguish data from controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import fftconvolve, hilbert, resample_poly

from .p4sig import CENTRE_HZ, CHIP_RATE, SPREAD16

SAMPLES_PER_CHIP = 16
SAMPLE_RATE = int(CHIP_RATE * SAMPLES_PER_CHIP)
# Exact 129-entry §10 array, including the terminal zero. Not an ideal RRC.
RRC_TAPS = np.array([
    226,215,194,161,117,61,-6,-84,-171,-263,-356,-446,-525,-586,-623,-627,
    -592,-514,-388,-215,4,262,550,855,1162,1452,1705,1898,2013,2030,1933,1711,
    1358,875,270,-440,-1230,-2070,-2919,-3733,-4464,-5058,-5466,-5638,-5530,-5103,-4330,-3194,
    -1689,174,2374,4873,7621,10554,13601,16682,19712,22603,25270,27634,29621,31168,32228,32767,
    32767,32228,31168,29621,27634,25270,22603,19712,16682,13601,10554,7621,4873,2374,174,-1689,
    -3194,-4330,-5103,-5530,-5638,-5466,-5058,-4464,-3733,-2919,-2070,-1230,-440,270,875,1358,
    1711,1933,2030,2013,1898,1705,1452,1162,855,550,262,4,-215,-388,-514,-592,
    -627,-623,-586,-525,-446,-356,-263,-171,-84,-6,61,117,161,194,215,226,0,
], dtype=float) / 32767.0
_SPREAD8_IQ = np.array([32767,0,30273,12539,0,32767,-30273,-12539,
                        32767,0,-30273,-12539,0,32767,30273,12539])
SPREAD8 = (_SPREAD8_IQ[::2] + 1j * _SPREAD8_IQ[1::2]) / 32767.0
MATCHED_DELAY = len(RRC_TAPS) - 1


def _vector(values, *, real=False):
    x = np.asarray(values)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("expected a nonempty finite one-dimensional signal")
    if real and np.iscomplexobj(x):
        raise ValueError("audio must be real; pass complex baseband directly to recovery")
    return x.astype(float if real else complex)


def audio_to_baseband(audio, fs: int, *, centre_hz: float = CENTRE_HZ):
    """Analytic real audio → complex baseband, resampled to SAMPLE_RATE.

    Integer input rate only. Hilbert/resampling edge transients require padding
    around the burst; reference coordinates are measured in the returned array.
    Positive-frequency audio is used, without conjugation or inversion search.
    """
    x = _vector(audio, real=True)
    if not np.isfinite(fs) or fs <= 0 or int(fs) != fs or not np.isfinite(centre_hz):
        raise ValueError("fs must be a positive integer and centre_hz finite")
    z = hilbert(x) * np.exp(-2j * np.pi * centre_hz * np.arange(x.size) / fs)
    g = gcd(SAMPLE_RATE, int(fs))
    return resample_poly(z, SAMPLE_RATE // g, int(fs) // g)


def matched_filter(baseband):
    """Full convolution with reversed published taps / pulse energy.

    If TX inserts chip impulse at n and shapes with RRC_TAPS, its matched peak
    is at n+128 (the trailing zero is retained). Isolated chip gain is unity.
    This returns all convolution samples, without trimming or time relabelling.
    """
    z = _vector(baseband)
    return fftconvolve(z, RRC_TAPS[::-1], mode="full") / np.dot(RRC_TAPS, RRC_TAPS)


@dataclass(frozen=True)
class RobustObservation:
    reference_start: int
    cfo_hz: float
    coherence: float
    candidates: np.ndarray  # rows: reference_start, CFO Hz, spread coherence
    matched: np.ndarray
    chip_indices: np.ndarray  # indices in matched array, including reference
    chips: np.ndarray
    symbols: np.ndarray  # reference first, then body; unknown complex channel gain
    symbol_coherence: np.ndarray
    differential: np.ndarray  # body * conj(previous); not decoded dibits
    phase_alphabet: np.ndarray
    phase_similarity: np.ndarray  # cos(angle(differential)-hypothesis), NOT LLRs
    phase_margin: np.ndarray  # best-minus-second similarity, zero for erasures


def recover_robust(baseband, *, reference_start: int, spread_factor: int,
                   data_symbols: int, timing_offsets=range(-8, 9), cfo_hz=(0.0,),
                   phase_alphabet=(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    """Recover soft phase observations around a supplied reference boundary.

    reference_start + timing_offset is the first reference chip's TX impulse
    index at SAMPLE_RATE, BEFORE TX pulse shaping. All offsets are integer
    samples. Need a complete reference plus body in the supplied recording.
    No automatic reference/header identification: caller owns that hypothesis.

    CFO correction is a constant complex rotation BEFORE matched filtering.
    Candidate rank is projection energy onto the published spreading sequence
    divided by chip energy across the whole supplied symbol block. This is an
    acquisition diagnostic, not a calibrated detector probability/acceptance.
    Multipath can bias the chosen timing/CFO grid point even at high coherence.
    Fixed chip clock, no time drift tracking or multipath equalizer. Deep fades
    and ISI can produce misleading phase confidence; noise alone can produce
    a high phase margin, which must never be used as packet confidence.
    The four default phase
    angles are an explicit unlabelled alphabet hypothesis, not published bits.
    """
    z = _vector(baseband)
    if not np.isscalar(spread_factor) or spread_factor not in (8, 16):
        raise ValueError("spread_factor must be 8 or 16")
    spread_factor = int(spread_factor)
    for value in (reference_start, data_symbols):
        if not np.isscalar(value) or not np.isfinite(value) or int(value) != value:
            raise ValueError("reference_start and data_symbols must be finite integers")
    if data_symbols < 1:
        raise ValueError("data_symbols must be a positive integer")
    offsets = np.asarray(tuple(timing_offsets), dtype=float)
    cfos = np.asarray(tuple(cfo_hz), dtype=float)
    phases = np.asarray(phase_alphabet, dtype=float)
    if (offsets.ndim != 1 or not offsets.size or not np.all(np.isfinite(offsets))
            or np.any(offsets != np.round(offsets))):
        raise ValueError("timing_offsets must contain finite integer sample offsets")
    if cfos.ndim != 1 or not cfos.size or not np.all(np.isfinite(cfos)):
        raise ValueError("cfo_hz must contain finite frequency hypotheses")
    if phases.ndim != 1 or phases.size < 2 or not np.all(np.isfinite(phases)):
        raise ValueError("phase_alphabet must contain at least two finite angles")
    spread = SPREAD8 if spread_factor == 8 else SPREAD16
    energy = float(np.vdot(spread, spread).real)
    n_chips = (int(data_symbols) + 1) * spread_factor
    chip_offsets = np.arange(n_chips) * SAMPLES_PER_CHIP
    candidates, best = [], None
    for cfo in cfos:
        mf = matched_filter(z * np.exp(-2j * np.pi * cfo * np.arange(z.size) / SAMPLE_RATE))
        for offset in offsets:
            start = int(reference_start + offset)
            # Require complete TX support rather than accept convolution padding.
            if start < 0 or start + chip_offsets[-1] + len(RRC_TAPS) > z.size:
                continue
            indices = start + MATCHED_DELAY + chip_offsets
            chips = mf[indices].reshape(-1, spread_factor)
            symbols = chips @ spread.conj() / energy
            received_energy = np.sum(np.abs(chips) ** 2, axis=1)
            projected = np.abs(symbols) ** 2 * energy
            coherence = float(projected.sum() / max(received_energy.sum(), 1e-30))
            candidates.append((start, cfo, coherence))
            if best is None or coherence > best[0]:
                best = (coherence, start, float(cfo), mf, indices, chips, symbols,
                        projected / np.maximum(received_energy, 1e-30))
    if best is None:
        raise ValueError("no candidate contains the complete reference and body")
    coherence, start, cfo, mf, indices, chips, symbols, symbol_coherence = best
    differential = symbols[1:] * symbols[:-1].conj()
    magnitude = np.abs(differential)
    unit = differential / np.maximum(magnitude, 1e-30)
    similarity = (unit[:, None] * np.exp(-1j * phases)).real
    ordered = np.sort(similarity, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return RobustObservation(start, cfo, coherence, np.asarray(candidates), mf,
                             indices, chips, symbols, symbol_coherence,
                             differential, phases, similarity, margin)
