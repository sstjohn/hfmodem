# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline normal-mode training/BPSK observations, never a packet decoder.

SCS P4 §6.4 construction code is followed (32 samples), not the inconsistent
33-sample prose. Header length, terminal training count and bit labels remain
hypotheses supplied by the caller. This module has no runtime/RF integration.
"""
from dataclasses import dataclass

import numpy as np

from .p4rx import MATCHED_DELAY, RRC_TAPS, SAMPLE_RATE, SAMPLES_PER_CHIP, matched_filter

CAZAC16 = np.array([1, -1, -1j, -1, -1, -1j, -1j, 1j,
                    1, 1, -1j, 1, -1, 1j, -1j, -1j], complex)


def _integer(value, name, minimum=0):
    if (not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value)
            or int(value) != value or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def training_sequence(sequence_number):
    """Unit-amplitude C[9..16],C[1..16],C[1..8], numbered from ONE."""
    number = _integer(sequence_number, 'sequence_number', 1)
    c = CAZAC16 if number % 2 else CAZAC16.conj()
    return np.r_[c[8:], c, c[:8]]


@dataclass(frozen=True)
class NormalLayout:
    variant: str
    block_symbols: int
    blocks: int
    terminal_training: bool

    @property
    def training_offsets(self):
        return np.arange(self.blocks + int(self.terminal_training)) * (32 + self.block_symbols)

    @property
    def data_offsets(self):
        return np.arange(self.blocks) * (32 + self.block_symbols) + 32

    @property
    def total_symbols(self):
        return self.blocks * (32 + self.block_symbols) + 32 * int(self.terminal_training)


def layout(variant='short', *, terminal_training):
    """Explicit B versus B+1 training hypothesis; first symbol is training #1.

    Short uses table value 176, not prose typo 175. Break-in only describes
    candidate body geometry: its special header/coding are not established.
    """
    dimensions = {'short': (176, 6), 'long': (207, 24), 'breakin': (210, 4)}
    if variant not in dimensions or type(terminal_training) is not bool:
        raise ValueError('choose short/long/breakin and an explicit boolean terminal_training')
    return NormalLayout(variant, *dimensions[variant], terminal_training)


@dataclass(frozen=True)
class NormalObservation:
    first_training_start: int
    cfo_hz: float
    coherence: float
    candidates: np.ndarray  # start, CFO, coherent training score
    layout: NormalLayout
    matched: np.ndarray
    symbol_indices: np.ndarray
    symbols: np.ndarray
    training_coherence: np.ndarray
    channel_gain: np.ndarray  # scalar fit, diagnostic only
    equalizer: np.ndarray  # complex FIR coefficients per training
    condition_number: np.ndarray  # unregularized received-window design matrix
    training_nmse: np.ndarray
    holdout_nmse: np.ndarray  # every third interior training target excluded from fit
    equalized: np.ndarray  # blocks x block_symbols
    bpsk_observation: np.ndarray  # real projection, not bits or calibrated LLRs
    quadrature: np.ndarray


def recover_bpsk(baseband, *, first_training_start, layout: NormalLayout,
                  timing_offsets=range(-8, 9), cfo_hz=(0.0,),
                  equalizer_half_width=2, ridge=1e-3, bpsk_axis=0.0):
    """Bounded offline training acquisition and per-block FIR equalization.

    Input is complex baseband at 28800 Hz. Start is the first training chip's
    impulse coordinate BEFORE TX shaping; matched delay128 is added internally.
    Search assumes constant CFO/clock and coherent channel across trainings.
    Equalizer coefficients refresh at each preceding training and remain fixed
    over its following data block. No decision-directed loop or channel drift
    interpolation. Last optional training supplies diagnostics, not extra data.

    A complex ridge least-squares FIR maps received symbol windows to known
    training targets. Interior guard excludes unknown data/pulse edges. Holdout
    NMSE uses every third interior target omitted from a separate diagnostic fit;
    overlapping windows mean this is not independent recording validation.
    Ridge can return finite coefficients even for degenerate/noise training;
    unregularized design condition numbers expose that failure mode, but do not
    gate outputs. All outputs exist for noise/wrong layout; none accepts a packet.
    Caller supplies BPSK axis hypothesis; + and - signs have no assigned bits.
    """
    if not isinstance(layout, NormalLayout) or layout != globals()['layout'](
            layout.variant, terminal_training=layout.terminal_training):
        raise ValueError('layout must be one of the explicit published dimension hypotheses')
    start = _integer(first_training_start, 'first_training_start')
    half = _integer(equalizer_half_width, 'equalizer_half_width')
    if half > 4:
        raise ValueError('equalizer_half_width must be <= 4 for bounded 32-symbol training')
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(bpsk_axis):
        raise ValueError('ridge must be positive and bpsk_axis finite')
    z = np.asarray(baseband, complex)
    if z.ndim != 1 or not z.size or not np.all(np.isfinite(z)):
        raise ValueError('baseband must be a nonempty finite vector')
    offsets = np.asarray(tuple(timing_offsets), float)
    cfos = np.asarray(tuple(cfo_hz), float)
    if (offsets.ndim != 1 or not offsets.size or not np.all(np.isfinite(offsets))
            or np.any(offsets != np.round(offsets))):
        raise ValueError('timing_offsets must contain integer sample offsets')
    if cfos.ndim != 1 or not cfos.size or not np.all(np.isfinite(cfos)):
        raise ValueError('cfo_hz must contain finite hypotheses')
    guard = max(4, half + 2)
    targets = np.arange(guard, 32 - guard)
    positions = layout.training_offsets[:, None] + targets
    expected = np.array([training_sequence(k + 1)[targets]
                         for k in range(len(layout.training_offsets))])
    candidates, best = [], None
    native_offsets = np.arange(layout.total_symbols) * SAMPLES_PER_CHIP
    for cfo in cfos:
        mf = matched_filter(z * np.exp(-2j * np.pi * cfo * np.arange(z.size) / SAMPLE_RATE))
        for dt in offsets:
            candidate_start = start + int(dt)
            if candidate_start < 0 or candidate_start + native_offsets[-1] + len(RRC_TAPS) > z.size:
                continue
            indices = candidate_start + MATCHED_DELAY + native_offsets
            symbols = mf[indices]
            received = symbols[positions]
            denominator = np.sum(np.abs(received)**2) * expected.size
            score = float(abs(np.vdot(expected, received))**2 / max(denominator, 1e-30))
            candidates.append((candidate_start, cfo, score))
            if best is None or score > best[0]:
                best = score, candidate_start, float(cfo), mf, indices, symbols
    if best is None:
        raise ValueError('no candidate contains the complete hypothesized layout')
    score, start, cfo, mf, indices, symbols = best
    shifts = np.arange(-half, half + 1)
    equalizers, training_error, holdout_error, coherence, gain = [], [], [], [], []
    conditioning = []
    for block, offset in enumerate(layout.training_offsets):
        target = expected[block]
        matrix = symbols[offset + targets[:, None] + shifts]
        conditioning.append(float(np.linalg.cond(matrix)))
        test = np.arange(len(target)) % 3 == 0

        def fit(a, b):
            return np.linalg.solve(a.conj().T @ a + ridge * np.eye(len(shifts)), a.conj().T @ b)

        diagnostic = fit(matrix[~test], target[~test])
        eq = fit(matrix, target)
        equalizers.append(eq)
        training_error.append(float(np.mean(abs(matrix @ eq - target)**2)))
        holdout_error.append(float(np.mean(abs(matrix[test] @ diagnostic - target[test])**2)))
        y = symbols[offset + targets]
        h = np.vdot(target, y) / len(target)
        gain.append(h)
        coherence.append(float(abs(h)**2 * len(target) / max(np.sum(abs(y)**2), 1e-30)))
    # Zero padding only affects the last half symbols if no terminal training.
    # Mark those observations NaN: their future input context is unobserved.
    padded = np.pad(symbols, (half, half))
    equalized = []
    for block, offset in enumerate(layout.data_offsets):
        positions = offset + np.arange(layout.block_symbols)
        matrix = padded[positions[:, None] + shifts + half]
        values = matrix @ equalizers[block]
        values[positions + half >= len(symbols)] = np.nan + 1j * np.nan
        equalized.append(values)
    equalized = np.asarray(equalized)
    rotated = equalized * np.exp(-1j * bpsk_axis)
    return NormalObservation(start, cfo, score, np.asarray(candidates), layout,
                             mf, indices, symbols, np.asarray(coherence), np.asarray(gain),
                             np.asarray(equalizers), np.asarray(conditioning), np.asarray(training_error),
                             np.asarray(holdout_error), equalized, rotated.real, rotated.imag)
