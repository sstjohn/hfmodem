# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive an explicitly located P4 normal-short BPSK body (SL5/SL6).

Six 32-symbol CAZAC-trained blocks of 176 BPSK symbols gather through the
archived component/symbol tables to 352-symbol systematic, C1 and C2 fields,
whose first 344 decisions pack LSB-first into the 43-byte P4-whitened,
X-25-checked frame. This path has no header interpretation or ARQ. The
caller must establish the body geometry, training origin, and frequency
offset. Returned data remains source coded.
"""
from __future__ import annotations

import numpy as np

from . import p4c2, p4decode, p4frame, p4normal, p4rx, tablegen


def component_permutation() -> np.ndarray:
    """Exact 352-entry direct-gather order, returned as a fresh array."""
    return p4c2.permutation(352).copy()


def decode_body(baseband, *, first_training_start: int,
                frequency_offset: float = 0.0) -> p4frame.Frame | None:
    """Decode complex 28800-Hz baseband, accepting only a unique CRC pass.

    The start is the training's impulse coordinate before transmit shaping,
    not the matched-filter peak. Require full body plus receive filter context.
    Incomplete/silent input or CRC failure returns None; malformed input raises
    ValueError. No CFO/timing search or implicit header/packet classification.
    Constituent states are free and unresolved parity tails are erased.
    """
    z = np.asarray(baseband, dtype=complex)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("baseband must be a finite one-dimensional vector")
    if type(first_training_start) is not int or first_training_start < 0:
        raise ValueError("first_training_start must be a nonnegative integer")
    if not np.isscalar(frequency_offset) or not np.isreal(frequency_offset) or not np.isfinite(frequency_offset):
        raise ValueError("frequency_offset must be finite and real")
    start = first_training_start
    if len(z) < start + 1248 * 16 + 128:
        return None
    # Bound numerical range; the channel fit handles absolute gain and phase.
    peak = np.max(abs(z))
    if peak == 0:
        return None
    z = z / peak * np.exp(-2j * np.pi * frequency_offset * np.arange(len(z)) / 28800)
    matched = p4rx.matched_filter(z)
    targets = np.arange(10, 22)
    training = np.array([p4normal.training_sequence(k + 1) for k in range(6)])
    positions = start + 128 + 16 * (208 * np.arange(6)[:, None] + targets)
    received = matched[positions]
    design = training[:, targets[:, None] + np.arange(-2, 3)]
    fit = [0, 1, 4, 5]
    matrix = design[fit].reshape(-1, 5)
    observed = received[fit].ravel()
    if not np.any(observed):
        return None
    gram = matrix.conj().T @ matrix
    channel = np.linalg.solve(gram + 1e-6 * np.trace(gram).real / 5 * np.eye(5),
                              matrix.conj().T @ observed)
    variance = float(np.mean(abs(observed - matrix @ channel)**2))
    # Choose a finite inverse by minimizing ISI plus receiver-noise energy.
    convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
    target = np.zeros(15)
    target[7] = 1
    if variance == 0:
        inverse = np.linalg.lstsq(convolution, target, rcond=None)[0]
    else:
        inverse = np.linalg.solve(convolution.conj().T @ convolution + variance * np.eye(11),
                                  convolution.conj().T @ target)
    residual = np.convolve(channel, inverse)
    gain = residual[7]
    residual[7] = 0
    noise = float(variance * np.vdot(inverse, inverse).real + np.vdot(residual, residual).real)
    if noise <= 0 or not np.isfinite(noise):
        return None
    body = (208 * np.arange(6)[:, None] + 32 + np.arange(176)).ravel()
    windows = start + 128 + 16 * (body[:, None] + np.arange(-5, 6))
    supported = ((body >= 15) & (body + 15 < 1248)
                 & (windows[:, 0] >= 128) & (windows[:, -1] < len(z)))
    scores = np.zeros(1056)
    equalized = matched[windows[supported]] @ inverse
    scores[supported] = 4 * (gain.conjugate() * equalized).real / noise
    fields = scores[tablegen.INTERLEAVERS['0xbaf20_1056']()]
    return decode_scores(fields)[0]


def decode_scores(fields, *, iterations=8):
    fields = np.asarray(fields, dtype=float)
    if fields.shape != (1056,) or not np.all(np.isfinite(fields)):
        raise ValueError("normal short requires 1056 finite scores")
    systematic, c1, c2 = np.split(fields, 3)
    c1, c2 = c1.copy(), c2.copy()
    c1[-8:] = c2[-8:] = 0
    decoded = p4decode.turbo_decode(systematic, c1, c2, component_permutation(),
                                   iterations=iterations)
    accepted = []
    # Complementing every input and state complements both parity fields.
    # Free boundary states therefore give the negated posterior for -fields.
    for polarity in (1, -1):
        bits = decoded.posterior[:344] * polarity < 0
        wire = np.packbits(bits, bitorder='little').tobytes()
        frame = p4frame.decode_field(wire, data_bytes=40)
        if frame is not None:
            accepted.append((frame, polarity))
    return accepted[0] if len(accepted) == 1 else (None, None)


GEOM = 2548
NORMAL_COHERENCE_KNEE = .30
NORMAL_ACQUIRE_KNEE = .35
TRAIN = np.array([p4normal.training_sequence(k + 1) for k in range(7)])
NARROW = np.arange(-24, 49, 2)
WIDE = np.arange(-1008, 1009, 2)          # +-35 ms
_RES = np.arange(-3.0, 3.0 + 1e-9, 0.1)
_TT = np.arange(4, 28)
_POS = (128 + 16 * (208 * np.arange(7)[:, None] + _TT)).ravel()
_EXP = TRAIN[:, _TT].ravel().conj()
_ROT = np.exp(-2j * np.pi * _POS[:, None] * _RES / 28800)
_LONG_TRAIN = np.array([p4normal.training_sequence(k + 1) for k in range(25)])
_LONG_POS = 128 + 16 * (239 * np.arange(25)[:, None] + _TT)
_LONG_EXP = _LONG_TRAIN[:, _TT].conj()
_LONG_ROT = {n: np.exp(-2j * np.pi * _LONG_POS[:n].ravel()[:, None] * _RES / 28800)
             for n in (5, 25)}


def complete(z, first_training_start):
    return first_training_start + 128 + 16 * 1275 + 64 < len(z)


def place_short(z, first_training_start, cfo_hz, *, acquisition=False, offsets=None):
    offsets = (WIDE if acquisition else NARROW) if offsets is None else offsets
    if not complete(z, first_training_start + int(max(offsets))):
        return None
    m = p4rx.matched_filter(z * np.exp(-2j * np.pi * cfo_hz * np.arange(len(z)) / 28800))
    return _place(m, first_training_start, cfo_hz, offsets, _POS, _EXP, _ROT)


def _place(m, first_training_start, cfo_hz, offsets, pos, expected, rotated):
    starts = first_training_start + np.asarray(offsets, dtype=int)
    starts = starts[(starts+pos[0] >= 0) & (starts+pos[-1] < len(m))]
    if not len(starts):
        return 0., first_training_start, cfo_hz
    values = m[starts[:, None]+pos]
    norm = np.sqrt(np.sum(abs(values)**2, axis=1) * len(pos))
    correlations = abs((values*expected) @ rotated)
    coherence = correlations / np.maximum(norm[:, None], 1e-30)
    row, column = np.unravel_index(coherence.argmax(), coherence.shape)
    if coherence[row, column] == 0:
        return 0., first_training_start, cfo_hz
    return float(coherence[row, column]), int(starts[row]), float(cfo_hz+_RES[column])


def place_long(z, first_training_start, cfo_hz, *, offsets=None, trainings=25):
    offsets = WIDE if offsets is None else offsets
    pos = _LONG_POS[:trainings].ravel()
    expected = _LONG_EXP[:trainings].ravel()
    rotated = _LONG_ROT.get(trainings)
    if rotated is None:
        rotated = np.exp(-2j * np.pi * pos[:, None] * _RES / 28800)
    m = p4rx.matched_filter(z * np.exp(-2j * np.pi * cfo_hz * np.arange(len(z)) / 28800))
    return _place(m, first_training_start, cfo_hz, offsets, pos, expected, rotated)


def short_scores(baseband, *, first_training_start, frequency_offset=0., refine=3.,
                 on_symbols=None, on_equalized=None):
    z = np.asarray(baseband, dtype=complex)
    start = first_training_start
    if z.ndim != 1 or not np.all(np.isfinite(z)) or type(start) is not int or start < 0:
        raise ValueError("finite baseband and nonnegative integer training start required")
    if not np.isfinite(frequency_offset) or refine not in (0, 3.):
        raise ValueError("finite carrier and refinement of 0 or 3 Hz required")
    if not complete(z, start) or not np.any(z):
        return None
    placed = place_short(z, start, frequency_offset, offsets=(0,)) if refine else None
    coherence, _, cfo = placed if placed else (0., start, frequency_offset)
    z = z / np.max(abs(z)) * np.exp(-2j * np.pi * cfo * np.arange(len(z)) / 28800)
    matched = p4rx.matched_filter(z)
    targets = np.arange(10, 22)
    positions = start + 128 + 16 * (208 * np.arange(7)[:, None] + targets)
    observed = matched[positions].ravel()
    matrix = TRAIN[:, targets[:, None] + np.arange(-2, 3)].reshape(-1, 5)
    gram = matrix.conj().T @ matrix
    channel = np.linalg.solve(gram + 1e-6 * np.trace(gram).real / 5 * np.eye(5),
                              matrix.conj().T @ observed)
    variance = float(np.mean(abs(observed - matrix @ channel)**2))
    convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
    target = np.zeros(15)
    target[7] = 1
    inverse = np.linalg.lstsq(convolution, target, rcond=None)[0] if variance == 0 else np.linalg.solve(
        convolution.conj().T @ convolution + variance * np.eye(11), convolution.conj().T @ target)
    residual = np.convolve(channel, inverse)
    gain = residual[7]
    residual[7] = 0
    noise = float(variance * np.vdot(inverse, inverse).real + np.vdot(residual, residual).real)
    if noise <= 0 or not np.isfinite(noise):
        return None
    body = (208 * np.arange(6)[:, None] + 32 + np.arange(176)).ravel()
    windows = start + 128 + 16 * (body[:, None] + np.arange(-5, 6))
    equalized = gain.conjugate() * (matched[windows] @ inverse)
    if on_symbols is not None and abs(gain)>0:
        on_symbols(equalized / abs(gain)**2, start + 16 * body)
    if on_equalized is not None and abs(gain) > 0:
        on_equalized(equalized / abs(gain)**2, noise / abs(gain)**2)
    scores = 4 * equalized.real / noise
    return coherence, cfo, scores[tablegen.INTERLEAVERS['0xbaf20_1056']()]
