# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive an explicitly located P4 normal-long BPSK body (3.75 s cycle).

The long counterpart of `p4short`, and the same standing limits: no header
interpretation, no ARQ, no acquisition. The caller establishes the body
geometry, the training origin and the frequency offset, and what comes back is
still source coded. Section 6.4 lays twenty-four 207-symbol data blocks behind
twenty-four 32-symbol trainings and closes with a twenty-fifth.

Functional evidence: docs/protocols/pactor/pactor4-framing.md.
"""
from __future__ import annotations

import numpy as np

from . import p4c2, p4decode, p4frame, p4normal, p4rx, tablegen

LAYOUT = p4normal.layout("long", terminal_training=True)
"""24 data blocks, 25 trainings, 5768 symbols."""

NET_BYTES = 206
"""Section 3's N for the normal long: 203 user bytes, the status byte, the CRC."""

SYSTEMATIC_BITS = 1656
"""D for the same row -- eight more than the field's 1648."""

def component_permutation() -> np.ndarray:
    """Exact 1656-entry direct-gather order, returned as a fresh array."""
    return p4c2.permutation(1656).copy()


def body_scores(baseband, *, first_training_start: int,
                frequency_offset: float = 0.0) -> np.ndarray | None:
    """The 4968 coded-bit LLRs a located body carries, in figure 7.2's order.

    The start is the training's impulse coordinate before transmit shaping, not
    the matched-filter peak. One five-tap channel is fitted across all
    twenty-five trainings and inverted under its own residual variance. The
    scale is the fit's, not a calibrated one. Incomplete or silent input
    returns None; malformed input raises ValueError.
    """
    z = np.asarray(baseband, dtype=complex)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("baseband must be a finite one-dimensional vector")
    if type(first_training_start) is not int or first_training_start < 0:
        raise ValueError("first_training_start must be a nonnegative integer")
    if (not np.isscalar(frequency_offset) or not np.isreal(frequency_offset)
            or not np.isfinite(frequency_offset)):
        raise ValueError("frequency_offset must be finite and real")
    start = first_training_start
    spacing = 32 + LAYOUT.block_symbols
    if len(z) < start + LAYOUT.total_symbols * p4rx.SAMPLES_PER_CHIP + 128:
        return None
    peak = np.max(abs(z))
    if peak == 0:
        return None
    z = z / peak * np.exp(-2j * np.pi * frequency_offset * np.arange(len(z)) / 28800)
    matched = p4rx.matched_filter(z)
    targets = np.arange(10, 22)
    trainings = len(LAYOUT.training_offsets)
    training = np.array([p4normal.training_sequence(k + 1) for k in range(trainings)])
    positions = start + 128 + 16 * (spacing * np.arange(trainings)[:, None] + targets)
    observed = matched[positions].ravel()
    if not np.any(observed):
        return None
    matrix = training[:, targets[:, None] + np.arange(-2, 3)].reshape(-1, 5)
    gram = matrix.conj().T @ matrix
    channel = np.linalg.solve(gram + 1e-6 * np.trace(gram).real / 5 * np.eye(5),
                              matrix.conj().T @ observed)
    variance = float(np.mean(abs(observed - matrix @ channel)**2))
    convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
    target = np.zeros(15)
    target[7] = 1
    if variance == 0:
        inverse = np.linalg.lstsq(convolution, target, rcond=None)[0]
    else:
        inverse = np.linalg.solve(
            convolution.conj().T @ convolution + variance * np.eye(11),
            convolution.conj().T @ target)
    residual = np.convolve(channel, inverse)
    gain = residual[7]
    residual[7] = 0
    noise = float(variance * np.vdot(inverse, inverse).real
                  + np.vdot(residual, residual).real)
    if noise <= 0 or not np.isfinite(noise):
        return None
    body = (spacing * np.arange(LAYOUT.blocks)[:, None] + 32
            + np.arange(LAYOUT.block_symbols)).ravel()
    windows = start + 128 + 16 * (body[:, None] + np.arange(-5, 6))
    scores = 4 * (gain.conjugate() * (matched[windows] @ inverse)).real / noise
    return scores[tablegen.INTERLEAVERS["0xb81c0_4968"]()]


def decode_body(baseband, *, first_training_start: int,
                frequency_offset: float = 0.0) -> p4frame.Frame | None:
    """Decode complex 28800-Hz baseband, accepting only a unique CRC pass.

    Incomplete/silent input or CRC failure returns None; malformed input raises
    ValueError. No CFO/timing search or implicit header/packet classification.
    Constituent states are free and unresolved parity tails are erased.
    """
    fields = body_scores(baseband, first_training_start=first_training_start,
                         frequency_offset=frequency_offset)
    if fields is None:
        return None
    accepted = [frame for polarity in (1, -1)
                if (frame := decode_scores(fields, iterations=8, polarity=polarity)) is not None]
    return accepted[0] if len(accepted) == 1 else None


def decode_scores(fields, *, iterations=8, polarity=1):
    fields = np.asarray(fields, dtype=float)
    if fields.shape != (3 * SYSTEMATIC_BITS,) or not np.all(np.isfinite(fields)):
        raise ValueError("normal long requires 4968 finite scores")
    if polarity not in (1, -1):
        raise ValueError("polarity must be +1 or -1")
    systematic, c1, c2 = np.split(fields * polarity, 3)
    c1[-8:] = c2[-8:] = 0
    decoded = p4decode.turbo_decode(systematic, c1, c2,
                                   component_permutation(), iterations=iterations)
    wire = np.packbits(decoded.bits[:8 * NET_BYTES], bitorder="little").tobytes()
    return p4frame.decode_field(wire, data_bytes=NET_BYTES - 3)
