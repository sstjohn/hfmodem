# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Located Y27 changeover body: four 210-symbol BPSK blocks and five trainings.

The first training is 432 chips after the word origin. The five-tap MMSE
channel fit follows p4long; only a unique CRC-valid decoding is returned.
"""

from __future__ import annotations

import numpy as np

from . import p4c2, p4decode, p4frame, p4normal, p4rx, tablegen

LAYOUT = p4normal.layout("breakin", terminal_training=True)
NET_BYTES = 34
SYSTEMATIC_BITS = 280
BODY_OFFSET_CHIPS = 432


def component_permutation() -> np.ndarray:
    """Exact 280-entry direct-gather order, returned as a fresh array."""
    return p4c2.permutation(280).copy()


def body_scores(
    baseband, *, first_training_start: int, frequency_offset: float = 0.0
) -> np.ndarray | None:
    """The 840 coded-bit LLRs a located body carries, in figure 7.2's order.

    The start is the training's impulse coordinate before transmit shaping, not
    the matched-filter peak. One five-tap channel is fitted across all
    five trainings and inverted under its own residual variance. The
    scale is the fit's, not a calibrated one. Incomplete or silent input
    returns None; malformed input raises ValueError.
    """
    z = np.asarray(baseband, dtype=complex)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("baseband must be a finite one-dimensional vector")
    if type(first_training_start) is not int or first_training_start < 0:
        raise ValueError("first_training_start must be a nonnegative integer")
    if (
        not np.isscalar(frequency_offset)
        or not np.isreal(frequency_offset)
        or not np.isfinite(frequency_offset)
    ):
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
    # The recorded body carrier differs slightly from its word's carrier.
    # Refine on the five known trainings before fitting one common channel;
    # otherwise that residual rotation is mistaken for channel noise.
    residuals = np.arange(-3.0, 3.01, 0.5)
    expected = training[:, targets].ravel()
    samples = matched[positions].ravel()
    rotations = np.exp(-2j * np.pi * positions.ravel()[:, None] * residuals / 28800)
    coherent = abs((samples * expected.conj()) @ rotations)
    residual = residuals[int(coherent.argmax())]
    matched *= np.exp(-2j * np.pi * residual * np.arange(len(matched)) / 28800)
    observed = matched[positions].ravel()
    if not np.any(observed):
        return None
    matrix = training[:, targets[:, None] + np.arange(-2, 3)].reshape(-1, 5)
    gram = matrix.conj().T @ matrix
    channel = np.linalg.solve(
        gram + 1e-6 * np.trace(gram).real / 5 * np.eye(5), matrix.conj().T @ observed
    )
    variance = float(np.mean(abs(observed - matrix @ channel) ** 2))
    convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
    target = np.zeros(15)
    target[7] = 1
    if variance == 0:
        inverse = np.linalg.lstsq(convolution, target, rcond=None)[0]
    else:
        inverse = np.linalg.solve(
            convolution.conj().T @ convolution + variance * np.eye(11),
            convolution.conj().T @ target,
        )
    residual = np.convolve(channel, inverse)
    gain = residual[7]
    residual[7] = 0
    noise = float(
        variance * np.vdot(inverse, inverse).real + np.vdot(residual, residual).real
    )
    if noise <= 0 or not np.isfinite(noise):
        return None
    body = (
        spacing * np.arange(LAYOUT.blocks)[:, None]
        + 32
        + np.arange(LAYOUT.block_symbols)
    ).ravel()
    windows = start + 128 + 16 * (body[:, None] + np.arange(-5, 6))
    scores = 4 * (gain.conjugate() * (matched[windows] @ inverse)).real / noise
    return scores[tablegen.INTERLEAVERS["0xba890_840"]()]


def decode_body(
    baseband, *, first_training_start: int, frequency_offset: float = 0.0
) -> p4frame.Frame | None:
    """Decode complex 28800-Hz baseband, accepting only a unique CRC pass.

    Incomplete/silent input or CRC failure returns None; malformed input raises
    ValueError. No timing search or implicit packet classification; the carrier
    is refined within +/-3 Hz on the five known trainings.
    Constituent states are free and unresolved parity tails are erased.
    """
    fields = body_scores(
        baseband,
        first_training_start=first_training_start,
        frequency_offset=frequency_offset,
    )
    if fields is None:
        return None
    return decode_scores(fields)


def decode_scores(fields):
    """Accept exactly one polarity's CRC, exploiting the free-state symmetry.

    Complementing input and all three state bits complements both RSC output
    bits. With uniform initial/final states and erased parity tails, BCJR on
    negated observations is therefore the negated posterior. The interleaver
    preserves that symmetry, so both polarity candidates need one turbo run,
    not two. A posterior tie remains bit zero in either candidate.
    """
    fields = np.asarray(fields, dtype=float)
    if fields.shape != (840,) or not np.all(np.isfinite(fields)):
        raise ValueError("expected 840 finite decoder-order LLRs")
    systematic, c1, c2 = np.split(fields.copy(), 3)
    c1[-8:] = c2[-8:] = 0
    decoded = p4decode.turbo_decode(systematic, c1, c2, component_permutation())
    accepted = []
    for polarity in (1, -1):
        bits = (decoded.posterior * polarity < 0).astype(np.uint8)
        wire = np.packbits(bits[: 8 * NET_BYTES], bitorder="little").tobytes()
        frame = p4frame.decode_field(wire, data_bytes=NET_BYTES - 3)
        if frame is not None:
            accepted.append(frame)
    return accepted[0] if len(accepted) == 1 else None



def search_scores(z, nominal, cfo, span=48, step=2, *, deadline=None):
    from time import monotonic

    best = None
    for offset in range(-span, span + 1, step):
        if deadline is not None and monotonic() >= deadline:
            return None
        if nominal + offset < 0:
            continue
        llr = body_scores(z, first_training_start=int(nominal + offset),
                          frequency_offset=cfo)
        if llr is None:
            continue
        q = float(np.mean(abs(llr)))
        if best is None or q > best[1]:
            best = offset, q, llr
    if deadline is not None and monotonic() >= deadline:
        return None
    return best


def distance(llr, codeword):
    return float(np.mean((np.asarray(llr) < 0) != np.asarray(codeword)))
