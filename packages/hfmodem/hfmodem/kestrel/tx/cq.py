# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""CQ encoding and 48 kHz synthesis."""
import numpy as np

from ..coding import turbo
from ..rx.cq import (CQ, HEADER, TRAINING, ALLOCATION, REFERENCES, DATA,
                     REFERENCE_BINS, GRAY, tables, build_frame)
from ..vara.vara_mfsk import synth_tones


BODY_LEVEL = .270 / .353


def column_bins(frame: bytes) -> np.ndarray:
    if len(frame) != 8:
        raise ValueError('CQ frame must contain eight bytes')
    place, perm, pn = tables()
    bits = np.unpackbits(np.frombuffer(frame, np.uint8)) ^ pn
    values = turbo.encode_r13(bits, perm)[place].reshape(-1, 3) @ np.array([4, 2, 1])
    bins = np.empty(CQ.columns, int)
    bins[REFERENCES] = REFERENCE_BINS
    bins[DATA] = 54 + (ALLOCATION[DATA] - 54 + 2 * GRAY[values]) % 21
    return bins


def synth_frame(frame: bytes, *, amplitude: float = 0.5, gap: float = .032) -> np.ndarray:
    header = synth_tones(HEADER, amplitude=amplitude)
    body_bins = np.concatenate((TRAINING, column_bins(frame)))
    n = np.arange(CQ.symbol_samples)
    body = amplitude * BODY_LEVEL * np.sin(2 * np.pi * body_bins[:, None] * n / CQ.symbol_samples)
    gap_samples = round(gap * CQ.sample_rate) - (len(header) - len(HEADER) * CQ.symbol_samples)
    if gap_samples < 0:
        raise ValueError('gap shorter than header shaping tail')
    return np.concatenate((header, np.zeros(gap_samples), body.ravel()))


def synth_cq(call: str, bandwidth: int, **kwargs) -> np.ndarray:
    return synth_frame(build_frame(call, bandwidth), **kwargs)
