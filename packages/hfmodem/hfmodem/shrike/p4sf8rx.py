# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Placed SF8 SL4 robust-long field reader."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import p4decode, p4frame

SYMBOLS = 696
SYSTEMATIC = 1160
PARITY = 116
ORDER = np.concatenate([np.arange(j, SYMBOLS, 29) for j in range(29)])


@dataclass(frozen=True)
class Reception:
    frame: p4frame.Frame
    fields: np.ndarray
    quarter_turn: int
    correction: str


def _frame(bits):
    field = np.packbits(bits[:1152], bitorder='little').tobytes()
    return p4frame.decode_field(field, data_bytes=141)


def _scores(differential, *, amplitude, stride):
    unit = differential / np.maximum(abs(differential), 1e-30)
    phase = np.exp(-1j * np.angle(np.mean(unit**4)) / 4)
    order = ORDER if stride == 29 else np.concatenate(
        [np.arange(j, SYMBOLS, stride) for j in range(stride)])
    data = unit[order] * phase
    if amplitude:
        data *= abs(differential[order]) / max(np.mean(abs(differential)), 1e-30)
    scale = 8 if amplitude else 4
    for turn in range(4):
        rotated = data * (1j**turn)
        yield turn, scale * np.stack((rotated.real + rotated.imag,
                                      rotated.real - rotated.imag), axis=-1).ravel()


def decode_symbols(symbols, *, c1=True, amplitude=True, stride=29,
                   turns=range(4)):
    raw = np.asarray(symbols, dtype=complex)
    allowed = tuple(turns)
    if (raw.ndim != 1 or not np.all(np.isfinite(raw)) or stride not in (26, 29)
            or len(set(allowed)) != len(allowed)
            or any(type(turn) is not int or turn not in range(4) for turn in allowed)):
        raise ValueError('finite symbols, supported stride and quarter-turns required')
    if len(raw) != SYMBOLS + 1 or not np.any(abs(raw)):
        return None
    differential = raw[1:] * raw[:-1].conjugate()
    candidates = []
    for turn, fields in _scores(differential, amplitude=False, stride=stride):
        if turn not in allowed:
            continue
        frame = _frame(fields < 0)
        if frame is not None:
            candidates.append(Reception(frame, fields, turn, 'systematic'))
    if len(candidates) == 1 or not c1:
        return candidates[0] if len(candidates) == 1 else None
    for weighted in (False, True) if amplitude else (False,):
        for turn, fields in _scores(differential, amplitude=weighted, stride=stride):
            if turn not in allowed:
                continue
            parity = np.zeros(SYSTEMATIC)
            parity[::10] = fields[SYSTEMATIC:SYSTEMATIC + PARITY]
            result = p4decode.component_decode(fields[:SYSTEMATIC], parity)
            frame = _frame(result.posterior < 0)
            if frame is not None:
                candidates.append(Reception(frame, fields, turn,
                                            'amplitude-c1' if weighted else 'unit-c1'))
    unique = {(row.frame.data, row.frame.status.byte, row.frame.crc)
              for row in candidates}
    return candidates[0] if len(unique) == 1 else None
