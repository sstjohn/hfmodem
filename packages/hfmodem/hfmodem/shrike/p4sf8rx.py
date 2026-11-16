# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Placed SF8 SL4 robust-long field reader."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import p4frame, p4longdecode, p4robustrx

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


def build_members(symbols, *, amplitude=True, stride=29, turns=range(4)):
    """Freeze SL4's turn/weight domain, ranked by period-ten C1 syndrome."""
    raw = np.asarray(symbols, dtype=complex)
    allowed = tuple(turns)
    if (raw.ndim != 1 or not np.all(np.isfinite(raw)) or stride not in (26, 29)
            or len(set(allowed)) != len(allowed)
            or any(type(turn) is not int or turn not in range(4) for turn in allowed)):
        raise ValueError('finite symbols, supported stride and quarter-turns required')
    if len(raw) != SYMBOLS + 1 or not np.any(abs(raw)):
        return ()
    differential = raw[1:] * raw[:-1].conjugate()
    unit = dict(_scores(differential, amplitude=False, stride=stride))
    weighted = dict(_scores(differential, amplitude=True, stride=stride)) if amplitude else {}
    pairs = []
    for turn in allowed:
        pair = [p4longdecode.LongMember(p4longdecode.geometry(4), unit[turn],
                                       quarter_turn=turn, weighting='unit')]
        if amplitude:
            pair.append(p4longdecode.LongMember(p4longdecode.geometry(4), weighted[turn],
                                               quarter_turn=turn, weighting='amplitude'))
        pairs.append((p4robustrx.syndrome(unit[turn], 4), pair))
    pairs.sort(key=lambda row: row[0], reverse=True)
    return tuple(member for _, pair in pairs for member in pair)


def decode_symbols(symbols, *, c1=True, amplitude=True, stride=29,
                   turns=range(4)):
    members = build_members(symbols, amplitude=amplitude, stride=stride, turns=turns)
    result = p4longdecode.LongSearch(members, iterations=8 if c1 else 0).advance()
    chosen = result.selected_provenance
    if result.frame is None or chosen is None:
        return None
    member = members[chosen.member_index]
    correction = ('systematic' if chosen.method == 'systematic' else
                  f'{member.weighting}-c1' if chosen.method == 'c1-only' else
                  f'{member.weighting}-turbo')
    return Reception(result.frame, member.fields, member.quarter_turn, correction)
