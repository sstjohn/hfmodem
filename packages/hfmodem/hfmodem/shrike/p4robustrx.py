# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Located SF16 robust-long observations for the shared exact LONG engine."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import p4frame, p4longdecode

SYMBOLS = 348


def _walk(n, stride):
    return np.concatenate([np.arange(j, n, stride) for j in range(stride)])


_SYMBOL_ORDER = _walk(348, 29)
_CHECKS = {
    2: (348, 2, (0, 2, 3), tuple(range(7))),
    3: (580, 10, (0, 1, 3), (0, 1, 2, 3, 6, 8, 9, 11, 12, 15,
                            17, 18, 19, 22, 24, 25, 26, 29, 30)),
}
_CHECKS[4] = (1160, *_CHECKS[3][1:])


@dataclass(frozen=True)
class Reception:
    frame: p4frame.Frame
    speedlevel: int
    field_start: int
    fields: np.ndarray
    syndrome_score: float
    corrected_bits: int

    @property
    def packet_symbol(self):
        return self.field_start - 20


def syndrome(fields, speedlevel):
    D, period, pc, sc = _CHECKS[speedlevel]
    bits = np.asarray(fields) < 0
    idx = np.arange(max(pc), D//period-4)
    expected = np.bitwise_xor.reduce(bits[idx[:, None]*period-np.array(sc)], axis=1)
    parity = bits[D:D+D//period]
    actual = np.bitwise_xor.reduce(parity[idx[:, None]-np.array(pc)], axis=1)
    return float(np.mean(expected == actual))


def build_members(symbols, *, starts=range(18, 25)):
    """Freeze every supported placement, turn, weighting and SL2/3 layout.

    C1 syndrome ranks work only. Each unit member is followed by its amplitude
    partner; no observation is excluded by a syndrome threshold.
    """
    raw = np.asarray(symbols, dtype=complex)
    starts = tuple(starts)
    if (raw.ndim != 1 or not np.all(np.isfinite(raw))
            or any(type(k) is not int or not 18 <= k <= 24 for k in starts)
            or len(set(starts)) != len(starts)):
        raise ValueError('finite symbols and unique starts18..24 required')
    if len(raw) < 367 or not np.any(abs(raw)):
        return ()
    differential = raw[1:] * raw[:-1].conjugate()
    unit = differential / np.maximum(abs(differential), 1e-30)
    phase = np.exp(-1j*np.angle(np.mean(unit**4))/4)
    pairs = []
    for start in starts:
        if start + SYMBOLS > len(differential):
            continue
        observed = differential[start:start + SYMBOLS]
        data = unit[start:start + SYMBOLS][_SYMBOL_ORDER] * phase
        weights = abs(observed[_SYMBOL_ORDER]) / max(np.mean(abs(observed)), 1e-30)
        for turn in range(4):
            rotated = data * (1j**turn)
            fields = 4*np.stack((rotated.real + rotated.imag,
                                 rotated.real - rotated.imag), axis=-1).ravel()
            amplitude = fields * np.repeat(weights, 2) * 2
            for level in (2, 3):
                pair = tuple(p4longdecode.LongMember(
                    p4longdecode.geometry(level), scores, quarter_turn=turn,
                    weighting=weighting, label=str(start))
                    for weighting, scores in (('unit', fields), ('amplitude', amplitude)))
                pairs.append((syndrome(fields, level), pair))
    pairs.sort(key=lambda row: row[0], reverse=True)
    return tuple(member for _, pair in pairs for member in pair)


def decode_symbols(symbols, *, starts=range(18, 25), iterations=2):
    """Read the complete placed SF16 domain through the exact LONG engine."""
    if type(iterations) is not int or not 0 <= iterations <= 2:
        raise ValueError('zero to two iterations required')
    members = build_members(symbols, starts=starts)
    result = p4longdecode.LongSearch(members, iterations=iterations).advance()
    chosen = result.selected_provenance
    if result.frame is None or chosen is None:
        return None
    member = members[chosen.member_index]
    corrected = chosen.agreements[0].hard_disagreements
    return Reception(result.frame, member.geometry.speedlevel, int(member.label),
                     member.fields, syndrome(member.fields, member.geometry.speedlevel), corrected)
