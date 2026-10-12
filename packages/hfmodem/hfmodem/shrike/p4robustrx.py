# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Located SF16 robust-long fields; measured symbol order, published coding.

This reader accepts SL2 via systematic CRC only. The exact C2 permutation is
available through p4c2 and the separate placed p4robustc2 bench decoder.
SL3 uses the published rate5/6 puncture and measured 580/24 component walk.
No acquisition, header meaning, ACK policy or host delivery is performed here.
Four recorded frames pin the measured symbol and component ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import p4decode, p4frame

SYMBOLS = 348
MIN_SYNDROME = .80


def _walk(n, stride):
    return np.concatenate([np.arange(j, n, stride) for j in range(stride)])


_SYMBOL_ORDER = _walk(348, 29)
_COMPONENT_ORDER = _walk(580, 24)
_CHECKS = {
    2: (348, 2, (0, 2, 3), tuple(range(7))),
    3: (580, 10, (0, 1, 3), (0, 1, 2, 3, 6, 8, 9, 11, 12, 15,
                            17, 18, 19, 22, 24, 25, 26, 29, 30)),
}


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


def _frame(bits, speedlevel):
    length = 43 if speedlevel == 2 else 72
    wire = np.packbits(bits[:8*length], bitorder='little').tobytes()
    return p4frame.decode_field(wire, data_bytes=length-3)


def decode_symbols(symbols, *, starts=range(18, 25), iterations=2):
    """Read complete placed symbols, choosing a layout by C1 before CRC.

    Starts are indices of the first data differential; nominally20. At most
    seven placements and four carrier quarter-turns are admitted. Only equally
    best C1 layouts above the measured null bound can enter CRC/turbo decoding.
    ``fields`` retains the696 signed soft bits for later offline combining.
    """
    raw = np.asarray(symbols, dtype=complex)
    starts = tuple(starts)
    if (raw.ndim != 1 or not np.all(np.isfinite(raw))
            or any(type(k) is not int or not 18 <= k <= 24 for k in starts)
            or len(set(starts)) != len(starts)
            or type(iterations) is not int or not 0 <= iterations <= 2):
        raise ValueError('finite symbols, unique starts18..24, and zero to two iterations required')
    if len(raw) < 367 or not np.any(abs(raw)):
        return None
    delta = raw[1:] * raw[:-1].conjugate()
    delta /= np.maximum(abs(delta), 1e-30)
    delta *= np.exp(-1j*np.angle(np.mean(delta**4))/4)
    candidates = []
    for start in starts:
        if start+SYMBOLS > len(delta):
            continue
        data = delta[start:start+SYMBOLS][_SYMBOL_ORDER]
        for turn in range(4):
            rotated = data * (1j**turn)
            fields = 4*np.stack([rotated.real+rotated.imag,
                                 rotated.real-rotated.imag], axis=-1).ravel()
            for speedlevel in (2, 3):
                score = syndrome(fields, speedlevel)
                if score >= MIN_SYNDROME:
                    candidates.append((score, speedlevel, start, fields))
    if not candidates:
        return None
    best = max(row[0] for row in candidates)
    candidates = [row for row in candidates if row[0] == best]
    if len({(level, start) for _, level, start, _ in candidates}) != 1:
        return None
    accepted = []
    for score, level, start, fields in candidates:
        bits = (fields < 0).astype(np.uint8)
        frame = _frame(bits, level)
        if frame is not None:
            accepted.append(Reception(frame, level, start, fields, score, 0))
    if accepted:
        return accepted[0] if len(accepted) == 1 else None
    if not iterations:
        return None
    soft_cache = []
    for score, level, start, fields in candidates:
        if level != 3:
            continue
        decoded = None
        for old_fields, old_posterior in soft_cache:
            if np.allclose(fields, -old_fields, atol=1e-12, rtol=0):
                decoded = (-old_posterior < 0).astype(np.uint8)
                break
        if decoded is None:
            c1, c2 = np.zeros(580), np.zeros(580)
            c1[::10], c2[4::10] = fields[580:638], fields[638:]
            c1[-4:] = c2[-4:] = 0
            result = p4decode.turbo_decode(fields[:580], c1, c2,
                                           _COMPONENT_ORDER, iterations=iterations)
            decoded = result.bits
            soft_cache.append((fields, result.posterior))
        frame = _frame(decoded, level)
        if frame is not None:
            corrected = int(np.count_nonzero(decoded != (fields[:580] < 0)))
            accepted.append(Reception(frame, level, start, fields, score, corrected))
    return accepted[0] if len(accepted) == 1 else None
