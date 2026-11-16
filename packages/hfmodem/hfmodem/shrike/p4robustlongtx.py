# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PACTOR-4 robust LONG transmitter: SL2/3 SF16, SL4 SF8.

PUBLISHED: spreading, chip rate, pulse, RSC components and puncturing.
DERIVED: field sizes, CRC/whitening and LSB-first packing from the readers.
VERIFIED against stock: termination, C2 permutations, serialisation, symbol
walk and Gray DQPSK labels. The reference anchors the body at phase zero.

The four measured SF16 headers are circular means over the
203 CRC-valid stock copies,
admitting only copies whose every header symbol reaches 0.6 of the body's
median amplitude. A first copy, and a recut (counter kept, level or data
changed, 3 of 3 seen), carries its level's header; every repeat carries the
one level-blind RQ header. Fitted with each station left out, the literals
still put all 203 copies in their own class (margin >= 0.22 in whole-header
correlation); the copies that miss the 0.6 correlation floor are faded or
weak ones. No fifth class or second mode appeared. How the headers are
generated is unknown.

An empty payload is all IDLE.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from . import p4c2, p4coding, p4frame, p4robusttx, p4rx, p4sig, spec


@dataclass(frozen=True)
class Geometry:
    data_bytes: int
    systematic: int
    padding: int
    rate: str
    data_symbols: int
    spread: int


GEOMETRY = {
    2: Geometry(40, 348, 1, '1/2', 348, 16),
    3: Geometry(69, 580, 1, '5/6', 348, 16),
    4: Geometry(141, 1160, 5, '5/6', 696, 8),
}
HEADER_STEPS = {
    # 138 admitted copies, 22 packets, 6 stations; steps 1–18 mean absolute circular deviation <= 6.4 deg.
    'RQ': (1, -170, -153, 57, 95, -37, 18, -95, -19, -114,
           -19, -95, 19, -37, 94, 58, -151, -173, 0),
    # 4/8 admitted/CRC copies, 4/8 packets, 3/5 stations; mean absolute circular deviation <= 8.3 deg.
    2: (7, 101, -16, -39, 58, -92, -134, -60, 135, 76,
        127, -51, -136, -95, 57, -42, -16, 115, 1),
    # 15/18 admitted/CRC copies, 15/18 packets, 5/6 stations; mean absolute circular deviation <= 8.4 deg.
    3: (0, -75, 130, -95, -35, -61, -151, 37, 154, -170,
        149, 43, -156, -55, -37, -96, 135, -79, -1),
    # 2/3 admitted/CRC copies, 2/3 packets, VE3KPG/K4SPB; mean absolute circular deviation <= 7.9 deg.
    4: (-2, -56, -177, 22, 155, -139, -110, -147, 107, -32,
        115, -154, -112, -135, 149, 19, -165, -56, -1),
}
REFERENCE_STEP = 0
# Measured header-to-reference circular means: RQ/SL2/SL3/SL4 = 2/2/-2/6 deg.
INITIAL_STATE = (0, 0, 0)
EMISSION_S = {
    level: (320 + (g.data_symbols + 1)*g.spread - 1)/p4rx.CHIP_RATE
           + (p4rx.RRC_TAPS.size - 1)/p4rx.SAMPLE_RATE
    for level, g in GEOMETRY.items()
}
# One gain for every packet, so a repeat's body samples equal its first copy's:
# peak 0.9 at the unscaled maximum 1.2410 over seeds 0-63, SL2/3/4, all four
# headers, 48 and 8 kHz.
_SCALE = .9 / 1.2410


def _geometry(level):
    if (isinstance(level, (bool, np.bool_)) or not isinstance(level, Integral)
            or level not in GEOMETRY):
        raise ValueError('level must be 2, 3 or 4')
    return GEOMETRY[level]


def _walk(n, stride):
    return np.concatenate([np.arange(j, n, stride) for j in range(stride)])


def coded_bits(field: bytes, level) -> np.ndarray:
    """Whitened field to systematic || kept C1 || kept C2, including the tail."""
    g = _geometry(level)
    if len(field) != g.data_bytes + 3:
        raise ValueError(f'field must contain {g.data_bytes + 3} bytes')
    bits = np.unpackbits(np.frombuffer(field, np.uint8), bitorder='little')
    padded = np.concatenate((bits, np.zeros(g.padding, np.uint8)))
    tail = p4coding.rsc_terminate(
        p4coding.rsc_encode(padded, initial_state=INITIAL_STATE).final_state)
    systematic = np.concatenate((padded, np.array(tail.systematic, np.uint8)))
    permutation = _walk(580, 24) if level == 3 else p4c2.permutation(g.systematic)
    c1 = p4coding.rsc_encode(systematic, initial_state=INITIAL_STATE).parity
    c2 = p4coding.rsc_encode(systematic[permutation], initial_state=INITIAL_STATE).parity
    return np.array(p4coding.puncture(systematic, c1, c2, rate=g.rate).serialized,
                    np.uint8)


def body_symbols(coded, level) -> np.ndarray:
    """Phase-zero reference and interleaved Gray DQPSK data symbols."""
    g = _geometry(level)
    bits = np.asarray(coded)
    if bits.shape != (2*g.data_symbols,) or not np.all((bits == 0) | (bits == 1)):
        raise ValueError(f'coded must contain {2*g.data_symbols} bits')
    pairs = bits.astype(np.uint8).reshape(-1, 2)
    steps = np.empty(g.data_symbols, np.int64)
    steps[_walk(g.data_symbols, 29)] = np.array([0, 1, -1, 2])[2*pairs[:, 0]+pairs[:, 1]]
    # Exact quarter turns avoid cumulative floating point drift.
    return np.array([1, 1j, -1, -1j])[np.r_[0, np.cumsum(steps)] % 4]


def header_symbols(header) -> np.ndarray:
    """Twenty measured symbols running backwards from the phase-zero reference."""
    if header not in HEADER_STEPS:
        raise ValueError("header must be 'RQ', 2, 3 or 4")
    phases = np.r_[0, np.cumsum(HEADER_STEPS[header])]
    return np.exp(1j*np.radians(phases - phases[-1] - REFERENCE_STEP))


def robust_long(payload: bytes, status: int, level: int, *, repeat: bool = False,
                header=None, fs: int = spec.SAMPLE_RATE) -> np.ndarray:
    """One robust long as audio. `header` overrides the evidence default, which
    is 'RQ' for a repeat and the level's own header otherwise."""
    g = _geometry(level)
    if len(payload) > g.data_bytes:
        raise ValueError(f'payload exceeds {g.data_bytes} bytes at level {level}')
    if isinstance(fs, bool) or not isinstance(fs, Integral) or fs <= 0:
        raise ValueError('fs must be a positive integer')
    field = p4frame.encode_field(payload + bytes([spec.IDLE])*(g.data_bytes-len(payload)),
                                p4frame.Status.from_byte(status))
    head = header_symbols(('RQ' if repeat else level) if header is None else header)
    body = body_symbols(coded_bits(field, level), level)
    spread = p4rx.SPREAD8 if g.spread == 8 else p4sig.SPREAD16
    chips = np.concatenate(((head[:, None]*p4sig.SPREAD16).ravel(),
                            (body[:, None]*spread).ravel()))
    # Resample_poly rounds the input sample count, including its last endpoint.
    # Keep the requested-rate samples through the physical pulse endpoint.
    return (_SCALE * p4robusttx.shape_chips(chips, fs))[:round(EMISSION_S[level]*fs)+1]
