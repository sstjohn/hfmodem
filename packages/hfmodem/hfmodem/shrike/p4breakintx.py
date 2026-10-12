# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Measured Y27-headed PACTOR-4 changeover packet, with idle-padded data."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import (
    p4coding,
    p4frame,
    p4normal,
    p4rx,
    p4robusttx,
    p4sig,
    p4words,
    spec,
    tablegen,
)
from .p4breakin import LAYOUT, NET_BYTES, SYSTEMATIC_BITS, component_permutation

PAD_BITS = SYSTEMATIC_BITS - 8 * NET_BYTES - 3
INITIAL_STATE = (0, 0, 0)
WORD_CHIPS = p4robusttx.CS_SYMBOLS * p4sig.SPREAD_FACTOR
EMISSION_CHIPS = WORD_CHIPS + LAYOUT.total_symbols
EMISSION_S = (EMISSION_CHIPS - 1) / p4rx.CHIP_RATE + p4robusttx.PULSE_S
WORD_GAIN = 10 ** (1.3 / 20)
_SYMBOL_ORDER = tablegen.INTERLEAVERS["0xba890_840"]()


def coded_bits(field: bytes) -> np.ndarray:
    """A whitened 34-byte field as 840 coded bits, V0 then C1 then C2 (7.2)."""
    if len(field) != NET_BYTES:
        raise ValueError("field must contain exactly 34 bytes")
    bits = np.unpackbits(np.frombuffer(field, np.uint8), bitorder="little")
    padded = np.concatenate([bits, np.zeros(PAD_BITS, np.uint8)])
    tail = p4coding.rsc_terminate(
        p4coding.rsc_encode(padded, initial_state=INITIAL_STATE).final_state
    )
    systematic = np.concatenate([padded, np.array(tail.systematic, np.uint8)])
    order = component_permutation()
    return np.concatenate(
        [
            systematic,
            p4coding.rsc_encode(systematic, initial_state=INITIAL_STATE).parity,
            p4coding.rsc_encode(systematic[order], initial_state=INITIAL_STATE).parity,
        ]
    ).astype(np.uint8)


def body_symbols(coded: np.ndarray) -> np.ndarray:
    """840 coded bits interleaved onto the four training-and-data blocks."""
    data = np.empty(coded.size, complex)
    data[_SYMBOL_ORDER] = 1 - 2.0 * coded
    out = np.empty(LAYOUT.total_symbols, complex)
    for block, offset in enumerate(LAYOUT.training_offsets):
        out[offset : offset + 32] = p4normal.training_sequence(block + 1)
    for block, offset in enumerate(LAYOUT.data_offsets):
        out[offset : offset + LAYOUT.block_symbols] = data[
            block * LAYOUT.block_symbols : (block + 1) * LAYOUT.block_symbols
        ]
    return out


def channel_bits(field: bytes) -> np.ndarray:
    """The 840 transmitted bits, in air order (not decoder order)."""
    out = np.empty(3 * SYSTEMATIC_BITS, np.uint8)
    out[_SYMBOL_ORDER] = coded_bits(field)
    return out


def changeover_packet(
    payload: bytes, status: int, word: str = "Y27", fs: int = spec.SAMPLE_RATE,
    *, word_render="off"
) -> np.ndarray:
    if word not in p4words.BANK or len(p4words.BANK[word]) != 27:
        raise ValueError("head must be a named 27-symbol word")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    data = payload[:31].ljust(31, bytes([spec.IDLE]))
    field = p4frame.encode_field(data, p4frame.Status.from_byte(status))
    word_chips = (p4words.BANK[word][:, None] * p4sig.SPREAD16).ravel() * WORD_GAIN
    p4robusttx.carrier_hz(word, word_render, word_chips)
    chips = np.concatenate([word_chips, body_symbols(coded_bits(field))])
    impulses = np.zeros(p4rx.SAMPLES_PER_CHIP * (chips.size - 1) + 1, complex)
    impulses[:: p4rx.SAMPLES_PER_CHIP] = chips
    g = gcd(int(fs), p4rx.SAMPLE_RATE)
    bb = resample_poly(
        np.convolve(impulses, p4rx.RRC_TAPS), int(fs) // g, p4rx.SAMPLE_RATE // g
    )
    out = (bb * np.exp(2j * np.pi * p4rx.CENTRE_HZ * np.arange(bb.size) / fs)).real
    return out / np.max(np.abs(out))
