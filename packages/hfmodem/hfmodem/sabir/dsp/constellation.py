# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Generic Gray-coded constellations for the OFDM engine.

This module is deliberately *parameterized and generic*, carrying no protocol of
its own. It provides the small family of linear modulations an OFDM system draws
from (BPSK / QPSK / square QAM), each Gray-coded and normalised to unit average
energy so that Eb/N0 accounting downstream is clean.

A ``Constellation`` maps groups of ``bits_per_symbol`` bits to complex points and
back. Demapping is nearest-neighbour, which is the ML decision on an AWGN
channel and works for *any* point set -- so an arbitrary custom alphabet can be
dropped in here without touching the OFDM engine.

The consistency guarantee: modulation and demodulation share one table
``points[i]``, where ``i`` is the integer formed by the symbol's bits (MSB
first). Round-trip is therefore exact by construction, independent of how the
Gray mapping is defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _gray_to_binary(g: int) -> int:
    """Decode a Gray-coded integer to its natural binary value."""
    b = g
    g >>= 1
    while g:
        b ^= g
        g >>= 1
    return b


def _square_qam_points(bits_per_symbol: int) -> np.ndarray:
    """Build a Gray-coded square QAM point table, indexed by raw bit integer.

    ``bits_per_symbol`` must be even (square constellation). The first half of
    the bits drive the in-phase axis, the second half the quadrature axis; each
    axis is an independently Gray-coded PAM. The whole set is normalised to unit
    average energy.
    """
    if bits_per_symbol % 2 != 0:
        raise ValueError("square QAM needs an even number of bits per symbol")
    bits_per_axis = bits_per_symbol // 2
    side = 1 << bits_per_axis  # points per axis
    m = 1 << bits_per_symbol

    def axis_amplitude(axis_bits: int) -> float:
        level = _gray_to_binary(axis_bits)          # 0 .. side-1
        return 2 * level - (side - 1)               # centred odd-integer PAM

    points = np.empty(m, dtype=np.complex128)
    axis_mask = side - 1
    for i in range(m):
        i_bits = (i >> bits_per_axis) & axis_mask
        q_bits = i & axis_mask
        points[i] = axis_amplitude(i_bits) + 1j * axis_amplitude(q_bits)

    points /= np.sqrt(np.mean(np.abs(points) ** 2))  # unit average energy
    return points


@dataclass
class Constellation:
    """A Gray-coded complex constellation with unit average symbol energy."""

    name: str
    bits_per_symbol: int
    points: np.ndarray  # shape (2**bits_per_symbol,), unit average energy

    # -- construction ---------------------------------------------------------
    @classmethod
    def create(cls, name: str) -> "Constellation":
        name = name.strip().lower()
        if name == "bpsk":
            pts = np.array([-1.0, 1.0], dtype=np.complex128)
            return cls("bpsk", 1, pts)
        if name == "qpsk":
            # Gray: bit pattern i = (i_bit<<1 | q_bit); axes independent.
            pts = np.array(
                [(-1 - 1j), (-1 + 1j), (1 - 1j), (1 + 1j)], dtype=np.complex128
            ) / np.sqrt(2)
            return cls("qpsk", 2, pts)
        if name in ("16qam", "qam16"):
            return cls("16qam", 4, _square_qam_points(4))
        if name in ("64qam", "qam64"):
            return cls("64qam", 6, _square_qam_points(6))
        if name in ("256qam", "qam256"):
            return cls("256qam", 8, _square_qam_points(8))
        raise ValueError(f"unknown constellation {name!r}")

    @property
    def order(self) -> int:
        return 1 << self.bits_per_symbol

    # -- mapping --------------------------------------------------------------
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Map a 1-D array of 0/1 bits to complex symbols.

        The number of bits must be a multiple of ``bits_per_symbol``.
        """
        bits = np.asarray(bits, dtype=np.int64).ravel()
        k = self.bits_per_symbol
        if bits.size % k:
            raise ValueError(f"bit count {bits.size} not a multiple of {k}")
        groups = bits.reshape(-1, k)
        weights = 1 << np.arange(k - 1, -1, -1)      # MSB first
        idx = groups @ weights
        return self.points[idx]

    def llr(self, symbols: np.ndarray, weight: np.ndarray | float = 1.0) -> np.ndarray:
        """Max-log soft demap: equalised symbols -> per-bit LLRs, LLR>0 <=> bit 0.

        ``weight`` is the per-symbol channel-state information |H|^2 / sigma^2
        (broadcastable to ``symbols``). It matters because a zero-forcing
        equaliser divides by H and colorizes the noise: after ``z = y/H`` the
        effective noise variance on ``z`` is sigma^2/|H|^2, so the LLR magnitude
        must scale with |H|^2/sigma^2 or a fading channel feeds the decoder
        over-confident garbage from faded carriers.
        """
        z = np.asarray(symbols, dtype=np.complex128).ravel()
        w = np.broadcast_to(np.asarray(weight, dtype=np.float64), z.shape)
        if self.name == "bpsk":
            return -4.0 * w * z.real
        if self.name == "qpsk":
            # Gray QPSK is two independent BPSKs at amplitude 1/sqrt(2); the
            # MSB rides the real axis, the LSB the imaginary axis.
            out = np.empty(2 * z.size)
            out[0::2] = -2.0 * np.sqrt(2.0) * w * z.real
            out[1::2] = -2.0 * np.sqrt(2.0) * w * z.imag
            return out
        # generic max-log over the point table: for each bit position, the
        # distance gap between the nearest bit-0 and nearest bit-1 point.
        # Reduces exactly to the closed forms above for BPSK/QPSK.
        k = self.bits_per_symbol
        d = np.abs(z[:, None] - self.points[None, :]) ** 2
        idx = np.arange(self.order)
        out = np.empty((z.size, k))
        for b in range(k):
            one = (idx >> (k - 1 - b)) & 1 == 1
            out[:, b] = d[:, one].min(axis=1) - d[:, ~one].min(axis=1)
        return (w[:, None] * out).ravel()

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Nearest-neighbour hard decision: complex symbols -> 0/1 bits."""
        symbols = np.asarray(symbols, dtype=np.complex128).ravel()
        # distance to every constellation point, argmin per received symbol
        d = np.abs(symbols[:, None] - self.points[None, :])
        idx = np.argmin(d, axis=1)
        k = self.bits_per_symbol
        shifts = np.arange(k - 1, -1, -1)
        return ((idx[:, None] >> shifts) & 1).astype(np.int64).ravel()
