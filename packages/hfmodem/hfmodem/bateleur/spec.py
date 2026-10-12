# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tables of MIL-STD-188-110 serial and STANAG 4285, and where each is written.

The serial-110 tables were transcribed from 110A/B text and remain unverified
against a foreign implementation. The 4285 Annex A tables were cross-checked
against hcab14's published decoder; Annex E message framing was not. Local
round trips establish consistency, not interoperability. The unresolved
readings and recordings needed to test them are in docs/protocols/bateleur/README.md.

Symbols throughout are 8-PSK *indices*: ``k`` is the phase ``k * 45`` degrees
(110B 5.3.2.3.9, 4285 Annex A 2(c)), and every constellation smaller than eight
points is a subset of those indices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The one channel both standards were drawn for.
BAUD = 2400
CARRIER_HZ = 1800
# The rate the modem works at internally: four samples a symbol, and 48 kHz / 5.
FS = 9600
SPS = FS // BAUD


def psk8(indices) -> np.ndarray:
    """8-PSK indices -> unit complex symbols."""
    return np.exp(1j * np.pi / 4 * np.asarray(indices, dtype=np.float64))


#: The points a known symbol may be, by index: the eight of 8-PSK, then the
#: five fifth roots of unity (indices 8-12), which MIL-STD-188-110C appendix
#: D's 48-symbol mini-probes are made of. Rotations act on the first eight.
KNOWN_POINTS = np.concatenate([psk8(np.arange(8)), np.exp(2j * np.pi * np.arange(5) / 5)])


# --------------------------------------------------------------------------
# The code both standards use: 110B figure 4, 4285 Annex E figure E-1.
#
# K = 7, rate 1/2, T1(x) = x^6+x^4+x^3+x+1 and T2(x) = x^6+x^5+x^4+x^3+1, the
# input entering the x^6 stage. As masks over a shift register whose bit t
# holds the input delayed by t, those are 0x6D and 0x4F — 133 and 171 octal,
# as Annex E labels them, read from the other end. T1 is sent first.
# --------------------------------------------------------------------------
CONV_K = 7
CONV_MASKS = (0x6D, 0x4F)


# ==========================================================================
# MIL-STD-188-110 serial (single-tone), fixed-frequency operation.
# ==========================================================================

# The 32-symbol expansions of the preamble's channel symbols 0-7 (table XVII),
# which are also 75 bit/s's Walsh sets (table XIV a and b). 0-3 are
# four-periodic; 4-7 are their eight-periodic complements.
_CHANNEL_BASE = ((0, 0, 0, 0), (0, 4, 0, 4), (0, 0, 4, 4), (0, 4, 4, 0))


def _channel_pattern(t: int) -> tuple[int, ...]:
    base = _CHANNEL_BASE[t & 3]
    if t < 4:
        return base * 8
    flip = tuple((s + 4) % 8 for s in base)
    return (base + flip) * 4


CHANNEL_PATTERNS = tuple(_channel_pattern(t) for t in range(8))

# The preamble's fixed-pattern sync symbols, then D1 D2, then the count C1-C3,
# then a closing 0: fifteen channel symbols to a 0.2 s segment (5.3.2.3.7.2.1).
PREAMBLE_SYNC = (0, 1, 3, 0, 1, 3, 1, 2, 0)
PREAMBLE_BLOCK_SYMBOLS = 15 * 32

# Added mod 8 to every preamble symbol, repeating every 32, 7 first (5.3.2.3.8.2).
PREAMBLE_SCRAMBLE = (7, 4, 3, 0, 5, 1, 5, 0, 2, 2, 1, 1, 5, 7, 4, 3,
                     5, 0, 2, 6, 2, 1, 6, 2, 0, 0, 5, 0, 5, 2, 6, 6)

# The data-phase randomiser (5.3.2.3.8.1, figure 6): a 12-stage register,
# loaded with 1011 1010 1101 (BAD) at the start of the data phase and again
# after every 160 symbols, shifted eight times for each symbol before its three
# low stages are read as the tribit (MSB, middle, LSB) added mod 8. Figure 6
# feeds the top stage back into stage 0 and exclusive-ors it in ahead of
# stages 1, 4 and 6: x^12 + x^6 + x^4 + x + 1.
SCRAMBLE110_SEED = 0xBAD
SCRAMBLE110_PERIOD = 160


def scramble110() -> np.ndarray:
    """Implemented reading: seed 0xBAD, left shift with mask 0x053, eight
    shifts before reading the low three bits. Foreign interoperability of
    this orientation is unverified; a known-payload serial recording is needed.
    """
    reg = SCRAMBLE110_SEED
    out = np.empty(SCRAMBLE110_PERIOD, dtype=np.int64)
    for i in range(SCRAMBLE110_PERIOD):
        for _ in range(8):
            carry = (reg >> 11) & 1
            reg = (reg << 1) & 0xFFF
            if carry:
                reg ^= 0x053            # taps x^6, x^4, x^1, and x^0 takes the carry
        out[i] = reg & 7
    return out


# Tribit / dibit / bit -> 8-PSK index: the modified-Gray decoder (tables XII and
# XIII) and symbol formation (5.3.2.3.7.1.1, figure 5).
GRAY110_8 = (0, 1, 3, 2, 7, 6, 4, 5)
GRAY110_4 = (0, 2, 6, 4)
GRAY110_2 = (0, 4)

# End of message, most significant digit first (5.3.2.3.1).
EOM110 = 0x4B65A5B2
EOM110_BITS = tuple((EOM110 >> (31 - i)) & 1 for i in range(32))
# Zero bits after the EOM: 144 for the coder, then whatever completes the
# interleaver block (5.3.2.3.2).
FLUSH110_BITS = 144


def find_eom(bits, start: int = 0, *, flush: int = 64, errors: int = 3) -> int:
    """EOM at any bit offset, with up to ``errors`` wrong marker bits.

    The serial 110/4285 readers require 60 of the next 64 flush bits to be
    zero: a near marker in random payload alone is too common to end it.
    4539 has only block fill, possibly empty, and uses an exact marker.
    """
    b = np.asarray(bits, dtype=np.uint8)
    start = max(0, start)
    width = len(EOM110_BITS) + flush
    if b.size - start < width:
        return -1
    w = np.lib.stride_tricks.sliding_window_view(b[start:], width)
    match = np.count_nonzero(w[:, :32] != EOM110_BITS, axis=1) <= errors
    if flush:
        match &= np.count_nonzero(w[:, 32:], axis=1) <= flush // 16
    hit = np.flatnonzero(match)
    return start + int(hit[0]) if hit.size else -1


@dataclass(frozen=True)
class Mode110:
    """One 188-110 serial data rate at one interleave setting (110B 5.3.2)."""

    rate: int                   # user bits per second
    interleave: str             # "short" | "long" | "zero" | "none" (4800)
    bits_per_symbol: int        # coded bits per channel symbol
    repeat: int                 # times each coded bit pair is sent
    unknown: int                # data symbols per frame (75: one 32-symbol set)
    known: int                  # probe symbols per frame
    rows: int                   # interleaver matrix; 0 when bypassed
    cols: int
    d1: int
    d2: int
    row_step: int = 9           # interleave load: row advance (5.3.2.3.4)
    col_step: int = 17          # interleave fetch: column decrement (5.3.2.3.5)

    @property
    def coded(self) -> bool:
        return self.rate != 4800

    @property
    def walsh(self) -> bool:
        """75 bit/s: each dibit is one of four 32-symbol sets, no probes."""
        return self.rate == 75

    @property
    def preamble_blocks(self) -> int:
        """0.6 s of preamble, or 4.8 s ahead of the long interleaver."""
        return 24 if self.interleave == "long" else 3

    @property
    def block_bits(self) -> int:
        return self.rows * self.cols

    @property
    def frames_per_block(self) -> int:
        """Frames between D1/D2 reinsertions — an interleaver block's worth.
        With no interleaver the text uses the 0.6 s block's pattern, 1440
        symbols (5.3.2.3.7.1.2 NOTE); at 75 bit/s it is the set count before
        an exceptional set, 45 or 360 (5.3.2.3.7.1.1)."""
        if self.walsh:
            return 360 if self.interleave == "long" else 45
        if not self.rows:
            return 1440 // (self.unknown + self.known)
        return self.block_bits // (self.bits_per_symbol * self.unknown)

    @property
    def name(self) -> str:
        return f"{self.rate}{self.interleave[0].upper()}"


def _m110(rate, il, bps, rep, unk, kn, rows, cols, d1, d2, rs=9, cs=17):
    return Mode110(rate, il, bps, rep, unk, kn, rows, cols, d1, d2, rs, cs)


# Rates, frames, interleaver sizes and D1/D2: 110B tables XV, XIX and X, the
# same numbers as 110A. Zero interleave (Z) is the short setting with the
# interleaver bypassed; it carries the short D1/D2, so a receiver has to be
# told (5.3.2.3.7.2.1 NOTE). 2400 bit/s digital voice (7, 7) is not here.
MODES110 = {
    m.name: m for m in (
        _m110(4800, "none", 3, 1, 32, 16, 0, 0, 7, 6),
        _m110(2400, "short", 3, 1, 32, 16, 40, 72, 6, 4),
        _m110(2400, "long", 3, 1, 32, 16, 40, 576, 4, 4),
        _m110(2400, "zero", 3, 1, 32, 16, 0, 0, 6, 4),
        _m110(1200, "short", 2, 1, 20, 20, 40, 36, 6, 5),
        _m110(1200, "long", 2, 1, 20, 20, 40, 288, 4, 5),
        _m110(1200, "zero", 2, 1, 20, 20, 0, 0, 6, 5),
        _m110(600, "short", 1, 1, 20, 20, 40, 18, 6, 6),
        _m110(600, "long", 1, 1, 20, 20, 40, 144, 4, 6),
        _m110(600, "zero", 1, 1, 20, 20, 0, 0, 6, 6),
        _m110(300, "short", 1, 2, 20, 20, 40, 18, 6, 7),
        _m110(300, "long", 1, 2, 20, 20, 40, 144, 4, 7),
        _m110(300, "zero", 1, 2, 20, 20, 0, 0, 6, 7),
        _m110(150, "short", 1, 4, 20, 20, 40, 18, 7, 4),
        _m110(150, "long", 1, 4, 20, 20, 40, 144, 5, 4),
        _m110(150, "zero", 1, 4, 20, 20, 0, 0, 7, 4),
        # 75 bit/s fixed frequency: rows advance by 7 and columns fall by 7.
        _m110(75, "short", 2, 1, 32, 0, 10, 9, 7, 5, 7, 7),
        _m110(75, "long", 2, 1, 32, 0, 20, 36, 5, 5, 7, 7),
        _m110(75, "zero", 2, 1, 32, 0, 0, 0, 7, 5, 7, 7),
    )
}
#: What a preamble names. The zero modes share the short modes' D1/D2.
MODES110_BY_D = {(m.d1, m.d2): m for m in MODES110.values() if m.interleave != "zero"}


def zero_of(mode: Mode110) -> Mode110:
    """The zero-interleave mode a short preamble stands for, when a receiver
    has been told the far end bypasses its interleaver."""
    return MODES110.get(f"{mode.rate}Z", mode) if mode.interleave == "short" else mode


def gray110(bits_per_symbol: int) -> tuple[int, ...]:
    return {3: GRAY110_8, 2: GRAY110_4, 1: GRAY110_2}[bits_per_symbol]


# ==========================================================================
# STANAG 4285.
# ==========================================================================

# 256 symbols: 80 of sync, then data 32 / reference 16 three times and a last
# 32 of data; the references are symbol 0 (Annex A 5, figure A-3).
FRAME4285 = 256
SYNC4285_LEN = 80
DATA4285_PER_FRAME = 128


def _frame4285_layout() -> np.ndarray:
    """Per data-phase symbol (176 of them): True where it is a reference."""
    return np.array([((i % 48) >= 32) for i in range(176)])


LAYOUT4285 = _frame4285_layout()


def sync4285() -> np.ndarray:
    """The 80-symbol sync (Annex A 6, figure A-4): x^5 + x^2 + 1 loaded with
    11010 at every frame, the first symbol the least significant bit, sent as
    2-PSK — 0 as phase 0 and 1 as phase 180."""
    state = [1, 1, 0, 1, 0]
    out = np.empty(SYNC4285_LEN, dtype=np.int64)
    for i in range(SYNC4285_LEN):
        out[i] = 4 * state[4]
        fb = state[2] ^ state[4]
        state = [fb] + state[:4]
    return out


def scramble4285() -> np.ndarray:
    """The 176-symbol randomiser over data and reference symbols (Annex A 7,
    figure A-5): x^9 + x^4 + 1 from all ones at every frame, the symbol
    n = 4 x2 + 2 x1 + x0 off the last three stages, three shifts apart."""
    state = [1] * 9
    out = np.empty(176, dtype=np.int64)
    for i in range(176):
        out[i] = 4 * state[6] + 2 * state[7] + state[8]
        for _ in range(3):
            fb = state[4] ^ state[8]
            state = [fb] + state[:8]
    return out


# Bits -> 8-PSK index, oldest bit most significant (Annex A 3, "transcoding").
# The 8-PSK labelling is not 188-110's.
GRAY4285_8 = (1, 0, 2, 3, 6, 7, 5, 4)
GRAY4285_4 = (0, 2, 6, 4)
GRAY4285_2 = (0, 4)


@dataclass(frozen=True)
class Mode4285:
    rate: int
    coded: bool
    bits_per_symbol: int
    interleave: str             # "short" | "long" | "none"
    increment: int              # the interleaver's row-delay step, in bit slots
    repeat: int = 1             # times each coded bit pair is sent (Annex E)

    @property
    def slots_per_symbol(self) -> int:
        """Interleaver slots a symbol spends. At 2400 bit/s the rate-1/2 stream
        goes into the interleaver whole and every fourth row is skipped on the
        way out (Annex E 3(a)), so a symbol spends four slots and carries
        three."""
        return 4 if (self.coded and self.bits_per_symbol == 3) else self.bits_per_symbol

    @property
    def flush_bits(self) -> int:
        """Implemented, foreign-unverified reading of Annex E 5(f): the interleaver's delay in
        message bits, 32 rows of it, plus 102 for the decoder's traceback."""
        return self.increment * 32 * 32 // (2 * self.repeat) + 102

    @property
    def name(self) -> str:
        return f"{self.rate}{self.interleave[0].upper()}"


# Unverified Annex E reading: increments and pair-wise repetition that
# carries 300, 150 and 75 bit/s on the 600 bit/s BPSK stream.
MODES4285 = {
    m.name: m for m in (
        Mode4285(2400, True, 3, "long", 48),
        Mode4285(2400, True, 3, "short", 4),
        Mode4285(1200, True, 2, "long", 24),
        Mode4285(1200, True, 2, "short", 2),
        Mode4285(600, True, 1, "long", 12),
        Mode4285(600, True, 1, "short", 1),
        Mode4285(300, True, 1, "long", 12, 2),
        Mode4285(300, True, 1, "short", 1, 2),
        Mode4285(150, True, 1, "long", 12, 4),
        Mode4285(150, True, 1, "short", 1, 4),
        Mode4285(75, True, 1, "long", 12, 8),
        Mode4285(75, True, 1, "short", 1, 8),
    )
}
# Uncoded: the channel rates of Annex A itself, keyed with a U so they do not
# collide with the coded modes that share their numbers.
MODES4285["3600U"] = Mode4285(3600, False, 3, "none", 0)
MODES4285["2400U"] = Mode4285(2400, False, 2, "none", 0)
MODES4285["1200U"] = Mode4285(1200, False, 1, "none", 0)

# Annex E 5(e): framing for a coded message, leftmost bit first.
SOM4285 = 0x03873C3C
SOM4285_BITS = tuple((SOM4285 >> (31 - i)) & 1 for i in range(32))


def gray4285(bits_per_symbol: int) -> tuple[int, ...]:
    return {3: GRAY4285_8, 2: GRAY4285_4, 1: GRAY4285_2}[bits_per_symbol]
