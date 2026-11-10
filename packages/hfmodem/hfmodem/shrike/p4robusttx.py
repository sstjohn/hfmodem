# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-4 robust-mode short packet and control signal, rendered by value.

MEASURED, NOT PUBLISHED. [SCS-P4] gives robust mode its geometry -- section 6.2
lays a 20-symbol header, one phase reference and a datablock behind it, all
spread by `p4sig.SPREAD16` to 1800 chips/s, and section 10 shapes every chip
with the same 129-tap pulse the normal mode uses. It gives no header sequence,
no control-signal field and no bit labels. Everything below the geometry came
off two recordings: `rf-corpus/sigidwiki/PACTORIVaudio1.mp3`, sigidwiki's
PACTOR IV sample captioned "Initial SELCALL with beginning PACTOR IV data
transmit", and signals.taunus.de's `PACTOR-IV_112-5_DQPSK.WAV`.

THE RECORDINGS. The tape opens on four PACTOR-1-shaped call bursts, then three
760.0 ms bursts on the 1.25 s raster at 4.863/6.113/7.364 s in the caller's
slot, then three 244.4 ms bursts at 9.214/10.464/11.714 s in the answerer's. 85
spread symbols render as 759.4 ms and 27 as 243.9 ms, each within 0.6 ms of the
measured envelope, which is what section 6.2's 20 + 1 + N reads as 20 + 1 + 64
and 20 + 1 + 6. The taunus file carries four more short packets on the same
raster at 0.257/1.507/2.757/4.007 s, from a different station and clean: they
spend 0.905-0.912 of their energy on the published spreading sequence where the
tape's three spend 0.35, and their dibits are unchanged across four analysis
windows.

THE DATABLOCK IS A [64,6,32] CODE. Section 6.2 calls the robust payload DQPSK,
and it is not: all 65 post-header differentials of all seven packets are 0 or
2, differential BPSK, which no wrong carrier branch and no wrong chip phase
produces. Read 0 -> 0 and 2 -> 1 in transmission order, the 64 datablock bits
are `BLOCK_MASK` XOR one row of the 64-point Sylvester Hadamard matrix, on
every one of the seven: any two blocks differ in exactly 32 places, and each
leaves ZERO residual against the one shared mask. A block therefore carries SIX
BITS spread 64 ways, and `block_value` reads them back. The tape's caller sent
6, then 9, then 8; the taunus station sent 13, 13, 49, 49.

TWO HEADERS, ALTERNATING, OFF THE QUARTER-TURN GRID. `HEADER_STEPS` holds
both. The tape sends A three times; taunus alternates A, B, A, B while its
block repeats across each pair, so the header is a packet parity and the value
does not depend on it. Unlike the block, the header's steps are not multiples
of 90 degrees: N9VC's caller on the 08-02 K5DAT-13 recording and the 09-25
Kiwi recording, taunus and the tape put each of its nineteen steps at the same
arbitrary phase within 5-10 degrees across fourteen packets, and those phases
are the literals. Quantised to quarter turns, which is what a reader decides
and `HEADER_DIBITS` holds, they read back every recorded header's decisions
but the tape's first step, which reads 2 there and 0 everywhere else: the
faded tape drops a symbol at the head. So a reader matches on `HEADER_WORD`.

WHAT STAYS UNREAD. What a value MEANS, and where the mask comes from.
`BLOCK_MASK` is not the P4 field whitener at any bit offset in either bit
order, not its raw LFSR state stream, not PACTOR-3's crctable whitening, not
any degree-6 m-sequence and not periodic at 8 or 16. So a value is pinned only
up to a fixed 6-bit XOR, and the mask itself only up to a global complement;
RELATIVE values are exact. The Chu root is not pinned either --
`p4sig.header_score` returns 5, 11 and 7 for the tape's three bursts, which
carry one header. Keying a value asserts something to a Dragon whose meaning this
station cannot yet read.

THE CONTROL SIGNALS ARE PROVISIONAL. `CS_DIBITS` holds the three answerer
bursts on the same treatment, but 26 differentials do not carry the averaging
84 do: their decision concentration is 0.19-0.26 against the caller packets'
0.71-0.81, and the runner-up carrier branch trails by only a tenth. Two or
three of their six post-reference steps are not 0 or 2 at all, so the six-bit
field they are exactly the right width for cannot be read off them. Their heads
correlate 0.50-0.62 with each other where the caller's three correlate 0.955,
so they are three different bursts rather than one answer repeated. Two
answers are measured, off VE3KPG and N9VC at SF16 locks the tape never
reached: `robust_word` keys `p4words`' X27 and Y27 at their measured phases,
as a header is keyed.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import p4rx, p4sig, p4words, spec
from .p4words import phasors

HEADER_SYMBOLS = 20
"""[SCS-P4] §6.2's header, spread by 16: 320 chips, 177.8 ms."""

BLOCK_SYMBOLS = 64
"""§6.2's short datablock, the variant it gives no user data field."""

SHORT_SYMBOLS = HEADER_SYMBOLS + 1 + BLOCK_SYMBOLS
"""20 header + 1 phase reference + the datablock."""

CS_SYMBOLS = 27
"""20 + 1 + 6. The six-symbol field is not in §6.2's 64/348/696 table."""

HEADER_STEPS = {
    'A': (5, 88, -78, -155, -131, -16, -171, 131, 171, -60,
          172, 133, -172, -18, -134, -153, -76, 92, 1),
    'B': (-1, 161, 151, -48, -94, 31, -13, 86, 29, 102,
          28, 89, -18, 36, -94, -55, 144, 171, 5),
}
"""The two measured 20-symbol headers, as phase steps in degrees. A is the
tape's three bursts, taunus's first and third and N9VC's caller; B is taunus's
second and fourth and N9VC's caller's alternate packets."""

HEADER_DIBITS = {name: tuple(round(step / 90) % 4 for step in steps)
                 for name, steps in HEADER_STEPS.items()}
"""The headers as a reader's quarter-turn decisions."""

HEADER_WORD = slice(1, 19)
"""The station-invariant part of a header, and the only part worth matching."""

REFERENCE_STEP = 0
"""Header to phase reference. Zero on all seven packets, both recordings."""

BLOCK_MASK = 0x063D3AA80E8AD124
"""The fixed 64-bit mask the datablock's Walsh row is sent under.

Bit `n` of the block is bit `63 - n` of this integer, so the first symbol off
the reference is the high bit. Weight 26, algebraic degree 5, and matching
nothing generated: see the module note.
"""

CS_DIBITS = (
    (1, 3, 3, 0, 1, 3, 0, 0, 1, 0, 3, 0, 3, 3, 2, 3, 0, 2, 3, 2, 0, 3, 3, 0, 1, 0),
    (3, 1, 2, 3, 3, 2, 3, 3, 2, 2, 2, 2, 1, 1, 2, 1, 0, 2, 2, 1, 0, 1, 0, 3, 2, 2),
    (2, 2, 1, 0, 0, 3, 0, 1, 2, 1, 2, 1, 2, 2, 2, 3, 0, 0, 2, 0, 2, 3, 2, 3, 2, 0),
)
"""The tape's three answerer bursts on the same treatment. Provisional."""

PULSE_S = (p4rx.RRC_TAPS.size - 1) / p4rx.SAMPLE_RATE

EMISSION_S = (SHORT_SYMBOLS * p4sig.SPREAD_FACTOR - 1) / p4rx.CHIP_RATE + PULSE_S
"""759.4 ms: 1359 chip periods plus the 4.44 ms the §10 pulse spans. The
recording's own envelope reads 760.0 ms across the same emission."""

_WALSH = np.array([[bin(v & n).count('1') & 1 for n in range(BLOCK_SYMBOLS)]
                   for v in range(BLOCK_SYMBOLS)], np.int64)
_MASK_BITS = np.array([BLOCK_MASK >> (BLOCK_SYMBOLS - 1 - n) & 1
                       for n in range(BLOCK_SYMBOLS)], np.int64)


def walsh(value: int) -> np.ndarray:
    """Row `value` of the 64-point Sylvester Hadamard matrix, as 0/1 bits."""
    if type(value) is not int or not 0 <= value < BLOCK_SYMBOLS:
        raise ValueError(f"value must be an integer in 0..{BLOCK_SYMBOLS - 1}")
    return _WALSH[value].copy()


def block_bits(value: int) -> np.ndarray:
    """The 64 datablock bits a robust short packet carries for `value`."""
    return _MASK_BITS ^ walsh(value)


def block_value(bits) -> tuple[int, int]:
    """`(value, peak)` for 64 received datablock bits; peak is 0..64.

    The peak is the unsigned Walsh correlation, so a block that arrived through
    the despreader's other differential branch -- every bit complemented -- still
    names its own value. A clean packet reaches 64; nothing else comes close.
    """
    b = np.asarray(bits, np.int64)
    if b.shape != (BLOCK_SYMBOLS,) or np.any((b < 0) | (b > 1)):
        raise ValueError(f"bits must be {BLOCK_SYMBOLS} zeros and ones")
    correlation = (1 - 2 * (b ^ _MASK_BITS)) @ (1 - 2 * _WALSH).T
    value = int(np.argmax(np.abs(correlation)))
    return value, int(abs(correlation[value]))


def short_steps(value: int, header: str = 'A') -> tuple[int, ...]:
    """The 84 phase steps of one robust short packet carrying `value`, in degrees."""
    if header not in HEADER_STEPS:
        raise ValueError(f"header must be one of {sorted(HEADER_STEPS)}")
    return (HEADER_STEPS[header] + (90 * REFERENCE_STEP,)
            + tuple(180 * int(b) for b in block_bits(value)))


def short_dibits(value: int, header: str = 'A') -> tuple[int, ...]:
    """`short_steps` as the quarter turns a reader decides."""
    return tuple(round(step / 90) % 4 for step in short_steps(value, header))


def symbols(dibits) -> np.ndarray:
    """Differential quarter turns as absolute symbols, from a phase-0 start."""
    steps = np.asarray(dibits, np.int64)
    if steps.ndim != 1 or not steps.size or np.any((steps < 0) | (steps > 3)):
        raise ValueError("dibits must be a nonempty sequence of 0..3")
    return phasors(90 * steps)


def carrier_hz(word, word_render, chips=None):
    """Look up a word's carrier and optionally apply its zero-origin chip ramp."""
    if word_render not in ('off', 'stock'):
        raise ValueError(f'unknown P4 word render mode: {word_render!r}')
    hz = p4words.CARRIER_HZ.get(word, 0.) if word_render == 'stock' else 0.
    if hz and chips is not None:
        chips *= np.exp(2j * np.pi * hz * np.arange(chips.size) / p4rx.CHIP_RATE)
    return hz


def _render(spread_symbols: np.ndarray, fs: int, *, word=None, word_render='off') -> np.ndarray:
    chips = (spread_symbols[:, None] * p4sig.SPREAD16).ravel()
    carrier_hz(word, word_render, chips)
    impulses = np.zeros(p4rx.SAMPLES_PER_CHIP * (chips.size - 1) + 1, complex)
    impulses[::p4rx.SAMPLES_PER_CHIP] = chips
    up, down = int(fs), p4rx.SAMPLE_RATE
    g = gcd(up, down)
    bb = resample_poly(np.convolve(impulses, p4rx.RRC_TAPS), up // g, down // g)
    out = (bb * np.exp(2j * np.pi * p4rx.CENTRE_HZ * np.arange(bb.size) / fs)).real
    return out / np.max(np.abs(out))


def robust_short(value: int, header: str = 'A',
                 fs: int = spec.SAMPLE_RATE) -> np.ndarray:
    """One robust short packet as audio: header, reference and `value`."""
    return _render(phasors(short_steps(value, header)), fs)


def robust_word(word: str, fs: int = spec.SAMPLE_RATE, *, word_render='off') -> np.ndarray:
    """A gateway's 27-symbol answer word, `p4words.STEPS`' X27 or Y27, as audio."""
    if word not in p4words.STEPS:
        raise ValueError(f"word must be one of {sorted(p4words.STEPS)}")
    return _render(phasors(p4words.STEPS[word]), fs,
                   word=word, word_render=word_render)


def robust_cs_clone(k: int, fs: int = spec.SAMPLE_RATE) -> np.ndarray:
    """The tape's k-th answerer burst, k in 0..2. Provisional; see the note."""
    if type(k) is not int or not 0 <= k < len(CS_DIBITS):
        raise ValueError(f"k must be an integer in 0..{len(CS_DIBITS) - 1}")
    return _render(symbols(CS_DIBITS[k]), fs)
