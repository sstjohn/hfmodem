# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-4 normal-mode short packet, speed levels 5 and 6: the transmitter.

Everything above the waveform is [SCS-P4] and already in this package. Section 5
orders the field -- payload, status, CRC-16 -- and `p4frame` whitens it. Section
7's two recursive systematic K=4 components are `p4coding`. Section 8.1's rate
for this variant is 1/3, which retains both parity fields whole, so there is
nothing to puncture and `p4coding.puncture` never runs. Section 6.4 lays six
176-symbol data blocks behind six 32-symbol CAZAC trainings and closes with a
seventh, which `p4normal` constructs. Section 10 shapes every chip with the
129-tap pulse `p4rx` prints, 16 samples to the chip, on the 1500 Hz centre.
`p4short` reads the result back.

Three things the document leaves open, and what the two CRC-valid Dragon frames
in `tests/shrike/fixtures/p4-short/audio.npz` say about them:

THE TAIL. Section 3's dimensions leave 352 systematic bits over a 43-byte
field's 344 and name no layout for the difference. Both recorded frames spend it
the same way: five zero bits, then the three `rsc_terminate` returns for the
state those 349 leave the first component in. Both components start from zero --
unique over all 64 pairs of initial states -- and the second, which reads the
permuted 352, is not terminated.

THE LABEL. Bit 0 keys +1 on the trainings' own phase. Both frames pass CRC on
that polarity and fail on the other.

WHAT SAYS THE CHAIN RUNS FORWARD. Re-encoding each recorded frame from its own
decoded bytes puts the parity fields 43 and 39 bits (the second frame: 41 and
38) from that recording's raw decisions -- the rate its systematic field already
carries, 33 and 50, so the distance is the channel and not the model. Every
wrong reading of the same chain lands at chance, 352/2: the second component
without its input permutation 176, the two components swapped 176, initial state
(1,0,0) 205, the symbol interleaver omitted 170.

THE HEADER -- MEASURED, NOT PUBLISHED. `_HEADER_IQ` is the first recorded
frame's header, lifted chip by chip through the channel estimate its own
trainings support, quantised to int16 re/im over 32767. The 140 chips and their
77.8 ms are measured on that recording: the impulse grid runs full amplitude
from 140 chips before training #1 and ramps in over the two before it. The two
frames send the SAME header -- 0.961 correlation raw, 0.941 chip by chip after
equalisation, against 0.14 for their bodies and 0.09 for the noise ahead of them
-- and their status counters differ, 1 and 2, where the header does not, so the
header does not carry the counter. Nothing else about it is resolved: SF16
despreading scores 0.04 and SF8 0.16, no spreading factor from 2 to 20 leaves a
rank-one block structure, and the sequence's cyclic autocorrelation is not
impulsive. `p4sig.header_score` cannot reach it at all -- its Chu19 x SF16
template is 304 chips, longer than the whole header.

A cloned header is therefore a claim this station cannot read. What it asserts
to a Dragon -- packet type, speed level, ARQ role -- is unknown, and rendering
it is not evidence that a Dragon accepts it.

Two more open items. The fill behind a short payload is `spec.field_fill`'s,
PACTOR-3's measured convention carried over as it is for the chirp, and no P4
source framing underwrites it. And `p4short.decode_body`'s window guard
erases the last fifteen body symbols although the seventh training now stands
behind them, so fifteen of the 1056 coded bits are read as erasures on the
return trip rather than as the observations they are.

Source recording and provenance: `tests/shrike/fixtures/p4-short/README.md`.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import p4coding, p4frame, p4normal, p4rx, p4short, placement, spec, tablegen

LAYOUT = p4normal.layout("short", terminal_training=True)
"""6.4's ordinary short: six blocks of 176 data symbols, each behind its own
32-symbol training, and figure 6.2's seventh training to close. 1280 symbols.
Both recorded frames send that seventh and then stop: it scores 0.66 and 0.56
where its six predecessors score 0.53 to 0.67, and the slot after it is
silence at a twentieth of the body's amplitude."""

NET_BYTES = 43
"""Section 3's N for SL5/SL6 short: 40 user bytes, the status byte, the CRC."""

SYSTEMATIC_BITS = 352
"""D, the systematic field the same table charges -- eight more than the field's
344. Five are zero and three terminate the first component."""

PAD_BITS = SYSTEMATIC_BITS - 8 * NET_BYTES - 3

INITIAL_STATE = (0, 0, 0)

HEADER_CHIPS = 140
"""Measured, not published: 77.8 ms of header ahead of the first training."""

PULSE_S = (p4rx.RRC_TAPS.size - 1) / p4rx.SAMPLE_RATE

EMISSION_S = (HEADER_CHIPS + LAYOUT.total_symbols - 1) / p4rx.CHIP_RATE + PULSE_S
"""792.8 ms: 1419 chip periods from the first pulse to the last, and the 4.44 ms
the section 10 pulse spans. The packet sits on the published 1.25 s short
raster."""

_HEADER_IQ = np.array([
    6506, -3165, 6835, -13774, 2417, -17244, -18962, 6575,
    16254, 4874, -9810, 2309, 23313, 10461, -241, 12009,
    11874, -12289, -14039, -8611, -8060, -18628, 2651, 14901,
    -11995, -7325, 9364, 9709, -2117, -20883, 23272, -7103,
    -7981, -6615, -208, 14159, 1278, 16943, 426, -21866,
    948, 10916, -9043, -11644, -18439, 27087, -18837, 6025,
    856, 11703, 4372, -8622, 7983, -9112, -24530, 4435,
    6938, 13925, -2461, -7901, 14334, 11777, 2989, -2578,
    18872, -7814, 2284, -26942, 7865, -22047, 39, -4757,
    -4513, -12073, 10836, 13674, 17153, -15833, 10554, -911,
    -3028, -13829, -16712, 2982, -8911, 18987, 9796, -2952,
    10321, -4758, -21158, -8521, -4054, 15234, -19110, 8229,
    -5479, 7396, -15179, 3255, -11621, -9098, -11310, 22805,
    -10365, 1258, 14321, 5114, -3818, -14133, 24470, -3506,
    -3086, -3279, 25714, 11977, 15398, 3237, 6850, -30145,
    -9574, 11719, 9963, -9151, -17604, 10320, -7290, -11142,
    1524, 8723, 1436, -1361, 12220, 7196, -15355, -5332,
    -4639, -14787, -20021, 12846, 19671, 7149, 9611, 16640,
    8980, -1556, 3978, -4545, -15954, 2645, 21964, 12027,
    14485, 2319, 4768, -24529, -18703, 2652, -16574, -12225,
    -3728, 2071, -5220, -15937, 3287, -20659, -24849, -2090,
    -2578, -12515, -5277, 10860, 16306, -9787, 12503, 14414,
    431, 1098, -7534, 25705, 1265, 21533, 26602, 665,
    -8793, -9188, 1326, 12182, -4245, -17942, 5492, -10954,
    -11890, -1441, -3838, 15759, -13446, 14843, 5415, -9419,
    -3236, 7761, 3733, -12320, -23517, 10204, -10621, -4643,
    -1549, 12244, 23157, -4933, 15449, -7103, -18015, 7081,
    10267, -3076, 1198, 1607, 27686, 86, 10119, 9078,
    11668, -1819, -12294, 3427, -10684, 3038, 18625, 8306,
    -2650, -14978, 5554, 3046, -808, -31883, 17, -11571,
    -17072, -41, -14896, 12796, -16240, 18325, 4865, -5633,
    6661, -10770, -13813, -351, 4990, 9345, -8395, 10984,
    -5744, 7373, -816, 17307, -10557, -4680, -615, 31961,
    14295, -5200, -3558, 2834, 18178, -4142, 5664, -6803,
    -9525, -1078, -16526, 1913, -17543, -15717, -5472, 13374,
], np.float64)

HEADER = _HEADER_IQ[0::2] + 1j * _HEADER_IQ[1::2]
HEADER /= np.sqrt(np.mean(abs(HEADER) ** 2))
"""The cloned header, unit RMS on the body's own amplitude."""

_HEADER_B_IQ = np.array([
    22383, -233, 12906, -7290, 7825, -21117, -8185, -24655,
    7230, -3357, -17854, 15877, 13535, -15003, 13295, 21620,
    -22231, 11414, -6883, 13942, 4710, 20724, 22463, 15225,
    6244, -26773, -11644, 19751, 9381, -14845, 7551, 4987,
    11836, 25785, 4490, 23389, 11366, 25029, 16376, -83,
    -27541, 4170, 20818, 8860, -15798, -6113, -3280, -19670,
    28457, -7243, 23529, -9370, 14864, -17175, -10227, -25528,
    -22742, -8646, 16320, 17508, -15179, -20018, 10859, -15266,
    26568, 9463, 22291, 12303, 26238, 3304, 7691, -24589,
    -22259, -5012, 21740, 10070, -16210, -12637, 3171, -17401,
    29418, 1677, 25076, 1969, 20782, -8522, -5101, -27063,
    -2944, -13544, -8553, 24550, 13229, -19265, 21670, 5914,
    -7643, 28475, -598, 28649, 9229, 19073, 25456, -4805,
    9315, -24409, -22147, 15533, 20528, -13797, 15834, 15055,
    -11725, 24331, -15271, 23470, 2220, 25899, 25421, 15469,
    11971, 14165, -10585, -21230, 10239, 23270, -22253, 9654,
    -16309, -21091, -13629, -24163, -24527, -10362, -18534, 19620,
    4605, 24807, 2964, -23974, -6065, 24080, -16552, 2249,
    -12626, -25436, -7610, -23508, -13187, -21308, -26334, 11628,
    -11924, 2139, 23337, -5465, -23953, -3366, -1938, -19090,
    27881, -9134, 23729, -9357, 18350, -14191, -11771, -19600,
    -22033, 16589, 26403, 250, -21562, 281, -7912, -19823,
    23276, -16510, 21650, -16932, 8985, -23059, -18967, -20113,
    -24838, -1446, 22536, 12238, -22115, -13338, 6592, -18645,
    27239, -1478, 25592, 4991, 26154, -6545, 7283, -21964,
    -4957, 21444, 9776, -16737, -8541, 14168, -19453, -1834,
    480, -30000, -1491, -25425, -10405, -20552, -24358, 6176,
    15871, 25409, -5930, -18615, 7686, 18905, -10912, 12914,
    -27038, -12060, -23189, -12810, -26093, -6546, -21903, 18672,
    -17685, -9390, 17095, 3595, -20564, -9095, 5113, -18933,
    29870, -684, 26807, 1624, 24143, -6102, 9471, -26401,
    21276, 3252, -16884, -3160, 18666, 3824, -1928, 22531,
    -28194, 5476, -26552, 2532, -23537, 13349, 4404, 25558,
    24695, 863, -23677, -8843, 20007, 12511, -841, 16785,
    -29302, 4685, -26661, 2480, -20760, 9365, 11416, 22481,
], np.float64)

HEADER_B = _HEADER_B_IQ[0::2] + 1j * _HEADER_B_IQ[1::2]
HEADER_B /= np.sqrt(np.mean(abs(HEADER_B) ** 2))
"""Measured, not published: the header of every retransmitted normal short.
The first copy of a packet carries HEADER; each repeat carries this one, the
mean of 62 CRC-valid repeat copies from WS8EOC on 2026-09-29 11:51, equalised
chip by chip and quantised as above. It correlates 0.055 with HEADER."""


def read_header(z, first_training_start, cfo_hz):
    """Which header a received normal short carries: 'A' (a first copy), 'B'
    (a repeat) or '?' when neither template leads by 0.1, with both scores."""
    positions = first_training_start + 128 + 16 * (np.arange(HEADER_CHIPS) - HEADER_CHIPS)
    lo = positions[0] - p4rx.RRC_TAPS.size
    if lo < 0 or positions[-1] >= len(z):
        return None
    n = np.arange(lo, positions[-1] + 1)
    chips = p4rx.matched_filter(z[lo:positions[-1] + 1] * np.exp(-2j * np.pi * cfo_hz * n / p4rx.SAMPLE_RATE))[positions - lo]
    energy = np.sqrt(np.vdot(chips, chips).real * HEADER_CHIPS)
    if energy == 0:
        return None
    a, b = (float(abs(np.vdot(t, chips)) / energy) for t in (HEADER, HEADER_B))
    return ('A' if a > b else 'B') if abs(a - b) >= .1 else '?', a, b

_SYMBOL_ORDER = tablegen.INTERLEAVERS["0xbaf20_1056"]()


def coded_bits(field: bytes) -> np.ndarray:
    """A whitened 43-byte field as 1056 coded bits, V0 then C1 then C2 (7.2)."""
    bits = np.unpackbits(np.frombuffer(field, np.uint8), bitorder="little")
    padded = np.concatenate([bits, np.zeros(PAD_BITS, np.uint8)])
    tail = p4coding.rsc_terminate(
        p4coding.rsc_encode(padded, initial_state=INITIAL_STATE).final_state)
    systematic = np.concatenate([padded, np.array(tail.systematic, np.uint8)])
    order = p4short.component_permutation()
    return np.concatenate([
        systematic,
        p4coding.rsc_encode(systematic, initial_state=INITIAL_STATE).parity,
        p4coding.rsc_encode(systematic[order], initial_state=INITIAL_STATE).parity,
    ]).astype(np.uint8)


def body_symbols(coded: np.ndarray) -> np.ndarray:
    """1056 coded bits interleaved onto the six training-and-data blocks."""
    data = np.empty(coded.size, complex)
    data[_SYMBOL_ORDER] = 1 - 2.0 * coded
    out = np.empty(LAYOUT.total_symbols, complex)
    for block, offset in enumerate(LAYOUT.training_offsets):
        out[offset:offset + 32] = p4normal.training_sequence(block + 1)
    for block, offset in enumerate(LAYOUT.data_offsets):
        out[offset:offset + LAYOUT.block_symbols] = (
            data[block * LAYOUT.block_symbols:(block + 1) * LAYOUT.block_symbols])
    return out


def short_packet(payload: bytes, status: int,
                 fs: int = spec.SAMPLE_RATE) -> np.ndarray:
    """One normal-short packet as audio: cloned header, then our own body."""
    info = placement.field_info(payload, NET_BYTES - 3, status)
    field = p4frame.encode_field(info[:-1], p4frame.Status.from_byte(info[-1]))
    symbols = np.concatenate([HEADER, body_symbols(coded_bits(field))])
    impulses = np.zeros(p4rx.SAMPLES_PER_CHIP * (symbols.size - 1) + 1, complex)
    impulses[::p4rx.SAMPLES_PER_CHIP] = symbols
    up, down = int(fs), p4rx.SAMPLE_RATE
    g = gcd(up, down)
    bb = resample_poly(np.convolve(impulses, p4rx.RRC_TAPS), up // g, down // g)
    out = (bb * np.exp(2j * np.pi * p4rx.CENTRE_HZ * np.arange(bb.size) / fs)).real
    return out / np.max(np.abs(out))
