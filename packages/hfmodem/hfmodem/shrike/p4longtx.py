# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-4 normal-mode LONG packet, the 3.75 s cycle: the transmitter.

The long counterpart of `p4shorttx`, and the same chain end to end. Section 5
orders the field -- 203 payload bytes, status, CRC-16 -- and `p4frame` whitens
it. Section 7's two K=4 components are `p4coding`; section 8.1's 1/3 keeps both
parity fields whole, so `p4coding.puncture` never runs here either. Section 6.4
lays twenty-four 207-symbol data blocks behind twenty-four trainings and closes
with a twenty-fifth, which `p4normal` constructs. Section 10 shapes every chip
with the 129-tap pulse on the 1500 Hz centre, and `p4long` reads it back.

What the two CRC-valid Dragon long frames in `tests/shrike/fixtures/p4-long`
say about the parts the document leaves open:

THE TAIL. Section 3 charges 1656 systematic bits for a 206-byte field's 1648,
and the eight are spent exactly as the short packet's eight are: five zero
bits, then the three `rsc_terminate` returns for the state those 1653 leave the
first component in. Both frames' turbo decoders return those eight bits
unconstrained and both return what this packing predicts. Both components start
from zero -- unique over all 64 pairs of initial states, 276 and 283 parity
disagreements against 1034 and 1056 for the runners-up -- and the second, which
reads the permuted 1656, is not terminated; it ends in (1,1,1) and (0,0,0).

THE PERMUTATION. C2 reads the 1656-entry table `p4long.component_permutation`
returns, and the body rides the 4968-entry symbol interleaver `tablegen` builds
from the published 11.6 recurrence.

WHAT SAYS THE CHAIN RUNS FORWARD. Re-encoding each recorded frame from its own
decoded bytes puts the parity fields 129 and 147 bits (the second frame: 144
and 139) from that recording's raw decisions -- the rate its systematic field
already carries, 104 and 147, so the distance is the channel and not the model.
Every wrong reading of the same chain lands at chance, 1656/2: C2 without its
input permutation 804 and 845, the two components swapped 800 and 862, initial
state (1,0,0) 918 and 896, the symbol interleaver omitted 790 to 857, and the
eight extra bits taken as plain zeros instead of five-plus-termination 659 and
639 -- that last one costs C1 nothing, which is why C2 is what settles it.

THE HEADER -- MEASURED, NOT PUBLISHED, AND NOT THE SHORT ONE. `_HEADER_IQ` is
the first recorded long frame's header, lifted chip by chip through the channel
estimate its own trainings support, quantised to int16 re/im over 32767. Its
140 chips and their 77.8 ms are the short packet's, measured again on these
recordings. The two long frames send the SAME header -- 0.970 correlation after
equalisation, against 0.03 and 0.15 for their bodies and 0.08 and 0.10 for the
noise ahead of them -- and their status counters differ, 3 and 0, where the
header does not.

It is NOT `p4shorttx.HEADER`: 0.245 against it, and 0.412 at the best of a
+/-900 Hz frequency scan, where permuting the short header's own chips already
reaches 0.215 median and 0.262 worst over the same scan. Some structure is
shared; what distinguishes them is unread. A cloned header remains a claim this
station cannot read -- packet type, speed level and ARQ role are unknown -- and
rendering it is not evidence that a Dragon accepts it.

The fill behind a short payload is `spec.field_fill`'s, PACTOR-3's measured
convention carried over as it is for the chirp; no P4 source framing
underwrites it.

Source recording and provenance: `tests/shrike/fixtures/p4-long/README.md`.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import p4coding, p4frame, p4long, p4normal, p4rx, placement, spec, tablegen

LAYOUT = p4long.LAYOUT
"""6.4's normal long: twenty-four 207-symbol data blocks, each behind its own
32-symbol training, and a twenty-fifth training to close. 5768 symbols."""

NET_BYTES = p4long.NET_BYTES

SYSTEMATIC_BITS = p4long.SYSTEMATIC_BITS

PAD_BITS = SYSTEMATIC_BITS - 8 * NET_BYTES - 3

INITIAL_STATE = (0, 0, 0)

HEADER_CHIPS = 140
"""Measured, not published: 77.8 ms of header ahead of the first training."""

PULSE_S = (p4rx.RRC_TAPS.size - 1) / p4rx.SAMPLE_RATE

EMISSION_S = (HEADER_CHIPS + LAYOUT.total_symbols - 1) / p4rx.CHIP_RATE + PULSE_S
"""3.286 s: 5907 chip periods from the first pulse to the last, and the 4.44 ms
the section 10 pulse spans. The packet sits on the published 3.75 s raster."""

_HEADER_IQ = np.array([
    -688, 6206, 14224, -4426, 10043, -13408, -17453, -13366,
    2800, 20397, -2877, -9728, -1163, 16597, -6873, -877,
    16703, -2714, 11023, -19650, 17679, -20205, -21968, -517,
    12129, 13152, -1855, -7170, 11876, 21389, -6663, 11991,
    12829, -11796, -11571, -25254, 4986, -13085, -14495, -559,
    -3430, 14636, 6059, -13896, -5112, 3781, -2565, -4522,
    7111, -10848, -4964, -29181, 14587, -9801, -15403, -8070,
    12713, 228, -12279, -5983, 15175, 21958, -8009, 28095,
    11053, -11464, -17373, -15465, -5414, -13116, 7968, 18463,
    -12545, -17462, 4237, 12675, 9320, -20900, 29307, 4002,
    -4113, -1467, -7377, 10344, 507, 13841, 18159, -19406,
    -13596, -3128, 1183, -2030, -25596, 234, -1028, -3690,
    -9372, 19668, 8711, 16171, 6686, 9751, 430, -3372,
    -2146, -13919, -15975, 8725, 5200, 11093, 1240, 12997,
    -364, 8681, 3540, 4329, -19089, 3965, 12990, 22520,
    22951, 5725, -7040, -12688, 6983, -1964, -14184, -13913,
    -4198, -5936, -16581, -8167, -8278, -22620, -21182, -181,
    -1103, 6187, 15914, -6779, -9115, -10194, 14064, -10950,
    -12695, -6786, 9966, 1848, 18480, 2862, -10850, -24672,
    -13892, 32515, 14801, -3160, -11832, 19931, -3570, 756,
    24280, -2141, 16095, -16328, 23435, -4899, -8058, 1025,
    2126, -7984, -13380, 14630, 28905, 13687, 6921, 22533,
    18762, -6921, -10167, -14733, -12741, -451, 20467, 17802,
    -7847, -13341, -5054, 8614, -8620, -24709, 3730, -3720,
    -12321, 4808, -7672, 14410, -5783, 11818, 11052, -3461,
    1180, -22561, -15368, 7785, -5502, -7086, -3707, 11712,
    -14033, 11606, -6093, 32767, -18518, 6983, 13652, 17145,
    -1106, -14210, 4941, 7882, 2696, -26101, 10969, -10861,
    -18448, 9754, -3504, 27575, -8373, 14146, 21310, -5597,
    1225, -10672, 6019, 2071, -14432, -19586, -1329, -4441,
    -18865, 13092, 945, 27484, -17471, 14388, 6387, 3224,
    11196, -5573, -6600, 1197, 4623, 4495, -13001, 1443,
    -1656, 9065, -293, 9955, -18476, -7085, -842, 26037,
    18567, 1307, -3680, 4035, 23846, -8651, -6907, -8959,
    -8861, -8173, -21266, 1599, -17371, -17499, -11637, 15716,
], np.float64)

HEADER = _HEADER_IQ[0::2] + 1j * _HEADER_IQ[1::2]
HEADER /= np.sqrt(np.mean(abs(HEADER) ** 2))
"""The cloned header, unit RMS on the body's own amplitude."""

_SYMBOL_ORDER = tablegen.INTERLEAVERS["0xb81c0_4968"]()


def coded_bits(field: bytes) -> np.ndarray:
    """A whitened 206-byte field as 4968 coded bits, V0 then C1 then C2 (7.2)."""
    bits = np.unpackbits(np.frombuffer(field, np.uint8), bitorder="little")
    padded = np.concatenate([bits, np.zeros(PAD_BITS, np.uint8)])
    tail = p4coding.rsc_terminate(
        p4coding.rsc_encode(padded, initial_state=INITIAL_STATE).final_state)
    systematic = np.concatenate([padded, np.array(tail.systematic, np.uint8)])
    order = p4long.component_permutation()
    return np.concatenate([
        systematic,
        p4coding.rsc_encode(systematic, initial_state=INITIAL_STATE).parity,
        p4coding.rsc_encode(systematic[order], initial_state=INITIAL_STATE).parity,
    ]).astype(np.uint8)


def body_symbols(coded: np.ndarray) -> np.ndarray:
    """4968 coded bits interleaved onto the twenty-four data blocks."""
    data = np.empty(coded.size, complex)
    data[_SYMBOL_ORDER] = 1 - 2.0 * coded
    out = np.empty(LAYOUT.total_symbols, complex)
    for block, offset in enumerate(LAYOUT.training_offsets):
        out[offset:offset + 32] = p4normal.training_sequence(block + 1)
    for block, offset in enumerate(LAYOUT.data_offsets):
        out[offset:offset + LAYOUT.block_symbols] = (
            data[block * LAYOUT.block_symbols:(block + 1) * LAYOUT.block_symbols])
    return out


def long_packet(payload: bytes, status: int,
                fs: int = spec.SAMPLE_RATE) -> np.ndarray:
    """One normal-long packet as audio: cloned header, then our own body."""
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
