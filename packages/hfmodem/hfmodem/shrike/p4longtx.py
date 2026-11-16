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

What the two CRC-valid Dragon long frames in
`tests/shrike/fixtures/sigidwiki/p4-long` say about the parts the document
leaves open:

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

THE HEADER. `_HEADER_IQ` is the complete 160-chip header of WS8EOC's
normal LONG at sample 2628637 in the 2026-10-27 capture, recovered through the
channel fitted to its own twenty-five trainings. The stock packet reference
puts its first training at baseband sample 2548. Its first header pulse begins
twelve samples before that reference; the returned audio trims those twelve
leading pulse samples. The earlier Sigidwiki clone retained only 140 chips and
put the body 308 baseband samples ahead of the live stock raster.

Every recorded stock SL5 normal LONG on the census ledger (twenty frames,
six captures) carries header energy over all 160 chips with its first
training at 2548, but their header chips differ from frame to frame: the other
nineteen correlate 0.08 to 0.56 with this one. Header meanings remain unknown. Cloning
a recorded SL5 header for a different payload or speed level does not
establish stock acceptance.

The fill behind a short payload is `spec.field_fill`'s, PACTOR-3's measured
convention carried over as it is for the chirp; no P4 source framing
underwrites it.

Body coding provenance: `tests/shrike/sigidwiki-p4-long.md`.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import p4coding, p4frame, p4long, p4normal, p4rx, p4short, placement, spec, tablegen

LAYOUT = p4long.LAYOUT
"""6.4's normal long: twenty-four 207-symbol data blocks, each behind its own
32-symbol training, and a twenty-fifth training to close. 5768 symbols."""

NET_BYTES = p4long.NET_BYTES

SYSTEMATIC_BITS = p4long.SYSTEMATIC_BITS

PAD_BITS = SYSTEMATIC_BITS - 8 * NET_BYTES - 3

INITIAL_STATE = (0, 0, 0)

HEADER_CHIPS = 160
HEADER_SAMPLES = p4short.GEOM

PULSE_S = (p4rx.RRC_TAPS.size - 1) / p4rx.SAMPLE_RATE

EMISSION_S = HEADER_SAMPLES / p4rx.SAMPLE_RATE + (LAYOUT.total_symbols - 1) / p4rx.CHIP_RATE + PULSE_S

_HEADER_IQ = np.array([
    19373, 1097, -14946, -5310, 15439, -8594, 15766, 9406,
    -19700, 12190, 4428, -6106, -3524, 1414, 1094, 16338,
    25110, -2707, -29561, 3670, 24909, -6410, 13635, 3597,
    -30332, 10654, -992, 336, 9637, 4706, 10605, 32767,
    -1572, -3785, 4246, 8864, -4405, 2913, -14771, -10323,
    5710, 4348, -2642, 9599, -1128, 11969, 11589, -16283,
    -1408, -5578, -9776, -4619, 6817, -1931, 5738, 17244,
    -10480, -8535, 3456, -13978, -4312, 1215, -11150, 12888,
    -259, 27847, 654, -7853, 601, 20487, -8653, 9484,
    -6704, -13436, -4926, 7535, 2937, 7359, -17286, 8258,
    -10978, -61, 7033, -4421, -14084, 4498, -9065, -12410,
    12366, -7851, -12053, 2282, -6424, 1578, 1821, -13307,
    -13152, 1459, 16099, -9538, -10439, 41, -10260, 822,
    17627, -15963, 5571, 20, -3791, -4154, -9801, -15762,
    -22496, 2530, 22180, -12875, -13031, 1489, -8520, -2175,
    16734, -14668, -5805, 799, -7193, -5291, -8031, -16419,
    -4669, 12162, 5758, -21449, 2446, 5664, -1727, 15337,
    -2826, -14885, 338, -3297, -316, -325, -18075, -8925,
    3820, 17034, -9545, -19797, 10458, 17033, 1963, 9951,
    -17945, -10973, -5232, 8032, -7710, 6715, -19592, 12280,
    8910, -6334, -13540, 215, 1364, -5776, 14199, -4536,
    -7344, 3393, -2937, -3536, 2923, 3290, 13012, 11051,
    13763, -18959, -8276, 17176, 7144, -18136, 8984, -8382,
    4982, 21324, 7855, 3848, 5070, -4732, 20919, 5919,
    -8660, -11762, 14318, 16735, -11308, 1339, -13368, -7146,
    20094, 8045, -1998, -2068, 825, -5589, 8029, -18850,
    -25377, -16114, 18839, 10380, -16921, 1364, -8077, -7170,
    19626, -1048, -7069, -5291, 1294, -7776, 6869, -12838,
    -13072, 365, 25033, 3004, -9082, 5934, -12183, -1819,
    22501, -8020, 209, -5282, 1654, -6408, -7766, -15499,
    -13250, -11153, 10507, 11544, -17427, -11911, 12032, -8893,
    11327, 9778, 236, -99, 2565, -11915, 10585, -5002,
    16569, -11609, -7128, 12441, 5583, -11947, 12264, 4936,
    4168, 14606, 2664, 650, 13600, 1641, 14723, 8311,
    -10887, -8, 8233, 10749, -3576, 7334, -14419, -5613,
    11479, -829, 4627, 7397, 5938, 4264, -1977, -12618,
    1535, -6788, -4770, -2964, 4166, -1526, 13199, 9155,
    -15112, 3512, 3398, -7589, -609, -3227, -7480, 9560,
    20576, 11126, -8598, 962, 12324, -574, 10929, 5491,
    -15060, 7892, -2928, 3094, 3172, 8503, 114, 16985,
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
    shaped = np.convolve(impulses, p4rx.RRC_TAPS)
    shaped = shaped[HEADER_CHIPS * p4rx.SAMPLES_PER_CHIP - HEADER_SAMPLES:]
    bb = resample_poly(shaped, up // g, down // g)
    out = (bb * np.exp(2j * np.pi * p4rx.CENTRE_HZ * np.arange(bb.size) / fs)).real
    return out / np.max(np.abs(out))
