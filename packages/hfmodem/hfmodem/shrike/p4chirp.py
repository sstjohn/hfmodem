# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-4 speed level 1, "chirp mode": the entry packet, from the published spec.

The geometry follows [SCS-P4] section 11, which declares the whole level
borrowed: "The PACTOR-4 speedlevel 1 is based on the PACTOR-2 speedlevel 1. The
main difference is the chirp of the two carrier frequencies." So the coding chain
IS `pactor2`'s and is reused from there rather than restated -- section 11.5's
"k = 9, R = 1/2, same as P2, speedlevel 1" polynomials are `coding.CODE_K9`
(bit-reversed into this package's register convention, recorded on that
constant), 11.6 reprints P2 Annex II's depth-16 pointer walk that
`pactor2.interleave_pointer` transcribes, 11.3's header is "Same as PACTOR-2
header for speedlevel 1 / long", and 11.7's alphabet is "just like P2" -- for
which `pactor2._DBPSK_STEP` carries the measured diagonals, 90 degrees from the
pair the document prints, with the measurement that settles it. What is new is
the waveform under all of that: two carriers at 550 and 1530 Hz (11.7.1),
66.66 Bd DBPSK (11.7: T = 15 ms), the lower carrier delayed T/2 -- the sentence
is load-bearing in 11.11's own packet-length arithmetic, so the sign is the
document's and not an inference -- both carriers chirping at 294.0 Hz/s (11.7.3)
under the 32-tap quasi-RRC symbol filter 11.8 prints as C source. The sweep
origin includes a calibrated T/4 advance of the section 11.7.2 reading.

THE CHIRP RUNS UPWARD, which the document does not say in words but fixes twice
over: downward takes the 550 Hz carrier through zero inside the packet, and
upward the two carriers average 1040 Hz at the start and 1957 at the end -- a
packet-long mean on the 1500 Hz centre the document gives the signal, with the
end of the sweep inside its 2400 Hz (@ -25 dB) bandwidth. Nothing stops the ramp
before the packet ends, so it runs to the end.

Sections 11.9 and 11.10 do not exist -- the numbering steps from 11.8 to 11.11
-- so no published text covers what fills a field with nothing to say. The empty
entry field takes `spec.field_fill`'s template, PACTOR-3's measured convention,
carried over on 11.4's word that the source encoder "is identically with the P3
source encoder". The calibrated sweep origin is empirical as well.

Packet whitening follows the functional evidence documented in
`docs/protocols/pactor/pactor4-framing.md`, checked against four recorded Dragon
chirps. `p4chirprx` decodes explicitly located chirps; it does not implement
connected ARQ. The test demodulator also checks the section-11 waveform
independently. Remote acceptance of this entry waveform remains unverified.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import coding, p4frame, pactor2, placement, spec

SYMBOL_S = 0.015
"""T. "2 carriers, modulated with 66,66 Bd (T = 15 ms) DBPSK each" [SCS-P4] 11.7."""

TONES_HZ = (550.0, 1530.0)          # 11.7.1, "before start of the Chirp"

CHIRP_HZ_PER_S = 294.0              # 11.7.3

CHIRP_START_T = 12.25
"""Empirical T/4 advance of the 12.5 T reading, calibrated against stock chirps."""

QUASI_RRC = np.array([
    0.0013033, 0.00099339, 0.00015167, -0.0021225,
    -0.0058615, -0.01035, -0.014008, -0.014558,
    -0.0095413, 0.002945, 0.023476, 0.050767,
    0.081603, 0.11134, 0.13489, 0.14791, 0.14791,
    0.13489, 0.11134, 0.081603, 0.050767, 0.023476,
    0.002945, -0.0095413, -0.014558, -0.014008,
    -0.01035, -0.0058615, -0.0021225, 0.00015167,
    0.00099339, 0.0013033,
])
"""11.8's `scsRrc_chirp`, verbatim: 32 real coefficients, 8 per symbol, so the
pulse spans the 4 T that 11.11's packet-length arithmetic charges for it."""

_SPS8 = 8                           # the filter's own rate: 8 samples per T

PATH = pactor2.Path("P4-chirp", 0, 416, 208, 16, 1, coding.RATE_1_2)
"""The chirp frame as one more `pactor2.Path`, which is what lets the whole P2
coding chain serve it unchanged. 11.11/11.12 fix every number: 208 data symbols
per carrier over 2 carriers is 416 transmitted bits, halved through the R=1/2
code to 26 bytes, less the flush byte the trellis termination spends --
`crc_bytes` comes out at 25: 22 user bytes, the status byte, the CRC-16 (11.2).
Depth 16 (11.6) divides 416. Level 0 in `pactor2`'s 0-based ladder is SL1."""

MARKER_K = pactor2.marker_index(PATH.level, long_frame=True)
"""11.3: "Same as PACTOR-2 header for speedlevel 1 / long" -- codeword 8.

Marker 8 starts a packet; marker 9 repeats it. The lane order alternates by
3.75 s slot independently of the marker (N9VC o0802, Trimode 2206/2226).
The four Dragon excerpts happened to correlate the two; they do not establish
a marker-to-lane mapping."""

CHIRP_CRC_XOR = 0x53E1
"""Historical residual when checking a still-whitened 25-byte chirp field.

Not a CRC convention: packet whitening explains this value exactly. The wire
field must whiten payload and status as well as CRC. Retained for historical
analysis; the transmitter uses ordinary X-25 followed by `p4frame.whiten`.
See docs/protocols/pactor/pactor4-framing.md.
"""

PACKET_S = 216 * SYMBOL_S + 4 * SYMBOL_S + SYMBOL_S / 2
"""3307.5 ms, 11.11's own sum: 217 pulses per carrier -- reference, eight header
chips, 208 data symbols -- spanning 216 T, the 4 T symbol filter, and the T/2 the
delayed lower carrier finishes late by."""


def entry_packet(payload: bytes, status: int,
                 fs: int = spec.SAMPLE_RATE, *, repeat: bool = False,
                 rank0_upper: bool = True) -> np.ndarray:
    """The chirp packet that answers a grant, as audio.

    The field rule is `placement.field_info`'s: an empty payload takes the
    template at the declared data type, a short one is padded with IDLE.
    The caller supplies repeat identity and the slot's lane order independently.
    The defaults retain the standalone entry waveform's arrangement.
    """
    info = placement.field_info(payload, PATH.crc_bytes - 3, status)
    field = p4frame.encode_field(info[:-1], p4frame.Status.from_byte(info[-1]))
    channel = np.zeros(PATH.n_buf, np.uint8)
    channel[pactor2.channel_of_code(PATH)] = pactor2.encode_frame(field, PATH)
    lanes = channel.reshape(PATH.n_symbols, 2)

    p2_lower, p2_upper = pactor2.marker_steps(MARKER_K + int(repeat))
    data_steps = pactor2.cell_steps(1)
    n_pulses = 1 + pactor2.MARKER_CHIPS + PATH.n_symbols
    delay_max = _SPS8 // 2
    n8 = (n_pulses - 1) * _SPS8 + QUASI_RRC.size + delay_max

    up, down = 3 * fs, round(_SPS8 / SYMBOL_S * 3)      # 8/T = 1600/3 Hz exactly
    g = gcd(up, down)
    # T/2 sets the symbol-boundary convention; the 32-tap filter support starts
    # 15.5 samples (1.9375 T) before its midpoint.
    chirp_at = (CHIRP_START_T * _SPS8 + (QUASI_RRC.size - 1) / 2
                - _SPS8 / 2) * SYMBOL_S / _SPS8 * fs

    out = None
    # Physical index 0 is the upper carrier; 11.7 delays the lower by T/2.
    # Only data lanes swap. The marker pair stays on these physical tones.
    # The header pair is CROSSED against PACTOR-2's: a P4dragon keys the word
    # `marker_steps` puts on P2's lower tone at 1530 Hz. Measured on the
    # sigidwiki recording -- four chirp packets score 0.93-1.00 crossed against
    # 0.55-0.70 the other way.
    for rank, marker, tone, delay8 in ((0, p2_lower, TONES_HZ[1], 0),
                                       (1, p2_upper, TONES_HZ[0], delay_max)):
        lane = rank if rank0_upper else 1 - rank
        walk = np.concatenate([marker, data_steps[lanes[:, lane]]])
        phasors = np.exp(1j * np.concatenate([[0.0], np.cumsum(walk)]))
        stuffed = np.zeros(n8, np.complex128)
        stuffed[delay8:delay8 + (n_pulses - 1) * _SPS8 + 1:_SPS8] = phasors
        bb = resample_poly(np.convolve(stuffed, QUASI_RRC)[:n8], up // g, down // g)
        i = np.arange(bb.size)
        t_past = np.maximum(i - chirp_at, 0.0) / fs
        theta = 2 * np.pi * (tone * i / fs + CHIRP_HZ_PER_S / 2 * t_past ** 2)
        carrier = (bb * np.exp(1j * theta)).real
        out = carrier if out is None else out + carrier
    return out / np.max(np.abs(out))
