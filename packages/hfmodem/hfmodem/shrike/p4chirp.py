# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-4 speed level 1, "chirp mode": the entry packet, from the published spec.

Everything here is [SCS-P4] section 11, which opens by declaring the whole level
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
from 4.5 T after the start of the last header symbol (11.7.2), under the 32-tap
quasi-RRC symbol filter 11.8 prints as C source.

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
source encoder"; it is the one bet in this file the document does not underwrite.

Rendered from the published description only. No decode path exists for what
comes back -- `p4sig` can name a PACTOR-4 answer, not read it -- so the round
trip in `tests/shrike/test_p4chirp.py` demodulates against the section-11
numbers independently, and interoperability waits on a station that answers.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import coding, pactor2, placement, spec

SYMBOL_S = 0.015
"""T. "2 carriers, modulated with 66,66 Bd (T = 15 ms) DBPSK each" [SCS-P4] 11.7."""

TONES_HZ = (550.0, 1530.0)          # 11.7.1, "before start of the Chirp"

CHIRP_HZ_PER_S = 294.0              # 11.7.3

CHIRP_START_T = 8 + 4.5
"""Symbol periods from the packet's first pulse to the chirp, both carriers at
once: 11.7.2 puts it "exactly 4,5 T (67,5 ms) after the start of the last header
symbol", and the last header symbol is pulse 8 of reference-then-eight (11.11)."""

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

Read off a P4dragon and confirmed at the level and length: of the four chirp
packets on the sigidwiki recording two carry 8 and two carry 9, the same codeword
with P2's `flag` bit set -- AND HERE THE FLAG IS THE CARRIER SWAP. At 8 channel
rank 0 is the upper carrier, at 9 the lower, four for four; `pactor2.marker_index`
records that the flag is not the swap in PACTOR-2, and that measurement stands.
This packet is keyed at 8 and arranges its lanes to match."""

CHIRP_CRC_XOR = 0x53E1
"""What a P4dragon's chirp CRC-16 is, on top of `coding.crc16`'s X-25.

Measured on four chirp packets off one station (sigidwiki Pactor_IV_chirps,
2026-09-16): the polynomial is X-25's and the residual is this constant on all
four, but one station cannot separate a non-standard fixed seed from a seed
derived from the addressed station. Evidence: the chirp validation of
2026-09-16 against that recording, in the Robust Connect round's note."""

PACKET_S = 216 * SYMBOL_S + 4 * SYMBOL_S + SYMBOL_S / 2
"""3307.5 ms, 11.11's own sum: 217 pulses per carrier -- reference, eight header
chips, 208 data symbols -- spanning 216 T, the 4 T symbol filter, and the T/2 the
delayed lower carrier finishes late by."""


def entry_packet(payload: bytes, status: int,
                 fs: int = spec.SAMPLE_RATE) -> np.ndarray:
    """The chirp packet that answers a grant, as audio.

    The field rule is `placement.field_info`'s: an empty payload takes the
    template at the declared data type, a short one is padded with IDLE. One
    arrangement only -- PACTOR-2's per-cycle carrier swap is a property of its
    ARQ cycle, and this packet is keyed before any PACTOR-4 cycle exists to
    swap on.
    """
    info = placement.field_info(payload, PATH.crc_bytes - 3, status)
    field = info + (coding.crc16(info) ^ CHIRP_CRC_XOR).to_bytes(2, "little")
    channel = np.zeros(PATH.n_buf, np.uint8)
    channel[pactor2.channel_of_code(PATH)] = pactor2.encode_frame(field, PATH)
    lanes = channel.reshape(PATH.n_symbols, 2)

    p2_lower, p2_upper = pactor2.marker_steps(MARKER_K)
    data_steps = pactor2.cell_steps(1)
    n_pulses = 1 + pactor2.MARKER_CHIPS + PATH.n_symbols
    delay_max = _SPS8 // 2
    n8 = (n_pulses - 1) * _SPS8 + QUASI_RRC.size + delay_max

    up, down = 3 * fs, round(_SPS8 / SYMBOL_S * 3)      # 8/T = 1600/3 Hz exactly
    g = gcd(up, down)
    # 11.7.2 counts from the START of a symbol; a shaped pulse's start is its
    # peak less T/2, and the undelayed upper carrier is the one whose grid the
    # simultaneous instant is read off.
    chirp_at = (CHIRP_START_T * _SPS8 + (QUASI_RRC.size - 1) / 2
                - _SPS8 / 2) * SYMBOL_S / _SPS8 * fs

    out = None
    # channel rank 0 is the UPPER carrier (`pactor2.channel_buffer`, measured);
    # 11.7 delays the lower one, so rank 1 takes tone 0 and the half symbol.
    # The header pair is CROSSED against PACTOR-2's: a P4dragon keys the word
    # `marker_steps` puts on P2's lower tone at 1530 Hz. Measured on the
    # sigidwiki recording -- four chirp packets score 0.93-1.00 crossed against
    # 0.55-0.70 the other way.
    for rank, marker, tone, delay8 in ((0, p2_lower, TONES_HZ[1], 0),
                                       (1, p2_upper, TONES_HZ[0], delay_max)):
        walk = np.concatenate([marker, data_steps[lanes[:, rank]]])
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
