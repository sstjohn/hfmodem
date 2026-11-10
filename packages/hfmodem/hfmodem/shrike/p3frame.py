# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-III packet header block, and the acquisition burst that precedes a
link rather than a packet.

Packet assembly itself lives in `placement`: a packet is one phase-reference
symbol, the eight-symbol header block built here, and then the frame laid out row
by row on the speed level's tones. The templates below are read in one direction
by `read_header` and written in the other by `header_steps`, off the same
`spec` tables, so a receiver and a transmitter cannot disagree about them.

The seed phasors that used to be applied to every transmitted symbol are gone.
They spread the eighteen carriers by rotating each 8-symbol block, which lands on
one grid row in eight and survives into the differential -- and the real signal
does not carry them: measured over a real speed-level-3 packet, the lag-1 phase
residual is 12-17 deg RMS and flat across all eight residue classes, where ours
put one class at 51. An independent monitor reports our packets with the rotation
removed and it reported them with the rotation present, so it was never paying
for itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import modem, spec, tablegen


ACQ_BAND_HZ = (1080.0, 1920.0)      # the two LOs the acquisition test mixes with
ACQ_REF = 1                         # complex reference index; case = (ref >> 1) & 3
ACQ_MATCHED_SYMBOLS = 8
ACQ_LEAD_SYMBOLS = 12
ACQ_TAIL_SYMBOLS = 0                # idle symbols between the match and the body
ACQ_SWAP_BANDS = False              # False: block 1 on 1080, correlator variant 3
ACQ_CONJUGATE = False               # True selects the conjugated variant pair (1/2)


def acquisition_chips(ref: int = ACQ_REF) -> tuple[np.ndarray, np.ndarray]:
    """The 8 complex chips each band must match, newest chip first.

    pre800 is 64 eight-tap groups; groups (2m, 2m+1) are the real and imaginary
    taps of one complex reference, so it holds 32 complex 8-chip sequences -- 16
    for each of the two halves the acquisition test correlates. Half 1 scores the
    1080 Hz band and half 0 the 1920 Hz band, which is variant 3; ACQ_SWAP_BANDS
    selects variant 0. Only `ref` decides the case, so all four variants of refs
    {0, 1, 8, 9} are case 0 and they differ in what they leave on tones 5/12."""
    raw = tablegen.pre800().astype(float)
    rows = raw.reshape(64, 8) / 800.0
    c = rows[0::2] + 1j * rows[1::2]
    pair = (c[ref], c[16 + ref]) if ACQ_SWAP_BANDS else (c[16 + ref], c[ref])
    return tuple(np.conj(x) for x in pair) if ACQ_CONJUGATE else pair


def preamble_audio(cfg: modem.ModConfig, ref: int | None = None) -> np.ndarray:
    """±800 packet-acquisition preamble, synthesised from pre800.

    The receiving side mixes the input down at 1080 and 1920 Hz, low-passes and
    decimates by 12 to 800 Hz, and takes the LAG-8 differential phase per FFT bin
    -- lag 8 at 800 Hz being exactly one 100 Bd symbol. It correlates eight of
    those, one per symbol, against each complex pre800 reference, and accepts when
    |sum|^2 >> 15 exceeds 5100 with the sum's phase inside ±49°.

    So the burst that acquires is two carriers on the band local oscillators whose
    per-symbol DIFFERENTIAL phases are the negated arguments of the reference
    chips, newest chip last in time -- negated and time-reversed, both, because
    the correlator matches the conjugate of what it was handed. A lead of
    unmodulated symbols fills the correlator's history.

    WHICH reference wins is the speed level, through case = (ref >> 1) & 3, so
    refs 0 and 1 are case 0 and refs 2 and 3 are case 1 -- pass `ref` to choose
    rather than leaning on the module-level default. The burst is emitted at unit
    differential phase for `ACQ_LEAD_SYMBOLS` symbols and then the eight matched
    symbols, i.e. 20 symbols at 100 Bd = 200 ms.
    """
    fs = cfg.sample_rate
    sps = int(round(fs / spec.SYMBOL_RATE_BD))
    n_sym = ACQ_LEAD_SYMBOLS + ACQ_MATCHED_SYMBOLS + ACQ_TAIL_SYMBOLS
    out = np.zeros(n_sym * sps)
    for f, chips in zip(ACQ_BAND_HZ, acquisition_chips(ACQ_REF if ref is None else ref)):
        dphi = np.concatenate([np.zeros(ACQ_LEAD_SYMBOLS), -np.angle(chips)[::-1],
                               np.zeros(ACQ_TAIL_SYMBOLS)])
        phase = np.repeat(np.cumsum(dphi), sps)
        out += np.cos(2 * np.pi * f * np.arange(len(phase)) / fs + phase)
    peak = np.max(np.abs(out))
    return out / peak * cfg.amplitude if peak > 0 else out


PATTERN_A = np.array([1, -1, 1, 1j, -1, -1, -1, 1j], dtype=complex)
"""The reference decoder's own 8-symbol pilot pattern: phases 0, 180, 0, 90, 180,
180, 180, 90 deg. Only its first symbol reaches the air -- a packet opens on one
phase-reference symbol -- but the rest is what a receiver's replica is built
from, so the whole pattern stays here rather than a lone constant."""


# ---------------------------------------------------------------------------
# The packet header block
# ---------------------------------------------------------------------------

HEADER_SYMBOLS = 8
"""DQPSK symbols the header block occupies, on every lit channel at once.

A variable header is 32 bits carried over the two VH channels and a constant
header is 16 bits on each of the other sixteen. Both come to eight dibits per
channel, so one block of eight symbols covers whatever the speed level lights --
the whole comb only at speed level 6. PT-III §4."""

DATA_OFFSET = 1 + HEADER_SYMBOLS
"""Symbols from a packet's phase-reference pulse to grid row 0.

A packet on an established link is [one phase-reference symbol][the header
block][one row per symbol] and nothing else. There is no acquisition burst inside
a cycle and no row is given up at the head; the whole 72-row field is on the air.

That comes to 81 symbols, and 81 is what the air carries. ONE SESSION says so and
it is written as one: `rf-corpus/PIII_Complete_1.wav` is the corpus's only
genuine PACTOR-3 reference -- its companion `PIII_Complete_2` holds no PACTOR-3
packet at all, and `ref_occ15_pactor3` is that same QSO re-encoded rather than a
second receiver. Its speed-level-1 entry packet lights channels 5 and 12 from the
phase reference through symbol 80 and collapses there, read through the matched
filter and again through a bare one-symbol DFT against the unlit channels. It
decodes CRC-valid in one trial, and its five payload bytes are the first five the
same station's level 3 packets carry four seconds later through a different code,
interleave and anchor.

A symbol past the field is a RUN-OUT, not an 82nd cell. All 35 bare codewords of
that session hold full amplitude through symbol 21 -- one past the last the
twenty bits need -- at a positive differential from symbol 20, so that symbol
repeats its predecessor and carries a zero bit no reader takes; by symbol 23 the
carrier is gone. Its packets carry one symbol past row 71 too, and that one is no
part of the field either. Nor does the answer move with it: the peer replies
889.4-891.9 ms after the phase reference over sixteen short cycles and answers
this 81-symbol packet at 891.2, inside that band. The answer slot belongs to the
cycle rather than to where the keying stopped."""

DIBIT_PHASOR = np.exp(0.25j * np.pi * np.array([3, 1, 5, 7]))
"""Dibit -> differential phasor, and one packet carries only this one map.

Which dibit goes where was read off the air rather than assumed. All 24 dibit
permutations were tried against both bit orders and both nibble conventions on a
real speed-level-3 packet; this one, LSB-first over the `spec.header_tx_form`
transmit form, lifts the sixteen published constant headers out at 92.5 of a
possible 96, where the mean over every hypothesis is 59.2 and the spread 2.9.

That measurement leaves the constellation's ABSOLUTE angle open -- its three
next-best entries were its own rotations, because `read_header` scores a
magnitude and a rotation is exactly what a magnitude cannot see. A receiver may
leave it open; a transmitter has to pick one, so the air settles it. Correlating
each channel against its own published word over five real packets lands at +2.5
to +5.5 deg of the map written here, spread 1.4 deg across twelve channels --
residual carrier offset and nothing else. The axis-aligned reading this constant
used to carry is 45 deg away and would have been transmitted wrong.

So the header sits on the same pi/4 diagonals as the data field, which is why
`placement` takes its DQPSK table from here instead of declaring a second one."""

VH_ORDER = spec.VH_CHANNELS[::-1]
"""The two variable-header carriers in virtual-carrier order: 12 first, then 5.

The 32-bit header's sixteen dibits alternate between the two, and which of them
takes dibit 0 is not a convention that can be assumed. Measured over five real
packets and unambiguous: on unswapped cycles only this order matches a published
word (0.94-0.97, against 0.49-0.55 for the other), and on swapped cycles only the
reverse -- which is this same order carried through `spec.CARRIER_SWAP`."""

CH_CHANNELS = tuple(c for c in range(spec.N_CHANNELS) if c not in spec.VH_CHANNELS)
"""The channel each of the sixteen constant headers belongs to: the non-VH
channels, low to high.

WHICH header a channel is found carrying is therefore a measurement of the
carrier swap, and it is how a receiver reads that state. Confirmed on real
traffic: on alternate ARQ cycles every channel carries the header of
`spec.CARRIER_SWAP`'s partner instead of its own, on all twelve channels the
duplicate at CH7/CH11 does not make ambiguous."""


def _header_dibits(word: int, width: int) -> np.ndarray:
    bits = spec.header_tx_form(word, width)
    b = [(bits >> i) & 1 for i in range(width)]
    return np.array([b[2 * k] * 2 + b[2 * k + 1] for k in range(width // 2)])


CONSTANT_TEMPLATES = np.array([DIBIT_PHASOR[_header_dibits(w, 16)]
                               for w in spec.CONSTANT_HEADERS])
VARIABLE_TEMPLATES = np.array([DIBIT_PHASOR[_header_dibits(w, 32)]
                               for w in spec.VARIABLE_HEADERS])


def variable_header(sl: int, *, swapped: bool = False,
                    long_cycle: bool = False,
                    request_status: bool | None = None) -> int:
    """Which of the sixteen variable headers a packet of this shape carries.

    Request-status bit 0 follows the packet's request state, independently of
    its physical carrier ordering. Direct callers omitting request_status retain
    the historical swap-derived value; packet renderers supply the status bit.
    """
    request = swapped if request_status is None else request_status
    return int(request) | ((sl - 1) & 3) << 1 | int(long_cycle) << 3


def header_steps(vh: int, channels: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Virtual carrier -> the eight differential phase steps of its header block.

    Keyed by HOME channel, before the carrier swap moves the carrier anywhere:
    the swap is one relabelling of the whole packet and `placement` applies it in
    a single place.

    The block covers the speed level's own channels and no others. That is
    measured, not assumed, and it is why the constant-header fit is a statement
    about the level as well as about the timing: through the header block of the
    five clean speed-level-3 packets in the occ15 recording, channels 2-15 stand
    four to eighteen times over the floor and channels 0, 1, 16 and 17 sit on it.

    It is measured at FOURTEEN channels, and the narrow levels are an inference
    from it. Speed level 1 corroborates it: offered cold an independent monitor
    reads nothing, but offered four cycles at level 3 and then four at level 1 on
    the same grid -- the only way a real link reaches it -- it reads. Speed level
    2 does not: the arrangement we call HOME is never read, and acceptance is
    payload-dependent on top of that -- both measured with repeats, because an
    independent decoder does not give the same answer twice at that level. See
    `tests/shrike/test_p3_oracle.py:NARROW_LEVELS_NOTE`, which keeps the measured
    part separate from what is only inferred from it.
    """
    var = VARIABLE_TEMPLATES[vh].reshape(HEADER_SYMBOLS, len(VH_ORDER))
    return {cn: np.angle(var[:, VH_ORDER.index(cn)] if cn in VH_ORDER
                         else CONSTANT_TEMPLATES[CH_CHANNELS.index(cn)])
            for cn in channels}


@dataclass(frozen=True)
class PacketHeader:
    """One packet's header block, read off the comb.

    Everything a receiver needs to place and shape the data field that follows:
    where the grid starts, which speed levels the variable header admits, how
    long the cycle is, which way round the carrier swap is this time, and at what
    angle the constellation arrived.
    """

    at: int
    """Sample index of the phase-reference symbol the packet opens on."""
    vh: int
    """Which of the sixteen variable headers was sent."""
    fit: float
    """Agreement with the published header block it was matched on, 0..1.

    The constant headers where a comb wide enough to carry them was read
    (`p3rx.HEADER_FIT`), the variable header where the level is too narrow for
    that and channels 5 and 12 are all there is (`p3rx.VH_FIT`). The two are
    scored over 192 bits and 32, so they are weighed against their own
    thresholds and never against each other."""
    rot: float = 0.0
    """Radians the differential constellation sits off the published map."""
    carrier_swapped: bool | None = None
    """Measured physical carrier order, independent of request-status bit 0.

    Receivers set this from the matched header ordering. None preserves the
    historical constructor used by callers describing old generated fixtures.
    Production readers always retain the measured order explicitly.
    """

    @property
    def swapped(self) -> bool:
        """Whether virtual carriers occupy their partner physical tones."""
        return (bool(self.vh & 1) if self.carrier_swapped is None
                else self.carrier_swapped)

    @property
    def long_cycle(self) -> bool:
        return bool(self.vh >> 3 & 1)

    @property
    def levels(self) -> tuple[int, ...]:
        """Speed levels this header admits, lowest first.

        The header carries the level MODULO FOUR, so it never separates 1 from 5
        or 2 from 6; the channel occupancy does, two tones against sixteen.
        """
        return tuple(sl for sl in spec.SPEED_LEVELS
                     if (sl - 1) & 3 == self.vh >> 1 & 3)

    def tones(self, channels: tuple[int, ...]) -> tuple[int, ...]:
        """`channels` as this cycle carries them, in virtual-carrier order."""
        return (tuple(spec.CARRIER_SWAP[c] for c in channels) if self.swapped
                else channels)


def read_header(diffs: dict[int, np.ndarray]) -> tuple[int, float, float]:
    """Compatibility view of the decoded value, fit and rotation."""
    return read_header_arrangement(diffs)[:3]


def read_header_arrangement(diffs: dict[int, np.ndarray]) -> tuple[int, float, float, bool]:
    """Unit differentials -> (variable header, fit, rotation, physical swap).

    The constant headers are scored assignment-free -- each channel against all
    sixteen words, best taken -- because which one a channel carries is exactly
    what the carrier swap changes, so demanding a particular assignment would
    make the detector blind every other cycle. The variable header is scored over
    both orderings of the two VH channels for the same reason, the swap
    exchanging 5 and 12 along with everything else.

    Preserve the winning ordering. WS8EOC's 2026-09-11 retries keep variable
    header bit 0 set while the two physical arrangements alternate; the request
    bit cannot substitute for this measurement.

    The fit is a MAGNITUDE, so it says nothing about where the constellation sat
    -- and the rotation is the argument the magnitude threw away. Every template
    here is unit-modulus, so a block received `theta` off the map correlates to
    `N exp(j theta)` and the sixteen channels agree on it: that is a residual
    carrier offset measured against 192 known bits, not an estimate off the data.
    It matters because a magnitude detector is happy at any angle while the data
    field is not. Measured on the `PIII_18` recording, whose header block fits
    0.97 on all three of its packets: the constellation sits 39 deg off, six
    degrees inside a DQPSK decision boundary, and the field reads at a 4% bit
    error rate until the angle is taken out -- after which it is CRC-valid.

    ON A COMB TOO NARROW FOR THE CONSTANT WORDS the fit and the rotation come
    off the VARIABLE header instead, which is what `PacketHeader.fit` has always
    claimed they do and what `p3rx.vh_anchors` does with its own copy of this
    arithmetic. It matters where a caller does not yet know the speed level:
    read on two channels, a level 1 packet scored 0.0 against a threshold it
    could never reach, so its swap and its angle -- the two things a case-0
    decode cannot be run without -- were thrown away and the packet read on the
    home order at zero rotation.
    """
    corrs = constant_header_corrs(diffs)
    fit, rot = constant_header_fit(corrs)
    vh, vh_fit, vh_rot, swapped = read_variable_header(diffs)
    return ((vh, fit, rot, swapped) if corrs
            else (vh, max(vh_fit, 0.0), vh_rot, swapped))


def constant_header_corrs(diffs: dict[int, np.ndarray]) -> dict[int, complex]:
    """Each measured channel's best-matching published constant header.

    The complex correlation rather than its magnitude, because both halves are
    wanted downstream and neither is recoverable from the other: the magnitude
    is that channel's agreement and the argument is the angle its block arrived
    at. Kept per channel so that a caller scoring one speed level's comb after
    another pays for the sixteen correlations once.
    """
    out = {}
    for c in CH_CHANNELS:
        d = diffs.get(c)
        if d is None:
            continue
        corr = CONSTANT_TEMPLATES.conj() @ d
        out[c] = complex(corr[int(np.argmax(np.abs(corr)))])
    return out


def constant_header_fit(corrs: dict[int, complex],
                        channels: tuple[int, ...] | None = None
                        ) -> tuple[float, float]:
    """Agreement and rotation over `channels`, or over everything measured.

    AVERAGED OVER THE CHANNELS SCORED, so which ones those are sets the ceiling.
    A speed level lighting twelve of the sixteen constant-header channels, read
    over all sixteen, caps at about 0.875 before any fading takes anything: the
    four dark ones return the ~0.5 a best-of-sixteen match on eight random
    phasors gives. `p3rx.HEADER_COMBS` is where a sweep chooses.
    """
    vals = [corrs[c] for c in (corrs if channels is None else channels)
            if c in corrs]
    if not vals:
        return 0.0, 0.0
    return (sum(abs(v) for v in vals) / (HEADER_SYMBOLS * len(vals)),
            float(np.angle(np.mean([v / (abs(v) + 1e-12) for v in vals]))))


def read_variable_header(diffs: dict[int, np.ndarray]
                         ) -> tuple[int, float, float, bool]:
    """The 32-bit word on channels 5 and 12: value, fit, rotation, physical swap.

    Both orderings are scored because the swap exchanges the two carriers along
    with everything else, and the winner is the measurement -- it is the only
    reading of the arrangement a comb too narrow for the constant words has.
    """
    vh, vh_fit, vh_rot, swapped = 0, -1.0, 0.0, False
    for order in (VH_ORDER, VH_ORDER[::-1]):
        if any(c not in diffs for c in order):
            continue
        v = np.stack([diffs[c] for c in order], 1).ravel()
        corr = VARIABLE_TEMPLATES.conj() @ v
        i = int(np.argmax(np.abs(corr)))
        if float(np.abs(corr[i])) / v.size > vh_fit:
            vh, vh_fit = i, float(np.abs(corr[i])) / v.size
            vh_rot = float(np.angle(corr[i]))
            swapped = order != VH_ORDER
    return vh, vh_fit, vh_rot, swapped
