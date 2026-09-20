# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-III data field receiver: audio in, payload bytes out, speed levels 1-6.

The demodulator was never the gap. What was missing was everything around it:
which speed level a burst is, where its grid begins, which channel each virtual
carrier is on, and how many candidate positions a CRC may be offered before a
passing CRC stops meaning anything.

A PACTOR-III packet answers the first three itself. It opens with one
phase-reference symbol and then an eight-symbol header block that carries the
sixteen published packet headers -- a 192-bit known pattern across the whole comb
-- so acquisition is a matched filter on a codeword rather than a search:
`header_anchors` places the packet to the sample, reads the speed level and the
cycle length out of the variable header, and reads the carrier swap out of which
constant header each channel is carrying. That last one is not a refinement. The
specification swaps every virtual carrier to a different tone on every ARQ cycle,
so a receiver holding one fixed tone order gathers the right energy in the wrong
order on alternate cycles and, having never applied the swap at all, on all of
them. This is what stood between shrike and a real data field: 13,148 blind
trials over speed level, cycle length, tone order and alignment returned nothing,
because none of those hypotheses was the one that was wrong.

The two narrow speed levels have their own codeword to be found by. Sixteen
published CONSTANT headers are 192 known bits across the whole comb, but only
where the whole comb is lit: speed level 2 carries four of them and speed level 1
none at all. What both carry is the VARIABLE header, 32 bits on the two channels
every level lights, and `vh_anchors` matches those sixteen words the same way --
which is how a real station's speed-level-1 entry packet is placed and read. An
envelope search survives behind both for a burst that matches neither.

WHAT A PASSING CRC IS WORTH. The CRC is 16 bits, so a scan offers it one chance in
65536 of accepting noise per position -- and a blind scan of a 30 s capture at
eighth-symbol resolution is 24,000 positions per speed level. Measured over 179,742
positions of real off-air audio carrying no PACTOR at all, the case-0 chain accepts
one in 36,000, and a 39-minute capture offers it two million. So an unbounded scan
MANUFACTURES decodes: swept over every capture the station holds, 12.4 hours of
audio with three genuine PACTOR-3 recordings taken out of it, the receiver claimed
a PACTOR-3 frame ten times an hour -- on FT8 calling frequencies, on an empty 80 m
channel, in the middle of a VARA sweep, with random payloads under plausible
status bytes.

THE RULE. A CRC accept is reported only when it survives moving the receiver's
clock -- `confirmed`, an eighth of a symbol either side, same field. That is not a
threshold on the signal, so it costs a weak station nothing: it is a statement
about the matched filter, whose output varies smoothly across a symbol, and it
therefore holds at every level below 5, and at speed level 5 down to 3 dB --
`confirmed` records where it stops. Measured: the real
KE5YTA case-0 header off 7101.5 kHz accepts at four consecutive eighth-symbol
alignments with an identical field; a rendered speed level 2, 3 or 4 data frame at
four to six from a clean channel down to 0 dB SNR, and the SL>=2 header a live link
reads at six even at -3 dB. Every accident in the first sweep -- five in the
negative audio above, three in 95,408 positions of white noise, and one apiece
inside the two genuine PACTOR-3 recordings we hold -- stood alone at exactly one
alignment; a larger sweep has since caught two, both at speed level 1, that do
not. The rule is a strong filter and not a proof, and `FALSE_RATE` carries the
measured residue.

`Scan.trials` is the second half, and it is arithmetic rather than judgement. Every
decode carries the number of positions offered to the CRC to find it;
`Scan.expected_false` turns that into how many CONFIRMED accepts the same scan
would be expected to produce on noise, and `TRIAL_BUDGET` is the most a decode may
carry and still be evidence. It is what retired the burst-wide SL1 scan, which on
one 88 s FT8 capture spent 19,484 trials to manufacture a five-byte payload.

In front of both, a gate that is a property of the signal: a speed level occupies a
KNOWN set of the eighteen 120 Hz channels, and `levels_present` throws out any level
whose channels are not lit. Two tones cannot be speed level 6 and eighteen cannot be
speed level 1, whatever the CRC says about either.

That gate never binds on speed level 1 itself -- a two-channel set has nothing for
`levels_present` to find missing -- so a case-0 accept answers the question at its
own scale instead: `sl1_carriers_present` asks whether the field the CRC accepted
put energy on BOTH of its carriers, which is what separates a packet from a CRC
accident assembled out of one channel's signal and the other channel's noise.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import p3frame, placement, rx, spec

FS = rx.FS_DEFAULT
SPS = rx.SPS_DEFAULT

GRID_OFFSET = 18
"""Symbols from an envelope's estimate of a packet's start to grid row 0.

The protocol's own figure is `p3frame.DATA_OFFSET`, nine: a packet is one phase
reference, the eight-symbol header block, and then one row per symbol. What this
constant carries on top of that is the edge detector's own slop, and the two
measurements it has to reach are far apart. A rendered packet's row 0 sits 10.5
symbols past its `bodies` edge on all six speed levels -- nine of that the header
block, the rest the eight-symbol transmit pulse against an edge called two and a
half symbols into the ramp. A capture whose burst never goes quiet puts the edge
wherever the run began rather than at the packet, and then no offset reaches it:
on `PIII_Complete_1` the edge search lands 0.3 s short of the speed-level-1 entry
packet and a window centred here never looks at it.

So this is a window's centre and not a claim, and the envelope is now the LAST
anchor rather than the narrow levels' only one: `header_anchors` places anything
wide enough to carry constant headers and `vh_anchors` places the rest, both to
the sample and neither needing an offset."""

CASE0_OFFSET = GRID_OFFSET - 1
"""...and where the case-0 decoder's own search starts.

One symbol earlier, because `rx.case0_accepts` counts from the pilot a row 0 is
differenced against rather than from row 0 itself."""

CONFIRM_STEP = SPS // 8
"""How far the receiver's clock moves to ask a CRC accept whether it is real.

An eighth of a symbol. Wide enough that noise almost never follows -- an accident
is one draw from a 16-bit lottery and its neighbour is another -- and narrow
enough that a frame can: the matched filter runs over eight symbols, so its
output varies smoothly across one, and a true accept survives a shift this size
at any level where it decodes at all.

Almost, because the blind speed-level-1 chain's filter is 1860 taps and 60
samples sits inside its correlation length, so an accident there can ride an
accepting plateau wider than the step -- the `neg_p3_phantom_sl1` fixture holds
one whose plateau measures 74 samples. Twice in 3,750,342 speed-level-1
positions (`FALSE_RATE`); never once in 1,571,320 at the header-anchored
levels."""

FALSE_RATE = 1 / 60_000
"""Confirmed false accepts per position, on audio carrying no PACTOR: a ceiling.

This constant was first set by rule of three on a sweep that found no confirmed
accepts at all. A larger sweep has since found them. Measured 2026-08-01 over the
corpus negatives plus that day's European 40 m and 80 m captures, per speed level:

    SL1    3,750,342 positions   39 accepts    2 CONFIRMED
    SL2-6  1,571,320 positions   31 accepts    0 confirmed

Both confirmed accepts came out of the blind envelope chain, which only speed
level 1 must always use -- it carries no constant headers, so it can never be
header-anchored -- and one of them is held as the regression fixture
`rf-corpus/regress/neg_p3_phantom_sl1`. It stood red from 2026-08-01 until
`sl1_carriers_present` landed, and it is green because that gate rejects the
accept, not because the scan narrowed: the fixture asserts its own position
count. The rates above were measured WITHOUT the gate and stay as stated -- the
gate closes the one accident that could be re-run, and the second was measured
on audio outside the corpus and has not been, so no smaller rate has been
demonstrated. The earlier sweep (zero confirmed in 179,742 off-air positions and
380,528 of white noise) was not wrong; it had not offered the chain enough
audio.

One constant and not six, because the measurement cannot tell the levels apart:
the SL1 rate is 5.3e-7 per position with a 95% upper bound of 1.7e-6, and zero in
1.57M puts SL2-6 under 1.9e-6 -- at SL1's own rate, an empty SL2-6 column comes up
43% of the time. What separates SL1 is exposure, not a resolved rate: the blind
chain offers it most of the positions, and both accidents were its.

The value stays a ceiling about 10x over both bounds, and the slack is
load-bearing rather than timid. At the measured rate the 19,484-trial burst-wide
SL1 scan would expect 0.010 confirmed -- a hair over `TRIAL_BUDGET` -- where the
ceiling rejects it 32-fold. A rate honest to the last decimal would put the worst
scan this module ever ran back on the line the budget exists to draw.

And what it bounds is one body's search, nothing longer. Every body is charged
against `TRIAL_BUDGET` separately, so a session accumulates exposure no check
ever sees: at the measured rate a 140k-position capture expects ~0.07 confirmed,
about one per fifteen captures of monitoring. A lone SL1 decode is answered by
corroboration -- a connect, a sequence advance, repetition on the 1.25 s raster --
not by this constant."""

TRIAL_BUDGET = 0.01
"""Confirmed false accepts a search may expect and still have found evidence.

600 positions at `FALSE_RATE`. An anchored window is 65 per speed level and a
burst-wide scan of a 39-minute capture is two million, which is the whole
distinction: the first can be believed and the second cannot, and the arithmetic
says so without anyone having to judge the audio.

Charged per candidate body, and never accumulated across a burst, a file or a
session -- `FALSE_RATE` carries what that boundary leaves uncovered."""


def confirmed(decode: Callable[[int], Any], start: int, field: Any) -> bool:
    """`field`, decoded at `start`, survives moving the clock an eighth of a symbol.

    `decode(pos)` returns whatever the CRC validated at `pos`, or None. One
    neighbour agreeing is enough, because a frame at the edge of its window has
    only one.

    What it costs, stated rather than hidden: a rendered speed level 5 frame
    validates at three alignments on a clean channel, two at 3 dB and one at 0 dB,
    so at 0 dB this declines a real one. Nothing below speed level 5 comes close
    to that edge, and a decoder that reports a frame it cannot find twice is the
    thing this module exists to stop.
    """
    return any(decode(start + d) == field
               for d in (-CONFIRM_STEP, CONFIRM_STEP))


TONE_HALFWIDTH_HZ = 45.0
MIN_TONE_EXCESS = 6.0
"""How far a channel must stand over the passband's quiet tenth to count as lit."""

TONE_REL = 0.1
"""...and how far below the strongest channel it may fall. -20 dB.

Both halves are needed and each covers the other's failure. The absolute test
alone breaks on noiseless synthesis, where the quiet tenth is floating-point dust
and every channel clears any multiple of it -- a two-tone speed level 1 packet
read as all eighteen lit. The relative test alone breaks on a fading real signal,
where the weakest tone of a wide level is genuinely 20 dB down on the strongest.

The measured separation is wide enough that the exact figure hardly matters: a
tone's skirt 120 Hz away runs 0.004 to 0.01 of its own peak, and the weakest
channel of a real fourteen-tone burst runs 0.6."""

ENVELOPE_SYMBOLS = 16
"""Analysis window of the burst envelope. Four symbols does not resolve a 120 Hz
grid at all -- 25 Hz bins under a Hann window smear a channel into its neighbours
-- and the detector reported a fourteen-tone burst as empty band."""


@dataclass(frozen=True)
class P3Packet:
    """One CRC-validated PACTOR-III data field."""

    sl: int
    status: int
    payload: bytes
    start: int
    """Sample index of grid row 0, within the buffer that was decoded."""
    trials: int
    """Positions the CRC was offered before this one passed."""
    info: bytes = b""
    """The field's information bytes as they arrived, fill and all.

    `payload` is these with the station's fill taken off, and the fill is
    evidence in its own right: it is what says a field carrying nothing is
    carrying `spec.TEMPLATE` rather than zeros or IDLE, which is the reading
    the entry packet of pactor3.md §17.1 rests on."""

    long_cycle: bool = False
    """Physical frame geometry, independent of the status bit requesting it."""
    carrier_swapped: bool | None = None
    """Physical carrier order of the CRC-valid decode, when measured."""

    @property
    def seq(self) -> int:
        return self.status & spec.STATUS_SEQ

    @property
    def data_type(self) -> int:
        return (self.status >> 2) & 0x7

    def report(self) -> str:
        flags = "".join(f" {n}" for n, m in
                        (("CHANGEOVER", spec.STATUS_CHANGEOVER),
                         ("QRT", spec.STATUS_QRT),
                         ("LONG-CYCLE", spec.STATUS_LONG_CYCLE)) if self.status & m)
        return (f"SL{self.sl} DATA {len(self.payload)}B status=0x{self.status:02x} "
                f"(seq={self.seq} type={self.data_type}{flags}) "
                f"{self.payload[:32]!r}  "
                f"[CRC-VALID, confirmed, {self.trials} trials]")


def packet_of(field: bytes, path: placement.Path, start: int,
              trials: int = 0) -> P3Packet:
    """A CRC-valid field on `path`, cut into status, payload and fill.

    Fill is not payload: the field is a fixed size for the level and both ends
    know it, so a station with less than that writes the rest out. A field that
    is nothing but `spec.TEMPLATE` is empty -- fourteen of PIII_Complete_1's
    are, and this decoder used to hand 58 bytes of walking bits to the host for
    each of them. Trailing IDLE goes the same way: the PACTOR-1 receiver in
    this package has always dropped it (`p1rx.decode_p1_packets`), and an
    independent monitor's own reported lengths follow the same rule --
    pactor1-data-packets.md §6.

    IT CAN TRIM A GENUINE BYTE, and that is the protocol's ambiguity rather than
    this line's: a field whose last payload byte is genuinely 0x1E is
    indistinguishable from a short one. The alternative is handing the host a
    field full of fill it cannot identify, and PACTOR-1 has made the same trade
    here since it was written. The template rule above is not that trade -- a
    whole field of walking bits is not a payload anything wrote.

    `path` is what fixes the field's length, which is why it is an argument and
    not a speed level: the long cycle carries 119 bytes on the level the short
    one carries 24 on.
    """
    info = field[:path.crc_bytes - 2]
    body = bytes(info[:-1])
    return P3Packet(path.speed_level, info[-1], spec.field_payload(body),
                    start, trials, body, path.long_cycle,
                    path.tones != placement.SPEED_PATHS[path.speed_level].tones)


def channel_energy(audio: np.ndarray, fs: int = FS) -> np.ndarray:
    """Energy at each of the eighteen channels, against the passband's quiet tenth.

    Neither of the two obvious references works. The passband MEDIAN fails on the
    wide levels, because an eighteen-tone signal is the passband and every channel
    then measures about one median. The midpoints BETWEEN channels fail too, and
    for a more interesting reason: a 100 Bd tone is some 200 Hz wide at the skirt,
    so on a dense tone set the gaps are as full as the channels -- measured 1.0 to
    1.6 across a live 14-tone burst, which separates nothing.

    What does work is the tenth percentile of the passband. Even speed level 6
    leaves 300-420 and 2580-2900 Hz empty, so a low quantile still lands on noise
    however many tones are up, and a channel that is lit stands over it by an
    order of magnitude.
    """
    if audio.size < 4 * (fs // 100):
        return np.zeros(spec.N_CHANNELS)
    F = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    f = np.fft.rfftfreq(audio.size, 1 / fs)
    floor = np.quantile(F[(f > 300) & (f < 2900)], 0.10) + 1e-12
    return np.array([F[np.abs(f - spec.channel_freq_hz(cn)) < TONE_HALFWIDTH_HZ].max()
                     / floor for cn in range(spec.N_CHANNELS)])


MAX_LEVELS = 4
"""Speed levels tried per packet. Six is affordable and four is enough: the tone
set never ranks the true level worse than third on anything measured here."""


def levels_present(energy: np.ndarray, excess: float = MIN_TONE_EXCESS,
                   max_levels: int = MAX_LEVELS) -> tuple[int, ...]:
    """Speed levels this channel occupancy admits, best fit first.

    It RANKS rather than filters, and the reason is a measurement. A 100 Bd tone
    puts real energy 120 Hz away, so a fourteen-tone speed level 3 burst lights
    sixteen channels -- and a rule that rejected a level for lighting a channel
    outside its set rejected the true level and admitted level 5 instead, on clean
    synthesised audio at full strength.

    So a level is scored by how badly it fits: a channel of its own that is dark
    counts double, one outside it that is lit counts single. Missing more than two
    of its own channels is disqualifying -- two tones cannot be speed level 6
    however the rest of the band reads -- and the rest is an ordering, settled by
    the CRC a few trials later.
    """
    lit = (energy > excess) & (energy > TONE_REL * energy.max())
    scored = []
    for sl, s in spec.SPEED_LEVELS.items():
        want = np.zeros(spec.N_CHANNELS, bool)
        want[list(s.channels)] = True
        missing = int((want & ~lit).sum())
        if missing > 2:
            continue
        scored.append((2 * missing + int((lit & ~want).sum()), sl))
    return tuple(sl for _, sl in sorted(scored)[:max_levels])


def bursts(audio: np.ndarray, fs: int = FS,
           min_s: float = 0.6) -> list[tuple[int, int]]:
    """(start, stop) samples of every stretch carrying multitone energy.

    The envelope is taken on the CHANNEL COMB rather than on the whole passband,
    so a co-channel signal that does not sit on the 120 Hz grid does not extend a
    burst or invent one. `min_s` is a short-cycle data field's own length; nothing
    briefer can carry one.
    """
    sps = fs // 100
    win = ENVELOPE_SYMBOLS * sps
    lvl = []
    for i in range(0, max(1, audio.size - win), sps):
        e = channel_energy(audio[i:i + win], fs)
        lvl.append(float(np.sort(e)[-2]))          # 2nd strongest: one tone is not a comb
    lvl = np.array(lvl)
    on = lvl > MIN_TONE_EXCESS
    out, i = [], 0
    while i < on.size:
        if on[i]:
            j = i
            while j < on.size and on[j]:
                j += 1
            if (j - i) * sps >= min_s * fs:
                out.append((i * sps, min(audio.size, j * sps + win)))
            i = j
        else:
            i += 1
    return out


@dataclass
class Scan:
    """What a decode attempt cost, so that what it found can be weighed."""

    trials: int = 0
    packets: list[P3Packet] = None
    false_rate: float = FALSE_RATE

    def __post_init__(self):
        if self.packets is None:
            self.packets = []

    @property
    def expected_false(self) -> float:
        return self.trials * self.false_rate


def path_for(sl: int, header: p3frame.PacketHeader | None) -> placement.Path:
    """The frame geometry a packet of this speed level is laid out on.

    Two things come from the header and nothing else can supply them: the cycle
    length picks the row count, and the carrier swap says which channel each
    virtual carrier is on THIS cycle. Reading a swapped cycle on the unswapped
    tone order gathers the right energy in the wrong order, which survives every
    check a decoder makes until the CRC.
    """
    if header is None:
        return placement.SPEED_PATHS[sl]
    path = (placement.LONG_PATHS if header.long_cycle else placement.SPEED_PATHS)[sl]
    return dataclasses.replace(path, tones=header.tones(path.tones))


def decode_at(audio: np.ndarray, start: int, sl: int, *,
              fs: int = FS, Z: dict | None = None,
              header: p3frame.PacketHeader | None = None) -> P3Packet | None:
    """One speed level, one alignment. `start` is the sample of grid row 0.

    `Z` is the per-tone baseband, already filtered. Passing it is not an
    optimisation so much as the difference between a usable decoder and an
    unusable one: the matched filter is 1860 taps over the whole buffer, so
    recomputing it per alignment made a twelve-file sweep outrun a ten-minute
    timeout.

    `header` is the packet header the field belongs to, when one was read. With
    it the cycle length, the carrier swap and the constellation's angle are known
    rather than guessed; without it the packet is read on the unswapped tone
    order, the short cycle and the published axes, which is all an envelope can
    offer.
    """
    path = path_for(sl, header)
    rot = 0.0 if header is None else header.rot
    if path.case == 0:
        field, ok = _case0_at(audio, start, rot, fs=fs, Z=Z, header=header,
                              path=path)
    else:
        chan = _cells(audio, start, path, rot, fs=fs, Z=Z)
        if chan is None:
            return None
        field, ok = rx.decode_frame_softs(placement.deinterleave(chan, path), path)
    return packet_of(field, path, start) if ok else None


def _case0_at(audio: np.ndarray, start: int, rot: float, *,
              fs: int = FS, Z: dict | None = None,
              header: p3frame.PacketHeader | None = None,
              path: placement.Path = placement.DETECT) -> tuple[bytes, bool]:
    """The case-0 field whose grid row 0 is at `start`, and whether its CRC passed.

    Case 0 is not `_cells`' shape and never was: it has its own cell order, its
    own permutation and a K=9 trellis, so `decode_frame_softs` -- which is K=7
    throughout -- can only ever have failed on it. What this adds over the blind
    `rx.case0_accepts` is the three things a header block knows and a scan
    cannot: which carrier the swap put cell 0 on, which of the two the split comb
    puts first, and the angle the constellation arrived at.
    """
    sps = fs // 100
    delay = (rx._pulse(sps).size - 1) // 2
    order = (p3frame.VH_ORDER if header is None
             else header.tones(p3frame.VH_ORDER))
    if Z is None:
        pulse = rx._pulse(sps)
        Z = {cn: rx._baseband(audio, cn, fs, pulse) for cn in order}
    softs = rx.case0_softs(Z, start - sps, fs=fs, delay=delay, order=order,
                           rot=rot, path=path,
                           lead=dict(zip(path.tones, path.clock_offsets(sps))))
    return ((b"", False) if softs is None
            else rx.decode_case0_softs(softs, path))


def _cells(audio: np.ndarray, start: int, path: placement.Path, rot: float, *,
           fs: int = FS, Z: dict | None = None) -> np.ndarray | None:
    """One frame's channel-order softs at `start`, or None off the audio's end.

    The demodulation `decode_at` runs, split out so `FieldMemory` can hold the
    same softs a failed decode was made from instead of demodulating its own."""
    if start < 0:
        return None
    sps = fs // 100
    try:
        if Z is None:
            return rx.demod_cells(audio, start, path, fs=fs,
                                  n_rows=path.n_symbols, rot=rot)
        delay = (rx._pulse(sps).size - 1) // 2
        return rx.cells_from_baseband(Z, start, path, fs=fs, delay=delay,
                                      n_rows=path.n_symbols, rot=rot)
    except ValueError:
        return None


def field_rotation(Z: dict, start: int, path: placement.Path, near: float, *,
                   fs: int = FS) -> tuple[float, float]:
    """The constellation angle the FIELD itself sits at, and how much of it agreed.

    A differential cell carries its data in the argument, so raising it to as
    many powers as the cell has states folds every state onto one point and
    leaves the rotation behind -- the estimator `ledger-80m` used to measure the
    peer's carrier offset, turned on the angle instead of the frequency. Where
    the header block offers eight symbols on whatever channels the level lights,
    this offers the whole 72-row field on all of them, and each cell weighs what
    its own amplitude is worth, so a faded carrier neither dominates the sum nor
    is thrown away.

    That matters because the block is the weaker instrument exactly where the
    angle is needed most. On WS8EOC's 80 m arm of 2026-09-13 the six speed
    level 2 carriers stand +0.3 to +1.8 dB over the median channel, its blocks
    read 0.669 to 0.871 against a gate of 0.80, and the angle they report
    scatters over the whole circle -- +24, -135, +25, -129, +9 -- while the
    field returns 41.6 to 45.4 deg on every cycle the packet is readable in.
    The station's angle is one number for the session, and this is what can see
    it.

    The cost of the fold is the ambiguity: the answer is modulo one state's
    turn, half a circle on the DBPSK levels and a quarter on the DQPSK ones.
    `near` resolves it -- the angle a previous frame was read at, or the header
    block's own reading before the session has one. A half turn wrong is every
    bit inverted, so this is the one thing the field cannot say for itself.

    The weight is the sum's magnitude per cell, in the audio's own units. It
    compares two readings of the same buffer and nothing else, which is what
    ranks the two carrier arrangements. On that arm the heavier reading names
    the arrangement the packet was sent in on all eight cycles whose field
    passes a CRC, by 1.6x to 10x, and follows the swap's cycle-by-cycle
    alternation through 31 of the 33 repeats -- the two it does not are the two
    whose weight is lowest, a fortieth of the arm's best. The block's own
    reading of the swap is a coin there at 0.42 to 0.55 variable-header fit.
    """
    sps = fs // 100
    delay = (rx._pulse(sps).size - 1) // 2
    power = 2 ** path.bits_per_cell
    rows = (np.arange(path.n_symbols + 1) - 1) * sps + delay
    offsets = path.clock_offsets(sps)
    total = 0j
    for rank, cn in enumerate(path.tones):
        z = Z[cn]
        idx = start + offsets[rank] + rows
        if idx[0] < 0 or idx[-1] >= z.size:
            return near, 0.0
        y = z[idx]
        d = y[1:] * np.conj(y[:-1])
        total += np.sum(d ** power / (np.abs(d) ** (power - 1) + 1e-12))
    if total == 0:
        return near, 0.0
    # Where the fold lands a frame that arrived on the published axes:
    # `rx.cells_from_baseband` slices DBPSK at -45 and +135 deg, so the square
    # of either is -90; DQPSK sits on all four pi/4 diagonals, whose fourth
    # power is 180.
    total *= 1j if path.bits_per_cell == 1 else -1
    rot = float(np.angle(total)) / power
    turn = 2 * np.pi / power
    return (rot + turn * round((near - rot) / turn),
            float(np.abs(total)) / (path.n_symbols * len(path.tones)))


HEADER_FIT = 0.80
"""Constant-header agreement an anchor must reach to count as a packet.

The sixteen published words are 192 known bits spread over the whole comb, so
this gate is a different kind of thing from an energy detector -- it asks what
the signal SAYS, not how loud it was. It lands in an empty gap rather than on a
judgement call: the 34 real packets of a full PACTOR-III session score 0.88 to
0.99, while 5,646 scanned positions of audio carrying no PACTOR-III -- the three
TI0BCR dwells, four FT8 captures and three of white noise -- reach 0.713 at
their very best. A quarter-symbol sweep of the real session averages 0.61."""


HEADER_COMB_LEVELS = (3, 4, 5, 6)
"""Speed levels whose own constant-header comb an anchor is scored over.

A LEVEL IS NOT CHARGED FOR THE CHANNELS IT DOES NOT LIGHT. `p3frame` averages
the agreement over the channels it scored, so a fourteen-carrier speed level 3
block read across all sixteen constant-header channels caps at about 0.875
before fading takes anything, and K0NTS's gateway of 2026-09-15 -- whose SL3
field decodes CRC-valid, byte-exact against the reference decoder, off two
independent recordings
-- never reached 0.775 on any of twenty-five cycles against this gate. Scored on
its own twelve channels the same block clears it.

WHICH COMBS ARE SWEPT IS MEASURED, not assumed, and the gate is untouched. Over
60 s of white noise and three off-air captures carrying no PACTOR-III, 43,000
quarter-symbol positions, the best fit any comb reaches is 0.710 on sixteen
channels, 0.727 on fourteen and 0.736 on twelve -- and 0.837 on the FOUR that
speed level 2 lights, which is above `HEADER_FIT` and would make an anchor out
of noise. So the narrow levels keep the readers they already had: speed levels 1
and 2 carry the variable header and are found by `vh_anchors` and the envelope,
exactly as before."""

HEADER_COMBS = tuple(sorted(
    {tuple(c for c in p3frame.CH_CHANNELS
           if c in placement.SPEED_PATHS[sl].tones)
     for sl in HEADER_COMB_LEVELS}, key=len))
"""The distinct constant-header channel sets `HEADER_COMB_LEVELS` lights."""


def _header_diffs(Z: dict, at: int, tones: tuple[int, ...],
                  offsets: tuple[int, ...], sps: int, delay: int) -> dict | None:
    """One header block's unit differentials, each carrier read on its own clock.

    None if the block runs off the end of the baseband. `offsets` is rank-indexed
    beside `tones`, so a level whose comb is staggered is sampled where its
    carriers actually are.
    """
    idx = at + np.arange(p3frame.HEADER_SYMBOLS + 1) * sps + delay
    diffs = {}
    for cn, off in zip(tones, offsets):
        z = Z[cn]
        j = idx + off
        if j[0] < 0 or j[-1] >= z.size:
            return None
        y = z[j]
        d = y[1:] * np.conj(y[:-1])
        diffs[cn] = d / (np.abs(d) + 1e-12)
    return diffs


def header_of(Z: dict, starts: Iterable[int], path: placement.Path, *,
              fs: int = FS) -> p3frame.PacketHeader | None:
    """The header of the packet whose grid row 0 is one of `starts`, or None.

    `header_anchors` is acquisition -- it has no timing and no speed level, so it
    sweeps the whole comb on one clock. A receiver tracking an established cycle
    has the opposite problem: it knows the level and knows row 0 to within a few
    symbols, and what it is missing is the two things only the header carries --
    the carrier swap and the cycle length. So this reads the block the level's own
    geometry puts in front of each candidate row 0 and returns the best-fitting
    one, leaving the CRC to say whether the packet is there at all.

    BOTH STAGGERS are read, and the swap is why: it relabels a virtual carrier's
    RANK onto a partner tone, and rank is what `spec.SUBBAND_LEAD` indexes, so on
    speed level 2 the swap moves the symbol clock as well as the carrier. The tone
    SET does not move -- every speed level's is closed under `spec.CARRIER_SWAP`
    -- so what separates the two is entirely which cluster leads. Measured on a
    rendered level 2 packet: read on the arrangement it was sent in the header
    fits 0.99, read on the other one it never passes 0.81, because all six
    carriers are then sampled half a symbol out. Levels with no lead read the same
    both ways, which costs one 16-by-8 correlation and buys one code path.

    `starts` should be finer than a symbol. The fit falls off inside one -- 0.99
    at the true instant, 0.93 a quarter symbol away, 0.62 half a symbol away, on
    an anchor gate of 0.80 -- while the CRC tolerates the whole symbol, so a
    caller stepping by symbols would put the reading below the gate exactly where
    the frame still decodes.

    The gate itself is `anchor_gate(path)`, because a two-channel level carries
    no constant words to score and `read_header` falls back to the variable
    header there.
    """
    sps = fs // 100
    delay = (rx._pulse(sps).size - 1) // 2
    offsets = path.clock_offsets(sps)
    arrangements = [path.tones,
                    tuple(spec.CARRIER_SWAP[c] for c in path.tones)]
    best = None
    for start in starts:
        at = start - p3frame.DATA_OFFSET * sps
        for tones in arrangements:
            diffs = _header_diffs(Z, at, tones, offsets, sps, delay)
            if diffs is None:
                continue
            vh, fit, rot, swapped = p3frame.read_header_arrangement(diffs)
            if best is None or fit > best.fit:
                best = p3frame.PacketHeader(at, vh, fit, rot, swapped)
    return best


def header_anchors(Z: dict, span: int, *, fs: int = FS,
                   step: int | None = None) -> list[p3frame.PacketHeader]:
    """Every packet header block in `span` samples of already-filtered baseband.

    EACH SPEED LEVEL'S COMB IS SCORED ON ITS OWN, because the sweep does not yet
    know which level it is looking at and an average over channels a level never
    lights is an average over noise -- `HEADER_COMBS` carries the arithmetic and
    the measurement. The gate is the same 0.80 for every one of them.

    This is acquisition by known codeword rather than by envelope, and it is what
    replaced a timing search. An envelope places a burst to within a few symbols
    and leaves the rest to the CRC; the header block places it exactly, and says
    which speed levels to try, whether the cycle is long, and which way the
    carrier swap fell -- none of which an envelope knows and the last two of
    which no amount of searching recovers.
    """
    sps = fs // 100
    step = step or sps // 4
    delay = (rx._pulse(sps).size - 1) // 2
    need = p3frame.DATA_OFFSET * sps + delay
    tones = tuple(Z)
    flat = (0,) * len(tones)
    out: list[p3frame.PacketHeader] = []
    for at in range(0, max(0, span - need), step):
        diffs = _header_diffs(Z, at, tones, flat, sps, delay)
        if diffs is None:
            break
        corrs = p3frame.constant_header_corrs(diffs)
        fit, rot = max((p3frame.constant_header_fit(corrs, comb)
                        for comb in HEADER_COMBS), key=lambda f: f[0])
        if fit >= HEADER_FIT:
            vh, _, _, swapped = p3frame.read_variable_header(diffs)
            out.append(p3frame.PacketHeader(at, vh, fit, rot, swapped))
    # One packet lights every position within half a symbol of its own, so keep
    # the best of each cluster rather than decoding the same field four times.
    peaks: list[p3frame.PacketHeader] = []
    for h in sorted(out, key=lambda h: -h.fit):
        if all(abs(h.at - p.at) > sps for p in peaks):
            peaks.append(h)
    return sorted(peaks, key=lambda h: h.at)


VH_FIT = 0.90
"""Variable-header agreement a narrow-level anchor must reach.

Sixteen published VARIABLE headers are 32 known bits on the two carriers every
speed level lights -- which is what the narrow levels have instead of the 192
`HEADER_FIT` scores. Six times less evidence, so it is a separate constant and a
much higher one, and where it sits is measured at quarter-symbol resolution:

  * real narrow-level packets reach 0.93 and up -- the four speed-level-1 entry
    packets to hand read 0.967 to 0.985, DL6MAA's at 0.977 and our own three arms
    through a web receiver, the 45 packets of DL6MAA's whole session 0.950 to
    0.994, and the `Q-6.0` packet on 7101 kHz 0.9349 with its ARQ repeat one
    cycle later at 0.9385;
  * audio carrying no PACTOR-III does not reach 0.86: nine captures -- the corpus
    negatives, three FT8 captures, a VARA session, the held speed-level-1 phantom
    -- and a minute of white noise top out at 0.855.

This stood at 0.95 on the reading that the 0.938 cluster in `7101k_234600` was
noise, and that the CRC-valid five-byte payload appearing one trial below the
threshold was the coincidence that fixed the number. Those five bytes are
`Q-6.0`; they survive moving the clock, and an independent monitor reads the same
five off the same audio three times over. The accept was the packet, and the
constant was set to exclude it. What 32 bits buys is the 0.855-to-0.935 gap, and
this sits in the middle of it.

An anchor is not a decode -- everything behind it still faces the CRC, `confirmed`
and `sl1_carriers_present`."""


def anchor_gate(path: placement.Path) -> float:
    """The fit a header block read on `path`'s comb has to reach to be believed.

    The two thresholds are scored over 192 bits and 32 and are never weighed
    against each other; which one applies is a property of the comb the block was
    read on, and `p3frame.read_header` reports whichever it could measure."""
    return VH_FIT if len(path.tones) <= len(spec.VH_CHANNELS) else HEADER_FIT


def vh_anchors(Z: dict, span: int, *, fs: int = FS,
               step: int | None = None) -> list[p3frame.PacketHeader]:
    """`header_anchors` for the levels whose comb is too narrow to match it.

    Speed level 1 lights two channels and speed level 2 six, so between them they
    carry none and four of the sixteen constant headers -- `HEADER_FIT` is scored
    over sixteen channels and neither can ever reach it. But BOTH carry the
    variable header, and so does every level above them: channels 5 and 12 are in
    every speed level's tone set, and the eight dibits they hold are one of
    sixteen published 32-bit words. That is a matched filter on a codeword, which
    is what an envelope search never was -- it places the packet to the sample,
    names the speed level modulo four, the cycle length and the carrier swap, and
    hands the field the angle the constellation arrived at.

    The returned `fit` is agreement with the VARIABLE header rather than the
    constant ones, so it is weighed against `VH_FIT`.

    BOTH CARRIERS ON ONE CLOCK IS ONLY ONE OF THREE HYPOTHESES. The two narrow
    levels are exactly the two `spec.SUBBAND_LEAD` splits, half a symbol, and the
    carrier swap decides which of the pair leads -- so a flat read samples one of
    them half a symbol off its own clock through the whole block. Measured on a
    rendered level 1 packet: flat reads 0.905 unswapped and 0.861 swapped, either
    side of the gate, and both senses tried reads 0.989. It costs three matched
    filters instead of one and buys nothing an accident could use: over 60 s of
    white noise the best fit moves from 0.823 to 0.828, against a gate of 0.90.
    DL6MAA's real entry packet reads 0.994 flat and 0.997 this way.
    """
    sps = fs // 100
    step = step or sps // 4
    delay = (rx._pulse(sps).size - 1) // 2
    lead = sps // 2
    need = p3frame.DATA_OFFSET * sps + delay + lead
    n = max(0, (min(span, min(Z[c].size for c in spec.VH_CHANNELS)) - need) // step)
    if not n:
        return []
    at = np.arange(n) * step
    idx = at[:, None] + np.arange(p3frame.HEADER_SYMBOLS + 1) * sps + delay
    unit = {}
    for cn in spec.VH_CHANNELS:
        for off in (0, lead):
            y = Z[cn][idx + off]
            d = y[:, 1:] * np.conj(y[:, :-1])
            unit[cn, off] = d / (np.abs(d) + 1e-12)
    best = np.zeros(n, complex)
    vh = np.zeros(n, int)
    swapped = np.zeros(n, bool)
    clocks = [{}] + [{cn: lead} for cn in spec.VH_CHANNELS]
    for order in (spec.VH_CHANNELS, spec.VH_CHANNELS[::-1]):
        for clock in clocks:
            # The dibits alternate between the two carriers, so the block reads in
            # cell order -- `read_header`'s own interleave, batched over positions.
            # The clock is per CHANNEL and the cell order per virtual carrier, so
            # the swap moves the two independently and both are swept.
            v = np.stack([unit[c, clock.get(c, 0)] for c in order],
                         2).reshape(n, -1)
            corr = v @ p3frame.VARIABLE_TEMPLATES.conj().T / v.shape[1]
            i = np.abs(corr).argmax(1)
            c = corr[np.arange(n), i]
            take = np.abs(c) > np.abs(best)
            best, vh = np.where(take, c, best), np.where(take, i, vh)
            swapped = np.where(take, order != p3frame.VH_ORDER, swapped)
    out = [p3frame.PacketHeader(int(at[k]), int(vh[k]), float(np.abs(best[k])),
                                float(np.angle(best[k])), bool(swapped[k]))
           for k in np.flatnonzero(np.abs(best) >= VH_FIT)]
    peaks: list[p3frame.PacketHeader] = []
    for h in sorted(out, key=lambda h: -h.fit):
        if all(abs(h.at - p.at) > sps for p in peaks):
            peaks.append(h)
    return sorted(peaks, key=lambda h: h.at)


PACKET_S = 0.9
"""How much of a short cycle a data field occupies -- the closest two packet
bodies can be."""

LEVEL_WINDOW_S = 0.5
"""Span of data field the tone set is read from, starting at grid row 0.

Well inside the field at both ends, and that is the whole point. A window that
overruns into the next packet, or back into a link's +-800 acquisition burst,
reads the wrong occupancy: the burst is wideband, so it reports every channel lit
and ranks a six-tone speed level 2 below three levels it cannot possibly be."""

TIMING_SYMBOLS = 10
"""Symbols either side of `GRID_OFFSET` that are tried.

The envelope places a burst to within a few symbols on quiet audio and to within
a couple of dozen on a channel that never goes quiet, and this is the slack that
covers both -- the two measurements behind `GRID_OFFSET` are 16.5 symbols apart.

It is also the whole trial budget, which is why it is a number rather than a
shrug: twenty symbols at a quarter symbol is 81 chances at the CRC per level
and at most 324 for the four levels a body is offered, against the 24,000 a blind
scan of the same window would take and the 600 the budget allows.
"""


def bodies(audio: np.ndarray, lo: int, hi: int, *, fs: int = FS) -> list[int]:
    """Where each packet in a burst starts: the rising edges of its envelope.

    There is nothing to correlate against on the two narrow levels: a packet
    opens on a SINGLE phase-reference symbol -- no repeated pilot block -- and
    their header blocks carry too few of the published words to match. So the
    anchor is the edge itself, refined by the CRC over the few alignments
    `TIMING_SYMBOLS` allows.

    The edge is taken on a one-symbol running RMS rather than on the comb
    detector's own envelope: that one integrates sixteen symbols, so it reports an
    onset up to fifteen symbols early, which is further than any affordable
    timing search reaches. Edges are taken no closer together than a packet is
    long, so one burst carrying one packet does not spend the trial budget on a
    dozen alignments of it.
    """
    sps = fs // 100
    seg = audio[lo:hi]
    if seg.size < 12 * sps:
        return []
    step = sps // 4
    n = (seg.size - sps) // step
    rms = np.sqrt(np.array([np.mean(seg[i * step:i * step + sps] ** 2)
                            for i in range(n)]))
    on = rms > 0.25 * np.quantile(rms, 0.9)
    out, guard = [], int(PACKET_S * fs)
    for i in range(n):
        if on[i] and (not out or lo + i * step - out[-1] >= guard):
            out.append(lo + i * step)
    return out


SL1_CARRIER_EXCESS = 2.0
"""How far over its own channel's quiet fifth the weaker carrier must stand..."""

SL1_CARRIER_BALANCE = 0.5
"""...or how close to the stronger carrier it may run. Clearing either is enough.

A speed level 1 field is a rate-1/2 code over 144 cells split evenly between
channels 5 and 12, so a genuine field cannot arrive on one of them: with channel
5 notched out entirely, a rendered packet's own field decodes at no alignment.
What CAN arrive on one channel is an accident. The confirmed accept held as
`rf-corpus/regress/neg_p3_phantom_sl1` rides skirt energy that an off-grid
neighbour near 1950-1990 Hz puts inside channel 12's window while channel 5 runs
at its own noise floor -- and it cleared the clock-shift rule, whose step sits
inside the 1860-tap matched filter's correlation length.

Both constants act on one statistic: the field's median |Z| against the same
channel's 20th-percentile |Z|, over the segment the decode was made from. The
excess threshold sits above what NO signal reads -- on a channel carrying only
stationary noise the statistic is a property of the noise's distribution shape,
not of its level, and settles near the Rayleigh figure of 1.76 (measured
1.49-1.89 across eight alignments of a rendered packet with channel 5 replaced
by noise). An accept is rejected only when it fails BOTH prongs, because each
keeps a real packet the other would refuse. Measured, through the production
segmentation:

  * the held accident: weaker-carrier excess 1.39, balance 0.34. Fails both.
  * the one real off-air case-0 header the corpus holds (KE5YTA, the fixture
    `pos_pactor1_local`, t=12.72): excess 2.19, balance 0.71. Clears both.
  * a rendered packet under one-sided fade on channel 5, swept 6 to 30 dB down
    at 3 and 10 dB SNR: 511 accepts, every one carrying the true field and none
    manufactured. Balance falls as low as 0.03, and down to 21 dB of fade the
    weaker carrier's excess never reads below 2.20 -- the excess prong keeps
    what the balance prong would refuse. WHAT THE GATE COSTS lives below that:
    at 24 to 30 dB of one-sided fade the true field still decodes, its excess
    reads 1.80-4.26, and the accepts under the threshold are refused with the
    accidents.
  * a packet weak on BOTH carriers keeps them comparable, which is the balance
    prong. An accident assembled from noise on BOTH channels is not this gate's
    to catch: it stands at a single alignment, and `confirmed` already rejects
    it -- three raw accepts in 476k noise positions, never once confirmed."""


def sl1_carriers_present(Z: dict, start: int, *, fs: int = FS) -> bool:
    """The case-0 accept whose pilot sits at `start` put energy on BOTH channels.

    `Z` and `start` are the baseband and alignment the accept was decoded from,
    so the question is asked of the same samples the CRC was. It is not a
    threshold on how loud the signal was: a field that genuinely decodes clears
    it at any depth measured (the constants above), and what it refuses is a
    field assembled from one channel's signal and the other channel's noise.
    """
    sps = fs // 100
    idx = start + np.arange(73) * sps + (rx._pulse(sps).size - 1) // 2
    exc = []
    for cn in rx.HEADER_TONES:
        z = np.abs(Z[cn])
        exc.append(float(np.median(z[idx]) / (np.quantile(z, 0.20) + 1e-12)))
    return min(exc) >= SL1_CARRIER_EXCESS \
        or min(exc) >= SL1_CARRIER_BALANCE * max(exc)


def _case0_in(seg: np.ndarray, window: range, scan: Scan, Z: dict,
              fs: int) -> P3Packet | None:
    """The first confirmed case-0 frame in `window`, charging `scan` for the search.

    Speed level 1 is the only case-0 path and it is the one that manufactured the
    corpus's phantoms, because it used to be scanned across a whole burst rather
    than across a window an envelope had placed. An accept here answers two
    questions beyond the CRC: it survives moving the clock, and it lit both of
    its carriers.
    """
    def at(pos: int) -> bytes | None:
        for _, f in rx.case0_accepts(seg, fs=fs, search=range(pos, pos + 1), Z=Z):
            return f
        return None

    for start, field in rx.case0_accepts(seg, fs=fs, search=window, Z=Z):
        if confirmed(at, start, field) and sl1_carriers_present(Z, start, fs=fs):
            scan.trials += 1 + (start - window.start) // window.step
            body = bytes(field[:5])
            return P3Packet(1, field[5], spec.field_payload(body), start,
                            scan.trials, body)
    scan.trials += len(window)
    return None


def _level_in(seg: np.ndarray, sl: int, window: range, scan: Scan, Z: dict,
              fs: int) -> P3Packet | None:
    """The first confirmed SL>=2 frame in `window`, charging `scan` for the search."""
    def at(pos: int) -> tuple | None:
        p = decode_at(seg, pos, sl, fs=fs, Z=Z)
        return None if p is None else (p.status, p.payload)

    for start in window:
        scan.trials += 1
        p = decode_at(seg, start, sl, fs=fs, Z=Z)
        if p is not None and confirmed(at, start, (p.status, p.payload)):
            return dataclasses.replace(p, trials=scan.trials)
    return None


def decode_body(audio: np.ndarray, body_at: int, *,
                levels: tuple[int, ...] | None = None, fs: int = FS) -> Scan:
    """The data field of the packet whose body starts at `body_at`."""
    sps = fs // 100
    scan = Scan()
    # The tone set is read from INSIDE the data field, not from the packet as a
    # whole: a window that catches a link's +-800 acquisition burst reports every
    # channel lit, and a six-tone speed level 2 then ranks below three levels it
    # cannot possibly be.
    on = audio[body_at + GRID_OFFSET * sps:
               body_at + GRID_OFFSET * sps + int(LEVEL_WINDOW_S * fs)]
    cand = levels if levels is not None else levels_present(channel_energy(on, fs))
    if not cand:
        return scan
    pulse = rx._pulse(sps)
    tones = sorted({cn for sl in cand for cn in placement.SPEED_PATHS[sl].tones}
                   | set(rx.HEADER_TONES))
    seg_lo = max(0, body_at - TIMING_SYMBOLS * sps)
    seg = audio[seg_lo:body_at + int(1.1 * fs)]
    Z = {cn: rx._baseband(seg, cn, fs, pulse) for cn in tones}
    body = body_at - seg_lo

    def window(offset: int) -> range:
        row0 = body + offset * sps
        return range(row0 - TIMING_SYMBOLS * sps, row0 + TIMING_SYMBOLS * sps + 1,
                     sps // 4)

    for sl in cand:
        p = (_case0_in(seg, window(CASE0_OFFSET), scan, Z, fs) if sl == 1 else
             _level_in(seg, sl, window(GRID_OFFSET), scan, Z, fs))
        if p is None:
            continue
        # Found, and the only question left is whether the search that found it
        # was short enough to mean anything. Either way this body is finished:
        # scanning the remaining levels can only lengthen the search.
        if scan.expected_false <= TRIAL_BUDGET:
            scan.packets.append(dataclasses.replace(
                p, start=seg_lo + p.start))
        break
    return scan


MEMORY_COPIES = 4
"""Frame copies `FieldMemory` holds at most, oldest dropped first.

`p2rx.MEMORY_COPIES`' bound, for its reasons: it caps what combining offers the
CRC, and it is how a mis-grouped copy leaves -- a field boundary the memory
missed poisons the sum, the CRC refuses it, and the sliding window ages the
stale copy out within this many cycles. A fade that outlasts four cycles has
the link changing speed level anyway, which changes the geometry and resets the
memory through `add`'s key check."""


class FieldMemory:
    """Soft combining across repeats of one unacknowledged PACTOR-III field.

    An unacked field is transmitted again, the same bytes under the same mod-4
    counter, until it is acked; a receiver that decodes each copy alone throws
    the earlier copies away. This sums the channel-order softs of consecutive
    failed HEADER-ANCHORED frames before the trellis sees them --
    `rx.decode_frame_softs` has always taken soft input, so combining is a
    buffer in front of the existing chain, which is exactly what
    `p2rx.BurstMemory` is to PACTOR-2.

    What makes the sum meaningful:

      * copies are grouped by CONSECUTIVE FAILURE within one GEOMETRY -- speed
        level and row count -- never by sequence number, which is exactly what
        an undecoded frame cannot supply. A peer that drops a speed level on a
        retransmission produces a copy whose layout is not combinable, and the
        key check resets the memory instead of mixing layouts.
      * each copy arrives through its own packet header: the header block
        names the carrier swap, so the softs are already in channel-rank order
        (`path_for` rewrote the tones), and its `rot` has taken the cycle's
        residual carrier phase off the constellation. Copies from alternate
        ARQ cycles ride opposite tones and different phases, and sum cell for
        cell regardless -- the PACTOR-III analogue of `p2rx.BurstMemory`'s
        lane mapping and per-lane de-rotation, done here by the header the
        anchor already read.
      * the softs are phase-only projections in [-1, 1], so every copy weighs
        the same and no normalisation exists to need.

    Only speed levels 2 and up can reach this memory: level 1 rides the case-0
    chain, whose false-accept history has its own gates
    (`sl1_carriers_present`), and its frames are never header-anchored at all.

    Trials the CRC is offered: ONE decode of the sum per `add` that holds two
    copies or more, charged to the caller's `Scan` like any other position, and
    corroborated like every accept in this module -- the copies carry their
    neighbours' softs an eighth of a symbol either side, so the combined accept
    must survive moving the clock, or else survive losing its oldest copy;
    `add` says what each is worth and `FALSE_RATE`'s arithmetic covers one
    position either way. A lone failure costs nothing: its only copy was
    already decoded single-shot.
    """

    def __init__(self) -> None:
        self._key: tuple[int, int] | None = None
        self._copies: list[dict[int, np.ndarray]] = []

    def clear(self) -> None:
        self._copies.clear()

    def add(self, cells: dict[int, np.ndarray],
            path: placement.Path) -> bytes | None:
        """Absorb one undecoded frame's softs; return the combined field, if any.

        `cells` maps clock offset -- 0 and +-`CONFIRM_STEP` -- to the frame's
        channel-order softs at that offset, all demodulated from the copy's own
        anchor. A CRC-valid, corroborated combined decode clears the memory:
        the field is delivered, and whatever follows stands alone again.

        TWO CORROBORATIONS, EITHER OF WHICH IS ONE. Moving the clock is the
        module's own rule and stands; a combined accept may also answer by
        surviving the loss of its OLDEST COPY, which is a different set of air
        rather than a different slicing of the same set. Both are asked for the
        same thing -- the same field twice -- and the second is what a sum at
        the edge of its margin can give: on WS8EOC's 80 m arm of 2026-09-13,
        where the clock shift confirms none of the 22 four-copy sums that carry
        the peer's greeting, dropping the oldest copy confirms 20 of them and
        the other two are the ones whose newest copy is noise. It needs three
        copies to mean anything, the sum of two minus its oldest being the
        single frame that already failed its own CRC."""
        key = (path.speed_level, path.n_symbols)
        if key != self._key:
            self._key = key
            self._copies.clear()
        self._copies.append(cells)
        del self._copies[:-MEMORY_COPIES]
        if len(self._copies) < 2:
            return None

        def at(off: int, copies: list | None = None) -> bytes | None:
            comb = np.sum([c[off] for c in (copies or self._copies)], axis=0)
            field, ok = rx.decode_frame_softs(
                placement.deinterleave(comb, path), path)
            return field if ok else None

        field = at(0)
        if field is None:
            return None
        if not (any(at(off) == field for off in (-CONFIRM_STEP, CONFIRM_STEP))
                or (len(self._copies) > 2 and at(0, self._copies[1:]) == field)):
            return None
        self.clear()
        return field


def decode_headed(audio: np.ndarray, lo: int, hi: int, *,
                  fs: int = FS, levels: tuple[int, ...] | None = None,
                  memory: FieldMemory | None = None) -> tuple[Scan, int]:
    """Every packet in a burst that opens with a header block.

    Returns the scan and how many anchors were found, because those are separate
    answers: anchors and no decode means a real PACTOR-III burst this receiver
    could not read, which is worth distinguishing from audio that never carried
    one. A caller that falls back to an envelope search must not do so on the
    first, or it will spend a blind sweep's trial budget on a signal whose timing
    was already known exactly.

    `memory` is a `FieldMemory` held across calls -- ARQ repeats arrive in
    consecutive cycles, usually consecutive bursts. A delivered frame clears
    it; an anchored frame that fails every admissible level feeds it one copy
    at the anchor's best-fitting geometry, and a combined decode joins
    `scan.packets` charged like any other position.
    """
    sps = fs // 100
    scan = Scan()
    pulse = rx._pulse(sps)
    seg = audio[lo:min(len(audio), hi + int(spec.CYCLE_LONG_S * fs))]
    Z = {cn: rx._baseband(seg, cn, fs, pulse) for cn in range(spec.N_CHANNELS)}
    # The variable header anchors the narrow levels and the constant ones anchor
    # everything else. A packet wide enough for both is kept on the
    # sixteen-channel reading, which is the better-evidenced of the two; a
    # narrow one has only the other, and a session that changes speed level
    # carries both kinds, so it is not a choice made once for the burst.
    heads = header_anchors(Z, hi - lo, fs=fs)
    heads += [h for h in vh_anchors(Z, hi - lo, fs=fs)
              if all(abs(h.at - w.at) > sps for w in heads)]
    heads.sort(key=lambda h: h.at)
    for h in heads:
        row0 = h.at + p3frame.DATA_OFFSET * sps
        # The channel occupancy is the only thing that separates speed level 1
        # from 5 and 2 from 6: the header carries the level modulo four.
        on = seg[row0:row0 + int(LEVEL_WINDOW_S * fs)]
        rank = levels_present(channel_energy(on, fs), max_levels=spec.N_CHANNELS)
        cand = [sl for sl in ([sl for sl in rank if sl in h.levels]
                              or list(h.levels))
                if levels is None or sl in levels]
        delivered = False
        for sl in cand:
            scan.trials += 1
            p = decode_at(seg, row0, sl, fs=fs, Z=Z, header=h)
            if p is None:
                continue
            def at(pos: int, sl: int = sl) -> tuple | None:
                q = decode_at(seg, pos, sl, fs=fs, Z=Z, header=h)
                return None if q is None else (q.status, q.payload)
            if confirmed(at, row0, (p.status, p.payload)):
                scan.packets.append(dataclasses.replace(
                    p, sl=sl, start=lo + row0, trials=scan.trials))
                delivered = True
            break
        if delivered:
            if memory is not None:
                memory.clear()
        elif memory is not None and cand and cand[0] >= 2:
            # The copy is held at the anchor's own best-fitting level: the
            # level is read off the channel occupancy, which a frame too weak
            # to pass its CRC still lights, so consecutive repeats group under
            # one geometry. A wrong guess is a wrong geometry, and the key
            # check plus the CRC fail it safe. Level 1 never reaches here --
            # see the class docstring.
            path = path_for(cand[0], h)
            cells = {off: _cells(seg, row0 + off, path, h.rot, fs=fs, Z=Z)
                     for off in (-CONFIRM_STEP, 0, CONFIRM_STEP)}
            if all(c is not None for c in cells.values()):
                scan.trials += 1
                field = memory.add(cells, path)
                if field is not None:
                    scan.packets.append(
                        packet_of(field, path, lo + row0, scan.trials))
    return scan, len(heads)


CHANGEOVER_SEARCH = 8
"""Eighth-symbols either side of a CS3 head that a changeover frame is looked for.

The head's own alignment comes from the codeword, which is a matched filter on the
same two carriers, so this is not an acquisition window -- it is the slack between
where the codeword reads best and where the frame reads best. The codeword is read
on ONE clock -- twenty bits summed over both carriers, which is what
`rx.nearest_control_signal` is handed -- so a split comb puts the head at the
midpoint of the two while the frame behind it is read on the earlier carrier's,
and the two disagree by half of the split.

MEASURED, on every changeover packet there is. The three in `PIII_Complete_1`
accept from -5 to +2 -- two on the home arrangement and the third on the swapped
one -- and our own render, whose head the same `rxfront._best_cs` places, from -2
to +4. Six eighths left the real material one step of slack at the bottom; eight
leaves three at each end for two milliseconds and eight more trials, and those
trials cost nothing that was being spent: 0 accepts in 400 windows of white noise
and 400 of off-air audio carrying no PACTOR-III, at either width.

The seventeen alignments on each carrier arrangement are spent only where a
codeword has already read at zero bit errors of twenty."""


def decode_changeover_details(audio: np.ndarray, head_at: int, *, fs: int = FS,
                             Z: dict | None = None) -> tuple[bytes, bool, bool | None]:
    """CS3 field, CRC result, and the confirmed physical carrier arrangement.

    The arrangement is None when no field validates; it is not inferred from
    the status counter or a transmit-grid parity.

    `placement.changeover_packet`'s opposite number: the codeword is the packet's
    first twenty symbols and `placement.CHANGEOVER_HEAD_SYMBOLS` later the frame's
    phase reference follows, in the same keying. A receiver that reads the
    codeword and stops has thrown away the three bytes the new sender put behind
    it -- the head of DL6MAA's greeting, in the recording this is fitted to.

    Both carrier arrangements are tried because a changeover packet carries no
    header block to name the swap, and the frame is a two-carrier one, so the
    swap is exactly an exchange of the two.

    AND THE ARRANGEMENT IS THE CLOCK, which is what makes this two sweeps rather
    than four. `spec.SUBBAND_LEAD` splits the comb half a symbol, and the lead
    belongs to the virtual carrier -- so the arrangement that says which tone
    carries cell 0 also says which tone runs half a symbol behind, and a sweep
    has nothing left to guess. Read on one clock the frame's optimum sits at the
    midpoint of the two carriers and the CRC still passes on the bench: measured
    against the render `placement.changeover_packet` keys, forty seeds on each
    arrangement, the one-clock read delivers 48 of 80 at -18 dB where each
    carrier on its own clock delivers 78, and 2 of 80 at -20 dB against 39 --
    a shade under 2 dB at the cliff. On the real material the sweep accepts at
    four or five alignments read on one clock and at seven or eight read on
    two, the same field either way.
    """
    sps = fs // 100
    pulse = rx._pulse(sps)
    delay = (pulse.size - 1) // 2
    if Z is None:
        Z = {cn: rx._baseband(audio, cn, fs, pulse) for cn in spec.VH_CHANNELS}
    path = placement.CHANGEOVER
    base = head_at + placement.CHANGEOVER_HEAD_SYMBOLS * sps

    for swapped in (False, True):
        order = p3frame.VH_ORDER[::-1] if swapped else p3frame.VH_ORDER
        tones = (tuple(spec.CARRIER_SWAP[cn] for cn in path.tones) if swapped
                 else path.tones)
        lead = dict(zip(tones, path.clock_offsets(sps)))

        def at(pos: int, order=order, lead=lead) -> bytes | None:
            softs = rx.case0_softs(Z, pos, fs=fs, delay=delay, order=order,
                                   path=path, lead=lead)
            if softs is None:
                return None
            field, ok = rx.decode_case0_softs(softs, path)
            return field if ok else None
        for k in range(-CHANGEOVER_SEARCH, CHANGEOVER_SEARCH + 1):
            pos = base + k * CONFIRM_STEP
            field = at(pos)
            if field is not None and confirmed(at, pos, field):
                return field, True, swapped
    return b"", False, None


def decode_changeover(audio: np.ndarray, head_at: int, *, fs: int = FS,
                      Z: dict | None = None) -> tuple[bytes, bool]:
    """Compatibility API: decoded CS3 field and CRC result.

    `decode_changeover_details` also exposes the measured carrier arrangement
    for receivers that must align a reply. Existing field-only callers keep
    their two-value result.
    """
    field, ok, _ = decode_changeover_details(audio, head_at, fs=fs, Z=Z)
    return field, ok


def decode_p3_packets(audio: np.ndarray, *, fs: int = FS,
                      levels: tuple[int, ...] | None = None,
                      envelope: bool = True) -> Scan:
    """Every PACTOR-III data field in `audio`, with the cost of finding them.

    Four things stand in front of the CRC and none of them is a threshold on how
    loud the signal was: multitone energy has to be present at all, the published
    packet headers have to be there to be matched, the channels the speed level
    uses have to be the channels that are lit, and the accept has to survive
    moving the clock. A capture with none of that is never scanned, so it cannot
    produce a decode -- which is what the corpus's fifteen phantom PACTOR-3
    headers were missing.

    A real packet costs the CRC one position per admissible speed level rather
    than a window of them, because the header block has already said where the
    grid starts. A whole PACTOR-III session -- 34 packets across five speed
    levels -- is 55 trials, about a tenth of the budget.

    Every packet is found inside a window an envelope placed. Speed level 1 used
    to be exempt, on the grounds that its trellis is cheap enough to sweep a whole
    burst with; that exemption cost 19,484 trials on one FT8 capture and returned
    a five-byte payload from a band no PACTOR station transmits on. Cheap to run
    is not the same as cheap to believe.

    `envelope` is the second anchor -- the rising edge, for a burst that carries
    no header block -- and a caller on a clock turns it off. The two are not
    comparable in cost. The header anchor knows where the grid starts, so it
    charges the CRC one position per admissible level; the edge knows only that
    something started, so it spends the whole of `TIMING_SYMBOLS` on every level
    the burst could be, whether or not the burst is PACTOR at all. MEASURED over
    a 1.25 s window of real off-air audio carrying no PACTOR-III: 84 ms for the
    headed pass against 1.85 s for the fallback behind it, which returns nothing.
    Nor is the fallback what finds the real thing -- every packet in the two
    PACTOR-III recordings the corpus holds (DL6MAA's eleven across levels 3, 4
    and 5, and PIII_18's single level 6 burst) is anchored on its header block
    and none on an edge. What it is FOR is the narrow levels, whose packets open
    on a single phase-reference symbol and so have no header block to match.
    """
    total = Scan()
    # One memory across the recording's bursts: ARQ repeats of one field arrive
    # in CONSECUTIVE cycles, which a per-burst memory could never see.
    memory = FieldMemory()
    for lo, hi in bursts(audio, fs):
        headed, n_heads = decode_headed(audio, lo, hi, fs=fs, levels=levels,
                                        memory=memory)
        total.trials += headed.trials
        total.packets.extend(headed.packets)
        if n_heads or not envelope:
            continue
        for body in bodies(audio, lo, hi, fs=fs):
            s = decode_body(audio, body, levels=levels, fs=fs)
            total.trials += s.trials
            total.packets.extend(s.packets)
    return total
