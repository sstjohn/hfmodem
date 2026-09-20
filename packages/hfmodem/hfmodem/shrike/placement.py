# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Forward placement of a decoder frame onto the (tone, symbol) grid.

The decoder gathers one soft byte per bit per active tone per symbol into a
channel-order buffer, permutes it into code order, and Viterbi-decodes the
result; a CRC-16/X25 over the recovered field is what gates the decoder's
status report. Transmitting a frame that survives that chain means running
every stage backwards:

    field bytes -> CRC-16/X25 -> conv K7 rate-1/2 -> code order
      -> inverse of the helical permutation -> channel order
      -> channel_index = (symbol * n_tones + tone_rank) * L + j
      -> differential phase steps on the path's tones

Stride, buffer size, tone set and bits-per-cell are not independent: they are one
descriptor per speed-level hypothesis, and getting a single one of them wrong
produces a frame that is well-formed and undecodable. `Path` bundles them so they
can only be chosen together.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace

import numpy as np

from . import coding, modem, p3frame, spec

FRAME_SYMBOLS = 72
"""Symbols per frame, the same on every short-cycle hypothesis.

72 is what makes the rest of the table consistent: every path's soft count
factors exactly as 72 x tones x bits-per-cell, which is the check `Path.n_symbols`
exposes and `tests/shrike/test_placement.py` asserts for all five paths."""

PACKET_S = (p3frame.DATA_OFFSET + FRAME_SYMBOLS) / spec.SYMBOL_RATE_BD
"""How long a short-cycle packet occupies the air, seconds.

`spec.P1_PACKET_S`'s counterpart, and what a cycle grid needs from this module:
one phase reference, a header block from `p3frame` and `FRAME_SYMBOLS` rows from
here, which is what a station reading them times its turnaround from. It counts
the GRID and not the keying: `ENTRY_TRAILER` puts one further symbol behind the
last row at every level, and on the two staggered levels the last carrier starts
half a symbol late, so what goes out runs 10 or 15 ms past this."""

CONV_GENERATORS = (0o171, 0o133)
"""K=7, rate 1/2, TERMINATED, bits packed LSB-first, over the WHITENED field.

The textbook K=7 pair. Which pair it is, and which bit order goes with it, are
settled by interoperability rather than by argument: six candidate pairs against
both bit orders were built into a case-1 header frame and offered to an
independent PACTOR-III decoder, and exactly one combination passes that decoder's
CRC and reads the chosen 26-byte field back byte-exact, against thirteen matched
negatives including a random codeword. This is that combination, and it is what a
real station's packets need: the 26 PACTOR-III packets of
`rf-corpus/PIII_Complete_1.wav`, speed levels 3 to 6, decode CRC-valid under this
pair in this order and under no other candidate in either order.

The frame the accepted pair produces is banked as a constant in
`tests/shrike/test_placement.py`, so the convention stays regression-tested here."""

CASE0_CODE = coding.CODE_K9
"""Case 0 does NOT share that code: it runs a 256-state, K=9 trellis.

The standard K=9 rate-1/2 pair is written with the newest bit at bit 0.
`coding.CODE_K9` holds the pair published for PACTOR-2. A field encoded this
way is accepted by an independent decoder, six cases out of six.

Case 0 is speed level 1, so this is the code on the slowest path the modem can
fall back to rather than a corner of the table."""

ENTRY_FLUSH = (0, 1, 1, 1, 1, 1, 0, 0)
"""The eight bits DL6MAA's entry packet ends its case-0 trellis on, not zeros.

MEASURED, and exactly. Our render of that packet and the packet itself agree on
every one of the first 128 code bits -- the same six info bytes, the same CRC,
the same whitening, transpose, header block and phase reference -- and disagree
on ten of the last sixteen, which are the trellis flush and nothing else. Case 0
neither punctures nor interleaves, so those sixteen bits land in sixteen named
cells and can be read straight off the air; the eight inputs that produce them
are unique among all 256, and with them the whole 81-symbol packet reproduces at
0 of 144 cells wrong. Confirmed on a second, independently encoded copy of the
same transmission (`occ15`), same tail, same zero.

A zero flush is what every other path here sends and what an ordinary receiver
expects, and the reference plainly does not send one. What generates these eight
is not known from one field -- as a ninth whitened field byte they read 0x78,
which is a byte of `spec.TEMPLATE` but not the one that follows this field's
five. So this is applied where it was measured, to the entry packet, and nowhere
else: that packet is the one whose every symbol a granting peer can predict
completely, which is the one place a wrong flush is a different packet rather
than a slightly noisier one."""

DATA_FLUSH: tuple[int, ...] | None = None
"""What an ORDINARY data packet ends its trellis on. None is zeros.

`ENTRY_FLUSH` is measured and applied where it was measured. This is the switch
that carries it onto every data packet instead, and it exists because of what is
left after the pulse: our speed-level-1 data packet and our entry packet are now
the same waveform -- same pulse, same stagger, same level, same two carriers --
and VE3KPG still answered CS1 thirty-five times at the one and read the other in
five keyings. What separates them is bits, and the flush is one of the two,
measured across those forty keyings rendered side by side. Zeros is the terminated
code every receiver here assumes and what every arm has flown, so it stays the
default until an arm says otherwise; `--p3-data-flush entry` and
`--p3-data-flush reference` (`REFERENCE_FLUSH`) are the two trials."""

REFERENCE_FLUSH = (0, 0, 0, 0, 1, 0)
"""The six bits DL6MAA's speed-level-3 data packets end their trellis on.

MEASURED off `rf-corpus/PIII_Complete_1.wav` through the production decode path
-- the packet's own header block for the swap and the angle, `p3rx._cells`,
`deinterleave` -- and scored against every candidate over the twelve positions a
flush can move. All six clean
short-cycle level-3 packets, two carrier arrangements, four status bytes, 0 B and
59 B payloads, agree on this tail at 0 of 12 movable bits wrong, where zeros
misses 3 and `ENTRY_FLUSH` misses 7. The same instrument returns `ENTRY_FLUSH`
exactly on that recording's entry packet, 0 of 144 code bits wrong.

IT IS NOT UNIVERSAL, and that is why it is a switch rather than a constant on the
render path. The reference's own level-4, level-5 and level-6 packets each fit a
different tail, its long-cycle level-3 packets fit another, and WS8EOC -- an SCS
modem on our own path, `captures/onair-0912-2321` -- ends its ordinary
speed-level-1 data packets on plain ZEROS, 0 of 16 wrong, on the two packets that
carry a clean CRC. So what a data packet ends on is field- or geometry-dependent
the way `ENTRY_FLUSH`'s docstring says, one repeatable tail per burst rather than
one constant per protocol, and zeros is what the one third-party modem this
station has actually exchanged packets with sends. This is the one-variable trial
`--p3-data-flush reference` flies at levels 2 to 6; case 0 keeps zeros, which is
both what WS8EOC keys there and the only tail width its K=9 trellis has."""

ENTRY_DELAY_S = 0.0324
"""How far past its own PACTOR-1 raster DL6MAA keys that same entry packet.

MEASURED on the one third-party station this package holds a COMPLETE handshake
from -- `rf-corpus/PIII_Complete_1.wav`, the session that goes on to carry 955-
and 965-byte payloads -- and measured twice, two ways that share no arithmetic:
envelope foot to foot puts the entry +32.4 ms past the cycle boundary its own
PACTOR-1 leg lands on, and the phase references the spec names put it +36.2 ms.

ONE STATION'S OBSERVED BEHAVIOUR, and nothing here claims a receiver requires
it. `ENTRY_FLUSH` above is what that station SENDS and it is byte-exact; this is
WHEN, and the two were never copied together -- every entry this package has
keyed went out on our own raster at +0.5 ms, and every one of those links
stalled. So this is a figure to FLY rather than a constant to build on:
`--p3-entry-delay` carries it onto the air and is off unless an arm line asks."""


@dataclass(frozen=True)
class Path:
    """One speed-level hypothesis of the header/confirm decode.

    A receiver picks a hypothesis from the preamble correlator, and that choice
    fixes the soft count, the field length the CRC covers, the bits-per-cell, the
    tone set (a channel mask) and the de-interleave stride together. Case 0 has no
    stride: it is the one path with no permutation at all, so its channel order
    and its code order coincide.
    """

    name: str
    case: int
    stride: int
    n_buf: int
    tones: tuple[int, ...]
    bits_per_cell: int
    crc_bytes: int
    puncture: coding.Puncture = coding.RATE_1_2
    """Which code bits the transmitter deletes. The two highest speed levels are
    the only ones that delete any; every lower path keeps the rate-1/2 stream
    whole, so `RATE_1_2` makes the puncture stage a no-op rather than a branch."""

    @property
    def n_symbols(self) -> int:
        return self.n_buf // (len(self.tones) * self.bits_per_cell)

    @property
    def n_pairs(self) -> int:
        """Trellis steps behind the `n_buf` bits that reach the air.

        Puncturing is what separates the two counts: the buffer holds what was
        transmitted, the trellis runs over what was encoded.
        """
        period, kept = self.puncture.period, sum(sum(r) for r in self.puncture.mask)
        return self.n_buf * period // kept

    @property
    def frame_bytes(self) -> int:
        """Bytes the Viterbi yields: rate 1/2, no termination."""
        return self.n_pairs // 8

    @property
    def speed_level(self) -> int:
        """The protocol's name for this path. The decoder's case numbers are one
        less throughout, and that off-by-one is a trap: everything on the air --
        the variable header above all -- counts from 1."""
        return self.case + 1

    @property
    def long_cycle(self) -> bool:
        return self.n_symbols == LONG_ROWS

    @property
    def subband_lead(self) -> tuple[float, ...]:
        """Symbols each virtual carrier runs ahead of the rest of the comb.

        One entry per rank in `tones`, and zeros on levels 3 to 6 --
        `spec.SUBBAND_LEAD` is where it comes from and why. Rank-indexed rather
        than channel-indexed so that it survives `p3rx.path_for`, which rewrites
        `tones` into the physical channels a swapped cycle puts them on.
        """
        return spec.SUBBAND_LEAD.get(self.speed_level, (0.0,) * len(self.tones))

    def clock_offsets(self, sps: int) -> tuple[int, ...]:
        """Samples each virtual carrier's symbol clock sits INTO the packet.

        The lead is a relative measurement, so something has to anchor it: the
        earliest carrier starts the packet, and the rest follow it by the lead
        they give up. That keeps a packet's first sample where it has always
        been, which is what every timing search and every anchored caller of
        `data_packet` already assumes; the packet ends that much later instead.
        """
        lead = self.subband_lead
        return tuple(round((max(lead) - x) * sps) for x in lead)

    @property
    def bit_phase(self) -> dict[int, float]:
        """Differential step per bit. Case 0 carries the opposite sign convention,
        so its two phases are exchanged relative to every other path."""
        return BIT_PHASE if self.case else {b: BIT_PHASE[1 - b] for b in BIT_PHASE}


DETECT = Path("detect", 0, 0, 144, (5, 12), 1, 8)
"""2 tones, DBPSK, 144 = 72 x 2 x 1, and NO interleave at all.

The permutation runs only where the stride is non-zero, and case 0's stride is
zero, so the channel buffer is read straight through. That leaves this path with
no permutation, no puncture and a two-tone grid -- the smallest complete frame in
the protocol."""

HEADER = Path("header", 1, 35, 432, (3, 5, 7, 10, 12, 14), 1, 26)
"""6 tones, DBPSK, 432 = 72 x 6 x 1.

This is the SL>=2 header frame, and it is the one path judged end to end from
audio by an independent decoder: `data_packet` on this path carries a chosen field
through that decoder's CRC, which is what makes it report a speed level and a
status line -- every cycle of an eight-cycle session, and nothing at all for the
cycles of the same session at any other case. The stride of 35 is measured rather
than assumed -- both sides of the de-interleave taken from one cycle agree
exactly."""

_MID_TONES = tuple(range(2, 16))                # mask 0xfffc, shared by cases 2/3

DATA3 = Path("data3", 2, 69, 1008, _MID_TONES, 1, 62)
"""14 tones, DBPSK, 1008 = 72 x 14 x 1. The spec's speed level 3.

Unpunctured, like cases 0 and 1, so it carries no unknown: tone mask and stride
are both fixed for the case, and the field length follows the rule the lower cases
already obey (n_buf / 16 - 1)."""

DATA4 = Path("data4", 3, 250, 2016, _MID_TONES, 2, 125)
"""The same 14 tones at 2 bits/cell -- DQPSK -- so 2016 = 72 x 14 x 2. Speed
level 4, and also unpunctured."""

# Both of the above are corroborated against the published table. Subtracting the
# CRC and the status byte from each field length gives the usable payload, and for
# every unpunctured case that lands exactly on M.1798 §1: 5, 23, 59 and 122 bytes
# for speed levels 1-4. The geometry here and the ITU Recommendation are arrived at
# independently, and they agree digit for digit across four cases -- not an
# agreement a wrong geometry survives.
#
# The same arithmetic explains why the two highest levels do NOT land on it: they
# are punctured, and the rate is what the difference measures. Level 5 needs
# 215 x 8 = 1720 info bits from 2304 softs, which is rate 3/4 (1720 x 4/3 = 2293);
# level 6 needs 2296 from 2592, which is 8/9 (2296 x 9/8 = 2583). Both agree with
# the published rates, so what is missing for those two is only the puncture
# PATTERN -- not the geometry, and not the rate.
CONFIRM = Path("confirm", 4, 563, 2304, tuple(range(1, 17)), 2, 215,
               coding.PUNCTURE_3_4)
"""16 tones, DQPSK, 2304 = 72 x 16 x 2. Speed level 5, and the first punctured one.

The permutation reproduces a captured cycle's buffer on all 2304 positions. The
puncture is `coding.PUNCTURE_3_4`, and it is no longer the open question this
docstring used to record: a decoder's depuncture stage deletes serial code bit j
whenever j mod 3 == 1, which is that mask read out in transmission order. 2304
softs open to 3456, i.e. 1728 trellis steps, i.e. 1722 information bits -- the
215-byte field with two bits to spare."""

DATA6 = Path("data6", 5, 909, 2592, tuple(range(18)), 2, 287,
             coding.PUNCTURE_8_9)
"""All 18 tones, DQPSK, 2592 = 72 x 18 x 2. Speed level 6, the fastest.

Stride 909 is the fifth entry of the same static table the four lower strides come
from, so it is read rather than fitted -- which matters, because every stride in
1..2591 permutes this buffer and nothing about the geometry would reject a wrong
one. The 8/9 puncture keeps 9 of every 16 serial code bits: 2592 softs open to
4608, 2304 steps, 2298 information bits, and the field is 287.

UNVALIDATED ON AIR. Speed level 6 needs a channel good enough that none of the
recordings here carries one, so this path has never met a real signal -- only
synthesis. Read the stride as the one value most likely to be wrong."""


LONG_ROWS = 320
"""Grid rows a 3.75 s cycle carries, against the short cycle's 72.

Derived rather than measured, and then checked hard. Every long-cycle field
length the geometry produces -- 320 rows times the level's tones times its
bits per cell, through its puncture and its trellis flush -- lands exactly on
`spec.SPEED_LEVELS[sl].payload_long`, on all six levels: 36, 116, 276, 556, 956,
1276. Six independent agreements with a table arrived at from the other
direction, and a wrong row count misses all six.
"""

LONG_PATHS: dict[int, Path] = {
    1: Path("long1", 0, 0, 640, (5, 12), 1, 39),
    2: Path("long2", 1, 181, 1920, HEADER.tones, 1, 119),
    3: Path("long3", 2, 321, 4480, _MID_TONES, 1, 279),
    4: Path("long4", 3, 1122, 8960, _MID_TONES, 2, 559),
    5: Path("long5", 4, 2407, 10240, tuple(range(1, 17)), 2, 959, coding.PUNCTURE_3_4),
    6: Path("long6", 5, 3876, 11520, tuple(range(18)), 2, 1279, coding.PUNCTURE_8_9),
}
"""The same six levels on the 3.75 s cycle.

The strides are the SECOND of the two static tables the de-interleaver selects
between, the one taken when its cycle-length argument is non-zero -- so the long
cycle is not a different mechanism, only a different row of the same one.

Levels 3-6 are VALIDATED against a third-party transmission: the ten long-cycle
fields of `rf-corpus/PIII_Complete_1` -- 276-byte SL3 x2, 556-byte SL4 x2,
956-byte SL5 x4, 1276-byte SL6 x2 -- decode CRC-valid through these paths, with
the mod-4 counter running unbroken across the short/long changes and the payload
decompressing to continuous prose. Long SL1 is independently decoded by the SCS
reference decoder (four valid fields, both carrier orders,
with four bad-CRC controls rejected). No stock long SL1 capture is available.
Level 2 still stands on the arithmetic and local round trips."""


SPEED_PATHS: dict[int, Path] = {1: DETECT, 2: HEADER, 3: DATA3, 4: DATA4,
                                5: CONFIRM, 6: DATA6}
"""Speed level -> the frame geometry its data field uses, short cycle.

The `case` numbers are one less throughout, and that is the trap: the
decoder's case 0 is the protocol's speed level 1. Callers should index by speed
level, which is what the header on the air actually carries.

Each path's usable payload -- `crc_bytes` less the status byte and the CRC -- is
`spec.SPEED_LEVELS[sl].payload_short` on all six, arrived at from the geometry
rather than from the table: 5, 23, 59, 122, 212, 284 both ways. Nothing holds
them equal -- `tests/shrike/test_placement.py` checks the stride bijection, the
interleave inverse and a soft round trip, and reads neither table."""

CHANGEOVER = Path("changeover", 0, 0, 112, (5, 12), 1, 6)
"""The frame a changeover packet carries behind its CS3 head.

Case 0 with sixteen rows taken away, and the sixteen are what the head costs:
an ordinary packet spends symbols 1-8 on the header block and 9-80 on 72 rows,
and a changeover packet spends symbols 1-24 on the codeword and its run-in
(`CHANGEOVER_HEAD_SYMBOLS`) and 25-80 on the 56 rows left. 56 rows x 2 carriers
is 112 cells, and `n_buf / 16 - 1` gives 6 field bytes -- three of payload, the
status byte and the CRC -- which is the rule every unpunctured case obeys.

MEASURED, on both directions of `rf-corpus/PIII_Complete_1.wav`. The IRS takes
the link at 5.5637 s and the ISS takes it back at 7.7681 s; each is ONE keying,
82 symbols, on channels 5 and 12 alone, opening on a phase reference and the CS3
codeword and carrying no packet header block anywhere (no variable header in
either burst reaches 0.75, against 0.977 for the same station's entry packet).
Through this geometry -- carrier order `p3frame.VH_ORDER`, the K=9 trellis, the
sixteen-by-seven transpose -- both decode CRC-valid and hold their field across
the whole two-dimensional span of sampling instants that covers both carriers:

    5.5637 s  0d 50 54  status 0x20  counter 0   the IRS's, `\\r P T` in ASCII
    7.7681 s  0f 8f 87  status 0x18  counter 0   the ISS's, the walking template

The first is the corroboration a CRC alone would not be: the greeting packet
that follows it decompresses to `C-II DSP/QUICC System - Maildrop QRV`, so those
three bytes are the missing head of that sentence and nothing else. The second is
the first three bytes of the header template its own speed-level-3 train carries,
at the data type that train declares. Both counters are 0 and both are followed by
counter 1, which is the reset `arq.PactorArq.on_cycle` already performs."""

CHANGEOVER_HEAD_SYMBOLS = 24
"""Symbol times between a changeover packet's phase reference and its frame's.

Twenty of them are the CS3 codeword, keyed the way `control_signal` keys a bare
one so that a receiver reading for an acknowledgement finds it there. The four
behind it carry no bit anything reads -- measured, they are four more zero bits of
the same DBPSK stream -- and what they are for is arithmetic: they put row 0
exactly sixteen rows into the ordinary case-0 grid, which is what makes the
shortened frame a whole number of interleaver columns."""


DIBIT_PHASE = np.angle(p3frame.DIBIT_PHASOR).reshape(2, 2)
"""Differential phase step per dibit for the two-soft (DQPSK) paths, [bit0][bit1].

A receiver takes bit0 from sin(delta) and bit1 from cos(delta), a positive soft
value meaning a zero bit; that gives bit0 = [sin(delta) < 0] and
bit1 = [cos(delta) > 0], which places the constellation on the pi/4 diagonals.

Taken from `p3frame` rather than declared again, because the header block sits on
this same constellation -- measured off the air, and the reason a transmitter can
emit a header at all."""

BIT_PHASE = {0: 1.75 * np.pi, 1: 0.75 * np.pi}
"""Differential phase step per bit, for the one-soft (DBPSK) paths.

Cases 0-2 take a single sample at delta - 45 deg with the sign rule inverted, so
the soft value is positive -- bit 0 -- exactly when sin(delta - 45 deg) < 0. The
two antipodal points that satisfy that most strongly are 315 deg for a zero and
135 deg for a one: the same diagonal the DQPSK constellation sits on, halved."""


@functools.cache
def _walk(path: Path) -> np.ndarray:
    if not path.stride:
        return np.arange(path.n_buf, dtype=np.int32)
    src = np.empty(path.n_buf, dtype=np.int32)
    cur = start = path.n_buf - path.stride
    for k in range(path.n_buf):
        src[k] = cur
        cur -= path.stride
        if cur < 0:
            start += 1
            cur = start
    return src


def channel_of_code(path: Path) -> np.ndarray:
    """`src[k]` = channel-order index whose soft byte is code-order position k.

    The walk goes backwards through the channel buffer in steps of the path's
    stride, dropping to the next start offset whenever it runs off the front. A
    zero stride skips the walk entirely and the two orders coincide.

    The walk depends on nothing but the path, so it is computed once per path and
    handed out as a copy: callers index with it, and the copy keeps the cached
    original safe from one that writes.
    """
    return _walk(path).copy()


def deinterleave(buf: np.ndarray, path: Path) -> np.ndarray:
    """Channel order -> code order (the decoder's direction)."""
    return np.asarray(buf)[channel_of_code(path)]


def interleave(code: np.ndarray, path: Path) -> np.ndarray:
    """Code order -> channel order (ours)."""
    out = np.zeros(path.n_buf, dtype=np.asarray(code).dtype)
    out[channel_of_code(path)] = code
    return out


def build_field(info: bytes, path: Path, tail: int = 0) -> bytes:
    """Info bytes -> the full field the decoder reconstructs.

    `path.crc_bytes` is the whole field, the last two bytes of it being the CRC --
    the length a receiver's CRC-16 covers. The trellis emits a little more
    than that (27 bytes for case 1's 432 softs) but the surplus is flush, so the
    field never needs padding.
    """
    if len(info) != path.crc_bytes - 2:
        raise ValueError(f"{path.name} carries {path.crc_bytes - 2} info bytes")
    return bytes(info) + coding.crc16(info).to_bytes(2, "little")


def encode_frame(field: bytes, path: Path,
                 flush: tuple[int, ...] | None = None) -> np.ndarray:
    """A full field -> `path.n_buf` coded bits, ready for `interleave`.

    Every case runs the same shape: whiten, pack LSB-first, convolutionally encode
    over the field and then over `flush`. Whitening comes FIRST because the decoder
    applies it to the Viterbi's output before the CRC reads it, and it is an
    involution -- so the bits that must go down the trellis are the whitened ones.
    Case 0 differs only in using the K=9 code.

    `flush` is the K-1 bits the trellis ends on, and zeros -- the terminated code
    every receiver here assumes -- is only the default. `ENTRY_FLUSH` is what one
    real station was measured sending.
    """
    code = CASE0_CODE if path.case == 0 else coding.ConvCode(7, CONV_GENERATORS)
    bits = coding.bytes_to_bits(coding.whiten(field), msb_first=False)
    n_info = path.n_pairs - (code.constraint_length - 1)
    bits = np.concatenate([bits[:n_info], np.zeros(max(0, n_info - bits.size), np.uint8)])
    tail = np.zeros(code.constraint_length - 1, np.uint8) if flush is None \
        else np.asarray(flush, np.uint8)
    raw = code.encode(np.concatenate([bits, tail]), terminate=False)
    return path.puncture.apply(raw)[:path.n_buf]


def build_grid(info: bytes, path: Path = DETECT, tail: int = 0, *,
               flush: tuple[int, ...] | None = None) -> np.ndarray:
    """(n_symbols, n_tones, bits_per_cell) bit grid carrying the frame."""
    code = encode_frame(build_field(info, path, tail), path, flush)
    return interleave(code, path).reshape(path.n_symbols, len(path.tones),
                                          path.bits_per_cell)


def grid_steps(grid: np.ndarray, path: Path) -> dict[int, np.ndarray]:
    """Bit grid -> per-tone differential phase steps (radians)."""
    if path.bits_per_cell == 1:
        phase = path.bit_phase
        step = np.array([phase[0], phase[1]])[grid[:, :, 0]]
    else:
        step = DIBIT_PHASE[grid[:, :, 0], grid[:, :, 1]]
    return {tone: step[:, rank] for rank, tone in enumerate(path.tones)}


# ---------------------------------------------------------------------------
# Packet
# ---------------------------------------------------------------------------

PREAMBLE_SYMBOLS = 20
"""Acquisition burst length. Twenty symbols is what an independent decoder
acquires on; below about that the correlator's history is never filled."""

PREAMBLE_RMS_RATIO = 0.89
"""Preamble RMS relative to the packet body.

Peak-normalising both parts separately makes the preamble almost three times as
loud as a 16-tone body, whose crest factor is far higher. 0.89 is the ratio a
signal carried that an independent decoder reported as speed level 3."""
ACQ_REF = {0: 1, 1: 2, 2: 5, 3: 7, 4: 1, 5: 3}
"""Case -> the pre800 reference its acquisition burst carries.

A receiver reads the speed level out of the reference that matched -- the case is
(ref >> 1) & 3, so two references serve each case -- and it is not indifferent to
which of the two. Measured against an independent monitor, one session per entry:
case 1 reports at 2 and only once in eight cycles at 3; case 2 reports every
cycle at 5 and not at all at 4; case 3 reports every cycle at 7. Case 0 has
reported at neither 0 nor 1 and is the one open entry here.

Cases 4 and 5 have no reference of their own -- theirs would set the long-cycle
bit -- so they borrow the burst of the case a receiver falls back from, and
neither has been read back."""


def _preamble(body: np.ndarray, cfg: modem.ModConfig, case: int) -> np.ndarray:
    """The acquisition burst a packet of this case opens with."""
    pre = p3frame.preamble_audio(cfg, ref=ACQ_REF[case])[:PREAMBLE_SYMBOLS * cfg.sps]
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    pre = pre * (PREAMBLE_RMS_RATIO * rms(body) / rms(pre))
    # The preamble is a whole number of symbols, so the body starts on a symbol
    # boundary of the grid the packet is anchored on; a half-symbol straddle lands
    # a receiver's sampling instant at the worst phase of the eight it takes.
    return np.concatenate([pre, np.zeros(-len(pre) % cfg.sps)])


START_PHASE = np.array([-np.pi * k * (k + 1) / spec.N_CHANNELS
                        for k in range(spec.N_CHANNELS)])
"""Angle each channel's phase-reference symbol is keyed at: Schroeder's quadratic.

A packet opens on ONE symbol that carries no data, and every lit carrier used to
enter it at the same angle. Fourteen adding in phase put that one symbol 14.44 dB
over the packet's RMS, a twentieth of a dB off the 10*log10(2N) that says they
were perfectly aligned, and `core.levels.at_drive` sets the drive by the packet's
peak -- so eighty-eight symbols were keyed 2.0 to 2.7 dB below what the operator
asked for, at every speed level wide enough for the reference symbol to have been
the peak. That is a share of the tenfold drop this station measured on PACTOR-3
against PACTOR-1 at one audio level: 11.5 dB of it before this, 8.8 dB after, and
what is left is the crest a fourteen-tone comb has and one FSK tone does not.

Schroeder's set is a formula rather than a search, and it costs a receiver
nothing: PACTOR-III reads each carrier differentially, so a constant angle per
carrier cancels in every difference the header block and the data field are taken
from. Both halves measured. Worst case over six speed levels, five payloads and
both carrier swaps, 15.7 dB of peak crest becomes 13.0. Over 48 renders across
the four wide levels and both swaps, the header anchor lands on the same sample
in 44 and one quarter-symbol search step away in four, never missing, never on
another variable header, worst fit 0.863 against `p3rx.HEADER_FIT`'s 0.80 -- and
the four that moved are the raised-cosine transmit pulse leaking between adjacent
channels, which the angle between them turns. An independent PACTOR-III monitor
reads every level 1 to 6 off these phases, all four arms of
`tests/shrike/test_p3_oracle.py`, with the negative arm still silent.

Newman's phases do the same job 0.2 dB worse here; the two are one time reversal
apart on pure carriers and differ only through the header block, which is a fixed
word rather than a random one. Speed levels 1 and 2 lose 0.1 and 0.7 dB, having
too few carriers for the reference symbol to have set the peak; two of them cannot
be helped at all, since they sweep through alignment inside every symbol whatever
angle they start at.

Keyed on the TONE a carrier is transmitted on rather than on its home channel, so
the crest is the same on both halves of the carrier swap."""


def assemble(steps: dict[int, np.ndarray], path: Path, *,
             cfg: modem.ModConfig | None = None,
             swapped: bool = False, acquire: bool = False,
             stagger: bool = True, request_status: bool | None = None) -> np.ndarray:
    """[phase reference][header block][one row per symbol] -> passband audio.

    `steps` is the data field's differential phase, keyed by HOME channel. The
    carrier swap is applied here and nowhere else: it moves a whole virtual
    carrier -- its header dibits and its data cells together -- onto the partner
    tone, which is the only thing the specification's frequency diversity does.
    The sub-band lead rides along with it, because both are keyed on the carrier's
    rank and the swap does not disturb rank.

    Public because a gate needs it: an interoperability test's negative arm has to
    put a field on the air whose CRC does not check out, and there is no other way
    to hand this a field the CRC was not computed over.

    `stagger` off keys the whole comb on one clock whatever the level's lead is.
    It is `CASE0_STAGGER`'s seam and nothing else asks for it.
    """
    cfg = cfg or modem.ModConfig()
    vh = p3frame.variable_header(path.speed_level, swapped=swapped,
                                 long_cycle=path.long_cycle,
                                 request_status=request_status)
    head = p3frame.header_steps(vh, path.tones)
    offsets = path.clock_offsets(cfg.sps) if stagger else (0,) * len(path.tones)
    tone_syms, delay = {}, {}
    for rank, cn in enumerate(path.tones):
        tone = spec.CARRIER_SWAP[cn] if swapped else cn
        start = np.angle(p3frame.PATTERN_A[0]) + START_PHASE[tone]
        ph = np.concatenate([[start], head[cn], steps[cn]])
        tone_syms[tone] = np.exp(1j * np.cumsum(ph))
        delay[tone] = offsets[rank]
    body = modem.modulate_tones(tone_syms, cfg, delay=delay)
    return np.concatenate([_preamble(body, cfg, path.case), body]) if acquire else body


ENTRY_TRAILER = -np.pi / 4
"""The 82nd symbol: a -45 degree step on every carrier, after the last data row.

MEASURED, off the entry packet of `rf-corpus/PIII_Complete_1.wav` and its
independently encoded copy, by the same one-symbol DFT that reads the packet's
extent. The real station's power holds through symbol 81 (within 0.6 dB of the
body) and collapses at 82, and the step from symbol 80 to 81 reads -43 and -46
degrees on channels 5 and 12 -- a modulated symbol, where a transmitter's
key-down ramp would hold the previous phase. The same instrument reads that
station's speed-level-3 packets at 82 symbols too, and a second station's
long-cycle level 6 (`pos_pactor3_sl6.wav`) through symbol 329. That is the
trailer pactor3.md §3 already published for the data levels as one fixed word
on all fourteen carriers, whatever the payload.

No 81-symbol packet has ever been recorded, ours included, and every level here
now keys the 82nd -- the entry, which flew twice on 2026-08-31, and the data
levels, which have not. The independent monitor reads levels 2 to 6 with it in
place -- twenty frames of twenty per run, each at the level it was sent at and
at the field length it was given, 23/59/122/212/284 bytes -- so it costs nothing
a receiver notices; whether one REQUIRES it is unmeasured either way, and the
answer slot demonstrably does not move with it: it flew twice on
2026-08-31 and the peer answered at the same instant both times.

The name is the entry's because that is where it was measured. `onair`'s cycle
arithmetic still counts `PACKET_S`, which is 81 symbols, so every data packet
now keys 10 ms past what that constant says -- 46 ms clear of the answer
instant, by the same measurement that cleared the entry."""


CASE0_STAGGER = True
"""Whether the two-carrier keyings split their carriers half a symbol.

The entry packet and the changeover packet, which are the only case-0 things
this station transmits. On by default because that is what the reference keys
(`spec.SUBBAND_LEAD`) and what our render had to become to match it; off, both
go out on one clock, which is the render four granted arms flew unread. It is a
module-level value rather than an argument because the caller that would pass it
is a command line, and `onair.RadioTx` reaches the renderer through three seams.

The level-2 stagger is not on this switch. It is corroborated from outside -- an
independent monitor reads our level 2 with it and nothing at all without -- so
there is no arm left for it to be the subject of."""


PROTOCOL_RISE = True
"""Whether everything this station keys shapes with the modem's OWN symbol pulse.

`modem.ModConfig` defaults to a generic raised cosine, rolloff 0.3 over eight
symbols, which is 3841 taps at 48 kHz and puts 40 ms of transmit filter in front
of the first symbol. `tablegen.symbol_pulse` -- the quasi-RRC [SCS-P4] s11.8
prints, the one `rx` already runs as its matched filter and `control_signal`
already keys with -- is 31 taps at eight per symbol and puts 19 ms there.

MEASURED, and the difference is the whole of the carrier's rise. On a ch5+ch12
envelope at 299 Hz analysis bandwidth, referred to each packet's own steady
level and to its own channel-5 symbol 0: DL6MAA's entry, its seven speed-level-3
packets and both its changeover frames all sit at the tape's noise floor
(-40 dB) until 16-18 ms before symbol 0 and cross 2 % / 10 % / 50 % / 90 % at
-5.9 / -4.4 / -3.0 / -1.9 ms. The raised cosine crosses at -29.6 / -10.6 /
-3.0 / +0.6, with a -34 dB shelf running from -40 ms to -20 ms that is 7 dB
above that tape's floor where the reference shows nothing at all. This pulse
crosses at -14.7 / -7.7 / -4.0 / +0.4 and its shelf is gone.

Three further readings agree and none of them is the rise. Whole-waveform
correlation against the reference entry goes 0.8086 -> 0.8769, past the 0.83 an
identical waveform was thought to be capped at by that recording's own SNR; the
nine-symbol head of a level-3 packet reads better on six of the reference's
seven, 0.329 -> 0.387 mean; and the envelope's coefficient of variation goes
0.182 -> 0.104 against the reference's 0.124, which fading and noise can only
have inflated. The peak-to-average also falls 9.26 -> 7.60 dB, so a
peak-normalised transmitter puts 1.7 dB more of this packet on the air.

Off, every keying goes out on the raised cosine, which is the render five
granted arms flew unread. `onair.ENTRY_END_N` is keyed to this: 40266 samples
on, 40888 off.

IT GOVERNS THE DATA PACKETS TOO, and that is what the name stopped saying. The
entry, the changeover and speed level 1 are case 0 and took this pulse from the
day it landed; levels 2 to 6 went on through `modem.ModConfig`'s default, which
is the only PACTOR-III thing this station keyed on the generic filter. Rendered
side by side, that cost a level-3 packet 20 ms of energy in front of the keying
where the reference's own
data packets sit on their tape's floor, a phase reference 27.8 ms past the
boundary against the entry's 13.9, and 29 ms of extra keying -- so a peer that
acquired on our entry met every data packet 1.4 symbol periods late, through a
receiver whose matched filter is the pulse we did not use. The reference shows
no such difference: its data packets have its entry's attack, packet for
packet."""


def protocol_config() -> modem.ModConfig:
    """The modulator every PACTOR-III keying uses, under `PROTOCOL_RISE`."""
    return modem.ModConfig(matched_pulse=PROTOCOL_RISE)


def data_packet(info: bytes, path: Path = HEADER, *,
                cfg: modem.ModConfig | None = None,
                swapped: bool = False, acquire: bool = False,
                flush: tuple[int, ...] | None = None) -> np.ndarray:
    """One packet carrying `info` through `path`'s frame geometry -> audio.

    What goes out is what a station on an established link puts on the air: one
    phase-reference symbol, the eight-symbol header block, all 72 data rows and
    the trailer behind the grid, on the tones this ARQ cycle's carrier swap puts
    them on. Eighty-two symbols, which is what every packet on tape measures at
    every level -- see `ENTRY_TRAILER`.

    `acquire` prepends the +-800 burst a monitor locks a LINK up on. It is not
    part of a packet and no real cycle carries one, so it is off by default; it
    costs no rows either way, because the burst now sits ahead of the phase
    reference rather than over the head of the grid.

    Interoperability: an independent PACTOR-III monitor reads this packet with no
    burst in front of it and reports it as the speed level it was sent at. Levels
    3, 4, 5 and 6 each printed a status line and the payload BYTE-EXACT at 59,
    122, 212 and 284 bytes, over four ARQ cycles apiece with the carrier swap
    alternating -- and the top two had never been read by anything outside shrike,
    because at rate 3/4 and 8/9 they do not survive the rows the old acquisition
    burst cost. Levels 1 and 2 are read once the monitor has locked on a wider
    level first, which is the only way a real link reaches either: level 2 at
    eight cycles of eight, three runs, both arrangements -- and at none at all
    until `spec.SUBBAND_LEAD` put its two three-tone clusters half a symbol apart.
    `tests/shrike/test_p3_oracle.py:NARROW_LEVELS_NOTE`.

    The payload prints verbatim on a monitor only if the field's LAST byte is a
    status byte whose data type is 8-bit ASCII: the monitor reads the compression
    table out of those three bits and will happily expand plain text through a
    Huffman table into 232 bytes of German-looking noise, which is not a decode
    failure but it reads like one.
    """
    steps = grid_steps(build_grid(info, path, flush=flush), path)
    steps = {cn: np.append(s, ENTRY_TRAILER) for cn, s in steps.items()}
    return assemble(steps, path, cfg=cfg or protocol_config(), swapped=swapped,
                    acquire=acquire, request_status=bool(info[-1] & 1))


def case0_packet(info: bytes, *, cfg: modem.ModConfig | None = None,
                 swapped: bool = False, acquire: bool = False,
                 flush: tuple[int, ...] | None = None,
                 long_cycle: bool = False) -> np.ndarray:
    """One speed level 1 packet -> passband audio.

    The case-0 geometry is its own: cells run tone 12 then tone 5, and the code
    order is a sixteen-column transpose rather than a helical walk (nine rows
    short, forty long). Everything
    around the field -- the phase reference, the header block, the swap -- is the
    shape every other level uses, and speed level 1 has both variable-header
    carriers and no constant one, so its header block is the variable header
    alone. `rx.decode_case0_header` inverts the field.
    """
    path = LONG_PATHS[1] if long_cycle else DETECT
    cell_order = replace(path, tones=p3frame.VH_ORDER)
    steps = grid_steps(case0_cells(info, path, flush).reshape(path.n_buf // 2, 2, 1),
                       cell_order)
    steps = {cn: np.append(s, ENTRY_TRAILER) for cn, s in steps.items()}
    return assemble(steps, path, cfg=cfg or protocol_config(), swapped=swapped,
                    acquire=acquire, stagger=CASE0_STAGGER,
                    request_status=bool(info[-1] & 1))


def terminal_packet(*, swapped: bool = False,
                    cfg: modem.ModConfig | None = None,
                    header_bit: int = 1) -> np.ndarray:
    """Measured post-QRT marker: an SL1 header and an uncoded known body.

    `PIII_Complete_1` at 75.874375 s follows the QRT packet's CS1 with
    VH1, 72 alternating +135/-45 degree rows on both carriers, and a
    trailer of -45 degrees on home channel 5, +135 on home channel 12.
    CS2 follows this marker. No ordinary field, whitening or CRC is present.

    Only the unswapped VH1 example is recorded. Other header bits and the
    swapped arrangement are explicit experimental variants: the latter moves
    each virtual carrier's header, stagger and trailer together, following
    the ordinary P3 carrier-swap rule. This renderer does not implement or
    claim the complete terminal handshake.
    """
    if header_bit not in (0, 1):
        raise ValueError("terminal header_bit must be 0 or 1")
    body = np.tile([3 * np.pi / 4, -np.pi / 4], FRAME_SYMBOLS // 2)
    steps = {5: np.append(body, -np.pi / 4),
             12: np.append(body, 3 * np.pi / 4)}
    return assemble(steps, DETECT, cfg=cfg or protocol_config(),
                    swapped=swapped, stagger=CASE0_STAGGER,
                    request_status=bool(header_bit))


def case0_cells(info: bytes, path: Path,
                flush: tuple[int, ...] | None = None) -> np.ndarray:
    """`info` -> case 0's channel-order cells, one bit each."""
    cells = np.zeros(path.n_buf, np.uint8)
    cells[case0_map(path)] = encode_frame(build_field(info, path), path, flush)
    return cells


@functools.cache
def case0_map(path: Path = DETECT) -> np.ndarray:
    """Case 0's code-order -> cell-order permutation: sixteen columns.

    A transpose of sixteen columns by `n_buf // 16` rows, so the depth follows
    the buffer: nine for a whole speed-level-1 frame and seven for `CHANGEOVER`'s
    shortened one. Both are read off the air -- the seven-row depth is what
    decodes DL6MAA's two changeover packets, and no other permutation of that
    buffer decodes either.
    """
    n = path.n_buf
    return np.array([(n // 16) * (k % 16) + k // 16 for k in range(n)]) % n


# --------------------------------------------------------------------------- #
# What a station on an established PACTOR-III link actually keys
# --------------------------------------------------------------------------- #

def link_packet(sl: int, payload: bytes, status: int, *,
                swapped: bool = False, acquire: bool = False, long_cycle: bool = False,
                flush: tuple[int, ...] | None = None) -> np.ndarray:
    """The data packet an ISS transmits this cycle at speed level `sl`.

    `long_cycle` selects `LONG_PATHS` -- 320 rows on the 3.75 s cycle -- and the
    variable header follows from the path, so the receiver is told which frame it
    is looking at by the packet itself. At SL1 the long field uses the same
    sixteen-column transpose as the short field, with 320 data rows and a
    36-byte payload; its header must describe that long geometry too.

    Every level has its OWN frame geometry, so a level-3 packet sent through
    level 2's path is reported by a monitor as level 2 -- or, where the header
    and the frame disagree, not reported at all. The field is fixed-size for the
    level, so a station with less than that to say fills the rest of it out and
    the receiver drops what it wrote.

    AN EMPTY FIELD IS THE TEMPLATE INSTEAD, and that one is measured rather than
    inferred: fourteen fields of `rf-corpus/PIII_Complete_1` carry
    `spec.TEMPLATE` and nothing else, at speed levels 1, 3 and 6, and the entry
    packet §17.1 turns on is one of them. A station keying five bytes of user
    text where the reference keys the template is not sending a degraded entry
    packet, it is sending a different packet.

    THE SHORT-FIELD PADDING IS INFERRED. No published PACTOR-III text says what
    fills a field a station only part-filled, and the two in the reference resume
    the template at byte 4 and byte 14 of it, which no offset rule reproduces. So
    that case keeps 0x1E, the character stream's IDLE (PACTOR-1's description
    says so, and this package's PACTOR-1 renderer has always padded with it and
    its receiver has always stripped it), and an independent monitor's own
    reported lengths drop trailing IDLE (pactor1-data-packets.md §6). Nothing
    outside this package has been shown one: `tests/shrike/test_p3_oracle.py`
    fills its payloads exactly.

    A LENGTH BYTE USED TO SIT AT THE END OF THIS FIELD, and it did three kinds of
    damage. It was not interoperable -- a real field is a fixed size both ends
    know, so no length travels, and an SCS modem read it as one more payload
    byte. Only one of the two receive paths honoured it, so the same packet
    delivered eleven bytes through the tracked decoder and eleven bytes followed
    by a hundred NULs through the scanning one. And it cost a byte a packet
    silently: the ARQ layer chunks `spec.SPEED_LEVELS[sl].payload_short` bytes,
    which is exactly what the field holds, so the last one was truncated away
    every cycle.

    One home for this because two ends of one link must build the same bytes:
    `onair.RadioTx` keys it on the air and `tests/shrike/test_qso.py` keys it into
    the loopback, and while each built its own the rehearsal passed over a
    waveform the radio never sent.
    """
    path = (LONG_PATHS if long_cycle else SPEED_PATHS)[sl]
    info = field_info(payload, path.crc_bytes - 3, status)
    flush = DATA_FLUSH if flush is None else flush
    if path.case == 0:
        # A flush is K-1 bits of ONE trellis. Case 0's is eight bits long and a
        # six-bit tail there would encode two steps short, so the frame would
        # come out shorter than `n_buf` -- silently, and unreadable.
        if flush is not None and len(flush) != CASE0_CODE.constraint_length - 1:
            flush = None
        return case0_packet(info, swapped=swapped, acquire=acquire, flush=flush,
                            long_cycle=long_cycle)
    return data_packet(info, path, swapped=swapped, acquire=acquire, flush=flush)


def field_info(payload: bytes, n: int, status: int) -> bytes:
    """`n` payload bytes and the status byte behind them.

    Public because the entry packet's field is a claim about a stranger's modem
    and a test has to be able to read it as bytes -- `tests/shrike/
    test_p3_upgrade.py` puts it beside the one DL6MAA keyed.
    """
    filled = spec.field_fill(n) if not payload else payload[:n].ljust(
        n, bytes([spec.IDLE]))
    return filled + bytes([status & 0xFF])


CONTROL_STAGGER_S = 0.005
"""Half a symbol, the carrier stagger measured at 5.06 +/- 0.05 ms on 34 real
control bursts from three stations."""


def pulse_lead(audio: np.ndarray) -> int:
    """Leading carrier's phase-reference center after the driver's 2% trim.

    Reserve this before the pulse boundary, rather than placing the beginning
    of filter padding on the boundary and sending every symbol late.

    Taken from the waveform being keyed rather than from its codeword, because
    a control moved onto the peer's raster crosses that threshold at its own
    sample: the trim is an instantaneous test and the carriers are what moved.
    """
    trim = int(np.flatnonzero(abs(audio) > .02 * max(abs(audio)))[0])
    return int(np.argmax(modem.matched_pulse(spec.SAMPLE_RATE // 100))) - trim


@functools.lru_cache(maxsize=48)
def control_burst(cs_index: int, *, tail: bool,
                  swapped: bool | None) -> np.ndarray:
    """One control codeword on channels 5 and 12, built to a named shape.

    `tail` repeats the twentieth data symbol at full amplitude -- the
    twenty-second symbol every measured emitter sends, at a phase step of
    +0.4 degrees. `swapped` places the half-symbol carrier stagger: False leads
    with channel 5, True leads with channel 12, and None keys both tones on one
    clock, an arrangement no measured station sends. Neither changes the
    nominal 1.25/3.75 s cycle.
    """
    w = spec.CONTROL_SIGNALS[cs_index]
    bits = np.array([(w >> i) & 1 for i in range(spec.CS_BITS_PER_TONE)], np.uint8)
    syms = modem.differential_encode(bits, 1)
    if tail:
        syms = np.r_[syms, syms[-1]]
    audio = modem.modulate_tones(
        {5: syms, 12: syms},
        modem.ModConfig(sample_rate=spec.SAMPLE_RATE, matched_pulse=True),
        delay=None if swapped is None else
        {5 if swapped else 12: round(CONTROL_STAGGER_S * spec.SAMPLE_RATE)})
    audio.flags.writeable = False
    return audio


@functools.lru_cache(maxsize=48)
def control_burst_pulse_lead(cs_index: int, *, tail: bool,
                             swapped: bool | None) -> int:
    return pulse_lead(control_burst(cs_index, tail=tail, swapped=swapped))


def control_signal(cs_index: int, *, swapped: bool = False) -> np.ndarray:
    """A measured SCS control: staggered DBPSK tones and one repeated tail symbol.

    PIII_Complete_1's controls exchange the leading carrier every cycle. The
    home arrangement has channel 5 leading channel 12 by 5 ms; swapping moves
    that clock with its virtual carrier. CS6 at 17.42 and 54.92 seconds has the
    swapped arrangement.
    """
    return control_burst(cs_index, tail=True, swapped=swapped)


def control_pulse_lead(cs_index: int, *, swapped: bool = False) -> int:
    return control_burst_pulse_lead(cs_index, tail=True, swapped=swapped)


def historical_control_signal(cs_index: int) -> np.ndarray:
    """September 11's progressing WS8EOC control, for explicit A/B trials.

    This preserves 7129bb83's synchronous carriers and original ending. It is
    a local interoperability baseline, not the measured SCS control template.
    """
    return control_burst(cs_index, tail=False, swapped=None)


def historical_control_pulse_lead(cs_index: int) -> int:
    """Actual pulse offset after trim, even when aiming the audio start."""
    return control_burst_pulse_lead(cs_index, tail=False, swapped=None)


BREAKIN_CS = 2
"""CS3's index into `spec.CONTROL_SIGNALS`. `arq.CS_BREAKIN` is the same number
read from the link layer, which a renderer is below and cannot ask."""


def changeover_packet(payload: bytes, status: int, *,
                      cfg: modem.ModConfig | None = None,
                      swapped: bool = False) -> np.ndarray:
    """The packet a station takes the channel with: CS3 and a field, one keying.

    `link_packet`'s opposite number, and the one thing on a PACTOR-III link that
    is not keyed at a speed level: `CHANGEOVER` is the same two carriers and the
    same three bytes whatever level the traffic behind it runs at, so a station
    breaking in does not have to guess what the peer will follow it down to. Both
    of DL6MAA's are that shape at speed levels 3 and 5 respectively.

    CS3 IS NEVER A BARE BURST. The codeword is the packet's first twenty symbols
    and the field follows it with no gap and no re-key, which is what makes the
    break-in safe: a peer reading its answer slot finds a control signal there and
    stops transmitting, and the same 0.81 s of carrier carries the first bytes of
    what the new sender has to say. A bare CS3 leaves it nothing to switch to
    receive FOR -- the mistake `onair.RadioTx.send_p1_breakin`'s docstring records
    against the PACTOR-1 renderer, and this is the same packet a level up.

    BOTH CARRIERS ARE NOT KEYED TOGETHER, and that is measured here rather than
    carried over: on `rf-corpus/PIII_Complete_1` the IRS's changeover splits
    channel 5 ahead of channel 12 by 0.512 symbol and the ISS's by 0.505, on the
    same instrument that reads our own one-clock render at 0.001. Nine of ours
    were keyed on one clock and none was ever acknowledged. `CASE0_STAGGER` is
    the switch back.

    The head and the frame do NOT share a sign rule. Both ride the +-45 degree
    diagonal, as both stations on the air key them, but the codeword keeps the
    ordinary `BIT_PHASE` -- a zero at 315 degrees, which is where `rx.cs_bits`
    slicing the real axis needs it -- while the frame behind it takes case 0's
    inverted one. Keying the head through the frame's map complements every bit of
    it, which lands eight from the nearest codeword: a break-in nobody reads as
    one, on a channel the peer goes on transmitting into.
    """
    cfg = cfg or protocol_config()
    info = field_info(payload, CHANGEOVER.crc_bytes - 3, status)
    cell_order = replace(CHANGEOVER, tones=p3frame.VH_ORDER)
    rows = grid_steps(case0_cells(info, CHANGEOVER).reshape(
        CHANGEOVER.n_symbols, 2, 1), cell_order)

    w = spec.CONTROL_SIGNALS[BREAKIN_CS]
    bits = np.zeros(CHANGEOVER_HEAD_SYMBOLS, np.uint8)
    bits[:spec.CS_BITS_PER_TONE] = [(w >> i) & 1
                                    for i in range(spec.CS_BITS_PER_TONE)]
    head = np.array([BIT_PHASE[0], BIT_PHASE[1]])[bits]

    offsets = (CHANGEOVER.clock_offsets(cfg.sps) if CASE0_STAGGER
               else (0,) * len(CHANGEOVER.tones))
    tones, delay = {}, {}
    for rank, cn in enumerate(CHANGEOVER.tones):
        tone = spec.CARRIER_SWAP[cn] if swapped else cn
        start = np.angle(p3frame.PATTERN_A[0]) + START_PHASE[tone]
        tones[tone] = np.exp(1j * np.cumsum(
            np.concatenate([[start], head, rows[cn]])))
        delay[tone] = offsets[rank]
    return modem.modulate_tones(tones, cfg, delay=delay)
