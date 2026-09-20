# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-III protocol constants — the single source of truth.

The waveform is published. Citations read `PT-III §N` and refer to SCS, "The
PACTOR-III Protocol" (Helfert & Rink); ITU-R Rec. M.1798 reproduces the same
material with the same section structure, so either document resolves them.

A handful of constants the specification describes but does not tabulate — the
six control-signal codewords, the header transmit transform — are stated here
with the property that pins them, because that property is what a reader needs
to check the value rather than take it on faith.

Values that are neither published nor pinned are NOT guessed here. They live in
`unknowns.py` with their candidate sets, so a value merely *assumed* can never be
mistaken for one the specification states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Protocol(StrEnum):
    """Which PACTOR level a burst is in. One name, spelled once.

    A StrEnum, and the members are the exact strings `rxfront.Event.protocol`
    reports, so a decoded frame names a member directly and there is no lookup
    table between what a decoder said and what the link layer believes. The
    protocol used to be a boolean called `p1_link`, which could say "PACTOR-1" and
    "not PACTOR-1" and had no room for the third answer at all.

    PACTOR-4 is deliberately absent: nothing in this package reads or writes it,
    and a member that no decoder can ever produce is a name for a thing that does
    not exist here.
    """

    PACTOR1 = "PACTOR-1"
    PACTOR2 = "PACTOR-2"
    PACTOR3 = "PACTOR-3"

# ---------------------------------------------------------------------------
# Physical layer
# ---------------------------------------------------------------------------

SYMBOL_RATE_BD = 100.0
"""Symbols/sec, identical on every speed level. PT-III §2."""

TONE_SPACING_HZ = 120.0
"""PT-III §2."""

TONE0_HZ = 480.0
"""Frequency of channel number 0 (the "lowest" channel). PT-III §2."""

N_CHANNELS = 18
CENTER_FREQ_HZ = 1500.0
"""Center of the whole signal, and consistent with the channel plan: the 18
channels run 480..2520 Hz, whose midpoint is 480 + 120*8.5 = 1500 Hz exactly.
PT-III §2."""

EMISSION_DESIGNATOR = "2K20J2D"
"""PT-III §1."""


def channel_freq_hz(cn: int) -> float:
    """Center frequency of channel `cn` (0..17)."""
    if not 0 <= cn < N_CHANNELS:
        raise ValueError(f"channel number out of range: {cn}")
    return TONE0_HZ + TONE_SPACING_HZ * cn


# Channels carrying the *variable* packet headers and the control signals.
# Every speed level includes both -- checked in tests/shrike/test_spec.py.
VH_CHANNELS = (5, 12)
"""PT-III §4."""


# ---------------------------------------------------------------------------
# Speed levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeedLevel:
    sl: int
    channels: tuple[int, ...]   # channel numbers in use
    bits_per_symbol: int        # 1 = DBPSK, 2 = DQPSK
    constraint_length: int      # K
    code_rate: tuple[int, int]  # (num, den)
    pdr_bps: int                # physical (raw) data rate
    ndr_bps: float              # net uncompressed user data rate
    crest_factor_db: float
    payload_short: int          # usable payload bytes, 1.25 s cycle
    payload_long: int           # usable payload bytes, 3.75 s cycle

    @property
    def modulation(self) -> str:
        return "DBPSK" if self.bits_per_symbol == 1 else "DQPSK"

    @property
    def n_tones(self) -> int:
        return len(self.channels)


# Tone maps from the figure "number and position of the used channels", PT-III
# §2. Read by column position: a plain-text rendering of that figure loses the
# column alignment and produces maps that look plausible and are wrong.
#
# Cross-checks that all pass (tests/shrike/test_spec.py):
#   * n_tones * bits_per_symbol * 100 Bd == PDR, for every SL;
#   * every map is symmetric about channel 8.5 (i.e. about 1500 Hz);
#   * every map contains channels 5 and 12 (needed for VH + CS).
SPEED_LEVELS: dict[int, SpeedLevel] = {
    1: SpeedLevel(1, (5, 12),
                  1, 9, (1, 2),  200,   76.8, 1.9,    5,   36),
    2: SpeedLevel(2, (3, 5, 7, 10, 12, 14),
                  1, 7, (1, 2),  600,  247.5, 2.6,   23,  116),
    3: SpeedLevel(3, tuple(range(2, 16)),
                  1, 7, (1, 2), 1400,  588.8, 3.1,   59,  276),
    4: SpeedLevel(4, tuple(range(2, 16)),
                  2, 7, (1, 2), 2800, 1186.1, 3.8,  122,  556),
    5: SpeedLevel(5, tuple(range(1, 17)),
                  2, 7, (3, 4), 3200, 2039.5, 5.2,  212,  956),
    6: SpeedLevel(6, tuple(range(0, 18)),
                  2, 7, (8, 9), 3600, 2722.1, 5.7,  284, 1276),
}
"""Speed-level table.
PT-III §3."""


# ---------------------------------------------------------------------------
# Sample rate
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
"""The one working sample rate, and the module every other one chains to.

It was declared eight times under three different names -- `FS`, `FS_DEFAULT` and
`DEFAULT_SAMPLE_RATE` -- with the last a float while the rest were ints. Nothing
had gone wrong yet, but a rate that appears eight times is a rate that can differ
in seven places, and an int is what a sample index wants.

It lives in `spec` because `spec` imports nothing: every module can chain here
without a cycle. It is NOT a protocol constant -- PACTOR does not specify a sound
card -- but it is the audio-side convention the whole package shares, and
`modem.ModConfig.sample_rate` remains a real parameter for anything that needs to
run at another rate.
"""


# ---------------------------------------------------------------------------
# Cycle timing
# ---------------------------------------------------------------------------

CYCLE_SHORT_S = 1.25
CYCLE_LONG_S = 3.75
# PACTOR-1's cycle, itemised. The description gives all four numbers and they
# add up: a packet occupies 0.96 s and the answer window the remaining 0.29 s.
#
#   Gesamt-Zyklusdauer                 : 1.25 sec
#   Paketdauer                         : 0.96 sec
#   Fenster fuer Kontrollsignalempfang : 0.29 sec
#   Kontrollsignaldauer                : 0.12 sec
#
# Separate from the cycle length because shrike listened for
# 1.2 s inside a 1.25 s cycle and then transmitted for another second, which is
# not a raster at all -- the operator could hear us keying over the far end.
P1_PACKET_S = 0.96
CS_WINDOW_S = 0.29
P1_CS_S = 0.12
CYCLE_SHORT_LONGPATH_S = 1.4
CYCLE_LONG_LONGPATH_S = 4.2
"""PT-III §3."""

PULSE_SLOT_S = 0.010
""""Every packet and CS is preceded by a single phase reference pulse. All pulses
occupy a time slot of 10 ms." -- i.e. exactly one symbol period at 100 Bd.
PT-III §4."""

CS_BITS_PER_TONE = 20
"""Six 20-bit controls on channels 5 and 12: forty coded bits (PT-III §4).

Recorded SCS controls stagger the carriers by 5 ms and repeat the last symbol
as runout; neither adds a coded bit. See placement.control_signal.
"""


# ---------------------------------------------------------------------------
# Packet headers
# ---------------------------------------------------------------------------

# Sixteen 32-bit "variable packet headers", sent alternately on channels 12 and 5,
# channel 12 first (`p3frame.VH_ORDER`, MEASURED). They encode 4 bits, and the
# numbering below is 0-BASED off the index -- PT-III §4 counts from one, which
# read literally puts the speed level and the cycle length one bit too high:
#
#   bit 0   request/ACK phase, matching status-byte sequence parity in the
#           recorded short and long frames; independent of physical carrier swap
#   bits 1-2  speed level, mod 4 -- SL5/SL6 are separated by also analysing the
#             constant headers
#   bit 3   cycle duration, 0 short and 1 long
#
# `p3frame.variable_header` builds exactly these positions, and they are what
# reads the reference recording's long cycles.
# PT-III §4.
VARIABLE_HEADERS: tuple[int, ...] = (
    0x1873174F, 0xFC0F6047, 0x0A4C7EA7, 0x09BCE11F,
    0x8E67C43C, 0x7268A47B, 0x842BBA9B, 0x87DB2523,
    0x4D55AA6A, 0xB15ACA2D, 0x4719D4CD, 0x44E94B75,
    0x3CCD91A9, 0xC0C2F1EE, 0x3681EF0E, 0x357170B6,
)

# Sixteen 16-bit "constant packet headers" for the 16 non-VH channels. They carry
# no information; they exist for QRG tracking, memory-ARQ, listen-mode, and to
# distinguish SL5 from SL6.
# PT-III §4.
#
# !! DOCUMENTED ANOMALY, DO NOT "FIX" SILENTLY !!
# CH7 and CH11 are BOTH 0x5a3c in SCS's own PDF (and in the ITU reproduction).
# Sixteen headers exist to characterise sixteen distinct channels, so a genuine
# duplicate makes two channels indistinguishable -- and that is what it is, not a
# typo. Off-air corroboration, which is what this note used to ask for: on a real
# packet each of the other fourteen channels matches its own word alone, while
# the two channels carrying 0x5a3c match CH7 and CH11 at exactly equal strength.
# See unknowns.py:CH_DUPLICATE.
CONSTANT_HEADERS: tuple[int, ...] = (
    0xC324, 0xF987, 0xB1C8, 0xF370,
    0x801D, 0x7C3D, 0xD8F1, 0x5A3C,
    0x792D, 0x8397, 0x33AA, 0x5A3C,   # <-- CH11 duplicates CH7 in the source
    0x823C, 0x073F, 0xF798, 0xD801,
)


# ---------------------------------------------------------------------------
# Carrier swap (frequency diversity)
# ---------------------------------------------------------------------------

# "the digital data stream that constitutes a specific virtual carrier is swapped
# to a different tone with every ARQ cycle". The published pairing is an
# involution over all 18 channels, and it maps 5 <-> 12, so the header/CS
# channels stay the header/CS channels. Checked in tests/shrike/test_spec.py.
# PT-III §2.
CARRIER_SWAP: dict[int, int] = {
    0: 17, 1: 16, 2: 9, 3: 10, 4: 11,
    5: 12, 6: 13, 7: 14, 8: 15,
}
CARRIER_SWAP.update({v: k for k, v in list(CARRIER_SWAP.items())})


# ---------------------------------------------------------------------------
# Sub-band lead
# ---------------------------------------------------------------------------

SUBBAND_LEAD: dict[int, tuple[float, ...]] = {1: (0.5, 0.0),
                                              2: (0.5,) * 3 + (0.0,) * 3}
"""Speed level -> symbols each of its virtual carriers runs AHEAD of the others.

Speed level 2 lights six channels that on the air are two three-tone clusters,
840/1080/1320 Hz and 1680/1920/2160 Hz, and they are NOT transmitted together:
the low cluster leads by half a symbol, phase reference, header block and data
field alike. Speed level 1 is two carriers and splits them the same way, channel
5 ahead of channel 12. Levels 3 to 6 are one comb on one clock, so they have no
entry here and `placement.Path.subband_lead` reads that as zeros.

Indexed by the carrier's RANK in the level's tone tuple rather than by channel,
because the lead belongs to the virtual carrier and `CARRIER_SWAP` moves a
virtual carrier to its partner tone every ARQ cycle while leaving its rank alone.
On the home arrangement ranks 0-2 are channels 3, 5 and 7; on a swapped cycle
they are 10, 12 and 14, and the lead goes with them. One entry serves both cycle
lengths because a speed level's tone tuple is the same on each.

MEASURED, on two real off-air level 2 packets, per carrier and with no frame
geometry in it: each lit channel's header block carries a known eight-dibit word,
so the instant a channel best matches its own word is that carrier's symbol
clock. The low cluster came out 0.46 and 0.52 symbol early, with 0.013 to 0.034
symbol of scatter inside each cluster. Five real level 3 packets and two real
level 6 ones measure 0.02 to 0.09 across all their carriers, which is what no
lead looks like on the same instrument.

MEASURED at speed level 1, on the only entry packet a peer is known to have
acquired and on both changeover frames of the same session. The instrument is
the level-2 one -- a carrier's own known differential steps scored against the
recording at each trial clock -- read at the CENTRE of the plateau rather than
at its argmax, because 24 of a changeover's 80 steps are the antipodal CS3 word
and the score holds a flat top most of a symbol wide across them. Channel 5
leads channel 12 by 0.514 symbol on the entry, 0.512 and 0.505 on the two
changeovers, against 0.001 on our own one-clock render read back the same way.
Whole-waveform correlation of our entry against the recording rises 0.742 ->
0.828 with the lead in, peaks at exactly half a symbol, and falls to 0.535 with
the sign reversed -- against a ~0.83 ceiling set by the tape's own SNR.

MEASURED, from outside: an independent decoder, eight cycles a file, counts of
accepted level 2 frames with repeats behind every column, because that decoder is
not deterministic at this level. The first three columns are a hand-rolled render
in the home arrangement; the last two are `placement.data_packet` keyed by
`onair.RadioTx` with the carrier swap alternating as a link does, four cycles in
each arrangement.

                                        hand-rolled     transmit path
    all six carriers together            0, 0, 0          0, 0
    low cluster half a symbol EARLY      4, 8, 8          8, 8, 8
    low cluster half a symbol LATE       0, 0, 0

THE SIGN IS THE OTHER WAY ROUND FROM THE ONLY PUBLISHED STATEMENT OF THIS
MECHANISM, and a reader who assumes otherwise gets a waveform that reads as
nothing at all. PACTOR-4 §11.7 says of its own two-carrier speed level 1 that
"symbols on the carrier with the lower frequency always appear delayed by T / 2",
and carries the half symbol through its packet-length arithmetic in §11.11. Here
the lower carriers are EARLY, not delayed; late was rendered and offered, and it
scores zero. Beyond half a symbol nothing is settled: a whole symbol has been
measured at 6 of 8 in one session and 0 of 8 in another, and one and a half
symbols scores zero.

INFERRED, and marked so because no published PACTOR-III text says anything
whatever about carrier timing:

  * that the figure is exactly one half. Two off-air packets bracket it, and half
    a symbol is the only offset that has reached a full score from outside.
  * that this is PACTOR-4's mechanism seen at a different speed level rather than
    a coincidence. The waveforms are not the same -- level 1 there is two 66.66 Bd
    carriers, this is two clusters of three at 100 Bd -- and the sign differs, so
    the only thing carried across is that the house staggers carriers at all.
  * that it is there for the envelope. PACTOR-III's own specification claims a
    crest factor "fairly comparable to single-carrier modes" up to speed level 4,
    2.6 dB at level 2 on six tones, which equal-amplitude simultaneous multitone
    does not reach; staggering symbol transitions so the envelope dips do not
    coincide is the standard way to buy it. That is read off a published
    PERFORMANCE figure, not off a published statement of how it is achieved.
"""


# ---------------------------------------------------------------------------
# Status byte
# ---------------------------------------------------------------------------

class DataType:
    """Status-byte bits 2-4.
    PT-III §4.

    THREE BITS FROM PACTOR-2 ON. PACTOR-1's own field is bits 2-3, so only the
    first four values reach that layer -- the PMC modes need a bit the 1990
    description leaves unassigned. `p1rx.Packet.data_type` reads the narrow field.
    """
    ASCII_8BIT = 0b000       # <-- the transmitter sends ONLY this; the receiver
                             #     reads every mode a peer declares (compress.py)
    HUFFMAN = 0b001
    HUFFMAN_SWAPPED = 0b010
    RESERVED = 0b011
    PMC_GERMAN = 0b100
    PMC_GERMAN_SWAPPED = 0b101
    PMC_ENGLISH = 0b110
    PMC_ENGLISH_SWAPPED = 0b111


IDLE = 0x1E
"""The character a short data field is padded out with, and the padding a
receiver drops. RS -- "dient als IDLE-Symbol", not 0x00.

It lives here rather than in `pactor1` because it is a property of the character
stream and not of one protocol level: the PACTOR-1 renderer has always padded
with it and its receiver has always stripped it, and PACTOR-3's fixed field needs
exactly the same thing. Without it PACTOR-3 carried a length byte of its own,
which no other implementation reads and which cost a payload byte a packet."""


TEMPLATE = bytes((0x0F, 0x8F, 0x87, 0xC7, 0xC3, 0xE3, 0xE1, 0xF1,
                  0xF0, 0x78, 0x78, 0x3C, 0x3C, 0x1E, 0x1E))
"""The walking-bit pattern a PACTOR-III station writes where it has no data.

MEASURED off `rf-corpus/PIII_Complete_1`, which is the only genuine PACTOR-III
reference we hold, and it is the same fifteen bytes in three different places:
the 62-byte header field of pactor3.md §14, the entry packet of §17.1
(`0f 8f 87 c7 c3 1a 66 89` -- five bytes of it, the status byte, the CRC), and
every field in that session a station had nothing to put in. Fourteen of the
recording's packets carry it and nothing else, at speed levels 1, 3 and 6, always
from byte 0 of the pattern. An independent monitor reports such a field as
`LEN: 0`: to the firmware the pattern IS empty.

Not `IDLE`. 0x1E was this package's own guess at what fills a PACTOR-3 field and
no receiver outside it has ever been shown one -- and an entry packet is the one
packet whose whole field is fill, keyed at a peer that has to acquire the
waveform cold off it."""


def field_fill(n: int) -> bytes:
    """`n` bytes of the pattern a station with nothing to say writes."""
    return (TEMPLATE * (n // len(TEMPLATE) + 1))[:n]


def field_payload(info: bytes) -> bytes:
    """A field's information bytes, less whatever fill the station wrote.

    A field that is nothing BUT the pattern carries nothing -- that is what the
    reference session's idle packets are and what our own empty field now is.
    Anything else keeps the old rule, trailing IDLE and no more: the two
    partly-filled fields in the reference resume the pattern at byte 4 and byte
    14 of it, which no offset rule reproduces, so there is nothing here to match
    them against and a loose match would eat a payload ending in `x` or `<`.
    """
    return b"" if info == field_fill(len(info)) else info.rstrip(bytes([IDLE]))


STATUS_SEQ = 0b0000_0011
STATUS_LONG_CYCLE = 0b0010_0000
STATUS_CHANGEOVER = 0b0100_0000
STATUS_QRT = 0b1000_0000
"""Masks for reading a received status byte; the assembler below is their inverse.
PT-III §4."""


def status_byte(packet_count: int, data_type: int = DataType.ASCII_8BIT,
                long_cycle_request: bool = False,
                changeover_request: bool = False,
                qrt: bool = False) -> int:
    """Assemble the status byte.

    bits 0-1: modulo-4 packet counter (detects repetitions)
    bits 2-4: data type / compression
    bit 5   : suggests switching to data mode
    bit 6   : changeover request
    bit 7   : QRT (link termination)
    """
    if not 0 <= packet_count <= 3:
        raise ValueError("packet counter is modulo-4")
    if not 0 <= data_type <= 7:
        raise ValueError("data type is 3 bits")
    return (packet_count
            | (data_type << 2)
            | (int(long_cycle_request) << 5)
            | (int(changeover_request) << 6)
            | (int(qrt) << 7))


# CRC: "16-bit CRC calculated according to the CCITT-CRC16 standard."
# The *exact* CCITT variant (init value, reflection, xorout) is NOT stated by the
# spec -- see unknowns.py:CRC_VARIANT.
# PT-III §4.
CRC_POLY = 0x1021


# ---------------------------------------------------------------------------
# Control signals -- the six 20-bit codewords (CS1..CS6)
# ---------------------------------------------------------------------------

# The six control signals are an equidistant code: all 15 pairwise Hamming
# distances are exactly 12 of 20 (24 of 40 over both tones), which is the Plotkin
# bound for six words at this length. That is the property to check a candidate
# table against -- a transcription slip in any one word breaks the equidistance
# immediately, and nothing else about the set would look wrong.
#
# CS1/CS2 = ACK/request, CS3 = break-in, CS4 = speed up, CS5 = NAK+speed down,
# CS6 = cycle-length toggle. Each CS is sent DBPSK on BOTH tone 5 and tone 12
# (same 20 bits ⇒ 40-bit block).
CONTROL_SIGNALS: tuple[int, ...] = (
    0x52D56, 0xAABA2, 0xAD45A, 0x95AC5, 0x74339, 0x4B4AD,
)

# Canonical index -> name for the six CS, so the modem, monitor and decoders all
# read the same label. arq.py maps these indices to FSM events.
CS_NAMES: tuple[str, ...] = ("ACK", "REQ", "BREAK-IN", "SPEED-UP", "NAK", "CYCLE-TOG")

# PACTOR-1's four are a DIFFERENT, older set and do not share these meanings, so
# labelling a decoded P1 codeword out of CS_NAMES above misreports it: index 3 is
# PACTOR-3's speed-up but PACTOR-1's CS4. CS1 and CS2 are BOTH acknowledgement;
# what acknowledges is the alternation between them, and repeating either is the
# request for a repeat. The monitor printed "SPEED-UP" for a station asking us to
# send the block again.
#
# CS4 is named for what it ASKS FOR rather than for either of the two things it
# means, because the decoder does not know which context it is in and "reject"
# read as a refusal to a station that had in fact just accepted the call. It
# answers a connect to accept it at 100 Bd (CS1 accepts at 200); it answers a
# faulty packet mid-session to discard it and have the information again at 100.
# Both are "go to 100 Bd". PACTOR-1 has no refusal codeword at all -- a station
# that will not talk to you transmits nothing.
#
# The last two are the words PACTOR-1 assigns no meaning, and they are named for
# themselves because there is nothing else true to call them. A decoder that folds
# them into the nearest control signal reports a six-error CS1 and throws the
# burst away -- see `pactor1.UNASSIGNED_SIGNALS`.
P1_CS_NAMES: tuple[str, ...] = ("CS1/ack", "CS2/ack", "CS3/break-in", "CS4/100Bd",
                                "0x6A9/unassigned", "0x59A/unassigned")


# ---------------------------------------------------------------------------
# Header transmit transform
# ---------------------------------------------------------------------------

def header_tx_form(logical: int, width: int) -> int:
    """Map a published (logical) header codeword to its on-air (transmit) form.

    On air the two middle nibbles of a header codeword are exchanged; that swap
    is the interleaving between the two header tones. For a 32-bit variable
    header the swap is bit-field [12:15] <-> [16:19]; for a 16-bit constant
    header it is [4:7] <-> [8:11]. It is its own inverse, so transmit form and
    logical form use the same function.

    One transform accounts for all sixteen published variable headers at once,
    which is what fixes the field positions: a different pair of nibbles would
    have to be wrong on at least one of the sixteen.
    """
    if width == 32:
        lo, hi = 12, 16
    elif width == 16:
        lo, hi = 4, 8
    else:
        raise ValueError("header width must be 16 or 32")
    a = (logical >> lo) & 0xF
    b = (logical >> hi) & 0xF
    cleared = logical & ~((0xF << lo) | (0xF << hi))
    return cleared | (b << lo) | (a << hi)


# Headers are DQPSK: the (transmit-form) codeword is taken 2 bits at a time as
# 16 dibits -> 8 DQPSK symbols per tone on tones 5 and 12. Headers stay DQPSK on
# every speed level, including the levels whose payload is DBPSK: the header is
# what carries the speed level, so it cannot be modulated according to it.
HEADER_MODULATION = "DQPSK"
