# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-3 ARQ link layer — transport-agnostic stop-and-wait FSM.

Structure follows kestrel's generic ARQ FSM (``hfmodem.kestrel.arq.fsm``): the
five states, the connect handshake, stop-and-wait with per-cycle ACK/NAK, speed
(gear) adaptation and graceful disconnect are the same skeleton — both VARA and PACTOR are stop-and-wait/ISS-IRS/retry/gear-shift links.
The pattern is shared and the code is not: four state machines, four gearshift
rules, and a hoist that would abstract over protocols whose event vocabularies
differ rather than over drafts of one interface [ARCHITECTURE.md].

What is PACTOR-SPECIFIC here (bound to the cracked PHY facts, spec.py):
  * CYCLE-LOCKED timing: the ISS sends exactly one data packet per 1.25 s ARQ
    cycle (3.75 s in data mode); the IRS answers with ONE 20-bit control signal.
    A receiver tracks that grid continuously from the connect and loses lock
    across an off-grid gap, so timing here is cycle-driven, not turnaround-timer.
  * CONTROL = the six 20-bit CS codewords (spec.CONTROL_SIGNALS), each sent DBPSK
    on tones 5 & 12. Semantics are published for CS3..CS6; the
    CS1/CS2 split ("used to acknowledge/request packets") is ours.
  * DATA sequence rides the STATUS BYTE mod-4 packet counter (spec.status_byte),
    NOT a separate marker byte. Memory-ARQ = the peer keeps soft copies of a
    failed packet and combines retransmissions (modelled here as a plain retry).
  * CONNECT is a PACTOR-1 FSK burst (shrike.pactor1). The link is PACTOR-1 until
    something upgrades it, and the upgrade is a DECISION, not a handshake: PACTOR
    has no capability field, no negotiation and no refusal, so capability is
    expressed by what a station transmits and a target has to be discovered by
    trying. This layer decides only WHEN -- `_on_ack`, once PACTOR-1 has carried a
    packet -- and `ptc.UPGRADE_TARGETS` decides which protocol and reverses the
    choice if the peer turns out to be speaking PACTOR-1 after all.

CHANGEOVER (ISS <-> IRS turnaround) — every Winlink session turns the link around
constantly, so this is the core of the link layer, not a garnish. Provenance,
kept explicit because a plausible guess here would pass a self-test and fail on
air:

  VERIFIED, ITU-R M.1798 / PT-III.pdf §4. The ISS asks to hand over by setting
  status-byte BIT 6, "indicates a changeover request", in a data packet; BIT 7
  "initiates the QRT protocol". The IRS takes the link by sending CS3, which
  "forces a break-in". Those are the only two changeover mechanisms on the wire.

  CS3 IS NOT A BARE BURST. "In contrast to AMTOR, CS3 is transmitted as head
  portion of a special changeover packet" -- it is the first 120 ms of the new
  ISS's own 960 ms packet, and the station yielding switches to receive on hearing
  it and reads the remaining 840 ms of that same transmission. A break-in is
  therefore emitted here as a PACKET carrying the `breakin` flag, not as a control
  signal, and an ISS that decodes one yields and then handles it exactly as the IRS
  would. The counter resets to 0 across the reversal. Both stations rotate the
  cycle grid by 960 - 120 = 840 ms in the same cycle; that half belongs to the PHY,
  see `onair._MasterGrid.reverse`.

  MEASURED, WS8EOC 2026-08-03 (captures/p1-ws8eoc-0803-*): bit 6 is a REQUEST
  and nothing more. This layer used to treat the ACK of the packet carrying it
  as the turnaround itself — a derivation from "no CS means changeover granted"
  — and a real gateway refutes that: it acknowledges the packet and REMAINS the
  IRS. In every session that reached an acknowledged bit-6 packet, the peer went
  on emitting 120-135 ms control signals (CS2 repeated, then CS4) and never
  transmitted a packet — zero 960 ms bursts, zero CRC-valid frames, across every
  listening window — while this end had already switched itself to IRS. Both
  ends then hold the receiving role and exchange control signals until the
  retry budget ends the link: one acknowledgement, then death, six times in six.

  So the changeover on the wire is ALWAYS the IRS's CS3-headed packet, in both
  directions of intent. An ISS asking to hand over keeps the link — idle
  packets, bit 6 standing — until the peer's break-in arrives; an IRS that
  decodes bit 6 answers by scheduling its own break-in. The counter reset and
  the grid rotation then ride the one changeover mechanism there is.

  INFERRED — see unknowns.ARQ_CHANGEOVER. Each end keeps its own speed level; a
  break-in requeues the ISS's unacknowledged packet; and a turnaround lost on
  the air (a changeover packet that never decodes) strands both ends believing
  they are the ISS. Since the link is half-duplex on a cycle grid, both then
  transmit and neither can hear the other, so nothing on the wire can break the
  deafness: an ISS that exhausts its retries yields the link once and listens
  instead of aborting, and only aborts if the peer stays silent after that too.
  DEAFNESS IS THE PREMISE and the budget is spent on silence alone -- an ISS
  holding the channel for a peer it can hear every cycle keeps holding it, bit 6
  standing, however long the peer takes to break in. See `_on_nak`.

The FSM is pure logic driven by events (host commands, decoded RX frames, cycle
ticks) through the injected :class:`ArqIO`; it renders no waveform itself.
"""
from __future__ import annotations

import os

from enum import StrEnum
from dataclasses import dataclass
from typing import Optional

from . import compress, pactor2, spec

# ---- control-signal semantics (spec.py) -------------------------------------
# CS index (0..5) -> logical ARQ event. The 20-bit codewords themselves are in
# spec.CONTROL_SIGNALS. CS3..CS6 are published verbatim; splitting CS1/CS2 into
# ACK and repeat-request is the reconstructed part.
#
# PUBLISHED, M.1798 §4, quoted in docs/protocols/pactor/pactor3.md §7: "CS3
# forces a break-in, CS4 demands an increase to the next higher speed level, CS5
# is a NACK asking for a repetition and a reduction to the next lower level, and
# CS6 toggles the cycle length". So CS4 raising the level is the specification's
# own direction and not a transcription slip -- the same index in PACTOR-1 is
# CS4/Speedchange, whose 100 Bd branch is a DROP, and the two tables must not be
# read through one another. See spec.P1_CS_NAMES.
CS_ACK        = 0     # CS1
CS_REQUEST    = 1     # CS2  (request repeat / keepalive)
CS_BREAKIN    = 2     # CS3  "forces a break-in" -- IRS takes the link [spec 4]
CS_SPEED_UP   = 3     # CS4  (acknowledge, and go one speed level faster)
CS_NAK        = 4     # CS5  (decode failed -> retransmit, drop a speed level)
CS_CYCLE_TOG  = 5     # CS6  (short<->long cycle)

P1_SPEED_LEVEL = 0
"""What the PACTOR-1 seam reports as a decoded packet's speed level.

PACTOR-1 has no speed level -- it has a bit rate, and the packet does not carry
it (`rxfront._p1_packet_event` reports 0). It is the seam the IRS's gear commands
are gated on: CS4 means "next higher speed level" in PACTOR-3 and "drop to 100
Bd" in PACTOR-1, so asking a PACTOR-1 peer to speed up asks it to slow down. It
is therefore NOT which protocol carried the packet; that reaches the link layer
as `on_rx_packet`'s own argument.

`onair._SessionRx._p2_packet` reported it for a PACTOR-2 frame too, on the
grounds that a station which cannot key PACTOR-2 has no gear to ask a PACTOR-2
peer for. It keys one now, so the frame's own level travels and `P2_LADDER` is
what the gear commands are counted against."""
_CS_NAMES = {CS_ACK: "ACK", CS_REQUEST: "REQ", CS_BREAKIN: "BRK",
             CS_SPEED_UP: "SPU", CS_NAK: "NAK", CS_CYCLE_TOG: "CYC"}

SEQ_MOD = 4                      # status-byte packet counter is mod-4 [spec 4]

ISS, IRS = "iss", "irs"

ENTRY_RUNGS = ("template", "burst", "data", "p4chirp", "p2sl1")
"""What `ArqConfig.entry_ladder` may be built out of, described there."""

ENTRY_RUNGS_GROUNDED = {
    "burst": "it renders 1.074 s -- the packet plus a 205 ms acquisition "
             "preamble -- into a 1.25 s cycle against a 210 ms listen floor, so "
             "`_keyable_slot` cannot aim it at consecutive boundaries and it "
             "keys one slot in two. On 2026-08-26 a third party's receiver "
             "measured it taking 6 of the 15 cycles it spanned, with the tail of "
             "every one of the six running into the gateway's answer",
}
"""Rungs this station cannot key well enough to offer, and the measurement why.

A LINK GOES AS FAR AS BOTH ENDS ALLOW, and our own ability is one of the two
bounds. This is where that bound is written down, so that "as far as we can" is
a fact an operator can read rather than a claim. `check_entry_ladder` refuses a
ladder holding one of these, at the constructor and at the command line, and the
way to fly it again is to fix the rung -- or the cycle it has to fit in -- and
take it out of here."""


def check_entry_ladder(rungs) -> tuple[str, ...]:
    """The ladder, or a ValueError naming the rung and the arithmetic."""
    rungs = tuple(rungs)
    for r in rungs:
        if r not in ENTRY_RUNGS:
            raise ValueError(f"entry rungs are {', '.join(ENTRY_RUNGS)}; not {r}")
        if r in ENTRY_RUNGS_GROUNDED:
            raise ValueError(f"the '{r}' entry rung does not fly: "
                             f"{ENTRY_RUNGS_GROUNDED[r]}")
    if not rungs:
        raise ValueError("an entry ladder needs a rung")
    return rungs


@dataclass(frozen=True)
class Ladder:
    """The speed levels of one protocol: what a packet carries at each rung.

    The ARQ layer is protocol-independent everywhere except here. Chunking,
    the CS4 climb, the CS5 drop and the long-cycle ask all need a number that
    belongs to the waveform the link is in, and every one of them used to read
    `spec.SPEED_LEVELS` -- PACTOR-3's six rungs -- whatever the link was
    running. PACTOR-2 has four rungs and smaller fields at every one of them --
    32 bytes at speed level 3 against PACTOR-3's 59, and 156 long against 276 --
    so a PACTOR-2 link chunked to PACTOR-3's table hands its renderer nearly
    twice what the field holds and settles all of it. `PactorArq._sent` is what
    caught that same fault inside one protocol, and it can only report a
    truncation the seam admits to.

    `payloads` is `(short, long)` per rung, `long` None where that rung has no
    long frame at all. `top` is the highest rung this station will CLIMB to,
    which is not always the last of them. `entry_sl` is where a phase of this
    protocol opens, or None to take `ArqConfig.entry_sl`.
    """

    protocol: spec.Protocol
    payloads: tuple[tuple[int, Optional[int]], ...]
    top: int
    entry_sl: Optional[int] = None

    def payload(self, sl: int, long: bool) -> int:
        short, extended = self.payloads[sl - 1]
        return extended if long and extended is not None else short

    def long_frame(self, sl: int) -> Optional[int]:
        return self.payloads[sl - 1][1]


P3_LADDER = Ladder(
    spec.Protocol.PACTOR3,
    tuple((lv.payload_short, lv.payload_long)
          for lv in spec.SPEED_LEVELS.values()),
    top=6)
"""PACTOR-3's six rungs, `spec.SPEED_LEVELS` read as a ladder.

All six levels carry the published long field. SL1's 36-byte case-0 waveform
is independently checked by the SCS monitor on four valid fields in both
carrier orders, with four bad-CRC controls rejected.
A CS6 grant must retain the negotiated cycle even at the lowest level."""

P2_LADDER = Ladder(
    spec.Protocol.PACTOR2,
    tuple((short.crc_bytes - 3, long.crc_bytes - 3)
          for short, long in zip(pactor2.PATHS, pactor2.PATHS_LONG)),
    top=3, entry_sl=1)
"""PACTOR-2's four rungs, off the frame geometry: 5/14/32/59 short and
36/76/156/276 long, the field less its status byte and its CRC.

TOP IS 3 AND THE LADDER HAS FOUR. Speed level 4 renders and does not survive
even our own front end: its 16-DPSK cells sit closer together than the analysis
kernel's residual phase error, so nothing at that level is validated in either
direction and `_gear_cs` may not ask a peer for it. The rung stays in the table
because a PEER may transmit at it and the receiver reads the level off the frame
marker either way.

ENTRY AT THE FLOOR, which is not what `ArqConfig.entry_sl` does for PACTOR-3.
That constant is 3 because levels 1 and 2 are not acquirable from a standing
start -- level 1's header block is the variable header alone. PACTOR-2 keys a
nine-pulse frame marker in front of every burst and `p2rx.find_markers` reads
the level AND the frame length back off it at any rung, so the reason does not
reach here: the floor is DBPSK on both carriers, it is the most robust rung
there is, and CS4 climbs off it."""


LADDERS = {spec.Protocol.PACTOR2: P2_LADDER, spec.Protocol.PACTOR3: P3_LADDER}
"""Which ladder a link in each protocol chunks to. PACTOR-1 has none -- it has a
bit rate, and `ptc.PtcHost._p1_field` overrides the field size outright -- so a
PACTOR-1 link keeps whichever ladder it last held, exactly as it keeps a stale
speed level."""


LONG_TICKS = 3
"""Grid slots to one long ARQ cycle: 3.75 s is exactly three of the 1.25 s short
cycle, so the driver keeps ticking on the short raster and this layer acts on
every third tick while the link runs long. The reference session keeps one
raster across both lengths -- its long-cycle packets sit on the same phase
comb as its short ones, 3.750 s apart (PIII_Complete_1, 17.86 through 51.62 s)
-- which is what lets the cycle length change without a resynchronisation.
"""

UPGRADE_SILENCE_CYCLES = 4
"""Consecutive unanswered cycles after an upgrade before falling back.

SILENCE IS THE ONLY REFUSAL PACTOR HAS. There is no capability field, no
negotiation and no codeword meaning "I cannot do that" -- a station declines a
mode by never transmitting it -- so a peer that could not follow an upgrade tells
us by going quiet, and nothing else. Both ends are then keying waveforms the
other is not reading, and the link is dead in both directions with nothing on the
air to say so. 83 of 558 Winlink PACTOR channels are PACTOR-2 only and cannot
follow a PACTOR-3 upgrade at all, so this is a third of the network rather than
an edge.

AND IT IS THE WHOLE OF THAT DECISION -- which is what it was written to be and
was not. `ptc.PtcHost._follow_peer` reads a PACTOR-1 frame on an upgraded link as
the peer's contradiction, and a peer that answers at all reached it first: on the
cycle after the upgrade, with a codeword it had built from the cycle before, so
the target was ruled out before the peer had seen a single PACTOR-3 frame and
this count never ran. `ptc.PtcHost._stale_pactor1` now holds the contradiction
back until the far end has been heard in the protocol the upgrade moved to, and
the PACTOR-3 packet repeats into the count until then. Those repeats are what
memory-ARQ is for -- a receiver combines its looks at a marginal frame, and one
look is the least a new waveform can be acquired on.

FOUR, and it is the protocol's own number for this shape of decision -- a rate
change the peer may not be following, backed off on silence alone. It is
hf-pactor's `PACTOR_RETRY_HISPEED`, the budget its 200 Bd states run on against
thirty before a link is dropped altogether (pactor1-timing.md §4, which carries
which role spends which). One silent cycle is ordinary on HF; four in a row, from a
station that owes a control signal EVERY cycle, is not. It is also deliberately
below `ArqConfig.max_retries`, so the fallback is spent before the
yield-and-listen recovery, which cannot help against a peer that simply cannot
read us. It is also what a PACTOR-1-only peer costs, once per link: four cycles
of a waveform it cannot read, and then PACTOR-3 is ruled out and the link is
PACTOR-1 for the rest of the contact.

It counts only while an upgrade is UNCONFIRMED: any decoded frame that reaches
the FSM disarms it, so a fade later in the session cannot flap a link that was
working. A PACTOR-1 frame inside the window is not one of those and does not
reach here at all. Presence alone (`note_peer_heard`) does not disarm it -- a
peer we can hear and cannot decode after an upgrade is precisely the signature
this exists for.

It is not the only way the window ends, and the other one is not silence at all:
see `note_upgrade_unread`. This count is spent on quiet cycles only."""

ENTRY_GRANT_CYCLES = int(os.environ.get("HFMODEM_ENTRY_GRANT_CYCLES", "14"))
"""Repeated grants an unconfirmed entry campaign may draw before the
repeated-grant rule (`note_upgrade_unread`'s count, not silence) ends it.

RAISED FROM EIGHT ON THE STRENGTH OF WHAT EIGHT NEVER LET US SEE. Three arms
against WS8EOC on 2026-09-11, on two bands, all keyed the identical entry
waveform. The only sign the peer's behaviour ever changed at all was its own
PACTOR-1 answer position going quiet, at cycles 11-12 of a twelve-entry run on
80 m. The 40 m arm never reached that point: this rule, at the old value of
`ArqConfig.max_retries` (8), ended it two cycles short, at ten keyed entries.
So no arm has ever run long enough to see what a granting gateway does past a
dozen tries -- every one of them gave up first. FOURTEEN clears cycle twelve
with margin instead of falling two short of it.

ITS OWN CONSTANT, DELIBERATELY NOT `max_retries`. That field is spent
everywhere an ordinary packet goes unacknowledged; widening it to let one
entry campaign run longer would loosen every other retry ceiling on the link
along with it, which nothing measured calls for. This bound belongs to the one
decision it was raised for.

IT DOES NOT MOVE `UPGRADE_SILENCE_CYCLES`, and does not need to: a peer that
answers every cycle never touches the silence count at all -- a repeated grant
resets `_unanswered_upgrade` to zero -- so a gateway that keeps granting is
bounded by this constant alone, and one that goes quiet is still caught by the
silence rule at its usual four cycles, unchanged. The two counts do not share
a cycle, and `_upgrade_requests` never resets on silence either, so a peer
that alternates grants and quiet stretches is still bounded by this constant
in total across the whole campaign -- raising it is what widens the window,
and nothing about the silence rule has to move to reach fourteen.

STILL A LOCAL BUDGET, NOT A CAPABILITY VERDICT ON ITS OWN TERMS -- see
`entry_ladder`. The final ladder rung draws this many requests; earlier rungs
advance after `UPGRADE_SILENCE_CYCLES` apiece (`_step_entry`)."""

UNREPAIRED_BUDGET = 2
"""Packets that reached us and would not decode, answered with a NAK before the
link is ended deliberately.

The NAK is right and it is what PACTOR gives us: memory-ARQ keeps the samples,
the peer sends the packet again, and a fade over one packet is repaired. What was
missing is the end of it. A repeat that fails the same way is no longer the
channel's fault -- it is this receiver's, and the peer cannot mend that by
transmitting more of the same. Left unbounded it also blinded the budget that
would otherwise have caught it: a bad-CRC packet sets `_rx_this_cycle`, which
zeroes `_silent_cycles`, so a peer whose every packet was unreadable read as a
perfectly live link for as long as it kept keying.

Two, for the reason the count is consecutive: the first repeat is the one worth
asking for and the second is the benefit of the doubt. Past that we are spending
a shared band, and a gateway's transmitter as well as our own, trading NAKs for
nothing.
"""

GOODBYE_CYCLES = 4
"""Repeats of the QRT packet against a peer that has stopped answering.

A LINK THIS END GIVES UP ON IS STILL UP AT THE OTHER END, and PACTOR gives the
far station nothing to time that out with: the only figure in the corpus is the
30 s of grid extrapolation a station will still answer a late QRT across
(pactor1-timing.md §5), which bounds how stale a goodbye may be and not how long
to wait for one. So a peer left without it holds the channel until an operator
ends the session by hand -- WS8EOC spent 34 cycles asking a vanished station for
its next packet on 2026-08-03 and then signed off in CW.

Four cycles is five seconds, well inside the 30 s the far end will still answer
across, and the same cap `onair.QRT_CYCLES` already puts on a host-commanded
teardown. Counted in cycles from the QRT reaching the air (`on_cycle`) rather
than in retries of the packet carrying it: a peer that breaks in takes the link
back and clears that packet with it, so a retry count is not a clock the far end
cannot stop.
"""

GOODBYE_PLACE_TICKS = GOODBYE_CYCLES * LONG_TICKS
"""Base-raster ticks allowed to place the first terminal packet: 15 seconds.

Starts at a terminal disconnect request, independently of packet retries and
peer traffic. A refused emission spends no retry, but cannot lease the link
indefinitely. Once a goodbye keys, GOODBYE_CYCLES owns its answer timeout.
"""


UNREAD_RUNG_REPEATS = 16
"""Answered repeat requests one PACTOR-3 data packet may draw, after an entry
packet was keyed and before any ordinary field of the new waveform has been
acknowledged, before the link retreats to PACTOR-1.

A REPEAT TRAIN AGAINST AN UNREAD RUNG CANNOT END ON ITS OWN. `_on_nak`'s budget
is right to forgive a repeat request -- the reverse channel is working -- but
pactor3.md §7 reads the same codeword from the other side: "a station that
answers a peer with the same codeword every cycle is telling it, correctly, to
send the same packet forever". A peer that read the entry packet and has never
taken one packet of the rung behind it is doing exactly that, and the ISS has no
protocol remedy: the level is the IRS's to command (CS5), and a held CS1
commands nothing. KB5LZK, 2026-09-11, twenty-seven keyings of one field.

SIXTEEN, AND IT IS TWICE THE SILENCE BUDGET. The two counts measure different
failures of the same cycle -- `ArqConfig.max_retries` a cycle the peer did not
answer, this one a cycle it answered by asking again -- and a fade that takes
out the forward path alone runs the second without the first, so a bound at the
silence budget would cut an 80 m night path that is merely slow. A first data
packet is also the one worth the most patience: the peer has to acquire the rung
on it, which is close to the same allowance `on_cycle` already gives the entry
packet (`ENTRY_GRANT_CYCLES` requests while `entry_pending`), spent twice over.

AND IT IS NOT A VERDICT. Exhaustion falls back WITHOUT ruling PACTOR-3 out
(`ptc.PtcHost.fall_back`), because fourteen or sixteen cycles of a fade is not
evidence about what a station can decode -- so the link can climb again, and the
peer's next grant is taken. A gear command resets it: a peer working the ladder
is a peer doing its job, not a peer stuck.

UNDER `--no-p3-fallback` THE BOUND ENDS THE LINK INSTEAD. There is no lower rung
to retreat to once the operator has closed that door, and what the flag used to
do here was nothing: the retreat was refused, the bound was one-shot, and the
repeat train carried on. KB5LZK 40 m, 2026-09-15, spent 22 further cycles and
19 s of carrier on a packet the peer had refused 17 times. So the bound says what
it is doing and QRTs in PACTOR-3 (`_give_up`), which is the same ending every
other budget in this module has. WHAT A REFERENCE ISS DOES WHEN THE IRS HOLDS ITS
CODEWORD IS ON NO TAPE WE HAVE -- neither reference session contains a held
codeword at all -- so this is a local policy, not a reading of one.
"""


RECLAIM_CODEWORDS = 3
"""Codewords decoded from the peer while THIS station also holds the receiving
role, before the IRS takes the link back.

BOTH ENDS RECEIVING WAS A FAULT WITH NO EXIT. `_take_link` was unreachable from
a stranded IRS -- the only ways out were the hold budget and the QRT -- so the
strand was signed off rather than ended. KB5LZK, 2026-08-28: from TX[19],
sixteen consecutive cycles of this station sending nothing but 120 ms
acknowledgements while the far end sent no data at all, `rx CS REQ` standing in
the host log through the whole of it.

WHAT AUTHORISES IT IS WHAT THE PEER SENT, NOT THIS END'S COUNTER, which is the
correction `_on_nak` already carries and this is its mirror. An IRS owes the ISS
one control signal a cycle and sends no packets, so a bare codeword from the
station we are receiving from is that station holding the receiving role --
`on_rx_cs` reads it as exactly that and then had nothing to do with it. A retry
count says nothing about the far end and a changeover is a statement about the
far end.

CS3 IS NOT ONE OF THEM and keeps its own reading: it is the head of the peer's
changeover packet, so the peer is the sending station and there is no strand at
all. It resets this count where it resets the silence budget.

Three, and consecutive. One codeword is a fade artefact or a word out of
somebody else's link that landed in our window; two could be a single answer
repeated; three running, with no data packet between them, is a station that
held the receiving role across the whole span. Deliberately below
`ArqConfig.max_retries`, as `UPGRADE_SILENCE_CYCLES` is: the reclaim is spent
before the give-up, so a strand ends by taking the link rather than by hanging
up.
"""


# --------------------------------------------------------------------------- #
# Connect operators -- what a host's `C` argument asks for
# --------------------------------------------------------------------------- #

CONNECT_PREFIXES = {"%": "robust", "!": "longpath", ";": "longpath"}
"""The SCS connect operators a `C` argument may carry [PTC-IIIusb 4.1 §6.22].

`C %DL6MAA` asks for a Robust Connect and `C !DL6MAA` -- or `;DL6MAA` -- for a
longpath call, and the manual says so of the host interface too: "this is also
valid for WA8DED hostmode". The two are EXCLUSIVE, because a robust call is
"only allowed within the normal PACTOR time frame" and never on the longpath
cycle, so the PTC refuses `C !%CALL` and `C %!CALL` rather than choosing one.
"""

CONTYPE_ACCEPTS = {
    0: (),
    1: ("normal", "longpath"),
    2: ("robust",),
    3: ("normal", "longpath", "robust"),
}
"""Which incoming calls each `CONType` answers [PTC-IIIusb 4.1 §6.22.1].

0 takes no connect at all; 1 only "the usual PACTOR-I connects", which is the
one FSK connect frame under either of its call types; 2 only robust calls; 3,
the default, all of them.
"""


def parse_connect_arg(arg: str) -> tuple[str, str]:
    """Split a `C` argument into its callsign and the call it asks for."""
    call, variant = arg.strip(), "normal"
    while call[:1] in CONNECT_PREFIXES:
        if variant != "normal":
            raise ValueError("a connect takes one operator, % or !")
        variant, call = CONNECT_PREFIXES[call[0]], call[1:]
    if not call:
        raise ValueError("connect needs a callsign")
    return call.upper(), variant


class State(StrEnum):
    """The link states, shared verbatim with the other modems in the flock.

    A StrEnum rather than a bare class of constants: it still compares equal to
    its own string, so `state == "CONNECTED"` and every f-string keep working,
    but a typo is now an AttributeError instead of a silently false comparison,
    and the set is enumerable. These five names are byte-identical across three
    independent FSMs, which is the one part of the ARQ layer that genuinely is
    common -- the state machines around them are not.
    """

    DISCONNECTED = "DISCONNECTED"
    LISTENING = "LISTENING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTING = "DISCONNECTING"


@dataclass
class ArqConfig:
    # Cycle grid (pactor3.md / spec.py). One data packet per cycle.
    cycle_s: float = spec.CYCLE_SHORT_S            # 1.25 s [spec]
    data_cycle_s: float = spec.CYCLE_LONG_S        # 3.75 s [spec]
    max_retries: int = 8                           # [ours — spec gap]
    max_connect_retries: int = 6                   # [ours]
    """Cycles with no sign of the peer before a connect attempt is abandoned.

    THE BUDGET HAS TO OUTLAST THE EVIDENCE RULE. `onair._ConnectEvidence` needs
    three accepts at one turnaround before it will call a codeword a station, and
    a real gateway supplies them sparsely: WS8EOC answered `onair-0730-2036` on
    cycles 3, 5 and 7. A cycle the peer was heard in buys one back
    (`note_peer_heard`), but it is charged first and forgiven at the NEXT tick,
    so reaching cycle 7 costs five. At four this aborted the session one cycle
    before the rule could close it -- measured, not reasoned. Six leaves a cycle
    of margin, and it is what `tools/lib/attempts.sh` passes on the pactor
    attempt.
    """
    # Speed adaptation over the 6 published speed levels, and it is the RECEIVING
    # station that drives it: M.1798 §4 gives CS4 ("increase to the next higher
    # speed level") and CS5 ("repetition and a reduction to the next lower level")
    # and gives the sending station no gear command at all. So this threshold is
    # spent at the IRS, which is the end that has the evidence -- it is the one
    # decoding. [spec 4 gives the levels and the commands; the threshold is ours]
    speed_up_after: int = 3                        # [ours — "several" clean cycles]
    p3_max_try: int = 2  # SCS MAXTry: total emitted attempts at the trial speed, 1..9.
    p3_max_down: int = 6  # SCS MAXDown: consecutive receive errors before CS5, 2..30.
    speed_up: str = "auto"                         # [ours — "auto" or "hold"]
    """Whether this station may ask the peer for a rung at all.

    "hold" keeps the link at the level the entry opened it on, and it exists
    because the count above is blind to how the cycles it spans actually went.
    MEASURED, WS8EOC 80 m 2026-09-13 21:35, off the arm's own cycle ledger:
    `speed_up_after` closed on three accepted packets spread over
    seventeen cycles, twelve of which read nothing, and the CS4 it drew took a
    link delivering the greeting at 38% legibility to 3% -- 1 read cycle of 35,
    against a witness that proves the gateway transmitted in every one of them
    and repeated the same SL2 block 33 times. The consecutive-cycle rule in
    `_gear_cs` is the fix for that arm; this is the arm-level override for a
    path already known to be marginal, and it binds BOTH gear seams, the
    stalled-run exit of `repeat_gear` included -- a CS4 keyed to break a stall
    is still a rung the peer climbs.
    """
    repeat_gear: int = 0                           # [ours — off by default]
    """Identical repeats answered identically before the IRS asks for a gear.

    THE COUNT ABOVE NEVER REACHES A LINK THAT IS NOT MOVING. `speed_up_after`
    spends ACCEPTED traffic packets, and a repeat is not one -- so a peer that
    repeats forever produces no clean run, the link never leaves the level it
    stalled on, and the one control word that asks a stalled link to change
    anything is never keyed.

    MEASURED, WS8EOC 2026-09-13, off the arm's own cycle ledger. The arm that
    stalled read `SL1 status=0x21 seq=1` on thirty-six consecutive
    cycles, answered every one of them CS2 -- the alternation word for an odd
    counter, and the same word the two arms that delivered the whole greeting
    keyed against the same byte -- and the gateway advanced on none of them. It
    is the only arm on file whose transmit log holds no CS4 at all.

    So this counts ANSWERS rather than acceptances: N identical packets answered
    with one identical codeword, and the next repeat gets CS4 instead. "CS4 and
    CS6 substitute in that slot" (pactor3.md 7) -- the request acknowledges the
    counter it replaces, so the alternation is not broken and the counter law is
    untouched. Off by default: it substitutes a codeword on a link that today
    keys the reference's own answer, and no arm has flown it.

    NOT THE LONG CYCLE, which is the other thing the repeated packet asks for.
    Every longest repeat run in all three ledgered arms sits on a bit-5 packet
    (x12, x16, x36), but granting that ask at SL1 is what `_answer_cs` refuses
    above, for the reason 0911 measured: the ladder has no long frame there, so
    our raster goes long while the peer stays on its short comb.
    """
    min_sl: int = 1
    max_sl: int = 6
    entry_sl: int = 3
    """The speed level the PACTOR-3 phase of a link OPENS at.

    Not `min_sl`. An UNINVITED upgrade has to be acquirable from a standing start
    by a peer with no reason to be expecting one, and our own monitor reads levels
    1 and 2 only through a wider level first -- level 1's header block is the
    variable header alone and level 2 carries four of the sixteen constant
    headers.

    A GRANTED upgrade is a different question and has its own answer,
    `ptc.GRANT_ENTRY_SL`: a peer that sent `0x59A` is waiting for the entry
    packet, and the entry packet both references key is level 1 on tones 5 and 12
    -- the pair every level shares, which is what makes it acquirable off a
    PACTOR-1 raster. "Levels 1 and 2 cannot be acquired cold" was a property of
    the monitor and not of the protocol; the real IRS acquired that burst cold and
    answered it.

    MEASURED, twice and independently. A real off-air PACTOR-1 -> PACTOR-3 session
    (`rf-corpus/PIII_Complete_1`, DL6MAA) opens its PACTOR-3 phase at
    speed level 3 and climbs 3 -> 4 -> 5 -> 6 across the session, on the raster its
    PACTOR-1 connect answer set. And an independent monitor acquires level 3 from
    a standing start on every run, while it reads levels 1 and 2 only through a
    wider one -- `tests/shrike/test_p3_oracle.py:NARROW_LEVELS_NOTE`."""
    traffic_sl: int = 3
    """The speed level a GRANTED PACTOR-3 phase carries traffic at, once the entry
    packet has been answered.

    A SEPARATE NUMBER FROM `entry_sl`, because the two answer different questions.
    `entry_sl` is what an UNINVITED upgrade opens at, keyed at a peer with no
    reason to expect one; this is what a peer that has just DEMONSTRATED it read
    the entry packet is handed next. Reading that demonstration as permission for
    a level the peer has said nothing about is a guess, and it is the guess that
    ended two sessions: VE3KPG took the speed-level-1 entry packet twice on
    2026-09-13 and answered the speed-level-3 traffic behind it with the same
    codeword 32 and 17 times, accepting none, then went silent. What the entry
    packet proves is that the peer acquired THAT level.

    So `--p3-traffic-sl` holds the traffic where the peer has just been read, and
    the peer's own CS4 -- the only gear command the protocol gives the receiving
    station -- pulls it up from there. The default is `entry_sl`'s number, so an
    arm that does not name it flies unchanged."""
    entry_ladder: tuple[str, ...] = ("template",)
    """The entry packets a granted upgrade keys, in order.

    Repeated grants advance an optional rung after `UPGRADE_SILENCE_CYCLES`
    requests. The final rung gets `ENTRY_GRANT_CYCLES` requests; silence has its own
    consecutive-cycle limit. Refusals have a separate local placement budget.

    A LADDER BECAUSE THE PEER OFFERS THE CYCLES. Four gateways across three bands
    granted the upgrade and then repeated `0x59A` thirteen to seventeen times
    apiece while this station keyed four packets and fell back -- so the rungs
    below cost nothing that was not already on the table, and the grant is the
    only place the alternatives can be tried at all.

    `template` is the shape pactor3.md §17.1 records: an empty field, so `spec.TEMPLATE`
    fills it, at the data type the reference declares, unswapped. `burst` is the
    same packet behind `placement`'s acquisition preamble, for a receiver that
    turns out to need one. `data` is the shape this station keyed through the
    2026-08-26 slot -- user text at type 0 -- kept nameable so an arm can put the
    control back on the air rather than reason about it.

    ONE RUNG BY DEFAULT, and it is the one that was measured clear at both ends:
    `template` took four of the four cycles available to it. `burst` is grounded
    outright in `ENTRY_RUNGS_GROUNDED` -- it keys one slot in two, and a rung
    offered on half the cadence of the one it is compared against is not
    comparable to it. That was prose in this docstring while the field it
    documents armed the rung by default, so it is a refusal now:
    `check_entry_ladder` will not build a ladder holding it, and the arithmetic
    is in the message.

    `data` is flyable and not offered, which is the other half of the same
    honesty -- no gateway has taken it, so it costs cycles an alternative that
    has never been tried could have.

    `p2sl1` is a PACTOR-2 speed-level-1 short frame in the entry's slot: two
    100 Bd DPSK carriers at 1400 and 1600 Hz, the tones the peer's PACTOR-1
    tracker is already following, unswapped, an empty field at the same fill as
    the PACTOR-3 template. In the August 31 entry experiment, the
    general P2-bridge reading was retired on the independent monitor, but every
    gateway that grants us and never takes the entry is a P4dragon, and both
    completions on tape are against PTC-II class peers -- so the P4dragon-specific
    form is untested, and this is the one-variable candidate against `template`
    as control.

    `p4chirp` is not a PACTOR-3 packet at all: it is PACTOR-4 speed level 1's
    two-tone chirp entry (`p4chirp`), for the possibility seven sessions never
    examined -- that a gateway rostered "Pactor 3,4" grants at ITS maximum, and
    the level §17.1's "the answering station commands the change" commands was
    only ever read off stations whose maximum was PACTOR-3. It renders 3307.5 ms
    into a 1.25 s cycle, so it keys one slot in THREE -- which is not `burst`'s
    grounding arithmetic in a longer coat: PACTOR-4 chirp mode's own cycle is
    3.75 s, exactly three of these slots, and the 442.5 ms left before that
    boundary clears the 210 ms listen floor that grounds `burst`. What it costs
    instead is the peer's two intervening answer slots, transmitted over -- the
    price of asking a 3.75 s question on a 1.25 s raster, and the ask itself."""

    long_cycle: bool = True
    """Whether this station will take the link onto the 3.75 s cycle.

    OFF IS A REFUSAL THIS STATION CAN MAKE AND THE PROTOCOL CANNOT SAY. Neither
    end commands the length alone: the ISS suggests with status bit 5 and the IRS
    commands with CS6, so declining is simply never raising the bit and answering
    a peer's raised one with the ordinary gear codeword -- which is what the
    reference IRS does when it holds a request for a cycle (DL6MAA's bit-5 drop
    at 47.87 s is answered CS1, and the CS6 comes one packet later).

    A CS6 THAT ARRIVES ANYWAY IS STILL FOLLOWED. There is no codeword meaning
    "I cannot", and a station that holds the short raster through a CS6
    desynchronises the session -- so this governs what we ASK for and what we
    GRANT, and never what we do with a command already on the air.
    """

    # Long-path cycle variants exist (1.4 / 4.2 s); off by default.
    longpath: bool = False

    contype: int = 3
    """Which incoming calls this station answers -- SCS `CONType` and its own
    default [PTC-IIIusb 4.1 §6.22.1]. The table is `CONTYPE_ACCEPTS`."""

    conintegrity: int = 0
    """SCS `CONIntegrity`: search for robust calls with less error tolerance,
    which costs "roughly 2 dB" of the SNR the search reaches down to (P4dragon
    firmware 2.40).

    STORED AND NOT YET CONSUMED. The tolerance it tightens belongs to the
    branch-B reader that finds a robust call in the first place, and nothing in
    this package reads one yet, so a station that sets this is setting a
    preference we can honour rather than a behaviour we have.
    """

    def __post_init__(self) -> None:
        check_entry_ladder(self.entry_ladder)
        if not 1 <= self.p3_max_try <= 9:
            raise ValueError("p3_max_try must be 1..9")
        if not 2 <= self.p3_max_down <= 30:
            raise ValueError("p3_max_down must be 2..30")
        if self.contype not in CONTYPE_ACCEPTS:
            raise ValueError(f"contype is 0..3, not {self.contype!r}")
        if self.conintegrity not in (0, 1):
            raise ValueError(f"conintegrity is 0 or 1, not {self.conintegrity!r}")


#: What `ArqIO.send_packet` returns for a burst the seam declined to key.
#:
#: Distinct from `None`, which is a seam that does not measure what it rendered,
#: and from a byte count, which is a seam that keyed. A guard that drops a burst
#: takes no air, so the cycle it would have spent is not a retry: on 2026-09-03
#: the ACK guard and the QRM guard dropped one changeover packet each and both
#: were charged, so the strand ended two cycles before it had spent that much
#: air -- the arm's transmit numbering jumps TX[14] to TX[17] with nothing keyed
#: between them.
REFUSED = -1


class ArqIO:
    """Sink the FSM drives. A real modem implements this over the PHY + host."""

    breakin_rides_a_packet = True
    """Whether taking the link keys a CS3-headed PACKET or the bare codeword.

    True for PACTOR-1 and PACTOR-3, where a changeover packet is a codeword head
    and a shortened field behind it; a seam that says nothing keeps that. False
    where the protocol has no such frame -- `ptc.PtcHost` answers it per link,
    and PACTOR-2 is the case. See `on_cycle`."""

    asking_for_a_grant = False
    """Whether a grant is the only way this link reaches PACTOR-3.

    `ptc.PtcHost` answers it per link, and only for the arm the grant is the
    whole ask of -- not for every PACTOR-1 packet whose status byte carries the
    default announcement. A seam that says nothing keeps today's behaviour."""

    p3_qrt_confirm = False
    """Opt in to the captured P3 terminal-marker experiment after a QRT ACK."""

    p3_tx_gear_hold = False
    """Trial: repeated P3 gear commands retain the unacknowledged packet."""

    def connect_burst(self, mycall: str, dxcall: str) -> None: ...   # P1 FSK connect
    # ...and it answers with the payload bytes it actually RENDERED, `REFUSED`
    # for a burst it declined to key, or None for a seam that does not measure.
    # `PactorArq._sent` reads it: a renderer handed a field it cannot carry
    # truncates in silence -- `placement.link_packet` cut a 276-byte long-cycle
    # field to the short path's 59 and the ARQ settled all 276 -- and the FSM is
    # the only layer that can tell the two lengths apart.
    def send_packet(self, sl: int, payload: bytes, status: int,
                    breakin: bool = False) -> Optional[int]: ...  # P3 data pkt
    def send_cs(self, cs_index: int) -> Optional[int]: ...  # REFUSED, or accepted
    def send_p3_terminal(self, header_bit: int) -> Optional[int]:
        """Emit the short SL1 terminal marker; REFUSED if not keyed."""
        return REFUSED
    def send_p1_cs(self, index: int) -> None: ...                    # 12-bit PACTOR-1 CS
    def upgrade(self, payload_waiting: bool) -> bool: ...   # leave PACTOR-1; did we?
    def fall_back(self, why: str, *,
                  rule_out: bool = True) -> Optional[bool]:
        """Apply fallback; False retains the protocol and its pending upgrade."""
    def breakin_now(self) -> None: ...                # a changeover packet next
    def connected(self, mycall: str, dxcall: str) -> None: ...
    def disconnected(self) -> None: ...
    def defer_rx_close(self) -> bool:
        """Whether the peer's final ACK is queued until the driver keys it."""
        return False
    def deliver(self, blob: bytes) -> None: ...                      # -> host data port
    def buffer(self, nbytes: int) -> None: ...
    def log(self, msg: str) -> None: ...


@dataclass
class _Packet:
    status: int
    payload: bytes
    sl: int
    retries: int = 0
    repeats: int = 0
    breakin: bool = False
    entry: bool = False
    sent_sl: Optional[int] = None
    sent_at_level: int = 0
    trial_from: Optional[int] = None

    @property
    def seq(self) -> int:
        return self.status & spec.STATUS_SEQ


@dataclass
class _ReceiveOpportunity:
    token: int
    silence_pending: bool = False
    crc: bool = False
    occupied: bool = False
    gear_error_counted: bool = False


class PactorArq:
    """Cycle-locked stop-and-wait ARQ for PACTOR-3 (one packet per cycle)."""

    def __init__(self, io: ArqIO, cfg: Optional[ArqConfig] = None):
        self.io = io
        self.cfg = cfg or ArqConfig()
        self.state = State.DISCONNECTED
        self.role: Optional[str] = None            # ISS (sending) | IRS (receiving)
        # Which end of the call this station sits at, which `role` does not say:
        # ISS and IRS swap at every changeover and this does not swap at all.
        # PACTOR-1's read instant is asymmetric in exactly this and nothing else
        # -- see `tests.shrike.test_ackplace`.
        self.answering = False
        self.mycall = self.dxcall = ""
        self._listen = False

        # Sending side
        self._outbuf = bytearray()                 # host byte stream awaiting TX
        self._next_seq = 0
        self._inflight: Optional[_Packet] = None
        self._p3_tx_gear_command: int | None = None
        self._sl = self.cfg.min_sl
        self._ladder = P3_LADDER
        self._clean_run = 0
        self._p3_rx_errors = 0
        self._p3_rx_sl: int | None = None
        # The ISS follows received CS6. The IRS separates the observed peer
        # cycle from its requested target and actual command emission.
        self._cycle_long = False
        self._cycle_request: bool | None = None
        self._cycle_command_emitted = False
        self._subtick = 0                          # driver ticks into a long cycle
        self._buffer_raw = 0
        self._connect_retries = 0
        self._refused_burst = False                # see REFUSED and `_sent`
        self._peer_heard = False
        self.peer_reads = 0                        # frames decoded from the peer
        self._peer_asked_for_channel = False
        self._peer_wants_us_sending = False
        self._peer_frame_cycle = False             # a peer packet read while ISS
        self._peer_raster_burst = False            # ...or a burst on its own comb
        self._breakin_listen = False               # see `breakin_listen_due`
        self._peer_receiving = 0                   # see RECLAIM_CODEWORDS

        # Receiving side
        self._expected_seq = 0
        self._rx_seen = False                      # anything fed under this counter
        self._rx_ahead = 0                         # host characters past the peer's cut
        self._rx_accepted = 0                      # packets taken, never reset
        self._repeat_answer: tuple[int, ...] | None = None  # see cfg.repeat_gear
        self._repeat_run = 0
        self._repeat_pending: tuple[tuple[int, ...], int] | None = None
        self._last_rx_field: tuple[int, bytes] | None = None
        self._rx_decoder = compress.Decoder()      # wire coding -> host characters
        self._rx_si = compress.Supervisor()        # ...and characters -> host bytes

        # Turnaround intents
        self._over_pending = False                 # hand over once the buffer drains
        self._breakin_pending = False              # take the link at the next cycle
        self._breakin_armed = False                # ...but only behind a good packet
        self._turn_ack_owed = False
        self._turn_ack_pending = False
        self._turn_head_wait = self._turn_ack_cycle = False
        self._stint_tail = False                   # a packet may still be in the air
        self._qrt_pending = False                  # terminate once the buffer drains
        self._rx_close_pending = False             # peer QRT still owes an ACK
        self._rx_close_cycles = self._rx_close_subtick = 0
        self._recovering = False                   # yielded after a silent peer
        # Survives the teardown on purpose -- `_finish_disconnected` clears the
        # intents, and the caller reads this afterwards to say what the ending
        # was. Reset where the next link begins instead.
        self.said_goodbye = self.goodbye_acked = False
        self._goodbye_cycles = self._goodbye_subtick = 0
        self._terminal_confirm_pending = self._terminal_confirm_emitted = False
        self._terminal_confirm_ticks = 0
        self._terminal_confirm_seq = 1
        self._disconnect_ticks: int | None = None
        self._budget_goodbye = False
        self.goodbye_unplaceable = False
        self._silent_cycles = 0
        self._refused_cycles = 0                   # ...and cycles WE could not key
        self._rx_this_cycle = False
        self._burst_at_anchor = False              # ...where an answer is due
        self._receive_opportunity: _ReceiveOpportunity | None = None
        self._receive_opportunity_serial = 0
        self._unrepaired = 0                       # packets that arrived unreadable
        # Cycles since an upgrade the peer has not answered, or None when there
        # is no unconfirmed upgrade outstanding. Cycles rather than seconds
        # because on this link the cycle IS the clock (UPGRADE_SILENCE_CYCLES).
        self._unanswered_upgrade: Optional[int] = None
        # ...and the other way the same window can end: cycles the peer spent
        # asking for the entry packet again. See `note_upgrade_unread`.
        self._upgrade_asked = False
        self._upgrade_requests = 0
        self._upgrade_refusals = 0
        # Consecutive cycles whose only answer was an unassigned word, and a
        # flag the present cycle sets. `note_upgrade_unread` credits the retry
        # budget only while the run is inside `ENTRY_GRANT_CYCLES`.
        self._grant_run = 0
        self._granted_cycle = False
        # ...and what the channel carried in those cycles. See
        # `note_burst`.
        self._upgrade_bursts: list[float] = []
        # ...and the cycles that carried something no reader here could shape.
        # See `note_unreadable_answer`.
        self._unreadable_answers = 0
        self._entry_rung = 0
        # An entry packet has been keyed and nothing ordinary has been
        # acknowledged since. It is what arms `UNREAD_RUNG_REPEATS`, and it is
        # armed by the KEYING rather than by the upgrade: an uninvited upgrade
        # keys no entry packet, so it has no entry to exempt and the bound
        # would spend itself on a link nobody asked for the waveform on.
        self._entry_keyed = False

    # -- helpers ----------------------------------------------------------
    def _to(self, s: str) -> None:
        if s != self.state:
            self.io.log(f"state {self.state} -> {s}")
            self.state = s

    @property
    def ladder(self) -> Ladder:
        """The speed levels of the protocol this link is in (`LADDERS`)."""
        return self._ladder

    @ladder.setter
    def ladder(self, ladder: Ladder) -> None:
        """Written by the seam that changes the link's protocol, as `cycle_long`
        is. The level comes down with it where the new ladder is shorter, because
        a rung that does not exist chunks to nothing the renderer can build."""
        self._ladder = ladder
        self._sl = min(self._sl, self._top)

    @property
    def _top(self) -> int:
        """The highest rung this link may climb to: the ladder's, or the
        config's where that is lower."""
        return min(self.cfg.max_sl, self._ladder.top)

    @property
    def entry_level(self) -> int:
        """The level a phase of this link's own protocol opens at.

        `ArqConfig.entry_sl` is PACTOR-3's, and so is its reason; a ladder that
        does not share it names its own (`P2_LADDER`).
        """
        opens = self._ladder.entry_sl
        return min(self.cfg.entry_sl if opens is None else opens, self._top)

    @property
    def traffic_level(self) -> int:
        """The level a granted phase runs traffic at once the entry packet has
        been answered: `ArqConfig.traffic_sl`, where the reason is."""
        return min(self.cfg.traffic_sl, self._top)

    @property
    def speed_level(self) -> int:
        """Current speed level, 1..`ladder.top`."""
        return self._sl

    @speed_level.setter
    def speed_level(self, sl: int) -> None:
        """Set by the seam that knows the link changed protocol under us.

        A PACTOR-1 link has no speed level, so the one this holds while the link
        is in PACTOR-1 is stale, and the level a PACTOR-3 phase opens at is not
        `min_sl` -- see `ArqConfig.entry_sl` and `entry_level`.
        """
        self._sl = min(self._top, max(self.cfg.min_sl, sl))
        self._clean_run = 0
        self._p3_rx_errors, self._p3_rx_sl = 0, None
        if self._inflight is not None:
            self._inflight.trial_from = None

    @property
    def cycle_long(self) -> bool:
        """Whether the link is on the 3.75 s cycle (spec.CYCLE_LONG_S).

        The ISS follows received CS6; the IRS follows the peer's observed
        frame header. A requested or emitted CS6 is separate transition state,
        not proof that the peer has changed. The renderer reads this to pick
        geometry (`placement.LONG_PATHS`) and the driver may read it for the
        raster; the seam that changes the link's protocol writes it, because a
        PACTOR-1 link has no long cycle for it to be true of.
        """
        return self._cycle_long

    @cycle_long.setter
    def cycle_long(self, long: bool) -> None:
        self._cycle_long = bool(long)
        self._subtick = 0
        self._cycle_request = None
        self._cycle_command_emitted = False

    @property
    def cycle_request(self) -> bool | None:
        """The IRS's requested target, pending observation of that peer cycle."""
        return self._cycle_request

    @property
    def cycle_command_emitted(self) -> bool:
        """Whether the outstanding CS6 actually keyed since the last peer frame.

        The receiver can use this to cover the possible new-length packet while
        keeping the observed cycle distinct. It is not permission to transmit
        into that packet: the normal placement and occupancy guards still apply.
        """
        return self._cycle_command_emitted

    def observe_peer_cycle(self, long: bool) -> None:
        """Reconcile a CRC-valid ordinary IRS packet's physical cycle header.

        An old-cycle repeat ends the wait for the previous command, restoring
        that cycle's reply opportunity. Answer selection acknowledges a repeated
        packet normally instead of renewing CS6; a new packet may negotiate
        again. A matching header confirms the transition.
        """
        if self.role != IRS or self.state not in (State.CONNECTED, State.DISCONNECTING):
            return
        self._cycle_long = bool(long)
        self._subtick = 0
        self._cycle_command_emitted = False
        if self._cycle_request == self._cycle_long:
            self._cycle_request = None

    @property
    def rx_progress(self) -> int:
        """How many packets this station has ACCEPTED, and nothing else -- a
        duplicate leaves it where it was, and so do a failed CRC and a sequence
        gap, both of which are answered with a NAK rather than an
        acknowledgement. It counts deliveries, so a stint's worth of it is what
        a byte stream can be graded against.

        NOT THE ALTERNATION, which every level reads off `rx_seq` instead. This
        one is free-running and the numbering an acknowledgement answers is not:
        a changeover restarts the counter and leaves this where it was, so a
        codeword keyed off it lands on whichever face the link happens to have
        reached. PACTOR-1's did, and on 2026-09-02 it came up CS2 -- the repeat
        request -- and VE1YZ sent the same seven bytes 45 times.
        """
        return self._rx_accepted

    @property
    def rx_seq(self) -> int:
        """The mod-4 counter of the last packet ACCEPTED from the peer.

        EVERY LEVEL'S ALTERNATION IS A FUNCTION OF THIS: the codeword answering
        a packet is CS1 for an even counter and CS2 for an odd one
        (`ptc.PtcHost._counter_cs_for`, and `_p1_cs_for` through it), so what it
        has to follow is the number in the peer's own status byte -- which
        restarts at every changeover, along with the sequence it numbers.

        Before anything has been accepted it reads one BELOW the counter this end
        expects, which is the useful answer rather than a placeholder: the
        codeword it produces is the one that does not acknowledge the packet the
        link is waiting for, and an IRS still owes a codeword in the cycles
        before it has read anything. The two seeds differ because the two
        numberings do -- a link counts from one and a changeover from zero -- so
        a called station holds CS1, the connect answer, and a station that has
        just yielded holds CS2, which is the request for the changeover packet.
        """
        return (self._expected_seq - 1) % SEQ_MOD

    @property
    def tx_seq(self) -> Optional[int]:
        """The mod-4 counter of the packet awaiting acknowledgement, or None.

        The other half of the same rule, read from the sending end: the ISS knows
        which counter it put on the air, so it knows which of CS1 and CS2 answers
        it, and the other one is the peer still answering the packet before.
        """
        if self._terminal_confirm_pending:
            return self._terminal_confirm_seq
        return None if self._inflight is None else self._inflight.seq

    @property
    def terminal_confirm_pending(self) -> bool:
        """The opted-in P3 QRT marker still needs emission or its answer."""
        return self._terminal_confirm_pending

    @property
    def terminal_confirm_emitted(self) -> bool:
        """At least one terminal marker reached the transmit seam successfully."""
        return self._terminal_confirm_emitted

    @property
    def upgrade_unanswered(self) -> bool:
        """The link changed protocol and nothing has been heard in the new one.

        The window UPGRADE_SILENCE_CYCLES is counted over, and it is public
        because `ptc.PtcHost` has to tell a PACTOR-1 frame that is a verdict on
        the upgrade from one that is older than the upgrade itself.
        """
        return self._unanswered_upgrade is not None

    payload_bytes_override: int | None = None
    """Field size for a link that is not PACTOR-3. A PACTOR-1 packet carries 8
    bytes at 100 Bd and 20 at 200; the PACTOR-3 speed-level table says 5 at SL1,
    so an un-upgraded link chopped its own callsign announcement to `1W9SS`."""

    breakin_bytes_override: int | None = None
    """...and what is left of that field once the CS3 head has taken its share: 7
    bytes at 100 Bd, 18 at 200. It is a second number and not a derivation because
    the head is 2 bytes at one rate and 3 at the other. Chunking a break-in packet
    at the full field size silently drops the overflow -- the renderer truncates."""

    entry_pending: bool = False
    """The upgrade was granted and the entry packet has not been answered yet.

    Set and cleared by `ptc.PtcHost`, which owns the grant; read here and by the
    renderer, because each has half of what an entry packet is. THE FIELD IS THE
    HALF THAT LIVES HERE: the entry packet carries `spec.TEMPLATE` and no user
    data, so `_start_next_packet` holds the buffer back while this stands.

    MEASURED, and it is the whole of what a working entry packet has ever been
    seen to be. DL6MAA's -- the only one in recorded history that a peer read --
    is `0f 8f 87 c7 c3 1a 66 89`: the template, a status byte at data type 6, the
    CRC. This station keyed `39 53 53 4a 20 02 74 f8` instead, five bytes of user
    text at type 0, and four gateways across three bands answered every one of
    them with the `0x59A` they had granted with. That word says the peer is
    asking again; §7 says it is what a receiver repeats both when nothing arrived
    AND when it will not take what did, so the air cannot tell the two apart.
    What the air can say is that no independent receiver has ever accepted a
    type-0 entry packet cold, and that a peer which has just granted knows every
    one of the 81 symbols it is waiting for -- against which a wrong field is not
    a degraded entry packet but a different one."""

    @property
    def entry_variant(self) -> str:
        """Which rung of `cfg.entry_ladder` this grant is standing on."""
        return self.cfg.entry_ladder[min(self._entry_rung,
                                         len(self.cfg.entry_ladder) - 1)]

    def _payload_bytes(self, breakin: bool = False) -> int:
        if breakin and self.breakin_bytes_override:
            return self.breakin_bytes_override
        if self.payload_bytes_override:
            return self.payload_bytes_override
        # Field sizes follow the negotiated cycle and the active protocol.
        return self._ladder.payload(self._sl, self._cycle_long)

    # ==================================================================== #
    # Host-driven events
    # ==================================================================== #
    def on_host_listen(self, on: bool) -> None:
        self._listen = on
        if on and self.state == State.DISCONNECTED:
            self._to(State.LISTENING)
        elif not on and self.state == State.LISTENING:
            self._to(State.DISCONNECTED)

    connect_variant: str = "normal"
    """Which call `io.connect_burst` keys: a value of `CONNECT_PREFIXES`.

    Read by the keying seam as it renders, the way `entry_variant` is, so the
    host's connect operator reaches the waveform without every layer between
    carrying it -- and so the calls a retry resends are the call the host asked
    for rather than the default."""

    last_connect: Optional[tuple[str, str]] = None
    """Callsign and operator of the last connect THIS station initiated.

    A bare `C` re-dials it as it was dialled: the PTC re-inserts the `%` when
    the last link this end initiated was a Robust Connect, and the longpath `!`
    persists the same way [PTC-IIIusb 4.1 §6.22.2]. Kept apart from `dxcall`,
    which a station calling US overwrites."""

    def on_host_connect(self, mycall: str, dxcall: str) -> None:
        """Call a station, `dxcall` carrying any SCS connect operator.

        An empty `dxcall` is the PTC's bare `C`: re-dial the last remote, with
        the operator it was called with.
        """
        if dxcall.strip():
            dxcall, variant = parse_connect_arg(dxcall)
        elif self.last_connect is not None:
            dxcall, variant = self.last_connect
        else:
            raise ValueError("no station has been called yet")
        self.mycall, self.dxcall = mycall.upper(), dxcall
        self.connect_variant = variant
        self.last_connect = (dxcall, variant)
        self.role, self.answering = ISS, False
        self._to(State.CONNECTING)
        self._connect_retries = 0
        self._peer_heard = False
        self.io.connect_burst(self.mycall, self.dxcall)     # P1 FSK connect [spec]
        self.io.log(f"connect burst -> {self.dxcall}"
                    + ("" if variant == "normal" else f" ({variant} call)"))

    def on_host_data(self, blob: bytes) -> None:
        """Queue host data. Accepted in either role: an IRS holds it until it
        becomes the ISS, which is how a host can write ahead of a changeover.

        NOT ONCE THE GOODBYE IS OWED. The QRT rides the packet that drains the
        buffer, so a host still writing keeps postponing it -- and on the paths
        that end a link this end has given up on, the far end is not going to
        deliver those bytes anyway. A write during teardown used to hold the
        goodbye off indefinitely, `said_goodbye` never became true, and the
        teardown had nothing to bound it.

        AND IT SAYS SO NOW. Both refusals lose host bytes for good -- the B2F
        layer answers a proposal block with one line and has no resend -- and
        the only symptom either produces is a stage that stops advancing.
        `UNREPAIRED_BUDGET` reaches here from inside the receive that hands
        those very bytes up, so the drop can be one call away from the delivery
        that caused it. The count and the reason, never the bytes: what a host
        wrote is its own business, and `;PR:` carries the login.

        WHAT GOES IN THE BUFFER IS A CHARACTER STREAM, not the host's bytes.
        Every level below this one drops 0x1E and reads 0x1C as the opening of a
        supervisor block -- PACTOR-1 by the 1990 description and MEASURED
        against SCS's own monitor at every field position, PACTOR-3 because
        `placement.field_info` pads a part-filled field with IDLE and every
        receive path takes it back off (`spec.field_payload`). So `transparent`
        runs at the top of the stack and `_rx_si` undoes it at the top of the
        other one, and the level the link is at never comes into it. A B2F body
        is lzhuf output, near enough uniform that each value turns up about once
        in every 256 bytes of it.

        The escape is what `_buffer_raw` counts, because it is what has to be
        transmitted before the goodbye can ride out: `_on_ack` subtracts the
        field it sent, which is these bytes and not the host's.
        """
        if self.state != State.CONNECTED or self._qrt_pending:
            self.io.log(f"{len(blob)} bytes from the host discarded -- "
                        + ("the goodbye is owed" if self._qrt_pending
                           else f"the link is {self.state}"))
            return
        wire = compress.transparent(blob)
        self._outbuf += wire
        self._buffer_raw += len(wire)
        self.io.buffer(self._buffer_raw)

    def on_host_over(self) -> None:
        """Hand the link to the peer once the transmit buffer is sent and
        confirmed. Latent while we are the IRS. [PTC hostmode %Q, and %O as ISS]"""
        if self.state == State.CONNECTED:
            self._over_pending = True

    def on_host_breakin(self) -> None:
        """Take the link now, by force. [PTC hostmode %I, and %O as IRS]"""
        if self.state == State.CONNECTED and self.role == IRS:
            self._breakin_pending = True

    def on_host_changeover(self) -> None:
        """Reverse the link: break in if receiving, hand over if sending. [%O]"""
        (self.on_host_breakin if self.role == IRS else self.on_host_over)()

    def on_host_disconnect(self) -> None:
        if (self.state == State.CONNECTED and self.entry_pending
                and getattr(self.io, "no_p3_fallback", False)):
            # The entry waveform has no QRT field. Closing an unconfirmed
            # attempt must not invent a confirmation or send a P1 goodbye.
            self.io.log("closing unconfirmed P3 attempt without a P1 goodbye "
                        "(--no-p3-fallback)")
            self.on_host_abort()
            return
        if self.state == State.CONNECTING:
            self.on_host_abort()
            return
        if self.state != State.CONNECTED:
            return
        if self._disconnect_ticks is None and not self.said_goodbye:
            self._disconnect_ticks = 0
        self._qrt_pending = True
        if self.role == IRS:
            self._breakin_pending = True     # QRT rides a packet, so we must send
        elif self._inflight is not None and not self._inflight.payload:
            # The goodbye rides a packet too, and an idle one in flight would
            # hold it for as long as the peer goes on asking for the repeat --
            # which, against a peer that answers every cycle, is now forever
            # (`_on_nak`). An empty field has nothing to lose, so it goes back and
            # the QRT is built over it, under the counter the peer is expecting.
            self.requeue_inflight()

    def on_rx_ack_emitted(self) -> None:
        """The driver emitted the ACK owed to a received QRT."""
        if self._rx_close_pending:
            self._finish_disconnected()

    def on_cs_emitted(self, cs_index: int) -> None:
        """A control actually reached the transmit seam, rather than its queue."""
        self._refused_cycles = 0            # the seam keyed: it is not stuck
        if cs_index == CS_NAK and self._p3_rx_errors >= self.cfg.p3_max_down:
            self.io.log(f"P3 MAXDown={self.cfg.p3_max_down}: "
                        f"{self._p3_rx_errors} receive errors at SL{self._p3_rx_sl} "
                        "-> CS5 sent; request one level down")
            self._p3_rx_errors = 0
        pending, self._repeat_pending = self._repeat_pending, None
        if cs_index in (CS_BREAKIN, CS_SPEED_UP, CS_NAK, CS_CYCLE_TOG):
            self._repeat_answer, self._repeat_run = None, 0
            if (pending is not None and pending[1] == CS_SPEED_UP
                    and cs_index == CS_SPEED_UP and pending[0][2] == CS_ACK):
                self.io.log(f"{self.cfg.repeat_gear} identical packets answered "
                            f"identically -> ask the peer for SL{pending[0][0] + 1} instead")
        elif (pending is not None and pending[1] == CS_ACK
              and cs_index in (CS_ACK, CS_REQUEST)):
            # CS2 may be the physical ACK for an odd counter. The pending
            # packet records its logical intent before the PTC parity mapping.
            key = pending[0]
            self._repeat_run = self._repeat_run + 1 if key == self._repeat_answer else 1
            self._repeat_answer = key
        if self._turn_ack_pending:
            self._turn_ack_pending = self._turn_ack_owed = False
        if cs_index == CS_CYCLE_TOG and self.role == IRS \
                and self._cycle_request is not None:
            self._cycle_command_emitted = True
        # The radio reports physical CS1/CS2, whose index1 is also the ARQ's
        # logical REQUEST. The existing terminal-pending gate admits only the
        # final ACK; do not reinterpret that physical index as a logical NAK.
        self.on_rx_ack_emitted()

    def on_host_abort(self) -> None:
        self._finish_disconnected()

    # ==================================================================== #
    # RX events — a decoded control signal or a decoded data packet
    # ==================================================================== #
    def on_rx_connect(self, mycall: str, dxcall: str,
                      variant: str = "normal") -> None:
        """Peer's connect burst decoded (we are being called).

        `variant` is the kind of call it was, as `CONNECT_PREFIXES` names them;
        a robust call reaches here once the branch-B reader labels one.
        """
        if self.state not in (State.LISTENING, State.CONNECTING):
            return
        if variant.lower() not in CONTYPE_ACCEPTS[self.cfg.contype]:
            self.io.log(f"CONType {self.cfg.contype} does not answer a "
                        f"{variant.lower()} call -- ignored")
            return
        if self.state == State.CONNECTING and mycall.upper() != self.dxcall:
            # A CONNECT BURST NAMES THE CALLED PARTY AND NOBODY ELSE
            # (docs/protocols/pactor/pactor-connect-frames.md sec 3 and sec 6:
            # one address section, and it is the callsign being called), so off
            # the air `mycall` is empty and the station keying it cannot be
            # identified at all. That is enough to answer while we are LISTENING,
            # which is what a called station does with anything naming it. It is
            # not enough to abandon a call already on the air for: taking it made
            # this station drop its own outgoing call, become the IRS to a station
            # it could not name, and report the link with an empty dxcall.
            #
            # A call is answered with CS1 or CS4, or with the peer's first data
            # packet. A connect burst is neither, and refusing it leaves our own
            # call standing.
            self.io.log("a connect burst is not an answer to our call "
                        "-- still calling")
            self.note_peer_heard()
            return
        self.role, self.answering = IRS, True
        # from our view the peer's dxcall is us, its mycall is the far end
        self.mycall, self.dxcall = dxcall.upper(), mycall.upper()
        self._enter_connected()
        self.io.send_cs(CS_ACK)                             # confirm the link

    def _peer_answered(self) -> None:
        """A decoded frame from the peer: whatever we last changed, it followed.

        The bar is a DECODE, which is why this sits in the two RX handlers rather
        than beside `note_peer_heard`. An upgrade is confirmed by the far end
        being read, once; after that a fade is a fade and not a refusal.

        It records PRESENCE as well, because a decode is the strongest presence
        there is and the two arrive in the same instant. That matters most for
        the frames this station reads and then discards on its role: a stranded
        ISS is handed the other ISS's packets and drops them below, and that
        discard is the deafness `_on_nak`'s yield exists to break.
        """
        self._unanswered_upgrade = None
        self._subtick = 0                    # a long cycle restarts at an answer
        # AND IT COUNTS, because one decision needs the evidence to have a floor
        # rather than a last instant: `ptc._p1_cs_for` may only command a peer's
        # rate down on a link that has read that peer at all, and a link on which
        # nothing has ever decoded has no peer to command (see `_command_100`).
        # Per link -- `_enter_connected` clears it -- so what a previous contact
        # read can never stand in for this one.
        self.peer_reads += 1
        self.note_peer_heard()

    def on_rx_cs(self, cs_index: int) -> None:
        """A 20-bit control signal decoded from the IRS."""
        self.io.log(f"rx CS {_CS_NAMES.get(cs_index, cs_index)}")
        if self._rx_close_pending:
            return
        self._peer_answered()
        if self._terminal_confirm_pending:
            # PtcHost maps the marker's selected physical response to ACK.
            # Repeating the preceding QRT's answer cannot settle this distinct
            # marker, and neither can a gear command.
            if cs_index == CS_ACK and self._terminal_confirm_emitted:
                self.goodbye_acked = True
                self.io.log("P3 terminal marker acknowledged -> link down")
                self._finish_disconnected()
            return
        if self.state == State.CONNECTING and self.role == ISS and cs_index == CS_ACK:
            self._enter_connected()   # confirm via CS -- secondary; the robust path
            return                    # is a CRC-valid frame (see on_rx_packet)
        if self.state == State.CONNECTING and self.role == ISS and \
                cs_index == CS_BREAKIN:
            # A BREAK-IN is also an answer: a Winlink RMS hears the call, takes
            # the channel, and sends its greeting. Only ACK was accepted here, so
            # such an answer was decoded and then discarded, and the link never
            # came up.
            #
            # This rests on the published "forces a break-in" and on nothing else.
            # It used to cite the off-air captures as well -- every link-setup
            # reply shrike had decoded read as a break-in -- and that reading is
            # withdrawn: those are PACTOR-1 bursts, and PACTOR-1's codewords are
            # closed under bit reversal, so they were the speed-change signal read
            # backwards. See ptc._logical_cs, which is where a PACTOR-1 link's
            # codewords are translated; nothing on this path is PACTOR-1.
            self._enter_connected()
            self._yield_link()        # it asked for the channel; give it
            return
        if self.state not in (State.CONNECTED, State.DISCONNECTING):
            return
        if self.role != ISS:
            if cs_index == CS_BREAKIN:
                # ALREADY THE IRS. CS3 is never bare, so the only thing that can
                # put one on the air is the peer's own changeover packet, and the
                # peer is the station already holding the link -- this is the head
                # of a packet whose 840 ms we did not read, not a request for a
                # channel we have already given up. The answer to it is the repeat
                # request `on_cycle` sends below, and the answer it must NOT get is
                # the link-dead budget: `_silent_cycles` is there for a peer that
                # has stopped transmitting, and a codeword at zero bit errors from
                # a station keying a packet every cycle is the refutation of that.
                #
                # MEASURED, WS8EOC 2026-08-09, three sessions. Each yielded
                # correctly, read one packet, and then spent its whole budget on
                # cycles in which the peer's changeover packet was on the air with
                # its head decoding at zero errors -- "no decodable traffic from
                # the peer -> abort", against a station the operator could hear.
                # This is `_on_nak`'s rule read from the receiving side: the budget
                # counts SILENCE, and this is not silence.
                #
                # Only CS3, and it also clears the strand count below: a packet
                # head is the peer holding the SENDING role, which is the one
                # reading under which nothing is stranded at all.
                self._silent_cycles = 0
                self._peer_receiving = 0
                return
            # ANY OTHER CODEWORD IS THE PEER BEHAVING AS AN IRS TOO -- the strand
            # both ends of a lost turnaround fall into, and until now the budget
            # was the only thing that ended it, which ends it by hanging up on a
            # station that is answering every cycle. `RECLAIM_CODEWORDS` carries
            # the reading and the count; what reaches here is one bare codeword,
            # decoded at zero errors in the slot the grid says an answer is due
            # in, from a station we already hold the receiving role against.
            self._peer_receiving += 1
            if self._peer_receiving == RECLAIM_CODEWORDS:
                self.io.log(f"{RECLAIM_CODEWORDS} codewords and no packet -- "
                            f"the peer is receiving too -> take the link back")
            return
        if cs_index == CS_BREAKIN:
            self._yield_link(acked=True)                    # "forces a break-in"
            return
        # EVERY OTHER CODEWORD ANSWERS THIS CYCLE'S PACKET, and each answers HERE,
        # in the cycle the control signal arrived in -- which is the cycle the peer
        # is timing us against. The tick below would otherwise send a SECOND packet
        # on top of it, into the 0.29 s the peer answers in, and charge a second
        # retry for it; two 0.96 s packets do not fit in a 1.25 s cycle at all.
        #
        # Unreachable until the control-signal decoder started working: no answer
        # decoded means no path through here, so every cycle of every session so
        # far was driven by the tick alone.
        #
        # It is also where the stint we took closes: a codeword from the peer is
        # the peer receiving, so nothing more of its own is in the air behind it.
        self._rx_this_cycle = True
        self._stint_tail = False
        self.note_peer_answering()
        gear_hold = (getattr(self.io, "p3_tx_gear_hold", False)
                     and getattr(self.io, "protocol", None) == spec.Protocol.PACTOR3)
        if (gear_hold and self._inflight is not None
                and cs_index in (CS_SPEED_UP, CS_NAK)
                and cs_index == self._p3_tx_gear_command):
            # A held command may be the answer to the packet BEFORE this one.
            # The K0NTS 0920 trial keeps bytes/counter pending until an actual
            # ACK or a different command disambiguates the exchange. The
            # speed-up trial's MAXTry bound still applies.
            # This is opt-in: the public description omits lost-command state.
            self.io.log(f"P3 TX GEAR HOLD: repeated CS{cs_index + 1}; "
                        f"repeat seq={self._inflight.seq} from SL{self._sl} "
                        f"{len(self._inflight.payload)}B unacknowledged")
            self._on_nak(speed_down=False)
            return
        if cs_index == CS_NAK:
            self._on_nak(speed_down=True)                   # CS5 = NACK + drop one
            if gear_hold:
                self._p3_tx_gear_command = cs_index
        elif cs_index == CS_REQUEST:
            self._on_nak(speed_down=False)                  # repeat; MAXTry bounds a trial
        else:
            # CS4 AND CS6 ACKNOWLEDGE, and that is derived rather than quoted. The
            # IRS emits exactly ONE control signal per cycle, and M.1798 §4 spends
            # all six meanings: two acknowledge/request, one forces a break-in, one
            # demands a higher speed level, one is a NACK asking for a repetition
            # and a lower level, one toggles the cycle length. None of the six is
            # "acknowledged, and separately change gear", so a gear command that
            # did not also acknowledge would leave the link unable to both advance
            # and change gear -- it would climb a level per cycle while repeating
            # one packet forever. Which is what this did: CS4 raised `_sl` and fell
            # through, `_inflight` stayed set, and the tick retransmitted.
            trial_from = None
            if cs_index == CS_SPEED_UP:
                previous_sl = self._sl
                self._sl = min(self._top, self._sl + 1)
                if (getattr(self.io, "protocol", None) == spec.Protocol.PACTOR3
                        and self._sl > previous_sl):
                    trial_from = previous_sl
                self.io.log(f"peer asked for the next level -> SL{self._sl}")
            elif cs_index == CS_CYCLE_TOG:
                # CS6 acknowledges AND toggles: the reference IRS's first CS6
                # (17.41 s) answers the packet that raised bit 5, and the ISS's
                # very next packet is a long one on the same raster -- new
                # counter, not a repeat. So the ack below settles the inflight
                # and the next chunk is cut to the new field size by
                # `_payload_bytes`. `LONG_TICKS` carries the raster arithmetic.
                self._cycle_long = not self._cycle_long
                self._subtick = 0
                self.io.log("peer toggled the cycle length -> "
                            f"{'LONG 3.75 s' if self._cycle_long else 'short 1.25 s'}")
            self._on_ack()
            if (trial_from is not None and self._inflight is not None
                    and self._inflight.payload and not self._inflight.breakin
                    and not self._inflight.entry and self.role == ISS
                    and self._inflight.sl > trial_from):
                self._inflight.trial_from = trial_from
                self.io.log(f"P3 MAXTry={self.cfg.p3_max_try}: trial SL{self._sl} "
                            f"from SL{trial_from}, seq={self._inflight.seq}")
            if gear_hold and cs_index == CS_SPEED_UP:
                self._p3_tx_gear_command = cs_index

    @property
    def terminal_pending(self) -> bool:
        """Terminal intent, including the interval before QRT can be keyed.

        A pending goodbye can still have CONNECTED state while its packet is
        waiting for a slot. Role reversal and ordinary receive deadlines do
        not end a link and are deliberately absent from this predicate.
        """
        return (self._qrt_pending or self._rx_close_pending
                or self._terminal_confirm_pending
                or self._disconnect_ticks is not None
                or self.state == State.DISCONNECTING)

    def on_rx_grant(self) -> None:
        """The peer commanded a waveform (`ptc.PtcHost._take_grant`).

        IT ACKNOWLEDGES THE ANNOUNCEMENT IT ARRIVES BEHIND. The grant lands
        where the codeword answering our packet would, and pactor3.md §17.1
        reads that slot literally -- "it acknowledges that packet, and the
        granted station's very next burst is PACTOR-3"; pactor1-control-
        signals.md §3 says the same. So the packet in flight is settled here,
        exactly as an acknowledgement settles it, however many times the peer
        asked for it first.

        AND NOTHING CROSSES INTO THE NEW WAVEFORM WITH IT. Neither reference
        caller carries its PACTOR-1 announcement into PACTOR-3. W4DNA announced
        `1w4dna\\r` in PACTOR-1 and put five fresh bytes, `Q-6.0`, in its first
        PACTOR-3 field (7101k_234600, 21.185 s), filling it exactly; DL6MAA keyed
        six template-filled packets after its entry -- acknowledged, with the
        gateway taking the turn on the first of them -- and only then loaded a
        field (PIII_Complete_1, 9.113 to 16.613 s). Requeued, ours put `1w9ss`,
        `1w9ssj\\r` cut mid-callsign at the speed-level-1 field, in the first
        PACTOR-3 field and left `j\\r` to lead the login a turn later
        (`peer-cedes-the-turn-with-a-bit6-packet` recorded stale `1w9` bytes
        leading one). What follows the entry now is idle packets until the host
        writes -- which is the state the reference peer breaks in on.

        THE RESIDUE IS THE ONE THE GRANT ITSELF LEAVES: if the peer never read
        the packet it granted behind, those bytes are gone. In every arm on
        record that packet is the callsign announcement `ptc.PtcHost.
        _answer_link_setup` queues, which the PACTOR-3 phase has no use for, and
        the grant is the only answer PACTOR-1 gives it.

        THE COUNTER DOES NOT GO BACK EITHER. What follows a grant is the entry
        packet, and its counter is the one the PACTOR-1 phase had reached -- 2 in
        both reference sessions, behind that phase's one data packet (pactor3.md,
        "a changeover restarts the counter").
        """
        if self._rx_close_pending:
            return
        self.io.log("rx GRANT")
        self._peer_answered()
        self.note_peer_answering()
        # UNREACHABLE, and the guards that make it so are the caller's:
        # `ptc.PtcHost._take_grant` clears `_ruled_out` immediately above this
        # and gates on PACTOR-1, ISS and CONNECTED, so `upgrade`'s target loop
        # always finds one. Returning here would consume the peer's answer
        # without marking the cycle answered, and the tick would then charge a
        # retry against a station that transmitted at zero bit errors.
        if not self._offer_upgrade(payload_waiting=bool(self._outbuf)):
            return
        self._settle_inflight()
        self._rx_this_cycle = True
        self._stint_tail = False
        self._start_next_packet()

    def on_rx_packet(self, sl: int, payload: bytes, status: int, crc_ok: bool,
                     breakin: bool = False,
                     protocol: str | None = None,
                     cycle_long: bool | None = None,
                     repeated_stint: bool = False) -> None:
        """A decoded data packet at the IRS -- or a changeover packet at the ISS.

        `protocol` is what the DECODER read (`rxfront.Event.protocol`), and it is
        a separate argument from `sl` because the two answer different questions:
        the status byte's layout follows the protocol, the gear commands follow
        the level, and a seam may honestly report a PACTOR-1 level for a packet
        that is not PACTOR-1 (`onair._SessionRx._p2_packet`). Unset -- the
        simulated peers, which render no waveform -- the level answers both.

        `cycle_long` is the decoded current-cycle header, when available. It
        must not be filled from the status bit that requests the next cycle.

        `repeated_stint` requires driver evidence: an emitted, unacknowledged
        local P3 CS3 followed by the identical peer CS3 on its previous raster.
        Payload equality alone cannot distinguish this from a genuine reversal.
        """
        if self._terminal_confirm_pending:
            # No data or role-request packet can substitute for this marker's
            # control answer or restart its bounded termination clock.
            return
        # Once the peer has closed, only its repeated QRT can replace the
        # final ACK. Other traffic cannot reopen the link or erase that answer.
        if self._rx_close_pending and not (crc_ok and status & spec.STATUS_QRT):
            return
        # A new packet replaces the previous deferred answer. Only an ACK
        # selected for this valid packet can release the recovery barrier.
        self._turn_ack_pending = False
        self._turn_head_wait = False
        self._repeat_pending = None
        # A CHANGEOVER PACKET WHILE WE ARE CALLING IS NOT PRESENCE TO SPEND. It is
        # refused as an answer below, for the reason spelled out there -- CS3 is
        # never bare, so a packet headed by one is a link changing hands between
        # two stations that have one -- and `note_peer_heard` still counted it, so
        # a third party's QSO bought back a connect retry and held our call on the
        # frequency longer. The budget exists for undecoded energy that MIGHT be the
        # station we called; a frame this well identified is evidence of the
        # opposite, and on a shared band that is a reason to stop calling sooner.
        if not (breakin and self.state == State.CONNECTING):
            self._peer_answered()
        # A break-in reaches us as the peer's OWN packet, so yielding and reading
        # it are one event and not two. "Empfaengt der TX ein CS3, schaltet er
        # sofort in den RX-Modus und liest das Restpaket vollstaendig ein": there
        # is no cycle in between, and an implementation that yields here and waits
        # for the next one has thrown the packet away.
        if breakin and crc_ok and self.role == ISS and \
                self.state in (State.CONNECTED, State.DISCONNECTING):
            if repeated_stint and protocol == spec.Protocol.PACTOR3 \
                    and self.unconfirmed_breakin:
                self._resume_peer_stint()
            else:
                self._yield_link(acked=True)                # "forces a break-in"
        elif crc_ok and self.entry_pending and self.role == ISS and \
                self.state in (State.CONNECTED, State.DISCONNECTING):
            # AND THE ENTRY PACKET'S ANSWER IS THE PEER TRANSMITTING, whatever
            # part of it we caught. The reference answers a good entry packet
            # with a CS3-headed packet and then a full data packet in the other
            # direction; the head is twenty symbols at the front of an 0.81 s
            # burst and the data packet behind it is the whole cycle, so the
            # branch above is the half of that answer a receiver is least likely
            # to hold. Without this the other half fell through to the discard
            # below -- a station that had just proved it could read our waveform,
            # graded on our own role and thrown away, with the entry packet keyed
            # again over its greeting until the retry budget ended the link.
            #
            # Only while `entry_pending` stands, and that is what keeps it narrow.
            # A data packet arriving at an ISS is otherwise the stranded-ISS
            # signature, which `_on_nak` spends against a budget rather than
            # acting on -- there the peer may be a station that never agreed to
            # anything. Here we asked one question in the previous slot, and an
            # IRS that answers it by sending a packet has both acquired the
            # waveform and taken the channel.
            self._yield_link(acked=True)
        # A CRC-valid frame from the peer while we are still CONNECTING is
        # unforgeable proof the link came up and the peer has taken the ISS role --
        # e.g. a Winlink RMS answering a connect by sending its greeting first.
        # This is the robust confirm: off-air measurement shows a real answer is
        # CRC'd frames on the header tones, not the bare CS_ACK that on_rx_cs waits
        # for, and the CS decoder is not discriminative at real SNR (a 20-bit
        # codeword lands within a few bits of noise). We become the IRS and ack.
        # NOT A CHANGEOVER PACKET, THOUGH. CS3 is never bare and a packet headed
        # by one is the first packet of a link changing hands -- a link that
        # already exists, between two stations neither of which can be answering
        # a call we have not had an answer to. It is the same reading `ptc`
        # gives a bare CS3 arriving in this phase: an exchange we have walked in
        # on. A peer answering OUR call sends an ordinary first packet.
        if self.state == State.CONNECTING and self.role == ISS and crc_ok \
                and not breakin:
            self.io.log("peer answered with a CRC-valid frame -> link up (IRS)")
            self.role = IRS
            self._enter_connected()
        if self.state not in (State.CONNECTED, State.DISCONNECTING) or self.role != IRS:
            # THE STRANDED ISS, AND THE ONLY THING THAT EVIDENCES IT. A data
            # packet is what an ISS transmits and an IRS never does, so a
            # CRC-valid one arriving while this end also holds the sending role
            # is the peer saying it has the channel -- the lost turnaround
            # `_on_nak`'s yield exists to break, and the discard on the line
            # above is the deafness itself. Recorded rather than acted on: the
            # yield belongs to the budget, not to one frame.
            #
            # AND NOT A PACKET WITH BIT 6 STANDING, WHICH SAYS THE OPPOSITE. An
            # ISS that has run out of things to say holds a changeover request up
            # on idle packets until our changeover packet arrives, so a peer
            # repeating one while we hold the sending role is a peer that has NOT
            # read the break-in we already sent it. Graded as the strand, it
            # yields the channel back to the station asking us to take it, and
            # the peer asks again. MEASURED, WS8EOC 3596500 kHz 2026-08-29: the
            # gateway took the link, sent one empty packet with bit 6 up, and
            # this end broke in, yielded on the strand, broke in again and
            # yielded again -- sixteen empty 0.96 s packets keyed over a station
            # that was waiting for one.
            #
            # AND THE TAIL OF THE STINT WE JUST TOOK IS NOT THE STRAND AT ALL,
            # it is the last packet of the direction that has only this instant
            # closed. `_take_link` flips the role a whole cycle before the peer
            # can read the CS3 that announces it, so a peer with one more packet
            # already cut keys it into a station that has stopped being its IRS
            # -- and then reads the changeover packet, which arrives in that
            # packet's acknowledgement slot and IS its acknowledgement
            # (`_yield_link(acked=True)` is our own side of the same rule). The
            # field was settled at the far end and dropped here: acknowledged
            # bytes that never reached the host, once per turnaround, on a link
            # that turns around every few lines of a Winlink exchange.
            #
            # The counter is what tells the two apart, and it tells them apart
            # exactly. `_take_link` leaves the receive stream where the ended
            # stint left it, so the tail packet carries the number this end is
            # still waiting for; a repeat of it does not, and neither does a
            # station that has renumbered under a changeover of its own.
            if crc_ok and self.role == ISS \
                    and self.state in (State.CONNECTED, State.DISCONNECTING):
                self._peer_frame_cycle = True
                tail = (self._stint_tail
                        and (status & spec.STATUS_SEQ) == self._expected_seq)
                if status & spec.STATUS_CHANGEOVER:
                    self._peer_wants_us_sending = True
                else:
                    self._peer_asked_for_channel = True
                if tail:
                    self._accept_field(sl, payload, status, protocol)
                elif (protocol != spec.Protocol.PACTOR3
                        and status & spec.STATUS_CHANGEOVER and self.unconfirmed_breakin
                        and not status & spec.STATUS_QRT):
                    # P3's standing request is not an acknowledgement of our
                    # CS3 payload. WS8EOC 0919-2325 repeated the same 0x40
                    # ordinary packet on its old raster; accepting it here
                    # discarded the unacknowledged ;FW and started counter 1
                    # over the peer's transmission. Keep P3's counter-zero
                    # changeover pending until its actual control answer.
                    # The following legacy interpretation remains for P1/P2.
                    self._stint_tail = False
                    self._rx_this_cycle = True
                    self._peer_wants_us_sending = False
                    self.io.log("the peer is asking for the channel behind our "
                                "changeover -> take the turn and key the data "
                                "it is waiting for")
                    self._on_ack()
            return
        # Past this point every path answers -- ACK, or NAK for a bad CRC or a
        # sequence gap, or the changeover packet on the tick when a break-in is
        # pending. Recording it here rather than beside each send is what keeps the
        # cycle tick from sending a SECOND control signal on top of it; the IRS
        # emits exactly one.
        self._rx_this_cycle = True
        p3 = ((protocol if protocol is not None else getattr(self.io, "protocol", None))
              == spec.Protocol.PACTOR3)
        if not crc_ok:
            self._clean_run = 0
            opportunity = self._receive_opportunity
            if not p3 or opportunity is None or not opportunity.gear_error_counted:
                self._unrepaired += 1
            # P3 must reach MAXDown before a two-error legacy guard ends it.
            # Its existing link timeout remains independent and configurable.
            limit = self.cfg.max_retries if p3 else UNREPAIRED_BUDGET
            if self._unrepaired > limit:
                self.io.log(f"{self._unrepaired} packets arrived and would not "
                            f"decode -> saying goodbye")
                self.on_host_disconnect()
                return
            answer = CS_NAK
            if p3:
                answer = self._p3_error_reply(sl if not breakin else 1)
            emitted = self.io.send_cs(answer)
            if (p3 and answer == CS_NAK and emitted != REFUSED
                    and not getattr(self.io, "defer_rx_close", lambda: False)()):
                self.on_cs_emitted(answer)
            return
        self._p3_rx_errors = 0
        self._p3_rx_sl = sl if p3 else None
        if self._receive_opportunity is not None:
            self._receive_opportunity.crc = True
        if cycle_long is not None and not breakin and sl != P1_SPEED_LEVEL:
            self.observe_peer_cycle(cycle_long)
        self._recovering = False                 # the peer is talking; link is sane
        self._unrepaired = 0                     # a packet read is the repair asked for
        self._peer_receiving = 0                 # ...and a packet is the peer SENDING
        seq = status & spec.STATUS_SEQ
        # THE ONLY COUNTER THAT IS NOT NEW IS THE ONE THIS END JUST TOOK.
        # Stop-and-wait cannot skip a number: the ISS holds its packet until
        # something answers it, so the two values a working link can put on the
        # air are the one we are waiting for and the one before it. A third value
        # is not a gap -- there is no packet in between for it to be a gap around
        # -- it is a peer that has renumbered, and every changeover renumbers.
        #
        # MEASURED, DL6MAA. Its speed-level-1 entry packet carries counter 2 and
        # the train behind it runs 1, 1, 2, 3, 0: the IRS broke in between the
        # two, so the entry packet is the last of the caller's PACTOR-1 numbering
        # and the train is the first of the numbering the changeover started.
        # Graded as a gap, the greeting that follows the entry packet -- 212
        # bytes, and the only field in `ref_occ15_pactor3.wav` carrying
        # characters -- was answered "send that again" and never delivered.
        new_packet = not self._rx_seen or seq != (self._expected_seq - 1) % SEQ_MOD
        if new_packet:
            self._accept_field(sl, payload, status, protocol)
        else:
            self._note_recut(payload, self._data_type(sl, status, protocol))
        # NOT FROM A CHANGEOVER PACKET'S OWN STATUS BYTE. That packet is the peer
        # taking the channel; a request to hand it straight back cannot also be
        # what it carries, and acting on one keys a break-in into the burst we
        # just yielded to. Only a host request can arm one in the cycle after a
        # yield.
        if status & spec.STATUS_CHANGEOVER and not status & spec.STATUS_QRT \
                and not breakin:
            # An invitation, not a transfer. The link changes hands only through
            # the CS3-headed packet (module header, MEASURED), and that packet is
            # ours to send: the flags below hand it to the cycle tick, which
            # resets the counter and takes the link -- the same machinery a host
            # break-in uses. Flipping the role here, with no break-in on the air,
            # left the peer transmitting to a station that had silently stopped
            # acknowledging. Read for a duplicate too, and idempotently: the peer
            # holds its request up on every packet until we act on it.
            #
            # AND IT IS ANSWERED IN THE CYCLE IT ARRIVED IN. This used to pend
            # only, so the break-in waited for the NEXT packet that decoded --
            # and a peer that has asked for the channel back may have nothing
            # more to send. `PIII_Complete_1` is that case exactly: the station
            # that took the link after the entry packet keys one 212-byte
            # greeting with bit 6 standing, is answered by the caller's own CS3
            # in the very next turnaround, and transmits nothing in between. Held
            # for a second packet, this end answers CS2 into a channel the peer
            # is waiting to be handed and the link dies on the silence budget.
            self._breakin_pending = True
        # AND NOT WHEN THE PACKET SAYS QRT. A changeover is something to do with
        # a link and bit 7 is the end of one, so the goodbye is answered here
        # even with a break-in standing -- from the host, or from bit 6 on the
        # peer's previous packet. Ranked the other way, the peer's sign-off got
        # no acknowledgement, this end took a link the far end had left, and with
        # no QRT of its own pending it held the channel with idle packets until
        # the retry budget expired.
        if self._breakin_pending and not self._turn_ack_owed \
                and not status & spec.STATUS_QRT:
            # The changeover packet goes out instead, on the tick -- and THIS
            # packet decoding is what licenses it. "The IRS sends it after a
            # correctly received packet": the CS3 replaces this cycle's
            # acknowledgement, so the peer reads it AS one and settles the
            # packet it was holding. Break in behind a failed packet and the
            # peer marks undelivered bytes delivered -- payload loss.
            self._breakin_armed = True
            return
        # A repeat proves reception, but not forward progress: its ACK may
        # never have reached the peer. The shortened changeover field is also
        # not evidence for the traffic ladder. WS8EOC's repeated three-byte
        # RMS changeover otherwise requested SL2 every few lost ACKs.
        answer = CS_ACK if self._turn_ack_owed else self._answer_cs(
            sl, status, breakin=breakin, new_packet=new_packet,
            traffic=bool(payload) and new_packet and not breakin)
        answer = self._repeat_gear_cs(sl, status, answer)
        if (breakin and protocol == spec.Protocol.PACTOR3
                and not status & spec.STATUS_QRT and not self._turn_ack_owed
                and getattr(self.io, "p3_changeover_cs5", False)):
            # Experiment for CHANGEOVER_ANSWER_CS5: progressing reference
            # changeovers receive CS5, but its startup-rate/ACK meaning is open.
            # Keep the normal timing and ordinary-packet answer policy.
            answer = CS_NAK
            self.io.log("experimental P3 changeover reply -> CS5")
        self._turn_ack_pending = self._turn_ack_owed
        self._turn_ack_cycle = self._turn_ack_owed
        emitted = self.io.send_cs(answer)
        if emitted == REFUSED:
            self._turn_ack_pending = False
            self._repeat_pending = None
        if emitted != REFUSED and not getattr(self.io, "defer_rx_close", lambda: False)():
            self.on_cs_emitted(answer)
        if status & spec.STATUS_QRT:
            if getattr(self.io, "defer_rx_close", lambda: False)():
                # Keep the accepted protocol, IRS role and clock alive until
                # the queued final ACK actually keys. Closing here resets all
                # three before the driver's emit_pending_cs can use them.
                self._rx_close_pending = True
                self._to(State.DISCONNECTING)
            else:
                self._finish_disconnected()

    def _accept_field(self, sl: int, payload: bytes, status: int,
                      protocol: str | None) -> None:
        """Take a packet into the receive stream and put its field on the host
        port -- the one place a field is delivered, from either role.

        The status byte's own data-type field says how the payload is coded, and
        the host gets characters, not wire coding: a real peer sends Huffman/PMC
        whenever it shortens the field, and an idle fill -- what an ISS with an
        empty buffer transmits -- decodes to nothing and is not delivered. Our
        own transmit is ASCII_8BIT, with run/supervisor control bytes quoted. Only counters
        this end has not fed reach here, so a memory-ARQ repeat cannot
        double-feed the decoder -- or the supervisor behind it, which is stateful
        across fields on purpose: a block can put its 0x1C at the end of one
        packet and its function code at the start of the next, MEASURED. It spans
        a changeover as readily, so it is reset where the decoder is and nowhere
        else: one link direction is one character stream, however many stints it
        takes.

        WHAT THE PEER HAS ALREADY BEEN GIVEN IS HELD BACK HERE. `_note_recut`
        is where that distance comes from and why it exists.
        """
        data_type = self._data_type(sl, status, protocol)
        if self._is_replay(payload, data_type):
            self.io.log(f"the peer opened its stint with the {len(payload)} "
                        f"bytes it closed the last one with -> acknowledged, "
                        f"not delivered")
        else:
            chars = self._rx_decoder.feed(payload, data_type)
            if self._rx_ahead:
                held = min(self._rx_ahead, len(chars))
                self._rx_ahead -= held
                chars = chars[held:]
                self.io.log(f"{held} of this field's characters reached the host "
                            f"under the peer's previous cut -> not delivered again")
            data = self._rx_si.feed(chars)
            if data:
                self.io.deliver(data)
            self._last_rx_field = (data_type, bytes(payload))
        self._expected_seq = ((status & spec.STATUS_SEQ) + 1) % SEQ_MOD
        self._rx_seen = True
        self._rx_accepted += 1

    def _data_type(self, sl: int, status: int, protocol: str | None) -> int:
        """How the status byte says this packet's field is coded.

        THE FIELD IS TWO BITS WIDE IN PACTOR-1 AND THREE FROM PACTOR-2 ON, so
        this serves two layouts and the PACKET'S OWN PROTOCOL is what separates
        them. Not `sl`: that is the gear seam, which reports PACTOR-1 for a
        PACTOR-2 frame on purpose, and read as the layout it sent every PMC
        field HB9AK transmitted to the Huffman tree -- `in the ` reaching the
        host as `I  ESdEREV0 G D G D`. The 1990 description gives the Datenmodus
        as bits 2-3 and calls bit 4 unassigned; SCS's own PACTOR-2 figure widens
        it to bits 2-4 for the PMC modes, which PACTOR-1 has no compressor for.
        MEASURED, and both PACTOR-1 stations the corpus can check agree with the
        split: DL6MAA's 200 Bd announcement, status 0x35, is plain Huffman under
        two bits and decompresses to `1dl6maa`, the same nine characters an
        independent decoder reads off that audio, while three bits call it PMC
        German swapped and produce `1DIT)   WS DUNZ0 AN`; W4DNA's 0x31 is ASCII
        `1w4dna` under two and PMC German under three. Bit 4 there belongs to the
        capability declaration bits 4-5 carry -- the thing a modern gateway
        answers with a grant -- and reading it as data mode put a decompressor's
        garbage on the host data port.
        """
        p1 = (sl == P1_SPEED_LEVEL if protocol is None
              else spec.Protocol(protocol) is spec.Protocol.PACTOR1)
        return (status >> 2) & (0b11 if p1 else 0b111)

    def _note_recut(self, payload: bytes, data_type: int) -> None:
        """A repeat that re-cuts the field this end has already delivered.

        A SPEED CHANGE REBUILDS THE PACKET IN FLIGHT AT THE NEW FIELD SIZE and
        the counter comes back with it -- `rechunk_inflight` is this end's half
        of that rule, and the peer's half is what the host reads. So the repeat
        of a counter can be a PREFIX of the field taken under it, and everything
        the peer sends afterwards is measured from the new, shorter boundary:
        the counters behind it carry characters the host has already had, under
        numbers the duplicate check is right to call new.

        MEASURED, WS8EOC 2026-09-13 17:14 and 17:48. A 23-byte speed-level-2
        greeting was answered CS1, repeated four times as the five-byte
        speed-level-1 packet `0\r\nW9` -- the acknowledgement never reached the
        gateway -- and the three counters behind it re-sent the remaining
        eighteen bytes. The host read `W9SSJ has 82 daily mSSJ has 82 daily
        minutes remaining`, and the repeat itself was correctly suppressed on
        both arms: the counter law caught the copy and could not see the re-cut.

        DELIVERED CHARACTERS CANNOT BE TAKEN BACK, so what this end is ahead by
        it holds against the fields that follow (`_rx_ahead`, spent in
        `_accept_field`). Idempotent -- a re-cut packet is repeated like any
        other, and every copy measures the same distance from the same accepted
        field -- and silent under a coded mode, where a prefix of the wire says
        nothing about a count of characters.

        A repeat that EXTENDS the accepted field is the speed-UP re-cut, and
        what it carries past the old boundary is characters the host has never
        had. Suppressed whole they are gone with no line on the log, and the
        layer above is a checksummed protocol: eighteen bytes out of a SID line,
        a `;PQ:` challenge or a compressed body is the session. Only the tail
        goes to the stream -- the prefix has been delivered once already -- and
        the counter is spent either way, so this changes nothing about what is
        acknowledged.
        """
        if not payload or self._last_rx_field is None \
                or compress.bit_stream(data_type):
            return
        last_type, last = self._last_rx_field
        if data_type != last_type:
            return
        if len(payload) < len(last) and last.startswith(payload):
            self._rx_ahead = len(last) - len(payload)
            self.io.log(f"the peer re-cut counter {self.rx_seq} from "
                        f"{len(last)} bytes to {len(payload)} -> the host is "
                        f"{self._rx_ahead} characters ahead of its stream")
        elif len(payload) > len(last) and payload.startswith(last):
            tail = payload[len(last):]
            data = self._rx_si.feed(self._rx_decoder.feed(tail, data_type))
            if data:
                self.io.deliver(data)
            self._last_rx_field = (data_type, bytes(payload))
            self.io.log(f"the peer re-cut counter {self.rx_seq} from "
                        f"{len(last)} bytes to {len(payload)} -> {len(tail)} "
                        f"characters the host had not had")

    def _answer_cs(self, sl: int, status: int, *, breakin: bool,
                   traffic: bool, new_packet: bool = True) -> int:
        """The IRS's one codeword for a packet that decoded, cycle length first.

        CS6 IS THE IRS GRANTING THE ISS'S OWN SUGGESTION. The sending station has
        no gear command of any kind; what it has is status bit 5, and the IRS
        answers a packet whose bit disagrees with the cycle the link is on by
        toggling -- which also acknowledges (`on_rx_cs` has the derivation).
        Measured on the reference: the first bit-5 packet (16.61 s) is answered
        CS6 in its own answer slot (17.41 s), and the ISS's next packet is long.
        The IRS keeps discretion the other way -- DL6MAA's bit-5 drop at 47.87 s
        is answered CS1 and the CS6 comes one packet later -- so granting on the
        first ask is inside the observed envelope, not beyond it.

        NOT A CHANGEOVER PACKET'S BIT, though. The one changeover packet on tape
        with bit 5 up (5.56 s, status 0x20) is followed by SHORT-cycle traffic,
        so whatever that bit is doing there, it is not this request; a changeover
        is answered on the traffic behind it, not geared.

        `cfg.long_cycle` off declines the ask -- the gear codeword goes out
        instead, which is the reference's own way of holding a request -- but it
        never declines the way BACK. A link already long has to be able to come
        home whatever this station's policy for entering one is.
        """
        # Not onto a rung the ladder gives no long frame, either -- SL1 has none
        # (`placement.link_packet` refuses to construct one), and granting it
        # anyway moves OUR raster to the long period while the peer, unable to
        # render the geometry, stays on its short comb. That is how one grant
        # cost 0911 its receive position: 4:4 confirmed before, 6:96 after.
        # The way back to short is still unconditional -- a link already long
        # has to be able to come home regardless of what the ladder holds now.
        if (sl != P1_SPEED_LEVEL and not breakin
                # VE3KPG 2026-09-20 repeated the same valid short SL2 field
                # 28 times against our CS6s. Its observed header still owns
                # the cycle: ACK a duplicate on that cycle and clear the
                # unconfirmed request below, rather than negotiating forever.
                # Keep P2's separate command-time latch unchanged.
                and (new_packet or self._ladder is P2_LADDER)
                and bool(status & spec.STATUS_LONG_CYCLE) != self._cycle_long
                and (self._cycle_long
                     or (self.cfg.long_cycle
                         and self._ladder.long_frame(sl) is not None))):
            target = bool(status & spec.STATUS_LONG_CYCLE)
            if self._ladder is P2_LADDER:
                # Preserve P2's existing command-time cycle latch. Its live
                # reader does not publish the observed header length, and its
                # driver has no pending-transition receive window. Applying the
                # P3 waiting state here would request CS6 forever.
                self.cycle_long = target
                self.io.log("granting the PACTOR-2 cycle-length change -> "
                            f"{'LONG 3.75 s' if target else 'short 1.25 s'}")
                return CS_CYCLE_TOG
            self._cycle_request = target
            self._cycle_command_emitted = False
            self.io.log("requesting the peer's cycle-length change -> "
                        f"{'LONG 3.75 s' if self._cycle_request else 'short 1.25 s'}; "
                        "waiting for command emission and peer header")
            return CS_CYCLE_TOG
        if not breakin:
            self._cycle_request = None
            self._cycle_command_emitted = False
        return self._gear_cs(sl, traffic=traffic)

    def _gear_cs(self, sl: int, *, traffic: bool = True) -> int:
        """The IRS's one control signal for a packet that decoded: CS1, or CS4.

        THE GEAR IS THE RECEIVING STATION'S TO CHOOSE. M.1798 §4 gives the sending
        station no speed command at all -- CS4 demands the next higher level and
        CS5 demands the next lower one, and both travel from the IRS. So the run of
        clean cycles is counted HERE, at the end that is doing the decoding and so
        is the only one with evidence about the channel. The ISS used to climb on
        its own after `speed_up_after` acknowledgements, which transmits at a level
        the peer never asked for and leaves shrike's own IRS unable to ask at all.

        Only for a packet on a ladder. `sl` is `P1_SPEED_LEVEL` on the PACTOR-1
        seam, where the same index means CS4/Speedchange -- which a station at
        100 Bd reads as a repeat request and a station at 200 Bd reads as a
        REJECT. Asking a PACTOR-1 peer to speed up asks it to slow down, or to
        send that again.

        WHICH LADDER'S TOP IT STOPS AT is `_top`, and it is the link's own: a
        PACTOR-2 peer asked for SL4 would be asked for a rung this station
        cannot read back (`P2_LADDER`), and PACTOR-3's six do not exist there
        at all.

        AND THE RUN IS CONSECUTIVE CYCLES, NOT A TALLY OF ACCEPTS. A count that
        only ever goes up measures how many packets got through and not what
        fraction of the peer's transmissions did, so it closes on a link that is
        barely legible just as surely as on a clean one -- it only takes longer.
        WS8EOC 80 m 2026-09-13 21:35 closed it on three accepts spread over
        seventeen cycles, twelve of them unread, and the rung it asked for took
        the link from 38% of the peer's cycles read to 3%. So a cycle that reads
        nothing ends the run (`on_cycle`), and so does a repeat or an idle field
        below: what reaches `speed_up_after` is `speed_up_after` cycles in a row
        that each delivered a new packet, which is the only evidence this end has
        that the channel would carry the level above.
        """
        if sl == P1_SPEED_LEVEL:
            return CS_ACK
        if not traffic:
            # An idle field proves nothing about the channel at the next level
            # and a climb costs the ISS an acquisition it gets nothing for. The
            # reference IRS does exactly this: six empty SL3 fields in a row
            # (9.11-15.37 s) are answered CS1/CS2 only, and its CS4s come once
            # the fields carry content. It is also a cycle the run did not
            # advance in -- a repeat arrives here too -- so the run ends with it.
            self._clean_run = 0
            return CS_ACK
        self._clean_run += 1
        if (self._clean_run >= self.cfg.speed_up_after and sl < self._top
                and self.cfg.speed_up == "auto"):
            self._clean_run = 0
            self.io.log(f"clean run -> ask the peer for SL{sl + 1}")
            return CS_SPEED_UP
        return CS_ACK

    def _repeat_gear_cs(self, sl: int, status: int, answer: int) -> int:
        """The way out of a run of identical packets answered identically.

        A STALL IS NOT A CHANNEL READING, so `_gear_cs` above cannot see it: that
        count spends accepted traffic, and a peer whose counter never moves
        supplies none. This one spends the answer itself. While the packet
        (level, status byte, counter) and our codeword for it are all unchanged,
        the run grows; anything that differs -- the peer advancing, a gear or
        cycle command going out, a changeover -- starts a new one.

        CS4 IS WHAT SUBSTITUTES. It is a control signal in the acknowledgement's
        own slot, so the repeated counter is still acknowledged and the
        alternation resumes from wherever the peer takes the link next; the run
        restarts, so a peer that goes on repeating is asked again N repeats
        later rather than every cycle. Only where a rung above exists to ask
        for, and never on the PACTOR-1 seam, where CS4's index means
        Speedchange -- a 100 Bd station reads it as a repeat request and a
        200 Bd one as a reject (`_gear_cs`).

        NOT OVER A GOODBYE. Bit 7 is the peer closing, and its packet is owed
        the acknowledgement `_finish_disconnected` is waiting to key.

        AND THE PEER'S CHANGEOVER PACKET STALLS THE SAME WAY, so it is counted
        here like any other. WS8EOC 2026-09-13 18:49: the gateway took the link
        and then repeated its three-byte CS3-headed `RMS` packet, counter 0, for
        35 cycles against 46 CS1s and went silent -- while the 18:37 arm, whose
        stall was on an ordinary counter-1 packet, drew CS4 on its fourth repeat
        and read the whole greeting. A changeover packet rides the two-carrier
        comb and reports speed level 1 for it, which is a PACTOR-3 level and not
        the PACTOR-1 seam, so the rung above it exists to ask for.

        OUR OWN BREAK-IN IS NOT ON THIS PATH AT ALL: `on_rx_packet` returns at
        `_breakin_armed` a dozen lines above, and the changeover packet goes out
        instead of any codeword.

        AND IT IS STILL A RUNG, so `cfg.speed_up == "hold"` stops it: the exit
        from a stall is the same CS4 the climb keys and the peer answers it the
        same way, by transmitting at a level this end has just been shown it
        cannot read. Under hold the stalled run keeps its ordinary alternation
        word and the stall ends the way it did before this seam existed -- on
        `_silent_cycles` once the peer stops, or on the caller's hold budget.
        """
        # Several physical repeats may decode while the driver gives up a
        # slot. They replace one deferred reply, not several transmitted ACKs.
        # Preview the answer here; only on_cs_emitted commits its history.
        key = (sl, status, answer, self.rx_seq)
        previous = self._repeat_run if key == self._repeat_answer else 0
        if (self.cfg.repeat_gear and previous >= self.cfg.repeat_gear
                and self.cfg.speed_up == "auto"
                and answer == CS_ACK and sl != P1_SPEED_LEVEL
                and sl < self._top and not status & spec.STATUS_QRT):
            answer = CS_SPEED_UP
        self._repeat_pending = key, answer
        return answer

    def _p3_error_reply(self, sl: int | None = None, *, count: bool = True) -> int:
        """MAXDown counts received errors, not outgoing repeats or elapsed slots.

        Failed CRC candidates in one driver window count once. An occupied
        window with no successful decode is counted after all readers finish.
        A valid packet (including a duplicate) clears the consecutive run.
        """
        if sl is not None and sl != self._p3_rx_sl:
            self._p3_rx_sl, self._p3_rx_errors = sl, 0
        opportunity = self._receive_opportunity
        if count and (opportunity is None or (
                not opportunity.gear_error_counted and not opportunity.crc)):
            if opportunity is not None:
                opportunity.gear_error_counted = True
            if self._p3_rx_sl is not None and self._p3_rx_sl > 1:
                self._p3_rx_errors += 1
        if self._p3_rx_sl is None or self._p3_rx_sl <= 1:
            return CS_REQUEST
        return CS_NAK if self._p3_rx_errors >= self.cfg.p3_max_down else CS_REQUEST

    def begin_receive_opportunity(self) -> int:
        """Open one driver RX window whose post-key evidence is not ready yet.

        Optional: ordinary callers retain on_cycle's immediate accounting.
        The driver must finish this token once, after all readers and weak
        observations for the window. Its outer elapsed-slot hold budget still
        bounds a channel that repeatedly carries undecodable energy.
        """
        if self._receive_opportunity is not None:
            raise RuntimeError("the previous receive opportunity is still open")
        self._receive_opportunity_serial += 1
        token = self._receive_opportunity_serial
        self._receive_opportunity = _ReceiveOpportunity(token)
        self._burst_at_anchor = False  # No prior window's weak evidence leaks in.
        return token

    def finish_receive_opportunity(self, token: int) -> None:
        """Settle one IRS silence decision without emitting another response."""
        opportunity = self._receive_opportunity
        if opportunity is None:
            raise RuntimeError("no receive opportunity is open")
        if type(token) is not int or token != opportunity.token:
            raise ValueError("the receive opportunity token does not match")
        if (self.role == IRS and self.state in (State.CONNECTED, State.DISCONNECTING)
                and not self._qrt_pending and not self._rx_close_pending
                and getattr(self.io, "protocol", None) == spec.Protocol.PACTOR3
                and opportunity.occupied and not opportunity.crc
                and not opportunity.gear_error_counted):
            self._p3_error_reply()
        self._receive_opportunity = None
        self._burst_at_anchor = False
        if (self.role != IRS or self.state not in (State.CONNECTED, State.DISCONNECTING)
                or self._qrt_pending or self._rx_close_pending):
            return
        if opportunity.crc:
            self._silent_cycles = 0
            self._p3_rx_errors = 0
        elif opportunity.silence_pending:
            if opportunity.occupied and not self._recovering:
                self.io.log("a burst where the answer is due "
                            "-> not counting this cycle")
            else:
                self._charge_irs_silence()

    def _charge_irs_silence(self) -> None:
        """Queue the current miss, or charge it immediately for legacy callers."""
        if self._receive_opportunity is not None:
            self._receive_opportunity.silence_pending = True
            return
        self._silent_cycles += 1
        if self._silent_cycles > self.cfg.max_retries:
            self._give_up("no decodable traffic from the peer")

    def note_peer_heard(self) -> None:
        """Somebody is transmitting on this frequency, contents unknown.

        Deliberately the weakest thing in the module: it carries no callsign, no
        codeword and no sequence, so it can never bring a link up or advance one.
        What it does carry is that the channel is not empty, and two decisions
        turn on nothing stronger -- the connect budget, which shrike was measured
        spending while WS8EOC was still calling it, and the yield-and-listen
        recovery in `_on_nak`, whose premise is a station we can hear and cannot
        read. Both are questions about the air rather than about the far end.

        SO "CONTENTS UNKNOWN" IS A CONDITION ON THE CALLER, not a description of
        this line. Weak evidence is admitted here because the unreadable station
        might be the one we are trying to raise; a frame that decoded and was
        then refused has answered that question the other way, and passing it
        here spends the budget of a call on the occupant we should be yielding
        to. `ptc.PtcHost.on_rx_event` refuses three frames on exactly that
        ground -- one in PACTOR-2 or PACTOR-3, a bare CS2, a bare CS3 -- and
        forwards none of them. The refusal below is not one: a connect burst
        that reaches it NAMES THIS STATION, so the caller may well be the
        station we are calling, and only the direction is wrong.
        """
        self._peer_heard = True

    def note_peer_answering(self) -> None:
        """A codeword from the peer IN THE ANSWER SLOT: it is the receiving end.

        The refutation of `_peer_asked_for_channel`, and it is stronger evidence
        than the thing it refutes. An IRS owes the ISS one control signal a
        cycle and sends nothing else; a station transmitting one is holding the
        receiving role, whatever this end's retry counter has reached.

        WS8EOC, 2026-08-26. The gateway sent `0x59A` at zero bit errors on
        thirty-six consecutive cycles -- ten of them read here -- and the link
        reversed anyway, because the only thing the yield asked was whether
        somebody was keying. Both ends then held the receiving role, and the
        rotation into a role the peer had not agreed to put this station's
        codeword 69 ms in front of the gateway's answer instant for five cycles.
        An unassigned word counts: `pactor1.CS_59A` is not an acknowledgement
        and drives nothing else here, but a twelve-bit word read at zero of
        twelve in the slot the IRS owes us is that station answering.
        """
        self._peer_asked_for_channel = self._peer_wants_us_sending = False

    def note_upgrade_unread(self) -> None:
        """The peer asked again for the packet the upgrade was supposed to carry.

        NOT SILENCE, AND POINTING THE OTHER WAY. `UPGRADE_SILENCE_CYCLES` counts
        a peer that said nothing, on the reading that silence is the only refusal
        PACTOR has. This is a peer that answered in the slot it owes us, at zero
        bit errors, with the word that commanded the mode in the first place. So
        it spends a count of its own, and the window ends with that finding
        rather than with a claim nothing was heard -- which is what three on-air
        arms logged over cycles carrying a codeword apiece.

        A repeated codeword requests retransmission (pactor3.md §7). Count it
        as a decoded request: it resets consecutive silence and advances the
        entry-request budget. Packet contents and entry detection remain
        unconfirmed until the peer supplies a higher-mode response.

        Optional entry variants still advance after four requests. The last
        variant uses the normal retry allowance, independently of silence.

        AND THE CYCLE IS ANSWERED WHETHER OR NOT A WINDOW IS OPEN. The count
        above belongs to the upgrade; the reset below belongs to the link. A
        twelve-bit word read at zero of twelve in the slot the far end owes us is
        that station transmitting on our raster, and a cycle carrying one is not
        a cycle the in-flight retry budget may charge -- which it did, because
        `_unanswered_upgrade` is None on a plain PACTOR-1 link and this method
        then did nothing at all. KB5LZK on 30 m, 2026-09-16
        (`pactor-current-kb5lzk-30-pounce-20260916T155343Z`): ten of sixteen
        in-link slots carry `0x59A` at zero bit errors, the session read none of
        them, spent this budget and signed off with a QRT over a gateway that was
        answering two cycles in three.

        ONLY HERE, AND NOT IN `note_peer_answering`, which every assigned
        codeword passes through as well. An acknowledgement already clears this
        count in `_on_ack`; a REPEAT REQUEST must not, because the retry budget
        is the only bound on an ISS repeating a packet the peer keeps asking for
        -- the two sessions of 2026-08-14 in `on_cycle`'s own comment repeated
        packet #1 nine times apiece. The unassigned word asks for nothing, so
        crediting the cycle it arrived in cannot unbind that.

        AND THE CREDIT IS BOUNDED BY THE BUDGET THAT ALREADY BOUNDS A REPEATED
        GRANT. A station answering every cycle with a word that acknowledges
        nothing still resets the retry count every cycle, and an unbounded reset
        removes the only ceiling on an ISS keying the same packet at a peer that
        will never take it -- which is the campaign `ENTRY_GRANT_CYCLES` was
        written to bound in the first place. So the run of consecutive
        grant-credited cycles is spent against that same constant; past it the
        cycle charges as any other, the budget exhausts, and the changeover
        decision in `_on_nak` is reached with the finding intact. KB5LZK's ten
        answered slots of 2026-09-16 sit at a longest run of nine, well inside
        it: this bounds a stalled campaign, not an answering gateway.
        """
        if self._unanswered_upgrade is not None:
            self._upgrade_asked = True
        self._granted_cycle = True
        if self._inflight is not None and self._grant_run < ENTRY_GRANT_CYCLES:
            self._inflight.retries = 0

    def note_burst(self, gap_ms: float, *, at_anchor: bool = False) -> None:
        """A burst this cycle carried, at the turnaround it implies.

        Presence with a POSITION on it, and the position is the whole point. This
        layer cannot see the channel, so when the upgrade window ran out it said
        "nothing decoded" -- which on 2026-08-26 stood over four arms whose cycles
        each carried a zero-error codeword 150 ms further out than the window was
        looking. The line said the peer had gone quiet; the peer had not moved.

        The turnarounds are kept rather than counted, because where they were is
        what tells a reader whether the window was in the wrong place or the far
        end was genuinely somewhere else.

        `at_anchor` IS WHAT THE LINK-DEAD BUDGET SPENDS ITSELF AGAINST, and it is
        the caller's band test -- the bounds `onair._MasterGrid._acquire` accepts
        a turnaround inside, which is where an answer to our own transmission has
        to fall and nowhere else in the cycle. It is a claim about a slot and not
        about a station, and that is enough: the answer instant is a property of
        THIS link, corroborated at the connect and re-measured off every decode
        since, so a station outside the link has no reason to key on it.

        `note_peer_heard` is the same evidence with the position thrown away, and
        the difference is measured. KB5LZK, 2026-08-28: ten cycles of a dead link
        forgiven on presence alone -- two of them with the grid reporting nothing
        heard at all, and the rest putting the burst at -17, -79, -92, -107, -156
        or -216 ms, where an answer has to fall between 40 and 130. Not one was
        in the band, and a negative gap is a burst that began under our own
        carrier, which `_upgrade_window_ended` says nothing about the far end is
        knowable across.
        """
        if self._receive_opportunity is not None:
            self._receive_opportunity.occupied |= at_anchor
            self._burst_at_anchor = self._receive_opportunity.occupied
        else:
            self._burst_at_anchor = at_anchor
        if self._unanswered_upgrade is not None:
            self._upgrade_bursts.append(gap_ms)

    def note_unreadable_answer(self) -> None:
        """The slot an answer is due in was OCCUPIED, by nothing this station reads.

        `note_burst` with the turnaround taken away, because there is no
        turnaround to give: the evidence behind this is a channel standing over
        its own floor across the whole band an answer to our transmission has to
        land in, and nothing here found an edge to time. Both ways of losing one
        are on file -- WS8EOC's bursts on 2026-08-28 are already up when the T/R
        mute lifts, and KB5LZK's on 2026-08-29 arrive at an ordinary 61-76 ms and
        are rejected by a detector shaped like PACTOR-1. Fifteen consecutive
        listen windows apiece, every one logged as `nothing heard`, and the retry
        budget spent against a station transmitting in all of them.

        THE CYCLE IS SPENT OR IT IS NOT, and that is the whole of what this
        decides. It reaches `_burst_at_anchor`, which is the finding `on_cycle`
        forgives an IRS cycle on, and `_peer_heard`, which holds the connect
        budget open -- both questions about whether anyone is transmitting,
        neither of them an identification. It advances no counter, acknowledges
        nothing and authorises no changeover, because a burst nobody read is not
        a codeword.

        AND IT DOES NOT END A LINK. It did, for one release: an ISS holding the
        channel to repeat a packet the far end had stopped acknowledging signed
        off on this alone. What that reached on 2026-09-04 was band noise -- the
        two loudest windows of a distribution spanning 8.6 dB, neither carrying
        anything any decoder on this station or SCS's own could name -- and a
        gateway that had granted PACTOR-3 four times was left calling into a
        link we had torn down. The strand ends where it always did, on the retry
        budget, and this holds a cycle.
        """
        self._burst_at_anchor = True
        if self._receive_opportunity is not None:
            self._receive_opportunity.occupied = True
        self._peer_heard = True
        if self._unanswered_upgrade is not None:
            self._unreadable_answers += 1

    def _upgrade_window_ended(self) -> str:
        """Which state the RECEIVER was in when the upgrade window ran out.

        Three of them, and until the turnaround was reported signed they were
        one sentence -- `the peer never took the entry packet` -- printed over
        all three. They call for opposite things: a dead band, a receive path to
        fix, and a transmitter to get off the peer's raster.

        A NEGATIVE TURNAROUND IS OUR OWN CARRIER. `note_burst` is fed
        `_MasterGrid.nearest_gap_n`, samples from OUR data ending to the burst,
        and a negative one is a burst that began while we were still keyed.
        Nothing about the far end is knowable on such a cycle: the head of its
        codeword is under our emission and is not in any recording of ours.
        WS8EOC 2026-08-26 spent the whole of one window there -- the log read
        `nearest at 1212 ms` for eleven cycles, which is the same -38 ms
        measured around the cycle instead of about it.

        The peer's own answers are not in this branch at all: a cycle that
        carried a codeword spends `_upgrade_requests`, above.
        """
        grants = "grant" if self._upgrade_requests == 1 else "grants"
        prefix = (f"After {self._upgrade_requests} repeated {grants}, "
                  f"{self._unanswered_upgrade} consecutive unanswered attempts: "
                  if self._upgrade_requests else "")
        if not self._upgrade_bursts:
            if self._unreadable_answers:
                # AND IT IS NOT SILENCE, which is the one thing this branch used
                # to be able to say. A slot standing over its own floor with no
                # onset in it is a peer transmitting something this station has
                # no reader for; calling that "nothing at all" is what made every
                # `--p1-status-bits45 2` arm on record unreadable.
                return (prefix + f"THE ANSWER SLOT WAS OCCUPIED on "
                        f"{self._unreadable_answers} of the "
                        f"{self._unanswered_upgrade} cycles in the current silent run and "
                        f"nothing in it was PACTOR-1 -- no codeword, no burst at "
                        f"the tones, and a channel over its own floor exactly "
                        f"where our answer is due. What it was is not on the air")
            return (prefix + f"WE HEARD NOTHING AT ALL in {self._unanswered_upgrade} "
                    f"cycles in the current silent run -- no codeword, and no burst at "
                    f"the peer's tones either")
        lo, hi = min(self._upgrade_bursts), max(self._upgrade_bursts)
        where = f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f} to {hi:.0f}"
        under = [g for g in self._upgrade_bursts if g < 0]
        if under:
            return (prefix + f"WE WERE TRANSMITTING OVER THE ANSWER on {len(under)} of "
                    f"the {len(self._upgrade_bursts)} cycles that carried a "
                    f"burst -- it began up to {-min(under):.0f} ms before our "
                    f"own audio ended, so its head is under our carrier and no "
                    f"receiver recovers that. Turnarounds {where} ms. This is a "
                    f"finding about this station, not about the peer")
        return (prefix + f"WE READ NO CODEWORD in {self._unanswered_upgrade} cycles since "
                f"the upgrade, though the window carried "
                f"{len(self._upgrade_bursts)} burst(s) at a turnaround of "
                f"{where} ms that no reader took")

    # ==================================================================== #
    # Cycle tick — the ISS drives one packet per grid cycle
    # ==================================================================== #
    def on_cycle(self, *, elapsed_ticks: int = 1, cycle_ticks: int = 1) -> None:
        """Age elapsed base slots, then service at most one present opportunity.

        ``elapsed_ticks`` charges teardown time, including skipped slots.
        ``cycle_ticks`` advances only the present opportunity's cycle gate:
        a driver already stepping complete long cycles passes LONG_TICKS.
        Historical slots never replay packet emissions or replace a queued ACK.
        Both counts must be positive integers. Time preceding a newly requested
        disconnect must be accounted before that decision, not charged to it.
        """
        for name, ticks in (("elapsed_ticks", elapsed_ticks),
                            ("cycle_ticks", cycle_ticks)):
            if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._grant_run = self._grant_run + 1 if self._granted_cycle else 0
        self._granted_cycle = False
        if self.state == State.CONNECTING and self.role == ISS:
            # A cycle in which the peer was heard does not count against the
            # budget. Observed on the air: WS8EOC kept calling once a second long
            # after shrike had given up and dropped the link -- the operator could
            # hear it answering a transmitter that had stopped listening. Its
            # bursts are often too poor at this SNR to raise `p1reply`, so the
            # retry counter ran out while the far end was plainly still there.
            #
            # Presence is not an answer and does not connect anything; it only
            # buys another cycle. The total is still bounded by the caller's
            # --max-cycles, so this cannot call forever at a dead frequency.
            if self._peer_heard:
                self._peer_heard = False
                # SOMEBODY, not the peer: `note_peer_heard` says so in its own
                # first line, and what sets it is undecoded energy carrying no
                # callsign, codeword or sequence. Calling that "the peer" in a
                # line the operator reads asserts an identification nothing here
                # made -- and this one drives behaviour, since it spends a retry.
                self.io.log("somebody still transmitting -> not counting this retry")
            else:
                self._connect_retries += 1
            if self._connect_retries > self.cfg.max_connect_retries:
                self.io.log("connect retries exhausted -> abort")
                self.on_host_abort()
                return
            self.io.connect_burst(self.mycall, self.dxcall)  # resend connect
            return
        if self.state not in (State.CONNECTED, State.DISCONNECTING):
            return
        if self._terminal_confirm_pending:
            self._terminal_confirm_ticks += elapsed_ticks
            if self._terminal_confirm_ticks > GOODBYE_CYCLES:
                self.io.log("P3 terminal marker confirmation timed out -> link down")
                self._finish_disconnected()
                return
            # The final marker always has a short SL1 geometry. Emit at most
            # one current opportunity; elapsed historical slots only age it.
            rendered = self.io.send_p3_terminal(self._terminal_confirm_seq)
            if rendered != REFUSED:
                self._terminal_confirm_emitted = True
            return
        if self._disconnect_ticks is not None and not self.said_goodbye:
            self._disconnect_ticks += elapsed_ticks
            if self._disconnect_ticks >= GOODBYE_PLACE_TICKS:
                self.goodbye_unplaceable = True
                self.io.log("goodbye could not be placed -> link down")
                self._finish_disconnected()
                return
        period = LONG_TICKS if self._cycle_long else 1
        # Teardown clocks must advance even when no current transmission is
        # eligible. Their residual base ticks are independent of RX phase:
        # a fresh frame cannot keep restarting an unanswered-goodbye timeout.
        if self._rx_close_pending:
            cycles, self._rx_close_subtick = divmod(
                self._rx_close_subtick + elapsed_ticks, period)
            self._rx_close_cycles += cycles
            if self._rx_close_cycles > GOODBYE_CYCLES:
                self.io.log("the received goodbye ACK could not be emitted -> link down")
                self._finish_disconnected()
            return
        if self.said_goodbye:
            # IN CYCLES, not in retries, because the retry budget is not the only
            # clock a teardown runs on. A decoded break-in yields the link and
            # `_give_link` re-arms the QRT we still owe, which clears the packet
            # in flight with it -- so against a peer that breaks in every cycle,
            # which WS8EOC did for a whole session, the two ends trade the
            # channel and the goodbye is keyed forever. The count starts where
            # the QRT reached the AIR: `on_host_disconnect` may be waiting on a
            # buffer that still has bytes in it, and a bound that started there
            # would cut the transfer the goodbye is queued behind.
            cycles, self._goodbye_subtick = divmod(
                self._goodbye_subtick + elapsed_ticks, period)
            self._goodbye_cycles += cycles
            if self._goodbye_cycles > GOODBYE_CYCLES:
                self.io.log(f"the goodbye went unanswered for "
                            f"{GOODBYE_CYCLES} cycles -> link down")
                self._finish_disconnected()
                return
        if self._cycle_long:
            # A caller on the short raster advances one tick; a caller already
            # at the next long opportunity advances LONG_TICKS. Collapse any
            # crossed boundaries into this one action, retaining phase.
            cycles, self._subtick = divmod(self._subtick + cycle_ticks, LONG_TICKS)
            if not cycles:
                return
        if self._turn_head_wait:
            self._turn_head_wait = self._rx_this_cycle = False
            return  # A refused earlier retry does not license keying over this head.
        if self.role == IRS:
            if self._turn_ack_cycle:
                self._turn_ack_cycle = self._rx_this_cycle = False
                self._breakin_armed = False
                self._silent_cycles = 0
                return  # Includes synchronous ACK emission before this tick.
            armed, self._breakin_armed = self._breakin_armed, False
            if not self._turn_ack_owed and (
                    (self._breakin_pending and (armed or self._qrt_pending))
                    or self._reclaiming):
                # The link is taken by SENDING THE FIRST PACKET OF IT, whose head
                # is CS3. Nothing else goes out this cycle, and nothing waits for
                # the next one: the packet occupies the control-signal slot, which
                # is what the 840 ms grid rotation is for.
                #
                # ONLY BEHIND A PACKET THAT DECODED, because the CS3 IS that
                # packet's acknowledgement: the peer settles its inflight on
                # hearing one (`_yield_link(acked=True)`), which is only true if
                # we read it. `armed` is set where the ack was withheld for
                # exactly this burst, and consumed here whether or not it fires.
                #
                # The QRT goodbye is exempt, because it must not depend on the
                # peer still being readable: a hold budget expires against a
                # faded or silent station too, and the alternative to a blind
                # CS3 there is vanishing without the sign-off. The settle it
                # forces at the peer costs nothing the teardown does not
                # already cost -- an inflight the link ends under is lost with
                # or without the courtesy.
                #
                # AND SO IS THE RECLAIM, on a stronger version of the same
                # argument: the finding that authorises it (`RECLAIM_CODEWORDS`)
                # is that the peer is sending no packets at all, so there is no
                # inflight at the far end for the blind CS3 to settle wrongly.
                # Waiting for a packet to ride behind would be waiting for the
                # very thing whose absence is the fault.
                #
                # UNLESS THE PROTOCOL HAS NO SUCH PACKET, which PACTOR-2 does
                # not: a changeover packet is a codeword head and a SHORT field
                # behind it -- `placement.CHANGEOVER` is 56 rows where an
                # ordinary PACTOR-3 packet is 72 -- and PACTOR-2's frame marker
                # can name a level and a length and nothing else, so there is no
                # 51-row frame to build and no way to tell a receiver it is
                # looking at one. There the bare CS3 IS the break-in, which is
                # what "CS3 forces a break-in" says on its own, and our first
                # packet goes out on the next boundary.
                if getattr(self.io, "breakin_rides_a_packet", True):
                    # A refused changeover never tells the peer to yield. Keep
                    # receiving on the old grid until a packet actually keys;
                    # otherwise the driver rotates its receive window by 840 ms
                    # while the peer continues sending on the original raster.
                    seq, inflight = self._next_seq, self._inflight
                    state, goodbye = self.state, self.said_goodbye
                    self._next_seq = 0
                    rendered = self._start_next_packet(breakin=True)
                    if rendered == REFUSED:
                        self._outbuf[:0] = self._inflight.payload
                        self._inflight, self._next_seq = inflight, seq
                        self.state, self.said_goodbye = state, goodbye
                        self._refused_burst = False
                        if self._qrt_pending and self._budget_goodbye:
                            # The retry budget already ended this P3 link. Give
                            # its QRT one placement opportunity, not another
                            # train of CS2 requests after a refused goodbye.
                            # Operator-requested closes retain their normal
                            # acquisition/fallback and elapsed-time deadline.
                            self.goodbye_unplaceable = True
                            self.io.log("budget-ended goodbye could not be placed "
                                        "-> link down without a repeat request")
                            self._finish_disconnected()
                            return
                        # THE CONTROL THE CHANGEOVER DISPLACED STILL GOES OUT.
                        # A cycle that keys nothing withholds the very evidence
                        # the placement was refused for wanting: the PACTOR-3
                        # reply position is a pair, an EMITTED control phase and
                        # a CRC packet phase (`onair._observe_p3_reply_timing`),
                        # and the changeover occupies the control slot. So the
                        # refusal starves itself -- 0913-2050 refused eleven
                        # consecutive cycles with the goodbye owed and 0913-1837
                        # eighty-eight, each with the last emitted control ageing
                        # out underneath it and no way back. A control is not a
                        # changeover: it hands the link to nobody and it is what
                        # the IRS owes every cycle anyway, so the seam that
                        # cannot place the packet can still answer, and the next
                        # decoded frame re-corroborates the pair.
                        #
                        # CS_ACK ONLY WHERE ONE IS OWED. `armed` marks the
                        # CRC-valid packet whose acknowledgement CS3 was meant to
                        # replace. With the goodbye owed nothing at the far end
                        # may be settled on the way out (and a reclaim has no
                        # packet to settle), so those ask for the repeat instead
                        # -- which is an answer, and keeps the reverse channel up.
                        ack = armed and not self._qrt_pending
                        self.io.log("unkeyed changeover refused -> "
                                    + ("ACK received packet" if ack else
                                       "repeat request holds the reverse channel"))
                        emitted = self.io.send_cs(CS_ACK if ack else CS_REQUEST)
                        # A fallback the seam also refused took no air, so
                        # the cycle it would have answered stays unanswered.
                        keyed = emitted != REFUSED
                        if ack:
                            self._rx_this_cycle = False
                            if keyed:
                                self._silent_cycles = 0
                                if not getattr(self.io, "defer_rx_close",
                                               lambda: False)():
                                    self.on_cs_emitted(CS_ACK)
                        # A CYCLE THAT PUT NOTHING ON THE AIR IS SPENT, and this
                        # seam had no budget that counted them. A refused
                        # changeover deliberately costs no retry, because the
                        # cycle it gives up is the one that re-acquires the peer
                        # -- true while something else keys in it, which is what
                        # the fallback above is. Where nothing keys at all it
                        # buys nothing: 0913-1556 printed `CHANGEOVER NOT
                        # PLACED` for 87 consecutive cycles with a silent peer,
                        # no carrier, no retry and no progress, and ended on the
                        # hold budget in cycle 347 with the mail stage still at
                        # "their turn". `_refused_cycles` is the control seam's
                        # own counter and this is the same fact about the other
                        # seam.
                        #
                        # THE GOODBYE IS NOT COUNTED HERE. It has its own
                        # decision clock (`GOODBYE_PLACE_TICKS`), deliberately
                        # longer than this budget, so that a sign-off is given
                        # every chance to reach a peer that is already leaving.
                        if keyed or self._qrt_pending:
                            self._refused_cycles = 0
                        else:
                            self._refused_cycles += 1
                            if self._refused_cycles > self.cfg.max_retries:
                                self._give_up(
                                    "the changeover seam refused every cycle")
                            return
                        # AND WHAT ENDS THIS LINK IS THE PEER'S SILENCE, never
                        # our own unplaced changeover. With the control on the
                        # air the seam is no longer starving itself, so an
                        # unanswered cycle is the same fact about the far end
                        # that the ordinary IRS control seam counts below, and it
                        # is counted the same way: a burst in the answer band
                        # forgives the cycle, a decode clears the count. Ending a
                        # link in ten cycles while a peer is still serving it is
                        # worse than re-placing the changeover for as long as
                        # fresh frames keep arriving -- the caller's hold budget
                        # bounds the session either way.
                        #
                        # THE GOODBYE IS EXEMPT HERE TOO, for the reason it is
                        # exempt above: `GOODBYE_PLACE_TICKS` is its clock and is
                        # deliberately the longer one, so that a sign-off gets
                        # every chance to reach a peer that is already leaving.
                        answered, self._burst_at_anchor = self._burst_at_anchor, False
                        if ack or self._qrt_pending or self._rx_this_cycle or answered:
                            return
                        self._charge_irs_silence()
                        return
                    self._refused_cycles = 0
                    self._take_link()
                else:
                    self._take_link()
                    self._next_seq = 0
                    self.io.send_cs(CS_BREAKIN)
                return
            if self._rx_this_cycle:
                self._rx_this_cycle = False      # on_rx_packet already answered it
                self._silent_cycles = 0
                return
            # A CYCLE WE READ NOTHING IN ENDS THE CLEAN RUN. `_gear_cs` counts
            # packets and this is the only place that sees the gaps between
            # them, so a run that survives here is one the peer's every
            # transmission got through -- which is what a request for the next
            # level is claiming about the channel.
            self._clean_run = 0
            # The IRS owes the ISS a control signal EVERY cycle, decode or no
            # decode -- that is what the reverse channel IS. shrike only ever
            # answered a packet it had decoded, so when the peer's traffic did not
            # resolve it heard nothing back at all, which is indistinguishable
            # from a dead station. Measured against WS8EOC: it took the link with
            # CS3, got silence, and broke in again, repeatedly, for the whole
            # session. "I did not get that, send it again" is a real answer and
            # keeps the link up; saying nothing ends it.
            #
            # AND A CYCLE THE ANSWER SLOT WAS FILLED IN IS NOT A SILENT ONE.
            # This budget counts silence, so what forgives a cycle has to be
            # evidence about the far end -- and the only such evidence this layer
            # is given short of a decode is WHERE a burst was. `note_burst` marks
            # the cycles that carried one inside the band an answer to our own
            # transmission has to fall in; nothing else in the cycle counts,
            # however loud.
            #
            # PRESENCE USED TO DO IT AND MUST NOT. `note_peer_heard` identifies
            # nobody -- it says so in its own first line -- and the two stations
            # it could be call for opposite actions: the peer we are asking to
            # keep transmitting, and an occupant we should be yielding to. Read
            # here it kept a dead link alive on a third station's energy ten
            # times over in one session (KB5LZK, 2026-08-28), on cycles the grid
            # reported as empty or as bursts outside the answer band, never
            # inside it. It keeps the two decisions
            # its own docstring claims -- the connect budget and the
            # yield-and-listen premise, both questions about the air -- and it no
            # longer forgives this one, which is a question about the far end.
            #
            # MEASURED against WS8EOC on 2026-08-18, 7101500 kHz, the clamped arm.
            # The link went down in hold cycle 28, and eight of the nine cycles
            # the budget was spent on carry an `fsk` or `detect` line; the
            # operator could hear the gateway still calling minutes after we had
            # unkeyed. What those cycles wrote to disk settles what it was. Each
            # capture in captures/onair-0818-2235 holds one burst of about a
            # second at 1400/1600 Hz keyed 200 Bd on the idle byte, and hold_08,
            # hold_12 and hold_17 decode CRC-valid off disk as PACTOR-1
            # changeover packets. Not a dialect we lack -- our own protocol at
            # our own speed level, read once in six. Those are the cycles this
            # branch exists to hold open, and a burst of the peer's own packet
            # length answering our own transmission is inside the band by
            # construction; the shape lines that came with them are not what
            # says so.
            #
            # AND THE CONTROL, the 2026-08-15 session, the same gateway on
            # 2026-08-15: 42 hold cycles, WS8EOC's greeting delivered in four
            # pieces at cycles 11, 12, 25 and 33, and thirteen straight cycles
            # between the second and the third with nothing but shape lines in
            # the log. That link lived because the peer was keying 100 Bd and its
            # repeat decoded most cycles, which reset the counter; the only
            # difference on the 18th is that the same gateway keyed 200 Bd and
            # the same reader got one in six. A give-up rule that turns on the
            # read rate is not measuring whether anyone is there.
            #
            # Not while recovering. That branch in `_on_nak` has already spent
            # a reading of the channel to hand it over, and a station that took
            # it would be decoding; forgiving the budget again on anything short
            # of a decode is how both ends of a lost turnaround sit and listen
            # forever.
            #
            # A slot that stays filled ends nothing on its own, and does not have
            # to: the caller's --hold still bounds the session, `RECLAIM_CODEWORDS`
            # ends the strand this branch cannot, and every cycle through here
            # puts the repeat request on the air.
            answered, self._burst_at_anchor = self._burst_at_anchor, False
            self._turn_ack_pending = False  # A quiet repeat request is not the owed ACK.
            # ASKED BEFORE IT IS CHARGED. This budget counts cycles the peer did
            # not answer, and a cycle we never put a request into is not one of
            # them: the seam refuses an instant, not the link, and charging the
            # far end for our own refusal ends links the peer is still serving.
            # AND OUR OWN REFUSAL IS BOUNDED BY THE SAME BUDGET. Forgiving the
            # peer for a cycle we never keyed is right; leaving the link with no
            # bound at all is not, and a seam that refuses is not a passing
            # instant when its reason is standing. `onair.p3_control_refusal`
            # holds "no fresh packet clock" for as long as nothing decodes --
            # which is exactly the case this branch runs in -- so the two
            # conditions feed each other and neither counter would ever move.
            # A station that can hear a peer and cannot key for eight cycles is
            # as done as one hearing nothing; only the reason differs, so the
            # line names the seam rather than the far end.
            answer = CS_REQUEST
            if getattr(self.io, "protocol", None) == spec.Protocol.PACTOR3:
                # The live driver settles occupied failures after its readers;
                # legacy callers already supplied this cycle's burst evidence.
                answer = self._p3_error_reply(
                    count=answered and self._receive_opportunity is None)
            emitted = self.io.send_cs(answer)
            if emitted == REFUSED:
                self._refused_cycles += 1
                if self._refused_cycles > self.cfg.max_retries:
                    self._give_up("the transmit seam refused every cycle")
                return
            self._refused_cycles = 0
            if (answer == CS_NAK
                    and not getattr(self.io, "defer_rx_close", lambda: False)()):
                self.on_cs_emitted(answer)
            if answered and not self._recovering:
                self.io.log("a burst where the answer is due "
                            "-> not counting this cycle")
            else:
                self._charge_irs_silence()
            return
        asked, self._upgrade_asked = self._upgrade_asked, False
        if self._rx_this_cycle:
            self._rx_this_cycle = False     # on_rx_cs already answered this cycle
            if self._inflight is not None and not self._refused_burst:
                return                      # ...and the answer WAS a packet
            # EITHER the acknowledgement cleared the last packet and left nothing
            # behind it -- an idle ISS, not a finished one, and the slot still has
            # to be filled (see `_start_next_packet`) -- OR the answer was built
            # and the seam refused it. The suppression itself is right: a cycle
            # whose answer is already on the air owes the peer nothing more, and
            # the next cycle is the one that waits for it. But it reads that off
            # `_inflight` standing, and a REFUSED burst leaves `_inflight`
            # standing too (`_sent` returns ahead of its own bookkeeping). On
            # 2026-09-11 the grant's entry packet was refused by the transmit
            # admission guard at slot 37, and slot 38 was then spent waiting for
            # an answer to a packet that had never been on the air -- one
            # refusal, two slots, at the one moment a gateway was waiting for a
            # consecutive entry sequence. `_on_nak` below puts it back on the air
            # and consumes the flag without spending a retry, because the budget
            # counts air.
        elif self._unanswered_upgrade is not None:
            # The last thing this station did was change the protocol it
            # transmits in, and the peer has not confirmed it. Two ways that ends
            # and they are different findings: silence, and a peer still asking
            # for the entry packet (`note_upgrade_unread`). A cycle that carried
            # the request is not a cycle that carried nothing, so it spends the
            # second count and not the first.
            if asked:
                # A repeated grant is a working reverse channel. It breaks a
                # silent run, but cannot confirm reception of our entry.
                self._unanswered_upgrade = 0
                self._upgrade_bursts = []
                self._unreadable_answers = 0
                if self._inflight is not None:
                    self._inflight.retries = 0
                if not self._refused_burst:
                    self._upgrade_requests += 1
            elif not self._refused_burst:
                self._unanswered_upgrade += 1
            self._upgrade_refusals = (self._upgrade_refusals + 1
                                      if self._refused_burst else 0)
            ended = None
            if self._upgrade_requests >= UPGRADE_SILENCE_CYCLES \
                    and self._step_entry():
                return
            # Four repeated grants advance an optional entry rung. The final
            # granted-entry waveform gets `ENTRY_GRANT_CYCLES` requests, not the
            # ordinary retry allowance: VE3KPG was still granting when the old
            # four-request limit made us switch back to P1, and
            # the 2026-09-11 WS8EOC arms measured the old eight-cycle value
            # ending a campaign two cycles short of the only point a gateway
            # was ever seen to react. This is a local budget, not a capability
            # verdict.
            request_limit = (ENTRY_GRANT_CYCLES if self.entry_pending
                             else UPGRADE_SILENCE_CYCLES)
            if self._upgrade_refusals >= self.cfg.max_retries:
                ended = (f"entry placement budget exhausted after "
                         f"{self._upgrade_refusals} consecutive refused transmissions; "
                         f"no entry was keyed during those cycles")
            elif asked and self._upgrade_requests >= request_limit:
                ended = (f"entry retry budget exhausted (repeated-grant rule, "
                         f"ENTRY_GRANT_CYCLES={ENTRY_GRANT_CYCLES}): it answered "
                         f"{self._upgrade_requests} cycles since the upgrade "
                         f"with repeated grants at zero bit errors, but no "
                         f"P3 entry confirmation was decoded; remote entry "
                         f"reception and mode compatibility remain unknown")
            elif self._unanswered_upgrade >= UPGRADE_SILENCE_CYCLES:
                ended = (f"silence rule (UPGRADE_SILENCE_CYCLES="
                         f"{UPGRADE_SILENCE_CYCLES}): "
                         + self._upgrade_window_ended())
            if ended is not None:
                # ONE LINE, and it is the finding. This used to log the finding
                # and then let `io.fall_back` print a verdict of its own, which
                # on 2026-08-26 stood directly under it and contradicted it.
                if self.io.fall_back(ended) is not False:
                    self._unanswered_upgrade = None
                    self._upgrade_bursts = []
                    self._unreadable_answers = 0
                # A retained entry stays unconfirmed. Repeated grants can
                # still restart its retry wait; they never acknowledge it.
        if self._inflight is not None:
            # No ACK arrived: retransmit. A repeated grant reset the packet's
            # retry count above; this attempt starts its next wait.
            self._on_nak(speed_down=False, answered=False)
            return
        self._start_next_packet()

    # -- ISS engine -------------------------------------------------------
    def _start_next_packet(self, breakin: bool = False) -> Optional[int]:
        """This cycle's data packet -- with an empty field if that is all we have.

        HAVING NOTHING TO SAY IS NOT A REASON TO STAY SILENT and leave the peer
        transmitting. PACTOR is cycle-synchronous: the IRS reads our packet at a
        fixed offset in a slot it computes from its own clock and does not look
        elsewhere for it, so "a skipped cycle presents the responder with an empty
        packet slot" (pactor1-timing.md §8.1) -- it scores a failed packet
        and spends the cycle running its timing-recovery estimator on noise.

        So an idle ISS sends an IDLE PACKET: an ordinary data packet whose field
        the renderer pads out with 0x1E and the far end strips again, advancing
        the mod-4 counter and acknowledged like any other. That is what a link
        with nothing to carry looks like -- the status byte has a name for it,
        `ptc._status_bytes` reports TRAFFIC or IDLE -- and it is not what a link
        that has ended looks like. This used to return here on an empty buffer,
        so an ISS whose host stopped writing transmitted NOTHING, indefinitely,
        while `on_cycle`'s IRS branch was answering every single cycle.

        The changeover stays the host's to ask for. An empty buffer is not the
        same fact as being finished, and a link that handed the channel back
        between two host writes would spend a turnaround on each of them;
        `on_host_over` is where that decision belongs.

        AN ENTRY PACKET IS THE ONE PACKET WITH NOTHING IN IT ON PURPOSE. While
        `entry_pending` stands the buffer is not touched at all: the field is
        empty so the renderer fills it with `spec.TEMPLATE`, and the data type is
        the one the reference declares over that template rather than the one our
        own text would want. The peer has not acquired the waveform yet, so
        anything the field carried would be bytes offered to a station that
        cannot read them and duplicated behind whatever finally does.

        AND THE DECLARATION IS A PROPERTY OF THE EMPTY FIELD, NOT OF THE ENTRY
        PACKET. `spec.DataType.PMC_ENGLISH` was reached for here because the
        entry packet is where it was first measured, which made this station's
        own idle packet and its own entry packet disagree about what an empty
        field is. In `PIII_Complete_1` they do not: all fifteen fields the
        reference writes the template into declare 6, at speed levels 1, 3 and 6
        and in three separate runs of them -- the entry packet, six cycles after
        the changeover and eight at the end -- while the traffic between those
        runs declares 4 and 7. Its changeover packet's template field is a
        sixteenth. So the type travels with the template.

        STATUS BIT 5 RIDES A LOADED FIELD WITH SOMETHING BEHIND IT: raised when
        this packet carries application bytes AND the buffer is not empty after
        it, clear otherwise. That is the rule both reference callers fit.
        DL6MAA raises it on its first loaded packet (PIII_Complete_1, 16.613 s,
        0x33, a full 59-byte SL3 field), holds it through every loaded packet to
        44.115 s, and clears it at 47.865 s (0x10) on the partial field that
        empties its buffer; W4DNA raises it on a five-byte speed-level-1 field
        (7101k_234600, 21.185 s, 0x21). Every template-filled field carries it
        clear -- DL6MAA's idle 0x19/0x1a and our own entry's 0x1a -- as do its
        changeover (0x5d) and its QRT (0x98). "The field is exactly full" fits
        both tapes equally and is not taken: neither tape separates the two, and
        this one is what the bit says it is, a stream still coming.

        Bit 5 requests a long cycle whenever more bytes remain after this
        packet. It does not require a full long field to be queued: DL6MAA's
        SL5 field at 44.115 s asks with only 459 bytes left. All six P3 levels
        can now render the grant, including the 36-byte SL1 long field.

        NEVER IN PACTOR-1, whose status byte spends bits 4 and 5 on the
        capability declaration and whose link has no long cycle (the 0x21 arms
        are the measured cost); `payload_bytes_override` is what says the link is
        in it.
        """
        if breakin:
            # BEFORE the field is chunked. A changeover packet gives its head to
            # the CS3 codeword, so its field is a different size from an ordinary
            # packet's -- and which size depends on the protocol the io is about
            # to render it in, which this layer does not know and the io may
            # change on being told (`ptc.PtcHost.breakin_now`).
            self.io.breakin_now()
        entry = self.entry_pending and self.entry_variant != "data"
        n = 0 if entry else self._payload_bytes(breakin)
        chunk = bytes(self._outbuf[:n]); del self._outbuf[:n]
        drained = not self._outbuf
        qrt = drained and self._qrt_pending and not entry
        wants_long = (self.cfg.long_cycle
                      and self.payload_bytes_override is None and not entry
                      and not breakin and not qrt
                      and bool(chunk) and not drained)
        # AND BIT 6 DOES NOT RIDE THE ASK FOR A WAVEFORM. A PACTOR-1 packet
        # carrying both the capability bits and the changeover request gives the
        # peer two things to act on and the turn is the cheaper one: VE3KPG,
        # September 13, took the channel in PACTOR-1 off our very first 0x71 --
        # "GRID REVERSED -> IRS" on packet #1 -- and no entry packet was ever
        # keyed. The run before it, announcing with bit 6 clear, drew a read
        # entry after twenty keyings. That arm flew `--p1-grant-only`, which is
        # what `io.asking_for_a_grant` is narrowed to; an ordinary PACTOR-1 link
        # still offers the turn, because an ISS with a drained buffer has no
        # other way to end its over.
        # AND NOT ON AN ENTRY THE PEER HAS NOT READ: an entry still pending is an
        # entry nothing has answered, and handing the link over on it asks a
        # station that cannot yet read us to take the channel.
        hand_over = (drained and self._over_pending and not qrt and not entry
                     and not self.entry_pending
                     and not self.io.asking_for_a_grant)
        status = spec.status_byte(self._next_seq % SEQ_MOD,
                                  data_type=(spec.DataType.PMC_ENGLISH if not chunk
                                             else spec.DataType.ASCII_8BIT),
                                  long_cycle_request=wants_long,
                                  changeover_request=hand_over,
                                  qrt=qrt)
        self._next_seq += 1
        self._peer_frame_cycle = False      # a new packet is a new question
        self._inflight = _Packet(status=status, payload=chunk, sl=self._sl,
                                 breakin=breakin, entry=entry)
        if qrt:
            self._to(State.DISCONNECTING)
        rendered = self.io.send_packet(self._inflight.sl,
                                       self._inflight.payload, status,
                                       breakin=breakin)
        self._sent(rendered)
        # ON THE AIR, NOT MERELY BUILT. `_entry_keyed` is what tells an unread
        # rung from an uninvited upgrade, and a burst the seam refused left no
        # entry packet anywhere for a peer to have failed to read.
        self._entry_keyed |= entry and rendered != REFUSED
        return rendered

    def _sent(self, rendered: Optional[int]) -> None:
        """Hold the accounting to what the RENDERER carried, not to what we cut.

        A seam that cannot build the field it was handed truncates it and says
        nothing: `ptc.PtcHost.send_packet` falls through to the short renderer
        when the peer has no long-cycle method, `placement.link_packet` cuts a
        276-byte field to the short path's 59, and the ISS then settles all 276
        on the acknowledgement of the 59. Measured on the audio harness at
        PACTOR-3: 217 bytes removed from the middle of a mail stream, no error
        anywhere in the link layer, and the B2F session died on framing several
        blocks later.

        SO THE BYTES GO BACK, at the front of the buffer and in order, and the
        packet in flight is cut to what was actually on the air. Nothing is lost
        and nothing is delivered twice -- the next packet carries the remainder.
        Refusing outright would be the other answer; it is not taken, because the
        seam that truncates is a defect of ours and the peer is owed the bytes
        either way.

        A REFUSAL IS THE OTHER THING THE SEAM CAN SAY, and it is a statement
        about the air rather than about the field: nothing went out, the packet
        in flight is untouched, and `on_cycle` owes it no retry.
        """
        self._refused_burst = rendered == REFUSED
        if self._refused_burst:
            return
        self._refused_cycles = 0            # the seam keyed: it is not stuck
        if self._inflight is not None:
            packet = self._inflight
            if packet.sent_sl != packet.sl:
                packet.sent_sl = packet.sl
                packet.sent_at_level = 0
            packet.sent_at_level += 1
        if self._inflight is not None and self._inflight.status & spec.STATUS_QRT:
            self.said_goodbye = True
            self._disconnect_ticks = None
        if rendered is None or self._inflight is None:
            return
        short = len(self._inflight.payload) - rendered
        if short <= 0:
            return
        self._outbuf[:0] = self._inflight.payload[rendered:]
        self._inflight.payload = self._inflight.payload[:rendered]
        self.io.log(f"the transmit seam rendered {rendered} of "
                    f"{rendered + short} bytes -- {short} put back at the front "
                    f"of the buffer rather than accounted as delivered")

    def _step_entry(self) -> bool:
        """Advance an explicitly configured entry variant after four requests.

        Keep the packet counter unchanged and hold unacknowledged user bytes.
        The final variant uses the normal retry budget. The template waveform
        follows pactor3.md §17.1; the rung timeout is a local retry policy.
        """
        if not self.entry_pending \
                or self._entry_rung + 1 >= len(self.cfg.entry_ladder):
            return False
        self._entry_rung += 1
        self._upgrade_requests = 0
        if self._inflight is not None:
            self._next_seq -= 1
            self._inflight = None
        self.io.log(f"{UPGRADE_SILENCE_CYCLES} more requests for the entry "
                    f"packet -> keying the '{self.entry_variant}' entry")
        self._start_next_packet()
        return True

    def _on_ack(self) -> None:
        if self._inflight is None:
            return
        self._p3_tx_gear_command = None
        status = self._inflight.status
        self._buffer_raw = max(0, self._buffer_raw - len(self._inflight.payload))
        self.io.buffer(self._buffer_raw)
        if not self._inflight.entry:
            self._entry_keyed = False       # the rung carries: see `_on_nak`
        self._inflight = None
        # No gear change on an ordinary ACK. CS4/CS5 command changes in
        # on_rx_cs; _on_nak also bounds an unacknowledged speed-up trial.
        #
        # THE UPGRADE OFFER, and whether it is a decision or a handshake is the
        # io's business rather than this layer's. Unasked it is a decision: what a
        # station can do is what it transmits, so there is nothing to be told and
        # the only question here is WHEN it is safe to stop transmitting PACTOR-1
        # -- which is the moment the PACTOR-1 link is known to be carrying data,
        # not the moment somebody answered the call. Asked, it is half a
        # handshake: the peer's `0x59A` grant reaches `ptc.PtcHost` and comes back
        # through here as this acknowledgement, carrying the answer with it.
        # WHICH protocol to move to is `io.upgrade`'s either way.
        #
        # An acknowledged data packet is exactly that evidence and the connect
        # answer is not: the answer is a verdict on the caller's 200 Bd redundancy
        # section (pactor1-control-signals.md §5) and says only that a station heard the
        # call. A real session bears the ordering out -- PIII_Complete_1 opens with
        # two connect bursts, a CS1 answer at 3.16 s and one more control signal a
        # cycle later, and its PACTOR-3 phase begins after that.
        #
        # Offered on EVERY acknowledgement rather than once, because an upgrade is
        # reversible: a peer that cannot follow is put back in PACTOR-1 by
        # `io.fall_back` when the window above runs out, or by
        # `ptc.PtcHost._follow_peer` once the upgrade has been answered and then
        # contradicted, and the target is ruled out on either path. The next
        # offer is what lets the link try the one after it. `io.upgrade` is a no-op
        # unless the link is in PACTOR-1 with a target left, and it REPORTS
        # whether it moved the link -- a change nobody answers has to be taken
        # back, which is UPGRADE_SILENCE_CYCLES, counted in `on_cycle`.
        #
        # AND ONLY WITH SOMETHING TO CARRY, which is what `payload_waiting` says.
        # The acknowledged packet's payload left the buffer three lines up, so an
        # upgrade taken on a drained one is followed by an IDLE packet: the peer's
        # first look at a waveform it has to acquire from nothing is a field of
        # 0x1E fill, and the acquisition is spent on information neither end
        # needed. Every archived session that upgraded did exactly that -- one
        # empty speed-level-3 packet keyed, and the link back in PACTOR-1 a cycle
        # later. WHETHER there is payload is this layer's to report; what to do
        # about it is `io.upgrade`'s, and a rig test overrides it there.
        self._offer_upgrade(payload_waiting=bool(self._outbuf))
        # An acknowledged CHANGEOVER REQUEST is an acknowledged packet and
        # nothing else -- the peer stays the IRS until its own CS3-headed
        # break-in arrives (measured, WS8EOC 2026-08-03; see the module header).
        # Yielding here stranded both ends receiving, so the request simply
        # stands: `_over_pending` keeps bit 6 on every packet until `_give_link`
        # runs from the break-in paths.
        if status & spec.STATUS_QRT:
            protocol = getattr(self.io, "protocol", self._ladder.protocol)
            if (getattr(self.io, "p3_qrt_confirm", False)
                    and protocol == spec.Protocol.PACTOR3):
                self._terminal_confirm_pending = True
                self._terminal_confirm_emitted = False
                self._terminal_confirm_ticks = 0
                # Stock QRT counter zero is followed by header bit one and
                # CS2. Using the opposite bit after an odd-counter QRT is an
                # unverified extension of this opt-in experiment; it keeps a
                # repeated QRT ACK from masquerading as marker confirmation.
                self._terminal_confirm_seq = (status ^ 1) & 1
                self._sl = 1
                self._cycle_long = False
                self._cycle_request = None
                self._cycle_command_emitted = False
                self._subtick = 0
                self._rx_this_cycle = False
                self.io.log("QRT acknowledged -> waiting for P3 terminal marker confirmation")
            else:
                self.goodbye_acked = True
                self._finish_disconnected()
        elif self._outbuf or self._over_pending or self._qrt_pending:
            # The next packet goes out in the cycle this acknowledgement arrived
            # in, which is the slot the peer is already timing us against. An
            # IDLE packet does not come from here: it belongs to the cycle tick,
            # the one thing that runs exactly once per slot. Chained from here it
            # would answer its own acknowledgement, and an idle link would run as
            # fast as the two ends could talk rather than at one packet a cycle.
            self._start_next_packet()

    def _offer_upgrade(self, payload_waiting: bool) -> bool:
        if not self.io.upgrade(payload_waiting=payload_waiting):
            return False
        self._unanswered_upgrade = self._upgrade_requests = 0
        self._upgrade_refusals = 0
        self._upgrade_bursts = []
        self._unreadable_answers = 0
        self._entry_rung = 0
        return True

    def _on_nak(self, speed_down: bool, *, answered: bool = True) -> None:
        """Repeat the packet, with separate silence and speed-up trial budgets.

        THE RETRY BUDGET COUNTS SILENCE, NOT REPEAT REQUESTS, and the two reach
        here by different doors: `on_rx_cs` when the peer asked for the packet
        again, `on_cycle` when nothing came back at all. Only the second is what
        the budget is about -- pactor1-timing.md §4, "on a failed receive
        the master re-sends the same packet in the next cycle and decrements a
        retry counter". A codeword decoded at zero errors is not a failed
        receive; it is the reverse channel working.

        MEASURED, WS8EOC 2026-08-09. The caller reached the state the module
        header prescribes -- ISS, buffer drained, idle packets with bit 6
        standing, waiting for the gateway's break-in -- and then gave the link
        away on its own. The gateway answered every cycle by repeating its
        codeword, which `ptc._logical_cs` reads as a request (correctly: in
        PACTOR-1 only a CHANGE of codeword acknowledges), nine of those spent the
        budget on a packet with an EMPTY FIELD, and the yield below flipped this
        end to IRS with no break-in on the air. The gateway had never asked for
        the channel, so both ends then held the receiving role and the session
        died -- the same strand the module header records for 2026-08-03,
        self-inflicted from the other side. Nothing was ever at risk: there is no
        information in an idle packet to fail to deliver.
        """
        unread = False                     # see the retransmit at the end
        self._breakin_listen = False
        if self._inflight is None:
            # A PACTOR-1 speed change requeues the packet before this runs, so
            # there is nothing to retransmit and the information goes out
            # RE-CHUNKED for the new rate instead -- in this cycle, which is the
            # one the peer asked in. Returning here left the station silent from
            # the first CS4 onward, which is a link that answers nothing.
            self._start_next_packet()
            return
        self._clean_run = 0
        packet = self._inflight
        if (not speed_down
                and getattr(self.io, "protocol", None) == spec.Protocol.PACTOR3
                and self.role == ISS and packet.payload
                and not packet.entry and not packet.breakin
                and packet.trial_from is not None
                and packet.sent_sl == self._sl
                and packet.sent_at_level >= self.cfg.p3_max_try
                and self._sl > self.cfg.min_sl):
            # MAXTry bounds only the first packet after CS4. Return to the
            # preceding speed with the same counter; _sent requeues any suffix.
            # Established-speed repeat requests do not command another drop.
            speed_down = True
            self.io.log(f"P3 MAXTry={self.cfg.p3_max_try}: seq={packet.seq} "
                        f"unacknowledged after {packet.sent_at_level} transmissions "
                        f"at SL{self._sl} -> return to SL{packet.trial_from}")
        if speed_down:
            packet.trial_from = None
        if speed_down and self._sl > self.cfg.min_sl:
            self._sl -= 1
            # A new level is a new question, so the rung budget below starts
            # again on it: a peer walking us down the ladder is the IRS doing
            # its job, not a peer that cannot read us.
            self._inflight.repeats = 0
            self.io.log(f"gear down -> SL{self._sl}")
        if not answered:
            refused, self._refused_burst = self._refused_burst, False
            # AND A LEVEL IN THE ANSWER SLOT DOES NOT END A LINK. It used to:
            # `note_unreadable_answer` ended the strand here on the finding that
            # the slot was occupied by something no reader took, and on
            # 2026-09-04 that hung up on WS8EOC over the two loudest windows of
            # the session's own band noise -- one of them 5.1 s BEFORE the
            # gateway granted PACTOR-3, the other 9.9 s after its last codeword,
            # with `0 PACTOR-4`, `0 PACTOR-3`, `0 PACTOR-2` and SCS's own decoder
            # reading nothing on the tape but our three PACTOR-1 packets. A
            # station that cannot name what it heard cannot sign off on it. What
            # the finding does now is hold the cycle (`_burst_at_anchor`), and
            # the ending is the retry budget's below or the operator's.
            # THE BUDGET COUNTS AIR. A cycle whose burst the seam refused --
            # `onair.RadioTx._refused`, and the changeover packet is what
            # reaches it -- put nothing on the channel for the peer to answer,
            # so an unanswered cycle is this station's own doing and not a loss.
            # Charging it spends the strand at the one moment we are trying to
            # get a packet placed.
            # AND OUR OWN REFUSALS ARE BOUNDED BY THE SAME BUDGET, which is the
            # IRS control seam's rule read from the sending side. Forgiving the
            # peer for a cycle we never keyed is right; leaving the link with no
            # bound at all is not, and the two conditions feed each other -- the
            # placement gate refuses for as long as nothing decodes, and nothing
            # can decode while nothing is on the air. 0913-1837 printed
            # `CHANGEOVER NOT PLACED` for 88 consecutive cycles against a peer
            # that had ceded the channel, spent no retry on any of them, and
            # died on the driver's hold budget with the login still in `_outbuf`
            # and no sign-off. A station that holds the link and cannot key for
            # the whole budget is as done as one hearing nothing.
            #
            # THE GOODBYE IS NOT COUNTED HERE, and the changeover seam's reason
            # is the same one: a teardown has its own decision clock
            # (`GOODBYE_PLACE_TICKS`), deliberately longer than this budget, so
            # that a sign-off is given every chance to reach a peer that is
            # already leaving. Spending this on it would end the link at the
            # shorter number and leave the QRT unplaced.
            #
            # AND NOT WHILE THE PEER IS STILL KEYING PACKETS AT US. That budget
            # is a statement about a seam nothing can get past, and a CRC-valid
            # frame in the same cycle is the refutation of it: the reply position
            # the placement wants is re-anchored on exactly those frames
            # (`onair._MasterGrid._observe_p3_reply_timing`), so the refusal is
            # a passing instant rather than a standing reason. This is round 23's
            # rule read from the sending side -- there a control still keyed and
            # the ending became the far end's silence; here nothing keys at all,
            # so the far end's silence IS the count, and the cycles that carried
            # a frame are the ones the changeover gets placed in.
            #
            # AND A CRC FRAME IS NOT THE ONLY WAY TO KNOW THAT. Keying 810 ms of
            # a 1250 ms cycle leaves 365 ms to read the peer's 815 ms packet in,
            # so the frame the budget wants is one our own transmitter makes
            # unreadable -- 0913-2320 fired this one line before the peer's
            # packet reached the ARQ. A burst on the peer's own corroborated
            # raster is the same refutation on the evidence the grid already
            # trusts for it: `onair._MasterGrid.note_peer_bursts` accepts one
            # only inside `MAX_PULL_S` of the comb, and `p3_control_refusal`
            # reads exactly that as "the peer is still transmitting".
            heard, self._peer_frame_cycle = self._peer_frame_cycle, False
            burst, self._peer_raster_burst = self._peer_raster_burst, False
            if self._qrt_pending:
                self._refused_cycles = 0
            elif refused:
                if (heard or burst) and self._inflight.breakin:
                    self._refused_cycles = 0
                else:
                    self._refused_cycles += 1
                    if self._refused_cycles > self.cfg.max_retries:
                        self._give_up("the transmit seam refused every cycle")
                        return
            if not refused:
                self._inflight.retries += 1
            # THE CYCLE AFTER AN UNANSWERED CHANGEOVER IS SPENT LISTENING, and
            # it is THIS cycle's keying that decides it rather than the last
            # one's. See `breakin_listen_due`; `refused` above is the seam's
            # answer to the burst before this one, and a rule taken on it hushes
            # twice in a row and re-keys nothing.
            unread = self.unconfirmed_breakin and not heard
            if self._inflight.retries > self.cfg.max_retries:
                # A silent peer is more often a turnaround we lost than a dead
                # link: if our changeover ACK or the peer's CS3 went missing, both
                # ends now believe they are the ISS and both transmit, so neither
                # can hear the other. Yielding once breaks that deafness -- if the
                # peer really did take the link its packets arrive and the session
                # resumes. If nothing arrives, the IRS side gives up below.
                #
                # DEAFNESS IS THE WHOLE PREMISE, which is why the branch sits
                # under the silence rather than beside it -- and why it asks
                # whether the premise holds. A station we cannot hear because it
                # is transmitting is on the air for 0.96 s of every 1.25; a
                # station that is not there is not.
                #
                # AND THE PREMISE IS ABOUT WHAT THE PEER SENT, NOT ABOUT OUR OWN
                # COUNTER. This asked `note_peer_heard` -- somebody is keying,
                # identified as nobody -- which is true of the peer answering us
                # correctly every cycle, of a third station, and of our own T/R
                # tail. So the handover fired on this end's retries with nothing
                # from the far end asking for the channel, and a handover is a
                # statement about the far end.
                #
                # WS8EOC, 2026-08-26, and it is the measured cost of the old
                # rule. The gateway sent `0x59A` at zero bit errors on
                # thirty-six consecutive cycles and never once keyed a data
                # packet: it held the receiving role for the whole session. The
                # yield fired anyway, `reverse` rotated the transmit anchor by
                # its correct 840 ms into a role the peer had not agreed to, and
                # this station's 120 ms codeword then landed 69 ms in FRONT of
                # the gateway's answer instant for five cycles. Five of the
                # twenty codewords we transmitted over that session are on the
                # far side of this line. The rotation was right; the role was
                # not.
                #
                # WHAT COUNTS INSTEAD is `_peer_asked_for_channel`: a CRC-valid
                # data packet while this end also holds the sending role. That
                # is the stranded-ISS signature and nothing else produces it --
                # an IRS sends one control signal a cycle and no packets. A CS3
                # never reaches here at all; `on_rx_cs` yields on it the instant
                # it decodes, which is the other half of the same rule.
                #
                # THE OPERATOR HEARD THIS BEFORE ANY LOG DID -- "we acted like
                # the flow direction reversed but nothing was going on
                # on-channel". Two of the four sessions that reached a peer on
                # 2026-08-14 changed over on nothing at all:
                # rig-session-20260814-211344 yielded after nine straight
                # `HOLD n RX (quiet)` cycles, having heard nothing since the
                # connect, and -212300 after nine of its own. Both had repeated
                # packet #1 nine times; neither peer transmitted again; both
                # sessions then spent their remaining budget sending 120 ms
                # codewords into an empty channel and died 10 and 6 cycles
                # later, the counter still at #1. A yield is a handover of the
                # channel and there was nothing to hand it to.
                #
                # The evidence is spent by the yield, so it is consumed under
                # the guard rather than beside it. A station already recovering
                # reaches this line only behind a host QRT -- the goodbye has to
                # go out as the ISS -- and that path aborts below on the same
                # silence, so the flag it leaves standing is torn down with the
                # link. Recovery itself ends only at a CRC-valid packet, which
                # is presence of the strongest kind and re-arms this branch on
                # its own account; shape-only energy heard while listening
                # cannot end a recovery and so can never carry into a second
                # handover.
                if not self._recovering and self._peer_asked_for_channel:
                    self._peer_asked_for_channel = self._peer_heard = False
                    self.io.log("max retries, and the peer sent a data packet "
                                "of its own while we held the sending role "
                                "-> yield the link and listen")
                    self._recovering = True
                    self._silent_cycles = 0
                    self._yield_link()
                    return
                # SINCE THE LINK CAME UP, not "on the channel". `_peer_heard` is
                # cleared at the CONNECTED transition and by the yield above --
                # not by `_yield_link`, which serves the acked break-ins too and
                # is reached from the very decode that sets it. So this branch
                # knows only that nothing has been heard within the link -- the
                # gateway may have answered the call minutes earlier in the same
                # session. An operator reads "nothing on the channel" as an
                # empty frequency and a reason to QSY.
                #
                # AND THE DECLINED HANDOVER IS A LINE RATHER THAN AN ABSENCE.
                # `_peer_heard` is not what says it: it is a latch, set by the
                # last thing decoded and read cycles later, and `note_peer_heard`
                # identifies nobody -- reporting it as "the peer" here would make
                # the same claim the trigger just stopped making. What the
                # last branch below can say is that nothing the far end sent
                # asked for the channel, so there is nothing to hand it to; the
                # first says the opposite, and neither claims an empty channel.
                self._peer_heard = False
                if self._peer_wants_us_sending:
                    why = ("the peer is still asking us to send, so our "
                           "changeover packet is not being read; yielding "
                           "would hand the channel back to the station "
                           "asking us to take it")
                elif self._recovering:
                    why = "the yield bought nothing"
                else:
                    why = ("nothing the peer sent asked for the channel, so "
                           "there is nothing to hand it to; NOT reversing")
                self._give_up(f"max retries ({why})")
                return
        else:
            # A REPEAT REQUEST COUNTS WHEREVER IT ARRIVES, and the bound below
            # is the only part of it that belongs to an entry packet: the count
            # runs on a PACTOR-1 packet as readily as on a PACTOR-3 one, and
            # `_entry_keyed` is what decides which links the bound applies to.
            self._inflight.repeats += 1
            # AND A WAVEFORM THE PEER CANNOT READ IS NOT SILENCE EITHER. The
            # budget above is right to forgive a repeat request -- the reverse
            # channel is working -- but a peer that holds its acknowledgement
            # against every packet of a rung it has never taken one packet of
            # is telling us the rung is unreadable, and there is nothing at the
            # other end of that loop. KB5LZK, 2026-09-11: the grant, the entry
            # packet read and answered, then the same counter-3 field keyed 27
            # times against a held CS1. `UNREAD_RUNG_REPEATS` carries the size
            # and why it is not a verdict; `_entry_keyed` carries which links
            # it applies to, which is the ones that asked the question an entry
            # packet asks. An uninvited upgrade keys no entry packet and is
            # bounded by `UPGRADE_SILENCE_CYCLES` instead.
            if (self._entry_keyed
                    and self._inflight.repeats > UNREAD_RUNG_REPEATS):
                self._entry_keyed = False
                why = (f"the peer read the entry packet and then asked "
                       f"{self._inflight.repeats} times for the first PACTOR-3 "
                       f"data packet behind it without acknowledging one")
                if getattr(self.io, "no_p3_fallback", False):
                    self._give_up(f"{why}, and --no-p3-fallback leaves no rung "
                                  f"to retreat to")
                    return
                self.io.fall_back(why, rule_out=False)
                if self._inflight is None:
                    self._start_next_packet()
                    return
        self._inflight.sl = self._sl
        # A repeat of the CHANGEOVER packet is still a changeover packet -- "a
        # repeat of the BK packet is requested with CS2" -- so the head goes with
        # it. Retransmitting it as an ordinary frame would leave the peer, which is
        # still the ISS until it decodes one, with nothing to yield to.
        self._sent(self.io.send_packet(  # retransmit
            self._inflight.sl, self._inflight.payload, self._inflight.status,
            breakin=self._inflight.breakin))
        # ...and now the burst is on the air, which is what makes the next cycle
        # unreadable. A cycle the seam refused put nothing there and leaves the
        # receiver the whole of it already.
        self._breakin_listen = unread and not self._refused_burst

    # -- changeover -------------------------------------------------------
    @property
    def unconfirmed_breakin(self) -> bool:
        """An emitted local turn packet still awaits peer acknowledgement."""
        return (self.state in (State.CONNECTED, State.DISCONNECTING)
                and self.role == ISS and not self.entry_pending
                and self._inflight is not None and self._inflight.breakin)

    @property
    def breakin_listen_due(self) -> bool:
        """Give this cycle to the receiver: the last one keyed an unread changeover.

        As PACTOR-3 ISS we key 810 ms of a 1250 ms cycle, so the peer's 815 ms
        packet on its own raster can only land in the 365 ms left over -- under
        `live.MIN_DECODE_S` and under `RollingRx.flush`'s half-second floor. Its
        answer to our changeover is therefore unreadable BY CONSTRUCTION, and
        re-keying the changeover guarantees the deafness that made it so.
        WS8EOC, 0913-2320: the gateway held one 0-byte `status=0x43` packet up on
        its comb for 25 consecutive cycles, `note_p3_packet` was never once
        called, the reply-timing pair aged out at `ONSET_MAX_CYCLES`, and this
        end signed off with the login still in the buffer.

        So the cycle after an unanswered changeover keys nothing and listens the
        whole 1.25 s, which is the one window the peer's packet fits in; the
        changeover goes out again on the cycle after that. The alternation needs
        no bound of its own -- the listening cycle is refused at the seam and
        charges `_refused_cycles`, the keyed one charges `_inflight.retries`, and
        each ends the link at `max_retries` exactly as it did before.
        """
        return self._breakin_listen and self.unconfirmed_breakin

    def note_peer_raster_burst(self) -> None:
        """The driver heard a burst on the peer's corroborated raster.

        Not a decode and not attributed by a reader: `onair._MasterGrid.
        note_peer_bursts` accepts an onset only inside `MAX_PULL_S` of the comb
        three CRC frames established, which is the same evidence
        `p3_control_refusal` reads as "the peer is still transmitting". Held for
        one cycle, and spent where a CRC frame is spent -- see `_on_nak`.
        """
        self._peer_raster_burst = True

    def on_unconfirmed_breakin_head(self) -> None:
        """Wait for the P3 body without settling bytes or rotating the clock."""
        if self.unconfirmed_breakin:
            self._rx_this_cycle = True
            self._turn_head_wait = True

    def _resume_peer_stint(self) -> None:
        """The driver proved the peer repeated its old CS3 on its old raster.

        WS8EOC 2026-09-12 13:37 repeated RMS within 1.8 ms of that raster
        after our local CS3. This is not acknowledgement of our three bytes
        or a new receive stint: retain its counter and streaming decoder.
        """
        self.requeue_inflight()
        self.role = IRS
        self._over_pending = self._breakin_armed = self._stint_tail = False
        self._cycle_request = None
        self._cycle_command_emitted = False
        self._clean_run = self._peer_receiving = 0
        self._turn_ack_owed = True
        self._breakin_pending = bool(self._outbuf) or self._qrt_pending
        self.io.log("peer repeated its previous turn -> IRS; bytes unacknowledged, ACK first")

    @property
    def _reclaiming(self) -> bool:
        """The strand this station is entitled to end by taking the link back.

        DERIVED, not pended. A reclaim that set `_breakin_pending` would leave a
        flag standing after a CS3 withdrew the finding, and `on_rx_packet` arms
        any pending break-in behind the next packet that decodes -- so a strand
        that resolved itself would still have cost a changeover. This condition
        is exactly as true as the evidence under it.
        """
        return self.role == IRS and self._peer_receiving >= RECLAIM_CODEWORDS

    @property
    def taking_link(self) -> bool:
        """This station's standing intent to seize the channel, either reason.

        THE STANDING INTENT, NOT THE ARMING. What licenses the packet is a
        decode inside the window that has to close early enough to reach the
        instant it keys at, so the driver asks this BEFORE the window and errs
        towards a window that closes early rather than a key it cannot make --
        `ptc.PtcHost.breakin_due` is the one caller and `onair.RadioTx.breakin_due`
        is what it feeds. Public because it was `_breakin_pending or
        _reclaiming` read through two underscores from another module.

        AND IT STANDS FOR THE WHOLE OF THE CHANGEOVER, NOT ITS FIRST CYCLE.
        `_take_link` clears `_breakin_pending` and turns us into the ISS the
        moment the packet is built, so from the second keying on this read False
        while a changeover packet was still the burst in hand -- eight of the
        nine on each of 2026-09-04's mail arms. The driver then sized the whole
        cycle for a boundary key, leaving `settle - early` for the render where a
        boundary keying gets the settle whole, and the placement it made was
        unreachable by construction. The inflight says which burst it is.
        """
        return not self._turn_ack_owed and (self._breakin_pending or self._reclaiming
                or (self._inflight is not None and self._inflight.breakin))

    def _take_link(self) -> None:
        """Become the ISS. We transmit from the next cycle, not from here.

        And the peer is still the ISS for one more cycle, because that is how
        long it takes our changeover packet to tell it otherwise -- so the
        receive stream stays open behind us (`_stint_tail`, read in
        `on_rx_packet`) rather than closing with the role.
        """
        self._breakin_pending = self._breakin_armed = False
        self._peer_receiving = 0
        self._stint_tail = True
        self._p3_tx_gear_command = None
        self._p3_rx_errors, self._p3_rx_sl = 0, None
        self.role = ISS
        self._cycle_request = None
        self._cycle_command_emitted = False
        self._clean_run = 0
        self.io.log("changeover -> ISS")

    def _give_link(self) -> None:
        """Become the IRS: our changeover request was acknowledged."""
        self._over_pending = False
        self._p3_tx_gear_command = None
        self.role = IRS
        self._cycle_request = None
        self._cycle_command_emitted = False
        self._clean_run = self._peer_receiving = 0
        self._stint_tail = False
        self._reset_rx_seq()
        self.io.log("changeover -> IRS")
        if self._qrt_pending:
            self._breakin_pending = True     # we still owe a QRT packet

    def _reset_rx_seq(self) -> None:
        self._p3_rx_errors, self._p3_rx_sl = 0, None
        self._repeat_answer, self._repeat_run = None, 0
        self._repeat_pending = None
        # The peer's counter resets to 0 as it takes the link, so the expectation
        # this end grades against resets with it -- a packet one short of a stale
        # expectation reads as a repeat, and a repeat is acknowledged with its
        # payload thrown away. It has to happen on EVERY path that hands the link
        # over, which is why it lives here: it used to sit beside two of the four
        # calls, and the connect answered by a break-in -- a Winlink RMS taking the
        # channel to send its greeting -- went through neither of them.
        #
        # Both halves clear together. Until something has been delivered under the
        # new numbering there is nothing for a repeat to be a repeat OF, so the
        # first packet of the new direction is taken at whatever counter it
        # carries; demanding a particular one is a guess, and a wrong guess costs
        # the payload rather than a cycle. `_is_replay` is what the bytes say
        # where the counter has nothing left to say, and `_last_rx_field`
        # deliberately does NOT clear here -- it is the whole of the identity
        # that survives a reversal.
        #
        # SO THE SEED IS NOT AN EXPECTATION, it is the codeword owed while the
        # new numbering is still empty, read back through `rx_seq`. Zero reads
        # back as counter 3 and keys CS2, and CS2 is what an IRS waiting for the
        # changeover packet owes AT EVERY LEVEL: the packet it is waiting for
        # carries counter 0, so CS1 is that packet's acknowledgement, and a
        # station that has not read it must not key the word that confirms it.
        # A symmetric peer grades our codeword against the counter it has on the
        # air (`ptc._counter_logical_cs`): CS1 in that slot releases the
        # changeover packet's field -- a Winlink greeting -- and the peer never
        # sends it again. CS1 arrives on the cycle one decodes and not before,
        # because `_accept_field` has advanced the counter by then.
        self._expected_seq = 0
        self._rx_seen = False
        self._rx_ahead = 0

    def _is_replay(self, payload: bytes, data_type: int) -> bool:
        """Match the first nonempty field's coding mode and bytes after a reversal."""
        return bool(payload) and not self._rx_seen \
            and (data_type, bytes(payload)) == self._last_rx_field

    def requeue_inflight(self) -> None:
        """Put the unacknowledged packet back at the head of the send queue.

        For the RECOVERY yield -- the end that gave up on a silent peer -- and
        for the goodbye that has to displace an idle packet (`on_host_disconnect`).
        The information was never confirmed, so it goes out again when we next
        hold the link. A genuine new peer CS3 must NOT come here: the CS3 is
        the acknowledgement of the packet in flight, and requeueing it re-sent
        delivered bytes under a counter the break-in had reset, past a
        duplicate check that is seq-keyed -- see `_yield_link`. A driver-proven
        repeat of the previous peer turn is different: `_resume_peer_stint`
        restores those unacknowledged bytes without resetting receive numbering.

        THE COUNTER COMES BACK WITH THE PAYLOAD. It did not, for the grant: the
        packet that replaced this one was the entry packet and the numbering
        carried on so the announcement rode the field behind it. A grant settles
        its packet now (`on_rx_grant`) and no caller is left that wants the
        bytes back under a number they have never been keyed under.

        What remains here is the ARQ residue no protocol removes: if the peer
        delivered this packet and then every acknowledgement faded for the whole
        retry budget, the requeue re-sends delivered bytes under a fresh
        counter. That needs max_retries consecutive one-way losses on a link
        that still works in the other direction; the CS3 case needed one.

        `rechunk_inflight` is the other reason to unbuild a packet, and the two
        are not the same decision.
        """
        if self._inflight is None:
            return
        self._outbuf[:0] = self._inflight.payload
        self._next_seq -= 1
        self._inflight = None

    def rechunk_inflight(self) -> None:
        """The field under the packet in flight changed size: build it again.

        A PACTOR-1 speed-down carries the SAME information at the new field size,
        20 bytes becoming three 8-byte packets, and a protocol fallback does the
        same from 59; nothing else in the FSM can split a packet already built.
        The sequence number comes back with the payload, so a peer that DID
        decode the original catches the repeat on its duplicate check.

        AN EMPTY FIELD STAYS IN FLIGHT. Re-chunking nothing is nothing, and the
        counter is then the whole difficulty: wound back and refilled from the
        buffer, it hands a number the peer may already have acknowledged to
        DIFFERENT information, which the peer's duplicate check throws away --
        measured, a payload lost on the cycle a link fell back out of PACTOR-3
        after an idle packet. Advancing past it instead presents the peer with a
        sequence GAP, which it can only answer with a NAK it will never be
        satisfied on. Leaving it alone retransmits the same empty packet under
        the same counter, rendered in whatever protocol the link is now in, which
        is exactly what a repeat means and fits any field there is.
        """
        if self._inflight is None or not self._inflight.payload:
            return
        self.requeue_inflight()

    def drop_queued_prefix(self, blob: bytes) -> bool:
        """Take unsent link-setup bytes back off the FRONT of the queue.

        The one thing that may be unqueued, and it is unqueued because nothing
        wants it any more rather than because it failed. The PACTOR-1 callsign
        announcement is queued by the spec (`ptc._answer_link_setup`) -- correct
        while the link is still setting up, and stale the moment an application
        has an answer behind it: the changeover packet's
        field is three bytes, so those three bytes are the whole of what a peer
        waiting for a login gets, and `1w9` is not a login. MEASURED, 0913-2143:
        eight changeover packets carrying `b'1w9'`, the B2F login behind them in
        `_outbuf`, and not one login byte on the air in the whole stint.

        Only from the front, only if it is still there whole, and never out of
        the packet in flight -- bytes that have been offered to the peer are the
        ARQ's, and taking them back mid-flight is how a counter and a field stop
        agreeing. `blob` is what the host wrote, escaped here the way
        `on_host_data` escaped it. Reports whether it dropped anything.
        """
        wire = compress.transparent(blob)
        if not wire or not self._outbuf.startswith(wire):
            return False
        del self._outbuf[:len(wire)]
        self._buffer_raw = max(0, self._buffer_raw - len(wire))
        self.io.buffer(self._buffer_raw)
        return True

    def _yield_link(self, *, acked: bool = False) -> None:
        """Give way to the peer's break-in.

        `acked` is for the yields a decoded CS3 triggers, and it is the
        duplicate-suppression half of the changeover: "the IRS sends it after a
        correctly received packet", so the CS3 arrives in the acknowledgement
        slot of whatever we had in flight and IS its acknowledgement. Requeueing
        that packet re-sent bytes the peer had already delivered, under a
        counter the break-in had just reset -- five duplicate bytes into the
        payload stream, caught by this module's own loopback. The
        retry-exhaustion yield stays a requeue: there the peer has answered
        nothing, and unconfirmed information belongs in the buffer.
        """
        if acked:
            self._settle_inflight()
        else:
            self.requeue_inflight()
        # Yielding changes the role, not the lifetime of the link. A pending
        # goodbye survives in _qrt_pending and _give_link re-arms its packet;
        # exposing CONNECTED here resurrected the host state during teardown.
        self._give_link()

    def _settle_inflight(self) -> None:
        """The packet in flight was delivered: account for it and let it go.

        `_on_ack` without the consequences -- no upgrade offer, no next packet,
        no QRT handling. A station mid-yield is not starting anything; the flags
        that survive (`_qrt_pending`, `_over_pending`) are re-examined from the
        IRS side by `_give_link`.
        """
        if self._inflight is None:
            return
        self._buffer_raw = max(0, self._buffer_raw - len(self._inflight.payload))
        self.io.buffer(self._buffer_raw)
        self._inflight = None

    # -- lifecycle --------------------------------------------------------
    def _enter_connected(self) -> None:
        self._p3_tx_gear_command = None
        self._p3_rx_errors, self._p3_rx_sl = 0, None
        self._terminal_confirm_pending = self._terminal_confirm_emitted = False
        self._terminal_confirm_ticks = 0
        self._terminal_confirm_seq = 1
        self._turn_ack_owed = self._turn_ack_pending = False
        self._turn_head_wait = self._turn_ack_cycle = False
        self._to(State.CONNECTED)
        self.said_goodbye = self.goodbye_acked = False
        self._goodbye_cycles = self._goodbye_subtick = 0
        self._disconnect_ticks = None
        self.goodbye_unplaceable = False
        self._budget_goodbye = False
        self._rx_close_pending = False
        self._rx_close_cycles = self._rx_close_subtick = 0
        self._sl = self.cfg.min_sl
        self._clean_run = 0
        self._repeat_answer = None
        self._repeat_run = 0
        self._repeat_pending = None
        self._cycle_long = False
        self._cycle_request = None
        self._cycle_command_emitted = False
        self._subtick = 0
        # A burst heard while CALLING is not a burst heard on the link, and the
        # connect budget is the only thing entitled to spend it. Left standing,
        # rig-session-20260814-212300's single `peer burst heard (not decoded)`
        # of cycle 6 was still true nine silent hold cycles later and would have
        # bought the yield in `_on_nak` by itself.
        self._peer_heard = self._peer_asked_for_channel = False
        self.peer_reads = 0
        self._peer_wants_us_sending = self._peer_frame_cycle = False
        self._peer_raster_burst = self._breakin_listen = False
        self._peer_receiving = 0
        self._stint_tail = False
        self._burst_at_anchor = False
        # Counting from ONE. "Das erste normale Datenpaket mit Head=AA (HEX) und
        # Paketzaehler=1" -- the connect burst is packet zero (its own header is
        # the 0x55 sync byte) and the first data packet is one. Starting at zero
        # and patching it up at the PACTOR-1 seam, which is what `(status & 3) or
        # 1` in ptc.send_packet did, gave the first TWO packets of every link the
        # same counter, so the far end's duplicate check had to throw one away.
        self._next_seq = self._expected_seq = 1
        self._rx_seen = False
        self._rx_ahead = 0
        self._last_rx_field = None
        self._rx_decoder = compress.Decoder()      # a new link is a new stream
        self._rx_si = compress.Supervisor()
        self.io.connected(self.mycall, self.dxcall)

    def _give_up(self, why: str) -> None:
        """End a live link by saying so, rather than by ceasing to transmit.

        Every budget in this module used to reach `on_host_abort`, which is the
        HOST's "stop now" -- right for a host that has stopped caring and wrong
        for a station that has. It tears the FSM down without putting anything on
        the air, and the peer goes on owing its side of a cycle to somebody who
        left; the protocol gives it no inactivity timeout to notice with, so it
        holds the channel until an operator ends it (`GOODBYE_CYCLES`).

        MEASURED, four sessions on 2026-08-22 (act-01, act-04, act-06, act-07):
        the upgrade went unanswered, the link fell back to PACTOR-1, the retry
        budget yielded the channel to a peer that had never asked for it, and
        this end then vanished from the IRS role -- both stations holding the
        receiving role, which is what the operator heard as "both of us acting as
        though we were receiving".

        `on_host_disconnect` is the whole of the goodbye and knows both roles: an
        IRS breaks in to send one, because QRT rides a packet. What is left here
        is the discard that lets that packet be built -- the buffer and the
        packet in flight were never confirmed and there is no cycle left to
        confirm them in -- and the idempotence, so that a budget running out
        again during the teardown does not start a second goodbye. What ends the
        first one is `GOODBYE_CYCLES`, in `on_cycle`.
        """
        if self._qrt_pending:
            ending = ("the goodbye went unanswered" if self.said_goodbye
                      else "goodbye could not be placed")
            self.goodbye_unplaceable = not self.said_goodbye
            self.io.log(f"{why} -> abort, {ending}")
            self.on_host_abort()
            return
        self.io.log(f"{why} -> QRT")
        self._outbuf.clear()
        self._inflight = None
        self._budget_goodbye = (
            getattr(self.io, "protocol", None) == spec.Protocol.PACTOR3)
        self.on_host_disconnect()

    def _finish_disconnected(self) -> None:
        self._terminal_confirm_pending = self._terminal_confirm_emitted = False
        self._terminal_confirm_ticks = 0
        self._terminal_confirm_seq = 1
        self._repeat_answer, self._repeat_run = None, 0
        self._repeat_pending = None
        self._turn_ack_owed = self._turn_ack_pending = False
        self._turn_head_wait = self._turn_ack_cycle = False
        self._disconnect_ticks = None
        self._budget_goodbye = False
        self._cycle_request = None
        self._cycle_command_emitted = False
        self._outbuf.clear()
        self._inflight = None
        self._buffer_raw = 0
        self._over_pending = self._breakin_pending = self._qrt_pending = False
        self._rx_close_pending = False
        self._rx_close_cycles = self._rx_close_subtick = 0
        self._breakin_armed = self._stint_tail = False
        self._recovering = False
        self._peer_asked_for_channel = False
        self._peer_wants_us_sending = self._peer_frame_cycle = False
        self._peer_raster_burst = self._breakin_listen = False
        self._peer_receiving = 0
        self._unanswered_upgrade = None
        self.role, self.answering = None, False
        self.io.disconnected()
        self._to(State.LISTENING if self._listen else State.DISCONNECTED)


# --------------------------------------------------------------------------- #
# Length-prefixed reassembly (same scheme as kestrel frames.py) — deliver
# exactly the bytes the host wrote, ignoring packet zero-padding.
# --------------------------------------------------------------------------- #
# In-memory loopback self-test: two PactorArq endpoints exchange a real session.
def _selftest() -> None:
    """Both changeover directions, a pre-empted live packet, a lost break-in, a
    break-in read from its CS3 head alone and a QRT, asserting byte-exact
    delivery both ways.

    A function rather than a bare ``__main__`` block, because a self-test that
    only runs when someone types the module name is how five duplicate bytes
    rode through a green suite on 2026-08-03: pytest never collects
    ``__main__``, so the one gate that covered the requeue-across-changeover
    path never ran. `tests/shrike/test_arq.py` collects this; the module name
    still runs it directly.
    """

    class LoopIO(ArqIO):
        """Routes this endpoint's outputs into the peer's input events.

        The 'channel' is a shared event queue; a driver clocks the cycle grid.
        `drop_next` corrupts the next data packet once, `drop_cs` loses the next
        control signal, `drop_breakin` loses the next changeover packet and
        `breakin_to_cs` strips the next changeover packet down to its CS3 head,
        to exercise the NAK, the unacknowledged, the lost-turnaround and the
        head-only break-in paths.

        `on_packet` fires as this end transmits one, which is how a test arms a
        loss against a TRANSMISSION rather than against a cycle count. `pump`
        settles a whole exchange, so one keepalive can carry a station's entire
        buffer inside a single call and a drop armed between two hand-clocked
        cycles lands on whatever burst that leaves.
        """
        def __init__(self, name):
            self.name = name
            self.delivered = bytearray()
            self.q: list = []                # (kind, args) events for the peer
            self.drop_next = False
            self.drop_cs = False
            self.drop_breakin = False
            self.breakin_to_cs = False
            self.on_packet = None

        def connect_burst(self, mycall, dxcall):
            self.q.append(("connect", (mycall, dxcall)))
        def send_packet(self, sl, payload, status, breakin=False):
            ok, self.drop_next = not self.drop_next, False
            if self.drop_breakin and breakin:
                self.drop_breakin = False
                return
            if self.breakin_to_cs and breakin:
                # The body is lost to the channel; the CS3 head decodes alone,
                # so it arrives at the peer as a bare control signal.
                self.breakin_to_cs = False
                self.q.append(("cs", (CS_BREAKIN,)))
                if self.on_packet is not None:
                    self.on_packet()
                return
            # Snapshot the transmitted header's cycle, not its status request.
            self.q.append(("packet", (sl, payload, status, ok, breakin,
                                       None, self.fsm.cycle_long)))
            if self.on_packet is not None:
                self.on_packet()
        def send_cs(self, cs_index):
            if self.drop_cs:
                self.drop_cs = False
                return
            self.q.append(("cs", (cs_index,)))
        def connected(self, mycall, dxcall):
            print(f"  [{self.name}] CONNECTED {mycall} <- {dxcall}")
        def disconnected(self):
            print(f"  [{self.name}] DISCONNECTED")
        def deliver(self, blob):
            self.delivered += blob
        def buffer(self, n): pass
        def log(self, m): pass  # print(f"  [{self.name}] {m}")

    a_io, b_io = LoopIO("A"), LoopIO("B")
    A, B = PactorArq(a_io), PactorArq(b_io)
    a_io.fsm, b_io.fsm = A, B
    ends = ((A, a_io), (B, b_io))

    def pump():
        """Deliver all queued events between the two endpoints until quiescent."""
        for _ in range(200):
            moved = False
            for src, dst in ((a_io, B), (b_io, A)):
                while src.q:
                    moved = True
                    kind, args = src.q.pop(0)
                    getattr(dst, {"connect": "on_rx_connect", "packet": "on_rx_packet",
                                  "cs": "on_rx_cs"}[kind])(*args)
            if not moved:
                return
        raise RuntimeError("pump did not settle")

    def run(cycles=40, until=None):
        """Clock the cycle grid, both ends per cycle, until `until` holds."""
        for _ in range(cycles):
            for fsm, _io in ends:
                fsm.on_cycle()
                pump()
            if until is not None and until():
                return
        if until is not None:
            raise AssertionError("session did not settle")

    def idle(fsm):
        return not fsm._outbuf and fsm._inflight is None

    msg_a = b"HELLO DE SHRIKE - PACTOR-3 ARQ LOOPBACK TEST - " * 6      # ~282 bytes
    msg_b = b"; SHRIKE ECHO - THE OTHER DIRECTION - " * 5               # ~190 bytes
    msg_a2 = b"FF\r73 DE N0CALL SK\r"
    msg_b2 = b"; STRANDED-ISS RECOVERY PROBE\r"
    msg_b3 = b"73 OM"           # one packet at any speed level (5 bytes at SL1)

    B.on_host_listen(True)
    A.on_host_connect("N0CALL", "N0DX")
    pump()
    assert A.state == State.CONNECTED and B.state == State.CONNECTED, "connect failed"
    assert (A.role, B.role) == (ISS, IRS), "roles not assigned by the connect"

    # -- A -> B, with one corrupted packet to exercise the NAK path -------
    A.on_host_data(msg_a)
    a_io.drop_next = True
    run(until=lambda: idle(A))

    # -- A hands over (%O as ISS): the request rides the next status byte --
    A.on_host_over()
    run(until=lambda: A.role == IRS and B.role == ISS)
    print(f"  changeover 1: A={A.role} B={B.role}")

    # -- B -> A, in the other direction. Losing A's ACK of the first packet
    #    leaves B holding it, so the break-in below has to pre-empt a live
    #    packet -- the case that tests the requeue and the duplicate check.
    B.on_host_data(msg_b)
    # WHICH control signal is lost decides what this tests. The IRS owes the ISS
    # one every cycle whether or not it decoded anything, so a drop armed by the
    # clock is spent on A's keepalive and B's packet comes back acknowledged --
    # which is not the state the break-in below is meant to pre-empt. Arm it on
    # B's own first transmission instead, where it cannot land anywhere else.
    def lose_the_answer():
        a_io.drop_cs = True
        b_io.on_packet = None

    b_io.on_packet = lose_the_answer
    # ONE cycle, and it used to take two: an ISS whose acknowledgement had left it
    # with an empty buffer spent the following cycle transmitting nothing at all,
    # so B's first packet did not go out until the cycle after the host wrote it.
    run(cycles=1)
    assert B._inflight is not None, "B should be holding an unacknowledged packet"

    # -- A takes the link back by force (%I / CS3 break-in) ----------------
    A.on_host_breakin()
    run(until=lambda: A.role == ISS and B.role == IRS)
    print(f"  changeover 2: A={A.role} B={B.role} (pre-empted a live packet)")

    A.on_host_data(msg_a2)
    run(until=lambda: idle(A))

    # -- back to B so it can finish what the break-in interrupted ----------
    A.on_host_over()
    run(until=lambda: B.role == ISS and idle(B))

    # -- a lost break-in: both ends believe they are the ISS ---------------
    # Half-duplex on a cycle grid, so neither can hear the other transmitting.
    # The stalled ISS has to yield on its own to break the deafness.
    B.on_host_data(msg_b2)
    a_io.drop_breakin = True                  # A's changeover packet never lands
    A.on_host_breakin()
    # TWO cycles to the strand, not one: a break-in only goes out behind a
    # packet that decoded -- the CS3 replaces that packet's acknowledgement --
    # so A spends the first cycle reading B before it can seize the second.
    run(cycles=2)
    assert A.role == ISS and B.role == ISS, "the lost CS3 should strand both as ISS"
    run(until=lambda: (A.role, B.role) in ((ISS, IRS), (IRS, ISS)),
        cycles=4 * (A.cfg.max_retries + 2))
    print(f"  recovered from a lost break-in: A={A.role} B={B.role}")
    assert A.state == State.CONNECTED and B.state == State.CONNECTED, "link lost"
    # The probe is still queued at B, which comes out of the recovery as the IRS,
    # so the link has to be turned around once more before it can go out -- and
    # the recovery is only worth anything if the link carries traffic afterwards.
    # This used to settle without the turnaround, and it settled by the link
    # DYING: A had an empty buffer, an ISS with an empty buffer transmitted
    # nothing at all, and B eventually gave up on a silent peer and cleared the
    # probe out of its own buffer on the way down.
    if A.role == ISS:
        A.on_host_over()
        run(until=lambda: B.role == ISS)
    run(until=lambda: idle(A) and idle(B))

    # -- a break-in recognised from its CS3 head alone ---------------------
    # The changeover packet's body is lost to the channel; what survives is the
    # codeword in the acknowledgement slot, so the yield runs from `on_rx_cs`
    # rather than from a decoded packet -- the other place the fix in
    # `_yield_link` has to hold. The head is still the acknowledgement of the
    # packet B has in flight: B must settle it, not requeue it, or the bytes A
    # already delivered go out again once the link turns around.
    B.on_host_data(msg_b3)

    def lose_this_answer_too():
        a_io.drop_cs = True
        b_io.on_packet = None

    b_io.on_packet = lose_this_answer_too
    run(cycles=1)
    assert B._inflight is not None, "B should be holding an unacknowledged packet"
    a_io.breakin_to_cs = True         # A's changeover packet arrives head-only
    A.on_host_breakin()
    run(until=lambda: A.role == ISS and B.role == IRS)
    assert not a_io.breakin_to_cs, "the head-only break-in never went out"
    assert B._inflight is None and not B._outbuf, \
        "the CS3 head acknowledges the packet in flight: settle it, not requeue"
    print(f"  changeover 3: A={A.role} B={B.role} (break-in read from its CS3 head)")
    run(until=lambda: idle(A) and idle(B))
    # ...and hand the link back, so anything a requeue put back in B's buffer
    # would go out again and break the byte-exact accounting below.
    A.on_host_over()
    run(until=lambda: B.role == ISS and idle(B))

    # -- QRT, from the receiving end: break in first, then send it ---------
    A.on_host_disconnect()
    run(until=lambda: A.state in (State.DISCONNECTED, State.LISTENING))

    to_b, to_a = bytes(b_io.delivered), bytes(a_io.delivered)
    print(f"A->B {len(to_b)}/{len(msg_a + msg_a2)} bytes, "
          f"B->A {len(to_a)}/{len(msg_b + msg_b2 + msg_b3)} bytes")
    print(f"final states: A={A.state} B={B.state}; final SL={A._sl}")
    assert to_b == msg_a + msg_a2, "A->B payload not byte-exact"
    assert to_a == msg_b + msg_b2 + msg_b3, "B->A payload not byte-exact"
    assert A.state in (State.DISCONNECTED, State.LISTENING), "A not disconnected"
    assert B.state == State.LISTENING, "B not disconnected"
    print("ALL PASS")


if __name__ == "__main__":
    _selftest()
