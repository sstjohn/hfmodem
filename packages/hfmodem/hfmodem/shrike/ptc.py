# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""SCS PTC-IIIusb emulation: the host interface Winlink already knows how to drive.

shrike presents itself to host software as an **SCS PTC-IIIusb, firmware 4.1 /
BIOS 2.90** on a serial port. That model is the obvious one to be: it is the
PACTOR-3-capable member of the PTC family that every Winlink client of the era
supports, its manual is public, and its hostmode is the one Pat's PACTOR driver
targets. The version string is the manual's own banner (Display 10.1.1) and is
what ``%V`` reports; see unknowns.HOST_IDENT for what we could not confirm.

There are two layers here and they are separate on purpose:

  * a **terminal-mode command interpreter** -- the ``cmd:`` prompt. Every client
    configures the modem here (MYCALL, PTCHN, tone/mode settings) and then sends
    ``JHOST4`` to hand over to
  * **CRC hostmode** (shrike.hostmode), where one reserved channel -- the PTCHN
    channel, default 4 -- *is* the PACTOR link. Connect, disconnect, data both
    ways and link state all ride that channel; channel 254 carries the status
    bytes and 255 is the extended poll.

Nothing in here knows how a waveform is made and nothing in ARQ knows about
serial ports: PtcHost implements :class:`shrike.arq.ArqIO` and hands the FSM's
PHY-bound calls to an injected peer -- :class:`onair.RadioTx` on a radio, and
:class:`SimPeer`, a stand-in station on an ideal channel, without one.

WHICH PROTOCOL a burst is rendered in is decided here and nowhere else. A link
opens in PACTOR-1 because a connect burst is PACTOR-1, and `upgrade` moves it to
the best of `UPGRADE_TARGETS` the peer has not contradicted -- or to the one the
peer asked for, when a grant arrives and `_take_grant` is armed to take it.
Neither the ARQ FSM above nor the peer below knows the answer: the FSM says what
it wants sent and the peer renders whatever it is handed.

Driven by a real client, not just by its own test: Pat 1.0.0 (whose PACTOR
driver is harenber/ptc-go) opened this on a pty, ran its whole init script,
switched to CRC hostmode, connected, exchanged a Winlink B2F handshake in both
directions and disconnected cleanly. See unknowns.HOST_* for what that run did
not touch.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from . import hostmode, pactor1, placement, spec
from .arq import (CS_ACK, CS_BREAKIN, CS_REQUEST, IRS, ISS, LADDERS,
                  P1_SPEED_LEVEL, ArqIO, PactorArq, State)
from .spec import Protocol

FIRMWARE = "4.1"
BIOS = "2.90"
BANNER = (
    "\r\n****************************************\r\n"
    "*    SCS PTC-IIIusb Multimode Controller                 *\r\n"
    f"*    Firmware     Version V.{FIRMWARE}      Level 3         *\r\n"
    f"*    BIOS Version {BIOS}                               *\r\n"
    "*    (C) 1994-2012 SCS GmbH - Germany                *\r\n"
    "****************************************\r\n"
)
PROMPT = "\r\ncmd: "
WAIT_SECONDS = 10.0        # the %W WAIT state always times out [10.4.37]
TRX_CHANNEL = 253          # transparent channel to the transceiver port [10.7]
BUSY_HOLD_S = 3.5          # "it remains so for at least 3.5 seconds" [6.90]

# PACTOR-1 does not have a NAK codeword, and looking for one is what made every
# answer shrike sent equivalent.
#
# The protocol description settles it: "CS1..3 have the same function as their
# AMTOR counterparts; CS4 serves as the speed change control." In AMTOR/SITOR-A
# the receiving station acknowledges by ALTERNATING between two control signals --
# the acknowledgement is the TOGGLE, not the codeword. Repeating the previous one
# is how a retransmission is requested. A constant CS2, which is what shrike sent
# for a whole session, reads as "send that again", forever, which is exactly what
# a real gateway did in response to it.
#
# So the mapping from the FSM's logical intent is stateful, and lives with the
# link rather than in a lookup table.  (PACTOR-1 protocol description §3.)


# --------------------------------------------------------------------------- #
# Which protocols a link may be run in
# --------------------------------------------------------------------------- #

TRANSMITTABLE: frozenset[Protocol] = frozenset(
    {Protocol.PACTOR1, Protocol.PACTOR2, Protocol.PACTOR3})
"""Protocols shrike will key. Every other one is receive-only or absent.

PACTOR-2 IS HERE AS OF 2026-09-02, and the condition this docstring set for it is
the one that was met. The waveform was never what stood in the way:
`pactor2.data_burst` renders marker and data as one phase walk, and an
independent monitor prints the payload of a real link's 32 bursts re-rendered
field for field, at speed levels 1 through 3 (pactor2.md §7.2, §7.12). What was
missing was the CONTROL SIGNALS -- an ARQ link is codewords in both directions
on every cycle, and a station that can send data bursts and cannot acknowledge
is not one a peer holds a link with.

`pactor2.control_signal` and `p2rx.control_signal_at` are that renderer and that
reader, AND THEY ARE A HYPOTHESIS. Nothing in the corpus carries a PACTOR-2
control signal at all, so the keying is PACTOR-3's -- the same six codewords,
measured on a stranger's tape -- put on PACTOR-2's two carriers at the packet's
own length plus the same 70 ms turnaround. What corroborates it is our own loop
and nothing else: `tests/shrike/test_p2link.py` reads 12 of 12 fields and 12 of
12 codewords off one raster, and an independent monitor cannot grade a reverse
channel because a monitor never answers. What that monitor DID settle is the
other half -- a peer keying in the answer slot costs our packets no frame and no
level, at all three levels, whether the codewords sit at 0.880 s, at 0.890, or
are not there at all.

So a PACTOR-2 link is offered where the peer leads into it and behind a flag
where it does not (`UPGRADE_TARGETS`, `PtcHost.offer_pactor2`), and the bound
that is honest to state is this one: our packets are read from outside, our
codewords are not. Speed level 4 stays out on its own account -- it renders, but
its 16-DPSK cells do not survive even our own front end, and `arq.P2_LADDER` tops
out at 3 for that reason."""

UPGRADE_TARGETS: tuple[Protocol, ...] = (Protocol.PACTOR3,)
"""What a PACTOR-1 link may OFFER to upgrade into, uninvited, best first.

Best first because the choice is usually made by trying: `PtcHost.upgrade` takes
the head of this list that the link has not already been contradicted on, and a
peer answering in PACTOR-1 is the contradiction. With one entry that is "try
PACTOR-3 once, and stay in PACTOR-1 if the peer cannot follow"; with two it walks
the list.

Usually, because a commercial gateway will say so outright: `0x59A` in the answer
slot is the grant, and a station that has one does not have to guess which
protocol the peer can read. `PtcHost._take_grant` is that path, and it is taken
by default: the peer's grant is a bound on how far the link goes, and the only
honest answer to a bound the far end has just widened is to use it.

PACTOR-2 IS NOT IN THIS LIST AND IS TRANSMITTABLE, which is the one place the two
facts come apart. Following a peer into PACTOR-2 answers a station that has
already keyed the waveform; offering one uninvited keys a codeword nothing
outside this package has ever graded, at a station that has said nothing about
it. The
corpus is the reason and it is a null: not one PACTOR-1-to-PACTOR-2 transition on
tape, and both graded third-party completions go 1 -> 3 in a single step. So the
offer is `PtcHost.offer_pactor2`, default off, and it is appended AFTER
PACTOR-3 -- tried only once this link has been contradicted on PACTOR-3, never
in front of it.

Of 558 Winlink PACTOR channels, 473 offer PACTOR-3 and 85 more can be improved on
only with PACTOR-2 -- 83 of those offering PACTOR-2 and nothing else above
PACTOR-1. That is what the flag is for, and it is a third of the network."""


GRANT_ENTRY_SL = 1
"""The speed level the PACTOR-3 phase opens at when the peer GRANTED it.

Not `ArqConfig.entry_sl`, and the difference is what the grant buys. Level 1 is
the case-0 packet on tones 5 and 12 -- the two tones every speed level shares --
so it is the one packet a receiver that has been tracking a PACTOR-1 raster can
acquire, and that is what both references key: `p3rx.levels_present` scores the
first PACTOR-3 segment of `rf-corpus/regress/fixtures/pos_pactor1_local.wav` and
of `rf-corpus/PIII_Complete_1.wav` at level 1, channels 5 and 12 lit and the
PACTOR-1 FSK pair dark. The level the traffic then runs at is `ArqConfig.traffic_sl`,
taken once the entry packet has been acknowledged -- which is the point at which
the peer has demonstrably acquired the waveform.

An UNINVITED upgrade still opens at `entry_sl`, because there is nobody known to
be listening for the entry packet and level 3 is the level an independent monitor
acquires cold."""


P1_HISPEED_RETRIES = 4
"""Repeat requests at 200 Bd before a PACTOR-1 link drops itself back to 100.

`PACTOR_RETRY_HISPEED`, off hf-pactor's `hfkernel/fsk/pactor.c` -- the budget
every 200 Bd state there is entered with, against the 30 of `params.pactor.retry`
at 100 Bd. THE REFERENCE SPENDS IT AT THE IRS, whose exhaustion is answered with
the CS4 that orders the sender down; its ISS never gears itself down at all, and
on both links this figure was wanted for we were the ISS. CS4 is not ours to send
-- M.1798 sec 4 gives the sending station no gear command -- but the rate we key
is, so the same figure bounds the same fact from the other end.
docs/protocols/pactor/pactor1-timing.md sec 4 carries the split.

It bounds the one thing the ordinary retry budget cannot see: a peer that answers
every cycle, at zero bit errors, asking again for a packet it is not decoding."""


# WA8DED link states as reported in field f of the L command [WA8DED guide].
LINK_DISCONNECTED, LINK_SETUP, LINK_DISCONNECT_REQ, LINK_INFO = 0, 1, 3, 4

# Terminal commands whose canonical spelling we know; the capitals are the
# minimum abbreviation the PTC accepts, exactly as the manual writes them.
_CANONICAL = [
    "MYcall", "PTCHn", "MAXError", "REMote", "CHOBell", "TONes", "MARk", "SPAce",
    "CWid", "CONType", "CONIntegrity", "Connect", "Disconnect", "DD",
    "MODe", "DAte", "TIme",
    "STatus", "LIsten", "MAXSum",
    "SERBaud", "TXdelay", "PDuplex", "HCr", "ADDlf", "TERm", "BAud", "FSKAmpl",
    "PSKAmpl", "MYSelcall", "MYLevel", "MAXErr", "CHeck", "CMsg", "Quit",
    "RESTart", "VERsion", "JHOST", "PT", "PR", "PAC",
]
_ABBREV = {c.upper(): "".join(ch for ch in c if ch.isupper()) for c in _CANONICAL}


def _split_command(text: str) -> tuple[str, str]:
    """A hostmode command word and its argument.

    Hostmode commands are one letter, or `@X`/`%X`, and the argument follows
    with or without a space: BPQ32 sends `I<call>` and `%W0`, ptc-go sends
    `C <call>` and `L`.
    """
    head = 2 if text[:1] in "@%" or text[:2].upper() == "DD" else 1
    return text[:head].upper(), text[head:].strip()


@dataclass
class _Channel:
    """One hostmode channel's outbound queues."""
    link: list[str] = field(default_factory=list)      # code 3 events, oldest first
    rx: bytearray = field(default_factory=bytearray)   # code 7 payload

    def pending(self) -> bool:
        return bool(self.link or self.rx)


class PtcHost(ArqIO):
    """A PTC-IIIusb on a wire: terminal menu, CRC hostmode, and a PACTOR link."""

    wait_until: Optional[float] = None
    """Deadline of the %W WAIT state [10.4.37], or None outside it.

    %W0's "1" answer promises the scanner that no call will be taken while it
    retunes, and the promise always lapses after ten seconds whether or not %W1
    releases it."""

    busy_until: Optional[float] = None
    """How long the channel goes on counting as busy after a decoded signal [6.90]."""

    _wait_armed = False

    def __init__(self, peer=None, mycall: str = "SCSPTC", ptchn: int = 4):
        self.peer = peer
        self.settings: dict[str, str] = {"MYCALL": mycall.upper(), "PTCHN": str(ptchn)}
        # The announcement's callsign case. Both completions on tape send it
        # lowercase (`1dl6maa`, `1w4dna`); every arm of ours sent it upper.
        self.announce_lower = True
        self.hostmode = False
        self.listening = False
        self.autostatus = False
        self.arq = PactorArq(self)
        self.settings["CONTYPE"] = str(self.arq.cfg.contype)
        self.settings["CONINTEGRITY"] = str(self.arq.cfg.conintegrity)
        # Which bit rate this PACTOR-1 link runs at, decided by the codeword the
        # called station answers with and NOT by us. CS1 means it read our 200 Bd
        # redundancy section cleanly and the link starts at 200; CS4 means it did
        # not and the link starts at 100. Answering a CS1 at 100 Bd transmits into
        # a receiver listening at 200, and the link is then deaf with nothing on
        # the air to break the deadlock. See docs/protocols/pactor/pactor1-timing.md.
        #
        # 100 until told otherwise: a link that never got an answer never got a
        # speed either, and 100 is the slower, safer default. Set before the
        # protocol below, which reads it for the field size.
        self.p1_baud = 100
        # The link starts in PACTOR-1 -- that is what a connect burst is -- and
        # stays there until it is upgraded. Setting it also sets the field size:
        # PACTOR-1's 100 Bd field is 8 bytes, 7 in a changeover packet where the
        # CS3 head takes the other one, and PACTOR-3's comes from the speed level.
        self.protocol = Protocol.PACTOR1
        # Upgrade targets this link has tried and been contradicted on. Per link,
        # not per station: a peer that cannot follow says so by answering in
        # PACTOR-1, and that says nothing about the next station we call.
        self._ruled_out: set[Protocol] = set()
        # Two different memories of the reverse channel, because the protocol
        # asks two different questions of it. `_last_rx_cs` is the alternation --
        # the last acknowledgement in the alternation. A CS4 speed-up consumes
        # its next position even though that CS1/CS2 was replaced by CS4.
        # `_prev_rx_cs` is the codeword immediately before this one, for the
        # rule that needs it: "CS4 in Folge (ohne zwischenzeitliches richtiges CS)
        # werden als 'Request' interpretiert".
        self._last_rx_cs = None
        self._prev_rx_cs = None
        self._p1_speedup_pending = False
        # Header phase is per sending turn: setup #1/headAA, post-BK #1/head55.
        self._p1_header_inverted = False
        # A block whose acknowledgement is CS1 outright rather than by the
        # alternation: the one a CS4 REJECT re-chunks down to 100 Bd, which the
        # description gives its own rule -- "Bestaetigung erfolgt mittels CS1
        # oder CS3". Armed in `_logical_cs`, spent on the codeword that takes it.
        self._p1_first_block = False
        # Whether a CS4 read at 100 Bd is the speed offer at all. It answers "jedem
        # richtig empfangenen 100-Bd-Paket", so a link that has just geared down to
        # 100 with nothing acknowledged since is asking for that block again.
        self._p1_speedup_armed = True
        self._hispeed_tries = 0
        self.stay_in_pactor1 = False
        """Hold the link in PACTOR-1, whatever it could be upgraded to.

        Two reasons, and neither is a debug switch. 83 of 558 Winlink PACTOR
        channels are PACTOR-2 only, and every upgrade offered to one of those is
        four cycles spent transmitting a waveform nobody is reading. And an
        upgrade fires on the first acknowledged packet with traffic behind it --
        so a session run to find out whether the peer acknowledges at all
        changes protocol on exactly the cycle it is trying to observe. One
        variable at a time is the whole of the diagnosis.
        """
        self.stay_in_pactor3 = False
        """Take the upgrade at the first acknowledgement and hold the link there.

        The mirror of `stay_in_pactor1` and it exists for the second of that
        flag's two reasons: an upgrade that happens on its own, on whichever
        cycle the buffer happens to be non-empty, and is taken back by whichever
        of two rules the peer trips first, is not a variable an operator can hold
        still for a rig test. Under this the offer is taken with or without
        traffic behind it and neither rule can undo it, so what the session
        measures is what a gateway does with our PACTOR-3.

        A BREAK-IN IS STILL FOLLOWED. The peer seizing the channel says nothing
        about what it can decode -- there is no PACTOR-3 changeover packet to
        seize it with, so ours go out in PACTOR-1 too -- and a station that will
        not follow one holds a link nobody else is on. The upgrade is offered
        again on the next acknowledged packet, which under this flag is the next
        one there is.
        """
        self.no_p3_fallback = False
        """After entering P3, retain it through retries and teardown.

        Unlike stay_in_pactor3, this does not change when an upgrade is offered.
        Lower-protocol replies cannot acknowledge P3 traffic or force a return
        to P1. The driver's hold limit and ARQ teardown still end the contact.
        """
        self._p3_fallback_reported = False
        self.p3_changeover_cs5 = False
        """Opt-in reference-CS5 reply experiment for CRC-valid P3 changeovers."""
        self.p3_qrt_confirm = False
        """Opt-in post-QRT marker experiment; wait for its separate ACK reply."""
        self.p3_changeover_p1_cs = False
        """Opt-in: answer a PACTOR-3 changeover packet with a PACTOR-1 codeword.

        The link stays at PACTOR-3 and so does the alternation -- only the
        renderer changes, and only for the CS1/CS2 that answers the changeover.
        Wider than that there is nothing to render: PACTOR-1 has four codewords,
        its CS4 is a speed-DOWN to 100 Bd where PACTOR-3's demands a speed-up,
        and it has no CS5 or CS6 at all, so a gear command keyed in PACTOR-1
        would say something other than what the FSM asked for."""
        self._p3_changeover_answer = False
        self.p1_6a9 = False  # Alternate P1 response experiment, off by default.
        self.p1_act_on_grant = True
        """Take the peer's `0x59A` as the command to key PACTOR-3 that it is.

        ON, because a link goes as far as both ends can carry it. Two bounds set
        that distance and nothing else does: what this station can key, and what
        the peer will have. `0x59A` in the answer slot is the far end naming the
        second bound out loud -- it can read PACTOR-3 and it is asking for the
        entry packet -- so a station that decodes that word and stays in
        PACTOR-1 is declining a capability the peer just offered it, at a
        gateway that may serve nothing else (WS8EOC: "only accept Pactor levels
        above 2 - Disconnecting"). Refusing is the thing that needs saying out
        loud, so it is the flag; taking it is the posture, so it is the default.

        The first bound is `arq.ENTRY_RUNGS_GROUNDED` and the protocols missing
        from `TRANSMITTABLE`: what we cannot key reliably is written down where
        it is used rather than left to a docstring, and it is the only thing
        that stops us short of what the peer allowed.

        Off, the grant is named in the log and nothing else happens -- which is
        how every PACTOR-3 this station has ever transmitted went out uninvited,
        and what "the peer refuses our PACTOR-3" was measured against: peers
        with no reason to be listening for one. See `_take_grant`, and
        `p1_grant_only` for the other half of what the old opt-in flag meant.
        """
        self.offer_pactor2 = False
        """Offer PACTOR-2 uninvited, once PACTOR-3 is ruled out for this link.

        OFF, and the reason is a null in the corpus rather than a doubt about the
        waveform. Our PACTOR-2 packets are read from outside at three speed
        levels; our PACTOR-2 CODEWORDS are graded by nothing but our own loop
        (`TRANSMITTABLE`), and no recording anywhere holds a PACTOR-1 link
        transitioning into PACTOR-2 -- both graded third-party completions go
        1 -> 3 in one step. So there is no measurement saying what a station
        expects to see when it is offered this, and the offer is what an operator
        turns on for an arm rather than what a session does on its own.

        What FOLLOWING a peer into PACTOR-2 costs is a different question and it
        is answered the other way (`_follow_peer`, default on): there the peer
        has already keyed the waveform, so the ladder is one it named itself.

        On, `UPGRADE_TARGETS` gains PACTOR-2 behind PACTOR-3 and nothing else
        changes -- the rung is tried only where PACTOR-3 has been ruled out for
        this link, which is a peer that answered our PACTOR-3 in PACTOR-1. What
        it keys is `arq.P2_LADDER.entry_sl`: a speed-level-1 short packet,
        DBPSK on both carriers, with the nine-pulse frame marker in front of it
        naming its own level and length.
        """
        self.p1_grant_only = False
        """Refuse the UNINVITED upgrade, so a grant is the only door into PACTOR-3.

        Not a posture -- an experiment control. An arm that asks what a gateway
        does with an INVITED PACTOR-3 has to be the only thing that can put one
        on the air, and an uninvited upgrade off the next acknowledged packet is
        a second variable which arrives first. `stay_in_pactor1` cannot do this:
        it declines the grant along with everything else, and it is also a
        promise the channel sense reads as 1200-1800 Hz (`core.occupied`), which
        no grant may break. So the two contradict each other, `_take_grant`
        declines under `stay_in_pactor1`, and the parser refuses the pair.
        """
        # Once per link, and armed for `upgrade` to spend on the next
        # acknowledgement -- which is the one the grant itself carries.
        self._grant_taken = False
        self._grant_pending = False
        # An UNINVITED PACTOR-3 phase is open and nothing has come back from it.
        # The granted half of the same state is `arq.entry_pending`, which stays
        # off this door: that flag holds the buffer back and routes the render
        # through `send_entry_packet`, and an uninvited upgrade keys its traffic
        # instead -- 5aa3f1bc, because the peer's first look at a waveform it
        # must acquire cold should be a full field rather than fill. The packet
        # asks the same question either way, so `_entry_answered` takes either.
        self._uninvited_entry = False
        self._burst_at = None
        self._burst_sample = None
        self.log_lines: list[str] = []

        # The station-side application this link's payload belongs to, when the
        # station is its own client rather than serving one — winlink's
        # MailClient is the shape: `link_up()` on connect, `on_link_data(blob)`
        # per delivery, `done` when its exchange is over. With an app attached,
        # delivered payload goes to it instead of the hostmode channel buffer:
        # a link has one reader, and a station speaking its own mail has no
        # hostmode client polling the channel.
        self.app = None
        self._modem = hostmode.Modem(self._dispatch)
        self._line = bytearray()
        self._channels: dict[int, _Channel] = {}
        self._txbuf = 0            # bytes accepted from the host, not yet acked
        self._setup_bytes = b""    # the PACTOR-1 announcement, while it is owed
        # PAYLOAD, EACH WAY, AND ONLY PAYLOAD. Both move on an acknowledgement
        # that carried a field: idle fill decodes to nothing and never reaches
        # `deliver`, and an empty packet's ack leaves `_txbuf` where it was. So a
        # pair that has not changed across a cycle is a link on which nothing was
        # said, whatever went out over it -- which is what `onair._HoldBudget`
        # asks them, and why a codeword cannot answer for them.
        self.phase_power = None
        """The arm's RF-power plan (`onair._PhasePower`), or None for nobody's.

        A PACTOR-1 packet and a PACTOR-3 entry cannot be put on the air at two
        powers from the audio side of a rig whose ALC reads their crest factors
        differently, so an arm that wants them at two powers hands this layer
        something that writes the rig's own level -- and this layer is where it
        belongs, because the transitions are the link's rather than the
        session loop's. Three of them call in (`_phase_power`); nothing else
        does, and with none of it here the rig is never written to at all.
        """
        self.sent_total = 0
        self._sent_base = 0
        self.rcvd_total = 0
        self._laststatus: Optional[bytes] = None
        self._link_up = False
        self._disconnect_asked = False
        # Per-channel MYCALL [10.4.5]; channel 0's is the one a channel falls
        # back to when its link ends. BPQ32's SCS driver sets one per stream.
        self._callsigns: dict[int, str] = {}
        if peer is not None:
            peer.attach(self)

    # -- which protocol this link is in -----------------------------------
    @property
    def protocol(self) -> Protocol:
        """The PACTOR level this link's packets and control signals are in.

        The one place the answer lives, because three things have to agree on it
        and they used to be set apart: which renderer a burst goes to, which
        demodulators the receiver runs, and how many bytes a packet's field holds.
        A PACTOR-3 field size on a PACTOR-1 link chopped shrike's own callsign
        announcement to `1W9SS`.
        """
        return self._protocol

    @protocol.setter
    def protocol(self, p: Protocol) -> None:
        self._protocol = p
        if p in LADDERS:
            self.arq.ladder = LADDERS[p]
        if p is Protocol.PACTOR1:
            self._p1_field(self.p1_baud)
        else:
            self._p1_speedup_pending = False
            self._hispeed_tries = 0
            # A ladder's field size is the speed level's, so the override comes
            # off rather than being set to something.
            self.arq.payload_bytes_override = None
            self.arq.breakin_bytes_override = None

    def _phase_power(self, protocol: Protocol, why: str) -> None:
        """Tell the arm's power plan which protocol the link is in now.

        THREE TRANSITIONS AND NOT THE SETTER, which is the seam it looks like it
        should be. `protocol` is assigned from `__init__`, from `_end_link`'s
        reset and from `upgrade` -- and `upgrade` runs on the acknowledgement
        path, which chains straight into `arq._start_next_packet` and keys in
        the same cycle. A CAT write there would be a write in the pre-key
        window, which is the one place this must never be. So the calls are made
        where the transition is READ instead: `_take_grant`, a cycle's
        turnaround in front of the entry packet, and the two places a link comes
        back to PACTOR-1 with nothing keyed behind them.
        """
        if self.phase_power is not None:
            self.phase_power.select(protocol, why)

    @property
    def breakin_due(self) -> bool:
        """Whether this cycle's tick may key a changeover packet.

        Asked BEFORE the listen window, because the changeover is the one burst
        placed against the peer's transmission rather than on our own boundary
        (`onair.RadioTx._breakin_key`) and the window in front of it has to close
        early enough to reach that instant. What licenses the packet is decided
        inside that same window -- the CS3 is only sent behind a packet that
        decoded -- so the answer here is the FSM's standing intent rather than
        its arming, and it errs towards a window that closes early instead of a
        key that cannot be made. A cycle that then keys a control signal keys it
        on the boundary, unmoved.

        BOTH PROTOCOLS, and the PACTOR-3 half is the same measurement as the
        PACTOR-1 one. The reference peer's CS3 answering DL6MAA's 82-symbol
        entry lands at +891.2 ms from that packet's phase reference -- 71.2 ms
        after its audio ends, the lead `onair.BREAKIN_LEAD_S` carries -- so the
        instant is a property of the turnaround rather than of the waveform. It
        was gated on PACTOR-1 only while the PACTOR-3 changeover packet did not
        exist to place; `placement.CHANGEOVER` is read off both directions of a
        real session now, and `breakin_now` says so.
        """
        return self.arq.taking_link

    def upgrade(self, payload_waiting: bool) -> bool:
        """Leave PACTOR-1 for the best protocol not already ruled out. [arq.ArqIO]

        THE TARGET IS A CHOICE, not a constant, and unless the peer names it it has
        to be discovered rather than looked up. PACTOR's connect answer is one bit
        and it is a rate, and there is no codeword or state meaning "I cannot" --
        so absent a grant the only way to learn what a peer can do is to transmit
        something and see what comes back. `UPGRADE_TARGETS` is the order to try
        them in, `_ruled_out` is what this link has already been contradicted on,
        and the contradicting is done by `_follow_peer` when the peer answers in
        PACTOR-1 and by `fall_back` when it stops answering at all.

        A GRANT IS THE OTHER DOOR. `_take_grant` arms `_grant_pending` on the
        peer's `0x59A`, which arrives as an acknowledgement, so the offer below
        is taken on that acknowledgement whether or not there is traffic behind
        it -- the peer asked for the waveform, and the slot it asked for it in is
        the next one. It opens at `GRANT_ENTRY_SL` rather than `entry_sl`, and
        under `p1_grant_only` it is the ONLY door: the offer below is refused
        so the arm has one variable and does not need `stay_in_pactor1`, which
        would decline the grant along with everything else.

        AND AN EMPTY BUFFER DOES NOT DECLINE IT. The payload guard below belongs
        to the uninvited offer: it asks whether the upgrade is worth a cycle of
        the peer's acquisition, and a grant has already answered that -- the peer
        spent its own answer slot asking for the waveform. A granted link with
        nothing to carry keys an IDLE packet and waits, and the IRS takes the
        channel off it with the CS3 the reference peer answers an entry packet
        with (pactor3.md, "the answer is a break-in, and a break-in is an
        answer"), which is how a caller waiting on a greeting gets one. It is
        the ordinary case: `arq.PactorArq.on_rx_grant` carries nothing out of the
        PACTOR-1 phase, so a granted link keys idle packets until the host
        writes, which is what the reference caller does after its own entry.

        The FSM decides WHEN, on every acknowledgement of a data packet
        (`arq.PactorArq._on_ack`), so a link that has fallen back gets another
        offer as soon as PACTOR-1 has again proved it carries data. This is a
        no-op unless the link is in PACTOR-1 with a target left to try, and it
        answers TRUE only when it actually moved the link -- the FSM arms its
        silence timer on that, because an unanswered change has to be undone.

        NOT ONTO AN EMPTY BUFFER. `payload_waiting` is what the FSM has left to
        send at the moment of the offer, and an upgrade taken without it hands
        the peer an IDLE PACKET as its first PACTOR-3 frame -- 59 bytes of fill
        to acquire a new waveform on, carrying nothing either end can use. The
        upgrade is worth a cycle of the peer's acquisition only when there is
        traffic behind it to be carried at four times the rate, so it waits for
        the traffic; a link that never has any is a link PACTOR-1 was already
        fast enough for. `stay_in_pactor3` is the one thing that overrides it,
        because a rig test wants the waveform on the air whether or not there is
        traffic to put under it.
        """
        # A grant armed before an operator stop must not be consumed by the
        # final ACK. Discard only the offer, never terminal intent or its timer.
        if self.arq.terminal_pending:
            self._grant_pending = False
            return False
        # `_on_ack` is the one caller, so this is the acknowledgement half of
        # the entry packet's answer. `on_rx_event` carries the other half.
        self._entry_answered()
        if self.protocol is not Protocol.PACTOR1 or self.stay_in_pactor1:
            return False
        granted, self._grant_pending = self._grant_pending, False
        if not granted:
            if self.p1_grant_only:
                return False
            if not payload_waiting and not self.stay_in_pactor3:
                return False
        for target in self._upgrade_targets():
            if target not in self._ruled_out:
                self.protocol = target
                self.arq.speed_level = GRANT_ENTRY_SL if granted \
                    else self.arq.entry_level
                self.arq.entry_pending = granted and target is Protocol.PACTOR3
                self._uninvited_entry = not granted \
                    and target is Protocol.PACTOR3
                why = "on the peer's grant" if granted else "uninvited"
                self.log(f"link upgraded to {target} at "
                         f"SL{self.arq.speed_level} {why}")
                return True
        return False

    def _upgrade_targets(self) -> tuple[Protocol, ...]:
        """`UPGRADE_TARGETS`, with the PACTOR-2 rung appended where it is armed.

        AFTER, never before: PACTOR-3 is contradicted by a peer answering in
        PACTOR-1, and that contradiction is what makes the second rung worth
        keying at all. `offer_pactor2` is where the null this stands on is
        written down."""
        return UPGRADE_TARGETS + ((Protocol.PACTOR2,) if self.offer_pactor2
                                  else ())

    def _entry_answered(self) -> None:
        """The peer read the entry packet: run the traffic at `traffic_sl`.

        The entry packet asks one question -- can you acquire this waveform? --
        and until something comes back the link holds `GRANT_ENTRY_SL`, which is
        the level a station tracking a PACTOR-1 raster can find rather than a
        level to carry traffic at: five bytes a cycle.

        BOTH DOORS ASK IT. This was gated on `arq.entry_pending` alone, which
        `upgrade` sets only on a grant, so the milestone -- and the `entry read`
        column the console scores off its line -- could not fire on an uninvited
        upgrade at all. All 27 uninvited entries on record scored zero there by
        construction. `_uninvited_entry` is the other door's half of the same
        state, and it is a second flag rather than a widening of the first
        because `entry_pending` also decides what gets RENDERED: the uninvited
        door opens at `ArqConfig.entry_sl` with the traffic in the field, and
        what comes back answers it just as much.

        AND THE ANSWER IS NOT ALWAYS AN ACKNOWLEDGEMENT. In both reference
        recordings the one speed-level-1 packet on the air is answered CS3
        BREAK-IN, in the turnaround the acknowledgement would have come back in
        -- the IRS reads the entry packet, and what it has to say is its own
        greeting rather than a codeword. A break-in settles the packet in flight
        through `arq.PactorArq._yield_link`, which never reaches `_on_ack`, so a
        spend that lived only on the acknowledgement path was never made against
        the answer the recordings actually carry, and the link held speed level 1
        for the whole of its PACTOR-3 phase.

        IT IS NOT ALWAYS A BREAK-IN EITHER, and that is a property of the
        receiver rather than of the protocol. The break-in's twenty-symbol head
        has to be aimed at; the data packet a cycle behind it is a whole cycle
        wide and comes out of the blind pass. `rxfront.decode_events` over
        `PIII_Complete_1` finds the second and not the first. So any PACTOR-3
        packet the peer sends while the entry packet stands spends this, and
        `arq.PactorArq.on_rx_packet` yields the link on the same frame.
        """
        if self.protocol is Protocol.PACTOR1 \
                or not (self.arq.entry_pending or self._uninvited_entry):
            return
        self._uninvited_entry = False
        self.arq.entry_pending = False
        self.arq.speed_level = self.arq.traffic_level
        self.log(f"the peer answered the entry packet -> SL{self.arq.speed_level}")

    def fall_back(self, why: str, *, rule_out: bool = True) -> bool | None:
        """The upgrade window ran out: go back to what was carrying data.
        [arq.ArqIO, after `arq.UPGRADE_SILENCE_CYCLES` cycles]

        `_follow_peer` handles the peer that CONTRADICTS the upgrade, which it
        may do only once the upgrade has been answered at all. This is every
        other case: a station that could not follow keeps answering in a protocol
        our own upgraded receiver has largely stopped listening for, so what
        reaches this layer is nothing at all -- and a PACTOR-1 frame that does
        reach it inside the window is held back by `_stale_pactor1`, so the count
        runs either way.

        `why` IS THE CALLER'S FINDING AND IS NOT RESTATED. This end used to
        write its own verdict here -- `the peer never took the entry packet` --
        on top of whatever `_on_nak`'s window had concluded, and it names the
        far end in every one of the three states the receiver can be in. On
        2026-08-26 it printed while WS8EOC was sending the PACTOR-3 grant at zero
        bit errors, and went on doing so for twenty-one more cycles; that
        sentence is what a whole investigation then read as a refusal. Absence is
        evidence on this link in a way it is almost nowhere else -- the IRS owes
        the ISS a control signal every cycle -- but it is only evidence about the
        peer once our own carrier and our own readers are out of the way, and
        `arq._upgrade_window_ended` is what says which of those it was.

        `rule_out` IS WHETHER THE FINDING IS ABOUT THE STATION OR ABOUT THE
        CHANNEL. Silence across the upgrade window is a verdict on what the peer
        can decode and the target is closed for the link. A repeat train
        (`arq.UNREAD_RUNG_REPEATS`) is not: the peer is answering every cycle,
        and what the count measures is that our field is not arriving. So that
        caller declines the verdict, the target stays open, and the link can
        climb again on the next grant.
        """
        if self.protocol is not Protocol.PACTOR1:
            return self._fall_back_to_pactor1(why, rule_out=rule_out)

    def _fall_back_to_pactor1(self, why: str, *, rule_out: bool = True) -> bool | None:
        """Run the link in PACTOR-1 again, and stop offering what it was in.

        Ruled out for the rest of this LINK, which is what keeps it from
        flapping: the FSM offers an upgrade on every acknowledgement, so without
        this the very next one would put the link straight back into the protocol
        that just failed. `upgrade` then walks on to the next target, or leaves
        the link in PACTOR-1 when there is none.

        Re-chunking is the other half. The packet in flight was built to the old
        protocol's field -- 59 bytes at speed level 3 against 8 at 100 Bd -- and
        the renderer truncates silently, so it goes back into the buffer to be
        split rather than out as it stands.

        `rule_out` is also what separates a verdict on the peer's CAPABILITY from
        a changeover, which is why `stay_in_pactor3` is refused here rather than
        at the two call sites: the flag suspends the two rules that read a peer
        as unable to follow, and leaves the break-in that reads it as taking the
        channel alone.
        """
        if self.no_p3_fallback and self.protocol is Protocol.PACTOR3:
            if not self._p3_fallback_reported:
                self.log(f"{why}; --no-p3-fallback retains PACTOR-3 "
                         "until the link or hold budget ends")
                self._p3_fallback_reported = True
            return False
        if rule_out and self.stay_in_pactor3:
            self.log(f"{why}, but the link is held in {self.protocol}")
            return
        if rule_out:
            self._ruled_out.add(self.protocol)
        else:
            # AND THE GRANT COMES BACK WITH IT. `_grant_taken` is a latch, so a
            # retreat that is not a verdict would still forfeit the waveform
            # for the rest of the contact -- the peer may key `0x59A` every
            # cycle and this station would never act on one again. A fallback
            # that declines to rule the target out is saying the evidence was
            # about the channel; the peer's next grant is new evidence.
            self._grant_taken = False
        self.protocol = Protocol.PACTOR1
        self.arq.cycle_long = False          # PACTOR-1 has no long cycle
        # The entry packet holds the buffer back, and a link that has stopped
        # trying to enter has to let it go again or it keys empty PACTOR-1
        # packets at a peer that can read them. The entry packet itself does not
        # come down with it: an empty field is not information, and nothing
        # acknowledged it -- that is why we are here -- so it goes back rather
        # than out again at 8 bytes a cycle with nothing in it.
        self._uninvited_entry = False
        if self.arq.entry_pending:
            self.arq.requeue_inflight()
            self.arq.entry_pending = False
        self.arq.rechunk_inflight()
        self._phase_power(Protocol.PACTOR1, "the link fell back")
        self.log(f"{why} -> link falls back to PACTOR-1")

    def _follow_peer(self, protocol: Optional[str],
                     breakin: bool = False) -> None:
        """Transmit whatever the peer is transmitting.

        CAPABILITY IS WHAT A STATION TRANSMITS. PACTOR has no capability field, no
        negotiation and no refusal, so there is nothing to ask a peer and nothing
        it can tell us -- but a decoded frame is a statement about the far end that
        cannot be forged, and it is the whole of what we can know. So the link runs
        in the protocol the peer was last heard using, in both directions:

          * a station that decodes a higher-level frame follows the peer up, which
            is how the answering station -- which never sends a data packet and so
            never reaches the ISS's decision -- ever leaves PACTOR-1 at all;
          * a station that decodes a PACTOR-1 frame on a link it believes is
            higher goes back, AND RULES THAT TARGET OUT for the rest of the link.
            That is what makes the ISS's unilateral upgrade recoverable against a
            peer that cannot follow, and the next upgrade offer moves on to the
            next target. It may not act on the FIRST PACTOR-1 frame after an
            upgrade, though: a PACTOR-1-only station answers every cycle whether
            or not it decoded us, so that frame is older than the upgrade and is
            a verdict on nothing. `_stale_pactor1` is what it meets instead, and
            `arq.UPGRADE_SILENCE_CYCLES` is what ends the link's patience.

        A peer transmitting something we cannot transmit back is NOT followed.
        Adopting a protocol we have no renderer for would key nothing at all,
        which is worse than answering in PACTOR-1 -- the peer can still read that,
        and it is the one protocol every PACTOR station has.

        PACTOR-2 IS FOLLOWED AS OF 2026-09-02, and it is the branch this rule was
        written for. A peer leading into it has keyed the waveform, which is the
        strongest statement PACTOR has: it has said what it can read and it is
        waiting for an answer in it. Hearing that and answering in PACTOR-1 was
        this station declining a bound the far end had just named -- and it is a
        different question from OFFERING PACTOR-2 uninvited, which is off by
        default because no recording holds one (`offer_pactor2`). The frame's own
        marker names the level and the length, so `arq.P2_LADDER` is what the
        traffic is chunked to from the moment the follow happens.

        A CHANGEOVER PACKET IS THE ONE PACTOR-1 FRAME THAT RULES NOTHING OUT. A
        PACTOR-1 break-in says the peer is seizing the channel, not that it could
        not follow us -- a PACTOR-1-only station and one simply taking its turn
        key the same frame. The link still goes back, because that is what the
        peer is transmitting; it just gets to climb again on the next
        acknowledged packet. (Ours no longer drops to PACTOR-1 to break in --
        see `breakin_now` -- so this is a reading of the PEER and not a mirror of
        anything we do.)

        A speed-level-1 packet cannot drive any of this on its own -- see
        `on_rx_event`.
        """
        heard = None if protocol is None else Protocol(protocol)
        if heard is None or heard is self.protocol:
            return
        if self.no_p3_fallback and self.protocol is Protocol.PACTOR3:
            return
        if heard is Protocol.PACTOR1:
            self._fall_back_to_pactor1(
                "peer sent PACTOR-1" + ("" if not breakin else " to change over"),
                rule_out=not breakin)
        elif heard in TRANSMITTABLE:
            if self.arq.role == ISS and not breakin:
                # Stale news from a turnaround race, not a peer to follow: a
                # higher-protocol frame heard while WE hold the link is the
                # packet the peer had in flight before it decoded our break-in.
                # Following it up put this end in PACTOR-3 for exactly one
                # cycle, and the peer's next PACTOR-1 acknowledgement -- of our
                # own PACTOR-1 packet -- then read as a contradiction and ruled
                # PACTOR-3 out for the rest of the link. The upward follow is
                # how the ANSWERING station leaves PACTOR-1, and the answering
                # station is the IRS; the ISS leaves it through `upgrade`, on
                # its own acknowledged packet.
                return
            # A changeover explicitly hands us the receiving role. Follow
            # before ARQ yields and renders its ACK, including a late entry
            # response received after our local fallback to PACTOR-1.
            self.protocol = heard
            self.arq.speed_level = self.arq.entry_level
            self.log(f"peer is transmitting {heard} -> link follows")
        else:
            self.log(f"peer is transmitting {heard}, which shrike cannot "
                     f"transmit -- answering in {self.protocol}")

    def _stale_pactor1(self, ev) -> bool:
        """Keep P1 replies from acknowledging an unconfirmed higher-mode packet.

        Before this gate, the recorded uninvited upgrades keyed one SL3 packet
        and returned to P1 a cycle later on CS1. The IRS builds its reply from
        the preceding received packet: that first CS1 could acknowledge the P1
        packet ahead of the upgrade, while the new P3 packet was still awaiting
        acquisition. Following the P1 reply immediately ended every such offer;
        letting it acknowledge the new packet would retire unconfirmed bytes.

        Keep repeating the pending packet through the bounded ARQ window so a
        peer can combine successive receptions. Later P1 replies are not
        necessarily old, but cannot acknowledge a P3 packet either. A decoded
        higher-mode response confirms the transition. P1 break-in remains exempt:
        it takes the channel instead of acknowledging the entry.
        """
        if ev.protocol != Protocol.PACTOR1 or not self.arq.upgrade_unanswered:
            return False
        if ev.breakin if ev.kind == "packet" else ev.cs == pactor1.CS_CHANGEOVER:
            return False
        self.log("peer sent PACTOR-1 while the upgrade remains unconfirmed "
                 "-- keeping the P3 packet pending; not a P3 acknowledgement")
        return True

    def _take_grant(self, ev) -> None:
        """The peer commanded PACTOR-3: key the entry packet in the next slot.

        `0x59A` IN THE ANSWER SLOT IS THE GRANT, and it is an acknowledgement as
        much as a command -- it arrives where the codeword answering our
        announcement packet would, and the granted station's very next burst is
        PACTOR-3. Measured twice: grant at 12.443 s and PACTOR-3 tone energy at
        12.60 s in `rf-corpus/regress/fixtures/pos_pactor1_local.wav`, 4.412 s
        and 4.610 s in `rf-corpus/PIII_Complete_1.wav`. One turnaround, not one
        cycle, and neither station changes its slot timing to do it.

        SO IT GOES IN IN THE CYCLE IT ARRIVED IN and nothing here computes a
        delay: `arq.PactorArq.on_rx_grant` moves the link, sets `GRANT_ENTRY_SL`
        and keys the entry packet, and `onair.RadioTx._tx` lands the burst on the
        grid's next boundary as it does every other. The entry packet takes the
        slot the next PACTOR-1 packet would have had.

        AND IT IS THE ANSWER TO THE PACKET IN FLIGHT, which in every arm on
        record is the callsign announcement -- nine keyings of it at KB5LZK on
        2026-09-11, each answered with a repeat request, then the grant.
        `on_rx_grant` settles it there and carries nothing into PACTOR-3, which
        is what both reference callers do.

        ONLY `0x59A`. The other word `pactor1.UNASSIGNED_SIGNALS` names has no
        precedent anywhere in the corpus and nothing knows what it would ask for.

        ONLY IN PACTOR-1, ONLY AS THE CALLER, ONLY ONCE. Every measured grant is
        a called station commanding the caller that announced the capability; a
        link already in PACTOR-3 has nothing left to grant, and a station that
        answered a call has no transmit anchor of its own to key a new waveform
        from (`onair.RadioTx.answered_refusal`). Once, because a peer waiting for
        the entry packet repeats the word every cycle -- and because two readers
        find the same burst, the anchored one and the sweep.

        AND A GRANT THAT ARRIVES IN A TEARDOWN IS REFUSED OUT LOUD. Only
        `CONNECTED` can take one. The entry packet is the first burst of a link
        CONTINUING, and by `DISCONNECTING` our QRT is already on the air and
        repeating every cycle until the peer answers it (`arq.GOODBYE_CYCLES`);
        PACTOR-1 has nothing that un-says a goodbye, so taking the grant here
        would key a new waveform at a station that may have torn the link down
        on our own word. The refusal stands; the silence does not.

        WS8EOC, 2026-09-11: five `0x59A`
        at zero bit errors all arrived behind `max retries -> QRT`, every one
        dropped by this gate with no line anywhere, so the arm could only report
        that no entry packet was keyed and not that the peer had asked for one.
        A witness 160 mi from the gateway puts its FIRST grant a full cycle in
        FRONT of that QRT. So what a grant read in `DISCONNECTING` is evidence
        of is the teardown: a peer still commanding PACTOR-3 has not read our
        goodbye, and the budget that decided to send one counted its silence
        wrong. The line says that, because it is the only place the session gets
        told where to look.
        """
        experimental = self.p1_6a9
        expected = pactor1.CS_6A9 if experimental else pactor1.CS_59A
        if not self.p1_act_on_grant or ev.spare != expected:
            return
        trigger = "0x6A9 experimental entry response" if experimental else "0x59A grant"
        if self._grant_taken or self.stay_in_pactor1:
            return
        if self.protocol is not Protocol.PACTOR1 or self.arq.answering \
                or self.arq.role != ISS:
            return
        if self.arq.state != State.CONNECTED:
            self.log(f"{trigger} REFUSED in {self.arq.state}: an entry packet "
                     f"is the first burst of a link continuing, and only a "
                     f"CONNECTED link has one to continue. A goodbye already "
                     f"keyed is not retracted -- and a peer still commanding "
                     f"PACTOR-3 has not read it, so the budget that ended this "
                     f"link is what to go and look at")
            return
        # CONNECTED also covers a requested QRT whose packet has not yet been
        # keyed. A fresh entry cannot begin while that terminal deadline runs.
        # Gate before either grant latch, phase power, or ARQ side effects.
        if self.arq.terminal_pending:
            self.log(f"{trigger} REFUSED: termination already pending; "
                     "no new entry and the goodbye deadline is unchanged")
            return
        self._grant_taken = self._grant_pending = True
        # AND IT OVERRULES WHAT SILENCE IMPLIED. `_ruled_out` is inference: the
        # peer answered an uninvited PACTOR-3 in PACTOR-1, or stopped answering,
        # and neither says it cannot read one. `0x59A` is the peer saying it can,
        # which is the better evidence -- so an uninvited attempt that failed
        # first may not be what closes the door on an invitation that came after.
        self._ruled_out.clear()
        if experimental:
            self.log("0x6A9 experimental entry response -> selected entry waveform")
        else:
            self.log("0x59A grant -> PACTOR-3 entry packet in the next transmit slot")
        # HERE, and not behind `on_rx_grant`: that call queues the entry packet
        # and the cycle keys it on the next boundary, so this is the last
        # instant with a whole turnaround in front of the carrier the new level
        # is for.
        self._phase_power(Protocol.PACTOR3, f"the peer's {trigger}")
        self.arq.on_rx_grant()

    # -- identity ---------------------------------------------------------
    @property
    def ptchn(self) -> int:
        return int(self.settings["PTCHN"])

    @property
    def mycall(self) -> str:
        return self.settings["MYCALL"]

    def channel(self, n: int) -> _Channel:
        return self._channels.setdefault(n, _Channel())

    def call_for(self, ch: int) -> str:
        """The callsign this channel connects under, channel 0's being the one a
        channel falls back to after a disconnect [10.4.5]."""
        return self._callsigns.get(ch) or self.mycall

    def set_listen(self, on: bool) -> None:
        """PACTOR listen mode, default 1 [10.4.31 %L, terminal LIsten 6.49].

        Listen is monitoring, and it arms the receiver. Turning it off does not
        stop a PTC in standby answering a call addressed to MYCALL -- the manual
        names no command that does, and BPQ32's SCS driver sends `LISTEN 0` in
        its init script while serving inbound connects all day -- so `%L 0`
        leaves the FSM where it is.
        """
        self.listening = on
        self.settings["LISTEN"] = "1" if on else "0"
        if on:
            self.wait_until = None
            self.arq.on_host_listen(True)

    def enter_wait(self) -> None:
        """Take no calls while the scanner retunes [10.4.37]."""
        if self.wait_until is None:      # a second %W0 extends it, it does not
            self._wait_armed = self.arq.state == State.LISTENING   # forget the state
        self.wait_until = time.monotonic() + WAIT_SECONDS
        self.arq.on_host_listen(False)

    def release_wait(self) -> None:
        """%W1, or the state's own ten-second lapse [10.4.37]."""
        if self.wait_until is None:
            return
        self.wait_until = None
        if self._wait_armed:
            self.arq.on_host_listen(True)

    # ==================================================================== #
    # Serial byte stream
    # ==================================================================== #
    def open(self) -> bytes:
        """Bytes the modem emits when the port comes up."""
        return (BANNER + PROMPT).encode("latin-1")

    def feed(self, chunk: bytes) -> bytes:
        """Consume host bytes, return what the modem answers."""
        if self.hostmode:
            return self._modem.feed(chunk)
        out = bytearray()
        for i, b in enumerate(chunk):
            if self.hostmode:                       # JHOST4 mid-chunk
                return bytes(out) + self._modem.feed(chunk[i:])
            out += self._terminal_byte(b)
        return bytes(out)

    def tick(self, *, elapsed_ticks: int = 1, cycle_ticks: int = 1) -> None:
        """Age base slots and service one opportunity; never replay missed TX.

        A driver stepping whole cycles supplies its current raster's tick count
        as ``cycle_ticks`` and the actual elapsed slots as ``elapsed_ticks``.
        The default remains one tick on the 1.25 s grid.
        """
        if self.wait_until is not None and time.monotonic() >= self.wait_until:
            self.release_wait()
        self.arq.on_cycle(elapsed_ticks=elapsed_ticks, cycle_ticks=cycle_ticks)
        if self.peer is not None:
            self.peer.pump()
            self.peer.cycle()

    def feed_audio(self, audio, fs: int = 48000) -> None:
        """Live receive: decode incoming audio and drive the FSM with real events.

        The on-air counterpart to the SimPeer loopback. It runs the same decode
        path shrike.monitor renders as text (shrike.rxfront) straight into
        PactorArq, so the monitor is a faithful dry-run of this receiver:

          connect -> on_rx_connect   a PACTOR-1 link-setup burst naming us
          cs      -> on_rx_cs         the control signals a session actually
                                      turns on -- the connect-answer, ACK, NAK
          packet  -> on_rx_packet     a CRC-valid header; its status byte drives
                                      the turn-around (seq / QRT / changeover)

        A packet event carries whatever the frame can actually deliver: a PACTOR-1
        data frame hands up its field, while a P3 case-0 header has no user data of
        its own (and the P3 data-field decoder is unbuilt), so it delivers nothing
        and only drives the sequence and turnaround. Header bytes are never passed
        off as received text. detect / fsk events are informational.
        """
        from . import rxfront
        if fs != rxfront.FS:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(fs), rxfront.FS)
            audio = resample_poly(audio, rxfront.FS // g, int(fs) // g)
        for ev in rxfront.decode_events(audio):
            self.on_rx_event(ev)

    def on_rx_event(self, ev) -> None:
        """Drive the FSM from one decoded rxfront event. Shared by feed_audio (a
        whole buffer at once) and the streaming sound-card loop in shrike.live, so a
        live session and a replayed WAV take the identical path into PactorArq."""
        # Anything the front end reports is a signal on the channel that is not
        # noise, which is what the status byte's Channel-Busy means [6.90].
        self.busy_until = time.monotonic() + BUSY_HOLD_S
        if ev.kind in ("cs", "packet") and self.arq.state == State.CONNECTING \
                and ev.protocol != Protocol.PACTOR1:
            # NOT AN ANSWER TO OUR CALL, whatever it decoded to. Every PACTOR-2
            # and PACTOR-3 session is entered through a PACTOR-1 connect, and the
            # higher phase opens about three cycles AFTER the PACTOR-1 answer
            # (docs/protocols/pactor/pactor2.md sec 6,
            # docs/protocols/pactor/pactor3.md sec 17.1) -- so a station keying one
            # already holds a link, and it is not ours, because ours is not up.
            # Neither a data packet nor a codeword carries an address, so the
            # protocol a frame arrives in is the only thing this phase has to
            # separate the station we called from an exchange we can hear.
            #
            # The scan that found it is left wide on purpose: a receiver reporting
            # what is on the frequency is the operator's business, and being
            # unable to bring our link up is this station's.
            #
            # AND IT DOES NOT BUY A RETRY. `note_peer_heard` forgives a cycle on
            # energy that identifies nobody, because nobody is who it could be --
            # including the station we called, keying too poorly to decode [see
            # arq.note_peer_heard and the WS8EOC session behind it]. This frame is
            # the opposite: it decoded, and what it decoded to is a station three
            # cycles into somebody else's link. Spending it makes an occupied
            # frequency the reason we go on calling over the occupant, which
            # inverts the yield.
            #
            # MEASURED on rig-session-20260813-235522, calling WS8EOC at
            # 10146200 Hz. One control signal decoded in the whole session -- a
            # PACTOR-3 CYCLE-TOG at zero bit errors in cycle 15, refused at line
            # 96 -- and the presence it recorded was spent five cycles later, at
            # line 119, as "still transmitting -> not counting this retry". The
            # call then ran to cycle 30: 13 bursts, 12.5 s of carrier, 32% of the
            # air, on a frequency whose only readable occupant was not ours.
            self.log(f"{ev.protocol} is not an answer to a call -- still calling")
            return
        if self.arq._rx_close_pending and not (
                ev.kind == "packet" and ev.packet is not None
                and ev.packet[3] and ev.packet[1] & spec.STATUS_QRT
                and ev.protocol == self.protocol):
            return
        if (self.no_p3_fallback and self.protocol is Protocol.PACTOR3
                and ev.kind in ("cs", "packet")
                and ev.protocol in (Protocol.PACTOR1, Protocol.PACTOR2)):
            # A lower-mode ACK or break-in cannot settle P3 bytes or change
            # our role under this option. Repeated grants use the unassigned
            # path below and remain evidence of an answering peer.
            self.log(f"rx {ev.protocol} {ev.kind}: --no-p3-fallback "
                     "keeps the PACTOR-3 state unchanged")
            return
        if ev.kind in ("cs", "packet") and self._stale_pactor1(ev):
            return
        if ev.kind == "connect":
            called = ev.connect.callsign.upper()
            if called == self.mycall:                # the burst names the called party
                self.arq.on_rx_connect("", called, ev.connect.variant)
        elif ev.kind == "p1reply":
            # PRESENCE ONLY -- and presence does not bring a link up.
            #
            # This used to call the link established on the arrival of any
            # correctly-timed PACTOR-1 burst, on the reasoning that nothing else
            # would be transmitting in the peer's slot. On the air that reported
            # CONNECTED to a station whose reply was never decoded: one sweep
            # arm logged a connect to W6IDS off 14 bursts and not one zero-error
            # control signal. A burst is a burst; it can be QRM, another station,
            # or the tail of someone else's exchange.
            #
            # The bar is a DECODED control signal, which the receiver now
            # delivers at zero errors against a distance-8 code. Presence is
            # still worth logging -- it says the frequency is not dead and the
            # call may be being heard -- but it is not an answer.
            if self.arq.state == State.CONNECTING:
                self.log("peer burst heard (not decoded) -- still calling")
        elif ev.kind == "cs":
            if self.arq.state == State.CONNECTING:
                # WHILE WE ARE CALLING, THE PHASE DECIDES THIS AND NOT `protocol`.
                # A call is a PACTOR-1 connect, its answer is a PACTOR-1 codeword,
                # and the guard above has already refused anything that arrived in
                # another protocol -- while `self.protocol` is what WE would
                # transmit, which a link that has not come up yet has no business
                # reading a peer's codeword through. Asking it sent the answer to
                # a call down the in-session path whenever a previous link had
                # left the protocol upgraded, where CS3 -- never bare, so never an
                # answer -- brings a link up and hands the channel away.
                #
                # A connect is answered with CS1 or CS4 and with NOTHING ELSE, and
                # what the answer calls for is a PACKET. "Sobald das erste
                # gueltige CS empfangen und synchronisiert ist, wird das erste
                # normale Datenpaket mit Head=AA (HEX) und Paketzaehler=1
                # ausgesendet."
                #
                # This used to accept any of the four, and two of them are not
                # answers: CS2 is the other half of the in-session acknowledge
                # alternation, and CS3 is never bare at all -- it is the head of
                # the break-in packet. Either one arriving here is a decode off a
                # station that is not talking to us, or an exchange we have walked
                # in on. Both brought a link up.
                #
                # Not routed through on_rx_cs, because during setup the codeword
                # is not carrying its in-session meaning: CS1 says our 200 Bd
                # redundancy section arrived clean, CS4 says it did not, and CS4
                # mid-session means something else entirely. Treating a setup
                # answer as an in-session codeword made shrike hand over a link
                # that had not been established and then sit acknowledging a
                # station that was waiting for a packet.
                # Refused, and it buys no retry either, for the reason the
                # PACTOR-2/3 guard above gives: a codeword read at radius 1 in a
                # slot we are not the sender of names an exchange, and it is not
                # ours.
                if ev.cs not in (pactor1.CS_ACK_A, pactor1.CS_SPEED):
                    self.log(f"CS{(ev.cs or 0) + 1} is not an answer to a call "
                             f"-- still calling")
                    return
                # Remember it: the peer repeating this same codeword next cycle
                # is asking for the packet again, not breaking in. Without this
                # the first repeat read as a fresh break-in and handed over a
                # link that was one packet old.
                #
                # AND IT IS WHY A RUN OF THE ANSWER MEANS "NOT YET", which is worth
                # spelling out because the opposite reading is the natural one and
                # it is wrong. Suspecting that a peer's unbroken CS1 run was really
                # an acknowledgement we were deaf to -- that seeding this deadlocked
                # the link -- is refuted without needing the air. "An accepting
                # station sends its answer every cycle until the caller's first data
                # packet decodes" (docs/protocols/pactor/pactor1-control-signals.md
                # sec 5). So if the answer was CS1 and the acknowledgement of packet
                # #1 were ALSO CS1, the caller could not tell "still waiting" from
                # "got it", and the protocol would have no way to start. Acceptance
                # has to be a CHANGE of codeword, which is exactly the rule below.
                #
                # An unbroken run of the connect answer at zero errors therefore
                # says one thing only: our first data packet has not decoded, once
                # per cycle, for the length of the run.
                # Both memories, and the second is what carries the first-block
                # exception: "CS4 dient als 'REQUEST'-CS fuer den ersten
                # 100-Bd-Block, Bestaetigung erfolgt mittels CS1 oder CS3." A CS4
                # answer followed by another CS4 is that request, not a speed
                # offer, and the speed-up branch is unreachable until CS1 or CS3
                # has acknowledged the first block.
                self._last_rx_cs = self._prev_rx_cs = ev.cs
                self._answer_link_setup(ev.cs)
            else:
                # A 20-bit codeword at mutual distance 12, read at radius 1 or
                # less, is evidence about the far end that nothing else in the
                # receiver produces -- so a control signal may carry the link
                # across in either direction on its own.
                if self.arq.state == State.CONNECTED or (
                        self.arq.state == State.DISCONNECTING
                        and ev.cs == pactor1.CS_CHANGEOVER):
                    # A break-in HEAD is the front of a changeover packet, so it
                    # carries the same exemption the packet does.
                    self._follow_peer(ev.protocol,
                                      breakin=ev.cs == pactor1.CS_CHANGEOVER)
                cs = self._logical_cs(ev)
                if cs == CS_BREAKIN:
                    self._entry_answered()
                    if ev.protocol == Protocol.PACTOR3 and self.arq.unconfirmed_breakin:
                        # A CS3 head with an unread body can be the peer's old
                        # turn repeated. Keep bytes and the saved IRS epoch until
                        # a CRC-valid body/phase or a real ACK settles the turn.
                        self.arq.on_unconfirmed_breakin_head()
                        return
                self.arq.on_rx_cs(cs)
        elif ev.kind == "packet":
            sl, status, payload, ok = ev.packet
            if ev.protocol == Protocol.PACTOR3:
                # This frame owns the next answer. It can replace an older
                # queued ACK with a CS3 packet, which may itself be refused;
                # that refusal must not release the old packet's ACK instead.
                cancel = getattr(self.peer, "cancel_pending_cs", None)
                if cancel is not None:
                    cancel()
            # A LONE SPEED-LEVEL-1 ACCEPT MAY NOT MOVE THE LINK. Level 1 carries no
            # constant headers, so it can never be header-anchored and comes out of
            # real band noise at about 5e-7 per position -- roughly one per fifteen
            # captures. Levels 2 to 6 are anchored and clean: zero confirmed in
            # 1.57 million positions. So the protocol follow, which is a permanent
            # change to what this station transmits, is driven by the levels that
            # cannot be manufactured, and a level 1 frame still delivers its
            # payload and its sequence like any other.
            #
            # A CHANGEOVER PACKET IS EXEMPT, and its head is why: it reports the
            # two-carrier geometry as level 1 because that is the comb it rides,
            # but nothing about it was found by a blind scan. `rxfront._cs_event`
            # only reaches this frame from a CS3 already read at zero bit errors
            # of twenty, and the frame still faced the CRC and `p3rx.confirmed`
            # behind it.
            #
            # AND IT IS PACTOR-3'S LEVEL 1, not level 1 as a number. PACTOR-2's
            # floor is not found by a blind scan either: `p2rx.find_markers`
            # arms on a nine-pulse codeword at 0.94 of its ceiling and the level
            # is read off that marker, so a level-1 PACTOR-2 frame is anchored
            # where a level-1 PACTOR-3 one is precisely not.
            if ok and (self.arq.state == State.CONNECTED
                       or (ev.breakin and self.arq.state == State.DISCONNECTING)) \
                    and (sl != 1 or ev.breakin
                         or ev.protocol != Protocol.PACTOR3):
                self._follow_peer(ev.protocol, breakin=ev.breakin)
            # The same spend a bare CS3 makes above: a break-in IS the answer to
            # the entry packet in both reference recordings, and reading the rest
            # of the packet must not cost the link what reading only its head
            # bought (`_entry_answered`). A PACTOR-3 packet from the peer with no
            # break-in head on it spends it too, and for the reason
            # `arq.PactorArq.on_rx_packet` yields on one: it is the same answer,
            # read a cycle later, and it says the same thing about the waveform.
            entry = self.arq.entry_pending
            if ok and ev.protocol == Protocol.PACTOR3 and not ev.breakin \
                    and self.arq.role == IRS and ev.cycle_long is not None:
                # A CS6 we sent may have been lost. The variable header, not
                # status bit 5, tells us which cycle the peer actually used.
                # Reconcile before answering so an unfulfilled request gets
                # CS6 again, rather than a speed-up on the assumed new length.
                self.arq.observe_peer_cycle(ev.cycle_long)
            previous_role = self.arq.role
            repeated_stint = False
            if ok and ev.breakin and ev.protocol == Protocol.PACTOR3 \
                    and self.arq.unconfirmed_breakin:
                recover = getattr(self.peer, "recover_p3_turn", None)
                if recover is not None:
                    repeated_stint = bool(recover(ev))
            self._p3_changeover_answer = (ev.breakin
                                          and ev.protocol == Protocol.PACTOR3)
            try:
                self.arq.on_rx_packet(sl, payload, status, ok,
                                      breakin=ev.breakin, protocol=ev.protocol,
                                      repeated_stint=repeated_stint)
            finally:
                self._p3_changeover_answer = False
            if previous_role == ISS and self.arq.role == IRS:
                # A whole changeover packet can arrive without its head ever
                # passing _logical_cs. Its accepted role change ends our trial
                # just as a separately decoded CS3 does.
                self._p1_speedup_pending = False
                self._hispeed_tries = 0
            if ok and entry and self.protocol is not Protocol.PACTOR1:
                self._entry_answered()
        else:
            self.log(f"rx {ev.kind}: {ev.text}")
            if ev.kind == "unassigned":
                # A REPEAT OF THE GRANT IS A DIAGNOSIS, and `_take_grant` is a
                # no-op after the first one, so acting on the word only there
                # threw every later copy away. What it says is in that method's
                # own reading of `0x59A`: the peer commands PACTOR-3 once a cycle
                # for as long as it is waiting for the entry packet, so a copy
                # arriving after we have keyed one is the far end saying it did
                # not read what we sent.
                if ev.spare == pactor1.CS_59A:
                    self.arq.note_upgrade_unread()
                self._take_grant(ev)
                # AND IT IS THE PEER BEING THE IRS. Twelve bits at zero errors
                # in the slot the receiving station owes us is that station
                # answering, whatever the word means -- so it refutes the
                # stranded-ISS reading the changeover rests on, which is a
                # different question from what the word asks for above.
                self.arq.note_peer_answering()
        # Bare PACTOR-1 FSK is not an answer -- no callsign, no codeword, nothing
        # that could bring a link up -- but it does say somebody is keying on this
        # frequency, and a caller that gives up while the far end is still calling
        # never completes anything. It only holds the retry budget open.
        #
        # ALL THREE SHAPE KINDS, because presence is a question about the air and
        # every one of them answers it. `detect` was left out on the strength of
        # its name; what it measures is a carrier pair over the guard bands, which
        # `rxfront._p2_present` says in its own docstring is a geometry and not a
        # protocol. Over the nine hold cycles that spent the link-dead budget in
        # the 2026-08-18 clamped arm it is the only kind to reach
        # here in two of them, and the traffic underneath it decodes off disk as
        # PACTOR-1.
        #
        # AND `unassigned`, which is stronger than any of them and was weaker
        # than all three: a codeword read at zero of twelve bits is a station
        # transmitting on our raster, and leaving it out meant a cycle carrying
        # one counted as a silent cycle while undecoded energy in the same slot
        # would not have. Three arms on 2026-08-22 spent a give-up budget that
        # way, `0x59A` at zero bit errors in every cycle of it. What the word
        # ASKS FOR is `_take_grant`'s business above; that somebody sent it is
        # this line's.
        if ev.kind in ("p1reply", "fsk", "detect", "unassigned"):
            self.arq.note_peer_heard()

    # ==================================================================== #
    # Terminal mode -- the cmd: prompt
    # ==================================================================== #
    def _terminal_byte(self, b: int) -> bytes:
        if b == 10:                                  # <LF> ignored on input [5.1]
            return b""
        if b == 13:
            line, self._line[:] = bytes(self._line), b""
            return b"\r\n" + self._command(line.decode("latin-1")) + PROMPT.encode()
        if b == 8:
            if self._line:
                del self._line[-1]
                return b"\b \b"
            return b""
        self._line.append(b)
        return bytes((b,)) if 32 <= b < 127 else b""

    def _command(self, line: str) -> bytes:
        # "The command that switches a TNC/PTC into hostmode is <ESC>JHOST4<CR>"
        # [10.9.6]. ptc-go sends the bare word, so both forms have to read.
        word, _, arg = line.strip().lstrip("\x1b").partition(" ")
        arg = arg.strip()
        if not word:
            return b""
        name = self._resolve(word)
        if name is None:
            # Unreadable input -- a stray hostmode frame, line noise. The host is
            # mid-handshake; answering with an error would only confuse it.
            return b""
        if name == "JHOST":
            # JHOST4 and JHOST 4 both occur. 1 is plain WA8DED hostmode, 4 and 5
            # the CRC one [10.4.6]; 5 differs only in a FAX channel we do not
            # have, so it is served as 4 [unknowns.HOST_JHOST5].
            param = word[5:] or arg
            if param in ("1", "4", "5"):
                self.hostmode = True
                self._modem.reset(crc=param != "1")   # [10.9.6]
                self.settings["%M"] = "0"            # reset at hostmode start [10.4.32]
                self.channel(0).rx += BANNER.encode("latin-1")
            return b""
        if name == "QUIT":
            return b"*** ERROR: no submenu active\r\n"   # already in the main menu
        if name == "VERSION":
            return f"*** {self._version()}\r\n".encode("latin-1")
        if name == "RESTART":
            self.settings["%M"] = "0"
            return BANNER.encode("latin-1")
        if name == "CONNECT":
            return self._call(arg)
        if name == "DISCONNECT":               # [6.37], and DD [6.34]
            self.arq.on_host_disconnect()
            return b""
        if name == "DD":
            self.arq.on_host_abort()
            return b""
        if name in ("PT", "PR", "PAC"):
            return b""
        if arg:
            try:
                self._store(name, arg)
            except ValueError as exc:
                return f"*** ERROR: {exc}\r\n".encode("latin-1")
            return b""
        return f"*** {name}: {self.settings.get(name, '0')}\r\n".encode("latin-1")

    def _call(self, arg: str) -> bytes:
        """Terminal `C [%|!]CALL`, and a bare `C` re-dials [PTC-IIIusb 4.1 §6.22]."""
        try:
            self.arq.on_host_connect(self.mycall, arg.split()[0] if arg else "")
        except ValueError as exc:
            return f"*** ERROR: {exc}\r\n".encode("latin-1")
        # The manual prints the operator it took -- "(ROBUST CONNECT)" is its
        # wording; the longpath marker follows its shape.
        marker = {"robust": " (ROBUST CONNECT)", "longpath": " (LONGPATH)"}
        return (f"*** NOW CALLING {self.arq.dxcall}"
                f"{marker.get(self.arq.connect_variant, '')}\r\n").encode("latin-1")

    def _store(self, name: str, arg: str) -> None:
        """Keep a setting. The two acceptance settings also reach the ARQ, where
        `CONType` decides which calls this station answers."""
        if name in ("CONTYPE", "CONINTEGRITY"):
            # Validated by the config's own rules, and before it is stored: a
            # value the ARQ refuses is one the host never gets told we took.
            self.arq.cfg = replace(self.arq.cfg, **{name.lower(): int(arg)})
        if name == "LISTEN":
            self.set_listen(arg != "0")
        if name == "STATUS":
            # BPQ32's init script opens with STATUS 2 and its driver says outright
            # that automatic status must be enabled for it [6.90, 10.6.1].
            self.autostatus = arg.strip() == "2"
        self.settings[name] = arg.upper() if name == "MYCALL" else arg

    def _resolve(self, word: str) -> Optional[str]:
        """Map a possibly-abbreviated command word to its canonical upper-case name."""
        if not word.replace("%", "").replace("@", "").isalnum():
            return None
        w = word.upper()
        if w.startswith("JHOST"):
            return "JHOST"
        for canon, abbrev in _ABBREV.items():
            if canon.startswith(w) and w.startswith(abbrev):
                return canon
        # Unknown settings are stored, not rejected: refusing a command a client
        # believes is essential aborts the session, while quietly accepting one
        # we do not model costs nothing.
        return w if w.isalpha() else None

    def _version(self) -> str:
        return f"SCS PTC-IIIusb Firmware {FIRMWARE} BIOS {BIOS}"

    # ==================================================================== #
    # Hostmode -- channel dispatch
    # ==================================================================== #
    def _dispatch(self, pkt: hostmode.Packet) -> bytes:
        # JHOST0 hands the port back to the terminal parser from any channel, so
        # a client that reconnects can re-run its whole open sequence [10.2].
        if pkt.is_command and pkt.text.strip().upper().startswith("JHOST0"):
            self.hostmode = False
            return hostmode.reply(pkt.channel, hostmode.OK)
        if pkt.channel == 255:
            return self._poll_all()
        if pkt.channel == 254:
            return self._status(pkt)
        if pkt.channel == self.ptchn:
            return self._pactor(pkt)
        return self._other(pkt)

    def _poll_all(self) -> bytes:
        """Extended hostmode: which channels have something waiting [10.5]."""
        live = [n for n, ch in sorted(self._channels.items()) if ch.pending()]
        # Auto status is triggered by a change in byte 1 or byte 3 [10.6] -- the
        # PACTOR level in byte 2 moving on an upgrade is not a status change.
        cur, last = self._status_bytes(), self._laststatus
        if self.autostatus and (last is None or (cur[0], cur[2]) != (last[0], last[2])):
            live.append(254)
        return hostmode.reply(255, hostmode.MSG, bytes(n + 1 for n in live))

    def _status(self, pkt: hostmode.Packet) -> bytes:
        """Status channel: G[0..3] returns that many status bytes plus one [10.6]."""
        arg = pkt.text[1:].strip()
        n = int(arg) + 1 if arg.isdigit() else 1
        self._laststatus = self._status_bytes()
        return hostmode.reply(254, hostmode.DATA, self._laststatus[:n])

    def _status_bytes(self) -> bytes:
        """The four status bytes of manual 10.6, byte 1 built per 6.90."""
        connected = self.arq.state in (State.CONNECTED, State.DISCONNECTING)
        if self.arq.state == State.CONNECTING:
            mode, status = 0b010, 0b110                     # PACTOR-ARQ, SYNCH
        elif connected:
            mode, status = 0b010, (0b010 if self._txbuf else 0b011)   # TRAFFIC/IDLE
        elif self.listening:
            mode, status = 0b110, 0b111                     # LISTEN
        elif self.busy_until is not None and time.monotonic() < self.busy_until:
            # "An occupied HF channel is indicated by a status value of 247 ...
            # only given in the STBY condition, and not in Listen mode" [6.90],
            # which is what a scanning client holds its transmission off.
            mode, status = 0b111, 0b111                     # Channel-Busy
        else:
            mode, status = 0b000, 0b111                     # STANDBY / IGNORE
        direction = 0b1000 if connected and self.arq.role == ISS else 0
        # Byte 2 is the PACTOR level the link is actually running at, which is 1
        # until it upgrades. Reporting 3 for a PACTOR-1 link told the host the one
        # thing about the session it could not see for itself, wrongly.
        level = int(self.protocol[-1]) if connected else 0
        # Byte 3 has to be derived the same way. The ARQ's stored speed level
        # survives a fallback (`arq.PactorArq.speed_level` says so), and a link
        # that fell back to PACTOR-1 -- which `fall_back` makes routine -- kept
        # reporting the PACTOR-3 entry level it was no longer running at.
        sl = (P1_SPEED_LEVEL if self.protocol is Protocol.PACTOR1
              else self.arq.speed_level - 1)
        # Byte 4 is the receive frequency offset, and 128 is the manual's "no
        # estimate yet" [10.6]. A zero here is a claim to be dead on frequency,
        # which no part of this modem has measured.
        return bytes((0x80 | mode << 4 | direction | status,
                      level,
                      sl if connected else 0,
                      128))

    def _other(self, pkt: hostmode.Packet) -> bytes:
        """Any channel that is not the PACTOR one: idle Packet-Radio channels."""
        if pkt.channel == TRX_CHANNEL and not pkt.is_command:
            # There is no transceiver behind this port, and answering OK to CAT
            # bytes we then drop is the one answer a host cannot recover from
            # [10.7, unknowns.HOST_TRX_CHANNEL].
            return hostmode.reply(pkt.channel, hostmode.FAIL)
        if not pkt.is_command:
            return hostmode.reply(pkt.channel, hostmode.OK)
        text = pkt.text.strip()
        verb, arg = _split_command(text)
        if verb == "G":
            return self._get(pkt.channel, text)
        if verb == "L":
            return self._answer_link_status(pkt.channel, arg)
        if verb == "I":
            return self._callsign(pkt.channel, arg)
        if verb in ("C", "D", "DD"):
            return hostmode.reply(pkt.channel, hostmode.OK)   # no link on this channel
        return self._setting(pkt.channel, verb, arg)

    # -- the PACTOR channel ----------------------------------------------
    def _pactor(self, pkt: hostmode.Packet) -> bytes:
        ch = pkt.channel
        if not pkt.is_command:
            self.arq.on_host_data(pkt.data)
            return hostmode.reply(ch, hostmode.OK)

        text = pkt.text.strip()
        if text.startswith("#"):
            # BPQ32's SCS driver calls this "a hidden feature where you can send
            # any normal mode command in host mode by preceding it with a #", and
            # sends its dirty disconnect and its MYLevel that way.
            self._command(text[1:])
            return hostmode.reply(ch, hostmode.OK)

        verb, arg = _split_command(text)
        if verb == "G":
            return self._get(ch, text)
        if verb == "L":
            return self._answer_link_status(ch, arg)
        if verb == "C":
            if self.arq.state in (State.CONNECTED, State.DISCONNECTING):
                return hostmode.reply_text(ch, hostmode.FAIL,
                                           "CHANNEL ALREADY CONNECTED")
            try:
                self.arq.on_host_connect(self.call_for(ch),
                                         arg.split()[0] if arg else "")
            except ValueError as exc:
                return hostmode.reply_text(ch, hostmode.FAIL, str(exc))
            return hostmode.reply(ch, hostmode.OK)
        if verb == "DD":
            # Not in chapter 10, but ptc-go's forceDisconnect sends exactly this
            # and BPQ32 sends `#DD`, so the verb stays
            # [unknowns.HOST_UNDOCUMENTED_COMMANDS].
            self.arq.on_host_abort()
            return hostmode.reply(ch, hostmode.OK)
        if verb == "D":
            # "If the Disconnect command is given twice, one after the other,
            # then the link is broken immediately" [10.4.2]. "One after the
            # other" is per link, not per packet: every client polls G and L
            # between its commands, so spending the first D on a poll would put
            # the hard break out of reach of the clients it is there for.
            (self.arq.on_host_abort if self._disconnect_asked
             else self.arq.on_host_disconnect)()
            self._disconnect_asked = True
            return hostmode.reply(ch, hostmode.OK)
        if verb == "I":
            return self._callsign(ch, arg)
        if verb == "V":
            return hostmode.reply_text(ch, hostmode.MSG, self._version())
        if verb == "@B":
            return hostmode.reply_text(ch, hostmode.MSG, "65536")
        if verb.startswith("%"):
            return self._percent(ch, text)
        return self._setting(ch, verb, arg)

    def _percent(self, ch: int, text: str) -> bytes:
        code, arg = text[1:2].upper(), text[2:].strip()
        if code == "V":
            return hostmode.reply_text(ch, hostmode.MSG, f"{FIRMWARE} {BIOS}")
        if code == "T":
            if arg:
                self._sent_base = self.sent_total
            return hostmode.reply_text(ch, hostmode.MSG,
                                       str(self.sent_total - self._sent_base))
        if code == "L":
            if not arg:
                return hostmode.reply_text(ch, hostmode.MSG,
                                           self.settings.get("LISTEN", "1"))
            self.set_listen(arg != "0")
            return hostmode.reply(ch, hostmode.OK)
        if code == "B":
            return hostmode.reply_text(ch, hostmode.MSG, "SCS - DSP MULTI MODEM 1200")
        if code == "W":
            # Scan sync [10.4.37]: 1 = safe to retune, and the answer itself is
            # the promise -- the WAIT state it enters takes no call for ten
            # seconds, whether %W1 releases it or the state lapses on its own.
            if arg == "1":
                self.release_wait()
                return hostmode.reply_text(ch, hostmode.MSG, "")
            free = self.arq.state in (State.DISCONNECTED, State.LISTENING)
            if free:
                self.enter_wait()
            return hostmode.reply_text(ch, hostmode.MSG, "1" if free else "0")
        if code == "M":
            if not arg:
                return hostmode.reply_text(ch, hostmode.MSG,
                                           self.settings.get("%M", "0"))
            if arg != "0":
                # "an error message (code byte=2) which contains the maximum
                # possible argument value" [10.4.32]. No expansion level is
                # implemented here, so the maximum this modem can honour is 0 --
                # a client that set %M1 would wait for code-8 echoes for ever.
                return hostmode.reply_text(ch, hostmode.FAIL, "0")
            self.settings["%M"] = arg
            return hostmode.reply(ch, hostmode.OK)
        # Changeover [PTC hostmode manual]: %O reverses the link either way -- it
        # breaks in when receiving, and hands over when the transmit buffer has
        # been sent and confirmed. %I only breaks in (IRS only), %Q only hands
        # over: it puts an over token at the end of the transmit buffer and never
        # breaks in.
        if code == "O":
            self.arq.on_host_changeover()
            return hostmode.reply(ch, hostmode.OK)
        if code == "I":
            self.arq.on_host_breakin()
            return hostmode.reply(ch, hostmode.OK)
        if code == "Q":
            self.arq.on_host_over()
            return hostmode.reply(ch, hostmode.OK)
        return hostmode.reply(ch, hostmode.OK)

    def _answer_link_status(self, ch: int, arg: str) -> bytes:
        """L, for the channel in the argument or the one the packet arrived on.

        10.4.8 gives L a channel parameter; the WA8DED guide sends a bare L to
        the channel of interest. ptc-go and BPQ32 both send the bare form, and
        this tree's own client sends `L <ch>` on channel 0, so both must read.
        """
        target = int(arg) if arg.isdigit() else ch
        if target == self.ptchn:
            return hostmode.reply_text(ch, hostmode.MSG, self._link_status())
        chan = self._channels.get(target)
        waiting = (len(chan.link), math.ceil(len(chan.rx) / hostmode.MAX_DATA)) \
            if chan else (0, 0)
        # "Only items a and b are displayed for channel 0" [WA8DED guide].
        fields = waiting if target == 0 else waiting + (0, 0, 0, LINK_DISCONNECTED)
        return hostmode.reply_text(ch, hostmode.MSG, " ".join(str(v) for v in fields))

    def _callsign(self, ch: int, arg: str) -> bytes:
        """I, the station callsign, which is set per channel [10.4.5]."""
        if not arg:
            return hostmode.reply_text(ch, hostmode.MSG, self.call_for(ch))
        if ch == 0:
            self.settings["MYCALL"] = arg.upper()
        else:
            self._callsigns[ch] = arg.upper()
        return hostmode.reply(ch, hostmode.OK)

    def _setting(self, ch: int, verb: str, arg: str) -> bytes:
        """Any other parameter command: store it, or read it back.

        "If the command is successful and returns information (like any command
        without an argument), a code 1 transmission will be the reply" [WA8DED
        guide]; every hostmode parameter in 10.4 is documented with such a
        read-back form.
        """
        if not arg:
            return hostmode.reply_text(ch, hostmode.MSG, self.settings.get(verb, "0"))
        try:
            self._store(verb, arg)
        except ValueError as exc:
            return hostmode.reply_text(ch, hostmode.FAIL, str(exc))
        return hostmode.reply(ch, hostmode.OK)

    def _get(self, ch: int, text: str) -> bytes:
        """G poll: link status events first, then received data [WA8DED guide]."""
        arg = text[1:].strip()
        chan = self.channel(ch)
        if chan.link and arg != "0":
            return hostmode.reply_text(ch, hostmode.LINK, chan.link.pop(0))
        if chan.rx and arg != "1":
            n = min(len(chan.rx), hostmode.MAX_DATA)
            blob, chan.rx[:] = bytes(chan.rx[:n]), chan.rx[n:]
            return hostmode.reply(ch, hostmode.DATA, blob)
        return hostmode.reply(ch, hostmode.OK)

    def _link_status(self) -> str:
        """The L command's six fields [WA8DED guide, confirmed by ptc-go]."""
        state = {State.CONNECTING: LINK_SETUP, State.CONNECTED: LINK_INFO,
                 State.DISCONNECTING: LINK_DISCONNECT_REQ}.get(
                     self.arq.state, LINK_DISCONNECTED)
        chan = self.channel(self.ptchn)
        unsent = math.ceil(self._txbuf / hostmode.MAX_DATA)
        return " ".join(str(v) for v in (
            len(chan.link),                                    # a: link msgs waiting
            math.ceil(len(chan.rx) / hostmode.MAX_DATA),       # b: rx frames waiting
            unsent,                                            # c: not transmitted
            1 if unsent else 0,                                # d: not acknowledged
            0,                                                 # e: retries
            state))                                            # f: link state

    # ==================================================================== #
    # ArqIO -- the FSM's view of us
    # ==================================================================== #
    def connect_burst(self, mycall: str, dxcall: str) -> None:
        if self.peer is not None:
            self.peer.connect_burst(mycall, dxcall)

    def _p1_field(self, baud: int) -> None:
        """Tell the link layer how much a packet at this rate can carry."""
        self.arq.payload_bytes_override = pactor1.DATA_FIELD[baud]
        self.arq.breakin_bytes_override = pactor1.BREAKIN_FIELD[baud]

    def breakin_now(self) -> None:
        """We are about to seize the channel. [arq.ArqIO]

        Called BEFORE the field is chunked, which is the whole reason it is a
        seam of its own rather than a branch in `send_packet`: a changeover
        packet's field is 7 bytes at 100 Bd and 3 in PACTOR-3, against the 59 an
        ordinary speed-level-3 packet carries, and a chunk built to the wrong one
        is silently truncated by the renderer.

        THIS USED TO DROP THE LINK TO PACTOR-1, on the grounds that nothing had
        established how a CS3 head joins a PACTOR-3 frame and that a plausible
        construction offered to the air is how a wrong waveform gets frozen in.
        `placement.CHANGEOVER` is now read off both directions of a real session
        rather than constructed, so the packet goes out in the protocol the link
        is in and the link stays where it is. What the fall-back cost was not the
        cycle it spent: under `p1_grant_only` the grant is the only door back
        into PACTOR-3 and `_grant_taken` has closed it, so the first changeover
        ended the session's PACTOR-3 phase for good.

        The field is three bytes whatever the speed level, because the changeover
        packet is not keyed at one -- it is two carriers and 56 rows however fast
        the traffic behind it runs.

        PACTOR-2 HAS NO SUCH PACKET, so there is no size to override: the
        break-in there is the bare CS3 and the field that follows it, a cycle
        later, is an ordinary one. See `breakin_rides_a_packet`.
        """
        if self.protocol in (Protocol.PACTOR1, Protocol.PACTOR2):
            return
        self.arq.breakin_bytes_override = placement.CHANGEOVER.crc_bytes - 3

    @property
    def breakin_rides_a_packet(self) -> bool:
        """Whether taking the link keys a packet or a bare codeword. [arq.ArqIO]

        A changeover packet is a CS3 head and a SHORT field behind it --
        `placement.CHANGEOVER` is 56 rows where an ordinary packet is 72, and
        PACTOR-1's is 7 bytes where an ordinary one is 8 -- because the head has
        to fit in the same 1.25 s cycle as the data.

        PACTOR-2 cannot be given one. Its frame is 72 pulses behind a nine-pulse
        marker and that marker's codeword index carries a speed level and a frame
        length and nothing else, so a 51-pulse frame is neither constructible nor
        announceable, and no recording holds one to read the geometry off. There
        the codeword IS the break-in, which is all "CS3 forces a break-in" ever
        said; the new sender's first packet goes out on the next boundary, one
        cycle later than PACTOR-1 or PACTOR-3 would key it."""
        return self.protocol is not Protocol.PACTOR2

    @property
    def asking_for_a_grant(self) -> bool:
        """Whether a grant is still the only way this link reaches PACTOR-3.
        [arq.ArqIO]

        NARROWER THAN "the announcement is on the air", and the September 13
        arm is why: VE3KPG was called with `--p1-grant-only`, so the 0x71 packet
        that handed the turn away on #1 came from an arm where the grant was the
        ONLY door and the announcement was the whole of the ask. Status bits 4-5
        alone do not say that. They are the default on every arm that is not
        carrying mail (`onair._arm_defaults`), including arms whose PACTOR-3
        comes from an uninvited upgrade and arms that never leave PACTOR-1 --
        and on those the hold costs a changeover nothing else in the session can
        ask for: an ISS with a drained buffer holds the link until the peer
        breaks in, and bit 6 is the only thing that invites one.

        So the hold bites exactly while `_take_grant` could still fire -- a
        grant this link would act on, with no grant taken yet -- and the other
        half of the rule lives where it belongs, on the entry packet.
        `arq.PactorArq._start_next_packet` holds the request while
        `entry_pending` stands, whatever drew the entry, so an ANNOUNCING arm
        that draws a grant keeps bit 6 down from the grant until the peer has
        read the entry.
        """
        return (self.protocol is Protocol.PACTOR1 and self.p1_grant_only
                and self.p1_act_on_grant and not self.stay_in_pactor1
                and not self._grant_taken)

    def send_packet(self, sl: int, payload: bytes, status: int,
                    breakin: bool = False) -> Optional[int]:
        """Send a data packet in whatever protocol the link is actually in.

        Same seam as send_cs, and for the same reason: an un-upgraded link is
        PACTOR-1, and rendering a PACTOR-3 frame onto it is as good as silence.

        `status` arrives in PACTOR-3's layout and the two are NOT the same byte:
        P3 spends bits 2-4 on the data type and bit 5 on a long-cycle request,
        P1 spends bits 2-3 on the Datenmodus and has no long cycle at all. Only
        the mod-4 counter sits in the same place. So the flags are translated
        rather than forwarded -- and only the counter used to survive the trip,
        which left the changeover request and the QRT with nowhere to go and no
        way for shrike to give a PACTOR-1 peer its turn.

        The break-in case is settled a step earlier, in `breakin_now`.

        THE ENTRY PACKET IS ITS OWN CALL, because its arrangement is pinned
        rather than taken from the cycle -- see `onair.RadioTx.send_entry_packet`
        for what that costs when it is not. Which rung of the ladder we are on is
        `arq.PactorArq.entry_variant`, and the only thing it changes here is the
        acquisition preamble; the field it changes in the FSM.

        THE FIELD EACH SEAM RENDERED COMES BACK THROUGH HERE, unread, and so
        does a seam's `arq.REFUSED`. This is a router and not a renderer, so what
        it knows is which seam a packet went to and not how many bytes that seam
        could build, nor whether the guard let the burst out -- and the difference
        matters: `SimPeer` is a second link layer that carries whatever it is
        handed, where `onair.RadioTx` and the audio harness build a fixed-size
        frame and cut the rest away. A router that answered for both would
        report a truncation that never happened. `arq.PactorArq._sent` reads
        whatever the seam says, and a seam that says nothing is unmeasured.
        """
        if self.peer is None:
            return None
        p1 = self.protocol is Protocol.PACTOR1
        if self.protocol is Protocol.PACTOR2 \
                and hasattr(self.peer, "send_p2_packet"):
            # NO BREAK-IN ARM. PACTOR-2's changeover is the bare CS3 of the
            # cycle before (`breakin_rides_a_packet`), so what reaches here on
            # the cycle after it is an ordinary packet with an ordinary field.
            if self.arq.cycle_long and hasattr(self.peer, "send_p2_long_packet"):
                return self.peer.send_p2_long_packet(sl, payload, status)
            return self.peer.send_p2_packet(sl, payload, status)
        if not p1 and self.arq.entry_pending and not breakin:
            if self.arq.entry_variant == "p4chirp" \
                    and hasattr(self.peer, "send_p4_entry_packet"):
                return self.peer.send_p4_entry_packet(payload, status)
            if self.arq.entry_variant == "p2sl1" \
                    and hasattr(self.peer, "send_p2_entry_packet"):
                return self.peer.send_p2_entry_packet(payload, status)
            if hasattr(self.peer, "send_entry_packet"):
                return self.peer.send_entry_packet(
                    sl, payload, status,
                    acquire=self.arq.entry_variant == "burst")
        if breakin and p1 and hasattr(self.peer, "send_p1_breakin"):
            self._p1_header_inverted = True
            return self.peer.send_p1_breakin(
                payload, self.p1_baud, status & spec.STATUS_SEQ,
                qrt=bool(status & spec.STATUS_QRT))
        if p1 and hasattr(self.peer, "send_p1_packet"):
            # The first packet after BK is #1/head55. That reverses the header
            # phase for the WHOLE sending turn, including repeats: #2 must then
            # carry AA. A one-packet override repeated head55 on new information.
            header = (pactor1.DATA_HEADER if status & 1 else pactor1.SYNC_HEADER)
            if self._p1_header_inverted:
                header ^= 0xFF
            return self.peer.send_p1_packet(
                payload, self.p1_baud, status & spec.STATUS_SEQ, header=header,
                changeover_request=bool(status & spec.STATUS_CHANGEOVER),
                qrt=bool(status & spec.STATUS_QRT))
        # The cycle length is link state the FSM holds, not a flag in the status
        # byte -- bit 5 is the REQUEST, and a long packet can carry it clear
        # (DL6MAA at 47.87 s) while a short one carries it set (16.61 s). The
        # changeover packet is exempt: it is the same 0.81 s shape on both cycle
        # lengths, measured either side of the reference's long train.
        if self.arq.cycle_long and not breakin \
                and hasattr(self.peer, "send_long_packet"):
            return self.peer.send_long_packet(sl, payload, status)
        return self.peer.send_packet(sl, payload, status, breakin=breakin)

    def send_p3_terminal(self, header_bit: int = 1):
        """Render the opt-in terminal marker without queuing host bytes."""
        from .arq import REFUSED
        send = getattr(self.peer, "send_p3_terminal", None)
        if self.protocol is not Protocol.PACTOR3 or send is None:
            return REFUSED
        return send(header_bit)

    def send_cs(self, cs_index: int) -> None:
        """Send a control signal in whatever protocol the link is actually in.

        The two are different waveforms -- PACTOR-1's are 12-bit codewords in
        FSK, PACTOR-3's 20-bit in DBPSK on tones 5 and 12 -- and a link that has
        not been upgraded is still PACTOR-1. Routing every control signal to the
        PACTOR-3 renderer meant answering WS8EOC's PACTOR-1 break-in with a
        waveform it has no reason to be listening for, which is the same as not
        answering. The FSM speaks in logical control signals and should not have
        to know; this is where the protocol is known.

        The alternation is part of that translation in ALL THREE, so every
        renderer is reached through it. A peer with no `send_p1_cs` renders
        nothing at all -- `SimPeer` is a second link layer on an ideal channel,
        not a station -- and takes the FSM's own alphabet unchanged.

        PACTOR-2's codeword is the same twenty bits on a different physical
        layer, so it shares `_counter_cs_for` outright and differs only in which
        seam renders it. That mapping is the packet counter's, which every
        protocol above PACTOR-1 puts in the status byte, so there is nothing
        PACTOR-3 about it but its old name.

        `p3_changeover_p1_cs` is the one place the link's protocol and the
        burst's are allowed to differ, and `_p3_changeover_answer` is the cycle
        it is allowed in.
        """
        if self.peer is None:
            return
        if not hasattr(self.peer, "send_p1_cs"):
            return self.peer.send_cs(cs_index)
        elif self.protocol is Protocol.PACTOR1:
            return self.peer.send_p1_cs(self._p1_cs_for(cs_index))
        elif self.protocol is Protocol.PACTOR2 \
                and hasattr(self.peer, "send_p2_cs"):
            return self.peer.send_p2_cs(self._counter_cs_for(cs_index))
        else:
            index = self._counter_cs_for(cs_index)
            if (self.p3_changeover_p1_cs and self._p3_changeover_answer
                    and index in (CS_ACK, CS_REQUEST)):
                return self.peer.send_cs(index, p1_codeword=True)
            return self.peer.send_cs(index)

    def _answer_link_setup(self, cs_index: int) -> None:
        """The called station answered: bring the link up and send packet #1.

        The caller does not send control signals -- they travel the other way,
        from the receiving station. Answering with one leaves the far end waiting
        for a packet that never arrives, which it reports by repeating its answer
        until it gives up.
        """
        # The answer names the speed, and BOTH answers are accepts. CS1 (index 0)
        # = the 200 Bd redundancy section arrived clean, so the link runs at 200
        # and our first packet carries 20 data bytes; CS4 (index 3) = it did not,
        # and the link runs at 100 with 8. "Wird es fehlerfrei empfangen, wird mit
        # CS1 geantwortet, ansonsten mit CS4, was zur sofortigen
        # Geschwindigkeitsreduzierung auf 100 Baud fuehrt." A verdict on the
        # channel, not on the caller.
        #
        # CS4 here is emphatically NOT a refusal, and reading it as one would have
        # been the more expensive mistake: the protocol has no refusal codeword
        # and no refusal state -- a station that does not want a call says
        # nothing. See docs/protocols/pactor/pactor1-control-signals.md, "A
        # refused call is silence".
        if cs_index == pactor1.CS_ACK_A:
            self.p1_baud = 200
            # AND THE 200 IS ON TRIAL FROM HERE, exactly as the one a CS4 offers
            # is. The answer scored a 240 ms redundancy section, not the 960 ms
            # packet behind it, and an unbroken run of the answer says once per
            # cycle that the packet has not decoded (pactor1-control-signals.md
            # §5). Both sessions that lost a gateway to an unreadable emission
            # spent that run at full rate: WS8EOC answered CS1 six times at zero
            # bit errors on 2026-08-28 and KB5LZK seven on 2026-08-29, and
            # packet #1 went out at 200 Bd on every one of those cycles and on
            # every cycle after.
            self._hispeed_tries = 1
        else:
            self.p1_baud = 100
        self._p1_field(self.p1_baud)
        self.log(f"peer answered (CS{cs_index + 1}) -> link at {self.p1_baud} Bd, "
                 f"first data packet Head=AA count=1")
        self.arq.on_rx_cs(CS_ACK)          # link up, we remain the sending station
        if self.protocol is Protocol.PACTOR1 \
                and self.arq.state == State.CONNECTED:
            # Queued rather than transmitted directly, so the FSM owns it: an
            # unacknowledged packet is retransmitted on every cycle that does not
            # acknowledge it, which IS the fixed time raster PACTOR runs on. Sent
            # straight down to the peer it went out once, nothing was in flight,
            # and the next cycle had nothing to say -- so the far end asked again
            # into silence. A field of IDLE (0x1E) is what a packet with nothing
            # to carry carries.
            # The content is specified, not arbitrary: "Die ersten uebertragenen
            # Zeichen enthalten Informationen ueber maximal verfuegbare
            # Software-Levelnummer, gefolgt vom MASTER-Rufzeichen, abgeschlossen
            # mit CR (Beispiel: 1DF4KV <CR>)." A real off-air packet confirms the
            # form exactly -- W4DNA's first packet reads b"1w4dna\r".
            #
            # Level 1: that is what shrike speaks, and every link starts at the
            # lowest level anyway ("daher wird jede Verbindung zunaechst auf dem
            # kleinsten Level (1) gestartet").
            call = self.mycall.lower() if self.announce_lower else self.mycall
            self._setup_bytes = f"1{call}\r".encode("ascii")
            self.arq.on_host_data(self._setup_bytes)

    def _logical_cs(self, ev) -> int:
        """A decoded control signal as the FSM's logical event.

        The mirror of :meth:`_p1_cs_for`, and it has to exist for the same
        reason: the acknowledgement is the ALTERNATION, so CS1 and CS2 are BOTH
        acknowledgements and differ only in which packet they answer. Passing the
        codeword index straight through made CS2 arrive as a repeat-request, so a
        peer acknowledging on the wrong phase looked like it was rejecting us --
        which it did, on the very first handshake, once the transmit side started
        alternating properly.

        EVERY PROTOCOL ABOVE PACTOR-1 says the same thing and says it with the
        counter, so both get :meth:`_counter_logical_cs` rather than the rules
        below. Those are PACTOR-1's own: its CS4 carries a speed offer with a
        spec clause of its own, and the acknowledgement is read from a REPEAT
        because PACTOR-1's alternation is a local toggle rather than a number on
        the air.
        """
        from .arq import CS_BREAKIN
        proto = getattr(ev, "protocol", None)
        if proto in (Protocol.PACTOR2, Protocol.PACTOR3):
            return self._counter_logical_cs(ev)
        if proto != "PACTOR-1":
            return ev.cs
        held, self._prev_rx_cs = ev.cs == self._prev_rx_cs, ev.cs
        # CS3 IS NOT IN THE ALTERNATION, so the repeat rule below does not reach
        # it: the alternation is how an acknowledgement says WHICH packet it
        # answers, and CS3 answers none -- it is the first 120 ms of the peer's
        # own changeover packet. A second one is therefore a repeated PACKET,
        # which is what a station whose acknowledgement went missing sends
        # ("a repeat of the BK packet is requested with CS2" -- the request
        # travels the other way, and the head stays CS3).
        #
        # MEASURED, WS8EOC 2026-08-09: after this end had already yielded, three
        # more changeover packets arrived with their heads at zero bit errors and
        # every one of them reached the FSM as a repeat request. See
        # `arq.on_rx_cs`, which is where the IRS acts on the head.
        if ev.cs == pactor1.CS_CHANGEOVER:
            self._p1_speedup_pending = False
            self._hispeed_tries = 0
            return CS_BREAKIN
        if ev.cs == pactor1.CS_SPEED:
            # CS4 IS RESOLVED BY THE SPEED THE LINK IS ALREADY RUNNING AT, which
            # is context the sender has, and so there is no state in which it has
            # to guess -- pactor1-control-signals.md sec 4.3. The receiving
            # station "kann immer nach einem fehlerhaften Paket ein CS4 senden,
            # was auf der TX-Seite immer als 'REJECT' interpretiert wird", and it
            # "kann nach jedem richtig empfangenen 100-Bd-Paket ... mit CS4
            # bestaetigen, was den TX zur Umschaltung auf 200 Baud zwingt". At 200
            # the packet is rejected and not acknowledged; at 100 it is
            # acknowledged and the next one goes out at 200.
            #
            # Consecutive CS4s are neither: "CS4 in Folge (ohne zwischenzeitliches
            # richtiges CS) werden als 'Request' interpretiert, also ignoriert."
            # This is also what holds the first-block exception -- the connect
            # answer seeds `_prev_rx_cs`, so a CS4 train behind a CS4 answer asks
            # for the block again rather than offering a speed the link has not
            # earned yet.
            if held:
                return self._request_at_speed()
            if self.p1_baud != 100:
                # "Send it again at 100 Bd" is a speed-down on a link that came up
                # at 200 -- which is every link a gateway answers CS1 -- and
                # ignoring the speed half of it is what `captures/witness.wav`
                # recorded: WS8EOC answered our call CS1 (4 cycles, zero errors),
                # then sent CS4 and repeated it for 25 further cycles at 1.25 s,
                # 30 s of them after our transmitter had stopped, and then
                # identified in CW. It was asking for the same information at half
                # the rate and shrike kept sending it at 200.
                #
                # The re-chunked block that follows is the link's FIRST 100 Bd
                # block, and this CS4 is the REQUEST for it -- so the codeword
                # that takes it is named absolutely rather than by the
                # alternation. See `_p1_first_block`.
                self._fall_back_to_100("REJECT", reset_header=True)
                self._p1_first_block = True
                return CS_REQUEST
            if not self._p1_speedup_armed:
                # "Danach kann bereits wieder ein CS4 als 'speedup'-Signal
                # gesendet werden" -- AFTER a 100 Bd block has been taken, and a
                # link that geared itself down has had none. `held` carries that
                # alone only where the link reached 100 Bd behind a CS4; one that
                # fell back on its own has a CS1 as the codeword before, so the
                # next CS4 read as an offer and the link bounced straight back to
                # the rate it had just left, every fifth cycle, for as long as
                # the station held CS4.
                return CS_REQUEST
            # The speed offer is addressed to the sending station -- it forces
            # "den TX" to switch -- and a station that reads one while it is
            # itself the IRS is in the strand where both ends believe they are
            # receiving. Falling back is safe from either role and rising is not:
            # it would put our own changeover packet on the air at a rate the peer
            # has no reason to be listening for.
            #
            # NOR ON A LINK THAT IS NOT UP. `_end_link` resets `p1_baud` to 100,
            # `_prev_rx_cs` to None and `_p1_speedup_armed` to True -- the exact
            # state that reads the next codeword off the channel as an offer --
            # and this seam runs on every decoded control signal, including while
            # an arm observes after teardown. Three sessions rose a dead link to
            # 200 Bd in their transcripts with `** LINK DOWN **` above the line
            # and nothing keyed under it, and an audit nearly scored all three as
            # speed-ups the peer had commanded.
            #
            # DISCONNECTING IS UP. A goodbye is acknowledged by whatever codeword
            # answers it, CS4 included, and a link that cannot read its own
            # acknowledgement cannot close (`test_qrtack`).
            if self.arq.role == IRS or self.arq.state not in (
                    State.CONNECTED, State.DISCONNECTING):
                return CS_REQUEST
            self.p1_baud = 200
            self._p1_field(200)
            self._hispeed_tries = 1
            self._p1_speedup_pending = True
            # CS4 acknowledges the 100 Bd block in place of its alternating
            # CS1/CS2. hf-pactor's tx_100_cs1 -> tx_100_200_cs2 takes CS1 as
            # the first 200 Bd ACK, and CS2 as the speed-up refusal. Leaving
            # CS1 standing here stalled K0NTS and KB5LZK after login BK on
            # 2026-09-10: both answered the first 200 Bd packet CS1.
            if self._last_rx_cs in (pactor1.CS_ACK_A, pactor1.CS_ACK_B):
                self._last_rx_cs ^= 1
            # CS_ACK AND NOT A REQUEST, because the packet the peer answered is
            # delivered: on a link that came up at 100 -- which is every link a
            # station answers CS4 -- a repeat request here is a sequence that
            # cannot advance past any packet the peer acknowledges, and the link
            # stalls on the one it just carried.
            self.log("peer sent CS4 (speed-up) -> link rises to 200 Bd, "
                     "20-byte field")
            return CS_ACK
        # "Wiederholung des gleichen CS bedeutet 'REQUEST'" -- a repeat of the
        # SAME codeword asks for the block again. Only a CHANGE carries the
        # codeword's own meaning. This is the receive-side mirror of the
        # alternation on transmit, and without it a station that repeats itself is
        # misread every time: a gateway sends the same codeword over and over,
        # shrike read each one as a fresh break-in, handed over a link that had
        # barely been established, and then sat chirping control signals at a
        # station that was asking it to retransmit.
        #
        # A repeat is a Request and NOT a NAK: PACTOR-1 has no NAK codeword and
        # nothing here asks for a speed drop, so the retransmission holds the
        # level it was sent at.
        #
        # The comparison follows acknowledged blocks, including the 100 Bd block
        # that CS4 acknowledged. During the first 200 Bd trial, the same logical
        # ACK instead declines the speed-up: "verschieden vom letzten CS vor dem
        # CS4" (Level-1, Geschwindigkeitserhoehung). It settles no 200 Bd data.
        if self._p1_speedup_pending and ev.cs == self._last_rx_cs:
            self._fall_back_to_100("peer declined the speed-up")
            return CS_REQUEST
        #
        # ONE POSITION IS OUTSIDE THE ALTERNATION, and it is the position a
        # REJECT creates: "CS4 dient als 'REQUEST'-CS fuer den ersten
        # 100-Bd-Block, Bestaetigung erfolgt mittels CS1 oder CS3, danach kann
        # bereits wieder ein CS4 als 'speedup'-Signal gesendet werden." The
        # acknowledgement of that one block is CS1, whatever went before it, so
        # the reference standing from a CS1 connect answer must not veto it.
        # K4MSU on 3595 kHz, `captures/onair-0819-2048`: CS1 answer, three CS4,
        # then eleven CS1 at zero bit errors and `#1 x16` on the air. Held the
        # other way, `captures/onair-0819-2207` is the same gateway answering
        # CS4, where the reference is a CS4 and the first CS1 advances the
        # counter on the cycle it arrives.
        # A counter-zero BK has an absolute answer: CS1 accepts it, CS2 asks
        # for it again (Level-1 Senderichtungswechsel). The previous sending
        # turn's alternation cannot veto that CS1. KB5LZK 2026-09-07 answered
        # our login BK with CS1 repeatedly while we classified every one as
        # REQUEST and never advanced beyond the first seven bytes.
        packet = self.arq._inflight
        if packet is not None and packet.breakin:
            if ev.cs == pactor1.CS_ACK_B:
                return self._request_at_speed()
            if ev.cs == pactor1.CS_ACK_A:
                self._last_rx_cs = None
        repeat = ev.cs == self._last_rx_cs
        self._last_rx_cs = ev.cs
        if self._p1_first_block and ev.cs == pactor1.CS_ACK_A:
            self._p1_first_block = False
        elif repeat:
            return self._request_at_speed()
        self._hispeed_tries = 0
        self._p1_speedup_pending = False
        self._p1_speedup_armed = True
        return CS_ACK

    def _counter_logical_cs(self, ev) -> int:
        """A twenty-bit codeword as the FSM's event, read by the packet counter.

        MEASURED off `PIII_Complete_1.wav`, thirty consecutive confirmations at
        zero bit errors of twenty at mutual distance twelve: the answer to a
        packet whose status-byte counter is EVEN is CS1 and the answer to an ODD
        one is CS2, unbroken through the SL3-SL6 ladder, both cycle lengths, and
        across every changeover. So CS2 is an acknowledgement, and reading it as
        a repeat request deadlocks the link on the first packet a real peer
        confirms -- which is packet 1, the first one there is.

        The counter is what the alternation is FOR, so the counter is what
        resolves it: the codeword whose parity matches the packet in flight
        confirms it, and the other one is the peer still answering the packet
        before, which is "not yet". PACTOR-1 has to infer the same thing from a
        repeat because it has no such number to compare against.

        CS3 to CS6 keep the meanings M.1798 §4 gives them and pass through. Each
        of them ALSO answers this cycle's packet -- the IRS emits exactly one
        control signal per cycle -- and `arq.on_rx_cs` is where that is acted on.
        """
        from .arq import CS_ACK, CS_REQUEST
        seq = self.arq.tx_seq
        if seq is None or ev.cs not in (CS_ACK, CS_REQUEST):
            return ev.cs
        return CS_ACK if ev.cs == (seq & 1) else CS_REQUEST

    def _counter_cs_for(self, cs_index: int) -> int:
        """The FSM's logical control signal as a codeword index, by the counter.

        The transmit half of :meth:`_counter_logical_cs`, and the reason a real
        ISS can read us at all: an IRS that answers every packet CS1 tells the
        peer, on every odd counter, to send that packet again.

        There is no toggle to keep because there is nothing to keep it in step
        with -- the phase is the peer's own counter, arriving in every status
        byte. A duplicate is answered with the codeword the accepted packet took,
        which is "still on that one", and a changeover restarts both the counter
        and the alternation with it.

        SO A REPEAT REQUEST IS THE HELD ACKNOWLEDGEMENT, exactly as in PACTOR-1:
        the codeword is a function of what has been accepted, and a cycle that
        accepted nothing leaves it where it was. There is no separate "again"
        word to key -- CS2 is not one, it is the answer to an odd counter -- and
        keying it against an even packet in flight would confirm a packet that
        never arrived. CS5 is the exception the six meanings do carry: a NAK, and
        one that drops a speed level with it.

        The other five codewords have their own meanings and their own indices,
        which the FSM's alphabet already numbers (`arq.CS_ACK` .. `CS_CYCLE_TOG`
        ARE the six codewords); only the acknowledgement is ambiguous.
        """
        from .arq import CS_ACK, CS_REQUEST
        if cs_index not in (CS_ACK, CS_REQUEST):
            return cs_index
        return CS_REQUEST if self.arq.rx_seq & 1 else CS_ACK

    def _request_at_speed(self) -> int:
        """A repeat request, counted while 200 Bd is still unacknowledged.

        A RUN OF REQUESTS AT 200 Bd IS THE 200 FAILING, and nothing else bounds
        it: the retry budget counts SILENCE, and a codeword at zero bit errors is
        the reverse channel working, so a peer that can hear us at 100 and not at
        200 would ask for the same packet for as long as the session lasted.

        BOTH WAYS A LINK REACHES 200 Bd ARM IT, and for a while only the CS4
        speed-up did, on the reading that a CS1 connect answer's run is "not yet"
        rather than a rate the peer cannot read. It is both, and the run is how
        long "not yet" is allowed to last: memory-ARQ has combined that many looks
        at one 960 ms packet by the fourth cycle, and the reverse channel arriving
        clean every cycle throughout is the channel saying the rate is the
        difference. What the old reading cost is on file twice -- WS8EOC six CS1
        on 2026-08-28, KB5LZK seven on 2026-08-29, packet #1 at 200 Bd on every
        cycle of both, and each gateway then left PACTOR-1 for an emission nothing
        here reads.

        THE COUNT SPENDS DECODED CODEWORDS ONLY, so it cannot fire on a fade or a
        dead band, and a single acknowledgement -- any CHANGE of codeword --
        disarms it. It never reaches a cycle nothing was read in: those are
        `arq._on_nak`'s, where the answer-band sign-off and `max_retries` live.
        """
        if self._hispeed_tries:
            self._hispeed_tries += 1
            if self._hispeed_tries > P1_HISPEED_RETRIES:
                self._fall_back_to_100("no 200 Bd packet acknowledged")
        return CS_REQUEST

    def _fall_back_to_100(self, why: str, *, reset_header: bool = False) -> None:
        self.p1_baud = 100
        self._p1_field(100)
        self._hispeed_tries = 0
        self._p1_speedup_pending = False
        # Level-1, Geschwindigkeitsverminderung: both ends reset the header to
        # 55 after speed-down. Keep that phase for this counter and its repeats.
        seq = self.arq.tx_seq
        if reset_header and seq is not None:
            self._p1_header_inverted = bool(seq & 1)
        # AND THE SPEED-UP IS SPENT WITH IT: "kann nach jedem richtig empfangenen
        # 100-Bd-Paket ... mit CS4 bestaetigen" -- the offer answers a packet
        # that arrived, and a link that has just geared down has had none since.
        # A CS4 at 100 Bd here is the REQUEST for the re-chunked block.
        self._p1_speedup_armed = False
        # The 20 bytes in flight become three 8-byte packets carrying the same
        # information -- "das gerade gesendete, nicht bestaetigte Paket wird
        # verworfen und die in ihm enthaltene Information erneut im
        # 100-Baud-Modus ausgesendet". Requeueing rather than retransmitting is
        # the only way to get the re-chunk; the sequence number is reused, so a
        # peer that did decode the 200 Bd packet throws the repeat away as a
        # duplicate.
        self.arq.rechunk_inflight()
        self.log(f"link falls back to 100 Bd ({why})")

    def _command_100(self, why: str) -> int:
        """CS4 as the receiving station's REJECT: send that again at 100 Bd.

        The mirror of :meth:`_fall_back_to_100`, which is the same clause read
        from the sending side -- "kann immer nach einem fehlerhaften Paket ein
        CS4 senden, was auf der TX-Seite immer als 'REJECT' interpretiert wird",
        pactor1-control-signals.md sec 4.3. This station has acted on a peer's
        CS4 since the link layer existed and has never keyed one, so an IRS whose
        200 Bd packets would not decode answered with the held acknowledgement --
        "send that again, at the rate that just failed" -- until the silence
        budget signed the link off. KB5LZK 40 m 2026-09-17 13:54: thirteen peer
        packets at 200 Bd, two decoded, eleven CS1 keyed, QRT on the eighth
        silent cycle.

        THE LINE IS THE POINT. The far ends here are gateways of unknown build
        and no two PACTOR firmwares are obliged to read this the same way, so
        what we can command is settled on the air and not from the desk: 390 peer
        packets across the working record arrive at 100 Bd, which proves these
        stations TRANSMIT at the rate and not that any of them will drop to it
        when asked. Naming the instant in the transcript is how the next arm
        answers that -- the peer's own rate is in every packet line after it.
        """
        self.log(f"commanding the peer down to 100 Bd (CS4 = REJECT): {why}")
        return pactor1.CS_SPEED

    def _p1_cs_for(self, cs_index: int) -> int:
        """The FSM's logical control signal as a PACTOR-1 codeword index.

        THE SAME RULE :meth:`_counter_cs_for` KEYS FOR EVERY LATER LEVEL, because
        it is the same rule: the acknowledgement is the alternation, the
        alternation is the peer's packet counter, and PACTOR-1 carries that
        counter in bits 0-1 of the status byte exactly as PACTOR-3 does. Even
        counter -> CS1, odd -> CS2, held on the air while no new counter arrives,
        which is what "Wiederholung des gleichen CS bedeutet 'REQUEST'" asks for.

        A FREE-RUNNING TOGGLE CANNOT SURVIVE A CHANGEOVER, and that is what this
        was. `arq.rx_seq` restarts with the numbering it reads; a count of
        acceptances does not, so which codeword answered the peer's counter-0
        packet was a coin flip. Both faces are on file, against the same seven
        bytes: KB5LZK's `RMS Tri` break-in drew CS1 on 2026-08-22 and the whole
        banner crossed, and drew CS2 from VE1YZ on 2026-09-02, which is the
        repeat request -- "eine Wiederholung des BK-Paketes wird mittels CS2
        angefordert" -- and the peer sent those seven bytes 45 more times.

        THE SEEDS BOTH FALL OUT OF THE COUNTER, and they are different seeds.
        `arq._enter_connected` counts from one, so a called station that has
        accepted nothing reads counter 0 and keys CS1 -- the connect answer, and
        §5 holds it every cycle until the caller's first packet decodes.
        `arq._reset_rx_seq` counts from zero, so an IRS that has accepted nothing
        SINCE A CHANGEOVER reads counter 3 and keys CS2, which is hf-pactor's
        `tx_rx_100` verbatim: CS2 in every cycle it waits for the changeover
        packet, CS1 on the cycle it decodes. EVERY LEVEL WAITS ON CS2 for the
        same reason: the packet it is waiting for carries counter 0, CS1 is that
        packet's acknowledgement, and keying it before one decodes confirms a
        field that never arrived.

        CS3 and CS4 have codewords of their own and answer no packet, so they
        pass through and leave the alternation where it stands.

        CS5 HAS A CODEWORD AT 200 Bd AND NONE AT 100. A NAK is the one event
        meaning a packet reached the decoder and would not come out of it, and
        sec 4.3 gives the receiving station exactly one thing to say about that:
        CS4, which a sender at 200 Bd reads as REJECT and answers by re-chunking
        the information at 100 (`_command_100`). At 100 Bd the same codeword is
        the speed-UP offer -- the rate is what disambiguates it -- and there is
        nothing below to ask for, so the request is the held codeword as before.
        """
        from .arq import CS_BREAKIN, CS_CYCLE_TOG, CS_NAK, CS_REQUEST, CS_SPEED_UP
        if cs_index == CS_BREAKIN:
            return pactor1.CS_CHANGEOVER
        if cs_index in (CS_SPEED_UP, CS_CYCLE_TOG):
            return pactor1.CS_SPEED
        if cs_index == CS_NAK:
            if self.p1_baud == 200:
                return self._command_100("a packet would not decode at 200 Bd")
            cs_index = CS_REQUEST
        # AND A RUN OF CYCLES THAT DECODE NOTHING IS THE SAME FINDING ABOUT THE
        # SAME CHANNEL, arrived at with no packet to fail a CRC: the demodulator
        # produced no frame at all, so nothing reached `on_rx_packet` and the
        # cycle answers with the held acknowledgement. That is the state that
        # cost KB5LZK, and it is exactly the evidence `_request_at_speed` spends
        # in the other direction -- a run at 200 Bd IS the 200 failing -- so it
        # is spent here on the same count, `P1_HISPEED_RETRIES`, and the fifth
        # cycle asks.
        #
        # THE AIRTIME IS IDENTICAL. A codeword is keyed in this cycle either way,
        # so this is a substitution and not an addition, and the protocol bounds
        # the rest itself: "CS4 in Folge (ohne zwischenzeitliches richtiges CS)
        # werden als 'Request' interpretiert", which is what the held codeword
        # already means. The first one acts and every repeat degrades to the
        # cycle we would have keyed anyway -- which is why it goes out every
        # cycle rather than once, on a channel that is losing transmissions.
        #
        # ON EVIDENCE OF A PEER AND NOT ON SILENCE ALONE. "Nothing decoded"
        # cannot tell a fading station from a departed one, and a codeword keyed
        # at a departed one is noise into an empty channel. `arq.peer_reads` is
        # the floor: at least one frame of the peer's read on THIS link, and
        # since the run of dead cycles is consecutive and any decode resets it,
        # the reading it stands on is at most this many cycles old. A station
        # that is fading has it (KB5LZK carried four codewords in 27 s); a link
        # that never read the far end at all does not.
        #
        # NOT `note_peer_raster_burst`, which the ISS spends for the same kind of
        # refutation. Its comb is corroborated by three CRC frames, so on a link
        # where nothing decodes it is the reading we do not have; it names no
        # station, where a decode names the protocol and the counter; and it is
        # consumed in `_on_nak` alone, so read from here it would latch. The
        # ranking is `note_peer_heard`'s own.
        if (cs_index == CS_REQUEST and self.p1_baud == 200
                and self.arq.role is IRS and self.arq.peer_reads
                and self.arq._silent_cycles >= P1_HISPEED_RETRIES):
            return self._command_100(
                f"{self.arq._silent_cycles} cycles decoded nothing at 200 Bd, "
                f"{self.arq.peer_reads} frames read on this link")
        return self._counter_cs_for(cs_index)

    def connected(self, mycall: str, dxcall: str) -> None:
        self._link_up = True
        self.channel(self.ptchn).link.append(f"({self.ptchn}) CONNECTED to {dxcall}")
        if self.app is not None:
            self.app.link_up()

    def app_turns(self, *, allow_breakin: bool = True) -> None:
        """The link's turn discipline for an attached application, once a cycle.

        A hostmode client works the channel itself — %O when it has said its
        piece, %I when it has something to say — and an attached app has
        nobody to press those keys. The discipline is the same one an operator
        runs, driven by the one thing this layer can see, the buffer: an ISS
        whose data is sent and confirmed owes the peer its turn, and an IRS
        holding an answer takes the channel to give it. B2F is request and
        response all the way down, so "buffer drained" is exactly "our turn is
        over". Idempotent by way of the ARQ's own pending flags, so calling it
        every cycle is the intended use.

        A receive-first application may withhold its automatic break-in while
        waiting for a reply to become ready. Queued link setup bytes alone do
        not establish that the application has an answer. This does not cancel
        explicit host commands, handing over a drained buffer, or teardown.
        """
        # B2F can queue its final FQ and become done in the same callback.
        # An IRS must still take the turn to send it. _txbuf includes bytes
        # already in flight and falls only as the peer acknowledges them.
        if (self.app is None or self.arq.state != State.CONNECTED
                or (self.app.done and self._txbuf == 0)):
            return
        if self.arq.role == ISS and self._txbuf == 0:
            self.arq.on_host_over()
        elif allow_breakin and self.arq.role == IRS and self._txbuf > 0:
            self._drop_stale_setup()
            self.arq.on_host_breakin()

    def _drop_stale_setup(self) -> None:
        """The changeover carries the application's first bytes, not the link's.

        The announcement is the PACTOR-1 phase's own business -- level and
        callsign, queued by `_answer_link_setup`. Behind an application that has
        an answer ready it is neither: the changeover
        packet's field is three bytes, and a gateway waiting for a login gets
        `1w9` instead of `;FW`. 0913-2143 keyed those three bytes eight times
        with the whole B2F login queued behind them.

        ONLY WITH SOMETHING BEHIND IT, which is what makes this a reordering
        rather than a loss: an empty changeover is worse than a stale one, and
        before the greeting completes the announcement is still the only thing
        the link has to say. The bytes are unsent -- `drop_queued_prefix`
        refuses anything that has been offered to the peer -- so nothing the
        far end has seen is affected.
        """
        setup = self._setup_bytes
        if not setup or self._txbuf <= len(setup):
            return
        if self.arq.drop_queued_prefix(setup):
            self._setup_bytes = b""
            self.log(f"the application has an answer -> {len(setup)} unsent "
                     f"link-setup bytes dropped ahead of it")

    def defer_rx_close(self) -> bool:
        return (self.protocol == Protocol.PACTOR3
                and bool(getattr(self.peer, "defer_p3_cs", False)))

    def on_rx_ack_emitted(self) -> None:
        self.arq.on_rx_ack_emitted()

    def on_cs_emitted(self, cs_index: int) -> None:
        self.arq.on_cs_emitted(cs_index)

    def disconnected(self) -> None:
        """The link is down, and everything that described it goes down with it.

        A CONNECT is not a fresh process. `onair` runs one call per run, but the
        hostmode `C` command does not -- it is how Winlink Express and Pat drive
        this modem, and they call station after station down a channel list on one
        PtcHost. Every field below is about a link rather than about this station:
        which protocol the packets are in, which rate the PACTOR-1 phase runs at, which
        upgrades the far end refused, which half of the CS1/CS2 alternation is
        owed, what the last codeword was. Held across a teardown, the next call
        opens in the previous contact's waveform -- rendering PACTOR-3 at a station
        whose connect burst was PACTOR-1, chopping the callsign announcement to the
        wrong field size, and reading a stranger's codeword through a protocol this
        link never had. `PactorArq._finish_disconnected` already does exactly this
        for its own half; this is the other half, and the seam it calls.

        `_ruled_out` is per link by its own comment -- "a peer that cannot follow
        says so by answering in PACTOR-1, and that says nothing about the next
        station we call" -- which was true of the comment and not of the code.
        """
        peer = self.arq.dxcall or "?"
        # A call that never came up did not disconnect, it failed, and that is
        # the distinction a client's retry logic reads [10.4.10].
        event = ("DISCONNECTED fm" if self._link_up else "LINK FAILURE with")
        self.channel(self.ptchn).link.append(f"({self.ptchn}) {event} {peer}")
        self._txbuf = 0
        self._end_link()

    def _end_link(self) -> None:
        """Forget everything that was true of the link that just ended.

        `protocol` is the one with teeth, because three readers downstream take
        it as this call's waveform:

          * `onair._SessionRx._readers` puts it at the head of the frame scan --
            the reader that runs on the audio the cycle's answer hangs on -- so a
            call following an upgraded link scans PACTOR-3 first for an answer
            that can only arrive as a PACTOR-1 codeword;
          * `send_cs` and `send_packet` render in it, and a call is a PACTOR-1
            exchange in both directions until it is upgraded;
          * `_answer_link_setup` queues the spec's mandatory first data packet
            only on a PACTOR-1 link, so a call that DID get its answer then
            transmits nothing and the peer repeats until it gives up.

        The link is what ends, not the station: `stay_in_pactor1` is the
        operator's and stays set.
        """
        cancel = getattr(self.peer, "cancel_pending_cs", None)
        if cancel is not None:
            cancel()
        self._ruled_out.clear()
        self.p1_baud = 100
        self.protocol = Protocol.PACTOR1
        self._phase_power(Protocol.PACTOR1, "the link ended")
        self._p3_fallback_reported = False
        # A grant is spent once per contact, not once per modem process.
        # Hostmode clients reuse this object when calling the next gateway.
        self._grant_taken = self._grant_pending = False
        self._uninvited_entry = False
        self._hispeed_tries = 0
        self._p1_header_inverted = False
        self._p1_speedup_pending = False
        self._p1_first_block = False
        self._p1_speedup_armed = True
        self._setup_bytes = b""
        self._last_rx_cs = self._prev_rx_cs = None
        self._link_up = False
        self._disconnect_asked = False
        # "The value is reset at the end of each connection" [10.4.35 %T]. It is
        # the host's register that restarts, not the counter itself: `sent_total`
        # is what `onair._Link` reads once a cycle to tell a link that is still
        # moving from one that has gone quiet, and a counter that rewinds at
        # teardown reads there as a final cycle of traffic. It pushed a hold's
        # idle deadline twelve cycles past the goodbye that had already been sent.
        self._sent_base = self.sent_total
        # The channel goes back to channel 0's callsign after a disconnect [10.4.5].
        self._callsigns.pop(self.ptchn, None)

    def deliver(self, blob: bytes) -> None:
        self.rcvd_total += len(blob)
        if self.app is not None:
            self.app.on_link_data(blob)
        else:
            self.channel(self.ptchn).rx += blob
        # The channel buffer serves a hostmode client; an onair session has none,
        # so without this line received data reached nobody -- a link could carry
        # a whole message and the operator never saw a byte of it. The link layer
        # has already decoded the wire coding, so this is the message itself.
        self.log(f"rx data {len(blob)}B: {blob.decode('cp437')!r}")

    def buffer(self, nbytes: int) -> None:
        if nbytes < self._txbuf:
            self.sent_total += self._txbuf - nbytes
        self._txbuf = nbytes

    def log(self, msg: str) -> None:
        self.log_lines.append(msg)


# --------------------------------------------------------------------------- #
# A stand-in remote station, until the PACTOR-3 transmitter lands.

class SimPeer:
    """A remote station on an ideal channel: a second :class:`PactorArq` at the
    far end, decoding what we send and answering with real control signals.

    `reply` is the far end's application: what it returns is short-circuited into
    the host's receive queue, which keeps a request/response test one-directional
    and cheap. :meth:`send` is the honest path -- the far end queues the data on
    its own link layer and transmits it once it holds the channel, so a test can
    drive a real changeover in both directions.
    """

    def __init__(self, reply=None, greeting: bytes = b""):
        self.host: Optional[PtcHost] = None
        self.received = bytearray()
        self.reply = reply                  # callable(bytes) -> bytes | None
        self.greeting = greeting            # sent once the far end sees the link
        self._far = PactorArq(_FarIO(self))
        self._far.on_host_listen(True)
        self._to_far: list[tuple] = []
        self._to_near: list[tuple] = []

    def attach(self, host: PtcHost) -> None:
        self.host = host

    @property
    def role(self) -> Optional[str]:
        return self._far.role

    def send(self, blob: bytes) -> None:
        """Queue data at the far end, to go out when it holds the channel."""
        self._far.on_host_data(blob)

    def breakin(self) -> None:
        """Far end grabs the channel (CS3)."""
        self._far.on_host_breakin()

    def over(self) -> None:
        """Far end hands the channel back once its buffer is confirmed."""
        self._far.on_host_over()

    @property
    def sending(self) -> bool:
        """Bytes still queued or in flight at the far end."""
        return bool(self._far._outbuf) or self._far._inflight is not None

    def connect_burst(self, mycall, dxcall):
        self._to_far.append(("connect", (mycall, dxcall,
                                         self.host.arq.connect_variant)))

    def send_packet(self, sl, payload, status, breakin=False):
        # The physical header describes this emission; status requests the next.
        self._to_far.append(("packet", (sl, payload, status, True, breakin,
                                        None, self.host.arq.cycle_long)))

    def send_cs(self, cs_index):
        self._to_far.append(("cs", (cs_index,)))

    def cycle(self) -> None:
        """Clock the far end's cycle grid, then settle the exchange."""
        self._far.on_cycle()
        self.pump()

    def pump(self) -> None:
        for _ in range(200):
            if not (self._to_far or self._to_near):
                return
            for queue, fsm in ((self._to_far, self._far), (self._to_near, self.host.arq)):
                while queue:
                    kind, args = queue.pop(0)
                    getattr(fsm, {"connect": "on_rx_connect", "packet": "on_rx_packet",
                                  "cs": "on_rx_cs"}[kind])(*args)
        raise RuntimeError("SimPeer did not settle")

    def _far_connected(self) -> None:
        if self.greeting:
            self.host.deliver(self.greeting)

    def _far_delivered(self, blob: bytes) -> None:
        self.received += blob
        if self.reply is not None:
            answer = self.reply(blob)
            if answer:
                self.host.deliver(answer)


class _FarIO(ArqIO):
    def __init__(self, peer: SimPeer):
        self.peer = peer

    def connect_burst(self, mycall, dxcall):
        self.peer._to_near.append(("connect", (mycall, dxcall,
                                               self.peer._far.connect_variant)))

    def send_packet(self, sl, payload, status, breakin=False):
        self.peer._to_near.append(("packet", (sl, payload, status, True, breakin,
                                             None, self.peer._far.cycle_long)))

    def send_cs(self, cs_index):
        self.peer._to_near.append(("cs", (cs_index,)))

    def connected(self, mycall, dxcall):
        self.peer._far_connected()

    def deliver(self, blob):
        self.peer._far_delivered(blob)


# --------------------------------------------------------------------------- #
# Serial front end

def serve(host: PtcHost, device: Optional[str] = None, *, cycle_s: float = 1.25,
          verbose: bool = True) -> None:
    """Run `host` on a pty (or on `device`, an existing serial port) until closed."""
    import os
    import select
    import termios
    import tty

    if device:
        fd = os.open(device, os.O_RDWR | os.O_NOCTTY)
        name = device
    else:
        fd, replica = os.openpty()
        name = os.ttyname(replica)
        tty.setraw(replica)
    tty.setraw(fd, termios.TCSANOW)

    # A modem on a wire is a modem that answers: Listen defaults to 1 [6.49,
    # 10.4.31], and a client that wants it off says so. Until this was applied a
    # served PtcHost took no inbound call until some client sent %L1, which
    # neither of the two drivers we can read ever sends.
    host.set_listen(True)

    if verbose:
        print(f"PTC-IIIusb (fw {FIRMWARE}) on {name}   MYCALL {host.mycall} "
              f"PTCHN {host.ptchn}")
    os.write(fd, host.open())

    next_tick = time.monotonic() + cycle_s
    try:
        while True:
            timeout = max(0.0, next_tick - time.monotonic())
            ready, _, _ = select.select([fd], [], [], timeout)
            if ready:
                chunk = os.read(fd, 4096)
                if chunk:
                    out = host.feed(chunk)
                    if out:
                        os.write(fd, out)
            if time.monotonic() >= next_tick:
                host.tick()
                next_tick += cycle_s
    except KeyboardInterrupt:
        pass
    finally:
        os.close(fd)


def main() -> int:
    """Serve the PTC emulation on a pty. A named function so it can be a console
    script; `python -m hfmodem.shrike.ptc` still reaches it through the block below."""
    import argparse

    ap = argparse.ArgumentParser(description="shrike as an SCS PTC-IIIusb")
    ap.add_argument("--device", help="bind an existing serial device instead of a pty")
    ap.add_argument("--mycall", default="N0CALL")
    ap.add_argument("--cycle", type=float, default=1.25, help="ARQ cycle seconds")
    args = ap.parse_args()

    # The stand-in station greets like a Winlink RMS because, as far as B2F is
    # concerned, it is one: a real answering session out of hfmodem.winlink,
    # so a client on the pty can run an actual mail exchange end to end. It
    # used to be a canned SID string and an echo, which looked like a gateway
    # for exactly one line.
    from ..winlink import B2FSession
    rms = B2FSession("SHRIKE", role="answering", target=args.mycall)
    peer = SimPeer(reply=rms.feed, greeting=rms.start())
    serve(PtcHost(peer, mycall=args.mycall), args.device, cycle_s=args.cycle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
