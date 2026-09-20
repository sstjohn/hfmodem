# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One rule for a frame that arrived and would not read, held against every stack.

The distinction the modems were missing is three-way and they each collapsed it to
two. The channel was empty; a frame arrived and failed to decode; a frame arrived
in a modulation we do not implement. On 2026-08-18 all three met the middle case
and answered it differently and wrongly — besra ACKed it as a duplicate, shrike
called it "no decodable traffic from the peer" and dropped a live link, kestrel
logged its own evidence and went silent — and an ACK, an abort and a silence are
all wrong for the same reason: none of them is what the protocol has for "say that
again".

What follows from the distinction is each protocol's own business, which is why
this is a shared TEST and not a shared class. The stacks do not even agree on
whether a frame can be ADDRESSED — no PACTOR data packet or codeword carries an
address at all — so a common superclass would have to abstract over that and would
get it wrong. What they have in common is a rule and the need to be held to it:

  * a frame this modem could not read draws a protocol-legal repeat request,
    not silence, not an abort, and never an acknowledgement;
  * the asking is bounded on a link this receiver has never read, and the frames
    below arrive on exactly that link: an unbounded repeat train at a peer we
    have got nothing from is the keepalive problem with a different frame in it,
    and it spends a shared band and the peer's transmitter as well as ours. What
    a stack does when the link IS delivering is its own business and outside this
    rule — besra read that case as its own deafness and hung up on W6IDS on
    2026-08-26, four frames short of a message it had already accepted;
  * a frame that does read clears the count, because a link that loses one frame
    to a fade is a working link — and the frame that clears it below is one that
    reads without DELIVERING, because a delivery holds besra's link by the rule
    above whatever the count is doing and would carry that arm on its own.

A stack with nothing protocol-legal to key for "say that again" is not excused
from the rule; it is RECORDED against it. ``_NO_REPEAT_REQUEST`` carries the
reason, the arm is still held to everything else the rule asks — an empty
turnaround and never an acknowledgement, never an abort inside the budget, and a
ladder that closes the link rather than leaving the peer transmitting to nobody —
and the entry is asserted rather than described, so a stack that acquires a
repeat request fails the test that says it has none. Silence that a reader can
find the reason for is a measured limitation; silence a suite was relaxed around
is the 2026-08-18 fault again with a passing report on top of it.

The register is empty. It held kestrel at BW2300, on the ground that no burst
available there could be shown to be a NAK, and the entry named the measurement
that would settle it: a real VARA pair driven into an over that will not decode,
the responder's turnaround read against its answer to one that does. That bench
ran on 2026-09-07 — two stock VARA HF 4.9.0 on the virtual cables, noise into a
receiver's input until it registered a burst and failed CRC — and the turnaround
is an 8-symbol two-tone burst the sender answers by dropping a speed level and
re-sending. kestrel keys it, so the entry is struck and kestrel is graded on the
asking like everything else.

Each stack brings its own adapter. Nothing here opens a device, a port or a rig.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest


@dataclass(frozen=True, slots=True)
class Stack:
    """One modem's answer to the three questions this rule is made of."""

    name: str
    #: Each of these acts on an open link with this station in the receiving role.
    unreadable: Callable[[], None]      #: deliver a frame that arrived and failed
    readable: Callable[[], None]        #: deliver one that decoded
    repeats: Callable[[], int]          #: repeat requests sent so far
    keyed: Callable[[], int]            #: transmissions of any kind so far
    holding: Callable[[], bool]         #: still a link we will go on answering on
    budget: int


def _besra() -> Stack:
    from hfmodem.besra.arq.session import (BREAK, _DATA_LADDER, _UNDELIVERED_BUDGET,
                                           _is_datanak)
    from hfmodem.tests.besra.test_arq_robustness import Pair

    pair = Pair()
    pair.connect()
    # The caller comes up as the ISS, and this rule is about receiving: the
    # responder breaks in and sends, which leaves the caller the IRS.
    iss, irs = pair.responder, pair.caller
    block = b"the peer"                 # one frame's worth at the base rung
    # No clock between the delivery and the turnover. A peer that chirps IDLE
    # first has settled the payload, and besra clears it there so a later copy
    # of the same bytes is the new delivery it is; the frame `readable` puts
    # back below is one this end answered into a fade, from an ISS deposed
    # before it ever heard about it.
    iss.queue_data(block)
    pair.pump()
    assert bytes(pair.obs_a.received) == block == irs._last_rx_payload
    # besra grades its budget against what the CURRENT stint has read, so the
    # frames below have to arrive on one that has read nothing — this end answers
    # and the peer takes the link straight back, which is the turnover a mail
    # client makes on every forward.
    irs.queue_data(b"and this end answers")
    pair.pump()
    assert bytes(pair.obs_b.received)
    irs.on_receive(BREAK, b"", irs._session, True)
    assert irs._last_rx_type < 0 and irs._last_rx_payload == block
    even = _DATA_LADDER[500][0][0]      # the base rung: what an IRS meets first
    marks = len(pair.relay.log)

    def unreadable() -> None:
        # Never the type we last ACKed: that one is the peer asking whether its
        # frame landed, which is a different question with a different answer.
        ft = even + 1 if irs._last_acked_type == even else even
        irs.on_receive(ft, b"", irs._session, False)

    def readable() -> None:
        # The frame this stint has already answered, put back on the air: a
        # deposed ISS keeps the one it never heard an ACK for, and this stint
        # meets it again under a type that proves nothing. It READS, so it clears
        # the count; it dedupes rather than delivering (`_is_replay` first, then
        # type alternation), so the count is the only thing left holding the link.
        irs.on_receive(even, block, irs._session, True)

    return Stack(
        "besra", unreadable, readable,
        lambda: sum(1 for who, ft in pair.relay.log[marks:]
                    if who == "W9SSJ" and _is_datanak(ft)),
        lambda: sum(1 for who, _ in pair.relay.log[marks:] if who == "W9SSJ"),
        lambda: irs.connected and not irs._disc_repeating,
        # The stint above is built to have read nothing, and besra grades an
        # undelivered stint against the longer of its two budgets.
        _UNDELIVERED_BUDGET)


def _shrike() -> Stack:
    from hfmodem.shrike.arq import (CS_BREAKIN, CS_NAK, SEQ_MOD, UNREPAIRED_BUDGET,
                                    ArqConfig, ArqIO, PactorArq)

    class _IO(ArqIO):
        def __init__(self):
            self.cs: list[int] = []
            self.gone = False

        def send_cs(self, i):
            self.cs.append(i)

        def connect_burst(self, mycall, dxcall):
            pass

        def connected(self, mycall, dxcall):
            pass

        def disconnected(self):
            self.gone = True

        def log(self, msg):
            pass

    io = _IO()
    arq = PactorArq(io, ArqConfig())
    arq.on_host_connect("W9SSJ", "WS8EOC")
    arq.on_rx_cs(CS_BREAKIN)                       # the peer takes the link
    seq = [0]

    def readable() -> None:
        arq.on_rx_packet(1, b"hi", seq[0] % SEQ_MOD, True)
        seq[0] += 1

    return Stack(
        "shrike", lambda: arq.on_rx_packet(1, b"", 0, False), readable,
        lambda: sum(1 for i in io.cs if i == CS_NAK),
        lambda: len(io.cs),
        lambda: not io.gone and not arq._qrt_pending,
        UNREPAIRED_BUDGET)


def _kestrel() -> Stack:
    """kestrel reaches the same rule by its own route, and the route is the point.

    Its evidence is a reference-column count: 16 of 24 with a failed CRC is
    positive proof a real over hit the receiver, which is the middle case named
    outright. `_peer_data_over` returns the `_UNDECODED` sentinel rather than the
    None it uses for "no over here", so the two stop arriving at `_answer_data_over`
    as the same value, and the budget hangs off the difference.

    One thing the rule must not be read into, and kestrel is right not to: it
    leaves its echo window OPEN on an unreadable over. Such an over names nobody —
    our own transmission back through the rig's monitor included — so clearing the
    bodies we keyed on the strength of it would let outgoing mail return as
    received mail. The contract is about what goes on the air and what the deadline
    counts. It is never about identity, because the frame at issue has none.

    Driven at the seam the other two are driven at: the recogniser's verdict,
    injected. Synthesising an over that identifies and will not decode is the
    receive chain's own test (`test_data_over_gate`), not this rule's.

    The arm is BW2300 because that is the bandwidth kestrel connects gateways on
    and the only one whose receiver can reach the middle case at all. What it
    answers an unreadable over with there is the role-keyed 8-symbol two-tone
    burst measured off the cables on 2026-09-07; BW500's NAK is corroborated
    too, but its over recogniser is BW2300-only, so at BW500 an over that will
    not decode is not identified as an over and the middle case cannot arise.
    """
    import numpy as np

    from hfmodem.kestrel.vara import vara_arq as VA
    from hfmodem.tests.kestrel.test_data_over_gate import _connected

    hs, io = _connected(bw="2300")
    verdict: list[object] = [VA._UNDECODED]
    hs._peer_data_over = lambda samples: [verdict[0]]
    quiet = np.zeros(int(3.0 * VA.MK.FS))

    def unreadable() -> None:
        verdict[0] = VA._UNDECODED
        hs._answer_data_over(quiet)

    def readable() -> None:
        # Full length, as a real over is: `arq.phy.payload` indexes the unpadded
        # body while padding a copy, so a short one raises IndexError rather than
        # reading short.
        verdict[0] = b"an over that decoded".ljust(90, b"\0")
        hs._answer_data_over(quiet)

    return Stack(
        "kestrel", unreadable, readable,
        lambda: sum(1 for m in io.msgs if m.startswith("tx NAK")),
        lambda: len(io.sent),
        lambda: hs.state == VA.VaraState.CONNECTED,
        VA._OVER_NAK_MAX)


STACKS = ("besra", "shrike", "kestrel")
_BUILD = {"besra": _besra, "shrike": _shrike, "kestrel": _kestrel}

#: The stacks that have no repeat request to key, and what is missing. An entry
#: here is a measured capability gap, not a waiver: the arm still runs, and the
#: rest of the rule still binds it. Empty since kestrel acquired one — see above.
_NO_REPEAT_REQUEST: dict[str, str] = {}
_ASKS = tuple(n for n in STACKS if n not in _NO_REPEAT_REQUEST)


@pytest.fixture(params=STACKS)
def stack(request) -> Stack:
    return _BUILD[request.param]()


@pytest.fixture(params=_ASKS)
def asking_stack(request) -> Stack:
    """A stack the protocol gives something to say. See `_NO_REPEAT_REQUEST`."""
    return _BUILD[request.param]()


@pytest.fixture(params=sorted(_NO_REPEAT_REQUEST))
def silent_stack(request) -> Stack:
    """One it does not, for as long as the entry against it stands."""
    return _BUILD[request.param]()


def test_an_unreadable_frame_draws_a_repeat_request(asking_stack: Stack):
    """Not silence and not an acknowledgement. Silence stalls a peer that is
    waiting to be told; an acknowledgement tells it the payload landed, and it
    moves on with the bytes still aboard — which is how 64 bytes of a Winlink
    forward block went missing and the parser met `0xb1` where it wanted SOH."""
    before = asking_stack.repeats()
    asking_stack.unreadable()
    assert asking_stack.repeats() == before + 1, (
        f"{asking_stack.name} did not ask for the frame again")
    assert asking_stack.holding(), (
        f"{asking_stack.name} gave up on the first unreadable frame")


def test_a_stack_with_no_repeat_request_keys_nothing(silent_stack: Stack):
    """The other half of the rule, for a stack whose protocol gives it nothing to
    ask with. The turnaround is left EMPTY: not an acknowledgement, and not a
    waveform invented to fill it either — one of those was keyed at a live gateway
    on 2026-08-20 and the gateway answered it the way it answers noise. The peer's
    own repeat timer is what recovers the over, and the ladder still closes the
    link, so the arm is graded on both.

    This assertion runs in the direction the entry can go stale in. A stack that
    acquires a real repeat request fails here rather than passing quietly, and the
    fix is to strike its `_NO_REPEAT_REQUEST` entry, which puts it back in front of
    the test above."""
    before = silent_stack.keyed()
    for _ in range(silent_stack.budget):
        silent_stack.unreadable()
    assert silent_stack.repeats() == 0, (
        f"{silent_stack.name} keyed a repeat request, and the limitation recorded "
        f"against it is stale:\n  {_NO_REPEAT_REQUEST[silent_stack.name]}")
    assert silent_stack.keyed() == before, (
        f"{silent_stack.name} put something on the air in a turnaround it has no "
        "frame for")
    assert silent_stack.holding(), (
        f"{silent_stack.name} gave up inside its own budget")

    silent_stack.unreadable()
    assert not silent_stack.holding(), (
        f"{silent_stack.name} left the peer transmitting to nobody past "
        f"{silent_stack.budget} overs it could not read")


def test_the_asking_is_bounded(stack: Stack):
    """Nothing on this link has ever read, so there is nothing to set against the
    count: a peer we have got nothing from is one to stop asking, rather than
    trade repeat requests with for nothing on a frequency somebody else could be
    using.

    The bound is the ladder and not the asking, which is why every stack is held to
    it: one that keys nothing counts the same overs and closes on the same rung,
    and a peer whose repeat timer is doing the work still has to be told when this
    receiver has stopped."""
    for _ in range(stack.budget):
        stack.unreadable()
    assert stack.holding(), f"{stack.name} stopped inside its own budget"

    stack.unreadable()
    assert not stack.holding(), (
        f"{stack.name} went on asking past {stack.budget} unanswered repeats")


def test_the_asking_stops_with_the_link(asking_stack: Stack):
    """The rung that closes the link is a close and not one more repeat request:
    a stack that spends its budget and asks again on the way out is asking a peer
    it has just stopped listening to."""
    for _ in range(asking_stack.budget + 1):
        asking_stack.unreadable()
    assert asking_stack.repeats() == asking_stack.budget, (
        f"{asking_stack.name} kept asking on the way out")


def test_a_frame_that_reads_clears_the_count(stack: Stack):
    """The count is consecutive. A link that loses one frame an hour to a fade is
    a working link and must not accumulate its way to a disconnect."""
    for _ in range(stack.budget * 4):
        stack.unreadable()
        stack.readable()
    assert stack.holding(), f"{stack.name} disconnected a link that was repairing"
