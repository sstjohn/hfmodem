# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""When a VARA session stops, and what a new call does not inherit from the old.

TWO FAULTS, ONE SHAPE. Both are a value outliving the thing it described.

On 2026-08-16 this station called KC9GHZ, exchanged three CRC-clean data overs,
and then the gateway stopped transmitting. `working/s2-vara-kc9ghz-force.log`
holds what followed: three turn-requests, twenty-three keepalives, twenty-six
keyings in all, and not one `rx` line after line 74. The idle cadence held a link
open on the strength of its own transmissions, into a shared 40 m channel the
operator could hear was empty, until a person stopped it — the log has no
session-end line at all, and there was no timeout to fire. So the thing that
described the link, `state == CONNECTED`, outlived the link.

The other is the same sentence about a contact: `_delivered_body` names what the
host last received, the turn names whose transmission is next, and `originate`
carried both into the following call. A gateway greets every session with its SID
line, so a second call to the same station sends a body this end has already seen
— dropped as a repeat, and the exchange dies on its first frame.

Nothing here opens a device, and nothing keys a radio.
"""
from __future__ import annotations

import numpy as np

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

_MYCALL = "W9SSJ"
_CALLED = "KC9GHZ"


class _IO(VA.VaraIO):
    """Records what was keyed, and can refuse to key at all."""

    def __init__(self, transmits: bool = True):
        self.transmits = transmits
        self.msgs: list[str] = []
        self.sent: list[np.ndarray] = []
        self.delivered: list[bytes] = []

    def key(self, on): ...

    def tx(self, samples):
        if self.transmits:
            self.sent.append(np.asarray(samples, float))

    def tx_went_out(self) -> bool:
        return self.transmits

    def pending(self): ...

    def connected(self, *a): ...

    def data(self, payload): self.delivered.append(bytes(payload))

    def log(self, msg): self.msgs.append(msg)


def _connected(transmits: bool = True):
    io = _IO(transmits)
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def _greeting(text: bytes = b";FW: W9SSJ\rKC9GHZ-2 DE W9SSJ QTC 0\r") -> bytes:
    """The body a Winlink gateway opens every session with — the same bytes each
    time, which is exactly why a delivered body must not outlive its contact."""
    return _phy.vara_body(text, _MYCALL)


def _idle_bursts(hs) -> int:
    """Idle-cadence ticks until the link is no longer up, driven the way the
    session loop drives it. Bounded well above the budget so a regression is a
    failed assertion rather than a hung test."""
    for tick in range(1, 200):
        hs.idle_keepalive()
        if hs.state is not VA.VaraState.CONNECTED:
            return tick
    return -1


def test_the_idle_cadence_stops_keying_at_a_peer_that_has_gone():
    """#104, from the state the KC9GHZ session was left in."""
    hs, io = _connected()
    ticks = _idle_bursts(hs)
    assert ticks == VA._MAX_WITHOUT_PROGRESS + 1, (
        f"the link was still up after {ticks} idle bursts with nothing heard")
    assert hs.state is VA.VaraState.DISCONNECTED
    assert any("closing the stalled link" in m for m in io.msgs), io.msgs
    # And the close is a close, not a pause: the cadence that comes round again
    # after it must not put the station back on the air.
    io.sent.clear()
    hs.idle_keepalive()
    hs.idle_keepalive()
    assert not io.sent, "kept keying the idle cadence after giving the link up"


def test_the_turn_request_train_is_charged_to_the_same_budget():
    """The route the real session took. Three turn-requests, the turn handed back,
    and then keepalives — one cadence, and one bill for all of it."""
    hs, io = _connected()
    hs.send(b"x" * 89)
    assert hs.turn == VA._TURN_ASKED
    ticks = _idle_bursts(hs)
    assert ticks <= VA._MAX_WITHOUT_PROGRESS + 2, (
        f"{ticks} idle bursts: the turn-request train paid nothing into the budget")
    assert any("giving the turn back" in m for m in io.msgs), io.msgs
    assert hs._txq, "the queue was thrown away rather than left for the operator"


def test_a_burst_the_transmitter_refused_is_not_charged():
    """A budget spends transmissions, not intentions: a burst the transport
    declined put nothing on the air for the peer to answer, so the silence that
    followed is not the peer's.

    Read on a branch that KEYS, which since the peer's turn went silent means
    holding the turn with a queue: there the cadence keys the turn-idle frame, and
    a refused one must not be charged. The peer's-turn tick is a different clock —
    it keys nothing, so there is no transmission to credit or refuse, and what it
    counts is the peer's own silence  [see idle_keepalive].
    """
    hs, _ = _connected(transmits=False)
    hs.turn, hs._txq = VA._TURN_OURS, [b"x" * 89]
    for _ in range(VA._MAX_WITHOUT_PROGRESS * 3):
        hs.idle_keepalive()
    assert hs.state is VA.VaraState.CONNECTED
    assert hs._since_progress == 0


def test_an_answered_idle_holds_the_link_open():
    """The peer answering our keepalive is an answer to something we sent, and a
    live VARA answers every one: ten occurrences across the two off-air gateway
    sessions, 0.112-0.150 s behind our burst. A link like that is not dead."""
    hs, _ = _connected()
    answer = MK.synth_burst(_CALLED, VF.SESSION_IDLE_RESPONSE)
    for _ in range(VA._MAX_WITHOUT_PROGRESS * 3):
        hs.idle_keepalive()
        hs.on_rx_audio(answer)
        assert hs.state is VA.VaraState.CONNECTED, "dropped a link the peer answers"


def test_a_peer_that_says_only_idle_is_not_progress():
    """The other half of the distinction, and the one that cost 100 s at the bench
    on 2026-08-31: after a reply over that would not decode, the responder keyed
    `session-responder-idle` eighteen times. Read as progress each one zeroed the
    budget, so the session had no route out but the mail timeout. The frame says
    the peer is transmitting, which is what a stalled session looks like too."""
    hs, io = _connected()
    idle = MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE)
    for tick in range(1, 200):
        hs.idle_keepalive()
        for i in range(0, len(idle), VA._STREAM_BLOCK):
            hs.on_rx_stream(idle[i:i + VA._STREAM_BLOCK])
        if hs.state is not VA.VaraState.CONNECTED:
            break
    assert any("session-responder-idle" in m for m in io.msgs), io.msgs
    assert tick == VA._MAX_WITHOUT_PROGRESS + 1, (
        f"idle repeats bought {tick} idle bursts and held a stalled link open")


def test_a_repeated_over_is_traffic_and_not_progress():
    """The distinction the whole clock turns on. A gateway resending the body we
    already delivered is transmitting, and the session is not moving: an activity
    clock reads that as a healthy link and keys at it until somebody intervenes."""
    hs, io = _connected()
    body = _greeting()
    hs._deliver([body])
    assert io.delivered, "the first delivery of a body must reach the host"
    for _ in range(VA._MAX_WITHOUT_PROGRESS):
        hs._deliver([body])                      # answered again, never re-delivered
    assert len(io.delivered) == 1
    assert hs._since_progress == 0, "a delivery is progress and must clear the count"
    ticks = _idle_bursts(hs)
    assert ticks == VA._MAX_WITHOUT_PROGRESS + 1, (
        f"repeats of a delivered body bought {ticks} idle bursts")


def test_a_new_call_does_not_inherit_the_last_contacts_delivered_body():
    """A gateway's SID greeting is the same bytes every session, so the duplicate
    filter left standing across a call drops the first frame of the next one."""
    hs, io = _connected()
    greeting = _greeting()
    hs._deliver([greeting])
    assert len(io.delivered) == 1

    hs.originate(_CALLED)
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs._deliver([greeting])
    assert len(io.delivered) == 2, (
        "the new session's greeting was suppressed as a repeat of the old one's")


def test_a_new_call_does_not_inherit_the_turn_and_keys_nothing_in_the_peers_turn():
    """A connect starts with the turn at the peer — the gateway speaks first in
    every session on record. Inheriting `ours` keys turn-idle at a station that is
    mid-greeting and reads its overs as intrusions.

    And in the peer's turn a stock caller keys nothing on its own clock: the
    unprompted keepalive cadence is gone, and the peer's own over-idle draws the
    single reactive keepalive instead  [see idle_keepalive, _stream_answer]."""
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs._asked, hs._into_our_turn, hs._since_progress = 2, 2, 4
    hs._over = 5

    hs.originate(_CALLED)
    assert hs.turn == VA._TURN_PEER
    assert (hs._asked, hs._into_our_turn, hs._since_progress, hs._over) == (0, 0, 0, 0)

    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    io.msgs.clear()
    io.sent.clear()
    hs.idle_keepalive()
    assert not io.sent, (
        f"a fresh session keyed on its own clock in the peer's turn: {io.msgs}")


def test_a_call_in_progress_keeps_the_queue_the_host_wrote_ahead():
    """The one thing `originate` must NOT throw away. Bytes written before the
    link came up are the caller's own, `_connected` transmits them, and the
    connect-request ladder re-enters `originate` on every retry."""
    hs, _ = _connected()
    hs.state = VA.VaraState.DISCONNECTED
    hs.send(b"x" * 89)
    queued = len(hs._txq)
    assert queued
    hs.originate(_CALLED)
    hs.originate(_CALLED)                        # CR #2, the same attempt
    assert len(hs._txq) == queued, "a connect-request retry discarded the queue"
