# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the re-acknowledgement ladder costs, and what stops it.

Every rung is a transmission into a shared band, so the ladder is bounded twice
over: by its own budget, and by the give-up budget that closes a link nobody is
answering on. The two have to agree — a ladder longer than the link's life would
close the session mid-recovery and put the failure in a silence instead of in the
log, which is the shape of every VARA session this file's neighbours were written
for.

The peer's own turnaround is the clock. The ~10 s cadence is the backstop behind
it and not a second ladder: a rung keyed in the peer's gap and another keyed on
our clock a moment later is two bursts into one turnaround, which is the one
thing 118 changes of transmitter across three complete sessions never show.

Nothing here opens a device or keys a radio.
"""
from __future__ import annotations

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.tests.kestrel.test_reack_over import (_mid_delivery, answering,
                                                   idle, keyed, over)


def test_the_ladder_fits_inside_the_give_up_budget():
    """Stated rather than trusted: the link must outlive its own recovery."""
    assert VA._REACK_MAX < VA._MAX_WITHOUT_PROGRESS


def test_the_ladder_says_when_it_is_spent():
    """A peer that cannot hear us must not be keyed at forever, and the operator
    reads which over it was and how much of the delivery was still behind it."""
    hs, io = answering()
    hs.on_rx_audio(tx.synth_burst(
        _phy.vara_body(b"A" * 89, "W9SSJ", tail=0x95), over=0))
    hs.on_rx_audio(tx.synth_burst(
        _phy.vara_body(b"B" * 89, "W9SSJ", tail=0x99), over=1))
    for _ in range(VA._REACK_MAX + 1):
        hs.on_rx_audio(idle())
    assert hs._answer_owed is None
    assert hs._reacks == 0
    assert any("over #2 unacknowledged after 4 attempt(s) (the peer had said the "
               "end of the delivery is not near)" in m for m in io.msgs), io.msgs


def test_a_spent_ladder_asks_for_the_turn_when_the_host_is_waiting():
    """The queue is not thrown away with the ladder: taking the channel is the
    other way to move a session the peer has stopped moving."""
    hs, io = answering(b"FC EM ABCDEF 100 90 0\rF> 3b\r")
    hs.on_rx_audio(over(b"A" * 89))
    assert hs._txq, io.msgs
    for _ in range(VA._REACK_MAX + 1):
        hs.on_rx_audio(idle())
    assert keyed(io) == "ask", io.msgs
    assert hs.turn == VA._TURN_ASKED


def test_the_cadence_carries_the_ladder_when_the_peer_says_nothing():
    """A peer whose cadence this build cannot name still owes an answer, and a
    keepalive is not what asks for one: a stock responder answers the keepalives
    0 of 6."""
    hs, io = _mid_delivery()
    hs.idle_keepalive()
    assert keyed(io) == hs._reack_frame, io.msgs
    assert hs._reacks == 1
    assert hs._since_progress == 1, "the rung was not charged to the give-up budget"


def test_a_rung_keyed_in_the_peers_gap_is_not_keyed_again_on_the_cadence():
    """One burst per turnaround. The cadence comes round on its own clock and the
    ladder has already spoken in the gap the peer opened."""
    hs, io = _mid_delivery()
    hs.on_rx_audio(idle())
    assert hs._reacks == 1
    at = len(io.sent)
    hs.idle_keepalive()
    assert hs._reacks == 1, io.msgs
    assert len(io.sent) == at, io.msgs
    assert not hs._keyed_on_peer_burst
    hs.idle_keepalive()
    assert hs._reacks == 2, "a quiet peer must not freeze the cadence fallback"


def test_the_ladder_does_not_hold_a_dead_link_open():
    """The whole reason the ladder is bounded. A peer that keys its idle cadence
    and nothing else is what a stalled session looks like, and the give-up budget
    closes it on the same tick it always did  [see test_vara_giveup]."""
    hs, io = _mid_delivery()
    for tick in range(1, 200):
        hs.idle_keepalive()
        hs.on_rx_audio(idle())
        if hs.state is not VA.VaraState.CONNECTED:
            break
    assert tick == VA._MAX_WITHOUT_PROGRESS + 1, io.msgs
    assert any("closing the stalled link" in m for m in io.msgs), io.msgs
