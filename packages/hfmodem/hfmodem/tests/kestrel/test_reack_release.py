# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The turn we are owed at the end of the peer's delivery, and how it is asked for.

A queue of our own used to replace the acknowledgement of the peer's LAST over
with a turn-request. Whether that frame also acknowledges the over it answers is
in no recording, and on 2026-09-08 it did not: the peer's release follows our
acknowledgement [vara_frames, SESSION_TURN_RELEASE_RESPONDER], so with none keyed
no release came, the gateway fell into its own idle cadence, and the three asks
that followed went out on our ~10 s clock — two of them across the peer's
transmissions, and the one it heard clean drew the grant 0.19 s later.

So the over draws the frame it asks for, and the ask waits for the peer's own
listening gap: the control burst again first, because the likeliest reason no
release came is that the acknowledgement was lost, and then the turn-request,
which takes the channel rather than waiting for it.

Nothing here opens a device or keys a radio.
"""
from __future__ import annotations

import numpy as np

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel.test_reack_over import (CALLED, answering, idle,
                                                   keyed, over)

_REPLY = b"FC EM ABCDEF 100 90 0\rF> 3b\r"


def _delivery_ends(reply: bytes = _REPLY):
    """The peer's closing over, with the host answering into the delivery."""
    hs, io = answering(reply)
    hs.on_rx_audio(over(b"KC9GHZ-2 DE W9SSJ QTC 4\r"))
    return hs, io


def _awaiting_release():
    hs, io = _delivery_ends()
    assert hs._answer_owed == VA._OWED_RELEASE, io.msgs
    return hs, io


# --------------------------------------------------------------------------- #
def test_the_last_over_is_acknowledged_and_the_ask_is_held():
    """The whole of event 1 of 2026-09-08, stated as a law: the acknowledgement
    goes out, the queue stays, and nothing takes the channel yet."""
    hs, io = _awaiting_release()
    assert keyed(io) == "control", io.msgs
    assert hs.turn == VA._TURN_PEER
    assert hs._txq, "the host's reply was thrown away with the ask"
    assert any("release is owed" in m for m in io.msgs), io.msgs


def test_an_unasked_release_ends_it_with_nothing_asked():
    """What a stock responder does 0.13-0.15 s behind our control burst, three
    arms of 2026-08-29: it hands the channel over, and the ask was never needed."""
    hs, io = _awaiting_release()
    hs.on_rx_audio(MK.synth_burst(CALLED, VF.SESSION_TURN_RELEASE_RESPONDER))
    assert hs.turn == VA._TURN_OURS, io.msgs
    assert hs._answer_owed is None
    assert hs._asked == 0, "asked for a turn the peer had already handed over"
    assert any("tx DATA over" in m for m in io.msgs), io.msgs


def test_the_first_rung_is_the_acknowledgement_again():
    """No release in the peer's cadence: the likeliest reason is the burst that
    draws one, so that is what goes out — into the gap the idle just opened."""
    hs, io = _awaiting_release()
    hs.on_rx_audio(idle())
    assert keyed(io) == "control", io.msgs
    assert hs.turn == VA._TURN_PEER, "the channel was taken on the first rung"
    assert any("re-keying the 11-symbol control burst" in m for m in io.msgs), io.msgs


def test_the_ask_follows_in_the_next_gap():
    """And then the channel is taken rather than waited for. 21 of 21 grants at a
    stock responder came from an ask that ended inside the peer's own gap; every
    one of the 14 ignored asks was still on the air when it keyed."""
    hs, io = _awaiting_release()
    hs.on_rx_audio(idle())
    hs.on_rx_audio(idle())
    assert keyed(io) == "ask", io.msgs
    assert hs.turn == VA._TURN_ASKED
    assert hs._asked == 1
    assert any("asking for the turn (1/3)" in m for m in io.msgs), io.msgs


def test_the_asks_are_charged_to_the_one_budget():
    """The turn-request is the same burst whichever clock keys it, so a ladder
    that asked on its own account would spend a budget twice over."""
    hs, io = _awaiting_release()
    for _ in range(VA._REACK_MAX + 2):
        hs.on_rx_audio(idle())
    assert hs._asked <= VA._TURN_MAX_ASK, io.msgs
    assert hs._answer_owed is None, "the ladder ran on past its own budget"
    assert any("never released the turn" in m for m in io.msgs), io.msgs


def test_a_grant_ends_the_ladder():
    """The peer answering the ask is the end of the whole question: the turn is
    ours and there is nothing left to be owed."""
    hs, io = _awaiting_release()
    hs.on_rx_audio(idle())
    hs.on_rx_audio(idle())
    assert hs.turn == VA._TURN_ASKED
    hs.on_rx_audio(np.concatenate([np.zeros(MK.FS // 4),
                                   MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)]))
    assert hs.turn == VA._TURN_OURS, io.msgs
    assert hs._answer_owed is None
    assert hs._reacks == 0


def test_an_empty_queue_still_repeats_the_final_ack_without_asking_for_turn():
    """N5TW's incomplete greeting could not queue a host reply. That does not
    remove the final-ACK obligation when the peer missed the first copy."""
    hs, io = _delivery_ends(reply=b"")
    assert hs._answer_owed == VA._OWED_RELEASE, io.msgs
    before = len(io.sent)
    hs.on_rx_audio(idle())
    hs.on_rx_audio(idle())
    assert len(io.sent) == before + 2, io.msgs
    assert keyed(io) == "control"
    assert hs._asked == 0
