# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Saying an acknowledgement again when the peer did not take it.

An acknowledgement is one burst into one turnaround and nothing repeated it. On
2026-09-08 KE8LVA proposed four messages, keyed seven full overs of the first,
and did not take our answer to the seventh — keyed at the same +0.07 s as the
eleven it did take, into a channel a narrowband occupant was using where that
frame's tones fall. The gateway idled at 3.55 s for 57 s with the rest of the
mailbox in hand and closed; the field its overs carry counts four or more full
overs still to come at every one of the seven. Three messages never arrived.

Nothing in the state machine could reach that: no state recorded that an over had
been answered and another was owed, the cadence fell to keepalive-a/b — which a
stock responder answers 0 of 6 — and the re-ack, the NAK and the ask were all
unreachable from it. What this file holds is the ladder that answers it, keyed
into the peer's own listening gap rather than on our ~10 s clock.

The rungs are the three continue-class frames and then the NAK. They differ in
length by a factor of four (0.34 s, 0.68, 1.37), so a peer that read none of one
may read another, and the NAK is what a stock sender answers by re-sending the
over rather than by waiting out its own timer.

Nothing here opens a device or keys a radio.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel.test_data_over_gate import _HostIO, _connected

MYCALL, CALLED = "W9SSJ", "KC9GHZ"

#: Bursts of two-tone symbols, which carry no callsign and are named by their
#: tones alone  [see vara_frames].
_PAIRS = {
    VA.OVER_CONTINUE_CAPTURED: VF.OVER_CONTINUE_CALLER_2300,
    "control": VF.CONTROL_BURST_CALLER_2300,
    "nak": VF.NAK_CALLER_2300,
}
#: And the generated ones, which are keyed to a callsign.
_KEYED = {
    VA.OVER_CONTINUE_GENERATED: (VF.SESSION_OVER_RESPONSE, CALLED),
    VA.OVER_CONTINUE_SHORT: (VF.SESSION_OVER_RESPONSE_SHORT, CALLED),
    "ask": (VF.SESSION_TURN_REQUEST, MYCALL),
}


def keyed(io) -> str:
    """What the last burst on the air was, by its own symbols."""
    x = io.sent[-1]
    n = round(len(x) / MK.HOP)
    pairs = [tuple(sorted(p)) for p in MK.demod_tone_pairs(x, n)]
    for name, want in _PAIRS.items():
        if pairs == [tuple(sorted(p)) for p in want]:
            return name
    tones = MK.demod_tones(x, n)
    for name, (kind, call) in _KEYED.items():
        if (n == len(kind.preamble) + kind.n_payload
                and tones == VF.handshake_tones(call, kind)):
            return name
    return f"unnamed {n}-symbol burst"


def idle(kind: VF.BurstKind = VF.SESSION_RESPONDER_OVER_IDLE) -> np.ndarray:
    """One burst of the gateway's own idle cadence."""
    return MK.synth_burst(CALLED, kind)


def over(payload: bytes, index: int = 0) -> np.ndarray:
    """One DATA over of the peer's, framed the way a sender frames it."""
    return tx.synth_burst(_phy.vara_body(payload, MYCALL), over=index)


def _field_over(payload: bytes, field: int, index: int) -> np.ndarray:
    """The same, carrying a named per-frame field  [see `arq.phy.over_field`]."""
    return tx.synth_burst(_phy.vara_body(payload, MYCALL, tail=field), over=index)


def answering(reply: bytes = b"", **kw):
    """A connected station mid-session, its host holding ``reply``."""
    return _connected(io=_HostIO(reply), **kw)


def stream(hs, x: np.ndarray, lead: float = 0.5, trail: float = 2.0) -> None:
    """Feed one burst to the raw-stream route, in the transport's own blocks."""
    x = np.concatenate([np.zeros(int(lead * MK.FS)), x,
                        np.zeros(int(trail * MK.FS))])
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])


def _mid_delivery(setting: str | None = None):
    """One intermediate over answered: the state the peer's next burst arrives in."""
    hs, io = answering(over_continue=setting)
    hs.on_rx_audio(over(b"A" * 89))
    assert hs._answer_owed == VA._OWED_OVER, io.msgs
    return hs, io


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("setting", VA.OVER_CONTINUE_ANSWERS)
def test_an_answered_intermediate_over_leaves_an_answer_owed(setting):
    """The state that did not exist. `_tx_over_response` set nothing, so the peer
    going quiet mid-delivery was indistinguishable from a peer with nothing to
    say."""
    hs, io = _mid_delivery(setting)
    assert hs._reack_frame == setting
    assert hs._reacks == 0
    assert hs._peer_over == 1


@pytest.mark.parametrize("setting", VA.OVER_CONTINUE_ANSWERS)
def test_the_first_rung_is_the_frame_that_answered_the_over(setting):
    """The peer may have missed one burst on a channel somebody else is using; the
    cheapest hypothesis is that the frame was right and the copy was lost."""
    hs, io = _mid_delivery(setting)
    hs.on_rx_audio(idle())
    assert keyed(io) == setting, io.msgs
    assert hs._reacks == 1
    assert any("re-acknowledging over #1 (1/4)" in m for m in io.msgs), io.msgs


@pytest.mark.parametrize("setting", VA.OVER_CONTINUE_ANSWERS)
def test_the_ladder_walks_the_other_two_frames_and_then_the_nak(setting):
    """Four rungs, four different bursts. The three continue-class frames are
    0.34, 0.68 and 1.37 s long, so which of them a peer reads through an occupant
    is not something this end can know in advance; the NAK is the one frame a
    stock sender answers by re-keying the over  [see test_vara_nak]."""
    hs, io = _mid_delivery(setting)
    rungs = []
    for _ in range(VA._REACK_MAX):
        hs.on_rx_audio(idle())
        rungs.append(keyed(io))
    assert rungs[0] == setting
    assert set(rungs[:3]) == set(VA.OVER_CONTINUE_ANSWERS), io.msgs
    assert rungs[3] == "nak", io.msgs
    assert hs._reacks == VA._REACK_MAX


def test_both_frames_of_the_peers_cadence_open_the_gap():
    """683 is what a responder with nothing left keys; 745 what it keys between
    finishing an over and its peer's answer to it. 745 was in the codebook and
    read by nothing, which is the frame KE8LVA spent 57 s on."""
    for kind in (VF.SESSION_RESPONDER_IDLE, VF.SESSION_RESPONDER_OVER_IDLE):
        hs, io = _mid_delivery()
        hs.on_rx_audio(idle(kind))
        assert any(kind.name in m for m in io.msgs), io.msgs
        assert hs._reacks == 1, (kind.name, io.msgs)


def test_the_rung_goes_out_on_the_stream_route_too():
    """The two routes must not disagree about a stall: the live transport feeds
    the raw stream and the loopback brackets, and the peer is the same peer."""
    hs, io = _mid_delivery()
    stream(hs, idle())
    assert keyed(io) == hs._reack_frame, io.msgs
    assert hs._reacks == 1


def test_an_idle_named_late_is_not_a_gap_to_key_into():
    """The gap is 1.7-1.9 s and a frame named a second behind the newest audio is
    one the peer has already keyed past. A rung into that lands on the peer's next
    transmission — which is what the operator heard twice on 2026-09-08 — so the
    scan that arrives late says so and keys nothing  [see `_GRANT_FRESH_S`]."""
    hs, io = _mid_delivery()
    late = np.concatenate([np.zeros(MK.FS // 4), idle(), np.zeros(int(1.5 * MK.FS))])
    before = len(io.sent)
    hs.on_rx_stream(late)                    # one late scan over the whole gap
    assert any("session-responder-over-idle" in m for m in io.msgs), io.msgs
    assert hs._idle_held > VA._GRANT_FRESH_S, hs._idle_held
    assert len(io.sent) == before, io.msgs
    assert hs._reacks == 0


def test_an_over_that_decodes_ends_the_ladder():
    """The peer sending on is the acknowledgement having landed: a sender that was
    not acknowledged repeats instead."""
    hs, io = _mid_delivery()
    hs.on_rx_audio(idle())
    assert hs._reacks == 1
    hs.on_rx_audio(over(b"B" * 89, 1))
    assert hs._reacks == 0
    assert hs._answer_owed == VA._OWED_OVER, "the new over owes its own answer"
    assert hs._peer_over == 2
    assert io.host == [b"A" * 89, b"B" * 89], io.msgs


def test_the_over_the_nak_draws_back_is_not_delivered_twice():
    """What the last rung costs: a stock sender answers the NAK by re-sending, and
    the over it re-sends is one the host already has  [see `_deliver`]."""
    hs, io = _mid_delivery()
    for _ in range(VA._REACK_MAX):
        hs.on_rx_audio(idle())
    assert keyed(io) == "nak"
    hs.on_rx_audio(over(b"A" * 89))
    assert io.host == [b"A" * 89], io.msgs
    assert any("repeats one already delivered" in m for m in io.msgs), io.msgs


def test_a_nak_from_the_peer_keys_the_next_rung():
    """The other way the peer says it did not read us. With an over of ours
    outstanding a NAK asks for that over again; with none outstanding the only
    thing it can be failing on is the answer we keyed  [see `_took_nak`]."""
    hs, io = _mid_delivery()
    assert hs._tx_pending is None
    hs.on_rx_audio(np.concatenate([np.zeros(int(0.3 * MK.FS)),
                                   MK.synth_tone_pairs(VF.NAK_RESPONDER_2300),
                                   np.zeros(int(0.25 * MK.FS))]))
    assert any("did not read our answer" in m for m in io.msgs), io.msgs
    assert keyed(io) == hs._reack_frame, io.msgs
    assert hs._reacks == 1


def test_the_peers_field_says_how_far_the_delivery_has_to_run():
    """Read for the log and for nothing else. The seven overs of 2026-09-08 all
    carried 0x99, which is where the countdown saturates: the stall was nowhere
    near the end of the mailbox and the transcript said so nowhere.

    The first over of a delivery is not on the countdown at all — three bench
    arms gave 0x95, 0x99 and 0x19 for it — so the second is the earliest one that
    can be placed  [see `arq.phy.overs_after`].
    """
    hs, io = answering()
    hs.on_rx_audio(_field_over(b"A" * 89, 0x95, 0))
    assert hs._overs_hint is None, "the first over of a delivery was placed"
    io.msgs.clear()
    hs.on_rx_audio(_field_over(b"B" * 89, 0x99, 1))
    assert hs._overs_hint == _phy.OVERS_AFTER_MAX
    assert any("the peer says the end of the delivery is not near" in m
               for m in io.msgs), io.msgs
    io.msgs.clear()
    hs.on_rx_audio(_field_over(b"C" * 89, 0x89, 2))
    assert hs._overs_hint == 0
    assert any("the peer says this is its last full over" in m
               for m in io.msgs), io.msgs


@pytest.mark.parametrize("field", [0x9d, 0x19, 0x00])
def test_a_field_off_the_countdown_says_nothing(field):
    """Anything the arithmetic cannot place is not guessed at."""
    hs, io = answering()
    hs.on_rx_audio(_field_over(b"A" * 89, 0x95, 0))
    io.msgs.clear()
    hs.on_rx_audio(_field_over(b"B" * 89, field, 1))
    assert hs._overs_hint is None
    assert not any("the peer says" in m for m in io.msgs), io.msgs


@pytest.mark.parametrize("field", [0x81, 0x89])
def test_both_bottom_steps_are_the_last_full_over(field):
    """The ladder has two floors and the difference between them is the record
    the closing over comes at, not a count: `0x81` announces a close at record 2
    and `0x89` one at the base level  [see `arq.phy.close_level`,
    `vara_arq._frame_field`]."""
    hs, io = answering()
    hs.on_rx_audio(_field_over(b"A" * 89, 0x95, 0))
    io.msgs.clear()
    hs.on_rx_audio(_field_over(b"B" * 89, field, 1))
    assert hs._overs_hint == 0
    assert any("this is its last full over" in m for m in io.msgs), io.msgs
