# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CHAT caller at the answerer's last over, and the empty turn handed back.

A VARA HF caller says idle with frame 1 at the answerer's last over and
traffic with frame 0 (BW500 pairs 2026-10-18; s2s-2300 and s2s-2750 50.9 s,
2026-10-22), and holds the empty turn the answerer hands back, polling it
11.15 s on. Kestrel's caller said traffic at every close, and the answerer
granted it an over it did not have. Kestrel answering Kestrel, OpenSlot air
loop and the pair harness, 2026-11-10.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock as _Clock, _IO as _IO
from .test_responder_control_stream import feed

CALLER, CALLED = 'W9SSJ-14', 'W9SSJ-15'
BWS = ['500', '2300', '2750']


def caller(bw, *, turn=A._TURN_PEER):
    io = _IO()
    hs = A.VaraStationHandshake([CALLER], io, bw=bw, chat=True)
    hs.role, hs.called, hs.caller = 'initiator', CALLED, CALLER
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, turn
    return hs, io


def answered(bw):
    """The answerer's last over answered with our idle."""
    hs, io = caller(bw)
    hs._tx_over_response(last=True)
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(CALLER, bw, 1)[0]))
    return hs, io


def hand_back(bw, high):
    return (M.synth_tone_pairs(F.responder_post_delivery_idle(CALLED, bw)) if high
            else M.synth_burst(CALLED, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER_SHORT, bw)))


@pytest.mark.parametrize('bw', BWS)
@pytest.mark.parametrize('queued', [False, True])
def test_a_chat_caller_says_idle_or_traffic_at_the_answerers_last_over(bw, queued):
    hs, io = caller(bw)
    if queued:
        hs._queue_host_payload(b'caller text')
    hs._tx_over_response(last=True)
    frame = M.synth_tone_pairs(F.control_bursts(CALLER, bw, int(not queued))[0])
    assert np.array_equal(io.sent[-1], frame), io.msgs[-3:]


@pytest.mark.parametrize('bw', BWS)
@pytest.mark.parametrize('high', [False, True], ids=['f20', 'idle-pair'])
def test_the_answerers_empty_hand_back_is_held_and_polled_on_its_cadence(bw, high):
    hs, io = answered(bw)
    sent = len(io.sent)
    feed(hs, np.r_[np.zeros(4800), hand_back(bw, high), np.zeros(9600)])
    assert hs.turn == A._TURN_OURS and hs._answer_owed is None, io.msgs[-3:]
    assert len(io.sent) == sent, io.msgs[-3:]
    assert hs.idle_poll_due_in() == pytest.approx(A._IDLE_POLL_FIRST_S, abs=.5)


@pytest.mark.parametrize('bw', BWS)
def test_text_written_before_the_hand_back_asks_and_goes_out_on_the_drained_frame(bw):
    """s2s-2300 73.0 s: from a turn it holds VARA HF asks with frame 0 and
    delivers behind the answerer's drained frame."""
    hs, io = answered(bw)
    hs._queue_host_payload(b'caller text')
    feed(hs, np.r_[np.zeros(4800), hand_back(bw, False), np.zeros(9600)])
    assert hs._tx_pending is None and hs.turn == A._TURN_ASKED, io.msgs[-3:]
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(CALLER, bw, 0)[0]))
    drained = M.synth_burst(CALLED, F.for_bw(F.SESSION_DRAINED_RESPONDER, bw))
    feed(hs, np.r_[np.zeros(4800), drained, np.zeros(9600)])
    assert hs._tx_pending is not None, io.msgs[-3:]


@pytest.mark.parametrize('bw', BWS)
def test_a_stale_hand_back_is_not_taken(bw):
    hs, io = answered(bw)
    hs._ans_due = 10 ** 9
    feed(hs, np.r_[np.zeros(4800), hand_back(bw, True), np.zeros(M.FS)])
    hs._ans_due = 0
    feed(hs, np.zeros(960))
    assert hs.turn == A._TURN_PEER, io.msgs[-3:]


@pytest.mark.parametrize('high', [False, True], ids=['f20', 'idle-pair'])
def test_a_transport_that_only_brackets_takes_the_hand_back(high):
    hs, io = answered('500')
    hs.on_rx_audio(np.r_[np.zeros(4800), hand_back('500', high), np.zeros(5120)])
    assert hs.turn == A._TURN_OURS and hs._answer_owed is None, io.msgs[-3:]


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_the_answers_tail_is_read_before_our_last_over_is_settled(bw):
    """Named off its preamble, the answer's frame came one scan later than its
    tail: taken unread, it drew our release for traffic instead of the f20."""
    hs, io = caller(bw, turn=A._TURN_OURS)
    hs.send(b'73')
    assert hs._tx_pending is not None
    answer = M.synth_tone_pairs(F.control_bursts(CALLER, bw, 1)[1])
    feed(hs, np.r_[np.zeros(24000), answer, np.zeros(12000)])
    assert hs._tx_pending is None and hs.turn == A._TURN_PEER, io.msgs[-3:]
    assert np.array_equal(io.sent[-1], M.synth_burst(
        CALLED, F.for_bw(F.SESSION_TURN_IDLE_SHORT, bw))), io.msgs[-3:]


@pytest.mark.parametrize('bw', BWS)
def test_the_hand_back_behind_a_repeated_idle_is_taken(bw):
    """Our idle unheard, the answerer's release never came and we keyed it
    again: the hand-back behind the repeat is the one we asked for."""
    hs, io = answered(bw)
    hs._tx_control_burst(0)
    assert not hs._hand_back_due
    hs._tx_control_burst(1)
    assert hs._hand_back_due
    feed(hs, np.r_[np.zeros(4800), hand_back(bw, False), np.zeros(9600)])
    assert hs.turn == A._TURN_OURS, io.msgs[-3:]


def test_a_hand_back_read_behind_audio_still_to_come_is_stale():
    hs, io = answered('500')
    burst = np.r_[np.zeros(4800), hand_back('500', False), np.zeros(960)]
    hs.on_rx_stream(np.r_[burst, np.zeros(2 * M.FS)])
    assert hs.turn == A._TURN_PEER, io.msgs[-3:]
