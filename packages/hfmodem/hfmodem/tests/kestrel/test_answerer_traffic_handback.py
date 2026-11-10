# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CHAT answerer with text queued, against a caller that has traffic too.

Kestrel answering Kestrel over the OpenSlot air loop, clean channel,
2026-11-09. Our level-6 over drew the caller's frame 0, we granted the turn
with 31 bytes still queued, and 0.3 s on keyed the 1.39 s turn-request across
the caller's over: 37 of its 112 symbols went under our own transmission. Under
CHAT a VARA HF station answers with frame 0 when it has traffic, and a VARA HF
receiver never asks while an over of the peer's is owed: the sender takes the
ask as that over's acknowledgement (2026-10-21, 19 VARA HF pairs).

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock as _Clock
from .test_responder_control_stream import feed, station


def _requests(io) -> int:
    ask = M.synth_burst(io.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, '500'))
    return sum(len(s) == len(ask) and np.allclose(s, ask) for s in io.sent)


def _over(hs, payload: bytes, host: int) -> np.ndarray:
    level = phy.level_key('500', host)
    body = phy.vara_body(payload, hs.caller, body_len=phy.body_size('500', level), level=level)
    return A.OF.data_over_tx(body, level=level, bw='500',
                             lead=TX.VB6Rnd(F.lead_state(hs.caller, True, '500')))


def _answered(frame: int, expect: int = 4):
    hs, io = station()
    io.caller = hs.caller
    hs.chat = hs._chat_profile = True
    if expect > 4:
        hs._link_high_speed, hs._expect_level = True, expect
    hs.turn = A._TURN_OURS
    hs.send(b'K9TST last over')
    assert hs._tx_pending is not None and A._phy.over_is_last(hs._tx_pending[0], hs.caller)
    hs.send(b'more text queued behind it')
    assert hs._txq
    answer = M.synth_tone_pairs(F.control_bursts(hs.caller, '500', frame)[0])
    feed(hs, np.concatenate([np.zeros(4800), answer, np.zeros(4800)]))
    assert hs._tx_pending is None and hs.turn == A._TURN_PEER, io.msgs[-4:]
    return hs, io


def test_frame_zero_on_the_last_over_grants_the_callers_traffic():
    hs, io = _answered(0)
    assert hs._granted_over_unread(), io.msgs[-4:]
    feed(hs, np.zeros(3 * M.FS))
    hs.idle_keepalive()
    assert _requests(io) == 0, io.msgs[-4:]
    assert hs._ask_owed and hs._txq
    sent = len(io.sent)
    feed(hs, np.r_[_over(hs, b' ok\r', 4), np.zeros(36000)])
    assert io.host == [b' ok\r'], io.msgs[-4:]
    traffic = M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 0)[1])
    assert any(np.array_equal(s, traffic) for s in io.sent[sent:]), io.msgs[-4:]
    assert _requests(io) == 0 and not hs._granted_over_unread(), io.msgs[-4:]


def test_frame_one_leaves_the_ask_free():
    hs, io = _answered(1)
    assert not hs._granted_over_unread()


def test_a_request_granted_from_a_held_turn_arms_the_reader_for_its_over():
    """The same session, 48.3 s: the caller handed back an empty turn with its
    f20 after its last over, we held it with text queued, and its request 0.15 s
    on was granted. The caller's level-6 over that followed went unread for want
    of an armed reader, and its level-4 resend was read."""
    hs, io = station()
    hs.chat = hs._chat_profile = True
    hs._link_high_speed, hs._expect_level, hs._lv_owed = True, 6, False
    hs.send(b'text queued behind the caller\'s close')
    hs._hold_released_turn()
    assert hs.turn == A._TURN_ASKED
    request = M.synth_burst(hs.caller, F.for_bw(F.SESSION_TURN_REQUEST_SHORT, '500'))
    feed(hs, np.concatenate([np.zeros(2400), request, np.zeros(4800)]))
    grant = M.synth_tone_pairs(F.turn_grant(hs.called, '500'))
    assert any(np.array_equal(s, grant) for s in io.sent), io.msgs[-4:]
    assert hs.turn == A._TURN_PEER and hs._lv_armed, io.msgs[-4:]
    feed(hs, np.r_[np.zeros(1200), _over(hs, b' ok\r', 6), np.zeros(36000)])
    assert io.host == [b' ok\r'], io.msgs[-4:]


@pytest.mark.parametrize('turn', [A._TURN_PEER, A._TURN_ASKED])
def test_a_clean_base_resend_is_read_with_our_ask_outstanding(turn):
    """The same session, 75.7 s: that ask left the turn asked, the over drew
    our NAK, and its three level-1 resends, each whole and CRC-clean, were
    located and dropped because the base window belonged only to a peer's
    turn. An ask leaves the turn with the peer until it hands it over."""
    hs, io = station()
    hs.chat = hs._chat_profile = True
    hs._link_high_speed = True
    hs._rx_resend, hs._rx_retry_rates = (4, (True,)), {4, 1}
    hs._expect_level, hs._lv_owed = 1, True
    hs.turn = turn
    hs._key(False)
    feed(hs, np.r_[np.zeros(24000), _over(hs, b' ok\r', 1), np.zeros(36000)])
    assert io.host == [b' ok\r'], io.msgs[-4:]
    assert not any('not read' in m for m in io.msgs), io.msgs[-4:]


def _silence_until_closed(hs, io, seconds):
    for _ in range(seconds):
        feed(hs, np.zeros(M.FS))
        hs.idle_keepalive()
        if hs.state is not A.VaraState.CONNECTED:
            return


def test_granted_traffic_that_never_comes_closes_the_link_without_asking():
    """Asking would acknowledge an over we never read: a caller silent behind
    the grant ends the link on the inactivity bound."""
    hs, io = _answered(0)
    _silence_until_closed(hs, io, A._MAX_WITHOUT_PROGRESS * 13)
    assert _requests(io) == 0, io.msgs[-4:]
    assert hs.state is not A.VaraState.CONNECTED, io.msgs[-4:]


@pytest.mark.parametrize('level', [4, 6])
def test_text_queued_under_the_callers_over_waits_for_its_answer(level):
    """Kestrel answering Kestrel, OpenSlot clients, 2026-11-09 52.26 s: our host
    wrote under the caller's level-4 over, behind a plain release, and the ask
    went out 0.01 s past the over's end instead of its answer."""
    hs, io = _answered(1, expect=level)
    assert hs._lv_armed == (level > 4)
    hs._txq.clear()
    over = _over(hs, b' ok\r', level)
    feed(hs, np.zeros(1200))
    io.receiving = True
    feed(hs, over[:len(over) // 2])
    hs.send(b'text written under the over')
    feed(hs, over[len(over) // 2:])
    io.receiving = False
    feed(hs, np.zeros(36000))
    assert io.host == [b' ok\r'], io.msgs[-6:]
    traffic = M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 0)[1])
    assert np.array_equal(io.sent[-1], traffic), io.msgs[-6:]
    assert _requests(io) == 0, io.msgs[-6:]


def test_the_callers_poll_behind_a_traffic_grant_draws_our_ask():
    """Kestrel answering a Kestrel caller with OpenSlot, 2026-11-09 58.9 s: the
    caller answers every close with frame 0, took our grant with nothing queued
    and polled, and our text written 2.5 s later waited for an over that was
    never coming. Its frame-1 poll holds an empty turn, and our queue asks."""
    hs, io = _answered(0)
    hs.send(b'text written behind the grant')
    assert _requests(io) == 0
    poll = M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 1)[0])
    feed(hs, np.r_[np.zeros(2400), poll, np.zeros(9600)])
    assert not hs._granted_over_unread(), io.msgs[-4:]
    assert _requests(io) == 1, io.msgs[-4:]


def _naked(hs):
    hs._txq.clear()
    hs._rx_resend, hs._rx_retry_rates, hs._expect_level = (4, (True,)), {4, 1}, 1
    hs.send(b'text written behind our NAK')


def test_an_over_we_naked_holds_the_ask_until_its_resend_is_read():
    hs, io = _answered(1)
    _naked(hs)
    feed(hs, np.zeros(3 * M.FS))
    hs.idle_keepalive()
    assert _requests(io) == 0 and hs._ask_owed, io.msgs[-4:]
    feed(hs, np.r_[_over(hs, b' ok\r', 1), np.zeros(36000)])
    assert io.host == [b' ok\r'] and hs._rx_resend is None, io.msgs[-4:]


def test_an_over_we_naked_and_never_resent_closes_the_link_without_asking():
    hs, io = _answered(1)
    _naked(hs)
    _silence_until_closed(hs, io, A._MAX_WITHOUT_PROGRESS * 13)
    assert _requests(io) == 0, io.msgs[-4:]
    assert hs.state is not A.VaraState.CONNECTED, io.msgs[-4:]


def test_the_callers_drained_hand_over_behind_our_final_answer_is_taken():
    """Below the base level a caller hands the turn over with session-drained
    behind our answer to its last over. A weak-path Kestrel pair whose caller
    opened at its setup level 2: the frame was named and dropped because our
    turn was not asked, and both ends waited for the other until the link
    closed."""
    hs, io = station()
    hs.chat = hs._chat_profile = True
    hs.turn, hs._answer_owed = A._TURN_PEER, A._OWED_RELEASE
    hs._queue_host_payload(b'answerer text')
    drained = M.synth_burst(hs.called, F.for_bw(F.SESSION_DRAINED, '500'))
    feed(hs, np.r_[np.zeros(2400), drained, np.zeros(9600)])
    assert hs._tx_pending is not None, io.msgs[-3:]
    assert any(m.startswith('rx caller handover') for m in io.msgs), io.msgs[-3:]


def test_the_callers_drained_hand_back_behind_a_traffic_grant_is_taken():
    """A Kestrel caller keys frame 0 at every close; handed our grant with
    nothing queued it hands back with session-drained below the base level,
    and our text goes into that turn (weak-path Kestrel pair)."""
    hs, io = _answered(0)
    drained = M.synth_burst(hs.called, F.for_bw(F.SESSION_DRAINED, '500'))
    feed(hs, np.r_[np.zeros(2400), drained, np.zeros(9600)])
    assert not hs._granted_over_unread() and hs._tx_pending is not None, io.msgs[-3:]


def test_a_caller_repeating_its_final_ack_is_not_silent():
    """Each repeat of the caller's final ACK behind our initial-greeting grant
    is named and re-granted without progress; it is still the caller heard."""
    hs, io = _answered(0)
    hs._initial_answer_turn = True
    ack = M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 0)[0])
    for _ in range(A._MAX_WITHOUT_PROGRESS * 2):
        feed(hs, np.r_[np.zeros(2400), ack, np.zeros(2400)])
        for _ in range(9):
            feed(hs, np.zeros(M.FS))
            hs.idle_keepalive()
    assert hs.state is A.VaraState.CONNECTED, io.msgs[-3:]
    assert _requests(io) == 0, io.msgs[-3:]


@pytest.mark.parametrize('form', ['f20', 'idle'])
def test_a_held_turns_frame_0_is_traffic_until_its_over_is_read(form):
    """Behind our empty hand-back the caller's frame 0 asks with traffic: no
    ask of ours may answer its over before that over is read."""
    hs, io = station()
    hs.chat = hs._chat_profile = True
    hs.turn, hs._released_form = A._TURN_PEER, form
    hs._caller_holds_control(0)
    assert hs._granted_over_unread(), io.msgs[-3:]


def test_a_held_turns_repeated_frame_0_behind_a_lost_grant_is_granted_again():
    """Our grant of its held-turn ask unheard, the caller asks again with frame
    0 on its cadence; the grant goes again and the over stays guarded."""
    hs, io = station()
    hs.chat = hs._chat_profile = True
    hs.turn, hs._released_form = A._TURN_PEER, 'f20'
    hs._caller_holds_control(0)
    granted = len(io.sent)
    ask = M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 0)[0])
    feed(hs, np.r_[np.zeros(2400), ask, np.zeros(9600)])
    assert len(io.sent) == granted + 1 and hs._granted_over_unread(), io.msgs[-3:]
