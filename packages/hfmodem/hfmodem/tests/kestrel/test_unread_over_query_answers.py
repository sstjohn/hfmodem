# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A query names an over this end has not read: it draws that over's NAK,
not a replay of the previous ACK, after a refused NAK and after a handover."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock, _IO  # noqa: F401
from .test_responder_control_stream import feed
from .test_limited_responder_query_recovery import (pending, receiver, on_air, query,
                                                    high_speed_pair, hear_with_tail)


@pytest.mark.parametrize('high_speed', [True, False])
def test_refused_crc_nak_keeps_the_unread_over_out_of_a_query_reack(high_speed):
    if high_speed:
        h, hi, c, ci = high_speed_pair(bytes(range(256)) * 8)
    else:
        h, hi = pending(payload=bytes(range(256)) * 8)
        c, ci = receiver(h)
    for _ in range(2):
        hear_with_tail(c, hi)
        hear_with_tail(h, ci)
    delivered = len(b''.join(ci.host))
    assert h.payload_ledger.confirmed == delivered and h._tx_pending[1] == 3
    # Over #3 arrives CRC-bad; the NAK for it is refused by the transmitter.
    ci.tx_went_out = lambda: False
    c._irs_nak(A._phy.LevelRead(c._expect_level, (), None, 1))
    ci.tx_went_out = lambda: True
    # The sender's actual phase-0 query for over #3, and the receiver's reply.
    query(h, hi, 11)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    assert len(ci.sent) == sent + 1, ci.msgs
    hear_with_tail(h, ci)
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == delivered, hi.msgs


@pytest.mark.parametrize('high_speed', [False, True])
def test_lost_first_data_of_a_later_delivery_draws_an_answer_to_its_query(high_speed):
    h, hi = pending(final=True)
    c, ci = receiver(h)
    h._link_high_speed = c._link_high_speed = high_speed
    c.over_continue_after = c.over_continue
    assert c._answer_data_over(on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 9 and h._tx_pending is None, hi.msgs
    h.send(bytes(range(110)))
    feed(c, on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    assert h.turn == A._TURN_OURS and h._tx_pending is not None, hi.msgs
    # Erase the new delivery's first DATA, then exchange its frame-11 query.
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, 11)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    assert len(ci.sent) == sent + 1, '\n'.join(ci.msgs)
    hear_with_tail(h, ci)
    assert h.payload_ledger.confirmed == 9 and h._tx_retries == 1, hi.msgs


def later_turn(high_speed):
    h, hi = pending(final=True)
    c, ci = receiver(h)
    h._link_high_speed = c._link_high_speed = high_speed
    c.over_continue_after = c.over_continue
    assert c._answer_data_over(on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    h.send(bytes(range(110)))
    feed(c, on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    assert h.turn == A._TURN_OURS and h._tx_pending is not None
    return h, hi, c, ci


@pytest.mark.parametrize('high_speed', [False, True])
def test_granted_turn_keeps_the_saved_ack_without_arming_unasked_recovery(high_speed):
    _, _, c, ci = later_turn(high_speed)
    sent, progress, answer = len(ci.sent), c.progress, c._over_answer
    assert answer is not None and c._answer_owed is None
    assert not c._first_over_owed()
    feed(c, np.zeros(6 * M.FS))
    assert c._rx_window.A.kind == 'nak' and not c._rx_window.A.sent
    assert c._reack_on_idle()
    assert len(ci.sent) == sent + 1 and c.progress == progress
    assert c._rx_window.A.kind == 'nak' and c._rx_window.A.sent
    assert c._rx_resend is not None


@pytest.mark.parametrize('high_speed', [False, True])
def test_later_delivery_recovers_wholly_lost_data_then_delivers_exactly_once(high_speed):
    h, hi, c, ci = later_turn(high_speed)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, 11)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    hear_with_tail(h, ci)
    assert h._tx_retries == 1 and h.payload_ledger.confirmed == 9
    for _ in range(30):
        hear_with_tail(c, hi)
        hear_with_tail(h, ci)
        if h._tx_pending is None:
            break
    assert h._tx_pending is None, hi.msgs
    assert b''.join(ci.host) == b'last over' + bytes(range(110)), ci.msgs
    assert h.payload_ledger.confirmed == 119


def test_refused_handback_does_not_admit_a_missing_data_query():
    h, _ = pending()
    c, ci = receiver(h)
    c.turn = A._TURN_OURS
    c._over_answer = ('control', 0)
    ci.tx_went_out = lambda: False
    assert not c._release_turn()
    sent = len(ci.sent)
    c._answer_caller_query(None, query_phase=0)
    assert c.turn == A._TURN_OURS and len(ci.sent) == sent
    ci.tx_went_out = lambda: True
    assert c._release_turn()
    c._took_answer_query(0)
    assert np.array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_NAK, '500')))


@pytest.mark.parametrize('sent', [False, True])
def test_caller_direct_nak_preserves_unread_debt_without_an_ack_ladder(sent):
    h, _ = pending()
    c, ci = receiver(h)
    assert c._answer_data_over(on_air(h.io.sent[-1]))
    delivered = ci.host.copy()
    c._owe_nothing()
    ci.tx_went_out = lambda: sent
    c._irs_nak(A._phy.LevelRead(c._expect_level, (), None, 1))
    assert c._answer_owed == A._OWED_OVER and c._over_answer is None
    assert bool(c._rx_resend) == sent
    ci.tx_went_out = lambda: True
    before = len(ci.sent)
    feed(c, on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', 1 - int(sent)))))
    assert len(ci.sent) == before + 1, ci.msgs
    assert c._rx_failed and c._answer_owed == A._OWED_OVER
    assert ci.host == delivered


def test_refused_query_answer_preserves_the_granted_turn_and_its_budget():
    _, hi, c, ci = later_turn(False)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.zeros(len(hi.sent[-1])))
    before = c.progress, c._reacks, c._rx_resend, c._expect_level
    ci.tx_went_out = lambda: False
    c._took_answer_query(0)
    assert (c.progress, c._reacks, c._rx_resend, c._expect_level) == before
    ci.tx_went_out = lambda: True
    for phase in (0, 1, 0, 1):
        c._took_answer_query(phase)
    assert c._reacks == A._REACK_MAX and c.progress == before[0]
    c._took_answer_query(0)
    assert c.state != A.VaraState.CONNECTED or c._close_asked_at is not None
    assert b''.join(ci.host) == b'last over'


@pytest.mark.parametrize('high_speed', [False, True])
@pytest.mark.parametrize('prompt', ['frame12', 'idle'])
def test_refused_nak_cannot_replay_an_ack_through_the_poll_or_idle_path(high_speed, prompt):
    h, hi, c, ci = high_speed_pair(bytes(range(256)) * 8)
    h._link_high_speed = c._link_high_speed = high_speed
    hear_with_tail(c, hi)
    hear_with_tail(h, ci)
    delivered = b''.join(ci.host)
    ci.tx_went_out = lambda: False
    c._irs_nak(A._phy.LevelRead(c._expect_level, (), None, 1))
    ci.tx_went_out = lambda: True
    if prompt == 'frame12':
        query(h, hi, 12)
        _Clock.t -= len(hi.sent[-1]) / M.FS
        hear_with_tail(c, hi)
    else:
        c._idle_end = _Clock.t
        assert c._reack_on_idle()
    assert np.array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_NAK_SHORT, '500'))), ci.msgs
    hear_with_tail(h, ci)
    assert h.payload_ledger.confirmed == len(delivered)
    assert b''.join(ci.host) == delivered


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('high_speed', [False, True])
def test_responder_handover_answers_the_short_query_for_wholly_lost_data(bw, high_speed):
    # A VARA HF caller's eight-symbol first query: test_varac_answer_1020,
    # r3c-queries at 41.395, before its 16- and 32-symbol queries.
    io = _IO()
    c = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    c.role, c.called, c.caller = 'responder', 'W9SSJ', 'W1AW'
    c.state, c.turn = A.VaraState.CONNECTED, A._TURN_OURS
    c._link_high_speed = high_speed
    c._peer_over = 2
    c._over_answer = ('control', 0)
    assert c._release_turn()
    assert c._answer_owed is None and not c._first_over_owed()
    sent, progress = len(io.sent), c.progress
    feed(c, np.zeros(4 * M.FS))  # The caller's new DATA vanished completely.
    assert len(io.sent) == sent
    pairs = F.over_poll(c.called, bw)[0]
    assert len(pairs) == 8
    poll = M.synth_tone_pairs(pairs)
    feed(c, np.r_[on_air(poll), np.zeros(M.FS)])
    assert len(io.sent) == sent + 1, io.msgs
    np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_NAK_RESPONDER_SHORT, bw)))
    assert not io.host and c.progress == progress
    assert c.payload_ledger.confirmed == 0 and c.turn == A._TURN_PEER


@pytest.mark.parametrize('high_speed', [False, True])
def test_send_after_refused_nak_waits_for_unread_data_and_answers_its_query(high_speed):
    if high_speed:
        h, hi, c, ci = high_speed_pair(bytes(range(256)) * 8)
    else:
        h, hi = pending(payload=bytes(range(256)) * 8)
        c, ci = receiver(h)
    for _ in range(2):
        hear_with_tail(c, hi)
        hear_with_tail(h, ci)
    delivered = b''.join(ci.host)
    assert h._tx_pending[1] == 3 and c._rx_resend is None
    ci.tx_went_out = lambda: False
    c._irs_nak(A._phy.LevelRead(c._expect_level, (), None, 1))
    ci.tx_went_out = lambda: True
    sent = len(ci.sent)
    c.send(b'reply')
    assert len(ci.sent) == sent and c.turn == A._TURN_PEER, ci.msgs
    assert c._ask_owed
    # Even after the ACK ladder is cleared, the definite unread DATA remains.
    c._owe_nothing()
    assert not c._owes_turn()
    c._ask_if_owed()
    assert len(ci.sent) == sent and c.turn == A._TURN_PEER
    c.idle_keepalive()
    assert not c._tx_turn_request()
    assert len(ci.sent) == sent and c.turn == A._TURN_PEER
    query(h, hi, 11)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    hear_with_tail(c, hi)
    assert len(ci.sent) == sent + 1, ci.msgs
    hear_with_tail(h, ci)
    assert h.payload_ledger.confirmed == len(delivered) and h._tx_retries == 1
    assert b''.join(ci.host) == delivered
    hear_with_tail(c, hi)
    assert c._rx_resend is None and c._over_answer is not None
    assert c._ask_for_turn() and c.turn == A._TURN_ASKED


@pytest.mark.parametrize('high_speed', [False, True])
def test_host_can_ask_for_a_possibly_idle_granted_turn(high_speed):
    _, _, c, ci = later_turn(high_speed)
    feed(c, np.zeros(4 * M.FS))
    sent = len(ci.sent)
    c.send(b'reply')
    assert len(ci.sent) == sent + 1 and c.turn == A._TURN_ASKED, ci.msgs
    assert c._rx_resend is None


@pytest.mark.parametrize('high_speed', [False, True])
@pytest.mark.parametrize('block', [960, 4096, 24000])
@pytest.mark.parametrize('lit_tail', [False, True])
def test_ordinary_bw500_query_reack_uses_the_complete_query_end(monkeypatch, high_speed, block, lit_tail):
    from .test_answer_query_recovery import caller_owing
    c, ci = caller_owing(monkeypatch, high_speed=high_speed)
    # Keep the saved ACK, with no missing-DATA obligation.
    sent, answer = len(ci.sent), c._over_answer
    if lit_tail:
        # The envelope can still read lit after the whole frame fits.
        monkeypatch.setattr(c, '_band_quiet_s', lambda: 0.0)
    poll = M.synth_tone_pairs(F.answer_query(c.called, '500'))
    start = _Clock.t
    lead = M.FS // 5
    feed(c, np.r_[np.zeros(lead), poll, np.zeros(M.FS // 2)], block)
    assert len(ci.sent) == sent + 1 and c._over_answer == answer, ci.msgs
    age = ci.at[-1] - (start + (lead + len(poll)) / M.FS)
    assert 0 <= age <= A._ANSWER_HELD_S + block / M.FS + .03
    assert any('answering again' in m for m in ci.msgs)


@pytest.mark.parametrize('marked', [False, True])
def test_refused_nak_peer_close_then_incoming_chat_setup_answers_with_peer_idle(marked):
    from .test_setup_query_recovery import padded, query as setup_query
    h, _ = pending()
    c, ci = receiver(h)
    ci.tx_went_out = lambda: False
    c._irs_nak(A._phy.LevelRead(c._expect_level, (), None, 1))
    ci.tx_went_out = lambda: True
    assert c._query_data_due and c._nak_data_due and c._rx_resend is None
    feed(c, padded(M.synth_burst(c.called, F.for_bw(
        F.SESSION_DISCONNECT_REQ_RESPONDER_SHORT, '500'))))
    assert c.state == A.VaraState.DISCONNECTED, ci.msgs
    c.listen()
    assert c.state == A.VaraState.LISTENING
    c.chat = c.high_speed = True
    request = F.connect_request('500')
    c.on_rx_tones(F.handshake_tones(c.mycalls[0], request), request)
    assert c.role == 'responder' and c.state == A.VaraState.CONNECTING
    assert c._take_link_setup(F.link_setup_frame('VE6DNL', bw='500', ask=1,
                                               high_speed=marked))
    assert c.state == A.VaraState.CONNECTED and c.turn == A._TURN_PEER
    assert c._link_high_speed == marked
    sent, progress = len(ci.sent), c.progress
    feed(c, padded(setup_query(c.called)))
    assert len(ci.sent) == sent + 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        'VE6DNL', F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, '500')))
    assert not c._query_data_due and not c._nak_data_due
    assert not ci.host and c.progress == progress and c.payload_ledger.confirmed == 0


@pytest.mark.parametrize('boundary', ['peer-close', 'peer-ident', 'caller-close-stream',
                                    'caller-close-tones', 'local-close', 'listen',
                                    'incoming', 'outgoing'])
def test_unread_query_evidence_ends_at_each_link_boundary(boundary):
    io = _IO()
    c = A.VaraStationHandshake(['W9SSJ'], io, bw='500')
    c.role, c.called, c.caller = 'initiator', 'W1AW', 'W9SSJ'
    c.state, c.turn = A.VaraState.CONNECTED, A._TURN_PEER
    if boundary in ('listen', 'incoming'):
        c.state = A.VaraState.DISCONNECTING if boundary == 'listen' else A.VaraState.LISTENING
    c._query_data_due = c._nak_data_due = True
    if boundary == 'peer-close':
        c._took_peer_close()
    elif boundary == 'peer-ident':
        c._took_peer_ident()
    elif boundary.startswith('caller-close'):
        c.role = 'responder'
        if boundary == 'caller-close-stream':
            c._answer_responder_control(F.SESSION_DISCONNECT_REQ)
        else:
            c.on_rx_tones(F.handshake_tones(c.called, F.for_bw(F.SESSION_DISCONNECT_REQ, '500')),
                          F.SESSION_DISCONNECT_REQ)
    elif boundary == 'local-close':
        c.disconnect()
    elif boundary == 'listen':
        c.listen()
    elif boundary == 'incoming':
        request = F.connect_request('500')
        c.on_rx_tones(F.handshake_tones('W9SSJ', request), request)
        assert c.state == A.VaraState.CONNECTING
    else:
        assert c.originate('VE6DNL')
    expected = (A.VaraState.CONNECTING if boundary in ('incoming', 'outgoing') else
                A.VaraState.LISTENING if boundary == 'listen' else A.VaraState.DISCONNECTED)
    assert c.state == expected
    assert not c._query_data_due and not c._nak_data_due


def test_unread_query_evidence_survives_a_refused_close_and_receive_reset():
    io = _IO()
    c = A.VaraStationHandshake(['W9SSJ'], io, bw='500')
    c.role, c.called, c.caller = 'initiator', 'W1AW', 'W9SSJ'
    c.state, c.turn = A.VaraState.CONNECTED, A._TURN_PEER
    c._query_data_due = c._nak_data_due = True
    c.rx_discontinuity()
    assert c._query_data_due and c._nak_data_due
    io.tx_went_out = lambda: False
    c.disconnect()
    assert c.state == A.VaraState.DISCONNECTING
    assert c._query_data_due and c._nak_data_due
    io.tx_went_out = lambda: True
    c.disconnect()
    assert c.state == A.VaraState.DISCONNECTED
    assert not c._query_data_due and not c._nak_data_due
