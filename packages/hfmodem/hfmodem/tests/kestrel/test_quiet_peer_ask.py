# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Queued answerer traffic asks a quiet caller without a prior release."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import _station
from .test_setup_query_recovery import setup, query, padded, feed, _Clock, clock  # noqa: F401


def recovered(bw, *, chat=False):
    hs, io = setup(bw=bw, chat=chat)
    feed(hs, padded(query(bw=bw)))
    assert len(io.sent) == 2
    np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
        hs.caller, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, bw)))
    feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, bw))))
    assert len(io.sent) == 3
    np.testing.assert_array_equal(io.sent[-1], M.synth_tone_pairs(
        F.control_bursts(hs.caller, bw, 1)[1]))
    assert hs._released_form is None and hs.turn == A._TURN_PEER
    return hs, io


def tick_at(hs, at):
    _Clock.t = at
    if hs.idle_poll_due_in() == 0:
        hs.idle_keepalive()


def assert_ask(hs, io):
    np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
        hs.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, hs.bw)))
    assert hs.turn == A._TURN_ASKED and hs._tx_pending is None


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('grant', ['control', 'release', 'confirm-32'])
def test_queued_payload_asks_after_own_quiet_then_repeats_and_takes_grant(bw, grant):
    hs, io = recovered(bw)
    unkeyed = hs._unkeyed_at
    _Clock.t += 1.5
    hs.send(b'queued reply')
    progress, progress_at = hs.progress, hs._progress_at
    assert len(io.sent) == 3 and hs._tx_pending is None
    assert hs._awaiting_caller_confirm
    assert hs.idle_poll_due_in() == pytest.approx(unkeyed + 5.7 - _Clock.t)
    tick_at(hs, unkeyed + 5.7 - .001)
    assert len(io.sent) == 3
    tick_at(hs, unkeyed + 5.7)
    assert len(io.sent) == 4, io.msgs
    assert_ask(hs, io)
    assert hs.idle_poll_due_in() == pytest.approx(3.64)
    tick_at(hs, unkeyed + 5.7 + 3.64 - .001)
    assert len(io.sent) == 4
    tick_at(hs, unkeyed + 5.7 + 3.64)
    assert len(io.sent) == 5, io.msgs
    assert_ask(hs, io)
    assert (hs.progress, hs._progress_at) == (progress, progress_at)
    if grant == 'control':
        samples = M.synth_tone_pairs(F.caller_turn_grant('W9SSJ', bw))
    else:
        kind = F.SESSION_TURN_RELEASE if grant == 'release' else A._CONFIRM_LONG[4]
        samples = M.synth_burst('W9SSJ', F.for_bw(kind, bw))
    feed(hs, padded(samples))
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None, io.msgs


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_requeue_on_an_already_quiet_channel_is_due_at_once(bw):
    hs, io = recovered(bw)
    _Clock.t = hs._unkeyed_at + 8.8
    hs.send(b'requeued reply')
    assert hs.idle_poll_due_in() == 0
    tick_at(hs, _Clock.t)
    assert len(io.sent) == 4
    assert_ask(hs, io)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('turns', [(A._TURN_PEER,), (A._TURN_OURS, A._TURN_PEER),
                                   (A._TURN_PEER, A._TURN_ASKED)])
def test_new_host_write_after_a_turn_change_waits_for_own_quiet(bw, turns):
    hs, io = recovered(bw)
    hs.send(b'first reply')
    tick_at(hs, hs._unkeyed_at + 5.7)
    assert_ask(hs, io)
    unkeyed = hs._unkeyed_at
    assert hs.idle_poll_due_in() == pytest.approx(3.64)
    for turn in turns:
        hs.turn = turn
    hs._txq.clear()
    _Clock.t = unkeyed + 4.0
    hs.send(b'new reply')
    assert hs.idle_poll_due_in() == pytest.approx(1.7)
    for delta in (4.0, 5.7 - .001):
        tick_at(hs, unkeyed + delta)
        assert len(io.sent) == 4 and hs._tx_pending is None
    tick_at(hs, unkeyed + 5.7)
    assert len(io.sent) == 5
    assert_ask(hs, io)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_chat_on_recovered_queue_stays_silent_on_quiet_ticks(bw):
    hs, io = recovered(bw, chat=True)
    hs.send(b'queued reply')
    unkeyed = hs._unkeyed_at
    for delta in (3.64, 5.7, 9.34, 10.0):
        _Clock.t = unkeyed + delta
        assert hs.idle_poll_due_in() is None
        hs.idle_keepalive()
        assert len(io.sent) == 3 and hs._tx_pending is None, io.msgs


@pytest.mark.parametrize('hold', ['nak', 'answer', 'held-answer', 'receiving', 'empty',
                                  'pending', 'close', 'release', 'poll', 'resend',
                                  'ours', 'initiator', 'disconnected', 'released-form', 'chat'])
def test_receive_debts_and_other_owners_hold_the_unprompted_ask(hold):
    hs, io = recovered('2300')
    hs.send(b'queued reply')
    if hold == 'nak':
        hs._nak_data_due = True
    elif hold == 'answer':
        hs._answer_owed = A._OWED_OVER
    elif hold == 'held-answer':
        hs._held_answer = (False, False)
    elif hold == 'receiving':
        io.receiving = True
    elif hold == 'empty':
        hs._txq.clear()
    elif hold == 'pending':
        hs._tx_pending = (b'pending', 0, 1)
    elif hold == 'close':
        hs._close_asked_at = _Clock.t
    elif hold == 'release':
        hs._release_owed = True
    elif hold == 'poll':
        hs._poll_at = _Clock.t + 100
    elif hold == 'resend':
        hs._rx_resend = b'owed'
    elif hold == 'ours':
        hs.turn = A._TURN_OURS
    elif hold == 'initiator':
        hs.role = 'initiator'
    elif hold == 'disconnected':
        hs.state = A.VaraState.DISCONNECTED
    elif hold == 'chat':
        hs.chat = hs._chat_profile = True
    else:
        hs._released_form = 'f20'
        hs._ask_not_before = _Clock.t + 100
    _Clock.t = hs._unkeyed_at + 20
    assert hs.idle_poll_due_in() is None
    hs.idle_keepalive()
    assert len(io.sent) == 3, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_unheard_ack_keeps_only_its_existing_train(bw):
    hs, io = setup(bw=bw, chat=False)
    hs.send(b'queued reply')
    acked_at = hs._acked_at[1]
    for n, delta in enumerate((5.7, 9.34, 12.98), 1):
        tick_at(hs, acked_at + delta - .001)
        assert len(io.sent) == n
        tick_at(hs, acked_at + delta)
        assert len(io.sent) == n + 1
        assert hs._ack_unheard() and hs.turn == A._TURN_PEER
        assert hs.idle_poll_due_in() == pytest.approx(3.64)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_a_keepalive_behind_a_train_ask_still_draws_frame_zero(bw):
    hs, io = recovered(bw)
    hs.send(b'queued reply')
    tick_at(hs, hs._unkeyed_at + 5.7)
    assert len(io.sent) == 4
    assert_ask(hs, io)
    _Clock.t += 1.5
    feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, bw))))
    assert len(io.sent) == 5, io.msgs
    np.testing.assert_array_equal(io.sent[-1], M.synth_tone_pairs(
        F.control_bursts(hs.caller, bw, 0)[1]))
    assert hs.turn == A._TURN_ASKED and hs._tx_pending is None
    unkeyed = hs._unkeyed_at
    assert hs.idle_poll_due_in() == pytest.approx(unkeyed + 5.7 - _Clock.t)
    tick_at(hs, unkeyed + 5.7 - .001)
    assert len(io.sent) == 5
    tick_at(hs, unkeyed + 5.7)
    assert len(io.sent) == 6
    assert_ask(hs, io)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_repeated_setup_query_behind_a_quiet_ask_restarts_own_quiet(bw):
    hs, io = setup(bw=bw, chat=False)
    idle = M.synth_burst(hs.caller, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, bw))
    feed(hs, padded(query(bw=bw)))
    assert len(io.sent) == 2
    np.testing.assert_array_equal(io.sent[-1], idle)
    hs.send(b'queued reply')
    tick_at(hs, hs._unkeyed_at + 5.7)
    assert len(io.sent) == 3
    assert_ask(hs, io)
    before = (hs.progress, hs._progress_at, hs.payload_ledger.pending,
              hs.payload_ledger.confirmed)
    _Clock.t += 1.5
    for count in (4, 5):
        feed(hs, padded(query(bw=bw)))
        assert len(io.sent) == count, io.msgs
        np.testing.assert_array_equal(io.sent[-1], idle)
        assert hs.turn == A._TURN_ASKED and hs._tx_pending is None
        assert (hs.progress, hs._progress_at, hs.payload_ledger.pending,
                hs.payload_ledger.confirmed) == before
        assert hs.idle_poll_due_in() == pytest.approx(hs._unkeyed_at + 5.7 - _Clock.t)
    unkeyed = hs._unkeyed_at
    assert hs.idle_poll_due_in() == pytest.approx(unkeyed + 5.7 - _Clock.t)
    tick_at(hs, unkeyed + 5.7 - .001)
    assert len(io.sent) == 5
    tick_at(hs, unkeyed + 5.7)
    assert len(io.sent) == 6
    assert_ask(hs, io)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('case', ['context', 'expired', 'answer', 'query-data', 'held',
                                  'close', 'released', 'poll', 'keepalive'])
def test_setup_query_behind_a_quiet_ask_keeps_its_recovery_guards(bw, case):
    hs, io = setup(bw=bw, chat=False)
    feed(hs, padded(query(bw=bw)))
    hs.send(b'queued reply')
    tick_at(hs, hs._unkeyed_at + 5.7)
    assert_ask(hs, io)
    if case == 'context':
        hs._setup_answer_for = ('W1AW', hs.called, hs.progress)
    elif case == 'expired':
        _Clock.t = hs._progress_at + A._MAX_WITHOUT_PROGRESS * A._IDLE_POLL_S
    elif case == 'answer':
        hs._answer_owed = A._OWED_OVER
    elif case == 'query-data':
        hs._query_data_due = True
        hs._rx_window.event('G')
    elif case == 'held':
        hs._deliver([A._phy.vara_body(b'', hs.caller)])
        hs._perform_rx_actions(hs._rx_window.event('E', single=True, peer_keying=True))
        hs._held_answer = (False, False)
    elif case == 'close':
        hs._close_asked_at = _Clock.t
    elif case == 'released':
        hs._released_form = 'f20'
    elif case == 'poll':
        hs._poll_at = _Clock.t + 100
    else:
        feed(hs, padded(M.synth_burst('W9SSJ', F.for_bw(F.SESSION_KEEPALIVE_B, bw))))
        assert hs._setup_answer_for is None
    count = len(io.sent)
    feed(hs, padded(query(bw=bw)))
    if case in ('query-data', 'held'):
        assert len(io.sent) == count + 1, io.msgs
        assert hs._rx_window.A.kind == ('nak' if case == 'query-data' else 'final')
        assert hs._rx_window.A.sent and not io.host
    else:
        assert len(io.sent) == count, io.msgs


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_a_poll_behind_a_released_form_ask_is_answered_without_its_cadence_gate(bw):
    hs, io = _station(bw)
    hs.role = 'responder'
    hs.chat = hs._chat_profile = False
    hs.turn = A._TURN_OURS
    assert hs._release_turn()
    assert hs._released_form == 'f13' and hs.turn == A._TURN_PEER
    poll = M.synth_tone_pairs(F.control_bursts(hs.caller, bw, 1)[0])
    feed(hs, padded(poll))
    assert len(io.sent) == 2, io.msgs
    assert hs._released_form == 'f20'
    _Clock.t = hs._unkeyed_at + 8.8
    hs.send(b'requeued reply')
    tick_at(hs, _Clock.t)
    assert len(io.sent) == 3, io.msgs
    assert_ask(hs, io)
    asked = io.at[-1]
    not_before = hs._ask_not_before
    assert not_before == pytest.approx(asked + 12.06)
    _Clock.t = asked + 1.9 - len(poll) / M.FS
    feed(hs, np.r_[poll, np.zeros(24000)])
    assert len(io.sent) == 4, io.msgs
    assert_ask(hs, io)
    assert 1.9 <= io.at[-1] - asked <= 1.9 + .1
    assert io.at[-1] < not_before


def test_unanswered_asks_do_not_extend_the_no_progress_close():
    hs, io = recovered('2300')
    hs.send(b'queued reply')
    progress = hs.progress
    deadline = hs._progress_at + A._MAX_WITHOUT_PROGRESS * A._IDLE_POLL_S
    asks = 0
    while hs.state == A.VaraState.CONNECTED:
        due = hs.idle_poll_due_in()
        assert due is not None
        tick_at(hs, _Clock.t + due)
        asks += 1
    assert asks > A._TURN_MAX_ASK
    assert deadline <= _Clock.t < deadline + 3.64
    assert hs.progress == progress and hs.state == A.VaraState.DISCONNECTED, io.msgs


@pytest.mark.parametrize('sent', [False, True])
def test_late_asks_preserve_the_listening_gap_behind_actual_unkey(sent):
    hs, io = recovered('2300')
    original_tx = io.tx

    def timed_tx(samples):
        original_tx(samples)
        _Clock.t += len(samples) / M.FS

    io.tx = timed_tx
    io.tx_went_out = lambda: sent
    hs.send(b'queued reply')
    _Clock.t = hs._unkeyed_at + 5.7
    io.receiving = True
    assert hs.idle_poll_due_in() is None
    hs.idle_keepalive()
    assert len(io.sent) == 3
    _Clock.t = hs._unkeyed_at + 9.2
    io.receiving = False
    assert hs.idle_poll_due_in() == 0
    hs.idle_keepalive()
    assert len(io.sent) == 4
    if sent:
        assert_ask(hs, io)
    else:
        assert hs.turn == A._TURN_PEER and hs._tx_pending is None
    duration = len(io.sent[-1]) / M.FS
    interval = 3.64 - duration if sent else 5.7
    assert hs.idle_poll_due_in() == pytest.approx(interval)
    tick_at(hs, _Clock.t + hs.idle_poll_due_in())
    assert len(io.sent) == 5
    assert io.at[-1] - io.at[-2] == pytest.approx(duration + interval)
