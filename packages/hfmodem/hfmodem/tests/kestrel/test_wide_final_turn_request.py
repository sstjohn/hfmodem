# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA HF wide final-over ACK/query/request/handover, 2026-10-30."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_wide_answer_query import (
    clock, final_pending, pending, receiver, query, timed_transmit, on_air, feed, _Clock,  # noqa: F401
)
from .test_idle_poll import _IO, _station


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_lost_opening_greeting_and_queries_hold_host_ask_at_runtime_idle_tick(bw):
    from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
    from .test_idle_tick_reply_window import Modem, Transport
    from .test_responder_wide_confirm import acquired

    h, hi = acquired(bw, immediate=b'last over')
    c, ci = receiver(h)
    c.state, c.step, c.chat = A.VaraState.CONNECTING, A._I_LINKSETUP_SENT, True
    timed_transmit(hi)
    timed_transmit(ci)
    hi.final_timed = True
    c.on_rx_audio(hi.sent[-1])
    assert c.state == A.VaraState.CONNECTED and c._confirm_open(), ci.msgs
    runtime = AudioHostRuntime(Transport(), clock=_Clock.monotonic)
    runtime.modem = Modem(c)
    runtime.step()
    deadline = runtime._idle_at
    assert deadline == pytest.approx(c._unkeyed_at + 10)
    replay(h, ci)
    assert h._tx_pending is not None and h.payload_ledger.pending == 9, hi.msgs
    original = h._tx_pending
    solicit(h, hi)
    _Clock.t = h._answer_query_at + A._FINAL_QUERY_REPLY_S
    assert h.intermediate_query_due_in() == pytest.approx(0)
    assert h.query_intermediate_answer(), hi.msgs
    unkey = h._answer_query_sent_at
    _Clock.t = unkey + .2
    sent = len(ci.sent)
    c.send(b'caller traffic')
    assert len(ci.sent) == sent and c._ask_owed, ci.msgs
    assert .2 < deadline - unkey < 1
    _Clock.t = deadline - .001
    runtime.step()
    assert len(ci.sent) == sent
    _Clock.t = deadline
    runtime.step()
    if len(ci.sent) > sent:
        replay(h, ci)
    assert h._tx_pending is original and h.payload_ledger.confirmed == 0, (
        h.payload_ledger.confirmed, ci.host, deadline - unkey, hi.msgs)
    assert h.payload_ledger.pending == 9 and not ci.host
    assert len(ci.sent) == sent and c._ask_owed, ci.msgs
    assert c._handover_unheard and c._query_data_due
    assert runtime._idle_at == pytest.approx(_Clock.t + 10)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('level', [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('form', ['short', 'repeat-short', 'long', 'repeat-long'])
@pytest.mark.parametrize('sent', [False, True])
def test_connect_confirm_forms_arm_only_successful_wide_data_authorizations(bw, level, form, sent):
    h, io = _station(bw)
    h._link_high_speed = True
    h._setup_ask = level
    h.low_confirm = True
    h._heard_db = {5: 16, 6: 20}.get(level)
    timed_transmit(io)
    if form != 'short':
        h._tx_session_confirm()
    if form == 'repeat-long':
        h._took_confirm_ack()
    h._handover_unheard = h._query_data_due = False
    io.tx_went_out = lambda: sent
    if form in ('short', 'repeat-short'):
        h._tx_session_confirm()
    else:
        h._took_confirm_ack()
    assert h._confirm_start == level
    expected = sent and bw != '500'
    assert h._handover_unheard is expected, io.msgs
    assert h._query_data_due is expected
    io.tx_went_out = lambda: False
    if form in ('short', 'repeat-short'):
        h._tx_session_confirm()
    else:
        h._took_confirm_ack()
    assert h._handover_unheard is expected and h._query_data_due is expected


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('long', [False, True])
def test_heard_opening_idle_allows_host_ask_after_confirm(bw, long):
    h, io = _station(bw)
    h._start_compression('OFF')
    timed_transmit(io)
    h._tx_session_confirm()
    if long:
        h._took_confirm_ack()
    idle = M.synth_tone_pairs(F.control_bursts(h.caller, bw, 1)[1])
    feed(h, np.r_[on_air(idle), np.zeros(M.FS // 2)])
    assert not h._handover_unheard, io.msgs
    _Clock.t = max(_Clock.t, h._ask_not_before) + .1
    sent = len(io.sent)
    h.send(b'new traffic')
    assert len(io.sent) == sent + 1, io.msgs
    np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
        h.caller, F.for_bw(F.SESSION_TURN_REQUEST, bw)))


def sender(bw, role, level=4, payload=b'last over', final=True):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.caller = role, 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    if role == 'initiator':
        h.step = A._I_CONNECTED
    h.tx_level = level
    h._rec500 = A._phy.level_key(bw, level)
    if level != 4:
        h._link_high_speed = True
        h._lead = A.VB6Rnd(F.lead_state(h.caller, role == 'responder', bw))
    h._queue_host_payload(payload)
    timed_transmit(io)
    io.final_timed = True
    h._tx_data_over()
    assert A._phy.over_is_last(h._tx_pending[0], h.caller) is final
    assert h._tx_pending[2] == A._phy.level_key(bw, level)
    return h, io


def solicit(h, io):
    if not getattr(io, "final_timed", False):
        timed_transmit(io)
        io.final_timed = True
    _Clock.t = h._pending_answer_at
    due = h.intermediate_query_due_in()
    assert due == pytest.approx(2.25 if h.role == 'responder' else 1.32)
    feed(h, np.zeros(round(due * M.FS) + 1))
    assert h.intermediate_query_due_in() == pytest.approx(0, abs=1e-8)
    assert h.query_intermediate_answer(), io.msgs
    if h.role == 'responder':
        np.testing.assert_array_equal(io.sent[-1], M.synth_tone_pairs(
            F.answer_query(h.called, h.bw, (h._delivery_keyed - 1) % 2)))
    else:
        np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
            h.called, F.for_bw(F.SESSION_FINAL_ANSWER_QUERY, h.bw)))
    return _Clock.t


def request(h, short=False):
    kind = (F.SESSION_TURN_REQUEST_SHORT if short else F.SESSION_TURN_REQUEST)
    if h.role == 'initiator':
        kind = F.SESSION_TURN_REQUEST_RESPONDER
    return M.synth_burst(h.caller, F.for_bw(kind, h.bw))


def replay(h, io, block=960):
    _Clock.t = io.at[-1] - 4800 / M.FS
    feed(h, np.r_[on_air(io.sent[-1]), np.zeros(M.FS // 2)], block=block)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('short', [False, True])
@pytest.mark.parametrize('level', [4, 6])
def test_responder_queried_final_request_grants_and_settles_once(bw, short, level):
    h, io = sender(bw, 'responder', level)
    debt = h.payload_ledger.pending
    unkey = solicit(h, io)
    x = request(h, short)
    sent, progress = len(io.sent), h.progress
    feed(h, np.r_[np.zeros(int(.06 * M.FS)), x, np.zeros(M.FS // 2)])
    assert h._tx_pending is None and h.payload_ledger.confirmed == debt, io.msgs
    assert h.payload_ledger.pending == 0 and h.turn == A._TURN_PEER
    assert h.progress > progress and h._answer_query_for is None
    assert len(io.sent) == sent + 1
    np.testing.assert_array_equal(io.sent[-1], M.synth_tone_pairs(F.turn_grant(h.called, bw)))
    assert .19 <= io.at[-1] - (unkey + .06 + len(x) / M.FS) <= .21
    progress = h.progress
    h.on_rx_audio(on_air(x))
    assert h.payload_ledger.confirmed == debt and len(io.sent) == sent + 1
    assert h.progress == progress


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('level', [1, 4, 6])
def test_caller_queried_final_request_drains_every_final_record_once(bw, level):
    h, io = sender(bw, 'initiator', level, b'last' if level == 1 else b'x' * 60)
    assert len(h._tx_pending[0]) == {1: 10, 4: 90, 6: 174}[level]
    debt = h.payload_ledger.pending
    unkey = solicit(h, io)
    x = request(h)
    sent = len(io.sent)
    feed(h, np.r_[np.zeros(int(.06 * M.FS)), x, np.zeros(M.FS // 2)])
    assert h._tx_pending is None and h.payload_ledger.confirmed == debt, io.msgs
    assert h.payload_ledger.pending == 0 and h.turn == A._TURN_PEER
    assert len(io.sent) == sent + 1
    np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
        h.called, F.for_bw(F.SESSION_DRAINED, bw)))
    assert .19 <= io.at[-1] - (unkey + .06 + len(x) / M.FS) <= .21
    progress = h.progress
    h.on_rx_audio(on_air(x))
    assert h.payload_ledger.confirmed == debt and len(io.sent) == sent + 1
    assert h.progress == progress


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_queued_receiver_acks_final_then_requests_queried_turn_and_delivers(bw, role):
    h, hi = sender(bw, role)
    if role == 'responder':
        c, ci = receiver(h)
    else:
        ci = _IO()
        c = A.VaraStationHandshake([h.called], ci, bw=bw)
        c.role, c.called, c.caller = 'responder', h.called, h.caller
        c.state, c.turn = A.VaraState.CONNECTED, A._TURN_PEER
        c.chat = c._chat_profile = c._link_high_speed = False
        c._start_compression('OFF')
    timed_transmit(ci)
    c.turn = A._TURN_OURS
    c._hand_turn_over()
    ci.receiving = True
    c.send(b'queued before the final over')
    ci.receiving = False
    replay(c, hi)
    assert b''.join(ci.host) == b'last over'
    assert len(ci.sent) == 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], M.synth_tone_pairs(
        F.control_bursts(c.caller, bw, 0)[c.role == 'responder']))
    assert h.payload_ledger.confirmed == 0
    solicit(h, hi)
    replay(c, hi)
    assert len(ci.sent) == 2, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], request(h))
    replay(h, ci)
    assert h._tx_pending is None and h.payload_ledger.confirmed == 9, hi.msgs
    replay(c, hi)
    assert c._tx_pending is not None, ci.msgs
    replay(h, ci)
    assert b''.join(hi.host) == b'queued before the final over', hi.msgs
    assert b''.join(ci.host) == b'last over'


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('case', ['none', 'wall', 'samples', 'other-over', 'prequery'])
def test_request_requires_exact_fresh_query_and_postquery_start(bw, role, case):
    h, io = final_pending(bw) if role == 'responder' else sender(bw, role)
    x = request(h)
    if case == 'prequery':
        feed(h, np.r_[np.zeros(4800), x[:M.FS // 2]])
    if case != 'none':
        solicit(h, io)
    field = '_answer_query' if role == 'responder' else '_final_query'
    if case == 'wall':
        _Clock.t += A._FINAL_QUERY_REPLY_S + 1
    elif case == 'samples':
        setattr(h, field + '_samples', int((A._FINAL_QUERY_REPLY_S + 1) * M.FS))
    elif case == 'other-over':
        setattr(h, field + '_for', tuple(list(h._tx_pending)))
    debt, old = h.payload_ledger.pending, h._tx_pending
    feed(h, np.r_[x[M.FS // 2:] if case == 'prequery' else on_air(x), np.zeros(24000)])
    assert h._tx_pending is old and h.payload_ledger.confirmed == 0, io.msgs
    assert h.payload_ledger.pending == debt


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_lost_final_and_two_queries_hold_host_ask_until_recovery(bw, monkeypatch):
    hi = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], hi, bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_PEER
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h.send(b'last over')
    c, ci = receiver(h)
    settle = h._settle_host_payload

    def settle_delivered():
        assert b''.join(ci.host) == b'last over'
        settle()

    monkeypatch.setattr(h, '_settle_host_payload', settle_delivered)
    c.turn = A._TURN_OURS
    timed_transmit(hi)
    timed_transmit(ci)
    assert c._release_turn()
    replay(h, ci)
    for rung in (1, 2):
        _Clock.t = (h._pending_answer_at + A._ANSWER_QUERY_S if rung == 1 else
                    h._answer_query_at + A._FINAL_QUERY_REPLY_S)
        query(h, hi, rung)
    _Clock.t = h._answer_query_sent_at + .2
    sent = len(ci.sent)
    c.send(b'caller traffic queued')
    assert len(ci.sent) == sent and c._ask_owed and c.turn == A._TURN_PEER, ci.msgs
    c.idle_keepalive()
    feed(c, np.zeros(4800))
    c.on_rx_audio(np.zeros(4800))
    assert len(ci.sent) == sent and not ci.host
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 9
    _Clock.t = h._answer_query_at + A._FINAL_QUERY_REPLY_S
    query(h, hi, 3)
    replay(c, hi)
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_NAK_SHORT, bw)))
    replay(h, ci)
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 9
    assert h._tx_retries == 1 and not ci.host
    replay(c, hi)
    assert b''.join(ci.host) == b'last over', ci.msgs
    assert h.payload_ledger.confirmed == 0
    replay(h, ci)
    assert h.payload_ledger.confirmed == 9 and h.payload_ledger.pending == 0
    replay(c, hi)
    if c._tx_pending is None:
        _Clock.t = c._ask_not_before + .1
        c.idle_keepalive()
        replay(h, ci)
        replay(c, hi)
    assert c._tx_pending is not None, ci.msgs
    replay(h, ci)
    assert b''.join(hi.host) == b'caller traffic queued', hi.msgs
    assert b''.join(ci.host) == b'last over'


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['send', 'stream', 'bracket', 'idle'])
def test_handover_unheard_holds_every_host_ask_route(bw, role, route):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.caller = role, 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h._progressed()
    assert h._release_turn()
    sent = len(io.sent)
    _Clock.t = h._ask_not_before + .2
    h.send(b'host data')
    if route == 'stream':
        feed(h, np.zeros(M.FS))
    elif route == 'bracket':
        h.on_rx_audio(np.zeros(M.FS))
    elif route == 'idle':
        h.idle_keepalive()
        if role == 'initiator':
            assert h._since_progress == 1
    assert len(io.sent) == sent and h._ask_owed, io.msgs
    assert h.turn == A._TURN_PEER and h._txq and h._query_data_due


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_caller_final_with_another_delivery_queued_settles_only_the_queried_over(bw):
    h, io = sender(bw, 'initiator')
    h.send(b'another delivery')
    debt = h.payload_ledger.pending
    assert debt > 9 and h._txq
    solicit(h, io)
    feed(h, np.r_[on_air(request(h)), np.zeros(M.FS // 2)])
    assert h._tx_pending is None and h.payload_ledger.confirmed == 9, io.msgs
    assert h.payload_ledger.pending == debt - 9 and h._txq
    assert h.turn == A._TURN_PEER


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_refused_queried_handover_preserves_payload(bw, role):
    h, io = sender(bw, role)
    solicit(h, io)
    io.tx_went_out = lambda: False
    feed(h, np.r_[on_air(request(h)), np.zeros(M.FS // 2)])
    assert h._tx_pending is not None and h.payload_ledger.confirmed == 0, io.msgs
    assert h.payload_ledger.pending == 9 and h.turn == A._TURN_OURS


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_heard_idle_after_handover_allows_immediate_host_ask(bw, role):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.caller = role, 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h._progressed()
    assert h._release_turn()
    if role == 'initiator':
        idle = M.synth_tone_pairs(F.control_bursts(h.caller, bw, 1)[1])
    else:
        idle = M.synth_tone_pairs(F.control_bursts(h.caller, bw, 1)[0])
    feed(h, np.r_[on_air(idle), np.zeros(M.FS // 2)])
    sent = len(io.sent)
    _Clock.t = max(_Clock.t, h._ask_not_before) + .1
    h.send(b'new traffic')
    assert len(io.sent) == sent + 1, io.msgs
    kind = F.SESSION_TURN_REQUEST if role == 'initiator' else F.SESSION_TURN_REQUEST_RESPONDER
    np.testing.assert_array_equal(io.sent[-1], M.synth_burst(h.caller, F.for_bw(kind, bw)))


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('route', ['send', 'stream', 'bracket', 'idle'])
def test_read_intermediate_lost_final_and_query_hold_host_ask(bw, route):
    payload = bytes(range(110))
    h, hi = pending(bw, payload)
    c, ci = receiver(h)
    timed_transmit(hi)
    hi.final_timed = True
    timed_transmit(ci)
    feed(c, on_air(hi.sent[-1]))
    assert b''.join(ci.host) == payload[:89]
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 89 and h.payload_ledger.pending == 21
    assert A._phy.over_is_last(h._tx_pending[0], h.caller)
    assert c._answer_owed == A._OWED_OVER and not c._handover_unheard
    solicit(h, hi)
    _Clock.t = h._answer_query_sent_at + .2
    sent = len(ci.sent)
    c.send(b'host traffic')
    if route == 'stream':
        feed(c, np.zeros(4800))
    elif route == 'bracket':
        c.on_rx_audio(np.zeros(4800))
    elif route == 'idle':
        c.idle_keepalive()
    assert c._ask_owed and len(ci.sent) == sent, ci.msgs
    assert h.payload_ledger.confirmed == 89 and h.payload_ledger.pending == 21
    assert b''.join(ci.host) == payload[:89]


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('debt', ['over', 'delivery'])
@pytest.mark.parametrize('route', ['send', 'stream', 'bracket', 'idle'])
def test_incomplete_peer_delivery_holds_every_host_ask(bw, role, debt, route):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.caller = role, 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_PEER
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h._answer_owed = A._OWED_OVER if debt == 'over' else None
    h._peer_delivery_open = debt == 'delivery'
    h._released_form = 'grant' if role == 'responder' else None
    h._handover_unheard = False
    h._progressed()
    h._owed_block = debt == 'over'
    _Clock.t += 4
    sent = len(io.sent)
    h.send(b'queued')
    if route == 'stream':
        feed(h, np.zeros(4800))
    elif route == 'bracket':
        h.on_rx_audio(np.zeros(4800))
    elif route == 'idle':
        h.idle_keepalive()
    asks = [M.synth_burst(h.caller, F.for_bw(k, bw)) for k in (
        F.SESSION_TURN_REQUEST, F.SESSION_TURN_REQUEST_RESPONDER)]
    assert not any(np.array_equal(x, ask) for x in io.sent[sent:] for ask in asks)
    assert h._ask_owed and h.turn == A._TURN_PEER, io.msgs


def receive_final(bw, role):
    h, hi = sender(bw, 'responder' if role == 'initiator' else 'initiator')
    if role == 'initiator':
        c, ci = receiver(h)
    else:
        ci = _IO()
        c = A.VaraStationHandshake([h.called], ci, bw=bw)
        c.role, c.called, c.caller = role, h.called, h.caller
        c.state, c.turn = A.VaraState.CONNECTED, A._TURN_PEER
        c.chat = c._chat_profile = c._link_high_speed = False
        c._start_compression('OFF')
    timed_transmit(ci)
    c._queue_host_payload(b'queued before final')
    replay(c, hi)
    assert b''.join(ci.host) == b'last over'
    assert c._answer_owed == A._OWED_RELEASE
    return h, hi, c, ci


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('busy', [False, True], ids=['lost-request', 'gate-busy'])
def test_next_final_query_reanswers_lost_or_gate_deferred_request(bw, role, busy):
    h, hi, c, ci = receive_final(bw, role)
    solicit(h, hi)
    ci.receiving = busy
    sent = len(ci.sent)
    replay(c, hi)
    if busy and role == 'responder':
        assert len(ci.sent) == sent and c._ask_owed, ci.msgs
    else:
        assert len(ci.sent) == sent + 1 and c.turn == A._TURN_ASKED, ci.msgs
    ci.receiving = False
    feed(c, np.zeros(4800))
    c.on_rx_audio(np.zeros(4800))
    _Clock.t = max(_Clock.t, (h._answer_query_sent_at if h.role == 'responder'
                             else h._final_query_at) + A._FINAL_QUERY_REPLY_S + .02)
    assert h.intermediate_query_due_in() == pytest.approx(0)
    assert h.query_intermediate_answer(), hi.msgs
    sent, progress = len(ci.sent), c.progress
    replay(c, hi)
    assert len(ci.sent) == sent + 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], request(h))
    assert c.turn == A._TURN_ASKED and c.progress == progress
    assert b''.join(ci.host) == b'last over'
    replay(h, ci)
    assert h.payload_ledger.confirmed == 9 and h.payload_ledger.pending == 0


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_next_query_while_asked_preserves_unread_data_precedence(bw, role):
    h, hi, c, ci = receive_final(bw, role)
    solicit(h, hi)
    replay(c, hi)
    assert c.turn == A._TURN_ASKED
    c._rx_window.event('G')
    c._query_data_due = c._owed_block = True
    c._answer_owed = A._OWED_OVER
    c._rx_nak_form = 'missing'
    assert c._rx_window.A.kind == 'nak' and not c._rx_window.A.sent
    _Clock.t += 4
    assert h.query_intermediate_answer(), hi.msgs
    sent = len(ci.sent)
    replay(c, hi)
    assert len(ci.sent) == sent + 1, ci.msgs
    kind = (F.SESSION_OVER_NAK_SHORT if role == 'initiator'
            else F.SESSION_OVER_NAK_RESPONDER_SHORT)
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(kind, bw)))
    assert b''.join(ci.host) == b'last over'


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('queued', [False, True])
def test_fresh_request_rekeys_erased_handover_without_credit_or_progress(bw, role, queued):
    h, io = sender(bw, role)
    solicit(h, io)
    x = np.r_[on_air(request(h)), np.zeros(M.FS // 2)]
    feed(h, x)
    assert h._tx_pending is None and h.payload_ledger.confirmed == 9
    handover = io.sent[-1].copy()
    if queued:
        h.send(b'next delivery')
    debt = h.payload_ledger.pending
    sent, progress, progress_at = len(io.sent), h.progress, h._progress_at
    h.on_rx_audio(x)
    assert len(io.sent) == sent and h.progress == progress
    _Clock.t += 4
    feed(h, x)
    assert len(io.sent) == sent + 1, io.msgs
    np.testing.assert_array_equal(io.sent[-1], handover)
    assert h.payload_ledger.confirmed == 9 and h.payload_ledger.pending == debt
    assert (h.progress, h._progress_at) == (progress, progress_at)
    assert h.turn == A._TURN_PEER and h._handover_unheard
    h.on_rx_audio(x)
    assert len(io.sent) == sent + 1


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_gate_deferred_final_query_request_keys_in_its_turnaround(bw):
    h, hi, c, ci = receive_final(bw, 'responder')
    solicit(h, hi)
    ci.receiving = True
    _Clock.t = hi.at[-1] + len(hi.sent[-1]) / M.FS - len(on_air(hi.sent[-1])) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(1920)])
    assert c._ask_owed and len(ci.sent) == 1, ci.msgs
    ci.receiving = False
    feed(c, np.zeros(960))
    assert len(ci.sent) == 2 and c.turn == A._TURN_ASKED, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], request(h))
    feed(c, np.zeros(4800))
    assert len(ci.sent) == 2


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_wide_final_query_deadline_uses_public_scheduler(bw, role):
    h, io = sender(bw, role)
    pending_at = h._pending_answer_at
    solicit(h, io)
    delay = io.at[-1] - pending_at
    assert (1.30 <= delay <= 1.34) if role == 'initiator' else (2.22 <= delay <= 2.28)
    assert h.payload_ledger.pending == 9 and h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_refused_public_final_query_keeps_debt_and_a_future_deadline(bw, role):
    h, io = sender(bw, role)
    old = h._tx_pending
    due = h.intermediate_query_due_in()
    assert due is not None
    feed(h, np.zeros(round(due * M.FS) + 1))
    io.tx_went_out = lambda: False
    assert not h.query_intermediate_answer()
    assert h._tx_pending is old and h.payload_ledger.confirmed == 0
    assert h.payload_ledger.pending == 9
    assert h._final_query_attempts == h._answer_queries == 0
    assert h.intermediate_query_due_in() > 0


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_disconnect_acquire_cannot_credit_unread_final_after_intermediate(bw):
    h, hi = sender(bw, 'initiator', payload=b'x' * 89 + b'y' * 21, final=False)
    ci = _IO()
    c = A.VaraStationHandshake([h.called], ci, bw=bw)
    c.role, c.called, c.caller = 'responder', h.called, h.caller
    c.state, c.turn, c.chat = A.VaraState.CONNECTED, A._TURN_PEER, True
    c.over_continue = A.OVER_CONTINUE_BURST
    c._start_compression('OFF')
    timed_transmit(ci)
    replay(c, hi)
    assert b''.join(ci.host) == b'x' * 89
    replay(h, ci)
    assert h.payload_ledger.confirmed == 89 and h.payload_ledger.pending == 21
    assert A._phy.over_is_last(h._tx_pending[0], h.caller)
    h.send(b'next delivery')
    solicit(h, hi)
    _Clock.t = h._final_query_at + .2
    sent = len(ci.sent)
    c.disconnect(acquire=True)
    assert len(ci.sent) == sent and c._close_owed_at is not None, ci.msgs
    assert c._close_asked_at is None and c.turn == A._TURN_PEER
    for _ in range(31):
        feed(c, np.zeros(4800))
    assert c.state == A.VaraState.DISCONNECTED, ci.msgs
    assert len(ci.sent) == sent + 1
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_DISCONNECT_REQ_RESPONDER, bw)))
    replay(h, ci)
    assert h.payload_ledger.confirmed == 89 and h.payload_ledger.pending == 34


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('debt', ['over', 'delivery', 'unheard'])
def test_close_acquisition_hold_has_a_deadline_without_more_audio(bw, role, debt):
    h, io = sender(bw, role)
    h._tx_pending = None
    h.turn, h.chat = A._TURN_PEER, True
    h._answer_owed = A._OWED_OVER if debt == 'over' else None
    h._peer_delivery_open = debt == 'delivery'
    h._handover_unheard = debt == 'unheard'
    sent = len(io.sent)
    h.disconnect(acquire=True)
    assert len(io.sent) == sent and h._close_asked_at is None, io.msgs
    assert h.idle_poll_due_in() == pytest.approx(A._CLOSE_WAIT_S)
    _Clock.t += A._CLOSE_WAIT_S
    h.idle_keepalive()
    assert h.state == A.VaraState.DISCONNECTED and len(io.sent) == sent + 1


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('handover', [False, True], ids=['idle-only', 'explicit-release'])
def test_only_accepted_handover_retires_intermediate_delivery_hold(bw, handover):
    h, hi = pending(bw)
    c, ci = receiver(h)
    timed_transmit(ci)
    feed(c, on_air(hi.sent[-1]))
    assert c._peer_delivery_open and c._answer_owed == A._OWED_OVER
    if handover:
        release = M.synth_burst(c.called, F.for_bw(F.SESSION_TURN_RELEASE_RESPONDER, bw))
        feed(c, np.r_[on_air(release), np.zeros(4800)])
        assert c.turn == A._TURN_PEER
    idle = M.synth_tone_pairs(F.control_bursts(c.caller, bw, 1)[1])
    feed(c, np.r_[on_air(idle), np.zeros(4800)])
    assert not c._handover_unheard
    sent = len(ci.sent)
    c.send(b'new traffic')
    assert len(ci.sent) == sent + int(handover), ci.msgs
    assert c._peer_delivery_open is not handover


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('block', [960, 4800, 24000])
@pytest.mark.parametrize('lead', [.02, .06, .14])
def test_validated_request_handover_deadline_precedes_next_audio(bw, role, block, lead):
    h, io = sender(bw, role)
    unkey = solicit(h, io)
    x = request(h)
    samples = np.r_[np.zeros(round(lead * M.FS)), x, np.zeros(M.FS)]
    sent = len(io.sent)
    deadline = None
    for at in range(0, len(samples), block):
        feed(h, samples[at:at + block], block=block)
        if h._tx_pending is None:
            break
        due = h.intermediate_query_due_in()
        if due is not None and due < block / M.FS:
            deadline = _Clock.t + due
            _Clock.t += due + 1e-9
            assert h.query_intermediate_answer(), io.msgs
            break
    assert h._tx_pending is None and h.payload_ledger.confirmed == 9, io.msgs
    assert len(io.sent) == sent + 1 and h.turn == A._TURN_PEER
    if deadline is not None:
        assert io.at[-1] == pytest.approx(deadline, abs=1e-8)
    age = io.at[-1] - (unkey + lead + len(x) / M.FS)
    assert .19 <= age <= max(.21, block / M.FS)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('case', ['pending', 'query', 'wall', 'samples', 'discontinuity'])
def test_held_request_deadline_preserves_identity_and_freshness(bw, role, case):
    h, io = sender(bw, role)
    solicit(h, io)
    x = np.r_[np.zeros(round(.06 * M.FS)), request(h), np.zeros(M.FS)]
    feed(h, x[:72000], block=24000)
    due = h.intermediate_query_due_in()
    assert due is not None and 0 < due < .2
    old = h._tx_pending
    field = '_final_query' if role == 'initiator' else '_answer_query'
    _Clock.t += due + 1e-9
    if case == 'pending':
        h._tx_pending = tuple(list(old))
    elif case == 'query':
        setattr(h, field + '_for', tuple(list(old)))
    elif case == 'wall':
        _Clock.t += 1
    elif case == 'samples':
        setattr(h, field + '_samples', round((A._FINAL_QUERY_REPLY_S + 1) * M.FS))
    else:
        h.rx_discontinuity()
    h.query_intermediate_answer()
    assert h._tx_pending is not None and h.payload_ledger.confirmed == 0, io.msgs
    assert h.payload_ledger.pending == 9


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_deferred_close_keys_in_accepted_peer_handover(bw, role):
    h, io = sender(bw, role)
    h._tx_pending = None
    h.turn, h.chat, h._peer_delivery_open = A._TURN_PEER, True, True
    h.disconnect(acquire=True)
    sent = len(io.sent)
    kind = (F.SESSION_TURN_RELEASE_RESPONDER if role == 'initiator'
            else F.SESSION_TURN_RELEASE)
    feed(h, np.r_[on_air(M.synth_burst(h.called, F.for_bw(kind, bw))), np.zeros(4800)])
    assert h.state == A.VaraState.DISCONNECTED and len(io.sent) == sent + 1, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_deferred_close_keeps_deadline_after_delivery_hold_ends(bw, role):
    h, io = sender(bw, role)
    h._tx_pending = None
    h.turn, h.chat, h._peer_delivery_open = A._TURN_PEER, True, True
    h.disconnect(acquire=True)
    sent = len(io.sent)
    _Clock.t += .2
    h._peer_delivery_open = False
    assert h.idle_poll_due_in() == pytest.approx(A._CLOSE_WAIT_S - .2)
    _Clock.t += A._CLOSE_WAIT_S - .2
    h.idle_keepalive()
    assert h.state == A.VaraState.DISCONNECTED and len(io.sent) == sent + 1, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_chat_regrant_holds_host_ask_when_peer_data_and_query_are_lost(bw):
    hi = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], hi, bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn, h.chat = A.VaraState.CONNECTED, A._TURN_PEER, True
    h._start_compression('OFF')
    c, ci = receiver(h)
    c.chat, c.turn = True, A._TURN_OURS
    timed_transmit(hi)
    timed_transmit(ci)
    hi.final_timed = True
    assert c._release_turn()
    replay(h, ci)
    assert h.turn == A._TURN_OURS and h._tx_pending is None
    _Clock.t += .6
    h.idle_keepalive()
    replay(c, hi)
    assert c.turn == A._TURN_PEER and not c._handover_unheard
    h.send(b'last over')
    replay(c, hi)
    replay(h, ci)
    assert h._tx_pending is not None and h.payload_ledger.pending == 9, hi.msgs
    assert c.turn == A._TURN_PEER and not ci.host
    solicit(h, hi)
    _Clock.t = h._answer_query_sent_at + .2
    sent = len(ci.sent)
    c.send(b'caller traffic')
    assert len(ci.sent) == sent and c._ask_owed, ci.msgs
    c.idle_keepalive()
    feed(c, np.zeros(4800))
    assert len(ci.sent) == sent and not ci.host
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 9


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('short', [False, True])
def test_chat_request_from_held_empty_turn_receives_high_speed_data(bw, short):
    hi = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], hi, bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_PEER
    h.chat = h._link_high_speed = True
    h._expect_level = 6
    h._start_compression('OFF')
    c, ci = receiver(h)
    c.chat, c._link_high_speed = short, True
    c.tx_level = 6
    c._rec500 = A._phy.level_key(bw, 6)
    c._lead = A.VB6Rnd(F.lead_state(c.caller, False, bw))
    c.turn = A._TURN_OURS
    timed_transmit(hi)
    timed_transmit(ci)
    assert c._release_turn()
    c.send(b'farewell')
    assert c._ask_owed and c._handover_unheard
    replay(h, ci)
    assert h._holding_idle() and not h._lv_armed, hi.msgs
    replay(c, hi)
    assert c.turn == A._TURN_ASKED, ci.msgs
    replay(h, ci)
    assert h.turn == A._TURN_PEER and h._handover_unheard, hi.msgs
    replay(c, hi)
    assert c._tx_pending is not None and c.payload_ledger.pending == 8, ci.msgs
    replay(h, ci)
    assert b''.join(hi.host) == b'farewell', hi.msgs
    replay(c, hi)
    assert c.payload_ledger.confirmed == 8 and c.payload_ledger.pending == 0, ci.msgs


class _SegmentedRequests:
    def __init__(self, h, io, samples, block):
        from hfmodem.kestrel.monitor import Segmenter
        self.io, self.samples, self.block = io, samples, block
        self.segmenter = Segmenter()
        self.segmenter.push(np.random.default_rng(0).normal(0, .001, M.FS))
        self.start, self.cursor = _Clock.t, 0
        self.keyed_receiving = []
        self.rig = None
        self.hs = h

    def __getattr__(self, name):
        return getattr(self.io, name)

    @property
    def receiving(self):
        return bool(self.segmenter.inb)

    def tx(self, samples):
        self.keyed_receiving.append(self.receiving)
        self.io.tx(samples)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('block', [4800, 24000])
@pytest.mark.parametrize('driver', ['audio', 'mail'])
@pytest.mark.parametrize('invalid', [None, 'pending', 'query', 'wall', 'samples', 'discontinuity'])
def test_real_drivers_service_retained_request_inside_segmenter_tail(
        bw, role, block, driver, invalid, monkeypatch):
    from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
    from .test_idle_tick_reply_window import Modem
    import threading
    h, io = sender(bw, role)
    unkey = solicit(h, io)
    x = request(h)
    lead = .02 if block == 4800 else .06
    samples = np.r_[np.zeros(round(lead * M.FS)), x, np.zeros(M.FS)]
    transport = _SegmentedRequests(h, io, samples, block)
    h.io = transport
    sent = len(io.sent)
    invalidated = []

    def invalidate():
        if invalid is None or invalidated or h._held_final_request is None:
            return
        invalidated.append(invalid)
        field = '_final_query' if role == 'initiator' else '_answer_query'
        if invalid == 'pending':
            h._tx_pending = tuple(list(h._tx_pending))
        elif invalid == 'query':
            setattr(h, field + '_for', tuple(list(h._tx_pending)))
        elif invalid == 'wall':
            _Clock.t += 1
        elif invalid == 'samples':
            setattr(h, field + '_samples', round((A._FINAL_QUERY_REPLY_S + 1) * M.FS))
        else:
            h.rx_discontinuity()

    if driver == 'audio':
        from collections import deque
        from types import SimpleNamespace
        import queue
        from hfmodem.kestrel.host import station_runtime as SR
        monkeypatch.setattr(SR, 'time', _Clock)
        capture = SR.StationTransport.__new__(SR.StationTransport)
        capture.contact = SimpleNamespace(receive_boundary=(0, 0))
        capture._init_receive_state()
        capture.hs = h
        capture._segmenter, capture._bursts = transport.segmenter, deque()
        capture._ui_channel = SR.UIChannel(capture._segmenter)
        capture.before_step = invalidate

        class Frames(queue.Queue):
            def get(self, timeout):
                at = transport.start + (transport.cursor + block) / M.FS
                end = _Clock.t + timeout
                if at > end + 1e-9 or transport.cursor >= len(samples):
                    _Clock.t = end
                    raise queue.Empty
                _Clock.t = max(_Clock.t, at)
                start = transport.cursor
                transport.cursor += block
                return 0, start, samples[start:start + block]

        capture.frames = Frames()
        runtime = AudioHostRuntime(capture, clock=_Clock.monotonic)
        runtime.modem = Modem(h)
        until = _Clock.t + 1.8
        while h._tx_pending is not None and _Clock.t < until:
            runtime.step()
    else:
        from pathlib import Path
        root = next(p for p in Path(__file__).resolve().parents if (p / 'tools').is_dir())
        monkeypatch.syspath_prepend(str(root / 'tools'))
        import kestrel_connect as KC
        monkeypatch.setattr(KC.time, 'time', _Clock.monotonic)
        monkeypatch.setattr(KC.time, 'monotonic', _Clock.monotonic)
        capture = KC.AudioVaraIO.__new__(KC.AudioVaraIO)
        capture._buf, capture._consumed, capture._bursts = [], 0, []
        capture._lock = threading.Lock()
        capture._rx_generation, capture._rx_inflight, capture._gap_from = 0, False, None
        capture._seg, capture._band = transport.segmenter, KC.BandGate()
        capture._ui_channel = KC.UIChannel(capture._seg)
        capture._report_rx_gap = lambda x: None
        capture.hs = h

        def sleep(delay):
            _Clock.t += delay
            at = transport.start + (transport.cursor + block) / M.FS
            if at <= _Clock.t + 1e-9 and transport.cursor < len(samples):
                capture._buf.append(samples[transport.cursor:transport.cursor + block])
                transport.cursor += block

        def receive(timeout, hs):
            invalidate()
            return capture.next_rx_burst(timeout, hs)

        monkeypatch.setattr(KC.time, 'sleep', sleep)
        transport.next_rx_burst = receive

        class Client:
            @property
            def done(self):
                return h._tx_pending is None

        assert KC.mail_session(h, transport, Client(), timeout=1.8) is (invalid is None), io.msgs
    if invalid is not None:
        assert invalidated == [invalid]
        assert h._tx_pending is not None and h.payload_ledger.confirmed == 0, io.msgs
        assert h.payload_ledger.pending == 9 and h._held_final_request is None
        return
    assert h._tx_pending is None and h.payload_ledger.confirmed == 9, io.msgs
    assert h.payload_ledger.pending == 0 and h.turn == A._TURN_PEER
    assert len(io.sent) == sent + 1
    assert transport.keyed_receiving == [True]
    age = io.at[-1] - (unkey + lead + len(x) / M.FS)
    assert .19 <= age <= .21
    expected = (M.synth_burst(h.called, F.for_bw(F.SESSION_DRAINED, bw))
                if role == 'initiator' else M.synth_tone_pairs(F.turn_grant(h.called, bw)))
    np.testing.assert_array_equal(io.sent[-1], expected)


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_chat_drained_responder_grant_keeps_wholly_lost_data_pending(bw):
    h, hi = sender(bw, 'responder', level=6)
    c, ci = receiver(h)
    h.chat = c.chat = True
    c._link_high_speed = True
    c._expect_level, c._lv_owed = 6, True
    c._key(False)
    timed_transmit(ci)
    replay(c, hi)
    assert ci.sent, ci.msgs
    replay(h, ci)
    assert h._released_form == 'grant', hi.msgs
    replay(c, hi)
    assert c._holding_idle(), ci.msgs
    replay(h, ci)
    assert h._released_form == 'idle', hi.msgs
    replay(c, hi)
    c.send(b'last over')
    replay(h, ci)
    np.testing.assert_array_equal(hi.sent[-1], M.synth_burst(
        h.called, F.for_bw(F.SESSION_DRAINED_RESPONDER, bw)))
    replay(c, hi)
    assert c._tx_pending is not None and c.payload_ledger.pending == 9, ci.msgs
    assert not hi.host
    solicit(c, ci)
    c.send(b'later delivery')
    _Clock.t = c._final_query_at + .2
    sent = len(hi.sent)
    h.send(b'responder host traffic')
    if len(hi.sent) > sent:
        replay(c, hi)
    assert c.payload_ledger.confirmed == 0 and c.payload_ledger.pending == 23, ci.msgs
    assert len(hi.sent) == sent and h._ask_owed, hi.msgs
    assert not hi.host


_HANDOVER_SITES = [
    (role, site) for role in ('initiator', 'responder')
    for site in ('release', 'idle-release', 'release-on-ask', 'acked-release')
] + [
    ('initiator', site) for site in ('final-drained', 'repeat-drained',
                                    'caller-grant', 'caller-idle-answer', 'asked-release',
                                    'asked-regrant', 'accepted-release', 'drained-poll-ask',
                                    'peer-control-grant', 'idle-release-retry', 'idle-empty-release')
] + [
    ('responder', site) for site in ('initial-grant', 'high-grant', 'high-idle',
                                   'final-grant', 'request-grant', 'repeat-grant',
                                   'drained-grant', 'caller-control-grant',
                                   'held-empty-ask', 'repeat-initial-ack')
]


def handover_site(bw, role, site, already_peer=False, refused=False, previous=False):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.caller = role, 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    h.chat = True
    h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    if role == 'initiator':
        h.step = A._I_CONNECTED
    h._progressed()
    io.kinds = []
    io.tx_metadata = lambda **m: io.kinds.append(m['packet_kind'])
    timed_transmit(io)
    if site in ('final-drained', 'final-grant', 'acked-release'):
        h._queue_host_payload(b'previous delivery')
        assert h._tx_data_over()
        h._final_ack_confirmed = h._tx_pending
    if already_peer:
        h.turn = A._TURN_PEER
    if previous:
        h._peer_over = 1
        h._over_answer, h._answer_owed = ('control', 0), A._OWED_RELEASE
    if refused:
        io.tx_went_out = lambda: False
    if site == 'release':
        h._release_turn()
    elif site == 'idle-release':
        h._release_turn(idle=True)
    elif site == 'release-on-ask':
        h._release_on_ask()
    elif site == 'final-drained':
        h._finish_final_answer()
    elif site == 'repeat-drained':
        h.turn, h._final_request_granted = A._TURN_PEER, True
        h._took_turn_request(raw=True)
    elif site == 'caller-grant':
        h._grant_held_turn()
    elif site == 'caller-idle-answer':
        h._answer_final_ack_idle()
    elif site == 'initial-grant':
        h._initial_answer_turn = True
        h._release_turn()
    elif site in ('high-grant', 'high-idle'):
        h._last_tx_level = 6
        h._release_turn(idle=site == 'high-idle')
    elif site == 'final-grant':
        h._finish_responder_final_request()
    elif site in ('request-grant', 'repeat-grant'):
        if site == 'repeat-grant':
            h.turn, h._final_request_granted = A._TURN_PEER, True
        h._answer_responder_control(F.SESSION_TURN_REQUEST)
    elif site == 'acked-release':
        h._took_control_burst()
    elif site in ('asked-release', 'asked-regrant'):
        if site == 'asked-regrant':
            h.turn, h._released = A._TURN_PEER, True
        h._took_turn_request(raw=True)
    elif site == 'accepted-release':
        h._took_responder_release()
    elif site == 'drained-poll-ask':
        # An empty drained handover is held and polled; the peer's ask is what
        # the release then answers  [test_drained_empty_turn_1031].
        h.turn, h._answer_owed = A._TURN_PEER, A._OWED_RELEASE
        h._took_drained_handover()
        assert h.turn == A._TURN_OURS and h._poll_at is not None, io.msgs
        h._took_turn_request(raw=True)
    elif site == 'peer-control-grant':
        h.turn = A._TURN_PEER
        h._took_peer_control(0)
    elif site in ('idle-release-retry', 'idle-empty-release'):
        h._release_owed = site == 'idle-release-retry'
        h.idle_keepalive()
    elif site in ('held-empty-ask', 'repeat-initial-ack'):
        if site == 'held-empty-ask':
            h.turn, h._poll_at = A._TURN_OURS, _Clock.t
        else:
            h.turn = A._TURN_PEER
            h._initial_answer_turn = h._released = True
        x = M.synth_tone_pairs(F.control_bursts(h.caller, bw, 0)[0])
        feed(h, np.r_[on_air(x), np.zeros(4800)])
    elif site == 'caller-control-grant':
        h.turn, h._released_form = A._TURN_PEER, 'f13'
        h._caller_holds_control(0)
    elif site == 'drained-grant':
        h.turn, h._released_form = A._TURN_PEER, 'idle'
        h._caller_holds_control(0)
    else:
        raise AssertionError(site)
    assert io.kinds[-1] in {
        F.SESSION_TURN_RELEASE.name, F.SESSION_TURN_RELEASE_RESPONDER.name,
        F.SESSION_DRAINED.name, F.SESSION_DRAINED_RESPONDER.name,
        F.SESSION_TURN_IDLE_SHORT.name, F.SESSION_TURN_IDLE_RESPONDER_SHORT.name,
        'session-turn-grant', 'caller-turn-grant',
        'session-post-delivery-idle', 'caller-idle-answer',
    }, io.msgs
    return h, io


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('chat', [False, True], ids=['chat-off', 'chat-on'])
def test_close_after_keyed_empty_release_only_holds_turn_acquisition(bw, role, chat):
    h, io = handover_site(bw, role, 'release')
    h.chat = chat
    assert h.turn == A._TURN_PEER and h._handover_unheard
    assert not h._peer_delivery_open and h._answer_owed is None
    assert h._tx_pending is None and not h._txq
    sent, now = len(io.sent), _Clock.t
    h.disconnect(acquire=True)
    if chat:
        assert len(io.sent) == sent and h.state == A.VaraState.CONNECTED, io.msgs
        assert h._close_owed_at == now and h._close_asked_at is None
        assert h.idle_poll_due_in() == pytest.approx(A._CLOSE_WAIT_S)
    else:
        assert len(io.sent) == sent + 1 and h.state == A.VaraState.DISCONNECTED, io.msgs
        assert io.at[-1] == now
        kind = (F.SESSION_DISCONNECT_REQ if role == 'initiator'
                else F.SESSION_DISCONNECT_REQ_RESPONDER)
        np.testing.assert_array_equal(io.sent[-1], M.synth_burst(h.called, F.for_bw(kind, bw)))
        assert h._close_owed_at is None and h._close_asked_at is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role,site,already_peer', [
    (role, site, peer) for role, site in _HANDOVER_SITES for peer in (False, True)
    if not peer or site not in ('final-drained', 'final-grant', 'idle-release',
                                'high-idle', 'caller-idle-answer', 'acked-release',
                                'idle-release-retry', 'idle-empty-release')
])
def test_every_keyed_handover_holds_host_ask_for_lost_peer_data(bw, role, site, already_peer):
    h, io = handover_site(bw, role, site, already_peer)
    peer, pi = sender(bw, 'responder' if role == 'initiator' else 'initiator')
    solicit(peer, pi)
    peer.send(b'later delivery')
    _Clock.t = (peer._answer_query_sent_at if peer.role == 'responder'
                else peer._final_query_at) + .2
    sent = len(io.sent)
    h.send(b'new host traffic')
    if len(io.sent) > sent:
        replay(peer, io)
    assert peer.payload_ledger.confirmed == 0 and peer.payload_ledger.pending == 23, pi.msgs
    assert len(io.sent) == sent and h._ask_owed, io.msgs
    h.idle_keepalive()
    feed(h, np.zeros(4800))
    assert len(io.sent) == sent and not io.host, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role,site', _HANDOVER_SITES)
def test_refused_handover_does_not_arm_unheard_hold(bw, role, site):
    h, io = handover_site(bw, role, site, refused=True)
    assert not h._handover_unheard, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('debt', [None, 'delivery', 'over', 'unread'])
def test_bracket_only_three_recognized_idles_end_only_nothing_heard_hold(bw, debt):
    h, io = handover_site(bw, 'initiator', 'release')
    h.chat = False
    if debt == 'delivery':
        h._peer_delivery_open = True
    elif debt == 'over':
        from hfmodem.kestrel.arq import rxwindow as RW
        h._rx_window.A = RW.Answer(0, 0, 'continue', sent=True)
        h._answer_owed = A._OWED_OVER
    elif debt == 'unread':
        h._owe_unread_over()
    h.send(b'queued host traffic')
    assert h._ask_owed and not h._stream_owns_answer
    idle = on_air(M.synth_burst(h.called, F.for_bw(F.SESSION_RESPONDER_OVER_IDLE, bw)))
    for i in range(3):
        _Clock.t += 4
        sent = len(io.sent)
        h.on_rx_audio(idle)
        assert not h._handover_unheard, io.msgs
        requests = sum(np.array_equal(x, M.synth_burst(
            h.caller, F.for_bw(F.SESSION_TURN_REQUEST, bw))) for x in io.sent[sent:])
        assert requests == int(debt is None), (i, io.msgs)
    assert h._peer_delivery_open == (debt == 'delivery')
    if debt == 'unread' and bw == '2750':
        assert h.state == A.VaraState.DISCONNECTED
    else:
        assert h._nak_data_due == (debt == 'unread')
        if debt == 'over':
            assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
            assert h._rx_window.T is None
    assert io.host == []


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role,frame', [
    ('initiator', 'idle-response'), ('initiator', 'responder-idle'),
    ('initiator', 'control-poll'), ('responder', 'caller-keepalive'),
    ('responder', 'caller-keepalive-b'),
])
def test_bracket_peer_heard_matches_stream_without_retiring_delivery(bw, role, frame):
    h, io = handover_site(bw, role, 'release')
    h.chat = False
    h._peer_delivery_open = True
    if frame == 'control-poll':
        x = M.synth_tone_pairs(F.control_bursts(h.caller, bw, 1)[1])
    else:
        kind = {'idle-response': F.SESSION_IDLE_RESPONSE,
                'responder-idle': F.SESSION_RESPONDER_IDLE,
                'caller-keepalive': F.SESSION_KEEPALIVE_A,
                'caller-keepalive-b': F.SESSION_KEEPALIVE_B}[frame]
        x = M.synth_burst(h.called, F.for_bw(kind, bw))
    h.on_rx_audio(on_air(x))
    assert not h._handover_unheard, io.msgs
    assert h._peer_delivery_open
    sent = len(io.sent)
    h.send(b'host write after recognized peer frame')
    assert len(io.sent) == sent and h._ask_owed, io.msgs
    assert not h._stream_owns_answer and not h._stream_owns_responder_control


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_empty_turn_confirmation_allows_host_ask_without_another_grant(bw, role):
    h, io = handover_site(bw, role, 'idle-release', already_peer=True)
    assert not h._handover_unheard
    _Clock.t = h._ask_not_before + .2
    sent = len(io.sent)
    h.send(b'host traffic')
    assert len(io.sent) == sent + 1, io.msgs
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 12


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role,site', [
    ('initiator', 'caller-grant'), ('responder', 'drained-grant'),
])
@pytest.mark.parametrize('previous', [False, True], ids=['first-delivery', 'previous-ack'])
def test_regrant_query_cannot_reuse_previous_delivery_ack(bw, role, site, previous):
    h, io = handover_site(bw, role, site, already_peer=True, previous=previous)
    peer, pi = sender(bw, 'responder' if role == 'initiator' else 'initiator')
    solicit(peer, pi)
    _Clock.t = (peer._answer_query_sent_at if peer.role == 'responder'
                else peer._final_query_at) + .2
    h.send(b'host write with fresh DATA unread')
    sent = len(io.sent)
    feed(h, np.r_[on_air(pi.sent[-1]), np.zeros(4800)])
    assert len(io.sent) == sent + 1, io.msgs
    reply = io.sent[-1]
    nak = F.SESSION_OVER_NAK_SHORT if role == 'initiator' else F.SESSION_OVER_NAK_RESPONDER_SHORT
    np.testing.assert_array_equal(reply, M.synth_burst(h.called, F.for_bw(nak, bw)))
    replay(peer, io)
    assert peer.payload_ledger.confirmed == 0 and peer.payload_ledger.pending == 9, pi.msgs
    assert not io.host


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('signal', ['partial-idle', 'silence'])
def test_unfinished_idle_or_silence_cannot_end_nothing_heard_hold(bw, route, signal):
    h, io = handover_site(bw, 'initiator', 'release')
    h.send(b'host traffic')
    x = M.synth_burst(h.called, F.for_bw(F.SESSION_RESPONDER_OVER_IDLE, bw))
    x = x[:round(len(x) * .6)] if signal == 'partial-idle' else np.zeros(len(x))
    sent = len(io.sent)
    if route == 'stream':
        feed(h, np.r_[np.zeros(4800), x])
    else:
        h.on_rx_audio(x)
    assert h._handover_unheard and len(io.sent) == sent, io.msgs
