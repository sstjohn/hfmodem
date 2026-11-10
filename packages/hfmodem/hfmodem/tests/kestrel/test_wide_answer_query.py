# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Wide responder recovery preserves the pending over's identity and phase."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock, _IO  # noqa: F401
from .test_limited_responder_query_recovery import on_air
from .test_responder_control_stream import feed


def pending(bw, payload=bytes(range(110))):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h.send(payload)
    assert not A._phy.over_is_last(h._tx_pending[0], h.caller)
    return h, io


def receiver(h):
    io = _IO()
    c = A.VaraStationHandshake([h.caller], io, bw=h.bw)
    c.role, c.called, c.caller = 'initiator', h.called, h.caller
    c.state, c.step, c.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_PEER
    c.over_continue = A.OVER_CONTINUE_BURST
    c._start_compression('OFF')
    return c, io


def query(h, io, rung=1, phase=0):
    assert h._query_answerer_over(), io.msgs
    expected = M.synth_tone_pairs(F.answer_query(h.called, h.bw, phase))
    np.testing.assert_array_equal(io.sent[-1], expected)


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_two_kestrels_recover_lost_first_ack_once_without_duplicate_delivery(bw):
    payload = bytes(range(110))
    h, hi = pending(bw, payload)
    c, ci = receiver(h)
    first = h._tx_pending
    feed(c, on_air(hi.sent[-1]))
    h._pending_answer_at = c._unkeyed_at
    assert len(ci.sent) == 1 and len(ci.sent[-1]) == 7 * M.HOP + M.STRIDE
    assert b''.join(ci.host) == payload[:len(first[0]) - 1], ci.msgs
    # The first ACK is erased; deadlines are measured from the over's unkey.
    _Clock.t = h._pending_answer_at
    assert h.intermediate_query_due_in() == pytest.approx(A._ANSWER_QUERY_S)
    assert h._answer_query_due_in() == pytest.approx(A._ANSWER_QUERY_S)
    _Clock.t += A._ANSWER_QUERY_S
    assert h.intermediate_query_due_in() == pytest.approx(0)
    assert h.query_intermediate_answer(), hi.msgs
    np.testing.assert_array_equal(hi.sent[-1], M.synth_tone_pairs(
        F.over_continue_frames(h.called, bw, (11,))[11][1]))
    sent = len(ci.sent)
    # The fixture's tx is instantaneous; replay from the real key-down time.
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    assert len(ci.sent) == sent + 1, ci.msgs
    reply = on_air(ci.sent[-1])
    feed(h, reply)
    assert h.payload_ledger.confirmed == len(first[0]) - 1, hi.msgs
    assert h._tx_pending[1] == 2 and h._tx_retries == 0
    second, sent = h._tx_pending, len(hi.sent)
    feed(h, reply)
    assert h._tx_pending is second and len(hi.sent) == sent
    assert h.payload_ledger.confirmed == len(first[0]) - 1
    while h._tx_pending is not None:
        feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS // 2)])
        feed(h, on_air(ci.sent[-1]))
    assert b''.join(ci.host) == payload, ci.msgs
    assert h.payload_ledger.confirmed == len(payload)


def timed_transmit(io):
    tx = io.tx

    def transmit(samples):
        tx(samples)
        _Clock.t += len(samples) / M.FS

    io.tx = transmit


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_wide_query_ladder_cadence_phases_and_budget(bw):
    h, io = pending(bw, bytes(range(256)) * 3)
    first_credit = len(h._tx_pending[0]) - 1
    timed_transmit(io)
    for phase in (0, 1):
        for rung in range(1, A._BW500_QUERY_MAX + 1):
            deadline = (h._pending_answer_at + A._ANSWER_QUERY_S if rung == 1 else
                        h._answer_query_at + A._FINAL_QUERY_REPLY_S)
            assert h._answer_query_due_in() == pytest.approx(max(0, deadline - _Clock.t))
            _Clock.t = deadline
            assert h.intermediate_query_due_in() == pytest.approx(0)
            query(h, io, rung, phase)
            assert h._answer_query_sent_at == pytest.approx(
                io.at[-1] + len(io.sent[-1]) / M.FS)
            assert h._caller_query_answer_fresh()
            _Clock.t += A._FINAL_QUERY_REPLY_S - .01
            assert h._caller_query_answer_fresh()
            assert h._answer_query_due_in() == pytest.approx(.01)
            assert h.state is A.VaraState.CONNECTED
            if phase == 0 and rung == 3:
                feed(h, on_air(M.synth_tone_pairs(
                    F.over_continue_frames(h.called, bw, (0,))[0][0])))
                assert h._tx_pending[1] == 2
                break
    assert h._answer_queries == A._BW500_QUERY_MAX
    assert h.payload_ledger.confirmed == first_credit
    _Clock.t = h._answer_query_at + A._FINAL_QUERY_REPLY_S
    assert not h.query_intermediate_answer()
    assert h.state is A.VaraState.DISCONNECTED, io.msgs
    assert any('query budget' in msg for msg in io.msgs)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('stale', [None, 'wall', 'samples'])
def test_long_caller_query_answer_requires_fresh_window_and_credits_once(bw, stale):
    h, io = pending(bw)
    first = h._tx_pending
    query(h, io)
    if stale == 'wall':
        _Clock.t += A._FINAL_QUERY_REPLY_S + 1
    if stale == 'samples':
        h._answer_query_samples = (A._FINAL_QUERY_REPLY_S + 1) * M.FS
    reply = on_air(M.synth_burst(h.called, F.for_bw(A._CALLER_QUERY_ANSWERS[0], bw)))
    sent = len(io.sent)
    feed(h, reply)
    assert h.payload_ledger.confirmed == (0 if stale else len(first[0]) - 1), io.msgs
    assert len(io.sent) == sent + (not stale)
    after = h._tx_pending
    feed(h, reply)
    assert h._tx_pending is after
    assert h.payload_ledger.confirmed == (0 if stale else len(first[0]) - 1)


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_short_caller_nak_draws_one_resend_without_credit(bw):
    h, io = pending(bw)
    first, debt = h._tx_pending, h.payload_ledger.pending
    query(h, io)
    sent = len(io.sent)
    reply = on_air(M.synth_burst(h.called, F.for_bw(F.SESSION_OVER_NAK_SHORT, bw)))
    feed(h, reply)
    assert len(io.sent) == sent + 1 and h._tx_retries == 1, io.msgs
    assert h._tx_pending[1] == first[1]
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == debt
    feed(h, reply)
    assert len(io.sent) == sent + 1 and h._tx_retries == 1
    assert h.payload_ledger.confirmed == 0


def test_bw500_queries_keep_eight_symbols_and_original_gap():
    h, io = pending('500')
    timed_transmit(io)
    _Clock.t = h._pending_answer_at + A._ANSWER_QUERY_S
    for rung in (1, 2, 3):
        query(h, io, rung)
        assert len(io.sent[-1]) == 7 * M.HOP + M.STRIDE
        assert h._answer_query_due_in() == pytest.approx(A._BW500_QUERY_GAP_S)
        previous_unkey = _Clock.t
        _Clock.t += A._BW500_QUERY_GAP_S
        assert h._answer_query_due_in() == pytest.approx(0)
        assert _Clock.t - previous_unkey == pytest.approx(A._BW500_QUERY_GAP_S)


def damaged_continue(h, case):
    pairs = F.over_continue_frames(h.called, h.bw, (0,))[0][0]
    erased = {'first': (0,), 'interior': (4,), 'last': (7,), 'two': (1, 4),
              'wrong': (), 'cut': (4,)}[case]
    x = M.synth_tone_pairs(pairs)
    for i in erased:
        symbol = M.synth_tone_pairs((pairs[i],))
        x[i * M.HOP:i * M.HOP + M.STRIDE] -= symbol
    if case == 'wrong':
        i = 4
        wrong = (M.band_for(h.bw)[0] + 3, M.band_for(h.bw)[1] - 3)
        x[i * M.HOP:i * M.HOP + M.STRIDE] += (
            M.synth_tone_pairs((wrong,)) - M.synth_tone_pairs((pairs[i],)))
    x = on_air(x)
    x += np.random.default_rng(18).normal(0, .001, len(x))
    track = A._top3_track(x, band=h._band)
    _, at = A._frame_fit(A._band_power(x, h._band), pairs, h._band)
    indices = np.rint((at + np.arange(8) * M.HOP + M._WOFF) / A._ACK_GRID).astype(int)
    clear = track[1][indices, 1]
    weak = clear < A._CLEAR_DB
    assert np.flatnonzero(weak).tolist() == list(erased), clear
    heard = np.sort(track[0][indices, :2], axis=1)
    exact = np.all(heard == pairs, axis=1)
    assert np.all(exact[~weak] == (np.arange(8)[~weak] != 4 if case == 'wrong' else True))
    if case == 'cut':
        x = x[4800 + M.HOP:]
    return x


@pytest.mark.parametrize('bw', ['2300', '2750', '500'])
@pytest.mark.parametrize('case', ['first', 'interior', 'last', 'two', 'wrong', 'cut'])
def test_continue_accepts_only_one_wide_interior_erasure(bw, case):
    h, io = pending(bw)
    first = h._tx_pending
    x = damaged_continue(h, case)
    feed(h, x)
    credit = len(first[0]) - 1 if bw != '500' and case == 'interior' else 0
    assert h.payload_ledger.confirmed == credit, io.msgs
    after, sent = h._tx_pending, len(io.sent)
    feed(h, x)
    assert h.payload_ledger.confirmed == credit
    assert h._tx_pending is after and len(io.sent) == sent


def final_pending(bw):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h.send(b'last over')
    assert A._phy.over_is_last(h._tx_pending[0], h.caller)
    return h, io


def caller_turn_request(h, kind):
    kind = F.for_bw(kind, h.bw)
    assert kind.keyed_by == 'caller'
    return on_air(M.synth_burst(h.caller, kind))


@pytest.mark.parametrize('bw', ['2300', '2750', '500'])
@pytest.mark.parametrize('case', ['none', 'wall', 'samples', 'nonfinal', 'other-over'])
def test_caller_turn_request_cannot_settle_pending_data(bw, case):
    h, io = pending(bw) if case == 'nonfinal' else final_pending(bw)
    _Clock.t = h._pending_answer_at + A._ANSWER_QUERY_S
    if case != 'none':
        query(h, io)
    if case == 'wall':
        _Clock.t += A._FINAL_QUERY_REPLY_S + 1
    if case == 'samples':
        h._answer_query_samples = (A._FINAL_QUERY_REPLY_S + 1) * M.FS
    if case == 'other-over':
        h._answer_query_for = tuple(list(h._tx_pending))
    old, sent, progress = h._tx_pending, len(io.sent), h.progress
    debt = h.payload_ledger.pending
    reply = caller_turn_request(h, F.SESSION_TURN_REQUEST)
    for _ in range(2):
        feed(h, reply)
        assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == debt, io.msgs
        assert h._tx_pending is old and h._tx_retries == 0
        assert h.turn == A._TURN_OURS and h.progress == progress
        assert len(io.sent) == sent


@pytest.mark.parametrize('kind', [F.SESSION_TURN_REQUEST, F.SESSION_TURN_REQUEST_SHORT])
def test_bw500_caller_turn_request_does_not_settle_fresh_final_query(kind):
    h, io = final_pending('500')
    query(h, io)
    old, debt, sent, progress = h._tx_pending, h.payload_ledger.pending, len(io.sent), h.progress
    for _ in range(2):
        feed(h, caller_turn_request(h, kind))
        assert h._tx_pending is old and h.payload_ledger.pending == debt, io.msgs
        assert h.payload_ledger.confirmed == 0 and h._tx_retries == 0
        assert h.turn == A._TURN_OURS and h.progress == progress
        assert len(io.sent) == sent


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('recovery_rung', [2, 3])
def test_two_kestrels_later_query_rungs_credit_once_without_resend(bw, recovery_rung):
    h, hi = pending(bw)
    c, ci = receiver(h)
    first = h._tx_pending
    feed(c, on_air(hi.sent[-1]))
    h._pending_answer_at = c._unkeyed_at
    delivered = b''.join(ci.host)
    # Lose the original ACK and each query reply before the selected rung.
    for rung in range(1, recovery_rung + 1):
        _Clock.t = (h._pending_answer_at + A._ANSWER_QUERY_S if rung == 1 else
                    h._answer_query_at + A._FINAL_QUERY_REPLY_S)
        query(h, hi, rung)
        sent = len(ci.sent)
        _Clock.t -= len(hi.sent[-1]) / M.FS
        feed(c, on_air(hi.sent[-1]))
        assert len(ci.sent) == sent + 1, ci.msgs
        assert b''.join(ci.host) == delivered
        assert h.payload_ledger.confirmed == 0 and h._tx_pending is first
    reply = on_air(ci.sent[-1])
    feed(h, reply)
    assert h.payload_ledger.confirmed == len(delivered) == len(first[0]) - 1, hi.msgs
    assert h._tx_pending[1] == 2 and h._tx_retries == 0
    second, sent = h._tx_pending, len(hi.sent)
    feed(h, reply)
    assert h._tx_pending is second and len(hi.sent) == sent
    assert h.payload_ledger.confirmed == len(delivered)


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_limited_wide_frame12_query_repeats_second_ack_once_without_resend(bw):
    h, hi = pending(bw, bytes(range(256)) * 3)
    c, ci = receiver(h)
    feed(c, on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    first_credit = h.payload_ledger.confirmed
    assert h._tx_pending[1] == 2
    second = h._tx_pending
    feed(c, on_air(hi.sent[-1]))
    delivered = b''.join(ci.host)
    assert c._peer_received_phase == c._peer_answered == 1
    # Lose the second ACK; frame 12 must repeat it instead of requesting DATA.
    query(h, hi, phase=1)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    assert len(ci.sent) == sent + 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_RESPONSE, bw)))
    reply = on_air(ci.sent[-1])
    feed(h, reply)
    assert h.payload_ledger.confirmed == first_credit + len(second[0]) - 1, hi.msgs
    assert h.payload_ledger.confirmed == len(delivered)
    assert h._tx_pending[1] == 3 and h._tx_retries == 0
    third, sent = h._tx_pending, len(hi.sent)
    feed(h, reply)
    assert h._tx_pending is third and len(hi.sent) == sent
    assert h.payload_ledger.confirmed == len(delivered) and b''.join(ci.host) == delivered


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_lost_final_and_first_query_with_caller_traffic_requests_missing_data(bw):
    hi = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], hi, bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_PEER
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h.send(b'last over')
    c, ci = receiver(h)
    # The caller grants a turn before the responder's wholly lost nine-byte final.
    c.turn = A._TURN_OURS
    assert c._release_turn()
    feed(h, on_air(ci.sent[-1]))
    assert h.turn == A._TURN_OURS and A._phy.over_is_last(h._tx_pending[0], h.caller)
    assert c.turn == A._TURN_PEER and c._query_data_due and not c._nak_data_due
    c._txq = [b'caller traffic queued']
    assert c._owes_turn()
    assert h.payload_ledger.pending == 9 and not ci.host
    old, debt = h._tx_pending, h.payload_ledger.pending
    _Clock.t = h._pending_answer_at + A._ANSWER_QUERY_S
    query(h, hi)
    # Lose the first query as well. The next rung stays an eight-symbol query.
    _Clock.t = h._answer_query_at + A._FINAL_QUERY_REPLY_S
    query(h, hi, rung=2)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    assert len(ci.sent) == sent + 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_NAK_SHORT, bw)))
    assert c.turn == A._TURN_PEER and c._rx_resend is not None
    assert not any('asked for the turn' in msg for msg in ci.msgs), ci.msgs
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == debt, hi.msgs
    assert h._tx_pending[1] == old[1] and h._tx_retries == 1
    assert not ci.host


def lost_final_behind_deferred_ask(bw):
    hi = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], hi, bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_PEER
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h.send(b'last over')
    c, ci = receiver(h)
    c.turn = A._TURN_OURS
    timed_transmit(hi)
    timed_transmit(ci)
    assert c._release_turn()
    _Clock.t = ci.at[-1] - 4800 / M.FS
    feed(h, on_air(ci.sent[-1]))
    assert h.turn == A._TURN_OURS and h.payload_ledger.pending == 9
    assert A._phy.over_is_last(h._tx_pending[0], h.caller)
    # Lose the entire final DATA. A busy transport defers the host's ask until
    # the next receive callback, which contains the second query in one block.
    ci.receiving = True
    c.send(b'caller traffic queued')
    assert c._ask_owed and c.turn == A._TURN_PEER
    ci.receiving = False
    assert c._query_data_due and not c._nak_data_due
    _Clock.t = h._pending_answer_at + A._ANSWER_QUERY_S
    query(h, hi)
    # Lose the entire first query too.
    _Clock.t = h._answer_query_at + A._FINAL_QUERY_REPLY_S
    query(h, hi, rung=2)
    sent = len(ci.sent)
    _Clock.t = hi.at[-1] - 4800 / M.FS
    feed(c, on_air(hi.sent[-1]), block=24000)
    assert len(ci.sent) == sent + 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_NAK_SHORT, bw)))
    assert c.turn == A._TURN_PEER and c._rx_resend is not None
    assert not ci.host
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 9
    old, sent = h._tx_pending, len(hi.sent)
    nak = on_air(ci.sent[-1])
    _Clock.t = ci.at[-1] - 4800 / M.FS
    feed(h, nak)
    assert h._tx_pending[1] == old[1] and h._tx_retries == 1, hi.msgs
    assert len(hi.sent) == sent + 1
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 9
    feed(h, nak)
    # The NAK starts 0.183 s after retry unkey (peer: 0.075–0.126 s plus
    # margin), so it is a new NAK and earns a second retry.
    assert len(hi.sent) == sent + 2 and h._tx_retries == 2
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 9
    return h, hi, c, ci


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_deferred_host_ask_reads_query_before_acknowledging_lost_final(bw):
    lost_final_behind_deferred_ask(bw)


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_lost_final_behind_deferred_ask_is_delivered_once(bw):
    h, hi, c, ci = lost_final_behind_deferred_ask(bw)
    resend, resend_at = hi.sent[-1], hi.at[-1]
    _Clock.t = resend_at - 4800 / M.FS
    feed(c, np.r_[on_air(resend), np.zeros(M.FS // 2)])
    assert b''.join(ci.host) == b'last over', '\n'.join(ci.msgs)
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == 9
    _Clock.t = ci.at[-1] - 4800 / M.FS
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 9 and h.payload_ledger.pending == 0, hi.msgs
    assert h._tx_pending is None
    feed(c, np.r_[on_air(resend), np.zeros(M.FS // 2)])
    assert b''.join(ci.host) == b'last over' * 2


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('high_speed', [False, True])
@pytest.mark.parametrize('route', ['direct-send', 'stream', 'bracket', 'idle-cadence'])
def test_idle_peer_query_data_due_does_not_block_host_turn_request(bw, high_speed, route):
    h = A.VaraStationHandshake(['W9SSJ'], _IO(), bw=bw)
    h.role, h.called, h.caller = 'responder', 'W9SSJ', 'W1AW'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    assert h._tx_pending is None and not h._txq
    c, ci = receiver(h)
    c._link_high_speed = high_speed
    c.turn = A._TURN_OURS
    assert c._release_turn()
    assert c._query_data_due and not c._nak_data_due and c._rx_resend is None
    h._tx_control_burst(1)
    feed(c, np.r_[on_air(h.io.sent[-1]), np.zeros(M.FS // 2)])
    sent = len(ci.sent)
    _Clock.t = c._ask_not_before + .1
    ci.receiving = route != 'direct-send'
    c.send(b'caller traffic queued')
    ci.receiving = False
    if route != 'direct-send':
        assert c._ask_owed and c.turn == A._TURN_PEER
        _Clock.t = max(_Clock.t, c._ask_not_before) + .1
        if route == 'stream':
            feed(c, np.zeros(4800))
        elif route == 'bracket':
            c.on_rx_audio(np.zeros(4800))
        else:
            c.idle_keepalive()
    assert len(ci.sent) == sent + 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.caller, F.for_bw(F.SESSION_TURN_REQUEST, bw)))
    assert c.turn == A._TURN_ASKED and not c._ask_owed
    assert c._query_data_due and not c._nak_data_due and c._rx_resend is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('high_speed', [False, True])
def test_wide_frame12_with_second_over_unread_requests_missing_data(bw, high_speed):
    h, hi = pending(bw, bytes(range(256)) * 3)
    c, ci = receiver(h)
    h._link_high_speed = c._link_high_speed = high_speed
    if high_speed:
        h._lead = A.VB6Rnd(F.lead_state(h.caller, True, bw))
    h._key(False)
    feed(c, on_air(hi.sent[-1]))
    # Recover the first over through its query before losing the second DATA.
    query(h, hi)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    _Clock.t = ci.at[-1] - 4800 / M.FS
    feed(h, on_air(ci.sent[-1]))
    first_credit = h.payload_ledger.confirmed
    second = h._tx_pending
    assert second[1] == 2 and c._peer_received_phase == c._peer_answered == 0
    assert not high_speed or c._lv_owed
    # Entire second DATA is lost: the same frame 12 now names unread DATA.
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, phase=1)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS // 2)])
    assert len(ci.sent) == sent + 1, ci.msgs
    np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
        c.called, F.for_bw(F.SESSION_OVER_NAK_SHORT, bw)))
    assert c._rx_window.A.kind == 'nak' and c._rx_window.A.sent
    assert c._rx_window.A.phase == 1 and not c._rx_window.trusted
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == first_credit, hi.msgs
    assert h._tx_pending[1] == second[1] and h._tx_retries == 1
