# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy, rxwindow as RW
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.rx import varahf500 as rx500
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock, _feed  # noqa: F401
from .receive_window_helpers import (station, window, replay_window, receive, feed,
                                     assert_missing_nak, assert_bracket_recovery_nak,
                                     burst, slot, top_window, query500,
                                     query, full_body, answered_metadata)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('held', [0.0, .4])
def test_responder_uses_query_start_before_shortest_over_boundary(bw, held):
    h, io = station(bw, 'responder')
    feed(h, window(bw, b'A' * 89))
    kind = F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]
    start = min(h._unkeyed_at + A._SHORTEST_OVER / M.FS,
                h._decoded_frame_end + h._companion_samples() / M.FS) - .1
    _Clock.t = start + A._span(kind) / M.FS + held
    before = len(io.sent)
    h._answer_caller_query(kind, query_held=held, query_start=start)
    assert len(io.sent) == before + 1, io.msgs
    expected = M.synth_burst(h.called, F.for_bw(F.SESSION_INTERMEDIATE_QUERY_ANSWER, bw))
    np.testing.assert_array_equal(io.sent[-1], expected)
    assert not h._owed_block and not h._rx_failed



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('delay', [.2, 6.4])
def test_peer_nak_reanswer_uses_its_own_start(bw, streamed, delay):
    h, io = station(bw, 'initiator')
    if streamed:
        feed(h, window(bw, b'A' * 89))
    else:
        data = window(bw, b'A' * 89)
        _Clock.t += len(data) / M.FS
        h.on_rx_audio(data)
    before = len(io.sent)
    _Clock.t = h._unkeyed_at + delay
    audio = np.r_[M.synth_tone_pairs(F.nak(h.called, bw)[1]), np.zeros(M.FS // 4)]
    if streamed:
        _feed(h, audio)
    else:
        _Clock.t += len(audio) / M.FS
        h.on_rx_audio(audio)
    if delay == .2:
        assert len(io.sent) == before + 1, io.msgs
        expected = M.synth_burst(h.called, F.for_bw(F.SESSION_OVER_RESPONSE, bw))
        np.testing.assert_array_equal(io.sent[-1], expected)
    else:
        assert_missing_nak(h, io, before)
    assert io.host == [b'A' * 89]



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('side', [-.1, .1], ids=['before', 'after'])
@pytest.mark.parametrize('trail,lag', [(.05, 0.0), (.1, .3), (.4, .3)])
def test_responder_two_tone_poll_preserves_fitted_onset(monkeypatch, bw, side, trail, lag):
    h, io = station(bw, 'responder')
    feed(h, window(bw, b'A' * 89))
    poll = M.synth_tone_pairs(F.over_poll(h.called, bw)[0])
    audio = np.r_[np.zeros(int(.2 * M.FS)), poll, np.zeros(int(trail * M.FS))]
    start = min(h._unkeyed_at + A._SHORTEST_OVER / M.FS,
                h._decoded_frame_end + h._companion_samples() / M.FS) + side
    _Clock.t = start + len(poll) / M.FS + trail + lag
    h._callback_lag = round(lag * M.FS)
    h._heard += len(audio)
    starts = []
    consume = h._answer_unheard

    def query_bound(query_start=None):
        starts.append(query_start)
        return consume(query_start=query_start)

    monkeypatch.setattr(h, '_answer_unheard', query_bound)
    before = len(io.sent)
    accepted = h._stream_responder_control(audio)
    if trail + lag > A._GRANT_FRESH_S:
        assert not accepted and len(io.sent) == before
        assert h._rx_window.W is None
        assert b''.join(io.host) == b'A' * 89
        return
    assert accepted
    assert len(starts) == 1 and abs(starts[0] - start) < .02
    # Frame 12 has phase 1; the first accepted DATA answer has phase 0.
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._owed_block
    assert b''.join(io.host) == b'A' * 89



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('release', ['pair', 'ceiling'])
@pytest.mark.parametrize('first', [3.5, 6.4], ids=['early', 'late'])
def test_held_answer_needs_first_idle_before_second_block(bw, release, first):
    h, io = station(bw, 'initiator')
    a = b'A' * 89
    h._rx_read = 4, None
    h._deliver([full_body(a, h.caller)], hold=True)
    assert h._rx_window.state == 'OPEN' and not io.host
    idle = np.r_[M.synth_burst(h.called, F.for_bw(F.SESSION_RESPONDER_IDLE, bw)),
                 np.zeros(M.FS // 10)]
    _Clock.t += first + A._span(F.for_bw(F.SESSION_RESPONDER_IDLE, bw)) / M.FS
    assert h._peer_responder_idle(idle) and h._idle_complete_exact
    assert h._rx_window.W.expected == frozenset((1,))
    actions = h._rx_window.event('E', single=True, peer_keying=True)
    assert not actions and h._rx_window.state == 'ENDED-COMPLETE'
    assert not io.host
    h._perform_rx_actions(h._rx_window.event('C' if release == 'ceiling' else 'E', single=True))
    assert b''.join(io.host) == a and len(io.sent) == 1
    if first == 6.4:
        before = len(io.sent)
        query(h, delay=6.4)
        assert_missing_nak(h, io, before)
        assert h._rx_window.T is not None
        feed(h, window(bw, a, a))
        assert b''.join(io.host) == a + a



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('release', ['pair', 'ceiling'])
def test_permissive_idle_inside_unread_block_cannot_prove_early_onset(monkeypatch, bw, release):
    h, io = station(bw, 'initiator')
    a = b'A' * 89
    h._rx_read = 4, None
    h._deliver([full_body(a, h.caller)], hold=True)
    h._held_samples = round(3.5 * M.FS)
    with monkeypatch.context() as patch:
        patch.setattr(h, '_peer_frame', lambda *args: (None, 0, True))
        assert h._peer_responder_idle(np.zeros(A._span(F.for_bw(F.SESSION_RESPONDER_IDLE, bw))))
    assert not h._idle_complete_exact and not h._a_gap()
    assert h._rx_window.state == 'OPEN' and not io.host
    if release == 'pair':
        idle = np.r_[M.synth_burst(h.called, F.for_bw(F.SESSION_RESPONDER_IDLE, bw)),
                     np.zeros(M.FS // 10)]
        _Clock.t += 8
        assert h._peer_responder_idle(idle) and h._idle_complete_exact
        h._perform_rx_actions(h._rx_window.event('E', single=True, peer_keying=True))
        assert h._rx_window.state == 'ENDED-COMPLETE' and not io.host
        h._perform_rx_actions(h._rx_window.event('E', single=True))
        assert b''.join(io.host) == a
    else:
        h._perform_rx_actions(h._rx_window.event('C', peer_keying=True))
        assert b''.join(io.host) == b'' and h._rx_window.W is None
    before = len(io.sent)
    query(h, delay=6.4)
    assert_missing_nak(h, io, before)
    feed(h, window(bw, a))
    assert b''.join(io.host) == a



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_generated32_cannot_hide_companion_before_our_unkey(monkeypatch, bw, role):
    h, io = station(bw, role)
    original = h._tx_continue_answer
    monkeypatch.setattr(h, '_tx_continue_answer',
                        lambda answer, step=None: original(A.OVER_CONTINUE_GENERATED, step))
    if bw == '500':
        h._expect_level = h._next_500 = 2
        data = burst(2, b'A' * rx500._PAYLOAD[2])
    else:
        data = window(bw, b'A' * 89)
    feed(h, data)
    end = getattr(h, '_decoded_frame_end', io.at[0] - (.180 if bw == '500' else .2))
    companion = (slot(2) if bw == '500' else
                 rx.RECORDS[phy.level_key(bw, 4)].ncols
                 * rx.RECORDS[phy.level_key(bw, 4)].dw50) / M.FS
    assert end is not None and end < h._unkeyed_at
    if bw == '500' and hasattr(h, '_decoded_frame_end'):
        assert h._unkeyed_at - end == pytest.approx(1.546, abs=.06)
    start = end + companion + (1.5 if bw == '500' else .5)
    _Clock.t = start + 1.5
    next_span = ((rx500.INDEX_RECORDS[2].lead + rx500.INDEX_RECORDS[2].ncols)
                 * rx500.INDEX_RECORDS[2].dw if bw == '500' else A._SHORTEST_OVER) / M.FS
    assert start - h._unkeyed_at < next_span
    assert h._answer_unheard(query_start=start)
    saved = h._rx_window.A
    before = len(io.sent)
    h._idle_kind = F.SESSION_RESPONDER_IDLE
    h._idle_end = start + A._span(h._idle_kind) / M.FS
    _Clock.t = h._idle_end
    if role == 'initiator':
        assert h._reack_over()
    else:
        h._reanswer_over(query_start=start)
    assert len(io.sent) == before + 1 and h._rx_window.A.kind == 'continue'
    assert h._rx_window.A == saved
    np.testing.assert_array_equal(io.sent[-1], io.sent[-2])
    assert b''.join(io.host) == (b'A' * (22 if bw == '500' else 89))



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_companion_and_next_over_boundaries_are_independent(bw, role):
    h, io = station(bw, role)
    h._rx_read = (2 if bw == '500' else 4), None
    h._decoded_frame_end = _Clock.t
    h._unkeyed_at = _Clock.t + 1.546
    answered_metadata(h)
    next_span = ((rx500.INDEX_RECORDS[h._next_500].lead
                  + rx500.INDEX_RECORDS[h._next_500].ncols)
                 * rx500.INDEX_RECORDS[h._next_500].dw if bw == '500' and h._next_500 < 4
                 else (rx500._PREAMBLE + rx500.NSYM) * rx500.H if bw == '500'
                 else A._SHORTEST_OVER) / M.FS
    assert h._answer_unheard(query_start=h._unkeyed_at + next_span - .001)
    assert not h._answer_unheard(query_start=h._unkeyed_at + next_span + .001)
    for verdict, hole, complete in [('NONE', False, True), ('CERTAIN', False, False),
                                    ('POSSIBLE', False, True), ('NONE', True, False)]:
        machine = RW.ReceiveWindow()
        body = full_body(b'A' * 22, h.caller, 23)
        machine.event('F', frame=RW.Frame(body), verdict=verdict, envelope='single')
        if hole:
            machine.event('X')
        machine.event('E', single=True, peer_keying=True)
        assert (machine.state == 'ENDED-COMPLETE') is complete
    assert b''.join(io.host) == b''



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_unlocated_bracket_head_waits_for_ordered_recovery(monkeypatch, bw, role):
    h, io = station(bw, role)
    a, b = bytes(range(89)), bytes(range(89, 178))
    record = phy.level_key(bw, 4)
    r = rx.RECORDS[record]
    span = r.ncols * r.dw50
    first = window(bw, a)
    erased = np.random.default_rng(64).normal(0, np.std(first), span)
    original = h._index_alignment

    def locate(x):
        if len(x) <= span + M.FS // 2:
            return record, 0, 24, None
        return original(x)

    monkeypatch.setattr(h, '_index_alignment', locate)
    assert h._answer_data_over(np.r_[erased, window(bw, b), np.zeros(M.FS // 4)])
    assert io.host == [], io.msgs
    assert_bracket_recovery_nak(h, io, 0)
    monkeypatch.setattr(h, '_index_alignment', original)
    feed(h, window(bw, a, b))
    assert b''.join(io.host) == a + b, io.msgs



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('erased_retries', [1, 2])
def test_silent_erased_predicted_bracket_companion_draws_nak(bw, role, erased_retries):
    h, io = station(bw, role)
    singles = [bytes([i]) * 89 for i in range(1, 7)]
    for payload in singles:
        data = window(bw, payload)
        _Clock.t += (len(data) + M.FS // 2) / M.FS
        assert h._answer_data_over(np.r_[data, np.zeros(M.FS // 2)])
    assert h._peer_doubles(), io.msgs
    a, b = bytes(range(89)), bytes(range(89, 178))
    data = window(bw, a, b)
    data[len(window(bw, a)):] = 0
    for attempt in range(erased_retries):
        before = len(io.sent)
        _Clock.t += len(data) / M.FS
        assert h._answer_data_over(data)
        assert io.host == singles + ([a] if attempt else [])
        if attempt:
            assert len(io.sent) == before + 1 and h._rx_window.A.kind == 'continue'
        else:
            assert_missing_nak(h, io, before)
    data = window(bw, a, b)
    _Clock.t += len(data) / M.FS
    assert h._answer_data_over(data)
    assert b''.join(io.host) == b''.join(singles) + a * erased_retries + b, io.msgs



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('query_form', ['idle', 'poll'])
def test_seventh_unplaced_frame_requests_recovery_on_early_query(monkeypatch, bw, role, query_form):
    h, io = station(bw, role)
    for i in range(1, 7):
        feed(h, window(bw, bytes([i]) * 89))
    assert h._peer_doubles()
    monkeypatch.setattr(A, '_window_state', lambda mag, r: A._QUIET)
    before = len(io.sent)
    feed(h, window(bw, b'A' * 89))
    assert h._held_answer is not None and len(io.sent) == before
    start = h._decoded_frame_end + 3.5
    _feed(h, np.zeros(max(0, round((start - _Clock.t) * M.FS))))
    kind = F.SESSION_RESPONDER_IDLE if role == 'initiator' else F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]
    audio = (M.synth_burst(h.called, F.for_bw(kind, bw)) if query_form == 'idle' else
             M.synth_tone_pairs(F.answer_query(h.called, bw)) if role == 'initiator' else
             M.synth_tone_pairs(F.over_poll(h.called, bw)[0]))
    feed(h, audio)
    assert_missing_nak(h, io, before)
    assert h._owed_block and h._held_answer is None and h._rx_window.W is None
    assert io.host == [bytes([i]) * 89 for i in range(1, 7)]



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_six_singles_below_peer_doubling_level_do_not_predict_double(bw, level):
    h, io = station(bw, 'initiator')
    size = phy.body_size(bw, phy.level_key(bw, level)) - 1
    for i in range(1, 7):
        feed(h, window(bw, bytes([i]) * size, level=level))
    prior = b''.join(bytes([i]) * size for i in range(1, 7))
    assert h._peer_doubles() and b''.join(io.host) == prior
    seventh = window(bw, bytes([7]) * size, level=level)
    feed(h, seventh)
    assert b''.join(io.host) == prior
    feed(h, np.zeros(A._ANSWER_HOLD_MAX))
    if not h._rx_window.A.sent:
        query(h, delay=12.)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert h._rx_window.T is None
    feed(h, seventh)
    assert b''.join(io.host) == b''.join(bytes([i]) * size for i in range(1, 8))



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_decoded_frame_end_excludes_bracket_tail_and_our_answer(bw, role, route):
    h, io = station(bw, role)
    data = burst(2, b'A' * rx500._PAYLOAD[2]) if bw == '500' else window(bw, b'A' * 89)
    h._expect_level = h._next_500 = 2 if bw == '500' else 4
    start = _Clock.t
    if route == 'stream':
        feed(h, data)
    else:
        padded = np.r_[data, np.zeros(2 * M.FS)]
        _Clock.t += len(padded) / M.FS
        assert h._answer_data_over(padded)
    assert h._decoded_frame_end == pytest.approx(start + len(data) / M.FS, abs=.08)
    assert h._decoded_frame_end < io.at[0] < h._unkeyed_at



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('bracket', [False, True], ids=['stream', 'bracket'])
def test_top_level_single_header_under_double_expectation_is_incomplete(bw, role, bracket):
    from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd

    h, io = station(bw, role)
    h._link_high_speed = True
    top = phy.TOP[bw]
    h._expect_level = top
    size = phy.body_size(bw, phy.level_key(bw, top))
    for i in range(1, 7):
        body = full_body(bytes([i]) * (size - 1), h.caller, body_len=size, tail=0x85)
        h._level_read = phy.LevelRead(top, (body,), None, 1, end=0)
        assert h._answer_data_over(np.zeros(0), hold=False)
        h._level_read = None
        assert len(io.sent) == i and h._rx_window.A.step == 0
    assert h._peer_doubles()
    body = full_body(b'A' * (size - 1), h.caller, body_len=size)
    data = np.r_[np.zeros(M.FS // 2), phy.render_level((body,), bw, top, VB6Rnd(43)),
                 np.zeros(M.FS // 2)]
    read = phy.decode_level(data, bw, top)
    assert read is not None and read.located == len(read.frames) == 1
    assert read.end is not None
    if bracket:
        data = np.r_[data, np.zeros(phy.burst_samples(bw, top, 2))]
    before = len(io.sent)
    _Clock.t += len(data) / M.FS
    h._level_read = read
    try:
        assert h._answer_data_over(data, hold=not bracket)
    finally:
        h._level_read = None
    assert h._decoded_frame_end == pytest.approx(_Clock.t - (len(data) - read.end) / M.FS
                                               - sum(len(x) / M.FS for x in io.sent[before:]))
    if bracket:
        assert_missing_nak(h, io, before)
    else:
        assert len(io.sent) == before and h._held_answer is not None



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_f5f_single_l2_lost_ack_idle_at_2_9_seconds_stays_positive(bw, role):
    from .receive_window_helpers import query

    h, io = station(bw, role)
    feed(h, window(bw, b'A' * 22, level=2))
    before, answer = len(io.sent), io.sent[-1]
    query(h, delay=2.9)
    assert not h._owed_block and len(io.sent) == before + 1, io.msgs
    if role == 'responder':
        answer = M.synth_burst(h.called, F.for_bw(F.SESSION_INTERMEDIATE_QUERY_ANSWER, bw))
    np.testing.assert_array_equal(io.sent[-1], answer)
    assert io.host == [b'A' * 22]



@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('delay', [1.5, 2.6])
def test_k9kdj_low_record_lost_answer_queries_fit_both_bounds(level, delay):
    from .receive_window_helpers import query

    h, io = station('2300', 'initiator')
    h._link_high_speed = True
    size = phy.body_size('2300', phy.level_key('2300', level)) - 1
    feed(h, window('2300', b'A' * size, level=level))
    before = len(io.sent)
    query(h, delay=delay)
    assert len(io.sent) == before + 1 and not h._owed_block, io.msgs
    assert io.host == [b'A' * size]



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_alignment_and_decode_cost_do_not_move_frame_end(monkeypatch, bw, route):
    h, _ = station(bw, 'initiator')
    data = window(bw, b'A' * 89)
    start = _Clock.t
    original = h._index_alignment

    def costly(x):
        result = original(x)
        _Clock.t += .4
        return result

    monkeypatch.setattr(h, '_index_alignment', costly)
    padded = np.r_[data, np.zeros(M.FS // 2)]
    _Clock.t += len(padded) / M.FS
    h._rx_audio_at = _Clock.t
    if route == 'stream':
        h._stream_over(padded)
    else:
        assert h._answer_data_over(padded)
    assert h._decoded_frame_end == pytest.approx(start + len(data) / M.FS, abs=.08)



@pytest.mark.parametrize('level', [2, 4])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_bw500_pair_retains_the_second_decoded_frame_end(level, role, route):
    h, io = station('500', role)
    h._next_500 = h._expect_level = level
    h._peer_over_state, h._peer_run_500 = (level, 0x99), 6
    size = rx500._PAYLOAD[level]
    data = burst(level, b'A' * size, b'B' * size)
    start = _Clock.t
    if route == 'stream':
        feed(h, data)
    else:
        _Clock.t += len(data) / M.FS + .5
        assert h._answer_data_over(np.r_[data, np.zeros(M.FS // 2)])
    assert b''.join(io.host) == b'A' * size + b'B' * size
    assert h._decoded_frame_end == pytest.approx(start + len(data) / M.FS, abs=.08)



def reanswer(h, start):
    h._idle_kind = (F.SESSION_RESPONDER_IDLE if h.role == 'initiator' else
                    F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0])
    h._idle_end = start + A._span(h._idle_kind) / M.FS
    _Clock.t = h._idle_end
    if h.role == 'responder':
        h._reanswer_over(query_start=start)
    elif h._link_high_speed:
        assert h._reack_answer(query_start=start)
    else:
        assert h._reack_over(query_start=start)



def answered(bw, path):
    h, io = station(bw, 'responder' if path == 'responder' else 'initiator')
    h._link_high_speed = path == 'high-speed'
    h._expect_level = h._next_500 = 2
    data = burst(2, b'A' * rx500._PAYLOAD[2]) if bw == '500' else window(bw, b'A' * 22, level=2)
    feed(h, data)
    return h, io



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('path', ['limited', 'high-speed', 'responder'])
def test_early_first_query_proves_companion_for_later_reanswer(bw, path):
    h, io = answered(bw, path)
    frame_end = h._decoded_frame_end
    reanswer(h, frame_end + 2.88)
    assert not h._owed_block
    start = h._unkeyed_at + 2.449
    assert start - frame_end > 3
    assert start > frame_end + h._companion_samples() / M.FS
    before = len(io.sent)
    reanswer(h, start)
    assert len(io.sent) == before + 1 and not h._owed_block and h._rx_resend is None, io.msgs
    assert h._rx_window.A.kind == 'continue' and io.host == [b'A' * 22]
    before = len(io.sent)
    reanswer(h, h._unkeyed_at + 6.35)
    assert_missing_nak(h, io, before)
    assert h._rx_window.W is None



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('path', ['limited', 'high-speed', 'responder'])
def test_late_first_query_never_proves_companion_on_later_queries(monkeypatch, bw, path):
    h, io = answered(bw, path)
    original = h._tx_continue_answer
    monkeypatch.setattr(h, '_tx_continue_answer',
                        lambda answer, step=None: original(A.OVER_CONTINUE_GENERATED, step))
    h._tx_continue_answer(A.OVER_CONTINUE_GENERATED)
    start = h._decoded_frame_end + h._companion_samples() / M.FS + .01
    assert start - h._unkeyed_at < (slot(2) if bw == '500' else A._SHORTEST_OVER) / M.FS
    for at in (start, None, None):
        before = len(io.sent)
        reanswer(h, h._unkeyed_at + .2 if at is None else at)
        assert len(io.sent) == before + 1 and not h._owed_block
        assert h._rx_window.A.kind == 'continue'
    assert io.host == [b'A' * 22]



@pytest.mark.parametrize('bw,level', [(bw, lv) for bw in ('500', '2300', '2750')
                                    for lv in (max(l for b, l in A._TONE.ROWS if b == int(bw)),
                                               min(l for b, l in A._OFDM.ROWS if b == int(bw)),
                                               phy.TOP[bw])])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('late', [False, True])
def test_only_fixed_single_frame_levels_need_only_next_over_bound(bw, level, role, late):
    h, io = station(bw, role)
    h._link_high_speed = True
    h._expect_level = level
    h._rx_read = level, None
    h._decoded_frame_end = h._unkeyed_at - 10
    h._answer_owed = A._OWED_OVER
    h._reack_frame = A.OVER_CONTINUE_BURST
    h._over_answer = ('continue', A.OVER_CONTINUE_BURST)
    span = phy.burst_samples(bw, level) / M.FS
    answered_metadata(h)
    before = len(io.sent)
    reanswer(h, h._unkeyed_at + span + (.001 if late else -.001))
    if late:
        assert_missing_nak(h, io, before)
    else:
        assert len(io.sent) == before + 1 and not h._owed_block, io.msgs
        assert h._rx_resend is None



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('event', ['data', 'frame', 'nak', 'unread', 'reset'])
def test_companion_proof_clears_on_data_nak_and_new_window(bw, event):
    h, io = answered(bw, 'limited')
    reanswer(h, h._decoded_frame_end + 2.88)
    assert h._rx_window.A.kind == 'continue' and h._rx_window.W is None
    anchor = h._rx_window.K
    if event == 'data':
        data = burst(2, b'B' * 22) if bw == '500' else window(bw, b'B' * 22, level=2)
        feed(h, data)
        assert h._rx_window.K != anchor
    elif event == 'frame':
        h._note_frame_end(M.FS, M.FS)
        assert h._rx_window.K == anchor
    elif event == 'nak':
        h._nak_owed_over()
        assert h._rx_window.A.kind == 'nak'
    elif event == 'unread':
        h._irs_nak(phy.LevelRead(2, (), None, 1))
        assert h._rx_window.A.kind == 'nak'
    else:
        h.originate(h.called)
        assert h._rx_window.W is None and h._rx_window.T is None
    assert b''.join(io.host) == b'A' * 22 + (b'B' * 22 if event == 'data' else b'')



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('proof', [None, False, True])
def test_companion_predicate_remains_pure_with_latched_proof(bw, proof):
    from copy import deepcopy
    h, io = answered(bw, 'limited')
    h._unkeyed_at = h._decoded_frame_end + 10
    _Clock.t = h._unkeyed_at + .2
    before = deepcopy(h._rx_window.__dict__)
    assert h._answer_unheard(query_start=_Clock.t)
    assert not h._answer_unheard(query_start=h._unkeyed_at + 6.35)
    assert h._rx_window.__dict__ == before
    assert b''.join(io.host) == b'A' * 22



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_f5f_l2_query_at_6_35_seconds_stays_nak(bw, role):
    from .receive_window_helpers import query

    h, io = station(bw, role)
    feed(h, window(bw, b'A' * 22, level=2))
    before = len(io.sent)
    query(h, delay=6.35)
    assert_missing_nak(h, io, before)
    assert h._owed_block and io.host == [b'A' * 22]

    feed(h, window(bw, b'A' * 22, b'B' * 22, level=2))
    assert b''.join(io.host) == b'A' * 22 + b'B' * 22
    assert not h._owed_block


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_erased_top_qam_companion_needs_frame_relative_proof(bw, role):
    h, io = station(bw, role)
    h._link_high_speed = True
    h._expect_level = phy.TOP[bw]
    size = phy.body_size(bw, phy.level_key(bw, phy.TOP[bw])) - 1
    data = top_window(h, b'A' * size, b'B' * size)
    end = M.FS // 2 + phy.burst_samples(bw, phy.TOP[bw])
    data[end:] = np.random.default_rng(843).normal(0, np.std(data) * .05, len(data) - end)
    read = phy.decode_level(np.r_[data, np.zeros(M.FS // 2)], bw, phy.TOP[bw])
    assert read is not None and read.located == len(read.frames) == 1 and not read.lost
    _Clock.t += (end + M.FS // 10) / M.FS
    h._level_read = read
    try:
        assert h._answer_data_over(data[:end + M.FS // 10], hold=False)
    finally:
        h._level_read = None
    assert io.host == [b'A' * size]
    start = h._decoded_frame_end + h._companion_samples() / M.FS + .1
    assert start - h._unkeyed_at < phy.burst_samples(bw, phy.TOP[bw]) / M.FS
    before = len(io.sent)
    (query500 if bw == '500' else query)(h, delay=start - h._unkeyed_at)
    assert len(io.sent) == before + 1 and h._rx_window.A.kind == 'continue'



@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_false_idle_after_quiet_ack_cannot_latch_companion(monkeypatch, bw):
    h, io = station(bw, 'initiator')
    monkeypatch.setattr(A, '_window_state', lambda *args: A._QUIET)
    feed(h, window(bw, b'A' * 89))
    end = h._decoded_frame_end
    assert h._held_answer is None and len(io.sent) == 1
    false_start = end + 1.4
    _Clock.t = false_start + A._span(F.SESSION_RESPONDER_IDLE) / M.FS
    with monkeypatch.context() as patch:
        patch.setattr(h, '_peer_frame', lambda *args: (None, 0, True))
        assert h._peer_responder_idle(np.zeros(A._span(F.SESSION_RESPONDER_IDLE)))
    assert not h._idle_complete_exact
    before = len(io.sent)
    assert not h._a_gap()
    assert len(io.sent) == before
    _Clock.t = h._unkeyed_at + A._SHORTEST_OVER / M.FS + 1
    h._rx_query(event='R', recognized=False, query_start=_Clock.t)
    assert_missing_nak(h, io, before)
    assert h._rx_window.W is None
    before = len(io.sent)
    query(h, delay=max(.2, end + 5.71 - h._unkeyed_at))
    assert_missing_nak(h, io, before)
    assert h._rx_window.W is None and io.host == [b'A' * 89]



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('damage', ['tail', 'stale', 'payload'])
def test_query_proof_requires_a_complete_fresh_payload(bw, role, damage):
    h, io = station(bw, role)
    if role == 'initiator':
        pairs = F.answer_query(h.called, bw)
        audio = M.synth_tone_pairs(pairs)
    else:
        kind = F.for_bw(F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0], bw)
        audio = M.synth_burst(h.called, kind)
    if damage == 'tail':
        audio = audio[:-M.HOP // 2]
    elif damage == 'stale':
        audio = np.r_[audio, np.zeros(M.FS)]
    else:
        audio[-M.HOP:] = 0
    h._answer_owed = A._OWED_OVER
    h._over_answer = ('continue', A.OVER_CONTINUE_BURST)
    h._decoded_frame_end = _Clock.t - .2
    _Clock.t += len(audio) / M.FS
    if role == 'initiator':
        assert h._peer_answer_query_phase(audio) is None
    else:
        h._stream_responder_control(audio)
    assert not io.sent and h._rx_window.W is None



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('route', ['idle', 'named', 'poll'])
@pytest.mark.parametrize('side', [-.1, .1], ids=['before', 'after'])
@pytest.mark.parametrize('processing', [0, .2, .6])
def test_query_audio_onset_survives_processing_delay(monkeypatch, bw, route, side, processing):
    h, io = station(bw, 'initiator' if route == 'idle' else 'responder')
    feed(h, window(bw, b'A' * 89))
    start = min(h._unkeyed_at + A._SHORTEST_OVER / M.FS,
                h._decoded_frame_end + h._companion_samples() / M.FS) + side
    if route == 'poll':
        signal = M.synth_tone_pairs(F.over_poll(h.called, bw)[0])
        method = '_peer_pairs_at'
    else:
        kind = F.SESSION_RESPONDER_IDLE if route == 'idle' else F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]
        signal = M.synth_burst(h.called, F.for_bw(kind, bw))
        method = '_complete_query_fit' if route == 'idle' else '_peer_answer_query_phase'
    audio = np.r_[np.zeros(M.FS // 5), signal, np.zeros(M.FS // 20)]
    _Clock.t = h._rx_audio_at = start + (len(signal) + M.FS // 20) / M.FS
    original = getattr(h, method)
    delayed = []

    def recognize(*args, **kwargs):
        if not delayed:
            _Clock.t += processing
            delayed.append(True)
        result = original(*args, **kwargs)
        return result

    monkeypatch.setattr(h, method, recognize)
    queries = []
    consume = h._rx_query

    def query_event(**kwargs):
        queries.append(kwargs)
        return consume(**kwargs)

    monkeypatch.setattr(h, '_rx_query', query_event)
    h._heard += len(audio)
    before = len(io.sent)
    if route == 'idle':
        assert h._peer_responder_idle(audio)
        if h._idle_complete_exact:
            assert abs(h._idle_end - A._span(h._idle_kind) / M.FS - start) < .02
            assert h._a_gap()
            h._reack_over()
    else:
        h._stream_responder_control(audio)
    assert delayed
    if processing > A._GRANT_FRESH_S:
        assert len(io.sent) == before and h._rx_window.W is None and not queries
    else:
        assert len(queries) == 1
        if route == 'idle':
            assert queries[0].get('phase') is None and not queries[0]['explicit']
        else:
            assert abs(queries[0]['query_start'] - start) < .02
            assert queries[0]['phase'] == (1 if route == 'poll' else 0)
        if route == 'poll':
            assert_missing_nak(h, io, before)
            assert h._rx_window.A.kind == 'nak' and h._owed_block
        else:
            assert len(io.sent) == before + 1 and not h._owed_block, io.msgs
            assert h._rx_window.A.kind == 'continue'
        if route == 'named':
            expected = M.synth_burst(h.called, F.for_bw(F.SESSION_INTERMEDIATE_QUERY_ANSWER, bw))
            np.testing.assert_array_equal(io.sent[-1], expected)
    assert b''.join(io.host) == b'A' * 89


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('side', [-.1, .1], ids=['before', 'after'])
@pytest.mark.parametrize('processing', [0, .2, .6])
def test_phaseless_idle_query_onset_survives_processing_delay(monkeypatch, bw, side, processing):
    test_query_audio_onset_survives_processing_delay(monkeypatch, bw, 'idle', side, processing)



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('route', ['phase', 'nak'])
@pytest.mark.parametrize('side', [-.1, .1], ids=['before', 'after'])
@pytest.mark.parametrize('processing', [0, .2, .6])
def test_initiator_pair_dispatch_keeps_audio_onset_and_processing_age(monkeypatch, bw, route, side, processing):
    h, io = station(bw, 'initiator')
    size = 43 if bw == '500' else 89
    payload = b'A' * size
    if bw == '500':
        from .receive_window_helpers import burst
        feed(h, burst(4, payload))
    else:
        feed(h, window(bw, payload))
    start = min(h._unkeyed_at + (h._companion_samples() / M.FS if bw == '500'
                else A._SHORTEST_OVER / M.FS),
                h._decoded_frame_end + h._companion_samples() / M.FS) + side
    if route == 'phase':
        pairs = F.over_continue_frames(h.called, bw, (11,))[11][1]
    else:
        pairs = F.nak(h.called, bw)[1]
    signal = M.synth_tone_pairs(pairs)
    trailing = round((.14 if bw == '500' and route == 'phase' else .05) * M.FS)
    audio = np.r_[np.zeros(M.FS // 5), signal, np.zeros(trailing)]
    _Clock.t = h._rx_audio_at = start + (len(signal) + trailing) / M.FS
    h._heard += len(audio)
    h._ans_buf = np.zeros(0)
    h._ans_due = h._ans_skip = 0
    h._control_schedule = None
    h._callback_lag = 0
    original = h._peer_pairs_at
    delayed = []
    starts = []
    consumer = h._answer_unheard

    def recognize(*args, **kwargs):
        at = original(*args, **kwargs)
        if at is not None and not delayed:
            _Clock.t += processing
            delayed.append(True)
        return at

    def consume(query_start=None):
        if query_start is not None:
            if bw == '500' and route == 'phase':
                assert h._answer_held_from is None
            starts.append(query_start)
        return consumer(query_start=query_start)

    monkeypatch.setattr(h, '_peer_pairs_at', recognize)
    monkeypatch.setattr(h, '_answer_unheard', consume)
    before = len(io.sent)
    h._stream_answer(audio)
    assert delayed
    assert io.host == [payload]
    if processing > A._GRANT_FRESH_S:
        assert len(io.sent) == before and not starts and h._rx_window.W is None
    else:
        assert len(starts) == 1 and abs(starts[0] - start) < .02
        assert len(io.sent) == before + 1 and not h._owed_block, io.msgs
        expected = M.synth_burst(h.called, F.for_bw(F.SESSION_OVER_RESPONSE, bw))
        np.testing.assert_array_equal(io.sent[-1], expected)
        assert h._rx_window.A.kind == 'continue'



@pytest.mark.parametrize('damage', ['none', 'tail', 'stale', 'payload'])
def test_strict_query_fit_returns_only_validated_fits(damage):
    h, io = station('2300', 'initiator')
    kind = F.SESSION_RESPONDER_IDLE
    want = np.asarray(F.handshake_tones(h.called, kind))
    audio = M.synth_burst(h.called, kind)
    if damage == 'tail':
        audio = audio[:-M.HOP // 2]
    elif damage == 'stale':
        audio = np.r_[audio, np.zeros(M.FS)]
    elif damage == 'payload':
        audio[-M.HOP:] = 0
    fit = h._complete_query_fit(audio, kind, want)
    assert (fit is not None) == (damage == 'none')
    if fit is not None:
        assert isinstance(fit, int)
    assert not io.sent



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('recovering', [False, True])
def test_next_decoded_data_proves_previous_singleton_only_without_recovery(bw, role, recovering):
    h, io = station(bw, role)
    receive(h, replay_window(bw, b'A' * 22, level=2), 'stream')
    anchor = h._rx_window.K
    assert h._rx_window.A.kind == 'continue' and h._rx_window.W is None
    if recovering:
        query(h, delay=12.)
        assert h._rx_window.A.kind == 'nak' and h._rx_window.T is not None
    h._note_frame_end(0, 0)
    assert h._rx_window.K == anchor
    assert b''.join(io.host) == b'A' * 22



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('format', ['tone', 'psk'])
def test_fixed_single_frame_format_records_completion_proof(bw, role, format):
    h, io = station(bw, role)
    level = (max(l for b, l in A._TONE.ROWS if b == int(bw)) if format == 'tone'
             else min(l for (b, l), row in A._OFDM.ROWS.items() if b == int(bw) and not row.qam))
    h._rx_read = level, None
    size = phy.body_size(bw, phy.level_key(bw, level))
    payload = b'A' * (size - 1)
    h._deliver([full_body(payload, h.caller, size, 0x99)])
    assert h._rx_window.state == 'OPEN' and b''.join(io.host) == b''
    h._perform_rx_actions(h._rx_window.event('E', single=True))
    assert h._rx_window.W is None and h._rx_window.K is not None
    assert b''.join(io.host) == payload


from .receive_window_helpers import rx_evidence  # noqa: F401
