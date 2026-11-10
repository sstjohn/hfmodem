# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from copy import deepcopy

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock, _feed  # noqa: F401
from .receive_window_helpers import (station, feed, assert_missing_nak, burst, slot,
                                     top_window, top_station, query500, query500 as query, full_body, answered_metadata,
                                     commit_bodies, arm_replay, rx_evidence)  # noqa: F401


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_unpredicted_erased_partner_query_cannot_retire_the_pair(role, level, route):
    h, io = station('500', role)
    h._next_500 = h._expect_level = level
    size = rx._PAYLOAD[level]
    a, b = bytes(range(size)), bytes(range(size, 2 * size))
    double = burst(level, a, b)
    split = len(double) - slot(level)
    double[split:] = 0
    assert not h._peer_pairs_500(level)
    if route == 'stream':
        feed(h, double)
    else:
        _Clock.t += len(double) / M.FS
        assert h._answer_data_over(np.pad(double, (0, M.FS // 2)))
    assert io.host == [a] and len(io.sent) == 1, io.msgs
    assert not h._pair_loss_500 and not h._owed_block
    assert not h._pair_pending_500() and not h._unread_over_500()
    before = len(io.sent)
    query(h, delay=slot(level) / M.FS + 1.5)
    assert_missing_nak(h, io, before)
    assert io.host == [a] and h._owed_block
    before = len(io.sent)
    query(h, delay=.1)
    assert_missing_nak(h, io, before)
    retry = h._expect_level if h._rx_resend else level
    size = rx._PAYLOAD[retry]
    payload = a + b
    chunks = [payload[start:start + size] for start in range(0, len(payload), size)]
    while chunks:
        count = 2 if h._peer_pairs_500(retry) and len(chunks) > 1 else 1
        feed(h, burst(retry, *chunks[:count]))
        chunks = chunks[count:]
    assert b''.join(io.host) == payload, io.msgs
    assert not h._owed_block



@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_high_speed_lower_record_predicted_pair_still_needs_query_proof(role, level):
    h, io = station('500', role)
    h._link_high_speed = True
    h._next_500 = h._expect_level = level
    h._peer_over_state, h._peer_run_500 = (level, 0x99), 6
    h._holds_any = 6
    size = rx._PAYLOAD[level]
    a, b = bytes(range(size)), bytes(range(size, 2 * size))
    double = burst(level, a, b)
    double[len(double) - slot(level):] = 0
    assert h._peer_double_verdict() == 'POSSIBLE'
    feed(h, double)
    assert io.host == [a] and len(io.sent) == 1, io.msgs
    assert not h._pair_loss_500 and not h._owed_block
    before = len(io.sent)
    query(h, delay=10)
    assert h._rx_window.closed and h.state == A.VaraState.DISCONNECTED
    assert len(io.sent) == before and h._rx_window.T is None
    report = h._receive_reports[-1]
    assert report.code == 'R8' and report.rejected == b''
    assert (report.start, report.frontier) == (0, size)
    assert b''.join(io.host) == a



@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('high_speed', [False, True])
def test_unlocated_next_over_then_poll_requests_data(role, level, high_speed):
    h, io = station('500', role)
    h._next_500 = h._expect_level = level
    feed(h, burst(level, bytes([31]) * rx._PAYLOAD[level]))
    h._link_high_speed = high_speed
    before = len(io.sent)
    query(h, delay=len(burst(h._next_500, b'B')) / M.FS + .3)
    assert_missing_nak(h, io, before)
    assert len(io.host) == 1



@pytest.mark.parametrize('high_speed', [False, True])
@pytest.mark.parametrize('route', ['_reack_over', '_reack_answer', '_reanswer_over'])
def test_refused_bw500_late_nak_keeps_the_next_query_negative(high_speed, route):
    h, io = station('500', 'responder' if route == '_reanswer_over' else 'initiator')
    feed(h, burst(4, bytes(range(rx._PAYLOAD[4]))))
    h._link_high_speed = high_speed
    _feed(h, np.zeros(6 * M.FS))
    h._idle_kind, h._idle_end = F.SESSION_RESPONDER_IDLE, _Clock.t
    io.tx_went_out = lambda: False
    before = len(io.sent)
    getattr(h, route)()
    assert_missing_nak(h, io, before)
    assert h._owed_block
    io.tx_went_out = lambda: True
    before = len(io.sent)
    query(h, delay=.1)
    assert_missing_nak(h, io, before)
    assert len(io.host) == 1



@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('held', [.08, .14])
def test_bw500_record_bound_and_held_query_stay_positive(role, level, held):
    h, io = station('500', role)
    h._next_500 = h._expect_level = level
    feed(h, burst(level, bytes([31]) * rx._PAYLOAD[level]))
    before = len(io.sent)
    kind = (F.SESSION_RESPONDER_IDLE if role == 'initiator' else
            F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0])
    start = max(h._decoded_frame_end + 1.5, h._unkeyed_at + .1)
    _Clock.t = start + A._span(F.for_bw(kind, '500')) / M.FS + held
    if role == 'initiator':
        audio = np.r_[M.synth_burst(h.called, F.for_bw(kind, '500')),
                      np.zeros(round(held * M.FS))]
        assert h._peer_responder_idle(audio) and h._idle_complete_exact
        assert h._a_gap() and h._reack_over()
    else:
        audio = np.r_[M.synth_burst(h.called, F.for_bw(kind, '500')),
                      np.zeros(round(held * M.FS))]
        assert h._stream_responder_control(audio)
    assert len(io.sent) == before + 1 and not h._owed_block, io.msgs
    assert io.at[-1] - (start + A._span(F.for_bw(kind, '500')) / M.FS) == pytest.approx(held)
    if role == 'responder':
        expected = M.synth_burst(h.called, F.for_bw(F.SESSION_INTERMEDIATE_QUERY_ANSWER, '500'))
        np.testing.assert_array_equal(io.sent[-1], expected)
    else:
        np.testing.assert_array_equal(io.sent[-1], io.sent[0])



@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('debt', [None, '_owed_block', '_rx_failed', '_query_data_due'])
def test_read_whole_predicate_has_no_side_effects(bw, debt):
    h, io = station(bw, 'initiator')
    answered_metadata(h, kind='nak' if debt else 'continue', sent=debt is None)
    if debt:
        setattr(h, debt, True)
    h._decoded_frame_end = h._unkeyed_at
    early = h._unkeyed_at + .2
    late = h._unkeyed_at + 10
    before = deepcopy(h._rx_window.__dict__)
    assert h._answer_unheard(query_start=early)
    assert not h._answer_unheard(query_start=late)
    assert h._rx_window.__dict__ == before
    h._idle_kind = F.for_bw(F.SESSION_RESPONDER_IDLE, bw)
    _Clock.t = h._idle_end = early + A._span(h._idle_kind) / M.FS
    keys = len(io.sent)
    h._rx_query(query_start=early)
    assert len(io.sent) == keys + 1
    assert h._rx_window.A.kind == ('nak' if debt else 'continue')
    assert b''.join(io.host) == b''



@pytest.mark.parametrize('level', [1, 2, 3, 4, 5, 6, 13])
def test_bw500_bound_uses_the_record_slot_or_requested_high_speed_burst(level):
    h, io = station('500', 'initiator')
    h._link_high_speed = level > 4
    h._next_500 = h._expect_level = level
    h._rx_read = level, None
    h._decoded_frame_end = h._unkeyed_at
    answered_metadata(h)
    span = (phy.burst_samples('500', level) if level > 4 else h._companion_samples()) / M.FS
    _Clock.t = h._unkeyed_at + span + 1
    if level == 13:
        assert span == pytest.approx(3.816)
        assert h._answer_unheard(query_start=h._unkeyed_at + 3.815)
        assert not h._answer_unheard(query_start=h._unkeyed_at + 3.817)
    else:
        assert h._answer_unheard(query_start=h._unkeyed_at + span - .001)
        assert not h._answer_unheard(query_start=h._unkeyed_at + span + .001)
    assert b''.join(io.host) == b''



@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [2, 4])
def test_bw500_equal_payload_advanced_field_recovery_reaches_host(role, level):
    h, io = station('500', role)
    h._expect_level = h._next_500 = level
    payload = b'A' * rx._PAYLOAD[level]
    feed(h, burst(level, payload))
    before = len(io.sent)
    query500(h, delay=10)
    assert_missing_nak(h, io, before)
    frame = (tx.build_frame(payload, 0x91) if level == 4 else
             tx.build_l3_frame(payload, 0x91, level))
    feed(h, tx.synth_burst([frame], level=level, onset=24000))
    # LEDGER #row-101: fields cannot distinguish equal new bytes after a NAK.
    assert b''.join(io.host) == payload, io.msgs
    assert h._rx_window.T is None and not h._owed_block



@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_bw500_nested_speculative_replay_keeps_advanced_field(role):
    h, io = station('500', role)
    h._rx_read = 4, None
    payload = b'A' * rx._PAYLOAD[4]
    old, new = full_body(payload, h.caller, 44, 0x95), full_body(payload, h.caller, 44, 0x91)
    commit_bodies(h, [old])
    arm_replay(h, io, query=query500)
    commit_bodies(h, [new])
    assert b''.join(io.host) == payload
    assert h._rx_window.T is None



@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_bw500_automatic_nested_replay_keeps_consecutive_equal_chunks(role):
    h, io = station('500', role)
    p, r, s = b'P' * 22, b'R' * 22, b'S' * 20
    original = p + p + r + s
    h._rx_read = 4, None
    commit_bodies(h, [full_body(original[:43], h.caller, 44, 0x95),
                      full_body(original[43:], h.caller, 44, 0x95)])
    assert b''.join(io.host) == original
    arm_replay(h, io, query=query500)
    h._rx_read = 2, None
    for payload, field in ((p, 0x95), (p, 0x91), (r, 0x95), (s, 0x82)):
        commit_bodies(h, [full_body(payload, h.caller, 23, field)])
        assert b''.join(io.host) == original
    assert h._rx_window.T is None



@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('release', ['early', 'late', 'timeout', 'companion'])
def test_bw500_top_hold_lifecycle(role, release):
    h, io = top_station(role, predicted=True)
    prior, sent = b''.join(io.host), len(io.sent)
    size = phy.body_size('500', phy.level_key('500', phy.TOP['500'])) - 1
    data = top_window(h, b'A' * size, b'B' * size)
    cut = M.FS // 2 + phy.burst_samples('500', phy.TOP['500'])
    # Row 104's companion belongs to this same keying; feed adds silence.
    (_feed if release == 'companion' else feed)(h, data[:cut])
    assert h._held_answer is not None and len(io.sent) == sent and b''.join(io.host) == prior, io.msgs
    if release == 'companion':
        feed(h, data[cut:])
        assert h._held_answer is None and len(io.sent) == sent + 1, '\n'.join(io.msgs)
        assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
        assert b''.join(io.host) == prior + b'A' * size + b'B' * size, io.msgs
        return
    if release == 'timeout':
        feed(h, np.zeros(A._ANSWER_HOLD_MAX))
        assert h._held_answer is None and h._owed_block
        assert_missing_nak(h, io, sent)
    start = max(_Clock.t, h._decoded_frame_end + (.2 if release == 'early' else 10))
    _feed(h, np.zeros(max(0, round((start - _Clock.t) * M.FS))))
    if release == 'late':
        # DESIGN §2: quiet airs the incomplete CERTAIN window's NAK.
        assert_missing_nak(h, io, sent)
    before_query = len(io.sent)
    kind = F.SESSION_RESPONDER_IDLE if role == 'initiator' else F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]
    feed(h, M.synth_burst(h.called, F.for_bw(kind, '500')))
    if h._held_answer is not None:
        feed(h, np.zeros(A._ANSWER_HOLD_MAX))
        query500(h, delay=.2)
    # A later validated Q(-) repeats the already-aired NAK (§2 :241).
    assert len(io.sent) == sent + (1 if release == 'early' else 2), '\n'.join(io.msgs)
    assert_missing_nak(h, io, before_query)
    assert b''.join(io.host) == prior and h._held_answer is None
    payload = b'A' * size + b'B' * size
    level = h._expect_level
    body_size = phy.body_size('500', phy.level_key('500', level))
    for at in range(0, len(payload), body_size - 1):
        field = 0x95 if at // (body_size - 1) % 2 == 0 else 0x91
        body = full_body(payload[at:at + body_size - 1], h.caller,
                            body_len=body_size, tail=field)
        audio = np.r_[np.zeros(M.FS // 2), phy.render_level((body,), '500', level, VB6Rnd(at))]
        feed(h, audio)
    assert b''.join(io.host) == prior + payload, io.msgs
    assert not h._owed_block and h._held_answer is None



@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_bw500_late_idle_pair_cannot_release_a_positive_hold(role):
    h, io = station('500', role)
    payload = b'A' * 43
    body = full_body(payload, h.caller, 44)
    h._rx_read = 4, None
    h._decoded_frame_end = _Clock.t
    h._deliver([body], hold=True)
    assert h._rx_window.state == 'OPEN' and not io.host
    h._perform_rx_actions(h._rx_window.event('E', single=True, envelope='single'))
    assert b''.join(io.host) == payload and h._rx_window.A.sent
    assert h._unkeyed_at <= _Clock.t
    before = len(io.sent)
    query500(h, delay=6.)
    assert_missing_nak(h, io, before)
    assert h._rx_window.W is None and h._rx_window.A.kind == 'nak'
    assert b''.join(io.host) == payload
    before = len(io.sent)
    query500(h, delay=.2)
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak'
    assert b''.join(io.host) == payload



@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('same', [False, True])
def test_bw500_continuous_companion_survives_early_end_read(monkeypatch, role, same):
    h, io = top_station(role, predicted=True)
    prior, sent = b''.join(io.host), len(io.sent)
    size = phy.body_size('500', phy.level_key('500', phy.TOP['500'])) - 1
    a, b = b'A' * size, (b'A' if same else b'B') * size
    data = top_window(h, a, b)
    answered = []
    answer = h._answer_data_over

    def whole_over(samples, hold=False):
        answered.append(h._level_read.frames)
        return answer(samples, hold)

    monkeypatch.setattr(h, '_answer_data_over', whole_over)
    cut = M.FS // 2 + phy.burst_samples('500', phy.TOP['500'])
    _feed(h, data[:cut])
    if h._held_answer is None:
        h._lv_due = len(h._lv_buf)
        h._lv_ended = True
        h._lv_credit = 1
        # A partial buffer registers the head, but has not taken a whole over.
        assert not h._read_expected_level(np.zeros(0))
    assert h._held_answer is not None and len(io.sent) == sent and b''.join(io.host) == prior
    head = h._rx_window.W.frames
    assert len(head) == 1 and phy.vara_payload(head[0].body, caller=h.caller,
                                             body_len=size + 1) == a
    assert answered == []
    # Rereading this buffer cannot register its head a second time.
    h._lv_due = len(h._lv_buf)
    h._lv_ended = True
    h._lv_credit = 1
    assert not h._read_expected_level(np.zeros(0))
    assert h._rx_window.W.frames == head and answered == []
    feed(h, data[cut:])
    assert h._held_answer is None and len(io.sent) == sent + 1, io.msgs
    assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
    assert len(answered) == 1 and len(answered[0]) == 2
    assert answered[0][0] == head[0].body
    assert b''.join(io.host) == prior + a + b



def test_bw500_frame_ten_repeat_requires_a_validated_query(monkeypatch):
    h, io = station('500', 'initiator')
    h._pair_loss_500 = h._owed_block = True
    h._answer_owed = A._OWED_OVER
    monkeypatch.setattr(h, '_peer_frame', lambda *args: (None, 0, True))
    assert h._peer_responder_idle(np.zeros(A._span(F.SESSION_RESPONDER_IDLE)))
    assert not h._idle_complete_exact and not h._a_gap()
    assert not h._reack_over()
    assert not io.sent and h._rx_window.W is None



@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('bw', ['500', '2750'])
def test_continuous_top_double_keyed_four_seconds_after_unkey_holds_head_until_companion_ends(bw, role):
    h, io = station(bw, role)
    h._link_high_speed = True
    h._expect_level = level = phy.TOP[bw]
    size = phy.body_size(bw, phy.level_key(bw, level)) - 1
    for i in range(1, 7):
        body = full_body(bytes([i]) * size, h.caller, size + 1, 0x85)
        h._level_read = phy.LevelRead(level, (body,), None, 1, end=0)
        try:
            assert h._answer_data_over(np.zeros(0))
        finally:
            h._level_read = None
        assert len(io.sent) == i and h._rx_window.A.step == 0
    h._lv_armed = True
    assert h._peer_double_verdict() == 'CERTAIN'
    prior, sent = b''.join(io.host), len(io.sent)
    a, b = b'A' * size, b'B' * size
    data = np.r_[np.zeros(4 * M.FS), top_window(h, a, b)[M.FS // 2:]]
    head_end = 4 * M.FS + phy.burst_samples(bw, level) + round(1.2 * M.FS)
    _feed(h, data[:head_end])
    assert h._lv_registered == 1
    assert h._held_answer is not None and len(io.sent) == sent and b''.join(io.host) == prior, '\n'.join(io.msgs)
    head = h._rx_window.W.frames
    assert len(head) == 1 and phy.vara_payload(head[0].body, caller=h.caller,
                                             body_len=size + 1) == a
    # The unkey-relative buffer is full while this same keying still carries B.
    cut = len(data) - round(.2 * M.FS)
    assert head_end < A._EXPECT_WINDOW[bw] < cut < len(data)
    _feed(h, data[head_end:cut])
    assert h._held_answer is not None and len(io.sent) == sent and b''.join(io.host) == prior, '\n'.join(io.msgs)
    assert h._rx_window.W.frames == head
    feed(h, data[cut:])
    assert h._held_answer is None and len(io.sent) == sent + 1, '\n'.join(io.msgs)
    assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
    assert b''.join(io.host) == prior + a + b, '\n'.join(io.msgs)
