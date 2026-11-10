# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Committed anchors and replay cursors remain independent of incomplete DATA."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock, _feed  # noqa: F401
from .receive_window_helpers import (
    station, window, replay_window, receive, feed, query, full_body,
    assert_missing_nak, commit_bodies, arm_replay, assert_r8, terminal_retry, rx_evidence,  # noqa: F401
)

@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('same', [False, True], ids=['distinct', 'identical'])
@pytest.mark.parametrize('failed_retry', [False, True])
@pytest.mark.parametrize('retry_level', [4, 2])
def test_incomplete_window_replay_prefix_is_consumed_once(bw, same, failed_retry, retry_level, unextended=False):
    h, io = station(bw, 'initiator')
    a = bytes(range(89))
    b = a if same else bytes(range(89, 178))
    body = full_body(a, h.caller)
    for _ in range(2 if failed_retry else 1):
        h._deliver([body], hold=True)
        assert h._rx_window.state == 'OPEN' and b''.join(io.host) == b''
        h._finish_delivery_window(complete=False)
        query(h, delay=12.)
        assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
        assert h._rx_window.T is None and b''.join(io.host) == b''
    size = phy.body_size(bw, phy.level_key(bw, retry_level)) - 1
    chunks = [(a + b)[i:i + size] for i in range(0, len(a + b), size)]
    if retry_level == 4:
        feed(h, window(bw, *chunks))
    else:
        for chunk in chunks:
            feed(h, window(bw, chunk, level=retry_level))
        terminal_retry(h, io, chunks[-1], a + b, unextended=unextended)
        if unextended:
            assert b''.join(io.host) == (a + b)[:176]
            return
    assert b''.join(io.host) == a + b, io.msgs
    assert h._rx_window.T is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('streamed', [False, True], ids=['bracket', 'stream'])
@pytest.mark.parametrize('same', [False, True], ids=['distinct', 'equal'])
def test_completed_double_late_query_replay_is_consumed_once(bw, role, streamed, same):
    h, io = station(bw, role)
    a, b = b'A' * 89, (b'A' if same else b'B') * 89
    data = window(bw, a, b)
    feed(h, data)
    assert b''.join(io.host) == a + b
    assert tuple(f.body for f in h._rx_window.K.frames) == (full_body(a, h.caller), full_body(b, h.caller))
    arm_replay(h, io)
    receive(h, data, 'stream' if streamed else 'bracket')
    assert b''.join(io.host) == a + b and h._rx_window.T is None
    feed(h, window(bw, b'C' * 89))
    assert b''.join(io.host) == a + b + b'C' * 89


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('streamed', [False, True], ids=['bracket', 'stream'])
def test_lost_equal_next_over_with_advanced_field_is_new_data(bw, role, streamed):
    h, io = station(bw, role)
    record = phy.level_key(bw, 4)
    payload = b'A' * 89
    a = full_body(payload, h.caller, field=0x95)
    b = full_body(payload, h.caller, field=0x91)
    feed(h, tx.synth_burst(a, record, 0))
    arm_replay(h, io)
    receive(h, tx.synth_burst(b, record, 1), 'stream' if streamed else 'bracket')
    # LEDGER #row-49; R1: a changed field cannot prove equal payload new.
    assert b''.join(io.host) == payload
    assert h._rx_window.T is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('resolution', ['repeat', 'different'])
def test_silent_tail_bracket_keeps_matching_replay_head_ambiguous(bw, role, resolution):
    h, io = station(bw, role)
    a, b, c = b'A' * 22, b'B' * 22, b'C' * 22
    receive(h, replay_window(bw, a, b, level=2), 'bracket')
    assert b''.join(io.host) == a + b
    arm_replay(h, io)
    head = replay_window(bw, a, level=2)
    retry = replay_window(bw, a, b, level=2)
    retry[len(head):] = 0
    assert rx.detect_overs(retry, min_len=0) == rx.detect_overs(np.r_[head, np.zeros(len(retry) - len(head))], min_len=0)
    receive(h, retry, 'bracket')
    assert b''.join(io.host) == a + b and h._rx_window.T.cursor == 22
    arm_replay(h, io)
    before = len(io.sent)
    receive(h, replay_window(bw, a, c if resolution == 'different' else b, level=2), 'bracket')
    assert b''.join(io.host) == a + b
    if resolution == 'different':
        assert_r8(h, io, before, a + b, a + c)
    else:
        assert h._rx_window.T is None and not h._rx_window.closed


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_initial_silent_tail_bracket_recovers_without_repeating_head(bw, role):
    h, io = station(bw, role)
    a, b = b'A' * 22, b'B' * 22
    head = replay_window(bw, a, level=2)
    data = replay_window(bw, a, b, level=2)
    damaged = data.copy()
    damaged[len(head):] = 0
    assert rx.detect_overs(damaged, min_len=0) == rx.detect_overs(np.r_[head, np.zeros(len(damaged) - len(head))], min_len=0)
    receive(h, damaged[:len(head)], 'bracket')
    _Clock.t += (len(damaged) - len(head)) / M.FS
    assert b''.join(io.host) == a
    arm_replay(h, io)
    receive(h, data, 'bracket')
    assert b''.join(io.host) == a + b


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['bracket', 'stream'])
@pytest.mark.parametrize('closing', [False, True], ids=['full-C', 'closing-C'])
def test_answered_ambiguous_prefix_cannot_join_next_window(bw, role, route, closing):
    h, io = station(bw, role)
    a, b, c = b'A' * 22, b'B' * 22, b'C' * (7 if closing else 22)
    data = replay_window(bw, a, b, level=2)
    receive(h, data, 'bracket')
    arm_replay(h, io)
    damaged = data.copy()
    damaged[len(replay_window(bw, a, level=2)):] = 0
    receive(h, damaged, 'bracket')
    assert h._rx_window.T.cursor == 22 and b''.join(io.host) == a + b
    before = len(io.sent)
    receive(h, replay_window(bw, c, level=2), route)
    assert_r8(h, io, before, a + b, c)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('evidence', ['early-query', 'next-DATA'])
def test_bracket_singleton_proof_requires_independent_end_evidence(bw, role, evidence):
    h, io = station(bw, role)
    a, c = b'A' * 22, b'C' * 22
    receive(h, replay_window(bw, a, level=2), 'bracket')
    assert h._rx_window.W is None and h._rx_window.A.kind == 'continue'
    if evidence == 'early-query':
        before = len(io.sent)
        query(h, delay=.2)
        assert len(io.sent) == before + 1 and not h._owed_block
    else:
        receive(h, replay_window(bw, c, level=2), 'bracket')
    assert b''.join(io.host) == a + (c if evidence == 'next-DATA' else b'')


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('answer_before', [False, True])
@pytest.mark.parametrize('declined', [False, True])
def test_ambiguous_prefix_retires_only_at_transmitted_answer(monkeypatch, bw, role, answer_before, declined):
    h, io = station(bw, role)
    h._rx_read = 2, None
    a, b, c = [full_body(p * 22, h.caller, 23, 0x99) for p in (b'A', b'B', b'C')]
    commit_bodies(h, [a, b])
    arm_replay(h, io)
    original_commit = h._rx_window.commit
    def ordered(prepared, **kwargs):
        return original_commit(prepared, **dict(kwargs, answer_before=answer_before))
    monkeypatch.setattr(h._rx_window, 'commit', ordered)
    with monkeypatch.context() as patch:
        patch.setattr(io, 'tx_went_out', lambda: not declined, raising=False)
        commit_bodies(h, [a])
        assert h._rx_window.W is None and h._rx_window.T.cursor == 22
        assert h._rx_window.A.sent is not declined
    if declined:
        query(h, delay=.2)
    before = len(io.sent)
    commit_bodies(h, [c])
    assert_r8(h, io, before, b'A' * 22 + b'B' * 22, b'C' * 22)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('partial', [False, True], ids=['accounted', 'unaccounted'])
def test_bracket_proves_end_only_with_accounted_extent(
        monkeypatch, record_property, bw, role, partial):
    h, io = station(bw, role)
    a, b = b'A' * 22, b'B' * 22
    head = replay_window(bw, a, level=2)
    data = replay_window(bw, a, b, level=2)
    end = len(head) + ((len(data) - len(head)) // 3 if partial else 0)
    audio = data[:end] if partial else np.r_[data[:end], np.zeros(M.FS // 10)]
    assert phy.burst_extent(audio, bw, 2) == ('inconclusive' if partial else 'single')

    events = []
    history = []
    event = h._rx_window.event
    perform = h._perform_rx_actions
    sent = h._rx_window.sent

    def observed(kind, **kwargs):
        before = h._rx_window.state
        actions = event(kind, **kwargs)
        events.append((kind, before, h._rx_window.state, repr(actions)))
        history.append({'event': kind, 'before': before, 'after': h._rx_window.state,
                        'inputs': kwargs})
        return actions

    def performed(actions, **kwargs):
        history.extend({'action': type(action).__name__, 'value': asdict(action)}
                       for action in actions)
        return perform(actions, **kwargs)

    def aired(key, reached_air):
        history.append({'sent': asdict(key), 'reached_air': reached_air})
        return sent(key, reached_air)

    monkeypatch.setattr(h._rx_window, 'event', observed)
    monkeypatch.setattr(h, '_perform_rx_actions', performed)
    monkeypatch.setattr(h._rx_window, 'sent', aired)
    receive(h, audio, 'bracket')
    record_property('initial_receive', json.dumps({
        'history': history, 'host_hex': b''.join(io.host).hex(),
        'R_hex': h._rx_window.R.hex(), 'A': asdict(h._rx_window.A),
        'envelope': phy.burst_extent(audio, bw, 2),
        'head_samples': len(head), 'bracket_samples': len(audio),
    }, default=lambda value: {'hex': value.hex()} if isinstance(value, bytes)
       else asdict(value)))
    if partial:
        # LEDGER #row-56; DESIGN §§2–4: a truncated active companion is X.
        assert [e[0] for e in events] == ['F', 'X', 'E'], (events, io.host, io.msgs)
        assert b''.join(io.host) == b''
        assert h._rx_window.A.kind == 'nak'
        assert h._rx_window.A.sent is not (bw == '2750' and role == 'initiator')
    else:
        # LEDGER #row-55; DESIGN §3: measured singleton quiet supplies E.
        assert [e[0] for e in events] == ['F', 'E'], events
        assert b''.join(io.host) == a
        assert h._rx_window.A.kind == 'continue'
        assert h._rx_window.A.sent
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert b''.join(io.host) == (b'' if partial else a)
    if partial:
        assert h._rx_window.T is None
    else:
        assert (h._rx_window.T.start, h._rx_window.T.frontier) == (0, 22)
    receive(h, data, 'bracket')
    assert b''.join(io.host) == a + b, io.msgs
    assert h._rx_window.T is None
    assert h._rx_window.W is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_ambiguous_new_identical_head_ack_heard_then_erased_next_window_resolves_as_replay(monkeypatch, bw, role):
    authorized = False
    h, io = station(bw, role)
    a, b, c = b'A' * 22, b'B' * 22, b'C' * 22
    receive(h, replay_window(bw, a, b, level=2), 'bracket')
    if authorized:
        arm_replay(h, io)
    with monkeypatch.context() as patch:
        patch.setattr(A, '_window_state', lambda *args: A._QUIET)
        receive(h, replay_window(bw, a, level=2), 'stream')
    assert b''.join(io.host) == a + b + (b'' if authorized else a)
    feed(h, np.zeros(len(replay_window(bw, c, level=2))))
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    before = len(io.sent)
    receive(h, replay_window(bw, c, level=2), 'bracket')
    if authorized:
        assert_r8(h, io, before, a + b, c)
    else:
        assert b''.join(io.host) == a + b + a + c


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_ambiguous_new_identical_head_ack_heard_then_erased_next_window_resolves_as_replay_authorized_resend(monkeypatch, bw, role):
    authorized = True
    h, io = station(bw, role)
    a, b, c = b'A' * 22, b'B' * 22, b'C' * 22
    receive(h, replay_window(bw, a, b, level=2), 'bracket')
    if authorized:
        arm_replay(h, io)
    with monkeypatch.context() as patch:
        patch.setattr(A, '_window_state', lambda *args: A._QUIET)
        receive(h, replay_window(bw, a, level=2), 'stream')
    assert b''.join(io.host) == a + b + (b'' if authorized else a)
    feed(h, np.zeros(len(replay_window(bw, c, level=2))))
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    before = len(io.sent)
    receive(h, replay_window(bw, c, level=2), 'bracket')
    if authorized:
        assert_r8(h, io, before, a + b, c)
    else:
        assert b''.join(io.host) == a + b + a + c


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('repeat_partial', [False, True])
def test_repacked_equal_chunk_with_advanced_field_advances_original_prefix(bw, repeat_partial):
    h, io = station(bw, 'initiator')
    a = b'A' * 89
    h._rx_read = 4, None
    commit_bodies(h, [full_body(a, h.caller, field=0x95)])
    arm_replay(h, io)
    h._rx_read = 2, None
    partial = full_body(a[:22], h.caller, 23, 0x95)
    commit_bodies(h, [partial])
    assert h._rx_window.T.cursor == 22 and b''.join(io.host) == a
    arm_replay(h, io)
    before = len(io.sent)
    next_body = partial if repeat_partial else full_body(a[22:44], h.caller, 23, 0x91)
    commit_bodies(h, [next_body])
    assert_r8(h, io, before, a, a[:22])


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('streamed', [False, True], ids=['bracket', 'stream'])
def test_nested_double_replay_consumes_one_obligation(bw, role, streamed):
    h, io = station(bw, role)
    a = bytes(range(89))
    feed(h, window(bw, a))
    arm_replay(h, io)
    route = 'stream' if streamed else 'bracket'
    partial = window(bw, a[:22], a[22:44], level=2)
    receive(h, partial, route)
    assert h._rx_window.T.cursor == 44 and b''.join(io.host) == a
    arm_replay(h, io)
    receive(h, partial, route)
    assert h._rx_window.T.cursor == 44 and b''.join(io.host) == a
    for payload in (a[44:66], a[66:88], a[88:] + b'new'):
        receive(h, window(bw, payload, level=2), route)
    assert b''.join(io.host) == a + b'new'


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('retry_level', [4, 2])
@pytest.mark.parametrize('failure', ['stream', 'expected'])
def test_crc_failure_retires_held_window_before_prompt_retry(monkeypatch, bw, role, retry_level, failure, unextended=False):

    h, io = station(bw, role)
    a, b = bytes(range(89)), bytes(range(89, 178))
    double = window(bw, a, b)
    if failure == 'stream':
        original = rx.frame_at
        damaged = full_body(b, h.caller)
        observed = []
        undecoded = h._undecoded_over

        def fail_companion(mag, level):
            fr = original(mag, level)
            return replace(fr, crc_ok=False) if bytes(fr.payload) == damaged else fr

        def failed_over():
            observed.append((h._held_answer, h._held_samples))
            return undecoded()

        with monkeypatch.context() as patch:
            patch.setattr(rx, 'frame_at', fail_companion)
            patch.setattr(h, '_undecoded_over', failed_over)
            feed(h, double)
        assert len(observed) == 1 and observed[0][0] is not None, io.msgs
        assert 0 < observed[0][1] < A._ANSWER_HOLD_MAX
    else:
        for at in range(0, len(double), M.FS // 50):
            _feed(h, double[at:at + M.FS // 50])
            if h._held_answer is not None:
                break
        assert h._held_answer is not None and b''.join(io.host) == b'', io.msgs
        h._link_high_speed = True
        h._irs_nak(phy.LevelRead(4, (), None, 1))
        h._link_high_speed = False
    assert b''.join(io.host) == b'', io.msgs
    assert h._held_samples < A._ANSWER_HOLD_MAX
    assert h._held_answer is None and h._rx_window.W is None, io.msgs
    assert h._rx_window.K is None and h._rx_window.T is None
    if not io.sent:
        query(h, delay=.2)
    if failure == 'expected' or role == 'responder':
        assert len(io.sent) == 1, io.msgs
        np.testing.assert_array_equal(io.sent[-1], M.synth_tone_pairs(F.nak(h.called, bw)[role == 'responder']))
    else:
        assert_missing_nak(h, io, 0)
    if failure == 'expected' and retry_level == 2:
        retry_level = h._expect_level
    size = phy.body_size(bw, phy.level_key(bw, retry_level)) - 1
    chunks = [(a + b)[i:i + size] for i in range(0, len(a + b), size)]
    if retry_level == 4:
        feed(h, window(bw, *chunks))
    else:
        for chunk in chunks:
            feed(h, window(bw, chunk, level=retry_level))
        if retry_level == 2:
            terminal_retry(h, io, chunks[-1], a + b, unextended=unextended)
            if unextended:
                assert b''.join(io.host) == (a + b)[:176]
                return
    assert b''.join(io.host) == a + b, io.msgs
    assert not h._held_answer and not h._owed_block


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('close', ['pair', 'ceiling', 'failed', 'ended', 'handover'])
@pytest.mark.parametrize('repeat', [False, True])
def test_window_closure_classifies_a_buffered_prefix(
        monkeypatch, record_property, bw, close, repeat):
    h, io = station(bw, 'initiator')
    a, b = b'A' * 89, b'B' * 89
    bodies = tuple(full_body(p, h.caller, 90) for p in (a, b))
    h._rx_read = 4, None
    h._expect_level = 4
    h._deliver(bodies)
    h._answer_over(False, False, False)
    assert b''.join(io.host) == a + b
    assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
    assert h._rx_window.T is None

    if close != 'failed':
        h._deliver(bodies if repeat else bodies[:1], hold=True)
        assert h._rx_window.state == 'OPEN' and b''.join(io.host) == a + b
        if close == 'handover':
            h.turn = A._TURN_OURS
            assert h._rx_window.W is None and b''.join(io.host) == a + b
            if not repeat:
                h.turn = A._TURN_PEER
                h._perform_rx_actions(h._rx_window.event('G'))
                commit_bodies(h, bodies[:1])
        else:
            kind = F.for_bw(F.SESSION_RESPONDER_IDLE, bw)
            idle = np.r_[M.synth_burst(h.called, kind), np.zeros(M.FS // 10)]
            _Clock.t += 3.5 + A._span(kind) / M.FS
            assert h._peer_responder_idle(idle) and h._idle_complete_exact
            if close == 'ended':
                h._note_frame_end(0, 0)
                assert h._rx_window.state == 'OPEN'
            h._perform_rx_actions(h._rx_window.event('E', single=True, peer_keying=True))
            assert h._rx_window.state == 'ENDED-COMPLETE' and b''.join(io.host) == a + b
            h._perform_rx_actions(h._rx_window.event('C' if close == 'ceiling' else 'E', single=True))
        expected = a + b + (b'' if close == 'handover' and repeat else a + b if repeat else a)
        assert b''.join(io.host) == expected
        return

    history = []
    event = h._rx_window.event

    class WindowStates(type(h._rx_window)):
        def __setattr__(self, name, value):
            if name == 'state':
                history.append({'state': value})
            super().__setattr__(name, value)

    def observed(kind, **kwargs):
        entry = {'event': kind, 'before': h._rx_window.state}
        history.append(entry)
        try:
            return event(kind, **kwargs)
        finally:
            entry['after'] = h._rx_window.state

    monkeypatch.setattr(h._rx_window, 'event', observed)
    monkeypatch.setattr(h._rx_window, '__class__', WindowStates)
    h._deliver(bodies if repeat else bodies[:1], hold=True)
    assert h._rx_window.state == 'OPEN'
    assert tuple(f.body for f in h._rx_window.W.frames) == (bodies if repeat else bodies[:1])
    assert b''.join(io.host) == a + b
    before = len(io.sent)
    try:
        # LEDGER #row-61Ffailed/#row-61Tfailed; DESIGN §2: failure drops W.
        h._finish_delivery_window(complete=False)
    finally:
        record_property('failed_open_window', json.dumps({
            'history': history, 'host_hex': b''.join(io.host).hex(),
            'state': h._rx_window.state,
            'W': asdict(h._rx_window.W) if h._rx_window.W else None,
            'A': asdict(h._rx_window.A), 'new_keyings': len(io.sent) - before,
        }, default=lambda value: {'hex': value.hex()} if isinstance(value, bytes)
           else sorted(value) if isinstance(value, (set, frozenset)) else asdict(value)))
    assert {'state': 'ENDED-INCOMPLETE'} in history
    assert b''.join(io.host) == a + b
    assert h._rx_window.W is None and h._rx_window.T is None
    assert h._rx_window.A.kind == 'nak'
    if not h._rx_window.A.sent:
        query(h, delay=12.)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert_missing_nak(h, io, before)
    h._deliver(bodies)
    h._answer_over(False, False, False)
    assert b''.join(io.host) == a + b + a + b


def test_completed_window_classifies_once_and_owns_its_prefix(monkeypatch):
    h, io = station('2300', 'initiator')
    a, b = [full_body(p * 89, h.caller) for p in (b'A', b'B')]
    original = h._rx_window.commit
    calls = []
    def committed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(h._rx_window, 'commit', committed)
    commit_bodies(h, [a, b])
    assert len(calls) == 1 and b''.join(io.host) == b'A' * 89 + b'B' * 89
    assert tuple(f.body for f in h._rx_window.K.frames) == (a, b)
    arm_replay(h, io)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('longer', [False, True])
def test_erased_next_window_recurring_head_uses_complete_identity(bw, role, route, longer):
    h, io = station(bw, role)
    singles = [bytes([i]) * 89 for i in range(1, 7)]
    for payload in singles:
        receive(h, replay_window(bw, payload), route)
    a, b = singles[-1], b'B' * 89
    assert h._rx_window.K.frames[-1].body[-1] == 0x99
    query(h, delay=.2)
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert (h._rx_window.T is None) is (role == 'responder')
    receive(h, replay_window(bw, a, b) if longer else replay_window(bw, a), route)
    # LEDGER #row-63/#row-65; R1: equal new bytes match K after a phaseless NAK.
    expected = b''.join(singles) + (a if role == 'responder' else b'') + (b if longer else b'')
    assert b''.join(io.host) == expected, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route,damage', [('stream', 'crc'), ('stream', 'timeout'), ('bracket', 'crc')])
@pytest.mark.parametrize('same', [False, True], ids=['new-tail', 'repeat'])
def test_incomplete_repeat_prefix_stays_ambiguous_until_recovery(monkeypatch, bw, role, route, damage, same):
    authorized = False
    h, io = station(bw, role)
    level = 2
    size = phy.body_size(bw, phy.level_key(bw, level)) - 1
    a, b = b'A' * size, b'B' * size
    receive(h, replay_window(bw, a, b, level=level), route)
    assert b''.join(io.host) == a + b
    if authorized:
        arm_replay(h, io)
    identity = h._rx_window.K
    data = replay_window(bw, a, b, level=level)
    head = replay_window(bw, a, level=level)
    tail = full_body(b, h.caller, tail=0x99, body_len=size + 1)
    original = rx.frame_at

    def bad_tail(*args):
        frame = original(*args)
        return replace(frame, crc_ok=False) if bytes(frame.payload) == tail else frame

    with monkeypatch.context() as patch:
        if damage == 'crc':
            patch.setattr(rx, 'frame_at', bad_tail)
        else:
            data[len(head):] = 0
        if route == 'stream':
            patch.setattr(A, '_window_state', lambda *args: A._KEYING)
            step = M.FS // 50
            for at in range(0, len(data), step):
                _feed(h, data[at:at + step])
                if h._held_answer is not None:
                    break
            assert h._held_answer is not None, io.msgs
            _feed(h, data[at + step:])
            feed(h, np.zeros(A._ANSWER_HOLD_MAX))
        else:
            receive(h, data, route)
    assert b''.join(io.host) == a + b, io.msgs
    assert h._rx_window.K == identity and h._rx_window.A.kind == 'nak'
    assert h._rx_window.W is None
    assert h._rx_window.W is None
    c = b if same else b'C' * size
    receive(h, replay_window(bw, a, c, level=level), route)
    assert b''.join(io.host) == a + b + (b'' if authorized and same else a + c), io.msgs
    assert not h._owed_block


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route,damage', [('stream', 'crc'), ('stream', 'timeout'), ('bracket', 'crc')])
@pytest.mark.parametrize('same', [False, True], ids=['new-tail', 'repeat'])
def test_incomplete_repeat_prefix_stays_ambiguous_until_recovery_authorized_resend(monkeypatch, bw, role, route, damage, same):
    authorized = True
    h, io = station(bw, role)
    level = 2
    size = phy.body_size(bw, phy.level_key(bw, level)) - 1
    a, b = b'A' * size, b'B' * size
    receive(h, replay_window(bw, a, b, level=level), route)
    assert b''.join(io.host) == a + b
    if authorized:
        arm_replay(h, io)
    identity = h._rx_window.K
    data = replay_window(bw, a, b, level=level)
    head = replay_window(bw, a, level=level)
    tail = full_body(b, h.caller, tail=0x99, body_len=size + 1)
    original = rx.frame_at

    def bad_tail(*args):
        frame = original(*args)
        return replace(frame, crc_ok=False) if bytes(frame.payload) == tail else frame

    with monkeypatch.context() as patch:
        if damage == 'crc':
            patch.setattr(rx, 'frame_at', bad_tail)
        else:
            data[len(head):] = 0
        if route == 'stream':
            patch.setattr(A, '_window_state', lambda *args: A._KEYING)
            step = M.FS // 50
            for at in range(0, len(data), step):
                _feed(h, data[at:at + step])
                if h._held_answer is not None:
                    break
            assert h._held_answer is not None, io.msgs
            _feed(h, data[at + step:])
            feed(h, np.zeros(A._ANSWER_HOLD_MAX))
        else:
            receive(h, data, route)
    assert b''.join(io.host) == a + b, io.msgs
    assert h._rx_window.K == identity and h._rx_window.A.kind == 'nak'
    assert h._rx_window.W is None
    assert h._rx_window.W is None
    c = b if same else b'C' * size
    receive(h, replay_window(bw, a, c, level=level), route)
    assert b''.join(io.host) == a + b + (b'' if authorized and same else a + c), io.msgs
    assert not h._owed_block


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_data_dispatch_classifies_once_for_delivery_and_progress(monkeypatch, role, route):
    h, io = station('2300', role)
    calls, progress = [], []
    commit, progressed = h._rx_window.commit, h._progressed
    def committed(*args, **kwargs):
        calls.append(True)
        return commit(*args, **kwargs)
    def advanced():
        progress.append(True)
        return progressed()
    monkeypatch.setattr(h._rx_window, 'commit', committed)
    monkeypatch.setattr(h, '_progressed', advanced)
    receive(h, replay_window(h.bw, b'A' * 89, b'B' * 89), route)
    assert len(calls) == 1 and progress
    assert b''.join(io.host) == b'A' * 89 + b'B' * 89


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_replay_match_cursor_does_not_count_as_host_delivery(bw, role):
    h, io = station(bw, role)
    body = full_body(b'A' * 89, h.caller)
    h._deliver([body], hold=True)
    assert h._rx_window.state == 'OPEN' and not io.host
    h._finish_delivery_window(complete=False)
    query(h, delay=12.)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert h._rx_window.T is None
    h._deliver([body], hold=True)
    assert h._rx_window.state == 'OPEN' and not io.host
    h._perform_rx_actions(h._rx_window.event('E', single=True))
    assert b''.join(io.host) == b'A' * 89


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('proof', ['unproven', 'query', 'held', 'after-held'])
def test_singleton_proof_decides_recurring_head_recovery(monkeypatch, bw, role, route, proof):
    h, io = station(bw, role)
    a, b = b'A' * 22, b'B' * 22
    if proof in ('held', 'after-held'):
        payload = b'P' * 22 if proof == 'after-held' else a
        with monkeypatch.context() as patch:
            patch.setattr(A, '_window_state', lambda *args: A._KEYING)
            feed(h, replay_window(bw, payload, level=2))
        assert h._held_answer is not None
        _Clock.t = h._decoded_frame_end + .7
        kind = (F.SESSION_RESPONDER_IDLE if role == 'initiator'
                else F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0])
        feed(h, M.synth_burst(h.called, F.for_bw(kind, bw)))
        assert h._held_answer is None and not h._owed_block, io.msgs
    if proof != 'held':
        receive(h, replay_window(bw, a, level=2),
                'stream' if proof in ('unproven', 'after-held') else route)
        if proof == 'query':
            query(h, delay=.2)
    prefix = b'P' * 22 if proof == 'after-held' else b''
    assert b''.join(io.host) == prefix + a
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == prefix + a + (a if role == 'responder' and proof == 'after-held' else b'') + b, io.msgs
    assert not h._owed_block and h._rx_window.T is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_f5f_l2_quiet_misread_recovers_missing_tail(monkeypatch, bw, role, route):
    h, io = station(bw, role)
    singles = [bytes([i]) * 22 for i in range(1, 7)]
    for payload in singles:
        feed(h, replay_window(bw, payload, level=2))
    a, b = b'defghij\r\nHello Kestrel', b' from Xara. abcdefghij'
    assert len(a) == len(b) == 22
    data = replay_window(bw, a, b, level=2)
    before = len(io.sent)
    with monkeypatch.context() as patch:
        patch.setattr(A, '_window_state', lambda *args: A._QUIET)
        step = M.FS // 50
        for at in range(0, len(data), step):
            _feed(h, data[at:at + step])
            if h._held_answer is not None or len(io.sent) > before:
                break
    assert io.host == singles and h._held_answer is not None, io.msgs
    assert at + step < len(data)
    feed(h, np.zeros(len(data) - at - step))
    before = len(io.sent)
    query(h, delay=6.35)
    assert_missing_nak(h, io, before)
    assert io.host == singles
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == b''.join(singles) + a + b, io.msgs
    assert not h._owed_block and h._rx_window.T is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route,damage', [('stream', 'crc'), ('stream', 'timeout'), ('bracket', 'crc')])
@pytest.mark.parametrize('nested', [False, True])
def test_damaged_repeat_repacked_lower_keeps_complete_candidate(monkeypatch, bw, role, route, damage, nested):
    authorized = False
    h, io = station(bw, role)
    a, b = bytes(range(22)), bytes(range(22, 44))
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == a + b
    if authorized:
        arm_replay(h, io)
    data = replay_window(bw, a, b, level=2)
    head = replay_window(bw, a, level=2)
    tail = full_body(b, h.caller, tail=0x99, body_len=23)
    original = rx.frame_at

    def bad_tail(*args):
        frame = original(*args)
        return replace(frame, crc_ok=False) if bytes(frame.payload) == tail else frame

    with monkeypatch.context() as patch:
        if damage == 'crc':
            patch.setattr(rx, 'frame_at', bad_tail)
        else:
            data[len(head):] = 0
        if route == 'stream':
            patch.setattr(A, '_window_state', lambda *args: A._KEYING)
            feed(h, data, np.zeros(A._ANSWER_HOLD_MAX))
        else:
            receive(h, data, route)
    assert h._owed_block and h._rx_window.W is None
    if not h._rx_window.A.sent:
        query(h, delay=6.35)
    size = phy.body_size(bw, phy.level_key(bw, 1)) - 1
    chunks = [(a + b)[i:i + size] for i in range(0, len(a + b), size)]
    receive(h, replay_window(bw, chunks[0], level=1), route)
    if nested:
        if route == 'stream':
            feed(h, np.zeros(M.FS))
        assert h._rx_window.W is None and h._rx_window.A.sent
        before = len(io.sent)
        query(h, delay=12.)
        assert_missing_nak(h, io, before)
        receive(h, replay_window(bw, chunks[0], level=1), route)
    for chunk in chunks[1:]:
        receive(h, replay_window(bw, chunk, level=1), route)
    receive(h, replay_window(bw, b'new', level=1), route)
    assert b''.join(io.host) == a + b + (b'' if authorized else a + b) + b'new', io.msgs
    assert h._rx_window.T is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route,damage', [('stream', 'crc'), ('stream', 'timeout'), ('bracket', 'crc')])
@pytest.mark.parametrize('nested', [False, True])
def test_damaged_repeat_repacked_lower_keeps_complete_candidate_authorized_resend(monkeypatch, bw, role, route, damage, nested):
    authorized = True
    h, io = station(bw, role)
    a, b = bytes(range(22)), bytes(range(22, 44))
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == a + b
    if authorized:
        arm_replay(h, io)
    data = replay_window(bw, a, b, level=2)
    head = replay_window(bw, a, level=2)
    tail = full_body(b, h.caller, tail=0x99, body_len=23)
    original = rx.frame_at

    def bad_tail(*args):
        frame = original(*args)
        return replace(frame, crc_ok=False) if bytes(frame.payload) == tail else frame

    with monkeypatch.context() as patch:
        if damage == 'crc':
            patch.setattr(rx, 'frame_at', bad_tail)
        else:
            data[len(head):] = 0
        if route == 'stream':
            patch.setattr(A, '_window_state', lambda *args: A._KEYING)
            feed(h, data, np.zeros(A._ANSWER_HOLD_MAX))
        else:
            receive(h, data, route)
    assert h._owed_block and h._rx_window.W is None
    if not h._rx_window.A.sent:
        query(h, delay=6.35)
    size = phy.body_size(bw, phy.level_key(bw, 1)) - 1
    chunks = [(a + b)[i:i + size] for i in range(0, len(a + b), size)]
    receive(h, replay_window(bw, chunks[0], level=1), route)
    if nested:
        if route == 'stream':
            feed(h, np.zeros(M.FS))
        assert h._rx_window.W is None and h._rx_window.A.sent
        before = len(io.sent)
        query(h, delay=12.)
        assert_missing_nak(h, io, before)
        receive(h, replay_window(bw, chunks[0], level=1), route)
    for chunk in chunks[1:]:
        receive(h, replay_window(bw, chunk, level=1), route)
    receive(h, replay_window(bw, b'new', level=1), route)
    assert b''.join(io.host) == a + b + (b'' if authorized else a + b) + b'new', io.msgs
    assert h._rx_window.T is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('end', ['repeat-retry', 'different-retry', 'early-query', 'bracket'])
def test_quiet_replay_head_waits_for_its_own_window_end(monkeypatch, bw, role, route, end):
    h, io = station(bw, role)
    a, b, c = b'A' * 22, b'B' * 22, b'C' * 22
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == a + b
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert b''.join(io.host) == a + b and h._rx_window.T is not None
    data = replay_window(bw, a, b, level=2)
    before = len(io.sent)
    with monkeypatch.context() as patch:
        patch.setattr(A, '_window_state', lambda *args: A._QUIET)
        step = M.FS // 50
        for at in range(0, len(data), step):
            _feed(h, data[at:at + step])
            if len(io.sent) > before:
                break
    assert at + step < len(data) and len(io.sent) == before + 1
    assert b''.join(io.host) == a + b and (h._rx_window.K.start, h._rx_window.K.end) == (0, 22), io.msgs
    assert h._rx_window.T is not None and h._rx_window.T is not None
    feed(h, np.zeros(0 if end == 'early-query' else len(data) - at - step))
    assert b''.join(io.host) == a + b
    before = len(io.sent)
    if end in ('repeat-retry', 'different-retry'):
        query(h, delay=12.)
        assert_missing_nak(h, io, before)
        assert b''.join(io.host) == a + b
        rejected_before = len(io.sent)
        receive(h, replay_window(bw, a, c if end == 'different-retry' else b, level=2), route)
        assert b''.join(io.host) == a + b, io.msgs
    elif end == 'early-query':
        query(h, delay=.2)
        assert len(io.sent) == before + 1 and not h._owed_block, io.msgs
        assert b''.join(io.host) == a + b, io.msgs
        assert h._rx_window.T is not None and h._rx_window.T is not None
        rejected_before = len(io.sent)
        receive(h, replay_window(bw, a, b, level=2), route)
        assert b''.join(io.host) == a + b, io.msgs
    else:
        query(h, delay=12.)
        assert_missing_nak(h, io, before)
        assert b''.join(io.host) == a + b
        receive(h, replay_window(bw, a, level=2), 'bracket')
        assert b''.join(io.host) == a + b and h._rx_window.T is not None, io.msgs
        query(h, delay=.2)
        assert b''.join(io.host) == a + b, io.msgs
        assert h._rx_window.T is not None and h._rx_window.T is not None
        rejected_before = len(io.sent)
        receive(h, replay_window(bw, a, b, level=2), route)
        assert b''.join(io.host) == a + b, io.msgs
    assert b''.join(io.host) == a + b
    if end != 'repeat-retry':
        assert len(io.sent) == rejected_before
        assert h._rx_window.closed and h._receive_reports[-1].code == 'R8'
        assert (h._receive_reports[-1].start, h._receive_reports[-1].frontier) == (0, 44)
    else:
        assert h._rx_window.T is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('empty', [False, True], ids=['damaged', 'nested-empty'])
def test_singleton_candidate_preserves_completeness_proof(monkeypatch, bw, role, route, empty):
    h, io = station(bw, role)
    a, b = b'A' * 22, b'B' * 22
    receive(h, replay_window(bw, a, level=2), route)
    query(h, delay=.2)
    tail = full_body(b, h.caller, tail=0x99, body_len=23)
    original = rx.frame_at

    def bad_tail(*args):
        frame = original(*args)
        return replace(frame, crc_ok=False) if bytes(frame.payload) == tail else frame

    with monkeypatch.context() as patch:
        patch.setattr(rx, 'frame_at', bad_tail)
        if route == 'stream':
            patch.setattr(A, '_window_state', lambda *args: A._KEYING)
            feed(h, replay_window(bw, a, b, level=2), np.zeros(A._ANSWER_HOLD_MAX))
        else:
            receive(h, replay_window(bw, a, b, level=2), route)
    assert io.host == [a] and h._owed_block, io.msgs
    if empty:
        h._finish_delivery_window(complete=False)
        assert h._rx_window.K is not None and b''.join(io.host) == a
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert io.host == [a]
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == a + a + b, io.msgs
    assert h._rx_window.T is None and not h._owed_block


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_unproven_candidate_survives_empty_nested_closure(bw, role):
    h, io = station(bw, role)
    h._rx_read = 2, None
    a, b = [full_body(payload, h.caller, 23, 0x99) for payload in (b'A' * 22, b'B' * 22)]
    commit_bodies(h, [a])
    arm_replay(h, io)
    anchor = h._rx_window.K
    h._deliver([a], hold=True)
    h._finish_delivery_window(complete=False)
    h._finish_delivery_window(complete=False)
    assert h._rx_window.K == anchor and b''.join(io.host) == b'A' * 22
    query(h, delay=12.)
    assert h._rx_window.T is not None
    commit_bodies(h, [a, b])
    assert b''.join(io.host) == b'A' * 22 + b'B' * 22


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_failed_companion_after_quiet_replay_head_retires_current_prefix(monkeypatch, bw, role, route):
    h, io = station(bw, role)
    a, b = b'A' * 22, b'B' * 22
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == a + b
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert b''.join(io.host) == a + b and h._rx_window.T is not None
    data = replay_window(bw, a, b, level=2)
    before = len(io.sent)
    with monkeypatch.context() as patch:
        patch.setattr(A, '_window_state', lambda *args: A._QUIET)
        step = M.FS // 50
        for at in range(0, len(data), step):
            _feed(h, data[at:at + step])
            if len(io.sent) > before:
                break
    assert at + step < len(data) and len(io.sent) == before + 1
    assert b''.join(io.host) == a + b
    assert h._rx_window.W is None and h._rx_window.T is not None
    feed(h, np.zeros(len(data) - at - step))
    assert b''.join(io.host) == a + b
    before = len(io.sent)
    h._undecoded_over()
    assert h._rx_window.W is None and h._owed_block
    assert b''.join(io.host) == a + b and h._rx_window.T is not None
    assert h._rx_window.T is not None
    assert len(io.sent) == before + int(h._nak_on_tape()), io.msgs
    if h._nak_on_tape():
        np.testing.assert_array_equal(io.sent[-1], M.synth_tone_pairs(F.nak(h.called, bw)[role == 'responder']))
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert b''.join(io.host) == a + b and h._rx_window.T is not None
    before = len(io.sent)
    receive(h, replay_window(bw, a, b, level=2), route)
    assert b''.join(io.host) == a + b and h._rx_window.closed, io.msgs
    assert len(io.sent) == before and h._rx_window.T is None
    assert h.state == A.VaraState.DISCONNECTED
    assert h._receive_reports[-1].code == 'R8'
    assert (h._receive_reports[-1].start, h._receive_reports[-1].frontier) == (0, 44)
    receive(h, replay_window(bw, b'new', level=2), route)
    assert b''.join(io.host) == a + b


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_double_plus_active_remainder_owes_nak_without_host_bytes_then_clean_retry_recovers_exactly(
        monkeypatch, bw, role):
    h, io = station(bw, role)
    a, b = bytes(range(22)), bytes(range(22, 44))
    clean = replay_window(bw, a, b, level=2)
    triple = replay_window(bw, a, b, b'C' * 22, level=2)
    audio = triple[:len(clean) + (len(triple) - len(clean)) // 3]
    reads = h._window_reads(audio)
    assert len(reads) == 3
    assert all(item[3].crc_ok for item in reads[:2])
    assert [bytes(item[3].payload) for item in reads[:2]] == [
        full_body(p, h.caller, 23, 0x99) for p in (a, b)]
    assert reads[2][3] is None
    record = rx.RECORDS[reads[2][0]]
    assert 0 < reads[2][5] - reads[2][4] < record.ncols * record.dw50

    events, states = [], []
    event = h._rx_window.event

    class WindowStates(type(h._rx_window)):
        def __setattr__(self, name, value):
            if name == 'state':
                states.append(value)
            super().__setattr__(name, value)

    def observed(kind, **kwargs):
        w = h._rx_window.W
        events.append((kind, len(w.frames) if w is not None else 0))
        return event(kind, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(h._rx_window, 'event', observed)
        patch.setattr(h._rx_window, '__class__', WindowStates)
        receive(h, audio, 'bracket')
    # The active tail closes the pair incomplete; it never becomes slot 3,
    # and no E can authorize a positive answer or host output for that pair.
    assert events == [('F', 0), ('F', 1), ('C', 2)]
    assert 'ENDED-INCOMPLETE' in states and 'ENDED-COMPLETE' not in states
    assert b''.join(io.host) == b'' and io.sent == []
    assert h._rx_window.W is None and h._rx_window.K is None
    assert h._rx_window.A.kind == 'nak' and not h._rx_window.A.sent
    assert h._owed_block and h._nak_data_due and h._query_data_due
    assert h._rx_window.T is None
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert b''.join(io.host) == b''
    receive(h, clean, 'bracket')
    assert b''.join(io.host) == a + b, io.msgs
    assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
    assert h._rx_window.W is None and h._rx_window.T is None
    assert not h._owed_block and not h._nak_data_due and not h._query_data_due


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_double_plus_quiet_remainder_completes_and_emits_exact_pair(monkeypatch, bw, role):
    h, io = station(bw, role)
    a, b = bytes(range(22)), bytes(range(22, 44))
    clean = replay_window(bw, a, b, level=2)
    triple = replay_window(bw, a, b, b'C' * 22, level=2)
    audio = np.r_[clean, np.zeros((len(triple) - len(clean)) // 3)]
    reads = h._window_reads(audio)
    assert len(reads) == 2 and all(item[3].crc_ok for item in reads)
    head = replay_window(bw, a, level=2)
    level, _, _, mag = h._index_alignment(audio[len(head):])
    assert A._window_state(mag, rx.RECORDS[level]) == A._QUIET

    events = []
    event = h._rx_window.event

    def observed(kind, **kwargs):
        events.append(kind)
        return event(kind, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(h._rx_window, 'event', observed)
        receive(h, audio, 'bracket')
    assert events == ['F', 'F', 'E']
    assert b''.join(io.host) == a + b, io.msgs
    assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
    assert h._rx_window.W is None and len(h._rx_window.K.frames) == 2
    assert not h._owed_block and not h._nak_data_due and not h._query_data_due


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_below_threshold_remainder_ends_a_single_window(bw, role):
    from hfmodem.kestrel.rx import varahf2300 as rx
    from hfmodem.kestrel.vara import vara_arq as A

    h, io = station(bw, role)
    a, b = b'A' * 22, b'B' * 22
    head = replay_window(bw, a, level=2)
    data = replay_window(bw, a, b, level=2)
    audio = data[:len(head) + (len(data) - len(head)) // 3].copy()
    noise = np.random.default_rng(56).normal(0, .01, len(audio) - len(head))
    audio[len(head):] = audio[len(head):] * .001 + noise
    level, _, _, mag = h._index_alignment(audio)
    assert A._window_state(mag, rx.RECORDS[level]) == A._QUIET
    receive(h, audio, 'bracket')
    assert b''.join(io.host) == a
    assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
    assert h._rx_window.W is None



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('repeat', [False, True])
def test_open_window_ceiling_drops_uncommitted_prefix(bw, repeat):
    h, io = station(bw, 'initiator')
    a, b = b'A' * 89, b'B' * 89
    bodies = [full_body(p, h.caller) for p in (a, b)]
    commit_bodies(h, bodies)
    h._deliver(bodies if repeat else bodies[:1], hold=True)
    assert h._rx_window.state == 'OPEN' and b''.join(io.host) == a + b
    h._perform_rx_actions(h._rx_window.event('C', peer_keying=True))
    assert h._rx_window.W is None and h._rx_window.A.kind == 'nak'
    assert h._rx_window.T is None and b''.join(io.host) == a + b


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('same', [False, True], ids=['distinct', 'identical'])
@pytest.mark.parametrize('failed_retry', [False, True])
def test_incomplete_terminal_unextended(bw, same, failed_retry):
    test_incomplete_window_replay_prefix_is_consumed_once(bw, same, failed_retry, 2, unextended=True)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_crc_failure_terminal_unextended(monkeypatch, bw, role):
    test_crc_failure_retires_held_window_before_prompt_retry(
        monkeypatch, bw, role, 2, 'stream', unextended=True)



def test_double_found_from_its_second_frame_reads_both_frames_and_no_hole():
    # Two clean frames tie on reference columns, so the search may land on the
    # second; the audio ahead of it then ends in that frame's own lead-in.
    from hfmodem.kestrel.vara import vara_ofdm as OF
    h, io = station('2300', 'initiator')
    payloads = [bytes([31]) * 89, bytes([73]) * 89]
    bodies = [phy.vara_body(p, h.caller, tail=0x99) for p in payloads]
    head = OF.data_over_tx(bodies[0], over=1)
    audio = np.concatenate([head, OF.data_over_tx(bodies[1], over=2)])
    level, hits, of, mag = h._index_alignment(audio[len(head):])
    reads = h._window_reads(audio, (level, hits, of, mag, None))
    assert [item[3] is not None and item[3].crc_ok for item in reads] == [True, True]
    assert [bytes(item[3].payload) for item in reads] == bodies
