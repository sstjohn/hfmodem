# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A late query cannot acknowledge unread wide DATA."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from dataclasses import replace

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock, _feed  # noqa: F401
from .receive_window_helpers import (station, window, feed, assert_missing_nak,
                                     assert_bracket_recovery_nak, query, full_body)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('retry_level', [4, 2])
@pytest.mark.parametrize('caller', ['W9SSJ', 'N0CALL'])
def test_quiet_misread_double_query_requests_missing_data(
        monkeypatch, request, record_property, bw, role, retry_level, caller, unextended=False):
    """QUIET cannot discharge a CERTAIN companion."""
    h, io = station(bw, role, caller)
    history = []
    event, perform, sent = h._rx_window.event, h._perform_rx_actions, h._rx_window.sent

    def snapshot():
        machine = h._rx_window
        return {'state': machine.state, 'R': machine.R, 'alt': machine.alt,
                'trusted': machine.trusted,
                **{name: asdict(value) if value is not None else None
                   for name in ('W', 'A', 'K', 'T')
                   for value in (getattr(machine, name),)}}

    def observed(kind, **kwargs):
        entry = {'event': kind, 'input': kwargs, 'before': snapshot()}
        history.append(entry)
        actions = event(kind, **kwargs)
        entry['after'] = snapshot()
        return actions

    def performed(actions, **kwargs):
        history.extend({'action': type(a).__name__, 'value': asdict(a)} for a in actions)
        return perform(actions, **kwargs)

    def aired(key, reached_air):
        history.append({'sent': asdict(key), 'reached_air': reached_air})
        return sent(key, reached_air)

    def encoded(value):
        if isinstance(value, bytes):
            return {'hex': value.hex()}
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        return asdict(value)

    monkeypatch.setattr(h._rx_window, 'event', observed)
    monkeypatch.setattr(h, '_perform_rx_actions', performed)
    monkeypatch.setattr(h._rx_window, 'sent', aired)
    request.addfinalizer(lambda: record_property('receive_evidence', json.dumps({
        'history': history, 'final': snapshot(), 'host_hex': b''.join(io.host).hex(),
        'messages': io.msgs, 'keyings': len(io.sent),
        'expected_hex': (b''.join(bytes([i]) * 89 for i in range(1, 7))
                         + (b'' if caller == 'N0CALL' and role == 'initiator' and bw == '2750'
                            else bytes(range(176 if unextended else 178)))).hex(),
    }, default=encoded)))
    singles = [bytes([i]) * 89 for i in range(1, 7)]
    for payload in singles:
        before = len(io.sent)
        feed(h, window(bw, payload, caller=caller))
        assert len(io.sent) == before + 1 and io.host[-1] == payload, io.msgs
    a, b = bytes(range(89)), bytes(range(89, 178))
    double = window(bw, a, b, caller=caller)
    before = len(io.sent)
    original = A._window_state
    misread = []

    def quiet_once(mag, record):
        if not misread:
            misread.append(True)
            return A._QUIET
        return original(mag, record)

    monkeypatch.setattr(A, '_window_state', quiet_once)
    step = M.FS // 50
    for at in range(0, len(double), step):
        _feed(h, double[at:at + step])
        if h._held_answer is not None or len(io.sent) > before:
            break
    assert misread and io.host == singles, io.msgs
    assert h._held_answer is not None and len(io.sent) == before
    monkeypatch.setattr(A, '_window_state', original)
    feed(h, np.zeros(max(0, len(double) - at - step) + A._ANSWER_HOLD_MAX))
    assert h._held_answer is None and h._owed_block
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert h._rx_window.T is None
    before = len(io.sent)
    query(h, delay=6.4)
    if caller == 'N0CALL' and role == 'initiator' and bw == '2750':
        assert h.state == A.VaraState.DISCONNECTED
        assert b''.join(io.host) == b''.join(singles)
        report = h._receive_reports[-1]
        assert report.code == 'R8'
        assert (report.start, report.frontier) == (5 * 89, 6 * 89)
        assert report.rejected == b''
        assert any('unconfirmed' in msg.lower() for msg in io.msgs)
        return
    assert_missing_nak(h, io, before)
    assert io.host == singles
    before = len(io.sent)
    query(h, delay=.2)
    assert_missing_nak(h, io, before)
    assert io.host == singles
    size = phy.body_size(bw, phy.level_key(bw, retry_level)) - 1
    resend = a + b
    chunks = [resend[i:i + size] for i in range(0, len(resend), size)]
    if retry_level == 4:
        feed(h, window(bw, *chunks, level=retry_level, caller=caller))
    else:
        for chunk in chunks:
            feed(h, window(bw, chunk, level=retry_level, caller=caller))
        from .receive_window_helpers import terminal_retry
        terminal_retry(h, io, chunks[-1], b''.join(singles) + a + b, unextended=unextended)
        if unextended:
            assert b''.join(io.host) == b''.join(singles) + bytes(range(176))
            return
    assert b''.join(io.host) == b''.join(singles) + a + b, '\n'.join(io.msgs)
    assert not h._rx_failed and not h._owed_block




@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_lost_next_over_query_requests_data(bw, role):
    h, io = station(bw, role)
    feed(h, window(bw, b'A' * 89))
    before = len(io.sent)
    query(h, delay=len(window(bw, b'B' * 89)) / M.FS + .3)
    assert_missing_nak(h, io, before)
    assert io.host == [b'A' * 89]



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_query_inside_turnaround_repeats_lost_answer(bw, role):
    h, io = station(bw, role)
    feed(h, window(bw, b'A' * 89))
    before = len(io.sent)
    first = io.sent[-1]
    query(h, delay=.2)
    assert len(io.sent) == before + 1, io.msgs
    if role == 'responder':
        first = M.synth_burst(h.called, F.for_bw(F.SESSION_INTERMEDIATE_QUERY_ANSWER, bw))
    np.testing.assert_array_equal(io.sent[-1], first)
    assert io.host == [b'A' * 89] and not h._rx_failed



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('high_speed', [False, True])
@pytest.mark.parametrize('route', ['_reack_over', '_reack_answer', '_reanswer_over'])
def test_refused_late_nak_cannot_turn_the_next_query_positive(bw, high_speed, route):
    role = 'responder' if route == '_reanswer_over' else 'initiator'
    h, io = station(bw, role)
    feed(h, window(bw, b'A' * 89))
    h._link_high_speed = high_speed
    _Clock.t = h._unkeyed_at + 6.4
    h._idle_kind = F.SESSION_RESPONDER_IDLE
    h._idle_end = _Clock.t
    answer = getattr(h, route)
    io.tx_went_out = lambda: False
    before = len(io.sent)
    answer()
    assert_missing_nak(h, io, before)
    io.tx_went_out = lambda: True
    before = len(io.sent)
    query(h, delay=.2)
    assert_missing_nak(h, io, before)
    assert io.host == [b'A' * 89]



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('bad_first', [False, True], ids=['bad-tail', 'bad-head'])
def test_bracket_with_identified_bad_companion_is_incomplete(monkeypatch, bw, bad_first):
    h, io = station(bw, 'initiator')
    original = rx.frame_at
    damaged = full_body(b'B' * 89, h.caller)

    def fail_companion(mag, level):
        from dataclasses import replace
        fr = original(mag, level)
        return replace(fr, crc_ok=False) if bytes(fr.payload) == damaged else fr

    monkeypatch.setattr(rx, 'frame_at', fail_companion)
    payloads = [b'B' * 89, b'A' * 89] if bad_first else [b'A' * 89, b'B' * 89]
    h.on_rx_audio(window(bw, *payloads))
    assert h._owed_block and not h._held_answer, io.msgs
    assert b''.join(io.host) == b''
    assert_bracket_recovery_nak(h, io, 0)
    if not bad_first:
        monkeypatch.setattr(rx, 'frame_at', original)
        feed(h, window(bw, b'A' * 89))
    assert b''.join(io.host) == (b'' if bad_first else b'A' * 89), io.msgs



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_bracket_with_unlocated_erased_companion_is_incomplete(monkeypatch, bw, role):
    h, io = station(bw, role)
    a, b = bytes(range(89)), bytes(range(89, 178))
    record = phy.level_key(bw, 4)
    r = rx.RECORDS[record]
    span = r.ncols * r.dw50
    first = window(bw, a)
    erased = np.random.default_rng(43).normal(0, np.std(first), span)
    original = h._index_alignment
    unlocated = []

    def locate(x):
        if len(x) <= span + M.FS // 2:
            unlocated.append(True)
            return record, 0, 24, None
        return original(x)

    monkeypatch.setattr(h, '_index_alignment', locate)
    assert h._answer_data_over(np.r_[np.zeros(M.FS // 4), first, erased, np.zeros(M.FS // 4)])
    assert unlocated
    assert b''.join(io.host) == b'', io.msgs
    assert h._owed_block and h._rx_window.W is None and not h._held_answer, io.msgs
    assert_bracket_recovery_nak(h, io, 0)
    monkeypatch.setattr(h, '_index_alignment', original)
    feed(h, window(bw, a, b))
    assert b''.join(io.host) == a + b, io.msgs



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_single_over_bracket_padding_is_not_an_erased_companion(bw, role):
    h, io = station(bw, role)
    a = bytes(range(89))
    assert h._answer_data_over(np.r_[np.zeros(3 * M.FS), window(bw, a), np.zeros(5 * M.FS)])
    assert io.host == [a] and len(io.sent) == 1, io.msgs
    assert (h._rx_window.A is not None and h._rx_window.A.kind == 'continue') and not h._owed_block



@pytest.mark.parametrize('recording', ['weak-0920', 'greeting-0820', 'challenge-0820'])
def test_measured_single_over_bracket_stays_positive(recording):
    from . import corpora
    from .test_weak_data_0920 import audio, PAYLOAD

    if recording == 'weak-0920':
        data = audio()[:round(5.2 * M.FS)]
        h, io = station('2750', 'initiator')
    else:
        path = corpora.ONAIR_NAKED_OVER
        if not path.is_file():
            pytest.skip(f'original KC9GHZ 2026-08-20 recording is unavailable: {path}')
        start, end = ((63.443, 68.157) if recording == 'greeting-0820' else (69.924, 74.638))
        data = corpora.wav_mono(path)[round(start * M.FS):round(end * M.FS)] / 32768.0
        h, io = station('2300', 'initiator')
    assert h._answer_data_over(data)
    assert io.host and len(io.sent) == 1, io.msgs
    assert (h._rx_window.A is not None and h._rx_window.A.kind == 'continue') and not h._owed_block, io.msgs
    if recording == 'weak-0920':
        assert io.host == [PAYLOAD]



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('missing', ['head', 'tail'])
@pytest.mark.parametrize('closing', [False, True])
def test_expected_silent_frame_waits_for_ordered_recovery(monkeypatch, bw, role, route, missing, closing, repeat_previous=False, literal=False):
    h, io = station(bw, role)
    singles = [bytes([i]) * 89 for i in range(1, 7)]
    for p in singles:
        if route == 'bracket':
            data = window(bw, p)
            _Clock.t += (len(data) + M.FS // 2) / M.FS
            assert h._answer_data_over(np.r_[data, np.zeros(M.FS // 2)])
        else:
            feed(h, window(bw, p))
    assert h._peer_doubles()
    a = singles[-1] if repeat_previous else bytes(range(89))
    b = b'close' if closing else bytes(range(89, 178))
    data = window(bw, a, b)
    cut = len(window(bw, a))
    data[:cut] = 0 if missing == 'head' else data[:cut]
    if missing == 'tail':
        data[cut:] = 0
    original = A._window_state
    monkeypatch.setattr(A, '_window_state', lambda *args: A._QUIET)
    before = len(io.sent)
    if route == 'bracket':
        quiet = h._companion_samples() + M.FS // 2
        _Clock.t += (len(data) + quiet) / M.FS
        assert h._answer_data_over(np.r_[data, np.zeros(quiet)])
        assert_missing_nak(h, io, before)
    else:
        feed(h, data)
        assert b''.join(io.host) == b''.join(singles)
        assert h._rx_window.W is not None or h._rx_window.A.kind == 'nak'
        if h._rx_window.A is not None and h._rx_window.A.kind == 'nak' and h._rx_window.A.sent:
            assert_missing_nak(h, io, before)
            before = len(io.sent)
        start = max(_Clock.t, h._decoded_frame_end + .2)
        _Clock.t = start
        kind = F.SESSION_RESPONDER_IDLE if role == 'initiator' else F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]
        feed(h, M.synth_burst(h.called, F.for_bw(kind, bw)))
        assert_missing_nak(h, io, before)
    assert io.host == singles
    assert h._rx_window.W is None and h._rx_window.A.kind == 'nak'
    assert h._rx_window.T is None and h._owed_block
    monkeypatch.setattr(A, '_window_state', original)
    feed(h, window(bw, a, b))
    assert b''.join(io.host) == b''.join(singles) + a + b, io.msgs
    assert not h._owed_block and h._held_answer is None
    before_host = b''.join(io.host)
    if literal:
        assert h._rx_window.T is None
        feed(h, window(bw, a, b))
        assert b''.join(io.host) == before_host + a + b
        return
    if closing:
        assert h._rx_window.A.kind == 'final'
        query(h, delay=12.)
    else:
        before = len(io.sent)
        query(h, delay=12.)
        assert_missing_nak(h, io, before)
        assert h._rx_window.T is not None
        feed(h, window(bw, a, b))
    assert b''.join(io.host) == before_host
    if repeat_previous:
        assert b''.join(io.host) == b''.join(singles) + a + bytes(range(89, 178))



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('remaining', [0, 1])
def test_fewer_than_two_remaining_loads_do_not_predict_a_double(bw, role, remaining):
    h, io = station(bw, role)
    for i in range(1, 6):
        feed(h, window(bw, bytes([i]) * 89))
    record = phy.level_key(bw, 4)
    body = full_body(bytes([6]) * 89, h.caller, tail=0x89 + 4 * remaining)
    feed(h, tx.synth_burst(body, record, 43))
    assert phy.fit_hint(body[-1]) == (3 if remaining else 2)
    assert h._holds_epoch == (6 if remaining else 0)
    last_full = full_body(b'A' * 89, h.caller, tail=0x89)
    feed(h, tx.synth_burst(last_full, record, 44))
    if remaining:
        assert b''.join(io.host) == b''.join(bytes([i]) * 89 for i in range(1, 7))
        feed(h, np.zeros(A._ANSWER_HOLD_MAX))
        if not h._rx_window.A.sent:
            query(h, delay=12.)
        assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
        assert h._rx_window.T is None
        feed(h, tx.synth_burst(last_full, record, 45))
    assert h._held_answer is None
    feed(h, window(bw, b'close'))
    assert b''.join(io.host) == b''.join(bytes([i]) * 89 for i in range(1, 7)) + b'A' * 89 + b'close'
    assert not h._owed_block



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_unplaced_window_does_not_register_the_previous_over_as_its_prefix(monkeypatch, bw, role, route):
    test_expected_silent_frame_waits_for_ordered_recovery(
        monkeypatch, bw, role, route, 'tail', False, repeat_previous=True)



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('damage', ['bad', 'unlocated'])
def test_empty_current_prefix_cannot_suppress_previous_equal_head(monkeypatch, bw, role, damage):
    h, io = station(bw, role)
    singles = [bytes([i]) * 89 for i in range(1, 7)]
    for payload in singles:
        training = window(bw, payload)
        _Clock.t += (len(training) + M.FS // 2) / M.FS
        assert h._answer_data_over(np.r_[training, np.zeros(M.FS // 2)])
        assert len(io.sent) == len(io.host) and h._rx_window.A.step == 0
    a, b = singles[-1], b'B' * 89
    data = window(bw, a, b)
    original_frame = rx.frame_at
    head = full_body(a, h.caller)

    def bad_frame(*args):
        fr = original_frame(*args)
        return replace(fr, crc_ok=False) if bytes(fr.payload) == head else fr

    if damage == 'unlocated':
        cut = len(window(bw, a))
        data[:cut] = np.random.default_rng(1043).normal(0, .3, cut)

    with monkeypatch.context() as patch:
        if damage == 'bad':
            patch.setattr(rx, 'frame_at', bad_frame)
        before = len(io.sent)
        _Clock.t += len(data) / M.FS
        assert h._answer_data_over(data)
        assert_bracket_recovery_nak(h, io, before, committed=b''.join(singles))
    assert io.host == singles and h._owed_block
    feed(h, window(bw, a, b))
    assert b''.join(io.host) == b''.join(singles) + a + b, io.msgs
    assert not h._owed_block



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('missing', ['head', 'tail'])
def test_confirm_limited_peer_predicts_seventh_window(monkeypatch, bw, missing):
    h, io = station(bw, 'initiator')
    monkeypatch.setitem(phy.RX_CEILING, bw, phy.TOP[bw])
    h.high_speed = h._link_high_speed = True
    h._heard_db = 30.
    h._tx_session_confirm()
    assert h._expect_level == h._confirm_start == 6
    singles = [bytes([i]) * 89 for i in range(1, 7)]
    for payload in singles:
        training = window(bw, payload)
        _Clock.t += (len(training) + M.FS // 2) / M.FS
        assert h._answer_data_over(np.r_[training, np.zeros(M.FS // 2)])
    assert not h._link_high_speed
    a, b = b'A' * 89, b'B' * 89
    data = window(bw, a, b)
    cut = len(window(bw, a))
    data[:cut] = 0 if missing == 'head' else data[:cut]
    if missing == 'tail':
        data[cut:] = 0
    before = len(io.sent)
    quiet = h._companion_samples() + M.FS // 2
    _Clock.t += (len(data) + quiet) / M.FS
    assert h._answer_data_over(np.r_[data, np.zeros(quiet)])
    assert_missing_nak(h, io, before)
    assert io.host == singles and h._owed_block
    assert h._rx_read[0] == 4
    feed(h, window(bw, a, b))
    assert b''.join(io.host) == b''.join(singles) + a + b and not h._owed_block, io.msgs
    assert h._expect_level == 4



@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_accepted_earlier_retry_reconciles_receive_level(bw, role):
    h, io = station(bw, role)
    feed(h, window(bw, b'A' * 89))
    h._advance_resend(missing=True)
    assert h._expect_level == 2
    h._advance_resend(missing=True)
    assert h._expect_level == 1
    retry = b'B' * (phy.body_size(bw, phy.level_key(bw, 2)) - 1)
    feed(h, window(bw, retry, level=2))
    assert io.host == [b'A' * 89, retry]
    assert h._expect_level == h._rx_read[0] == 2 and h._rx_resend is None


from .receive_window_helpers import rx_evidence  # noqa: F401


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('missing', ['head', 'tail'])
@pytest.mark.parametrize('closing', [False, True])
def test_expected_silent_frame_literal_new_data(monkeypatch, bw, role, route, missing, closing):
    test_expected_silent_frame_waits_for_ordered_recovery(
        monkeypatch, bw, role, route, missing, closing, literal=True)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_unplaced_window_literal_new_data(monkeypatch, bw, role, route):
    test_expected_silent_frame_waits_for_ordered_recovery(
        monkeypatch, bw, role, route, 'tail', False, repeat_previous=True, literal=True)


@pytest.mark.parametrize('bw,role,caller', [
    ('2300', 'initiator', 'W9SSJ'), ('2750', 'initiator', 'W9SSJ'),
    ('2300', 'responder', 'W9SSJ'), ('2750', 'responder', 'W9SSJ'),
    ('2300', 'initiator', 'N0CALL'), ('2300', 'responder', 'N0CALL'),
    ('2750', 'responder', 'N0CALL')])
def test_quiet_misread_terminal_unextended(monkeypatch, request, record_property, bw, role, caller):
    test_quiet_misread_double_query_requests_missing_data(
        monkeypatch, request, record_property, bw, role, 2, caller, unextended=True)
