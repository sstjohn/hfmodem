# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""UI exclusion precedes receive-window and payload state on every route."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf500 as RX500
from hfmodem.kestrel.rx.observation import CaptureContext, LocatedFrame
from hfmodem.kestrel.rx.receive_context import ReceiveContext, HOLD_SAMPLES, INPUT_STEP
from hfmodem.kestrel.rx.ui_spans import UISpan, UISpanLedger
from hfmodem.kestrel.vara import vara_arq as A
from .receive_window_helpers import full_body
from .test_rxwindow_adapter import station


def receiver(*, mode='OFF', high=True, bw='500', role='responder'):
    hs, io = station(mode=mode, high=high, bw=bw, role=role)
    hs._rx_read = (4, None)
    hs.receive_context = SimpleNamespace(ledger=UISpanLedger(hs._tune_epoch),
        watermark=10**9, input_clock=1000000)
    hs.receive_context.retune = lambda epoch, start: hs.receive_context.ledger.retune(epoch)
    return hs, io


def located(hs, ordinal=0, *, start=100000, body=None, level=4, bw=500):
    if body is None:
        body = full_body((b'peer-' + bytes([65 + ordinal])).ljust(43, b'.'), hs.caller, size=44,
                         field=0x85)
    record = RX500.INDEX_RECORDS.get(level) if bw == 500 else None
    extent = ((record.lead + record.ncols) * record.dw if record is not None else
              (9 + RX500.NSYM) * RX500.H + 1 if bw == 500 and level == 4 else 200000)
    return LocatedFrame(ordinal, hs.capture_context().origin(
        start, start + extent, level, bw), body)


def contest(hs, observation, span_id=1):
    span = UISpan(span_id, hs._tune_epoch, observation.origin.level,
                  observation.origin.training_origin - 67072)
    hs.receive_context.ledger.publish(span)
    return span


def read_attempt(hs, observations, *, hold=False):
    read = phy.LevelRead(observations[0].origin.level, observations=tuple(observations))
    hs._level_read = read
    try:
        return hs._answer_data_over(np.zeros(10), hold=hold)
    finally:
        hs._level_read = None


def trace_window(monkeypatch, hs):
    events = []
    original = hs._rx_window.event
    def event(kind, **kwargs):
        events.append((kind, kwargs))
        return original(kind, **kwargs)
    monkeypatch.setattr(hs._rx_window, 'event', event)
    return events


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_unindexed_level_read_keeps_legacy_window_proof(monkeypatch, bw):
    hs, io = station(bw=bw)
    hs.rx_discontinuity()
    size = phy.body_size(bw, phy.level_key(bw, 4))
    payload = b'peer-A'.ljust(size - 1, b'.')
    body = full_body(payload, hs.caller, size=size, field=0x85)
    observation = LocatedFrame(0, CaptureContext(0, 0, 0, A.MK.FS).origin(
        100000, 200000, 4, int(bw)), body)
    assert not hs.origin_is_current(observation.origin)
    hs._level_read = phy.LevelRead(4, observations=(observation,), end=1)
    monkeypatch.setattr(phy, 'burst_extent', lambda *args: 'single')
    try:
        assert hs._answer_data_over(np.zeros(10))
    finally:
        hs._level_read = None
    assert hs._pending_receive is None
    assert hs._decoded_frame_end is not None
    assert hs._rx_window.A is not None and hs._rx_window.A.kind != 'nak'
    assert b''.join(io.host) == payload


@pytest.mark.parametrize('route', ['level', 'segmented', 'located', 'expected'])
def test_whole_window_nak_uses_provenance_before_peer_state(monkeypatch, route):
    hs, io = receiver()
    observation = located(hs)
    contest(hs, observation)
    events = trace_window(monkeypatch, hs)
    before = hs._peer_over, hs._read_streak, hs._peer_turn_proven, hs._close_low
    if route == 'level':
        assert read_attempt(hs, [observation])
    elif route == 'expected':
        read = phy.LevelRead(4, observations=(observation,))
        assert hs._take_expected_read(read, np.zeros(10), A.time.monotonic(), True, True, 10)
    else:
        frame = RX500.FrameResult(0, (0, 256), 9, 1., True,
            observation.body + b'\0\0', level=4, origin=observation.origin)
        result = RX500.DecodeResult([frame], [frame])
        if route == 'segmented':
            monkeypatch.setattr(RX500, 'decode_stream', lambda *a, **kw: result)
            assert hs._answer_data_over(np.zeros(10))
        else:
            over = SimpleNamespace(generation=hs._front_generation, frames=[frame],
                result=result, end=10, decision=10, candidate=SimpleNamespace(start=0))
            monkeypatch.setattr(hs, '_read_ident_500', lambda end: False)
            hs._front_heard, hs._base_ready = 10, [over]
            assert hs._take_located_500()
    assert (hs._peer_over, hs._read_streak, hs._peer_turn_proven, hs._close_low) == before
    assert not io.host and hs._rx_window.R == b'' and hs._rx_window.K is None
    assert hs._rx_window.W is None and hs._rx_window.A.kind == 'nak'
    x = next(kwargs for kind, kwargs in events if kind == 'X')
    assert x['level'] == 4 and x.get('frame') is None
    assert not any(kind == 'F' for kind, _ in events)


@pytest.mark.parametrize('contested', [0, 1])
def test_both_frame10_cases(contested):
    hs, io = receiver(high=False)
    observations = [located(hs, i, start=100000 + i * RX500.NSYM * RX500.H)
                    for i in range(2)]
    contest(hs, observations[contested])
    assert read_attempt(hs, observations)
    if contested == 0:
        assert not io.host and hs._rx_window.A.kind == 'nak'
        assert hs._rx_window.K is None and hs._rx_window.R == b''
    else:
        assert hs._rx_window.A.kind == 'frame10'
        assert hs._rx_window.K.frames[0].body == observations[0].body
        assert len(hs._rx_window.K.frames) == 1
        assert b''.join(io.host) == b'peer-A'.ljust(43, b'.')


@pytest.mark.parametrize('keying,accept', [(True, True), (False, False)])
def test_nak_is_deferred_or_refused_without_positive_state(keying, accept):
    hs, io = receiver()
    io.accept = accept
    observation = located(hs)
    contest(hs, observation)
    assert read_attempt(hs, [observation], hold=keying)
    assert hs._rx_window.A.kind == 'nak' and not hs._rx_window.A.sent
    assert hs._rx_window.K is None and not hs._rx_window.R and not io.host


def test_held_companion_blocks_head_preparation_and_success_hints():
    hs, io = receiver(high=False)
    head, tail = located(hs), located(hs, 1, start=301728)
    contest(hs, tail)
    hs.receive_context.watermark = head.origin.end_sample + 1
    before = hs._peer_over, hs._read_streak, hs._rx_read, hs._peer_hint, hs._close_low
    assert not read_attempt(hs, [head, tail])
    assert hs._pending_receive is not None and hs._rx_window.W is None
    assert not io.host and hs._rx_window.K is None
    assert (hs._peer_over, hs._read_streak, hs._rx_read, hs._peer_hint, hs._close_low) == before
    hs.receive_context.watermark = tail.origin.end_sample + 1
    hs._release_receive_attempt()
    assert hs._pending_receive is None and hs._rx_window.A.kind == 'frame10'
    assert b''.join(io.host) == b'peer-A'.ljust(43, b'.')


@pytest.mark.parametrize('mode', ['TEXT', 'FILES'])
def test_contested_poison_body_never_reaches_preparation(monkeypatch, mode):
    hs, io = receiver(mode=mode)
    observation = located(hs, body=b'poison compressed UI body\xa1')
    contest(hs, observation)
    def poison(*args, **kwargs):
        raise AssertionError('contested payload reached preparation')
    monkeypatch.setattr(phy, 'vara_payload', poison)
    monkeypatch.setattr(A.COMP, 'decode_text', poison)
    monkeypatch.setattr(hs._rx_compression, 'preview_files', poison)
    assert read_attempt(hs, [observation])
    assert not io.host and not io.warnings and not hs._rx_window.closed
    assert hs._rx_window.A.kind == 'nak'
    assert not hs._rx_window.files.pending and not hs._rx_window.files.blocks


@pytest.mark.parametrize('reset', ['T', 'rx_discontinuity', 'key', 'retune'])
def test_reset_drops_hold_without_admitting(reset):
    hs, io = receiver()
    observation = located(hs)
    hs.receive_context.watermark = 0
    assert not read_attempt(hs, [observation])
    if reset == 'T':
        hs._perform_rx_actions(hs._receive_window_event('T'))
    elif reset == 'key':
        hs._key(True)
        hs._key(False)
    elif reset == 'retune':
        hs.retuned()
    else:
        hs.rx_discontinuity()
    assert hs._pending_receive is None
    hs._release_receive_attempt()
    assert hs._rx_window.W is None and hs._rx_window.K is None and not io.host


@pytest.mark.parametrize('pending', ['head-and-companion', 'entire-attempt'])
def test_ceiling_resolves_pending_ordinals_before_c(monkeypatch, pending):
    hs, io = receiver(high=False)
    head, tail = located(hs), located(hs, 1, start=301728)
    hs.receive_context.watermark = 0
    events = trace_window(monkeypatch, hs)
    if pending == 'head-and-companion':
        assert not read_attempt(hs, [head])
    assert not read_attempt(hs, [head, tail])
    hs._perform_rx_actions(hs._receive_window_event('C'))
    kinds = [kind for kind, _ in events]
    assert kinds[:3] == ['X', 'X', 'C']
    assert 'F' not in kinds and 'E' not in kinds
    assert hs._pending_receive is None and not io.host
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.K is None


def test_stale_result_after_tx_is_discarded_at_the_seam():
    hs, io = receiver()
    observation = located(hs)
    hs._key(True)
    hs._key(False)
    assert hs._admit_receive_observation(observation) == 'DISCARD'
    assert not read_attempt(hs, [observation])
    assert hs._pending_receive is None and hs._rx_window.W is None and not io.host


def test_stale_floor_becomes_x_after_span_retirement():
    hs, io = receiver()
    observation = located(hs)
    span = contest(hs, observation)
    ledger = hs.receive_context.ledger
    ledger.set_frontier('inactive', span.classification_end + 1)
    ledger.advance(span.deadline + 1, span.deadline + 1)
    assert not ledger.spans
    assert hs._admit_receive_observation(observation) == 'X'
    assert read_attempt(hs, [observation])
    assert hs._rx_window.A.kind == 'nak' and not io.host


@pytest.mark.parametrize('outcome', ['failed', 'truncated', 'group'])
def test_later_ui_outcomes_never_upgrade_an_x(outcome):
    hs, io = receiver()
    observation = located(hs)
    span = contest(hs, observation)
    assert read_attempt(hs, [observation])
    answer = hs._rx_window.A
    kwargs = dict(count=1, validated_end=observation.origin.end_sample) if outcome == 'group' else {}
    hs.receive_context.ledger.resolve(span.id, outcome, **kwargs)
    hs._release_receive_attempt()
    assert hs._rx_window.A is answer and answer.kind == 'nak'
    assert hs._rx_window.K is None and not io.host


def test_hold_expiry_produces_x_never_f(monkeypatch):
    hs, io = receiver()
    observation = located(hs)
    hs.receive_context.watermark = 0
    events = trace_window(monkeypatch, hs)
    start = hs.receive_context.input_clock
    assert not read_attempt(hs, [observation])
    hs.receive_context.input_clock = start + HOLD_SAMPLES - 1
    hs._release_receive_attempt()
    assert hs._pending_receive is not None and not events
    hs.receive_context.input_clock += 1
    hs._release_receive_attempt()
    assert hs._pending_receive is None
    assert [kind for kind, _ in events][:2] == ['X', 'E']
    assert hs._rx_window.A.kind == 'nak' and not io.host


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('level', [1, 2, 4, 5, 6])
def test_wideband_never_contested_or_held(bw, level):
    hs, io = receiver(bw=bw)
    narrow = located(hs, level=4)
    contest(hs, narrow)
    observation = replace(narrow, origin=replace(narrow.origin, bandwidth_hz=int(bw), level=level))
    hs.receive_context.watermark = 0
    assert hs._admit_receive_observation(observation) == 'F'
    assert hs._pending_receive is None


@pytest.mark.parametrize('phase', [0, 1, 127, 511, 4799])
@pytest.mark.parametrize('level', [1, 2, 4])
def test_worst_phase_clean_data_ack_latency_with_real_scanner(phase, level):
    hs, io = receiver()
    context = hs.receive_context = ReceiveContext(tune_epoch=hs._tune_epoch)
    observation = located(hs, start=100000 + phase, level=level)
    end = observation.origin.end_sample
    for at in range(0, end, INPUT_STEP):
        context.feed(np.zeros(min(INPUT_STEP, end - at)), at)
    start = context.input_clock
    read_attempt(hs, [observation])
    while hs._pending_receive is not None and context.input_clock - start < HOLD_SAMPLES:
        context.feed(np.zeros(INPUT_STEP), context.input_clock)
        hs._release_receive_attempt()
    assert hs._pending_receive is None
    assert context.input_clock - start <= HOLD_SAMPLES <= round(.75 * 48000)
    assert hs._rx_window.K is not None and hs._rx_window.A.kind != 'nak'
    assert b''.join(io.host) == b'peer-A'.ljust(43, b'.')


def test_bound_is_pinned_to_scanner_history_batch_and_resampler_geometry():
    from hfmodem.kestrel.rx.cq_stream import FS, HOP, SEARCH_HISTORY, SEARCH_WAIT, RESAMPLE_DELAY
    history = SEARCH_HISTORY * HOP * 48000 // FS
    wait, delay, rounding = SEARCH_WAIT * 48000 // FS, RESAMPLE_DELAY, 4
    lag = history + wait + INPUT_STEP + delay + rounding - (67072 - 16384)
    detector = ReceiveContext().detector
    assert len(detector._filter) == 2 * RESAMPLE_DELAY + 1
    assert HOLD_SAMPLES == ((lag + INPUT_STEP - 1) // INPUT_STEP) * INPUT_STEP == 28800


@pytest.mark.parametrize('route', ['segmented', 'expected'])
def test_broadcast_early_veto_cannot_hide_a_contested_companion(monkeypatch, route):
    from .test_broadcast_not_data import frame, OBSERVED
    hs, io = receiver(high=False)
    tail = frame(OBSERVED[0][0])
    head_body = full_body(b'H' * RX500._PAYLOAD[2], hs.caller,
                          size=RX500._PAYLOAD[2] + 1, field=0x85)
    head = RX500.FrameResult(0, (0,), 9, 1., True, head_body + b'\0\0', level=2,
        ordinal=0, frames_in_burst=2,
        origin=hs.capture_context().origin(100000, 339104, 2, 500))
    tail.ordinal, tail.frames_in_burst = 1, 2
    tail.origin = hs.capture_context().origin(333472, 572576, 2, 500)
    result = RX500.DecodeResult([head, tail], [head, tail])
    contest(hs, tail.observation)
    monkeypatch.setattr(RX500, 'decode_stream', lambda *a, **kw: result)
    if route == 'segmented':
        assert hs._answer_data_over(np.zeros(10))
    else:
        read = phy.decode_level(np.zeros(10), '500', 2, context=hs.capture_context(),
                                receive_observations=True)
        assert read.broadcast and len(read.observations) == 2
        assert hs._take_expected_read(read, np.zeros(10), A.time.monotonic(), True, True, 10)
    assert hs._rx_window.A.kind == 'frame10'
    assert b''.join(io.host) == b'H' * RX500._PAYLOAD[2]
    assert tuple(f.body for f in hs._rx_window.K.frames) == (head_body,)


def test_query_is_deferred_until_held_observations_reach_window(monkeypatch):
    hs, io = receiver()
    observation = located(hs)
    hs.receive_context.watermark = 0
    events = trace_window(monkeypatch, hs)
    assert not read_attempt(hs, [observation], hold=True)
    assert hs._receive_window_event('Q', peer_keying=True) == []
    assert not events and hs._rx_window.W is None
    hs.receive_context.watermark = observation.origin.end_sample + 1
    hs._release_receive_attempt()
    kinds = [kind for kind, _ in events]
    assert kinds[0] == 'F' and kinds[-1] == 'Q'
    assert hs._pending_receive is None


def test_ceiling_can_commit_clear_head_but_expired_tail_is_x(monkeypatch):
    hs, io = receiver(high=False)
    head, tail = located(hs), located(hs, 1, start=301728)
    hs.receive_context.watermark = head.origin.end_sample + 1
    events = trace_window(monkeypatch, hs)
    assert not read_attempt(hs, [head, tail])
    hs._perform_rx_actions(hs._receive_window_event('C'))
    assert [kind for kind, _ in events][:3] == ['F', 'X', 'C']
    assert hs._rx_window.A.kind == 'frame10'
    assert b''.join(io.host) == b'peer-A'.ljust(43, b'.')


def test_ceiling_of_entirely_pending_attempt_does_not_require_more_input(monkeypatch):
    hs, io = receiver()
    observation = located(hs)
    hs.receive_context.watermark = 0
    assert not read_attempt(hs, [observation])
    hs._pending_receive['started'] -= A._ANSWER_HOLD_MAX / A.MK.FS + 1
    monkeypatch.setattr(hs, '_band_busy', lambda: False)
    hs._release_held_answer(0, clock=True)
    assert hs._pending_receive is None and hs._rx_window.W is None
    assert hs._rx_window.A.kind == 'nak' and not io.host


@pytest.mark.parametrize('level', [5, 6])
def test_actual_narrow_high_levels_are_never_contested(level):
    hs, _ = receiver()
    base = located(hs)
    contest(hs, base)
    observation = replace(base, origin=replace(base.origin, level=level))
    hs.receive_context.watermark = 0
    assert hs._admit_receive_observation(observation) == 'F'


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_wideband_admission_never_calls_narrow_decoder(monkeypatch, bw):
    hs, io = receiver(bw=bw)
    observation = located(hs, bw=int(bw))
    contest(hs, replace(observation, origin=replace(observation.origin, bandwidth_hz=500)))
    def narrow(*args, **kwargs):
        raise AssertionError('wideband session entered narrow decoder')
    monkeypatch.setattr(RX500, 'decode_stream', narrow)
    monkeypatch.setattr(A._RX2300, 'decode_over', lambda *a, **kw: SimpleNamespace(
        crc_ok=True, payload=observation.body, quality=None,
        end=observation.origin.end_sample, origin=observation.origin,
        observation=observation))
    read = phy.decode_level(np.zeros(10), bw, 4, context=hs.capture_context(),
                            receive_observations=True)
    hs._level_read = read
    try:
        assert hs._answer_data_over(np.zeros(10))
    finally:
        hs._level_read = None
    assert hs._rx_window.K is not None and hs._rx_window.A.kind != 'nak'
    assert b''.join(io.host) == b'peer-A'.ljust(43, b'.')


def test_fallback_hold_retains_input_and_next_frame_absolute_origin(monkeypatch):
    hs, io = receiver()
    hs._lv_armed, hs._expect_level = True, 5
    hs.receive_context.watermark = 0
    hs.receive_context.input_clock = 100000
    hs._capture_start = 100000
    hs._lv_due = hs._lv_next = 0
    hs._lv_found = True
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    monkeypatch.setattr(hs, '_read_levels', lambda: (5,))
    origins = []
    def decode(buf, bw, level, *args, context, **kwargs):
        offset = 100 if not origins else int(np.flatnonzero(buf == 77)[0])
        origin = context.origin(offset, offset + 1000, 4, 500)
        origins.append(origin)
        return phy.LevelRead(4, broadcast=True,
            observations=(LocatedFrame(0, origin, located(hs).body),))
    monkeypatch.setattr(phy, 'decode_level', decode)
    assert hs._read_expected_level(np.ones(4800))
    assert hs._pending_receive is not None
    held = np.full(4800, 2.)
    held[100] = 77
    hs._capture_start = 104800
    assert hs._read_expected_level(held)
    assert len(origins) == 1
    assert len(hs._lv_buf) == hs._lv_heard == 9600
    hs.receive_context.watermark = 10**9
    hs._release_receive_attempt()
    assert hs._pending_receive is None and len(hs._lv_buf) == 9600
    hs._capture_start = 109600
    hs._lv_due = hs._lv_next = 0
    hs._lv_found = True
    hs._lv_credit = A._EXPECT_CREDIT
    assert not hs._read_expected_level(np.full(4800, 3.))
    assert len(origins) == 2 and origins[-1].training_origin == 104900
    assert hs._lv_start == 100000 and len(hs._lv_buf) == hs._lv_heard == 14400
    assert not io.host and hs._rx_window.K is None


@pytest.mark.parametrize('route', ['level', 'segmented', 'located', 'expected'])
@pytest.mark.parametrize('rejected', ['echo', 'broadcast', 'connect', 'rate'])
def test_clear_non_data_head_with_contested_tail_never_commits(monkeypatch, route, rejected):
    from .test_broadcast_not_data import frame, OBSERVED
    hs, io = receiver(high=False)
    head = frame(OBSERVED[0][0]) if rejected == 'broadcast' else frame(b'H' * 22, marker=0x85)
    head.ordinal, head.frames_in_burst = 0, 2
    head.origin = hs.capture_context().origin(100000, 339104, 2, 500)
    tail = frame(b'T' * 22, marker=0x85)
    tail.ordinal, tail.frames_in_burst = 1, 2
    tail.origin = hs.capture_context().origin(333472, 572576, 2, 500)
    if rejected == 'echo':
        hs._keyed_bodies.add(head.observation.body)
    elif rejected == 'connect':
        head.is_connect = True
    elif rejected == 'rate':
        hs._rx_resend, hs._expect_level = (4, ()), 4
    result = RX500.DecodeResult([head, tail], [f for f in (head, tail) if not f.is_connect])
    contest(hs, tail.observation)
    events = trace_window(monkeypatch, hs)
    monkeypatch.setattr(RX500, 'decode_stream', lambda *a, **kw: result)
    if route in ('level', 'expected'):
        read = phy._decode_base(np.zeros(10), '500', context=hs.capture_context(),
                                receive_observations=True)
        if route == 'level':
            hs._level_read = read
            try:
                assert hs._answer_data_over(np.zeros(10))
            finally:
                hs._level_read = None
        else:
            assert hs._take_expected_read(read, np.zeros(10), A.time.monotonic(), True, True, 10)
    elif route == 'segmented':
        assert hs._answer_data_over(np.zeros(10))
    else:
        over = SimpleNamespace(generation=hs._front_generation, frames=[head, tail],
            result=result, end=10, decision=10, candidate=SimpleNamespace(start=0))
        monkeypatch.setattr(hs, '_read_ident_500', lambda end: False)
        hs._front_heard, hs._base_ready = 10, [over]
        assert hs._take_located_500()
    assert [kind for kind, _ in events if kind in ('F', 'X')] == ['X', 'X']
    assert not io.host and hs._rx_window.K is None and hs._rx_window.R == b''
    assert hs._rx_window.A.kind == 'nak'


@pytest.mark.parametrize('level', [5, 6])
def test_unqualified_fallback_energy_without_header_creates_no_data_debt(monkeypatch, level):
    hs, io = receiver()
    samples = np.r_[np.zeros(48000),
        .2 * np.cos(2 * np.pi * 1500 * np.arange(240000) / A.MK.FS), np.zeros(48000)]
    monkeypatch.setattr(phy._tone, 'decode_burst', lambda *a, **kw: phy._tone.ToneResult(
        level, (), SimpleNamespace(snr_db=-5), lock=0.0, onset=0))
    monkeypatch.setattr(phy, '_tone_burst', lambda *a, **kw: False)
    monkeypatch.setattr(A._locator500.Receiver, 'feed', lambda *a, **kw: [])
    result = RX500.decode_stream(samples, context=hs.capture_context())
    assert result.unread_shape and result.observations
    assert not result.refined_lock and not any(f.crc_ok for f in result.frames)
    read = phy._decode_base(samples, '500', context=hs.capture_context(),
                            receive_observations=True)
    assert read.observations and not read.acquired
    assert not hs.receive_context.ledger.spans
    hs._expect_level = level
    before = (hs._rx_window.A, hs._rx_resend, hs._rx_retry_rates.copy(), hs._expect_level,
              hs._rx_failed, hs._undecoded, hs._reacks, io.keys)
    events = trace_window(monkeypatch, hs)
    assert phy.decode_level(samples, '500', level, under=len(samples),
        context=hs.capture_context(), receive_observations=True) is None
    assert not hs._take_expected_read(read, samples, A.time.monotonic(), True, False, len(samples))
    hs._lv_armed = hs._lv_ended = hs._lv_found = True
    hs._lv_due = hs._lv_next = hs._lv_quiet = 0
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    assert not hs._read_expected_level(samples)
    assert hs._rx_window.W is None and hs._pending_receive is None
    assert (hs._rx_window.A, hs._rx_resend, hs._rx_retry_rates, hs._expect_level,
            hs._rx_failed, hs._undecoded, hs._reacks, io.keys) == before
    assert not events and not io.sent and not io.host


def test_fallback_hold_deferred_nak_then_streaming_data_keeps_axis(monkeypatch):
    hs, io = receiver()
    io.broadcasts = []
    io.cq = io.broadcasts.append
    hs._lv_armed, hs._expect_level = True, 5
    io.receiving = True
    context = hs.receive_context
    context.input_clock = 100000
    context.feed = lambda samples, start, **kw: setattr(
        context, 'input_clock', start + len(samples))
    context.advance = lambda *a: None
    seen = []
    body = located(hs).body

    def feed(rx, samples, **kwargs):
        start = rx.raw.samples
        rx.raw.feed(samples)
        seen.append((rx, rx.context.sample_start + start, np.array(samples)))
        hits = np.flatnonzero(samples == 77)
        if not len(hits):
            return []
        at = start + int(hits[0])
        frame = RX500.FrameResult(at, (0, 256), 9, 1., True, body + b'\0\0',
            level=4, origin=rx.context.origin(at, at + 1000, 4, 500))
        return [A._locator500.Over(A._locator500.Candidate(4, at, 1., at, at),
            generation=rx.generation, frames=[frame], status='closed',
            end=at + 1000, decision=at + 1000)]

    monkeypatch.setattr(A._locator500.Receiver, 'feed', feed)
    monkeypatch.setattr(hs, '_stream_responder_control', lambda *a: False)
    monkeypatch.setattr(hs, '_read_ident_500', lambda *a: False)
    hs.on_rx_stream(np.zeros(1024), sample_start=100000)
    previous = hs._base_rx
    observation = located(hs)
    contest(hs, observation)
    context.watermark = 0
    read = phy.LevelRead(4, observations=(observation,))
    assert hs._take_expected_read(read, np.zeros(10), A.time.monotonic(), True, True, 10)
    assert hs._pending_receive is not None
    events = trace_window(monkeypatch, hs)
    context.watermark = 10**9
    hs.on_rx_stream(np.full(1024, 7.), sample_start=101024)
    assert hs._pending_receive is None and hs._rx_window.A.kind == 'nak'
    assert not hs._rx_window.A.sent and not io.sent
    assert hs._base_rx is not previous
    assert hs._base_rx.context.decoder_epoch == hs._decoder_epoch
    assert seen[-1][1] == 101024
    np.testing.assert_array_equal(seen[-1][2], np.full(1024, 7.))
    assert hs._front_start == 101024 and hs._front_heard == 1024
    admitted = []
    gate = hs._admit_receive_observation
    def admit(observation, **kwargs):
        decision = gate(observation, **kwargs)
        admitted.append((observation.origin, decision))
        return decision
    monkeypatch.setattr(hs, '_admit_receive_observation', admit)
    block = np.zeros(2048)
    block[100] = 77
    hs.on_rx_stream(block, sample_start=102048)
    assert seen[-2][1] == 102048 and seen[-1][1] == 103072
    assert any(o.training_origin == 102148 and d == 'F' for o, d in admitted)
    assert all(d != 'DISCARD' for _, d in admitted)
    assert [k for k, _ in events if k in ('F', 'X')] == ['X', 'F']
    assert b''.join(io.host) == b'peer-A'.ljust(43, b'.')
    assert hs._rx_window.K is not None and hs._rx_window.A.kind != 'nak'


def test_ceiling_resolves_companion_after_already_ended_complete_held_head(monkeypatch):
    hs, io = receiver()
    head, tail = located(hs), located(hs, 1, start=301728)
    hs.receive_context.watermark = head.origin.end_sample + 1
    assert read_attempt(hs, [head], hold=True)
    hs._perform_rx_actions(hs._receive_window_event('E', single=True, peer_keying=True))
    assert hs._rx_window.state == 'ENDED-COMPLETE' and hs._rx_window.W.held
    assert hs._rx_window.W.frames[0].body == head.body
    assert not io.host and hs._rx_window.K is None
    events = trace_window(monkeypatch, hs)
    assert not read_attempt(hs, [tail])
    assert hs._pending_receive is not None
    assert hs._rx_window.state == 'ENDED-COMPLETE' and hs._rx_window.W.held
    hs._perform_rx_actions(hs._receive_window_event('C'))
    assert [k for k, _ in events] == ['X', 'C']
    assert hs._pending_receive is None and hs._rx_window.W is None
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.K is None
    assert not io.host and not hs._rx_window.R


@pytest.mark.parametrize('route', ['deadline', 'absent'])
def test_stale_pending_head_owes_no_nak_after_a_decoder_reset(monkeypatch, route):
    hs, io = receiver(high=True)
    hs._expect_level = 6
    head = located(hs, level=6, bw=500)
    hs._lv_pending = phy.LevelRead(6, observations=(head,), end=1000)
    hs._lv_pending_deadline = 0
    hs._invalidate_receive_decoders()
    assert not hs.origin_is_current(head.origin)
    events = trace_window(monkeypatch, hs)
    before = hs._rx_read, hs._rx_window.A, hs._rx_resend
    if route == 'deadline':
        hs._lv_armed = True
        monkeypatch.setattr(hs, '_track_emission', lambda: None)
        hs._read_expected_level(np.zeros(4800))
    else:
        assert not hs._take_expected_read(None, np.zeros(10), A.time.monotonic(),
                                          True, False, 10)
    assert hs._lv_pending is None and hs._lv_pending_deadline is None
    assert not events and not io.sent
    assert (hs._rx_read, hs._rx_window.A, hs._rx_resend) == before
