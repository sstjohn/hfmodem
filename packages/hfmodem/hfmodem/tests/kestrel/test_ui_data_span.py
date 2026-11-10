# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P-DATA-SPAN real-route replays and explicitly unaccepted R-UI residuals."""
from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core.wav import source
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host.audio_runtime import KissObserver
from hfmodem.kestrel.host.station_runtime import StationTransport
from hfmodem.kestrel.rx import broadcast as UI, bw500_locator as locator, varahf500 as RX
from hfmodem.kestrel.rx.beacon import HEADER as LEVEL2_HEADER
from hfmodem.kestrel.rx.receive_context import ReceiveContext, HOLD_SAMPLES, INPUT_STEP
from hfmodem.kestrel.rx.ui_spans import CLEAR, CONTESTED
from hfmodem.kestrel.tx import varahf500_tx as TX
from hfmodem.kestrel.tx.ui import compact, packetize, packets
from hfmodem.kestrel.vara import vara_arq as A
from hfmodem.kestrel.vara.vara_mfsk import synth_tones
from hfmodem.tests.evidence import CORPUS, require_input
from .test_receive_context_runtime import owner
from .test_rxwindow_adapter import station
from .test_ui_receive import PEER, NATIVE, peer_captures, frame, normalized, crc
from .test_ui_window_admission import trace_window

ROUTES = ('streaming', 'segmented', 'fallback')
RECORDED_ROWS = ('len400', 'q-core23', 'q-core50', 'len809', 'len812', 'base')


@dataclass
class Replay:
    audio: np.ndarray
    level: int
    encoded: list


def render(level, encoded):
    header = synth_tones(LEVEL2_HEADER if level == 2 else UI.HEADERS[level], amplitude=.2)
    parts = [header, np.zeros(67072 - len(header))]
    for i, raw in enumerate(encoded):
        if i:
            parts.append(np.zeros(1536))
        body = (TX.synth_burst(raw, ncol=403) if level == 4 else
                TX.synth_l3_burst(raw, level=level))
        parts.append(body * (.2 / max(np.sqrt(2 * np.mean(body**2)), 1e-30)))
    return np.r_[np.concatenate(parts), np.zeros(48000)]


@lru_cache(maxsize=10)
def recorded_replay(name):
    case = next(r for r in PEER if r['case'] == name)
    path = require_input(CORPUS / 'kestrel-kiss-1031' / case['session'] / f'{name}.wav')
    rate, audio = wavfile.read(source(path))
    assert rate == 48000 and audio.ndim == 1
    if audio.dtype == np.int16:
        audio = audio.astype(float) / 32768
    return Replay(np.r_[audio, np.zeros(48000)], case['level'],
                  [bytes.fromhex(p) for p in case['packets']])


@pytest.fixture(scope='module', params=RECORDED_ROWS)
def recorded_case(request):
    if not PEER:
        pytest.skip('recording unavailable: peer-packets.json')
    return request.param, recorded_replay(request.param)


@pytest.fixture(scope='module')
def native_len400():
    case = next(r for r in NATIVE if r['case'] == 'n-len400')
    encoded = [bytes.fromhex(p) for p in case['packets']]
    return Replay(render(case['level'], encoded), case['level'], encoded)


def connected():
    hs, io = station(bw='500', role='initiator')
    io.broadcasts = []
    io.cq = io.broadcasts.append
    hs.receive_context = ReceiveContext(tune_epoch=hs._tune_epoch)
    hs.cq_detector = hs.receive_context.detector
    return hs, io


def feed_to(hs, audio, end, delivered, *, release=True):
    context = hs.receive_context
    start = context.input_clock
    for at in range(start, end, INPUT_STEP):
        context.feed(audio[at:min(at + INPUT_STEP, end)], at, sink=delivered.append)
        if release:
            pending = hs._pending_receive
            hs._release_receive_attempt()
            if pending is not None and hs._pending_receive is None:
                assert context.watermark > max(o.origin.end_sample for o in pending['observations'])
                assert context.input_clock <= pending['deadline']


def accepted_header(hs, replay, delivered):
    context = hs.receive_context
    for at in range(0, len(replay.audio), INPUT_STEP):
        context.feed(replay.audio[at:at + INPUT_STEP], at, sink=delivered.append)
        spans = [s for s in context.ledger.spans if s.level == replay.level]
        if spans:
            span = spans[0]
            assert not delivered and context.detector.turbo_attempts == 0
            assert span.possible_packet_count == 18
            return span
    pytest.fail('real scanner did not accept the UI header')


def packet_audio(replay, span, index):
    training = span.first_training_origin + index * span.period
    lo = max(0, training - span.tolerance - 12000)
    hi = min(len(replay.audio), training + UI.BODY_SAMPLES[replay.level] + 12000)
    return lo, replay.audio[lo:hi]


def take_route(hs, route, audio, start, *, expected=6):
    """Only the named decoder owns this attempt; retries use a fresh capture."""
    context = hs.capture_context(start)
    if route == 'streaming':
        receiver = locator.Receiver(generation=hs._front_generation, context=context)
        overs = []
        for at in range(0, len(audio), INPUT_STEP):
            overs.extend(receiver.feed(audio[at:at + INPUT_STEP], may_pair=False))
        overs.extend(receiver.feed(np.empty(0), may_pair=False, eof=True))
        assert len(overs) == 1
        over, = overs
        observations = over.result.observations
        hs._base_rx, hs._front_raw = receiver, receiver.raw
        hs._front_heard, hs._front_start = len(audio), start
        hs._base_ready = [over]
        hs._take_located_500()
    elif route == 'segmented':
        captured = []
        gate = hs._gate_receive_attempt
        def capture(observations, resume):
            captured.extend(observations)
            return gate(observations, resume)
        hs._gate_receive_attempt = capture
        hs._decode_context = context
        try:
            hs._answer_data_over(audio)
        finally:
            hs._decode_context = None
            hs._gate_receive_attempt = gate
        observations = tuple(captured)
    else:
        # Above-base callers must really try their expected family before the
        # BW500 base fallback; the silent lead satisfies that family's horizon.
        pad = max(0, 1000000 - len(audio))
        buf = np.pad(audio, (pad, 0))
        read = phy.decode_level(buf, '500', expected, fresh=4800, under=4800,
            expect_pair=False, context=hs.capture_context(start - pad),
            receive_observations=True)
        assert read is not None and read.level <= 4
        observations = read.observations
        hs._take_expected_read(read, buf, A.time.monotonic(), True, True, len(buf))
    assert observations and all(o.origin.bandwidth_hz == 500 for o in observations)
    return observations


def assert_excluded(hs, io, events):
    assert not io.host and not io.warnings
    assert not hs._rx_window.R and hs._rx_window.K is None and hs._rx_window.T is None
    assert hs._pending_receive is None and hs._rx_window.W is None
    assert hs._rx_window.A.kind == 'nak'
    assert any(kind == 'X' for kind, _ in events)
    assert not any(kind == 'F' for kind, _ in events)


def peer_retry(hs, io, level, *, start=None):
    size = RX._PAYLOAD[level] + 1
    payload = b'peer retry'[:size - 1].ljust(size - 1, b'.')
    raw = (TX.build_frame(payload, 0x85) if level == 4 else
           TX.build_l3_frame(payload, 0x8D if level == 1 else 0x85, level))
    body = (TX.synth_burst(raw, onset=12000, ncol=403) if level == 4 else
            TX.synth_l3_burst(raw, onset=12000, level=level))
    audio = np.pad(body, (0, 48000))
    start = hs.receive_context.input_clock if start is None else start
    hs.receive_context.feed(audio, start)
    hs.receive_context.feed(np.zeros(HOLD_SAMPLES), start + len(audio))
    hs._decode_context = hs.capture_context(start)
    try:
        assert hs._answer_data_over(audio)
    finally:
        hs._decode_context = None
    hs._release_receive_attempt()
    assert b''.join(io.host) == payload
    assert hs._rx_window.R == payload and hs._rx_window.K is not None
    assert hs._rx_window.A.kind != 'nak'


def replay_attempt(monkeypatch, replay, route, index, *, schedule='data-first', expected=6):
    hs, io = connected()
    delivered = []
    span = accepted_header(hs, replay, delivered)
    start, audio = packet_audio(replay, span, index)
    end = (len(replay.audio) if schedule == 'ui-first' else
           span.first_training_origin + index * span.period + UI.BODY_SAMPLES[replay.level])
    feed_to(hs, replay.audio, end, delivered)
    events = trace_window(monkeypatch, hs)
    before = hs._peer_over, hs._read_streak, hs._peer_turn_proven, hs._close_low
    observations = take_route(hs, route, audio, start, expected=expected)
    clean = [o for o in observations if o.body is not None]
    assert [o.body for o in clean] == [replay.encoded[index][:-2]]
    assert all(abs(o.origin.training_origin - (span.first_training_origin + index * span.period))
               <= span.tolerance for o in clean)
    assert all(hs.receive_context.ledger.classify(o.origin) == CONTESTED for o in observations)
    if schedule == 'data-first':
        assert hs._pending_receive is not None and not events and not io.host
    else:
        assert_excluded(hs, io, events)
    assert (hs._peer_over, hs._read_streak, hs._peer_turn_proven, hs._close_low) == before
    feed_to(hs, replay.audio, len(replay.audio), delivered)
    hs._release_receive_attempt()
    assert_excluded(hs, io, events)
    assert (hs._peer_over, hs._read_streak, hs._peer_turn_proven, hs._close_low) == before
    return hs, io, span, observations, delivered


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('schedule', ['data-first', 'ui-first'])
def test_recorded_matrix_real_routes_and_delayed_results(monkeypatch, recorded_case, route, schedule):
    name, replay = recorded_case
    index = 1 if name == 'len400' else len(replay.encoded) - 1
    hs, io, span, observations, delivered = replay_attempt(
        monkeypatch, replay, route, index, schedule=schedule)
    assert len(delivered) == 1
    assert delivered[0].ax25() == UI.parse_packets(replay.encoded, level=replay.level).ax25()
    if name in ('len809', 'len812'):
        current = next(s for s in hs.receive_context.ledger.spans if s.id == span.id)
        assert current.ambiguous_2_or_18 and current.possible_packet_count == 18
    peer_retry(hs, io, replay.level)


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('damage', ['middle', 'fade', 'early-final'])
@pytest.mark.parametrize('source_kind', ['recorded', 'native'])
def test_len400_failed_assembly_then_uncontested_retry(monkeypatch, native_len400, route, damage,
                                                       source_kind):
    if source_kind == 'recorded' and not PEER:
        pytest.skip('recording unavailable: peer-packets.json')
    replay = recorded_replay('len400') if source_kind == 'recorded' else native_len400
    probe, _ = connected()
    span = accepted_header(probe, replay, [])
    encoded = replay.encoded.copy()
    audio = replay.audio.copy()
    if damage == 'early-final':
        encoded[1] = crc(bytes([encoded[1][0] & 0x7F]) + encoded[1][1:-2])
        assert UI.parse_packets(encoded[:2], level=4) is None
        replacement = TX.synth_burst(encoded[1], ncol=403)
        replacement *= .2 / np.sqrt(2 * np.mean(replacement**2))
        at = span.first_training_origin + span.period
        audio[at:at + len(replacement)] = replacement
    else:
        first, last = (3, 4) if damage == 'middle' else (2, 7)
        audio[span.first_training_origin + first * span.period:
              span.first_training_origin + last * span.period] = 0
    damaged = Replay(audio, 4, encoded)
    hs, io, _, _, delivered = replay_attempt(monkeypatch, damaged, route, 7)
    assert delivered == []
    peer_retry(hs, io, 4)


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('name', ['q-core23', 'q-core50'])
@peer_captures
def test_index_origins_survive_lost_first_packet(monkeypatch, name, route):
    replay = recorded_replay(name)
    hs, _ = connected()
    span = accepted_header(hs, replay, [])
    audio = replay.audio.copy()
    lo = span.first_training_origin
    hi = lo + UI.BODY_SAMPLES[replay.level]
    audio[lo:hi] = 0
    hs, io, span, observations, delivered = replay_attempt(
        monkeypatch, replace(replay, audio=audio), route, 1)
    assert not delivered
    current = next(s for s in hs.receive_context.ledger.spans if s.id == span.id)
    assert current.possible_packet_count == 18
    assert observations[0].origin.level == replay.level
    peer_retry(hs, io, replay.level)


@pytest.mark.parametrize('level', [1, 2, 4])
@pytest.mark.parametrize('route', ROUTES)
def test_native_eighteen_packet_groups(monkeypatch, level, route):
    size = {1: 9, 2: 22, 4: 43}[level]
    for n in range(18 * (size - 1) - 20, 18 * (size - 1)):
        message = frame(b'\x80' + bytes((i % 251 for i in range(n))))
        core, mode = compact(message)
        if len(core) > 18 * (size - 1):
            break
        actual, encoded = packets(core, mode, level=level)
        if len(encoded) == 18:
            break
    assert actual == level and len(encoded) == 18
    replay = Replay(render(level, encoded), level, encoded)
    hs, io, _, _, delivered = replay_attempt(monkeypatch, replay, route, 17)
    assert [r.ax25() for r in delivered] == [normalized(message).ax25()]
    peer_retry(hs, io, level)


@pytest.mark.parametrize('route', ROUTES)
def test_valid_ambiguous_two_packet_collision_keeps_later_slots(monkeypatch, route):
    short = frame(b'\x80' + b'x' * 58 + b'\x00')
    long = frame(b'\x80' + b'x' * 738 + b'\x08')
    _, two = packetize(short, level=4)
    _, eighteen = packetize(long, level=4)
    assert len(two) == 2 and len(eighteen) == 18 and two[0] == eighteen[0]
    assert two[0][1] == 0x50
    encoded = [*two, *eighteen[2:]]
    replay = Replay(render(4, encoded), 4, encoded)
    hs, io, span, _, delivered = replay_attempt(monkeypatch, replay, route, 12)
    assert [r.ax25() for r in delivered] == [normalized(short).ax25()]
    current = next(s for s in hs.receive_context.ledger.spans if s.id == span.id)
    assert current.resolution == 'group' and current.possible_packet_count == 18
    peer_retry(hs, io, 4)


@pytest.mark.parametrize('expected', [5, 6])
def test_high_speed_expectation_cannot_bypass_narrow_span(monkeypatch, native_len400, expected):
    hs, io, _, observations, _ = replay_attempt(monkeypatch, native_len400, 'fallback', 4,
                                                expected=expected)
    assert observations[0].origin.level == 4
    peer_retry(hs, io, 4)


@pytest.mark.parametrize('level', [5, 6])
def test_actual_high_speed_audio_stays_clear(monkeypatch, native_len400, level):
    from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
    hs, io = connected()
    span = accepted_header(hs, native_len400, [])
    size = phy.body_size('500', phy.level_key('500', level))
    payload = b'H' * (size - 1)
    body = phy.vara_body(payload, hs.caller, tail=0x85, body_len=size)
    audio = np.r_[np.zeros(12000), phy.render_level((body,), '500', level, VB6Rnd(43)),
                  np.zeros(48000)]
    start = span.first_training_origin - 12000
    read = phy.decode_level(audio, '500', level, expect_pair=False,
                            context=hs.capture_context(start), receive_observations=True)
    assert read is not None and read.level == level and read.frames == (body,)
    assert all(hs.receive_context.ledger.classify(o.origin) == CLEAR and
               hs._admit_receive_observation(o) == 'F' for o in read.observations)
    events = trace_window(monkeypatch, hs)
    hs._level_read = read
    try:
        assert hs._answer_data_over(audio)
    finally:
        hs._level_read = None
    assert any(kind == 'F' for kind, _ in events) and not hs._pending_receive
    assert b''.join(io.host) == payload


@peer_captures
@pytest.mark.parametrize('route', ROUTES)
def test_base_closure_followed_by_new_link(monkeypatch, route):
    replay = recorded_replay('base')
    hs, io, span, _, delivered = replay_attempt(monkeypatch, replay, route, 1)
    current = next(s for s in hs.receive_context.ledger.spans if s.id == span.id)
    assert current.resolution == 'group' and current.possible_packet_count == 2
    assert not current.ambiguous_2_or_18 and len(delivered) == 1
    hs._perform_rx_actions(hs._receive_window_event('T', new_session=True))
    following, next_io = station(bw='500', role='initiator')
    following.receive_context = hs.receive_context
    following.cq_detector = hs.receive_context.detector
    peer_retry(following, next_io, 1,
               start=current.first_training_origin + 2 * current.period - 12000)
    assert not io.host


@pytest.mark.parametrize('route', ROUTES)
@peer_captures
@pytest.mark.parametrize('name,index', [('len400', 1), ('i-object-kill', 0),
                                      ('q-core23', 0), ('via1-used', 0)])
def test_R_UI_header_erased_residual_pin_not_acceptance(monkeypatch, name, index, route):
    replay = recorded_replay(name)
    probe, _ = connected()
    span = accepted_header(probe, replay, [])
    start, audio = packet_audio(replay, span, index)
    hs, io = connected()
    # A decoder can finish before the scanner covers its frame even when the
    # header was erased; absence of a span is not a negative acquisition proof.
    hs.receive_context.gap(start)
    hs.receive_context.feed(audio[:1], start)
    assert not hs.receive_context.ledger.spans
    events = trace_window(monkeypatch, hs)
    observations = take_route(hs, route, audio, start)
    assert [o.body for o in observations] == [replay.encoded[index][:-2]]
    assert hs._pending_receive is not None
    hs.receive_context.feed(audio[1:], start + 1)
    hs.receive_context.feed(np.zeros(HOLD_SAMPLES), start + len(audio))
    assert not hs.receive_context.ledger.spans
    assert all(hs.receive_context.ledger.classify(o.origin) == CLEAR for o in observations)
    assert all(hs._admit_receive_observation(o) == 'F' for o in observations)
    hs._release_receive_attempt()
    assert any(kind == 'F' for kind, _ in events)
    assert not any(kind == 'X' for kind, _ in events)
    expected = phy.vara_payload(replay.encoded[index][:-2], caller=hs.caller,
                                body_len=len(replay.encoded[index]) - 2)
    assert b''.join(io.host) == expected and expected


def test_station_transport_header_fade_nak_tx_reset_and_remaining_slots(monkeypatch, native_len400):
    backend = SimpleNamespace(station=SimpleNamespace(
        arbiter=SimpleNamespace(receive_boundary=(0, 0)),
        audio=SimpleNamespace(samples=0)), playback_ended_at=None)
    emissions = []
    def emit(samples, bandwidth, **kwargs):
        emissions.append((len(samples), bandwidth))
        backend.station.audio.samples += len(samples)
        return True
    backend.emit = emit
    transport = StationTransport(backend)
    hs, io = connected()
    context = hs.receive_context
    transport.bind_receive_context(context)
    transport.bind_station(hs)
    transport.cq_sink = io.cq
    io.key, io.tx, io.tx_metadata = transport.key, transport.tx, transport.tx_metadata
    io.tx_went_out = transport.tx_went_out
    events = trace_window(monkeypatch, hs)
    audio = native_len400.audio.copy()
    fade_end = 67072 + 3 * UI.PERIOD[4]
    audio[67072:fade_end] = 0

    def capture(start, end):
        for at in range(start, end, INPUT_STEP):
            chunk = audio[at:min(at + INPUT_STEP, end)]
            backend.station.audio.samples = at + len(chunk)
            transport.feed(at, chunk)
            while not transport.frames.empty() or transport._bursts:
                burst = transport.next_rx_burst(.01, hs)
                if burst is not None:
                    hs.on_rx_audio(burst, sample_start=transport.burst_start)

    capture(0, 96000)
    span, = context.ledger.spans
    assert span.level == 4 and not emissions and not io.host
    capture(96000, fade_end)
    before = hs._decoder_epoch, transport._rx_generation
    hs._perform_rx_actions(hs._receive_window_event('X', level=4))
    hs._perform_rx_actions(hs._receive_window_event('E', single=True))
    assert emissions and emissions[-1][1] == '500'
    assert hs._decoder_epoch > before[0] and transport._rx_generation > before[1]
    assert any(s.id == span.id for s in context.ledger.spans)
    assert hs._pending_receive is None
    resume = max(fade_end, transport._skip_before)
    capture(resume, len(audio))
    hs._release_receive_attempt(ceiling=True)
    later = [kwargs for kind, kwargs in events[2:] if kind in ('F', 'X')]
    assert later and all(kwargs.get('frame') is None for kwargs in later)
    assert not io.host and not hs._rx_window.R and hs._rx_window.K is None
    assert any(s.id == span.id for s in context.ledger.spans)
    assert hs.receive_context is transport.receive_context is context


def test_real_audio_hostless_attach_detach_bandwidth_and_overlap_deliver_once():
    runtime, transport, delivered = owner()
    runtime.step(0)
    context = runtime.receive_context
    detector, ledger = context.detector, context.ledger
    message = frame(b'owner lifecycle')
    level, encoded = packetize(message)
    audio = render(level, encoded)
    at = 0

    def capture(end):
        nonlocal at
        while at < end:
            stop = min(at + INPUT_STEP, end)
            transport.chunks.append((at, audio[at:stop]))
            runtime.step(0)
            at = stop
            assert runtime.modem.station.receive_context is context
            assert context.detector is detector and context.ledger is ledger
            assert len(ledger.spans) <= 1

    capture(96000)
    assert len(ledger.spans) == 1 and not delivered
    attached = runtime.modem_factory()
    attached.start(KissObserver(runtime))
    runtime.step(0)
    capture(min(len(audio), 160000))
    attached.set_bandwidth('500')
    runtime.step(0)
    attached.stop()
    runtime.step(0)
    capture(len(audio))
    assert delivered == [(message.command, normalized(message).encode())]
    for _ in range(3):
        transport.chunks.append((0, audio[:INPUT_STEP]))
        runtime.step(0)
        assert delivered == [(message.command, normalized(message).encode())]
        assert len(ledger.spans) == 1
    span, = ledger.spans
    context.gap(span.deadline + 48000)
    context.feed(np.zeros(INPUT_STEP), context.input_clock)
    context.advance(dict.fromkeys(ROUTES, context.input_clock))
    assert not ledger.spans


@pytest.mark.parametrize('route', ROUTES)
def test_native_intact_data_shaped_middles(monkeypatch, native_len400, route):
    hs, io, _, observations, delivered = replay_attempt(monkeypatch, native_len400, route, 4)
    assert observations[0].body[-1] & 1
    assert len(delivered) == 1
    peer_retry(hs, io, 4)


def test_pending_stream_result_cannot_be_bypassed_by_another_route(monkeypatch, native_len400):
    hs, io = connected()
    delivered = []
    span = accepted_header(hs, native_len400, delivered)
    start, audio = packet_audio(native_len400, span, 4)
    end = span.first_training_origin + 4 * span.period + UI.BODY_SAMPLES[4]
    feed_to(hs, native_len400.audio, end, delivered)
    events = trace_window(monkeypatch, hs)
    for route in ROUTES:
        observations = take_route(hs, route, audio, start)
        assert [o.body for o in observations] == [native_len400.encoded[4][:-2]]
        assert hs._pending_receive is not None and not events and not io.host
    feed_to(hs, native_len400.audio, len(native_len400.audio), delivered)
    assert_excluded(hs, io, events)
    assert sum(kind == 'X' for kind, _ in events) == 1
    assert len(delivered) == 1
