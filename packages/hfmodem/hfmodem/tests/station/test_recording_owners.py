# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Operating owners share policy selection, clocks and serial storage."""
import importlib
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import config, wav
from hfmodem.core.audio import Block
from hfmodem.core import recording
from hfmodem.station import capture, process
from hfmodem.tests.station.recording_support import (
    TIMEOUT, audio_files, bounded, flush, index, pcm, readable, samples,
)

OWNERS = ('station', 'kestrel', 'vara', 'pactor', 'ardop')
RATE = 8


class Stream:
    def __init__(self, **kwargs):
        self.callback = kwargs['callback']
        self.latency = 0
        self.blocksize = kwargs.get('blocksize', 128)
        self.rate = kwargs['samplerate']
        self.position = 0
        self.duplex = isinstance(kwargs.get('device'), tuple)
        self.output = []

    def feed(self, block):
        at = self.position / self.rate
        times = NS(currentTime=at, inputBufferAdcTime=at, outputBufferDacTime=at)
        status = NS(input_overflow=False, input_underflow=False,
                    output_underflow=False, priming_output=False)
        if self.duplex:
            out = np.zeros((len(block), 1), np.float32)
            self.callback(np.asarray(block)[:, None], out, len(block), times, status)
            self.output.append(out[:, 0].copy())
        else:
            self.callback(np.asarray(block)[:, None], len(block), times, status)
        self.position += len(block)

    def start(self):
        self.feed(pcm(0, 1))

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def owner_factory(tools, monkeypatch, tmp_path):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(capture, 'CARD_RATE_HZ', RATE)
    monkeypatch.setattr(wav, '_flac', lambda *a: (_ for _ in ()).throw(OSError('encoder unavailable')))
    monkeypatch.setattr(wav, 'store_later', lambda *a: None)
    monkeypatch.setattr(wav, 'store_detached', lambda *a: None)
    monkeypatch.setitem(sys.modules, 'sounddevice', NS(
        InputStream=Stream, Stream=Stream, query_devices=lambda *a: {'max_output_channels': 1}))
    kestrel = importlib.import_module('kestrel_connect')
    pactor = importlib.import_module('hfmodem.shrike.onair')
    ardop = importlib.import_module('hfmodem.besra.radio')
    monkeypatch.setattr(kestrel, 'FS', RATE)
    sessions, owners = [], []
    original = recording.RecordingSession

    class Session(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sessions.append(self)

    monkeypatch.setattr(recording, 'RecordingSession', Session)
    for module in (capture, kestrel, pactor, ardop):
        if getattr(module, 'RecordingSession', None) is original:
            monkeypatch.setattr(module, 'RecordingSession', Session)

    def make(name, policy):
        monkeypatch.setenv('HFMODEM_RECORDING', policy)
        root = tmp_path / name
        if name == 'station':
            owner = capture.StationCapture(root, 'synthetic')
            feed = lambda first, block: owner.push(Block(first / RATE, first, block))
        elif name == 'kestrel':
            owner = kestrel.Recorder(root / 'receive.wav')
            feed = lambda first, block: owner.write(block)
        elif name == 'vara':
            owner = kestrel.AudioVaraIO('synthetic-out', 'synthetic-in', record=root / 'receive.wav')
            feed = lambda first, block: owner._stream.feed(block)
        elif name == 'pactor':
            owner = pactor._LiveInput('synthetic', 'synthetic-out', fs=RATE,
                                      blocksize=1, record=root / 'stream.wav')
            feed = lambda first, block: owner._stream.feed(block)
        else:
            monkeypatch.setattr(ardop, 'FS_RADIO', RATE)
            modem = NS(expected_session=None, rx_epoch=0, status=lambda *a: None)
            owner = ardop.RadioLink(modem, in_device='synthetic', record=root / 'stream.wav')
            feed = lambda first, block: owner._capture(block)
        owners.append(owner)
        return owner, feed, root, sessions

    make.sessions = sessions
    yield make
    for owner in reversed(owners):
        if isinstance(owner, kestrel.AudioVaraIO):
            if owner.rec is not None:
                bounded(owner.rec.close)
            owner._stream.close()
        else:
            bounded(owner.close)
    for rec in reversed(sessions):
        bounded(rec.close)


@pytest.mark.parametrize('name', OWNERS)
@pytest.mark.parametrize('policy', recording.POLICIES)
def test_owner_idle_pcm_obeys_policy(owner_factory, name, policy):
    owner, feed, root, sessions = owner_factory(name, policy)
    feed(0, pcm(0, RATE))
    if hasattr(owner, 'drain'):
        owner.drain()
    if name == 'ardop':
        owner._drain()
    if name == 'vara':
        bounded(owner.rec.close)
    else:
        bounded(owner.close)
    if policy in ('off', 'contacts'):
        assert audio_files(root) == [], f'{name} persisted idle audio under {policy}'
        assert not [path for path in root.rglob('*') if path.is_file() and
                    path.name.endswith(('.wav.part', '.flac.part', '.pcm', '.raw'))]
    else:
        assert audio_files(root), f'{name} did not retain continuous audio'
    assert len(sessions) == 1, f'{name} must use one RecordingSession'
    rec = sessions[0]
    bounded(rec.close)
    assert rec.policy == policy
    assert rec.index['source'] == 'inherited'
    assert json.loads((rec.directory / 'recording-policy.json').read_text()) == {
        'policy': policy, 'source': 'inherited'}
    assert any(lane.position >= RATE for lane in rec.lanes.values())
    if policy != 'off':
        assert index(rec.directory)['policy'] == policy
    else:
        assert all(lane.ring is None and lane.writer is None for lane in rec.lanes.values())


@pytest.mark.parametrize('name', ('vara', 'pactor'))
def test_standalone_acquisition_controls_survive_delayed_delivery_and_replacement(
        owner_factory, tools, monkeypatch, tmp_path, name):
    station = importlib.import_module('consolelib.station')
    now = [1015.]
    deadline = time.perf_counter() + TIMEOUT
    def clock():
        assert time.perf_counter() < deadline, 'standalone control clock exceeded its timeout'
        return now[0]
    monkeypatch.setattr(recording.time, 'monotonic', clock)
    def start(stream):
        times = NS(inputBufferAdcTime=100., currentTime=115., outputBufferDacTime=115.)
        status = NS(input_overflow=False, input_underflow=False, output_underflow=False,
                    priming_output=False)
        block = pcm(0, RATE)[:, None]
        stream.callback(block, np.zeros_like(block), RATE, times, status)
    monkeypatch.setattr(Stream, 'start', start)
    plan = dict(env={'HFMODEM_RECORDING': 'contacts'})
    _, rd, sender = station.attached_control(plan)
    monkeypatch.setenv('HFMODEM_RECORDING_CONTROL_OUT_FDS', plan['env']['HFMODEM_RECORDING_CONTROL_OUT_FDS'])
    owner, _, _, opened = owner_factory(name, 'contacts')
    parent = opened[0]
    children, descriptors = [], [rd]
    status = NS(input_overflow=False, input_underflow=False, output_underflow=False,
                priming_output=False)

    def capture(adc, current):
        times = NS(inputBufferAdcTime=adc, currentTime=current, outputBufferDacTime=current)
        block = pcm(0, RATE)[:, None]
        if name == 'vara':
            owner._stream.callback(block, RATE, times, status)
        else:
            owner._stream.callback(block, np.zeros_like(block), RATE, times, status)
        flush(parent)

    def receiver(fd, origin, clock):
        child = recording.RecordingSession('contacts', tmp_path / clock, attached_children=False)
        children.append(child)
        lane = child.stream('witness', rate=RATE, clock=clock)
        lane.row.update(monotonic_origin=origin, measured_rate=RATE)
        return child, recording.ChildControlReceiver(child, fd)

    try:
        child, control = receiver(rd, 1000., 'witness-1')
        if name == 'vara':
            capture(100., 115.)
        parent.begin('delayed', {parent.lanes['rx'].clock: 16})
        assert readable(rd)
        control.drain()
        flush(child)
        assert child.index['contacts']['delayed']['positions'] == {'witness-1': 16}
        assert parent.lanes['rx'].row['clock_uncertainty_s'] is None
        _, rd2, _ = station.attached_control(plan, sender=sender)
        descriptors.append(rd2)
        replacement, control2 = receiver(rd2, 1010., 'witness-2')
        assert readable(rd2)
        control2.drain()
        flush(replacement)
        assert replacement.index['contacts']['delayed']['positions'] == {'witness-2': -64}
        now[0] = 1020.
        capture(101., 120.)
        parent.end('delayed', {parent.lanes['rx'].clock: 96})
        assert readable(rd2)
        control2.drain()
        flush(replacement)
        assert replacement.index['contacts']['delayed']['end_positions'] == {'witness-2': 16}
    finally:
        bounded(parent.close)
        station.close_attached_controls(plan)
        sender.close(timeout=TIMEOUT)
        for child in children:
            bounded(child.close)
        for fd in descriptors:
            os.close(fd)


@pytest.mark.parametrize('name', OWNERS)
@pytest.mark.parametrize('policy', ('contacts', 'everything'))
def test_owner_contact_spans_rollover(owner_factory, monkeypatch, name, policy):
    owner, feed, root, sessions = owner_factory(name, policy)
    assert len(sessions) == 1, f'{name} bypasses RecordingSession'
    rec = sessions[0]
    rx = next(lane for lane in rec.lanes.values() if 'rx' in lane.name.lower()
              or 'receive' in lane.name.lower() or 'stream' in lane.name.lower())
    start = rx.position
    origin = rx.origin if rx.origin is not None else start
    for first in range(start, start + 2 * RATE, RATE):
        feed(first, pcm(first, first + RATE))
        flush(rec)
    rec.begin('c1', {rx.clock: start + 2 * RATE}, peer='K9ZZT')
    for first in range(start + 2 * RATE, start + 65 * RATE, RATE):
        feed(first, pcm(first, first + RATE))
        flush(rec)
    rec.end('c1', {rx.clock: start + 65 * RATE})
    for first in range(start + 65 * RATE, start + 70 * RATE, RATE):
        feed(first, pcm(first, first + RATE))
        flush(rec)
    bounded(owner.rec.close if name == 'vara' else owner.close)
    bounded(rec.close)
    ref = index(rec.directory)['contacts']['c1']['lanes'][rx.name]
    assert ref['actual'] == [origin, start + 70 * RATE]
    assert ref['missing_pre_roll'] > 0
    assert len(ref['segments']) >= 2
    if policy == 'contacts':
        assert len(samples(rec.directory, ref)) == start + 70 * RATE - origin
    else:
        assert ref['source'] == rx.row['source']


@pytest.mark.parametrize('destination', ('', 'chosen'))
@pytest.mark.parametrize('audio_type', (config.Audio, config.ChainAudio))
@pytest.mark.parametrize('station_policy,inherited,policy', [
    ('off', None, 'off'), ('contacts', None, 'contacts'),
    ('everything', None, 'everything'), ('off', 'contacts', 'contacts'),
    ('contacts', 'off', 'off'),
])
def test_station_capture_destination_does_not_select_policy(
        tmp_path, monkeypatch, destination, audio_type, station_policy, inherited, policy):
    monkeypatch.delenv('HFMODEM_RECORDING', raising=False)
    if inherited:
        monkeypatch.setenv('HFMODEM_RECORDING', inherited)
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path / 'data'))
    cfg = config.StationConfig(
        station=config.Station(mycall='W0QXR', transmit=False),
        rig=config.Rig(cat='none'), protocols={},
        audio=audio_type(input='synthetic', capture_dir=str(tmp_path / destination) if destination else ''),
        recording=config.Recording(policy=station_policy))
    opened = []

    class Audio:
        def __init__(self, **kwargs):
            self.lanes = []

        def subscribe(self, lane):
            self.lanes.append(lane)

        def open(self):
            opened.extend(self.lanes)

    monkeypatch.setattr(process, 'StationAudio', Audio)
    monkeypatch.setattr(process.devices, 'find_device', lambda *a, **kw: 'synthetic')
    st = process.Station(cfg)
    monkeypatch.setattr(st, 'validate_options', lambda: None)
    monkeypatch.setattr(wav, 'store_later', lambda *a: None)
    monkeypatch.setattr(wav, '_flac', lambda *a: (_ for _ in ()).throw(OSError('encoder unavailable')))
    try:
        st.open_audio()
        selected = os.environ.get('HFMODEM_RECORDING')
        assert selected == policy, 'station must resolve at the run boundary'
        if policy != 'off':
            assert st.capture is not None, 'default run root must capture without capture_dir'
            assert st.capture in opened, 'recording must subscribe before samples arrive'
        if st.capture is not None:
            st.capture.push(Block(0, 0, pcm(0, 480)))
            st.capture.drain()
            bounded(st.capture.close)
        files = audio_files(tmp_path)
        assert bool(files) == (policy == 'everything')
        if policy != 'off':
            manifests = list(tmp_path.rglob('recording.json'))
            assert len(manifests) == 1, 'station capture must publish policy provenance'
            facts = json.loads(manifests[0].read_text())
            assert (facts['policy'], facts['source']) == (policy, 'inherited' if inherited else 'station')
    finally:
        if st.capture is not None:
            bounded(st.capture.close)


def test_kestrel_barrier_and_future_completion_use_storage_worker(owner_factory, monkeypatch):
    owner, feed, root, sessions = owner_factory('kestrel', 'everything')
    release, entered = threading.Event(), threading.Event()
    calls, threads = [], []

    def publish():
        entered.set()
        if sessions:
            assert next(iter(sessions[0].lanes.values())).consumed == RATE
        assert release.wait(TIMEOUT), 'publication was not released'
        calls.append('publication')
        threads.append(threading.current_thread())
        return 17

    feed(0, pcm(0, RATE))
    first = owner.after_pending(publish)
    assert entered.wait(TIMEOUT)
    second = owner.after_pending(lambda: calls.append('second'))
    first.add_done_callback(lambda future: threads.append(threading.current_thread()))
    try:
        assert not first.done() and not second.done()
        assert len(sessions) == 1, 'ordered publication must be handed to RecordingSession'
        rec = sessions[0]
        drained = threading.Event()
        consume = rec._consume

        def consumed(*args):
            consume(*args)
            drained.set()

        monkeypatch.setattr(rec, '_consume', consumed)
        feed(RATE, pcm(RATE, 2 * RATE))
        assert drained.wait(TIMEOUT), 'publication blocked PCM draining'
        assert threads == []
        assert next(iter(rec.lanes.values())).position == 2 * RATE
    finally:
        release.set()
    assert first.result(timeout=TIMEOUT) == 17
    second.result(timeout=TIMEOUT)
    assert calls == ['publication', 'second']
    assert threads == [rec._storage_thread, rec._storage_thread]


@pytest.mark.parametrize('policy', recording.POLICIES)
def test_varac_receive_path_retains_text_under_every_policy(owner_factory, tools, monkeypatch, tmp_path, policy):
    launch = importlib.import_module('varac_chat')
    root = tmp_path / 'sessions' / 'varac'
    root.mkdir(parents=True)
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    args = launch.parser().parse_args([
        '--mycall', 'W0QXR', '--band', '40', '--session', str(root),
        '--expect-model', 'synthetic', '--ptt-device', 'synthetic',
        '--tx-device', 'synthetic-out', '--rx-device', 'synthetic-in', '--no-bbs'])
    monkeypatch.setattr(launch, 'Terminal', lambda *a: io.StringIO())

    class StopSession(Exception):
        pass

    @contextmanager
    def host(audio):
        try:
            audio._stream.feed(pcm(0, RATE))
            raise StopSession
            yield
        finally:
            if audio.rec is not None:
                bounded(audio.rec.close)

    monkeypatch.setattr(launch, 'Host', host)
    with pytest.raises(StopSession):
        launch.run(args, NS(shutdown=lambda: None, keyed=False), NS(opened=lambda: None))
    for filename in ('host.jsonl', 'events.jsonl', 'contacts.jsonl', 'console.log'):
        assert (root / filename).is_file()
    assert bool(audio_files(root)) == (policy == 'everything'), 'VarAC persisted unselected receive audio'
    assert len(owner_factory.sessions) == 1, 'VarAC receive path bypasses RecordingSession'


def test_station_lane_preserves_block_offsets_and_keyed_intervals(owner_factory):
    owner, feed, root, sessions = owner_factory('station', 'everything')
    assert len(sessions) == 1, 'station capture must preserve Block.n0 through a shared lane'
    rec = sessions[0]
    feed(1000, pcm(1000, 1008))
    flush(rec)
    owner.flush_to(1008)
    owner.resync(1008)
    feed(1024, pcm(1024, 1032))
    flush(rec)
    bounded(owner.close)
    bounded(rec.close)
    rx = next(lane for lane in rec.lanes.values() if lane.position == 1032)
    facts = index(rec.directory)['lanes'][rx.name]
    assert (facts['origin'], facts['position'], facts['gaps']) == (1000, 1032, [[1008, 1024]])


@pytest.mark.parametrize('name', ('vara', 'ardop', 'pactor'))
@pytest.mark.parametrize('policy', recording.POLICIES)
def test_submitted_tx_has_a_separate_lane(owner_factory, monkeypatch, name, policy):
    owner, feed, root, sessions = owner_factory(name, policy)
    submitted = []
    if name == 'vara':
        module = importlib.import_module('kestrel_connect')
        monkeypatch.setattr(module, '_tx_end_in_capture', lambda audio, earliest: earliest)
        monkeypatch.setattr(module, '_tx_last_sample', lambda audio, end: end)
        def play(block, *args, **kwargs):
            submitted.append(block[:, 0].copy())
            kwargs['submitted'](0, block[:, 0])
        monkeypatch.setattr(module, 'play_drained', play)
        monkeypatch.setattr(module, '_bounded', lambda budget, fn, **kw: fn())
        owner.tx_tail = 0
        owner.tx(np.array([0., .1, -.1, 0.], np.float32))
    elif name == 'ardop':
        from hfmodem.besra import radio
        monkeypatch.setattr(radio, 'warm_output', lambda *a: None)
        monkeypatch.setattr(radio, 'to_card', lambda block, *a: block)
        def play(block, *args, **kwargs):
            submitted.append(block.copy())
            kwargs['submitted'](0, block)
        monkeypatch.setattr(radio, 'play_drained', play)
        owner._transmit(np.array([0., .1, -.1, 0.], np.float32))
    else:
        monkeypatch.setattr(owner, '_sleep_until', lambda *a: [
            owner._stream.feed(np.zeros(RATE, np.float32)) for _ in range(5)])
        owner.transmit(np.array([0., .1, -.1, 0.], np.float32))
        submitted = owner._stream.output
    assert submitted and any(np.any(block) for block in submitted)
    assert len(sessions) == 1, 'submitted TX must share the owner RecordingSession'
    rec = sessions[0]
    flush(rec)
    tx = [lane for lane in rec.lanes.values() if 'tx' in lane.name.lower()]
    assert len(tx) == 1, 'submitted output must have a separate TX lane'
    assert any(lane is not tx[0] for lane in rec.lanes.values()), 'RX and submitted TX must remain distinct'
    assert tx[0].position >= 4
    bounded(owner.rec.close if name == 'vara' else owner.close)
    bounded(rec.close)
    if policy in ('off', 'contacts'):
        assert audio_files(root) == []
    else:
        assert tx[0].row.get('source')


@pytest.mark.parametrize('prior_tx', (0, RATE + 3))
def test_tx_anchor_uses_one_source_snapshot(owner_factory, monkeypatch, prior_tx):
    from hfmodem.station.contactlog import ContactLog
    module = importlib.import_module('kestrel_connect')
    owner, feed, root, _ = owner_factory('vara', 'contacts')
    rec = owner.rec.session
    feed(0, pcm(0, 8))
    if prior_tx:
        owner.rec.tx.push(prior_tx - 4, pcm(0, 4))
    flush(rec)
    source_start = owner.samples
    tx_start = max(prior_tx, source_start)
    stamps = []

    def timing(kind, **details):
        if kind == 'play_start':
            stamps.append(details['capture_samples'])

    def diagnostic(kind, **details):
        if kind == 'tx_start':
            feed(source_start, pcm(8, 12))

    monkeypatch.setattr(owner, 'trace_timing_event', timing)
    monkeypatch.setattr(owner, '_diagnostic', diagnostic)
    monkeypatch.setattr(module, '_tx_end_in_capture', lambda audio, earliest: earliest)
    monkeypatch.setattr(module, '_tx_last_sample', lambda audio, end: end)
    monkeypatch.setattr(module, '_bounded', lambda budget, fn, **kw: fn())
    monkeypatch.setattr(module, 'play_drained', lambda block, *a, **kw:
                        kw['submitted'](0, block[:, 0]))
    owner.tx_tail = 0
    owner.tx(np.array([0., .1, -.1, 0.], np.float32))
    flush(rec)
    log = ContactLog(rec.directory, recording=rec)
    contact = dict(n=1, peer='K9ZZT', dial_hz=7105000, bandwidth_hz=500,
                   started_utc='2026-11-01T12:00:00+00:00', how='outgoing')
    log.update(contact, sample=source_start, sample_rate=RATE)
    log.update(contact | {'ended_utc': '2026-11-01T12:00:01+00:00'},
               sample=source_start + 4, sample_rate=RATE)
    bounded(owner.rec.close)
    selected = index(rec.directory)['contacts']['vara-1']
    assert stamps == [source_start]
    assert owner.rec.tx.row['source_anchor'] == {
        'clock': owner.rec.rx.clock, 'sample': 0, 'position': 0}
    assert selected['positions']['submitted-tx'] == source_start
    assert selected['end_positions']['submitted-tx'] == source_start + 4
    assert selected['lanes']['submitted-tx']['requested'] == [source_start - 60 * RATE,
                                                           source_start + 4 + 5 * RATE]
    assert selected['lanes']['submitted-tx']['actual'] == [prior_tx - 4 if prior_tx else tx_start,
                                                        tx_start + 4]


def test_vara_submitted_tx_after_adc_loss_uses_absolute_card_axis(owner_factory, monkeypatch):
    module = importlib.import_module('kestrel_connect')
    owner, feed, _, _ = owner_factory('vara', 'contacts')
    feed(0, pcm(0, 8))
    owner._stream.position += 8
    feed(16, pcm(16, 24))
    rec = owner.rec.session
    assert owner.samples == 16 and owner.rec.position == 24
    monkeypatch.setattr(module, '_bounded', lambda budget, fn, **kw: fn())
    monkeypatch.setattr(module, '_tx_end_in_capture', lambda audio, earliest: earliest)
    monkeypatch.setattr(module, '_tx_last_sample', lambda audio, end: end)
    monkeypatch.setattr(module, 'play_drained', lambda block, *a, **kw:
                        kw['submitted'](0, block[:, 0]))
    owner.tx_tail = 0
    owner.tx(np.array([0., .1, -.1, 0.], np.float32))
    rec.begin('loss', owner.rec.positions())
    feed(24, pcm(24, 28))
    rec.end('loss', owner.rec.positions())
    bounded(owner.rec.close)
    assert owner.samples == 20
    assert owner.rec.tx.row['source_anchor'] == dict(clock='card', sample=0, position=0)
    row = index(rec.directory)['contacts']['loss']
    assert row['positions'] == {'card': 24, 'submitted-tx': 24}
    assert row['end_positions'] == {'card': 28, 'submitted-tx': 28}
    assert row['lanes']['submitted-tx']['actual'] == [24, 28]


def test_vara_submitted_tx_queued_behind_earlier_tx_keeps_identity_card_axis(owner_factory, monkeypatch):
    module = importlib.import_module('kestrel_connect')
    owner, feed, _, _ = owner_factory('vara', 'contacts')
    feed(0, pcm(0, 8))
    owner._stream.position += 8
    feed(16, pcm(16, 24))
    owner.rec.tx.push(24, pcm(0, 4))
    rec = owner.rec.session
    assert owner.rec.position == 24 and owner.rec.tx.position == 28
    monkeypatch.setattr(module, '_bounded', lambda budget, fn, **kw: fn())
    monkeypatch.setattr(module, '_tx_end_in_capture', lambda audio, earliest: earliest)
    monkeypatch.setattr(module, '_tx_last_sample', lambda audio, end: end)
    monkeypatch.setattr(module, 'play_drained', lambda block, *a, **kw:
                        kw['submitted'](0, block[:, 0]))
    owner.tx_tail = 0
    owner.tx(np.array([0., .1, -.1, 0.], np.float32))
    assert owner.rec.tx.position == 32
    rec.begin('queued', owner.rec.positions())
    feed(24, pcm(24, 28))
    rec.end('queued', owner.rec.positions())
    bounded(owner.rec.close)
    assert owner.rec.tx.row['source_anchor'] == dict(clock='card', sample=0, position=0)
    row = index(rec.directory)['contacts']['queued']
    assert row['positions'] == {'card': 24, 'submitted-tx': 24}
    assert row['end_positions'] == {'card': 28, 'submitted-tx': 28}


def test_varac_releases_rig_and_audio_before_blocked_storage(owner_factory, tools, monkeypatch, tmp_path):
    launch = importlib.import_module('varac_chat')
    kestrel = importlib.import_module('kestrel_connect')
    audio, _, _, _ = owner_factory('vara', 'everything')
    args = launch.parser().parse_args(['--mycall', 'W0QXR', '--band', '40',
        '--session', str(tmp_path / 'session'), '--expect-model', 'synthetic',
        '--ptt-device', 'synthetic', '--tx-device', 'synthetic-out',
        '--rx-device', 'synthetic-in', '--no-bbs-service'])
    args.session.mkdir()
    events = []
    rig = NS(shutdown=lambda: events.append('rig-released'))
    monkeypatch.setattr(launch, 'Terminal', lambda *a: io.StringIO())
    audio_type = kestrel.AudioVaraIO
    monkeypatch.setattr(kestrel, 'AudioVaraIO', lambda *a, **kw: audio)
    monkeypatch.setattr(kestrel, 'drop_output', lambda: events.append('output-released'))
    monkeypatch.setattr(audio._stream, 'stop', lambda: events.append('input-stopped'))
    monkeypatch.setattr(audio._stream, 'close', lambda: events.append('input-released'))
    monkeypatch.setattr(kestrel, '_bounded', lambda budget, *actions, **kw: [fn() for fn in actions])
    host_type = launch.Host
    monkeypatch.setattr(host_type, '__enter__', lambda self: self)
    def host(transport, **kwargs):
        result = host_type(transport, runtime_factory=lambda _: NS(modem=None), **kwargs)
        result.server = NS(cmd_port=1, data_port=2, stop=lambda: events.append('host-stopped'))
        return result
    monkeypatch.setattr(launch, 'Host', host)
    class StopSession(Exception):
        pass
    @contextmanager
    def client(**kwargs):
        raise StopSession
        yield
    monkeypatch.setattr(launch.Client, 'open', client)
    entered, release = threading.Event(), threading.Event()
    close = audio.rec.close
    def storage():
        events.append('storage-wait')
        entered.set()
        assert release.wait(3)
        close()
    monkeypatch.setattr(audio.rec, 'close', storage)
    errors = []
    def run():
        try:
            launch.run(args, rig, NS(opened=lambda: None))
        except StopSession:
            pass
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(2), errors
        assert 'rig-released' in events, events
        assert events.index('host-stopped') < events.index('rig-released') < events.index('storage-wait')
        assert events.index('input-released') < events.index('storage-wait')
        assert events.index('output-released') < events.index('storage-wait')
    finally:
        release.set()
        thread.join(3)
        monkeypatch.setattr(kestrel, 'AudioVaraIO', audio_type)
    assert not thread.is_alive()
    assert errors == []


@pytest.mark.parametrize('name', OWNERS)
def test_owner_installs_boundary_before_source_start_and_closes_it(owner_factory, name):
    owner, feed, root, sessions = owner_factory(name, 'off')
    rec = owner.rec if name == 'vara' else owner
    edges = getattr(rec, 'boundaries', None)
    assert isinstance(edges, recording.ContactBoundaries)
    assert edges.session is sessions[0]
    edges.established((edges.run, 'vara', 'synthetic', 'K9ZZT', 'link', 1), 'W0QXR',
                      rec.positions())
    if name == 'ardop':
        assert owner.modem.recording_boundaries is edges
    if name == 'vara':
        assert owner.recording_boundaries is edges
        bounded(rec.close)
    else:
        bounded(owner.close)
    row, = sessions[0].index['contacts'].values()
    assert row['state'] == 'partial'
    assert row.get('end_positions') is not None


@pytest.mark.parametrize('name', ['kestrel', 'ardop'])
def test_owner_boundaries_map_idle_tx_from_source_anchor(owner_factory, name):
    from hfmodem.core.recording import contact_established, contact_ended

    owner, feed, _, sessions = owner_factory(name, 'contacts')
    rec, = sessions
    rx, tx = rec.lanes['rx'], rec.lanes['submitted-tx']
    tx.row['source_anchor'] = dict(clock=rx.clock, sample=16, position=24)
    tx.push(24, pcm(24, 32))
    for first in range(0, 80, RATE):
        feed(first, pcm(first, first + RATE))
        flush(rec)
    source = NS(recording_boundaries=owner.boundaries, recording_positions=owner.positions)
    contact_established(source, 'vara' if name == 'kestrel' else 'ardop', 'K9ZZT', 'W0QXR')
    for first in range(80, 88, RATE):
        feed(first, pcm(first, first + RATE))
    contact_ended(source)
    for first in range(88, 120, RATE):
        feed(first, pcm(first, first + RATE))
        flush(rec)
    contact_established(source, 'vara' if name == 'kestrel' else 'ardop', 'K9ZZT', 'N0ZQT')
    feed(120, pcm(120, 128))
    contact_ended(source)
    bounded(owner.close)
    rows = sorted(rec.index['contacts'].values(), key=lambda row: row['positions'][rx.clock])
    assert [row['positions'] for row in rows] == [
        {rx.clock: 80, tx.clock: 88}, {rx.clock: 120, tx.clock: 128}]
    assert [row['end_positions'][tx.clock] for row in rows] == [96, 136]
    assert tx.position == 32
    assert [row['lanes']['submitted-tx']['requested'] for row in rows] == [
        [88 - 60 * RATE, 96 + 5 * RATE], [128 - 60 * RATE, 136 + 5 * RATE]]


@pytest.mark.parametrize('name', ['station', 'pactor', 'ardop'])
def test_production_owner_publishes_boundary_protocol_metadata(owner_factory, name):
    from hfmodem.core.recording import contact_established, contact_ended

    owner, _, root, sessions = owner_factory(name, 'off')
    rec, = sessions
    boundaries = owner.boundaries
    source = NS(recording_boundaries=boundaries, recording_positions=owner.positions)
    protocols = ('ale', 's5066') if name == 'station' else (name,)
    for protocol in protocols:
        contact_established(source, protocol, 'K9ZZT', 'W0QXR')
        contact_ended(source, 'owner retired')
    bounded(owner.close)
    assert rec.publish is not None
    journal = rec.publish.__self__
    assert len(journal.records) == len(protocols)
    assert {row['protocol'] for row in journal.records.values()} == set(protocols)
    for row in journal.records.values():
        assert row['protocol'] == row['recording']['key'][1]
        assert row['recording']['state'] == 'partial'
        published = json.loads((Path(row['directory']) / 'provenance.json').read_text())
        assert published['recording']['id'] == row['recording']['id']
    assert not audio_files(root)
