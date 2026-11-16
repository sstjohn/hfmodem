# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Capture ownership proofs use only replay audio and injected consumers."""
import json
import threading
import wave

import numpy as np
import pytest

from hfmodem.core.audio import Block, ReplayAudio, StationAudio
from hfmodem.station.chat_capture import ChatCapture, CaptureStartupCleanup


class Recorder:
    def __init__(self):
        self.facts = {}
        self.samples = []
        self.threads = set()
        self.closed = False

    def push(self, samples):
        self.threads.add(threading.get_ident())
        self.samples.extend(samples)

    def drain(self):
        self.threads.add(threading.get_ident())

    def close(self):
        self.threads.add(threading.get_ident())
        self.closed = True


class Monitor:
    def __init__(self):
        self.items = []
        self.closed = False

    def put(self, samples):
        self.items.append(samples)

    def close(self):
        self.closed = True


def test_recording_keeps_keyed_audio_monitor_masks_and_callback_copies():
    audio = ReplayAudio(np.zeros(20), blocksize=4)
    audio.pump()
    recorder, monitor = Recorder(), Monitor()
    keyed = [False]
    capture = ChatCapture(audio, recorder, monitor, keyed=lambda: keyed[0])
    try:
        samples = np.array([.1, .2], np.float32)
        capture.push(Block(0, 4, samples))
        samples[:] = .9
        keyed[0] = True
        capture.push(Block(0, 6, np.array([.3, .4], np.float32)))
        capture.flush_to(8)
        assert recorder.samples == []
        assert capture.flush_through(8) == 4
        np.testing.assert_allclose(recorder.samples, [.1, .2, .3, .4])
        np.testing.assert_allclose(monitor.items[0], [.1, .2])
        np.testing.assert_array_equal(monitor.items[1], [0, 0])
        assert capture.origin == 4
        assert capture.position == 4
        with pytest.raises(ValueError, match='predates'):
            capture.file_offset(3)
    finally:
        capture.close()
    assert recorder.closed and monitor.closed
    assert len(recorder.threads) == 1
    assert threading.get_ident() not in recorder.threads
    assert audio._lanes == []


def test_index_gaps_pad_recording_and_reach_monitor_with_provenance():
    audio = ReplayAudio(np.zeros(10))
    recorder, monitor = Recorder(), Monitor()
    capture = ChatCapture(audio, recorder, monitor)
    capture.push(Block(0, 0, np.array([.1, .2])))
    capture.resync(2)
    capture.push(Block(0, 4, np.array([.5])))
    capture.close()
    np.testing.assert_allclose(recorder.samples, [.1, .2, 0, 0, .5])
    assert monitor.items[1:3] == [('gap', 0), ('gap', 2)]
    assert recorder.facts['station_origin_sample'] == 0
    assert recorder.facts['station_end_sample'] == 5
    assert recorder.facts['capture_gaps'] == [
        {'at': 2, 'end': None, 'cause': 'capture resync'},
        {'at': 2, 'end': 4, 'cause': 'capture index gap'},
    ]


def test_queue_overflow_is_failure_even_after_cleanup():
    audio = ReplayAudio(np.zeros(10))
    recorder = Recorder()
    capture = ChatCapture(audio, recorder, capacity_samples=2)
    capture.push(Block(0, 0, np.ones(2)))
    capture.push(Block(0, 2, np.ones(1)))
    with pytest.raises(RuntimeError, match='overflow'):
        capture.close()
    assert recorder.closed
    assert recorder.facts['chat_capture_incomplete']
    assert audio._lanes == []
    with pytest.raises(RuntimeError, match='overflow'):
        capture.close()


def test_active_disk_drain_counts_against_queue_bound_and_close_can_retry():
    audio = ReplayAudio(np.zeros(10))
    started, release = threading.Event(), threading.Event()

    class BlockingRecorder(Recorder):
        def drain(self):
            started.set()
            assert release.wait(2)
            super().drain()

    recorder = BlockingRecorder()
    capture = ChatCapture(audio, recorder, capacity_samples=2)
    try:
        capture.push(Block(0, 0, np.ones(2)))
        future = capture.drain()
        assert started.wait(2)
        assert capture.drain() is future
        with pytest.raises(RuntimeError, match='flush still pending'):
            capture.close(timeout=0)
        assert not recorder.closed
        assert not future.done()
        assert capture._detached
        # An active batch cannot be hidden by repeatedly scheduling drain().
        assert capture._queued_samples == 2
    finally:
        release.set()
        capture.close()
    assert recorder.closed
    assert all(not worker.is_alive() for worker in capture._worker._threads)


def test_finalizer_timeout_retains_worker_and_retries():
    audio = ReplayAudio(np.zeros(1))
    started, release = threading.Event(), threading.Event()

    class BlockingRecorder(Recorder):
        def close(self):
            started.set()
            assert release.wait(2)
            super().close()

    recorder = BlockingRecorder()
    capture = ChatCapture(audio, recorder)
    try:
        with pytest.raises(RuntimeError, match='retirement still pending'):
            capture.close(timeout=0)
        assert started.wait(2)
        assert not recorder.closed
    finally:
        release.set()
        capture.close()
    assert recorder.closed


def test_partial_start_consumers_retire_on_bounded_worker_and_continue_after_error():
    started, release = threading.Event(), threading.Event()
    events = []

    class Slots:
        def close(self):
            events.append('slots')
            started.set()
            assert release.wait(2)
            raise RuntimeError('decoder close failed')

    class PartialRecorder:
        def close(self):
            events.append('recorder')

    cleanup = CaptureStartupCleanup(Slots(), PartialRecorder())
    try:
        with pytest.raises(RuntimeError, match='startup cleanup still pending'):
            cleanup.close(timeout=0)
        assert started.wait(2)
    finally:
        release.set()
        with pytest.raises(RuntimeError, match='decoder close failed'):
            cleanup.close()
    assert events == ['slots', 'recorder']
    assert all(not worker.is_alive() for worker in cleanup._worker._threads)


def test_backlog_bound_includes_samples_already_handed_to_recording_worker():
    audio = ReplayAudio(np.zeros(10))
    started, release = threading.Event(), threading.Event()

    class BlockingRecorder(Recorder):
        def drain(self):
            started.set()
            assert release.wait(2)

    recorder = BlockingRecorder()
    capture = ChatCapture(audio, recorder, capacity_samples=2)
    capture.push(Block(0, 0, np.ones(2)))
    future = capture.drain()
    try:
        assert started.wait(2)
        capture.push(Block(0, 2, np.ones(1)))
        assert capture._failure is not None
        assert capture.drain() is future
        assert capture.received_through == 2
    finally:
        release.set()
        with pytest.raises(RuntimeError, match='overflow'):
            capture.close()
    assert recorder.closed


def test_contact_cutout_starts_after_ending_samples_flush():
    audio = ReplayAudio(np.zeros(20), blocksize=4)
    audio.pump()
    recorder = Recorder()
    calls = []
    finished = threading.Event()

    class ContactLog:
        def update(self, record, **fields):
            calls.append(('update', fields))
            return fields

        def finish(self, native):
            assert len(recorder.samples) >= native['sample']
            calls.append(('finish', native['sample']))
            finished.set()

    capture = ChatCapture(audio, recorder, contact_log=ContactLog())
    try:
        capture.contact({'n': 1}, card_index=4, offset_hz=250)
        capture.push(Block(0, 4, np.ones(4)))
        capture.contact({'n': 1, 'ended_utc': 'now'}, card_index=8, offset_hz=250)
        assert finished.wait(2)
    finally:
        capture.close()
    assert calls[0] == ('update', {'sample': 0, 'sample_rate': 48000, 'offset_hz': 250})
    assert calls[1][1]['sample'] == 4
    assert calls[2] == ('finish', 4)


def test_failed_contact_save_still_closes_owned_consumers():
    audio = ReplayAudio(np.zeros(1))
    recorder, monitor = Recorder(), Monitor()

    class ContactLog:
        def update(self, record, **fields):
            return fields

        def finish(self, native):
            raise ValueError('cutout failed')

    capture = ChatCapture(audio, recorder, monitor, contact_log=ContactLog())
    capture.contact({'ended_utc': 'now'})
    with pytest.raises(RuntimeError, match='cutout failed'):
        capture.close()
    assert recorder.closed and monitor.closed


def test_real_recorder_has_file_offsets_and_keyed_samples(tmp_path, monkeypatch):
    from hfmodem.core import wav
    monkeypatch.setattr(wav, 'store', lambda path, **kwargs: {'wav': path.name})
    audio = ReplayAudio(np.zeros(20), blocksize=4)
    audio.pump()
    path = tmp_path / 'receive.wav'
    recorder = wav.CaptureRecorder(path, rate=48000)
    capture = ChatCapture(audio, recorder, keyed=lambda: True)
    capture.push(Block(0, 4, np.array([.25, -.5])))
    capture.close()
    segment, = recorder._out.paths
    with wave.open(str(segment)) as recording:
        assert recording.getnframes() == 2
        assert list(np.frombuffer(recording.readframes(2), dtype='<i2')) == [8192, -16384]
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['station_origin_sample'] == 4
    assert side['station_end_sample'] == 6


@pytest.mark.parametrize('kind', ['live', 'replay'])
def test_removable_audio_subscription_barrier_and_stale_callback_snapshot(kind):
    audio = (StationAudio(input_device=None, output_device=None) if kind == 'live'
             else ReplayAudio(np.zeros(4), blocksize=4))
    started, release = threading.Event(), threading.Event()
    calls = []

    class Tap:
        def push(self, block):
            started.set()
            assert release.wait(2)
            calls.append(block.n0)

    tap = Tap()
    assert audio.subscribe(tap, removable=True) is tap
    stale = audio._lanes[0]
    worker = threading.Thread(target=stale.push, args=(Block(0, 0, np.ones(4)),))
    worker.start()
    try:
        assert started.wait(2)
        with pytest.raises(TimeoutError, match='callback still pending'):
            audio.unsubscribe(tap, timeout=0)
        assert audio._lanes == [stale]
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    audio.unsubscribe(tap, timeout=1)
    stale.push(Block(0, 4, np.ones(4)))
    audio.unsubscribe(tap, timeout=0)
    assert calls == [0]
    assert audio._lanes == []
