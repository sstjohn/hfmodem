# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Storage work never holds acquisition or segment-opening locks."""
import errno
import os
from copy import deepcopy
from collections import namedtuple
from pathlib import Path
import sys
import threading
import wave

import pytest

from hfmodem.core import capture_identity as ci, wav
from hfmodem.tests.station.recording_support import (
    RATE, TIMEOUT, audio_files, bounded, flush, index, interval, pcm, push, same_path,
)

DiskUsage = namedtuple('DiskUsage', 'total used free')


def test_push_only_writes_first_origin_marker_and_has_a_two_second_queue(sessions, monkeypatch):
    reports = []
    rec, root = sessions('everything', report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    tx = rec.stream('submitted-tx', rate=RATE, clock='output')
    entered, release = threading.Event(), threading.Event()
    original = wav.SegmentedWriter.writeframes
    def blocked(out, data):
        entered.set()
        assert release.wait(TIMEOUT)
        return original(out, data)
    monkeypatch.setattr(wav.SegmentedWriter, 'writeframes', blocked)
    lane.push(0, pcm(0, 8))
    assert entered.wait(TIMEOUT)
    opening = Path.open
    callback_thread = threading.get_ident()
    markers = []
    def guarded(path, *args, **kwargs):
        if threading.get_ident() == callback_thread:
            assert path.suffix == '.lane', 'unexpected file I/O in Lane.push'
            markers.append(path)
        return opening(path, *args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'open', guarded)
            for start in (8, 16, 24):
                lane.push(start, pcm(start, start + 8))
            tx.push(0, pcm(0, 8))
        assert lane.position == 32
        assert tx.position == 8
        assert len(markers) == 1
        assert markers[0].name.endswith('-o' + '0' * 20 + '.lane')
    finally:
        release.set()
    flush(rec)
    rec.begin('c1', {'card': 32})
    rec.end('c1', {'card': 32})
    push(rec, lane, 32, 72)
    bounded(rec.close)
    assert index(root)['lanes']['rx']['gaps'] == [[24, 32]]
    assert index(root)['lanes']['submitted-tx']['gaps'] == []
    assert interval(root)['gaps'] == [[24, 32]]
    assert any('drop' in str(r).lower() or 'overflow' in str(r).lower() for r in reports)


def test_overlarge_block_is_dropped_whole_without_clock_compression(sessions):
    reports = []
    rec, root = sessions('everything', report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    lane.push(0, pcm(0, 2 * RATE + 1))
    flush(rec)
    push(rec, lane, 17, 25)
    bounded(rec.close)
    assert lane.position == 25
    assert index(root)['lanes']['rx']['gaps'] == [[0, 17]]
    assert reports


@pytest.mark.parametrize('block', ['hash', 'store', 'publish'])
def test_storage_and_publication_share_one_worker_after_pcm(sessions, monkeypatch, block):
    rec_threads, publications = [], []
    entered, release = threading.Event(), threading.Event()
    worker = []
    original_hash, original_store = wav._sha256, wav.store
    def gated(fn, *args, **kwargs):
        worker.append(threading.get_ident())
        entered.set()
        assert release.wait(TIMEOUT)
        return fn(*args, **kwargs)
    def publish(row):
        publications.append(row)
        if block == 'publish':
            gated(lambda: None)
    rec, root = sessions('everything', publish=publish)
    capture = ci.allocate({'capture': root / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav',
                                 clock_id='card', continuous=True)
    lane = rec.stream('rx', rate=RATE, clock='card', ownership=owner)
    write = wav.SegmentedWriter.writeframes
    consumed = threading.Event()
    def observed(out, data):
        rec_threads.append(threading.get_ident())
        result = write(out, data)
        consumed.set()
        return result
    monkeypatch.setattr(wav.SegmentedWriter, 'writeframes', observed)
    if block == 'hash':
        monkeypatch.setattr(wav, '_sha256', lambda *a, **kw: gated(original_hash, *a, **kw))
    elif block == 'store':
        monkeypatch.setattr(wav, 'store', lambda *a, **kw: gated(original_store, *a, **kw))
    barrier, barrier_thread = threading.Event(), []
    def after_storage():
        barrier_thread.append(threading.get_ident())
        barrier.set()
    try:
        rec.begin('c1', {'card': 0})
        rec.end('c1', {'card': 0})
        for first in range(0, 60 * RATE, RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + RATE))
            assert consumed.wait(TIMEOUT)
        assert entered.wait(TIMEOUT)
        for first in range(60 * RATE, 64 * RATE, RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + RATE))
            assert consumed.wait(TIMEOUT)
        rec.after_pending(after_storage)
        assert not barrier.is_set()
        assert lane.position == 64 * RATE
        assert len(audio_files(root)) == 2
        if block != 'publish':
            assert index(root)['segments'][0]['state'] == 'storing'
            assert ci.audio_metadata(lane.writer.paths[0])['capture_state'] == 'storing'
        assert not set(worker).intersection(rec_threads)
    finally:
        release.set()
    assert barrier.wait(TIMEOUT)
    bounded(rec.close)
    assert index(root)['lanes']['rx']['gaps'] == []
    assert publications and len(set(worker)) == 1
    assert barrier_thread == [worker[0]]
    assert ci.audio_metadata(lane.writer.paths[0])['capture_state'] == 'finalized'


def test_descriptor_overflow_is_durable_pending_and_drains_serially(sessions, monkeypatch):
    entered, release, consumed = (threading.Event() for _ in range(3))
    store = wav.store
    jobs, concurrent = [], []
    active = 0
    def blocked(path, **kwargs):
        nonlocal active
        active += 1
        concurrent.append(active)
        jobs.append((Path(path), threading.get_ident()))
        try:
            entered.set()
            assert release.wait(TIMEOUT)
            return store(path, **kwargs)
        finally:
            active -= 1
    monkeypatch.setattr(wav, 'store', blocked)
    write = wav.SegmentedWriter.writeframes
    def observed(out, data):
        result = write(out, data)
        consumed.set()
        return result
    monkeypatch.setattr(wav.SegmentedWriter, 'writeframes', observed)
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    try:
        for first in range(0, 12 * 60 * RATE, RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + RATE))
            assert consumed.wait(TIMEOUT)
            if first + RATE == 60 * RATE:
                assert entered.wait(TIMEOUT)
        assert entered.wait(TIMEOUT)
        with rec._state_lock:
            descriptors = deepcopy(rec.index['segments'])
        assert len(descriptors) == 12
        assert sum(row['state'] == 'storing' for row in descriptors) == 9
        assert sum(row['state'] == 'pending' for row in descriptors) == 3
        assert all((root / row['source']).exists() for row in descriptors)
        assert len(jobs) == 1
    finally:
        release.set()
    flush(rec)
    assert sum(d['state'] == 'pending' for d in index(root)['segments']) == 3
    bounded(rec.close)
    assert len(jobs) == 12 and max(concurrent) == 1
    assert len({thread for _, thread in jobs}) == 1
    assert all(row['state'] == 'finalized' for row in index(root)['segments'])


def test_low_space_checks_each_minute_and_warns_every_ten_minutes(sessions):
    now, calls, reports = [0.0], [], []
    def usage(path):
        calls.append(now[0])
        return DiskUsage(total=10 << 30, used=9 << 30, free=1 << 30)
    rec, root = sessions('everything', clock=lambda: now[0], disk_usage=usage,
                         report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    for moment in (0, 59, 60, 599, 600, 659, 660, 1200):
        now[0] = moment
        first = lane.position
        lane.push(first, pcm(first, first + RATE))
        flush(rec)
    assert calls == [0, 60, 599, 659, 1200]
    low = [r for r in reports if 'space' in str(r).lower()]
    assert len(low) == 2
    bounded(rec.close)
    assert audio_files(root)


def test_space_threshold_includes_two_hours_of_all_active_native_pcm(sessions):
    reports, now = [], [0]
    rec, _ = sessions('everything', clock=lambda: now[0],
                      disk_usage=lambda p: DiskUsage(8 << 30, 5 << 30, 3 << 30), report=reports.append)
    lanes = [rec.stream(f'rx-{i}', rate=96000, clock=f'card-{i}') for i in range(3)]
    now[0] = 60
    for lane in lanes:
        lane.push(0, pcm(0, 8))
    flush(rec)
    assert any('space' in str(r).lower() for r in reports)


def test_enospc_disables_only_the_failed_lane_for_the_run(sessions, monkeypatch):
    reports = []
    rec, root = sessions('everything', report=reports.append)
    rx = rec.stream('rx', rate=RATE, clock='card')
    tx = rec.stream('submitted-tx', rate=RATE, clock='output')
    failed_path = root / index(root)['lanes']['rx']['source']
    write, failures = wav.SegmentedWriter.writeframes, []
    def fail_rx(out, data):
        if same_path(out.path, failed_path):
            failures.append(out.samples)
            raise OSError(errno.ENOSPC, 'synthetic disk full')
        return write(out, data)
    monkeypatch.setattr(wav.SegmentedWriter, 'writeframes', fail_rx)
    for first in range(0, 65 * RATE, RATE):
        rx.push(first, pcm(first, first + RATE))
        tx.push(first, pcm(first, first + RATE))
        flush(rec)
    bounded(rec.close)
    assert rx.position == tx.position == 65 * RATE
    assert len(failures) == 1
    lanes = index(root)['lanes']
    assert lanes['rx']['state'] == 'disabled'
    assert lanes['rx']['gaps'] == [[0, 65 * RATE]]
    assert lanes['submitted-tx']['gaps'] == []
    assert len([r for r in reports if 'disk full' in str(r).lower()]) == 1
    paths = [root / row['source'] for row in index(root)['segments']
             if row['lane'] == 'submitted-tx']
    count = 0
    for path in paths:
        with wave.open(str(path)) as src:
            count += src.getnframes()
    assert count == 65 * RATE


def test_a_lane_whose_writer_cannot_be_built_records_every_sample_as_a_gap(sessions, monkeypatch):
    reports = []
    rec, root = sessions('everything', report=reports.append)
    build = wav.SegmentedWriter.__init__
    def refuse(out, *args, **kwargs):
        raise ModuleNotFoundError("No module named 'hfhost.recordings'")
    monkeypatch.setattr(wav.SegmentedWriter, '__init__', refuse)
    rx = rec.stream('rx', rate=RATE, clock='card')
    monkeypatch.setattr(wav.SegmentedWriter, '__init__', build)
    tx = rec.stream('submitted-tx', rate=RATE, clock='output')
    for first in range(3 * RATE, 8 * RATE, RATE):
        rx.push(first, pcm(first, first + RATE))
        tx.push(first, pcm(first, first + RATE))
        flush(rec)
    rec.begin('K9ZZT-1', {'card': 4 * RATE, 'output': 4 * RATE})
    rec.end('K9ZZT-1', {'card': 6 * RATE, 'output': 6 * RATE})
    flush(rec)
    bounded(rec.close)
    lanes = index(root)['lanes']
    assert lanes['rx']['state'] == 'failed'
    assert lanes['rx']['gaps'] == [[3 * RATE, 8 * RATE]]
    assert lanes['submitted-tx']['state'] != 'failed'
    assert lanes['submitted-tx']['gaps'] == []
    assert not [row for row in index(root)['segments'] if row['lane'] == 'rx']
    contact = index(root)['contacts']['K9ZZT-1']
    assert contact['lanes']['rx']['state'] == 'partial'
    assert contact['lanes']['rx']['gaps'] == [[3 * RATE, 8 * RATE]]
    assert contact['lanes']['rx']['segments'] == []
    assert len([r for r in reports if 'hfhost.recordings' in str(r)]) == 1


def test_failed_verification_keeps_hashed_wav_and_finalizes_it(sessions, monkeypatch):
    rec, root = sessions('everything')
    def corrupt(*args):
        target = Path(args[args.index('-o') + 1])
        target.write_bytes(b'wrong decoded bytes' if '-d' in args else b'fake flac')
    monkeypatch.setattr(wav, '_flac', corrupt)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 16)
    bounded(rec.close)
    [descriptor] = index(root)['segments']
    assert descriptor['state'] == 'finalized'
    path = root / descriptor['source']
    assert path.suffix == '.wav' and path.exists()
    assert descriptor['sha256'] == wav._sha256(path)
    assert not list(root.rglob('*.flac')) and not list(root.rglob('*.part'))


def test_barrier_publication_runs_on_storage_worker_and_preserves_order(sessions):
    rec, _ = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    callbacks, done = [], threading.Event()
    lane.push(0, pcm(0, 8))
    rec.after_pending(lambda: callbacks.append((1, threading.get_ident(), lane.position)))
    rec.after_pending(lambda: (callbacks.append((2, threading.get_ident(), lane.position)), done.set()))
    assert done.wait(TIMEOUT)
    assert [row[0] for row in callbacks] == [1, 2]
    assert len({row[1] for row in callbacks}) == 1
    assert callbacks[0][1] != threading.get_ident()
    assert all(row[2] == 8 for row in callbacks)


def test_close_drains_more_than_one_full_queue_of_pending_descriptors(sessions, monkeypatch):
    entered, release, consumed = (threading.Event() for _ in range(3))
    store, write = wav.store, wav.SegmentedWriter.writeframes
    jobs = []
    def blocked(path, **kwargs):
        jobs.append(path)
        entered.set()
        assert release.wait(TIMEOUT)
        return store(path, **kwargs)
    def observed(out, data):
        result = write(out, data)
        consumed.set()
        return result
    monkeypatch.setattr(wav, 'store', blocked)
    monkeypatch.setattr(wav.SegmentedWriter, 'writeframes', observed)
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    try:
        for first in range(0, 24 * 60 * RATE, 2 * RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + 2 * RATE))
            assert consumed.wait(TIMEOUT)
            if first + 2 * RATE == 60 * RATE:
                assert entered.wait(TIMEOUT)
        assert entered.wait(TIMEOUT) and len(jobs) == 1
        with rec._state_lock:
            assert sum(d['state'] == 'pending' for d in rec.index['segments']) == 15
    finally:
        release.set()
    flush(rec)
    assert sum(d['state'] == 'pending' for d in index(root)['segments']) == 15
    bounded(rec.close)
    assert len(jobs) == 24
    assert all(d['state'] == 'finalized' for d in index(root)['segments'])


def test_segment_coverage_maps_absolute_gaps_to_stored_offsets(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 1000, 1016)
    push(rec, lane, 1024, 1040)
    bounded(rec.close)
    descriptors = index(root)['segments']
    assert [d['actual'] for d in descriptors] == [[1000, 1016], [1024, 1040]]
    assert [d['spans'] for d in descriptors] == [
        [{'absolute': [1000, 1016], 'offset': 0}],
        [{'absolute': [1024, 1040], 'offset': 0}],
    ]
    assert index(root)['lanes']['rx']['gaps'] == [[1016, 1024]]


def test_recording_workers_breathe_at_normal_priority(sessions, monkeypatch):
    """A worker holding the GIL at background QoS kept the callback out for 8-28 ms."""
    from hfmodem.core import gil
    monkeypatch.setattr(os, 'setpriority', lambda *args: pytest.fail(f'setpriority{args}'), raising=False)
    rec, _ = sessions('everything')
    rec.stream('rx', rate=RATE, clock='card')
    flush(rec)
    workers = {t.ident for t in threading.enumerate() if t.name in ('recording-pcm', 'recording-store')}
    assert len(workers) == 2 and workers <= set(gil._due)


def test_a_full_disk_says_so_once_not_on_every_index_write(sessions, monkeypatch):
    from hfmodem.tests.station.recording_support import module
    reports = []
    rec, _ = sessions('everything', report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    def full(path, facts):
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(module(), '_durable_json', full)
    push(rec, lane, 0, 20 * RATE)
    bounded(rec.close)
    assert len([r for r in reports if 'Recording storage' in r]) == 1, reports


def test_storage_priority_on_windows_lowers_only_the_current_worker_thread(monkeypatch):
    from types import SimpleNamespace
    from hfmodem.core import winproc
    from hfmodem.tests.station.recording_support import module
    recording = module()
    calls = []
    api = SimpleNamespace(GetCurrentThread=lambda: 'this thread',
                          SetThreadPriority=lambda *args: calls.append(args) or True)
    monkeypatch.setattr(recording.sys, 'platform', 'win32')
    monkeypatch.setattr(winproc, '_api', lambda: api)
    recording._background(lambda message: pytest.fail(message))
    assert calls == [('this thread', winproc.THREAD_PRIORITY_BELOW_NORMAL)]


@pytest.mark.skipif(sys.platform != 'win32', reason='reads back a Windows thread priority')
def test_storage_worker_runs_below_normal_on_windows():
    from hfmodem.core import winproc
    from hfmodem.tests.station.recording_support import module
    recording = module()
    api = winproc._api()
    seen = {}
    def worker():
        recording._background(lambda message: pytest.fail(message))
        seen['worker'] = api.GetThreadPriority(api.GetCurrentThread())
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(10)
    assert seen == {'worker': winproc.THREAD_PRIORITY_BELOW_NORMAL}
    assert api.GetThreadPriority(api.GetCurrentThread()) == 0
