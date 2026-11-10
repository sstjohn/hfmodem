# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Per-block recording cost stays bounded, and every boundary still reaches the disk."""
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import recording
from hfmodem.tests.station.recording_support import TIMEOUT, bounded, flush, index, pcm, samples

FS, BLOCK = 48000, 128


def _feed(lane, start, stop, step=BLOCK):
    for first in range(start, stop, step):
        lane.push(first, pcm(first, first + BLOCK))


def _on_disk(root, ready):
    deadline = time.monotonic() + TIMEOUT
    while True:
        try:
            facts = index(root)
            if ready(facts):
                return facts
        except (FileNotFoundError, KeyError, IndexError):
            pass
        assert time.monotonic() < deadline, 'recording.json never showed it'
        time.sleep(0.02)


@pytest.fixture
def no_timer(monkeypatch):
    """Only boundaries may snapshot the index."""
    monkeypatch.setattr(recording, '_INDEX_INTERVAL', 3600, raising=False)


@pytest.mark.parametrize('seed', range(5))
def test_gap_merge_matches_a_full_sort(seed):
    rng = np.random.default_rng(seed)
    gaps, every = [], []
    for _ in range(300):
        first = int(rng.integers(0, 2000)) if rng.random() < .3 else (gaps[-1][1] if gaps else 0) + int(rng.integers(-3, 20))
        last = first + int(rng.integers(-1, 15))
        recording._gap(gaps, first, last)
        if last > first:
            every.append([first, last])
        merged = []
        for start, stop in sorted(every):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(stop, merged[-1][1])
            else:
                merged.append([start, stop])
        assert gaps == merged


def test_blocks_between_boundaries_do_not_each_copy_the_index(sessions, monkeypatch, no_timer):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    flush(rec)
    saves = []
    save = rec._save
    monkeypatch.setattr(rec, '_save', lambda: (saves.append(1), save()))
    _feed(lane, 0, 400 * BLOCK)
    flush(rec)
    assert len(saves) <= 5
    assert index(root)['lanes']['rx']['position'] == 400 * BLOCK


def test_callback_blocks_reach_the_writer_in_batches_and_whole(sessions, monkeypatch):
    """A 128-frame block that woke the drainer put a GIL contender on every
    audio callback: on 2026-11-05 a Windows arm recording `everything` lost
    nine of fifteen PACTOR slots that the same arm with recording off kept."""
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    flush(rec)
    consumed = []
    consume = rec._consume
    monkeypatch.setattr(rec, '_consume', lambda *a: (consumed.append(len(a[2])), consume(*a)))
    start = time.monotonic()
    for n, first in enumerate(range(0, FS // 2, BLOCK)):
        while time.monotonic() < start + n * BLOCK / FS:
            time.sleep(BLOCK / FS / 4)
        lane.push(first, pcm(first, first + BLOCK))
    flush(rec)
    total = len(range(0, FS // 2, BLOCK)) * BLOCK
    assert sum(consumed) == total and len(consumed) <= 10
    written = samples(root, {'segments': index(root)['lanes']['rx']['expected']})
    assert np.array_equal(written, np.round(pcm(0, total) * 32768).astype('<i2'))


def test_a_batch_that_cannot_be_joined_is_written_block_by_block(sessions, monkeypatch):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    flush(rec)
    monkeypatch.setattr(recording, '_runs', lambda blocks: (_ for _ in ()).throw(MemoryError()))
    _feed(lane, 0, 40 * BLOCK)
    flush(rec)
    written = samples(root, {'segments': index(root)['lanes']['rx']['expected']})
    assert np.array_equal(written, np.round(pcm(0, 40 * BLOCK) * 32768).astype('<i2'))
    assert lane.queued == 0 and not lane.gaps


def test_after_a_crash_the_disk_is_no_older_than_the_index_interval(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    flush(rec)
    started = time.monotonic()
    _feed(lane, 0, 100 * BLOCK)
    # `position` is the producer's; the drainer's own progress is the segment's coverage.
    facts = _on_disk(root, lambda f: f['lanes']['rx']['expected'][0]['actual'] == [0, 100 * BLOCK])
    assert time.monotonic() - started < 4 * recording._INDEX_INTERVAL + 1
    expected = facts['lanes']['rx']['expected']
    assert len(expected) == 1 and facts['lanes']['rx']['position'] == 100 * BLOCK
    assert (root / expected[0]['source']).exists()


def test_a_new_segment_reaches_the_index_before_the_timer(sessions, monkeypatch, no_timer):
    monkeypatch.setattr(recording.wav, 'SEGMENT_SAMPLES', 10 * BLOCK)
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    _feed(lane, 0, 15 * BLOCK)
    facts = _on_disk(root, lambda f: len(f['lanes']['rx']['expected']) == 2)
    assert [e['actual'][0] for e in facts['lanes']['rx']['expected']] == [0, 10 * BLOCK]


def test_a_contact_end_reaches_the_disk_without_the_timer(sessions, no_timer):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    _feed(lane, 0, 50 * BLOCK)
    rec.begin('c1', {'card': 10 * BLOCK})
    _feed(lane, 50 * BLOCK, 100 * BLOCK)
    rec.end('c1', {'card': 60 * BLOCK})
    facts = _on_disk(root, lambda f: f['contacts']['c1'].get('end_positions') == {'card': 60 * BLOCK})
    assert facts['lanes']['rx']['position'] >= 60 * BLOCK


def test_close_leaves_the_final_index_on_disk(sessions, no_timer):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    _feed(lane, 0, 200 * BLOCK)
    bounded(rec.close)
    facts = index(root)
    row = facts['lanes']['rx']
    assert row['closed'] is True and row['position'] == 200 * BLOCK and row['gaps'] == []
    assert facts['segments'] and all(d['state'] == 'finalized' for d in facts['segments'])


def test_gaps_inside_a_coalescing_window_cost_no_storage_jobs(sessions, monkeypatch, no_timer):
    reports = []
    rec, root = sessions('everything', report=reports.append)
    lane = rec.stream('rx', rate=FS, clock='card')
    flush(rec)
    jobs = []
    enqueue = rec._enqueue
    monkeypatch.setattr(rec, '_enqueue', lambda fn, **kw: (jobs.append(fn), enqueue(fn, **kw)))
    # Two seconds of blocks, each one sample after the last ended.
    _feed(lane, 0, 750 * (BLOCK + 1), step=BLOCK + 1)
    flush(rec)
    # The space check rides every batch the drainer takes, gaps or none, so
    # its count is this machine's scheduling rather than the gaps' cost.
    assert len([job for job in jobs if job != rec._check_space]) < 40
    bounded(rec.close)
    assert len([r for r in reports if 'coalesced' in r]) <= 3
    assert len(index(root)['segments']) <= 5


def test_the_callback_publishes_clock_facts_without_writing_the_row(sessions, monkeypatch, no_timer):
    rec, _ = sessions('everything', attached_children=False)
    lane = rec.stream('rx', rate=FS, clock='adc')
    held, release = threading.Event(), threading.Event()
    consume = rec._consume
    def stalled(*args):
        held.set()
        release.wait(TIMEOUT)
        consume(*args)
    monkeypatch.setattr(rec, '_consume', stalled)
    for n in range(3):
        adc = n * BLOCK / FS
        lane.push_adc(pcm(n * BLOCK, (n + 1) * BLOCK), NS(inputBufferAdcTime=adc, currentTime=adc),
                      captured_at=1000. + adc)
    assert held.wait(TIMEOUT)
    try:
        assert 'monotonic_origin' not in lane.row and 'adc_origin' not in lane.row
        assert lane.timing()['monotonic_origin'] == pytest.approx(1000.)
    finally:
        release.set()
    flush(rec)
    assert lane.row['monotonic_origin'] == pytest.approx(1000.)
    assert lane.row['clock_uncertainty_s'] is None


def test_stamps_without_converter_time_place_blocks_by_count(sessions):
    rec, _ = sessions('everything', attached_children=False)
    lane = rec.stream('rx', rate=FS, clock='card')
    for n in range(40):
        lane.push_adc(pcm(0, BLOCK), None, captured_at=1000. + n * 0.01)
    lane.push_adc(pcm(0, BLOCK), None, overflow=True, captured_at=1000.5)
    lane.push_adc(pcm(0, BLOCK), None, captured_at=1000.51)
    flush(rec)
    first = 40 * BLOCK
    assert lane.gaps == [[first, first + 1]]
    assert lane.position == 42 * BLOCK + 1
    assert lane.row['uncertain']['from_sample'] == first
    # Loss re-anchors the clock on the first block after it.
    assert lane.row['monotonic_origin'] == pytest.approx(1000.5 - (first + 1) / FS)
