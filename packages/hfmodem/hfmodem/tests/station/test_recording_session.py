# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Policy selection preserves absolute source clocks and available padding."""
from pathlib import Path
import threading
import weakref

import numpy as np
import pytest

from hfmodem.tests.station.recording_support import (
    RATE, audio_files, bounded, flush, index, interval, pcm, push, samples,
    snapshot,
)


def test_off_idle_has_no_files_recording_buffers_or_workers(sessions, monkeypatch):
    threads, blocks = [], []
    start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, 'start', lambda t: threads.append(t) or start(t))
    def allocated(*args, **kwargs):
        pytest.fail('off allocated a recording buffer')
    monkeypatch.setattr(np, 'empty', allocated)
    monkeypatch.setattr(np, 'zeros', allocated)
    rec, root = sessions('off')
    lane = rec.stream('rx', rate=RATE, clock='card')
    for first in range(0, 1000, RATE):
        block = pcm(first, first + RATE)
        blocks.append(weakref.ref(block))
        lane.push(first, block)
    del block
    rec.close()
    assert lane.position == 1000
    assert not root.exists() or list(root.rglob('*')) == []
    assert threads == []
    assert all(block() is None for block in blocks)


def test_contacts_idle_audio_has_no_disk_spool(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 90 * RATE)
    assert lane.position == 90 * RATE
    assert audio_files(root) == []
    assert not any(p.suffix in ('.pcm', '.raw', '.part') for p in root.rglob('*'))
    bounded(rec.close)
    assert audio_files(root) == []


@pytest.mark.parametrize('start', [2 * RATE, 60 * RATE, 75 * RATE + 3, 130 * RATE + 1])
def test_exact_ring_wrap_and_pre_post_roll(sessions, start):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, start)
    rec.begin('c1', {'card': start})
    end = start + 2 * RATE + 1
    push(rec, lane, start, end)
    rec.end('c1', {'card': end})
    push(rec, lane, end, end + 6 * RATE)
    ref = interval(root)
    actual = [max(0, start - 60 * RATE), end + 5 * RATE]
    assert ref['requested'] == [start - 60 * RATE, end + 5 * RATE]
    assert ref['actual'] == actual
    assert ref['rate'] == RATE and ref['gaps'] == []
    assert ref['missing_pre_roll'] == max(0, 60 * RATE - start)
    np.testing.assert_array_equal(samples(root, ref), np.arange(*actual) % 30000)


def test_late_lane_origin_does_not_invent_pre_roll(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 1000, 1024)
    rec.begin('c1', {'card': 1024})
    rec.end('c1', {'card': 1024})
    push(rec, lane, 1024, 1072)
    ref = interval(root)
    assert ref['requested'] == [544, 1064]
    assert ref['actual'] == [1000, 1064]
    assert ref['missing_pre_roll'] == 456
    np.testing.assert_array_equal(samples(root, ref), np.arange(1000, 1064))


def test_close_records_short_post_roll_without_padding(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 24)
    rec.begin('c1', {'card': 24})
    push(rec, lane, 24, 40)
    rec.end('c1', {'card': 40})
    push(rec, lane, 40, 43)
    flush(rec)
    bounded(rec.close)
    ref = interval(root)
    assert ref['requested'] == [-456, 80]
    assert ref['actual'] == [0, 43]
    assert ref['missing_post_roll'] == 37
    assert ref['state'] == 'partial'
    np.testing.assert_array_equal(samples(root, ref), np.arange(43))


def test_barrier_does_not_wait_for_future_post_roll(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 16)
    rec.begin('c1', {'card': 16})
    rec.end('c1', {'card': 16})
    flush(rec)
    assert interval(root)['state'] != 'finalized'
    push(rec, lane, 16, 56)
    assert interval(root)['actual'] == [0, 56]


def test_duplicate_events_and_overlapping_contacts_have_separate_sinks(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 600)
    rec.begin('c1', {'card': 600})
    rec.begin('c1', {'card': 600})
    push(rec, lane, 600, 616)
    rec.begin('c2', {'card': 616})
    rec.end('c1', {'card': 616})
    rec.end('c1', {'card': 616})
    push(rec, lane, 616, 680)
    rec.end('c2', {'card': 680})
    push(rec, lane, 680, 728)
    first, second = interval(root), interval(root, 'c2')
    assert set(index(root)['contacts']) == {'c1', 'c2'}
    assert first['actual'] == [120, 656]
    assert second['actual'] == [136, 720]
    assert set(first['segments']).isdisjoint(second['segments'])
    np.testing.assert_array_equal(samples(root, first), np.arange(120, 656))
    np.testing.assert_array_equal(samples(root, second), np.arange(136, 720))


def test_independent_lane_clocks_and_native_rates(sessions):
    rec, root = sessions()
    rx = rec.stream('rx', rate=8, clock='card')
    tx = rec.stream('submitted-tx', rate=12, clock='output')
    rx.push(1000, pcm(1000, 1008))
    tx.push(2000, pcm(2000, 2012))
    flush(rec)
    rec.begin('c1', {'card': 1008, 'output': 2012})
    rec.end('c1', {'card': 1008, 'output': 2012})
    for _ in range(5):
        rx.push(rx.position, pcm(rx.position, rx.position + 8))
        tx.push(tx.position, pcm(tx.position, tx.position + 12))
        flush(rec)
    assert interval(root, lane='rx')['actual'] == [1000, 1048]
    assert interval(root, lane='submitted-tx')['actual'] == [2000, 2072]
    assert interval(root, lane='submitted-tx')['rate'] == 12


def test_upstream_discontinuity_is_an_absolute_gap(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 16)
    rec.begin('c1', {'card': 16})
    push(rec, lane, 24, 40)
    rec.end('c1', {'card': 40})
    push(rec, lane, 40, 80)
    ref = interval(root)
    assert lane.position == 80
    assert ref['actual'] == [0, 80]
    assert ref['gaps'] == [[16, 24]]
    np.testing.assert_array_equal(samples(root, ref), np.r_[0:16, 24:80])


def test_everything_publishes_fixed_references_without_contact_copies(sessions):
    published = []
    rec, root = sessions('everything', source='station', publish=published.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 600)
    rec.begin('c1', {'card': 600})
    push(rec, lane, 600, 640)
    rec.end('c1', {'card': 640})
    push(rec, lane, 640, 680)
    ref = interval(root)
    assert ref['kind'] == 'interval'
    assert ref['requested'] == ref['actual'] == [120, 680]
    assert ref['lane'] == 'rx' and ref['rate'] == RATE
    assert ref['source_id'] and ref['index_id']
    assert not Path(ref['source']).is_absolute()
    assert (root / ref['source']).with_suffix('.json').exists()
    assert len(audio_files(root)) == 2
    assert len(published) == 1 and published[0]['id'] == 'c1'
    frozen = {k: v for k, v in ref.items() if k not in ('state', 'segments')}
    push(rec, lane, 680, 1000)
    bounded(rec.close)
    after = interval(root)
    assert {k: v for k, v in after.items() if k not in ('state', 'segments')} == frozen
    assert after['state'] == 'finalized'
    assert len(published) == 1
    assert index(root)['policy'] == 'everything' and index(root)['source'] == 'station'


@pytest.mark.parametrize('policy', ['off', 'contacts', 'everything'])
def test_no_prior_finalized_audio_is_deleted(sessions, tmp_path, policy):
    prior = tmp_path / 'prior'
    prior.mkdir()
    (prior / 'kept.wav').write_bytes(b'finalized audio')
    (prior / 'kept.json').write_text('{"capture_state":"finalized"}')
    before = snapshot(prior)
    rec, _ = sessions(policy)
    lane = rec.stream('rx', rate=RATE, clock='card')
    lane.push(0, pcm(0, 8))
    rec.begin('c1', {'card': 8}, provisional=True)
    rec.end('c1', {'card': 8}, keep=False)
    bounded(rec.close)
    assert snapshot(prior) == before


def test_finalized_contact_audio_is_immutable_after_later_discard_and_close(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 16)
    rec.begin('c1', {'card': 16})
    rec.end('c1', {'card': 16})
    push(rec, lane, 16, 56)
    kept = {p: p.read_bytes() for p in audio_files(root)}
    sidecars = {p.with_suffix('.json'): p.with_suffix('.json').read_bytes() for p in kept}
    rec.begin('c2', {'card': 56}, provisional=True)
    push(rec, lane, 56, 72)
    rec.end('c2', {'card': 72}, keep=False)
    bounded(rec.close)
    assert kept
    assert all(p.read_bytes() == data for p, data in (kept | sidecars).items())


def test_off_barriers_use_storage_worker_without_audio_or_files(sessions):
    rec, root = sessions('off')
    lane = rec.stream('rx', rate=RATE, clock='card')
    lane.push(0, pcm(0, 8))
    callbacks = []
    rec.after_pending(lambda: callbacks.append(threading.get_ident()))
    flush(rec)
    bounded(rec.close)
    assert callbacks and callbacks[0] != threading.get_ident()
    assert lane.ring is None
    assert not root.exists()
