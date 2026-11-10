# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Publication ownership, retention decisions and unavailable source clocks."""
import errno
import json
import os
from pathlib import Path
import threading

import pytest

from hfmodem.core import capture_identity as ci, wav
from hfmodem.tests.station.recording_support import (
    RATE, TIMEOUT, audio_files, bounded, flush, index, interval, module, pcm, push,
)


def test_keep_is_frozen_before_manifest_preparation(sessions, monkeypatch):
    reports = []
    rec, root = sessions(report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, 58 * RATE)
    contact = rec.contacts['c1']
    original = rec._candidate_files
    def failed(c, writer, additional=0):
        if additional >= 5 * RATE:
            raise OSError('synthetic manifest preparation failure')
        return original(c, writer, additional)
    monkeypatch.setattr(rec, '_candidate_files', failed)
    rec.end('c1', {'card': 58 * RATE}, keep=True)
    flush(rec)
    bounded(rec.close)
    try:
        assert contact['decision'] is True
        assert audio_files(contact['directory'])
        assert contact['row']['state'] == 'pending'
        with pytest.raises(ValueError, match='active owner'):
            module().discard_candidate(contact['directory'])
    finally:
        if contact['lock'] is not None:
            contact['lock'].close()


def test_any_blocked_index_publication_leaves_idle_pcm_draining(sessions, monkeypatch):
    rec, root = sessions('contacts')
    lane = rec.stream('rx', rate=RATE, clock='card')
    flush(rec)
    entered, release, consumed = (threading.Event() for _ in range(3))
    durable, consume = module()._durable_json, rec._consume
    def blocked(path, facts):
        if path.name == 'recording.json' and not entered.is_set():
            entered.set()
            assert release.wait(TIMEOUT)
        return durable(path, facts)
    def observed(*args):
        consume(*args)
        consumed.set()
    monkeypatch.setattr(module(), '_durable_json', blocked)
    monkeypatch.setattr(rec, '_consume', observed)
    try:
        lane.push(0, pcm(0, RATE))
        assert entered.wait(TIMEOUT)
        for first in range(RATE, 5 * RATE, RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + RATE))
            assert consumed.wait(TIMEOUT / 2), 'publication stalled PCM'
    finally:
        release.set()
    bounded(rec.close)
    assert index(root)['lanes']['rx']['gaps'] == []


def test_blocked_owned_sidecar_leaves_rollover_draining(sessions, monkeypatch):
    rec, root = sessions('everything')
    capture = ci.allocate({'capture': root / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav',
                                 clock_id='card', continuous=True)
    lane = rec.stream('rx', rate=RATE, clock='card', ownership=owner)
    flush(rec)
    entered, release, consumed = (threading.Event() for _ in range(3))
    update, consume = owner.sidecar.slot.update, rec._consume
    def blocked(facts):
        if any(v.get('capture_state') == 'finalized'
               for v in facts.get('segment_artifacts', {}).values()) and not entered.is_set():
            entered.set()
            assert release.wait(TIMEOUT)
        return update(facts)
    def observed(*args):
        consume(*args)
        consumed.set()
    monkeypatch.setattr(owner.sidecar.slot, 'update', blocked)
    monkeypatch.setattr(rec, '_consume', observed)
    try:
        for first in range(0, 60 * RATE, RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + RATE))
            assert consumed.wait(TIMEOUT)
        assert entered.wait(TIMEOUT)
        for first in range(60 * RATE, 124 * RATE, RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + RATE))
            assert consumed.wait(TIMEOUT / 2), 'owned metadata stalled rollover'
        assert len(audio_files(root)) == 3
        with rec._state_lock:
            staged = rec.index['segments'][1]['pending_source']
            assert (root / staged).is_file()
    finally:
        release.set()
    bounded(rec.close)
    assert index(root)['lanes']['rx']['gaps'] == []

    assert not list(root.rglob('*.pending.wav'))
    assert sum(d['samples'] for d in index(root)['segments']) == 124 * RATE
    for descriptor in index(root)['segments']:
        assert descriptor['state'] == 'finalized'
        assert 'pending_source' not in descriptor
        assert ci.audio_metadata(root / descriptor['source'])['capture_state'] == 'finalized'


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_dropped_first_block_cannot_finalize_false_coverage(sessions, policy):
    rec, root = sessions(policy)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 1000})
    rec.end('c1', {'card': 1008})
    flush(rec)
    lane.push(1100, pcm(1100, 1120))
    flush(rec)
    bounded(rec.close)
    ref = interval(root)
    assert ref['actual'][0] == ref['actual'][1]
    assert ref['gaps'] == [[1000, 1048]]
    assert ref['missing_post_roll'] == 40
    assert ref['state'] == 'partial'


def test_initial_continuous_enospc_is_isolated(sessions, monkeypatch):
    reports = []
    rec, root = sessions('everything', report=reports.append)
    original = rec._writer
    def failed(lane, *args, **kwargs):
        if lane.name == 'rx':
            raise OSError(errno.ENOSPC, 'synthetic initial sink full')
        return original(lane, *args, **kwargs)
    monkeypatch.setattr(rec, '_writer', failed)
    rx = rec.stream('rx', rate=RATE, clock='card')
    tx = rec.stream('tx', rate=RATE, clock='tx')
    push(rec, rx, 1000, 1024)
    push(rec, tx, 0, 24)
    bounded(rec.close)
    assert rx.disabled and not tx.disabled
    assert rx.consumed == 1024
    assert index(root)['lanes']['rx']['gaps'] == [[1000, 1024]]
    assert index(root)['lanes']['rx']['state'] == 'disabled'
    assert len([r for r in reports if 'synthetic initial sink full' in r]) == 1


def test_discard_is_terminal_in_run_index(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, RATE)
    rec.end('c1', {'card': RATE}, keep=False)
    bounded(rec.close)
    assert index(root)['contacts']['c1']['state'] == 'discarded'
    assert audio_files(root) == []


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_all_durable_publication_belongs_to_storage_worker(sessions, monkeypatch, policy):
    rec, root = sessions(policy)
    capture = ci.allocate({'capture': root / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav',
                                 clock_id='card', continuous=True)
    calls = []
    marking = threading.local()
    mark_origin = module().Lane._mark_origin
    def marker(lane):
        marking.active = True
        try:
            return mark_origin(lane)
        finally:
            marking.active = False
    monkeypatch.setattr(module().Lane, '_mark_origin', marker)
    def watch(label, fn):
        def checked(*args, **kwargs):
            if not getattr(marking, 'active', False):
                calls.append((label, threading.current_thread().name))
            else:
                assert label == 'fsync'
                assert threading.current_thread().name != 'recording-pcm'
            return fn(*args, **kwargs)
        return checked
    for obj, name in [(os, 'fsync'), (os, 'replace'), (wav, '_write_json'),
                      (wav, '_sha256'), (module(), '_durable_json'),
                      (ci.RecordingMetadata, 'update'), (ci, '_publish'),
                      (json, 'dump'), (Path, 'write_text')]:
        monkeypatch.setattr(obj, name, watch(name, getattr(obj, name)))
    lane = rec.stream('rx', rate=RATE, clock='card',
                      ownership=owner if policy == 'everything' else None)
    rec.begin('c1', {'card': 0}, provisional=policy == 'contacts')
    push(rec, lane, 0, 62 * RATE)
    rec.end('c1', {'card': 62 * RATE})
    push(rec, lane, 62 * RATE, 67 * RATE)
    bounded(rec.close)
    assert calls
    assert set(thread for _, thread in calls) == {'recording-store'}, calls


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_publication_lock_never_nests_with_drainer_locks(sessions, monkeypatch, policy):
    rec, root = sessions(policy)
    capture = ci.allocate({'capture': root / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav',
                                 clock_id='card', continuous=True)
    edges, held = set(), threading.local()
    class CheckedLock:
        def __init__(self, name, lock):
            self.name, self.lock = name, lock
        def acquire(self, *args, **kwargs):
            if self.name in ('_publication_lock', 'ownership'):
                assert threading.current_thread() is not rec._pcm_thread
            stack = getattr(held, 'stack', [])
            edges.update((name, self.name) for name in stack if name != self.name)
            result = self.lock.acquire(*args, **kwargs)
            if result:
                held.stack = stack + [self.name]
            return result
        def release(self):
            held.stack.pop()
            self.lock.release()
        def __enter__(self):
            self.acquire()
            return self
        def __exit__(self, *args):
            self.release()
    for name in ('_input_lock', '_state_lock', '_publication_lock'):
        monkeypatch.setattr(rec, name, CheckedLock(name, getattr(rec, name)))
    monkeypatch.setattr(owner.sidecar, 'lock', CheckedLock('ownership', owner.sidecar.lock))
    lane = rec.stream('rx', rate=RATE, clock='card',
                      ownership=owner if policy == 'everything' else None)
    rec.begin('c1', {'card': 0}, provisional=policy == 'contacts')
    push(rec, lane, 0, 64 * RATE)
    rec.end('c1', {'card': 64 * RATE})
    bounded(rec.close)
    assert not any('_publication_lock' in edge and any(
        lock in edge for lock in ('_input_lock', '_state_lock')) for edge in edges), edges
    def visit(node, path):
        assert node not in path, edges
        for source, target in edges:
            if source == node:
                visit(target, path | {node})
    for node in {n for edge in edges for n in edge}:
        visit(node, set())


@pytest.mark.parametrize('failure', ['publisher', 'storage'])
def test_failed_kept_contact_publication_retains_lifetime_protection(sessions, monkeypatch, failure):
    def failed(row):
        raise OSError('synthetic publication failure')
    if failure == 'storage':
        monkeypatch.setattr(wav, 'store', lambda *args, **kwargs: failed(None))
    rec, root = sessions(publish=failed if failure == 'publisher' else None,
                         report=lambda message: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, RATE)
    rec.end('c1', {'card': RATE})
    push(rec, lane, RATE, 6 * RATE)
    bounded(rec.close)
    contact = rec.contacts['c1']
    try:
        assert contact['decision'] is True and contact['lock'] is not None
        assert index(root)['contacts']['c1']['state'] == 'pending'
        assert json.loads((contact['directory'] / 'candidate.json').read_text())['state'] == 'keep'
        assert audio_files(contact['directory'])
    finally:
        if contact['lock'] is not None:
            contact['lock'].close()


@pytest.mark.parametrize('overlapping', [False, True])
def test_candidate_preparation_is_bounded_and_does_not_stall_other_lanes(sessions, monkeypatch, overlapping):
    rec, root = sessions()
    rx = rec.stream('rx', rate=RATE, clock='card')
    tx = rec.stream('tx', rate=RATE, clock='output')
    if overlapping:
        rec.begin('c0', {'card': 0})
        flush(rec)
    entered, release, consumed = (threading.Event() for _ in range(3))
    durable, consume = module()._durable_json, rec._consume
    def blocked(path, facts):
        if path.name == 'candidate.identity':
            entered.set()
            assert release.wait(TIMEOUT)
        return durable(path, facts)
    def observed(*args):
        consume(*args)
        consumed.set()
    monkeypatch.setattr(module(), '_durable_json', blocked)
    monkeypatch.setattr(rec, '_consume', observed)
    rec.begin('c1', {'card': 0}, provisional=True)
    try:
        assert entered.wait(TIMEOUT)
        for first in range(0, 4 * RATE, RATE):
            for lane in (rx, tx):
                consumed.clear()
                lane.push(first, pcm(first, first + RATE))
                assert consumed.wait(TIMEOUT)
        with rec._state_lock:
            assert rec.contacts['c1']['buffers']['rx']['samples'] <= 2 * RATE
        assert len(audio_files(root)) == int(overlapping)
        assert tx.consumed == 4 * RATE
        rec.end('c1', {'card': 4 * RATE})
    finally:
        release.set()
    bounded(rec.close)
    ref = interval(root)
    assert ref['gaps'] == [[2 * RATE, 4 * RATE]]
    assert ref['actual'] == [0, 2 * RATE]
    assert ref['missing_post_roll'] == 5 * RATE

    if overlapping:
        assert interval(root, 'c0')['gaps'] == []
        assert interval(root, 'c0')['actual'] == [0, 4 * RATE]


def test_no_origin_at_close_has_empty_coverage_and_full_missing_post_roll(sessions):
    rec, root = sessions('everything')
    rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 1000})
    rec.end('c1', {'card': 1008})
    bounded(rec.close)
    ref = interval(root)
    assert ref['actual'] == [1000, 1000]
    assert ref['gaps'] == [[1000, 1048]]
    assert ref['missing_post_roll'] == 5 * RATE


def test_failed_owned_reservation_keeps_staged_pcm_unfinalized(sessions, monkeypatch):
    rec, root = sessions('everything', report=lambda message: None)
    capture = ci.allocate({'capture': root / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav',
                                 clock_id='card', continuous=True)
    lane = rec.stream('rx', rate=RATE, clock='card', ownership=owner)
    push(rec, lane, 0, RATE)
    def failed(ticket):
        raise ValueError('synthetic reservation failure')
    monkeypatch.setattr(ci.QueuedSegment, 'reserve', failed)
    stores, store = [], wav.store
    monkeypatch.setattr(wav, 'store', lambda path, **kw: stores.append(path) or store(path, **kw))
    push(rec, lane, RATE, 64 * RATE)
    bounded(rec.close)
    first, second = index(root)['segments']
    assert first['state'] == 'finalized' and second['state'] == 'partial'
    assert stores == [lane.writer.paths[0]]
    assert (root / second['pending_source']).is_file()
    assert not (root / second['source']).exists()


@pytest.mark.parametrize('failure', [False, True])
def test_kept_candidate_lock_covers_pending_descriptor_disposition(sessions, monkeypatch, failure):
    rec, root = sessions(report=lambda message: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, RATE)
    rec.end('c1', {'card': RATE})
    flush(rec)
    monkeypatch.setattr(rec, '_admit', lambda job: None)
    push(rec, lane, RATE, 6 * RATE)
    contact = rec.contacts['c1']
    assert contact['lock'] is not None
    assert index(root)['contacts']['c1']['state'] == 'pending'
    if failure:
        def failed(*args, **kwargs):
            raise OSError('synthetic pending storage failure')
        monkeypatch.setattr(wav, 'store', failed)
    bounded(rec.close)
    try:
        assert (contact['lock'] is not None) == failure
        assert index(root)['contacts']['c1']['state'] == ('pending' if failure else 'finalized')
        assert audio_files(contact['directory'])
    finally:
        if contact['lock'] is not None:
            contact['lock'].close()


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_close_failure_reconciles_gaps_before_freezing_bounds(sessions, policy):
    rec, root = sessions(policy, report=lambda message: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0})
    push(rec, lane, 0, RATE)
    rec.end('c1', {'card': RATE})
    flush(rec)
    writer = rec.contacts['c1']['writers'].get('rx', lane.writer)
    handle = writer._handle
    class FailedHandle:
        def close(self):
            handle.close()
            raise OSError(errno.ENOSPC, 'synthetic final flush failure')
    writer._handle = FailedHandle()
    bounded(rec.close)
    ref = interval(root)
    assert ref['gaps'] == [[0, RATE]]
    assert ref['actual'][0] == ref['actual'][1]
    assert ref['missing_post_roll'] == 5 * RATE
    assert ref['state'] == 'partial'
