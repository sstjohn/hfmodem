# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Contact disposition and availability follow durable storage facts."""
import json
import os
from pathlib import Path
import stat
import threading

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_support import (
    RATE, TIMEOUT, audio_files, bounded, flush, index, interval, module, pcm,
    push, same_path, samples, snapshot,
)


def test_failed_keep_marker_preserves_lock_and_unfinalized_audio(sessions, monkeypatch):
    published, reports, stores = [], [], []
    rec, root = sessions(publish=published.append, report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, 16)
    directory = root / index(root)['contacts']['c1']['directory']
    replace = os.replace
    def fail_keep(source, target, **kwargs):
        if (same_path(target, directory / 'candidate.json')
                and json.loads(Path(source).read_text())['state'] == 'keep'):
            raise OSError('synthetic keep marker failure')
        return replace(source, target, **kwargs)
    monkeypatch.setattr(os, 'replace', fail_keep)
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: stores.append(a))
    rec.end('c1', {'card': 16})
    push(rec, lane, 16, 56)
    bounded(rec.close)
    try:
        assert reports and not stores
        assert published == []
        assert index(root)['contacts']['c1']['state'] == 'pending'
        assert json.loads((directory / 'candidate.json').read_text())['state'] == 'undecided'
        before = snapshot(directory)
        with pytest.raises(ValueError, match='active owner'):
            module().discard_candidate(directory)
        assert snapshot(directory) == before
    finally:
        lock = rec.contacts['c1']['lock']
        if lock is not None:
            lock.close()


def test_off_publishes_metadata_and_index_without_audio(sessions, monkeypatch):
    published, threads = [], []
    rec, root = sessions('off', publish=lambda row: published.append((row, threading.get_ident())))
    lane = rec.stream('rx', rate=RATE, clock='card')
    monkeypatch.setattr(module().np, 'array', lambda *a, **kw: pytest.fail('off copied PCM'))
    lane.push(1000, pcm(1000, 1008))
    rec.begin('c1', {'card': 1000}, provisional=True, peer='')
    rec.enrich('c1', peer='K9ZZT')
    rec.end('c1', {'card': 1008}, reason='owner teardown')
    rec.end('c1', {'card': 1008})
    rec.after_pending(lambda: threads.append(threading.get_ident()))
    bounded(rec.close)
    assert len(published) == 1
    row, worker = published[0]
    assert worker == threads[0] != threading.get_ident()
    assert row['peer'] == 'K9ZZT' and row['state'] == 'partial'
    assert row['positions'] == {'card': 1000} and row['end_positions'] == {'card': 1008}
    assert row['lanes'] == {} and index(root)['contacts']['c1'] == row
    assert lane.ring is None and lane.writer is None and not hasattr(rec, '_pcm')
    assert [p.relative_to(root) for p in root.rglob('*')] == [Path('recording.json')]


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_contact_before_first_pcm_records_only_available_origin(sessions, policy):
    rec, root = sessions(policy)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 1000})
    flush(rec)
    push(rec, lane, 1100, 1116)
    rec.end('c1', {'card': 1116})
    push(rec, lane, 1116, 1156)
    bounded(rec.close)
    ref = interval(root)
    assert ref['requested'] == [520, 1156]
    assert ref['actual'] == [1100, 1156]
    assert ref['missing_pre_roll'] == 580
    assert ref['gaps'] == [[1000, 1100]]
    assert ref['state'] == 'partial'
    if policy == 'contacts':
        np.testing.assert_array_equal(samples(root, ref), np.arange(1100, 1156))


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_unreadable_storage_cannot_publish_finalized_audio(sessions, monkeypatch, policy):
    published = []
    monkeypatch.setattr(wav, 'store', lambda path, **kw: {'wav': Path(path).name,
                                                       'kept': 'unreadable: synthetic failure'})
    rec, root = sessions(policy, publish=published.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 16)
    rec.begin('c1', {'card': 16})
    rec.end('c1', {'card': 16})
    push(rec, lane, 16, 56)
    bounded(rec.close)
    assert index(root)['segments'][0]['state'] == 'partial'
    assert interval(root)['state'] == 'partial'
    assert all(row['lanes']['rx']['state'] != 'finalized' for row in published)
    assert audio_files(root)


def test_open_continuous_segment_is_pending_with_fixed_bounds(sessions):
    published = []
    rec, root = sessions('everything', publish=published.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 16)
    rec.begin('c1', {'card': 16})
    rec.end('c1', {'card': 16})
    push(rec, lane, 16, 56)
    before = interval(root)
    assert before['state'] == 'pending' and published[0]['lanes']['rx']['state'] == 'pending'
    bounded(rec.close)
    after = interval(root)
    assert after['state'] == 'finalized'
    for field in ('requested', 'actual', 'gaps', 'missing_pre_roll', 'missing_post_roll'):
        assert after[field] == before[field]


def test_pending_overflow_keeps_contact_audio_pending_until_drain(sessions, monkeypatch):
    entered, release, consumed = (threading.Event() for _ in range(3))
    store, write = wav.store, wav.SegmentedWriter.writeframes
    def blocked(path, **kwargs):
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
        for first in range(0, 12 * 60 * RATE, 2 * RATE):
            if first == 10 * 60 * RATE:
                rec.begin('c1', {'card': first})
                rec.end('c1', {'card': first})
            consumed.clear()
            lane.push(first, pcm(first, first + 2 * RATE))
            assert consumed.wait(TIMEOUT)
            if first + 2 * RATE == 60 * RATE:
                assert entered.wait(TIMEOUT)
        assert entered.wait(TIMEOUT)
        with rec._state_lock:
            assert rec.index['contacts']['c1']['lanes']['rx']['state'] == 'pending'
    finally:
        release.set()
    flush(rec)
    assert interval(root)['state'] == 'pending'
    bounded(rec.close)
    assert interval(root)['state'] == 'finalized'


def test_every_index_replacement_syncs_file_and_directory(sessions, monkeypatch):
    rec, root = sessions('everything')
    target = root / 'recording.json'
    events = []
    replace, fsync = os.replace, os.fsync
    def replaced(source, destination, **kwargs):
        inode = Path(source).stat().st_ino
        result = replace(source, destination, **kwargs)
        if same_path(destination, target):
            events.append(('replace', inode))
        return result
    def synced(fd):
        facts = os.fstat(fd)
        if stat.S_ISREG(facts.st_mode):
            events.append(('file', facts.st_ino))
        elif stat.S_ISDIR(facts.st_mode) and facts.st_ino == root.stat().st_ino:
            events.append(('directory', facts.st_ino))
        return fsync(fd)
    monkeypatch.setattr(os, 'replace', replaced)
    monkeypatch.setattr(os, 'fsync', synced)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 60 * RATE)
    bounded(rec.close)
    replacements = [(i, inode) for i, (kind, inode) in enumerate(events) if kind == 'replace']
    assert replacements
    for i, inode in replacements:
        assert events[i - 1] == ('file', inode)
        # Windows cannot fsync a directory; NTFS journals the rename.
        if os.name != 'nt':
            assert events[i + 1] == ('directory', root.stat().st_ino)
