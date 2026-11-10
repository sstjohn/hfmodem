# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Candidate state survives crashes and cleanup stays within its ownership.

Crash images contain only bytes durable at an ordered barrier; restoring the
same exact path after owner shutdown leaves no live lifetime lock.
"""
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import threading

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_support import (
    RATE, TIMEOUT, audio_files, bounded, flush, index, interval, module, pcm,
    push, restore, same_path, samples, snapshot,
)


def candidate(root, contact='c1'):
    return root / index(root)['contacts'][contact]['directory']


def state(directory):
    return json.loads((directory / 'candidate.json').read_text())


def cli(directory):
    try:
        result = module().main(['discard-candidate', str(directory)])
    except SystemExit as exc:
        return exc.code
    return 0 if result is None else result


def abandoned(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 16)
    rec.begin('c1', {'card': 16}, provisional=True)
    push(rec, lane, 16, 32)
    directory = candidate(root)
    saved = snapshot(directory)
    bounded(rec.close)
    restore(directory, saved)
    return root, directory


@pytest.mark.parametrize('keep', [False, True])
def test_candidate_longer_than_ring_and_segments_resolves_once(sessions, monkeypatch, keep):
    stores, published = [], []
    store = wav.store
    monkeypatch.setattr(wav, 'store', lambda p, **kw: stores.append(Path(p)) or store(p, **kw))
    rec, root = sessions(publish=published.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 80 * RATE)
    rec.begin('c1', {'card': 80 * RATE}, provisional=True)
    push(rec, lane, 80 * RATE, 220 * RATE)
    directory = candidate(root)
    assert directory != root and root in directory.parents
    assert state(directory)['state'] == 'undecided'
    assert len(audio_files(directory)) >= 3
    assert stores == [] and published == []
    rec.end('c1', {'card': 220 * RATE}, keep=keep)
    rec.end('c1', {'card': 221 * RATE}, keep=not keep)
    push(rec, lane, 220 * RATE, 226 * RATE)
    bounded(rec.close)
    if keep:
        ref = interval(root)
        assert ref['actual'] == [20 * RATE, 225 * RATE]
        np.testing.assert_array_equal(samples(root, ref), np.arange(20 * RATE, 225 * RATE))
        assert stores and len(published) == 1
        assert state(directory)['state'] == 'keep'
    else:
        assert audio_files(root) == []
        assert stores == [] and published == []


def test_candidate_identity_and_expected_paths_exist_before_pcm(sessions, monkeypatch):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    checks = []
    write = wav.SegmentedWriter.writeframes
    def checked(out, data):
        directory = candidate(root)
        facts = state(directory)
        assert facts['contact_id'] == 'c1'
        assert facts['candidate_id'] and facts['state'] == 'undecided'
        assert facts['owned_files']
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts
                   for name in facts['owned_files'])
        checks.append(facts)
        return write(out, data)
    monkeypatch.setattr(wav.SegmentedWriter, 'writeframes', checked)
    rec.begin('c1', {'card': 0}, provisional=True)
    lane.push(0, pcm(0, 8))
    flush(rec)
    assert checks


def test_keep_marker_is_atomic_and_precedes_finalization(sessions, monkeypatch):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, 70 * RATE)
    directory = candidate(root)
    seen, replacements = [], []
    store, replace = wav.store, os.replace
    def checked(path, **kwargs):
        assert state(directory)['state'] == 'keep'
        assert replacements
        seen.append(path)
        return store(path, **kwargs)
    def replaced(source, target, **kwargs):
        if same_path(target, directory / 'candidate.json'):
            assert state(directory)['state'] in ('undecided', 'keep')
            replacements.append(json.loads(Path(source).read_text()))
        return replace(source, target, **kwargs)
    monkeypatch.setattr(wav, 'store', checked)
    monkeypatch.setattr(os, 'replace', replaced)
    rec.end('c1', {'card': 70 * RATE}, keep=True)
    push(rec, lane, 70 * RATE, 75 * RATE)
    bounded(rec.close)
    assert seen
    assert any(row['state'] == 'keep' for row in replacements)


def test_discard_candidate_cleans_crash_before_keep_without_touching_foreign_files(sessions):
    root, directory = abandoned(sessions)
    foreign = directory / 'unowned.wav'
    foreign.write_bytes(b'unrelated audio')
    outside = root / 'unrelated.json'
    outside.write_bytes(b'unrelated metadata')
    assert state(directory)['state'] == 'undecided'
    assert cli(directory) == 0
    assert foreign.read_bytes() == b'unrelated audio'
    assert outside.read_bytes() == b'unrelated metadata'
    assert not (directory / 'candidate.json').exists()
    assert audio_files(directory) == [foreign]


def test_discard_candidate_module_entry_point(sessions, monkeypatch):
    _, directory = abandoned(sessions)
    monkeypatch.setattr(sys, 'argv', ['hfmodem.core.recording', 'discard-candidate', str(directory)])
    monkeypatch.delitem(sys.modules, 'hfmodem.core.recording')
    try:
        runpy.run_module('hfmodem.core.recording', run_name='__main__')
    except SystemExit as exc:
        assert exc.code in (None, 0)
    assert audio_files(directory) == []
    assert not (directory / 'candidate.json').exists()


def test_discard_candidate_refuses_active_owner(sessions, capsys):
    rec, root = sessions()
    rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    flush(rec)
    directory = candidate(root)
    before = snapshot(directory)
    assert cli(directory) != 0
    assert any(word in capsys.readouterr().err.lower() for word in ('active', 'lock', 'owner'))
    assert snapshot(directory) == before


def test_discard_candidate_refuses_directory_without_candidate_identity(tmp_path, capsys):
    module()
    directory = tmp_path / 'unknown'
    directory.mkdir()
    (directory / 'receive.wav').write_bytes(b'unowned audio')
    before = snapshot(directory)
    assert cli(directory) != 0
    assert capsys.readouterr().err.strip()
    assert snapshot(directory) == before


@pytest.mark.parametrize('kind', ['keep', 'unknown', 'identity', 'escape', 'symlink'])
def test_discard_candidate_refusals_preserve_all_bytes(sessions, tmp_path, capsys, kind):
    _, directory = abandoned(sessions)
    path = directory / 'candidate.json'
    facts = state(directory)
    if kind in ('keep', 'unknown'):
        facts['state'] = kind
    elif kind == 'identity':
        facts['candidate_id'] = 'different-identity'
    elif kind == 'escape':
        facts['owned_files'].append('../outside.wav')
    else:
        target = tmp_path / 'outside.wav'
        target.write_bytes(b'foreign audio')
        owned = directory / next(name for name in facts['owned_files'] if name.endswith('.wav'))
        owned.unlink()
        owned.symlink_to(target)
    path.write_text(json.dumps(facts))
    before = snapshot(tmp_path)
    assert cli(directory) != 0
    assert capsys.readouterr().err.strip()
    assert snapshot(tmp_path) == before


def test_crash_after_keep_before_storage_is_never_candidate_garbage(sessions, monkeypatch, capsys):
    entered, release = threading.Event(), threading.Event()
    store = wav.store
    def blocked(path, **kwargs):
        entered.set()
        assert release.wait(TIMEOUT)
        return store(path, **kwargs)
    monkeypatch.setattr(wav, 'store', blocked)
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, 65 * RATE)
    rec.end('c1', {'card': 65 * RATE}, keep=True)
    try:
        assert entered.wait(TIMEOUT)
        directory = candidate(root)
        saved = snapshot(directory)
        assert state(directory)['state'] == 'keep'
    finally:
        release.set()
    bounded(rec.close)
    shutil.rmtree(directory)
    restore(directory, saved)
    before = snapshot(directory)
    assert cli(directory) != 0
    message = capsys.readouterr().err.lower()
    assert 'keep' in message or 'kept' in message
    assert snapshot(directory) == before


def test_discard_candidate_retries_partial_cleanup_with_identity_removed_last(sessions, monkeypatch):
    _, directory = abandoned(sessions)
    unlink, removed = Path.unlink, []
    def interrupted(path, *args, **kwargs):
        if path.suffix == '.wav' and not removed:
            unlink(path, *args, **kwargs)
            removed.append(path)
            raise OSError('synthetic cleanup interruption')
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', interrupted)
        assert cli(directory) != 0
    assert removed and (directory / 'candidate.json').exists()
    assert cli(directory) == 0
    assert audio_files(directory) == []
    assert not (directory / 'candidate.json').exists()


def test_failed_publication_cannot_undo_keep_or_duplicate_storage(sessions, monkeypatch):
    reports, stores, publications = [], [], []
    store = wav.store
    def fail(row):
        publications.append(row)
        raise OSError('synthetic contact append failure')
    monkeypatch.setattr(wav, 'store', lambda p, **kw: stores.append(p) or store(p, **kw))
    rec, root = sessions(report=reports.append, publish=fail)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, 16)
    rec.end('c1', {'card': 16}, keep=True)
    push(rec, lane, 16, 56)
    flush(rec)
    before = {p: p.read_bytes() for p in audio_files(root)}
    rec.end('c1', {'card': 16}, keep=False)
    bounded(rec.close)
    assert before and all(p.read_bytes() == data for p, data in before.items())
    assert state(candidate(root))['state'] == 'keep'
    assert len(publications) == 1 and len(stores) == len(before)
    assert any('synthetic contact append failure' in str(r) for r in reports)


def test_keep_barrier_waits_for_all_previously_closed_candidate_segments(sessions, monkeypatch):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, 130 * RATE)
    entered, release, done = (threading.Event() for _ in range(3))
    store = wav.store
    def blocked(path, **kwargs):
        entered.set()
        assert release.wait(TIMEOUT)
        return store(path, **kwargs)
    monkeypatch.setattr(wav, 'store', blocked)
    try:
        rec.end('c1', {'card': 130 * RATE}, keep=True)
        assert entered.wait(TIMEOUT)
        rec.after_pending(done.set)
        assert not done.is_set()
    finally:
        release.set()
    assert done.wait(TIMEOUT)
    assert all(d['state'] == 'finalized' for d in index(root)['segments'])


def test_cleanup_retries_after_identity_file_removal(sessions, monkeypatch):
    _, directory = abandoned(sessions)
    unlink = Path.unlink
    def interrupted(path, *args, **kwargs):
        result = unlink(path, *args, **kwargs)
        if path.name == 'candidate.identity':
            raise OSError('synthetic identity cleanup interruption')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', interrupted)
        assert cli(directory) != 0
    assert (directory / 'candidate.json').exists()
    assert cli(directory) == 0


def test_cleanup_refuses_finalized_audio_even_in_an_undecided_manifest(sessions):
    _, directory = abandoned(sessions)
    facts = state(directory)
    owned = directory / next(name for name in facts['owned_files'] if name.endswith('.wav'))
    owned.with_suffix('.json').write_text('{"capture_state": "finalized"}')
    before = snapshot(directory)
    assert cli(directory) != 0
    assert snapshot(directory) == before


def test_candidate_lock_refuses_a_second_owner_and_leaves_the_identity_readable(tmp_path):
    from hfmodem.core import recording
    path = tmp_path / 'candidate.lock'
    with path.open('x+b') as owner:
        recording._lock_candidate(owner)
        owner.write(b'{"candidate_id": "c"}')
        owner.flush()
        assert path.read_bytes() == b'{"candidate_id": "c"}'
        with path.open('r+b') as other:
            with pytest.raises(BlockingIOError):
                recording._lock_candidate(other)
    with path.open('r+b') as other:
        recording._lock_candidate(other)
    assert path.stat().st_size == len(b'{"candidate_id": "c"}')
