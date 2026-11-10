# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""File-backed clocks and isolated persistence failures."""
import errno
import json
import subprocess
import sys
import threading
import warnings
import wave

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_support import (
    RATE, bounded, flush, index, interval, module, pcm, push, samples,
)


@pytest.mark.parametrize('escape', ['absolute', 'parent', 'symlink', 'pending'])
def test_cached_paths_cannot_store_outside_session(sessions, tmp_path, monkeypatch, escape):
    root, foreign = tmp_path / 'run', tmp_path / 'foreign'
    root.mkdir()
    foreign.mkdir()
    audio = foreign / 'receive-20261101T0000Z.wav'
    with wave.open(str(audio), 'wb') as out:
        out.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        out.writeframes(bytes(2 * RATE))
    (root / 'linked').symlink_to(foreign, target_is_directory=True)
    source = {'absolute': str(audio), 'parent': '../foreign/' + audio.name,
              'symlink': 'linked/' + audio.name, 'pending': 'receive.wav'}[escape]
    descriptor = dict(source=source, samples=RATE, rate=RATE, state='pending')
    if escape == 'pending':
        descriptor['pending_source'] = str(audio)
    (root / 'recording.json').write_text(json.dumps({'segments': [descriptor]}))
    stored = []
    monkeypatch.setattr(wav, 'store', lambda *args, **kwargs: stored.append(args) or {})
    rec, _ = sessions('everything', directory=root, report=lambda msg: None)
    bounded(rec.close)
    assert rec.index['segments'] == []
    assert stored == []
    assert audio.is_file()


def test_rollover_failure_invalidates_completed_contact(sessions):
    rec, root = sessions('everything', report=lambda msg: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0})
    push(rec, lane, 0, RATE)
    rec.end('c1', {'card': RATE})
    push(rec, lane, RATE, 6 * RATE)
    assert rec.contacts['c1']['done']
    handle = lane.writer._handle
    class FailedHandle:
        def close(self):
            handle.close()
            raise OSError(errno.ENOSPC, 'synthetic rollover failure')
    lane.writer._handle = FailedHandle()
    push(rec, lane, 6 * RATE, 60 * RATE)
    ref = interval(root)
    assert ref['actual'] == [0, 0]
    assert ref['gaps'] == [[0, 6 * RATE]]
    assert ref['missing_post_roll'] == 5 * RATE
    assert lane.gaps == [[0, 60 * RATE]]


def test_initial_index_enospc_returns_usable_session(sessions, monkeypatch):
    reports = []
    durable = module()._durable_json
    def full(path, facts):
        if path.name == 'recording.json':
            raise OSError(errno.ENOSPC, 'synthetic startup full')
        return durable(path, facts)
    with monkeypatch.context() as patch:
        patch.setattr(module(), '_durable_json', full)
        rec, root = sessions('everything', report=reports.append)
        lane = rec.stream('rx', rate=RATE, clock='card')
        lane.push(1000, pcm(1000, 1008))
    flush(rec)
    bounded(rec.close)
    assert lane.position == lane.consumed == 1008
    assert any('synthetic startup full' in r for r in reports)
    assert not rec._pcm_thread.is_alive() and not rec._storage_thread.is_alive()


def test_persistent_index_enospc_still_stops_workers(sessions, monkeypatch):
    def full(*args):
        raise OSError(errno.ENOSPC, 'synthetic permanent index full')
    monkeypatch.setattr(module(), '_durable_json', full)
    rec, _ = sessions('everything', report=lambda msg: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    lane.push(0, pcm(0, RATE))
    bounded(rec.close)
    assert not rec._pcm_thread.is_alive() and not rec._storage_thread.is_alive()


def test_candidate_authorizes_new_offset_after_gap_without_losing_pcm(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, RATE)
    push(rec, lane, 3 * RATE, 4 * RATE)
    rec.end('c1', {'card': 4 * RATE})
    bounded(rec.close)
    descriptors = index(root)['segments']
    assert [d['actual'] for d in descriptors] == [[0, RATE], [3 * RATE, 4 * RATE]]
    assert interval(root)['gaps'] == [[RATE, 3 * RATE]]


def test_candidate_preroll_authorizes_each_contiguous_span(sessions):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, RATE)
    push(rec, lane, 3 * RATE, 10 * RATE)
    rec.begin('c1', {'card': 10 * RATE}, provisional=True)
    flush(rec)
    rec.end('c1', {'card': 10 * RATE})
    bounded(rec.close)
    descriptors = index(root)['segments']
    assert [d['actual'] for d in descriptors] == [[0, RATE], [3 * RATE, 10 * RATE]]
    assert interval(root)['gaps'] == [[RATE, 3 * RATE]]


def test_closely_spaced_holes_cost_only_themselves(sessions):
    """Losses a few blocks apart once cost the recording the rest of a second each."""
    reports = []
    rec, root = sessions('everything', report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 1000, 1008)
    push(rec, lane, 1012, 1016)
    push(rec, lane, 1018, 1032)
    push(rec, lane, 1050, 1056)
    bounded(rec.close)
    assert lane.gaps == [[1008, 1012], [1016, 1018], [1032, 1050]]
    descriptors = index(root)['segments']
    assert [d['actual'] for d in descriptors] == [[1000, 1008], [1012, 1032], [1050, 1056]]
    assert [s['absolute'] for s in descriptors[1]['spans']] == [[1012, 1016], [1018, 1032]]
    got = samples(root, {'segments': descriptors[1:2]})
    want = np.round(pcm(1012, 1032) * 32768).astype('<i2')
    want[4:6] = 0
    np.testing.assert_array_equal(got, want)
    assert lane.unrecorded() == 0
    assert not [msg for msg in reports if 'coalesced' in msg]


def test_filled_holes_survive_reload_and_a_nonfinite_block_after_one(sessions):
    rec, root = sessions('everything', report=lambda msg: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    lane.push(0, pcm(0, 3))
    lane.push(4, pcm(4, 6))
    block = pcm(8, 11)
    block[1] = np.nan
    lane.push(8, block)
    flush(rec)
    bounded(rec.close)
    assert lane.gaps == [[3, 4], [6, 8], [9, 10]]
    descriptors = index(root)['segments']
    assert [d['actual'] for d in descriptors] == [[0, 3], [4, 11]]
    assert [s['absolute'] for s in descriptors[1]['spans']] == [[4, 6], [8, 9], [10, 11]]
    assert [s['offset'] for s in descriptors[1]['spans']] == [0, 4, 6]
    assert len(samples(root, {'segments': descriptors[1:]})) == 7
    loaded = module().load_recording_index(root)
    assert loaded['lanes']['rx']['gaps'] == lane.gaps
    assert [[s['absolute'] for s in d['spans']] for d in loaded['segments'] if d.get('lane') == 'rx'] == [
        [[0, 3]], [[4, 6], [8, 9], [10, 11]]]


def test_storage_revalidates_cached_paths_after_load(sessions, monkeypatch, tmp_path):
    rec, root = sessions('everything', report=lambda msg: None)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (root / 'linked').symlink_to(foreign, target_is_directory=True)
    audio = foreign / 'receive.wav'
    with wave.open(str(audio), 'wb') as out:
        out.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        out.writeframes(bytes(RATE * 2))
    descriptor = dict(source='linked/receive.wav', rate=RATE, samples=RATE, state='pending')
    rec.index['segments'].append(descriptor)
    stored = []
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: stored.append(a) or {})
    bounded(rec.close)
    assert stored == [] and descriptor['state'] == 'partial'


def test_nonfinite_samples_are_gaps_without_drainer_output(sessions, monkeypatch):
    output, reports = [], []
    rec, root = sessions('everything', report=lambda msg: reports.append((msg, threading.current_thread().name)))
    lane = rec.stream('rx', rate=RATE, clock='card')
    original = warnings.showwarning
    def observed(*args, **kwargs):
        output.append(threading.current_thread().name)
        return original(*args, **kwargs)
    monkeypatch.setattr(warnings, 'showwarning', observed)
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        lane.push(1000, np.array([0.1, np.nan, np.inf, -np.inf, 0.2], dtype=np.float32))
        flush(rec)
    bounded(rec.close)
    assert 'recording-pcm' not in output
    assert lane.gaps == [[1001, 1004]]
    assert lane.position == lane.consumed == 1005
    assert any('nonfinite' in msg.lower() for msg, _ in reports)
    assert all(thread == 'recording-store' for _, thread in reports)
    assert [d['actual'] for d in index(root)['segments']] == [[1000, 1001], [1004, 1005]]


@pytest.mark.parametrize('owned', [False, True])
def test_crash_recovers_exact_clocks_from_files_with_gaps(tmp_path, owned):
    root = tmp_path / 'crashed'
    script = '''
import json, os, sys, threading
from hfmodem.core import capture_identity as ci, wav
from hfmodem.core.recording import RecordingSession
from hfmodem.tests.station.recording_support import RATE, TIMEOUT, pcm
wav.SEGMENT_SAMPLES = 2 * 48000
rec = RecordingSession('everything', sys.argv[1], report=lambda msg: None)
owner = None
if sys.argv[2] == 'True':
    capture = ci.allocate({'capture': rec.directory / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav', clock_id='card', continuous=True)
lane = rec.stream('rx', rate=RATE, clock='card', ownership=owner)
entered, release, consumed = (threading.Event() for _ in range(3))
def blocked():
    entered.set()
    assert release.wait(30)
rec._enqueue(blocked)
assert entered.wait(TIMEOUT)
consume = rec._consume
def observed(*args):
    consume(*args)
    consumed.set()
rec._consume = observed
def feed(first, last):
    for start in range(first, last, RATE):
        consumed.clear()
        lane.push(start, pcm(start, min(start + RATE, last)))
        assert consumed.wait(TIMEOUT)
feed(1000, 1032)
feed(1040, 1050)
feed(1052, 1084)
# A block larger than the bounded queue is unavailable.
lane.push(1084, pcm(1084, 1108))
feed(1116, 1148)
lane.writer.flush()
facts = {'origin': lane.origin, 'position': lane.position, 'gaps': lane.gaps,
         'segments': [{'actual': c['actual'], 'spans': c['spans']} for c in lane.writer.coverage.values()]}
print(json.dumps(facts), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), str(owned)],
                            check=True, timeout=30, capture_output=True, text=True)
    expected = json.loads(result.stdout)
    assert index(root)['segments'] == []
    loaded = module().load_recording_index(root)
    row = loaded['lanes']['rx']
    assert {k: row.get(k) for k in ('origin', 'position', 'gaps')} == {
        k: expected[k] for k in ('origin', 'position', 'gaps')}
    actual = sorted(({'actual': d['actual'], 'spans': d['spans']} for d in loaded['segments']),
                    key=lambda d: d['actual'][0] if d['actual'] else -1)
    assert actual == expected['segments']
    assert row['gaps'] == [[1032, 1040], [1050, 1052], [1084, 1116]]
    assert all(d['samples'] <= 2 * RATE and d['state'] == 'pending' for d in loaded['segments'])
