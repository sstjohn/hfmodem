# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Session-scoped recovery and failure-safe terminal publication."""
import errno
import json
from pathlib import Path
import subprocess
import sys
import threading
import wave

import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_support import (
    RATE, bounded, flush, index, interval, module, pcm, push,
)


@pytest.mark.parametrize('owned', [False, True])
def test_restart_adopts_segments_after_crash_with_storage_blocked(sessions, tmp_path, owned):
    root = tmp_path / 'crashed'
    script = '''
import os, sys, threading
from hfmodem.core import capture_identity as ci, wav
from hfmodem.core.recording import RecordingSession
from hfmodem.tests.station.recording_support import RATE, TIMEOUT, pcm
wav.SEGMENT_SAMPLES = 60 * 48000
rec = RecordingSession('everything', sys.argv[1])
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
for first in range(0, 12 * 60 * RATE, RATE):
    consumed.clear()
    lane.push(first, pcm(first, first + RATE))
    assert consumed.wait(TIMEOUT)
assert len(rec.index['segments']) == 12
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(root), str(owned)], check=True, timeout=30)
    assert index(root)['segments'] == []
    files = {p.relative_to(root).as_posix() for p in root.rglob('*.wav')}
    assert len(files) == 12
    rec, _ = sessions('everything', directory=root)
    assert {d.get('pending_source', d['source']) for d in rec.index['segments']} == files
    assert all(d['state'] == 'pending' and d['samples'] == 60 * RATE
               for d in rec.index['segments'])
    assert all(d['rate'] == RATE for d in rec.index['segments'])
    assert index(root)['segments'] == rec.index['segments']


def test_stream_isolates_enospc_in_audio_and_index(sessions, monkeypatch):
    reports = []
    rec, root = sessions('everything', report=reports.append)
    with monkeypatch.context() as patch:
        original = Path.open
        def full(path, *args, **kwargs):
            if path.suffix == '.wav':
                raise OSError(errno.ENOSPC, 'synthetic PCM full')
            return original(path, *args, **kwargs)
        patch.setattr(Path, 'open', full)
        def full_index(*args):
            raise OSError(errno.ENOSPC, 'synthetic metadata full')
        patch.setattr(module(), '_durable_json', full_index)
        lane = rec.stream('rx', rate=RATE, clock='card')
        push(rec, lane, 1000, 1024)
    bounded(rec.close)
    assert lane.disabled and lane.position == lane.consumed == 1024
    assert index(root)['lanes']['rx']['gaps'] == [[1000, 1024]]
    assert any('synthetic metadata full' in r for r in reports)


@pytest.mark.parametrize('failure', ['write', 'close'])
def test_writer_errors_never_write_or_flush_stderr_on_drainer(sessions, monkeypatch, failure):
    calls, reports = [], []
    rec, root = sessions('everything', report=lambda msg: reports.append(threading.current_thread().name))
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, RATE)
    class Output:
        def write(self, text):
            calls.append(('write', threading.current_thread().name))
        def flush(self):
            calls.append(('flush', threading.current_thread().name))
    monkeypatch.setattr(sys, 'stderr', Output())
    def failed(*args):
        raise OSError(errno.ENOSPC, 'synthetic output full')
    if failure == 'write':
        monkeypatch.setattr(lane.writer._wav, 'writeframes', failed)
    else:
        handle = lane.writer._handle
        class Handle:
            def close(self):
                handle.close()
                failed()
        lane.writer._handle = Handle()
    push(rec, lane, RATE, 2 * RATE)
    bounded(rec.close)
    assert all(thread != 'recording-pcm' for _, thread in calls), calls
    assert reports and set(reports) == {'recording-store'}


def test_final_index_failure_keeps_candidate_pending_and_locked(sessions, monkeypatch):
    rec, root = sessions(report=lambda msg: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, RATE)
    rec.end('c1', {'card': RATE})
    flush(rec)
    monkeypatch.setattr(rec, '_admit', lambda job: None)
    push(rec, lane, RATE, 6 * RATE)
    contact = rec.contacts['c1']
    assert contact['publication_done'] and contact['lock'] is not None
    durable = module()._durable_json
    failed = []
    def fail_final(path, facts):
        if path.name == 'recording.json' and facts['contacts']['c1']['state'] == 'finalized':
            failed.append(True)
            raise OSError('synthetic final index failure')
        return durable(path, facts)
    monkeypatch.setattr(module(), '_durable_json', fail_final)
    try:
        rec._storage_call(rec._pending)
    except OSError:
        pass
    try:
        assert failed
        assert contact['lock'] is not None
        assert contact.get('retention_error')
        assert contact['row']['state'] == 'pending'
    finally:
        monkeypatch.setattr(module(), '_durable_json', durable)
        bounded(rec.close)
        if contact['lock'] is not None:
            contact['lock'].close()


def test_completed_reference_reconciles_later_segment_close_failure(sessions):
    rec, root = sessions('everything', report=lambda msg: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0})
    push(rec, lane, 0, RATE)
    rec.end('c1', {'card': RATE})
    push(rec, lane, RATE, 6 * RATE)
    contact = rec.contacts['c1']
    assert contact['done']
    selected = interval(root)['actual']
    handle = lane.writer._handle
    class Handle:
        def close(self):
            handle.close()
            raise OSError(errno.ENOSPC, 'synthetic final flush failure')
    lane.writer._handle = Handle()
    bounded(rec.close)
    ref = interval(root)
    assert ref['gaps'] == [[0, 6 * RATE]]
    assert ref['actual'][0] == ref['actual'][1]
    assert ref['missing_post_roll'] == 5 * RATE
    assert ref['selection'] == selected
    assert ref['state'] == 'partial'


def test_discard_reconciles_after_closing_writer(sessions, monkeypatch):
    rec, root = sessions()
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0}, provisional=True)
    push(rec, lane, 0, RATE)
    calls = []
    reconcile = rec._reconcile
    def observed(contact):
        calls.append(contact['writers']['rx']._finished)
        return reconcile(contact)
    monkeypatch.setattr(rec, '_reconcile', observed)
    rec.end('c1', {'card': RATE}, keep=False)
    bounded(rec.close)
    assert True in calls
    assert index(root)['contacts']['c1']['state'] == 'discarded'


def test_off_teardown_finishes_all_contacts_when_publisher_fails(sessions):
    published, reports = [], []
    def failing(row):
        published.append(row['id'])
        raise OSError('synthetic publisher failure')
    rec, root = sessions('off', publish=failing, report=reports.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0})
    rec.begin('c2', {'card': 0})
    lane.push(0, pcm(0, RATE))
    flush(rec)
    bounded(rec.close)
    assert all(c['done'] and c['decision'] is True for c in rec.contacts.values())
    assert set(published) == {'c1', 'c2'}
    assert all(row['end_positions'] == {'card': RATE} for row in index(root)['contacts'].values())
    assert len([r for r in reports if 'synthetic publisher failure' in r]) == 2


@pytest.mark.parametrize('state', ['undecided', 'discard', 'keep'])
def test_index_load_adopts_candidate_sidecar_without_resuming_it(tmp_path, state):
    root = tmp_path / 'run'
    directory = root / ('contact-' + 'a' * 32)
    lane = directory / ('b' * 32)
    lane.mkdir(parents=True)
    identity = {'candidate_id': 'c' * 32, 'contact_id': 'c1', 'directory': str(directory)}
    for name in ('candidate.lock', 'candidate.identity'):
        (directory / name).write_text(json.dumps(identity))
    source = lane / 'receive-20261101T0000Z.wav'
    facts = identity | {'state': state, 'owned_files': [source.relative_to(directory).as_posix()]}
    (directory / 'candidate.json').write_text(json.dumps(facts))
    source.with_suffix('.json').write_text(json.dumps({
        'wav': source.name, 'samplerate': RATE, 'samples': 48, 'capture_state': 'storing'}))
    loaded = module().load_recording_index(root)
    [descriptor] = loaded['segments']
    assert descriptor['source'] == source.relative_to(root).as_posix()
    assert descriptor['state'] == 'pending'
    assert descriptor['candidate_state'] == state
    assert descriptor['contact_id'] == 'c1'
    assert descriptor['actual'] is None and descriptor['spans'] == []
    assert not (root / 'recording.json').exists()


def test_index_load_adopts_unindexed_sidecars_and_staged_pcm_only_in_session_dirs(tmp_path):
    root = tmp_path / 'run'
    own = root / ('a' * 32)
    own.mkdir(parents=True)
    physical = own / '.receive-20261101T0000Z.pending.wav'
    with wave.open(str(physical), 'wb') as audio:
        audio.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        audio.writeframes(bytes(2 * RATE))
    sidecar = own / 'receive-20261101T0001Z.json'
    sidecar.write_text(json.dumps({'wav': sidecar.with_suffix('.flac').name,
                                   'samplerate': RATE, 'samples': RATE, 'capture_state': 'finalized'}))
    foreign = root / 'unrelated'
    foreign.mkdir()
    (foreign / 'receive-20261101T0002Z.json').write_text(sidecar.read_text())
    (root / ('b' * 32)).symlink_to(foreign, target_is_directory=True)
    loaded = module().load_recording_index(root)
    assert len(loaded['segments']) == 2
    assert {d['state'] for d in loaded['segments']} == {'pending'}
    staged = next(d for d in loaded['segments'] if 'pending_source' in d)
    assert staged['pending_source'] == physical.relative_to(root).as_posix()
    assert staged['samples'] == RATE
    assert len(module().load_recording_index(root)['segments']) == 2


def test_recovery_preserves_finalized_owned_segments_and_their_journal(sessions, monkeypatch):
    from hfmodem.core import capture_identity as ci
    rec, root = sessions('everything')
    capture = ci.allocate({'capture': root / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav',
                                 clock_id='card', continuous=True)
    lane = rec.stream('rx', rate=RATE, clock='card', ownership=owner)
    push(rec, lane, 0, 61 * RATE)
    bounded(rec.close)
    facts = index(root)
    expected = facts.pop('segments')
    facts['segments'] = []
    (root / 'recording.json').write_text(json.dumps(facts))
    before = {p: p.read_bytes() for p in (root / 'owned').rglob('*') if p.is_file()}
    stores = []
    monkeypatch.setattr(wav, 'store', lambda *args, **kwargs: stores.append(args) or {})
    recovered, _ = sessions('everything', directory=root)
    assert len(recovered.index['segments']) == len(expected)
    assert all(d['state'] == 'pending' for d in recovered.index['segments'])
    bounded(recovered.close)
    assert stores == []
    assert {p: p.read_bytes() for p in (root / 'owned').rglob('*') if p.is_file()} == before


def test_every_index_load_retains_cached_clock_spans_without_duplicates(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 1000, 1008)
    push(rec, lane, 1016, 1024)
    bounded(rec.close)
    facts = index(root)
    expected = facts['segments']
    facts['segments'] = []
    (root / 'recording.json').write_text(json.dumps(facts))
    loaded = module().load_recording_index(root)
    actual = loaded['segments']
    assert len(actual) == len(expected) == 2
    assert all(d['lane'] == 'rx' for d in actual)
    assert [d['actual'] for d in actual] == [d['actual'] for d in expected]
    assert [d['spans'] for d in actual] == [d['spans'] for d in expected]
    (root / 'recording.json').write_text(json.dumps(loaded))
    assert module().load_recording_index(root) == loaded


def test_index_load_uses_verified_replacement_when_cache_still_names_wav(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, RATE)
    bounded(rec.close)
    facts = index(root)
    [descriptor] = facts['segments']
    source = root / descriptor['source']
    converted = source.with_suffix('.flac')
    converted.write_bytes(b'synthetic finalized container')
    source.unlink()
    side = json.loads(source.with_suffix('.json').read_text())
    side['wav'] = converted.name
    source.with_suffix('.json').write_text(json.dumps(side))
    descriptor['state'] = 'storing'
    (root / 'recording.json').write_text(json.dumps(facts))
    [recovered] = module().load_recording_index(root)['segments']
    assert recovered['state'] == 'pending'
    assert recovered['source'] == converted.relative_to(root).as_posix()
