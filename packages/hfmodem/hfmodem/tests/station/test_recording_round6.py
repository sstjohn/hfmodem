# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Retained payloads, durable lane origins and protected candidate publication."""
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


def test_manifest_failure_after_keep_retains_lock(sessions, monkeypatch):
    rec, root = sessions(report=lambda message: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 1000}, provisional=True)
    push(rec, lane, 1000, 1008)
    rec.end('c1', {'card': 1008})
    flush(rec)
    contact = rec.contacts['c1']
    assert contact['kept']
    durable = module()._durable_json
    failures = []
    def failed(path, facts):
        if path.name == 'candidate.json':
            failures.append(path)
            raise OSError('synthetic manifest failure after keep')
        return durable(path, facts)
    with monkeypatch.context() as patch:
        patch.setattr(module(), '_durable_json', failed)
        push(rec, lane, 1024, 1048)
        bounded(rec.close)
    try:
        assert failures
        assert contact.get('retention_error')
        assert contact['lock'] is not None
        assert index(root)['contacts']['c1']['state'] == 'pending'
        assert json.loads((contact['directory'] / 'candidate.json').read_text())['state'] == 'keep'
    finally:
        if contact['lock'] is not None:
            contact['lock'].close()


def test_completed_contact_recovery_clips_bounds_and_gaps(tmp_path):
    root = tmp_path / 'crashed'
    script = '''
import os, sys, threading
from hfmodem.core.recording import RecordingSession
from hfmodem.tests.station.recording_support import RATE, TIMEOUT, flush, pcm
rec = RecordingSession('everything', sys.argv[1], report=lambda message: None)
lane = rec.stream('rx', rate=RATE, clock='card')
rec.begin('c1', {'card': 1000})
rec.end('c1', {'card': 1008})
flush(rec)
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
for first in [1000, 1008, 1024, 1032, 1040, 1048, 1064, 1072]:
    consumed.clear()
    lane.push(first, pcm(first, first + RATE))
    assert consumed.wait(TIMEOUT)
assert rec.contacts['c1']['done']
lane.writer.flush()
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(root)], check=True, timeout=30)
    assert interval(root)['actual'] == [1000, 1000]
    loaded = module().load_recording_index(root)
    ref = loaded['contacts']['c1']['lanes']['rx']
    assert ref['actual'] == [1000, 1048]
    assert ref['gaps'] == [[1016, 1024]]
    assert ref['missing_post_roll'] == 0
    assert ref['state'] == loaded['contacts']['c1']['state'] == 'partial'
    assert all(520 <= span['absolute'][0] < span['absolute'][1] <= 1048
               for item in ref['coverage'] for span in item['spans'])


@pytest.mark.parametrize('promised', [64, 80])
def test_recovery_counts_payload_frames(tmp_path, promised):
    family = tmp_path / ('a' * 32)
    family.mkdir()
    path = family / f'receive-20261101T0000Z-s{1000:020d}.wav'
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        audio.writeframes(bytes(2 * promised))
    with path.open('r+b') as audio:
        audio.truncate(44 + 2 * 72 + 1)
    source = str((family / 'receive.wav').relative_to(tmp_path))
    (tmp_path / 'recording.json').write_text(json.dumps({'lanes': {'rx': {'source': source}}}))
    loaded = module().load_recording_index(tmp_path)
    [descriptor] = loaded['segments']
    assert descriptor['samples'] == 72
    assert descriptor['actual'] == [1000, 1072]
    assert descriptor['state'] == 'partial'
    assert loaded['lanes']['rx']['unknown_from'] == 1072


@pytest.mark.parametrize('owned', [False, True])
@pytest.mark.parametrize('cache', [False, True])
def test_crash_recovers_leading_drop_and_lane_identity(tmp_path, owned, cache):
    root = tmp_path / 'crashed'
    script = '''
import os, sys, threading
from hfmodem.core import capture_identity as ci
from hfmodem.core.recording import RecordingSession
from hfmodem.tests.station.recording_support import RATE, TIMEOUT, pcm
rec = RecordingSession('everything', sys.argv[1], report=lambda message: None)
lanes = []
for name in ['rx', 'submitted/tx']:
    owner = None
    if sys.argv[2] == 'True':
        capture = ci.allocate({'capture': rec.directory / name.replace('/', '-')}, {'modem': 'capture'})
        owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav', clock_id=name, continuous=True)
    lanes.append(rec.stream(name, rate=RATE, clock=name, ownership=owner))
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
for lane, origin in zip(lanes, [1000, 2000]):
    lane.push(origin, pcm(origin, origin + 24))
    for first in [origin + 24, origin + 40]:
        consumed.clear()
        lane.push(first, pcm(first, first + RATE))
        assert consumed.wait(TIMEOUT)
    lane.writer.flush()
if sys.argv[3] == 'False':
    (rec.directory / 'recording.json').unlink()
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(root), str(owned), str(cache)],
                   check=True, timeout=30)
    loaded = module().load_recording_index(root)
    assert set(loaded['lanes']) == {'rx', 'submitted/tx'}
    for name, origin in [('rx', 1000), ('submitted/tx', 2000)]:
        row = loaded['lanes'][name]
        assert row['origin'] == origin
        assert row['position'] == row['unknown_from'] == origin + 48
        assert row['gaps'] == [[origin, origin + 24], [origin + 32, origin + 40]]
        assert row['rate'] == RATE
        assert all(d['lane'] == name for d in loaded['segments'] if Path(d['source']).parent == Path(row['source']).parent)


def test_clean_reload_preserves_leading_drop(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    lane.push(1000, pcm(1000, 1024))
    push(rec, lane, 1024, 1032)
    bounded(rec.close)
    loaded = module().load_recording_index(root)
    assert loaded['lanes']['rx']['origin'] == 1000
    assert loaded['lanes']['rx']['gaps'] == [[1000, 1024]]


def test_repeated_holes_reopen_segments_at_most_once_a_second(sessions):
    rec, root = sessions('everything', report=lambda message: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    for first, last in [(1000, 1008), (1016, 1018), (1020, 1026), (1027, 1029), (1032, 1040)]:
        push(rec, lane, first, last)
    bounded(rec.close)
    segments = index(root)['segments']
    assert [d['actual'] for d in segments] == [[1000, 1008], [1016, 1026], [1027, 1040]]
    assert [[s['absolute'] for s in d['spans']] for d in segments[1:]] == [
        [[1016, 1018], [1020, 1026]], [[1027, 1029], [1032, 1040]]]
    assert lane.gaps == [[1008, 1016], [1018, 1020], [1026, 1027], [1029, 1032]]
    assert lane.unrecorded() == 0


def test_preroll_gap_close_enospc_disables_lane_and_records_remaining_loss(sessions, monkeypatch):
    rec, root = sessions(report=lambda message: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 1000, 1008)
    push(rec, lane, 1016, 1032)
    create = rec._writer
    def failed_close(*args, **kwargs):
        writer = create(*args, **kwargs)
        finish = writer._finish
        def full():
            active = writer._wav is not None
            finish()
            if active:
                raise OSError(errno.ENOSPC, 'synthetic discontinuity close full')
        writer._finish = full
        return writer
    monkeypatch.setattr(rec, '_writer', failed_close)
    rec.begin('c1', {'card': 1032})
    flush(rec)
    assert lane.disabled
    assert rec.contacts['c1']['refs']['rx']['gaps'] == [[1000, 1032]]
    rec.end('c1', {'card': 1032})
    bounded(rec.close)
    assert interval(root)['actual'] == [1000, 1000]


def test_owned_lane_family_keeps_journal_identity_and_readers(sessions):
    from hfhost import recordings
    from hfmodem.core import capture_identity as ci
    from hfmodem.station.contactlog import cut_audio
    rec, root = sessions('everything')
    capture = ci.allocate({'capture': root / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav', clock_id='card', continuous=True)
    lane = rec.stream('rx', rate=RATE, clock='card', ownership=owner)
    push(rec, lane, 1000, 1016)
    bounded(rec.close)
    [descriptor] = index(root)['segments']
    path = root / descriptor['source']
    assert descriptor['state'] == 'finalized'
    assert ci.validate_artifact(path)['status'] == 'bound'
    assert ci.audio_metadata(path)['capture_state'] == 'finalized'
    # Capture identity names Windows paths in extended-length form.
    assert ci._absolute(recordings.base(path)) == ci._absolute(owner.base)
    assert [ci._absolute(p) for p in wav.segments(owner.base)] == [ci._absolute(path)]
    cut_audio(owner.base, root / 'cut.wav', 1002, 1004)
    with wave.open(str(root / 'cut.wav')) as audio:
        assert audio.getnframes() == 2


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_origin_markers_precede_pcm_on_owner_and_producer(sessions, monkeypatch, policy):
    rec, root = sessions(policy)
    calls = []
    mark = module().Lane._mark_origin
    def observed(lane):
        mark(lane)
        calls.append((lane.origin, threading.current_thread().name))
    monkeypatch.setattr(module().Lane, '_mark_origin', observed)
    lane = rec.stream('rx', rate=RATE, clock='card')
    assert len(list(root.rglob('*-ounknown.lane'))) == 1
    consumed = rec._consume
    def check(*args):
        assert len(list(root.rglob(f'*-o{1000:020d}.lane'))) == 1
        consumed(*args)
    monkeypatch.setattr(rec, '_consume', check)
    producer = threading.Thread(target=lambda: lane.push(1000, pcm(1000, 1008)), name='test-producer')
    producer.start()
    producer.join(5)
    assert not producer.is_alive()
    flush(rec)
    assert calls == [(None, threading.current_thread().name), (1000, 'test-producer')]
    assert lane.consumed == 1008
    assert not lane.disabled


def test_encoded_segment_stems_recover_lanes_without_markers_or_cache(tmp_path):
    for name, origin in [('rx', 1000), ('tx', 2000)]:
        directory = tmp_path / (('a' if name == 'rx' else 'b') * 32)
        directory.mkdir()
        for first in [origin, origin + 16]:
            path = directory / f'receive-l{name.encode().hex()}-20261101T0000Z-s{first:020d}.wav'
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
                audio.writeframes(bytes(2 * RATE))
    loaded = module().load_recording_index(tmp_path)
    for name, origin in [('rx', 1000), ('tx', 2000)]:
        lane = loaded['lanes'][name]
        assert lane['origin'] == origin
        assert lane['position'] == origin + 24
        assert lane['gaps'] == [[origin + 8, origin + 16]]
