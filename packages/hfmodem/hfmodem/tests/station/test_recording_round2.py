# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Storage failures preserve availability facts and independent sample clocks."""
import errno
import json
import os
from pathlib import Path
import stat
import threading

import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_support import (
    RATE, TIMEOUT, bounded, flush, index, interval, module, pcm, push,
)


@pytest.mark.parametrize('damage', ['header', 'truncated'])
def test_hashable_invalid_audio_remains_partial(sessions, monkeypatch, damage):
    rec, root = sessions('contacts')
    segment = rec._segment
    def damaged(lane, writer, path, contact):
        data = path.read_bytes()
        path.write_bytes(b'INVALID!' + data[8:] if damage == 'header' else data[:-2])
        segment(lane, writer, path, contact)
    monkeypatch.setattr(rec, '_segment', damaged)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 0})
    push(rec, lane, 0, 16)
    rec.end('c1', {'card': 16})
    push(rec, lane, 16, 56)
    bounded(rec.close)
    [descriptor] = index(root)['segments']
    assert descriptor['state'] == 'partial'
    assert interval(root)['state'] == 'partial'
    assert (root / descriptor['source']).is_file()
    assert not list(root.rglob('*.flac'))


def test_durable_json_syncs_temporary_before_replacing_old_index(tmp_path, monkeypatch):
    target = tmp_path / 'recording.json'
    target.write_text('{"old": true}')
    events = []
    replace, fsync = os.replace, os.fsync
    def synced(fd):
        events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        return fsync(fd)
    def replaced(source, destination, **kwargs):
        assert Path(destination) == target
        assert json.loads(target.read_text()) == {'old': True}
        events.append('replace')
        return replace(source, destination, **kwargs)
    monkeypatch.setattr(os, 'fsync', synced)
    monkeypatch.setattr(os, 'replace', replaced)
    module()._durable_json(target, {'new': True})
    # Windows cannot fsync a directory; NTFS journals the rename.
    assert events == ['file', 'replace'] + ([] if os.name == 'nt' else ['directory'])
    assert json.loads(target.read_text()) == {'new': True}


@pytest.mark.parametrize('publication', ['index', 'keep'])
def test_blocked_storage_metadata_does_not_block_pcm(sessions, monkeypatch, publication):
    entered, release, consumed = (threading.Event() for _ in range(3))
    rec, root = sessions('everything' if publication == 'index' else 'contacts')
    lane = rec.stream('rx', rate=RATE, clock='card')
    start = 58 * RATE if publication == 'keep' else 0
    if publication == 'keep':
        rec.begin('c1', {'card': 0}, provisional=True)
        push(rec, lane, 0, start)
    flush(rec)
    durable, consume = module()._durable_json, rec._consume
    worker_ids, publications = [], []
    def blocked(path, facts):
        worker = threading.current_thread() is rec._storage_thread
        selected = (path.name == 'recording.json' if publication == 'index' else
                    path.name == 'candidate.json' and facts['state'] == 'keep')
        if worker and selected and not entered.is_set():
            worker_ids.append(threading.get_ident())
            entered.set()
            assert release.wait(TIMEOUT)
        result = durable(path, facts)
        if path.name == 'recording.json':
            publications.append(facts['lanes']['rx']['position'])
        return result
    def observed(lane, first, data):
        result = consume(lane, first, data)
        consumed.set()
        return result
    monkeypatch.setattr(module(), '_durable_json', blocked)
    monkeypatch.setattr(rec, '_consume', observed)
    if publication == 'index':
        rec._enqueue(rec._save)
    else:
        rec.end('c1', {'card': start})
    try:
        assert entered.wait(TIMEOUT)
        for first in range(start, start + 4 * RATE, RATE):
            consumed.clear()
            lane.push(first, pcm(first, first + RATE))
            assert consumed.wait(TIMEOUT / 2), 'storage metadata blocked PCM consumption'
        assert lane.consumed == start + 4 * RATE
        if publication == 'keep':
            directory = root / index(root)['contacts']['c1']['directory']
            facts = json.loads((directory / 'candidate.json').read_text())
            assert facts['state'] == 'undecided'
            assert all(p.relative_to(directory).as_posix() in facts['owned_files']
                       for p in directory.rglob('*.wav'))
    finally:
        release.set()
    bounded(rec.close)
    assert worker_ids == [rec._storage_thread.ident]
    assert index(root)['lanes']['rx']['gaps'] == []
    assert publications == sorted(publications)
    assert publications[-1] == start + 4 * RATE
    if publication == 'keep':
        directory = root / index(root)['contacts']['c1']['directory']
        assert json.loads((directory / 'candidate.json').read_text())['state'] == 'keep'


def test_contact_sink_creation_enospc_disables_one_lane_without_retries(tmp_path, monkeypatch):
    reports, attempts = [], []
    root = tmp_path / 'run'
    rec = module().RecordingSession('contacts', root, report=reports.append)
    monkeypatch.setattr(wav, '_flac', lambda *a: (_ for _ in ()).throw(OSError('synthetic encoder unavailable')))
    rx = rec.stream('rx', rate=RATE, clock='card')
    tx = rec.stream('tx', rate=RATE, clock='output')
    writer = rec._writer
    def no_space(lane, path, contact=None):
        if lane is rx:
            attempts.append(path)
            raise OSError(errno.ENOSPC, 'synthetic sink full')
        return writer(lane, path, contact)
    monkeypatch.setattr(rec, '_writer', no_space)
    for cid, start in [('c1', 0), ('c2', 16)]:
        rec.begin(cid, {'card': start, 'output': start})
        rec.end(cid, {'card': start + 8, 'output': start + 8})
        for first in range(start, start + 16, RATE):
            rx.push(first, pcm(first, first + RATE))
            tx.push(first, pcm(first, first + RATE))
            flush(rec)
    bounded(rec.close)
    assert rx.disabled and not tx.disabled
    assert rx.consumed == tx.consumed == 32
    assert len(attempts) == 1
    assert index(root)['lanes']['rx']['gaps'] == [[0, 32]]
    assert index(root)['lanes']['rx']['state'] == 'disabled'
    for cid in ('c1', 'c2'):
        assert interval(root, cid)['state'] in ('partial', 'unavailable')
        assert interval(root, cid)['segments'] == []
        assert interval(root, cid, 'tx')['segments']
    assert len([r for r in reports if 'synthetic sink full' in r]) == 1
    assert not any('KeyError' in r or "'rx'" in r for r in reports)


@pytest.mark.parametrize('failure', ['wav-close', 'handle-close'])
def test_segment_close_failure_is_visible_before_callback(sessions, monkeypatch, failure):
    rec, root = sessions('everything', report=lambda message: None)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 8)
    writer = lane.writer
    seen, stores = [], []
    callback = writer._closed_segment
    def closed(path):
        seen.append(writer._segment_failed)
        callback(path)
    monkeypatch.setattr(writer, '_closed_segment', closed)
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: stores.append(a) or {})
    if failure == 'wav-close':
        close = writer._wav.close
        def failed_close():
            close()
            raise OSError(errno.ENOSPC, 'synthetic WAV close failure')
        monkeypatch.setattr(writer._wav, 'close', failed_close)
    else:
        handle = writer._handle
        class FailedHandle:
            def close(self):
                handle.close()
                raise OSError(errno.ENOSPC, 'synthetic handle flush failure')
        writer._handle = FailedHandle()
    bounded(rec.close)
    assert seen == [True]
    assert stores == []
    assert index(root)['segments'][0]['state'] == 'partial'


def test_off_owner_teardown_uses_final_input_clock(sessions):
    published = []
    rec, root = sessions('off', publish=published.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 1000})
    lane.push(1000, pcm(1000, 1008))
    lane.push(1016, pcm(1016, 1024))
    bounded(rec.close)
    assert published[0]['end_positions'] == {'card': 1024}
    assert index(root)['lanes']['rx']['gaps'] == [[1008, 1016]]
    assert lane.ring is None and lane.writer is None and not hasattr(rec, '_pcm')


@pytest.mark.parametrize('warning', ['overflow', 'overlap'])
def test_push_warnings_are_coalesced_off_acquisition(sessions, warning):
    entered, release = threading.Event(), threading.Event()
    reports, callers = [], []
    def report(message):
        if 'overflow' in message or 'overlapping' in message:
            reports.append(message)
            callers.append(threading.get_ident())
            entered.set()
            assert release.wait(TIMEOUT)
    rec, _ = sessions('everything', report=report)
    rx = rec.stream('rx', rate=RATE, clock='card')
    tx = rec.stream('tx', rate=RATE, clock='output')
    if warning == 'overlap':
        push(rec, rx, 0, 8)
    acquisition_ids = []
    def acquire():
        acquisition_ids.append(threading.get_ident())
        for i in range(100):
            first = i * (2 * RATE + 1) if warning == 'overflow' else 0
            rx.push(first, pcm(first, first + (2 * RATE + 1 if warning == 'overflow' else RATE)))
        tx.push(0, pcm(0, RATE))
    try:
        bounded(acquire)
        assert entered.wait(TIMEOUT)
        assert tx.position == RATE
        assert not set(callers).intersection(acquisition_ids)
    finally:
        release.set()
    bounded(rec.close)
    assert 1 <= len(reports) <= 2
