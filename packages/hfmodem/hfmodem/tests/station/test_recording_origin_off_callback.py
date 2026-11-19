# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""First-block origin publication leaves audio producers free of disk I/O."""
import builtins
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS

import pytest

from hfmodem.core import recording
from hfmodem.tests.station.recording_support import TIMEOUT, flush, pcm


# The marker's own fsync, then its directory and the one above, which Windows cannot open.
SYNCS = 1 if os.name == 'nt' else 3


def _watch_io(monkeypatch):
    calls = []
    def watch(name, fn):
        def observed(*args, **kwargs):
            calls.append((name, threading.get_ident()))
            return fn(*args, **kwargs)
        return observed
    for obj, name in ((os, 'open'), (builtins, 'open'), (Path, 'open'),
                      (os, 'fsync'), (recording, '_sync_directory')):
        monkeypatch.setattr(obj, name, watch(name, getattr(obj, name)))
    return calls


def _callback(fn):
    errors = []
    def run():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, name='test-audio-callback', daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive(), 'audio producer did not return'
    if errors:
        raise errors[0]
    return thread.ident


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_first_blocks_do_no_disk_io_on_callback(sessions, monkeypatch, policy):
    rec, _ = sessions(policy)
    lanes = [rec.stream('rx', rate=48000, clock='card'),
             rec.stream('submitted-tx', rate=48000, clock='output')]
    blocks = [(first, pcm(first, first + 128)) for first in range(0, 512, 128)]
    calls = _watch_io(monkeypatch)
    def produce():
        for first, block in blocks:
            for lane in lanes:
                lane.push(first, block)
    callback = _callback(produce)
    flush(rec)
    assert not [call for call in calls if call[1] == callback]


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_origin_marker_names_bytes_and_recovery_match_base(sessions, policy):
    rec, root = sessions(policy)
    lanes = [rec.stream('rx', rate=48000, clock='card'),
             rec.stream('submitted-tx', rate=48000, clock='output')]
    expected = {}
    for lane, origin in zip(lanes, (1000, 2000)):
        facts = {key: lane.row[key] for key in ('rate', 'clock', 'source_id', 'index_id')}
        facts['source'] = lane.row.get('source', recording._relative(lane.family, rec._root))
        for suffix in ('unknown', f'{origin:020d}'):
            path = lane.family.with_name(f'{lane.family.stem}-o{suffix}.lane')
            expected[path.relative_to(rec._root)] = json.dumps(facts).encode('utf-8')
        for first in range(origin, origin + 384, 128):
            lane.push(first, pcm(first, first + 128))

    def check():
        assert {path.relative_to(root): path.read_bytes()
                for path in root.rglob('*.lane')} == expected
        recovered = recording.load_recording_index(root)
        for lane in lanes:
            assert recovered['lanes'][lane.name]['origin'] == lane.origin
    flush(rec)
    check()
    rec.close()
    check()


@pytest.mark.realtime
def test_first_push_stays_under_one_ms_with_half_second_marker_fsyncs(sessions, monkeypatch):
    rec, root = sessions('contacts')
    lane = rec.stream('rx', rate=48000, clock='card')
    marking = threading.local()
    mark, fsync = lane._sync_origin, os.fsync
    synced, elapsed = [], []
    def marker(path):
        marking.active = True
        try:
            return mark(path)
        finally:
            marking.active = False
    def slow(fd):
        if getattr(marking, 'active', False):
            synced.append(threading.current_thread().name)
            time.sleep(.5)
        return fsync(fd)
    monkeypatch.setattr(lane, '_sync_origin', marker)
    monkeypatch.setattr(os, 'fsync', slow)
    block = pcm(0, 128)
    def produce():
        started = time.perf_counter()
        lane.push(0, block)
        elapsed.append(time.perf_counter() - started)
    _callback(produce)
    assert elapsed[0] < .001, f'first push took {elapsed[0]:.6f}s'
    flush(rec)
    assert synced == ['recording-store'] * SYNCS
    assert list(root.rglob('*-o00000000000000000000.lane'))


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
@pytest.mark.parametrize('failure', ['write', *range(1, SYNCS + 1)])
def test_origin_failure_disables_lane_and_reports_on_storage(sessions, monkeypatch, policy, failure):
    reports = []
    rec, _ = sessions(policy, report=lambda message: reports.append(
        (message, threading.current_thread().name)))
    lane = rec.stream('rx', rate=48000, clock='card')
    path = lane.family.with_name(f'{lane.family.stem}-o{1000:020d}.lane')
    opening, sync, fsync, consume = Path.open, lane._sync_origin, os.fsync, rec._consume
    marking = threading.local()
    attempts, disabled = [], []
    def opened(target, *args, **kwargs):
        if failure == 'write' and target == path and args == ('xb',):
            assert threading.current_thread().name == 'recording-pcm'
            raise OSError('synthetic origin failure')
        return opening(target, *args, **kwargs)
    def syncing(target):
        marking.active = Path(target.name) == path
        try:
            return sync(target)
        finally:
            marking.active = False
    def synced(fd):
        if getattr(marking, 'active', False):
            assert threading.current_thread().name == 'recording-store'
            attempts.append(fd)
            if len(attempts) == failure:
                raise OSError('synthetic origin failure')
        return fsync(fd)
    def consumed(*args):
        disabled.append(lane.disabled)
        return consume(*args)
    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(lane, '_sync_origin', syncing)
    monkeypatch.setattr(os, 'fsync', synced)
    monkeypatch.setattr(rec, '_consume', consumed)
    _callback(lambda: lane.push(1000, pcm(1000, 1128)))
    flush(rec)
    rec.close()
    assert lane.disabled
    assert reports == [('Recording rx origin: synthetic origin failure', 'recording-store')]
    if failure == 'write':
        assert disabled == [True]
        assert lane.gaps == [[1000, 1128]]
    else:
        assert len(attempts) == failure


def test_besra_first_submitted_tx_block_does_no_disk_io(monkeypatch, tmp_path):
    from hfmodem.besra import radio

    monkeypatch.setattr(radio, 'warm_output', lambda device: None)
    monkeypatch.setattr(radio, 'to_card', lambda samples, rate: samples)
    modem = NS(expected_session=lambda: None, rx_epoch=lambda: 0, status=lambda text: None)
    link = radio.RadioLink(modem, rig=None, record=tmp_path / 'besra.wav',
                           resolved=NS(policy='everything', source='test'))
    submitted_blocks = []
    def play(audio, rate, device, *, submitted):
        assert link.muted
        assert link.tx_lane.origin is None
        submitted(0, audio)
        submitted_blocks.append(len(audio))
    monkeypatch.setattr(radio, 'play_drained', play)
    try:
        calls = _watch_io(monkeypatch)
        callback = _callback(lambda: link._transmit(pcm(0, 128)))
        flush(link.recording)
        assert submitted_blocks == [128]
        assert link.tx_lane.origin == 0
        assert not [call for call in calls if call[1] == callback]
    finally:
        link.close()


def test_the_marker_is_synced_through_the_descriptor_that_wrote_it(sessions, monkeypatch):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=48000, clock='card')
    opening = Path.open
    def write_only(target, mode='r', *args, **kwargs):
        if Path(target).suffix == '.lane' and mode != 'xb':
            raise PermissionError('marker reopened')
        return opening(target, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', write_only)
    _callback(lambda: lane.push(1000, pcm(1000, 1128)))
    flush(rec)
    assert not lane.disabled
    assert list(root.rglob(f'*-o{1000:020d}.lane'))
