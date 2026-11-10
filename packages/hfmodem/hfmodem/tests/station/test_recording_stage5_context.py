# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Published replay context is evaluated at the selected source clock position."""
import json
import threading
from pathlib import Path

import pytest

from hfmodem.station.contactlog import ContactLog
from hfmodem.tests.station.recording_stage5_support import (
    RATE, START, STOP, bounded, clone, context, entry, index, push,
    reader, save_selection, select,
)


@pytest.mark.parametrize('policy', ['contacts', 'everything'])
def test_publication_builds_context_at_retained_preroll_on_storage_worker(sessions, tmp_path, policy):
    root = tmp_path / 'working/run'
    supplied = context(tmp_path)
    published = []
    rec, _ = sessions(policy, directory=root,
        publish=lambda row: published.append((row, threading.current_thread())))
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, START - 16, START)
    rec.begin('c2', {'card': START}, context=supplied, endpoint_generation=7)
    rec.end('c2', {'card': START + 8})
    push(rec, lane, START, STOP + 16)
    bounded(rec.close)
    row, worker = published[0]
    assert worker is rec._storage_thread
    ctx = row['context']
    assert ctx['contact_id'] == 'c2' and ctx['endpoint_generation'] == 7
    assert ctx['lanes']['rx']['source_id'] == lane.row['source_id'], 'context must bind the actual lane'
    assert ctx['lanes']['rx']['index_id'] == lane.row['index_id']
    assert ctx['retained'] == row['lanes']['rx']['actual']
    assert ctx['initial']['expected_session'] == 0x22
    assert ctx['transcript']['range'] == supplied['transcript']['range']
    assert ctx['tx'][0]['ptt_start_sample'] == 7996
    assert index(root)['contacts']['c2']['context'] == ctx


def test_contactlog_persists_context_with_reference_after_postroll(sessions, tmp_path):
    root = tmp_path / 'working/run'
    rec, _ = sessions('everything', directory=root)
    lane = rec.stream('rx', rate=RATE, clock='card')
    log = ContactLog(root, recording=rec)
    facts = dict(n=2, peer='W0QXR', dial_hz=7105000, bandwidth_hz=2750,
                 started_utc='2026-11-01T12:16:40Z', how='incoming', context=context(tmp_path))
    push(rec, lane, START - 16, START)
    log.update(facts, sample=START, sample_rate=RATE)
    native = log.update(facts | {'ended_utc': '2026-11-01T12:16:41Z'},
                        sample=START + 8, sample_rate=RATE)
    push(rec, lane, START, STOP + 16)
    bounded(rec.close)
    saved = json.loads((Path(native['directory']) / 'provenance.json').read_text())
    assert saved['recording'].get('context'), 'ContactLog must publish replay context beside interval references'
    assert saved['recording']['context']['local_role'] == 'called'
    assert saved['recording']['context']['retained'] == saved['recording']['lanes']['rx']['actual']


@pytest.mark.parametrize('start,session,epoch,muted,mode', [
    (8000, 0x22, 4, True, 'normal'), (8010, 0x22, 4, False, 'normal'),
    (8018, 0x22, 5, False, 'P4'), (8026, None, 5, False, 'P4'),
])
def test_selection_seeds_predecessor_state_without_quarter_hour_fold(tmp_path, start, session, epoch, muted, mode):
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    result = select(selected, path, tmp_path, start=start, stop=STOP)
    assert result.bounds == (start, STOP)
    assert result.state['expected_session'] == session
    assert result.state['arq_epoch'] == epoch
    assert result.state['muted'] is muted and result.state['mode'] == mode
    assert result.log.predicted(None) == pytest.approx(start / RATE)
    assert result.clock_generation == 2
    assert result.context_id == 'context-c2' and result.context_version == 1


def test_selection_clips_tx_once_and_keeps_original_bounds(tmp_path):
    selected = entry(tmp_path)
    result = select(selected, save_selection(tmp_path, selected), tmp_path)
    [tx] = result.tx
    assert (tx['t0'], tx['t1']) == (0, 1)
    assert tx['original_bounds']['ptt_start_sample'] == 7996
    assert tx['original_bounds']['audio_start_sample'] == 7998
    assert result.participation == (8008, 8024)


def test_gap_restart_and_clock_generation_exclude_neighbor_history(tmp_path):
    selected = entry(tmp_path, gaps=[[8010, 8016]])
    selected['recording']['context']['changes'].append(
        {'sample': 8017, 'clock_generation': 1, 'expected_session': 0x11, 'arq_epoch': 99})
    result = select(selected, save_selection(tmp_path, selected), tmp_path)
    assert [span.bounds for span in result.spans] == [(8000, 8010), (8016, 8032)]
    assert all(span.reset_decoder for span in result.spans)
    assert result.spans[1].state['arq_epoch'] == 5
    assert result.spans[1].state['expected_session'] == 0x22
    assert result.spans[1].state['grid']['anchor'] == 7992
    assert result.gaps == ((8010, 8016),)


@pytest.mark.parametrize('damage', ['context', 'missing-transcript', 'digest', 'source-id', 'index-id', 'clock', 'probe'])
def test_invalid_context_is_incomplete_without_neighbor_discovery(tmp_path, monkeypatch, damage):
    selected = entry(tmp_path, protocol='pactor')
    path = save_selection(tmp_path, selected)
    ctx = selected['recording']['context']
    if damage == 'context':
        del selected['recording']['context']
    elif damage == 'missing-transcript':
        (tmp_path / ctx['transcript']['path']).unlink()
    elif damage == 'digest':
        ctx['transcript']['sha256'] = '0' * 64
    elif damage == 'probe':
        ctx['probe']['sha256'] = '0' * 64
    else:
        field = {'source-id': 'source_id', 'index-id': 'index_id', 'clock': 'clock_generation'}[damage]
        ctx['lanes']['rx'][field] = 'other'
    module = reader('hfmodem.station.sessionlog')
    monkeypatch.setattr(module, 'find', lambda *a, **k: pytest.fail('selected context searched neighboring logs'))
    result = select(selected, path, tmp_path)
    assert not result.complete and result.reason
    assert result.state['expected_session'] is None
    assert result.tx == () and result.grading is None


def test_context_range_digest_survives_transcript_append(tmp_path):
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    log = tmp_path / selected['recording']['context']['transcript']['path']
    with log.open('a') as dst:
        dst.write('2026-11-01 13:00:00,000 INFO RX IDLE sess=0x33 ok=True\n')
    result = select(selected, path, tmp_path)
    assert result.complete and {frame.session for frame in result.log.frames} == {0x22}


def test_legacy_read_keeps_current_majority_and_bandwidth_discovery(tmp_path):
    selected = entry(tmp_path)
    module = reader('hfmodem.station.sessionlog')
    log = module.read(tmp_path / selected['recording']['context']['transcript']['path'])
    assert log.session == 0x22 and log.bw == '500'
    assert log.predicted(log.zero) == 0


def test_overlapping_contact_padding_does_not_share_decoder_state(tmp_path):
    first = entry(tmp_path)
    second = clone(first)
    second['recording']['id'] = 'c3'
    second['recording']['context'].update(id='context-c3', contact_id='c3', endpoint_generation=8)
    second['recording']['context']['initial']['expected_session'] = 0x33
    second['recording']['context']['initial']['arq_epoch'] = 9
    second['recording']['context']['changes'] = []
    path = save_selection(tmp_path, first)
    a = select(first, path, tmp_path)
    b = select(second, path, tmp_path)
    assert a.bounds == b.bounds
    assert a.state['expected_session'] == 0x22 and b.state['expected_session'] == 0x33
    assert a.context_id != b.context_id


def test_legacy_selection_helper_keeps_existing_log_discovery(tmp_path, monkeypatch):
    from hfmodem.tests.station.recording_stage5_support import wave_file
    selected = entry(tmp_path)
    path = wave_file(tmp_path / 'receive.wav')
    log = tmp_path / selected['recording']['context']['transcript']['path']
    with log.open('a') as stream:
        stream.write(f'session stream: {path}\n')
    module = reader('hfmodem.station.sessionlog')
    monkeypatch.setattr(module, '_roots', lambda: [log.parent])
    result = select({'recording': {}}, path, tmp_path)
    assert result.legacy and result.log.bw == '500'
    assert result.log.path == log
