# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Viewer replay and manifests retain selection context for each independent source."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.tests.station.recording_stage5_support import (
    RATE, START, STOP, clone, entry, reader, save_selection, wave_file, write_json,
)


def build_fixture(root, monkeypatch, protocol='vara'):
    view = reader('logbook_view.vara_view')
    selected = entry(root, protocol=protocol)
    path = save_selection(root, selected)
    wave_file(path, 64)
    side = json.loads(path.with_suffix('.json').read_text())
    side.update(start_stream_sample=START - 16, end_stream_sample=STOP + 16, samples=64)
    write_json(path.with_suffix('.json'), side)
    face = reader('desk.face')
    book = face.Logbook(root, station={'mycall': 'K9ZZT'})
    calls = []
    def replay(audio, bw, callsigns, out, **kwargs):
        calls.append((audio.copy(), bw, callsigns, kwargs))
        return []
    monkeypatch.setattr(view, 'FS', RATE)
    monkeypatch.setattr(view, 'replay', replay)
    monkeypatch.setattr(reader('logbook_view.pactor_view'), 'replay', replay)
    monkeypatch.setattr(reader('pactor_view'), 'replay', replay)
    monkeypatch.setattr(reader('logbook_view.ardop_view'), 'replay', replay)
    monkeypatch.setattr(reader('ardop_view'), 'replay', replay)
    monkeypatch.setattr(view, 'waterfall', lambda *a: {})
    monkeypatch.setattr(view, 'signal_view', lambda *a: None)
    monkeypatch.setattr(view, 'source_fingerprint', lambda: ('test-source', {}))
    return view, book, selected, path, calls


@pytest.mark.parametrize('protocol', ['vara', 'pactor', 'ardop'])
def test_view_build_clips_before_replay_and_uses_recorded_calls_roles_bandwidth(tmp_path, monkeypatch, protocol):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch, protocol)
    artifact = view.build(book, selected, tmp_path / 'derived')
    assert len(calls) == 1
    audio, bw, callsigns, kwargs = calls[0]
    np.testing.assert_array_equal(audio, np.arange(16, 48) / 32768)
    assert callsigns == ['W0QXR', 'N0CALL']
    assert bw == {'vara': 2750, 'pactor': 2400, 'ardop': 500}[protocol]
    assert kwargs['selection']['context']['id'] == 'context-c2'
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['duration_s'] == 4
    assert manifest['selection']['bounds'] == [START, STOP]
    assert manifest['selection']['clock_generation'] == 2
    assert manifest['selection']['context_id'] == 'context-c2'
    assert manifest['selection']['context_version'] == 1
    assert manifest['caller'] == 'W0QXR' and manifest['called'] == 'N0CALL'


@pytest.mark.parametrize('change', ['bounds', 'clock', 'context-id', 'context-version'])
def test_view_cache_key_separates_selection_identity(tmp_path, monkeypatch, change):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    first = view.build(book, selected, tmp_path / 'derived')
    other = clone(selected)
    rec = other['recording']
    if change == 'bounds':
        rec['lanes']['rx']['selection'] = [8008, STOP]
    elif change == 'clock':
        rec['lanes']['rx']['clock_generation'] = 3
        rec['context']['lanes']['rx']['clock_generation'] = 3
        side = json.loads(path.with_suffix('.json').read_text())
        side['clock_generation'] = 3
        write_json(path.with_suffix('.json'), side)
    elif change == 'context-id':
        rec['context']['id'] = 'context-c3'
    else:
        rec['context']['version'] = 2
    second = view.build(book, other, tmp_path / 'derived')
    assert first != second


@pytest.mark.parametrize('damage', ['transcript', 'probe'])
def test_view_missing_context_is_explicit_incomplete_without_hint_or_grading(tmp_path, monkeypatch, damage):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch, 'pactor')
    ctx = selected['recording']['context']
    (tmp_path / ctx[damage]['path']).unlink()
    artifact = view.build(book, selected, tmp_path / 'derived')
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['context']['state'] == 'incomplete'
    assert manifest['comparison']['logged_rx_overs'] is None
    assert calls[0][3].get('selection', {}).get('complete') is False
    assert not calls[0][3].get('tx')
    assert calls[0][2] == []


def test_view_ignores_unbound_sibling_probe(tmp_path, monkeypatch):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch, 'pactor')
    sibling = write_json(path.with_name('p4-probe.json'), {'sample_rate': RATE,
        'emissions': [{'audio_start': 0, 'audio_end': 2, 'word': 'other-contact'}]})
    original = view.sha
    def sha(candidate, **kwargs):
        assert candidate != sibling, 'selected replay must not discover a neighboring probe'
        return original(candidate, **kwargs)
    monkeypatch.setattr(view, 'sha', sha)
    view.build(book, selected, tmp_path / 'derived')


@pytest.mark.parametrize('failure', ['stat', 'hash', 'open'])
def test_view_isolates_unavailable_secondary_before_and_during_read(tmp_path, monkeypatch, failure):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    other = wave_file(path.with_name('witness.wav'))
    selected['audio'].append({'kind': 'sdr witness', 'url': '/file/working/run/witness.wav'})
    if failure == 'stat':
        original = book.served_path
        def served(rel):
            if str(rel).endswith('witness.wav'):
                raise FileNotFoundError(str(other))
            return original(rel)
        monkeypatch.setattr(book, 'served_path', served)
    else:
        name = 'sha' if failure == 'hash' else 'load_audio'
        original = getattr(view, name)
        def unavailable(candidate, *a, **k):
            if candidate == other:
                raise FileNotFoundError(str(other))
            return original(candidate, *a, **k)
        monkeypatch.setattr(view, name, unavailable)
    artifact = view.build(book, selected, tmp_path / 'derived')
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert any(s['kind'] == 'rig rx' for s in manifest['sources'])
    assert any(s['kind'] == 'sdr witness' and s['state'] == 'unavailable' for s in manifest['sources'])


def test_secondary_context_requires_its_own_clock_mapping(tmp_path, monkeypatch):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    wave_file(path.with_name('witness.wav'))
    selected['audio'].append({'kind': 'sdr witness', 'url': '/file/working/run/witness.wav',
        'selection': {'source_id': 'witness', 'clock': 'kiwi', 'clock_generation': 9,
                      'actual': [16000, 16032], 'rate': RATE}})
    monkeypatch.setattr(view, 'align_audio', lambda *a: (None, 'no clock anchor'))
    monkeypatch.setattr(view, 'align_packets', lambda *a: (None, 'no packet anchor'))
    monkeypatch.setattr(view, 'align_handshakes', lambda *a: (None, 'no handshake anchor'))
    artifact = view.build(book, selected, tmp_path / 'derived')
    assert len(calls) == 2
    assert calls[1][3].get('selection', {}).get('complete') is False
    assert not calls[1][3].get('tx')
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['sources'][1]['offset_s'] is None
    assert manifest['sources'][1]['context']['state'] == 'incomplete'


def test_pactor_view_uses_selected_mode_instead_of_whole_transcript(tmp_path, monkeypatch):
    module = reader('logbook_view.pactor_view')
    selected = entry(tmp_path, protocol='pactor')
    selected['recording']['context']['changes'] = []
    monkeypatch.setattr(module.rxfront, 'decode_events', lambda *a, **k: [])
    monkeypatch.setattr(module.p3rx, 'decode_p3_packets', lambda *a, **k: SimpleNamespace(packets=[]))
    monkeypatch.setattr(module.p1rx, 'decode_p1_packets', lambda *a, **k: [])
    monkeypatch.setattr(module.p2rx, 'find_markers', lambda *a, **k: [])
    monkeypatch.setattr(module.p1rx, 'acquire_robust_answer_search',
                        lambda *a, **k: pytest.fail('another contacts robust mode enabled selected decoding'))
    monkeypatch.setattr(module, 'p4_replay', lambda *a, **k: pytest.fail('another contacts P4 mode enabled selected decoding'))
    module.replay(np.zeros(48000), 2400, ['W0QXR', 'N0CALL'], tmp_path,
        transcript='robust P4 from a different contact', selection=selected['recording'],
        progress=lambda *a, **k: None)


def test_view_gap_restarts_decoder_and_retains_display_time(tmp_path, monkeypatch):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch, 'ardop')
    selected['recording']['lanes']['rx']['gaps'] = [[8008, 8016]]
    artifact = view.build(book, selected, tmp_path / 'derived')
    assert len(calls) == 2, 'each available span requires a new decoder/combining state'
    assert [len(call[0]) for call in calls] == [8, 16]
    assert calls[1][3]['selection']['bounds'] == [8016, STOP]
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['duration_s'] == 4 and manifest['gaps'] == [[1., 2.]]


def test_padding_is_not_automatically_attributed_as_peer_traffic(tmp_path, monkeypatch):
    view, book, selected, path, _ = build_fixture(tmp_path, monkeypatch)
    def replay(*args, **kwargs):
        return [{'id': 'padding', 't0': .1, 't1': .5, 'kind': 'unassigned',
                 'direction': 'observed', 'status': 'candidate', 'symbols': None},
                {'id': 'inside', 't0': 1.1, 't1': 1.5, 'kind': 'unassigned',
                 'direction': 'observed', 'status': 'candidate', 'symbols': None}]
    monkeypatch.setattr(view, 'replay', replay)
    artifact = view.build(book, selected, tmp_path / 'derived')
    events = json.loads((artifact / 'events.json').read_text())
    padding = next(e for e in events if e['id'] == 'padding')
    assert padding['direction'] == 'observed' and not padding.get('in_qso', False)


def test_secondary_move_after_hash_is_local_and_restoration_needs_no_log_edit(tmp_path, monkeypatch):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    other = wave_file(path.with_name('witness.wav'))
    selected['audio'].append({'kind': 'sdr witness', 'url': '/file/working/run/witness.wav'})
    original = view.sha
    moved = tmp_path / 'away.wav'
    def sha(candidate, **kwargs):
        digest = original(candidate, **kwargs)
        if candidate == other and not moved.exists():
            other.rename(moved)
        return digest
    monkeypatch.setattr(view, 'sha', sha)
    before = json.dumps(selected, sort_keys=True)
    try:
        artifact = view.build(book, selected, tmp_path / 'derived')
        manifest = json.loads((artifact / 'manifest.json').read_text())
        assert manifest['sources'][1]['state'] == 'unavailable'
    finally:
        if moved.exists():
            moved.rename(other)
    assert json.dumps(selected, sort_keys=True) == before
    assert book.served_path(other) == other
