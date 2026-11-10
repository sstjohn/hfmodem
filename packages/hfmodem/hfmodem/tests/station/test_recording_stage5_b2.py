# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Adapter reads retain source bounds and restart decoding after gaps."""
import json

import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_stage5_support import entry, reader, save_selection


@pytest.mark.parametrize('adapter,protocol', [('besra', 'ardop'), ('kestrel', 'vara'), ('shrike', 'pactor')])
def test_adapter_bounds_pcm_before_read_and_restarts_at_gaps(tmp_path, monkeypatch, adapter, protocol):
    selected = entry(tmp_path, protocol=protocol, gaps=[[8008, 8016]])
    path = save_selection(tmp_path, selected)
    side = json.loads(path.with_suffix('.json').read_text())
    side.update(peak=.1, rms=.01, lost_samples=0, grid_loss_seen=False, xruns=0)
    path.with_suffix('.json').write_text(json.dumps(side))
    module = reader('tools.rehear.' + adapter)
    reads = []
    original = wav.read
    def bounded(path, *args, **kwargs):
        assert kwargs.get('start') is not None and kwargs.get('stop') is not None
        reads.append((kwargs['start'], kwargs['stop']))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(wav, 'read', bounded)
    if adapter == 'besra':
        monkeypatch.setattr(module, '_rolling', lambda *a: [])
    elif adapter == 'kestrel':
        monkeypatch.setattr(module, '_scanned', lambda *a: [])
        monkeypatch.setattr(module, '_search', lambda *a, **k: ([], None))
    else:
        for name in ('_rolling', '_ungated', '_anchored'):
            monkeypatch.setattr(module, name, lambda *a: [])
    result = module.rehear(path, selection=selected['recording'])
    assert reads == [(8000, 8008), (8016, 8032)]
    assert result.seconds == 4
    assert not result.deaf


def test_cli_incomplete_context_does_not_dispatch_filename_hints(tmp_path, monkeypatch):
    selected = entry(tmp_path)
    selected['recording']['context']['transcript']['sha256'] = 'wrong'
    path = save_selection(tmp_path, selected)
    module = reader('tools.rehear')
    monkeypatch.setattr(module.sessionlog, 'find', lambda *a: pytest.fail('neighbor lookup'))
    result = module.rehear(path)
    assert result.against_log is None
    assert 'incomplete' in result.note.lower()


def test_monitor_restarts_children_after_gap_and_preserves_display_clock(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from creance.monitor import driver
    from hfhost import audio
    selected = entry(tmp_path, gaps=[[8008, 8016]])
    path = save_selection(tmp_path, selected)
    module = reader('logbook_view.rehear_arms')
    instances = []
    spec = SimpleNamespace(name='besra', unavailable=lambda: None)
    monkeypatch.setattr(driver, 'default_specs', lambda *a, **k: [spec])
    class Monitor:
        def __init__(self, spec, *a, **k):
            self.spec, self.fed, self.dropped_samples = spec, [], 0
            instances.append(self)
        def start(self): pass
        def feed(self, first, pcm): self.fed.append(first)
        def stop(self, grace): return True
    monkeypatch.setattr(driver, 'ModemMonitor', Monitor)
    origin = 1000 * audio.FS
    monkeypatch.setattr(audio, 'WavReplaySource', lambda *a, **k: SimpleNamespace(
        frames=lambda: iter([(origin, bytes(audio.FS*2)), (origin+2*audio.FS, bytes(audio.FS*4))])))
    args = SimpleNamespace(wav=path, selection=selected['recording'], out=tmp_path,
        base=tmp_path, deep=False, runners=['besra'], p4=False, grace=1)
    assert module.decode(args) == 0
    assert [m.fed for m in instances] == [[origin], [origin+2*audio.FS]]
    summary = json.loads((tmp_path / 'monitor.json').read_text())
    assert summary['seconds'] == 4
    assert summary['gaps'] == [[1., 2.]]
    assert summary['selection']['bounds'] == [8000, 8032]


def test_plain_replay_keeps_absolute_origin_and_storage_gaps(sessions):
    from hfhost.audio import FS, WavReplaySource
    from hfmodem.tests.station.recording_support import flush, bounded
    import numpy as np
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=FS, clock='card')
    lane.push(1000*FS, np.zeros(FS, dtype=np.float32))
    flush(rec)
    lane.push(1002*FS, np.zeros(FS, dtype=np.float32))
    flush(rec)
    bounded(rec.close)
    from hfhost.recordings import available
    assert available(root / lane.row['source'])
    with WavReplaySource(root / lane.row['source'], realtime=False) as source:
        frames = list(source.frames())
    assert frames[0][0] == 1000 * FS
    assert sum(len(pcm)//2 for _,pcm in frames) == 2 * FS
    assert any(first == 1002 * FS for first,_ in frames)


def test_queued_viewer_job_keeps_its_context_when_contact_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import copy
    import threading
    selected = entry(tmp_path)
    save_selection(tmp_path, selected)
    jobs_module = reader('logbook_view.viewer_jobs')
    class Thread:
        def __init__(self, **kwargs): pass
        def is_alive(self): return True
        def start(self): pass
    monkeypatch.setattr(threading, 'Thread', Thread)
    book = SimpleNamespace(roots=(tmp_path,), viewer_root=tmp_path / 'derived',
                           entries=lambda: {'entries': [selected]})
    jobs = jobs_module.Replays(book)
    first = jobs.start('c2')
    original = copy.deepcopy(first['selection'])
    selected['recording']['context']['id'] = 'context-c3'
    second = jobs.start('c2')
    assert first['id'] != second['id']
    assert jobs.status(first['id'])['selection'] == original
    assert jobs.status(second['id'])['context_id'] == 'context-c3'


def test_cli_missing_context_reports_incomplete(tmp_path):
    selected = entry(tmp_path)
    del selected['recording']['context']
    path = save_selection(tmp_path, selected)
    result = reader('tools.rehear').rehear(path)
    assert result is not None
    assert result.against_log is None and 'incomplete' in result.note.lower()


def test_secondary_uses_its_own_bounds_and_restarts_after_gap(tmp_path, monkeypatch):
    from hfmodem.tests.station.test_recording_stage5_views import build_fixture
    from hfmodem.tests.station.recording_stage5_support import wave_file, write_json
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    rec = selected['recording']
    ref = dict(rec['lanes']['rx'], lane='witness', source='working/run/witness.wav',
               source_id='source-witness', index_id='index-witness', clock='kiwi', clock_generation=9,
               actual=[16000, 16032], selection=[16000, 16032], gaps=[[16008, 16016]])
    rec['lanes']['witness'] = ref
    rec['context']['lanes']['witness'] = {k:ref[k] for k in ('source_id','index_id','clock','clock_generation','rate')}
    rec['context']['lanes']['witness']['anchor'] = dict(sample=16000, transcript_s=1000., uncertainty_s=.002)
    rec['context']['lane_states'] = {'witness': dict(initial=rec['context']['initial'], changes=[])}
    witness = wave_file(path.with_name('witness.wav'), count=64)
    write_json(witness.with_suffix('.json'), dict(start_stream_sample=15984, end_stream_sample=16048,
        source_id='source-witness', index_id='index-witness', clock='kiwi', clock_generation=9))
    selected['audio'].append(dict(kind='sdr witness', url='/file/working/run/witness.wav', selection=ref))
    artifact = view.build(book, selected, tmp_path / 'derived')
    assert [len(call[0]) for call in calls] == [32, 8, 16]
    assert [call[3]['selection']['bounds'] for call in calls[1:]] == [[16000,16008], [16016,16032]]
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['sources'][1]['offset_s'] == 0
    assert manifest['sources'][1]['context']['state'] == 'complete'


def test_desk_named_family_never_adds_unbounded_duplicates(sessions, tmp_path, monkeypatch):
    from hfmodem.tests.station.test_recording_stage5_desk import book_and_arm
    from hfmodem.tests.station.recording_stage5_support import recorded, write_json
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4*48000)
    _, root, saved = recorded(sessions)
    _, book, provenance, _ = book_and_arm(tmp_path, monkeypatch)
    facts = json.loads(provenance.read_text())
    rec = dict(saved['contacts']['c2'], policy='everything', run_directory=str(root))
    rec['lanes']['rx']['selection'] = [40, 80]
    facts['recording'] = rec
    write_json(provenance, facts)
    (provenance.parent / 'arm.log').write_text('session stream: '+str(root / rec['lanes']['rx']['source'])+'\n')
    book.roots = (*book.roots, root)
    book.served = (*book.served, root)
    [shown] = book.entries()['entries']
    assert shown['audio']
    for resource in shown['audio']:
        assert resource['duration_s'] == 5
        assert min(p['start_s'] for p in resource['segments']) == 0
        assert max(p['start_s']+p['duration_s'] for p in resource['segments']) == 5
