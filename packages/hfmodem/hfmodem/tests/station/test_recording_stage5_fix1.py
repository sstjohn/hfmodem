# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Reader regressions for selected recording identity, clocks and playback."""
import copy
import json
import os
from pathlib import Path
import shutil
import threading
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.tests import hermetic
from hfmodem.tests.station.recording_stage5_support import (
    RATE, START, STOP, entry, reader, save_selection, wave_file, write_json,
)
from hfmodem.tests.station.recording_support import bounded, flush, push, same_path
from hfmodem.tests.station.test_recording_stage5_desk import book_and_arm
from hfmodem.tests.station.test_recording_stage5_views import build_fixture


def test_context_read_does_not_block_pcm(sessions, monkeypatch):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    log = root / 'arm.log'
    log.write_text('selected transcript\n')
    entered, release, consumed = threading.Event(), threading.Event(), threading.Event()
    original_open, original_consume = Path.open, rec._consume
    def slow_open(path, *args, **kwargs):
        if same_path(path, log) and args == ('rb',):
            entered.set()
            assert release.wait(5)
        return original_open(path, *args, **kwargs)
    def consume(*args):
        original_consume(*args)
        consumed.set()
    monkeypatch.setattr(Path, 'open', slow_open)
    monkeypatch.setattr(rec, '_consume', consume)
    push(rec, lane, 0, 8)
    rec.begin('c', {'card': 8}, context={'transcript': {
        'path': 'arm.log', 'source_id': 'log', 'range': [0, log.stat().st_size]}})
    rec.end('c', {'card': 8})
    push(rec, lane, 8, 40)
    lane.push(40, np.zeros(8))
    assert entered.wait(5)
    consumed.clear()
    try:
        lane.push(48, np.zeros(8))
        assert consumed.wait(1), 'context hashing holds the PCM state lock'
        for first in range(56, 96, 8):
            consumed.clear()
            lane.push(first, np.zeros(8))
            assert consumed.wait(1)
    finally:
        release.set()
    bounded(rec.close)
    assert not lane.gaps


def test_desk_preserves_submitted_tx_kind(tmp_path, monkeypatch):
    _, book, provenance, selected = book_and_arm(tmp_path, monkeypatch)
    rec = selected['recording']
    rec['lanes']['tx'] = dict(rec['lanes'].pop('rx'), lane='tx')
    save_selection(tmp_path, entry(tmp_path))
    facts = json.loads(provenance.read_text())
    facts['recording'] = rec
    write_json(provenance, facts)
    [shown] = book.entries()['entries']
    assert [a['kind'] for a in shown['audio']] == ['submitted tx']
    assert shown['audio'][0]['label'] == 'Submitted TX'


def test_desk_policy_unmatched_source_is_unavailable(tmp_path, monkeypatch):
    _, book, provenance, selected = book_and_arm(tmp_path, monkeypatch)
    save_selection(tmp_path, selected)
    other = wave_file(provenance.parent / 'whole-session.wav')
    (provenance.parent / 'arm.log').write_text(f'session stream: {other}\n')
    [shown] = book.entries()['entries']
    unmatched = next(a for a in shown['audio'] if a.get('path') == str(other)
                     or a.get('url', '') == book.url(other))
    assert unmatched.get('url') is None, 'policy contact exposed a whole-session source'
    assert unmatched['state'] == 'unavailable'


@pytest.mark.parametrize('change', ['bounds', 'generation', 'source', 'gap'])
def test_identity_covers_secondary_selection(tmp_path, change):
    rec = entry(tmp_path)['recording']
    rec['lanes']['witness'] = dict(rec['lanes']['rx'], source='working/run/witness.wav')
    helper = reader('hfmodem.station.sessionlog')
    first = helper.identity(rec)
    ref = rec['lanes']['witness']
    if change == 'bounds':
        ref['selection'] = [8008, STOP]
    elif change == 'generation':
        ref['clock_generation'] += 1
    elif change == 'source':
        ref['source_id'] = 'replacement'
    else:
        ref['gaps'] = [[8008, 8016]]
    assert helper.identity(rec) != first


def test_besra_grading_excludes_padding_sessions(tmp_path):
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    helper = reader('hfmodem.station.sessionlog')
    chosen = helper.select(selected['recording'], path, root=tmp_path)
    assert chosen.complete
    assert len(chosen.grading) < len(chosen.log.frames)
    assert reader('tools.rehear.besra')._against(chosen.log, 1000., []) == (0, len(chosen.grading))


@pytest.mark.parametrize('damage', ['source_id', 'index_id', 'clock_generation', 'rate', 'geometry', 'root'])
def test_transport_refuses_selected_source_mismatch(tmp_path, damage):
    from hfhost.audio import WavReplaySource, AudioError
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    rec = selected['recording']
    rec['run_directory'] = str(tmp_path)
    side = json.loads(path.with_suffix('.json').read_text())
    if damage in ('source_id', 'index_id', 'clock_generation'):
        side[damage] = 'wrong'
    elif damage == 'rate':
        wave_file(path, count=32, rate=16)
    elif damage == 'geometry':
        side['end_stream_sample'] += 1
    else:
        path = save_selection(tmp_path / 'another-run', selected)
    if damage != 'root':
        write_json(path.with_suffix('.json'), side)
    with pytest.raises(AudioError, match='identity|mismatch|lane|geometry|rate|clock|source'):
        with WavReplaySource(path, realtime=False, selection=rec) as source:
            list(source.frames())


def test_station_replay_resolves_path_lane(tmp_path, monkeypatch):
    from hfmodem.core import config
    from hfmodem.station import process
    selected = entry(tmp_path)
    rec = selected['recording']
    rec['lanes']['tap'] = dict(rec['lanes']['rx'], lane='tap', source='working/run/tap.wav',
                              actual=[START, START+16], selection=[START, START+16])
    tap_entry = copy.deepcopy(selected)
    tap_entry['recording']['lanes'] = {'rx': rec['lanes']['tap']}
    path = save_selection(tmp_path, tap_entry)
    wave_file(path, count=16)
    side = json.loads(path.with_suffix('.json').read_text())
    side.update(recording=rec, end_stream_sample=START+16, samples=16)
    write_json(path.with_suffix('.json'), side)
    cfg = config.StationConfig(station=config.Station(mycall='N0CALL'), rig=config.Rig(cat='none'),
        audio=config.Audio(input='replay:'+str(path)), protocols={})
    station = process.Station(cfg)
    monkeypatch.setattr(station, 'validate_options', lambda: None)
    monkeypatch.setattr(cfg.__class__, 'require_audio', lambda *a, **k: None)
    audio = station.open_audio()
    try:
        assert len(station.replay) == 2*process.CARD_RATE_HZ
    finally:
        audio.close()


def secondary(root, selected, *, complete=True):
    rec = selected['recording']
    ref = dict(rec['lanes']['rx'], lane='witness', source='working/run/witness.wav',
               source_id='source-witness', index_id='index-witness', clock='kiwi',
               clock_generation=9, participation=[8016,8024])
    rec['lanes']['witness'] = ref
    rec['context']['lanes']['witness'] = {k:ref[k] for k in
        ('source_id','index_id','clock','clock_generation','rate')}
    if complete:
        rec['context']['lanes']['witness']['anchor'] = dict(sample=START, transcript_s=1000., uncertainty_s=.01)
    rec['context']['lane_states'] = {'witness': {'initial': rec['context']['initial'], 'changes': []}}
    path = wave_file(root / ref['source'], count=64)
    write_json(path.with_suffix('.json'), dict(start_stream_sample=START-16,
        end_stream_sample=STOP+16, samples=64, **{k:ref[k] for k in
        ('source_id','index_id','clock','clock_generation','rate')}))
    selected['audio'].append(dict(kind='sdr witness', url='/file/'+ref['source'], selection=ref))
    return path


def test_secondary_incomplete_mapping_never_aligns_or_attributes(tmp_path, monkeypatch):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    witness = secondary(tmp_path, selected, complete=False)
    def forbidden(*args):
        pytest.fail('policy secondary used legacy alignment')
    monkeypatch.setattr(view, 'align_audio', forbidden)
    original = view.load_audio
    def load(candidate, **kwargs):
        if candidate == witness:
            assert kwargs == {'start': START, 'stop': STOP}, 'unbounded secondary PCM'
        return original(candidate, **kwargs)
    monkeypatch.setattr(view, 'load_audio', load)
    artifact = view.build(book, selected, tmp_path / 'derived')
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['sources'][1]['offset_s'] is None
    assert calls[1][2] == []
    assert calls[1][3]['selection']['complete'] is False


def test_secondary_participation_does_not_inherit_primary(tmp_path, monkeypatch):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    secondary(tmp_path, selected)
    def replay(*args, **kwargs):
        return [dict(id='frame', t0=1.1, t1=1.5, kind='candidate', direction='observed',
                     status='candidate', symbols=None, role='caller')]
    monkeypatch.setattr(view, 'replay', replay)
    attributed = []
    original = view.vara_timing.attribute
    def attribute(events, *args, **kwargs):
        original(events, *args, **kwargs)
        attributed.append(events)
    monkeypatch.setattr(view.vara_timing, 'attribute', attribute)
    view.build(book, selected, tmp_path / 'derived')
    observed = attributed[1][0]
    assert observed['direction'] == 'observed' and not observed['in_qso']


@pytest.mark.parametrize('binding', ['selection', 'log'])
def test_incoming_ardop_caller_is_not_local_tx(tmp_path, monkeypatch, binding):
    view = reader('logbook_view.ardop_view')
    selected = entry(tmp_path)
    frame = SimpleNamespace(type=0x31, name='ConReq', symbol_rows=[], unverified=True,
        header_only=False, truncated=False, combined=False, ok=True, session_id=0x22,
        quality=90, cfo_hz=0, caller='W0QXR', target='N0CALL', payload=b'')
    class Rolling:
        def __init__(self, decoder, heard, **kwargs): self.heard = heard
        def push(self, audio): pass
        def pump(self): pass
        def flush(self): self.heard(0, frame)
    monkeypatch.setattr(view, 'RollingDecoder', Rolling)
    options = ({'selection': selected['recording']} if binding == 'selection'
               else {'log': SimpleNamespace(selection=selected['recording'])})
    events = view.replay(np.zeros(48000), 500, ['W0QXR','N0CALL'], tmp_path,
        **options, progress=lambda *a, **k: None)
    assert events[0]['direction'] != 'tx' and not events[0].get('tx_confirmed')


def test_shrike_native_grid_offsets_convert_once(tmp_path, monkeypatch):
    selected = entry(tmp_path, protocol='pactor')
    rec = selected['recording']
    rec['lanes']['rx']['rate'] = 12000
    rec['lanes']['rx']['selection'] = rec['lanes']['rx']['actual'] = [12000,24000]
    ctx = rec['context']
    ctx['lanes']['rx'].update(rate=12000, anchor=dict(sample=12000, transcript_s=1000., uncertainty_s=.01))
    ctx['grid'] = [dict(sample=15000, clock_generation=2, word='CS1')]
    path = save_selection(tmp_path, selected)
    wave_file(path, count=12000, rate=12000)
    side = json.loads(path.with_suffix('.json').read_text())
    side.update(start_stream_sample=12000, end_stream_sample=24000, samples=12000,
                peak=0., rms=0., samplerate=12000)
    write_json(path.with_suffix('.json'), side)
    helper = reader('hfmodem.station.sessionlog')
    chosen = helper.select(rec, path)
    assert chosen.complete
    shrike = reader('tools.rehear.shrike')
    seen = []
    monkeypatch.setattr(shrike, '_anchored', lambda audio, anchors: seen.append(anchors) or [])
    monkeypatch.setattr(shrike, '_rolling', lambda *a: [])
    monkeypatch.setattr(shrike, '_ungated', lambda *a: [])
    shrike._rehear(path, chosen.log, selection=rec)
    assert seen == [[3000*shrike.FS//12000]]


@pytest.mark.parametrize('consumer', ['build', 'jobs'])
def test_unavailable_placeholder_isolated_before_url_handling(tmp_path, monkeypatch, consumer):
    view, book, selected, path, calls = build_fixture(tmp_path, monkeypatch)
    selected['audio'].insert(0, dict(kind='rig rx', url=None, state='unavailable'))
    if consumer == 'build':
        view.build(book, selected, tmp_path/'derived')
        assert len(calls) == 1
    else:
        class Thread:
            def __init__(self, **kwargs): pass
            def start(self): pass
            def is_alive(self): return True
        monkeypatch.setattr(threading, 'Thread', Thread)
        book.entries = lambda: {'entries': [selected]}
        job = reader('logbook_view.viewer_jobs').Replays(book).start('c2')
        assert job['state'] == 'queued'


def test_selected_view_ignores_missing_unselected_segment(sessions, tmp_path, monkeypatch):
    from hfmodem.core import wav
    view, book, selected, _, calls = build_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4*48000)
    rec, root = sessions('everything', directory=tmp_path/'working/recorded')
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, START, START+96)
    bounded(rec.close)
    ref = selected['recording']['lanes']['rx']
    path = root / lane.row['source']
    ref.update(source=str(path.relative_to(tmp_path)), source_id=lane.row['source_id'], index_id=lane.row['index_id'])
    selected['recording']['context']['lanes']['rx'].update(source_id=ref['source_id'], index_id=ref['index_id'])
    selected['audio'] = [dict(kind='rig rx', url='/file/'+ref['source'], selection=ref)]
    parts = wav.segments(path)
    parts[-1].unlink()
    view.build(book, selected, tmp_path/'derived')
    assert len(calls) == 1 and len(calls[0][0]) == STOP-START


def test_contactlog_owner_publishes_readable_context_without_fixture(sessions, tmp_path):
    from hfmodem.core.contactlog import ContactLog
    rec, root = sessions('everything', directory=tmp_path/'working/run')
    lane = rec.stream('rx', rate=RATE, clock='card')
    (root/'arm.log').write_text('Contact with W0QXR\n')
    owner = ContactLog(root, recording=rec)
    facts = dict(n=1, peer='W0QXR', mycall='N0CALL', dial_hz=7105000, bandwidth_hz=2750,
                 started_utc='2026-11-01T12:00:00Z', how='incoming')
    push(rec, lane, 0, 24)
    owner.update(facts, sample=24, sample_rate=RATE)
    native = owner.update(facts | {'ended_utc': '2026-11-01T12:00:01Z'}, sample=32, sample_rate=RATE)
    push(rec, lane, 24, 80)
    bounded(rec.close)
    saved = json.loads((Path(native['directory'])/'provenance.json').read_text())['recording']
    chosen = reader('hfmodem.station.sessionlog').select(saved, root/saved['lanes']['rx']['source'])
    assert not chosen.complete
    assert all(name in chosen.reason for name in ('transcript', 'ARQ epoch', 'TX', 'mute'))
    assert chosen.bounds == (0, 72) and chosen.spans
    assert chosen.log is None and chosen.grading is None
    assert chosen.calls == {'local': 'N0CALL', 'peer': 'W0QXR'}
    at_edge = reader('hfmodem.station.sessionlog').select(saved, root/saved['lanes']['rx']['source'], start=24)
    assert chosen.local_role == 'called' and at_edge.state == {'expected_session': None}
    assert any(c.get('bandwidth_hz') == 2750 for c in saved['context']['changes'])


@pytest.mark.parametrize('waveform', ['voice','cw'])
def test_voice_replay_finishes_each_span_through_output(voice_owner, waveform, monkeypatch):
    made = voice_owner('off', waveform)
    owner = made.session
    selected = entry(made.directory, gaps=[[8008,8016]])
    owner.latest = dict(audio=np.arange(32, dtype=np.float32)/32768, rate=RATE,
                        recording=selected['recording'])
    starts = []
    output = reader('hfmodem.tests.station.recording_stage4_support').Output
    original = output.start
    def start(stream):
        starts.append(owner.playing['draft']['source_start_sample'])
        original(stream)
    monkeypatch.setattr(output, 'start', start)
    owner.replay_heard()
    assert starts == [8000,8016]
    assert owner.playing is None


@pytest.mark.parametrize('consumer', ['witness','kestrel','besra','shrike'])
def test_analysis_loaders_choose_bounds_before_allocation(tmp_path, monkeypatch, consumer):
    from hfmodem.core import wav
    path = wave_file(tmp_path/'legacy.wav', count=96, rate=48000)
    original = wav.read
    def read(path, *args, **kwargs):
        assert kwargs.get('start') == 0 and kwargs.get('stop') == 96, 'analysis allocated unbounded family'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(wav, 'read', read)
    if consumer == 'witness':
        reader('tools.witness_align').read(path)
    elif consumer == 'kestrel':
        reader('tools.rehear.kestrel')._load(path)
    elif consumer == 'besra':
        module = reader('tools.rehear.besra')
        monkeypatch.setattr(module, 'read_pcm', lambda *a, **k: pytest.fail('whole-segment PCM allocation'), raising=False)
        module._capture(path)
    else:
        reader('hfmodem.shrike.session').load_wav(str(path))


def test_monitor_replay_streams_before_consuming_next_frame(tmp_path, monkeypatch):
    from hfhost import audio
    view = reader('logbook_view.rehear_arms')
    driver = reader('creance.monitor.driver')
    fed = []
    spec = SimpleNamespace(name='besra', unavailable=lambda: None)
    monkeypatch.setattr(driver, 'default_specs', lambda *a, **k: [spec])
    class Monitor:
        def __init__(self, spec, *a, **k): self.spec, self.dropped_samples = spec, 0
        def start(self): pass
        def feed(self, first, pcm): fed.append(first)
        def stop(self, grace): return True
    monkeypatch.setattr(driver, 'ModemMonitor', Monitor)
    def frames():
        yield 0, bytes(4800*2)
        assert fed == [0], 'monitor materialized audio before starting readers'
        yield 4800, bytes(4800*2)
    monkeypatch.setattr(audio, 'WavReplaySource', lambda *a, **k: SimpleNamespace(frames=frames))
    args = SimpleNamespace(wav=tmp_path/'unused.wav', selection=None, out=tmp_path,
        base=tmp_path, deep=False, runners=['besra'], p4=False, grace=1)
    assert view.decode(args) == 0


def test_armsource_follows_operating_tap_index(sessions, tmp_path, monkeypatch):
    from hfmodem.core import capture_identity as ci
    arm = reader('consolelib.armsource')
    root = tmp_path/'arms/vara-c2'
    rec, _ = sessions('everything', directory=root/'tap')
    lane = rec.stream('tap', rate=48000, clock='tap')
    lane.push(0, np.zeros(4800))
    flush(rec)
    bounded(rec.close)
    write_json(root/'tap.json', dict(lane.row, recording_policy='everything',
                                    recording_index='tap/recording.json'))
    write_json(root/'provenance.json', dict(arm=root.name, dial_hz=7105000))
    (root/'arm.log').write_text('')
    monkeypatch.setattr(arm.rigsource, 'station_config', lambda *a: None)
    with arm.ArmSource(root) as source:
        assert source.tap == ci._absolute(root/'tap'/lane.row['source'])
        assert sum(len(pcm)//2 for _,pcm in source.frames()) == 4800


@pytest.mark.posix_only("witness_stop is tools/onair.sh bash; Windows stops the witness in consolelib.launch_capture")
def test_selected_witness_follows_operating_index(sessions, tmp_path):
    from hfmodem.tests.station.test_recording_stage5_witness import stop
    local = entry(tmp_path, protocol='pactor', source='working/run/stream.wav')
    local_path = save_selection(tmp_path, local)
    write_json(tmp_path/'recording.json', dict(policy='contacts', contacts={'c2':local['recording']}))
    rec, root = sessions('contacts', directory=tmp_path/'witness')
    lane = rec.stream('witness', rate=RATE, clock='kiwi-1')
    push(rec, lane, 0, 16)
    rec.begin('c2', {'kiwi-1':16}, anchors={'witness': dict(clock='kiwi-1', parent_clock='card',
        sample=8016, position=16, rate=RATE, parent_rate=RATE, uncertainty_s=.01)})
    rec.end('c2', {'kiwi-1':24})
    push(rec, lane, 16, 64)
    bounded(rec.close)
    write_json(tmp_path/'witness.json', dict(lane.row, recording_policy='contacts',
        recording_index='witness/recording.json', sample_rate=RATE, samplerate=RATE))
    (tmp_path/'witness.log').write_text('WITNESS COVERAGE: contact-only; retained 64 samples\n')
    result = stop(tmp_path, 'contacts')
    assert result.returncode == 0, result.stderr
    assert 'ALIGN' in result.stdout and str(local_path) in result.stdout
    assert '/witness/contact-' in result.stdout


def test_rehear_job_identity_changes_with_secondary(tmp_path):
    selected = entry(tmp_path)
    save_selection(tmp_path, selected)
    secondary(tmp_path, selected)
    module = reader('logbook_view.rehear_arms')
    pin = SimpleNamespace(root=tmp_path)
    job = module.plan(selected, tmp_path)
    first = module.identity(job, pin, 'N0CALL')
    selected['recording']['lanes']['witness']['clock_generation'] += 1
    replacement = module.plan(selected, tmp_path)
    assert module.identity(replacement, pin, 'N0CALL') != first


def test_grid_uses_requested_lane_generation(tmp_path):
    selected = entry(tmp_path, protocol='pactor')
    save_selection(tmp_path, selected)
    secondary(tmp_path, selected)
    rec = selected['recording']
    rec['context']['grid'] += [dict(sample=8018, clock_generation=9, word='CS2')]
    helper = reader('hfmodem.station.sessionlog')
    grid = helper.shrike_grid(tmp_path/rec['context']['transcript']['path'], RATE,
                              selection=dict(rec, lane='witness'))
    assert grid.clock_generation == 9 and grid.aimed == (8018,)


def test_rehear_plan_and_hash_ignore_unselected_family_damage(sessions, tmp_path, monkeypatch):
    from hfmodem.core import wav
    selected = entry(tmp_path)
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4*48000)
    owner, root = sessions('everything', directory=tmp_path/'working/source')
    lane = owner.stream('rx', rate=RATE, clock='card')
    push(owner, lane, START, START+96)
    bounded(owner.close)
    path = root/lane.row['source']
    ref = selected['recording']['lanes']['rx']
    ref.update(source=str(path.relative_to(tmp_path)), source_id=lane.row['source_id'], index_id=lane.row['index_id'])
    selected['recording']['context']['lanes']['rx'].update(source_id=ref['source_id'], index_id=ref['index_id'])
    selected['audio'] = [dict(kind='rig rx', url='/file/'+ref['source'])]
    module = reader('logbook_view.rehear_arms')
    parts = wav.segments(path)
    parts[-1].unlink()
    job = module.plan(selected, tmp_path)
    assert job is not None, 'whole-family availability rejected selected source'
    assert module.sha256(path, selection=selected['recording'])


@pytest.mark.parametrize('expected', [None, 0x22])
def test_live_boundary_publishes_its_own_source_state(sessions, expected):
    from hfmodem.core.recording import ContactBoundaries
    published = []
    owner, root = sessions('everything', publish=published.append)
    lane = owner.stream('rx', rate=RATE, clock='card')
    boundaries = ContactBoundaries(owner, run='r')
    key = ('r','VARA','endpoint','N0CALL','W0QXR',1)
    push(owner, lane, 0, 16)
    if expected is not None:
        owner.state('rx', 8, expected_session=expected, arq_epoch=4, mode='normal')
    cid = boundaries.established(key, 'W0QXR', {'card':16})
    boundaries.ended(key, {'card':24}, 'disconnect')
    push(owner, lane, 16, 72)
    bounded(owner.close)
    [record] = published
    assert record['id'] == cid
    result = reader('hfmodem.station.sessionlog').select(dict(record, policy='everything'),
        root/record['lanes']['rx']['source'])
    assert not result.complete
    assert all(name in result.reason for name in ('transcript', 'TX', 'mute'))
    assert 'ARQ epoch' in result.reason
    assert result.calls == {'local':'N0CALL','peer':'W0QXR'}
    at_edge = reader('hfmodem.station.sessionlog').select(dict(record, policy='everything'),
        root/record['lanes']['rx']['source'], start=16)
    assert ('ARQ epoch' in at_edge.reason) == (expected is None)
    assert at_edge.state == {'expected_session': None} and result.grading is None
    changes = record['context']['changes']
    assert next((c['expected_session'] for c in changes if 'expected_session' in c), None) == expected


def test_armsource_follows_contact_only_tap_index(sessions, tmp_path, monkeypatch):
    arm = reader('consolelib.armsource')
    root = tmp_path/'arms/vara-c2'
    rec, _ = sessions('contacts', directory=root/'tap')
    lane = rec.stream('tap', rate=48000, clock='tap')
    lane.push(0, np.zeros(4800))
    flush(rec)
    rec.begin('c2', {'tap':4800})
    rec.end('c2', {'tap':4800})
    bounded(rec.close)
    write_json(root/'tap.json', dict(lane.row, recording_policy='contacts',
                                    recording_index='tap/recording.json'))
    write_json(root/'provenance.json', dict(arm=root.name, dial_hz=7105000))
    (root/'arm.log').write_text('')
    monkeypatch.setattr(arm.rigsource, 'station_config', lambda *a: None)
    with arm.ArmSource(root) as source:
        assert sum(len(pcm)//2 for _,pcm in source.frames()) == 4800


@pytest.mark.parametrize('missing', [0,2])
def test_selected_unindexed_family_ignores_other_segments(tmp_path, monkeypatch, missing):
    from hfmodem.core import wav
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4*48000)
    path = tmp_path/'family.wav'
    writer = wav.SegmentedWriter(path, rate=RATE, sample_clock=True, closed=lambda _: None)
    writer.set_position(START)
    writer.writeframes(np.arange(96,dtype='<i2').tobytes())
    writer.close()
    writer.paths[missing].unlink()
    view = reader('logbook_view.vara_view')
    monkeypatch.setattr(view, 'FS', RATE)
    result, rate = view.load_audio(path, start=START+32, stop=START+64)
    assert rate == RATE
    np.testing.assert_array_equal(result, np.arange(32,64)/32768)


def test_unhinted_secondary_pactor_has_no_inherited_calls(tmp_path, monkeypatch):
    module = reader('logbook_view.pactor_view')
    seen = []
    def decode(*args, **kwargs):
        seen.append(kwargs.get('mycall'))
        return []
    monkeypatch.setattr(module.rxfront, 'decode_events', decode)
    monkeypatch.setattr(module.p3rx, 'decode_p3_packets', lambda *a, **k: SimpleNamespace(packets=[]))
    monkeypatch.setattr(module.p1rx, 'decode_p1_packets', lambda *a, **k: [])
    monkeypatch.setattr(module.p2rx, 'find_markers', lambda *a, **k: [])
    module.replay(np.zeros(4800), 2400, [], tmp_path,
        selection=dict(complete=False, context={}), progress=lambda *a, **k: None)
    assert seen and all(call is None for call in seen)


def test_rehear_plan_never_claims_submitted_tx_as_receiver(tmp_path):
    selected = entry(tmp_path)
    rx = save_selection(tmp_path, selected)
    tx = wave_file(tmp_path/'working/run/submitted.wav', count=32)
    selected['audio'].append(dict(kind='submitted tx', url='/file/working/run/submitted.wav'))
    selected['recording']['lanes']['tx'] = dict(selected['recording']['lanes']['rx'],
        lane='tx', source='working/run/submitted.wav', source_id='submitted', index_id='tx-index')
    write_json(tx.with_suffix('.json'), dict(source_id='submitted', index_id='tx-index',
        start_stream_sample=START, end_stream_sample=STOP, samples=32))
    job = reader('logbook_view.rehear_arms').plan(selected, tmp_path)
    assert job.claimed == [rx], 'submitted TX was scheduled as receiver audio'
    assert job.audio() == [('rig rx', rx)]


def test_station_discovers_healthy_selected_part_of_moved_family(sessions, tmp_path, monkeypatch):
    from hfmodem.core import config, wav
    from hfmodem.station import process
    selected = entry(tmp_path)
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4*48000)
    owner, root = sessions('everything', directory=tmp_path/'working/source')
    lane = owner.stream('rx', rate=RATE, clock='card')
    push(owner, lane, START, START+96)
    bounded(owner.close)
    path = root/lane.row['source']
    ref = selected['recording']['lanes']['rx']
    ref.update(source=str(path.relative_to(tmp_path)), source_id=lane.row['source_id'], index_id=lane.row['index_id'])
    selected['recording']['run_directory'] = str(tmp_path)
    selected['recording']['context']['lanes']['rx'].update(source_id=ref['source_id'], index_id=ref['index_id'])
    side = json.loads(path.with_suffix('.json').read_text())
    write_json(path.with_suffix('.json'), dict(side, recording=selected['recording']))
    wav.segments(path)[-1].unlink()
    assert process.recording(config.Audio(input='replay:'+str(path))) == path
    assert len(process._read_replay(path)) == 4*process.CARD_RATE_HZ


def test_a_long_legacy_source_is_analysed_whole_within_its_bounds(tmp_path, monkeypatch):
    from hfmodem.core import wav
    path = wave_file(tmp_path/'long.wav', count=1801, rate=1)
    original = wav.read
    def read(candidate, *args, **kwargs):
        assert kwargs.get('start') == 0 and kwargs.get('stop') == 1801
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(wav, 'read', read)
    assert len(reader('hfmodem.shrike.session').load_wav(str(path))) == 1801


def test_witness_unknown_flac_length_is_counted_before_bounded_read(tmp_path, monkeypatch):
    from hfmodem.core import wav
    path = wave_file(tmp_path/'unknown.wav', count=64, rate=RATE)
    encoded = path.with_suffix('.flac')
    if os.name == 'nt' and shutil.which('flac') is None:
        pytest.skip('flac is not installed on this Windows host')
    argv = ['flac','--silent','-o',str(encoded),str(path)]
    hermetic.guard.admit(argv, os.environ)
    subprocess.run(argv, check=True, timeout=5)
    data = bytearray(encoded.read_bytes())
    packed = int.from_bytes(data[18:26], 'big') & ~((1<<36)-1)
    data[18:26] = packed.to_bytes(8, 'big')
    encoded.write_bytes(data)
    original = wav.read
    def read(candidate, *args, **kwargs):
        assert kwargs == dict(start=0, stop=64)
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(wav, 'read', read)
    result, rate = reader('tools.witness_align').read(encoded)
    assert rate == RATE
    np.testing.assert_array_equal(result, np.arange(64)/32768)
    assert encoded.read_bytes() == data


def test_unknown_length_invalid_flac_is_unavailable_without_abort(tmp_path):
    path = tmp_path/'invalid.flac'
    packed = (RATE<<44) | (15<<36)
    path.write_bytes(b'fLaC'+b'\x80'+(34).to_bytes(3,'big')+bytes(10)+packed.to_bytes(8,'big')+bytes(16))
    assert reader('logbook_view.audio').duration_s(path, {}) is None


def test_identity_uses_effective_bounds_for_every_lane(tmp_path):
    rec = entry(tmp_path)['recording']
    rec['lanes']['rx']['selection'] = [START-8, STOP+8]
    rec['lanes']['witness'] = dict(rec['lanes']['rx'], actual=[START+8, STOP-8])
    ident = reader('hfmodem.station.sessionlog').identity(rec)
    assert ident['bounds'] == [START, STOP]
    assert ident['sources']['witness']['bounds'] == [START+8, STOP-8]
