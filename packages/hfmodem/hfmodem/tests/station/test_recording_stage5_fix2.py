# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Publication identity and bounded replay retain source facts."""
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core.contactlog import ContactLog
from hfmodem.core.recording import ContactBoundaries
from hfmodem.tests.station.recording_stage5_support import entry, reader, save_selection
from hfmodem.tests.station.recording_support import RATE, bounded, push


@pytest.mark.parametrize('successor_start', [24, 40])
def test_overlapping_postroll_keeps_each_contact_identity(sessions, successor_start):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    owner = ContactLog(root, recording=rec)
    a = dict(n=1, peer='W0QXR', mycall='N0CALL', bandwidth_hz=2750,
             dial_hz=7105000, started_utc='2026-11-01T12:00:00Z', how='incoming')
    b = dict(a, n=2, peer='K9ZZT', how='outgoing')
    push(rec, lane, 0, 24)
    owner.update(a, sample=24, sample_rate=RATE)
    if successor_start == 24:
        owner.update(b, sample=successor_start, sample_rate=RATE)
    first = owner.update(a | {'ended_utc': '2026-11-01T12:00:01Z'}, sample=32, sample_rate=RATE)
    if successor_start != 24:
        owner.update(b, sample=successor_start, sample_rate=RATE)
    second = owner.update(b | {'ended_utc': '2026-11-01T12:00:03Z'}, sample=48, sample_rate=RATE)
    push(rec, lane, 24, 96)
    bounded(rec.close)
    for native, peer, role in ((first, 'W0QXR', 'called'), (second, 'K9ZZT', 'caller')):
        saved = json.loads((Path(native['directory']) / 'provenance.json').read_text())['recording']
        chosen = reader('hfmodem.station.sessionlog').select(saved, root / saved['lanes']['rx']['source'])
        assert chosen.calls == {'local': 'N0CALL', 'peer': peer}
        assert chosen.local_role == role
        assert any(c.get('calls', {}).get('peer') == 'K9ZZT' for c in saved['context']['changes'])


@pytest.mark.parametrize('protocol', ['VARA', 'PACTOR'])
def test_live_owner_missing_replay_facts_are_explicit(sessions, protocol):
    published = []
    rec, root = sessions('everything', publish=published.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    owner = ContactBoundaries(rec, run='r')
    key = ('r', protocol, 'endpoint', 'N0CALL', 'W0QXR', 1)
    push(rec, lane, 0, 16)
    owner.established(key, 'W0QXR', {'card': 16})
    owner.ended(key, {'card': 24}, 'disconnect')
    push(rec, lane, 16, 72)
    bounded(rec.close)
    [saved] = published
    chosen = reader('hfmodem.station.sessionlog').select(dict(saved, policy='everything'),
        root / saved['lanes']['rx']['source'])
    assert not chosen.complete, 'boundary state is not a transcript or timing history'
    missing = ['transcript', 'ARQ epoch', 'TX', 'mute']
    if protocol == 'PACTOR':
        missing += ['probe', 'grid']
    assert all(name in chosen.reason for name in missing), chosen.reason
    assert chosen.bounds == (0, 64) and chosen.spans
    assert chosen.grading is None and chosen.log is None and not chosen.tx
    assert 'transcript' not in saved['context']


def monitor(tmp_path, monkeypatch, pcm, *, p4=False, gaps=(), event_times=()):
    from hfhost import audio
    view, driver = reader('logbook_view.rehear_arms'), reader('creance.monitor.driver')
    spec = SimpleNamespace(name='shrike', unavailable=lambda: None)
    monkeypatch.setattr(driver, 'default_specs', lambda *a, **k: [spec])
    @dataclass
    class Event:
        t: float
        kind: str = 'local leakage'
        modem: str = 'kestrel'
    class Monitor:
        def __init__(self, spec, found, **kwargs):
            self.spec, self.found, self.dropped_samples = spec, found, 0
        def start(self): pass
        def feed(self, first, raw):
            for t in event_times:
                if first <= round(t * 48000) < first + len(raw) // 2:
                    self.found.put(Event(t))
        def stop(self, grace): return True
    monkeypatch.setattr(driver, 'ModemMonitor', Monitor)
    def frames():
        first = 0
        for part in np.array_split(pcm, max(1, len(pcm) // 4800)):
            first += dict(gaps).get(first, 0)
            yield first, part.astype('<i2').tobytes()
            first += len(part)
    monkeypatch.setattr(audio, 'WavReplaySource', lambda *a, **k: SimpleNamespace(frames=frames))
    args = SimpleNamespace(wav=tmp_path / 'unused.wav', selection=None, out=tmp_path,
        base=tmp_path, deep=False, runners=['shrike'], p4=p4, grace=1)
    assert view.decode(args) == 0
    summary = json.loads((tmp_path / 'monitor.json').read_text())
    return view, summary, [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]


def test_monitor_long_keying_excludes_local_leakage(tmp_path, monkeypatch):
    fs = 48000
    rng = np.random.default_rng(7)
    quiet = rng.normal(0, 1000, fs)
    noise = rng.normal(0, 1, 8 * fs) * np.r_[np.full(fs, 2), np.full(6 * fs, 120), np.full(fs, 2)]
    leakage = noise + 500 * np.sin(2 * np.pi * 1500 * np.arange(8 * fs) / fs)
    view, summary, events = monitor(tmp_path, monkeypatch, np.r_[quiet, leakage, quiet],
        event_times=np.arange(2., 8., .2))
    assert len(events) == 30
    peers = []
    def classify(call, mycall, peer, *args):
        peers.extend(peer)
        return {}, {}
    monkeypatch.setattr(view, 'vara', classify)
    view.heard('vara', 'W0QXR', 'N0CALL', events, summary['spans'], None, source='rig rx')
    assert not peers, f'local leakage reached peer: {peers}'


def test_monitor_gap_resets_mute_baseline(tmp_path, monkeypatch):
    fs = 48000
    rng = np.random.default_rng(8)
    pcm = np.r_[rng.normal(0, 1000, fs), rng.normal(0, 2, 8 * fs)]
    _, summary, _ = monitor(tmp_path, monkeypatch, pcm, gaps=[(5 * fs, 2 * fs)])
    assert summary['spans'] == [[1., 5.]]


def test_monitor_complete_chirp_crossing_window_decodes_once(tmp_path, monkeypatch):
    from hfmodem.shrike import p4chirp
    detector = reader('corpuslib.detector')
    original = detector.chirp_events
    monkeypatch.setattr(detector, 'chirp_events', lambda audio, offset: original(audio, 0))
    packet = p4chirp.entry_packet(b'window crossing'.ljust(22, b'!'), 0x62)
    fs = 48000
    start = fs + fs // 2
    pcm = np.r_[np.zeros(start), packet * 12000, np.zeros(round(p4chirp.PACKET_S * fs))]
    _, _, events = monitor(tmp_path, monkeypatch, pcm, p4=True)
    chirps = [e for e in events if e['kind'].startswith('P4-CHIRP')]
    assert len(chirps) == 1, chirps
    assert chirps[0]['kind'] == 'P4-CHIRP-CRC'
    assert abs(chirps[0]['t'] - start / fs) < .02


@pytest.mark.parametrize('missing', ['transcript', 'arq_epoch', 'tx', 'muted', 'grid', 'probe'])
def test_selected_reader_requires_each_replay_fact(tmp_path, missing):
    selected = entry(tmp_path, protocol='pactor')
    ctx = selected['recording']['context']
    if missing in ('arq_epoch', 'muted'):
        ctx['initial'].pop(missing)
        for change in ctx['changes']:
            change.pop(missing, None)
    else:
        ctx.pop(missing)
    path = save_selection(tmp_path, selected)
    chosen = reader('hfmodem.station.sessionlog').select(selected, path, root=tmp_path)
    assert not chosen.complete and chosen.reason
    assert chosen.bounds == (8000, 8032) and chosen.spans
    assert chosen.grading is None and chosen.log is None and not chosen.tx


def test_state_json_cannot_supply_a_transcript(tmp_path):
    selected = entry(tmp_path)
    data = json.dumps(selected['recording']['context']['initial'])
    selected['recording']['context']['transcript'] = dict(text=data, range=[0, len(data)],
        source_id='state', sha256=hashlib.sha256(data.encode()).hexdigest(), first_s=0.)
    path = save_selection(tmp_path, selected)
    chosen = reader('hfmodem.station.sessionlog').select(selected, path, root=tmp_path)
    assert not chosen.complete and 'transcript' in chosen.reason
    assert chosen.log is None and chosen.grading is None


@pytest.mark.parametrize('missing', ['arq_epoch', 'muted', 'grid'])
def test_future_or_other_generation_state_does_not_complete_context(tmp_path, missing):
    selected = entry(tmp_path, protocol='pactor')
    ctx = selected['recording']['context']
    if missing == 'grid':
        for grid in ctx['grid']:
            grid['clock_generation'] = 9
    else:
        value = ctx['initial'].pop(missing)
        ctx['changes'].append(dict(sample=8004, clock_generation=2, **{missing: value}))
    path = save_selection(tmp_path, selected)
    chosen = reader('hfmodem.station.sessionlog').select(selected, path, root=tmp_path)
    assert not chosen.complete and chosen.reason


def test_monitor_incomplete_context_cannot_produce_attribution(tmp_path, monkeypatch):
    selected = entry(tmp_path, protocol='vara')
    selected['recording']['context'].pop('tx')
    save_selection(tmp_path, selected)
    module = reader('logbook_view.rehear_arms')
    job = module.plan(selected, tmp_path)
    pin = SimpleNamespace(root=tmp_path, out=tmp_path / 'derived', python=Path('/fake/python'),
        stage=tmp_path / 'stage', tree=tmp_path / 'tree', check=lambda *a: None,
        env=lambda scratch: {'PYTHONPATH': 'fake', 'PYTHONPYCACHEPREFIX': str(scratch)})
    pin.out.mkdir()
    def run_tool(argv, *, out, **kwargs):
        if 'decode' in argv:
            (out / 'monitor.json').write_text(json.dumps(dict(spans=[], residency=dict(set_sha256='set', modules={}))))
            (out / 'events.jsonl').write_text('')
        else:
            (out / 'rehear.json').write_text(json.dumps(dict(loaded={}, rehearings=[])))
    monkeypatch.setattr(module, 'run_tool', run_tool)
    out = module.rehear_arm(job, pin, dict(commit='test', set_sha256='set', label='test', lists_sha256='lists'),
        selected, mycall='N0CALL', audit=lambda *a: None)
    assert out['heard'] is None
    assert out['context']['state'] == 'incomplete' and 'TX' in out['context']['reason']


def test_monitor_cache_refuses_incomplete_context(tmp_path):
    selected = entry(tmp_path, protocol='vara')
    selected['recording']['context'].pop('tx')
    save_selection(tmp_path, selected)
    module = reader('logbook_view.rehear_arms')
    job = module.plan(selected, tmp_path)
    pin = SimpleNamespace(root=tmp_path, out=tmp_path / 'derived')
    pin.out.mkdir()
    old = dict(failed=[], heard={'reached': ['connect']}, identity=module.identity(job, pin, 'N0CALL'),
        audio=[dict(path=str(path.relative_to(tmp_path)), sha256=module.sha256(path, selection=job.selection))
               for _, path in job.audio()], rehearer=dict(reader=module.READER))
    (pin.out / 'c2.json').write_text(json.dumps(old))
    assert not module.done_already(pin, job, mycall='N0CALL')
