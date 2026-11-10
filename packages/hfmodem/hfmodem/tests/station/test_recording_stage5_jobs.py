# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Replay work is keyed by contact selection, source generation and context."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.core.capture_identity import IdentityError
from hfmodem.tests.station.recording_stage5_support import (
    START, STOP, clone, entry, reader, save_selection, wave_file, write_json,
)


def planned(root, protocol='vara'):
    module = reader('logbook_view.rehear_arms')
    selected = entry(root, protocol=protocol)
    path = save_selection(root, selected)
    job = module.plan(selected, root, mycall='N0CALL')
    assert job is not None
    return module, selected, path, job


@pytest.mark.parametrize('protocol', ['vara', 'ardop', 'pactor'])
def test_plan_dispatches_renamed_cutout_and_carries_selection(tmp_path, protocol):
    module, selected, path, job = planned(tmp_path, protocol)
    assert job.protocol == protocol and job.primary == ('rig rx', path)
    assert job.claimed == [path], 'selected recordings must be claimed by protocol, regardless of filename'
    assert job.selection == selected['recording']
    assert job.bounds == (START, STOP) and job.clock_generation == 2
    assert job.context_id == 'context-c2' and job.context_version == 1


@pytest.mark.parametrize('change', ['bounds', 'clock', 'context-id', 'context-version'])
def test_rehear_identity_separates_same_audio_with_different_context(tmp_path, change):
    module, selected, path, first = planned(tmp_path)
    second = clone(selected)
    rec = second['recording']
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
    other = module.plan(second, tmp_path, mycall='N0CALL')
    pin = SimpleNamespace(root=tmp_path)
    assert module.identity(first, pin, 'N0CALL') != module.identity(other, pin, 'N0CALL')


def test_rehear_arm_passes_selection_to_monitor_and_adapter_subprocesses(tmp_path, monkeypatch):
    module, selected, path, job = planned(tmp_path)
    job.claimed = [path]
    pin = SimpleNamespace(root=tmp_path, out=tmp_path / 'derived', python=Path('/fake/python'),
        stage=tmp_path / 'stage', tree=tmp_path / 'tree',
        env=lambda scratch: {'PYTHONPATH': 'fake', 'PYTHONPYCACHEPREFIX': str(scratch)})
    pin.out.mkdir()
    calls = []
    def run_tool(argv, **kwargs):
        payload = json.loads(Path(argv[argv.index('--selection') + 1]).read_text()) if '--selection' in argv else None
        calls.append((argv, payload))
        return 'decoder intentionally absent'
    monkeypatch.setattr(module, 'run_tool', run_tool)
    out = module.rehear_arm(job, pin, {'commit': 'test', 'set_sha256': 'set',
        'label': 'test', 'lists_sha256': 'lists'}, selected, mycall='N0CALL', audit=lambda *a: None)
    assert len(calls) == 4
    for argv, payload in calls:
        assert '--selection' in argv, 'monitor and adapter subprocesses both require the selected interval/context'
        assert payload == selected['recording']
    assert out['identity']['context_id'] == 'context-c2'
    assert out['identity']['bounds'] == [START, STOP]


def test_own_windows_filters_identity_clock_generation_and_interval(tmp_path, monkeypatch):
    module = reader('logbook_view.rehear_arms')
    identity = reader('hfmodem.core.capture_identity')
    selected = entry(tmp_path, protocol='pactor')
    selected['capture_id'] = 'one-capture'
    selected['window_clock_id'] = 'card'
    paths = [wave_file(tmp_path / f'hold_{i:02d}.wav') for i in range(4)]
    rows = [dict(source_id='source-rx', clock_generation=2, actual=bounds)
            for bounds in ([7990, 7998], [8001, 8007], [8028, 8038])]
    rows.append(dict(source_id='source-rx', clock_generation=1, actual=[8001, 8007]))
    for path, row in zip(paths, rows):
        write_json(path.with_suffix('.json'), row | {'capture_id': 'one-capture', 'clock_id': 'card'})
    monkeypatch.setattr(identity, 'recording_windows', lambda found, **kwargs: list(found))
    got = module.own_windows(selected, tmp_path, paths, None)
    assert got == paths[1:3], 'capture ID alone must not select another interval or clock generation'


@pytest.mark.parametrize('operation', ['plan', 'sort', 'hash'])
def test_batch_run_isolates_each_sources_early_failure(tmp_path, monkeypatch, operation):
    module, good, path, job = planned(tmp_path)
    bad = clone(good)
    bad['id'] = 'bad'
    bad_path = wave_file(tmp_path / 'working/run/bad.wav')
    bad['audio'] = [{'kind': 'rig rx', 'url': '/file/working/run/bad.wav'}]
    pin = SimpleNamespace(out=tmp_path / 'derived', root=tmp_path,
        prepare=lambda: {'commit': 'test'}, untouched=lambda: None)
    monkeypatch.setattr(module.Pinned, 'at', lambda *a: pin)
    monkeypatch.setattr(module, 'logbook', lambda *a: ({'call': 'N0CALL'}, [bad, good]))
    original_plan = module.plan
    def plan(e, *a, **k):
        if operation == 'plan' and e['id'] == 'bad':
            raise IdentityError('bad source during plan')
        return original_plan(e, *a, **k)
    monkeypatch.setattr(module, 'plan', plan)
    original_seconds = module.seconds
    def seconds(p):
        if operation == 'sort' and p == bad_path:
            raise FileNotFoundError(str(p))
        return original_seconds(p)
    monkeypatch.setattr(module, 'seconds', seconds)
    def done(pin, job, **kwargs):
        if operation == 'hash' and job.arm == 'bad':
            raise FileNotFoundError('moved while hashing')
        return False
    monkeypatch.setattr(module, 'done_already', done)
    monkeypatch.setattr(module, 'Gate', lambda *a: SimpleNamespace(wait=lambda: None, pauses=[], paused_s=0))
    completed = []
    monkeypatch.setattr(module, 'rehear_arm', lambda job, *a, **k:
        completed.append(job.arm) or {'heard': {'reached': []}, 'failed': []})
    args = SimpleNamespace(root=tmp_path, commit='test', arm=[], workers=1)
    assert module.run(args) == 0
    assert 'c2' in completed


def test_cli_dispatches_recorded_protocol_without_neighbor_log_lookup(tmp_path, monkeypatch):
    selected = entry(tmp_path, protocol='ardop')
    path = save_selection(tmp_path, selected)
    module = reader('tools.rehear')
    calls = []
    monkeypatch.setattr(module.sessionlog, 'find', lambda *a: pytest.fail('selected replay searched a neighboring transcript'))
    def run(path, log, *args, **kwargs):
        calls.append((path, log, kwargs))
        return 'selected result'
    monkeypatch.setattr(module, 'adapters', lambda: [('besra', lambda path: False, run)])
    assert module.rehear(path) == 'selected result'
    assert calls[0][0] == path


def test_cli_main_continues_after_root_discovery_failure(tmp_path, monkeypatch):
    module = reader('tools.rehear')
    broken, healthy = tmp_path / 'broken', tmp_path / 'healthy'
    broken.mkdir()
    path = wave_file(healthy / 'good.wav')
    original = Path.rglob
    def scan(root, pattern):
        if root == broken:
            raise FileNotFoundError('root moved')
        return original(root, pattern)
    monkeypatch.setattr(Path, 'rglob', scan)
    calls = []
    monkeypatch.setattr(module, 'rehear', lambda p, *a, **k: calls.append(p))
    assert module.main([str(broken), str(healthy), '--quiet']) == 0
    assert calls == [path]


def test_viewer_worker_identity_and_command_include_selection(tmp_path):
    module = reader('logbook_view.viewer_jobs')
    selected = entry(tmp_path, protocol='vara')
    save_selection(tmp_path, selected)
    commands = []
    done = threading.Event()
    def run(argv, **kwargs):
        commands.append(argv)
        done.set()
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    book = SimpleNamespace(roots=(tmp_path,), viewer_root=tmp_path / 'derived',
        entries=lambda: {'entries': [selected]})
    jobs = module.Replays(book, script=tmp_path / 'vara_view.py', run=run)
    job = jobs.start('c2')
    assert done.wait(3)
    worker = jobs.worker
    if worker is not None:
        worker.join(3)
        assert not worker.is_alive()
    assert job.get('context_id') == 'context-c2'
    assert job.get('clock_generation') == 2 and job.get('bounds') == [START, STOP]
    assert '--selection' in commands[0]


def test_derived_lookup_rejects_another_context_for_the_same_arm(tmp_path):
    module = reader('logbook_view.viewer')
    selected = entry(tmp_path)
    write_json(tmp_path / 'c2/old/manifest.json', {'schema': 1, 'arm': 'c2',
        'created_utc': '2026-11-01', 'decoder': {'source_sha256': 'same'},
        'selection': {'bounds': [0, 16], 'clock_generation': 1, 'context_id': 'context-c1', 'context_version': 1}})
    assert module.artifacts(tmp_path, 'c2', selection=selected['recording']) == []


def test_desk_rehearing_lookup_rejects_another_contact_context(tmp_path):
    face = reader('desk.face')
    selected = entry(tmp_path)
    write_json(tmp_path / 'working/rehear/old/c2.json', {'arm': 'c2', 'rehearer': {'commit': 'old', 'run_utc': '2026-11-01'},
        'heard': {}, 'limits': [], 'identity': {'bounds': [0, 16], 'clock_generation': 1, 'context_id': 'context-c1', 'context_version': 1}})
    book = face.Logbook(tmp_path, station={'mycall': 'N0CALL'})
    assert book.rehears({'c2': selected['recording']}) == {}


@pytest.mark.parametrize('command', ['adapter', 'monitor', 'viewer'])
def test_replay_cli_accepts_the_selection_sent_by_jobs(tmp_path, monkeypatch, command):
    selected = entry(tmp_path, protocol='vara')
    path = save_selection(tmp_path, selected)
    payload = write_json(tmp_path / 'selection.json', selected['recording'])
    calls = []
    if command == 'adapter':
        module = reader('tools.rehear')
        monkeypatch.setattr(module, 'rehear', lambda *a, **k: calls.append((a, k)))
        argv = [str(path), '--quiet', '--selection', str(payload)]
    elif command == 'monitor':
        module = reader('logbook_view.rehear_arms')
        monkeypatch.setattr(module, 'decode', lambda args: calls.append(args) or 0)
        argv = ['decode', str(path), '--out', str(tmp_path), '--base', str(tmp_path),
                '--runners', 'kestrel', '--selection', str(payload)]
    else:
        module = reader('logbook_view.vara_view')
        face = reader('desk.face')
        monkeypatch.setattr(face, 'Logbook', lambda *a, **k:
            SimpleNamespace(entries=lambda: {'entries': [selected]}))
        monkeypatch.setattr(module, 'build', lambda *a, **k: calls.append((a, k)))
        argv = ['--root', str(tmp_path), '--arm', 'c2', '--selection', str(payload)]
    assert module.main(argv) == 0
    assert len(calls) == 1
