from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid
import pytest
import json
from hfmodem.core import capture_identity as ci

ROOT=Path(__file__).resolve().parents[5]
def test_registered_mirrors_are_selected_by_frozen_role(tmp_path):
 c=ci.allocate_arm(tmp_path,native_root=tmp_path/'native'/'arm')
 roots=[c.root(role) for role in c.record['roots']]
 assert ci.select_roots(roots)==c.root('capture')
 assert ci.select_roots(roots,preferred_role='native')==c.root('native')
 with pytest.raises(ci.IdentityError):ci.select_roots(roots,preferred_role='absent')

@pytest.mark.parametrize('case',['two_legacy','different_registry','copied_identity','missing_new_manifest'])
def test_duplicate_or_incomplete_roots_cannot_choose_first(tmp_path,case):
 c=ci.allocate_arm(tmp_path/'one',native_root=tmp_path/'one-native'/'arm')
 if case=='two_legacy':
  paths=[tmp_path/'legacy1',tmp_path/'legacy2']
  for p in paths:p.mkdir()
 elif case=='different_registry':
  other=ci.allocate_arm(tmp_path/'two');paths=[c.root('capture'),other.root('capture')]
 elif case=='copied_identity':
  extra=tmp_path/'extra';extra.mkdir();(extra/ci.MANIFEST).write_bytes((c.root('capture')/ci.MANIFEST).read_bytes());paths=[c.root('capture'),extra]
 else:
  p=tmp_path/'native--00000000-0000-0000-0000-000000000001';p.mkdir();paths=[p]
 with pytest.raises(ci.IdentityError):ci.select_roots(paths)

@pytest.mark.parametrize('source_bound',[False,True])
def test_attached_monitor_never_reuses_legacy_and_records_source(tmp_path,source_bound):
 if source_bound:source=ci.allocate_arm(tmp_path/'repo').root('capture')
 else:source=tmp_path/'legacy';source.mkdir()
 old=source/'monitor';old.mkdir();original=old/'events.jsonl';original.write_bytes(b'synthetic old evidence')
 stamp=datetime(2026,10,4,12,13,14,tzinfo=timezone.utc)
 a=ci.allocate_analysis(source,now=stamp);b=ci.allocate_analysis(source,now=stamp)
 assert a.capture_id!=b.capture_id
 assert a.root('analysis').parent==source/'monitor-runs'
 assert b.root('analysis').name==a.root('analysis').name+'-2'
 assert original.read_bytes()==b'synthetic old evidence'
 assert a.record['source']['status']==('bound' if source_bound else 'legacy-unbound')
 assert bool(a.record['source']['capture_id'])==source_bound
 assert bool(a.record['source']['registry_sha256'])==source_bound
 assert a.kind=='monitor-analysis'

@pytest.mark.parametrize('fault',['source_symlink','parent_symlink','incomplete_source','exhausted'])
def test_attached_monitor_allocation_refuses_ambiguous_destinations(tmp_path,fault):
 source=tmp_path/'legacy';source.mkdir()
 stamp=datetime(2026,10,4,12,13,14,tzinfo=timezone.utc)
 if fault=='source_symlink':
  link=tmp_path/'linked';link.symlink_to(source,target_is_directory=True);source=link
 elif fault=='parent_symlink':
  foreign=tmp_path/'foreign';foreign.mkdir();(source/'monitor-runs').symlink_to(foreign,target_is_directory=True)
 elif fault=='incomplete_source':
  source=tmp_path/'native--00000000-0000-0000-0000-000000000001';source.mkdir()
 else:
  parent=source/'monitor-runs';parent.mkdir()
  base='monitor-20261004T121314Z'
  for n in range(1,100):(parent/(base if n==1 else base+'-'+str(n))).mkdir()
 with pytest.raises((ci.IdentityError,FileExistsError)):ci.allocate_analysis(source,now=stamp)

def test_missing_analysis_manifest_is_incomplete(tmp_path):
 p=tmp_path/'monitor-runs'/'monitor-20261004T121314Z';p.mkdir(parents=True)
 assert ci.provenance(p)['status']=='incomplete'

def test_single_legacy_reader_remains_unbound(tmp_path):
 p=tmp_path/'legacy';p.mkdir()
 assert ci.select_roots([p,p])==p
 assert ci.provenance(p)['status']=='legacy-unbound'

@pytest.mark.parametrize('duplicate',[False,True])
def test_arm_reader_uses_explicit_path_or_refuses_duplicate_legacy(tmp_path,duplicate,tools):
 from consolelib import armsource
 a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
 first=a/'legacy';first.mkdir()
 if duplicate:
  (b/'legacy').mkdir()
  with pytest.raises(armsource.AudioError):armsource.find('legacy',roots=[a,b])
 else:assert armsource.find('legacy',roots=[a,b])==first
 assert armsource.find(str(first),roots=[a,b])==first

@pytest.mark.parametrize('source_error',[False,True])
def test_monitor_claims_new_child_before_opening_attached_source(tmp_path,monkeypatch,source_error,tools):
 from consolelib import monitor
 source_root=tmp_path/'legacy';source_root.mkdir();old=source_root/'monitor';old.mkdir();(old/'events.jsonl').write_bytes(b'synthetic old')
 calls=[]
 class Source:
  directory=source_root
  window=SimpleNamespace(label='synthetic')
  def started(self):return None
  def __enter__(self):return self
  def __exit__(self,*args):pass
 def open_source(args):
  children=list((source_root/'monitor-runs').iterdir());assert len(children)==1
  assert (children[0]/'.capture-writers'/'analysis.json').exists()
  calls.append(children[0])
  if source_error:raise RuntimeError('synthetic open failure')
  return Source(),0,None,{'kind':'arm'}
 from contextlib import contextmanager
 @contextmanager
 def fake_stopping():
  yield SimpleNamespace(is_set=lambda: False), []
 monkeypatch.setattr(monitor, "stopping", fake_stopping)
 monkeypatch.setattr(monitor,'validate',lambda args:None)
 monkeypatch.setattr(monitor,'open_source',open_source)
 monkeypatch.setattr(monitor,'window_record',lambda window:{'kind':'synthetic'})
 monkeypatch.setattr(monitor,'Ours',lambda *args:SimpleNamespace(relabel=lambda d:d))
 monkeypatch.setattr(monitor,'Channel',lambda *args,**kwargs:None)
 monkeypatch.setattr(monitor,'Tapped',lambda *args:SimpleNamespace(heard_s=0))
 monkeypatch.setattr(monitor,'default_specs',lambda **_:[])
 monkeypatch.setattr(monitor.aggregate,'run',lambda *args,**kwargs:None)
 monkeypatch.setattr(monitor.arm,'git_head',lambda:'synthetic')
 args=SimpleNamespace(arm=str(source_root),wav=False,kiwi=False,dispatch='whole-band',cap=1,dial=None,no_record=True,console_argv=[],minutes=0,list_modes=False,no_rtty_usos=False)
 if source_error:
  with pytest.raises(RuntimeError):monitor.report(args)
 else:assert monitor.report(args)==0
 assert len(calls)==1
 assert (old/'events.jsonl').read_bytes()==b'synthetic old'
 cap=ci.load(calls[0]);assert cap.ended()
 if not source_error:
  import json
  facts=json.loads((calls[0]/'provenance.json').read_text())
  assert facts['capture_id']==cap.capture_id
  assert facts['capture_state']=='finalized'
  assert facts['metadata_version']==2
  assert (calls[0]/'events.jsonl').is_file()

@pytest.mark.parametrize('signal',['native_name','reserved_namespace','identity_sidecar'])
def test_artifact_cannot_fall_back_to_legacy_after_registry_loss(tmp_path,signal):
 root=tmp_path/('native--00000000-0000-0000-0000-000000000001' if signal=='native_name' else 'explicit')
 root.mkdir();path=root/'tap.wav';path.write_bytes(b'synthetic')
 if signal=='reserved_namespace':(root/'.capture-writers').mkdir()
 if signal=='identity_sidecar':path.with_suffix('.json').write_text('{"capture_id":"00000000-0000-0000-0000-000000000001"}')
 with pytest.raises(ci.IdentityError):ci.validate_artifact(path)


def test_existing_default_monitor_keeps_99_exclusive_attempts(tmp_path,monkeypatch,tools):
 from consolelib import monitor
 base=tmp_path/'monitor-default'
 monkeypatch.setattr(monitor,'run_directory',lambda args,source:base)
 args=SimpleNamespace(arm=None)
 assert monitor.make_run_directory(args,None)==base
 assert monitor.make_run_directory(args,None)==base.with_name(base.name+'-2')
 for n in range(3,100):base.with_name(base.name+'-'+str(n)).mkdir()
 with pytest.raises(FileExistsError):monitor.make_run_directory(args,None)


def test_general_artifact_binding_keeps_writer_clock_and_mirror_identity(tmp_path):
    capture = ci.allocate_arm(tmp_path, native_root=tmp_path / 'native' / 'arm')
    rows = []
    for role, filename in [('launcher-transcript', 'launcher.log'),
                           ('native-transcript', 'arm.log')]:
        writer = capture.claim(role)
        artifact = writer.reserve(filename, clock_id=role + ':host-clock')
        with artifact.open() as stream:
            stream.write(b'synthetic log\n')
        rows.append(ci.artifact_binding(artifact.path))
    assert all(row['status'] == 'bound' for row in rows)
    assert rows[0]['capture_id'] == rows[1]['capture_id'] == capture.capture_id
    assert rows[0]['registry_sha256'] == rows[1]['registry_sha256']
    assert rows[0]['artifact_id'] != rows[1]['artifact_id']
    assert rows[0]['clock_id'] != rows[1]['clock_id']


@pytest.mark.parametrize('fault', ['unreserved', 'symlink', 'wrong_registry'])
def test_general_artifact_binding_refuses_invalid_new_artifact(tmp_path, fault):
    import json
    capture = ci.allocate_default(tmp_path)
    artifact = capture.claim('modem').reserve('session.log', clock_id='host')
    with artifact.open() as stream:
        stream.write(b'synthetic log\n')
    if fault == 'unreserved':
        path = artifact.path.with_name('foreign.log')
        path.write_bytes(b'synthetic log\n')
    elif fault == 'symlink':
        artifact.path.unlink()
        target = tmp_path / 'foreign.log'
        target.write_bytes(b'synthetic log\n')
        artifact.path.symlink_to(target)
        path = artifact.path
    else:
        path = artifact.path
        for record in (capture.root('capture') / '.capture-artifacts').glob('*.json'):
            row = json.loads(record.read_text())
            if row['artifact_id'] == artifact.record['artifact_id']:
                row['registry_sha256'] = '0' * 64
                record.write_text(json.dumps(row))
    with pytest.raises((ci.IdentityError, FileNotFoundError)):
        ci.artifact_binding(path)


def test_wav_reader_checks_new_sidecar_before_decode_and_keeps_legacy(tmp_path, monkeypatch):
    import json
    import numpy as np
    from hfmodem.core import wav
    from .test_capture_identity_wav import recording
    _, owner = recording(tmp_path / 'bound')
    wav.write(owner.audio.path, np.array([.25, -.25]), 48000, ownership=owner)
    assert np.array_equal(wav.read(owner.audio.path), np.array([.25, -.25]))
    row = json.loads(owner.sidecar.artifact.path.read_text())
    row['clock_id'] = 'foreign-clock'
    owner.sidecar.artifact.path.write_text(json.dumps(row))
    with pytest.raises(ci.IdentityError):
        wav.source(owner.audio.path)
    legacy = tmp_path / 'legacy.wav'
    wav.write(legacy, np.array([.25, -.25]), 48000)
    assert np.array_equal(wav.read(legacy), np.array([.25, -.25]))
    assert ci.validate_artifact(legacy) == {'status': 'legacy-unbound'}


def test_wav_reader_validates_original_and_container_hashes_without_native(tmp_path, monkeypatch):
    import numpy as np
    from hfmodem.core import wav
    from .test_capture_identity_wav import recording, fake_flac
    _, owner = recording(tmp_path)
    side = wav.write(owner.audio.path, np.array([.25, -.25]), 48000, ownership=owner)
    fake_flac(monkeypatch)
    result = wav.store(owner.audio.path, ownership=owner)
    owner.sidecar.update({k: v for k, v in side.items()
                          if k not in owner.audio.record and k not in
                          ['metadata_version', 'previous_sha256']} | result)
    assert not owner.audio.path.exists()
    assert np.array_equal(wav.read(owner.audio.path), np.array([.25, -.25]))
    owner.container.path.write_bytes(b'foreign')
    with pytest.raises(ci.IdentityError):
        wav.source(owner.audio.path)


def test_audio_reader_refuses_missing_registry_before_decode(tmp_path):
    import numpy as np
    from hfmodem.core import wav
    capture = ci.allocate_default(tmp_path)
    owner = ci.Recording.reserve(capture.claim('modem'), 'tap.wav', clock_id='tap')
    wav.write(owner.audio.path, np.zeros(8), 48000, ownership=owner)
    (capture.root('capture') / ci.MANIFEST).unlink()
    with pytest.raises(ci.IdentityError):
        wav.source(owner.audio.path)


def _owned_transcript(capture, role, filename, *, footer=''):
    from hfmodem.tests.core.capture_transcript_support import HEAD, cycle
    artifact = capture.claim(role).reserve(filename, clock_id=role + ':host')
    text = HEAD.format(outdir=str(capture.root('capture'))) + cycle(2, 93.) + footer
    with artifact.open() as stream:
        stream.write(text.encode())
    return artifact.path


def test_transcript_reader_never_collapses_distinct_legacy_minute_arms(tmp_path,tools):
    from consolelib import fingerprints
    from hfmodem.tests.core.capture_transcript_support import arm, cycle
    a = arm(tmp_path, 'onair-old/a.log', cycle(2, 93.), outdir='/captures/onair-1043')
    b = arm(tmp_path, 'onair-old/b.log', cycle(3, 93.), outdir='/captures/onair-1043')
    before = {path: path.read_bytes() for path in (a, b)}
    assert len(fingerprints.transcripts(tmp_path)) == 2
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_transcript_reader_prefers_declared_native_mirror(tmp_path,tools):
    from consolelib import fingerprints
    capture = ci.allocate_arm(tmp_path, native_root=tmp_path / 'working' / 'arms' / 'call')
    _owned_transcript(capture, 'launcher-transcript', capture.record['transcript']['path'])
    native = _owned_transcript(capture, 'native-transcript', 'arm.log', footer='\n[model] native footer\n')
    selected = fingerprints.transcripts(tmp_path)
    assert len(selected) == 1
    assert selected[0][0] == native


def test_same_packet_observations_cannot_merge_distinct_new_arms(tmp_path,tools):
    from consolelib import fingerprints
    # Even identical packet observations are separate captures, not content aliases.
    a = ci.allocate_arm(tmp_path, transcripts=tmp_path / 'working')
    b = ci.allocate_arm(tmp_path, transcripts=tmp_path / 'working')
    _owned_transcript(a, 'launcher-transcript', a.record['transcript']['path'])
    _owned_transcript(b, 'launcher-transcript', b.record['transcript']['path'])
    assert a.capture_id != b.capture_id
    assert len(fingerprints.transcripts(tmp_path)) == 2


@pytest.mark.parametrize('fault', ['missing_registry', 'extra_root', 'undeclared_transcript'])
def test_transcript_reader_refuses_unreadable_new_association(tmp_path, fault, tools):
    import shutil
    from consolelib import fingerprints
    capture = ci.allocate_arm(tmp_path, transcripts=tmp_path / 'working')
    name = 'extra.log' if fault == 'undeclared_transcript' else capture.record['transcript']['path']
    _owned_transcript(capture, 'launcher-transcript', name)
    if fault == 'missing_registry':
        (capture.root('transcript') / ci.MANIFEST).unlink()
    elif fault == 'extra_root':
        shutil.copytree(capture.root('transcript'), tmp_path / 'working' / 'extra')
    with pytest.raises(ci.IdentityError):
        fingerprints.transcripts(tmp_path)


def test_evidence_reference_uses_frozen_time_and_exact_root_not_name(tmp_path):
    capture = ci.allocate_arm(tmp_path, now=datetime(2026, 10, 3, 10, 43, tzinfo=timezone.utc))
    log = _owned_transcript(capture, 'launcher-transcript', capture.record['transcript']['path'])
    row = ci.evidence_reference(log)
    assert row['capture_id'] == capture.capture_id
    assert row['allocated_utc'] == '2026-10-03T10:43:00+00:00'
    assert row['transcript'] == str(log)
    assert row['primary_root'] == str(capture.root('capture'))
    assert ci.evidence_reference(capture.root('capture'))['capture_id'] == capture.capture_id
    assert ci.evidence_reference(tmp_path / 'onair-1003-1043') == {'status': 'legacy-unbound'}


def test_evidence_reference_refuses_missing_new_identity(tmp_path):
    capture = ci.allocate_default(tmp_path)
    root = capture.root('capture')
    (root / ci.MANIFEST).unlink()
    with pytest.raises(ci.IdentityError):
        ci.evidence_reference(root)


def test_metadata_record_checks_identity_without_rewriting_legacy(tmp_path):
    import json
    capture = ci.allocate_default(tmp_path)
    artifact = capture.claim('modem').reserve('record.json', clock_id='host')
    slot = ci.MetadataSlot(artifact)
    slot.update({'state': 'synthetic'})
    assert ci.metadata_record(artifact.path)['capture_id'] == capture.capture_id
    row = json.loads(artifact.path.read_text())
    row['writer_id'] = str(uuid.uuid4())
    artifact.path.write_text(json.dumps(row))
    with pytest.raises(ci.IdentityError):
        ci.metadata_record(artifact.path)
    legacy = tmp_path / 'legacy.json'
    legacy.write_text(json.dumps({'state': 'synthetic'}))
    before = legacy.read_bytes()
    assert ci.metadata_record(legacy) == {'state': 'synthetic'}
    assert legacy.read_bytes() == before
    legacy.write_text(json.dumps({'capture_id': capture.capture_id}))
    with pytest.raises(ci.IdentityError):
        ci.metadata_record(legacy)


def _owned_windows(tmp_path, *, clocks=('local-stream', 'local-stream')):
    import numpy as np
    from hfmodem.core import wav
    capture = ci.allocate_default(tmp_path)
    writer = capture.claim('modem')
    windows = []
    for i, clock in enumerate(clocks):
        owner = ci.Recording.reserve(writer, f'window-{i:04}.wav', clock_id=clock)
        side = wav.write(owner.audio.path, np.zeros(8), 48000, ownership=owner)
        owner.sidecar.update({k: v for k, v in side.items()
                              if k not in owner.audio.record and k not in
                              ['metadata_version', 'previous_sha256']} |
                             {'end_stream_sample': 80 + 40 * i})
        windows.append(owner.audio.path)
    return capture, windows


def test_new_window_selection_is_independent_of_sidecar_mtime(tmp_path):
    import os
    capture, windows = _owned_windows(tmp_path)
    expected = list(windows)
    assert ci.recording_windows(windows[::-1], capture_id=capture.capture_id) == expected
    for i, path in enumerate(windows):
        os.utime(path.with_suffix('.json'), (i + 1, i + 1))
    assert ci.recording_windows(windows[::-1], capture_id=capture.capture_id) == expected


@pytest.mark.parametrize('fault', ['foreign_capture', 'foreign_clock', 'mixed_clocks', 'missing_samples'])
def test_new_window_selection_refuses_unowned_or_unplaced_input(tmp_path, fault):
    import json
    capture, windows = _owned_windows(tmp_path, clocks=('local-stream', 'foreign')
                                      if fault == 'mixed_clocks' else ('local-stream', 'local-stream'))
    identity = str(uuid.uuid4()) if fault == 'foreign_capture' else capture.capture_id
    clock = 'foreign' if fault == 'foreign_clock' else None
    if fault == 'missing_samples':
        side = windows[0].with_suffix('.json')
        row = json.loads(side.read_text())
        row.pop('end_stream_sample')
        side.write_text(json.dumps(row))
    with pytest.raises(ci.IdentityError):
        ci.recording_windows(windows, capture_id=identity, clock_id=clock)


def test_evidence_reference_end_is_arm_end_not_capture_completeness(tmp_path):
    capture = ci.allocate_default(tmp_path)
    assert 'arm_ended_utc' not in ci.evidence_reference(capture.root('capture'))
    capture.end()
    row = ci.evidence_reference(capture.root('capture'))
    assert datetime.fromisoformat(row['arm_ended_utc']).tzinfo is not None
    assert 'capture_complete' not in row


def test_reader_adapters_call_identity_helpers_without_importing_logbook_view(tools):
    import ast
    root = Path(__file__).resolve().parents[5]
    original_path = list(sys.path)
    # Source-only adapter pins; these files are never imported or on sys.path.
    checks = {
        'tools/logbook_view/legacy.py': {'evidence_reference', 'metadata_record'},
        'tools/logbook_view/delivered.py': {'associated_transcripts', 'metadata_record'},
        'tools/logbook_view/audio.py': {'validate_artifact'},
        'tools/logbook_view/rehear.py': {'evidence_reference', 'recording_windows', 'validate_artifact'},
        'tools/consolelib/logbook.py': {'metadata_record'},
    }
    for relative, needed in checks.items():
        tree = ast.parse((root / relative).read_text())
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert needed <= called
    assert sys.path == original_path


def test_delivery_reference_refuses_legacy_basename_last_match(tmp_path):
    paths = [tmp_path / 'a.log', tmp_path / 'b.log']
    for path in paths:
        path.write_bytes(b'synthetic transcript\n')
    with pytest.raises(ci.IdentityError):
        ci.associated_transcripts(tmp_path, tmp_path / 'legacy' / 'provenance.json', {}, paths)
    assert ci.associated_transcripts(tmp_path, tmp_path / 'legacy' / 'provenance.json', {}, paths[:1]) == paths[:1]


def test_delivery_reference_uses_registered_mirrors_and_ignores_basename(tmp_path):
    capture = ci.allocate_arm(tmp_path, native_root=tmp_path / 'working' / 'arms' / 'same')
    launcher = _owned_transcript(capture, 'launcher-transcript', capture.record['transcript']['path'])
    native = _owned_transcript(capture, 'native-transcript', 'arm.log')
    artifact = capture.claim('native-metadata').reserve('provenance.json', clock_id='host')
    row = ci.MetadataSlot(artifact).update({'arm': 'same'})
    other = tmp_path / 'foreign.log'
    other.write_bytes(b'synthetic foreign transcript\n')
    assert ci.associated_transcripts(tmp_path, artifact.path, row, [other]) == [native, launcher]
    with pytest.raises(ci.IdentityError):
        ci.associated_transcripts(tmp_path, artifact.path, row | {'capture_id': str(uuid.uuid4())})


def test_delivery_reference_derived_copy_cannot_replace_bound_original(tmp_path):
    capture = ci.allocate_arm(tmp_path)
    original = _owned_transcript(capture, 'launcher-transcript', capture.record['transcript']['path'])
    derived = tmp_path / 'derived'
    derived.mkdir()
    copy = derived / 'arm.log'
    copy.write_bytes(b'synthetic derived copy\n')
    record = {'source': {'converted_from': str(original),
                         'capture_reference': ci.evidence_reference(original)}}
    assert ci.associated_transcripts(tmp_path, derived / 'provenance.json', record) == [original]


def test_delivery_reference_never_downgrades_orphan_identity_to_basename(tmp_path):
    candidate = tmp_path / 'same.log'
    candidate.write_bytes(b'synthetic transcript\n')
    with pytest.raises(ci.IdentityError):
        ci.associated_transcripts(tmp_path, tmp_path / 'legacy' / 'provenance.json',
                                  {'capture_id': str(uuid.uuid4())}, [candidate])


def test_default_monitor_allocator_keeps_names_and_adds_distinct_identity(tmp_path, monkeypatch, tools):
    from consolelib import monitor
    monkeypatch.setattr(monitor, 'run_directory', lambda args, source: tmp_path / 'monitor-fixed')
    a = monitor.make_run_directory(SimpleNamespace(arm=None), None)
    b = monitor.make_run_directory(SimpleNamespace(arm=None), None)
    assert a.name == 'monitor-fixed'
    assert b.name == 'monitor-fixed-2'
    first, second = ci.load(a), ci.load(b)
    assert first.kind == second.kind == 'monitor-analysis'
    assert first.capture_id != second.capture_id


def test_owned_monitor_event_names_its_clock_and_sample_coordinate(tmp_path,tools):
    import json
    from consolelib import monitor
    capture = ci.allocate({'analysis': tmp_path / 'analysis'}, {'analysis': 'analysis'},
                          primary_root='analysis', kind='monitor-analysis')
    artifact = capture.claim('analysis').reserve('events.jsonl', clock_id='event-samples')
    events = monitor.Events(artifact.path, ownership=artifact)
    events({'t': .5, 'kind': 'synthetic-candidate'})
    events.close()
    row = json.loads(artifact.path.read_text())
    assert row['t'] == .5 and row['kind'] == 'synthetic-candidate'
    assert row['event_stream_sample'] == 24000
    assert row['capture_id'] == capture.capture_id
    assert row['artifact_id'] == artifact.record['artifact_id']
    assert row['clock_id'] == 'event-samples'


def test_legacy_event_writer_does_not_invent_identity(tmp_path,tools):
    import json
    from consolelib import monitor
    path = tmp_path / 'legacy-events.jsonl'
    events = monitor.Events(path)
    events({'t': .5, 'kind': 'synthetic-candidate'})
    events.close()
    assert json.loads(path.read_text()) == {'t': .5, 'kind': 'synthetic-candidate'}


@pytest.mark.parametrize('source_error', [False, True])
def test_default_monitor_claims_identity_before_source_open(tmp_path, monkeypatch, source_error, tools):
    from contextlib import contextmanager
    from consolelib import monitor
    calls = []
    class Source:
        window = SimpleNamespace(label='synthetic')
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def opening(args):
        root = tmp_path / 'monitor-fixed'
        capture = ci.load(root)
        assert (root / '.capture-writers' / 'analysis.json').is_file()
        calls.append(capture)
        if source_error:
            raise RuntimeError('synthetic source refusal')
        return Source(), 0, None, {'kind': 'wav'}
    @contextmanager
    def stopped():
        yield SimpleNamespace(is_set=lambda: False), []
    monkeypatch.setattr(monitor, 'validate', lambda args: None)
    monkeypatch.setattr(monitor, 'run_directory', lambda *args: tmp_path / 'monitor-fixed')
    monkeypatch.setattr(monitor, 'open_source', opening)
    monkeypatch.setattr(monitor, 'stopping', stopped)
    monkeypatch.setattr(monitor, 'window_record', lambda *args: {})
    monkeypatch.setattr(monitor, 'Channel', lambda *args, **kwargs: None)
    monkeypatch.setattr(monitor, 'Tapped', lambda *args: SimpleNamespace(heard_s=0))
    monkeypatch.setattr(monitor, 'default_specs', lambda **_: [])
    monkeypatch.setattr(monitor.aggregate, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(monitor.arm, 'git_head', lambda: 'synthetic')
    args = SimpleNamespace(arm=None, wav=True, kiwi=False, dispatch='whole-band', cap=1,
                           dial=None, no_record=True, console_argv=[], minutes=0,
                           list_modes=False, no_rtty_usos=False)
    if source_error:
        with pytest.raises(RuntimeError):
            monitor.report(args)
    else:
        assert monitor.report(args) == 0
    assert len(calls) == 1 and calls[0].ended()
    assert ci.load(tmp_path / 'monitor-fixed').kind == 'monitor-analysis'
    lifecycle=calls[0].lifecycle()
    assert lifecycle['capture_complete'] == (not source_error)
    assert lifecycle['writers']['analysis']['state']==('partial' if source_error else 'finalized')


def _monitor_recorder(tmp_path, *, segment_s=4/48000):
    from consolelib import monitor
    capture = ci.allocate({'analysis': tmp_path/'analysis'}, {'analysis':'analysis'}, primary_root='analysis', kind='monitor-analysis')
    writer = capture.claim('analysis')
    clock = capture.capture_id + ':monitor-samples'
    return capture, monitor.Recorder(writer.root, segment_s=segment_s, writer=writer, clock_id=clock), clock


def test_owned_monitor_segments_preserve_pcm_and_event_clock(tmp_path,tools):
    import numpy as np, wave
    capture, recorder, clock = _monitor_recorder(tmp_path)
    pcm = np.arange(-5, 5, dtype='<i2').tobytes()
    recorder(700, pcm[:6])
    recorder(703, pcm[6:])
    recorder.close()
    content = []
    for path, bounds in zip(recorder.paths, [(700,704),(704,708),(708,710)]):
        side = json.loads(path.with_suffix('.json').read_text())
        assert side['capture_id'] == capture.capture_id and side['clock_id'] == clock
        assert (side['start_stream_sample'],side['end_stream_sample']) == bounds
        assert side['samples'] == bounds[1]-bounds[0] and side['capture_state']=='finalized'
        assert ci.validate_artifact(path, clock_id=clock)['status']=='bound'
        with wave.open(str(path)) as audio:
            assert audio.getframerate()==48000 and audio.getsampwidth()==2 and audio.getnchannels()==1
            content.append(audio.readframes(audio.getnframes()))
    assert b''.join(content)==pcm
    assert len({s['artifact_id'] for s in recorder.segments})==3
    assert all(s['clock_id']==clock for s in recorder.segments)


def test_owned_monitor_gap_keeps_source_sample_coordinates(tmp_path,tools):
    capture, recorder, clock = _monitor_recorder(tmp_path,segment_s=1)
    recorder(0,b'\x00\x00'*3)
    recorder(11,b'\x00\x00'*2)
    recorder.close()
    assert [(s['start_stream_sample'],s['end_stream_sample']) for s in recorder.segments]==[(0,3),(11,13)]
    assert ci.recording_windows(recorder.paths,capture_id=capture.capture_id,clock_id=clock)==recorder.paths


@pytest.mark.parametrize('bad', ['clock','partial-sample','foreign'])
def test_owned_monitor_cannot_publish_foreign_or_unplaced_segment(tmp_path,bad,tools):
    capture, recorder, clock = _monitor_recorder(tmp_path)
    root=capture.root('analysis')
    if bad=='foreign':
        (root/'rx-000.wav').write_bytes(b'old')
        with pytest.raises(FileExistsError):recorder(0,b'\x00\x00')
        assert (root/'rx-000.wav').read_bytes()==b'old'
    elif bad=='clock':
        with pytest.raises(ValueError):recorder(.5,b'\x00\x00')
        assert not recorder.paths
    else:
        with pytest.raises(ValueError):recorder(0,b'\x00')
        assert not recorder.paths


def test_owned_monitor_partial_close_is_not_complete(tmp_path,tools):
    capture, recorder, clock=_monitor_recorder(tmp_path)
    recorder(0,b'\x00\x00'*2)
    recorder.close(state='partial')
    path=recorder.paths[0]
    assert json.loads(path.with_suffix('.json').read_text())['capture_state']=='partial'
    assert recorder.segments[0]['capture_state']=='partial'
    assert ci.validate_artifact(path,clock_id=clock)['status']=='bound'
    recorder.close()
    assert json.loads(path.with_suffix('.json').read_text())['capture_state']=='partial'


@pytest.mark.parametrize('failure',['source','enter','aggregate'])
def test_monitor_failure_remains_partial_and_ends_identity(tmp_path,monkeypatch,failure,tools):
    from contextlib import contextmanager
    from consolelib import monitor
    class Source:
        window=SimpleNamespace(label='synthetic')
        def __enter__(self):
            if failure=='enter':raise RuntimeError('synthetic enter refusal')
            return self
        def __exit__(self,*args):pass
    def opening(args):
        if failure=='source':raise RuntimeError('synthetic source refusal')
        return Source(),0,None,{'kind':'wav'}
    @contextmanager
    def stopped():yield SimpleNamespace(is_set=lambda:False),[]
    def aggregate(*args,**kwargs):raise RuntimeError('synthetic aggregation refusal')
    monkeypatch.setattr(monitor,'validate',lambda args:None)
    monkeypatch.setattr(monitor,'run_directory',lambda *args:tmp_path/'monitor-fixed')
    monkeypatch.setattr(monitor,'open_source',opening)
    monkeypatch.setattr(monitor,'stopping',stopped)
    monkeypatch.setattr(monitor,'window_record',lambda *args:{})
    monkeypatch.setattr(monitor,'Channel',lambda *args,**kwargs:None)
    monkeypatch.setattr(monitor,'Tapped',lambda *args:SimpleNamespace(heard_s=0))
    monkeypatch.setattr(monitor,'default_specs',lambda **_:[])
    monkeypatch.setattr(monitor.aggregate,'run',aggregate)
    monkeypatch.setattr(monitor.arm,'git_head',lambda:'synthetic')
    args=SimpleNamespace(arm=None,wav=True,kiwi=False,dispatch='whole-band',cap=1,dial=None,no_record=True,console_argv=[],minutes=0,list_modes=False,no_rtty_usos=False)
    with pytest.raises(RuntimeError):monitor.report(args)
    root=tmp_path/'monitor-fixed'
    assert ci.load(root).ended()
    assert ci.metadata_record(root/'provenance.json')['capture_state']=='partial'



@pytest.fixture
def p4_ledger(tools):
    from hfmodem.tests.evidence import require_input

    require_input(tools / "p4_ledger.py")
    import p4_ledger
    return p4_ledger


def test_bound_ledger_uses_declared_transcript_and_probe_not_text_outdir(tmp_path, monkeypatch, p4_ledger):
    import json
    from hfmodem.shrike import onair
    cap = ci.allocate_arm(tmp_path, native_root=tmp_path / "native", arguments=[])
    transcript = cap.claim("launcher-transcript")
    text = transcript.reserve(cap.record["transcript"]["path"])
    text.publish(b"synthetic transcript\n")
    writer = cap.claim("modem")
    monkeypatch.setattr(onair, "_CAPTURE_ROOTS", {writer.root: writer})
    # Original report file name remains the established consumer contract.
    import ast
    module = ast.parse(Path(p4_ledger.__file__).read_text())
    probe = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_probe")
    assignment = next(n for n in probe.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "here" for t in n.targets))
    path = writer.root / assignment.value.right.value
    onair._capture_write_text(path, json.dumps({"queries": []}))
    assert p4_ledger._log(cap.root("capture")) == text.path
    actual, result = p4_ledger._probe(cap.root("capture"), "irrelevant legacy outdir text")
    assert actual == path and result == {"queries": []}


def test_bound_ledger_cannot_accept_unclaimed_named_log(tmp_path,p4_ledger):
    cap = ci.allocate_arm(tmp_path, arguments=[])
    (cap.root("capture") / "modem.log").write_text("synthetic foreign log")
    with pytest.raises((ci.IdentityError, FileNotFoundError)):
        p4_ledger._log(cap.root("capture"))


def test_legacy_ledger_ambiguous_transcripts_are_not_silently_selected(tmp_path,p4_ledger):
    (tmp_path / "first.log").write_text("first")
    (tmp_path / "second.log").write_text("second")
    with pytest.raises(ValueError):
        p4_ledger._log(tmp_path)


def test_owned_witness_side_is_validated_before_rate_fallback(tmp_path,tools):
    import witness_align
    from hfmodem.core import wav
    import numpy as np
    cap = ci.allocate({"source": tmp_path / "owned"}, {"witness": "source"})
    writer = cap.claim("witness")
    recording = ci.Recording.reserve(writer, "witness.wav", clock_id=cap.capture_id + ":audio")
    wav.write(recording.audio.path, np.zeros(16), 48000, ownership=recording, sample_rate=47999.5)
    assert witness_align.witness_rate(recording.audio.path, 48000) == 47999.5
    side = recording.audio.path.with_suffix(".json")
    import json
    row = json.loads(side.read_text());row["capture_id"] = str(__import__('uuid').uuid4());side.write_text(json.dumps(row))
    with pytest.raises(ci.IdentityError):
        witness_align.witness_rate(recording.audio.path, 48000)


def test_legacy_witness_missing_side_keeps_header_rate(tmp_path,tools):
    import witness_align
    assert witness_align.witness_rate(tmp_path / "old.wav", 48000) == 48000



def test_corpus_manifest_discovery_keeps_renamed_identity_and_segment_refs(tmp_path,tools):
    from corpuslib import inventory
    from hfmodem.core import wav
    import numpy as np
    cap = ci.allocate({"analysis": tmp_path / "renamed-owned-visit"}, {"analysis": "analysis"}, kind="monitor-analysis")
    writer = cap.claim("analysis")
    import json
    metadata = writer.reserve("session.json")
    metadata.publish(json.dumps(metadata.metadata(dict(capture=True, complete=True, analyzed_seconds=1))).encode())
    recording = ci.Recording.reserve(writer, "rx-000.wav", clock_id=cap.capture_id + ":audio")
    wav.write(recording.audio.path, np.zeros(48), 48000, ownership=recording)
    result = inventory.build(tmp_path)
    assert result["schema"] == 2 and len(result["visits"]) == 1
    visit = result["visits"][0]
    assert visit["capture_identity"]["capture_id"] == cap.capture_id
    assert visit["recording_identities"][0]["artifact_id"] == recording.audio.record["artifact_id"]
    assert result["totals"]["wav_files"] == 1


def test_corpus_flac_conversion_keeps_one_segment_identity(tmp_path, monkeypatch, tools):
    from corpuslib import inventory
    from hfmodem.core import wav
    import numpy as np
    cap = ci.allocate({"analysis": tmp_path / "visit-owned"}, {"analysis": "analysis"}, kind="monitor-analysis")
    writer = cap.claim("analysis")
    import json
    metadata = writer.reserve("session.json")
    metadata.publish(json.dumps(metadata.metadata(dict(capture=True, complete=True, analyzed_seconds=1))).encode())
    recording = ci.Recording.reserve(writer, "rx-000.wav", clock_id=cap.capture_id + ":audio")
    wav.write(recording.audio.path, np.zeros(48), 48000, ownership=recording)
    # Fake the existing codec interface; its FLAC source contract returns BytesIO.
    import hashlib
    original_wav = recording.audio.path.read_bytes()
    temporary, stream = recording.container.temporary()
    stream.write(b"synthetic codec container");stream.close()
    recording.container.publish_file(temporary)
    facts = json.loads(recording.audio.path.with_suffix(".json").read_text())
    recording.sidecar.update(facts | dict(stored=recording.container.path.name,
        container_artifact_id=recording.container.record["artifact_id"],
        container_sha256=hashlib.sha256(recording.container.path.read_bytes()).hexdigest()))
    recording.audio.path.unlink()
    def fake_codec(*arguments):
        Path(arguments[-2]).write_bytes(original_wav)
    monkeypatch.setattr(wav, "_flac", fake_codec)
    import witness_align
    assert witness_align.witness_rate(recording.audio.path, 48000) == 48000
    result = inventory.build(tmp_path)
    assert result["totals"]["wav_files"] == 1
    assert result["totals"]["audio_seconds"] == .001
    assert result["visits"][0]["recording_identities"][0]["capture_id"] == cap.capture_id


def test_corpus_legacy_only_preserves_original_schema_and_fields(tmp_path,tools):
    from corpuslib import inventory
    import json
    visit = tmp_path / "000941Z-7000000";visit.mkdir()
    (visit / "session.json").write_text(json.dumps(dict(capture=True, complete=True)))
    result = inventory.build(tmp_path)
    assert result["schema"] == 1 and len(result["visits"]) == 1
    assert "capture_identity" not in result["visits"][0]
    assert "recording_identities" not in result["visits"][0]
