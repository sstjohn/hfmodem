# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Deferred capture work remains bound to its original root and sample clock."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import capture_identity as ci


def owned(tmp_path, name):
    capture = ci.allocate({'capture': tmp_path / name}, {'modem': 'capture'})
    return capture, capture.claim('modem')


def test_queued_binding_does_no_filesystem_work_on_caller(tmp_path, monkeypatch):
    capture, writer = owned(tmp_path, 'old')
    def forbidden(*args, **kwargs):
        raise AssertionError('filesystem I/O on queue caller')
    with monkeypatch.context() as patch:
        for method in ['open', 'mkdir', 'lstat', 'read_bytes', 'write_bytes', 'exists']:
            patch.setattr(Path, method, forbidden)
        job = ci.Recording.queued(writer, 'rx_01.wav', clock_id='original-stream')
    capture.end()
    newer, _ = owned(tmp_path, 'newer')
    recording = job.reserve()
    recording.audio.publish(b'old recording')
    assert recording.audio.path.parent == writer.root
    assert recording.audio.record['clock_id'] == 'original-stream'
    assert not (newer.root('capture') / 'rx_01.wav').exists()
    with pytest.raises(ci.IdentityError):
        job.reserve()
    with pytest.raises(ci.IdentityError):
        ci.Recording.queued(writer, 'after_end.wav', clock_id='original-stream')


def test_external_end_only_admits_preexisting_ticket(tmp_path):
    capture, writer = owned(tmp_path, 'old')
    before = writer.ticket('before.json')
    ci.load(writer.root).end()
    # A different process's end is checked on the worker; the scheduling path
    # remains free of filesystem I/O and cannot grant after-end publication.
    after = writer.ticket('after.json')
    before.reserve().publish(b'old')
    with pytest.raises(ci.IdentityError):
        after.reserve()
    assert not (writer.root / 'after.json').exists()


def test_changed_end_clock_domain_is_conservative_without_timestamp_guess(tmp_path):
    capture, writer = owned(tmp_path, 'old')
    ticket = writer.ticket('before.json')
    capture.end()
    marker = writer.root / '.capture-ended.json'
    record = json.loads(marker.read_text())
    record['clock_host'] = 'different-clock-domain'
    marker.write_text(json.dumps(record))
    with pytest.raises(ci.IdentityError):
        ticket.reserve()


def test_actual_async_helper_passes_fixed_job_and_old_window_bytes(tmp_path, monkeypatch):
    from hfmodem.shrike import onair, session
    capture, writer = owned(tmp_path, 'old')
    _, newer = owned(tmp_path, 'newer')
    queue = []
    monkeypatch.setattr(onair, '_CAPTURE_ROOTS', {writer.root: writer, newer.root: newer})
    monkeypatch.setattr(onair, '_CAPTURE_RECORDINGS', {})
    monkeypatch.setattr(onair, '_WRITER', SimpleNamespace(put=queue.append))
    path = writer.root / 'rx_01.wav'
    samples = np.array([0.25, -0.4, 0.8], dtype=np.float32)
    original = tmp_path / 'legacy.wav'
    session.write_wav(str(original), samples)
    def forbidden(*args, **kwargs):
        raise AssertionError('filesystem I/O on actual async capture caller')
    with monkeypatch.context() as patch:
        for method in ['open', 'mkdir', 'lstat', 'read_bytes', 'write_bytes', 'exists']:
            patch.setattr(Path, method, forbidden)
        onair._save_capture_async(path, samples, xruns=2, end=9876, lost=3)
    assert len(queue) == 1 and len(queue[0]) == 6
    assert isinstance(queue[0][-1], ci.QueuedRecording)
    capture.end()
    file, *arguments = queue.pop()
    recording = onair._save_capture(file, *arguments)
    assert file.read_bytes() == original.read_bytes()
    side = json.loads(file.with_suffix('.json').read_text())
    assert (side['end_stream_sample'], side['lost_samples'], side['xruns']) == (9876, 3, 2)
    assert side['normalised_on_write'] is True and side['grid_loss_seen'] is True
    assert side['capture_id'] == capture.capture_id
    assert side['clock_id'] == capture.capture_id + ':local-stream'
    assert recording.audio.path == file
    assert not (newer.root / file.name).exists()


def test_direct_output_defaults_offline_and_explicit_reuse_refusal(tmp_path):
    capture, writer = ci.prepare_output(offline=True, default_parent=tmp_path)
    assert capture.kind == 'offline-replay'
    assert writer.root.name.endswith(capture.capture_id)
    with pytest.raises(FileExistsError):
        ci.prepare_output(writer.root, offline=True)


@pytest.mark.parametrize('wrong', ['id', 'role', 'outdir', 'missing-root'])
def test_direct_modem_delegation_has_no_id_only_bypass(tmp_path, wrong):
    capture = ci.allocate_arm(tmp_path)
    keywords = dict(capture_id=capture.capture_id, root=capture.root('capture'), role='modem')
    outdir = capture.root('capture')
    if wrong == 'id':
        keywords['capture_id'] = '00000000-0000-0000-0000-000000000000'
    elif wrong == 'role':
        keywords['role'] = 'launcher-witness'
    elif wrong == 'outdir':
        outdir = tmp_path / 'unregistered'
    else:
        keywords['root'] = None
    with pytest.raises(ci.IdentityError):
        ci.prepare_output(outdir, **keywords)
    assert not (capture.root('capture') / '.capture-writers/modem.json').exists()



def test_arm_end_does_not_certify_pending_or_partial_writer(tmp_path):
    capture=ci.allocate_default(tmp_path)
    writer=capture.claim('modem')
    writer.update_status(state='ongoing',pending=2,written=1)
    capture.end()
    row=capture.lifecycle()
    assert row['arm_ended'] and not row['capture_complete']
    assert row['writers']['modem']['pending']==2
    writer.update_status(state='partial',pending=0,written=2,lost=1)
    row=capture.lifecycle()
    assert row['writers']['modem']['lost']==1 and not row['claimed_writers_complete']
    with pytest.raises(ci.IdentityError):writer.update_status(state='finalized',lost=1)


def test_original_writer_can_finish_reserved_status_after_arm_end(tmp_path):
    capture=ci.allocate_default(tmp_path)
    writer=capture.claim('modem')
    slot=writer._status['slot'].artifact.path
    original=slot.read_bytes()
    capture.end()
    writer.update_status(state='finalized',written=3)
    assert capture.lifecycle()['claimed_writers_complete']
    assert slot.read_bytes()!=original
    assert ci.metadata_record(slot)['written']==3
    assert len(list((slot.parent/('.'+slot.name+'.versions')).glob('*.json')))==2
    with pytest.raises(ci.IdentityError):writer.reserve('new-after-end.json')


def test_unclaimed_writer_is_explicit_unknown_not_complete(tmp_path):
    capture=ci.allocate({'main':tmp_path/'root'},{'modem':'main','tap':'main'},primary_root='main')
    writer=capture.claim('modem')
    writer.update_status(state='finalized')
    capture.end()
    row=capture.lifecycle()
    assert row['claimed_writers_complete']
    assert row['unclaimed_writers']==['tap'] and not row['capture_complete']


def test_old_writer_status_cannot_finish_new_capture(tmp_path):
    old=ci.allocate_default(tmp_path);new=ci.allocate_default(tmp_path)
    a=old.claim('modem');new.claim('modem')
    old.end();new.end()
    a.update_status(state='finalized',written=2)
    assert old.lifecycle()['writers']['modem']['written']==2
    assert new.lifecycle()['writers']['modem']['state']=='ongoing'
    assert not new.lifecycle()['capture_complete']


@pytest.mark.parametrize('count',[-1,True,1.5])
def test_writer_lifecycle_rejects_invalid_counts(tmp_path,count):
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    with pytest.raises(ci.IdentityError):writer.update_status(state='ongoing',pending=count)



def _deferred_worker():
    import queue,threading
    from hfmodem.shrike import onair
    worker=onair._CaptureWriter.__new__(onair._CaptureWriter)
    worker._q=queue.Queue();worker._idle=threading.Condition()
    worker.queued=worker.written=0
    worker.lost=[];worker.unstored=[];worker._owned={};worker.status_errors={}
    return onair,worker


def _one_worker_item(worker,monkeypatch):
    original=worker._q.get;calls=[]
    def take():
        if calls:raise StopIteration
        calls.append(1);return original()
    with monkeypatch.context() as m:
        m.setattr(worker._q,'get',take)
        with pytest.raises(StopIteration):worker._run()


def _enqueue_window(worker,writer,name):
    import numpy as np
    job=ci.Recording.queued(writer,name,clock_id=writer.capture.capture_id+':local-stream')
    worker.put((job.audio.writer.root/name,np.zeros(8),0,8,0,job))
    return job


def test_deferred_queue_per_capture_counts_do_no_filesystem_work(tmp_path,monkeypatch):
    onair,worker=_deferred_worker()
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    def forbidden(*args,**kwargs):raise AssertionError('queue attempted filesystem work')
    monkeypatch.setattr(ci,'_publish',forbidden)
    _enqueue_window(worker,writer,'window.wav')
    assert worker.snapshot(writer)==dict(pending=1,written=0,lost=0,partial=0)


def test_old_deferred_worker_completion_cannot_finish_new_root(tmp_path,monkeypatch):
    from hfmodem.core import wav
    onair,worker=_deferred_worker()
    old=ci.allocate_default(tmp_path);new=ci.allocate_default(tmp_path)
    a=old.claim('modem');b=new.claim('modem')
    _enqueue_window(worker,a,'window.wav');_enqueue_window(worker,b,'window.wav')
    old.end();new.end()
    def save(path,seg,xruns,end,lost,job):
        owner=job.reserve();wav.write(path,seg,48000,ownership=owner,end_stream_sample=end);return owner
    monkeypatch.setattr(onair,'_save_capture',save)
    stored=[]
    monkeypatch.setattr(onair,'_store_capture',lambda path,owner:stored.append((path,owner.audio.writer.capture.capture_id)))
    _one_worker_item(worker,monkeypatch)
    assert worker.snapshot(a)==dict(pending=1,written=1,lost=0,partial=0)
    assert worker.snapshot(b)==dict(pending=1,written=0,lost=0,partial=0)
    assert not old.lifecycle()['capture_complete']
    worker.store()
    assert stored==[(a.root/'window.wav',old.capture_id)]
    assert old.lifecycle()['capture_complete']
    assert not new.lifecycle()['capture_complete'] and b._status['slot'].version==2


@pytest.mark.parametrize('failure',['write','store'])
def test_deferred_failure_stays_under_original_capture(tmp_path,monkeypatch,failure):
    from hfmodem.core import wav
    onair,worker=_deferred_worker()
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    _enqueue_window(worker,writer,'window.wav');capture.end()
    def save(path,seg,xruns,end,lost,job):
        if failure=='write':raise RuntimeError('synthetic write refusal')
        owner=job.reserve();wav.write(path,seg,48000,ownership=owner);return owner
    def store(*args):raise RuntimeError('synthetic storage refusal')
    monkeypatch.setattr(onair,'_save_capture',save);monkeypatch.setattr(onair,'_store_capture',store)
    _one_worker_item(worker,monkeypatch)
    if failure=='store':
        with pytest.raises(RuntimeError):worker.store()
        assert writer.root.joinpath('window.wav').is_file()
    row=capture.lifecycle()
    assert row['arm_ended'] and not row['capture_complete']
    assert row['writers']['modem']['lost' if failure=='write' else 'partial']==1


def test_native_shutdown_exception_stays_partial_after_late_worker_finish(tmp_path,monkeypatch):
    import ast
    from hfmodem.shrike import onair
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    monkeypatch.setattr(onair,'_CAPTURE_ROOTS',{writer.root:writer})
    monkeypatch.setattr(onair,'_WRITER',None)
    @onair._capture_output_lifetime
    def fails(args):
        args._capture_identity=capture
        raise RuntimeError('synthetic native refusal')
    from types import SimpleNamespace
    with pytest.raises(RuntimeError):fails(SimpleNamespace())
    assert capture.lifecycle()['writers']['modem']['partial']==1
    writer.update_status(state='finalized',written=3)
    assert not capture.lifecycle()['capture_complete']
    module=ast.parse(__import__('inspect').getsource(onair))
    run=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    helper=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_capture_output_lifetime')
    assert [ast.unparse(n) for n in run.decorator_list]==['progress_to_stdout()','_capture_output_lifetime']
    assert not helper.decorator_list



def test_shared_writer_components_cannot_erase_unfinished_work(tmp_path):
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    writer.update_status(state='ongoing',pending=2,component='deferred-windows')
    writer.update_status(state='finalized',written=1,component='tap')
    capture.end()
    row=capture.lifecycle()['writers']['modem']
    assert row['state']=='ongoing' and row['pending']==2 and row['written']==1
    assert not capture.lifecycle()['capture_complete']
    writer.update_status(state='finalized',written=2,component='deferred-windows')
    row=capture.lifecycle()['writers']['modem']
    assert row['state']=='finalized' and row['written']==3
    assert capture.lifecycle()['capture_complete']


def test_partial_component_stays_visible_when_other_component_finishes(tmp_path):
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    writer.update_status(state='partial',lost=1,component='tap')
    writer.update_status(state='finalized',written=2,component='deferred-windows')
    capture.end()
    row=capture.lifecycle()['writers']['modem']
    assert row['state']=='partial' and row['lost']==1 and row['written']==2
    assert not capture.lifecycle()['capture_complete']



def test_ending_journal_cannot_certify_missing_recording(tmp_path):
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    owner=ci.Recording.reserve(writer,'never-written.wav',clock_id='synthetic')
    writer.update_status(state='finalized');capture.end()
    row=capture.lifecycle()
    assert row['claimed_writers_complete'] and not row['artifacts_complete']
    assert not row['capture_complete']
    assert row['artifact_states'][owner.audio.record['artifact_id']]=='missing'


def test_valid_wav_keeps_unused_flac_reservation_explicit(tmp_path):
    from hfmodem.core import wav
    import numpy as np
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    owner=ci.Recording.reserve(writer,'original.wav',clock_id='synthetic')
    wav.write(owner.audio.path,np.zeros(3),48000,ownership=owner)
    writer.update_status(state='finalized',written=1);capture.end()
    row=capture.lifecycle()
    assert row['artifacts_complete'] and row['capture_complete']
    assert row['artifact_states'][owner.container.record['artifact_id']]=='alternative-not-used'
    owner.sidecar.update({'capture_state':'partial'})
    assert not capture.lifecycle()['artifacts_complete']


@pytest.mark.parametrize('wrong', ['registry_sha256', 'clock_id', 'writer_id'])
def test_unused_container_must_match_its_reserved_pair(tmp_path, wrong):
    from hfmodem.core import wav
    import hashlib
    import json
    import numpy as np
    capture = ci.allocate_default(tmp_path)
    writer = capture.claim('modem')
    owner = ci.Recording.reserve(writer, 'pair.wav', clock_id='synthetic-clock')
    wav.write(owner.audio.path, np.zeros(3), 48000, ownership=owner)
    writer.update_status(state='finalized', written=1)
    capture.end()
    assert capture.lifecycle()['capture_complete']
    key = hashlib.sha256(owner.container.record['artifact_path'].encode()).hexdigest()
    reservation = writer.root / '.capture-artifacts' / (key + '.json')
    row = json.loads(reservation.read_text())
    row[wrong] = '00000000-0000-4000-8000-000000000001'
    reservation.write_text(json.dumps(row))
    with pytest.raises(ci.IdentityError):
        capture.lifecycle()



def test_owned_finite_artifacts_have_exclusive_binding_and_completion(tmp_path, monkeypatch):
    from hfmodem.shrike import onair
    capture, writer = owned(tmp_path, "finite")
    monkeypatch.setattr(onair, "_CAPTURE_ROOTS", {writer.root: writer})
    path = writer.root / "result.json"
    assert onair._capture_write_text(path, "{\"count\": 2}\n") == len(path.read_bytes())
    binding = ci.artifact_binding(path)
    assert binding["capture_id"] == capture.capture_id and binding["writer_id"] == writer.writer_id
    import hashlib
    component = writer._status["components"]["finite-" + hashlib.sha256(b"result.json").hexdigest()]
    assert component["state"] == "finalized" and component["written"] == 1
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        onair._capture_write_text(path, "replacement")
    assert path.read_bytes() == original
    assert writer.capture.lifecycle()["writers"][writer.role]["state"] == "partial"


def test_finite_legacy_text_and_exclusive_bytes_keep_original_io(tmp_path, monkeypatch):
    from hfmodem.shrike import onair
    monkeypatch.setattr(onair, "_CAPTURE_ROOTS", {})
    text = "test unicode \u03bb\n"
    path = tmp_path / "legacy.txt"
    assert onair._capture_write_text(path, text) == len(text)
    assert path.read_text() == text
    assert onair._capture_write_text(path, "next") == 4
    assert path.read_text() == "next"
    binary = tmp_path / "retained.bin"
    onair._capture_write_bytes(binary, b"synthetic bytes", legacy_exclusive=True)
    with pytest.raises(FileExistsError):
        onair._capture_write_bytes(binary, b"other", legacy_exclusive=True)
    assert binary.read_bytes() == b"synthetic bytes"


def test_expired_owner_cannot_create_new_finite_artifact_or_redirect(tmp_path, monkeypatch):
    from hfmodem.shrike import onair
    capture, writer = owned(tmp_path, "old-finite")
    _, newer = owned(tmp_path, "new-finite")
    monkeypatch.setattr(onair, "_CAPTURE_ROOTS", {writer.root: writer, newer.root: newer})
    capture.end()
    with pytest.raises(ci.IdentityError):
        onair._capture_write_text(writer.root / "late.json", "{}")
    assert not (writer.root / "late.json").exists()
    assert not (newer.root / "late.json").exists()


def test_owned_retained_transfer_keeps_bytes_and_manifest_shape(tmp_path, monkeypatch):
    from hfmodem.shrike import onair
    capture, writer = owned(tmp_path, "retained")
    monkeypatch.setattr(onair, "_CAPTURE_ROOTS", {writer.root: writer})
    retained = dict(field=b"synthetic field", suffix=b"synthetic suffix", counter=2, status=1, sl=2)
    result = onair._write_unconfirmed_transfer(writer.root, SimpleNamespace(transfer_unconfirmed=retained))
    assert result.read_bytes() == retained["field"] + retained["suffix"]
    assert ci.artifact_binding(result)["capture_id"] == capture.capture_id
    artifacts = [p for p in writer.root.glob("*.json")
                 if {"field", "suffix"}.issubset(json.loads(p.read_text()))]
    assert len(artifacts) == 1
    report = json.loads(artifacts[0].read_text())
    assert report["field"]["bytes"] == len(retained["field"])
    assert report["suffix"]["offset"] == len(retained["field"])
    assert ci.artifact_binding(artifacts[0])["capture_id"] == capture.capture_id
    with pytest.raises(FileExistsError):
        onair._write_unconfirmed_transfer(writer.root, SimpleNamespace(transfer_unconfirmed=retained))
    assert result.read_bytes() == retained["field"] + retained["suffix"]


def test_finite_json_failure_keeps_original_exception_and_partial_owner(tmp_path, monkeypatch):
    from hfmodem.shrike import onair
    _, writer = owned(tmp_path, "write-failure")
    monkeypatch.setattr(onair, "_CAPTURE_ROOTS", {writer.root: writer})
    original = OSError("synthetic publication failure")
    def fail(*args, **kwargs):
        raise original
    monkeypatch.setattr(ci.Artifact, "publish", fail)
    with pytest.raises(OSError) as caught:
        onair._capture_write_text(writer.root / "failed.json", "{}")
    assert caught.value is original
    assert writer.capture.lifecycle()["writers"][writer.role]["state"] == "partial"
    assert not (writer.root / "failed.json").exists()


def test_actual_tx_diagnostic_wavs_and_results_bind_existing_owner():
    import ast
    source = Path(__file__).resolve().parents[2] / "shrike/onair.py"
    tree = ast.parse(source.read_text())
    tx = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_tx_admitted")
    calls = [n for n in ast.walk(tx) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "wav" and n.func.attr == "write"]
    assert len(calls) == 2
    for call in calls:
        ownership = next(k.value for k in call.keywords if k.arg == "ownership")
        assert isinstance(ownership, ast.Call) and ownership.func.id == "_capture_recording_for"
        assert ast.dump(ownership.args[0]) == ast.dump(call.args[0])
    for name in ["_run_robust_bridge", "_drive_session"]:
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_capture_write_text"
                   for n in ast.walk(function))
