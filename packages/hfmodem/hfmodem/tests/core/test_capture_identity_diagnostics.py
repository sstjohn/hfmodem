# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Finite filesystem-only capture repairs; no receiver or hardware calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import capture_identity as ci
from hfmodem.shrike import onair, p3trial, p4experiment


def _capture(path):
    capture, writer = ci.prepare_output(path, offline=True)
    return capture, writer


def _probe(monkeypatch):
    arrays = [np.arange(4, dtype=np.float32), np.arange(6, dtype=np.float64)]
    probe = SimpleNamespace(received=[(i, i * 100, x) for i, x in enumerate(arrays)],
        emissions=[], listen_seconds=1, trigger_word=None, requested=False,
        grant_sample=None, payload=bytes(), status=None, guard_exceptions=[],
        windows=[], reason='synthetic incomplete')
    def observe(audio, start, fs):
        return dict(start_sample=start, end_sample=start+len(audio),
                    p1_controls=[], p4_candidate=None)
    monkeypatch.setattr(p4experiment, 'observe', observe)
    return probe, arrays


def _assert_binding(path, capture, writer):
    binding = ci.artifact_binding(path)
    assert binding['capture_id'] == capture.capture_id
    assert binding['writer_id'] == writer.writer_id
    assert binding['writer_role'] == writer.role


@pytest.mark.parametrize('nested', [False, True])
def test_specific_root_preserves_sibling_nested_and_delayed_old_ticket(tmp_path, monkeypatch, nested):
    monkeypatch.setattr(onair, '_CAPTURE_ROOTS', {})
    old, old_writer = _capture(tmp_path/'old')
    old_path = old_writer.root/'retained.bin'
    old_writer.reserve(old_path.name).publish(bytes(range(4)))
    old_bytes = old_path.read_bytes()
    ticket = old_writer.ticket('delayed.bin')
    onair._CAPTURE_ROOTS[old_writer.root] = old_writer
    old.end()
    path = old_writer.root/'new' if nested else tmp_path/'new'
    new, new_writer = _capture(path)
    onair._CAPTURE_ROOTS[new_writer.root] = new_writer
    output = new_writer.root/'current.bin'
    assert onair._capture_writer_for(output) is new_writer
    onair._capture_write_bytes(output, bytes(range(6)))
    _assert_binding(output, new, new_writer)
    assert old_path.read_bytes() == old_bytes
    ticket.reserve().publish(bytes(range(2)))
    _assert_binding(old_writer.root/'delayed.bin', old, old_writer)
    assert output.read_bytes() == bytes(range(6))
    assert not (new_writer.root/'delayed.bin').exists()


def test_ended_specific_owner_never_falls_back_to_active_ancestor(tmp_path, monkeypatch):
    old, old_writer = _capture(tmp_path/'old')
    new, new_writer = _capture(old_writer.root/'new')
    monkeypatch.setattr(onair, '_CAPTURE_ROOTS',
                        {old_writer.root:old_writer, new_writer.root:new_writer})
    new.end()
    output = new_writer.root/'refused.bin'
    before = sorted((old_writer.root/'.capture-artifacts').iterdir())
    assert onair._capture_writer_for(output) is new_writer
    with pytest.raises(ci.IdentityError):
        onair._capture_write_bytes(output, bytes(range(2)))
    assert not output.exists() and not old.ended()
    assert sorted((old_writer.root/'.capture-artifacts').iterdir()) == before


@pytest.mark.parametrize('kind', [p3trial.TimingTrial, p3trial.EntryTimingTrial])
def test_trial_exact_legacy_facts_have_owned_binding(tmp_path, kind):
    legacy = tmp_path/'legacy';legacy.mkdir()
    capture, writer = _capture(tmp_path/'owned')
    trial = kind('B')
    trial.write(legacy)
    trial.write(writer.root, capture_writer=writer)
    path = writer.root/trial.filename
    assert path.read_bytes() == (legacy/trial.filename).read_bytes()
    _assert_binding(path, capture, writer)
    assert writer._status['components'][path.stem] == dict(
        state='finalized', pending=0, written=1, lost=0, partial=0)


@pytest.mark.parametrize('kind', [p3trial.TimingTrial, p3trial.EntryTimingTrial])
@pytest.mark.parametrize('failure', ['foreign', 'ended'])
def test_trial_refuses_foreign_or_ended_without_changing_bytes(tmp_path, kind, failure):
    capture, writer = _capture(tmp_path/'owned')
    trial = kind('B');path=writer.root/trial.filename
    foreign = bytes(range(8))
    if failure == 'foreign':path.write_bytes(foreign)
    else:capture.end()
    with pytest.raises(FileExistsError if failure == 'foreign' else ci.IdentityError):
        trial.write(writer.root, capture_writer=writer)
    assert path.read_bytes() == foreign if failure == 'foreign' else not path.exists()
    assert writer._status['failed'] and writer._status['components'][path.stem]['state'] == 'partial'


@pytest.mark.parametrize('kind', [p3trial.TimingTrial, p3trial.EntryTimingTrial])
@pytest.mark.parametrize('failure', ['publish', 'final-journal'])
def test_trial_publication_error_propagates_and_marks_partial(tmp_path, monkeypatch, kind, failure):
    capture, writer = _capture(tmp_path/'owned');trial=kind('B')
    original = ci.Artifact.publish
    def refuse(artifact, raw, *, published=None):
        if artifact.path.name == trial.filename:raise OSError('synthetic publication failure')
        return original(artifact, raw)
    if failure == 'publish':
        monkeypatch.setattr(ci.Artifact, 'publish', refuse)
    else:
        original_status = type(writer).update_status
        def refuse_status(owner, **kwargs):
            if owner is writer and kwargs['state'] == 'finalized':raise OSError('synthetic publication failure')
            return original_status(owner, **kwargs)
        monkeypatch.setattr(type(writer), 'update_status', refuse_status)
    with pytest.raises(OSError, match='synthetic publication failure'):
        trial.write(writer.root, capture_writer=writer)
    path=writer.root/trial.filename
    assert path.exists() == (failure == 'final-journal')
    if path.exists():_assert_binding(path, capture, writer)
    assert writer._status['components'][path.stem]['written'] == int(path.exists())
    assert writer._status['failed']


@pytest.mark.parametrize('recording', [False, True])
def test_old_queued_ticket_cannot_enter_new_nested_capture(tmp_path, recording):
    old, old_writer = _capture(tmp_path/'old')
    if recording:
        ticket = ci.Recording.queued(old_writer, 'new/deferred.wav', clock_id='synthetic')
    else:
        ticket = old_writer.ticket('new/deferred.bin')
    allowed = old_writer.ticket('retained.bin')
    old.end()
    new, new_writer = _capture(old_writer.root/'new')
    before = sorted(p.relative_to(new_writer.root) for p in new_writer.root.rglob('*'))
    with pytest.raises(ci.IdentityError, match='^Directory has no complete ownership record$'):
        ticket.reserve()
    assert sorted(p.relative_to(new_writer.root) for p in new_writer.root.rglob('*')) == before
    allowed.reserve().publish(bytes(range(3)))
    _assert_binding(old_writer.root/'retained.bin', old, old_writer)
    new_writer.reserve('current.bin').publish(bytes(range(4)))
    _assert_binding(new_writer.root/'current.bin', new, new_writer)


@pytest.mark.parametrize('marker', [ci.MANIFEST, '.capture-writers', '.capture-artifacts'])
def test_old_ticket_refuses_malformed_or_partial_descendant_root(tmp_path, marker):
    capture, writer = _capture(tmp_path/'old')
    ticket = writer.ticket('new/refused.bin');capture.end()
    child=writer.root/'new';child.mkdir()
    boundary=child/marker
    if marker == ci.MANIFEST:boundary.write_bytes(bytes(range(5)))
    else:boundary.mkdir()
    with pytest.raises(ci.IdentityError, match='^Directory has no complete ownership record$'):
        ticket.reserve()
    assert not (child/'refused.bin').exists()


def test_reserved_publication_rechecks_descendant_boundary(tmp_path):
    capture, writer = _capture(tmp_path/'old')
    artifact=writer.reserve('child/held.bin')
    temporary, stream=artifact.temporary()
    with stream:stream.write(bytes(range(3)))
    child=artifact.path.parent
    (child/'.capture-writers').mkdir()
    with pytest.raises(ci.IdentityError, match='cross capture roots'):
        artifact.publish_file(temporary)
    assert not artifact.path.exists() and temporary.read_bytes() == bytes(range(3))


def test_experiment_exact_json_npz_content_and_both_bindings(tmp_path, monkeypatch):
    probe, arrays = _probe(monkeypatch)
    legacy=tmp_path/'legacy';legacy.mkdir()
    capture, writer = _capture(tmp_path/'owned')
    expected = p4experiment.write_report(probe, legacy, mode='replay')
    actual = p4experiment.write_report(probe, writer.root, mode='replay', capture_writer=writer)
    assert actual == expected
    report=writer.root/'p4-mail-experiment.json';receive=writer.root/'p4-mail-rx.npz'
    assert report.read_bytes() == (legacy/report.name).read_bytes()
    assert json.loads(report.read_text()) == actual
    for path in (report, receive):_assert_binding(path, capture, writer)
    with np.load(receive, allow_pickle=False) as stored:
        assert stored.files == [f'probe{i}' for i in range(len(arrays))]
        for i, expected in enumerate(arrays):
            actual=stored[f'probe{i}']
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert np.array_equal(actual, expected)
    assert writer._status['components']['p4-mail-experiment'] == dict(
        state='finalized', pending=0, written=2, lost=0, partial=0)


@pytest.mark.parametrize('foreign_name', ['p4-mail-experiment.json', 'p4-mail-rx.npz'])
def test_experiment_reserves_both_before_publication(tmp_path, monkeypatch, foreign_name):
    probe, _ = _probe(monkeypatch);capture, writer=_capture(tmp_path/'owned')
    foreign = writer.root/foreign_name;raw=bytes(range(8));foreign.write_bytes(raw)
    with pytest.raises(FileExistsError):
        p4experiment.write_report(probe, writer.root, capture_writer=writer)
    assert foreign.read_bytes() == raw
    other = 'p4-mail-rx.npz' if foreign_name.endswith('.json') else 'p4-mail-experiment.json'
    assert not (writer.root/other).exists()
    assert writer._status['failed']


def test_experiment_ended_creates_neither_output(tmp_path, monkeypatch):
    probe, _ = _probe(monkeypatch);capture, writer=_capture(tmp_path/'owned');capture.end()
    with pytest.raises(ci.IdentityError):
        p4experiment.write_report(probe, writer.root, capture_writer=writer)
    assert not (writer.root/'p4-mail-rx.npz').exists()
    assert not (writer.root/'p4-mail-experiment.json').exists()


@pytest.mark.parametrize('failure', ['serialize', 'report'])
def test_experiment_original_error_preserves_partial_evidence(tmp_path, monkeypatch, failure):
    probe, _ = _probe(monkeypatch);capture, writer=_capture(tmp_path/'owned')
    if failure == 'serialize':
        def refuse(*args, **kwargs):raise OSError('synthetic serialization failure')
        monkeypatch.setattr(p4experiment.np, 'savez_compressed', refuse)
    else:
        original = ci.Artifact.publish
        def refuse(artifact, raw, *, published=None):
            if artifact.path.name.endswith('experiment.json'):raise OSError('synthetic report failure')
            return original(artifact, raw)
        monkeypatch.setattr(ci.Artifact, 'publish', refuse)
    with pytest.raises(OSError, match='synthetic'):
        p4experiment.write_report(probe, writer.root, capture_writer=writer)
    assert not (writer.root/'p4-mail-experiment.json').exists()
    if failure == 'report':
        _assert_binding(writer.root/'p4-mail-rx.npz', capture, writer)
        assert writer._status['components']['p4-mail-experiment']['written'] == 1
    else:
        assert not (writer.root/'p4-mail-rx.npz').exists()
        assert len(list(writer.root.glob('.*.partial'))) == 1
    assert writer._status['failed']


@pytest.mark.parametrize('winner', ['allocator', 'writer'])
def test_directory_exclusive_winner_orders_preserve_each_owner(tmp_path, monkeypatch, winner):
    capture, writer = _capture(tmp_path / 'old')
    target = writer.root / 'new'
    queued = writer.ticket('new/deferred.bin')
    sibling = writer.ticket('sibling.bin')
    capture.end()
    original = type(target).mkdir
    paused = []
    def mkdir(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == target and not paused:
            paused.append(winner)
            if winner == 'allocator':
                with pytest.raises(ci.IdentityError):
                    queued.reserve()
            else:
                with pytest.raises(FileExistsError):
                    _capture(target)
        return value
    monkeypatch.setattr(type(target), 'mkdir', mkdir)
    if winner == 'allocator':
        new, own = _capture(target)
        own.reserve('current.bin').publish(bytes(range(3)))
        _assert_binding(target / 'current.bin', new, own)
        assert not (target / 'deferred.bin').exists()
    else:
        item = queued.reserve()
        item.publish(bytes(range(3)))
        _assert_binding(item.path, capture, writer)
        assert not (target / ci.MANIFEST).exists()
    assert paused == [winner]
    sibling.reserve().publish(bytes(range(2)))
    _assert_binding(writer.root / 'sibling.bin', capture, writer)


def test_initial_parent_child_allocation_does_not_adopt_child(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    original = type(parent).mkdir
    children = []
    def mkdir(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == parent and not children:
            children.append(None)
            children[0] = _capture(parent / 'child')
        return value
    monkeypatch.setattr(type(parent), 'mkdir', mkdir)
    capture, writer = _capture(parent)
    child, own = children[0]
    with pytest.raises(ci.IdentityError):
        writer.reserve('child/forbidden.bin')
    writer.reserve('current.bin').publish(bytes(range(2)))
    own.reserve('current.bin').publish(bytes(range(3)))
    _assert_binding(parent / 'current.bin', capture, writer)
    _assert_binding(parent / 'child/current.bin', child, own)
    assert not (parent / 'child/forbidden.bin').exists()


def test_same_root_roles_claim_concurrently_and_share_only_bound_folders(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    capture = ci.allocate({'capture': tmp_path / 'same'},
                          {'first': 'capture', 'second': 'capture'})
    assert (capture.root('capture') / '.capture-writer-status' / ci.DIRECTORY_OWNER).is_file()
    ready = threading.Barrier(2, timeout=5)
    initialize = ci.Writer._initialize_status
    def initialize_together(writer):
        ready.wait()
        return initialize(writer)
    monkeypatch.setattr(ci.Writer, '_initialize_status', initialize_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(capture.claim, 'first')
        second_future = pool.submit(capture.claim, 'second')
        first, second = first_future.result(), second_future.result()
    bind = ci._bind_directory
    paused = []
    def paused_bind(root, path, context):
        if path == first.root / 'shared' and not paused:
            paused.append(True)
            with pytest.raises(ci.IdentityError):
                second.reserve('shared/second.bin')
        return bind(root, path, context)
    monkeypatch.setattr(ci, '_bind_directory', paused_bind)
    first.reserve('shared/first.bin').publish(bytes(range(2)))
    second.reserve('shared/second.bin').publish(bytes(range(3)))
    assert paused == [True]
    a = ci.artifact_binding(first.root / 'shared/first.bin')
    b = ci.artifact_binding(second.root / 'shared/second.bin')
    assert a['writer_id'] == first.writer_id and b['writer_id'] == second.writer_id
    assert a['writer_id'] != b['writer_id']


def test_same_identity_nested_registered_roots_have_distinct_directory_authority(tmp_path):
    capture = ci.allocate({'outer': tmp_path / 'outer', 'inner': tmp_path / 'outer/nested'},
                          {'first': 'outer', 'second': 'inner'})
    outer, inner = capture.claim('first'), capture.claim('second')
    with pytest.raises(ci.IdentityError):
        outer.reserve('nested/forbidden.bin')
    outer.reserve('sibling.bin').publish(bytes(range(2)))
    inner.reserve('current.bin').publish(bytes(range(3)))
    _assert_binding(outer.root / 'sibling.bin', capture, outer)
    _assert_binding(inner.root / 'current.bin', capture, inner)
    assert not (inner.root / 'forbidden.bin').exists()


@pytest.mark.parametrize('failure', ['unbound', 'malformed', 'foreign'])
def test_directory_authority_never_adopts_partial_or_foreign_state(tmp_path, failure):
    capture, writer = _capture(tmp_path / 'capture')
    folder = writer.root / 'nested'
    if failure == 'unbound':
        folder.mkdir()
    else:
        writer.reserve('nested/retained.bin').publish(bytes(range(2)))
        marker = folder / ci.DIRECTORY_OWNER
        if failure == 'malformed':
            marker.write_bytes(bytes(range(4)))
        else:
            row = json.loads(marker.read_text())
            row['root_role'] = 'foreign'
            marker.write_bytes(ci._json(row))
    before = {str(p.relative_to(writer.root)): p.read_bytes()
              for p in writer.root.rglob('*') if p.is_file()}
    with pytest.raises(ci.IdentityError):
        writer.reserve('nested/new.bin')
    after = {str(p.relative_to(writer.root)): p.read_bytes()
             for p in writer.root.rglob('*') if p.is_file()}
    assert after == before and not (folder / 'new.bin').exists()


@pytest.mark.parametrize('operation', ['open', 'publish', 'temporary', 'publish_file'])
def test_artifact_operations_recheck_directory_authority(tmp_path, operation):
    _, writer = _capture(tmp_path / 'capture')
    artifact = writer.reserve('nested/current.bin')
    temporary = None
    if operation == 'publish_file':
        temporary, sink = artifact.temporary()
        with sink:
            sink.write(bytes(range(3)))
    (artifact.path.parent / ci.DIRECTORY_OWNER).unlink()
    with pytest.raises(ci.IdentityError):
        if operation == 'publish':
            artifact.publish(bytes(range(3)))
        elif operation == 'publish_file':
            artifact.publish_file(temporary)
        else:
            getattr(artifact, operation)()
    assert not artifact.path.exists()
    if temporary is not None:
        assert temporary.read_bytes() == bytes(range(3))


@pytest.mark.parametrize('kind', ['transcript-retained', 'analysis-corridor'])
def test_supported_relative_directory_and_analysis_outputs_remain_owned(tmp_path, kind):
    if kind == 'analysis-corridor':
        capture, writer = _capture(tmp_path / 'capture')
        corridor = writer.root / 'monitor-runs'
        corridor.mkdir()
        child = ci.allocate_analysis(writer.root)
        own = child.claim('analysis')
        own.reserve('nested/current.bin').publish(bytes(range(3)))
        _assert_binding(own.root / 'nested/current.bin', child, own)
        assert not (corridor / ci.DIRECTORY_OWNER).exists()
        with pytest.raises(ci.IdentityError):
            writer.reserve('monitor-runs/forbidden.bin')
    else:
        capture = ci.allocate_arm(tmp_path, log_label='relative/transcript')
        writer = capture.claim('launcher-transcript')
        name = capture.record['transcript']['path']
        artifact = writer.reserve(name)
        with artifact.open() as stream:
            stream.write(bytes(range(3)))
            stream.flush()
            retained = ci.retained_writer(capture, writer.role, name, stream.fileno())
            retained.update_status(state='finalized', written=1)
        _assert_binding(artifact.path, capture, writer)
        assert capture.lifecycle()['writers'][writer.role]['written'] == 1


def test_directory_binding_failure_is_preserved_and_cannot_be_adopted(tmp_path, monkeypatch):
    _, writer = _capture(tmp_path / 'capture')
    bind = ci._bind_directory
    primary = OSError('synthetic directory binding failure')
    def failed(root, path, context):
        if path == writer.root / 'partial':
            raise primary
        return bind(root, path, context)
    monkeypatch.setattr(ci, '_bind_directory', failed)
    with pytest.raises(OSError) as caught:
        writer.reserve('partial/current.bin')
    assert caught.value is primary and (writer.root / 'partial').is_dir()
    monkeypatch.setattr(ci, '_bind_directory', bind)
    with pytest.raises(ci.IdentityError):
        writer.reserve('partial/current.bin')
    assert not (writer.root / 'partial/current.bin').exists()
    writer.reserve('unrelated.bin').publish(bytes(range(2)))
    for name in (ci.DIRECTORY_OWNER, 'unrelated/' + ci.DIRECTORY_OWNER):
        with pytest.raises(ci.IdentityError):
            writer.reserve(name)
        with pytest.raises(ci.IdentityError):
            writer.ticket(name)


def _publication_witness(monkeypatch, output, events):
    def wrap(method):
        original = getattr(ci.Artifact, method)
        def invoke(artifact, value, *, published=None):
            if artifact.path != output or published is None:
                return original(artifact, value, published=published)
            def notify():
                events.append('publication')
                published()
            return original(artifact, value, published=notify)
        monkeypatch.setattr(ci.Artifact, method, invoke)
    wrap('publish')
    wrap('publish_file')


@pytest.mark.parametrize('kind', [p3trial.TimingTrial, p3trial.EntryTimingTrial])
def test_trial_link_success_cleanup_failure_retains_publication_credit(tmp_path, monkeypatch, kind):
    capture, writer = _capture(tmp_path / 'capture')
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    trial = kind('B')
    trial.write(legacy)
    expected = (legacy / trial.filename).read_bytes()
    events = []
    _publication_witness(monkeypatch, writer.root / trial.filename, events)
    original_unlink = type(writer.root).unlink
    primary = OSError('synthetic post-link cleanup failure')
    notified = []
    def unlink(path, *args, **kwargs):
        if path.parent == writer.root and path.name.startswith('.' + trial.filename + '.') and path.suffix == '.partial':
            notified.append(path)
            events.append('cleanup')
            raise primary
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(type(writer.root), 'unlink', unlink)
    with pytest.raises(OSError) as caught:
        trial.write(writer.root, capture_writer=writer)
    output = writer.root / trial.filename
    assert caught.value is primary and len(notified) == 1
    assert events == ['publication', 'cleanup']
    assert output.read_bytes() == expected == notified[0].read_bytes()
    _assert_binding(output, capture, writer)
    state = capture.lifecycle()['writers'][writer.role]
    assert state['state'] == 'partial' and state['written'] == 1
    assert state['pending'] == 0 and state['partial'] == 1


@pytest.mark.parametrize('output_name', ['p4-mail-rx.npz', 'p4-mail-experiment.json'])
def test_experiment_link_success_cleanup_failure_retains_exact_count(tmp_path, monkeypatch, output_name):
    capture, writer = _capture(tmp_path / 'capture')
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    probe, _ = _probe(monkeypatch)
    p4experiment.write_report(probe, legacy)
    expected = (legacy / output_name).read_bytes()
    events = []
    _publication_witness(monkeypatch, writer.root / output_name, events)
    original_unlink = type(writer.root).unlink
    primary = OSError('synthetic post-link cleanup failure')
    retained = []
    def unlink(path, *args, **kwargs):
        if path.parent == writer.root and path.name.startswith('.' + output_name + '.') and path.suffix == '.partial':
            retained.append(path)
            events.append('cleanup')
            raise primary
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(type(writer.root), 'unlink', unlink)
    with pytest.raises(OSError) as caught:
        p4experiment.write_report(probe, writer.root, capture_writer=writer)
    output = writer.root / output_name
    written = 1 if output_name.endswith('.npz') else 2
    assert caught.value is primary and len(retained) == 1
    assert events == ['publication', 'cleanup']
    assert output.read_bytes() == expected == retained[0].read_bytes()
    _assert_binding(output, capture, writer)
    state = capture.lifecycle()['writers'][writer.role]
    assert state['state'] == 'partial' and state['written'] == written
    assert state['pending'] == 2 - written and state['partial'] == 1
    if output_name.endswith('.npz'):
        assert not (writer.root / 'p4-mail-experiment.json').exists()
    else:
        _assert_binding(writer.root / 'p4-mail-rx.npz', capture, writer)


@pytest.mark.parametrize('route', ['timing', 'entry', 'npz', 'json'])
def test_failed_link_has_no_publication_notification_credit(tmp_path, monkeypatch, route):
    capture, writer = _capture(tmp_path / 'capture')
    trial = (p3trial.TimingTrial if route == 'timing' else p3trial.EntryTimingTrial)('B')
    probe, _ = _probe(monkeypatch)
    output_name = trial.filename if route in ('timing', 'entry') else ('p4-mail-rx.npz' if route == 'npz' else 'p4-mail-experiment.json')
    events = []
    _publication_witness(monkeypatch, writer.root / output_name, events)
    original_link = ci.os.link
    primary = OSError('synthetic final-link failure')
    attempted = []
    def link(source, destination, *args, **kwargs):
        if Path(destination) == writer.root / output_name:
            attempted.append(Path(source))
            raise primary
        return original_link(source, destination, *args, **kwargs)
    monkeypatch.setattr(ci.os, 'link', link)
    with pytest.raises(OSError) as caught:
        if route in ('timing', 'entry'):
            trial.write(writer.root, capture_writer=writer)
        else:
            p4experiment.write_report(probe, writer.root, capture_writer=writer)
    expected_written = 1 if route == 'json' else 0
    total = 1 if route in ('timing', 'entry') else 2
    assert caught.value is primary and len(attempted) == 1
    assert events == []
    assert attempted[0].is_file() and not (writer.root / output_name).exists()
    state = capture.lifecycle()['writers'][writer.role]
    assert state['state'] == 'partial' and state['written'] == expected_written
    assert state['pending'] == total - expected_written and state['partial'] == 1
    if route == 'json':
        _assert_binding(writer.root / 'p4-mail-rx.npz', capture, writer)
