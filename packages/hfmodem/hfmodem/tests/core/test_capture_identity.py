# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fresh-filesystem ownership tests; no launcher, device or recording inputs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest

from hfmodem.core import capture_identity as ci

NOW = datetime(2026, 10, 4, 1, 2, 3, 456789, tzinfo=timezone.utc)


def capture(tmp_path, **kwargs):
    return ci.allocate({'capture': tmp_path / 'new'},
                       {'modem': 'capture', 'witness': 'capture'}, now=NOW, **kwargs)


def test_unique_simultaneous_clock_and_rollback_preserves_old_bytes(tmp_path):
    def arm(_):
        item = ci.allocate_default(tmp_path, now=NOW)
        item.claim('modem').reserve('stream.wav', clock_id='local-stream').publish(b'old')
        return item
    with ThreadPoolExecutor(max_workers=8) as pool:
        arms = list(pool.map(arm, range(8)))
    assert len({a.capture_id for a in arms}) == 8
    assert len({a.root('capture') for a in arms}) == 8
    older = ci.allocate_default(tmp_path, now=NOW - timedelta(days=1))
    assert older.capture_id not in {a.capture_id for a in arms}
    assert all((a.root('capture') / 'stream.wav').read_bytes() == b'old' for a in arms)
    assert all(a.root('capture').name.endswith(a.capture_id) for a in arms)


def test_uuid_collision_and_bounded_exhaustion(tmp_path):
    fixed = uuid.uuid4()
    first = ci.allocate_default(tmp_path, now=NOW, uuid_factory=lambda: fixed)
    choices = iter([fixed, uuid.uuid4()])
    second = ci.allocate_default(tmp_path, now=NOW, uuid_factory=lambda: next(choices))
    assert second.capture_id != first.capture_id
    calls = []
    def repeated():
        calls.append(1)
        return fixed
    with pytest.raises(FileExistsError):
        ci.allocate_default(tmp_path, now=NOW, uuid_factory=repeated)
    assert len(calls) == 32


@pytest.mark.parametrize('kind', ['empty', 'nonempty', 'file', 'symlink', 'dangling'])
def test_explicit_reuse_refused_before_any_producer(tmp_path, kind):
    root = tmp_path / 'new'
    if kind in ('empty', 'nonempty'):
        root.mkdir()
        if kind == 'nonempty':
            (root / 'foreign').write_bytes(b'protected')
    elif kind == 'file':
        root.write_bytes(b'protected')
    else:
        target = tmp_path / 'target'
        if kind == 'symlink':
            target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    before = (root / 'foreign').read_bytes() if kind == 'nonempty' else None
    with pytest.raises(FileExistsError):
        capture(tmp_path)
    assert before is None or (root / 'foreign').read_bytes() == before
    assert not (root / ci.MANIFEST).exists()


def test_partial_multi_root_allocation_preserved_on_failure(tmp_path, monkeypatch):
    original = Path.mkdir
    def mkdir(path, *args, **kwargs):
        if path == tmp_path / 'second':
            raise PermissionError('fake second-root refusal')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', mkdir)
    with pytest.raises(PermissionError):
        ci.allocate({'first': tmp_path / 'first', 'second': tmp_path / 'second'},
                    {'modem': 'first'}, now=NOW)
    assert (tmp_path / 'first').is_dir()
    assert not (tmp_path / 'first' / ci.MANIFEST).exists()


def test_default_only_retries_collisions_not_other_faults(tmp_path, monkeypatch):
    calls = []
    def refused(*args, **kwargs):
        calls.append(1)
        raise OSError('fake unsupported filesystem')
    monkeypatch.setattr(ci, 'allocate', refused)
    with pytest.raises(OSError):
        ci.allocate_default(tmp_path, now=NOW)
    assert calls == [1]


def test_registered_mirrors_agree_and_registry_is_immutable(tmp_path):
    item = ci.allocate({'capture': tmp_path / 'capture', 'transcript': tmp_path / 'working',
                        'native': tmp_path / 'native'},
                       {'modem': 'capture', 'transcript': 'transcript',
                        'tap': 'native', 'console-witness': 'native', 'launcher-witness': 'capture'},
                       now=NOW, transcript={'root_role': 'transcript', 'path': 'nested/label.log'})
    for role in item.record['roots']:
        assert ci.load(item.root(role)).capture_id == item.capture_id
    with pytest.raises(TypeError):
        item.record['roots']['capture']['path'] = str(tmp_path / 'other')
    for role in item.record['writers']:
        claim = item.claim(role)
        assert claim.writer_id == item.record['writers'][role]['writer_id']
    assert item.record['writers']['launcher-witness']['writer_id'] != item.record['writers']['console-witness']['writer_id']


@pytest.mark.parametrize('wrong', ['id', 'root', 'role', 'duplicate'])
def test_delegation_is_explicit_and_exclusive(tmp_path, wrong):
    item = capture(tmp_path)
    if wrong == 'id':
        with pytest.raises(ci.IdentityError):
            item.claim('modem', capture_id=str(uuid.uuid4()))
    elif wrong == 'root':
        with pytest.raises(ci.IdentityError):
            item.claim('modem', root=tmp_path)
    elif wrong == 'role':
        with pytest.raises(ci.IdentityError):
            item.claim('unregistered')
    else:
        item.claim('modem')
        with pytest.raises(FileExistsError):
            ci.load(item.root('capture')).claim('modem')


def test_role_and_artifact_races_have_one_owner(tmp_path):
    item = capture(tmp_path)
    def claim(_):
        try:
            return item.claim('modem')
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(claim, range(8)))
    writer, = [x for x in claims if x is not None]
    def reserve(_):
        try:
            return writer.reserve('stream.wav')
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        artifacts = list(pool.map(reserve, range(8)))
    assert len([x for x in artifacts if x is not None]) == 1
    with pytest.raises(FileExistsError):
        item.claim('witness').reserve('stream.wav')


@pytest.mark.parametrize('name', ['', '/absolute', '../outside', 'a/../outside', 'a//b', 'a/', './x', 'x' * 256])
def test_invalid_artifact_or_transcript_reference_preallocation(tmp_path, name):
    with pytest.raises(ci.IdentityError):
        ci.allocate({'capture': tmp_path / 'new'}, {'modem': 'capture'},
                    transcript={'root_role': 'capture', 'path': name}, now=NOW)
    assert not (tmp_path / 'new').exists()


def test_relative_subfolders_and_symlink_refusal(tmp_path):
    writer = capture(tmp_path).claim('modem')
    writer.reserve('nested/name.log').publish(b'one')
    assert (writer.root / 'nested/name.log').read_bytes() == b'one'
    (writer.root / 'outside').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ci.IdentityError):
        writer.reserve('outside/foreign')


@pytest.mark.parametrize('changed', ['symlink', 'directory', 'manifest', 'claim'])
def test_changed_root_and_writer_refused(tmp_path, changed):
    item = capture(tmp_path)
    writer = item.claim('modem')
    artifact = writer.reserve('stream.wav')
    root = writer.root
    if changed in ('directory', 'symlink'):
        root.rename(tmp_path / 'preserved')
        if changed == 'directory':
            root.mkdir()
        else:
            root.symlink_to(tmp_path / 'preserved', target_is_directory=True)
    elif changed == 'manifest':
        (root / ci.MANIFEST).write_text('{}')
    else:
        (root / '.capture-writers/modem.json').write_text('{}')
    with pytest.raises((ci.IdentityError, FileNotFoundError)):
        artifact.publish(b'new')
    assert not artifact.path.exists()


def test_parent_worktree_symlink_is_pinned_and_later_change_refused(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    parent = tmp_path / 'worktree'
    parent.symlink_to(real, target_is_directory=True)
    item = ci.allocate({'capture': parent / 'new'}, {'modem': 'capture'}, now=NOW)
    item.claim('modem').reserve('stream.wav').publish(b'same')
    parent.unlink()
    other = tmp_path / 'other'
    other.mkdir()
    parent.symlink_to(other, target_is_directory=True)
    with pytest.raises(ci.IdentityError):
        item.root('capture')


def test_existing_output_no_replace_and_partial_retained(tmp_path):
    writer = capture(tmp_path).claim('modem')
    artifact = writer.reserve('stream.wav')
    artifact.path.write_bytes(b'foreign')
    with pytest.raises(FileExistsError):
        artifact.publish(b'new')
    assert artifact.path.read_bytes() == b'foreign'
    assert list(writer.root.glob('.stream.wav.*.partial'))


def test_ongoing_handle_then_delayed_original_reservation_after_end(tmp_path):
    old = capture(tmp_path)
    writer = old.claim('modem')
    stream = writer.reserve('stream.wav', clock_id='original-local')
    late = writer.reserve('late.json')
    handle = stream.open()
    handle.write(b'partial')
    old.end()
    newer = ci.allocate_default(tmp_path, now=NOW)
    with pytest.raises(ci.IdentityError):
        writer.reserve('new.json')
    with pytest.raises(ci.IdentityError):
        old.claim('witness')
    handle.write(b'-tail')
    handle.close()
    late.publish(b'late')
    assert stream.path.read_bytes() == b'partial-tail'
    assert late.path.read_bytes() == b'late'
    assert not (newer.root('capture') / 'late.json').exists()


def test_versions_keep_originals_and_reject_foreign_missing_or_mismatched_slot(tmp_path):
    writer = capture(tmp_path).claim('modem')
    artifact = writer.reserve('stream.wav', clock_id='local-stream')
    slot = artifact.versions('stream.json')
    slot.update({'state': 'ongoing', 'samples': 0})
    stream = artifact.open()
    stream.write(b'fake-audio')
    stream.close()
    slot.update({'state': 'finalized', 'samples': 10})
    original = (writer.root / '.stream.json.versions/00000000.json').read_bytes()
    current = (writer.root / 'stream.json').read_bytes()
    assert json.loads(original)['state'] == 'ongoing'
    assert json.loads(current)['state'] == 'finalized'
    assert json.loads(current)['previous_sha256'] == hashlib.sha256(original).hexdigest()
    assert ci.validate_artifact(artifact.path, clock_id='local-stream')['status'] == 'bound'
    assert artifact.path.read_bytes() == b'fake-audio'
    (writer.root / 'stream.json').write_text('{}')
    with pytest.raises(ci.IdentityError):
        slot.update({'state': 'finalized'})
    with pytest.raises(ci.IdentityError):
        ci.validate_artifact(artifact.path)
    (writer.root / 'stream.json').unlink()
    with pytest.raises(FileNotFoundError):
        slot.update({'state': 'finalized'})


def test_new_name_requires_manifest_legacy_remains_unbound(tmp_path):
    item = ci.allocate_default(tmp_path, now=NOW)
    assert ci.provenance(item.root('capture'))['status'] == 'bound'
    (item.root('capture') / ci.MANIFEST).unlink()
    assert ci.provenance(Path(item.record['roots']['capture']['path']))['status'] == 'incomplete'
    legacy = tmp_path / 'onair-1003-1043'
    legacy.mkdir()
    assert ci.provenance(legacy)['status'] == 'legacy-unbound'


@pytest.mark.parametrize('conflict', ['extra-root', 'registry', 'writer'])
def test_conflicting_registry_and_extra_root_are_ambiguous(tmp_path, conflict):
    item = capture(tmp_path)
    root = item.root('capture')
    raw = (root / ci.MANIFEST).read_bytes()
    if conflict == 'extra-root':
        extra = tmp_path / 'extra'
        extra.mkdir()
        (extra / ci.MANIFEST).write_bytes(raw)
        assert ci.provenance(extra)['status'] == 'ambiguous'
    else:
        record = json.loads(raw)
        if conflict == 'registry':
            record['roots']['capture']['resolved_parent'] += '/moved'
        else:
            record['writers']['modem']['writer_id'] = str(uuid.uuid4())
        (root / ci.MANIFEST).write_text(json.dumps(record))
        assert ci.provenance(root)['status'] == 'ambiguous'


@pytest.mark.parametrize('arguments,okay', [([], True), (['--outdir', 'same'], True),
    (['--outdir=other', '--outdir=same'], True), (['--outdir=same', '--outdir=other'], False),
    (['--outdir'], False), (['--outdir', '--wrong'], False)])
def test_effective_final_output_preflight_before_producer(tmp_path, monkeypatch, arguments, okay):
    monkeypatch.chdir(tmp_path)
    producers = []
    def launch():
        ci.effective_outdir(arguments, tmp_path / 'same')
        producers.append('fake producer')
    if okay:
        launch()
        assert producers == ['fake producer']
    else:
        with pytest.raises(ci.IdentityError):
            launch()
        assert not producers


def test_artifact_clock_mismatch_refuses_association_without_rebasing(tmp_path):
    writer = capture(tmp_path).claim('modem')
    artifact = writer.reserve('stream.wav', clock_id='stream-48000')
    artifact.versions('stream.json').update({'samplerate': 48000, 'sample_start': 123, 'sample_end': 456})
    artifact.publish(b'fake')
    with pytest.raises(ci.IdentityError):
        ci.validate_artifact(artifact.path, clock_id='witness-12000')
    side = json.loads((writer.root / 'stream.json').read_text())
    assert (side['samplerate'], side['sample_start'], side['sample_end']) == (48000, 123, 456)


def test_reader_duplicate_same_role_claim_is_ambiguous(tmp_path):
    item = capture(tmp_path)
    writer = item.claim('modem')
    root = writer.root
    claim = root / '.capture-writers/modem.json'
    (claim.parent / 'copy.json').write_bytes(claim.read_bytes())
    assert ci.provenance(root)['status'] == 'ambiguous'


def _copied_back(path):
    held = path.with_name(path.name + '.held')
    path.rename(held)
    shutil.copytree(held, path, symlinks=True)
    shutil.rmtree(held)


def test_roots_copied_away_and_back_keep_their_authority(tmp_path):
    item = ci.allocate({'capture': tmp_path / 'captures/new', 'transcript': tmp_path / 'working/new'},
                       {'modem': 'capture', 'log': 'transcript'}, now=NOW)
    item.claim('modem').reserve('nested/stream.wav').publish(b'old')
    item.claim('log').reserve('arm.log').publish(b'log')
    before = {role: item.root(role).stat().st_ino for role in item.record['roots']}
    for role in item.record['roots']:
        _copied_back(item.root(role))
    loaded = ci.load(tmp_path / 'captures/new')
    assert {role: loaded.root(role).stat().st_ino for role in loaded.record['roots']} != before
    assert ci.provenance(tmp_path / 'working/new')['status'] == 'bound'
    assert ci.artifact_binding(loaded.root('capture') / 'nested/stream.wav')['writer_role'] == 'modem'
    with (loaded.root('capture') / 'nested/stream.wav').open('ab') as stream:
        writer = ci.retained_writer(loaded, 'modem', 'nested/stream.wav', stream.fileno())
    writer.reserve('nested/later.bin').publish(b'new')
    assert ci.evidence_reference(loaded.root('capture'))['status'] == 'bound'
    assert 'inode' not in (loaded.root('capture') / ci.MANIFEST).read_text()


def test_legacy_inode_records_load_and_are_never_rewritten(tmp_path):
    item = ci.allocate({'capture': tmp_path / 'new'}, {'modem': 'capture'}, now=NOW)
    root = item.root('capture')
    record = json.loads((root / ci.MANIFEST).read_text())
    record['roots']['capture']['identity'] = {'device': 1, 'inode': 2}
    raw = ci._json(record)
    (root / ci.MANIFEST).write_bytes(raw)
    for marker in root.rglob(ci.DIRECTORY_OWNER):
        row = json.loads(marker.read_text())
        row.update(registry_sha256=hashlib.sha256(raw).hexdigest(),
                   root_identity={'device': 1, 'inode': 2},
                   directory_identity={'device': 1, 'inode': 3})
        marker.write_bytes(ci._json(row))
    legacy = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    loaded = ci.load(root)
    loaded.claim('modem').reserve('stream.wav').publish(b'new')
    assert ci.artifact_binding(root / 'stream.wav')['writer_role'] == 'modem'
    assert all(path.read_bytes() == data for path, data in legacy.items())
