"""Fake-exec-only launcher handoff; no live script or producer is invoked."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from hfmodem.core import capture_identity as ci

ROOT = Path(__file__).resolve().parents[5]
HELPER = ROOT / 'tools/capture_identity.py'
if not HELPER.is_file():
    pytest.skip('private launcher helper is absent from this distribution', allow_module_level=True)
SPEC = importlib.util.spec_from_file_location('capture_launcher_under_test', HELPER)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_launch_fake_exec_has_actual_owned_descriptor_and_unique_log(tmp_path):
    calls = []
    def fake_exec(script, words, environment):
        assert script == 'never-executed'
        assert words[1] == '--capture-prepared'
        capture, path, rest = launcher.prepared(words[2:], environment['HFMODEM_CAPTURE_FD'])
        assert rest == ['pactor', 'synthetic-peer']
        assert path.name == 'label--' + capture.capture_id + '.log'
        assert path.parent == capture.root('transcript') / 'nested'
        os.write(int(environment['HFMODEM_CAPTURE_FD']), b'fake transcript')
        calls.append(path)
    launcher.launch(tmp_path, 'never-executed', ['pactor', 'synthetic-peer', '--log', 'nested/label'], exec_fn=fake_exec)
    assert len(calls) == 1 and calls[0].read_bytes() == b'fake transcript'


def test_delegated_console_launch_keeps_same_id_and_claims_once(tmp_path):
    capture = ci.allocate_arm(tmp_path, log_label='native', native_root=tmp_path / 'arms/planned')
    arguments = ['pactor', 'synthetic-peer', '--capture-id', capture.capture_id,
                 '--capture-root', str(capture.root('capture'))]
    item, artifact, sink, rest = launcher.prepare(tmp_path, arguments)
    try:
        assert item.capture_id == capture.capture_id
        assert artifact.path.name.endswith(capture.capture_id + '.log')
        assert rest == ['pactor', 'synthetic-peer']
        with pytest.raises(FileExistsError):
            launcher.prepare(tmp_path, arguments)
    finally:
        sink.close()


@pytest.mark.parametrize('wrong', ['id', 'root', 'outdir', 'kind', 'environment-alone'])
def test_bad_delegation_and_outdir_refuse_before_fake_exec(tmp_path, wrong):
    capture = ci.allocate_arm(tmp_path)
    arguments = ['pactor', 'synthetic-peer', '--capture-id', capture.capture_id,
                 '--capture-root', str(capture.root('capture'))]
    if wrong == 'id':
        arguments[3] = '00000000-0000-0000-0000-000000000000'
    elif wrong == 'root':
        arguments[5] = str(tmp_path / 'unregistered')
    elif wrong == 'outdir':
        arguments += ['--outdir', str(tmp_path / 'other')]
    elif wrong == 'kind':
        arguments[:1] = ['pounce', 'pactor']
    else:
        with pytest.raises(ci.IdentityError):
            launcher.prepared(['pactor', 'synthetic-peer'], -1)
        return
    called = []
    with pytest.raises((ci.IdentityError, FileNotFoundError)):
        launcher.launch(tmp_path, 'never-executed', arguments, exec_fn=lambda *args: called.append(1))
    assert not called
    assert not list(capture.root('transcript').glob('*.log'))


def test_unowned_descriptor_and_changed_claim_refuse(tmp_path):
    capture, artifact, sink, rest = launcher.prepare(tmp_path, ['pactor', 'synthetic-peer'])
    arguments = ['--capture-id', capture.capture_id, '--capture-root', str(capture.root('capture')), *rest]
    try:
        foreign = tmp_path / 'foreign'
        with foreign.open('wb') as unrelated:
            with pytest.raises(ci.IdentityError):
                launcher.prepared(arguments, unrelated.fileno())
        claim = artifact.writer.root / '.capture-writers/launcher-transcript.json'
        row = json.loads(claim.read_text())
        row['writer_id'] = '00000000-0000-0000-0000-000000000000'
        claim.write_text(json.dumps(row))
        with pytest.raises(ci.IdentityError):
            launcher.prepared(arguments, sink.fileno())
    finally:
        sink.close()


def test_default_log_and_distinct_child_parent_relationship(tmp_path):
    group, parent, sink, _ = launcher.prepare(tmp_path, ['pounce', 'pactor', 'synthetic-peer'])
    sink.close()
    child_ids = []
    for _ in range(5):
        child, artifact, sink, _ = launcher.prepare(tmp_path, ['pactor', 'synthetic-peer',
            '--capture-parent-id', group.capture_id, '--capture-parent-kind', 'launch-group'])
        try:
            assert child.capture_id != group.capture_id
            assert child.record['parent_capture_id'] == group.capture_id
            assert child.kind == 'arm' and group.kind == 'launch-group'
            assert artifact.path != parent.path
            child_ids.append(child.capture_id)
        finally:
            sink.close()
    assert len(set(child_ids)) == 5


def test_other_mode_not_claimed_and_log_label_not_arbitrary_path(tmp_path):
    with pytest.raises(ci.IdentityError):
        launcher.prepare(tmp_path, ['vara', 'synthetic-peer'])
    with pytest.raises(ci.IdentityError):
        launcher.prepare(tmp_path, ['pactor', '--log', '../outside'])
    assert not (tmp_path / 'captures').exists()


@pytest.mark.parametrize('arguments', [
    ['--witness', 'pactor', 'synthetic'],
    ['--witness', 'pounce', 'pactor', 'synthetic'],
    ['pactor-answer', '--witness'],
])
def test_witness_global_flag_keeps_actual_pactor_preflight(tmp_path, arguments):
    assert launcher.pactor(arguments)
    capture, artifact, sink, rest = launcher.prepare(tmp_path, arguments)
    try:
        assert rest == arguments
        assert capture.kind == ('launch-group' if 'pounce' in arguments else 'arm')
        prepared, _, unchanged = launcher.prepared(
            ['--capture-id', capture.capture_id, '--capture-root', str(capture.root('capture')), *rest], sink.fileno())
        assert prepared.capture_id == capture.capture_id and unchanged == rest
    finally:
        sink.close()
    assert not launcher.pactor(['vara', 'pactor'])


def test_pounce_retained_journal_has_five_unique_ordered_children(tmp_path, monkeypatch):
    children = []
    def group_exec(script, words, environment):
        parent, transcript, _ = launcher.prepared(words[2:], environment['HFMODEM_CAPTURE_FD'])
        for key in ('HFMODEM_CAPTURE_GROUP_FD', 'HFMODEM_CAPTURE_GROUP_ROOT', 'HFMODEM_CAPTURE_GROUP_ID'):
            monkeypatch.setenv(key, environment[key])
        for index in range(1, 6):
            def child_exec(script, words, child_environment):
                child, path, rest = launcher.prepared(words[2:], child_environment['HFMODEM_CAPTURE_FD'])
                assert child.kind == 'arm'
                assert child.record['parent_capture_id'] == parent.capture_id
                assert child.record['parent_kind'] == 'launch-group'
                os.write(int(child_environment['HFMODEM_CAPTURE_FD']), b'synthetic child transcript\n')
                # Existing inherited stdout pipe forwards these child lines.
                os.write(int(environment['HFMODEM_CAPTURE_FD']), b'synthetic forwarded line\n')
                children.append((child, path))
            launcher.launch(tmp_path, '/fake-onair', ['--witness', 'pactor', 'synthetic',
                '--capture-parent-id', parent.capture_id, '--capture-parent-kind', 'launch-group'], exec_fn=child_exec)
            lines = (parent.root('transcript') / 'children.jsonl').read_text().splitlines()
            assert len(lines) == index
        rows = [json.loads(line) for line in lines]
        assert [r['sequence'] for r in rows] == list(range(1, 6))
        assert len({r['child_capture_id'] for r in rows}) == 5
        assert all(r['registry_sha256'] == parent.digest for r in rows)
        assert [r['child_capture_id'] for r in rows] == [c.capture_id for c, _ in children]
        assert len({path for _, path in children}) == 5
        assert all(path != transcript for _, path in children)
        assert transcript.read_bytes().count(b'synthetic forwarded line') == 5
        extra = ci.allocate_arm(tmp_path, parent_capture_id=parent.capture_id, parent_kind='launch-group')
        with pytest.raises(ci.IdentityError):
            launcher.child_registered(parent, extra, environment['HFMODEM_CAPTURE_GROUP_FD'])
        assert len((parent.root('transcript') / 'children.jsonl').read_text().splitlines()) == 5
        assert not parent.lifecycle()['capture_complete']
    launcher.launch(tmp_path, '/fake-onair', ['pounce', 'pactor', 'synthetic'], exec_fn=group_exec)
    assert len(children) == 5


def test_refused_child_launch_is_listed_with_partial_evidence(tmp_path, monkeypatch):
    original = OSError('synthetic child exec refusal')
    def group_exec(script, words, environment):
        parent, _, _ = launcher.prepared(words[2:], environment['HFMODEM_CAPTURE_FD'])
        for key in ('HFMODEM_CAPTURE_GROUP_FD', 'HFMODEM_CAPTURE_GROUP_ROOT', 'HFMODEM_CAPTURE_GROUP_ID'):
            monkeypatch.setenv(key, environment[key])
        def refuse(*args):
            raise original
        with pytest.raises(OSError) as caught:
            launcher.launch(tmp_path, '/fake-onair', ['pactor', 'synthetic',
                '--capture-parent-id', parent.capture_id, '--capture-parent-kind', 'launch-group'], exec_fn=refuse)
        assert caught.value is original
        rows = [json.loads(line) for line in (parent.root('transcript') / 'children.jsonl').read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['sequence'] == 1
        child = ci.load(rows[0]['child_capture_root'])
        assert not child.lifecycle()['capture_complete']
        assert child.record['parent_capture_id'] == parent.capture_id
        assert child.ended()
        assert child.lifecycle()['writers']['launcher-transcript']['state'] == 'partial'
    launcher.launch(tmp_path, '/fake-onair', ['pounce', 'pactor', 'synthetic'], exec_fn=group_exec)


def test_prepared_readonly_descriptor_and_ended_launch_refuse(tmp_path):
    capture, artifact, sink, rest = launcher.prepare(tmp_path, ['pactor', 'synthetic'])
    args = ['--capture-id', capture.capture_id, '--capture-root', str(capture.root('capture')), *rest]
    try:
        with artifact.path.open('rb') as reader:
            with pytest.raises(ci.IdentityError):
                launcher.prepared(args, reader.fileno())
        capture.end()
        with pytest.raises(ci.IdentityError):
            launcher.prepared(args, sink.fileno())
    finally:
        sink.close()


def test_retained_writer_preserves_status_and_refuses_foreign_descriptor(tmp_path):
    capture, artifact, sink, _ = launcher.prepare(tmp_path, ['pactor', 'synthetic'])
    writer = artifact.writer
    writer.update_status(state='ongoing', pending=2, component='original')
    try:
        continued = ci.retained_writer(capture, writer.role, artifact.record['artifact_path'], sink.fileno())
        continued.update_status(state='finalized', written=1, component='continuation')
        row = capture.lifecycle()['writers'][writer.role]
        assert row['state'] == 'ongoing' and row['pending'] == 2
        assert row['written'] == 1
        with artifact.path.open('rb') as reader:
            with pytest.raises(ci.IdentityError):
                ci.retained_writer(capture, writer.role, artifact.record['artifact_path'], reader.fileno())
        unrelated = tmp_path / 'foreign.log'
        with unrelated.open('xb') as foreign:
            with pytest.raises(ci.IdentityError):
                ci.retained_writer(capture, writer.role, artifact.record['artifact_path'], foreign.fileno())
        capture.end()
        continued = ci.retained_writer(capture, writer.role, artifact.record['artifact_path'], sink.fileno())
        continued.update_status(state='finalized', written=2, component='original')
        assert capture.lifecycle()['writers'][writer.role]['state'] == 'finalized'
        with pytest.raises(ci.IdentityError):
            capture.claim(writer.role)
    finally:
        sink.close()


@pytest.mark.parametrize('wrong', ['metadata_version', 'previous_sha256'])
def test_retained_status_tamper_refuses_before_continuation(tmp_path, wrong):
    capture, artifact, sink, _ = launcher.prepare(tmp_path, ['pactor', 'synthetic'])
    try:
        status = artifact.writer.root / '.capture-writer-status' / (artifact.writer.role + '.json')
        row = json.loads(status.read_text())
        row[wrong] = 20 if wrong == 'metadata_version' else 'synthetic-foreign-hash'
        status.write_text(json.dumps(row))
        with pytest.raises(ci.IdentityError):
            ci.retained_writer(capture, artifact.writer.role, artifact.record['artifact_path'], sink.fileno())
    finally:
        sink.close()


def test_busy_parent_journal_refuses_child_before_fake_exec(tmp_path, monkeypatch):
    import fcntl
    def group_exec(script, words, environment):
        parent, _, _ = launcher.prepared(words[2:], environment['HFMODEM_CAPTURE_FD'])
        for key in ('HFMODEM_CAPTURE_GROUP_FD', 'HFMODEM_CAPTURE_GROUP_ROOT', 'HFMODEM_CAPTURE_GROUP_ID'):
            monkeypatch.setenv(key, environment[key])
        def busy(*args):
            raise BlockingIOError('synthetic concurrent registration')
        monkeypatch.setattr(fcntl, 'lockf', busy)
        attempted = []
        with pytest.raises(BlockingIOError):
            launcher.launch(tmp_path, '/fake-onair', ['pactor', 'synthetic',
                '--capture-parent-id', parent.capture_id, '--capture-parent-kind', 'launch-group'], exec_fn=lambda *a: attempted.append(1))
        assert not attempted
        assert not (parent.root('transcript') / 'children.jsonl').read_bytes()
    launcher.launch(tmp_path, '/fake-onair', ['pounce', 'pactor', 'synthetic'], exec_fn=group_exec)


def test_shell_pounce_keeps_exact_attempt_bound_and_parent_flags():
    source = (ROOT / 'tools/onair.sh').read_text()
    branch = source.rindex('\npounce)')
    pounce = source[branch:source.index('\nlisten)', branch)]
    assert 'for _ in 1 2 3 4 5; do' in pounce
    assert '--capture-parent-id "$CAPTURE_ID" --capture-parent-kind launch-group' in pounce
    assert '[ "$rc" -eq 3 ]' in pounce
    assert '[ "$rc" -eq "$CHANNEL_BUSY" ] || exit "$rc"' in pounce
    preflight = source.index('classify --repo')
    assert preflight < source.index('witness_start()') < branch


@pytest.mark.parametrize("returncode", [0, 3, 17])
def test_launcher_end_and_later_transcript_eof_keep_independent_completion(tmp_path, returncode):
    seen = []
    def fake_exec(script, words, environment):
        cap, _, _ = launcher.prepared(words[2:], environment["HFMODEM_CAPTURE_FD"])
        selected = ["--capture-id", cap.capture_id, "--capture-root", str(cap.root("capture"))]
        cap.end()  # Native completion may precede the launcher.
        before = (cap.root("capture") / ".capture-ended.json").read_bytes()
        launcher.end_launch(selected, environment["HFMODEM_CAPTURE_METADATA_FD"], returncode=returncode)
        assert (cap.root("capture") / ".capture-ended.json").read_bytes() == before
        state = cap.lifecycle()
        assert state["writers"]["launcher-metadata"]["state"] == "finalized"
        assert state["writers"]["launcher-transcript"]["state"] == "ongoing"
        os.write(int(environment["HFMODEM_CAPTURE_FD"]), b"fake evidence\n")
        launcher.complete_transcript(selected, environment["HFMODEM_CAPTURE_FD"])
        assert cap.lifecycle()["writers"]["launcher-transcript"]["state"] == "finalized"
        rows = [json.loads(line) for line in (cap.root("transcript") / "launcher-events.jsonl").read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["event"] == "launcher-exit"
        assert rows[0]["returncode"] == returncode and rows[0]["capture_id"] == cap.capture_id
        assert not cap.lifecycle()["capture_complete"]  # Unlaunched delegated roles remain explicit.
        seen.append(True)
    launcher.launch(tmp_path, "/fake/onair.sh", ["pactor", "peer"], exec_fn=fake_exec)
    assert seen == [True]


def test_transcript_failure_remains_partial_after_launcher_ending(tmp_path):
    def fake_exec(script, words, environment):
        cap, _, _ = launcher.prepared(words[2:], environment["HFMODEM_CAPTURE_FD"])
        selected = ["--capture-id", cap.capture_id, "--capture-root", str(cap.root("capture"))]
        launcher.complete_transcript(selected, environment["HFMODEM_CAPTURE_FD"], outcome="partial")
        launcher.end_launch(selected, environment["HFMODEM_CAPTURE_METADATA_FD"], returncode=4)
        assert cap.lifecycle()["writers"]["launcher-transcript"]["state"] == "partial"
        assert not cap.lifecycle()["capture_complete"]
    launcher.launch(tmp_path, "/fake/onair.sh", ["pactor", "peer"], exec_fn=fake_exec)


def test_end_foreign_anchor_refuses_and_duplicate_exit_has_one_row(tmp_path):
    def fake_exec(script, words, environment):
        cap, _, _ = launcher.prepared(words[2:], environment["HFMODEM_CAPTURE_FD"])
        selected = ["--capture-id", cap.capture_id, "--capture-root", str(cap.root("capture"))]
        with pytest.raises(ci.IdentityError):
            launcher.end_launch(selected, environment["HFMODEM_CAPTURE_FD"], returncode=0)
        assert not cap.ended()
        launcher.end_launch(selected, environment["HFMODEM_CAPTURE_METADATA_FD"], returncode=0)
        with pytest.raises(ci.IdentityError):
            launcher.end_launch(selected, environment["HFMODEM_CAPTURE_METADATA_FD"], returncode=0)
        assert len((cap.root("transcript") / "launcher-events.jsonl").read_text().splitlines()) == 1
    launcher.launch(tmp_path, "/fake/onair.sh", ["pactor", "peer"], exec_fn=fake_exec)


def test_group_end_finalizes_children_journal_but_waits_for_transcript(tmp_path):
    def fake_exec(script, words, environment):
        cap, _, _ = launcher.prepared(words[2:], environment["HFMODEM_CAPTURE_FD"])
        selected = ["--capture-id", cap.capture_id, "--capture-root", str(cap.root("capture"))]
        launcher.end_launch(selected, environment["HFMODEM_CAPTURE_METADATA_FD"], returncode=0)
        status = cap.lifecycle()["writers"]["launcher-metadata"]
        assert status["components"]["children"]["state"] == "finalized"
        assert status["state"] == "finalized"
        assert not cap.lifecycle()["capture_complete"]
        launcher.complete_transcript(selected, environment["HFMODEM_CAPTURE_FD"])
        assert cap.lifecycle()["capture_complete"]
    launcher.launch(tmp_path, "/fake/onair.sh", ["pounce", "pactor", "peer"], exec_fn=fake_exec)


@pytest.mark.parametrize("returncode", [-1, 256, True, None])
def test_invalid_launcher_exit_refuses_without_side_effect(tmp_path, returncode):
    def fake_exec(script, words, environment):
        cap, _, _ = launcher.prepared(words[2:], environment["HFMODEM_CAPTURE_FD"])
        selected = ["--capture-id", cap.capture_id, "--capture-root", str(cap.root("capture"))]
        with pytest.raises(ci.IdentityError):
            launcher.end_launch(selected, environment["HFMODEM_CAPTURE_METADATA_FD"], returncode=returncode)
        assert not cap.ended()
        assert (cap.root("transcript") / "launcher-events.jsonl").stat().st_size == 0
    launcher.launch(tmp_path, "/fake/onair.sh", ["pactor", "peer"], exec_fn=fake_exec)


def test_shell_owned_exit_and_eof_keep_legacy_cleanup_and_exit_status():
    source = (ROOT / "tools/onair.sh").read_text()
    assert "capture_rc=${1:-$?}" in source
    assert "capture_trap_rc=$?;" in source
    # rigctld setup, PACTOR, and ARDOP's child and post-child EXIT handlers.
    assert source.count('capture_exit "$capture_trap_rc"') == 4
    assert source.count('capture_transcript_finish "$capture_tee_rc"') == 2
    assert "--returncode" in source and "--outcome" in source
    assert "for _ in $(seq 20)" in source



def test_witness_stdout_is_reserved_and_bound_before_fake_producer(tmp_path):
    def fake_exec(script, words, environment):
        cap, _, _ = launcher.prepared(words[2:], environment["HFMODEM_CAPTURE_FD"])
        descriptor = int(environment["HFMODEM_CAPTURE_WITNESS_LOG_FD"])
        path = launcher.prepared_witness_log(cap, descriptor)
        assert path == cap.root("transcript") / "witness.log"
        assert os.get_inheritable(descriptor)
        os.write(descriptor, b"fake witness metadata\n")
        assert path.read_bytes() == b"fake witness metadata\n"
        binding = ci.artifact_binding(path)
        assert binding["writer_role"] == "launcher-metadata"
        assert binding["capture_id"] == cap.capture_id
        selected = ["--capture-id", cap.capture_id, "--capture-root", str(cap.root("capture"))]
        launcher.end_launch(selected, environment["HFMODEM_CAPTURE_METADATA_FD"], returncode=0)
        assert cap.lifecycle()["writers"]["launcher-metadata"]["components"]["witness-output"]["state"] == "finalized"
        with pytest.raises(ci.IdentityError):
            launcher.prepared_witness_log(cap, descriptor)
    launcher.launch(tmp_path, "/fake/onair.sh", ["--witness", "pactor", "peer"], exec_fn=fake_exec)


def test_witness_stdout_foreign_and_readonly_descriptors_refuse(tmp_path):
    def fake_exec(script, words, environment):
        cap, _, _ = launcher.prepared(words[2:], environment["HFMODEM_CAPTURE_FD"])
        with pytest.raises(ci.IdentityError):
            launcher.prepared_witness_log(cap, environment["HFMODEM_CAPTURE_FD"])
        path = launcher.prepared_witness_log(cap, environment["HFMODEM_CAPTURE_WITNESS_LOG_FD"])
        with path.open("rb") as readonly:
            with pytest.raises(ci.IdentityError):
                launcher.prepared_witness_log(cap, readonly.fileno())
        assert not cap.ended() and path.stat().st_size == 0
    launcher.launch(tmp_path, "/fake/onair.sh", ["pactor", "peer"], exec_fn=fake_exec)


def test_shell_witness_stdout_uses_original_descriptor_without_replacing_cleanup():
    source = (ROOT / "tools/onair.sh").read_text()
    branch = source[source.index('witness_start()'):source.index('witness_stop() {')]
    assert 'witness_log_target="/dev/fd/$HFMODEM_CAPTURE_WITNESS_LOG_FD"' in branch
    assert '>"$witness_log_target" 2>&1 &' in branch
    assert '>"$SESSION/witness.log" 2>&1 &' not in branch
    assert 'capture_exit "$capture_trap_rc"' in branch
