#!/usr/bin/env python3
"""Pure PACTOR evidence preflight and retained transcript descriptor handoff.

The CLI itself contains no device, rig, audio, network or modem imports. Tests
replace exec with a fake; a real launcher is never run by the offline suite.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packages/hfmodem'))
from hfmodem.core import capture_identity as ci


def options(arguments):
    """Consume only launcher identity/label flags, preserving all other words."""
    selected, rest = {}, []
    names = {'--log': 'log', '--capture-id': 'capture_id', '--capture-root': 'capture_root',
             '--capture-parent-id': 'parent_id', '--capture-parent-kind': 'parent_kind'}
    at = 0
    while at < len(arguments):
        word = arguments[at]
        head, separator, value = word.partition('=')
        if head not in names:
            rest.append(word)
            at += 1
            continue
        if not separator:
            at += 1
            if at == len(arguments) or arguments[at].startswith('--'):
                raise ci.IdentityError('Missing launcher identity argument')
            value = arguments[at]
        key = names[head]
        if key != 'log' and key in selected and selected[key] != value:
            raise ci.IdentityError('Conflicting launcher identity arguments')
        selected[key] = value
        at += 1
    if bool(selected.get('capture_id')) != bool(selected.get('capture_root')):
        raise ci.IdentityError('Delegation requires both capture identity and registered root')
    return selected, rest


def verb_arguments(arguments):
    # onair.sh consumes this flag globally before selecting the verb.
    return [word for word in arguments if word != '--witness']


def pactor(arguments):
    words = verb_arguments(arguments)
    return bool(words) and (words[0] in ('pactor', 'pactor-answer')
                            or words[:2] == ['pounce', 'pactor'])


def child_registered(parent, child, descriptor):
    """Append one serial attempted child to the retained parent journal."""
    if (parent.kind != 'launch-group' or child.kind != 'arm'
            or child.record.get('parent_capture_id') != parent.capture_id
            or child.record.get('parent_kind') != 'launch-group'
            or parent.ended()):
        raise ci.IdentityError('launch-group child binding mismatch')
    writer = ci.retained_writer(parent, 'launcher-metadata', 'children.jsonl', descriptor)
    path = writer.root / 'children.jsonl'
    # Process-scoped nonblocking record lock: concurrent registration refuses
    # before a child exec rather than losing an ordinal or overwriting evidence.
    fcntl.lockf(int(descriptor), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        existing = [json.loads(line) for line in path.read_text().splitlines()]
        if len(existing) >= 5:
            raise ci.IdentityError('launch-group child ceiling reached')
        for index, row in enumerate(existing, 1):
            if (row.get('capture_id') != parent.capture_id
                    or row.get('registry_sha256') != parent.digest
                    or row.get('sequence') != index):
                raise ci.IdentityError('launch-group journal binding mismatch')
        if any(row.get('child_capture_id') == child.capture_id for row in existing):
            raise ci.IdentityError('launch-group child already registered')
        row = dict(capture_id=parent.capture_id, registry_sha256=parent.digest,
                   writer_id=writer.writer_id, writer_role=writer.role,
                   sequence=len(existing) + 1, child_capture_id=child.capture_id,
                   child_registry_sha256=child.digest,
                   child_capture_root=str(child.root('capture')),
                   child_kind=child.kind)
        raw = (json.dumps(row, sort_keys=True) + '\n').encode()
        os.lseek(int(descriptor), 0, os.SEEK_END)
        if os.write(int(descriptor), raw) != len(raw):
            raise OSError('partial launch-group journal write')
        return row

    finally:
        fcntl.lockf(int(descriptor), fcntl.LOCK_UN)


def _selected_capture(arguments):
    selected, rest = options(list(arguments))
    if not selected.get('capture_id') or rest:
        raise ci.IdentityError('explicit capture completion binding required')
    capture = ci.load(selected['capture_root'])
    if capture.capture_id != selected['capture_id']:
        raise ci.IdentityError('capture completion identity mismatch')
    return capture


def complete_transcript(arguments, descriptor, *, outcome='finalized'):
    if outcome not in ('finalized', 'partial'):
        raise ci.IdentityError('invalid transcript completion outcome')
    capture = _selected_capture(arguments)
    transcript = capture.record['transcript']
    writer = ci.retained_writer(capture, 'launcher-transcript', transcript['path'], descriptor)
    writer.update_status(state=outcome, written=1, partial=int(outcome == 'partial'),
                         component='transcript')


def prepared_witness_log(capture, descriptor):
    if capture.ended() or capture.kind != 'arm':
        raise ci.IdentityError('witness producer does not own an active arm')
    writer = ci.retained_writer(capture, 'launcher-metadata', 'witness.log', descriptor)
    return writer.root / 'witness.log'


def end_launch(arguments, descriptor, *, returncode):
    if (not isinstance(returncode, int) or isinstance(returncode, bool)
            or not 0 <= returncode <= 255):
        raise ci.IdentityError('invalid launcher exit status')
    capture = _selected_capture(arguments)
    writer = ci.retained_writer(capture, 'launcher-metadata', 'launcher-events.jsonl', descriptor)
    row = dict(capture_id=capture.capture_id, registry_sha256=capture.digest,
               writer_id=writer.writer_id, writer_role=writer.role,
               event='launcher-exit', returncode=returncode)
    try:
        # Shared native driver may have already ended this same registered arm.
        # Its first ending clock is retained; this is a producer completion record.
        capture.end()
        existing = [json.loads(line) for line in (writer.root / 'launcher-events.jsonl').read_text().splitlines()]
        if any(r.get('event') == 'launcher-exit' for r in existing):
            raise ci.IdentityError('launcher already completed')
        raw = (json.dumps(row, sort_keys=True) + '\n').encode()
        os.lseek(int(descriptor), 0, os.SEEK_END)
        if os.write(int(descriptor), raw) != len(raw):
            raise OSError('partial launcher ending write')
        if capture.kind == 'launch-group':
            writer.update_status(state='finalized', written=1, component='children')
        else:
            writer.update_status(state='finalized', written=1, component='witness-output')
        writer.update_status(state='finalized', written=1, component='launcher')
    except BaseException:
        writer._status['failed'] = True
        try:
            writer.update_status(state='partial', partial=1, component='launcher')
        except BaseException as trouble:
            writer._status['journal_error'] = type(trouble).__name__
        raise


def prepare(repo, arguments):
    selected, rest = options(list(arguments))
    if not pactor(rest):
        raise ci.IdentityError('Capture launcher delegation is scoped to PACTOR')
    kind = 'launch-group' if verb_arguments(rest)[0] == 'pounce' else 'arm'
    if selected.get('capture_id'):
        capture = ci.load(selected['capture_root'])
        if capture.capture_id != selected['capture_id'] or capture.kind != kind:
            raise ci.IdentityError('Launcher delegation identity or kind differs')
        ci.effective_outdir(rest, capture.root('capture'))
    else:
        capture = ci.allocate_arm(repo, transcripts=Path(repo) / 'working',
                                  log_label=selected.get('log'), kind=kind,
                                  arguments=rest, source={'unknown': True},
                                  parent_capture_id=selected.get('parent_id'),
                                  parent_kind=selected.get('parent_kind'))
    transcript = capture.record['transcript']
    writer = capture.claim('launcher-transcript')
    if writer.root != capture.root(transcript['root_role']):
        raise ci.IdentityError('Transcript writer has the wrong registered root')
    artifact = writer.reserve(transcript['path'])
    sink = artifact.open()
    os.set_inheritable(sink.fileno(), True)
    return capture, artifact, sink, rest


def launch(repo, script, arguments, *, exec_fn=os.execvpe):
    capture, artifact, sink, rest = prepare(repo, arguments)
    environment = dict(os.environ)
    environment['HFMODEM_CAPTURE_FD'] = str(sink.fileno())
    journal = None
    metadata_sink = None
    witness_sink = None
    metadata_writer = None
    try:
        metadata_writer = capture.claim('launcher-metadata')
        metadata_sink = metadata_writer.reserve('launcher-events.jsonl').open()
        os.set_inheritable(metadata_sink.fileno(), True)
        environment['HFMODEM_CAPTURE_METADATA_FD'] = str(metadata_sink.fileno())
        metadata_writer.update_status(state='ongoing', pending=1, component='launcher')
        if capture.kind == 'arm':
            witness_sink = metadata_writer.reserve('witness.log').open()
            os.set_inheritable(witness_sink.fileno(), True)
            environment['HFMODEM_CAPTURE_WITNESS_LOG_FD'] = str(witness_sink.fileno())
            metadata_writer.update_status(state='ongoing', pending=1, component='witness-output')
        if capture.kind == 'launch-group':
            writer = metadata_writer
            journal = writer.reserve('children.jsonl').open()
            os.set_inheritable(journal.fileno(), True)
            writer.update_status(state='ongoing', pending=1, component='children')
            environment.update(HFMODEM_CAPTURE_GROUP_FD=str(journal.fileno()),
                               HFMODEM_CAPTURE_GROUP_ROOT=str(capture.root('capture')),
                               HFMODEM_CAPTURE_GROUP_ID=capture.capture_id)
        elif capture.record.get('parent_capture_id') is not None:
            if not all(environment.get(k) for k in ('HFMODEM_CAPTURE_GROUP_ROOT', 'HFMODEM_CAPTURE_GROUP_ID', 'HFMODEM_CAPTURE_GROUP_FD')):
                raise ci.IdentityError('missing launch-group retained binding')
            parent = ci.load(environment['HFMODEM_CAPTURE_GROUP_ROOT'])
            if parent.capture_id != environment.get('HFMODEM_CAPTURE_GROUP_ID'):
                raise ci.IdentityError('launch-group environment mismatch')
            child_registered(parent, capture, environment['HFMODEM_CAPTURE_GROUP_FD'])
        words = [str(script), '--capture-prepared', '--capture-id', capture.capture_id,
                 '--capture-root', str(capture.root('capture')), *rest]
        exec_fn(str(script), words, environment)
    except BaseException:
        for writer in (artifact.writer, metadata_writer):
            if writer is None:
                continue
            writer._status['failed'] = True
            try:
                writer.update_status(state='partial', partial=1, component='launch')
            except BaseException as trouble:
                writer._status['journal_error'] = type(trouble).__name__
        try:
            capture.end()
        except BaseException as trouble:
            artifact.writer._status['ending_error'] = type(trouble).__name__
        raise
    finally:
        if journal is not None:
            journal.close()
        if witness_sink is not None:
            witness_sink.close()
        if metadata_sink is not None:
            metadata_sink.close()
        sink.close()


def prepared(arguments, descriptor):
    selected, rest = options(arguments)
    if not selected.get('capture_id') or not pactor(rest):
        raise ci.IdentityError('Missing explicit prepared launcher delegation')
    capture = ci.load(selected['capture_root'])
    if capture.capture_id != selected['capture_id']:
        raise ci.IdentityError('Prepared launcher identity differs')
    ci.effective_outdir(rest, capture.root('capture'))
    if capture.ended() or capture.kind != ('launch-group' if verb_arguments(rest)[0] == 'pounce' else 'arm'):
        raise ci.IdentityError('prepared launch lifetime/kind mismatch')
    if fcntl.fcntl(int(descriptor), fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY:
        raise ci.IdentityError('prepared transcript descriptor is read-only')
    transcript = capture.record['transcript']
    root = capture.root(transcript['root_role'])
    path = root / ci.relative(transcript['path'])
    actual = os.fstat(int(descriptor))
    expected = path.lstat()
    if (not stat.S_ISREG(actual.st_mode) or not stat.S_ISREG(expected.st_mode)
            or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)):
        raise ci.IdentityError('Prepared descriptor is not the registered transcript')
    claim = json.loads((root / '.capture-writers/launcher-transcript.json').read_text())
    binding = capture.record['writers']['launcher-transcript']
    if (claim['capture_id'] != capture.capture_id
            or claim['writer_id'] != binding['writer_id']
            or claim['registry_sha256'] != capture.digest):
        raise ci.IdentityError('Prepared transcript writer differs')
    return capture, path, rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'prepared', 'classify', 'ended', 'transcript-complete'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--script', type=Path)
    parser.add_argument('--returncode', type=int)
    parser.add_argument('--outcome', choices=['finalized', 'partial'], default='finalized')
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    if args.action == 'ended':
        end_launch(remaining, os.environ.get('HFMODEM_CAPTURE_METADATA_FD', '-1'), returncode=args.returncode)
        return
    if args.action == 'transcript-complete':
        complete_transcript(remaining, os.environ.get('HFMODEM_CAPTURE_FD', '-1'), outcome=args.outcome)
        return
    if args.action == 'classify':
        raise SystemExit(0 if pactor(options(remaining)[1]) else 1)
    if args.action == 'launch':
        if args.script is None:
            parser.error('launch requires an explicit script')
        launch(args.repo, args.script, remaining)
        return
    capture, transcript, _ = prepared(remaining, os.environ.get('HFMODEM_CAPTURE_FD', '-1'))
    ci.retained_writer(capture, 'launcher-metadata', 'launcher-events.jsonl',
                       os.environ.get('HFMODEM_CAPTURE_METADATA_FD', '-1'))
    if capture.kind == 'arm':
        prepared_witness_log(capture, os.environ.get('HFMODEM_CAPTURE_WITNESS_LOG_FD', '-1'))
    # Fixed assignment names and shell quoting; no arbitrary arguments, modem
    # fields or transcript content enter this metadata handoff.
    values = {'SESSION': str(capture.root('capture')), 'ARM_LOG': str(transcript),
              'CAPTURE_ID': capture.capture_id, 'CAPTURE_ROOT': str(capture.root('capture')),
              'CAPTURE_KIND': capture.kind}
    for key, value in values.items():
        print(key + '=' + shlex.quote(value))


if __name__ == '__main__':
    main()
