# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from contextlib import contextmanager
from datetime import datetime
import json
import io
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from hfmodem.core import capture_identity as ci
from hfmodem.core.claim import ClaimError
from hfmodem.core.recording import control_launch

from .launch_io import ENCODING, ERRORS
ROOT = Path(__file__).resolve().parents[2]


class Evidence:
    def __init__(self, arguments):
        import capture_identity as helper
        self.children = None
        if arguments[:1] == ['--capture-prepared']:
            descriptor = int(os.environ['HFMODEM_CAPTURE_FD'])
            self.capture, path, self.words = helper.prepared(arguments[1:], descriptor)
            self.artifact = SimpleNamespace(path=path, writer=ci.retained_writer(self.capture,
                'launcher-transcript', str(path.relative_to(self.capture.root('transcript'))), descriptor))
            self.metadata = ci.retained_writer(self.capture, 'launcher-metadata', 'launcher-events.jsonl',
                                              os.environ['HFMODEM_CAPTURE_METADATA_FD'])
            self.sink, self.events = (os.fdopen(os.dup(int(fd)), 'wb') for fd in
                (descriptor, os.environ['HFMODEM_CAPTURE_METADATA_FD']))
            self.witness_log = self.witness_path = None
            if self.capture.kind == 'launch-group':
                self.children = os.fdopen(os.dup(int(os.environ['HFMODEM_CAPTURE_GROUP_FD'])), 'wb')
            else:
                self.witness_path = helper.prepared_witness_log(self.capture, os.environ['HFMODEM_CAPTURE_WITNESS_LOG_FD'])
                self.witness_log = os.fdopen(os.dup(int(os.environ['HFMODEM_CAPTURE_WITNESS_LOG_FD'])), 'wb')
            self.session = self.capture.root('capture')
            self.binding = ['--capture-id', self.capture.capture_id, '--capture-root', str(self.session)]
            self.modem_args = [*self.binding, '--capture-role', 'modem']
            return
        data = Path(os.environ.get('HFMODEM_DATA_DIR', ROOT / 'operator-data'))
        self.capture, self.artifact, self.sink, self.words = helper.prepare(data, arguments)
        self.session = self.capture.root('capture')
        self.binding = ['--capture-id', self.capture.capture_id, '--capture-root', str(self.session)]
        self.modem_args = [*self.binding, '--capture-role', 'modem']
        self.metadata = self.events = self.witness_log = self.witness_path = None
        try:
            self.metadata = self.capture.claim('launcher-metadata')
            self.events = self.metadata.reserve('launcher-events.jsonl').open()
            self.metadata.update_status(state='ongoing', pending=1, component='launcher')
            if self.capture.kind == 'launch-group':
                self.children = self.metadata.reserve('children.jsonl').open()
                self.metadata.update_status(state='ongoing', pending=1, component='children')
            else:
                self.witness_log = self.metadata.reserve('witness.log').open()
                self.witness_path = Path(self.witness_log.name)
                self.metadata.update_status(state='ongoing', pending=1, component='witness-output')
            if self.capture.record.get('parent_capture_id') is not None:
                parent = ci.load(os.environ['HFMODEM_CAPTURE_GROUP_ROOT'])
                if parent.capture_id != os.environ['HFMODEM_CAPTURE_GROUP_ID']:
                    raise ci.IdentityError('launch-group environment mismatch')
                from hfmodem.core import claim
                descriptor = claim.extra('capture-group') if os.name == 'nt' else os.environ['HFMODEM_CAPTURE_GROUP_FD']
                helper.child_registered(parent, self.capture, descriptor)
        except BaseException:
            for writer in (self.artifact.writer, self.metadata):
                if writer is not None:
                    writer.update_status(state='partial', partial=1, component='launch')
            self.finish(1)
            raise

    @contextmanager
    def transcript(self):
        from .launch import Transcript
        stdout, stderr = sys.stdout, sys.stderr
        output = io.TextIOWrapper(self.sink, encoding=ENCODING, errors=ERRORS, write_through=True)
        sys.stdout, sys.stderr = Transcript(stdout, output), Transcript(stderr, output)
        try:
            print(f'  transcript: {self.artifact.path}', flush=True)
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            output.detach()

    def finish(self, code):
        from .launch import CLAIM_CLOSE_S, unwind_bound
        with unwind_bound(CLAIM_CLOSE_S):
            self._finish(code)

    def _finish(self, code):
        try:
            self.capture.end()
            if self.metadata is not None:
                if self.events is not None:
                    self.events.write((json.dumps(dict(capture_id=self.capture.capture_id,
                        registry_sha256=self.capture.digest, writer_id=self.metadata.writer_id,
                        writer_role=self.metadata.role, event='launcher-exit', returncode=code),
                        sort_keys=True) + '\n').encode())
                    self.events.flush()
                self.metadata.update_status(state='finalized', written=1,
                    component='children' if self.capture.kind == 'launch-group' else 'witness-output')
                self.metadata.update_status(state='finalized', written=1, component='launcher')
            self.artifact.writer.update_status(state='finalized', written=1, component='transcript')
        finally:
            for stream in (self.events, self.witness_log, self.children, self.sink):
                if stream is not None:
                    stream.close()

    @contextmanager
    def witness(self, launcher, dial, peer, *, enabled):
        if not enabled:
            yield
            return
        if policy() == 'off':
            print(INTENTIONAL_ABSENCE, flush=True)
            try:
                yield
            finally:
                print(INTENTIONAL_ABSENCE, flush=True)
            return
        wav, log = self.session / 'witness.wav', self.witness_path
        data = Path(os.environ.get('HFMODEM_DATA_DIR', ROOT / 'operator-data'))
        argv = [sys.executable, str(ROOT / 'tools/kiwi_witness.py'), '--dial', f'{dial / 1000:.3f}',
                '--seconds', os.environ.get('WITNESS_MAX_S', '3600'), '--out', str(wav),
                '--candidates', os.environ.get('HFMODEM_WITNESS_CANDIDATES', str(data / 'kiwi-candidates.csv')),
                *self.binding, *(['--capture-role', 'launcher-witness'] if self.binding else [])]
        if peer:
            argv += ['--peer', peer]
        # The bash FIFO's two ends: the witness reads one, and each modem this
        # launcher starts is handed the other through HFMODEM_RECORDING_CONTROL_OUT_FDS.
        reader, writer = os.pipe()
        env, passed = control_launch({k: v for k, v in os.environ.items() if k != CONTROL_OUT}, read=reader)
        try:
            child = launcher.claim.spawn(launcher.descriptor, argv, role='recorder', env=env, **passed,
                                         stdout=self.witness_log, stderr=subprocess.STDOUT)
        except (ClaimError, OSError, subprocess.SubprocessError) as exc:
            child, failure = None, exc
        finally:
            os.close(reader)
        if child is None:
            os.close(writer)
            self.witness_log.write(f'WITNESS NONE: {failure}\n'.encode())
            self.witness_log.flush()
            print(no_witness(f'{log} says why'), flush=True)
            yield
            return
        print(f'  witness: kiwi_witness pid {child.pid} -> {wav} (log {log})', flush=True)
        previous = os.environ.get(CONTROL_OUT)
        os.environ[CONTROL_OUT] = ','.join(filter(None, (previous, str(writer))))
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(CONTROL_OUT, None)
            else:
                os.environ[CONTROL_OUT] = previous
            os.close(writer)
            report(launcher, child, self.session, wav, log,
                   getattr(self, 'witness_stream', self.session / 'stream.wav'))


INTENTIONAL_ABSENCE = '  witness: recording off; intentional absence'
CONTROL_OUT = 'HFMODEM_RECORDING_CONTROL_OUT_FDS'


def policy():
    return os.environ.get('HFMODEM_RECORDING') or 'everything'


def no_witness(why):
    # "!! NO WITNESS" is what the console's arm summary looks for.
    return (f'  !! NO WITNESS for this arm -- witness unavailable: capture failed; {why}. The arm\n'
            '   above stands on its own, and nothing in it was delayed or refused by this.')


def align(*words):
    from .launch import WITNESS_ALIGN_S, unwind_bound
    try:
        with unwind_bound(WITNESS_ALIGN_S):
            aligned = subprocess.run([sys.executable, str(ROOT / 'tools/witness_align.py'), *map(str, words)],
                capture_output=True, text=True, encoding=ENCODING, errors=ERRORS, timeout=WITNESS_ALIGN_S)
    except (OSError, subprocess.SubprocessError):
        return
    for line in aligned.stdout.splitlines():
        if line.startswith('WITNESS'):
            print('  ' + line, flush=True)


def report(launcher, child, session, wav, log, stream):
    from hfhost.recordings import available
    from .launch import CHILD_STOP_S, unwind_bound
    import witness_select
    try:
        with unwind_bound(CHILD_STOP_S):
            stopped = launcher.claim.end_child(child) is not False
    except (ClaimError, OSError):
        stopped = False
    try:
        text = log.read_text(encoding=ENCODING, errors=ERRORS)
    except OSError:
        text = ''
    for line in text.splitlines():
        if line.startswith('WITNESS'):
            print('  ' + line, flush=True)
    if not stopped or child.poll() != 0 or re.search(r'^WITNESS (SHORT|NONE)', text, re.M):
        print(no_witness(f'{log} says why'), flush=True)
        return
    if policy() == 'contacts':
        chosen = witness_select.selected(launcher.destination or session, wav)
        for local, remote, selection in chosen:
            align(local, remote, '--selection', json.dumps(selection))
        if chosen:
            return
    if available(wav):
        if available(stream):
            align(stream, wav)
        print(f'  witness: {wav}\n  the whole lag table is:\n'
              f'    {sys.executable} {ROOT}/tools/witness_align.py {stream} {wav}', flush=True)
    elif re.search(r'^WITNESS COVERAGE: everything; retained [1-9]', text, re.M):
        print(f'  witness unavailable here (possibly moved); continuous coverage indexed in {session}/witness',
              flush=True)
    elif re.search(r'^WITNESS COVERAGE: contact-only; retained [1-9]', text, re.M):
        print(f'  witness unavailable here (possibly moved); contact-only coverage indexed in {session}/witness',
              flush=True)
    elif re.search(r'^WITNESS COVERAGE: contact-only; retained 0 ', text, re.M):
        print(f'  witness: contact-only; no retained contact ({log})', flush=True)
    elif policy() == 'contacts':
        print(no_witness(f'missing completion in {log}'), flush=True)
    else:
        print(f'  witness unavailable here (possibly moved); {log} says why', flush=True)


class LegacyEvidence:
    def __init__(self):
        data = Path(os.environ.get('HFMODEM_DATA_DIR', ROOT / 'operator-data'))
        self.session = data / 'captures' / datetime.now().strftime('onair-%m%d-%H%M')
        self.binding = []

    @contextmanager
    def witness(self, launcher, dial, peer, *, enabled):
        if not enabled or policy() == 'off':
            with Evidence.witness(self, launcher, dial, peer, enabled=enabled):
                yield
            return
        self.session.mkdir(parents=True, exist_ok=True)
        self.witness_path = self.session / 'witness.log'
        with self.witness_path.open('wb') as self.witness_log:
            with Evidence.witness(self, launcher, dial, peer, enabled=True):
                yield
