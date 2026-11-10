# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One bounded offline replay worker for the local logbook."""
from collections import deque
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid


def roots(book):
    return [a for r in book.roots for a in ('--root', str(r))]


class Replays:
    def __init__(self, book, script=None, run=None):
        self.book = book
        self.script = script or Path(__file__).with_name('vara_view.py')
        self.run = run
        self.lock = threading.Lock()
        self.jobs = {}
        self.queue = deque()
        self.worker = None

    def start(self, arm):
        with self.lock:
            entries = self.book.entries()['entries']
            entry = next((e for e in entries if e['id'] == arm), None)
            selection = deepcopy((entry or {}).get('recording'))
            for job in self.jobs.values():
                if job['arm'] == arm and job['state'] in ('queued', 'running') and job.get('selection') == selection:
                    return dict(job)
            if sum(j['state'] in ('queued', 'running') for j in self.jobs.values()) >= 8:
                raise ValueError('Eight replays are already queued; wait for one to finish.')
            if not entry or entry['protocol'] not in ('vara','pactor','ardop'):
                raise ValueError('Choose a VARA, PACTOR or ARDOP QSO in this logbook.')
            available = [r for r in entry.get('audio', []) if r.get('url')
                         and r['kind'] in ('rig rx', 'rig tap', 'sdr witness')]
            if not available:
                raise ValueError('This QSO has no recording to rehear.')
            while len(self.jobs) >= 32:
                old = next((k for k, j in self.jobs.items() if j['state'] in ('complete', 'failed')), None)
                if old is None:
                    break
                del self.jobs[old]
            key = uuid.uuid4().hex
            job = {'id': key, 'arm': arm, 'state': 'queued', 'output': '', 'started': None}
            if selection:
                from hfmodem.station import sessionlog
                source = available[0]
                path = Path(self.book.roots[0]) / source['url'].removeprefix('/file/')
                sessionlog.select(selection, path, root=self.book.roots[0])
                job.update(sessionlog.identity(selection))
                job['selection'] = selection
            self.jobs[key] = job
            self.queue.append(key)
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._work, daemon=True)
                self.worker.start()
            return dict(job)

    def status(self, key):
        with self.lock:
            return dict(self.jobs[key]) if key in self.jobs else None

    def _run_streamed(self, command, key):
        """Keep a bounded progress tail visible while the offline decoder runs."""
        process = subprocess.Popen(command, cwd=self.script.parent, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        def read():
            for line in process.stdout:
                with self.lock:
                    job = self.jobs[key]
                    job['output'] = (job['output'] + line)[-6000:]
                    if line.strip():
                        job['progress'] = line.strip()[:240]
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            reader.join()
            raise
        finally:
            if process.poll() is not None:
                reader.join()
                process.stdout.close()
        with self.lock:
            output = self.jobs[key]['output']
        return subprocess.CompletedProcess(command, code, output, '')

    def _work(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.worker = None
                    return
                key = self.queue.popleft()
                job = self.jobs[key]
                job.update(state='running', started=time.time())
                arm = job['arm']
            command = [sys.executable, str(self.script), *roots(self.book),
                       '--arm', arm, '--out', str(self.book.viewer_root)]
            try:
                with tempfile.TemporaryDirectory(prefix='replay-selection-') as scratch:
                    if job.get('selection'):
                        payload = Path(scratch) / 'selection.json'
                        payload.write_text(json.dumps(job['selection']))
                        command += ['--selection', str(payload)]
                    result = (self.run(command, cwd=self.script.parent, capture_output=True,
                                   text=True, timeout=1800) if self.run is not None
                          else self._run_streamed(command, key))
                state = 'complete' if result.returncode == 0 else 'failed'
                output = (result.stdout + '\n' + result.stderr)[-6000:]
            except subprocess.TimeoutExpired:
                state, output = 'failed', 'Replay exceeded the 30-minute processing limit.'
            except OSError as exc:
                state, output = 'failed', str(exc)
            with self.lock:
                self.jobs[key].update(state=state, output=output)
