# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

import pytest

from hfmodem.tests.core.fakeyaesu import FakeYaesu

from .observer import Inotify, LinuxObserver, MacOSObserver, Trace, TracedProcess, linux_run_start_floor
from .oracle import Evidence, read_events


TREE = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent


def pytest_configure(config):
    config.addinivalue_line('markers', 'proof: requires the validated tty0tty line surrogate')
    config.addinivalue_line('markers', 'noline: coexistence row without a line proof')


class Harness:
    def __init__(self, root):
        self.root = root
        # This context creates the UUID and launches new descendants. It never
        # adopts/execs a pre-existing process into the run's participant set.
        self.process_start_floor = linux_run_start_floor() if sys.platform == 'linux' else None
        self.run_id = uuid.uuid4().hex
        self.events_path = root / 'events.jsonl'
        self.events_path.touch()
        self.barriers = root / 'barriers'
        self.barriers.mkdir()
        self.audio = root / 'fake-audio'
        self.audio.touch()
        self.shim = root / 'shim'
        self.shim.mkdir()
        (self.shim / 'sounddevice.py').symlink_to(HERE / 'shim.py')
        site = root / 'site'
        site.mkdir()
        shutil.copyfile(TREE / 'publish/coexist.pth', site / 'coexist.pth')
        bootstrap = root / 'bootstrap'
        bootstrap.mkdir()
        (bootstrap / 'sitecustomize.py').write_text(f'import site\nsite.addsitedir({str(site)!r})\n')
        binaries = root / 'bin'
        binaries.mkdir()
        ffmpeg = binaries / 'ffmpeg'
        ffmpeg.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable)
                          + ' -m hfmodem.tests.coexist.ffmpeg_stub "$@"\n')
        ffmpeg.chmod(0o700)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith('HFMODEM_COEXIST_') and key != 'HFMODEM_STATE_DIR'}
        self.env.update(HFMODEM_COEXIST_HARNESS='1', HFMODEM_COEXIST_RUN=self.run_id,
                        HFMODEM_COEXIST_LOG=str(self.events_path), HFMODEM_STATE_DIR=str(root),
                        HFMODEM_COEXIST_BARRIERS=str(self.barriers),
                        HFMODEM_COEXIST_AUDIO=str(self.audio), HFMODEM_COEXIST_SHIM=str(self.shim),
                        PYTHONDONTWRITEBYTECODE='1',
                        PYTHONPATH=os.pathsep.join(map(str, (bootstrap, TREE / 'packages/hfmodem', TREE))),
                        PATH=str(binaries) + os.pathsep + self.env.get('PATH', ''))
        if self.process_start_floor is not None:
            self.env['HFMODEM_COEXIST_START_TICKS'] = str(self.process_start_floor)
        self.children = []
        self.traced = []
        self.observer = None
        self.watcher = None

    def observe(self, paths):
        self.paths = set(map(str, paths))
        observer = LinuxObserver if sys.platform == 'linux' else MacOSObserver
        self.observer = (observer(self.run_id, started_after=self.process_start_floor)
                         if sys.platform == 'linux' else observer(self.run_id))
        if sys.platform == 'linux':
            self.watcher = Inotify(paths)
        self.observer.start()
        return self.observer

    def launch(self, argv, **kwargs):
        if sys.platform != 'linux':
            return self.spawn(argv, **kwargs)
        traced = TracedProcess(argv, directory=self.root / f'trace-{len(self.traced)}',
                               env=kwargs.pop('env', self.env), **kwargs)
        self.traced.append(traced)
        self.children.append(traced.process)
        return traced.process

    def collect(self, timeout=10):
        trace = Trace()
        codes = []
        for child in self.traced:
            observed = child.finish(timeout)
            trace.records.extend(observed.records)
            trace.pids.update(observed.pids)
            trace.exited.update(observed.exited)
            codes.append(child.process.returncode)
        if any(child.poll() is None for child in self.children):
            raise RuntimeError('participant tree is still running')
        if self.observer is None:
            raise RuntimeError('observe must precede participant launch')
        self.observer.capture('finished')
        self.observer.stop()
        snapshots = self.observer.snapshots
        self.observer = None
        notifications = self.watcher.drain() if self.watcher else []
        return Evidence(self.run_id, self.paths, sys.platform, self.events(), snapshots,
                        trace if self.traced else None, notifications,
                        max(codes) if codes else None, tree_finished=True,
                        lost=self.events_path.with_name(self.events_path.name + '.lost').exists())

    def install(self, monkeypatch):
        for key, value in self.env.items():
            if key.startswith('HFMODEM_'):
                monkeypatch.setenv(key, value)

    def spawn(self, argv, **kwargs):
        process = subprocess.Popen(argv, env=kwargs.pop('env', self.env),
                                   stdout=kwargs.pop('stdout', subprocess.PIPE),
                                   stderr=kwargs.pop('stderr', subprocess.PIPE), **kwargs)
        self.children.append(process)
        return process

    def python(self, source, **kwargs):
        return self.spawn([sys.executable, '-c', source], **kwargs)

    def pin(self, point):
        if Path(point).name != point:
            raise ValueError('invalid barrier name')
        (self.barriers / point).touch()

    def reached(self, point, timeout=5, *, quiescent=False):
        deadline = time.monotonic() + timeout
        while not (self.barriers / (point + '.reached')).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'barrier not reached: {point}')
            time.sleep(.005)
        event = next(event for event in reversed(self.events())
                     if event['event'] == 'barrier' and event['point'] == point)
        if self.observer is not None:
            if isinstance(self.observer, MacOSObserver):
                self.observer.barrier(point, participant_monotonic_ns=event['monotonic_ns'],
                                      claim_path=self.root / 'rig.claim' if quiescent else None,
                                      quiescent=quiescent)
            else:
                self.observer.barrier(point, event['monotonic_ns'])
        if self.watcher is not None:
            self.watcher.drain()
        return event

    def release(self, point):
        (self.barriers / (point + '.release')).touch()

    def events(self):
        return read_events(self.events_path)

    def close(self):
        for process in reversed(self.children):
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=2)
        try:
            if self.observer is not None:
                self.observer.stop()
        finally:
            if self.watcher is not None:
                self.watcher.close()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    run = Harness(tmp_path)
    run.install(monkeypatch)
    try:
        yield run
    finally:
        run.close()


@pytest.fixture
def fake_yaesu():
    fake = FakeYaesu()
    try:
        yield fake
    finally:
        fake.close()


@pytest.fixture
def tty0tty_pair():
    if os.environ.get('HFMODEM_COEXIST_LINE_PROOF') != '1':
        pytest.skip('modem-line proof not requested (portable software validation)')
    if sys.platform != 'linux':
        pytest.skip('tty0tty modem-line validation requires Linux; macOS proof needs attended hw')
    paths = tuple(os.environ.get('HFMODEM_COEXIST_TTY0TTY', '/dev/tnt0,/dev/tnt1').split(','))
    if len(paths) != 2 or paths[0] == paths[1] or not all(re.fullmatch(r'/dev/tnt\d+', p) for p in paths):
        pytest.fail('only a distinct tty0tty /dev/tntN pair is permitted')
    if not all(Path(path).exists() for path in paths):
        pytest.fail('modem-line proof requested but tty0tty is not installed; '
                    'provision /dev/tnt0,/dev/tnt1 in hfmodem-ci')
    yield paths
