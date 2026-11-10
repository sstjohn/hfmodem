# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Device-free operating owners with bounded storage barriers."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import config, recording, wav
from hfmodem.tests.station.recording_support import (
    audio_files, bounded, flush, samples,
)
from hfmodem.tests.station.recording_live_support import Clock, LOCAL, PEER


def forbidden(*args, **kwargs):
    raise AssertionError('unexpected device or child access')


@pytest.fixture
def stage4(monkeypatch, tmp_path, tools):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 48000)
    monkeypatch.setattr(wav, '_flac', lambda *a: (_ for _ in ()).throw(OSError('encoder unavailable')))
    monkeypatch.setattr(wav, 'store_later', lambda *a, **k: None)
    monkeypatch.setattr(wav, 'store_detached', lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, 'sounddevice', NS(
        rec=forbidden, wait=forbidden, stop=forbidden, InputStream=forbidden,
        OutputStream=forbidden, Stream=forbidden, query_devices=forbidden))
    sessions = []
    initialize = recording.RecordingSession.__init__

    def opened(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(recording.RecordingSession, '__init__', opened)
    state = NS(root=tmp_path, sessions=sessions, clock=Clock(1000.))
    yield state
    for session in reversed(sessions):
        bounded(session.close)


def finish(state):
    for session in state.sessions:
        bounded(session.close)


def one_session(state, policy):
    assert len(state.sessions) == 1, 'owner must use one shared RecordingSession'
    session = state.sessions[0]
    assert session.policy == policy
    return session


def flush_all(state):
    for session in state.sessions:
        if not session._closed:
            flush(session)


def lane_audio(session, lane):
    names = [d['source'] for d in session.index['segments'] if d['lane'] == lane.name]
    return samples(session.directory, {'segments': names})


def assert_idle(state, policy, count=None):
    finish(state)
    files = audio_files(state.root)
    assert bool(files) == (policy == 'everything'), f'idle audio persisted under {policy}'
    assert not [p for p in state.root.rglob('*') if p.suffix in ('.raw', '.pcm', '.part')]
    session = one_session(state, policy)
    assert not session.index['contacts'], 'listening or a peer label is not a contact'
    if count is not None:
        assert sum(lane.position - (lane.origin or 0) for lane in session.lanes.values()) == count
    if policy == 'off':
        assert all(lane.ring is None and lane.writer is None for lane in session.lanes.values())
    for path in files:
        params, _ = wav.read_pcm(path)
        assert params.nframes <= wav.SEGMENT_SAMPLES * params.framerate // 48000
    return session


def voice_cfg():
    rig = config.Rig(max_key_s=180)
    return NS(station=config.Station(mycall=LOCAL, grid='AA00aa'), rig=rig,
              selected_rig='ftdx10', voice=config.Voice(), models=config.Models(),
              recording=config.Recording(), declared_power_w=rig.power_w)


class SpeechPCM:
    ready = {'rss_gb': 0.}

    def __init__(self, text=''):
        self.text, self.requests = text, []

    def request(self, operation, **values):
        from hfmodem.contact import runtime
        self.requests.append((operation, values))
        if operation == 'synthesize':
            audio = np.tile(np.array([0., .1, -.1, 0.], np.float32), 1200)
            if values.get('path'):
                runtime.write_wav(values['path'], audio)
            return dict(audio=audio.tolist(), rate=48000, duration_s=len(audio)/48000, elapsed_s=0.)
        if operation == 'transcribe':
            return dict(text=self.text, words=[], probability=.98)
        return dict(probability=0., ok=True)

    def close(self):
        pass


class Output:
    tick = staticmethod(lambda: None)

    class Stop(Exception):
        pass

    def __init__(self, **kwargs):
        self.callback = kwargs['callback']
        self.finished = kwargs['finished_callback']
        self.blocks = []
        self.limit = kwargs.get('limit')

    def start(self):
        for _ in range(20000):
            out = np.empty((1024, 1), np.float32)
            try:
                self.callback(out, len(out), NS(), False)
            except self.Stop:
                self.blocks.append(out[:, 0].copy())
                self.finished()
                return
            self.blocks.append(out[:, 0].copy())
            if len(self.blocks) % 32 == 0:
                self.tick()
            if self.limit and sum(map(len, self.blocks)) >= self.limit:
                return
        raise AssertionError('output callback did not finish within its sample budget')

    def stop(self):
        pass

    abort = close = stop


@pytest.fixture
def shell_guard(tmp_path):
    import os
    bindir = tmp_path / 'guard'
    bindir.mkdir()
    log = tmp_path / 'unexpected-children'
    for name in ('rigctl', 'rigctld', 'python', 'python3', 'sox', 'rec', 'play',
                 'arecord', 'aplay', 'ffmpeg', 'nc', 'timeout'):
        path = bindir / name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$STAGE4_GUARD_LOG"\nexit 97\n')
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               STAGE4_GUARD_LOG=str(log))
    from hfmodem.tests import hermetic
    hermetic.guard.admit(['/bin/bash', '-c'], env)
    yield env
    assert not log.exists() or not log.read_text(), 'shell reached a guarded device command'


@pytest.fixture
def voice_owner(stage4, monkeypatch):
    from hfmodem.contact import runtime
    from hfmodem.contact.terminal import Terminal
    owners = []
    monkeypatch.setattr(runtime, 'pcm_device', lambda name, direction: name)
    sd = sys.modules['sounddevice']
    monkeypatch.setattr(sd, 'OutputStream', Output)
    monkeypatch.setattr(Output, 'tick', staticmethod(lambda: flush_all(stage4)))
    monkeypatch.setattr(sd, 'CallbackStop', Output.Stop, raising=False)
    monkeypatch.setattr(sd, 'CallbackAbort', Output.Stop, raising=False)
    monkeypatch.setattr(runtime.time, 'monotonic', stage4.clock)
    def make(policy='contacts', waveform='voice', name=None):
        monkeypatch.setenv('HFMODEM_RECORDING', policy)
        directory = stage4.root / (name or f'{waveform}-{len(owners)}')
        directory.mkdir()
        session = runtime.Session(voice_cfg(), directory, waveform=waveform,
            input_device='synthetic-in', output_device='synthetic-out',
            preview_device='synthetic-speaker', dial_hz=7105000)
        session.speech = (session.waveform.open_modem(None, directory) if waveform == 'cw'
                          else SpeechPCM(f'{LOCAL} DE {PEER} K'))
        session.language = NS(extract=lambda *a: dict(de=PEER, to=LOCAL, end='over'), close=lambda: None)
        session.rig = NS(key=lambda *a, **k: None, unkey=lambda *a: None, close=lambda: None)
        terminal = Terminal(session)
        monkeypatch.setattr(terminal, 'exchange', lambda: None)
        owners.append(session)
        return NS(session=session, terminal=terminal, directory=directory, state=stage4)
    yield make
    for session in reversed(owners):
        bounded(session.close)

