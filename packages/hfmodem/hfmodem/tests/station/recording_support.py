# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Synthetic PCM and bounded synchronization for policy-core tests.

Test-facing metadata uses contacts keyed by ID, lanes keyed by name, and
descriptors in ``segments``. Lane intervals carry ``requested``/``actual``
half-open bounds and ``gaps``. Paths are relative to the run directory.
"""
import importlib
import json
import threading
import wave

import numpy as np
import pytest

from hfmodem.core import wav

TIMEOUT = 5


def same_path(left, right):
    """One file whichever form names it: recordings open Windows paths extended-length."""
    from hfmodem.core.capture_identity import _absolute
    return _absolute(left) == _absolute(right)


def readable(fd, timeout=TIMEOUT):
    """True once a pipe has data or EOF; Windows select() takes only sockets."""
    import os
    if os.name != 'nt':
        import select
        return bool(select.select([fd], [], [], timeout)[0])
    import ctypes, msvcrt, time
    from ctypes import wintypes
    available = wintypes.DWORD()
    deadline = time.monotonic() + timeout
    while True:
        if not ctypes.windll.kernel32.PeekNamedPipe(wintypes.HANDLE(msvcrt.get_osfhandle(fd)), None, 0,
                                                    None, ctypes.byref(available), None):
            return True
        if available.value:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(.01)
RATE = 8


def module():
    return importlib.import_module('hfmodem.core.recording')


def bounded(fn):
    errors = []
    def run():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive(), 'recording operation did not finish'
    if errors:
        raise errors[0]


def flush(rec):
    done = threading.Event()
    rec.after_pending(done.set)
    assert done.wait(TIMEOUT), 'ordered storage barrier did not finish'


def push(rec, lane, start, stop):
    for first in range(start, stop, RATE):
        last = min(first + RATE, stop)
        lane.push(first, pcm(first, last))
        flush(rec)


def pcm(start, stop):
    return ((np.arange(start, stop) % 30000) / 32768).astype(np.float32)


def index(root):
    return json.loads((root / 'recording.json').read_text())


def interval(root, contact='c1', lane='rx'):
    return index(root)['contacts'][contact]['lanes'][lane]


def audio_files(root):
    return sorted(p for p in root.rglob('*') if p.suffix in ('.wav', '.flac'))


def samples(root, reference):
    chunks = []
    for name in reference['segments']:
        path = root / (name['source'] if isinstance(name, dict) else name)
        with wav.opened_source(path) as stream, wave.open(stream) as src:
            chunks.append(np.frombuffer(src.readframes(src.getnframes()), '<i2'))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype='<i2')


@pytest.fixture
def sessions(monkeypatch, tmp_path):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav, '_flac', lambda *a: (_ for _ in ()).throw(OSError('synthetic encoder unavailable')))
    opened = []
    def make(policy='contacts', **kwargs):
        recording = module()
        root = kwargs.pop('directory', tmp_path / f'run-{len(opened)}')
        rec = recording.RecordingSession(policy, root, **kwargs)
        opened.append(rec)
        return rec, root
    yield make
    for rec in reversed(opened):
        bounded(rec.close)


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def restore(root, saved):
    root.mkdir(parents=True, exist_ok=True)
    for name, data in saved.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
