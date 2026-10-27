# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Explicit on-demand process/pipe failures; no upstream build is needed."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

import pytest

import run as h


def test_short_writes_preserve_every_command_byte(monkeypatch):
    reader, writer = os.pipe()
    actual_write = os.write
    monkeypatch.setattr(h.os, 'write', lambda fd, blob: actual_write(fd, blob[:3]))
    with os.fdopen(reader, 'rb', buffering=0) as source, os.fdopen(writer, 'wb', buffering=0) as sink:
        blob = bytes(range(100))
        h.write_all(sink, blob)
        assert h.read_exact(source, len(blob)) == blob
        assert os.get_blocking(sink.fileno())


def test_nonreading_peer_cannot_block_a_command_forever():
    reader, writer = os.pipe()
    try:
        with os.fdopen(writer, 'wb', buffering=0) as sink:
            with pytest.raises(TimeoutError, match='consume'):
                h.write_all(sink, b'x' * 1_000_000, timeout=.01)
            assert os.get_blocking(sink.fileno())
    finally:
        os.close(reader)


def test_zero_length_write_is_eof_instead_of_a_spin(monkeypatch):
    reader, writer = os.pipe()
    monkeypatch.setattr(h.os, 'write', lambda *args: 0)
    try:
        with os.fdopen(writer, 'wb', buffering=0) as sink:
            with pytest.raises(EOFError, match='consuming'):
                h.write_all(sink, b'CONNECT\n')
    finally:
        os.close(reader)


def endpoint(monkeypatch, script, **kwargs):
    actual_popen = subprocess.Popen
    monkeypatch.setattr(h, 'validate_probe', lambda: None)
    monkeypatch.setattr(h.subprocess, 'Popen', lambda *args, **kw:
                        actual_popen([sys.executable, '-c', script], **kw))
    return h.Xara(500, False, 'OFF', lambda *args: None, **kwargs)


@pytest.mark.parametrize('reply,error', [
    (b'\x05\x00\x00\x00ab', EOFError),
    ((1_000_000).to_bytes(4, 'little'), ValueError),
])
def test_invalid_reply_reaps_the_child_and_closes_both_pipes(monkeypatch, reply, error):
    x = endpoint(monkeypatch,
                 f'import sys; sys.stdin.buffer.readline(); sys.stdout.buffer.write({reply!r}); sys.stdout.flush()')
    with pytest.raises(error):
        x.command('CONNECT')
    assert x.process.poll() is not None
    assert x.process.stdin.closed and x.process.stdout.closed
    x.close()
    with pytest.raises(ValueError, match='closed'):
        x.command('CONNECT')


def test_reply_timeout_reaps_the_child(monkeypatch):
    x = endpoint(monkeypatch,
                 'import sys,time; sys.stdin.buffer.readline(); time.sleep(.1)', timeout=.01)
    with pytest.raises(TimeoutError):
        x.command('CONNECT')
    assert x.process.poll() is not None
    assert x.process.stdin.closed and x.process.stdout.closed


def test_close_broken_pipe_still_kills_and_reaps_once():
    events = []
    class Input:
        def close(self):
            events.append('stdin')
            raise BrokenPipeError()
    class Process:
        stdin = Input()
        stdout = SimpleNamespace(close=lambda: events.append('stdout'))
        def wait(self, timeout):
            events.append('wait')
            if events.count('wait') == 1:
                raise subprocess.TimeoutExpired('adapter', timeout)
        def kill(self):
            events.append('kill')
    x = h.Xara.__new__(h.Xara)
    x.process, x._closed = Process(), False
    x.close()
    x.close()
    assert events == ['stdin', 'wait', 'kill', 'wait', 'stdout']


def trace_handles(monkeypatch):
    handles = []
    actual_open = Path.open
    def opened(path, *args, **kwargs):
        handle = actual_open(path, *args, **kwargs)
        if path.name == 'trace.jsonl':
            handles.append(handle)
        return handle
    monkeypatch.setattr(Path, 'open', opened)
    return handles


def test_second_endpoint_startup_failure_closes_the_first_and_trace(monkeypatch, tmp_path):
    handles = trace_handles(monkeypatch)
    first = SimpleNamespace(closed=False)
    def close():
        first.closed = True
    first.close = close
    created = []
    def launch(*args):
        if created:
            raise OSError('second endpoint failed')
        created.append(first)
        return first
    monkeypatch.setattr(h, 'Xara', launch)
    args = SimpleNamespace(out=tmp_path, bw=[500], registered=False, compression='OFF', provenance={})
    with pytest.raises(OSError, match='second endpoint'):
        h.controls(args)
    assert first.closed
    assert handles and all(handle.closed for handle in handles)


def test_session_startup_failure_closes_trace_wavs_and_stops_modem(monkeypatch, tmp_path):
    from hfmodem.kestrel.host.vara_modem import VaraHostModem
    handles = trace_handles(monkeypatch)
    stopped = []
    actual_stop = VaraHostModem.stop
    def stop(modem):
        stopped.append(modem)
        actual_stop(modem)
    monkeypatch.setattr(VaraHostModem, 'stop', stop)
    def failed(*args):
        raise OSError('endpoint startup failed')
    monkeypatch.setattr(h, 'Xara', failed)
    args = SimpleNamespace(out=tmp_path, registered=False, compression='OFF', record_audio=True)
    with pytest.raises(OSError, match='startup failed'):
        h.session_case(args, 500, 'kestrel')
    assert len(stopped) == 1
    assert handles and all(handle.closed for handle in handles)
    for path in tmp_path.rglob('*.wav'):
        with wave.open(str(path), 'rb') as recorded:
            assert recorded.getnframes() == 0
