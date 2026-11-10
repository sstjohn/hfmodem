# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import io
import json

from hfmodem.bbs.config import BBSConfig
from hfmodem.bbs.core import BBSCore
from hfmodem.bbs.frontends.local import main, run
from hfmodem.bbs.store import Store


def test_local_classic_menu_transcript_and_eof(tmp_path):
    store = Store(tmp_path / 'state')
    core = BBSCore(BBSConfig(state_root=store.root), store)
    output = io.StringIO()
    transcript = tmp_path / 'session.jsonl'
    try:
        assert run(core, 'K1ABC', incoming=io.StringIO('D\nQ\n\nG\n'), outgoing=output, transcript=transcript) == 0
        assert 'Main:' in output.getvalue() and 'Doors' in output.getvalue()
        records = [json.loads(line) for line in transcript.read_text().splitlines()]
        assert any(r.get('text') == 'BBS MORE' for r in records)
        assert (tmp_path / 'session.jsonl.txt').exists()
        assert core.quiescent
    finally:
        store.close()


def test_missing_config_clean_error(tmp_path, capsys):
    assert main(['--config', str(tmp_path / 'missing.toml')]) == 2
    assert 'BBS local:' in capsys.readouterr().err


class _Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class _DeadlineCore:
    """Tiny clock-injected core probe: deadline enforcement requires tick()."""
    def __init__(self, event=None):
        from types import SimpleNamespace
        self.clock = _Clock()
        self.session = SimpleNamespace(id='local-generation', state='lobby', paused=False)
        self.message_ui = SimpleNamespace(busy=False)
        self.event = event
        self.delivered = []
        self.inputs = []
        self.closed = []
        self.quiescent = True

    def open(self, peer, presentation, mode):
        return self.session.id

    def tick(self):
        self.clock.now += .1
        if self.clock() >= 1 and self.session.state != 'closed':
            self.close(self.session.id, 'hard connection deadline')

    def poll(self, sid):
        result = [self.event] if self.event is not None else []
        self.event = None
        return result

    def output_done(self, sid, output_id, disposition):
        self.delivered.append((output_id, disposition))

    def input(self, sid, input_id, action):
        self.inputs.append(action)

    def close(self, sid, reason):
        self.session.state = 'closed'
        self.closed.append(reason)

    def shutdown(self):
        pass


def _run_thread(core, incoming, outgoing, **options):
    import threading
    result = []
    errors = []
    def worker():
        try:
            result.append(run(core, 'K1ABC', incoming=incoming, outgoing=outgoing, **options))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, result, errors


def test_partial_pipe_input_cannot_block_hard_deadline():
    import fcntl
    import os
    read_fd, write_fd = os.pipe()
    incoming = os.fdopen(read_fd, 'r')
    original = fcntl.fcntl(read_fd, fcntl.F_GETFL)
    os.write(write_fd, b'partial command with no newline')
    core = _DeadlineCore()
    thread, result, errors = _run_thread(core, incoming, io.StringIO())
    try:
        thread.join(2)
        assert not thread.is_alive(), 'partial line blocked core deadline ticks'
        assert result == [0] and not errors
        assert core.closed == ['hard connection deadline']
        assert core.inputs == []
        assert fcntl.fcntl(read_fd, fcntl.F_GETFL) == original
    finally:
        os.close(write_fd)
        thread.join(2)
        incoming.close()


def test_full_output_pipe_cannot_block_deadline_or_claim_delivery(tmp_path):
    import fcntl
    import os
    from hfmodem.bbs.model import OutputEvent
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    try:
        while True:
            os.write(write_fd, b'x' * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(write_fd, True)
    outgoing = os.fdopen(write_fd, 'w')
    original = fcntl.fcntl(write_fd, fcntl.F_GETFL)
    event = OutputEvent('page', 'blocked-page', 'MUST BE DISCARDED', 'local-generation')
    core = _DeadlineCore(event)
    transcript = tmp_path / 'blocked.jsonl'
    thread, result, errors = _run_thread(core, io.StringIO(''), outgoing, transcript=transcript)
    try:
        thread.join(2)
        assert not thread.is_alive(), 'full output pipe blocked core deadline ticks'
        assert result == [0] and not errors
        assert core.closed == ['hard connection deadline']
        assert core.delivered == []
        assert fcntl.fcntl(write_fd, fcntl.F_GETFL) == original
        os.set_blocking(read_fd, False)
        written = bytearray()
        while True:
            try:
                data = os.read(read_fd, 65536)
            except BlockingIOError:
                break
            if not data:
                break
            written.extend(data)
        assert b'MUST BE DISCARDED' not in written
        records = [json.loads(line) for line in transcript.read_text().splitlines()]
        assert any(record.get('disposition') == 'cancelled' for record in records)
    finally:
        os.close(read_fd)
        thread.join(2)
        outgoing.close()


def test_socket_aliases_restore_original_flags_on_exception():
    import fcntl
    import os
    import socket
    import pytest
    from hfmodem.bbs.frontends.local import _FDStreams
    local, remote = socket.socketpair()
    duplicate = os.dup(local.fileno())
    incoming = local.makefile('r')
    outgoing = os.fdopen(duplicate, 'w')
    original = fcntl.fcntl(local.fileno(), fcntl.F_GETFL)
    try:
        with pytest.raises(RuntimeError, match='injected'):
            with _FDStreams(incoming, outgoing):
                assert not os.get_blocking(local.fileno())
                assert not os.get_blocking(duplicate)
                raise RuntimeError('injected failure')
        assert fcntl.fcntl(local.fileno(), fcntl.F_GETFL) == original
        assert fcntl.fcntl(duplicate, fcntl.F_GETFL) == original
    finally:
        incoming.close()
        outgoing.close()
        local.close()
        remote.close()


def test_unknown_non_fd_stream_rejected_without_reading_or_opening_core():
    import pytest
    class BlockingStream:
        def readline(self):
            pytest.fail('unknown stream was read')
    core = _DeadlineCore()
    with pytest.raises(ValueError, match='file descriptor or StringIO'):
        run(core, 'K1ABC', incoming=BlockingStream(), outgoing=io.StringIO())
    assert core.clock() == 0


def test_partial_output_is_completed_before_output_done(monkeypatch):
    import os
    from hfmodem.bbs.model import OutputEvent
    core = _DeadlineCore(OutputEvent('page', 'partial', 'hello', 'local-generation'))
    # Give the partial writer sufficient fake time to complete before expiry.
    original_tick = core.tick
    def tick():
        core.clock.now -= .09
        original_tick()
    core.tick = tick
    read_fd, write_fd = os.pipe()
    outgoing = os.fdopen(write_fd, 'w')
    original_write = os.write
    calls = []
    def partial(fd, data):
        calls.append(bytes(data))
        if len(calls) == 1:
            assert core.delivered == []
            return original_write(fd, data[:2])
        assert core.delivered == []
        return original_write(fd, data)
    monkeypatch.setattr(os, 'write', partial)
    try:
        assert run(core, 'K1ABC', incoming=io.StringIO(''), outgoing=outgoing) == 0
        assert core.delivered == [('partial', 'complete')]
        assert os.read(read_fd, 20) == b'hello\n'
        assert calls == [b'hello\n', b'llo\n']
    finally:
        outgoing.close()
        os.close(read_fd)


def test_pipe_eof_discards_unsubmitted_partial_command():
    import os
    read_fd, write_fd = os.pipe()
    incoming = os.fdopen(read_fd, 'r')
    os.write(write_fd, b'north')
    os.close(write_fd)
    core = _DeadlineCore()
    try:
        assert run(core, 'K1ABC', incoming=incoming, outgoing=io.StringIO()) == 0
        assert core.inputs == []
        assert core.closed == ['local EOF']
    finally:
        incoming.close()


def test_real_core_connection_clock_expires_during_partial_pipe(tmp_path):
    import os
    from hfmodem.bbs.config import Limits
    store = Store(tmp_path / 'state')
    clock = _Clock()
    core = BBSCore(BBSConfig(state_root=store.root, limits=Limits(idle_seconds=1, connection_seconds=1)), store, clock=clock)
    original_tick = core.tick
    def tick():
        clock.now += .025
        original_tick()
    core.tick = tick
    read_fd, write_fd = os.pipe()
    incoming = os.fdopen(read_fd, 'r')
    os.write(write_fd, b'partial without Enter')
    thread, result, errors = _run_thread(core, incoming, io.StringIO())
    try:
        thread.join(2)
        assert not thread.is_alive()
        assert result == [0] and not errors
        assert clock() >= 1
        assert core.session.state == 'closed'
        assert core.session.input_ids == set()
        assert core.quiescent
    finally:
        os.close(write_fd)
        thread.join(2)
        incoming.close()
        store.close()
