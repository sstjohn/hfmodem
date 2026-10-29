# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real PTY lifecycle/backpressure checks; no physical serial or RF devices."""
import errno
import os
import queue
import select
import threading
import time

import pytest

from hfmodem.core import hostmode
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost, SimPeer, serve
from hfmodem.shrike.serial_endpoint import PtcSerialEndpoint


class Endpoint:
    def __init__(self, host, **kwargs):
        self.ready = queue.Queue()
        self.errors = []
        self.stop = threading.Event()

        def run():
            try:
                serve(host, verbose=False, on_ready=self.ready.put,
                      stop_event=self.stop, **kwargs)
            except Exception as exc:
                self.errors.append(exc)

        self.thread = threading.Thread(target=run, name='test-ptc-serial')
        self.thread.start()
        self.name = self.ready.get(timeout=3)

    def close(self):
        self.stop.set()
        self.thread.join(3)
        assert not self.thread.is_alive()


def read_until(fd, expected, timeout=3):
    result = bytearray()
    deadline = time.monotonic() + timeout
    while expected not in result and time.monotonic() < deadline:
        if select.select([fd], [], [], max(0, deadline - time.monotonic()))[0]:
            result += os.read(fd, 65536)
    assert expected in result, bytes(result[-200:])
    return bytes(result)


def test_last_client_close_retires_endpoint_and_partial_terminal_command():
    host = PtcHost(SimPeer())
    endpoint = Endpoint(host)
    fd = os.open(endpoint.name, os.O_RDWR | os.O_NOCTTY)
    try:
        read_until(fd, b'cmd:')
        os.write(fd, b'MYCALL STALE')
        read_until(fd, b'STALE')
        os.close(fd)
        fd = None
        endpoint.thread.join(3)
        assert not endpoint.thread.is_alive()
        assert not endpoint.errors
        assert not host.hostmode and not host._line
        assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
        assert b'STALE' not in host.feed(b'\r')
    finally:
        if fd is not None:
            os.close(fd)
        endpoint.close()


def test_stop_before_client_attach_closes_both_pty_descriptors():
    host = PtcHost(SimPeer())
    endpoint = Endpoint(host)
    endpoint.close()
    assert not endpoint.errors
    with pytest.raises(OSError):
        os.open(endpoint.name, os.O_RDWR | os.O_NOCTTY)


class LargeReply(PtcHost):
    payload = bytes(range(256)) * 512

    def __init__(self):
        super().__init__(SimPeer())
        self.arq.cfg.cycle_s = 0.02
        self.ticks = []

    def feed(self, chunk):
        return self.payload

    def tick(self, *, elapsed_ticks=1, cycle_ticks=1):
        self.ticks.append(elapsed_ticks)
        super().tick(elapsed_ticks=elapsed_ticks, cycle_ticks=cycle_ticks)


def test_partial_writes_preserve_every_byte():
    host = LargeReply()
    endpoint = Endpoint(host)
    fd = os.open(endpoint.name, os.O_RDWR | os.O_NOCTTY)
    try:
        read_until(fd, b'cmd: ')
        os.write(fd, b'x')
        result = bytearray()
        deadline = time.monotonic() + 5
        while len(result) < len(host.payload) and time.monotonic() < deadline:
            if select.select([fd], [], [], 0.2)[0]:
                result += os.read(fd, 2048)
        assert bytes(result) == host.payload
        assert not endpoint.errors
    finally:
        os.close(fd)
        endpoint.close()


def test_stopped_reader_times_out_while_owner_keeps_ticking():
    host = LargeReply()
    endpoint = Endpoint(host, drain_timeout=0.2)
    fd = os.open(endpoint.name, os.O_RDWR | os.O_NOCTTY)
    try:
        read_until(fd, b'cmd: ')
        os.write(fd, b'x')
        endpoint.thread.join(3)
        assert not endpoint.thread.is_alive()
        assert len(endpoint.errors) == 1
        assert isinstance(endpoint.errors[0], TimeoutError)
        assert len(host.ticks) >= 3
        assert any('unread serial output' in line for line in host.log_lines)
    finally:
        os.close(fd)
        endpoint.close()


def test_output_overflow_retires_instead_of_blocking():
    host = LargeReply()
    endpoint = Endpoint(host, output_limit=4096)
    fd = os.open(endpoint.name, os.O_RDWR | os.O_NOCTTY)
    try:
        read_until(fd, b'cmd: ')
        os.write(fd, b'x')
        endpoint.thread.join(3)
        assert not endpoint.thread.is_alive()
        assert len(endpoint.errors) == 1
        assert isinstance(endpoint.errors[0], BufferError)
    finally:
        os.close(fd)
        endpoint.close()


def test_late_owner_ages_slots_without_replaying_opportunities():
    class SlowReply(LargeReply):
        def feed(self, chunk):
            time.sleep(0.14)
            return b'OK'

    host = SlowReply()
    endpoint = Endpoint(host)
    fd = os.open(endpoint.name, os.O_RDWR | os.O_NOCTTY)
    try:
        read_until(fd, b'cmd: ')
        os.write(fd, b'x')
        read_until(fd, b'OK')
        assert any(elapsed >= 5 for elapsed in host.ticks)
        # One late servicing call ages the gap. A catch-up loop would record
        # only elapsed=1, repeatedly emitting historical opportunities.
    finally:
        os.close(fd)
        endpoint.close()


def test_loss_resets_hostmode_partial_crc_and_retransmission_cache():
    host = PtcHost(SimPeer())
    host.feed(b'JHOST4\r')
    host.feed(b'\xaa')
    host._modem._buffered = b'old response'
    host.settings['%M'] = '1'
    host.attachment_lost()
    assert not host.hostmode
    assert not host._attachment_started
    assert host.settings['%M'] == '0'
    assert b'PTC-IIIusb' in host.feed(b'VER\r')
    host.feed(b'JHOST4\r')
    assert host.hostmode
    assert host._modem.feed(hostmode.REQUEST) == b''


def test_passive_endpoint_leaves_every_tick_on_the_existing_modem_owner():
    class OwnedHost(PtcHost):
        def __init__(self):
            super().__init__(SimPeer())
            self.owner_calls = []
            self.ticks = []

        def feed(self, chunk):
            self.owner_calls.append(('feed', threading.get_ident()))
            return super().feed(chunk)

        def tick(self, *, elapsed_ticks=1, cycle_ticks=1):
            self.ticks.append((elapsed_ticks, cycle_ticks, threading.get_ident()))
            return super().tick(elapsed_ticks=elapsed_ticks, cycle_ticks=cycle_ticks)

        def attachment_lost(self):
            self.owner_calls.append(('lost', threading.get_ident()))
            return super().attachment_lost()

    host = OwnedHost()
    endpoint = PtcSerialEndpoint(host)
    owner = threading.get_ident()
    name = endpoint.open()
    fd = os.open(name, os.O_RDWR | os.O_NOCTTY)
    try:
        assert endpoint.service()
        read_until(fd, b'cmd: ')
        os.write(fd, b'MYCALL LOCAL\r')
        for _ in range(3):
            assert endpoint.service(timeout=.01)
        read_until(fd, b'LOCAL')
        assert host.mycall == 'LOCAL'
        assert host.ticks == []  # Host traffic does not invent a radio cycle.
        host.tick(elapsed_ticks=7, cycle_ticks=3)
        assert host.ticks == [(7, 3, owner)]
        os.close(fd)
        fd = None
        assert not endpoint.service(timeout=.1)
        assert endpoint.closed
        assert host.owner_calls == [('feed', owner), ('lost', owner)]
        with pytest.raises(RuntimeError, match='only one attachment'):
            endpoint.open()
    finally:
        if fd is not None:
            os.close(fd)
        endpoint.close()


def test_passive_endpoint_refuses_reader_thread_state_machine_access():
    host = PtcHost(SimPeer())
    endpoint = PtcSerialEndpoint(host)
    endpoint.open()
    errors = []

    def other_thread():
        for operation in (endpoint.service, endpoint.close):
            try:
                operation()
            except RuntimeError as exc:
                errors.append(str(exc))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(3)
    try:
        assert not thread.is_alive()
        assert len(errors) == 2
        assert all('modem owner' in message for message in errors)
        assert not endpoint.closed
        assert endpoint.service()
    finally:
        endpoint.close()


def test_passive_output_fault_retires_without_relying_on_caller_finally():
    host = LargeReply()
    endpoint = PtcSerialEndpoint(host, output_limit=4096)
    fd = os.open(endpoint.open(), os.O_RDWR | os.O_NOCTTY)
    owned_fd = endpoint._fd
    try:
        endpoint.service()
        read_until(fd, b'cmd: ')
        os.write(fd, b'x')
        with pytest.raises(BufferError, match='OVERFLOW'):
            endpoint.service(timeout=.1)
        assert endpoint.closed
        assert endpoint.pending_bytes == 0
        assert host.ticks == []
        with pytest.raises(OSError):
            os.fstat(owned_fd)
    finally:
        try:
            os.close(fd)
        except OSError as exc:
            assert exc.errno in (errno.EIO, errno.ENXIO)
        endpoint.close()


def test_failed_endpoint_bind_does_not_abort_an_unattached_contact(tmp_path):
    host = PtcHost(SimPeer())
    host.arq.on_host_connect('N0CALL', 'N0PEER')
    assert host.arq.state is State.CONNECTING
    endpoint = PtcSerialEndpoint(host)
    with pytest.raises(FileNotFoundError):
        endpoint.open(str(tmp_path/'missing-device'))
    assert endpoint.closed
    assert host.arq.state is State.CONNECTING


@pytest.mark.parametrize('kind', ['file', 'symlink', 'directory'])
def test_named_endpoint_exclusive_bind_preserves_existing_operator_paths(tmp_path, kind):
    path = tmp_path / 'ptc'
    if kind == 'file':
        path.write_bytes(b'operator-owned')
    elif kind == 'symlink':
        path.symlink_to(tmp_path / 'missing-target')
    else:
        path.mkdir()
    host = PtcHost()
    host.arq.state = State.CONNECTED
    endpoint = PtcSerialEndpoint(host, endpoint=path)
    with pytest.raises(FileExistsError):
        endpoint.open()
    assert endpoint.closed and endpoint._replica is None
    assert host.arq.state == State.CONNECTED
    if kind == 'file':
        assert path.read_bytes() == b'operator-owned'
    elif kind == 'symlink':
        assert os.readlink(path) == str(tmp_path / 'missing-target')
    else:
        assert path.is_dir()


def test_named_endpoint_removes_only_the_alias_it_created(tmp_path):
    path = tmp_path / 'ptc'
    endpoint = PtcSerialEndpoint(PtcHost(), endpoint=path)
    assert endpoint.open() == str(path) and path.is_symlink()
    target = os.readlink(path)
    assert target == endpoint._slave_name
    endpoint.close()
    assert not path.is_symlink() and endpoint.alias_error is None
    successor = PtcSerialEndpoint(PtcHost(), endpoint=path)
    try:
        successor.open()
        endpoint.close()
        assert path.is_symlink() and os.readlink(path) == successor._slave_name
    finally:
        successor.close()
    assert not path.is_symlink()


def test_named_endpoint_preserves_an_operator_replacement_during_close(tmp_path):
    path = tmp_path / 'ptc'
    endpoint = PtcSerialEndpoint(PtcHost(), endpoint=path)
    endpoint.open()
    path.unlink()
    path.write_bytes(b'new operator content')
    endpoint.close()
    assert endpoint.closed and endpoint._replica is None
    assert path.read_bytes() == b'new operator content'
    assert endpoint.alias_error == 'PTC endpoint name replaced while active'


def test_named_endpoint_cannot_also_open_a_serial_device(tmp_path, monkeypatch):
    endpoint = PtcSerialEndpoint(PtcHost(), endpoint=tmp_path / 'ptc')
    with monkeypatch.context() as scope:
        scope.setattr(os, 'open', lambda *a, **kw: pytest.fail('serial device opened'))
        with pytest.raises(ValueError, match='cannot also open'):
            endpoint.open('/dev/not-a-device')
    assert endpoint.closed and endpoint._replica is None
    assert not (tmp_path / 'ptc').is_symlink()
