# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import pytest

from hfmodem.ax25.kiss import Parser, encode
from hfmodem.brubru.host.server import KissServer


class Socket:
    def __init__(self):
        self.wire = bytearray()
        self.closed = False
        self.blocked = False
        self.limit = 65536

    def setblocking(self, value):
        assert value is False

    def send(self, data):
        if self.blocked:
            raise BlockingIOError
        count = min(len(data), self.limit)
        self.wire.extend(data[:count])
        return count

    def close(self):
        self.closed = True


@pytest.fixture
def server():
    queued, params, drops = [], [], []
    link = SimpleNamespace(enqueue=queued.append, host=None,
                           set_parameter=lambda c, v: params.append((c, v)),
                           drop_pending=lambda: drops.append(True))
    server = KissServer(link, log=lambda _: None)
    server.queued, server.params, server.drops = queued, params, drops
    server._accept(Socket())
    return server


def test_one_client_and_stock_framing_without_greeting(server):
    second = Socket()
    assert not server._accept(second) and second.closed
    assert not server._client.wire
    frame = b'hello\xc0\xdb'
    for byte in encode(frame):
        server._receive(bytes([byte]))
    assert server.queued == [frame]
    server.send_frame(frame)
    server._send(0)
    assert Parser().feed(server._client.wire) == [(0, 0, frame)]


@pytest.mark.parametrize('opening', [b'GET / HTTP/1.1\r\n', b'POST / HTTP/1.0\r\n',
                                    b'OPTIONS * HTTP/1.1\r\n', b'PRI * HTTP/2.0\r\n'])
def test_http_opening_rejected_even_one_byte_at_a_time(server, opening):
    client = server._client
    for byte in opening:
        if server._client is None:
            break
        server._receive(bytes([byte]))
    assert client.closed and not server.queued


def test_commands_ports_bad_escape_and_oversize_recovery(server):
    for command in (1, 2, 3, 5, 6):
        server._receive(encode(b'\x11', command))
    server._receive(encode(b'ignored', port=1))
    server._receive(b'\xc0\x00bad\xdbxignored\xc0')
    server._receive(b'\xc0\x00' + b'x' * 2049 + b'\xc0')
    server._receive(encode(b'next'))
    assert server.params == [(1, 17), (2, 17), (3, 17)]
    assert server.queued == [b'next']
    assert server._parser.dropped == 2


def test_outbound_bound_drop_oldest_and_blocked_sender(server):
    for n in range(33):
        server.send_frame(bytes([n]))
    assert len(server._out) == 32
    assert Parser().feed(server._out[0]) == [(0, 0, b'\x01')]
    assert 'dropped oldest' in server.events[-1]
    client = server._client
    client.blocked = True
    server._send(5)
    server._send(6.999)
    assert not client.closed
    server._send(7)
    assert client.closed and not server._out and server.drops


def test_partial_frame_survives_queue_overflow(server):
    client = server._client
    client.limit = 1
    server.send_frame(b'first')
    server._send(0)
    for n in range(40):
        server.send_frame(bytes([n]))
    client.limit = 65536
    while server._out:
        server._send(0.5)
    frames = Parser().feed(client.wire)
    assert frames[0] == (0, 0, b'first')
    assert len(frames) == 32


def test_stop_drops_queue_and_closes_client(server):
    client = server._client
    server.send_frame(b'frame')
    server.stop()
    assert client.closed and not server._out and server.drops


def test_bind_failure_is_synchronous(monkeypatch, server):
    class Listener(Socket):
        def setsockopt(self, *args):
            pass

        def bind(self, address):
            assert address[0] == '127.0.0.1'
            raise OSError('port occupied')

    listener = Listener()
    monkeypatch.setattr('hfmodem.brubru.host.server.socket.socket', lambda *args: listener)
    with pytest.raises(OSError, match='port occupied'):
        server.start()
    assert listener.closed and server._thread is None


def test_bound_port_is_reported(monkeypatch, server):
    class Listener(Socket):
        def setsockopt(self, *args):
            pass
        def bind(self, address):
            assert address == ('127.0.0.1', 0)
        def listen(self, count):
            pass
        def getsockname(self):
            return ('127.0.0.1', 45123)
    class Thread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
    monkeypatch.setattr('hfmodem.brubru.host.server.socket.socket', lambda *args: Listener())
    monkeypatch.setattr('hfmodem.brubru.host.server.threading.Thread', Thread)
    server.start()
    assert server.port == 45123 and server.link.host is server
    server.stop()
