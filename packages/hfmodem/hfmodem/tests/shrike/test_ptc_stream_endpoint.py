# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real stream transport and retirement without a second PACTOR clock."""
import socket
import threading
import time

import pytest

from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.shrike.stream_endpoint import PtcStreamEndpoint


class Host(PtcHost):
    payload = bytes(range(256))*512

    def __init__(self):
        super().__init__(SimPeer())
        self.calls = []

    def feed(self, data):
        self.calls.append(('feed', threading.get_ident()))
        return self.payload

    def tick(self, **kwargs):
        raise AssertionError('stream I/O must not tick the ARQ')

    def attachment_lost(self):
        self.calls.append(('lost', threading.get_ident()))
        super().attachment_lost()


def opened(**options):
    server, client = socket.socketpair()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    client.setblocking(False)
    host = Host()
    endpoint = PtcStreamEndpoint(host, server, **options)
    endpoint.open()
    endpoint.service()
    assert b'cmd:' in client.recv(4096)
    return host, endpoint, client


def test_partial_writes_preserve_binary_and_only_call_the_native_owner():
    host, endpoint, client = opened()
    try:
        owner = threading.get_ident()
        client.sendall(b'give me the payload')
        received = bytearray()
        deadline = time.monotonic() + 2
        while len(received) < len(host.payload):
            assert time.monotonic() < deadline
            endpoint.service()
            try:
                received += client.recv(2048)
            except BlockingIOError:
                pass
        assert bytes(received) == host.payload
        client.close()
        assert not endpoint.service() and endpoint.hangup
        assert host.calls == [('feed', owner), ('lost', owner)]
        assert not host._attachment_started
        endpoint.close()
        assert host.calls == [('feed', owner), ('lost', owner)]
        with pytest.raises(RuntimeError, match='only one attachment'):
            endpoint.open()
    finally:
        client.close()
        endpoint.close()


@pytest.mark.parametrize('failure', ['overflow', 'stopped-reader'])
def test_backpressure_fails_and_retires_instead_of_blocking(failure):
    host, endpoint, client = opened(output_limit=4096 if failure == 'overflow' else 256*1024,
                                    drain_timeout=.03)
    try:
        client.sendall(b'payload')
        deadline = time.monotonic() + 1
        with pytest.raises(BufferError if failure == 'overflow' else TimeoutError):
            while time.monotonic() < deadline:
                endpoint.service()
                time.sleep(.005)
        assert endpoint.closed and host.calls[-1][0] == 'lost'
        assert any('unread stream output' in line for line in host.log_lines) or failure == 'overflow'
    finally:
        client.close()
        endpoint.close()


def test_broker_revocation_retires_the_parser_on_the_owner():
    host, endpoint, client = opened()
    endpoint.stream.close()
    try:
        assert not endpoint.service()
        assert endpoint.hangup and endpoint.closed
        assert host.calls == [('lost', threading.get_ident())]
    finally:
        client.close()
        endpoint.close()


def test_backend_value_error_is_visible_and_is_not_mistaken_for_socket_revocation():
    host, endpoint, client = opened()
    def bad(data):
        raise ValueError('backend defect')
    host.feed = bad
    try:
        client.sendall(b'x')
        with pytest.raises(ValueError, match='backend defect'):
            endpoint.service()
        assert endpoint.closed and not endpoint.hangup
    finally:
        client.close()
        endpoint.close()


def test_revocation_before_owner_adoption_does_not_open_or_retire_a_host():
    server, client = socket.socketpair()
    host = Host()
    endpoint = PtcStreamEndpoint(host, server)
    server.close()
    try:
        assert endpoint.open() is False
        assert endpoint.closed and endpoint.hangup and not host.calls
        assert not host._attachment_started
    finally:
        endpoint.close()
        client.close()
