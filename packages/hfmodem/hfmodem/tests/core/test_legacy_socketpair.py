# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Legacy attachments over real TCP; no physical endpoints."""
import socket
import struct
import time

import pytest

from hfmodem.besra.host.server import HostServer
from hfmodem.kestrel.host.server import VaraServer


@pytest.fixture(params=("vara", "ardop"))
def server(request):
    if request.param == "vara":
        srv = VaraServer(cmd_port=0, data_port=0, half_open_timeout=.15)
        srv.start_background()
        cmd = srv.cmd_port
    else:
        srv = HostServer(control_port=0, data_port=0, half_open_timeout=.15, quiet=True)
        srv.start()
        cmd = srv.control_port
    yield srv, cmd
    srv.stop()


def connect(port):
    return socket.create_connection(("127.0.0.1", port), timeout=2)


def until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.005)


def read_until(sock, token):
    data = bytearray()
    while token not in data:
        block = sock.recv(4096)
        assert block
        data.extend(block)
    return bytes(data)


def test_data_first_and_either_socket_loss_retires_both(server):
    srv, cmd_port = server
    with connect(srv.data_port) as data, connect(cmd_port) as cmd:
        cmd.sendall(b"VERSION\r")
        assert b"VERSION" in read_until(cmd, b"VERSION")
        data.shutdown(socket.SHUT_RDWR)
        data.close()
        assert cmd.recv(4096) == b""
    until(lambda: srv._pairs._active is None)
    with connect(cmd_port) as cmd, connect(srv.data_port) as data:
        cmd.sendall(b"VERSION\r")
        read_until(cmd, b"VERSION")
        cmd.shutdown(socket.SHUT_RDWR)
        cmd.close()
        assert data.recv(4096) == b""


def test_half_open_lifetime_is_bounded(server):
    srv, cmd_port = server
    with connect(cmd_port) as cmd:
        assert cmd.recv(4096) == b""
    with connect(srv.data_port) as data:
        assert data.recv(4096) == b""


def test_detectable_half_open_overlap_is_rejected(server):
    srv, cmd_port = server
    with connect(cmd_port) as first:
        until(lambda: len(srv._pairs._pending) == 1)
        with connect(cmd_port) as second:
            assert second.recv(4096) == b""
            assert first.recv(4096) == b""


def test_competing_complete_attachment_never_replaces_incumbent(server):
    srv, cmd_port = server
    with connect(cmd_port) as cmd, connect(srv.data_port):
        cmd.sendall(b"VERSION\r")
        read_until(cmd, b"VERSION")
        generation = srv._pairs.generation
        with connect(cmd_port) as other_cmd, connect(srv.data_port) as other_data:
            assert other_cmd.recv(4096) == b""
            assert other_data.recv(4096) == b""
        assert srv._pairs.generation == generation
        cmd.sendall(b"VERSION\r")
        read_until(cmd, b"VERSION")


def test_ardop_output_blocks_preserve_bytes_with_configured_limit():
    srv = HostServer(control_port=0, data_port=0, output_block_limit=16, quiet=True)
    srv.start()
    try:
        with connect(srv.control_port) as cmd, connect(srv.data_port) as data:
            cmd.sendall(b"VERSION\r")
            read_until(cmd, b"VERSION")
            original = bytes(range(256))
            srv.modem_data_received("ARQ", original)
            received = bytearray()
            for _ in range(16):
                hdr = HostServer._recv_exactly(data, 2)
                length = struct.unpack(">H", hdr)[0]
                block = HostServer._recv_exactly(data, length)
                assert length == 19 and block[:3] == b"ARQ"
                received.extend(block[3:])
            assert bytes(received) == original
    finally:
        srv.stop()


def test_bounded_output_fails_attachment_without_blocking_producer():
    from hfmodem.core.hostio import Connection, MAX_OUTPUT
    client, peer = socket.socketpair()
    conn = Connection(peer, nodelay=False)
    try:
        assert not conn.send_bounded(b"x" * (MAX_OUTPUT + 1))
        assert conn.closed
        assert client.recv(1) == b""
    finally:
        conn.close()
        client.close()


def test_old_ardop_echo_cannot_enter_a_successive_attachment():
    from hfmodem.besra.host.modem_core import LoopbackModem
    modem = LoopbackModem()
    modem.OTA_DELAY = .2
    srv = HostServer(modem, control_port=0, data_port=0, quiet=True)
    srv.start()
    try:
        with connect(srv.control_port) as cmd, connect(srv.data_port) as data:
            cmd.sendall(b"MYCALL W9SSJ\rARQCALL K7ABC 5\r")
            read_until(cmd, b"CONNECTED K7ABC")
            data.sendall(struct.pack(">H", 3) + b"old")
            read_until(cmd, b"BUFFER 3")
            data.shutdown(socket.SHUT_RDWR)
            data.close()
            read_until_eof = bytearray()
            while block := cmd.recv(4096):
                read_until_eof.extend(block)
        until(lambda: srv._pairs._active is None)
        with connect(srv.control_port) as cmd, connect(srv.data_port) as data:
            cmd.sendall(b"MYCALL W9SSJ\rARQCALL K7ABC 5\r")
            read_until(cmd, b"CONNECTED K7ABC")
            data.sendall(struct.pack(">H", 3) + b"new")
            hdr = HostServer._recv_exactly(data, 2)
            block = HostServer._recv_exactly(data, struct.unpack(">H", hdr)[0])
            assert block == b"ARQnew"
    finally:
        srv.stop()
