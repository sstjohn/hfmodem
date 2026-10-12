# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Station host openings over IPv4 TCP, with the default bind unchanged."""
import ctypes
import errno
import ipaddress
import socket
import struct
import sys
import threading
import time
from contextlib import ExitStack

import pytest

KINDS = ("bateleur", "ale", "selcall", "sis")
PORTS = (("bateleur", 0), ("bateleur", 1), ("ale", 0), ("selcall", 0), ("sis", 0))
TIMEOUT = 10.0


@pytest.fixture
def tcp_host(kind):
    from hfmodem.tests.core.test_station_hosts import _server
    server = _server(kind)
    try:
        (getattr(server, "start_background", None) or server.start)()
        deadline = time.monotonic() + TIMEOUT
        while not server._running:
            assert time.monotonic() < deadline, "accept loop did not start"
            time.sleep(0.005)
        yield server
    finally:
        server.stop()


def _listeners(kind, server):
    if kind == "bateleur":
        return tuple(server._srv)
    return (server._srv,)


def _connect(listener):
    conn = socket.create_connection(("127.0.0.1", listener.getsockname()[1]), timeout=TIMEOUT)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn


def _exact(conn, expected):
    deadline = time.monotonic() + TIMEOUT
    got = bytearray()
    while len(got) < len(expected):
        left = deadline - time.monotonic()
        assert left > 0, bytes(got)
        conn.settimeout(left)
        chunk = conn.recv(len(expected) - len(got))
        assert chunk, bytes(got)
        got.extend(chunk)
    assert got == expected


def _exchange(kind, server, conns):
    cmd = conns[0]
    payload = b"hello\x00\xff\r\n"
    if kind != "sis":
        cmd.sendall(b"VERSION\r")
        version = {"bateleur": b"bateleur 1", "ale": b"bateleur-ale 1",
                   "selcall": b"hfmodem-selcall 1"}[kind]
        _exact(cmd, b"OK " + version + b"\r\n")
        if kind == "bateleur":
            conns[1].sendall(payload)
            deadline = time.monotonic() + TIMEOUT
            while server._txbuf != payload:
                assert time.monotonic() < deadline
                time.sleep(.005)
            server.rx_data(payload)
            _exact(conns[1], payload)
    else:
        from hfmodem.bateleur.s5066 import primitives as P
        cmd.sendall(P.encode(P.prim(P.BIND_REQUEST, sap=6,
                                    service=P.ServiceType(P.MODE_NON_ARQ))))
        _exact(cmd, P.encode(P.prim(P.BIND_ACCEPTED, sap=6, mtu=2048)))
        keepalive = P.encode(P.prim(P.KEEP_ALIVE))
        cmd.sendall(keepalive)
        _exact(cmd, keepalive)
        cmd.sendall(P.encode(P.prim(P.UNIDATA_REQUEST, dest_sap=6, dest=2,
                                    mode=P.ServiceType(P.MODE_DEFAULT), u_pdu=payload)))
        cmd.sendall(keepalive)
        _exact(cmd, keepalive)
        assert [out.u_pdu for out in server.node.sis.queue] == [payload]
        cmd.sendall(P.encode(P.prim(P.UNBIND_REQUEST)) + keepalive)
        _exact(cmd, keepalive)
        assert 6 not in server.node.sis.clients


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("delay", [False, True], ids=["immediate", "idle-first"])
def test_tcp_stock_opening_and_data(kind, delay, tcp_host, monkeypatch):
    from hfmodem.core.hostio import Connection
    accepted = []
    condition = threading.Condition()
    original = Connection.__init__
    ports = {s.getsockname()[1] for s in _listeners(kind, tcp_host)}

    def accept(conn, sock, **kwargs):
        port = sock.getsockname()[1]
        original(conn, sock, **kwargs)
        if port in ports:
            with condition:
                accepted.append(conn)
                condition.notify_all()

    monkeypatch.setattr(Connection, "__init__", accept)
    with ExitStack() as stack:
        conns = [stack.enter_context(_connect(s)) for s in _listeners(kind, tcp_host)]
        with condition:
            assert condition.wait_for(lambda: len(accepted) == len(conns), TIMEOUT)
        if delay:
            # Cross the host reader's 0.5 s polling interval without sending a byte.
            time.sleep(0.65)
        _exchange(kind, tcp_host, conns)


@pytest.mark.parametrize("kind", KINDS)
def test_tcp_default_bound_address(kind, tcp_host):
    for listener in _listeners(kind, tcp_host):
        assert listener.family == socket.AF_INET
        assert listener.getsockname()[0] == "127.0.0.1"
        with _connect(listener) as conn:
            assert conn.getpeername() == listener.getsockname()


def _non_loopback_addresses():
    class Ifaddrs(ctypes.Structure):
        pass
    Ifaddrs._fields_ = [("next", ctypes.POINTER(Ifaddrs)), ("name", ctypes.c_char_p),
                       ("flags", ctypes.c_uint), ("addr", ctypes.c_void_p),
                       ("netmask", ctypes.c_void_p), ("destination", ctypes.c_void_p),
                       ("data", ctypes.c_void_p)]
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "getifaddrs"):
        pytest.skip("interface enumeration requires getifaddrs")
    libc.getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(Ifaddrs))]
    libc.getifaddrs.restype = ctypes.c_int
    libc.freeifaddrs.argtypes = [ctypes.POINTER(Ifaddrs)]
    libc.freeifaddrs.restype = None
    head = ctypes.POINTER(Ifaddrs)()
    if libc.getifaddrs(ctypes.byref(head)):
        raise OSError(ctypes.get_errno(), "getifaddrs")
    addresses = set()
    try:
        entry = head
        while entry:
            item = entry.contents
            if item.addr and item.flags & 1:
                raw = ctypes.string_at(item.addr, 8)
                family = raw[1] if sys.platform == "darwin" else int.from_bytes(raw[:2], sys.byteorder)
                if family == socket.AF_INET:
                    address = ipaddress.ip_address(raw[4:8])
                    if not address.is_loopback and not address.is_unspecified:
                        addresses.add(str(address))
            entry = item.next
    finally:
        libc.freeifaddrs(head)
    if not addresses:
        pytest.skip("no active non-loopback IPv4 interface")
    return sorted(addresses)


@pytest.mark.parametrize("kind", KINDS)
def test_tcp_non_loopback_interface_cannot_connect(kind, tcp_host):
    for address in _non_loopback_addresses():
        for listener in _listeners(kind, tcp_host):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
                conn.settimeout(TIMEOUT)
                result = conn.connect_ex((address, listener.getsockname()[1]))
                assert result in (errno.ECONNREFUSED, errno.ENETUNREACH,
                                  errno.EHOSTUNREACH, errno.ETIMEDOUT, errno.EAGAIN), (address, result)


def _observe_dispatch(monkeypatch, kind, server):
    executed = []

    def observe(obj, method):
        original = getattr(obj, method)
        def call(*args, **kwargs):
            executed.append((method, args))
            return original(*args, **kwargs)
        monkeypatch.setattr(obj, method, call)

    if kind != "sis":
        observe(server, "command")
        body = b"VERSION\r"
    else:
        from hfmodem.bateleur.s5066 import primitives as P
        original = server.node.attach
        def attach(send):
            attachment = original(send)
            observe(attachment, "handle")
            return attachment
        monkeypatch.setattr(server.node, "attach", attach)
        body = P.encode(P.prim(P.BIND_REQUEST, sap=6, service=P.ServiceType(P.MODE_NON_ARQ)))
    return executed, body


HTTP_PORTS = [(kind, channel, paired) for kind, channel in PORTS
              for paired in ([False, True] if kind == "bateleur" else [False])]


@pytest.mark.parametrize("kind,channel,paired", HTTP_PORTS)
@pytest.mark.parametrize("request_line", [b"GET / HTTP/1.0", b"POST / HTTP/1.1",
                                         b"OPTIONS / HTTP/1.1"])
@pytest.mark.parametrize("fragmented", [False, True], ids=["whole", "fragmented"])
def test_tcp_http_opening_closes_without_dispatch(
        kind, channel, paired, request_line, fragmented, tcp_host, monkeypatch):
    executed, body = _observe_dispatch(monkeypatch, kind, tcp_host)
    listeners = _listeners(kind, tcp_host)
    if channel == 1:
        body = b"message"
    authority = f"127.0.0.1:{listeners[channel].getsockname()[1]}".encode()
    request = (request_line + b"\r\nHost: " + authority + b"\r\nOrigin: http://example.test\r\n"
               b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    from hfmodem.core.hostio import Opening
    first_seen = threading.Event()
    original = Opening.push

    def push(opening, chunk):
        data = original(opening, chunk)
        if bytes(opening.pending) == request[:1]:
            first_seen.set()
        return data

    if fragmented:
        monkeypatch.setattr(Opening, "push", push)
    with ExitStack() as stack:
        conn = stack.enter_context(_connect(listeners[channel]))
        if paired:
            stack.enter_context(_connect(listeners[1 - channel]))
        if fragmented:
            conn.sendall(request[:1])
            assert first_seen.wait(TIMEOUT), "first HTTP fragment was not retained"
            conn.sendall(request[1:3])
            conn.sendall(request[3:])
        else:
            conn.sendall(request)
        got = bytearray()
        deadline = time.monotonic() + TIMEOUT
        while True:
            left = deadline - time.monotonic()
            assert left > 0, "HTTP connection stayed open"
            conn.settimeout(left)
            try:
                chunk = conn.recv(4096)
            except ConnectionResetError:
                break
            if not chunk:
                break
            got.extend(chunk)
        assert bytes(got) == b""
    assert executed == []
    if kind == "bateleur":
        assert tcp_host._txbuf == b""


@pytest.mark.parametrize("shutdown_first", [False, True])
def test_tcp_error_line_arrives_before_eof(shutdown_first, monkeypatch):
    from hfmodem.core import hostio
    write, closing = threading.Event(), threading.Event()
    original = hostio.Connection._write

    def delayed_writer(conn):
        assert write.wait(TIMEOUT)
        original(conn)

    monkeypatch.setattr(hostio.Connection, "_write", delayed_writer)
    with hostio.listen("127.0.0.1", 0) as listener, _connect(listener) as client:
        sock, _ = listener.accept()
        conn = hostio.Connection(sock)

        def reply_then_close():
            conn.sendall(b"ERROR bad command\r\n")
            closing.set()
            if shutdown_first:
                conn.shutdown(socket.SHUT_RDWR)
            conn.close()

        closer = threading.Thread(target=reply_then_close)
        closer.start()
        try:
            assert closing.wait(TIMEOUT)
            closer.join(.02)
            write.set()
            _exact(client, b"ERROR bad command\r\n")
            assert client.recv(1) == b""
            closer.join(TIMEOUT)
            assert not closer.is_alive()
        finally:
            write.set()
            conn.close()
            closer.join(TIMEOUT)



@pytest.mark.parametrize("kind", ["sis"])
@pytest.mark.parametrize("resume", [True, False], ids=["resume", "disconnect"])
def test_tcp_sis_large_transfer_survives_a_stalled_reader(kind, resume, tcp_host):
    from hfmodem.bateleur.s5066 import primitives as P

    payloads = [i.to_bytes(4, "big") + bytes(range(256)) * 7 + bytes(252)
                for i in range(2304)]
    frames = [P.encode(P.prim(P.UNIDATA_INDICATION, priority=0, dest_sap=6, dest=1,
                              mode=P.MODE_NON_ARQ, src_sap=6, src=2, u_pdu=data))
              for data in payloads]
    produced = threading.Event()
    errors = []

    def receive(start, end):
        for data in payloads[start:end]:
            tcp_host.node.sis.spdu_received(2, 1, b"\x00\x66\x00" + data,
                                           expedited=False, arq=False)

    def remainder():
        try:
            receive(256, len(payloads))
        except Exception as exc:
            errors.append(exc)
        finally:
            produced.set()

    with _connect(tcp_host._srv) as client:
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        client.sendall(P.encode(P.prim(P.BIND_REQUEST, sap=6,
                                       service=P.ServiceType(P.MODE_NON_ARQ))))
        _exact(client, P.encode(P.prim(P.BIND_ACCEPTED, sap=6, mtu=2048)))
        conn = next(iter(tcp_host._conns))
        conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
        receive(0, 256)
        _exact(client, b"".join(frames[:256]))
        producer = threading.Thread(target=remainder, name="sis-receive")
        producer.start()
        try:
            assert produced.wait(TIMEOUT), "node reception waited for the SIS client"
            assert errors == []
            deadline = time.monotonic() + TIMEOUT
            while not (conn._q.full() and conn._delivery) and not conn.closed:
                assert time.monotonic() < deadline, "output never reached backpressure"
                time.sleep(.005)
            # Stay stalled beyond the socket's receive/send polling timeout.
            time.sleep(.75)
            assert not conn.closed
            if not resume:
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                client.close()
                deadline = time.monotonic() + TIMEOUT
                while not conn.closed or conn in tcp_host._conns:
                    assert time.monotonic() < deadline, "stalled client stayed connected"
                    time.sleep(.005)
                receive(0, 1)
                return
            _exact(client, b"".join(frames[256:]))
            keepalive = P.encode(P.prim(P.KEEP_ALIVE))
            client.sendall(keepalive)
            _exact(client, keepalive)
            assert not conn.closed
        finally:
            conn.close()
            producer.join(TIMEOUT)
        assert not producer.is_alive()


def test_a_write_error_releases_a_waiting_producer():
    # Over TCP, not a socketpair: Linux AF_UNIX never fails a send blocked
    # behind its own SHUT_WR, while TCP raises EPIPE on every platform.
    from hfmodem.core import hostio
    started, done = threading.Event(), threading.Event()
    with hostio.listen("127.0.0.1", 0) as listener, _connect(listener) as client:
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        sock, _ = listener.accept()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        conn = hostio.Connection(sock)

        def produce():
            started.set()
            conn.sendall(b"x" * (4 * hostio.MAX_OUTPUT))
            done.set()

        producer = threading.Thread(target=produce)
        producer.start()
        try:
            assert started.wait(2)
            assert not done.wait(.1)
            sock.shutdown(socket.SHUT_WR)
            assert done.wait(2)
            assert conn.closed
        finally:
            conn.close()
            producer.join(2)
