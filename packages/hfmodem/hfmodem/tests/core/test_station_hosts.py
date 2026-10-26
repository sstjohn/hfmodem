# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Host openings over socket pairs, without a listening TCP port."""
import queue
import socket
from types import SimpleNamespace

import pytest


class _Socket:
    def __init__(self, sock):
        self.sock = sock
        self.options = []

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def setsockopt(self, level, option, value):
        self.options.append((level, option, value))
        if level != socket.IPPROTO_TCP:
            self.sock.setsockopt(level, option, value)


class _Listener:
    def __init__(self, port):
        self.port = port
        self.pending = queue.Queue()
        self.timeout = .2
        self.sockets = []

    def getsockname(self):
        return "127.0.0.1", self.port

    def settimeout(self, timeout):
        self.timeout = timeout

    def accept(self):
        try:
            conn = self.pending.get(timeout=self.timeout)
        except queue.Empty:
            raise socket.timeout from None
        if conn is None:
            raise OSError("listener closed")
        return conn, ("127.0.0.1", 1)

    def connect(self):
        client, server = socket.socketpair()
        client.settimeout(2)
        server = _Socket(server)
        self.sockets.append(server)
        self.pending.put(server)
        return _Socket(client)

    def shutdown(self, how):
        self.close()

    def close(self):
        self.pending.put(None)


def _server(kind):
    if kind == "bateleur":
        from hfmodem.bateleur.host import BateleurHost
        return BateleurHost(SimpleNamespace(abort=lambda: None), cmd_port=0, data_port=0)
    if kind == "ale":
        from hfmodem.bateleur.ale.host import AleHost
        ctl = SimpleNamespace(cfg=SimpleNamespace(channels=["A"]))
        return AleHost(SimpleNamespace(ctl=ctl), port=0)
    if kind == "selcall":
        from hfmodem.bateleur.selcall.host import SelcallHost
        return SelcallHost(SimpleNamespace(), port=0)
    if kind == "sis":
        from hfmodem.bateleur.s5066.node import Node, Timing
        from hfmodem.bateleur.s5066.server import SisServer
        return SisServer(Node(1, Timing(1200)), lambda fn: fn(), port=0)
    raise ValueError(kind)


def _closed(conn):
    try:
        assert conn.recv(4096) == b""
    except ConnectionResetError:
        pass
    finally:
        conn.close()


KINDS = ["bateleur", "ale", "selcall", "sis"]


def socketpair_network(monkeypatch):
    listeners = []
    raw_socket = socket.socket
    raw_connect = socket.create_connection

    class Listener(_Listener):
        def bind(self, address):
            assert address[0] == "127.0.0.1"
            if address[1]:
                self.port = address[1]

        def listen(self, backlog):
            pass

        def setsockopt(self, *args):
            pass

    def make_socket(family=socket.AF_INET, type=socket.SOCK_STREAM, *args, **kwargs):
        if family != socket.AF_INET:
            return raw_socket(family, type, *args, **kwargs)
        srv = Listener(20000 + len(listeners))
        listeners.append(srv)
        return srv

    def connect(address, *args, **kwargs):
        for listener in listeners:
            if address == ("127.0.0.1", listener.port):
                return listener.connect()
        return raw_connect(address, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", make_socket)
    monkeypatch.setattr(socket, "create_connection", connect)
    return listeners


@pytest.fixture
def listeners(monkeypatch):
    return socketpair_network(monkeypatch)


def _start(srv):
    (getattr(srv, "start_background", None) or srv.start)()


def _read(conn, expected):
    result = bytearray()
    while len(result) < len(expected):
        chunk = conn.recv(len(expected) - len(result))
        assert chunk
        result.extend(chunk)
    assert result == expected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fragmented", [False, True])
def test_http_post_with_valid_command_is_refused_on_every_port(kind, fragmented, listeners):
    srv = _server(kind)
    executed = []
    if kind in ("bateleur", "ale", "selcall"):
        srv.command = lambda line: executed.append(line) or "OK"
        body = b"VERSION\r\n"
    elif kind == "sis":
        from hfmodem.bateleur.s5066 import primitives as P
        original = srv.node.attach
        def attach(send):
            att = original(send)
            att.handle = lambda p: executed.append(p)
            return att
        srv.node.attach = attach
        body = P.encode(P.prim(P.BIND_REQUEST, sap=6, service=P.ServiceType(P.MODE_NON_ARQ)))
    _start(srv)
    try:
        for index, listener in enumerate(listeners):
            conn = listener.connect()
            payload = body
            if index == 1:
                payload = b"message"
            request = (b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: " +
                       str(len(payload)).encode() + b"\r\n\r\n" + payload)
            try:
                if fragmented:
                    conn.sendall(b"P")
                    conn.sendall(b"OS")
                    conn.sendall(request[3:])
                else:
                    conn.sendall(request)
                _closed(conn)
            finally:
                conn.close()
        assert executed == []
        if kind == "bateleur":
            assert srv._txbuf == b""
    finally:
        srv.stop()


@pytest.mark.parametrize("kind", KINDS)
def test_normal_opening_without_prelude(kind, listeners):
    srv = _server(kind)
    _start(srv)
    conns = [listener.connect() for listener in listeners]
    try:
        cmd = conns[0]
        if kind == "sis":
            from hfmodem.bateleur.s5066 import primitives as P
            cmd.sendall(P.encode(P.prim(P.BIND_REQUEST, sap=6,
                                        service=P.ServiceType(P.MODE_NON_ARQ))))
            _read(cmd, P.encode(P.prim(P.BIND_ACCEPTED, sap=6, mtu=2048)))
            good = P.encode(P.prim(P.KEEP_ALIVE))
            cmd.sendall(good)
            _read(cmd, good)
        else:
            cmd.sendall(b"VERSION\r")
            expected = {"bateleur": b"OK bateleur 1\r\n",
                        "ale": b"OK bateleur-ale 1\r\n", "selcall": b"OK hfmodem-selcall"}[kind]
            _read(cmd, expected)
        if kind == "bateleur":
            conns[1].sendall(b"binary\x00\xff")
            import time
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with srv._lock:
                    if srv._txbuf == b"binary\x00\xff":
                        break
                time.sleep(.005)
            assert srv._txbuf == b"binary\x00\xff"
    finally:
        for conn in conns:
            conn.close()
        srv.stop()


def test_sis_stop_waits_for_attachment_cleanup(monkeypatch):
    import threading
    import time
    from hfmodem.core import hostio
    from hfmodem.bateleur.s5066.server import SisServer
    listener = _Listener(20000)
    monkeypatch.setattr(hostio, "listen", lambda *args, **kwargs: listener)
    attached = threading.Event()
    closed = []
    def close():
        time.sleep(.05)
        closed.append(True)
    def attach(send):
        attached.set()
        return SimpleNamespace(close=close)
    srv = SisServer(SimpleNamespace(attach=attach), lambda fn: fn())
    srv.start_background()
    conn = listener.connect()
    try:
        assert attached.wait(2)
        srv.stop()
        assert closed
    finally:
        conn.close()
        srv.stop()


def test_peer_close_cancels_command_waiting_on_modem():
    import threading
    from hfmodem.core import hostio
    from hfmodem.bateleur.host import BateleurHost
    pending = []
    ready = threading.Event()
    def post(fn):
        pending.append(fn)
        ready.set()
    modem = SimpleNamespace(post=post, listen=True)
    srv = BateleurHost(modem)
    srv._running = True
    client, peer = socket.socketpair()
    conn = hostio.Connection(_Socket(peer))
    worker = threading.Thread(target=srv._serve_cmd, args=(conn,))
    worker.start()
    try:
        client.sendall(b"LISTEN OFF\r\n")
        assert ready.wait(2)
        client.close()
        worker.join(2)
        assert not worker.is_alive()
        pending[0]()
        assert modem.listen
    finally:
        client.close()
        conn.close()
        worker.join(2)


def test_data_server_tolerates_a_connection_replaced_before_it_runs():
    from hfmodem.core import hostio
    from hfmodem.bateleur.host import BateleurHost
    srv = BateleurHost(SimpleNamespace())
    srv._running = True
    client, peer = socket.socketpair()
    conn = hostio.Connection(_Socket(peer))
    conn.close()
    try:
        srv._serve_data(conn)
    finally:
        client.close()


@pytest.mark.parametrize("line", [False, True])
@pytest.mark.parametrize("opening_bytes", [b"POST / HTTP/1.1\r\n\r\nVERSION\r",
                                     b"GET /", b"OPTIONS * HTTP/1.0\r\n"])
def test_http_opening_is_rejected_at_every_fragment_boundary(line, opening_bytes):
    from hfmodem.core.hostio import Opening
    for split in range(1, len(opening_bytes)):
        opening = Opening(line=line)
        with pytest.raises(ConnectionAbortedError):
            assert opening.push(opening_bytes[:split]) == b""
            opening.push(opening_bytes[split:])


@pytest.mark.parametrize("opening_bytes", [b"CUSTOM target HTTP/1.1\r\nVERSION\r",
                                     b"WHATEVER /path HTTP/1.0\r\nVERSION\r"])
def test_http_extension_request_never_reaches_command_parser(opening_bytes):
    from hfmodem.core.hostio import lines
    for split in range(1, len(opening_bytes)):
        chunks = iter([opening_bytes[:split], opening_bytes[split:], b""])
        conn = SimpleNamespace(recv=lambda size: next(chunks))
        assert list(lines(conn, lambda: True)) == []


@pytest.mark.parametrize("payload", [b"\x00", b"\x90\xeb", b"binary", b"POSTCARD", b"hello"])
def test_binary_opening_needs_no_line_ending(payload):
    from hfmodem.core.hostio import Connection
    client, peer = socket.socketpair()
    conn = Connection(_Socket(peer))
    try:
        peer.settimeout(.2)
        client.sendall(payload)
        assert conn.recv(65536) == payload
        client.sendall(b"POST / HTTP/1.1\r\n")
        assert conn.recv(65536) == b"POST / HTTP/1.1\r\n"
    finally:
        conn.close()
        client.close()


def test_ale_allcall_suffix_is_a_normal_first_command():
    from hfmodem.core.hostio import lines
    chunks = iter([b"ALLCALL /X message\r", b""])
    conn = SimpleNamespace(recv=lambda size: next(chunks))
    assert list(lines(conn, lambda: True)) == ["ALLCALL /X message"]


@pytest.mark.parametrize("shutdown_first", [False, True])
def test_close_drains_the_queued_reply(shutdown_first, monkeypatch):
    import threading
    from hfmodem.core.hostio import Connection
    write, closing = threading.Event(), threading.Event()
    original = Connection._write

    def delayed_writer(conn):
        assert write.wait(2)
        original(conn)

    monkeypatch.setattr(Connection, "_write", delayed_writer)
    client, peer = socket.socketpair()
    client.settimeout(2)
    conn = Connection(_Socket(peer))

    def reply_then_close():
        conn.sendall(b"ERROR bad command\r\n")
        closing.set()
        if shutdown_first:
            conn.shutdown(socket.SHUT_RDWR)
        conn.close()

    closer = threading.Thread(target=reply_then_close)
    closer.start()
    try:
        assert closing.wait(2)
        write.set()
        _read(client, b"ERROR bad command\r\n")
        assert client.recv(1) == b""
        closer.join(2)
        assert not closer.is_alive()
    finally:
        write.set()
        conn.close()
        client.close()
        closer.join(2)


def test_close_has_a_bound_when_the_writer_is_stalled():
    import threading
    import time
    from hfmodem.core.hostio import Connection
    writing, release = threading.Event(), threading.Event()

    class StalledSocket(_Socket):
        def send(self, data):
            writing.set()
            assert release.wait(2)
            return len(data)

    client, peer = socket.socketpair()
    client.settimeout(2)
    conn = Connection(StalledSocket(peer))
    try:
        conn.sendall(b"reply")
        assert writing.wait(2)
        start = time.monotonic()
        conn.close()
        assert time.monotonic() - start < 1
        assert conn.closed
        assert client.recv(1) == b""
    finally:
        release.set()
        conn.close()
        client.close()
        conn._writer.join(2)
    assert not conn._writer.is_alive()


@pytest.mark.parametrize("limit", ["bytes", "writes"])
def test_output_waits_for_capacity_without_closing(limit, monkeypatch):
    import threading
    from hfmodem.core import hostio
    release, full, done = (threading.Event() for _ in range(3))
    original = hostio.Connection._write

    def writer(conn):
        assert release.wait(5)
        original(conn)

    monkeypatch.setattr(hostio.Connection, "_write", writer)
    client, peer = socket.socketpair()
    client.settimeout(5)
    conn = hostio.Connection(_Socket(peer))
    first = b"a" * hostio.MAX_OUTPUT if limit == "bytes" else b"a"
    count = 1 if limit == "bytes" else hostio.MAX_PENDING
    last = bytes(range(256)) * 2049

    def produce():
        for _ in range(count):
            conn.sendall(first)
        full.set()
        conn.sendall(last)
        done.set()

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        assert full.wait(2)
        assert not done.wait(.05)
        assert not conn.closed
        release.set()
        _read(client, first * count + last)
        assert done.wait(2)
        assert not conn.closed
    finally:
        release.set()
        conn.close()
        client.close()
        producer.join(2)
    assert not producer.is_alive()


def test_partial_write_timeout_retries_only_the_unwritten_bytes():
    import threading
    from hfmodem.core.hostio import Connection
    finished = threading.Event()
    got = bytearray()

    class Socket:
        calls = 0
        def settimeout(self, value): pass
        def setsockopt(self, *args): pass
        def shutdown(self, *args): pass
        def close(self): pass
        def sendall(self, data):
            view = memoryview(data)
            while view:
                view = view[self.send(view):]
        def send(self, data):
            self.calls += 1
            if self.calls == 2:
                raise socket.timeout
            n = min(3, len(data))
            got.extend(data[:n])
            if got == b"abcdefghij":
                finished.set()
            return n

    conn = Connection(Socket())
    try:
        conn.sendall(b"abcdefghij")
        assert finished.wait(2)
        assert not conn.closed
        assert got == b"abcdefghij"
    finally:
        conn.close()


@pytest.mark.parametrize("ending", ["close", "disconnect", "http"])
def test_closing_releases_a_waiting_producer(ending):
    import threading
    from hfmodem.core.hostio import Connection, MAX_OUTPUT
    started, done = threading.Event(), threading.Event()
    client, peer = socket.socketpair()
    peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    conn = Connection(_Socket(peer))

    def produce():
        started.set()
        conn.sendall(b"x" * (4 * MAX_OUTPUT))
        done.set()

    def read():
        try:
            conn.recv(1024)
        except OSError:
            pass
        finally:
            conn.close()

    producer = threading.Thread(target=produce)
    reader = threading.Thread(target=read)
    producer.start()
    reader.start()
    try:
        assert started.wait(2)
        assert not done.wait(.1)
        if ending == "close":
            conn.close()
        elif ending == "disconnect":
            client.close()
        else:
            client.sendall(b"GET / HTTP/1.1\r\n\r\n")
        assert done.wait(2)
        producer.join(2)
        reader.join(2)
        assert conn.closed
        assert not reader.is_alive()
    finally:
        conn.close()
        client.close()
        producer.join(2)
        reader.join(2)


@pytest.mark.parametrize("kind", ["bateleur", "ale", "selcall", "sis"])
def test_stalled_host_delivery_does_not_block_modem_work(kind, listeners, monkeypatch):
    import threading
    from hfmodem.bateleur.worker import ModemWorker
    from hfmodem.bateleur.s5066 import primitives as P
    from hfmodem.bateleur.s5066.node import Node, Timing
    from hfmodem.bateleur.s5066.server import SisServer
    from hfmodem.bateleur.ale.controller import Event as AleEvent
    from hfmodem.bateleur.selcall.modem import Event as SelcallEvent

    class Worker(ModemWorker):
        card_rate = 48000
        _log = staticmethod(lambda *args: None)
        def _audio(self, samples):
            audio_seen.set()
        def abort(self): pass

    stall, writing, release, progressed, audio_seen = (threading.Event() for _ in range(5))
    original = _Socket.__getattr__

    def lookup(sock, name):
        if name in ("send", "sendall"):
            def send(data):
                if stall.is_set():
                    writing.set()
                    assert release.wait(5)
                return getattr(sock.sock, name)(data)
            return send
        return original(sock, name)

    monkeypatch.setattr(_Socket, "__getattr__", lookup)
    worker = Worker()
    worker._init_worker()
    worker.start()
    if kind == "sis":
        server = SisServer(Node(1, Timing(1200)), worker.post, port=0)
    else:
        server = _server(kind)
    _start(server)
    conns = [listener.connect() for listener in listeners]
    try:
        if kind == "sis":
            conns[0].sendall(P.encode(P.prim(P.BIND_REQUEST, sap=6,
                                           service=P.ServiceType(P.MODE_NON_ARQ))))
            _read(conns[0], P.encode(P.prim(P.BIND_ACCEPTED, sap=6, mtu=2048)))
            conn = next(iter(server._conns))
            payload = bytes(range(256)) * 8
            def emit():
                server.node.sis.spdu_received(2, 1, b"\x00\x66\x00" + payload,
                                             expedited=False, arq=False)
            expected = P.encode(P.prim(P.UNIDATA_INDICATION, priority=0, dest_sap=6,
                                       dest=1, mode=P.MODE_NON_ARQ, src_sap=6, src=2,
                                       u_pdu=payload))
        else:
            conns[0].sendall(b"VERSION\r")
            version = {"bateleur": b"bateleur 1", "ale": b"bateleur-ale 1",
                       "selcall": b"hfmodem-selcall 1"}[kind]
            _read(conns[0], b"OK " + version + b"\r\n")
            if kind == "bateleur":
                conns[1].sendall(b"ready")
                import time
                deadline = time.monotonic() + 2
                while server._txbuf != b"ready":
                    assert time.monotonic() < deadline
                    time.sleep(.005)
                conn = server._data
                expected = bytes(range(256)) * 8
                emit = lambda: server.rx_data(expected)
            elif kind == "ale":
                conn = server._conn
                emit = lambda: server.event(AleEvent("SOUNDED", channel=0))
                expected = b"SOUNDED CHANNEL A\r\n"
            else:
                conn = server._conn
                emit = lambda: server.event(SelcallEvent("SEND_FAILED", None, "stalled"))
                expected = b"SEND FAILED  stalled\r\n"
        stall.set()
        worker.post(lambda: [emit() for _ in range(256)])
        worker.post(progressed.set)
        worker._feed(([0],))
        assert writing.wait(2)
        assert progressed.wait(2), "modem command waited for host output"
        assert audio_seen.wait(2), "modem audio waited for host output"
        assert not conn.closed
        release.set()
        _read(conns[1] if kind == "bateleur" else conns[0], expected * 256)
        assert not conn.closed
    finally:
        release.set()
        for client in conns:
            client.close()
        server.stop()
        worker.stop()


def test_close_drains_output_handed_off_by_the_modem():
    from hfmodem.core.hostio import Connection
    client, peer = socket.socketpair()
    client.settimeout(2)
    conn = Connection(_Socket(peer))
    try:
        conn.send_later(b"final indication")
        conn.close()
        _read(client, b"final indication")
        assert client.recv(1) == b""
        conn._delivery_thread.join(2)
        assert not conn._delivery_thread.is_alive()
    finally:
        conn.close()
        client.close()


def test_socketpair_stall_retains_every_byte_after_send_timeout():
    import threading
    from hfmodem.core.hostio import Connection
    client, peer = socket.socketpair()
    client.settimeout(5)
    peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    conn = Connection(_Socket(peer))
    payload = bytes(range(256)) * 16384
    done = threading.Event()

    def produce():
        conn.sendall(payload)
        done.set()

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        assert not done.wait(.75)
        assert not conn.closed
        _read(client, payload)
        assert done.wait(2)
        assert not conn.closed
    finally:
        conn.close()
        client.close()
        producer.join(2)
    assert not producer.is_alive()
