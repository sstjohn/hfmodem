# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import errno
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from hfmodem.core import hostio
from hfmodem.core.operations import LeaseTimeout
from hfmodem.bateleur.worker import ModemWorker
from hfmodem.bateleur.modem import BateleurModem, Mode
from hfmodem.bateleur.ale.modem import AleModem
from hfmodem.bateleur.ale.controller import Config
from hfmodem.bateleur.selcall.modem import SelcallModem
from hfmodem.bateleur.s5066.onair import OnAir
from hfmodem.bateleur.s5066.server import NodeService
from hfmodem.brubru.host.server import KissServer
from hfmodem.station.link import BateleurLink, AleLink, SelcallLink
from hfmodem.tests.core.test_station_hosts import _Socket, _server, socketpair_network


@pytest.fixture(autouse=True)
def loopback_bind(monkeypatch):
    bind = socket.socket.bind

    def checked(sock, address):
        try:
            return bind(sock, address)
        except OSError as exc:
            if sock.family == socket.AF_INET and address == ('127.0.0.1', 0) \
                    and exc.errno in (errno.EPERM, errno.EACCES):
                sock.close()
                pytest.skip(f'sandbox refuses loopback bind: {exc}')
            raise

    monkeypatch.setattr(socket.socket, 'bind', checked)


class Pin:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread = None

    def __call__(self, *args):
        self.thread = threading.current_thread()
        self.entered.set()
        assert self.release.wait(10), 'test did not release its worker'
        return 'OK'


def bounded_false(stop):
    start = time.monotonic()
    assert stop(.02) is False
    assert time.monotonic() - start < .5


def connection_threads(conn):
    return [t for t in (conn._writer, conn._delivery_thread, conn._opening_thread) if t]


def host_threads(server):
    threads = list(server._threads) if hasattr(server, '_threads') else [server._thread, *server._workers]
    for conn in server._connections:
        threads.extend(connection_threads(conn))
    return threads


@pytest.mark.parametrize('method', ['_write', '_deliver', '_read_opening'])
@pytest.mark.parametrize('pinned', [False, True])
def test_connection_keeps_and_checks_every_thread(monkeypatch, method, pinned):
    pin = Pin()
    if not pinned:
        pin.release.set()
    original = getattr(hostio.Connection, method)

    def run(conn):
        pin()
        original(conn)

    monkeypatch.setattr(hostio.Connection, method, run)
    client, peer = socket.socketpair()
    conn = hostio.Connection(peer, eager=True, nodelay=False)
    try:
        assert pin.entered.wait(2)
        start = time.monotonic()
        conn.close()
        assert time.monotonic() - start < .5
        assert conn.closed
        assert conn._ready.is_set()
        if pinned:
            bounded_false(conn.join)
            assert pin.thread in connection_threads(conn)
            assert pin.thread.is_alive()
            bounded_false(conn.join)
        else:
            assert conn.join(2) is True
    finally:
        pin.release.set()
        conn.close()
        client.close()
        assert conn.join(2) is True
    assert not any(t.is_alive() for t in connection_threads(conn))
    assert conn.join(0) is True


def test_connection_opening_is_cancelled_while_waiting_for_first_bytes():
    reading = threading.Event()
    client, peer = socket.socketpair()

    class Socket(_Socket):
        def recv(self, size):
            reading.set()
            return self.sock.recv(size)

    conn = hostio.Connection(Socket(peer), eager=True, line=True)
    try:
        assert reading.wait(2)
        assert not conn._ready.is_set()
        assert conn._opening_thread.is_alive()
        conn.close()
        assert conn.join(2) is True
        assert conn.recv(1) == b''
    finally:
        conn.close()
        client.close()
        assert conn.join(2)


@pytest.mark.parametrize('kind', ['bateleur', 'ale', 'selcall', 'sis'])
@pytest.mark.parametrize('pinned', [False, True])
def test_host_checks_client_handler_and_connection(monkeypatch, kind, pinned):
    listeners = socketpair_network(monkeypatch)
    server = _server(kind)
    pin = Pin()
    if not pinned:
        pin.release.set()
    if kind == 'sis':
        attach = server.node.attach

        def blocked_attach(send):
            pin()
            return attach(send)

        monkeypatch.setattr(server.node, 'attach', blocked_attach)
    else:
        monkeypatch.setattr(server, 'command', pin)
    server.start_background()
    client = listeners[0].connect()
    try:
        client.sendall(b'WAIT\r') if kind != 'sis' else None
        assert pin.entered.wait(2)
        if pinned:
            bounded_false(server.stop)
            assert pin.thread in host_threads(server)
            assert pin.thread.is_alive()
            assert all(conn.closed for conn in server._connections)
            bounded_false(server.stop)
        else:
            assert server.stop(2) is True
    finally:
        pin.release.set()
        client.close()
        assert server.stop(2) is True
    assert not any(t.is_alive() for t in host_threads(server))
    assert server.stop(0) is True


@pytest.mark.parametrize('method', ['_accept_cmd', '_accept_data', '_autosender', '_read_data'])
@pytest.mark.parametrize('pinned', [False, True])
def test_bateleur_checks_accept_autosend_and_data(monkeypatch, method, pinned):
    listeners = socketpair_network(monkeypatch)
    server = _server('bateleur')
    pin = Pin()
    if not pinned:
        pin.release.set()
    original = getattr(server, method)

    def run(*args):
        pin()
        original(*args)

    monkeypatch.setattr(server, method, run)
    server.start_background()
    client = listeners[1].connect() if method == '_read_data' else None
    try:
        assert pin.entered.wait(2)
        if pinned:
            bounded_false(server.stop)
            assert pin.thread in host_threads(server)
            assert pin.thread.is_alive()
        else:
            assert server.stop(2) is True
    finally:
        pin.release.set()
        if client:
            client.close()
        assert server.stop(2) is True
    assert not any(t.is_alive() for t in host_threads(server))


@pytest.mark.parametrize('kind', ['bateleur', 'ale', 'selcall'])
def test_replaced_client_remains_in_teardown(monkeypatch, kind):
    listeners = socketpair_network(monkeypatch)
    server = _server(kind)
    pin, second = Pin(), threading.Event()

    def command(line):
        if line == 'WAIT':
            return pin()
        second.set()
        return 'OK'

    monkeypatch.setattr(server, 'command', command)
    server.start_background()
    first = listeners[0].connect()
    replacement = None
    try:
        first.sendall(b'WAIT\r')
        assert pin.entered.wait(2)
        old = server._connections[0]
        replacement = listeners[0].connect()
        replacement.sendall(b'NEXT\r')
        assert second.wait(2)
        assert old.closed
        bounded_false(server.stop)
        assert old in server._connections
        assert pin.thread in host_threads(server)
    finally:
        pin.release.set()
        first.close()
        if replacement:
            replacement.close()
        assert server.stop(2)


@pytest.mark.parametrize('kind', ['bateleur', 'ale', 'selcall', 'sis'])
def test_host_checks_connection_writer_after_handler_exits(monkeypatch, kind):
    listeners = socketpair_network(monkeypatch)
    pin = Pin()
    original = hostio.Connection._write

    def write(conn):
        pin()
        original(conn)

    monkeypatch.setattr(hostio.Connection, '_write', write)
    server = _server(kind)
    server.start_background()
    client = listeners[0].connect()
    try:
        assert pin.entered.wait(2)
        bounded_false(server.stop)
        assert pin.thread in host_threads(server)
        assert pin.thread.is_alive()
    finally:
        pin.release.set()
        client.close()
        assert server.stop(2)


def make_onair():
    modem = SimpleNamespace(max_key_s=10, post=lambda fn: None)
    return OnAir(modem, 1, Mode('110', '1200S'))


@pytest.mark.parametrize('pinned', [False, True])
def test_onair_checks_ticker(monkeypatch, pinned):
    onair = make_onair()
    pin = Pin()
    if not pinned:
        pin.release.set()
    monkeypatch.setattr(onair.modem, 'post', pin)
    onair.start()
    try:
        assert pin.entered.wait(2)
        if pinned:
            bounded_false(onair.stop)
            assert onair._thread is pin.thread
            assert onair._thread.is_alive()
            bounded_false(onair.stop)
        else:
            assert onair.stop(2) is True
    finally:
        pin.release.set()
        assert onair.stop(2) is True
    assert onair.stop(0) is True


@pytest.mark.parametrize('pinned_part', [None, 'sis', 'onair'])
def test_node_service_checks_both_workers(monkeypatch, pinned_part):
    listeners = socketpair_network(monkeypatch)
    modem = SimpleNamespace(max_key_s=10, post=lambda fn: None)
    service = NodeService(modem, 1, Mode('110', '1200S'), port=0)
    pins = {part: Pin() for part in ('sis', 'onair')}
    for part, pin in pins.items():
        if part != pinned_part:
            pin.release.set()
    monkeypatch.setattr(service.sis, '_serve', pins['sis'])
    monkeypatch.setattr(service.onair, '_ticker', pins['onair'])
    service.start_background()
    client = listeners[0].connect()
    try:
        assert all(pin.entered.wait(2) for pin in pins.values())
        if pinned_part:
            bounded_false(service.stop)
            assert pins[pinned_part].thread.is_alive()
            assert pins[pinned_part].thread in [*host_threads(service.sis), service.onair._thread]
            assert all(conn.closed for conn in service.sis._connections)
            bounded_false(service.stop)
        else:
            assert service.stop(2) is True
    finally:
        for pin in pins.values():
            pin.release.set()
        client.close()
        assert service.stop(2) is True
    assert not any(t.is_alive() for t in [*host_threads(service.sis), service.onair._thread])


def make_worker(kind):
    if kind == 'bateleur':
        return BateleurModem(lambda *a: None, card_rate=48000, max_key_s=10)
    if kind == 'ale':
        return AleModem(Config(['OWN'], ['A']), tune=lambda ch: 0, send=lambda *a: None)
    if kind == 'selcall':
        return SelcallModem(['1234'], send=lambda *a: None)
    worker = ModemWorker()
    worker.card_rate = 48000
    worker._log = lambda *args: None
    worker._init_worker()
    return worker


@pytest.mark.parametrize('kind', ['worker', 'bateleur', 'ale', 'selcall'])
@pytest.mark.parametrize('pinned', [False, True])
def test_modem_stop_checks_worker(kind, pinned):
    worker = make_worker(kind)
    pin = Pin()
    if not pinned:
        pin.release.set()
    worker.start()
    worker.post(pin)
    try:
        assert pin.entered.wait(2)
        if pinned:
            bounded_false(worker.stop)
            assert worker._thread is pin.thread
            assert worker._thread.is_alive()
            bounded_false(worker.stop)
        else:
            assert worker.stop(2) is True
    finally:
        pin.release.set()
        assert worker.stop(2) is True
    assert worker.stop(0) is True


@pytest.mark.parametrize('started', [False, True])
def test_worker_cleanup_is_also_bounded(started):
    worker = make_worker('worker')
    pin = Pin()
    if started:
        worker.start()
    worker.post_cleanup(pin)
    try:
        bounded_false(worker.stop)
        assert pin.entered.wait(2)
        assert worker._thread is pin.thread
        assert worker._thread.is_alive()
    finally:
        pin.release.set()
        assert worker.stop(2) is True


@pytest.mark.parametrize('kind,link_type', [('bateleur', BateleurLink), ('ale', AleLink),
                                           ('selcall', SelcallLink)])
@pytest.mark.parametrize('pinned', [False, True])
def test_lane_checks_actual_modem_stop(monkeypatch, kind, link_type, pinned):
    lane = object.__new__(link_type)
    lane.modem = make_worker(kind)
    rig = SimpleNamespace(max_key_s=.3, ptt=SimpleNamespace(settle_s=.07))
    lane.station = SimpleNamespace(rig=None, arbiter=None, cfg=SimpleNamespace(rig=rig))
    lane.modem._audio_limit = lane.modem.card_rate // 2
    pin = Pin()
    if not pinned:
        pin.release.set()
    stop = lane.modem.stop
    timeouts = []

    def tiny_stop(timeout):
        timeouts.append(timeout)
        return stop(.02 if pinned else 2)

    monkeypatch.setattr(lane.modem, 'stop', tiny_stop)
    lane.start = lane.modem.start
    lane.start()
    lane.modem.post(pin)
    try:
        assert pin.entered.wait(2)
        if pinned:
            with pytest.raises(LeaseTimeout, match=lane.waveform.name):
                lane.stop()
            assert lane.modem._thread is pin.thread
            assert pin.thread.is_alive()
        else:
            assert lane.stop() is None
        assert timeouts == [pytest.approx(.87)]
    finally:
        pin.release.set()
        assert stop(2) is True
    lane.stop()


@pytest.mark.parametrize('pinned', [False, True])
def test_kiss_checks_retained_thread(monkeypatch, pinned):
    from hfmodem.ax25.kiss import encode
    pin = Pin()
    if not pinned:
        pin.release.set()
    link = SimpleNamespace(drop_pending=lambda: None, host=None, enqueue=pin)
    server = KissServer(link)
    client, peer = socket.socketpair()
    wake, listening = socket.socketpair()

    class Listener(_Socket):
        def bind(self, address):
            assert address == ('127.0.0.1', 0)
        def listen(self, backlog):
            pass
        def getsockname(self):
            return '127.0.0.1', 12345
        def accept(self):
            self.sock.recv(1)
            return peer, ('127.0.0.1', 54321)

    monkeypatch.setattr(socket, 'socket', lambda *args: Listener(listening))
    server.start()
    wake.sendall(b'x')
    client.sendall(encode(b'blocked command'))
    try:
        assert pin.entered.wait(2)
        if pinned:
            bounded_false(server.stop)
            assert server._thread is pin.thread
            assert server._thread.is_alive()
            assert peer.fileno() == -1
            assert listening.fileno() == -1
            bounded_false(server.stop)
        else:
            assert server.stop(2) is True
    finally:
        pin.release.set()
        client.close()
        wake.close()
        assert server.stop(2) is True
    assert server.stop(0) is True


@pytest.mark.parametrize('kind', ['connection', 'bateleur', 'ale', 'selcall', 'sis', 'onair', 'kiss', 'worker'])
def test_stop_from_retained_thread_reports_it_alive(monkeypatch, kind):
    done = threading.Event()
    result = []
    go = threading.Event()
    if kind == 'connection':
        def run(conn):
            assert go.wait(2)
            conn.close()
            result.append(conn.join(.02))
            done.set()
        monkeypatch.setattr(hostio.Connection, '_write', run)
        client, peer = socket.socketpair()
        obj = hostio.Connection(peer, nodelay=False)
        stop = obj.join
        go.set()
    else:
        if kind in ('bateleur', 'ale', 'selcall', 'sis'):
            obj = _server(kind)
        elif kind == 'onair':
            obj = make_onair()
        elif kind == 'kiss':
            obj = KissServer(SimpleNamespace(drop_pending=lambda: None, host=None))
        else:
            obj = make_worker('worker')
        stop = obj.stop

        def run():
            result.append(stop(.02))
            done.set()

        thread = threading.Thread(target=run)
        if hasattr(obj, '_threads'):
            obj._threads.append(thread)
        else:
            obj._thread = thread
        thread.start()
    try:
        assert done.wait(2)
        assert result == [False]
    finally:
        if kind == 'connection':
            obj.close()
            client.close()
        assert stop(2) is True


def test_connection_to_nonreading_client_can_be_cancelled():
    client, peer = socket.socketpair()
    peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    writing = threading.Event()

    class Socket(_Socket):
        def send(self, data):
            writing.set()
            return self.sock.send(data)

    conn = hostio.Connection(Socket(peer))
    producer = threading.Thread(target=lambda: conn.sendall(b'x' * (4 * hostio.MAX_OUTPUT)))
    producer.start()
    try:
        assert writing.wait(2)
        conn.close()
        assert conn.join(2) is True
        producer.join(2)
        assert not producer.is_alive()
    finally:
        conn.close()
        client.close()
        producer.join(2)
        assert conn.join(2)


def test_kiss_real_loop_and_nonreading_client(monkeypatch):
    accepted = threading.Event()
    server = KissServer(SimpleNamespace(drop_pending=lambda: None, host=None))
    original = server._accept

    stalled = threading.Event()

    class Socket(_Socket):
        def send(self, data):
            try:
                return self.sock.send(data)
            except BlockingIOError:
                stalled.set()
                raise

    def accept(client):
        client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        result = original(Socket(client))
        accepted.set()
        return result

    monkeypatch.setattr(server, '_accept', accept)
    server.start()
    client = socket.create_connection(('127.0.0.1', server.port), timeout=2)
    try:
        assert accepted.wait(2)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        send = server._send

        def fill(now):
            for _ in range(32):
                server.send_frame(b'x' * 2048)
            for _ in range(32):
                send(now)

        monkeypatch.setattr(server, '_send', fill)
        assert stalled.wait(2)
        assert server.stop(2) is True
        assert not server._thread.is_alive()
    finally:
        client.close()
        assert server.stop(2)


@pytest.mark.parametrize('kind', ['bateleur', 'ale', 'selcall', 'sis'])
def test_host_joins_share_one_deadline_and_close_first(monkeypatch, kind):
    clock = [100.0]
    calls = []
    closed = []
    monkeypatch.setattr(hostio.time, 'monotonic', lambda: clock[0])

    class Thread:
        def join(self, timeout):
            assert closed == ['listener', 'client']
            calls.append(timeout)
            clock[0] += timeout
        def is_alive(self):
            return True

    class Connection(hostio.Connection):
        def __init__(self):
            pass
        def cancel(self):
            closed.append('client')
        def join(self, timeout):
            calls.append(timeout)
            return False

    server = _server(kind)
    listener = SimpleNamespace(close=lambda: closed.append('listener'))
    server._srv = [listener] if kind == 'bateleur' else listener
    server._connections.append(Connection())
    if hasattr(server, '_threads'):
        server._threads.extend([Thread(), Thread()])
    else:
        server._thread = Thread()
        server._workers.add(Thread())
    assert server.stop(.25) is False
    assert calls == [.25, 0.0, 0.0]


def test_node_service_shares_deadline_and_checks_ticker_after_sis_timeout(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(hostio.time, 'monotonic', lambda: clock[0])
    service = object.__new__(NodeService)

    def sis_stop(timeout):
        assert calls == ['closed']
        assert service.onair._running is False
        calls.append(timeout)
        clock[0] += timeout
        return False

    service.sis = SimpleNamespace(_cancel=lambda: calls.append('closed'), stop=sis_stop)
    service.onair = SimpleNamespace(_running=True, stop=lambda timeout: calls.append(timeout) or False)
    assert service.stop(.25) is False
    assert calls == ['closed', .25, 0.0]


def test_cancel_during_write_shutdown(monkeypatch):
    pin = Pin()
    client, peer = socket.socketpair()
    conn = hostio.Connection(peer, nodelay=False)
    monkeypatch.setattr(conn, '_drain_output', pin)

    def shutdown():
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            assert conn.closed

    thread = threading.Thread(target=shutdown)
    thread.start()
    try:
        assert pin.entered.wait(2)
        conn.cancel()
        assert conn.closed
        assert conn.join(2) is True
        assert peer.fileno() == -1
    finally:
        pin.release.set()
        thread.join(2)
        conn.cancel()
        client.close()
        assert conn.join(2)
    assert not thread.is_alive()


@pytest.mark.parametrize('kind', ['bateleur', 'ale', 'selcall', 'sis'])
def test_host_stop_never_drains(monkeypatch, kind):
    listeners = socketpair_network(monkeypatch)
    server = _server(kind)
    server.start_background()
    registered = threading.Event()
    original = hostio.Connection._drain_output
    calls = []

    def drain(conn):
        if threading.current_thread() is threading.main_thread():
            calls.append(conn)
        original(conn)

    monkeypatch.setattr(hostio.Connection, '_drain_output', drain)
    if kind == 'sis':
        attach = server.node.attach
        monkeypatch.setattr(server.node, 'attach', lambda send: registered.set() or attach(send))
    else:
        monkeypatch.setattr(server, 'command', lambda line: registered.set() or 'OK')
    client = listeners[0].connect()
    try:
        if kind != 'sis':
            client.sendall(b'NEXT\r')
        assert registered.wait(2)
        assert server.stop(2)
        assert calls == []
    finally:
        client.close()
        server.stop(2)


def test_bateleur_refuses_accept_returning_during_stop():
    server = _server('bateleur')
    pin = Pin()
    client, peer = socket.socketpair()

    def accept():
        pin()
        return peer, ('127.0.0.1', 1)

    listener = SimpleNamespace(accept=accept, close=lambda: None)
    server._srv = [listener]
    server._running = True
    taken = []
    thread = threading.Thread(target=lambda: taken.extend(server._take(listener)))
    server._threads.append(thread)
    thread.start()
    try:
        assert pin.entered.wait(2)
        bounded_false(server.stop)
        pin.release.set()
        assert server.stop(2)
        assert taken == []
        assert server._connections == []
        assert peer.fileno() == -1
    finally:
        pin.release.set()
        client.close()
        server.stop(2)


def test_cleanup_posted_after_final_clean_prevents_success(monkeypatch):
    worker = make_worker('worker')
    final = Pin()
    original = worker._clean

    def clean():
        original()
        if not worker._running:
            final()

    monkeypatch.setattr(worker, '_clean', clean)
    worker.start()
    results = []
    stopper = threading.Thread(target=lambda: results.append(worker.stop(2)))
    stopper.start()
    try:
        assert final.entered.wait(2)
        cleaned = threading.Event()
        worker.post_cleanup(cleaned.set)
        final.release.set()
        stopper.join(2)
        assert results == [False] or (results == [True] and cleaned.is_set())
        assert worker.stop(2)
        assert cleaned.is_set()
    finally:
        final.release.set()
        stopper.join(2)
        worker.stop(2)


@pytest.mark.parametrize('kind,link_type', [('bateleur', BateleurLink), ('ale', AleLink),
                                           ('selcall', SelcallLink)])
@pytest.mark.parametrize('overrun', [False, True])
def test_lane_stop_waits_for_keyup(monkeypatch, kind, link_type, overrun):
    from hfmodem.station.arbiter import TxArbiter
    from hfmodem.tests.core.test_arbiter import Card, LineRig, req

    card = Card()
    card.wait_gate = threading.Event()
    rig = LineRig(card)
    rig.max_key_s = 2.2
    rig.ptt = SimpleNamespace(settle_s=.03)
    arbiter = TxArbiter(card, rig)
    lane = object.__new__(link_type)
    lane.modem = make_worker(kind)
    lane.modem._audio_limit = lane.modem.card_rate // 50
    lane.station = SimpleNamespace(rig=None, arbiter=arbiter, cfg=SimpleNamespace(rig=rig))
    joining = threading.Event()
    results = []
    stop = lane.modem.stop
    limit = rig.max_key_s + rig.ptt.settle_s + .02

    def join_worker(timeout):
        assert lane.waveform.name in arbiter.disabled
        assert rig.keyed
        assert timeout == pytest.approx(limit)
        joining.set()
        return stop(timeout)

    monkeypatch.setattr(lane.modem, 'stop', join_worker)

    def stop_lane():
        try:
            results.append(lane.stop())
        except Exception as exc:
            results.append(exc)

    stopper = threading.Thread(target=stop_lane)
    lane.modem.start()
    lane.modem.post(lambda: arbiter.submit(req(protocol=lane.waveform.name)))
    try:
        assert card.wait_entered.wait(2)
        started = time.monotonic()
        assert rig.keyed
        stopper.start()
        assert joining.wait(2)
        if not overrun:
            card.wait_gate.set()
        stopper.join(limit + 2)
        elapsed = time.monotonic() - started
        assert not stopper.is_alive()
        if overrun:
            assert elapsed >= limit
            assert len(results) == 1
            assert isinstance(results[0], LeaseTimeout)
            assert str(results[0]) == lane.waveform.name
            assert rig.keyed
            assert lane.modem._thread.is_alive()
        else:
            assert elapsed < rig.max_key_s
            assert results == [None]
            assert not rig.keyed
            assert card.finished == 1
            assert not lane.modem._thread.is_alive()
        assert not card.cancelled
    finally:
        card.wait_gate.set()
        if stopper.ident is not None:
            stopper.join(2)
        assert stop(2)
        arbiter.close()


@pytest.mark.parametrize('deferred', [False, True])
def test_close_waits_for_output_before_cancelling(monkeypatch, deferred):
    client, peer = socket.socketpair()
    conn = hostio.Connection(peer, nodelay=False)
    draining = threading.Event()
    original = conn._drain_output
    pin = Pin()

    def drain():
        draining.set()
        original()

    monkeypatch.setattr(conn, '_drain_output', drain)
    # Hold the writer at the actual send, so closing must wait for it.
    class Socket(_Socket):
        def send(self, data):
            pin()
            return self.sock.send(data)

    conn.sock = Socket(peer)
    (conn.send_later if deferred else conn.sendall)(b'reply\r\n')
    closer = threading.Thread(target=conn.close)
    try:
        assert pin.entered.wait(2)
        closer.start()
        assert draining.wait(2)
        assert not conn.closed
        pin.release.set()
        client.settimeout(2)
        assert client.recv(7) == b'reply\r\n'
        closer.join(2)
        assert conn.closed
    finally:
        pin.release.set()
        conn.cancel()
        closer.join(2)
        client.close()
        assert conn.join(2)


@pytest.mark.parametrize('kind', ['bateleur', 'ale', 'selcall', 'sis'])
def test_registration_prunes_only_ended_entries(monkeypatch, kind):
    listeners = socketpair_network(monkeypatch)
    server = _server(kind)
    pin = Pin()
    alive = threading.Thread(target=pin)
    ended = threading.Thread(target=lambda: None)
    alive.start()
    ended.start()
    ended.join(2)
    dead_conn = SimpleNamespace(join=lambda timeout: True)
    live_conn = SimpleNamespace(join=lambda timeout: False, cancel=lambda: None)
    server._connections.extend([dead_conn, live_conn])
    retained = server._threads if hasattr(server, '_threads') else server._workers
    if isinstance(retained, list):
        retained.extend([alive, ended])
    else:
        retained.update([alive, ended])
    registered = threading.Event()
    if kind == 'sis':
        attach = server.node.attach
        monkeypatch.setattr(server.node, 'attach', lambda send: registered.set() or attach(send))
    else:
        monkeypatch.setattr(server, 'command', lambda line: registered.set() or 'OK')
    server.start_background()
    client = listeners[0].connect()
    try:
        if kind != 'sis':
            client.sendall(b'NEXT\r')
        assert registered.wait(2)
        assert dead_conn not in server._connections
        assert live_conn in server._connections
        retained = server._threads if hasattr(server, '_threads') else server._workers
        assert alive in retained
        assert ended not in retained
    finally:
        pin.release.set()
        with server._lock:
            server._connections.remove(live_conn)
        client.close()
        assert server.stop(2)


def test_registration_keeps_threads_not_yet_started(monkeypatch):
    listeners = socketpair_network(monkeypatch)
    server = _server('bateleur')
    registered = threading.Event()
    init = hostio.Connection.__init__

    def registering(conn, *args, **kwargs):
        init(conn, *args, **kwargs)
        registered.set()

    monkeypatch.setattr(hostio.Connection, '__init__', registering)
    own = (server._accept_cmd, server._accept_data, server._autosender)
    clients = []
    started = []
    start = threading.Thread.start

    def gated(thread):
        if thread._target in own:
            if thread._target == server._accept_data:
                clients.append(listeners[0].connect())
                assert registered.wait(2)
            started.append(thread)
        start(thread)

    monkeypatch.setattr(threading.Thread, 'start', gated)
    server.start_background()
    try:
        assert len(started) == 3
        assert all(t in server._threads for t in started)
    finally:
        for client in clients:
            client.close()
        assert server.stop(2)
