# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KISS fan-out with bounded output and independent client progress."""
import socket
import threading
import time

import pytest

from hfmodem.ax25 import kiss
from hfmodem.kestrel.host.kiss import KissPort, OUTPUT_FRAMES, _Client


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(.005)
    assert predicate()


def receive(client, count):
    parser, frames = kiss.Parser(), []
    while len(frames) < count:
        data = client.recv(8192)
        assert data
        frames.extend(parser.feed(data))
    return frames


@pytest.fixture
def fanout():
    now, logs, submitted = [0.0], [], []
    port = KissPort(lambda text: None, port=0, clock=lambda: now[0],
                    on_frame=lambda *args: submitted.append(args),
                    log=lambda *args: logs.append(args))
    slow, peer = socket.socketpair()
    slow.setblocking(False)
    slow.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    while True:
        try:
            slow.send(b'x' * 4096)
        except BlockingIOError:
            break
    state = _Client()
    port._clients[slow] = state
    port.start()
    healthy = socket.create_connection((port.host, port.port), timeout=2)
    wait_for(lambda: len(port._clients) == 2)
    try:
        yield port, slow, peer, state, healthy, now, logs, submitted
    finally:
        healthy.close()
        peer.close()
        assert port.stop()


def test_nonreading_client_drops_oldest_without_affecting_healthy_client(fanout):
    port, _, _, state, healthy, _, logs, _ = fanout
    for i in range(OUTPUT_FRAMES + 12):
        payload = i.to_bytes(2, 'big')
        port.deliver_frame(0, payload)
        healthy.sendall(kiss.encode(b'', 15))
        assert receive(healthy, 1) == [(0, 0, payload)]
    with port._lock:
        assert state.dropped == 12
        assert list(state.output) == [
            kiss.encode(i.to_bytes(2, 'big')) for i in range(12, OUTPUT_FRAMES + 12)]
    wait_for(lambda: any('dropped 12 oldest frames' in msg for _, msg in logs))


def test_full_queue_disconnects_after_thirty_seconds(fanout):
    port, slow, _, state, healthy, now, logs, submitted = fanout
    for _ in range(OUTPUT_FRAMES):
        port.deliver_frame(0, b'full')
        healthy.sendall(kiss.encode(b'', 15))
        assert receive(healthy, 1) == [(0, 0, b'full')]
    assert state.full_since == 0
    now[0] = 29.999
    healthy.sendall(kiss.encode(b'wake'))
    wait_for(lambda: submitted == [(0, b'wake')])
    with port._lock:
        assert slow in port._clients
    now[0] = 30
    wait_for(lambda: slow not in port._clients)
    assert any('queue full for 30 s' in msg for _, msg in logs)
    port.deliver_frame(1, b'still here')
    assert receive(healthy, 1) == [(0, 1, b'still here')]


def test_delivery_returns_with_a_full_socket_buffer(fanout):
    port, _, _, state, _, _, _, _ = fanout
    done = threading.Event()

    def deliver():
        for _ in range(OUTPUT_FRAMES + 10):
            port.deliver_frame(0, b'queued')
        done.set()

    worker = threading.Thread(target=deliver, daemon=True)
    worker.start()
    try:
        assert done.wait(.5)
        assert len(state.output) == OUTPUT_FRAMES
        assert state.dropped == 10
    finally:
        worker.join(3)


def test_disconnect_concurrent_with_delivery_keeps_reader_alive(fanout):
    port, slow, peer, _, healthy, _, _, submitted = fanout
    entered, resume, done = threading.Event(), threading.Event(), threading.Event()

    def deliver():
        port.deliver_frame(0, b'before')
        entered.set()
        resume.wait(2)
        port.deliver_frame(1, b'after')
        done.set()

    worker = threading.Thread(target=deliver, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        peer.close()
        resume.set()
        assert done.wait(2)
        assert receive(healthy, 2) == [(0, 0, b'before'), (0, 1, b'after')]
        wait_for(lambda: slow not in port._clients)
        healthy.sendall(kiss.encode(b'submitted', 1))
        wait_for(lambda: submitted == [(1, b'submitted')])
        assert port._thread.is_alive()
    finally:
        resume.set()
        worker.join(3)


class ShortSocket:
    def __init__(self, sock):
        self.sock = sock
        self.calls = 0
        self.on_recv = None

    def fileno(self):
        return self.sock.fileno()

    def send(self, wire):
        self.calls += 1
        if self.calls % 2 == 0:
            raise BlockingIOError()
        return self.sock.send(wire[:3])

    def recv(self, size):
        data = self.sock.recv(size)
        if self.on_recv:
            self.on_recv()
        return data

    def close(self):
        self.sock.close()


def test_partial_writes_continue_without_corrupting_frames():
    port = KissPort(lambda text: None, port=0)
    sock, peer = socket.socketpair()
    sock.setblocking(False)
    peer.settimeout(2)
    short = ShortSocket(sock)
    port._clients[short] = _Client()
    payloads = [b'one\xc0\xdb', b'two' * 20, b'three']
    for payload in payloads:
        port.deliver_frame(1, payload)
    port.start()
    try:
        assert receive(peer, 3) == [(0, 1, payload) for payload in payloads]
        assert short.calls > 3
    finally:
        peer.close()
        assert port.stop()


def test_full_timer_resets_on_progress_and_overflow_preserves_partial_frame():
    now = [0.0]
    port = KissPort(lambda text: None, port=0, clock=lambda: now[0])
    sock, peer = socket.socketpair()
    sock.setblocking(False)
    short, state = ShortSocket(sock), _Client()
    port._clients[short] = state
    try:
        for _ in range(OUTPUT_FRAMES):
            port.deliver_frame(1, b'first frame')
        assert state.full_since == 0
        port._write(short)
        assert state.full_since is None
        partial = state.pending
        assert partial == kiss.encode(b'first frame', 1)[3:]
        now[0] = 29
        port.deliver_frame(1, b'refills queue')
        port.deliver_frame(1, b'overflows queue')
        assert state.full_since == 29 and state.dropped == 1
        port._write(short)
        assert state.full_since == 29 and state.pending == partial
    finally:
        peer.close()
        sock.close()
        port._listener.close()


def test_client_removed_during_recv_is_skipped():
    submitted = []
    port = KissPort(lambda text: None, port=0,
                    on_frame=lambda *args: submitted.append(args))
    sock, peer = socket.socketpair()
    sock.setblocking(False)
    short = ShortSocket(sock)
    short.on_recv = lambda: port._drop(short)
    port._clients[short] = _Client()
    port.start()
    healthy = socket.create_connection((port.host, port.port), timeout=2)
    try:
        peer.sendall(kiss.encode(b'removed'))
        wait_for(lambda: short not in port._clients)
        healthy.sendall(kiss.encode(b'healthy'))
        wait_for(lambda: submitted == [(0, b'healthy')])
        port.deliver_frame(0, b'received')
        assert receive(healthy, 1) == [(0, 0, b'received')]
        assert port._thread.is_alive()
    finally:
        healthy.close()
        peer.close()
        assert port.stop()


def test_submitted_frames_are_not_echoed_to_any_local_client(fanout):
    port, _, _, _, healthy, _, _, submitted = fanout
    other = socket.create_connection((port.host, port.port), timeout=2)
    try:
        wait_for(lambda: len(port._clients) == 3)
        healthy.sendall(kiss.encode(b'outbound', 1))
        wait_for(lambda: submitted == [(1, b'outbound')])
        for client in (healthy, other):
            client.settimeout(.1)
            with pytest.raises(socket.timeout):
                client.recv(8192)
    finally:
        other.close()
