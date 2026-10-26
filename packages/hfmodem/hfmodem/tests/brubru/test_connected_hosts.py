# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import os
import select
import socket
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from hfmodem.ax25.connected import Connection, State
from hfmodem.ax25.frame import Frame, address, parse
from hfmodem.brubru.host.agwpe import AgwHost, Decoder, Packet
from hfmodem.brubru.host.connected_server import AgwServer, TrackerServer
from hfmodem.brubru.host.runtime import PacketRuntime
from hfmodem.station.arbiter import RequestState


def peer_frame(control, payload=b''):
    return Frame((address('W9SSJ', flag=control & 1 == 0),
                  address('N0CALL', last=True, flag=control & 1 != 0)),
                 control, 0xF0 if control & 1 == 0 else None, payload).encode()


def wait(predicate):
    end = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < end, 'condition timed out'
        time.sleep(.005)


class Link:
    def __init__(self):
        self.host = self.packet_engine = None
        self.txdelay, self.level, self.drive_scale = 100, 0, 1.
        self.access = SimpleNamespace(persistence=32, slot_ms=10, _lock=threading.Lock())
        self._condition = threading.Condition()
        self.requests = []
    def enqueue(self, frame):
        request = SimpleNamespace(frame=frame, completion=None, state=RequestState())
        self.requests.append(request)
        return request
    def drop_pending(self):
        for request in self.requests:
            if request.completion is None:
                request.state.cancel()
    def complete(self, index=-1, *, okay=True):
        self.requests[index].completion = ('sent', '') if okay else ('refused', 'no transmitter')


def test_tracker_reopen_applies_its_reported_engine_and_rf_defaults():
    link = Link()
    runtime = PacketRuntime(link, mycall='W9SSJ')
    runtime.start()
    try:
        old = runtime.attach('tracker')
        assert not runtime.engine.listen and old.host.values['Y'] == 0
        runtime.feed(old, b'\x1bO7\r\x1b%XR3000\r\x1bT10\r\x1bP64\r')
        assert runtime.engine.window == 7 and link.drive_scale == 1
        runtime.detach(old)
        new = runtime.attach('tracker')
        assert runtime.engine.window == new.host.values['O'] == 2
        assert link.drive_scale == new.host.values['%XR']/3000
        assert link.txdelay == new.host.values['T']*10 == 250
        assert link.access.persistence == new.host.values['P'] == 32
        assert link.access.slot_ms == new.host.values['W'] == 10
        assert not runtime.engine.listen
    finally:
        runtime.stop()


def exchange(sock, packet):
    sock.sendall(packet.encode())
    decoder = Decoder()
    while True:
        data = sock.recv(4096)
        assert data
        frames = decoder.feed(data)
        if frames:
            assert len(frames) == 1
            return frames[0]


def test_agw_header_is_36_bytes_with_little_endian_length_and_zero_reserves():
    frame = Packet(0, 'D', 0xF0, 'W9SSJ', 'N0CALL', bytes(range(256)))
    raw = frame.encode()
    assert len(raw) == 36 + 256
    assert raw[:8] == b'\0\0\0\0D\0\xf0\0'
    assert raw[8:18] == b'W9SSJ\0\0\0\0\0'
    assert raw[28:36] == b'\0\x01\0\0\0\0\0\0'
    decoder = Decoder()
    assert [f for byte in raw for f in decoder.feed(bytes([byte]))] == [frame]
    assert Decoder().feed(raw + raw) == [frame, frame]


def test_agw_oversize_length_rejected_before_body_allocation():
    raw = bytearray(Packet(0, 'D').encode())
    raw[28:32] = struct.pack('<I', 65537)
    with pytest.raises(ValueError):
        Decoder().feed(raw)


def test_agw_registration_capabilities_and_no_false_1200_baud_advertisement():
    output = []
    engine = Connection('NOCALL', lambda _: True)
    host = AgwHost(engine, lambda data: output.extend(Decoder().feed(data)))
    host.feed(Packet(0, 'g').encode())
    assert len(output[-1].data) == 12 and output[-1].data[6] == 2
    assert output[-1].data[0] == 255
    host.feed(Packet(0, 'G').encode())
    assert b'R300' in output[-1].data
    host.feed(Packet(0, 'X', source='W9SSJ').encode())
    assert output[-1].data == b'\x01' and engine.listen
    host.feed(Packet(0, 'X', source='N0CALL').encode())
    assert output[-1].data == b'\x00' and engine.mycall == 'W9SSJ'
    host.feed(Packet(0, 'x', source='W9SSJ').encode())
    assert not engine.listen and not host.registered


def test_agw_incoming_and_outgoing_connect_text_is_distinct():
    output = []
    engine = Connection('W9SSJ', lambda _: True)
    host = AgwHost(engine, lambda data: output.extend(Decoder().feed(data)))
    host.handle(Packet(0, 'X', source='W9SSJ'))
    engine.remote = 'N0CALL'
    engine._state(State.CONNECTED)
    assert output[-1].data == b'*** CONNECTED To N0CALL\r'
    assert output[-1].source == 'N0CALL' and output[-1].destination == 'W9SSJ'
    engine.abort()
    host.handle(Packet(0, 'C', source='W9SSJ', destination='N0CALL'))
    engine.on_frame(peer_frame(0x73))
    assert output[-1].data == b'*** CONNECTED With N0CALL\r'


def test_agw_block_limit_preserves_every_original_byte():
    output = []
    engine = Connection('W9SSJ', lambda _: True)
    host = AgwHost(engine, lambda data: output.extend(Decoder().feed(data)))
    engine.remote = 'N0CALL'
    host._receive(bytes(range(256)) * 3)
    assert all(len(p.data) == 128 and p.kind == 'D' and p.pid == 0xF0 for p in output)
    assert b''.join(p.data for p in output) == bytes(range(256)) * 3


def test_agw_output_failure_retires_originals_and_blocks_late_commands():
    engine = Connection('W9SSJ', lambda _: True)
    host = AgwHost(engine, lambda _: False)
    engine.remote = 'N0CALL'
    engine.state = State.CONNECTED
    engine.write(b'unconfirmed')
    with pytest.raises(ConnectionAbortedError):
        host._receive(b'received')
    assert engine.state is State.DISCONNECTED
    assert engine.last_unconfirmed == b'unconfirmed' and not engine.delivered
    with pytest.raises(ConnectionAbortedError):
        host.feed(Packet(0, 'C', source='W9SSJ', destination='N0CALL').encode())


def test_packet_owner_waits_for_real_transmit_completion_before_t1():
    link, now = Link(), [0.]
    runtime = PacketRuntime(link, mycall='W9SSJ', clock=lambda: now[0])
    runtime.start()
    try:
        attachment = runtime.attach('tracker')
        runtime.feed(attachment, b'\x1bS1\r\x1bC N0CALL\r')
        wait(lambda: bool(link.requests))
        now[0] = 10000
        time.sleep(.05)
        assert runtime.engine.retries == 0 and len(link.requests) == 1
        link.complete()
        wait(lambda: runtime._transmission is None)
        assert runtime.engine._deadline > now[0]
        runtime.detach(attachment)
        assert not runtime.engine.active
        assert link.packet_engine is runtime
    finally:
        runtime.stop()
    assert link.packet_engine is link.host is None


def test_detach_cancels_pending_physical_request_and_rejects_old_token():
    link = Link()
    runtime = PacketRuntime(link, mycall='W9SSJ')
    runtime.start()
    try:
        old = runtime.attach('tracker')
        runtime.feed(old, b'\x1bS1\r\x1bC N0CALL\r')
        wait(lambda: bool(link.requests))
        runtime.detach(old)
        assert link.requests[0].state.value == 'cancelled'
        with pytest.raises(ValueError):
            runtime.attach('tracker')
        link.complete()
        wait(lambda: runtime._transmission is None)
        new = runtime.attach('tracker')
        with pytest.raises(ConnectionAbortedError):
            runtime.feed(old, b'\x1bC N0CALL\r')
        assert new.alive and not runtime.engine.active
    finally:
        runtime.stop()


def test_agw_real_tcp_enqueue_ack_and_clean_disconnect():
    link = Link()
    runtime = PacketRuntime(link, mycall='W9SSJ')
    server = AgwServer(runtime, port=0)
    server.start()
    try:
        with socket.create_connection(('127.0.0.1', server.port), timeout=2) as client:
            assert exchange(client, Packet(0, 'R')).data == struct.pack('<HHHH', 0, 0, 1, 0)
            assert exchange(client, Packet(0, 'X', source='W9SSJ')).data == b'\x01'
            client.sendall(Packet(0, 'C', source='W9SSJ', destination='N0CALL').encode())
            wait(lambda: bool(link.requests))
            assert parse(link.requests[-1].frame).control == 0x3F
            link.complete()
            runtime.send_frame(peer_frame(0x73))
            reader = Decoder()
            event, = reader.feed(client.recv(4096))
            assert event.kind == 'C' and event.data.startswith(b'*** CONNECTED With ')
            client.sendall(Packet(0, 'D', 0xF0, 'W9SSJ', 'N0CALL', b'final').encode())
            pending = exchange(client, Packet(0, 'Y', source='W9SSJ', destination='N0CALL'))
            assert int.from_bytes(pending.data, 'little') == 1
            wait(lambda: len(link.requests) >= 2)
            link.complete()
            runtime.send_frame(peer_frame(0x31))  # cumulative N(R)=1, F=1
            wait(lambda: runtime.engine.delivered == 5)
            assert int.from_bytes(exchange(client, Packet(0, 'Y', source='W9SSJ', destination='N0CALL')).data, 'little') == 0
            client.sendall(Packet(0, 'd', source='W9SSJ', destination='N0CALL').encode())
            wait(lambda: parse(link.requests[-1].frame).control == 0x53)
            link.complete()
            runtime.send_frame(peer_frame(0x73))
            event, = reader.feed(client.recv(4096))
            assert event.kind == 'd'
    finally:
        server.stop()


def test_agw_rf_failure_closes_both_owner_and_endpoint_without_another_read():
    link = Link()
    runtime = PacketRuntime(link, mycall='W9SSJ')
    server = AgwServer(runtime, port=0)
    server.start()
    try:
        with socket.create_connection(('127.0.0.1', server.port), timeout=2) as client:
            exchange(client, Packet(0, 'X', source='W9SSJ'))
            client.sendall(Packet(0, 'C', source='W9SSJ', destination='N0CALL').encode())
            wait(lambda: bool(link.requests))
            link.complete(okay=False)
            data = client.recv(4096)
            if data:
                assert Decoder().feed(data)[0].kind == 'd'
                assert client.recv(1) == b''
            wait(lambda: server._client is None)
            assert not runtime.engine.active
    finally:
        server.stop()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX PTY endpoint')
def test_tracker_real_pty_reopen_and_owned_alias_cleanup(tmp_path):
    link = Link()
    runtime = PacketRuntime(link, mycall='W9SSJ')
    alias = tmp_path / 'tracker'
    server = TrackerServer(runtime, endpoint=str(alias))
    server.start()
    try:
        for _ in range(2):
            fd = os.open(alias, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            try:
                import tty
                tty.setraw(fd)       # serial clients configure the port on every open
                os.write(fd, b'\x1bS1\r\x1bJHOST1\r')
                response = bytearray()
                deadline = time.monotonic() + 2
                while b'CHANNEL 1' not in response:
                    assert time.monotonic() < deadline
                    assert select.select([fd], [], [], 2)[0]
                    response.extend(os.read(fd, 4096))
                assert b'CHANNEL 1' in response
                # SCS 1.7 boots with terminal echo enabled; ESC/CR echo '.'.
                assert response.startswith(b'.S1.\rCHANNEL')
            finally:
                os.close(fd)
            wait(lambda: runtime._attachment is None)
        assert alias.is_symlink()
    finally:
        server.stop()
    assert not alias.exists()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX PTY endpoint')
def test_tracker_never_replaces_existing_endpoint(tmp_path):
    alias = tmp_path / 'operator'
    alias.write_text('keep')
    link = Link()
    server = TrackerServer(PacketRuntime(link), endpoint=str(alias))
    with pytest.raises(FileExistsError):
        server.start()
    assert alias.read_text() == 'keep'
    assert link.host is None
