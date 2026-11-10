# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A backend's fast RF callbacks cannot overtake command admission replies."""
import threading

import pytest

from .test_host_callback_order import endpoint as endpoint


@pytest.mark.parametrize('threaded', [False, True])
def test_admission_reply_precedes_resulting_events_without_blocking_the_backend(endpoint, threaded):
    session, modem, lines = endpoint
    trace = []
    session._send_cmd = lambda line: trace.append(('command', line))
    session._send_data = lambda data: trace.append(('data', data))
    returned = threading.Event()
    def callback():
        session.modem_connected('W9SSJ', 'K7ABC', '500')
        session.modem_data_received(b'first\x00\xff')
        session.modem_buffer(0)
        returned.set()
    def connect(*args):
        if threaded:
            worker = threading.Thread(target=callback)
            worker.start()
            worker.join(1)
            assert not worker.is_alive(), 'RF callback blocked on its host command'
        else:
            callback()
        assert returned.is_set()
    modem.connect = connect
    session._dispatch('CONNECT W9SSJ K7ABC')
    assert trace == [('command', 'OK'), ('command', 'CONNECTED W9SSJ K7ABC 500'),
                     ('data', b'first\x00\xff'), ('command', 'BUFFER 0')]


def test_callbacks_arriving_while_events_drain_preserve_their_order(endpoint):
    session, modem, lines = endpoint
    trace = []
    def command(line):
        trace.append(line)
        if line.startswith('CONNECTED '):
            session.modem_buffer(7)
    session._send_cmd = command
    def connect(*args):
        session.modem_connected('W9SSJ', 'K7ABC', '500')
        session.modem_buffer(3)
    modem.connect = connect
    session._dispatch('CONNECT W9SSJ K7ABC')
    assert trace == ['OK', 'CONNECTED W9SSJ K7ABC 500', 'BUFFER 3', 'BUFFER 7']


@pytest.mark.parametrize('overflow', ['bytes', 'events'])
def test_command_callback_overflow_retires_both_sockets_without_blocking_backend(endpoint, monkeypatch, overflow):
    from hfmodem.kestrel.host import server
    session, modem, lines = endpoint
    monkeypatch.setattr(server, 'MAX_OUTPUT', 16)
    monkeypatch.setattr(server, 'MAX_PENDING', 4)
    def connect(*args):
        if overflow == 'bytes':
            session.modem_data_received(b'x' * 17)
        else:
            for index in range(5):
                session.modem_ptt(index % 2 == 0)
    modem.connect = connect
    session._dispatch('CONNECT W9SSJ K7ABC')
    assert session._closed
    assert session._cmd.closed and session._data.closed
    assert not session._events and session._event_bytes == 0


def test_filling_the_byte_bound_exactly_preserves_binary_output(endpoint, monkeypatch):
    from hfmodem.kestrel.host import server
    session, modem, lines = endpoint
    monkeypatch.setattr(server, 'MAX_OUTPUT', 16)
    sent = []
    session._send_data = sent.append
    original = bytes(range(16))
    modem.connect = lambda *args: session.modem_data_received(original)
    session._dispatch('CONNECT W9SSJ K7ABC')
    assert lines == ['LINK REGISTERED', 'OK'] and sent == [original]
    assert not session._closed


def test_continuous_callbacks_during_drain_have_a_finite_retirement_boundary(endpoint, monkeypatch):
    from hfmodem.kestrel.host import server
    session, modem, lines = endpoint
    monkeypatch.setattr(server, 'MAX_PENDING', 4)
    def command(line):
        lines.append(line)
        if line == 'PTT ON':
            session.modem_ptt(True)
    session._send_cmd = command
    modem.connect = lambda *args: session.modem_ptt(True)
    session._dispatch('CONNECT W9SSJ K7ABC')
    assert lines == ['LINK REGISTERED', 'OK'] + ['PTT ON'] * 4
    assert session._closed and not session._events


def test_fast_connection_reply_and_notification_reach_the_real_command_socket():
    import socket
    from hfmodem.kestrel.host.server import HostSession
    from .test_host_callback_order import Immediate
    command, peer = socket.socketpair()
    data, data_peer = socket.socketpair()
    modem = Immediate()
    session = HostSession(command, data, modem)
    peer.settimeout(1)
    try:
        modem.start(session)
        session._dispatch('CONNECT W9SSJ K7ABC')
        received = bytearray()
        while received.count(b'\r') < 3:
            received.extend(peer.recv(4096))
        assert received == b'LINK REGISTERED\rOK\rCONNECTED W9SSJ K7ABC 500\r'
    finally:
        session.close()
        session.ensure_retired()
        peer.close()
        data_peer.close()
