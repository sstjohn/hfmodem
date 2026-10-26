# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Backend callbacks may precede return from their initiating host command."""
import socket

import pytest

from hfmodem.kestrel.host.modem_core import LoopbackModem
from hfmodem.kestrel.host.server import HostSession, SessionState


class Immediate(LoopbackModem):
    def connect(self, src, dst, vias=None):
        with self._lock:
            self._connected = True
        self._observer.modem_connected(src, dst, self._bandwidth)

    def disconnect(self):
        self._finish_disconnect()


@pytest.fixture
def endpoint():
    command, command_peer = socket.socketpair()
    data, data_peer = socket.socketpair()
    modem = Immediate()
    session = HostSession(command, data, modem)
    lines = []
    session._send_cmd = lines.append
    modem.start(session)
    yield session, modem, lines
    session.close()
    session.ensure_retired()
    command_peer.close()
    data_peer.close()


def test_fast_connect_and_disconnect_callbacks_survive_command_return_and_repeat(endpoint):
    session, _, lines = endpoint
    for _ in range(2):
        session._dispatch('CONNECT W9SSJ K7ABC')
        assert session.state == SessionState.CONNECTED
        assert lines[-2:] == ['OK', 'CONNECTED W9SSJ K7ABC 500']
        session._dispatch('DISCONNECT')
        assert session.state == SessionState.DISCONNECTED
        assert lines[-3:] == ['OK', 'DISCONNECTED', 'BUSY OFF']
    assert 'WRONG' not in lines


def test_connect_refusal_rolls_back_without_storing_an_initiated_state(endpoint):
    session, modem, lines = endpoint
    session.state = SessionState.LISTENING
    def refuse(*args):
        raise ValueError('no admission')
    modem.connect = refuse
    session._dispatch('CONNECT W9SSJ K7ABC')
    assert session.state == SessionState.LISTENING and lines[-1] == 'WRONG'


def test_disconnect_refusal_rolls_back_and_callback_refusal_is_not_overwritten(endpoint):
    session, modem, lines = endpoint
    session._dispatch('CONNECT W9SSJ K7ABC')
    def refuse():
        raise ValueError('not admitted')
    modem.disconnect = refuse
    session._dispatch('DISCONNECT')
    assert session.state == SessionState.CONNECTED and lines[-1] == 'WRONG'
    modem.disconnect = lambda: session.modem_disconnect_refused('DISCONNECT REFUSED')
    session._dispatch('DISCONNECT')
    assert session.state == SessionState.CONNECTED
    assert lines[-2:] == ['OK', 'DISCONNECT REFUSED']


@pytest.mark.parametrize('on', [False, True])
def test_listen_close_callback_sees_selected_next_idle_state(endpoint, on):
    session, modem, lines = endpoint
    session._dispatch('CONNECT W9SSJ K7ABC')
    session.listening = not on
    modem.set_listen = lambda selected: modem._finish_disconnect()
    session._dispatch('LISTEN ON' if on else 'LISTEN OFF')
    assert session.listening is on
    assert session.state == (SessionState.LISTENING if on else SessionState.DISCONNECTED)
    assert lines[-3:] == ['OK', 'DISCONNECTED', 'BUSY OFF']


@pytest.mark.parametrize('on', [False, True])
def test_refused_listen_preserves_frontend_setting_and_contact(endpoint, on):
    session, modem, lines = endpoint
    session._dispatch('CONNECT W9SSJ K7ABC')
    session.listening = not on
    def refuse(selected):
        raise ValueError('owner queue full')
    modem.set_listen = refuse
    session._dispatch('LISTEN ON' if on else 'LISTEN OFF')
    assert session.listening is not on
    assert session.state == SessionState.CONNECTED
    assert lines[-1] == 'WRONG'


@pytest.mark.parametrize('connected', [False, True])
@pytest.mark.parametrize('command', ['PUBLIC ON', 'PUBLIC OFF', 'CWID ON', 'CWID OFF'])
def test_stock_compatibility_controls_ack_without_mutating_modem_state(endpoint, connected, command):
    session, modem, lines = endpoint
    if connected:
        session._dispatch('CONNECT W9SSJ K7ABC')
    modem._compression, modem._bandwidth, modem._listen = 'TEXT', '2300', True
    before = (session.state, session.compression, session.bandwidth, session.listening,
              modem.connected, modem._compression, modem._bandwidth, modem._listen)
    lines.clear()
    session._dispatch(command)
    assert lines == ['OK']
    assert before == (session.state, session.compression, session.bandwidth, session.listening,
                      modem.connected, modem._compression, modem._bandwidth, modem._listen)


@pytest.mark.parametrize('command', [
    'PUBLIC', 'PUBLIC YES', 'PUBLIC ON EXTRA', 'CWID', 'CWID YES', 'CWID OFF EXTRA',
])
def test_compatibility_controls_do_not_accept_unknown_or_extra_arguments(endpoint, command):
    session, _, lines = endpoint
    assert lines == ['LINK REGISTERED']
    lines.clear()
    session._dispatch(command)
    assert lines == ['WRONG']
