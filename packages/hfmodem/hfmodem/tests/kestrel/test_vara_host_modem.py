# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Host lifecycle and the live launcher's factory, with no audio devices."""
import importlib.util
from pathlib import Path

import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, compression as C
from .test_compression import IO


class Observer:
    def __init__(self):
        self.closes = 0
        self.warnings = []
        self.buffers = []
        self.data = []
    def modem_disconnected(self): self.closes += 1
    def modem_warning(self, message): self.warnings.append(message)
    def modem_disconnect_refused(self, message): self.modem_warning(message)
    def modem_buffer(self, depth): self.buffers.append(depth)
    def modem_connected(self, *args): pass
    def modem_ptt(self, on): pass
    def modem_data_received(self, data): self.data.append(data)


@pytest.fixture
def modem(monkeypatch):
    monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a, **kw: True)
    m = VaraHostModem(IO())
    m.start(Observer())
    return m


@pytest.mark.parametrize('state', [A.VaraState.DISCONNECTED, A.VaraState.LISTENING,
                                   A.VaraState.CONNECTING])
def test_defensive_disconnect_cannot_close_next_connection(modem, state):
    modem.station.state = state
    modem.disconnect()
    modem.poll()
    assert not modem._closing
    assert modem.observer.closes == 1
    modem.connect('W9SSJ', 'WI9YZA')
    modem.poll()
    modem.station._connected(confirm=False)
    modem.poll()
    assert modem.connected
    assert modem.station.state == A.VaraState.CONNECTED
    assert not modem._closing


def test_called_station_disconnect_is_reported_not_left_pending(modem):
    modem.station.role = 'responder'
    modem.station.state = A.VaraState.CONNECTED
    modem._connected = True
    modem.disconnect()
    modem.poll()
    assert not modem._closing
    assert modem.observer.warnings == ['WRONG DISCONNECT: called-station close is not implemented']
    modem.station.role = 'initiator'  # a later role must not inherit a close
    modem.poll()
    assert modem.station.state == A.VaraState.CONNECTED


def test_caller_disconnect_waits_for_queued_data_then_clears(modem):
    hs = modem.station
    hs.role = 'initiator'
    hs.caller, hs.called = 'W9SSJ', 'WI9YZA'
    hs.state = A.VaraState.CONNECTED
    hs._answering = True
    modem._connected = True
    modem.transmit(b'finish this')
    modem.disconnect()
    modem.poll()
    assert modem._closing and hs.pending_bytes == len(b'finish this')
    hs._txq.clear()  # acknowledged by the peer
    modem.poll()
    assert not modem._closing and not modem.connected
    assert modem.observer.closes == 1


def test_listen_preserves_host_writes_until_caller_mode_known(modem, monkeypatch):
    modem.transmit(b'first')
    modem.transmit(b'second')
    modem.set_bandwidth('500')
    modem.set_listen(True)
    modem.poll()
    hs = modem.station
    assert hs.bw == '500'
    assert hs.preconnect_data == (b'first', b'second')
    frame = F.link_setup_frame('W9SSJ', bw='500', high_speed=True, compression='FILES')
    monkeypatch.setattr(A.OF, 'link_setup_read', lambda *a, **kw: frame)
    monkeypatch.setattr(hs, '_tx_connected_ack', lambda: True)
    hs.called = 'WI9YZA'
    assert hs._rx_link_setup([])
    assert hs.preconnect_data == ()
    rx = C.ReceiveCompression('FILES')
    assert rx.feed(b''.join(hs._txq), 0x82, True) == (b'firstsecond', None)
    assert hs.pending_bytes == len(C.encode_files(b'first')) + len(C.encode_files(b'second'))


def test_default_host_announces_files_and_explicit_speed_limit_disables_it(modem):
    modem.set_compression('FILES')
    modem.connect('W9SSJ', 'WI9YZA')
    modem.poll()
    hs = modem.station
    assert hs.high_speed and hs.session_compression == 'FILES'
    assert F.link_setup_frame(hs.caller, high_speed=hs.high_speed,
                              compression=hs.session_compression)[7] == 0x85
    limited = VaraHostModem(IO(), high_speed=False)
    limited.set_compression('FILES')
    limited.connect('W9SSJ', 'WI9YZA')
    limited.poll()
    assert limited.station.session_compression == 'OFF'


def runtime_class():
    path = Path(__file__).resolve().parents[5] / 'tools' / 'kestrel_host.py'
    require_input(path)
    spec = importlib.util.spec_from_file_location('kestrel_audio_host_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AudioHostRuntime


def test_live_factory_routes_host_mode_into_audio_driver(monkeypatch):
    import socket
    from hfmodem.kestrel.host.server import HostSession
    class Audio(IO):
        def next_rx_burst(self, timeout, hs=None):
            self.seen_station = hs
            return None
    audio = Audio()
    runtime = runtime_class()(audio)
    modem = runtime.modem_factory()  # same factory given to the live VaraServer
    cmd, appcmd = socket.socketpair()
    data, appdata = socket.socketpair()
    appcmd.settimeout(1)
    try:
        session = HostSession(cmd, data, modem)
        modem.start(session)
        session._dispatch('COMPRESSION FILES')
        assert appcmd.recv(4096) == b'OK\r'
        monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a, **kw: True)
        session._dispatch('CONNECT W9SSJ WI9YZA')
        assert appcmd.recv(4096) == b'OK\r'
        runtime.step()
        assert audio.seen_station is modem.station
        assert modem.station.session_compression == 'FILES'
        modem.station._connected(confirm=False)
        modem.station._answering = True
        modem.transmit(b'to the audio path')
        runtime.step()
        rx = C.ReceiveCompression('FILES')
        assert rx.feed(b''.join(modem.station._txq), 0x82, True) == (b'to the audio path', None)
    finally:
        for sock in (cmd, appcmd, data, appdata): sock.close()


def test_incoming_connect_timeout_starts_after_listen_not_at_startup():
    class Audio(IO):
        def next_rx_burst(self, timeout, hs=None): return None
    now = [0.0]
    runtime = runtime_class()(Audio(), clock=lambda: now[0])
    modem = runtime.modem_factory()
    modem.start(Observer())
    modem.set_listen(True)
    runtime.step()
    now[0] = 600.0
    modem.station.state = A.VaraState.CONNECTING
    modem.station.role = 'responder'
    hs = modem.station
    runtime.step()
    assert modem.station is hs
    assert modem.observer.closes == 0
    now[0] = 661.0
    runtime.step()
    assert modem.station is not hs
    assert modem.station.state == A.VaraState.LISTENING
    assert modem.observer.closes == 1


def test_refused_close_keeps_host_session_connected(modem):
    import socket
    from hfmodem.kestrel.host.server import HostSession, SessionState
    cmd, appcmd = socket.socketpair()
    data, appdata = socket.socketpair()
    appcmd.settimeout(1)
    try:
        session = HostSession(cmd, data, modem)
        modem.start(session)
        modem._connected = True
        modem.station.role = 'responder'
        modem.station.state = A.VaraState.CONNECTED
        session.state = SessionState.CONNECTED
        session._dispatch('DISCONNECT')
        assert appcmd.recv(4096) == b'OK\r'
        modem.poll()
        assert appcmd.recv(4096).startswith(b'WRONG DISCONNECT:')
        assert session.state == SessionState.CONNECTED
        assert not modem._closing
    finally:
        for sock in (cmd, appcmd, data, appdata): sock.close()
