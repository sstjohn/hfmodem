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


def command_line(sock):
    """Read one CR line without assuming TCP callback boundaries."""
    line = bytearray()
    while not line.endswith(b'\r'):
        chunk = sock.recv(1)
        assert chunk, 'command socket closed before CR'
        line += chunk
    return bytes(line)


class Observer:
    def __init__(self):
        self.closes = 0
        self.warnings = []
        self.buffers = []
        self.data = []
        self.pending_events = []
    def modem_disconnected(self): self.closes += 1
    def modem_warning(self, message): self.warnings.append(message)
    def modem_buffer(self, depth): self.buffers.append(depth)
    def modem_connected(self, *args): pass
    def modem_ptt(self, on): pass
    def modem_data_received(self, data): self.data.append(data)
    def modem_pending(self, cancel=False): self.pending_events.append(cancel)


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


def test_called_station_disconnect_keys_the_observed_responder_close(modem, monkeypatch):
    sent = []
    monkeypatch.setattr(modem.station, '_send_burst', lambda kind: sent.append(kind) or True)
    modem.station.role = 'responder'
    modem.station.state = A.VaraState.CONNECTED
    modem._connected = True
    modem.disconnect()
    modem.poll()
    assert not modem._closing
    assert not modem.observer.warnings
    assert sent == [F.SESSION_DISCONNECT_REQ_RESPONDER]
    assert modem.station.state == A.VaraState.DISCONNECTED
    assert modem.observer.closes == 1


def test_chat_on_called_station_asks_for_the_turn_to_close_in(modem, monkeypatch):
    """Stock's CHAT ON answerer asks with its 32-symbol request and closes in
    the turn the caller hands it (bench 3 s2s-2300 221.8)."""
    sent = []
    monkeypatch.setattr(modem.station, '_send_burst', lambda kind: sent.append(kind) or True)
    modem.station.role = 'responder'
    modem.station.chat = True
    modem.station.state = A.VaraState.CONNECTED
    modem._connected = True
    modem.disconnect()
    modem.poll()
    assert sent == [F.SESSION_TURN_REQUEST_RESPONDER]
    assert modem.station.state == A.VaraState.CONNECTED and modem._closing
    assert modem.observer.closes == 0


def test_refused_responder_close_retries_without_false_disconnection(modem, monkeypatch):
    outcomes = iter((False, True))
    sent = []
    def transmit(kind):
        sent.append(kind)
        return next(outcomes)
    monkeypatch.setattr(modem.station, '_send_burst', transmit)
    modem.station.role = 'responder'
    modem.station.state = A.VaraState.CONNECTED
    modem._connected = True
    modem.disconnect()
    modem.poll()
    assert modem.connected and modem._closing
    assert modem.station.state == A.VaraState.DISCONNECTING
    assert modem.observer.closes == 0
    modem.poll()
    assert sent == [F.SESSION_DISCONNECT_REQ_RESPONDER] * 2
    assert not modem.connected and not modem._closing
    assert modem.observer.closes == 1


@pytest.mark.parametrize('on', [False, True])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_listen_during_contact_uses_real_close_and_retains_next_listen_selection(modem, monkeypatch, on, role):
    sent = []
    hs = modem.station
    monkeypatch.setattr(hs, '_send_burst', lambda kind: sent.append(kind) or True)
    hs.role = role
    hs.state = A.VaraState.CONNECTED
    modem._connected = True
    modem._listen = on  # Repeating the current selection still closes a link.
    modem.set_listen(on)
    with pytest.raises(ConnectionAbortedError, match='closing'):
        modem.transmit(b'after LISTEN admission')
    modem.poll()
    expected = F.SESSION_DISCONNECT_REQ if role == 'initiator' else F.SESSION_DISCONNECT_REQ_RESPONDER
    assert sent == [expected]
    assert modem.observer.closes == 1
    assert not modem.connected
    assert modem.station.state == (A.VaraState.LISTENING if on else A.VaraState.DISCONNECTED)


@pytest.mark.parametrize('on', [False, True])
def test_listen_cannot_erase_a_refused_close_or_reopen_payload_admission(modem, monkeypatch, on):
    hs = modem.station
    hs.role = 'responder'
    hs.state = A.VaraState.CONNECTED
    modem._connected = True
    sent = []
    monkeypatch.setattr(hs, '_send_burst', lambda kind: sent.append(kind) or False)
    modem.disconnect()
    modem.poll()
    assert hs.state == A.VaraState.DISCONNECTING
    modem.set_listen(on)
    modem.poll()
    assert modem.station is hs
    assert hs.state == A.VaraState.DISCONNECTING
    assert modem.connected and modem.observer.closes == 0
    with pytest.raises(ConnectionAbortedError, match='closing'):
        modem.transmit(b'cannot pass the close boundary')
    assert sent == [F.SESSION_DISCONNECT_REQ_RESPONDER] * 2


@pytest.mark.parametrize('on', [False, True])
def test_listen_retires_pending_originals_without_delivery_credit(modem, monkeypatch, on):
    hs = modem.station
    hs.role = 'initiator'
    hs.caller, hs.called = 'W9SSJ', 'WI9YZA'
    hs.state = A.VaraState.CONNECTED
    hs._answering = True  # Host enqueue while composing a peer response.
    modem._connected = True
    original = bytes(range(256)) * 2
    modem.transmit(original)
    modem.poll()
    assert hs.data_pending and modem.accepted_total == len(original)
    sent = []
    monkeypatch.setattr(hs, '_send_burst', lambda kind: sent.append(kind) or True)
    modem.set_listen(on)
    modem.poll()
    assert sent == [F.SESSION_DISCONNECT_REQ]
    assert modem.observer.closes == 1
    assert modem.confirmed_total == 0
    assert modem.discarded_total == len(original)
    assert modem.retired[-1]['unconfirmed'] == original
    assert modem.retired[-1]['reason'] == ('LISTEN ON' if on else 'LISTEN OFF')
    assert modem.station.pending_bytes == 0


def test_refused_listen_close_preserves_originals_until_real_retirement(modem, monkeypatch):
    hs = modem.station
    hs.role = 'responder'
    hs.caller, hs.called = 'W9SSJ', 'WI9YZA'
    hs.state = A.VaraState.CONNECTED
    hs._answering = True
    modem._connected = True
    original = b'unconfirmed\x00\xff'
    modem.transmit(original)
    modem.poll()
    outcomes = iter((False, True))
    monkeypatch.setattr(hs, '_send_burst', lambda kind: next(outcomes))
    modem.set_listen(True)
    modem.poll()
    assert modem.station is hs and modem.connected
    assert modem.confirmed_total == modem.discarded_total == 0
    assert hs.payload_ledger.pending == len(original)
    assert not modem.retired and modem.observer.closes == 0
    modem.poll()
    assert modem.discarded_total == len(original)
    assert modem.retired[-1]['unconfirmed'] == original


def test_stop_cancels_unapplied_commands_on_the_owner(modem, monkeypatch):
    originated = []
    monkeypatch.setattr(A.VaraStationHandshake, 'originate',
                        lambda *args: originated.append(args))
    modem.connect('W9SSJ', 'WI9YZA')
    modem.transmit(b'old attachment')
    modem.stop()
    modem.poll()
    assert not originated
    assert modem.observer is None
    assert not modem.station.pending_bytes
    with pytest.raises(RuntimeError, match='fresh attachment'):
        modem.start(Observer())


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_disconnect_waits_for_queued_data_then_clears(modem, role):
    hs = modem.station
    hs.role = role
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


@pytest.mark.parametrize('invalid', ['INVALID', 'ON EXTRA', 'TEXT EXTRA', ''])
def test_live_compression_refusal_is_transactional(invalid):
    import socket
    from hfmodem.kestrel.host.server import HostSession
    modem = VaraHostModem(IO())
    command, peer = socket.socketpair()
    data, data_peer = socket.socketpair()
    peer.settimeout(1)
    session = HostSession(command, data, modem)
    modem.start(session)
    notifications = []
    def response(text):
        session._dispatch(text)
        while True:
            line = command_line(peer)
            if line == b'REGISTERED HFMODEM\r':
                notifications.append(line)
            else:
                return line
    try:
        assert session.compression == modem._wanted_compression
        assert response('COMPRESSION FILES') == b'OK\r'
        modem.poll()
        pending = modem._events.qsize()
        assert response('COMPRESSION ' + invalid) == b'WRONG\r'
        assert notifications == [b'REGISTERED HFMODEM\r']
        assert session.compression == modem._wanted_compression == modem._compression == 'FILES'
        assert modem._events.qsize() == pending
        modem.poll()
        assert response('COMPRESSION OFF') == b'OK\r'
        modem.poll()
        assert session.compression == modem._wanted_compression == modem._compression == 'OFF'
    finally:
        session.close()
        session.ensure_retired()
        peer.close()
        data_peer.close()


def test_live_compression_default_selects_an_effective_transmit_mode():
    modem = VaraHostModem(IO())
    assert modem._compression == modem._wanted_compression == 'TEXT'
    assert C.caller_mode(modem._compression, True) == modem._compression


@pytest.mark.parametrize('selection', ['TEXT', 'ON'])
def test_live_text_selection_reaches_the_owner_and_negotiated_mode(selection, monkeypatch):
    import socket
    from hfmodem.kestrel.host.server import HostSession
    modem = VaraHostModem(IO())
    cmd, appcmd = socket.socketpair()
    data, appdata = socket.socketpair()
    appcmd.settimeout(1)
    session = HostSession(cmd, data, modem)
    modem.start(session)
    monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a, **kw: True)
    try:
        session._dispatch('COMPRESSION OFF')
        assert command_line(appcmd) == b'OK\r'
        modem.poll()
        assert command_line(appcmd) == b'REGISTERED HFMODEM\r'
        session._dispatch('COMPRESSION '+selection)
        assert command_line(appcmd) == b'OK\r'
        assert session.compression == modem._wanted_compression == 'TEXT'
        assert modem._compression == 'OFF'  # The owner has not applied it yet.
        modem.poll()
        assert modem._compression == modem.station.compression == 'TEXT'
        session._dispatch('CONNECT W9SSJ WI9YZA')
        assert command_line(appcmd) == b'OK\r'
        modem.poll()
        assert modem.station.session_compression == 'TEXT'
        frame = F.link_setup_frame('W9SSJ', high_speed=True,
                                   compression=modem.station.session_compression)
        assert frame[7] == 0x83
        pending = modem._events.qsize()
        session._dispatch('COMPRESSION FILES')
        lines = appcmd.recv(4096).split(b'\r')
        assert b'WRONG' in lines
        assert session.compression == modem._wanted_compression == 'TEXT'
        assert modem._events.qsize() == pending
    finally:
        session.close()
        session.ensure_retired()
        appcmd.close()
        appdata.close()


def test_direct_legacy_compression_selection_is_canonical_on_the_owner():
    modem = VaraHostModem(IO())
    modem.start(Observer())
    modem.set_compression('ON')
    modem.poll()
    assert modem._compression == modem._wanted_compression == modem.station.compression == 'TEXT'


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


def test_unavailable_role_close_keeps_host_session_connected(modem):
    import socket
    from hfmodem.kestrel.host.server import HostSession, SessionState
    cmd, appcmd = socket.socketpair()
    data, appdata = socket.socketpair()
    appcmd.settimeout(1)
    try:
        session = HostSession(cmd, data, modem)
        modem.start(session)
        modem._connected = True
        modem.station.role = None
        modem.station.state = A.VaraState.CONNECTED
        session.state = SessionState.CONNECTED
        session._dispatch('DISCONNECT')
        assert command_line(appcmd) == b'OK\r'
        modem.poll()
        assert command_line(appcmd) == b'REGISTERED HFMODEM\r'
        assert command_line(appcmd).startswith(b'WRONG DISCONNECT:')
        assert session.state == SessionState.CONNECTED
        assert not modem._closing
    finally:
        for sock in (cmd, appcmd, data, appdata): sock.close()
