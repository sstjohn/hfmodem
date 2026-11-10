# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Failed real acquisition retires the scanner's early pending warning."""
import pytest

from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F
from .test_compression import IO
from .test_vara_host_modem import Observer


@pytest.fixture
def incoming(monkeypatch):
    monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a: True)
    modem = VaraHostModem(IO(), bandwidth='500')
    observer = Observer()
    observer.pending_events = []
    observer.modem_pending = lambda cancel=False: observer.pending_events.append(cancel)
    modem.start(observer)
    modem.set_mycall(['W9SSJ'])
    modem.set_listen(True)
    modem.poll()
    hs = modem.station
    kind = F.for_bw(F.connect_request('500'), '500')
    hs.on_rx_tones(F.handshake_tones('W9SSJ', kind), F.connect_request('500'))
    assert hs.state == A.VaraState.CONNECTING
    assert observer.pending_events == [False]
    return modem, observer


@pytest.mark.parametrize('action', ['abort', 'disconnect', 'failure'])
def test_failed_acquisition_cancels_pending_once_and_retires_originals(incoming, action):
    modem, observer = incoming
    original = b'not acknowledged\x00\xff'
    modem.transmit(original)
    modem.poll()
    if action == 'failure':
        modem.station.state = A.VaraState.DISCONNECTED
    else:
        getattr(modem, action)()
    modem.poll()
    modem.poll()
    assert observer.pending_events == [False, True]
    assert modem.confirmed_total == 0 and modem.discarded_total == len(original)
    assert modem.retired[-1]['unconfirmed'] == original
    assert modem.station.state == A.VaraState.LISTENING


def test_establishment_retires_pending_without_reporting_acquisition_failure(incoming):
    modem, observer = incoming
    modem.station.state = A.VaraState.CONNECTED
    modem.station.io.connected('K7ABC', 'W9SSJ', '500')
    modem.abort()
    modem.poll()
    assert observer.pending_events == [False]
    assert observer.closes == 1


def test_lost_attachment_clears_pending_without_publishing_to_retired_observer(incoming):
    modem, observer = incoming
    modem.stop()
    modem.poll()
    assert observer.pending_events == [False]
    assert modem.observer is None
    assert not modem.incoming_pending


def test_new_acquisition_gets_its_own_pending_and_cancellation(incoming):
    modem, observer = incoming
    modem.abort()
    modem.poll()
    hs = modem.station
    kind = F.for_bw(F.connect_request('500'), '500')
    hs.on_rx_tones(F.handshake_tones('W9SSJ', kind), F.connect_request('500'))
    modem.abort()
    modem.poll()
    assert observer.pending_events == [False, True, False, True]


@pytest.mark.parametrize('bandwidth', ['500', '2300', '2750'])
def test_owner_times_incoming_failure_from_detection_before_its_response_output(bandwidth):
    from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
    from .test_session_retry import Audio, Clock

    clock = Clock()
    audio = Audio(clock)
    owner = AudioHostRuntime(audio, clock=clock)
    modem = owner.modem_factory()
    observer = Observer()
    modem.start(observer)
    modem.set_bandwidth(bandwidth)
    modem.set_mycall(['W9SSJ'])
    modem.set_listen(True)
    owner.step(timeout=0)
    hs = modem.station
    detected_at = clock.now
    kind = F.for_bw(F.connect_request(bandwidth), bandwidth)
    hs.on_rx_tones(F.handshake_tones('W9SSJ', kind), F.connect_request(bandwidth))
    assert clock.now > detected_at  # Real response rendering and output occurred.
    owner.step(timeout=0)
    assert owner._attempt_end == pytest.approx(detected_at + 30)
    clock.now = detected_at + 30 - .001
    owner.step(timeout=0)
    assert observer.pending_events == [False]
    clock.now = detected_at + 30
    owner.step(timeout=0)
    owner.step(timeout=0)
    assert observer.pending_events == [False, True]
    assert observer.closes == 0  # The incoming contact was never established.
    assert modem.station.state == A.VaraState.LISTENING


def test_new_recognized_request_renews_the_incoming_clock_and_warning():
    from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
    from .test_session_retry import Audio, Clock

    clock = Clock()
    audio = Audio(clock)
    owner = AudioHostRuntime(audio, clock=clock)
    modem = owner.modem_factory()
    observer = Observer()
    modem.start(observer)
    modem.set_bandwidth('500')
    modem.set_mycall(['W9SSJ'])
    modem.set_listen(True)
    owner.step(timeout=0)
    hs = modem.station
    kind = F.for_bw(F.connect_request('500'), '500')
    tones = F.handshake_tones('W9SSJ', kind)
    hs.on_rx_tones(tones, F.connect_request('500'))
    owner.step(timeout=0)
    first_deadline = owner._attempt_end
    clock.now = 10
    hs.on_rx_tones(F.handshake_tones('N0ZZZ', kind), F.connect_request('500'))
    owner.step(timeout=0)
    assert owner._attempt_end == first_deadline
    assert observer.pending_events == [False]
    clock.now = 15
    hs.on_rx_tones(tones, F.connect_request('500'))
    owner.step(timeout=0)
    assert owner._attempt_end == 45
    clock.now = first_deadline
    owner.step(timeout=0)
    assert observer.pending_events == [False, False]
    assert modem.station is hs and hs.state == A.VaraState.CONNECTING
    clock.now = 45
    owner.step(timeout=0)
    assert observer.pending_events == [False, False, True]
    assert observer.closes == 0


def test_outgoing_attempt_keeps_its_distinct_timeout_and_terminal_notification():
    from .test_session_retry import runtime

    owner, modem, audio, clock = runtime(mode='P2P')
    owner.step(timeout=0)
    deadline = owner._attempt_end
    assert deadline > clock.now + 59
    clock.now = deadline
    owner.step(timeout=0)
    assert modem.observer.pending_events == []
    assert modem.observer.closes == 1


def test_native_acquisition_and_failure_reach_the_actual_command_socket(monkeypatch):
    import socket
    from hfmodem.kestrel.host.server import HostSession

    monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a: True)
    modem = VaraHostModem(IO(), bandwidth='500')
    command, peer = socket.socketpair()
    data, data_peer = socket.socketpair()
    peer.settimeout(.05)
    session = HostSession(command, data, modem)
    try:
        modem.start(session)
        session._dispatch('MYCALL W9SSJ')
        session._dispatch('LISTEN ON')
        modem.poll()
        kind = F.for_bw(F.connect_request('500'), '500')
        modem.station.on_rx_tones(F.handshake_tones('W9SSJ', kind), F.connect_request('500'))
        modem.abort()
        modem.poll()
        modem.poll()
        wire = bytearray()
        while True:
            try:
                wire.extend(peer.recv(4096))
            except socket.timeout:
                break
        assert wire.split(b'\r') == [b'OK', b'OK', b'REGISTERED HFMODEM',
                                     b'PENDING', b'CANCELPENDING', b'']
    finally:
        session.close()
        modem.poll()
        session.ensure_retired()
        peer.close()
        data_peer.close()
