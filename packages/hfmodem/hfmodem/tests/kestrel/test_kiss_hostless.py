# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Hostless KISS owner cadence with fake capture, playback and time."""
import threading
from types import SimpleNamespace

import pytest

from hfmodem.ax25 import kiss
from hfmodem.kestrel.broadcast import Broadcast, encode_ui
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime, KissObserver
from hfmodem.kestrel.host.kiss import KissPort, _Client
from hfmodem.kestrel.host.kiss_tx import KissTxDisposition as Outcome
from hfmodem.kestrel.host.server import HostSession, VaraServer
from hfmodem.kestrel.host.station_runtime import StationTransport
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.rx.beacon import BeaconResult
from hfmodem.kestrel.vara.vara_arq import VaraIO, VaraState


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Transport(VaraIO):
    def __init__(self):
        self.hs = None
        self.receiving = False
        self.sent, self.keys, self.notes = [], [], []
        self.admission = None
        self.result = Outcome.SENT
        self.unkeyed = True
        self.failure = None
        self.received = []

    def next_rx_burst(self, timeout, hs):
        for message in self.received:
            hs.io.cq(message)
        self.received.clear()
        return None

    def admit_kiss(self, modem):
        assert modem._lock._is_owned()
        assert modem.kiss_tx_queue._claimed is not None
        return self.admission

    def key(self, on):
        self.keys.append(on)

    def tx(self, audio):
        if self.failure:
            raise self.failure
        self.sent.append(audio)

    def tx_metadata(self, **details):
        self.notes.append(details)

    def kiss_tx_result(self):
        return self.result

    def unkey_confirmed(self):
        return self.unkeyed


class Client:
    def __init__(self):
        self.wire = []

    def settimeout(self, timeout): pass
    def sendall(self, wire): self.wire.append(wire)


@pytest.fixture
def owner(monkeypatch):
    from hfmodem.kestrel.host import vara_modem
    from hfmodem.kestrel.tx import broadcast
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda message: message)
    monkeypatch.setattr(vara_modem, 'synth_beacon', lambda text: text)
    clock, transport, logs = Clock(), Transport(), []
    runtime = AudioHostRuntime(transport, clock=clock)
    server = VaraServer(cmd_port=0, data_port=0, modem_factory=runtime.modem_factory,
                        log=lambda *args: logs.append(args))
    yield runtime, server, transport, clock, logs
    server.stop()


def submit(server, *, command=0, source='K1ABC', info=b'hello'):
    server._kiss_frame(command, encode_ui(source, 'ALL', info, command=command))


def test_hostless_ui_keys_on_standing_modem(owner):
    runtime, server, transport, clock, _ = owner
    submit(server)
    runtime.step(0)
    modem = runtime.modem
    assert isinstance(modem.observer, KissObserver)
    assert modem._mycall == [] and not modem._listen
    assert modem.station.state == VaraState.DISCONNECTED
    assert not transport.sent and len(server.kiss_tx_queue) == 1
    clock.now = 1.999
    runtime.step(0)
    assert not transport.sent
    clock.now = 2
    runtime.step(0)
    assert len(transport.sent) == 1 and transport.keys == [True, False]
    assert transport.sent[0].command == 0 and transport.sent[0].source == 'K1ABC'
    assert not server.kiss_tx_queue and server._session is None


def test_hostless_receive_fans_out_broadcast_and_beacon(owner):
    runtime, server, transport, _, _ = owner
    clients = [Client(), Client()]
    port = object.__new__(KissPort)
    port._clients = {client: _Client() for client in clients}
    port._clock = Clock()
    port._lock = threading.Lock()
    runtime.bind_kiss(port, server.kiss_tx_queue)
    message = Broadcast(1, 'K1ABC', 'ALL', b'hello')
    transport.received = [message, BeaconResult('20OEN63BCW9SSJ')]
    runtime.step(0)
    for client in clients:
        assert list(kiss.Parser().feed(b''.join(port._clients[client].output))) == [
            (0, 1, message.encode()), (0, 2, '20OEN63BCW9SSJ'.ljust(20).encode())]
    assert not transport.sent


def test_attach_detach_retires_standing_and_preserves_queue(owner, monkeypatch):
    runtime, server, transport, clock, _ = owner
    submit(server)
    runtime.step(0)
    standing = runtime.modem
    retired = []
    stop = standing.stop
    def retire():
        retired.append(True)
        stop()
    monkeypatch.setattr(standing, 'stop', retire)
    clock.now = .5
    attached = runtime.modem_factory()
    attached.start(KissObserver(SimpleNamespace(_kiss=None)))
    runtime.step(0)
    assert retired == [True] and standing.observer is None and not standing._ptt_active
    assert runtime.modem is attached and len(server.kiss_tx_queue) == 1
    clock.now = 1
    attached.stop()
    runtime.step(0)
    assert runtime.modem not in (standing, attached)
    assert runtime.modem._mycall == [] and len(server.kiss_tx_queue) == 1
    clock.now = 2
    runtime.step(0)
    assert len(transport.sent) == 1 and not server.kiss_tx_queue


def test_attach_requires_confirmed_standing_unkey(owner):
    runtime, _, transport, _, _ = owner
    runtime.step(0)
    standing = runtime.modem
    attached = runtime.modem_factory()
    attached.start(KissObserver(SimpleNamespace(_kiss=None)))
    transport.unkeyed = False
    with pytest.raises(RuntimeError, match='unkey was not confirmed'):
        runtime.step(0)
    assert standing._stopping and runtime.modem is standing
    assert runtime.modem is not attached and not transport.sent


def test_occupancy_fall_and_activity_inside_clearance_holdoff_postpone(owner):
    runtime, server, transport, clock, _ = owner
    submit(server)
    runtime.step(0)
    clock.now, transport.receiving = 1, True
    runtime.step(0)
    clock.now = 1.25
    runtime.step(0)
    assert runtime.modem._channel_busy
    clock.now, transport.receiving = 3, False
    runtime.step(0)
    clock.now = 4.5
    runtime.step(0)
    assert not runtime.modem._channel_busy and server.kiss_tx_queue.quiet_at == 3
    clock.now = 4.99
    runtime.step(0)
    assert not transport.sent
    clock.now, transport.receiving = 4.99, True
    runtime.step(0)
    clock.now = 5.25
    runtime.step(0)
    clock.now, transport.receiving = 6, False
    runtime.step(0)
    clock.now = 7.5
    runtime.step(0)
    clock.now = 7.99
    runtime.step(0)
    assert not transport.sent
    clock.now = 8
    runtime.step(0)
    assert len(transport.sent) == 1


def test_fifo_waits_two_seconds_from_previous_unkey(owner):
    runtime, server, transport, clock, _ = owner
    submit(server, info=b'first')
    submit(server, command=1, info=b'second')
    clock.now = 2
    runtime.step(0)
    assert [m.info for m in transport.sent] == [b'first']
    clock.now = 3.999
    runtime.step(0)
    assert len(transport.sent) == 1
    clock.now = 4
    runtime.step(0)
    assert [m.info for m in transport.sent] == [b'first', b'second']


@pytest.mark.parametrize('gate', ['linked', 'connecting', 'incoming', 'keyed', 'intent'])
def test_keying_gates_keep_head(owner, gate):
    runtime, server, transport, clock, _ = owner
    submit(server)
    runtime.step(0)
    modem = runtime.modem
    if gate in ('linked', 'connecting'):
        modem.station.state = VaraState.CONNECTED if gate == 'linked' else VaraState.CONNECTING
    elif gate == 'incoming':
        modem._incoming_pending_at = 0
    elif gate == 'keyed':
        modem._ptt_active = True
    else:
        modem._intent_active = True
    clock.now = 10
    assert modem.service_kiss_queue() is None
    assert not transport.sent and len(server.kiss_tx_queue) == 1
    modem.station.state = VaraState.DISCONNECTED
    modem._incoming_pending_at = None
    modem._ptt_active = modem._intent_active = False
    assert modem.service_kiss_queue() is Outcome.SENT
    assert len(transport.sent) == 1


def test_bandwidth_change_and_link_end_keep_frame(owner):
    runtime, server, transport, clock, _ = owner
    submit(server)
    runtime.step(0)
    modem = runtime.modem
    old_station = modem.station
    modem.set_bandwidth('500')
    runtime.step(0)
    assert modem.station is not old_station and len(server.kiss_tx_queue) == 1
    modem.station.state = VaraState.CONNECTED
    modem._connected = modem._intent_active = True
    clock.now = 3
    assert modem.service_kiss_queue() is None
    modem.abort()
    runtime.step(0)
    assert len(transport.sent) == 1 and not server.kiss_tx_queue


@pytest.mark.parametrize('outcome', [Outcome.CONTENDED, Outcome.REFUSED_BEFORE_KEY])
def test_admission_retry_keeps_head(owner, outcome):
    runtime, server, transport, clock, logs = owner
    submit(server)
    runtime.step(0)
    transport.admission = outcome
    clock.now = 2
    assert runtime.modem.service_kiss_queue() is outcome
    head = server.kiss_tx_queue._requests[0]
    assert head.disposition is outcome and not transport.keys
    clock.now = 599
    assert runtime.modem.service_kiss_queue() is outcome
    assert server.kiss_tx_queue._requests[0] is head
    transport.admission = None
    assert runtime.modem.service_kiss_queue() is Outcome.SENT
    assert not server.kiss_tx_queue and len(transport.sent) == 1
    if outcome is Outcome.CONTENDED:
        assert sum('contended' in message for _, message in logs) == 1


def test_refused_head_expires_at_ten_minutes_but_contention_does_not(owner):
    runtime, server, transport, clock, logs = owner
    submit(server)
    runtime.step(0)
    transport.admission, clock.now = Outcome.CONTENDED, 600
    runtime.modem.service_kiss_queue()
    assert len(server.kiss_tx_queue) == 1
    transport.admission = Outcome.REFUSED_BEFORE_KEY
    runtime.modem.service_kiss_queue()
    assert not server.kiss_tx_queue and not transport.keys
    assert any('10 minutes' in message for _, message in logs)


@pytest.mark.parametrize('failure', ['partial', 'exception', 'unkey'])
def test_terminal_failure_drops_head_and_raises_owner_failure(owner, failure):
    runtime, server, transport, clock, logs = owner
    submit(server)
    runtime.step(0)
    if failure == 'partial':
        transport.result = Outcome.PARTIAL
    elif failure == 'exception':
        transport.failure = OSError('playback failed')
    else:
        transport.unkeyed = False
    clock.now = 2
    with pytest.raises((OSError, RuntimeError)):
        runtime.step(0)
    assert not server.kiss_tx_queue and transport.keys == [True, False]
    expected = 'UNKEY_UNCONFIRMED' if failure == 'unkey' else 'PARTIAL'
    assert any(expected in message for _, message in logs)
    transport.failure, transport.unkeyed = None, True
    runtime.step(0)
    assert transport.keys == [True, False]


def test_queue_bound_drops_newest_and_stop_logs_once(owner):
    _, server, _, _, logs = owner
    for i in range(65):
        submit(server, info=str(i).encode())
    assert len(server.kiss_tx_queue) == 64
    assert server.kiss_tx_queue._requests[-1].value.info == b'63'
    assert sum('full' in message for _, message in logs) == 1
    server.kiss_tx_queue.stop()
    assert not server.kiss_tx_queue
    assert sum('discarded 64' in message for _, message in logs) == 1


@pytest.mark.parametrize('close', ['ABORT', 'DISCONNECT'])
def test_kiss_queue_survives_attachment_close(owner, close):
    runtime, server, transport, clock, _ = owner
    modem = runtime.modem_factory()
    cmd, data = Client(), Client()
    session = HostSession(cmd, data, modem)
    modem.start(session)
    session._dispatch('MYCALL W9SSJ')
    runtime.step(0)
    cmd.wire.clear()
    submit(server, command=1)
    session._dispatch(close)
    runtime.step(0)
    assert not transport.sent and len(server.kiss_tx_queue) == 1
    assert b'OK\r' in b''.join(cmd.wire)
    clock.now = 2
    runtime.step(0)
    assert [message.info for message in transport.sent] == [b'hello']
    assert transport.sent[0].source == 'K1ABC' and not server.kiss_tx_queue
    submit(server, command=1, info=b'again')
    runtime.step(0)
    assert len(transport.sent) == 1
    clock.now = 4
    runtime.step(0)
    assert [message.info for message in transport.sent] == [b'hello', b'again']
    assert not server.kiss_tx_queue


@pytest.mark.parametrize('cancel', ['ABORT', 'DISCONNECT', 'detach'])
def test_beaconforce_stays_off_the_kiss_queue(owner, cancel):
    runtime, server, transport, clock, _ = owner
    modem = runtime.modem_factory()
    cmd, data = Client(), Client()
    session = HostSession(cmd, data, modem)
    modem.start(session)
    session._dispatch('MYCALL W9SSJ')
    runtime.step(0)
    modem._channel_busy = True
    session._dispatch('BEACONFORCE 20OEN63BCW9SSJ')
    assert not server.kiss_tx_queue
    if cancel == 'detach':
        modem.stop()
    else:
        session._dispatch(cancel)
    modem._channel_busy = False
    for now in (1, 5, 30):
        clock.now = now
        runtime.step(0)
    assert not server.kiss_tx_queue
    assert transport.sent in ([], ['20OEN63BCW9SSJ'])
    assert len(transport.sent) <= 1


def test_owner_running_before_the_port_is_bound_still_delivers(owner):
    runtime, server, transport, _, _ = owner
    runtime._kiss = None
    runtime.step(0)
    standing = runtime.modem
    client = Client()
    port = object.__new__(KissPort)
    port._clients = {client: _Client()}
    port._clock = Clock()
    port._lock = threading.Lock()
    runtime.bind_kiss(port, server.kiss_tx_queue)
    message = Broadcast(1, 'K1ABC', 'ALL', b'late bind')
    transport.received = [message]
    runtime.step(0)
    assert runtime.modem is standing
    assert list(kiss.Parser().feed(b''.join(port._clients[client].output))) == [(0, 1, message.encode())]


def test_head_is_claimed_before_transport_admission_against_another_owner(owner):
    runtime, server, transport, clock, _ = owner
    submit(server)
    runtime.step(0)
    other = VaraHostModem(Transport(), clock=clock, kiss_tx_queue=server.kiss_tx_queue)
    other.start(KissObserver(SimpleNamespace(_kiss=None)))
    attempts = []
    def admit(modem):
        assert modem._lock._is_owned()
        attempts.append(other.service_kiss_queue())
        return None
    transport.admit_kiss = admit
    clock.now = 2
    runtime.step(0)
    assert attempts == [None] and not other._transport.sent
    assert len(transport.sent) == 1 and not server.kiss_tx_queue


def test_station_transport_distinguishes_contention_refusal_and_partial():
    owned = [False]
    contact = SimpleNamespace(receive_boundary=(0, 0), admit=lambda **kw: owned[0])
    link = SimpleNamespace(station=SimpleNamespace(
        arbiter=SimpleNamespace(receive_boundary=(0, 0), disabled=set())))
    transport = StationTransport(link)
    transport.contact = contact
    modem = object()
    assert transport.admit_kiss(modem) is Outcome.CONTENDED
    owned[0] = True
    assert transport.admit_kiss(modem) is None
    transport.fault.set()
    assert transport.admit_kiss(modem) is Outcome.REFUSED_BEFORE_KEY
    transport._sent, transport._kiss_keyed = False, False
    assert transport.kiss_tx_result() is Outcome.REFUSED_BEFORE_KEY
    transport._report_ptt(True)
    assert transport.kiss_tx_result() is Outcome.PARTIAL
    assert not transport.unkey_confirmed()
    transport._report_ptt(False)
    transport._sent = True
    assert transport.kiss_tx_result() is Outcome.SENT and transport.unkey_confirmed()
    transport.fault.clear()
    link.station.arbiter.disabled.add('vara')
    assert transport.admit_kiss(modem) is Outcome.REFUSED_BEFORE_KEY


def test_audio_transport_typed_result_uses_key_refusal():
    transport = VaraIO()
    transport._refused = True
    assert transport.kiss_tx_result() is Outcome.REFUSED_BEFORE_KEY
    transport._refused = False
    assert transport.kiss_tx_result() is Outcome.SENT
    transport.tx_went_out = lambda: False
    assert transport.kiss_tx_result() is Outcome.PARTIAL
    transport.rig = SimpleNamespace(retired=False, keyed=True)
    assert transport.admit_kiss(None) is Outcome.CONTENDED
    transport.rig.retired = True
    assert transport.admit_kiss(None) is Outcome.REFUSED_BEFORE_KEY


def test_unkey_holdoff_survives_detach_and_station_replacement(owner):
    runtime, server, transport, clock, _ = owner
    modem = runtime.modem_factory()
    modem.start(KissObserver(SimpleNamespace(_kiss=None)))
    submit(server)
    runtime.step(0)
    clock.now = 5
    modem.station.io.report_ptt(True)
    clock.now = 6
    modem.station.io.report_ptt(False)
    modem.stop()
    clock.now = 7
    runtime.step(0)
    assert not transport.sent and server.kiss_tx_queue.unkey_at == 6
    clock.now = 8
    runtime.step(0)
    assert len(transport.sent) == 1


def test_station_runtime_wires_server_queue_and_releases_retired_claimant(monkeypatch):
    from hfmodem.kestrel.host.station_runtime import StationVaraRuntime
    from hfmodem.kestrel.tx import broadcast
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda message: message)
    reserved, released = [], []
    arbiter = SimpleNamespace(receive_boundary=(0, 0),
        claim_contact=lambda protocol, token, **kw: token or object(),
        release_contact=lambda protocol, token: released.append(token))
    station = SimpleNamespace(arbiter=arbiter, audio=SimpleNamespace(samples=0))
    def emit(samples, bandwidth, *, on_ptt, responding, marks):
        reserved.append(bandwidth)
        on_ptt(True)
        for k in range(len(marks[0])):
            marks[1](k)
        on_ptt(False)
        return True
    runtime = StationVaraRuntime(SimpleNamespace(station=station, emit=emit))
    clock = Clock()
    runtime.owner.clock = clock
    server = VaraServer(cmd_port=0, data_port=0, modem_factory=runtime.modem_factory)
    try:
        submit(server)
        runtime.owner.step(0)
        old = runtime.owner.modem
        assert runtime.owner.kiss_tx_queue is server.kiss_tx_queue
        assert runtime.transport.admit_kiss(old) is None
        attached = runtime.modem_factory()
        attached.start(KissObserver(SimpleNamespace(_kiss=None)))
        runtime.owner.step(0)
        assert released and runtime.transport._claimant is None
        attached.stop()
        clock.now = 2
        runtime.owner.step(0)
        assert reserved == ['500'] and not server.kiss_tx_queue
    finally:
        server.stop()
        runtime.stop()


def test_transport_key_refusal_retries_without_an_unkey_holdoff(owner):
    runtime, server, transport, clock, _ = owner
    submit(server)
    runtime.step(0)
    transport.kiss_tx_result = lambda: VaraIO.kiss_tx_result(transport)
    refused = [True]
    def key(on):
        if on:
            transport._refused = refused[0]
    def tx(audio):
        if not transport._refused:
            transport.sent.append(audio)
    transport.key, transport.tx = key, tx
    clock.now = 2
    assert runtime.modem.service_kiss_queue() is Outcome.REFUSED_BEFORE_KEY
    assert len(server.kiss_tx_queue) == 1 and not transport.sent
    assert server.kiss_tx_queue.unkey_at is None
    refused[0] = False
    assert runtime.modem.service_kiss_queue() is Outcome.SENT
    assert len(transport.sent) == 1 and not server.kiss_tx_queue
