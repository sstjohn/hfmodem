# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA original-byte disposition, distinct from wire-container BUFFER."""
import socket

import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.vara import vara_arq as A, compression as C
from hfmodem.kestrel.vara.ledger import PayloadLedger
from hfmodem.tests.kestrel.test_compression import IO, station
from hfmodem.tests.kestrel.test_vara_host_modem import Observer, runtime_class


def test_plain_and_atomic_container_credit_and_retirement():
    ledger = PayloadLedger()
    ledger.queue(b'plain', b'plain')
    ledger.queue(b'container', b'original'*20, atomic=True)
    assert ledger.settle(8) == b'plain'
    assert ledger.confirmed == 5
    assert ledger.unconfirmed == b'original'*20
    assert ledger.settle(5) == b''
    assert ledger.settle(1) == b'original'*20
    assert ledger.confirmed == ledger.accepted == 165
    ledger.queue(b'unsent', b'unsent')
    assert ledger.retire() == b'unsent'
    assert ledger.confirmed == 165 and ledger.discarded == 6
    assert ledger.retire() == b''
    assert ledger.last_unconfirmed == b'unsent'


def test_negotiation_reframes_without_admitting_or_confirming_again():
    hs, _ = station('FILES')
    original = bytes(range(256))*128 + b'last block'
    hs.send(original)
    assert hs.payload_ledger.accepted == len(original)
    hs._start_compression('FILES')
    assert hs.payload_ledger.accepted == len(original)
    assert hs.payload_ledger.pending == len(original)
    assert hs.payload_ledger.confirmed == 0
    assert hs.payload_ledger.wire_pending == hs.pending_bytes
    assert b''.join(hs._txq) == C.encode_files(original)
    assert hs.payload_ledger.unconfirmed == original


def test_default_admission_bound_rejects_transactionally():
    hs, _ = station()
    hs.send(b'A'*65536)
    queue = tuple(hs._txq)
    with pytest.raises(BufferError, match='queue full'):
        hs.send(b'late')
    assert tuple(hs._txq) == queue
    assert hs.preconnect_data == (b'A'*65536,)
    assert hs.payload_ledger.accepted == 65536


@pytest.mark.parametrize('mode', ['OFF', 'FILES'])
def test_only_validated_over_settlement_credits_originals(mode, monkeypatch):
    hs, _ = station(mode)
    hs.state = A.VaraState.CONNECTED
    hs.caller = 'W9SSJ'
    hs._answering = True
    hs._start_compression(mode)
    original = bytes(range(256))*2
    hs.send(original)
    monkeypatch.setattr(hs, '_tx_data_over', lambda: False)
    monkeypatch.setattr(hs, '_release_turn', lambda idle=False: True)
    credited = []
    while hs._txq:
        wire = hs._txq.pop(0)
        size = phy.body_size(hs.bw, hs._tx_level)
        body = phy.vara_body(wire, hs.caller, body_len=size)
        hs._tx_pending = (body, 1, hs._tx_level)
        before = hs.payload_ledger.confirmed
        assert hs.payload_ledger.wire_pending >= len(wire)
        hs._took_control_burst(continue_reply=True)
        credited.append(hs.payload_ledger.confirmed-before)
    assert hs.payload_ledger.confirmed == len(original)
    assert hs.payload_ledger.wire_pending == 0
    if mode == 'FILES':
        assert credited[:-1] and not any(credited[:-1])
        assert credited[-1] == len(original)
    else:
        assert credited[0] > 0


def test_unapplied_stop_retains_originals_and_cannot_reopen():
    modem = VaraHostModem(IO())
    modem.start(Observer())
    modem.transmit(b'old original')
    modem.stop()
    modem.poll()
    assert modem.accepted_total == modem.discarded_total == 12
    assert modem.confirmed_total == 0
    assert modem.retired[-1]['unconfirmed'] == b'old original'
    generation = modem.generation
    modem.poll()
    assert modem.generation == generation
    with pytest.raises(ConnectionAbortedError, match='retired'):
        modem.transmit(b'late')


def test_rebinding_preserves_admission_and_abort_records_exact_originals():
    modem = VaraHostModem(IO())
    modem.start(Observer())
    modem.transmit(b'first')
    modem.transmit(b'second')
    modem.set_listen(True)
    modem.poll()
    assert modem.accepted_total == modem.station.payload_ledger.accepted == 11
    assert modem.station.payload_ledger.unconfirmed == b'firstsecond'
    modem.abort()
    modem.poll()
    assert modem.confirmed_total == 0
    assert modem.discarded_total == modem.accepted_total == 11
    assert modem.retired[-1]['unconfirmed'] == b'firstsecond'
    assert modem.station.payload_ledger.accepted == 0


def test_owner_queue_and_closing_admission_are_bounded():
    modem = VaraHostModem(IO())
    for _ in range(64):
        modem.set_listen(False)
    with pytest.raises(ConnectionAbortedError, match='command queue full'):
        modem.transmit(b'never admitted')
    assert modem.accepted_total == 0
    modem.poll()
    modem.transmit(b'accepted')
    modem.disconnect()
    with pytest.raises(ConnectionAbortedError, match='closing'):
        modem.transmit(b'late')
    assert modem.accepted_total == 8


def test_receiver_wait_cannot_deliver_into_retired_attachment(monkeypatch):
    class Audio(IO):
        def next_rx_burst(self, timeout, hs=None):
            runtime.modem.stop()
            return [0]*100
    runtime = runtime_class()(Audio())
    modem = runtime.modem_factory()
    modem.start(Observer())
    received = []
    monkeypatch.setattr(modem.station, 'on_rx_audio', received.append)
    runtime.step()
    assert not received
    assert modem.observer is None


def test_host_never_reports_ok_or_stores_settings_for_backend_refusal():
    from hfmodem.kestrel.host.server import HostSession, SessionState
    from hfmodem.kestrel.host.modem_core import LoopbackModem
    class Refuse(LoopbackModem):
        def set_compression(self, mode): raise ValueError('unavailable')
        def connect(self, *args): raise ValueError('unavailable')
    cmd, app = socket.socketpair()
    data, appdata = socket.socketpair()
    app.settimeout(1)
    try:
        session = HostSession(cmd, data, Refuse())
        previous = session.compression
        session._dispatch('COMPRESSION FILES')
        assert app.recv(100) == b'WRONG\r'
        assert session.compression == previous
        session._dispatch('CONNECT W9SSJ K7ABC')
        assert app.recv(100) == b'WRONG\r'
        assert session.state == SessionState.DISCONNECTED
    finally:
        for conn in (cmd, app, data, appdata): conn.close()


def test_active_or_queued_contact_rejects_configuration_replacement(monkeypatch):
    monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a, **kw: True)
    modem = VaraHostModem(IO())
    modem.set_mycall(['W9SSJ'])
    modem.set_bandwidth('500')
    modem.set_compression('FILES')
    modem.connect('W9SSJ', 'K7ABC')
    # Exact repeats of an already admitted control preserve their effect.
    modem.set_mycall(['W9SSJ'])
    modem.set_bandwidth('500')
    modem.set_compression('FILES')
    with pytest.raises(ValueError, match='already active'):
        modem.connect('W9SSJ', 'N0OTHER')
    with pytest.raises(ValueError, match='idle contact'):
        modem.set_bandwidth('2300')
    with pytest.raises(ValueError, match='idle contact'):
        modem.set_mycall(['N0OTHER'])
    with pytest.raises(ValueError, match='idle contact'):
        modem.set_compression('OFF')
    modem.poll()
    assert modem.station.bw == '500'
    assert modem.station.mycalls == ['W9SSJ']
    assert modem.station.session_compression == 'FILES'


def test_reentrant_stop_cancels_unapplied_bytes_and_prevents_transmission(monkeypatch):
    class Audio(IO):
        def __init__(self):
            super().__init__()
            self.transmitted = []
        def tx(self, samples): self.transmitted.append(samples)
    class StopOnPending(Observer):
        def modem_pending(self): modem.stop()
    audio = Audio()
    modem = VaraHostModem(audio)
    modem.start(StopOnPending())
    def originate(hs, *args):
        hs.io.pending()
        hs.io.key(True)
        try:
            hs.io.tx([1, 2, 3])
            assert not hs.io.tx_went_out()
        finally:
            hs.io.key(False)
    monkeypatch.setattr(A.VaraStationHandshake, 'originate', originate)
    modem.connect('W9SSJ', 'K7ABC')
    modem.transmit(b'queued behind connect')
    modem.poll()
    assert not audio.transmitted
    assert modem.observer is None
    assert modem.confirmed_total == 0
    assert modem.discarded_total == modem.accepted_total == 21
    assert modem.retired[-1]['unconfirmed'] == b'queued behind connect'
