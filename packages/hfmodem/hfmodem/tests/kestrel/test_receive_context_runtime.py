# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The audio owner keeps UI acquisition across host lifecycle boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
from hfmodem.kestrel.broadcast import Broadcast
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime, KissObserver
from hfmodem.kestrel.host.station_runtime import StationTransport
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.rx.ui_spans import UISpan
from hfmodem.kestrel.vara.vara_arq import VaraIO
from .test_rxwindow_adapter import station


class Transport(VaraIO):
    def __init__(self):
        self.hs = None
        self.input_clock = self.tune_epoch = 0
        self.receiving = False
        self.chunks = []

    def next_rx_burst(self, timeout, hs):
        if self.chunks:
            at, samples = self.chunks.pop(0)
            self.input_clock = max(self.input_clock, at + len(samples))
            self.receive_context.feed(samples, at, sink=hs.io.cq)
            hs.on_rx_stream(samples, sample_start=at)
        return None

    def retuned(self):
        self.tune_epoch += 1
        self.receive_context.retune(self.tune_epoch, self.input_clock)
        self.hs.retuned(tune_epoch=self.tune_epoch)

    def unkey_confirmed(self): return True
    def key(self, on): pass
    def log(self, message): pass


def owner():
    transport = Transport()
    runtime = AudioHostRuntime(transport, clock=lambda: 100.)
    delivered = []
    runtime.bind_kiss(SimpleNamespace(
        deliver_frame=lambda command, payload: delivered.append((command, payload))), None)
    return runtime, transport, delivered


def test_standalone_modems_and_station_replacements_share_transport_context():
    transport = Transport()
    modem = VaraHostModem(transport)
    context = modem.receive_context
    assert context is transport.receive_context is modem.station.receive_context
    assert modem.station.cq_detector is context.detector
    context.feed(np.zeros(4800), 0)
    context.ledger.publish(UISpan(1, 0, 4, 0))
    previous = modem.station
    modem.set_bandwidth('500')
    modem.poll()
    assert modem.station is not previous and modem.station.bw == '500'
    assert modem.station.receive_context is context
    assert modem.station.cq_detector is context.detector
    replacement = VaraHostModem(transport)
    assert replacement.receive_context is context
    assert replacement.station.cq_detector is context.detector
    runtime = AudioHostRuntime(transport)
    assert runtime.receive_context is context
    assert context.input_clock == 4800 and len(context.ledger.spans) == 1


def test_attach_detach_bw_change_keep_one_detector_and_ledger():
    runtime, transport, _ = owner()
    runtime.step(0)
    context = runtime.receive_context
    detector, ledger = context.detector, context.ledger
    ledger.publish(UISpan(1, 0, 4, 0))
    standing = runtime.modem.station
    attached = runtime.modem_factory()
    attached.start(KissObserver(runtime))
    runtime.step(0)
    assert attached.station is not standing
    attached.set_bandwidth('500')
    runtime.step(0)
    assert attached.station.receive_context is context
    assert attached.station.cq_detector is detector
    attached.stop()
    runtime.step(0)
    assert runtime.modem is not attached
    assert runtime.modem.station.cq_detector is detector
    assert runtime.receive_context.ledger is ledger and len(ledger.spans) == 1


def test_retune_clears_acquisition_and_ledger_but_keeps_owner_objects():
    runtime, transport, _ = owner()
    runtime.step(0)
    context = runtime.receive_context
    context.feed(np.zeros(4800), 0)
    transport.input_clock = 4800
    context.ledger.publish(UISpan(1, 0, 4, 0))
    detector = context.detector
    transport.retuned()
    assert context.detector is detector and context.input_clock == 4800
    assert not context.ledger.spans and context.ledger.tune_epoch == 1
    assert detector.tune_epoch == 1 and detector._source_count == 0
    assert detector._candidates == [] and detector._groups == []


def test_key_and_decoder_discontinuity_do_not_reset_detector(monkeypatch):
    hs, _ = station(bw='500')
    calls = []
    hs.cq_detector = SimpleNamespace(reset=lambda **kw: calls.append(kw))
    hs._key(True)
    hs._key(False)
    hs.rx_discontinuity()
    assert calls == []


def test_hostless_attach_detach_and_bw_change_do_not_duplicate_kiss_or_spans(monkeypatch):
    runtime, transport, delivered = owner()
    context = runtime.receive_context
    message = Broadcast(1, 'K1ABC', 'ALL', b'continuous owner')
    original = context.detector.feed
    def feed(samples, **kwargs):
        previous = context.input_clock
        result = original(samples, **kwargs)
        if previous < 4800 <= context.input_clock:
            context.ledger.publish(UISpan(1, 0, 4, 0))
            result.append(message)
        return result
    monkeypatch.setattr(context.detector, 'feed', feed)
    transport.chunks.append((0, np.zeros(4800)))
    runtime.step(0)
    assert delivered == [(message.command, message.encode())]
    attached = runtime.modem_factory()
    attached.start(KissObserver(runtime))
    for change in ('attach', 'bandwidth', 'detach'):
        if change == 'bandwidth':
            attached.set_bandwidth('500')
        elif change == 'detach':
            attached.stop()
        transport.chunks.append((0, np.zeros(4800)))
        runtime.step(0)
        assert runtime.modem.station.receive_context is context
        assert delivered == [(message.command, message.encode())]
        assert len(context.ledger.spans) == 1


def test_station_transport_batches_before_connected_decode_and_sends_gaps(monkeypatch):
    backend = SimpleNamespace(station=SimpleNamespace(
        arbiter=SimpleNamespace(receive_boundary=(0, 0)),
        audio=SimpleNamespace(samples=0)))
    transport = StationTransport(backend)
    runtime = AudioHostRuntime(transport, clock=lambda: 100.)
    runtime.step(0)
    context = runtime.receive_context
    order = []
    feed = context.feed
    def indexed(samples, start, **kwargs):
        order.append(('detector', start, len(samples)))
        return feed(samples, start, **kwargs)
    monkeypatch.setattr(context, 'feed', indexed)
    hs = runtime.modem.station
    stream = hs.on_rx_stream
    def receive(samples, **kwargs):
        order.append(('decode', kwargs['sample_start'], len(samples)))
        return stream(samples, **kwargs)
    monkeypatch.setattr(hs, 'on_rx_stream', receive)
    transport.feed(0, np.zeros(9600))
    runtime.step(.01)
    assert order[0] == ('detector', 0, 4800)
    assert order[1] == ('decode', 0, 4800)
    assert context.input_clock == 4800 and transport.frames.qsize() == 1
    context.ledger.publish(UISpan(1, 0, 4, 0))
    transport._reset_rx(14400)
    assert context.input_clock == 14400 and len(context.ledger.spans) == 1
    assert hs._pending_receive is None
    transport.retuned()
    assert context.ledger.tune_epoch == 1 and not context.ledger.spans


def test_connected_ui_delivery_uses_shared_detector_and_kiss_once(monkeypatch):
    from hfmodem.kestrel.vara.vara_arq import VaraState, _TURN_PEER
    runtime, transport, delivered = owner()
    runtime.step(0)
    hs = runtime.modem.station
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'K1ABC'
    hs.state, hs.turn = VaraState.CONNECTED, _TURN_PEER
    message = Broadcast(1, 'K1ABC', 'ALL', b'inside link')
    detector = runtime.receive_context.detector
    original = detector.feed
    def feed(samples, **kwargs):
        assert kwargs.get('ui', True)
        result = original(samples, **kwargs)
        result.append(message)
        return result
    monkeypatch.setattr(detector, 'feed', feed)
    transport.chunks.append((0, np.zeros(4800)))
    runtime.step(0)
    assert delivered == [(message.command, message.encode())]
    assert hs._rx_window.K is None and hs._rx_window.R == b''


def test_station_contact_boundary_retunes_context_and_handshake():
    from hfmodem.station.arbiter import TxArbiter
    from hfmodem.kestrel.rx.observation import LocatedFrame
    from hfmodem.kestrel.rx.ui_spans import CLEAR, CONTESTED
    audio = SimpleNamespace(samples=4800)
    arbiter = TxArbiter(audio, SimpleNamespace())
    transport = StationTransport(SimpleNamespace(station=SimpleNamespace(
        arbiter=arbiter, audio=audio)))
    runtime = AudioHostRuntime(transport, clock=lambda: 100.)
    runtime.step(0)
    hs, context = runtime.modem.station, runtime.receive_context
    context.ledger.publish(UISpan(1, transport.tune_epoch, 4, 0))
    old = hs.capture_context().origin(67072, 270000, 4, 500)
    assert context.ledger.classify(old) == CONTESTED
    hs._pending_receive = {'observations': (LocatedFrame(0, old, b'old'),)}
    transport.feed(4800, np.zeros(4800))
    # The arbiter publishes this boundary for scanner/ALE ownership, without
    # AudioHostRuntime's explicit retune service being involved.
    token = arbiter.acquire_scan_hold('ale', eligible=lambda: True)
    assert token is not None
    audio.samples = 14400
    arbiter.release_scan_hold(token)
    transport.before_step()
    assert hs._tune_epoch == transport.tune_epoch == context.ledger.tune_epoch == 1
    assert context.detector.tune_epoch == 1 and not context.ledger.spans
    assert not hs.origin_is_current(old) and hs._pending_receive is None
    assert transport._skip_before == context.input_clock == 14400
    fresh = hs.capture_context(0).origin(67072, 270000, 4, 500)
    assert context.ledger.classify(fresh) == CLEAR
    assert transport._channel_boundary == transport.contact.receive_boundary
    epoch = hs._decoder_epoch
    transport.before_step()
    assert hs._decoder_epoch == epoch and transport.tune_epoch == 1
    context.ledger.publish(UISpan(2, 1, 4, 0))
    transport._reset_rx(19200)
    assert hs._tune_epoch == transport.tune_epoch == context.ledger.tune_epoch == 1
    assert len(context.ledger.spans) == 1 and context.ledger.classify(fresh) == CONTESTED


@pytest.mark.parametrize('connected', [False, True])
def test_unindexed_transport_keeps_connected_ui_gate(monkeypatch, connected):
    from hfmodem.kestrel.vara.vara_arq import VaraState
    transport = VaraIO()
    modem = VaraHostModem(transport)
    hs = modem.station
    assert modem.receive_context is hs.receive_context is None
    if connected:
        hs.state = VaraState.CONNECTED
    calls = []
    monkeypatch.setattr(hs.cq_detector, 'feed',
        lambda samples, **kwargs: calls.append(kwargs['ui']) or [])
    monkeypatch.setattr(hs, '_feed_shared_front', lambda samples: None)
    monkeypatch.setattr(hs, '_stream_connect_request', lambda samples: True)
    hs.on_rx_stream(np.zeros(1024))
    assert calls == [not connected]
    assert hs._pending_receive is None
