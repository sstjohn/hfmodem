# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real VARA station owner with synthetic input and a recording arbiter."""
import threading
import socket
import time
from types import SimpleNamespace

import numpy as np

from hfmodem.kestrel.host.station_runtime import StationTransport
from hfmodem.kestrel.vara import vara_arq as A, vara_mfsk as M, vara_frames as F
from hfmodem.station.arbiter import Refused, TxArbiter
from hfmodem.station.link import KestrelLink
from hfmodem.tests.kestrel.test_vara_host_modem import Observer


class Notices(Observer):
    def __init__(self):
        super().__init__()
        self.ptt = []
        self.pending = 0
        self.cancels = 0
    def modem_ptt(self, on): self.ptt.append(on)
    def modem_pending(self, cancel=False):
        if cancel:
            self.cancels += 1
        else:
            self.pending += 1


def wait_for(predicate):
    until = time.monotonic()+10
    while not predicate():
        assert time.monotonic() < until
        time.sleep(.01)


def station(*, refuse=False):
    sent = []
    st = SimpleNamespace(audio=SimpleNamespace(samples=0),
        rig=SimpleNamespace(dial=lambda: 7100000),
        cfg=SimpleNamespace(rig=SimpleNamespace(power_w=10, ptt=SimpleNamespace(settle_s=0))))
    def submit(req):
        if refuse:
            raise Refused('test refusal')
        sent.append((req, threading.current_thread().name))
        if req.on_ptt: req.on_ptt(True)
        st.audio.samples += len(req.audio)
        if req.on_ptt: req.on_ptt(False)
        return st.audio.samples
    st.arbiter = TxArbiter(st.audio, st.rig)
    st.arbiter.submit = submit
    return st, sent


def open_modem(link):
    link.start()
    modem = link.runtime.modem_factory()
    observer = Notices()
    modem.start(observer)
    modem.set_mycall(['W9SSJ'])
    modem.set_bandwidth('500')
    modem.set_compression('OFF')
    return modem, observer


def test_host_commands_reach_real_owner_and_only_station_arbiter():
    st, sent = station()
    link = KestrelLink(st)
    modem, observer = open_modem(link)
    try:
        modem.connect('W9SSJ', 'K7ABC')
        wait_for(lambda: sent)
        req, thread = sent[0]
        assert thread == 'station-vara-owner'
        assert req.protocol == 'vara' and req.rate == 48000
        assert np.issubdtype(req.audio.dtype, np.floating)
        assert req.emission.audio_lo_hz == 1080
        assert req.emission.audio_hi_hz == 1870
        assert observer.ptt == [True, False]
        assert isinstance(modem.station, A.VaraStationHandshake)
        assert link.active and not modem.connected
        assert link.runtime.failure is None
        modem.stop()
        wait_for(lambda: modem.observer is None)
        assert not link.active
    finally:
        link.stop()
    assert not link.runtime._thread.is_alive()


def test_refusal_has_no_ptt_or_delivery_credit_and_stop_retains_originals():
    st, _ = station(refuse=True)
    link = KestrelLink(st)
    modem, observer = open_modem(link)
    try:
        modem.transmit(b'original before connect')
        modem.connect('W9SSJ', 'K7ABC')
        wait_for(lambda: link.refusals)
        assert not observer.ptt and not modem.connected
        assert modem.confirmed_total == 0
        modem.stop()
        wait_for(lambda: modem.observer is None)
        assert modem.discarded_total == modem.accepted_total == 23
        assert modem.retired[-1]['unconfirmed'] == b'original before connect'
    finally:
        link.stop()


def test_contact_boundary_drops_old_indexed_capture():
    st, _ = station()
    link = KestrelLink(st)
    transport = link.runtime.transport
    old, received = [], []
    hs = SimpleNamespace(on_rx_stream=old.append, _reset_stream=lambda: None)
    transport.hs = hs
    transport.feed(0, np.ones(4800, np.float32)*.001)
    st.audio.samples = 4800
    next_hs = SimpleNamespace(on_rx_stream=received.append, _reset_stream=lambda: None)
    transport.hs = next_hs
    transport.feed(4800, np.ones(4800, np.float32)*.002)
    transport.next_rx_burst(.01, hs=next_hs)
    assert not old and len(received) == 1
    assert np.allclose(received[0], .002)
    link.stop()


def test_receive_energy_detector_publishes_busy_edges_without_a_contact():
    st, sent = station()
    link = KestrelLink(st)
    modem, observer = open_modem(link)
    edges = []
    observer.modem_busy = edges.append
    try:
        wait_for(lambda: link.runtime.transport.hs is modem.station)
        # Feed real indexed PCM through the production owner and segmenter.
        noise = np.random.default_rng(1020).normal(0, .001, 48000)
        tone = .1*np.sin(2*np.pi*1500*np.arange(24000)/48000)
        audio = np.concatenate([noise, tone, noise])
        # BUSY holds and clears on the clock, as stock's does, so the PCM arrives in time.
        for at in range(0, len(audio), 2400):
            link.runtime.transport.feed(at, audio[at:at + 2400])
            time.sleep(.05)
        wait_for(lambda: True in edges and edges[-1] is False)
        assert edges.count(True) == 1
        assert not sent and not modem.connected
        assert modem.confirmed_total == 0 and link.runtime.failure is None
    finally:
        link.stop()


def test_station_flush_retires_old_brackets_without_failing_contact():
    st, _ = station()
    link = KestrelLink(st)
    transport = link.runtime.transport
    received, reset = [], []
    hs = SimpleNamespace(on_rx_stream=received.append, _reset_stream=lambda: reset.append(True))
    transport.hs = hs
    transport.feed(0, np.ones(4800, np.float32)*.001)
    transport.flushed(4800)
    transport.feed(4800, np.ones(4800, np.float32)*.002)
    transport.next_rx_burst(.01, hs=hs)
    assert reset == [True]
    assert len(received) == 1 and np.allclose(received[0], .002)
    assert not transport.fault.is_set()
    link.stop()


def test_synthetic_incoming_call_is_decoded_and_answered_by_same_owner():
    st, sent = station()
    link = KestrelLink(st)
    modem, observer = open_modem(link)
    try:
        modem.set_listen(True)
        wait_for(lambda: modem.station.state == A.VaraState.LISTENING)
        audio = np.concatenate([np.zeros(48000), M.synth_burst('W9SSJ', F.for_bw(F.CR, '500')),
                                np.zeros(48000)])
        link.runtime.transport.feed(0, audio)
        wait_for(lambda: sent)
        assert observer.pending > 0
        assert modem.station.role == 'responder'
        assert sent[0][1] == 'station-vara-owner'
        assert link.runtime.failure is None
    finally:
        link.stop()


def test_receive_fault_retires_originals_and_reports_the_failed_attempt():
    st, sent = station()
    link = KestrelLink(st)
    modem, observer = open_modem(link)
    try:
        modem.transmit(b'not confirmed')
        modem.connect('W9SSJ', 'K7ABC')
        wait_for(lambda: sent)
        link.runtime.transport.fault.set()
        wait_for(lambda: modem.discarded_total == 13)
        assert modem.confirmed_total == 0
        assert modem.retired[-1]['unconfirmed'] == b'not confirmed'
        assert observer.warnings
        assert observer.closes == 1
        assert not link.active
    finally:
        link.stop()


def test_station_host_factory_is_the_link_owner_factory():
    from hfmodem.station.hosts import _build_one
    st, _ = station()
    link = KestrelLink(st)
    st.lanes = {'vara': link}
    served = _build_one('vara', SimpleNamespace(host='vara', cmd_port=0, data_port=0), st)
    assert served.server.modem_factory == link.runtime.modem_factory
    assert served.keys
    modem = served.server.modem_factory()
    modem.transmit(b'unstarted owner')
    link.stop()
    assert modem.discarded_total == 15 and modem.confirmed_total == 0


def test_station_capture_queue_overflow_fails_explicitly():
    st, _ = station()
    link = KestrelLink(st)
    transport = StationTransport(link)
    transport.feed(0, np.zeros(4800*129, np.float32))
    assert transport.frames.qsize() == 128
    assert transport.fault.is_set()
    link.stop()


def test_real_host_sockets_use_station_owner_and_retire_binary_originals():
    from hfmodem.kestrel.host.server import VaraServer
    st, sent = station()
    link = KestrelLink(st)
    link.start()
    server = VaraServer(cmd_port=0, data_port=0, modem_factory=link.runtime.modem_factory)
    server.start_background()
    cmd = data = None
    try:
        cmd = socket.create_connection(('127.0.0.1', server.cmd_port), timeout=3)
        data = socket.create_connection(('127.0.0.1', server.data_port), timeout=3)
        for line in (b'MYCALL W9SSJ', b'BW500', b'COMPRESSION FILES', b'CONNECT W9SSJ K7ABC'):
            cmd.sendall(line+b'\r')
            received = bytearray()
            while b'OK\r' not in received:
                received.extend(cmd.recv(4096))
        wait_for(lambda: sent)
        modem = link.runtime.owner.modem
        original = bytes(range(256))*2
        data.sendall(original)
        wait_for(lambda: modem.station.payload_ledger.accepted == len(original))
        assert modem.station.payload_ledger.unconfirmed == original
        assert modem.confirmed_total == 0
        cmd.close()
        wait_for(lambda: modem.observer is None)
        assert modem.retired[-1]['unconfirmed'] == original
        assert modem.discarded_total == len(original)
        assert data.recv(1) == b''
    finally:
        for conn in (cmd, data):
            if conn is not None: conn.close()
        server.stop()
        link.stop()
