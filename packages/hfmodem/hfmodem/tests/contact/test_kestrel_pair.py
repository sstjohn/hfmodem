# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Two current chat stacks through rendered audio, without radio hardware.

The host sockets are deterministic in-memory byte streams; neither application
messages nor modem replies are invented. Each endpoint has a monotonic virtual
clock, including blocking TX time. Audio is serialized with a quiet lead/tail,
fed in 1024-sample blocks, then bracketed as in the live receive path. Production
AudioHostRuntime supplies the retry cadence. This checks protocol interoperability,
not RF collisions, the live audio segmenter, hardware timing, or TCP threads.
"""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.contact import varac as v
from hfmodem.contact.bbs import HostCatalog
from hfmodem.kestrel.host.server import HostSession
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.vara import vara_arq as arq


@pytest.mark.parametrize('snr,wire', [(125., '<R+99>'), (-125., '<R-99>'), (12.4, '<R+12>')])
def test_opening_report_fits_observed_two_digit_field(snr, wire):
    from hfmodem.tests.contact.test_varac import Desk

    desk = Desk()
    desk.hear(f'SN {snr}', 'CONNECTED W9SSJ W1AW 500')
    desk.flow.opening()
    assert desk.texts[-1] == f'{len("de W9SSJ " + wire)} de W9SSJ {wire}'
    assert desk.host.client.snr == snr


class Stream:
    def __init__(self, send=lambda data: None, chunk=65536):
        self.sendall = send
        self.pending = bytearray()
        self.chunk = chunk

    def recv(self, size):
        n = min(size, self.chunk, len(self.pending))
        result = bytes(self.pending[:n])
        del self.pending[:n]
        return result

    def close(self):
        pass


class Channel:
    def __init__(self, snr, chunk, lose_request=False):
        self.times = [1000., 1000.]
        self.active = 0
        self.snr, self.chunk = snr, chunk
        self.rng = np.random.default_rng(1025)
        self.air = deque()
        self.peers = []
        self.trace = deque(maxlen=300)
        self.lose_request = lose_request
        self.dropped = 0

    def clock(self):
        return self.times[self.active]

    @property
    def now(self):
        return max(self.times)

    def step(self):
        for index, peer in enumerate(self.peers):
            self.active = index
            peer.session.tick(0)
            while peer.client.command_socket.pending or peer.client.data_socket.pending:
                peer.session.tick(0)
            peer.runtime.transport.receiving = bool(self.air)
            peer.runtime.step(0)
        if self.air:
            sender, start, samples = self.air.popleft()
            if self.lose_request and sender == 0 and not self.dropped:
                assert self.peers[sender].modem.station.connect_request_pending
                self.dropped += 1
                return
            self.active = 1 - sender
            self.times[self.active] = max(self.clock(), start)
            receiver = self.peers[1 - sender]
            receiver.runtime.transport.receiving = False
            power = np.mean(samples ** 2)
            samples = np.pad(samples, (24000, 12000))
            if self.snr is not None:
                noise_power = power / 10 ** (self.snr / 10)
                samples += self.rng.normal(0, np.sqrt(noise_power), len(samples))
            for at in range(0, len(samples), 1024):
                block = samples[at:at + 1024]
                self.times[self.active] += len(block) / 48000
                receiver.modem.station.on_rx_stream(block)
            receiver.modem.station.on_rx_audio(samples)
            receiver.modem.station.on_rx_stream(np.empty(0), flush=True)
        else:
            now = self.now + .1
            for index, peer in enumerate(self.peers):
                self.active = index
                self.times[index] = now
                station = peer.modem.station
                station.on_rx_stream(np.zeros(4800))

    def until(self, predicate, label, seconds=240):
        deadline = self.now + seconds
        for _ in range(10000):
            if predicate():
                return
            if self.now >= deadline:
                break
            self.step()
        state = [(p.call, p.modem.station.state, p.modem.station.turn,
                  p.modem.station.pending_bytes, p.said[-3:]) for p in self.peers]
        pytest.fail(f'{label} timed out at {self.now:.1f}: {state}\n' + '\n'.join(self.trace))


class Transport(arq.VaraIO):
    receiving = False

    def __init__(self, channel, index):
        self.channel, self.index = channel, index

    def tx(self, samples):
        self.channel.air.append((self.index, self.channel.clock(), np.asarray(samples, float).copy()))
        self.channel.times[self.index] += len(samples) / 48000

    def log(self, text):
        self.channel.trace.append(f'{self.index}: {text}')

    def next_rx_burst(self, timeout, hs):
        return None


class Peer:
    def __init__(self, channel, call, bw, directory, catalog):
        self.call = call
        self.channel, self.index = channel, len(channel.peers)
        channel.active = self.index
        self.runtime = AudioHostRuntime(Transport(channel, len(channel.peers)), clock=channel.clock)
        self.modem = self.runtime.modem_factory()
        command, data = Stream(chunk=channel.chunk), Stream(chunk=channel.chunk)
        host = HostSession(Stream(command.pending.extend), Stream(data.pending.extend), self.modem)
        self.modem.start(host)

        def dispatch(blob):
            for line in blob.decode('ascii').split('\r'):
                if line:
                    host._dispatch(line)
            self.modem.poll()

        command.sendall = dispatch
        data.sendall = self.modem.transmit

        def ready(streams, timeout):
            waiting = [s for s in streams if s.pending]
            if not waiting:
                channel.times[self.index] += timeout
            return waiting

        self.client = v.Client(command, data, clock=channel.clock,
                               ready=ready)
        self.client.initialize(call, bw)
        self.flow = v.Flow(self.client, lambda hz: None)
        self.mailbox = v.Mailbox(directory)
        self.said, self.events, self.notices = [], [], []
        self.session = v.Session(self.client, self.flow, say=self.said.append,
                                 mailbox=self.mailbox, bandwidth=bw, bbs_catalog=catalog,
                                 emit=lambda kind, **detail: self.notices.append((kind, detail)),
                                 line_state=lambda: False)

        def event(item):
            self.events.append(item)
            self.session.handle(item)

        self.client.event = event
        self.session.monitor()
        channel.peers.append(self)

    def command(self, text):
        self.channel.active = self.index
        return self.session.command(text)


@pytest.mark.parametrize('bw', [500, 2300, 2750])
@pytest.mark.parametrize('snr,chunk,lose_request', [
    pytest.param(None, 65536, False, id='clean'),
    pytest.param(12., 1, False, id='noise-fragmented'),
    pytest.param(20., 37, True, id='lost-request'),
])
def test_two_chats_bbs_vmail_and_disconnect(tmp_path, monkeypatch, bw, snr, chunk, lose_request):
    channel = Channel(snr, chunk, lose_request)
    monkeypatch.setattr(arq, 'time', SimpleNamespace(monotonic=channel.clock))
    public = tmp_path / 'public'
    public.mkdir()
    (public / 'hello.txt').write_text('A file from the other Kestrel.\n')
    catalog = HostCatalog()
    catalog.enable(public)
    try:
        a = Peer(channel, 'W9SSJ', bw, tmp_path / 'a', HostCatalog())
        b = Peer(channel, 'W1AW', bw, tmp_path / 'b', catalog)
        b.command('/cq 11')
        channel.until(lambda: b.flow.state == v.State.WAITING and not channel.air, 'CQ slot wait')
        a.command('/call W1AW 11')
        channel.until(lambda: all(p.client.connected and p.session.steps is None
                                  for p in (a, b)), 'connect')
        assert a.flow.dial == b.flow.dial == v.slot_frequency(a.flow.band, 11)
        for sender, receiver, text in [(a, b, 'Hello from W9SSJ'), (b, a, 'Hello from W1AW')]:
            sender.command(text)
            channel.until(lambda: any(text in line for line in receiver.said), f'chat delivery from {sender.call}')
            assert sum(text in line for line in receiver.said) == 1
        a.command('/bbs list')
        channel.until(lambda: any('hello.txt' in line for line in a.said), 'BBS list')
        assert [entry.name for entry in a.session.files.entries] == ['hello.txt']
        for sender, receiver in [(a, b), (b, a)]:
            sender.command('/vmail send Pair test')
            sender.command('A message through two real modem state machines.')
            sender.command('.')
            channel.until(lambda: any(e['direction'] == 'in' for e in receiver.mailbox.entries())
                          and not sender.session._vmail_sent, 'VMail and acknowledgement')
            incoming = [receiver.mailbox.read(i) for i, entry in enumerate(receiver.mailbox.entries(), 1)
                        if entry['direction'] == 'in']
            assert len(incoming) == 1
            assert (incoming[0]['from'], incoming[0]['to'], incoming[0]['subject'], incoming[0]['body']) == (
                sender.call, receiver.call, 'Pair test', 'A message through two real modem state machines.')
        (b if chunk == 1 else a).command('/bye')
        channel.until(lambda: all(not p.client.connected and p.session.contact is None
                                  and p.session.steps is None for p in (a, b)), 'disconnect')
        assert all(p.modem.station.pending_bytes == 0 for p in (a, b))
        assert all(p.modem.discarded_total == 0 for p in (a, b))
        assert all(p.modem.accepted_total == p.modem.confirmed_total for p in (a, b))
        assert not any(e.kind == 'command' and e.raw == 'ABORT' for p in (a, b) for e in p.events), '\n'.join(
            [f'{p.call}: {line}' for p in (a, b) for line in p.said[-12:]] + list(channel.trace))
        assert not [(p.call, detail) for p in (a, b) for kind, detail in p.notices if kind == 'error']
        assert not any(e.kind == 'framing_error' for p in (a, b) for e in p.events)
        assert channel.dropped == int(lose_request)
    finally:
        catalog.close()
