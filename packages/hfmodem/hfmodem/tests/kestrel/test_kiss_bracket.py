# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P-BRACKET through both production transports and the console's real wait."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from hfmodem.ax25 import kiss
from hfmodem.core.audio import BurstInvalidated, ReplayAudio, play_drained
from hfmodem.kestrel.host.kiss_tx import KissTxDisposition as Outcome, KissTxQueue
from hfmodem.kestrel.host.server import HostSession
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.rx.broadcast import PERIOD
from hfmodem.kestrel.tx.ui import packetize
from hfmodem.kestrel.ui import UIAddress, UIFrame
from hfmodem.station.arbiter import Refused
from hfmodem.station.link import KestrelLink
from hfmodem.tests.contact.test_varac import Host
from hfmodem.tests.core.test_arbiter import arb as arb, rig as rig
from . import test_trace_timing
from .test_cq import MockSocket

audio_transport = test_trace_timing.transport


@pytest.fixture(params=['station', 'audio'])
def playback(request, monkeypatch):
    lines, clock = [], SimpleNamespace(now=0.0)
    failure = SimpleNamespace(kind=None)
    if request.param == 'station':
        arbiter = request.getfixturevalue('arb')
        arbiter.audio = ReplayAudio(np.zeros(48000 * 30, np.float32))
        st = SimpleNamespace(audio=arbiter.audio, arbiter=arbiter, rig=arbiter.rig,
            cfg=SimpleNamespace(declared_power_w=10,
                                rig=SimpleNamespace(ptt=SimpleNamespace(settle_s=0))))
        link = KestrelLink(st)
        transport = link.runtime.transport
        stamp = lambda: arbiter.audio.samples / 48000
        report = transport._report_ptt
        def ptt(on):
            if on or failure.kind != 'unkey':
                report(on)
        monkeypatch.setattr(transport, '_report_ptt', ptt)
        wait = arbiter._wait_dac
        validate = arbiter._validate
        def checked(req, **kw):
            if failure.kind == 'refused':
                raise Refused('before key')
            return validate(req, **kw)
        def advance(frame, **kw):
            if failure.kind == 'partial' and frame >= 67072 + PERIOD[4]:
                raise BurstInvalidated('playback failed')
            return wait(frame, **kw)
        monkeypatch.setattr(arbiter, '_validate', checked)
        monkeypatch.setattr(arbiter, '_wait_dac', advance)
    else:
        kc, audio_clock, make = request.getfixturevalue('audio_transport')
        transport, _, _ = make()
        transport.tx_tail = 0
        transport.rig.keyed = False
        transport.rig.line_held = True
        transport.rig.line_state = lambda: False
        stamp = lambda: audio_clock.now
        key = transport.rig.key
        def keyed(on):
            if failure.kind == 'refused' and on:
                return False
            if failure.kind == 'unkey' and not on:
                raise RuntimeError('unkey failed')
            return key(on)
        transport.rig.key = keyed
        played = SimpleNamespace(samples=0, queued=0, started=audio_clock.now)
        class Output:
            latency = .1
            active = True
            def __init__(self, **kw): pass
            def start(self):
                played.started = audio_clock.now
            @property
            def time(self):
                played.samples = max(0, round((audio_clock.now - played.started - self.latency) * 48000))
                if failure.kind == 'partial' and played.samples >= 67072 + PERIOD[4] - 1:
                    raise OSError('playback failed')
                return audio_clock.now
            def stop(self):
                audio_clock.now = max(audio_clock.now,
                                      played.started + self.latency + played.queued / 48000)
            def close(self): pass
            def write(self, block):
                played.queued += len(block)
                audio_clock.now = max(audio_clock.now, played.started + played.queued / 48000)
                transport.samples += len(block)
                transport._buf.append(np.zeros(len(block)))
        monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(OutputStream=Output))
        monkeypatch.setattr(kc, 'play_drained', play_drained)
        monkeypatch.setattr(kc, '_tx_end_in_capture', lambda audio, expected: expected)
    started = []
    tx = transport.tx
    def transmit(samples, *, marks):
        offsets, callback = marks
        def mark(k):
            started.append((k, stamp()))
            callback(k)
        return tx(samples, marks=(offsets, mark))
    monkeypatch.setattr(transport, 'tx', transmit)
    modem = VaraHostModem(transport, clock=lambda: clock.now)
    session = HostSession(MockSocket(), MockSocket(), modem)
    session._send_cmd = lambda line: lines.append((stamp(), line))
    modem.start(session)
    modem.poll()
    transport.hs = modem.station
    queue = KissTxQueue(clock=lambda: clock.now)
    modem.kiss_tx_queue = queue
    lines.clear()
    yield SimpleNamespace(modem=modem, session=session, queue=queue, lines=lines,
                          failure=failure, transport=transport, stamp=stamp, marks=started,
                          clock=clock)


def message(info=b'x' * 70):
    return UIFrame(1, (UIAddress('ALL'), UIAddress('W9SSJ', flags=0x61)), 3, 0xF0, info)


@pytest.mark.parametrize('info', [b'hello', b'hello everyone', b'x' * 70])
def test_p_bracket_reports_each_body_at_playback_marks(playback, info):
    frame = message(info)
    level, packets = packetize(frame)
    assert playback.queue.broadcast(frame.command, frame.ax25())
    assert playback.modem.service_kiss_queue() is None
    playback.clock.now += playback.queue.holdoff
    assert playback.modem.service_kiss_queue() is Outcome.SENT
    lines = [line for _, line in playback.lines]
    assert lines[:3] == ['PENDING', 'PTT ON', 'BUSY ON']
    assert lines[-3:] == ['PTT OFF', 'CANCELPENDING', 'BUSY OFF']
    rates = [(at, line) for at, line in playback.lines if line.startswith('BITRATE')]
    bps = {1: 18, 2: 41, 4: 88}[level]
    assert [line for _, line in rates] == [f'BITRATE ({level})  {bps} bps TX'] * len(packets)
    assert [k for k, _ in playback.marks] == list(range(len(packets) + 1))
    onset = playback.marks[0][1]
    for k, (at, _) in enumerate(rates):
        assert at - onset == pytest.approx((67072 + k * PERIOD[level]) / 48000, abs=.003)
    assert rates[0][0] - playback.lines[1][0] >= 67072 / 48000


@pytest.mark.parametrize('kind', ['refused', 'partial', 'unkey'])
def test_p_bracket_failure_shapes(playback, monkeypatch, kind):
    frame = message()
    assert playback.queue.broadcast(frame.command, frame.ax25())
    assert playback.modem.service_kiss_queue() is None
    playback.clock.now += playback.queue.holdoff
    playback.failure.kind = kind
    if kind == 'unkey':
        monkeypatch.setattr(playback.transport, 'unkey_confirmed', lambda: False)
    if kind == 'refused':
        assert playback.modem.service_kiss_queue() is Outcome.REFUSED_BEFORE_KEY
        assert not playback.lines and len(playback.queue) == 1
        return
    with pytest.raises((OSError, RuntimeError, BurstInvalidated)):
        playback.modem.service_kiss_queue()
    lines = [line for _, line in playback.lines]
    assert lines[:3] == ['PENDING', 'PTT ON', 'BUSY ON']
    assert lines[-3:] == ['PTT OFF', 'CANCELPENDING', 'BUSY OFF']
    assert sum(line.startswith('BITRATE') for line in lines) == (1 if kind == 'partial' else 2)
    assert not playback.queue


def test_real_console_queued_kiss_steps_completes_from_kestrel_bracket(playback):
    host = Host(kiss=True)
    frame = message()
    def write(wire):
        for _, command, raw in kiss.Parser().feed(wire):
            assert playback.queue.broadcast(command, raw)
    host.kiss.respond = write
    playback.session._send_cmd = host.lines
    steps = host.client.queued_kiss_steps(frame)
    next(steps)
    assert playback.modem.service_kiss_queue() is None
    playback.clock.now += playback.queue.holdoff
    assert playback.modem.service_kiss_queue() is Outcome.SENT
    while host.command.incoming:
        host.client.poll()
    with pytest.raises(StopIteration):
        next(steps)
    assert len(host.kiss.sent) == 1 and not playback.queue
