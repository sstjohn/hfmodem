# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P-BUSY-LONG: real receive paths, VARA HF long groups and bounded clearance."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime, KissObserver
from hfmodem.kestrel.host.kiss_tx import KissTxQueue
from hfmodem.kestrel.host.station_runtime import StationTransport
from hfmodem.kestrel.host.ui_channel import UIChannel
from hfmodem.kestrel.broadcast import encode_ui
from hfmodem.kestrel.monitor import FS, SEG_FRAME, Segmenter
from hfmodem.kestrel.rx.broadcast import BODY_SAMPLES, PERIOD, deadline
from hfmodem.kestrel.rx.cq_stream import CQDetector
from hfmodem.kestrel.tx.broadcast import synth_broadcast
from hfmodem.kestrel.tx.cq import synth_cq
from hfmodem.kestrel.ui import UIFrame
from hfmodem.tests import evidence
from hfmodem.kestrel.vara.vara_arq import VaraStationHandshake, VaraState
from hfmodem.tests.evidence import require_input
from .test_kiss_hostless import Clock

FIXTURES = Path(__file__).with_name('fixtures') / 'kiss-ui-1031'
PEER_FILE = FIXTURES / 'peer-packets.json'
CORPUS = Path(os.environ.get('HFMODEM_KISS_CORPUS', evidence.CORPUS / 'kestrel-kiss-1031'))


@pytest.fixture(scope='module', params=['len400', 'len700'])
def capture(request):
    # The VARA HF captures stay out of the distribution.
    captures = json.loads(require_input(PEER_FILE).read_text())['captures']
    row = next(r for r in captures if r['case'] == request.param)
    relative = Path(row['session']) / (row['case'] + '.wav')
    path = next((p for p in (CORPUS / relative, FIXTURES / relative) if p.is_file()), None)
    frame = UIFrame.from_ax25(bytes.fromhex(row['sent']), row['command'])
    if path is None:
        warnings.warn(f'P-BUSY-LONG: {relative} missing; using Kestrel encoder', stacklevel=1)
        audio = synth_broadcast(frame)
    else:
        rate, audio = wavfile.read(path)
        assert rate == FS and audio.ndim == 1
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(float) / (1 << (np.iinfo(audio.dtype).bits - 1))
    audio = audio / max(float(np.max(np.abs(audio))), 1e-12) * .5
    nonzero = np.flatnonzero(np.abs(audio) > .001)
    onset = int(nonzero[0])
    return row, frame, audio, onset


@pytest.fixture(params=['station', 'audio'])
def receiver(request, monkeypatch, tools):
    clock, keyed, delivered = Clock(), [], []
    if request.param == 'station':
        station = SimpleNamespace(audio=SimpleNamespace(samples=0), arbiter=SimpleNamespace(
            receive_boundary=(0, 0), disabled=set(),
            claim_contact=lambda protocol, token, **kw: token or object(),
            release_contact=lambda *args: None))
        def emit(samples, bandwidth, *, on_ptt, responding, marks):
            on_ptt(True)
            for k in range(len(marks[0])):
                marks[1](k)
            on_ptt(False)
            return True
        transport = StationTransport(SimpleNamespace(station=station, emit=emit))
        def feed(at, block):
            station.audio.samples = at + len(block)
            transport.feed(at, block)
    else:
        import kestrel_connect as kc
        class Output:
            time = 0.0
            latency = 0.0
            active = True
            def start(self): pass
            def stop(self): pass
            def close(self): pass
            def write(self, block):
                self.time += len(block) / FS
        class Card:
            def query_devices(self, *args): return {'max_output_channels': 1}
            def InputStream(self, **kw): return SimpleNamespace()
            def OutputStream(self, **kw):
                return Output()
        monkeypatch.setitem(sys.modules, 'sounddevice', Card())
        transport = kc.AudioVaraIO('output', 'input', tx_tail=0)
        def feed(at, block):
            transport.samples = at + len(block)
            transport._buf.append(block)
    runtime = AudioHostRuntime(transport, clock=clock)
    queue = KissTxQueue(clock=clock)
    runtime.bind_kiss(None, queue)
    modem = runtime.modem_factory()
    class Observer(KissObserver):
        def modem_ptt(self, on):
            if on:
                keyed.append(clock.now)
        def modem_broadcast(self, result):
            delivered.append(result)
    modem.start(Observer(runtime))
    runtime.step(0)
    def push(at, block):
        clock.now = (at + len(block)) / FS
        feed(at, block)
        runtime.step(.001)
    return SimpleNamespace(runtime=runtime, transport=transport, modem=modem,
                           queue=queue, push=push, clock=clock, keyed=keyed,
                           delivered=delivered)


@pytest.mark.parametrize('corrupt', [False, True], ids=['intact', 'damaged-middle'])
@pytest.mark.parametrize('background', ['silence', 'steady', 'raised', 'late-burst'])
def test_p_busy_long_waits_for_remote_and_has_bounded_clearance(receiver, capture, corrupt, background):
    row, frame, audio, onset = capture
    rng = np.random.default_rng(1031)
    audio = audio.copy()
    if corrupt:
        start = onset + 67072 + (len(row['packets']) // 2) * PERIOD[row['level']]
        stop = min(len(audio), start + BODY_SAMPLES[row['level']])
        rms = np.sqrt(np.mean(audio[start:stop] ** 2))
        audio[start:stop] = rng.normal(0, rms, stop - start)
    prime = 30 * FS
    low = 0 if background == 'silence' else .0001
    for at in range(0, prime, 4800):
        receiver.push(at, rng.normal(0, low, 4800))
    segmenter = (receiver.transport._segmenter if hasattr(receiver.transport, '_segmenter')
                 else receiver.transport._seg)
    old_history = tuple(segmenter.hist)
    assert len(old_history) == segmenter.hist.maxlen
    end = (prime + len(audio)) / FS
    extent = deadline(prime + onset, row['level'], len(row['packets'])) / FS
    queued = False
    frozen_floor = frozen_history = None
    saw_header = saw_late_burst = saw_closed_before_deadline = False
    for at in range(prime, int((extent + 46) * FS), 4800):
        relative = at - prime
        block = np.zeros(4800)
        take = audio[relative:relative + 4800]
        block[:len(take)] = take
        t = at / FS
        raised = background in ('raised', 'late-burst') and relative > 6 * FS
        if background == 'late-burst' and extent - .4 <= t < extent - .1:
            raised = False
        block += rng.normal(0, .004 if raised else low, len(block))
        if not queued and relative >= 3 * FS:
            receiver.queue.broadcast(frame.command, frame.ax25())
            queued = True
        receiver.push(at, block)
        channel = receiver.transport._ui_channel
        if channel.headers:
            saw_header = True
            if frozen_floor is None:
                frozen_floor = segmenter.nf
                frozen_history = tuple(segmenter.hist)
            assert segmenter.nf == frozen_floor
            assert tuple(segmenter.hist) == frozen_history
        if background == 'late-burst' and extent - .3 <= t < extent:
            saw_closed_before_deadline |= not segmenter.inb
            saw_late_burst |= saw_closed_before_deadline and segmenter.inb
        if receiver.keyed:
            break
    assert queued and saw_header and len(receiver.keyed) == 1
    assert receiver.keyed[0] >= end + 2
    if background in ('silence', 'steady'):
        assert receiver.keyed[0] <= end + .5 + 2 + 1
    else:
        assert receiver.keyed[0] <= extent + 30 + 12 + 2 + 1
    if background == 'late-burst':
        assert saw_late_burst
    assert not receiver.queue
    assert len(receiver.delivered) == (0 if corrupt else 1)


def test_ui_floor_freezes_through_sustained_closes_and_recovers_without_rebasing():
    segmenter = Segmenter()
    channel = UIChannel(segmenter)
    block = np.full(SEG_FRAME, .001)
    for _ in range(segmenter.hist.maxlen):
        segmenter.push(block)
    history = tuple(segmenter.hist)
    floor = segmenter.nf
    sample = segmenter.t0 + segmenter.pos
    channel._header(sample, 4)
    channel._deadline(sample, sample + 15 * FS)
    for _ in range(int(14 * FS / SEG_FRAME)):
        segmenter.push(np.full(SEG_FRAME, .02))
    assert segmenter.nf == floor and tuple(segmenter.hist) == history
    for _ in range(int(2 * FS / SEG_FRAME)):
        segmenter.push(np.full(SEG_FRAME, .02))
    assert not channel.headers and not segmenter.floor_frozen
    assert tuple(segmenter.hist) == history


@pytest.mark.parametrize('level', [1, 2, 4])
def test_ui_clearance_uses_exit_threshold_for_half_a_second(level):
    segmenter = Segmenter()
    channel = UIChannel(segmenter)
    segmenter.hist.extend([.001] * segmenter.hist.maxlen)
    segmenter.nf = .001
    channel._header(0, level)
    threshold = channel.threshold
    for at in range(0, 23 * SEG_FRAME, SEG_FRAME):
        channel._level(at, (.001 + threshold) / 2)
    assert channel.active
    channel._level(23 * SEG_FRAME, (.001 + threshold) / 2)
    assert not channel.headers
    channel._header(FS, level)
    for at in range(FS, 2 * FS, SEG_FRAME):
        channel._level(at, threshold * 1.01)
    assert channel.active and segmenter.floor_frozen


@pytest.mark.parametrize('kind', ['cq', 'ui'])
@pytest.mark.parametrize('deliver', [False, True], ids=['standalone', 'explicit-sink'])
def test_standalone_handshake_receives_unrelated_decodable_frames(monkeypatch, kind, deliver):
    from .test_rx_segmenter import _transport, _record
    transport = _transport(monkeypatch)
    hs = VaraStationHandshake(['W9SSJ'], transport, bw='500', mfsk_only=True)
    frame = UIFrame.from_ax25(encode_ui('K1ABC', 'ALL', b'hello everyone', command=0), 0)
    signal = synth_cq('K1ABC', 500) if kind == 'cq' else synth_broadcast(frame)
    audio = np.concatenate((np.zeros(FS), signal, np.zeros(2 * FS)))
    expected = CQDetector().feed(audio)
    assert len(expected) == 1
    delivered, bursts = [], []
    if deliver:
        transport.cq_sink = delivered.append
    for at in range(0, len(audio), FS // 10):
        _record(transport, audio[at:at + FS // 10])
        burst = transport.next_rx_burst(0, hs=hs)
        if burst is not None:
            bursts.append(burst)
            hs.on_rx_audio(burst)
    assert hs.io is transport and hs.state == VaraState.DISCONNECTED
    assert bursts
    assert len(delivered) == int(deliver)
    if deliver:
        if kind == 'cq':
            assert delivered[0].call == 'K1ABC'
        else:
            assert delivered[0].encode() == expected[0].encode()


@pytest.mark.parametrize('signal', ['out-of-band', 'noise', 'in-band'])
def test_audio_bw500_busy_preserves_band_gate(monkeypatch, signal):
    from .test_rx_segmenter import _transport, _record
    transport = _transport(monkeypatch)
    clock, reports = Clock(), []
    runtime = AudioHostRuntime(transport, clock=clock)
    modem = runtime.modem_factory()
    class Observer(KissObserver):
        def modem_busy(self, active):
            reports.append(active)
    modem.start(Observer(runtime))
    modem.set_bandwidth('500')
    queue = KissTxQueue(clock=clock)
    runtime.bind_kiss(None, queue)
    rng = np.random.default_rng(1043)
    raw_active = qualified = False
    for at in range(0, 6 * FS, FS // 10):
        block = rng.normal(0, .0001, FS // 10)
        if at >= 3 * FS:
            if signal == 'noise':
                block *= 10
            else:
                hz = 2250 if signal == 'out-of-band' else 1500
                block += .1 * np.sin(2 * np.pi * hz * np.arange(at, at + len(block)) / FS)
        clock.now = (at + len(block)) / FS
        _record(transport, block)
        runtime.step(.001)
        if at >= 3 * FS:
            raw_active |= transport._seg.inb
            qualified |= transport._ui_channel.qualified
            assert transport.channel_active('500') == transport._band.active
    assert raw_active and not qualified
    assert queue.busy
    assert (True in reports) == (signal == 'in-band')
