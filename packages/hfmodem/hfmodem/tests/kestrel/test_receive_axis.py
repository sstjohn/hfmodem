# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Indexed runtime input with fake capture and no device owners."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host.station_runtime import StationTransport
from hfmodem.kestrel.monitor import Segmenter, SEG_FRAME
from hfmodem.kestrel.vara import vara_arq as arq
from hfmodem.tests.kestrel import corpora


class Handshake:
    state = arq.VaraState.DISCONNECTED
    def __init__(self):
        self.chunks, self.gaps, self.frontiers = [], [], {}
        self.io = SimpleNamespace(channel_activity=lambda active: None)
        self.bw = '500'
    def on_rx_stream(self, samples, *, sample_start):
        self.chunks.append((sample_start, samples.copy()))
    def rx_gap(self, sample_start): self.gaps.append(sample_start)
    def set_receive_frontier(self, route, sample): self.frontiers[route] = sample
    def _reset_stream(self): pass


class Brackets:
    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.pos = 0
    @property
    def active(self):
        return self.segmenter.inb
    def push(self, samples, **kwargs):
        at, self.pos = self.pos, self.pos + len(samples)
        self.segmenter.push(np.zeros(len(samples)))
        return [(at + 100, samples[100:200]), (at + 500, samples[500:600])]


def _transport():
    station = SimpleNamespace(arbiter=None, audio=SimpleNamespace(samples=0))
    transport = StationTransport(SimpleNamespace(station=station))
    transport.cq_sink = None
    return transport, station


def test_station_queued_bursts_and_attachment_share_capture_axis():
    transport, _ = _transport()
    first, second = Handshake(), Handshake()
    transport.hs = first
    transport._ui_channel = Brackets(transport._segmenter)
    transport.feed(10000, np.ones(4800))
    burst = transport.next_rx_burst(.1, first)
    assert transport.burst_start == 10100 and len(burst) == 100
    transport.hs = second
    burst = transport.next_rx_burst(.1, second)
    assert transport.burst_start == 10500 and len(burst) == 100
    transport.feed(14800, np.ones(4800) * 2)
    transport.next_rx_burst(.1, second)
    assert transport.burst_start == 14900
    assert [at for at, _ in first.chunks] == [10000]
    assert [at for at, _ in second.chunks] == [14800]
    assert transport.input_clock == 19600
    assert second.frontiers['segmented'] <= 10500


def test_station_overlap_and_explicit_gap_do_not_concatenate():
    transport, _ = _transport()
    hs = Handshake()
    transport.hs = hs
    transport.feed(1000, np.ones(4800))
    transport.feed(3400, np.ones(4800) * 2)
    transport.feed(12000, np.ones(4800) * 3)
    transport.next_rx_burst(.02, hs)
    assert [(at, len(x)) for at, x in hs.chunks] == [(1000, 4800), (5800, 2400), (12000, 4800)]
    assert hs.gaps == [12000]
    assert transport._segment_start == 12000
    assert transport.input_clock == 16800


def test_station_tx_discard_and_retune_preserve_clock_and_queue_frontier():
    transport, station = _transport()
    hs = Handshake()
    transport.hs = hs
    transport.feed(1000, np.ones(4800))
    transport.next_rx_burst(.02, hs)
    transport.feed(5800, np.ones(4800))
    transport._reset_rx(20000)
    transport.feed(18000, np.ones(4800))
    transport.next_rx_burst(.02, hs)
    assert hs.gaps[-1] == 20000
    assert [at for at, _ in hs.chunks] == [1000, 20000]
    assert len(hs.chunks[-1][1]) == 2800
    assert transport.input_clock == 22800
    station.audio.samples = 30000
    transport.retuned()
    transport.feed(30000, np.ones(4800))
    transport.next_rx_burst(.02, hs)
    assert hs.chunks[-1][0] == 30000
    assert transport.input_clock == 34800


def test_segmenter_frontier_keeps_idle_preroll_and_partial_frame():
    segmenter = Segmenter()
    segmenter.push(np.zeros(5 * SEG_FRAME + 17))
    assert segmenter.oldest_retained_sample == segmenter.t0 + segmenter.pos - len(segmenter.ring) * SEG_FRAME
    assert segmenter.oldest_retained_sample < 5 * SEG_FRAME
    with pytest.raises(AttributeError):
        segmenter.oldest_retained_sample = 100000
    segmenter.inb, segmenter.start = True, 200
    assert segmenter.oldest_retained_sample == 200


def test_recursive_station_chunks_keep_absolute_starts_and_overlap(monkeypatch):
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    station.state = arq.VaraState.CONNECTED
    chunks = []
    monkeypatch.setattr(station, '_feed_shared_front',
                        lambda x: chunks.append((station._capture_start, len(x))))
    monkeypatch.setattr(station, '_stream_connect_request', lambda x: True)
    station.on_rx_stream(np.zeros(10000), sample_start=50000)
    assert chunks == [(50000 + at, min(1024, 10000 - at)) for at in range(0, 10000, 1024)]
    station.on_rx_stream(np.zeros(4800), sample_start=59000)
    assert chunks[-1] == (60000, 3800)
    assert station._input_end == station._input_clock == 63800
    epoch = station._decoder_epoch
    station.rx_gap(90000)
    station.on_rx_stream(np.zeros(1000), sample_start=90000)
    assert chunks[-1] == (90000, 1000)
    assert station._decoder_epoch > epoch and station._input_clock == 91000


@pytest.fixture
def standalone():
    cli = corpora.harness('kestrel_connect')
    io = cli.AudioVaraIO.__new__(cli.AudioVaraIO)
    io._buf, io._consumed, io._buffer_start = [], 0, 0
    io._rx_end, io.burst_start = None, None
    io._lock = threading.Lock()
    io._seg = cli._BracketSegmenter()
    io._ui_channel = cli.UIChannel(io._seg)
    io._band = SimpleNamespace(push=lambda x: None, restart=lambda: None, busy=False)
    io._bursts, io._rx_generation = [], 0
    io._gap, io._gap_from, io._gap_lead = None, None, 0
    io._rx_inflight = False
    io._report_rx_gap = lambda x: None
    io.channel_active = lambda bw: False
    io.cq_sink = None
    io.hs, io.samples = Handshake(), 0
    return cli, io


def test_standalone_compaction_restart_and_queued_bursts(standalone):
    cli, io = standalone
    io._buffer_start = 50000
    io._buf = [np.ones(4800), np.ones(4800) * 2]
    io._consumed = 4800
    io._compact()
    assert io._buffer_start == 54800 and io._consumed == 0
    io._ui_channel = Brackets(io._seg)
    assert len(io.next_rx_burst(0, io.hs)) == 100
    assert io.burst_start == 54900
    assert io.hs.chunks[0][0] == 54800
    assert io._buffer_start == 59600 and io._consumed == 0
    assert len(io.next_rx_burst(0, io.hs)) == 100
    assert io.burst_start == 55300
    io._restart_receive(80000)
    assert io.hs.gaps == [80000]
    assert io._seg.capture_start == 80000
    io._buffer_start, io._consumed = 80000, 0
    io._buf = [np.zeros(4800)]
    io.next_rx_burst(0, io.hs)
    assert io.hs.chunks[-1][0] == 80000


def test_standalone_retune_and_mute_guard_survive_empty_buffer(standalone):
    _, io = standalone
    io.samples = 100000
    io._buf = [np.ones(4800)]
    io._bursts = [(1, np.ones(100))]
    io.retuned()
    assert io._buffer_start == io._seg.capture_start == 100000
    assert io._rx_generation == 1 and not io._bursts
    io._buf, io._consumed = [np.ones(4800)], 7000
    io._restart_receive(107000)
    io._compact()
    assert io._buffer_start == 104800 and io._consumed == 2200
    io._buf = [np.ones(4800)]
    io.next_rx_burst(0, io.hs)
    assert io.hs.chunks[-1][0] == 107000
    assert len(io.hs.chunks[-1][1]) == 2600


def test_standalone_callback_keeps_absolute_base_after_compaction(standalone):
    _, io = standalone
    io.samples = 90000
    io._paced, io._ahead, io._fit = None, None, []
    io._blk, io.lost, io.xruns = 128, 0, 0
    io.timestamp_limitation = None
    io._run = threading.Event()
    io._run.set()
    io.rec, io.taps, io.offset_hz = None, [], 0
    status = SimpleNamespace(input_overflow=False, input_underflow=False)
    io._on_rx(np.ones((128, 1)), 128, SimpleNamespace(inputBufferAdcTime=10.), status)
    assert io.samples == 90128 and io._buffer_start == 90000
    io._consumed = 128
    io._compact()
    io._on_rx(np.ones((128, 1)), 128, SimpleNamespace(inputBufferAdcTime=10. + 128 / 48000), status)
    assert io.samples == 90256 and io._buffer_start == 90128


def test_bracket_restart_anchor_and_floor_are_independent(standalone):
    _, io = standalone
    segmenter = io._seg
    segmenter.push(np.zeros(10000))
    segmenter.nf = .001
    segmenter.restart(100000)
    assert segmenter.capture_start == 100000
    assert segmenter.oldest_retained_sample == 0 and segmenter.nf == .001
    segmenter.push(np.zeros(3001))
    segmenter.restart()
    assert segmenter.capture_start == 103001
    assert not len(segmenter.buf) and not segmenter.ring


def test_station_frontiers_include_fallback_and_returned_observations():
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    station._input_clock = 100000
    station._lv_start, station._lv_buf = 50000, np.zeros(100)
    context = station.capture_context(10000)
    from hfmodem.kestrel.rx.observation import LocatedFrame
    station._level_read = phy.LevelRead(4, observations=(LocatedFrame(0,
        context.origin(-100, 1000, 4, 500), None),))
    station.set_receive_frontier('segmented', 400)
    frontiers = station.receive_frontiers
    assert frontiers['fallback'] <= 9900
    assert frontiers['fallback'] <= 50000 - phy.RECEIVE_BACKWARD_REACH
    assert frontiers['segmented'] == 400


def test_station_tx_mute_uses_capture_clock_with_fake_emission():
    transport, station = _transport()
    hs = Handshake()
    hs.role = 'initiator'
    transport.hs = hs
    transport.feed(0, np.zeros(4800))
    transport.next_rx_burst(.02, hs)
    station.audio.samples = 4800
    def emit(samples, bw, **kwargs):
        transport.feed(4800, np.ones(10000))
        station.audio.samples = 14800
        return True
    transport.link.emit = emit
    transport.tx_metadata(bandwidth_hz=500)
    transport.tx(np.ones(1000))
    transport.feed(14800, np.zeros(9600))
    transport.next_rx_burst(.02, hs)
    assert transport.input_clock == 24400
    assert hs.chunks[-1][0] == 19600
    assert len(hs.chunks[-1][1]) == 4800
    assert hs.gaps[-1] == 19600
    assert [at for at, _ in hs.chunks] == [0, 19600]


def test_attachment_preserves_tune_epoch_and_advances_decoder_epoch():
    transport, _ = _transport()
    first = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    transport.hs = first
    transport.retuned()
    first.retuned(tune_epoch=transport.tune_epoch)
    tune, decoder = first._tune_epoch, first._decoder_epoch
    second = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='2300')
    transport.hs = second
    assert second._tune_epoch == tune == 1
    assert second._decoder_epoch > decoder
    assert transport.tune_epoch == 1


def test_recursive_split_discards_pre_tx_tail_after_gap(monkeypatch):
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    station.state = arq.VaraState.CONNECTED
    chunks = []
    def feed(samples):
        chunks.append((station._capture_start, len(samples)))
        if len(chunks) == 1:
            station.rx_gap(100000)
    monkeypatch.setattr(station, '_feed_shared_front', feed)
    monkeypatch.setattr(station, '_stream_connect_request', lambda x: True)
    station.on_rx_stream(np.zeros(10000), sample_start=50000)
    assert chunks == [(50000, 1024)]
    assert station._input_clock == 100000
    station.on_rx_stream(np.zeros(1000), sample_start=100000)
    assert chunks[-1] == (100000, 1000)
    assert station._input_clock == 101000


def test_empty_flush_keeps_control_processing_without_advancing_axis(monkeypatch):
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    station._input_clock = 123456
    seen = []
    monkeypatch.setattr(station, '_feed_shared_front', lambda x: seen.append(('front', len(x))))
    monkeypatch.setattr(station, '_stream_connect_request', lambda x: seen.append(('control', len(x))) or True)
    station.on_rx_stream((), flush=True)
    assert seen == [('front', 0), ('control', 0)]
    assert station._input_clock == 123456


def test_joined_capture_reanchors_input_heard_on_another_axis(monkeypatch):
    from hfmodem.kestrel.rx.receive_context import ReceiveContext
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    station.state = arq.VaraState.CONNECTED
    station.io.cq = lambda result: None
    gaps = []
    rx_gap = station.rx_gap
    monkeypatch.setattr(station, 'rx_gap', lambda at: gaps.append(at) or rx_gap(at))
    monkeypatch.setattr(station, '_feed_shared_front', lambda x: None)
    monkeypatch.setattr(station, '_stream_connect_request', lambda x: True)
    station.on_rx_stream(np.zeros(1000))
    station.on_rx_stream(np.zeros(10000))
    assert gaps == [] and station._input_end == 11000
    station.rx_discontinuity()

    transport, _ = _transport()
    context = ReceiveContext()
    transport.bind_receive_context(context)
    transport.hs = station
    assert station.receive_context is context
    assert gaps == [0] and station._input_clock == 0
    station.on_rx_stream((), flush=True)
    station.on_rx_stream(np.zeros(0), sample_start=0)
    assert context.ledger.result_floor <= 0
    block = np.zeros(2048)
    context.feed(block, 0)
    station.on_rx_stream(block, sample_start=0)
    assert gaps == [0]
    assert station._input_end == station._input_clock == context.input_clock == 2048
    station.on_rx_stream(block, sample_start=2048)
    assert gaps == [0] and station._input_end == 4096
    station.on_rx_stream((), flush=True)
    assert station._input_end == station._input_clock == 4096


def test_receive_boundaries_void_raw_control_audio_but_keep_owed_waits():
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='2300')
    held = object()
    for boundary in (lambda: station.rx_gap(station._input_clock + 4800),
                     station.rx_discontinuity):
        station._grant_buf = station._ans_buf = station._connect_ask_buf = np.ones(4800)
        station._ans_held.append(np.ones(4800))
        station._ans_held_n = 4800
        station._held_answer, station._ans_wake = held, True
        station._responder_nak_wait_samples = 960
        boundary()
        assert not len(station._grant_buf)
        assert not len(station._ans_buf) and not station._ans_held and not station._ans_held_n
        assert not len(station._connect_ask_buf)
        assert station._held_answer is held and station._ans_wake
        assert station._responder_nak_wait_samples == 960
