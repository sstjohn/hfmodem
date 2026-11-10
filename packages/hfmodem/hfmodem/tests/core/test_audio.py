# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The card, driven by a scripted stream so none of this needs hardware.

What `FakeStream` can do that a real card cannot: hand out chosen ADC/DAC
timestamps, inject an overflow or an underrun exactly where a test wants one, and
run 22208 callbacks in milliseconds. What it cannot do is tell you what a CoreAudio
device really reports, which is why the offset, the clock ppm and the T/R deaf
window stay `hardware`.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hfmodem.core import audio
from hfmodem.core.audio import (
    AudioError, Block, BurstInvalidated, CycleLane, RollingLane, StationAudio,
    ReplayAudio, drop_output, play_drained,
)

REPO = Path(__file__).resolve().parents[5]


def test_playback_callback_submission_includes_final_padding(monkeypatch):
    import sys
    class Stop(Exception): pass
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(CallbackStop=Stop))
    submitted, complete = [], []
    callback = audio.playback_callback(np.arange(6, dtype=np.float32),
        submitted=lambda at, pcm: submitted.append((at, pcm.copy())),
        complete=lambda: complete.append(True))
    out = np.empty((4, 1), np.float32)
    callback(out, 4, None, False)
    assert not complete
    with pytest.raises(Stop):
        callback(out, 4, None, False)
    assert complete == [True]
    assert [at for at, _ in submitted] == [0, 4]
    np.testing.assert_array_equal(np.concatenate([pcm for _, pcm in submitted]),
                                  [0, 1, 2, 3, 4, 5, 0, 0])


def test_playback_callback_failed_buffer_has_no_submission(monkeypatch):
    import sys
    class Abort(Exception): pass
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(CallbackAbort=Abort))
    submitted, failed = [], []
    callback = audio.playback_callback(np.ones(20, np.float32),
        submitted=lambda *a: submitted.append(a), failed=failed.append)
    out = np.empty((4, 1), np.float32)
    with pytest.raises(Abort):
        callback(out, 4, None, 'synthetic underflow')
    assert failed == ['synthetic underflow'] and not submitted
    assert not out.any()


class _Status:
    def __init__(self, *, overflow=False, underflow=False, priming=False):
        self.input_overflow = overflow
        self.output_underflow = underflow
        self.priming_output = priming

    def __bool__(self):
        return self.input_overflow or self.output_underflow


class _Times:
    def __init__(self, adc, dac):
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = dac


class FakeStream:
    """Enough of sounddevice.Stream to drive the callback deterministically."""

    def __init__(self, cb, *, blocksize=128, lat=1152, latency=(0.01, 0.01)):
        self.cb, self.blocksize, self.lat, self.latency = cb, blocksize, lat, latency
        self.t = 0.0
        self.out_history: list[np.ndarray] = []

    def start(self):
        return None

    def stop(self):
        return None

    def close(self):
        return None

    def run(self, blocks=1, *, data=None, status=None, lat=None):
        for i in range(blocks):
            n = self.blocksize
            indata = np.zeros((n, 1), np.float32)
            if data is not None:
                chunk = data[i * n:(i + 1) * n]
                indata[:len(chunk), 0] = chunk
            outdata = np.zeros((n, 1), np.float32)
            offset = (self.lat if lat is None else lat) / audio.CARD_RATE_HZ
            # `status if None` rather than `status or ...`: a priming-only status
            # is falsy, so `or` would silently replace the state under test.
            self.cb(indata, outdata, n, _Times(self.t, self.t + offset),
                    _Status() if status is None else status)
            self.out_history.append(outdata[:, 0].copy())
            self.t += n / audio.CARD_RATE_HZ


def station(monkeypatch, **kw) -> tuple[StationAudio, FakeStream]:
    made = {}

    class FakeSd:
        @staticmethod
        def Stream(callback=None, blocksize=128, **_):
            made["s"] = FakeStream(callback, blocksize=blocksize, **kw)
            return made["s"]

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSd)
    sa = StationAudio(input_device="fake-in", output_device="fake-out")
    return sa, made


# --- the ADC/DAC offset is a median, not one reading ---------------------------

def _fix23_stamps(*, varying=True, ages=None, packets=24):
    """The probe's 480/128 FIFO model, with raw median 128 and spread 480.

    From working/windows-phase3/build/wasapi-probe/test_fix23_model.py:
    capture padding is the next packet length, not capture age. The true
    2048-frame offset and packet ages are unavailable to StationAudio.
    """
    fs, user, host, physical_lat = audio.CARD_RATE_HZ, 128, 480, 2048
    if ages is None:
        ages = (1200, 1680, 1560, 1440, 1320) if varying else (1440,)
    n0 = 0
    for packet in range(packets):
        h = packet * host
        retained = h - n0
        age = ages[packet % len(ages)]
        current = 100 + (h + age) / fs
        output_padding = physical_lat - age - user
        step = 0
        while n0 + user <= h + host:
            yield SimpleNamespace(
                currentTime=current,
                inputBufferAdcTime=current + (host - retained + step) / fs,
                outputBufferDacTime=current + (output_padding + user - retained + step) / fs)
            n0 += user
            step += user


def _fix23_station(monkeypatch, *, host="Windows WASAPI", varying=True,
                   callbacks=audio._LAT_MEDIAN_N, latency=(.0123, .0246), stamps=None):
    """Open-path mock shared by the audio and preflight pinning tests."""
    stamps = (list(_fix23_stamps(varying=varying))[:callbacks]
              if stamps is None else list(stamps))

    class ScriptedStream(FakeStream):
        device = (17, 18)
        active = True

        def start(self):
            for t in stamps:
                data = np.zeros((self.blocksize, 1), np.float32)
                out = np.ones_like(data)
                self.cb(data, out, self.blocksize, t, _Status())
                self.out_history.append(out[:, 0].copy())
            self.t = stamps[-1].inputBufferAdcTime + self.blocksize / audio.CARD_RATE_HZ

    made = []

    def stream(callback, blocksize, **kw):
        s = ScriptedStream(callback, blocksize=blocksize, latency=latency)
        made.append(s)
        return s

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", SimpleNamespace(
        Stream=stream, query_devices=lambda device: {"hostapi": 3},
        query_hostapis=lambda api: {"name": host}))
    return StationAudio(input_device="fake-in", output_device="fake-out", blocksize=128), made


def test_wasapi_rig_spread_opens_with_nominal_reported_latency(monkeypatch, caplog):
    raw = [round((t.outputBufferDacTime - t.inputBufferAdcTime) * audio.CARD_RATE_HZ)
           for t in list(_fix23_stamps())[:audio._LAT_MEDIAN_N]]
    assert max(raw) - min(raw) == 480
    assert np.median(raw) == 128
    sa, made = _fix23_station(monkeypatch)
    sa.open()
    try:
        assert sa.failure is None
        assert sa.lat == round(sum(made[0].latency) * audio.CARD_RATE_HZ)
        assert sa.lat_provenance == "nominal"
        assert sa._lat_samples == []
        assert sa.holdback == 128 + round(.0123 * audio.CARD_RATE_HZ)
        assert all(not block.any() for block in made[0].out_history)
        warnings = [r.message for r in caplog.records if "transmit timing is nominal" in r.message]
        assert len(warnings) == 1
    finally:
        sa.close()


def test_wasapi_startup_needs_a_block_but_no_settling_callbacks(monkeypatch):
    sa, _ = _fix23_station(monkeypatch, callbacks=1)
    sa.open()
    try:
        assert sa.samples == 128 and sa.lat_provenance == "nominal"
    finally:
        sa.close()


@pytest.mark.parametrize("callbacks", [1, audio._LAT_MEDIAN_N])
def test_wasapi_startup_callback_failure_closes_before_raising(monkeypatch, callbacks):
    sa, made = _fix23_station(monkeypatch, callbacks=callbacks)
    failure = RuntimeError("subscribed lane failed at startup")
    closed = []
    monkeypatch.setattr(FakeStream, "stop", lambda self: closed.append("stop"))
    monkeypatch.setattr(FakeStream, "close", lambda self: closed.append("close"))

    class FailingLane:
        def push(self, block):
            if sa.samples == callbacks * sa.blocksize:
                raise failure

    sa.subscribe(FailingLane())
    try:
        with pytest.raises(AudioError, match="audio callback failed: subscribed lane failed at startup"):
            sa.open()
        assert sa.failure is failure
        assert sa.samples == callbacks * sa.blocksize
        assert sa.lat_provenance == "nominal" and sa.lat is not None
        assert sa._stream is None
        assert closed == ["stop", "close"]
        assert not made[0].out_history[-1].any()
    finally:
        sa.close()


@pytest.mark.parametrize("host", ["Core Audio", "ALSA"])
def test_non_wasapi_rig_spread_still_fails(monkeypatch, host):
    sa, _ = _fix23_station(monkeypatch, host=host)
    with pytest.raises(AudioError, match=r"varies by 480 samples over 64 callbacks \(median 128\)"):
        sa.open()
    assert sa.lat is None and sa._stream is None


@pytest.mark.parametrize("host", ["Core Audio", "ALSA"])
def test_non_wasapi_constant_stamps_still_settle_to_median(monkeypatch, caplog, host):
    sa, _ = _fix23_station(monkeypatch, host=host, varying=False)
    sa.open()
    try:
        assert sa.lat == 128 and sa.lat_provenance == "measured"
        assert len(sa._lat_samples) == audio._LAT_MEDIAN_N
        assert not caplog.records
    finally:
        sa.close()


def test_nominal_offset_places_stream_through_loop_lat(monkeypatch):
    sa, made = _fix23_station(monkeypatch)
    sa.tx_latency_n = 960
    sa.open()
    try:
        nominal = round(sum(made[0].latency) * audio.CARD_RATE_HZ)
        assert sa.loop_lat == nominal + 960
        at = sa.samples + nominal + 960 + 3 * sa.blocksize + 17
        tx = sa.arm_stream(at, capacity=512)
        tx.write(np.ones(200, np.float32))
        tx.close()
        made[0].run(6)
        first = int(np.flatnonzero(np.concatenate(made[0].out_history))[0])
        assert first + sa.loop_lat == at
        assert tx.emitted == 200 and tx.end == at + 200
    finally:
        sa.close()


def test_the_offset_is_the_median_of_many_callbacks(monkeypatch):
    """Measured stable to zero over 22208 callbacks — which is why it is safe to
    use, and not a reason to read it once."""
    sa, _ = station(monkeypatch, lat=1152)
    # Driven synchronously: open() spins until the offset settles, so the stream
    # is primed here rather than raced against.
    sa._stream = FakeStream(sa._callback, lat=1152)
    sa._stream.run(audio._LAT_MEDIAN_N)
    assert sa.lat == 1152
    assert len(sa._lat_samples) >= audio._LAT_MEDIAN_N


def test_a_single_outlier_does_not_move_the_offset(monkeypatch):
    """A transient in one callback would make the station eat the opening symbols
    of every reply, forever, self-consistently. The median absorbs it."""
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    s.run(1, lat=1100)                       # one bad reading
    s.run(audio._LAT_MEDIAN_N - 1, lat=1152)
    assert sa.lat == 1152


def test_a_varying_offset_is_refused_rather_than_averaged(monkeypatch):
    """A spread is not noise to average away — it says the offset is not constant
    on this hardware, and everything downstream assumes it is.

    The callback stores the failure instead of raising, because sounddevice wraps
    it with `error=paAbort` and a raise stops the stream permanently while
    `sample_now()` goes on extrapolating. So the assertion is that the station
    knows it is broken, not that an exception escaped a C callback."""
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    for i in range(audio._LAT_MEDIAN_N):
        s.run(1, lat=1000 + i * 20)          # drifts far past a block
    assert sa.failure is not None and "varies by" in str(sa.failure)
    with pytest.raises(AudioError, match="varies by"):
        sa.alive()


def test_priming_callbacks_are_not_measured(monkeypatch):
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    s.run(10, status=_Status(priming=True), lat=99)
    assert sa._lat_samples == []


# --- fan-out ------------------------------------------------------------------

def test_one_block_reaches_every_lane_at_its_own_rate(monkeypatch):
    sa, _ = station(monkeypatch)
    sa._stream = FakeStream(sa._callback)
    lanes = [sa.subscribe(CycleLane(48000)), sa.subscribe(CycleLane(12000))]
    sa._stream.run(4)
    for lane in lanes:
        assert lane._seen == 4


def test_a_slow_lane_drops_its_oldest_and_counts(monkeypatch):
    """A decoder that falls behind must not stall the card or its neighbours."""
    sa, _ = station(monkeypatch)
    sa._stream = FakeStream(sa._callback)
    slow = sa.subscribe(CycleLane(48000, depth=4))
    fast = sa.subscribe(CycleLane(48000, depth=4096))
    sa._stream.run(20)
    assert slow.dropped > 0
    assert fast.dropped == 0, "one lane's backlog reached another"


def test_index_arithmetic_is_exact_across_rates():
    from hfmodem.core import rates
    for n in (0, 128, 4800, 48000):
        assert rates.to_native(n, 12000) * 4 == n


# --- level-blind --------------------------------------------------------------

def test_nothing_is_gated_on_level(monkeypatch):
    """besra gated at 200 int16 while this station's quiet channel measures 3935.
    The gate opened on the first block and never closed, and a perfect answer
    would have been discarded ahead of the demodulator."""
    seen: list[int] = []
    lane = RollingLane(48000, decode=lambda buf: seen.append(len(buf)) or [])
    faint = (np.random.default_rng(0).normal(0, 2.0, 48000)).astype(np.int16)
    lane.push(Block(0.0, 0, faint))
    lane.poll()
    assert seen and seen[0] == len(faint), "a level gate swallowed the buffer"


# --- half duplex is enforced once --------------------------------------------

def test_finishing_a_burst_advances_every_lane_past_our_own_carrier(monkeypatch):
    sa, _ = station(monkeypatch)
    sa._stream = FakeStream(sa._callback, lat=1152)
    sa._stream.run(audio._LAT_MEDIAN_N)
    a = sa.subscribe(CycleLane(48000))
    b = sa.subscribe(RollingLane(48000, decode=lambda buf: []))
    sa.arm_burst(np.zeros(4800, np.float32), at=10_000)
    end = sa.finish_burst()
    assert end == 10_000 + 4800
    assert a.pos >= end
    assert b._buf.size == 0


def test_finite_dac_placement_reports_on_air_endpoint(monkeypatch):
    """The arbiter subtracts TX latency before arm_burst; the callback only
    removes the driver offset. finish_burst includes TX latency for RX flushes.
    """
    def lead(tx_latency_n: int) -> int:
        sa, _ = station(monkeypatch)
        sa.tx_latency_n = tx_latency_n
        s = FakeStream(sa._callback, lat=1152)
        sa._stream = s
        s.run(audio._LAT_MEDIAN_N)
        on_air = sa.samples + 5000
        sa.arm_burst(np.ones(480, np.float32), at=on_air - tx_latency_n)
        s.run(40)
        assert sa.finish_burst() == on_air + 480
        first_frame = int(np.flatnonzero(np.concatenate(s.out_history))[0])
        assert first_frame + sa.lat + tx_latency_n == on_air
        return on_air - first_frame

    assert lead(0) == 1152
    assert lead(960) == 1152 + 960


def test_the_replay_transmitter_takes_no_such_correction():
    r = ReplayAudio(np.zeros(4800, np.float32))
    assert r.lat == 0 and r.tx_latency_n == 0


def test_nothing_is_emitted_between_bursts(monkeypatch):
    """The invariant besra's per-burst close was protecting. One stream satisfies
    it by zero-filling, and keeps the transmit index arithmetic."""
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    s.run(audio._LAT_MEDIAN_N)
    s.out_history.clear()
    s.run(8)
    assert all(np.all(blk == 0.0) for blk in s.out_history), "output between bursts"


def test_an_underrun_during_a_burst_invalidates_it(monkeypatch):
    """A dropout mid-burst is a hole in what we transmitted. Abandon it rather
    than report it sent — ARQ exists to retry."""
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    s.run(audio._LAT_MEDIAN_N)
    at = sa.arm_burst(np.zeros(4800, np.float32), at=sa.samples)
    # Emit part of the burst first. An underrun during the lead before `at`, or
    # after the last sample, is not a hole in what went out — invalidating a burst
    # that was transmitted intact is the wrong direction to be wrong.
    while sa.samples < at + 2400:
        s.run(1)
    assert 0 < sa._emitted < 4800
    s.run(1, status=_Status(underflow=True))
    with pytest.raises(BurstInvalidated, match="not what was composed"):
        sa.finish_burst()


def test_an_underrun_outside_the_emitted_range_does_not_invalidate(monkeypatch):
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    s.run(audio._LAT_MEDIAN_N)
    sa.arm_burst(np.zeros(4800, np.float32), at=sa.samples + 48000)
    s.run(1, status=_Status(underflow=True))     # still in the lead
    sa.finish_burst()                            # no raise


# --- an open-ended transmission ------------------------------------------------

def _streaming(monkeypatch):
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    s.run(audio._LAT_MEDIAN_N)
    s.out_history.clear()
    return sa, s


def test_a_stream_goes_out_in_order_from_its_index(monkeypatch):
    sa, s = _streaming(monkeypatch)
    start = sa.samples
    tx = sa.arm_stream(start + 3000, capacity=48000)
    ramp = np.arange(1, 2001, dtype=np.float32) / 4000
    tx.write(ramp[:700])
    tx.write(ramp[700:])
    tx.close()
    while not sa.burst_done():
        s.run(1)
    out = np.concatenate(s.out_history)
    first = int(np.flatnonzero(out)[0])
    assert tx.at - (start + first) == 1152
    np.testing.assert_array_equal(out[first:first + 2000], ramp)
    assert not np.any(out[first + 2000:])
    assert sa.finish_burst() == tx.at + 2000


def test_a_stream_is_bounded(monkeypatch):
    sa, _ = _streaming(monkeypatch)
    tx = sa.arm_stream(sa.samples, capacity=1000)
    tx.write(np.zeros(1000, np.float32))
    with pytest.raises(AudioError, match="full"):
        tx.write(np.zeros(1, np.float32))
    tx.close()
    with pytest.raises(AudioError, match="closed"):
        tx.write(np.zeros(1, np.float32))


def test_a_stream_that_runs_dry_while_open_is_invalid_and_says_where(monkeypatch):
    sa, s = _streaming(monkeypatch)
    tx = sa.arm_stream(sa.samples, capacity=48000)
    tx.write(np.full(600, .1, np.float32))
    while tx.emitted < 600:
        s.run(1)
    s.run(4)
    assert tx.starved_at is not None and not sa.burst_done()
    assert sa.samples - tx.starved_at <= 5 * 128
    tx.write(np.full(300, .1, np.float32))
    s.run(1)
    assert tx.starved_at is None
    with pytest.raises(BurstInvalidated):
        sa.finish_burst()


def test_a_stream_waiting_for_its_index_is_not_starved(monkeypatch):
    sa, s = _streaming(monkeypatch)
    tx = sa.arm_stream(sa.samples + 48000, capacity=48000)
    s.run(10)
    assert tx.starved_at is None and tx.emitted == 0
    tx.close()
    assert sa.finish_burst() == tx.at


def test_finishing_a_stream_flushes_every_lane_and_drops_the_rest(monkeypatch):
    sa, s = _streaming(monkeypatch)
    a = sa.subscribe(CycleLane(48000))
    b = sa.subscribe(RollingLane(48000, decode=lambda buf: []))
    tx = sa.arm_stream(sa.samples, capacity=48000)
    tx.write(np.full(4800, .2, np.float32))
    while tx.emitted < 1000:
        s.run(1)
    end = sa.finish_burst()
    assert end == tx.at + tx.emitted < tx.at + 4800
    assert a.pos >= end and b._buf.size == 0
    s.out_history.clear()
    s.run(40)
    assert not any(np.any(blk) for blk in s.out_history)
    assert sa.burst_done()


def test_an_underrun_mid_stream_invalidates_it(monkeypatch):
    sa, s = _streaming(monkeypatch)
    tx = sa.arm_stream(sa.samples, capacity=48000)
    tx.write(np.zeros(4800, np.float32))
    while tx.emitted == 0:
        s.run(1)
    s.run(1, status=_Status(underflow=True))
    with pytest.raises(BurstInvalidated):
        sa.finish_burst()


def test_a_reader_mid_callback_never_sees_a_stream_drained_before_its_end(monkeypatch):
    sa, s = _streaming(monkeypatch)
    tx = sa.arm_stream(sa.samples, capacity=48000)
    tx.write(np.full(300, .1, np.float32))
    while tx.emitted < 300:
        s.run(1)
    s.run(3)
    tx.write(np.full(200, .1, np.float32))
    tx.close()
    seen, read = [], tx._read

    def watched(n):
        got = read(n)
        seen.append((tx.drained, tx.end))
        return got

    monkeypatch.setattr(tx, '_read', watched)
    while not sa.burst_done():
        s.run(1)
    assert all(end >= tx.end for drained, end in seen if drained)


def test_the_card_index_of_every_sample_counts_the_holes_before_it(monkeypatch):
    sa, s = _streaming(monkeypatch)
    start = sa.samples
    tx = sa.arm_stream(start + 500, capacity=48000)
    tx.write(np.arange(1, 301, dtype=np.float32) / 1000)
    while tx.emitted < 300:
        s.run(1)
    s.run(3)
    tx.write(np.arange(301, 501, dtype=np.float32) / 1000)
    tx.close()
    while not sa.burst_done():
        s.run(1)
    out = np.concatenate(s.out_history)
    progress = tx.progress
    assert len(progress.runs) == 2
    for position in (1, 150, 300, 301, 450, 500):
        at = int(np.flatnonzero(np.isclose(out, position / 1000))[0])
        assert progress.card_end(position) == start + at + 1 + sa.loop_lat
    assert progress.card_end(500) == tx.end
    assert progress.card_end(0) is None and progress.card_end(501) is None


# --- xrun -------------------------------------------------------------------

@pytest.mark.parametrize("host", ["Windows WASAPI", "MME", "Core Audio", "ALSA",
                                  None, "unavailable"])
@pytest.mark.parametrize("input_only", [True, False])
def test_host_api_selects_timestamp_loss_detection(monkeypatch, host, input_only):
    class StartingStream(FakeStream):
        def start(self):
            self.run(audio._LAT_MEDIAN_N)

    class FakeSd:
        @staticmethod
        def Stream(callback, blocksize, **_):
            stream = StartingStream(callback, blocksize=blocksize)
            if host is not None:
                stream.device = (7, 8)
            return stream

        @staticmethod
        def InputStream(callback, blocksize, **_):
            stream = StartingStream(lambda data, out, frames, t, status:
                                    callback(data, frames, t, status),
                                    blocksize=blocksize, latency=0.01)
            if host is not None:
                stream.device = 7
            return stream

        @staticmethod
        def query_devices(device):
            assert device == 7
            if host == "unavailable":
                raise RuntimeError("device metadata unavailable")
            return {"hostapi": 3}

        @staticmethod
        def query_hostapis(index):
            assert index == 3
            return {"name": host}

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSd)
    sa = StationAudio(input_device="fake-in",
                      output_device=None if input_only else "fake-out")
    faults = []
    sa.subscribe(SimpleNamespace(push=lambda block: None,
                                 fault=lambda at, cause: faults.append((at, cause))))
    sa.open()
    try:
        assert isinstance(sa._floor_loss, audio.rates.FloorLoss) == (host == "Windows WASAPI")
        if host == "Windows WASAPI":
            assert sa.timestamp_limitation == "WASAPI timestamps carry no converter time"
        else:
            assert sa.timestamp_limitation is None
        data = np.zeros((128, 1), np.float32)

        def deliver(ahead):
            adc = sa.samples / audio.CARD_RATE_HZ + ahead
            sa._input_callback(data, 128, _Times(adc, adc + 0.024), _Status())
            assert sa.failure is None

        for i in range(450):
            deliver(0.012 if i % 2 == 0 else 0.0)
        if host == "Windows WASAPI":
            assert sa.lost == 0 and faults == []
            for _ in range(450):
                deliver(0.1)
            assert sa.lost == 0
            assert sa.estimated_lost == 4800
            assert faults == []
        else:
            assert sa.lost == 225 * 576
            assert len(faults) == 225
            assert all(cause == "timestamp loss" for _, cause in faults)
    finally:
        sa.close()


def test_an_overflow_resyncs_every_lane(monkeypatch):
    sa, _ = station(monkeypatch)
    s = FakeStream(sa._callback, lat=1152)
    sa._stream = s
    s.run(audio._LAT_MEDIAN_N)
    cyc = sa.subscribe(CycleLane(48000))
    roll = sa.subscribe(RollingLane(48000, decode=lambda buf: []))
    s.run(1, status=_Status(overflow=True))
    assert sa.overflows == 1
    assert cyc.resyncs == 1 and roll.resyncs == 1


def test_a_lost_block_invalidates_a_cycle_grid_but_not_a_rolling_window(monkeypatch):
    """The grid is *made of* the sample count, so a lost block moves every later
    boundary. A rolling window loses one window and carries on."""
    cyc, roll = CycleLane(48000), RollingLane(48000, decode=lambda b: [])
    assert cyc.grid_valid
    cyc.resync(9999)
    roll.resync(9999)
    assert not cyc.grid_valid, "the grid claims to be valid after a discontinuity"
    # resync publishes an index; the consumer applies it. A read-modify-write from
    # the audio callback races the decode thread doing the same, and a turnaround
    # is exactly when both happen.
    assert cyc._resync_at == 9999 and roll._resync_at == 9999
    cyc.push(Block(0.0, 9999, np.zeros(128, np.float32)))
    cyc.take_until(10_000)
    assert cyc.pos is not None and cyc.pos >= 9999


# --- the clock fit ----------------------------------------------------------

def test_the_clock_fit_ignores_delivery_jitter(monkeypatch):
    """3 ms rms of callback-delivery jitter must not move the ppm estimate.

    The fit pairs the converter's own ADC timestamp with its sample index --
    formed before any Python delivery delay -- so a loaded machine reads the
    same crystal as an idle one. Anchored on callback-entry time instead, this
    exact scenario reads tens of ppm of pure fiction: the -32 ppm (sigma 17)
    one 50 s session reported was the jitter and nothing else, on a crystal
    two clean long fits place within a few ppm of true.
    """
    sa, _ = station(monkeypatch)
    rng = np.random.default_rng(7)
    ppm_true = 2.4
    frames = 128
    indata = np.zeros((frames, 1), np.float32)
    outdata = np.zeros((frames, 1), np.float32)
    entry = {"now": 0.0}
    monkeypatch.setattr(audio.time, "monotonic", lambda: entry["now"])
    n = 0
    while n < 50 * audio.CARD_RATE_HZ:                      # 50 s of stream
        adc = n / (audio.CARD_RATE_HZ * (1 + ppm_true * 1e-6))
        # Delivery trails the hardware instant; it never leads it.
        entry["now"] = adc + abs(rng.normal(0.0, 0.003))
        sa._callback(indata, outdata, frames,
                     _Times(adc, adc + 1152 / audio.CARD_RATE_HZ), _Status())
        n += frames
    ppm, sigma = sa.clock_ppm()
    assert abs(ppm - ppm_true) < 0.1, f"{ppm=} against {ppm_true} planted"
    assert sigma < 0.1


# --- startup ----------------------------------------------------------------

def test_a_stream_that_delivers_nothing_is_fatal_and_says_why(monkeypatch):
    """The silent-total failure this module exists to make loud."""
    monkeypatch.setattr(audio, "STARTUP_DEADLINE_S", 0.05)

    class DeadSd:
        @staticmethod
        def Stream(callback=None, blocksize=128, **_):
            return FakeStream(callback, blocksize=blocksize)

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", DeadSd)
    sa = StationAudio(input_device="dead", output_device="dead")
    with pytest.raises(AudioError, match="cannot hear"):
        sa.open()


# --- replay -----------------------------------------------------------------

def test_the_whole_thing_runs_on_a_recording():
    """No device opened. This is how the transmit-adjacent code gets exercised
    without a transmitter."""
    tone = (np.sin(2 * np.pi * 1500 * np.arange(48000) / 48000) * 8000).astype(np.int16)
    r = ReplayAudio(tone)
    got: list[int] = []
    r.subscribe(RollingLane(48000, decode=lambda buf: got.append(len(buf)) or []))
    while r.pump(64):
        pass
    assert r.exhausted() and r.samples == len(tone)
    assert got == [] or all(n > 0 for n in got)


def test_replay_holds_no_holdback():
    assert ReplayAudio(np.zeros(10, np.int16)).holdback == 0


# --- the transmit call ------------------------------------------------------


class _Out:
    """An output stream that records the order it was driven in."""

    def __init__(self, log, **kw):
        self.log = log
        log.append(("open", kw["channels"]))

    def start(self):
        self.log.append(("start",))

    def write(self, block):
        self.log.append(("write", block.shape))

    def stop(self):
        self.log.append(("stop",))

    def close(self):
        self.log.append(("close",))


def _card(monkeypatch) -> list:
    log: list = []

    class FakeSd:
        @staticmethod
        def OutputStream(**kw):
            return _Out(log, **kw)

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSd)
    return log


def test_the_stream_is_stopped_before_it_is_closed(monkeypatch):
    """`Stream.stop()` is `Pa_StopStream`, which waits for the pending buffers.
    Closing a stream nobody stopped is `paAbort`, and the tail of the
    transmission is discarded rather than played."""
    log = _card(monkeypatch)
    play_drained(np.zeros(1200, np.float32), 48000, "fake-out")
    assert log == [("open", 1), ("start",), ("write", (1200, 1)),
                   ("stop",), ("close",)]


def test_a_multichannel_block_keeps_its_channels(monkeypatch):
    """A mono stream is inaudible on a multi-channel virtual device, so a caller
    that filled the device's channel count must get the stream it shaped for."""
    log = _card(monkeypatch)
    play_drained(np.zeros((1200, 4), np.float32), 48000, "fake-out")
    assert ("open", 4) in log and ("write", (1200, 4)) in log


class _Ring:
    """A running stream's ring buffer: a write fills it and the callback takes it
    back one poll at a time, which is the only thing a held stream can drain on."""

    ROOM = 4096

    def __init__(self, log, **kw):
        self.log = log
        self.pending = 0
        log.append(("open", kw["channels"]))

    @property
    def write_available(self):
        room = self.ROOM - self.pending
        self.pending = max(self.pending - 1, 0)
        return room

    def start(self):
        self.log.append(("start",))

    def write(self, block):
        self.log.append(("write", block.shape))
        self.pending = 3

    def stop(self):
        self.log.append(("stop",))

    def close(self):
        self.log.append(("close",))


def _held_card(monkeypatch, channels: int = 1) -> list:
    log: list = []

    class FakeSd:
        @staticmethod
        def OutputStream(**kw):
            return _Ring(log, **kw)

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSd)
    audio.hold_output("fake-out", 48000, channels)
    return log


def test_a_held_stream_is_written_to_and_outlives_the_burst(monkeypatch):
    """The whole of what holding buys: no open and no start under the key. On a
    virtual cable that start is 0.15 s in front of every answer this station
    keys, against 0.02 s once the stream is up."""
    log = _held_card(monkeypatch)
    try:
        play_drained(np.zeros(1200, np.float32), 48000, "fake-out")
        assert log == [("open", 1), ("start",), ("write", (1200, 1))]
    finally:
        drop_output()
    assert log[-2:] == [("stop",), ("close",)]


def test_a_held_stream_is_drained_before_the_call_returns(monkeypatch):
    """`stop` is what drains a per-burst stream and a held one never stops, so
    the ring emptying is what stands in for it — the caller unkeys on this
    returning and must not do it over its own last samples."""
    _held_card(monkeypatch)
    try:
        play_drained(np.zeros(1200, np.float32), 48000, "fake-out")
        held = audio._HELD[("fake-out", 1)]
        assert held.pending == 0
    finally:
        drop_output()


def test_a_held_stream_serves_only_the_shape_it_was_opened_for(monkeypatch):
    """A burst that fills a different channel count is a different stream, and
    writing it into this one would put the signal on the wrong carriers."""
    log = _held_card(monkeypatch, channels=1)
    try:
        play_drained(np.zeros((1200, 4), np.float32), 48000, "fake-out")
        assert log[-5:] == [("open", 4), ("start",), ("write", (1200, 4)),
                            ("stop",), ("close",)]
    finally:
        drop_output()


#: Everything PortAudio will let a caller push samples out of, and `sd.play`,
#: which is the module-global one.
_OUTPUTS = {"OutputStream", "RawOutputStream", "Stream", "RawStream"}
_BLOCKING = {"play", "playrec"}
#: What the module is reached through. `self._sd.play(...)` is `sd.play(...)` with
#: the module parked on an attribute, and a caller that does that is not using a
#: different API.
_HANDLES = {"sd", "_sd", "sounddevice"}


def _hands_audio_out(node: ast.AST) -> bool:
    """Whether this call hands a rendered array to a sound card to transmit.

    The distinction is the callback, and it is the whole of what separates a
    transmission from the two device-to-device bridges (`tools/rigaudio.py`,
    `tools/vara_rig_bridge.py`). A stream given a callback is a continuous pump:
    PortAudio asks it for the next block and nobody ever writes a burst into it,
    so there is no tail for a missing drain to lose. A stream given none exists
    to have a finished array written into it and then stopped — which is the
    sequence `play_drained` is, and the sequence `tools/ptt_tail_check.py`
    measures on the radio.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    recv = node.func.value
    held = (recv.id if isinstance(recv, ast.Name)
            else recv.attr if isinstance(recv, ast.Attribute) else None)
    if held not in _HANDLES:
        return False
    if node.func.attr in _BLOCKING:
        return True
    return (node.func.attr in _OUTPUTS
            and not any(kw.arg == "callback" for kw in node.keywords))


def _sealed(rel: str) -> bool:
    """Directories the walk does not enter, neither of them for convenience.

    Closed historical analyses and third-party reference checkouts are outside
    this scan of active transmit code.
    """
    parts = rel.split("/")[:-1]
    return (rel.startswith("working/ardop/reference/")
            or any(d.endswith("-disasm") for d in parts))


def _transmit_path(text: str) -> tuple[dict[str, str], int]:
    """A module's sound-card calls: the source of every top-level function that
    makes one, and how many there are in the file altogether.

    What clears an archived copy of `core.audio` used to be identity with the
    whole live file, and that made a dozen frozen bisect snapshots -- which
    nobody may edit, because they are the record of what flew -- fail the gate
    the moment the live module changed anywhere in it. What is blessed is the
    transmit path, so the transmit path is what is compared: a copy whose
    `play_drained` or `hold_output` has DRIFTED is still walked, and so is one
    that grew a sound-card call somewhere else, which is what the count is for.
    """
    tree = ast.parse(text)
    return ({node.name: ast.get_source_segment(text, node) for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and any(_hands_audio_out(n) for n in ast.walk(node))},
            sum(1 for n in ast.walk(tree) if _hands_audio_out(n)))


def _audio_archives():
    ledger = REPO / "working/audio-source-archives.json"
    if not ledger.is_file():
        return {}
    return {REPO / row["path"]: row["sha256"]
            for row in json.loads(ledger.read_text())["archives"]}


def _attested_basetemp(directory: Path):
    """Only an adjacent completed pytest report can attest a generated fixture tree."""
    status = directory / "status.json"
    report = directory / "results.xml"
    log = directory / "stdout.log"
    if not all(p.is_file() for p in (status, report, log)):
        return None
    try:
        record = json.loads(status.read_text())
        command = record["command"]
        if (record.get("finished") is None or not isinstance(record.get("exit_code"), int)
                or not isinstance(command, list) or not log.stat().st_size):
            return None
        if not any(isinstance(arg, str) and "pytest" in Path(arg).name for arg in command):
            return None
        if "--basetemp" in command:
            target_arg = command[command.index("--basetemp") + 1]
        else:
            target_arg = next(arg.split("=", 1)[1] for arg in command
                              if isinstance(arg, str) and arg.startswith("--basetemp="))
        target = Path(target_arg).resolve()
        if not target.is_relative_to(directory.resolve()) or target == directory.resolve():
            return None
        if f"--junitxml={report.resolve()}" not in command:
            return None
        if not ET.parse(report).findall(".//testcase"):
            return None
        return target
    except (ValueError, KeyError, IndexError, TypeError, OSError, ET.ParseError, StopIteration):
        return None


def _python_sources(root: Path):
    """Maintained instruments, excluding independent repos and attested artifacts.

    Exact-hash historical sources remain unchanged as evidence, without being
    certified as current live instruments. Changed bytes are always scanned.
    """
    archives = _audio_archives()
    basetemps = set()
    for directory, subdirs, files in root.walk():
        attested = _attested_basetemp(directory)
        if attested is not None:
            basetemps.add(attested)

        def separate_repository(name):
            candidate = directory / name
            return ((candidate / ".git").exists()
                    or all((candidate / marker).is_file() for marker in (
                        "pyproject.toml", "packages/hfmodem/pyproject.toml",
                        "packages/hfmodem/hfmodem/core/audio.py")))

        subdirs[:] = sorted(name for name in subdirs if not separate_repository(name)
                            and (directory / name).resolve() not in basetemps)
        for name in sorted(files):
            src = directory / name
            if src.suffix == ".py" and src.is_file():
                if src in archives and sha256(src.read_bytes()).hexdigest() == archives[src]:
                    continue
                yield src


def test_python_source_walk_keeps_instruments_inside_py_named_directories(tmp_path):
    artifact = tmp_path / "reported-case.py"
    artifact.mkdir()
    instrument = artifact / "instrument.py"
    instrument.write_text("import sounddevice as sd\nsd.play([1])\n")
    ordinary = tmp_path / "ordinary.py"
    ordinary.write_text("value = 1\n")
    copied = tmp_path / "independent-copy"
    copied_audio = copied / "packages/hfmodem/hfmodem/core/audio.py"
    copied_audio.parent.mkdir(parents=True)
    copied_audio.write_text("import sounddevice as sd\nsd.play([1])\n")
    (copied / "pyproject.toml").write_text("[project]\nname='copy'\n")
    (copied / "packages/hfmodem/pyproject.toml").write_text("[project]\nname='hfmodem'\n")
    worktree = tmp_path / "independent-worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /historical/worktree\n")
    (worktree / "instrument.py").write_text(instrument.read_text())

    sources = list(_python_sources(tmp_path))
    assert set(sources) == {instrument, ordinary}
    assert any(_hands_audio_out(node) for node in ast.walk(ast.parse(instrument.read_text())))


def test_audio_source_archives_are_exact_and_do_not_cover_neighbors(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__, fromlist=["REPO"]), "REPO", tmp_path)
    working = tmp_path / "working"
    working.mkdir()
    archived = working / "saved.py"
    neighbor = working / "current.py"
    bad = "import sounddevice as sd\nsd.play([1])\n"
    archived.write_text(bad)
    neighbor.write_text(bad)
    (working / "audio-source-archives.json").write_text(json.dumps({"archives": [{
        "path": "working/saved.py", "sha256": sha256(archived.read_bytes()).hexdigest()}]}))
    assert list(_python_sources(working)) == [neighbor]
    archived.write_text(bad + "# changed\n")
    sources = list(_python_sources(working))
    assert set(sources) == {archived, neighbor}
    assert all(any(_hands_audio_out(n) for n in ast.walk(ast.parse(p.read_text())))
               for p in sources)


@pytest.mark.parametrize("joined", [False, True])
def test_audio_source_walk_requires_completed_adjacent_pytest_provenance(tmp_path, joined):
    fixture = tmp_path / "temporary/candidate.py"
    fixture.parent.mkdir()
    fixture.write_text("intentionally invalid fixture\n")
    instrument = tmp_path / "instrument.py"
    instrument.write_text("import sounddevice as sd\nsd.play([1])\n")
    status = tmp_path / "status.json"
    report = tmp_path / "results.xml"
    report.write_text('<testsuites><testsuite><testcase name="fixture"/></testsuite></testsuites>')
    (tmp_path / "stdout.log").write_text("1 passed\n")
    record = {"finished": 1, "exit_code": 0, "command": ["python", "-m", "pytest",
        "--basetemp", str(fixture.parent), f"--junitxml={report}"]}
    if joined:
        record["command"][3:5] = [f"--basetemp={fixture.parent}"]
    status.write_text(json.dumps(record))
    assert list(_python_sources(tmp_path)) == [instrument]
    if joined:
        record["command"][3] = f"--basetemp={tmp_path.parent / 'outside'}"
    else:
        record["command"][4] = str(tmp_path.parent / "outside")
    status.write_text(json.dumps(record))
    assert set(_python_sources(tmp_path)) == {fixture, instrument}
    if joined:
        record["command"][3] = f"--basetemp={fixture.parent}"
    else:
        record["command"][4] = str(fixture.parent)
    record.pop("finished")
    status.write_text(json.dumps(record))
    assert set(_python_sources(tmp_path)) == {fixture, instrument}


def test_only_core_audio_hands_a_transmission_to_a_sound_card():
    """One transmit call, and `sd.play` is only one of the ways of not being it.

    The property is not "nobody calls `sd.play`": that is one API of several, and
    a gate on the name alone passes a module that opens its own `sd.OutputStream`
    and writes a burst into it. What `tools/ptt_tail_check.py` measures on the
    radio is `play_drained`, so a second copy of it is a path the instrument has
    never seen — and `play_drained`'s claim to be the only one is this test.

    Walked as syntax rather than searched as text: what `sd.play` does to the end
    of a transmission is written down in four modems' comments and docstrings, and
    a gate that fires on the reasoning is a gate people switch off.

    `working/` is walked too, though it is scratch and is excluded from ruff and
    from pytest collection for being scratch. The bench instruments live there,
    and a probe there played every burst it ever
    transmitted short for exactly as long as this walk stopped at `tools/`. A
    transmit path that does not drain is not a measurement instrument, whichever
    directory it sits in.

    Independent repository copies are separate candidates. Exact-hash archived
    source in the maintained private ledger is historical evidence, not a
    certified current live instrument; changed bytes and unlisted neighbors are
    scanned. Adjacent completed pytest reports can attest their exact descendant
    fixture directory, which may deliberately contain invalid Python.
    """
    one = Path(audio.__file__).resolve()
    blessed = _transmit_path(one.read_text(encoding="utf-8"))
    hits = []
    for root in (REPO / "packages", REPO / "tools", REPO / "working"):
        for src in _python_sources(root):
            rel = src.relative_to(REPO).as_posix()
            if ({"tests", "__pycache__", "build", "dist"} & set(src.parts)
                    or src.resolve() == one or _sealed(rel)):
                continue
            try:
                text = src.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except SyntaxError as e:
                hits.append(f"{rel}:{e.lineno}: will not parse, so cannot be cleared")
                continue
            # A bisect archive holds this module, a dozen revisions of it. What
            # is blessed is the file, not the directory it sits in, so the
            # exemption is by content -- see `_transmit_path`.
            if src.name == one.name and _transmit_path(text) == blessed:
                continue
            hits += [f"{rel}:{node.lineno}"
                     for node in ast.walk(tree) if _hands_audio_out(node)]
    assert not hits, (
        "a blocking output stream is a transmission, and there is one of those in "
        "this project: `core.audio.play_drained`, which is what "
        "`tools/ptt_tail_check.py` puts on the radio. `sd.play` cannot be it — its "
        "callback raises CallbackAbort when the array runs out, which is "
        "PortAudio's paAbort, so what the device still holds is discarded rather "
        "than played and the transmission goes out short. A hand-rolled "
        "OutputStream is not it either: it is a path the instrument does not "
        "measure, and each one has so far picked its own channel count.\n  "
        + "\n  ".join(hits))


# --- capture the interpreter was too busy to accept ---------------------------

def test_blocks_the_interpreter_never_accepted_are_counted(monkeypatch):
    """`0 xruns` is not evidence the stream is intact. Eight pure-Python threads
    cost this station 87% of a 90 s capture with `input_overflow` false on every
    callback that ran; the only trace is the converter's own timestamp running
    ahead of the sample count Python was handed. Without it `clock_ppm` reads the
    shortfall as a slow crystal and preflight refuses a healthy card."""
    sa, _ = station(monkeypatch)
    frames = 128
    indata = np.zeros((frames, 1), np.float32)
    outdata = np.zeros((frames, 1), np.float32)
    adc, missed = 0.0, 0
    for i in range(400):
        if i and i % 10 == 0:           # one block in ten is captured and dropped
            adc += frames / audio.CARD_RATE_HZ
            missed += frames
        sa._callback(indata, outdata, frames,
                     _Times(adc, adc + 1152 / audio.CARD_RATE_HZ), _Status())
        adc += frames / audio.CARD_RATE_HZ
    assert sa.lost == missed


def test_an_unstarved_stream_loses_nothing(monkeypatch):
    """Crystal drift may never accumulate into the counter: the step has to clear
    half a block before it is loss, and a card 100 ppm out never does."""
    sa, _ = station(monkeypatch)
    frames = 128
    indata = np.zeros((frames, 1), np.float32)
    outdata = np.zeros((frames, 1), np.float32)
    for i in range(2000):
        adc = i * frames / (audio.CARD_RATE_HZ * (1 + 100e-6))
        sa._callback(indata, outdata, frames,
                     _Times(adc, adc + 1152 / audio.CARD_RATE_HZ), _Status())
    assert sa.lost == 0


@pytest.mark.parametrize('queued', [False, True])
def test_stream_flush_never_delivers_empty_chunks_or_loses_faults(queued):
    lane = audio.StreamLane(audio.CARD_RATE_HZ)
    if queued:
        lane.push(Block(0, 0, np.zeros(128, np.float32)))
    lane.flush_to(128)
    assert lane.poll() == []
    assert lane.poll() == []
    lane.push(Block(0, 128, np.ones(128, np.float32)))
    chunk, = lane.poll()
    assert chunk.at == 128
    assert chunk.samples.dtype == np.float32 and len(chunk.samples) == 128
    assert [(f.index, f.cause) for f in chunk.faults] == [(128, 'flush')]
    assert lane.poll() == []


def test_dac_anchor_uses_each_callback_not_startup_median(monkeypatch):
    sa, _ = station(monkeypatch)
    stream = FakeStream(sa._callback)
    stream.run(audio._LAT_MEDIAN_N)
    median = sa.lat
    stream.run(1, lat=median + 64)
    timestamp, frame = sa._dac
    assert timestamp == pytest.approx(sa._last[0] + (median + 64) / 48000)
    assert sa.lat == median
    assert sa.dac_time(frame + 480) == pytest.approx(timestamp + .01)
    assert sa.dac_frame_at(timestamp + .01) == pytest.approx(frame + 480)


def test_a_wasapi_dac_anchor_does_not_move_later_after_the_burst_end_is_waited_out(monkeypatch):
    """WASAPI stamps the DAC as callback time plus queued padding, and shared-mode
    padding falls a whole engine period at a time, so a callback early in the
    period stamps early. The 11-06 rig recheck's second RPR burst was waited out
    on one callback's anchor and judged on a later one's."""
    sa, _ = station(monkeypatch)
    sa.lat = 1152
    sa.timestamp_limitation = "WASAPI timestamps carry no converter time"
    s = FakeStream(sa._callback)
    early = [0, 120, 240, 360]
    k = 0
    def run(n):
        nonlocal k
        for _ in range(n):
            s.run(1, lat=sa.lat - early[k % len(early)])
            k += 1
    run(8)
    at = sa.reserve_burst(np.ones(1000, np.float32), at=0, deadline=10**9)
    while sa._consumed < 1000 or early[(k - 1) % len(early)] != 360:
        run(1)
    now = sa.dac_time(at - sa.loop_lat + 1000) + 1e-4
    k = 0
    run(1)
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now)
    monkeypatch.setattr(sa, 'check_callbacks', lambda: None)
    assert sa.finish_burst() == at + 1000


def test_dac_wait_uses_live_deadline_health_and_cancellation(monkeypatch):
    sa = StationAudio(input_device=None, output_device=None)
    sa._dac = (10., 1000)
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(sa, 'sample_now', lambda: 10**9)
    checks = []
    monkeypatch.setattr(sa, 'check_callbacks', lambda: checks.append(1))
    sa.wait_until_dac(1480, healthy=True, deadline=11)
    assert now[0] >= 10.01 and checks
    with pytest.raises(BurstInvalidated, match='cancelled'):
        sa.wait_until_dac(0, cancelled=lambda: True)
    with pytest.raises(BurstInvalidated, match='deadline'):
        sa.wait_until_dac(48000, deadline=now[0])


def test_cancel_discards_stream_and_closes_the_old_writer(monkeypatch):
    sa, stream = _streaming(monkeypatch)
    tx = sa.arm_stream(sa.samples, capacity=1000)
    tx.write(np.ones(900))
    sa.cancel_burst()
    assert sa._tx_stream is None and tx.closed
    stream.out_history.clear()
    stream.run(20)
    assert not any(np.any(block) for block in stream.out_history)
    with pytest.raises(AudioError, match='closed'):
        tx.write(np.ones(1))


@pytest.mark.parametrize('consumed,elapsed,valid', [(100, True, True), (99, True, False),
                                                   (100, False, False)])
def test_reserved_completion_uses_dac_and_keeps_emission_end(monkeypatch, consumed, elapsed, valid):
    sa = StationAudio(input_device="fake-in", output_device="fake-out", tx_latency_n=960)
    sa.lat = 1152
    sa._dac = (10., 0)
    at = sa.reserve_burst(np.ones(100), at=3000, deadline=10000)
    sa._consumed = consumed
    deadline = sa.dac_time(at - sa.loop_lat + 100)
    monkeypatch.setattr(audio.time, 'monotonic', lambda: deadline + (.001 if elapsed else -.001))
    monkeypatch.setattr(sa, 'check_callbacks', lambda: None)
    monkeypatch.setattr(sa, 'sample_now', lambda: at + 100 - 960)
    if valid:
        assert sa.finish_burst() == at + 100
    else:
        with pytest.raises(BurstInvalidated):
            sa.finish_burst()


@pytest.mark.parametrize('consumed,elapsed,why', [
    (99, True, "the card took 99 of 100 samples"),
    (100, False, "before its last sample reached the DAC")])
def test_a_reserved_burst_invalidated_without_an_underrun_says_why(monkeypatch, consumed,
                                                                  elapsed, why):
    sa = StationAudio(input_device="fake-in", output_device="fake-out")
    sa.lat = 1152
    sa._dac = (10., 0)
    at = sa.reserve_burst(np.ones(100), at=3000, deadline=10000)
    sa._consumed = consumed
    deadline = sa.dac_time(at - sa.loop_lat + 100)
    monkeypatch.setattr(audio.time, 'monotonic', lambda: deadline + (.001 if elapsed else -.001))
    monkeypatch.setattr(sa, 'check_callbacks', lambda: None)
    with pytest.raises(BurstInvalidated) as raised:
        sa.finish_burst()
    assert why in str(raised.value) and 'underrun' not in str(raised.value)


def test_silence_idle_allows_holes_but_not_device_underruns(monkeypatch):
    sa, stream = _streaming(monkeypatch)
    tx = sa.arm_stream(sa.samples, capacity=1000)
    tx.idle = 'silence'
    tx.write(np.ones(100))
    stream.run(30)
    assert tx.starved_at is not None and not sa._tx_invalid
    tx.write(np.ones(100))
    stream.run(1, status=_Status(underflow=True))
    assert sa._tx_invalid
    with pytest.raises(BurstInvalidated):
        sa.finish_burst()


def test_cancelled_ident_aborts_audio_before_the_remaining_tones(monkeypatch):
    log = _card(monkeypatch)
    monkeypatch.setattr(_Out, 'abort', lambda self: log.append(('abort',)), raising=False)
    with pytest.raises(InterruptedError, match='identification cancelled'):
        play_drained(np.ones(48000, np.float32), 48000, 'fake-out',
                     cancelled=lambda: any(item[0] == 'write' for item in log))
    assert log == [('open', 1), ('start',), ('write', (2400, 1)), ('abort',), ('close',)]


def test_cancelled_ident_retires_its_held_stream(monkeypatch):
    log = _held_card(monkeypatch)
    monkeypatch.setattr(_Ring, 'abort', lambda self: log.append(('abort',)), raising=False)
    try:
        with pytest.raises(InterruptedError, match='identification cancelled'):
            play_drained(np.ones(48000, np.float32), 48000, 'fake-out',
                         cancelled=lambda: any(item[0] == 'write' for item in log))
        assert log[-2:] == [('abort',), ('close',)]
        assert ('fake-out', 1) not in audio._HELD
    finally:
        drop_output()


@pytest.fixture
def marked_output(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    class Output:
        latency = .1
        def __init__(self):
            self.queued = 0
            self.started = clock.now
            self.underflow = False
            self.underflow_at = None
            self.writes = []
            self.closed = False
            self.active = True
        @property
        def time(self):
            return clock.now
        @property
        def played(self):
            return max(0, round((clock.now - self.started - self.latency) * 1000))
        @property
        def write_available(self):
            return 4096 - max(0, self.queued - self.played)
        def start(self):
            self.started = clock.now
        def write(self, block):
            if self.played >= self.queued:
                self.started = clock.now - self.queued / 1000
            self.writes.append(block.copy())
            self.queued += len(block)
            clock.now = max(clock.now, self.started + self.queued / 1000)
            underflow, self.underflow = self.underflow, False
            return underflow or len(self.writes) == self.underflow_at
        def stop(self):
            clock.now = max(clock.now, self.started + self.latency + self.queued / 1000)
        def close(self):
            self.closed = True
    stream = Output()
    monkeypatch.setitem(__import__('sys').modules, 'sounddevice',
                        SimpleNamespace(OutputStream=lambda **kw: stream))
    monkeypatch.setattr(audio.time, 'sleep', lambda seconds:
                        setattr(clock, 'now', clock.now + seconds))
    return stream, clock


@pytest.mark.parametrize('failed', [False, True])
def test_submission_hook_has_bounded_exact_pcm_and_excludes_failed_writes(marked_output, monkeypatch, failed):
    stream, _ = marked_output
    original = stream.write
    submitted = []
    def write(block):
        if failed and stream.queued >= 100:
            raise OSError('submission refused')
        return original(block)
    monkeypatch.setattr(stream, 'write', write)
    values = np.arange(250, dtype=np.float32) / 1000
    def play():
        audio.play_drained(values, 1000, 'synthetic',
                           submitted=lambda first, pcm: submitted.append((first, pcm.copy())))
    if failed:
        with pytest.raises(OSError, match='submission refused'):
            play()
    else:
        play()
    assert [first for first, _ in submitted] == list(range(0, 100 if failed else 250, 50))
    assert all(len(pcm) <= 50 for _, pcm in submitted)
    np.testing.assert_array_equal(np.concatenate([pcm for _, pcm in submitted]), values[:100] if failed else values)
    assert stream.closed


def test_submission_offsets_include_marked_playback_padding(marked_output):
    stream, _ = marked_output
    values = np.arange(120, dtype=np.float32) / 1000
    submitted = []
    audio.play_drained(values, 1000, 'synthetic', marks=([0, 100], lambda index: None),
                       submitted=lambda first, pcm: submitted.append((first, pcm.copy())))
    expected = np.r_[np.zeros(50, np.float32), values]
    assert [first for first, _ in submitted] == list(np.cumsum([0] + [len(pcm) for _, pcm in submitted[:-1]]))
    np.testing.assert_array_equal(np.concatenate([pcm for _, pcm in submitted]), expected)
    np.testing.assert_array_equal(np.concatenate(stream.writes)[:, 0], expected)


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('latency', [.005, .1])
def test_packet_marks_wait_for_dac_instead_of_enqueue(marked_output, held, latency):
    stream, clock = marked_output
    stream.latency = latency
    seen = []
    if held:
        audio.hold_output('fake-out', 1000)
    try:
        play_drained(np.ones(1000), 1000, 'fake-out', marks=([0, 400, 800],
            lambda k: seen.append((k, stream.played, stream.queued, clock.now))))
        assert np.array_equal(stream.writes[0], np.zeros((50, 1)))
        assert np.array_equal(np.concatenate(stream.writes[1:]), np.ones((1000, 1)))
        assert [k for k, *_ in seen] == [0, 1, 2]
        for (k, played, queued, at), offset in zip(seen, [0, 400, 800]):
            assert offset + 50 <= played <= offset + 51
            assert queued > played
            assert at >= stream.started + stream.latency + (offset + 50) / 1000
        assert stream.played >= 1050
    finally:
        drop_output()


@pytest.mark.parametrize('held', [False, True])
def test_playback_failure_before_buffered_mark_does_not_report_it(monkeypatch, marked_output, held):
    stream, _ = marked_output
    seen = []
    def fail(seconds):
        assert stream.queued > 50 and stream.played <= 50
        raise OSError('DAC failed')
    monkeypatch.setattr(audio.time, 'sleep', fail)
    if held:
        audio.hold_output('fake-out', 1000)
    with pytest.raises(OSError, match='DAC failed'):
        play_drained(np.ones(1000), 1000, 'fake-out', marks=([0], seen.append))
    assert seen == [] and stream.closed
    assert ('fake-out', 1) not in audio._HELD


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('write_number,reported', [(2, []), (3, [0])])
def test_output_underrun_invalidates_buffered_packet_marks(marked_output, held, write_number, reported):
    stream, _ = marked_output
    stream.underflow_at = write_number
    seen = []
    if held:
        audio.hold_output('fake-out', 1000)
    with pytest.raises(BurstInvalidated, match='output underrun'):
        play_drained(np.ones(1000), 1000, 'fake-out', marks=([0, 400], seen.append))
    assert seen == reported and stream.closed
    assert ('fake-out', 1) not in audio._HELD


def test_marked_burst_after_idle_held_output_clears_preburst_underrun(marked_output):
    stream, clock = marked_output
    audio.hold_output('fake-out', 1000)
    try:
        play_drained(np.ones(1000), 1000, 'fake-out')
        clock.now += 10
        stream.underflow = True
        first_write = len(stream.writes)
        seen = []
        play_drained(np.ones(1000), 1000, 'fake-out', marks=([0, 400, 800],
            lambda k: seen.append((k, stream.played))))
        assert [k for k, _ in seen] == [0, 1, 2]
        for (_, played), offset in zip(seen, [0, 400, 800]):
            assert 1050 + offset <= played <= 1051 + offset
        assert np.array_equal(stream.writes[first_write], np.zeros((50, 1)))
        assert np.array_equal(np.concatenate(stream.writes[first_write + 1:]), np.ones((1000, 1)))
        assert stream.played >= 2050
        assert audio._HELD['fake-out', 1] is stream and not stream.closed
    finally:
        drop_output()


def test_stopped_output_does_not_report_buffered_packet_marks(monkeypatch, marked_output):
    stream, clock = marked_output
    seen = []
    def stop(seconds):
        clock.now += 1
        stream.active = False
    monkeypatch.setattr(audio.time, 'sleep', stop)
    with pytest.raises(AudioError, match='output stream stopped'):
        play_drained(np.ones(1000), 1000, 'fake-out', marks=([0], seen.append))
    assert seen == [] and stream.closed
