# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A VARA attempt says what its own recording is missing.

Every advance this receive chain has made came out of a recording, and every
conclusion drawn from one rests on the recording being the air — sample k of the
file being sample k of the stream, with nothing spliced out of the middle. shrike
has said so at the end of every session since `lost` existed. kestrel never did:
`kestrel_connect.py` recorded from its own PortAudio callback and printed only how
many seconds it wrote, so not one VARA recording on this record carries a reading
of the timebase it is indexed on. Same interpreter, same PortAudio, same rig, and
the loss that produced -4860 ppm on a shrike arm is invisible on a kestrel one.

The counter it needed already exists in `core.rates`, which is the point of
sharing it: the third call site is what the sharing was for.

Run: pytest hfmodem/tests/kestrel/test_capture_clock.py
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora

kestrel_connect = corpora.harness("kestrel_connect")
clock_shortfall = corpora.harness("clock_shortfall")

FS = kestrel_connect.FS
BLOCK = 128


class _Times:
    def __init__(self, adc: float):
        self.currentTime = adc
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = adc


class _Status:
    input_overflow = input_underflow = output_underflow = False
    priming_output = False


@pytest.fixture
def io(monkeypatch):
    """`AudioVaraIO` with its input stream replaced, so the test drives the
    callback with timestamps of its own choosing."""
    class _FakeStream:
        blocksize = BLOCK
        latency = 0.01

        def __init__(self, callback):
            self.callback = callback

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    made = {}

    class _FakeSd:
        @staticmethod
        def InputStream(callback=None, **_):
            made["stream"] = _FakeStream(callback)
            return made["stream"]

        @staticmethod
        def query_devices(*_):
            return {"max_output_channels": 2}

    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd)
    return kestrel_connect.AudioVaraIO("fake-out", "fake-in", record=None)


def _drive(io, *, seconds: float, drop_every: int = 0) -> None:
    """Feed the callback `seconds` of converter time, block by block.

    `drop_every`: the Python callback is not invoked for one block in this many,
    while the converter's timestamps march on regardless — what a starved
    interpreter does to a real stream, with no status flag behind it.
    """
    ind = np.zeros((BLOCK, 1), np.float32)
    made, k = 0, 0
    while made < seconds * FS:
        adc = made / FS
        made += BLOCK
        k += 1
        if drop_every and k % drop_every == 0:
            continue
        io._on_rx(ind, BLOCK, _Times(adc), _Status())


def test_an_attempt_reports_the_timebase_its_recording_is_indexed_on(io):
    _drive(io, seconds=40.0)
    line = io.clock_report()
    assert line.startswith("capture clock:")
    assert "no loss we can see" in line
    assert "SPLICED" not in line


def test_a_starved_kestrel_stream_says_so_where_the_ranking_reads_it(io):
    """0.49% of a 60 s arm is 293 ms missing from the middle of the file, and
    `0 xruns` beside it reads as a clean stream."""
    _drive(io, seconds=60.0, drop_every=205)
    line = io.clock_report()
    assert io.xruns == 0, "the driver raised the flag this test needs it to miss"
    assert abs(io.lost / FS - 0.293) < 0.02, f"{io.lost} samples short"
    assert "SPLICED" in line, line
    row = clock_shortfall.CLOCK.search(line)
    assert row is not None, (
        "the line has to be the one tools/clock_shortfall.py ranks the record "
        f"on, or a kestrel capture is invisible to it: {line}")
    assert int(row.group(1)) == io.samples


def test_the_recording_and_the_reading_come_off_one_callback(io, tmp_path):
    """The count is only the file's if it is taken where the file is written. A
    reading assembled anywhere else describes a stream, not this WAV."""
    src = (kestrel_connect.AudioVaraIO._on_rx.__code__.co_names,
           kestrel_connect.AudioVaraIO.clock_report.__code__.co_names)
    assert "rec" in src[0] and "lost_step" in src[0], src[0]
    assert "clock_report" in src[1], src[1]


def test_the_constructor_asks_the_stream_for_nothing(monkeypatch):
    """`sd.InputStream` is opened without a `blocksize`, so PortAudio picks one
    and the object does not carry the answer -- the callback's `frames` does.
    Reading it off the stream instead cost `test_tx_drive.py` two arms, whose
    stub is a bare namespace because a transmit test has no reason to build a
    capture stream.
    """
    class _FakeSd:
        @staticmethod
        def InputStream(**_):
            return types.SimpleNamespace(start=lambda: None, stop=lambda: None,
                                         close=lambda: None)

        @staticmethod
        def query_devices(*_):
            return {"max_output_channels": 2}

    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd)
    io = kestrel_connect.AudioVaraIO("fake-out", "fake-in")
    ind = np.zeros((64, 1), np.float32)
    io._on_rx(ind, 64, _Times(0.0), _Status())
    io._on_rx(ind, 64, _Times(64 / FS), _Status())
    assert io.lost == 0 and io.samples == 128
    assert "capture clock:" in io.clock_report()


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
def test_kestrel_rig_replay_and_delivery_rate_estimate(monkeypatch, tmp_path, host):
    import json
    from hfmodem.tests.core.test_audio import _fix23_stamps
    from hfmodem.core import rates

    class Input:
        device = 17
        def __init__(self, callback, **kw):
            self.callback = callback
        def start(self):
            for t in list(_fix23_stamps())[:64]:
                self.callback(np.zeros((128, 1), np.float32), 128, t, _Status())
        def stop(self):
            pass
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(
        InputStream=Input,
        query_devices=lambda *a: {'hostapi': 3, 'max_output_channels': 2},
        query_hostapis=lambda api: {'name': host}))
    monkeypatch.setattr(kestrel_connect, '_bounded', lambda *a, **kw: None)
    path = tmp_path / 'receive.wav'
    io = kestrel_connect.AudioVaraIO('out', 'selected', record=path)
    io._stream.start()
    report = io.clock_report()
    try:
        assert io.samples == 8192
        if host == 'Windows WASAPI':
            assert io.lost == 0 and io.estimated_lost == 1920
            assert 'uncertain' in report and '1920' in report
            assert 'SPLICED' not in report
        else:
            assert io.lost == 1920 and 'SPLICED' in report
    finally:
        io.stop()
        io.stop_recording()
    if host == 'Windows WASAPI':
        meta = json.loads((io.rec.session.directory / 'recording.json').read_text())['lanes']['rx']
        assert meta['timestamp_limitation'] == io.timestamp_limitation
        assert meta['estimated_lost_samples'] == 1920 and meta['lost_samples'] == 0
        assert meta['continuous'] is None

    # Exact 48 kHz samples, delivery age decreasing 2 ms per second:
    # stamps fit +2004 ppm even though no converter samples are missing.
    io = kestrel_connect.AudioVaraIO('out', 'selected')
    for n in range(0, 5*FS, BLOCK):
        io._on_rx(np.zeros((BLOCK, 1), np.float32), BLOCK,
                  _Times(100 + .998*n/FS), _Status())
    ppm, _ = rates.rate_fit(io._fit, FS)
    assert ppm == pytest.approx(2004.008, abs=.01)
    report = io.clock_report()
    if host == 'Windows WASAPI':
        assert 'uncertain stamp estimate +2004.01 ppm' in report
        assert 'vs the system clock' not in report
        assert clock_shortfall.CLOCK.search(report) is None
    else:
        assert '+2004.01 ppm vs the system clock' in report
