# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A deeper ALSA capture ring moves no transmit timing in `_LiveInput`.

The same FTDX10 geometry opened on Core Audio and on ALSA, where `open_stream`
deepens the input latency from 512 to 2048 frames: the holdback, the ADC->DAC
offset, every figure in the clock line, and the PTT and emission indices of a
keyed burst must come out identical.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.tests.core.test_alsa_capture_ring import PortAudio

STATUS = SimpleNamespace(input_overflow=False, input_underflow=False,
                         output_underflow=False, priming_output=False)


def _live(monkeypatch, host):
    clock = SimpleNamespace(now=0.0)

    class Device:
        def __init__(self, callback, *, blocksize, latency):
            self.callback, self.blocksize, self.latency = callback, blocksize, latency
            self.n = 0

        def pump(self):
            out = np.zeros((128, 1), np.float32)
            adc = self.n / 48000
            self.callback(np.zeros_like(out), out, 128,
                          SimpleNamespace(inputBufferAdcTime=adc,
                                          outputBufferDacTime=adc + 720 / 48000,
                                          currentTime=clock.now), STATUS)
            self.n += 128

        def start(self):
            self.pump()

    sd = PortAudio(monkeypatch, host=host, stream=Device)
    monkeypatch.setattr(onair, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(onair.threading, "Thread",
                        lambda **kw: SimpleNamespace(start=lambda: None))
    live = onair._LiveInput("USB AUDIO  CODEC", "USB AUDIO  CODEC", tx_latency_n=336)
    live._breathing.close()

    class Done:
        def clear(self):
            pass

        def set(self):
            pass

        def wait(self, limit):
            while live._tx is not None:
                live._stream.pump()
            return True

    live._tx_done = Done()
    live._sleep_until = lambda t: setattr(clock, "now", t)
    live.flush_to = lambda n: setattr(live, "pos", n)
    live.wait_until = lambda n: 0.0
    return live, clock, sd


def _trace(live, clock):
    keys = []
    first, end = live.transmit(
        np.full(46080, .1, np.float32), at=120000, settle=.04,
        key=lambda state: keys.append((state, round(clock.now * 48000))))
    return live._tx_at, first, end, keys


@pytest.fixture
def pair(monkeypatch, capsys):
    with monkeypatch.context() as m:
        core, core_clock, _ = _live(m, "Core Audio")
        expected = (core.holdback, core._lat, onair._clock_line(core, 1920),
                    _trace(core, core_clock))
    capsys.readouterr()
    alsa, alsa_clock, sd = _live(monkeypatch, "ALSA")
    sd.printed = capsys.readouterr().out
    return expected, alsa, alsa_clock, sd


def test_alsa_ring_is_deeper(pair):
    _, alsa, _, sd = pair
    assert [round(o.stream.latency[0] * 48000) for o in sd.opened] == [512, 2048]
    assert alsa._stream is sd.opened[-1].stream
    assert "capture ring: input latency 42.7 ms, 10.7 ms on \"low\"" in sd.printed


def test_alsa_ring_moves_no_timing(pair):
    expected, alsa, clock, _ = pair
    assert expected[0] == 640
    assert (alsa.holdback, alsa._lat, onair._clock_line(alsa, 1920),
            _trace(alsa, clock)) == expected
