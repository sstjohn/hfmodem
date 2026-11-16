# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ALSA gets a deeper capture ring at the same period, and nothing else moves.

`PortAudio` answers stream latency with pa_linux_alsa.c's arithmetic (the same
in 19.6 and 19.7) over a device that takes whole periods, as snd-usb-audio does.
Its "low" is 384 frames at the 44.1 kHz PortAudio probes first, which at 48 kHz
opens the 256-frame period and 512-frame input latency shackbook logged as
`holdback 640`.
"""
from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace

import pytest

from hfmodem.core import audio
from hfmodem.core.audio import StationAudio
from hfmodem.tests.core.test_audio import FakeStream

FS = audio.CARD_RATE_HZ
LOW = 384 / 44100


def _period(latency: float, periods: int, blk: int = 128) -> int:
    host = (blk + max(blk, int(latency * FS))) // periods
    return -(-host // blk) * blk


def _ring(latency: float, period: int) -> int:
    return round((latency * FS + period) / period) * period


class PortAudio:
    """sounddevice on one host API, opening streams as PortAudio sizes them."""

    _libname = "libportaudio.so.2"

    def __init__(self, monkeypatch, host="ALSA", low=LOW, stream=FakeStream,
                 playback_periods=1024):
        self.host, self.low, self.stream = host, low, stream
        self.playback_periods = playback_periods
        self.periods, self.opened, self.closed = 4, [], []
        monkeypatch.setitem(sys.modules, "sounddevice", self)
        monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace(
            PaAlsa_SetNumPeriods=lambda n: setattr(self, "periods", n)))

    def query_devices(self, device, kind=None):
        return {"hostapi": 2, "default_low_input_latency": self.low}

    def query_hostapis(self, index):
        return {"name": self.host}

    def open(self, latency, callback=None, blocksize=128):
        lin, lout = latency if isinstance(latency, tuple) else (latency, latency)
        lin, lout = (self.low if x == "low" else x for x in (lin, lout))
        # Fixed-block duplex sizes the period off playback, at its own latency,
        # when playback cannot take the periods asked for.
        period = (_period(lout, self.playback_periods)
                  if self.periods > self.playback_periods else _period(lin, self.periods))
        s = self.stream(callback, blocksize=blocksize,
                        latency=((_ring(lin, period) - period) / FS,
                                 (_ring(lout, period) - period) / FS))
        s.close = lambda: self.closed.append(s)
        self.opened.append(SimpleNamespace(request=latency, periods=self.periods,
                                           period=period, ring=_ring(lin, period), stream=s))
        return s

    def Stream(self, *, latency, callback=None, blocksize=128, **_):
        return self.open(latency, callback, blocksize)


@pytest.mark.parametrize("low", [LOW, 512 / FS, 384 / FS, 0.0087, 0.020, -1.0])
def test_alsa_deepens_the_ring_and_keeps_the_period(monkeypatch, low):
    sd = PortAudio(monkeypatch, low=low)
    stream, delivery = audio.open_stream(sd.open, "USB AUDIO  CODEC", 128)
    shallow, deep = sd.opened
    assert (shallow.request, shallow.periods) == ("low", 4)
    assert sd.closed == [shallow.stream] and stream is deep.stream
    assert deep.request[1] == "low" and sd.periods == 4
    assert deep.period == shallow.period
    assert stream.latency[0] >= audio.ALSA_CAPTURE_S - deep.period / FS / 2
    assert stream.latency[1] == shallow.stream.latency[1]
    assert delivery == shallow.stream.latency[0]


def test_shackbook_grant():
    """What the FTDX10's codec should grant: 42.7 ms in, 10.7 ms out as before."""
    period = _period(LOW, 4)
    deep = (4 * (128 + int(LOW * FS)) - 128 + 0.5) / FS
    assert (period, _period(deep, 16)) == (256, 256)
    assert (_ring(LOW, period) - period, _ring(deep, period) - period) == (512, 2048)
    assert _ring(deep, period) == 9 * period


@pytest.mark.parametrize("host", ["Core Audio", "Windows WASAPI", None])
def test_other_hosts_open_once_on_low(monkeypatch, host):
    sd = PortAudio(monkeypatch, host=host)
    if host is None:
        sd.query_devices = lambda device, kind=None: (_ for _ in ()).throw(OSError())
    stream, delivery = audio.open_stream(sd.open, "dev", 128)
    assert [(o.request, o.periods) for o in sd.opened] == [("low", 4)]
    assert not sd.closed and delivery == stream.latency[0]


def test_alsa_restores_the_periods_when_the_open_fails(monkeypatch):
    sd = PortAudio(monkeypatch)
    opened = sd.open

    def refuse(latency):
        if isinstance(latency, tuple):
            raise OSError("device busy")
        return opened(latency)

    with pytest.raises(OSError):
        audio.open_stream(refuse, "dev", 128)
    assert sd.periods == 4


def test_alsa_keeps_low_where_the_playback_would_move(monkeypatch):
    sd = PortAudio(monkeypatch, playback_periods=8)
    stream, delivery = audio.open_stream(sd.open, "dev", 128)
    assert [(o.request == "low", o.periods) for o in sd.opened] == [
        (True, 4), (False, 16), (True, 4)]
    assert sd.closed == [o.stream for o in sd.opened[:2]] and stream is sd.opened[2].stream
    assert stream.latency == sd.opened[0].stream.latency and sd.periods == 4


class _Starting(FakeStream):
    def start(self):
        self.run(audio._LAT_MEDIAN_N)


@pytest.mark.parametrize("host", ["ALSA", "Core Audio"])
def test_station_holdback_is_the_low_streams(monkeypatch, host):
    sd = PortAudio(monkeypatch, host=host, stream=_Starting)
    sa = StationAudio(input_device="in", output_device="out")
    sa.open()
    try:
        assert sa.holdback == 640
        assert round(sa._stream.latency[0] * FS) == (2048 if host == "ALSA" else 512)
        assert [o.request for o in sd.opened][-1] == (
            ((4 * (128 + int(LOW * FS)) - 128 + 0.5) / FS, "low") if host == "ALSA" else "low")
        assert sa.lat == 1152
    finally:
        sa.close()
