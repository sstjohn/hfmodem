# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`test_capture_loss`'s bench with the station's own callback in the loop.

The device there is a pipe and a thread; here the thread drives
`StationAudio._callback` with the ticker's stamps as the converter's, so what is
measured is the station's capture path -- the copy, the lane push, the `ahead`
accounting -- under the same listening-cycle load, and `StationAudio.lost` is
read beside the device's own count of blocks it had to overwrite.

`breathe()` is the station's, claimed on the thread that runs the readers, and
the session under it wraps nothing of its own: the claim is that a breather held
by the capture covers every reader the decode thread runs, including the ones
that wrap none.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from hfmodem.core import audio, gil
from hfmodem.core.audio import StationAudio, StreamLane
from hfmodem.tests.core.test_capture_loss import (
    BLOCK, CYCLES, RING_S, _TICKER, _Session, _per_cycle)


class _Times:
    def __init__(self, adc):
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = adc + 1152 / audio.CARD_RATE_HZ


class _Status:
    input_overflow = output_underflow = priming_output = False

    def __bool__(self):
        return False


class _Device:
    """sounddevice.Stream, as a ticker process and a callback thread.

    A block whose stamp is older than the ring when the thread gets to it is a
    block the converter overwrote: the callback is not called for it, and the
    next stamp it does see steps ahead of the sample count -- which is exactly
    what `rates.lost_step` reads off a real card.
    """

    blocksize = BLOCK
    latency = (RING_S, RING_S)
    active = True

    def __init__(self, callback=None, **_):
        self.cb = callback
        self.overwritten = 0
        self.worst = 0.0
        self.counting = False
        self._stop = threading.Event()
        self._proc = subprocess.Popen([sys.executable, "-c", _TICKER],
                                      stdout=subprocess.PIPE)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._proc.kill()
        self._proc.stdout.close()

    def close(self):
        pass

    def _run(self):
        indata = np.zeros((BLOCK, 1), np.float32)
        outdata = np.zeros((BLOCK, 1), np.float32)
        out = self._proc.stdout
        while not self._stop.is_set():
            b = out.read(8)
            if len(b) < 8:
                break
            stamp = struct.unpack("d", b)[0]
            lag = time.monotonic() - stamp
            if self.counting:
                self.worst = max(self.worst, lag)
            if lag > RING_S:
                self.overwritten += self.counting
                continue
            self.cb(indata, outdata, BLOCK, _Times(stamp), _Status())


def _station(monkeypatch) -> tuple[StationAudio, _Device]:
    made = {}

    class FakeSd:
        @staticmethod
        def Stream(**kw):
            made["d"] = _Device(**kw)
            return made["d"]

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSd)
    sa = StationAudio(input_device="ticker", output_device="ticker")
    sa.open()
    return sa, made["d"]


def _arm(monkeypatch, *, breathe: bool) -> tuple[int, int, float]:
    """(blocks the device overwrote, `StationAudio.lost`, worst stall)."""
    sa, device = _station(monkeypatch)
    lane = sa.subscribe(StreamLane(audio.CARD_RATE_HZ))
    session = _Session(breathe=False)
    try:
        if breathe:
            sa.breathe()
        session.listening()
        lane.poll()
        time.sleep(0.3)
        device.counting, base = True, sa.lost
        for _ in range(CYCLES):
            session.listening()
            lane.poll()
        device.counting = False
        return device.overwritten * BLOCK, sa.lost - base, device.worst
    finally:
        sa.close()


@pytest.mark.realtime
def test_the_station_loses_capture_under_the_listening_cycle_and_counts_it(monkeypatch):
    overwritten, counted, worst = _arm(monkeypatch, breathe=False)
    assert _per_cycle(overwritten) > 150, (
        f"the unbreathed station lost {_per_cycle(overwritten):.0f} samples a "
        "cycle, and the air's listening cycles lose 400-1400")
    assert counted == pytest.approx(overwritten, rel=0.05), (
        f"the device overwrote {overwritten} samples and `StationAudio.lost` "
        f"read {counted}")
    assert worst > 4 * gil.PERIOD


@pytest.mark.realtime
def test_breathing_from_the_polling_thread_takes_the_stations_loss_to_zero(monkeypatch):
    overwritten, counted, worst = _arm(monkeypatch, breathe=True)
    assert (overwritten, counted) == (0, 0), (
        f"{overwritten} samples went with the station breathing "
        f"({counted} off `lost`)")
    assert worst < 5 * gil.PERIOD, (
        f"the callback was kept out {1e3 * worst:.1f} ms against a "
        f"{1e3 * gil.PERIOD:.0f} ms period")


def test_close_releases_the_breather(monkeypatch):
    """The window is process-wide and one at a time: a station that closed and
    left it claimed would leave the next stream decoding unbreathed."""
    sa, _ = _station(monkeypatch)
    sa.breathe()
    assert gil._tool is not None
    sa.close()
    assert gil._tool is None
