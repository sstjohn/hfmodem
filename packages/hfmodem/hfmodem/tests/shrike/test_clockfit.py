# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The session's clock fit reads the crystal, not the scheduler.

`_LiveInput.clock_ppm` is the number `clock_report` prints at the end of every
on-air session, and `-32.38 ppm (fit sigma 17.12)` was printed by it against a
crystal two clean long fits place at +2.4 +/- 7.8 ppm. The instrument was
anchoring its fit on CALLBACK-ENTRY time -- the clock the callback's own comment
says carries 3 ms rms delivery jitter -- while the converter's jitter-free
timestamps were computed two lines above and went unused. Simulated at the
measured jitter, entry anchoring reproduces the observed sigma exactly; the
converter anchor holds +/-1.5 ppm at 50 s and is load-independent.

So the scenario here is the failure's own: a stream whose hardware runs a known
few ppm fast, delivered to Python late by |N(0, 3 ms)| every callback. The fit
must read the planted crystal and not the delivery.

Run: pytest hfmodem/tests/shrike/test_clockfit.py
"""
from __future__ import annotations

import sys

import numpy as np

from hfmodem.core import rates
from hfmodem.shrike import onair

FS = onair.FS
PPM_TRUE = 2.4
BLOCK = 128


class _Times:
    def __init__(self, current: float, adc: float):
        self.currentTime = current
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = adc + 1152 / FS


class _Status:
    input_overflow = input_underflow = output_underflow = False
    priming_output = False


class _FakeStream:
    """Enough of sounddevice.InputStream to bring `_LiveInput` up.

    `start` fires one callback so the constructor's delivered-nothing gate
    passes; every later block is driven by the test, with chosen timestamps.
    """

    blocksize = BLOCK
    latency = 0.01

    def __init__(self, callback):
        self.callback = callback

    def start(self):
        ind = np.zeros((BLOCK, 1), np.float32)
        self.callback(ind, BLOCK, _Times(0.0, 0.0), _Status())

    def stop(self):
        pass

    def close(self):
        pass


def _live(monkeypatch) -> onair._LiveInput:
    class _FakeSd:
        @staticmethod
        def InputStream(callback=None, **_):
            return _FakeStream(callback)

    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd)
    return onair._LiveInput("fake", None)


def test_the_fit_is_anchored_on_the_converter_not_on_delivery(monkeypatch):
    # The fake clock goes in before the stream comes up, so the startup
    # callback and the driven ones share one time base.
    entry = {"now": 0.0}
    monkeypatch.setattr(onair.time, "monotonic", lambda: entry["now"])
    live = _live(monkeypatch)
    rng = np.random.default_rng(7)
    ind = np.zeros((BLOCK, 1), np.float32)
    stream = live._stream
    n = live.samples
    while n < 50 * FS:                                 # 50 s of stream
        adc = n / (FS * (1 + PPM_TRUE * 1e-6))         # the hardware instant
        # Delivery trails the hardware; `currentTime` and `monotonic` are read
        # at the same (late) instant, which is what makes their difference the
        # clean base the callback relies on.
        entry["now"] = adc + abs(rng.normal(0.0, 0.003))
        stream.callback(ind, BLOCK, _Times(entry["now"], adc), _Status())
        n = live.samples
    ppm, sigma = rates.rate_fit(live._fit, FS)
    live.close()
    assert abs(ppm - PPM_TRUE) < 0.1, f"{ppm=:+.2f} against {PPM_TRUE} planted"
    assert sigma < 0.1, f"{sigma=}"


def test_the_report_no_longer_launders_the_sigma(monkeypatch):
    """The parenthetical "repeat runs scatter ~1 ppm, which is the real
    uncertainty" was measured on >=25-minute idle runs of the jittered
    instrument, and it is why nobody questioned -32 ppm. It is gone."""
    entry = {"now": 0.0}
    monkeypatch.setattr(onair.time, "monotonic", lambda: entry["now"])
    live = _live(monkeypatch)
    stream, ind = live._stream, np.zeros((BLOCK, 1), np.float32)
    n = live.samples
    while n < 35 * FS:
        adc = entry["now"] = n / FS
        stream.callback(ind, BLOCK, _Times(adc, adc), _Status())
        n = live.samples
    report = live.clock_report()
    live.close()
    assert "fit sigma" in report
    assert "repeat runs scatter" not in report
    assert "real uncertainty" not in report


class _Now(dict):
    """`time.monotonic` and the handle the test steers it with, in one object."""

    def __call__(self) -> float:
        return self["now"]


def _live_late(monkeypatch) -> tuple[onair._LiveInput, _Now]:
    now = _Now(now=0.0)
    monkeypatch.setattr(onair.time, "monotonic", now)
    return _live(monkeypatch), now


def _drive(live, now, *, seconds: float, drop_every: int = 0,
           entry_ramp_s: float = 0.0) -> None:
    """Feed the stream `seconds` of converter time, block by block.

    `drop_every`: the Python callback is not invoked for one block in this many,
    while the converter's timestamps march on regardless -- what a starved
    interpreter does to a real stream, with no status flag behind it.
    `entry_ramp_s`: how late Python is to the callback by the end, against the
    IOProc instant `currentTime` was read at.
    """
    ind = np.zeros((BLOCK, 1), np.float32)
    stream, k, made = live._stream, 0, live.samples
    while made < seconds * FS:
        adc = made / (FS * (1 + PPM_TRUE * 1e-6))
        made += BLOCK
        k += 1
        if drop_every and k % drop_every == 0:
            continue
        cur = adc + 0.011                          # the IOProc's own instant
        now["now"] = cur + entry_ramp_s * adc / seconds
        stream.callback(ind, BLOCK, _Times(cur, adc), _Status())


def test_a_starved_stream_reads_as_lost_capture_and_not_as_a_crystal(monkeypatch):
    """The -4860.19 ppm of the 2026-08-19 daytime arms.

    Measured on this station: eight pure-Python threads starve the callback and
    the card delivers 567040 samples in 90 s of stream -- 87% of the capture
    never reaching Python -- while `status.input_overflow` stays false for every
    one of the 4431 callbacks that do run. The fit sees only that the sample
    count fell behind the converter's own timeline and calls the shortfall a
    crystal. 0.49% of a 60 s arm is 293 ms of audio missing from the middle of
    the recording every window in that session indexes into, and a report that
    says `0 xruns` beside it reads as a clean stream.
    """
    live, now = _live_late(monkeypatch)
    _drive(live, now, seconds=60.0, drop_every=205)      # 0.49% of the blocks
    report = live.clock_report()
    lost = live.lost
    live.close()
    assert live.xruns == 0, "the driver raised the flag this test needs it to miss"
    assert abs(lost / FS - 0.293) < 0.02, f"{lost} samples short"
    assert "SPLICED" in report, report


def test_the_fit_anchor_does_not_carry_the_delivery_delay(monkeypatch):
    """`currentTime` is the IOProc's instant, not Python's.

    The callback formed its anchor as `inputBufferAdcTime + (monotonic() -
    currentTime)` -- the converter's instant plus however late Python was to the
    callback. Measured idle on this station that term is 90 us and harmless;
    under a contended interpreter it reaches half a second. It has no business
    in a rate fit either way, and the converter's own timestamp is the anchor
    the docstring promises.
    """
    live, now = _live_late(monkeypatch)
    _drive(live, now, seconds=60.0, entry_ramp_s=0.3)
    ppm = live.clock_ppm()
    live.close()
    assert abs(ppm - PPM_TRUE) < 0.1, f"{ppm=:+.2f} against {PPM_TRUE} planted"
