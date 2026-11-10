# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A converter stamp that runs ahead of delivery does not hold the last wait past the key.

PortAudio's WASAPI stream stamps a block's ADC instant AFTER the callback that
delivers it. Measured on the FT-891 codec, 2026-11-05: the host hands over
three or four 128-frame blocks every 10 ms, and their `inputBufferAdcTime`
runs 8.7 to 16.7 ms past `currentTime`. `sample_now()` then trails the
delivered count by 450 to 930 samples, so `wait_until(key - lead)` slept until
the delivered count was that far past the read deadline, and the admission
clamp, taken on the delivered count, found the key +1.9 to +6.9 ms late. With
5 ms forgiven, the slot grid held or fell by whichever side of it the stream
opened on; recording `everything` lengthened the callbacks and tipped it over.

A block cannot be captured after it is delivered, and that bound is all the
wait needs.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_clockfit import BLOCK, _FakeStream, _Now, _Status, _Times

FS = onair.FS
PERIOD_S = 0.010          # the WASAPI shared-mode event
AHEAD_S = 0.00867         # first block's ADC stamp past `currentTime`, measured
SETTLE_N = round(0.040 * FS)


class _Wasapi:
    """Delivers the stream as the box's WASAPI does: whole blocks, one event
    every 10 ms, each block stamped ahead of the event that hands it over."""

    def __init__(self, monkeypatch, entry_s: float = 0.0):
        self.clock = _Now(now=0.0)
        monkeypatch.setattr(onair.time, "monotonic", self.clock)

        class _FakeSd:
            @staticmethod
            def InputStream(callback=None, **_):
                return _FakeStream(callback)

        monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd)
        # The rig arm's own clocks: DAC notice 2624, so the notice and not the
        # settle sets the read deadline.
        self.live = onair._LiveInput("fake", None, tx_latency_n=960, geometry_n=960)
        self.live._lat = 2240
        self.live._sleep_until = self._sleep_until
        self.entry_s = entry_s
        self.event = 0
        self.ind = np.zeros((BLOCK, 1), np.float32)

    def _fire(self):
        self.event += 1
        cur = self.event * PERIOD_S
        due = int(cur * FS) // BLOCK * BLOCK
        k = 0
        while self.live.samples < due:
            # Python reaches each block's callback `entry_s` after the last.
            self.clock["now"] = cur + k * self.entry_s
            self.live._stream.callback(self.ind, BLOCK,
                                       _Times(cur, cur + AHEAD_S + k * BLOCK / FS), _Status())
            k += 1
        self.clock["now"] = cur + k * self.entry_s

    def take_until(self, index):
        while self.live.samples < index:
            self._fire()

    def _sleep_until(self, t):
        while (self.event + 1) * PERIOD_S <= t:
            self._fire()
        self.clock["now"] = max(self.clock["now"], t)


@pytest.mark.parametrize("entry_s", [0.0, 0.0005], ids=["idle", "long-callbacks"])
def test_the_last_wait_leaves_every_slot_phase_keyable(monkeypatch, entry_s):
    stream = _Wasapi(monkeypatch, entry_s)
    live = stream.live
    stream.take_until(FS)
    late = []
    for residue in range(0, 4 * BLOCK, 7):
        key = live.samples + FS // 2 + residue
        lead = onair._prekey_lead(live, SETTLE_N)
        stream.take_until(key - lead)
        live.wait_until(key - lead)
        late.append(live.clamp_late(key))
    live.close()
    assert max(late) == 0, f"{sum(1 for x in late if x)} of {len(late)} phases late, worst +{max(late) / FS * 1e3:.1f} ms"


def test_the_converter_position_never_trails_what_has_been_delivered(monkeypatch):
    stream = _Wasapi(monkeypatch)
    stream.take_until(FS)
    live = stream.live
    assert live.sample_now() >= live.samples
    live.close()
