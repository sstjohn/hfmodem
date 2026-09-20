# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The station's own transmit latency, on the index every burst is keyed at.

`_LiveInput` schedules a burst so that output frame `at - _lat` reaches the DAC
as capture index `at` reaches the ADC, where `_lat` is the converter's own
`outputBufferDacTime - inputBufferAdcTime`. That is not the whole path: the
codec's in-and-out delay is not in the driver's number, and what is left over
went out on every burst this station has ever keyed.

`arm-v23-A-40-ws8eoc` measured the residue twice on one arm and got the same
answer both ways -- 19.8 ms from where WS8EOC's PACTOR-1 answer sat against our
960 ms packet, and about 20 ms from what the PACTOR-3 witness chain needs to
close against an SCS pair's 894.1 ms. So it is a property of the audio path,
spent on every emission of every protocol, and it is carried here rather than
inside a protocol constant: `P3_REPLY_S = 0.890` is measured off SCS traffic and
is right.

What this file holds is the arithmetic, on a stream with no device under it.
Three claims: the audio moves by exactly the correction and by nothing without
it; the key moves with the audio, so the PTT lead the session reports is the
same number either way; and the budget between the cycle's last read and the
admission check -- the tick, the render and the drain -- is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.shrike import onair

FS = onair.FS
# `arm-v23-A-40-ws8eoc`'s own clocks line: block 128, ADC->DAC 1268 samples,
# DAC notice 1652, holdback 660 -- and the 40 ms settle the FT-891 is set for.
BLK, LAT, SETTLE_N = 128, 1268, round(0.04 * FS)
#: What the v24 arm flies, in samples. 20 ms, measured two ways.
CORRECTION_N = round(0.020 * FS)


class _Done:
    """`threading.Event` for a burst that is over before it is armed."""

    def clear(self) -> None:
        pass

    def wait(self, _timeout: float) -> bool:
        return True


class _Stream(onair._LiveInput):
    """`_LiveInput`'s transmit arithmetic with no card and no clock.

    Every sleep is recorded instead of taken, so the instants are the
    measurement rather than a race: `slept[0]` is where PTT would be asserted.
    """

    def __init__(self, *, tx_latency_n: int = 0) -> None:
        self.fs = FS
        self._duplex = True
        self._blk = BLK
        self._lat = LAT
        self.tx_latency_n = tx_latency_n
        self.samples = 100 * BLK
        self.holdback = 660
        # A fixed epoch, so two streams are comparable to the sample.
        self._dac = (0.0, 0)
        self._last = (0.0, self.samples)
        self._tx = None
        self._tx_at = 0
        self._tx_end_time = 0.0
        self._tx_done = _Done()
        self.keyed_s = 0.0
        self.keyed_at = None
        self.slept: list[float] = []

    def _sleep_until(self, t: float) -> None:
        self.slept.append(t)


def _key(stream: _Stream, at: int) -> tuple[int, float]:
    """(output frame the burst starts on, wall clock PTT goes up) for `at`."""
    stream.slept.clear()
    stream.transmit(np.zeros(round(0.21 * FS), np.float32), at=at,
                    settle=SETTLE_N / FS)
    return stream._tx_at, stream.slept[0]


def test_the_correction_moves_the_audio_and_nothing_moves_without_it():
    at = 400 * BLK
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    assert _key(plain, at)[0] == at - LAT
    assert _key(corrected, at)[0] == at - LAT - CORRECTION_N


def test_the_burst_still_reports_the_capture_index_it_was_aimed_at():
    # `first` is what `tx_audio_start` becomes and what every peer interval is
    # measured against. The burst lands there either way -- the correction is
    # the whole loop's, so it moves the emission, not the bookkeeping.
    at = 400 * BLK
    audio = np.zeros(round(0.21 * FS), np.float32)
    for stream in (_Stream(), _Stream(tx_latency_n=CORRECTION_N)):
        first, end = stream.transmit(audio, at=at, settle=SETTLE_N / FS)
        assert first == at
        assert end == at + audio.size


def test_the_key_moves_with_the_audio_so_the_ptt_lead_is_untouched():
    # The arm reports PTT LEAD SHORT at 25 ms of the 40 it is set for. Moving
    # the audio 20 ms early without moving the key would spend the correction
    # out of what is left of that lead and cut the head off every burst.
    at = 400 * BLK
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    start_p, key_p = _key(plain, at)
    start_c, key_c = _key(corrected, at)
    lead_p = plain._dac_time(start_p) - key_p
    lead_c = corrected._dac_time(start_c) - key_c
    assert lead_p == pytest.approx(SETTLE_N / FS, abs=1e-9)
    assert lead_c == pytest.approx(lead_p, abs=1e-9)
    assert key_p - key_c == pytest.approx(CORRECTION_N / FS, abs=1e-9)


def test_the_enqueue_deadline_moves_by_exactly_the_correction():
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    assert corrected.key_notice - plain.key_notice == CORRECTION_N
    # An instant already inside the plain stream's notice, so both clamps speak.
    at = plain.samples + plain.key_notice - 300
    assert plain.clamp_late(at) == 300
    assert corrected.clamp_late(at) - plain.clamp_late(at) == CORRECTION_N
    # ...and a slot that no longer leaves the notice is refused rather than
    # slid into the peer's packet, at the new deadline.
    late = corrected.samples + corrected.key_notice - 1
    with pytest.raises(onair._MissedTxSlot):
        corrected.transmit(np.zeros(BLK, np.float32), at=late, settle=0.0)


def test_the_cycles_budget_between_the_last_read_and_the_key_is_invariant():
    # `_prekey_lead` is one number for the read deadline and for the drain, and
    # what separates it from `key_notice` is the whole budget for the tick, the
    # render and the drain. The correction moves both edges together.
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    lead_p = onair._prekey_lead(plain, SETTLE_N)
    lead_c = onair._prekey_lead(corrected, SETTLE_N)
    assert lead_c - lead_p == CORRECTION_N
    assert lead_c - corrected.key_notice == lead_p - plain.key_notice
    # And the session says so unconditionally, in the line every transmit
    # deadline in `onair` is built from.
    assert (f"+ {CORRECTION_N} samples (20.0 ms) of station transmit latency"
            in onair._clock_line(corrected, SETTLE_N))


def test_a_replay_takes_neither_term():
    # No DAC to give notice to and no clock to lose it on. The settle alone is
    # the deadline there, exactly as it was.
    replay = onair._ReplayInput.__new__(onair._ReplayInput)
    assert onair._prekey_lead(replay, SETTLE_N) == SETTLE_N
