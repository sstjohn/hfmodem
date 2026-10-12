# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.brubru import rx, tx
from hfmodem.tests.brubru.bench import Inconclusive, crossing, impair, sensitivity
from hfmodem.tests.brubru.vectors import frames_for


@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('ppm,drift', [(-200, 0), (-50, 0), (50, 0), (200, 0),
                                     (0, -0.5), (0, 0.5)])
def test_longest_class_tracks_clock_and_carrier(level, ppm, drift):
    frames = frames_for(825)
    burst, = tx.transmit(frames, level)
    audio = impair(burst.audio, snr=(7, 10)[level], seed=123, ppm=ppm, drift=drift)
    got = rx.receive(audio, centre=1500)
    assert [list(d.frames) for d in got] == [frames]


@pytest.mark.parametrize('duration,depth', [(1, 0.1), (0.3, 0)])
def test_midframe_fade_and_dropout(duration, depth):
    frames = frames_for(825)
    burst, = tx.transmit(frames)
    audio = impair(burst.audio, snr=10, seed=17, notch_s=duration, depth=depth)
    got = rx.receive(audio, centre=1500)
    assert [list(d.frames) for d in got] == [frames]


def test_short_frames_never_use_tracking(monkeypatch):
    burst, = tx.transmit(frames_for(15))
    def forbidden(*args):
        raise AssertionError('tracking used for a short frame')
    monkeypatch.setattr(rx, 'tracked_values', forbidden)
    assert rx.receive(burst.audio, centre=1500)


def test_crossing_interpolation_monotonicity_and_paired_seeds():
    assert crossing({-1: 8, -0.5: 12}) == pytest.approx(-0.75)
    with pytest.raises(Inconclusive):
        crossing({-1: 8, -0.5: 7, 0: 12})
    calls = []
    def measure(snr, seed):
        calls.append((snr, seed))
        return seed % 20 < round(np.clip((snr + 2) * 5, 0, 20))
    result = sensitivity(measure, lower=-4, upper=4)
    assert result['point'] == 0
    for snr in result['cells']:
        assert [seed for cell, seed in calls if cell == snr] == list(range(20))


def test_false_long_header_hold_is_bounded_and_releases():
    from hfmodem.brubru import waveform as w
    burst, = tx.transmit(frames_for(825))
    receiver = rx.Receiver()
    receiver.feed(burst.audio[:48 * w.SPS])
    assert receiver.held
    assert not receiver.feed(np.zeros(receiver.MAX_SAMPLES + receiver.WINDOW))
    assert not receiver.held and receiver.events
    assert receiver.buffered <= receiver.MAX_SAMPLES
