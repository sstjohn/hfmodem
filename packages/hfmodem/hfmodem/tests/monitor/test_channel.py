# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A track cut out of its window and moved home: where it lands, what it keeps
out, and that sample i out is still sample i in."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.monitor.channel import Channel

FS = 48000


def _through(chan: Channel, x: np.ndarray, chunk: int) -> np.ndarray:
    return np.concatenate([chan.push(x[i:i + chunk]) for i in range(0, len(x), chunk)]
                          + [chan.flush()])


def _level_db(y: np.ndarray, hz: float) -> float:
    w = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * w)) / w.sum() * 2
    f = np.fft.rfftfreq(len(y), 1 / FS)
    return float(20 * np.log10(spec[np.abs(f - hz) < 3].max() + 1e-12))


@pytest.mark.parametrize("chunk", [4800, 4801, 333])
def test_the_track_lands_on_the_nominal_centre_and_its_neighbour_does_not(chunk):
    t = np.arange(2 * FS) / FS
    x = np.sin(2 * np.pi * 8237.0 * t) + np.sin(2 * np.pi * 7537.0 * t)
    y = _through(Channel(8200.0, 1500.0, 300.0), x, chunk)
    assert len(y) == len(x)
    assert _level_db(y, 1537.0) == pytest.approx(0.0, abs=0.1)
    assert _level_db(y, 837.0) < -70


def test_the_delay_is_taken_out():
    x = np.zeros(FS)
    x[20000:24800] = np.sin(2 * np.pi * 2200.0 * np.arange(4800) / FS)
    y = _through(Channel(2200.0, 1500.0, 400.0), x, 4800)
    on = np.flatnonzero(np.abs(y) > 0.5)
    assert abs(on[0] - 20000) <= 30 and abs(on[-1] - 24799) <= 30


def test_an_offset_is_read_from_where_the_track_sits():
    chan = Channel(2200.0, 1500.0, 400.0)
    assert chan.offset(-46.9) == 653.1
    assert chan.offset(None) is None


def test_the_passband_never_folds_through_zero():
    assert Channel(400.0, 1500.0, 1400.0).half_hz == 340.0
    assert Channel(1500.0, 1500.0, 1400.0).half_hz == 1400.0
