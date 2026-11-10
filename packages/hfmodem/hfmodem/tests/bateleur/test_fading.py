# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.bateleur.sim import _fading
from hfmodem.bateleur.spec import FS

SPREADS = (0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0)


def tap_statistics(spread, trials=1024):
    lag = round(FS / (np.pi * spread))
    samples = []
    pairs = []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        g = _fading(FS, spread, rng)
        samples.append(g[::120])
        long = _fading(2 * lag + 1, spread, rng)
        pairs.append(long[[0, lag, 2 * lag]])
    samples = np.concatenate(samples)
    pairs = np.asarray(pairs)
    power = np.mean(np.abs(samples) ** 2)
    return dict(mean_abs=float(np.mean(np.abs(samples))), power=float(power),
                below_03=float(np.mean(np.abs(samples) < 0.3)),
                mean_real=float(samples.real.mean()), mean_imag=float(samples.imag.mean()),
                covariance=[complex(np.mean(pairs[:, 0].conj() * pairs[:, i]))
                            for i in (1, 2)])


@pytest.mark.parametrize('spread', SPREADS)
def test_short_bursts_keep_rayleigh_amplitudes_and_gaussian_doppler(spread):
    stats = tap_statistics(spread)
    assert stats['mean_abs'] == pytest.approx(np.sqrt(np.pi) / 2, abs=0.055)
    assert stats['power'] == pytest.approx(1, abs=0.10)
    assert stats['below_03'] == pytest.approx(1 - np.exp(-0.09), abs=0.025)
    assert abs(stats['mean_real']) < 0.075
    assert abs(stats['mean_imag']) < 0.075
    for multiple, covariance in enumerate(stats['covariance'], 1):
        assert covariance.real == pytest.approx(np.exp(-multiple ** 2 / 2), abs=0.10)
        assert abs(covariance.imag) < 0.10


def test_fading_repeats_by_seed_and_paths_are_independent():
    rng = np.random.default_rng(91)
    first = _fading(FS, 0.1, rng)
    second = _fading(FS, 0.1, rng)
    np.testing.assert_array_equal(first, _fading(FS, 0.1, np.random.default_rng(91)))
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(_fading(17, 0, rng), np.ones(17))
