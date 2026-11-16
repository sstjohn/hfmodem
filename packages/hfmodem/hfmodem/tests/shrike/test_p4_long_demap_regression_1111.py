# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The faster shared likelihood pass preserves every constellation bit class."""
import numpy as np
import pytest
from scipy.special import logsumexp

from hfmodem.shrike import p4longrates


@pytest.mark.parametrize('sl', range(6, 11))
@pytest.mark.parametrize('noise', [1e-6, .01, 1., 1000.])
def test_shared_exponentials_match_separately_normalized_bit_classes(sl, noise):
    rng = np.random.default_rng(1111 + sl)
    points = p4longrates.constellation(sl)
    z = points[rng.integers(len(points), size=4968)]
    z = z + .13*(rng.normal(size=len(z)) + 1j*rng.normal(size=len(z)))
    z[:8] = 0  # Includes phase/label ties rather than only decided symbols.
    metrics = -abs(z[:, None]-points)**2/noise
    labels = np.arange(len(points))
    width = p4longrates.geometry(sl).width
    reference = np.column_stack([
        logsumexp(metrics[:, ((labels >> bit) & 1) == 0], axis=1)
        - logsumexp(metrics[:, ((labels >> bit) & 1) == 1], axis=1)
        for bit in range(width-1, -1, -1)
    ])[p4longrates.symbol_permutation()].ravel()
    actual = p4longrates.demap(z, noise, sl)
    np.testing.assert_allclose(actual, reference, atol=2e-9, rtol=2e-12)
    np.testing.assert_array_equal(actual < -1e-9, reference < -1e-9)


@pytest.mark.parametrize('sl', range(7, 11))
def test_extreme_class_underflow_uses_finite_separately_shifted_sums(sl):
    z = np.full(4968, 10+7j)
    points = p4longrates.constellation(sl)
    metrics = -abs(z[:, None]-points)**2/1e-100
    labels = np.arange(len(points))
    width = p4longrates.geometry(sl).width
    reference = np.column_stack([
        logsumexp(metrics[:, ((labels >> bit) & 1) == 0], axis=1)
        - logsumexp(metrics[:, ((labels >> bit) & 1) == 1], axis=1)
        for bit in range(width-1, -1, -1)
    ])[p4longrates.symbol_permutation()].ravel()
    actual = p4longrates.demap(z, 1e-100, sl)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, reference)
