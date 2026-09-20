# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Independent short paths, exchange invariants and noisy offline codewords."""
from itertools import product

import numpy as np
import pytest

from hfmodem.shrike import p4decode


def _parity(bits, state):
    a, b, c = state
    result = []
    for bit in bits:
        u = int(bit) ^ b ^ c
        result.append(u ^ a ^ c)
        a, b, c = u, a, b
    return np.array(result)


def _enumerated(systematic, parity, prior):
    sums = np.zeros((len(systematic), 2))
    for start in product((0, 1), repeat=3):
        for bits in product((0, 1), repeat=len(systematic)):
            b = np.array(bits)
            p = _parity(bits, start)
            weight = np.exp(.5 * np.dot(1 - 2*b, systematic + prior)
                            + .5 * np.dot(1 - 2*p, parity))
            sums[np.arange(len(b)), b] += weight
    return np.log(sums[:, 0] / sums[:, 1])


def test_bcjr_matches_all_short_paths_and_nonzero_prior():
    rng = np.random.default_rng(963)
    for n in range(1, 7):
        systematic, parity, prior = rng.normal(size=(3, n))
        decoded = p4decode.component_decode(systematic, parity, prior)
        expected = _enumerated(systematic, parity, prior)
        np.testing.assert_allclose(decoded.posterior, expected, atol=2e-12)
        np.testing.assert_allclose(decoded.extrinsic, expected-systematic-prior, atol=2e-12)


def test_erased_parity_never_amplifies_systematic_information():
    systematic = np.linspace(-3, 3, 41)
    decoded = p4decode.turbo_decode(systematic, np.zeros(41), np.zeros(41),
                                     np.random.default_rng(8).permutation(41), iterations=12)
    np.testing.assert_allclose(decoded.posterior, systematic, atol=1e-12)
    assert len(decoded.diagnostics) == 12
    neutral = p4decode.turbo_decode(np.zeros(41), np.zeros(41), np.zeros(41), np.arange(41))
    np.testing.assert_array_equal(neutral.posterior, np.zeros(41))
    np.testing.assert_array_equal(neutral.bits, np.zeros(41))


def test_one_iteration_gather_scatter_and_extrinsic_only():
    rng = np.random.default_rng(62)
    systematic, parity1, parity2 = rng.normal(size=(3, 17))
    order = rng.permutation(17)
    assert not np.array_equal(order, np.argsort(order))
    first = p4decode.component_decode(systematic, parity1)
    second = p4decode.component_decode(systematic[order], parity2, first.extrinsic[order])
    expected = np.empty(17)
    expected[order] = second.posterior
    actual = p4decode.turbo_decode(systematic, parity1, parity2, order, iterations=1)
    np.testing.assert_array_equal(actual.posterior, expected)


def test_noisy_codeword_recovers_and_global_polarity_complements():
    rng = np.random.default_rng(924)
    bits = rng.integers(0, 2, 256)
    order = rng.permutation(len(bits))
    coded = np.array([bits, _parity(bits, (1, 0, 1)), _parity(bits[order], (0, 1, 1))])
    sigma = .7
    llr = 2*((1-2*coded) + sigma*rng.normal(size=coded.shape))/sigma**2
    channel_errors = np.count_nonzero((llr[0] < 0) != bits)
    decoded = p4decode.turbo_decode(*llr, order, iterations=8)
    assert channel_errors >= 10
    np.testing.assert_array_equal(decoded.bits, bits)
    complement = p4decode.turbo_decode(*(-llr), order, iterations=8)
    np.testing.assert_allclose(complement.posterior, -decoded.posterior, atol=1e-10)
    np.testing.assert_array_equal(complement.bits, 1-bits)


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[1.]], [1+1j]])
def test_invalid_component_inputs_rejected(bad):
    with pytest.raises(ValueError):
        p4decode.component_decode(bad, bad)


def test_shapes_and_prior_validation():
    with pytest.raises(ValueError):
        p4decode.component_decode([1, 2], [1])
    with pytest.raises(ValueError):
        p4decode.component_decode([1, 2], [1, 2], [np.nan, 0])
    with pytest.raises(ValueError):
        p4decode.turbo_decode([1, 2], [1], [1, 2], [0, 1])


@pytest.mark.parametrize("order", [[0, 0], [1, 2], [0., 1.], [False, True], [[0, 1]], [0]])
def test_invalid_permutations_rejected(order):
    with pytest.raises(ValueError):
        p4decode.turbo_decode([1, 2], [1, 2], [1, 2], order)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True, np.nan])
def test_invalid_iterations_rejected(iterations):
    with pytest.raises(ValueError):
        p4decode.turbo_decode([1, 2], [1, 2], [1, 2], [0, 1], iterations)
