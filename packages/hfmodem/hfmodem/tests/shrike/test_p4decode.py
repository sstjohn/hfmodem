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


def test_bcjr_parity_extrinsic_matches_all_short_paths():
    systematic = np.array([.4, -1.2, .7, .3])
    parity = np.array([-.5, .2, 1.1, -.8])
    prior = np.array([.3, -.1, -.6, .4])
    sums = np.zeros((4, 2))
    for start in product((0, 1), repeat=3):
        for bits in product((0, 1), repeat=4):
            b = np.array(bits)
            p = _parity(bits, start)
            weight = np.exp(.5*np.dot(1-2*b, systematic+prior)
                            + .5*np.dot(1-2*p, parity))
            sums[np.arange(4), p] += weight
    decoded = p4decode.component_decode(systematic, parity, prior, parity_output=True)
    np.testing.assert_allclose(decoded.parity_extrinsic,
                               np.log(sums[:, 0]/sums[:, 1])-parity, atol=2e-12)


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


def test_flat_recurrence_mapping_and_workspace_aliases():
    expected = np.empty((2, 8, 2), dtype=int)
    for state in range(8):
        a, b, c = (state >> 2) & 1, (state >> 1) & 1, state & 1
        for branch in range(2):
            expected[0, state, branch] = (b << 2) | (c << 1) | branch
            expected[1, state, branch] = 8 + (((branch ^ b ^ c) << 2) | (a << 1) | b)
    np.testing.assert_array_equal(p4decode._FLAT_NEIGHBORS, expected)
    assert expected.min() == 0 and expected.max() == 15
    metrics = np.empty((4, 2, 8))
    flat_metrics = metrics.reshape(4, 16)
    workspace = np.empty((2, 8, 2), dtype=np.float64)
    branch0, branch1 = workspace[:, :, 0], workspace[:, :, 1]
    edges = np.arange(96, dtype=float).reshape(3, 2, 8, 2) / -17
    metrics[0] = np.arange(16).reshape(2, 8) / -13
    reference = metrics[0].copy()
    assert np.shares_memory(metrics, flat_metrics)
    assert not np.shares_memory(workspace, metrics)
    assert not np.shares_memory(workspace, edges)
    assert np.shares_memory(branch0, workspace) and np.shares_memory(branch1, workspace)
    assert not np.shares_memory(branch0, branch1)
    for t in range(3):
        assert not np.shares_memory(metrics[t], metrics[t + 1])
        original = reference[p4decode._DIRECTIONS, p4decode._NEIGHBORS] + edges[t]
        reference = np.logaddexp(original[:, :, 0], original[:, :, 1])
        reference -= np.logaddexp.reduce(reference, axis=1)[:, None]
        flat_metrics[t].take(p4decode._FLAT_NEIGHBORS, out=workspace, mode='clip')
        np.add(workspace, edges[t], out=workspace)
        np.testing.assert_array_equal(branch0, original[:, :, 0])
        np.testing.assert_array_equal(branch1, original[:, :, 1])
        row = metrics[t + 1]
        np.logaddexp(branch0, branch1, out=row)
        row -= np.logaddexp.reduce(row, axis=1)[:, None]
        np.testing.assert_array_equal(row, reference)
