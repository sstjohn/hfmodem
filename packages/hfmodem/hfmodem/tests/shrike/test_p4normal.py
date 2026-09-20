# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Normal-mode primitive checks; synthetic results are not interoperability."""
import numpy as np
import pytest

from hfmodem.shrike import p4normal as normal
from hfmodem.shrike.p4rx import RRC_TAPS


def test_cazac_independent_invariants_and_construction():
    c = normal.CAZAC16
    np.testing.assert_array_equal([np.vdot(c, np.roll(c, k)) for k in range(16)], [16] + [0]*15)
    assert np.vdot(c, c.conj()) == 0
    literal = np.array([1, 1, -1j, 1, -1, 1j, -1j, -1j,
                        1, -1, -1j, -1, -1, -1j, -1j, 1j,
                        1, 1, -1j, 1, -1, 1j, -1j, -1j,
                        1, -1, -1j, -1, -1, -1j, -1j, 1j])
    np.testing.assert_array_equal(normal.training_sequence(1), literal)
    np.testing.assert_array_equal(normal.training_sequence(2), literal.conj())
    np.testing.assert_array_equal(normal.training_sequence(3), literal)


@pytest.mark.parametrize('variant,blocks,size', [('short',6,176), ('long',24,207), ('breakin',4,210)])
def test_layout_hypotheses(variant, blocks, size):
    for terminal in [False, True]:
        candidate = normal.layout(variant, terminal_training=terminal)
        assert candidate.total_symbols == blocks*(size+32)+32*terminal
        assert len(candidate.training_offsets) == blocks+terminal
        assert candidate.data_offsets[-1]+size == blocks*(size+32)


def _wave(*, terminal=True, echo=False, noise=0.05, wrong_conjugation=False):
    candidate = normal.layout('short', terminal_training=terminal)
    rng = np.random.default_rng(609)
    data = 2*rng.integers(0, 2, (6,176))-1
    symbols = np.zeros(candidate.total_symbols, complex)
    # Explicit published 16-entry fixture; no production training generator.
    c = np.array([1,-1,-1j,-1,-1,-1j,-1j,1j,1,1,-1j,1,-1,1j,-1j,-1j])
    train = np.tile(np.roll(c, 8), 2)
    for i, offset in enumerate(candidate.training_offsets):
        symbols[offset:offset+32] = train.conj() if i%2 and not wrong_conjugation else train
    for i, offset in enumerate(candidate.data_offsets):
        symbols[offset:offset+176] = data[i]
    impulses = np.zeros(len(symbols)*16, complex)
    impulses[::16] = symbols
    z = np.pad(np.convolve(impulses,RRC_TAPS), (93, 200))
    if echo:
        z[16:] += 0.38*np.exp(0.8j)*z[:-16].copy()
    z *= np.exp(1j*(0.7+2*np.pi*7*np.arange(len(z))/28800))
    z += noise*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
    return z, candidate, data


@pytest.mark.parametrize('echo', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
def test_equalization_unseen_data_noise_cfo_and_multipath(echo, terminal):
    z, candidate, expected = _wave(echo=echo, terminal=terminal)
    observed = normal.recover_bpsk(z, first_training_start=90, layout=candidate,
                                   timing_offsets=range(0,7), cfo_hz=(4,7,10))
    assert abs(observed.first_training_start-93) <= (2 if echo else 0)
    assert observed.cfo_hz == 7
    off_cfo = observed.candidates[observed.candidates[:,1] != 7,2]
    assert off_cfo.max() < observed.coherence / 2
    assert np.max(observed.condition_number) < 3
    assert observed.coherence > (0.80 if echo else 0.98)
    live = np.isfinite(observed.bpsk_observation)
    np.testing.assert_array_equal(np.sign(observed.bpsk_observation[live]), expected[live])
    assert np.mean(abs(observed.equalized[live]-expected[live])**2) < 0.04
    assert np.max(observed.holdout_nmse) < 0.04
    assert np.count_nonzero(~live) == (0 if terminal else 2)


def test_training_refreshed_each_block_and_echo_equalization_helps():
    z, candidate, expected = _wave(echo=True, noise=0.02)
    scalar = normal.recover_bpsk(z, first_training_start=93, layout=candidate,
                                 timing_offsets=(0,), cfo_hz=(7,), equalizer_half_width=0)
    fir = normal.recover_bpsk(z, first_training_start=93, layout=candidate,
                              timing_offsets=(0,), cfo_hz=(7,), equalizer_half_width=2)
    assert np.mean(abs(fir.equalized-expected)**2) < np.mean(abs(scalar.equalized-expected)**2)/3
    assert fir.equalizer.shape == (7,5)


def test_wrong_conjugation_and_noise_are_not_accepted():
    z, candidate, _ = _wave(wrong_conjugation=True)
    wrong = normal.recover_bpsk(z, first_training_start=93, layout=candidate,
                                timing_offsets=(0,), cfo_hz=(7,))
    assert wrong.coherence < 0.5
    assert np.max(wrong.holdout_nmse) > 0.4
    rng = np.random.default_rng(802)
    for samples in (np.zeros_like(z), rng.normal(size=len(z))+1j*rng.normal(size=len(z))):
        observed = normal.recover_bpsk(samples, first_training_start=93, layout=candidate,
                                       timing_offsets=(0,))
        assert observed.coherence < 0.05
        assert np.mean(observed.holdout_nmse) > 0.5


def test_wrong_terminal_hypothesis_has_bad_last_training():
    z, _, _ = _wave(terminal=False)
    # Extra surrounding noise makes the longer hypothesis evaluable, not true.
    z = np.pad(z, (0, 512))
    observed = normal.recover_bpsk(z, first_training_start=93,
                                  layout=normal.layout(terminal_training=True),
                                  timing_offsets=(0,), cfo_hz=(7,))
    assert observed.training_coherence[-1] < 0.2
    assert observed.holdout_nmse[-1] > 0.5


def test_invalid_inputs():
    with pytest.raises(ValueError):
        normal.training_sequence(0)
    with pytest.raises(ValueError):
        normal.layout(terminal_training=1)
    with pytest.raises(ValueError, match='complete'):
        normal.recover_bpsk(np.zeros(100), first_training_start=0,
                            layout=normal.layout(terminal_training=True))
    with pytest.raises(ValueError, match='<= 4'):
        normal.recover_bpsk(np.zeros(100), first_training_start=0,
                            layout=normal.layout(terminal_training=True), equalizer_half_width=5)
