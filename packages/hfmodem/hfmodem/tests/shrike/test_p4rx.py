# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Published-table checks plus independently assembled channel experiments.

These test DSP convention and bounded impairments, not Dragon interoperability.
"""
from pathlib import Path
import re

import numpy as np
import pytest
from scipy.signal import resample_poly

from hfmodem.shrike import p4rx


def test_published_tables_and_impulse_delay():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / 'working/pactor/spec/refs/scs/PACTOR-4_Protocol.txt').exists()), None)
    if root is None:
        pytest.skip("local SCS reference text unavailable")
    text = (root / 'working/pactor/spec/refs/scs/PACTOR-4_Protocol.txt').read_text()
    for name, expected in [('scsRrcf', p4rx.RRC_TAPS),
                           ('scsSpread8', p4rx.SPREAD8), ('scsSpread16', p4rx.SPREAD16)]:
        raw = re.search(r'short ' + name + r'\[\d+\]\s*=\s*\{([^}]+)', text).group(1)
        values = np.array([int(v) for v in re.findall(r'-?\d+', raw)]) / 32767
        actual = values if name == 'scsRrcf' else values[::2] + 1j * values[1::2]
        np.testing.assert_allclose(expected, actual, atol=1e-16, rtol=0)
    y = p4rx.matched_filter(p4rx.RRC_TAPS)
    assert len(p4rx.RRC_TAPS) == 129
    assert p4rx.RRC_TAPS[-1] == 0
    assert y.argmax() == 128
    assert abs(y[128] - 1) < 1e-12
    np.testing.assert_allclose(y, np.correlate(p4rx.RRC_TAPS, p4rx.RRC_TAPS, 'full') /
                               np.sum(p4rx.RRC_TAPS ** 2), atol=1e-14)


def _channel(sf, *, multipath=False):
    # Independent explicit phase-state recursion, no production TX/encoder.
    rng = np.random.default_rng(82 + sf)
    increments = rng.integers(0, 4, 80)
    symbols = np.array([1, 1j, -1, -1j])[np.r_[0, np.cumsum(increments)] % 4]
    spread = p4rx.SPREAD8 if sf == 8 else p4rx.SPREAD16
    impulses = np.zeros(symbols.size * sf * 16, complex)
    impulses[::16] = (symbols[:, None] * spread).ravel()
    shaped = np.convolve(impulses, p4rx.RRC_TAPS)
    start = 173
    z = np.pad(shaped, (start, 150)) * np.exp(0.7j)
    if multipath:
        z[11:] += 0.23 * np.exp(0.4j) * z[:-11].copy()
    z *= np.exp(2j * np.pi * 13 * np.arange(z.size) / 28800)
    z += 0.17 * (rng.normal(size=z.size) + 1j * rng.normal(size=z.size))
    return z, start, increments


@pytest.mark.parametrize('sf', [8, 16])
@pytest.mark.parametrize('multipath', [False, True])
def test_bounded_timing_cfo_noise_and_echo(sf, multipath):
    z, start, expected = _channel(sf, multipath=multipath)
    result = p4rx.recover_robust(z, reference_start=start - 3, spread_factor=sf,
                               data_symbols=len(expected), timing_offsets=range(-2, 9),
                               cfo_hz=(7, 10, 13, 16, 19))
    assert abs(result.reference_start - start) <= (2 if multipath else 0)
    # A short echo biases this spreading-only CFO objective by one grid step.
    assert abs(result.cfo_hz - 13) <= (3 if multipath else 0)
    assert result.coherence > (0.94 if multipath else 0.98)
    np.testing.assert_array_equal(result.phase_similarity.argmax(axis=1), expected)
    assert np.quantile(result.phase_margin, 0.05) > (0.70 if multipath else 0.85)
    assert result.chips.shape == (81, sf)
    assert result.candidates.shape == (55, 3)


def test_audio_frequency_sign_and_resampling():
    fs = 48000
    t = np.arange(fs) / fs
    audio = 0.4 * np.cos(2 * np.pi * 1573 * t + 0.9)
    bb = p4rx.audio_to_baseband(audio, fs)
    expected = 0.4 * np.exp(1j * (2 * np.pi * 73 * np.arange(28800) / 28800 + 0.9))
    np.testing.assert_allclose(bb[300:-300], expected[300:-300], atol=0.001)


def test_real_audio_path_retains_robust_phases():
    z, start, expected = _channel(16)
    at_48k = resample_poly(z, 5, 3)
    audio = (at_48k * np.exp(2j * np.pi * 1500 * np.arange(at_48k.size) / 48000)).real
    bb = p4rx.audio_to_baseband(audio, 48000)
    result = p4rx.recover_robust(bb, reference_start=start, spread_factor=16,
                               data_symbols=len(expected), cfo_hz=(13,))
    np.testing.assert_array_equal(result.phase_similarity.argmax(axis=1), expected)
    assert result.coherence > 0.97


def test_zero_and_noise_do_not_imply_confidence():
    rng = np.random.default_rng(13)
    for z in [np.zeros(10000), rng.normal(size=10000) + 1j * rng.normal(size=10000)]:
        result = p4rx.recover_robust(z, reference_start=100, spread_factor=16,
                                   data_symbols=30, timing_offsets=(0,))
        assert result.coherence < 0.15
        if not np.any(z):
            assert not np.any(result.phase_similarity)
            assert not np.any(result.phase_margin)


def test_invalid_or_truncated_input_rejected():
    with pytest.raises(ValueError, match='complete'):
        p4rx.recover_robust(np.zeros(100), reference_start=0, spread_factor=16, data_symbols=30)
    with pytest.raises(ValueError, match='spread_factor'):
        p4rx.recover_robust(np.zeros(100), reference_start=0, spread_factor=7, data_symbols=30)
    with pytest.raises(ValueError, match='integer sample'):
        p4rx.recover_robust(np.zeros(100), reference_start=0, spread_factor=16, data_symbols=30,
                           timing_offsets=(0.5,))
    with pytest.raises(ValueError, match='finite'):
        p4rx.matched_filter([1, np.nan])
    with pytest.raises(ValueError, match='real'):
        p4rx.audio_to_baseband(np.ones(100, complex), 48000)


@pytest.mark.parametrize('field,value', [('reference_start', np.nan), ('reference_start', np.inf),
                                         ('data_symbols', np.inf), ('data_symbols', 1.5)])
def test_invalid_counts(field, value):
    kwargs = dict(reference_start=0, spread_factor=8, data_symbols=2)
    kwargs[field] = value
    with pytest.raises(ValueError, match='finite integers'):
        p4rx.recover_robust(np.zeros(1000), **kwargs)


def test_integral_float_spreading_factor():
    result = p4rx.recover_robust(np.zeros(1000), reference_start=0, spread_factor=8.0,
                               data_symbols=2, timing_offsets=(0,))
    assert result.chips.shape == (3, 8)
