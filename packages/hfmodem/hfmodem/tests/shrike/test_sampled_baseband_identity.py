# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from time import perf_counter

import numpy as np
import pytest

from hfmodem.shrike import placement, rx, rxfront, spec
from hfmodem.shrike.rxfront import FS, SPS, _matched_filter, _sampled_kernels


def _reference_carrier(cn, fs, start, count):
    period = fs / spec.TONE_SPACING_HZ
    n = start + np.arange(count)
    if period != int(period):
        return np.exp(-2j * np.pi * spec.channel_freq_hz(cn) * n / fs)
    period = int(period)
    turn = np.exp(-2j * np.pi * spec.channel_freq_hz(cn)
                  * np.arange(period) / fs)
    return np.resize(np.roll(turn, -start % period), count)


def _reference_sampled_baseband(audio: np.ndarray, tones, idx: np.ndarray) -> dict:
    """`rx._baseband` for each tone, evaluated ONLY where `idx` reads it.

    The matched filter is 1860 taps and the convolution produces one output per
    input sample -- some 43,000 for the window one data frame lives in, of which
    the grid reads 81. A sweeping receiver has no choice, because it does not know
    which 81. A receiver that has locked does know, and a filter evaluated only at
    its sample instants is one small matrix product instead of six transforms of
    the window. Measured on the station Pi, that convolution was 75% of a tracked
    decode.

    Returns full-length arrays with only `idx` populated, so `rx.cells_from_baseband`
    and `rx.cs_bits` index them exactly as they index the real thing. Real
    histories are gathered once for all tones; mixing is in cached FIR kernels.
    The full-filter equivalence pins agree at those positions to 2e-12.
    """
    taps = _matched_filter()
    m = taps.size
    lo, hi = int(idx.min()), int(idx.max())
    b0, b1 = max(0, lo - m + 1), min(len(audio), hi + 1)
    pad = np.zeros(m - 1, audio.dtype)
    rows = idx - b0
    tones = tuple(tones)
    win = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad, audio[b0:b1], pad]), m)[rows]
    values = win @ _sampled_kernels(tones)
    values = values[:, :len(tones)] + 1j * values[:, len(tones):]
    out = {}
    for rank, cn in enumerate(tones):
        z = np.zeros(len(audio) + m - 1, complex)
        phase = _reference_carrier(cn, FS, lo - m + 1, hi - lo + 1)
        z[idx] = values[:, rank] * phase[idx - lo]
        out[cn] = z
    return out


@pytest.mark.parametrize('fs', [48000, 12000, 44100, 47999])
@pytest.mark.parametrize('start,count', [(-1860, 0), (-1859, 1), (0, 401), (47999, 48001)])
def test_carrier_table_is_identical(fs, start, count):
    for cn in range(spec.N_CHANNELS):
        new = rx.carrier(cn, fs, start, count)
        reference = _reference_carrier(cn, fs, start, count)
        assert new.dtype == reference.dtype
        assert new.shape == reference.shape
        assert new.tobytes() == reference.tobytes()


def _tracked_indices():
    path = placement.SPEED_PATHS[6]
    delay = (_matched_filter().size - 1) // 2
    return 9600 + (np.arange(path.n_symbols + 9) - 1) * SPS + delay


@pytest.mark.parametrize('tones', [(5, 12), placement.SPEED_PATHS[3].tones,
                                    tuple(range(spec.N_CHANNELS)), (17, 0, 5)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('index_dtype', [np.int64, np.int32, np.uint64])
@pytest.mark.parametrize('pattern', ['near-zero', 'tracked', 'unsorted', 'tail'])
def test_sampled_baseband_bits_agree(tones, dtype, index_dtype, pattern):
    rng = np.random.default_rng(72)
    audio = rng.normal(size=FS).astype(dtype)
    m = _matched_filter().size
    idx = {'near-zero': np.array([0, 1, 3, 120, 479, 480]),
           'tracked': _tracked_indices(),
           'unsorted': np.array([5000, 3, 5000, 499, 1000]),
           'tail': np.array([audio.size - 1, audio.size, audio.size + m - 2])}[pattern]
    idx = idx.astype(index_dtype)
    new = rxfront._sampled_baseband(audio, tones, idx)
    reference = _reference_sampled_baseband(audio, tones, idx)
    assert list(new) == list(reference) == list(tones)
    for c in tones:
        assert new[c].shape == reference[c].shape == (len(audio) + m - 1,)
        assert new[c].dtype == reference[c].dtype == np.complex128
        assert new[c].tobytes() == reference[c].tobytes()


def test_sampled_baseband_noninteger_period_fallback(monkeypatch):
    monkeypatch.setattr(rxfront, 'FS', 47999)
    # The copied implementation resolves FS in this test module.
    monkeypatch.setitem(_reference_sampled_baseband.__globals__, 'FS', 47999)
    audio = np.random.default_rng(52).normal(size=48000)
    idx = _tracked_indices()
    tones = (5, 12)
    _sampled_kernels.cache_clear()
    try:
        new = rxfront._sampled_baseband(audio, tones, idx)
        reference = _reference_sampled_baseband(audio, tones, idx)
    finally:
        _sampled_kernels.cache_clear()
    assert list(new) == list(reference) == list(tones)
    for c in tones:
        assert new[c].dtype == reference[c].dtype
        assert new[c].shape == reference[c].shape
        assert new[c].tobytes() == reference[c].tobytes()


@pytest.mark.wall_clock
def test_sampled_baseband_indexes_one_carrier_period():
    audio = np.random.default_rng(2026).normal(size=FS)
    tones, idx = placement.SPEED_PATHS[6].tones, _tracked_indices()
    functions = (rxfront._sampled_baseband, _reference_sampled_baseband)
    for function in functions:
        function(audio, tones, idx)
    times = [[], []]
    for repeat in range(11):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = perf_counter()
            functions[index](audio, tones, idx)
            times[index].append(perf_counter() - start)
    new, reference = map(min, times)
    print(f'sampled baseband: new={new:.6f}s reference={reference:.6f}s ratio={new / reference:.4f}')
    # The window product both share dominates on Windows: 0.78 there, 0.23 on macOS.
    assert new <= reference * 0.9
