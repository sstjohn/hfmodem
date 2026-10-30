# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import p4oracle, p4rx, p4sig
from hfmodem.shrike.p4oracle import (
    _DESPREAD_CFOS, _DESPREAD_MIN_BLOCKS, _SPREAD_CONJ, _SPREAD_ENERGY,
)

FX = Path(__file__).parent / 'fixtures'
RECORDINGS = sorted(
    [p.relative_to(FX).as_posix() for pattern in (
        'n9vc-0802-*.wav', 'kb5lzk-0923-robust-answer.wav',
        'p4-breakin/*.wav', 'n9vc-0925-kiwi-*.wav',
    ) for p in FX.glob(pattern)])


def _reference_coarse_fractions(chips: np.ndarray) -> np.ndarray:
    """`_despread_at`'s fraction over the whole coarse grid, as [carrier, start].

    Every 16-chip window is correlated once per carrier, and each start takes
    the mean over the windows on its own block grid.
    """
    turns = np.arange(chips.size) / p4rx.CHIP_RATE
    tuned = chips * np.exp(-2j * np.pi * _DESPREAD_CFOS[:, None] * turns)
    windows = np.lib.stride_tricks.sliding_window_view(
        tuned, p4sig.SPREAD_FACTOR, axis=1)
    power = np.maximum(np.lib.stride_tricks.sliding_window_view(
        np.abs(chips) ** 2, p4sig.SPREAD_FACTOR).sum(1), 1e-30) * _SPREAD_ENERGY
    ratio = np.abs(windows @ _SPREAD_CONJ) ** 2 / power
    fractions = np.zeros((_DESPREAD_CFOS.size, p4sig.SPREAD_FACTOR))
    for start in range(p4sig.SPREAD_FACTOR):
        usable = (chips.size - start) // p4sig.SPREAD_FACTOR
        if usable >= _DESPREAD_MIN_BLOCKS:
            fractions[:, start] = ratio[:, start::p4sig.SPREAD_FACTOR][:, :usable].mean(1)
    return fractions


@pytest.mark.parametrize('recording', RECORDINGS)
def test_recorded_coarse_fractions_and_locks_agree(recording, monkeypatch):
    root = Path(__file__).resolve().parents[5]
    assert Path(p4oracle.__file__).resolve().is_relative_to(root)
    fs, audio = wavfile.read(FX / recording)
    x = audio.astype(np.float64) / (np.iinfo(audio.dtype).max + 1)
    if x.ndim > 1:
        x = x.mean(1)
    found = p4oracle.bursts(x, fs)
    burst_count = len(found)
    # Weak or tightly cropped recordings may have no edge above the burst knee.
    if not found:
        found = [(0.0, x.size / fs)]
    max_relative = 0.0
    for onset, duration in found:
        seg = x[round(onset * fs):round((onset + duration) * fs)]
        z = p4rx.audio_to_baseband(np.asarray(seg, np.float64), fs)
        matched = p4rx.matched_filter(z)[p4rx.MATCHED_DELAY:]
        grids = [matched[phase::p4rx.SAMPLES_PER_CHIP]
                 for phase in range(p4rx.SAMPLES_PER_CHIP)]
        for chips in grids:
            new = p4oracle._coarse_fractions(chips)
            reference = _reference_coarse_fractions(chips)
            # Reference phase grows to 2*pi*1488*N/1800 rad, rounded by float64
            # to about N*1e-12 rad: O(1e-12..1e-10) relative over recordings.
            # 1e-9 leaves margin; any _lock change fails the exact check below.
            np.testing.assert_allclose(new, reference, rtol=1e-9, atol=1e-12)
            relative = np.abs(new - reference) / np.maximum(np.abs(reference), 1e-30)
            max_relative = max(max_relative, float(relative.max()))
        new_lock = p4oracle._lock(seg, fs)
        new_word = p4oracle.despread_word(seg, fs)
        with monkeypatch.context() as patch:
            patch.setattr(p4oracle, '_coarse_fractions', _reference_coarse_fractions)
            reference_lock = p4oracle._lock(seg, fs)
            reference_word = p4oracle.despread_word(seg, fs)
        assert np.array_equal(new_word, reference_word)
        assert len(new_lock) == len(reference_lock) == 5
        for new, reference in zip(new_lock, reference_lock):
            if isinstance(new, np.ndarray):
                assert np.array_equal(new, reference)
            else:
                assert new == reference
    print(f'{recording}: max relative difference {max_relative:.3e}; '
          f'{burst_count} bursts covered, {len(found)} segments including fallback')


@pytest.mark.parametrize('size', [16, 63, 64, 65, 79, 80])
@pytest.mark.parametrize('zero', [False, True])
@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
def test_short_and_zero_chip_grids_agree(size, zero, dtype):
    rng = np.random.default_rng(42)
    chips = np.zeros(size, complex) if zero else rng.normal(size=size) + 1j * rng.normal(size=size)
    chips = chips.astype(dtype)
    new = p4oracle._coarse_fractions(chips)
    reference = _reference_coarse_fractions(chips)
    assert new.shape == reference.shape == (125, 16)
    # complex64 quantization and power are shared; complex128 taps keep correlation rounding float64.
    np.testing.assert_allclose(new, reference, rtol=1e-9, atol=1e-12)
    starts = np.arange(p4sig.SPREAD_FACTOR)
    unusable = (size - starts) // p4sig.SPREAD_FACTOR < _DESPREAD_MIN_BLOCKS
    assert np.all(new[:, unusable] == 0)
    if zero:
        assert np.all(new == 0)


@pytest.mark.wall_clock
def test_the_coarse_grid_is_one_contiguous_product():
    rng = np.random.default_rng(2026)
    chips = rng.normal(size=5400) + 1j * rng.normal(size=5400)
    assert p4oracle._COARSE_TAPS.shape == (16, 125)
    assert p4oracle._COARSE_TAPS.dtype == np.complex128
    assert p4oracle._COARSE_TAPS.flags.c_contiguous
    functions = (p4oracle._coarse_fractions, _reference_coarse_fractions)
    for function in functions:
        function(chips)
    times = [[], []]
    for repeat in range(11):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = perf_counter()
            functions[index](chips)
            times[index].append(perf_counter() - start)
    new, reference = map(min, times)
    print(f'coarse grid: new={new:.6f}s reference={reference:.6f}s ratio={new / reference:.4f}')
    assert new <= reference / 2
