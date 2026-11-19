# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import importlib
import importlib.util
import sys

import numpy as np
import pytest

from hfmodem import cli
from hfmodem.core import grid_fold
from hfmodem.shrike import p3acquire, placement, rx, spec
from hfmodem.shrike.p3acquire import SEARCH_FS, STEP, _fold_length, _fold_weight, fft, ifft
from hfmodem.tests import native as native_helpers
from hfmodem.tests.shrike.archive import requires_ws8eoc_p3
from hfmodem.tests.shrike.test_p3_gateway_acquisition import audio


def inputs(width, scale=1, seed=123):
    rng = np.random.default_rng(seed)

    def values(shape, factor):
        a = np.empty(shape, dtype=np.complex128)
        a.real = rng.uniform(-0.125, 0.125, shape) * factor
        a.imag = rng.uniform(-0.125, 0.125, shape) * factor
        return a

    spectra = [values(10 * width, scale) for _ in range(3)]
    # Keep subnormal products nonzero; large values exercise large products too.
    weight = values((10, width), 1 if scale == 1e-310 else scale)
    picks = [2, 0, 1, 2, 1, 0]
    shifts = [-30 * width - 1, 0, 10 * width - 1, 50 * width + 3, 2**62 + 7, -2**62 - 7]
    tones = [0, 30 * width - 7]
    return spectra, picks, shifts, tones, weight


def original_grid(analytic: np.ndarray, offsets, first: int,
          count: int) -> dict[int, np.ndarray]:
    """`p3acquire._grid` as it was before the one-call fold, kept as the reference."""
    n = len(analytic)
    nfft = _fold_length(n + len(rx._pulse(SEARCH_FS // 100)) - 1)
    weight = _fold_weight(nfft, first % STEP)
    width = nfft // STEP
    spectrum = fft(analytic, nfft)
    twice, mirror = (np.concatenate([spectrum, spectrum]),
                     np.tile(np.conj(np.roll(spectrum[::-1], 1)), 2))
    rows = {cn: np.empty((len(offsets), width), complex)
            for cn in spec.VH_CHANNELS}
    for i, hz in enumerate(offsets):
        shift = hz * nfft / SEARCH_FS
        if abs(shift - round(shift)) < 1e-9:
            line, image, shift = twice, mirror, round(shift)
        else:
            moved = fft(analytic * np.exp(-2j*np.pi*hz*np.arange(n)/SEARCH_FS),
                        nfft)
            line, image, shift = (np.concatenate([moved, moved]),
                                  np.tile(np.conj(np.roll(moved[::-1], 1)), 2),
                                  0)
        for cn in spec.VH_CHANNELS:
            tone = round(spec.channel_freq_hz(cn) * nfft / SEARCH_FS)
            a, b = (shift + tone) % nfft, (tone - shift) % nfft
            z = line[a:a + nfft] + image[b:b + nfft]
            rows[cn][i] = np.sum(z.reshape(STEP, width) * weight, axis=0)
    at = first // STEP
    return {cn: ifft(v, axis=1)[:, at:at + count] for cn, v in rows.items()}


@pytest.mark.parametrize('width', [1, 2, 160, 4800])
@pytest.mark.parametrize('scale', [0, 1e-310, 1, 1e154])
def test_fold_bits(width, scale):
    args = inputs(width, scale)
    spectra, picks, shifts, tones, weight = args
    expected = np.empty((2, len(picks), width), complex)
    for c, tone in enumerate(tones):
        for i, (pick, shift) in enumerate(zip(picks, shifts)):
            s = spectra[pick]
            twice = np.concatenate([s, s])
            mirror = np.tile(np.conj(np.roll(s[::-1], 1)), 2)
            a, b = (shift + tone) % len(s), (tone - shift) % len(s)
            z = twice[a:a + len(s)] + mirror[b:b + len(s)]
            expected[c, i] = np.sum(z.reshape(STEP, width) * weight, axis=0)
    for fold in (grid_fold.fold_numpy, grid_fold.fold_grid):
        out = np.empty_like(expected)
        fold(*args, out)
        assert out.tobytes() == expected.tobytes()


def test_native_admitted():
    native_helpers.dsp()
    assert grid_fold.NATIVE.startswith('native: yes (Rust grid fold, ')
    assert grid_fold.fold_grid is not grid_fold.fold_numpy


def test_native_dispatch():
    native = native_helpers.dsp()
    calls = []

    def counted(*args):
        calls.append(args[-1])
        native.fold_grid(*args)

    fold, status = grid_fold._select(counted)
    assert status.startswith('native: yes')
    calls.clear()
    args = inputs(160)
    out = np.empty((2, 6, 160), complex)
    fold(*args, out)
    assert len(calls) == 1
    expected = np.empty_like(out)
    grid_fold.fold_numpy(*args, expected)
    assert out.tobytes() == expected.tobytes()


def test_wrong_kernel_refused(monkeypatch):
    def wrong(*args):
        args[-2].fill(42)
    assert grid_fold.probe(wrong) is None
    with pytest.warns(RuntimeWarning, match='does not match this NumPy') as caught:
        fold, status = grid_fold._select(wrong)
    assert len(caught) == 1
    assert fold is grid_fold.fold_numpy
    assert status == 'native: no (extension does not match this NumPy)'
    native = native_helpers.dsp()
    try:
        with monkeypatch.context() as m:
            m.setattr(native, 'fold_grid', wrong)
            with pytest.warns(RuntimeWarning, match='does not match this NumPy') as caught:
                importlib.reload(grid_fold)
            assert len(caught) == 1
            assert grid_fold.fold_grid is grid_fold.fold_numpy
            assert grid_fold.NATIVE == status
    finally:
        importlib.reload(grid_fold)


def test_extension_absent(monkeypatch):
    try:
        with monkeypatch.context() as m:
            m.setitem(sys.modules, 'hfmodem.core._dsp', None)
            importlib.reload(grid_fold)
            assert grid_fold.fold_grid is grid_fold.fold_numpy
            assert grid_fold.NATIVE == 'native: no (extension not built)'
    finally:
        importlib.reload(grid_fold)


def test_version(capsys, monkeypatch):
    for spec_found, line in ((object(), 'native: built (admitted against NumPy on first use)'),
                             (None, 'native: no (extension not built)')):
        monkeypatch.setattr(importlib.util, 'find_spec', lambda name: spec_found)
        assert cli.cmd_version(None) == 0
        assert capsys.readouterr().out.splitlines() == [cli.__version__, line]


@pytest.mark.parametrize('poison', ['nan', 'inf', 'overflow'])
def test_non_finite_grid_matches_numpy(poison):
    spectra, picks, shifts, tones, weight = inputs(160)
    if poison == 'overflow':
        spectra[0] *= 1e300
        weight = weight * 1e10
    else:
        spectra[1][[3, 700]] = complex(float(poison), -float(poison))
    expected = np.empty((2, len(picks), 160), complex)
    out = np.empty_like(expected)
    with np.errstate(all='ignore'):
        grid_fold.fold_numpy(spectra, picks, shifts, tones, weight, expected)
        grid_fold.fold_grid(spectra, picks, shifts, tones, weight, out)
    assert not np.isfinite(expected).all()
    assert out.tobytes() == expected.tobytes()


def test_probe_ignores_caller_float_errors():
    native = native_helpers.dsp()
    with np.errstate(all='raise'):
        assert grid_fold.probe(native.fold_grid) is not None


def search_bytes(found):
    assert found is not None
    return tuple(a.tobytes() for a in (
        found.starts, found.accepted, found.cis, found.qualities, found.diffs,
        found.trial_signs.signs, found.trial_signs.cis, found.analytic))


def compare_acquisition(x, monkeypatch):
    offsets = (0., -25., 25., -12.1)
    analytic = p3acquire.hilbert(p3acquire.resample_poly(x, 1, p3acquire.DECIMATION))
    expected_grid = original_grid(analytic, offsets, 153, 100)
    with monkeypatch.context() as m:
        m.setattr(p3acquire, '_grid', original_grid)
        expected_search = search_bytes(p3acquire._search(x, offsets, range(6), 21, 1641))
        expected_candidate = p3acquire.changeover(x, offsets=offsets)
    for fold in (grid_fold.fold_numpy, grid_fold.fold_grid):
        with monkeypatch.context() as m:
            calls = []
            def counted(*args):
                calls.append(args)
                fold(*args)
            m.setattr(p3acquire, '_fold', counted)
            got = p3acquire._grid(analytic, offsets, 153, 100)
            assert len(calls) == 1
            assert len(calls[0][0]) == 2
            assert {cn: a.tobytes() for cn, a in got.items()} == {
                cn: a.tobytes() for cn, a in expected_grid.items()}
            assert search_bytes(p3acquire._search(x, offsets, range(6), 21, 1641)) == expected_search
            assert p3acquire.changeover(x, offsets=offsets) == expected_candidate
    return expected_search, expected_candidate


def test_synthetic_acquisition(monkeypatch):
    x = np.pad(placement.control_signal(placement.BREAKIN_CS), (4800, 4800))
    x = p3acquire.compensate(x, 12.1)
    found, candidate = compare_acquisition(x, monkeypatch)
    assert found
    assert candidate is not None


@requires_ws8eoc_p3
@pytest.mark.parametrize('name', [
    '0908-p3-head-4', '0908-p3-head-5', '0908-p3-rms', '0909-p3-banner', '0909-p3-ode',
    '0909-p3-ode-repeat', '0909-p3-ode-zero', '0909-p3-rms', '0909-p3-trim'])
def test_tape_acquisition(name, monkeypatch):
    compare_acquisition(audio(f'ws8eoc-{name}.wav'), monkeypatch)
