# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Raw buffer contract and independently rounded arithmetic for the optional kernel."""
from concurrent.futures import ThreadPoolExecutor
import math
from threading import Barrier

import numpy as np
import pytest

native = pytest.importorskip('hfmodem.core._dsp')
_fma = np.frompyfunc(math.fma, 3, 1)


def reference(spectra, picks, shifts, tones, weight, fused):
    width = weight.shape[1]
    nfft = 10 * width
    out = np.zeros((len(tones), len(picks), width), dtype=np.complex128)
    for c, tone in enumerate(tones):
        for i, (pick, shift) in enumerate(zip(picks, shifts)):
            spectrum = spectra[pick]
            twice = np.concatenate([spectrum, spectrum])
            mirror = np.tile(np.conj(np.roll(spectrum[::-1], 1)), 2)
            a, b = (shift + tone) % nfft, (tone - shift) % nfft
            z = (twice[a:a + nfft] + mirror[b:b + nfft]).reshape(10, width)
            for k in range(10):
                zr, zi = z[k].real, z[k].imag
                wr, wi = weight[k].real, weight[k].imag
                if fused:
                    pr = _fma(zr, wr, -(zi * wi)).astype(np.float64)
                    pi = _fma(zr, wi, zi * wr).astype(np.float64)
                else:
                    pr = zr * wr - zi * wi
                    pi = zr * wi + zi * wr
                out[c, i].real += pr
                out[c, i].imag += pi
    return out


def assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


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


@pytest.mark.parametrize('width', [1, 2, 7, 160, 4800])
@pytest.mark.parametrize('scale', [0, 1e-310, 1, 1e154])
@pytest.mark.parametrize('fused', [False, True])
def test_exact_rounding(width, scale, fused):
    args = inputs(width, scale)
    expected = reference(*args, fused)
    out = np.full(expected.shape, complex(42, -13))
    args[0][0].flags.writeable = False
    args[4].flags.writeable = False
    assert native.fold_grid(*args, out, fused) is True
    assert_bits(out, expected)


@pytest.mark.parametrize('fused', [False, True])
def test_numpy_expression_when_products_are_exact(fused):
    args = inputs(7)
    spectra, picks, shifts, tones, weight = args
    # Powers of two make this comparison independent of NumPy's contraction.
    for a in [*spectra, weight]:
        a.real = np.round(a.real * 32) / 8
        a.imag = np.round(a.imag * 32) / 8
    out = np.empty((2, 6, 7), dtype=np.complex128)
    native.fold_grid(*args, out, fused)
    for c, tone in enumerate(tones):
        for i, (pick, shift) in enumerate(zip(picks, shifts)):
            s = spectra[pick]
            twice = np.concatenate([s, s])
            mirror = np.tile(np.conj(np.roll(s[::-1], 1)), 2)
            a, b = (shift + tone) % len(s), (tone - shift) % len(s)
            z = twice[a:a + len(s)] + mirror[b:b + len(s)]
            assert_bits(out[c, i], np.sum(z.reshape(10, 7) * weight, axis=0))


@pytest.mark.parametrize('target', ['spectrum', 'weight', 'out'])
@pytest.mark.parametrize('defect', ['dtype', 'format', 'big_endian', 'strided', 'unaligned', 'rank'])
def test_invalid_buffers_leave_output_untouched(target, defect):
    spectra, picks, shifts, tones, weight = inputs(7)
    out = np.full((2, 6, 7), complex(42, -13))
    original = {'spectrum': spectra[1], 'weight': weight, 'out': out}[target]
    if defect == 'dtype':
        bad = original.astype(np.complex64)
    elif defect == 'format':
        bad = original.view('V16')
    elif defect == 'big_endian':
        bad = original.astype('>c16')
    elif defect == 'strided':
        bad = np.repeat(original, 2, axis=-1)[..., ::2]
    elif defect == 'unaligned':
        bad = np.ndarray(original.shape, dtype=np.complex128,
                         buffer=bytearray(original.nbytes + 1), offset=1)
        bad[:] = original
    else:
        bad = original.reshape((*original.shape, 1))
    if target == 'spectrum':
        spectra[1] = bad
    elif target == 'weight':
        weight = bad
    else:
        out = bad
    before = out.tobytes()
    with pytest.raises((ValueError, BufferError)):
        native.fold_grid(spectra, picks, shifts, tones, weight, out, False)
    assert out.tobytes() == before


@pytest.mark.parametrize('defect', [
    'readonly', 'spectrum_alias', 'weight_alias', 'unpicked_alias', 'partial_alias',
    'spectrum_length', 'weight_rows', 'zero_width', 'out_shape', 'pick',
    'negative_pick', 'huge_pick', 'mismatch', 'fractional_shift', 'fractional_pick',
])
def test_invalid_arguments_leave_output_untouched(defect):
    width = 7
    spectra, picks, shifts, tones, weight = inputs(width)
    out = np.full((2, 6, width), complex(42, -13))
    if defect == 'readonly':
        out.flags.writeable = False
    elif defect in {'spectrum_alias', 'weight_alias', 'unpicked_alias', 'partial_alias'}:
        picks, shifts, tones = [0], [0], [0]
        source = weight if defect == 'weight_alias' else spectra[0]
        if defect == 'unpicked_alias':
            source = spectra[2]
        offset = 3 if defect == 'partial_alias' else 0
        out = source.ravel()[offset:offset + width].reshape(1, 1, width)
    elif defect == 'spectrum_length':
        spectra[2] = spectra[2][:-1]
    elif defect == 'weight_rows':
        weight = weight[:9]
    elif defect == 'zero_width':
        weight = np.empty((10, 0), dtype=np.complex128)
    elif defect == 'out_shape':
        out = out[:, :5].copy()
    elif defect in {'pick', 'negative_pick', 'huge_pick', 'fractional_pick'}:
        picks[0] = {'pick': 3, 'negative_pick': -1, 'huge_pick': 10**60,
                    'fractional_pick': 0.5}[defect]
    elif defect == 'mismatch':
        shifts.pop()
    else:
        shifts[0] = 0.5
    before = out.tobytes()
    with pytest.raises((ValueError, BufferError)):
        native.fold_grid(spectra, picks, shifts, tones, weight, out, True)
    assert out.tobytes() == before


@pytest.mark.parametrize('fused', [False, True])
def test_four_concurrent_calls(fused):
    args = inputs(160)
    expected = reference(*args, fused)
    barrier = Barrier(4)

    def worker(_):
        out = np.empty_like(expected)
        barrier.wait()
        for _ in range(8):
            native.fold_grid(*args, out, fused)
            assert_bits(out, expected)
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(worker, range(4)):
            assert_bits(result, expected)


@pytest.mark.parametrize('n_ch,n_off', [(0, 3), (2, 0), (0, 0)])
def test_empty_grid(n_ch, n_off):
    out = np.empty((n_ch, n_off, 1), dtype=np.complex128)
    native.fold_grid([np.ones(10, dtype=np.complex128)], [0] * n_off,
                     [0] * n_off, [0] * n_ch, np.ones((10, 1), dtype=np.complex128),
                     out, False)


def test_integer_sequences_and_buffer_protocol():
    spectra, picks, shifts, tones, weight = inputs(7)
    shifts = [-71, 0, 69, 73, 2, -2]
    expected = reference(spectra, picks, shifts, tones, weight, False)
    out = np.empty_like(expected)
    native.fold_grid([memoryview(s) for s in spectra], np.array(picks),
                     np.array(shifts), np.array(tones), memoryview(weight),
                     memoryview(out), False)
    assert_bits(out, expected)


def test_fusion_changes_rounding():
    args = inputs(7)
    unfused = reference(*args, False)
    fused = reference(*args, True)
    assert np.any(unfused.view(np.uint64) != fused.view(np.uint64))
    for mode, expected in [(False, unfused), (True, fused)]:
        out = np.empty_like(expected)
        native.fold_grid(*args, out, mode)
        assert_bits(out, expected)


@pytest.fixture
def frontend():
    if not hasattr(native, 'sliding_magnitudes'):
        pytest.skip('front-end extension not built')
    return native


def _frontend_inputs():
    x = np.linspace(-1, 1, 1024)
    tables = [np.exp(-2j * np.pi * np.arange(31) / 31)]
    out = np.full((1, 1017), 17.)
    return x, 8, tables, out, 'hypot'


@pytest.mark.parametrize('target', ['audio', 'oscillator', 'out'])
@pytest.mark.parametrize('defect', ['dtype', 'strided', 'unaligned', 'rank', 'endian'])
def test_frontend_sliding_invalid_layout(frontend, target, defect):
    x, n, tables, out, mode = _frontend_inputs()
    original = {'audio': x, 'oscillator': tables[0], 'out': out}[target]
    if defect == 'dtype':
        bad = original.astype(np.complex64 if target == 'oscillator' else np.float32)
    elif defect == 'strided':
        bad = np.repeat(original, 2, axis=-1)[..., ::2]
    elif defect == 'unaligned':
        bad = np.ndarray(original.shape, dtype=original.dtype,
                         buffer=bytearray(original.nbytes + 1), offset=1)
        bad[:] = original
    elif defect == 'rank':
        bad = original.reshape((*original.shape, 1))
    else:
        bad = original.astype(original.dtype.newbyteorder('>'))
    if target == 'audio':
        x = bad
    elif target == 'oscillator':
        tables[0] = bad
    else:
        out = bad
    before = out.tobytes()
    with pytest.raises(ValueError):
        frontend.sliding_magnitudes(x, n, tables, out, mode)
    assert out.tobytes() == before


@pytest.mark.parametrize('defect', ['audio_overlap', 'oscillator_overlap', 'readonly',
                                  'zero_window', 'empty_table', 'output_shape', 'mode'])
def test_frontend_sliding_invalid_arguments(frontend, defect):
    x, n, tables, out, mode = _frontend_inputs()
    if defect == 'audio_overlap':
        out = x[:1017].reshape(1, -1)
    elif defect == 'oscillator_overlap':
        tables = [np.ones(1024, complex)]
        out = tables[0].view(float)[:1017].reshape(1, -1)
    elif defect == 'readonly':
        out.flags.writeable = False
    elif defect == 'zero_window':
        n = 0
    elif defect == 'empty_table':
        tables = [np.empty(0, complex)]
    elif defect == 'output_shape':
        out = out[:, :-1]
    else:
        mode = 'approximate'
    before = out.tobytes()
    with pytest.raises(ValueError):
        frontend.sliding_magnitudes(x, n, tables, out, mode)
    assert out.tobytes() == before


@pytest.mark.parametrize('defect', ['negative_start', 'end_start', 'stride', 'frames_overlap',
                                  'mask_overlap', 'outputs_overlap', 'skip', 'readonly'])
def test_frontend_frame_validation(frontend, defect):
    mark, space = np.ones(1000), np.zeros(1000)
    starts = np.array([0, 1, 2], np.int64)
    frames = np.full((3, 12), 37, np.uint8)
    masks = np.full((2, 2, 3), 37, np.uint8)
    skips = [1, 2]
    if defect == 'negative_start':
        starts[0] = -1
    elif defect == 'end_start':
        starts[0] = 1000
    elif defect == 'stride':
        starts = np.repeat(starts, 2)[::2]
    elif defect == 'frames_overlap':
        frames = mark.view(np.uint8)[:36].reshape(3, 12)
    elif defect == 'mask_overlap':
        masks = space.view(np.uint8)[:12].reshape(2, 2, 3)
    elif defect == 'outputs_overlap':
        masks = frames.ravel()[:12].reshape(2, 2, 3)
    elif defect == 'skip':
        skips[0] = 11
    else:
        masks.flags.writeable = False
    before = frames.tobytes(), masks.tobytes()
    with pytest.raises(ValueError):
        frontend.gather_frames(mark, space, starts, 2, 12, skips, frames, masks,
                               0x8408, 0xffff, 0xffff, True)
    assert (frames.tobytes(), masks.tobytes()) == before


@pytest.mark.parametrize('kind', ['nan', 'inf', 'overflow'])
def test_frontend_nonfinite_rejected(frontend, kind):
    x, n, tables, out, mode = _frontend_inputs()
    x[10] = {'nan': np.nan, 'inf': np.inf, 'overflow': np.finfo(float).max}[kind]
    if kind == 'overflow':
        x[11] = x[10]
        tables = [np.ones(1, complex)]
    assert frontend.sliding_magnitudes(x, n, tables, out, mode) is False
    mark = x
    space = -x
    env = np.zeros_like(x)
    assert frontend.eye_argmax(mark, space, env, np.array([0], np.int64),
                               1, 12, 2)[0] is False


@pytest.mark.parametrize('defect', ['bounds', 'overlap', 'readonly', 'shape', 'dtype'])
def test_frontend_slice_validation(frontend, defect):
    audio = np.zeros(256)
    bounds = np.array([[0, 128], [7, 143]], np.int64)
    tables = [np.ones(3, complex), np.ones(7, complex)]
    bits = np.full((2, 12), 37, np.uint8)
    if defect == 'bounds':
        bounds[1, 1] = 257
    elif defect == 'overlap':
        bits = audio.view(np.uint8)[:24].reshape(2, 12)
    elif defect == 'readonly':
        bits.flags.writeable = False
    elif defect == 'shape':
        bits = bits[:, :11]
    else:
        bounds = bounds.astype(np.int32)
    before = bits.tobytes()
    with pytest.raises(ValueError):
        frontend.slice_eyes(audio, bounds, 4, tables, 12, 3, bits, 'hypot')
    assert bits.tobytes() == before
