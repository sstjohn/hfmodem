# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import warnings

import numpy as np

STEP = 10


def fold_numpy(spectra, picks, shifts, tones, weight, out):
    width = weight.shape[1]
    nfft = STEP * width
    images = {}
    for pick in set(picks):
        spectrum = spectra[pick]
        images[pick] = (np.concatenate([spectrum, spectrum]),
                        np.tile(np.conj(np.roll(spectrum[::-1], 1)), 2))
    for c, tone in enumerate(tones):
        for i, (pick, shift) in enumerate(zip(picks, shifts)):
            twice, mirror = images[pick]
            a, b = (shift + tone) % nfft, (tone - shift) % nfft
            z = twice[a:a + nfft] + mirror[b:b + nfft]
            out[c, i] = np.sum(z.reshape(STEP, width) * weight, axis=0)


def _probe_cases():
    rng = np.random.default_rng(6100)
    for width, scale in ((160, 1), (480, 1), (160, 0), (160, 1e-310)):
        def values(shape, factor):
            a = np.empty(shape, complex)
            a.real = rng.uniform(-.125, .125, shape) * factor
            a.imag = rng.uniform(-.125, .125, shape) * factor
            return a
        spectra = [values(STEP * width, scale) for _ in range(3)]
        weight = values((STEP, width), 1)
        yield (spectra, [0, 2, 1, 0], [-21*width-1, 0, 10*width-1, 31*width+3],
               [width, 2*width + 7], weight)


def probe(kernel):
    # The subnormal case must not trip a caller's np.seterr at import.
    with np.errstate(all='ignore'):
        return _probe(kernel)


def _probe(kernel):
    cases = []
    for args in _probe_cases():
        expected = np.empty((len(args[3]), len(args[1]), args[4].shape[1]), complex)
        fold_numpy(*args, expected)
        cases.append((args, expected))
    for fused in (False, True):
        for args, expected in cases:
            out = np.empty_like(expected)
            try:
                kernel(*args, out, fused)
            except Exception:
                break
            if out.tobytes() != expected.tobytes():
                break
        else:
            return fused
    return None


def _select(kernel):
    if kernel is None:
        return fold_numpy, 'native: no (extension not built)'
    fused = probe(kernel)
    if fused is None:
        status = 'native: no (extension does not match this NumPy)'
        warnings.warn(status + '; using NumPy grid fold', RuntimeWarning, stacklevel=2)
        return fold_numpy, status

    def fold(spectra, picks, shifts, tones, weight, out):
        # NumPy pairwise-sums a singleton column, complex64 adds round earlier, and
        # a non-finite result may carry NaN signs NumPy's would not.
        if (weight.shape[1] == 1 or any(s.dtype != np.complex128 for s in spectra)
                or not kernel(spectra, picks, shifts, tones, weight, out, fused)):
            fold_numpy(spectra, picks, shifts, tones, weight, out)

    mode = 'fused' if fused else 'plain'
    return fold, f'native: yes (Rust grid fold, {mode} multiply)'


try:
    from ._dsp import fold_grid as _kernel
except ImportError:
    _kernel = None

fold_grid, NATIVE = _select(_kernel)
