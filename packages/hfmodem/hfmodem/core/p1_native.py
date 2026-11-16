# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import warnings

import numpy as np

from . import crc


def magnitudes_numpy(audio, n, tables):
    rows = []
    for table in tables:
        lo = np.tile(table, -(-audio.size // table.size))[:audio.size]
        acc = np.concatenate(([0.0], np.cumsum(audio * lo)))
        rows.append(np.abs(acc[n:] - acc[:-n]))
    return np.asarray(rows).reshape(len(tables), max(0, audio.size - n + 1))


def gather_frames_numpy(mark, space, starts, sps, nbytes, skips, variant='x25'):
    idx = starts[:, None] + np.arange(nbytes * 8) * sps
    bits = mark[idx] > space[idx]
    frames = bits.reshape(-1, nbytes, 8).view(np.uint8) @ (
        1 << np.arange(8)).astype(np.uint8)
    ok = np.empty((len(skips), 2, starts.size), dtype=bool)
    for polarity, rows in enumerate((frames, ~frames)):
        trailer = (rows[:, -1].astype(np.uint32) << 8) | rows[:, -2]
        for i, skip in enumerate(skips):
            ok[i, polarity] = crc.crc16_rows(rows[:, skip:-2], variant) == trailer
    return frames, ok


def eye_argmax_numpy(mark, space, env, starts, sps, nbits, tail):
    eye = starts[:, None] + np.arange(nbits) * sps
    end = starts[:, None] + (nbits + np.arange(tail)) * sps
    score = (np.abs(mark[eye] - space[eye]).sum(axis=1)
             - env[end].sum(axis=1))
    return int(np.argmax(score))


def slice_eyes_numpy(audio, bounds, n, tables, nbits, tail):
    bits = np.empty((len(bounds), nbits), dtype=np.uint8)
    for i, (lo, hi) in enumerate(bounds):
        x = np.concatenate([audio[lo:hi], np.zeros(tail * n)])
        mark, space = magnitudes_numpy(x, n, tables)
        starts = np.arange(mark.size - (nbits + tail - 1) * n)
        origin = eye_argmax_numpy(mark, space, mark + space, starts, n, nbits, tail)
        idx = origin + np.arange(nbits) * n
        bits[i] = mark[idx] > space[idx]
    return bits


def _array(value, dtype, ndim=1):
    return (isinstance(value, np.ndarray) and value.dtype == dtype
            and value.ndim == ndim and value.flags.c_contiguous
            and value.flags.aligned)


def magnitudes(audio, n, tables):
    if (_dsp is not None and ABS_VARIANT is not None
            and _array(audio, np.float64)
            and all(_array(t, np.complex128) for t in tables)):
        out = np.empty((len(tables), max(0, audio.size - n + 1)), dtype=np.float64)
        try:
            if _dsp.sliding_magnitudes(audio, n, tables, out, ABS_VARIANT):
                return out
        except (ValueError, BufferError):
            pass
    return magnitudes_numpy(audio, n, tables)


def gather_frames(mark, space, starts, sps, nbytes, skips, variant='x25'):
    if variant not in crc.CRC_VARIANTS:
        raise ValueError(f"unknown CRC variant {variant!r}; "
                         f"choose from {sorted(crc.CRC_VARIANTS)}")
    if (_dsp is not None and ABS_VARIANT is not None
            and _array(mark, np.float64) and _array(space, np.float64)
            and _array(starts, np.int64)):
        frames = np.empty((starts.size, nbytes), dtype=np.uint8)
        ok = np.empty((len(skips), 2, starts.size), dtype=np.uint8)
        p = crc.CRC_VARIANTS[variant]
        try:
            if _dsp.gather_frames(mark, space, starts, sps, nbytes, skips,
                                  frames, ok, 0x8408 if p['refin'] else 0x1021,
                                  p['init'], p['xorout'], p['refin']):
                return frames, ok.view(bool)
        except (ValueError, BufferError):
            pass
    return gather_frames_numpy(mark, space, starts, sps, nbytes, skips, variant)


def eye_argmax(mark, space, env, starts, sps, nbits, tail):
    if (_dsp is not None and ABS_VARIANT is not None
            and all(_array(a, np.float64) for a in (mark, space, env))
            and _array(starts, np.int64)):
        try:
            finite, index = _dsp.eye_argmax(mark, space, env, starts, sps, nbits, tail)
            if finite:
                return index
        except (ValueError, BufferError):
            pass
    return eye_argmax_numpy(mark, space, env, starts, sps, nbits, tail)


def slice_eyes(audio, bounds, n, tables, nbits, tail):
    if (_dsp is not None and ABS_VARIANT is not None
            and _array(audio, np.float64) and _array(bounds, np.int64, 2)
            and all(_array(t, np.complex128) for t in tables)):
        bits = np.empty((len(bounds), nbits), dtype=np.uint8)
        try:
            if _dsp.slice_eyes(audio, bounds, n, tables, nbits, tail, bits, ABS_VARIANT):
                return bits
        except (ValueError, BufferError):
            pass
    return slice_eyes_numpy(audio, bounds, n, tables, nbits, tail)


def _probe_cases():
    rng = np.random.default_rng(6101)
    tables = [np.exp(-2j * np.pi * f / 48000 * np.arange(p))
              for f, p in ((1400, 240), (1600, 30), (1373, 48000))]
    for size, n in ((1, 1), (7, 1), (31, 7), (241, 240), (1023, 480),
                    (4097, 240)):
        x = rng.normal(size=size)
        yield x, n, tables[:2]
    for kind in ('zero', 'subnormal', 'cancel', 'impulse', 'clip', 'padding'):
        x = rng.normal(size=1031)
        if kind == 'zero':
            x[:] = 0
            x[::2] = -0.0
        elif kind == 'subnormal':
            x *= 1e-310
        elif kind == 'cancel':
            x = np.resize(np.array([1., -1., 1e-15, -1e-15]), x.size)
        elif kind == 'impulse':
            x[:] = 0
            x[[0, 239, 240, 479, 1030]] = [1, -1, .5, -.5, 1]
        elif kind == 'clip':
            x = np.clip(x * 4, -1, 1)
        else:
            x[:173] = 0
            x[-251:] = 0
        yield x, 240, tables


def _probe_frames(kernel):
    rng = np.random.default_rng(6102)
    for variant, p in crc.CRC_VARIANTS.items():
        for nbytes, skips in ((3, [0, 1]), (12, [1, 2]), (24, [1, 3])):
            rows = rng.integers(0, 256, (11, nbytes), dtype=np.uint8)
            for i, skip in enumerate(skips):
                for polarity in (0, 1):
                    at = 2 * i + polarity
                    value = crc.crc16(rows[at, skip:-2].tobytes(), variant)
                    rows[at, -2:] = [value & 255, value >> 8]
                    if polarity:
                        rows[at] = ~rows[at]
            bits = np.unpackbits(rows, axis=1, bitorder='little').ravel()
            mark = bits.astype(float)
            space = 1 - mark
            mark[-1] = space[-1]
            starts = np.arange(rows.shape[0], dtype=np.int64) * nbytes * 8
            expected, expected_ok = gather_frames_numpy(
                mark, space, starts, 1, nbytes, skips, variant)
            frames = np.empty_like(expected)
            ok = np.empty(expected_ok.shape, np.uint8)
            if not kernel(mark, space, starts, 1, nbytes, skips, frames, ok,
                          0x8408 if p['refin'] else 0x1021,
                          p['init'], p['xorout'], p['refin']):
                return False
            if (frames.tobytes() != expected.tobytes()
                    or ok.tobytes() != expected_ok.tobytes()):
                return False
    return True


def _probe_eye(kernel):
    rng = np.random.default_rng(6103)
    for nbits in (1, 7, 8, 12, 96, 128, 129, 192, 257):
        for tail in (0, 1, 2, 7, 9):
            size = (nbits + tail) * 3 + 257
            starts = np.array([0, 1, 7, 91, 122, 256], dtype=np.int64)
            for scale in (0, 1e-310, 1, 1e100):
                m, s = rng.random((2, size)) * scale
                env = m + s
                expected = eye_argmax_numpy(m, s, env, starts, 3, nbits, tail)
                finite, actual = kernel(m, s, env, starts, 3, nbits, tail)
                if not finite or actual != expected:
                    return False
    # Permutations have equal exact sums; only the reduction's rounding picks
    # a winner, so a sequential sum cannot pass on an otherwise wide-open eye.
    rng = np.random.default_rng(411)
    for nbits in (8, 12, 96, 128, 192, 264):
        values = np.full(nbits, 2**-54)
        values[0] = 1.
        mark = np.concatenate([rng.permutation(values) for _ in range(257)])
        space = np.zeros_like(mark)
        starts = np.arange(257, dtype=np.int64) * nbits
        expected = eye_argmax_numpy(mark, space, space, starts, 1, nbits, 0)
        finite, actual = kernel(mark, space, space, starts, 1, nbits, 0)
        if not finite or actual != expected:
            return False
    return True


def _probe_slices(dsp, mode):
    rng = np.random.default_rng(6104)
    tables = [np.exp(-2j * np.pi * f / 48000 * np.arange(p))
              for f, p in ((1400, 240), (1600, 30))]
    bounds = np.array([[0, 31], [1, 32], [37, 201], [239, 400]], dtype=np.int64)
    for scale in (0, 1e-310, 1):
        audio = rng.normal(size=401) * scale
        audio[::2] *= -1
        for tail in (0, 2, 7):
            args = audio, bounds, 2, tables, 12, tail
            expected = slice_eyes_numpy(*args)
            out = np.empty_like(expected)
            if (not dsp.slice_eyes(*args, out, mode)
                    or out.tobytes() != expected.tobytes()):
                return False
    return True


def probe(dsp):
    with np.errstate(all='ignore'):
        try:
            if not _probe_frames(dsp.gather_frames) or not _probe_eye(dsp.eye_argmax):
                return None
            cases = [(args, magnitudes_numpy(*args)) for args in _probe_cases()]
            for mode in ('hypot', 'fused', 'plain'):
                for args, expected in cases:
                    out = np.empty_like(expected)
                    if (not dsp.sliding_magnitudes(*args, out, mode)
                            or out.tobytes() != expected.tobytes()):
                        break
                else:
                    if _probe_slices(dsp, mode):
                        return mode
        except Exception:
            return None
    return None


def _select(dsp):
    if dsp is None or not hasattr(dsp, 'sliding_magnitudes'):
        return None, 'native: no (extension not built)'
    mode = probe(dsp)
    if mode is None:
        status = 'native: no (extension does not match this NumPy)'
        warnings.warn(status + '; using NumPy front end', RuntimeWarning, stacklevel=2)
        return None, status
    return mode, f'native: yes (Rust front end, {mode} abs)'


try:
    from . import _dsp
except ImportError:
    _dsp = None

ABS_VARIANT, NATIVE = _select(_dsp)
