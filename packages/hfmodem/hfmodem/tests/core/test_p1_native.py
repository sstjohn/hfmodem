# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import asdict, is_dataclass
from math import gcd

import numpy as np
import pytest

from hfmodem.core import crc, p1_native
from hfmodem.shrike import onair, p1rx, pactor1, rxfront, session
from hfmodem.tests import evidence, native as native_helpers
from hfmodem.tests.kestrel import corpora


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if padded(actual.dtype):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual.tobytes() == expected.tobytes()


def padded(dtype):
    # x86-64 stores a long double's 80 bits in 16 bytes and leaves the rest unset.
    if dtype.kind not in 'fc':
        return False
    info = np.finfo(dtype)
    return info.nmant + info.nexp + 1 < 8 * dtype.itemsize // (2 if dtype.kind == 'c' else 1)


def tables():
    return [p1rx._lo(p1rx.FS // gcd(hz, p1rx.FS), hz)
            for hz in (0, 1400, 1600, 1783)]


def magnitudes_numpy(audio, n, oscillators):
    rows = []
    for one in oscillators:
        lo = np.tile(one, -(-audio.size // one.size))[:audio.size]
        acc = np.concatenate(([0.0], np.cumsum(audio * lo)))
        rows.append(np.abs(acc[n:] - acc[:-n]))
    return np.asarray(rows)


def frames_numpy(mark, space, starts, sps, nbytes, skips, variant='x25'):
    idx = starts[:, None] + np.arange(nbytes * 8) * sps
    bits = mark[idx] > space[idx]
    frames = np.packbits(bits, axis=1, bitorder='little')
    ok = np.empty((len(skips), 2, len(starts)), np.uint8)
    for i, skip in enumerate(skips):
        for invert in (0, 1):
            f = frames ^ (255 * invert)
            trailer = f[:, -2].astype(np.uint32) | (f[:, -1].astype(np.uint32) << 8)
            ok[i, invert] = crc.crc16_rows(f[:, skip:-2], variant) == trailer
    return frames, ok


def eye_numpy(mark, space, env, starts, sps, nbits, tail_bits):
    eye = starts[:, None] + np.arange(nbits) * sps
    tail = starts[:, None] + (nbits + np.arange(tail_bits)) * sps
    score = (np.abs(mark[eye] - space[eye]).sum(axis=1)
             - env[tail].sum(axis=1))
    return int(np.argmax(score))


@pytest.fixture
def native():
    return native_helpers.dsp()


@pytest.mark.parametrize('kind', ['random', 'zeros', 'cancellation', 'subnormal',
                                  'impulse', 'clipping'])
@pytest.mark.parametrize('n', [1, 7, 240, 480])
def test_sliding_magnitudes_match_numpy(native, kind, n):
    rng = np.random.default_rng(1111)
    x = rng.normal(size=5003)
    if kind == 'zeros':
        x[:] = np.resize([0., -0.], x.size)
    elif kind == 'cancellation':
        x[:] = np.resize([1., -1., 2**-52, -2**-52], x.size)
    elif kind == 'subnormal':
        x *= 1e-310
    elif kind == 'impulse':
        x[:] = 0
        x[[0, 239, 240, 4999, 5002]] = [1, -1, 1, -1, 1]
    elif kind == 'clipping':
        x = np.clip(x * 8, -1, 1)
    oscillators = tables()
    expected = magnitudes_numpy(x, n, oscillators)
    out = np.empty_like(expected)
    assert native.sliding_magnitudes(x, n, oscillators, out, p1_native.ABS_VARIANT)
    assert_bits(out, expected)


@pytest.mark.parametrize('variant', ['x25', 'kermit', 'ccitt-false'])
@pytest.mark.parametrize('nbytes', [3, 12, 24])
def test_gather_crc_both_polarities_and_skips(native, variant, nbytes):
    rng = np.random.default_rng(89)
    rows = rng.integers(0, 256, (18, nbytes), dtype=np.uint8)
    skips = [0, 1, nbytes - 2]
    for i, row in enumerate(rows):
        check = crc.crc16(row[skips[i % len(skips)]:-2].tobytes(), variant)
        row[-2:] = [check & 255, check >> 8]
        if i & 1:
            row ^= 255
    bits = np.unpackbits(rows, axis=1, bitorder='little').ravel()
    mark = np.repeat(bits.astype(float), 3)
    space = 1 - mark
    starts = np.arange(rows.shape[0], dtype=np.int64) * nbytes * 8 * 3
    expected, want_ok = frames_numpy(mark, space, starts, 3, nbytes, skips, variant)
    frames = np.empty_like(expected)
    ok = np.empty_like(want_ok)
    p = crc.CRC_VARIANTS[variant]
    assert native.gather_frames(mark, space, starts, 3, nbytes, skips, frames, ok,
                                0x8408 if p['refin'] else 0x1021,
                                p['init'], p['xorout'], p['refin'])
    assert_bits(frames, expected)
    assert_bits(ok, want_ok)
    assert np.count_nonzero(ok) >= len(rows)


@pytest.mark.parametrize('nbits', [1, 7, 8, 12, 96, 127, 128, 129, 192, 264])
@pytest.mark.parametrize('tail_bits', [0, 1, 2, 7, 9])
def test_eye_reduction_and_first_tie(native, nbits, tail_bits):
    rng = np.random.default_rng(991)
    size = (nbits + tail_bits) * 5 + 300
    mark = rng.normal(size=size) * rng.choice([1., 2**-40, 2**40], size)
    space = rng.normal(size=size)
    env = mark + space
    starts = np.arange(3, 201, 3, dtype=np.int64)
    expected = eye_numpy(mark, space, env, starts, 5, nbits, tail_bits)
    finite, got = native.eye_argmax(mark, space, env, starts, 5, nbits, tail_bits)
    assert finite and got == expected
    mark.fill(0)
    space.fill(-0.)
    env.fill(0)
    assert native.eye_argmax(mark, space, env, starts, 5, nbits, tail_bits) == (True, 0)


def freeze(value):
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if is_dataclass(value):
        return freeze(asdict(value))
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def memory_state(memory):
    return {baud: (lane.flipped, freeze(lane.copies))
            for baud, lane in memory._lanes.items()}


def read_window(x, memory):
    short = x[:round(.29 * p1rx.FS)]
    return freeze((
        p1rx.decode_p1_packets(x),
        p1rx.decode_p1_packets(x, breakin=True),
        rxfront.decode_expected_p1_packet(x, memory=memory),
        memory_state(memory),
        p1rx.packet_softs(x, baud=100),
        p1rx.packet_softs(x, baud=200),
        p1rx.cs_bits(short, .09, .12),
        p1rx.decode_control_signal(short, .09, .12),
        p1rx.cs_anchored(short, .09),
        p1rx.cs_head(short, .09),
        onair._anchored_answer(short, 0),
    ))


def replay(monkeypatch, windows):
    native_memory, numpy_memory = p1rx.PacketMemory(), p1rx.PacketMemory()
    for i, x in enumerate(windows):
        actual = read_window(x, native_memory)
        with monkeypatch.context() as patch:
            patch.setattr(p1_native, '_dsp', None)
            expected = read_window(x, numpy_memory)
        assert actual == expected, f'read window {i}'


@pytest.mark.parametrize('baud', [100, 200])
@pytest.mark.parametrize('breakin', [False, True])
def test_rendered_packets_and_memory_replay(monkeypatch, baud, breakin):
    render = pactor1.breakin_signal if breakin else pactor1.packet_signal
    clean = render(b'NATIVE', baud, lead_s=.09, tail_s=.2)
    rng = np.random.default_rng(1234)
    windows = [clean]
    for k in range(5):
        x = render(b'NATIVE', baud, invert=bool(k & 1), lead_s=.09, tail_s=.2)
        windows.append(x + rng.normal(0, .55, x.size))
    windows.extend([clean[:-800], np.zeros_like(clean), clean])
    replay(monkeypatch, windows)


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('invert', [False, True])
def test_rendered_control_replay(monkeypatch, index, invert):
    burst = pactor1.control_signal(index, invert=invert)
    x = np.concatenate([np.zeros(round(.09 * p1rx.FS)), burst,
                        np.zeros(round(.08 * p1rx.FS))])
    replay(monkeypatch, [x, x[17:], x[:-87]])


RECORDINGS = ('pos_pactor1_local.wav', 'pos_p1cs_ws8eoc.wav',
              'pos_p1_twosided_14110.wav', 'neg_noise_chatham.wav')


@pytest.mark.parametrize('name', RECORDINGS)
def test_recorded_read_replay(monkeypatch, name):
    path = evidence.require_input(corpora.REGRESS_FIXTURES / name)
    audio = p1rx.load_wav(str(path))
    width = round(1.25 * p1rx.FS)
    replay(monkeypatch, [audio[start:start + width]
                         for start in range(0, audio.size, width)])


def test_recorded_hold_replay(monkeypatch):
    paths = sorted({p for pattern in ('p1-ws8eoc-*/**/hold_*.wav',
                                     'p1-ws8eoc-*/**/hold_*.flac')
                    for p in evidence.CAPTURES.glob(pattern)})
    if not paths:
        pytest.skip(f'off-air hold windows absent: {evidence.CAPTURES}/p1-ws8eoc-*')
    replay(monkeypatch, [session.load_wav(str(evidence.require_input(p)), p1rx.FS)
                         for p in paths])


@pytest.mark.parametrize('name', RECORDINGS)
def test_recorded_kernel_outputs(native, name):
    path = evidence.require_input(corpora.REGRESS_FIXTURES / name)
    audio = p1rx.load_wav(str(path))
    oscillators = tables()[1:3]
    for start in (0, audio.size // 2 + 17, max(0, audio.size - 60001)):
        x = np.asarray(audio[start:start + 60000], dtype=float)
        for sps in (240, 480):
            if x.size < sps:
                continue
            expected = magnitudes_numpy(x, sps, oscillators)
            out = np.empty_like(expected)
            assert native.sliding_magnitudes(x, sps, oscillators, out, p1_native.ABS_VARIANT)
            assert_bits(out, expected)
            mark, space = out
            env = mark + space
            nbytes = 24 if sps == 240 else 12
            nbits, tail = nbytes * 8, 3
            stop = mark.size - (nbits + tail - 1) * sps
            if stop > 0:
                starts = np.unique(np.linspace(0, stop - 1, 71, dtype=np.int64))
                skips = [1, 3 if sps == 240 else 2]
                expected_frames, expected_ok = frames_numpy(
                    mark, space, starts, sps, nbytes, skips)
                frames = np.empty_like(expected_frames)
                ok = np.empty_like(expected_ok)
                assert native.gather_frames(mark, space, starts, sps, nbytes, skips,
                                            frames, ok, 0x8408, 0xffff, 0xffff, True)
                assert_bits(frames, expected_frames)
                assert_bits(ok, expected_ok)
                expected_eye = eye_numpy(mark, space, env, starts, sps, nbits, tail)
                assert native.eye_argmax(mark, space, env, starts, sps,
                                         nbits, tail) == (True, expected_eye)
            if x.size < 12 * sps:
                continue
            span = min(x.size, 21 * sps)
            origins = sorted({0, min(17, x.size - span), x.size - span})
            bounds = np.array([(origin, origin + span) for origin in origins], np.int64)
            expected_bits = []
            for lo, hi in bounds:
                padded = np.concatenate([x[lo:hi], np.zeros(tail * sps)])
                local_m, local_s = magnitudes_numpy(padded, sps, oscillators)
                starts = np.arange(local_m.size - (12 + tail - 1) * sps)
                origin = eye_numpy(local_m, local_s, local_m + local_s,
                                   starts, sps, 12, tail)
                idx = origin + np.arange(12) * sps
                expected_bits.append(local_m[idx] > local_s[idx])
            expected_bits = np.asarray(expected_bits, np.uint8)
            bits = np.empty_like(expected_bits)
            assert native.slice_eyes(x, bounds, sps, oscillators, 12, tail,
                                     bits, p1_native.ABS_VARIANT)
            assert_bits(bits, expected_bits)


@pytest.mark.parametrize('nbits', [8, 12, 96, 128, 192, 264])
def test_eye_argmax_resolves_rounding_close_scores(native, nbits):
    rng = np.random.default_rng(411)
    values = np.full(nbits, 2**-54)
    values[0] = 1.
    mark = np.concatenate([rng.permutation(values) for _ in range(257)])
    space = np.zeros_like(mark)
    starts = np.arange(257, dtype=np.int64) * nbits
    expected = eye_numpy(mark, space, space, starts, 1, nbits, 0)
    assert native.eye_argmax(mark, space, space, starts, 1, nbits, 0) == (True, expected)


def test_probe_selects_and_rejects(native):
    assert p1_native.probe(native) in ('hypot', 'fused', 'plain')
    assert p1_native.ABS_VARIANT is not None
    assert p1_native.NATIVE.startswith('native: yes')
    assert p1_native._select(None)[0] is None

    class Mismatch:
        sliding_magnitudes = staticmethod(lambda *args: False)
        gather_frames = staticmethod(lambda *args: False)

    with pytest.warns(RuntimeWarning, match='using NumPy') as warnings:
        mode, status = p1_native._select(Mismatch())
    assert mode is None and status.startswith('native: no')
    assert len(warnings) == 1


@pytest.mark.parametrize('defect', ['float32', 'longdouble', 'strided', 'unaligned', 'nonfinite'])
def test_adapter_numpy_fallback(monkeypatch, defect):
    rng = np.random.default_rng(71)
    x = rng.normal(size=1073)
    if defect == 'float32':
        x = x.astype(np.float32)
    elif defect == 'longdouble':
        x = x.astype(np.longdouble)
    elif defect == 'strided':
        x = np.repeat(x, 2)[::2]
    elif defect == 'unaligned':
        bad = np.ndarray(x.shape, dtype=float, buffer=bytearray(x.nbytes + 1), offset=1)
        bad[:] = x
        x = bad
    else:
        x[37] = np.nan
    oscillators = tables()[1:3]
    with np.errstate(all='ignore'):
        assert_bits(p1_native.magnitudes(x, 240, oscillators),
                    magnitudes_numpy(x, 240, oscillators))
    mark = np.abs(x)
    space = np.ones_like(mark)
    starts = np.array([0, 1, 17], dtype=np.int64)
    frames, ok = p1_native.gather_frames(mark, space, starts, 1, 12, [1, 2])
    expected, expected_ok = frames_numpy(mark, space, starts, 1, 12, [1, 2])
    assert_bits(frames, expected)
    assert_bits(ok, expected_ok.view(bool))
    with np.errstate(all='ignore'):
        assert p1_native.eye_argmax(mark, space, mark + space, starts, 1, 12, 3) == (
            eye_numpy(mark, space, mark + space, starts, 1, 12, 3))


def test_pure_python_and_noninteger_frequency(monkeypatch):
    x = np.random.default_rng(3).normal(size=1003).astype(np.float32)
    with monkeypatch.context() as patch:
        patch.setattr(p1_native, '_dsp', None)
        tones = p1rx._Tones(x, 240)
        assert_bits(tones.mark, np.abs(p1rx._sliding_bin(x, 240, p1rx.MARK)))
        assert_bits(tones.space, np.abs(p1rx._sliding_bin(x, 240, p1rx.SPACE)))
    tones = p1rx._Tones(x, 240, 1400.25, 1600.25)
    assert_bits(tones.mark, np.abs(p1rx._sliding_bin(x, 240, 1400.25)))
    assert_bits(tones.space, np.abs(p1rx._sliding_bin(x, 240, 1600.25)))


def test_combined_passes_preserve_data_before_breakin_and_baud_order():
    parts = [pactor1.packet_signal(b'ONE', 200),
             pactor1.breakin_signal(b'TWO', 100),
             pactor1.packet_signal(b'THREE', 100),
             pactor1.breakin_signal(b'FOUR', 200)]
    x = np.concatenate(parts)
    expected = [p1rx.decode_p1_packets(x), p1rx.decode_p1_packets(x, breakin=True)]
    assert all(expected)
    assert list(p1rx._packet_passes(x)) == expected


def test_grid_continues_after_rejected_zero_error_trial(monkeypatch):
    x = np.zeros(round(.29 * p1rx.FS))
    words = [p1rx.CS(0, 0, 0), p1rx.CS(1, 0, 1)]
    positions = []

    def trial_reads(audio, trials, dur):
        positions.extend(trials)
        return iter(words + [None] * (len(trials) - 2))

    visited = []

    def excess(audio, at):
        visited.append(at)
        return 0 if len(visited) == 1 else onair.ANSWER_CODEWORD_X + 1

    monkeypatch.setattr(p1rx, '_control_signal_trials', trial_reads)
    monkeypatch.setattr(onair, '_candidate_excess', excess)
    got = onair._anchored_answer(x, 0)
    assert got == (words[1], positions[1], onair.ANSWER_CODEWORD_X + 1)
    assert visited == positions[:2]
    at = max(0., onair.INLINK_READ_BAND[0])
    expected = []
    hi = min(onair.INLINK_READ_BAND[1], (x.size - .12 * p1rx.FS) / p1rx.FS)
    while at <= hi:
        expected.append(at)
        at += onair.INLINK_READ_HOP_S
    assert positions == expected


@pytest.mark.parametrize('tail', [0, 1, 2, 7])
def test_slice_local_batch_matches_individual_numpy_reads(native, tail):
    rng = np.random.default_rng(731)
    audio = rng.normal(size=20003)
    audio[733:1900] = 0
    bounds = np.array([[0, 5777], [17, 8231], [239, 12333], [14401, 20003]], np.int64)
    n, nbits = 240, 12
    oscillators = tables()[1:3]
    expected = []
    for lo, hi in bounds:
        x = np.concatenate([audio[lo:hi], np.zeros(tail * n)])
        mark, space = magnitudes_numpy(x, n, oscillators)
        starts = np.arange(mark.size - (nbits + tail - 1) * n)
        origin = eye_numpy(mark, space, mark + space, starts, n, nbits, tail)
        idx = origin + np.arange(nbits) * n
        expected.append(mark[idx] > space[idx])
    expected = np.asarray(expected, np.uint8)
    out = np.empty_like(expected)
    assert native.slice_eyes(audio, bounds, n, oscillators, nbits, tail, out,
                             p1_native.ABS_VARIANT)
    assert_bits(out, expected)


def test_control_trials_equal_original_individual_reads():
    rng = np.random.default_rng(177)
    audio = rng.normal(size=round(.29 * p1rx.FS))
    positions = [0., .001, .031, .09, .0999999999, .14, .19]
    assert list(p1rx._control_signal_trials(audio, positions, .12)) == [
        p1rx.decode_control_signal(audio, at, .12) for at in positions]


def test_control_trials_chunked_order_and_early_exit(native, monkeypatch):
    rng = np.random.default_rng(178)
    audio = rng.normal(size=round(.4 * p1rx.FS))
    positions = [i * .002 for i in range(3 * p1rx.CS_TRIAL_CHUNK + 3)]
    calls = []
    real = p1_native.slice_eyes

    def counted(audio, bounds, *args):
        calls.append(len(bounds))
        return real(audio, bounds, *args)

    monkeypatch.setattr(p1_native, 'slice_eyes', counted)
    expected = [p1rx.decode_control_signal(audio, at, .12) for at in positions]
    assert list(p1rx._control_signal_trials(audio, positions, .12)) == expected
    k = p1rx.CS_TRIAL_CHUNK
    assert calls == [k, k, k, 3]
    calls.clear()
    trials = p1rx._control_signal_trials(audio, positions, .12)
    assert next(trials) == expected[0]
    assert calls == [k]


def test_read_crossings_are_batched(native, monkeypatch):
    from collections import Counter

    counts = Counter()

    class Counted:
        def __getattr__(self, name):
            kernel = getattr(native, name)

            def call(*args):
                counts[name] += 1
                return kernel(*args)
            return call

    monkeypatch.setattr(p1_native, '_dsp', Counted())
    audio = np.random.default_rng(101).normal(size=round(1.25 * p1rx.FS))
    assert rxfront.decode_expected_p1_packet(audio, memory=p1rx.PacketMemory()) is None
    assert counts == {'sliding_magnitudes': 4, 'gather_frames': 2, 'eye_argmax': 2}
    counts.clear()
    p1rx.cs_bits(audio[:round(.29 * p1rx.FS)], .09, .12)
    assert counts == {'sliding_magnitudes': 1, 'eye_argmax': 1}
    counts.clear()
    onair._anchored_answer(audio[:round(.29 * p1rx.FS)], 0)
    assert counts == {'slice_eyes': 7}


@pytest.mark.parametrize('dtype', [np.complex128, np.longdouble, np.int16])
def test_tones_preserve_unsupported_audio_dtype(dtype):
    rng = np.random.default_rng(831)
    audio = (rng.normal(size=1033) * 100).astype(dtype)
    if dtype == np.complex128:
        audio += 1j * rng.normal(size=audio.size)
    tones = p1rx._Tones(audio, 240)
    # Values, not bytes: x86-64 stores a long double's 80 bits in 16 and leaves
    # the rest unset, so two equal arrays need not share their bytes.
    for got, want in ((tones.mark, np.abs(p1rx._sliding_bin(audio, 240, p1rx.MARK))),
                      (tones.space, np.abs(p1rx._sliding_bin(audio, 240, p1rx.SPACE)))):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('fractional', [False, True])
def test_tone_pair_sweep_is_batched_exactly(native, monkeypatch, fractional):
    audio = np.random.default_rng(91).normal(size=2041).astype(np.float32)
    pairs = [(1400, 1600), (1373, 1573), (1600, 1400)]
    if fractional:
        pairs[1] = (1373.25, 1573.25)
    calls = []

    class Counted:
        def sliding_magnitudes(self, *args):
            calls.append(args)
            return native.sliding_magnitudes(*args)

    monkeypatch.setattr(p1_native, '_dsp', Counted())
    monkeypatch.setattr(p1rx, 'TONE_PAIR_BATCH', 2)
    tones = list(p1rx._tone_pairs(audio, 240, iter(pairs)))
    assert len(tones) == len(pairs)
    assert len(calls) == (0 if fractional else 2)
    for tone, (mark, space) in zip(tones, pairs):
        expected_m = np.abs(p1rx._sliding_bin(audio, 240, mark))
        expected_s = np.abs(p1rx._sliding_bin(audio, 240, space))
        assert_bits(tone.mark, expected_m)
        assert_bits(tone.space, expected_s)
        assert_bits(tone.env, expected_m + expected_s)
        assert tone.n == expected_m.size


@pytest.mark.parametrize('variant', ['normal', 'longpath', 'robust'])
def test_connect_readers_differential(monkeypatch, variant):
    audio = pactor1.connect_signal('N1CALL', variant=variant, invert=True,
                                    lead_s=.05, tail_s=.05)

    def reads():
        return freeze((p1rx.decode_connect(audio),
                       p1rx.search_calls(audio, 'N1CALL'),
                       p1rx.decode_call_b(audio)))

    actual = reads()
    with monkeypatch.context() as patch:
        patch.setattr(p1_native, '_dsp', None)
        expected = reads()
    assert actual == expected
    assert actual[1]
    assert actual[2 if variant == 'robust' else 0] is not None


def test_unknown_crc_variant_has_numpy_error(monkeypatch):
    mark = np.ones(1000)
    space = np.zeros_like(mark)
    starts = np.array([0, 1], np.int64)
    args = mark, space, starts, 1, 12, [1]
    with pytest.raises(ValueError, match='unknown CRC variant') as actual:
        p1_native.gather_frames(*args, variant='unknown')
    monkeypatch.setattr(p1_native, '_dsp', None)
    with pytest.raises(ValueError, match='unknown CRC variant') as expected:
        p1_native.gather_frames(*args, variant='unknown')
    assert str(actual.value) == str(expected.value)


def test_assert_bits_ignores_padding_but_not_values():
    if not padded(np.dtype(np.longdouble)):
        pytest.skip('long double has no padding on this platform')
    a = np.linspace(1, 2, 5).astype(np.longdouble)
    b = a.copy()
    b.view(np.uint8).reshape(5, -1)[:, -1] ^= 0xFF
    assert a.tobytes() != b.tobytes()
    assert_bits(a, b)
    b[2] += 1
    with pytest.raises(AssertionError):
        assert_bits(a, b)
