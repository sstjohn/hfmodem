# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import fields
from pathlib import Path
from struct import pack
from time import perf_counter

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import p3frame, p3rx, placement, rx, spec
from hfmodem.shrike.p3rx import FS, HEADER_COMBS, HEADER_FIT
from hfmodem.tests.shrike.archive import requires_ws8eoc_p3

FX = Path(__file__).parent / 'fixtures'


def _reference_header_diffs(Z: dict, at: int, tones: tuple[int, ...],
                  offsets: tuple[int, ...], sps: int, delay: int) -> dict | None:
    """One header block's unit differentials, each carrier read on its own clock.

    None if the block runs off the end of the baseband. `offsets` is rank-indexed
    beside `tones`, so a level whose comb is staggered is sampled where its
    carriers actually are.
    """
    idx = at + np.arange(p3frame.HEADER_SYMBOLS + 1) * sps + delay
    diffs = {}
    for cn, off in zip(tones, offsets):
        z = Z[cn]
        j = idx + off
        if j[0] < 0 or j[-1] >= z.size:
            return None
        y = z[j]
        d = y[1:] * np.conj(y[:-1])
        diffs[cn] = d / (np.abs(d) + 1e-12)
    return diffs


def _reference_header_anchors(Z: dict, span: int, *, fs: int = FS,
                   step: int | None = None) -> list[p3frame.PacketHeader]:
    """Every packet header block in `span` samples of already-filtered baseband.

    EACH SPEED LEVEL'S COMB IS SCORED ON ITS OWN, because the sweep does not yet
    know which level it is looking at and an average over channels a level never
    lights is an average over noise -- `HEADER_COMBS` carries the arithmetic and
    the measurement. The gate is the same 0.80 for every one of them.

    This is acquisition by known codeword rather than by envelope, and it is what
    replaced a timing search. An envelope places a burst to within a few symbols
    and leaves the rest to the CRC; the header block places it exactly, and says
    which speed levels to try, whether the cycle is long, and which way the
    carrier swap fell -- none of which an envelope knows and the last two of
    which no amount of searching recovers.
    """
    sps = fs // 100
    step = step or sps // 4
    delay = (rx._pulse(sps).size - 1) // 2
    need = p3frame.DATA_OFFSET * sps + delay
    tones = tuple(Z)
    flat = (0,) * len(tones)
    out: list[p3frame.PacketHeader] = []
    for at in range(0, max(0, span - need), step):
        diffs = _reference_header_diffs(Z, at, tones, flat, sps, delay)
        if diffs is None:
            break
        corrs = p3frame.constant_header_corrs(diffs)
        fit, rot = max((p3frame.constant_header_fit(corrs, comb)
                        for comb in HEADER_COMBS), key=lambda f: f[0])
        if fit >= HEADER_FIT:
            vh, _, _, swapped = p3frame.read_variable_header(diffs)
            out.append(p3frame.PacketHeader(at, vh, fit, rot, swapped))
    # One packet lights every position within half a symbol of its own, so keep
    # the best of each cluster rather than decoding the same field four times.
    peaks: list[p3frame.PacketHeader] = []
    for h in sorted(out, key=lambda h: -h.fit):
        if all(abs(h.at - p.at) > sps for p in peaks):
            peaks.append(h)
    return sorted(peaks, key=lambda h: h.at)


def _assert_header_lists(actual, expected):
    assert len(actual) == len(expected)
    for new, reference in zip(actual, expected):
        assert type(new) is type(reference)
        for field in fields(reference):
            value, old = getattr(new, field.name), getattr(reference, field.name)
            assert type(value) is type(old), field.name
            if isinstance(value, np.generic):
                assert value.tobytes() == old.tobytes(), field.name
            elif isinstance(value, float):
                assert pack('!d', value) == pack('!d', old), field.name
            else:
                assert value == old, field.name


def _assert_anchors(Z, span, monkeypatch, **kwargs):
    header = p3frame.PacketHeader
    read = p3frame.read_variable_header
    candidates, readings = [], []

    def packet(*args):
        result = header(*args)
        candidates.append(result)
        return result

    def variable(diffs):
        readings.append(diffs)
        return read(diffs)

    with monkeypatch.context() as patch:
        patch.setattr(p3frame, 'PacketHeader', packet)
        patch.setattr(p3frame, 'read_variable_header', variable)
        new = p3rx.header_anchors(Z, span, **kwargs)
        new_candidates, new_readings = candidates[:], readings[:]
        candidates.clear()
        readings.clear()
        reference = _reference_header_anchors(Z, span, **kwargs)
    _assert_header_lists(new, reference)
    # Pin all candidates before clustering, including their fit and rotation.
    _assert_header_lists(new_candidates, candidates)
    assert len(new_readings) == len(readings)
    for actual, expected in zip(new_readings, readings):
        assert list(actual) == list(expected)
        for c in actual:
            assert actual[c].dtype == expected[c].dtype
            assert actual[c].shape == expected[c].shape
            assert actual[c].tobytes() == expected[c].tobytes()
    return new


def _basebands(audio, fs=FS):
    pulse = rx._pulse(fs // 100)
    return {c: rx._baseband(audio, c, fs, pulse) for c in range(spec.N_CHANNELS)}


@pytest.mark.parametrize('sl,long_cycle', [(1, False), (2, False), (3, False),
                                          (4, False), (5, False), (6, False),
                                          (3, True), (6, True)])
@pytest.mark.parametrize('snr', [0, 12, 30])
@pytest.mark.parametrize('swapped', [False, True])
def test_rendered_header_anchors_are_identical(sl, long_cycle, snr, swapped, monkeypatch):
    path = (placement.LONG_PATHS if long_cycle else placement.SPEED_PATHS)[sl]
    payload = bytes(0x41 + i % 26 for i in range(path.crc_bytes - 3))
    body = placement.link_packet(sl, payload, 0x21, swapped=swapped, long_cycle=long_cycle)
    audio = np.concatenate([np.zeros(FS // 5), body, np.zeros(FS // 5)])
    sigma = float(np.sqrt(np.mean(np.asarray(body, np.float64) ** 2))) / 10 ** (snr / 20)
    audio += np.random.default_rng(37).normal(0, sigma, audio.size)
    heads = _assert_anchors(_basebands(audio), audio.size, monkeypatch)
    if sl >= 3 and snr >= 12:
        assert heads
        assert any(h.swapped == swapped for h in heads)


@requires_ws8eoc_p3
@pytest.mark.parametrize('recording', ['ws8eoc-0908-p3-head-4.wav',
                                        'ws8eoc-0908-p3-head-5.wav'])
def test_recorded_header_anchors_are_identical(recording, monkeypatch):
    fs, audio = wavfile.read(FX / recording)
    audio = audio.astype(np.float64) / (np.iinfo(audio.dtype).max + 1)
    if audio.ndim > 1:
        audio = audio.mean(1)
    _assert_anchors(_basebands(audio, fs), audio.size, monkeypatch, fs=fs)


def test_noise_only_header_anchors_are_identical(monkeypatch):
    audio = np.random.default_rng(57).normal(0, 0.1, 2 * FS)
    assert _assert_anchors(_basebands(audio), audio.size, monkeypatch) == []


@pytest.mark.parametrize('size', [0, 100, 4770, 4771, 4890, 4891, 7000])
@pytest.mark.parametrize('span', [0, 4800, 15000])
def test_header_scan_stops_at_first_incomplete_block(size, span, monkeypatch):
    rng = np.random.default_rng(123)
    Z = {c: rng.normal(size=size + c * 3) + 1j * rng.normal(size=size + c * 3)
         for c in range(spec.N_CHANNELS)}
    _assert_anchors(Z, span, monkeypatch)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('tones', [tuple(range(18)), (5, 12), (0, 5, 12, 17),
                                    (17, 12, 5, 0)])
@pytest.mark.parametrize('zero', [False, True])
def test_every_position_has_identical_scores_and_differentials(dtype, tones, zero, monkeypatch):
    rng = np.random.default_rng(707)
    Z = {}
    for c in tones:
        z = np.zeros(20000, complex) if zero else (rng.normal(size=20000)
                                                   + 1j * rng.normal(size=20000))
        Z[c] = z.astype(dtype)[::2]
    # Open the gate to pin every position, rather than only detected peaks.
    monkeypatch.setattr(p3rx, 'HEADER_FIT', 0.)
    monkeypatch.setitem(_reference_header_anchors.__globals__, 'HEADER_FIT', 0.)
    _assert_anchors(Z, 8500, monkeypatch)


def test_empty_basebands_have_no_anchors(monkeypatch):
    assert _assert_anchors({}, FS, monkeypatch) == []
    assert p3rx.header_anchors({}, 10**30) == []


@pytest.mark.parametrize('size', [0, 100, 4770, 4771, 4890, 4891, 7000])
@pytest.mark.parametrize('step', [None, 37, 300, -1, 2**62])
def test_oversized_span_is_bounded_by_channel_lengths(size, step, monkeypatch):
    Z = {0: np.zeros(size + 17, complex), 5: np.zeros(size, complex)}
    _assert_anchors(Z, 10**30, monkeypatch, step=step)


@pytest.mark.parametrize('fs', [12000, 48000])
@pytest.mark.parametrize('step', [None, 0, 37, 300, -1])
def test_header_scan_rates_and_steps_are_identical(fs, step, monkeypatch):
    rng = np.random.default_rng(808)
    size = 12 * (fs // 100)
    Z = {c: rng.normal(size=size + c) + 1j * rng.normal(size=size + c)
         for c in range(spec.N_CHANNELS)}
    monkeypatch.setattr(p3rx, 'HEADER_FIT', 0.)
    monkeypatch.setitem(_reference_header_anchors.__globals__, 'HEADER_FIT', 0.)
    _assert_anchors(Z, fs, monkeypatch, fs=fs, step=step)


@pytest.mark.wall_clock
def test_header_anchors_batch_the_position_grid():
    rng = np.random.default_rng(2026)
    Z = {c: rng.normal(size=FS) + 1j * rng.normal(size=FS) for c in range(18)}
    functions = (p3rx.header_anchors, _reference_header_anchors)
    for function in functions:
        function(Z, FS)
    times = [[], []]
    for repeat in range(11):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = perf_counter()
            functions[index](Z, FS)
            times[index].append(perf_counter() - start)
    new, reference = map(min, times)
    print(f'header anchors: new={new:.6f}s reference={reference:.6f}s ratio={new / reference:.4f}')
    assert new <= reference / 2


def test_long_basebands_keep_the_fused_complex_multiply(monkeypatch):
    # Thousands of positions: the batch rounds as the per-position loop did only
    # while its operands stay contiguous (strided, macOS dropped the FMA).
    rng = np.random.default_rng(400)
    size = 400_000
    Z = {c: (rng.normal(size=size) + 1j * rng.normal(size=size)) * np.exp(rng.normal(size=size))
         for c in range(spec.N_CHANNELS)}
    monkeypatch.setattr(p3rx, 'HEADER_FIT', 0.)
    monkeypatch.setitem(_reference_header_anchors.__globals__, 'HEADER_FIT', 0.)
    _assert_anchors(Z, size, monkeypatch)
