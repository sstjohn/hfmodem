# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.kestrel.rx.control_schedule import ControlSchedule, HardScores, SoftScores
from hfmodem.kestrel.rx.bw500_features import FeatureTrack


@pytest.mark.parametrize('offset', [0., -.49, .23, .31])
def test_spectra_are_exact_across_fft_batch_partitions(monkeypatch, offset):
    windows = np.random.default_rng(94).normal(0, .01, (257, 256))
    attributes = ('bins', 'clear', 'residual', 'power', 'pair_floor', 'denominator', 'live')
    expected = ControlSchedule(offset=offset)
    expected.live[:] = False
    expected._spectra(windows, np.arange(len(windows)))
    original = np.fft.fft if offset else np.fft.rfft
    calls = []

    def fft(block, *args, **kwargs):
        calls.append(len(block))
        return original(block, *args, **kwargs)

    monkeypatch.setattr(np.fft, 'fft' if offset else 'rfft', fft)
    for size in (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                 63, 64, 65, 127, 128, 129, 255, 256):
        schedule = ControlSchedule(offset=offset)
        schedule.live[:] = False
        calls.clear()
        logical_rows = []
        for first in range(0, len(windows), size):
            stop = min(first + size, len(windows))
            logical_rows.append(stop - first)
            schedule._spectra(windows[first:stop], np.arange(first, stop))
        for attr in attributes:
            np.testing.assert_array_equal(getattr(schedule, attr)[:len(windows)],
                                          getattr(expected, attr)[:len(windows)])
        assert all(0 <= actual - logical <= 7
                   for actual, logical in zip(calls, logical_rows, strict=True))
        assert schedule.fft_rows == sum(calls)
        assert schedule.fft_rows <= len(windows) + 7 * len(calls)


@pytest.mark.parametrize('offset', [0., -.49, .23, .49])
@pytest.mark.parametrize('layout', [(1024,), (1, 511, 73, 439), (1023, 17, 700, 113)])
def test_rows_match_independent_scalar_spectrum_and_raw_power(offset, layout):
    rng = np.random.default_rng(65)
    x = rng.normal(0, .0003, 16384)
    x[3500:13000] += .02 * np.sin(2 * np.pi * 65.23 * np.arange(9500) / 2048)
    front = FeatureTrack(stage='front')
    schedule = ControlSchedule(origin=117, generation=6, offset=offset)
    low = []
    position = 0
    iteration = 0
    delivered = []
    while position < len(x):
        block = x[position:position + layout[iteration % len(layout)]]
        front.feed(block)
        low.extend(front.last_low)
        first, stop = schedule.feed(front.last_low, block)
        delivered.extend(range(first, stop))
        available = min(position + len(block), front.samples)
        assert stop == max(0, (available - 2112) // 32 + 1)
        position += len(block)
        iteration += 1
    low = np.asarray(low)
    assert delivered == list(range(schedule.transforms))
    assert schedule.transforms == (len(x) - 2112) // 32 + 1
    win = np.hanning(2048)[::8] * np.exp(-2j * np.pi * offset * np.arange(256) / 256)
    for row in delivered:
        slot = row % schedule.capacity
        start = row * 32
        spectrum = abs(np.fft.fft(low[start // 8 + 8:start // 8 + 264] * win)) * 8
        expected_power = spectrum[29:99] ** 2
        np.testing.assert_allclose(schedule.power[slot], expected_power, atol=1e-11, rtol=2e-12)
        floor = np.partition(schedule.power[slot], 67)[:67].mean() + 1e-30
        assert schedule.pair_floor[slot] == floor
        np.testing.assert_allclose(schedule.denominator[slot], expected_power.sum(), atol=1e-11)
        mag = spectrum[29:99].copy()
        peaks = []
        for k in range(3):
            j = int(mag.argmax())
            assert schedule.bins[slot, k] == 29 + j
            peaks.append(mag[j])
            if k == 0:
                left, mid, right = spectrum[j + 28:j + 31]
                residual = 2 * (right - left) / (left + 2 * mid + right + 1e-20)
                assert schedule.residual[slot] == pytest.approx(residual, abs=1e-11)
            mag[max(0, j - 1):j + 2] = 0
        expected_clear = 20 * np.log10((np.array(peaks[:2]) + 1e-20) / (np.array(peaks[1:]) + 1e-20))
        np.testing.assert_allclose(schedule.clear[slot], expected_clear, atol=1e-10)
        expected_live = 20 * np.log10(np.sqrt(np.mean(x[start:start + 2048] ** 2)) + 1e-15) > -50
        assert schedule.live[slot] == expected_live


def test_ring_wrap_and_generation_reset():
    schedule = ControlSchedule()
    x = np.zeros(1024)
    low = np.zeros(128)
    end = 0
    for _ in range(300):
        first, end = schedule.feed(low, x)
        slots = schedule.interval(first, end)
        np.testing.assert_array_equal(schedule.tags[slots], np.arange(first, end))
    assert end > schedule.capacity
    assert schedule.delivered == schedule.transforms == end
    with pytest.raises(RuntimeError, match='history'):
        schedule.interval(0, 1)
    slots = schedule.interval(end - schedule.capacity, end)
    assert len(slots) == schedule.capacity
    assert not np.any(schedule.live[slots])
    replacement = ControlSchedule(origin=307200, generation=1, offset=.25)
    assert replacement.transforms == replacement.delivered == replacement.next_row == 0
    assert np.all(replacement.tags == -1)


def test_causal_first_row_and_explicit_batch_failure():
    schedule = ControlSchedule()
    assert schedule.feed(np.zeros(128), np.zeros(1024)) == (0, 0)
    assert schedule.feed(np.zeros(128), np.zeros(1024)) == (0, 0)
    assert schedule.feed(np.zeros(7), np.zeros(56)) == (0, 0)
    assert schedule.feed(np.zeros(1), np.zeros(8)) == (0, 1)
    with pytest.raises(ValueError, match='1024'):
        schedule.feed(np.zeros(256), np.zeros(2048))



def source(n, seed=9):
    s = ControlSchedule()
    r = np.random.default_rng(seed)
    s.next_row = n
    slots = np.arange(n) % s.capacity
    s.tags[slots] = np.arange(n)
    s.bins[slots, 0] = r.integers(29, 99, n)
    s.clear[slots, 0] = r.choice([1., 2., 4.], n)
    s.power[slots] = r.uniform(.01, 4., (n, 70))
    s.denominator[slots] = s.power[slots].sum(1)
    return s


def scalar_fit(s, tones, pre, first, stop, low, high):
    best = None
    for shift in range(-2, 3):
        for start in range(low, high + 1):
            c = m = 0
            for k in range(pre, len(tones)):
                row = round((start * 32 + k * 2048 + 16) / 32)
                if first <= row < stop and s.clear[row % 8192, 0] >= 2:
                    c += 1
                    m += s.bins[row % 8192, 0] == tones[k] + shift
            rank = (m >= 13 or (c >= 8 and m == c), m, m - c)
            if best is None or rank > best[0]:
                best = rank, start, shift
    return best[1:]


@pytest.mark.parametrize('pre', [0, 1, 3])
def test_incremental_entry_expiry_and_aliases(pre):
    s = source(1000)
    tones = [40, 46, 53, 64, 89]
    a = HardScores(tones, pre)
    for first, stop in [(0, 17), (0, 129), (17, 211), (82, 399), (193, 713)]:
        a.advance(s, first, stop)
        low, high = first - 256, stop - 1
        assert a.best(low, high) == scalar_fit(s, tones, pre, first, stop, low, high)
        for start in range(low, high + 1):
            rows = [round((start * 32 + k * 2048 + 16) / 32) for k in range(pre, 5)]
            valid = [r for r in rows if first <= r < stop and s.clear[r % 8192, 0] >= 2]
            assert a.counts[start % 8192] == len(valid)
            for shift in range(-2, 3):
                expect = sum(first <= row < stop and s.clear[row % 8192, 0] >= 2
                             and s.bins[row % 8192, 0] == tones[k] + shift
                             for k, row in zip(range(pre, 5), rows))
                assert a.matches[start % 8192, shift + 2] == expect
    before = a.entries, a.expiries
    a.advance(s, 193, 713)
    assert before == (a.entries, a.expiries)


def test_soft_entry_expiry_and_rank():
    s = source(1000)
    tones = [(40, 49), (60, 63), (60, 63), (87, 90)]
    a = SoftScores(tones, -1)
    for first, stop in [(0, 100), (0, 400), (103, 663), (301, 1000)]:
        a.advance(s, first, stop)
        expected = []
        for start in range(first, stop - 192):
            values = []
            for k, pair in enumerate(tones):
                row = start + k * 64
                p = sum(s.power[row, t - 1 - 29] for t in pair)
                values.append(np.log(max(70 * p / s.denominator[row], 1e-9) / 2))
            expected.append(sum(values) / 4)
        score, at = a.best(first, stop)
        if expected:
            assert score == pytest.approx(max(expected), abs=1e-13)
            assert at == first + np.argmax(expected)
        else:
            assert score == 0


def test_ring_wrap_retains_all_eligible_hypotheses():
    s = source(8192)
    a = HardScores([40, 45, 55], 1)
    a.advance(s, 0, 4000)
    a.advance(s, 3000, 8000)
    for first in range(8192, 16000, 32):
        rows = np.arange(first, first + 32)
        s.tags[rows % 8192] = rows
        s.bins[rows % 8192, 0] = 45
        s.clear[rows % 8192, 0] = 3
        s.next_row = first + 32
        if first == 8192:
            a.advance(s, 6000, 8192)
        a.advance(s, first - 2000, first + 32)
    assert a.best(14500, 15000) == scalar_fit(s, [40, 45, 55], 1, a.first, a.stop, 14500, 15000)


def test_pair_accumulators_match_unchanged_array_rules():
    from hfmodem.kestrel.rx.control_schedule import PairScores
    from hfmodem.kestrel.vara import vara_arq as v
    s = source(1800)
    rng = np.random.default_rng(21)
    s.bins[:1800, 1:] = rng.integers(29, 99, (1800, 2))
    s.clear[:1800, 1] = rng.choice([1., 2., 5.], 1800)
    s.live[:1800] = rng.choice([False, True, True], 1800)
    # A plateau crossing delivery boundaries, including a crowded middle pair.
    for k, pair in enumerate(v._ACK_MATCH):
        rows = np.arange(500, 580) + v._ACK_MATCH_OFF[k]
        s.bins[rows, :2] = pair
        s.live[rows] = True
        s.clear[rows] = [1., 3.]
    a = PairScores('ack', v._ACK_MATCH, v._ACK_MATCH_OFF)
    for first, stop in [(0, 530), (0, 650), (200, 901), (517, 1100), (800, 1800)]:
        a.advance(s, first, stop)
        got = a.held(first, stop, v._ACK_MIN_CARRIERS, v._ACK_MAX_CROWDED)
        want = v._ack_held(s.bins[first:stop], s.clear[first:stop], s.live[first:stop])
        np.testing.assert_array_equal(got, want)
        assert v._widest(got) == v._widest(want)


def test_grid_rebuild_matches_fresh_generation():
    from hfmodem.kestrel.rx.bw500_features import FeatureTrack
    from hfmodem.kestrel.rx.control_track import ControlTrack
    x = np.random.default_rng(45).normal(0, .1, 330752)
    front = FeatureTrack(stage='front')
    retained = ControlTrack()
    old, fresh = ControlSchedule(), ControlSchedule(offset=.49)
    for start in range(0, len(x), 1024):
        raw = x[start:start + 1024]
        front.feed(raw)
        retained.feed(front.last_low, start + len(raw), raw=raw)
        old.feed(front.last_low, raw)
        fresh.feed(front.last_low, raw)
    old.rebuild(.49, retained)
    assert old.rebuilds == 1
    assert old.next_row == fresh.next_row
    slots = old.interval(old.next_row - 8192, old.next_row)
    np.testing.assert_array_equal(old.bins[slots], fresh.bins[slots])
    np.testing.assert_allclose(old.power[slots], fresh.power[slots])
    np.testing.assert_array_equal(old.live[slots], fresh.live[slots])
    assert not old.hard and not old.soft and not old.pairs


@pytest.mark.parametrize('phase', range(32))
def test_request_waveform_all_raw_phases(phase):
    from hfmodem.kestrel.rx.bw500_features import FeatureTrack
    from hfmodem.kestrel.vara import vara_frames as f, vara_mfsk as m
    kind = f.for_bw(f.SESSION_TURN_REQUEST_RESPONDER, '500')
    tones = f.handshake_tones('W9SSJ', kind)
    x = np.r_[np.zeros(4096 + phase), m.synth_burst('W9SSJ', kind), np.zeros(4096)]
    front, schedule = FeatureTrack(stage='front'), ControlSchedule()
    for pos in range(0, len(x), 1024):
        raw = x[pos:pos + 1024]
        front.feed(raw)
        schedule.feed(front.last_low, raw)
    heard, at = schedule.view(0, len(x), 'request').payload_fit(tones, len(kind.preamble))
    assert f.recognize(heard, 'W9SSJ', kind)
    assert np.count_nonzero(heard >= 0) >= 24
    assert abs(at - (4096 + phase)) < 1024


def test_overflow_is_an_error_not_an_eviction():
    s = source(1000)
    view = s.view(0, 32000)
    for k in range(s.hard_limit):
        view.payload_fit([40, 41 + k, 70], 1)
    with pytest.raises(RuntimeError, match='registry overflow'):
        view.payload_fit([40, 90, 70], 1)
    assert len(s.hard) == s.hard_limit
    with pytest.raises(RuntimeError, match='history'):
        s.interval(-1, 2)


def test_batch_registry_keeps_twelve_templates_and_existing_contributions():
    s = source(1000)
    view = s.view(0, 32000)
    templates = [[40, 42 + k, 75, 55, 60] for k in range(12)]
    for tones in templates:
        view.payload_fit(tones, 1)
    batch = next(iter(s.hard.values())).batch
    assert len({id(score.batch) for score in s.hard.values()}) == 1
    before = batch.entries.copy()
    for tones in templates:
        heard, at = view.payload_fit(tones, 1)
        low, high = -256, (32000 - 2080) // 32
        expected_at, shift = scalar_fit(s, tones, 1, view.first, view.stop, low, high)
        assert at == expected_at * 32
        assert len(heard) == len(tones)
    np.testing.assert_array_equal(batch.entries, before)
    for score in s.hard.values():
        single = HardScores(score.tones, score.npre)
        single.advance(s, view.first, view.stop)
        assert score.entries == single.entries
        np.testing.assert_array_equal(score.counts, single.counts)
        np.testing.assert_array_equal(score.matches, single.matches)


def test_reader_context_reset_preserves_other_consumers():
    s = source(1000)
    s.set_context('answer', ('pending', 1))
    for reader in ('answer', 'held', 'request'):
        s.view(0, 32000, reader).payload_fit([40, 51, 70], 1)
    retained = dict(s.hard)
    s.set_context('answer', ('pending', 1))
    assert s.hard == retained
    s.set_context('answer', ('pending', 2))
    assert {key[0] for key in s.hard} == {'held', 'request'}
    assert all(s.hard[k] is retained[k] for k in s.hard)
    s.view(0, 32000).payload_fit([40, 51, 70], 1)
    assert all(s.hard[k] is not retained[k] for k in s.hard if k[0] == 'answer')


@pytest.mark.parametrize('pair', [False, True])
def test_exact_candidates_match_the_materialized_search_as_views_move(pair):
    from hfmodem.kestrel.rx.bw500_features import exact_control_candidates
    s = source(3000)
    r = np.random.default_rng(31)
    s.bins[:3000, 1] = r.integers(29, 99, 3000)
    s.clear[:3000, 1] = r.choice([1., 2., 4.], 3000)
    templates = np.array([[[50, 60], [52, 61], [47, 70], [55, 66]],
                          [[40, 44], [41, 45], [42, 46], [43, 47]]])[:, :, :2 if pair else 1]
    shifts = [-1, 0, 2]
    for at, (j, shift) in {700: (0, 0), 1500: (1, 2), 2400: (0, -1)}.items():
        rows = at + np.arange(4) * 64
        s.bins[rows, :templates.shape[2]] = templates[j][:, ::-1] + shift
        s.clear[rows] = 4.
    found = set()
    for start, end in [(0, 60000), (16000, 80000), (16016, 80016), (22000, 84000),
                       (40000, 90000), (40000, 96000), (30000, 96000), (60000, 96000)]:
        view = s.view(start, end)
        got = view.exact_candidates(templates if pair else templates[:, :, 0], shifts, pair)
        bins, clear, _ = view
        want = (exact_control_candidates(np.sort(bins[:, :2], axis=1), clear[:, 1], templates, shifts)
                if pair else exact_control_candidates(bins[:, 0], clear[:, 0], templates[:, :, 0], shifts))
        if (start - s.origin) % 32 == 16:
            assert got is None
            continue
        np.testing.assert_array_equal(got, want)
        found |= {j for j in range(len(templates)) if got[j]}
    assert found == {0, 1}
