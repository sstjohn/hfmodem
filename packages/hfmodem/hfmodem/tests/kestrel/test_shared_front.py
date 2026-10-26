# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Shared control arithmetic, feature fit ranking, and stream lifecycle."""
import numpy as np
import pytest

from hfmodem.kestrel.rx.bw500_features import FeatureTrack, payload_fit
from hfmodem.kestrel.rx.control_track import ControlTrack
from hfmodem.kestrel.vara import vara_arq as arq, vara_frames as VF


@pytest.fixture
def front_case():
    rng = np.random.default_rng(67)
    raw = rng.normal(0, .003, 16384)
    raw += .02 * np.sin(2 * np.pi * 65.23 * np.arange(len(raw)) / 2048)
    raw[:3072] = 0
    feature = FeatureTrack(stage='front')
    low = []
    control = ControlTrack()
    for start in range(0, len(raw), 1024):
        block = raw[start:start + 1024]
        feature.feed(block)
        low.append(feature.last_low.copy())
        control.feed(feature.last_low, start + len(block), raw=block)
    return raw, np.concatenate(low), control


def _sampled_reference(low, starts, band, offset):
    win = np.hanning(2048)[::8]
    win = win * np.exp(-2j * np.pi * offset * np.arange(256) / 256)
    full = np.array([abs(np.fft.fft(low[start // 8 + 8:start // 8 + 264] * win)) * 8
                     for start in starts])
    lo, hi = band
    power = full[:, lo:hi + 1] ** 2
    bins = np.empty((len(starts), 3), dtype=np.int32)
    clear = np.empty((len(starts), 2))
    resid = np.empty(len(starts))
    for row, spectrum in enumerate(full):
        work = spectrum[lo:hi + 1].copy()
        peaks = []
        for k in range(3):
            j = int(np.argmax(work))
            bins[row, k] = lo + j
            peaks.append(work[j])
            if not k:
                left, mid, right = spectrum[lo + j - 1:lo + j + 2]
                resid[row] = 2 * (right - left) / (left + 2 * mid + right + 1e-20)
            work[max(0, j - 1):j + 2] = 0
        clear[row] = 20 * np.log10((np.array(peaks[:2]) + 1e-20)
                                   / (np.array(peaks[1:]) + 1e-20))
    return bins, clear, resid, power


@pytest.mark.parametrize('offset', [0., -.37, .23])
@pytest.mark.parametrize('band', [(54, 74), (29, 98), (22, 105)])
def test_control_matches_independent_symmetric_sampled_hann(front_case, offset, band):
    _, low, control = front_case
    start, stop = 32, 8192
    starts = np.arange(start, stop - 2048 + 1, 32)
    expected = _sampled_reference(low, starts, band, offset)
    actual = (*control.track(start, stop, band, bin_offset=offset),
              control.power(start, stop, band, offset))
    np.testing.assert_array_equal(actual[0], expected[0])
    for got, want in zip(actual[1:], expected[1:]):
        np.testing.assert_allclose(got, want, atol=1e-11, rtol=2e-12)


def test_live_mask_uses_original_raw_rms_and_causal_completion(front_case):
    raw, _, control = front_case
    starts = np.arange(0, control.effective_end - 2048 + 1, 32)
    expected = np.array([20 * np.log10(np.sqrt(np.mean(raw[s:s + 2048] ** 2)) + 1e-15)
                         > -50 for s in starts])
    np.testing.assert_array_equal(control.live(0, len(raw)), expected)
    assert control.effective_end == len(raw) - 64
    quiet = ControlTrack()
    quiet.feed(np.zeros(264), 2112, raw=np.zeros(2112))
    np.testing.assert_array_equal(quiet.live(0, 2112), [False])
    assert len(quiet.track(0, 2112)[0]) == 1


def test_track_power_and_fractional_overlap_transform_each_window_once(front_case):
    _, _, control = front_case
    band, offset = (54, 74), -.23
    a = control.track(0, 8192, band, bin_offset=offset)
    first = control.transforms
    assert first == len(a[0])
    p = control.power(0, 8192, band, offset)
    assert control.transforms == first
    control.power(32, 9216, band, offset)
    assert control.transforms == first + 1024 // 32
    count = control.transforms
    b = control.track(32, 9216, band, bin_offset=offset)
    assert control.transforms == count
    np.testing.assert_array_equal(a[0][1:], b[0][:-32])
    np.testing.assert_array_equal(p[1:], control.power(32, 8192, band, offset))
    assert control.transforms == count
    assert control.cache_hits >= len(a[0])


def test_numeric_cache_allocations_are_bounded_and_reset_clears_history(front_case):
    raw, low, _ = front_case
    control = ControlTrack(max_cache=256)
    control.feed(low, len(raw), raw=raw)
    for offset in np.arange(12) / 20:
        control.power(0, 8192, (54, 74), offset)
        control.live(0, 8192)
        assert control.cache_size <= 256
        assert sum(len(c.starts) for c in control._cache.values()) <= 256
        assert sum(len(c.starts) for c in control._live_cache.values()) <= 256
    arrays = [control._low, control._power, control._window]
    for cache in (control._cache, control._live_cache):
        for rows in cache.values():
            arrays.extend(getattr(rows, name) for name in rows.__slots__
                          if getattr(rows, name) is not None)
    assert all(a.dtype != object for a in arrays)
    assert sum(a.nbytes for a in arrays) < 3 * 1024 * 1024
    control.reset(32000)
    assert control.effective_end == control.oldest_start == 32000
    assert control.transforms == control.cache_hits == control.cache_size == 0
    assert not control._live_cache
    assert control.power(32000, 32000, (54, 74)).shape == (0, 21)
    with pytest.raises(ValueError, match='history'):
        control.track(0, 32000)


def _python_payload_fit(xlen, kind, tones, track):
    bins, clear, _ = track
    n_sym, npre = len(tones), len(kind.preamble)
    lo, hi = -(n_sym - 1) * arq.MK.HOP, xlen - arq.MK.STRIDE
    if not len(bins) or hi < lo:
        return np.full(n_sym, -1, dtype=np.int32), 0
    starts = np.arange(lo, hi + 1, arq._ACK_GRID)
    gi = np.rint((starts[:, None] + np.arange(n_sym) * arq.MK.HOP + arq.MK._WOFF)
                 / arq._ACK_GRID).astype(np.int64)
    j = np.clip(gi, 0, len(bins) - 1)
    ok = (gi >= 0) & (gi < len(bins)) & (clear[j, 0] >= arq._CLEAR_DB)
    heard = np.where(ok, bins[j, 0], -1).astype(np.int32)
    hd, valid, want = heard[:, npre:], ok[:, npre:], tones[npre:]
    comparable = valid.sum(1)
    best = None
    for shift in arq._RESP_SHIFTS:
        matches = ((hd == want + shift) & valid).sum(1)
        answers = ((matches >= arq._RESP_MIN_TONES)
                   | ((comparable >= arq._RESP_MIN_HEARD) & (matches == comparable)))
        row = int(np.lexsort((comparable - matches, -matches, ~answers))[0])
        rank = bool(answers[row]), int(matches[row]), int(matches[row] - comparable[row])
        if best is None or rank > best[0]:
            best = rank, row, shift
    _, row, shift = best
    return np.where(heard[row] >= 0, heard[row] - shift, -1).astype(np.int32), int(starts[row])


@pytest.mark.parametrize('case', ['random', 'shifted', 'partial', 'deaf', 'empty'])
def test_feature_payload_fit_preserves_independent_python_ranking(case):
    rng = np.random.default_rng(102)
    kind = VF.CONNECT_RESPONSE
    tones = rng.integers(40, 90, len(kind.preamble) + kind.n_payload, dtype=np.int32)
    xlen = 48000
    n = max(0, (xlen - arq.MK.NFFT) // arq._ACK_GRID + 1)
    bins = rng.integers(29, 99, (n, 3), dtype=np.int32)
    clear = rng.uniform(0, 12, (n, 2))
    if case in ('shifted', 'partial'):
        origin = -6 * arq.MK.HOP if case == 'partial' else 512
        for symbol, tone in enumerate(tones):
            j = int(round((origin + symbol * arq.MK.HOP + arq.MK._WOFF)
                          / arq._ACK_GRID))
            if 0 <= j < n:
                bins[j, 0], clear[j, 0] = tone + 1, 20
    if case == 'deaf':
        clear[:] = 0
    if case == 'empty':
        bins, clear = bins[:0], clear[:0]
    track = bins, clear, np.zeros(len(bins))
    expected = _python_payload_fit(xlen, kind, tones, track)
    actual = arq._payload_fit(np.zeros(xlen), kind, tones, track)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def _independent_grid_fit(bins, clear, tones, shifts, npre, xlen, hop, woff,
                          stride, grid, clearance, min_tones=13, min_heard=8):
    lo, hi = -(len(tones) - 1) * hop, xlen - stride
    if not len(bins) or hi < lo:
        return np.full(len(tones), -1, np.int32), 0
    starts = np.arange(lo, hi + 1, grid)
    indices = np.rint((starts[:, None] + np.arange(len(tones)) * hop + woff) / grid).astype(int)
    clipped = np.clip(indices, 0, len(bins) - 1)
    valid = (indices >= 0) & (indices < len(bins)) & (clear[clipped] >= clearance)
    heard = np.where(valid, bins[clipped], -1).astype(np.int32)
    counts = valid[:, npre:].sum(axis=1)
    best = None
    for shift in shifts:
        matches = ((heard[:, npre:] == tones[npre:] + shift) & valid[:, npre:]).sum(axis=1)
        accepted = (matches >= min_tones) | ((counts >= min_heard) & (matches == counts))
        row = int(np.lexsort((counts - matches, -matches, ~accepted))[0])
        rank = bool(accepted[row]), int(matches[row]), int(matches[row] - counts[row])
        if best is None or rank > best[0]:
            best = rank, row, int(shift)
    _, row, shift = best
    return np.where(heard[row] >= 0, heard[row] - shift, -1).astype(np.int32), int(starts[row])


@pytest.mark.parametrize('seed', range(12))
def test_feature_fit_randomized_aligned_and_generalized_grids(seed):
    rng = np.random.default_rng(8670 + seed)
    for case in range(12):
        grid = int(rng.integers(1, 9))
        aligned = case % 2 == 0
        hop = grid * int(rng.integers(1, 9)) if aligned else int(rng.integers(1, 25))
        woff = grid * int(rng.integers(-4, 5)) if aligned else int(rng.integers(-13, 14))
        nsym = int(rng.integers(1, 25))
        npre = int(rng.integers(0, nsym + 1))
        xlen, stride = int(rng.integers(0, 501)), int(rng.integers(1, 31))
        bins = rng.integers(-1, 80, int(rng.integers(0, 150)), dtype=np.int32)
        clear = rng.choice(np.array([-np.inf, 0., 4., np.nan, np.inf]), len(bins))
        tones = rng.integers(20, 70, nsym, dtype=np.int32)
        shifts = rng.permutation(np.array([0, -2, 1, -1, 2], np.int32))
        if case % 3 == 0 and len(bins):
            origin = -(nsym // 2) * hop
            for k, tone in enumerate(tones):
                j = int(np.rint((origin + k * hop + woff) / grid))
                if 0 <= j < len(bins):
                    bins[j], clear[j] = tone + shifts[0], 4.
        expected = _independent_grid_fit(bins, clear, tones, shifts, npre, xlen,
                                         hop, woff, stride, grid, 4.)
        actual = payload_fit(bins, clear, tones, shifts, npre, xlen, nsym,
                             hop, woff, stride, grid, 4.)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1], (seed, case, aligned)


def test_feature_fit_preserves_first_shift_before_earliest_global_origin():
    bins = np.array([48, 52, 48, 52], np.int32)
    clear = np.full(4, 4.)
    tones = np.array([50, 50], np.int32)
    shifts = np.array([2, -2, 0], np.int32)
    actual = payload_fit(bins, clear, tones, shifts, 1, 4, 2, 1, 0, 1, 1, 4.)
    np.testing.assert_array_equal(actual[0], [46, 50])
    assert actual[1] == 0


def test_station_deduplicates_overlaps_and_resets_on_sample_gap():
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    station.on_rx_stream(np.zeros(1024), sample_start=1000)
    generation = station._front_generation
    station.on_rx_stream(np.zeros(1024), sample_start=1000)
    assert station._front_heard == 1024
    assert station._front_generation == generation
    station.on_rx_stream(np.zeros(1024), sample_start=1512)
    assert station._front_heard == 1536
    station.on_rx_stream(np.zeros(1024), sample_start=4096)
    assert station._front_generation > generation
    assert station._front_heard == 1024
    assert station._control.transforms == 0
    station.rx_discontinuity()
    station.on_rx_stream(np.zeros(1024), sample_start=0)
    assert station._front_heard == 1024
    assert station._input_end == 1024


def test_station_turn_mode_change_resets_shared_front_without_decoding():
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='500')
    station.state = arq.VaraState.CONNECTED
    station.role = 'initiator'
    station.caller, station.called = 'W9SSJ', 'W1AW'
    station.turn = arq._TURN_PEER
    station._feed_shared_front(np.zeros(1024))
    generation = station._front_generation
    control = station._control
    station.turn = arq._TURN_OURS
    station._feed_shared_front(np.zeros(1024))
    assert station._front_generation > generation
    assert station._control is not control
    assert station._front_heard == 1024
    assert station._rx_front.samples == 8192 + station._front_heard
    assert station._control.transforms == 0


@pytest.mark.parametrize('reason', ['off-grid', 'expired'])
def test_frontend_wrapper_falls_back_for_unavailable_coordinates(front_case, reason):
    raw, low, control = front_case
    if reason == 'expired':
        control = ControlTrack(retain_seconds=.05)
        control.feed(low, len(raw), raw=raw)
        origin = 0
        assert origin < control.oldest_start
    else:
        origin = 3
    x = raw[origin:origin + 4096]
    expected = arq._top3_track(x, band=(54, 74), bin_offset=.17)
    actual = arq._top3_track(x, band=(54, 74), bin_offset=.17,
                             frontend=control, origin=origin)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    assert control.transforms == 0


def test_frontend_wrapper_fills_unavailable_tail_without_losing_rows(front_case):
    raw, _, control = front_case
    band, offset = (54, 74), -.19
    cached = control.track(0, len(raw), band, bin_offset=offset)
    boundary = len(cached[0]) * arq._ACK_GRID
    tail = arq._top3_track(raw[boundary:], band=band, bin_offset=offset)
    actual = arq._top3_track(raw, band=band, bin_offset=offset,
                             frontend=control, origin=0)
    assert len(actual[0]) == (len(raw) - arq.MK.NFFT) // arq._ACK_GRID + 1
    assert len(tail[0]) > 0
    for got, before, after in zip(actual, cached, tail):
        np.testing.assert_array_equal(got[:len(before)], before)
        np.testing.assert_array_equal(got[len(before):], after)
    assert control.transforms == len(cached[0])


def _awaiting_station():
    station = arq.VaraStationHandshake(['W9SSJ'], arq.VaraIO(), bw='2300')
    station.state = arq.VaraState.CONNECTING
    station.role = 'initiator'
    station.caller, station.called = 'W9SSJ', 'W1AW'
    station.step = arq._I_CR_SENT
    return station


@pytest.mark.parametrize('reason', ['off-grid', 'expired'])
def test_response_stream_falls_back_without_changing_coordinates(monkeypatch, reason):
    station = _awaiting_station()
    if reason == 'expired':
        monkeypatch.setattr(arq, 'ControlTrack',
                            lambda **kw: ControlTrack(retain_seconds=.05, **kw))
        origin = 0
    else:
        station._feed_shared_front(np.zeros(1027))
        station._reset_stream()
        origin = 1027
    x = np.zeros(32768)
    station.on_rx_stream(x)
    nwin = (len(x) - arq.MK.NFFT) // arq._ACK_GRID + 1
    span = arq.MK.NFFT + (nwin - 1) * arq._ACK_GRID
    expected = arq._top3_track(x[:span], band=station._band)
    for actual, want in zip((station._st_track, station._st_clear, station._st_resid), expected):
        np.testing.assert_array_equal(actual, want)
    np.testing.assert_array_equal(station._st_live, arq._live_track(x[:span]))
    assert station._response_origin == origin + nwin * arq._ACK_GRID
    assert len(station._st_buf) == len(x) - nwin * arq._ACK_GRID
    assert station._control.transforms == 0


def test_response_scan_keeps_original_raw_batching_boundary():
    station = _awaiting_station()
    threshold = arq.MK.NFFT + arq._STREAM_BLOCK
    station.on_rx_stream(np.zeros(threshold - 1))
    assert len(station._st_track) == 0
    station.on_rx_stream(np.zeros(1))
    nwin = (threshold - arq.MK.NFFT) // arq._ACK_GRID + 1
    assert len(station._st_track) == len(station._st_clear) == len(station._st_live) == nwin
    assert station._response_origin == nwin * arq._ACK_GRID
    assert station._control.effective_end < threshold
