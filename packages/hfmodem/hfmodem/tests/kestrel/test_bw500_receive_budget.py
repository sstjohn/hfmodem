# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import numpy as np
import pytest

from hfmodem.kestrel.coding import turbo
from hfmodem.kestrel.rx import bw500_locator as loc, varahf500 as rx
from hfmodem.kestrel.rx.bw500_features import FeatureTrack
from hfmodem.kestrel.rx.control_schedule import ControlSchedule
from hfmodem.kestrel.rx.control_track import ControlTrack


def drain(work):
    yields = 0
    while True:
        try:
            next(work)
            yields += 1
        except StopIteration as done:
            return done.value, yields


@pytest.mark.parametrize('seed', range(3))
def test_sliced_turbo_preserves_all_iterations_and_bits(monkeypatch, seed):
    llr = np.random.default_rng(seed).normal(0, 2, 748)
    expected = turbo.decode_from_coded_llr(llr, 368, check=rx._frame_crc_ok)
    original = turbo._bcjr
    calls = []
    def bcjr(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(turbo, '_bcjr', bcjr)
    work = turbo.decode_from_coded_llr_steps(llr, 368, check=rx._frame_crc_ok)
    for _ in range(2):
        before = len(calls)
        next(work)
        assert len(calls) - before == 8
    with pytest.raises(StopIteration) as done:
        next(work)
    np.testing.assert_array_equal(done.value.value, expected)
    assert len(calls) == 24


def test_clean_turbo_finishes_without_yielding():
    bits = np.random.default_rng(16).integers(0, 2, 368)
    llr = 8. * (1 - 2 * turbo.encode(bits))
    decoded, yields = drain(turbo.decode_from_coded_llr_steps(
        llr, 368, check=lambda got: np.array_equal(got, bits)))
    np.testing.assert_array_equal(decoded, bits)
    assert yields == 0


def test_full_alignment_slices_are_identical():
    samples = np.random.default_rng(91).normal(0, .01, 240000)
    expected, onsets, score = rx._demod_aligned(samples, 12000)
    (got, offsets, scored), yields = drain(rx._demod_aligned_steps(samples, 12000))
    assert offsets == onsets and scored == score
    assert yields > 2
    for band in expected:
        np.testing.assert_array_equal(got[band], expected[band])


def test_failed_candidate_yields_between_trials_and_drains_at_eof(monkeypatch):
    receiver = loc.Receiver()
    receiver.raw.feed(np.zeros(230000))
    epochs = (loc.Epoch(0, .2, 0), loc.Epoch(64, .19, 0))
    over = loc.Over(loc.Candidate(4, 0, .2, 0, 64, epochs))
    monkeypatch.setattr(loc, 'demod_local', lambda *args: (None, (0, 0), .2))
    monkeypatch.setattr(rx, '_frame_gh', lambda *args: object())
    trials, alignments = [], []
    def full(*args):
        alignments.append(1)
        yield
        return None, {0: 0, 1: 0}, .2
    def decode(gh, fk, c0):
        trials.append(c0)
        yield
        return .5, c0, False, b'', False
    monkeypatch.setattr(rx, '_demod_aligned_steps', full)
    monkeypatch.setattr(rx, '_decode_cells_steps', decode)
    for _ in range(8):
        before = len(trials)
        receiver._advance(over, False, False)
        assert len(trials) - before <= 1
        assert over.status == 'open'
    receiver._advance(over, False, True)
    assert over.status == 'closed'
    assert not over.result.complete
    assert trials == [9, 8, 10, 7, 11] * 3
    assert alignments == [1]
    assert over.work is None


@pytest.mark.parametrize('offset', [0., -.49, .23])
def test_lazy_spectra_match_eager_across_gaps_wrap_and_rebuild(offset):
    x = np.random.default_rng(94).normal(0, .01, 310272)
    front, retained = FeatureTrack(stage='front'), ControlTrack()
    eager = ControlSchedule(offset=offset)
    lazy = ControlSchedule(offset=offset, retained=retained)
    for start in range(0, len(x), 1024):
        raw = x[start:start + 1024]
        front.feed(raw)
        retained.feed(front.last_low, start + len(raw), raw=raw)
        eager.feed(front.last_low, raw)
        lazy.feed(front.last_low, raw)
        assert lazy.next_row == eager.next_row
        if start in (3072, 29696, 270336, 309248):
            begin = max(0, start - 12000)
            view = lazy.view(begin, start + len(raw))
            slots = lazy.interval(view.first, view.stop)
            for attr in ('bins', 'clear', 'residual', 'power', 'denominator', 'live'):
                np.testing.assert_array_equal(getattr(lazy, attr)[slots], getattr(eager, attr)[slots])
    assert lazy.transforms < eager.transforms // 3
    assert lazy.fft_rows < eager.fft_rows // 3
    lazy_fft_rows, eager_fft_rows = lazy.fft_rows, eager.fft_rows
    lazy.rebuild(.31, retained)
    assert lazy.fft_rows == lazy_fft_rows
    eager.rebuild(.31, retained)
    assert eager.fft_rows > eager_fft_rows
    view = lazy.view(len(x) - 12000, len(x))
    slots = lazy.interval(view.first, view.stop)
    for attr in ('bins', 'clear', 'residual', 'power', 'denominator', 'live'):
        np.testing.assert_array_equal(getattr(lazy, attr)[slots], getattr(eager, attr)[slots])
    assert lazy.fft_rows < eager.fft_rows // 3


def test_hard_rank_aliases_keep_each_template_lower_bound():
    from hfmodem.tests.kestrel.test_control_schedule import source, scalar_fit
    templates = [([40, 51, 70], 1), ([40, 51, 60, 70, 80], 0)]
    for first in (0, 1, 62, 63, 100):
        for high in (499, 500):
            s = source(1000)
            view = s.view(first * 32, (high + 65) * 32)
            for tones, pre in templates:
                heard, at = view.payload_fit(tones, pre)
                low = first - (len(tones) - 1) * 64
                expected, shift = scalar_fit(s, tones, pre, view.first, view.stop, low, high)
                assert at == expected * 32 - view.start


@pytest.mark.parametrize('shift', [-2, 0, 2])
@pytest.mark.parametrize('heard_count,errors', [(8, 1), (8, 0), (15, 2), (15, 4)])
def test_match_filter_preserves_rank_and_seven_of_eight(shift, heard_count, errors):
    from hfmodem.tests.kestrel.test_control_schedule import source
    s = source(3000)
    tones = np.random.default_rng(73).integers(35, 90, 16)
    s.clear[:, 0] = 0
    rows = 100 + np.arange(1, heard_count + 1) * 64
    s.bins[rows, 0] = tones[1:heard_count + 1] + shift
    s.clear[rows, 0] = 4
    s.bins[rows[:errors], 0] = 29
    for end in (40000, 65000, 96000):
        view = s.view(0, end)
        expected, at = view.payload_fit(tones, 1)
        heard, found = view.payload_fit(tones, 1, require_match=True)
        if np.count_nonzero(expected[1:] == tones[1:]) >= 7:
            np.testing.assert_array_equal(heard, expected)
            assert found == at
        else:
            assert np.all(heard == -1)
    for first in (16000, 32000, 64000):
        view = s.view(first, 96000)
        expected, at = view.payload_fit(tones, 1)
        heard, found = view.payload_fit(tones, 1, require_match=True)
        if np.count_nonzero(expected[1:] == tones[1:]) >= 7:
            np.testing.assert_array_equal(heard, expected)
            assert found == at
        else:
            assert np.all(heard == -1)


@pytest.mark.parametrize("size,steps", [(1024, 1), (3000, 3), (5000, 5)])
def test_callback_budget_is_shared_by_pending_candidates(monkeypatch, size, steps):
    receiver = loc.Receiver()
    receiver.pending = [loc.Over(loc.Candidate(4, start, .2, 0, start)) for start in (0, 2048)]
    calls = []
    def advance(over, *args):
        calls.append(over.candidate.start)
    monkeypatch.setattr(receiver, '_advance', advance)
    turns = [2048, 0] * 8
    receiver.feed(np.zeros(size))
    assert calls == turns[:steps]
    receiver.feed(np.zeros(size))
    assert calls == turns[:2 * steps]
    receiver.feed(np.empty(0), batches=[], eof=True)
    assert calls == turns[:2 * steps + 2]


def test_a_lock_inside_a_running_over_does_not_starve_it_for_its_partner(monkeypatch):
    """L4 −4 dB seed 26 pair, random callbacks: a junk lock inside the first
    frame took every step from that frame's nearly finished decode, until the
    second frame located on its own, decoded, and retired the over that would
    have read both. The running over keeps its share beside the lock."""
    slot = rx.NSYM * rx.H
    first, lock, partner = (loc.Over(loc.Candidate(4, at, .2, 0, at))
                            for at in (0, slot // 2, slot))
    def advance(over, *args):
        if over is lock or over is first and over.steps < 6:
            return
        frame = rx.FrameResult(over.candidate.start, (0,), 9, 1., True, b'')
        over.frames = [frame] * (2 if over is first else 1)
        over.end, over.status = over.candidate.start + 2 * slot, 'closed'
    receiver = loc.Receiver()
    monkeypatch.setattr(receiver, '_advance', advance)
    out = []
    for arrival in (first, lock, partner):
        receiver.pending.append(arrival)
        for _ in range(4 if arrival is first else 8):
            out += receiver.feed(np.zeros(1024), batches=[])
    assert out == [first] and len(first.frames) == 2
    assert not receiver.pending


def test_match_registry_extension_preserves_live_candidates():
    from hfmodem.tests.kestrel.test_control_schedule import source
    s = source(3000)
    s.clear[:, 0] = 0
    templates = [np.arange(40, 56), np.arange(55, 71)]
    for at, tones in zip((100, 1400), templates):
        rows = at + np.arange(1, 16) * 64
        s.bins[rows, 0], s.clear[rows, 0] = tones[1:], 4
    view = s.view(0, 96000)
    for tones in templates * 2:
        expected, at = view.payload_fit(tones, 1)
        heard, found = view.payload_fit(tones, 1, require_match=True)
        np.testing.assert_array_equal(heard, expected)
        assert found == at


def test_sparse_match_candidates_expire_at_ring_wrap():
    from hfmodem.tests.kestrel.test_control_schedule import source
    s = source(8192)
    s.clear[:, 0] = 0
    tones = np.arange(40, 56)
    rows = 100 + np.arange(1, 16) * 64
    s.bins[rows, 0], s.clear[rows, 0] = tones[1:], 4
    view = s.view(0, 40000)
    assert np.count_nonzero(view.payload_fit(tones, 1, require_match=True)[0] == tones) == 15
    view = s.view(6000 * 32, 8192 * 32)
    assert np.all(view.payload_fit(tones, 1, require_match=True)[0] == -1)
    for first in range(8192, 16000, 128):
        rows = np.arange(first, first + 128)
        s.tags[rows % s.capacity] = rows
        s.bins[rows % s.capacity, 0] = 29
        s.clear[rows % s.capacity, 0] = 4
        s.next_row = first + 128
        heard, _ = s.view((first - 2000) * 32, (first + 128) * 32).payload_fit(
            tones, 1, require_match=True)
        assert np.all(heard == -1)
    assert not next(iter(s.hard.values())).batch.candidates[0]
