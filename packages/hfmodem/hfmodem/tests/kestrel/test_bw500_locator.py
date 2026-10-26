# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Independent multirate arithmetic and causal stream accounting."""
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import firwin, upfirdn

from hfmodem.kestrel.rx import bw500_locator as locator, varahf500 as rx


def _collect(x, chunk, stage='all'):
    track = locator.FeatureTrack(stage=stage)
    rows = {lv: [] for lv in range(1, 5)}
    low = []
    for j in range(0, len(x), chunk):
        before = track.samples // 8
        for lv, starts, scores, available in track.feed(x[j:j + chunk]):
            rows[lv].append((starts, scores, np.full(len(starts), available)))
        assert track.low_start == before
        assert len(track.pending) < 1024
        low.append(track.last_low.copy())
    result = {lv: (np.concatenate([r[0] for r in values]),
                   np.concatenate([r[1] for r in values], axis=-1),
                   np.concatenate([r[2] for r in values]))
              for lv, values in rows.items() if values}
    return track, result, np.concatenate(low)


def _reference(x):
    n = len(x) // 1024 * 1024
    low = upfirdn(firwin(129, 2800, fs=48000), x[:n], down=8)[:n // 8]
    phase = np.arange(len(low)) % 640
    mix = np.exp(-2j * np.pi * np.array([1350, 1650])[:, None] * phase / 6000)
    z = upfirdn(firwin(101, 120, fs=6000), low * mix, down=8, axis=-1)[:, :n // 64]
    d = z * np.concatenate((np.zeros((2, 8)), z[:, :-8]), axis=1).conj()
    u = np.zeros_like(d)
    np.divide(d * d, np.abs(d) ** 2, out=u, where=np.abs(d) ** 2 > 1e-100)
    refs = np.array([rx._GRID480[rx.grid480_index(s, rx.NSYM)] for s, _ in rx.SUBBANDS])
    dref = refs[:, 1:] * refs[:, :-1].conj()
    q = np.zeros_like(dref)
    np.divide(dref * dref, np.abs(dref) ** 2, out=q, where=np.abs(dref) > 0)
    assert np.all(np.count_nonzero(q[:, :393], axis=1) == 384)
    span = 393 * 8
    epochs = np.arange(max(0, u.shape[1] - span))
    windows = sliding_window_view(u[:, 8:], span - 8 + 1, axis=-1)[:, :, ::8]
    scores = abs(np.einsum('sjn,sn->sj', windows, q[:, :393], optimize=False)) / 384
    available = ((epochs + span) // 16 + 1) * 1024
    rows = {4: (epochs * 64 - 464, scores, available)}
    for lv, rec in rx.INDEX_RECORDS.items():
        dw, hop = rec.dw // 8, rec.dw // 64
        transforms = sliding_window_view(low, dw)[::hop]
        power = abs(np.fft.rfft(transforms, axis=1)[:, rec.first_bin:rec.first_bin + rec.span]) ** 2
        lit = power.argmax(axis=1)
        lit[power.max(axis=1) < 1e-100] = -1
        ref, bins = rx._l3_tables(lv)[:2]
        epochs = np.arange(max(0, len(lit) - int(ref.max()) * 8))
        hits = (lit[epochs[:, None] + ref * 8] == bins - rec.first_bin).sum(axis=1)
        ends = (epochs + int(ref.max()) * 8) * hop + dw
        available = ((ends + 127) // 128) * 1024
        rows[lv] = (epochs * hop * 8 - 64 - rec.lead * rec.dw, hits, available)
    return low, rows


@pytest.fixture(scope='module')
def arithmetic_case():
    x = np.random.default_rng(19).normal(size=300001)
    low, rows = _reference(x)
    return x, low, rows


@pytest.mark.parametrize('chunk', [1024, 4096, 1003, 5003, 300001])
def test_feature_matches_independent_polyphase_and_rectangular_fft(arithmetic_case, chunk):
    x, low, expected = arithmetic_case
    track, actual, decimated = _collect(x, chunk)
    np.testing.assert_allclose(decimated, low, atol=2e-14, rtol=2e-14)
    assert actual.keys() == expected.keys()
    for lv, (starts, scores, available) in actual.items():
        want_starts, want_scores, want_available = expected[lv]
        np.testing.assert_array_equal(starts, want_starts)
        np.testing.assert_allclose(scores, want_scores, atol=2e-12, rtol=2e-12)
        np.testing.assert_array_equal(available, want_available)
        assert len(np.unique(starts)) == len(starts)
        np.testing.assert_array_equal(np.diff(starts), 64 if lv == 4 else rx.INDEX_RECORDS[lv].dw // 8)
    assert track.samples == len(x) // 1024 * 1024
    assert len(track.pending) == len(x) % 1024


def test_front_only_retains_partial_callbacks_and_shares_the_same_low_stream(arithmetic_case):
    x, low, _ = arithmetic_case
    track, rows, actual = _collect(x, 1003, stage='front')
    assert not rows
    np.testing.assert_allclose(actual, low, atol=2e-14, rtol=2e-14)


def test_dead_input_has_no_reference_support_or_candidate():
    track, rows, low = _collect(np.zeros(300001), 5003)
    assert not low.any()
    grouper = locator.Grouper()
    for lv, (starts, values, available) in rows.items():
        assert len(starts)
        assert not values.any()
        assert not grouper.feed([(lv, starts, values, int(available[-1]))])
    assert not grouper.feed([], eof=True)
    assert not grouper.groups


def test_group_closes_only_after_full_gap_and_records_actual_decision():
    grouper = locator.Grouper()
    assert not grouper.feed([(3, np.array([0, 128, 1024]), np.array([12, 20, 0]), 4096)])
    ready = grouper.feed([(3, np.array([1152, 1280]), np.array([0, 0]), 5120)])
    assert len(ready) == 1
    candidate = ready[0]
    assert (candidate.level, candidate.start, candidate.score, candidate.last) == (3, 128, 20, 128)
    assert candidate.decision == 5120
    assert not grouper.feed([], eof=True)
    assert grouper.count == 1


def test_callback_arrival_is_later_than_internal_feature_availability():
    grouper = locator.Grouper()
    ready = grouper.feed([(3, np.array([0, 128, 1280]), np.array([12, 20, 0]), 2048)],
                         available=5003)
    assert ready[0].decision == 5003
    assert ready[0].epochs[0].available == 2048


def test_large_receiver_callback_keeps_its_actual_arrival(monkeypatch):
    receiver = locator.Receiver()
    def group(*args, **kwargs):
        if receiver.raw.samples == 1024:
            return [locator.Candidate(4, 0, .5, kwargs['available'], 0)]
        return []
    monkeypatch.setattr(receiver.grouper, 'feed', group)
    def finish(over, may_pair, eof):
        over.status = 'closed'
    monkeypatch.setattr(receiver, '_advance', finish)
    found = receiver.feed(np.zeros(5003), eof=True)
    assert len(found) == 1
    assert found[0].candidate.decision == found[0].decision == 5003


def test_level4_stagger_uses_separate_band_maxima_and_right_context():
    grouper = locator.Grouper()
    starts = np.arange(64) * 64
    values = np.zeros((2, len(starts)))
    values[0, 10], values[1, 18] = .2, .3
    ready = []
    for j in range(0, len(starts), 8):
        found = grouper.feed([(4, starts[j:j + 8], values[:, j:j + 8], 10000 + j * 64)])
        if j < 32:
            assert not found
        ready.extend(found)
        assert len(grouper.tail[0]) <= 8
        assert len(grouper.groups) <= 1
    assert len(ready) == 1
    assert ready[0].start == 14 * 64
    assert ready[0].score == pytest.approx(.2)
    assert ready[0].decision >= 10000 + 32 * 64


def test_raw_carry_preserves_absolute_positions_and_bounded_storage():
    carry = locator.RawCarry(capacity=8192)
    for j in range(50):
        carry.feed(np.arange(j * 1024, (j + 1) * 1024, dtype=float))
        assert carry.samples == (j + 1) * 1024
        assert len(carry.raw) <= carry.capacity
        np.testing.assert_array_equal(carry.raw, np.arange(carry.origin, carry.samples))
    with pytest.raises(ValueError, match='bounded'):
        carry.feed(np.zeros(carry.capacity + 1))


def test_pending_queue_overflow_is_explicit_and_never_suppresses(monkeypatch):
    receiver = locator.Receiver()
    candidates = [locator.Candidate(3, 100000 + j * 20000, 24, 1024, 100000 + j * 20000)
                  for j in range(5)]
    monkeypatch.setattr(receiver.grouper, 'feed', lambda *args, **kwargs: candidates)
    with pytest.raises(RuntimeError, match='overflow'):
        receiver.feed(np.zeros(1024), batches=[])
    assert receiver.overflows == 1
    assert len(receiver.pending) == 4


def test_first_short_level4_batch_does_not_drop_its_first_four_epochs():
    grouper = locator.Grouper()
    starts = np.arange(32) * 64
    scores = np.zeros((2, 32))
    scores[:, 0] = .2
    ready = grouper.feed([(4, starts[:8], scores[:, :8], 1024)])
    assert not ready
    assert grouper.groups[4].start == 0
    ready = grouper.feed([(4, starts[8:], scores[:, 8:], 2048)])
    assert len(ready) == 1
    assert ready[0].start == 0
    assert [epoch.start for epoch in ready[0].epochs] == [0, 64]
    assert len(grouper.tail[0]) <= 8


def test_eof_scores_unfinished_right_edge_once_and_uses_eof_decision():
    grouper = locator.Grouper()
    starts = np.arange(4) * 64
    scores = np.zeros((2, 4))
    scores[0, 0], scores[1, 3] = .2, .3
    assert not grouper.feed([(4, starts, scores, 1024)])
    assert not grouper.groups
    ready = grouper.feed([], eof=True, available=1731)
    assert len(ready) == 1
    assert ready[0].start == 0
    assert ready[0].decision == 1731
    assert [epoch.start for epoch in ready[0].epochs] == [0, 64]
    assert not grouper.tail[0].size
    assert not grouper.feed([], eof=True, available=1731)
    assert grouper.count == 1


def test_group_retains_only_two_distinct_epochs_and_separate_band_scores():
    grouper = locator.Grouper()
    starts = np.arange(24) * 64
    scores = np.zeros((2, 24))
    scores[:, 8] = [.31, .4]
    assert not grouper.feed([(4, starts[:8], scores[:, :8], 1024)])
    assert not grouper.feed([(4, starts[8:], scores[:, 8:], 2048)])
    ready = grouper.feed([], eof=True, available=3000)
    assert len(ready) == 1
    epochs = ready[0].epochs
    assert len(epochs) == 2
    assert [e.start for e in epochs] == [4 * 64, 5 * 64]
    assert epochs[0].bands == pytest.approx((.31, .4))
    assert epochs[0].available == 2048


def test_alternate_group_epoch_realigns_only_after_primary_crc_failure(monkeypatch):
    class Front:
        def feed(self, samples): return []

    receiver = locator.Receiver(front=Front())
    rec = rx.INDEX_RECORDS[3]
    primary, alternate = 2000, 3000
    epochs = (locator.Epoch(primary, 24., 10000), locator.Epoch(alternate, 23., 10000))
    candidate = locator.Candidate(3, primary, 24., 10000, alternate, epochs)
    over = locator.Over(candidate)
    aligns = []

    def align(samples, start, stop, level):
        aligns.append(start)
        return start, rec.lead, 24, None

    def frames(start, stop, offset, lead, hits, band, level, **kwargs):
        return [rx.FrameResult(start=offset, onset=(0,), c0=lead, self_consistency=.9,
                               crc_ok=offset == alternate, frame_bytes=b'x' * 48, level=level)]

    monkeypatch.setattr(rx, '_l3_align', align)
    monkeypatch.setattr(rx, '_l3_frames', frames)
    receiver.raw.feed(np.zeros(alternate + (rec.lead + rec.ncols + 10) * rec.dw))
    receiver._advance(over, may_pair=False, eof=False)
    assert aligns == [primary, primary - rec.dw // 2, alternate]
    assert over.epoch_try == 1
    assert over.frames[0].crc_ok
    assert over.candidate.start == alternate
    assert over.alignment[0] == alternate
    assert over.status == 'closed'
    assert receiver.refinements == 2
    assert receiver.turbo_attempts == 3
