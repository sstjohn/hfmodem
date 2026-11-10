# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import numpy as np
import pytest

from hfmodem.kestrel.rx.bw500_features import exact_control_candidates


@pytest.mark.parametrize('carriers', [1, 2])
@pytest.mark.parametrize('shift', [-2, -1, 0, 1, 2])
@pytest.mark.parametrize('start', [0, 1, 63, 127, 319])
def test_exact_control_gate_preserves_shifted_sequences(carriers, shift, start):
    rng = np.random.default_rng(723)
    templates = rng.integers(35, 90, (19, 4, carriers))
    bins = np.zeros((512, carriers), int)
    clear = np.zeros(512)
    rows = start + np.arange(4) * 64
    bins[rows] = templates[7] + shift
    clear[rows] = 2
    found = exact_control_candidates(bins, clear, templates, range(-2, 3))
    assert np.flatnonzero(found).tolist() == [7]
    clear[rows[2]] = 1.999
    assert not exact_control_candidates(bins, clear, templates, range(-2, 3)).any()


def test_control_gate_does_not_accept_disjoint_symbol_hits():
    templates = np.array([[40, 50, 60, 70], [40, 50, 60, 71]])
    bins = np.zeros(512, int)
    bins[[0, 64, 128, 193]] = templates[0]
    assert not exact_control_candidates(bins, np.full(512, 3.), templates, [0]).any()
    bins[192] = 70
    np.testing.assert_array_equal(
        exact_control_candidates(bins, np.full(512, 3.), templates, [0]), [True, False])


def test_control_gate_empty_and_short_tracks():
    for size in (0, 50, 192):
        assert not exact_control_candidates(np.zeros(size, int), np.zeros(size),
                                            np.array([[40, 50, 60, 70]]), [0]).any()


@pytest.mark.parametrize('layout', [(1024,), (1, 511, 1903, 73, 5000)])
def test_l4_pair_reuses_shared_low_history_without_full_alignment(monkeypatch, layout):
    from hfmodem.kestrel.rx import bw500_locator as loc, varahf500 as rx
    from hfmodem.kestrel.tx import varahf500_tx as tx
    frames = [tx.build_frame(bytes([k]) * rx.FRAME_PAYLOAD, 0x95) for k in (31, 73)]
    x = np.pad(tx.synth_burst(frames, onset=24000, level=4), (0, 24000))
    receiver = loc.Receiver()
    def forbidden(*args, **kwargs):
        pytest.fail('clean located pair repeated full-rate alignment')
    monkeypatch.setattr(rx, '_demod_aligned', forbidden)
    overs = []
    at = i = 0
    while at < len(x):
        block = x[at:at + layout[i % len(layout)]]
        overs.extend(receiver.feed(block, may_pair=True))
        at += len(block)
        i += 1
    overs.extend(receiver.feed(np.empty(0), may_pair=True, eof=True))
    complete = [o for o in overs if o.result.complete]
    assert len(complete) == 1
    assert [f.frame_bytes for f in complete[0].frames] == frames
    assert complete[0].low_rate
    assert receiver.low.samples * 8 == receiver.front.samples
    assert len(receiver.low.raw) <= receiver.low.capacity


def test_failed_l4_epochs_share_one_full_alignment(monkeypatch):
    from hfmodem.kestrel.rx import bw500_locator as loc, varahf500 as rx
    receiver = loc.Receiver()
    epochs = (loc.Epoch(0, .2, 0), loc.Epoch(64, .19, 0))
    over = loc.Over(loc.Candidate(4, 0, .2, 0, 64, epochs))
    receiver.raw.feed(np.zeros(230000))
    calls = []
    monkeypatch.setattr(loc, 'demod_local', lambda *a: (None, (0, 0), .2))
    def full(*args):
        calls.append(args[2:] or (None,))
        yield
        return None, {0: 0, 1: 0}, .2
    monkeypatch.setattr(rx, '_demod_aligned_steps', full)
    monkeypatch.setattr(rx, '_frame_gh', lambda *a: object())
    def decode(gh, fk, c0):
        yield
        return 0., c0, False, b'', False
    monkeypatch.setattr(rx, '_decode_cells_steps', decode)
    receiver._advance(over, False, True)
    assert calls == [(None,)]
    assert over.epoch_try == 1
    assert not over.result.complete


def test_responder_control_uses_shared_spectra_after_odd_carry(monkeypatch):
    from hfmodem.kestrel.vara import vara_arq as arq
    from hfmodem.tests.kestrel.test_bw500_locator_ownership import _station
    station = _station('responder')
    original = arq._fullrate_top3_track
    lengths = []
    def fullrate(samples, *args, **kwargs):
        lengths.append(len(samples))
        return original(samples, *args, **kwargs)
    monkeypatch.setattr(arq, '_fullrate_top3_track', fullrate)
    x = np.random.default_rng(129).normal(0, .0001, 4 * 48000)
    for start in range(0, len(x), 1024):
        station.on_rx_stream(x[start:start + 1024])
    assert max(lengths, default=0) < 4096
    assert station._control.transforms > 0


@pytest.mark.parametrize('at', [-2048, 0, 1024])
@pytest.mark.parametrize('shift', [-2, 0, 2])
def test_exact_short_close_keeps_missing_head_and_carrier_shifts(at, shift):
    from hfmodem.kestrel.vara import vara_arq as arq
    from hfmodem.tests.kestrel.test_bw500_locator_ownership import _station
    station = _station('initiator')
    kind = arq.VF.SESSION_DISCONNECT_REQ_RESPONDER_SHORT
    tones = np.asarray(station._burst_tones(station.called, kind))
    x = np.zeros(at + arq._span(kind) + 1024)
    n = (len(x) - arq.MK.STRIDE) // 32 + 1
    bins, clear = np.zeros((n, 3), int), np.zeros((n, 2))
    rows = np.rint((at + np.arange(len(tones)) * arq.MK.HOP + arq.MK._WOFF) / 32).astype(int)
    valid = (rows >= 0) & (rows < n)
    bins[rows[valid], 0], clear[rows[valid], 0] = tones[valid] + shift, 6
    track = bins, clear, np.zeros(n)
    assert station._peer_short_close(x, track)
    bins[rows[-1], 0] += 10
    assert not station._peer_short_close(x, track)


@pytest.mark.parametrize('cut', range(4))
@pytest.mark.parametrize('shift', [-1, 0, 1])
def test_exact_pair_gate_keeps_three_missing_leading_symbols(monkeypatch, cut, shift):
    from hfmodem.kestrel.vara import vara_arq as arq
    from hfmodem.tests.kestrel.test_bw500_locator_ownership import _station
    station = _station('responder')
    station._peer_shift = shift
    pairs = arq.VF.caller_turn_grant(station.called, station.bw)
    want = np.asarray(pairs[cut:])
    x = np.zeros((len(want) - 1) * arq.MK.HOP + arq.MK.STRIDE)
    n = (len(x) - arq.MK.STRIDE) // 32 + 1
    bins, clear = np.zeros((n, 3), int), np.zeros((n, 2))
    rows = np.rint((np.arange(len(want)) * arq.MK.HOP + arq.MK._WOFF) / 32).astype(int)
    bins[rows, :2], clear[rows, 1] = want + shift, 6
    monkeypatch.setattr(arq, '_frame_fit', lambda *a: (1., 0))
    track = bins, clear, np.zeros(n)
    assert station._complete_responder_pair(x, pairs, track, np.zeros((n, 70)))
    clear[rows[-1], 1] = 1
    assert not station._complete_responder_pair(x, pairs, track, np.zeros((n, 70)))
