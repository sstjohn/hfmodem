# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.kestrel.rx import bw500_locator as loc, varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd


def _power(x):
    f = np.fft.rfftfreq(len(x), 1 / rx.FS)
    return 2 * np.sum(abs(np.fft.rfft(x)[(f >= 1150) & (f < 1850)]) ** 2) / len(x) ** 2


def _band(level, snr, seed, notched=False, cfo=0):
    r = rx.INDEX_RECORDS[level]
    frame = tx.build_l3_frame(np.random.default_rng(11000 + seed).bytes(r.payload), 0, level)
    end = 24000 + (r.lead + r.ncols) * r.dw
    signal = tx.synth_burst(frame, onset=24000, length=end + 24000,
                            level=level, lead=VB6Rnd(seed + 1))
    if cfo:
        signal = np.real(hilbert(signal) * np.exp(2j * np.pi * cfo * np.arange(len(signal)) / rx.FS))
    z = np.fft.rfft(np.random.default_rng(21000 + seed).normal(size=len(signal)))
    f = np.fft.rfftfreq(len(signal), 1 / rx.FS)
    z[(f < 300) | (f >= 2700)] = 0
    if notched:
        z *= 1 - .95 * np.exp(-((f - 1500) / 100) ** 2)
    noise = np.fft.irfft(z, len(signal))
    noise *= np.sqrt(_power(signal[24000:end]) / _power(noise[24000:end]) / 10 ** (snr / 10))
    return rx._l3_band(signal + noise, 24000, r.lead + r.ncols, level, raw=True), frame


@pytest.mark.parametrize('level,snr,previous,pooled', [(1, -10, 8, 10), (2, -7, 3, 8), (3, -5, 9, 9)])
def test_fine_seed_recoveries(level, snr, previous, pooled):
    counts = [0, 0, 0]
    for seed in range(10):
        band, frame = _band(level, snr, seed)
        lead = rx.INDEX_RECORDS[level].lead
        results = [rx._l3_decode_llr(rx._l3_llr(band, lead, level, pooled=p), level)
                   for p in (False, True)] + [rx._l3_decode(band, lead, level, pooled=None)]
        for k, result in enumerate(results):
            counts[k] += result == (frame, True)
        assert not results[0][1] or results[2] == results[0]
    assert counts == [previous, pooled, pooled]


def test_notched_noise_retains_per_bin_reads(monkeypatch):
    modes = []
    original = rx._l3_llr
    def llr(*args, pooled):
        modes.append(pooled)
        return original(*args, pooled=pooled)
    monkeypatch.setattr(rx, '_l3_llr', llr)
    for seed in range(10):
        band, frame = _band(1, -10, seed, notched=True)
        assert rx._l3_decode(band, rx.INDEX_RECORDS[1].lead, 1, pooled=None) == (frame, True)
        assert rx._l3_decode(band, rx.INDEX_RECORDS[1].lead, 1, pooled=True) == (b'', False)
    assert modes == [False] * 10


@pytest.mark.parametrize('cfo,seed', [(-5, 3), (-5, 6), (5, 1)])
def test_flat_noise_preserves_offset_reads_when_pooled_fails(cfo, seed):
    band, frame = _band(1, -10, seed, cfo=cfo)
    lead = rx.INDEX_RECORDS[1].lead
    assert not rx._l3_decode_llr(rx._l3_llr(band, lead, 1, pooled=True), 1)[1]
    assert rx._l3_decode(band, lead, 1, pooled=None) == (frame, True)


@pytest.mark.parametrize('level', [1, 2, 3])
def test_pooled_metrics_are_soft_bounded_and_scale_invariant(level):
    band, _ = _band(level, {1: -10, 2: -7, 3: -5}[level], 0)
    lead = rx.INDEX_RECORDS[level].lead
    llr = rx._l3_llr(band, lead, level, pooled=True)
    assert np.all(np.isfinite(llr)) and np.max(abs(llr)) <= 8
    assert np.any((abs(llr) > .01) & (abs(llr) < 7.99))
    np.testing.assert_allclose(rx._l3_llr(band * 7, lead, level, pooled=True), llr, atol=1e-12)


@pytest.mark.parametrize('observe_pairs', [False, True])
@pytest.mark.parametrize('level,seed,snr,complete', [(2, 43, -7, True), (2, 89, -6, True),
                                                   (1, 37, -9, False)])
def test_pooled_pair_recovery_preserves_alignment_and_partner_identity(level, seed, snr, complete, observe_pairs):
    r = rx.INDEX_RECORDS[level]
    rng = np.random.default_rng(11000 + seed)
    frames = [tx.build_l3_frame(rng.bytes(r.payload), 0x95, level) for _ in range(2)]
    end = 24000 + (r.lead + 2 * r.ncols) * r.dw
    signal = tx.synth_burst(frames, onset=24000, length=end + 24000,
                            level=level, lead=VB6Rnd(seed + 1))
    f = np.fft.rfftfreq(len(signal), 1 / rx.FS)
    z = np.fft.rfft(np.random.default_rng(21000 + seed).normal(size=len(signal)))
    z[(f < 300) | (f >= 2700)] = 0
    noise = np.fft.irfft(z, len(signal))
    noise *= np.sqrt(_power(signal[24000:end]) / _power(noise[24000:end]))
    samples = signal + noise * 10 ** (-snr / 20)
    receiver, overs, at = loc.Receiver(observe_pairs=observe_pairs), [], 0
    cuts = np.random.default_rng(31000 + seed)
    while at < len(samples):
        stop = at + int(cuts.integers(1, 5001))
        overs.extend(receiver.feed(samples[at:stop], may_pair=True))
        at = stop
    overs.extend(receiver.feed(np.empty(0), may_pair=True, eof=True))
    assert frames[1] in [f.frame_bytes for o in overs for f in o.frames if f.crc_ok]
    completed = [o for o in overs if o.result.complete]
    assert len(completed) <= 1
    assert all([f.frame_bytes for f in o.frames] == frames for o in completed)
    if complete:
        assert completed


@pytest.mark.parametrize('marker,complete', [(0x95, True), (0x97, False)])
def test_late_first_frame_recovery_obeys_pair_marker(monkeypatch, marker, complete):
    r = rx.INDEX_RECORDS[1]
    body = tx.build_l3_frame(bytes(r.payload), marker, 1)
    partner = tx.build_l3_frame(b'x' * r.payload, 0x95, 1)
    first = rx.FrameResult(0, (0,), r.lead, 1., False, b'', level=1, frames_in_burst=2)
    second = rx.FrameResult(0, (0,), r.lead + r.ncols, 1., True, partner, level=1, frames_in_burst=2)
    receiver = loc.Receiver()
    end = (r.lead + r.ncols) * r.dw
    over = loc.Over(loc.Candidate(1, 0, 24., 0, 0), frames=[first], pair=True,
                    end=end, alignment=(0, r.lead, 24))
    receiver.raw.feed(np.zeros(end + r.ncols * r.dw + rx.FS // 2))
    monkeypatch.setattr(rx, '_l3_band', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_l3_frames', lambda *a, **kw: [first, second])
    monkeypatch.setattr(rx, '_l3_decode', lambda *a, **kw: (body, True))
    receiver._advance(over, may_pair=True, eof=True)
    assert over.result.complete == complete
    assert over.frames[1].crc_ok and over.frames[1].frame_bytes == partner
