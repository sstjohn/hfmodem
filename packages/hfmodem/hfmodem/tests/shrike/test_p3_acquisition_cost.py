# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The P3 acquisition search, held to the dense form it replaced.

`p3acquire._search` once built every hypothesis' bands at the full 8 kHz and
the [offsets, starts, bits] tensors over them: 80 ms for the 4 s, 32-offset
cold changeover search on this Mac, about 200 ms on the shackbook's i5. It now
reads one sample of every `STEP` through `_grid` and takes the tensors as views
of one line, and the full-rate extent check filters 21 symbols rather than the
window (`p3acquire._inside_burst`). What it accepts must not move: codeword,
alignment, quality and refined offset, over every PCM16 fixture in the tree.
"""
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve, hilbert, resample_poly

from hfmodem.shrike import p3acquire, placement, rx, rxfront, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_slot_deadline import crops

FIXTURES = Path(__file__).with_name('fixtures')
FS = p3acquire.FS
CODES = tuple(range(len(rx._CS_TABLE)))
COLD = tuple(hz for hz in p3acquire.ENTRY_CONTROL_OFFSETS_HZ
             if hz not in (*p3acquire.OFFSETS_HZ, -100.0, 100.0, 5.0, -5.0))


def _dense(audio, offsets, codes, symbols, reach):
    """`p3acquire._search` as of 900809b2e: full-rate bands, dense tensors."""
    small = resample_poly(np.asarray(audio, dtype=float), 1, p3acquire.DECIMATION)
    analytic = hilbert(small)
    clock = np.arange(len(small)) / p3acquire.SEARCH_FS
    sps = p3acquire.SEARCH_FS // 100
    pulse = rx._pulse(sps)
    delay = (len(pulse)-1)//2
    starts = np.arange(0, len(small)-reach, sps//8)
    idx = starts[:, None] + np.arange(symbols)[None, :]*sps + delay
    table = rx._CS_TABLE[list(codes)][:, :symbols-1]
    signs = 1.0-2.0*table
    shifted = (analytic[None, :] * np.exp(
        -2j*np.pi*np.asarray(offsets, float)[:, None]*clock[None, :])).real
    bands = {cn: fftconvolve(shifted * rx.carrier(cn, p3acquire.SEARCH_FS, 0,
                                                  shifted.shape[1]),
                             pulse[None, :], axes=1) for cn in spec.VH_CHANNELS}
    y = np.asarray([bands[cn][:, idx] for cn in spec.VH_CHANNELS])
    diffs = y[..., 1:]*np.conj(y[..., :-1])
    combined = diffs.real.sum(axis=0)
    errors = np.sum((combined[:, None] < 0) != table[None, :, None], axis=3)
    cis = np.argmin(errors, axis=1)
    exact = errors[np.arange(len(offsets))[:, None], cis,
                   np.arange(len(starts))[None, :]] == 0
    paired = exact[:, :-1] & exact[:, 1:] & (cis[:, :-1] == cis[:, 1:])
    adjacent = np.zeros_like(exact)
    adjacent[:, 1:] |= paired
    adjacent[:, :-1] |= paired
    adjacent &= exact
    trial_signs = signs[cis]
    qualities = np.sum(combined*trial_signs, axis=2) / np.maximum(
        np.sum(np.abs(diffs), axis=(0, 3)), 1e-30)
    carrier_quality = np.sum(diffs.real*trial_signs, axis=3) / np.maximum(
        np.sum(np.abs(diffs), axis=3), 1e-30)
    accepted = (adjacent & (qualities >= .65)
                & (carrier_quality.max(axis=0) >= .85))
    return accepted, cis, qualities, diffs, trial_signs


def _recordings():
    for path in sorted(FIXTURES.rglob('*.wav')):
        try:
            with wave.open(str(path)) as wav:
                if (wav.getframerate(), wav.getsampwidth(),
                        wav.getnchannels()) != (FS, 2, 1):
                    continue
                raw = wav.readframes(wav.getnframes())
        except wave.Error:
            continue
        yield path.name, np.frombuffer(raw, '<i2').astype(np.float32) / 32768


def _e07():
    row = crops()['e07']
    return recorded_pcm({'file': f"slot-deadline-0910/{row['file']}",
                         'sha256': row['pcm_sha256']})


def test_the_grid_is_the_bands_it_samples():
    """On the 5 Hz grid by bin shifts, off it by a transform of its own: both
    the bands `_bands` filters, at the alignments the search reads."""
    audio = np.asarray(_e07()[:round(3.9 * FS)], dtype=float)
    analytic = hilbert(resample_poly(audio, 1, p3acquire.DECIMATION))
    clock = np.arange(len(analytic)) / p3acquire.SEARCH_FS
    offsets = (*p3acquire.ENTRY_CONTROL_OFFSETS_HZ, -12.1, 2.0)
    delay = (len(rx._pulse(p3acquire.SEARCH_FS // 100)) - 1) // 2
    count = (len(analytic) - delay) // p3acquire.STEP
    bands = p3acquire._bands(analytic, clock, offsets,
                             rx._pulse(p3acquire.SEARCH_FS // 100))
    grid = p3acquire._grid(analytic, offsets, delay, count)
    for cn in spec.VH_CHANNELS:
        want = bands[cn][:, delay::p3acquire.STEP][:, :count]
        assert np.abs(grid[cn] - want).max() <= 1e-11 * np.abs(want).max()


@pytest.mark.parametrize('codes, symbols, reach, seconds', [
    (CODES, 21, round(.22 * p3acquire.SEARCH_FS), 4.0),
    ((placement.BREAKIN_CS,), 21, round(.22 * p3acquire.SEARCH_FS), 4.0),
    (CODES, p3acquire.AIM_BITS + 1, p3acquire.AIM_BITS * 80 + 41, 1.17),
])
def test_the_search_accepts_what_the_dense_tensors_did(codes, symbols, reach,
                                                       seconds):
    """Every fixture in the tree, cut to the receive windows the search is
    handed, on the entry receiver's 41 hypotheses: the same alignments
    accepted, the same codeword at each, the quality to rounding and the
    refined offset to the tenth of a hertz it reports."""
    n, windows, hits = round(seconds * FS), 0, 0
    for name, audio in _recordings():
        for lo in range(0, max(1, audio.size - n // 2), n):
            seg = audio[lo:lo + n]
            if seg.size < round(.5 * FS):
                break
            offsets = p3acquire.ENTRY_CONTROL_OFFSETS_HZ
            got = p3acquire._search(seg, offsets, codes, symbols, reach)
            accepted, cis, qualities, diffs, signs = _dense(
                seg, offsets, codes, symbols, reach)
            windows += 1
            assert (got.starts == np.arange(accepted.shape[1]) * p3acquire.STEP).all()
            assert (got.accepted == accepted).all(), (name, lo)
            trial, k = np.nonzero(accepted)
            hits += len(trial)
            assert (got.cis[trial, k] == cis[trial, k]).all(), (name, lo)
            assert np.abs(got.qualities[trial, k]
                          - qualities[trial, k]).max(initial=0) < 1e-9
            for t, j in zip(trial, k):
                assert (p3acquire._refined(got.diffs, got.trial_signs, t, j,
                                           offsets[t])
                        == p3acquire._refined(diffs, signs, t, j, offsets[t]))
    # The published tree ships fewer fixtures than the dev tree (47 windows at
    # 4 s, against more than 60), so the floor is what both trees clear.
    assert windows > 40 and hits > 20, (windows, hits)


def test_the_extent_check_reads_the_whole_windows_envelope():
    """`_inside_burst` over its 21 symbols against the envelope of the whole
    window, at every alignment the search accepts in the e07 crop."""
    audio = np.asarray(_e07(), dtype=float)
    found = p3acquire._search(audio[:round(3.9 * FS)], (0.0,), CODES, 21,
                              round(.22 * p3acquire.SEARCH_FS))
    shifted = audio[:round(3.9 * FS)]
    pulse = rx._pulse(p3acquire.SPS)
    z = {cn: rx._baseband(shifted, cn, FS, pulse) for cn in spec.VH_CHANNELS}
    env = rxfront._tone_envelope(z, (len(pulse) - 1) // 2, 0, len(shifted))
    ats = list(found.starts * p3acquire.DECIMATION)
    ats += [len(shifted) - 20 * p3acquire.SPS - 1, len(shifted) - 20 * p3acquire.SPS]
    inside = [rxfront._inside_burst(env, 0, at) for at in ats]
    assert any(inside) and not all(inside)
    assert [p3acquire._inside_burst(shifted, at) for at in ats] == inside


def test_identical_copies_rank_by_rule_not_by_rounding():
    """Two copies of one CS3 a cycle apart score alike to 1e-15, and which one
    won was whichever the transforms' rounding favoured -- the later for
    `changeover` and the earlier for `head_aims` at 900809b2e. Ranked to
    `RANK_DIGITS`, a tie keeps each caller's own order for an exact one, at
    any level the audio arrives at."""
    word = placement.control_signal(placement.BREAKIN_CS)
    for gain in (1.0, .7, .013):
        audio = np.zeros(120000)
        for at in (120, 60120):
            audio[at:at + len(word)] = gain * word
        assert p3acquire.changeover(audio).event.start == 61140
        assert p3acquire.control_signal(audio[:100000]).event.start == 61140
        assert p3acquire.head_aims(audio) == [(61140, 0.0), (1140, 0.0)]
        assert p3acquire.cut_word(audio, 1140, 0.0) == placement.BREAKIN_CS


def test_a_bodyless_tie_across_offsets_keeps_the_first(monkeypatch):
    """A head with no body behind it, accepted on two hypotheses whose scores
    differ past `RANK_DIGITS` only: the first one supplied is the candidate,
    however the rounding fell."""
    audio = np.zeros(48000)
    word = placement.control_signal(placement.BREAKIN_CS)
    audio[4800:4800 + len(word)] = word
    search = p3acquire._search

    def nudged(*args):
        found = search(*args)
        both = found.accepted.all(axis=0)
        assert both.any()
        found.accepted[:] = both
        found.qualities[1, both] = found.qualities[0, both] + 1e-15
        return found

    monkeypatch.setattr(p3acquire, '_search', nudged)
    got = p3acquire.changeover(audio, offsets=(0.0, 2.5))
    assert got.event.packet is None and got.coarse_hz == 0.0


def _best_ms(call, reps=3):
    call()
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        took = time.perf_counter() - t0
        best = took if best is None else min(best, took)
    return best * 1e3


def test_the_cold_changeover_search_costs_a_fraction_of_the_dense_one():
    """The receiver's main offender -- `_p3_changeover_packet`'s cold list over
    the 4 s it keeps -- measured against the dense form in this process, so the
    ratio is the code's and not the machine's. 9.8x on this Mac, 81 ms to 8.3."""
    audio = np.asarray(_e07()[:round(4.0 * FS) - 1], dtype=float)
    reach = round(.22 * p3acquire.SEARCH_FS)
    breakin = (placement.BREAKIN_CS,)
    dense = _best_ms(lambda: _dense(audio, COLD, breakin, 21, reach))
    fast = _best_ms(lambda: p3acquire._search(audio, COLD, breakin, 21, reach))
    assert fast * 4 < dense, (fast, dense)
