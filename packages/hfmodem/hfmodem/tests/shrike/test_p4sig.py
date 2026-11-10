# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the PACTOR-4 signature detector may name, and what it must not.

The emission both gateways switch to was on tape for two days as `nothing
heard`. `p4sig` names it from two published sequences -- the SF-16 spreading
train and the Chu19 header of [SCS-P4] §6.2 -- and the risk a matched filter
carries is the same one every codeword search here has paid for: enough
templates over enough alignments mint accepts from noise. So the synthetic half
proves the construction round-trips with the root and shift recovered exactly,
and the capture half holds the knee against the two real emissions and every
negative population the sessions carry, pinned by name for the same reason
`test_answer_band.py` pins its windows -- a detector that regresses on them
regresses silently.

Run:  pytest hfmodem/tests/shrike/test_p4sig.py
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.shrike import p4sig
from hfmodem.tests import evidence

FS = 48000


def _synth_header(root: int, shift: int, *, cfo: float = 0.0,
                  snr_db: float = 20.0, lead: float = 0.05,
                  seed: int = 1) -> np.ndarray:
    k = np.arange(p4sig.CHU_LEN)
    chu = np.exp(-1j * np.pi * root * k * (k + 1) / p4sig.CHU_LEN)
    chips = (np.repeat(chu[(k + shift) % p4sig.CHU_LEN], p4sig.SPREAD_FACTOR)
             * np.tile(p4sig.SPREAD16, p4sig.CHU_LEN))
    n = int((2 * lead + p4sig.HEADER_CHIPS / p4sig.CHIP_RATE) * FS)
    t = np.arange(n) / FS
    idx = np.floor((t - lead) * p4sig.CHIP_RATE).astype(int)
    live = (idx >= 0) & (idx < p4sig.HEADER_CHIPS)
    z = np.zeros(n, complex)
    z[live] = chips[idx[live]]
    x = np.real(z * np.exp(2j * np.pi * (p4sig.CENTRE_HZ + cfo) * t))
    rng = np.random.default_rng(seed)
    rms = np.sqrt((x ** 2).mean())
    return (x + rng.normal(0, rms / 10 ** (snr_db / 20), n)).astype(np.float32)


def test_spread_score_reads_the_construction():
    assert p4sig.spread_score(_synth_header(7, 5)) >= p4sig.SPREAD_KNEE
    assert p4sig.spread_score(_synth_header(14, 3, cfo=8.0)) >= p4sig.SPREAD_KNEE


def test_header_score_recovers_root_and_shift():
    r, root, shift, _ = p4sig.header_score(_synth_header(14, 3, cfo=6.0))
    assert (root, shift) == (14, 3)
    assert r > p4sig.spread_score(_synth_header(14, 3, cfo=6.0))


def test_noise_and_fsk_stay_under_the_knee():
    rng = np.random.default_rng(3)
    n = int(0.28 * FS)
    assert p4sig.spread_score(
        rng.normal(0, 0.05, n).astype(np.float32)) < p4sig.SPREAD_KNEE
    t = np.arange(n) / FS
    tone = np.where(np.floor(t * 100).astype(int) % 2 == 0, 1400.0, 1600.0)
    fsk = np.sin(2 * np.pi * np.cumsum(tone) / FS).astype(np.float32)
    assert p4sig.spread_score(0.3 * fsk) < p4sig.SPREAD_KNEE


def test_a_short_window_scores_zero():
    assert p4sig.spread_score(np.zeros(int(0.1 * FS), np.float32)) == 0.0


#: The two real emissions, by session and listen window, and the windows of the
#: same sessions the PACTOR-1 reader answered in. The named subsets are the ones
#: the knee was measured to hold on; the emissions also fill windows the knee
#: does not reach (0.185-0.27), which is the miss the knee's own note prices in.
NAMED = {
    "onair-0829-1343": [f"hold_{n:02d}" for n in (8, 9, 11, 12, 13, 14, 15, 16,
                                                  17, 18, 19, 20, 21, 22)],
    "onair-0828-1844": [f"hold_{n:02d}" for n in (7, 8, 9, 10, 11, 16, 18, 19,
                                                  20, 21)],
}
READ = {
    "onair-0829-1343": [f"hold_{n:02d}" for n in range(1, 8)]
                       + [f"rx_{n:02d}" for n in range(1, 7)],
    "onair-0828-1844": [f"hold_{n:02d}" for n in range(1, 7)],
}


@pytest.mark.parametrize("name", sorted(NAMED))
def test_the_real_emissions_clear_the_knee(name):
    from hfmodem.shrike import rxfront
    d = evidence.CAPTURES / name
    if not d.is_dir():
        pytest.skip(f"{d} not on this machine")
    for stem in NAMED[name]:
        r = p4sig.spread_score(rxfront.load_wav(str(d / f"{stem}.wav")))
        assert r >= p4sig.SPREAD_KNEE, f"{name}/{stem} regressed to {r:.3f}"
    for stem in READ[name]:
        r = p4sig.spread_score(rxfront.load_wav(str(d / f"{stem}.wav")))
        assert r < p4sig.SPREAD_KNEE, f"{name}/{stem} false-named at {r:.3f}"


def test_genuine_pactor3_is_not_named():
    from hfmodem.shrike import rxfront
    ref = evidence.CORPUS / "PIII_Complete_1.wav"
    if not ref.exists():
        pytest.skip(f"{ref} not on this machine")
    x = rxfront.load_wav(str(ref))
    for a, b in ((4.66, 4.85), (6.82, 7.05), (9.02, 9.25), (48.0, 48.25)):
        r = p4sig.spread_score(x[int(a * FS):int(b * FS)])
        assert r < p4sig.SPREAD_KNEE, f"P3 at {a}s named at {r:.3f}"


def _screens_in(seg, fs=48000.0):
    z, d = p4sig.comb_screen(seg, fs)
    return z >= p4sig.COMB_KNEE and p4sig.comb_family(d)


#: Windows whose whole-window comb clears the screen. Three named windows sit
#: outside it (0829 hold_17/hold_21, 0828 hold_16) -- the comb reads the
#: repetitive signaling, not the Chu header, and a window that only clears the
#: correlation knee is exactly the miss the two-track screen prices in.
COMBED = {
    "onair-0829-1343": [f"hold_{n:02d}" for n in (8, 9, 11, 12, 13, 14, 15,
                                                  16, 18, 19, 20, 22)],
    "onair-0828-1844": [f"hold_{n:02d}" for n in (7, 8, 9, 10, 11, 18, 19,
                                                  20, 21)],
}


def test_comb_screen_is_scoped_to_the_signaling():
    assert not _screens_in(_synth_header(14, 3))
    rng = np.random.default_rng(3)
    assert not _screens_in(rng.normal(0, 0.05, int(0.28 * FS)))


@pytest.mark.parametrize("name", sorted(COMBED))
def test_the_real_emissions_screen_in(name):
    from hfmodem.shrike import rxfront
    d = evidence.CAPTURES / name
    if not d.is_dir():
        pytest.skip(f"{d} not on this machine")
    for stem in COMBED[name]:
        seg = rxfront.load_wav(str(d / f"{stem}.wav"))
        assert _screens_in(seg), f"{name}/{stem} fell out of the screen"
    for stem in READ[name]:
        seg = rxfront.load_wav(str(d / f"{stem}.wav"))
        assert not _screens_in(seg), f"{name}/{stem} screened in"


def test_the_pactor3_grid_is_rejected_by_the_family():
    from hfmodem.shrike import rxfront
    ref = evidence.CORPUS / "PIII_Complete_1.wav"
    if not ref.exists():
        pytest.skip(f"{ref} not on this machine")
    x = rxfront.load_wav(str(ref))
    for a, b in ((4.66, 4.85), (6.82, 7.05), (9.02, 9.25), (48.0, 48.25)):
        assert not _screens_in(x[int(a * FS):int(b * FS)]), f"P3 at {a}s"
    z, d = p4sig.comb_screen(x[int(9.02 * FS):int(9.25 * FS)])
    assert z >= p4sig.COMB_KNEE and not p4sig.comb_family(d)
