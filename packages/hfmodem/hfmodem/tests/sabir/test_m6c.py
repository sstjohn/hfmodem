# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M6c acceptance gates. Full sweeps with numbers: ``python -m hfmodem.sabir.sim.m6c``."""

import numpy as np
from dataclasses import replace

from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.modem import CP, LATTICE_DF, LATTICE_STAG, N_FFT
from hfmodem.sabir.sim.impulse import add_impulse_noise
from hfmodem.sabir.sim.m6c import IMP, fer, net_bps, papr_db


# -- pilot overhead and timing margin ------------------------------------------
def test_default_lattice_capacity_and_timing_margin():
    phy = Phy(GEARS["workhorse34"])
    m = phy.lattice_mask(9)
    s, r = np.meshgrid(np.arange(9), np.arange(24), indexing="ij")
    assert (m == ((r - LATTICE_STAG * s) % LATTICE_DF == 0)).all()
    assert phy.n_fft == N_FFT and phy.cp == CP
    assert phy.detector.early_bias == 96
    assert phy.capacity(9) == 9 * 24 * 2 - 2 * m.sum()  # pilots cost 2 bits


# -- 1. decoder-aided iterative estimation ---------------------------------------
def test_decoder_feedback_beats_plain_and_dd():
    kw = dict(n_frames=40, seed=90)
    plain = fer("workhorse34", "poor", 9.0, **kw)
    dd1 = fer("workhorse34", "poor", 9.0, dd=1, **kw)
    fb2 = fer("workhorse34", "poor", 9.0, fb_iters=2, **kw)
    assert fb2 <= plain - 0.10
    assert fb2 <= dd1


# -- 2. impulse noise + blanker --------------------------------------------------
def test_impulse_model_adds_bursty_power():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(48000) + 1j * rng.standard_normal(48000))
    y = add_impulse_noise(x, np.random.default_rng(1), **IMP)
    excess = np.abs(y) ** 2 - np.abs(x) ** 2
    # heavy-tailed: the added power is concentrated in a small duty cycle
    hot = np.abs(y) > 4 * np.median(np.abs(y))
    assert excess.sum() > 0 and 0 < hot.mean() < 0.05


def test_blanker_rescues_qam_under_impulses():
    kw = dict(n_frames=15, seed=160, impulses=IMP)
    raw = fer("fast", "moderate", 16.0, **kw)
    blanked = fer("fast", "moderate", 16.0, blank=3.5, **kw)
    assert raw >= 0.7
    assert blanked <= 0.4


def test_blanker_inert_on_clean_air():
    kw = dict(n_frames=20, seed=140)
    off = fer("fast", "moderate", 14.0, **kw)
    on = fer("fast", "moderate", 14.0, blank=3.5, **kw)
    assert on <= off + 0.10


# -- 3. doppler mid-tier ---------------------------------------------------------
def test_doppler_gear_holds_polar_where_workhorse_dies():
    wk = fer("workhorse", "polar", 14.0, n_frames=12, seed=140)
    dop = fer("doppler", "polar", 14.0, n_frames=12, seed=140)
    assert wk >= 0.75
    assert dop <= 0.25


def test_doppler_gear_subpolar_and_rate():
    assert fer("doppler", "subpolar", 8.0, n_frames=8, seed=80) == 0.0
    assert net_bps("doppler") > 2 * net_bps("robust")


# -- 4. ACE ----------------------------------------------------------------------
def test_ace_cuts_papr_and_still_decodes():
    for name, deep in (("fast", 4.5), ("max", 6.0)):
        base = GEARS[name]
        ace = replace(base, clip_papr_db=deep, ace=True)
        assert papr_db(ace) < papr_db(base) - 1.0
        assert fer(ace, None, 30.0, n_frames=3, seed=1) == 0.0


# -- 5. sparse pilots ------------------------------------------------------------
def test_sparse34_rate_gain_and_decode():
    assert net_bps("sparse34") / net_bps("workhorse34") > 1.08
    # Sparse pilots save airtime but trade channel-estimation margin. Compare
    # under the same channel/SNR; neither finite sample implies zero FER.
    dense = fer("workhorse34", "good", 10.0, n_frames=64, seed=100)
    sparse = fer("sparse34", "good", 10.0, n_frames=64, seed=100)
    assert sparse <= .05 and sparse <= dense + .05
