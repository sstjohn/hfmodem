# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M2a acceptance gates. Full sweeps with larger volumes: ``python -m hfmodem.sabir.sim.m2``."""

import numpy as np
import pytest

from hfmodem.sabir.fec import CODES, QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.dsp import Constellation
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.modem import FS
from hfmodem.sabir.sim.m2 import Rung, add_noise_snr3k, notch, rung_fer
from hfmodem.sabir.sim.watterson import PROFILES, Watterson, tap_process

# -- Watterson channel model ---------------------------------------------------
def test_watterson_tap_statistics():
    rng = np.random.default_rng(0)
    for spread in (0.5, 1.0, 10.0):
        fs = 64.0 * spread
        n = int(fs * 2000.0 / spread)       # ~2000 fade time-constants
        h = tap_process(n, fs, spread, rng=rng)
        p = np.mean(np.abs(h) ** 2)
        assert abs(p - 1.0) < 0.15                          # unit power
        kurt = np.mean(np.abs(h) ** 4) / p**2
        assert abs(kurt - 2.0) < 0.15                       # Rayleigh envelope
        H = np.abs(np.fft.fft(h * np.hanning(n))) ** 2
        f = np.fft.fftfreq(n, 1 / fs)
        mu = np.sum(f * H) / np.sum(H)
        two_sigma = 2 * np.sqrt(np.sum((f - mu) ** 2 * H) / np.sum(H))
        assert abs(two_sigma - spread) / spread < 0.15      # Gaussian Doppler
        assert abs(mu) < 0.1 * spread                       # no phantom shift


def test_watterson_shift_and_independence():
    rng = np.random.default_rng(1)
    fs, n = 64.0, 64 * 2000
    h = tap_process(n, fs, 1.0, shift_hz=3.0, rng=rng)
    H = np.abs(np.fft.fft(h * np.hanning(n))) ** 2
    f = np.fft.fftfreq(n, 1 / fs)
    assert abs(np.sum(f * H) / np.sum(H) - 3.0) < 0.1
    a = tap_process(n, fs, 1.0, rng=rng)
    b = tap_process(n, fs, 1.0, rng=rng)
    xc = abs(np.mean(a * np.conj(b)))
    assert xc < 0.05                                        # independent taps


def test_watterson_profiles_and_power():
    assert set(PROFILES) >= {"good", "moderate", "poor", "nvis", "polar"}
    w = Watterson("poor", FS, np.random.default_rng(2))
    x = np.exp(2j * np.pi * 1500 * np.arange(48000) / FS)
    y = w(x)
    assert y.shape == x.shape
    assert len(w.taps) == 2
    assert 0.01 < np.mean(np.abs(y) ** 2) < 5.0             # one realisation


# -- higher-order constellations -----------------------------------------------
def test_generic_llr_matches_qpsk_closed_form():
    q = Constellation.create("qpsk")
    rng = np.random.default_rng(3)
    z = rng.standard_normal(300) + 1j * rng.standard_normal(300)
    w = rng.uniform(0.1, 5.0, 300)
    generic = Constellation("q", 2, q.points).llr(z, w)    # forces generic path
    assert np.allclose(q.llr(z, w), generic)


@pytest.mark.parametrize("name", ["16qam", "64qam"])
def test_qam_llr_soft_demap(name):
    c = Constellation.create(name)
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, 600 * c.bits_per_symbol)
    s = c.modulate(bits)
    llr = c.llr(s, 8.0)
    assert ((llr > 0) == (bits == 0)).all()                 # noiseless signs
    zn = s + 0.05 * (rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size))
    assert (c.demodulate(zn) == (c.llr(zn, 1.0) < 0)).all()  # max-log = ML hard


# -- the QC-LDPC ladder ---------------------------------------------------------
@pytest.mark.parametrize("rate", list(CODES))
def test_ldpc_ladder(rate):
    code = QCLDPC(rate)
    rng = np.random.default_rng(5)
    info = rng.integers(0, 2, code.k)
    cw = code.encode(info)
    assert (cw[: code.k] == info).all()                     # systematic
    assert code.syndrome_ok(cw)
    hard, ok, iters = code.decode((1.0 - 2.0 * cw) * 8.0)
    assert ok.all() and iters[0] == 1 and (hard[0] == cw).all()


@pytest.mark.parametrize("rate,repeat", [("r13", 2), ("r34", 1), ("r56", 1)])
def test_frame_codec_rates_and_repeat(rate, repeat):
    fc = FrameCodec(QCLDPC(rate), repeat)
    rng = np.random.default_rng(6)
    payload = rng.integers(0, 256, 2 * fc.data_bytes + 7, dtype=np.uint8).tobytes()
    bits = fc.encode(payload)
    assert bits.size == repeat * 3 * fc.code.n
    got, stats = fc.decode((1.0 - 2.0 * bits) * 4.0)
    assert got == payload and all(stats["crc_ok"])


# -- scattered-pilot loopback ----------------------------------------------------
@pytest.mark.parametrize("name", ["workhorse", "fast", "max"])
def test_scattered_loopback_clean(name):
    rung = Rung(name, n_codewords=2)
    rng = np.random.default_rng(7)
    payload = rng.integers(0, 256, rung.payload_len, dtype=np.uint8).tobytes()
    bits = rung.fc.encode(payload)
    tx = rung.phy.transmit(bits)
    rx = rung.phy.from_audio(rung.phy.to_audio(tx))
    hard, llr, res = rung.phy.receive(rx, n_symbols=rung.phy.n_symbols_for(bits.size))
    assert (hard[: bits.size] == bits).all()                # zero raw BER
    got, _ = rung.fc.decode(llr)
    assert got == payload
    assert res.masked.sum() == 0


def test_no_false_alarm_on_noise():
    phy = Phy(GEARS["workhorse"])
    rng = np.random.default_rng(8)
    for _ in range(10):
        noise = rng.standard_normal(60000) + 1j * rng.standard_normal(60000)
        assert phy.detector.detect(noise) is None


# -- the ladder over Watterson fading --------------------------------------------
def _fading_ok(name: str, profile: str | None, snr_db: float, n_frames: int,
               seed: int) -> int:
    rung = Rung(name, n_codewords=1)
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n_frames):
        payload, tx, n_syms = rung.frame(rng)
        y = tx if profile is None else Watterson(profile, FS, rng)(tx)
        y = add_noise_snr3k(y, snr_db, rng)
        try:
            _, llr, _ = rung.phy.receive(y, n_symbols=n_syms)
            got, _ = rung.fc.decode(llr)
        except ValueError:
            got = None
        ok += got == payload
    return ok


@pytest.mark.parametrize("name,profile,snr,seed,need", [
    ("robust", "poor", -2.0, 11, 7),
    ("workhorse", "poor", 8.0, 22, 6),
    ("workhorse34", "poor", 12.0, 33, 5),
    ("fast", "moderate", 16.0, 44, 6),
    ("max", "good", 24.0, 55, 6),
])
def test_rung_decodes_on_fading(name, profile, snr, seed, need):
    assert _fading_ok(name, profile, snr, 8, seed) >= need


def test_ladder_ordering_robust_below_workhorse():
    # at the robust rung's operating point the workhorse must be mostly gone
    assert _fading_ok("workhorse", "poor", -2.0, 8, 66) <= 3
    assert _fading_ok("robust", "poor", -2.0, 8, 11) >= 7


# -- dead-carrier masking ---------------------------------------------------------
def test_masking_helps_on_notched_channel():
    gear = GEARS["workhorse34"]
    dead = np.arange(8, 12)
    f_lo = gear.carrier_hz[dead[0]] - FS / gear.n_fft / 2
    f_hi = gear.carrier_hz[dead[-1]] + FS / gear.n_fft / 2
    kw = dict(n_frames=12, seed=99, notch_hz=(f_lo, f_hi))
    fer_plain = rung_fer("workhorse34", None, 8.0, rx_masking=False, **kw)
    fer_rx = rung_fer("workhorse34", None, 8.0, rx_masking=True, **kw)
    tx_mask = np.ones(gear.n_carriers, dtype=bool)
    tx_mask[dead] = False
    fer_tx = rung_fer("workhorse34", None, 8.0, tx_mask=tx_mask, **kw)
    assert fer_plain > 0.2                  # the notch bites at rate 3/4
    assert fer_rx < fer_plain               # RX erasure measurably helps
    assert fer_tx == 0.0                    # TX masking clears it entirely


def test_rx_detects_dead_carriers():
    rung = Rung("workhorse34")
    gear = rung.gear
    f_lo = gear.carrier_hz[8] - FS / gear.n_fft / 2
    f_hi = gear.carrier_hz[11] + FS / gear.n_fft / 2
    rng = np.random.default_rng(12)
    payload, tx, n_syms = rung.frame(rng)
    y = add_noise_snr3k(notch(tx, f_lo, f_hi), 10.0, rng)
    _, _, res = rung.phy.receive(y, n_symbols=n_syms)
    got = set(np.nonzero(res.masked)[0].tolist())
    # the fully-dead interior carriers are caught; the notch's edge carriers
    # retain real leakage energy and legitimately may stay live
    assert {9, 10} <= got <= {8, 9, 10, 11}


# -- TX-mask bookkeeping ----------------------------------------------------------
def test_tx_mask_capacity_and_roundtrip():
    phy = Phy(GEARS["workhorse"])
    tx_mask = np.ones(24, dtype=bool)
    tx_mask[8:12] = False
    full = phy.capacity(30)
    masked = phy.capacity(30, tx_mask)
    assert masked < full
    fc = FrameCodec(QCLDPC("r12"))
    rng = np.random.default_rng(13)
    payload = rng.integers(0, 256, fc.data_bytes, dtype=np.uint8).tobytes()
    bits = fc.encode(payload)
    tx = phy.transmit(bits, tx_mask)
    rx = add_noise_snr3k(tx, 15.0, rng)
    _, llr, _ = phy.receive(rx, n_symbols=phy.n_symbols_for(bits.size, tx_mask),
                            tx_mask=tx_mask)
    got, _ = fc.decode(llr)
    assert got == payload
