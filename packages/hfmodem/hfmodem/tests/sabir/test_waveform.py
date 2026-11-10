# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Carrier placement, emitted spectra and acquisition regression gates."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.sabir.dsp.fading import common_phase
from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.rate import CENTER_HZ, FS
from hfmodem.sabir.sim.design import spectrum


@pytest.mark.parametrize('name', GEARS)
def test_exact_midpoint_real_audio_and_spectral_skirts(name):
    gear = GEARS[name]
    spacing = FS / gear.n_fft
    assert np.mean(gear.carrier_hz) == CENTER_HZ
    assert gear.carrier_hz[0] == gear.first_carrier * spacing + gear.mixer_hz
    assert gear.mixer_hz - CENTER_HZ == spacing / 2
    phy = Phy(gear)
    codec = FrameCodec(QCLDPC(gear.code), gear.repeat)
    payload = np.random.default_rng(19).bytes(4 * codec.data_bytes)
    bits = codec.encode(payload)
    wave = phy.transmit(bits)
    measured = spectrum(wave)
    assert measured['outside_100_2900_db'] < -60
    assert measured['band99_hz'][1] - measured['band99_hz'][0] < 2800
    _, llr, _ = phy.receive(phy.from_audio(phy.to_audio(wave)),
                            n_symbols=phy.n_symbols_for(bits.size), dd=1)
    assert codec.decode(llr)[0] == payload


def test_filter_is_linear_without_wraparound():
    phy = Phy(GEARS['fast'])
    x = np.zeros(12000, dtype=complex)
    x[-100] = 1
    y = phy._bandpass(x)
    assert np.max(abs(y[:7000])) < 1e-14
    assert np.max(abs(y[-500:])) > .01


def test_silence_is_not_a_preamble_or_control():
    from hfmodem.sabir.floor.mfsk import FloorModem
    phy = Phy()
    silence = np.zeros(48000 * 10, dtype=complex)
    assert phy.detector.detect(silence) is None
    block, stats = FloorModem().receive(silence, 44)
    assert block is None and stats['sync'] is None


def test_half_bin_is_required_for_demodulation():
    phy = Phy()
    codec = FrameCodec()
    payload = bytes(range(61))
    bits = codec.encode(payload)
    wave = phy.transmit(bits)
    phy.mixer_hz = CENTER_HZ  # negative control: omit the half-bin correction
    _, llr, _ = phy.receive(wave, n_symbols=phy.n_symbols_for(bits.size))
    assert codec.decode(llr)[0] != payload


def test_phase_tracking_across_wrapped_increment():
    # Long frame at 3.5 Hz/s, with increments crossing pi during the burst.
    phy = Phy()
    symbols = np.arange(180)
    t = symbols * (phy.n_fft + phy.cp) / FS
    expected = 2 * np.pi * (.4 * t + .5 * 3.5 * t*t)
    mask = phy.lattice_mask(symbols.size)
    pilots = np.exp(1j * expected[:, None]) * mask
    actual = common_phase(pilots, mask)
    np.testing.assert_allclose(np.exp(1j * actual), np.exp(1j * expected), atol=1e-10)


def test_invalid_allocations_cannot_hang_capacity_search():
    with pytest.raises(ValueError, match='even'):
        replace(GEARS['workhorse'], n_carriers=23)
    phy = Phy()
    with pytest.raises(ValueError, match='no data capacity'):
        phy.n_symbols_for(100, tx_mask=np.zeros(24, dtype=bool))
    with pytest.raises(ValueError, match='loading'):
        phy.n_symbols_for(100, loading=np.ones(24))
    with pytest.raises(ValueError, match='coded bits'):
        phy.transmit(np.array([], dtype=int))
