# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the modulator puts on the air, measured."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import mil110, phy, s4285, s4539, spec, tone16, tone39
from hfmodem.bateleur.measure import spectrum
from hfmodem.bateleur.modem import Mode
from hfmodem.bateleur.offair import OCCUPIED_AUDIO_HZ


_minus_26db_span = spectrum

_CASES = ([(mil110.transmit, m) for m in spec.MODES110]
          + [(s4285.transmit, m) for m in spec.MODES4285]
          + [(s4539.transmit, m) for m in s4539.MODES4539])


@pytest.mark.parametrize("tx,mode", _CASES, ids=[f"{t.__module__[16:]}-{m}" for t, m in _CASES])
@pytest.mark.parametrize("bw", ("3000", "3200"))
def test_every_mode_stays_inside_its_declared_span(tx, mode, bw):
    lo, hi = OCCUPIED_AUDIO_HZ[bw]
    family = {mil110.transmit: "110", s4285.transmit: "4285", s4539.transmit: "4539"}[tx]
    for seed, n in ((0, 1), (1, max(1, Mode.parse(family, mode).most_bytes(30)))):
        data = bytes(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8))
        a, b = _minus_26db_span(tx(data, mode, trim=bw == "3000"))
        assert lo <= a and b <= hi, (mode, a, b)


@pytest.mark.parametrize("mode", ["2400/288/S", "1200/63/S", "600/33/A10.8", "300/17/SF",
                                  "150/9/S", "75/36/S", "600/33/SX"])
@pytest.mark.parametrize("bw", ("3000", "3200"))
def test_the_39_tone_mode_stays_inside_the_same_span(mode, bw):
    """The host can switch the modem to it at any time, so bateleur's
    declared span must hold it as well; the pilot and keyed skirts are included."""
    lo, hi = OCCUPIED_AUDIO_HZ[bw]
    for seed, n in ((0, 1), (1, max(1, Mode.parse("39", mode).most_bytes(30)))):
        data = bytes(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8))
        a, b = _minus_26db_span(tone39.transmit(data, mode))
        assert lo <= a and b <= hi, (mode, a, b)


@pytest.mark.parametrize("rate", [2400, 1200, 600, 300, 150, 75])
@pytest.mark.parametrize("bw", ("3000", "3200"))
def test_the_16_tone_mode_stays_inside_the_same_span(rate, bw):
    lo, hi = OCCUPIED_AUDIO_HZ[bw]
    for seed, n in ((0, 1), (1, max(1, Mode.parse("16", str(rate)).most_bytes(30)))):
        data = bytes(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8))
        a, b = _minus_26db_span(tone16.transmit(data, rate))
        assert lo <= a and b <= hi, (rate, a, b)


@pytest.mark.parametrize("name", ["75W", "75W/1700", "50N", "300W", "75V2", "75V3D", "75V14"])
@pytest.mark.parametrize("bw", ("3000", "3200"))
def test_fsk_modes_the_modem_keys_stay_inside_the_same_span(name, bw):
    from hfmodem.bateleur import fsk
    from hfmodem.bateleur.modem import Mode
    Mode.parse("FSK", name)                      # the modem would key it
    lo, hi = OCCUPIED_AUDIO_HZ[bw]
    data = bytes(np.random.default_rng(1).integers(0, 256, 100, dtype=np.uint8))
    a, b = _minus_26db_span(fsk.transmit(data, name))
    assert lo <= a and b <= hi, (name, a, b)


@pytest.mark.parametrize("name", ["1800U", "600L/1100", "600L/2300"])
@pytest.mark.parametrize("bw", ("3000", "3200"))
def test_4529_modes_the_modem_keys_stay_inside_the_same_span(name, bw):
    from hfmodem.bateleur import s4529
    from hfmodem.bateleur.modem import Mode
    Mode.parse("4529", name)                     # the modem would key it
    lo, hi = OCCUPIED_AUDIO_HZ[bw]
    data = bytes(np.random.default_rng(1).integers(0, 256, 300, dtype=np.uint8))
    a, b = _minus_26db_span(s4529.transmit(data, name))
    assert lo <= a and b <= hi, (name, a, b)


def test_the_untrimmed_pulse_is_wider_and_keeps_the_same_timing():
    """Either filter setting puts each symbol at the same sample, so the
    receiver needs no telling which the far end chose."""
    idx = np.random.default_rng(4).integers(0, 8, 400)
    a, b = phy.modulate(idx), phy.modulate(idx, trim=False)
    assert a.size == b.size
    assert _minus_26db_span(b)[1] > _minus_26db_span(a)[1] + 50
    at = phy.leadin_samples()
    for x in (a, b):
        got = phy.downconvert(x)[at: at + 400 * spec.SPS: spec.SPS]
        dec = np.round(np.angle(got) / (np.pi / 4)).astype(int) % 8
        assert np.array_equal(dec[30:-30], idx[30:-30])


def test_matched_filter_recovers_the_symbols():
    idx = np.random.default_rng(3).integers(0, 8, 400)
    bb = phy.downconvert(phy.modulate(idx))
    at = phy.leadin_samples()
    got = bb[at: at + 400 * spec.SPS: spec.SPS]
    got = got / np.mean(np.abs(got))
    dec = np.round(np.angle(got) / (np.pi / 4)).astype(int) % 8
    assert np.array_equal(dec[30:-30], idx[30:-30])


def test_acquisition_finds_offset_and_position():
    idx = mil110.preamble(spec.MODES110["600S"])
    audio = np.concatenate([np.zeros(3000), phy.modulate(idx)])
    t = np.arange(audio.size) / spec.FS
    from scipy.signal import hilbert
    shifted = np.real(hilbert(audio) * np.exp(2j * np.pi * -47.0 * t))
    hit = phy.find(phy.downconvert(shifted), mil110.SYNC_REF)
    assert hit.sample == 3000 + phy.leadin_samples()
    assert hit.freq_hz == pytest.approx(-47.0, abs=0.5)
