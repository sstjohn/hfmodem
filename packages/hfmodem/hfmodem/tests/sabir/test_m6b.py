# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M6b acceptance gates: the narrow-tone beacon. Full sweeps: ``python -m hfmodem.sabir.sim.m6b``."""

import numpy as np
import pytest

from hfmodem.sabir.floor import (BEACON_BYTES, BEACON_GEARS, FLOOR_GEARS, BeaconModem,
                                 BeaconPayload, receive_combining, recv_beacon, send_beacon)
from hfmodem.sabir.phy import LADDER
from hfmodem.sabir.phy.modem import FS, Phy
from hfmodem.sabir.sim.m2b import occupied_bw
from hfmodem.sabir.sim.m6b import PAYLOAD, add_noise_snr2k5, beacon_trial


# -- payload -------------------------------------------------------------------
@pytest.mark.parametrize("pl", [
    BeaconPayload("W1AW", "FN31", 173),
    BeaconPayload("K1ABC/6", "AA00", 0),
    BeaconPayload("A", "RR99", 255),
])
def test_payload_roundtrip(pl):
    if len(pl.callsign) > 6:
        with pytest.raises(ValueError):
            pl.pack()
        return
    buf = pl.pack()
    assert len(buf) == BEACON_BYTES
    assert BeaconPayload.unpack(buf) == pl


def test_payload_rejects():
    with pytest.raises(ValueError):
        BeaconPayload("W1AW", "XX99").pack()       # S-Z: not a grid field
    with pytest.raises(ValueError):
        BeaconPayload("TOOLONG1", "FN31").pack()
    buf = bytearray(PAYLOAD.pack())
    buf[3] ^= 0x40
    assert BeaconPayload.unpack(bytes(buf)) is None


# -- the mode registry ---------------------------------------------------------
def test_beacon_is_its_own_mode():
    # the beacon rungs live outside both the ARQ ladder and the interactive
    # floor: non-interactive by nature, never a gearshift target
    for name in BEACON_GEARS:
        assert name not in LADDER and name not in FLOOR_GEARS
    durs = [BeaconModem(g).duration_s(BEACON_BYTES)
            for g in BEACON_GEARS.values()]
    assert durs == sorted(durs)                    # deeper rung = longer burst


# -- waveform ------------------------------------------------------------------
def test_beacon_loopback_clean_audio():
    gear = BEACON_GEARS["beacon_short"]
    tx = send_beacon(PAYLOAD, gear)
    got, stats = recv_beacon(Phy.from_audio(Phy.to_audio(tx)), gear)
    assert got == PAYLOAD
    assert abs(stats["cfo_hz"]) < 0.5 and stats["rank"] == 0


def test_beacon_constant_envelope_and_bandwidth():
    tx = send_beacon(PAYLOAD, BEACON_GEARS["beacon_med"])
    body = tx[np.abs(tx) > 0]
    p = np.abs(body) ** 2
    assert 10 * np.log10(p.max() / p.mean()) < 0.5
    lo, hi = occupied_bw(tx)
    assert hi - lo < 50.0                          # ~20 Hz of spectrum


@pytest.mark.parametrize("cfo,drift", [(75.0, 0.0), (-60.0, 0.05)])
def test_beacon_cfo_envelope(cfo, drift):
    gear = BEACON_GEARS["beacon_short"]
    tx = send_beacon(PAYLOAD, gear)
    t = np.arange(tx.size) / FS
    y = tx * np.exp(2j * np.pi * (cfo * t + 0.5 * drift * t**2))
    rng = np.random.default_rng(4)
    y = add_noise_snr2k5(np.concatenate(
        [np.zeros(24000), y, np.zeros(24000)]), -18.0, rng)
    got, stats = recv_beacon(y, gear)
    assert got == PAYLOAD
    assert abs(stats["cfo_hz"] - cfo) < 1.0


# -- sensitivity ---------------------------------------------------------------
def test_beacon_short_negative_snr():
    gear = BEACON_GEARS["beacon_short"]
    modem = BeaconModem(gear)
    tx = send_beacon(PAYLOAD, gear)
    rng = np.random.default_rng(6)
    assert sum(beacon_trial(modem, tx, None, -20.0, rng)
               for _ in range(4)) >= 3


def test_beacon_deep_field_exact():
    # a single burst, field-exact, at -26 dB in 2.5 kHz -- ~11 dB below the
    # interactive floor; the sweep in sim.m6b puts the FER-0.5 crossing lower
    gear = BEACON_GEARS["beacon_deep"]
    tx = send_beacon(PAYLOAD, gear)
    y = add_noise_snr2k5(np.concatenate(
        [np.zeros(48000), tx, np.zeros(48000)]), -26.0,
        np.random.default_rng(1))
    got, stats = recv_beacon(y, gear)
    assert got == PAYLOAD and stats["ok"]


def test_beacon_deep2_field_exact():
    # the deepest rung: in-burst repeat x2, energy-combined, -28 dB
    gear = BEACON_GEARS["beacon_deep2"]
    tx = send_beacon(PAYLOAD, gear)
    y = add_noise_snr2k5(np.concatenate(
        [np.zeros(48000), tx, np.zeros(48000)]), -28.0,
        np.random.default_rng(2))
    got, stats = recv_beacon(y, gear)
    assert got == PAYLOAD and stats["ok"]


# -- broadcast combining -------------------------------------------------------
def test_broadcast_combining_rescues():
    # no-ARQ one-to-many: each burst alone fails at -23.5 dB, the LLR
    # combiner across repeats decodes -- and stops as soon as it has enough
    gear = BEACON_GEARS["beacon_short"]
    modem = BeaconModem(gear)
    tx = send_beacon(PAYLOAD, gear)
    rng = np.random.default_rng(2)
    caps = [add_noise_snr2k5(np.concatenate(
        [np.zeros(24000), tx, np.zeros(24000)]), -23.5, rng)
        for _ in range(3)]
    for cap in caps:
        block, _ = modem.receive(cap)
        assert block is None or BeaconPayload.unpack(block) != PAYLOAD
    block, stats = receive_combining(caps, gear=gear)
    assert block is not None and BeaconPayload.unpack(block) == PAYLOAD
    assert stats["combined"] <= 3


def test_beacon_ignores_pure_noise():
    modem = BeaconModem(BEACON_GEARS["beacon_short"])
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(1_000_000) + 1j * rng.standard_normal(1_000_000)
    block, stats = modem.receive(noise)
    assert block is None and stats["sync"] is None
