# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M2b acceptance gates: the noncoherent floor. Full sweeps: ``python -m hfmodem.sabir.sim.m2b``."""

import numpy as np
import pytest

from hfmodem.sabir.fec import CATBCC, ConvCode
from hfmodem.sabir.floor import FLOOR_GEARS, FloorModem
from hfmodem.sabir.frame import crc16, crc_ok
from hfmodem.sabir.arq import wire
from hfmodem.sabir.phy import GEARS, LADDER
from hfmodem.sabir.phy.modem import FS, Phy
from hfmodem.sabir.sim.m2 import add_noise_snr3k
from hfmodem.sabir.sim.m2b import code_duel, floor_fer, occupied_bw, ofdm_fer

CONTROL = wire.Control(wire.DATA, 0x81, seq=1, gear=4,
                       mask=wire.cw_mask([0]), aux=wire.data_aux(154, 1, (0,) * 8))
BLOCK = CONTROL.pack()


# -- the convolutional substrate ------------------------------------------------
def test_convcode_free_distance():
    assert ConvCode().free_distance() == 15          # Larsen optimum, K=12
    assert ConvCode(7, (0o133, 0o171)).free_distance() == 10   # textbook check


def test_tailbiting_is_circular():
    cc = ConvCode()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 64)
    pairs = cc.encode_tb(bits).reshape(-1, 2)
    rolled = cc.encode_tb(np.roll(bits, 5)).reshape(-1, 2)
    assert (rolled == np.roll(pairs, 5, axis=0)).all()


def test_catbcc_roundtrip():
    tb = CATBCC()
    rng = np.random.default_rng(1)
    payload = rng.integers(0, 256, 14, dtype=np.uint8).tobytes()
    block = payload + crc16(payload).to_bytes(2, "big")
    coded = tb.encode(block)
    assert coded.size == 16 * len(block)
    got, stats = tb.decode((1.0 - 2.0 * coded) * 8.0, len(block), crc_ok)
    assert got == block and stats["ok"] and stats["rank"] == 0


def test_crc_list_rescues_plain_viterbi():
    # same noisy LLRs through the plain Viterbi (list 1, no screen) and the
    # CRC-aided list: the list must strictly dominate and rescue >= 1 block
    ca = CATBCC()
    plain = CATBCC(list_size=1, state_list=1)
    rng = np.random.default_rng(2)
    payload = rng.integers(0, 256, 14, dtype=np.uint8).tobytes()
    block = payload + crc16(payload).to_bytes(2, "big")
    coded = ca.encode(block)
    sigma = np.sqrt(1 / 10 ** (1.5 / 10))
    e_plain = e_ca = rescued = 0
    for _ in range(30):
        llr = 2 * ((1.0 - 2.0 * coded) + sigma * rng.standard_normal(256)) / sigma**2
        p, _ = plain.decode(llr, len(block))
        c, _ = ca.decode(llr, len(block), crc_ok)
        e_plain += p != block
        e_ca += c != block
        rescued += p != block and c == block
    assert e_ca <= e_plain and rescued >= 1


def test_catbcc_beats_short_ldpc():
    # the reason the floor code is convolutional: at (256,128) the CA-TBCC
    # clears the same-length QC-LDPC by ~1 dB (full curve in sim.m2b)
    bt, bl = code_duel(2.5, 80, seed=9)
    assert bt < bl and bl >= 3 / 80


# -- floor waveform -------------------------------------------------------------
def test_floor_session_control_loopback_clean():
    tx = FloorModem().transmit(BLOCK)
    got, stats = FloorModem().receive(Phy.from_audio(Phy.to_audio(tx)), wire.BLOCK_BYTES)
    assert got == BLOCK
    assert abs(stats["cfo_hz"]) < 1.0 and stats["rank"] == 0


def test_floor_generic_block_roundtrip():
    # the floor carries any CRC-terminated block, including connectionless controls
    modem = FloorModem(FLOOR_GEARS["floor"])
    rng = np.random.default_rng(3)
    payload = rng.integers(0, 256, 22, dtype=np.uint8).tobytes()
    block = payload + crc16(payload).to_bytes(2, "big")
    y = add_noise_snr3k(modem.transmit(block), -8.0, rng)
    got, _ = modem.receive(y, len(block))
    assert got == block


@pytest.mark.parametrize("cfo,drift", [(-75.0, 3.5), (75.0, -3.5), (40.0, 0.0)])
def test_floor_cfo_envelope(cfo, drift):
    tx = FloorModem().transmit(BLOCK)
    t = np.arange(tx.size) / FS
    y = tx * np.exp(2j * np.pi * (cfo * t + 0.5 * drift * t**2))
    rng = np.random.default_rng(4)
    y = add_noise_snr3k(np.concatenate(
        [np.zeros(24000), y, np.zeros(24000)]), -8.0, rng)
    got, stats = FloorModem().receive(y, wire.BLOCK_BYTES)
    assert got == BLOCK
    assert abs(stats["cfo_hz"] - cfo) < 6.0


def test_floor_awgn_sensitivity():
    # strongly negative in-band SNR, far below the OFDM robust rung
    assert floor_fer("floor", None, -13.0, 6, seed=51) <= 1 / 6
    assert floor_fer("floor2", None, -15.0, 6, seed=52) <= 2 / 6


def test_dual_waveform_gate_polar_disturbed():
    # the reason the design is dual-waveform: 30 Hz Doppler spread kills the
    # coherent OFDM rungs at the tested SNRs while the floor can decode below 0 dB
    assert ofdm_fer("workhorse", "polar_disturbed", 15.0, 4, seed=61) == 1.0
    assert ofdm_fer("robust", "polar_disturbed", 10.0, 4, seed=62) == 1.0
    assert floor_fer("floor", "polar_disturbed", 0.0, 4, seed=63) <= 1 / 4


def test_floor_occupied_bandwidth_and_papr():
    tx = FloorModem().transmit(BLOCK)
    lo, hi = occupied_bw(tx)
    assert hi - lo <= 500.0                          # ACDS-legal control tier
    body = tx[np.abs(tx) > 0]
    p = np.abs(body) ** 2
    assert 10 * np.log10(p.max() / p.mean()) < 0.5   # constant envelope


def test_floor_ignores_pure_noise():
    modem = FloorModem(FLOOR_GEARS["floor"])
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)
    got, _ = modem.receive(noise, wire.BLOCK_BYTES)
    assert got is None


# -- the dual-family gear table -------------------------------------------------
def test_ladder_spans_both_families():
    for name in LADDER:
        assert (name in GEARS) != (name in FLOOR_GEARS)
    assert LADDER[0] in FLOOR_GEARS and LADDER[-1] in GEARS
