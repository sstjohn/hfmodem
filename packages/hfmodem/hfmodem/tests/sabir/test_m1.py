# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M1 acceptance gates. Full sweeps with larger volumes: ``python -m hfmodem.sabir.sim.m1``."""

import numpy as np
import pytest

from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.fec.ldpc import Z
from hfmodem.sabir.frame import FrameCodec, crc16, pn9
from hfmodem.sabir.sim.m1 import (cfo_sweep, coded_fer, loopback, papr_bandwidth,
                            uncoded_sweep)


@pytest.fixture(scope="module")
def code():
    return QCLDPC()


# -- component sanity ---------------------------------------------------------
def test_crc16_check_value():
    assert crc16(b"123456789") == 0x29B1        # CRC-16/CCITT-FALSE


def test_pn9_sequence():
    seq = pn9(511 + 9)
    assert seq[:9].tolist() == [1] * 9          # seed 0x1FF shifts out ones
    assert seq[511:].tolist() == seq[:9].tolist()   # maximal period 511


def test_ldpc_encode_valid(code):
    rng = np.random.default_rng(0)
    for _ in range(4):
        info = rng.integers(0, 2, code.k)
        cw = code.encode(info)
        assert (cw[: code.k] == info).all()     # systematic
        assert code.syndrome_ok(cw)


def test_ldpc_girth_at_least_six(code):
    cols: dict[int, dict[int, int]] = {}
    for r, c, s in code.entries:
        cols.setdefault(c, {})[r] = s
    for c1 in cols:
        for c2 in cols:
            if c2 <= c1:
                continue
            shared = sorted(set(cols[c1]) & set(cols[c2]))
            for i in range(len(shared)):
                for j in range(i + 1, len(shared)):
                    r1, r2 = shared[i], shared[j]
                    delta = (cols[c1][r1] - cols[c2][r1]
                             + cols[c2][r2] - cols[c1][r2]) % Z
                    assert delta != 0           # a zero sum would be a 4-cycle


def test_ldpc_decodes_noiseless(code):
    rng = np.random.default_rng(1)
    info = rng.integers(0, 2, code.k)
    cw = code.encode(info)
    hard, ok, iters = code.decode((1.0 - 2.0 * cw) * 8.0)
    assert ok.all() and iters[0] == 1
    assert (hard[0] == cw).all()


def test_frame_roundtrip_pure():
    fc = FrameCodec()
    rng = np.random.default_rng(2)
    for n in (0, 1, 61, 62, 200):
        payload = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        bits = fc.encode(payload)
        got, stats = fc.decode((1.0 - 2.0 * bits) * 8.0)
        assert got == payload
        assert all(stats["crc_ok"])


# -- gate (a): byte-exact loopback through real acquisition ------------------
def test_gate_a_byte_exact():
    rng = np.random.default_rng(3)
    for n, via_audio, ebn0 in ((200, True, None), (61, True, None),
                               (1, True, None), (300, False, 10.0)):
        payload = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        got, res, _ = loopback(payload, ebn0, via_audio=via_audio, seed=n)
        assert got == payload, f"{n} bytes via {'audio' if via_audio else 'AWGN'}"
        assert abs(res.cfo_hz) < 2.0


# -- gate (b): uncoded BER tracks the Q-function -----------------------------
def test_gate_b_uncoded_ber_vs_qfunction():
    rows = uncoded_sweep([2.0, 4.0, 6.0], n_bits=60_000, seed=4)
    for ebn0, ber, theory, offset_db in rows:
        assert ber > 0, f"need measurable BER at {ebn0} dB"
        assert abs(offset_db) < 1.0, (
            f"Eb/N0 {ebn0}: BER {ber:.3e} vs theory {theory:.3e} "
            f"-> offset {offset_db:+.2f} dB")


# -- gate (c): coded waterfall, clipping ON ----------------------------------
def test_gate_c_coded_waterfall():
    fer_lo, _ = coded_fer(2.0, 50, seed=5)
    fer_mid, _ = coded_fer(3.0, 50, seed=6)
    fer_hi, _ = coded_fer(4.5, 50, seed=7)
    assert fer_lo > 0.4                 # below threshold: mostly erased
    assert fer_mid < fer_lo             # falling edge
    assert fer_hi <= 0.02               # above threshold: (near-)error-free


# -- gate (d): acquisition across +/-75 Hz and +/-3.5 Hz/s -------------------
def test_gate_d_cfo_drift_sweep():
    rows = cfo_sweep(ebn0_db=5.0, trials=2, seed=8, payload_len=180)
    for cfo, drift, ok, n, err_max in rows:
        assert ok == n, f"decode failed at CFO {cfo:+.0f} Hz, drift {drift:+.1f} Hz/s"
        assert err_max < 3.0, f"CFO error {err_max:.2f} Hz at {cfo:+.0f}/{drift:+.1f}"


# -- PAPR clip + spectral containment ----------------------------------------
def test_papr_and_occupied_bandwidth():
    pb = papr_bandwidth()
    assert pb["papr_off"] > 8.0                 # raw OFDM is peaky
    assert pb["papr_on"] < 6.5                  # clip+filter tames it
    assert pb["papr_off"] - pb["papr_on"] > 3.0
    assert pb["occupied_99_hz"] < 1700.0        # stays a ~1.4 kHz-tier signal
    assert pb["band_lo_hz"] > 500.0
    assert pb["band_hi_hz"] < 2500.0            # far inside the 2.8 kHz cap
