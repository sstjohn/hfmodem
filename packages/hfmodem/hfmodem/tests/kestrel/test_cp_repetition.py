# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A const-law over is named by its cyclic prefix, and nothing without one is.

The positive is kestrel's own BW2300 L10 (record 9) transmitter under noise; the negatives are real VARA overs at the base level,
which carry no prefix, and band noise."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.rx import varaofdm
from hfmodem.kestrel.tx import varaofdm_tx
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_connected_ack_stream import _IO, _awaiting_ack

FS = rx.FS


def _rec9_over(snr_db: float, seed: int = 9) -> np.ndarray:
    """A rec9 over with white noise at ``snr_db`` in 2500 Hz, half a second either side."""
    rng = np.random.default_rng(seed)
    r = varaofdm.ROWS[2300, 10]
    s = varaofdm_tx.synth_burst(rng.integers(0, 256, r.frame_bytes - 3, dtype=np.uint8).tobytes(),
                                2300, 10, 0x99, VB6Rnd(seed), caller="N0CALL")
    s = np.concatenate([np.zeros(FS // 2), s, np.zeros(FS // 2)])
    sigma = np.sqrt(np.mean(s ** 2) / 10 ** (snr_db / 10) * (FS / 2) / 2500)
    return s + rng.normal(0.0, sigma, len(s))


def _cp_ofdm(useful: int, snr_db: float, seed: int = 50) -> np.ndarray:
    """5 s of QPSK CP-OFDM on a ``useful``-sample symbol filling 300-2900 Hz: the
    shape BW2750's high-SL over is believed to have, at a length no record fixes."""
    rng = np.random.default_rng(seed)
    bins = np.arange(round(300 * useful / FS), round(2900 * useful / FS))
    syms = []
    for _ in range(5 * FS // (useful + rx.CP_SAMPLES)):
        sp = np.zeros(useful, complex)
        sp[bins] = 1j ** rng.integers(0, 4, len(bins))
        t = np.fft.ifft(sp).real
        syms += [t[-rx.CP_SAMPLES:], t]
    s = np.concatenate(syms)
    sigma = np.sqrt(np.mean(s ** 2) / 10 ** (snr_db / 10) * (FS / 2) / 2500)
    return s + rng.normal(0.0, sigma, len(s))


def test_a_rec9_over_at_moderate_snr_repeats_its_prefix_at_1024():
    useful, metric = rx.cp_repetition(_rec9_over(3.0))
    assert useful == 1024, (useful, metric)
    assert metric >= 1.5 * rx.CP_FLOOR, metric


def test_a_prefix_is_found_at_a_useful_length_off_every_grid():
    useful, metric = rx.cp_repetition(_cp_ofdm(1283, 3.0))
    assert useful == 1283, (useful, metric)
    assert metric >= 1.5 * rx.CP_FLOOR, metric


def test_band_noise_carries_no_prefix():
    rng = np.random.default_rng(1)
    for _ in range(4):
        useful, metric = rx.cp_repetition(rng.normal(0.0, 1.0, 5 * FS))
        assert metric < rx.CP_FLOOR / 2, (useful, metric)


@corpora.requires_bw2300_capture
def test_real_base_level_overs_carry_no_prefix():
    a = np.asarray(corpora.wav_mono(corpora.BW2300_CAPTURE), float)
    overs = rx.detect_overs(a)
    assert len(overs) == 7
    for s, e in overs:
        useful, metric = rx.cp_repetition(a[s:e])
        assert metric < rx.CP_FLOOR / 2, (s / FS, useful, metric)


@pytest.mark.parametrize("deep", [False, True])
def test_the_monitor_names_an_undecoded_rec9_over_by_its_prefix(deep):
    monitor = corpora.harness("vara_monitor")
    r = monitor.classify(_rec9_over(3.0), [], decode_wideband=deep, level_db=8.0)
    assert r.kind == "high-SL over", r
    assert "1024+128-sample" in r.info and "record unknown" in r.info, r


def test_the_session_log_reads_the_prefix_off_an_over_it_cannot_lock():
    io = _IO()
    hs = _awaiting_ack(io)
    assert hs._peer_data_over(_rec9_over(3.0)) == []
    assert "not a wideband over, not answering, CP-OFDM 1024+128 at" in io.log_lines[-1], io.log_lines
