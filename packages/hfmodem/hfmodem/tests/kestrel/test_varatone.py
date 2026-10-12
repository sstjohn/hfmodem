# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tone family (BW2300 L5-9, BW2750 L5-10, BW500 L5-6): TX -> RX, classification,
and the known-answer cells the working/ KPA models produce for seeded random frames
(fixtures/tone-kat/cells.json, from working/vara-registered-0926/tone_kat.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.kestrel.rx import varatone as rx
from hfmodem.kestrel.rx import tablegen
from hfmodem.kestrel.tx import varatone_tx as tx
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd

KAT_PATH = Path(__file__).parent / "fixtures" / "tone-kat" / "cells.json"
KAT = (json.loads(KAT_PATH.read_text()) if KAT_PATH.is_file() else
       [pytest.param(None, id="missing-tone-kat", marks=pytest.mark.skip(
           reason="fixtures/tone-kat/cells.json is not included in this distribution"))])
ROWS = sorted(rx.ROWS)


def _burst(key, seed, control=0x99, payload=None, caller="N0CALL"):
    rec = rx.ROWS[key]
    rng = np.random.default_rng(seed)
    pl = payload if payload is not None else rng.integers(0, 256, rec.payload_bytes).astype(np.uint8).tobytes()
    a = tx.synth_burst(pl, *key, control, VB6Rnd(int(rng.integers(0, 1 << 24))),
                       caller=caller)
    return pl, np.concatenate([np.zeros(int(rng.integers(0, 4000))), a, np.zeros(2000)])


@pytest.mark.parametrize("key", ROWS, ids=lambda k: f"bw{k[0]}-L{k[1]}")
def test_row_roundtrip_byte_exact(key):
    control = 0x19 + key[1]
    pl, audio = _burst(key, seed=key[0] + key[1], control=control)
    r = rx.decode_burst(audio, *key)
    assert r.ok and r.level == key[1]
    assert r.payload == pl and r.control == control
    assert r.quality.raw_errors == 0 and r.quality.snr_db > 20


@pytest.mark.parametrize("key", ROWS, ids=lambda k: f"bw{k[0]}-L{k[1]}")
def test_classification_lands_on_the_row(key):
    pl, audio = _burst(key, seed=7 * key[1] + key[0])
    r = rx.decode_burst(audio, key[0])
    assert r.level == key[1] and r.ok and r.payload == pl


@pytest.mark.parametrize("caller,crc", [("N0CALL", 0x09), ("W9SSJ", 0xF6)])
@pytest.mark.parametrize("key", [(2300, 7), (2750, 10), (500, 5)], ids=lambda k: f"bw{k[0]}-L{k[1]}")
def test_short_frame_carries_the_callers_fill(key, caller, crc):
    rec = rx.ROWS[key]
    data = bytes(range(40))
    _, audio = _burst(key, seed=3, control=0x82, payload=data, caller=caller)
    r = rx.decode_burst(audio, *key)
    end = 0x07 if key[0] == 500 else 0x06
    assert r.control == 0x82
    assert r.payload == data + bytes([0x14, crc]) + bytes(rec.payload_bytes - 43) + bytes([end])


def _shifted(a, df):
    return np.real(hilbert(a) * np.exp(2j * np.pi * df * np.arange(len(a)) / rx.FS))


OFFSET_ROWS = [(2300, 5), (2300, 9), (2750, 5), (2750, 10), (500, 5), (500, 6)]


@pytest.mark.parametrize("key", OFFSET_ROWS, ids=lambda k: f"bw{k[0]}-L{k[1]}")
@pytest.mark.parametrize("df", [-20, -12, -4, 3, 9, 20])
def test_carrier_offset_reads_byte_exact(key, df):
    pl, audio = _burst(key, seed=key[1])
    r = rx.decode_burst(_shifted(audio, df), *key)
    assert r.ok and r.payload == pl and r.quality.raw_errors == 0
    assert abs(r.quality.offset_hz - df) < 0.5


@pytest.mark.parametrize("df", [-17, 12])
def test_carrier_offset_leaves_classification_alone(df):
    pl, audio = _burst((2750, 8), seed=21)
    r = rx.decode_burst(_shifted(audio, df), 2750)
    assert r.level == 8 and r.payload == pl


# Per row, noise (in the burst's own power) 1 dB below the most the on-grid read
# carries on these seeds.
ONGRID_EDGE_DB = {(2300, 9): -8.5, (2750, 10): -7.0, (500, 6): -14.0}


@pytest.mark.parametrize("key", sorted(ONGRID_EDGE_DB), ids=lambda k: f"bw{k[0]}-L{k[1]}")
@pytest.mark.parametrize("df", [-12, 12])
def test_carrier_offset_costs_under_a_decibel(key, df):
    pl, audio = _burst(key, seed=key[1])
    rng = np.random.default_rng(7)
    p = np.mean(audio[audio != 0] ** 2)
    noise = rng.normal(0, np.sqrt(p / 10 ** ((ONGRID_EDGE_DB[key] + 1) / 10)), len(audio))
    for f in (0, df):
        r = rx.decode_burst(_shifted(audio, f) + noise, *key)
        assert r.payload == pl and r.quality.raw_errors > 0


def test_noise_fails_crc():
    rng = np.random.default_rng(5)
    r = rx.decode_burst(100 * rng.standard_normal(200000), 2300, 5)
    assert not r.ok and r.lock < 0.3


def test_decodes_through_noise():
    pl, audio = _burst((2750, 10), seed=11)
    rng = np.random.default_rng(12)
    r = rx.decode_burst(audio + 1.3 * np.std(audio) * rng.standard_normal(len(audio)), 2750, 10)
    assert r.ok and r.payload == pl and r.quality.raw_errors > 0


def test_symbol_counts_follow_the_records():
    assert {k: r.nsym for k, r in rx.ROWS.items()} == {
        (2300, 5): 383, (2300, 6): 372, (2300, 7): 362, (2300, 8): 359, (2300, 9): 356,
        (2750, 5): 383, (2750, 6): 372, (2750, 7): 362, (2750, 8): 359, (2750, 9): 356, (2750, 10): 356,
        (500, 5): 402, (500, 6): 383}
    assert rx.ROWS[2750, 10].n_info == 4466 and rx.ROWS[2750, 10].frame_bytes == 558


@pytest.mark.parametrize("kat", KAT, ids=lambda k: f"bw{k['bw']}-L{k['level']}")
def test_row_matches_the_kpa_model(kat):
    rec = rx.ROWS[kat["bw"], kat["level"]]
    assert rec.nsym == kat["nsym"] and rec.pulse == kat["pulse"]
    assert list(rec.stagger) == kat["stagger"]
    assert np.allclose(rec.centres, kat["centres"], atol=1e-9)
    frame = bytes.fromhex(kat["frame"])
    assert tx.build_frame(frame[:-3], kat["control"], rec, caller="N0CALL") == frame
    want = np.array([list(bytes.fromhex(b)) for b in kat["cells"]])
    got = np.round(np.angle(tx.body(frame, rec, np.zeros(rec.nb, int))) * 64 / np.pi).astype(int) % 128
    measured = rx.grid(rec.nb)[:, :rec.nsym - rx.C0] >= 0
    assert (want[~measured] != 255).sum() == 0
    # The model keyed level 7's last three columns on rows 4 and 5 off BW2300
    # L7's noisy read; the rows now carry BW2750 L7's clean one.
    tail = np.zeros_like(measured)
    if rec.level == 7:
        tail[4:6, -3:] = True
    assert (want[measured & ~tail] == got[measured & ~tail]).all()


def test_lead_is_nine_time_major_rows_of_the_session_stream():
    rec = rx.ROWS[2750, 10]
    rnd = VB6Rnd(4107106)
    c = tx.cells(bytes(rec.frame_bytes), rec, rnd)
    s, v = 4107106, []
    for _ in range(9 * rec.nb):
        s = tablegen.step(s)
        v.append(tablegen.draw(s, 4))
    assert rnd.state == s
    q = rx.GRAY[np.array(v).reshape(9, rec.nb).T]
    g = rx.grid(rec.nb)[:, :rx.C0]
    assert np.allclose(c[:, :rx.C0], 1j ** q[:, 1:] * np.exp(-2j * np.pi * g / 128))
    assert np.allclose(c[:, rx.C0], 1j ** q[:, 0] * np.exp(-2j * np.pi * g[:, 0] / 128))
