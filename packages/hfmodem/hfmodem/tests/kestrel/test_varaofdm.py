# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The OFDM family (BW2300 L10-16, BW2750 L11-17, BW500 L7-13): the row table, TX -> RX,
classification, double overs, the symbol shape, and the known-answer cells the working/ KPA
models produce for seeded random frames (fixtures/ofdm-kat/cells.json, from
working/vara-registered-0926/ofdm_kat.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.rx import varaofdm as rx
from hfmodem.kestrel.tx import varaofdm_tx as tx
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd

_KAT = Path(__file__).parent / "fixtures" / "ofdm-kat" / "cells.json"
KAT = json.loads(_KAT.read_text()) if _KAT.exists() else {"rows": {}}
ROWS = sorted(rx.ROWS)
_ids = {"ids": lambda k: f"bw{k[0]}-L{k[1]}"}

# (frame bytes, coded bits) per row, as the KPA reports measured them
MEASURED = {
    2300: {10: (899, 14406), 11: (1199, 14406), 12: (1439, 14406), 13: (1799, 21609),
           14: (2159, 21609), 15: (2713, 26068), 16: (3257, 32585)},
    2750: {11: (1083, 17346), 12: (1444, 17346), 13: (1733, 17346), 14: (2167, 26019),
           15: (2600, 26019), 16: (3268, 31388), 17: (3922, 39235)},
    500: {7: (197, 3168), 8: (263, 3168), 9: (315, 3168), 10: (395, 4752), 11: (474, 4752),
          12: (594, 5720), 13: (713, 7150)},
}


def _frames(key, seed, n=1, control=None):
    r = rx.ROWS[key]
    rng = np.random.default_rng(seed)
    return [tx.build_frame(rng.integers(0, 256, r.frame_bytes - 3, dtype=np.uint8).tobytes(),
                           int(rng.integers(0, 256)) if control is None else control, *key,
                           caller="N0CALL")
            for _ in range(n)]


def _air(a, seed, offset=0):
    """``a`` in a buffer whose burst starts ``offset`` samples from where a caller
    expects it (negative: the head is cut), with a little noise."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(max(offset, 0)), a[max(-offset, 0):], np.zeros(3000)])
    return x + rng.normal(0, 0.01, len(x))


def test_rows_are_the_measured_coding_and_fill_every_cell():
    assert {(bw, lv) for bw, rows in MEASURED.items() for lv in rows} == set(rx.ROWS)
    for (bw, lv), r in rx.ROWS.items():
        assert (r.frame_bytes, r.coded) == MEASURED[bw][lv]
        kind, fill = r.layout
        assert (kind == rx.DATA).sum() * r.bpc == r.coded
        assert sorted(fill[kind == rx.DATA]) == list(range(r.coded // r.bpc))
        assert r.gap_cells.shape == (r.carriers, 4 if bw == 500 else 1)


@pytest.mark.parametrize("key", ROWS, **_ids)
@pytest.mark.parametrize("offset", [-64, 64])
def test_row_roundtrip_byte_exact(key, offset):
    (frame,) = _frames(key, seed=key[0] + key[1])
    res = rx.decode_burst(_air(tx.render(frame, *key, VB6Rnd(key[1])), key[1], offset), key[0])
    assert res.ok and res.level == key[1] and res.frames == (frame,)
    assert res.payload == frame[:-3] and res.control == frame[-3]
    assert res.quality.raw_errors == 0 and res.quality.snr_db > 20


def _shifted(a, df):
    return np.real(hilbert(a) * np.exp(2j * np.pi * df * np.arange(len(a)) / rx.FS))


OFFSET_ROWS = [(2300, 10), (2300, 13), (2300, 16), (2750, 11), (2750, 14), (2750, 17),
               (500, 7), (500, 10), (500, 13)]


@pytest.mark.parametrize("key", OFFSET_ROWS, **_ids)
@pytest.mark.parametrize("df", [-20, -12, -4, 3, 9, 20])
def test_carrier_offset_reads_byte_exact(key, df):
    (frame,) = _frames(key, seed=key[1])
    a = _shifted(_air(tx.render(frame, *key, VB6Rnd(3)), key[1], 500), df)
    res = rx.decode_burst(a, key[0])
    assert res.level == key[1] and res.frames == (frame,)
    assert res.quality.raw_errors == 0


@pytest.mark.parametrize("key", [(2300, 12), (2300, 15), (2750, 17), (500, 9), (500, 13)], **_ids)
@pytest.mark.parametrize("df", [-20, -7.3, 0, 11.6, 20])
def test_carrier_offset_estimate_within_half_a_hertz(key, df):
    frames = _frames(key, seed=5, n=2 if rx.ROWS[key].qam else 1)
    res = rx.decode_burst(_shifted(tx.render(frames, *key, VB6Rnd(4)), df), key[0])
    assert res.ok and abs(res.quality.offset_hz - df) < 0.5


# Per row, noise (in the burst's own power) 1 dB below the most the on-grid read
# carries on these seeds.
ONGRID_EDGE_DB = {(2300, 13): -0.5, (2750, 17): 6.5, (500, 13): -1.0}


@pytest.mark.parametrize("key", sorted(ONGRID_EDGE_DB), **_ids)
@pytest.mark.parametrize("df", [-12, 12])
def test_carrier_offset_costs_under_a_decibel(key, df):
    (frame,) = _frames(key, seed=key[1])
    a = np.concatenate([np.zeros(1500), tx.render(frame, *key, VB6Rnd(1)), np.zeros(3000)])
    rng = np.random.default_rng(7)
    noise = rng.normal(0, np.sqrt(np.mean(a[a != 0] ** 2) / 10 ** ((ONGRID_EDGE_DB[key] + 1) / 10)), len(a))
    for f in (0, df):
        res = rx.decode_burst(_shifted(a, f) + noise, key[0])
        assert res.frames == (frame,) and res.quality.raw_errors > 0


@pytest.mark.parametrize("key", ROWS, **_ids)
def test_classification_lands_on_the_row_whatever_the_expected_level(key):
    """The expected level is only where the walk starts: a burst at another row of
    the class is still read at its own."""
    bw, level = key
    wrong = next(lv for (b, lv), r in rx.ROWS.items()
                 if b == bw and lv != level and r.qam == rx.ROWS[key].qam)
    (frame,) = _frames(key, seed=3 * level)
    res = rx.decode_burst(_air(tx.render(frame, *key, VB6Rnd(1)), level), bw, wrong)
    assert res.level == level and res.frames == (frame,)


@pytest.mark.parametrize("key", [(2300, 16), (2750, 17), (500, 13), (2300, 15)], **_ids)
def test_double_over_yields_two_frames(key):
    frames = _frames(key, seed=key[1], n=2)
    res = rx.decode_burst(_air(tx.render(frames, *key, VB6Rnd(5)), 2), key[0])
    assert res.level == key[1] and res.frames == tuple(frames)
    assert res.payload == frames[0][:-3] + frames[1][:-3]


@pytest.mark.parametrize("bw", [2300, 2750, 500])
def test_noise_fails_cleanly(bw):
    rng = np.random.default_rng(bw)
    res = rx.decode_burst(rng.normal(0, 0.25, 360000), bw)
    assert not res.ok and res.payload == b"" and res.control is None


@pytest.mark.parametrize("key", [(2300, 10), (500, 12)], **_ids)
def test_other_bandwidth_does_not_read(key):
    (frame,) = _frames(key, seed=9)
    a = _air(tx.render(frame, *key, VB6Rnd(1)), 9)
    for bw in {2300, 2750, 500} - {key[0]}:
        assert not rx.decode_burst(a, bw).ok


@pytest.mark.parametrize("name", sorted(KAT["rows"]))
def test_cells_match_the_kpa_model(name):
    """Every body cell, lead excluded, as the KPA model transmits it: pins the
    scramble and its sign, the fill order, the gap cells, the pilots, the LUT and
    the puncture, none of which a round trip can see."""
    bw, level = map(int, name.split("/"))
    row = KAT["rows"][name]
    r = rx.ROWS[bw, level]
    rng = np.random.default_rng(row["seed"])
    frames = []
    for _ in range(row["frames"]):
        body = rng.integers(0, 256, r.frame_bytes - 3, dtype=np.uint8).tobytes() + bytes([KAT["control"]])
        frames.append(body + crc16_genibus(body).to_bytes(2, "big"))
    got = tx.cells(frames, bw, level, VB6Rnd(1))[r.lead:]
    k = np.array(row["cells"]).reshape(row["shape"])
    phase = np.exp(2j * np.pi * (k % 128) / 128)
    want = phase if not r.qam else np.append(r.lut, 1.0)[k // 128] * phase
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-9


@pytest.mark.parametrize("key", [(2300, 16), (2750, 11), (500, 8)], **_ids)
def test_every_window_reads_its_cells_phase_exact(key):
    """Symbol s's FFT window opens 128 samples into it and reads its cells exactly:
    the prefix, the origin and the blend (confined to the prefix's first and the
    suffix's 32 samples) are the tape's."""
    (frame,) = _frames(key, seed=1)
    r = rx.ROWS[key]
    want = tx.cells([frame], *key, VB6Rnd(2))
    a = tx.render([frame], *key, VB6Rnd(2))
    assert len(a) == len(want) * rx.SYM
    at = rx.CP + rx.SYM * np.arange(len(want) - 1)
    got = np.fft.fft(a[at[:, None] + np.arange(rx.TU)], axis=1)[:, r.bins]
    ratio = got / want[:-1]
    assert np.allclose(ratio, ratio[0, 0], rtol=1e-9)
    ramp = np.sin(np.pi * (np.arange(32) + 0.5) / 64) ** 2
    u = np.fft.ifft(np.pad(want[0], (r.first_bin, rx.TU - r.first_bin - r.carriers))).real
    body = a[rx.CP:rx.CP + rx.TU]
    scale = body @ u / (u @ u)
    assert np.allclose(a[:32], scale * ramp * u[-rx.CP:][:32])


def test_lead_draws_one_block_per_frame_and_keys_the_last():
    for key, n in (((2300, 10), 1), ((2300, 16), 2), ((500, 13), 2), ((2750, 17), 1)):
        r = rx.ROWS[key]
        frames = _frames(key, seed=4, n=n)
        rnd, ref = VB6Rnd(777), VB6Rnd(777)
        lead = tx.cells(frames, *key, rnd)[:r.lead]
        draws = [ref.draw(r.bpc) for _ in range(n * r.lead * r.carriers)]
        assert rnd.state == ref.state
        block = np.array(draws[-r.lead * r.carriers:]).reshape(r.carriers, r.lead).T
        assert np.array_equal(lead, r.lut[block])


@pytest.mark.parametrize("caller,key", [("N0CALL", 0x09), ("W9SSJ", 0xF6)])
@pytest.mark.parametrize("bw,level,end", [(2300, 10, 0x06), (2750, 12, 0x06), (500, 8, 0x07)])
def test_short_frame_carries_the_tapes_fill(bw, level, end, caller, key):
    r = rx.ROWS[bw, level]
    cap = r.frame_bytes - 3
    data = bytes(range(40))
    frame = tx.build_frame(data, 0x82, bw, level, caller=caller)
    assert frame[:-2] == data + bytes([0x14, key]) + bytes(cap - 43) + bytes([end, 0x82])
    two = tx.build_frame(bytes(cap - 2), 0x82, bw, level, caller=caller)
    assert two[-5:-2] == bytes([0x14, end, 0x82])
    full = tx.build_frame(bytes(cap), 0x99, bw, level, caller=caller)
    assert full[-3] == 0x99 and crc16_genibus(full[:-2]).to_bytes(2, "big") == full[-2:]
    with pytest.raises(ValueError):
        tx.build_frame(bytes(cap + 1), 0x99, bw, level, caller=caller)
