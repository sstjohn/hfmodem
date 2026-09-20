# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW500 records 1 and 0 — host ``BITRATE (2)`` and ``(1)`` — decoded off stock audio.

A stock VARA HF 4.9.0 responder whose greeting the caller could not read re-sends
it from the lowest level: 9 bytes at level 1 (18 bps, 5.39 s), then 22 at level 2
(41 bps, 4.96 s), then climbs. Two such recoveries were recorded on the bench on
2026-09-11 (``working/vara-evening-en63bc-0910/analysis/stock500-recovery``), both
arms symbol-identical, the greeting delivered 225/225 byte-exact — so the four
records here carry known payloads: the greeting's bytes 0..8 and 9..30.

Level 2 is record 1: record 2's comb on two more columns, turbo rate 1/3, its
tables the draws of its own interleaver gap. Level 1 is record 0: 2048-sample
columns over 21 bins at stride 2, 124 columns, rate 1/3, its 124 base bins read
off the tape. Both carry marker 0x99 in these recordings.

A burst can hold more than one frame. The stock responder's seventh level-1
over of the chain arm (``analysis/stock500-chain``, 2026-09-11) keyed two in
one PTT: 1002 columns of 512, the second frame opening at column 2 + 124 on the
first frame's grid, no training of its own, no gap, the same tables — 24 of 24
reference columns at c0 = 2 and c0 = 126 and no more than 6 anywhere else on
either copy of the tape. Greeting bytes 54..71, byte-exact.
"""
import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.rx import tablegen
from hfmodem.kestrel.rx import varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx

FIXTURES = Path(__file__).with_name("fixtures") / "bw500-low-levels"
RECORDING_NAMES = ["bw500-level1-arm1", "bw500-level1-arm2", "bw500-level1-two-frames",
                   "bw500-level2-arm1", "bw500-level2-arm2"]
_MARKER = 0x5A


@cache
def recordings():
    path = FIXTURES / "provenance.json"
    if not path.exists():
        pytest.skip(f'stock BW500 low-level metadata absent: {path}')
    rows = json.loads(path.read_text())
    assert set(rows) == set(RECORDING_NAMES)
    return rows


def _audio(name):
    path = FIXTURES / f"{name}.wav"
    if not path.exists():
        pytest.skip(f'stock BW500 low-level recording absent: {path}')
    rate, x = wavfile.read(path)
    assert rate == 48000 and x.dtype == np.int16
    return x.astype(float) / 32768


def _frame_columns(meta):
    return meta.get("frame_columns", [meta["lead"]])


@pytest.mark.parametrize("name", RECORDING_NAMES)
def test_a_recorded_low_level_resend_decodes_byte_exact(name):
    meta = recordings()[name]
    res = rx.decode_stream(_audio(name))
    assert res.complete and len(res.data_frames) == len(_frame_columns(meta))
    for f in res.data_frames:
        assert f.level == meta["host_bitrate"] and f.crc_ok
        assert f.marker == 0x99 and f.self_consistency == 1.0
        assert f.frames_in_burst == len(_frame_columns(meta))
    assert [f.c0 for f in res.data_frames] == _frame_columns(meta)
    assert b"".join(f.payload for f in res.data_frames) == bytes.fromhex(
        meta["expected_payload_hex"])


def test_a_two_frame_burst_is_two_frames_on_one_grid():
    """No training, no gap, no second table: the reference columns score 24 at
    the first frame's lead and again one frame length on, and under 12 at every
    other column. The first frame reads the same whether or not the second is
    asked for."""
    meta = recordings()["bw500-level1-two-frames"]
    audio = _audio("bw500-level1-two-frames")
    (a, b), = rx.burst_spans(audio)
    r = rx.INDEX_RECORDS[1]
    off = rx._l3_grid(audio, a, b, 1)
    band = rx._l3_band(audio, off, (b + r.dw - off) // r.dw, 1)
    hits = [rx._l3_ref_hits(band, c, 1) for c in range(len(band) - r.ncols + 1)]
    assert [c for c, h in enumerate(hits) if h >= rx._L3_GUARD_MIN] == [2, 126]
    assert max(h for c, h in enumerate(hits) if c not in (2, 126)) <= 6
    first, second = rx.decode_burst_frames(audio, a, b)
    assert (first.burst_columns, first.frames_in_burst) == (meta["burst_columns"], 2)
    assert first == rx.decode_l3_burst(audio, a, b, 1)
    assert second.c0 == first.c0 + r.ncols and second.crc_ok


@pytest.mark.parametrize("name", RECORDING_NAMES)
def test_the_reference_columns_name_the_record(name):
    """Levels 2 and 3 share a comb and all three are within a second of each other
    in length: the record is told by its own 24 reference columns, 24 of 24
    against no more than 8 at either other record."""
    audio = _audio(name)
    (a, b), = rx.burst_spans(audio)
    hits = {lv: rx._l3_align(audio, a, b, lv)[2] for lv in rx.INDEX_RECORDS}
    own = recordings()[name]["host_bitrate"]
    assert hits[own] == 24
    assert all(h < rx._L3_GUARD_MIN for lv, h in hits.items() if lv != own)


def test_the_gaps_pay_for_the_columns():
    """Records 1 and 0 close on the next record's seed after ``columns + 24``
    draws, record 2's own arithmetic — and record 0's measured bins sit within
    -1..+2 of the draws they are not, the wide-comb near miss BW2300 shows."""
    for rec in (0, 1, 2):
        tablegen.bw500_map2(rec)
        ref = tablegen.bw500_ref_cols(rec)
        assert len(set(ref.tolist())) == 24
        assert ref.max() < len(tablegen.bw500_alloc(rec))
    s = tablegen._bw500_gap(0)
    draw = []
    for _ in range(124):
        s = tablegen.step(s)
        draw.append(54 + tablegen.draw(s, 21))
    delta = ((tablegen.bw500_alloc(0) - np.array(draw) + 10) % 21) - 10
    assert set(delta.tolist()) == {-1, 0, 1, 2}


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("frames", [1, 2, 3])
def test_a_rendered_low_level_burst_reads_back(level, frames):
    """Round trip, payload-independent, through the level-blind dispatch — one
    frame, or several on the tape's layout: training once, then frames end to
    end on the record's grid."""
    r = rx.INDEX_RECORDS[level]
    rng = np.random.default_rng(level * 8 + frames)
    payloads = [bytes(rng.integers(0, 256, r.payload, dtype=np.uint8)) for _ in range(frames)]
    audio = tx.synth_l3_burst([tx.build_l3_frame(p, _MARKER, level) for p in payloads],
                              onset=1000, level=level)
    assert len(audio) == 1000 + (r.lead + frames * r.ncols) * r.dw
    got = rx.decode_burst_frames(audio, 1000, len(audio))
    assert [f.c0 for f in got] == [r.lead + k * r.ncols for k in range(frames)]
    for f, payload in zip(got, payloads):
        assert f.level == level and f.crc_ok and f.self_consistency == 1.0
        assert f.payload == payload and f.marker == _MARKER
        assert f.frames_in_burst == frames


@pytest.mark.parametrize("name", RECORDING_NAMES)
def test_the_rendered_body_is_the_recordings_own_columns(name):
    """Bin for bin against the tape: the transmitter is the decoder's inverse at
    these records too, the training columns aside."""
    audio = _audio(name)
    (a, b), = rx.burst_spans(audio)
    level = recordings()[name]["host_bitrate"]
    r = rx.INDEX_RECORDS[level]
    frames = rx.decode_burst_frames(audio, a, b, level)
    off = rx._l3_grid(audio, a, b, level)
    lit = rx._l3_band(audio, off, r.lead + len(frames) * r.ncols, level).argmax(1) + r.first_bin
    rendered = np.concatenate([tx.l3_column_bins(f.frame_bytes, level) for f in frames])
    assert (lit[r.lead:] == rendered).all()
