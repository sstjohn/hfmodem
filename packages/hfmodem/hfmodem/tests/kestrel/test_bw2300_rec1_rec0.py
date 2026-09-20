# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Records 1 and 0 — the bottom two rungs of the free ladder — off real VARA audio.

Host ``BITRATE (2)`` and ``BITRATE (1)``, 41 and 24 bps. A stock pair was walked
down to them on 2026-09-04 by band-limited noise on the *caller's* input, since a
transmitter's level follows its receiver's feedback, and the noise was cut
mid-session so the record the responder had dropped to stayed keyed for another
over or two with nothing on top of it. Those are the overs here.

Both records ride a rate-1/3 turbo the shipped chain had no path to, and both
carry three bits a column rather than four: record 1 on record 2's comb and
column count, record 0 on 2048-sample symbols reading 64 bins of a wider one. The
bin tables are measured, so these are the tests that hold them — the overs decode
to the counter the responder was pushing, each record's reference columns pin at
its own level and nowhere else, and the transmit chain renders a burst its own
receiver reads back.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from hfmodem.kestrel.rx import tablegen
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.tests.kestrel import corpora

#: RMS on the caller's input above which the noise that commanded the gear-down is
#: still running. A stock over lands at 0.048 there and the loudest rung's noise at
#: 0.24, so anything between is an over the noise has not been cut for.
_QUIET = 0.10


def _clean_overs(level: int):
    """``(audio, start, end)`` per over of ``level`` the noise was already cut for.

    A key-up is stamped with the last level its host announced, and on a gear
    change that announcement can be one over ahead of the waveform: three of the
    release tape's key-ups stamped record 0 are 232 and 236 blocks of 1024 long,
    which is a record-1 over. The emission length settles it."""
    man = json.loads((corpora.BW2300_LADDER / "manifest.json").read_text())
    out = []
    for rung in man["rungs"]:
        if not rung.get("b2a_wav"):
            continue
        x = np.fromfile(rung["b2a_wav"], dtype=np.float32, offset=44).astype(float)
        for o in rung["responder_overs"]:
            s, e = o["sample_start"], min(o["sample_end"], len(x))
            r = rx.RECORDS[level]
            emitted = (r.lead - rx._BASE_LEADIN + r.ncols) * r.dw50
            if o["record"] != level or e - s < emitted - r.dw50:
                continue
            if np.sqrt((x[s:e] ** 2).mean()) <= _QUIET:
                out.append((x, s, e))
    return out


def _counter_run(frame: bytes) -> int:
    n = 1
    while n < len(frame) and frame[n] == (frame[n - 1] + 1) & 0xFF:
        n += 1
    return n


@pytest.mark.parametrize("level", [0, 1])
@corpora.requires_bw2300_ladder
def test_the_record_decodes_its_overs_to_the_counter_the_peer_was_pushing(level):
    overs = _clean_overs(level)
    assert len(overs) >= 3, level
    runs = []
    for audio, s, e in overs:
        fr = rx.decode_over(audio, max(0, s - 8192), min(len(audio), e + 8192),
                            level=level)
        assert fr.crc_ok, fr.frame_bytes.hex()
        runs.append(_counter_run(fr.payload))
    # the ARQ layer's own control byte closes the payload, so a mid-transfer
    # frame is one byte short of a whole counter run
    assert sum(r >= rx.payload_bytes(level) - 1 for r in runs) >= 3, runs


@pytest.mark.parametrize("level", [0, 1])
@corpora.requires_bw2300_ladder
def test_the_reference_columns_pin_at_its_own_record_and_nowhere_else(level):
    audio, s, e = _clean_overs(level)[0]
    seg = audio[max(0, s - 8192):min(len(audio), e + 8192)]
    scores = {lv: hits for lv, hits, _, _ in rx.index_guard(seg)}
    assert scores[level] == 24, scores
    assert max(v for lv, v in scores.items() if lv != level) <= 9, scores


@corpora.requires_bw2300_ladder
def test_a_whole_session_decodes_at_the_record_it_dropped_to():
    # the segmenter and the level fallback together, with nothing told about the
    # record: the level is not signalled anywhere but in the waveform.
    audio, s, e = _clean_overs(0)[0]
    got = rx.decode_overs(audio[max(0, s - 24000):min(len(audio), e + 24000)])
    assert got and all(f.crc_ok for f in got)
    assert any(f.level == 0 for f in got)


@pytest.mark.parametrize("level", [0, 1])
def test_the_transmit_chain_renders_a_burst_its_receiver_reads_back(level):
    payload = bytes((i * 7 + 3) & 0xFF for i in range(rx.payload_bytes(level)))
    burst = tx.synth_burst(payload, level=level)
    assert len(burst) == rx.burst_length(level)
    fr = rx.decode_burst(burst, level=level)
    assert fr.crc_ok and fr.payload == payload


@pytest.mark.parametrize("level", [0, 1])
def test_the_record_takes_the_rate_one_third_branch(level):
    r = rx.RECORDS[level]
    assert r.coded == 3 * r.n_info + 12
    assert r.frame_bytes * 8 == r.n_info
    assert r.bpc == 3 and r.stride == 2


def test_record_1_takes_record_2s_reference_layout_and_record_0_has_its_own():
    # record 1 lights the same 24 columns of the same 228 as record 2, so
    # `map1_col2` gates both records and there is no second gate to know
    assert (rx._ROLES[1][0] == rx._ROLES[2][0]).all()
    assert len(rx._ROLES[0][0]) == 24
    assert not set(rx._ROLES[0][0].tolist()) & set(rx._ROLES[2][0].tolist()[:1])
    assert len(tablegen.base_bins_col0()) == rx.RECORDS[0].ncols == 124
    assert len(tablegen.base_bins_col1()) == rx.RECORDS[1].ncols == 228
    assert tablegen.map1_col0().sum() == 24
