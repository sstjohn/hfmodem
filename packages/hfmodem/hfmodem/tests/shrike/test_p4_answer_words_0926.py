# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""X27 and Y27: what VE3KPG and N9VC answer robust shorts with, read and keyed.

onair-0926-1218's answers stood 2-7 dB over the floor and its arm log read
both windows as `none`; 1228's and the Kiwi's are clean."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys

import numpy as np
import pytest

from hfmodem.shrike import onair, p4oracle, p4robusttx, p4words
from hfmodem.tests.shrike.test_n9vc_links import _load
from hfmodem.tests.shrike.test_p1_robust_0926 import _peer_wav

TAPES = ('ve3kpg-0926-1218-answers', 've3kpg-0926-1228-answers', 'n9vc-0925-kiwi-answers')


def _named(rows):
    return [(round(row['onset_ms'] / 1e3, 2), row['word']) for row in rows
            if row['reply_class'].startswith('p4-word:')]


@pytest.mark.parametrize('name', TAPES)
def test_every_answer_on_the_tape_is_named_and_nothing_else(name):
    meta, x, fs = _load(name)
    heard = [(at, word) for at, word in _named(p4oracle.read_window(x, fs, meta.get('offset_hz')))
             if word != 'CALLER20']
    assert heard == [(round(w['at_s'], 2), w['word']) for w in meta['words']]


def test_the_faded_arm_reads_as_answers_in_its_query_lines():
    meta, x, fs = _load('ve3kpg-0926-1218-answers')
    row = p4oracle.query(1, None, 0, 0, x[round(5.5 * fs):round(6.1 * fs)], fs)
    assert row['reply_class'] == 'p4-word:X27'
    assert any('p4-word:X27 (X27 r 0.99' in line for line in p4oracle.lines(row))


def test_a_scan_names_no_word_in_noise_or_link_traffic():
    meta, x, fs = _load('ve3kpg-0926-1218-answers')
    start, end = meta['noise']
    assert p4oracle.word_scan(x[round(start * fs):round(end * fs)], fs) == []
    noise = np.random.default_rng(26).standard_normal(20 * fs).astype(np.float32)
    assert p4oracle.word_scan(noise, fs) == []


def test_the_words_are_new_to_the_bank_and_gateway_is_x27s_tail():
    for word in p4words.STEPS:
        assert p4words.match(p4words.BANK[word]) == (word, pytest.approx(1.0))
    tail = p4words.BANK['X27'][9:]
    turns = np.arange(tail.size) / p4words.SYMBOL_RATE
    fit = max(abs(np.vdot(p4words.BANK['GATEWAY'], tail * np.exp(2j * np.pi * hz * turns)))
              for hz in np.arange(-20, 20.5, 0.5)) / tail.size
    assert fit > 0.99


@pytest.mark.parametrize('name,index', [('ve3kpg-0926-1228-answers', 0),
                                        ('ve3kpg-0926-1228-answers', 1),
                                        ('n9vc-0925-kiwi-answers', 0),
                                        ('n9vc-0925-kiwi-answers', 1)])
def test_our_render_is_the_gateways_word(name, index):
    meta, x, fs = _load(name)
    word = meta['words'][index]
    at = round((word['at_s'] - 0.005) * fs)
    heard = p4oracle.despread_word(x[at:at + round(0.27 * fs)], fs)
    ours = p4oracle.despread_word(p4robusttx.robust_word(word['word'], fs), fs)
    assert p4words.match(ours) == (word['word'], pytest.approx(1.0, abs=1e-6))
    assert p4words.correlate(ours[:27], heard[:30])[0] > 0.9


def test_reply_words_parse_by_letter_or_bank_name():
    assert onair._probe_replies('x, Y27,X') == ('X27', 'Y27', 'X27')
    for bad in ('Q', 'BREAKIN', 'X,X,X,X,X', ''):
        with pytest.raises(Exception):
            onair._probe_replies(bad)


def test_the_replies_key_in_the_slots_after_the_last_short(tmp_path, monkeypatch):
    wav = tmp_path / "peer.wav"
    _peer_wav(wav)
    keyed, key = [], onair.RadioTx._tx

    def tx(self, audio, what, *a, **k):
        before = len(self.slots_used)
        key(self, audio, what, *a, **k)
        if len(self.slots_used) > before:
            keyed.append((self.slot, what))

    monkeypatch.setattr(onair.RadioTx, "_tx", tx)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", [
        "shrike.onair", "--replay", str(wav), "--max-cycles", "3",
        "--mycall", "W9SSJ", "--dxcall", "VE3KPG", "--dial", "7089500",
        "--outdir", str(tmp_path / "out"), "--after-connect", "p1-robust",
        "--p1-status-bits45", "2", "--p4-probe-values", "7,38",
        "--p4-probe-headers", "A,A", "--p4-probe-cadence", "consecutive",
        "--p4-probe-reply", "X,X,X"])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    log = log.getvalue()
    aired = [what for _, what in keyed if not what.startswith("connect")]
    assert [what.split(" (")[0] for what in aired[1:]] == [
        "P4 PROBE ROBUST 1/2 v=7 hdr=A", "P4 PROBE ROBUST 2/2 v=38 hdr=A",
        "P4 PROBE REPLY 1/3 X27", "P4 PROBE REPLY 2/3 X27", "P4 PROBE REPLY 3/3 X27"], log
    probe = json.loads((tmp_path / "out" / "p4-probe.json").read_text())
    slots = [e["slot"] for e in probe["emissions"]]
    assert slots == list(range(slots[0], slots[0] + 5)), (slots, log)
    assert [e["word"] for e in probe["emissions"]] == [None, None, "X27", "X27", "X27"]
    x27 = hashlib.sha256(p4robusttx.robust_word("X27").tobytes()).hexdigest()
    assert all(e["waveform_sha256"] == x27 for e in probe["emissions"][2:])
    assert probe["replies"] == ["X27", "X27", "X27"]
    assert "receiving continuously" in log.split("reply 3/3")[1], log
