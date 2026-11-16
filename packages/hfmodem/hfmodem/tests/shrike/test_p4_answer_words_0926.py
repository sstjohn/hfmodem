# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""X27 and Y27: what VE3KPG and N9VC answer robust shorts with, read and keyed.

onair-0926-1218's answers stood 2-7 dB over the floor and its arm log read
both windows as `none`; 1228's and the Kiwi's are clean."""
from __future__ import annotations


import numpy as np
import pytest

from hfmodem.shrike import p4oracle, p4robusttx, p4words
from hfmodem.tests.shrike.test_n9vc_links import _load

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
