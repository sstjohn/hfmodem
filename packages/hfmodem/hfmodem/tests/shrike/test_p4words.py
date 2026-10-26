# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The measured word bank: each word is itself, none is another, none is noise."""
import itertools

import numpy as np
import pytest

from hfmodem.shrike import p4words


def test_every_word_is_itself():
    for name, word in p4words.BANK.items():
        assert p4words.correlate(word, word)[0] == pytest.approx(1.0)
        assert p4words.match(word) == (name, pytest.approx(1.0))


def test_no_word_is_another():
    """Ten-word bank: pairs stay below 0.72 except V27/GATEWAY at 0.821;
    all stay below the 0.88 knee."""
    for a, b in itertools.combinations(p4words.BANK, 2):
        r = p4words.correlate(p4words.BANK[a], p4words.BANK[b])[0]
        assert (r >= 0.80 if {a, b} == {'V27', 'GATEWAY'} else r <= 0.72), (a, b, r)
        assert r < p4words.MATCH_KNEE


def test_the_lengths_are_the_measured_ones():
    assert [len(w) for w in p4words.BANK.values()] == [27, 27, 27, 27, 27, 27, 27, 27, 20, 18]
    assert all(np.allclose(np.abs(w), 1.0) for w in p4words.BANK.values())


def test_a_noise_word_is_named_nothing():
    """Ten-word null: peak 0.882009, rejected at the search boundary."""
    rng = np.random.default_rng(101)
    reads = [p4words.match(np.exp(2j * np.pi * rng.random(n)))
             for n in (18,20,27) for _ in range(1000)]
    assert all(name is None for name,score in reads)
    assert max(score for name,score in reads) == pytest.approx(.8820090266,abs=1e-9)
    assert p4words.match(np.zeros(20, complex) + 1e-9)[0] is None


def test_a_word_survives_the_carrier_the_lock_cannot_see():
    """The grid lock is blind to tens of hertz, so the match takes them out."""
    word = p4words.BANK['CALLER20']
    for hz in (-18.0, -4.0, 4.0, 18.0):
        turns = np.arange(word.size) / p4words.SYMBOL_RATE
        heard = word * np.exp(2j * np.pi * hz * turns) * np.exp(1j * 0.7)
        assert p4words.match(heard) == ('CALLER20', pytest.approx(1.0, abs=1e-6))


def test_a_word_too_short_to_be_one_is_not_matched():
    assert p4words.match(p4words.BANK['BREAKIN'][:8]) == (None, 0.0)


def test_a_fit_on_the_search_boundary_names_nothing():
    """A maximum at the edge is the search out of room, not a word."""
    word = p4words.BANK['CALLER20']
    turns = np.arange(word.size) / p4words.SYMBOL_RATE
    railed = word * np.exp(2j * np.pi * (p4words.MATCH_HZ + 6.0) * turns)
    name, score = p4words.match(railed)
    assert name is None
    assert not p4words.interior(*p4words.correlate(word, railed)[1:])
    inside = word * np.exp(2j * np.pi * (p4words.MATCH_HZ - 4.0) * turns)
    assert p4words.match(inside)[0] == 'CALLER20'
    assert p4words.interior(*p4words.correlate(word, inside)[1:])


def test_the_boundary_test_is_on_both_axes():
    assert p4words.interior(0, 0.0)
    assert not p4words.interior(p4words.MATCH_SHIFT, 0.0)
    assert not p4words.interior(-p4words.MATCH_SHIFT, 0.0)
    assert not p4words.interior(0, p4words.MATCH_HZ)
    assert not p4words.interior(0, -p4words.MATCH_HZ)


def test_correlate_says_where_it_found_its_maximum():
    word = p4words.BANK['GATEWAY']
    r, shift, hz = p4words.correlate(word, word)
    assert (r, shift, hz) == (pytest.approx(1.0), 0, pytest.approx(0.0))


@pytest.mark.parametrize('name', ['T27', 'X27', 'V27'])
def test_cached_word_audio_round_trip_with_zero_offset_placement(name):
    from types import SimpleNamespace
    from hfmodem.shrike import onair, p4oracle, p4robusttx
    from hfmodem.shrike.p4receiver import Kind
    fs = 48000
    probe = SimpleNamespace(values=())
    cache = onair._p4_link_waveforms(probe)
    wave = cache[(Kind.WORD, name, None, 'A', b'', 0)]
    assert np.array_equal(wave, p4robusttx.robust_word(name))
    assert onair._p4_link_waveforms(probe) is cache
    audio = np.r_[np.zeros(fs//10), wave, np.zeros(fs//10)]
    coherence, symbols, first = p4oracle.placed_despread(audio, fs, .1, 0., count=27)
    matched, score, _ = p4words.match_placed(symbols, cfo_hz=0., carrier_offsets={})
    assert coherence > p4oracle.ROBUST_FRACTION_KNEE
    assert matched == name and score > p4words.MATCH_KNEE
    assert abs(first-fs//10) <= 2
    assert p4words.CARRIER_HZ['T27'] == 3.80
