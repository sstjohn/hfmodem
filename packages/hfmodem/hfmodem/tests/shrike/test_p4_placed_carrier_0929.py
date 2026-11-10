# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.shrike import p4oracle, p4words
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, P4Receiver, Registered
from hfmodem.tests.shrike import p4_stock_peer as stock

FS = 48000
WORDS = {'X27', 'V27', 'Y27', 'U27', 'T27'}


@pytest.mark.parametrize('cfo', [-10., 0., 10.])
@pytest.mark.parametrize('snr', [5., 10.])
def test_stock_words_acquire_then_alternate_on_the_body_carrier(cfo, snr):
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rng = np.random.default_rng(92904)
    for index, name in enumerate(['Y27', 'X27', 'V27', 'X27', 'U27', 'X27', 'T27']):
        at = FS + index*60000
        wave = stock.word(name, cfo=cfo)
        audio = np.pad(wave, (2400, 4800))
        audio += rng.normal(0, np.sqrt(np.mean(wave**2)*10**(-snr/10)), len(audio))
        origin = at-2400
        early = p4oracle.word_at(audio[:2400+round(.215*FS)], FS, origin, .05,
                                cfo_hz=rx.cfo_hz, provisional=True, expected_words=WORDS)
        assert early is not None and early.word == name and early.provisional
        assert early.score >= .88
        rx._emit(early, [])
        assert rx.cfo_hz == pytest.approx(cfo, abs=.4)
        full = p4oracle.word_at(audio, FS, origin, .05, cfo_hz=rx.cfo_hz, expected_words=WORDS)
        assert full is not None and full.word == name and not full.provisional
        assert full.cfo_hz == pytest.approx(cfo+p4words.CARRIER_HZ[name], abs=.4)
        records = []
        rx._emit(full, records)
        assert records[0].outcome == Registered.CONFIRMED
        assert rx.cfo_hz == pytest.approx(cfo, abs=.4)
        assert rx.clock.cfo_hz == rx.cfo_hz


@pytest.mark.parametrize('name', sorted(WORDS))
@pytest.mark.parametrize('count', [20, 27])
@pytest.mark.parametrize('residual', [-3.5, -3., 0., 3., 3.5])
def test_each_word_keeps_the_same_residual_boundary(name, count, residual):
    word = p4words.BANK[name][:count]
    heard = word*np.exp(2j*np.pi*(p4words.CARRIER_HZ[name]+residual)*np.arange(count)/112.5)
    matched, score, _ = p4words.match_placed(heard, cfo_hz=0.)
    fitted, scores = p4words.placed_scores(heard[None, :])
    assert fitted[0][0] == name
    assert (abs(fitted[0][2]) < 2.75) == (residual == 0)
    assert matched == (name if residual == 0 else None)
    if residual == 0:
        assert score == pytest.approx(1.) and scores[0] == pytest.approx(1.)


@pytest.mark.parametrize('snr', [5., 10.])
@pytest.mark.parametrize('hold', [False, True])
@pytest.mark.parametrize('cfo', [-10., 0., 10.])
def test_stock_head_enriches_without_learning_its_offset_twice(hold, cfo, snr):
    wave = stock.body(b'body carrier', 0x40, cfo=cfo)
    audio = np.pad(wave, (FS, FS))
    audio += np.random.default_rng(92905).normal(0, np.sqrt(np.mean(wave**2)*10**(-snr/10)), len(audio))
    rx = P4Receiver(FS, admitted=True, cfo_hz=cfo, breakin=True)
    if hold:
        rx.enable_hold()
    rx.expected = True
    rx.clock = Clock(Expect.WORD, FS, .935, 1.25, cfo)
    regs = rx.read_partial(audio[:FS+round(.215*FS)], 0, deadline=FS+round(.215*FS))
    regs += rx.read_window(audio, 0, whole=True)
    bodies = [r.observation for r in regs if r.observation.kind == Kind.BODY]
    assert len(bodies) == 1 and bodies[0].frame.data.startswith(b'body carrier')
    assert bodies[0].cfo_hz == pytest.approx(cfo, abs=.25)
    assert rx.cfo_hz == pytest.approx(cfo, abs=.25)


def test_offsets_are_the_stock_measurements():
    assert p4words.CARRIER_HZ == {'X27': 3.65, 'V27': -1.45, 'Y27': 1.4, 'U27': -3.3, 'T27': 3.8}


def test_unmeasured_word_does_not_move_a_calibrated_body_carrier():
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.expected = True
    for index, name in enumerate(['Z27', 'X27', 'Z27', 'V27', 'X27', 'U27']):
        at = FS+index*60000
        wave = stock.word(name, cfo=2.3 if name == 'Z27' else 0.)
        audio = np.pad(wave, (2400, 4800))
        origin = at-2400
        if index == 0:
            obs = p4oracle.word_at(audio, FS, origin, .05, cfo_hz=None,
                                  provisional=True, expected_words={'Z27'})
            assert obs is not None and obs.word == 'Z27'
            rx._emit(obs, [])
            assert not rx._body_carrier_known
        else:
            rx.clock.next_due = at
            partial = rx.read_partial(audio[:2400+round(.215*FS)], origin,
                                      deadline=at+round(.215*FS))
            assert partial and partial[0].observation.word == name
            full = rx.read_window(audio, origin, whole=True)
            assert any(r.outcome == Registered.CONFIRMED for r in full)
            assert rx.cfo_hz == pytest.approx(0., abs=.1)
            assert rx.clock.cfo_hz == rx.cfo_hz
    rx.new_call()
    assert not rx._body_carrier_known and rx._word_cfo_hz is None


@pytest.mark.parametrize('name', sorted(WORDS))
@pytest.mark.parametrize('cfo', [0., 10.])
def test_full_scan_acquisition_learns_the_body_carrier(name, cfo):
    audio = np.pad(stock.word(name, cfo=cfo), (2400, 4800))
    hits = p4oracle.word_scan(audio, FS, origin=0, whole=True)
    assert len(hits) == 1 and hits[0].word == name
    assert not hits[0].provisional and hits[0].score >= .99
    rx = P4Receiver(FS, admitted=True)
    rx._emit(hits[0], [])
    assert rx.cfo_hz == pytest.approx(cfo, abs=.25)


@pytest.mark.parametrize('index', [0, 1])
def test_recorded_unnamed_ladder_head_keeps_absolute_carrier_until_confirmation(index, monkeypatch):
    from hfmodem.shrike.p4receiver import Emission, Role
    from hfmodem.tests.shrike.test_p4_probe_handoff_0929 import capture

    audio, meta = capture()
    emission = meta['emissions'][index]
    at, origin = emission['audio_start'], emission['audio_end']
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.expected_words = {'X27', 'Y27', 'Z27'}
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, at, (origin-at)/FS))
    end = at+round(1.138*FS)
    early, = rx.read_partial(audio[origin:end], origin, deadline=end)
    head = early.observation
    assert head.word is None and head.provisional
    assert rx.cfo_hz is None and rx.clock.cfo_hz == 0.
    assert rx._word_cfo_hz is None and not rx._body_carrier_known
    placed = p4oracle.word_at
    references = []

    def track_reference(*args, **kwargs):
        references.append((kwargs['cfo_hz'], kwargs.get('carrier_offsets')))
        return placed(*args, **kwargs)

    monkeypatch.setattr(p4oracle, 'word_at', track_reference)
    end = head.stream_sample+round(.335*FS)
    records = rx.read_window(audio[origin:end], origin, whole=True)
    full = next(r for r in records if r.outcome == Registered.CONFIRMED)
    assert full.responds_to == at and full.observation.word == 'Y27'
    # Full estimation can refine the partial CFO. The recheck must borrow the
    # absolute head carrier without adding Y27's measured offset a second time.
    assert references[0] == (head.cfo_hz, {})
    assert rx.cfo_hz == pytest.approx(full.observation.cfo_hz-1.4)
    assert rx.clock.cfo_hz == rx.cfo_hz and rx._body_carrier_known
