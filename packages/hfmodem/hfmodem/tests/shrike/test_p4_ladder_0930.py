# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""S5(b,c): exact recorded short/word sequences, plus the 2031 audio head."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
from hfmodem.shrike.p4turn import State
from hfmodem.shrike.p4observation import Registered
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role, Kind
from hfmodem.tests.shrike.test_p4turn import machine, word, FS

LADDER = Path(__file__).parent/'fixtures/p4_ladder_0930.json'
if not (LADDER).exists():
    pytest.skip('p4_ladder_0930.json is not in this tree', allow_module_level=True)
SEQUENCES = json.loads(LADDER.read_text())
ROOT = Path(__file__).resolve().parents[5]


def opening(at=0):
    m, _ = machine()
    m.seed_opening([dict(audio_start=at, value=7)], [], values=(7, 38))
    return m


@pytest.mark.parametrize('arm', ['0929-1158', '0929-2055', '0929-2105', '0929-2138', '0930-0111'])
def test_recorded_ladder_keeps_key_positions_and_survives_every_answer(arm):
    seq = SEQUENCES[arm]
    m = opening(seq['emissions'][0]['audio_start'])
    emissions = seq['emissions'][1:]
    if seq['words'][-1]['word'] == 'Z27':
        # The tape ends at the budget failure. Demand the next 1.25 s slot.
        last = seq['emissions'][-1]
        emissions = emissions + [dict(audio_start=last['audio_start']+60000,
                                     value=38, header='A')]
    events = [(h['capture_end_sample'], 'read', h) for h in seq['words']]
    events += [(e['audio_start']-m.lead, 'key', e) for e in emissions]
    partials = {}
    keys = [m.last.stream_sample]
    for now, event, record in sorted(events, key=lambda e: e[0]):
        if event == 'read':
            h = record
            reg = word(h['word'], h['onset_sample'], partial=h['provisional'],
                       outcome=Registered(h['outcome']),
                       replaced=partials.get(h['responds_to']),
                       answered=h['answered'], responds_to=h['responds_to'])
            # Eligibility is recomputed with this branch's expected-word set;
            # the old receiver marked the recorded T27 answers unexpected.
            reg = replace(reg, observation=replace(
                reg.observation, unexpected=h['word'] not in m.expected_words()))
            if h['provisional']:
                partials[h['responds_to']] = reg.observation
            m.observe(reg, now=now)
        else:
            m.silence(now)
            p = m.decide(now)
            assert m.state != State.DOWN, (arm, now, m.reason)
            assert p is not None
            assert (p.stream_sample, p.value) == (record['audio_start'], record['value'])
            # The recorded transmitter alternated A/B incorrectly. Preserve
            # its raster and values, with B on every repeated value.
            seen = {e.value for e in m._emissions.values()}
            assert p.header == ('B' if p.value in seen else 'A')
            assert p.stream_sample-keys[-1] == 60000
            keys.append(m.keyed().stream_sample)
    assert m.last.value == 38 and m.state == State.ISS_LADDER
    if seq['words'][-1]['word'] == 'Z27':
        assert m.repeats == 0
    else:
        assert m._blocked_redue  # The recorded Y27 ends the bare-word ladder.


@pytest.mark.parametrize('name', ['T27', 'Z27', 'X27'])
def test_opening_expects_peer_words_even_on_v38(name):
    m = opening()
    assert name in m.expected_words()
    m.index = 1
    assert name in m.expected_words()


def test_2031_provisional_breakin_only_anchors_and_never_selects_a_step():
    seq = SEQUENCES['0929-2031']
    h = seq['words'][0]
    m = opening(seq['emissions'][0]['audio_start'])
    early = word('BREAKIN', h['onset_sample'], partial=True, responds_to=h['responds_to'])
    early = replace(early, observation=replace(early.observation, unexpected=True))
    m.observe(early, now=h['capture_end_sample'])
    decision = 840601  # The real, premature decision for key 842733.
    assert m.decide(decision) is None
    m.silence(decision)
    p = m.decide(decision)
    assert p is not None and p.value == 7
    assert p.stream_sample == h['onset_sample']+round(.940*FS)
    assert p.stream_sample != 842733


@pytest.mark.parametrize('keyed', [False, True])
def test_1158_provisional_x27_speculates_and_rolls_back_on_contradiction(keyed):
    seq = SEQUENCES['0929-1158']
    h = next(h for h in seq['words'] if h['word'] == 'X27' and h['provisional'])
    m = opening(h['responds_to'])
    early = word('X27', h['onset_sample'], partial=True, responds_to=h['responds_to'])
    m.observe(early, now=h['capture_end_sample'])
    due = h['responds_to']+60000
    p = m.decide(due-m.lead)
    assert (p.value, p.stream_sample) == (38, due)
    assert m.index == 0
    if keyed:
        m.keyed()
        assert m.index == 1
    m.observe(word('Z27', h['onset_sample'], outcome=Registered.CONTRADICTED,
                   replaced=early.observation, answered=due if keyed else None,
                   responds_to=h['responds_to']), now=due if keyed else due-m.lead)
    p = m.decide(due if keyed else due-m.lead)
    assert (p.value, p.stream_sample) == (7, due+60000 if keyed else due)


def test_v38_x27_is_an_answer_but_does_not_release_data():
    m = opening()
    m._short(FS, 1)
    m.keyed()
    at = FS+round(.94*FS)
    m.observe(word('X27', at, responds_to=FS), now=at+round(.3*FS))
    assert m.decide(at+round(.866*FS)) is None
    assert m.state == State.ISS_LADDER and m._opening_wait
    assert m._opening and m._next is None
    assert m.repeats == 0
    deadline = FS+round(10*1.25*FS)
    m.silence(deadline-1)
    assert m.decide(deadline-1) is None
    assert m.state == State.ISS_LADDER
    m.silence(deadline)
    assert m.decide(deadline) is None
    assert m.state == State.DOWN and m.reason == 'opening wait expired'


def test_2055_late_answer_credits_the_short_already_repeated():
    from hfmodem.shrike.p4observation import Registered
    seq = SEQUENCES['0929-2055']
    h = next(h for h in seq['words'] if not h['provisional'])
    first, second = seq['emissions'][:2]
    m = opening(first['audio_start'])
    early = word(h['word'], h['onset_sample'], partial=True, responds_to=h['responds_to'])
    m.observe(early)
    m._short(second['audio_start'], 0, repeat=True)
    m.keyed()
    m.observe(word(h['word'], h['onset_sample'], outcome=Registered.CONFIRMED,
                   replaced=early.observation, answered=h['answered'], responds_to=h['responds_to']),
              now=second['audio_start']+round(.1*FS))
    assert m.retry_budget()[0] == 0
    assert m._next is None


def test_2031_late_repeat_stays_on_the_peer_raster():
    h = SEQUENCES['0929-2031']['words'][0]
    m = opening(h['responds_to'])
    m.observe(word('BREAKIN', h['onset_sample'], partial=True, responds_to=h['responds_to']))
    now = h['onset_sample']+FS
    m.silence(now)
    assert m.decide(now).stream_sample == h['onset_sample']+round((.940+1.25)*FS)


def test_silence_is_still_bounded_and_stale_words_cannot_reset_it():
    m = opening()
    m._short(FS, 0, repeat=True)
    m.keyed()
    m._short(2*FS, 0, repeat=True)
    m.keyed()
    at = round(.94*FS)
    m.observe(word('Z27', at, responds_to=0), now=2*FS)
    assert m.repeats == 1
    for _ in range(8):
        m.silence(m.last.stream_sample+round(.9*FS))
        if m.state == State.DOWN:
            break
        m.keyed()
    assert m.reason == 'retry budget exhausted'


@pytest.mark.parametrize('name', ['T27', 'X27', 'Z27'])
def test_1158_bare_ladder_words_are_named_before_the_key_decision(name):
    from hfmodem.shrike import session
    seq = SEQUENCES['0929-1158']
    h = next(h for h in seq['words'] if h['word'] == name and h['provisional'])
    p = ROOT/'captures/onair-0929-1158/stream.flac'
    if not p.exists():
        pytest.skip('local 1158 recording absent')
    origin = h['onset_sample']-2400
    end = h['capture_end_sample']
    assert end < h['responds_to']+60000-round(.074*FS)
    audio = session.load_wav(str(p), FS)[origin:end]
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, h['responds_to'], .758))
    rx.expected_words = {'X27', 'Y27', 'Z27', 'T27'}
    records = rx.read_partial(audio, origin, deadline=end)
    assert any(r.observation.word == name and r.observation.provisional for r in records)


def test_1158_driver_keys_on_the_recorded_1250ms_raster():
    from hfmodem.shrike import onair, session
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    seq = SEQUENCES['0929-1158']
    p = ROOT/'captures/onair-0929-1158/stream.flac'
    if not p.exists():
        pytest.skip('local 1158 recording absent')
    stream = Stream(session.load_wav(str(p), FS)[:seq['words'][-1]['capture_end_sample']])
    tx = Tx(stream)
    entry = seq['emissions'][0]
    end = entry['audio_start']+36367
    tx.p4_probe.emissions = [dict(entry, audio_end=end)]
    tx.p4_probe.values = (7, 38)
    tx.settle = .044
    stream.pos = end
    onair._run_p4_link(tx, stream, onair._MasterGrid(0,60000,0,packet_n=36000,
                                                  cs_n=12000,d_max_n=4800))
    shorts = [e for e in tx.p4_probe.emissions if e.get('value') in (7, 38)]
    expected = [dict(e, header='B' if i and e['value'] in
                     {p['value'] for p in seq['emissions'][:i]} else 'A')
                for i, e in enumerate(seq['emissions'])]
    assert [{k: e[k] for k in ('audio_start', 'value', 'header')} for e in shorts] == expected
    assert any(h['word'] == 'Y27' and not h['provisional'] for h in tx.p4_probe.heard)


@pytest.mark.parametrize('elapsed', [.215, .275, .299, .335])
def test_2031_audio_word_is_unnamed_until_300ms(elapsed):
    from hfmodem.shrike import session
    p = ROOT/'captures/onair-0929-2031/stream.flac'
    if not p.exists():
        pytest.skip('local 2031 recording absent')
    onset = 827825
    origin = onset-2400
    audio = session.load_wav(str(p), FS)[origin:onset+round(elapsed*FS)]
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=4.)
    rx.note_emission(Emission(Kind.SHORT,
                              Role.ISS, 782733, .758))
    rx.expected_words = {'X27', 'Y27', 'Z27', 'T27'}
    if elapsed < .300:
        records = rx.read_partial(audio, origin, deadline=origin+len(audio))
        assert records
        assert all(r.observation.word is None and r.observation.provisional for r in records)
    else:
        records = rx.read_window(audio, origin, whole=True)
        assert any(r.observation.word == 'Y27' and not r.observation.provisional for r in records)


def test_2031_driver_reads_full_y27_before_keying_again():
    from hfmodem.shrike import onair, session
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    p = ROOT/'captures/onair-0929-2031/stream.flac'
    if not p.exists():
        pytest.skip('local 2031 recording absent')
    stream = Stream(session.load_wav(str(p), FS)[:round(19.*FS)])
    tx = Tx(stream)
    start = 782733
    tx.p4_probe.emissions = [dict(audio_start=start, audio_end=start+36367, value=7, header='A')]
    tx.p4_probe.values = (7, 38)
    tx.settle = .044
    stream.pos = start+36367
    onair._run_p4_link(tx, stream, onair._MasterGrid(0,60000,0,packet_n=36000,
                                                  cs_n=12000,d_max_n=4800))
    heard = tx.p4_probe.heard
    assert any(h['word'] == 'Y27' and not h['provisional']
               and abs(h['onset_sample']-827825) < 48 for h in heard)
    assert all(h['word'] is None for h in heard if h['provisional'] and h['kind'] == 'word')
    assert all(e['audio_start'] >= 827825+round(.920*FS)-48
               for e in tx.p4_probe.emissions[1:])
    assert not any(e.get('value') == 7 for e in tx.p4_probe.emissions[1:])
