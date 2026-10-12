"""E5: hear the gateway's autonomous ladder without answering it."""
from dataclasses import replace
import numpy as np
import pytest

from hfmodem.shrike import onair, p4breakintx
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, body, machine, word
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
from hfmodem.tests.shrike.test_p4_s5_chirp_0930 import observation, recording


def short(at, value, header='B'):
    return RegisteredObservation(Observation(
        Kind.SHORT, FS, 0, at, at, .758, 0., .9,
        value=value, header=header, walsh_peak=64), Registered.NEW)


@pytest.mark.parametrize('greeting', [False, True])
def test_eight_silent_cycles_keep_every_registration_then_answer(greeting):
    m, _ = machine(role='irs', irs_silent_cycles=8)
    # 0928-2234's measured 13/A,B -> 49/A,B -> 53/B train; E5
    # removes our answers. A greeting may precede the first ladder short.
    rows = [short(FS+k*60000, v, 'A' if k in (0, 2, 4) else 'B')
            for k, v in enumerate([13, 13, 49, 49, 53, 53, 53, 53, 53])]
    if greeting:
        rows[0] = body(FS)
    for k, reg in enumerate(rows):
        at = reg.observation.stream_sample
        m.observe(reg, now=at+round(.81*FS))
        m.observe(reg, now=at+round(.82*FS))  # rereading is not another cycle
        m.silence(at+round(.846*FS))
        p = m.decide(at+round(.92*FS)-m.lead)
        if k < 8:
            assert p is None and not m._emissions
        else:
            assert p.word == 'U27'
            m.keyed()
    assert len(m.events) == len(rows)
    assert [e['event'] for e in m.link_events if e['event'].startswith('irs-silence-')] == [
        'irs-silence-start', 'irs-silence-end']
    assert m.state == State.IRS_ANSWER


def test_silent_greeting_takes_irs_without_keying_an_ack():
    m, _ = machine(irs_silent_cycles=2)
    m.seed_opening([dict(audio_start=0, value=38)], [], values=(7, 38))
    m.observe(body(FS), now=round(1.81*FS))
    assert m.decide(round(1.92*FS)-m.lead) is None
    assert m.state == State.IRS_ANSWER
    assert m.last.value == 38
    for k in (1, 2):
        at = FS+k*60000
        m.observe(body(at), now=at+round(.81*FS))
        p = m.decide(at+round(.92*FS)-m.lead)
        if k == 1:
            assert p is None
        else:
            assert p.word == 'X27'


def test_chirp_changes_the_silent_cycle_length():
    m, _ = machine(role='irs', irs_silent_cycles=3)
    m.observe(short(FS, 53))
    assert m.decide(round(1.92*FS)-m.lead) is None
    row = next(r for r in recording('0930-0012')['rows'] if r['kind'] == 'chirp')
    for k in range(3):
        obs = observation(dict(row, onset_sample=round((2.25+k*3.75)*FS)))
        m.observe(RegisteredObservation(obs, Registered.NEW), now=obs.stream_sample+round(3.3*FS))
        p = m.decide(obs.stream_sample+round(3.4*FS)-m.lead)
        if k < 2:
            assert p is None
        else:
            assert p.word == 'X27'


def test_silence_expires_without_releasing_a_deferred_greeting():
    m, _ = machine(role='irs', irs_silent_cycles=1)
    m.observe(body(FS), now=2*FS)  # CRC resolves after its decision deadline
    assert m.decide(2*FS) is None
    m.silence(round(2.25*FS))
    answer = m.decide(round(2.25*FS))
    assert answer.purpose is None and not answer.current_slot_accepted
    assert m._deferred is None
    m.keyed()
    m.observe(body(round(2.25*FS)), now=round(3.06*FS))
    assert m.decide(round(3.17*FS)-m.lead).word == 'X27'


def test_provisional_short_and_refinements_do_not_spend_extra_cycles():
    m, _ = machine(role='irs', irs_silent_cycles=2)
    full = short(FS, 13)
    partial = replace(full, observation=replace(full.observation, provisional=True))
    m.observe(partial)
    assert m._irs_silent_start == FS
    assert m.decide(round(1.92*FS)-m.lead) is None
    m.observe(replace(full, outcome=Registered.CONFIRMED, replaced=partial.observation))
    m.observe(replace(full, observation=replace(full.observation, stream_sample=FS+8),
                      outcome=Registered.ENRICHED, replaced=full.observation))
    for k in (1, 2):
        at = FS+k*60000-8
        m.observe(short(at, 49))
        p = m.decide(at+round(.92*FS)-m.lead)
        assert (p is None) == (k == 1)
    assert p.word == 'X27'


def test_queued_mail_cannot_request_from_unread_shorts_after_silence():
    m, queue = machine(role='irs', data='body', irs_silent_cycles=2)
    queue.on_host_data(b'queued mail')
    for k in range(3):
        at = FS+k*60000
        m.observe(short(at, 13))
        m.silence(at+round(.846*FS))
        p = m.decide(at+round(.92*FS)-m.lead)
        if k < 2:
            assert p is None
        else:
            assert p.kind == Kind.WORD and p.word == 'X27'
    assert queue._buffer_raw == len(b'queued mail')


def test_zero_is_off_and_words_do_not_start_silence():
    m, _ = machine(role='irs', irs_silent_cycles=2)
    m.observe(word('Y27', 0, partial=True))
    assert not [e for e in m.link_events if e['event'] == 'irs-silence-start']
    off, _ = machine(role='irs', irs_silent_cycles=0)
    off.observe(body(FS))
    assert off.decide(round(1.92*FS)-off.lead).word == 'X27'


def test_bare_request_at_the_end_of_silence_gets_its_normal_answer():
    m, _ = machine(role='irs', irs_silent_cycles=1)
    m.observe(body(FS))
    at = FS+60000
    m.observe(word('Y27', at, extent=.24), now=at+round(.81*FS))
    p = m.decide(at+round(.92*FS)-m.lead)
    assert p.kind == Kind.BODY and p.status == 0x40


def test_driver_keys_only_after_silence_and_logs_every_greeting(capsys):
    audio = np.zeros(7*FS)
    wave = p4breakintx.changeover_packet(b'hello', 0)
    onsets = [round((.923+k*1.25)*FS) for k in range(4)]
    for at in onsets:
        audio[at:at+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_irs_silent_cycles = 2
    tx.p4_link_options.p4_link_timeout = 6.
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    keys = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert [e['word'] for e in keys] == ['X27', 'X27']
    assert all(abs(e['at']-at-round(.92*FS)) <= 1 for e, at in zip(keys, onsets[2:]))
    heard = [r for r in tx.p4_probe.heard if r['kind'] == Kind.BODY]
    assert all(any(abs(r['onset_sample']-at) <= 1 for r in heard) for at in onsets)
    log = capsys.readouterr().out
    assert 'irs-silence-start' in log and 'irs-silence-end' in log


@pytest.mark.parametrize('ceiling,expected', [(3., 'session ceiling'), (40., 'peer silent')])
def test_driver_silence_keeps_receive_log_and_existing_stop_rules(capsys, ceiling, expected):
    audio = np.zeros(45*FS)
    at = round(.923*FS)
    wave = p4breakintx.changeover_packet(b'hello', 0)
    audio[at:at+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_irs_silent_cycles = 100
    tx.p4_link_options.p4_link_timeout = ceiling
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    reason = onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert reason == expected
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert any(r['kind'] == Kind.BODY for r in tx.p4_probe.heard)
    assert delivered == [b'hello']
    text = capsys.readouterr().out
    assert 'irs-silence-start' in text and 'irs-silence-end' in text


def test_negative_cycle_count_is_invalid():
    with pytest.raises(ValueError, match='silent'):
        machine(irs_silent_cycles=-1)
