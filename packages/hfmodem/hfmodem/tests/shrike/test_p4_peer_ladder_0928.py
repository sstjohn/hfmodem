# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair, p4robusttx, p4oracle
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role, Kind, Observation, Registered
from hfmodem.shrike.p4observation import RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import machine
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx, FS

ROOT = Path(__file__).resolve().parents[5]
START = 1546767


def capture(arm='2234'):
    path = ROOT/f'captures/onair-0928-{arm}/stream.flac'
    require_input(path)
    return onair.session.load_wav(str(path), FS)


def receiver(at, cfo=4.):
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    rx.enable_hold()
    rx.note_emission(Emission(Kind.WORD, Role.IRS, at-round(.330*FS), .244))
    rx.peer_onset = at-60000
    rx._peer_heads = [at-120000, at-60000]
    return rx


@pytest.mark.parametrize('k,value', [(k, 13 if k < 2 else 49 if k < 4 else 53) for k in range(11)])
def test_recorded_controls_resolve_without_an_energy_envelope(k, value):
    audio = capture()
    at = START+k*60000
    rx = receiver(at)
    origin = at-round(.05*FS)
    early, end = at+round(.810*FS), at+round(.846*FS)
    rx.read_partial(audio[origin:early], origin, deadline=early)
    records = rx.read_window(audio[origin:end], origin, whole=True)
    full = [r.observation for r in records if r.outcome in (Registered.NEW, Registered.CONFIRMED)
            and not r.observation.provisional]
    assert len(full) == 1
    obs = full[0]
    assert obs.kind == Kind.SHORT and obs.value == value
    assert obs.walsh_peak >= 62 and obs.placed
    assert abs(obs.stream_sample-at) <= 8
    assert rx.finish_cycle(at) == 0
    assert rx.clock.cycle_s == 1.25
    assert rx.read_window(audio[origin:end], origin, whole=True) == []


@pytest.mark.parametrize('value', [53])
def test_peer_53_gets_measured_stock_answer(value):
    m, _ = machine()
    m.state = State.IRS_ANSWER
    at = FS
    obs = Observation(Kind.SHORT, FS, 0, at, at, .758, 4., .5,
                      header='B', value=value, walsh_peak=64, placed=True)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.81*FS))
    assert m.decide(at+round(.920*FS)-m.lead).word == 'U27'
    assert any(e.get('event') == 'peer-control-no-progress' and e['value'] == value
               for e in m.link_events)


@pytest.mark.parametrize('header', ['A', 'B'])
@pytest.mark.parametrize('value', [13,49])
def test_known_peer_control_still_gets_its_measured_answer(header, value):
    at = FS
    rx = receiver(at)
    signal = p4robusttx.robust_short(value, header)
    origin = at-2400
    audio = np.r_[np.zeros(2400), signal, np.zeros(4800)]
    m, _ = machine()
    m.state = State.IRS_ANSWER
    for reg in rx.read_window(audio, origin, whole=True):
        m.observe(reg, now=at+round(.81*FS))
    sent = m.decide(at+round(.920*FS)-m.lead)
    assert sent.word == 'X27' and abs(sent.stream_sample-at-44160) <= 2


@pytest.mark.parametrize('arm,start,opening,expected', [
    ('2234',START,1262627,[13,13,49,49]+[53]*7),
    ('2258',1786878,1622725,[13,13]+[49]*9),
])
def test_recorded_driver_answers_measured_peer_controls(arm,start,opening,expected):
    audio = capture(arm)[:start+10*60000+FS]
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_probe.emissions = [dict(audio_start=opening, audio_end=opening+36367, value=7, header='A')]
    tx.p4_probe.values = (7, 38)
    stream.pos = opening+36367
    tx.settle = .044
    key = tx._tx
    def flushing(audio, what):
        key(audio, what)
        stream.pos = tx.tx_end
    tx._tx = flushing
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    onair._run_p4_link(tx, stream, onair._MasterGrid(0,60000,0,packet_n=36000,
                                                  cs_n=12000,d_max_n=4800))
    assert b'RMS Trimode 1.4.4.0 KB5LZK at t' in b''.join(delivered)
    replies = [e for e in tx.p4_probe.emissions if e['audio_start'] > start]
    assert len(replies) == len(expected)
    assert [e['word'] for e in replies] == ['U27' if v == 53 else 'X27' for v in expected]
    assert abs(replies[0]['audio_start']-(start+44160)) <= 12
    assert all(abs(e['audio_start']-(start+k*60000+44160)) <= 12
               for k,e in enumerate(replies))
    controls = [r.observation for r in tx.sessrx.p4.registrations
                if r.observation.kind == Kind.SHORT and not r.observation.provisional]
    assert [o.value for o in controls] == expected


@pytest.mark.parametrize('k,value', [(k,13 if k < 2 else 49) for k in range(15)])
def test_later_arm_keeps_repeating_49_without_an_answer(k, value):
    audio = capture('2258')
    at = 1786878+k*60000
    rx = receiver(at)
    origin = at-2400
    end = at+round(.846*FS)
    records = rx.read_window(audio[origin:end], origin, whole=True)
    assert len(records) == 1
    obs = records[0].observation
    assert obs.kind == Kind.SHORT and obs.value == value
    assert abs(obs.stream_sample-at) <= 12
    assert rx.packet_read['reason'] == 'decoded'
    assert rx.finish_cycle(at) == 0


def test_unread_later_copy_reports_failed_codeword_instead_of_live_energy():
    audio = capture('2258')
    at = 1786878+15*60000
    rx = receiver(at)
    origin = at-2400
    records = rx.read_window(audio[origin:at+round(.846*FS)], origin, whole=True)
    assert not any(r.observation.value is not None for r in records)
    assert rx.packet_read['reason'] == 'no-walsh'
    assert rx.packet_read['walsh_peak'] == 24
    assert rx.finish_cycle(at) == 1


@pytest.mark.parametrize('signal', ['silence', 'noise', 'early', 'late'])
def test_placed_short_requires_a_whole_codeword_at_its_clock(signal):
    at = FS
    rx = receiver(at)
    origin = at-2400
    if signal == 'noise':
        audio = np.random.default_rng(928).normal(0,.1,FS)
    elif signal == 'silence':
        audio = np.zeros(FS)
    else:
        offset = 2400 if signal == 'early' else 2400+round(.15*FS)
        audio = np.r_[np.zeros(offset), p4robusttx.robust_short(13), np.zeros(4800)]
        if signal == 'early':
            audio = audio[:round(.60*FS)]
    records = rx.read_window(audio,origin,whole=True)
    assert not any(r.observation.value is not None for r in records)
    assert rx.finish_cycle(at) == 1


def test_unprogressed_53_counts_physical_copies_once_without_stopping():
    m, _ = machine()
    m.state = State.IRS_ANSWER
    for k in range(8):
        at = FS+k*60000
        obs = Observation(Kind.SHORT, FS, 0, at, at, .758,4.,.5,
                          header='?',value=53,walsh_peak=64,placed=True)
        m.observe(RegisteredObservation(replace(obs,provisional=True),Registered.NEW))
        assert len(m._unsupported_copies) == k
        m.observe(RegisteredObservation(obs,Registered.CONFIRMED))
        m.observe(RegisteredObservation(replace(obs,stream_sample=at+2),Registered.DUPLICATE))
        assert len(m._unsupported_copies) == k+1
        proposal = m.decide(at+44160-m.lead)
        assert proposal.word == 'U27'
        m.keyed()
    assert m.state == State.IRS_ANSWER
    assert m.reason is None
    assert [e['copies'] for e in m.link_events if e['event'] == 'peer-control-no-progress'] == list(range(1,9))


def test_supported_packet_resets_unsupported_control_budget():
    m, _ = machine()
    m.state = State.IRS_ANSWER
    obs = Observation(Kind.SHORT, FS, 0, FS, FS, .758,4.,.5,
                      header='B',value=53,walsh_peak=64,placed=True)
    m.observe(RegisteredObservation(obs,Registered.NEW))
    m.observe(RegisteredObservation(replace(obs,stream_sample=FS+60000,value=49),Registered.NEW))
    assert m._unsupported_copies == []


def test_recorded_49_answer_is_followed_by_53_on_the_word_turnaround():
    audio = capture()
    probe = json.loads((ROOT/'captures/onair-0928-2234/p4-probe.json').read_text())
    keyed = next(e for e in probe['emissions'] if e['audio_start'] == 1770925)
    assert keyed['word'] == 'X27'
    for delta, value in [(-44160,49),(15840,53)]:
        at = keyed['audio_start']+delta
        field = p4oracle.robust_at(audio[at-2400:at+43200],FS,.05,4.)
        assert field['p4_value'] == value and field['p4_walsh_peak'] == 64
        assert abs(field['first_chip']-2400) <= 4


@pytest.mark.parametrize('arm,at,cfo,value', [
    ('2255',2088020,-4.,5),
    ('2255',2148020,-4.,5),
    ('2301',1488064,4.,13),
    ('2301',7368064,-4.,52),
    ('2301',8568064,4.,55),
    ('2307',2206746,4.,49),
    ('2317',1127782,-4.,13),
    ('2317',1187782,-4.,13),
])
def test_later_ve3kpg_controls_keep_their_actual_values(arm,at,cfo,value):
    audio = capture(arm)
    rx = receiver(at,cfo)
    origin = at-2400
    end = at+round(.846*FS)
    (reg,) = rx.read_window(audio[origin:end],origin,whole=True)
    assert reg.observation.value == value
    assert reg.observation.walsh_peak >= 60
    assert abs(reg.observation.stream_sample-at) <= 40
    assert rx.finish_cycle(at) == 0


def test_2031_opening_ladder_answers_v61_with_x27_and_keeps_v53_u27():
    cap = ROOT/'captures/onair-0930-2031'
    require_input(cap/'stream.flac')
    require_input(cap/'p4-probe.json')
    audio = onair.session.load_wav(str(cap/'stream.flac'), FS)
    probe = json.loads((cap/'p4-probe.json').read_text())
    controls = [h for h in probe['heard'] if h['kind'] == 'short'
                and not h['provisional']]
    assert [h['value'] for h in controls] == [61]*5 + [49]*2 + [53]*3
    m, _ = machine(role='irs', normal_ack='parity')
    for h in controls:
        at = h['onset_sample']
        rx = receiver(at, h['cfo_hz'])
        records = rx.read_window(audio[at-2400:at+round(.846*FS)], at-2400, whole=True)
        full = [r for r in records if r.observation.kind == Kind.SHORT
                and not r.observation.provisional]
        assert len(full) == 1
        reg, = full
        assert reg.observation.value == h['value']
        assert abs(reg.observation.stream_sample-at) <= 12
        m.observe(reg, now=at+round(.81*FS))
        p = m.decide(at+round(.846*FS))
        assert p.word == ('U27' if h['value'] == 53 else 'X27')
        assert abs(p.stream_sample-at-round(.920*FS)) <= 12
        m.keyed()
