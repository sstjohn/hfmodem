# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from pathlib import Path

import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike.p4observation import RESPONSE_S, RegisteredObservation
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role, Kind, Observation, Registered
from hfmodem.tests.shrike.test_p4turn import machine
from hfmodem.tests.shrike.test_p4_normal_short_0928 import normal, FS
from hfmodem.tests.shrike.test_p4_mail_0927 import options

ROOT = Path(__file__).resolve().parents[5]
LEAD = 2132


def recorded_answers(arm):
    path = ROOT/f'captures/onair-0929-{arm}/p4-probe.json'
    require_input(path)
    decision, answers = None, []
    for event in json.loads(path.read_text())['link_events']:
        if event['event'] == 'Decision':
            decision = event
        elif (event['event'] == 'Keyed' and event['word'] == 'V27'
              and decision['state'] == 'IRS_ANSWER'):
            answers.append((decision['at'], decision['decision_at'], event['at']))
    return answers


def answer(onset, decision_at, **kw):
    m, _ = machine(role='irs', prekey_lead=LEAD/FS, **kw)
    m.first_body = False
    m.observe(RegisteredObservation(normal(onset), Registered.NEW), now=onset+round(.810*FS))
    return m.decide(decision_at)


@pytest.mark.parametrize('arm,count', [('1235', 71), ('1151', 93)])
def test_default_offset_replays_the_recorded_v27_keys(arm, count):
    answers = recorded_answers(arm)
    assert len(answers) == count
    for onset, decision_at, keyed in answers:
        p = answer(onset, decision_at)
        assert (p.word, p.stream_sample) == ('V27', keyed)


@pytest.mark.parametrize('offset', [.900, .940])
def test_offset_moves_every_recorded_v27_key(offset):
    for onset, decision_at, keyed in recorded_answers('1235'):
        due = onset+round(offset*FS)
        early = answer(onset, due-LEAD-1, answer_offset=offset)
        assert early is None
        p = answer(onset, due-LEAD, answer_offset=offset)
        assert (p.word, p.stream_sample) == ('V27', due)
        assert p.stream_sample-keyed == round((offset-.920)*FS)


@pytest.mark.parametrize('offset', [.900, .920, .940])
def test_irs_answer_to_a_robust_short_keys_at_the_offset(offset):
    m, _ = machine(role='irs', answer_offset=offset)
    at = 480000
    obs = Observation(Kind.SHORT, FS, 0, at, at, .758, 0, .9, value=7)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.5*FS))
    p = m.decide(at+round(offset*FS)-m.lead)
    assert (p.word, p.stream_sample) == ('X27', at+round(offset*FS))


@pytest.mark.parametrize('offset', [.900, .920, .940])
def test_offset_leaves_the_peers_answer_bracket_alone(offset):
    m, _ = machine(answer_offset=offset)
    r = P4Receiver(FS, admitted=True)
    r.note_emission(Emission(Kind.SHORT, Role.ISS, 96000, .758))
    assert m.t[Kind.SHORT] == offset
    assert r.clock.next_due == 96000+round(RESPONSE_S*FS)
    assert r._response_target(Observation(Kind.WORD, FS, 0, r.clock.next_due, r.clock.next_due,
                                          .24, 0, .9, word='V27')) == 96000


@pytest.mark.parametrize('extra,offset', [((), .920), (('--p4-answer-offset', '0.9'), .900)])
def test_modem_flag_reaches_the_link_options(monkeypatch, extra, offset):
    assert options(monkeypatch, *extra).p4_answer_offset == offset


@pytest.mark.parametrize('offset', [.900, .920, .940])
def test_1235_body_to_normal_keeps_peer_raster_in_driver(monkeypatch, offset):
    from hfmodem.shrike import onair, p4turn
    from hfmodem.shrike.p4receiver import Clock, Expect
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    path = ROOT/'captures/onair-0929-1235/stream.flac'
    require_input(path)
    audio = onair.session.load_wav(str(path), FS)
    first = 1367728
    stream = Stream(audio)
    stream.pos = first-round(.1*FS)
    tx = Tx(stream)
    tx.settle = LEAD/FS
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_answer_offset = offset
    tx.p4_link_options.p4_link_timeout = 4.
    rx = tx.sessrx.p4
    rx.cfo_hz = .628728
    rx.clock = Clock(Expect.PACKET, first, .330, 1.25, rx.cfo_hz)
    init = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: init(*a, **dict(kw, role='irs')))
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    heard = [r for r in tx.p4_probe.heard if r['kind'] == Kind.NORMAL and 'status' in r]
    assert heard and abs(heard[0]['onset_sample']-1427751) <= 48
    assert heard[0]['status'] == 1
    assert bytes.fromhex(heard[0]['data']).startswith(b'91 daily minutes remaining')
    keyed = [r for r in tx.p4_probe.link_events if r['event'] == 'Keyed']
    for r in heard:
        assert any(e['at'] == r['onset_sample']+round(offset*FS) for e in keyed)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Decision' and e['missed_slots']]


@pytest.mark.parametrize('offset', [.900, .920, .940])
def test_measured_late_peer_ack_settles_bytes_at_every_own_offset(offset):
    import numpy as np
    from hfmodem.tests.shrike import p4_stock_peer as stock
    from hfmodem.shrike.p4turn import State
    m, queue = machine(data='body', ladder='none', answer_offset=offset)
    queue.on_host_data(b'outgoing')
    m._data(0, State.ISS_DATA_ACK)
    m.keyed()
    r = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=0.)
    r.enable_hold()
    r.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8))
    audio = np.zeros(round(1.24*FS))
    wave = stock.word('X27')
    at = round(.950*FS)
    audio[at:at+len(wave)] = wave
    regs = r.read_window(audio, 0, whole=True)
    ack = next(reg for reg in regs if reg.observation.word == 'X27')
    assert ack.responds_to == 0 and not ack.observation.provisional
    m.observe(ack, now=len(audio))
    assert not m.outbox.inflight and queue._buffer_raw == 0
    assert m.counter == 1
