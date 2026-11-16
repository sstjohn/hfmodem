# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p4frame, p4oracle, p4turn, session
from hfmodem.shrike.p4host import P4Host, P4Outbox
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Clock, Expect
from hfmodem.shrike.p4turn import P4Turn, Proposal, State
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.evidence import CAPTURES
from hfmodem.tests.shrike.test_p4turn import IO
from hfmodem.tests.shrike import p4_stock_peer as stock

ROOT = Path(__file__).resolve().parents[5]
CAPTURE = CAPTURES/'onair-0929-1024/stream.flac'
LEAD = 2132
# 1024: VE3KPG's X27 to our v13 at 98.149 began at 99.099, +0.950.
RECORDED_X27 = round(99.099*FS)


def recorded_x27():
    if not CAPTURE.exists():
        pytest.skip('local September29 1024 capture absent')
    audio = session.load_wav(str(CAPTURE), FS)
    from hfmodem.shrike import p4oracle
    piece = audio[RECORDED_X27-round(.020*FS):RECORDED_X27+round(.270*FS)]
    head = p4oracle.word_at(piece, FS, 0, .020, cfo_hz=5.-3.65, provisional=True,
                            expected_words={'X27'})
    assert head is not None and head.word == 'X27'
    # Place the measured head inside the inclusive edge by one baseband sample.
    return piece, head.first_chip+round(FS/28800)


def synthetic_x27():
    return stock.word('X27'), 0


def ladder_step(monkeypatch, wave, head, tau, cost_ms):
    audio = np.zeros(5*FS, np.float32)
    at = round(tau*FS)-head
    audio[at:at+len(wave)] = wave
    stream = Stream(audio)
    stream.wall = 0
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.key_notice = 1652
    stream.transmit = True
    stream.clamp_late = lambda t: max(0, stream.sample_now()+stream.key_notice-t)
    tx = Tx(stream)
    tx.p4_probe.emissions[0].update(value=7, header='A')
    tx.p4_probe.values = (7, 38)
    tx.p4_link_options.p4_link_timeout = 4.
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: LEAD)

    def costly(read):
        def run(*args, **kwargs):
            result = read(*args, **kwargs)
            if stream.pos < 60000:
                stream.wall = stream.sample_now()+round(cost_ms/1000*FS)
            return result
        return run

    for name in ('read_partial', 'read_window'):
        monkeypatch.setattr(tx.sessrx.p4, name, costly(getattr(tx.sessrx.p4, name)))
    key = tx._tx

    def keying(wave, what):
        key(wave, what)
        stream.pos = stream.wall = tx.tx_end

    tx._tx = keying
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return tx


@pytest.mark.parametrize('source', [synthetic_x27, recorded_x27])
@pytest.mark.parametrize('tau', [.950, .970])
@pytest.mark.parametrize('cost_ms,at', [(0, 60000), (40, 60000), (70, 120000)])
def test_late_word_steps_on_the_raster_its_budget_reaches(monkeypatch, source, tau, cost_ms, at):
    tx = ladder_step(monkeypatch, *source(), tau, cost_ms)
    emitted = tx.p4_probe.emissions[1:]
    assert emitted[0]['value'] == 38
    assert emitted[0]['audio_start'] == at
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']
    decision = next(e for e in tx.p4_probe.link_events if e['event'] == 'Decision')
    assert decision['registrations'] >= 1


def replay_1024(monkeypatch, first, costly_onsets, cost_ms=60, seconds=8.):
    if not CAPTURE.exists():
        pytest.skip('local September29 1024 capture absent')
    audio = session.load_wav(str(CAPTURE), FS)
    at = round(first*FS)

    class ClockedStream(Stream):
        now = 0

        def sample_now(self):
            return max(self.now, self.pos)

    live = ClockedStream(audio)
    live.key_notice = 1796
    live.clamp_late = lambda at: max(0, live.sample_now()+live.key_notice-at)
    live.pos = at-round(.6*FS)
    tx = Tx(live)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = seconds
    tx.p4_link_options.p4_data = 'body'
    received = []
    tx.p4_link_host.deliver = received.append
    rx = tx.sessrx.p4
    read_window = rx.read_window
    costly = [round(t*FS) for t in costly_onsets]

    def slow(audio, origin, **kw):
        out = read_window(audio, origin, **kw)
        due = rx.clock.next_due
        if (origin+len(audio) >= due+round(3.28*FS)
                and any(abs(due-t) < FS//10 for t in costly)):
            live.now = max(live.now, live.pos)+round(cost_ms*FS/1000)
        return out

    monkeypatch.setattr(rx, 'read_window', slow)
    rx.cfo_hz = 3.03
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, 3.03)
    tx.p4_probe.handoff_audio = at-6*FS, audio[at-6*FS:live.pos].copy()
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **{**kw, 'role': 'irs'}))
    key = Tx._tx

    def keying(wave, what):
        key(tx, wave, what)
        live.pos = tx.tx_end

    tx._tx = keying
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    events = [e for e in tx.p4_probe.link_events if e['event'] in (
        'answer-deferred', 'deferred-answer-cancelled', 'Refused')]
    return tx, received, events


def full_long_at(tx, onset):
    rows = [h for h in tx.p4_probe.heard if h['kind'] == Kind.LONG
            and not h['provisional'] and 'status' in h
            and abs(h['onset_sample']-round(onset*FS)) < FS//10]
    assert len(rows) == 1
    return rows[0]


# The live link bounds LONG work by the wall clock, so a loaded host changes what it decides.
@pytest.mark.realtime
@pytest.mark.parametrize('costly', [(63.499,), (63.499, 67.249)])
def test_1024_prompt_long_decoded_late_is_answered_once_on_its_repeat(monkeypatch, costly):
    tx, received, events = replay_1024(monkeypatch, 63.499, costly)
    # Host acceptance checks the deadline after
    # receiver publication. The injected cost precedes turn/host admission,
    # so an expired first copy cannot create delivered or held-answer credit.
    # A timely complete repeat has its own deadline and source receipt (:172).
    decisions = [e for e in tx.p4_probe.link_events if e['event'] == 'Decision']
    assert len(decisions) == 2
    assert all(e['nominal_due'] == e['at']+round(3.4*FS) for e in decisions)
    assert any(e['event'] == 'long-engine' and e['crc_valid'] and e['timely']
               and abs(e['at']-round(63.499*FS)) < FS//10
               for e in tx.p4_probe.link_events)
    accepted = [e for e in tx.p4_probe.link_events if e['event'] == 'source-accept']
    expected = b'[WL2K-5.0-B2FWIHJM$]\r;PQ: 25286779\rCMS via VE3KPG >\r'
    if len(costly) == 1:
        repeat = full_long_at(tx, 67.249)
        at = repeat['onset_sample']
        assert [(e['word'], e['audio_start']) for e in tx.p4_probe.emissions] == [
            ('X27', at+round(3.4*FS))]
        keyed, = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
        assert keyed['purpose'] is None
        assert keyed['accepted_target'] == keyed['current_slot'] == at
        assert keyed['current_slot_accepted']
        assert keyed['ack_identity'] == (3, expected.ljust(69, b'\x1e').hex())
        assert len(accepted) == 1 and accepted[0]['at'] == at
        assert [e['event'] for e in events] == ['Refused']
        assert events[0]['at'] == decisions[0]['nominal_due']
        assert received == [expected]
    else:
        assert tx.p4_probe.emissions == []
        assert [e['event'] for e in events] == ['Refused', 'Refused']
        assert [e['at'] for e in events] == [e['nominal_due'] for e in decisions]
        assert not accepted and received == []


# The live link bounds LONG work by the wall clock, so a loaded host changes what it decides.
@pytest.mark.realtime
def test_1024_idle_long_decoded_late_keys_its_policy_answer_over_the_unread_repeat(monkeypatch):
    tx, received, events = replay_1024(monkeypatch, 70.999, (70.999,), seconds=8.5)
    # Design :98,170,203 forbids acceptance of the first expired copy. Full
    # coded-region agreement (:172) validates its repeat on that copy's own
    # deadline, so this is a current-slot T27 rather than a held expired ACK.
    assert [e['event'] for e in events] == ['Refused']
    repeat = full_long_at(tx, 74.749)
    at = repeat['onset_sample']
    assert repeat['status'] == 0x40 and repeat['data'] == (b'\x1e'*69).hex()
    assert any(e['event'] == 'long-engine' and e['state'] == 'repeat'
               and e['crc_valid'] and e['timely']
               and abs(e['at']-at) < FS//10 for e in tx.p4_probe.link_events)
    accepted = [e for e in tx.p4_probe.link_events if e['event'] == 'source-accept']
    assert len(accepted) == 1 and accepted[0]['at'] == at
    answer, = tx.p4_probe.emissions
    # The former held answer over an unread repeat keyed at 3751140. The
    # validated current repeat refines its own physical onset by two samples;
    # pin both absolute coordinates independently of the observed onset.
    assert (answer['word'], answer['status'], answer['payload'], answer['audio_start']) == (
        'X27', 0, '', 3751138)
    assert at == 3587938
    keyed, = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert keyed['state'] == 'IRS_ANSWER'
    assert keyed['accepted_target'] == keyed['current_slot'] == at
    assert keyed['current_slot_accepted']
    assert keyed['ack_identity'] == (0x40, (b'\x1e'*69).hex())
    assert received == []
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'turn']
    assert [e['purpose'] for e in tx.p4_probe.link_events if e['event'] == 'Keyed'][:1] == [None]


def test_1024_idle_repeat_complete_support_onset_coordinates():
    if not CAPTURE.exists():
        pytest.skip('local September29 1024 capture absent')
    source = session.load_wav(str(CAPTURE), FS)
    predicted = 3587940
    origin = predicted-24000
    audio = source[origin:predicted+round(3.355*FS)]
    prefix = audio[:24000+round(3.230*FS)]
    prepared = p4oracle.prepare_despread_long(prefix, FS, .5, 3.03)
    assert prepared is not None
    starts = {}
    for count in (369, 373):
        _, symbols, first, reason = p4oracle.finish_despread_long(
            audio, FS, prepared, count=count)
        assert reason == 'prepared' and len(symbols) == count
        starts[count] = origin+first
    # The old held answer used the predicted phase, not either body fit.
    # Keep the broader legacy support beside the live locked-support case.
    assert starts == {369: 3587938, 373: 3586658}



def irs():
    a = arq.PactorArq(IO())
    a.state = arq.State.CONNECTED
    delivered = []
    host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    m = P4Turn(P4Outbox(a), P4Host(host), data='body', role='irs', prekey_lead=LEAD/FS)
    return m, delivered


def long(at, data=b'prompt >\r', status=3, readable=True):
    frame = p4frame.decode_field(p4frame.encode_field(
        data.ljust(69, b'\x1e'), p4frame.Status.from_byte(status)), data_bytes=69)
    return RegisteredObservation(Observation(Kind.LONG, FS, 0, at, at, 3.28, 0, .9,
                                             frame=frame if readable else None), Registered.NEW)


T_LONG = round(3.4*FS)
CYCLE_LONG = round(3.75*FS)


def late_prompt(m):
    m.observe(long(FS), now=FS+T_LONG-LEAD+1)
    answer = m.decide(FS+T_LONG-LEAD+1)
    assert (answer.word, answer.stream_sample, answer.purpose) == ('X27', FS+T_LONG, None)
    assert m._deferred is None
    m.refused('deadline')
    assert m._deferred is not None and m._ack_chain is None
    m.silence(FS+T_LONG-LEAD)
    assert m._next is None
    return FS+T_LONG+CYCLE_LONG


def test_unread_repeat_keeps_the_deferred_answer():
    m, delivered = irs()
    m.observe(long(FS, readable=False))
    request = m.decide(FS+T_LONG-LEAD)
    assert request.word == 'V27' and request.purpose == 'repeat-request'
    m.refused('deadline')
    instant = FS+T_LONG+CYCLE_LONG
    repeat = FS+CYCLE_LONG
    m.observe(long(repeat, readable=False), now=repeat+round(3.3*FS))
    proposal = m.decide(repeat+T_LONG-LEAD)
    assert (proposal.word, proposal.stream_sample) == ('V27', instant)
    assert proposal.ack_identity is None and proposal.purpose == 'repeat-request'
    m.keyed()
    assert delivered == []
    assert m._last_ack is None and m._ack_chain is None


@pytest.mark.parametrize('late', [False, True])
def test_readable_repeat_consumes_the_deferred_answer_once(late):
    m, delivered = irs()
    instant = late_prompt(m)
    repeat = FS+CYCLE_LONG
    m.observe(long(repeat), now=repeat+T_LONG-LEAD+(1 if late else -FS//10))
    proposal = m.decide(repeat+T_LONG-LEAD)
    assert (proposal.word, proposal.stream_sample) == ('X27', instant)
    m.keyed()
    assert m.decide(repeat+T_LONG+CYCLE_LONG) is None
    assert delivered == [b'prompt >\r']
    assert [e['event'] for e in m.link_events
            if e['event'] not in ('source-accept', 'source-replay', 'source-unclassified', 'long-answer')] == ['answer-deferred']


def test_different_packet_cancels_the_deferred_answer():
    m, delivered = irs()
    late_prompt(m)
    after = FS+CYCLE_LONG
    m.observe(long(after, b'next line\r', status=0), now=after+round(3.3*FS))
    proposal = m.decide(after+T_LONG-LEAD)
    assert (proposal.word, proposal.stream_sample) == ('X27', after+T_LONG)
    assert m._next.ack_identity[0] == 0
    events = [e for e in m.link_events if e['event'] not in ('source-accept', 'source-replay', 'source-unclassified', 'long-answer')]
    assert [e['event'] for e in events] == ['answer-deferred', 'deferred-answer-cancelled']
    assert events[-1]['reason'] == 'different packet'
    assert delivered == [b'prompt >\r', b'next line\r']


@pytest.mark.parametrize('change', ['word', 'our key', 'link down'])
def test_mode_role_or_link_change_cancels_the_deferred_answer(change):
    m, _ = irs()
    instant = late_prompt(m)
    if change == 'word':
        m.observe(RegisteredObservation(Observation(Kind.WORD, FS, 0, FS+CYCLE_LONG, FS+CYCLE_LONG,
                                                    .24, 0, .9, word='Y27', score=.99, extent_s=.8),
                                        Registered.NEW))
    elif change == 'our key':
        m._next = Proposal(Kind.WORD, FS+T_LONG+FS, word='X27', state=State.IRS_ANSWER)
        m.keyed()
    else:
        m.link_down('test')
    assert m._deferred is None
    proposal = m.decide(instant-LEAD)
    assert proposal is None or proposal.stream_sample != instant
