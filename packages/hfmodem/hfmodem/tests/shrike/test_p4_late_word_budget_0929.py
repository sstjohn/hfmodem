# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p4frame, p4turn, session
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


def deferred_instant(tx, onset):
    heard = next(h['onset_sample'] for h in tx.p4_probe.heard if not h['provisional']
                 and abs(h['onset_sample']-round(onset*FS)) < FS//10)
    return heard+round(3.4*FS)+round(3.75*FS)


@pytest.mark.parametrize('costly', [(63.499,), (63.499, 67.249)])
def test_1024_prompt_long_decoded_late_is_answered_once_on_its_repeat(monkeypatch, costly):
    tx, received, events = replay_1024(monkeypatch, 63.499, costly)
    if len(costly) == 1:
        assert [(e['word'], e['audio_start']) for e in tx.p4_probe.emissions] == [
            ('X27', deferred_instant(tx, 63.499))]
        assert [e['purpose'] for e in tx.p4_probe.link_events if e['event'] == 'Keyed'] == [None]
        assert [e['event'] for e in events] == ['Refused', 'answer-deferred']
    else:
        assert tx.p4_probe.emissions == []
        assert len([e for e in events if e['event'] == 'Refused']) == 2
        assert [e['at'] for e in events if e['event'] == 'Refused'] == [
            deferred_instant(tx, 63.499)-CYCLE_LONG, deferred_instant(tx, 63.499)]
        assert len([e for e in events if e['event'] == 'answer-deferred']) == 1
    assert len(received) == 1 and received[0].endswith(b'CMS via VE3KPG >\r')


def test_1024_idle_long_decoded_late_keys_its_grant_over_the_unread_repeat(monkeypatch):
    tx, received, events = replay_1024(monkeypatch, 70.999, (70.999,), seconds=8.5)
    assert [e['event'] for e in events] == ['Refused', 'answer-deferred']
    assert not any(e['event'] == 'long decode' and e.get('crc_valid')
                   and abs(e['at']-round(74.749*FS)) < FS//10 for e in tx.p4_probe.link_events)
    grant = tx.p4_probe.emissions[0]
    assert (grant['status'], grant['audio_start']) == (0x40, deferred_instant(tx, 70.999))
    assert [e['purpose'] for e in tx.p4_probe.link_events if e['event'] == 'Keyed'][:1] == [None]



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
    assert m._deferred is not None and m.ack_word is None
    m.silence(FS+T_LONG-LEAD)
    assert m._next is None
    return FS+T_LONG+CYCLE_LONG


def test_unread_repeat_keeps_the_deferred_answer():
    m, delivered = irs()
    instant = late_prompt(m)
    repeat = FS+CYCLE_LONG
    m.observe(long(repeat, readable=False), now=repeat+round(3.3*FS))
    proposal = m.decide(repeat+T_LONG-LEAD)
    assert (proposal.word, proposal.stream_sample) == ('X27', instant)
    m.keyed()
    assert delivered == [b'prompt >\r']
    assert m._last_ack[0] == (3, b'prompt >\r'.ljust(69, b'\x1e'))


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
            if e['event'] not in ('source-accept', 'source-replay', 'source-unclassified')] == ['answer-deferred']


def test_different_packet_cancels_the_deferred_answer():
    m, delivered = irs()
    late_prompt(m)
    after = FS+CYCLE_LONG
    m.observe(long(after, b'next line\r', status=0), now=after+round(3.3*FS))
    proposal = m.decide(after+T_LONG-LEAD)
    assert (proposal.word, proposal.stream_sample) == ('X27', after+T_LONG)
    assert m._next.ack_identity[0] == 0
    events = [e for e in m.link_events if e['event'] not in ('source-accept', 'source-replay', 'source-unclassified')]
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
