# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Queued BODY compatibility; observations are synthetic, registry/turn are real."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, machine, body, word
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, CYCLE, packet


def context(*, managed=False, explicit=False, data=None):
    kw = dict(role='irs', managed_application=managed)
    if explicit: kw['data'] = 'hold'
    if data is not None: kw['data'] = data
    m, a = machine(**kw)
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    rx = P4Receiver(FS, admitted=True, chirp=True, breakin=True)
    return m, a, rx, delivered


def feed(m, rx, kind, at, status=0, data=b'hello'):
    if kind == Kind.BODY:
        head = rx.register(word('Y27', at, extent=.8, responds_to=None).observation)
        m.observe(head, now=at+12000)
        r = rx.register(body(at, data, status).observation)
        assert r.outcome == Registered.ENRICHED
    else:
        width = {Kind.NORMAL:128, Kind.LONG:40, Kind.CHIRP:22}[kind]
        r = rx.register(packet(at, kind, status, data.ljust(width,b'\x1e')).observation)
    now = at + round(m.t[kind]*FS) - m.lead
    m.observe(r, now=now)
    return r, now


def key_answer(m, rx, p):
    assert m.can_key(p)
    sent = m.keyed(p.stream_sample)
    assert sent == p
    rx.note_emission(Emission(p.kind, Role.IRS, p.stream_sample, .244,
        accepted_target=p.accepted_target, current_slot=p.current_slot,
        current_slot_accepted=p.current_slot_accepted))
    rx.sync_peer_watch(m.peer_watch, receiving=True)


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('kind', [Kind.BODY, Kind.NORMAL])
def test_ordinary_hold_keyed_ack_then_queue_requests_body(explicit, managed, kind):
    m, a, rx, delivered = context(managed=managed, explicit=explicit)
    assert not m.sends_data
    first, now = feed(m,rx,kind,FS)
    p = m.decide(now)
    assert (p.kind,p.state,p.accepted_target) == (Kind.WORD,State.IRS_ANSWER,FS)
    assert m._accepted_proposal[0] is p
    key_answer(m,rx,p)
    epoch = m._receive_epoch
    a.on_host_data(b'ready')
    r, now = feed(m,rx,kind,FS+60000)
    assert not m._long_peer_open()
    assert m.pending[r.observation.stream_sample]['accepted']
    p = m.decide(now)
    assert (p.kind,p.status,p.state,p.payload) == (Kind.BODY,0x40,State.IRS_BREAKIN,b'')
    assert p.ack_identity == (r.observation.frame.status.byte,r.observation.frame.data)
    assert bytes(a._outbuf) == b'ready' and not m.outbox.inflight and a.tx_acked_bytes == 0
    assert m._receive_epoch == epoch and delivered == [b'hello']
    m.keyed(p.stream_sample)
    assert m.breakins == 1 and not m.grants and a.tx_acked_bytes == 0


@pytest.mark.parametrize('kind,status,expected', [(Kind.LONG,0x42,'X27'),(Kind.CHIRP,0x41,'X27')])
@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_constructor_hold_long_phase_survives_empty_body(kind,status,expected,managed,explicit):
    m,a,rx,delivered = context(managed=managed,explicit=explicit)
    a.on_host_data(b'ready')
    r,now = feed(m,rx,kind,AT,status,b'peer\r')
    p=m.decide(now)
    assert (p.kind,p.word,p.state,p.stream_sample)==(Kind.WORD,expected,State.IRS_ANSWER,KEY)
    key_answer(m,rx,p)
    phase,held,epoch=m.peer_phase,m._held_answer,m._receive_epoch
    assert m._long_peer_open() and not m.sends_data
    at=KEY+60000
    r,now=feed(m,rx,Kind.BODY,at,0x40,b'')
    assert m.peer_phase is phase and r.responds_to is None
    p=m.decide(now)
    assert (p.kind,p.word,p.state,p.stream_sample)==(Kind.WORD,expected,State.IRS_ANSWER,KEY+CYCLE)
    assert p.accepted_target==held.onset and not p.current_slot_accepted
    assert p.receive_epoch==held.answer.receive_epoch and p.source_root==held.answer.source_root
    key_answer(m,rx,p)
    assert m.peer_phase is phase and m._receive_epoch==epoch
    assert m.peer_watch.attempts==0 and m.breakins==m.grants==0
    assert bytes(a._outbuf)==b'ready' and not m.outbox.inflight and a.tx_acked_bytes==0
    assert delivered==[b'peer\r'] and not any(e['event']=='turn' for e in m.link_events)


@pytest.mark.parametrize('kind', [Kind.LONG,Kind.CHIRP])
@pytest.mark.parametrize('managed', [False,True])
def test_real_refinement_retains_long_gate_then_normal_replaces_it(kind,managed):
    m,a,rx,delivered=context(managed=managed,explicit=True)
    status=0x42 if kind==Kind.LONG else 0x41
    o=packet(AT,kind,status,b'peer\r'.ljust(40 if kind==Kind.LONG else 22,b'\x1e')).observation
    header=rx.register(replace(o,frame=None,provisional=True))
    m.observe(header,now=AT+24000)
    full=rx.register(replace(o,stream_sample=AT-24))
    assert full.replaced is header.observation
    now=AT-24+163200-m.lead
    m.observe(full,now=now)
    p=m.decide(now)
    key_answer(m,rx,p)
    assert m.peer_phase.onset==AT-24 and m._long_peer_open()
    a.on_host_data(b'ready')
    phase=m.peer_phase
    _,now=feed(m,rx,Kind.BODY,KEY+60000,0,b'')
    p=m.decide(now)
    assert p.kind==Kind.WORD and p.state==State.IRS_ANSWER and m.peer_phase is phase
    key_answer(m,rx,p)
    at=AT+2*CYCLE
    r,now=feed(m,rx,Kind.NORMAL,at,0,b'')
    assert m.peer_phase.kind==Kind.NORMAL and m.peer_phase.onset==at
    assert m.peer_watch is None and not m._long_peer_open()
    p=m.decide(now)
    assert (p.kind,p.status,p.state)==(Kind.BODY,0x40,State.IRS_BREAKIN)
    assert not m.outbox.inflight and a.tx_acked_bytes==0 and delivered==[b'peer\r']


@pytest.mark.parametrize('control', ['empty','gated','inflight','terminal','inflight_terminal','close'])
@pytest.mark.parametrize('kind', [Kind.BODY,Kind.NORMAL])
def test_ordinary_queue_exclusions_keep_accepted_answer(control,kind):
    m,a,rx,delivered=context(explicit=True)
    _,now=feed(m,rx,kind,FS)
    key_answer(m,rx,m.decide(now))
    if control not in ('empty','terminal'):
        a.on_host_data(b'ready')
    if control in ('inflight','inflight_terminal'):
        assert m.outbox.transfer(31)==b'ready' and not m.outbox.queued
    if control=='gated': m.allow_breakin=False
    if control in ('terminal','inflight_terminal'): m.app_done=True
    if control=='close':
        m._body(FS+60000,0x20,State.ISS_CLOSE)
        ctx=m.close_context
        m.refused('semantic close refusal')
        assert m.state==State.IRS_ANSWER and m.close_context is ctx
    r,now=feed(m,rx,kind,FS+60000)
    p=m.decide(now)
    assert (p.kind,p.state,p.accepted_target)==(Kind.WORD,State.IRS_ANSWER,FS+60000)
    assert p.current_slot_accepted and m._accepted_proposal[0] is p
    assert m.breakins==m.grants==0 and a.tx_acked_bytes==0
    assert delivered==[b'hello']
    if control=='close': assert m.close_context is ctx


def test_real_data_reply_retains_reply_grant_with_new_queue():
    m,a,rx,delivered=context(data='body')
    a.on_host_data(b'sent')
    m._data(FS)
    sent=m.keyed(FS)
    rx.note_emission(Emission(sent.kind,Role.ISS,FS,.800))
    a.on_host_data(b'ready')
    at=FS+44880
    r,now=feed(m,rx,Kind.BODY,at,0x40,b'')
    assert r.responds_to==FS and m.pending[at]['data_reply']
    assert a.tx_acked_bytes==4 and bytes(a._outbuf)==b'ready' and not m.outbox.inflight
    p=m.decide(now)
    assert (p.kind,p.status,p.state,p.payload)==(Kind.BODY,0x40,State.ISS_GRANT,b'')
    assert not m.breakins and bytes(a._outbuf)==b'ready'


@pytest.mark.parametrize('kind', [Kind.LONG,Kind.CHIRP])
@pytest.mark.parametrize('managed', [False,True])
@pytest.mark.parametrize('terminal', [False,True])
def test_long_hold_inflight_or_terminal_does_not_become_queued_data(kind,managed,terminal):
    m,a,rx,delivered=context(managed=managed,explicit=True)
    a.on_host_data(b'ready')
    assert m.outbox.transfer(31)==b'ready'
    # Semantic terminal intent without invoking its distinct close-request route.
    m.app_done=terminal
    _,now=feed(m,rx,kind,AT,0x42 if kind==Kind.LONG else 0x41,b'peer\r')
    p=m.decide(now)
    assert (p.kind,p.word,p.state)==(Kind.WORD,'X27',State.IRS_ANSWER)
    assert p.accepted_target==AT and p.current_slot_accepted
    assert not m.outbox.queued and m.outbox.inflight==b'ready'
    assert not m.sends_data and m.breakins==m.grants==0 and a.tx_acked_bytes==0
    assert delivered==[b'peer\r']


@pytest.mark.parametrize('kind', [Kind.LONG,Kind.CHIRP])
@pytest.mark.parametrize('managed', [False,True])
def test_long_sends_data_inflight_recovery_keeps_allow_breakin_scope(kind,managed):
    m,a,rx,delivered=context(managed=managed,data='body')
    a.on_host_data(b'ready')
    assert m.outbox.transfer(31)==b'ready'
    m.allow_breakin=False
    _,now=feed(m,rx,kind,AT,0x42 if kind==Kind.LONG else 0x41,b'peer\r')
    p=m.decide(now)
    assert (p.kind,p.word,p.state,p.stream_sample)==(Kind.WORD,'Y27',State.IRS_BREAKIN,KEY)
    assert p.peer_watch_generation==m.peer_watch.generation and m.can_key(p)
    m.keyed(KEY)
    assert m.peer_watch.attempts==1 and m.breakins==1 and m.grants==0
    assert m.outbox.inflight==b'ready' and not m.outbox.queued and a.tx_acked_bytes==0
