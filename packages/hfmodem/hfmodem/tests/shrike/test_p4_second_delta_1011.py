# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Second-delta semantic controls. All observations here are synthetic."""
from dataclasses import replace, asdict
from types import SimpleNamespace
import pytest
from hfmodem.shrike import onair
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Emission, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, CYCLE, FS, setup, packet
from hfmodem.tests.shrike.second_delta_support_1011 import record

def test_d1_empty_cooldown_slot_uses_physical_due():
    m,a = setup(b'TEST\r')
    m.observe(packet(),now=AT+158400)
    for due in (KEY,KEY+CYCLE,KEY+2*CYCLE,KEY+3*CYCLE):
        m.decide(due-m.lead)
        m.keyed(due)
        m.silence(due+60000)
    m.keyed(8911870)
    assert m._next is None and m._accepted_proposal is None
    pending = [s for s in m.pending if s not in m._decided]
    assert not pending and m._deferred is None
    assert m.peer_watch.attempts == 4 and m.next_offer_at == 9691870
    now, due = 9091870-m.lead, 9091870
    m.reschedule_breakin(now)
    actual = onair._p4_link_due(m, pending, due-163200, 3.400, now, m.lead, False)
    record('d1_first_slot', due=actual, selected=asdict(m._next) if m._next else None)
    assert actual == due, 'current held-answer due must precede future cooldown acquisition'
    p = m.decide(now)
    if p is None:
        m.silence(now, answer_due=actual)
        p = m.decide(now)
    assert (p.word, p.state, p.stream_sample) == ('X27', State.IRS_ANSWER, due)
    assert m.can_key(p)
    m.keyed(due)
    for due in (9271870,9451870,9631870):
        m.reschedule_breakin(due-m.lead)
        p = m.decide(due-m.lead)
        if p is None:
            m.silence(due-m.lead, answer_due=due)
            p = m.decide(due-m.lead)
        assert p.word == 'X27' and m.next_offer_at == 9691870 and m.peer_watch.attempts == 4
        m.keyed(due)
    m.reschedule_breakin(9811870-m.lead)
    p = m.decide(9811870-m.lead)
    assert p.word == 'Y27' and p.stream_sample == 9811870
    assert m.next_offer_at == 9691870 and m.peer_watch.attempts == 4
    m.keyed(9811870)
    assert m.next_offer_at is None and m.peer_watch.attempts == 1 and a.tx_acked_bytes == 0

def emit(rx, p):
    rx.note_emission(Emission(p.kind, Role.IRS, p.stream_sample, .245,
        accepted_target=p.accepted_target, current_slot=p.current_slot,
        current_slot_accepted=p.current_slot_accepted))

@pytest.mark.parametrize('kind,status,weak,policy', [
    (Kind.CHIRP,0x61,False,'fixed-x27'),
    (Kind.NORMAL,0x01,True,'fixed-x27'),
    (Kind.NORMAL,0x01,False,'fixed-x27'),
    (Kind.LONG,0x01,False,'fixed-x27'),
    (Kind.LONG,0x01,False,'counter-odd-t27'),
    (Kind.LONG,0x02,False,'counter-even-t27')])
@pytest.mark.parametrize('advance,refuse', [(False,False),(True,False),(False,True),(True,True)])
def test_d2_selected_repeat_binds_current_context(kind,status,weak,policy,advance,refuse):
    m, a = setup(long_ack=policy)
    m.managed_application = False
    rx = P4Receiver(FS, admitted=True, chirp=True)
    width = {Kind.CHIRP:22,Kind.LONG:40,Kind.NORMAL:128}[kind]
    data = b'peer source\r'.ljust(width,b'\x1e')
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75 if kind != Kind.NORMAL else 1.25, -.19)
    first = rx.register(packet(AT,kind,status,data).observation)
    m.observe(first,now=AT+round(m.t[kind]*FS)-m.lead)
    p = m.decide(AT+round(m.t[kind]*FS)-m.lead)
    assert p and p.kind == Kind.WORD and p.state == State.IRS_ANSWER
    m.keyed(p.stream_sample)
    emit(rx,p)
    old = m._last_ack
    at = rx.clock.next_due
    # A real lifecycle operation, explicitly semantic rather than an on-air claim.
    if advance:
        at += 2400  # Independent new epoch source, outside the old emission's association window.
        m._open_receive_epoch(at)
        assert m._last_ack is old
    obs = packet(at,kind,status,data).observation
    # A LONG/CHIRP envelope is a complete frame-less classification, allowing
    # its checked refinement under the real registry's existing rules.
    header = rx.register(replace(obs,frame=None,provisional=kind == Kind.NORMAL))
    m.observe(header,now=at+12000)
    repeat = rx.register(replace(obs,weak=weak))
    m.observe(repeat,now=at+round(m.t[kind]*FS)-m.lead)
    pending = m.pending[at]
    assert pending['accepted'] and pending['body'] is repeat.observation
    assert ('answer' not in pending) if weak else pending.get('answer') is not None
    due = at+round(m.t[kind]*FS)
    p = m.decide(due-m.lead)
    assert any(e['event']=='acked-repeat' and e['at']==at for e in m.link_events)
    record('d2_binding', kind=kind, weak_synthetic=weak, advance=advance, refuse=refuse,
           proposal=asdict(p), epoch=m._receive_epoch, pending_epoch=pending['receive_epoch'],
           pending_root=pending['source_root'], delivered=[b.hex() for b in delivered],
           answer_kinds=str(m._answer_kinds), accepted_context=str(m._accepted_proposal))
    assert p.receive_epoch == pending['receive_epoch'], 'selected repeat owns current receive epoch'
    assert p.source_root == pending['source_root']
    assert due in m._answer_kinds
    assert p.ack_identity == (status,data)
    assert p.state == State.IRS_ANSWER and p.accepted_target == at, 'selected repeat owns new accepted target'
    assert p.current_slot == at and p.current_slot_accepted
    assert m._accepted_proposal[0] is p and m._accepted_proposal[1] == at
    assert m._accepted_proposal[2]['body'] is repeat.observation
    if refuse:
        m.refused('deadline')
        assert m._deferred and m._deferred[0] == at
        due += round((1.25 if kind == Kind.NORMAL else 3.75)*FS)
        p = m.decide(due-m.lead)
        assert p.accepted_target == at and not p.current_slot_accepted
    assert m.can_key(p)
    sent = m.keyed(p.stream_sample)
    emit(rx,sent)
    assert sent.receive_epoch == m._receive_epoch
    record('d2_emission', current_slot=sent.current_slot, answered=rx._answered,
           answered_generations=rx._answered_generations,
           provisional_remaining=[o.stream_sample for o in rx.observations if o.provisional])
    assert delivered == [b'peer source\r'] and a.tx_acked_bytes == 0 and not m.outbox.inflight
