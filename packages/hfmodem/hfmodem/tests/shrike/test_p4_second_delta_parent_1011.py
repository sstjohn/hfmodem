# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Identical ordinary scenarios executed against real isolated production revisions.

Semantic observations are synthetic, with genuine field encode/decode/register.
Extra watch metadata is reported outside the exact ordinary comparison schema.
"""
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from types import SimpleNamespace
import pytest
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Emission, Role
from hfmodem.shrike.p4hold import BreakinHold
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, CYCLE, FS, setup, packet
from hfmodem.tests.shrike.second_delta_support_1011 import record

WATCH_METADATA = frozenset(('peer_watch_generation','primary_packet_due','watch_recovery'))

class Recorder:
    def __init__(self,m,rx,a):
        self.m,self.rx,self.a = m,rx,a
        self.refs,self.keep,self.rows,self.delivered = {},[],[],[]

    def value(self,v):
        if v is None or isinstance(v,(str,int,float,bool)): return v
        if isinstance(v,bytes): return {'bytes':v.hex()}
        if isinstance(v,Enum): return v.value
        if isinstance(v,(tuple,list)): return [self.value(x) for x in v]
        if isinstance(v,(set,frozenset)): return sorted((self.value(x) for x in v),key=str)
        if isinstance(v,dict): return {str(k):self.value(x) for k,x in sorted(v.items(),key=lambda p:str(p[0])) if k not in WATCH_METADATA}
        ident=id(v)
        if ident in self.refs: return {'ref':self.refs[ident]}
        ref=len(self.refs)+1
        self.refs[ident]=ref; self.keep.append(v)
        if is_dataclass(v):
            return {'id':ref,'type':type(v).__name__,'fields':{f.name:self.value(getattr(v,f.name)) for f in fields(v) if f.name not in WATCH_METADATA}}
        return {'id':ref,'type':type(v).__name__}

    def snapshot(self,label,reg=None):
        m,rx,a=self.m,self.rx,self.a
        names=('state','_receive_active','_receive_epoch','_receive_boundary','_last_source_onset',
               '_accepted_roots','_root_epoch','pending','_decided','_dropped','unread','_next',
               '_accepted_proposal','_deferred','_last_ack','_held_answer','_answer_kinds','acked_counter',
               'ack_word','_peer_generation','_peer_consumed')
        rnames=('clock','_responds_to','_response_evidence','_answered','_answered_generations',
                'turnarounds','_tau_seen','observations','emissions','peer_onset','silent_cycles',
                '_receive_generation','_train_generation')
        values={'turn':{n:getattr(m,n,None) for n in names},
                'receiver':{n:getattr(rx,n,None) for n in rnames},
                'hold':dict(unread=rx.hold.unread,epoch=rx.hold.epoch,events=rx.hold.events),
                'registry':reg,'delivered':list(self.delivered),
                'source_result':m.adapter.result,'receipt':m._receipt,
                'queued':bytes(a._outbuf),'inflight':m.outbox.inflight,'credited':a.tx_acked_bytes}
        self.rows.append(dict(label=label,values=self.value(values)))

    def deliver(self,b):
        self.snapshot('source_callback_before')
        self.delivered.append(b)
        self.snapshot('source_callback_after')

    def finish(self,case):
        record('ordinary_exact',case=case,rows=self.rows,diagnostics=dict(
               schema=sorted(WATCH_METADATA),sync_peer_watch_present=hasattr(self.rx,'sync_peer_watch'),
               peer_watch=str(getattr(self.rx,'peer_watch',None))))

@pytest.mark.parametrize('kind',[Kind.LONG,Kind.CHIRP])
@pytest.mark.parametrize('status',[0x42,0x62])
@pytest.mark.parametrize('variant',['good','crc','provisional','weak_header','weak_full','header_full','displaced_duplicates','unassociated'])
def test_c4_exact_ordinary_registry_source(kind,status,variant):
    m,a=setup()
    rx=P4Receiver(FS,admitted=True,chirp=True)
    rx.hold=BreakinHold(FS)
    rx.clock=Clock(Expect.PACKET,AT,.330,3.75,-.19)
    recorder=Recorder(m,rx,a)
    m.adapter.host=SimpleNamespace(deliver=recorder.deliver,supervisor=lambda *a:None)
    seed=rx.register(packet(AT,kind,status).observation)
    m.observe(seed,now=AT+158400)
    p=m.decide(KEY-m.lead)
    assert p and p.kind==Kind.WORD
    sent=m.keyed(KEY)
    rx.note_emission(Emission(sent.kind,Role.IRS,KEY,.245,accepted_target=sent.accepted_target,
        current_slot=sent.current_slot,current_slot_accepted=sent.current_slot_accepted))
    at=KEY+15840+(2400 if variant=='unassociated' else 0)
    # note_emission sets the primary packet clock; no test clock repair.
    assert rx.clock.next_due==AT+CYCLE and rx.clock.expects==Expect.PACKET
    if hasattr(rx,'sync_peer_watch'): rx.sync_peer_watch(m.peer_watch)
    recorder.snapshot('preceding_actual_emission')
    width=40 if kind==Kind.LONG else 22
    o=packet(at,kind,status,b'ordinary source\r'.ljust(width,b'\x1e')).observation
    observations=[]
    if variant in ('crc','provisional','weak_header'):
        observations=[replace(o,frame=None,provisional=variant=='provisional',weak=variant=='weak_header')]
    elif variant=='weak_full': observations=[replace(o,weak=True)]
    elif variant in ('header_full','displaced_duplicates'):
        header=replace(o,frame=None)
        full=replace(o,stream_sample=at+24,first_chip=at+24) if variant=='displaced_duplicates' else o
        observations=[header,full,full,full]
    else: observations=[o]
    for obs in observations:
        reg=rx.register(obs)
        if variant=='unassociated': assert reg.responds_to is None
        else: assert reg.responds_to==KEY, 'real preceding IRS emission must establish association'
        recorder.snapshot('registered',reg)
        m.observe(reg,now=at+158400)
        recorder.snapshot('observed',reg)
    recorder.snapshot('before_decide')
    current=min((s for s in m.pending if s not in m._decided),default=at)
    due=current+163200
    p=m.decide(due-m.lead)
    recorder.snapshot('after_decide')
    if p is not None and m.can_key(p):
        sent=m.keyed(p.stream_sample)
        rx.note_emission(Emission(sent.kind,Role.IRS,sent.stream_sample,.245,
            accepted_target=sent.accepted_target,current_slot=sent.current_slot,
            current_slot_accepted=sent.current_slot_accepted))
        recorder.snapshot('keyed_and_emitted')
    if variant=='good':
        assert recorder.delivered==[b'ordinary source\r']
        assert rx.turnarounds['word']==[.330]
    assert a.tx_acked_bytes==0 and not m.outbox.inflight
    recorder.finish([kind,status,variant])
