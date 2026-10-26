# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PCM driver, recorded receive and explicit semantic grant boundary controls."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from hfmodem.shrike import onair, p4turn, p4robusttx, p4shorttx
from hfmodem.shrike.p4receiver import Emission, Expect, Kind, Role
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx


@pytest.mark.parametrize('queued,return_tail,prior_data,revoke',[
    (False,False,False,False),(True,False,False,False),(False,True,False,False),
    (True,False,True,False),(False,True,False,True)])
def test_real_two_short_receive_prefix_then_normal_no_output_grant(queued,return_tail,prior_data,revoke):
    # Synthetic PCM, ordinary admitted contact. No decoded observations or
    # peer heads/phase are planted. The initial IRS WORD supplies association.
    first=48000
    audio=np.zeros(40*FS,np.float32)
    for at,wave in ((first,p4robusttx.robust_short(13,'A')),
                    (first+60000,p4robusttx.robust_short(13,'A')),
                    (first+120000+round(308*FS/28800),p4shorttx.short_packet(b'',0x40))):
        audio[at:at+len(wave)]=wave
    tail=first+7*60000
    if return_tail:
        wave=p4shorttx.short_packet(b'',0)
        at=tail+round(308*FS/28800)
        audio[at:at+len(wave)]=wave
    if prior_data:
        wave=p4robusttx.robust_word('V27')
        at=first+120000+44160+44880
        audio[at:at+len(wave)]=wave
    stream=Stream(audio)
    stream.pos=first-12000
    tx=Tx(stream);tx.settle=.025
    tx.p4_probe.emissions=[];tx.p4_probe.peer_breakin=True
    opts=tx.p4_link_options
    opts.p4_data='body';opts.p4_ladder='none';opts.p4_hold_cycles=16
    opts.p4_normal_ack='parity';opts.p4_link_timeout=35
    rx=tx.sessrx.p4
    rx.note_emission(Emission(Kind.WORD,Role.IRS,first-round(.330*FS),.242))
    machines=[];before=[];physical=[];registered=[];delivered=[];side_reads=[];revocations=[]
    tx.p4_link_host.deliver=delivered.append
    ctor=p4turn.P4Turn
    def factory(*a,**kw):
        kw['role']='irs'
        m=ctor(*a,**kw);machines.append(m)
        if prior_data:
            decide=m.decide
            injected=False
            def negative_decision(now):
                nonlocal injected
                result=decide(now)
                if not injected and m.last is not None and m.last.payload:
                    injected=True
                    # Explicit external synthetic grant AFTER real driver
                    # release and actual local payload. Never a positive prefix.
                    m._body(m.next_raster(now),0x40,p4turn.State.ISS_GRANT)
                    result=m._next
                return result
            m.decide=negative_decision
        return m
    note=rx.note_emission
    def emitted(em,**kw):
        m=machines[0]
        before.append(dict(emission=asdict(em),clock=asdict(rx.clock) if rx.clock else None,
            clock_id=id(rx.clock),peer_onset=rx.peer_onset,heads=list(rx._peer_heads),
            admitted=rx.admitted,expected=rx.expected,registry=id(rx.observations),
            generation=rx._train_generation,epoch=m._receive_epoch,
            active=m._receive_active,turn_last=asdict(m.last),
            map_member=m._emissions.get(m.last.stream_sample) is m.last))
        result=note(em,**kw)
        physical.append(em)
        # Real application data, after the first actually keyed grant.
        if queued and em.kind==Kind.BODY and m.last.state==p4turn.State.ISS_GRANT and m.grants==1:
            tx.p4_link_host.arq.on_host_data(b'queued after genuine key\r')
        return result
    rx.note_emission=emitted
    register=rx.register
    def registration(obs):
        result=register(obs)
        registered.append(dict(kind=obs.kind,onset=obs.stream_sample,value=obs.value,
            provisional=obs.provisional,weak=obs.weak,outcome=result.outcome,
            status=obs.frame.status.byte if obs.frame else None,
            responds_to=result.responds_to,heads=list(rx._peer_heads)))
        return result
    rx.register=registration
    side=rx._read_peer_normal
    def side_read(audio,origin,result):
        clock=rx.clock;owner=rx.peer_normal_grant
        fields=asdict(clock) if clock else None
        count=len(result)
        value=side(audio,origin,result)
        if len(result)>count:
            side_reads.append(dict(before=id(clock),after=id(rx.clock),fields=fields,
                after_fields=asdict(rx.clock),owner=owner,after_owner=rx.peer_normal_grant,
                origin=origin,end=origin+len(audio),onsets=[r.observation.stream_sample for r in result[count:]]))
            if revoke and not revocations and any(r.observation.stream_sample==tail for r in result[count:]):
                old=rx._train_generation
                rx._train_generation+=1
                revocations.append((old,rx._train_generation,onair._p4_grant_read_owned(machines[0],rx)))
        return value
    rx._read_peer_normal=side_read
    key=tx._tx
    def keyed(wave,what):
        result=key(wave,what)
        stream.pos=max(stream.pos,tx.tx_end)
        return result
    tx._tx=keyed
    with patch.object(p4turn,'P4Turn',factory):
        onair._run_p4_link(tx,stream,onair._MasterGrid(0,60000,0,packet_n=36000,cs_n=12000,d_max_n=4800))
    m=machines[0]
    data=dict(queued=queued,return_tail=return_tail,prior_data=prior_data,revocations=revocations,registered=registered,pre_emissions=before,side_reads=side_reads,
              emissions=[asdict(e) for e in physical],heard=tx.p4_probe.heard,
              events=tx.p4_probe.link_events,reason=tx.p4_probe.reason,
              delivered=[b.hex() for b in delivered],credited=tx.p4_link_host.arq.tx_acked_bytes)
    if os.environ.get('GRANT_ARTIFACT'):
        Path(os.environ['GRANT_ARTIFACT']+('-revoked' if revoke else '-prior-data' if prior_data else '-queued' if queued else '-tail' if return_tail else '-empty')+'.json').write_text(json.dumps(data,indent=2,default=lambda x:x.hex() if isinstance(x,bytes) else str(x))+'\n')
    shorts=[r for r in registered if r['kind']==Kind.SHORT and not r['provisional'] and not r['weak'] and r['outcome']=='confirmed']
    assert len(shorts)==2 and shorts[1]['onset']-shorts[0]['onset']==60000
    grants=[r for r in before if r['emission']['kind']==Kind.BODY and r['turn_last']['state']==p4turn.State.ISS_GRANT]
    assert grants, 'production P2 NORMAL must select and actually key the empty grant'
    origin=grants[0]
    assert origin['clock']['expects']==Expect.PACKET and origin['clock']['cycle_s']==1.25
    assert origin['admitted'] and origin['expected'] and origin['map_member']
    assert len(origin['heads'])==2 and origin['heads'][1]-origin['heads'][0]==60000
    if prior_data:
        payloads=[r for r in before if r['turn_last']['payload']]
        assert payloads and any(e['event']=='turn' for e in tx.p4_probe.link_events)
        later=[r for r in grants if r['emission']['stream_sample']>payloads[0]['emission']['stream_sample']]
        assert later and later[0]['clock']['expects']==Expect.WORD
        assert tx.p4_link_host.arq.tx_acked_bytes==0
        assert any(r['turn_last']['state']==p4turn.State.ISS_CLOSE for r in before)
        assert m._grant_receive_origin is None
        return
    assert len(grants)==4 and tx.p4_link_host.arq.tx_acked_bytes==0
    assert not m.outbox.inflight and not delivered
    assert tx.p4_probe.reason=='peer silent' and rx.silent_cycles==16
    assert not any(e.kind==Kind.SHORT or e.kind==Kind.BODY and m._emissions[e.stream_sample].status!=0x40 for e in physical)
    if return_tail:
        successful=[s for s in side_reads if tail in s['onsets']]
        assert successful and any(s['before']!=s['after'] for s in successful)
        assert all(s['owner']==s['after_owner'] and s['fields']==s['after_fields'] for s in successful)
        if revoke:
            assert revocations and revocations[0][1]==revocations[0][0]+1 and revocations[0][2] is False
            assert not any(e.kind==Kind.WORD and e.stream_sample==tail+44160 for e in physical)
        else:
            assert any(e.kind==Kind.WORD and e.stream_sample==tail+44160 for e in physical)


def test_actual_long_invitation_then_explicit_synthetic_short_f6_grant():
    from hfmodem.tests.evidence import CAPTURES, require_input
    from hfmodem.shrike.p4receiver import P4Receiver, Clock
    from hfmodem.tests.shrike.test_p4turn import machine, word
    m,a=machine(role='irs',data='body',ladder='none',prekey_lead=.025)
    a.on_host_data(b'TEST\r')
    r=P4Receiver(FS,admitted=True,chirp=True,breakin=True)
    r.enable_hold()
    at=8748327
    r.clock=Clock(Expect.PACKET,at,.330,3.75,-.19)
    r.cfo_hz=-.19
    r.expected=True
    path=require_input(CAPTURES/'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590'/'stream.flac')
    source=onair.session.load_wav(str(path),FS)
    audio=source[at-24000:at+round(3.326*FS)]
    records=r.read_window(audio,at-24000,whole=True)
    good=[reg for reg in records if reg.observation.kind==Kind.LONG and reg.observation.frame]
    assert good and all(reg.observation.frame.status.byte==0x42 and len(reg.observation.frame.data)==40 for reg in good)
    for reg in good:m.observe(reg,now=reg.observation.stream_sample+round(3.3*FS))
    p=m.decide(good[-1].observation.stream_sample+round(3.4*FS)-m.lead)
    assert p.kind==Kind.WORD and p.word=='Y27' and p.state==p4turn.State.IRS_BREAKIN
    actual=m.keyed()
    role=(Role.IRS if (actual.kind==Kind.WORD
          and actual.state!=p4turn.State.IRS_BREAKIN)
          or actual.purpose=='close-grant' else Role.ISS)
    assert role==Role.ISS
    r.note_emission(Emission(actual.kind,role,actual.stream_sample,.245))
    onair._p4_sync_grant_receive_origin(m,r)
    assert m._grant_receive_origin is None
    # Production ISS WORD replaces the received LONG's primary owner.
    assert r.clock.expects==Expect.WORD and r.clock.cycle_s==1.25
    # This control deliberately injects an external short grant after the
    # genuine LONG/Y27 episode. It is not an ordinary positive planner path.
    m._body(actual.stream_sample+60000,0x40,p4turn.State.ISS_GRANT)
    pre=asdict(r.clock)
    actual=m.keyed()
    r.note_emission(Emission(Kind.BODY,Role.ISS,actual.stream_sample,.798,packet_reply=True))
    onair._p4_sync_grant_receive_origin(m,r)
    assert m._grant_receive_origin is None
    assert pre['expects']==Expect.WORD and pre['cycle_s']==1.25
    assert r.grant_emission_context.predecessor.role==Role.ISS
    assert r.grant_emission_context.predecessor.kind==Kind.WORD
    assert m.peer_watch is None
    for _ in range(3):
        m._repeat();actual=m.keyed()
        r.note_emission(Emission(Kind.BODY,Role.ISS,actual.stream_sample,.798,packet_reply=True))
        onair._p4_sync_grant_receive_origin(m,r)
        assert m._grant_receive_origin is None
    m._repeat()
    onair._p4_sync_grant_receive_origin(m,r)
    assert m._grant_receive_origin is None
    close,ctx=m._next,m.close_context
    assert close.state==p4turn.State.ISS_CLOSE and ctx is not None
    limits=ctx.end,ctx.wall_started
    m.observe(word('V27',m.last.stream_sample+44880,responds_to=m.last.stream_sample,extent=.245),now=m.last.stream_sample+58000)
    assert m._next is close and m.close_context is ctx and (ctx.end,ctx.wall_started)==limits
    assert not m.outbox.inflight and a.tx_acked_bytes==0
    assert not any(e['event']=='turn' for e in m.link_events)
    m.decide(ctx.end)
    assert m.state==p4turn.State.DOWN and m.close_context is ctx
    assert (ctx.end,ctx.wall_started)==limits and m._grant_receive_origin is None
    assert not m.outbox.inflight and a.tx_acked_bytes==0


@pytest.mark.parametrize('bad', ['absent','word','long','no-heads','off-heads',
    'unadmitted','unexpected','off-key','unkeyed','payload','wrong-kind','wrong-role',
    'stale-generation','new-call','registry-replacement','terminal','selected-close',
    'prior-local-data'])
def test_synthetic_boundary_negatives_never_install_receive_origin(bad):
    from hfmodem.shrike.p4receiver import P4Receiver, Clock
    from hfmodem.tests.shrike.test_p4turn import machine
    m,a=machine(role='irs',data='body',ladder='none')
    r=P4Receiver(FS,admitted=True)
    if bad=='prior-local-data':
        a.on_host_data(b'prior data')
        m._data(1,p4turn.State.ISS_DATA_ACK);m.keyed()
        r.note_emission(Emission(Kind.BODY,Role.ISS,1,.798,packet_reply=True))
    # Explicit synthetic candidate matching the unchanged 1116 seed. This is
    # adversarial predicate coverage, never a claimed ordinary production prefix.
    r.expected=True;r.clock=Clock(Expect.PACKET,2*FS,.330,1.25,0.)
    r.peer_onset=2*FS;r._peer_heads=[2*FS-60000,2*FS]
    if bad=='absent':r.clock=None
    elif bad=='word':r.clock.expects=Expect.WORD
    elif bad=='long':r.clock.cycle_s=3.75
    elif bad=='no-heads':r._peer_heads=[]
    elif bad=='off-heads':r._peer_heads[0]-=481
    elif bad=='unadmitted':r.admitted=False
    elif bad=='unexpected':r.expected=False
    at=2*FS+44160+(1681 if bad=='off-key' else 0)
    m._body(at,0x40,p4turn.State.ISS_GRANT,b'payload' if bad=='payload' else b'')
    if bad!='unkeyed':m.keyed()
    r.note_emission(Emission(Kind.WORD if bad=='wrong-kind' else Kind.BODY,
        Role.IRS if bad=='wrong-role' else Role.ISS,at,.798,packet_reply=True))
    if bad=='stale-generation':r._train_generation+=1
    elif bad=='new-call':r.new_call()
    elif bad=='registry-replacement':r.observations=list(r.observations)
    elif bad=='terminal':m.finish_application(at)
    elif bad=='selected-close':m._body(at+60000,0x20,p4turn.State.ISS_CLOSE)
    onair._p4_sync_grant_receive_origin(m,r)
    assert m._grant_receive_origin is None
    assert not a.tx_acked_bytes


def semantic_receive_origin():
    """Synthetic pre-key owner fixture; consumed keys, no PCM/RF claim."""
    from hfmodem.shrike.p4receiver import P4Receiver, Clock
    from hfmodem.tests.shrike.test_p4turn import machine
    m,a=machine(role='irs',data='body',ladder='none')
    r=P4Receiver(FS,admitted=True)
    r.expected=True;r.clock=Clock(Expect.PACKET,2*FS,.330,1.25,0.)
    r.peer_onset=2*FS;r._peer_heads=[2*FS-60000,2*FS]
    m._body(2*FS+44160,0x40,p4turn.State.ISS_GRANT);m.keyed()
    r.note_emission(Emission(Kind.BODY,Role.ISS,m.last.stream_sample,.798,packet_reply=True))
    onair._p4_sync_grant_receive_origin(m,r)
    assert m._grant_receive_origin is not None
    return m,a,r


def test_semantic_neutral_word_retires_metadata_and_preserves_receive_bookkeeping():
    # Synthetic semantic owner/observation fixture, consumed production keys.
    # No recording/RF provenance, regression explanation or allocation benchmark.
    from hfmodem.tests.shrike.test_p4turn import word
    m,a,r=semantic_receive_origin()
    original=m._grant_receive_origin
    context=r.grant_emission_context
    pending=word('Y27',m.last.stream_sample+60000,partial=True).observation
    reg=r.register(pending)
    assert reg.observation is pending and pending in r.observations
    r.enable_hold()
    hold=r.hold
    hold_epoch=hold.epoch
    heads=dict(r._body_heads)
    assert heads[pending.stream_sample] is pending
    answered=dict(r._answered),dict(r._answered_generations)
    assert pending.stream_sample not in r._answered
    a.on_host_data(b'queued semantic source\r')
    assert m.outbox.transfer()==b'queued semantic source\r'
    source=m.adapter.result
    credit=m.outbox.inflight,bytes(a._outbuf),a._buffer_raw,a.tx_acked_bytes
    due=pending.stream_sample+44160
    m._next=m._held_proposal(due)
    proposal=m._next
    assert (proposal.kind,proposal.state,proposal.current_slot,
            proposal.current_slot_accepted)==(Kind.WORD,p4turn.State.IRS_ANSWER,None,False)
    assert m._grant_receive_origin is original and r.grant_emission_context is context
    actual=m.keyed()
    assert actual is m.last and m._next is None
    assert m._emissions[actual.stream_sample] is actual
    assert m._grant_receive_origin is None
    neutral=Emission(actual.kind,Role.IRS,actual.stream_sample,.245,
        accepted_target=actual.accepted_target,current_slot=actual.current_slot,
        current_slot_accepted=actual.current_slot_accepted)
    r.note_emission(neutral)
    onair._p4_sync_grant_receive_origin(m,r)
    assert r.grant_emission_context is None and m._grant_receive_origin is None
    assert r.last_emission is neutral and r.emissions[-1] is neutral
    assert r.hold is hold and hold.epoch==hold_epoch and r._body_heads==heads
    assert (r._answered,r._answered_generations)==answered
    assert pending.stream_sample not in r._answered
    assert m.adapter.result is source
    assert (m.outbox.inflight,bytes(a._outbuf),a._buffer_raw,a.tx_acked_bytes)==credit
    assert a.tx_acked_bytes==0
    # Clearing unused context leaves the actual WORD as the next predecessor.
    generation=r._train_generation
    m._body(actual.stream_sample+60000,0x40,p4turn.State.ISS_GRANT)
    candidate=m.keyed()
    r.note_emission(Emission(Kind.BODY,Role.ISS,candidate.stream_sample,.798,packet_reply=True))
    assert r.grant_emission_context.predecessor is neutral
    assert r.grant_emission_context.predecessor_generation==generation


def test_semantic_new_call_sync_retires_installed_origin_without_other_changes():
    # Synthetic lifecycle control with a real receiver reset; no RF, import
    # count, allocation or timing claim. Sync must retire the stale owner.
    m,a,r=semantic_receive_origin()
    original=m._grant_receive_origin
    registry=r.observations
    assert original.context.registry is registry
    a.on_host_data(b'queued lifecycle source\r')
    assert m.outbox.transfer()==b'queued lifecycle source\r'
    m._repeat()
    r.new_call()
    assert r.grant_emission_context is None and r.observations is not registry
    assert m._grant_receive_origin is original
    proposal,last=m._next,m.last
    assert proposal is not None and last is original.actual
    proposal_fields,last_fields=asdict(proposal),asdict(last)
    generations=r._train_generation,m._peer_generation
    epoch=m._receive_epoch,m._receive_active,m._receive_boundary
    readers=(r.clock,r.peer_watch,r._watch_read_generation,r.hold,
        r.peer_normal_pending,r.peer_normal_grant,r.packet_read)
    current_registry=r.observations
    source=m.adapter.result
    credit=m.outbox.inflight,bytes(a._outbuf),a._buffer_raw,a.tx_acked_bytes
    for _ in range(2):
        onair._p4_sync_grant_receive_origin(m,r)
        assert m._grant_receive_origin is None
        assert r.grant_emission_context is None and r.observations is current_registry
        assert m._next is proposal and m.last is last
        assert (asdict(m._next),asdict(m.last))==(proposal_fields,last_fields)
        assert (r._train_generation,m._peer_generation)==generations
        assert (m._receive_epoch,m._receive_active,m._receive_boundary)==epoch
        assert (r.clock,r.peer_watch,r._watch_read_generation,r.hold,
            r.peer_normal_pending,r.peer_normal_grant,r.packet_read)==readers
        assert m.adapter.result is source
        assert (m.outbox.inflight,bytes(a._outbuf),a._buffer_raw,a.tx_acked_bytes)==credit
        assert a.tx_acked_bytes==0


@pytest.mark.parametrize('gap,retained',[
    (60240,True), (60000-1680,True), (60000+1680,True),
    (60000-1681,False), (60000+1681,False),
    (0,False), (240,False), (1680,False)])
def test_semantic_consumed_retry_existing_bracket_and_minimum_gap(gap,retained):
    # keyed(at) consumes the selected repeat and updates the actual map. The
    # owner fixture and displacement are semantic, not synthesized RF timing.
    m,a,r=semantic_receive_origin()
    original=m._grant_receive_origin
    predecessor=r.last_emission
    m._repeat()
    proposal=m._next
    actual=m.keyed(original.actual.stream_sample+gap)
    assert actual is m.last and m._next is None and actual.repeat
    assert m._emissions[actual.stream_sample] is actual
    assert actual.stream_sample==original.actual.stream_sample+gap
    assert proposal.stream_sample==original.actual.stream_sample+60000
    r.note_emission(Emission(Kind.BODY,Role.ISS,actual.stream_sample,.798,packet_reply=True))
    context=r.grant_emission_context
    assert context.predecessor is predecessor
    assert context.predecessor_generation==original.receiver_generation
    readers=r.peer_normal_pending,r.peer_normal_grant
    onair._p4_sync_grant_receive_origin(m,r)
    if retained:
        origin=m._grant_receive_origin
        assert origin==replace(original,actual=actual,
            receiver_generation=context.generation,turn_generation=m._peer_generation)
        assert origin.context is original.context
        assert origin.receive_epoch==original.receive_epoch
    else:
        assert m._grant_receive_origin is None
    assert (r.peer_normal_pending,r.peer_normal_grant)==readers
    assert not onair._p4_grant_read_owned(m,r)
    for _ in range(2):
        m._repeat();actual=m.keyed()
        r.note_emission(Emission(Kind.BODY,Role.ISS,actual.stream_sample,.798,packet_reply=True))
        onair._p4_sync_grant_receive_origin(m,r)
        assert (m._grant_receive_origin is not None)==retained
        if retained:assert m._grant_receive_origin.context is original.context
    m._repeat()
    assert m.grants==4
    if retained:
        assert m._next is None and m.close_context is None
    else:
        assert m._next.state==p4turn.State.ISS_CLOSE and m.close_context is not None
    assert not m.outbox.inflight and a.tx_acked_bytes==0
    assert not any(e['event']=='turn' for e in m.link_events)


@pytest.mark.parametrize('bad',['receiver-generation','turn-generation','epoch',
                              'map','registry','selected-close'])
def test_semantic_shifted_retry_cannot_retain_stale_or_closing_origin(bad):
    m,a,r=semantic_receive_origin()
    original=m._grant_receive_origin
    m._repeat();actual=m.keyed(original.actual.stream_sample+60240)
    r.note_emission(Emission(Kind.BODY,Role.ISS,actual.stream_sample,.798,packet_reply=True))
    if bad=='receiver-generation':r._train_generation+=1
    elif bad=='turn-generation':m._peer_generation+=1
    elif bad=='epoch':m._receive_epoch+=1
    elif bad=='map':m._emissions[actual.stream_sample]=replace(actual)
    elif bad=='registry':r.observations=list(r.observations)
    else:m._body(actual.stream_sample+60000,0x20,p4turn.State.ISS_CLOSE)
    close,ctx=m._next,m.close_context
    readers=r.peer_normal_pending,r.peer_normal_grant
    onair._p4_sync_grant_receive_origin(m,r)
    assert m._grant_receive_origin is None
    assert (r.peer_normal_pending,r.peer_normal_grant)==readers
    assert not onair._p4_grant_read_owned(m,r)
    if bad=='selected-close':
        assert m._next is close and m.close_context is ctx
    assert not m.outbox.inflight and a.tx_acked_bytes==0


@pytest.mark.parametrize('cleanup',['peer-phase','link-down'])
def test_semantic_installed_origin_clears_on_admitted_phase_or_link_down(cleanup):
    m,a,r=semantic_receive_origin()
    if cleanup=='peer-phase':
        from hfmodem.tests.shrike.test_p4_normal_short_0928 import normal
        # Explicit receive-role fixture and encoded/decoded field, then actual
        # production registration/turn admission. No positive RF claim.
        m.state=p4turn.State.IRS_ANSWER
        obs=normal(m.last.stream_sample+60000,status=0,data=b'')
        reg=r.register(obs)
        m.observe(reg,now=obs.stream_sample+round(.810*FS))
        assert m.peer_phase is not None and m.peer_phase.onset==obs.stream_sample
        assert reg in m.events
    else:
        m.link_down('semantic cleanup control')
        assert m.state==p4turn.State.DOWN
    assert m._grant_receive_origin is None
    assert not m.outbox.inflight and a.tx_acked_bytes==0
    assert not onair._p4_grant_read_owned(m,r)


def test_reader_revocation_disables_reads_without_reclassifying_actual_history():
    m,a,r=semantic_receive_origin()
    original=m._grant_receive_origin
    for _ in range(3):
        m._repeat();m.keyed()
        r.note_emission(Emission(Kind.BODY,Role.ISS,m.last.stream_sample,.798,packet_reply=True))
        onair._p4_sync_grant_receive_origin(m,r)
        assert m._grant_receive_origin.context is original.context
    onair._p4_peer_normal_reader(m,r)
    r._train_generation+=1
    assert not onair._p4_grant_read_owned(m,r)
    origin=m._grant_receive_origin
    onair._p4_sync_grant_receive_origin(m,r)
    assert m._grant_receive_origin is origin
    m._repeat()
    assert m._next is None and m.close_context is None and m.grants==4
    assert not a.tx_acked_bytes


@pytest.mark.parametrize('kind,width',[(Kind.LONG,40),(Kind.CHIRP,22)])
@pytest.mark.parametrize('mode',['empty','greeting-gated','hold','eligible-body','eligible-y27'])
def test_registered_long_policy_and_real_constructor_eligibility(kind,width,mode):
    from hfmodem.shrike.p4receiver import P4Receiver, Clock
    from hfmodem.tests.shrike.test_p4turn import machine
    from hfmodem.tests.shrike.test_p4_late_long_ack_1010 import accepted
    hold=mode=='hold'
    m,a=machine(role='irs',data='hold' if hold else 'body',
        ladder='long' if hold else 'none',prekey_lead=.025,
        breakin='y27' if mode=='eligible-y27' else 'body',chirp_ack='parity')
    if mode!='empty':a.on_host_data(b'queued application\r')
    m.allow_breakin=mode!='greeting-gated'
    delivered=[]
    from types import SimpleNamespace
    m.adapter.host=SimpleNamespace(deliver=delivered.append,supervisor=lambda *a:None)
    r=P4Receiver(FS,admitted=True,chirp=True)
    r.expected=True;r.clock=Clock(Expect.PACKET,FS,.330,3.75,0.)
    # Explicit semantic PCM-independent fields, actual CRC/width and production
    # registry/admission. Non-fill source makes once-only delivery nonvacuous.
    reg=r.register(replace(accepted(FS,b'one delivery\r'.ljust(width,b'\x1e'),0x40,kind).observation,placed=True))
    m.observe(reg,now=FS+round(3.3*FS))
    p=m.decide(FS+163200-m.lead)
    source=m.adapter.result;epoch=m._receive_epoch,m._receive_active,m._receive_boundary
    assert delivered==[b'one delivery\r']
    assert not m.outbox.inflight and a.tx_acked_bytes==0
    if mode.startswith('eligible'):
        assert (p.kind,p.word,p.state,p.payload)==(Kind.WORD,'Y27',p4turn.State.IRS_BREAKIN,b'')
        m.keyed()
        assert m.breakins==1 and not m.outbox.inflight and a.tx_acked_bytes==0
    else:
        policy_word='V27' if kind==Kind.CHIRP else 'X27'
        assert (p.kind,p.word,p.state,p.status,p.payload)==(Kind.WORD,policy_word,p4turn.State.IRS_ANSWER,0,b'')
        assert p.accepted_target==FS and p.current_slot==FS and p.current_slot_accepted
        m.refused('deadline')
        assert m._deferred[0]==FS and m._deferred[1]['body'] is reg.observation
        assert m._deferred[1]['due']==391200
        repeated=m.decide(391200-m.lead)
        assert repeated.stream_sample==391200 and repeated.accepted_target==FS
        assert not repeated.current_slot_accepted and repeated.ack_identity==p.ack_identity
        assert m.adapter.result is source
        m.keyed()
        assert (m._receive_epoch,m._receive_active,m._receive_boundary)==epoch
        assert m.grants==m.breakins==0 and not m.outbox.inflight and a.tx_acked_bytes==0
    assert delivered==[b'one delivery\r'] and m.adapter.result is source
