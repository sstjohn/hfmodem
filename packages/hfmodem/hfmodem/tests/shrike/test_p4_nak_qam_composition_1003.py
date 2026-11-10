# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Actual neutral keys revoke saved QAM work; refusal preserves ownership."""
import pytest
from hfmodem.shrike.p4receiver import Clock,Emission,Expect,Role
from hfmodem.shrike.p4observation import Kind
from hfmodem.tests.shrike.test_p4turn import machine
from hfmodem.tests.shrike.test_p4_qam16_live_1003 import recording as recording,receiver,read_row,FS

@pytest.mark.parametrize('action',['none','refused','keyed'])
def test_neutral_request_and_saved_qam_ownership(recording,action):  # noqa: F811
    audio,rows,_=recording
    row=rows[1];rx=receiver(row)
    rx.hold.now=lambda:0.
    rx.normal_deadline=lambda at:.100
    rx._normal_costs[4]=[1.]
    assert read_row(audio,row,rx)==[]
    entry=next(e for e in rx._normal_cache.values() if e.get('decoder'))
    candidate=entry['decoder']
    assert candidate.pending and not candidate.done
    generation=rx._train_generation
    m,arq=machine(role='irs')
    arq.on_host_data(b'queued composition control')
    queued=arq._buffer_raw
    history=(m.ack_word,m.acked_counter,m._last_ack,m._last_body_ack,m._held_answer)
    due=row['onset_sample']+round(.920*FS)
    m.silence(due-m.lead,answer_due=due)
    request=m.decide(due-m.lead)
    assert (request.kind,request.word,request.purpose)==(Kind.WORD,'V27',None)
    if action=='refused':
        m.refused('deadline')
        rx.note_refusal(request.kind,request.stream_sample)
    elif action=='keyed':
        sent=m.keyed(request.stream_sample)
        rx.note_emission(Emission(sent.kind,Role.IRS,sent.stream_sample,.244,
                                 current_slot_accepted=False))
        assert rx._train_generation>generation
        assert rx._normal_cache[row['onset_sample']] is entry
    rx._normal_costs[4].clear()
    records=rx.refine(deadline=.100)
    assert len(records)==(0 if action=='keyed' else 1)
    assert candidate.done
    assert (candidate.frame is None)==(action=='keyed')
    if records:
        assert records[0].observation.speedlevel=='SL9/10-short'
    assert (m.ack_word,m.acked_counter,m._last_ack,m._last_body_ack,m._held_answer)==history
    assert arq._buffer_raw==queued and arq._inflight is None
    assert m.last is None if action!='keyed' else m.last.purpose==None
    if action=='keyed':
        # The discarded search cannot starve a distinct later recorded copy.
        later=rows[7];at=later['onset_sample']
        rx.clock=Clock(Expect.PACKET,at,.330,1.25,later['cfo_hz'])
        recovered=read_row(audio,later,rx)
        assert len(recovered)==1
        assert recovered[0].observation.frame is not None
        assert recovered[0].observation.speedlevel=='SL9/10-short'
        assert entry['obs'].frame is None  # Old cache key may expire after six seconds.
        assert (m.ack_word,m.acked_counter,m._last_ack,m._last_body_ack,m._held_answer)==history
        assert arq._buffer_raw==queued and arq._inflight is None
