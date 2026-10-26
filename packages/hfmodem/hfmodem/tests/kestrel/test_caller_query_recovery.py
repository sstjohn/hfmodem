# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A complete solicited caller answer recovers a missed nonfinal continue."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.contact.slots import Mixer
from hfmodem.contact.varac import frame_message
from hfmodem.tests.evidence import CORPUS,require_input
from hfmodem.kestrel.vara import vara_arq as A,vara_frames as F,vara_mfsk as M
from .test_responder_control_stream import station,feed
from .test_idle_poll import clock as clock,_Clock as _Clock

PAYLOAD=frame_message('<AWAY>\r\nAway from the station; please leave a VMail.')

def pending(*,final=False,query=True):
    hs,io=station('W9SSJ');hs.caller='VE6SCL';hs.chat=True
    hs.turn=A._TURN_OURS;hs._tx_level=4;hs._start_compression('TEXT')
    hs.send(b'2 hi' if final else PAYLOAD)
    if query:assert hs._query_answerer_over()
    return hs,io

def native():
    p=Path(__file__).parent/'fixtures/ve6scl-query-1024/provenance.json'
    meta=json.loads(p.read_text());path=require_input(CORPUS/meta['corpus_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    fs,x=wavfile.read(path);assert fs==M.FS
    return Mixer(1500)(x.astype(float)/32768)

def reply(frame=5,call='W9SSJ',seed=60):
    kind=F.BurstKind('test-query-answer',(74,),31,seed,1+62*frame,1)
    return np.r_[np.zeros(4800),M.synth_burst(call,F.for_bw(kind,'500')),np.zeros(4800)]

@pytest.mark.parametrize('block',[960,4096,24000])
def test_native_solicited_reply_settles_only_first_block_and_steps_down(block):
    hs,io=pending();before=len(io.sent);x=native();feed(hs,x,block)
    assert hs.payload_ledger.confirmed==43,io.msgs
    assert hs.payload_ledger.pending==12 and hs._tx_pending is not None
    assert hs._tx_pending[2]==2 and hs._over==2
    assert len(io.sent)==before+1
    feed(hs,x,block)
    assert hs.payload_ledger.confirmed==43 and len(io.sent)==before+1

@pytest.mark.parametrize('frame',[5,8])
def test_proven_positive_indices_use_existing_continuation(frame):
    hs,io=pending();feed(hs,reply(frame))
    assert hs.payload_ledger.confirmed==43 and hs.payload_ledger.pending==12

@pytest.mark.parametrize('case',['no-query','refused-query','refused-renewal','stale-wall','stale-samples',
    'identical-new-pending','final-over','wrong-call','wrong-family','missing-data',
    'truncated-tail','corrupt-tail','other-role','other-turn','chat-off'])
def test_no_delivery_credit_outside_exact_solicited_nonfinal_context(case):
    hs,io=pending(final=case=='final-over',query=case not in ('no-query','refused-query'))
    if case in ('refused-query','refused-renewal'):
        io.tx_went_out=lambda:False
        assert not hs._query_answerer_over()
        io.tx_went_out=lambda:True
    if case=='stale-wall':_Clock.t+=3
    if case=='stale-samples':hs._answer_query_samples=3*M.FS
    if case=='identical-new-pending':hs._tx_pending=tuple(list(hs._tx_pending))
    if case=='other-role':hs.role='initiator'
    if case=='other-turn':hs.turn=A._TURN_PEER
    if case=='chat-off':hs.chat=False
    x=reply(frame=18 if case=='missing-data' else 5,
            call='N0ZZZ' if case=='wrong-call' else 'W9SSJ',
            seed=288 if case=='wrong-family' else 60)
    if case=='truncated-tail':x=x[:4800+29*M.HOP]
    if case=='corrupt-tail':
        kind=F.for_bw(A._CALLER_QUERY_ANSWERS[5],'500')
        tones=F.handshake_tones('W9SSJ',kind);tones[-1]=50 if tones[-1]!=50 else 52
        x=np.r_[np.zeros(4800),M.synth_tones(tones),np.zeros(4800)]
    before=hs.payload_ledger.confirmed;feed(hs,x)
    assert hs.payload_ledger.confirmed==before,io.msgs

def test_native_reply_cannot_key_before_its_complete_tail():
    hs,io=pending();before=len(io.sent);x=native();cut=int(1.3*M.FS)
    feed(hs,x[:cut]);assert hs.payload_ledger.confirmed==0 and len(io.sent)==before
    feed(hs,x[cut:]);assert hs.payload_ledger.confirmed==43


def test_extra_long_answer_fits_are_bounded_to_fresh_query_window(monkeypatch):
    hs,io=pending();calls=[];original=A._payload_fit
    def fit(x,kind,*args,**kwargs):
        if kind in A._CALLER_QUERY_ANSWERS.values():calls.append(len(x))
        return original(x,kind,*args,**kwargs)
    monkeypatch.setattr(A,'_payload_fit',fit)
    feed(hs,np.zeros(3*M.FS))
    assert 0<len(calls)<=12 and min(calls)>=A._SESSION_NEED
    assert hs.payload_ledger.confirmed==0


@pytest.mark.parametrize('frame',[8,18])
def test_recorded_stock_positive_and_missing_data_answer(frame):
    hs,io=station('W9SSJ');hs.caller='VE6SCL';hs.chat=True;hs.turn=A._TURN_OURS
    hs._link_high_speed=True;hs._rec500=5
    hs._lead=A.VB6Rnd(F.lead_state(hs.caller,True,'500'));hs._start_compression('TEXT')
    original=b'512 '+bytes(np.random.default_rng(1024).integers(33,127,512,dtype=np.uint8))
    hs.send(original);assert hs._query_answerer_over()
    meta=json.loads((Path(__file__).parent/'fixtures/ve6scl-query-1024/provenance.json').read_text())['stock'][str(frame)]
    path=require_input(CORPUS/meta['corpus_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    fs,x=wavfile.read(path);assert fs==M.FS;feed(hs,Mixer(1500)(x.astype(float)))
    assert hs.payload_ledger.confirmed==(90 if frame==8 else 0),io.msgs
    assert hs.payload_ledger.pending==len(original)-hs.payload_ledger.confirmed
    if frame==8:assert hs._tx_pending[2]==A._phy.level_key('500',9)
