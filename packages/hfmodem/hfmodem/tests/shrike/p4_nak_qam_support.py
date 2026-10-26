# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Capture actual neutral keys while asserting history and byte neutrality."""
from contextlib import contextmanager
from unittest.mock import patch
import hashlib
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.shrike.p4turn import P4Turn,State

@contextmanager
def neutral_keys(*,refused=None):
    original=P4Turn.keyed
    original_refused=P4Turn.refused
    original_emission=P4Receiver.note_emission
    records=[]
    def keyed(self,*args,**kwargs):
        neutral=self._next is not None and not self._next.current_slot_accepted
        if neutral:
            names=('ack_word','acked_counter','_last_ack','_last_body_ack',
                   '_held_answer','_packet_ack')
            history=tuple(getattr(self,n) for n in names)
            permission=(self.counter,self.grants,self.breakins,self.first_body,
                        self._receive_epoch,self._peer_generation,self._receipt,
                        self._negotiation,self._peer_word,self._peer_word_at)
            bounds=(self._irs_silent_until,self._irs_silent_done,self._hold_until,
                    self.session_end,self.session_wall_end)
            a=self.outbox.arq
            silent=a._silent_cycles
            outgoing=(self.outbox.inflight,bytes(a._outbuf),a._buffer_raw,a._inflight)
        sent=original(self,*args,**kwargs)
        if neutral:
            assert sent.kind==Kind.WORD and sent.state==State.IRS_ANSWER
            assert sent.payload == b''
            assert tuple(getattr(self,n) for n in names)==history
            assert (self.counter,self.grants,self.breakins,self.first_body,
                    self._receive_epoch,self._peer_generation,self._receipt,
                    self._negotiation,self._peer_word,self._peer_word_at)==permission
            assert (self._irs_silent_until,self._irs_silent_done,self._hold_until,
                    self.session_end,self.session_wall_end)==bounds
            assert a._silent_cycles==silent
            assert (self.outbox.inflight,bytes(a._outbuf),a._buffer_raw,a._inflight)==outgoing
            records.append(dict(at=sent.stream_sample,word=sent.word,
                                purpose=sent.purpose,kind=sent.kind,state=sent.state))
        return sent
    def refusal(self,reason):
        proposal=self._next
        if proposal is None or proposal.current_slot_accepted:
            return original_refused(self,reason)
        names=('ack_word','acked_counter','_last_ack','_last_body_ack',
               '_held_answer','_packet_ack','counter',
               'grants','breakins','first_body','_receive_epoch','_peer_generation',
               '_receipt','_negotiation','_peer_word','_peer_word_at')
        history=tuple(getattr(self,n) for n in names)
        bounds=(self._irs_silent_until,self._irs_silent_done,self._hold_until,
                self.session_end,self.session_wall_end)
        arq=self.outbox.arq
        silent=arq._silent_cycles
        outgoing=(self.outbox.inflight,bytes(arq._outbuf),arq._buffer_raw,arq._inflight)
        last=self.last
        result=original_refused(self,reason)
        assert proposal.kind==Kind.WORD and proposal.payload==b''
        assert not proposal.current_slot_accepted
        assert tuple(getattr(self,n) for n in names)==history
        assert (self._irs_silent_until,self._irs_silent_done,self._hold_until,
                self.session_end,self.session_wall_end)==bounds
        assert arq._silent_cycles==silent
        assert (self.outbox.inflight,bytes(arq._outbuf),arq._buffer_raw,arq._inflight)==outgoing
        assert self.last is last
        if refused is not None:
            refused.append(dict(at=proposal.stream_sample,word=proposal.word,
                                purpose=proposal.purpose,kind=proposal.kind,
                                state=proposal.state,reason=reason))
        return result
    def emission(self,em,**kwargs):
        if em.current_slot_accepted:
            return original_emission(self,em,**kwargs)
        presence=(self.silent_cycles,self._counted_cycle,self._counted_live_onset,
                  tuple(self.live_onsets))
        hold=(self.hold.epoch,self.hold.unread,self.hold.last) if self.hold else None
        result=original_emission(self,em,**kwargs)
        assert (self.silent_cycles,self._counted_cycle,self._counted_live_onset,
                tuple(self.live_onsets))==presence
        assert ((self.hold.epoch,self.hold.unread,self.hold.last) if self.hold else None)==hold
        return result
    with patch.object(P4Turn,'keyed',keyed),patch.object(P4Turn,'refused',refusal), \
         patch.object(P4Receiver,'note_emission',emission):
        yield records


def without_neutral_requests(monkeypatch):
    """Remove only the approved neutral proposals and their early read shortcut."""
    from hfmodem.shrike import onair
    silence=P4Turn.silence
    def no_request(self,*args,**kwargs):
        silence(self,*args,**kwargs)
        if self._next is not None and not self._next.current_slot_accepted:
            self._next=None
    monkeypatch.setattr(P4Turn,'silence',no_request)
    monkeypatch.setattr(P4Turn,'_answer_unresolved',lambda *a:None)
    monkeypatch.setattr(onair,'_p4_early_empty_answer',lambda *a:False)


@contextmanager
def waveform_keys():
    """Hash the wave actually handed to the offline physical key seam."""
    from hfmodem.tests.shrike.test_p4_link_0927 import Tx
    original=Tx._tx
    rows=[]
    def keyed(self,audio,what):
        original(self,audio,what)
        rows.append(dict(at=self.tx_audio_start,samples=len(audio),dtype=audio.dtype.str,
                         sha256=hashlib.sha256(audio.tobytes()).hexdigest()))
    with patch.object(Tx,'_tx',keyed):
        yield rows
