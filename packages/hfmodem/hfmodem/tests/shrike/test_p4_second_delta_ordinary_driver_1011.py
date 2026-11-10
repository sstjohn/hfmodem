# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real-reader ordinary repeat and unread transfer, same harness on each revision."""
from unittest.mock import patch
import numpy as np
import pytest
from hfmodem.shrike import onair,p4turn,p4chirp,p4longtx
from hfmodem.shrike.p4receiver import Clock,Expect
from hfmodem.tests.shrike.test_p4_link_0927 import Tx,FS
from hfmodem.tests.shrike.test_p4_second_delta_c_harness_1011 import AbsoluteStream
from hfmodem.tests.shrike.test_p4_second_delta_parent_1011 import Recorder
from hfmodem.tests.shrike.second_delta_support_1011 import record

@pytest.mark.parametrize('variant',['repeat','unread_refinement'])
def test_c4_real_driver_source_and_hold_unread(variant):
    onset,cycle=480000,180000
    audio=np.zeros(3*cycle+168000,np.float32)
    first=p4chirp.entry_packet(b'ordinary source\r',0x61)
    second=(first if variant=='repeat' else p4longtx.long_packet(b'ordinary more\r',0x62))
    if variant=='unread_refinement':
        second=second.copy()
        second[round(.500*FS):]=0  # Explicit synthetic incomplete body, genuine header PCM/readers.
    audio[24000:24000+len(first)]=first
    audio[24000+cycle:24000+cycle+len(second)]=second
    live=AbsoluteStream(audio,onset-24000)
    tx=Tx(live)
    tx.settle=.025
    tx.p4_probe.emissions,tx.p4_probe.peer_breakin=[],True
    tx.p4_link_options.p4_ladder,tx.p4_link_options.p4_data='none','body'
    tx.p4_link_options.p4_link_timeout=len(audio)/FS
    rx=tx.sessrx.p4
    rx.clock=Clock(Expect.PACKET,onset,.330,3.75,0.)
    rx.expected=True
    machines,recorders,trace,keys=[],[],[],[]
    cls=p4turn.P4Turn
    def factory(*a,**kw):
        kw['role']='irs'
        m=cls(*a,**kw); machines.append(m)
        r=Recorder(m,rx,tx.p4_link_host.arq); recorders.append(r)
        observe,decide=m.observe,m.decide
        def feed(reg,**kw):
            r.snapshot('driver_registered',reg)
            result=observe(reg,**kw)
            r.snapshot('driver_observed',reg)
            return result
        def decision(*a,**kw):
            before=m.unread,rx.hold.unread
            r.snapshot('driver_before_decide')
            result=decide(*a,**kw)
            trace.append(('decide',before,(m.unread,rx.hold.unread)))
            r.snapshot('driver_after_decide')
            return result
        m.observe,m.decide=feed,decision
        return m
    def deliver(b):
        trace.append(('callback',machines[0].unread,rx.hold.unread))
        recorders[0].deliver(b)
    tx.p4_link_host.deliver=deliver
    note=rx.note_emission
    def emitted(em,**kw):
        recorders[0].snapshot('driver_before_note_emission')
        result=note(em,**kw)
        recorders[0].snapshot('driver_after_note_emission')
        return result
    rx.note_emission=emitted
    sink=tx._tx
    def key(wave,what):
        if tx.p4_tx_admit: tx.p4_tx_admit()
        current=machines[0]._next
        trace.append(('sink',machines[0].unread,rx.hold.unread))
        recorders[0].snapshot('driver_after_final_admission')
        sink(wave,what)
        keys.append(current)
    tx._tx=key
    with patch.object(p4turn,'P4Turn',factory):
        onair._run_p4_link(tx,live,onair._MasterGrid(0,60000,0,packet_n=36000,cs_n=12000,d_max_n=4800))
    r=recorders[0]
    record('ordinary_driver',case=variant,rows=r.rows,trace=trace,
           keys=[(p.stream_sample,p.word,p.accepted_target,p.current_slot,p.current_slot_accepted) for p in keys],
           events=tx.p4_probe.link_events,delivered=[b.hex() for b in r.delivered])
    assert r.delivered and r.delivered[0]==b'ordinary source\r'
    assert len(keys)>=2 and keys[0].accepted_target is not None
    if variant=='repeat':
        assert len(r.delivered)==1
        assert keys[1].accepted_target!=keys[0].accepted_target
        assert any(e['event']=='acked-repeat' for e in tx.p4_probe.link_events)
    else:
        changes=[(a,b) for name,a,b in trace if name=='decide' and b[0]>a[0]]
        assert changes, 'actual machine unread must increment in production decide'
        assert any(e['event']=='BREAKIN_UNREAD' for e in tx.p4_probe.link_events)
        assert any(name=='sink' and h>0 for name,u,h in trace), 'production must transfer unread to hold before sink'
    assert tx.p4_link_host.arq.tx_acked_bytes==0
