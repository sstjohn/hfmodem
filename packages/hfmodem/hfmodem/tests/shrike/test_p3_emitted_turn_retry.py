"""A real WS8EOC CS2 retains its unacknowledged CS3's emitted pulse raster."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3acquire, p3rx, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex

FS, CYCLE = 48000, 60000
CAP = Path(__file__).resolve().parents[5]/'captures/onair-0919-2342'


def pending(tmp_path):
    g = onair._MasterGrid(32581,CYCLE,0,packet_n=38880,cs_n=10080,d_max_n=6240)
    g.protocol,g.sending = spec.Protocol.PACTOR3,False
    g.acquired=g.corroborated=True
    g.d_n,g.d_ref_n=4800,38880
    g.cycles=63
    clock=ReplyClock(entry_phase=3964483,pulse_epoch=3993433)
    g.reply_clock=clock
    g.note_p3_packet(7128585,39120,CYCLE,swapped=False)
    g.note_p3_control(7173433)
    g.note_p3_packet(7188585,39120,CYCLE,swapped=True)
    sess=_Session(role=arq.ISS)
    tx=duplex(g,tmp_path)
    tx.reply_clock=clock
    sess.host.peer=tx
    tx.attach(sess.host)
    packet=arq._Packet(status=0,payload=b';FW',sl=1,breakin=True)
    sess.host.arq._inflight=packet
    sess.host.arq._next_seq=1
    sess.host.arq._buffer_raw=3
    tx.aim(g,121)
    tx.breakin_due=True
    tx.live.now=tx.live.pos=7280000
    tx.send_packet(1,b';FW',0,breakin=True)
    assert not tx.refused
    assert tx.tx_audio_start+min(tx.tx_pulse_offsets)==7293433
    assert tx._p3_emitted_turn is not None
    g.reverse(to_iss=True)
    g.turn_accepted=False
    g.cycles=65
    tx.aim(g,124)
    tx.breakin_due=True
    tx.live.now=tx.live.pos=7467800
    return sess,tx,g,packet


def test_recorded_cs2_repeats_same_field_after_old_irs_geometry_is_cleared(tmp_path):
    path=CAP/'hold_65.wav'
    if not path.exists():
        pytest.skip('2342 capture absent')
    sess,tx,g,packet=pending(tmp_path)
    meta=json.loads(path.with_suffix('.json').read_text())
    fs,pcm=wavfile.read(path)
    assert fs==FS
    audio=pcm[:,0].astype(float)/32768
    base=meta['end_stream_sample']-len(audio)
    lo=7457526-base-4800
    got=p3acquire.control_signal(audio[lo:],offsets=range(-25,1))
    assert got and got.event.cs==arq.CS_REQUEST and got.event.packet is None
    ev=replace(got.event,t=(base+lo+got.event.start)/FS)
    before=len(tx.live.emissions)
    sess.rx._on(ev,anchored=True)
    assert g._p3_reply_timing() is None  # Obsolete IRS geometry stays invalid.
    assert len(tx.live.emissions)==before+1
    assert not tx.refused
    assert tx.tx_audio_start+min(tx.tx_pulse_offsets)==7473433
    assert sess.host.arq._inflight is packet
    assert (packet.seq,packet.payload)==(0,b';FW')
    assert sess.host.arq._next_seq==1 and sess.host.arq._buffer_raw==3
    fs,wave=wavfile.read(sorted(tmp_path.glob('tx_*.wav'))[-1])
    padded=np.pad(wave.astype(float)/32768,(4800,4800))
    field,valid,_=p3rx.decode_changeover_details(padded,4800+min(tx.tx_pulse_offsets))
    assert valid and field==bytes.fromhex('3b465700693e')


@pytest.mark.parametrize('bad',[
    'quiet','stale','unkeyed','different_packet','mutated_field','wrong_protocol',
    'wrong_phase','under_own_tx','control_tail_in_tx','cs3','changed_cycle',
    'changed_clock','not_pending'])
def test_only_fresh_answer_to_same_actual_pending_turn_can_preserve_clock(tmp_path,bad):
    sess,tx,g,packet=pending(tmp_path)
    phase=7457526
    g.note_peer_codeword(phase,10080,'REQ','WS8EOC',protocol=spec.Protocol.PACTOR3)
    assert tx._p3_retry_phase(g,124)==7473433
    if bad=='quiet': g.peer_cs=None
    elif bad=='stale': g.cycles+=onair.ISS_GUARD_CYCLES+1
    elif bad=='unkeyed': tx._p3_emitted_turn=None
    elif bad=='different_packet': sess.host.arq._inflight=replace(packet)
    elif bad=='mutated_field': packet.payload=b'BAD'
    elif bad=='wrong_protocol': g.peer_cs=g.peer_cs._replace(protocol=spec.Protocol.PACTOR1)
    elif bad=='wrong_phase': g.peer_cs=g.peer_cs._replace(at=phase-20000)
    elif bad=='under_own_tx': tx.keyings.append((phase-1,phase+1))
    elif bad=='control_tail_in_tx': tx.keyings.append((phase+5000,phase+15000))
    elif bad=='cs3': g.peer_cs=g.peer_cs._replace(name='BREAK-IN')
    elif bad=='changed_cycle': g.slot_n=180000
    elif bad=='changed_clock': tx.reply_clock.pulse_epoch+=1000
    elif bad=='not_pending': packet.breakin=False
    assert tx._p3_retry_phase(g,124) is None
    assert tx._place_breakin()==''
    assert tx.unplaceable is not None
    assert len(tx.live.emissions)==1


def test_valid_repeat_never_bypasses_final_tx_deadline(tmp_path):
    sess,tx,g,packet=pending(tmp_path)
    g.note_peer_codeword(7457526,10080,'REQ','WS8EOC',protocol=spec.Protocol.PACTOR3)
    assert tx._p3_retry_phase(g,124)==7473433
    # Pin the requested slot: exercise the final guard independently of _flip's
    # legitimate choice to spend a missed slot and aim at the next one.
    tx._flip=lambda:False
    tx.live.now=tx.live.pos=7473433+1000
    tx.send_packet(1,b';FW',0,breakin=True)
    assert tx.refused
    assert len(tx.live.emissions)==1


def test_missed_next_control_retains_phase_but_silence_cannot_refresh_evidence(tmp_path):
    sess,tx,g,packet=pending(tmp_path)
    g.note_peer_codeword(7457526,10080,'REQ','WS8EOC',protocol=spec.Protocol.PACTOR3)
    original=tx._p3_emitted_turn
    tx.send_packet(1,b';FW',0,breakin=True)
    assert not tx.refused
    assert tx._p3_emitted_turn is original
    tx.aim(g,125)
    g.cycles+=1
    assert tx._p3_retry_phase(g,125)==7533433
    # Keying again must not turn our own traffic into fresh peer evidence.
    tx.aim(g,134)
    g.cycles+=9
    assert tx._p3_retry_phase(g,134) is None


def test_valid_repeat_never_bypasses_peer_collision_guard(tmp_path):
    sess,tx,g,packet=pending(tmp_path)
    g.note_peer_codeword(7457526,10080,'REQ','WS8EOC',protocol=spec.Protocol.PACTOR3)
    assert tx._p3_retry_phase(g,124)==7473433
    # The pulse remains correctly placed, but this much PTT lead would key
    # across the decoded peer control. Retaining a clock cannot authorize it.
    tx.settle=.250
    tx.live.now=tx.live.pos=7450000
    tx.send_packet(1,b';FW',0,breakin=True)
    assert tx.refused
    assert len(tx.live.emissions)==1
