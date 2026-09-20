# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A caller's CS3 replaces its response permutation, independent of grid parity."""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex

ENTRY = 2044487
PEER = 5628499
FIRST_SLOT = 94
PULSE = 5673437


def recorded_turn(tmp_path, *, peer_swapped=False, answering=False, grid_parity=False):
    # onair-0919-2325 TX73→first CS3 TX74: previous ACK5613437,
    # peer0x02 prompt frame5628499, desired outgoing pulse5673437.
    clock = ReplyClock(entry_phase=ENTRY)
    clock.target(0)
    grid = onair._MasterGrid(32585,60000,0,packet_n=38880,cs_n=10080,d_max_n=6240)
    grid.protocol,grid.sending = spec.Protocol.PACTOR3,False
    grid.acquired=grid.corroborated=True
    grid.d_n,grid.d_ref_n=4800,38880
    grid.reply_clock=clock
    grid.note_p3_packet(PEER-60000,39120,60000,swapped=not peer_swapped)
    grid.note_p3_control(PULSE-60000)
    grid.note_p3_packet(PEER,39120,60000,swapped=peer_swapped)
    tx=duplex(grid,tmp_path)
    tx.reply_clock=clock
    tx.host.arq.answering=answering
    # Same physical receive history with either arbitrary local parity. The
    # local counter still alternates over skipped slots and subsequent packets.
    grid.shift_slot=FIRST_SLOT-int(grid_parity)
    tx.live.now=tx.live.pos=PEER+30000
    return tx,grid


def emitted(tmp_path, tx):
    meta=json.loads((tmp_path/f'tx_{tx.n:02d}.json').read_text())
    fs,raw=wavfile.read(tmp_path/f'tx_{tx.n:02d}.wav')
    assert fs==48000
    audio=raw.astype(float)/32768 if raw.dtype==np.int16 else raw.astype(float)
    return meta,np.pad(audio,(4800,4800))


@pytest.mark.parametrize('peer_swapped',[False,True])
@pytest.mark.parametrize('grid_parity',[False,True])
@pytest.mark.parametrize('skipped',[0,1,2])
def test_recorded_changeover_uses_projected_peer_carriers(
        tmp_path,peer_swapped,grid_parity,skipped):
    tx,grid=recorded_turn(tmp_path,peer_swapped=peer_swapped,
                          grid_parity=grid_parity)
    slot=FIRST_SLOT+skipped
    tx.aim(grid,slot)
    tx.breakin_due=True
    tx.send_packet(1,b'ABC',0,breakin=True)
    assert not tx.refused and len(tx.live.emissions)==1
    meta,audio=emitted(tmp_path,tx)
    got=rxfront.SyncedRx().control_signal_at(
        audio,4800+min(meta['pulse_offsets']),details=True)
    assert got is not None and got.packet[:3]==(1,0,b'ABC')
    peer_now=peer_swapped ^ bool(skipped&1)
    expected=not peer_now
    assert got.carrier_swapped==expected
    pulse=meta['audio_start']+min(meta['pulse_offsets'])
    assert pulse==PULSE+skipped*60000
    assert (pulse-ENTRY)%60000==28950
    # The physical leading carrier agrees with the independently decoded body.
    assert (meta['pulse_offsets'][0]>meta['pulse_offsets'][1])==expected


@pytest.mark.parametrize('grid_parity',[False,True])
@pytest.mark.parametrize('peer_swapped',[False,True])
def test_next_ordinary_packets_continue_the_emitted_changeover_arrangement(
        tmp_path,grid_parity,peer_swapped):
    tx,grid=recorded_turn(tmp_path,grid_parity=grid_parity,peer_swapped=peer_swapped)
    tx.aim(grid,FIRST_SLOT)
    tx.breakin_due=True
    tx.send_packet(1,b'ABC',0,breakin=True)
    assert not tx.refused
    first=not peer_swapped
    grid.reverse(to_iss=True)
    tx.host.arq.role=arq.ISS
    for elapsed in (1,2,4):
        # Synthetic reactive peer control after accepted CS3; the real2325
        # gateway did not accept the erroneous arrangement sent that evening.
        peer_control=PEER+88800+(elapsed-1)*60000
        grid.note_peer_codeword(peer_control,10080,'ACK','WS8EOC',
                                protocol=spec.Protocol.PACTOR3)
        tx.aim(grid,FIRST_SLOT+elapsed)
        tx.send_packet(1,b'HELLO',elapsed&3)
        assert not tx.refused
        meta,audio=emitted(tmp_path,tx)
        row=4800+min(meta['pulse_offsets'])+4320
        sync=rxfront.SyncedRx()
        sync.packet_at,sync.packet_level=row,1
        got=sync.packet(audio,preferred_only=True)
        assert got is not None and got.packet[:3]==(1,elapsed&3,b'HELLO')
        assert got.carrier_swapped==(first ^ bool(elapsed&1))
        assert meta['audio_start']+min(meta['pulse_offsets'])==PULSE+elapsed*60000


@pytest.mark.parametrize('peer_swapped',[False,True])
@pytest.mark.parametrize('grid_parity',[False,True])
@pytest.mark.parametrize('skipped',[0,1,2])
def test_answerer_retains_projected_peer_arrangement(
        tmp_path,peer_swapped,grid_parity,skipped):
    tx,grid=recorded_turn(tmp_path,peer_swapped=peer_swapped,answering=True,
                          grid_parity=grid_parity)
    # Production currently refuses all actual RF on answered links because
    # the answerer's TX anchor has not been implemented. Test the independently
    # specified carrier relation without bypassing that emission guard.
    expected=peer_swapped ^ bool(skipped&1)
    assert tx._p3_reply_carrier_order(PULSE+skipped*60000,grid_parity)==expected
    assert not tx.live.emissions


@pytest.mark.parametrize('grid_parity',[False,True])
def test_refused_breakin_does_not_rephase_subsequent_packets(tmp_path,grid_parity):
    tx,grid=recorded_turn(tmp_path,peer_swapped=grid_parity,grid_parity=grid_parity)
    tx.aim(grid,FIRST_SLOT)
    tx.breakin_due=True
    tx.listening=True
    tx.send_packet(1,b'ABC',0,breakin=True)
    assert tx.refused and not tx.live.emissions
    grid.reverse(to_iss=True)
    tx.host.arq.role=arq.ISS
    grid.note_peer_codeword(PEER+88800,10080,'ACK','WS8EOC',
                            protocol=spec.Protocol.PACTOR3)
    tx.listening=False
    tx.aim(grid,FIRST_SLOT+1)
    tx.send_packet(1,b'HELLO',1)
    assert not tx.refused
    meta,audio=emitted(tmp_path,tx)
    sync=rxfront.SyncedRx()
    sync.packet_at=4800+min(meta['pulse_offsets'])+4320
    sync.packet_level=1
    got=sync.packet(audio,preferred_only=True)
    assert got is not None and got.packet[2]==b'HELLO'
    assert got.carrier_swapped==(not grid_parity)


@pytest.mark.parametrize('grid_parity',[False,True])
@pytest.mark.parametrize('peer_swapped',[False,True])
def test_long_packet_retains_the_emitted_changeover_calibration(
        tmp_path,grid_parity,peer_swapped):
    tx,grid=recorded_turn(tmp_path,grid_parity=grid_parity,peer_swapped=peer_swapped)
    tx.aim(grid,FIRST_SLOT)
    tx.breakin_due=True
    tx.send_packet(1,b'ABC',0,breakin=True)
    assert not tx.refused
    grid.reverse(to_iss=True)
    tx.host.arq.role=arq.ISS
    grid.regear(True)
    grid.note_peer_codeword(PEER+88800,10080,'ACK','WS8EOC',
                            protocol=spec.Protocol.PACTOR3)
    tx.aim(grid,FIRST_SLOT+1)
    dac=[]
    send=tx.live.transmit
    def record(audio,**kwargs):
        dac.append(audio.copy())
        return send(audio,**kwargs)
    tx.live.transmit=record
    tx.send_long_packet(3,b'LONG FIELD',1)
    assert not tx.refused and len(dac)==1 and len(tx.live.emissions)==2
    audio=np.pad(dac[0],(4800,4800))
    sync=rxfront.SyncedRx()
    # Long sends retain the normal audio-start reference; use the known
    # rendered leading phase, independent of the carrier choice under test.
    from hfmodem.shrike import placement
    row=4800+placement.pulse_lead(audio[4800:-4800])+4320
    sync.packet_at,sync.packet_level=row,3
    got=sync.packet(audio,preferred_only=True)
    assert got is not None and got.packet[2]==b'LONG FIELD'
    assert got.cycle_long
    assert got.carrier_swapped==peer_swapped


@pytest.mark.parametrize('grid_parity',[False,True])
@pytest.mark.parametrize('peer_swapped',[False,True])
def test_final_late_key_keeps_calibrated_packet_on_same_physical_parity(
        tmp_path,grid_parity,peer_swapped):
    tx,grid=recorded_turn(tmp_path,grid_parity=grid_parity,peer_swapped=peer_swapped)
    tx.aim(grid,FIRST_SLOT)
    tx.breakin_due=True
    tx.send_packet(1,b'ABC',0,breakin=True)
    assert not tx.refused
    grid.reverse(to_iss=True)
    tx.host.arq.role=arq.ISS
    grid.note_peer_codeword(PEER+88800,10080,'ACK','WS8EOC',
                            protocol=spec.Protocol.PACTOR3)
    tx.aim(grid,FIRST_SLOT+1)
    send=tx._tx
    def overrun(audio,label,**kwargs):
        # Rendering has already fixed the carrier order. Force the final
        # admission check to miss without bypassing its real slot selection.
        tx.live.now=tx.live.pos=tx.boundary-tx.live.key_notice+480
        return send(audio,label,**kwargs)
    tx._tx=overrun
    tx.send_packet(1,b'HELLO',1)
    assert not tx.refused and tx.slot==FIRST_SLOT+3
    meta,audio=emitted(tmp_path,tx)
    sync=rxfront.SyncedRx()
    sync.packet_at=4800+min(meta['pulse_offsets'])+4320
    sync.packet_level=1
    got=sync.packet(audio,preferred_only=True)
    assert got is not None and got.packet[2]==b'HELLO'
    assert got.carrier_swapped==peer_swapped
    assert meta['audio_start']+min(meta['pulse_offsets'])==PULSE+3*60000
