# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded late ACK through regrid, real receiver/ARQ, and fake DAC.

This bounded recovery seam does not model the whole live loop or RF feedback.
The initial clock and pending counter-zero CS3 timing are TX61's recorded
state; its application payload is modeled by ABC followed by queued DEF. All
subsequent receive samples come from the continuous WS8EOC recording.
"""
import json

import numpy as np
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3acquire, p3rx, placement, rxfront
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_p3_late_turn_answer_2300 import (
    recording as recording, pending, START, END, CYCLE)


def recover(recording, tmp_path, monkeypatch, *, legacy=False):
    session, grid = pending()
    host, receiver = session.host, session.rx
    live = _Bench(seconds=180)
    live.audio[:len(recording)] = recording
    live.now = live.pos = 5313024  # End of actual listening hold47.
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(host)
    host.peer = tx
    tx.live, tx.raster, tx.sessrx = live, grid, receiver
    tx.defer_p3_cs = True
    tx.reply_clock = ReplyClock(entry_phase=2404592, pulse_epoch=2433542)
    tx.p3_follow_offset = 'all'
    tx.tx_audio_start, tx.tx_end = START, END
    tx.breakin_due = True
    tx.breakin_cost_n = 96
    grid.remember_p3_turn(START+677)
    grid.note_p3_packet(5148691,39120,CYCLE,swapped=True)
    grid._p3_peer_confirmed = True
    receiver._p3_row0 = receiver._p3_delivered_at = 5153011
    receiver._p3_clock_role = arq.IRS
    receiver._p3_cycle_n = CYCLE
    receiver._p3_span = rxfront._frame_span(placement.SPEED_PATHS[1])
    receiver.sync.packet_level = 1
    calls=[]
    control = receiver.control_signal_in
    scan = receiver.deep_scan

    def read_control(audio, origin, raster):
        calls.append(('control', origin, origin+len(audio)))
        # Original admission could inspect only the first reply interval of
        # TX61; a later captured interval was ineligible despite pending CS3.
        if legacy and origin >= START+CYCLE:
            result=(None,None)
        else:
            result=control(audio,origin,raster)
        live.spend(144)
        return result

    def read_packet(audio):
        calls.append(('tracked' if receiver._tracked_only else 'blind',
                      receiver._scan_origin, receiver._scan_origin+len(audio)))
        result=scan(audio)
        live.spend(288)
        return result

    monkeypatch.setattr(receiver,'control_signal_in',read_control)
    monkeypatch.setattr(receiver,'deep_scan',read_packet)
    slot=88
    seg=np.zeros(0,np.float32)
    origin=live.pos
    for _ in range(3):
        receiver.new_cycle()
        grid.cycles += 1
        tx.aim(grid,slot)
        # Recover an already lost slot, as the actual loop did. Advancing only
        # wall time retains every sample to be read by production _regrid.
        live.now=max(live.now,tx.key_instant(grid,slot)+480-live.key_notice)
        slot,seg,origin=onair._regrid(live,grid,tx,host,receiver,slot,
                                    seg,origin,1920)
        if not host.arq.unconfirmed_breakin:
            break
        live.spend(1200)
    return session,tx,grid,calls


def test_regrid_reads_recorded_ack_and_emits_first_ordinary_data(
        recording,tmp_path,monkeypatch):
    session,tx,grid,calls=recover(recording,tmp_path,monkeypatch)
    assert not session.host.arq.unconfirmed_breakin
    assert session.host.arq._inflight.seq == 1
    assert tx.live.emissions and not tx.refused
    assert not any(kind=='blind' for kind,_,_ in calls)
    assert any(ev.kind=='cs' and ev.cs==arq.CS_ACK for ev in session.events)
    assert abs(session.rx._p3_answer_at-5357640)<480
    saved=json.loads((tmp_path/f'tx_{tx.n:02d}.json').read_text())
    fs,pcm=wavfile.read(tmp_path/f'tx_{tx.n:02d}.wav')
    corrected=p3acquire.compensate(pcm.astype(float)/32768,saved['tx_offset_hz'])
    decoded=p3rx.decode_p3_packets(np.pad(corrected,(2400,2400))).packets
    assert decoded and decoded[0].payload==b'DEF' and decoded[0].status&3==1
    assert session.host.arq.state is arq.State.CONNECTED
    assert not session.host.arq._qrt_pending


def test_original_single_interval_admission_stalls_same_recorded_recovery(
        recording,tmp_path,monkeypatch):
    session,tx,grid,calls=recover(recording,tmp_path,monkeypatch,legacy=True)
    assert session.host.arq.unconfirmed_breakin
    assert not tx.live.emissions
    assert not any(ev.kind=='cs' and ev.cs==arq.CS_ACK for ev in session.events)
    assert any(kind=='control' for kind,_,_ in calls)
