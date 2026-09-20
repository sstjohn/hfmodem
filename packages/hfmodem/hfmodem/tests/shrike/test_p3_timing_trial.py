# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Timing A/B through real entry, RX/ARQ, scheduler and fake duplex seams."""
import json
import sys
import numpy as np
import pytest
from hfmodem.shrike import arq, onair, placement, spec
from hfmodem.shrike.p3trial import TimingTrial, CYCLE, FS, ROTATION
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.archive import requires_ws8eoc_p3


def setup(tmp_path, arm, delay=0):
    s = _Session(role=arq.ISS, entry_pending=True)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    s.host.arq.cfg.long_cycle = False
    live = _Bench(seconds=60)
    g = onair._MasterGrid(10*FS, CYCLE, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.raster, tx.sessrx = live, g, s.rx
    tx.timing_trial = TimingTrial(arm)
    tx.defer_p3_cs = True
    tx.p3_follow_offset = 'none'
    tx.entry_delay_n = delay
    tx.aim(g, 0)
    live.now = live.pos = 9*FS
    tx.send_entry_packet(1, b'', 0x1a)
    assert not tx.refused and tx.timing_trial.entry_phase is not None
    return s, tx, live, g


def receive(s, tx, live, g, head):
    # Synthetic CRC-valid peer CS3; recorded acquisition is exercised below.
    raw = placement.changeover_packet(b'RMS', 0, swapped=False)
    center = int(np.argmax(placement.protocol_config().pulse()))
    pad = 9600-center
    x = np.pad(raw, (pad, FS//10))
    origin = head-9600
    live.audio[origin:origin+len(x)] = x
    live.now = live.pos = head+round(.84*FS)
    tx.aim(g, 2)
    s.rx.control_signal(x, origin, head)
    assert s.packets and s.packets[-1].packet[2] == b'RMS'
    assert s.host.arq.role == arq.IRS
    g.reverse(to_iss=False)


@pytest.mark.parametrize('arm', ['A', 'B'])
@pytest.mark.parametrize('gap', [895, 925, 945])
@pytest.mark.parametrize('delay', [0, 336])
def test_real_emission_uses_selected_epoch(tmp_path, arm, gap, delay):
    s, tx, live, g = setup(tmp_path, arm, delay)
    trial = tx.timing_trial
    entry = trial.entry_phase
    assert entry == 10*FS+delay+666
    sidecar = json.loads((tmp_path/'tx_01.json').read_text())
    assert sidecar['audio_start']+min(sidecar['pulse_offsets']) == entry
    head = entry+round(gap/1000*FS)
    receive(s, tx, live, g, head)
    tx.aim(g, 1)
    onair._p3_place_reply(g, tx, 1)
    tx.emit_pending_cs()
    assert not tx.refused and len(live.emissions) == 2
    pulse = tx.tx_audio_start+min(tx.tx_pulse_offsets)
    if arm == 'B':
        assert (pulse-entry)%CYCLE == ROTATION
    else:
        assert pulse-g._p3_peer[0] == round(.890*FS)+852
    assert trial.opportunities == 1
    assert trial.replies[-1]['cs'] == 1


def test_duplicate_quiet_cs2_and_offset_keep_entry_phase(tmp_path):
    s, tx, live, g = setup(tmp_path, 'B')
    trial = tx.timing_trial
    receive(s, tx, live, g, trial.entry_phase+round(.925*FS))
    opened = trial.opened
    tx.p3_follow_offset = 'all'
    s.rx.p3_receive_offset_hz = -13.5
    for slot, ci in [(1,0), (2,0), (4,1)]:
        tx.aim(g, slot)
        tx._pending_p3_cs = ci
        onair._p3_place_reply(g, tx, slot)
        tx.emit_pending_cs()
        assert not tx.refused
        pulse = tx.tx_audio_start+min(tx.tx_pulse_offsets)
        assert (pulse-trial.entry_phase)%CYCLE == ROTATION
    assert trial.opened == opened and trial.opportunities == 4
    assert [r['cs'] for r in trial.replies] == [1,1,2]


def test_last_guard_refuses_expired_trial_and_close_runs_once(tmp_path):
    s, tx, live, g = setup(tmp_path, 'B')
    t = tx.timing_trial
    receive(s, tx, live, g, t.entry_phase+round(.925*FS))
    tx.aim(g, 1)
    onair._p3_place_reply(g, tx, 1)
    live.now = live.pos = t.opened+20*FS
    before = len(live.emissions)
    tx.emit_pending_cs()
    assert tx.refused and len(live.emissions) == before
    assert onair._timing_trial_close(tx, s.host, live.now)
    assert t.closing and s.host.arq._disconnect_ticks == 0
    assert s.host.arq._qrt_pending and s.host.arq._breakin_pending
    assert onair._timing_trial_close(tx, s.host, live.now) is None


def test_refused_entry_does_not_establish_origin(tmp_path):
    s, tx, live, g = setup(tmp_path, 'B')
    tx.timing_trial = TimingTrial('B')
    tx.listening = True
    tx.send_entry_packet(1, b'', 0x1a)
    assert tx.refused and tx.timing_trial.entry_phase is None


def test_enqueue_forgiveness_cannot_cross_trial_deadline(tmp_path):
    s, tx, live, g = setup(tmp_path, 'B')
    t = tx.timing_trial
    t.packet(t.entry_phase+44400, 0, b'RMS', breakin=True, long_cycle=False)
    deadline = t.opened+20*FS
    tx.tx_pulse_offsets = (852, 852)
    tx.boundary = deadline-852-100
    live.now = live.pos = tx.boundary-1000
    assert not tx._timing_trial_refusal()
    assert tx._timing_trial_refusal(delay=100)
    assert t.reason == '20 second trial limit'


def test_trial_scoring_duplicates_limits_and_reset(tmp_path):
    t = TimingTrial('B')
    t.entry(1000, 1)
    t.entry(61010, 2)
    t.packet(105000, 0, b'RMS', breakin=True, long_cycle=False)
    t.target(105000, 852)
    origin, first = t.entry_phase, t.first_reply
    t.packet(165000, 0, b'RMS', breakin=True, long_cycle=False)
    assert t.entry_phase == origin and t.first_reply == first
    for seq, text in [(1,b' Trim'), (1,b' Trim'), (2,b'ode 1'), (3,b'.4.3.')]:
        t.packet(t.packets[-1]['phase']+CYCLE, 0x20|seq, text,
                 breakin=False, long_cycle=False)
    assert len(t.progress) == 3 and 'progression' in t.reason
    t.write(tmp_path)
    assert json.loads((tmp_path/'p3-timing-trial.json').read_text())['progression']
    assert TimingTrial('B').entry_phase is None
    for arm in ('A','B'):
        t = TimingTrial(arm)
        t.entry(1000,1)
        t.packet(45000,0,b'RMS',breakin=True,long_cycle=False)
        t.target(105000,852)  # A's first deadline must not follow the duplicate.
        assert t.first_reply < 105000
        assert t.attempt(t.first_reply+11*CYCLE)
        assert t.opportunities == 12
        assert not t.attempt(t.first_reply+12*CYCLE)


@pytest.mark.parametrize('status,long,why', [(0x80,False,'QRT'),
    (0x61,False,'role reversal'), (0x21,True,'long cycle')])
def test_other_turn_or_cycle_ends_scoring(status, long, why):
    t = TimingTrial('B')
    t.entry(1000,1)
    t.packet(45000,0,b'RMS',breakin=True,long_cycle=False)
    t.packet(105000,status,b'next',breakin=False,long_cycle=long)
    assert why in t.reason


def test_ambiguous_or_missing_entry_is_not_silently_reanchored():
    t = TimingTrial('B')
    t.entry(1000,1)
    t.entry(62000,2)
    assert 'ambiguous' in t.reason
    empty = TimingTrial('B')
    empty.packet(45000,0,b'RMS',breakin=True,long_cycle=False)
    assert 'no emitted entry' in empty.reason


def test_preset_and_conflicts_are_checked_before_io(monkeypatch):
    got = []
    monkeypatch.setattr(onair, 'run', lambda args: got.append(args) or 0)
    monkeypatch.setattr(sys, 'argv', ['onair', '--p3-timing-trial', 'B'])
    assert onair.main() == 0
    a = got[0]
    onair._timing_trial_defaults(a)
    assert a.p1_grant_only and a.over and a.hold == 32
    assert a.p3_speed_up == 'hold' and a.p3_repeat_gear == 0
    a.mail_fetch = True
    with pytest.raises(SystemExit, match='without mail'):
        onair._timing_trial_defaults(a)


@requires_ws8eoc_p3
@pytest.mark.parametrize('arm', ['A','B'])
def test_complete_recorded_session_loop_closes_trial(monkeypatch, tmp_path, arm):
    from hfmodem.tests.shrike import test_late_entry_loop as loop
    main = onair.main

    def trial_main():
        sys.argv += ['--p3-timing-trial', arm]
        return main()

    monkeypatch.setattr(onair, 'main', trial_main)
    result = loop.run_late(monkeypatch, tmp_path, 1, False)
    trial = json.loads((tmp_path/'p3-timing-trial.json').read_text())
    assert trial['opened'] is not None, result['log']
    assert trial['closing'] and trial['reason'], result['log']
    assert trial['opportunities'] <= 12
    assert result['host'].arq.state in (arq.State.DISCONNECTED, arq.State.DISCONNECTING)
    assert 'PTT off.' in result['log'] and 'the goodbye' in result['log']
    assert result['log'].strip().splitlines()[-1].startswith('verdict: P3 TIMING TRIAL')
    assert trial['replies'], result['log']
    for reply in trial['replies']:
        assert reply['cs'] in (1,2)
        pulse = reply['audio_start']+min(reply['pulse_offsets'])
        assert pulse < trial['opened']+20*FS
        if arm == 'B':
            # The loop may forgive a late enqueue by <=5 ms; it must not
            # reset the entry epoch to a newly decoded duplicate's phase.
            error = (pulse-trial['entry_phase']-ROTATION+CYCLE//2)%CYCLE-CYCLE//2
            assert abs(error) <= round(.005*FS), (reply, trial)
