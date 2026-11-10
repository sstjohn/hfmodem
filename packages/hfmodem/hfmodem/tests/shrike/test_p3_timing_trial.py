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


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
@pytest.mark.parametrize('protocol', ['P2', 'P3'])
def test_tx_sidecar_retains_sample_timing(tmp_path, host, protocol):
    from hfmodem.core.audio import timestamp_limitation
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.live = _Bench(seconds=5)
    tx.live.timestamp_limitation = timestamp_limitation(host)
    tx.live.lat_provenance = 'nominal' if tx.live.timestamp_limitation else 'measured'
    tx.live.driver_offset_samples = 1771 if tx.live.timestamp_limitation else 756
    tx._p2_phase_lead = 100 if protocol == 'P2' else None
    tx._tx(np.ones(480, np.float32), protocol,
           **({'pulse_offsets': (2, 4)} if protocol == 'P3' else {}))
    row = json.loads((tmp_path / 'tx_01.json').read_text())
    assert (row['audio_start'], row['audio_end'], row['ptt_up']) == (
        tx.tx_audio_start, tx.tx_end, tx.tx_key_up)
    if tx.live.timestamp_limitation:
        assert row['timestamp_limitation'] == tx.live.timestamp_limitation
        assert row['driver_offset_provenance'] == 'nominal'
        assert row['driver_offset_samples'] == 1771
        assert row['timing_estimated'] is True
        assert 'estimated' in row['timing_basis'] and 'nominal' in row['timing_basis']
    else:
        assert not {'timestamp_limitation', 'timing_estimated', 'timing_basis'} & row.keys()


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
@pytest.mark.parametrize('entry', [False, True])
def test_trial_retains_sample_timing(tmp_path, host, entry):
    from hfmodem.core.audio import timestamp_limitation
    from hfmodem.shrike.p3trial import EntryTimingTrial
    trial = (EntryTimingTrial if entry else TimingTrial)('B')
    limitation = timestamp_limitation(host)
    trial.timing = (dict(timestamp_limitation=limitation,
                         driver_offset_provenance='nominal', driver_offset_samples=1771,
                         timing_estimated=True, timing_basis='estimated sample timing; nominal driver offset')
                    if limitation else {})
    trial.entry(1000, 1)
    trial.packet(50000, 0, b'RMS', breakin=True, long_cycle=False)
    trial.write(tmp_path)
    row = json.loads((tmp_path / trial.filename).read_text())
    assert row['entry_phase'] == 1000 and row['opened'] == 50000
    if limitation:
        for record in (row, row['entries'][0], row['packets'][0]):
            assert record['timestamp_limitation'] == limitation
            assert record['timing_estimated'] is True
            assert 'nominal' in record['timing_basis']
    else:
        assert not {'timestamp_limitation', 'timing_estimated', 'timing'} & row.keys()


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
        assert pulse-g._p3_peer[0] == round(onair.P3_REPLY_S*FS)+852
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


@pytest.mark.parametrize('pending', [False, True])
def test_last_guard_refuses_expired_trial_and_close_runs_once(tmp_path, pending):
    s, tx, live, g = setup(tmp_path, 'B')
    t = tx.timing_trial
    receive(s, tx, live, g, t.entry_phase+round(.925*FS))
    if pending:
        s.host.arq.on_host_data(b'pending application bytes beyond the trial')
    tx.aim(g, 1)
    onair._p3_place_reply(g, tx, 1)
    live.now = live.pos = t.opened+20*FS
    before = len(live.emissions)
    tx.emit_pending_cs()
    assert tx.refused and len(live.emissions) == before
    assert onair._timing_trial_close(tx, s.host, live.now)
    assert t.closing and s.host.arq._disconnect_ticks == 0
    assert s.host.arq._qrt_pending and s.host.arq._breakin_pending
    assert not s.host.arq._outbuf and s.host.arq._inflight is None
    payload, status = s.host.arq.next_breakin_packet(3)
    assert not payload and status & 0x80
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
    assert a.p3_control_waveform == 'historical'
    assert a.p1_grant_only and a.over and a.hold == 32
    assert a.p3_speed_up == 'hold' and a.p3_repeat_gear == 0
    a.mail_fetch = True
    with pytest.raises(SystemExit, match='without mail'):
        onair._timing_trial_defaults(a)
    monkeypatch.setattr(sys, 'argv', ['onair', '--p3-timing-trial', 'B',
                                      '--p3-control-waveform', 'current'])
    assert onair.main() == 0
    with pytest.raises(SystemExit, match='requires the historical'):
        onair._timing_trial_defaults(got[1])
    monkeypatch.setattr(sys, 'argv', ['onair'])
    assert onair.main() == 0
    onair._timing_trial_defaults(got[2])
    assert got[2].p3_control_waveform == 'current'


@requires_ws8eoc_p3
@pytest.mark.parametrize('arm', ['A','B'])
def test_complete_recorded_session_loop_closes_trial(monkeypatch, tmp_path, arm):
    from hfmodem.tests.shrike import test_late_entry_loop as loop
    monkeypatch.setattr(onair.ota, 'find_rigctl', lambda *_a: 'fake-rigctl')
    main = onair.main
    rigs = []

    class ObservedRig(loop._Rig):
        stopped = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            rigs.append(self)

        def stop(self):
            self.stopped = True
            return super().stop()

    def trial_main():
        sys.argv += ['--p3-timing-trial', arm, '--p3-control-waveform', 'historical']
        return main()

    monkeypatch.setattr(onair, 'main', trial_main)
    monkeypatch.setattr(loop, '_Rig', ObservedRig)
    result = loop.run_late(monkeypatch, tmp_path, 1, False)
    trial = json.loads((result['outdir']/'p3-timing-trial.json').read_text())
    assert trial['opened'] is not None, result['log']
    assert trial['closing'] and trial['reason'], result['log']
    assert trial['opportunities'] <= 12
    assert result['host'].arq.state in (arq.State.DISCONNECTED, arq.State.DISCONNECTING)
    rig, = rigs
    assert rig.stopped and rig.edges and rig.edges[-1] is False
    assert 'PTT off.' in result['log']
    a = result['host'].arq
    assert a.said_goodbye or a.goodbye_unplaceable
    if a.goodbye_unplaceable:
        assert 'goodbye could not be placed' in result['log']
    else:
        # A retained transfer names the session's ending; the QRT is on the log.
        assert 'the goodbye' in result['log'] or '-> QRT' in result['log']
    verdicts = [line for line in result['log'].splitlines() if line.startswith('verdict:')]
    assert any(line.startswith(f'verdict: P3 TIMING TRIAL {arm}:') for line in verdicts)
    if a.transfer_unconfirmed is not None:
        assert verdicts[-1] == 'verdict: transfer unconfirmed'
        retained = a.transfer_unconfirmed
        assert ((result['outdir']/'p3-transfer-unconfirmed.bin').read_bytes()
                == retained['field'] + retained['suffix'])
    else:
        assert verdicts[-1].startswith('verdict: P3 TIMING TRIAL')
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
