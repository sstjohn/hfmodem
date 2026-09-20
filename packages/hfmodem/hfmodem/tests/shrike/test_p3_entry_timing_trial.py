# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Entry-only trial across actual TX, CRC reception and guarded teardown seams."""
import json
import numpy as np
import pytest
from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import EntryTimingTrial, FS, CYCLE
from hfmodem.tests.shrike.test_p3_timing_trial import receive
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig


def entry_setup(tmp_path, arm):
    s = _Session(role=arq.ISS, entry_pending=True)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    s.host.arq.cfg.long_cycle = False
    live = _Bench(seconds=60)
    grid = onair._MasterGrid(10*FS, CYCLE, 8880, packet_n=46080, cs_n=5760, d_max_n=6240)
    grid.protocol = spec.Protocol.PACTOR3
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.raster, tx.sessrx = live, grid, s.rx
    tx.defer_p3_cs = True
    tx.p3_follow_offset = 'none'
    tx.timing_trial = EntryTimingTrial(arm)
    tx.entry_delay_n = round(tx.timing_trial.delay_ms*48)
    tx.aim(grid, 0)
    live.now = live.pos = 9*FS
    tx.send_entry_packet(1, b'', 0x1a)
    assert not tx.refused
    return s, tx, live, grid


def test_entry_b_changes_only_epoch_at_actual_emission(tmp_path):
    results = []
    for arm in ('A', 'B'):
        path = tmp_path/arm
        path.mkdir()
        s, tx, live, grid = entry_setup(path, arm)
        meta = json.loads((path/'tx_01.json').read_text())
        results.append((tx.timing_trial.entry_phase, live.emissions[-1], meta))
    assert results[1][0]-results[0][0] == 234
    assert results[1][2]['audio_start']-results[0][2]['audio_start'] == 234
    # Compare captured DAC reference samples; the trial changes no waveform.
    from scipy.io import wavfile
    assert np.array_equal(wavfile.read(tmp_path/'A/tx_01.wav')[1],
                          wavfile.read(tmp_path/'B/tx_01.wav')[1])


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_crc_changeover_stops_before_ack_and_preserves_teardown(tmp_path, arm):
    s, tx, live, grid = entry_setup(tmp_path, arm)
    t = tx.timing_trial
    before = len(live.emissions)
    receive(s, tx, live, grid, t.entry_phase+round(.925*FS))
    assert t.reason == 'CRC-valid inbound P3 packet'
    tx.emit_pending_cs()
    assert tx.refused and len(live.emissions) == before
    assert onair._timing_trial_close(tx, s.host, live.now)
    assert t.closing and s.host.arq._qrt_pending
    assert onair._timing_trial_close(tx, s.host, live.now) is None
    t.write(tmp_path)
    result = json.loads((tmp_path/t.filename).read_text())
    assert result['p3_crc_verified'] and result['entries_before_first_p3'] == 1
    assert not result['progression_tested']


def test_retries_do_not_extend_deadline_and_last_guard_refuses(tmp_path):
    s, tx, live, grid = entry_setup(tmp_path, 'B')
    t = tx.timing_trial
    first = t.entry_phase
    tx.aim(grid, 2)
    tx.send_entry_packet(1, b'', 0x1a)
    assert len(t.entries) == 2
    assert t.check(first+20*FS-1) is None
    live.now = live.pos = first+20*FS
    before = len(live.emissions)
    tx.aim(grid, 18)
    tx.send_entry_packet(1, b'', 0x1a)
    assert tx.refused and len(live.emissions) == before
    assert len(t.entries) == 2 and t.reason == '20 second entry-acquisition limit'
    assert onair._timing_trial_close(tx, s.host, live.now)


def test_control_only_confirmation_is_not_crc_success(tmp_path):
    s, tx, live, grid = entry_setup(tmp_path, 'A')
    s.host.arq.entry_pending = False
    assert tx._timing_trial_refusal()
    assert tx.timing_trial.reason == 'entry state confirmed; no inbound CRC yet'
    tx.timing_trial.write(tmp_path)
    assert not json.loads((tmp_path/tx.timing_trial.filename).read_text())['p3_crc_verified']


def test_local_disconnect_is_not_peer_confirmation(tmp_path):
    s, tx, live, grid = entry_setup(tmp_path, 'A')
    s.host.arq.on_host_disconnect()
    s.host.arq.entry_pending = False
    onair._entry_trial_control_confirmation(tx, s.host)
    assert tx.timing_trial.reason is None


def test_pre_entry_audio_and_own_echo_do_not_open_trial():
    t = EntryTimingTrial('B')
    t.packet(FS, 0, b'RMS', breakin=True, long_cycle=False)
    t.entry(2*FS, 1)
    for phase in (FS, 2*FS, 2*FS+4000):
        t.packet(phase, 0x1a, b'', breakin=False, long_cycle=False)
    assert not t.packets and t.reason is None
    t.packet(2*FS+round(.925*FS), 0, b'RMS', breakin=True, long_cycle=False)
    assert len(t.packets) == 1
