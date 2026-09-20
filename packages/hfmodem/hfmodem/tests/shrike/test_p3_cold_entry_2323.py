# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Captured cold WS8EOC entry must establish a CRC clock before a mail ACK."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

ROOT = Path(__file__).resolve().parents[5]
CAP = ROOT / 'captures/onair-0919-2323'
ENTRY, ORIGIN, FIRST_HEAD = 1204493, 1244305, 1248745


@pytest.fixture(scope='module')
def recording():
    path = CAP / 'hold_06.wav'
    if not path.exists():
        pytest.skip('recorded WS8EOC 23:23 hold 6 is unavailable')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        'ce6c51ecd46789bfacb6f767c8cc9accf63e323ebbe3b1c3e7a84019832f5ff0')
    fs, raw = wavfile.read(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert fs == onair.FS and raw.dtype == np.int16
    assert meta['end_stream_sample']-len(raw) == ORIGIN
    return raw[:, 0].astype(float)/32768


def cold_mail(tmp_path):
    s = _Session(role=arq.ISS, entry_pending=True)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    s.host.arq.cfg.long_cycle = False
    live = _Bench(seconds=60)
    grid = onair._MasterGrid(ENTRY-900, 60000, 8880, packet_n=46080,
                            cs_n=5760, d_max_n=6240)
    grid.protocol = spec.Protocol.PACTOR3
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.raster, tx.sessrx = live, grid, s.rx
    tx.reply_clock = ReplyClock()
    tx.defer_p3_cs = True
    tx.p3_follow_offset = 'all'
    tx.entry_delay_n = 234
    tx.aim(grid, 0)
    live.now = live.pos = ENTRY-60000
    tx.send_entry_packet(1,b'',0x1a)
    assert not tx.refused and len(live.emissions)==1
    assert tx.reply_clock.entry_phase == ENTRY
    assert s.rx.p3_receive_offset_hz == 0
    assert s.rx._p3_row0 is None and grid._p3_peer is None
    tx.aim(grid, 2)
    return s,tx,live,grid


@pytest.mark.parametrize('cold_grid', [False, True])
def test_cold_capture_establishes_current_crc_then_emits_mail_ack(
        recording,tmp_path,monkeypatch,cold_grid):
    s,tx,live,grid = cold_mail(tmp_path)
    # First complete recorded body, scanned early with the next slot in front
    # of it. No CFO, packet phase, speed, or CRC has been seeded into the RX.
    if not cold_grid:
        fits = s.rx._p3_acquisition_fits
        monkeypatch.setattr(s.rx,'_p3_acquisition_fits',
                            lambda reserve: reserve != s.rx.P3_COLD_RESERVE_S and fits(reserve))
    first = recording[:60000]
    live.now = live.pos = ORIGIN+len(first)
    onair._scan_frame(s.rx,first,ORIGIN,tracked_only=False)
    if not cold_grid:
        assert not s.packets and s.rx._p3_row0 is None
        assert s.rx.p3_receive_offset_hz == 0 and grid._p3_peer is None
        tx.emit_pending_cs()
        assert len(live.emissions)==1
        return
    assert s.packets and s.packets[-1].packet[:3] == (1,0,b'RMS')
    assert -17 < s.rx.p3_receive_offset_hz < -13
    assert s.host.arq.role == arq.IRS and not s.host.arq.entry_pending
    assert tx._pending_p3_cs == arq.CS_ACK
    onair._grid_reversal(grid,s.host)
    tx.aim(grid,2)
    onair._p3_place_reply(grid,tx,2)
    # The following physical repetition is decoded with the acquired CFO on
    # the retained clock before emission, using only captured samples.
    s.rx.new_cycle()
    live.now = live.pos = ORIGIN+len(recording)
    onair._scan_frame(s.rx,recording,ORIGIN,tracked_only=True)
    assert len(s.packets)==2
    assert abs(s.rx._p3_delivered_at-(FIRST_HEAD+60000)) <= 120
    tx.emit_pending_cs()
    assert not tx.refused and len(live.emissions)==2
    pulse = tx.tx_audio_start+min(tx.tx_pulse_offsets)
    assert (pulse-ENTRY)%60000 == 28950
    assert tx._pending_p3_cs is None
    sidecar=json.loads((tmp_path/'tx_02.json').read_text())
    assert sidecar['audio_start']+min(sidecar['pulse_offsets'])==pulse


def test_erased_body_cannot_authorize_ack_from_head_only(recording,tmp_path):
    s,tx,live,grid = cold_mail(tmp_path)
    first=recording[:60000].copy()
    first[FIRST_HEAD-ORIGIN+21*480:] = 0
    live.now=live.pos=ORIGIN+len(first)
    onair._scan_frame(s.rx,first,ORIGIN,tracked_only=False)
    assert not s.packets and s.rx._p3_row0 is None and grid._p3_peer is None
    # A readable CS3 head can yield IRS but supplies no CRC packet clock.
    s.rx.control_signal(first,ORIGIN,FIRST_HEAD)
    assert not s.packets
    assert s.host.arq.role == arq.IRS
    onair._grid_reversal(grid,s.host)
    tx.aim(grid,2)
    onair._p3_place_reply(grid,tx,2)
    s.host.tick()
    assert tx._pending_p3_cs == arq.CS_REQUEST
    tx.emit_pending_cs()
    assert tx.refused and len(live.emissions)==1
    assert grid._p3_peer is None
