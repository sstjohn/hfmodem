# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""First SL2 after CS4 must reach the ACK in that packet's own cycle."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3acquire, p3frame, placement, rxfront
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid
from hfmodem.tests.shrike.test_slot_deadline import charging

FIXTURE = Path(__file__).with_name('fixtures') / 'gear-0920'
if not (FIXTURE / "metadata.json").is_file():
    pytest.skip("PACTOR SL2 gear-change recordings are not included in this distribution",
                allow_module_level=True)
ROWS = json.loads((FIXTURE / 'metadata.json').read_text())


def recorded(row):
    path = FIXTURE / row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    fs, raw = wavfile.read(path)
    assert fs == onair.FS and raw.ndim == 1
    return raw.astype(float) / 32768


def receiver(row):
    s = _Session(role=arq.IRS)
    s.rx.p3_wideband_prekey = True
    s.rx._p3_row0 = row['row0'] - 60000
    s.rx._p3_delivered_at = s.rx._p3_row0
    s.rx._p3_clock_role = arq.IRS
    s.rx.sync.packet_level = 1
    s.rx.p3_receive_offset_hz = row['offset_hz']
    s.host.arq.cfg.long_cycle = False
    s.host.arq._rx_seen = True
    s.host.arq._expected_seq = 3 if row['file'].startswith('k0nts') else 0
    return s


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['file'])
def test_recorded_first_sl2_is_decoded_before_reply(row, monkeypatch):
    s = receiver(row)
    audio = recorded(row)
    # The previous preferred-only read tries SL1 and fails on this exact PCM.
    old = rxfront.SyncedRx()
    old.packet_level, old.packet_at = 1, row['row0'] - row['start']
    assert old.packet(p3acquire.compensate(audio, row['offset_hz']), preferred_only=True) is None
    s.rx.note_p3_speedup_emitted()
    original = s.rx._read_p3_packet
    monkeypatch.setattr(s.rx, '_read_p3_packet', lambda *a, **k: pytest.fail('late/general fallback'))
    expected = s.host.arq._expected_seq
    onair._scan_frame(s.rx, audio, row['start'], tracked_only=True)
    assert len(s.packets) == 1
    assert s.packets[0].packet[:2] == (2, 0x20 | expected)
    assert s.host.peer.sent[-1] == ('cs', expected & 1)
    assert s.rx.sync.packet_level == 2
    # Overlapping re-reads must not accept this physical packet twice.
    monkeypatch.setattr(s.rx, '_read_p3_packet', original)
    s.rx.new_cycle()
    onair._scan_frame(s.rx, audio, row['start'], tracked_only=True)
    assert len(s.packets) == 1


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['file'])
def test_recorded_sl2_crc_tick_and_key_fit_original_reply_slot(row, tmp_path, monkeypatch):
    s = receiver(row)
    s.rx.note_p3_speedup_emitted()
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer, tx.sessrx = tx, s.rx
    tx.defer_p3_cs = True
    tx.reply_clock = ReplyClock()
    tx.reply_clock.entry_phase = 0
    tx.reply_clock.pulse_epoch = row['key'] + 852
    grid = irs_grid(0)
    # Preserve the recorded key instant and peer phase; do not buy time by
    # moving the reply later. This fixture uses the shipping audio-start CS.
    grid.anchor += row['key'] - grid.boundary(row['slot'])
    grid.note_p3_packet(row['row0'] - 60000 - 4320, 38880, 60000)
    bench = _Bench(seconds=100, blk=128, holdback=660, lat_in=532)
    bench._lat, bench.tx_latency_n = 1268, 960
    tx.live = bench
    tx.aim(grid, row['slot'])
    deadline = onair._p3_decode_deadline(bench, row['key'], 1920)
    ready = onair._p3_frame_ready(s.rx, deadline, bench.holdback)
    delivered = (ready - bench.lat_in) // 128 * 128
    assert delivered <= row['end']
    audio = recorded(row)[:delivered-row['start']]
    bench.audio[row['start']:delivered] = audio
    bench.now, bench.pos = delivered + bench.lat_in, delivered
    cost = {}
    with charging(bench, cost):
        onair._scan_frame(s.rx, audio, row['start'], tracked_only=True)
        assert len(s.packets) == 1
        expected = s.packets[0].packet[1] & 1
        assert tx._pending_p3_cs == expected
        start = time.perf_counter()
        s.host.tick()
        bench.spend(round((time.perf_counter()-start)*onair.FS))
        original_tx = tx._tx
        def charge_render(*args, **kwargs):
            bench.spend(round((time.perf_counter()-start)*onair.FS))
            return original_tx(*args, **kwargs)
        monkeypatch.setattr(tx, '_tx', charge_render)
        start = time.perf_counter()
        tx.emit_pending_cs()
    assert not tx.refused and tx.n == 1
    assert tx.slot == row['slot']
    assert abs(tx.tx_audio_start - row['key']) <= 240


@pytest.mark.parametrize('swapped', [False, True])
def test_bounded_sl2_reader_takes_both_orders_and_rejects_other_frames(swapped):
    cfg = placement.protocol_config()
    at = 4800 + (len(cfg.pulse())-1)//2 + p3frame.DATA_OFFSET*480
    for sl in (1, 2, 3, 4, 5, 6):
        audio = np.pad(placement.link_packet(sl, b'gear test', 3, swapped=swapped), (4800, 4800))
        sync = rxfront.SyncedRx()
        event = sync.sl2_packet_at(audio, at)
        if sl == 2:
            assert event.packet == (2, 3, b'gear test', True)
            assert event.carrier_swapped == swapped
        else:
            assert event is None
            assert sync.packet_level is None


def test_failed_sl2_trial_does_not_create_lock():
    sync = rxfront.SyncedRx()
    sync.packet_level, sync.packet_at, sync.rotation = 1, 9000, .1
    rng = np.random.default_rng(920)
    for _ in range(100):
        assert sync.sl2_packet_at(rng.normal(0, .1, 44000), 7000) is None
    assert (sync.packet_level, sync.packet_at, sync.rotation) == (1, 9000, .1)


def test_sl2_expectation_expires_and_only_emitted_cs4_arms_it(tmp_path, monkeypatch):
    s = receiver(ROWS[0])
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.attach(s.host)
    tx.sessrx = s.rx
    refuse = True
    def send(*a, **k):
        tx.refused = refuse
        tx.n += not refuse
    monkeypatch.setattr(tx, '_tx', send)
    tx._send_p3_control(3)
    assert not s.rx.sl2_prekey_expected(s.rx._p3_row0 + 60000)
    refuse = False
    tx._send_p3_control(3)
    assert s.rx.sync.packet_level == 1  # Request is not a validated lock.
    assert s.rx.sl2_prekey_expected(s.rx._p3_row0 + 60000)
    assert not s.rx.sl2_prekey_expected(s.rx._p3_row0 + 240000)
