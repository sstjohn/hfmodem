# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded skipped sends and repeated changeovers, on an elapsed sample clock."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_cs6_stalls import _duplex

FIX = Path(__file__).parent / 'fixtures/ws8eoc-2008'
META = json.loads((FIX / 'metadata.json').read_text())
FS = 48000


def pcm(row):
    p = FIX / row['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    return rxfront.load_wav(str(p))


def sender(tmp_path, row):
    d = row['previous']
    s = _Session(role=arq.ISS)
    a = s.host.arq
    a.cycle_long = 'LONG' in d['control']
    a.speed_level = int(re.search(r'SL(\d)', d['control'])[1])
    status = int(re.search(r'status=0x(\w+)', d['control'])[1], 16)
    size = int(re.search(r'pkt (\d+)B', d['control'])[1])
    a._inflight = arq._Packet(status=status, payload=b'X' * size, sl=a.speed_level)
    a._next_seq, a._buffer_raw = (status + 1) % 4, 1000 + size
    a._outbuf.extend(b'A' * 1000)
    a.payload_ledger.queue(a._inflight.payload + bytes(a._outbuf))
    g = onair._MasterGrid(0, 60000, 0, packet_n=38880, cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, True
    g.regear(a.cycle_long)
    pulse = d['audio_start'] + min(d['pulse_offsets'])
    pulse -= (pulse - d['p3_mail_reply_epoch'] + 30000) % 60000 - 30000
    g.anchor = pulse - d['slot'] * 60000 - 852
    g.d_n, g.d_ref_n = 4800, 46320 + (120000 if a.cycle_long else 0)
    g.keyed_slot = d['slot']
    g._p3_keyed_reply = (d['slot'], d['audio_start'], d['audio_end'], g.cycle_n)
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path, settle=.04)
    tx.reply_clock = g.reply_clock = ReplyClock(
        entry_phase=2284590, pulse_epoch=d['p3_mail_reply_epoch'])
    s.host.peer = tx
    tx.attach(s.host)
    tx.sessrx = s.rx
    tx.slots_used = [d['slot']]
    tx.aim(g, g.next_slot(d['slot']))
    s.rx.p3_receive_offset_hz = -9.1 if row['tx'] < 48 else -8.9
    s.rx._p3_answer_at = row['phase'] - g.cycle_n
    clock = _duplex(tx, s, np.zeros(0), 2612)
    clock.holdback, clock.tx_latency_n, clock._blk = 660, 960, 128
    clock._callback_timing = [(128*i, 128, i*.00001, i/375) for i in range(3)]
    return s, tx, g, clock


@pytest.mark.parametrize('row', META['controls'], ids=lambda r: f"after-tx{r['tx']}")
def test_recorded_late_work_never_costs_two_extra_long_slots(tmp_path, monkeypatch, row):
    s, tx, g, clock = sender(tmp_path, row)
    slot = tx.slot
    old_close = tx.boundary - onair._prekey_lead(clock, 1920)
    ready = s.rx.control_collect_until(old_close, row['phase'] + 10000, g)
    assert ready <= old_close
    # Never shorten a window already too tight for full tracking support.
    # Otherwise preserve every symbol, filter support and tracking radius.
    end = ready - clock.holdback
    audio = pcm(row)[:end - row['start']]
    if ready < old_close:
        assert end - row['phase'] >= 20 * 480 + rxfront.MATCHED_DELAY_N + 480
    # Reproduce the recorded total overrun, charged at rendering rather than
    # relying on machine speed. Include the observed 3-callback delivery batch.
    pulse = g.boundary(slot) + 852
    old_samples = old_close - clock.holdback + 384
    old_audio = pulse - (666 if s.host.arq.speed_level == 1 else 654)
    work = max(0, old_audio - old_samples - clock.key_notice) + round(row['late_ms'] * 48)
    render = placement.link_packet
    def charged(*args, **kwargs):
        result = render(*args, **kwargs)
        clock.samples += work
        return result
    monkeypatch.setattr(placement, 'link_packet', charged)
    clock.samples = end + 384
    heard, _ = s.rx.control_signal_in(audio, row['start'], g)
    assert heard == row['cs']
    packet = s.host.arq._inflight
    assert packet is not None
    if tx.refused:
        # A real overrun must return promptly to RX/regrid, not sleep 7.5 s
        # waiting for the rendered carrier arrangement to recur.
        assert clock.samples < g.boundary(slot) + 2400
        assert not clock.emissions
        tx.aim(g, g.next_slot(slot))
        clock.samples = tx.boundary - 10000
        s.host.tick(elapsed_ticks=g.ticks, cycle_ticks=g.ticks)
        assert s.host.arq._inflight is packet
    assert not tx.refused and len(clock.emissions) == 1
    assert tx.slots_used[-1] in (slot, slot + g.ticks)
    error = (tx.tx_audio_start + min(tx.tx_pulse_offsets) - pulse) % g.cycle_n
    assert min(error, g.cycle_n - error) <= 240


@pytest.mark.parametrize('to_long', [False, True])
def test_period_changed_by_tick_corrects_already_advanced_next_slot(tmp_path, to_long):
    s, tx, g, clock = sender(tmp_path, META['controls'][0])
    g.regear(not to_long)
    s.host.arq.cycle_long = to_long
    tx.aim(g, 209)
    step = g.ticks
    next_slot = g.next_slot(209)  # Driver advances before host.tick renders.
    tx._prepare_p3_cycle(to_long)
    next_slot, _ = onair._regear_next_slot(g, next_slot, to_long, stepped_ticks=step)
    assert next_slot == 209 + (3 if to_long else 1)


@pytest.mark.parametrize('row', META['changeovers'], ids=lambda r: str(r['hold']))
def test_current_changeover_crc_is_read_before_speculative_data(tmp_path, monkeypatch, row):
    s = _Session(role=arq.IRS)
    r = s.rx
    r.p3_receive_offset_hz = -8.9
    r._seed_p3_changeover_clock(row['phase'] - 60000)
    r.sync.packet_level = 1
    g = onair._MasterGrid(0, 60000, 0, packet_n=38880, cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    d = row['tx']
    g.anchor = d['audio_start'] - d['slot'] * 60000
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path, settle=.04)
    tx.reply_clock = g.reply_clock = ReplyClock(entry_phase=2284590, pulse_epoch=2371140)
    s.host.peer = tx
    tx.attach(s.host)
    tx.sessrx, tx.defer_p3_cs = r, True
    tx.aim(g, d['slot'])
    clock = _duplex(tx, s, np.zeros(0), 2612)
    clock.samples = row['end']
    ordinary = r.sync.sl1_packet_at
    def expensive(*args, **kwargs):
        clock.samples += 288  # 6 ms on an ordinary-header trial that fails.
        return ordinary(*args, **kwargs)
    monkeypatch.setattr(r.sync, 'sl1_packet_at', expensive)
    onair._scan_frame(r, pcm(row), row['start'], tracked_only=True)
    assert abs(r._p3_delivered_at - row['phase']) <= 120
    assert r._p3_prekey_crc == (r._p3_delivered_at, d['slot'])
    assert tx._pending_p3_cs is not None
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used == [d['slot']]


@pytest.mark.parametrize('damage', ['erase-body', 'truncate-head'])
def test_changeover_without_valid_body_cannot_authorize_ack(tmp_path, damage):
    row = META['changeovers'][0]
    s = _Session(role=arq.IRS)
    r = s.rx
    r.p3_receive_offset_hz = -8.9
    r._seed_p3_changeover_clock(row['phase'] - 60000)
    audio = pcm(row).copy()
    at = row['phase'] - row['start']
    if damage == 'erase-body':
        audio[at + 24 * 480:] = 0
    else:
        audio = audio[:at + 10 * 480]
    corrected = r._corrected(audio)
    ev = rxfront._cs_event(corrected, arq.CS_BREAKIN, 0, at, at / FS, '')
    assert ev.packet is None
