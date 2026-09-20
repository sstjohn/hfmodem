# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""First-slot ACKs for VE3KPG's three recorded reversals, without RF devices."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3acquire, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_cs6_stalls import ElapsedPCM, _buffered_listen
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Rig

FIXTURES = Path(__file__).with_name('fixtures') / 've3kpg-changeover-0920'
if not (FIXTURES / "metadata.json").is_file():
    pytest.skip("VE3KPG off-air fixtures are not included in this distribution",
                allow_module_level=True)
ROWS = json.loads((FIXTURES / 'metadata.json').read_text())
FS = 48000


class Capture(ElapsedPCM):
    holdback = 660
    _blk = 128
    _callback_timing = ((0, 128, 0.0, 0.0), (128, 128, .00004, .002667),
                        (256, 128, .00008, .005333))

    def take_until(self, until):
        # The real capture read trails the converter. Model callback batching
        # as well as time spent decoding; neither may be rewound by a read.
        end = min(len(self.audio), max(self.pos, int(until)))
        if end > self.pos:
            self.samples = max(self.samples, end + 384)
        audio = self.audio[self.pos:end]
        self.pos = end
        return audio

    def sample_now(self):
        return self.samples + self.holdback


def arm(tmp_path, row, damage=False):
    fs, pcm = wavfile.read(FIXTURES / row['file'])
    assert fs == FS
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == row['pcm_sha256']
    audio = pcm.astype(float) / 32768
    head = row['head'] - row['origin']
    if damage:
        audio[head + 12000:] = 0
    s = _Session(role=arq.ISS)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    s.host.arq.cfg.long_cycle = False
    s.rx.p3_receive_offset_hz = row['offset_hz']
    ack = row['first_ack_audio'] - row['origin']
    g = onair._MasterGrid(ack - 28800, 60000, 0, packet_n=38880,
                         cs_n=10080, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.acquired = g.corroborated = g.turn_accepted = True
    g.d_n, g.d_ref_n = 4800, 38880
    # The already emitted mail phase rotates with this accepted ISS turn.
    clock = ReplyClock(entry_phase=0, pulse_epoch=ack + 852 - 28800)
    g.reply_clock = clock
    tx = onair.RadioTx(_Rig(), transmit=True, settle=.04, outdir=tmp_path)
    tx.reply_clock = clock
    tx.attach(s.host)
    s.host.peer = tx
    live = Capture(audio, 1536)
    tx.live, tx.raster, tx.sessrx = live, g, s.rx
    tx.defer_p3_cs = True
    tx.p3_follow_offset = 'all'
    tx.aim(g, 0)
    # Decode only the head, as the ISS control window does. Its end must not
    # contain the body the new IRS is about to collect.
    live.pos = head + 12000
    live.samples = live.pos
    seg = audio[:live.pos]
    ev = s.rx.sync.control_signal_tracked(s.rx._corrected(seg), head, details=False)
    assert ev is not None and ev.cs == arq.CS_BREAKIN and ev.packet is None
    s.rx._on(ev, anchored=True)
    assert s.host.arq.role == arq.IRS
    return s, tx, live, g, seg


@pytest.mark.parametrize('row,payload', list(zip(ROWS, (b'RMS', b'FS ', b'***'))))
@pytest.mark.parametrize('reserve_s', (0.0, onair.PREKEY_RESERVE_S))
def test_recorded_changeover_ack_uses_first_absolute_slot(
        tmp_path, monkeypatch, row, payload, reserve_s):
    s, tx, live, g, seg = arm(tmp_path, row)
    delivered = []
    s.host.deliver = delivered.append
    def no_search(*args, **kwargs):
        pytest.fail('a known CS3 must not reacquire time/frequency before its ACK')
    monkeypatch.setattr(p3acquire, 'changeover', no_search)
    decode = rxfront._cs_event
    def charged_decode(*args, **kwargs):
        result = decode(*args, **kwargs)
        live.samples += round(s.rx.CHANGEOVER_BODY_RESERVE_S * FS)
        return result
    monkeypatch.setattr(rxfront, '_cs_event', charged_decode)
    seg = onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0,
                                    np.zeros(0), 0, 1920, reserve_s=reserve_s)
    assert b''.join(delivered) == payload
    assert s.rx.frame_seen and s.rx._p3_clock_role == arq.IRS
    assert tx._pending_p3_cs == arq.CS_ACK
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    slot, _, _ = onair._regrid(live, g, tx, s.host, s.rx, 0, seg, 0, 1920)
    assert slot == 0
    s.host.tick()
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used == [0]
    assert len(live.emissions) == 1
    assert tx.tx_audio_start + row['origin'] == row['first_ack_audio']
    # A later pass over the same physical burst neither redelivers nor queues
    # another ACK, while the fresh clock survives the role-change bookkeeping.
    s.rx.changeover_body(seg, 0, s.rx.cs_at)
    assert b''.join(delivered) == payload
    s.rx.new_cycle()
    assert s.rx._p3_row0 is not None


@pytest.mark.parametrize('row', ROWS)
def test_valid_head_with_erased_body_is_never_acked(tmp_path, row):
    s, tx, live, g, seg = arm(tmp_path, row, damage=True)
    onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0,
                              np.zeros(0), 0, 1920)
    assert not s.packets
    assert s.rx._p3_delivered_at is None
    assert tx._pending_p3_cs != arq.CS_ACK
    tx.emit_pending_cs()
    assert not live.emissions


def test_following_recorded_data_gets_current_counter_ack(tmp_path):
    # Preserve the actual recording's six-cycle interval before seq1. This is
    # not a claim about what the gateway would send after our corrected ACK.
    row = ROWS[2]
    s, tx, live, g, seg = arm(tmp_path, row)
    delivered = []
    s.host.deliver = delivered.append
    onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0,
                              np.zeros(0), 0, 1920)
    onair._p3_place_reply(g, tx, 0)
    tx.emit_pending_cs()
    assert not tx.refused
    s.rx.new_cycle()
    meta = json.loads((FIXTURES / 'error-next.json').read_text())
    _, pcm = wavfile.read(FIXTURES / 'error-next.wav')
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == meta['pcm_sha256']
    audio = pcm.astype(float) / 32768
    origin = meta['origin'] - row['origin']
    tx.aim(g, 6)
    live.audio = np.pad(live.audio, (0, origin + len(audio) - len(live.audio)))
    live.audio[origin:origin + len(audio)] = audio
    live.pos = live.samples = origin
    until = onair._p3_frame_ready(
        s.rx, onair._p3_decode_deadline(live, tx.key_instant(g, 6), 1920),
        live.holdback)
    audio = live.take_until(until)
    onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    assert b''.join(delivered) == b'*** Erro'
    assert tx._pending_p3_cs == arq.CS_REQUEST  # physical CS2, odd-counter ACK
    onair._p3_place_reply(g, tx, 6)
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used == [0, 6]
    assert abs(tx.tx_audio_start + row['origin']
               - row['first_ack_audio'] - 6 * 60000) <= 1


def test_diagnostic_exposes_whole_cycle_delay(tmp_path, capsys):
    row = ROWS[2]
    s, tx, live, g, seg = arm(tmp_path, row)
    onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0,
                              np.zeros(0), 0, 1920)
    tx.tx_audio_start = row['first_ack_audio'] - row['origin'] + 60000
    tx.tx_pulse_offsets = (852, 852)
    tx._ack_gap_line()
    assert '1 whole cycle(s)' in capsys.readouterr().out


def test_already_buffered_body_is_delivered_without_another_read(tmp_path):
    s, tx, live, g, _ = arm(tmp_path, ROWS[2])
    delivered = []
    s.host.deliver = delivered.append
    live.pos = live.samples = s.rx.cs_at + round(.834 * FS)
    seg = live.audio[:live.pos]
    before = live.pos
    onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0,
                              np.zeros(0), 0, 1920)
    assert live.pos == before
    assert b''.join(delivered) == b'***'
    assert tx._pending_p3_cs == arq.CS_ACK


def test_expired_reply_never_forces_a_late_ack(tmp_path, monkeypatch):
    s, tx, live, g, seg = arm(tmp_path, ROWS[2])
    live.samples = ROWS[2]['first_ack_audio'] - ROWS[2]['origin']
    def no_late_decode(*args, **kwargs):
        pytest.fail('body work must not consume an expired reply opportunity')
    monkeypatch.setattr(rxfront, '_cs_event', no_late_decode)
    onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0,
                              np.zeros(0), 0, 1920)
    assert tx._pending_p3_cs != arq.CS_ACK
    assert not live.emissions
