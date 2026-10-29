# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC 0611 long receive geometry with a hypothetical short CS3 reply."""
import numpy as np
import pytest
from hfmodem.shrike import arq, onair, placement, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex


@pytest.mark.parametrize('delay_ms', [115, 130, 145])
@pytest.mark.parametrize('stale_answer', [False, True])
def test_long_irs_short_breakin_can_read_first_short_reply(tmp_path, delay_ms, stale_answer):
    # Recorded coordinates translated by 15,000,000 samples; the answer below
    # is synthetic. The recording contains no clean ACK to this break-in.
    grid = onair._MasterGrid(29326, 60000, 8880, packet_n=38880,
                            cs_n=10080, d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, False
    grid.d_n, grid.d_ref_n = 4800, 38880
    grid.acquired = grid.corroborated = True
    grid.regear(True)
    grid.cycles = 17
    identity = (False, 1, 0x43, b'last packet')
    grid.note_p3_packet(707061, 158160, 180000, identity=identity)
    grid.note_p3_control(870178)
    grid.note_p3_packet(887061, 158160, 180000, identity=identity)
    clock = ReplyClock(entry_phase=1, pulse_epoch=1050178)
    grid.reply_clock = clock
    tx = duplex(grid, tmp_path)
    tx.reply_clock = clock
    session = _Session(role=arq.IRS)
    session.host.arq._inflight = arq._Packet(0, b'FF\r', 1, breakin=True)
    session.host.arq._buffer_raw = 3
    session.host.arq.payload_ledger.queue(b'FF\r', b'FF\r')
    session.host.arq._next_seq = 1
    session.host.arq._cycle_long = True
    session.host.peer = tx
    tx.attach(session.host)
    tx.aim(grid, 17)
    tx.breakin_due = True
    tx._flip = lambda: False
    tx.live.now = tx.live.pos = 1045000
    tx.send_packet(1, b'FF\r', 0, breakin=True)
    assert not tx.refused
    pulse = tx.tx_audio_start+min(tx.tx_pulse_offsets)
    assert pulse == 1050178
    grid.reverse(to_iss=True)
    session.host.arq.role = arq.ISS
    session.rx.new_cycle()
    phase = tx.tx_end+round(delay_ms*48)
    if stale_answer:
        session.rx._p3_answer_at = phase-600000+9600
    origin, end = tx.tx_end, phase+15000
    audio = np.zeros(end-origin, np.float32)
    raw = placement.control_burst(0, tail=False, swapped=None)
    start = phase-int(np.argmax(placement.protocol_config().pulse()))-origin
    audio[start:start+len(raw)] = raw
    tx.aim(grid, 20)
    tx.live.now = tx.live.pos = end+660
    heard, _ = session.rx.control_signal_in(audio, origin, grid)
    assert heard == 0
    assert abs(round(session.rx.cs_log[-1].t*48000)-phase) < 480
