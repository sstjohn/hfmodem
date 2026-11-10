# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Every IRS reply to a data packet stands on the answer slot of the packet it
answers; a reply to the peer's changeover stands on our own raster
(`test_p3_changeover_own_raster_1108`).

VE3KPG 2026-11-07 15:39Z, on a Kiwi witness 139 mi off: the gateway took our
TX16 and TX17 with their phase reference 889.9 and 890.2 ms past its packet's,
then refused TX19-54 at 901.4-904.8 ms, 35 of 35, after `P3 MAIL reply moved
+7.021 ms onto the emitted entry/turn clock`: they stood 69.5 ms past
"reading + 39120", 902.25 ms on the packet's own reference. Its data packets
read 240 samples earlier on its comb than the changeover copies before them. The
witness agrees with our own readings to the tenth of a millisecond: the pulse
centre we logged less the CRC frame we read is the delay the gateway saw.
"""
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p3trial import ROTATION, ReplyClock
from hfmodem.tests.shrike.test_p3_reply_armed_ahead import (
    CYCLE, WIDTH, frame, mail_link, owe, top_of_cycle)
from hfmodem.tests.shrike.test_p3_timing_trial import receive, setup

FS = onair.FS
ANSWER_N = round(0.890 * FS)          # stock's 889.4-891.9; taken through 899, refused from 901


def answered(s, tx, live, grid, slot):
    """Key `slot`'s reply and return its phase reference."""
    slot = top_of_cycle(tx, live, grid, slot)
    owe(s, tx, tx._p3_armed.cs)
    live.now = live.pos = grid.boundary(slot) - 5000
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used[-1] == slot
    return tx.tx_audio_start + min(tx.tx_pulse_offsets)


def turned(tmp_path):
    """The gateway's stint opened by its changeover, as VE3KPG's was."""
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    receive(s, tx, live, grid, entry + 44400)
    return s, tx, live, grid, grid._p3_peer[0]


@pytest.mark.parametrize('step', [-240, 240])
def test_every_data_reply_stands_890_ms_past_the_packet_it_answers(tmp_path, step):
    """Changeover copies on our own raster, then data packets `step` samples
    off them on the peer's comb: -240 is VE3KPG's, +240 the other way."""
    s, tx, live, grid, phase = turned(tmp_path)
    entry = tx.reply_clock.entry_phase
    for k in range(2, 9):
        form = (True, 1) if k < 4 else (False, 1)
        phase += CYCLE + (step if k == 4 else 0)
        grid.cycles = k
        grid.note_p3_packet(phase, WIDTH, CYCLE, swapped=None,
                            identity=(*form, k % 4, b'test'))
        centre = answered(s, tx, live, grid, k + 1)
        if k < 4:
            assert (centre - entry) % CYCLE == ROTATION, k
        else:
            assert abs(centre - (phase + CYCLE) - ANSWER_N) <= FS // 1000, (k, centre - phase - CYCLE)


def test_the_reply_never_brings_its_ptt_inside_the_packet(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    tx.settle = 0.080
    phase = origin + index * CYCLE
    grid.cycles = index + 1
    frame(grid, phase, index + 1)
    centre = answered(s, tx, live, grid, index + 2)
    assert tx.tx_key_up >= phase + CYCLE + WIDTH
