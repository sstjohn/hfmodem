# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""
Unread CS6 copies use the last accepted counter's parity to avoid crediting
unread payload.
"""

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.p3_cycle_support import emit, scene


@pytest.mark.parametrize(
    'anchor, command, last_slot, last_end, phase, level, payload, word_at, word',
    [
        (811176, 15, 22, 2142853, 1548660, 2, b'MS via WS8EOC >\r',
         2122560, 'ACK'),
        (571274, 19, 32, 2502924, 1548960, 1, b'S via WS8EOC >\r',
         2483618, 'CYCLE-TOG'),
    ],
    ids=['ws8eoc-0923-2244', 'ws8eoc-0924-0455'],
)
def test_pending_short_controls_survive_unplaced_breakin(
        tmp_path, monkeypatch, anchor, command, last_slot, last_end, phase,
        level, payload, word_at, word):
    s, tx, grid, live = scene(tmp_path, monkeypatch, old_long=True)
    a = s.host.arq
    grid.anchor = anchor
    tx.reply_clock = grid.reply_clock = ReplyClock(
        entry_phase=1, pulse_epoch=anchor + 675)
    tx.p3_control_instant = 'peer'
    tx.p3_control_stagger = 'lead-12'
    tx.aim(grid, command)
    live.spend(grid.boundary(command) - 10000)
    row = phase + 4320
    s.rx._on(rxfront.Event(
        t=row / onair.FS, start=row, kind='packet', text='',
        protocol=spec.Protocol.PACTOR3, packet=(level, 1, payload, True),
        cycle_long=True, carrier_swapped=False))
    assert emit(s, tx, grid, live) == (command, arq.CS_CYCLE_TOG)
    assert grid._p3_interval.pending and grid._p3_interval.period == 1
    a.on_host_data(b';FW: W9SSJ\r')
    a.on_host_breakin()
    a._breakin_armed = False
    grid._p3_peer_confirmed = True
    grid.note_peer_codeword(word_at, onair.P3_CS_N, word, 'WS8EOC',
                            protocol=spec.Protocol.PACTOR3)
    tx.slots_used.append(last_slot)
    grid.keyed_slot = last_slot
    tx.tx_end = last_end
    live.spend(last_end - live.now)
    live.flush_to(last_end)
    tx.aim(grid, last_slot)
    tx.breakin_due = s.host.breakin_due
    assert tx.breakin_due
    assert grid.breakin_refusal(grid.peer_packet_end(last_slot + 1)) is not None

    keyed = []
    for expected in range(last_slot + 1, last_slot + 4):
        slot, _ = onair._regear_next_slot(
            grid, expected, a.cycle_long, stepped_ticks=1)
        tx.aim(grid, slot)
        tx.prepare_p3_breakin()
        slot = onair._p3_place_reply(grid, tx, slot)
        early = grid.boundary(slot) - tx.key_instant(grid, slot)
        slot = onair._keyable_slot(live, grid, slot, 1920 + early)
        assert slot == expected
        tx.aim(grid, slot)
        tx.arm_p3_reply()
        slot, audio, origin, _ = onair._p3_transition_window(
            live, grid, tx, s.host, s.rx, slot, np.zeros(0), live.pos, 1920)
        slot, _, _, _ = onair._regrid(
            live, grid, tx, s.host, s.rx, slot, audio, origin, 1920)
        keyed.append(emit(s, tx, grid, live))
    parity = s.host._counter_cs_for(arq.CS_REQUEST)
    assert keyed == [(slot, parity)
                     for slot in range(last_slot + 1, last_slot + 4)]
    assert a.role == arq.IRS and a.cycle_command_emitted
