# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VE3KPG 80 m 2026-11-17 06:42Z: the entry re-keyed behind an answer we held as a candidate.

`operator-data/working/onair-20261117T064249676838Z-a80982a4-…`: TX[15], the
entry at slot 14, drew the candidate ACK at 18.538 s (log 232); TX[16], the
entry again at slot 15, drew the ACK at 19.78 s that corroborated it (log 249).
Our first data packet, TX[17] at slot 16, alternated from slot 15 and went out
swapped; the gateway, counting from the entry it had answered, asked for counter 3
seventeen times and never read one of our packets.
"""
import pytest

from hfmodem.shrike import arq
from hfmodem.shrike.p3trial import CYCLE
from hfmodem.tests.shrike.test_p3_entry_iss_phase import bare_ack, check_packet, entry_link


def answered_then_rekeyed(tmp_path, reference, grid_parity):
    session, tx, live, grid = entry_link(tmp_path, grid_parity=grid_parity)
    tx.p3_entry_swap_reference = reference
    entry = tx.reply_clock.entry_phase
    rx = session.rx
    rx.new_cycle()                      # slot 14's answer window: a candidate ACK
    head = grid.boundary(0) + (entry - grid.anchor) % CYCLE + 45000
    rx._p3_head_candidate = (rx._p3_acquisition_cycle, head, 0.0, arq.CS_ACK)
    tx.aim(grid, 1)                     # slot 15: the entry again
    live.now = live.pos = entry + CYCLE - 6000
    tx.send_entry_packet(1, b'', 0x1a)
    assert not tx.refused and session.host.arq.entry_pending
    assert bare_ack(session, tx, live, grid, 2) == arq.CS_ACK
    assert not session.host.arq.entry_pending and not tx.refused
    packet, pulse = check_packet(tx, entry)
    assert pulse == entry + 2 * CYCLE   # slot 16
    return packet


@pytest.mark.parametrize('grid_parity', [False, True])
def test_data_alternates_from_the_entry_the_gateway_answered(tmp_path, grid_parity):
    assert not answered_then_rekeyed(tmp_path, 'answered', grid_parity).carrier_swapped


@pytest.mark.parametrize('grid_parity', [False, True])
def test_last_restores_the_flown_reference(tmp_path, grid_parity):
    assert answered_then_rekeyed(tmp_path, 'last', grid_parity).carrier_swapped
