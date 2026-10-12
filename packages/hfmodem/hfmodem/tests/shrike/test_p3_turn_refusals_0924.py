# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The four refusals of VE3KPG-80 0924-0033, each where the log printed it.

P3 REPLY CLOCK right after both CS3s (log 339, 1632), LISTENING CYCLE on the data
packet a CS4 released (510/512), CHANGEOVER NOT PLACED under a carried long cycle
(733), and REPLY CLOCK again once the peer went silent (1759-1771).
"""
from dataclasses import replace
from types import SimpleNamespace

from hfmodem.shrike import arq, onair, p3acquire, rxfront, spec
from hfmodem.tests.shrike.test_p3_reply_placement import (
    CYCLE, IDENTITY, PACKET_N, REPLY_N, irs_grid)
from hfmodem.tests.shrike.test_ws8eoc_long_recovery_0920 import (
    META, pending_turn, recording)

FS = onair.FS


def slot_after(g, at):
    return -(-(at + REPLY_N - g.anchor) // g.slot_n)


def iss_after_a_stint():
    """We held the link for twenty cycles; the peer's last packet is that old."""
    g = irs_grid()
    g.note_p3_packet(1_500_000, PACKET_N, CYCLE, identity=IDENTITY)
    g.reverse(to_iss=True)
    g.cycles += 20
    return g, 1_500_000 + 20 * CYCLE


def test_the_cs3_head_seats_the_reply_clock_before_its_body_decodes():
    g, head = iss_after_a_stint()
    g.note_p3_head(head, CYCLE)
    g.reverse(to_iss=False)
    slot = slot_after(g, head)
    assert g.p3_control_refusal(slot) is None
    assert g.p3_reply_phase() == head


def test_without_the_head_the_same_reply_is_refused():
    g, head = iss_after_a_stint()
    g.reverse(to_iss=False)
    assert "no fresh packet clock" in g.p3_control_refusal(slot_after(g, head))


def test_the_changeover_frame_replaces_the_head_it_arrived_behind():
    g, head = iss_after_a_stint()
    g.note_p3_head(head, CYCLE)
    g.reverse(to_iss=False)
    g.note_p3_packet(head - 17, PACKET_N, CYCLE, swapped=False, identity=IDENTITY)
    assert g._p3_peer[0] == head - 17 and g._p3_peer_swap is False
    g.note_p3_head(head - 10, CYCLE)             # the same packet's head, again
    assert g._p3_peer[0] == head - 17


def test_a_silent_peer_is_still_asked_on_its_comb():
    g = irs_grid()
    for k in range(3):
        g.note_p3_packet(1_500_000 + k * CYCLE, PACKET_N, CYCLE, identity=IDENTITY)
        g.cycles += 1
    last = 1_500_000 + 2 * CYCLE
    g.cycles += onair.RASTER_PROJECT_CYCLES + 4
    slot = slot_after(g, last + (onair.RASTER_PROJECT_CYCLES + 4) * CYCLE)
    assert g.p3_control_refusal(slot) is None


def test_a_reversal_off_the_long_cycle_regears_the_grid_short():
    g = irs_grid()
    g.reverse(to_iss=True)
    g.regear(True)
    assert g.cycle_long
    host = SimpleNamespace(protocol=spec.Protocol.PACTOR3, arq=SimpleNamespace(
        state=arq.State.CONNECTED, entry_pending=False, role=arq.IRS,
        cycle_long=False, unconfirmed_breakin=False))
    line = onair._grid_reversal(g, host)
    assert "GRID REVERSED -> IRS" in line and "cycle length -> short" in line
    assert not g.cycle_long


def test_a_cs4_answering_our_changeover_ends_the_cede(tmp_path):
    s, tx, g = pending_turn(tmp_path)
    tx.aim(g, 37)
    tx.live.now = tx.live.pos = 2248700
    tx.listening = tx.cede_listening = True
    phase = META['controls'][0]
    pcm = p3acquire.compensate(recording('repeat-controls.wav')[:16800], -13.1)
    ev = rxfront.SyncedRx().control_signal_at(pcm, META['control_lead'])
    s.rx._on(replace(ev, t=phase / FS, cs=arq.CS_SPEED_UP), anchored=True)
    assert not tx.listening
    assert not s.host.arq.unconfirmed_breakin


def test_the_cede_names_itself_when_it_refuses(tmp_path, capsys):
    s, tx, g = pending_turn(tmp_path)
    tx.aim(g, 37)
    tx.listening = tx.cede_listening = True
    tx.send_packet(1, b';FW', 0, breakin=True)
    out = capsys.readouterr().out
    assert "LISTENING CYCLE" in out and "arq.breakin_listen_due" in out
    assert "--listen-every" not in out
