# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The IRS reply follows the peer's packets, each receiving stint on its own raster.

Measured 2026-09-20 on WS8EOC (1015, 1119, 1233): the peer's raster came back
601.7-607 ms on per round trip while the mail clock rotated by 600.0, so every
later stint's reply landed 2-7 ms earlier against the packet it answered.
"""
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_p3_reply_placement import (
    CYCLE, FS, IDENTITY, PACKET_N, REPLY_N, irs_grid)

QUANTUM = 120  # the frame search reads a packet to 2.5 ms on this station
ROTATION = 28800


def reply_error(g, slot, onset):
    """The reply boundary against the fixed delay past the peer's TRUE onset."""
    n = (g.boundary(slot) - onset - REPLY_N) % g.slot_n
    return n - g.slot_n if n > g.slot_n // 2 else n


@pytest.mark.parametrize("quantum", [30, QUANTUM])
@pytest.mark.parametrize("ms_per_min", [1.0, 3.0])
def test_a_drifting_peer_is_followed_within_a_millisecond(ms_per_min, quantum):
    """Within 1 ms of the truth at the fine quantum; the coarse one bounds the
    reading itself at 1.25 ms and the tracker adds no more than 0.25 ms to it."""
    g = irs_grid()
    per_cycle = ms_per_min / 60 * 1.25 * FS / 1000
    tracked, read = [], []
    for k in range(80):
        true = 1_500_000 + k * CYCLE + k * per_cycle
        reading = round(true / quantum) * quantum
        g.note_p3_packet(reading, PACKET_N, CYCLE, identity=IDENTITY)
        g.p3_reply_shift(30)
        tracked.append(abs(reply_error(g, 30, round(true))))
        read.append(abs(reading - round(true)))
        g.cycles += 1
    assert g._p3_raster_run == 80  # smoothed, never re-originated
    assert max(tracked[4:]) <= max(read) + FS // 4000
    assert quantum > 2 * FS // 1000 or max(tracked[4:]) <= FS // 1000


def test_a_step_past_the_pull_re_origins():
    g = irs_grid()
    for k in range(4):
        g.note_p3_packet(1_500_000 + k * CYCLE, PACKET_N, CYCLE, identity=IDENTITY)
        g.cycles += 1
    stepped = 1_500_000 + 4 * CYCLE + round(0.030 * FS)
    g.note_p3_packet(stepped, PACKET_N, CYCLE, identity=IDENTITY)
    assert (g._p3_raster_origin, g._p3_raster_run) == (stepped, 1)
    g.p3_reply_shift(30)
    assert reply_error(g, 30, stepped) == 0


def test_a_reversal_reseats_the_raster_from_the_new_stints_first_packet():
    g = irs_grid()
    for k in range(3):
        g.note_p3_packet(1_500_000 + k * CYCLE, PACKET_N, CYCLE, identity=IDENTITY)
        g.cycles += 1
    g.p3_reply_shift(30)
    assert g._p3_raster_corroborated
    g.reverse(to_iss=True)
    g.cycles += 10
    g.reverse(to_iss=False)
    assert g._p3_raster_origin is None and g.p3_reply_phase() is None
    assert g.p3_reply_shift(40) is None  # the rotation stands until a packet seats it
    back = 1_500_000 + 13 * CYCLE + ROTATION + round(0.007 * FS)
    g.note_p3_packet(back, PACKET_N, CYCLE, identity=IDENTITY)
    assert (g._p3_raster_origin, g._p3_raster_run) == (back, 1)
    g.p3_reply_shift(40)
    assert reply_error(g, 40, back) == 0


def test_the_changeover_that_turns_the_link_seats_the_new_stint():
    g = irs_grid()
    g.reverse(to_iss=True)
    g.cycles += 6
    changeover = 1_500_000 + 6 * CYCLE
    g.note_p3_packet(changeover, PACKET_N, CYCLE, identity=IDENTITY)
    g.reverse(to_iss=False)
    assert (g._p3_raster_origin, g._p3_raster_run) == (changeover, 1)
    assert g.p3_reply_phase() == changeover


def test_mail_reply_keeps_its_gap_past_the_packet_in_every_stint(tmp_path):
    from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive
    s, tx, live, g = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock()
    tx.reply_clock.entry(entry)
    receive(s, tx, live, g, entry + 44400)
    tx.aim(g, 1)
    onair._p3_place_reply(g, tx, 1)
    tx.emit_pending_cs()
    assert not tx.refused
    pulse = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    lead = pulse - tx.tx_audio_start
    first = g._p3_peer[0]
    gap = (pulse - first) % CYCLE
    assert tx.reply_clock.gap == gap
    g.reverse(to_iss=True)
    g.cycles += 12
    g.turn_accepted = True
    g.reverse(to_iss=False)
    # The peer's raster returns 602.5 ms on; the rotation alone is 600.0.
    second = first + 12 * CYCLE + ROTATION + QUANTUM
    g.note_p3_packet(second, PACKET_N, CYCLE, identity=IDENTITY)
    tx.aim(g, 15)
    onair._p3_place_reply(g, tx, 15)
    assert (g.boundary(15) + lead - second) % CYCLE == gap
