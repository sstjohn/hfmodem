# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One late reading does not latch the short-cycle reply later.

WS8EOC 2026-11-07 05:11Z: the gateway read our ACKs at +66.6..+68.5 ms past its
packet and advanced on copy 1-2. At TX[80] one frame was read a single 2.5 ms
search step late; the floor of that day judged the reply against that one
reading, moved the whole clock +5.7 ms, and nothing brought it back. The next
99 ACKs sat at +74 ms and the gateway needed 5, 13, 11, 31 and 34 copies.

The peer-relative aim follows the smoothed raster in both directions. The
default control instant holds our own raster while that aim still places CS3.
"""
import numpy as np
import pytest

from hfmodem.shrike import onair, placement
from hfmodem.shrike.p3trial import ReplyClock, CYCLE, ROTATION
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive

WIDTH = 39120
# Stint 2's readings as offsets from the raster they settled on, on the frame
# search's 120-sample grid; frame 9 (index 8) reads one step late.
READINGS = [-120, -240, -120, -240, -120, -120, 0, 0, 120] + [0] * 20
LATE = 8


def _replies(tmp_path, instant):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    if instant == 'peer':
        tx.p3_control_instant = instant
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    receive(s, tx, live, grid, entry + 44400)
    origin = grid._p3_peer[0]
    past = []
    pulses, clocks = [], []
    lead = placement.historical_control_pulse_lead(0)
    for index, offset in enumerate(READINGS, 2):
        grid.cycles = index
        grid.note_p3_packet(origin + (index - 1) * CYCLE + offset, WIDTH, CYCLE,
                            swapped=None, identity=(False, 1, index % 4, b'test'))
        tx.aim(grid, index)
        onair._p3_place_reply(grid, tx, index)
        end = (grid._peer_raster_position(grid._p3_peer[0], CYCLE)
               or grid._p3_peer[0]) + WIDTH
        past.append((grid.boundary(index) - end) / onair.FS * 1e3)
        pulses.append(grid.boundary(index) + lead - index * CYCLE - entry)
        clocks.append(tx.reply_clock.pulse_epoch)
    return past, pulses, clocks


@pytest.mark.parametrize('instant', ['raster', 'peer'])
def test_one_late_reading_does_not_latch_the_reply_later(tmp_path, instant):
    past, pulses, _ = _replies(tmp_path, instant)
    # A moving peer reference cannot measure the stability of raster controls.
    # Peer placement keeps its smoothed aim.
    if instant == 'raster':
        assert pulses == [ROTATION] * len(READINGS)
    else:
        before = np.mean(past[4:LATE])
        after = past[LATE:]
        assert max(abs(p - before) for p in after) <= 1.0, past
        assert abs(np.mean(after[-10:]) - before) <= 0.5, past


def test_late_reading_keeps_same_breakin_clock_under_both_instants(tmp_path):
    peer_path, raster_path = tmp_path / 'peer', tmp_path / 'raster'
    peer_path.mkdir()
    raster_path.mkdir()
    _, peer_pulses, peer_clock = _replies(peer_path, 'peer')
    _, raster_pulses, raster_clock = _replies(raster_path, 'raster')
    assert raster_clock == peer_clock
    assert raster_pulses != peer_pulses
