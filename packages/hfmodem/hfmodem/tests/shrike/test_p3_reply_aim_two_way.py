# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One late reading does not latch the short-cycle reply later.

WS8EOC 2026-11-07 05:11Z: the gateway read our ACKs at +66.6..+68.5 ms past its
packet and advanced on copy 1-2. At TX[80] one frame was read a single 2.5 ms
search step late; the floor of that day judged the reply against that one
reading, moved the whole clock +5.7 ms, and nothing brought it back. The next
99 ACKs sat at +74 ms and the gateway needed 5, 13, 11, 31 and 34 copies.

The reply now sits on the aim past the smoothed raster, so the late reading
moves it a quarter of a step and the next readings take that back.
"""
import numpy as np

from hfmodem.shrike import onair
from hfmodem.shrike.p3trial import ReplyClock, CYCLE
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive

WIDTH = 39120
# Stint 2's readings as offsets from the raster they settled on, on the frame
# search's 120-sample grid; frame 9 (index 8) reads one step late.
READINGS = [-120, -240, -120, -240, -120, -120, 0, 0, 120] + [0] * 20
LATE = 8


def _replies(tmp_path):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    receive(s, tx, live, grid, entry + 44400)
    origin = grid._p3_peer[0]
    past = []
    for index, offset in enumerate(READINGS, 2):
        grid.cycles = index
        grid.note_p3_packet(origin + (index - 1) * CYCLE + offset, WIDTH, CYCLE,
                            swapped=None, identity=(False, 1, index % 4, b'test'))
        tx.aim(grid, index)
        onair._p3_place_reply(grid, tx, index)
        end = (grid._peer_raster_position(grid._p3_peer[0], CYCLE)
               or grid._p3_peer[0]) + WIDTH
        past.append((grid.boundary(index) - end) / onair.FS * 1e3)
    return past


def test_one_late_reading_does_not_latch_the_reply_later(tmp_path):
    past = _replies(tmp_path)
    before = np.mean(past[4:LATE])
    after = past[LATE:]
    # The late reading moves the reply only by the tracker's quarter gain, and
    # it settles back where it was: no +5.7 ms step and no climb.
    assert max(abs(p - before) for p in after) <= 1.0, past
    assert abs(np.mean(after[-10:]) - before) <= 0.5, past
