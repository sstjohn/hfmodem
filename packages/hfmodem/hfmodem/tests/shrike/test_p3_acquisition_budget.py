# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the fall-through changeover acquisition must have left before it runs.

`_SessionRx._p3_acquisition_fits` is the one admission check in front of the
only read that delivered a PACTOR-3 packet on either 2026-09-13 arm, and it was
spending its reserve against `clamp_late`, which allows `key_notice` and nothing
else. Every other read in the loop stops at `_p3_decode_deadline`, six
milliseconds earlier; this one did not, so it was admitted with about a tenth of
a millisecond in hand and an overrun cost the transmit slot.

The room figures below are `onair-0913-0014`'s own, measured against the
deadline the loop enforces, on the four receive windows that report one. Three
of them are the crops under `fixtures/ws8eoc-0913-*`.
"""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

FS = onair.FS
CYCLE = 60000
SLOT = 10
DECODER_RESERVE_N = round(onair.P3_DECODE_RESERVE_S * FS)
ARM_NOTICE_N = 1652        # `onair-0913-0014`'s own `clocks:` line

# hold 38, 39, 44, 46 of `onair-0913-0014`: how far the window closed INSIDE
# `_p3_decode_deadline`. Negative is a window that closed past it.
ARM_ROOM_MS = (-31.4, 2.6, -3.4, 0.6)


@pytest.fixture
def seam(tmp_path):
    """A transmitter aimed at a slot, and the receiver's view of its clock."""
    g = onair._MasterGrid(0, CYCLE, 8880, packet_n=46080, cs_n=5760,
                          d_max_n=6240)
    g.d_n, g.d_ref_n = 4413, 46080
    g.keyed_slot = 0
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path,
                       settle=.04)
    tx.live, tx.raster = _Bench(seconds=120), g
    tx.aim(g, SLOT)
    rx = SimpleNamespace(host=SimpleNamespace(peer=tx))
    return tx, g, rx


def admits(tx, rx, reserve_s, at):
    """The check as it now stands: `_p3_decode_deadline` less the reserve."""
    tx.live.now = tx.live.pos = at
    return onair._SessionRx._p3_acquisition_fits(rx, reserve_s)


def admitted_before(tx, reserve_s, at):
    """The check as it stood: `clamp_late(key_instant - reserve)`, bare."""
    tx.live.now = tx.live.pos = at
    return not tx.live.clamp_late(
        tx.key_instant(tx.raster, SLOT) - round(reserve_s * FS))


def last_admitted(tx, rule):
    """The latest clock position `rule` still lets the acquisition run at."""
    key = tx.key_instant(tx.raster, SLOT)
    lo, hi = key - CYCLE, key
    assert rule(lo) and not rule(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if rule(mid) else (lo, mid)
    return lo


def test_the_acquisition_now_keeps_the_reserve_every_other_read_keeps(seam):
    """`clamp_late` allows `key_notice`; the loop's reads keep six more.

    So the flip lands a reserve plus that six milliseconds plus the DAC's own
    notice in front of the key, where before it was the reserve and the notice.
    """
    tx, g, rx = seam
    key, blk = tx.key_instant(g, SLOT), tx.live._blk
    reserve_n = round(onair._SessionRx.P3_ACQUIRE_RESERVE_S * FS)
    now = last_admitted(
        tx, lambda at: admits(tx, rx, onair._SessionRx.P3_ACQUIRE_RESERVE_S, at))
    owed = reserve_n + DECODER_RESERVE_N + tx.live.key_notice
    # `_Bench.samples` is block-quantised, as the codec's delivered count is
    # and always downwards, so the flip lands inside one block LATE of the
    # instant the arithmetic names and never early of it.
    assert -blk < (key - now) - owed <= 0
    # ...and the refinement behind it gets the same treatment, one reserve later.
    fine = last_admitted(
        tx, lambda at: admits(tx, rx, onair._SessionRx.P3_FINE_RESERVE_S, at))
    assert now - fine == pytest.approx(
        round((onair._SessionRx.P3_FINE_RESERVE_S
               - onair._SessionRx.P3_ACQUIRE_RESERVE_S) * FS), abs=blk)


def test_the_reserves_cover_the_searches_they_guard():
    """Measured, single worker on this machine, and both constants clear it.

    `p3acquire.changeover` on the coarse list: 9.2-17.9 ms over the thirty-five
    receive windows of `onair-0913-0014`, and 10.7, 13.1, 14.3 ms median on the
    three crops of that arm in `fixtures/ws8eoc-0913-*`. The 5 Hz refinement
    behind it reaches 26.7 ms on its own and 29.9 ms measured with the coarse
    pass in front of it on the same three crops.
    """
    assert onair._SessionRx.P3_ACQUIRE_RESERVE_S * 1e3 >= 17.9
    assert onair._SessionRx.P3_FINE_RESERVE_S * 1e3 >= 29.9
    assert onair._SessionRx.P3_FINE_RESERVE_S > onair._SessionRx.P3_ACQUIRE_RESERVE_S


@pytest.mark.parametrize("room_ms", ARM_ROOM_MS)
def test_the_reserve_costs_the_arms_own_windows_no_acquisition(room_ms, seam):
    """The change declines nothing on 0913-0014 that `.018` did not decline first.

    The best of the four windows closed 2.6 ms inside the deadline the loop
    enforces -- 40.4 ms in front of the key, on that arm's 34.4 ms of DAC notice
    -- against an old rule that wanted 18 ms on top of the notice and a new one
    that wants 25 plus 6. Both refuse all four. Raising the constant and
    carrying the decoder reserve therefore costs this arm no acquisition at all;
    what it buys is on the windows that DO have room, where the acquisition used
    to be admitted with about a tenth of a millisecond to spend.
    """
    tx, g, rx = seam
    tx.live._lat = ARM_NOTICE_N - 3 * tx.live._blk      # the arm's own clocks
    key = tx.key_instant(g, SLOT)
    close = key - tx.live.key_notice - DECODER_RESERVE_N - round(
        room_ms / 1e3 * FS)
    assert not admitted_before(tx, .018, close)
    assert not admits(tx, rx, onair._SessionRx.P3_ACQUIRE_RESERVE_S, close)


def test_the_band_the_new_rule_declines_and_the_old_admitted(seam):
    """And here is what it does cost, measured rather than left to be found."""
    tx, g, rx = seam
    blk = tx.live._blk
    before = last_admitted(tx, lambda at: admitted_before(tx, .018, at))
    now = last_admitted(
        tx, lambda at: admits(tx, rx, onair._SessionRx.P3_ACQUIRE_RESERVE_S, at))
    assert before - now == pytest.approx(
        DECODER_RESERVE_N + round(
            (onair._SessionRx.P3_ACQUIRE_RESERVE_S - .018) * FS), abs=blk)
    # Every clock position between the two is one the old rule admitted and this
    # one declines: 13.0 ms of a 1250 ms cycle.
    for at in (now + 1, (now + before) // 2, before):
        assert admitted_before(tx, .018, at)
        assert not admits(tx, rx, onair._SessionRx.P3_ACQUIRE_RESERVE_S, at)
