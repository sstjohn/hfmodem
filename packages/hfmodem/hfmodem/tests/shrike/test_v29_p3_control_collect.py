# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Cold-control collection geometry; synthetic clocks, no devices or RF."""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench


def scene():
    session = _Session(entry_pending=True)
    grid = SimpleNamespace(sending=True, keyed_slot=0,
                           _p3_keyed_reply=(0, 0, 40464, 60000))
    return session, grid


def test_cold_collection_allocates_budget_without_weakening_guard():
    session, grid = scene()
    old_close = 58080
    earlier = session.rx.control_collect_until(old_close, 56000, grid)
    assert earlier == old_close - round(.012 * onair.FS)
    live = _Bench(seconds=0)
    live.limit = 1 << 40
    peer = session.host.peer
    peer.live, peer.raster, peer.slot = live, grid, 1
    peer.key_instant = lambda raster, slot: 60000
    live.now = live.pos = old_close
    assert not session.rx._p3_acquisition_fits(.012)
    live.now = live.pos = earlier
    assert session.rx._p3_acquisition_fits(.012)


def test_first_bridge_cannot_consume_the_cold_final_reserve():
    session, grid = scene()
    first, final = 57400, 57024
    target = session.rx.control_collect_until(final, 54000, grid)
    assert session.rx.control_bridge_until(first, final, 54000, grid) == target
    assert session.rx.control_collect_until(final, target + 64, grid) == target
    session.host.arq.entry_pending = False
    session.rx._p3_answer_at = 43000
    assert session.rx.control_bridge_until(first, final, 54000, grid) == first


@pytest.mark.parametrize("change", ["cycle", "emission", "final"])
def test_collection_plan_cannot_outlive_its_receive_window(change):
    session, grid = scene()
    final = 57024
    session.rx.control_bridge_until(57400, final, 54000, grid)
    if change == "cycle":
        session.rx.new_cycle()
    elif change == "emission":
        grid._p3_keyed_reply = (1, 60000, 100464, 60000)
        grid.keyed_slot = 1
    else:
        final += 128
    assert session.rx.control_collect_until(final, final, grid) == final


@pytest.mark.parametrize("condition", ["established", "irs", "unsent",
                                       "mismatched", "consumed", "short",
                                       "head_tail"])
def test_other_collection_and_required_support_are_preserved(condition):
    session, grid = scene()
    until, now = 58080, 56000
    if condition == "established":
        session.host.arq.entry_pending = False
        session.rx._p3_answer_at = 43000
    elif condition == "irs":
        session.host.arq.role = arq.IRS
        grid.sending = False
    elif condition == "unsent":
        grid._p3_keyed_reply = None
    elif condition == "mismatched":
        grid.keyed_slot = 1
    elif condition == "consumed":
        now = until
    elif condition == "short":
        grid._p3_keyed_reply = (0, 0, 48000, 60000)
    elif condition == "head_tail":
        session.rx._p3_head_candidate = (0, 48000, -86.0)
    assert session.rx.control_collect_until(until, now, grid) == until
