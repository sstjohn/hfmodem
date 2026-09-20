# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The receive window holds the peer's instant across an UNACQUIRED upgrade.

WS8EOC, 7101.5 kHz, 2026-09-11 22:14. The peer granted PACTOR-3 with `0x59A`,
we keyed ten entry packets at SL1, and it answered thirteen times in a 7 ms
spread around 1055 ms past our slot boundary -- our 960 ms PACTOR-1 packet plus
the 95-97 ms turnaround, unmoved by our own packet dropping to 810.

That session had corroborated a turnaround in cycle 11, so `rx_ref_n` had a
reference to hold. The route this covers is the other one: the anchored codeword
reader runs at the nominal `d` and reads a grant whether or not the envelope
detector ever gave `_acquire` a burst, so a link can upgrade with nothing pinned
-- and the fall-through to `packet_n` then walks the window 150 ms early, onto
an instant the peer does not key at.
"""
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.spec import Protocol

FS = onair.FS
SLOT_N = round(1.25 * FS)
P1_PACKET_N = round(0.960 * FS)
P1_CS_N = round(0.120 * FS)
D_MAX_N = round(0.130 * FS)
PULL_N = round(onair.MAX_PULL_S * FS)

ANSWERS_MS = [1054.3, 1058.3, 1059.3, 1055.3, 1055.3,
              1052.3, 1054.0, 1054.0, 1057.0, 1056.0]
"""Where the peer answered in each of the ten entry cycles, from the slot
boundary, off `working/onair-0911-2214/evening-E10-ws8eoc-p3-short-clear.log`."""

FIRST_SLOT = 21


def _grid():
    g = onair._MasterGrid(0, SLOT_N, 0, packet_n=P1_PACKET_N,
                          cs_n=P1_CS_N, d_max_n=D_MAX_N)
    g.keyed_slot = FIRST_SLOT
    return g


def _answers(grid):
    return [(FIRST_SLOT + i, grid.boundary(FIRST_SLOT + i)
             + round(ms / 1e3 * FS)) for i, ms in enumerate(ANSWERS_MS)]


def test_an_unacquired_upgrade_leaves_the_window_on_the_peers_instant():
    g = _grid()
    held = g.rx_ref_n
    line = g.keying(Protocol.PACTOR3)
    assert g.data_n == P1_PACKET_N - round(0.150 * FS)
    assert g.rx_ref_n == held
    for slot, at in _answers(g):
        assert abs(g.rx_due(slot) - at) <= PULL_N
    assert "HOLDS ITS INSTANT" in line


def test_our_own_shortened_packet_reaches_none_of_those_answers():
    """The null: the position the fall-through aims at, off the same grid."""
    g = _grid()
    g.keying(Protocol.PACTOR3)
    for slot, at in _answers(g):
        stale = g.boundary(slot) + onair.P3_PACKET_N + g.d
        assert abs(stale - at) > 5 * PULL_N


def test_the_pinned_anchor_measures_the_turnaround_the_link_was_on():
    """`_acquire` and the tracker read the same gap they read in PACTOR-1."""
    g = _grid()
    g.keying(Protocol.PACTOR3)
    for slot, at in _answers(g):
        g.keyed_slot = slot
        g.update([at])
    assert g.acquired and g.corroborated
    assert 0.090 * FS <= g.d_n <= 0.100 * FS
    slot, at = _answers(g)[-1]
    assert abs(g.rx_due(slot) - at) <= PULL_N


def test_a_peer_that_did_follow_us_still_takes_the_window_back():
    """The pin is not a lock: three cycles of bursts off our PACTOR-3 packet
    release it, which is what `_acquire`'s second pass is for."""
    g = _grid()
    g.keying(Protocol.PACTOR3)
    d_n = round(0.095 * FS)
    for slot in range(FIRST_SLOT, FIRST_SLOT + 3):
        g.keyed_slot = slot
        g.update([g.boundary(slot) + onair.P3_PACKET_N + d_n])
    assert g.d_ref_n == onair.P3_PACKET_N
    assert abs(g.rx_due(slot) - (g.boundary(slot) + onair.P3_PACKET_N + d_n)) <= PULL_N


@pytest.mark.parametrize("sending", [True, False])
def test_the_fallback_to_pactor_1_moves_the_window_no_further(sending):
    """A protocol change is not the peer's to follow in either direction, and
    the pin standing already is never overwritten by a later one."""
    g = _grid()
    g.sending = sending
    g.keying(Protocol.PACTOR3)
    pinned = g.d_ref_n
    assert pinned == (P1_PACKET_N if sending else P1_CS_N)
    g.keying(Protocol.PACTOR1)
    assert g.d_ref_n == pinned
