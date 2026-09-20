# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A rung is asked for on measured legibility, not on a count of accepts.

WS8EOC, 80 m, 2026-09-13 21:35 (`working/pactor3-header-0913/ledger-80m`). The
link took the greeting at speed level 1 and read 5 of the 13 cycles the peer
transmitted in -- 38%. At k=19 `_gear_cs` printed `clean run -> ask the peer for
SL2`: `speed_up_after` had closed on three accepted packets spread over
seventeen cycles, twelve of which read nothing at all. The peer delivered SL2 at
k=20 and from there we read 1 cycle of 35, while a witness 160 miles from the
gateway decoded a CRC-valid frame in every one of them and the same SL2 block
repeated 33 times. The link went down with the gateway still calling us.

The count could not see that, because a tally of accepts measures how many
packets got through and never what fraction did. The run is now consecutive
cycles -- a repeat, an idle field or a cycle nothing was read in ends it -- so
what reaches `speed_up_after` is a stretch of the peer's transmissions that all
arrived, which is the only evidence this end has about the level above.

`--p3-speed-up hold` is the arm-level override for a path already known to be
marginal, and it binds both gear seams: the stalled-run exit of
`--p3-repeat-gear` keys the same CS4 and the peer climbs on it the same way.

Everything below drives the real `PtcHost` seam and the real cycle tick, so the
codewords asserted on are the physical ones the counter alternation produces.

Run:  python -m pytest hfmodem/tests/shrike/test_gear_gate_speed_up.py
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import arq
from hfmodem.tests.shrike.test_repeat_gear_stall import (STALL_SL, arrives,
                                                         linked, status_at)

#: What the ledger measured at speed level 1 before the rung was asked for:
#: three packets accepted, and four cycles read nothing in between each pair.
UNREAD_BETWEEN = 4


def linked_at(speed_up: str = "auto", **kw):
    host, seam = linked(**kw)
    host.arq.cfg.speed_up = speed_up
    return host, seam


def delivers(host, seq: int) -> None:
    """One cycle of the peer's carrying a new packet, and its tick."""
    arrives(host, status=status_at(seq, long_cycle=False),
            field=bytes([0x41 + seq]))
    host.arq.on_cycle()


def unread(host, cycles: int = 1) -> None:
    """Cycles the peer transmitted in and this receiver read nothing."""
    for _ in range(cycles):
        host.arq.on_cycle()


# -- the run is cycles, not accepts ---------------------------------------


def test_three_accepts_with_unread_cycles_between_them_draw_no_rung():
    """The 21:35 arm, at its own cadence: 38% legibility asks for nothing."""
    host, seam = linked_at(speed_up_after=3)
    for seq in (1, 2, 3):
        delivers(host, seq)
        unread(host, UNREAD_BETWEEN)
    assert arq.CS_SPEED_UP not in seam.words
    assert host.arq.state is arq.State.CONNECTED
    assert host.arq.rx_progress == 3


def test_three_consecutive_cycles_do_draw_it():
    """The positive control, and the only evidence the level above gets."""
    host, seam = linked_at(speed_up_after=3)
    for seq in (1, 2, 3):
        delivers(host, seq)
    assert seam.words[-1] == arq.CS_SPEED_UP
    assert host.arq.rx_progress == 3


def test_one_unread_cycle_is_enough_to_end_the_run():
    """The gap is not a budget: legibility is the whole claim CS4 makes."""
    host, seam = linked_at(speed_up_after=3)
    delivers(host, 1)
    delivers(host, 2)
    unread(host)
    delivers(host, 3)
    assert arq.CS_SPEED_UP not in seam.words
    delivers(host, 0)
    delivers(host, 1)
    assert seam.words[-1] == arq.CS_SPEED_UP


def test_a_repeat_between_accepts_ends_the_run_too():
    """A repeat proves the peer is there and nothing about the next level."""
    host, seam = linked_at(speed_up_after=3)
    delivers(host, 1)
    delivers(host, 2)
    arrives(host, status=status_at(2, long_cycle=False), field=b"B")
    host.arq.on_cycle()
    delivers(host, 3)
    assert arq.CS_SPEED_UP not in seam.words
    assert host.arq.rx_progress == 3


def test_an_idle_field_ends_it_as_well():
    host, seam = linked_at(speed_up_after=3)
    delivers(host, 1)
    delivers(host, 2)
    arrives(host, status=status_at(3, long_cycle=False), field=b"")
    host.arq.on_cycle()
    delivers(host, 0)
    assert arq.CS_SPEED_UP not in seam.words


# -- hold binds both seams ------------------------------------------------


def test_hold_keys_no_rung_from_the_climb():
    host, seam = linked_at("hold", speed_up_after=3)
    for seq in (1, 2, 3, 0, 1, 2):
        delivers(host, seq)
    assert arq.CS_SPEED_UP not in seam.words
    assert host.arq.rx_progress == 6


def test_hold_keys_no_rung_from_the_repeat_seam_either():
    """The stalled-run exit is a rung too, and the peer climbs on it."""
    host, seam = linked_at("hold", repeat_gear=3)
    for _ in range(8):
        arrives(host)
        host.arq.on_cycle()
    assert seam.words == [arq.CS_REQUEST] * 8
    assert host.arq.rx_progress == 1


def test_the_repeat_stall_under_hold_is_the_transcript_the_arm_keyed():
    """36 copies of `SL1 status=0x21 seq=1`, answered CS2 every time.

    Held, the stall ends where it ended before `--p3-repeat-gear` existed: on
    `_silent_cycles` once the peer stops, and on the caller's hold budget while
    it has not. Nothing here asks a link reading 38% of its peer for SL2.
    """
    host, seam = linked_at("hold", repeat_gear=3, speed_up_after=3)
    for _ in range(36):
        arrives(host)
    assert seam.words == [arq.CS_REQUEST] * 36
    assert arq.CS_SPEED_UP not in seam.words
    assert host.arq.speed_level == STALL_SL
    assert host.arq.state is arq.State.CONNECTED

    cycles = 0
    while host.arq.state is arq.State.CONNECTED and cycles < 40:
        host.arq.on_cycle()
        cycles += 1
    assert host.arq.cfg.max_retries < cycles <= host.arq.cfg.max_retries + 3


@pytest.mark.parametrize("repeat_gear", [0, 3])
def test_speed_hold_still_allows_long_cycle_negotiation(repeat_gear):
    """Holding speed suppresses CS4, while an enabled long-cycle request gets CS6."""
    host, seam = linked_at("hold", repeat_gear=repeat_gear,
                           speed_up_after=1, long_cycle=True)
    arrives(host)
    assert seam.words == [arq.CS_CYCLE_TOG]
    assert host.arq.cycle_request is True
    assert not host.arq.cycle_long  # A grant is not an observed transition.

    for _ in range(8):
        arrives(host, status=status_at(2), field=b"ode", cycle_long=True)
    assert seam.words == [arq.CS_CYCLE_TOG] + [arq.CS_ACK] * 8
    assert host.arq.cycle_long and host.arq.cycle_request is None
    assert host.arq.speed_level == STALL_SL
    assert host.arq.rx_progress == 2
    assert bytes(host.channel(host.ptchn).rx) == b" Trimode"


@pytest.mark.parametrize("speed_up", ["auto", "hold"])
def test_the_peers_own_gear_commands_are_untouched(speed_up):
    """Hold is about what THIS station asks for. M.1798 gives the IRS the
    commands, and against a peer that is the IRS we are the end that obeys."""
    host, seam = linked_at(speed_up, speed_up_after=3)
    host.arq.role = arq.ISS
    host.arq.speed_level = 1
    host.arq.on_rx_cs(arq.CS_SPEED_UP)
    assert host.arq.speed_level == 2


def test_the_default_is_auto():
    assert arq.ArqConfig.speed_up == "auto"
