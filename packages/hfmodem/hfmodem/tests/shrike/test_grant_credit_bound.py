# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""How long a station may key at a peer whose only answer acknowledges nothing.

`0x59A` in the answer slot is the gateway transmitting on our raster, so the
cycle it arrived in is not a cycle the in-flight retry budget may charge -- that
is `note_upgrade_unread`, and KB5LZK on 30 m, 2026-09-16, is what it was written
off: ten of sixteen in-link slots carried the word at zero bit errors and the
session signed off with a QRT over a gateway answering two cycles in three.

But the credit is a reset, and a reset every cycle is no bound at all: a peer
that only ever grants would hold an ISS repeating one packet for as long as it
cared to keep answering. The repo already says how long a repeated-grant
campaign may run -- `ENTRY_GRANT_CYCLES`, fourteen cycles, raised to clear the
only point a granting gateway was ever seen to react -- and that is the constant
the run of consecutive credited cycles is spent against here.

Run:  pytest hfmodem/tests/shrike/test_grant_credit_bound.py
"""
from __future__ import annotations

from hfmodem.shrike import pactor1, rxfront
from hfmodem.shrike.arq import ENTRY_GRANT_CYCLES, ISS, State
from hfmodem.tests.shrike.test_qrtack import calling_station

#: The 09-16 call, slot by slot: which of the sixteen in-link holds carried the
#: word. `test_inlink_anchored_read_0916.KB5LZK_INLINK` is where these come
#: from; the longest unanswered run in them is three.
KB5LZK_ANSWERED = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16})


def _grant(t: float) -> rxfront.Event:
    """An unassigned `0x59A` at the grid, as `_SessionRx._p1_cs` delivers one."""
    return rxfront.Event(t, "unassigned", "0x59A  (0 bit errors, PACTOR-1, "
                         "read at the grid, shift normal)",
                         protocol="PACTOR-1", spare=pactor1.CS_59A, sense=0)


def _linked():
    """A linked ISS with traffic in flight -- the KB5LZK call at its first hold."""
    host = calling_station()
    host.arq.on_host_data(b"payload " * 40)
    return host


def _cycles(host, answered) -> list[int | None]:
    """Run `answered` cycles, granting where it says so, until the link ends."""
    out = []
    for i, granted in enumerate(answered):
        if granted:
            host.on_rx_event(_grant(2.0 + i))
        host.tick()
        out.append(None if host.arq._inflight is None else host.arq._inflight.retries)
        if host.arq.state is not State.CONNECTED:
            break
    return out


def test_a_gateway_answering_every_cycle_holds_the_link_through_the_grant_budget():
    """Inside the budget the word is worth exactly what it was measured to be.

    The count reaches 1 and not 0 -- the reset lands when the word arrives and
    the tick charges the transmission that goes out behind it -- and it never
    climbs, so nothing here reaches the changeover.
    """
    host = _linked()
    seen = _cycles(host, [True] * ENTRY_GRANT_CYCLES)
    assert host.arq.state is State.CONNECTED, host.arq.state
    assert host.arq.role is ISS
    assert max(r for r in seen if r is not None) <= 1, seen
    assert not any("max retries" in ln for ln in host.log_lines), host.log_lines[-3:]


def test_past_the_budget_the_cycle_charges_again_and_the_decision_is_reached():
    """...and a peer that only ever grants does not get the channel for free.

    Past `ENTRY_GRANT_CYCLES` consecutive credited cycles the retry count runs
    as it does on any other unanswered transmission, so the budget exhausts and
    `_on_nak`'s changeover decision is taken -- declined, because nothing the
    peer sent asked for the channel, and said out loud rather than left as an
    absence.
    """
    host = _linked()
    seen = _cycles(host, [True] * (ENTRY_GRANT_CYCLES + host.arq.cfg.max_retries + 2))
    assert host.arq.state is not State.CONNECTED, seen
    assert host.arq.role is ISS, "the link reversed onto a peer that was answering"
    assert max(r for r in seen if r is not None) >= host.arq.cfg.max_retries, seen
    told = [ln for ln in host.log_lines if "NOT reversing" in ln]
    assert told, "the declined changeover was silent: " + str(host.log_lines[-3:])
    assert not any("yield the link and listen" in ln for ln in host.log_lines)


def test_the_bound_is_the_run_and_not_the_total():
    """The 09-16 call's own shape, which is what the credit exists for.

    Ten grants across sixteen slots, the longest silent stretch three -- so no
    run of credited cycles comes near the budget and the link stands, exactly as
    `test_inlink_anchored_read_0916` reads it off the arm's own recordings. A
    total rather than a run would bind here too, and this call is the one case
    the credit was written for.
    """
    host = _linked()
    _cycles(host, [hold in KB5LZK_ANSWERED for hold in range(1, 17)])
    assert host.arq.state is State.CONNECTED, \
        "the link still went down over a gateway answering 11 of 16"
    assert host.arq._inflight.retries <= 1
