# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A cycle given to the receiver on purpose, and what it may not cost.

`hold_15` of `captures/onair-0904-1659` is the single most informative window of
that arm: 1.256 s with the receiver open, where the other twenty-seven windows of
the session were 0.24 s peepholes between our own carriers. It exists because the
teardown decision happened to cost a cycle. Everything the arm wanted to know
about what was on the channel -- whether an emission has an onset, whether it
repeats on the peer's raster, whether `p3rx`'s envelope path can be reached at all
(it needs 0.6 s) -- wants that window and cannot be answered inside 0.24 s.

So `--listen-every N` schedules one, and because it changes what goes on the air
-- the peer gets a cycle with no packet in it -- it is off unless flown. What is
under test here is the price:

  * nothing is keyed, through the production transmitter and the production FSM;
  * no retry is spent, by the same `arq.REFUSED` convention a guard's drop uses,
    so a listening station cannot spend a link down by listening -- up to the
    link budget, which bounds our own refusals as well as the peer's silence,
    and which the schedule below keeps a listening arm nowhere near;
  * one slot of the hold is charged, because the ceiling is a promise about how
    long this transmitter holds a shared channel and a cycle spent listening is
    still a cycle of it;
  * and the schedule counts KEYINGS, so two silences never land together.

Run:  pytest hfmodem/tests/shrike/test_listen_cycle.py
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import arq, onair, ptc, rxfront

FS = onair.FS


class _Live:
    """The emission path's stream, reduced to a clock that is never late."""
    pos = 0
    holdback = 0

    def clamp_late(self, at):
        return 0

    def take_until(self, at):
        return np.zeros(0, np.float32)

    def wait_until(self, at):
        pass

    def sample_now(self):
        return self.pos


def _linked(tmp_path) -> tuple[ptc.PtcHost, onair.RadioTx]:
    """A station that called, was answered, and has a packet to repeat."""
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path, settle=0.040)
    tx.live = _Live()
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(rxfront.Event(
        1.0, "cs", "CS1", protocol="PACTOR-1", cs=0, sense=False))
    host.arq.on_host_data(b"the budget counts air")
    return host, tx


def test_a_listening_cycle_keys_nothing(capsys, tmp_path):
    """The whole of what goes on the air, asked of the transmitter itself."""
    host, tx = _linked(tmp_path)
    before = (tx.n, len(tx.keyings), len(list(tmp_path.glob("tx_*.wav"))))
    tx.listening = True
    host.tick()
    out = capsys.readouterr().out
    assert "LISTENING CYCLE" in out and "NOT KEYING" in out
    assert (tx.n, len(tx.keyings),
            len(list(tmp_path.glob("tx_*.wav")))) == before
    assert tx.refused


def test_a_listening_cycle_spends_no_retry(capsys, tmp_path):
    """...and the FSM is told, by the one convention that says so.

    `arq.REFUSED` back from the seam, exactly as a guard's drop returns it: the
    cycle put nothing on the channel for the peer to answer, so an unanswered
    cycle is this station's own doing and not a loss. A budget spent listening
    would end links the moment an arm asked for a look at the channel.

    Up to the link budget, which is the bound below: the schedule never takes
    two silences together, so a listening arm never approaches it.
    """
    host, tx = _linked(tmp_path)
    keyed = tx.n
    tx.listening = True
    for _ in range(host.arq.cfg.max_retries):
        host.tick()
    capsys.readouterr()
    assert host.arq._inflight is not None
    assert host.arq._inflight.retries == 0
    assert host.arq.state is arq.State.CONNECTED
    assert tx.n == keyed

    # ...and the packet is re-placed rather than paused: the next cycle keys it.
    tx.listening = False
    host.tick()
    assert tx.n == keyed + 1
    assert "LISTENING CYCLE" not in capsys.readouterr().out


def test_listening_cannot_outlast_the_link_budget(capsys, tmp_path):
    """A cycle that keys nothing is still a cycle the peer heard nothing in.

    `arq._on_nak` bounds our own refusals by the same budget it bounds the
    peer's silence with, so a station that cannot key for the whole of it signs
    off rather than holding a link it is not on. Nothing schedules that here:
    `_listen_due` never takes two silences in a row, so the count is cleared by
    the keyed cycle between any two listens.
    """
    host, tx = _linked(tmp_path)
    keyed = tx.n
    tx.listening = True
    for cycles in range(1, 4 * host.arq.cfg.max_retries):
        host.tick()
        if host.arq._qrt_pending:
            break
    capsys.readouterr()
    assert cycles == host.arq.cfg.max_retries + 2
    assert tx.n == keyed


def test_the_schedule_counts_keyings(tmp_path):
    """One cycle in N, and never two silences in a row.

    A hold spends cycles on hushes, on refused changeovers and on the listen
    cycles themselves. Counted in cycles, the count would run on through a
    silence and take the next one as well.
    """
    assert not _listens(0, 12)
    assert _listens(3, 12) == [4, 8, 12]
    assert _listens(1, 4) == [2, 4]
    # A cycle that keyed nothing does not advance the keying count, and the
    # decision is not taken twice off the same one.
    assert not onair._listen_due(3, 3, last=3)


def _listens(every: int, cycles: int) -> list[int]:
    """Which cycles of a hold `--listen-every N` would take, counting keyings."""
    keyed, last, out = 0, 0, []
    for h in range(1, cycles + 1):
        if onair._listen_due(every, keyed, last):
            last = keyed
            out.append(h)
        else:
            keyed += 1
    return out


def test_a_listening_cycle_costs_a_slot_and_buys_none(tmp_path):
    """The hold budget charges it and the idle timeout does not credit it.

    Both halves matter. The ceiling counts grid slots this station held a shared
    channel for, and a cycle spent listening is still a cycle of it. The idle
    timeout counts cycles that MOVED something, and a cycle with nothing on the
    air moved nothing -- crediting it would let an arm hold a channel open by
    listening to it.
    """
    budget = onair._HoldBudget(6)
    was = onair._Link(0, 0, arq.ISS)
    deadline = budget.deadline
    for h in range(1, 4):
        budget.spend(1)
        budget.cycle(h, was, was)
    assert budget.slots == 3
    assert budget.deadline == deadline


def test_the_flag_is_off_unless_flown():
    """It changes what goes on the air, so an arm that did not ask keeps none.

    `--listen-every` defaults to 0 and 0 is asked at the one place the hold loop
    consults it, so an unflown arm transmits in every cycle it always did.
    """
    assert not any(onair._listen_due(0, keyed, 0) for keyed in range(20))
