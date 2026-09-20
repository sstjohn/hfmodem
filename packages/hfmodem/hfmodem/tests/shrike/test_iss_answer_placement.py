# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The first acknowledgement a PACTOR-3 ISS is owed, and where the next packet goes.

`captures/onair-0913-1837` is the arm this file is built from. It connected,
upgraded, took the whole `CMS via WS8EOC >` greeting as IRS -- a stint that needs
no codeword from the peer at all -- and then broke in to send the Winlink login.
From that instant the station read nothing: `_p3_cs`'s acquiring search was gated
on `entry_pending`, spent as soon as the entry was confirmed, so the only reader
pointed at the peer's answer was the anchored one and the only anchor it had was
the PACTOR-1 turnaround measured at 22.36 s. 0 PACTOR-3 control signals decoded
in 377 s, against 57 PACTOR-3 frames; three of about sixty-five login bytes went
out once and the rest stayed in `PactorArq._outbuf`.

The geometry is why the anchor could not be aimed. The retained reference puts
the answer a 960 ms PACTOR-1 packet plus the turnaround past our boundary, and a
PACTOR-3 packet is 810 ms -- so the codeword arrives about 150 ms in FRONT of the
instant being read, five times `SyncedRx`'s bracket, on a carrier the anchored
read is not told about.

What the scene drives is the hold loop's own seam, in its own order:
`_MasterGrid.rx_due_in` for the instant, `_SessionRx.control_signal` for the
word, `_forecast_next_key` to hand it to the grid, and `RadioTx.aim` for the
next cycle's carrier. The coordinates are `test_p3_reply_placement`'s recorded
comb, reversed to ISS.

Run:  python -m pytest hfmodem/tests/shrike/test_iss_answer_placement.py
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import onair, rx, rxfront, spec
from hfmodem.shrike.arq import CS_ACK, ISS
from hfmodem.tests.shrike.test_entry_answer import _Session, _answer
from hfmodem.tests.shrike.test_p3_reply_placement import (CYCLE, IDENTITY,
                                                          PACKET_N, driver,
                                                          irs_grid)

FS = onair.FS
PEER = 1_500_000
"""The greeting packet's phase, on `irs_grid`'s comb."""

TURNAROUND_N = 4413
"""The turnaround that grid measured in PACTOR-1, and the peer still answers in.

It is the same physical switching time either side of the upgrade; what changed
is the packet in front of it, which is why the retained anchor misses.
"""

CS_PHASE_N = (rx._pulse(rxfront.SPS).size - 1) // 2
"""Where a reader's phase reference sits inside a rendered control signal.

`placement.control_signal` is matched-filtered, so its first data symbol is a
pulse delay into the audio and that is the sample `_p3_cs` reports -- never the
first sample of the burst. The residual inside `SPS // 2` is the half-symbol
stagger `spec.SUBBAND_LEAD` puts between the two carriers.
"""


def iss_after_the_turn(tmp_path):
    """The station of 18:37: greeting read, link taken, no P3 answer ever measured."""
    g = irs_grid()
    g.note_p3_packet(PEER, PACKET_N, CYCLE, swapped=False, identity=IDENTITY)
    tx = driver(g, tmp_path)
    g.reverse(to_iss=True)
    sess = _Session(role=ISS, entry_pending=False)
    assert sess.rx._p3_answer_at is None
    return g, tx, sess


def peer_answers(g, slot: int, burst: np.ndarray):
    """The cycle's audio, with the peer's codeword where PACTOR-3 puts it."""
    at = g.boundary(slot) + g.data_n + TURNAROUND_N
    seg = np.zeros(CYCLE, np.float32)
    seg[at - g.boundary(slot):at - g.boundary(slot) + burst.size] += burst
    return seg, g.boundary(slot), at


def seam(g, tx, sess, slot: int, burst: np.ndarray):
    """One cycle of the hold loop, from our key to the aim of the next one."""
    tx.aim(g, slot)
    g.keyed_slot = slot
    seg, seg_start, at = peer_answers(g, slot, burst)
    due = g.rx_due_in(seg_start, seg_start + seg.size)
    sess.rx.new_cycle()
    heard = sess.rx.control_signal(seg, seg_start, due)
    if heard is not None:
        onair._forecast_next_key(sess.rx, tx, g, seg_start)
    g.cycles += 1
    return heard, due, at


def test_the_iss_reads_its_first_p3_answer_and_keys_the_next_packet_from_it(
        tmp_path, capsys):
    """The arm's own seam, with the acquiring search reachable after the turn.

    The first cycle produces a candidate and nothing else -- a body-less word
    is not a clock until a second physical cycle puts it at the same phase and
    the same offset. The second delivers it, and from there the ISS is keying
    against a transmission it has READ: `peer_at` is the codeword's own sample,
    `_bare_peer_width` gives the 210 ms a PACTOR-3 control lasts rather than a
    packet's 810, and the next carrier comes up `peer_read_gap` past the end of
    that transmission with `key_refusal` silent.

    `peer_at` is the codeword's phase reference and its audio begins
    `CS_PHASE_N` in front of that, which is why the gap is measured against the
    burst the scene keyed rather than against the reading of it.
    """
    g, tx, sess = iss_after_the_turn(tmp_path)
    burst = _answer(CS_ACK)
    slot = (PEER + CYCLE - g.anchor) // CYCLE
    assert g.peer_at == PEER, \
        "the premise: the newest transmission the grid holds is the greeting"

    heard, due, at = seam(g, tx, sess, slot, burst)
    assert due - at == round(spec.P1_PACKET_S * FS) - g.data_n, \
        "the premise: the retained PACTOR-1 anchor is a packet's difference late"
    assert heard is None and sess.rx._p3_answer_at is None
    assert "awaiting a distinct corroborating cycle" in capsys.readouterr().out

    heard, due, at = seam(g, tx, sess, slot + 1, burst)
    assert heard == CS_ACK
    assert sess.rx.p3_receive_offset_hz == 15.0
    assert abs(sess.rx._p3_answer_at - (at + CS_PHASE_N)) <= rxfront.SPS // 2
    assert [ev.cs for ev in sess.events if ev.kind == "cs"] == [CS_ACK]

    assert g.peer_cs.protocol == spec.Protocol.PACTOR3
    assert g.peer_at == sess.rx._p3_answer_at
    assert g._bare_peer_width(g.peer_at) == g.cs_n

    tx.aim(g, slot + 2)
    assert tx.boundary - (at + g.cs_n) == g.peer_read_gap
    assert g.key_refusal(tx.boundary, g.data_n) is None


def test_the_acquisition_is_spent_once_and_the_anchor_follows_the_peer(tmp_path):
    """...and the cycles after it are read at the instant this one measured.

    The search is a 460 ms bracket over 31 frequencies in front of the key. It
    buys the peer's clock, and `_p3_cs` reads there from the next cycle on --
    which is what makes running it on a link with no measured answer affordable
    rather than a standing cost.
    """
    g, tx, sess = iss_after_the_turn(tmp_path)
    burst = _answer(CS_ACK)
    slot = (PEER + CYCLE - g.anchor) // CYCLE
    for k in range(2):
        seam(g, tx, sess, slot + k, burst)
    acquired = sess.rx._p3_answer_at

    for k in range(2, 5):
        heard, _, at = seam(g, tx, sess, slot + k, burst)
        assert heard == CS_ACK, f"cycle {k} went unanswered"
        assert abs(sess.rx._p3_answer_at - (acquired + (k - 1) * CYCLE)) \
            <= rxfront.SPS // 2


def test_a_changeover_retry_is_still_refused_on_the_answer_it_just_read(tmp_path):
    """What the read does NOT buy, named where the arm spent 88 cycles on it.

    The 18:37 station went on offering the same changeover packet because
    `arq.PactorArq._on_nak` retransmits an inflight burst with the break-in flag
    it was built with. That placement stands on `peer_read_gap`, which without a
    corroborated PACTOR-3 packet/control phase pair is the retained PACTOR-1
    turnaround wearing a PACTOR-3 cycle's arithmetic -- and `breakin_refusal`
    declines it, correctly, whatever the acquiring search has read.

    A codeword answering our packet is not that pair and cannot become one: the
    peer's last transmission is 210 ms of control, not a packet, and
    `note_peer_codeword` clears the old pair for exactly that reason. So the
    acknowledged login advances as an ordinary data packet or not at all, which
    is an `arq` question and is not settled here.
    """
    g, tx, sess = iss_after_the_turn(tmp_path)
    burst = _answer(CS_ACK)
    slot = (PEER + CYCLE - g.anchor) // CYCLE
    for k in range(2):
        seam(g, tx, sess, slot + k, burst)
    assert sess.rx._p3_answer_at is not None

    tx.aim(g, slot + 2)
    end = g.peer_packet_end(slot + 2)
    assert end.at == g.peer_at and end.end == g.peer_at + g.cs_n
    assert "no fresh corroborated" in g.breakin_refusal(end)
    assert tx._place_breakin() == "" and not tx.placed
