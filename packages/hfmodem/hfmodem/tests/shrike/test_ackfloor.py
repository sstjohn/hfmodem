# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Which burst the grid believes is the peer, benched over the corpus's pairs.

A reading opens the receive window; it does not by itself become the session's
turnaround. `_TurnaroundEvidence` holds that to two candidates from different
cycles inside `SPAN` agreeing to `TOL_N`, and to there being no SECOND gap in
the window the same could be said of -- because two witnessed turnarounds is a
thing a master cannot resolve from timing at all.

What it costs when that rule is missing is measured rather than argued.
`_acquire` used to take the first onset of a cycle whose gap fell in the band,
so arrival order decided between candidates: benched 2026-08-14 over all 899
(spurious 41-69 ms, real 90-120 ms) pairs with both bursts in every cycle, the
spurious one won all 899. The receive window then opens on an echo while the
peer answers 40 ms away, `key_refusal` measures phase against that same echo,
and the tracker never looks at the real burst again -- it only searches inside
`MAX_PULL_S` once `d` is set.

`D_MIN_S` is the other gate and the near one: a gap no station could have
turned around in is not a peer. KB5LZK's break-ins on 2026-08-14 came back at
raw 28-37 ms and every one was refused; so was the 39.1 ms echo that took the
2026-08-10 link down.

Run:  pytest hfmodem/tests/shrike/test_ackfloor.py
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

from hfmodem.shrike import onair, spec
from hfmodem.shrike.onair import RadioTx

FS = onair.FS
SLOT_N = round(1.25 * FS)
DATA_N = round(spec.P1_PACKET_S * FS)
CS_N = round(spec.P1_CS_S * FS)
SETTLE_N = round(0.040 * FS)
#: What the rig is keyed for to send one acknowledgement: the settle and the
#: codeword behind it, which is what `_refused` passes and what the peer shares
#: the channel with.
AIR_N = SETTLE_N + CS_N

# The raw gap behind the log's "d = 38.4": `_acquire` gates on the codeword
# position (42.0 ms) and the -3.6 ms edge statistic lands after the gate.
# The bench has no audio, so no edge term, so the raw gap is the scene.
FAST_GAP_N = round(0.042 * FS)


def _irs_grid() -> onair._MasterGrid:
    g = onair._MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                          packet_n=DATA_N, cs_n=CS_N, d_max_n=round(0.13 * FS))
    g.reverse(to_iss=False)      # we hold the IRS side: our burst is the CS
    return g


def _answer(g: onair._MasterGrid, gap_n: int, k: int = 5) -> int:
    """Where a peer that turns around `gap_n` after our RF ends puts its onset."""
    return g.anchor + k * SLOT_N + g.packet_n + gap_n


def _acquired(g: onair._MasterGrid, gap_n: int, k: int = 5) -> tuple[int, str]:
    """Answer twice at `gap_n`, which is what corroborates a turnaround."""
    g.update([_answer(g, gap_n, k - 1)])
    onset = _answer(g, gap_n, k)
    return onset, g.update([onset])


def test_the_fast_peer_is_acquired_and_answered_where_it_reads():
    # d = 42 ms clears `D_MIN_S`. The turnaround is the peer's and it is kept;
    # the ack keys 170 ms - d past its data end, clear of the peer's packet, like
    # every other turnaround the search accepts.
    g = _irs_grid()
    onset, line = _acquired(g, FAST_GAP_N)
    assert g.locked and g.acquired and g.corroborated and g.peer_onset == onset
    assert "TURNAROUND ACQUIRED" in line
    for slot in (7, 9, 14):
        b = g.boundary(slot)
        assert (b - onset - DATA_N) % SLOT_N == SLOT_N - DATA_N - CS_N - FAST_GAP_N
        assert g.key_refusal(b - SETTLE_N, AIR_N) is None


def test_the_stall_rejects_answers_under_d_min():
    # The nine unanswered break-ins: the peer kept answering, at raw 28-37 ms,
    # and every burst fell under `D_MIN_S` -- rejected before any codeword
    # search could witness it. Lowering it needs a discriminator the grid does
    # not have yet; this pins today's behaviour.
    g = _irs_grid()
    g.update([_answer(g, FAST_GAP_N)])       # answered before: acquired latches
    for _ in range(onair._MasterGrid.MAX_MISSES):
        line = g.update([])
    assert "released" in line and not g.locked and g.acquired
    line = g.update([_answer(g, round(0.032 * FS), k=9)])
    assert not g.locked
    assert "has to fall between 40 and 130" in line
    assert "staying on the air" in line


def _phase_to(g: onair._MasterGrid, boundary: int, onset: int) -> float:
    """Signed ms from that burst's data end to our CARRIER for `boundary`.

    The carrier, not the boundary, because that is the instant the emission path
    asks the guard about and the instant the peer shares the channel with.
    """
    p = (boundary - SETTLE_N - onset - DATA_N) % SLOT_N
    return (p - SLOT_N if p > SLOT_N / 2 else p) / FS * 1e3


def test_the_spurious_burst_no_longer_takes_the_band_from_the_peer():
    # The 899-pair sweep, standing: a burst anywhere in the sub-knee band
    # (41-69 ms) against a real answer anywhere at 90-120 ms, BOTH in every
    # cycle. Two witnessed gaps in the window is a thing timing cannot
    # resolve, so nothing is corroborated, the tracker never latches onto
    # either, and every cycle re-searches the whole band.
    for cycles in (1, 2, 6):
        worst = []
        for spur_ms in range(41, 70):
            for real_ms in range(90, 121):
                g = _irs_grid()
                for k in range(5, 5 + cycles):
                    g.update(sorted([_answer(g, round(spur_ms * 1e-3 * FS), k),
                                     _answer(g, round(real_ms * 1e-3 * FS), k)]))
                real = _answer(g, round(real_ms * 1e-3 * FS), 5 + cycles - 1)
                assert not g.corroborated, (cycles, spur_ms, real_ms)
                worst.append(_phase_to(g, g.boundary(5 + cycles), real + SLOT_N))
        # ...and the ack a cycle like that keys is still clear of the real
        # peer's packet, which is the whole of what the grid owes it. The whole
        # sweep sits inside the band; the ten slowest peers in it -- 111 to
        # 120 ms -- are the ones a 20 ms courtesy margin used to refuse.
        assert min(worst) >= 0, (cycles, min(worst))
        assert (round(min(worst)), round(max(worst))) == (10, 40), cycles


def test_a_burst_that_does_not_come_back_does_not_take_the_band_either():
    # The other side of the same rule, and the one that makes it a fix rather
    # than a refusal to decide: a peer answering every cycle beats a burst that
    # answered once, over the whole sweep.
    for spur_ms in range(41, 70):
        for real_ms in range(90, 121):
            g = _irs_grid()
            for k in (5, 6, 7):
                real = _answer(g, round(real_ms * 1e-3 * FS), k)
                once = [_answer(g, round(spur_ms * 1e-3 * FS), k)] if k == 5 else []
                g.update(sorted(once + [real]))
            assert g.corroborated and g.peer_onset == real, (spur_ms, real_ms)
            b = g.boundary(8)
            assert g.key_refusal(b - SETTLE_N, AIR_N) is None
            assert round(_phase_to(g, b, real + SLOT_N)) == 130 - real_ms


def test_a_peer_is_corroborated_inside_the_cycles_a_real_session_offers():
    # The cost, measured against every rig session on file that ever acquired
    # (2026-08-13/14): the nine corroborating pairs sit
    # 1, 1, 1, 1, 1, 2, 2, 2 and 3 cycles apart.
    #
    # A peer that answers every cycle pays one cycle. KB5LZK answered in cycles
    # 11 and 13 of rig-session-20260814-005758 and in neither 12, and pays two
    # -- which is what `SPAN` is eight for.
    for k, quiet in ((5, 0), (5, 1), (5, 3)):
        g = _irs_grid()
        line = g.update([_answer(g, FAST_GAP_N, k)])
        assert g.locked and not g.corroborated and "CANDIDATE" in line
        for _ in range(quiet):
            g.update([])
        onset = _answer(g, FAST_GAP_N, k + quiet + 1)
        line = g.update([onset])
        assert g.corroborated and "TURNAROUND ACQUIRED" in line, quiet
        assert "corroborated in cycle" in line
        assert g.peer_onset == onset


def test_the_uncorroborated_reading_still_opens_the_receive_window():
    # What is NOT paid for, and it is the whole reason the corroboration gates
    # the tracker rather than the acquisition. Five of the corpus's eleven
    # acquisitions came off a burst no later cycle ever agreed with -- three of
    # them in the last cycle of the call -- so a rule that withheld `d` until
    # agreement would have left those sessions with no receive window at all.
    g = _irs_grid()
    onset = _answer(g, round(0.105 * FS))
    line = g.update([onset])
    assert g.locked and g.acquired and g.peer_onset == onset
    assert abs(g.d - round(0.105 * FS)) <= 1
    assert g.rx_due(9) == g.boundary(9) + g.packet_n + g.d
    assert not g.corroborated and "CANDIDATE" in line


#: Turnarounds this station actually acquired, off the logs in this tree. Each
#: of the four readings above 110 ms is a link a 20 ms courtesy margin would
#: have refused on the receiving side. Its whole measured cost: over the 753 acknowledgements
#: keyed as IRS across 57 sessions it refuses nine -- eight of them every IRS
#: cycle of the WM4RB link, and one of the twenty-seven of `cal-pactor-ws8eoc`.
RECORDED = (
    ("KI0BK", 105.2, "working/rig-session-20260813-191422/session.log"),
    ("W6IDS", 108.6, "working/rig-session-20260813-233917/session.log"),
    ("WS8EOC", 114.0, "working/rig-session-20260814-212300/session.log"),
    ("WS8EOC", 119.2, "working/cal-pactor-ws8eoc.log"),
    ("WM4RB", 120.9, "working/rig-session-20260814-000029/session.log"),
    ("WM4RB", 121.7, "working/rig-session-20260814-000029/session.log"),
)


def test_the_slowest_peers_on_the_record_can_still_be_answered():
    """The receiving side's band is `_d_max_n`, and these are what pays for it.

    `key_refusal` carried a 20 ms margin on this side until 2026-08-28, which
    closed the workable turnaround at d <= 110 while `_d_max_n` admitted 130 --
    so a peer answering slowly was one we refused EVERY cycle of, in the role a
    gateway takes to hand us mail. WS8EOC's 114.0 ms is corroborated in its own
    cycle 12; WM4RB's 121.7 is a single-cycle acquire the tracker then read at
    120.9. Both are at this station's 40 ms settle, which is the only settle any
    session in the tree was flown at.
    """
    for peer, d_ms, where in RECORDED:
        g = _irs_grid()
        onset, line = _acquired(g, round(d_ms / 1e3 * FS))
        assert g.acquired and g.peer_onset == onset, (peer, line)
        b = g.boundary(6)
        assert g.key_refusal(b - SETTLE_N, AIR_N) is None, (peer, d_ms, where)
        # ...and the clearance is the arithmetic, not an accident of the bench.
        assert abs(_phase_to(g, b, onset) - (130 - d_ms)) < 0.1, (peer, d_ms)


def test_the_band_stops_where_the_grid_stops_acquiring():
    """One turnaround past `_d_max_n` the peer's packet really is still on the
    air when our carrier comes up, and the refusal says so in its terms."""
    g = _irs_grid()
    b = g.boundary(6)
    g.peer_onset = b - SETTLE_N - DATA_N          # d = 130 ms exactly: clear
    assert g.key_refusal(b - SETTLE_N, AIR_N) is None
    g.peer_onset += 1
    why = g.key_refusal(b - SETTLE_N, AIR_N)
    assert why is not None and "still on the air" in why, why


def _irs_refusals(n: int) -> tuple[list[bool], str]:
    """`n` acknowledgements into a grid that refuses every one of them."""
    g = _irs_grid()
    b = g.boundary(6)
    g.peer_onset = b - SETTLE_N - DATA_N + 1
    tx = RadioTx(None, transmit=False, outdir=Path("."), settle=0.040)
    tx.raster = g
    with contextlib.redirect_stdout(io.StringIO()) as log:
        dropped = [tx._refused(b - SETTLE_N, AIR_N, "P1 CS1") for _ in range(n)]
    return dropped, log.getvalue()


def test_the_receiving_sides_guard_cannot_hold_the_link_down_either():
    """It could, and that is the defect. The sending side's refusal rests on a
    projection that a moving peer falsifies; this one rests on a phase that is
    `_d_max_n` less a turnaround, and a turnaround does not move inside a
    session. So a first refusal here was every refusal, for ever, on a link the
    peer was working perfectly. Three drops, then one goes out."""
    dropped, log = _irs_refusals(9)
    assert dropped == [True, True, True, False] * 2 + [True], dropped
    assert log.count("STOOD DOWN") == 2, log
    assert log.count("ACK GUARD") == 9 and "QRM GUARD" not in log
    assert "--no-qrm-guard" not in log, "that flag is the sending side's"
    assert "1 of 3 in a row" in log
