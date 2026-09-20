# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A gateway answered two cycles in three and the session called it silence.

`pactor-current-kb5lzk-30-pounce-20260916T155343Z`, 30 m, 2026-09-16. The link
came up, the peer answered the raster with `0x59A` -- its standing grant word --
and this station logged `no control codewords decoded inside the link`, spent the
live-link retry budget, sent a QRT and hung up. Read off the same captures with
the codeword reader pointed at the grid instead of at an onset, ten of the
sixteen in-link slots hold that word at ZERO bit errors, at d = 20-46 ms and
1.95-3.44x over their own window.

WHAT STOOD BETWEEN THEM AND THE BUDGET IS A LEVEL. `rxfront._p1_runs` wants 4.0x
before a burst is offered to a decoder at all; these read 2.0-3.4x, so nothing
ever asked them what they said. `INLINK_READ_BAND` asks -- one read per alignment
across the band an answer is due in, zero bit errors, `ANSWER_CODEWORD_X` in
front of it -- and the word is then delivered exactly as the detector's own would
be, which is where a codeword's consequences belong.

THE BAND IS 20-130 ms AND COVERS TWO STATIONS. KB5LZK answers at 20-46 ms on 30 m
and `PEER_TURNAROUND_S` puts WS8EOC at 87-134 on 40 m. Unlike the connect search
this runs below `ACQUIRE_FLOOR_S`: the rig's mute is 14 ms, the peer is a station
already identified, and a false read costs one cycle of a budget of eight rather
than a link that was never there.

Run: pytest hfmodem/tests/shrike/test_inlink_anchored_read_0916.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1, rxfront, spec
from hfmodem.shrike.arq import State
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_qrtack import calling_station

FS = onair.FS
DATA_N = round(spec.P1_PACKET_S * FS)
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
ARM = (evidence.WORKING / "pactor-level-air-0916/arms"
       / "pactor-current-kb5lzk-30-pounce-20260916T155343Z" / "arm.log")

#: Every in-link slot of that call, through `_anchored_answer`: (hold, ms, word).
#: `0x59A` is `spec.P1_CS_NAMES` index 5; the CS1 of hold 1 is the peer's
#: acknowledgement arriving at an ordinary turnaround.
KB5LZK_INLINK = ((1, 96.0, "CS1/ack"), (2, 46.0, "0x59A/unassigned"),
                 (3, 30.0, "0x59A/unassigned"), (4, 28.0, "0x59A/unassigned"),
                 (5, 20.0, "0x59A/unassigned"), (6, 20.0, "0x59A/unassigned"),
                 (7, 34.0, "0x59A/unassigned"), (8, 20.0, "0x59A/unassigned"),
                 (9, 20.0, "0x59A/unassigned"), (12, 20.0, "0x59A/unassigned"),
                 (16, 20.0, "0x59A/unassigned"))


def _codeword(idx: int, level: float, at: float, n: int) -> np.ndarray:
    """One control signal at `at` seconds in a window of noise, scaled to `level`.

    The gate is a ratio against the window's own passband median, so the scale is
    searched rather than asserted: what a given amplitude reads as depends on the
    noise it sits in.
    """
    rng = np.random.default_rng(5)
    cs = onair._trim_silence(pactor1.control_signal(idx)).astype(np.float64)
    for gain in np.linspace(0.02, 3.0, 60):
        seg = rng.normal(0, 0.02, n)
        i = int(at * FS)
        seg[i:i + cs.size] += cs * gain
        out = seg.astype(np.float32)
        if onair._candidate_excess(out, at) >= level:
            return out
    raise AssertionError(f"no gain reached {level}x")


def test_the_read_is_at_the_grid_and_not_at_an_onset():
    """A codeword the onset detector would not offer is read anyway."""
    n = round(0.215 * FS)
    seg = _codeword(pactor1.CS_ACK_B, 2.0, 0.038, n)
    assert "nothing at the FSK tones" in rxfront.cs_evidence(seg) \
        or "not a control signal's length" in rxfront.cs_evidence(seg), \
        f"the onset detector would have offered this one: {rxfront.cs_evidence(seg)}"
    got = onair._anchored_answer(seg, 0)
    assert got is not None, "the grid read found nothing"
    cs, at, level = got
    assert cs[0] == pactor1.CS_ACK_B and cs[1] == 0
    # WHERE THE READ STARTED, not where the burst did: `decode_control_signal`
    # locks the word inside the slice it is given, so the first alignment that
    # reads is the earliest one holding the whole of it.
    assert onair.INLINK_READ_BAND[0] <= at <= 0.038, at
    assert level >= onair.ANSWER_CODEWORD_X


def test_a_codeword_under_the_level_is_not_read():
    """The same gate the connect candidates clear, and for the same reason: an
    exact twelve bits is a match on noise as readily as on a station."""
    n = round(0.215 * FS)
    rng = np.random.default_rng(9)
    assert onair._anchored_answer(rng.normal(0, 0.05, n).astype(np.float32), 0) is None
    quiet = _codeword(pactor1.CS_ACK_B, onair.ANSWER_CODEWORD_X - 0.3, 0.038, n)
    assert onair._candidate_excess(quiet, 0.038) < onair.ANSWER_CODEWORD_X
    assert onair._anchored_answer(quiet, 0) is None


def test_the_band_holds_both_stations_on_file():
    """20-130 ms is not a preference: it is where two gateways actually answer."""
    lo, hi = onair.INLINK_READ_BAND
    assert lo < 0.038 < hi, "KB5LZK's 30 m turnaround is outside the band"
    assert lo < onair.PEER_TURNAROUND_S[1] < hi, \
        "the 40 m median turnaround is outside the band"
    assert lo < onair.ACQUIRE_FLOOR_S, \
        "the in-link read no longer reaches below the connect search's floor"
    # The top is the window's, not the peer's: a keyed 1.25 s cycle leaves about
    # 215 ms, and a codeword starting past this does not fit in it whole.
    window = spec.CYCLE_SHORT_S - 0.04 - spec.P1_PACKET_S
    assert hi + spec.P1_CS_S <= window + 0.02, \
        f"a codeword at the top of the band does not fit a {window * 1e3:.0f} ms window"
    assert onair.PEER_TURNAROUND_S[2] - hi < 0.005, \
        "the band no longer reaches the top of the measured turnaround spread"


def test_the_grid_read_costs_one_cycle_in_five_hundred():
    """The price, on real off-air energy addressed to nobody.

    Over the 32239 negative cycle-windows of `connect-ceiling-negatives.json` --
    1188 recordings of rf-corpus and this station's own ARDOP captures -- one
    read per alignment across 20-130 ms at zero bit errors accepts in 839 of them
    (2.602%), and 65 of those clear `ANSWER_CODEWORD_X` (0.202%). What a false
    one costs is one cycle of a budget of eight; what the eight bought on
    2026-09-16 was a QRT over an answering gateway.

    Asserted as a ratchet on the rule rather than re-swept here: the corpus is
    11.3 hours of audio and is not part of this distribution.
    """
    assert onair.INLINK_READ_BAND == (0.020, 0.130)
    assert onair.INLINK_READ_HOP_S == 0.002
    assert onair.ANSWER_CODEWORD_X == 1.4
    alignments = (onair.INLINK_READ_BAND[1] - onair.INLINK_READ_BAND[0]) \
        / onair.INLINK_READ_HOP_S
    assert 50 <= alignments <= 60, \
        f"{alignments:.0f} alignments is not the 56 the 0.202% was measured over"


# -- and what it does to a link ---------------------------------------------

def _cs_event(t: float, idx: int) -> rxfront.Event:
    """The event the loop builds for a word it read at the grid."""
    return rxfront.Event(t, "cs", f"{spec.P1_CS_NAMES[idx]}  (0 bit errors, "
                         f"PACTOR-1, read at the grid, shift normal)",
                         protocol="PACTOR-1", cs=idx, sense=0)


def _run(deliver: bool) -> list[tuple[str, object]]:
    """A linked ISS with a packet in flight, answered or not, cycle by cycle."""
    host = calling_station()
    host.arq.on_host_data(b"payload " * 40)
    acks, out = (pactor1.CS_ACK_B, pactor1.CS_ACK_A), []
    for i in range(host.arq.cfg.max_retries + 4):
        if deliver:
            host.on_rx_event(_cs_event(2.0 + i, acks[i % 2]))
        host.tick()
        fl = host.arq._inflight
        out.append((host.arq.state.name, None if fl is None else fl.retries))
    return out


def test_a_word_read_at_the_grid_is_a_cycle_the_budget_may_not_charge():
    """The outcome, through the real FSM: the same cycles, answered, hold the link.

    `_inflight.retries` is the count that reaches `max_retries` and sends the QRT
    -- `[host] max retries (nothing the peer sent asked for the channel…) -> QRT`
    is line 222 of the KB5LZK arm log. Silence spends it in eight cycles. The
    same eight cycles carrying a codeword this station can now read spend none of
    it, because the word reaches the host exactly as an onset-detected one does
    and the acknowledgement path does the rest.
    """
    silent, answered = _run(False), _run(True)
    assert any(state != State.CONNECTED.name for state, _ in silent), \
        "silence no longer tears the link down, so this proves nothing"
    assert all(state == State.CONNECTED.name for state, _ in answered), answered
    assert max(r for _, r in silent if r is not None) > host_budget(), silent
    assert all(r == 0 for _, r in answered if r is not None), answered


def host_budget() -> int:
    return calling_station().arq.cfg.max_retries - 1


def _unassigned_event(t: float) -> rxfront.Event:
    """The event the loop builds for an unassigned word read at the grid."""
    return rxfront.Event(t, "unassigned", "0x59A  (0 bit errors, PACTOR-1, "
                         "read at the grid, shift normal)",
                         protocol="PACTOR-1", spare=pactor1.CS_59A, sense=0)


def test_the_grant_word_is_a_cycle_the_budget_may_not_charge_either():
    """A plain PACTOR-1 link, no upgrade announced, `0x59A` every cycle.

    The word reaches `arq.note_upgrade_unread`, which used to count only while an
    upgrade window was open -- `_unanswered_upgrade is None` on a link that never
    offered one -- so the peer answered and the in-flight retry budget was
    charged anyway. It now credits the cycle whatever the window is doing, which
    is the whole of what a codeword in the answer slot says.

    The count reaches 1 rather than 0: the reset lands when the word arrives and
    the tick then charges the transmission that goes out behind it. What matters
    is that it never climbs.
    """
    host = calling_station()
    host.arq.on_host_data(b"payload " * 40)
    budget = host.arq.cfg.max_retries
    seen = []
    for i in range(budget + 4):
        host.on_rx_event(_unassigned_event(2.0 + i))
        host.tick()
        seen.append(host.arq._inflight.retries if host.arq._inflight else None)
    assert host.arq.state == State.CONNECTED, host.arq.state
    assert max(r for r in seen if r is not None) <= 1, seen
    assert host.arq._unanswered_upgrade is None, \
        "the credit opened an upgrade window, which is not its business"


# -- the arm it was read off -------------------------------------------------

@pytest.mark.skipif(not ARM.exists(),
                    reason=f"{ARM} is this station's own working record")
def test_the_kb5lzk_call_reads_its_gateway_after_all():
    """Every in-link slot of that call, through the production read.

    Ten 0x59A and one CS1 in sixteen slots, against the nothing the session
    logged. Holds 1-9 and 12 answer, so the unanswered run never reaches the
    eight that sent the QRT -- the goodbye of holds 13-16, and the teardown rule
    that counts it, are not touched here and do not have to be.
    """
    log = ARM.read_text()
    out = Path(re.search(r"--outdir (\S+)", log).group(1))
    if not out.is_dir():
        pytest.skip(f"{out} is not on this machine")
    first = int(re.search(r"\[grid\] slot 1 boundary @ sample (\d+)", log).group(1))
    holds = [(int(h), int(m)) for h, m
             in re.findall(r"\[grid\] hold (\d+) slot (\d+),", log)]
    assert len(holds) == 16
    got = []
    for hold, slot in holds:
        wav = out / f"hold_{hold:02d}.wav"
        seg = rxfront.load_wav(str(wav))
        side = json.loads(wav.with_suffix(".json").read_text())
        seg_start = side["end_stream_sample"] - side["samples"]
        since = seg_start - (first + (slot - 2) * SLOT_N + DATA_N)
        read = onair._anchored_answer(seg, since)
        if read is None:
            continue
        cs, at, level = read
        got.append((hold, round((at + since / FS) * 1e3, 1),
                    spec.P1_CS_NAMES[cs[0]]))
        assert level >= onair.ANSWER_CODEWORD_X
    assert tuple(got) == KB5LZK_INLINK, got
    answered = {h for h, _, _ in got}
    run = max_run(h for h, _ in holds if h not in answered)
    assert run < calling_station().arq.cfg.max_retries, \
        f"an unanswered run of {run} still reaches the budget that sent the QRT"

    # ...and those sixteen cycles through the FSM, answered where the tape says
    # they were answered: hold 16 is the last of the four the session spent its
    # goodbye on, and it carries a word that now credits the budget.
    host = calling_station()
    host.arq.on_host_data(b"payload " * 40)
    for hold, _ in holds:
        if hold in answered:
            host.on_rx_event(_unassigned_event(2.0 + hold))
        host.tick()
    assert host.arq.state == State.CONNECTED, \
        f"the link still went down over a gateway answering {len(answered)} of 16"
    assert 16 in answered and host.arq._inflight.retries <= 1


def max_run(missing) -> int:
    """The longest stretch of consecutive unanswered slots."""
    out = best = 0
    prev = None
    for h in sorted(missing):
        out = out + 1 if prev is not None and h == prev + 1 else 1
        best, prev = max(best, out), h
    return best


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
