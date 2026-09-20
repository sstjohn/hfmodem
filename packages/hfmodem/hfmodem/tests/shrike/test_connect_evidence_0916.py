# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Four calls to two gateways on 2026-09-16, and what they moved.

Three of them corroborated a peer and linked -- WS8EOC on 80 m twice
(`...80-pounce-20260916T132318Z`, `...80-force-20260916T135856Z`) and KB5LZK on
40 m (`...40-sense-20260916T150811Z`), which then read two 0x59A and a CS2 off
the peer in session. The fourth, `...80-sense-20260916T150542Z`, accepted five
zero-error codewords, corroborated none of them, and printed `no PACTOR-1 control
signal decoded` over its own log. Its tape says those five are noise: each sits
inside the tape's own band statistic, a witness receiver 160 miles from the
gateway shows our bursts and nothing in the gaps, and the search accepts in 6.6%
of that tape's windows with nothing in them at all.

WHAT THE SEARCH HAD NO ANSWER TO IS THE LEVEL, which is the change with the
number on it: twelve bits matched on the sign of an instantaneous frequency is
not evidence that anybody transmitted. `onair.ANSWER_CODEWORD_X` is that gate and
its pricing is `fixtures/connect-level-0916.json`, asserted below on both
populations.

Three smaller things came off the same arms. The final-call listen extended by
three cycles from wherever a candidate landed and so closed with three of its
eight authorised cycles unspent while the peer was answering. The cycle that ENDS
a hush was never searched, because `keyed_at` still named the carrier before the
hush -- cycles 11, 20 and 29 of the 150542Z arm. And the verdict line read off
what reached the ARQ, so it called a channel silent that had five accepts in it.

THE MEASUREMENTS HERE WERE ALL TAKEN WITH THIS STATION'S OWN RFI PRESENT: the tap
analysis of the same morning has the receive floor rising 7.6 dB broadband from
the first key-down until 0.64 s after our last transmission, combed at 1000.000
Hz, ahead of the rig's T/R mute. Every window after our first call is searched
about 8 dB deafer than the pre-key floor, so the false-accept budget priced here
is conservative and the positives are weaker than that peer really was.

The operator heard WS8EOC answering on rig audio through the 150542Z call while
the tape and the witness hold nothing. That conflict is open and nothing here
settles it.

Run: pytest hfmodem/tests/shrike/test_connect_evidence_0916.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1, spec
from hfmodem.tests.shrike.test_connect_gate import _closes
from hfmodem.tests.shrike.test_connect_tail import session

FS = onair.FS
SLOT_N, DATA_N = round(spec.CYCLE_SHORT_S * FS), round(spec.P1_PACKET_S * FS)
D_MAX_N = onair._d_max_n(spec.CYCLE_SHORT_S, 0.04)
_LEVELS_FILE = Path(__file__).with_name("fixtures") / "connect-level-0916.json"
if not _LEVELS_FILE.exists():
    # The pricing fixture is off-air data the distribution does not carry.
    pytest.skip("connect-level-0916.json is not in this tree", allow_module_level=True)
LEVELS = json.loads(_LEVELS_FILE.read_text())


# -- the level a candidate has to stand at ----------------------------------

def _linked(arm) -> list[list]:
    """The accepts that arm's corroboration actually closed on."""
    return [a for a in arm["accepts"] if a[0] in arm["linked_on"]]


def test_the_level_gate_keeps_every_link_and_takes_the_corpus_off_the_rule():
    """The price, both ways, and the ratchet on it.

    Negatives are the 699 accepts of `connect-ceiling-negatives.json` re-run
    through the same search -- every accept in every recording of that corpus
    able to hold three, which is the whole population `_ConnectEvidence` can
    fire on. Positives are the nine accepts three sessions of 2026-09-16 built a
    link out of. They separate: the negatives run 1.06 median and 2.11 at the
    99th percentile, and the weakest accept a link was ever made on stands at
    1.73.
    """
    rows = LEVELS["negatives"]["rows"]
    assert len(rows) == LEVELS["negatives"]["accepts"] == 699
    xs = np.array([r[6] for r in rows])
    assert round(float(np.median(xs)), 2) == 1.06
    assert round(float(np.percentile(xs, 99)), 2) == 2.11

    kept = [(onair.ANSWER_CODEWORD_X, 26), (1.5, 18), (1.7, 10), (2.3, 6)]
    assert [(g, int((xs >= g).sum())) for g, _ in kept] == kept, \
        "the corpus no longer prices the way the gate was chosen on"

    weakest = min(a[4] for arm in LEVELS["positives"].values()
                  for a in _linked(arm))
    assert weakest == 1.726, "the positives moved"
    assert weakest / onair.ANSWER_CODEWORD_X > 1.2, \
        "the gate is inside 20% of the weakest accept a real link was made on"
    assert onair.ANSWER_CODEWORD_X < weakest, \
        "the gate is at or above the weakest accept a real link was made on"
    for tag, arm in LEVELS["positives"].items():
        assert all(a[4] >= onair.ANSWER_CODEWORD_X for a in _linked(arm)), \
            f"the gate breaks the link {tag} made"


def test_the_gate_closes_the_last_false_link_in_the_corpus():
    """...and it is the rule's own input that changes, not the rule.

    `_ConnectEvidence` fires on one of the 1012 negative recordings ungated. No
    accept of that recording clears the gate, so the search stops handing the
    rule the coincidence, and nothing in `N`, `SPAN` or `TOL_S` has to move to
    get there.
    """
    by_rec: dict[str, list] = {}
    windows: dict[str, int] = {}
    for name, n, k, d, _cs, _sense, x in LEVELS["negatives"]["rows"]:
        by_rec.setdefault(name, []).append((k, d, x))
        windows[name] = n
    fires = {g: [name for name, acc in by_rec.items()
                 if _closes([(k, d) for k, d, x in acc if x >= g],
                            windows[name]) is not None]
             for g in (0.0, onair.ANSWER_CODEWORD_X)}
    assert len(fires[0.0]) == 1, fires[0.0]
    assert fires[onair.ANSWER_CODEWORD_X] == [], fires[onair.ANSWER_CODEWORD_X]


def test_the_disputed_call_loses_its_floor_accepts_and_keeps_its_pair():
    """What the gate does to the call that started this, and what it does not.

    Three of the five go: two that sat in the first 4 ms of the band and the one
    at 119 ms. The pair at 76 and 78 ms survives at 1.9-2.0x -- two accepts are
    not three, so the arm still does not connect, and the log now shows the
    operator two candidates and three discards instead of five answers.
    """
    arm = LEVELS["positives"]["ws8eoc-80-sense-20260916T150542Z"]
    assert arm["linked_on"] == []
    kept = [a for a in arm["accepts"] if a[4] >= onair.ANSWER_CODEWORD_X]
    assert [a[1] for a in kept] == [77.5, 76.3]
    assert onair._ConnectEvidence.N > len(kept), \
        "two accepts now name a station, which is not what the rule says"


def test_a_real_group_mixes_codeword_and_shift():
    """Why the corroborating accepts may not be made to match each other.

    Requiring the `N` to share a codeword, or a shift sense, reads as free
    strictness: over the negative corpus it costs nothing and closes the one
    false link as well. It costs two of the three links of this morning. The
    force arm corroborated on CS1, CS1 and CS4 -- both are legal answers to a
    call and a peer may send either -- and the shift alternates cycle to cycle
    in every one of the three, which is the protocol working rather than a peer
    changing its mind.
    """
    for tag, arm in LEVELS["positives"].items():
        group = _linked(arm)
        if not group:
            continue
        assert len({a[3] for a in group}) > 1, f"{tag} held one shift sense"
    force = _linked(LEVELS["positives"]["ws8eoc-80-force-20260916T135856Z"])
    assert len({a[2] for a in force}) > 1, "the force arm's group was one codeword"


def test_a_candidate_under_the_gate_never_reaches_the_rule(
        tmp_path, monkeypatch, capsys):
    """The wiring: a discarded codeword is not offered, and it is not silence."""
    monkeypatch.setattr(onair, "_candidate_excess",
                        lambda seg, at: onair.ANSWER_CODEWORD_X - 0.1)
    keys = session(tmp_path, monkeypatch, replies=5)
    out = capsys.readouterr().out
    assert "DISCARDED" in out
    assert "** CONNECTED to" not in out
    assert any(k[0].startswith("ID ") for k in keys), \
        "the call neither connected nor identified"


def test_the_gate_is_an_excess_and_not_a_level():
    """A codeword 40 dB down reads the same as one at full scale on a quiet
    channel; what changes it is noise in the same passband."""
    cs = onair._trim_silence(pactor1.control_signal(pactor1.CS_SPEED))
    rng = np.random.default_rng(11)
    for gain in (1.0, 0.01):
        seg = rng.normal(0, 1e-4, SLOT_N).astype(np.float32)
        seg[SLOT_N // 3:SLOT_N // 3 + cs.size] += cs * gain
        at = (SLOT_N // 3) / FS
        assert onair._candidate_excess(seg, at) > onair.ANSWER_CODEWORD_X, gain
    noise = rng.normal(0, 0.05, SLOT_N).astype(np.float32)
    assert onair._candidate_excess(noise, 0.3) < onair.ANSWER_CODEWORD_X


# -- where the search says the codeword is ----------------------------------

@pytest.mark.parametrize("place", [0.5, 0.5031, 0.5077, 0.5125])
def test_the_search_offset_lands_on_the_codeword(place):
    """`t0 + offset` is the position, and it is measured rather than derived.

    The slice starts `ACQUIRE_LEAD_S` in front of `t0` and the alignments are
    counted from there, which reads as 4 ms of double counting and is not: the
    scan returns the FIRST alignment that decodes and that runs early by about
    the same 4 ms. Planted in quiet audio, `t0 + offset` recovers the planting
    inside one hop while subtracting the lead is 4-5 ms early.

    The air said it first. On `...80-force-20260916T135856Z` the energy detector
    -- no alignments in it at all -- put the answer the session connected on at
    71.4 ms after our data ended; the search read 71.3, and lead-corrected that
    answer is 67.2.
    """
    cs = onair._trim_silence(pactor1.control_signal(pactor1.CS_SPEED))
    audio = np.zeros(int(2.0 * FS), np.float32)
    n = int(place * FS)
    audio[n:n + cs.size] = cs
    got = p1rx.acquire_control_signal(audio, 0.40, spec.P1_CS_S, span=0.25)
    assert got is not None and got[0] == pactor1.CS_SPEED
    assert abs(0.40 + got[1] - place) <= p1rx.ACQUIRE_HOP_S, \
        f"the search put the codeword at {(0.40 + got[1]) * 1e3:.2f} ms"
    lead_corrected = 0.40 - p1rx.ACQUIRE_LEAD_S + got[1]
    assert place - lead_corrected > 0.003, \
        "subtracting the lead is no longer the early reading it was measured as"


# -- the final-call listen --------------------------------------------------

def test_a_candidate_spends_the_whole_final_listen():
    """An exact candidate takes the tail to its ceiling, not three cycles on.

    The 150542Z arm took its first candidate early enough that three cycles ran
    to 46.350 s against a ceiling of 50.080, and closed with three authorised
    cycles unspent while the peer was answering in searched windows 29 and 31.
    """
    tail = onair._ConnectTail(0, SLOT_N)
    assert tail.end == onair._ConnectTail.INITIAL_CYCLES * SLOT_N
    assert tail.candidate(SLOT_N)
    assert tail.end == tail.limit == onair._ConnectTail.MAX_CYCLES * SLOT_N
    assert not tail.candidate(7 * SLOT_N), "the ceiling extended past itself"


def test_the_extension_line_reports_the_ceiling(tmp_path, monkeypatch, capsys):
    """And the operator is told what it now does, on the capture clock."""
    session(tmp_path, monkeypatch, replies=1)
    line = next(ln for ln in capsys.readouterr().out.splitlines()
                if "exact candidate extends listening" in ln)
    to = float(line.split("listening to ")[1].split("s")[0])
    ceiling = float(line.split("fixed ceiling ")[1].split("s")[0])
    assert to == ceiling, line


# -- the cycle that ends a hush ---------------------------------------------

def test_the_cycle_that_ends_a_hush_is_searched_on_its_slot():
    """`keyed_at` is a whole hush stale in the cycle that keys again.

    That cycle is not hushed -- it is about to transmit -- so it measured from
    our last real carrier, five or six cycles back, and `_acquisition_window`
    gave back nothing to search. One window per hush went unread: cycles 11, 20
    and 29 of the 150542Z arm, each of them a cycle in which a peer answering our
    raster is provably not covered by us.

    The second origin is a FALLBACK. A skipped slot or a re-aimed key moves the
    boundary without any hush, and a rule that reads those as stale searches off
    our own carrier into a band nothing is in.
    """
    boundary = 10 * SLOT_N
    own = boundary - SLOT_N + DATA_N              # our own last carrier
    first, second = onair._answer_origins(own, boundary, SLOT_N, DATA_N, False)
    assert (first, second) == (own, own), \
        "the two origins disagree in an ordinary keyed cycle"
    stale = boundary - 6 * SLOT_N + DATA_N        # ...six cycles of hush ago
    first, second = onair._answer_origins(stale, boundary, SLOT_N, DATA_N, False)
    assert (first, second) == (stale, own), "the slot is not the fallback"
    assert onair._answer_origins(stale, boundary, SLOT_N, DATA_N, True) \
        == (own, stale), "a hush no longer takes its slot first"

    seg_start = boundary - SLOT_N                 # the window runs from the slot
    assert onair._acquisition_window(SLOT_N, seg_start - stale, D_MAX_N,
                                     onair.ACQUIRE_READ_TAIL_S)[1] == 0.0, \
        "the stale anchor left something to search, so nothing was ever lost"
    t0, span = onair._acquisition_window(SLOT_N, seg_start - own, D_MAX_N,
                                         onair.ACQUIRE_READ_TAIL_S)
    assert span > 0
    lo = t0 + (seg_start - own) / FS
    assert round(lo, 6) == onair.TR_SWITCH_S, \
        f"the recovered band starts at {lo * 1e3} ms"


def test_the_search_floor_is_the_measured_mute_plus_what_the_corpus_charges():
    """The floor is 55 ms and the rig's mute is 14; both numbers are held here.

    Over 34 keyed slots of 2026-09-16 -- three arms, two bands -- the receive
    chain is back within 3 dB of its own window floor at d = 13-15 ms, so
    `TR_SWITCH_S`'s 55 is a transmit number and not a description of what we can
    hear. Dropping the search to 20 ms buys 35 ms of audio the corpus prices at
    one phantom link: 1161 accepting windows of 32239 become 1766, and after
    `ANSWER_CODEWORD_X` 65 become 104 -- which is where the rule stops holding,
    at `rf-corpus/offair/ws8eoc_witness_20260806/kiwi_witness.wav`.

    So the constant exists, is documented, and has not moved. This asserts the
    band it gives, which is what the pricing was taken on.
    """
    assert onair.ACQUIRE_FLOOR_S == 0.055
    t0, span = onair._acquisition_window(SLOT_N, -DATA_N, D_MAX_N,
                                         onair.ACQUIRE_READ_TAIL_S)
    lo, hi = t0 - DATA_N / FS, t0 + span - DATA_N / FS
    assert (round(lo * 1e3), round(hi * 1e3)) == (55, 130), \
        f"the band is {lo * 1e3:.0f}-{hi * 1e3:.0f} ms, not what was priced"


# -- and the verdict --------------------------------------------------------

def _verdict(ev: onair._ConnectEvidence, capsys) -> str:
    onair._summary([], [], ev.searched, [], spec.CYCLE_SHORT_S, [], ev,
                   "the connect budget ran out")
    return next(ln for ln in capsys.readouterr().out.splitlines()
                if ln.startswith("verdict:"))


def test_uncorroborated_accepts_are_not_a_silent_channel(capsys):
    """The 150542Z arm's verdict contradicted its own log five lines above it.

    `no PACTOR-1 control signal decoded` is what reaches the ARQ, and an accept
    the rule did not close on never does. The operator's next move differs:
    nothing accepted is a band question, accepts that did not agree is a
    question about this station's receive floor and the peer's schedule.
    """
    ev = onair._ConnectEvidence()
    for cycle, d in ((32, 0.0775), (34, 0.0763)):
        ev.note_search(0.075)
        ev.offer(cycle, d)
    line = _verdict(ev, capsys)
    assert "no PACTOR-1 control signal decoded" not in line
    assert "2 zero-error codeword(s) accepted" in line
    assert "closest pair 1.2 ms apart" in line


def test_one_accept_has_no_spread_and_says_so(capsys):
    ev = onair._ConnectEvidence()
    ev.note_search(0.075)
    ev.offer(7, 0.072)
    assert "a single accept, which has no spread" in _verdict(ev, capsys)


def test_a_channel_that_accepted_nothing_still_reads_as_one(capsys):
    """The wording that was always right is reserved for the case it is right
    for."""
    ev = onair._ConnectEvidence()
    ev.note_search(0.075)
    assert "no PACTOR-1 control signal decoded" in _verdict(ev, capsys)


def test_the_discards_are_counted_where_the_operator_reads_them():
    ev = onair._ConnectEvidence()
    ev.note_search(0.075)
    ev.note_discard()
    ev.note_discard()
    assert "2 more read out of the band's own noise" in ev.report()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
