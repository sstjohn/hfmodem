# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A gateway that answered, and a session that reported hearing nothing.

`captures/onair-0911-2332` is WS8EOC on 80 m, called for 25 cycles while the
operator listened to it answering on a monitor receiver. The production connect
search reads its answer three times -- CS4 at zero bit errors, 98, 100 and 99 ms
after our carrier dropped, a 2 ms group against the 2.5 ms the seven 2026-07-30
sessions put on `_ConnectEvidence.TOL_S` -- and the session threw all three away,
printed `no control word has been read this session` under every one of them, and
went off the air three times over a peer that was answering. It connected in
cycle 26 on a burst the anchored reader happened to catch in a long hush window,
which is luck; two other arms the same evening ran the same way and gave up.

TWO THINGS WERE WRONG AND THEY ARE DIFFERENT THINGS.

The span was counted in CYCLES. Its measurement is not: 31996 consecutive
cycle-windows of off-air energy, which are draws of the search, and this station
calls four cycles in ten. 13 of the 25 cycles were searched, so an eight-cycle
bound was spending four windows of the twelve it was priced for. In searches the
three answers are the 5th, 7th and 13th -- nine end to end, which is what a
4-call/6-hush rotation costs a peer that answers the last call of each run.

And the hush was armed anyway. `_MasterGrid._blind` goes quiet on two premises
together: our own carrier may be covering the peer's burst, and nothing on this
grid says anything is there. A twelve-bit codeword decoded at zero bit errors
where an answer to our own call was due refutes both, whatever the onset detector
made of the same audio -- so it cancels the hush, and it aims NOTHING. Corroboration
is still what places a receive window, and it is still three accepts. The two
questions are `answered` and `_ConnectEvidence` and they are asked separately.

Run: pytest hfmodem/tests/shrike/test_connect_gate.py
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from hfmodem.shrike import onair, p1rx, rxfront, spec
from hfmodem.tests import evidence

FS = onair.FS
FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.name} is not in this tree (it does not ship)")
    return path

SESSION = evidence.CAPTURES / "onair-0911-2332"
SLOT_N = round(1.25 * FS)
D_MAX_N = onair._d_max_n(1.25, 0.040)

#: What the live arm reported at its sign-off: three candidates over 13 searched
#: cycles, at 98, 100 and 99 ms, none corroborated.
ANSWERS = ((5, 98), (13, 100), (25, 99))
SEARCHED = 13


def _grid() -> onair._MasterGrid:
    return onair._MasterGrid(0, SLOT_N, round(0.185 * FS),
                             packet_n=round(spec.P1_PACKET_S * FS),
                             cs_n=round(spec.P1_CS_S * FS), d_max_n=D_MAX_N)


def _windows():
    """The session's receive windows, and whether an answer was due in each.

    A cycle we called in gets a peephole between our own carriers -- 235 to 297 ms
    of this session's 1.25 s -- and a cycle we did not gets the whole of it. That
    is the schedule as recorded, and it is handed to `_acquisition_window` as
    `since_tx` so the production gate decides for itself which windows hold the
    band: from the second cycle of a hush our last carrier is a cycle back and the
    band falls off the front. 13 of the 25, exactly as the arm reported.
    """
    for wav in sorted(SESSION.glob("rx_*.wav")):
        seg = rxfront.load_wav(str(wav))
        hushed = seg.size > SLOT_N // 2
        yield int(wav.stem.split("_")[1]), seg, SLOT_N if hushed else 0


def _replay(ev: onair._ConnectEvidence, tail: float = onair.ACQUIRE_TAIL_S):
    """Run the production connect search over the session; return (cycle, ms).

    `tail` is which top edge the band is cut at. The two below take the SLOT's
    reach, which is the band this session was searched over when the span bound
    was priced off it; `ACQUIRE_READ_TAIL_S` is 19 ms further out and is what the
    loop hands the search now. Both are measured here, on the same recording.
    """
    found, at = [], None
    for cycle, seg, since_tx in _windows():
        t0, span = onair._acquisition_window(seg.size, since_tx, D_MAX_N, tail)
        if span <= 0:
            continue
        ev.note_search(span)
        got = p1rx.acquire_control_signal(seg, t0, spec.P1_CS_S, span=span)
        if got is None:
            continue
        d = t0 + got[1]
        found.append((cycle, round(d * 1e3), got[0]))
        if ev.offer(cycle, d) and at is None:
            at = cycle
    return found, at


@pytest.mark.skipif(not SESSION.is_dir(),
                    reason=f"{SESSION} is this station's own recording of "
                           "WS8EOC 2026-09-11 and is in no other clone")
def test_the_gateway_that_answered_is_recognised() -> None:
    """The recorded answers, through the production search, now name a station."""
    ev = onair._ConnectEvidence()
    found, at = _replay(ev)
    assert ev.searched == SEARCHED, \
        f"the schedule reconstructs {ev.searched} searched windows, not {SEARCHED}"
    assert [(c, ms) for c, ms, _ in found] == list(ANSWERS), \
        f"the session's own answers are not where the arm reported them: {found}"
    assert {cs for _, _, cs in found} == {spec.CS_NAMES.index("SPEED-UP")}, \
        f"the answers are not the CS4 the arm read: {found}"
    assert at == 25, f"WS8EOC's three answers did not corroborate: {ev.report()}"
    assert "none corroborated" not in ev.report(), ev.report()


@pytest.mark.skipif(not SESSION.is_dir(), reason="station recording only")
def test_the_answers_span_more_than_eight_searches() -> None:
    """The counterexample, in the terms the bound is written in.

    This is the whole of why the span moved, so it is measured off the recording
    rather than asserted: eight searches cannot hold this peer and twelve can.
    """
    ev = onair._ConnectEvidence()
    _replay(ev)
    draws = [w for _, _, w in ev.candidates]
    assert draws == [5, 7, 13], f"the answers were drawn at {draws}"
    assert draws[-1] - draws[0] >= 8, \
        "the answers fit inside eight searches after all -- the span was not why"
    assert draws[-1] - draws[0] < onair._ConnectEvidence.SPAN


# -- the ceiling, which was the capture's end and not the peer's -------------

CEILING = FIXTURES / "ve3kpg-0913-connect-ceiling.json"

#: What the fifteen VE3KPG arms of 2026-09-13 got: the search topped out at
#: `seg - ACQUIRE_TAIL_S`, which on a 235-241 ms capture is 94-100 ms, and the
#: peer answers at 96-107 -- 42 codewords read across 357 windows holding 95.
#: This one call read four, one of them the 55 ms accept at the low edge that is
#: not an answer at all.
CLIPPED = ((5, 55.0), (13, 97.5), (16, 97.5), (24, 100.0))

#: ...and the same twenty-one windows at the reach a clipping search needs:
#: fifteen, fourteen of them inside the 96-107 ms this station turns around in.
READ = ((4, 101.25), (5, 55.0), (7, 105.0), (9, 98.75), (11, 100.0),
        (13, 97.5), (14, 100.0), (15, 102.5), (16, 97.5), (17, 101.25),
        (18, 106.25), (19, 98.75), (22, 105.0), (23, 105.0), (24, 100.0))


def _ceiling_windows() -> list:
    """The listen windows of the VE3KPG call of 2026-09-13 15:07, in order.

    Every cycle of that arm was keyed, so each window is what one call left --
    and its LENGTH is the thing under test, because the ceiling was cut from it.
    The fixture is those windows joined end to end; the manifest splits them
    apart again, and it has to account for all of the audio to do it.
    """
    manifest = json.loads(_fixture(CEILING).read_text())
    audio = rxfront.load_wav(str(CEILING.with_suffix(".wav")))
    out, at = [], 0
    for window in manifest["windows"]:
        out.append((window["cycle"], audio[at:at + window["samples"]]))
        at += window["samples"]
    assert at == audio.size, \
        f"the manifest describes {at} samples of a {audio.size}-sample fixture"
    return out


def _ceiling_replay(tail: float):
    """That call through the production search at `tail`; (accepts, closed-in)."""
    ev, found, at = onair._ConnectEvidence(), [], None
    for cycle, seg in _ceiling_windows():
        t0, span = onair._acquisition_window(seg.size, 0, D_MAX_N, tail)
        if span <= 0:
            continue
        ev.note_search(span)
        got = p1rx.acquire_control_signal(seg, t0, spec.P1_CS_S, span=span)
        if got is None:
            continue
        d = t0 + got[1]
        found.append((cycle, round(d * 1e3, 2), got[0]))
        if ev.offer(cycle, d) and at is None:
            at = cycle
    return found, at


def test_connect_ceiling_was_the_end_of_the_capture() -> None:
    """The defect, reproduced: the top edge dithers with the capture length.

    A keyed 1.25 s cycle on this rig leaves 235, 237, 239 or 241 ms depending on
    where the bridge fell, and asking for the smoother's 20 ms past the band took
    that straight off the peer -- so the same call is searched to 94, 96, 98 and
    100 ms on a four-cycle rotation, and VE3KPG answers at 96-107.
    """
    tops = sorted({round(onair._acquisition_window(
        seg.size, 0, D_MAX_N)[1] * 1e3 + onair.TR_SWITCH_S * 1e3, 1)
        for _, seg in _ceiling_windows()})
    assert tops == [94.2, 96.2, 98.2, 100.2, 130.0], \
        f"the slot's reach does not put the top where the arms found it: {tops}"
    found, at = _ceiling_replay(onair.ACQUIRE_TAIL_S)
    assert [(c, ms) for c, ms, _ in found] == list(CLIPPED), \
        f"the call did not read what the arm read: {found}"
    assert at == 24, f"the fourth codeword is what corroborated, not {at}"


def test_connect_ceiling_clears_the_peers_turnaround() -> None:
    """...and at the reach a clipping search needs, the answers are all there.

    Fourteen of the fifteen accepts land in 96-107 ms, which is where this
    station turns around; the fifteenth is the same 55 ms low-edge accept the
    clipped search already had, and it is why corroboration is three and not one.
    """
    found, at = _ceiling_replay(onair.ACQUIRE_READ_TAIL_S)
    assert [(c, ms) for c, ms, _ in found] == list(READ), \
        f"the widened band does not read the call the way it measures: {found}"
    answers = [ms for _, ms, _ in found if ms != 55.0]
    assert min(answers) >= 96 and max(answers) <= 107, \
        f"an accept landed outside the peer's own turnaround: {answers}"
    assert at == 9, f"VE3KPG was corroborated in cycle {at}, not 9"
    assert len(found) > len(CLIPPED) and at < 24, \
        "the wider band read no more of this call than the clipped one did"


def _closes(accepts, windows: int) -> int | None:
    """The window `_ConnectEvidence` closes in, driven as the hold loop drives it."""
    ev, at = onair._ConnectEvidence(), dict((k, d) for k, d in accepts)
    for k in range(windows):
        ev.searched += 1
        if k in at and ev.offer(k, at[k]):
            return k
    return None


def test_connect_ceiling_costs_no_extra_false_link() -> None:
    """The negative control, and the whole reason nothing was re-priced.

    `connect-ceiling-negatives.json` is one draw of the search per keyed cycle
    over every recording in rf-corpus and this station's ARDOP captures -- real
    off-air energy addressed to nobody. The wider band accepts in half again as
    many windows, and `_ConnectEvidence` closes on the SAME single recording
    either way, which is what says `N`, `SPAN` and `TOL_S` did not have to move.
    """
    sweep = json.loads(_fixture(FIXTURES / "connect-ceiling-negatives.json").read_text())
    assert sweep["rule"] == {"N": onair._ConnectEvidence.N,
                             "SPAN": onair._ConnectEvidence.SPAN,
                             "TOL_S": onair._ConnectEvidence.TOL_S}, \
        f"the rule moved since the sweep was taken: {sweep['rule']}"
    assert (sweep["bands"]["slot"]["tail_s"], sweep["bands"]["read"]["tail_s"]) \
        == (round(onair.ACQUIRE_TAIL_S, 6), round(onair.ACQUIRE_READ_TAIL_S, 6))

    fired = {}
    for band in ("slot", "read"):
        fired[band] = [c["recording"] for c in sweep["candidates"]
                       if _closes(c[band], c["windows"]) is not None]
    assert len(fired["slot"]) == sweep["bands"]["slot"]["fires"] == 1, fired
    assert fired["read"] == fired["slot"], \
        f"the wider band invented a link the narrow one did not: {fired}"


# -- a cycle we did not key still has an answer band -------------------------

def test_connect_ceiling_searches_a_hush_on_its_projected_slot() -> None:
    """The band hangs off the slot, not off the last carrier that really dropped.

    `keyed_at` freezes when the hush starts, so from the hush's SECOND cycle the
    window's first sample sits a whole cycle past the band and
    `_acquisition_window` gives back nothing to search. The peer does not stop:
    it is synchronised to our clock rather than to our PTT, and it keeps
    answering the slot we would have called on.
    """
    # A hush cycle's window runs from the slot boundary for a whole cycle, and
    # the call it would be answering ends `p1_data_n` into that slot -- so the
    # window's first sample sits that far IN FRONT of the band's origin.
    data_n = _grid().p1_data_n
    for back in range(1, 4):
        stale = back * SLOT_N - data_n        # our last real carrier, `back` back
        assert onair._acquisition_window(SLOT_N, stale, D_MAX_N,
                                         onair.ACQUIRE_READ_TAIL_S)[1] == 0.0, \
            f"the frozen anchor searched a hush cycle {back} cycles in"
    since_tx = -data_n
    t0, span = onair._acquisition_window(SLOT_N, since_tx, D_MAX_N,
                                         onair.ACQUIRE_READ_TAIL_S)
    assert span > 0, "the projected slot left nothing to search either"
    lo, hi = t0 + since_tx / FS, t0 + span + since_tx / FS
    assert (round(lo * 1e3), round(hi * 1e3)) == (55, 130), \
        f"the hush band is not the turnaround band: {lo * 1e3}-{hi * 1e3} ms"
    assert lo <= onair.PEER_TURNAROUND_S[1] <= hi, \
        "the median turnaround is outside the band a hush cycle is searched over"


def test_connect_ceiling_hush_codeword_cancels_the_hush() -> None:
    """...and what it is for: the decode ends the hush in the cycle it lands in.

    A hush that outlives the answer it was waiting for is a hush that has to be
    sat through, and `onair-0912-2329` sat through four decodable CS4s.
    """
    grid = _grid()
    for _ in range(onair._MasterGrid.BLIND_CYCLES + 1):
        grid.update([], hushed=False, since_tx=0)
    assert grid.hush_left, "the grid never went quiet"
    line = grid.update([], hushed=True, since_tx=SLOT_N, answered=(3, 0.09625))
    assert grid.hush_left == 0, line
    assert grid.answered_word[1:] == (3, 0.09625), grid.answered_word


HUSH_SESSION = evidence.CAPTURES / "onair-0912-2329"

#: `onair-0912-2329` cycles 29-32, which are logged HUSHED and hold a CS4 each.
#: The anchor arithmetic is the session's own: anchor 2738, 60000-sample slots,
#: a 46080-sample call, and cycle `c` answers the call slot `c - 1`.
HUSH_ANSWERS = ((29, 96.25), (30, 101.25), (31, 96.25), (32, 95.0))


@pytest.mark.skipif(not HUSH_SESSION.is_dir(),
                    reason=f"{HUSH_SESSION} is this station's own recording")
def test_connect_ceiling_reads_the_hush_cycles_of_2329() -> None:
    """The recording the stale anchor was found on, through the fixed loop."""
    anchor, slot_n, data_n = 2738, 60000, 46080
    frozen = 1608818                          # the last carrier that dropped
    found = []
    for cycle, _ in HUSH_ANSWERS:
        wav = HUSH_SESSION / f"rx_{cycle:02d}.wav"
        seg = rxfront.load_wav(str(wav))
        side = json.loads(wav.with_suffix(".json").read_text())
        seg_start = side["end_stream_sample"] - side["samples"]
        assert onair._acquisition_window(seg.size, seg_start - frozen, D_MAX_N,
                                         onair.ACQUIRE_READ_TAIL_S)[1] == 0.0, \
            f"cycle {cycle} was searchable off the frozen anchor after all"
        end = anchor + (cycle - 1) * slot_n + data_n
        t0, span = onair._acquisition_window(seg.size, seg_start - end, D_MAX_N,
                                             onair.ACQUIRE_READ_TAIL_S)
        got = p1rx.acquire_control_signal(seg, t0, spec.P1_CS_S, span=span)
        assert got is not None, f"cycle {cycle} read nothing on its own slot"
        found.append((cycle, round((t0 + got[1] + (seg_start - end) / FS)
                                   * 1e3, 2), got[0]))
    assert [(c, ms) for c, ms, _ in found] == list(HUSH_ANSWERS), found
    assert {cs for _, _, cs in found} == {spec.CS_NAMES.index("SPEED-UP")}, \
        f"the hush answers are not the CS4 the replay found: {found}"


# -- and the half that must not move ----------------------------------------

def test_one_codeword_aims_nothing() -> None:
    """A single zero-error accept keeps the air. It does not place a grid.

    The negative control for the whole change: the 2026-08-06 phantoms were one
    accept each, reported as links to three gateways that were never there.
    """
    ev = onair._ConnectEvidence()
    ev.note_search(0.075)
    assert not ev.offer(3, 0.098), "one codeword corroborated"
    assert ev.at is None
    g = _grid()
    line = g.update([], hushed=False, since_tx=0, answered=(3, 0.098))
    assert g.d_n is None and not g.corroborated and not g.acquired, \
        "a codeword placed the receive window"
    assert g.anchor == 0, "a codeword moved the transmit grid"
    assert "ANSWERED, NOT PLACED" in line and "aims nothing" in line, line


def test_one_timing_candidate_cannot_place_the_grid() -> None:
    """...and neither can one burst edge, which is the older half of the rule.

    `_TurnaroundEvidence` wants two cycles agreeing to 20 ms before a gap is a
    peer's turnaround. Bursts at 94 and 124 ms -- both inside the band, 30 ms
    apart -- are the reading that does not repeat: the receive window opens on
    each, the tracker never takes over, and the transmit anchor is not touched
    either way. This is the case the `answered` path is kept out of.
    """
    g = _grid()
    for cycle, d in ((1, 0.094), (2, 0.124)):
        onset = g.anchor + cycle * g.slot_n + g.p1_data_n + round(d * FS)
        line = g.update([onset], hushed=False, since_tx=0)
        assert not g.corroborated, f"one burst edge took over the tracker: {line}"
        assert "CANDIDATE" in line, line
    assert g.evidence.at is None, "two disagreeing gaps corroborated"
    assert g.anchor == 0, "an uncorroborated onset moved the transmit grid"


def test_the_reading_never_contradicts_a_zero_error_decode(capsys) -> None:
    """The line that cost an hour: it may not say nothing was read.

    `cs_log` holds what a decoder was given and an uncorroborated accept never
    reaches one, so the session printed `no control word has been read this
    session` under three zero-error codewords of its own.
    """
    class _Rx:
        cs_log: list = []
        cs_heard = None
        _p3_answer_at = None
        _p3_row0 = None

    class _Arq:
        role = "iss"
        entry_pending = False
        state = onair.State.CONNECTING

    class _Host:
        arq = _Arq()
        protocol = onair.Protocol.PACTOR1

    g = _grid()
    cold = onair._scheduler_reading(_Rx(), _Host(), g)
    assert "no control word has been read this session" in cold, cold

    said = onair._scheduler_reading(_Rx(), _Host(), g, (3, 0.098))
    assert "no control word has been read this session" not in said, said
    assert "CS4 read at zero bit errors" in said and "98 ms" in said, said

    g.update([], hushed=False, since_tx=0, answered=(3, 0.098))
    later = onair._scheduler_reading(_Rx(), _Host(), g)
    assert "no control word has been read this session" not in later, later
    assert "cycle 1's answer band" in later, later


def test_a_decoded_answer_keeps_the_station_on_the_air() -> None:
    """Four blind cycles arm the hush; an answer in the fourth does not.

    The hush is a link-setup move against a phase that hides the peer, and a
    codeword decoded in our own answer band is that phase demonstrably not
    hiding it. Six cycles off the air after the peer answers is six cycles it
    answers nothing, which is how three answers came to be twenty cycles apart.
    """
    deaf = _grid()
    for _ in range(onair._MasterGrid.BLIND_CYCLES):
        line = deaf.update([], hushed=False, since_tx=0)
    assert deaf.hush_left == onair._MasterGrid.HUSH_CYCLES, line
    assert "OFF THE AIR" in line, line

    heard = _grid()
    for _ in range(onair._MasterGrid.BLIND_CYCLES - 1):
        heard.update([], hushed=False, since_tx=0)
    line = heard.update([], hushed=False, since_tx=0, answered=(3, 0.098))
    assert heard.hush_left == 0, line
    assert "OFF THE AIR" not in line, line
    assert heard.blind == 1, "the blind count survived an answer"

    # ...and an answer arriving with the hush already running drains it, the way
    # a link coming up does. WS8EOC's first answer landed in exactly that cycle.
    draining = _grid()
    for _ in range(onair._MasterGrid.BLIND_CYCLES + 1):
        draining.update([], hushed=True, since_tx=0)
    assert draining.hush_left
    line = draining.update([], hushed=True, since_tx=0, answered=(3, 0.098))
    assert draining.hush_left == 0, line
    assert "listening in the clear" not in line, line


# -- a shape line is not a station, and it used to consume the search --------

EVENTS = FIXTURES / "ws8eoc-0913-connect-events.json"

#: Every zero-error accept the production search takes in that call, at the
#: offset it takes it at. Cycle 16 holds none and is in the fixture anyway,
#: because its burst raised a shape line and that is the thing under test.
ACCEPTS = ((6, 102.50), (13, 95.00), (20, 96.25), (24, 97.50),
           (27, 96.25), (28, 58.75), (30, 102.50))

#: The three cycles the front end raised a `p1reply` in -- and two of them hold
#: the cleanest reads of the arm. At a 0.125 ms hop cycles 13, 20 and 27 each put
#: a wide plateau of accepting alignments on 96.4 ms, repeatable to 0.12 ms,
#: which is what this gateway's 40 m turnaround is.
SHAPE_ONLY = (13, 16, 20)

#: What the loop hands the rolling decoder at a time. Any block shorter than its
#: slide reaches the same decodes; this is the one the arm ran.
CHUNK_N = FS // 10


class _StubArq:
    state = onair.State.CONNECTING
    role = "iss"


class _StubHost:
    """Enough of the host for `_SessionRx` to deliver an event to.

    Nothing here acts on one. The question the fixture asks is which events the
    FRONT END raises, and what the loop's own counters then do about them.
    """

    protocol = onair.Protocol.PACTOR1
    sent_total = 0
    rcvd_total = 0
    peer = None

    def __init__(self) -> None:
        self.arq = _StubArq()
        self.events: list = []

    def on_rx_event(self, ev) -> None:
        self.events.append(ev)


def _events_windows():
    """The fixture's listen windows: cycle, audio, and `since_tx` in samples.

    The band hangs off the carrier the window's call ended with, so the session's
    own grid is what places it: the call in slot `(end - anchor - data_n) //
    slot_n` ends `data_n` into that slot. A hush window starts BEFORE that
    instant, which is the negative `since_tx` `_acquisition_window` reads as a
    projected slot.
    """
    manifest = json.loads(_fixture(EVENTS).read_text())
    grid = manifest["grid"]
    audio = rxfront.load_wav(str(EVENTS.with_suffix(".wav")))
    out, at = [], 0
    for window in manifest["windows"]:
        seg = audio[at:at + window["samples"]]
        at += window["samples"]
        end = window["end_stream_sample"]
        slot = (end - grid["anchor"] - grid["data_n"]) // grid["slot_n"]
        carrier_end = grid["anchor"] + slot * grid["slot_n"] + grid["data_n"]
        out.append((window["cycle"], seg, end - seg.size - carrier_end))
    assert at == audio.size, \
        f"the manifest describes {at} samples of a {audio.size}-sample fixture"
    return grid["cycles"], out


@lru_cache(maxsize=1)
def _front_end() -> dict:
    """Each window through the session's own decoder: counters, and event kinds.

    A WINDOW IS NOT ONE DECODE. The loop feeds `RollingRx` in blocks and it
    decodes every slide over as much history as it holds, then flushes what the
    slides left -- and the flush is in front of the gate, so it is part of what
    the gate is answering. Run that way the fixture reproduces the arm's log
    exactly: a `p1reply` in cycles 13, 16 and 20, and no event at all in the
    other five. `decode_events` over the whole window is a different reading and
    is not the one the receiver made.

    One decoder per window, because the fixture carries eight of the arm's thirty
    and a rolling buffer reaching across the twenty-two that are not here would
    decode a splice that never went over the air -- which also makes the loop's
    top-of-cycle snapshot zero, so both gates below read these counters directly.
    """
    out = {}
    for cycle, seg, _ in _events_windows()[1]:
        sessrx = onair._SessionRx(_StubHost(), tag="FIX")
        for i in range(0, seg.size, CHUNK_N):
            sessrx.feed(seg[i:i + CHUNK_N])
        sessrx.flush()
        out[cycle] = (sessrx.count, len(sessrx.words_at),
                      sessrx.host.rcvd_total,
                      tuple(ev.kind for ev in sessrx.host.events))
    return out


def _raw_count_gate(cycle: int) -> bool:
    """The predicate as it stood: the front end raised NO EVENT this cycle."""
    return _front_end()[cycle][0] == 0


def _decoded_gate(cycle: int) -> bool:
    """...and as it stands: no DECODER took anything this cycle."""
    return _front_end()[cycle][1:3] == (0, 0)


def _events_replay(gate):
    """That call through the production search, with the cycles `gate` allows.

    Cycles the fixture carries no audio for are barren searches -- the arm printed
    `RX (nothing decoded)` in every one of them, and the 1.25 ms sweep behind the
    fixture confirms a band with no accept in it -- so the draw is counted and
    nothing else is. `note_search` is the loop's own call and wants the window's
    band to price `by_chance`, which is not what is measured here.
    """
    cycles, windows = _events_windows()
    have = {c: (seg, since) for c, seg, since in windows}
    ev, found, searched_in, at = onair._ConnectEvidence(), [], [], None
    for c in range(1, cycles + 1):
        if c not in have:
            ev.searched += 1
            continue
        if not gate(c):
            continue
        seg, since = have[c]
        t0, span = onair._acquisition_window(seg.size, since, D_MAX_N,
                                             onair.ACQUIRE_READ_TAIL_S)
        assert span > 0, f"cycle {c} left the search nothing to read"
        searched_in.append(c)
        ev.note_search(span)
        got = p1rx.acquire_control_signal(seg, t0, spec.P1_CS_S, span=span)
        if got is None:
            continue
        d = t0 + got[1] + (since / FS if since < 0 else 0.0)
        found.append((c, round(d * 1e3, 2), got[0]))
        if ev.offer(c, d):
            at = c
            break                       # the call is over; the link is coming up
    return ev, found, searched_in, at


def _closing_group(ev: onair._ConnectEvidence) -> list:
    """The offsets in ms the rule closed on, read back off the live candidates."""
    n, tol = onair._ConnectEvidence.N, onair._ConnectEvidence.TOL_S
    live = sorted(d for _, d, w in ev.candidates
                  if ev.searched - w < onair._ConnectEvidence.SPAN)
    for i in range(len(live) - n + 1):
        if live[i + n - 1] - live[i] <= tol:
            return [round(d * 1e3, 2) for d in live[i:i + n]]
    return []


def test_connect_events_shape_lines_are_the_only_ones_here() -> None:
    """What the front end made of the call: three shape lines and no station.

    `rxfront`'s contract says `detect`, `fsk` and `p1reply` are shape and that no
    line among them may be read as a station -- nothing in one reads a bit, and
    measured on the corpus `p1reply` also fires on VARA, on PACTOR-2 and on a
    500 Hz-class ARQ station. The three cycles here that raised one are the three
    whose bursts were strongest, and two of them carry a codeword at zero errors.
    """
    raised = {c: kinds for c, (_, _, _, kinds) in _front_end().items() if kinds}
    assert set(raised) == set(SHAPE_ONLY), \
        f"the fixture does not raise the events the arm raised: {raised}"
    assert set(sum(raised.values(), ())) == {"p1reply"}, \
        f"something other than a shape line came out of the call: {raised}"
    for cycle in SHAPE_ONLY:
        count, words, rcvd, _ = _front_end()[cycle]
        assert count and (words, rcvd) == (0, 0), \
            f"cycle {cycle}: {count} event(s), {words} word(s), {rcvd} received"
        assert not _raw_count_gate(cycle), \
            f"cycle {cycle}'s shape line did not consume the old gate after all"
        assert _decoded_gate(cycle), \
            f"cycle {cycle}'s shape line consumes the search anyway"


def test_connect_events_the_old_gate_spent_six_cycles() -> None:
    """The defect, reproduced: three windows skipped, and the link up in cycle 30.

    WS8EOC was on the air from cycle 6 at a turnaround repeatable to 0.12 ms.
    Cycles 13 and 20 hold clean reads of it and cycle 16 holds a burst; all three
    raised a `p1reply`, all three were skipped, and the rule closed fourteen
    searches later on a group 6.25 ms wide.
    """
    ev, found, searched_in, at = _events_replay(_raw_count_gate)
    assert not [c for c in SHAPE_ONLY if c in searched_in], \
        f"the old gate searched a shape cycle: {searched_in}"
    assert [(c, ms) for c, ms, _ in found] == \
        [(c, ms) for c, ms in ACCEPTS if c not in SHAPE_ONLY], \
        f"the replay does not read the call the way the arm did: {found}"
    assert {cs for _, _, cs in found} == {spec.CS_NAMES.index("SPEED-UP")}, \
        f"the accepts are not the CS4 the arm read: {found}"
    assert at == 30, f"the arm's link came up in cycle 30, not {at}"
    assert _closing_group(ev) == [96.25, 97.5, 102.5], _closing_group(ev)


def test_connect_events_the_new_gate_closes_six_cycles_earlier() -> None:
    """...and with the shape lines no longer consuming it, cycle 24.

    The two reads the old gate skipped are the tightest the call holds: 95.00 and
    96.25 ms against cycle 24's 97.50, a group 2.5 ms wide where the old gate
    needed 6.25 and four more cycles of calling to find it.
    """
    ev, found, searched_in, at = _events_replay(_decoded_gate)
    assert set(SHAPE_ONLY) <= set(searched_in), \
        f"a shape cycle was skipped anyway: {searched_in}"
    assert searched_in == sorted(c for c in
                                 {c for c, _ in ACCEPTS} | set(SHAPE_ONLY)
                                 if c <= 24), \
        f"the new gate did not search the call as it ran: {searched_in}"
    assert [(c, ms) for c, ms, _ in found] == \
        [(c, ms) for c, ms in ACCEPTS if c <= 24], \
        f"the searched shape cycles do not read what the sweep found: {found}"
    assert at == 24, f"the call closed in cycle {at}, not 24"
    group = _closing_group(ev)
    assert group == [95.0, 96.25, 97.5], group
    assert group[-1] - group[0] <= 2.5, group


def test_connect_events_a_read_word_still_consumes_the_search() -> None:
    """The other half: a window a DECODER took something in is not searched again.

    The gate is there so the acquisition search runs once per cycle and only
    where nothing has read the burst already -- a second copy of one codeword is
    a REQUEST in PACTOR-1, and the packet counter stops advancing over it. Both
    twelve-bit kinds close it, `unassigned` as much as `cs`: 0x59A in the answer
    slot is a transmission the peer made, whatever PACTOR-1 gives it for a name.
    """
    host = _StubHost()
    sessrx = onair._SessionRx(host, tag="FIX")

    sessrx._on(rxfront.Event(0.50, "p1reply", "1400/1600 Hz burst 230 ms"))
    assert sessrx.count == 1 and not sessrx.words_at, \
        "a shape line reached a decoder's counter"

    sessrx._on(rxfront.Event(0.60, "cs", "CS4/at anchor (0 bit errors)",
                             protocol="PACTOR-1", cs=3, sense=0))
    assert (len(sessrx.words_at), host.rcvd_total) != (0, 0), \
        "a decoded control signal left the connect search open"

    sessrx._on(rxfront.Event(1.85, "unassigned", "0x59A (0 bit errors)",
                             protocol="PACTOR-1"))
    assert len(sessrx.words_at) == 2, \
        "an unassigned word is not counted as a transmission the peer made"


def test_connect_events_cost_no_extra_false_link() -> None:
    """The negative control: searching every window buys no phantom station.

    `connect-events-negatives.json` is every zero-error alignment in every window
    of the same population `connect-ceiling-negatives.json` was priced on -- 1012
    recordings of real off-air energy addressed to nobody, swept with no shape
    gate in front of it, which is the regime the search now runs in. The rule
    closes on ONE recording there, the same single recording the one-accept sweep
    closes on, so nothing in it had to move.

    AND THE TOLERANCE IS WHY THE SPREAD WAS NOT ANSWERED BY WIDENING IT INSTEAD.
    The group this call closes on is 2.5 ms and the old gate's was 6.25; 15 ms
    covers both, and it fires on six of these recordings.
    """
    sweep = json.loads(_fixture(FIXTURES / "connect-events-negatives.json").read_text())
    assert sweep["rule"] == {"N": onair._ConnectEvidence.N,
                             "SPAN": onair._ConnectEvidence.SPAN,
                             "TOL_S": onair._ConnectEvidence.TOL_S}, \
        f"the rule moved since the sweep was taken: {sweep['rule']}"
    assert sweep["tail_s"] == round(onair.ACQUIRE_READ_TAIL_S, 6), \
        "the sweep was taken over a band the search no longer reads"

    one = json.loads(_fixture(FIXTURES / "connect-ceiling-negatives.json").read_text())
    assert sweep["population"]["windows_with_any_accept"] == \
        one["bands"]["read"]["accepts"], \
        "the two sweeps did not see the same accepting windows"
    assert sweep["rules"]["first/tol10"]["fires"] == \
        one["bands"]["read"]["fires"] == 1, sweep["rules"]
    assert sweep["rules"]["best/tol10"]["on"] == \
        sweep["rules"]["first/tol10"]["on"], \
        f"taking every accept in a window invented a link: {sweep['rules']}"
    assert sweep["rules"]["best/tol15"]["fires"] == 6 > \
        sweep["rules"]["best/tol10"]["fires"], sweep["rules"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
