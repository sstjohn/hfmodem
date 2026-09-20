# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What a listen window the PACTOR-1 reader found nothing in is allowed to say.

Three times a peer answered this station in a waveform it has no decoder for, and
three times the log said the same thing it says about an empty channel. On
2026-08-28 that cost a session: WS8EOC filled fifteen consecutive listen windows,
`NO CONTROL SIGNAL -- nothing heard` printed on twelve of them, the strand budget
was spent on a station transmitting in every one, and the link was signed off.

Then, on 2026-09-04, the measure that fixed those three ended a fourth session on
nothing at all. WS8EOC answered five held cycles, granted PACTOR-3 four times at
zero bit errors, and the arm hung up on the two loudest windows of a 28-window
distribution spanning 8.6 dB with nothing but the channel in it -- the first of
them 5.1 s BEFORE the grant, the second 9.9 s after the peer's last codeword,
neither carrying anything any decoder here or SCS's own could name. Both were
already up in the first bin the receiver could hear after unkey and flat to the
end of the window, where every answer that arm did take steps 10-13 dB at the
turnaround.

So the claim under test is narrow and the tests are in four parts: the three real
instances are still not read as silence, the two false ones are readings and not
findings, `-> QRT` is not reachable from either, and the price -- an energy test
is a new false-accept surface, and the population it is bought against is this
station's own listen windows from every arm in which no PACTOR-1 control signal
was ever decoded.

Run: python -m hfmodem.tests.shrike.test_answer_band
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, rxfront, spec

FS = rxfront.FS
REPO = Path(__file__).resolve().parents[5]
CAPTURES = REPO / "captures"

#: The turnaround band a 1.25 s schedule with this station's settle can search.
D_MAX_N = onair._d_max_n(1.25, 0.030)

#: The three sessions a peer filled the answer slot in, and the cycles of each in
#: which the PACTOR-1 reader took nothing. Named rather than derived: `captures/`
#: is ignored and holds whatever the last run wrote, so a glob measures the
#: machine while these measure the checkout.
#:
#:   onair-0828-1844  WS8EOC, 7101.5 kHz. lost_samples 0, +2.74 ppm at fit sigma
#:                    0.01 -- the cleanest receive path on file. Answered CS1 at
#:                    zero bit errors through hold 6, then filled every window to
#:                    the end while the log reported nothing heard.
#:   onair-0829-1343  KB5LZK arm 6, 10144.9 kHz. Answered CS1 through hold 7;
#:                    `max retries ... -> QRT` at hold 16. The operator heard the
#:                    peer leave PACTOR-1 during this arm.
#:   onair-0829-1345  KB5LZK arm 7b. hold_24 is the emission the arm ended
#:                    under -- one window, entirely covered, and `p1_burst_onsets`
#:                    returns nothing for it.
#:
#: They are one population rather than three coincidences: the emission carries the
#: same carrier signature at both gateways and on both bands, characterised
#: elsewhere and not re-derived here.
FILLED = {
    "onair-0828-1844": [f"hold_{n:02d}" for n in range(7, 22)],
    "onair-0829-1343": [f"hold_{n:02d}" for n in range(8, 23)],
    "onair-0829-1345": ["hold_24"],
}

#: ...and the cycles of those same sessions the peer answered PACTOR-1 in. Nothing
#: here may fire on one: a control signal raises a level test as surely as
#: anything else, and a cycle the reader took has nothing left to report.
READ = {
    "onair-0828-1844": [f"hold_{n:02d}" for n in range(1, 7)],
    "onair-0829-1343": [f"hold_{n:02d}" for n in range(1, 8)],
}

#: How many of the filled cycles each session has to keep. Measured at the
#: thresholds `onair.ANSWER_OCCUPIED_DB`, `ANSWER_ONSET_DB` and
#: `ANSWER_PERSIST_DB` stand at, and stated per session because the three are not
#: equally loud -- see the tables beside those constants.
#:
#: THIS IS THE PRICE OF THE ONSET RULE and it is recorded rather than smoothed
#: over: the bare level kept 14, 6 and 1. All three emissions are already under
#: way when the mute lifts, so none of them has an onset and every one of these
#: is kept by persistence -- which cannot see the FIRST cycle of a run, and sees
#: nothing at all of an emission that covers one window and stops. That last case
#: is the whole of what onair-0829-1345 was.
KEEPS = {"onair-0828-1844": 12, "onair-0829-1343": 4, "onair-0829-1345": 0}

#: The measured false-accept rate on the null arms below, as a percentage of
#: scored windows, with a little air over the 0.695% measured. A rate that has
#: crept up is the finding this bound exists to surface. The bare level test
#: measured 1.203% on the same 112 arms.
MAX_FALSE_PCT = 1.0

#: Arms `_null_arms` finds and this population may not be priced against, with
#: what the arm's own log said about its receiver. A null arm is evidence only
#: where the receiver could have heard an answer; where it could not, every
#: window it "accepts" prices the station's own gain and not the level test.
#: Both of these were read out of the 2026-09-13 population, where they carried
#: 66 of 168 accepts between them.
#:
#:   onair-0906-1547  `!! RX 2.009% SATURATED -- ... This capture cannot decode;
#:                    it is NOT evidence about the far end` at rx_03, again at
#:                    0.289% on rx_17, peak 0.99997 in three windows and rms
#:                    exactly 0.0 in two more. Its level moves four orders of
#:                    magnitude window to window and the excess against a floor
#:                    taken at rms 0.0 reaches +50.3 dB. 35 of 44.
#:   onair-0910-2249  Nothing railed, and the log's other disqualification:
#:                    `!! RX level 0.0049 RMS -- receiver may be DEAF ... A
#:                    silent window here is NOT evidence the gateway stayed
#:                    quiet`. The four floor windows are the receiver coming up
#:                    -- rms 0.0049, 0.0114, 0.0154, 0.0573 -- and it then sits
#:                    at 0.059 +-0.003 for every window after, so all 31 score
#:                    +21 dB over a quietest window that was the gain and not
#:                    the channel. 31 of 31: an arm that accepts on every scored
#:                    window is not a false-accept population.
NOT_NULL = {
    "onair-0906-1547": "SATURATED: this capture cannot decode",
    "onair-0910-2249": "floor taken while the receiver was DEAF",
}

#: THE ARM A LEVEL LINE ENDED, `captures/onair-0904-1659` and
#: `working/onair-0904-1659/pactor-eve-02-ws8eoc-observe.log`. FLAGS are the two
#: windows it printed `ANSWER SLOT OCCUPIED` for and took the link down on the
#: second of; ANSWERS are the cycles the peer really did answer in, read out of
#: the same log; AFTER are windows recorded once the link was down and nobody was
#: on the air. The tape carries no PACTOR of any kind in either flag -- no
#: PACTOR-3 header or variable-header anchor from a detector that finds all seven
#: of our own entries in the same file, `spread_score` never over 0.180 against a
#: knee of 0.28, and the independent monitor reading nothing but our own three frames.
ARM = "onair-0904-1659"
FLAGS = ("rx_06", "hold_13")
ANSWERS = ("rx_04", "rx_05", "rx_08", "hold_01", "hold_02")
AFTER = ("hold_25", "hold_90")

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _restore(d: Path, stem: str) -> np.ndarray:
    """A capture at the level it arrived on the air.

    `session.write_wav` scales every file to 0.8 peak, so the level this detector
    compares across windows is exactly what the on-disk file has thrown away. The
    sidecar keeps the pre-normalisation peak for this reason; undoing the scale
    reproduces the array the live session measured, and the `rms` beside it is
    what says so.
    """
    seg = rxfront.load_wav(str(d / f"{stem}.wav"))
    side = d / f"{stem}.json"
    if not side.exists():
        return seg
    j = json.loads(side.read_text())
    if not j.get("normalised_on_write"):
        return seg
    peak = float(np.abs(seg).max()) or 1.0
    return seg * (j["peak"] / peak)


def _stems(d: Path) -> list[str]:
    """Every listen window of a session, in the order the session collected them."""
    return (sorted((f.stem for f in d.glob("rx_*.wav")), key=lambda s: int(s[3:]))
            + sorted((f.stem for f in d.glob("hold_*.wav")), key=lambda s: int(s[5:])))


def _replay(d: Path, read: set[str] = frozenset()
            ) -> dict[str, onair._SlotReading | None]:
    """One session through `_AnswerBand`, window by window, as the loop feeds it."""
    band = onair._AnswerBand()
    out = {}
    for stem in _stems(d):
        seg = _restore(d, stem)
        out[stem] = band.sight(seg, 0, D_MAX_N, read=stem in read, answered=True)
    return out


def _onset(d: Path, stem: str) -> float | None:
    """The step at the turnaround, on the band the session's own loop asks over."""
    seg = _restore(d, stem)
    t0, span = onair._acquisition_window(seg.size, 0, D_MAX_N)
    found = rxfront.answer_onset(seg, t0, t0 + span + spec.P1_CS_S)
    return None if found is None else found.step_db


def _excess(d: Path) -> dict[str, float | None]:
    """The dB the same replay measured, for the lines that report a margin."""
    floor, seen, out = None, 0, {}
    for stem in _stems(d):
        seg = _restore(d, stem)
        t0, span = onair._acquisition_window(seg.size, 0, D_MAX_N)
        level = (rxfront.quiet_level_db(seg, t0, t0 + span + spec.P1_CS_S)
                 if span > 0 else None)
        if level is None:
            out[stem] = None
            continue
        seen += 1
        prev, floor = floor, level if floor is None else min(floor, level)
        out[stem] = (None if prev is None or seen <= onair.ANSWER_FLOOR_WINDOWS
                     else level - prev)
    return out


def _null_arms() -> list[Path]:
    """Capture directories of arms that ended `no PACTOR-1 control signal decoded`.

    Every window of these is a false accept by construction: nothing answered, so
    there was nothing in the answer slot to hear. Each session names its own
    capture directory in its own log, which is what makes this a population rather
    than a glob -- an arm with no log contributes nothing, and the count is
    printed so a shrunken population is visible instead of quietly passing.
    """
    out = set()
    for log in sorted((REPO / "working").rglob("*.log")):
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        if "no PACTOR-1 control signal decoded" not in text:
            continue
        m = re.search(r"captures:\s+(\S+)", text)
        if m and Path(m.group(1)).is_dir():
            out.add(Path(m.group(1)))
    return sorted(p for p in out
                  if any(p.glob("rx_*.wav")) and p.name not in NOT_NULL)


def main() -> int:
    present = [n for n in FILLED if (CAPTURES / n).is_dir()]
    if not present:
        print(f"  [SKIP] no pinned sessions under {CAPTURES}")
        return 2

    print("\nThe three sessions a peer filled the answer slot in")
    fired: dict[str, list[str]] = {}
    for name in present:
        d = CAPTURES / name
        lines = _replay(d, read=set(READ.get(name, ())))
        fired[name] = [s for s, v in lines.items() if v and v.occupied]
        margins = _excess(d)
        want = [s for s in FILLED[name] if s in lines]
        hit = [s for s in want if s in fired[name]]
        check(f"{name}: the filled cycles are not read as silence",
              len(hit) >= KEEPS[name],
              f"{len(hit)} of {len(want)} against {KEEPS[name]} required; "
              f"margins " + " ".join(f"{margins[s]:.1f}" for s in want
                                     if margins.get(s) is not None))
        raised = [s for s in hit if rxfront.p1_burst_onsets(_restore(d, s))]
        check(f"{name}: and the PACTOR-1 detector reads them as nothing",
              not raised, f"onsets in {raised}")
        answered = [s for s in READ.get(name, ()) if s in lines]
        if answered:
            check(f"{name}: a cycle the reader took reports nothing",
                  not [s for s in answered if lines[s] is not None],
                  str([s for s in answered if lines[s] is not None]))

    print("\nThe arm a level line ended, 2026-09-04")
    if not (CAPTURES / ARM).is_dir():
        print(f"  [SKIP] {ARM} is not on disk")
    else:
        d = CAPTURES / ARM
        steps = {s: _onset(d, s) for s in ANSWERS + FLAGS + AFTER}
        print("    " + " ".join(f"{s} {steps[s]:+.1f}" for s in ANSWERS))
        print("    " + " ".join(f"{s} {steps[s]:+.1f}" for s in FLAGS + AFTER))
        check("every cycle the peer answered in STEPS at the turnaround",
              all(steps[s] >= onair.ANSWER_ONSET_DB for s in ANSWERS),
              f"knee {onair.ANSWER_ONSET_DB}")
        check("...and neither flagged window has an edge anywhere in it",
              all(steps[s] < onair.ANSWER_ONSET_DB for s in FLAGS))
        check("...nor has a window recorded with nobody on the air",
              all(steps[s] < onair.ANSWER_ONSET_DB for s in AFTER))
        lines = _replay(d, read=set(ANSWERS))
        flags = [lines[s] for s in FLAGS]
        check("the two flags are still MEASURED and reported",
              all(v is not None for v in flags))
        check("...as a channel reading and not an occupied slot",
              not [v for v in flags if v is not None and v.occupied])
        check("...and they say so in the operator's own words",
              all(v is not None and "CHANNEL READING" in v.line for v in flags))
        after = [lines[s] for s in AFTER]
        check("a window with nobody on the air is neither",
              not [v for v in after if v is not None and v.occupied],
              str([s for s in AFTER if lines[s] and lines[s].occupied]))
        print(f"    {flags[0].line}")

    print("\nWhat the line may and may not say")
    d = CAPTURES / present[0]
    line = next(v.line for v in _replay(d).values() if v and v.occupied)
    print(f"    {line}")
    missing = [w for w in ("not a codeword", "not attributed", "link stays up",
                           "dB over the quietest", "onset") if w not in line]
    check("it reports a level and an onset, names nobody, and keeps the link up",
          not missing, f"missing {missing}")
    check("...and it is not the control-signal line", "CONTROL SIGNAL" not in line)

    print("\nThe line and the cycle come off one call")
    # Built rather than waited for: the wiring is what is under test here, and a
    # session that has to reach a real gateway to exercise it is a session that
    # does not exercise it. Five quiet windows to make a floor, then the two
    # shapes that may hold a cycle and the one that may not.
    rng = np.random.default_rng(7)
    n = int(0.243 * FS)
    quiet = [rng.normal(0, 0.02, n) for _ in range(5)]
    # Energy that ARRIVES, at a turnaround this station could be answered at...
    edge = quiet[0].copy()
    at = int(0.075 * FS)
    edge[at:] += rng.normal(0, 0.10, edge.size - at)
    # ...and energy that was already there when the band opened, which is the
    # shape both windows of 2026-09-04 came in.
    flat = quiet[0] + rng.normal(0, 0.10, n)
    band, told = onair._AnswerBand(), []
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for seg in quiet + [edge]:
            onair._report_answer_band(band, _Host(told), _Grid(True), seg, 0,
                                      D_MAX_N, read=False)
    check("a window energy arrives in prints once",
          log.getvalue().count("ANSWER SLOT OCCUPIED") == 1,
          f'{log.getvalue().count("ANSWER SLOT")} lines from six windows')
    check("...says which of the two things it stood on",
          "begins inside the window" in log.getvalue())
    check("...and the cycle budget is told once", told == ["unreadable"], str(told))
    band, told = onair._AnswerBand(), []
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for seg in quiet + [flat, flat]:
            onair._report_answer_band(band, _Host(told), _Grid(True), seg, 0,
                                      D_MAX_N, read=False)
    check("a window that was loud before it opened is a reading first",
          log.getvalue().count("CHANNEL READING") == 1, log.getvalue())
    check("...and the cycle budget is told nothing about that one",
          log.getvalue().index("CHANNEL READING")
          < log.getvalue().index("ANSWER SLOT OCCUPIED"))
    check("...but the same reading again, in the same place, is an occupancy",
          log.getvalue().count("ANSWER SLOT OCCUPIED") == 1
          and "in the cycle before" in log.getvalue(), log.getvalue())
    check("...and holds exactly one cycle", told == ["unreadable"], str(told))
    band, told = onair._AnswerBand(), []
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for seg in quiet + [edge]:
            onair._report_answer_band(band, _Host(told), _Grid(False), seg, 0,
                                      D_MAX_N, read=False)
    check("a grid nothing has answered on says nothing",
          not log.getvalue() and not told, log.getvalue())

    print("\nNothing downstream takes it for a codeword")
    a = arq.PactorArq(_Silent(), arq.ArqConfig())
    a.role, a.state = arq.IRS, arq.State.CONNECTED
    before = (a.state, a.role, a._next_seq, a._expected_seq, a._inflight,
              a._rx_this_cycle, a._breakin_pending)
    a.note_unreadable_answer()
    after = (a.state, a.role, a._next_seq, a._expected_seq, a._inflight,
             a._rx_this_cycle, a._breakin_pending)
    check("it advances no counter and authorises no changeover", before == after)
    check("...and the cycle is not a silent one", a._burst_at_anchor)
    check("...and the connect budget hears it", a._peer_heard)

    print("\nWhere the ISS strand ends, and it is not here")
    # The 2026-09-04 arm hung up on the second of two of these, five cycles after
    # a gateway granted PACTOR-3 four times over. A finding this station cannot
    # name may hold a cycle; the ending is the retry budget's, and a station
    # transmitting into every window ends at the same cycle as an empty channel.
    ends = {}
    for occupied in (False, True):
        told = []
        a = _iss(told)
        for n in range(1, 4 * a.cfg.max_retries):
            if occupied:
                a.note_unreadable_answer()
            a.on_cycle()
            if a._qrt_pending:
                ends[occupied] = (n, told[-1])
                break
    check("an occupied answer slot never ends the strand on its own",
          ends.get(True, (None,))[0] == a.cfg.max_retries + 1,
          str(ends.get(True)))
    check("...and ends it exactly where an empty channel would",
          ends.get(True, (None,))[0] == ends.get(False, (None,))[0],
          f"{ends.get(True)} against {ends.get(False)}")
    check("...on the retry budget, which is the only thing that says QRT",
          all("max retries" in line for _, line in ends.values()),
          str(ends))
    check("...and nothing anywhere still signs off on the level",
          not [line for _, line in ends.values()
               if "no reader here took" in line], str(ends))

    print("\nArms in which nothing ever answered")
    arms = _null_arms()
    for name, why in sorted(NOT_NULL.items()):
        print(f"    not a null arm: {name} -- {why}")
    if not arms:
        print("  [SKIP] no null-arm logs with their captures on disk")
    else:
        scored = accepts = readings = 0
        for d in arms:
            band = onair._AnswerBand()
            for stem in _stems(d):
                try:
                    seg = _restore(d, stem)
                except Exception:
                    continue
                # The band gate forced open -- none of these arms ever acquired a
                # turnaround, so in a session every one of them is silent here by
                # construction. What is being priced is the level test itself.
                seen = band.sight(seg, 0, D_MAX_N, read=False, answered=True)
                if band.windows > onair.ANSWER_FLOOR_WINDOWS:
                    scored += 1
                    readings += seen is not None
                    accepts += seen is not None and seen.occupied
        pct = accepts / max(scored, 1) * 100
        check(f"the false-accept rate stays bought ({len(arms)} arms)",
              scored > 1000 and pct <= MAX_FALSE_PCT,
              f"{accepts} of {scored} scored windows = {pct:.3f}% "
              f"against {MAX_FALSE_PCT}%, from {readings} level readings "
              f"({readings / max(scored, 1) * 100:.3f}%)")

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


class _Grid:
    """The one thing `_report_answer_band` asks the raster."""

    def __init__(self, acquired: bool) -> None:
        self.acquired = acquired


class _Host:
    """A host that records which of `arq`'s doors was opened, and nothing else."""

    def __init__(self, told: list[str], protocol=None, role=None,
                 state=None) -> None:
        self.arq = self
        self._told = told
        # A PACTOR-1 link by default, which is the one these windows were
        # recorded on: the reading is attributed to the peer only where our own
        # role says the peer owes us a data packet in the slot being measured.
        self.protocol = protocol
        self.role = role
        self.state = state

    def note_unreadable_answer(self) -> None:
        self._told.append("unreadable")


class _Silent:
    """The `io` an FSM needs to exist, doing none of it."""

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _Recording(_Silent):
    """...and the same, keeping the lines an operator would have read."""

    def __init__(self, told: list[str]) -> None:
        self._told = told

    def log(self, line: str) -> None:
        self._told.append(line)


def _iss(told: list[str]) -> "arq.PactorArq":
    """A station holding the sending role with one unacknowledged packet out.

    Where both sign-offs were: twenty bytes keyed as packet #1, never
    acknowledged, and the peer's answer slot the only thing left to read.
    """
    a = arq.PactorArq(_Recording(told), arq.ArqConfig())
    a.role, a.state = arq.ISS, arq.State.CONNECTED
    a._next_seq = a._expected_seq = 1
    a._outbuf.extend(b"the quick brown fox!")
    a._start_next_packet()
    return a


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"no pinned sessions under {CAPTURES}")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
