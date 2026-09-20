# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What a spliced recording can and cannot be asked, measured on both tools.

`capture clock: ... -4860.19 ppm` is lost capture, not a crystal, so the figure
converts to milliseconds of missing air. Two things about that conversion are
easy to get wrong and both change an answer:

  * the shortfall is a fraction of the ELAPSED time, not of the file, and the
    file is the shorter of the two;
  * a big ppm on a short arm is less lost air than a small one on a long arm,
    so ppm is the wrong sort order.

And `witness_align` is the only measurement that sees where the hole is. Its
claim -- flat runs with a step between them -- is worth nothing unless the
correlator returns a flat line when there is no step and the exact excision when
there is, so that is what is asserted here.

The rest is `--mark`, which is what puts that reading beside the audio for the
sessions recorded before the modem counted its own lost blocks. Those directories
cannot answer for themselves -- every sidecar in them says `xruns: 0` -- so the
mark is how that reading reaches them, and the checks on it are about what it
refuses to touch as much as what it writes. It is not the only way such a session
can be graded: where a peer transmitted on a raster, `tools/grid_phase.py` recovers
the delivered stream's phase against that transmitter and needs no counter and no
second receiver.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[5] / "tools"

pytestmark = pytest.mark.skipif(not _TOOLS.is_dir(),
                                reason="tools/ not present (installed-wheel run)")


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_shortfall_is_a_fraction_of_the_air_not_of_the_file():
    """The 0819 daytime arm, and the one reading where the difference is not small.

    `captures/onair-0819-1747` delivered 2889600 samples at -4860.19 ppm. Charging
    that rate to the 60.2 s of file understates by 70 samples, which nobody would
    notice; the 0819-0022 arm at -354881 ppm is 35% of its own elapsed time, and
    charging it to the file loses a third of the answer -- 30.8 s against 47.7.
    """
    cs = _tool("clock_shortfall")
    assert cs.shortfall_ms(-4860.19, 60.2) == pytest.approx(294.0, abs=0.5)
    naive = 4860.19e-6 * 60.2 * 1e3
    assert cs.shortfall_ms(-4860.19, 60.2) > naive

    big = cs.shortfall_ms(-354881.40, 86.8)
    assert big == pytest.approx(47749, rel=0.01)
    assert big / (354881.40e-6 * 86.8 * 1e3) > 1.5

    # A card that runs fast delivered everything it was asked for.
    assert cs.shortfall_ms(+7.4, 60.2) == 0.0
    assert cs.shortfall_ms(None, 60.2) == 0.0


def test_ppm_and_milliseconds_do_not_rank_the_record_the_same_way():
    """The reason the inventory sorts on one and not the other."""
    cs = _tool("clock_shortfall")
    short_arm = cs.shortfall_ms(-15285.0, 31.0)      # the record's worst ppm
    long_arm = cs.shortfall_ms(-6188.73, 60.2)       # a third of its ppm
    assert short_arm > long_arm                       # ...and still more air lost
    assert cs.shortfall_ms(-750.73, 34.8) < cs.shortfall_ms(-541.55, 63.8)


def _keyed(pattern: np.ndarray, fs: int, tone: float, frame: float) -> np.ndarray:
    n = int(frame * fs)
    t = np.arange(n) / fs
    burst = np.sin(2 * np.pi * tone * t)
    return np.concatenate([burst * v for v in pattern]).astype(float)


def test_the_witness_correlator_finds_an_excision_and_invents_none():
    """A recording that lost nothing must read flat, and one that lost 300 ms
    must read 300 ms -- at the step, not spread over the run."""
    wa = _tool("witness_align")
    fs, rate = 12000, 100.0
    rng = np.random.default_rng(7)
    keys = rng.random(600) < 0.4                 # 60 s of 100 ms keyed frames
    ref = _keyed(keys, fs, 1400.0, 0.1) + 0.01 * rng.standard_normal(600 * fs // 10)

    kb = wa.keydown(ref, fs, (1400.0, 1600.0), rate)
    flat = wa.lag_curve(kb, kb, rate, 0.0, 8.0, 4.0, 1.2)
    assert flat[:, 1].max() == 0.0 and flat[:, 1].min() == 0.0
    assert flat[:, 2].min() > 0.5

    cut, at = int(0.300 * fs), 30 * fs
    spliced = np.concatenate([ref[:at], ref[at + cut:]])
    ka = wa.keydown(spliced, fs, (1400.0, 1600.0), rate)
    lc = wa.lag_curve(ka, kb, rate, 0.0, 8.0, 4.0, 1.2)
    before = lc[lc[:, 0] < 24.0]
    after = lc[lc[:, 0] > 36.0]
    assert np.allclose(before[:, 1], 0.0, atol=0.011), before[:, 1]
    assert np.allclose(after[:, 1], 0.300, atol=0.011), after[:, 1]


def _capture(root: Path, name: str) -> Path:
    """A capture directory as the tool has to recognise one: audio in it."""
    d = root / "captures" / name
    d.mkdir(parents=True)
    (d / "hold_01.wav").write_bytes(b"RIFF-not-really-a-wav")
    return d


def _log(root: Path, name: str, arms) -> Path:
    """A session log in the shape the record's own are in.

    The `session stream:` line is what carried the 2026-08-19 evening's per-arm
    logs, and it is the only line in them that names a directory.
    """
    lines = []
    for stream, ppm, secs, *rest in arms:
        n, sigma = int(secs * 48000), rest[0] if rest else 1.00
        if len(rest) > 1:
            up, threads = rest[1]
            lines.append(f"host: {up} s since boot, {threads} threads in this "
                         f"process")
        lines += [f"capture clock: {n} samples ({secs} s), 0 xruns, 0 TX "
                  f"underruns, {ppm:+.2f} ppm vs the system clock "
                  f"(fit sigma {sigma:.2f})",
                  f"session stream: {stream}/stream.wav -- {secs} s, 4.1 MB"]
    p = root / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_only_the_arms_short_of_the_air_are_marked(tmp_path):
    """The record's own two arms: -4860.19 ppm is 294 ms gone and gets a mark,
    +2.86 ppm is a card running fast and gets nothing.

    Both arms predate `_LiveInput.lost`, so neither directory can answer for
    itself and both look identical from the inside -- `xruns: 0` in every sidecar.
    The session log is the only thing that separates them.
    """
    cs = _tool("clock_shortfall")
    short = _capture(tmp_path, "onair-0819-1747")
    fast = _capture(tmp_path, "onair-0814-2120")
    log = _log(tmp_path, "evening.log", [(short, -4860.19, 60.2),
                                         (fast, +2.86, 60.0)])

    rows = cs.readings([log])
    for r in [r for r in rows if r["lost_ms"] > 0]:
        cs.mark(r["dir"], [r])

    assert not (fast / cs.MARK).exists(), "a card running fast lost nothing"
    got = json.loads((short / cs.MARK).read_text())
    assert got["grid_loss_seen"] is True
    assert got["lost_ms"] == pytest.approx(294.0, abs=0.5)
    assert got["lost_is_a_floor"] is True
    assert got["readings"][0]["from_log"] == [str(log)]


def test_each_reading_carries_the_host_the_arm_ran_on(tmp_path):
    """The ramp is a regression of loss on host state, so the two have to arrive
    on the same row.

    A night's log is a concatenation of arms and every arm prints its own host
    line, so a reading takes the NEAREST ONE ABOVE it. Taking the file's only
    one -- which is what `captures:` gets away with -- would stamp the whole
    night with whatever the first arm ran on, and the night is the measurement.
    """
    cs = _tool("clock_shortfall")
    first = _capture(tmp_path, "onair-0821-2200")
    last = _capture(tmp_path, "onair-0822-0600")
    log = _log(tmp_path, "night.log", [(first, -812.00, 60.0, 1.00, (600, 14)),
                                       (last, -4860.19, 60.2, 1.00, (29400, 21))])

    rows = {r["dir"]: r for r in cs.readings([log])}
    assert (rows[first]["uptime_s"], rows[first]["threads"]) == (600, 14)
    assert (rows[last]["uptime_s"], rows[last]["threads"]) == (29400, 21)


def test_an_arm_that_never_said_answers_for_itself(tmp_path):
    """Every reading on the record before tonight, and the tool still ranks it."""
    cs = _tool("clock_shortfall")
    where = _capture(tmp_path, "onair-0819-1747")
    rows = cs.readings([_log(tmp_path, "old.log", [(where, -4860.19, 60.2)])])
    assert rows[0]["uptime_s"] is None and rows[0]["threads"] is None
    assert rows[0]["lost_ms"] == pytest.approx(294.0, abs=0.5)


def test_the_mark_is_the_only_file_written_and_never_over_someone_else_s(tmp_path):
    """The captures are off-air and unrepeatable, so the tool's whole write
    surface is one new filename, and even that yields to anything already there
    it did not write itself."""
    cs = _tool("clock_shortfall")
    d = _capture(tmp_path, "onair-0819-1747")
    audio = d / "hold_01.wav"
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    log = _log(tmp_path, "arm.log", [(d, -4860.19, 60.2)])
    row = cs.readings([log])[0]

    cs.mark(d, [row])
    assert {p.name for p in d.iterdir()} == set(before) | {cs.MARK}
    assert audio.read_bytes() == before[audio.name]

    # Re-marking its own file is how a corrected reading lands.
    cs.mark(d, [row])
    assert json.loads((d / cs.MARK).read_text())["tool"] == cs.SIGNATURE

    # Somebody else's file at that name is data, and stays data.
    (d / cs.MARK).write_text('{"mine": true}\n')
    assert "REFUSED" in cs.mark(d, [row])
    assert json.loads((d / cs.MARK).read_text()) == {"mine": True}


def test_the_marking_pass_reaches_the_evening_logs_outside_the_tree(tmp_path):
    """`readings` takes files, not only directories.

    Fourteen of this record's readings -- every arm of the 0819 evening the
    PACTOR forensics rests on -- were written to the operator's home and never
    moved into the checkout. A scanner that only walks `working/`, `logs/` and
    `captures/` does not see one of them.
    """
    cs = _tool("clock_shortfall")
    d = _capture(tmp_path, "onair-0819-2048")
    log = _log(tmp_path, "evening-p2-k4msu-pactor.log", [(d, -2262.67, 42.3)])

    rows = cs.readings([log])
    assert len(rows) == 1 and rows[0]["dir"] == d
    assert rows[0]["lost_ms"] == pytest.approx(95.9, abs=0.5)


def test_the_record_itself_ranks_the_two_arms_the_way_the_marking_does():
    """Against the recordings rather than a fixture: the 0819 daytime arm at
    -4860.19 is marked and the 0814 arm at +2.86 is not, read out of the logs as
    they sit on this disk. Nothing is written here.

    Both readings are in `working/`, which is tracked, so this runs on a fresh
    clone. The larger figures in the record -- the 47.7 s arm among them -- are
    quoted from logs that live only in `captures/`, which is not, and asserting
    on those would be a failure and not a skip for anyone who cloned the tree.
    """
    cs = _tool("clock_shortfall")
    if not cs.WORKING.is_dir():
        pytest.skip("the working record is not in this checkout")
    rows = cs.readings([cs.WORKING])
    by_ppm = {r["ppm"]: r for r in rows}
    assert by_ppm[-4860.19]["lost_ms"] == pytest.approx(294.0, abs=0.5)
    assert by_ppm[2.86]["lost_ms"] == 0.0
    # And the order the marking pass inherits is milliseconds throughout.
    short = [r for r in rows if r["lost_ms"] > 0]
    assert len(short) > 1 and short == sorted(short, key=lambda r: -r["lost_ms"])


def test_a_slope_inside_its_own_sigma_is_not_marked_as_loss(tmp_path):
    """`onair-0814-2119` reads -0.93 ppm at fit sigma 1.44 -- three samples in
    64 seconds, and a fit that cannot tell it from a card keeping perfect time.
    Condemning that directory would be the same fault as clearing a spliced one:
    a field saying more than the measurement behind it."""
    cs = _tool("clock_shortfall")
    assert not cs.resolved({"ppm": -0.93, "sigma": 1.44})
    assert cs.resolved({"ppm": -277.74, "sigma": 63.15})
    assert not cs.resolved({"ppm": None, "sigma": None})

    noise = _capture(tmp_path, "onair-0814-2119")
    real = _capture(tmp_path, "onair-0813-2346")
    log = _log(tmp_path, "session.log", [(noise, -0.93, 63.8, 1.44),
                                         (real, -277.74, 35.6, 63.15)])
    short = [r for r in cs.readings([log]) if r["lost_ms"] > 0]

    assert len(short) == 2, "both slopes are negative and both convert to air"
    assert [r["dir"] for r in short if cs.resolved(r)] == [real]


def test_a_second_pass_does_not_cite_the_marks_from_the_first(tmp_path):
    """The mark quotes the `capture clock` line verbatim so a reader has it, and
    the mark lands inside `captures/`, which is scanned. Reading its own output
    back in would have the tool citing itself as a source for the reading."""
    cs = _tool("clock_shortfall")
    d = _capture(tmp_path, "onair-0819-1747")
    log = _log(tmp_path / "captures", "arm.log", [(d, -4860.19, 60.2)])
    cs.mark(d, [cs.readings([tmp_path / "captures"])[0]])

    again = cs.readings([tmp_path / "captures"])
    assert len(again) == 1, "the mark came back as a second reading"
    assert again[0]["where"] == [str(log)]


def _silence(path: Path, seconds: float, fs: int = 12000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(np.zeros(int(seconds * fs), "<i2").tobytes())
    return path


def test_a_witness_that_does_not_overlap_is_an_answer_and_not_a_traceback(
        tmp_path, capsys, monkeypatch):
    """The question the tool exists to answer, asked of a recording that misses.

    Every window is skipped when the witness has no span to search at this lag,
    or when ours is never keyed, and both are the ordinary reading rather than a
    fault: an operator points this at a second receiver precisely to find out
    whether it caught the session at all. So `lag_curve` has to come back the
    right SHAPE when it comes back with nothing -- the columns are indexed by
    name in `main`, and a bare `np.array([])` is one-dimensional -- and `main`
    has to say so in words.
    """
    wa = _tool("witness_align")
    empty = wa.lag_curve(np.zeros(50), np.zeros(5000), 100.0, 0.0, 8.0, 2.0, 2.0)
    assert empty.shape == (0, 3)
    assert empty[empty[:, 2] > 0.5].shape == (0, 3)

    ours = _silence(tmp_path / "ours.wav", 5.0)
    witness = _silence(tmp_path / "witness.wav", 20.0)
    for argv in (["witness_align", str(ours), str(witness)],
                 ["witness_align", str(ours), str(witness), "--guess", "0"]):
        monkeypatch.setattr(sys, "argv", argv)
        wa.main()
        assert "no overlap" in capsys.readouterr().out


#: A session has to have run this far to carry four whole keyed/listening groups.
LOSS_MIN_WINDOWS = 40
#: A keyed cycle hands the reader about 0.24 s of audio and a listening one the
#: whole 1.25 s. Nothing on the record sits between 0.69 s and 1.00 s.
KEYED_MAX_S = 0.9


def _windows():
    """``(name, seconds, lost)`` per session, for every one long enough to read.

    `lost_samples` is cumulative at the moment a sidecar is written, so the step
    into window *n* is the loss taken while window *n* was being captured. The
    other reading -- charging it to the window before -- puts 209876 samples on
    keyed cycles that are quiet under this one, which is how the two are told
    apart.
    """
    from hfmodem.tests import evidence

    out = []
    for directory in sorted(evidence.CAPTURES.glob("onair-*")):
        facts = []
        for side in sorted(directory.glob("rx_*.json")):
            fact = json.loads(side.read_text())
            if "lost_samples" not in fact:
                break
            facts.append(fact)
        if len(facts) >= LOSS_MIN_WINDOWS:
            out.append((directory.name,
                        np.array([f["seconds"] for f in facts]),
                        np.diff(np.r_[0, [f["lost_samples"] for f in facts]])))
    return out


def test_the_capture_loses_in_the_listening_cycles_and_never_in_a_keyed_one():
    """WHERE THE LOST BLOCKS FALL, which is not anywhere and was assumed to be.

    `lost_samples` was read as a rate -- a session's total over its length, ranked
    against other sessions, and a day that ranked high was asked what was different
    about that day. Nothing was. The loss is phased on the session's own cadence,
    not on the wall clock and not on the day: a calling session runs four keyed
    cycles and then six listening ones, forever. The keyed cycles hand the reader
    about 0.24 s -- the rest of the cycle was our own carrier and was flushed --
    and the listening cycles hand it the whole 1.25 s, which the rolling decoder
    then spends holding the interpreter while the audio callback waits for it.
    Machine load has exactly this phase, which is why it was the wrong thing to
    rule out; a state of the converter does not, and stays ruled out.

    Over 66 sessions of 40 windows or more, six days: 1381 keyed windows carrying
    16.8 M samples lost NOTHING, and every one of the 871216 lost samples on the
    record fell in one of the 1777 listening windows.

    ASSERTED ON THE CADENCE AND NOT ON THE WINDOW INDEX, which is what this used
    to say and what the 0829 and 0830 arms overturned. The reading was that the
    loss repeats on a raster of ten windows with two phases that had never lost a
    sample; ten is only the group length, and a session's first group is eleven
    windows in 64 of the 66 (the keyed run ends on a short 0.30 s window of its
    own), so every later group already sat one index off and the "quiet phases"
    were an artefact of every session slipping by the SAME one. A session whose
    listening run is cut short slips by another: `onair-0829-2254` runs groups of
    11, 11, 9, 7 and put 1028 samples on phase 3, `onair-0822-1729` runs 12, 9,
    10, 10 and put 1504 on phase 1, and both windows are ordinary 1.26 s listening
    cycles. Nothing about the capture path changed -- the index was never what the
    finding was about.

    The phase is what a fix has to erase, and `core/gil.py`'s `breathing()` erases
    it on the bench without touching a byte of this historical data -- so this
    still passes, and should. IT HAS NOT ERASED IT ON THE AIR: `breathing()` went
    into `live.py` on 2026-08-26, and the arms flown after it lose 0.84% (0829)
    and 1.02% (0830) of their listening audio against 0.56-0.65% on the three days
    before. Whatever the bench measures, no arm on this record has yet been
    captured clean.
    """
    sessions = _windows()
    if len(sessions) < 10:
        pytest.skip(f"{len(sessions)} sessions carry a per-window loss count")

    secs = np.concatenate([s for _, s, _ in sessions])
    lost = np.concatenate([l for _, _, l in sessions])
    keyed = secs < KEYED_MAX_S

    assert not ((secs >= 0.75) & (secs <= 0.99)).any(), (
        "a window landed in the gap the keyed/listening split is drawn across, "
        "so the split is no longer reading the cadence")
    assert lost.sum() > 0, "no session on this record lost a block"
    assert keyed.sum() > 500, (
        "too few keyed cycles here for their silence to say anything")

    guilty = [(name, int(i) + 1, int(step[i]))
              for name, sec, step in sessions
              for i in np.flatnonzero((sec < KEYED_MAX_S) & (step > 0))]
    assert not guilty, (
        f"a keyed cycle lost a sample, where {keyed.sum()} of them carrying "
        f"{secs[keyed].sum() * 48000 / 1e6:.1f} M samples never have: {guilty}")
