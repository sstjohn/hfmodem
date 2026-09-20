# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Telling a spliced capture from a moving clock, on a phase series with no rig.

`onair-0809-2202` walked its turnaround 90 ms to 30 ms over 87 cycles and three
readings of that were on the table: the path moved, the peer's clock moved, or
the recording is short of the air. Its sidecars predate `_LiveInput.lost` by
eleven days, so the counter cannot answer, and the argument had to be settled on
shape instead -- which means the shape has to be shown to separate the two
BEFORE it is trusted on a session nothing else can grade.

So both inputs are synthesised here at the magnitude the question is asked at.
Losing a block moves the peer's burst once and leaves it where it landed; a
clock off by a few hundred ppm moves it a little every cycle and never stops. At
the detector's 5 ms grid a single cycle cannot tell those apart, and this asserts
what it takes before they can be: the same total movement, delivered both ways,
has to come back with different verdicts, and neither may be read off a plateau
shorter than the quantisation it is averaging down.

The 552 ppm arm is the one worth keeping honest. It is not a plausible
oscillator -- a peer following it would have to hold 5.5 ms of standing error
against an estimator that searches +/-5 -- and it is here precisely because it
is the reading that would have to be believed if the shape did NOT separate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[5] / "tools"

pytestmark = pytest.mark.skipif(not _TOOLS.is_dir(),
                                reason="tools/ not present (installed-wheel run)")


def _tool():
    spec = importlib.util.spec_from_file_location("_tool_grid_phase",
                                                  _TOOLS / "grid_phase.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


#: The detector's own grid, which every reading below is quantised onto: nothing
#: here may depend on resolution `rxfront.p1_burst_onsets` does not have.
GRID_MS = 5.0
#: Cycles, and the movement the 2026-08-09 session showed across them.
CYCLES, MOVED_MS = 87, -60.0


def _series(mod, phase_ms):
    """A phase series on the detector's grid, as `grade` takes it."""
    quantised = np.round(np.asarray(phase_ms) / GRID_MS) * GRID_MS
    return [(i, int(round(p / 1e3 * mod.FS)))
            for i, p in enumerate(quantised)]


def _splice(at, size_ms=MOVED_MS, n=CYCLES, base=90.0):
    return [base + (size_ms if i >= at else 0.0) for i in range(n)]


def _clock(rate_ms=MOVED_MS / CYCLES, n=CYCLES, base=90.0):
    return [base + rate_ms * i for i in range(n)]


def test_a_lost_block_reads_as_a_step_and_a_clock_does_not():
    """The discriminator, stated as the one assertion the session turns on."""
    mod = _tool()
    spliced = mod.grade(_series(mod, _splice(at=CYCLES // 2)))
    drifting = mod.grade(_series(mod, _clock()))

    assert "SPLICED" in spliced.verdict
    assert "RAMPED" in drifting.verdict
    # And NOT because one of them has a step and the other has none: on the
    # detector's grid a ramp is a staircase too, and this one is cut into
    # several. The verdicts turn on which model needs less to explain it.
    assert drifting.steps_ms
    assert drifting.ramp_ms_per_cycle == pytest.approx(MOVED_MS / CYCLES, abs=0.05)

    # Same total movement, so nothing here is separating them on magnitude.
    assert spliced.moved_ms == pytest.approx(drifting.moved_ms, abs=2 * GRID_MS)


def test_the_step_is_recovered_at_the_size_the_capture_lost():
    mod = _tool()
    for lost_ms in (-12.0, -25.0, -60.0):
        r = mod.grade(_series(mod, _splice(at=40, size_ms=lost_ms)))
        assert r.stepped_ms == pytest.approx(lost_ms, abs=GRID_MS), lost_ms


def test_each_model_is_the_better_fit_to_its_own_input():
    """The verdicts are not a threshold on movement; they are which fit wins."""
    mod = _tool()
    spliced = mod.grade(_series(mod, _splice(at=CYCLES // 2)))
    drifting = mod.grade(_series(mod, _clock()))
    assert spliced.plateau_rms_ms < spliced.ramp_rms_ms
    assert drifting.ramp_rms_ms <= drifting.plateau_rms_ms


def test_a_whole_capture_holds_its_phase_through_the_detector_grid():
    mod = _tool()
    rng = np.random.default_rng(0)
    held = 90.0 + rng.normal(0.0, GRID_MS / 2, CYCLES)
    r = mod.grade(_series(mod, held))
    assert not r.steps_ms
    assert abs(r.moved_ms) < 3.0
    assert "HELD" in r.verdict


def test_no_plateau_is_offered_shorter_than_the_quantisation_it_averages():
    """Every split has `PLATEAU_MIN` cycles behind it, wherever the step really is.

    Two cycles either side of a split is the detector's grid twice over and
    nothing else, so a step closer than that to an end is reported at the
    nearest place there is evidence for and at whatever size that supports --
    UNDER the truth, never over it. A capture is not condemned on noise.
    """
    mod = _tool()
    for at in range(1, CYCLES):
        cuts = mod._split(np.asarray(_splice(at=at), float))
        assert all(mod.PLATEAU_MIN <= c <= CYCLES - mod.PLATEAU_MIN for c in cuts)
    late = mod.grade(_series(mod, _splice(at=CYCLES - 3)))
    assert abs(late.stepped_ms) < abs(MOVED_MS)


def test_the_peer_and_our_own_tail_are_not_averaged_together():
    """`phase_series` keeps one burst a cycle, and it is the station's.

    Our T/R tail sits below the 40 ms turnaround floor while the peer answers
    near 90, so a window holding both offers two onsets an entire cycle apart in
    phase. Taking the nearest to a running estimate is what keeps the series on
    one of them; taking a mean would put it between two things and track neither.
    """
    mod = _tool()
    assert mod.NEAR_N < mod.SLOT_N // 2
    # Against ONE cycle's move, not the session's: the estimate is a filter and
    # follows the walk, so what has to clear the gate is the largest single
    # splice this station has made -- 1108 samples, on onair-0821-2110.
    assert mod.NEAR_N > 1108
