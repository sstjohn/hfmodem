# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`tools/alc_knee.py` off the radio: the buffer, the search and the transcript.

The instrument's claim is that what the operator drives past the knee is the
packet an arm keys and not something shaped like it, so that is the assertion
here -- the loop it hands the card, times the drive, is `core.levels.at_drive`
of `placement.link_packet`'s own entry, sample for sample. A knee measured on a
second render of the entry would be a knee for a waveform this station never
transmits, and nothing in the log would say so.

The rest is what the operator's hands touch: the drive clamp (0 and 1 are the
card's own bounds, and a held arrow key must not walk past either), the waveform
ring, the bisection that decides how many key-downs a leg costs, and the
transcript, which is the whole deliverable of item 12a.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import levels
from hfmodem.shrike import pactor1, placement, spec

_TOOLS = Path(__file__).resolve().parents[5] / "tools"

pytestmark = pytest.mark.skipif(not _TOOLS.is_dir(),
                                reason="tools/ not present (installed-wheel run)")


def _tool(name: str):
    spec_ = importlib.util.spec_from_file_location(f"_tool_{name}",
                                                   _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec_)
    sys.modules[spec_.name] = mod
    spec_.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def alc():
    return _tool("alc_knee")


def _wave(alc, name: str, gap: bool = True):
    return alc.render(name, tone_hz=1500, call="W9SSJ", gap=gap)


# -- the buffer ---------------------------------------------------------------

def test_the_entry_is_the_transmitters_own_render_at_every_drive(alc):
    """The packet a grant is answered with, scaled rather than rebuilt.

    `RadioTx._entry_burst` builds `link_packet(1, b"", 0x1a, flush=ENTRY_FLUSH)`
    and `_tx` puts it through `levels.at_drive`. The tool normalises once at 1.0
    and multiplies at play time, so the two have to agree at whatever drive the
    operator stops on -- which is what makes the reading a reading of the arm.
    """
    unit = _wave(alc, "entry", gap=False).unit
    keyed = placement.link_packet(alc.ENTRY_SL, b"", alc.ENTRY_STATUS,
                                  swapped=False, flush=placement.ENTRY_FLUSH)
    for drive in (0.30, 0.62, 0.97, 1.00):
        assert np.allclose(unit * np.float32(drive),
                           levels.at_drive(keyed, drive), atol=2e-7)


def test_the_announcement_is_the_one_the_call_goes_out_as(alc):
    unit = _wave(alc, "p1", gap=False).unit
    keyed = pactor1.connect_signal("W9SSJ", lead_s=0.0, tail_s=0.0)
    assert np.allclose(unit * np.float32(0.62), levels.at_drive(keyed, 0.62),
                       atol=2e-7)


def test_the_cycle_gap_pads_the_burst_and_never_touches_it(alc):
    """The duty an ARQ link keys at, with the packet left as it was.

    An entry is 0.86 s of the 1.25 s cycle. Padding is what gives the needle its
    recovery between packets; a pad that clipped or rescaled the burst would
    move the knee it is there to expose.
    """
    bare, cycled = _wave(alc, "entry", gap=False), _wave(alc, "entry")
    assert cycled.seconds == pytest.approx(spec.CYCLE_SHORT_S)
    assert np.array_equal(cycled.unit[:cycled.burst_n], bare.unit)
    assert not cycled.unit[cycled.burst_n:].any()
    assert 0.5 < cycled.duty < 0.8


def test_the_tone_wraps_without_a_discontinuity(alc):
    """A whole second of an integer frequency, so the loop seam is phase-continuous.

    The pump repeats this buffer under a live carrier. A seam is a step in the
    waveform, which is a click the ALC would meet and the band would hear.
    """
    unit = _wave(alc, "tone").unit
    joined = np.concatenate([unit, unit])
    step = np.abs(np.diff(joined))
    assert step[len(unit) - 1] <= step.max()
    assert unit.max() == pytest.approx(1.0, abs=1e-6)


# -- the operator's hands -----------------------------------------------------

@pytest.mark.parametrize("start, delta, want", [
    (0.50, 0.02, 0.52),
    (0.50, -0.02, 0.48),
    (0.95, 0.10, 1.00),
    (0.05, -0.10, 0.00),
    (1.00, 0.02, 1.00),
    (0.00, -0.02, 0.00),
])
def test_the_drive_steps_and_clamps_to_the_cards_own_bounds(alc, start, delta,
                                                            want):
    assert alc.stepped(start, delta) == pytest.approx(want)


def test_a_held_arrow_key_cannot_walk_off_either_end(alc):
    drive = 0.50
    for _ in range(80):
        drive = alc.stepped(drive, 0.02)
    assert drive == 1.0
    for _ in range(80):
        drive = alc.stepped(drive, -0.02)
    assert drive == 0.0


def test_the_waveform_ring_returns_to_where_it_started(alc):
    name = "tone"
    seen = [name]
    for _ in range(len(alc.WAVEFORMS) - 1):
        name = alc.cycle_waveform(name)
        seen.append(name)
    assert sorted(seen) == sorted(alc.WAVEFORMS)
    assert alc.cycle_waveform(name) == "tone"


# -- the search ---------------------------------------------------------------

def _run(alc, knee: float, step: float = 0.02):
    """Answer every probe as a rig whose ALC starts acting above `knee`."""
    hunt = alc.Bisect(step)
    while (probe := hunt.next_probe()) is not None:
        hunt.answer(probe, probe > knee + 1e-9)
    return hunt


def test_the_bisection_finds_the_knee_within_one_step(alc):
    for knee in (0.35, 0.50, 0.66, 0.81):
        hunt = _run(alc, knee)
        assert knee - 0.02 <= hunt.knee <= knee
        assert len(hunt.answers) <= 8


def test_a_rig_the_alc_never_touches_is_named_rather_than_given_a_knee(alc):
    hunt = alc.Bisect(0.02)
    while (probe := hunt.next_probe()) is not None:
        hunt.answer(probe, False)
    assert hunt.knee == 1.0
    assert "no knee" in hunt.verdict


def test_an_alc_already_acting_at_the_first_question_reports_no_linear_region(alc):
    hunt = alc.Bisect(0.02)
    while (probe := hunt.next_probe()) is not None:
        hunt.answer(probe, True)
    assert hunt.knee == 0.0
    assert "no linear region" in hunt.verdict


def test_a_contradictory_answer_does_not_move_the_knee(alc):
    """The arrow keys can walk outside the bracket, so an `n` above a `y` is
    reachable. It stays in the log and out of the reading."""
    hunt = alc.Bisect(0.02)
    hunt.answer(0.40, False)
    hunt.answer(0.50, True)
    hunt.answer(0.80, False)
    assert hunt.hi == 0.50
    assert hunt.knee == 0.40
    assert (0.80, False) in hunt.answers


# -- the transcript -----------------------------------------------------------

def test_the_transcript_carries_the_station_and_one_row_per_point(alc):
    """What the operator pastes back: header comments, the rows, the findings."""
    power = alc.Power(0.60, 60.0, 7101500.0)
    points = [alc.Point("2026-09-16T01:02:03Z", "tone@RFPOWER 0.60", "tone",
                        0.62, power, "0.750", True, "ALC flicker")]
    text = alc.transcript({"git HEAD": "abc1234", "codec": "USB Audio Device"},
                          points, ["knee 0.62"])
    lines = text.splitlines()
    assert lines[0] == "# git HEAD: abc1234"
    assert lines[1] == "# codec: USB Audio Device"
    assert lines[2].split("\t") == list(alc.COLUMNS)
    assert lines[3].split("\t") == ["2026-09-16T01:02:03Z", "tone@RFPOWER 0.60",
                                   "tone", "0.62", "0.60", "60", "0.750", "yes",
                                   "ALC flicker"]
    assert lines[-1] == "# knee 0.62"
    assert text.endswith("\n")


def test_an_unreadable_power_leaves_the_columns_empty_rather_than_lying(alc):
    """`rfpower-is-read-back-not-set`: a CAT read that did not answer is a hole
    in the table, and a hole an operator can see is worth more than a zero."""
    blank = alc.Power(None, None, None)
    assert str(blank) == "unreadable"
    row = alc.Point("t", "leg", "entry", 0.97, blank, "unreadable", False,
                    "").row()
    assert row[4] == "" and row[5] == ""
    assert row[7] == "no"


def test_the_findings_name_the_entrys_knee_against_the_tones(alc):
    """The number item 12b is waiting on: how much crest the ALC charges.

    A knee of 0.52 on the tone and 0.24 on the entry is 6.7 dB of headroom the
    DPSK does not get, and the arms' own 0.97 sits above both.
    """
    power = alc.Power(1.0, 100.0, 7101500.0)
    legs = [alc.Leg("tone", power, 1.0, _run(alc, 0.52), "100"),
            alc.Leg("entry", power, 1.0, _run(alc, 0.24), "75")]
    text = "\n".join(alc.findings(legs))
    assert "-6.7" in text or "-6.8" in text
    assert "ABOVE the knee" in text
    assert "highest linear drive for the P3 phases at RFPOWER 1.0: 0.24" in text
    assert "cannot carry the drive ratio" in text


def test_no_findings_without_a_leg(alc):
    assert "no knee to report" in alc.findings([])[0]


def test_a_silent_cat_still_names_the_power_the_operator_set(alc):
    """A run where the adapter never answered still has two powers in it.

    The read-back is one string for the whole run in that case, so the derived
    rows key off the power the operator was ASKED for; folding them by the CAT
    value would report one leg pair where four legs were keyed.
    """
    blank = alc.Power(None, None, None)
    legs = [alc.Leg("tone", blank, None, _run(alc, 0.52), ""),
            alc.Leg("entry", blank, None, _run(alc, 0.24), ""),
            alc.Leg("tone", blank, 1.0, _run(alc, 0.40), ""),
            alc.Leg("entry", blank, 1.0, _run(alc, 0.18), "")]
    text = "\n".join(alc.findings(legs))
    assert "at RFPOWER as set: the entry's knee is -6.7" in text
    assert "at RFPOWER 1.0 (by hand): the entry's knee is -6.9" in text
    assert "highest linear drive for the P3 phases at RFPOWER 1.0: 0.18" in text
