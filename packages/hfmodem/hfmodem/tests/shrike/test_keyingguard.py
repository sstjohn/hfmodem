# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What stands between a rendered burst and the peer's air, read off one session.

The 2026-08-26 WS8EOC arm ended on the operator's ear: they heard this station
transmitting over the gateway. Its log is the evidence and it is committed beside
this file, so every figure below is one a run printed rather than one a round
trip through our own encoder produced.

Three readings of the same keying disagree in that log -- `--settle 0.040`, a
measured `PTT keyed - audio` of 28 ms, and 51 ms of carrier ahead of the body on
the rig's own monitor tap -- and they reconcile because they are three different
intervals that sum: 28 ms of carrier ahead of the nominal DAC instant, plus the
20.8-23.9 ms our own audio arrives late through the transmit chain, against a tap
median of 51. What is testable without the tap is here: which
instant each guard was asked about, and what the numbers in the log are made of.

Run:  pytest hfmodem/tests/shrike/test_keyingguard.py
"""
from __future__ import annotations

import contextlib
import io
import re

import pytest

from hfmodem.shrike import spec
from hfmodem.shrike.onair import (FS, P3_CS_N, _MasterGrid,
                                  _report_collision)
from hfmodem.tests import evidence

LOG = evidence.WORKING / "pactor-day-02-ws8eoc-2026-08-26.log"
pytestmark = pytest.mark.skipif(not LOG.exists(), reason=f"no {LOG}")

CYCLE_S = 1.25
SLOT_N = round(CYCLE_S * FS)
SETTLE_S = 0.040                    # what the arm was launched with
BURST_N = round(spec.P1_CS_S * FS)


@pytest.fixture(scope="module")
def log() -> list[str]:
    return LOG.read_text().splitlines()


def test_the_wandering_guard_is_one_anchor_seen_modulo_the_cycle(log) -> None:
    """The session's `[predict]` leads read +24, +35, ..., +652 and once -11, and
    the report of the day called that a guard wandering 663 ms inside one session.

    It is not a guard and it does not wander. The printed lead is
    `boundary_after(nxt) - nxt - cs - settle`, and the first term is modular in
    the cycle: it sweeps the whole 1.25 s as the source's phase moves against our
    anchor, so the figure is a PHASE and its range is the cycle by construction.
    Folding every lead back through that arithmetic has to land on ONE anchor,
    and it does -- to under a millisecond, which is what the line rounds to.

    The transmit anchor moves once mid-session (`GRID REVERSED -> IRS: transmit
    anchor +840 ms`), and the leads after it fold onto a second anchor exactly
    840 ms from the first. The reversal is visible in the same arithmetic rather
    than an exception to it, and it is the only thing in the session that moves
    the phase these leads are made of.
    """
    rev = next(i for i, l in enumerate(log) if "GRID REVERSED -> IRS" in l)
    before, after = [], []
    for i, line in enumerate(log):
        m = re.search(r"repeat at (\d+): we key ([+-]\d+) ms", line)
        if m:
            nxt, lead = int(m.group(1)), int(m.group(2)) / 1e3
            (before if i < rev else after).append(
                (nxt + round((lead + spec.P1_CS_S + SETTLE_S) * FS)) % SLOT_N)
    assert (len(before), len(after)) == (17, 5)
    for name, group in (("before the reversal", before), ("after it", after)):
        span = (max(group) - min(group)) / FS * 1e3
        assert span < 1.0, (
            f"{name}: the predicted keys fold onto anchors {span:.1f} ms apart, "
            f"so the printed lead is not one anchor read modulo the cycle")
    step = (after[0] - before[0]) % SLOT_N / FS * 1e3
    assert abs(step - 840) < 1.0, (
        f"the anchor moved {step:.1f} ms where the reversal logged +840")


def test_the_rf_started_line_never_reported_the_carrier(log) -> None:
    """`RF started %+.1f ms from the slot boundary` was the audio's first sample
    against the boundary it was scheduled on, and `transmit` places one AT the
    other -- so the line could only ever print zero, whatever the rig was doing.

    Forty-six keyings, two distinct values, both inside two milliseconds. The same
    bursts logged a keyed lead of 28 to 33 ms, which is where the carrier actually
    was, and the connect bursts -- which never go through the gridded path -- kept
    the whole 40. A line that cannot vary is not a measurement, and this one was
    the only place the session claimed to say where the carrier came up.
    """
    body = "\n".join(log)
    rf = [float(v) for v in re.findall(r"RF started ([+-][\d.]+) ms", body)]
    leads = [round((float(k) - float(d)) * 1e3) for k, d in
             re.findall(r"PTT keyed ([\d.]+) s for ([\d.]+) s of audio", body)]
    assert len(rf) >= 40 and len(leads) >= len(rf)
    assert max(abs(v) for v in rf) < 2.0, sorted(set(rf))
    gridded = [v for v in leads if v < round(SETTLE_S * 1e3)]
    assert len(gridded) >= 30 and 26 <= min(gridded) <= max(gridded) <= 34, (
        f"the gridded keyed leads are {sorted(set(gridded))}, not the "
        "settle-less-holdback band the carrier actually sat in")
    assert max(leads) == round(SETTLE_S * 1e3), sorted(set(leads))


def _grid() -> _MasterGrid:
    return _MasterGrid(0, SLOT_N, round(0.185 * FS),
                       packet_n=round(spec.P1_PACKET_S * FS), cs_n=BURST_N,
                       d_max_n=round(0.19 * FS))


def test_a_burst_that_ends_inside_our_settle_reads_as_a_collision() -> None:
    """The overlap ran from the audio, so the whole PTT lead sat outside it: a
    source that stopped 30 ms after our key went up still ended 10 ms before our
    first modulated sample, and came out `[clear]`.

    Geometry from the arm -- a 40 ms settle and a 120 ms codeword -- and both
    readings are run over it so the difference is the argument.
    """
    settle_n = round(SETTLE_S * FS)
    onset = 3 * SLOT_N
    key_up = onset + BURST_N - round(0.030 * FS)   # their last 30 ms under us
    audio = key_up + settle_n
    end = audio + round(0.96 * FS)

    class _Tx:
        settle = SETTLE_S
        tx_key_up, tx_audio_start, tx_end = key_up, audio, end

    assert min(onset + BURST_N, end) - max(onset, audio) < 0, (
        "the audio-start reading has to call this geometry clear, or the test "
        "is not about the settle")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _report_collision([(onset, BURST_N)], _Tx())
    first = buf.getvalue().splitlines()[0]
    assert "[collide]" in first and "30 ms" in first, first


def test_the_burst_rung_cannot_be_keyed_every_cycle(log) -> None:
    """Why eight cycles ran BEHIND THE GRID, and it is not what closed task #20.

    Nothing discards anything now -- the message says so and the window is held.
    What the session did was key every OTHER slot for the whole `burst` rung, and
    `_keyable_slot`'s listen floor is why: the rung renders the entry packet
    behind a 200 ms acquisition preamble, 1.074 s of audio, which leaves 136 ms
    of channel in front of the next key against the 210 ms a PACTOR-3 codeword
    needs. The `template` rung's 0.869 s leaves 341 and keyed every cycle.

    So the two rungs of that grant were not offered to the peer on the same
    cadence, and neither can be scored against the other from this session.
    """
    room = {d: CYCLE_S - d - SETTLE_S for d in (0.869, 1.074)}
    assert room[0.869] * FS >= P3_CS_N > room[1.074] * FS, room

    steps, last, pending = {}, None, None
    for line in log:
        hold = re.search(r"hold (\d+) slot (\d+)", line)
        if hold:
            slot = int(hold.group(2))
            if last is not None and pending is not None:
                steps.setdefault(pending, []).append(slot - last)
            last = slot
        tx = re.search(r"TX\[\d+\] (.+?)\s+\(", line)
        if tx:
            pending = tx.group(1)
    template, burst = steps["SL1 ENTRY 0B"], steps["SL1 ENTRY 0B +burst"]
    assert len(template) >= 3 and set(template) == {1}, template
    assert len(burst) >= 5 and min(burst) >= 2, burst
