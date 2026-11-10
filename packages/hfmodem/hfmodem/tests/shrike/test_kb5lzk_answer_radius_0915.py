# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two peer answers the 40 m arm had in its window and did not read.

KB5LZK, 40 m, 2026-09-15 23:46Z (`arm-post-v31-B-assessed-40-kb5lzk-20260915T234606Z`).
The tape carries an answer on thirty-three cycles; the log printed `HOLD RX
(quiet)` on two of them, at 37.24 s and 53.50 s. Both words are in the arm's own
hold window, at the instant the peer's own cadence projects, and both sit one bit
from a control signal when the two carriers are summed -- one has channel 5 faded
and the other channel 12, so the surviving carrier carries the word and its
partner spoils a bit of it.

What refused them was `CS_EXPECTED_MAX_ERRORS`, PACTOR-1's twelve-bit rule,
reaching a twenty-bit PACTOR-III codeword by way of `control_signal_at`'s
default -- while the same reader holds a LOCK to radius 1. A projection from the
peer's last decoded answer is a sharper prior than a lock, so it gets the tighter
bracket and the code's own radius; `SyncedRx.ANSWER_TRACK_SYMBOLS` carries the
false-accept measurement.

The third quiet cycle of that arm, hold 28, carries no word at any radius on
either carrier. It is here as the negative control.

Run:  python -m pytest hfmodem/tests/shrike/test_kb5lzk_answer_radius_0915.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session

FIXTURE = Path(__file__).with_name("fixtures") / "kb5lzk-answer-0915"
DROPPED = (23, 36)
"""The two holds whose answer never reached the ARQ: 37.24 s and 53.50 s."""
SILENT = 28
CLEAN = (22, 35)


@pytest.fixture(scope="module")
def windows():
    if not (FIXTURE / "metadata.json").exists():
        pytest.skip(f"no {FIXTURE.name} cuts under {FIXTURE.parent}")
    meta = json.loads((FIXTURE / "metadata.json").read_text())
    out = {}
    for row in meta["rows"]:
        path = FIXTURE / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        fs, pcm = wavfile.read(path)
        assert fs == onair.FS and len(pcm) == row["end"] - row["start"]
        out[row["hold"]] = (row, pcm.astype(float) / 32768)
    return meta, out


def answered(meta, row, pcm, *, legacy=False):
    """One cycle through the session reader, at the peer's projected instant."""
    s = _Session(role=arq.ISS)
    s.rx.p3_receive_offset_hz = meta["correction_hz"]
    s.rx._p3_answer_at = row["previous_answer"]
    s.rx.new_cycle()
    if legacy:
        # The read this path made before the peer's own cadence was believed.
        s.rx.sync.control_signal_tracked = (
            lambda audio, at, **kw: s.rx.sync.control_signal_at(audio, at, **kw))
    got = s.rx.control_signal(pcm, row["start"],
                              row["previous_answer"] + meta["cycle"])
    return got, s.rx._p3_answer_at


@pytest.mark.parametrize("hold", DROPPED)
def test_a_faded_carrier_no_longer_costs_the_whole_answer(windows, hold):
    meta, rows = windows
    row, pcm = rows[hold]
    assert answered(meta, row, pcm, legacy=True)[0] is None
    cs, at = answered(meta, row, pcm)
    assert cs == arq.CS_ACK
    assert abs(at - (row["previous_answer"] + meta["cycle"])) <= rxfront.SPS


def test_the_cycle_that_carried_nothing_still_carries_nothing(windows):
    """The arm's third quiet hold, and the reason the other two are not noise."""
    meta, rows = windows
    row, pcm = rows[SILENT]
    assert answered(meta, row, pcm)[0] is None
    assert answered(meta, row, pcm, legacy=True)[0] is None


@pytest.mark.parametrize("hold", CLEAN)
def test_the_cycles_that_already_read_are_unchanged(windows, hold):
    meta, rows = windows
    row, pcm = rows[hold]
    assert answered(meta, row, pcm)[0] == answered(meta, row, pcm, legacy=True)[0] \
        == arq.CS_ACK


def test_the_narrowed_bracket_fabricates_no_codeword(windows):
    """What the radius costs, measured where the live read makes it.

    A twenty-bit word, six codewords, no CRC: every alignment offered is another
    chance to manufacture one, and the four-symbol bracket offers 129. On the
    one-symbol bracket a projection from the peer's own answer deserves, radius
    1 accepts nothing in 200 windows of white noise the size of the arm's own --
    the record radius 0 has at four symbols.
    """
    rng = np.random.default_rng(20260916)
    n = len(next(iter(windows[1].values()))[1])
    sync = rxfront.SyncedRx()
    assert not [i for i in range(200)
                if sync.control_signal_tracked(rng.normal(0, .1, n), 3700,
                                               details=False) is not None]


TAPE_MS = 902.4
"""Where the tape puts the answer on all thirty clean cycles, +- 1.5 ms."""


def test_the_position_instrument_reads_the_codeword_and_not_the_envelope(windows):
    """One number per cycle, and none at all for the cycle with no word in it.

    `ENTRY ANSWER POSITION` takes its milliseconds from the onset detector, and
    on this arm that reading moved 140 ms over five cycles while the peer did
    not move at all. The instant the codeword's matched filter has already
    placed is the same measurement done properly: the two clean cycles here land
    on the tape's figure, the cycle whose channel 5 faded lands a symbol early,
    and the silent cycle reports nothing rather than last cycle's answer.
    """
    meta, rows = windows
    got = {}
    for hold, (row, pcm) in rows.items():
        s = _Session(role=arq.ISS)
        s.rx.p3_receive_offset_hz = meta["correction_hz"]
        s.rx._p3_answer_at = row["previous_answer"]
        s.rx.new_cycle()
        due = row["previous_answer"] + meta["cycle"]
        s.rx.control_signal(pcm, row["start"], due)
        boundary = due - round(TAPE_MS / 1e3 * onair.FS)
        got[hold] = s.rx.tracked_answer_position_ms(boundary, meta["cycle"])
    assert got[SILENT] is None
    for hold in CLEAN:
        assert abs(got[hold] - TAPE_MS) <= 2, got
    assert max(v for v in got.values() if v is not None) \
        - min(v for v in got.values() if v is not None) <= 10
