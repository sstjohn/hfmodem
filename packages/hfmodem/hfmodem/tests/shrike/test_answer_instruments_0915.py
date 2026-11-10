# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The two slot instruments, wired to the measurements that already existed.

Both come off KB5LZK's 40 m arm of 2026-09-15
(`arm-post-v31-B-assessed-40-kb5lzk-20260915T234606Z`), and both were reports
about the channel where a report about the peer was available for free.

`ENTRY ANSWER POSITION` takes its milliseconds from the nearest onset, and an
envelope finds where the channel got louder rather than where the word is: over
five cycles of that arm the onset reading moved 140 ms -- 1015.3, 1044.0, 938.1,
1033.0, 904.0 -- while the peer answered at 902.4 ms every time and the session's
own matched filter had already placed it there. `_SessionRx.tracked_answer_position_ms`
is that instant folded onto the slot, and the cycles with no word in them still
get the onset, because an instrument reports what it measured.

`ANSWER SLOT LEVEL`/`OCCUPIED` said how loud a slot was and how wide, and width
is not a mode test -- band noise fills a passband and measures wider than the
signal it hides. `rxfront.p3_comb_db` asks the one question that needs no speed
level: are 1080 AND 1920 Hz standing over the channel comb. It grades nothing
here; the occupied verdict and its thresholds are untouched, and what the knee
buys is a line that says which side of it the window fell on.

Run:  python -m pytest hfmodem/tests/shrike/test_answer_instruments_0915.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_entry_answer import _Session

FS = onair.FS
FIXTURE = Path(__file__).with_name("fixtures") / "kb5lzk-answer-0915"

TAPE_MS = 902.4
"""Where the tape puts the peer's answer on every clean cycle of that arm."""

ONSET_MS = 1015.3
"""...and what the onset detector made of the first of them. The instrument the
grid had: 113 ms off the word, and it is the number the fallback prints."""

TRACKED = (22, 23, 35, 36)
SILENT = 28
"""The arm's third quiet hold, which carries no word on either carrier."""

D_MAX_N = onair._d_max_n(1.25, 0.030)
P1_PACKET_N = round(spec.P1_PACKET_S * FS)
P1_CS_N = round(spec.P1_CS_S * FS)


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
        assert fs == FS and len(pcm) == row["end"] - row["start"]
        out[row["hold"]] = (row, pcm.astype(float) / 32768)
    return meta, out


def _grid(meta, row, pcm, *, wired: bool) -> onair._MasterGrid:
    """One cycle of the arm's own geometry, with the hold read into it.

    The entry is keyed for slot 8, the peer's answer lands in that slot at the
    tape's position, and the onset handed to the grid is the one the arm's
    detector reported -- so the two instruments disagree by 113 ms and the line
    says which of them it is quoting.
    """
    session = _Session(role=arq.ISS)
    session.rx.p3_receive_offset_hz = meta["correction_hz"]
    session.rx._p3_answer_at = row["previous_answer"]
    session.rx.new_cycle()
    due = row["previous_answer"] + meta["cycle"]
    session.rx.control_signal(pcm, row["start"], due)

    slot_n = meta["cycle"]
    boundary = due - round(TAPE_MS / 1e3 * FS)
    grid = onair._MasterGrid(boundary - 8 * slot_n, slot_n, 0,
                             packet_n=P1_PACKET_N, cs_n=P1_CS_N,
                             d_max_n=round(0.130 * FS))
    grid.d_n, grid.d_ref_n = float(round(0.0924 * FS)), P1_PACKET_N
    grid.corroborated = True
    for k in (5, 6, 7):
        grid.update([grid.rx_due(k)])
    grid.keyed_slot = 8
    grid.keying(onair.Protocol.PACTOR3)
    if wired:
        grid.sessrx = session.rx
    grid.update([boundary + round(ONSET_MS / 1e3 * FS)])
    return grid


@pytest.mark.parametrize("hold", TRACKED)
def test_the_position_is_the_codeword_instant_where_there_is_one(windows, hold):
    """The number in the line is the matched filter's, and it says so."""
    meta, rows = windows
    grid = _grid(meta, *rows[hold], wired=True)
    tracked = grid.entry_tracked[grid.cycles]
    assert f"{tracked:.1f} (tracked)" in grid.entry_verdict()
    assert f"{ONSET_MS:.1f}" not in grid.entry_verdict()


@pytest.mark.parametrize("hold", (22, 35, 36))
def test_the_clean_cycles_land_on_the_tape(windows, hold):
    """Three of the four inside 2 ms of where the tape puts every answer.

    The fourth, hold 23, is the cycle whose channel 5 faded: the surviving
    carrier still carries the word and places it a symbol early. It is in the
    series above and not in this one.
    """
    meta, rows = windows
    grid = _grid(meta, *rows[hold], wired=True)
    assert abs(grid.entry_tracked[grid.cycles] - TAPE_MS) <= 2


def test_a_cycle_with_no_word_in_it_reports_the_onset(windows):
    """The negative control, and the reason the fold is checked.

    Hold 28's last tracked answer is a whole cycle back. Folding it would let
    that answer stand in for this cycle's, which is a quiet cycle reporting a
    position nothing measured in it; the position falls back to the onset and
    is not marked.
    """
    meta, rows = windows
    grid = _grid(meta, *rows[SILENT], wired=True)
    assert not grid.entry_tracked
    assert f"{ONSET_MS:.1f} ms after our slot boundary" in grid.entry_verdict()
    assert "(tracked)" not in grid.entry_verdict()


@pytest.mark.parametrize("hold", TRACKED)
def test_without_the_receiver_the_line_is_the_onset_it_always_was(windows, hold):
    """What the arm printed: the same cycles, the envelope's number, unmarked."""
    meta, rows = windows
    grid = _grid(meta, *rows[hold], wired=False)
    assert not grid.entry_tracked
    assert f"{ONSET_MS:.1f} ms after our slot boundary" in grid.entry_verdict()
    assert "(tracked)" not in grid.entry_verdict()


def _sighted(seg: np.ndarray, rng):
    """One window through the band, after enough quiet ones to have a floor."""
    band = onair._AnswerBand()
    for _ in range(onair.ANSWER_FLOOR_WINDOWS + 2):
        band.sight(rng.normal(0, .002, seg.size), 0, D_MAX_N,
                   read=False, answered=True)
    reading = band.sight(seg, 0, D_MAX_N, read=False, answered=True)
    assert reading is not None
    return reading


def _rise(freqs, n: int, rng) -> np.ndarray:
    """A steady rise on the named tones, up before the window opens."""
    t = np.arange(n) / FS
    out = rng.normal(0, .002, n)
    for hz in freqs:
        out += .05 * np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi))
    return out


def test_a_rise_off_the_pactor3_pair_is_named_off_carrier():
    """Two-thirds of a kHz of energy, none of it where PACTOR-III puts any."""
    rng = np.random.default_rng(20260916)
    line = _sighted(_rise((2200, 2280, 2340, 2400), round(0.34 * FS), rng),
                    rng).line
    assert "off-carrier" in line and "on-carrier" not in line


def test_a_rise_on_the_pactor3_pair_is_named_on_carrier():
    """The same level, the same width, on the two channels every level lights."""
    rng = np.random.default_rng(20260916)
    line = _sighted(_rise((1080, 1920), round(0.34 * FS), rng), rng).line
    assert "on-carrier" in line


@pytest.mark.parametrize("hold,word", ((35, "on-carrier"), (23, "off-carrier")))
def test_the_tape_names_the_slot_it_filled(windows, hold, word):
    """And on the arm's own holds, which is what the flag was wanted for.

    Hold 35 is a PACTOR-III answer the reader took; hold 23 is one it dropped,
    with a faded carrier, and the comb reads it at 3.1 dB. The flag is not a
    decoder and does not claim to be one -- what it separates is an emission on
    PACTOR-III's channels from a slot that merely got louder.
    """
    _, rows = windows
    assert word in _sighted(rows[hold][1], np.random.default_rng(20260916)).line


def test_the_flag_moves_no_verdict(windows):
    """The occupied call is the call it was: hold 35 occupied, 23 a reading."""
    _, rows = windows
    got = {hold: _sighted(rows[hold][1], np.random.default_rng(20260916))
           for hold in (23, 35)}
    assert got[35].occupied and not got[23].occupied
    assert got[23].line.startswith("ANSWER SLOT LEVEL")
    assert got[35].line.startswith("ANSWER SLOT OCCUPIED")
