# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The in-link grid read was never on the air.

`_anchored_answer` and `INLINK_READ_BAND` were written on 2026-09-16 for a peer
answering under the onset detector's threshold, and `test_inlink_anchored_read_0916`
reads the KB5LZK tape through them by hand and finds the eleven codewords that
call threw away. The production wiring went into the SETUP loop, under a guard
that reads `State.CONNECTED`. The setup loop leaves for the hold loop in the same
iteration the link comes up, so that guard is true for at most one cycle and the
hold loop -- the only loop that runs while a link is held -- never asks at all.
Across the 123 arm logs of `working/pactor-level-air-0916` the line the read
prints, `read at the grid from d =`, appears zero times.

K0NTS, 40 m, 2026-09-17 (`pactor-current-k0nts-40-sense-20260917T140438Z`) is the
arm with the arithmetic on it. The gateway answered CS4 at d = 64-66 ms after our
packet ended while the grid's nominal answer slot stood at 105 ms, so
`cs_anchored`'s three half-bits either side never covered it, and the bursts read
1.9-5.3x against the 4.0x `_p1_runs` wants before a burst reaches a decoder at
all. The session read 14 of its 23 keyed hold windows, spent the live-link retry
budget on the other nine, and signed off with `max retries (nothing the peer sent
asked for the channel...)` over a station that had asked in every one of them.
The grid read takes 13 of the 23, three of them cycles nothing else read.

Over every arm in that campaign holding in-link windows -- 651 keyed windows in
30 sessions -- the session read 415 and the grid read adds 58 more across 15
arms, three of which read NOTHING live and would have read 11, 13 and 7. Two of
the 58 are changeovers, which this reader reports and does not act on, so 56
reach the FSM.

So this drives the whole session, `shrike.onair --replay` against a station on
K0NTS's geometry: 65 ms turnaround, every answer under the detector, a codeword
at zero bit errors in all of them. What the loop does with that is the test.

Run: python -m pytest hfmodem/tests/shrike/test_inlink_grid_read_0917.py
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1, rxfront, spec

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
PACKET_N = round(spec.P1_PACKET_S * FS)

PEER_D_S = 0.065
"""K0NTS's turnaround on 40 m: `[grid] TURNAROUND ACQUIRED ... d = 65.1 ms after
our data ends`, twice in that arm, and 64-66 ms on every burst the detector saw.
Inside `INLINK_READ_BAND` and 40 ms clear of the nominal answer slot, which is
what puts it out of the anchored reader's reach."""

PEER_X = 2.0
"""What the answers stand at over their own window, in the middle of that arm's
1.9-5.3x and under the 4.0x the onset detector wants."""

NOISE = 0.02
MESSAGE = "TEST DE W9SSJ -- " + "THE QUICK BROWN FOX 0123456789 " * 8
HOLD = 14


def _gain(level: float) -> float:
    """The amplitude that puts a codeword at `level` over its own window.

    Searched rather than asserted: the gate is a ratio against the window's own
    passband median, so what an amplitude reads as depends on the noise under it.
    """
    rng = np.random.default_rng(3)
    cs = onair._trim_silence(
        np.asarray(pactor1.control_signal(pactor1.CS_SPEED), np.float32))
    at = round(PEER_D_S * FS)
    for gain in np.linspace(0.005, 2.0, 400):
        seg = rng.normal(0, NOISE, round(0.215 * FS)).astype(np.float32)
        seg[at:at + cs.size] += cs * gain
        if onair._candidate_excess(seg, at / FS) >= level:
            return float(gain)
    raise AssertionError(f"no gain reached {level}x")


def _peer_wav(path: Path, *, slots: int) -> None:
    """A PACTOR-1 gateway answering every cycle, too quietly to be detected.

    CS4 takes the call at 100 Bd and the acknowledgements alternate behind it,
    one per cycle, on the caller's own grid -- `test_grantslot`'s station moved
    to K0NTS's turnaround and K0NTS's level.
    """
    gain = _gain(PEER_X)
    rng = np.random.default_rng(7)
    out = rng.normal(0, NOISE, (slots + 2) * SLOT_N).astype(np.float32)
    for k in range(slots):
        cs = (pactor1.CS_SPEED if k == 0 else
              pactor1.CS_ACK_A if k % 2 else pactor1.CS_ACK_B)
        rf = onair._trim_silence(
            np.asarray(pactor1.control_signal(cs, invert=k % 2), np.float32))
        at = k * SLOT_N + PACKET_N + round(PEER_D_S * FS)
        out[at:at + rf.size] += rf * gain
    onair.session.write_wav(str(path), out)


@pytest.fixture(scope="module")
def log(tmp_path_factory) -> str:
    """One whole session over that station; its stdout is the evidence."""
    tmp = tmp_path_factory.mktemp("inlink-grid")
    wav = tmp / "peer.wav"
    _peer_wav(wav, slots=44)
    # The per-cycle capture write runs on its own thread and outlives the
    # session; nothing here reads the files, and leaving it on races the
    # temporary directory away from under it.
    save = onair._save_capture_async
    onair._save_capture_async = lambda *a, **kw: None
    argv, out = sys.argv, io.StringIO()
    sys.argv = ["shrike.onair", "--replay", str(wav), "--hold", str(HOLD),
                "--max-cycles", "3", "--mycall", "W9SSJ", "--dxcall", "K7ABC",
                "--dial", "7100000", "--outdir", str(tmp / "out"),
                "--message", MESSAGE, "--pactor1-only"]
    try:
        with contextlib.redirect_stdout(out):
            onair.main()
    finally:
        onair._save_capture_async = save
        sys.argv = argv
    return out.getvalue()


def test_the_station_is_the_one_the_detector_refuses():
    """The premise, before anything is claimed about the loop.

    A test that fed the reader an audible burst would pass on the wrong path and
    prove nothing about the grid: what is under examination is the window nothing
    else reads.
    """
    at = round(PEER_D_S * FS)
    rng = np.random.default_rng(11)
    seg = rng.normal(0, NOISE, round(0.215 * FS)).astype(np.float32)
    cs = onair._trim_silence(
        np.asarray(pactor1.control_signal(pactor1.CS_ACK_A), np.float32))
    seg[at:at + cs.size] += cs * _gain(PEER_X)
    assert not onair._peer_bursts(seg, 0), \
        f"the onset detector offers this burst after all: {rxfront.cs_evidence(seg)}"
    assert onair._candidate_excess(seg, PEER_D_S) >= onair.ANSWER_CODEWORD_X
    read = onair._anchored_answer(seg, 0)
    assert read is not None and read[0][1] == 0, \
        "the grid read cannot take this window either, so the loop is not what is on trial"


def test_a_changeover_read_at_the_grid_is_reported_and_not_acted_on():
    """A reversal owes the peer's packet, and twelve bits have not got it.

    The head read rotates the grid AND collects the 840 ms behind the word, which
    is the whole of what a changeover carries. `test_breakin`'s negative control
    -- the same loop with that read disabled -- stands on nothing else finding
    the head, and a bare codeword that reversed the grid there would yield to a
    station whose packet was then never read.
    """
    at = round(PEER_D_S * FS)
    rng = np.random.default_rng(13)
    seg = rng.normal(0, NOISE, round(0.215 * FS)).astype(np.float32)
    cs = onair._trim_silence(
        np.asarray(pactor1.control_signal(pactor1.CS_CHANGEOVER), np.float32))
    seg[at:at + cs.size] += cs * _gain(PEER_X)
    assert onair._anchored_answer(seg, 0) is not None, "the read found nothing"
    taken = []
    line = onair._grid_answer(SimpleNamespace(_on=taken.append), seg, 0, 0)
    assert line and "NOT acted on" in line, line
    assert not taken, "the changeover reached the front end after all"


def test_the_held_link_reads_the_answer_at_the_grid(log):
    """The whole of it: a linked cycle nothing decoded asks the grid.

    `RX ... read at the grid from d = ...` is the line, and it had never yet
    appeared in an arm log. The offset it carries is WHERE THE READ STARTED and
    not where the peer keyed -- `decode_control_signal` locks the word inside the
    slice it is handed -- so the band is what it is checked against.
    """
    read = [int(d) for d in
            re.findall(r"RX \S+ read at the grid from d = (-?\d+) ms", log)]
    assert read, ("the hold loop never asked the grid: "
                  f"{log.count('NO CONTROL SIGNAL')} cycles of "
                  "'NO CONTROL SIGNAL' over a peer answering every one of them")
    assert len(read) >= HOLD - 4, f"{len(read)} of {HOLD} held cycles read: {read}"
    lo, hi = (b * 1e3 for b in onair.INLINK_READ_BAND)
    assert all(lo <= d <= hi for d in read), read


def test_a_gateway_answering_every_cycle_keeps_its_link(log):
    """The consequence, which is the reason the read exists.

    `_inflight.retries` reaching `--link-retries` is what sends the QRT, and
    silence is what spends it. A codeword in the answer slot is a cycle the
    budget may not charge, whichever reader found it.
    """
    assert "max retries" not in log, \
        "the link still tore down over a station answering every cycle"
    assert "** LINK DOWN **" not in log, "the hold did not run its course"
    assert "the ARQ was given" in log, \
        "the words were printed and never handed to the FSM, which is the half " \
        "that costs the budget"
