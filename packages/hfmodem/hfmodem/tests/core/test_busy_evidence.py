# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 2026-08-10 recordings that caught the occupancy guard firing on nothing.

Three 30 s captures through the same receiver and antenna, the tuned frequency
verified by readback before each: a control at 6950 kHz — outside the amateur
band, nothing there — and two 40 m channels each carrying another station's
VARA session. Against the old ``BURST_DB = 5.5`` the control read busy on every
10 s window, margins +0.5 to +0.9, from ``burst`` alone: that night's static
holds its lifts past the 0.5 s hold that separated crashes from overs on the
2026-08-09 recordings. A guard that fires on an empty out-of-band frequency
fires everywhere, and a guard that fires everywhere gets ``--force``d — which
is how this station transmitted over two live sessions on 2026-08-09.

Measured on these captures, ``_BURST_HOLD_S = 0.5``, windows as labeled:

    control 6950   burst 4.18/5.93/4.43 (8 s)  6.38/5.98/6.32 (10 s)  6.87 (30 s)
                   shape <= 2.98              tone <= 1.18
    7101.5 kHz     burst >= 5.81 (8 s)         shape >= 9.08   tone <= 5.36
    7106.5 kHz     burst >= 8.65 (10 s)        shape >= 17.24  tone <= 4.83

Both sides are asserted here over the 8 s and 10 s windows the tools actually
listen in (onair.sh and kestrel_connect take 8 s, gwsurvey and vara_monitor
10 s). Over the whole 30 s the control's burst accumulates 6.87 against the 7.86
of the weakest occupant only ``burst`` catches, and 80 m band noise through the
same receiver reaches 9.98 over 8 s — so ``burst`` decides nothing
(``core.busy.DECIDING``) and the figures here are what a refusal is re-read by.

The captures live outside the tree like every other recording (working/ is the
operator's bench, the corpus keeps the archive); these tests skip visibly when
they are absent, the same bargain tests/kestrel/corpora.py strikes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.core import busy
from hfmodem.core.occupied import OCCUPIED_HZ
from hfmodem.tests.kestrel import corpora

REPO = Path(__file__).resolve().parents[5]
_NAME = "detector-evidence-20260811"
#: The archive first, the bench second. A threshold whose evidence lives only in
#: one working directory is a threshold nobody else can re-derive, so these
#: recordings are kept in the corpus alongside every other one; `working/` is
#: still searched because that is where a capture lands the night it is made.
_CORPUS = evidence.CORPUS
EVIDENCE = next((d for d in (_CORPUS / _NAME, REPO / "working" / _NAME)
                 if (d / "control-6950-outofband.wav").exists()),
                _CORPUS / _NAME)
CONTROL = EVIDENCE / "control-6950-outofband.wav"
OCCUPIED = (EVIDENCE / "channel-7101k5.wav", EVIDENCE / "channel-7106k5.wav")

requires_evidence = pytest.mark.skipif(
    not all(p.exists() for p in (CONTROL, *OCCUPIED)),
    reason=f"the 2026-08-10 detector evidence captures not present "
           f"({_CORPUS / _NAME}, nor {REPO / 'working' / _NAME})")

FS = 48000
#: the listen windows production uses: onair.sh/kestrel_connect 8 s, gwsurvey 10 s
WINDOWS_S = (8, 10)


def _wav(path: Path) -> np.ndarray:
    from scipy.io import wavfile

    x = np.asarray(wavfile.read(str(path))[1], float)
    return x / 32768.0 if np.abs(x).max() > 1.5 else x


def _windows(x: np.ndarray, secs: int) -> list[np.ndarray]:
    n = secs * FS
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)]


@requires_evidence
def test_the_out_of_band_control_reads_clear():
    """6950 kHz with nothing on it, and the defect of 2026-08-10: every 10 s
    window read busy on ``burst`` alone. Static that holds a lift for half a
    second is still static; the decision has to sit above it."""
    x = _wav(CONTROL)
    for secs in WINDOWS_S:
        for i, w in enumerate(_windows(x, secs)):
            s = busy.scores(w)
            assert s is not None, f"control window {i} ({secs} s) was not judged live"
            verdict, _levels, margin = busy.is_busy(w)
            assert not verdict, (
                f"the out-of-band control read busy on window {i} ({secs} s) at "
                f"margin {margin:+.2f} dB (burst {s['burst']:.2f}, "
                f"shape {s['shape']:.2f}, tone {s['tone']:.2f})")


@requires_evidence
@pytest.mark.parametrize("path", OCCUPIED, ids=lambda p: p.stem)
def test_the_channels_carrying_vara_sessions_read_busy(path):
    """The other side of the same night: 7101.5 kHz (a session a KiwiSDR decoded
    14 VARA frames on minutes earlier) and 7106.5 kHz. Raising the burst
    threshold must not trade these away — every occupied window here carries
    shape at 9 dB or more, which is what the verdict rides on when a weak over
    drops burst under its threshold."""
    x = _wav(path)
    for secs in WINDOWS_S:
        for i, w in enumerate(_windows(x, secs)):
            s = busy.scores(w)
            assert s is not None, f"{path.name} window {i} ({secs} s) was not judged live"
            verdict, _levels, margin = busy.is_busy(w)
            assert verdict, (
                f"{path.name} read clear on window {i} ({secs} s) at margin "
                f"{margin:+.2f} dB (burst {s['burst']:.2f}, shape {s['shape']:.2f}, "
                f"tone {s['tone']:.2f})")


# ------------------------------------------------- the band the modem will fill
#: Every band a keying path can ask for, plus the answer an unlisted mode gets.
BANDS = {f"{m} {b}".strip(): v for (m, b), v in OCCUPIED_HZ.items()} | {
    "unlisted": busy.FULL_BAND}


def _pretx(path: Path) -> np.ndarray:
    return _wav(path)[:int(corpora.SENSE_PRETX_S * FS)]


@requires_evidence
@pytest.mark.parametrize("band", BANDS.values(), ids=BANDS)
def test_no_band_a_modem_can_ask_for_makes_the_control_busy(band):
    """The trap in narrowing the sense: the thresholds are properties of the band
    they were measured on, and the noise reference moves with it. Measured on the
    6950 kHz control and the 7108.5 kHz channel the operator listened to and heard
    nothing on, worst over 5, 8 and 10 s windows:

        band Hz    2300   1617   1000    609    400
        burst      6.72   7.18   7.81   8.64   9.82
        shape      3.43   3.73   4.25   6.22   6.74
        tone       1.32   1.12   1.09   1.09   1.09

    ``burst`` is why it is not simply a matter of parameterising the band: at
    400 Hz an empty out-of-band frequency reaches 9.82 dB of it, past every
    occupant in the corpus, so no threshold there separates anything. It stays on
    the full passband, where the only occupant it can see lives anyway. ``shape``
    is floored at ``MIN_SHAPE_HZ``; only ``tone``, which reads one bin against its
    own neighbourhood, narrows the whole way and is flat while it does.
    """
    x = _wav(CONTROL)
    quiet = [(CONTROL.name, w) for secs in (int(busy.POUNCE_WINDOW_S), *WINDOWS_S)
             for w in _windows(x, secs)]
    if all(p.exists() for p in corpora.SENSE_CLEAR):
        quiet += [(p.name, _pretx(p)) for p in corpora.SENSE_CLEAR]
    for name, w in quiet:
        s = busy.scores(w, FS, band)
        assert s is not None, f"{name} was not judged live at {band}"
        verdict, _levels, margin = busy.is_busy(w, FS, band)
        assert not verdict, (
            f"{name} read busy sensing {band[0]:.0f}-{band[1]:.0f} Hz at margin "
            f"{margin:+.2f} dB (burst {s['burst']:.2f}, shape {s['shape']:.2f}, "
            f"tone {s['tone']:.2f})")


@requires_evidence
@pytest.mark.parametrize("path", OCCUPIED, ids=lambda p: p.stem)
@pytest.mark.parametrize("band", BANDS.values(), ids=BANDS)
def test_a_session_on_this_channel_reads_busy_whatever_we_would_key(path, band):
    """The other direction, and the one narrowing could quietly cost: a station
    working the channel we are pointed at must stay audible at every band, because
    every one of them overlaps a 2300 Hz occupant. Worst margin over both
    recordings falls from +3.08 dB sensing the whole passband to +2.78 dB sensing
    ARDOP 200's 250 Hz -- narrower, not blind."""
    x = _wav(path)
    for secs in WINDOWS_S:
        for i, w in enumerate(_windows(x, secs)):
            verdict, _levels, margin = busy.is_busy(w, FS, band)
            assert verdict, (
                f"{path.name} window {i} ({secs} s) read clear sensing "
                f"{band[0]:.0f}-{band[1]:.0f} Hz at margin {margin:+.2f} dB")


@corpora.requires_channel_sense
def test_whether_an_occupant_is_ours_to_wait_for_is_the_band_we_would_key():
    """7102.0 kHz on 2026-08-09, whose occupant sits around 2600 Hz of audio and a
    channel away in RF. Sensing the whole passband it reads busy at +3.89 dB, all
    of it ``tone`` at 8.89; sensing the 1375-1625 Hz an ARDOP 200 frame fills,
    -1.86 dB and clear. That is the reading the operator kept overruling.

    A PACTOR key is on the other side of it, and that is the correction: the table
    answered 1200-1800 for PACTOR until 2026-08-16 and read this channel clear,
    while the session it was about to start reaches 2620 Hz -- straight over this
    station. Both directions are asserted, because a sense that clears everything
    and a sense that clears nothing fail the same way."""
    x = _pretx(corpora.SENSE_BUSY[1])
    assert busy.is_busy(x)[0], (
        "the narrow occupant on 7102.0 no longer reads busy on the whole "
        "passband, so this recording has stopped being the case it is kept for")
    verdict, _levels, margin = busy.is_busy(x, FS, OCCUPIED_HZ[("ardop", "200")])
    assert not verdict, (
        f"an occupant a kilohertz clear of our tones still held the gate shut "
        f"({margin:+.2f} dB)")
    verdict, _levels, margin = busy.is_busy(x, FS, OCCUPIED_HZ[("pactor", "")])
    assert verdict, (
        f"a PACTOR session reaching 2620 Hz read this channel clear "
        f"({margin:+.2f} dB), which is the 2026-08-16 defect back again")


@corpora.requires_gateway_session
def test_the_window_is_the_shortest_that_will_not_let_a_caller_through():
    """``WINDOW_S`` is pinned from the occupied side, where the cost is somebody
    else's session and not a wasted slot.

    The unattended sense keys on two consecutive clear windows. Over a whole live
    gateway session on 7103.5 kHz, 8 s windows never give it two: 12 of 13 refuse
    and the thirteenth stands alone. Cut the window to 5 s and the same recording
    opens a run of two — 17 of 21 refuse and four of the misses fall together in
    the turnarounds, which is a station calling into a held link.

    A pounce keys on three consecutive clear ``POUNCE_WINDOW_S`` windows. At 3 s
    the same session refuses 29 of 35 and the longest run of clears is two, so a
    pounce never keys into it either; and ``MIN_LIVE_S`` is pinned from the same
    side, because at 2.0 the run reaches three.
    """
    x = _wav(corpora.OFFAIR / "KC9GHZ_2300" / "rig_rx.wav")

    def longest_clear(secs):
        best = run = 0
        for w in _windows(x, secs):
            run = 0 if busy.is_busy(w)[0] else run + 1
            best = max(best, run)
        return best

    assert longest_clear(5) >= 2, (
        "5 s windows no longer let a caller through this session, so nothing pins "
        "WINDOW_S and the launchers could go back to polling in them")
    assert longest_clear(int(busy.WINDOW_S)) < 2
    assert longest_clear(int(busy.POUNCE_WINDOW_S)) < 3, (
        "a pounce's three clear windows fit inside this session's turnarounds")
    assert busy.POUNCE_WINDOW_S >= busy.MIN_LIVE_S, (
        "a pounce window shorter than the live-audio floor can never read clear")


@requires_evidence
def test_the_shape_floor_is_what_keeps_the_empty_channel_empty(monkeypatch):
    """``MIN_SHAPE_HZ`` is not a rounding: at 609 Hz the control's ``shape``
    reaches 6.22 dB against a threshold of 6.0, and at 400 Hz 6.74. Take the floor
    away and the narrowest band in the table reads busy on an empty frequency,
    which is how ``--force`` became reflex in the first place."""
    band = OCCUPIED_HZ[("ardop", "200")]
    x = _wav(CONTROL)
    monkeypatch.setattr(busy, "MIN_SHAPE_HZ", band[1] - band[0])
    assert any(busy.scores(w, FS, band)["shape"] > busy.SHAPE_DB
               for secs in WINDOWS_S for w in _windows(x, secs)), (
        "an unfloored 200 Hz shape band no longer reads this control busy, so the "
        "floor is carrying nothing and the population it was measured on has moved")
