# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The card timebase. Exactness is the whole property being asserted."""
from __future__ import annotations

import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core.rates import (CARD_RATE_HZ, clock_report, host_report,
                                lost_step, rate_fit, to_native)

PACKAGE = Path(__file__).resolve().parents[2]
TOOLS = PACKAGE.parents[2] / "tools"


def test_the_card_index_maps_exactly_onto_a_native_index():
    assert to_native(CARD_RATE_HZ, 12000) == 12000
    assert to_native(4800, 12000) == 1200
    assert to_native(0, 12000) == 0


@pytest.mark.parametrize("rate", [8000, 12000, 16000, 48000])
def test_a_whole_second_of_card_is_a_whole_second_of_the_lane(rate):
    assert to_native(CARD_RATE_HZ * 7, rate) == rate * 7


def test_the_cards_own_rate_is_the_identity():
    assert to_native(123_457, CARD_RATE_HZ) == 123_457


def test_a_card_index_between_native_samples_truncates():
    """Integer division, not rounding: the index names a sample that has already
    arrived. Rounding up would report a frame position the lane cannot yet have
    decoded."""
    assert to_native(5, 12000) == 1               # 1.25 native samples
    assert to_native(7, 12000) == 1


def test_a_ten_hour_session_stays_exact():
    """The reason this is integer arithmetic. Nothing accumulates — a position is
    always converted from the card count — so a long session is as exact as a
    short one, which a float scale factor would not be."""
    n = 48000 * 3600 * 10
    assert to_native(n, 12000) == 12000 * 3600 * 10


# --- the estimator the capture paths share -----------------------------------
#
# Two capture front ends read the same card: `core.audio.StationAudio` and
# shrike's `_LiveInput`. The lost-block detector and the rate fit were written
# once and copied, and then each copy was found wrong in a DIFFERENT term and
# corrected on its own -- `core` had the anchor right and reported microseconds
# of residual scatter as a ppm; `onair` had the standard error right and
# anchored on a term that was really callback lateness. Neither correction
# reached the other, and they still answered differently on identical data.


def _points(seconds: float, ppm: float = 2.4, fs: int = CARD_RATE_HZ,
            noise: float = 0.0, seed: int = 7):
    """One (ADC instant, sample index) pair a second off a card `ppm` fast."""
    rng = np.random.default_rng(seed)
    return [(n / fs / (1 + ppm * 1e-6) + rng.normal(0.0, noise), n)
            for n in range(0, int(seconds * fs) + 1, fs)]


def test_one_estimator_and_not_two():
    """The gate the divergence earned.

    A second copy of either of these is how the last two corrections came to
    apply to one capture path and not the other."""
    paths = (PACKAGE / "core" / "audio.py",
             PACKAGE / "shrike" / "onair.py",
             TOOLS / "kestrel_connect.py")
    if absent := [p for p in paths if not p.exists()]:
        pytest.skip(f"{absent[0]} is not in this tree: the distribution ships no "
                    "tools/, and kestrel's capture path lives there")
    offences = [
        f"{path.name}: {mark}"
        for path in paths
        for mark in ("lstsq", "np.linalg.inv", "blocksize / 2 /")
        if mark in path.read_text(encoding="utf-8")
    ]
    assert not offences, (
        "the clock fit and the lost-block detector live in core.rates, and every "
        "capture path calls them:\n  " + "\n  ".join(offences))


def test_a_short_baseline_answers_and_says_how_well():
    """One rule, and it is the sigma rather than a threshold.

    shrike refused to fit under 30 s of baseline and `core` did not, so
    `preflight` -- whose stream is a dozen seconds by the time the fit runs --
    printed a confident ppm off a baseline the other copy called unmeasurable.
    Refusing is not the fix: preflight's question is whether the card is running
    at 48000 at all, a device handing back 44100 reads -81000 ppm, and a line
    nothing could measure fails the whole run. The slope's standard error already
    carries the baseline, so it is what both of them report.
    """
    runs = {secs: [rate_fit(_points(secs, noise=200e-6, seed=s), CARD_RATE_HZ)
                   for s in range(40)]
            for secs in (5.0, 60.0)}
    for secs, fits in runs.items():
        assert all(np.isfinite(f).all() for f in fits), secs
        covered = sum(abs(ppm - 2.4) < sigma for ppm, sigma in fits)
        assert covered >= 24, (
            f"{secs} s: the error bar covered the planted crystal {covered}/40 "
            "times, which is not what one sigma means")
    short, long = (float(np.median([s for _, s in runs[secs]]))
                   for secs in (5.0, 60.0))
    assert short > 5 * long, (
        f"a 5 s baseline must not report a 60 s baseline's confidence: "
        f"sigma {short:.1f} against {long:.1f}")


def test_a_fit_with_no_residual_degrees_of_freedom_declines():
    assert all(np.isnan(rate_fit(_points(0.0), CARD_RATE_HZ)))
    assert all(np.isnan(rate_fit(_points(1.0), CARD_RATE_HZ)))


def test_the_fit_reads_the_crystal_and_not_the_startup():
    """Differencing the endpoints charges the ~0.1 s of stream startup during
    which the clock runs and no samples arrive to the crystal, which fabricates
    about -1700 ppm. A fixed offset belongs in the intercept."""
    pts = [(t + 0.1, n) for t, n in _points(60.0)]
    ppm, _ = rate_fit(pts, CARD_RATE_HZ)
    assert abs(ppm - 2.4) < 0.01, f"{ppm=:+.2f}"


def test_a_dropped_block_is_the_only_trace_a_starved_interpreter_leaves():
    """Eight pure-Python threads cost this station 87% of a 90 s capture with
    `input_overflow` false on every callback that ran."""
    fs, blk = CARD_RATE_HZ, 128
    assert lost_step(blk / fs, 0.0, blk, fs) == blk
    assert lost_step(3 * blk / fs, 0.0, blk, fs) == 3 * blk


def test_drift_can_never_accumulate_into_the_lost_count():
    """Half a block, not a block: a card 100 ppm out gains 0.0000128 s per 128
    samples and would otherwise be counted as loss the moment the sum crossed
    one."""
    fs, blk = CARD_RATE_HZ, 128
    assert lost_step(0.4 * blk / fs, 0.0, blk, fs) == 0
    assert lost_step(0.6 * blk / fs, 0.0, blk, fs) == round(0.6 * blk)


def test_the_first_callback_has_nothing_to_compare_against():
    assert lost_step(0.5, None, 128, CARD_RATE_HZ) == 0


def test_a_clean_line_says_only_what_it_can_see():
    """`lost` is anchored on the converter's own timestamps, so loss upstream of
    them is invisible to it. Only a second receiver could see such a hole, and the
    one witnessed arm does not show one: +260 ms against a counter justifying 295,
    once the aligner's 8300 ppm framing error is corrected. The bound rests on
    where the count is anchored, not on that arm."""
    clean = clock_report(samples=60 * CARD_RATE_HZ, lost=0, xruns=0, underruns=0,
                         points=_points(60.0), blocksize=128, fs=CARD_RATE_HZ)
    assert "capture clock:" in clean
    assert "no loss we can see" in clean
    assert "SPLICED" not in clean
    spliced = clock_report(samples=60 * CARD_RATE_HZ, lost=14924, xruns=0,
                           underruns=0, points=_points(60.0), blocksize=128,
                           fs=CARD_RATE_HZ)
    assert "SPLICED" in spliced and "311 ms" in spliced


@pytest.mark.parametrize("seconds", [5.0, 20.0, 29.9, 60.0])
def test_the_two_capture_paths_answer_alike_on_identical_data(seconds):
    """The divergence, as it was found.

        dur=    5 n=  5  core=-4860.42  onair=nan
        dur=   20 n= 20  core=-4859.82  onair=nan
        dur= 29.9 n= 30  core=-4859.99  onair=nan
        dur=   60 n= 60  core=-4860.00  onair=-4860.00

    One copy carried a 30 s baseline rule and the other did not, so the same
    stream was unmeasurable to shrike and a confident figure to `preflight` --
    whose own stream is a dozen seconds by the time it asks.
    """
    from hfmodem.core.audio import StationAudio                # noqa: PLC0415
    from hfmodem.shrike.onair import _LiveInput                # noqa: PLC0415

    pts = _points(seconds, ppm=-4860.0)
    core = StationAudio.__new__(StationAudio)
    core._fit = pts
    shrike = _LiveInput.__new__(_LiveInput)
    shrike._fit, shrike.fs = pts, CARD_RATE_HZ

    assert core.clock_ppm()[0] == shrike.clock_ppm()
    assert abs(shrike.clock_ppm() + 4860.0) < 2.0, shrike.clock_ppm()


def test_the_count_is_one_way_and_says_what_that_costs():
    """Lost capture does not come back, so a down-step is not a credit.

    Which makes this a ratchet, and the ratchet is only harmless where `ahead`
    cannot swing: asking the built-in 44100 device for 48000 puts CoreAudio's
    rate converter in the path, sawtooths the timestamps +/-125 samples and banks
    2082 ms over 30 s of a stream `rate_fit` reads at -0.73 ppm. The station's
    codec is opened at the rate it runs at and has nothing to swing.
    """
    fs, blk = CARD_RATE_HZ, 128
    assert lost_step(0.0, 125 / fs, blk, fs) == 0, "a down-step is not a credit"
    saw, prev, banked = [0.0, 125 / fs] * 3 + [0.0], None, 0
    for a in saw:
        banked += lost_step(a, prev, blk, fs)
        prev = a
    assert banked == 375, (
        f"the sawtooth nets to zero and this banks {banked} -- true of the "
        "counter, and the reason a resampled stream must not be read off it")


# --- the host the reading was taken on ---------------------------------------
#
# The seven-day loss ramp -- daily medians 0.00 to 7.15 ms of air lost per second
# of stream, monotone across one boot session -- was reconstructed from file
# mtimes and `last reboot`, because no session said what machine it ran on.

HOST_LINE = re.compile(r"host: (\d+) s since boot, (\d+) threads in this process")


def test_the_host_line_says_the_two_things_the_ramp_could_not_separate():
    m = HOST_LINE.fullmatch(host_report())
    assert m, host_report()
    assert int(m.group(1)) > 0
    assert int(m.group(2)) == threading.active_count()


def test_the_uptime_is_the_machines_and_not_the_interpreters():
    """A clock that restarted with the process would read a few seconds on every
    arm ever run, which is a column of noise wearing the name of the variable."""
    child = subprocess.run(
        [sys.executable, "-c",
         "from hfmodem.core.rates import host_report; print(host_report())"],
        capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(PACKAGE.parent), "PATH": "/usr/bin:/bin"})
    fresh = int(HOST_LINE.search(child.stdout).group(1))
    assert abs(fresh - int(HOST_LINE.search(host_report()).group(1))) <= 5, (
        f"a process seconds old reports {fresh} s since boot -- that is its own "
        "age, not the machine's")


def test_the_thread_count_is_this_processs_own():
    """What starves the callback is the GIL, so the count that can matter is the
    one inside this interpreter -- not the machine's run queue."""
    before = int(HOST_LINE.search(host_report()).group(2))
    stop = threading.Event()
    extra = threading.Thread(target=stop.wait)
    extra.start()
    try:
        assert int(HOST_LINE.search(host_report()).group(2)) == before + 1
    finally:
        stop.set()
        extra.join()


def test_every_capture_path_says_what_host_it_ran_on():
    """Three front ends record on this station, and a field on one of them is a
    column with holes in it: the ramp is read across sessions, so an arm that
    does not answer is an arm the regression drops."""
    paths = (PACKAGE / "shrike" / "onair.py",
             PACKAGE / "besra" / "host" / "run_server.py",
             TOOLS / "kestrel_connect.py")
    if absent := [p for p in paths if not p.exists()]:
        pytest.skip(f"{absent[0]} is not in this tree: the distribution ships no "
                    "tools/, and kestrel's capture path lives there")
    silent = [str(path) for path in paths
              if "host_report()" not in path.read_text(encoding="utf-8")]
    assert not silent, (
        "every capture path prints core.rates.host_report beside its own "
        "teardown:\n  " + "\n  ".join(silent))
