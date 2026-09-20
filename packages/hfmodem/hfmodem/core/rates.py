# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Index arithmetic between the sound card and a mode's native rate, and what
the card's own clock is measured to be doing.

The card is the station's single timebase. One capture stream feeds every enabled
protocol, each at its own native rate, and a position reported by any of them has
to be an exact function of the card's sample count rather than of that lane's own
decode history — otherwise two lanes disagree about when the same signal
happened, and the disagreement grows with the session.

Exact means integer. A float conversion accumulates, and the position a frame is
reported at is what a session's cadence is measured against: shrike's peer holds
a 1.25 s raster to a tenth of a millisecond over 77 seconds, and a timebase that
drifts cannot see that.

Which leaves the two ways that timebase can be wrong, and they are here because
three capture front ends ask the same question of the same card and had been
answering it from three states of repair. `lost_step` is capture the converter
timestamped and Python was never handed; `rate_fit` is the rate itself. They are
read together — the fit's numerator is the count Python was handed, so a block
the interpreter was too busy to accept subtracts from it exactly as a slow
oscillator would, and only the loss count separates them.

`host_report` is neither, and is here because it is read on the same line: what
the machine was when the reading was taken, for want of which a seven-day ramp
in the loss had to be reconstructed from file mtimes afterwards.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np

#: The sound card's rate. Every native rate here divides it.
CARD_RATE_HZ = 48000


def to_native(n_card: int, rate: int) -> int:
    """The index in a `rate` stream of card sample `n_card`."""
    return n_card * rate // CARD_RATE_HZ


def lost_step(ahead: float, prev: float | None, blocksize: int, fs: int) -> int:
    """Samples the converter timestamped between two callbacks and never delivered.

    `ahead` is the converter's own instant for this block minus where the
    delivered count says it should be, and the previous callback's is `prev`.

    A starved interpreter loses whole blocks and nothing announces it: eight
    pure-Python threads on this station cost 87% of a 90 s stream with
    `input_overflow` false on every callback that did run. The only trace is this
    difference stepping by more than HALF a block — half, so that a crystal's
    drift, which is parts per million of one, can never accumulate into it.

    ONE-WAY, AND ONLY SOUND AT THE DEVICE'S NATIVE RATE. Lost capture does not
    come back, so a down-step is not a credit and this never returns one. That
    makes it a ratchet on any `ahead` series that swings both ways by more than
    half a block, which is what CoreAudio's rate converter produces: asking the
    built-in 44100 device for 48000 sawtooths the timestamps ±125 samples and
    banks 2082 ms of loss over 30 s while `rate_fit` reads the same stream at
    −0.73 ppm. On this station's codec, opened at the 48000 it runs at, the two
    agree within 9% and there is nothing to swing. A stream resampled by the host
    is a stream this count will libel, and the fit is the one to believe.
    """
    if prev is None or ahead - prev <= blocksize / 2 / fs:
        return 0
    return round((ahead - prev) * fs)


def rate_fit(points: Sequence[tuple[float, int]], fs: int) -> tuple[float, float]:
    """(ppm, 1 sigma) of the capture clock against the system clock.

    `points` are (ADC instant, sample index) pairs, thinned to about one a
    second; one a second is ample over minutes and keeps a list that lives as
    long as a session from growing into a leak at 375 callbacks a second.

    FITTED, NOT DIFFERENCED. Dividing total samples by elapsed time charges
    stream startup — ~0.1 s during which the clock runs and no samples arrive —
    to the crystal, which fabricates about −1700 ppm out of nothing and then
    converges to it too slowly to notice. A least-squares slope puts any fixed
    startup or delivery latency in the INTERCEPT, where it belongs.

    ANCHORED ON THE CONVERTER'S OWN TIMESTAMPS. Callback-entry time carries the
    driver's 3 ms rms delivery jitter, which is load-dependent and at 50 s of
    session is worth sigma 17 ppm of pure fiction: the 2026-08-13 report of
    −32 ppm was that jitter and nothing else, on a crystal two clean long fits
    place at +2.4 ± 7.8. So a caller passes the instant the converter stamped
    and not the instant Python reached the callback.

    THE SIGMA IS THE SLOPE'S STANDARD ERROR, in the same ppm as the figure it
    qualifies, and it falls as the baseline grows. It used to be the residual
    scatter over the rate — a spread in microseconds printed as a ppm, and tight
    on a short stream for a reason that had nothing to do with how well the rate
    was known. It is also the whole of the baseline rule: a fixed "under 30 s,
    refuse" was one copy's answer and the other's was to print a short fit's ppm
    with a confidence it had not earned, and neither is needed once the error bar
    is honest. `preflight`'s stream is a dozen seconds by the time it asks, and
    its question — whether the card is running at 48000 at all, where a device
    handing back 44100 reads about −81000 ppm — is one a dozen seconds answers.

    A SLOPE PAST A FEW TENS OF PPM IS NOT A CRYSTAL. The numerator is the sample
    count Python was handed, so blocks the interpreter was too busy to accept
    subtract from it exactly as a slow oscillator would, and the two are
    indistinguishable here. Sessions read −750 to −6091 ppm on a card preflight
    measured at +7.4 twenty minutes earlier, and every reading past ±20 ppm in
    this station's whole record is negative — which a crystal has no reason to be
    and lost capture can only be. `lost_step` is what separates them.

    The machine's built-in microphone — a different converter at a different rate
    — reads the same +1.3 to +1.4 ppm, so this is the offset of the whole audio
    clock domain against `time.monotonic()` on this Mac and not a fact about the
    dongle's crystal. Which crystal it actually belongs to is undetermined: no
    traceable reference was involved, only two clocks in the same laptop.
    """
    if len(points) < 3:
        return float("nan"), float("nan")
    t, n = np.array(points, float).T
    t -= t[0]
    if t[-1] <= 0.0:
        return float("nan"), float("nan")
    # On delivered-minus-nominal, so the numbers stay small; the raw product
    # would put 1 ppm of signal 16 decimal places down.
    resid = n - n[0] - fs * t
    A = np.vstack([t, np.ones_like(t)]).T
    coef = np.linalg.lstsq(A, resid, rcond=None)[0]
    err = resid - A @ coef
    var = float(err @ err) / (len(t) - 2)
    cov = var * np.linalg.inv(A.T @ A)
    return float(coef[0]) / fs * 1e6, float(np.sqrt(cov[0, 0])) / fs * 1e6


#: Seconds since boot. Darwin's CLOCK_MONOTONIC runs across sleep and agrees with
#: `last reboot` to 0.2 s over 8.7 hours of uptime; Linux's stops while suspended,
#: where CLOCK_BOOTTIME is the one that does not. Wall since boot rather than
#: awake time, because `last reboot` is what the record so far is dated against.
_SINCE_BOOT = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)


def host_report() -> str:
    """The machine a session ran on, printed beside its `capture clock` line.

    Every figure in the seven-day loss ramp -- daily medians 0.00 to 7.15 ms of
    air lost per second of stream, monotone across one uninterrupted boot
    session -- was recovered afterwards from file mtimes and `last reboot`,
    because nothing a session printed said what host it had. Two numbers make
    that a column beside the loss instead of an archaeology after it.

    THESE TWO, because they are the pair the seven days could not tell apart:
    the machine's uptime and the number of threads this process runs were both
    monotone in calendar time, one because the machine stayed up and the other
    because the modem grew lanes. A reboot resets one and not the other.

    NOT LOAD AVERAGE, which is the obvious third and is the one already
    excluded: a fresh-boot probe at load 4.1 with three cores pegged fit
    +4.30 ppm with zero shortfall, from the same code that read -4018 ppm during
    a session. What starves this callback is the GIL and not the run queue --
    eight pure-Python threads cost 87% of a 90 s stream -- so the count that can
    matter is the one inside this interpreter. What would put load back on the
    line: arms at equal uptime and equal thread count whose loss tracks the
    machine's instead.

    AT TEARDOWN, WHILE THE SESSION IS STILL ASSEMBLED, rather than in the
    opening banner. A capture path's threads start with its stream, so a count
    taken where the banner prints is 1 on every arm ever run; uptime does not
    care which end of a one-minute arm it is read at, and the length to correct
    it by is on the `capture clock` line already.

    Neither figure explains any loss. They are facts about the host, said where
    the loss figure is, so that a regression has both on one row.
    """
    return (f"host: {time.clock_gettime(_SINCE_BOOT):.0f} s since boot, "
            f"{threading.active_count()} threads in this process")


def clock_report(*, samples: int, lost: int, xruns: int, underruns: int,
                 points: Sequence[tuple[float, int]], blocksize: int,
                 fs: int = CARD_RATE_HZ) -> str:
    """The line every capture path prints about its own timebase.

    One wording, because `tools/clock_shortfall.py` parses it out of every log on
    the record to rank the sessions by milliseconds of missing air. A front end
    that prints its own phrasing is a front end the ranking cannot see, and until
    this was shared no VARA recording on the record carried the line at all.

    `xruns` is the driver's opinion and it is not enough: the loss named here is
    the one PortAudio does not flag, so the line has to carry it or `0 xruns`
    reads as a clean stream. What a clean line is worth is bounded the other way
    too — the count is anchored on the converter's own timestamps, so loss
    upstream of them is invisible to it. A second receiver is the only reading
    that could see such a hole, and the one witnessed arm on file does not show
    one: corrected for an 8300 ppm framing error in the aligner it reads +260 ms
    against this counter's 295, i.e. consistent with it. The bound stands on the
    anchoring, not on that arm.
    """
    ppm, sigma = rate_fit(points, fs)
    got = (f"{ppm:+.2f} ppm vs the system clock (fit sigma {sigma:.2f})"
           if ppm == ppm else "too short to measure the clock (a slope needs "
           "three points across a nonzero baseline)")
    gap = (f" -- and {lost} samples ({lost / fs * 1e3:.0f} ms, "
           f"{lost / (samples + lost) * 100:.2f}% of what was captured) the "
           f"converter timestamped never reached us. THE STREAM IS SPLICED: its "
           f"clock reads that far short of the air by the end, and a shortfall is "
           f"what any ppm above is measuring"
           if lost >= blocksize else
           " -- and no loss we can see, which is as far as this goes: the count "
           "is anchored on the converter's own timestamps and says nothing about "
           "capture lost upstream of them")
    return (f"capture clock: {samples} samples ({samples / fs:.1f} s), "
            f"{xruns} xruns, {underruns} TX underruns, {got}{gap}")
