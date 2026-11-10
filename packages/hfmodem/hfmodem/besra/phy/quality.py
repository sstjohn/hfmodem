# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The decode-quality number an ARDOP receiver reports back, and how it is measured.

Every DATAACK and DATANAK carries a five-bit code that its recipient reads as
``Q = 38 + 2·code``, and an ARDOP sender picks its next mode from a running average
of those numbers: it moves up the FSK→PSK→QAM ladder once the average clears a
per-mode bar (80, 80, 80, 76, 85 and 75 at 2000 Hz; 80, 84, 84, 75 and 79 at 500)
and two frames in a row have been acknowledged. A receiver that always reports 100
tells its peer the path is flawless, and the peer climbs until it is sending a mode
the receiver cannot read.

So the number has to be measured, and on the scale the peers were tuned against.
For 4FSK that is how much of each symbol's power fell on the tones not chosen; for
PSK, how far the differential phases sit from the constellation grid; for 16QAM,
that phase score scaled down by how loosely the two amplitude rings hold together.
The inputs are what besra already computes — tone magnitude² from
:func:`detect.tone_mag_series` and differential phase in milliradians — and each
score is a ratio, so it carries over with no calibration of our own. The constants
below (the 37-point cap and 2.7 slope for 4FSK, the half-step scale for phase, the
75% ring split for 16QAM) are what make our numbers read like every other ARDOP
station's; they are matched, not tuned.
"""

from __future__ import annotations

import math

import numpy as np

#: The least a frame is reported at when RS put it right with few repairs: errors
#: per carrier strictly below a quarter of the parity count. However loose the
#: constellation looked, the bytes came through, and ARDOP peers report such a
#: frame at no less than 80 at every mode.
#:
#: That is not the lowest bar on the ladder. The bars run {82,84,84,85} at 200 Hz,
#: {80,84,84,75,79} at 500, {80,80,80,80,75} at 1000, {80,80,80,76,85,75} at 2000
#: and {60,85,85} on FM, so an RS-clean but scruffy frame reported at 80 still
#: invites a shift up wherever the bar is 60, 75, 76 or 79. We report what the
#: peers report, because the point of the number is that a gateway reads it the way
#: it reads everyone else's.
RS_CLEAN_FLOOR = 80

#: Errors per carrier strictly below ``r // RS_CLEAN_DIVISOR`` count as "few". Both
#: sides of the comparison are whole numbers, as they are on the peers: at ``r`` = 50
#: or 150 a true quotient would move the bar by half a symbol.
RS_CLEAN_DIVISOR = 4

#: 4FSK: a symbol's shortfall is ⌊80 × off-tone share⌋ points, capped here, and the
#: score drops by this slope per point of mean shortfall.
_FSK_CAP = 37.0
_FSK_SLOPE = 2.7

#: 16QAM rings: a magnitude at or above this share of the strongest is outer.
_RING_SPLIT = 0.75


def _clip(q: float) -> int:
    """Clamp a quality to the 0-100 the five-bit wire code can carry.

    NaN is floored, not clamped: ``min(100.0, nan)`` is ``100.0``, so the plain
    two-sided clamp answered *flawless path* for a measurement that was not a
    number — reachable from :func:`qam_quality` whenever the inner ring reads zero
    (a dropped carrier), where the scatter term divides by its own mean."""
    if not math.isfinite(q):
        return 0
    return int(max(0.0, min(100.0, q)))


def fsk_quality(mags: np.ndarray) -> int:
    """How cleanly each 4FSK symbol landed on the tone that was decided.

    ``mags`` is magnitude² per tone per symbol, shape ``(4, nsym)``. A symbol whose
    power is all in the winning tone costs nothing; the share that fell on the
    other three costs ⌊80 × share⌋ points, at most 37. The score is 100 less 2.7
    times the mean cost.
    """
    if mags.shape[1] == 0:
        return 0
    total = mags.sum(axis=0)
    off_share = np.divide(total - mags.max(axis=0), total,
                          out=np.zeros_like(total), where=total > 0)
    shortfall = np.minimum(_FSK_CAP, np.floor(80.0 * off_share))
    return _clip(100.0 - _FSK_SLOPE * float(shortfall.mean()))


def psk_quality(dphase: np.ndarray, step_mrad: float) -> int:
    """How close a carrier's differential phases sit to the constellation grid.

    ``dphase`` is in milliradians and ``step_mrad`` is the grid spacing (1571 for
    4PSK, 785.4 for 8PSK). On the grid scores 100; half a step off, as far from a
    decision as a phase can be, scores 0.
    """
    return _clip(_phase_score(dphase, step_mrad))


def qam_quality(dphase: np.ndarray, mag: np.ndarray, step_mrad: float) -> int:
    """The 16QAM score: the phase score, scaled down by how far the two amplitude
    rings scatter, since half of a 16QAM symbol's information is in its radius.

    ``mag`` is the per-symbol magnitude alongside ``dphase``; its first entry is
    the reference symbol and takes no part. Symbols are counted into rings by a
    split at 75% of the strongest, but each ring's scatter is summed about its own
    mean over the symbols on its side of the midpoint between the two means. The
    two splits do not always agree, and the score is defined with both.
    """
    phase_score = _phase_score(dphase, step_mrad)
    body = np.asarray(mag, dtype=np.float64)[1:]
    if body.size == 0:
        return 0
    split = _RING_SPLIT * body.max()
    inner, outer = body[body < split], body[body >= split]
    if inner.size == 0 or outer.size == 0 or inner.mean() <= 0:
        # One ring carrying nothing is not scatter: a carrier that dropped out
        # leaves the amplitude half of a QAM symbol unmeasurable, and dividing that
        # ring's own error by its zero mean made the whole score NaN — which the
        # plain two-sided clamp below used to read back as a flawless 100.
        return _clip(phase_score)
    mid = (inner.mean() + outer.mean()) / 2.0
    outer_spread = np.abs(outer.mean() - body[body > mid]).sum()
    inner_spread = np.abs(inner.mean() - body[body <= mid]).sum()
    scatter = (inner_spread / (inner.size * inner.mean())
               + outer_spread / (outer.size * outer.mean()))
    return _clip((1.0 - scatter) * phase_score)


def _phase_score(dphase: np.ndarray, step_mrad: float) -> float:
    """100 less 200 × the mean distance to the grid over the step, unclipped.

    The first phase is the reference symbol's and is not measured, but it stays in
    the count: the sum over the rest is divided by them all. Fewer than two phases
    leave nothing to measure, which scores as the worst case."""
    p = np.asarray(dphase, dtype=np.float64)
    if p.size < 2:
        err = 0.5 * step_mrad
    else:
        body = p[1:]
        err = float(np.abs(body - step_mrad * np.round(body / step_mrad)).sum() / p.size)
    return 100.0 - 200.0 * err / step_mrad
