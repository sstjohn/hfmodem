# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The decode-quality number an ARDOP receiver reports back, and how it is measured.

Every DATAACK/DATANAK carries a five-bit code its recipient reads as ``Q = 38 + 2·code``
(``DecodeACKNAK``, SoundInput.c), and the reference climbs its FSK→PSK→QAM ladder on
an exponential average of exactly that number: ``Gearshift_9`` (ARQ.c) shifts up when
the average exceeds a per-mode threshold — ``GetShiftUpThresholds`` gives ``{80, 80,
80, 76, 85, 75}`` at 2000 Hz and ``{80, 84, 84, 75, 79}`` at 500 Hz — and two frames
running have been acknowledged. A receiver that reports a constant 100 tells its peer
the path is flawless, and the peer climbs until it is sending a mode the receiver
cannot read.

So the number is measured, and measured the way the reference measures it. These are
ardopcf's own formulas — ``Update4FSKConstellation`` and ``UpdatePhaseConstellation``
in SoundInput.c — over the same quantities besra already computes: 4FSK tone
magnitude² (there ``intToneMags``, filled by ``powf(re,2)+powf(im,2)``; here
:func:`detect.tone_mag_series`) and differential symbol phase in milliradians. Both
are ratio metrics, scale-free in the tone magnitudes and absolute in the phases, so
they carry over with no calibration of our own — which is the point: a gateway's
ladder was tuned against peers reporting these numbers.

The parts of this module ``NOTICE`` names as ardopcf's are under that project's
MIT licence, Copyright (c) 2014-2024 Rick Muething, John Wiseman, Peter LaRue;
the copyright and permission notice it requires ship in ``NOTICE``.
"""

from __future__ import annotations

import math

import numpy as np

#: A frame that RS-corrected with few errors is reported no worse than this, however
#: loose its constellation looked. Both the number and the rule are the reference's,
#: applied at every mode (SoundInput.c, immediately before ``returnframe``)::
#:
#:     if (blnDecodeOK && (totalRSErrors / intNumCar) < (intRSLen / 4)
#:         && intLastRcvdFrameQuality < 80)
#:             intLastRcvdFrameQuality = 80;
#:
#: It is *not* the lowest shift-up bar, which is what this note used to claim.
#: ``GetShiftUpThresholds`` (ARQ.c) holds {82,84,84,85} at 200 Hz, {80,84,84,75,79} at
#: 500, {80,80,80,80,75} at 1000, {80,80,80,76,85,75} at 2000 and {60,85,85} on FM: 80
#: clears six of those twenty-three bars outright, so an RS-clean but scruffy decode
#: reports 80 and invites a shift up wherever the bar is 60, 75, 76 or 79. That is the
#: reference's own behaviour and it is reproduced rather than corrected — the whole
#: point of the number is that a gateway's ladder meets the peer it was tuned against.
RS_CLEAN_FLOOR = 80

#: Errors per carrier strictly below ``r // RS_CLEAN_DIVISOR`` count as "few". Both
#: sides of that comparison are C integer divisions in the reference, and they are
#: integer here: at ``r`` = 50 or 150 a true quotient would move the bar half a symbol.
RS_CLEAN_DIVISOR = 4


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
    """``Update4FSKConstellation``: how much of each symbol's tone power landed off
    the tone that was decided.

    ``mags`` is magnitude² per tone per symbol, shape ``(4, nsym)``. The reference
    plots each symbol at a radius that shrinks as the three losing tones take a
    larger share of the total, and averages the shortfall from the ideal radius of
    42 — integer arithmetic throughout, so the per-symbol distance is
    ``min(37, ⌊80 · off_share⌋)``. The 2.7 scaling is its own empirical calibration.
    """
    if mags.shape[1] == 0:
        return 0
    total = mags.sum(axis=0)
    off = np.divide(total - mags.max(axis=0), total,
                    out=np.zeros_like(total), where=total > 0)
    distance = np.minimum(37.0, np.floor(80.0 * off))
    return _clip(100.0 - 2.7 * float(distance.mean()))


def psk_quality(dphase: np.ndarray, step_mrad: float) -> int:
    """``UpdatePhaseConstellation`` for PSK: mean distance of the differential phases
    from the constellation grid, as a fraction of half a symbol step.

    ``dphase`` is one carrier's differential phases in milliradians and ``step_mrad``
    the constellation spacing (1571 for 4PSK, 785.4 for 8PSK). A phase landing on a
    grid point scores 100; one landing half a step away — as far from a decision as it
    can be — scores 0. The reference skips its first stored phase and still divides by
    the full count, which is reproduced here.
    """
    return _clip(100.0 - 200.0 * _mean_phase_error(dphase, step_mrad) / step_mrad)


def qam_quality(dphase: np.ndarray, mag: np.ndarray, step_mrad: float) -> int:
    """``UpdatePhaseConstellation`` for 16QAM: the PSK phase score scaled down by how
    far the two amplitude rings scatter, since half of a QAM symbol's information is
    in its radius. ``mag`` is the per-symbol magnitude alongside ``dphase``.

    The reference splits the rings at 75% of the peak magnitude to count them but
    accumulates their errors about the midpoint of the two ring averages; both are
    kept, because together they are the definition of the number a peer gearshifts on.
    """
    phase_score = 100.0 - 200.0 * _mean_phase_error(dphase, step_mrad) / step_mrad
    body = np.asarray(mag, dtype=np.float64)[1:]
    if body.size == 0:
        return 0
    inner, outer = body[body < 0.75 * body.max()], body[body >= 0.75 * body.max()]
    if inner.size == 0 or outer.size == 0 or inner.mean() <= 0:
        # One ring carrying nothing is not scatter: a carrier that dropped out
        # leaves the amplitude half of a QAM symbol unmeasurable, and dividing that
        # ring's own error by its zero mean made the whole score NaN — which the
        # plain two-sided clamp below used to read back as a flawless 100.
        return _clip(phase_score)
    mid = (inner.mean() + outer.mean()) / 2.0
    err_outer = np.abs(outer.mean() - body[body > mid]).sum()
    err_inner = np.abs(inner.mean() - body[body <= mid]).sum()
    scatter = (err_inner / (inner.size * inner.mean())
               + err_outer / (outer.size * outer.mean()))
    return _clip((1.0 - scatter) * phase_score)


def _mean_phase_error(dphase: np.ndarray, step_mrad: float) -> float:
    p = np.asarray(dphase, dtype=np.float64)
    if p.size < 2:
        return 0.5 * step_mrad          # nothing to measure: worst case
    body = p[1:]
    err = np.abs(body - step_mrad * np.round(body / step_mrad))
    return float(err.sum() / p.size)
