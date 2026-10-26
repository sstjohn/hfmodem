# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Local L4 refinement on the locator's retained 6 kHz stream."""
import numpy as np
from scipy.signal import fftconvolve

from . import varahf500 as rx
from .bw500_features import LP, REF


def demod(low, origin, start, onsets=None):
    center = (start - origin) // 8
    first = max(0, center - 128)
    stop = min(len(low), center + (rx._NCOL_DEMOD + 8) * 64 + 128)
    segment = low[first:stop]
    if not len(segment):
        return {}, (), 0.
    phase = np.arange(len(segment)) % 640
    mix = np.exp(-2j * np.pi * np.array([f for _, f in rx.SUBBANDS])[:, None] * phase / 6000)
    bb = fftconvolve(segment * mix, LP[None, :], 'same', axes=-1)
    center -= first
    score = 0.
    if onsets is None:
        offsets = np.arange(-64, 65)
        offsets = offsets[(center + offsets >= 0)
                          & (center + offsets + (rx.NSYM - 1) * 64 < len(segment))]
        if not len(offsets):
            return {}, (), 0.
        indices = center + offsets[:, None] + np.arange(rx.NSYM) * 64
        values = bb[:, indices] * REF[:, None, :]
        live = abs(values) > .3 * np.median(abs(values), axis=2)[:, :, None]
        d = values[:, :, 1:] * values[:, :, :-1].conj()
        mask = live[:, :, 1:] & live[:, :, :-1]
        w = np.zeros_like(d)
        np.divide(d * d, abs(d) ** 2, out=w, where=mask)
        counts = mask.sum(axis=2)
        scores = np.full(counts.shape, -1.)
        np.divide(abs(w.sum(axis=2)), counts, out=scores, where=counts > 0)
        best = scores.argmax(axis=1)
        onsets = tuple(int(origin + (first + center + offsets[k]) * 8 - start) for k in best)
        score = float(scores[np.arange(2), best].min())
    cols = {}
    for k, onset in enumerate(onsets):
        indices = (start + onset - origin) // 8 - first + np.arange(rx._NCOL_DEMOD) * 64
        cols[k] = bb[k, indices[(indices >= 0) & (indices < len(segment))]]
    return cols, onsets, score
