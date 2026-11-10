# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Deterministic impairment cells and fixed-sample sensitivity crossings."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import hilbert

from hfmodem.brubru import waveform as w


class Inconclusive(ValueError):
    pass


def impair(audio, *, snr, seed, ppm=0, drift=0, notch_s=0, depth=1):
    positions = np.arange(len(audio)) / (1 + ppm * 1e-6)
    changed = (map_coordinates(audio, [positions], order=3, mode='constant')
               if ppm else audio.copy())
    t = np.arange(len(audio)) / w.RATE
    if drift:
        changed = (hilbert(changed) * np.exp(1j * np.pi * drift * t ** 2)).real
    if notch_s:
        half = round(notch_s * w.RATE / 2)
        mid = len(changed) // 2
        changed[mid - half:mid + half] *= depth
    power = float(np.mean(audio ** 2))
    sigma = np.sqrt(power * w.RATE / (2 * 3000) / 10 ** (snr / 10))
    return changed + np.random.default_rng(seed).normal(0, sigma, len(audio))


def crossing(cells):
    """Lowest stable >=50% cell, interpolated against its immediate predecessor."""
    points = sorted(cells.items())
    candidates = [i for i, (_, successes) in enumerate(points)
                  if successes >= 10 and all(n >= 10 for _, n in points[i:])]
    if not candidates or candidates[0] == 0:
        raise Inconclusive('sweep does not bracket a crossing')
    i = candidates[0]
    low, below = points[i - 1]
    high, above = points[i]
    if any(b > a for (_, b), (_, a) in zip(points, points[1:])):
        raise Inconclusive('refinement is not monotonic')
    return low + (10 - below) * (high - low) / (above - below)


def sensitivity(measure, *, lower=-12, upper=8):
    """Twenty paired trials per cell; repeat an inconclusive refinement once."""
    for seed_set in (0, 1):
        cells = {}
        def cell(snr):
            if snr not in cells:
                cells[snr] = sum(bool(measure(snr, seed_set * 10000 + trial))
                                 for trial in range(20))
            return cells[snr]
        lo, hi = lower, upper
        for _ in range(20):
            for snr in np.arange(lo, hi + 0.01, 2):
                cell(float(snr))
            ordered = sorted(cells)
            stable = [s for s in ordered if cells[s] >= 10 and
                      all(cells[h] >= 10 for h in ordered if h >= s)]
            if stable and stable[0] > lo:
                high = stable[0]
                low = max(s for s in ordered if s < high)
                for snr in np.arange(low, high + 0.01, 0.5):
                    cell(float(snr))
                refined = {s: cells[s] for s in cells if low <= s <= high}
                try:
                    return {'point': crossing(refined), 'cells': cells, 'seed_set': seed_set}
                except Inconclusive:
                    break
            if not stable:
                hi += 4
            elif stable[0] == lo:
                lo -= 4
        else:
            raise Inconclusive('sweep failed to bracket after twenty extensions')
    raise Inconclusive('two seed sets produced inconclusive refinements')
