# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Scattered-pilot channel estimation for OFDM over fading HF channels.

This module estimates a full time-frequency channel surface H[s, k] from a
scattered (DRM-style) pilot lattice:

1. **Common-phase tracking** -- residual CFO after acquisition shows up as a
   phase ramp common to all carriers. It is measured from conjugate products
   between pilots on the same carrier (the lattice repeats every ``step``
   symbols, so both factors are pilots), fitted with a low-order polynomial,
   and removed before interpolation -- complex interpolation across a rotating
   phase would otherwise combine destructively.
2. **2-D kernel smoothing** -- the channel surface is a Gaussian-kernel
   (Nadaraya-Watson) average of the pilot observations, separable in time and
   frequency. The kernel widths trade estimation-noise averaging against
   tracking bias; the defaults are matched to the Poor profile (1 Hz spread,
   2 ms delay). Normalising by the smoothed pilot mask handles interpolation,
   edge extrapolation, and the staggered lattice uniformly.
3. **Noise variance** from second differences along each pilot track: for
   white noise E|h[i+1] - 2h[i] + h[i-1]|^2 = 6 sigma^2. Channel variation also
   contributes, biasing this estimate upwards on faster channels. Relative
   per-cell weights |H|^2/sigma^2 preserve reliability differences for LDPC
   decoding and soft combining.
4. **Dead-carrier masking** -- carriers whose time-averaged SNR sits at the
   channel-estimation bias floor carry no information, but a biased |H|^2
   estimate would still feed the decoder confidently wrong LLRs. They are
   erased (weight 0). The threshold is relative to the median carrier SNR with
   an absolute cap, so a uniformly weak (but honest) channel masks nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import convolve


@dataclass
class FadingEstimate:
    noise_var: float
    common_phase: np.ndarray          # per-symbol derotation applied (rad)
    carrier_snr: np.ndarray           # per-carrier time-averaged SNR (linear)
    masked: np.ndarray                # bool per carrier: erased as dead
    H: np.ndarray = field(repr=False)        # (n_syms, n_carriers)
    weights: np.ndarray = field(repr=False)  # |H|^2/sigma^2, 0 on masked carriers


def _gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(np.ceil(3 * sigma)))
    t = np.arange(-r, r + 1)
    return np.exp(-t**2 / (2 * sigma**2))


def integer_cfo(grids: np.ndarray, carriers: np.ndarray, mask: np.ndarray,
                n_fft: int, search: int = 2, step: int = 3) -> int:
    """Integer carrier offset from pilot-track coherence.

    ``grids``: (n_syms, n_fft) full FFT rows; ``mask``: (n_syms, n_carriers)
    pilot lattice. At the true shift the pilots ``step`` symbols apart on the
    same carrier are channel-correlated; at a wrong shift they read
    uncorrelated data cells.
    """
    n_probe = min(grids.shape[0], 15)
    best_d, best = 0, -1.0
    for d in range(-search, search + 1):
        bins = (np.asarray(carriers) + d) % n_fft
        hp = np.where(mask[:n_probe], grids[:n_probe][:, bins], 0)
        num = np.abs(np.sum(hp[step:] * np.conj(hp[:-step])))
        den = np.sum(np.abs(hp) ** 2) + 1e-12
        if num / den > best:
            best, best_d = num / den, d
    return best_d


def common_phase(hp: np.ndarray, mask: np.ndarray, step: int = 3) -> np.ndarray:
    """Per-symbol common phase c[s] (residual CFO + drift) from pilot tracks.

    Conjugate products over ``step`` symbols give wrapped phase increments
    dphi[s] ~ c[s+step] - c[s]. Weighted linear fits to wrapped and unwrapped
    increments compete by circular residual error, producing a quadratic
    phase model. The initial increment has a CFO ambiguity every 1/(step*T),
    so acquisition must already be within half that interval (~6 Hz for the
    standard lattice).
    """
    n_syms = hp.shape[0]
    c = np.zeros(n_syms)
    if n_syms <= step:
        return c
    z = np.sum(np.where(mask[step:] & mask[:-step],
                        hp[step:] * np.conj(hp[:-step]), 0), axis=1)
    wgt = np.abs(z)
    if wgt.sum() < 1e-12:
        return c
    # These are phase increments across one pilot period. Drift can carry
    # them through +/-pi during a frame even when acquisition is accurate.
    # Unwrap increments before fitting; fitting wrapped angles reverses the
    # apparent drift and destroys the late symbols of long frames.
    dphi = np.angle(z)
    s = np.arange(dphi.size)
    if dphi.size >= 4:
        B, A = np.polyfit(s, dphi, 1, w=np.sqrt(wgt))
        unwrapped = np.unwrap(dphi)
        if not np.array_equal(dphi, unwrapped):
            bu, au = np.polyfit(s, unwrapped, 1, w=np.sqrt(wgt))
            # At low SNR unwrapping can accumulate random full turns. Select
            # the fit by its circular error on the actual pilot products,
            # rather than trusting that arbitrary unwrapped branch.
            def error(a, b):
                return np.sum(wgt * (1 - np.cos(dphi - (a + b * s))))
            if error(au, bu) < error(A, B):
                B, A = bu, au
    else:
        A, B = np.average(dphi, weights=wgt), 0.0
    # dphi(s) = c(s+step) - c(s) with c(s) = alpha s + beta s^2
    beta = B / (2 * step)
    alpha = (A - beta * step**2) / step
    sa = np.arange(n_syms)
    return alpha * sa + beta * sa**2


def timing_ramp(hp: np.ndarray, mask: np.ndarray, stag: int = 2,
                df: int = 6) -> float:
    """Common linear phase slope across carriers (rad/carrier), i.e. the bulk
    timing offset of the FFT window.

    Complex smoothing across frequency would average rotating phasors and
    shrink |H| -- fatal for QAM amplitude decisions -- so the bulk ramp must
    come off before interpolation. Estimated in two stages of conjugate
    products: stagger-adjacent pilots (1 symbol, 2 carriers apart) are
    unambiguous for |slope| < pi/2 (a quarter FFT of timing offset), then
    in-symbol gap-6 pairs refine. The lattice guarantees both members of each
    pair are pilots.
    """
    slope = 0.0
    for ds, dr in ((1, stag), (0, df)):
        both = mask[ds:, dr:] & mask[: mask.shape[0] - ds, : mask.shape[1] - dr]
        z = np.sum(np.where(both, hp[ds:, dr:] * np.conj(
            hp[: hp.shape[0] - ds, : hp.shape[1] - dr]
        ) * np.exp(-1j * slope * dr), 0))
        slope += np.angle(z) / dr
    return slope


def estimate(Y: np.ndarray, mask: np.ndarray, pilot_value: complex = 1 + 0j,
             sigma_t: float = 2.5, sigma_f: float = 1.5, step: int = 3,
             mask_rel: float = 0.15, mask_cap: float = 0.25,
             ref: np.ndarray | None = None, stag: int = 2, df: int = 6,
             robust: bool = False) -> FadingEstimate:
    """Estimate H[s, k], noise variance, and LLR weights from scattered pilots.

    ``Y``: (n_syms, n_carriers) equaliser input cells (active carriers only);
    ``mask``: pilot lattice, True where ``Y`` holds a pilot. Returns Y's frame:
    apply ``exp(-1j*common_phase)`` to Y before dividing by ``H``.

    ``ref`` generalises the pilot lattice to arbitrary known cells (the
    decision-directed second pass): a per-cell reference value wherever
    ``mask`` is True, dividing each observation by its own reference instead
    of the uniform ``pilot_value``.

    ``stag``/``df`` describe the lattice geometry for the timing-ramp pairs;
    ``robust`` swaps the noise estimator's mean for a median (|d2|^2 is
    exponential under Gaussian noise, so median/ln2 is unbiased there yet
    ignores the heavy tail an impulsive band or decision errors add).
    """
    if ref is None:
        hp = np.where(mask, Y / pilot_value, 0)
    else:
        hp = np.where(mask, Y * np.conj(ref)
                      / np.maximum(np.abs(ref) ** 2, 1e-12), 0)

    c = common_phase(hp, mask, step)
    hp = hp * np.exp(-1j * c)[:, None]
    ramp = timing_ramp(hp, mask, stag, df)
    rvec = np.exp(-1j * ramp * np.arange(mask.shape[1]))
    hp = hp * rvec[None, :]

    kt = _gauss_kernel(sigma_t)[:, None]
    kf = _gauss_kernel(sigma_f)[None, :]
    m = mask.astype(float)
    num = convolve(convolve(hp, kt, mode="same"), kf, mode="same")
    den = convolve(convolve(m, kt, mode="same"), kf, mode="same")
    H = num / np.maximum(den, 1e-12) * np.conj(rvec)[None, :]

    # noise from second differences along each pilot track
    d2s = []
    for r in range(mask.shape[1]):
        v = hp[mask[:, r], r]
        if v.size >= 3:
            d2s.append(np.abs(v[2:] - 2 * v[1:-1] + v[:-2]) ** 2)
    if not d2s:
        noise_var = 1e-9
    elif robust:
        noise_var = float(np.median(np.concatenate(d2s)) / (6 * np.log(2)))
    else:
        d2 = np.concatenate(d2s)
        noise_var = float(d2.sum() / (6 * d2.size))
    noise_var = max(noise_var, 1e-12)

    # Per-carrier SNR from raw received power -- data cells have unit average
    # energy just like pilots, so every cell is a power probe and every
    # carrier is measured at full resolution. The smoothed |H|^2 would leak
    # healthy neighbours into a dead carrier and hide it from the mask.
    raw_pwr = np.mean(np.abs(Y) ** 2, axis=0)
    carrier_snr = np.maximum(raw_pwr / noise_var - 1.0, 1e-6)
    thresh = min(mask_cap, mask_rel * float(np.median(carrier_snr)))
    masked = carrier_snr < thresh
    weights = np.abs(H) ** 2 / noise_var
    weights[:, masked] = 0.0

    return FadingEstimate(noise_var=noise_var, common_phase=c,
                          carrier_snr=carrier_snr, masked=masked,
                          H=H, weights=weights)
