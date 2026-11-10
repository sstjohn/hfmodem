# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two-tone FSK for the narrow modes, parametrised by baud rate and tone pair.

Continuous-phase modulation, a non-coherent per-sample soft decision, and symbol
timing for synchronous streams. The 45.45 Bd RTTY and 100 Bd SITOR paths both
run through here.
"""
from __future__ import annotations

import numpy as np

FS = 48000
_TRACK_WINDOW_S, _TRACK_STEP_S, _TRACK_SPAN_S = 2.0, 1.0, 30.0


def tone_pair(x, fs, *, shift, near=0.0, span=400.0, tol=5.0, max_s=60.0, keyed=False) -> tuple[float, float]:
    """Find a tone pair and its summed power / median 1 Hz band power in dB.

    Both tones must be spectral peaks; rank pairs by their weaker tone so a
    lone strong carrier cannot win. For long inputs, select the highest-band-
    energy window on a one-second grid, then estimate from at most max_s.
    With ``keyed=True``, require alternating tone envelopes; strong framing
    or phasing can rescue correlation degraded by neighbouring signals.
    Array-like inputs may provide dtype, size and slicing to read lazily.
    """
    pairs = tone_pairs(x, fs, shift=shift, near=near, span=span, tol=tol,
                       max_s=max_s, keyed=keyed, baud=1000 / 15, limit=1)
    return pairs[0] if pairs else (float(near), -3000.0)


def tone_pairs(x, fs, *, shift, near=0.0, span=400.0, tol=5.0,
               max_s=60.0, keyed=False, baud=45.45, limit=4):
    """Rank distinct tone pairs by weaker-tone power, returning (centre, dB).

    Keyed validation tries 15 ms envelopes and one-unit envelopes at ``baud``.
    Nearby spectral peaks
    belonging to the same pair are coalesced; at most ``limit`` pairs return.
    """
    from collections import deque
    from scipy.signal import find_peaks

    if fs <= 0 or shift <= 0 or span < 0 or tol < 0 or max_s <= 0 or baud <= 0 or limit < 1:
        raise ValueError("invalid tone-pair search parameters")
    if not hasattr(x, "dtype"):
        x = np.asarray(x)
    if not x.size:
        raise ValueError("empty tone-pair input")
    real = not np.issubdtype(x.dtype, np.complexfloating)
    low = near - span - (shift + tol) / 2
    high = near + span + (shift + tol) / 2
    if real:
        low = max(0.0, low)
    low, high = max(low, -fs / 2), min(high, fs / 2)
    if high <= low:
        raise ValueError("search band is outside the sampled spectrum")

    def spectrum(v):
        n = max(v.size, round(fs))
        if real:
            f = np.fft.rfftfreq(n, 1 / fs)
            X = np.fft.rfft(v * np.hanning(v.size), n=n)
        else:
            f = np.fft.fftfreq(n, 1 / fs)
            X = np.fft.fft(v * np.hanning(v.size), n=n)
        keep = (f >= low) & (f <= high)
        return f[keep], np.abs(X[keep]) ** 2

    cap = max(1, round(max_s * fs))
    start = 0
    if x.size > cap:
        step = max(1, min(round(fs), cap))
        blocks = max(1, cap // step)
        energies = deque()
        total, best = 0.0, -1.0
        for pos in range(0, x.size - step + 1, step):
            _, p = spectrum(np.asarray(x[pos:pos + step]))
            energy = float(p.sum())
            energies.append(energy)
            total += energy
            if len(energies) > blocks:
                total -= energies.popleft()
            if len(energies) == blocks and total > best:
                best = total
                start = pos - (blocks - 1) * step
        start = min(start, x.size - cap)
    f, p = spectrum(np.asarray(x[start:start + cap]))
    origin = int(np.floor(low))
    bins = np.floor(f).astype(int) - origin
    power = np.bincount(bins, weights=p, minlength=int(np.floor(high)) - origin + 1)
    peaks, _ = find_peaks(np.r_[0.0, power, 0.0])
    peaks = peaks - 1
    floor = max(float(np.median(power)), np.finfo(float).tiny)
    candidates = []
    for i in peaks:
        for j in peaks[(peaks > i) & (np.abs(peaks - i - shift) <= tol)]:
            mid = origin + (i + j) / 2
            if abs(mid - near) <= span:
                candidates.append((min(power[i], power[j]), i, j, mid))
    validation_audio = np.asarray(x[start:start + cap])
    validation_fs = fs
    validation_offset = 0.0
    if keyed and fs > 6000:
        validation_offset = (low + high) / 2
        validation_audio, validation_fs, _, _ = _pair_baseband(
            validation_audio, fs, high, low)
    pairs = []
    for score, i, j, mid in sorted(candidates, reverse=True):
        if keyed and (score / floor < 10):
            continue
        if any(abs(mid - centre) < max(60, 2 * tol) for centre, _ in pairs):
            continue
        if not keyed or any(keyed_pair(validation_audio, validation_fs,
                                       f_hi=origin + j - validation_offset,
                                       f_lo=origin + i - validation_offset, baud=rate)
                            for rate in dict.fromkeys((1000 / 15, baud))):
            ratio = 10 * np.log10(max(float(power[i] + power[j]), np.finfo(float).tiny) / floor)
            pairs.append((float(mid), float(ratio)))
            if len(pairs) >= limit:
                break
    return pairs


def fit_pair(audio, fs, *, midpoint, shift=170, baud=45.45):
    """Refine an interfered pair by integrating each tone through four units.

    Spectral skirts can displace a readable RTTY pair enough to distort data.
    Clean alternation and SITOR phasing keep their spectral estimates.
    """
    from hfmodem.lanner import sitor

    audio, fs, hi, lo = _pair_baseband(
        np.asarray(audio), fs, midpoint + shift / 2, midpoint - shift / 2)
    offset = midpoint - (hi + lo) / 2
    midpoint = (hi + lo) / 2
    soft = discriminate(audio, f_hi=hi, f_lo=lo, baud=sitor.BAUD, fs=fs)
    centres = track(soft, baud=sitor.BAUD, fs=fs)
    if sitor.find_phasing(soft[np.rint(centres).astype(int)] > 0, 0) is not None:
        return midpoint + offset
    window = max(1, round(fs / baud))
    if len(audio) <= 2 * window:
        return midpoint + offset
    t = np.arange(len(audio)) / fs
    a, b = [np.abs(moving_average(audio * np.exp(-2j * np.pi * hz * t), window)) ** 2
            for hz in (hi, lo)]
    a, b = a[window:-window], b[window:-window]
    on = a + b > np.max(a + b) * .15
    if (np.count_nonzero(on) < 8 or min(a[on].std(), b[on].std()) == 0
            or np.corrcoef(a[on], b[on])[0, 1] < -.8):
        return midpoint + offset
    best, quality = midpoint, -1.0
    for offset_hz in range(-8, 9):
        mid = midpoint + offset_hz
        power = tone_energy(audio, f_hi=mid + shift / 2, f_lo=mid - shift / 2,
                            baud=baud, fs=fs, units=4)
        score = float(np.mean(power))
        if score > quality:
            best, quality = mid, score
    return best + offset


def keyed_pair(audio, fs, *, f_hi, f_lo, baud=1000 / 15) -> bool:
    """Whether two tones alternate, using symbol-time envelopes for narrow FSK.

    Ignore carrier-off intervals so common silence cannot mimic alternation.
    Average over one unit at the supplied baud rate to suppress noise
    fluctuations before measuring correlation. Both
    envelopes must vary and their correlation must be negative. Each tone
    must also retain energy over two units, rejecting keyed spectral sidebands.
    Strong RTTY framing or SITOR phasing can rescue degraded correlation.
    """
    audio = np.asarray(audio)
    if len(audio) < max(1, round(fs * 0.005)):
        return False
    audio, fs, f_hi, f_lo = _pair_baseband(audio, fs, f_hi, f_lo)
    window = max(1, round(fs / baud))
    if len(audio) <= 2 * window:
        return False
    t = np.arange(len(audio)) / fs
    mixed = [audio * np.exp(-2j * np.pi * f * t) for f in (f_hi, f_lo)]
    short = [np.abs(moving_average(v, window)) ** 2 for v in mixed]
    # A keyed sideband can alternate like the carrier, but loses coherence
    # when integrated for two units. Require sustained energy at each tone.
    if any(np.mean(np.abs(moving_average(v, 2 * window)) ** 2)
           < .5 * np.mean(energy) for v, energy in zip(mixed, short)):
        return False
    energies = [energy[window:-window:window] for energy in short]
    if not len(energies[0]):
        return False
    total = energies[0] + energies[1]
    on = total > total.max() * 0.15
    a, b = (energy[on] for energy in energies)
    if len(a) < 8 or min(a.mean(), b.mean()) <= 0:
        return False
    if min(a.std() / a.mean(), b.std() / b.mean()) < 0.15:
        return False
    correlation = float(np.corrcoef(a, b)[0, 1])
    if correlation < -0.8:
        return True
    # Neighbours can weaken alternation without destroying framing. Retain
    # the coherence/variation gates and require protocol evidence on this pair.
    if correlation >= -.5:
        return False
    return framed_pair(audio, fs, f_hi=f_hi, f_lo=f_lo)


def _pair_baseband(audio, fs, f_hi, f_lo):
    """Limit protocol validation to 6 kHz while retaining both quadratures."""
    from scipy.signal import resample_poly
    from fractions import Fraction

    if fs > 6000:
        centre = (f_hi + f_lo) / 2
        audio = audio * np.exp(-2j * np.pi * centre * np.arange(len(audio)) / fs)
        f_hi, f_lo = f_hi - centre, f_lo - centre
        ratio = Fraction(6000 / fs).limit_denominator(10000)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
        fs *= ratio.numerator / ratio.denominator
    return audio, fs, f_hi, f_lo


def mark_idle(audio, fs, *, mark_hz, space_hz, power, baud=45.45) -> bool:
    """Coherent steady mark on an acquired RTTY pair preserves its lifetime.

    A block-long Hann spectrum resolves the carrier from neighbouring tones
    whose skirts can exceed the short symbol-envelope threshold. Allow the
    acquired tuning's small frequency error, and require mark to dominate space.
    Its one-unit envelope must also stay on: a keyed neighbour sharing the
    mark frequency is traffic, even when its other tone is outside this pair.
    ``power`` is this transmission's pair-local acquisition reference.
    """
    audio, fs, mark_hz, space_hz = _pair_baseband(
        np.asarray(audio), fs, mark_hz, space_hz)
    if len(audio) < round(.2 * fs):
        return False
    window = np.hanning(len(audio))
    size = max(len(audio), round(fs))
    spectrum = np.abs(np.fft.fft(audio * window, n=size) / window.sum()) ** 2
    frequencies = np.fft.fftfreq(size, 1 / fs)
    mark, space = (float(np.max(spectrum[np.abs(frequencies - hz) <= 10]))
                   for hz in (mark_hz, space_hz))
    if mark < power * .1 or mark <= 10 * space:
        return False
    unit = round(fs / baud)
    envelope = np.abs(moving_average(
        audio * np.exp(-2j * np.pi * mark_hz * np.arange(len(audio)) / fs), unit)) ** 2
    envelope = envelope[unit:-unit]
    return bool(len(envelope) and np.min(envelope) >= .5 * np.max(envelope))


def framed_pair(audio, fs, *, f_hi, f_lo, sitor_locked=False, baud=45.45) -> bool:
    """RTTY stop acceptance above chance, or SITOR phasing on this pair.

    An acquired SITOR lane can also show activity through constant-ratio words;
    ``sitor_locked`` does not relax acquisition's phasing requirement.
    """
    from scipy.special import bdtrc
    from hfmodem.lanner import rtty, sitor

    audio, fs, f_hi, f_lo = _pair_baseband(np.asarray(audio), fs, f_hi, f_lo)
    soft = discriminate(audio, f_hi=f_hi, f_lo=f_lo, baud=baud, fs=fs)
    unit = fs / baud
    # This is a statistical stop-bit test, not a decoder committing characters
    # or shifts. Its binomial null is a stop's sign, with probability one half;
    # applying the decoder's confidence gate changes that null and can hide
    # framing evidence on a keyed pair under interference.
    good, bad, _ = rtty._frames(soft, unit, 1, len(soft) - round(6.5 * unit),
                              confidence=0)
    power = tone_energy(audio, f_hi=f_hi, f_lo=f_lo, baud=baud, fs=fs, units=4)
    level = float(np.max(power)) if len(power) else 0
    # Framing evidence belongs to carrier-on frames, just as the envelope
    # correlation excludes silence. Earlier neighbours must not dilute it.
    def audible(frame):
        at = frame[0]
        return np.mean(power[at:at + round(6.5 * unit)]) >= level * .1

    good, bad = [frame for frame in good if audible(frame)], [frame for frame in bad if audible(frame)]
    n = len(good) + len(bad)
    # Under random stops, the upper binomial tail is the false-accept chance.
    if n >= 5 and len(good) / n >= .8 and bdtrc(len(good) - 1, n, .5) < .05:
        return True
    soft = discriminate(audio, f_hi=f_hi, f_lo=f_lo, baud=sitor.BAUD, fs=fs)
    if sitor_locked and len(soft):
        power = tone_energy(audio, f_hi=f_hi, f_lo=f_lo, baud=sitor.BAUD, fs=fs, units=4)
        soft = np.where(power >= np.max(power) * .01, soft, 0)
    centres = track(soft, baud=sitor.BAUD, fs=fs)
    decisions = soft[np.rint(centres).astype(int)]
    if sitor.find_phasing(decisions > 0, 0) is not None:
        return True
    return sitor_locked and sitor.word_activity(decisions)


def modulate(units, *, baud, f_hi, f_lo, fs=FS, amplitude=0.5):
    """Continuous-phase FSK; True is the higher tone, and symbol k starts at
    round(k * fs / baud) so a fractional rate such as 45.45 Bd never drifts."""
    units = np.fromiter(units, dtype=bool)
    lengths = np.diff(np.round(np.arange(units.size + 1) * fs / baud).astype(np.int64))
    freq = np.repeat(np.where(units, f_hi, f_lo), lengths)
    phase = 2 * np.pi * np.cumsum(freq, dtype=np.float64) / fs
    return (amplitude * np.sin(phase)).astype(np.float32)


def tone_energy(audio, *, f_hi, f_lo, baud, fs=FS, units=2):
    """Pair-local power through Hann windows spanning ``units`` symbols."""
    from scipy.signal import fftconvolve

    audio = np.asarray(audio, dtype=np.complex128 if np.iscomplexobj(audio) else np.float64)
    if not audio.size:
        return np.empty(0)
    t = np.arange(audio.size) / fs
    width = min(audio.size, max(3, round(units * fs / baud)))
    window = np.hanning(width) if width > 2 else np.ones(width)
    window /= window.sum()
    return sum(np.abs(fftconvolve(
        audio * np.exp(-2j * np.pi * hz * t), window, mode='same')) ** 2
        for hz in (f_hi, f_lo))


def discriminate(audio, *, f_hi, f_lo, baud, fs=FS):
    """Per-sample soft decision in [-1, 1]; positive favours the higher tone.

    Each tone is mixed to baseband and boxcar-averaged over one symbol; the
    output is the power difference normalised by the total, which cancels gain.
    """
    audio = np.asarray(audio, dtype=np.complex128 if np.iscomplexobj(audio) else np.float64)
    t = np.arange(audio.size, dtype=np.float64) / fs
    if not audio.size:
        return np.empty(0, dtype=np.float64)
    w = min(audio.size, max(1, int(round(fs / baud))))
    hi = np.abs(moving_average(audio * np.exp(-2j * np.pi * f_hi * t), w)) ** 2
    lo = np.abs(moving_average(audio * np.exp(-2j * np.pi * f_lo * t), w)) ** 2
    soft = np.zeros_like(hi)
    np.divide(hi - lo, hi + lo, out=soft, where=hi + lo > 0)
    return soft


def symbol_phase(soft, *, baud, fs=FS):
    """For a synchronous stream: the sample offset in [0, fs/baud) whose
    symbol-centre samples have the largest mean |soft|."""
    m = np.abs(np.asarray(soft, dtype=np.float64))
    if not m.size:
        return 0.0
    period = fs / baud
    k = np.arange(int((m.size - 1) / period) + 1)
    best, best_mean = 0.0, -1.0
    for s in range(int(np.ceil(period))):
        idx = np.round(s + k * period).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < m.size)]
        if not idx.size:
            continue
        mean = m[idx].mean()
        if mean > best_mean:
            best, best_mean = s, mean
    return float(best)


def track(soft, *, baud, fs=FS):
    """Estimate fractional symbol-centre positions for a clock that drifts
    slowly relative to `baud` and `fs`.

    Estimate phase between successive window starts, including the remaining
    tail in the final interval. Unwrap and smooth reliable estimates to follow
    clock drift. With no timing evidence, retain the nominal symbol grid so
    silence still occupies its elapsed symbol time. Rounded centres always
    lie within the input.
    """
    m = np.asarray(soft)
    period = fs / baud

    def bounded(centres):
        idx = np.rint(centres)
        return centres[(idx >= 0) & (idx < m.size)]

    if not m.size:
        return np.empty(0, dtype=np.float64)
    w, step = round(_TRACK_WINDOW_S * fs), round(_TRACK_STEP_S * fs)
    starts = np.arange(0, m.size - w + 1, step, dtype=np.int64)
    if starts.size == 0:
        phase = symbol_phase(soft, baud=baud, fs=fs)
        return bounded(phase + np.arange(int((m.size - 1) / period) + 1, dtype=np.float64) * period)

    nph = int(np.ceil(period))
    nwin = starts.size
    kmax = int((m.size - 1) / period) + 1
    best = np.empty(nwin, dtype=np.int64)
    quality = np.empty(nwin, dtype=np.float64)
    phases = np.arange(nph)[:, None]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < nwin else m.size
        k0 = max(0, int(np.floor((start - nph) / period)))
        k1 = min(kmax, int(np.ceil(end / period)) + 1)
        pos = np.round(phases + np.arange(k0, k1)[None, :] * period).astype(np.int64)
        keep = (pos >= start) & (pos < end)
        keys = np.broadcast_to(phases, pos.shape)[keep]
        values = np.abs(np.asarray(m[pos[keep]], dtype=np.float64))
        sums = np.bincount(keys, weights=values, minlength=nph)
        counts = np.bincount(keys, minlength=nph)
        mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        best[i] = mean.argmax()
        quality[i] = mean[best[i]]

    good = (quality > 0) & (quality >= 0.5 * np.median(quality))
    if not good.any():
        return bounded(np.arange(kmax, dtype=np.float64) * period)
    pg = np.unwrap(best[good].astype(np.float64), period=period)
    if pg.size < 2:
        return bounded(pg[0] + np.arange(kmax, dtype=np.float64) * period)
    tg = (starts[good] + w // 2) / fs
    half = _TRACK_SPAN_S / 2
    q = quality[good]
    cs = [np.concatenate(([0.0], np.cumsum(v))) for v in
          (q, q * tg, q * pg, q * tg * tg, q * tg * pg)]
    lo = np.searchsorted(tg, tg - half)
    hi = np.searchsorted(tg, tg + half, side="right")
    n = hi - lo
    W, X, Y, XX, XY = (c[hi] - c[lo] for c in cs)
    den = W * XX - X * X
    slope = np.where(n >= 2, (W * XY - X * Y) / np.where(den != 0, den, 1.0), 0.0)
    intercept = (Y - slope * X) / W
    smooth = np.where(n >= 2, intercept + slope * tg, pg)

    k = np.arange(kmax)
    centres = k * period + np.interp(k * period / fs, tg, smooth)
    return bounded(centres)


def moving_average(x, w):
    """Centred ``w``-sample mean, with the input's length.

    ``x`` is a one-dimensional real or complex array and ``1 <= w <= x.size``.
    Even windows use the earlier centre; edges repeat the nearest full mean.
    """
    c = np.concatenate(([0], np.cumsum(x)))
    y = (c[w:] - c[:-w]) / w
    h = (w - 1) // 2
    out = np.empty(x.size, dtype=y.dtype)
    out[h:h + y.size] = y
    out[:h] = y[0]
    out[h + y.size:] = y[-1]
    return out
