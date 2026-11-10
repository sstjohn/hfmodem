#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""
audio-nulltest -- prove an audio loopback is bit-transparent.

Before we trust any audio-loopback setup to capture a real-time OFDM waveform,
we must prove the loopback path is *bit-transparent*: no sample-rate conversion
(SRC), no dropped/inserted samples, no clock drift, no mixer/volume alteration.
A hidden resampler corrupts captures and is otherwise indistinguishable from a
real property of the signal under study.

This tool implements a null-test protocol:

  1. GENERATE a deterministic reference WAV (seeded full-scale PRN noise, or a
     band-edge sine) at a chosen format (default 48 kHz / 32-bit float / mono).
  2. PLAY it out one device and CAPTURE from another, simultaneously, at the
     identical format, via PortAudio (sounddevice).
  3. ANALYZE: cross-correlate capture vs reference, find the integer lag, trim,
     subtract, and CLASSIFY:
        bit-exact                -> residual identically zero (+ hash match)
        SRC / resampling         -> static nonzero residual, spectral imaging,
                                     broadened/fractional correlation peak
        dropped/inserted samples -> lag track STEPS partway through
        clock drift / async SRC  -> lag track RAMPS ~linearly over the capture
     Print a PASS/FAIL verdict + the failure mode, and save diagnostic plots.
  4. SINE mode: full-scale tone near the band edge (~2.9 kHz); report residual
     noise floor (bit-exact -> -inf; SRC/dither raises it).

The analysis functions (generate_reference, classify, etc.) depend only on
numpy + scipy and are unit-testable with synthetic inputs. Audio I/O
(sounddevice) and plotting (matplotlib) are imported lazily so the analysis and
its tests run on a machine without them.

Author: kestrel project, tools/audio-nulltest
"""

from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

# Map our subtype names to (numpy dtype, bytes, wav sampwidth or None for float)
SUBTYPES = {
    "float32": ("float32", 4),
    "int24": ("int24", 3),
    "int16": ("int16", 2),
}


@dataclass
class AudioFormat:
    samplerate: int = 48000
    channels: int = 1
    subtype: str = "float32"  # one of SUBTYPES

    @property
    def dtype(self) -> str:
        return "float32" if self.subtype == "float32" else "int32"  # int24 carried in int32

    def describe(self) -> str:
        return f"{self.samplerate} Hz / {self.subtype} / {self.channels} ch"


# ---------------------------------------------------------------------------
# Reference signal generation
# ---------------------------------------------------------------------------

def generate_prn(n_samples: int, seed: int = 0xC0FFEE, amplitude: float = 0.98) -> np.ndarray:
    """Deterministic full-scale pseudo-random reference, float32 in [-amp, amp].

    Full-scale *random* samples make any interpolation visible: an interpolated
    sample is a weighted average of neighbours, which for white noise is almost
    never equal to the true sample, so SRC produces a large, broadband residual.

    We use a uniform full-scale distribution (not Gaussian) so the sequence
    actually reaches near +/- full scale often; `amplitude` < 1.0 leaves a hair
    of headroom so honest unity-gain paths don't clip on the DAC/ADC.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)
    return (x * np.float32(amplitude)).astype(np.float32)


def generate_sine(n_samples: int, samplerate: int, freq: float = 2900.0,
                  amplitude: float = 0.98) -> np.ndarray:
    """Full-scale sine near the band edge for the noise-floor test.

    A near-band-edge tone is the hardest case for a cheap resampler: its images
    fold close to the passband edge and its steep phase makes interpolation
    error large, so SRC shows up strongly as a raised residual noise floor.
    """
    t = np.arange(n_samples, dtype=np.float64) / float(samplerate)
    x = amplitude * np.sin(2.0 * np.pi * freq * t)
    return x.astype(np.float32)


def generate_reference(mode: str, duration_s: float, fmt: AudioFormat,
                       seed: int = 0xC0FFEE, freq: float = 2900.0,
                       amplitude: float = 0.98) -> np.ndarray:
    """Return a mono float32 reference of the requested mode/duration."""
    n = int(round(duration_s * fmt.samplerate))
    if mode == "prn":
        return generate_prn(n, seed=seed, amplitude=amplitude)
    if mode == "sine":
        return generate_sine(n, fmt.samplerate, freq=freq, amplitude=amplitude)
    raise ValueError(f"unknown reference mode {mode!r}")


# ---------------------------------------------------------------------------
# WAV I/O (float32 or int PCM), stdlib wave + numpy; no soundfile dependency
# ---------------------------------------------------------------------------

def _float_to_pcm(x: np.ndarray, subtype: str) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    if subtype == "float32":
        return x.astype("<f4").tobytes()
    if subtype == "int16":
        return (x * 32767.0).round().astype("<i2").tobytes()
    if subtype == "int24":
        ints = (x * 8388607.0).round().astype(np.int64)
        b = np.empty((ints.size, 3), dtype=np.uint8)
        u = (ints & 0xFFFFFF).astype(np.uint32)
        b[:, 0] = u & 0xFF
        b[:, 1] = (u >> 8) & 0xFF
        b[:, 2] = (u >> 16) & 0xFF
        return b.tobytes()
    raise ValueError(subtype)


def write_wav(path: str, x: np.ndarray, fmt: AudioFormat) -> None:
    """Write mono/interleaved float or PCM WAV. `x` is float32 in [-1, 1]."""
    if x.ndim == 1:
        interleaved = x
    else:
        interleaved = x.reshape(-1)
    if fmt.subtype == "float32":
        # stdlib wave cannot write IEEE float; write a WAVE_FORMAT_IEEE_FLOAT
        # container by hand via a raw fallback.
        _write_wav_float(path, x, fmt)
        return
    with wave.open(path, "wb") as w:
        w.setnchannels(fmt.channels)
        w.setsampwidth(SUBTYPES[fmt.subtype][1])
        w.setframerate(fmt.samplerate)
        w.writeframes(_float_to_pcm(interleaved, fmt.subtype))


def _write_wav_float(path: str, x: np.ndarray, fmt: AudioFormat) -> None:
    import struct
    data = _float_to_pcm(x.reshape(-1), "float32")
    ch = fmt.channels
    sr = fmt.samplerate
    byte_rate = sr * ch * 4
    block_align = ch * 4
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 3))          # WAVE_FORMAT_IEEE_FLOAT
        f.write(struct.pack("<H", ch))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", 32))         # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def read_wav(path: str) -> tuple[np.ndarray, AudioFormat]:
    """Read a WAV written by write_wav; returns (float32 mono/interleaved, fmt)."""
    import struct
    with open(path, "rb") as f:
        riff = f.read(12)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        audio_fmt = None
        nch = sr = bits = 0
        data = b""
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, size = struct.unpack("<4sI", hdr)
            body = f.read(size)
            if size % 2 == 1:
                f.read(1)  # padding byte
            if cid == b"fmt ":
                audio_fmt, nch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
            elif cid == b"data":
                data = body
        if audio_fmt == 3 and bits == 32:
            x = np.frombuffer(data, dtype="<f4").astype(np.float32)
            sub = "float32"
        elif audio_fmt == 1 and bits == 16:
            x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            sub = "int16"
        elif audio_fmt == 1 and bits == 24:
            raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float32) / 8388608.0
            sub = "int24"
        else:
            raise ValueError(f"unsupported wav format {audio_fmt}/{bits}")
        fmt = AudioFormat(samplerate=sr, channels=nch, subtype=sub)
        if nch > 1:
            x = x.reshape(-1, nch)
        return x, fmt


# ---------------------------------------------------------------------------
# Correlation / lag estimation
# ---------------------------------------------------------------------------

def _to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def fft_xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int | None = None) -> tuple[int, np.ndarray]:
    """Integer lag that best aligns `a` to `b` via FFT cross-correlation.

    Returns (lag, corr) where a[n] ~ b[n - lag]; i.e. positive lag means `a` is
    delayed relative to `b`. `corr` is the full cross-correlation for optional
    peak-shape inspection.
    """
    a = _to_mono(a)
    b = _to_mono(b)
    a = a - a.mean()
    b = b - b.mean()
    n = len(a) + len(b) - 1
    nfft = 1 << (int(n - 1).bit_length())
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    corr = np.fft.irfft(fa * np.conj(fb), nfft)
    # full-length cross-correlation ordering
    corr = np.concatenate((corr[-(len(b) - 1):], corr[:len(a)]))
    lags = np.arange(-(len(b) - 1), len(a))
    if max_lag is not None:
        keep = np.abs(lags) <= max_lag
        corr = corr[keep]
        lags = lags[keep]
    peak = int(np.argmax(corr))
    return int(lags[peak]), corr


def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak location refinement around integer index i."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    ym1, y0, yp1 = y[i - 1], y[i], y[i + 1]
    denom = (ym1 - 2 * y0 + yp1)
    if denom == 0:
        return float(i)
    return i + 0.5 * (ym1 - yp1) / denom


def local_lag(cap_block: np.ndarray, ref_window: np.ndarray, center_offset: int,
              max_shift: int) -> tuple[float, float]:
    """Estimate fractional lag of cap_block within ref_window.

    ref_window is a slice of the reference starting `center_offset` before the
    nominal aligned position. Returns (fractional_lag_deviation, peak_corr_norm)
    where the deviation is relative to perfect alignment (0.0 == aligned).
    """
    a = _to_mono(cap_block)
    b = _to_mono(ref_window)
    a = a - a.mean()
    b = b - b.mean()
    # FFT-based valid correlation: O((B+2S) log) instead of O(B*S). Essential
    # for multi-minute captures where blocks are hundreds of thousands of samples.
    try:
        from scipy.signal import correlate
        corr = correlate(b, a, mode="valid", method="auto")
    except Exception:
        corr = np.correlate(b, a, mode="valid")
    if len(corr) == 0:
        return 0.0, 0.0
    i = int(np.argmax(corr))
    frac = _parabolic_peak(corr, i)
    # offset i corresponds to shift (i - center_offset)
    dev = frac - center_offset
    denom = (np.linalg.norm(a) * np.linalg.norm(b[i:i + len(a)]) + 1e-20)
    peak_norm = float(corr[i]) / denom
    return float(dev), peak_norm


@dataclass
class LagTrack:
    times: np.ndarray          # block center time in samples (reference frame)
    lags: np.ndarray           # fractional lag deviation per block
    peaks: np.ndarray          # normalized correlation peak per block
    slope: float               # samples of lag per sample of time (dimensionless)
    intercept: float
    r2: float                  # goodness of linear fit
    max_step: float            # largest single-block jump in lag
    step_index: int            # block index of that jump


def estimate_lag_track(capture: np.ndarray, reference: np.ndarray, global_lag: int,
                       n_blocks: int = 40, max_shift: int = 4000) -> LagTrack:
    """Track how the required lag varies across the capture.

    After removing the constant `global_lag`, we chop the overlap into blocks and
    measure each block's residual lag deviation. A flat track -> static; a linear
    ramp -> clock drift / async SRC; a single step -> dropped/inserted samples.
    """
    cap = _to_mono(capture)
    ref = _to_mono(reference)
    # Align: cap[i] corresponds to ref[i - global_lag].
    if global_lag >= 0:
        cap_a = cap[global_lag:]
        ref_a = ref
    else:
        cap_a = cap
        ref_a = ref[-global_lag:]
    n = min(len(cap_a), len(ref_a))
    cap_a = cap_a[:n]
    ref_a = ref_a[:n]

    block = max(1024, (n - 2 * max_shift) // max(1, n_blocks))
    times, lags, peaks = [], [], []
    pos = max_shift
    while pos + block + max_shift <= n:
        cap_block = cap_a[pos:pos + block]
        ref_window = ref_a[pos - max_shift:pos + block + max_shift]
        dev, peak = local_lag(cap_block, ref_window, max_shift, max_shift)
        times.append(pos + block / 2)
        lags.append(dev)
        peaks.append(peak)
        pos += block
    times = np.asarray(times, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.float64)

    slope = intercept = r2 = 0.0
    max_step = 0.0
    step_index = -1
    if len(times) >= 2:
        A = np.vstack([times, np.ones_like(times)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, lags, rcond=None)
        fit = slope * times + intercept
        ss_res = float(np.sum((lags - fit) ** 2))
        ss_tot = float(np.sum((lags - lags.mean()) ** 2)) + 1e-20
        r2 = 1.0 - ss_res / ss_tot
        diffs = np.abs(np.diff(lags))
        step_index = int(np.argmax(diffs))
        max_step = float(diffs[step_index])
    return LagTrack(times, lags, peaks, float(slope), float(intercept),
                    float(r2), max_step, step_index)


# ---------------------------------------------------------------------------
# Residual + spectral analysis
# ---------------------------------------------------------------------------

def aligned_pair(capture: np.ndarray, reference: np.ndarray, global_lag: int,
                 guard: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (cap_aligned, ref_aligned) trimmed to the common region."""
    cap = _to_mono(capture)
    ref = _to_mono(reference)
    if global_lag >= 0:
        cap_a = cap[global_lag:]
        ref_a = ref
    else:
        cap_a = cap
        ref_a = ref[-global_lag:]
    n = min(len(cap_a), len(ref_a))
    cap_a = cap_a[guard:n - guard]
    ref_a = ref_a[guard:n - guard]
    return cap_a, ref_a


def residual_db(cap_a: np.ndarray, ref_a: np.ndarray) -> float:
    """Residual RMS relative to reference RMS, in dB. -inf if identically zero."""
    # Best-fit scalar gain removes honest unity/volume scaling from the residual
    # question -- but for a *bit-transparent* test we do NOT want to hide a
    # volume change, so we report the raw residual. Gain is reported separately.
    resid = cap_a - ref_a
    rms_r = np.sqrt(np.mean(ref_a ** 2)) + 1e-30
    rms_e = np.sqrt(np.mean(resid ** 2))
    if rms_e == 0.0:
        return float("-inf")
    return 20.0 * np.log10(rms_e / rms_r)


def best_fit_gain(cap_a: np.ndarray, ref_a: np.ndarray) -> float:
    denom = float(np.dot(ref_a, ref_a)) + 1e-30
    return float(np.dot(cap_a, ref_a) / denom)


def residual_after_gain_db(cap_a: np.ndarray, ref_a: np.ndarray) -> float:
    g = best_fit_gain(cap_a, ref_a)
    resid = cap_a - g * ref_a
    rms_r = np.sqrt(np.mean(ref_a ** 2)) + 1e-30
    rms_e = np.sqrt(np.mean(resid ** 2))
    if rms_e == 0.0:
        return float("-inf")
    return 20.0 * np.log10(rms_e / rms_r)


def spectral_tilt(resid: np.ndarray, samplerate: int) -> float:
    """Slope of the residual power spectrum (dB per Nyquist fraction).

    A whitening SRC/imaging residual is not spectrally flat: interpolation error
    is high-frequency weighted (a positive tilt), so a strong non-zero tilt is
    evidence of resampling rather than flat additive noise/dither.
    """
    r = resid - resid.mean()
    if np.allclose(r, 0):
        return 0.0
    f, pxx = _welch(r, samplerate)
    mask = (f > 0) & (pxx > 0)
    if mask.sum() < 8:
        return 0.0
    xf = f[mask] / (samplerate / 2.0)
    yl = 10.0 * np.log10(pxx[mask])
    A = np.vstack([xf, np.ones_like(xf)]).T
    (slope, _), *_ = np.linalg.lstsq(A, yl, rcond=None)
    return float(slope)


def _welch(x: np.ndarray, fs: int, nperseg: int = 4096):
    try:
        from scipy.signal import welch
        return welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    except Exception:
        n = min(nperseg, len(x))
        w = np.hanning(n)
        seg = x[:n] * w
        f = np.fft.rfftfreq(n, 1.0 / fs)
        pxx = (np.abs(np.fft.rfft(seg)) ** 2)
        return f, pxx


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    passed: bool
    mode: str                 # bit-exact | src | dropped | drift | inconclusive
    detail: str
    residual_db: float
    residual_after_gain_db: float
    gain: float
    global_lag: int
    lag_slope_ppm: float      # linear ramp in parts-per-million
    lag_r2: float
    max_lag_step: float
    tilt_db_per_nyq: float
    bit_exact_hash_match: bool
    track: LagTrack | None = None
    noise_floor_db: float | None = None  # sine mode only

    def summary(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        lines = [f"VERDICT: {tag}  ({self.mode})", f"  {self.detail}"]
        lines.append(f"  integer lag           : {self.global_lag} samples")
        lines.append(f"  residual (raw)        : {self._fmt(self.residual_db)} dB")
        lines.append(f"  residual (gain-fit)   : {self._fmt(self.residual_after_gain_db)} dB")
        lines.append(f"  best-fit gain         : {self.gain:.6f} "
                     f"({20*np.log10(abs(self.gain)+1e-30):+.3f} dB)")
        lines.append(f"  lag ramp              : {self.lag_slope_ppm:+.2f} ppm "
                     f"(fit R^2={self.lag_r2:.3f})")
        lines.append(f"  max single-block step : {self.max_lag_step:.2f} samples")
        lines.append(f"  residual spectral tilt: {self.tilt_db_per_nyq:+.1f} dB/Nyquist")
        lines.append(f"  bit-exact hash match  : {self.bit_exact_hash_match}")
        if self.noise_floor_db is not None:
            lines.append(f"  sine residual floor   : {self._fmt(self.noise_floor_db)} dBFS")
        return "\n".join(lines)

    @staticmethod
    def _fmt(v: float) -> str:
        return "-inf" if v == float("-inf") else f"{v:.1f}"


# Decision thresholds (documented; tune per environment if needed).
BITEXACT_RESIDUAL_DB = -120.0   # below this and hash-match -> treat as bit-exact
FAIL_RESIDUAL_DB = -60.0        # above this residual -> definitely altered
DRIFT_PPM_THRESH = 2.0          # linear ramp steeper than this -> drift/async SRC
DRIFT_R2_THRESH = 0.80          # ramp must be a good linear fit
STEP_THRESH_SAMPLES = 2.0       # a lag jump bigger than this in one block -> drop
STEP_DOMINANCE = 4.0            # jump must dominate the rest of the track


def classify(capture: np.ndarray, reference: np.ndarray, samplerate: int,
             mode: str = "prn", freq: float = 2900.0,
             n_blocks: int = 40, max_shift: int = 4000,
             search_lag: int | None = None) -> Verdict:
    """Classify a capture against its reference. Pure numpy/scipy; unit-testable."""
    cap = _to_mono(capture)
    ref = _to_mono(reference)

    # 1) integer lag on a strong central segment (cheaper + robust)
    seg = min(len(cap), len(ref))
    lo = max(0, seg // 2 - 200000)
    hi = min(seg, seg // 2 + 200000)
    ml = search_lag if search_lag is not None else max(4000, seg // 4)
    global_lag, corr = fft_xcorr_lag(cap[lo:hi], ref[lo:hi], max_lag=ml)

    # 2) aligned residual
    cap_a, ref_a = aligned_pair(cap, ref, global_lag)
    if len(cap_a) < 16:
        return Verdict(False, "inconclusive", "no usable overlap after alignment",
                       0.0, 0.0, 0.0, global_lag, 0.0, 0.0, 0.0, 0.0, False)
    res_db = residual_db(cap_a, ref_a)
    res_g_db = residual_after_gain_db(cap_a, ref_a)
    gain = best_fit_gain(cap_a, ref_a)
    resid = cap_a - ref_a
    tilt = spectral_tilt(resid, samplerate)

    # 3) hash / byte compare of aligned payloads (quantize to catch true bit-exact)
    q = np.round(cap_a * 8388608.0).astype(np.int64)
    r = np.round(ref_a * 8388608.0).astype(np.int64)
    hash_match = bool(np.array_equal(q, r))
    exact_zero = bool(np.array_equal(cap_a.astype(np.float32), ref_a.astype(np.float32)))

    # 4) lag track for drift / drop discrimination
    track = estimate_lag_track(cap, ref, global_lag, n_blocks=n_blocks, max_shift=max_shift)
    slope_ppm = track.slope * 1e6  # lag samples per sample -> ppm
    # step dominance: is the biggest jump much bigger than the typical jump?
    if len(track.lags) > 3:
        diffs = np.abs(np.diff(track.lags))
        others = np.delete(diffs, track.step_index)
        med = float(np.median(others)) + 1e-9
        step_dominant = (track.max_step > STEP_THRESH_SAMPLES and
                         track.max_step > STEP_DOMINANCE * med)
    else:
        step_dominant = False

    # 5) noise floor for sine mode
    noise_floor = None
    if mode == "sine":
        noise_floor = _sine_noise_floor(cap_a, ref_a, samplerate, freq)

    # ---- Decision tree ----
    if (exact_zero or (hash_match and res_db <= BITEXACT_RESIDUAL_DB)):
        return Verdict(True, "bit-exact",
                       "Residual is identically zero after integer alignment; "
                       "aligned payloads hash-match. Loopback is bit-transparent.",
                       res_db, res_g_db, gain, global_lag, slope_ppm, track.r2,
                       track.max_step, tilt, hash_match, track, noise_floor)

    # Not bit-exact -> something altered the samples. Determine which.
    detail_bits = []

    # Drop/insert: a dominant single step in the lag track. A dominant single
    # jump only happens for a discontinuity: genuine drift produces *uniform*
    # per-block increments (max_step ~ median), so STEP_DOMINANCE separates them
    # regardless of the linear-fit R^2 that a step function coincidentally scores.
    if step_dominant:
        detail_bits.append(
            f"lag track jumps by {track.max_step:.1f} samples at block "
            f"{track.step_index}/{len(track.lags)} -> samples dropped or inserted.")
        return Verdict(False, "dropped",
                       " ".join(detail_bits), res_db, res_g_db, gain, global_lag,
                       slope_ppm, track.r2, track.max_step, tilt, hash_match,
                       track, noise_floor)

    # Clock drift / async SRC: lag ramps ~linearly across the capture.
    if abs(slope_ppm) > DRIFT_PPM_THRESH and track.r2 > DRIFT_R2_THRESH:
        detail_bits.append(
            f"required lag ramps linearly at {slope_ppm:+.1f} ppm (R^2={track.r2:.2f}) "
            "-> clock drift / asynchronous SRC between play and capture clocks.")
        return Verdict(False, "drift",
                       " ".join(detail_bits), res_db, res_g_db, gain, global_lag,
                       slope_ppm, track.r2, track.max_step, tilt, hash_match,
                       track, noise_floor)

    # Static nonzero residual with (usually) spectral tilt/imaging -> SRC.
    if res_g_db > FAIL_RESIDUAL_DB or res_db > FAIL_RESIDUAL_DB:
        detail_bits.append(
            f"static nonzero residual ({Verdict._fmt(res_db)} dB) with constant lag; "
            f"spectral tilt {tilt:+.1f} dB/Nyquist (interpolation imaging) "
            "-> sample-rate conversion / hidden resampler in the path.")
        return Verdict(False, "src",
                       " ".join(detail_bits), res_db, res_g_db, gain, global_lag,
                       slope_ppm, track.r2, track.max_step, tilt, hash_match,
                       track, noise_floor)

    # Small but nonzero residual, no clear structure: pure gain change, or dither.
    if abs(20 * np.log10(abs(gain) + 1e-30)) > 0.05 and res_g_db <= FAIL_RESIDUAL_DB:
        detail_bits.append(
            f"residual is explained by a level change of "
            f"{20*np.log10(abs(gain)+1e-30):+.3f} dB (mixer/volume not at unity).")
        return Verdict(False, "src", " ".join(detail_bits) +
                       " Not bit-transparent: fix gain to 100%/unity.",
                       res_db, res_g_db, gain, global_lag, slope_ppm, track.r2,
                       track.max_step, tilt, hash_match, track, noise_floor)

    detail_bits.append(
        f"residual {Verdict._fmt(res_db)} dB is nonzero but low-level and "
        "unstructured (likely dither/quantization); not bit-exact.")
    return Verdict(False, "src", " ".join(detail_bits), res_db, res_g_db, gain,
                   global_lag, slope_ppm, track.r2, track.max_step, tilt,
                   hash_match, track, noise_floor)


def _sine_noise_floor(cap_a: np.ndarray, ref_a: np.ndarray, fs: int, freq: float) -> float:
    """Residual (everything that is NOT the fundamental) in dBFS."""
    g = best_fit_gain(cap_a, ref_a)
    resid = cap_a - g * ref_a
    rms_e = np.sqrt(np.mean(resid ** 2))
    if rms_e == 0.0:
        return float("-inf")
    return 20.0 * np.log10(rms_e)  # relative to full scale (1.0)


# ---------------------------------------------------------------------------
# Diagnostic plots (matplotlib, lazy)
# ---------------------------------------------------------------------------

def save_plots(verdict: Verdict, capture: np.ndarray, reference: np.ndarray,
               samplerate: int, out_prefix: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plots skipped: matplotlib unavailable: {e}]", file=sys.stderr)
        return []
    cap_a, ref_a = aligned_pair(capture, reference, verdict.global_lag)
    resid = cap_a - ref_a
    paths = []

    # residual waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(len(resid)) / samplerate
    ax.plot(t, resid, lw=0.4)
    ax.set(title=f"Residual waveform ({verdict.mode})", xlabel="s", ylabel="amplitude")
    p = f"{out_prefix}_residual_wave.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
    plt.close(fig); paths.append(p)

    # residual spectrum
    f, pxx = _welch(resid - resid.mean(), samplerate)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(f, pxx + 1e-30, lw=0.7)
    ax.set(title="Residual spectrum", xlabel="Hz", ylabel="power")
    p = f"{out_prefix}_residual_spectrum.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
    plt.close(fig); paths.append(p)

    # lag vs time
    if verdict.track is not None and len(verdict.track.times):
        fig, ax = plt.subplots(figsize=(10, 3))
        tt = verdict.track.times / samplerate
        ax.plot(tt, verdict.track.lags, "o-", ms=3)
        ax.set(title=f"Required lag vs time (slope {verdict.lag_slope_ppm:+.1f} ppm)",
               xlabel="s", ylabel="lag deviation (samples)")
        p = f"{out_prefix}_lag_vs_time.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
        plt.close(fig); paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Audio I/O (sounddevice, lazy)
# ---------------------------------------------------------------------------

def list_devices() -> str:
    import sounddevice as sd
    return str(sd.query_devices())


def _sd_dtype(fmt: AudioFormat) -> str:
    return {"float32": "float32", "int16": "int16", "int24": "int24"}[fmt.subtype]


def play_and_capture(reference: np.ndarray, fmt: AudioFormat,
                     out_device, in_device, extra_tail_s: float = 1.0) -> np.ndarray:
    """Play `reference` out `out_device` while recording `in_device`.

    Returns the captured mono float32 array (longer than the reference by the
    tail). Requires sounddevice/PortAudio and a real loopback; not exercised in
    the sandbox.
    """
    import threading

    import sounddevice as sd

    sd.default.samplerate = fmt.samplerate
    play = np.asarray(reference, dtype=np.float32).reshape(-1, 1)
    tail = int(extra_tail_s * fmt.samplerate)
    n_capture = len(play) + tail
    captured = np.zeros((n_capture, 1), dtype=np.float32)

    rec_state = {"pos": 0}
    play_state = {"pos": 0}
    done = threading.Event()

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        # capture
        rp = rec_state["pos"]
        take = min(frames, n_capture - rp)
        captured[rp:rp + take, 0] = indata[:take, 0]
        rec_state["pos"] = rp + take
        # playback
        pp = play_state["pos"]
        give = min(frames, len(play) - pp)
        if give > 0:
            outdata[:give, 0] = play[pp:pp + give, 0]
        if give < frames:
            outdata[give:, 0] = 0.0
        play_state["pos"] = pp + give
        if rec_state["pos"] >= n_capture:
            done.set()
            raise sd.CallbackStop()

    with sd.Stream(samplerate=fmt.samplerate, channels=1,
                   dtype=_sd_dtype(fmt), device=(in_device, out_device),
                   callback=callback):
        done.wait(timeout=(n_capture / fmt.samplerate) + 5.0)
    return captured[:, 0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_device(v: str | None):
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def cmd_devices(args):
    print(list_devices())


def cmd_gen(args):
    fmt = AudioFormat(args.rate, 1, args.subtype)
    ref = generate_reference(args.mode, args.duration, fmt, seed=args.seed, freq=args.freq)
    write_wav(args.out, ref, fmt)
    print(f"wrote {args.out}: {fmt.describe()}, {len(ref)} samples "
          f"({len(ref)/fmt.samplerate:.1f} s), mode={args.mode}")


def cmd_run(args):
    fmt = AudioFormat(args.rate, 1, args.subtype)
    ref = generate_reference(args.mode, args.duration, fmt, seed=args.seed, freq=args.freq)
    print(f"reference: {fmt.describe()}, {args.duration:.0f}s, mode={args.mode}")
    print("playing + capturing (do not touch audio settings)...")
    cap = play_and_capture(ref, fmt,
                           out_device=_parse_device(args.out_device),
                           in_device=_parse_device(args.in_device))
    if args.save_capture:
        write_wav(args.save_capture, cap, fmt)
        print(f"saved capture -> {args.save_capture}")
    v = classify(cap, ref, fmt.samplerate, mode=args.mode, freq=args.freq)
    print(v.summary())
    if args.plots:
        paths = save_plots(v, cap, ref, fmt.samplerate, args.plots)
        for p in paths:
            print(f"  plot -> {p}")
    sys.exit(0 if v.passed else 1)


def cmd_analyze(args):
    cap, cfmt = read_wav(args.capture)
    ref, rfmt = read_wav(args.reference)
    fs = rfmt.samplerate
    v = classify(cap, ref, fs, mode=args.mode, freq=args.freq)
    print(v.summary())
    if args.plots:
        for p in save_plots(v, cap, ref, fs, args.plots):
            print(f"  plot -> {p}")
    sys.exit(0 if v.passed else 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nulltest",
        description="Prove an audio loopback is bit-transparent (null test).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("devices", help="list audio devices")
    pd.set_defaults(func=cmd_devices)

    def add_fmt(sp):
        sp.add_argument("--rate", type=int, default=48000)
        sp.add_argument("--subtype", choices=list(SUBTYPES), default="float32")
        sp.add_argument("--mode", choices=["prn", "sine"], default="prn")
        sp.add_argument("--duration", type=float, default=180.0,
                        help="seconds (default 180 = 3 min)")
        sp.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FFEE)
        sp.add_argument("--freq", type=float, default=2900.0, help="sine mode Hz")

    pg = sub.add_parser("gen", help="write a reference WAV")
    add_fmt(pg)
    pg.add_argument("--out", required=True)
    pg.set_defaults(func=cmd_gen)

    pr = sub.add_parser("run", help="play+capture through a loopback and classify")
    add_fmt(pr)
    pr.add_argument("--out-device", required=True, help="output device index or name")
    pr.add_argument("--in-device", required=True, help="input device index or name")
    pr.add_argument("--save-capture", help="also write the raw capture WAV here")
    pr.add_argument("--plots", help="prefix path for diagnostic PNGs")
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("analyze", help="classify an already-captured WAV pair")
    pa.add_argument("--reference", required=True)
    pa.add_argument("--capture", required=True)
    pa.add_argument("--mode", choices=["prn", "sine"], default="prn")
    pa.add_argument("--freq", type=float, default=2900.0)
    pa.add_argument("--plots")
    pa.set_defaults(func=cmd_analyze)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
