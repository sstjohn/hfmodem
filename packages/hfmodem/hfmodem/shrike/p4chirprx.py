# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Locate a predicted P4 SL1 header and decode an explicitly placed chirp.

This finite receive operation has no device, ARQ or mail side effects. The
caller supplies the carrier and either the waveform origin or its bounded
prediction. Both observed chirp marker words and lane orders are tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from . import p4chirp, p4frame, pactor2


@dataclass(frozen=True)
class Reception:
    frame: p4frame.Frame
    marker: int
    rank0_upper: bool
    header_score: float


@dataclass(frozen=True)
class Header:
    start_sample: int
    marker: int
    score: float


_HEADER_S = .200
_HEADER_PULSE = resample_poly(p4chirp.QUASI_RRC, 90, 1)


def acquire_header(audio, *, fs: int = 48000, nominal_sample: int,
                   frequency_offset: float = 0., slack_s: float = .035) -> Header | None:
    """Locate one marker in the predicted bracket, using only its first 200 ms."""
    raw = np.asarray(audio)
    if raw.ndim != 1 or np.iscomplexobj(raw) or not np.all(np.isfinite(raw)):
        raise ValueError("audio must be a finite real vector")
    if not isinstance(fs, int) or fs < 8000:
        raise ValueError("sample rate must be an integer of at least 8000 Hz")
    if not isinstance(nominal_sample, int) or nominal_sample < 0:
        raise ValueError("nominal_sample must be a nonnegative integer")
    if not np.isfinite(frequency_offset) or not 0 < slack_s <= .035:
        raise ValueError("finite carrier and timing slack in (0, .035] required")
    lo = max(0, nominal_sample-round(slack_s*fs))
    hi = nominal_sample+round(slack_s*fs)
    end = hi+round(_HEADER_S*fs)
    if len(raw) < end:
        return None
    x = np.asarray(raw[lo:end], dtype=float)
    peak = np.max(np.abs(x))
    if peak == 0:
        return None
    x = x/peak
    if fs != 48000:
        common = gcd(fs, 48000)
        x = resample_poly(x, 48000//common, fs//common)
    starts = np.arange(round((hi-lo)*48000/fs)+1)
    t = np.arange(len(x))/48000
    differentials = []
    for tone, delay in ((1530., 0), (550., 4)):
        matched = fftconvolve(x*np.exp(-2j*np.pi*(tone+frequency_offset)*t), _HEADER_PULSE)
        indices = starts[:, None]+90*(31+delay+8*np.arange(9))
        symbols = matched[indices]
        differentials.append(symbols[:, 1:]*symbols[:, :-1].conjugate())
    best = None
    for marker in (8, 9):
        correlations = []
        for heard, reference in zip(differentials, pactor2.marker_steps(marker)):
            correlations.append(abs(heard@np.exp(-1j*reference)) /
                                np.maximum(abs(heard).sum(axis=1), 1e-30))
        scores = np.minimum(*correlations)
        at = int(scores.argmax())
        if scores[at] >= .85 and (best is None or scores[at] > best.score):
            best = Header(lo+round(int(starts[at])*fs/48000), marker, float(scores[at]))
    return best


def decode_packet(audio, *, fs: int = 48000, start_sample: int = 0,
                  frequency_offset: float = 0.0) -> Reception | None:
    """Return only a uniquely decoded field; short/noisy/mismatched input is None.

    `start_sample` denotes the first sample of the shaped waveform, not its
    pulse peak. Frequency offset is in Hz relative to 550/1530 Hz. Input must
    contain the complete 3307.5 ms packet. Header correlation is an acquisition
    check, not a replacement for CRC or proof of a connected session.
    """
    raw = np.asarray(audio)
    if (raw.ndim != 1 or np.iscomplexobj(raw)
            or not np.all(np.isfinite(raw))):
        raise ValueError("audio must be a finite real vector")
    if not isinstance(fs, int) or fs < 8000:
        raise ValueError("sample rate must be an integer of at least 8000 Hz")
    if not isinstance(start_sample, int) or start_sample < 0:
        raise ValueError("start_sample must be a nonnegative integer")
    if not np.isfinite(frequency_offset):
        raise ValueError("frequency_offset must be finite")
    needed = round(p4chirp.PACKET_S * fs)
    if len(raw) - start_sample < needed:
        return None
    x = np.asarray(raw[start_sample:start_sample + needed], dtype=float)
    peak = np.max(np.abs(x))
    if peak == 0:
        return None
    # Bound numerical range without losing relative lane amplitudes.
    x = x / peak
    # The entire sweep fits below 3 kHz at these carriers. Decimate the real
    # input once before mixing both lanes, rather than dechirping 48 ksample/s
    # twice to retain only 533 1/3 samples/s.
    mix_fs = 8000 if abs(frequency_offset) <= 500 else 48000
    common = gcd(fs, mix_fs)
    if fs != mix_fs:
        x = resample_poly(x, mix_fs // common, fs // common)
    t = np.arange(len(x)) / mix_fs
    pulse = p4chirp.QUASI_RRC
    origin = (p4chirp.CHIRP_START_T + (len(pulse) - 1) / 16 - .5) * p4chirp.SYMBOL_S
    chirp_phase = .5 * p4chirp.CHIRP_HZ_PER_S * np.maximum(t - origin, 0)**2
    differentials = []
    for tone, delay in ((1530., 0), (550., 4)):
        mixed = x * np.exp(-2j * np.pi * ((tone + frequency_offset) * t + chirp_phase))
        baseband = resample_poly(mixed, 1, 3 * mix_fs // 1600)
        matched = np.convolve(baseband, pulse)
        symbols = matched[len(pulse) - 1 + delay::8][:217]
        if len(symbols) != 217:
            return None
        differentials.append(symbols[1:] * symbols[:-1].conjugate())

    accepted = []
    for marker in (8, 9):
        references = pactor2.marker_steps(marker)
        # Header word assignment is measured separately from data lane order.
        correlations = []
        for d, reference in zip(differentials, references):
            denom = np.sum(np.abs(d[:8]))
            correlations.append(float(abs(np.sum(d[:8] * np.exp(-1j * reference)))
                                      / denom) if denom else 0.)
        score = min(correlations)
        if score < .85:
            continue
        physical = [(d[8:] * np.exp(-3j * np.pi / 4)).real for d in differentials]
        for rank0_upper in (True, False):
            lanes = physical if rank0_upper else physical[::-1]
            channel = np.stack(lanes, axis=1).ravel()
            code = channel[pactor2.channel_of_code(p4chirp.PATH)]
            wire = pactor2.decode_frame(code, p4chirp.PATH)
            frame = p4frame.decode_field(wire, data_bytes=22)
            if frame is not None:
                accepted.append(Reception(frame, marker, rank0_upper, score))
    if len(accepted) != 1:
        return None
    return accepted[0]
