# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fixed R2 cb.py analyzer; origin and defaults stay independent of production."""
from math import gcd

import numpy as np
from scipy.signal import resample_poly

T = .015
PULSE = np.array([
    0.0013033, 0.00099339, 0.00015167, -0.0021225,
    -0.0058615, -0.01035, -0.014008, -0.014558,
    -0.0095413, 0.002945, 0.023476, 0.050767,
    0.081603, 0.11134, 0.13489, 0.14791, 0.14791,
    0.13489, 0.11134, 0.081603, 0.050767, 0.023476,
    0.002945, -0.0095413, -0.014558, -0.014008,
    -0.01035, -0.0058615, -0.0021225, 0.00015167,
    0.00099339, 0.0013033,
])
PACKET_S = 216 * T + 4 * T + T / 2
ORIGIN = (12.5 + 31 / 16 - .5) * T
RATE = 294.
TONES = ((1530., 0), (550., 4))
MIX = 8000
SPS = 8
BBFS = 1600 / 3
PRE = .1
MARKER = {
    8: (
        [0.7853981852531433, -2.356194496154785, 0.7853981852531433,
         -2.356194496154785, 0.7853981852531433, -2.356194496154785,
         -0.7853981852531433, -2.356194496154785],
        [0.7853981852531433, 0.7853981852531433, 0.7853981852531433,
         -2.356194496154785, 0.7853981852531433, -2.356194496154785,
         -2.356194496154785, 2.356194496154785],
    ),
    9: (
        [-0.7853981852531433, 2.356194496154785, 0.7853981852531433,
         0.7853981852531433, -0.7853981852531433, -2.356194496154785,
         2.356194496154785, 0.7853981852531433],
        [-2.356194496154785, 0.7853981852531433, 0.7853981852531433,
         0.7853981852531433, 2.356194496154785, -2.356194496154785,
         -2.356194496154785, -0.7853981852531433],
    ),
}


def segment(x, fs, start):
    a = int(round(start - PRE * fs))
    b = int(round(start + (PACKET_S + PRE) * fs))
    seg = np.asarray(x[max(a, 0):b], float)
    if a < 0:
        seg = np.concatenate([np.zeros(-a), seg])
    if len(seg) < b - a:
        seg = np.concatenate([seg, np.zeros(b - a - len(seg))])
    seg = seg / (np.max(np.abs(seg)) or 1)
    g = gcd(int(fs), MIX)
    if fs != MIX:
        seg = resample_poly(seg, MIX // g, int(fs) // g)
    return seg


def baseband(seg, cfo, tone, origin=ORIGIN, rate=RATE):
    t = np.arange(len(seg)) / MIX - PRE
    ph = .5 * rate * np.maximum(t - origin, 0) ** 2
    mixed = seg * np.exp(-2j * np.pi * ((tone + cfo) * (t + PRE) + ph))
    bb = resample_poly(mixed, 1, 3 * MIX // 1600)
    m = np.convolve(bb, PULSE)
    return m, round(PRE * BBFS)


def symbols(seg, cfo, origin=ORIGIN, rate=RATE, delays=(0, 4), extra=(0, 0)):
    out = []
    for (tone, _), delay in zip(TONES, delays):
        m, lead = baseband(seg, cfo, tone, origin, rate)
        first = lead + len(PULSE) - 1 + delay - SPS * extra[0]
        out.append(m[first::SPS][:217 + extra[0] + extra[1]])
    return out


def residual_phase(d, marker):
    ref = MARKER[marker]
    out = []
    for di, r in zip(d, ref):
        r = np.asarray(r)
        h = np.angle(di[:8] * np.exp(-1j * r))
        q = np.angle((di[8:] * np.exp(-3j * np.pi / 4)) ** 2) / 2
        out.append(np.concatenate([h, q]))
    return out


def fine_fit(d, marker, coarse):
    r = residual_phase(d, marker)
    tk = (np.arange(216) + 1) * T
    out = []
    for k, (ri, c) in enumerate(zip(r, coarse)):
        tt = tk + (T / 2 if k == 1 else 0)
        best = None
        for dorg in np.arange(c[1] - 0.003, c[1] + 0.00301, 0.00025):
            o = ORIGIN + dorg
            base = RATE * (np.maximum(tt - o, 0) - np.maximum(tt - ORIGIN, 0))
            for rate in np.arange(c[2] - 0.4, c[2] + 0.401, 0.02):
                model = base + (rate - RATE) * np.maximum(tt - o, 0)
                mz = np.mean(np.exp(2j * (ri - 2 * np.pi * T * model)))
                if best is None or abs(mz) > best[0]:
                    best = (float(abs(mz)), float(dorg), float(rate), float(np.angle(mz) / 2 / (2 * np.pi * T)))
        out.append(best)
    return out


def fine(x, fs, r):
    seg = segment(x, fs, round(r['start_s'] * fs))
    hi, lo = symbols(seg, r['cfo'])
    d = [s[1:] * s[:-1].conj() for s in (hi, lo)]
    return fine_fit(d, r['marker'], r['sweep'])
