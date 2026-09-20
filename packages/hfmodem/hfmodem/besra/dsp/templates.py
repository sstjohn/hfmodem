# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP transmit sample templates — the sample-domain foundation.

Every ARDOP frame is played from a small set of precomputed one-symbol
templates: a two-tone leader symbol, the four 4FSK tones at 50/100/600 baud, and
the nine PSK/QAM carriers. All are 12 kHz, 16-bit, centred on 1500 Hz. Rendered
here from the generating formulas in `docs/protocols/ardop/11-WAVEFORM.md` §1, §3, §4 —
reference ``CalcTemplates.c`` (MIT) — as original numpy, nothing copied.

Reproduction notes (the reference's checked-in ``ardopSampleArrays.c`` was
generated across several code eras, so its rounding is not internally consistent):

- **PSK/QAM** (`PSK_100BD`) reproduces ardopcf's array *bit-exact* — phase
  accumulated in float32 (``CalcTemplates.c:114`` declares ``float dblAngle``),
  the raised-cosine ``sin(pi*k/119)`` window and the sine in float64, rounded
  half-away-from-zero (``CalcTemplates.c:409``).
- **4FSK 50 baud** (`FSK_50BD`) reproduces bit-exact — phase in float64, amplitude
  ``26000 * 1.1`` (``CalcTemplates.c:148``), truncated toward zero as C does when
  assigning the ``double`` product to ``short``.
- **4FSK 100/600 baud and the leader** reproduce every sample to within ±2 of the
  reference array; the residual (42 of 480 FSK-100 samples, 96 of 240 leader
  samples, all |Δ|≤2) is ardopcf's historical rounding drift, called out in
  ``CalcTemplates.c:398-404`` (LaRue, June 2024). It is deterministic and folds
  into the transmitter's documented sample tolerance (see tests/besra/test_modulator.py).

`intAmp = 26000` is the reference's headroom amplitude (`CalcTemplates.c:41`).

The parts of this module ``NOTICE`` names as ardopcf's are under that project's
MIT licence, Copyright (c) 2014-2024 Rick Muething, John Wiseman, Peter LaRue;
the copyright and permission notice it requires ship in ``NOTICE``.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 12000
CENTRE_HZ = 1500
INT_AMP = 26000
LEADER_LEN = 240  # samples in one 20 ms / 50 baud leader symbol
TWO_PI = 2.0 * np.pi

_f32 = np.float32


def _to_short(x: float) -> int:
    """C ``(short)`` of a floating value: truncate toward zero, wrap to int16."""
    return int(np.int16(int(x)))


def _c_round(x: float) -> int:
    """C ``round()``: nearest integer, ties away from zero."""
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


def _leader_symbol() -> np.ndarray:
    """One 20 ms symbol of the 1475/1525 Hz two-tone leader (`CalcTemplates.c:61`).

    ``x = 26000 * 0.55 * (sin(a) - sin(b))`` with the two tone phases computed
    fresh per sample; the phase products are rounded to float32 (the reference's
    ``float`` intermediates) before the float64 sine, then rounded to int16."""
    out = np.empty(LEADER_LEN, dtype=np.int16)
    for i in range(LEADER_LEN):
        a = float(_f32(((CENTRE_HZ - 25) / CENTRE_HZ) * (i / 8.0 * TWO_PI)))
        b = float(_f32(((CENTRE_HZ + 25) / CENTRE_HZ) * (i / 8.0 * TWO_PI)))
        x = INT_AMP * 0.55 * (np.sin(a) - np.sin(b))
        out[i] = _to_short(x)
    return out


def _fsk_tones(freqs: tuple[int, ...], n: int) -> np.ndarray:
    """4FSK carrier templates: ``26000 * 1.1 * sin(phase)`` truncated to int16,
    phase accumulated in float64 and wrapped at 2*pi (`CalcTemplates.c:144-152`).
    The 1.1 factor keeps the FSK peak just under the two-tone leader peak."""
    out = np.empty((len(freqs), n), dtype=np.int16)
    for i, f in enumerate(freqs):
        inc = TWO_PI * f / SAMPLE_RATE
        angle = 0.0
        for k in range(n):
            out[i, k] = _to_short(INT_AMP * 1.1 * np.sin(angle))
            angle += inc
            if angle >= TWO_PI:
                angle -= TWO_PI
    return out


PSK_CARRIERS_HZ = (800, 1000, 1200, 1400, 1500, 1600, 1800, 2000, 2200)


def _psk_carriers() -> np.ndarray:
    """The nine PSK/QAM carrier templates, four phases (0/45/90/135 deg) each,
    120 samples (`CalcTemplates.c:387-414`). Phases 4-7 are the negatives of 0-3
    and are synthesised at play time, so only the positive quadrant is stored.

    Each sample is ``round(26000 * sin(pi*k/119) * sin(phase))``: the raised-cosine
    envelope tapers the symbol edges to zero (the spec's cyclic-prefix window),
    the phase runs in float32, the products in float64."""
    out = np.empty((9, 4, 120), dtype=np.int16)
    for ci, f in enumerate(PSK_CARRIERS_HZ):
        inc = _f32(TWO_PI * f / SAMPLE_RATE)
        for j in range(4):
            angle = _f32(TWO_PI * j / 8.0)
            for k in range(120):
                env = np.sin(np.pi * k / 119)
                out[ci, j, k] = _c_round(INT_AMP * env * np.sin(float(angle)))
                angle = _f32(angle + inc)
                if float(angle) >= TWO_PI:
                    angle = _f32(float(angle) - TWO_PI)
    return out


# The two-tone leader / sync symbol (1475 + 1525 Hz), 240 samples.
LEADER_50BD: np.ndarray = _leader_symbol()

# 4FSK tone tables — index [tone 0..3][sample]. 200 Hz-BW 50 baud (240 samples),
# 500 Hz-BW 100 baud (120), 2000 Hz-BW 600 baud (20, FM only). Named as well as
# rendered: the band `core.occupied` says an ARDOP key fills is held against these.
FSK_TONES_HZ = {50: (1425, 1475, 1525, 1575),
                100: (1350, 1450, 1550, 1650),
                600: (600, 1200, 1800, 2400)}
FSK_50BD: np.ndarray = _fsk_tones(FSK_TONES_HZ[50], 240)
FSK_100BD: np.ndarray = _fsk_tones(FSK_TONES_HZ[100], 120)
FSK_600BD: np.ndarray = _fsk_tones(FSK_TONES_HZ[600], 20)

# PSK/QAM carriers — index [carrier 0..8][phase 0..3][sample], 120 samples/symbol.
# Carrier 4 (1500 Hz) is single-carrier only; the others straddle it.
PSK_100BD: np.ndarray = _psk_carriers()

# The 1500 Hz phase-0 template also serves as the trailer tone (`Modulate.c:604`).
TRAILER_1500HZ: np.ndarray = PSK_100BD[4, 0]
