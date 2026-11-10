# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP transmit sample templates — the sample-domain foundation.

Every ARDOP frame is played from a small set of one-symbol templates: a two-tone
leader symbol, the four 4FSK tones at 50, 100 and 600 baud, and the nine PSK/QAM
carriers. All are 12 kHz, 16-bit and centred on 1500 Hz, rendered here from the
formulas in `docs/protocols/ardop/11-WAVEFORM.md` §1, §3 and §4.

A formula alone does not pin the last bit, so the numerics are chosen to land
where the tables other ARDOP stations play from land:

- **PSK/QAM** (`PSK_100BD`) matches them exactly with the carrier phase stepped
  in float32, the sine and the ``sin(πk/119)`` edge taper in float64, and the
  product rounded half away from zero.
- **4FSK 50 baud** (`FSK_50BD`) matches exactly with the phase stepped in
  float64, an amplitude of 26000 × 1.1, and the product truncated toward zero.
- **4FSK 100/600 baud and the leader** match every sample to within ±2 (42 of
  480 FSK-100 samples and 96 of 240 leader samples differ). Those tables were
  generated in more than one era with different rounding, so no single rule
  reproduces all of them; the residual is deterministic and sits inside the
  transmitter's sample tolerance (see tests/besra/test_modulator.py).

The amplitude 26000 leaves headroom below int16 full scale for summed carriers.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 12000
CENTRE_HZ = 1500
INT_AMP = 26000
LEADER_LEN = 240  # samples in one 20 ms / 50 baud leader symbol
TWO_PI = 2.0 * np.pi


def _truncate16(x: np.ndarray) -> np.ndarray:
    """Toward zero, then wrapped into int16."""
    return np.trunc(x).astype(np.int64).astype(np.int16)


def _round16(x: np.ndarray) -> np.ndarray:
    """Nearest, ties away from zero, into int16."""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int16)


def _sin(x: np.ndarray) -> np.ndarray:
    """``np.sin`` taken one sample at a time. The matched tables sit close to
    rounding boundaries in places, and a vectorised sine is free to round its last
    bit differently on some platforms."""
    return np.array([np.sin(v) for v in x.tolist()])


def _phase_ramp(start: float, step: float, n: int, single: bool) -> np.ndarray:
    """``n`` phases from ``start``, ``step`` apart, brought back by 2π each time
    one reaches it. They are accumulated one step at a time, in float32 when
    ``single``, because the matched tables were built that way and a closed form
    rounds differently."""
    cast = np.float32 if single else float
    phase, step = cast(start), cast(step)
    out = np.empty(n)
    for k in range(n):
        out[k] = phase
        phase = cast(phase + step)
        if float(phase) >= TWO_PI:
            phase = cast(float(phase) - TWO_PI)
    return out


def _leader_symbol() -> np.ndarray:
    """One 20 ms symbol of the 1475/1525 Hz two-tone leader.

    The two tones are taken in difference at 0.55 × 26000 each. Each tone's phase
    is computed directly from the sample index and held to float32 before its
    float64 sine; the difference is truncated to int16."""
    cycles = np.arange(LEADER_LEN) / 8.0 * TWO_PI   # 1500 Hz at 12 kHz: 8 samples a cycle
    low = ((CENTRE_HZ - 25) / CENTRE_HZ * cycles).astype(np.float32).astype(np.float64)
    high = ((CENTRE_HZ + 25) / CENTRE_HZ * cycles).astype(np.float32).astype(np.float64)
    return _truncate16(INT_AMP * 0.55 * (_sin(low) - _sin(high)))


def _fsk_tones(freqs: tuple[int, ...], n: int) -> np.ndarray:
    """One ``n``-sample symbol per 4FSK tone at 26000 × 1.1, which keeps the FSK
    peak just under the two-tone leader's."""
    return np.stack([
        _truncate16(INT_AMP * 1.1 * _sin(_phase_ramp(0.0, TWO_PI * f / SAMPLE_RATE, n, False)))
        for f in freqs])


PSK_CARRIERS_HZ = (800, 1000, 1200, 1400, 1500, 1600, 1800, 2000, 2200)


def _psk_carriers() -> np.ndarray:
    """The nine PSK/QAM carriers at the four phases 0/45/90/135°, 120 samples
    each. Phases 180-315° are these negated and are formed at play time.

    Each symbol is shaped by ``sin(πk/119)``, which takes its edges to zero (the
    spec's cyclic-prefix window)."""
    taper = INT_AMP * _sin(np.pi * np.arange(120) / 119)
    return np.stack([
        np.stack([_round16(taper * _sin(_phase_ramp(TWO_PI * j / 8.0, TWO_PI * f / SAMPLE_RATE,
                                                      120, True)))
                  for j in range(4)])
        for f in PSK_CARRIERS_HZ])


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

# The trailer is the 1500 Hz carrier at phase 0.
TRAILER_1500HZ: np.ndarray = PSK_100BD[4, 0]
