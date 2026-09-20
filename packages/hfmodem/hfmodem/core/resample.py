# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Rate conversion at the radio edge, and nowhere else.

Each mode keeps its own sample rate — ARDOP's 12 kHz is normative to ARDOP, and
harmonising it to the card's 48 kHz would break besra's byte-exact cross-decode
against the real ardopcf binary — so the sound card is the one boundary where a
rate changes. Both call sites today are inside `besra/radio.py`; this is here
because the station fans one 48 kHz capture out to lanes that do not share a
rate, and each of them needs the same conversion at the same edge.

Measured on ARDOP, which is the mode with the tightest claim on it: every mode —
4FSK, 4PSK, 8PSK and 16QAM through 2000 Hz — survives the round trip
payload-exact on realistic random payloads. ARDOP has always been a sound-card
modem and 16QAM is no exception. A pathological high-PAPR payload can clip in the
TX filter and self-corrupt, but that is ardopcf's own clip behaviour, bit-matched
here, and real compressed traffic clips negligibly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from .rates import CARD_RATE_HZ


def to_card(samples: np.ndarray, rate: int) -> np.ndarray:
    """int16 at `rate` → float32 for the sound card."""
    up = resample_poly(samples.astype(np.float64), CARD_RATE_HZ, rate)
    return (up / 32768.0).astype(np.float32)


def for_card(samples: np.ndarray, rate: int) -> np.ndarray:
    """Anything a protocol composed, in the units the card plays: card rate,
    float32 in [-1, 1].

    Rate and scale travel together on purpose. A protocol that works at some
    other rate also works in integer PCM — ARDOP's 12 kHz int16 is the only one
    here — so a caller offered two conversions would eventually do one and forget
    the other, and the result of forgetting is silence or a rail rather than a
    number that looks wrong.
    """
    a = np.asarray(samples)
    a = (a.astype(np.float32) / -float(np.iinfo(a.dtype).min)
         if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32))
    if rate != CARD_RATE_HZ:
        a = resample_poly(a.astype(np.float64), CARD_RATE_HZ, rate).astype(np.float32)
    return a


def from_card(samples: np.ndarray, rate: int) -> np.ndarray:
    """Sound-card float32 → int16 at `rate`."""
    down = resample_poly(samples.astype(np.float64), rate, CARD_RATE_HZ)
    return np.clip(np.round(down * 32768.0), -32768, 32767).astype("<i2")
