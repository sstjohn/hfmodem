# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The KISS broadcast frame at 48 kHz: its header, then one record-1 burst.

Stock's own emission (bench 3, 10-23) opens the body 32 ms after the header's last
symbol, as its CQ frame does, and keys it at the header's r.m.s. (KG9B, KC0TAF and
N9PMI at +9 to +12 dB read within 1.4 dB of their headers)."""
import numpy as np

from ..rx import beacon as B
from ..vara.vara_mfsk import synth_tones
from .varahf500_tx import L3_OUT_SCALE, build_l3_frame, synth_l3_burst

GAP_S = .032


def synth_beacon(text: str, *, amplitude: float = 0.5, marker: int = 0) -> np.ndarray:
    header = synth_tones(B.HEADER, amplitude=amplitude)
    frame = build_l3_frame(B.build_payload(text), marker, level=B.LEVEL)
    body = synth_l3_burst(frame, level=B.LEVEL, training=B.TRAINING) * (amplitude / L3_OUT_SCALE)
    gap = np.zeros(round(GAP_S * 48000) - (len(header) - len(B.HEADER) * 2048))
    return np.concatenate((header, gap, body))
