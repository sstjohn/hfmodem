# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Unlinked KISS UI broadcast synthesis."""

import numpy as np
from ..rx import broadcast as B
from ..rx.beacon import HEADER as LEVEL2_HEADER
from ..vara.vara_mfsk import synth_tones
from . import varahf500_tx as T
from .ui import packetize


def synth_broadcast(message, *, amplitude=0.5):
    level, frames = packetize(message)
    tones = LEVEL2_HEADER if level == 2 else B.HEADERS[level]
    header = synth_tones(tones, amplitude=amplitude)
    parts = [header, np.zeros(32 * 2048 + 1536 - len(header))]
    for i, frame in enumerate(frames):
        if i:
            parts.append(np.zeros(1536))
        if level == 4:
            body = T.synth_burst(frame, ncol=T.C0 + T.NSYM)
            body = body * (amplitude / max(np.sqrt(2 * np.mean(body**2)), 1e-30))
        else:
            body = T.synth_l3_burst(frame, level=level) * (amplitude / T.L3_OUT_SCALE)
        parts.append(body)
    return np.concatenate(parts)
