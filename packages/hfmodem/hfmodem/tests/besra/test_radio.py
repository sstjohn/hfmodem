# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Radio backend: the 12k<->48k resampling fidelity every mode depends on, without
hardware. The receive path either side of it lives in `test_radio_audio_path.py`."""

from __future__ import annotations

import numpy as np

from hfmodem.core.resample import from_card, to_card
from hfmodem.besra.phy import demodulator as D
from hfmodem.besra.phy import modulator as M


def test_resample_roundtrip_preserves_every_mode():
    # Every mode — through 16QAM.2000 — survives the 12k↔48k sound-card trip on
    # realistic (random) data. ARDOP is a sound-card modem; the ×4 resample is
    # faithful. (Payloads are random because a pathological high-PAPR pattern can
    # clip in ardopcf's own TX filter, which is a clip property, not the resampler.)
    rng = np.random.default_rng(0)
    for ft, k in [(0x4A, 64), (0x50, 128), (0x44, 108),
                  (0x46, 128), (0x54, 256), (0x64, 512), (0x74, 1024)]:
        payload = bytes(rng.integers(0, 256, k, dtype=np.uint8))
        x = M.render_frame(ft, payload=payload, session_id=0x5E)
        back = from_card(to_card(x, D.SAMPLE_RATE), D.SAMPLE_RATE)
        assert len(back) == len(x)
        pad = np.concatenate([np.zeros(2400, "<i2"), back, np.zeros(4800, "<i2")])
        got = D.decode(pad)
        assert got and got[0].type == ft and got[0].payload == payload, hex(ft)
