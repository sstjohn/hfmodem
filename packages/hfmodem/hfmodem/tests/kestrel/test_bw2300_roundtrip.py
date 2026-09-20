# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW2300 wideband OFDM codec round-trip — TX -> RX byte-exact, every speed level.

kestrel BW2300 transmitter synthesises a burst at each speed level (the index-mod
records 2 and 3 + the high-throughput ladder rec9-16); the BW2300 receiver recovers
the exact payload bytes CRC-clean. Built from spec/01 §2300, spec/03 §3.5.3-3.5.4,
spec/06 §6.1c-6.1d. See kestrel/rx/varahf2300.py for the value-law families.

A round trip is the weakest evidence there is — it says the two halves agree, not
that either matches VARA. What holds record 2 to the air is `test_bw2300_rec2.py`,
which decodes recorded VARA overs at that level; what holds record 3 is
`test_bw2300_real_audio.py`. This file is the regression net under both.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx


def _payload(level, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, rx.payload_bytes(level)).astype(np.uint8).tobytes()


@pytest.mark.parametrize("level", rx.LEVELS)
def test_level_roundtrip_byte_exact(level):
    pl = _payload(level, seed=1000 + level)
    audio = tx.synth_burst(pl, level)
    fr = rx.decode_burst(audio, level)
    assert fr.crc_ok, f"level {level} CRC failed"
    assert fr.payload == pl, f"level {level} payload not byte-exact"


def test_base_trial_decode_no_level_hint():
    """The base level auto-detects (unique burst length) and decodes CRC-clean."""
    pl = _payload(rx.BASE_LEVEL, seed=7)
    fr = rx.decode_burst(tx.synth_burst(pl, rx.BASE_LEVEL))   # level=None -> trial
    assert fr.level == rx.BASE_LEVEL and fr.crc_ok and fr.payload == pl


def test_noise_fails_crc():
    """Pure noise (no valid codeword) must NOT pass CRC — guards against a trivial
    always-pass. (The FEC deliberately corrects light corruption, so this uses a
    burst-shaped noise array rather than a few wiped symbols.)"""
    rng = np.random.default_rng(99)
    n = rx.burst_length(rx.BASE_LEVEL)
    noise = rng.standard_normal(n)
    fr = rx.decode_burst(noise, rx.BASE_LEVEL)
    assert not fr.crc_ok


if __name__ == "__main__":
    for lv in rx.LEVELS:
        pl = _payload(lv, 1000 + lv)
        fr = rx.decode_burst(tx.synth_burst(pl, lv), lv)
        print(f"level {lv:2d} {rx.RECORDS[lv].name:10s} "
              f"payload={rx.payload_bytes(lv):4d}B crc_ok={fr.crc_ok} "
              f"byte_exact={fr.payload == pl}")
