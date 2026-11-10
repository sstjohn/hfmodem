# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate besra's transmitter against ardopcf's own rendered audio.

The 59 ``txframe_*.wav`` fixtures are ardopcf renders, one per Phase-A frame type
(see ``groundtruth.py`` / ``txframe-manifest.txt``). For each, besra builds the
same frame's bytes and renders them, and we compare int16 sample streams.

Why a tolerance rather than bit-exactness
-----------------------------------------
Two deterministic, bounded sources keep besra a few LSBs off ardopcf's shipped
output — everything else (template values for PSK/QAM & 50-baud 4FSK, sign
conventions, differential phase, carrier scaling, soft clip, sample counts) is
reproduced exactly:

1. **The TX filter.** ardopcf's comb-resonator filter (``Modulate.c:771-914``)
   runs in C ``float``; besra evaluates it in float64. The resonators sit at pole
   radius 0.9995, so the two precisions land on opposite sides of the final
   ``(short)`` truncation for a minority of samples — a ±1 floor visible even on
   frames whose templates are bit-exact (IDLE/DISC/END/ConRej* land at ≤1).

2. **Historical template rounding.** ardopcf's checked-in ``ardopSampleArrays.c``
   was generated across several code eras; its leader (96/240 samples, |Δ|≤2) and
   100-/600-baud 4FSK (|Δ|≤1) arrays are not reproducible from one consistent
   formula (``CalcTemplates.c:398-404``, LaRue's note). besra generates templates
   from the formulas, so those samples differ by ≤2 and propagate through the
   filter gain.

The observed worst case is 6 LSBs (0.02% of full scale) on the wide-filter
600-baud FM frames; everything else is ≤5, with RMS error below ~1.1 LSB. The
bounds asserted here are tight enough to catch any byte-assembly or modulation
regression: a single wrong byte flips four tone symbols and drives the diff to
full scale, so passing these bounds is proof of exact framing and modulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmodem.besra.phy import modulator
from . import groundtruth as gt


def _is_fm_600(name: str) -> bool:
    return ".2000.600" in name


def _render(entry: dict) -> np.ndarray:
    meta = entry.get("meta", {})
    return modulator.render_frame(
        entry["type"], payload=entry.get("payload", b""),
        session_id=entry["session_id"], **meta)


@gt.requires_reference
@pytest.mark.parametrize("name", gt.wav_params())
def test_render_matches_fixture(name: str) -> None:
    entry = gt.txframe_manifest()[name]
    mine = _render(entry)
    ref = gt.load_wav(name)

    # Sample count is exact — leader + sync + header + data + trailer, byte-for-byte.
    assert len(mine) == len(ref), f"{name}: {len(mine)} vs {len(ref)} samples"

    diff = mine.astype(np.int64) - ref.astype(np.int64)
    max_abs = int(np.abs(diff).max())
    rms = float(np.sqrt((diff * diff).mean()))

    # See the module docstring for the two bounded error sources. The wide 2000 Hz
    # filter on the 600-baud FM frames widens the float floor slightly.
    max_bound, rms_bound = (6, 3.0) if _is_fm_600(name) else (5, 1.5)
    assert max_abs <= max_bound, f"{name}: max |Δ| = {max_abs}"
    assert rms <= rms_bound, f"{name}: RMS Δ = {rms:.2f}"


# Frames whose every template is bit-exact against ardopcf (50-baud 4FSK + the
# 1500 Hz trailer, no leader-dominated payload): these expose the filter float
# floor alone and must stay at ≤1 LSB, proving the ±1 claim above.
_FILTER_FLOOR = [
    "txframe_IDLE.wav", "txframe_DISC.wav", "txframe_END.wav",
    "txframe_ConRejBusy.wav", "txframe_ConRejBW.wav",
    "txframe_DataACK-q80.wav", "txframe_DataNAK-q60.wav",
]


@gt.requires_reference
@pytest.mark.parametrize("name", _FILTER_FLOOR)
def test_filter_float_floor_is_one_lsb(name: str) -> None:
    entry = gt.txframe_manifest()[name]
    diff = _render(entry).astype(np.int64) - gt.load_wav(name).astype(np.int64)
    assert int(np.abs(diff).max()) <= 1, f"{name}: filter floor exceeded"


@gt.requires_reference
def test_leader_and_sync_region() -> None:
    """The leader + sync + header prefix (shared by every frame) reproduces the
    fixture within the leader's ±2 template residual, over the full 22-symbol
    5280-sample prefix (minus the 60-sample filter warm-up)."""
    entry = gt.txframe_manifest()["txframe_IDLE.wav"]
    mine = _render(entry)
    ref = gt.load_wav(name="txframe_IDLE.wav")
    prefix = 12 * 240 + 10 * 240 - 60  # leader + header, less filter group delay
    diff = mine[:prefix].astype(np.int64) - ref[:prefix].astype(np.int64)
    assert int(np.abs(diff).max()) <= 2
