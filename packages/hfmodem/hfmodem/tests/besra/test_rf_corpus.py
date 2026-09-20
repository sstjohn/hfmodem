# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra against the shared off-air corpus: decode real ARDOP, reject everything else.

`rf-corpus/regress/fixtures` holds real recordings with known-enough answers
(`fixtures.toml`). This is besra's discrimination gate on genuine off-air audio,
the thing the synthetic fixtures cannot exercise:

  * it must recover the ARDOP oracle's callsigns byte-exact, and
  * it must find NO CRC-valid ARDOP frame in any VARA / PACTOR / FT8 / noise
    recording — the false-positive class where a lone bodyless control frame
    (a k=0 DATANAK/ACK) CRC-collides across a long acquisition search on a busy
    channel. One such VARA capture minted a DATANAK before the bodyless-frame
    crisp floor; this keeps it dead.

The corpus lives outside the repo; the module skips cleanly when it is absent.
"""

from __future__ import annotations

import os
import tomllib
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.besra.phy.demodulator import decode


def _find_corpus() -> Path | None:
    env = os.environ.get("HFMODEM_CORPUS") or os.environ.get("RF_CORPUS")
    cands = ([Path(env)] if env else []) + [Path(__file__).resolve().parents[6] / "rf-corpus"]
    return next((c for c in cands if (c / "regress" / "fixtures.toml").exists()), None)


_CORPUS = _find_corpus()
pytestmark = pytest.mark.skipif(_CORPUS is None, reason="rf-corpus not present")

_FIXTURES = _CORPUS / "regress" / "fixtures" if _CORPUS else None
_MANIFEST = (tomllib.loads((_CORPUS / "regress" / "fixtures.toml").read_text())["fixture"]
             if _CORPUS else {})


def _load(path) -> np.ndarray:
    """A recording as 12 kHz int16, resampling from its rate at the edge.

    The corpus carries float32 WAVs as well as PCM16 — `wave` rejects those, so
    read through scipy and scale, rather than skipping a fixture for its container."""
    try:
        with wave.open(str(path)) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)
    except wave.Error:
        sr, raw = wavfile.read(str(path))
        x = np.asarray(raw, dtype=np.float64)
        if x.ndim > 1:
            x = x[:, 0]
        if np.issubdtype(raw.dtype, np.floating):
            x *= 32768.0
    if sr != 12000:
        x = resample_poly(x, 12000, sr)
    return np.clip(np.round(x), -32768, 32767).astype("<i2")


def _ardop_frames(name: str) -> list:
    return [f for f in decode(_load(_FIXTURES / f"{name}.wav")) if f.ok]


def _corpus_frames(name: str) -> list:
    return [f for f in decode(_load(_CORPUS / f"{name}.wav")) if f.ok]


# Real off-air ARDOP in the top-level corpus (not the curated fixtures): besra
# recovered these ConReqs — valid callsigns, valid CRC — from KC3OWM where the
# VARA/PACTOR monitor logged "nothing heard". Genuine off-air ARDOP, a stronger
# positive than the synthetic oracle, and the true-positives the false-positive
# fix must not break.
_REAL_OFFAIR = {"7102k_065457": ("KC3OWM", "K4PAR-2"),
                "7101k_065546": ("KC3OWM", "AB4NW")}
# Busy VARA/PACTOR channels that minted a bare control frame (DATANAK, ConAck) on a
# 16-bit header CRC before the per-frame-class crisp floor. Must stay silent.
_WAS_FALSE_POSITIVE = ["14105k_102347", "14098k_003826"]


# Every real non-ARDOP recording besra must stay silent on. Excludes the ARDOP
# oracle, the ardop watch fixture (manifest: deliberately unasserted — besra
# frame-syncs there without validating), and two genuinely mode-ambiguous winlink
# captures that could legitimately carry ARDOP.
_SKIP = {"oracle_ardop_conreq", "edge_ardop_suspect", "edge_winlink_oceania", "edge_long_80m"}
_NEGATIVES = sorted(n for n in _MANIFEST if n not in _SKIP)


def test_decodes_the_ardop_oracle():
    """The one byte-level truth assertion: both callsigns, exactly."""
    frames = _ardop_frames("oracle_ardop_conreq")
    hit = [f for f in frames if f.caller == "M7TFF" and f.target == "GB7RDG-15"]
    assert hit, ("expected ConReq M7TFF > GB7RDG-15, got "
                 f"{[(hex(f.type), f.caller, f.target) for f in frames]}")


@pytest.mark.parametrize("name", _NEGATIVES)
def test_finds_no_ardop_in_non_ardop(name):
    """Discrimination: a CRC-valid ARDOP frame out of VARA/PACTOR/FT8/noise is a
    false positive — the defect that would send an operator to call on a dead
    channel, or print a phantom frame in the monitor."""
    if not (_FIXTURES / f"{name}.wav").exists():
        pytest.skip(f"{name}.wav absent")
    minted = _ardop_frames(name)
    assert not minted, (f"besra minted ARDOP from {name!r}: "
                        f"{[(hex(f.type), f.name) for f in minted]}")


@pytest.mark.parametrize("name,call", sorted(_REAL_OFFAIR.items()))
def test_decodes_real_offair_ardop(name, call):
    """Real off-air ARDOP ConReqs, callsigns exact — the corroboration that keeps
    the false-positive floor from eating genuine connect requests."""
    if not (_CORPUS / f"{name}.wav").exists():
        pytest.skip(f"{name}.wav absent")
    caller, target = call
    hit = [f for f in _corpus_frames(name) if f.caller == caller and f.target == target]
    assert hit, (f"expected ConReq {caller} > {target} from {name}, got "
                 f"{[(hex(f.type), f.caller, f.target) for f in _corpus_frames(name)]}")


@pytest.mark.parametrize("name", _WAS_FALSE_POSITIVE)
def test_no_bare_control_false_positive(name):
    """A busy VARA/PACTOR channel must not mint a bare control/ack frame — the
    exact regression the per-class crisp floor closed."""
    if not (_CORPUS / f"{name}.wav").exists():
        pytest.skip(f"{name}.wav absent")
    minted = _corpus_frames(name)
    assert not minted, (f"besra minted a phantom frame from {name!r}: "
                        f"{[(hex(f.type), f.name) for f in minted]}")
