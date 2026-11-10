# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The batched `_cs_profile` reads what the per-hop loop read.

The loop it replaced ran one 960-point transform and one median per millisecond
of audio, in Python, and on the P3 arms of 2026-11-06 held the interpreter long
enough to make the FTDX10's audio callback 5-11 ms late. `_reference` is that
loop verbatim; every segment here is real audio, with P1 control bursts in it and
without.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.shrike import rxfront
from hfmodem.tests.kestrel import corpora

FS = rxfront.FS
REPO = Path(__file__).resolve().parents[5]

#: WS8EOC answering, 2026-07-30: hold windows with and without a control signal.
HOLDS = [REPO / "captures" / "onair-0730-2036" / f"hold_0{n}.wav" for n in (1, 2, 5, 9)]

#: Tonight's arms through the FTDX10, 3.7 s windows. Not committed: a checkout
#: without `operator-data/` skips them.
ARMS = REPO / "operator-data" / "captures"
ARM_WINDOWS = [
    (ARMS / "onair-20261106T043227401104Z-85aaa031-9fa7-482a-9ec0-6e19ad0f8aa9"
     / "stream-l7278-20261106T0432Z-s00000000000000168577.flac", 25.0),
    (ARMS / "onair-20261106T044355270112Z-24568757-0330-428d-b108-8ec4eabd9b6f"
     / "stream-l7278-20261106T0444Z-s00000000000000000129.flac", 0.0),
    (ARMS / "onair-20261106T044355270112Z-24568757-0330-428d-b108-8ec4eabd9b6f"
     / "stream-l7278-20261106T0444Z-s00000000000000000129.flac", 40.0),
]


def _reference(seg):
    w, h = int(rxfront.CS_WIN_S * FS), int(rxfront.CS_HOP_S * FS)
    if seg.size < w * 4:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    f = np.fft.rfftfreq(w, 1 / FS)
    band = (f > 300) & (f < 2600)
    tone = ((f > 1360) & (f < 1440)) | ((f > 1560) & (f < 1640))
    wideband = (f > 1900) & (f < 2500)
    p1, wide, ts = [], [], []
    for i in range(0, seg.size - w, h):
        F = rxfront._Spectrum(seg[i:i + w]).mag
        g = np.median(F[band]) + 1e-12
        p1.append(F[tone].mean() / g)
        wide.append(F[wideband].mean() / g)
        ts.append((i + w / 2) / FS)
    return np.array(p1), np.array(wide), np.array(ts)


def _same(seg):
    want, got = _reference(seg), rxfront._cs_profile(seg)
    for a, b in zip(want, got):
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=0)
    return got


def _bursts(prof):
    return len(rxfront._cs_length(rxfront._p1_runs(prof)))


@pytest.mark.parametrize("path", HOLDS, ids=lambda p: p.stem)
def test_answered_holds(path):
    if not path.exists():
        pytest.skip(f"{path} absent")
    _same(rxfront.load_wav(str(path)))


def test_holds_include_control_signals():
    present = [p for p in HOLDS if p.exists()]
    if not present:
        pytest.skip("captures/onair-0730-2036 absent")
    assert sum(_bursts(_same(rxfront.load_wav(str(p)))) for p in present) >= 3


def test_band_noise():
    path = corpora.REGRESS_FIXTURES / "neg_noise_chatham.wav"
    if not path.exists():
        pytest.skip(f"{path} absent")
    a = rxfront.load_wav(str(path))
    n = int(3.7 * FS)
    for i in range(0, min(len(a) - n, 4 * n), n):
        assert _bursts(_same(a[i:i + n])) == 0


@pytest.mark.parametrize("path,start", ARM_WINDOWS, ids=lambda v: str(v)[-12:])
def test_arm_windows(path, start):
    if not path.exists():
        pytest.skip(f"{path} absent")
    a = wav.read(path, FS)
    _same(a[int(start * FS):int((start + 3.7) * FS)])


def test_edges():
    rng = np.random.default_rng(6)
    w = int(rxfront.CS_WIN_S * FS)
    for n in (w * 4 - 1, w * 4, w * 4 + 1, w + rxfront.CS_CHUNK * 48 + 7):
        _same(rng.standard_normal(n))
    _same(rng.standard_normal(w * 9).astype(np.float32))
