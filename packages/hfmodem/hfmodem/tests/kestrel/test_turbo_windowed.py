# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The parallel-window BCJR: it decodes what the whole-trellis recursion decodes, and
an L16 frame fits inside the ARQ turnaround."""
from __future__ import annotations

import time

import numpy as np
import pytest

from hfmodem.kestrel.coding import turbo
from hfmodem.kestrel.rx import tablegen


def _encode(info, perm, coded):
    N = len(info)
    if coded == 2 * N + 12:
        return turbo.encode(info, perm)
    if coded == 3 * N + 12:
        return turbo.encode_r13(info, perm)
    return turbo.encode_punctured(info, perm, coded)


def _decode(Lc, N, perm, coded, iters=8):
    if coded == 2 * N + 12:
        return turbo.decode_from_coded_llr(Lc, N, iters=iters, perm=perm)
    if coded == 3 * N + 12:
        return turbo.decode_r13(Lc, N, perm, iters=iters)
    return turbo.decode_punctured(Lc, N, perm, coded, iters=iters)


@pytest.mark.parametrize("N,coded", sorted({(n2, n1) for _, n1, n2 in tablegen.BW2300}))
def test_noiseless_round_trip(N, coded):
    rng = np.random.default_rng(N)
    perm = rng.permutation(N)
    info = rng.integers(0, 2, N)
    Lc = 8.0 * (1 - 2 * _encode(info, perm, coded))
    assert np.array_equal(_decode(Lc, N, perm, coded, iters=2), info)


def test_windows_decide_as_the_whole_trellis_does(monkeypatch):
    N = 368                                  # BW500 L4: three windows at _WIN = 128
    assert N + 3 > 2 * turbo._WIN
    perm = turbo.interleaver(N)
    rng = np.random.default_rng(11)
    sigma = 0.8
    for _ in range(8):
        info = rng.integers(0, 2, N)
        y = (1 - 2.0 * turbo.encode(info, perm)) + sigma * rng.standard_normal(2 * N + 12)
        Lc = np.clip(2 * y / sigma**2, -8, 8)
        windowed = turbo.decode_from_coded_llr(Lc, N, iters=8)
        with monkeypatch.context() as m:
            m.setattr(turbo, "_WIN", 10**9)
            whole = turbo.decode_from_coded_llr(Lc, N, iters=8)
        assert np.array_equal(whole, info)
        assert np.array_equal(windowed, whole)


@pytest.mark.realtime
def test_l16_frame_decodes_inside_the_turnaround():
    N, coded = 26056, 32585
    rng = np.random.default_rng(16)
    perm = rng.permutation(N)
    info = rng.integers(0, 2, N)
    Lc = np.clip(4.0 * (1 - 2 * _encode(info, perm, coded)) + 2.0 * rng.standard_normal(coded),
                 -8, 8)
    best = np.inf
    for _ in range(3):
        t = time.perf_counter()
        bits = _decode(Lc, N, perm, coded)
        best = min(best, time.perf_counter() - t)
    assert np.array_equal(bits, info)
    assert best < 0.5, f"L16 decode took {best:.3f} s with the full 8-iteration schedule"
