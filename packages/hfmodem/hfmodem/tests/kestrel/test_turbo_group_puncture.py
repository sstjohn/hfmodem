# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The group puncture read off high speed capable BW2300 and BW500 tapes."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.coding import turbo
from hfmodem.kestrel.rx import tablegen

_PAIRS = sorted({(n2, n1) for tab in (tablegen.BW2300, tablegen.BW500)
                 for _, n1, n2 in tab if n1 <= 2 * n2 + 12})


@pytest.mark.parametrize("N,coded", _PAIRS)
def test_group_puncture_round_trip(N, coded):
    rng = np.random.default_rng(N)
    perm = rng.permutation(N)
    info = rng.integers(0, 2, N)
    out = turbo.encode_punctured(info, perm, coded)
    assert len(out) == coded
    # Erase a twentieth of the channel so the parity and tail have to carry it:
    # a misplaced parity bit then costs the decode, not just a systematic lookup.
    Lc = 4.0 * (1 - 2 * out)
    Lc[rng.random(coded) < 0.05] = 0.0
    assert np.array_equal(turbo.decode_punctured(Lc, N, perm, coded, iters=4), info)


def test_rate_half_is_the_plain_mux():
    rng = np.random.default_rng(1)
    N = 368
    perm, info = turbo.interleaver(N), rng.integers(0, 2, N)
    assert np.array_equal(turbo.encode_punctured(info, perm, 2 * N + 12),
                          turbo.encode(info, perm))


def test_rate_two_thirds_reads_s_p1_s_then_s_p2_s():
    N, coded = 9596, 14406
    perm = np.random.default_rng(2).permutation(N)
    info = np.random.default_rng(3).integers(0, 2, N)
    p1, t1s, t1p = turbo._enc_constituent(info)
    p2, t2s, t2p = turbo._enc_constituent(info[perm])
    out = turbo.encode_punctured(info, perm, coded)
    for g in range(4):
        i = 2 * g
        par = p1[i] if g % 2 == 0 else p2[i]
        assert out[3 * g:3 * g + 3].tolist() == [info[i], par, info[i + 1]]
    tail = [b for j in range(3) for b in (t1s[j], t1p[j], t2s[j], t2p[j])]
    assert out[3 * N // 2:3 * N // 2 + 12].tolist() == tail


@pytest.mark.parametrize("tab,first,pads", [
    (tablegen.BW2300, 9, (0, 0, 4, 0, 2, 4, 3)),
    (tablegen.BW500, 6, (0, 0, 1, 0, 0, 2, 3)),
])
def test_zero_pad_counts(tab, first, pads):
    got = []
    for _, coded, N in tab[first:first + len(pads)]:
        _, _, _, tail = turbo._groups(N, coded)
        got.append(coded - tail - 12)
        rng = np.random.default_rng(N)
        out = turbo.encode_punctured(rng.integers(0, 2, N), rng.permutation(N), coded)
        assert not out[tail + 12:].any()
    assert tuple(got) == pads


def test_rate_one_third_is_refused():
    with pytest.raises(ValueError):
        turbo.encode_punctured(np.zeros(96, int), np.arange(96), 300)

