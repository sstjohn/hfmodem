# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The Reed-Solomon code of MIL-STD-188-110 Appendix B: (15,11) over GF(16),
shortened."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import rs16


def _word(rng, k):
    return rs16.encode([int(v) for v in rng.integers(0, 16, k)])


@pytest.mark.parametrize("k", [3, 10, 11])
def test_code_words_check(k):
    rng = np.random.default_rng(k)
    for _ in range(50):
        cw = _word(rng, k)
        assert len(cw) == k + 4
        assert not any(rs16.syndromes(cw))


def test_the_field_is_x4_x_1():
    # alpha^4 = alpha + 1
    assert rs16.EXP[4] == 0b0011
    assert all(rs16.mul(a, rs16.div(1, a)) == 1 for a in range(1, 16))


@pytest.mark.parametrize("k", [3, 10])
def test_two_errors_are_corrected(k):
    rng = np.random.default_rng(10 + k)
    for _ in range(100):
        cw = _word(rng, k)
        bad = list(cw)
        for i in rng.choice(len(cw), 2, replace=False):
            bad[i] ^= int(rng.integers(1, 16))
        got = rs16.decode(bad)
        assert got is not None and list(got[0]) == list(cw) and got[1] == 2


@pytest.mark.parametrize("k", [3, 10])
def test_four_erasures_are_filled(k):
    rng = np.random.default_rng(20 + k)
    for _ in range(100):
        cw = _word(rng, k)
        er = [int(i) for i in rng.choice(len(cw), 4, replace=False)]
        bad = list(cw)
        for i in er:
            bad[i] = int(rng.integers(0, 16))
        got = rs16.decode(bad, erasures=er)
        assert got is not None and list(got[0]) == list(cw)


def test_three_errors_are_not_silently_accepted_as_clean():
    rng = np.random.default_rng(3)
    for _ in range(100):
        cw = _word(rng, 3)
        bad = list(cw)
        for i in rng.choice(7, 3, replace=False):
            bad[i] ^= int(rng.integers(1, 16))
        got = rs16.decode(bad)
        assert got is None or list(got[0]) != list(bad)
