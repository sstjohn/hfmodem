# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tables, each checked against something other than itself.

The sequence generators are checked against a second formulation written from
the generator polynomials alone: a register reads as a linear recurrence on
the bits it emits, which shares nothing with the register code in ``spec`` but
the polynomial, so an edit to either that changes the sequence fails here.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import fec, interleave, spec


def _recurrence(seed, lags, n):
    """Bits of x^a + x^b + 1 as b[k] = b[k-a] ^ b[k-b], from ``seed``."""
    b = list(seed)
    while len(b) < n:
        b.append(b[-lags[0]] ^ b[-lags[1]])
    return b[:n]


def test_4285_sync_follows_its_polynomial():
    """x^5 + x^2 + 1: the first five bits out are the register as loaded,
    last stage first (1 1 0 1 0 read backwards), and 1 is sent as 180 degrees."""
    bits = _recurrence([0, 1, 0, 1, 1], (3, 5), 80)
    assert list(spec.sync4285()) == [4 * x for x in bits]


def test_4285_randomiser_follows_its_polynomial():
    """x^9 + x^4 + 1 from all ones, three bits out a symbol, the first of the
    three the least significant."""
    bits = _recurrence([1] * 9, (5, 9), 3 * 176)
    want = [bits[3 * k] + 2 * bits[3 * k + 1] + 4 * bits[3 * k + 2] for k in range(176)]
    assert list(spec.scramble4285()) == want
    assert spec.scramble4285()[0] == 7          # all ones reads as tribit 7


def test_188_110_randomiser_generator_is_maximal_length():
    """x^12 + x^6 + x^4 + x + 1 is primitive, so the register as written walks
    all 4095 nonzero states before repeating. This checks the cycle, not
    the seed orientation or which three stages the standard intends to read."""
    reg, seen = spec.SCRAMBLE110_SEED, set()
    while reg not in seen:
        seen.add(reg)
        carry = (reg >> 11) & 1
        reg = (reg << 1) & 0xFFF
        if carry:
            reg ^= 0x053
    assert len(seen) == 4095
    s = spec.scramble110()
    assert s.size == 160 and set(s) <= set(range(8))
    assert len(set(s)) == 8


def test_the_code_is_133_171_read_from_the_other_end():
    def rev7(x):
        return int(f"{x:07b}"[::-1], 2)
    assert {rev7(m) for m in spec.CONV_MASKS} == {0o133, 0o171}


@pytest.mark.parametrize("t", range(8))
def test_channel_patterns(t):
    p = np.array(spec.CHANNEL_PATTERNS[t])
    assert p.size == 32 and set(p) <= {0, 4}
    # 0-3 are the four Walsh rows, 4-7 their complements on alternate quads.
    base = np.array(spec.CHANNEL_PATTERNS[t & 3])
    if t < 4:
        assert (p == np.tile(p[:4], 8)).all()
    else:
        assert (p[:4] == base[:4]).all() and (p[4:8] == (base[:4] + 4) % 8).all()


def test_d1_d2_name_each_mode_once():
    """Table XV: every D1/D2 pair names one rate and interleave. The zero
    modes have none of their own (5.3.2.3.7.2.1)."""
    named = [m for m in spec.MODES110.values() if m.interleave != "zero"]
    assert len(spec.MODES110_BY_D) == len(named)
    assert spec.MODES110_BY_D[(7, 5)].name == "75S"
    assert spec.MODES110_BY_D[(5, 5)].name == "75L"


@pytest.mark.parametrize("mode", [m for m in spec.MODES110.values() if m.rows],
                         ids=lambda m: m.name)
def test_188_110_interleaver_is_a_permutation(mode):
    o = interleave.block110_order(mode.rows, mode.cols)
    assert np.array_equal(np.sort(o), np.arange(mode.rows * mode.cols))
    # One block fills a whole number of frames.
    from hfmodem.bateleur.mil110 import coded_bits_per_frame
    assert mode.block_bits % coded_bits_per_frame(mode) == 0
    # 0.6 s or 4.8 s of coded bits, as the preamble lengths promise.
    rate = mode.rate * 2 * mode.repeat
    assert mode.block_bits / rate == pytest.approx(0.6 if mode.interleave == "short" else 4.8)


@pytest.mark.parametrize("cols", [18, 36, 72, 144, 288, 576])
def test_the_other_fetch_reading_is_not_a_permutation(cols):
    """Why 5.3.2.3.5 says "one larger than the value it had when the row
    number was last zero": restarting one column right of where the last pass
    *ended* instead misses cells."""
    rows = 40
    visited = set()
    col = 0
    for _ in range(cols):
        for r in range(rows):
            visited.add((r, col))
            if r < rows - 1:
                col = (col - 17) % cols
        col = (col + 1) % cols
    assert len(visited) < rows * cols


@pytest.mark.parametrize("inc", [1, 2, 4, 12, 24, 48])
def test_4285_interleaver_round_trips(inc):
    rng = np.random.default_rng(inc)
    n = 32 * (31 * inc + 20)
    x = rng.integers(0, 2, n)
    y = interleave.interleave4285(x, inc)
    back = interleave.deinterleave4285(y, inc)
    assert np.array_equal(back, x[: back.size])
    assert back.size == n - 31 * inc * 32


def test_4285_interleaver_spreads_a_group_by_the_row_delay():
    x = np.zeros(32 * 400, dtype=int)
    x[:32] = 1
    y = interleave.interleave4285(x, 12).reshape(-1, 32)
    groups, positions = np.nonzero(y)
    assert sorted(positions) == list(range(32))
    assert all(g == 12 * c for g, c in zip(groups, positions))


def test_viterbi_undoes_the_encoder_and_corrects():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 2000)
    coded, _ = fec.encode(bits)
    soft = 1.0 - 2.0 * coded
    flip = rng.choice(soft.size, 60, replace=False)
    soft[flip] *= -1
    assert np.array_equal(fec.viterbi(soft), bits)


def test_viterbi_takes_erasures():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 1000)
    coded, _ = fec.encode(bits)
    soft = (1.0 - 2.0 * coded).reshape(-1, 4)
    soft[:, 3] = 0.0                                # 4285's rate 2/3
    assert np.array_equal(fec.viterbi(soft.ravel()), bits)
