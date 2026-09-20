# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Figure-derived trellis plus independent polynomial and field-order checks."""
from itertools import product

import pytest

from hfmodem.shrike.p4coding import (
    encode_components, puncture, rsc_encode, rsc_terminate,
)


def _state(number):
    return number >> 2, (number >> 1) & 1, number & 1


def test_complete_one_step_trellis():
    # Manually enumerate diagram 7.1; columns are (parity, next_state), x=0/1.
    table = [((0, 0), (1, 4)), ((0, 4), (1, 0)),
             ((1, 5), (0, 1)), ((1, 1), (0, 5)),
             ((1, 2), (0, 6)), ((1, 6), (0, 2)),
             ((0, 7), (1, 3)), ((0, 3), (1, 7))]
    for state, branches in enumerate(table):
        for bit, (parity, next_state) in enumerate(branches):
            result = rsc_encode([bit], initial_state=_state(state))
            assert result.parity == (parity,)
            assert result.final_state == _state(next_state)


def test_impulse_and_nonzero_state_vectors():
    result = rsc_encode([1, 0, 0, 0, 0, 0, 0, 0], initial_state=(0, 0, 0))
    assert result.parity == (1, 1, 1, 1, 0, 0, 1, 0)
    assert result.final_state == (1, 0, 0)
    result = rsc_encode([0, 1, 1, 0], initial_state=(1, 0, 1))
    assert result.parity == (1, 1, 0, 1)
    assert result.final_state == (1, 1, 0)


def test_polynomial_identity_for_all_short_inputs():
    # Check transfer-function convolution, not another shift-register encoder:
    # (1+D²+D³)P = (1+D+D³)X, including recursive response after the input.
    for source in product((0, 1), repeat=7):
        x = source + (0,) * 12
        p = rsc_encode(x, initial_state=(0, 0, 0)).parity
        def at(seq, i):
            return seq[i] if i >= 0 else 0
        for i in range(len(x)):
            assert p[i] ^ at(p, i - 2) ^ at(p, i - 3) == (
                x[i] ^ at(x, i - 1) ^ at(x, i - 3))


def test_all_states_terminate_and_streaming_preserves_state():
    for state in product((0, 1), repeat=3):
        tail = rsc_terminate(state)
        check = rsc_encode(tail.systematic, initial_state=state)
        assert len(tail.systematic) == 3
        assert check.parity == tail.parity
        assert check.final_state == tail.final_state == (0, 0, 0)
    tail = rsc_terminate((1, 0, 1))
    assert tail.systematic == (1, 1, 1)
    assert tail.parity == (0, 0, 1)
    source = (1, 1, 0, 1, 0, 0, 1)
    first = rsc_encode(source[:3], initial_state=(0, 1, 1))
    second = rsc_encode(source[3:], initial_state=first.final_state)
    whole = rsc_encode(source, initial_state=(0, 1, 1))
    assert first.parity + second.parity == whole.parity
    assert second.final_state == whole.final_state


def test_pair_requires_explicit_directional_permutation():
    # Non-involution: C2 gathers [x2,x0,x3,x1], never inverse/scatter order.
    first, second = encode_components(
        [1, 0, 0, 0], permutation=[2, 0, 3, 1],
        initial_states=((0, 0, 0), (0, 0, 0)))
    assert first.parity == (1, 1, 1, 1)
    assert second.parity == (0, 1, 1, 1)
    assert second.final_state == (1, 0, 1)


def test_nontrivial_puncturing_and_field_order():
    # Distinct 25-bit fields distinguish component, offset and period errors.
    v0 = tuple(map(int, "1100100101011000010110010"))
    v1 = tuple(map(int, "1010011010010101100011011"))
    v2 = tuple(map(int, "0110100011101010011000100"))
    fields = puncture(v0, v1, v2, rate="5/6")
    assert fields.c1 == (1, 0, 1)  # positions 1,11,21
    assert fields.c2 == (1, 1, 0)  # positions 5,15,25
    assert fields.serialized == v0 + (1, 0, 1, 1, 1, 0)
    fields = puncture(v0, v1, v2, rate="1/2")
    assert fields.c1 == tuple(map(int, "1101100010101"))
    assert fields.c2 == tuple(map(int, "100010001000"))
    assert fields.serialized == v0 + fields.c1 + fields.c2
    assert puncture(v0, v1, v2, rate="1/3").serialized == v0 + v1 + v2


@pytest.mark.parametrize("length,rate,total", [(348, "1/2", 696),
    (580, "5/6", 696), (1160, "5/6", 1392), (1656, "1/3", 4968)])
def test_published_field_dimensions(length, rate, total):
    field = [0] * length
    assert len(puncture(field, field, field, rate=rate).serialized) == total


def test_empty_inputs_and_validation():
    assert rsc_encode([], initial_state=(1, 0, 0)).final_state == (1, 0, 0)
    assert puncture([], [], [], rate="5/6").serialized == ()
    for bits in ([2], [-1], [0.0], ["1"]):
        with pytest.raises(ValueError):
            rsc_encode(bits, initial_state=(0, 0, 0))
    for state in ((0, 0), (0, 0, 0, 0), (0, 2, 0)):
        with pytest.raises(ValueError):
            rsc_terminate(state)
    for order in ([0, 0], [0], [0, 2], [0, -1], [0, 1.0], [False, True]):
        with pytest.raises(ValueError):
            encode_components([0, 1], permutation=order,
                              initial_states=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError):
        puncture([0], [], [0], rate="1/2")
    with pytest.raises(ValueError):
        puncture([], [], [], rate="2/3")
    with pytest.raises(TypeError):
        rsc_encode([0])
    with pytest.raises(TypeError):
        encode_components([0], initial_states=((0, 0, 0), (0, 0, 0)))
