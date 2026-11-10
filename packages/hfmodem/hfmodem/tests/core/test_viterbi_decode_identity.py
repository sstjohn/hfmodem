# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from time import perf_counter

import numpy as np
import pytest

from hfmodem.brubru import frame
from hfmodem.core.fec import ConvCode, _trellis, viterbi_decode
from hfmodem.shrike import coding, pactor2, placement


def _reference_viterbi_decode(soft: np.ndarray, code: ConvCode, terminated: bool = True,
                   pinned: bool = False) -> np.ndarray:
    """Soft Viterbi: positive values mean zero; zero values are erasures.

    Terminated removes K-1 tail bits. Pinned fixes both boundary states
    at zero; otherwise both boundaries are free.
    """
    K = code.constraint_length
    n_states = code.n_states
    soft = np.asarray(soft, dtype=np.float64)
    if soft.size % 2:
        raise ValueError("expected an even number of coded soft values")
    n_steps = soft.size // 2
    a0_0, a0_1, a1_0, a1_1, pred0, pred1 = _trellis(code)

    if pinned:
        metric = np.full(n_states, -1e12)
        metric[0] = 0.0
    else:
        metric = np.zeros(n_states)
    back = np.zeros((n_steps, n_states), dtype=np.int32)

    for t in range(n_steps):
        r0, r1 = soft[2 * t], soft[2 * t + 1]
        cand0 = metric[pred0] + (a0_0 * r0 + a0_1 * r1)
        cand1 = metric[pred1] + (a1_0 * r0 + a1_1 * r1)
        take1 = cand1 > cand0
        metric = np.where(take1, cand1, cand0)
        back[t] = np.where(take1, pred1, pred0)

    state = 0 if pinned else int(np.argmax(metric))
    bits = np.zeros(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        bits[t] = state & 1          # the input bit is the low bit of the state
        state = int(back[t, state])
    if terminated:
        bits = bits[: n_steps - (K - 1)]
    return bits


CODES = [ConvCode(7, placement.CONV_GENERATORS), frame.CODE, pactor2.CODE,
         coding.CODE_K9]


@pytest.mark.parametrize('code', CODES, ids=['p3-k7', 'brubru', 'pactor2', 'p3-case0'])
@pytest.mark.parametrize('pinned', [False, True])
@pytest.mark.parametrize('terminated', [False, True])
@pytest.mark.parametrize('size', [0, 2, 10, 12, 14, 162, 4096])
def test_viterbi_outputs_are_identical(code, pinned, terminated, size):
    rng = np.random.default_rng(20261030)
    random = rng.normal(size=size)
    bits = rng.integers(0, 2, size=size // 2, dtype=np.uint8)
    encoded = 1.0 - 2.0 * code.encode(bits, terminate=False)
    erased = encoded.copy()
    erased[::3] = 0
    for soft in (random, random[:, None], random.astype(np.float32), random.tolist(),
                 np.zeros(size), np.full(size, -0.0), np.ones(size),
                 np.resize([1., -1., -1., 1.], size), encoded, erased):
        new = viterbi_decode(soft, code, terminated, pinned)
        reference = _reference_viterbi_decode(soft, code, terminated, pinned)
        assert new.dtype == reference.dtype == np.uint8
        assert new.shape == reference.shape
        assert new.tobytes() == reference.tobytes()


@pytest.mark.parametrize('code', CODES)
@pytest.mark.parametrize('pinned', [False, True])
@pytest.mark.parametrize('terminated', [False, True])
def test_odd_soft_length_raises_identically(code, pinned, terminated):
    for function in (viterbi_decode, _reference_viterbi_decode):
        with pytest.raises(ValueError, match='expected an even number of coded soft values'):
            function(np.ones(17), code, terminated, pinned)


@pytest.mark.parametrize('pinned', [False, True])
@pytest.mark.parametrize('terminated', [False, True])
def test_strided_and_extreme_softs_agree(pinned, terminated):
    rng = np.random.default_rng(54)
    for scale in [1e-200, 1e200]:
        soft = (rng.normal(size=2056) * scale)[::2]
        new = viterbi_decode(soft, CODES[0], terminated, pinned)
        reference = _reference_viterbi_decode(soft, CODES[0], terminated, pinned)
        assert new.dtype == reference.dtype
        assert new.shape == reference.shape
        assert new.tobytes() == reference.tobytes()


@pytest.mark.wall_clock
@pytest.mark.parametrize('long_cycle', [False, True], ids=['sl6-short', 'sl6-long'])
def test_viterbi_batches_branch_metrics(long_cycle):
    path = (placement.LONG_PATHS if long_cycle else placement.SPEED_PATHS)[6]
    soft = np.random.default_rng(2026).normal(size=2 * path.n_pairs)
    functions = (viterbi_decode, _reference_viterbi_decode)
    code = CODES[0]
    for function in functions:
        function(soft, code)
    times = [[], []]
    for repeat in range(11):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = perf_counter()
            functions[index](soft, code)
            times[index].append(perf_counter() - start)
    new, reference = map(min, times)
    print(f'viterbi ({long_cycle=}): new={new:.6f}s reference={reference:.6f}s ratio={new / reference:.4f}')
    # Measured 0.44 on macOS and 0.61-0.65 on the Windows build.
    assert new <= reference * 0.8
