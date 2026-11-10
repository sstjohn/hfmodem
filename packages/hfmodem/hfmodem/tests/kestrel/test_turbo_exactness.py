# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The BCJR sentinel mask, pinned against the one input class that broke it.

When the recursions were vectorised over states, the scalar original's
``alpha[t,s] <= NEG/2`` skip was reproduced only for ``t < 3``, on the reasoning that
it is trellis-fill bookkeeping and every state is reachable from ``t = 3`` on. The
reachability is true. The reasoning is not: the same test is *also* a metric-underflow
guard, and it fires at ``t >= 3`` once the normalised alpha spread reaches the ``NEG``
sentinel scale — empirically around 5.6x ``max|LLR|``, so from ``max|LLR|`` of roughly
9e7. Restricted to ``t < 3`` the two implementations diverged above ~2e8, in 465 of 600
adversarial trials.

Nothing this receiver produces comes close: BW2300 feeds exactly +/-8.0, BW500 feeds
``6*dv/median(|dv|)``, and the largest channel LLR measured across the real captures is
12.0 — seven orders of magnitude of headroom. The guard stays unconditional anyway,
because it costs one gather per trellis step and the alternative is a correctness
argument that depends on the caller's dynamic range.

The vector below is the minimal reproducer that distinguished the two, so it is pinned
rather than described. It fails if the mask is ever narrowed again.
"""
from __future__ import annotations

import numpy as np

from hfmodem.kestrel.coding import turbo

# Minimal input that separates a masked-everywhere alpha recursion from a t<3-only one.
_LS = np.array([0e0, -1e9, -0e0, -2e9, 2e9])
_LP = np.array([1e9, -0e0, 1e9, 0e0, -1e9])
_TAIL_S = np.array([1e9, -0e0, -0e0])
_TAIL_P = np.array([-1e9, 0e0, -0e0])

# What the scalar reference returns. The t<3-only variant returns 4.99999999e+08 here.
_EXPECTED_LE4 = 5.0e8


def test_sentinel_mask_holds_at_high_llr():
    le = turbo._bcjr(_LS, _LP, np.zeros(5), _TAIL_S, _TAIL_P)
    assert le[4] == _EXPECTED_LE4, (
        f"Le[4] = {le[4]:.8e}, expected {_EXPECTED_LE4:.8e}. The NEG-sentinel mask in the "
        "alpha recursion has been narrowed — it must run at every t, not only the "
        "trellis-fill steps.")


def test_the_vector_is_actually_in_the_sensitive_regime():
    """Guard against the pin quietly becoming vacuous — if someone rescales the vector,
    it stops testing what it is named for."""
    assert max(np.abs(_LS).max(), np.abs(_LP).max()) >= 1e8


def test_ordinary_llr_magnitudes_are_unaffected():
    """The regime the receiver actually operates in, for contrast: nothing here is near
    the sentinel scale, and the decoder is ordinary."""
    rng = np.random.default_rng(3)
    ls = rng.standard_normal(64) * 6.0
    le = turbo._bcjr(ls, rng.standard_normal(64) * 6.0, np.zeros(64),
                     rng.standard_normal(3) * 6.0, rng.standard_normal(3) * 6.0)
    assert np.all(np.isfinite(le))
    assert np.abs(le).max() < 1e6


def test_early_termination_requires_two_consecutive_passes():
    """A 16-bit CRC evaluated once per iteration gets `iters` independent chances to
    accept noise instead of one. Requiring the same bits to check twice running is what
    keeps the false-accept rate below where it was before early termination existed."""
    seen: list[int] = []

    def check(_bits) -> bool:
        seen.append(1)
        return True                      # "passes" every single iteration

    rng = np.random.default_rng(5)
    n = 368                              # the real BW500 block; the default perm needs it
    lc = rng.standard_normal(2 * n + 12) * 4.0
    turbo.decode_from_coded_llr(lc, n, iters=12, check=check)
    assert len(seen) >= 2, "stopped on a single CRC pass; one accept is enough to be wrong"


def test_check_none_runs_the_full_schedule():
    """The default must stay exactly what it was before the hook existed."""
    rng = np.random.default_rng(7)
    n = 368
    lc = rng.standard_normal(2 * n + 12) * 4.0
    a = turbo.decode_from_coded_llr(lc, n, iters=12)
    b = turbo.decode_from_coded_llr(lc, n, iters=12, check=lambda _b: False)
    assert np.array_equal(a, b)
