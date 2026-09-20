# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Memory ARQ: soft combining across repeats of one unacknowledged PACTOR-2 field.

An unacked field is transmitted again, the same bytes under the same mod-4
counter, and `p2rx.BurstMemory` sums the soft windows of those copies before
the trellis sees them. The capability is worthless unless three things hold at
once, so each is planted here as its own arm, on one set of noisy renders:

  * copies that each FAIL alone must decode summed -- the gain itself, with the
    single-shot failures asserted rather than presumed, so the pass cannot come
    from a copy that was never marginal;
  * copies of two DIFFERENT fields must sum to nothing -- the grouping guard.
    When the memory has been fed across a field boundary the CRC is what stands
    between a mixed sum and a delivered wrong field, and this arm is the
    counterexample that fails if anyone relaxes it;
  * a copy from a SWAPPED ARQ cycle must combine when read in its own
    arrangement and must NOT when read in the home one -- the carrier swap
    takes channel rank and stagger with it, so a memory that skips the mapping
    has to go dark here rather than soft;
  * a copy turned by a constant rotation must still combine -- the per-lane
    differential alignment, which is what real carrier offset between repeats
    looks like by the time it reaches the windows.

Noise sigma and seeds are pinned; every decode below is deterministic at SL3.
The corpus fixtures (`rf-corpus/regress`, `pos_p2_sl3_hb9ak*`) hold the other
side of the bargain: on the real recordings, where nearly every burst decodes
single-shot, the memory must change nothing.
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import p2rx, pactor2

FS = 12000
PATH = pactor2.PATHS[2]                      # SL3 short: the level proven off-air
SIGMA = 0.8
"""Noise sigma against a unit-peak render: past the single-shot cliff (0 of 6
seeded copies of field A decode alone) and inside combining's reach."""

A = pactor2.build_field(bytes(range(1, 34)), PATH)
B = pactor2.build_field(bytes(range(200, 233)), PATH)

# One anchor for every render: the bursts all start at the same sample, and the
# marker of the clean home render places them (t, bin pair measured once, here
# pinned). The memory is the unit under test, not acquisition.
ANCHOR_T, BIN_PAIR = 0.335, 12

_cache: dict = {}


def _windows(field: bytes, seed: int, *, render_swapped=False, read_swapped=False):
    key = (field, seed, render_swapped, read_swapped)
    if key not in _cache:
        burst = pactor2.data_burst(field, PATH, swapped=render_swapped, fs=FS)
        base = np.concatenate([np.zeros(FS // 4), burst, np.zeros(FS // 4)])
        x = base + np.random.default_rng(seed).normal(0, SIGMA, base.size)
        _cache[key] = p2rx.burst_window(
            p2rx.bin_phasors(x, FS), BIN_PAIR, ANCHOR_T, PATH.n_symbols,
            1 << PATH.bits_per_cell, swapped=read_swapped)
    return _cache[key]


def _failing(field: bytes, seed: int, **kw):
    w = _windows(field, seed, **kw)
    assert w is not None
    assert p2rx.decode_burst(w, PATH) is None, \
        f"premise broken: seed {seed} decodes single-shot"
    return w


def test_copies_that_fail_alone_decode_summed():
    mem = p2rx.BurstMemory()
    assert mem.add(_failing(A, 1), PATH) is None      # one copy: nothing to sum
    got = mem.add(_failing(A, 2, render_swapped=True, read_swapped=True), PATH)
    assert got == A, "two marginal copies, one per arrangement, must sum to the field"
    # a delivered field clears the memory: the next lone copy stands alone again
    assert mem.add(_failing(A, 3), PATH) is None


def test_copies_of_different_fields_sum_to_nothing():
    mem = p2rx.BurstMemory()
    mem.add(_failing(A, 1), PATH)
    got = mem.add(_failing(B, 3), PATH)
    assert got is None, f"a mixed sum must decode to nothing, got {got!r}"


def test_the_arrangement_mapping_is_load_bearing():
    # the same swapped-cycle copy that combines in test 1, read at home
    mem = p2rx.BurstMemory()
    mem.add(_failing(A, 1), PATH)
    wrong = _windows(A, 2, render_swapped=True, read_swapped=False)
    assert wrong is not None
    assert mem.add(wrong, PATH) is None, \
        "a swapped-cycle copy read in the home arrangement must not combine"


def test_intercopy_rotation_is_absorbed():
    mem = p2rx.BurstMemory()
    mem.add(_failing(A, 1), PATH)
    turned = [w * np.exp(2j * np.pi / 3) for w in _failing(A, 2)]
    assert mem.add(turned, PATH) == A, \
        "a constant rotation between copies is what the alignment is for"
