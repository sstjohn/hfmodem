# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Resolved P4 generators only; no reader, decoder, or native oracle execution."""
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import tablegen


# Entire little-endian raw-coordinate arrays, independently verified before
# generator implementation. These hashes are evidence pins, not point values.
POINT_HASHES = {
    6: "24cccd676788da5f956150f1b199f368eebcbebbdc35f262003b544f7e073c0a",
    7: "e640af2763f0cbf06d0d4a00a3f19860acbadd3d1ff15cf2bf35c6b1906fe2ac",
    8: "bcf3c71422f2189699637d78eb1ac8324e01ed322911520ea041e90676f35b67",
    9: "af9d009ff38d3c17c33d54f0ca1f00ced89d596f612df1fee40dfa75a63e9e60",
    10: "c3f84937d1aba4f2d2b9a0766d644044d6d8e6b84b45657a8fdddeca7089ab3a",
}
SYMBOL_HASH = "6faa332dd74cb24fef548447dd60579e6a6bb7c6df84acc78a08d13de1af66dc"


@pytest.mark.parametrize("speedlevel", (6, 7, 8, 9, 10))
def test_label_order_shape_dtype_scale_and_all_raw_bytes(speedlevel):
    points = tablegen.p4_constellation_points(speedlevel)
    assert points.shape == (1 << (speedlevel - 5), 2)
    assert points.dtype.str == "<i2" and points.flags.c_contiguous
    assert tablegen.P4_POINT_SCALE == 4096
    assert sha256(points.tobytes()).hexdigest() == POINT_HASHES[speedlevel]
    assert len(set(map(tuple, points.tolist()))) == len(points)
    assert tablegen.p4_constellation_points(np.int64(speedlevel)) is points


@pytest.mark.parametrize("speedlevel", (6, 7, 8, 9, 10))
def test_cached_constellation_cannot_restore_write_permission(speedlevel):
    points = tablegen.p4_constellation_points(speedlevel)
    before = points.tobytes()
    assert not points.flags.writeable
    with pytest.raises(ValueError):
        points.setflags(write=True)
    with pytest.raises(ValueError):
        points[0, 0] = 0
    assert tablegen.p4_constellation_points(speedlevel).tobytes() == before


def test_symbol_gather_is_the_existing_direct_4968_walk():
    order = tablegen.p4_symbol_permutation()
    assert order.shape == (4968,) and order.dtype == np.dtype(np.intp)
    assert order.flags.c_contiguous
    assert np.array_equal(np.sort(order), np.arange(4968))
    # Fixed reference is direct, not the inverse; serialize the same portable
    # u16 extent used by the independent reference proof.
    assert sha256(order.astype("<u2").tobytes()).hexdigest() == SYMBOL_HASH
    assert sha256(np.argsort(order).astype("<u2").tobytes()).hexdigest() != SYMBOL_HASH
    existing = tablegen.INTERLEAVERS["0xb81c0_4968"]()
    assert np.array_equal(order, existing.astype(np.intp))


def test_cached_symbol_gather_cannot_restore_write_permission():
    order = tablegen.p4_symbol_permutation()
    before = order.tobytes()
    assert not order.flags.writeable
    with pytest.raises(ValueError):
        order.setflags(write=True)
    with pytest.raises(ValueError):
        order[0] = 1
    assert tablegen.p4_symbol_permutation() is order
    assert order.tobytes() == before


@pytest.mark.parametrize("invalid", (None, True, np.bool_(False), 5, 11, 6.0,
                                     np.float64(7), "6", 6 + 0j))
def test_public_validation_precedes_cached_aliases(invalid):
    # Populate both Python-int and NumPy-integer routes first. A float/bool
    # equal to a cache key still must reach the public validation check.
    for speedlevel in range(6, 11):
        tablegen.p4_constellation_points(speedlevel)
        tablegen.p4_constellation_points(np.int64(speedlevel))
    with pytest.raises(ValueError, match="integer from 6 through 10"):
        tablegen.p4_constellation_points(invalid)


def test_optional_full_original_arrays_have_explicit_reference_root():
    value = os.environ.get("P4_CONSTANT_REFERENCE_ROOT")
    if not value:
        pytest.skip("optional full-array reference root not supplied")
    root = Path(value)
    if not root.is_absolute():
        pytest.fail("reference root must be explicit and absolute")
    reference = root / "p4-normal-long.npz"
    if not reference.is_file():
        pytest.fail("declared original normal-LONG reference is absent")
    with np.load(reference, allow_pickle=False) as archive:
        for speedlevel in range(6, 11):
            original = archive[f"points_{speedlevel}"]
            generated = tablegen.p4_constellation_points(speedlevel)
            assert original.dtype.str == generated.dtype.str
            assert original.shape == generated.shape
            assert original.tobytes() == generated.tobytes()
        original = archive["symbols"]
        generated = tablegen.p4_symbol_permutation()
        assert np.array_equal(original.astype(np.intp), generated)
