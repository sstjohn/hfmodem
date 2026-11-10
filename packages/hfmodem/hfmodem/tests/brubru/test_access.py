# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.brubru.access import Access, RATE


def feed(access, seconds, kind='quiet', amplitude=1):
    start = access._next if access._next is not None else access.evaluated
    rng = np.random.default_rng(123)
    for at in range(start, start + round(seconds * RATE), access.WINDOW):
        noise = rng.normal(0, 0.001 * amplitude, access.WINDOW)
        if kind == 'tone':
            noise += 0.1 * np.cos(2 * np.pi * 1500 * np.arange(access.WINDOW) / RATE)
        access.feed(at, noise)


def test_shape_startup_and_sixty_seconds_of_occupancy():
    access = Access()
    feed(access, 60, 'tone')
    assert access.floor is None and not access.clear()
    feed(access, 2.3)
    assert access.clear()
    floor = access.floor
    feed(access, 60, 'tone')
    assert access.stale and access.floor == floor and not access.clear()
    feed(access, 1.9)
    assert access.stale
    feed(access, 0.2)
    assert not access.stale and access.clear()


def test_broadband_startup_qualifies_as_floor_then_lowers():
    access = Access()
    feed(access, 2, amplitude=100)
    assert access.clear()
    high = access.floor
    feed(access, 4)
    assert access.clear() and access.floor < high / 100


def test_shape_twenty_db_above_floor_never_updates_it():
    access = Access()
    feed(access, 3)
    floor = access.floor
    feed(access, 15, 'tone')
    assert not access.clear() and access.floor == floor
    feed(access, 0.1)
    assert not access.clear()
    feed(access, 0.1)
    assert access.clear()


def test_hold_tx_mask_and_post_fault_evidence():
    access = Access()
    feed(access, 3)
    floor = access.floor
    access.hold(True)
    feed(access, 1, amplitude=0)
    assert not access.clear() and access.floor == floor
    access.hold(False)
    access.tx(True, access.evaluated)
    feed(access, 1, 'tone')
    access.tx(False, access.evaluated)
    feed(access, 0.2)
    assert not access.clear()
    feed(access, 0.1)
    assert access.clear()
    access.fault(access.evaluated, 1)
    feed(access, 0.9)
    assert not access.clear(1)
    feed(access, 0.1)
    assert access.clear(1) and not access.clear(2)


@pytest.mark.parametrize('draw,expected', [(62, True), (63, True), (64, False)])
def test_persistence_is_inclusive_and_obeys_slots(draw, expected):
    access = Access(draw=lambda: draw)
    feed(access, 2)
    assert not access.contend(0)
    assert not access.contend(0.099)
    assert access.contend(0.1) == expected
    assert not access.contend(0.11)
