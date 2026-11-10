# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The dial convention. No regulator appears here — see test_regulatory.py."""
from __future__ import annotations

import pytest

from hfmodem.core import band as B


def test_the_dial_is_the_centre_minus_1500():
    assert B.dial_hz(7_101_500) == 7_100_000
    assert B.centre_hz(7_100_000) == 7_101_500
    assert B.DIAL_OFFSET_HZ == 1500


@pytest.mark.parametrize("centre", [3_585_000, 7_101_500, 14_105_000, 28_120_000])
def test_dial_arithmetic_round_trips(centre):
    assert B.centre_hz(B.dial_hz(centre)) == centre


def test_the_audio_centre_is_a_different_fact_that_shares_a_number():
    """1500 Hz is also the modulators' passband centre. That the two agree is a
    property of the sideband convention, and collapsing them would couple a
    waveform change to the dial convention."""
    from hfmodem.shrike import spec
    assert spec.CENTER_FREQ_HZ == 1500.0
    assert B.DIAL_OFFSET_HZ == 1500
