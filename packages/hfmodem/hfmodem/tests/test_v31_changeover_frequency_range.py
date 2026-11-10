# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Cold changeover edges are local to the session's CRC acquisition path."""
import pytest

from hfmodem.shrike import onair, p3acquire


@pytest.mark.parametrize("retained", [0.0, 4.2, -100.0, 100.0, -95.8])
def test_only_two_cold_edge_hypotheses_are_added(retained):
    session = onair._SessionRx.__new__(onair._SessionRx)
    session.p3_receive_offset_hz = retained
    shared = tuple(p3acquire.OFFSETS_HZ)
    got = session._p3_coarse_offsets()
    original = (retained, *(hz for hz in shared if hz != retained))
    assert got[:len(original)] == original
    assert got[0] == retained
    assert -100.0 in got and 100.0 in got
    assert len(got) == len(set(got))
    assert len(got) - len(original) <= 2
    assert tuple(p3acquire.OFFSETS_HZ) == shared
    assert shared == (0.0, -25.0, 25.0, -50.0, 50.0, -75.0, 75.0)
