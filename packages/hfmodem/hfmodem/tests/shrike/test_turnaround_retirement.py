# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Expired turnaround observations cannot accumulate during long contacts."""
from hfmodem.shrike.onair import _TurnaroundEvidence


def test_turnaround_keeps_only_its_existing_eight_cycle_admission_window():
    evidence = _TurnaroundEvidence()
    for cycle in range(10000):
        # Both distinct populations have multiple independent cycles, so neither
        # may become a unique placement merely because an old one was discarded.
        answer = evidence.offer(cycle, [1000., 10000.])
        if cycle:
            assert answer is None and evidence.witnessed == 2
        assert len(evidence.candidates) <= 2 * evidence.SPAN
        assert all(cycle - at < evidence.SPAN for at, _ in evidence.candidates)
    # A population expires under the existing protocol window, not an extra
    # timeout. Its replacement can then provide the unique current placement.
    for cycle in range(10000, 10000 + evidence.SPAN):
        answer = evidence.offer(cycle, [1000.])
    assert answer == 1000. and evidence.witnessed == 1
    assert evidence.at == 10000 + evidence.SPAN - 2
    assert evidence.offer(cycle + evidence.SPAN, []) is None
    assert evidence.candidates == [] and evidence.witnessed == 0
