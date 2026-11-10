# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Startup frequency coverage must not turn into a wide pre-key search."""
import numpy as np
import pytest

from hfmodem.shrike import onair, p3acquire, placement
from hfmodem.tests.shrike.test_entry_answer import _Session


@pytest.mark.parametrize("retained", [0.0, -15.3, -25.0, 75.0])
def test_cold_search_covers_every_entry_frequency_without_repeating(retained):
    rx = _Session(entry_pending=True).rx
    rx.p3_receive_offset_hz = retained
    earlier = set(rx._p3_coarse_offsets()) | set(rx._p3_fine_offsets())
    cold = rx._p3_cold_offsets()
    assert len(cold) == len(set(cold))
    assert not earlier.intersection(cold)
    assert set(p3acquire.ENTRY_CONTROL_OFFSETS_HZ) <= earlier | set(cold)


@pytest.mark.parametrize("offset", [-90., -65., -40., -15., 15., 40., 65., 90.])
def test_cold_changeover_between_coarse_bins_delivers_crc_payload(offset):
    session = _Session(entry_pending=True)
    audio = np.pad(placement.changeover_packet(b"RMS", 0), (4800, 4800))
    session.rx.new_cycle()
    onair._scan_frame(session.rx, p3acquire.compensate(audio, -offset), 0)
    assert bytes(session.host.channel(session.host.ptchn).rx) == b"RMS"
    assert session.rx._p3_row0 is not None


@pytest.mark.parametrize("tracked, acquired, room, expected", [
    (False, False, True, True),
    (True, False, True, False),
    (False, True, True, False),
    (False, False, False, False),
])
def test_complete_search_requires_cold_early_scan_and_its_own_reserve(
        monkeypatch, tracked, acquired, room, expected):
    rx = _Session(entry_pending=True).rx
    rx._tracked_only = tracked
    rx._p3_row0 = 4800 if acquired else None
    calls = []

    def acquire(audio, *, offsets):
        calls.append(tuple(offsets))
        return None

    monkeypatch.setattr(p3acquire, "changeover", acquire)
    monkeypatch.setattr(rx, "_p3_acquisition_fits",
                        lambda reserve: room or reserve < rx.P3_COLD_RESERVE_S)
    assert rx._p3_changeover_packet(np.zeros(60000)) is None
    assert (rx._p3_cold_offsets() in calls) == expected
    assert rx._p3_row0 == (4800 if acquired else None)


def test_no_new_frequency_is_committed_from_a_head_without_a_body(monkeypatch):
    rx = _Session(entry_pending=True).rx
    # The acquisition can return a coherent head even when its body fails.
    from types import SimpleNamespace
    candidate = SimpleNamespace(event=SimpleNamespace(packet=None), offset_hz=-15.)
    monkeypatch.setattr(p3acquire, "changeover", lambda *a, **k: candidate)
    assert rx._p3_changeover_packet(np.zeros(60000)) is None
    assert rx.p3_receive_offset_hz == 0.0
    assert rx._p3_row0 is None
