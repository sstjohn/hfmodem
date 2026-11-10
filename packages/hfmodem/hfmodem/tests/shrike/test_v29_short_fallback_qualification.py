# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Opt-in short hypothesis remains bounded and CRC qualified."""
from dataclasses import replace
import numpy as np
import pytest
from hfmodem.shrike import p3frame, p3rx, placement, rxfront


def rendered(sl=3, long_cycle=False):
    audio = np.pad(placement.link_packet(sl, b'v29 guard', 0x21,
                                       long_cycle=long_cycle), (4800, 4800))
    at = (4800 + (placement.protocol_config().pulse().size - 1)//2
          + p3frame.DATA_OFFSET*rxfront.SPS)
    return audio, at


@pytest.mark.parametrize('sl,long_cycle', [(3, True), (4, False), (5, False), (6, False)])
def test_credible_other_geometry_never_runs_short_crc(sl, long_cycle, monkeypatch):
    audio, at = rendered(sl, long_cycle)
    monkeypatch.setattr(p3rx, 'decode_at', lambda *a, **k: pytest.fail('other geometry CRC'))
    assert rxfront.SyncedRx().wideband_packet_at(
        audio, at, allow_short_fallback=True, can_decode=lambda: True) == (None, False)


def weak_header(monkeypatch, sync, audio, at):
    header = sync._wideband_header_candidate_at(audio, at)
    header = replace(header, fit=0.1)
    monkeypatch.setattr(sync, '_wideband_header_candidate_at', lambda *a: header)


def test_opt_in_and_final_admission_are_required(monkeypatch):
    audio, at = rendered()
    sync = rxfront.SyncedRx()
    weak_header(monkeypatch, sync, audio, at)
    monkeypatch.setattr(p3rx, 'decode_at', lambda *a, **k: pytest.fail('unadmitted CRC'))
    assert sync.wideband_packet_at(audio, at) == (None, False)
    assert sync.wideband_packet_at(audio, at, allow_short_fallback=True,
                                   can_decode=lambda: False) == (None, False)


def test_more_than_25ms_missing_rejects_before_crc(monkeypatch):
    audio, at = rendered()
    end = at + rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[3])) - 1201
    monkeypatch.setattr(p3rx, 'decode_at', lambda *a, **k: pytest.fail('missing-tail CRC'))
    assert rxfront.SyncedRx().wideband_packet_at(
        audio[:end], at, allow_short_fallback=True) == (None, False)


def test_weak_crc_miss_is_unowned_and_tries_each_arrangement_once(monkeypatch):
    """Two CRCs, one per physical carrier order, and nothing else opens.

    The block is too weak to name the swap, and K0NTS's gateway of 2026-09-15
    alternated it; one trial each is what the field's own CRC can settle. There
    is still no alignment, level or frequency ladder, the clock does not move,
    and a miss claims nothing.
    """
    audio, at = rendered()
    sync = rxfront.SyncedRx()
    weak_header(monkeypatch, sync, audio, at)
    before = (sync.packet_level, sync.tracked, sync.misses, sync.rotation)
    calls = []
    monkeypatch.setattr(p3rx, 'decode_at', lambda *a, **k: calls.append((a, k)))
    assert sync.wideband_packet_at(audio, at, allow_short_fallback=True) == (None, False)
    assert len(calls) == 2
    assert {k['header'].swapped for _, k in calls} == {False, True}
    assert {a[1] for a, _ in calls} == {at}
    assert (sync.packet_level, sync.tracked, sync.misses, sync.rotation) == before


def test_weak_crc_success_is_exact_retained_short_row(monkeypatch):
    audio, at = rendered()
    sync = rxfront.SyncedRx()
    weak_header(monkeypatch, sync, audio, at)
    ev, owned = sync.wideband_packet_at(audio, at, allow_short_fallback=True)
    assert owned and ev is not None
    assert ev.start == at and ev.cycle_long is False
    assert ev.packet == (3, 0x21, b'v29 guard', True)
    assert 'short fallback' in ev.text


@pytest.mark.parametrize('kind', ['quiet', 'noise', 'controls'])
def test_false_inputs_never_deliver_crc_packet(kind):
    rng = np.random.default_rng(290914)
    waves = [np.zeros(43000)] if kind == 'quiet' else (
        [rng.normal(0, .1, 43000) for _ in range(40)] if kind == 'noise' else
        [np.pad(placement.control_signal(cs), (2400, 48000)) for cs in range(6)])
    for audio in waves:
        assert rxfront.SyncedRx().wideband_packet_at(
            audio, 6720, allow_short_fallback=True)[0] is None
