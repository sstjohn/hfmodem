# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Opening headers retain geometry without pretending to deliver a packet."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import p3frame, p3rx, placement, rxfront


def rendered(sl=3, *, long_cycle=False, swapped=False):
    audio = np.pad(placement.link_packet(
        sl, b"header only", 0x03, long_cycle=long_cycle, swapped=swapped),
        (4800, 4800))
    at = (4800 + (placement.protocol_config().pulse().size - 1) // 2
          + p3frame.DATA_OFFSET * rxfront.SPS)
    return audio, at


@pytest.mark.parametrize("sl", [sl for sl in placement.SPEED_PATHS if sl >= 3])
@pytest.mark.parametrize("long_cycle", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
def test_opening_prefix_classifies_wideband_geometry(sl, long_cycle, swapped):
    audio, at = rendered(sl, long_cycle=long_cycle, swapped=swapped)
    sync = rxfront.SyncedRx()
    prefix = audio[:at + sync.WIDEBAND_HEADER_BODY_N]
    header = sync.wideband_header_at(prefix, at)
    assert header is not None
    assert sl in header.levels
    assert header.long_cycle == long_cycle
    assert header.swapped == swapped
    assert abs(header.at + p3frame.DATA_OFFSET * rxfront.SPS - at) <= rxfront.SPS // 4


def test_header_probe_has_no_crc_or_receiver_state_side_effects(monkeypatch):
    audio, at = rendered(long_cycle=True)
    sync = rxfront.SyncedRx()
    sync.packet_at, sync.packet_level, sync.cs_at = 1234, 4, 5678
    sync.misses, sync.tracked, sync.locks, sync.rotation = 1, 2, 3, .5
    before = dict(vars(sync))
    monkeypatch.setattr(p3rx, "decode_at", lambda *a, **k: pytest.fail("header ran CRC"))
    monkeypatch.setattr(sync.memory, "clear", lambda: pytest.fail("header cleared memory"))
    assert sync.wideband_header_at(audio, at).long_cycle
    assert vars(sync) == before


def test_short_prefix_and_missing_header_are_rejected_before_filtering(monkeypatch):
    audio, at = rendered()
    sync = rxfront.SyncedRx()
    monkeypatch.setattr(rxfront, "_sampled_baseband",
                        lambda *a, **k: pytest.fail("incomplete header support"))
    assert sync.wideband_header_at(audio[:at + sync.WIDEBAND_HEADER_BODY_N - 1], at) is None
    assert sync.wideband_header_at(audio, p3frame.DATA_OFFSET * rxfront.SPS) is None


def test_three_position_search_and_transform_extent_are_bounded(monkeypatch):
    audio, at = rendered(long_cycle=True)
    sync = rxfront.SyncedRx()
    calls = []
    sampled = rxfront._sampled_baseband
    read_header = p3rx.header_of

    def bounded_sample(pcm, tones, idx):
        calls.append((len(pcm), len(idx)))
        assert len(pcm) == at + sync.WIDEBAND_HEADER_BODY_N
        return sampled(pcm, tones, idx)

    def bounded_header(z, starts, path):
        assert list(starts) == [at - rxfront.SPS // 4, at, at + rxfront.SPS // 4]
        return read_header(z, starts, path)

    monkeypatch.setattr(rxfront, "_sampled_baseband", bounded_sample)
    monkeypatch.setattr(p3rx, "header_of", bounded_header)
    assert sync.wideband_header_at(audio, at) is not None
    assert len(calls) == 1


def test_original_anchor_gate_is_not_relaxed(monkeypatch):
    audio, at = rendered(long_cycle=True)
    sync = rxfront.SyncedRx()
    header = sync.wideband_header_at(audio, at)
    gate = p3rx.anchor_gate(placement.SPEED_PATHS[3])
    monkeypatch.setattr(p3rx, "header_of", lambda *a, **k: replace(header, fit=gate - 1e-6))
    assert sync.wideband_header_at(audio, at) is None
    monkeypatch.setattr(p3rx, "header_of", lambda *a, **k: replace(header, fit=gate))
    assert sync.wideband_header_at(audio, at) is not None


def test_narrow_header_is_not_claimed_even_at_high_fit(monkeypatch):
    audio, at = rendered()
    monkeypatch.setattr(p3rx, "header_of", lambda *a, **k: SimpleNamespace(fit=1., levels=(1, 2)))
    assert rxfront.SyncedRx().wideband_header_at(audio, at) is None


def test_noise_and_controls_do_not_become_geometry_evidence():
    sync = rxfront.SyncedRx()
    at = 6720
    n = at + sync.WIDEBAND_HEADER_BODY_N
    rng = np.random.default_rng(91427)
    for _ in range(100):
        assert sync.wideband_header_at(rng.normal(0, .1, n), at) is None
    for cs in range(6):
        audio = np.pad(placement.control_signal(cs), (2400, n))
        assert sync.wideband_header_at(audio, at) is None


def test_long_header_does_not_expand_the_early_crc_scope(monkeypatch):
    audio, at = rendered(long_cycle=True)
    sync = rxfront.SyncedRx()
    assert sync.wideband_header_at(audio, at).long_cycle
    monkeypatch.setattr(p3rx, "decode_at", lambda *a, **k: pytest.fail("long early CRC trial"))
    assert sync.wideband_packet_at(audio, at) == (None, False)
