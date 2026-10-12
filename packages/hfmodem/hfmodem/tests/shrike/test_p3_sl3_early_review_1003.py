# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Successive physical holds retain staging, fallback and final-slice admission."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, rxfront
from hfmodem.tests.shrike.test_p3_irs_sl3_ack_1003 import (
    audio_callback_load as audio_callback_load, budget, crop, listen_current, receiver, stage_sl3)


def test_staged_central_samples_match_independent_matched_filter():
    audio, p = crop(201)
    s = receiver(p['row'])
    stage_sl3(s, audio, p['origin'])
    key, prepared = s.rx._p3_early_sl3
    corrected = s.rx._corrected(audio[key[1]-p['origin']:])
    # Isolate the grid reduction from finite-window carrier correction.
    independent = rxfront._EarlySL3(prepared.at)
    actual = independent.baseband(corrected)
    expected = rxfront._sampled_baseband(corrected, independent.tones, independent.idx)
    for tone in independent.tones:
        np.testing.assert_allclose(actual[tone][independent.idx],
                                   expected[tone][independent.idx], rtol=0, atol=1e-12)


def test_changed_final_header_cannot_spend_staged_reserve_on_new_grid(monkeypatch):
    audio, p = crop(201)
    s = receiver(p['row'])
    stage_sl3(s, audio, p['origin'])
    key, prepared = s.rx._p3_early_sl3
    corrected = s.rx._corrected(audio[key[1]-p['origin']:])
    header = s.rx.sync._wideband_header_candidate_at(corrected, prepared.at)
    monkeypatch.setattr(s.rx.sync, '_wideband_header_candidate_at',
                        lambda *args: replace(header, at=header.at+120))
    monkeypatch.setattr(prepared, 'baseband',
                        lambda *args: pytest.fail('changed header consumed stale grid'))
    assert s.rx.sync.wideband_packet_at(corrected, prepared.at,
        prepared=prepared, can_decode=lambda: True) == (None, False)
    assert not s.packets and not s.host.peer.sent


@pytest.mark.parametrize('following', [202, 206])
def test_sl3_promotion_retains_next_reader_at_18ms(monkeypatch, following):
    _, p = crop(201)
    s = receiver(p['row'])
    for index, hold in enumerate((201, following)):
        if index:
            s.rx.new_cycle()
        audio, p = crop(hold)
        with monkeypatch.context() as patch:
            spent, trace = budget(s, patch, 18.)
            stage_sl3(s, audio, p['origin'])
            onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
        assert len(s.packets) == index+1
        assert len(s.host.peer.sent) == index+1
        assert s.host.peer.sent[-1] == ('cs', arq.CS_REQUEST)
        assert trace == (['sl2_packet_at'] if hold == 202 else ['wideband_packet_at'])
        assert spent[0]+6 <= 18.
        ev = s.packets[-1]
        assert ev.packet[:2] == (2 if hold == 202 else 3, 0x23)
        assert ev.packet[3]
        assert s.rx._p3_expected() == (2, 3)
        assert abs(s.rx._p3_delivered_at-p['row']) <= rxfront.SPS
        delivered = s.host.rcvd_total
        onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
        assert len(s.packets) == index+1 and len(s.host.peer.sent) == index+1
        assert s.host.rcvd_total == delivered


@pytest.mark.parametrize('following', [202, 206])
def test_successive_holds_charge_final_listener_slice(audio_callback_load, following):  # noqa: F811
    costs = []
    for _ in range(20):
        _, first = crop(201)
        s = receiver(first['row'])
        for index, hold in enumerate((201, following)):
            if index:
                s.rx.new_cycle()
            audio, p = crop(hold)
            elapsed = listen_current(s, audio, p['origin'])
            costs.append(elapsed)
            assert len(s.packets) == index+1
            assert len(s.host.peer.sent) == index+1
            assert s.packets[-1].packet[:2] == (2 if hold == 202 else 3, 0x23)
            assert s.packets[-1].packet[3]
            assert elapsed+1000*onair.P3_DECODE_RESERVE_S <= 18.
            assert s.rx._p3_acquisition_fits(0)
    print(f'SUCCESSIVE_METRIC following={following} total_max_ms={max(costs):.3f} '
          f'margin_ms={12-max(costs):.3f} acks={len(costs)}/{len(costs)}')
