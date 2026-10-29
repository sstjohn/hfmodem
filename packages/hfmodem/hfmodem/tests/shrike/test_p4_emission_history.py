# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.shrike import p4rx
from hfmodem.shrike.p4receiver import (
    EMISSION_LOOKBACK_S, Emission, Expect, Kind, Observation, P4Receiver, Role,
)

FS = 48000


@pytest.mark.parametrize('scan', ['legacy', 'incremental'])
def test_long_session_bounds_emissions_without_losing_current_reply(scan):
    rx = P4Receiver(FS, admitted=True, cfo_hz=0., scan=scan)
    audio = np.zeros(128, np.float32)
    for slot in range(3000):
        at = slot*60000
        em = Emission(Kind.SHORT, Role.ISS, at, .758)
        rx.note_emission(em)
        assert rx.read_window(audio, at+round(.758*FS)) == []
        assert len(rx.emissions) <= 15
        obs = Observation(Kind.WORD, FS, at, 0, at+round(.935*FS), .245,
                          0., 1., word='X27')
        assert rx._response_target(obs) == at
        assert rx.last_emission is em


def test_retention_includes_oldest_train_reply_in_six_second_audio():
    rx = P4Receiver(FS)
    em = Emission(Kind.SHORT, Role.ISS, 0, .758, train=True)
    rx.note_emission(em)
    rx.train_lease.max_slot = 9
    at = round((.935+9*1.25+.035)*FS)
    obs = Observation(Kind.WORD, FS, at, 0, at, .245, 0., 1., word='Y27')
    rx._trim_emissions(at+6*FS)
    assert rx._response_target(obs) == 0
    assert rx._word_horizon(obs) == .300
    boundary = round((6+EMISSION_LOOKBACK_S)*FS)
    rx._trim_emissions(boundary)
    assert rx.emissions == [em]
    rx._trim_emissions(boundary+1)
    assert rx.emissions == []


def test_waveform_mask_outlives_response_lookback():
    rx = P4Receiver(FS, scan='incremental')
    em = Emission(Kind.LONG, Role.ISS, 0, 30.)
    rx.note_emission(em)
    margin = round(p4rx.MATCHED_DELAY*FS/p4rx.SAMPLE_RATE)
    boundary = 36*FS+margin
    rx._trim_emissions(boundary)
    assert rx.emissions == [em]
    assert rx._scan_gaps(np.zeros(1), 30*FS+margin-1) == [(0, 1)]
    rx._trim_emissions(boundary+1)
    assert rx.emissions == []


def test_session_facts_survive_history_eviction_and_reset_for_new_call():
    rx = P4Receiver(FS)
    rx.enable_hold()
    rx.note_emission(Emission(Kind.WORD, Role.IRS, 0, .245))
    at = 30*FS
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, at, .758))
    assert all(e.role == Role.ISS for e in rx.emissions)
    rx.clock.expects = Expect.PACKET
    assert rx._normal_eligible()
    rx._trim_emissions(60*FS)
    assert rx.emissions == []
    obs = Observation(Kind.WORD, FS, at, 0, at, .245, 0., 1.,
                      word='Y27', unexpected=True)
    rx._live(rx.register(obs))
    assert rx.peer_onset is None
    rx.new_call()
    assert not rx._had_irs_emission and rx.last_emission is None
    assert rx._emission_revision == 0 and rx.emissions == []


def test_scan_revision_tracks_emissions_gaps_and_cfo_without_redundant_changes():
    rx = P4Receiver(FS, cfo_hz=0., scan='incremental')
    audio = np.zeros(128, np.float32)
    em = Emission(Kind.SHORT, Role.ISS, 0, .758)
    rx.note_emission(em)
    rx._retain_scan(audio, 0)
    initial = rx._scan_generation_number
    rx.note_emission(em)
    rx._retain_scan(audio, 0)
    assert rx._scan_generation_number == initial
    rx.note_gap(1, 2)
    rx._retain_scan(audio, 0)
    assert rx._scan_generation_number == initial+1
    rx._scan_last_block = 10
    rx.cfo_hz = 1.
    rx._retain_scan(audio, 0)
    assert rx._scan_generation_number == initial+2 and rx._scan_rescan == 10
    rx._trim_emissions(30*FS)
    rx._retain_scan(audio, 0)
    assert rx._scan_generation_number == initial+3


def test_opening_handoff_does_not_resubmit_evicted_probe_keys():
    rx = P4Receiver(FS)
    opening = [Emission(Kind.SHORT, Role.ISS, slot*60000, .758, train=True)
               for slot in range(30)]
    for em in opening:
        rx.note_emission(em)
    last, clock, lease = rx.last_emission, rx.clock, rx.train_lease
    revision = rx._emission_revision
    for em in opening:
        if not any(e.stream_sample == em.stream_sample for e in rx.emissions):
            rx.note_emission(em)
    assert rx.last_emission is last and rx.clock is clock and rx.train_lease is lease
    assert rx._emission_revision == revision and len(rx.emissions) == 15
