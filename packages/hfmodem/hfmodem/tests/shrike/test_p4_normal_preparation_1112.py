# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Causal normal filter preparation retains the final global pilot fit."""
import numpy as np
import pytest

from hfmodem.shrike import p4longrates, p4rx, p4short
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.shrike.test_p4_long_audio_1111 import FS, render_long_audio


def test_normal_long_nominal_uses_packet_geometry(monkeypatch):
    rx = P4Receiver(FS, admitted=True)
    rx.clock = Clock(Expect.PACKET, FS, .33, 3.75, 0.)
    nominals = []
    monkeypatch.setattr(p4rx, 'audio_to_baseband', lambda *args: np.ones(1000, complex))
    def place(z, nominal, cfo, **kwargs):
        nominals.append(nominal)
        return 0., nominal, cfo
    monkeypatch.setattr(p4short, 'place_long', place)
    rx._normal_long(np.zeros(FS), FS - 2400, partial=True)
    assert nominals == [1440 + p4short.GEOM]


def test_retained_matched_fit_is_the_same_all_training_fit(long_encoder):
    audio, onset = render_long_audio(long_encoder, 5, bytes(range(203)), 0x41,
                                    cfo=.12, seed=111213)
    assert onset == 12137
    rx = P4Receiver(FS, admitted=True, cfo_hz=0.)
    rx.clock = Clock(Expect.PACKET, onset, .33, 3.75, 0.)
    rx._normal_long(audio[:onset+round(.845*FS)], 0, partial=True, alias=True)
    rx.prepare_sf8(audio[:onset+round(3.230*FS)], 0)
    entry = next(row for row in rx._long_cache.values() if row.get('kind') == 'normal')
    finished = rx._finish_normal_long(entry['prepared'], audio[:onset+round(3.300*FS)], 0)
    assert finished is not None
    z, base, placements, matched_rows = finished
    assert base == entry['prepared']['base_origin'] and len(placements) == 3
    assert all(row is not None for row in matched_rows)
    for (_, start, cfo), matched in zip(placements, matched_rows):
        expected = p4longrates.observe_body(z, first_training_start=int(start),
                                          frequency_offset=cfo)
        actual = p4longrates.observe_matched_body(matched/np.max(abs(z)),
            first_training_start=int(start), frequency_offset=cfo, input_samples=len(z))
        assert expected is not None and actual is not None
        # Different FFT lengths and normalization order allow only roundoff.
        rotated = z*np.exp(-2j*np.pi*cfo*np.arange(len(z))/28800)
        np.testing.assert_allclose(matched, p4rx.matched_filter(rotated), rtol=0, atol=1e-13)
        np.testing.assert_allclose(actual.symbols, expected.symbols, rtol=0, atol=1e-13)
        for name in ('gain', 'fitted_noise', 'inverse_energy', 'residual_energy'):
            assert getattr(actual, name) == pytest.approx(getattr(expected, name), rel=1e-12, abs=1e-13)
        assert actual.coordinates == expected.coordinates


def test_causal_normal_preparation_recovers_current_complete_body(long_encoder):
    payload = bytes(range(203))
    audio, onset = render_long_audio(long_encoder, 5, payload, 0x41,
                                    cfo=.12, seed=111213)
    rx = P4Receiver(FS, admitted=True, cfo_hz=0.)
    rx.clock = Clock(Expect.PACKET, onset, .33, 3.75, 0.)
    prefix, _ = rx._normal_long(audio[:onset+round(.845*FS)], 0,
                                partial=True, alias=True)
    assert prefix is not None and prefix.provisional and not prefix.placed
    assert prefix.frame is None and not rx._long_jobs
    # Use the public common preparation opportunity; no fields/decoder results
    # are inserted into the receiver.
    rx.prepare_sf8(audio[:onset+round(3.230*FS)], 0)
    entry = next(row for row in rx._long_cache.values() if row.get('kind') == 'normal')
    assert 'prepared' in entry and not entry['complete']
    assert len(entry['prepared']['rows']) == 3
    assert not rx._long_jobs
    full, _ = rx._normal_long(audio[:onset+round(3.300*FS)], 0, alias=True)
    assert full is not None and full.placed and not full.provisional
    assert full.frame.data == payload and full.frame.status.byte == 0x41
    assert full.speedlevel == 'SL5-long'
    job = next(row for key, row in rx._long_jobs.items() if key[2] == 'normal')
    assert len(job['search'].members) == 24
    assert job['result'].state == 'unique'


def test_changed_overlap_refuses_cached_normal_filter_splice(long_encoder):
    audio, onset = render_long_audio(long_encoder, 5, b'a'*203, 0x41,
                                    cfo=.12, seed=111214)
    rx = P4Receiver(FS, admitted=True, cfo_hz=0.)
    rx.clock = Clock(Expect.PACKET, onset, .33, 3.75, 0.)
    rx._normal_long(audio[:onset+round(.845*FS)], 0, partial=True, alias=True)
    rx.prepare_sf8(audio[:onset+round(3.230*FS)], 0)
    entry = next(row for row in rx._long_cache.values() if row.get('kind') == 'normal')
    changed = audio[:onset+round(3.300*FS)].copy()
    changed[onset+3*FS:] = np.random.default_rng(111214).normal(0, .1, len(changed)-onset-3*FS)
    assert rx._finish_normal_long(entry['prepared'], changed, 0) is None
    assert not rx._long_jobs
