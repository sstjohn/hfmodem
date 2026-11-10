# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""RETAINED-EXPIRY-DELTA and RECORDED-START narrow receiver controls.

Synthetic real PCM; reader/converter wrappers delegate. Cleanup attachments are
explicit preconditions on the genuinely acquired observation, not decoding or
source evidence. No grant eligibility or recorded-history repair is installed.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from hfmodem.shrike import p4rx, p4robusttx, p4chirp
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4receiver import Emission, Expect, Role
from hfmodem.tests.shrike.test_p4_retained_chirp_prefix_1012 import (
    FS as FS, AT as AT, acquire as acquire, receiver as receiver, pcm as pcm, calls as calls)


@pytest.mark.parametrize('transition', ['watch', 'emission'])
def test_stale_real_acquisition_expires_once_and_cleans_owned_attachments(pcm, calls, transition):
    rx = receiver()
    h, key, reg = acquire(rx, pcm)
    old = reg.observation
    if transition == 'watch':
        rx.sync_peer_watch(SimpleNamespace(generation=7, onset=h-180000,
            cycle=180000, cfo_hz=0., attempts=0, last_attempt=None))
        watch = rx.peer_watch
        with rx.scoped_clock(watch.clock, watch_generation=watch.generation):
            current = rx.read_window(pcm[h+1:], h+1, whole=True)
            assert not any(r.outcome == Registered.EXPIRED for r in current)
        rx.finish_peer_watch(watch, h)
        assert watch.clock.next_due == h+180000
        scope = watch.clock
        primary = rx.clock
    else:
        wave = p4robusttx.robust_word('X27')
        rx.note_emission(Emission(Kind.WORD, Role.ISS, h+163200, len(wave)/FS))
        primary = scope = rx.clock
        assert primary.expects == Expect.WORD
    # Lease/body associations with this real provisional root provide a
    # nonvacuous cleanup contract, without planting a decoded frame.
    rx.arm_close_lease(rx.last_emission.stream_sample if rx.last_emission else 0,
                       end=h+10*FS)
    rx.close_lease.first_request = h
    rx._body_heads[h] = old
    rx.pending_body = old
    lease = rx.close_lease
    fields = replace(scope)
    origin = h+round(3.355*FS)+1
    expired = []
    for _ in range(2):
        with rx.scoped_clock(scope):
            rows = rx.read_window(pcm[origin:], origin, whole=True)
            expired.extend(r for r in rows if r.observation is old and r.outcome == Registered.EXPIRED)
            assert rx.clock is scope and rx.clock == fields
        assert rx.clock is primary
    assert len(expired) == 1 and expired[0].replaced is old
    assert old not in rx.observations
    assert lease.first_request is None and h not in rx._body_heads and rx.pending_body is None
    assert rx._chirp_headers[key] is None
    assert rx.clock is primary and calls['decode'] == []
    assert not any(e.get('event') == 'chirp header unread' and e.get('at') != h for e in rx.live_events)


@pytest.mark.parametrize('excess', [0, 1, 200])
def test_empty_conversion_slice_leaves_existing_entry_unchanged(monkeypatch, excess):
    rx = receiver()
    entry = dict(kind='normal', obs=object(), coherence=.8, complete=False,
                 end=AT-4000, alias=False, scores=None, iterations=0)
    rx._long_cache[(AT, 0.)] = entry
    audio = np.ones(20)
    lo = 20+excess
    origin = AT-round(.050*FS)-lo
    assert origin+len(audio) > entry['end']
    before = dict(entry)
    monkeypatch.setattr(p4rx, 'audio_to_baseband', lambda *a, **k: pytest.fail('empty converted'))
    assert rx._normal_long(audio, origin, alias=True) == (None, 0.)
    assert rx._long_cache == {(AT, 0.):entry} and entry == before
    assert rx.observations == [] and rx.registrations == []


def test_empty_deadline_after_genuine_acquired_pcm_has_no_negative_cache(pcm, monkeypatch):
    rx = receiver()
    acquire(rx, pcm)
    owner = rx.clock
    before = dict(rx._long_cache)
    monkeypatch.setattr(p4rx, 'audio_to_baseband', lambda *a, **k: pytest.fail('empty converted'))
    origin = AT+round(3.355*FS)
    assert rx._normal_long(pcm[origin:origin], origin) == (None, 0.)
    assert rx._long_cache == before and rx.clock is owner


def test_one_remaining_sample_delegates_strict_converter(monkeypatch):
    rx = receiver()
    converter = p4rx.audio_to_baseband
    counted = []
    def convert(audio, fs):
        counted.append(len(audio))
        return converter(audio, fs)
    monkeypatch.setattr(p4rx, 'audio_to_baseband', convert)
    rx._normal_long(np.ones(20), AT-round(.050*FS)-19)
    assert counted == [1]


@pytest.mark.parametrize('audio', [np.array([]), np.array([np.nan]), np.ones((2,2)), np.array(['x'])])
def test_strict_converter_stays_strict(audio):
    with pytest.raises((ValueError, TypeError)):
        p4rx.audio_to_baseband(audio, FS)


def test_genuine_repeat_renews_watch_without_second_delivery_and_bad_admissions_do_not():
    from hfmodem.tests.shrike.test_p4turn import machine
    m, _ = machine(role='irs', ladder='none', data='body', chirp_ack='alternate')
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    records = []
    for at in (AT, AT+180000):
        rx = receiver(at)
        audio = np.r_[np.zeros(at), p4chirp.entry_packet(b'repeat source\r', 0x01), np.zeros(FS//10)]
        reg = next(r for r in rx.read_window(audio, 0, whole=True)
                   if r.observation.kind == Kind.CHIRP and r.observation.frame is not None)
        records.append(reg)
        m.observe(reg, now=at+round(3.31*FS))
        assert m.peer_watch.onset == reg.observation.stream_sample
        answer = m.decide(at+163200-m.lead)
        assert answer is not None
        m.keyed()
    assert delivered == [b'repeat source\r']
    watch = m.peer_watch
    assert watch.onset > records[0].observation.stream_sample
    for bad in (replace(records[0], outcome=Registered.EXPIRED), records[0],
                replace(records[1], observation=replace(records[1].observation,
                    stream_sample=AT+360000, provisional=True))):
        m.observe(bad, now=AT+400000)
        assert m.peer_watch is watch
    assert delivered == [b'repeat source\r']


def test_empty_late_slice_preserves_genuinely_acquired_normal_long_entry():
    from hfmodem.shrike import p4longtx
    rx = receiver()
    audio = np.r_[np.zeros(AT), p4longtx.long_packet(b'normal-long source\r', 0x01), np.zeros(FS//10)]
    end = AT+round(.810*FS)
    obs, coherence = rx._normal_long(audio[:end], 0, partial=True)
    assert obs is not None and obs.provisional and coherence > .45
    entry = next(v for v in rx._long_cache.values() if v['kind'] == 'normal')
    before = dict(entry)
    assert rx._normal_long(audio[end+1:end+1], end+1, alias=True) == (None, 0.)
    assert entry == before
    obs, _ = rx._normal_long(audio, 0)
    assert obs is not None and obs.frame is not None
