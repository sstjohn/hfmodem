# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Remembered and combined work keeps each physical copy's provenance."""
import numpy as np
import pytest

from time import monotonic
from types import SimpleNamespace

from hfmodem.shrike import p4longdecode, p4oracle, p4receiver
from hfmodem.shrike.p4budget import LongDeadline
from hfmodem.shrike.p4receiver import P4Receiver, Kind, Observation, Clock, Expect
from hfmodem.tests.shrike.test_p4_long_decode_1111 import encoded


def test_remembered_repeat_selects_actual_member_among_equal_provenance():
    g, _, coded, frame = encoded(5)
    correct = p4longdecode.LongMember(g, 8*(1-2.0*coded), weighting='pilot')
    wrong = p4longdecode.LongMember(g, -correct.fields, weighting='pilot')
    # Fields deliberately do not participate in dataclass equality. Physical
    # selection must keep object identity rather than pick the first equal tag.
    assert correct == wrong
    rx = P4Receiver(48000, admitted=True, breakin=True)
    at = 48000
    rx._remember_long(frame, correct, at)
    onset = at+180000
    obs = Observation(Kind.LONG, 48000, 0, onset, onset, 3.282, 0., .8,
                      provisional=True, placed=True)
    job = rx._start_long((onset, 0., 'normal'), (wrong, correct), obs)
    result = rx._advance_long(job)
    assert result.frame == frame and result.stream_sample == onset
    assert job['selected_member'] == 1
    assert rx.live_events[-1]['state'] == 'repeat'
    assert job['search'].tier == 0  # Agreement did not run a second decoder.


@pytest.mark.parametrize('separation', ['reread', 'raster', 'cfo', 'status', 'body', 'gap', 'epoch'])
def test_unrelated_or_repeated_input_cannot_borrow_previous_channel_evidence(separation):
    g, _, coded, _ = encoded(5)
    fields = 8*(1-2.0*coded)
    rx = P4Receiver(48000, admitted=True, breakin=True)
    first = p4longdecode.LongMember(g, fields, weighting='pilot')
    rx._normal_long_members((first,), ((.9, 0, 0.),), 48000)
    onset, cfo = 228000, 0.
    current = fields.copy()
    if separation == 'reread':
        onset = 48000
    elif separation == 'raster':
        onset += 4800
    elif separation == 'cfo':
        cfo = 7.5
    elif separation == 'status':
        current[8*g.data_bytes] *= -1
    elif separation == 'body':
        current[g.systematic:] *= -1
    elif separation == 'gap':
        rx.capture_gaps.append((48001, 48002))
    elif separation == 'epoch':
        rx.invalidate_long_phase()
    member = p4longdecode.LongMember(g, current, weighting='pilot')
    domain, _ = rx._normal_long_members((member,), ((.9, 0, cfo),), onset)
    assert domain == (member,)
    np.testing.assert_array_equal(domain[0].fields, current)
    np.testing.assert_array_equal(first.fields, fields)


def test_combining_retains_at_most_four_distinct_physical_copies():
    g, _, coded, wanted = encoded(5)
    fields = 8*(1-2.0*coded)
    rx = P4Receiver(48000, admitted=True, breakin=True)
    for copy in range(6):
        member = p4longdecode.LongMember(g, fields, weighting='pilot')
        domain, _ = rx._normal_long_members((member,), ((.9, 0, 0.),), 48000+180000*copy)
        assert domain[0] is member
        if copy:
            assert len(domain) == 2
            np.testing.assert_array_equal(domain[1].fields, min(copy+1, 4)*fields)
            assert len(set(domain[1].label.removeprefix('copies:').split(','))) == min(copy+1, 4)
            result = p4longdecode.LongSearch(domain, component_budget=0).advance()
            assert result.frame == wanted and result.completed_tier == 0


def test_reply_gap_preserves_repeat_evidence_but_body_gap_invalidates_it():
    g, _, coded, wanted = encoded(5)
    member = p4longdecode.LongMember(g, 8*(1-2.0*coded), weighting='pilot')
    rx = P4Receiver(48000, admitted=True, breakin=True)
    at = 48000
    rx._remember_long(wanted, member, at)
    rx._normal_long_members((member,), ((.9, 0, 0.),), at)
    rx.note_gap(at+round(3.306*48000), at+round(3.65*48000))
    assert next(iter(rx._remembered_longs.values()))[0] == wanted
    assert len(rx._normal_long_copies) == 1
    rx.note_gap(at+48000, at+48001)
    assert rx._remembered_longs == {} and rx._normal_long_copies == []


def test_expired_frontend_retains_immutable_input_for_bounded_remembered_recovery(monkeypatch):
    at, fs = 48000, 48000
    rx = P4Receiver(fs, admitted=True, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, 0.)
    expired = LongDeadline(fs, at, lambda: at+round(3.5*fs), lead_s=.044)
    rx.long_deadline = lambda *a, **kw: expired
    monkeypatch.setattr(p4oracle, 'robust_at', lambda *a, **kw: dict(
        p4_walsh_peak=0, coherence=0., first_chip=None, p4_hdr='?', p4_value=None))
    monkeypatch.setattr(p4oracle, 'placed_despread', lambda *a, **kw: (0., [], None))
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [])
    audio = np.zeros(at+round(3.355*fs))
    assert rx._packet(audio, 0, long_fallback=True) is None
    saved, = rx._long_inputs.values()
    assert not saved['audio'].flags.writeable and not rx._long_jobs
    audio[:] = 1  # A caller reusing its buffer cannot change retained evidence.
    assert not np.any(saved['audio'])
    g, _, coded, wanted = encoded(2)
    member = p4longdecode.LongMember(g, 8*(1-2.0*coded), label='20')
    calls = []

    def finish_retained(data, origin, **kwargs):
        calls.append(rx._long_relaxed_deadline)
        assert not data.flags.writeable and not np.any(data)
        obs = Observation(Kind.LONG, fs, origin, at-origin, at, 3.28, 0., .9,
                          provisional=True, placed=True)
        entry = rx._start_long((at, 0., 'sf16'), (member,), obs)
        assert rx._advance_long(entry) is None

    monkeypatch.setattr(rx, '_packet', finish_retained)
    assert rx.refine(deadline=monotonic()-1) == [] and calls == []
    limit = monotonic()+.250
    assert rx.refine(deadline=limit) == [] and calls == [limit]
    job, = rx._long_jobs.values()
    assert job['result'].frame == wanted and job['obs'].frame is None
    assert not rx.observations and not rx.registrations
    assert rx.live_events[-1]['relaxed'] and not rx.live_events[-1]['timely']
    assert next(iter(rx._remembered_longs.values()))[0] == wanted
    assert rx.refine(deadline=monotonic()+.250) == [] and calls == [limit]


@pytest.mark.parametrize('prepared', [False, True])
def test_retained_frontend_reserves_work_before_starting(monkeypatch, prepared):
    fs, at = 48000, 48000
    rx = P4Receiver(fs, admitted=True, breakin=True)
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, 0.)
    audio = np.zeros(round(3.3*fs))
    audio.flags.writeable = False
    saved = dict(audio=audio, origin=at, end=at+len(audio), clock=rx.clock,
                 generation=rx._long_generation, tried=False)
    rx._long_inputs[(rx._long_generation, at, 0.)] = saved
    if prepared:
        rx._long_prepared[(at, 0.)] = object()
        rx._sf8_prepared[(at, 0.)] = object()
    reserve = .015+max(rx._long_frontend_seconds(sf8=True),
                       rx._long_frontend_seconds(fallback=True))
    rx.clock = Clock(Expect.PACKET, at+180000, .33, 3.75, 2.)
    now, calls = [10.], []
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])

    def frontend(data, origin, **kwargs):
        assert data is audio and origin == at
        calls.append(rx._long_relaxed_deadline)
        assert rx.clock is saved['clock']
        now[0] += reserve
        return None

    monkeypatch.setattr(rx, '_packet', frontend)
    for remaining in (.001, reserve-.0001):
        deadline = now[0]+remaining
        assert rx.refine(deadline=deadline) == []
        assert calls == [] and now[0] == 10. and now[0] <= deadline
        assert not saved['tried'] and not rx._long_jobs
    assert rx._long_relaxed_deadline is None
    assert rx.refine(deadline=10.270) == []
    assert calls == [10.270] and saved['tried'] and now[0] <= 10.270
    assert rx.refine(deadline=10.520) == []
    assert calls == [10.270]  # A completed negative is not repeated.
    assert not rx.observations and not rx.registrations and not rx._remembered_longs


@pytest.mark.parametrize('deadline', [None, 10.4])
def test_refine_shares_one_slice_across_hold_normal_and_long(monkeypatch, deadline):
    # Design 166 caps the entire relaxed grant, including a default grant.
    rx = P4Receiver(48000, admitted=True, breakin=True)
    now, calls = [10.], []
    limit = 10.250 if deadline is None else deadline
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])

    def hold(*, deadline):
        calls.append(('hold', deadline))
        assert deadline == limit
        now[0] = limit-.050

    def normal(deadline):
        calls.append(('normal', deadline))
        assert deadline == limit
        now[0] = limit+.001

    rx.hold = SimpleNamespace(kind='normal', refine=hold)
    obs = Observation(Kind.NORMAL, 48000, 0, 48000, 48000, .805, 0., .8)
    rx._normal_cache[48000] = dict(obs=obs,
        decoder=SimpleNamespace(done=False, decode=normal))
    rx._long_jobs[(48000, 0., 'normal')] = dict(obs=obs,
        search=SimpleNamespace(done=False))
    monkeypatch.setattr(rx, '_advance_long',
        lambda *a, **kw: pytest.fail('LONG received a second relaxed grant'))
    assert rx.refine(deadline=deadline) == []
    assert calls == [('hold', limit), ('normal', limit)]


def test_sf8_fallback_rechecks_budget_after_robust_classification(monkeypatch):
    fs, at = 48000, 48000
    rx = P4Receiver(fs, admitted=True, breakin=True)
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, 0.)
    now = [10.]
    rx._long_relaxed_deadline = 10.010
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])

    def classify(*args, **kwargs):
        now[0] += .020
        return dict(p4_walsh_peak=0., coherence=0., first_chip=None,
                    p4_hdr='?', p4_value=None)

    monkeypatch.setattr(p4oracle, 'robust_at', classify)
    monkeypatch.setattr(p4oracle, 'placed_despread', lambda *a, **kw: (0., [], None))
    monkeypatch.setattr(rx, '_normal_eligible', lambda: False)
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [])
    monkeypatch.setattr(rx, '_sf8_long',
        lambda *a, **kw: pytest.fail('SF8 frontend began after its cutoff'))
    monkeypatch.setattr(rx, '_long',
        lambda *a, **kw: pytest.fail('SF16 frontend began after its cutoff'))
    assert rx._packet(np.zeros(at+round(3.300*fs)), 0) is None
    assert now[0] == 10.020
    assert not rx._long_jobs and not rx.observations and not rx.registrations
