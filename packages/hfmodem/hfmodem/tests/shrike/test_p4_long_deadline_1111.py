# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Deterministic LONG deadline/publication and complete-repeat evidence gates."""
from time import monotonic

import numpy as np
import pytest

from hfmodem.shrike import p4longdecode
from hfmodem.shrike.p4budget import LongDeadline
from hfmodem.shrike.p4observation import Kind, Observation
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect
from hfmodem.tests.evidence import TREE, require_input
from hfmodem.tests.shrike.test_p4_long_decode_1111 import encoded


FS = 48000
ONSET = FS


def limits(*, sf8=False, anchor=100.):
    clock = dict(sample=ONSET, wall=anchor)
    result = LongDeadline(FS, ONSET, lambda: clock['sample'], lead_s=.050,
                          notice_s=.020, sf8=sf8,
                          wall_now=lambda: clock['wall'])
    return result, clock


@pytest.mark.parametrize('clock_kind', ['sample', 'wall'])
def test_frozen_deadline_expires_with_either_clock_alone(clock_kind):
    budget, clock = limits()
    initial_limit = budget.work_limit()
    assert budget.work.remaining() == pytest.approx(3.345)
    # Repeated queries cannot move the original wall or sample limits.
    if clock_kind == 'wall':
        clock['wall'] += 1
    else:
        clock['sample'] += FS
    assert budget.work_limit() == pytest.approx(initial_limit - (clock_kind == 'sample'))
    assert budget.work.remaining() == pytest.approx(2.345)
    if clock_kind == 'wall':
        clock['wall'] = budget.enqueue.wall_deadline + .001
    else:
        clock['sample'] = budget.enqueue.sample_deadline + 1
    assert budget.work.remaining() < 0
    assert not budget.publishable() and not budget.keyable()


@pytest.mark.parametrize('clock_kind', ['sample', 'wall'])
def test_work_publication_and_enqueue_have_separate_limits(clock_kind):
    budget, clock = limits()
    def just_after(part):
        if clock_kind == 'sample':
            clock['sample'] = part.sample_deadline + 1
        else:
            clock['wall'] = part.wall_deadline + .000001
    just_after(budget.work)
    assert budget.work.remaining() < 0
    assert budget.publishable() and budget.keyable()
    just_after(budget.publication)
    assert not budget.publishable() and budget.keyable()
    just_after(budget.enqueue)
    assert not budget.keyable()


def test_sf8_work_reserve_does_not_shorten_publication_or_enqueue():
    normal, _ = limits()
    sf8, _ = limits(sf8=True)
    assert sf8.work.sample_deadline == normal.work.sample_deadline - round(.010 * FS)
    assert sf8.work.wall_deadline == pytest.approx(normal.work.wall_deadline - .010)
    assert sf8.publication.sample_deadline == normal.publication.sample_deadline
    assert sf8.enqueue.sample_deadline == normal.enqueue.sample_deadline
    assert sf8.publication.wall_deadline == normal.publication.wall_deadline
    assert sf8.enqueue.wall_deadline == normal.enqueue.wall_deadline


def member_frame(sl=5):
    geometry, _, coded, frame = encoded(sl)
    return p4longdecode.LongMember(geometry, 8 * (1 - 2.0 * coded)), frame


def test_remembered_conflicts_are_not_resolved_by_arrival_order():
    first, frame = member_frame()
    g, _, coded, other = encoded(5, seed=1112)
    second = p4longdecode.LongMember(g, 8*(1-2.0*coded))
    receiver = P4Receiver(FS, admitted=True)
    receiver._remember_long(frame, first, ONSET)
    receiver._remember_long(other, second, ONSET)
    repeat = ONSET+round(3.75*FS)
    assert receiver._long_agreement([first, second], repeat) is None
    receiver._remember_long(other, second, repeat)
    receiver._remember_long(frame, first, ONSET)
    assert receiver._long_agreement([first], repeat+round(3.75*FS)) is None


def observation():
    return Observation(Kind.LONG, FS, 0, ONSET, ONSET, 3.282, 0., .95,
                       provisional=True, placed=True)


def test_expired_slot_does_not_start_another_long_frontend(monkeypatch):
    receiver = P4Receiver(FS, admitted=True)
    receiver.clock = Clock(Expect.PACKET, ONSET, .33, 3.75, 0.)
    budget, clock = limits(anchor=monotonic())
    clock['sample'] = budget.work.sample_deadline + FS
    receiver.long_deadline = lambda at, **kwargs: budget
    def forbidden(*args, **kwargs):
        raise AssertionError('expired slot started a new LONG frontend')
    monkeypatch.setattr(receiver, '_normal_long', forbidden)
    monkeypatch.setattr(receiver, '_sf8_long', forbidden)
    monkeypatch.setattr(receiver, '_long', forbidden)
    assert receiver._packet(np.zeros(round(4.4*FS)), 0) is None


@pytest.mark.parametrize('relaxed', [False, True])
def test_late_or_relaxed_crc_is_remembered_without_publication(monkeypatch, relaxed):
    member, frame = member_frame()
    receiver = P4Receiver(FS, admitted=True)
    budget, clock = limits(anchor=monotonic())
    receiver.long_deadline = lambda at, **kwargs: budget
    entry = receiver._start_long((ONSET, 0.), [member], observation())
    advance = entry['search'].advance
    def completed_late(**kwargs):
        result = advance(**kwargs)
        clock['wall'] = budget.publication.wall_deadline + .001
        return result
    monkeypatch.setattr(entry['search'], 'advance', completed_late)
    assert receiver._advance_long(entry, relaxed=relaxed) is None
    assert entry['result'].frame == frame
    assert entry['obs'].frame is None and entry['obs'].provisional
    repeat_at = ONSET + round(3.75 * FS)
    assert receiver._long_agreement([member], repeat_at) == (frame, member)
    assert receiver.live_events[-1]['crc_valid']
    assert not receiver.live_events[-1]['timely']
    assert receiver.registrations == []


def test_result_after_work_limit_but_before_publication_can_publish(monkeypatch):
    member, frame = member_frame()
    receiver = P4Receiver(FS, admitted=True)
    budget, clock = limits(anchor=monotonic())
    receiver.long_deadline = lambda at, **kwargs: budget
    entry = receiver._start_long((ONSET, 0.), [member], observation())
    advance = entry['search'].advance
    def completed_in_publication_reserve(**kwargs):
        result = advance(**kwargs)
        clock['wall'] = budget.work.wall_deadline + .001
        return result
    monkeypatch.setattr(entry['search'], 'advance', completed_in_publication_reserve)
    result = receiver._advance_long(entry)
    assert result.frame == frame and not result.provisional
    assert result.speedlevel == 'SL5-long'
    assert receiver.live_events[-1]['budget_overrun_keyable']


@pytest.mark.parametrize('sl', [2, 4, 5, 10])
@pytest.mark.parametrize('region', ['systematic', 'retained-c1', 'retained-c2'])
def test_repeat_requires_agreement_in_every_complete_region(sl, region):
    member, frame = member_frame(sl)
    receiver = P4Receiver(FS, admitted=True)
    receiver._remember_long(frame, member, ONSET)
    repeat_at = ONSET + round(3.75 * FS)
    assert receiver._long_agreement([member], repeat_at) == (frame, member)
    fields = member.fields.copy()
    _, lo, hi = next(r for r in member.geometry.regions if r[0] == region)
    # Corrupt the region suffix, beyond an early-packet-only comparison.
    fields[hi - max(1, (hi - lo) // 5):hi] *= -1
    damaged = p4longdecode.LongMember(member.geometry, fields)
    assert receiver._long_agreement([damaged], repeat_at) is None


def test_repeat_rejects_reliable_status_conflict_wrong_cadence_and_generation():
    member, frame = member_frame()
    receiver = P4Receiver(FS, admitted=True)
    receiver._remember_long(frame, member, ONSET)
    repeat_at = ONSET + round(3.75 * FS)
    fields = member.fields.copy()
    fields[8 * member.geometry.data_bytes] *= -1
    conflict = p4longdecode.LongMember(member.geometry, fields)
    assert receiver._long_agreement([conflict], repeat_at) is None
    assert receiver._long_agreement([member], repeat_at + FS // 10) is None
    receiver.new_call()
    assert receiver._long_agreement([member], repeat_at) is None


@pytest.mark.parametrize('sl,deadline,cost,calls', [(4, .015, .004, 3), (5, .020, .010, 2)])
def test_python_grant_stops_between_members_and_resumes(monkeypatch, sl, deadline, cost, calls):
    from hfmodem.shrike import p4longnative
    monkeypatch.setattr(p4longnative, 'backend', None)
    g = p4longdecode.geometry(sl)
    members = [p4longdecode.LongMember(g, np.random.default_rng(i).normal(size=g.serialized_size))
               for i in range(8)]
    job = p4longdecode.LongSearch(members, policy='exhaustive', component_budget=0)
    job.advance()
    now = [0.]
    step = p4longdecode._member_step
    def timed(*args):
        result = step(*args)
        now[0] += cost
        return result
    monkeypatch.setattr(p4longdecode.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4longdecode, '_member_step', timed)
    job.component_budget = None
    result = job.advance(deadline=deadline)
    assert result.state == 'incomplete' and result.completed_tier == 0
    assert result.component_calls == calls
    assert now[0] == cost*result.component_calls
    assert now[0] <= deadline
    expected_job = p4longdecode.LongSearch(members, policy='exhaustive')
    expected = expected_job.advance()
    actual = job.advance()
    assert actual.provenance == expected.provenance
    np.testing.assert_array_equal(job.priors, expected_job.priors)


def test_normal_registration_retains_job_and_remembered_codeword():
    member, frame = member_frame()
    receiver = P4Receiver(FS, admitted=True)
    entry = receiver._start_long((ONSET, 0., 'normal'), [member], observation())
    obs = receiver._advance_long(entry)
    assert obs.frame == frame
    generation = receiver._long_generation
    receiver.register(obs)
    assert receiver._long_generation == generation
    assert receiver._long_jobs[(ONSET, 0., 'normal')] is entry
    assert receiver._long_agreement([member], ONSET + round(3.75*FS)) == (frame, member)


@pytest.mark.parametrize('shift', [-FS//50, FS//50])
def test_placed_long_uses_existing_physical_slot_grant(shift):
    """Placement must not buy a fresh wall allowance after frontend work."""
    from dataclasses import replace
    member, _ = member_frame(2)
    receiver = P4Receiver(FS, admitted=True)
    budget, clock = limits()
    calls = []
    def deadline(at, **kwargs):
        calls.append(at)
        return budget if at == ONSET else limits(anchor=clock['wall'])[0]
    receiver.long_deadline = deadline
    # The prefix/dispatcher has already frozen this physical slot's budget.
    original_stop = budget.work_limit()
    clock['wall'] += .025
    placed = replace(observation(), stream_sample=ONSET+shift,
                     first_chip=ONSET+shift)
    entry = receiver._start_long((ONSET, 0., 'sf16'), [member], placed)
    assert calls == [ONSET]
    assert entry['limits'] is budget
    assert entry['limits'].work_limit() == original_stop
    assert entry['obs'].stream_sample == ONSET+shift
    # A wider retained field domain keeps the same grant too.
    receiver._start_long((ONSET, 0., 'sf16'), [member], placed, replace_members=True)
    assert calls == [ONSET] and entry['limits'] is budget


MEASURED = TREE / 'working/p4-long-rx-1111/python-components-current.json'


def python_component_measurements():
    import json
    return json.loads(require_input(MEASURED).read_text())


@pytest.mark.parametrize('row', python_component_measurements()['rows'] if MEASURED.is_file() else [None],
                         ids=lambda row: f"SL{row['speedlevel']}" if row else 'unmeasured')
def test_python_constituent_reserve_covers_measured_worst_with_margin(row):
    require_input(MEASURED)
    g = p4longdecode.geometry(row['speedlevel'])
    assert row['reserve_ms'] == p4longdecode._python_step_seconds(g, 1) * 1000
    for tier, calls in ((1, 1), (2, 1), (3, 2)):
        assert p4longdecode._python_step_seconds(g, tier) >= calls * row['max_ms'] * .00125


def test_python_batch_reserve_covers_probability_domain_measurement():
    row = python_component_measurements()['batch']
    g = p4longdecode.geometry(row['speedlevel'])
    assert row['reserve_ms'] == p4longdecode._python_step_seconds(g, 1, row['count']) * 1000
    for tier, calls in ((1, 1), (2, 1), (3, 2)):
        assert p4longdecode._python_step_seconds(g, tier, row['count']) >= calls * row['max_ms'] * .00125


@pytest.mark.parametrize('sl,count', [(3, 24), (10, 1)])
def test_python_fallback_reserves_complete_step(monkeypatch, sl, count):
    """A positive grant alone cannot admit an indivisible Python recurrence."""
    monkeypatch.setattr(p4longdecode.p4longnative, 'backend', None)
    g = p4longdecode.geometry(sl)
    members = [p4longdecode.LongMember(g, np.ones(g.serialized_size))]*count
    job = p4longdecode.LongSearch(members, policy='exhaustive', component_budget=0)
    job.advance()
    assert job.tier == 1
    job.component_budget = None
    monkeypatch.setattr(p4longdecode.time, 'monotonic', lambda: 0.)
    def forbidden(*args, **kwargs):
        raise AssertionError('Python recurrence started without a completion reserve')
    monkeypatch.setattr(p4longdecode, '_member_step', forbidden)
    monkeypatch.setattr(p4longdecode, '_batch_step', forbidden)
    result = job.advance(deadline=.000001)
    assert result.state == 'incomplete' and result.completed_tier == 0
    assert result.component_calls == 0 and job.member_index == 0 and not job.pending
    assert not np.any(job.priors) and not np.any(job.first)


def test_python_component_can_park_before_validation_without_recomputing(monkeypatch):
    monkeypatch.setattr(p4longdecode.p4longnative, 'backend', None)
    g, _, coded, frame = encoded(2)
    fields = 8*(1-2.0*coded)
    fields[7] *= -1
    job = p4longdecode.LongSearch([p4longdecode.LongMember(g, fields)],
                                 iterations=1, component_budget=0)
    job.advance()
    job.component_budget = None
    now = [0.]
    step, check = p4longdecode._member_step, p4longdecode._check
    def timed(*args):
        result = step(*args)
        now[0] += .0049
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError('validation started without its own completion reserve')
    monkeypatch.setattr(p4longdecode.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4longdecode, '_member_step', timed)
    monkeypatch.setattr(p4longdecode, '_check', forbidden)
    paused = job.advance(deadline=.005)
    assert paused.state == 'incomplete' and paused.frame is None
    assert paused.component_calls == 1 and job.member_index == 0 and 0 in job.pending
    monkeypatch.setattr(p4longdecode, '_check', check)
    resumed = job.advance(deadline=.020)
    assert resumed.frame == frame and resumed.completed_tier == 1
    assert resumed.component_calls == 1 and not job.pending


def test_retained_frontend_admission_deferral_remains_retryable(monkeypatch):
    from hfmodem.shrike import p4receiver
    rx = P4Receiver(FS, admitted=True)
    rx.clock = Clock(Expect.PACKET, ONSET, .33, 3.75, 0.)
    audio = np.zeros(round(3.3*FS))
    audio.flags.writeable = False
    saved = dict(audio=audio, origin=ONSET, end=ONSET+len(audio), clock=rx.clock,
                 generation=rx._long_generation, tried=False)
    rx._long_inputs[(rx._long_generation, ONSET, 0.)] = saved
    now = [10.]
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])
    calls = []
    def frontend(*args, **kwargs):
        calls.append(rx._long_relaxed_deadline)
        if len(calls) == 1:
            now[0] += .045
        rx._long_work_expired(ONSET, reserve_s=.030)
        return None
    monkeypatch.setattr(rx, '_packet', frontend)
    assert rx.refine(deadline=10.010) == []
    assert not saved['tried'] and calls == [] and now[0] == 10.
    assert rx.refine(deadline=10.070) == []
    assert not saved['tried'] and calls == [10.070] and now[0] <= 10.070
    assert rx.refine(deadline=10.270) == []
    assert saved['tried'] and calls == [10.070, 10.270]
    assert rx.refine(deadline=10.520) == []
    assert calls == [10.070, 10.270] and rx._long_relaxed_deadline is None


def test_python_batch_shrinks_to_grant_and_resumes_full_domain(monkeypatch):
    monkeypatch.setattr(p4longdecode.p4longnative, 'backend', None)
    g = p4longdecode.geometry(3)
    members = [p4longdecode.LongMember(g, np.random.default_rng(i).normal(size=g.serialized_size))
               for i in range(24)]
    job = p4longdecode.LongSearch(members, policy='exhaustive', component_budget=0, iterations=1)
    job.advance()
    job.component_budget = None
    now = [0.]
    batches = []
    batch = p4longdecode._batch_step
    def timed(job, indices, tier):
        result = batch(job, indices, tier)
        batches.append(tuple(indices))
        now[0] += p4longdecode._python_step_seconds(g, tier, len(indices))
        return result
    monkeypatch.setattr(p4longdecode.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4longdecode, '_batch_step', timed)
    paused = job.advance(deadline=.008)
    assert batches == [tuple(range(8))]
    assert paused.state == 'incomplete' and paused.completed_tier == 0
    assert paused.component_calls == 8 and now[0] <= .008
    assert len(job.members) == 24 and job.pending
    expected_job = p4longdecode.LongSearch(members, policy='exhaustive', iterations=1)
    expected = expected_job.advance()
    actual = job.advance()
    assert actual.provenance == expected.provenance
    assert actual.component_calls == expected.component_calls
    np.testing.assert_array_equal(job.priors, expected_job.priors)
    np.testing.assert_array_equal(job.first, expected_job.first)


@pytest.mark.parametrize('grant,calls', [(.002, 0), (.004, 1)])
def test_relaxed_python_grant_uses_stage_reserve_without_blanket_floor(monkeypatch, grant, calls):
    from hfmodem.shrike import p4receiver
    monkeypatch.setattr(p4longdecode.p4longnative, 'backend', None)
    g, _, coded, frame = encoded(2)
    fields = 8*(1-2.0*coded)
    fields[7] *= -1
    rx = P4Receiver(FS, admitted=True)
    entry = rx._start_long((ONSET, 0.), [p4longdecode.LongMember(g, fields)], observation())
    job = entry['search']
    job.component_budget = 0
    job.advance()
    job.component_budget = None
    now = [0.]
    step, check = p4longdecode._member_step, p4longdecode._check
    def timed_step(*args):
        result = step(*args)
        now[0] += p4longdecode._python_step_seconds(g, args[-1])
        return result
    def timed_check(*args):
        result = check(*args)
        now[0] += p4longdecode._python_check_seconds(g)
        return result
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4longdecode.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4longdecode, '_member_step', timed_step)
    monkeypatch.setattr(p4longdecode, '_check', timed_check)
    assert rx.refine(deadline=grant) == []
    assert entry['result'].component_calls == calls
    assert entry['result'].frame == (frame if calls else None)
    assert now[0] <= grant
    assert entry['obs'].frame is None and not rx.registrations
