# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Normal-long receive on the recorded VE3KPG 3.75 s raster."""
import hashlib
import json
from contextlib import redirect_stdout
from os import devnull

import numpy as np
import pytest

from hfmodem.shrike import (p4frame, p4longdecode as engine,
                            p4longnative, p4longtx, p4rx, p4short, p4turn, session)
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.p4_chirp_deadline_support import FS, prerender, protocol_heard, replay
from hfmodem.tests.shrike.test_p4_long_decode_1111 import encoded
from pathlib import Path
from hfmodem.tests.shrike.p4_nak_qam_support import (
    neutral_keys, waveform_keys, without_neutral_requests,
)


def unhurried():
    """A refine deadline these decode checks cannot reach on a loaded host.

    They assert what the engine decodes, not that the Python fallback finishes
    inside the live 250 ms slice; the slice itself is pinned with a fake clock below.
    """
    return engine.time.monotonic() + 60


FIRST = 2148660
SECOND = 2328660
PREFIX = b'(FN04VE)\r[WL2K-5.0-'


def recorded(arm, stop, **kwargs):
    with open(devnull, 'w') as sink, redirect_stdout(sink):
        return replay(arm, stop=stop, **kwargs)


def assert_long_exchange(result):
    heard = result['heard']
    c1 = [row for row in heard if row['kind'] == Kind.NORMAL
          and abs(row['onset_sample']-2088660) <= 1680 and not row['provisional']]
    assert c1 and c1[-1].get('status') == 1
    for at in (FIRST, SECOND):
        copies = [row for row in heard if row['kind'] == Kind.LONG
                  and abs(row['onset_sample']-at) <= 1680 and not row['provisional']]
        assert copies, f'no framed long near {at}'
        row = copies[-1]
        assert row.get('status') == 2
        assert bytes.fromhex(row['data']).startswith(PREFIX)
        assert abs(row['cfo_hz']+7.1) <= .5
    keyed = [row['at'] for row in result['events'] if row['event'] == 'Keyed'
             and row.get('kind') == Kind.WORD and row.get('word') == 'X27'
             and row.get('state') == 'IRS_ANSWER'
             and row['at'] >= FIRST]
    assert any(abs(at-expected) <= 1680 for at in keyed
               for expected in (2311860, 2491860)), keyed
    assert not [e for e in result['events'] if e['event'] == 'Refused']
    assert not [e for e in result['events'] if e['event'] == 'Decision'
                and e['at'] >= round(45.5*FS)
                and 'no live packet' in e.get('reason', '')]
    assert all(count <= 1 for at, count in result['cycles'] if at >= round(45.574*FS))
    return keyed[0]


def test_1049_long_replay_acknowledges_the_first_or_next_copy():
    keyed = assert_long_exchange(recorded('onair-1002-1049', 52*FS,
                                          work_costs={
                                              'read_partial': lambda at, _: .015 if at >= FIRST else .003,
                                              'read_window': lambda at, args: (
                                                  .045 if at >= FIRST and args[1]+len(args[0]) >= at+3*FS
                                                  else .008 if at >= FIRST else .004),
                                          }))
    assert abs(keyed-2311860) <= 1680 or abs(keyed-2491860) <= 1680


def test_1049_pin_fails_without_the_lattice_candidate(monkeypatch):
    original = P4Receiver._long

    def despread_only(self, audio, origin, **kwargs):
        if kwargs.get('placed'):
            return None, 0.
        return original(self, audio, origin, **kwargs)

    with monkeypatch.context() as mutation:
        mutation.setattr(P4Receiver, '_long', despread_only)
        result = recorded('onair-1002-1049', 52*FS)
        with pytest.raises(AssertionError, match='no framed long'):
            assert_long_exchange(result)
        assert not [row for row in result['heard'] if row['kind'] == Kind.LONG
                    and row['onset_sample'] >= FIRST]
        assert any(count >= 2 for at, count in result['cycles'] if at >= round(45.574*FS))


@pytest.mark.wall_clock  # measured performance runs serially at nice 0
def test_1049_sample_clock_receive_cost():
    if p4longnative.backend is None:
        pytest.skip('pure-Python LONG decode measured over the publication budget; the native backend meets it')
    result = recorded('onair-1002-1049', 52*FS, measured_work=True)
    decisions = [e for e in result['events'] if e['event'] == 'Decision'
                 and e['at'] >= round(45.5*FS)]
    assert decisions
    for row in decisions:
        work = row['work_ms']
        assert work.get('read_window', 0)+work.get('read_partial', 0) <= 100
        assert (row['decision_at']-row['now_sample'])*1000/FS >= -40
    for at in (FIRST, SECOND):
        # Design 148/150/246: quick success is now a completed common-engine
        # tier (systematic, C1 or turbo-1), rather than an SL5 wrapper call.
        quick = [e for e in result['events'] if e['event'] == 'long-engine'
                 and abs(e['at']-at) <= 1680
                 and not e['relaxed'] and e['crc_valid'] and e['timely']]
        assert quick and quick[0]['elapsed_ms'] <= 40
        expected = ('unique', 1, 0, 24) if at == FIRST else ('repeat', None, None, 28)
        assert (quick[0]['state'], quick[0]['tier'], quick[0]['iteration'],
                quick[0]['members']) == expected
        assert quick[0]['copies'] == 1 and quick[0]['polarity'] == 1
        frames = [row for row in result['heard'] if row['kind'] == Kind.LONG
                  and abs(row['onset_sample']-at) <= 1680 and not row['provisional']]
        assert frames and all(row['status'] == 2 and bytes.fromhex(row['data']).startswith(PREFIX)
                              for row in frames)


def receiver(at=FIRST):
    rx = P4Receiver(FS, admitted=True, cfo_hz=-6.94, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, -6.94)
    return rx


def audio():
    return session.load_wav(str(require_input(CAPTURES/'onair-1002-1049'/'stream.flac')), FS)


def test_recorded_long_quick_decode_and_held_repeat():
    tape = audio()
    rx = receiver()
    start = FIRST-round(.050*FS)
    partial, _ = rx._long(tape[start:FIRST+round(.810*FS)], start,
                          placed=True, partial=True)
    assert partial is not None and partial.provisional
    assert rx.clock.cycle_s == 3.75 and rx.clock.misses == 0
    full, _ = rx._long(tape[start:FIRST+round(3.355*FS)], start, placed=True)
    assert full is not None and full.frame is not None and full.frame.status.byte == 2
    assert full.frame.data.startswith(PREFIX)
    assert any(e['event'] == 'long-engine' and e['tier'] == 1
               and e['iteration'] == 0 and e['copies'] == 1 and e['polarity'] == 1
               and e['crc_valid'] and e['timely'] for e in rx.live_events)
    job = next(entry for key, entry in rx._long_jobs.items() if key[2] == 'normal')
    assert job['search'].members[job['result'].selected_provenance.member_index].polarity == 1
    assert any(e['event'] == 'long-header' for e in rx.live_events)
    rx.clock.next_due = SECOND
    start = SECOND-round(.050*FS)
    repeat, _ = rx._long(tape[start:SECOND+round(.810*FS)], start,
                         placed=True, partial=True)
    # Design 172: prefix agreement ranks a remembered packet; only its own
    # complete body can authorize the repeated copy's answer.
    assert repeat is not None and repeat.provisional and repeat.frame is None
    repeat, _ = rx._long(tape[start:SECOND+round(3.355*FS)], start, placed=True)
    assert repeat is not None and not repeat.provisional and repeat.frame == full.frame
    assert abs(repeat.stream_sample-SECOND) <= 1680
    assert any(e['event'] == 'long-engine' and e['state'] == 'repeat'
               and e['agreement'] >= .70
               and abs(e['at']-SECOND) <= 1680 and e['timely'] for e in rx.live_events)


def test_pending_long_stops_silence_and_late_frame_reanchors_clock():
    rx = receiver()
    pending = Observation(Kind.LONG, FS, 0, FIRST, FIRST, .810, -7.1, .7,
                          provisional=True, placed=True)
    rx.register(pending)
    rx.silent_cycles = 2
    assert rx.finish_cycle(FIRST) == 0
    frame = p4frame.decode_field(p4frame.encode_field(
        b'x'*203, p4frame.Status.from_byte(2)), data_bytes=203)
    rx.clock.next_due = FIRST+round(2*1.25*FS)
    confirmed = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                            frame=frame, placed=True)
    rx._emit(confirmed, [])
    assert rx.clock.cycle_s == 3.75
    assert rx.clock.next_due == FIRST+round(3.75*FS)


def test_expired_crc_miss_retains_placed_long_carrier_liveness():
    rx = receiver()
    rx.peer_onset = FIRST-round(1.25*FS)
    rx.clock.cycle_s = 3.75
    for cycle in range(5):
        at = FIRST+cycle*round(3.75*FS)
        if cycle:
            rx.new_cycle(at)
        pending = Observation(Kind.LONG, FS, 0, at, at, .810, -7.1, .75,
                              provisional=True, placed=True)
        rx.register(pending)
        rx.observations.remove(pending)
        rx._live(RegisteredObservation(pending, Registered.EXPIRED, pending))
        assert rx.finish_cycle(at) == 0
    assert len(rx.live_onsets) == 5


def test_second_and_third_1049_long_crc_misses_keep_carrier_live(monkeypatch):
    tape = audio()
    rx = receiver()
    rx.peer_onset = FIRST-round(1.25*FS)  # last short before the long raster
    first_start = FIRST-round(.050*FS)
    # Complete the real systematic/CRC tier, rather than expire the grant
    # before any decoder work. The recording requires error correction.
    initialize = engine.LongSearch.__init__
    def systematic(self, members, **kwargs):
        initialize(self, members, **dict(kwargs, iterations=0))
    monkeypatch.setattr(engine.LongSearch, '__init__', systematic)
    first, _ = rx._long(tape[first_start:FIRST+round(.810*FS)], first_start,
                        placed=True, partial=True)
    assert first is not None and first.provisional and not first.placed
    first, _ = rx._long(tape[first_start:FIRST+round(3.355*FS)], first_start,
                        placed=True)
    assert first is not None and first.provisional and first.placed and first.frame is None
    rx.register(first)
    assert rx.clock.cycle_s == 3.75
    assert rx.finish_cycle(FIRST) == 0

    for at in (SECOND, SECOND+round(3.75*FS)):
        rx.new_cycle(at)
        start = at-round(.050*FS)
        pending, _ = rx._long(tape[start:at+round(.810*FS)], start,
                              placed=True, partial=True)
        assert pending is not None and pending.provisional and not pending.placed
        rx.register(pending)
        missed, _ = rx._long(tape[start:at+round(3.355*FS)], start, placed=True)
        assert missed is not None and missed.frame is None
        job = next(entry for key, entry in rx._long_jobs.items()
                   if key[0] == at and key[2] == 'normal')
        result = job['result']
        assert result.state == 'unread' and result.complete and result.completed_tier == 0
        assert result.component_calls == 0
        assert len(result.provenance) == len(job['search'].members)
        assert all(not row.crc_valid for row in result.provenance)
        assert any(e['event'] == 'long-engine' and abs(e['at']-at) <= 1680
                   and not e['crc_valid'] for e in rx.live_events)
        rx.register(missed)
        assert missed.placed and any(abs(at-s) <= 1680 for s in rx.live_onsets)
        rx.observations.remove(missed)
        rx._live(RegisteredObservation(missed, Registered.EXPIRED, missed))
        assert rx.finish_cycle(at) == 0
    assert rx.peer_onset == FIRST-round(1.25*FS)


def test_new_long_frame_can_contradict_an_early_repeat():
    rx = receiver()
    first = p4frame.decode_field(p4frame.encode_field(
        b'x'*203, p4frame.Status.from_byte(2)), data_bytes=203)
    changed = p4frame.decode_field(p4frame.encode_field(
        b'y'*203, p4frame.Status.from_byte(2)), data_bytes=203)
    early = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                        frame=first, placed=True)
    late = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                       frame=changed, placed=True)
    rx.register(early)
    result = rx.register(late)
    assert result.outcome == Registered.CONTRADICTED
    assert rx.observations[-1].frame == changed
    assert any(e['event'] == 'long-repeat-contradicted' for e in rx.live_events)


def test_failed_quick_decode_gets_eight_iteration_refine(monkeypatch):
    if p4longnative.backend is None:
        pytest.skip('reference eight-member refine needs over 2 s; native fits the 250 ms grant')
    tape = audio()
    rx = receiver()
    start = FIRST-round(.050*FS)
    clean = tape[start:FIRST+round(3.355*FS)]
    noisy = clean+np.random.default_rng(14).normal(0, .75, len(clean))
    initialize = engine.LongSearch.__init__

    def quick(self, members, **kwargs):
        initialize(self, members, **dict(kwargs, component_budget=2*len(members)))

    with monkeypatch.context() as urgent:
        urgent.setattr(engine.LongSearch, '__init__', quick)
        rx._long(noisy[:round(.860*FS)], start, placed=True, partial=True)
        pending, _ = rx._long(noisy, start, placed=True)
    assert pending is not None and pending.frame is None
    job = next(entry for key, entry in rx._long_jobs.items() if key[2] == 'normal')
    assert job['result'].completed_tier == 2 and job['result'].component_calls == 16
    assert not any(p.crc_valid for p in job['result'].provenance)
    assert job['search'].policy == 'live' and job['search'].last_tier == 9
    assert not job['search'].done
    job['search'].component_budget = None
    # Design 170: relaxed success is remembered, never delivered into an old
    # slot. The full repeat receives its own registration and deadline.
    assert rx.refine(deadline=unhurried()) == []
    assert job['result'].frame is not None and job['result'].frame.status.byte == 2
    assert job['result'].completed_tier == 9 and job['search'].done
    assert job['result'].component_calls == 128
    assert len(job['result'].provenance) == 80
    assert {p.iteration for p in job['result'].provenance if p.method == 'turbo'} == set(range(1, 9))
    assert job['obs'].frame is None and rx.registrations == []
    assert any(e['event'] == 'long-engine' and e['relaxed']
               and e['crc_valid'] and not e['timely'] for e in rx.live_events)
    rx.clock.next_due = SECOND
    start = SECOND-round(.050*FS)
    prefix, _ = rx._long(tape[start:SECOND+round(.810*FS)], start,
                         placed=True, partial=True)
    assert prefix.provisional and prefix.frame is None
    repeated, _ = rx._long(tape[start:SECOND+round(3.355*FS)], start, placed=True)
    assert repeated.frame == job['result'].frame
    assert abs(repeated.stream_sample-SECOND) <= 1680


def independent_repeat_models(rx, *, shared_error=False):
    """Independently encoded SL5, with immutable raw observations per copy."""
    g, _, coded, wanted = encoded(5)
    first, second = (8*(1-2.0*coded) for _ in range(2))
    first[7] = -2*(1-2.0*coded[7])
    error = 7 if shared_error else 17
    second[error] = -2*(1-2.0*coded[error])
    placement = (.9, 0, -6.94)
    raw1 = engine.LongMember(g, first, label='first')
    raw2 = engine.LongMember(g, second, label='single')
    domain1, _ = rx._normal_long_members([raw1], [placement], FIRST)
    domain2, _ = rx._normal_long_members([raw2], [placement], SECOND)
    assert domain1 == (raw1,) and len(domain2) == 2
    assert domain2[0] is raw2 and domain2[1].label.startswith('copies:')
    np.testing.assert_array_equal(domain2[1].fields, first+second)
    np.testing.assert_array_equal(raw1.fields, first)
    np.testing.assert_array_equal(raw2.fields, second)
    return raw1, raw2, domain2, wanted


def test_failed_single_copies_get_one_iteration_sum(monkeypatch):
    rx = receiver()
    first, second, domain, wanted = independent_repeat_models(rx)
    # Design 174/194 keeps independent raw and combined observations in the
    # common engine, replacing the old decoder's three wrapper calls.
    # Component-budget zero admits the complete systematic tier only. Both
    # independent copies fail, while their channel sum repairs distinct errors.
    for raw in (first, second):
        result = engine.LongSearch([raw], component_budget=0).advance()
        assert result.frame is None and result.state == 'incomplete'
        assert result.completed_tier == 0 and result.component_calls == 0
    result = engine.LongSearch(domain, component_budget=0).advance()
    assert result.frame == wanted and result.completed_tier == 0
    assert result.selected_provenance.member_index == 1
    assert result.component_calls == 0 and result.complete
    assert len(result.provenance) == 2


@pytest.mark.parametrize('reference', [False, True])
def test_recorded_single_and_sum_models_reach_real_decoder_in_three_grants(monkeypatch, reference):
    if reference:
        monkeypatch.setattr(p4longnative, 'backend', None)
    tape, rx = audio(), receiver()
    initialize, advance = engine.LongSearch.__init__, engine.LongSearch.advance
    calls = []
    def limited(self, members, **kwargs):
        initialize(self, members, **dict(kwargs, iterations=1, component_budget=1))
    def counted(self, deadline=None):
        calls.append(len(self.members))
        return advance(self, deadline)
    monkeypatch.setattr(engine.LongSearch, '__init__', limited)
    monkeypatch.setattr(engine.LongSearch, 'advance', counted)
    jobs = []
    for at in (FIRST, SECOND):
        rx.clock.next_due = at
        lo = at-round(.050*FS)
        clean = tape[lo:at+round(3.355*FS)]
        seed = 27 if at == FIRST else 1027
        noisy = clean+np.random.default_rng(seed).normal(0, .4, len(clean))
        rx._long(noisy[:round(.860*FS)], lo, placed=True, partial=True)
        pending, _ = rx._long(noisy, lo, placed=True)
        assert pending.frame is None
        job = next(entry for key, entry in rx._long_jobs.items()
                   if key[0] == at and key[2] == 'normal')
        assert job['result'].state == 'incomplete'
        assert job['result'].completed_tier == 0 and job['result'].component_calls == 1
        assert not job['result'].identities
        assert job['search'].policy == 'live'
        jobs.append(job)
    first, second = (job['search'] for job in jobs)
    assert len(first.members) == 8 and len(second.members) == 10
    assert {(m.geometry.speedlevel, m.polarity) for m in second.members[8:]} == {
        (5, 1), (5, -1)}
    for combined in second.members[8:]:
        single = next(m for m in second.members[:8] if
            (m.geometry, m.polarity, m.weighting) ==
            (combined.geometry, combined.polarity, combined.weighting))
        previous = next(m for m in first.members if
            (m.geometry, m.polarity, m.weighting) ==
            (single.geometry, single.polarity, single.weighting))
        assert combined.label.startswith('copies:')
        np.testing.assert_array_equal(combined.fields, previous.fields+single.fields)
    retained = tuple(m.fields.copy() for m in second.members)
    second.component_budget = None
    assert rx._advance_long(jobs[1], deadline=unhurried(), relaxed=True) is None
    result = jobs[1]['result']
    assert calls == [8, 10, 10]
    assert result.complete and result.completed_tier == 1
    assert result.component_calls == 10 and len(result.provenance) == 20
    assert result.frame is not None and result.frame.status.byte == 2
    assert result.frame.data.startswith(PREFIX)
    assert {p.member_index for p in result.provenance} == set(range(10))
    selected = result.selected_provenance
    assert selected.member_index == 8 and selected.identity_index == 0
    assert second.members[selected.member_index].label.startswith('copies:')
    assert {p.member_index for p in result.provenance if p.identity_index == 0} == {8}
    assert rx.live_events[-1]['copies'] == 2
    for member, fields in zip(second.members, retained):
        np.testing.assert_array_equal(member.fields, fields)
    assert jobs[1]['obs'].frame is None and rx.registrations == []


@pytest.mark.parametrize('winner', ('single', 'sum'))
def test_both_single_and_sum_keep_independent_eight_iteration_priors(monkeypatch, winner):
    rx = receiver()
    first, single, domain, wanted = independent_repeat_models(rx, shared_error=True)
    job = engine.LongSearch(domain, policy='exhaustive', component_budget=0)
    urgent = job.advance()
    assert urgent.frame is None and urgent.completed_tier == 0
    assert urgent.state == 'incomplete' and job.last_tier == 9
    retained = tuple(member.fields.copy() for member in job.members)
    # Design 150/174: diagnostic exhaustive mode proves every raw/summed model
    # retains independent priors through eight iterations. Both real channel
    # models must recover; select each separately to prove neither was dropped.
    index = 0 if winner == 'single' else 1
    model = engine.LongSearch([domain[index]], policy='exhaustive')
    expected = model.advance()
    assert expected.frame == wanted and expected.completed_tier == 9
    assert expected.component_calls == 16
    job.component_budget = None
    pending = Observation(Kind.LONG, FS, 0, SECOND, SECOND, 3.282, -6.94, .9,
                          provisional=True, placed=True)
    entry = rx._start_long((SECOND, -6.94, 'normal'), domain, pending)
    entry.update(search=job, result=urgent)
    assert rx.refine(deadline=unhurried()) == []
    result = entry['result']
    assert result.frame == wanted and result.completed_tier == 9
    assert result.component_calls == 32
    assert {p.member_index for p in result.provenance if p.identity_index == 0} == {0, 1}
    np.testing.assert_array_equal(job.priors[index], model.priors[0])
    np.testing.assert_array_equal(job.first[index], model.first[0])
    for member, original in zip(job.members, retained):
        np.testing.assert_array_equal(member.fields, original)
    assert entry['obs'].frame is None and rx.registrations == []
    assert next(iter(rx._remembered_longs.values()))[0] == wanted


@pytest.mark.parametrize('payload,winner,seed', [
    ('recorded', 'single', 10), ('recorded', 'sum', 10),
    ('original-x', 'single', 8), ('original-x', 'sum', 68),
])
@pytest.mark.parametrize('reference', [False, True])
def test_failed_sum_keeps_single_copy_for_eight_iteration_refine(monkeypatch, winner, payload, seed, reference):
    if reference:
        monkeypatch.setattr(p4longnative, 'backend', None)
    tape, recorded_rx = audio(), receiver()
    lo = FIRST-round(.050*FS)
    recorded_rx._long(tape[lo:FIRST+round(.810*FS)], lo, placed=True, partial=True)
    observed, _ = recorded_rx._long(tape[lo:FIRST+round(3.355*FS)], lo, placed=True)
    wanted = observed.frame
    assert wanted is not None and wanted.status.byte == 2 and wanted.data.startswith(PREFIX)
    if payload == 'original-x':
        wanted = p4frame.decode_field(p4frame.encode_field(
            b'x'*203, p4frame.Status.from_byte(2)), data_bytes=203)
        assert wanted.data == b'x'*203 and wanted.status.byte == 2
    g = engine.geometry(5)
    wire = p4frame.encode_field(wanted.data, wanted.status)
    good = p4longtx.coded_bits(wire)
    # A valid convolutional codeword with a deliberately invalid CRC remains
    # unread after actual C1/C2 work. Pick a trailing CRC bit whose two parity
    # tails also satisfy the receiver's independent-region combining guard.
    candidates = []
    for bit in range(8*len(wire)-16, 8*len(wire)):
        changed = bytearray(wire)
        changed[bit//8] ^= 1 << (bit % 8)
        bad = p4longtx.coded_bits(bytes(changed))
        support = min(float(np.mean(good[a:b] == bad[a:b]))*2-1
                      for _, a, b in g.regions)
        candidates.append((support, bit, bytes(changed), bad))
    support, bit, damaged_wire, bad = max(candidates, key=lambda row: (row[0], row[1]))
    assert support >= .70 and p4frame.decode_field(damaged_wire, data_bytes=203) is None
    first_bits, current_bits = (bad, good) if winner == 'single' else (good, bad)
    # Shared interference preserves copy agreement while requiring all eight
    # turbo iterations; the stronger old copy controls the sum's CRC identity.
    noise = np.random.default_rng(seed).normal(0, 1.1, len(good))
    first = engine.LongMember(g, 2*(1-2.0*first_bits+noise), weighting='pilot')
    single = engine.LongMember(g, 1-2.0*current_bits+noise, weighting='pilot')
    rx = receiver()
    rx._normal_long_members((first,), ((.9, 0, -6.94),), FIRST)
    domain, _ = rx._normal_long_members((single,), ((.9, 0, -6.94),), SECOND)
    assert len(domain) == 2 and domain[0] is single
    np.testing.assert_array_equal(domain[1].fields, first.fields+single.fields)
    pending = Observation(Kind.LONG, FS, 0, SECOND, SECOND, 3.282, -6.94, .9,
                          provisional=True, placed=True)
    entry = rx._start_long((SECOND, -6.94, 'normal'), domain, pending)
    job = entry['search']
    assert job.policy == 'live' and job.last_tier == 9
    job.component_budget = 2*len(domain)
    assert rx._advance_long(entry, relaxed=True) is None
    assert entry['result'].state == 'incomplete' and entry['result'].completed_tier == 2
    assert entry['result'].component_calls == 4
    assert not any(p.crc_valid for p in entry['result'].provenance)
    retained = tuple(m.fields.copy() for m in job.members)
    job.component_budget = None
    assert rx.refine(deadline=unhurried()) == []
    result = entry['result']
    assert result.frame == wanted and result.completed_tier == 9
    assert result.component_calls == 32 and len(result.provenance) == 20
    selected = 0 if winner == 'single' else 1
    assert {p.member_index for p in result.provenance if p.identity_index == 0} == {selected}
    assert [(p.member_index, p.iteration) for p in result.provenance
            if p.method == 'turbo' and p.iteration == 8 and p.identity_index == 0] == [(selected, 8)]
    for member, original in zip(job.members, retained):
        np.testing.assert_array_equal(member.fields, original)
    assert entry['obs'].frame is None and rx.registrations == []
    assert next(iter(rx._remembered_longs.values()))[0] == wanted


def test_refine_admits_measured_pass_and_resumes_negative_polarity(monkeypatch):
    from hfmodem.shrike import p4receiver
    rx = receiver()
    g, _, coded, wanted = encoded(5)
    fields = 8*(1-2.0*coded)
    members = [engine.LongMember(g, -fields, polarity=1),
               engine.LongMember(g, fields, polarity=-1)]
    pending = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                          provisional=True, placed=True)
    entry = rx._start_long((FIRST, -6.94, 'normal'), members, pending)
    now, calls = [10.], []
    check = engine._check
    def measured(bits, member, member_index, tier, found):
        result = check(bits, member, member_index, tier, found)
        calls.append(member.polarity)
        now[0] += .052
        return result
    # A deterministic reference slice ends after one
    # admitted member and resumes without publishing a partial tier. Native
    # correctness is covered separately; no decoded frame is fabricated here.
    monkeypatch.setattr(p4longnative, 'backend', None)
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])
    monkeypatch.setattr(engine.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(engine, '_check', measured)
    monkeypatch.setattr(engine, '_python_check_seconds', lambda _: .052)
    assert rx.refine(deadline=now[0]) == [] and calls == []
    assert rx.refine(deadline=now[0]+.040) == []
    assert calls == [] and entry['result'].state == 'incomplete'
    assert entry['search'].member_index == 0 and entry['search'].tier == 0
    assert rx.refine(deadline=now[0]+.060) == []
    assert calls == [1] and entry['result'].state == 'incomplete'
    assert entry['search'].member_index == 1 and entry['result'].completed_tier == -1
    assert rx.refine(deadline=now[0]+.060) == []
    assert calls == [1, -1] and entry['result'].frame == wanted
    assert entry['result'].completed_tier == 0 and entry['obs'].frame is None
    assert next(iter(rx._remembered_longs.values()))[0] == wanted


def test_six_hz_step_uses_crc_to_discard_strong_training_alias():
    tape = audio()
    rx = receiver()
    start = FIRST-round(.050*FS)
    nominal = round((FIRST-start)*28800/FS)+p4short.GEOM
    for end, partial in ((FIRST+round(.810*FS), True),
                         (FIRST+round(3.355*FS), False)):
        z = p4rx.audio_to_baseband(tape[start:end], FS)
        z *= np.exp(-2j*np.pi*6*np.arange(len(z))/28800)
        rx._normal_baseband = z, start, nominal
        obs, _ = rx._long(tape[start:end], start, placed=True, partial=partial, alias=True)
    assert obs is not None and obs.frame is not None
    assert obs.frame.status.byte == 2 and obs.frame.data.startswith(PREFIX)
    assert -13.5 < obs.cfo_hz < -12.5


def test_alias_retry_requires_both_short_and_robust_failure(monkeypatch):
    rx = receiver()
    calls = []

    def placed(z, nominal, cfo, *, offsets, trainings):
        calls.append(cfo)
        return (.61 if cfo > 0 else .30), nominal, cfo

    monkeypatch.setattr(p4short, 'place_long', placed)
    silence = np.zeros(FIRST+round(.810*FS), np.float32)
    obs, _ = rx._long(silence[FIRST-round(.050*FS):], FIRST-round(.050*FS),
                      placed=True, partial=True, alias=False)
    assert obs is None and calls == [-6.94]
    obs, _ = rx._long(silence[FIRST-round(.050*FS):], FIRST-round(.050*FS),
                      placed=True, partial=True, alias=True)
    assert obs is not None and obs.provisional
    assert calls[1:] == pytest.approx([-6.94, -6.94-1800/239, -6.94+1800/239])


NEGATIVE_CONTROLS = [
    ('onair-1002-1013', 60, '59e22aa1f5b5b431872e3494f95436fde6dedff1d575f5ba4efbc60a1c8ffd6d'),
    ('onair-1001-1045', 77, 'ea7eac9ed69647c980eb2a45c300018f9ad2166bd8eb291faa370fb52afb83bc'),
]


def trace_digest(value):
    return hashlib.sha256(json.dumps(value,default=str,separators=(',', ':')).encode()).hexdigest()


@pytest.mark.parametrize('arm,stop,digest', NEGATIVE_CONTROLS)
def test_recorded_negative_controls_keep_their_decisions(arm, stop, digest, monkeypatch):
    if arm == 'onair-1002-1013':
        monkeypatch.setattr(p4turn.P4Turn, '_handback_seed', lambda self, reg, now: False)
    with neutral_keys() as neutral, waveform_keys() as waves:
        result = recorded(arm, stop*FS)
    assert not [row for row in result['heard'] if row['kind'] == Kind.LONG]
    rows = [(e['event'], e['at'], e.get('kind'), e.get('word'), e.get('state'), e.get('reason'))
            for e in result['events'] if e['event'] in ('Keyed', 'Decision')]
    pins=json.loads((Path(__file__).with_name('fixtures')/
                     'nak-stack1003e-controls.json').read_text())[arm]
    # The original no-request trace stays frozen; unread requests carry no stale ACK identity.
    assert pins['without_requests_decision_digest']==digest
    assert trace_digest(rows)==pins['decision_digest']
    assert trace_digest(protocol_heard(result['heard']))==pins['heard_digest']
    assert neutral==pins['neutral']
    neutral_at={e['at'] for e in neutral}
    ordinary=[e for e in prerender(result['emissions']) if e['audio_start'] not in neutral_at]
    ordinary_waves=[e for e in waves if e['at'] not in neutral_at]
    assert len(ordinary)==pins['ordinary_emission_count']
    assert trace_digest(ordinary)==pins['ordinary_emission_digest']
    assert len(ordinary_waves)==pins['ordinary_waveform_count']
    assert trace_digest(ordinary_waves)==pins['ordinary_waveform_digest']


@pytest.mark.parametrize('arm,stop,digest', NEGATIVE_CONTROLS)
def test_recorded_negatives_without_neutral_requests_keep_original_digest(arm,stop,digest,monkeypatch):
    if arm=='onair-1002-1013':
        monkeypatch.setattr(p4turn.P4Turn,'_handback_seed',lambda self,reg,now:False)
    without_neutral_requests(monkeypatch)
    with neutral_keys() as neutral, waveform_keys() as waves:
        result=recorded(arm,stop*FS)
    assert neutral==[]
    assert not [row for row in result['heard'] if row['kind']==Kind.LONG]
    rows=[(e['event'],e['at'],e.get('kind'),e.get('word'),e.get('state'),e.get('reason'))
          for e in result['events'] if e['event'] in ('Keyed','Decision')]
    assert trace_digest(rows)==digest
    pins=json.loads((Path(__file__).with_name('fixtures')/
                     'nak-stack1003e-controls.json').read_text())[arm]
    assert trace_digest(protocol_heard(result['heard']))==pins['without_requests_heard_digest']
    assert len(result['emissions'])==pins['ordinary_emission_count']
    assert trace_digest(prerender(result['emissions']))==pins['ordinary_emission_digest']
    assert len(waves)==pins['ordinary_waveform_count']
    assert trace_digest(waves)==pins['ordinary_waveform_digest']
