# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""LONG work deadlines, current-copy agreement and stale-result admission."""
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import p4longdecode, p4longnative, p4longtx, p4receiver
from hfmodem.core import wav
from hfmodem.shrike.p4budget import LongDeadline, watch_read_admission
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, Observation, P4Receiver
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.test_p4_long_decode_1111 import encoded
from hfmodem.tests.shrike.test_p4turn import machine

FS = 48000


@pytest.mark.parametrize('sl,cost', [(5, .006), (7, .027), (10, .058)])
@pytest.mark.parametrize('backend_state', ['absent', 'failed', 'failing'])
def test_python_fallback_preserves_repeat_request_slot(monkeypatch, sl, cost, backend_state):
    turn, _ = machine(role='irs', prekey_lead=.040)
    at = FS
    received = at + round(3.280*FS)
    now = [0.]
    sample_now = lambda: received + round(now[0]*FS)
    budget = LongDeadline(FS, at, sample_now, lead_s=turn.lead/FS,
                          wall_now=lambda: now[0])
    rx = P4Receiver(FS, admitted=True)
    rx.long_deadline = lambda *a, **kw: budget
    obs = Observation(Kind.LONG, FS, 0, at, at, 3.280, 0., .9,
                      provisional=True, placed=True, speedlevel=f'SL{sl}-long')
    g = p4longdecode.geometry(sl)
    members = [p4longdecode.LongMember(g, np.random.default_rng(i).normal(size=g.serialized_size))
               for i in range(8)]
    entry = rx._start_long((at, 0., 'normal'), members, obs)
    step = p4longdecode._member_step

    def charged(*args):
        result = step(*args)
        if args[-1]:
            now[0] += cost * (1 if args[-1] in (1, 2) else 2)
        return result

    def broken(job, deadline):
        job.priors[:] = 99
        raise RuntimeError('backend failure')

    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4longdecode.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4longdecode, '_member_step', charged)
    monkeypatch.setattr(p4longnative, 'backend', None if backend_state == 'absent' else broken)
    entry['search'].backend_failed = backend_state == 'failed'
    assert rx._advance_long(entry) is None
    assert entry['result'].state == 'incomplete'
    assert now[0] <= budget.work_limit()
    assert budget.keyable()
    turn.observe(p4receiver.RegisteredObservation(obs, p4receiver.Registered.NEW), now=sample_now())
    due = at + round(3.400*FS)
    answer = turn.decide(due-turn.lead)
    assert answer.word == 'T27' and answer.purpose == 'repeat-request'
    assert answer.stream_sample == due and not answer.current_slot_accepted
    assert turn.can_key(answer)
    assert turn.keyed().stream_sample == due


def test_combined_member_cannot_supply_current_copy_agreement():
    g, _, coded, frame = encoded(5)
    old_fields, current_fields = 8*(1-2.0*coded), 1-2.0*coded
    _, lo, hi = g.regions[-1]
    old_fields[lo:lo+(hi-lo)//10] *= -1
    current_fields[lo:lo+(hi-lo)//5] *= -1
    old = p4longdecode.LongMember(g, old_fields, weighting='pilot')
    current = p4longdecode.LongMember(g, current_fields, weighting='pilot')
    rx = P4Receiver(FS, admitted=True)
    rx._remember_long(frame, old, FS)
    rx._normal_long_members((old,), ((.9, 0, 0.),), FS)
    at = FS+180000
    domain, _ = rx._normal_long_members((current,), ((.9, 0, 0.),), at)
    assert len(domain) == 2
    support = lambda m: float(m.fields[lo:hi] @ (1-2.0*coded[lo:hi])
                              / np.sum(abs(m.fields[lo:hi])))
    assert support(current) == 1-2*((hi-lo)//5)/(hi-lo)
    assert support(current) < .70 <= support(domain[1])
    assert rx._long_agreement(domain, at) is None
    obs = Observation(Kind.LONG, FS, 0, at, at, 3.282, 0., .9,
                      provisional=True, placed=True)
    job = rx._start_long((at, 0., 'normal'), domain, obs)
    recovered = rx._advance_long(job)
    assert recovered.frame == frame
    assert job['result'] is not None and job['result'].completed_tier == 0
    assert rx.live_events[-1]['state'] == 'unique'


@pytest.mark.parametrize('prepared', [False, True])
def test_sf8_cut_at_3250_ms_with_strong_noise_cannot_reset_silence(prepared):
    at = 3528586
    origin, end = at-24000, at+round(3.300*FS)
    path = require_input(CAPTURES/'onair-1002-2154/stream.flac')
    crop = wav.read(str(path), FS, start=origin, stop=end).copy()
    cut = at-origin+round(3.250*FS)
    crop[cut:] = np.random.default_rng(108).normal(0, .05, len(crop)-cut)
    rx = P4Receiver(FS, admitted=True, cfo_hz=.69, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, .69)
    rx.sf8_deadline = lambda _: monotonic()-1
    if prepared:
        assert rx.prepare_sf8(crop[:at-origin+round(3.230*FS)], origin) is not None
    obs = rx._sf8_long(crop, origin)
    assert obs is None
    records = rx.read_window(crop, origin, whole=True)
    assert not any(r.observation.kind == Kind.LONG and r.observation.placed
                   for r in records)
    assert not rx._remembered_longs
    rx.silent_cycles = 3
    assert rx.finish_cycle(at) == 4


def test_normal_long_prefix_is_ready_but_cannot_reset_silence():
    at, origin = FS, FS-2400
    wave = p4longtx.long_packet(b'hello', 0x61, FS)
    audio = np.r_[np.zeros(2400), wave[:round(.810*FS)]]
    rx = P4Receiver(FS, admitted=True)
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, 0.)
    pending, _ = rx._normal_long(audio, origin, partial=True)
    assert pending is not None and pending.provisional and not pending.placed
    assert pending.frame is None and pending.duration_s == .810
    rx.register(pending)
    rx.silent_cycles = 3
    assert rx.finish_cycle(at) == 4
    assert not rx.live_onsets and not rx._long_jobs


@pytest.mark.parametrize('stage,reserve', [(0, 2400), (1, 2880)])
def test_peer_watch_prefix_reserves_work_against_parent_decision(stage, reserve):
    # Watched support is only 1 ms before the independent
    # WORD owner's decision. Stages 0/1 formerly returned True unconditionally.
    decision, horizon = 180000, 180000-48
    assert watch_read_admission(stage, horizon, horizon-4800, decision, FS) == (False, reserve)
    assert watch_read_admission(stage, decision-reserve, 0, decision, FS) == (True, reserve)
    assert watch_read_admission(stage, decision-reserve+1, 0, decision, FS) == (False, reserve)
    assert watch_read_admission(stage, horizon, horizon-4800, decision, FS,
                                (3840,)) == (False, 3840)


def test_final_codeword_work_cannot_cross_publication_limit(monkeypatch):
    g, _, coded, frame = encoded(2)
    member = p4longdecode.LongMember(g, 8*(1-2.0*coded))
    rx = P4Receiver(FS, admitted=True)
    now = [0.]
    monkeypatch.setattr(p4longnative, 'backend', None)
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])
    rx.long_deadline = lambda *a, **kw: SimpleNamespace(
        work_limit=lambda: monotonic()+1, publishable=lambda: now[0] < .010)
    remember = rx._remember_long

    def charged(*args):
        result = remember(*args)
        now[0] += .020
        return result

    monkeypatch.setattr(rx, '_remember_long', charged)
    obs = Observation(Kind.LONG, FS, 0, FS, FS, 3.28, 0., .9,
                      provisional=True, placed=True)
    job = rx._start_long((FS, 0., 'sf16'), (member,), obs)
    assert rx._advance_long(job) is None
    assert job['result'].frame == frame and job['obs'].frame is None
    assert next(iter(rx._remembered_longs.values()))[0] == frame
    assert rx.live_events[-1]['crc_valid'] and not rx.live_events[-1]['timely']
