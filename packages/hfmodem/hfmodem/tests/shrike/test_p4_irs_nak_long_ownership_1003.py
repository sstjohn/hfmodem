# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A neutral request must respect every unresolved LONG receive interval."""
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace

import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.tests.shrike.test_p4turn import machine, ack_frame, body, word, FS


def long_candidate(at):
    return replace(ack_frame(at, b'long', 0, Kind.LONG).observation,
                   frame=None, provisional=True)


def body_candidate(m, rx, at, *, now=None):
    head = word('Y27', at, extent=.796).observation
    m.observe(rx.register(head), now=at+round(.25*FS) if now is None else now)
    candidate = replace(body(at).observation, frame=None, provisional=True)
    reg = rx.register(candidate)
    assert reg.outcome == Registered.ENRICHED
    m.observe(reg, now=at+round(.810*FS) if now is None else now)
    return reg.observation


@pytest.mark.parametrize('lead', [.040, .049, .080])
@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('prior_ack', [False, True])
def test_provisional_body_must_not_key_inside_unresolved_long(lead, queued, prior_ack):
    """Exact twelve cases from the independent pass6 reproducer."""
    m, _ = machine(role='irs', prekey_lead=lead)
    m.require_long_crc = True
    rx = P4Receiver(FS)
    if prior_ack:
        m.observe(body(FS, b'prior'))
        m.decide(FS+round(.920*FS)-m.lead)
        m.keyed()
    if queued:
        # This is a queued empty-cycle WORD proposal, not transmitted mail.
        m.silence(round(3.096*FS))
        assert m.decide(round(3.096*FS)).purpose == None
    onset = round(2.25*FS)
    long = long_candidate(onset)
    m.observe(rx.register(long), now=round(3.10*FS))
    assert m._next is None
    at = round(3.5*FS)
    candidate = body_candidate(m, rx, at)
    history = m.last, m._held_answer, m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack
    due = at+round(m.t[Kind.BODY]*FS)
    decision = due-m.lead
    p = m.decide(decision)
    if p is None:
        m.silence(decision, answer_due=due)
        p = m.decide(decision)
    if p is not None:
        live = SimpleNamespace(pos=decision, clamp_late=lambda at:max(0, decision+1796-at))
        grid = onair._MasterGrid(due, round(1.25*FS), 0,
                               packet_n=round(.244*FS), cs_n=round(.244*FS), d_max_n=round(.1*FS))
        assert onair._keyable_slot(live, grid, 0, 0, listen=False, shift=False) == 0
        assert m.keyed().stream_sample == due
    assert p is None
    assert m.pending[at]['body'] == candidate and not m.pending[at]['accepted']
    assert onset not in m._decided
    assert (m.last, m._held_answer, m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack) == history
    long_due = onset+round(m.t[Kind.LONG]*FS)
    p = m.decide(long_due-m.lead)
    assert (p.stream_sample, p.word, p.purpose) == (long_due, 'X27' if prior_ack else 'V27', None)
    m.keyed()
    assert (m._held_answer, m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack) == history[1:]


@pytest.mark.parametrize('order', list(permutations(('owner', 'body', 'future'))))
def test_all_pending_long_intervals_are_checked_in_any_insertion_order(order):
    m, _ = machine(role='irs')
    m.require_long_crc = True
    rx = P4Receiver(FS)
    onset, at, future = round(2.25*FS), round(3.5*FS), 6*FS
    for event in order:
        if event == 'body':
            body_candidate(m, rx, at)
        else:
            obs = long_candidate(onset if event == 'owner' else future)
            m.observe(rx.register(obs), now=round(4.31*FS))
    due = at+round(m.t[Kind.BODY]*FS)
    assert m.decide(due-m.lead) is None
    m.silence(due-m.lead, answer_due=due)
    assert m.decide(due-m.lead) is None
    # A later nominated interval does not own the earlier LONG's answer.
    own_due = onset+round(m.t[Kind.LONG]*FS)
    proposal = m.decide(own_due-m.lead)
    assert (proposal.stream_sample, proposal.word, proposal.purpose) == (own_due, 'V27', None)


@pytest.mark.parametrize('resolution', ['expired', 'contradicted', 'decided'])
def test_resolved_long_does_not_hold_a_later_body_request_forever(resolution):
    m, _ = machine(role='irs')
    m.require_long_crc = True
    rx = P4Receiver(FS)
    onset, at = round(2.25*FS), round(3.5*FS)
    long = long_candidate(onset)
    m.observe(rx.register(long), now=round(3.10*FS))
    body_candidate(m, rx, at)
    if resolution == 'expired':
        m.observe(RegisteredObservation(long, Registered.EXPIRED, long), now=round(4.31*FS))
        assert onset not in m.pending
    elif resolution == 'contradicted':
        normal = replace(ack_frame(onset, b'normal', 0, Kind.NORMAL).observation,
                         frame=None, provisional=False)
        reg = rx.register(normal)
        assert reg.outcome == Registered.CONTRADICTED and reg.replaced == long
        m.observe(reg, now=round(4.31*FS))
        assert m.pending[onset]['body'].kind == Kind.NORMAL
    else:
        # A settled owner is retained for deduplication, not future admission.
        m._decided.add(onset)
    due = at+round(m.t[Kind.BODY]*FS)
    p = m.decide(due-m.lead)
    assert (p.stream_sample, p.word, p.purpose) == (due, 'V27', None)
    assert m.pending[at]['accepted'] is False
    m.keyed()
    assert m.acked_counter is None and m._held_answer is None


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_long_answer_boundary_is_excluded_from_receive_ownership(delta):
    m, _ = machine(role='irs')
    m.require_long_crc = True
    rx = P4Receiver(FS)
    onset = round(2.25*FS)
    own_due = onset+round(m.t[Kind.LONG]*FS)
    m.observe(rx.register(long_candidate(onset)), now=round(3.10*FS))
    due = own_due+delta
    at = due-round(m.t[Kind.BODY]*FS)
    body_candidate(m, rx, at)
    proposal = m.decide(due-m.lead)
    if delta < 0:
        assert proposal is None
        proposal = m.decide(own_due-m.lead)
        assert proposal.stream_sample == own_due
    else:
        assert proposal.stream_sample == due
    assert proposal.purpose == None and proposal.word == 'V27'


def test_contradiction_to_long_keeps_ownership_until_its_own_due():
    m, _ = machine(role='irs')
    m.require_long_crc = True
    rx = P4Receiver(FS)
    onset, at = round(2.25*FS), round(3.5*FS)
    old = replace(ack_frame(onset, b'normal', 0, Kind.NORMAL).observation,
                  frame=None, provisional=True)
    m.observe(rx.register(old), now=round(3.10*FS))
    long = replace(long_candidate(onset), provisional=False)
    reg = rx.register(long)
    assert reg.outcome == Registered.CONTRADICTED
    m.observe(reg, now=round(3.10*FS))
    body_candidate(m, rx, at)
    due = at+round(m.t[Kind.BODY]*FS)
    assert m.decide(due-m.lead) is None
    long_due = onset+round(m.t[Kind.LONG]*FS)
    assert m.decide(long_due-m.lead).stream_sample == long_due
