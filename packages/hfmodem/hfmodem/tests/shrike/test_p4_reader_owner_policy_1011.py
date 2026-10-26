# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Common P3 IRS reception and honest P2 seed policy/physical-owner controls."""
from dataclasses import asdict, replace
import os
from types import SimpleNamespace
import pytest
from hfmodem.shrike import p4robusttx, p4breakintx, p4shorttx
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Emission, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, FS, setup, packet
from hfmodem.tests.shrike.test_p4_second_delta_parent_1011 import Recorder
from hfmodem.tests.shrike.reader_owner_support_1011 import record

class PolicyRecorder(Recorder):
    def snapshot(self, label, reg=None):
        super().snapshot(label, reg)
        record('policy_raw_snapshot', case=self.case, row=self.rows[-1])


VARIANTS = ['good', 'crc', 'provisional', 'weak_header', 'weak_full',
            'header_full', 'displaced_duplicates', 'unassociated']


def emit_actual(m, rx, p):
    sent = m.keyed(p.stream_sample)
    waveform = (p4robusttx.robust_word(sent.word) if sent.kind == Kind.WORD else
                p4shorttx.short_packet(sent.payload, sent.status) if sent.kind == Kind.NORMAL else
                p4breakintx.changeover_packet(sent.payload, sent.status))
    role = Role.IRS if sent.kind == Kind.WORD and sent.state != State.IRS_BREAKIN else Role.ISS
    em = Emission(sent.kind, role, sent.stream_sample, len(waveform)/FS,
                  packet_reply=sent.kind in (Kind.BODY, Kind.NORMAL) and sent.state != State.ISS_CLOSE,
                  accepted_target=sent.accepted_target, current_slot=sent.current_slot,
                  current_slot_accepted=sent.current_slot_accepted)
    rx.note_emission(em)
    return sent, em


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('context', ['common-p3-irs', 'actual-p2-seed'])
@pytest.mark.parametrize('variant', VARIANTS)
def test_c4_policy_source_and_actual_association(kind, context, variant):
    parent = os.environ.get('CONDUCTOR_TEST_TREE') == '/private/tmp/hfmodem-chirp-second-eddd-1011'
    m, a = setup()
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, -.19)
    r = PolicyRecorder(m, rx, a)
    r.case = [kind, context, variant]
    m.adapter.host = SimpleNamespace(deliver=r.deliver, supervisor=lambda *a: None)
    width = 40 if kind == Kind.LONG else 22
    seed_status = 0x62 if context == 'common-p3-irs' else 0x42
    seed_data = (b'\x1e'*width if context == 'common-p3-irs' else b'seed source\r'.ljust(width, b'\x1e'))
    seed = rx.register(packet(AT, kind, seed_status, seed_data).observation)
    r.snapshot('seed_registered', seed)
    m.observe(seed, now=AT+158400)
    r.snapshot('seed_observed', seed)
    pending = m.pending[AT]
    assert pending['accepted'] and pending['body'] is seed.observation
    if context == 'actual-p2-seed':
        if parent:
            assert 'answer' not in pending and m._held_answer is None
        else:
            assert pending['answer'].kind == Kind.WORD
            assert m._held_answer.answer is pending['answer']
            assert m._held_answer.context['body'] is seed.observation
    p = m.decide(KEY-m.lead)
    r.snapshot('seed_selected')
    assert p is not None
    if context == 'actual-p2-seed' and parent:
        assert (p.kind, p.status, p.payload, p.state) == (Kind.BODY, 0x40, b'', State.ISS_GRANT)
        assert (p.receive_epoch, p.source_root, p.accepted_target, p.current_slot) == (None, None, None, None)
    else:
        assert (p.kind, p.word, p.state) == (Kind.WORD, 'X27' if kind == Kind.LONG else 'V27', State.IRS_ANSWER)
        assert p.receive_epoch == pending['receive_epoch'] and p.source_root == pending['source_root']
        assert p.accepted_target == p.current_slot == AT and p.current_slot_accepted
    assert p.ack_identity == (seed_status, seed_data)
    assert m._accepted_proposal[0] is p and m._accepted_proposal[1] == AT
    assert m._accepted_proposal[2]['body'] is seed.observation
    epoch, active, boundary = m._receive_epoch, m._receive_active, m._receive_boundary
    sent, em = emit_actual(m, rx, p)
    r.snapshot('seed_keyed_actual_emission')
    if context == 'actual-p2-seed' and parent:
        assert em.kind == Kind.BODY and em.role == Role.ISS and em.packet_reply
        assert rx.clock.expects == Expect.WORD and m.peer_phase is None
        assert not m._receive_active and m._receive_epoch == epoch
        assert m._handoff_ticket is not None
    else:
        assert em.kind == Kind.WORD and em.role == Role.IRS and not em.packet_reply
        assert rx.clock.expects == Expect.PACKET and m._receive_active == active
        assert m._receive_epoch == epoch and m.peer_phase is not None
        assert m._receive_boundary == boundary
        assert m._handoff_ticket is None
    associated_after = .330 if em.role == Role.IRS else .935
    at = em.stream_sample+round(associated_after*FS)+(2400 if variant == 'unassociated' else 0)
    # This explicit same-absolute-onset control never retimes either version.
    absolute = packet(KEY+15840, kind, 0x42, b'ordinary source\r'.ljust(width, b'\x1e')).observation
    target = rx._response_target(absolute)
    assert target == (None if em.role == Role.ISS else KEY)
    o = packet(at, kind, 0x42, b'ordinary source\r'.ljust(width, b'\x1e')).observation
    if variant in ('crc', 'provisional', 'weak_header'):
        observations = [replace(o, frame=None, provisional=variant == 'provisional', weak=variant == 'weak_header')]
    elif variant == 'weak_full':
        observations = [replace(o, weak=True)]
    elif variant in ('header_full', 'displaced_duplicates'):
        full = replace(o, stream_sample=at+24, first_chip=at+24) if variant == 'displaced_duplicates' else o
        observations = [replace(o, frame=None), full, full, full]
    else:
        observations = [o]
    source_before = len(r.delivered)
    outcomes = []
    for obs in observations:
        reg = rx.register(obs)
        assert reg.responds_to == (None if variant == 'unassociated' else KEY)
        outcomes.append((reg.outcome, reg.observation.stream_sample,
                         reg.replaced.stream_sample if reg.replaced else None))
        r.snapshot('tested_registered', reg)
        m.observe(reg, now=at+158400)
        r.snapshot('tested_observed', reg)
    r.snapshot('tested_before_decide')
    current = min((s for s in m.pending if s not in m._decided), default=at)
    p = m.decide(current+163200-m.lead)
    r.snapshot('tested_selected')
    selected = asdict(p) if p else None
    if p is not None and m.can_key(p):
        later, later_em = emit_actual(m, rx, p)
        r.snapshot('tested_keyed_actual_emission')
    else:
        later = later_em = None
    complete_good = variant in ('good', 'header_full', 'displaced_duplicates', 'unassociated')
    if complete_good:
        assert r.delivered[source_before:] == [b'ordinary source\r']
        if parent:
            # Common IRS P2 reception selects the old generic grant. A seeded
            # parent already owns ISS and runs its associated release path.
            assert later is not None
            if context == 'actual-p2-seed' and variant != 'unassociated':
                assert (later.kind, later.state, later.status, later.payload) == (Kind.NORMAL, State.ISS_HOLD, 0, b'')
            else:
                assert (later.kind, later.state, later.status, later.payload) == (Kind.BODY, State.ISS_GRANT, 0x40, b'')
        else:
            assert later is not None and (later.kind, later.state) == (Kind.WORD, State.IRS_ANSWER)
            assert later.word == ('X27' if kind == Kind.LONG else 'V27')
            assert later.accepted_target == current and later.current_slot == current
            assert later.ack_identity[1] == o.frame.data
    elif variant != 'weak_full':
        assert r.delivered[source_before:] == []
    weak_rejected = kind == Kind.CHIRP and variant in ('weak_header', 'weak_full')
    unread_case = variant in ('crc', 'provisional') or kind == Kind.LONG and variant == 'weak_header'
    accepted_case = complete_good or kind == Kind.LONG and variant == 'weak_full'
    if weak_rejected:
        assert p is None and current not in m.pending
        assert m.unread == 0 and current not in m._decided and current not in m._dropped
        assert r.delivered[source_before:] == []
    elif unread_case:
        assert m.unread == 1 and current in m._decided and current in m._dropped
        if parent and context == 'actual-p2-seed':
            assert p is None
        else:
            assert p.kind == Kind.WORD and p.state == State.IRS_ANSWER
            assert p.accepted_target == AT and p.current_slot == current and not p.current_slot_accepted
            assert p.ack_identity == (seed_status, seed_data)
    elif accepted_case:
        assert m.unread == 0 and current in m._decided and current not in m._dropped
        assert r.delivered[source_before:] == [b'ordinary source\r']
        if kind == Kind.LONG and variant == 'weak_full':
            if parent:
                assert p.kind == (Kind.NORMAL if context == 'actual-p2-seed' else Kind.BODY)
                assert p.ack_identity == (0x42, o.frame.data)
                assert p.receive_epoch is None and p.source_root is None and p.accepted_target is None and p.current_slot is None
            else:
                # Weak counted LONG is ordinary accepted source, but cannot
                # install the new qualified held answer. _grant recovers the
                # existing source; it must not manufacture current acceptance.
                assert p.kind == Kind.WORD and p.accepted_target == AT
                assert p.current_slot is None and not p.current_slot_accepted
                assert p.ack_identity == (seed_status, seed_data)
    assert m._deferred is None
    if variant == 'displaced_duplicates':
        assert outcomes[-1][0] == outcomes[-2][0] == 'duplicate'
    assert a.tx_acked_bytes == 0 and not m.outbox.inflight and bytes(a._outbuf) == b''
    record('c4_policy_matrix', case=[kind, context, variant], parent=parent,
           context_label=context if o.frame else context+' frame-less classification (no protocol bits)',
           rows=r.rows, seed_proposal=asdict(sent), seed_emission=asdict(em),
           tested_onset=at, associated_after=associated_after, same_absolute_target=target,
           selected=selected, final=asdict(later) if later else None,
           final_emission=asdict(later_em) if later_em else None, outcomes=outcomes,
           delivered=[b.hex() for b in r.delivered], decided=sorted(m._decided), dropped=sorted(m._dropped),
           unread=m.unread, queued=bytes(a._outbuf).hex(), inflight=m.outbox.inflight.hex(), credited=a.tx_acked_bytes,
           source_contract='eddd _accepted_answer 2003-2034 / _grant 2491-2507 / keyed 2555-2584; final _accepted_answer 2216-2245 / decide 2498-2505 / keyed')
