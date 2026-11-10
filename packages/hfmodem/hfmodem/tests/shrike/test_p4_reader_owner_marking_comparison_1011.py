# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Synthetic competing classification hypotheses through genuine registry/admission."""
from dataclasses import replace, asdict
from types import SimpleNamespace
import pytest
from hfmodem.shrike import p4robusttx
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Emission, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, FS, setup, packet
from hfmodem.tests.shrike.reader_owner_support_1011 import record


@pytest.mark.parametrize('refuse', [False])
def test_d2_exact_versions_complete_retained_candidate_marking(refuse):
    m, a = setup()
    m.managed_application = False
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, 0.)
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    old_first = AT + 3840
    history = Emission(Kind.SHORT, Role.ISS, old_first - 44880, .758,
                       train=True, train_words=frozenset(('X27', 'Y27', 'Z27')))
    rx.note_emission(history)
    source = b'candidate source\r'.ljust(22, b'\x1e')

    def candidates(n):
        full = packet(n, Kind.CHIRP, 0x61, source).observation
        left = replace(full, stream_sample=n-3840, first_chip=n-3840,
                       frame=None, provisional=True)
        right = replace(full, kind=Kind.WORD, stream_sample=n+3840,
                        first_chip=n+3840, frame=None, word='X27', provisional=True)
        l = rx.register(left)
        m.observe(l, now=n+12000)
        r = rx.register(right)
        m.observe(r, now=n+12000)
        accepted = rx.register(full)
        record('marking_registry_preconditions', center=n,
               left=asdict(l), right=asdict(r), accepted=asdict(accepted),
               evidence=dict(rx._response_evidence), responds=dict(rx._responds_to),
               observations=[asdict(o) for o in rx.observations])
        assert right.stream_sample-left.stream_sample == 7680
        assert accepted.replaced is left
        assert any(o is right for o in rx.observations)
        assert not any(o is left for o in rx.observations)
        assert accepted.observation is full
        m.observe(accepted, now=n+158400)
        assert m.pending[n]['accepted'] and m.pending[n]['body'] is full
        return right, r

    old, old_reg = candidates(AT)
    assert old_reg.responds_to == history.stream_sample
    assert old_reg.response_generation == rx.train_lease.generation
    assert old_reg.response_slot == 0 and old_reg.response_owner is rx
    old_generation = old_reg.response_generation
    p = m.decide(AT+163200-m.lead)
    assert p.kind == Kind.WORD and p.state == State.IRS_ANSWER
    sent = m.keyed(p.stream_sample)
    rx.note_emission(Emission(sent.kind, Role.IRS, sent.stream_sample,
        len(p4robusttx.robust_word(sent.word))/FS, accepted_target=sent.accepted_target,
        current_slot=sent.current_slot, current_slot_accepted=sent.current_slot_accepted))
    expected_pair = (old_generation, rx._train_generation)
    assert rx._answered[old.stream_sample] == sent.stream_sample
    assert rx._answered_generations[old.stream_sample] == expected_pair
    assert rx.train_lease is None
    old_binding = dict(rx._answered), dict(rx._answered_generations)
    n = sent.stream_sample + 15840
    new, new_reg = candidates(n)
    assert new_reg.responds_to is None and new_reg.response_generation is None
    assert rx._response_evidence[new.stream_sample] is None
    p = m.decide(n+163200-m.lead)
    record('marking_repeat_selection', refuse=refuse, proposal=asdict(p),
           old_binding=old_binding, current_candidate=new.stream_sample,
           delivered=[b.hex() for b in delivered])
    import os
    frozen = os.environ.get('CONDUCTOR_TEST_TREE') == '/private/tmp/hfmodem-chirp-second-frozen-1011'
    target = AT if frozen else n
    assert p.accepted_target == target and p.current_slot == target and p.current_slot_accepted
    assert m._accepted_proposal is None if frozen else m._accepted_proposal[0] is p
    if refuse:
        m.refused('deadline')
        p = m.decide(n+343200-m.lead)
        assert p.accepted_target == n and not p.current_slot_accepted
        third = replace(new, stream_sample=n+180000+3840, first_chip=n+180000+3840)
        reg = rx.register(third)
        m.observe(reg, now=p.stream_sample-m.lead)
        assert any(o is third for o in rx.observations)
    assert m.can_key(p)
    sent2 = m.keyed(p.stream_sample)
    before_hold = rx.hold.epoch
    rx.note_emission(Emission(sent2.kind, Role.IRS, sent2.stream_sample,
        len(p4robusttx.robust_word(sent2.word))/FS, accepted_target=sent2.accepted_target,
        current_slot=sent2.current_slot, current_slot_accepted=sent2.current_slot_accepted))
    record('marking_expected_version_final', frozen=frozen, refuse=refuse, answered=dict(rx._answered),
           generations=dict(rx._answered_generations), pair=expected_pair,
           hold_before=before_hold, hold_after=rx.hold.epoch, proposal=asdict(sent2))
    assert rx._answered[old.stream_sample] == old_binding[0][old.stream_sample]
    assert rx._answered_generations == old_binding[1]
    if refuse:
        assert new.stream_sample not in rx._answered and third.stream_sample not in rx._answered
        assert rx.hold.epoch == before_hold
    else:
        if frozen:
            assert new.stream_sample not in rx._answered
        else:
            assert rx._answered[new.stream_sample] == sent2.stream_sample
        assert rx.hold.epoch == before_hold+1
    assert delivered == [b'candidate source\r']
    assert a.tx_acked_bytes == 0 and not m.outbox.inflight
