# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Opt-in associated v59/W27 trial; a fixed tape cannot prove acceptance."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hfmodem.shrike import onair, p4robusttx, p4turn
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Emission, Role
from hfmodem.shrike.p4turn import Proposal, State
from hfmodem.tests.evidence import CAPTURES
from hfmodem.tests.shrike import p4_chirp_deadline_support as support
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
from hfmodem.tests.shrike.test_p4turn import FS, ack_frame, machine, propose_ack


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def control(m, *, value=59, header='A', **changes):
    at = m.last.stream_sample + round(.330 * FS)
    obs = Observation(Kind.SHORT, FS, 0, at, at, .758, 0., .95,
                      value=value, header=header, walsh_peak=64, placed=True)
    return RegisteredObservation(replace(obs, **changes), Registered.NEW,
                                 responds_to=m.last.stream_sample)


def answering(policy='w27', **options):
    m, a = machine(role='irs', v59_answer=policy, normal_ack='alternate-packet', **options)
    propose_ack(m, ack_frame(FS, b'ordinary peer bytes', 1))
    m.keyed()
    return m, a


def proposal(m, reg, *, late=False):
    at = reg.observation.stream_sample
    now = at + round(.920 * FS) - m.lead + int(late)
    m.observe(reg, now=now)
    return m.decide(now)


def ownership(m, a):
    return (m.state, m.counter, m.ack_word, m.acked_counter, m._last_ack,
            m._last_body_ack, m._receive_epoch,
            m.outbox.inflight, bytes(a._outbuf), a._buffer_raw)


@pytest.mark.parametrize('header', ['A', 'B'])
def test_associated_v59_repeats_select_w27_without_data_ack_or_ownership_changes(header):
    m, a = answering()
    delivered = []
    m.adapter.host = type('Host', (), {'deliver': lambda self, data: delivered.append(data),
                                     'supervisor': lambda *a: None})()
    before = ownership(m, a)
    for _ in range(3):
        reg = control(m, header=header)
        p = proposal(m, reg)
        assert (p.kind, p.word, p.state, p.peer_control, p.ack_reason) == (
            Kind.WORD, 'W27', State.IRS_ANSWER, 59, None)
        assert p.stream_sample == reg.observation.stream_sample + round(.920 * FS)
        assert p.receive_epoch == m._receive_epoch
        assert p.ack_identity is None and not p.payload
        m.keyed()
        assert ownership(m, a) == before
    assert delivered == []
    assert len([e for e in m.link_events if e['event'] == 'experimental-v59-answer']) == 3


@pytest.mark.parametrize('value', [8, 13, 37, 49, 51, 53, 58, 60, 62])
def test_other_values_keep_current_answers(value):
    proposals = []
    for policy in ('current', 'w27'):
        m, _ = answering(policy)
        proposals.append(proposal(m, control(m, value=value)))
    assert proposals[0] == proposals[1]
    assert proposals[1] is None or proposals[1].word != 'W27'


@pytest.mark.parametrize('allowed', [True, False])
def test_queued_mail_keeps_its_existing_turn_permission(allowed):
    for policy in ('current', 'w27'):
        m, a = answering(policy, data='body', ladder='none')
        a.on_host_data(b'queued mail')
        m.allow_breakin = allowed
        before = ownership(m, a)
        # A robust control has neither checked data status nor an idle/turn
        # invitation. Break-in permission alone cannot turn it into a grant.
        for header in ('A', 'B', 'B'):
            p = proposal(m, control(m, header=header))
            assert p.state == State.IRS_ANSWER and p.kind == Kind.WORD
            assert p.word == ('X27' if policy == 'current' else 'W27')
            assert p.ack_identity is None and p.ack_reason is None
            assert not p.payload and not m.outbox.inflight
            m.keyed()
            assert ownership(m, a) == before
            assert bytes(a._outbuf) == b'queued mail' and a._buffer_raw == 11
        assert not [e for e in m.link_events if e['event'] in ('turn', 'data-reply-ack')]


@pytest.mark.parametrize('policy', ['current', 'w27'])
@pytest.mark.parametrize('evidence', ['idle', 'invitation', 'answered-repeat'])
def test_queued_mail_requests_on_checked_link_evidence_and_waits_for_grant(policy, evidence):
    m, a = answering(policy, data='body', ladder='none')
    queued = b'queued mail'
    a.on_host_data(queued)
    m.allow_breakin = True
    assert proposal(m, control(m)).word == ('X27' if policy == 'current' else 'W27')
    m.keyed()
    assert m.state == State.IRS_ANSWER and not m.outbox.inflight
    at = m.last.stream_sample+round(.330*FS)
    status, payload = {'idle': (2, b''), 'invitation': (0x42, b''),
                       'answered-repeat': (1, b'ordinary peer bytes')}[evidence]
    p = propose_ack(m, ack_frame(at, payload, status, Kind.BODY))
    assert (p.kind, p.state, p.status) == (Kind.BODY, State.IRS_BREAKIN, 0x40)
    assert p.stream_sample == at+round(.920*FS)
    assert not p.payload and not m.outbox.inflight
    assert bytes(a._outbuf) == queued and a._buffer_raw == len(queued)
    m.keyed()
    retry_at = m.last.stream_sample+round(1.25*FS)-m.lead
    m.silence(retry_at)
    retry = m.decide(retry_at)
    assert (retry.kind, retry.state, retry.status) == (Kind.BODY, State.IRS_BREAKIN, 0x40)
    assert not retry.payload and not m.outbox.inflight
    assert bytes(a._outbuf) == queued and a._buffer_raw == len(queued)
    m.keyed()
    # Only a checked BODY grant associated with the keyed request releases the
    # queued bytes into an outbound proposal. They still lack peer ACK credit.
    grant = replace(ack_frame(m.last.stream_sample+round(.330*FS), b'', 0x40, Kind.BODY),
                    responds_to=m.last.stream_sample)
    data = propose_ack(m, grant)
    assert data.kind == Kind.BODY and data.state == State.ISS_DATA_ACK
    assert data.payload == m.outbox.inflight == queued
    assert not a._outbuf and a._buffer_raw == len(queued)
    m.keyed()
    assert m.outbox.inflight == queued and a._buffer_raw == len(queued)


@pytest.mark.parametrize('policy', ['current', 'w27'])
@pytest.mark.parametrize('evidence', ['idle', 'answered-repeat'])
def test_checked_link_evidence_cannot_request_turn_without_breakin_permission(policy, evidence):
    m, a = answering(policy, data='body', ladder='none')
    a.on_host_data(b'queued mail')
    m.allow_breakin = False
    proposal(m, control(m))
    m.keyed()
    status, payload = (2, b'') if evidence == 'idle' else (1, b'ordinary peer bytes')
    p = propose_ack(m, ack_frame(m.last.stream_sample+round(.330*FS),
                                payload, status, Kind.BODY))
    assert p.kind == Kind.WORD and p.state == State.IRS_ANSWER and not p.payload
    m.keyed()
    assert not m.outbox.inflight and bytes(a._outbuf) == b'queued mail'
    assert a._buffer_raw == 11


@pytest.mark.parametrize('case', ['unplaced', 'weak', 'unexpected', 'provisional',
                                  'ambiguous', 'missing-header', 'bare-word',
                                  'unassociated', 'wrong-key', 'duplicate', 'expired',
                                  'contradicted', 'stale-generation', 'late'])
def test_invalid_or_unassociated_control_cannot_select_trial(case):
    m, _ = answering()
    reg = control(m)
    fields = {'unplaced': {'placed': False}, 'weak': {'weak': True},
              'unexpected': {'unexpected': True}, 'provisional': {'provisional': True},
              'ambiguous': {'header': '?'}, 'missing-header': {'header': None},
              'bare-word': {'word': 'V27'}}
    if case in fields:
        reg = replace(reg, observation=replace(reg.observation, **fields[case]))
    elif case in ('unassociated', 'wrong-key'):
        reg = replace(reg, responds_to=None if case == 'unassociated' else -1)
    elif case in ('duplicate', 'expired', 'contradicted'):
        reg = replace(reg, outcome=Registered(case))
    elif case == 'stale-generation':
        reg = replace(reg, response_generation=-1)
    p = proposal(m, reg, late=case == 'late')
    assert p is None or p.word != 'W27'
    assert not [e for e in m.link_events if e['event'] == 'experimental-v59-answer']


@pytest.mark.parametrize('state', [s for s in State if s != State.IRS_ANSWER])
def test_trial_does_not_apply_to_other_turn_states(state):
    m, _ = answering()
    m.state = state
    m.last = replace(m.last, state=state)
    p = proposal(m, control(m))
    assert p is None or p.word != 'W27'
    assert not [e for e in m.link_events if e['event'] == 'experimental-v59-answer']


@pytest.mark.parametrize('kind', [k for k in Kind if k != Kind.SHORT])
def test_trial_does_not_apply_to_other_packet_families(kind):
    proposals = []
    for policy in ('current', 'w27'):
        m, _ = answering(policy)
        reg = control(m)
        reg = replace(reg, observation=replace(reg.observation, kind=kind))
        proposals.append(proposal(m, reg))
        assert not [e for e in m.link_events if e['event'] == 'experimental-v59-answer']
    assert proposals[0] == proposals[1]


def test_stale_capture_epoch_cannot_key_trial_proposal():
    m, a = answering()
    before = ownership(m, a)
    assert proposal(m, control(m)).word == 'W27'
    m._receive_epoch += 1
    with pytest.raises(RuntimeError, match='stale receive answer'):
        m.keyed()
    assert m.last.word == 'V27'
    assert ownership(m, a)[:6] == before[:6]


def test_refused_trial_is_not_keyed_or_recorded_as_data_acceptance():
    m, a = answering()
    before = ownership(m, a)
    old_last = m.last
    assert proposal(m, control(m)).word == 'W27'
    m.refused('deadline')
    assert m.last == old_last
    assert ownership(m, a) == before
    assert not [e for e in m.link_events if e['event'] in ('data-reply-ack', 'turn')]


@pytest.mark.parametrize('bad', ['unknown', 'W27', None])
def test_constructor_rejects_unknown_policy(bad):
    with pytest.raises(ValueError, match='v59 answer policy'):
        machine(v59_answer=bad)


def replay_trial(policy=None, *, change_after_start=False):
    tx_factory = support.Tx

    def configured(stream):
        tx = tx_factory(stream)
        if policy is not None:
            tx.p4_link_options.p4_v59_answer = policy
        if change_after_start:
            key = tx._tx

            def changed(*args, **kwargs):
                tx.p4_link_options.p4_v59_answer = 'current'
                return key(*args, **kwargs)

            tx._tx = changed
        return tx

    with patch.object(support, 'Tx', configured):
        return support.replay('onair-1003-1123', stop=round(47.2 * FS))


@pytest.fixture(scope='module')
def recorded_default():
    return replay_trial()


def test_1123_default_serialization_and_emissions_match_prechange_flight(recorded_default):
    fixture = Path(__file__).with_name('fixtures') / 'chirp-deadline' / 'onair-1003-1123'
    baseline = json.loads((fixture / 'v59-default.json').read_text())
    emissions = support.prerender(recorded_default['emissions'])
    # The opening fixture predates the timing diagnostics. Every subsequent
    # actual key keeps the original default-zero schedule and waveform.
    for emission in emissions[1:]:
        assert emission.pop('answer_residual_samples') == 0
        assert emission.pop('output_enqueue_sample') is None
        assert emission.pop('nominal_due') == emission['audio_start']
        assert emission.pop('local_due') == emission['audio_start']
    nak = json.loads((fixture / 'v59-nak-default.json').read_text())
    requests = {e['at']: e for e in recorded_default['events']
                if e['event'] == 'Keyed' and e.get('current_slot_accepted') is False}
    assert list(requests) == [1292875, 1352875, 1472871, 2192831]
    # B1: each unread slot repeats its held target's keyed answer verbatim.
    held = {e['accepted_target']: e for e in recorded_default['events']
            if e['event'] == 'Keyed' and e.get('current_slot_accepted')}
    assert all(e['kind'] == Kind.WORD and e['state'] == State.IRS_ANSWER and not e.get('payload')
               and (e['word'], e['ack_reason'], e['ack_identity'])
               == tuple(held[e['accepted_target']][k] for k in ('word', 'ack_reason', 'ack_identity'))
               for e in requests.values())
    assert [e for e in support.prerender(recorded_default['emissions'])
            if e['audio_start'] in requests] == nak['neutral_emissions']
    # Preserve every original ordinary clock, renderer hash and field.
    emissions = [e for e in emissions if e['audio_start'] not in requests]
    assert emissions == baseline['emissions']
    events = support.prerender(e for e in recorded_default['events']
                               if e['event'] in ('Decision', 'Keyed'))
    for event in events:
        due = event['scheduled_at'] if event['event'] == 'Decision' else event['at']
        assert event.pop('answer_residual_samples') == 0
        assert event.pop('nominal_due') == due
        assert event.pop('local_due') == due
    events = [{k: v for k, v in e.items() if k not in ('work_ms', 'postdecision_work_ms')}
              for e in events]
    # B1 adds held-target metadata to keys and drops the request label; the
    # projection back to the prechange schema must still match the flight.
    held_fields = ('accepted_target', 'current_slot', 'current_slot_accepted', 'ack_identity')
    prechange = [e if e['event'] != 'Keyed' else {
        **{k: v for k, v in e.items() if k not in held_fields},
        **({'purpose': 'repeat-request', 'ack_reason': None} if e['at'] in requests else {})}
        for e in events]
    assert digest(prechange) == nak['decisions_and_keys_sha256']
    assert nak['baseline_heard_sha256'] == baseline['heard_sha256']
    assert digest(recorded_default['heard']) == nak['heard_sha256']
    measured = ('cfo_hz', 'coherence', 'lattice')
    assert [{k: h[k] for k in measured if k in h}
            for h in recorded_default['heard']] == nak['measured_receive_fields']
    # The first request refreshes Clock.cfo_hz from tracked CFO. Only these
    # three measurements differ; all registration/association/payload fields
    # retain the exact flight digest (not a regenerated candidate digest).
    assert digest([{k: v for k, v in h.items() if k not in measured}
                   for h in recorded_default['heard']]) == nak['heard_without_measurements_sha256']
    original = json.loads((CAPTURES / 'onair-1003-1123' / 'p4-probe.json').read_text())
    # The fixture Tx records the full cached wave; flight's writer trims its
    # tail. Both paths retain the hash of that same untrimmed renderer.
    assert [{k: v for k, v in e.items() if k != 'audio_end'}
            for e in emissions] == [
                {k: v for k, v in e.items() if k != 'audio_end'}
                for e in original['emissions']]
    assert [a['audio_end'] - b['audio_end'] for a, b in zip(
        emissions, original['emissions'], strict=True)] == [
            0, 88, 89, 88, 88, *([89] * 11)]


def test_1123_opt_in_changes_only_associated_unambiguous_v59_answers(recorded_default):
    trial = replay_trial('w27')
    assert trial['heard'] == recorded_default['heard']
    full = {h['onset_sample']: h for h in trial['heard']
            if h['kind'] == Kind.SHORT and not h['provisional']}
    requests = {e['at']: e for e in trial['events']
                if e['event'] == 'Keyed' and e.get('current_slot_accepted') is False}
    changed = []
    changed_requests = []
    for default, chosen in zip(recorded_default['emissions'], trial['emissions'], strict=True):
        if default == chosen:
            continue
        if chosen['audio_start'] in requests:
            assert chosen['audio_start'] == 2192831
            assert requests[chosen['audio_start']]['ack_reason'] is None
            assert requests[chosen['audio_start']]['state'] == State.IRS_ANSWER
            # The active policy already keyed W27 to the prior v59. This
            # request repeats it without accepting the unread next control.
            assert default['word'] == 'X27' and chosen['word'] == 'W27'
            assert chosen['waveform_sha256'] == hashlib.sha256(
                p4robusttx.robust_word('W27').tobytes()).hexdigest()
            assert {k: v for k, v in chosen.items() if k not in ('word', 'waveform_sha256', 'audio_end')} == {
                k: v for k, v in default.items() if k not in ('word', 'waveform_sha256', 'audio_end')}
            changed_requests.append(chosen)
            continue
        peer = full[chosen['audio_start'] - round(.920 * FS)]
        assert peer['value'] == 59 and peer['header'] in ('A', 'B')
        assert peer['responds_to'] is not None
        assert default['word'] == 'X27' and chosen['word'] == 'W27'
        assert chosen['waveform_sha256'] == hashlib.sha256(
            p4robusttx.robust_word('W27').tobytes()).hexdigest()
        assert chosen['audio_end'] - chosen['audio_start'] == len(p4robusttx.robust_word('W27'))
        assert {k: v for k, v in chosen.items() if k not in ('word', 'waveform_sha256', 'audio_end')} == {
            k: v for k, v in default.items() if k not in ('word', 'waveform_sha256', 'audio_end')}
        changed.append(peer)
    assert len(changed) == 8 and {h['header'] for h in changed} == {'A', 'B'}
    assert len(changed_requests) == 1
    assert not [h for h in trial['heard'] if h['onset_sample'] == 2148671
                and h['kind'] == Kind.SHORT and not h['provisional']]
    assert not [e for e in trial['events'] if e['event'] == 'Refused']
    assert trial['buffered'] == recorded_default['buffered'] == 0
    assert trial['source'] == recorded_default['source'] == ''
    assert trial['cycles'] == recorded_default['cycles']
    assert len([e for e in trial['events'] if e['event'] == 'experimental-v59-answer']) == 8
    default_keys = [e for e in recorded_default['events'] if e['event'] == 'Keyed']
    trial_keys = [e for e in trial['events'] if e['event'] == 'Keyed']
    assert [(e['at'], e['state'], e.get('status'), e.get('payload')) for e in trial_keys] == [
        (e['at'], e['state'], e.get('status'), e.get('payload')) for e in default_keys]


def test_trial_configuration_is_frozen_when_link_starts():
    trial = replay_trial('w27', change_after_start=True)
    keys = [e for e in trial['events'] if e['event'] == 'Keyed' and e.get('word') == 'W27']
    assert len([e for e in keys if e.get('current_slot_accepted') is not False]) == 8
    assert [(e['at'], e['ack_reason']) for e in keys
            if e.get('current_slot_accepted') is False] == [(2192831, None)]
    assert len([e for e in trial['events'] if e['event'] == 'experimental-v59-answer']) == 8


def test_driver_renders_w27_and_does_not_record_a_refused_key(monkeypatch):
    wave = p4robusttx.robust_short(59, 'A')
    audio = np.zeros(3 * FS, np.float32)
    at = round(.330 * FS)
    audio[at:at + len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_v59_answer = 'w27'
    tx.p4_link_options.p4_link_timeout = 2.5
    receiver = tx.sessrx.p4
    constructor = p4turn.P4Turn

    def seeded(*args, **kwargs):
        m = constructor(*args, **dict(kwargs, role='irs'))
        m._next = Proposal(Kind.WORD, 0, word='V27', state=State.IRS_ANSWER)
        m.keyed()
        receiver.note_emission(Emission(Kind.WORD, Role.IRS, 0, .242))
        return m

    monkeypatch.setattr(p4turn, 'P4Turn', seeded)
    # Model admission rejection; the radio writer must never receive the wave.
    stream.clamp_late = lambda at: max(0, stream.sample_now() + round(.250 * FS) - at)
    monkeypatch.setattr(tx, '_tx', lambda *a, **kw: pytest.fail('refused key reached writer'))
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert any(e['event'] == 'experimental-v59-answer' for e in tx.p4_probe.link_events)
    assert any(e['event'] == 'Refused' for e in tx.p4_probe.link_events)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    cache = tx.p4_probe.link_waveforms
    np.testing.assert_array_equal(cache[(Kind.WORD, 'W27', None, 'A', b'', 0)],
                                  p4robusttx.robust_word('W27'))


@pytest.mark.parametrize('policy', ['current', 'w27'])
@pytest.mark.parametrize('header', ['A', 'B'])
def test_repeat_after_v59_is_neutral_and_cannot_accept_another_control(policy, header):
    m, a = answering(policy, data='body', ladder='none')
    a.on_host_data(b'queued mail')
    reg = control(m, header=header)
    accepted = proposal(m, reg)
    assert accepted.word == ('X27' if policy == 'current' else 'W27')
    m.keyed()
    before = ownership(m, a), dict(m._packet_words), m._packet_ack, m._held_answer
    experiment_count = len([e for e in m.link_events if e['event'] == 'experimental-v59-answer'])
    for shift in (0, 100):
        due = m.last.stream_sample+round(1.25*FS)
        m.silence(due-m.lead, answer_due=due)
        request = m.decide(due-m.lead)
        # B1: the unread slot repeats the held v59 answer without accepting it.
        assert (request.kind, request.state, request.word, request.purpose) == (
            Kind.WORD, State.IRS_ANSWER, accepted.word, None)
        assert request.ack_reason is request.ack_identity is None
        assert (request.peer_control, request.accepted_target) == (59, reg.observation.stream_sample)
        assert not request.current_slot_accepted
        assert not request.payload and not m.outbox.inflight
        m.keyed(due+shift)
        assert (ownership(m, a), dict(m._packet_words), m._packet_ack, m._held_answer) == before
        assert bytes(a._outbuf) == b'queued mail' and a._buffer_raw == 11
    assert len([e for e in m.link_events if e['event'] == 'experimental-v59-answer']) == experiment_count
    assert not [e for e in m.link_events if e['event'] in ('turn', 'data-reply-ack')]
