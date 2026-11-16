# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import FrozenInstanceError, replace
import socket
from threading import Thread

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_controller import Controller, Disposition
from hfmodem.contact.chat_fake import (
    ActionExpectation, BBS_ATTACK, CONNECTED, FakeClock, FakeSession,
    SCENARIOS, ScenarioEvent, ScenarioStep, patched,
)


def request(fake, action, *, id='r1', actor='view-a', generation=None):
    return s.Request(id, 'controller', fake.state.link.generation if generation is None else generation,
                     actor, None, None, action)


def connected():
    fake = FakeSession()
    fake.state = patched(fake.state, CONNECTED)
    fake.poll(0)
    fake._sync_generation()
    return fake


def execute(fake, action, **kwargs):
    return fake.execute(request(fake, action, **kwargs))


def advance(fake, seconds):
    barrier = fake.advance(seconds)
    fake.poll(0)
    assert barrier.result(0) == fake.poll_count


def terminal(fake, request_id):
    return [event.payload for event in fake.event_trace
            if event.request_id == request_id and isinstance(event.payload, Disposition)
            and event.payload.state in ('done', 'refused', 'failed', 'cancelled', 'unknown')]


def test_scenarios_are_shared_immutable_values_and_owner_only():
    assert set(SCENARIOS) == {'idle-monitoring', 'incoming-call', 'two-way-chat', 'qsy',
                              'vmail-email', 'bbs', 'broadcast', 'failure-close'}
    step = SCENARIOS['idle-monitoring'][0]
    with pytest.raises(FrozenInstanceError):
        step.at = 2
    with pytest.raises(TypeError, match='immutable'):
        FakeSession(scenario=(ScenarioStep(0, 'poll', None, (('requests', []),), ()),))
    fake = FakeSession()
    fake.project()
    errors = []

    def outside():
        for method in (fake.project, lambda: fake.preflight(s.ChatSend('x')), lambda: fake.poll(0), fake.quit):
            try:
                method()
            except RuntimeError as error:
                errors.append(str(error))
    thread = Thread(target=outside)
    thread.start()
    thread.join()
    assert errors == ['fake SessionPort accessed outside owner'] * 4


def test_preflight_has_no_effect_and_unexpected_actions_fail():
    fake = FakeSession()
    before = fake.project()
    assert fake.preflight(s.ChatSend('hello')).reason_code == 'link-required'
    assert fake.project() == before
    assert fake.trace == fake.effects == fake.event_trace == []
    scripted = FakeSession(scenario=(ScenarioStep(0, 'action', s.VerboseSet(True), (), ()),))
    with pytest.raises(AssertionError):
        execute(scripted, s.AutoSet(False))
    assert scripted.trace == []
    execute(scripted, s.VerboseSet(True))
    scripted.assert_complete()


def test_full_request_expectation_and_bound_authority():
    fake = connected()
    action = s.ChatSend('hello')
    expected = ActionExpectation(action, 1, 1, 'view-a', 'lease', 7)
    fake._steps = [ScenarioStep(0, 'chat.send', expected, (), ())]
    req = replace(request(fake, action), control_id='lease', control_generation=7)
    result = fake.execute(req)
    assert result.state == 'done'
    assert fake.trace == [req]
    assert fake.effects == [('r1', s.Effect.X)]
    assert result.result.text_id == fake.event_trace[0].payload.id


def test_injected_clock_and_explicit_barrier_release():
    fake = FakeSession(FakeClock(20), (ScenarioStep(21, 'barrier:continue', None,
        (('policy.verbose', True),), ()),))
    advance(fake, 1)
    assert not fake.state.policy.verbose
    released = fake.release('continue')
    fake.poll(0)
    assert released.done() and fake.state.policy.verbose
    with pytest.raises(ValueError):
        fake.clock.advance(-1)
    fake.assert_complete()


def test_idle_monitoring_heard_anonymous_slots_age_and_observed_coverage():
    fake = FakeSession(scenario='idle-monitoring')
    fake.poll(0)
    snap = fake.project()
    assert snap.receive.heard[0].n == 1 and snap.receive.heard[1].beacon
    assert snap.receive.slots[0].calls == ('?',)
    assert snap.receive.watched == (12, 13)
    assert snap.receive.coverage[0].share_60s == .05
    assert snap.receive.wide_offset == 125.
    observed = snap.receive.observed_at
    advance(fake, 1)
    assert fake.state.policy.auto_state == 'off'
    advance(fake, 1)
    assert fake.state.policy.auto_state == 'suspended'
    advance(fake, 59)
    assert fake.state.receive.coverage == () and fake.state.receive.observed_at == observed
    assert fake.trace == fake.effects == []
    fake.assert_complete()


def test_incoming_call_keeps_away_and_queued_directory_on_same_generation():
    fake = FakeSession(scenario='incoming-call')
    fake.poll(0)
    assert fake.state.link.pending and fake.state.link.generation == 1
    advance(fake, 1)
    assert fake.state.link.connected and fake.state.link.peer == 'N0XYZ'
    assert fake.state.identity.channel_hz == fake.state.identity.calling_hz
    assert fake.state.application.bbs_list_generation == 1
    assert fake.state.policy.away_owed_generation == 1
    advance(fake, 1)
    assert fake.state.link.generation == 1 and fake.state.policy.away_owed_generation is None
    assert [e.generation for e in fake.event_trace if isinstance(e.payload, s.TextRecord)] == [1, 1]
    assert fake.trace == []
    fake.assert_complete()


def test_two_way_chat_queued_partial_ack_unknown_and_aggregate_discard():
    fake = FakeSession(scenario='two-way-chat')
    fake.poll(0)
    result = execute(fake, s.ChatSend('hello'), id='send-1')
    assert result.state == 'done' and isinstance(result.result, s.WriteAccepted)
    advance(fake, .25)
    assert fake.state.tx.host_ptt and fake.state.tx.buffer_bytes == 0 and fake.state.tx.unreported
    for _ in range(3):
        advance(fake, .25)
    outcomes = [e.payload for e in fake.event_trace if isinstance(e.payload, s.TxTextOutcome)]
    assert [o.status for o in outcomes] == ['queued', 'partial', 'acknowledged', 'unknown']
    assert outcomes[1].acknowledged_bytes == 4 and outcomes[2].acknowledged_bytes == 9
    assert outcomes[-1].attribution == 'gap'
    assert any(isinstance(e.payload, s.TxCredit) and e.payload.disposition == 'discarded'
               for e in fake.event_trace)
    assert len(fake.trace) == len(fake.effects) == 1
    fake.assert_complete()


def test_qsy_script_readiness_invitation_acceptance_retune_opening():
    fake = FakeSession(scenario='qsy')
    fake.poll(0)
    assert execute(fake, s.ChatQSY(12)).state == 'running'
    phases = []
    for _ in range(4):
        advance(fake, .2)
        phases.append(fake.state.operation.subphase)
    assert phases == ['invitation delivery', 'peer acceptance', 'retune', 'opening']
    advance(fake, .2)
    assert fake.state.identity.slot == 12 and terminal(fake, 'r1')[0].state == 'done'
    fake.assert_complete()


def test_qsy_cancel_only_wait_refusal_and_stale_generation():
    fake = connected()
    execute(fake, s.ChatQSY(3))
    result = execute(fake, s.ChatCancelWait(), id='stop')
    assert result.result.effect == 'wait-only' and fake.state.link.connected
    assert terminal(fake, 'r1')[0].state == 'cancelled'
    assert execute(fake, s.ChatSend('stale'), id='old', generation=0).reason_code == 'link-stale'
    fake.state = patched(fake.state, (('link.connected', False),))
    assert execute(fake, s.ChatQSY(3), id='unlinked').reason_code == 'link-required'


def test_draft_owner_revisions_literal_period_and_local_reads():
    fake = FakeSession(scenario='vmail-email')
    fake.poll(0)
    first = execute(fake, s.VMailCompose('Greetings')).result.draft
    assert first.revision == 0 and first.owner == 'view-a'
    assert execute(fake, s.VMailCompose('Other'), id='other', actor='view-b').reason_code == 'draft-owned'
    updated = execute(fake, s.DraftAppend(first.id, 0, '.'), id='append').result.draft
    assert updated.lines == ('.',) and updated.revision == 1
    assert execute(fake, s.VMailSend(first.id, 0), id='stale').reason_code == 'draft-stale'
    execute(fake, s.VerboseSet(True), id='local')
    fake.legacy_line('/status')
    assert fake.state.application.draft == updated
    fake.legacy_line('/cq')
    assert fake.state.application.draft is None
    assert not any(isinstance(e.payload, s.TextRecord) for e in fake.event_trace)
    advance(fake, .1)
    fake.assert_complete()


def test_multiline_interruption_replacement_and_link_fence_never_send():
    fake = connected()
    draft = execute(fake, s.VMailCompose('body')).result.draft
    next_draft = execute(fake, s.DraftAppend(draft.id, 0, 'retained'), id='append').result.draft
    refused = execute(fake, s.DraftAppend(draft.id, 1, 'x' * 65537), id='too-large')
    assert refused.reason_code == 'draft-too-large' and fake.state.application.draft == next_draft
    replacement = execute(fake, s.VMailCompose('replacement'), id='replace').result.draft
    assert replacement.id != draft.id
    assert execute(fake, s.VMailSend(draft.id, 1), id='old-send').reason_code == 'draft-stale'
    fake.state = patched(fake.state, (('link.generation', 2),))
    fake._sync_generation()
    assert execute(fake, s.VMailSend(replacement.id, 0), id='new-link').reason_code == 'draft-stale'
    assert not any(isinstance(e.payload, s.TextRecord) for e in fake.event_trace)


@pytest.mark.parametrize('kind', ['vmail', 'email'])
def test_mail_consumes_reviewed_revision_and_application_confirmation_is_separate(kind):
    fake = connected()
    compose = s.VMailCompose('subject') if kind == 'vmail' else s.EmailCompose('a@example.org', 'b@example.org', 'subject')
    draft = execute(fake, compose).result.draft
    draft = execute(fake, s.DraftAppend(draft.id, 0, 'body'), id='body').result.draft
    action = s.VMailSend(draft.id, 1) if kind == 'vmail' else s.EmailSend(draft.id, 1)
    result = execute(fake, action, id='send')
    assert result.state == 'running' and fake.state.application.draft is None
    advance(fake, .25)
    assert terminal(fake, 'send') == []
    advance(fake, .25)
    assert terminal(fake, 'send')[0].result == s.MailResult(kind, 'smr' if kind == 'vmail' else 'er')
    assert fake.preflight(action).reason_code == 'reconnect-required'
    assert fake.preflight(action).reservation.state == 'spent'


def test_queued_bbs_cancel_releases_reservation_sent_cancel_requires_reconnect():
    fake = connected()
    fake.state = patched(fake.state, (('identity.channel_hz', 14105000),))
    execute(fake, s.BBSList())
    assert fake.preflight(s.BBSList()).reservation.state == 'pending'
    execute(fake, s.ChatCancelWait(), id='stop')
    assert terminal(fake, 'r1')[0].state == 'cancelled'
    assert fake.preflight(s.BBSList()).reservation.state == 'available'
    fake.state = patched(fake.state, (('identity.channel_hz', 14106500),))
    execute(fake, s.BBSList(), id='sent')
    execute(fake, s.ChatCancelWait(), id='stop2')
    assert terminal(fake, 'sent')[0].state == 'unknown'
    assert fake.preflight(s.BBSList()).reason_code == 'reconnect-required'


def test_bbs_scenario_directory_entry_hostile_page_and_autonomous_detach():
    fake = FakeSession(scenario='bbs')
    fake.poll(0)
    execute(fake, s.BBSList(), id='list')
    advance(fake, .25)
    assert fake.state.application.bbs_waiting
    advance(fake, .75)
    assert terminal(fake, 'list')[0].result.entries[0].name == 'hello.txt'
    execute(fake, s.BBSEnter(), id='enter')
    advance(fake, .1)
    assert any(e.payload == BBS_ATTACK for e in fake.event_trace)
    advance(fake, .9)
    assert not fake.state.application.bbs.remote_view
    fake.assert_complete()


def test_bbs_term_negotiation_and_per_link_reservation():
    fake = connected()
    execute(fake, s.BBSEnter())
    execute(fake, s.BBSTerm('ansi16', 80, 24), id='term')
    assert not fake.state.application.bbs.remote_ansi
    assert fake.preflight(s.BBSTerm('ansi16', 80, 24)).reason_code == 'service-pending'
    advance(fake, .5)
    assert fake.state.application.bbs.remote_ansi
    assert terminal(fake, 'term')[0].result == s.TermResult('ansi16', 80, 24)
    assert fake.preflight(s.BBSTerm('plain', 80, 24)).reason_code == 'reconnect-required'


def test_broadcast_explicit_actions_and_busy_high_unknown_refusals():
    fake = FakeSession(scenario='broadcast')
    fake.poll(0)
    assert fake.effects == [] and fake.project().receive.heard[0].n == 1
    execute(fake, s.BroadcastSelect(12), id='select')
    assert fake.state.broadcast.selected_hz == 14108000
    execute(fake, s.BroadcastSend('hello all'), id='send')
    advance(fake, 1)
    assert terminal(fake, 'send')[0].result == s.BroadcastResult(1, 'sent')
    assert fake.state.broadcast.possibly_outstanding == 1
    assert fake.effects == [('select', s.Effect.X), ('send', s.Effect.X)]
    fake.assert_complete()
    for line in (True, None):
        fake.state = patched(fake.state, (('tx.physical_line', line),))
        assert execute(fake, s.BroadcastSend('not sent'), id=f'line-{line}').reason_code == 'line-not-low'
    fake.state = patched(fake.state, (('tx.physical_line', False), ('receive.host_busy', True)))
    assert execute(fake, s.BroadcastSend('not sent'), id='busy').reason_code == 'channel-busy'


def test_failure_close_deadlines_survive_peer_loss_and_quit_records_order():
    fake = FakeSession(scenario='failure-close')
    fake.poll(0)
    deadline = fake.state.beacon.due
    advance(fake, 1)
    assert fake.state.beacon.due == deadline
    assert execute(fake, s.ChatClose()).state == 'failed'
    advance(fake, 1)
    assert fake.state.tx.physical_line is False
    advance(fake, 1)
    assert fake.state.link.host_closed and fake.state.ownership.health == 'failed'
    fake.quit()
    fake.quit()
    assert fake.state.beacon.minutes is None
    assert fake.shutdown_trace == ['session.quit', 'client.close', 'runtime.close', 'slots.close',
        'reporter.close', 'saves.wait', 'streams.close', 'terminal.close', 'catalog.close', 'rig.shutdown']
    fake.assert_complete()


def test_beacon_pending_positive_config_off_hop_and_one_terminal():
    fake = FakeSession()
    execute(fake, s.BeaconOnce(False), id='once')
    assert execute(fake, s.BeaconOnce(True), id='force').reason_code == 'beacon-pending'
    execute(fake, s.BeaconConfigure(10), id='configure')
    assert fake.state.beacon.once_request_id == 'once'
    execute(fake, s.BeaconConfigure(None), id='off')
    assert [r.state for r in terminal(fake, 'once')] == ['cancelled']
    execute(fake, s.BeaconOnce(True), id='again')
    execute(fake, s.BeaconHop((20, 40), False, 10), id='hop')
    advance(fake, 5)
    assert [r.state for r in terminal(fake, 'again')] == ['cancelled']
    assert fake.preflight(s.BeaconOnce(False)).reservation.state == 'available'


def test_cq_without_incoming_returns_nullable_binding_and_parent_id():
    fake = FakeSession()
    result = execute(fake, s.ChatCQ(12, False))
    assert result.bound_generation is None and result.attempt_id is None
    advance(fake, 1)
    done = terminal(fake, 'r1')[0]
    assert done.result == s.ConnectionResult(None, None, 'no-answer') and done.bound_generation is None


def test_fake_controller_never_opens_network_factories(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network factory invoked')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    fake = FakeSession(scenario='idle-monitoring')
    controller = Controller(fake, clock=fake.clock)
    try:
        controller.ready.result(2)
        fake.advance(2).result(2)
        controller.check(s.VerboseSet(True)).result(2)
        assert controller.projection().live.policy.auto_state == 'suspended'
        assert fake.trace == []
    finally:
        controller.shutdown().result(2)
        assert controller.join(2)


def test_scripted_failure_and_unknown_are_not_normal_exhaustion_success():
    fake = FakeSession(scenario=(
        ScenarioStep(0, 'chat.cq', s.ChatCQ(12, False), (),
                     (Disposition('failed', reason_code='synthetic-start-failure'),)),
        ScenarioStep(1, 'poll', None, (),
                     (ScenarioEvent(s.Lifecycle('failed', 'worker fault', 'unknown')),)),
    ))
    result = execute(fake, s.ChatCQ(12, False))
    assert result.state == 'failed' and result.reason_code == 'synthetic-start-failure'
    advance(fake, 1)
    assert fake.event_trace[-1].payload.unkey == 'unknown' and fake.event_trace[-1].generation is None
    fake.assert_complete()


OUTCOME_CASES = (
    (s.ChatSend('hello'), s.WriteAccepted),
    (s.ChatCall('N0XYZ', 12, False), s.ConnectionResult),
    (s.ChatAnswer(1), s.ConnectionResult),
    (s.ChatCQ(12, False), s.ConnectionResult),
    (s.ChatQSY(12), s.Applied),
    (s.ChatTag('INFO', 'test'), s.WriteAccepted),
    (s.ChatCancelWait(), s.WaitStopped),
    (s.ChatClose(), s.LinkClosed),
    (s.ChatExit(), s.LinkClosed),
    (s.BufferClear(), s.Applied),
    (s.FollowSet('all'), s.Applied),
    (s.BandSet(40), s.Applied),
    (s.BandReady(True), s.Applied),
    (s.BroadcastSelect(12), s.Applied),
    (s.BroadcastSend('everyone'), s.BroadcastResult),
    (s.BroadcastOff(), s.Applied),
    (s.BroadcastClear(), s.Applied),
    (s.BroadcastConfirm(False), s.Applied),
    (s.BeaconConfigure(10), s.Applied),
    (s.BeaconOnce(False), s.BeaconResult),
    (s.BeaconHop((20, 40), False, 10), s.Applied),
    (s.AwaySet('later'), s.Applied),
    (s.AutoSet(False), s.Applied),
    (s.AwayBack(), s.Applied),
    (s.VerboseSet(True), s.Applied),
    (s.PSKReporterOff(), s.Applied),
    (s.InquireHeard(), s.WriteAccepted),
    (s.VMailCompose('subject'), s.DraftAccepted),
    (s.EmailCompose('a@example.org', 'b@example.org', 'subject'), s.DraftAccepted),
    (s.DraftAppend('reviewed-draft', 2, 'line'), s.DraftAccepted),
    (s.DraftDrop('reviewed-draft', 2), s.DraftAccepted),
    (s.VMailSend('reviewed-draft', 2), s.MailResult),
    (s.EmailSend('reviewed-draft', 2), s.MailResult),
    (s.VMailCollect('N0XYZ'), s.ConnectionResult),
    (s.VMailDismiss('N0XYZ'), s.Applied),
    (s.BBSList(), s.DirectoryResult),
    (s.BBSGet('1'), s.FileResult),
    (s.BBSRequest('1'), s.FileResult),
    (s.BBSCapture(), s.FileResult),
    (s.BBSEnter(), s.WriteAccepted),
    (s.BBSTerm('ansi16', 80, 24), s.TermResult),
    (s.BBSInput('current-context', 3, '1'), s.WriteAccepted),
    (s.BBSReload(), s.ReloadResult),
)


@pytest.mark.parametrize('action,result_type', OUTCOME_CASES, ids=lambda value: getattr(value, 'kind', str(value)))
def test_every_operational_outcome_row(action, result_type):
    idle = action.kind in ('chat.call', 'chat.answer', 'chat.cq', 'vmail.collect', 'beacon.once', 'band.set')
    fake = FakeSession() if idle else connected()
    fake.state = patched(fake.state, (
        ('receive.heard', SCENARIOS['idle-monitoring'][0].patch[0][1]),
        ('broadcast.selected_hz', 14106500),
        ('application.files.entries', (s.FileEntry('hello.txt', '2026-10-28', 5),)),
        ('application.files.list_generation', fake.state.link.generation),
        ('application.files.generation', fake.state.link.generation),
        ('application.files.offer', 'hello.txt, 5 bytes'),
    ))
    if action.kind in ('draft.append', 'draft.drop', 'vmail.send', 'email.send'):
        draft = s.Draft('reviewed-draft', 2, 'view-a', 'email' if action.kind == 'email.send' else 'vmail',
            fake.state.link.generation, 'N0XYZ', 'subject', 'a@example.org', 'b@example.org', ('one', 'two'))
        fake.state = patched(fake.state, (('application.draft', draft),))
    if action.kind in ('bbs.term', 'bbs.input'):
        fake.state = patched(fake.state, (('application.bbs.remote_view', True),
                                        ('application.bbs.owned', True)))
    result = execute(fake, action)
    if result.state == 'running':
        assert terminal(fake, 'r1') == []
        advance(fake, 2)
        result, = terminal(fake, 'r1')
    assert result.state == 'done', result
    assert isinstance(result.result, result_type)
    if isinstance(result.result, s.ConnectionResult) and action.kind != 'chat.cq':
        assert result.result.result == 'connected'
        assert result.result.attempt_id is not None and result.result.generation is not None
    if isinstance(result.result, s.FileResult):
        assert result.result.result == {'bbs.get': 'validated-saved', 'bbs.request': 'offer-recorded',
                                       'bbs.capture': 'capture-started'}[action.kind]
    assert fake.effects == [('r1', s.effect(action))]


def test_outcome_fixture_inventory_is_the_complete_operational_union():
    assert {action.kind for action, _ in OUTCOME_CASES} == {
        kind for kind, (_, effect) in s.ACTIONS.items() if effect is not s.Effect.R}


@pytest.mark.parametrize('action,outcome', (
    (s.BBSList(), Disposition('failed', s.DirectoryResult((), 'rejected'), reason_code='directory-rejected')),
    (s.BBSList(), Disposition('unknown', s.DirectoryResult((), 'incomplete'), reason_code='directory-timeout')),
    (s.BBSRequest('1'), Disposition('failed', s.FileResult('failed', 'hello.txt', None, 0), reason_code='file-rejected')),
    (s.BBSTerm('ansi16', 80, 24), Disposition('unknown', reason_code='term-timeout')),
    (s.EmailSend('reviewed-draft', 2), Disposition('failed', s.MailResult('email', 'rejected'), reason_code='gateway-rejected')),
    (s.EmailSend('reviewed-draft', 2), Disposition('unknown', s.MailResult('email', 'unconfirmed'), reason_code='email-timeout')),
    (s.ChatQSY(12), Disposition('refused', reason_code='qsy-refused')),
))
def test_scripted_protocol_refusals_and_timeouts_have_explicit_outcomes(action, outcome):
    fake = connected()
    fake.state = patched(fake.state, (('application.files.list_generation', 1),
                                     ('application.bbs.remote_view', True)))
    fake._steps = [ScenarioStep(0, action.kind, action, (), (outcome,))]
    result = execute(fake, action)
    assert result == outcome
    assert not any(isinstance(e.payload, s.TextRecord) for e in fake.event_trace)
    fake.assert_complete()


def test_partial_credit_clear_and_replacement_preserve_known_bytes():
    fake = connected()
    execute(fake, s.ChatSend('hello'))
    advance(fake, .25)
    execute(fake, s.BufferClear(), id='clear')
    outcomes = [e.payload for e in fake.event_trace if isinstance(e.payload, s.TxTextOutcome)]
    assert outcomes[-1].status == 'discarded'
    assert outcomes[-1].acknowledged_bytes == 4 and outcomes[-1].discarded_bytes == 5
    advance(fake, 1)
    assert len([e.payload for e in fake.event_trace if isinstance(e.payload, s.TxTextOutcome)]) == 3
    execute(fake, s.ChatSend('next'), id='next')
    advance(fake, .25)
    fake.state = patched(fake.state, (('link.generation', 2),))
    fake._sync_generation()
    last = next(e.payload for e in reversed(fake.event_trace) if isinstance(e.payload, s.TxTextOutcome))
    assert last.status == 'unknown' and last.acknowledged_bytes == 4 and last.unknown_bytes == 4
    advance(fake, 1)
    assert next(e.payload for e in reversed(fake.event_trace) if isinstance(e.payload, s.TxTextOutcome)) == last


@pytest.mark.parametrize('line,unkey', ((True, 'high'), (None, 'unknown')))
def test_close_without_low_line_proof_fails(line, unkey):
    fake = connected()
    fake.state = patched(fake.state, (('tx.physical_line', line),))
    result = execute(fake, s.ChatClose())
    assert result.state == 'failed' and result.result.unkey == unkey
    assert fake.state.link.connected


@pytest.mark.parametrize('line,unkey', ((False, 'low'), (True, 'high'), (None, 'unknown')))
def test_shutdown_retires_fake_host_without_fabricating_low_line(line, unkey):
    fake = FakeSession()
    fake.state = patched(fake.state, (*CONNECTED, ('tx.physical_line', line), ('tx.host_ptt', True),
                                    ('link.pending', True)))
    controller = Controller(fake, clock=fake.clock)
    controller.ready.result(2)
    completion = controller.shutdown()
    if line is False:
        completion.result(2)
    else:
        with pytest.raises(RuntimeError):
            completion.result(2)
    assert controller.join(2)
    snapshot = controller.snapshot()
    assert not snapshot.link.connected and not snapshot.link.pending and snapshot.link.host_closed
    assert not snapshot.tx.host_ptt and snapshot.tx.physical_line is line
    lifecycle = [event.payload for event in controller.subscribe(0).drain()
                 if isinstance(event.payload, s.Lifecycle)][-1]
    assert lifecycle.unkey == unkey
    assert lifecycle.state == ('closed' if line is False else 'failed')


def test_owner_loss_and_shutdown_failure_remain_explicit():
    fake = connected()
    fake.state = patched(fake.state, (('ownership.mode', 'station'),
        ('ownership.control_id', 'lease'), ('ownership.control_generation', 2),
        ('ownership.health', 'revoked')))
    result = fake.execute(replace(request(fake, s.ChatSend('no')), control_id='lease', control_generation=2))
    assert result.state == 'refused' and result.reason_code == 'authority-stale'
    assert fake.effects == []
    fake.quit_error = RuntimeError('synthetic worker did not retire')
    with pytest.raises(RuntimeError, match='worker did not retire'):
        fake.quit()
    assert fake.shutdown_trace


def test_gap_and_pressure_script_delivers_bounded_controller_recovery():
    events = tuple(s.TextRecord(f'burst-{i}', 'rx', 'body', 'N0XYZ', 'chat') for i in range(160))
    fake = FakeSession(scenario=(ScenarioStep(1, 'poll', None, CONNECTED, events),))
    controller = Controller(fake, clock=fake.clock, max_records=20)
    try:
        controller.ready.result(2)
        subscription = controller.subscribe(0)
        fake.advance(1).result(2)
        controller.check(s.VerboseSet(True)).result(2)
        delivered = subscription.drain()
        assert len(delivered) <= 128
        assert any(isinstance(e.payload, s.Gap) for e in delivered)
        snapshot = controller.snapshot()
        assert len(snapshot.presentation.records) == 20
        assert snapshot.presentation.evicted_through is not None
        assert fake.trace == fake.effects == []
    finally:
        controller.shutdown().result(2)
        assert controller.join(2)


READ_ACTIONS = (
    s.ChatSnapshot(), s.ChatProjection(), s.ChatCheck(s.ChatSend('preview')), s.ChatOutcome('absent'),
    s.ChatHelp(None), s.HeardList(), s.SlotsList(), s.FilesList(), s.VMailList(), s.VMailRead(1),
    s.VMailPending(), s.BroadcastStatus(), s.BeaconHopStatus(), s.EmailStatus(), s.PSKReporterStatus(),
)


def test_every_read_outcome_row_and_read_bookkeeping_stability():
    assert {action.kind for action in READ_ACTIONS} == {
        kind for kind, (_, effect) in s.ACTIONS.items() if effect is s.Effect.R}
    fake = FakeSession()
    fake.state = patched(fake.state, (('application.mailbox', (
        s.MailHeader(1, 'mail-1', 'in', 'N0XYZ', 'W9SSJ', 'subject', '2026-10-28T00:00:00Z'),)),))
    controller = Controller(fake, clock=fake.clock, controller_id='controller')
    try:
        initial = controller.ready.result(2)
        for index, action in enumerate(READ_ACTIONS):
            req = s.Request(f'read-{index}', controller.controller_id, 0, 'view', None, None, action)
            assert controller.submit(req).queued
            for _ in range(10):
                controller.check(s.VerboseSet(True)).result(2)
                result = controller.outcome(req.id)
                if isinstance(result, s.RequestResult) and result.state.state == 'done':
                    break
            assert result.state.state == 'done', result
            assert isinstance(result.result, s.ReadValue)
        assert controller.snapshot().revision == initial.revision
        assert controller.projection().requests == ()
        assert all(effect is s.Effect.R for _, effect in fake.effects)
    finally:
        controller.shutdown().result(2)
        assert controller.join(2)


@pytest.mark.parametrize('action', (
    s.VerboseSet(True), s.AutoSet(False), s.AwaySet('later'), s.AwayBack(), s.FollowSet('all'),
    s.PSKReporterOff(), s.VMailDismiss('N0XYZ'), s.BBSEnter(),
))
def test_local_settings_and_bbs_interception_preserve_draft(action):
    fake = connected()
    draft = execute(fake, s.VMailCompose('subject')).result.draft
    execute(fake, action, id='preserve')
    assert fake.state.application.draft == draft


def test_typed_nonlocal_command_drops_draft_after_preflight_but_chat_send_refuses():
    fake = connected()
    draft = execute(fake, s.VMailCompose('subject')).result.draft
    refused = execute(fake, s.ChatSend('not a body append'), id='chat')
    assert refused.state == 'refused' and refused.reason_code == 'draft-active'
    assert fake.state.application.draft == draft
    execute(fake, s.ChatQSY(12), id='qsy')
    assert fake.state.application.draft is None
    assert not any(isinstance(e.payload, s.TextRecord) for e in fake.event_trace)


@pytest.mark.parametrize('action', (s.BroadcastSend('clear'), s.BroadcastSend('yes'),
                                    s.BroadcastSend('no')))
def test_broadcast_acknowledgement_refusals_preserve_outstanding_frames(action):
    fake = FakeSession()
    fake.state = patched(fake.state, (('broadcast.possibly_outstanding', 2),))
    before = fake.project()
    assert execute(fake, action).state == 'refused'
    assert fake.project() == before
    assert fake.effects == []


def test_fake_typed_broadcast_acknowledgements_unblock_retuning():
    pytest.importorskip('textual', reason='install hfmodem[tui] to run the Textual chat tests')
    from hfmodem.ui.textual_chat import parse_action
    fake = FakeSession()
    def act(line, id):
        return execute(fake, s.decode(s.encode(parse_action(line))), id=id)
    assert act('/all cf', 'select').state == 'done'
    assert act('/all hello', 'send').state == 'running'
    advance(fake, 1)
    assert fake.state.broadcast.possibly_outstanding == 1
    assert act('/all slot 1', 'pending').state == 'done'
    assert act('/all no', 'no').state == 'done'
    assert fake.state.broadcast.possibly_outstanding == 1
    assert act('/all slot 1', 'retry').state == 'done'
    assert act('/all yes', 'yes').state == 'done'
    assert fake.state.broadcast.possibly_outstanding == 0
    assert fake.state.broadcast.selected_hz != fake.state.identity.calling_hz
    fake.state = patched(fake.state, (('broadcast.possibly_outstanding', 2),))
    assert act('/all clear', 'clear').state == 'done'
    assert fake.state.broadcast.possibly_outstanding == 0
    assert fake.preflight(s.BandSet(40)).availability == 'allowed'
