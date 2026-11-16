# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest

from hfmodem.contact import chat_state as s
from hfmodem.contact import varac as v
from hfmodem.contact.chat_controller import Disposition, TERMINAL
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.tests.contact.test_bbs_download import MESSAGE, NAME, OFFER
from hfmodem.tests.contact.test_chat_controller import adapter, execute
from hfmodem.tests.contact.test_varac import CQ_DONE, Desk


def terminal(port, events, req):
    port._flush()
    result = port._results[req.id]
    assert result.state in TERMINAL
    states = [payload for payload, rid, _ in events
              if rid == req.id and isinstance(payload, Disposition) and payload.state in TERMINAL]
    assert states == [result]
    return result


def pump(port, predicate, *, count=2000):
    for _ in range(count):
        port.poll(.1)
        if predicate():
            return
    pytest.fail('fake transport did not reach the expected boundary')


def file_adapter(tmp_path):
    desk, port, events = adapter(connected=True, mailbox=tmp_path)
    desk.session.files.listed((v.BBSFile(NAME, '2026-05-31', 2516),))
    return desk, port, events


@pytest.mark.parametrize('action', (s.BBSRequest('1'), s.BBSGet('1')))
def test_file_rejection_is_terminal_failed_even_without_failure_flag(tmp_path, action):
    desk, port, events = file_adapter(tmp_path)
    req, initial = execute(port, action)
    assert initial.state == 'running'
    desk.receive('<BGJ>')
    port.poll(0)
    assert desk.session.files.failure is None
    result = terminal(port, events, req)
    assert result.state == 'failed' and result.result.result == 'failed'
    assert port.preflight(action).reservation.state == 'spent'


@pytest.mark.parametrize('action', (s.BBSRequest('1'), s.BBSGet('1')))
def test_file_offer_timeout_is_unknown_and_spends_family(tmp_path, action):
    desk, port, events = file_adapter(tmp_path)
    req, _ = execute(port, action)
    desk.host.now += 61
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == 'unknown'
    assert port.preflight(action).reason_code == 'reconnect-required'
    desk.session.files.end('test complete')


def test_file_validation_failure_survives_tick_clearing_failure(tmp_path):
    desk, port, events = file_adapter(tmp_path)
    req, _ = execute(port, s.BBSGet('1'))
    desk.receive(OFFER.raw)
    assert desk.session.files.receiving
    desk.receive(MESSAGE[:5] + '?' + MESSAGE[6:])
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == 'failed'
    assert not any('<SFOK>' in text for text in desk.texts)
    assert not tuple(tmp_path.glob('downloads/*/*'))


def test_file_receiving_deadline_is_failed_not_validated_download(tmp_path):
    desk, port, events = file_adapter(tmp_path)
    req, _ = execute(port, s.BBSGet('1'))
    desk.receive(OFFER.raw)
    desk.session.files.recorder.deadline = desk.host.now
    pump(port, lambda: req.id in port._results)
    assert terminal(port, events, req).state == 'failed'
    assert not any('<SFOK>' in text for text in desk.texts)


def test_file_evidence_finish_io_error_is_not_swallowed(tmp_path, monkeypatch):
    desk, port, events = file_adapter(tmp_path)
    req, _ = execute(port, s.BBSRequest('1'))

    def fail(reason):
        raise OSError('synthetic evidence disk failure')
    monkeypatch.setattr(desk.session.files.recorder, 'finish', fail)
    desk.receive(OFFER.raw)
    result = terminal(port, events, req)
    assert result.state == 'failed'
    assert any('File evidence write failed' in line for line in desk.said)


def test_file_offer_recorded_boundary_and_capture_start_are_separate(tmp_path):
    desk, port, events = file_adapter(tmp_path)
    req, _ = execute(port, s.BBSRequest('1'))
    desk.receive(OFFER.raw)
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result.result == 'offer-recorded'
    assert not any('<SFRD>' in text for text in desk.texts)
    capture, admitted = execute(port, s.BBSCapture())
    assert admitted.state == 'done' and admitted.result.result == 'capture-started'
    assert desk.session.files.opaque
    assert terminal(port, events, capture) == admitted
    desk.session.files.end('test complete')


def test_validated_file_save_survives_notification_failure(tmp_path, monkeypatch):
    desk, port, events = file_adapter(tmp_path)
    req, _ = execute(port, s.BBSGet('1'))
    desk.receive(OFFER.raw)
    original = desk.host.client.send_text

    def send(text):
        if text == '<SFOK>':
            raise OSError('synthetic notification failure')
        return original(text)
    monkeypatch.setattr(desk.host.client, 'send_text', send)
    desk.receive(MESSAGE)
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result.result == 'validated-saved'
    assert tuple(tmp_path.glob('downloads/*/*'))


def email_adapter():
    desk, port, events = adapter(connected=True)
    desk.receive('<S:E>')
    _, result = execute(port, s.EmailCompose('a@example.org', 'b@example.org', 'subject'))
    draft = result.result.draft
    req, sent = execute(port, s.EmailSend(draft.id, draft.revision))
    assert sent.state == 'running'
    return desk, port, events, req


@pytest.mark.parametrize('reply', ('<EJ>', '<SMF>', '<SMFP>'))
def test_email_observed_rejection_is_failed_not_unknown(reply):
    desk, port, events, req = email_adapter()
    desk.receive(reply)
    port.poll(0)
    result = terminal(port, events, req)
    assert result.state == 'failed' and result.result == s.MailResult('email', 'rejected')
    assert port.preflight(s.EmailSend('old', 0)).reservation.state == 'spent'


def test_email_er_finishes_while_smr_and_buffer_zero_do_not():
    desk, port, events, req = email_adapter()
    desk.hear('BUFFER 0')
    desk.receive('<SMR>')
    port.poll(0)
    assert req.id not in port._results
    assert not any(isinstance(payload, s.ApplicationOutcome) and payload.result == s.MailResult('email', 'modem-ack')
                   for payload, _, _ in events)
    desk.receive('<ER>')
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result == s.MailResult('email', 'er')


def test_email_timeout_is_unknown_and_late_er_cannot_change_terminal():
    desk, port, events, req = email_adapter()
    desk.host.now += 121
    port.poll(0)
    result = terminal(port, events, req)
    assert result.state == 'unknown'
    assert port.preflight(s.EmailSend('old', 0)).reason_code == 'reconnect-required'
    desk.receive('<ER>')
    assert terminal(port, events, req) == result


def test_queued_bbs_list_retains_context_through_sent_timeout():
    desk, port, events = adapter(connected=True)
    desk.flow.dial = v.CALLING_HZ[20]
    req, initial = execute(port, s.BBSList())
    assert initial.stage == 'queued'
    desk.flow.dial = 14106500
    port.poll(0)
    assert req.id in port._sent and '<BLR>' in desk.texts[-1]
    desk.host.now += 61
    port.poll(0)
    result = terminal(port, events, req)
    assert result.state == 'unknown'
    assert port.preflight(s.BBSList()).reservation.state == 'spent'


@pytest.mark.parametrize('reply,state', (('<BLJ>', 'failed'), ('<BL:one.txt|2026-01-01|5>', 'done')))
def test_directory_response_has_semantic_success_or_rejection(reply, state):
    desk, port, events = adapter(connected=True)
    req, _ = execute(port, s.BBSList())
    desk.receive(reply)
    result = terminal(port, events, req)
    assert result.state == state
    assert result.result.result == ('complete' if state == 'done' else 'rejected')


def test_legacy_tag_blr_contaminates_typed_directory_attribution():
    desk, port, _ = adapter(connected=True)
    port.legacy_line('/tag BLR')
    desk.receive('<BL:legacy.txt|2026-01-01|5>')
    cap = port.preflight(s.BBSList())
    assert cap.availability == 'blocked' and cap.reason_code == 'reconnect-required'
    assert cap.reservation.state == 'spent' and cap.reservation.request_id is None
    before = tuple(desk.texts)
    _, refused = execute(port, s.BBSList())
    assert refused.state == 'refused' and tuple(desk.texts) == before


def test_legacy_offer_alias_contaminates_typed_file_attribution(tmp_path):
    desk, port, _ = file_adapter(tmp_path)
    port.legacy_line('/bbs offer 1')
    desk.receive('<BGJ>')
    cap = port.preflight(s.BBSGet('1'))
    assert cap.reservation.state == 'spent' and cap.reservation.request_id is None
    desk.session.files.end('test complete')


def test_refused_ids_do_not_remain_active_for_future_semantic_callbacks():
    _, port, _ = adapter()
    req, refused = execute(port, s.ChatSend('unlinked'))
    assert refused.state == 'refused'
    assert port._active_operation('chat.send') is None
    assert port._results[req.id] == refused


def test_cancel_wait_does_not_finish_before_delayed_physical_unkey():
    desk, port, events = adapter(connected=True)
    desk.line = True
    desk.host.client.ptt = True
    req, initial = execute(port, s.ChatCancelWait())
    assert initial.state == 'running'
    port.poll(.1)
    assert req.id not in port._results
    desk.line = False
    desk.host.lines('PTT OFF')
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result.effect == 'link-closed'
    assert not desk.host.client.connected


def test_repeat_cq_keeps_parent_after_failed_incoming_child():
    desk, port, events = adapter()
    command = desk.flow.codec.cq_command('W9SSJ', 2, 500)
    desk.host.replies[command] = CQ_DONE
    desk.flow.opening_steps = lambda: iter(())
    req, _ = execute(port, s.ChatCQ(2, True))
    pump(port, lambda: desk.flow.state is v.State.WAITING)
    desk.host.lines('PENDING', 'DISCONNECTED')
    port.poll(.1)
    assert req.id not in port._results
    assert any(a['ended'] and a['request_id'] == req.id for a in port._attempts.values())
    pump(port, lambda: desk.host.client.listening and desk.flow.state in (v.State.WAITING, v.State.MONITORING))
    desk.host.lines('PENDING', 'CONNECTED W9SSJ W1AW 500')
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result.result == 'connected'
    assert result.result.attempt_id and result.result.generation is not None


def test_interrupted_frame_waits_for_late_completion_then_unkey_without_fake_incoming():
    desk, port, _ = adapter()
    generator = desk.host.client.cq_steps('CQFRAME W9SSJ-9 500')
    next(generator)
    generator.close()
    assert port._frame is not None
    for line in ('PENDING', 'PTT ON', 'CANCELPENDING'):
        desk.host.client._line(line)
    assert port._frame is not None and not port._attempts
    desk.host.client._line('PTT OFF')
    assert port._frame is None and not port._attempts
    desk.host.client._line('PENDING')
    assert len(port._attempts) == 1
    assert next(iter(port._attempts.values()))['request_id'] is None


def test_buffer_clear_missing_reply_after_command_submission_is_unknown():
    desk, port, events = adapter(connected=True, **{'CLEANTXBUFFER': []})
    req, result = execute(port, s.BufferClear())
    assert 'CLEANTXBUFFER' in desk.sent
    assert result.state == 'unknown'
    assert terminal(port, events, req) == result


@pytest.mark.parametrize('submitted', (False, True))
def test_broadcast_cancel_distinguishes_unwritten_from_possible_tx(submitted):
    desk, port, events = adapter(kiss=True)
    desk.host.kiss.respond = lambda _: None
    port.poll(0)
    _, selected = execute(port, s.BroadcastSelect(11))
    assert selected.state == 'done'
    req, initial = execute(port, s.BroadcastSend('hello everyone'))
    assert initial.state == 'running'
    if submitted:
        pump(port, lambda: bool(desk.host.kiss.sent))
    _, stopped = execute(port, s.ChatCancelWait())
    assert stopped.state in ('running', 'done')
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == ('unknown' if submitted else 'cancelled')
    pump(port, lambda: desk.session.steps is None)
    assert not desk.session.broadcast.active
    assert desk.session.broadcast.possibly_outstanding == int(submitted)


def test_broadcast_matching_bracket_completes_without_reception_claim():
    desk, port, events = adapter(kiss=True)
    desk.host.kiss.respond = lambda _: None
    port.poll(0)
    execute(port, s.BroadcastSelect(11))
    req, initial = execute(port, s.BroadcastSend('hello'))
    assert initial.state == 'running'
    pump(port, lambda: bool(desk.host.kiss.sent))
    from hfmodem.tests.contact.test_broadcast_client import observed
    observed(desk)
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result == s.BroadcastResult(1, 'sent')
    assert len(desk.host.kiss.sent) == 1


def test_broadcast_write_failure_is_unknown_after_possible_submission():
    desk, port, events = adapter(kiss=True)
    def fail(_):
        raise OSError('partial KISS write')
    desk.host.kiss.sendall = fail
    port.poll(0)
    execute(port, s.BroadcastSelect(11))
    req, initial = execute(port, s.BroadcastSend('hello'))
    assert initial.state == 'running'
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == 'unknown'
    assert desk.session.broadcast.possibly_outstanding == 1


@pytest.mark.parametrize('text', ('clear', 'yes', 'no'))
def test_broadcast_send_cannot_bypass_typed_acknowledgements(text):
    desk, port, _ = adapter(kiss=True)
    desk.session.broadcast.possibly_outstanding = 2
    dial = desk.flow.dial
    _, result = execute(port, s.BroadcastSend(text))
    assert result.state == 'refused'
    assert 'typed broadcast acknowledgement' in result.detail
    assert desk.session.broadcast.possibly_outstanding == 2
    assert desk.flow.dial == dial and desk.host.kiss.sent == []


@pytest.mark.parametrize('action', (s.BroadcastSelect(11), s.BroadcastOff(),
                                    s.ChatCall('W1AW', None, False), s.AutoSet(True),
                                    s.BandSet(40), s.FollowSet(None)))
def test_typed_broadcast_retunes_require_outstanding_acknowledgement(action):
    desk, port, _ = adapter(kiss=True)
    desk.session.broadcast.possibly_outstanding = 2
    dial = desk.flow.dial
    _, result = execute(port, action)
    confirming = isinstance(action, (s.BroadcastSelect, s.BroadcastOff))
    assert result.state == ('done' if confirming else 'refused')
    assert (desk.session.broadcast._retune_command is not None) is confirming
    assert desk.session.broadcast.possibly_outstanding == 2
    assert desk.flow.dial == dial and desk.host.kiss.sent == []


@pytest.mark.parametrize('action', (
    s.VerboseSet(True), s.AutoSet(False), s.AwaySet('Back later'), s.AwayBack(),
    s.PSKReporterOff(), s.BeaconConfigure(10), s.BeaconConfigure(None), s.FollowSet(None),
))
def test_routine_settings_finish_with_applied_result_without_delivery_claim(action):
    _, port, events = adapter()
    req, result = execute(port, action)
    assert result.state == 'done' and isinstance(result.result, s.Applied)
    assert terminal(port, events, req) == result
    assert not any(isinstance(payload, s.TxTextOutcome) for payload, _, _ in events)


def test_tag_lhr_uses_inquiry_guards_and_blr_uses_directory_contract():
    desk, port, events = adapter(connected=True)
    desk.flow.dial = v.CALLING_HZ[20]
    before = tuple(desk.texts)
    _, refused = execute(port, s.ChatTag('LHR', None))
    assert refused.state == 'refused' and tuple(desk.texts) == before
    desk.flow.dial = 14106500
    req, result = execute(port, s.ChatTag('BLR', None))
    assert result.state == 'running'
    desk.receive('<BL:one.txt|2026-01-01|5>')
    assert terminal(port, events, req).result.result == 'complete'


def test_mailbox_snapshot_initializes_existing_header_without_render_scan(tmp_path):
    desk = Desk(mailbox=tmp_path)
    record = desk.mailbox.store('in', 'N0XYZ', 'W9SSJ', v.VMail('saved subject', 'saved body'),
                                dial_hz=desk.flow.dial, snr=None)
    port = SessionAdapter(desk.session)
    headers = port.project().application.mailbox
    assert any(header.id == record['id'] and header.subject == 'saved subject' for header in headers)


def test_credit_and_link_transition_projection_reflect_observations():
    desk, port, _ = adapter(connected=True)
    before = port.project()
    desk.session.diagnostic('acknowledged', bytes=7)
    desk.session.diagnostic('discarded', bytes=3)
    snapshot = port.project()
    assert sum(credit.acknowledged_bytes for credit in snapshot.tx.credit) == 7
    assert sum(credit.discarded_bytes for credit in snapshot.tx.credit) == 3
    assert all(credit.attribution == 'unattributed' for credit in snapshot.tx.credit)
    desk.host.now += 1
    desk.session._closing = True
    snapshot = port.project()
    assert snapshot.link.last_transition_at > before.link.last_transition_at
    assert snapshot.link.generation == before.link.generation


@pytest.mark.parametrize('force', (False, True))
def test_beacon_wrong_without_pending_proves_not_sent_and_releases_reservation(force):
    command = 'CQFRAMEFORCE W9SSJ-9 500' if force else 'CQFRAME W9SSJ-9 500'
    desk, port, events = adapter(**{command: ['WRONG']})
    req, initial = execute(port, s.BeaconOnce(force))
    assert initial.state == 'running'
    pump(port, lambda: req.id in port._results)
    result = terminal(port, events, req)
    assert result.state == 'failed' and result.result == s.BeaconResult('not-sent')
    assert port._frame is None
    assert port.preflight(s.BeaconOnce(False)).reservation.state == 'available'
    assert not any(event.raw in ('PENDING', 'PTT ON') for event in desk.host.events)


@pytest.mark.parametrize('line', ('<unescaped>', 'x' * 65537))
def test_invalid_vmail_append_keeps_reviewed_draft_and_backend_body(line):
    desk, port, events = adapter(connected=True)
    _, composed = execute(port, s.VMailCompose('subject'))
    draft = composed.result.draft
    req, result = execute(port, s.DraftAppend(draft.id, 0, line))
    assert result.state == 'refused'
    assert port.project().application.draft == draft
    assert desk.session.compose == ('subject', [])
    assert not desk.texts
    assert terminal(port, events, req) == result


@pytest.mark.parametrize('guard', ('closing', 'files-active', 'bbs-owned', 'broadcast-active'))
def test_vmail_send_guard_retains_draft_before_any_effect(guard):
    desk, port, events = adapter(connected=True)
    _, composed = execute(port, s.VMailCompose('subject'))
    draft = composed.result.draft
    if guard == 'closing':
        desk.session._closing = True
    elif guard == 'files-active':
        desk.session.files.waiting = True
    elif guard == 'bbs-owned':
        desk.session.bbs.remote_view = True
    else:
        desk.session.broadcast.active = True
    req, result = execute(port, s.VMailSend(draft.id, 0))
    assert result.state == 'refused'
    assert port.project().application.draft == draft
    assert desk.session.compose == ('subject', [])
    assert not desk.texts
    assert terminal(port, events, req) == result


def test_vmail_release_size_limit_is_a_pure_typed_refusal():
    desk, port, events = adapter(connected=True)
    _, composed = execute(port, s.VMailCompose('subject'))
    draft = composed.result.draft
    action = s.DraftAppend(draft.id, 0, 'body' * 200)
    assert port.preflight(action).availability == 'blocked'
    req, result = execute(port, action)
    assert result.state == 'refused'
    assert '500-character admission limit' in result.detail
    assert port.project().application.draft == draft
    assert desk.session.compose == ('subject', [])
    assert not desk.texts
    assert terminal(port, events, req) == result


def test_all_read_actions_use_real_session_sources_without_mutation(tmp_path):
    from hfmodem.contact.chat_controller import Controller
    from hfmodem.tests.contact.test_chat_fake import READ_ACTIONS
    mailbox = v.Mailbox(tmp_path)
    mailbox.store('in', 'N0XYZ', 'W9SSJ', v.VMail('saved', 'body'), dial_hz=14105000, snr=None)
    owner = {}

    def factory():
        desk = Desk(mailbox=tmp_path)
        owner['desk'] = desk
        owner['writes'] = tuple(desk.sent), tuple(desk.texts), tuple(desk.tunes), tuple(desk.armed)
        return SessionAdapter(desk.session)

    controller = Controller(factory, clock=lambda: owner['desk'].host.now)
    try:
        initial = controller.ready.result(2)
        for index, action in enumerate(READ_ACTIONS):
            req = s.Request(f'read-{index}', controller.controller_id, 0, 'observer', None, None, action)
            assert controller.submit(req).queued
            for _ in range(100):
                controller.check(s.VerboseSet(False)).result(2)
                result = controller.outcome(req.id)
                if isinstance(result, s.RequestResult) and result.state.state in TERMINAL:
                    break
            assert result.state.state == 'done'
            assert isinstance(result.result, s.ReadValue)
            if isinstance(action, s.VMailRead):
                assert result.result.body == 'body' and result.result.header.subject == 'saved'
        assert controller.snapshot().requests == ()
        assert controller.snapshot().revision >= initial.revision
        desk = owner['desk']
        assert (tuple(desk.sent), tuple(desk.texts), tuple(desk.tunes), tuple(desk.armed)) == owner['writes']
    finally:
        controller.shutdown().result(2)
        assert controller.join(2)


@pytest.mark.parametrize('failure', (None, 'future', 'apply'))
def test_real_bbs_reload_applies_or_refuses_on_owner_poll(failure):
    from concurrent.futures import Future
    from types import SimpleNamespace
    desk, port, events = adapter()
    future = Future()
    applied = []

    def reload_policy(config):
        applied.append(config)
        if failure == 'apply':
            raise ValueError('invalid replacement policy')
        return 17
    desk.session.bbs.core = SimpleNamespace(tick=lambda: None, reload_policy=reload_policy)
    desk.session.bbs.reload = lambda: future
    req, result = execute(port, s.BBSReload())
    assert result.state == 'running' and applied == []
    assert port.project().application.bbs.reload_pending
    if failure == 'future':
        future.set_exception(OSError('reload worker could not read policy'))
    else:
        future.set_result('replacement-policy')
    port.poll(0)
    result = terminal(port, events, req)
    assert result.state == ('done' if failure is None else 'failed')
    if failure is None:
        assert result.result == s.ReloadResult(17)
    assert applied == ([] if failure == 'future' else ['replacement-policy'])
    assert not port.project().application.bbs.reload_pending


def test_real_term_timeout_is_unknown_and_late_reply_cannot_settle_reissue():
    desk, port, events = adapter(connected=True)
    desk.host.client.enable_writer = lambda: None
    execute(port, s.BBSEnter())
    req, result = execute(port, s.BBSTerm('ansi16', 80, 24))
    assert result.state == 'running'
    desk.host.now += 61
    port.poll(0)
    result = terminal(port, events, req)
    assert result.state == 'unknown'
    assert port.preflight(s.BBSTerm('ansi16', 80, 24)).reservation.state == 'spent'
    _, refused = execute(port, s.BBSTerm('ansi16', 80, 24))
    assert refused.state == 'refused'
    desk.receive('BBS TERM OK ANSI16 80 24')
    assert terminal(port, events, req) == result


def test_real_hop_configuration_cancels_one_shot_exactly_once():
    desk, port, events = adapter()
    req, _ = execute(port, s.BeaconOnce(False))
    _, configured = execute(port, s.BeaconHop((20, 40), False, 10))
    assert configured.state == 'done' and desk.session.hopping.active
    result = terminal(port, events, req)
    assert result.state == 'cancelled'
    assert port._beacon_id is None
    assert port.preflight(s.BeaconOnce(False)).reservation.state == 'available'
    for _ in range(5):
        port.poll(.1)
    assert terminal(port, events, req) == result


def test_real_expiry_failure_before_restore_fails_one_shot_once(monkeypatch):
    desk, port, events = adapter()
    execute(port, s.BeaconConfigure(10))
    req, _ = execute(port, s.BeaconOnce(False))
    desk.session.hopping.configure(['20', '40', 'every', '10'])
    desk.session._beacon_expires = desk.host.now - 1

    def fail_monitor():
        raise v.HostError('monitor failed during expiry')
    monkeypatch.setattr(desk.session, 'monitor', fail_monitor)
    with pytest.raises(v.HostError, match='during expiry'):
        desk.session._beacon_tick()
    result = terminal(port, events, req)
    assert result.state == 'failed' and result.reason_code == 'expiry-failed'
    assert desk.session._beacon_now is False and port._beacon_id is None
    assert port.preflight(s.BeaconOnce(False)).reservation.state == 'available'


@pytest.mark.parametrize('kind', ('bbs.list', 'bbs.queued', 'bbs.request', 'vmail.send', 'email.send',
                                  'bbs.term', 'broadcast.send', 'chat.call'))
def test_real_host_eof_settles_pending_service_without_replay(tmp_path, kind):
    from hfmodem.contact.chat_controller import Controller
    owner = {}

    def factory():
        d = Desk(mailbox=tmp_path, kiss=True)
        d.host.kiss.respond = lambda _: None
        p = SessionAdapter(d.session)
        owner['desk'] = d
        if kind not in ('broadcast.send', 'chat.call'):
            d.flow.dial = 14106500
            d.hear('CONNECTED W9SSJ N0XYZ 500')
        if kind == 'bbs.queued':
            d.flow.dial = v.CALLING_HZ[20]
        if kind == 'bbs.request':
            d.session.files.listed((v.BBSFile(NAME, '2026-05-31', 2516),))
        if kind == 'email.send':
            d.receive('<S:E>')
        if kind == 'bbs.term':
            d.host.client.enable_writer = lambda: None
            d.session.bbs.command('/bbs enter')
        if kind == 'broadcast.send':
            d.session.broadcast.command('slot 11')
        return p

    controller = Controller(factory, clock=lambda: owner['desk'].host.now)
    controller.ready.result(2)
    serial = 0

    def submit(action, *, terminal_state=False):
        nonlocal serial
        serial += 1
        req = s.Request(f'eof-{serial}', controller.controller_id, controller.snapshot().link.generation,
                        'view', None, None, action)
        assert controller.submit(req).queued
        for _ in range(100):
            controller.check(s.VerboseSet(False)).result(2)
            result = controller.outcome(req.id)
            if isinstance(result, s.RequestResult) and (
                    result.state.state in TERMINAL if terminal_state else result.state.state == 'running'):
                return req, result
        pytest.fail('request never reached expected admission state')

    try:
        if kind in ('vmail.send', 'email.send'):
            compose = (s.VMailCompose('subject') if kind == 'vmail.send' else
                       s.EmailCompose('a@example.org', 'b@example.org', 'subject'))
            _, accepted = submit(compose, terminal_state=True)
            draft = accepted.result.draft
            action = s.VMailSend(draft.id, 0) if kind == 'vmail.send' else s.EmailSend(draft.id, 0)
        else:
            action = {'bbs.list': s.BBSList(), 'bbs.queued': s.BBSList(), 'bbs.request': s.BBSRequest('1'),
                'bbs.term': s.BBSTerm('ansi16', 80, 24), 'broadcast.send': s.BroadcastSend('hello'),
                'chat.call': s.ChatCall('W1AW', None, False)}[kind]
        req, _ = submit(action)
        if kind == 'broadcast.send':
            for _ in range(100):
                controller.check(s.VerboseSet(False)).result(2)
                if controller.outcome(req.id).state.stage == 'sent':
                    break
            assert controller.outcome(req.id).state.stage == 'sent'
        # The journal enters running before backend execution reaches admission.
        controller.check(s.VerboseSet(False)).result(2)
        owner['desk'].host.command.incoming.append(b'')
        assert controller.join(2)
        with pytest.raises(v.HostError, match='(?i)host.*closed'):
            controller.shutdown().result(2)
        result = controller.outcome(req.id)
        assert result.state.state == ('cancelled' if kind == 'bbs.queued' else 'unknown')
        assert result.state.reason_code == 'shutdown'
        terminal_before = result.state
        writes_before = tuple(owner['desk'].sent), tuple(owner['desk'].texts)
        assert not controller.submit(req).reason == 'controller-stale'
        assert controller.outcome(req.id).state == terminal_before
        assert (tuple(owner['desk'].sent), tuple(owner['desk'].texts)) == writes_before
    finally:
        if controller._thread.is_alive():
            completion = controller.shutdown()
            try:
                completion.result(2)
            except v.HostError:
                pass
            assert controller.join(2)


def test_snapshot_every_session_field_uses_its_authoritative_source(monkeypatch):
    from concurrent.futures import Future
    from dataclasses import fields
    from types import SimpleNamespace
    from hfmodem.contact.slots import Said

    desk, port, _ = adapter(connected=True)
    se, c, f = desk.session, desk.host.client, desk.flow
    c.link_generation = 7
    _, composed = execute(port, s.VMailCompose('source subject'))
    draft = composed.result.draft
    _, appended = execute(port, s.DraftAppend(draft.id, 0, 'source body'))
    draft = appended.result.draft
    desk.host.now = 50.
    f.band, f.dial = 40, 7106500
    se.slot = -2
    se.idcall = 'W9SSJ-9'
    c.bandwidth = 2300
    se.band_setup = {'mode': 'USB', 'filter': 500}
    se.contact = {'n': 41, 'how': 'answer', 'started_utc': '2026-10-28T12:00:00Z'}
    c.pending = True
    se.intent, se.steps = 'answer', iter(())
    f.qsy_slot, f.qsy_receiving = -2, True
    se._close_failed = RuntimeError('close detail')
    monkeypatch.setattr(se, '_operator_phase', lambda: (('qsy', 'source phase'), 'source description', 'source next'))
    c.ptt, c.buffer, c.buffer_sequence, c.sequence, c._sent_after = True, 23, 7, 31, 9
    c.data_writer = SimpleNamespace(pending=True)
    c._stream_poisoned = True
    desk.line = True
    port._observe_line()
    se.diagnostic('acknowledged', bytes=7)
    se.diagnostic('discarded', bytes=3)
    se.auto, se.away, se._away_owed, se.new_vmails = True, 'away text', 7, 4
    se.following, se._follow_tuned, se.verbose = 'all', False, True
    se.reporter = SimpleNamespace(status=lambda: {'enabled': True, 'queued': 3})
    h = se.hopping
    h.bands, h.auto, h.minutes, h.due, h.pending = (20, 40), True, 17, 201., 40
    h.paused = 'operator readiness'
    h.readiness, h.last_tx = {20: 'unknown', 40: 'ready'}, {20: 10., 40: 20.}
    se.beacon, se._beacon_due, se._beacon_expires, se._beacon_last = 13, 101., 10000., 22.
    se._beacon_now, se._beacon_generation, port._beacon_id = 'force', 9, 'once-request'
    c.kiss_socket = object()
    b = se.broadcast
    b.selected, b.active, b.phase = 7105750, True, 'admission'
    b.cancelled, b.possibly_outstanding, b.result = True, 17, 'queued'
    se.mail_ack.generation, se.mail_ack.state, se.mail_ack.kind = 7, 'active', 'direct'
    se._bbs_list_queued, se._bbs_waiting = 7, True
    se.email.waiting, se.email.phase, se.email.sent_generation = True, 'email phase', 6
    b = se.bbs
    b.sid, b.remote_view, b.generation = 'local-bbs', True, 7
    b.remote_ansi, b.remote_dimensions = True, (100, 30)
    b.requested_term, b.reload_future = ('plain', 80, 24), Future()
    se.bbs_host.phase, se.bbs_host.name = 'sending', 'outgoing.txt'
    file = se.files
    file.waiting = file.downloading = file.receiving = file.opaque = True
    file.generation, file.list_generation, file.requested = 7, 6, 'requested.txt'
    file.entries = (v.BBSFile('entry.txt', '2026-10-27', 17),)
    file.offer, file.received_bytes = SimpleNamespace(wire_name='offer.txt'), 11
    file.offer_deadline, file.failure = 70., 'file detail'
    monkeypatch.setattr(se.mailbox, 'entries', lambda: [dict(id='stored-id', direction='in',
        subject='stored subject', utc='mail UTC', **{'from': 'K1ABC', 'to': 'W9SSJ'})])
    monkeypatch.setattr(se.mail_collection, '_load', lambda: [dict(source='K2ABC', destination='W9SSJ',
        utc='notice UTC', dial_hz=7106500, received=2)])
    port._mail_loaded = False
    se.heard = [v.Heard(9, 'K1ABC', 12, 500, -4., 30., True, 'AA00aa', 'VB', 'wide', 40)]
    row = SimpleNamespace(slot=12, audio_hz=1800, state='active', heard=32., calls=('K1ABC', 'K2ABC'),
        level=4, tags=('B', 'V'), home=True, caller='K1ABC', called='K2ABC', stage='connected',
        since=22., at=77., sn=-3., one_side=True)
    se.slots = SimpleNamespace(_generation=3, table=lambda: ([row], 77.))
    c.busy, c.snr, c.bitrate = True, -2., '1400 bps'
    port.project()
    se.emit('passband', edges={'voice': (300., 2700.)}, slots=[12], wide_offset=125.)
    se.semantic('slot_observed', said=Said(12, 77., 'coverage', 'source gap', duration_s=12.))
    se.semantic('monitor_failed', error=RuntimeError('monitor detail'))
    port._ownership = s.Ownership('station', 'operator', 'lease', 4, 'lease UTC', 'ready', True)
    port._reservations = {'directory': s.ServiceReservation('directory', 'pending', 'list-id'),
                          'mail': s.ServiceReservation('mail', 'spent', 'mail-id')}
    snapshot = port.project()
    expected = {
        'schema': 1, 'controller_id': '', 'revision': 0, 'through_seq': 0, 'sampled_at': 0., 'utc': '',
        'identity': s.Identity('W9SSJ', 'W9SSJ-9', 40, 7106500, 7105000, 12, 2300, False,
                               (('mode', 'USB'), ('filter', '500'))),
        'link': s.Link(7, 'established', s.State.CONNECTED, 'N0XYZ', 'host-connected', 41, 'answer',
                        '2026-10-28T12:00:00Z', True, True, False, 50.),
        'operation': s.Operation(s.Phase.QSY, 'source phase', 'source description', 'source next',
                                  'answer', True, -2, True, None, False, False, 'close detail'),
        'tx': s.Tx(True, True, 50., 23, 7, 31, 9, True, True, True,
            'transmitting a modem frame; 23 bytes queued or awaiting modem acknowledgement', True,
            (s.Credit(None, 7, 3, 'unattributed'),)),
        'policy': s.Policy(True, False, 'suspended', 'broadcast', 'away text', 7, 4, 'all', False, True,
                            (('enabled', 'True'), ('queued', '3'))),
        'beacon': s.Beacon(13, 101., 10000., 22., 'force', 'once-request', True, 9, 'broadcast-selection',
            s.Hop(True, True, (20, 40), 17, 201., 40, 'operator readiness',
                  ((20, 'unknown'), (40, 'ready')), ((20, 10.), (40, 20.)))),
        'broadcast': s.Broadcast(7105750, True, 'admission', True, 17, 'queued'),
        'application': s.Application(draft, True, 7, True, s.EmailState(True, 'email phase', 6),
            s.BBSState(True, True, True, 'local-bbs', 7, True, (100, 30), ('plain', 80, 24), True, 'outgoing.txt'),
            s.FileState(True, True, True, True, 7, 6, 'requested.txt',
                        (s.FileEntry('entry.txt', '2026-10-27', 17),), 'offer.txt', 11, 70., 'file detail'),
            (s.MailHeader(1, 'stored-id', 'in', 'K1ABC', 'W9SSJ', 'stored subject', 'mail UTC'),),
            (s.MailNotice('K2ABC', 'W9SSJ', 'notice UTC', 7106500, 2),),
            s.MailAckState(7, 'active', 'direct', None, None),
            s.ParkingState(True, None, True, True, 30, 100, 1000, True, 1, None, None, (), None, ())),
        'receive': s.Receive(True, -2., '1400 bps',
            (s.HeardRow(9, 'K1ABC', 12, 500, -4., 30., True, 'AA00aa', 'VB', 'wide', 40, 7106500),),
            (s.SlotRow(12, 1800, 'active', 32., ('K1ABC', 'K2ABC'), 4, ('B', 'V'), True,
                       'K1ABC', 'K2ABC', 'connected', 22., 77., -3., True),),
            (12,), (('voice', (300., 2700.)),), 125., 77., 50.,
            (s.Coverage(12, 'source gap', 50., 77., 12., .2, 'observed-events'),), 'monitor detail'),
        'ownership': s.Ownership('station', 'operator', 'lease', 4, 'lease UTC', 'ready', True),
        'requests': (), 'presentation': s.PresentationState((), None, None, 0, True, False),
        'omitted_sections': (),
    }
    families = {'bbs.list': 'directory', 'bbs.get': 'file', 'bbs.request': 'file',
                'vmail.send': 'mail', 'email.send': 'mail', 'bbs.term': 'term', 'beacon.once': 'beacon'}
    capabilities = []
    for kind, (_, effect) in s.ACTIONS.items():
        family = families.get(kind)
        reservation, code, reason = None, None, None
        availability = 'check-on-execution' if effect is s.Effect.X else 'allowed'
        if family:
            reservation = s.ServiceReservation(family, 'available', None)
        if family == 'directory':
            reservation = s.ServiceReservation('directory', 'pending', 'list-id')
            availability, code, reason = 'blocked', 'service-pending', 'Wait for the pending operation'
        elif family == 'mail':
            reservation = s.ServiceReservation('mail', 'spent', 'mail-id')
            availability, code, reason = 'blocked', 'reconnect-required', 'Reconnect before another request'
        capabilities.append(s.Capability(kind, effect, availability, reservation, code, reason))
    expected['capabilities'] = tuple(capabilities)
    assert {field.name for field in fields(snapshot)} == set(expected)
    for name, value in expected.items():
        assert getattr(snapshot, name) == value, name


def test_refused_writer_enqueue_is_failed_without_submission():
    from hfmodem.contact.data_writer import WriterUnavailable
    desk, port, events = adapter(connected=True)
    class FullWriter:
        def submit(self, payload, token): raise WriterUnavailable('queue full')
    desk.host.client.data_writer = FullWriter()
    req, result = execute(port, s.ChatSend('not enqueued'))
    assert result.state == 'failed'
    assert req.id not in port._sent
    assert not desk.host.data.sent
    assert not port._delivery_writes
