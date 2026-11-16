# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.kestrel.broadcast import Broadcast
from hfmodem.tests.contact.test_chat_controller import adapter, execute
from hfmodem.tests.contact.test_chat_outcomes import email_adapter, terminal


def vmail_adapter(*, buffer=0):
    desk, port, events = adapter(connected=True)
    desk.host.data.respond = lambda payload: desk.host.lines(f'BUFFER {buffer}')
    _, result = execute(port, s.VMailCompose('subject'))
    draft = result.result.draft
    req, sent = execute(port, s.VMailSend(draft.id, draft.revision))
    assert sent.state == 'running'
    return desk, port, events, req


@pytest.mark.parametrize('tag,state', [('SMR', 'done'), ('SMF', 'failed'), ('SMFP', 'failed')])
def test_vmail_outcomes_follow_durable_ack_owner(tag, state):
    desk, port, events, req = vmail_adapter()
    desk.receive(f'<{tag}>')
    result = terminal(port, events, req)
    assert result.state == state
    assert result.result == s.MailResult('vmail', tag.lower())
    assert desk.session.parking.store.records[0]['state'] == ('delivered' if tag == 'SMR' else 'queued')
    assert not port.project().application.vmail_ack_pending


@pytest.mark.parametrize('email', [False, True])
@pytest.mark.parametrize('reason', ['link changed', 'application acknowledgement timed out',
                                    'QSY interrupted the mail exchange', 'operator stopped exchange'])
def test_ack_interrupt_settles_unknown_immediately_and_keeps_reservation(email, reason):
    desk, port, events, req = email_adapter() if email else vmail_adapter()
    desk.session.mail_ack.interrupt(reason)
    result = terminal(port, events, req)
    assert result.state == 'unknown'
    assert result.reason_code == 'mail-unresolved' and result.detail == reason
    assert desk.session.mail_ack.busy
    app = port.project().application
    assert app.mail_ack.state == 'unresolved'
    assert app.mail_ack.kind == ('email' if email else 'direct')


def test_email_rejection_arrives_from_confirmation_without_generator_poll():
    desk, port, events, req = email_adapter()
    desk.receive('<SMR><ER><SMFP>')
    result = terminal(port, events, req)
    assert result.state == 'failed' and result.result == s.MailResult('email', 'rejected')


def test_vmail_wait_uses_progress_and_drain_deadlines_not_controller_120_seconds():
    desk, port, events, req = vmail_adapter(buffer=30)
    c = desk.host.client
    port.poll(0)
    c.buffer = 30
    desk.host.now += 121
    port.poll(0)
    assert req.id not in port._results
    c.buffer = 20
    desk.host.now += 121
    port.poll(0)
    assert req.id not in port._results
    c.buffer = 0
    c._writes.clear()
    c._sent_after = None
    port.poll(0)
    desk.host.now += 89
    port.poll(0)
    assert req.id not in port._results
    desk.host.now += 2
    port.poll(0)
    result = terminal(port, events, req)
    assert result.state == 'unknown' and result.detail == 'application acknowledgement timed out'
    assert desk.session.mail_ack.busy


@pytest.mark.parametrize('block', ['cf', 'owner', 'incomplete', 'content'])
def test_send_preflight_refuses_release_guards_without_side_effects(block):
    desk, port, _ = adapter(connected=True)
    _, result = execute(port, s.VMailCompose('subject'))
    draft = result.result.draft
    if block == 'cf':
        desk.flow.dial = desk.session.calling_hz
    elif block == 'owner':
        desk.session.mail_ack.reserve('parking', 17, lambda *args: None)
    elif block == 'incomplete':
        desk.host.client._messages.pending.extend(b'partial')
    else:
        port._draft = replace(draft, lines=('x' * 501,))
    before = (desk.sent, desk.texts, list(desk.armed), list(desk.tunes),
              desk.session.compose, desk.session.parking.store.records)
    action = s.VMailSend(draft.id, draft.revision)
    cap = port.preflight(action)
    assert cap.availability == 'blocked'
    _, refused = execute(port, action)
    assert refused.state == 'refused' and refused.detail == cap.reason
    assert (desk.sent, desk.texts, desk.armed, desk.tunes,
            desk.session.compose, desk.session.parking.store.records) == before
    assert port.project().application.draft is not None


@pytest.mark.parametrize('action', [s.InquireHeard(),
                                    s.EmailCompose('a@example.org', 'b@example.org', 'subject'),
                                    s.BBSList()])
def test_mail_owner_blocks_new_application_preflight(action):
    desk, port, _ = adapter(connected=True)
    desk.receive('<S:E>')
    desk.session.mail_ack.reserve('parking', 17, lambda *args: None)
    before = (desk.sent, desk.texts, desk.session.compose)
    assert port.preflight(action).availability == 'blocked'
    _, refused = execute(port, action)
    assert refused.state == 'refused'
    assert (desk.sent, desk.texts, desk.session.compose) == before


def test_parked_delivery_and_urgent_notice_are_projected_from_authoritative_state():
    desk, port, _ = adapter()
    store = desk.session.parking.store
    row, _ = store.accept(owner='W9SSJ', sender='K1ABC', recipient='N0XYZ',
                          via='W9SSJ', subject='parked', body='body')
    desk.flow.dial = 14106500
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    app = port.project().application
    assert app.vmail_ack_pending and app.mail_ack.kind == 'parking'
    assert app.parking.active and app.parking.started_generation == desk.host.client.link_generation
    assert app.parking.mail[0].id == row['id'] and app.parking.mail[0].state == 'in_flight'
    desk.receive('<SMR>')
    app = port.project().application
    assert app.parking.mail[0].state == 'delivered'
    desk.session.mail_collection.notice(Broadcast(1, 'K1ABC', 'W9SSJ', b'<SMU>'), dial=14105000)
    app = port.project().application
    assert app.notices[0].urgent
    assert s.decode(s.encode(s.project(port.project()))).application == app


def test_line_power_passes_through_and_preserves_draft_with_release_bytes():
    desk, port, _ = adapter(connected=True)
    from hfmodem.tests.contact.test_varac_power import Radio
    radio = desk.session.power = Radio(100)
    desk.session.power_clamp_w = 25
    _, result = execute(port, s.VMailCompose('subject'))
    draft = result.result.draft
    port.legacy_line('/power 80')
    assert radio.writes == [25]
    assert desk.said[-2:] == [
        'warning: /power asks 80 W, above the 25 W station clamp ([rig] power_clamp_w); setting 25 W',
        'power set to 25 W (clamp 25 W)']
    assert port.project().application.draft == draft


def test_email_smr_keeps_ack_owner_until_er_or_interrupt():
    desk, port, events, req = email_adapter()
    desk.receive('<SMR>')
    assert req.id not in port._results
    assert desk.session.mail_ack.state == 'email-reserved'
    desk.session.mail_ack.interrupt('QSY interrupted the mail exchange')
    result = terminal(port, events, req)
    assert result.state == 'unknown' and result.detail == 'QSY interrupted the mail exchange'
    assert port.project().application.mail_ack.state == 'unresolved'
    desk.receive('<ER>')
    assert terminal(port, events, req) == result


@pytest.mark.parametrize('reply', ['<ER><SMR>', '<SMR><ER>', '<ER>'])
@pytest.mark.parametrize('finish', ['complete', 'interrupt'])
def test_terminal_email_confirmation_survives_smr_and_wait_cleanup(reply, finish):
    from hfmodem.contact.varac import decode_application
    desk, port, events, req = email_adapter()
    c, owner = desk.host.client, desk.session.mail_ack
    owner.message(decode_application(reply), c.message_sequence + 1, c.link_generation)
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result == s.MailResult('email', 'er')
    assert owner.result == 'ER'
    owner.message(decode_application('<SMR>'), c.message_sequence + 2, c.link_generation)
    if finish == 'complete':
        port.poll(0)
        assert desk.session.steps is None
    else:
        desk.session.mail_ack.interrupt('operator stopped exchange')
        desk.session.steps.close()
    owner.message(decode_application('<SMR>'), c.message_sequence + 3, c.link_generation)
    desk.session.mail_ack.interrupt('later interruption')
    ack = port.project().application.mail_ack
    assert ack.state == 'email-reserved' and ack.result == 'ER' and ack.reason is None
    assert terminal(port, events, req) == result


@pytest.mark.parametrize('query', ['preflight', 'projection'])
def test_parking_queries_observe_committed_state_across_expiration_without_writes(monkeypatch, query):
    desk, port, _ = adapter(connected=True)
    parking = desk.session.parking
    store = parking.store
    now = [store.clock()]
    store.clock = lambda: now[0]
    row, _ = store.accept(owner='W9SSJ', sender='W9SSJ', recipient=desk.session.peer,
                          via='W9SSJ', subject='expires', body='body', origin='direct')
    parking.started = desk.host.client.link_generation
    writes = []
    commit = store._commit
    def track(data):
        writes.append(data)
        commit(data)
    monkeypatch.setattr(store, '_commit', track)
    before = store.path.read_bytes()
    for at in (row['expires_unix'] - 1, row['expires_unix'] + 1):
        now[0] = at
        if query == 'preflight':
            assert port.preflight(s.BBSList()).availability == 'blocked'
        else:
            projected = port.project().application.parking
            assert projected.active and projected.mail[0].state == 'queued'
        assert writes == [] and store.path.read_bytes() == before
    parking.tick()
    assert len(writes) == 1 and store.records[0]['state'] == 'expired'
    projected = port.project().application.parking
    assert not projected.active and projected.mail[0].state == 'expired'
    assert port.preflight(s.BBSList()).availability != 'blocked'


def test_vmail_stalled_writer_waits_for_owner_180_second_timer():
    desk, port, events, req = vmail_adapter(buffer=30)
    port.poll(0)
    desk.host.now += 179
    port.poll(0)
    assert req.id not in port._results
    desk.host.now += 2
    port.poll(0)
    result = terminal(port, events, req)
    assert result.state == 'unknown' and result.detail == 'application acknowledgement timed out'
    desk.receive('<SMR>')
    assert desk.session.parking.store.records[0]['state'] == 'delivered'
    assert terminal(port, events, req) == result


def test_failed_durable_ack_commit_is_unknown_not_delivered(monkeypatch):
    desk, port, events, req = vmail_adapter()
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(desk.session.parking.store, 'resolve', fail)
    desk.receive('<SMR>')
    result = terminal(port, events, req)
    assert result.state == 'unknown' and result.detail == 'disk full'
    assert port.project().application.vmail_ack_pending
    assert desk.session.parking.store.records[0]['state'] == 'in_flight'


def test_smr_with_reply_envelope_finishes_direct_request_before_envelope_dispatch():
    desk, port, events, req = vmail_adapter()
    desk.receive('<SMR>\r\n<SM><SBJ:reply><MSG:body>')
    result = terminal(port, events, req)
    assert result.state == 'done' and result.result == s.MailResult('vmail', 'smr')
    assert desk.mailbox.read(2)['body'] == 'body'


@pytest.mark.parametrize('email', [False, True])
def test_incomplete_frame_blocks_compose_without_mutation(email):
    desk, port, _ = adapter(connected=True)
    desk.receive('<S:E>')
    desk.host.client._messages.pending.extend(b'partial')
    action = (s.EmailCompose('a@example.org', 'b@example.org', 'subject') if email else
              s.VMailCompose('subject'))
    before = desk.sent, desk.texts, desk.session.compose, desk.session.email.draft
    cap = port.preflight(action)
    assert cap.availability == ('blocked' if email else 'allowed')
    assert (desk.sent, desk.texts, desk.session.compose, desk.session.email.draft) == before


def test_email_invalid_send_and_spent_generation_are_pure_refusals():
    desk, port, _ = adapter(connected=True)
    desk.receive('<S:E>')
    action = s.EmailCompose('a@example.org', 'b@example.org', 'subject')
    _, result = execute(port, action)
    draft = result.result.draft
    port._draft = replace(draft, lines=('x' * 501,))
    assert port.preflight(s.EmailSend(draft.id, 0)).availability == 'blocked'
    desk.session.email.sent_generation = desk.host.client.link_generation
    before = desk.session.email.draft
    assert port.preflight(action).availability == 'blocked'
    _, refused = execute(port, action)
    assert refused.state == 'refused'
    assert desk.session.email.draft is before


def test_received_mail_refreshes_collection_notice_count_without_a_read_action():
    desk, port, _ = adapter(connected=True)
    desk.session.mail_collection.notice(Broadcast(1, 'N0XYZ', 'W9SSJ', b'<SMU>'), dial=14105000)
    assert port.project().application.notices[0].received == 0
    desk.receive('<SM><SBJ:reply><MSG:body>')
    notice = port.project().application.notices[0]
    assert notice.received == 1 and notice.urgent


@pytest.mark.parametrize('email', [False, True])
@pytest.mark.parametrize('interrupt', ['qsy', 'stop'])
def test_backend_qsy_and_typed_stop_interrupt_pending_mail(email, interrupt):
    desk, port, events, req = email_adapter() if email else vmail_adapter()
    if interrupt == 'qsy':
        desk.receive('<QSYF>000014105750</QSYF>')
        reason = 'QSY interrupted the mail exchange'
    else:
        execute(port, s.ChatCancelWait())
        reason = 'operator stopped exchange'
    result = terminal(port, events, req)
    assert result.state == 'unknown' and result.detail == reason
    assert desk.session.mail_ack.busy
