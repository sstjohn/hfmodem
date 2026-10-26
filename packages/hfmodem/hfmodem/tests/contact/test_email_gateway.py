# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest
from hfmodem.contact import varac as v
from hfmodem.contact.email_gateway import Email
from hfmodem.tests.contact.test_varac import Desk


def gateway(service='<S:VE>'):
    d = Desk()
    d.hear('CONNECTED W9SSJ N0XYZ 500')
    d.flow.dial = 14_105_750
    if service:
        d.receive(service)
    d.host.data.sent.clear()
    return d


def compose(d):
    d.session.command('/email send recipient@example.org reply@example.net Test subject')
    d.session.command('Hello from W9SSJ')


def test_public_p113_envelope_order_and_reply_address():
    assert Email('a@example.org', 'b@example.net', 'Test', 'Hello').wire() == (
        '<SM><TO:a@example.org><SBJ:Test><MSG:Hello><EG><RE:b@example.net>')


@pytest.mark.parametrize('changes', [dict(recipient='x@y.org>'), dict(reply_to='a\r\n@example.org'),
    dict(subject='inject<EG>'), dict(body='x<ER>'), dict(body='\x00'), dict(body='x'*501),
    dict(recipient='a@example.org,b@example.org'), dict(reply_to='Person <a@example.org>')])
def test_invalid_envelope_rejected(changes):
    fields=dict(recipient='a@example.org', reply_to='b@example.org', subject='test', body='body')
    fields.update(changes)
    with pytest.raises(ValueError):
        Email(**fields).wire()


@pytest.mark.parametrize('service', ['<E>', '<EA>', '<S:VE>', 'hello <S:VEB>'])
def test_console_to_wire_acceptance_then_gateway_confirmation(service):
    d = gateway(service)
    def reply(data):
        d.host.lines('BUFFER 0')
        d.host.data.incoming.append(v.frame_message('<SMR>'))
    d.host.data.respond = reply
    compose(d)
    d.session.command('.')
    d.tick()
    wire=Email('recipient@example.org','reply@example.net','Test subject','Hello from W9SSJ').wire()
    assert d.host.data.sent[-1] == v.frame_message(wire[:-1]) + b'>'
    assert d.session.email.waiting
    assert not any(e.get('event')=='gateway_reported_sent' for e in d.kinds('email'))
    d.receive('notification <ER>')
    assert not d.session.email.waiting
    assert d.kinds('email')[-1]['event'] == 'gateway_reported_sent'
    assert any('recipient inbox delivery not verified' in s for s in d.said)


@pytest.mark.parametrize('service', [None, '<S:VAB>', '<EM:a@example.org>'])
def test_requires_current_connected_gateway_service(service):
    d=gateway(service)
    d.session.command('/email send a@example.org b@example.org Test')
    assert d.session.email.draft is None and not d.texts


def test_service_and_draft_cannot_cross_link_generation():
    d=gateway()
    compose(d)
    d.host.client._retire_link_messages()
    d.receive('<E>')
    d.session.command('.')
    assert not d.texts and 'link changed' in d.said[-1]


def test_old_service_does_not_authorize_new_link():
    d=gateway()
    d.hear('DISCONNECTED')
    d.hear('PENDING','CONNECTED W9SSJ OTHER 500')
    d.flow.dial=14_105_750
    d.session.command('/email send a@example.org b@example.org Test')
    assert not any('<EG>' in t for t in d.texts) and d.session.email.draft is None


def test_cf_and_active_exchange_rejected():
    d=gateway()
    d.flow.dial=14_105_000
    d.session.command('/email send a@example.org b@example.org Test')
    assert 'off the calling frequency' in d.said[-1] and not d.texts
    d.flow.dial=14_105_750
    d.session.steps=iter([.1])
    d.session.command('/email send a@example.org b@example.org Test')
    assert 'current exchange' in d.said[-1] and not d.texts


@pytest.mark.parametrize('response', ['<EJ>','<SMF>','<SMFP>','<ER><EJ>'])
def test_rejection_never_claims_sent(response):
    d=gateway()
    d.host.data.respond=lambda data: d.host.data.incoming.append(v.frame_message(response))
    compose(d)
    d.session.command('.')
    d.tick()
    assert any('rejected' in e['text'] for e in d.kinds('error'))
    assert not any(e.get('event')=='gateway_reported_sent' for e in d.kinds('email'))


def test_modem_ack_and_smr_without_er_timeout_unknown():
    d=gateway()
    compose(d)
    d.session.command('.')
    d.receive('<SMR>')
    d.host.now+=121
    d.tick()
    assert not d.session.email.waiting
    assert any('outcome unknown' in e['text'] for e in d.kinds('error'))
    assert not any(e.get('event')=='gateway_reported_sent' for e in d.kinds('email'))


@pytest.mark.parametrize('action', ['/stop','/bye'])
def test_cancel_wait_never_retries_email(action):
    d=gateway()
    compose(d)
    d.session.command('.')
    d.session.command(action)
    assert not d.session.email.waiting
    assert sum('<EG>' in text for text in d.texts)==1
    assert any('may still be sent' in s for s in d.said)


def test_disconnect_and_reconnect_cannot_supply_late_confirmation():
    d=gateway()
    compose(d)
    d.session.command('.')
    d.host.client._line('DISCONNECTED')
    d.host.client._line('PENDING')
    d.host.client._line('CONNECTED W9SSJ OTHER 500')
    d.host.client._text(v.frame_message('<E><ER>'))
    d.session._step()
    assert not any(e.get('event')=='gateway_reported_sent' for e in d.kinds('email'))


def test_stale_er_does_not_confirm_new_send_and_reply_retrieval_is_explicit():
    d=gateway()
    d.receive('<ER>')
    compose(d)
    d.session.command('.')
    d.tick()
    assert d.session.email.waiting
    d.session.command('/email get')
    assert 'not implemented' in d.said[-1]
    d.session.command('/stop')


def test_draft_bound_and_cancel_send_nothing():
    d=gateway()
    compose(d)
    d.session.command('x'*501)
    assert '500-byte' in d.said[-1]
    d.session.command('/stop')
    assert d.session.email.draft is None and not d.texts


def test_cancelled_send_cannot_misattribute_late_er_to_second_send():
    d=gateway()
    compose(d)
    d.session.command('.')
    d.session.command('/stop')
    d.receive('<ER>')
    d.session.command('/email send other@example.org reply@example.org Second')
    assert d.session.email.draft is None
    assert 'one email per link' in d.said[-1]
    assert sum('<EG>' in t for t in d.texts)==1


def test_normal_status_preserves_draft_and_explains_gateway_wait():
    d=gateway()
    compose(d)
    for command in ('/status','/help email','/email status'):
        d.session.command(command)
        assert d.session.email.draft is not None
    assert d.session._operator_phase()[0]=='email-compose'
    d.session.command('.')
    d.session.command('/status')
    assert d.session._operator_phase()[0][0]=='email'
    assert any('gateway outcome unknown' in line for line in d.said)
    d.session.command('/stop')


def test_tags_in_unrelated_vmail_body_do_not_confirm_email():
    d=gateway()
    compose(d)
    d.session.command('.')
    d.receive('<SM><SBJ:unrelated><MSG:literal <ER>>')
    assert d.session.email.waiting
    assert not any(e.get('event')=='gateway_reported_sent' for e in d.kinds('email'))
    d.session.command('/stop')


def test_disconnect_during_draft_never_queues_payload():
    d=gateway()
    compose(d)
    d.hear('DISCONNECTED')
    d.session.command('.')
    assert not any('<EG>' in text for text in d.texts)
    assert d.session.email.draft is not None
    assert any('established link' in e['text'] for e in d.kinds('error'))


def test_email_frame_roundtrips_existing_vmail_message_framer():
    email=Email('a@example.org','b@example.org','Test','body')
    text=email.wire()
    framed=v.frame_message(text[:-1])+b'>'
    parser=v.MessageParser()
    assert parser.feed(framed[:-1])==[]
    assert parser.feed(framed[-1:])==[text.encode()]
