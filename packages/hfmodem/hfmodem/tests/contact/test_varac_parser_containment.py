# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Malformed peer application bytes must not retire the modem host or beacon timer."""
import pytest
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk


def linked(**responses):
    d = Desk(**responses)
    d.session.set_beacon(15)
    d.session._beacon_due = d.host.now+900
    d.hear('CONNECTED VE6DNL W9SSJ 500')
    d.flow.dial = 14_105_750
    return d


# CRC-clean level2 DATA, session150046533013; decoded-data.json, samples8150000:8427520.
VE6DNL_PAYLOAD = bytes.fromhex('2164c23c598e5b43873984aad410204081020f140000')


def malformed(d, blob=VE6DNL_PAYLOAD):
    d.host.data.incoming.append(blob)
    d.tick()


def test_malformed_peer_contact_closes_but_host_and_beacon_schedule_survive():
    d = linked()
    due, expires = d.session._beacon_due, d.session._beacon_expires
    malformed(d)
    d.drive()
    assert not d.host.client.connected and not d.host.client.closed
    assert not d.host.command.closed and not d.host.data.closed
    assert d.flow.state is v.State.MONITORING and d.flow.dial == 14_105_000
    assert d.session.beacon == 15 and d.session._beacon_due == due and d.session._beacon_expires == expires
    assert 'DISCONNECT' in d.sent and 'LISTEN CQ' in d.sent
    evidence = d.kinds('peer_protocol_error')
    assert evidence[0]['hex'] == VE6DNL_PAYLOAD.hex()
    assert evidence[0]['bytes'] == 22 and not evidence[0]['truncated']
    assert not any(e['text'] == 'not framed peer bytes!' for e in d.kinds('rx'))
    # Timer is still operational, not merely retained in memory.
    d.host.replies['CQFRAME W9SSJ 500']=['OK','PTT ON','PTT OFF','CANCELPENDING']
    d.host.now = due+6
    d.tick()
    d.host.now+=6
    d.tick()
    d.drive()
    assert any(c.startswith('CQFRAME W9SSJ') for c in d.sent)


def test_ordinary_peer_disconnect_also_keeps_beacon_timer_and_host():
    d = linked()
    due, expires = d.session._beacon_due, d.session._beacon_expires
    d.hear('DISCONNECTED')
    assert d.flow.state is v.State.MONITORING
    assert not d.host.client.closed and d.session.beacon == 15
    assert (d.session._beacon_due, d.session._beacon_expires) == (due,expires)


def test_malformed_input_blocks_text_until_next_contact():
    d = linked()
    malformed(d)
    d.drive()
    d.host.data.incoming.append(v.frame_message('<AWQ>'))
    d.tick()
    assert not d.host.client.messages
    d.hear('PENDING','CONNECTED OTHER W9SSJ 500')
    d.receive('new contact')
    assert any(e['text'] == 'new contact' for e in d.kinds('rx'))


def test_inflight_ptt_keeps_monitoring_disabled_until_unkey():
    d = linked(DISCONNECT=['OK','DISCONNECTED'])
    d.hear('PTT ON')
    malformed(d)
    assert d.session.steps is not None and d.session._closing
    assert d.flow.state is not v.State.MONITORING
    assert not d.host.client.closed
    d.hear('PTT OFF')
    d.drive()
    assert d.flow.state is v.State.MONITORING


def test_physical_high_after_close_is_fatal_instead_of_resuming_beacons():
    d = linked()
    d.line = True
    with pytest.raises(v.HostError,match='keying line remains HIGH'):
        malformed(d)
        d.drive()
    assert d.flow.state is not v.State.MONITORING
    assert d.session._close_failed is not None


def test_failure_evidence_is_bounded_and_does_not_guess_resynchronization():
    d = linked()
    malformed(d,b'x'*10000+v.frame_message('<AWQ>'))
    d.drive()
    item = d.kinds('peer_protocol_error')[0]
    assert item['bytes'] == 10007 and len(item['hex']) == 8192 and item['truncated']
    assert not d.host.client.messages


def test_real_host_eof_remains_fatal():
    d = linked()
    d.host.data.incoming.append(b'')
    with pytest.raises(v.HostError,match='closed its command or data port'):
        d.tick()
    assert d.host.client.closed


def test_bbs_wait_cancelled_by_malformed_contact_without_request_retry():
    d = linked()
    d.session.command('/bbs list')
    assert d.session._bbs_waiting
    malformed(d)
    d.drive()
    assert not d.session._bbs_waiting and d.session.steps is None
    assert sum('<BLR>' in text for text in d.texts) == 1
    assert d.flow.state is v.State.MONITORING and not d.host.client.closed


def test_email_wait_is_cancelled_without_resend_or_false_confirmation_on_bad_data():
    d = linked()
    due, expires = d.session._beacon_due, d.session._beacon_expires
    d.receive('<S:VE>')
    d.session.command('/email send recipient@example.org reply@example.net Test')
    d.session.command('A test body')
    d.session.command('.')
    assert d.session.email.waiting
    malformed(d)
    d.drive()
    assert not d.session.email.waiting and not d.host.client.closed
    assert d.flow.state is v.State.MONITORING
    assert (d.session._beacon_due, d.session._beacon_expires) == (due, expires)
    assert sum('<EG>' in text for text in d.texts) == 1
    assert any(event['event'] == 'wait_cancelled' for event in d.kinds('email'))
    assert not any(event['event'] == 'gateway_reported_sent' for event in d.kinds('email'))


def test_host_eof_quit_records_local_loss_not_peer_disconnect():
    d = linked()
    d.host.data.incoming.append(b'')
    with pytest.raises(v.HostError):
        d.tick()
    with pytest.raises(v.HostError):
        d.session.quit()
    assert d.kinds('link')[-1]['event'] == 'host_lost'
    assert any('peer disconnect and final unkey are unconfirmed' in line for line in d.said)
    assert not any('VE6DNL disconnected' in line for line in d.said)
    assert 'observed before runtime shutdown' in d.kinds('contact')[-1]['line']


def test_late_bad_bytes_after_ordinary_disconnect_do_not_disable_beacons():
    d = linked()
    d.hear('DISCONNECTED')
    assert d.flow.state is v.State.MONITORING
    malformed(d)
    d.drive()
    assert d.flow.state is v.State.MONITORING and not d.session._closing
    assert not d.host.client.closed and d.session.beacon == 15
    d.host.replies['CQFRAME W9SSJ 500']=['OK','PTT ON','PTT OFF','CANCELPENDING']
    d.host.now = d.session._beacon_due+6
    d.tick()
    d.host.now+=6
    d.tick()
    d.drive()
    assert any(c.startswith('CQFRAME W9SSJ') for c in d.sent)


def test_old_fault_cannot_close_new_link_started_in_same_poll():
    d = linked()
    c = d.host.client
    c._text(VE6DNL_PAYLOAD)
    failed_generation = c.link_generation
    c._line('DISCONNECTED')
    c._line('PENDING')
    c._line('CONNECTED OTHER W9SSJ 500')
    c._text(v.frame_message('new link text'))
    before = len(d.sent)
    d.tick()
    assert c.link_generation != failed_generation and c.connected
    assert 'DISCONNECT' not in d.sent[before:]
    assert any(e['text'] == 'new link text' for e in d.kinds('rx'))
    assert d.kinds('peer_protocol_error')[0]['generation'] == failed_generation


def test_ordinary_disconnect_waits_for_host_and_physical_unkey_before_return():
    d = linked()
    due, expires = d.session._beacon_due, d.session._beacon_expires
    d.line = True
    d.hear('PTT ON', 'DISCONNECTED')
    assert d.session.contact is None and d.session.steps is None
    assert d.flow.state is v.State.CONNECTED and d.flow.dial == 14_105_750
    assert d.session.beacon == 15
    assert d.session._operator_phase()[0] == 'settling'
    d.session.command('/status')
    assert any('waiting for the keying line to settle' in line for line in d.said)
    d.hear('PTT OFF')
    assert d.flow.state is v.State.CONNECTED, 'physical HIGH still defers return'
    assert d.session._operator_phase()[0] == 'settling'
    d.line = False
    d.tick()
    assert d.flow.state is v.State.MONITORING and d.flow.dial == 14_105_000
    assert (d.session._beacon_due, d.session._beacon_expires) == (due, expires)
    d.host.replies['CQFRAME W9SSJ 500'] = ['OK', 'PTT ON', 'PTT OFF', 'CANCELPENDING']
    d.host.now = due + 6
    d.tick()
    d.host.now += 6
    d.tick()
    d.drive()
    assert any(c.startswith('CQFRAME W9SSJ') for c in d.sent)
