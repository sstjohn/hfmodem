# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
from pathlib import Path

import pytest

from hfmodem.ax25 import kiss
from hfmodem.contact import varac as v
from hfmodem.contact.broadcast import Broadcast, encode_ui
from hfmodem.tests.contact.test_varac import Desk, answered, vmail_arrives


CAPTURE = json.loads((Path(__file__).parent / 'fixtures' / 'forwarded-vmail-1024.json').read_text())
WIRE = bytes.fromhex(CAPTURE['wire_hex'])
MAIL = '<SM><TME:2026-10-25 12:00:00><FRM:K1ABC><TO:W9SSJ><SBJ:Hello><MSG:café\r\n73>'


def notice(d, source='KQ4QWH', destination='W9SSJ', payload=b'<SMP>', dial=14105000, command=1):
    d.session.broadcast.received(Broadcast(command, source, destination, payload), dial=dial)


def test_notice_from_kiss_is_addressed_persistent_and_never_transmits(tmp_path):
    d = Desk(mailbox=tmp_path, kiss=True)
    before = (list(d.sent), list(d.texts), list(d.tunes))
    d.host.kiss.incoming.append(kiss.encode(encode_ui('KQ4QWH', 'W9SSJ', b'<SMP>'), 1))
    d.tick()
    assert (d.sent, d.texts, d.tunes) == before
    assert len(d.kinds('vmail_notice')) == 1
    assert any('/vmail collect KQ4QWH' in line for line in d.said)
    notice(d)  # repeated notice updates the one row
    restarted = Desk(mailbox=tmp_path)
    restarted.do('/vmail pending')
    assert sum('KQ4QWH: mail notice' in line for line in restarted.said) == 1
    assert not restarted.mailbox.entries()
    restarted.do('/vmail dismiss KQ4QWH')
    restarted.do('/vmail pending')
    assert restarted.said[-1] == 'No parked-mail notices saved for this callsign'


@pytest.mark.parametrize('args', [
    dict(destination='ALL'), dict(destination='N0XYZ'), dict(destination='W9SSJ-1'),
    dict(source='W9SSJ'), dict(payload=b'<SMP>hello'), dict(payload=b'\xff'), dict(command=0),
])
def test_other_broadcasts_do_not_become_mail_or_trigger_collection(args):
    d = Desk()
    before = list(d.sent)
    notice(d, **args)
    assert not d.kinds('vmail_notice') and not d.mailbox.entries()
    assert d.sent == before and not d.texts


def test_notification_to_qsy_to_captured_forwarded_mail_and_read(tmp_path):
    d = Desk(mailbox=tmp_path)
    notice(d)
    def connect():
        d.host.data.incoming.append(v.frame_message('<AWQ>'))
        return ['OK', 'CONNECTED W9SSJ KQ4QWH 500', 'SN -3.0']
    d.host.replies['CONNECT W9SSJ KQ4QWH'] = connect
    def reply(data):
        d.host.lines('BUFFER 0')
        if data == v.frame_message('<TL>'):
            # Actual captured aggregate: QSY acceptance and forwarded mail.
            d.host.data.incoming.append(WIRE[:180])
            d.host.data.incoming.append(WIRE[180:])
    d.host.data.respond = reply
    d.do('/vmail collect kq4qwh')
    assert not d.kinds('error')
    assert d.flow.dial == v.slot_frequency(20, 1)
    assert d.session.steps is None and d.host.client.connected
    mail = d.mailbox.read(1)
    assert mail['from'] == 'KN6ZUA' and mail['to'] == 'W9SSJ' and mail['via'] == 'KQ4QWH'
    assert mail['composed_utc'] == '2026-01-04 23:02:35'
    assert mail['subject'] == 'Just want to say HOWDY!!'
    assert mail['body'].endswith('73\r\nJim\r\n')
    assert d.texts.count('5 <SMR>') == 1
    d.do('/vmail 1')
    assert 'relayed by KQ4QWH' in d.said
    assert 'composed: 2026-01-04 23:02:35 UTC (sender-declared)' in d.said
    d.do('/vmail pending')
    assert any('1 received since notice' in line for line in d.said)
    assert len(d.session.mail_collection._load()) == 1
    d.do('/vmail dismiss KQ4QWH')
    assert len(d.mailbox.entries()) == 1


def test_captured_counted_packet_reassembles_at_every_split():
    expected = WIRE.partition(b' ')[2]
    assert int(WIRE.partition(b' ')[0]) == len(expected)
    for split in range(len(WIRE) + 1):
        parser = v.MessageParser()
        got = parser.feed(WIRE[:split]) + parser.feed(WIRE[split:] + v.frame_message('next'))
        assert got == [expected, b'next']
        assert not parser.pending
        assert v.VMail.parse(got[0].decode()).sender == 'KN6ZUA'


def test_forwarded_excluded_terminator_utf8_and_multiple_mail_frames():
    d = answered()
    wire = v.frame_message(MAIL[:-1]) + b'>'
    d.host.data.incoming.append(wire[:-1])
    d.tick()
    assert not d.mailbox.entries() and '5 <SMR>' not in d.texts
    d.host.data.incoming.append(wire[-1:] + v.frame_message('<SM><MSG:second>'))
    d.tick()
    assert len(d.mailbox.entries()) == 2 and d.texts.count('5 <SMR>') == 2
    assert d.mailbox.read(1)['body'] == 'café\r\n73'
    assert d.mailbox.read(2)['from'] == 'N0XYZ'


def test_retry_after_reconnect_is_acknowledged_without_duplicate_saved_mail(tmp_path):
    d = answered(mailbox=tmp_path)
    vmail_arrives(d, MAIL)
    restarted = answered(mailbox=tmp_path)
    vmail_arrives(restarted, MAIL)
    assert len(restarted.mailbox.entries()) == 1
    assert restarted.texts.count('5 <SMR>') == 1
    assert any('already saved' in line for line in restarted.said)
    vmail_arrives(restarted, MAIL.replace('12:00:00', '12:01:00'))
    assert len(restarted.mailbox.entries()) == 2


@pytest.mark.parametrize('text', [
    MAIL.replace('<FRM:K1ABC>', '<FRM:K1ABC><FRM:K2ABC>'),
    MAIL.replace('2026-10-25', '2026-99-99'),
    MAIL.replace('<FRM:K1ABC>', '<FRM:>'),
    MAIL + '<EG>', MAIL + '<U>', MAIL.replace('<TME:', '<UNKNOWN:'),
])
def test_wrong_recipient_and_unsupported_envelopes_never_acknowledge(text):
    d = answered()
    d.receive(text)
    assert not d.mailbox.entries() and '5 <SMR>' not in d.texts
    assert d.kinds('rx')[-1]['text'] == text


def test_save_failure_does_not_acknowledge_or_lose_live_contact(monkeypatch):
    d = answered()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(d.mailbox, 'store', fail)
    d.receive(MAIL)
    assert '5 <SMR>' not in d.texts and d.host.client.connected
    assert any('custody refused (SMF): disk full' in line for line in d.said)


def test_failed_call_and_cancel_preserve_notice():
    d = Desk()
    notice(d)
    d.do('/vmail collect KQ4QWH')
    assert len(d.session.mail_collection._load()) == 1 and not d.mailbox.entries()
    assert any('timed out' in line for line in d.said)
    d.host.replies['CONNECT W9SSJ KQ4QWH'] = ['OK', 'CONNECTED W9SSJ KQ4QWH 500']
    d.session.command('/vmail collect KQ4QWH')
    d.drive(until=lambda: d.host.client.connected)
    d.tick()
    d.do('/stop')
    assert len(d.session.mail_collection._load()) == 1
    assert not any('<QSYF>' in text for text in d.texts)


@pytest.mark.parametrize('guard', ['linked', 'pending', 'ptt', 'physical', 'band-fault', 'steps', 'wrong-band', 'closed'])
def test_collection_guards_preserve_notice_and_do_not_tune_or_send(guard):
    d = Desk()
    notice(d, dial=7105000 if guard == 'wrong-band' else 14105000)
    if guard == 'linked':
        d.hear('CONNECTED N0XYZ W9SSJ 500')
    elif guard == 'pending':
        d.host.client.pending = True
    elif guard == 'ptt':
        d.host.client.ptt = True
    elif guard == 'physical':
        d.line = True
    elif guard == 'band-fault':
        d.session._band_fault = True
    elif guard == 'steps':
        d.session.steps = iter([None])
    elif guard == 'closed':
        d.host.client.closed = True
    before = (list(d.sent), list(d.texts), list(d.tunes))
    d.session.command('/vmail collect KQ4QWH')
    assert (d.sent, d.texts, d.tunes) == before
    assert d.kinds('error') and len(d.session.mail_collection._load()) == 1


def test_collection_returns_to_notice_cf_after_passive_tuning():
    d = Desk(**{'CONNECT W9SSJ KQ4QWH': ['OK', 'CONNECTED W9SSJ KQ4QWH 500']})
    notice(d)
    d.flow.dial = 14105750
    d.session.command('/vmail collect KQ4QWH')
    d.drive(until=lambda: d.host.client.connected)
    assert d.tunes == [14105000]


def test_corrupt_notification_store_is_preserved(tmp_path):
    d = Desk(mailbox=tmp_path)
    path = tmp_path / 'relay-notices.json'
    path.write_text('{bad')
    notice(d)
    d.do('/vmail pending')
    assert path.read_text() == '{bad' and not d.kinds('vmail_notice')
    assert not d.texts


def test_slot_notice_uses_saved_channel_and_heard_bandwidth():
    d = Desk(**{'CONNECT W9SSJ KQ4QWH': ['OK', 'CONNECTED W9SSJ KQ4QWH 2300', 'SN -5.0']})
    d.hear('CQFRAME KQ4QWH-12 2300')
    notice(d, dial=14106500)
    d.do('/vmail collect KQ4QWH')
    assert d.flow.dial == 14106500 and 'BW2300' in d.sent
    assert d.texts == ['15 de W9SSJ <R-05>']
    assert d.host.client.connected and not d.kinds('error')


def test_collection_without_notice_uses_current_band_cf():
    d = Desk(**{'CONNECT W9SSJ KQ4QWH': ['OK', 'CONNECTED W9SSJ KQ4QWH 500']})
    d.flow.dial = 14105750
    d.session.command('/vmail collect KQ4QWH')
    d.drive(until=lambda: d.host.client.connected)
    assert d.flow.dial == 14105000
    assert not d.mailbox.entries()


def test_shared_mailbox_notices_are_scoped_to_local_callsign(tmp_path):
    d = Desk(mailbox=tmp_path)
    notice(d)
    d.host.client.mycall = 'N0XYZ'
    d.do('/vmail pending')
    assert d.said[-1] == 'No parked-mail notices saved for this callsign'
    d.do('/vmail dismiss KQ4QWH')
    assert len(d.session.mail_collection._load()) == 1


def test_compact_old_direct_mail_retains_compose_time_and_email_author():
    d = answered()
    d.receive('<SM><TME:2026-10-25 12:00:00><SBJ:Old direct><MSG:hi>')
    assert d.mailbox.read(1)['from'] == 'N0XYZ'
    assert d.mailbox.read(1)['composed_utc'] == '2026-10-25 12:00:00'
    d.receive(MAIL.replace('K1ABC', 'sender@example.org'))
    assert d.mailbox.read(2)['from'] == 'sender@example.org'
    assert d.texts.count('5 <SMR>') == 2


def test_outgoing_direct_mail_is_not_displayed_as_relayed():
    d = answered()
    d.do('/vmail send Hello')
    d.do('73')
    d.do('.')
    d.do('/vmail 1')
    assert not any('relayed by' in line for line in d.said)
