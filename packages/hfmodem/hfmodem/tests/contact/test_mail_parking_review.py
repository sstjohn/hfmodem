# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from pathlib import Path

import pytest

from hfmodem.contact import mail_parking as p, mail_storage, varac as v
from hfmodem.contact.broadcast import Broadcast
from hfmodem.tests.contact.test_broadcast_client import answer, wire
from hfmodem.tests.contact.test_email_gateway import gateway, compose
from hfmodem.tests.contact.test_mail_parking_notifications import ready, deposit, beacon, notice
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.tests.contact.test_varac_cf_qsy import parked, advance_mail


def test_email_pending_frame_preserves_draft_and_does_not_reserve_generation():
    d = gateway()
    compose(d)
    d.host.data.incoming.append(b'40 <partial')
    d.tick()
    d.session.command('.')
    assert not d.session.email.waiting
    assert d.session.email.sent_generation is None
    assert d.session.email.draft is not None
    assert not d.session.mail_ack.busy


@pytest.mark.parametrize('command', ['/quit', '/bye', '/stop'])
def test_local_draft_does_not_intercept_link_controls(command):
    d = Desk()
    d.hear('CONNECTED K1ABC W9SSJ 500')
    d.session.command('/vmail park K2ABC hello')
    result = d.session.command(command)
    assert result is False if command == '/quit' else d.session._closing
    assert d.session.parking.draft is None
    assert any('draft dropped' in text for text in d.said)


@pytest.mark.parametrize('operation', ['cq', 'broadcast'])
def test_local_draft_stop_cancels_unlinked_operation(operation):
    d = ready(caps=False)
    if operation == 'cq':
        d.session.command('/cq 1 repeat')
    else:
        deposit(d)
        beacon(d)
    d.session.command('/vmail park K2ABC hello')
    d.session.command('/stop')
    assert d.session.broadcast.cancelled if operation == 'broadcast' else d.flow.cancelled


def test_c4_qsyr_precedes_receipt_ack():
    d = Desk()
    parked(d)
    d.hear('CONNECTED K1ABC W9SSJ 500', 'SN -7.0')
    d.receive('<QSYF>000014105750</QSYF>')
    d.receive('<TL>\r\n<SM><SBJ:reply><MSG:body>')
    advance_mail(d)
    assert d.texts.index('6 <QSYR>') < d.texts.index('5 <SMR>')
    assert d.mailbox.read(1)['body'] == 'body'


@pytest.mark.parametrize('prefix', ['<SMR>\r\n', '<SMF>\r\n', '<SMFP>\r\n',
                                  'de K1ABC <R-07> <S:V>\r\n',
                                  'de K1ABC <R-07> <S:V>\r\n<SMR>\r\n'])
@pytest.mark.parametrize('outside', [False, True])
def test_leading_ack_and_reply_mail_both_dispatched(prefix, outside):
    d = Desk()
    row = parked(d)
    d.flow.dial = v.slot_frequency(20, 11)
    d.hear('CONNECTED K1ABC W9SSJ 500')
    text = prefix + '<SM><SBJ:back><MSG:reply mail>'
    raw = v.frame_message(text[:-1]) + b'>' if outside else v.frame_message(text)
    for byte in raw:
        d.host.data.incoming.append(bytes([byte]))
        d.tick()
    state = ('queued' if '<SMF>' in prefix or '<SMFP>' in prefix else
             'delivered' if '<SMR>' in prefix else 'in_flight')
    assert d.session.parking.store.get(row['id'])['state'] == state
    assert d.mailbox.read(1)['body'] == 'reply mail'
    assert d.texts.count('5 <SMR>') == 1


@pytest.mark.parametrize('text', ['chat containing <SM> literally', 'BBS help: <SM><SBJ:x><MSG:body>'])
def test_mid_text_sm_stays_chat(text):
    d = Desk()
    d.hear('CONNECTED K1ABC W9SSJ 500')
    d.receive(text)
    assert v.decode_application(text).envelope is None
    assert any(text in line for line in d.said)
    assert not d.mailbox.entries()


def test_mid_text_sm_stays_visible_in_remote_bbs():
    d = Desk()
    d.hear('CONNECTED K1ABC W9SSJ 500')
    d.session.bbs.remote_view = True
    d.receive('BBS help: <SM> is a mail control')
    assert any('BBS help: <SM>' in line for line in d.said)
    assert not d.kinds('bbs_control_refused')


def test_notice_sent_commit_failure_retains_cooldown_and_terminal(monkeypatch):
    d = ready()
    deposit(d)
    beacon(d)
    reserved = notice(d)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(d.session.parking.store, '_commit', fail)
    answer(d, 'QUEUED', 'SENT')
    assert d.session.steps is None and not d.session.broadcast.active
    assert notice(d) == reserved
    assert not d.session.parking.store.eligible('W9SSJ', 'K1ABC')
    assert any('disk full' in text for text in d.said)
    assert 'ABORT' not in d.sent


@pytest.mark.parametrize('boundary', ['replace', 'directory_sync'])
def test_notice_sent_disk_failure_preserves_restart_cooldown(tmp_path, monkeypatch, boundary):
    d = ready(tmp_path)
    deposit(d)
    beacon(d)
    reserved = notice(d)
    def fail(*args):
        raise OSError('sent status disk failure')
    with monkeypatch.context() as patch:
        if boundary == 'replace':
            patch.setattr(p.os, 'replace', fail)
        else:
            patch.setattr(mail_storage, 'sync_directory', fail)
        answer(d, 'QUEUED', 'SENT')
    assert not d.session.parking.store.available
    assert d.session.steps is None and not d.session.broadcast.active
    assert 'ABORT' not in d.sent
    d.session.parking.store.close()
    store = p.ParkedStore(tmp_path)
    try:
        assert store.available and not store.eligible('W9SSJ', 'K1ABC')
        assert store.notice('W9SSJ', 'K1ABC')['next_eligible'] == reserved['next_eligible']
        assert store.get(1)['state'] == 'queued'
    finally:
        store.close()


@pytest.mark.parametrize('mycall,to,peer', [('W9SSJ/P', 'w9ssj/p-00', 'K1ABC/P'),
                                         ('TEAM-ALPHA', 'TEAM-ALPHA', 'TEAM-BRAVO')])
def test_portable_and_tactical_inbox_and_collection(mycall, to, peer):
    d = Desk()
    d.host.client.mycall = mycall
    d.hear(f'CONNECTED {peer} {mycall} 500')
    d.session.mail_collection.notice(Broadcast(1, peer, to, b'<SMP>'), dial=d.flow.dial)
    d.receive(f'<SM><TO:{to}><SBJ:hi><MSG:for you>')
    assert len(d.mailbox.entries()) == 1
    assert d.texts.count('5 <SMR>') == 1
    d.session.command('/vmail pending')
    assert any(peer + ': mail notice' in text for text in d.said)
    assert not any('could not be updated' in text for text in d.said)
    assert d.session.mail_collection._load()[0]['received'] == 1


@pytest.mark.parametrize('failure', ['locked', 'corrupt', 'error'])
def test_collection_survives_unavailable_parking_store(tmp_path, failure):
    d = Desk(mailbox=tmp_path)
    original = d.session.parking.store
    if failure == 'locked':
        second = p.ParkedStore(tmp_path)
        assert not second.available
        d.session.parking.store = second
    elif failure == 'corrupt':
        d.session.parking.store.close()
        (tmp_path / 'parked-mail.json').write_text('{broken')
        d.session.parking.store = p.ParkedStore(tmp_path)
    else:
        d.session.parking.store.error = 'disk full'
    d.session.mail_collection.notice(Broadcast(1, 'K1ABC', 'W9SSJ', b'<SMP>'), dial=d.flow.dial)
    assert len(d.session.mail_collection._load()) == 1
    assert any('unavailable' in text for text in d.said)
    original.close()


def test_locked_store_preserves_persisted_relay_off(tmp_path):
    d = Desk(mailbox=tmp_path)
    original = d.session.parking.store
    original.configure(relay=False)
    d.session.parking.store = p.ParkedStore(tmp_path)
    assert not d.session.parking.store.available
    d.session.mail_collection.notice(Broadcast(1, 'K1ABC', 'W9SSJ', b'<SMP>'), dial=d.flow.dial)
    assert not d.session.mail_collection._load()
    original.close()


def test_direct_mail_with_portable_calls_survives_restart(tmp_path):
    d = Desk(mailbox=tmp_path)
    d.host.client.mycall = 'W9SSJ/P'
    d.flow.dial = v.slot_frequency(20, 11)
    d.hear('CONNECTED K1ABC/P W9SSJ/P 500')
    for line in ('/vmail send hello', 'body', '.'):
        d.session.command(line)
    assert any('<SM>' in text for text in d.texts)
    d.receive('<SMR>')
    d.session.parking.store.close()
    store = p.ParkedStore(tmp_path)
    try:
        assert store.available and store.get(1)['state'] == 'delivered'
    finally:
        store.close()


def test_unready_notice_explained_once_and_in_summary():
    d = ready()
    deposit(d)
    d.session.hopping.readiness.clear()
    assert '/band ready' in d.session.parking.summary()
    d.session.command('/status')
    before = len(d.said)
    beacon(d)
    beacon(d)
    assert sum('/band ready' in text for text in d.said[before:]) == 1
    assert not wire(d) and not notice(d)


def test_directed_notice_disables_listen_and_restores_monitoring():
    d = ready()
    deposit(d)
    start = len(d.sent)
    beacon(d)
    commands = d.sent[start:]
    assert commands.index('LISTEN OFF') < commands.index('BROADCAST CAPS')
    answer(d, 'QUEUED', 'SENT')
    assert d.sent[-1] == 'LISTEN CQ'


def test_direct_compose_refuses_cf_before_body():
    d = Desk()
    d.hear('CONNECTED K1ABC W9SSJ 500')
    d.session.command('/vmail send subject')
    assert d.session.compose is None
    assert any('calling frequency' in text for text in d.said)


def test_receipt_dedup_reads_each_record_at_most_once(tmp_path, monkeypatch):
    box = v.Mailbox(tmp_path)
    for n in range(12):
        box.store('in', 'W9SSJ', 'K1ABC', v.VMail(str(n), 'body'), composed_utc='2026-10-28 12:00:00')
    d = Desk(mailbox=tmp_path)
    d.hear('CONNECTED K1ABC W9SSJ 500')
    reads = []
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path.name.startswith('202'):
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    d.receive('<SM><TME:2026-10-28 12:00:00><SBJ:11><MSG:body>')
    assert len(reads) <= 12
    assert len(d.mailbox.entries()) == 12


def test_linked_ticks_do_not_deepcopy_parking_snapshot(monkeypatch):
    d = Desk()
    parked(d)
    d.hear('CONNECTED K1ABC W9SSJ 500')
    calls = []
    original = p.copy.deepcopy
    def copy(value, *args):
        calls.append(value)
        return original(value, *args)
    monkeypatch.setattr(p.copy, 'deepcopy', copy)
    for _ in range(10):
        d.session.parking.pending()
        d.session.parking.active
        d.tick()
    assert not calls


def test_pending_views_cannot_mutate_custody_and_expire_on_deadline():
    d = ready(wall=1000000)
    row = deposit(d)
    store = d.session.parking.store
    view = store.pending('W9SSJ', 'K1ABC', snapshot=False)[0]
    with pytest.raises(TypeError):
        view['body'] = 'lost'
    with pytest.raises(TypeError):
        view['notification']['last_sent'] = 0
    d.wall[0] = row['expires_unix']
    assert not store.pending('W9SSJ', 'K1ABC', snapshot=False)
    assert store.get(row['id'])['state'] == 'expired'
    assert store.get(row['id'])['body'] == 'hello'


def test_cached_inbox_does_not_ack_deleted_custody(tmp_path):
    d = Desk(mailbox=tmp_path)
    row = d.mailbox.store('in', 'W9SSJ', 'K1ABC', v.VMail('hi', 'body'),
                          composed_utc='2026-10-28 12:00:00')
    d.mailbox.entries()
    (tmp_path / (row['id'] + '.json')).unlink()
    d.hear('CONNECTED K1ABC W9SSJ 500')
    d.receive('<SM><TME:2026-10-28 12:00:00><SBJ:hi><MSG:body>')
    assert '5 <SMF>' in d.texts and '5 <SMR>' not in d.texts


def test_content_budget_uses_vmail_max(monkeypatch):
    monkeypatch.setattr(v, 'VMAIL_MAX', 4)
    with pytest.raises(ValueError):
        p.validate_content('hi', 'abc')


def test_mailbox_and_parking_commit_use_shared_directory_sync(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mail_storage, 'sync_directory', lambda path: calls.append(Path(path)))
    d = Desk(mailbox=tmp_path)
    calls.clear()
    parked(d)
    assert tmp_path in calls
    calls.clear()
    d.mailbox._append_index([])
    assert calls == [tmp_path]


def test_ack_owner_can_release_only_unsubmitted_reservation():
    d = Desk()
    d.hear('CONNECTED K1ABC W9SSJ 500')
    owner = d.session.mail_ack
    owner.reserve('parking', 1, lambda *args: None)
    owner.release_unsent()
    assert not owner.busy and owner.callback is None
