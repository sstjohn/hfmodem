# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk, answered
from hfmodem.tests.contact.test_varac_cf_qsy import caller
from hfmodem.tests.contact.test_email_gateway import gateway, compose as compose_email


DEPOSIT = '<SM><TME:2026-10-28 12:00:00><FRM:K1ABC><TO:W9SSJ-1><SBJ:Hello><MSG:café\r\n73>'


def queued(d, destination='N0XYZ', subject='queued', body='body'):
    d.session.command(f'/vmail park {destination} {subject}')
    d.session.command(body)
    d.session.command('.')
    return d.session.parking.store.records[-1]


def off_channel_link(d, peer='N0XYZ'):
    d.flow.dial = v.slot_frequency(20, 3)
    d.hear(f'CONNECTED {peer} W9SSJ 500')


def mail_wires(d):
    return [raw for raw in d.host.data.sent if b'<SM>' in raw]


@pytest.mark.parametrize('command', ['/vmail send w9ssj-01 Frozen subject',
                                    '/vmail park W9SSJ-1 Frozen subject'])
def test_offline_queue_is_durable_and_never_calls_tunes_or_transmits(tmp_path, command):
    d = Desk(mailbox=tmp_path)
    before = (list(d.sent), list(d.texts), list(d.tunes), list(d.armed))
    d.session.command(command)
    composed = d.session.parking.draft['composed']
    d.session.command('line one')
    d.session.command('line two')
    d.session.command('.')
    assert (d.sent, d.texts, d.tunes, d.armed) == before
    row, = d.session.parking.store.records
    assert row['to'] == 'W9SSJ-1' and row['owner'] == row['from'] == row['via'] == 'W9SSJ'
    assert row['body'] == 'line one\r\nline two' and row['composed_utc'] == composed
    assert row['state'] == 'queued' and row['origin'] == 'local'
    assert json.loads((tmp_path / 'parked-mail.json').read_text())['records'] == [row]
    assert not d.mailbox.entries()


def test_local_draft_destination_and_time_stay_frozen_when_link_arrives():
    d = Desk()
    d.session.command('/vmail send W9SSJ-1 Frozen subject')
    composed = d.session.parking.draft['composed']
    off_channel_link(d, 'K2ABC')
    before = list(d.host.data.sent)
    d.session.command('still local')
    d.session.command('.')
    row, = d.session.parking.store.records
    assert row['to'] == 'W9SSJ-1' and row['composed_utc'] == composed
    assert d.host.data.sent == before
    assert row['state'] == 'queued'


@pytest.mark.parametrize('busy', ['steps', 'pending', 'closing', 'quitting', 'bbs',
                                 'file', 'broadcast', 'qsy', 'follow'])
def test_local_parking_reads_settings_cancel_and_draft_work_in_busy_states(busy):
    d = Desk()
    row = queued(d)
    if busy == 'steps':
        d.session.steps = iter([.1])
    elif busy == 'pending':
        d.host.client.pending = True
    elif busy == 'closing':
        d.session._closing = True
    elif busy == 'quitting':
        d.session._quitting = True
    elif busy == 'bbs':
        d.session.bbs.sid = 'test-lobby'
    elif busy == 'file':
        d.session.files.waiting = True
    elif busy == 'broadcast':
        d.session.broadcast.active = True
    elif busy == 'qsy':
        d.flow.qsy_phase = 'waiting for peer arrival'
    else:
        d.session.following = 3
    before = (list(d.sent), list(d.texts), list(d.tunes), list(d.armed))
    d.session.command('/vmail park W9SSJ-1 new draft')
    d.session.command('not chat or game input')
    draft = d.session.parking.draft
    for command in ('/vmail parking off', '/vmail relay off', '/vmail parking',
                    '/vmail relay', '/vmail parked', f'/vmail parked {row["id"]}'):
        d.session.command(command)
        assert d.session.parking.draft is draft
    d.session.command(f'/vmail cancel {row["id"]}')
    assert d.session.parking.store.get(row['id'])['state'] == 'cancelled'
    d.session.command('.')
    assert d.session.parking.store.records[-1]['body'] == 'not chat or game input'
    assert (d.sent, d.texts, d.tunes, d.armed) == before
    assert not d.kinds('error')


def test_cancel_local_draft_changes_neither_store_nor_transmitter():
    d = Desk()
    before = (list(d.sent), list(d.texts), list(d.tunes))
    d.session.command('/vmail park W9SSJ-1 draft')
    d.session.command('discard me')
    d.session.command('/stop')
    assert d.session.parking.draft is None
    assert not d.session.parking.store.records
    assert (d.sent, d.texts, d.tunes) == before


def test_linked_legacy_subject_beginning_with_callsign_is_not_a_destination():
    d = answered()
    d.session.command('/vmail send K1ABC told me this')
    d.session.command('ordinary direct mail')
    d.session.command('.')
    row, = d.session.parking.store.records
    assert row['to'] == 'N0XYZ' and row['subject'] == 'K1ABC told me this'
    assert b'<TO:' not in mail_wires(d)[0] and b'<TME:' not in mail_wires(d)[0]
    assert row['state'] == 'in_flight'
    d.receive('<SMR>')
    assert d.session.parking.store.get(row['id'])['state'] == 'delivered'


def test_unsupported_deposits_refuse_and_parking_off_preserves_inbox():
    d = answered()
    for destination in ('ALL', 'CQ', 'W9SSJ-16', 'X-W9SSJ', 'a@example.org'):
        d.receive(DEPOSIT.replace('W9SSJ-1', destination))
        assert d.texts[-1] == '5 <SMF>'
    assert not d.session.parking.store.records
    d.session.command('/vmail parking off')
    d.receive(DEPOSIT)
    assert d.texts[-1] == '6 <SMFP>'
    assert not d.session.parking.store.records
    d.receive(DEPOSIT.replace('W9SSJ-1', 'w9ssj-00'))
    assert d.texts[-1] == '5 <SMR>'
    assert len(d.mailbox.entries()) == 1
    assert d.mailbox.read(1)['from'] == 'K1ABC'


def test_deposit_canonical_ssids_attribution_and_duplicate_after_parking_off():
    d = answered()
    d.receive(DEPOSIT.replace('W9SSJ-1', 'w9ssj-01'))
    row, = d.session.parking.store.records
    assert row['to'] == 'W9SSJ-1' and row['from'] == 'K1ABC' and row['via'] == 'N0XYZ'
    assert row['composed_utc'] == '2026-10-28 12:00:00' and row['state'] == 'queued'
    assert not d.mailbox.entries() and d.texts[-1] == '5 <SMR>'
    d.session.command('/vmail parking off')
    d.receive(DEPOSIT)
    assert d.texts[-1] == '5 <SMR>' and len(d.session.parking.store.records) == 1
    d.receive(DEPOSIT.replace('café', 'other body'))
    assert d.texts[-1] == '6 <SMFP>'


def test_deposit_without_frm_infers_connected_peer_separate_from_relay_owner():
    d = answered()
    d.receive('<SM><TO:W9SSJ-1><MSG:subjectless body>')
    row, = d.session.parking.store.records
    assert row['from'] == row['via'] == 'N0XYZ' and row['owner'] == 'W9SSJ'
    assert row['composed_utc'] is None and row['subject'] == ''
    assert row['dedup_key'] is None and row['estimated_composed_utc'] == row['received_utc']
    d.session.command('/vmail parked 1')
    assert any('local receive-time estimate; original unknown' in line for line in d.said)


def test_missing_tme_dedup_only_within_contact():
    d = answered()
    wire = '<SM><FRM:K1ABC><TO:W9SSJ-1><SBJ:Hi><MSG:repeat>'
    d.receive(wire)
    d.receive(wire)
    assert len(d.session.parking.store.records) == 1
    d.hear('DISCONNECTED')
    off_channel_link(d)
    d.receive(wire)
    assert len(d.session.parking.store.records) == 2
    assert all(row['dedup_key'] is None for row in d.session.parking.store.records)


@pytest.mark.parametrize('prefix', ['', '<TL>\r\n', '<QSYR>\r\n'])
@pytest.mark.parametrize('inside', [False, True])
def test_aggregates_at_every_byte_split_keep_one_custody_ack_and_next_frame(prefix, inside):
    text = prefix + DEPOSIT
    wire = v.frame_message(text if inside else text[:-1]) + (b'' if inside else b'>')
    for split in range(len(wire) + 1):
        d = answered()
        before = len(d.texts)
        if split:
            d.host.data.incoming.append(wire[:split])
        d.tick()
        if split < len(wire):
            assert not d.session.parking.store.records
        d.host.data.incoming.append(wire[split:] + v.frame_message('next frame'))
        d.tick()
        assert len(d.session.parking.store.records) == 1
        assert d.texts[before:].count('5 <SMR>') == 1
        assert [event.raw for event in d.host.events if event.kind == 'message'][-2:] == [text, 'next frame']
        assert not d.host.client._messages.pending
        d.session.parking.store.close()


@pytest.mark.parametrize('prefix', ['', '<TL>\r\n', '<QSYR>\r\n', '<AWQ>\r\n', 'hello '])
@pytest.mark.parametrize('body', ['<SMR>', '<QSYF>000014107250</QSYF>', '<ER>', '<VRP>'])
def test_malformed_envelope_never_leaks_body_controls_or_ack(prefix, body):
    d = answered()
    queued(d)
    d.session.parking.start_delivery()
    d.tick()
    before = (list(d.texts), list(d.tunes))
    d.receive(prefix + '<SM><SBJ:bad><MSG:literal ' + body + '>')
    assert (d.texts, d.tunes) == before
    assert d.session.parking.store.get(1)['state'] == 'in_flight'
    assert d.session.mail_ack.busy
    assert not d.mailbox.entries()
    assert not d.kinds('vmail_parked')


@pytest.mark.parametrize('inside', [False, True])
def test_tl_mail_aggregate_satisfies_responder_and_custody_at_every_split(inside):
    text = '<TL>\r\n' + DEPOSIT
    wire = v.frame_message(text if inside else text[:-1]) + (b'' if inside else b'>')
    for split in range(len(wire) + 1):
        d = Desk()
        d.hear('SN -5.0', 'CONNECTED N0XYZ W9SSJ 500')
        d.receive('<QSYF>000014107250</QSYF>')
        if split:
            d.host.data.incoming.append(wire[:split])
        d.host.data.incoming.append(wire[split:] + v.frame_message('de N0XYZ <R-05>'))
        d.tick()
        d.drive()
        assert d.session.steps is None and d.flow.dial == 14107250
        assert d.texts.count('6 <QSYR>') == 1
        assert d.texts.count('5 <SMR>') == 1
        assert len(d.session.parking.store.records) == 1
        assert not d.kinds('error')
        d.session.parking.store.close()


@pytest.mark.parametrize('inside', [False, True])
def test_qsyr_mail_aggregate_satisfies_requester_and_custody(inside):
    d = caller()
    text = '<QSYR>\r\n' + DEPOSIT
    wire = v.frame_message(text if inside else text[:-1]) + (b'' if inside else b'>')
    def reply(data):
        d.host.lines('BUFFER 0')
        if data == v.frame_message('<TL>'):
            d.host.data.incoming.extend(bytes([byte]) for byte in wire)
    d.host.data.respond = reply
    d.do('/call KE8PLM')
    assert d.session.steps is None and d.flow.dial != d.session.calling_hz
    assert d.texts.count('5 <SMR>') == 1
    assert len(d.session.parking.store.records) == 1
    assert not d.kinds('error')


@pytest.mark.parametrize('subject_field', ['', '<SBJ:>'])
def test_unknown_composition_and_empty_subject_survive_restart_and_onward_delivery(tmp_path, subject_field):
    d = answered(mailbox=tmp_path)
    d.receive('<SM><FRM:K1ABC><TO:W9SSJ-1>' + subject_field + '<MSG:old but undated>')
    accepted = d.session.parking.store.get(1)
    d.session.parking.store.close()
    restarted = Desk(mailbox=tmp_path)
    restarted.session.parking.store.clock = lambda: accepted['received_unix'] + 86400
    off_channel_link(restarted, 'W9SSJ-1')
    wire, = mail_wires(restarted)
    assert b'<TME:' not in wire and b'<FRM:K1ABC>' in wire
    assert b'<SBJ:>' in wire and b'<TO:' not in wire
    assert v.VMail.parse(wire.partition(b' ')[2].decode()).subject == ''
    assert restarted.session.parking.store.get(1)['state'] == 'in_flight'
    restarted.receive('<SMR>')
    assert restarted.session.parking.store.get(1)['state'] == 'delivered'


def test_fifo_waits_for_smr_and_drain_and_report_plus_duplicate_ack_is_single_outcome():
    d = Desk()
    first = queued(d, subject='first')
    second = queued(d, subject='second')
    off_channel_link(d)
    assert len(mail_wires(d)) == 1
    assert d.session.parking.store.get(first['id'])['state'] == 'in_flight'
    assert d.session.parking.store.get(second['id'])['state'] == 'queued'
    d.host.data.incoming.append(v.frame_message('de N0XYZ <R-02> <SMR>') + v.frame_message('<SMR>'))
    d.tick()
    assert d.session.parking.store.get(first['id'])['state'] == 'delivered'
    assert d.session.parking.store.get(second['id'])['state'] == 'in_flight'
    assert len(mail_wires(d)) == 2
    d.receive('<SMR>')
    assert d.session.parking.store.get(second['id'])['state'] == 'delivered'


@pytest.mark.parametrize('ack', ['SMF', 'SMFP'])
def test_negative_ack_retains_custody_and_stops_fifo_for_current_link(ack):
    d = Desk()
    first = queued(d, subject='first')
    second = queued(d, subject='second')
    off_channel_link(d)
    d.receive(f'<{ack}>')
    assert d.session.parking.store.get(first['id'])['state'] == 'queued'
    assert d.session.parking.store.get(first['id'])['last_error'] == ack
    assert d.session.parking.store.get(second['id'])['state'] == 'queued'
    for _ in range(5):
        d.tick()
        d.session.parking.start_delivery()
    assert len(mail_wires(d)) == 1


def test_timeout_keeps_ack_owner_blocks_new_send_and_accepts_late_smr():
    d = Desk()
    first = queued(d, subject='first')
    queued(d, subject='second')
    off_channel_link(d)
    d.host.now += 91
    d.tick()
    assert d.session.parking.store.get(first['id'])['state'] == 'uncertain'
    assert d.session.mail_ack.state == 'unresolved'
    d.session.command('/vmail send blocked')
    d.session.command('not admitted')
    d.session.command('.')
    assert len(mail_wires(d)) == 1
    d.receive('<SMR>')
    assert d.session.parking.store.get(first['id'])['state'] == 'delivered'
    assert len(mail_wires(d)) == 1


def test_cancellation_inflight_keeps_owner_and_honors_smr():
    d = Desk()
    first = queued(d)
    off_channel_link(d)
    d.session.command(f'/vmail cancel {first["id"]}')
    row = d.session.parking.store.get(first['id'])
    assert row['state'] == 'in_flight' and row['cancel_requested']
    assert d.session.mail_ack.busy
    d.receive('<SMR>')
    assert d.session.parking.store.get(first['id'])['state'] == 'delivered'


def test_disconnect_reconnect_retains_uncertain_and_rejects_old_generation_ack():
    d = Desk()
    row = queued(d)
    off_channel_link(d)
    old_generation = d.host.client.link_generation
    d.hear('DISCONNECTED')
    assert d.session.parking.store.get(row['id'])['state'] == 'uncertain'
    off_channel_link(d)
    assert d.session.parking.store.get(row['id'])['attempts'] == 2
    current = d.host.client.message_sequence
    d.session.mail_ack.message(v.decode_application('<SMR>'), current + 1, old_generation)
    assert d.session.parking.store.get(row['id'])['state'] == 'in_flight'
    d.receive('<SMR>')
    assert d.session.parking.store.get(row['id'])['state'] == 'delivered'


@pytest.mark.parametrize('action', ['timeout', 'stop', 'er_before_smr'])
def test_delayed_email_smr_cannot_complete_parked_or_direct_mail(action):
    d = gateway()
    compose_email(d)
    d.session.command('.')
    d.tick()
    if action == 'timeout':
        d.host.now += 121
        d.tick()
    elif action == 'stop':
        d.session.command('/stop')
    else:
        d.receive('<ER>')
    assert not d.session.email.waiting
    queued(d, subject='parking stays queued')
    d.session.parking.start_delivery()
    d.tick()
    d.session.command('/vmail send direct must wait')
    d.session.command('no second SM transaction')
    d.session.command('.')
    assert len(mail_wires(d)) == 1 and b'<EG>' in mail_wires(d)[0]
    d.receive('<SMR>')
    assert d.session.mail_ack.kind == 'email' and d.session.mail_ack.busy
    assert all(row['state'] == 'queued' for row in d.session.parking.store.records)
    assert any(event.get('event') == 'vmail_accepted' for event in d.kinds('email'))
    if action == 'er_before_smr':
        assert 'gateway reported email sent' in d.session.email.phase
    else:
        d.receive('<ER>')
        assert any(event.get('event') == 'gateway_reported_sent' for event in d.kinds('email'))
    assert len(mail_wires(d)) == 1


def test_peek_is_explicit_negative_answer_once_per_complete_top_level_message():
    d = answered()
    before = len(d.texts)
    wire = v.frame_message('<VRP>')
    for byte in wire:
        d.host.data.incoming.append(bytes([byte]))
        d.tick()
    assert d.texts[before:] == ['6 <VRPJ>']
    assert not d.mailbox.entries() and not d.session.parking.store.records


def storage_fault(monkeypatch, operation, *, commit=1):
    from hfmodem.contact import mail_parking as p
    fsync, replace = p.os.fsync, p.os.replace
    fsync_calls = replace_calls = 0

    def sync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        target = operation.startswith('file_fsync') and fsync_calls == 2 * commit - 1
        if target and operation.endswith('before'):
            raise OSError('injected parking custody failure')
        fsync(fd)
        if target:
            raise OSError('injected parking custody failure')

    def move(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        target = operation.startswith('replace') and replace_calls == commit
        if target and operation.endswith('before'):
            raise OSError('injected parking custody failure')
        replace(source, destination)
        if target:
            raise OSError('injected parking custody failure')

    monkeypatch.setattr(p.os, 'fsync', sync)
    monkeypatch.setattr(p.os, 'replace', move)


@pytest.mark.parametrize('operation', ['file_fsync_before', 'file_fsync_after',
                                       'replace_before', 'replace_after'])
@pytest.mark.parametrize('stage', ['admission', 'attempt'])
def test_direct_custody_io_failure_is_contained_before_any_mail_bytes(
        tmp_path, monkeypatch, operation, stage):
    from hfmodem.contact.mail_parking import ParkedStore
    d = answered(mailbox=tmp_path)
    d.session.command('/vmail send retained draft')
    d.session.command('must be durable before transmission')
    before = list(mail_wires(d))
    with monkeypatch.context() as patch:
        storage_fault(patch, operation, commit=1 if stage == 'admission' else 2)
        d.session.command('.')
    assert mail_wires(d) == before
    assert d.host.client.connected and not d.host.client.closed
    assert not d.session.parking.store.available
    assert any('injected parking custody failure' in line for line in d.said)
    d.session.parking.store.close()
    restarted = ParkedStore(tmp_path)
    assert restarted.available
    if stage == 'attempt' or operation == 'replace_after':
        row, = restarted.records
        assert row['body'] == 'must be durable before transmission'
        assert row['state'] == ('uncertain' if stage == 'attempt' and operation == 'replace_after'
                                else 'queued')
    else:
        assert not restarted.records
    restarted.close()


@pytest.mark.parametrize('operation', ['file_fsync_before', 'file_fsync_after',
                                       'replace_before', 'replace_after'])
def test_qsy_mail_ready_store_failure_keeps_link_and_never_sends_uncommitted_attempt(
        monkeypatch, operation):
    from hfmodem.contact.mail_parking import ParkedStore
    d = caller()
    row = queued(d, 'KE8PLM')

    def reply(data):
        d.host.lines('BUFFER 0')
        if data == v.frame_message('<TL>'):
            d.host.data.incoming.append(v.frame_message('<QSYR>'))

    d.host.data.respond = reply
    with monkeypatch.context() as patch:
        storage_fault(patch, operation)
        d.do('/call KE8PLM')
    assert not mail_wires(d)
    assert d.host.client.connected and not d.host.client.closed
    assert d.session.steps is None
    assert not d.session.parking.store.available
    assert any('injected parking custody failure' in line for line in d.said)
    d.session.parking.store.close()
    restarted = ParkedStore(d.mailbox.root)
    assert restarted.available
    restored = restarted.get(row['id'])
    assert restored['state'] == ('uncertain' if operation == 'replace_after' else 'queued')
    assert restored['body'] == row['body']
    restarted.close()


def test_direct_writer_failure_after_possible_admission_retains_uncertainty_and_owner():
    d = answered()
    d.session.command('/vmail send ambiguous admission')
    d.session.command('the sender cannot know whether these bytes escaped')

    def fail(payload):
        raise OSError('write failed after possible admission')

    d.host.data.respond = fail
    d.session.command('.')
    row, = d.session.parking.store.records
    assert row['state'] == 'uncertain' and row['attempts'] == 1
    assert d.session.mail_ack.state == 'unresolved' and d.session.mail_ack.busy
    assert len(mail_wires(d)) == 1
    d.host.data.respond = lambda payload: d.host.lines('BUFFER 0')
    d.receive('<SMR>')
    assert d.session.parking.store.get(row['id'])['state'] == 'delivered'


def test_partial_duplicate_ack_must_finish_before_next_fifo_attempt():
    d = Desk()
    first = queued(d, subject='first')
    second = queued(d, subject='second')
    off_channel_link(d)
    ack = v.frame_message('<SMR>')
    d.host.data.incoming.append(ack + ack[:3])
    d.tick()
    assert d.session.parking.store.get(first['id'])['state'] == 'delivered'
    assert d.session.parking.store.get(second['id'])['state'] == 'queued'
    assert len(mail_wires(d)) == 1
    d.host.data.incoming.append(ack[3:])
    d.tick()
    assert d.session.parking.store.get(second['id'])['state'] == 'in_flight'
    assert len(mail_wires(d)) == 2
    d.receive('<SMR>')
    assert d.session.parking.store.get(second['id'])['state'] == 'delivered'
