# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Local reads and settings work in every state; refusals that remain protect the link."""
import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.tests.contact.test_varac_cf_qsy import caller
from hfmodem.tests.contact.test_varac_close import stuck_desk

LIST = '<BL:hello.txt|2026-01-01|5>'


def closing():
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    desk.hear('CQFRAME K7QB-2 500')
    desk.session.command('/bye')
    desk.tick()
    assert desk.session._closing
    return desk


def test_heard_lists_while_a_close_is_in_progress():
    desk = closing()
    desk.said.clear()
    assert desk.session.command('/heard')
    assert desk.said == [desk.session.listing(desk.session.heard[0])]
    desk.session.quit()


@pytest.mark.parametrize('line, said', [
    ('/slots', 'slot monitor off'),
    ('/vmail', 'no VMails'),
    ('/vmail pending', 'No parked-mail notices saved for this callsign'),
    ('/beacon hop status', 'Band readiness: '),
    ('/verbose on', 'verbose on: '),
    ('/follow off', 'follow off'),
    ('/back', 'back; 0 new VMails'),
])
def test_local_reads_and_settings_answer_while_closing(line, said):
    desk = closing()
    assert desk.session.command(line)
    assert desk.said[-1].startswith(said)
    desk.session.quit()


def test_auto_and_away_apply_while_closing_without_reopening_the_listener():
    desk = closing()
    sent = len(desk.sent)
    desk.session.command('/auto off')
    assert desk.said[-1] == 'auto-answer off from the next return to the calling frequency'
    assert not desk.session.auto and desk.session._closing and desk.sent[sent:] == []
    desk.session.command('/away gone')
    assert desk.session.away == 'gone' and desk.session.auto and desk.session._closing
    assert 'LISTEN CQ' not in desk.sent[sent:]
    desk.session.quit()


@pytest.mark.parametrize('line', ['/clear', 'chat text', '/bye again', '/beacon 10', '/cq'])
def test_link_and_transmit_commands_still_wait_for_the_close(line):
    desk = closing()
    texts = desk.texts[:]
    desk.session.command(line)
    assert desk.said[-1] == 'closing; /stop aborts without waiting for delivery'
    assert desk.texts == texts and 'CLEANTXBUFFER' not in desk.sent
    desk.session.quit()


def test_heard_while_composing_keeps_the_vmail_draft(tmp_path):
    desk = Desk(mailbox=tmp_path)
    desk.flow.dial = v.slot_frequency(20, 3)
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    desk.session.command('/vmail send Hi')
    desk.session.command('line one')
    desk.session.command('/heard')
    assert desk.session.compose == ('Hi', ['line one'])
    assert 'VMail dropped' not in desk.said


def test_heard_while_the_remote_bbs_owns_the_conversation():
    desk = Desk()
    desk.hear('CQFRAME K7QB-2 500')
    desk.session.bbs.sid = 'test'
    desk.session.command('/heard')
    assert desk.said[-1].startswith('  1  K7QB')
    desk.session.command('hello')
    assert desk.said[-1].startswith('Remote BBS owns this conversation')


def test_status_says_when_auto_answer_is_off():
    desk = Desk()
    desk.session.command('/auto off')
    desk.said.clear()
    desk.session.command('/status')
    assert desk.said[1] == 'Operation: not linked; monitoring; auto-answer off (/auto on to answer)'
    desk.session.command('/auto on')
    desk.session.command('/status')
    assert 'Operation: not linked; monitoring; auto-answer on' in desk.said


def test_a_second_call_names_the_linked_peer():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.do('/call K7QB 2')
    desk.do('/call N0XYZ 3')
    assert desk.said[-1] == 'error: linked with K7QB; /bye first'


def test_email_composes_during_an_exchange_and_waits_to_send():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    desk.flow.dial = 14_105_750
    desk.receive('<S:VE>')
    desk.session.steps = iter([.1] * 100)
    desk.session.command('/email send a@example.org b@example.org Test')
    assert desk.session.email.draft is not None
    assert desk.said[-1].startswith('Email via N0XYZ')


def test_calling_our_own_callsign_warns_and_proceeds():
    desk = Desk(**{'CONNECT W9SSJ W9SSJ': ['OK', 'CONNECTED W9SSJ W9SSJ 500']})
    desk.do('/call W9SSJ 2')
    assert "warning: W9SSJ is this station's own callsign; calling it anyway" in desk.said
    assert 'CONNECT W9SSJ W9SSJ' in desk.sent


def test_chat_tx_events_are_queued_until_the_modem_acknowledges():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    desk.session.command('hello')
    assert desk.kinds('tx')[-1] == dict(kind='tx', text='hello\n', status='queued')
    desk.session.diagnostic('acknowledged', bytes=6)
    assert desk.kinds('tx')[-1] == dict(kind='tx', status='acknowledged', bytes=6)


# -- away readiness and BBS listing from the calling frequency -------------------------

def test_away_answers_on_the_calling_frequency_with_awq_and_greets_after_the_qsy():
    desk = Desk()
    desk.session.command('/away Gone to the shed')
    desk.hear('SN -5.0', 'CONNECTED KB6CIO W9SSJ 500')
    assert desk.texts == ['5 <AWQ>']
    assert desk.legacy_said[-1] == 'away: <AWQ> sent to KB6CIO; the away message follows the QSY'
    desk.receive('<QSYF>000014107250</QSYF>')
    assert desk.tunes == [14_107_250] and desk.texts == ['5 <AWQ>']
    desk.receive('<TL>')
    desk.receive('de KB6CIO <R-05>')
    assert desk.session.steps is None and desk.flow.dial == 14_107_250
    assert desk.texts[-1] == '24 <AWAY>\r\nGone to the shed'
    assert desk.texts.count('24 <AWAY>\r\nGone to the shed') == 1


def test_responder_bbs_list_on_the_calling_frequency_waits_for_the_callers_qsy():
    desk = Desk()
    desk.hear('SN -5.0', 'CONNECTED KB6CIO W9SSJ 500')
    desk.session.command('/bbs list')
    assert desk.texts == ['3 <Q>']
    assert desk.legacy_said[-1] == ('BBS list queued: it is requested once this link settles off the calling '
                             'frequency; KB6CIO called, so its station starts the QSY. /stop cancels it')
    desk.receive('<QSYF>000014107250</QSYF>')
    desk.receive('<TL>')
    desk.receive('de KB6CIO <R-05>')
    assert desk.texts[-1] == '5 <BLR>' and desk.texts.count('5 <BLR>') == 1
    desk.receive(LIST)
    assert any('hello.txt' in line for line in desk.said)


def test_caller_bbs_list_during_the_readiness_wait_runs_after_the_move():
    desk = caller(readiness=None)
    def reply(data):
        desk.host.lines('BUFFER 0')
        if data == v.frame_message('<TL>'):
            desk.host.data.incoming.append(v.frame_message('<QSYR>'))
            desk.host.data.incoming.append(v.frame_message('de KE8PLM <R-02>'))
    desk.host.data.respond = reply
    desk.session.command('/call KE8PLM')
    desk.tick()
    assert desk.flow.qsy_phase.startswith('waiting for peer auto-QSY readiness')
    desk.session.command('/bbs list')
    assert desk.said[-1].startswith('BBS list queued') and not desk.kinds('error')
    desk.receive('<Q>')
    desk.drive(until=lambda: '5 <BLR>' in desk.texts)
    assert desk.flow.dial != v.CALLING_HZ[20] and desk.texts[-1] == '5 <BLR>'


def test_stop_cancels_a_queued_bbs_list():
    desk = Desk()
    desk.hear('SN -5.0', 'CONNECTED KB6CIO W9SSJ 500')
    desk.session.command('/bbs list')
    desk.session.command('/stop')
    assert desk.legacy_said[-1] == 'queued BBS list cancelled' and desk.host.client.connected
    desk.receive('<QSYF>000014107250</QSYF>')
    desk.receive('<TL>')
    desk.receive('de KB6CIO <R-05>')
    assert '5 <BLR>' not in desk.texts


def test_old_direct_draft_preserves_original_time_and_cannot_follow_another_peer():
    from hfmodem.tests.contact.test_varac import answered
    d = answered()
    d.session.parking.store.clock = lambda: 1800000000
    d.session.command('/vmail send composed earlier')
    d.session.command('body')
    composed = d.session._compose_link[2]
    d.session.parking.store.clock = lambda: 1800000601
    d.session.command('.')
    assert f'<TME:{composed}>' in d.texts[-1]
    d.receive('<SMR>')
    d.session.command('/vmail send never transfer draft to next contact')
    d.hear('DISCONNECTED')
    d.hear('CONNECTED K1ABC W9SSJ 500')
    before = list(d.texts)
    d.session.command('.')
    assert d.texts == before
    assert 'earlier contact' in d.said[-1]


def test_tactical_identity_keeps_status_and_collection_available():
    desk = Desk()
    desk.host.client.mycall = 'X-W9SSJ'
    desk.session.command('/status')
    assert any('VMail parking on' in line and '0 pending' in line for line in desk.said)
    desk.session.command('/vmail pending')
    assert any('No parked-mail notices saved' in line for line in desk.said)
    assert not desk.kinds('error')
