# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hfmodem.tests.contact.test_varac import Desk


def test_service_offer_is_explained_without_sending_bbs_or_claiming_readiness():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    before = list(desk.texts)
    greeting = 'Howdy from Colorful Colorado!! de KF0RKC <R-01> <S:VAB>'
    desk.receive(greeting)
    assert desk.kinds('rx')[-1]['text'] == greeting
    assert any('advertises: VMail, AI, BBS' in x for x in desk.said)
    assert any('availability not yet checked' in x for x in desk.said)
    assert any('does not announce auto-QSY readiness' in x for x in desk.said)
    assert desk.texts == before
    n = sum('advertises:' in x for x in desk.said)
    desk.receive('<S:VAB>')
    assert sum('advertises:' in x for x in desk.said) == n


def test_help_is_local_and_explains_text_and_bbs_limits():
    desk = Desk()
    before = list(desk.texts)
    desk.do('/help')
    assert any('other text is sent as chat' in x for x in desk.said)
    assert any('/bbs get N downloads a file (32 KiB compressed limit)' in x for x in desk.said)
    desk.do('/help bbs')
    assert desk.texts == before


def test_bbs_wait_diagnostic_does_not_send_or_offer_a_readiness_bypass():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.flow.qsy_phase = 'waiting for peer auto-QSY readiness (<AWQ> or <Q>)'
    desk.session.steps = iter([None])
    before = list(desk.texts)
    desk.session.command('/bbs list')
    assert 'peer has not announced auto-QSY readiness' in desk.said[-1]
    assert '/stop' in desk.said[-1] and '/qsy' not in desk.said[-1]
    assert desk.texts == before


def test_service_hint_off_cf_and_per_contact():
    desk = Desk()
    desk.flow.dial = 14106500
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.receive('<S:B>')
    assert not any('Still on the calling frequency' in x for x in desk.said)
    desk.hear('DISCONNECTED')
    desk.hear('CONNECTED W9SSJ W1AW 500')
    desk.receive('<S:B>')
    assert sum('advertises:' in x for x in desk.said) == 2


def status_lines(desk):
    start = len(desk.said)
    desk.session.command('/status')
    return desk.said[start:]


def test_actual_loop_kf0rkc_wait_cancel_and_ready_snapshot():
    desk = Desk(**{'CONNECT W9SSJ KF0RKC': ['OK']})
    desk.session.command('/call KF0RKC')
    desk.tick()
    assert status_lines(desk)[1:] == [
        'Operation: connecting: waiting for modem link establishment',
        'Data: no pending text reported (not proof the peer read it)',
        'No input needed; /stop cancels the attempt.']
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.receive('Howdy from Colorful Colorado!! de KF0RKC <R-01> <S:VAB>')
    lines = status_lines(desk)
    assert '14105.000 kHz (calling frequency)' in lines[0]
    assert 'modem link established' in lines[0]
    assert 'waiting for peer auto-QSY readiness' in lines[1]
    assert 'No input needed' in lines[3]
    assert not any('<Q:' in t or '<BLR>' in t for t in desk.texts)
    before = len(desk.said)
    for _ in range(20):
        desk.tick()
    assert desk.said[before:] == []
    desk.session.command('/stop')
    desk.tick()
    assert status_lines(desk)[1] == (
        'Operation: link established on the calling frequency; no exchange in progress')
    assert any('Ready for optional chat' in x for x in desk.said)


def test_status_tracks_buffer_and_frame_without_keepalive_notice_spam():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.host.data.respond = lambda data: None
    desk.session.command('hello')
    desk.tick()
    assert 'fresh modem buffer report' in status_lines(desk)[2]
    desk.hear('BUFFER 9', 'PTT ON')
    assert status_lines(desk)[2] == (
        'Data: transmitting a modem frame; 9 bytes queued or awaiting modem acknowledgement')
    desk.hear('PTT OFF')
    assert '9 bytes queued or awaiting modem acknowledgement' in status_lines(desk)[2]
    desk.hear('BUFFER 0')
    assert 'no pending text reported' in status_lines(desk)[2]
    start = len(desk.said)
    desk.hear('PTT ON')
    desk.hear('PTT OFF')
    assert desk.said[start:] == []


def test_status_bbs_wait_cancel_actual_dial_and_closing_composition():
    desk = Desk(**{'DISCONNECT': ['OK']})
    desk.flow.dial = 14106500
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.session.command('/bbs list')
    desk.tick()
    lines = status_lines(desk)
    assert '14106.500 kHz (slot 12)' in lines[0]
    assert lines[1] == 'Operation: waiting for the peer BBS directory response'
    desk.session.command('/stop')
    desk.tick()
    assert 'no exchange in progress' in status_lines(desk)[1]
    desk.session.compose = ('subject', ['unsent body'])
    before = desk.session.compose
    status_lines(desk)
    assert desk.session.compose == before
    desk.session.command('/bye')
    assert status_lines(desk)[1].startswith('Operation: closing:')


def test_early_services_replayed_only_for_current_link_and_unknown_ignored():
    desk = Desk(**{'CONNECT W9SSJ KF0RKC': ['OK']})
    desk.session.command('/call KF0RKC cf')
    desk.receive('<S:VBZ>')
    assert not any('advertises:' in line for line in desk.said)
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    assert any('KF0RKC advertises: VMail, BBS' in line for line in desk.said)
    assert not any('Z' in line for line in desk.said if 'advertises:' in line)
    desk.hear('DISCONNECTED')
    desk.hear('CONNECTED W9SSJ W1AW 500')
    assert not any('W1AW advertises:' in line for line in desk.said)


def test_normal_data_notices_snapshot_ignores_byte_count_changes_and_empty_ptt():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.host.data.respond = lambda data: None
    start = len(desk.said)
    desk.session.command('hello')
    desk.tick()
    desk.hear('BUFFER 9', 'PTT ON')
    desk.hear('BUFFER 7')
    desk.hear('PTT OFF')
    desk.hear('BUFFER 0')
    desk.hear('PTT ON')
    desk.hear('PTT OFF')
    assert [x for x in desk.said[start:] if x.startswith('Data: ')] == [
        'Data: text queued locally; waiting for a fresh modem buffer report',
        'Data: transmitting a modem frame; 9 bytes queued or awaiting modem acknowledgement',
        'Data: 7 bytes queued or awaiting modem acknowledgement',
        'Data: no pending text reported (not proof the peer read it)']


def test_vmail_input_and_application_ack_are_distinct_from_empty_buffer():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.session.command('/vmail send greetings')
    desk.session.command('hello')
    assert 'waiting for your input' in status_lines(desk)[1]
    before = desk.session.compose
    desk.session.command('/help')
    assert desk.session.compose == before
    desk.session.command('.')
    desk.tick()
    desk.hear('BUFFER 0')
    assert 'peer VMail acknowledgement' in status_lines(desk)[1]
    desk.receive('<SMR>')
    assert 'no exchange in progress' in status_lines(desk)[1]


def test_status_tracks_real_qsy_delivery_acceptance_and_opening_waits():
    from hfmodem.tests.contact.test_varac_cf_qsy import caller
    from hfmodem.contact import varac as v
    for phase in ('delivery', 'acceptance', 'opening'):
        desk = caller()
        if phase == 'delivery':
            desk.host.data.respond = lambda data: None
        elif phase == 'opening':
            def connected():
                desk.host.data.incoming.append(v.frame_message('<AWQ>'))
                return ['OK', 'CONNECTED W9SSJ KE8PLM 500']
            desk.host.replies['CONNECT W9SSJ KE8PLM'] = connected
            def reply(data):
                desk.host.lines('BUFFER 0')
                if data == v.frame_message('<Q>'):
                    desk.host.data.incoming.append(v.frame_message('<AWQ>'))
                if data == v.frame_message('<TL>'):
                    desk.host.data.incoming.append(v.frame_message('<QSYR>'))
            desk.host.data.respond = reply
        desk.session.command('/call KE8PLM')
        for _ in range(10):
            desk.tick()
        line = status_lines(desk)[1]
        assert {'delivery': 'invitation delivery', 'acceptance': 'peer acceptance',
                'opening': 'opening signal report'}[phase] in line
        assert any('Operation: QSY' in x for x in desk.said)


def test_new_contact_does_not_inherit_old_vmail_ack_wait():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.session.command('/vmail send greeting')
    desk.session.command('hello')
    desk.session.command('.')
    desk.hear('DISCONNECTED')
    desk.hear('CONNECTED W9SSJ W1AW 500')
    assert 'no exchange in progress' in status_lines(desk)[1]


def test_fresh_buffer_evidence_survives_host_history_rotation():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KF0RKC 500')
    desk.host.data.respond = lambda data: None
    desk.session.command('hello')
    desk.hear('BUFFER 0')
    for _ in range(4100):
        desk.host.client._line('IAMALIVE')
    assert 'no pending text reported' in status_lines(desk)[2]
