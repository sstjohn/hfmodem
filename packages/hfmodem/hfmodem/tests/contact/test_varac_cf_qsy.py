# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Calling-frequency opening: KE8PLM postmortem and P1 R08 Q/TL/QSYR."""
import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk


def caller(*, readiness="<AWQ>"):
    desk = Desk(loc='EN63bb', **{
        'CONNECT W9SSJ KE8PLM': ['OK', 'CONNECTED W9SSJ KE8PLM 500', 'SN -2.4']})
    if readiness is not None:
        def connected():
            desk.host.data.incoming.append(v.frame_message(readiness))
            return ['OK', 'CONNECTED W9SSJ KE8PLM 500', 'SN -2.4']
        desk.host.replies['CONNECT W9SSJ KE8PLM'] = connected
    return desk


def test_default_cf_call_invites_free_slot_then_opens_only_after_qsyr():
    desk = caller()
    desk.hear('CQFRAME K7QB-1 500')
    sent_on = []
    def reply(data):
        sent_on.append((desk.flow.dial, data))
        desk.host.lines('BUFFER 0')
        if data == v.frame_message('<TL>'):
            desk.host.data.incoming.append(v.frame_message('<QSYR>'))
    desk.host.data.respond = reply
    desk.do('/call KE8PLM')
    target = v.slot_frequency(20, 2)
    assert sent_on == [
        (14_105_000, v.frame_message('<QSYF>000014103500</QSYF>')),
        (target, v.frame_message('<TL>')),
        (target, v.frame_message('de W9SSJ <R-02>')),
        (target, v.frame_message('<LOC:EN63bb>')),
    ]
    assert desk.tunes == [target] and desk.session.slot == 2
    assert not desk.kinds('error')


def test_explicit_cf_call_keeps_plain_cf_chat():
    desk = caller()
    desk.do('/call KE8PLM cf')
    assert desk.texts == [] and desk.tunes == []
    assert desk.session.steps is None
    desk.do('hello')
    assert desk.texts[-1] == '6 hello\n'


@pytest.mark.parametrize('stalled', ['delivery', 'qsyr', 'snr'])
def test_auto_qsy_yields_to_operator_at_every_peer_wait(stalled):
    desk = caller()
    if stalled == 'delivery':
        desk.host.data.respond = lambda data: None
    elif stalled == 'snr':
        desk.host.replies['CONNECT W9SSJ KE8PLM'] = lambda: (
            desk.host.data.incoming.append(v.frame_message('<AWQ>')) or
            ['OK', 'CONNECTED W9SSJ KE8PLM 500'])
        def reply(data):
            desk.host.lines('BUFFER 0')
            if data == v.frame_message('<TL>'):
                desk.host.data.incoming.append(v.frame_message('<QSYR>'))
        desk.host.data.respond = reply
    began = desk.host.now
    desk.session.command('/call KE8PLM')
    for _ in range(10):
        desk.session.tick()
    assert desk.host.now - began < 3
    assert desk.session.steps is not None
    assert desk.session.command('/quit') is False
    assert not any('de W9SSJ' in text for text in desk.texts)
    assert desk.tunes == ([] if stalled == 'delivery' else [v.slot_frequency(20, 1)])
    before, tunes, start = len(desk.texts), desk.tunes[:], desk.host.now
    desk.session.quit()
    assert desk.texts[before:] == []
    assert desk.tunes == tunes
    assert desk.session.steps is None and not desk.host.client.connected
    assert desk.host.now - start < 1


def test_manual_qsy_is_stepped_and_cannot_replace_an_active_exchange():
    desk = caller()
    desk.do('/call KE8PLM cf')
    desk.host.data.respond = lambda data: None
    began = desk.host.now
    desk.session.command('/qsy 15')
    assert desk.host.now - began < 1 and desk.session.steps is not None
    desk.tick()
    desk.tick()
    desk.session.command('/qsy 14')
    assert 'QSY to slot 15 already in progress: waiting for invitation delivery' in desk.said[-1]
    assert desk.texts == ['25 <QSYF>000014108750</QSYF>']
    assert desk.session.command('/quit') is False


@pytest.mark.parametrize('ready', ['<AWQ>', '<Q>', 'hello <Q> <S:VEB>', '<AWQ>\r\n<VW>'])
def test_no_invitation_or_move_until_current_peer_readiness(ready):
    desk = caller(readiness=None)
    desk.session.command('/call KE8PLM')
    for _ in range(5): desk.tick()
    assert desk.texts == [] and not desk.tunes
    assert 'readiness' in desk.flow.qsy_phase
    desk.host.data.incoming.append(v.frame_message(ready))
    desk.tick()
    desk.tick()
    assert any('<QSYF>' in x for x in desk.texts)


def test_readiness_timeout_preserves_cf_link_without_invitation():
    desk = caller(readiness=None)
    desk.session.command('/call KE8PLM')
    desk.host.now += 91
    desk.tick()
    assert desk.session.steps is None and desk.host.client.connected
    assert desk.texts == [] and not desk.tunes
    assert desk.flow.dial == v.CALLING_HZ[20]
    assert any('did not announce auto-QSY readiness' in x for x in desk.said)


@pytest.mark.parametrize('action', ['stop', 'bye', 'quit', 'disconnect'])
def test_readiness_wait_cancels_without_q_or_tune(action):
    desk = caller(readiness=None)
    desk.session.command('/call KE8PLM')
    if action == 'disconnect':
        desk.hear('DISCONNECTED')
        desk.tick()
    elif action == 'quit':
        desk.session.quit()
    else:
        desk.session.command('/' + action)
        if action == 'bye': desk.drive()
    assert not any('<QSYF>' in x for x in desk.texts) and not desk.tunes
    assert desk.flow.qsy_phase is None
    if action == 'stop':
        assert desk.host.client.connected and desk.session.steps is None


def test_prior_link_readiness_cannot_authorize_new_call():
    desk = caller(readiness=None)
    desk.host.client._text(v.frame_message('<AWQ>'))
    desk.session.command('/call KE8PLM')
    for _ in range(5): desk.tick()
    assert desk.texts == [] and not desk.tunes


def test_readiness_after_connect_request_but_before_connected_is_retained():
    desk = caller(readiness=None)
    def connected():
        desk.host.client._text(v.frame_message('<Q>'))
        return ['OK', 'PENDING', 'CONNECTED W9SSJ KE8PLM 500', 'SN -2.4']
    desk.host.replies['CONNECT W9SSJ KE8PLM'] = connected
    desk.session.command('/call KE8PLM')
    for _ in range(5): desk.tick()
    assert any('<QSYF>' in x for x in desk.texts)


def test_manual_cf_qsy_waits_and_coalesced_acceptance_opens():
    desk = caller(readiness=None)
    desk.do('/call KE8PLM cf')
    desk.session.command('/qsy 15')
    assert desk.texts == [] and not desk.tunes
    def reply(data):
        desk.host.lines('BUFFER 0')
        if data == v.frame_message('<TL>'):
            desk.host.data.incoming.append(v.frame_message('<QSYR> welcome'))
    desk.host.data.respond = reply
    desk.host.data.incoming.append(v.frame_message('<Q>'))
    desk.drive()
    assert desk.flow.dial == v.slot_frequency(20, 15)
    assert desk.session.steps is None and any('de W9SSJ' in x for x in desk.texts)


def test_inbound_pending_discards_old_readiness_but_preserves_new_before_connected():
    desk = caller(readiness=None)
    c = desk.host.client
    c._text(v.frame_message('<AWQ>'))
    c._line('PENDING')
    before = c.link_message_start
    c._text(v.frame_message('<Q>'))
    c._line('CONNECTED W9SSJ KE8PLM 500')
    assert c.link_message_start == before
    desk.session.command('/qsy 15')
    desk.tick()
    desk.tick()
    assert any('<QSYF>' in x for x in desk.texts)


def test_disconnected_peer_cannot_satisfy_wait_from_cached_readiness():
    desk = caller(readiness=None)
    desk.session.command('/call KE8PLM')
    c = desk.host.client
    c._text(v.frame_message('<AWQ>'))
    c.connected = False  # emulate link loss before the generator next advances
    desk.tick()
    assert not any('<QSYF>' in x for x in desk.texts) and not desk.tunes


def test_confirmed_abort_retires_complete_and_partial_capability():
    desk = caller(readiness=None)
    c = desk.host.client
    c.connected = True
    c._text(v.frame_message('<AWQ>'))
    c._text(b'5 <AW')
    desk.host.replies.update({'DISCONNECT': [], 'VERSION': ['VERSION test-host'], 'ABORT': ['OK']})
    c.finish(desk.flow.close_steps(timeout=.2))
    assert not c.connected and not c.messages and not c._messages.pending
    c._line('PENDING')
    c._line('CONNECTED W9SSJ WR9R 500')
    desk.session.command('/qsy 15')
    assert not any('<QSYF>' in x for x in desk.texts) and not desk.tunes


def test_duplicate_pending_preserves_current_partial_and_complete_readiness():
    desk = caller(readiness=None)
    c = desk.host.client
    c._line('PENDING')
    c._text(b'3 <Q')
    c._line('PENDING')
    c._text(b'>')
    c._line('PENDING')
    c._line('CONNECTED W9SSJ WR9R 500')
    desk.session.command('/qsy 15')
    desk.tick()
    desk.tick()
    assert any('<QSYF>' in x for x in desk.texts)


@pytest.mark.parametrize('first_end', ['stop', 'timeout'])
def test_readiness_wait_can_be_retried_then_closed(first_end):
    desk = caller(readiness=None)
    desk.session.command('/call KE8PLM')
    if first_end == 'stop':
        desk.session.command('/stop')
    else:
        desk.host.now += 91
        desk.tick()
    assert desk.host.client.connected and desk.session.steps is None
    desk.host.data.incoming.append(v.frame_message('<Q>'))
    desk.tick()
    desk.session.command('/qsy 15')
    desk.tick()
    desk.tick()
    assert any('<QSYF>' in x for x in desk.texts)
    desk.session.command('/bye')
    desk.drive()
    assert not desk.host.client.connected


@pytest.mark.parametrize('change', ['new_link', 'frequency'])
def test_suspended_readiness_wait_cannot_act_on_changed_context(change):
    desk = caller(readiness=None)
    desk.session.command('/call KE8PLM')
    c = desk.host.client
    if change == 'new_link':
        c._line('DISCONNECTED')
        c._line('PENDING')
        c._line('CONNECTED W9SSJ WR9R 500')
    else:
        desk.flow.dial = v.slot_frequency(20, 2)
    c._text(v.frame_message('<AWQ>'))
    desk.tick()
    assert not any('<QSYF>' in x for x in desk.texts) and not desk.tunes
    assert desk.session.steps is None


def test_rejected_outbound_attempt_does_not_keep_inbound_boundary_open():
    desk = caller(readiness=None)
    desk.host.replies['CONNECT W9SSJ KE8PLM'] = ['WRONG']
    desk.session.command('/call KE8PLM')
    c = desk.host.client
    assert not c._outbound_connect_pending
    c._text(v.frame_message('<AWQ>'))
    c._line('PENDING')
    c._line('CONNECTED W9SSJ WR9R 500')
    desk.session.command('/qsy 15')
    assert not any('<QSYF>' in x for x in desk.texts)
