# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A manual QSY must drain pending local chat before sending its invitation."""
import pytest
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac_cf_qsy import caller


def test_pending_chat_and_invitation_stay_in_separate_modem_deliveries():
    d = caller()
    pending, deliveries = [], []
    d.host.data.respond = pending.append
    d.do('/call KE8PLM cf')
    d.host.client.send_text('hello')
    d.session.command('/qsy 1')
    for _ in range(8):
        d.tick()
    assert pending == [v.frame_message('hello')]
    assert not d.tunes
    assert d.flow.qsy_phase == 'waiting for local delivery'
    # BUFFER counts and partial progress cannot release the invitation.
    d.hear('BUFFER 5', 'BUFFER 2')
    for _ in range(5):
        d.tick()
    assert pending == [v.frame_message('hello')]
    deliveries.append(b''.join(pending)); pending.clear()
    d.hear('BUFFER 0')
    for _ in range(5):
        d.tick()
    assert pending == [v.frame_message('<QSYF>000014104250</QSYF>')]
    assert not d.tunes
    deliveries.append(b''.join(pending)); pending.clear()
    d.hear('BUFFER 0')
    for _ in range(5):
        d.tick()
    assert d.tunes == [v.slot_frequency(20, 1)]
    assert pending == [v.frame_message('<TL>')]
    assert deliveries == [b'5 hello', b'25 <QSYF>000014104250</QSYF>']
    assert not any('de W9SSJ' in text for text in d.texts)


@pytest.mark.parametrize('action', ['bye', 'disconnect', 'timeout'])
def test_pending_local_chat_cannot_send_invitation_after_link_or_wait_ends(action):
    d = caller()
    d.host.data.respond = lambda data: None
    d.do('/call KE8PLM cf')
    d.host.client.send_text('hello')
    d.session.command('/qsy 1')
    for _ in range(5):
        d.tick()
    if action == 'bye':
        d.session.command('/bye')
    elif action == 'disconnect':
        d.hear('DISCONNECTED')
    else:
        d.host.now += 91
    d.tick()
    assert not d.tunes
    assert not any('<QSYF>' in text or '<TL>' in text for text in d.texts)


@pytest.mark.parametrize('change', ['new-link', 'frequency', 'discard'])
def test_local_chat_drain_is_bound_to_original_link_and_delivery(change):
    d = caller()
    d.host.data.respond = lambda data: None
    d.do('/call KE8PLM cf')
    d.host.client.send_text('hello')
    d.session.command('/qsy 1')
    for _ in range(5):
        d.tick()
    if change == 'new-link':
        d.hear('DISCONNECTED', 'PENDING', 'CONNECTED W9SSJ N0XYZ 500')
    elif change == 'frequency':
        d.flow.dial = v.slot_frequency(20, 2)
    else:
        d.hear('CLEANTXBUFFER BUFFEREMPTY')
    for _ in range(5):
        d.tick()
    assert not any('<QSYF>' in text for text in d.texts)
    assert not d.tunes


def test_acceptance_before_invitation_cannot_complete_the_later_move():
    d = caller()
    d.host.data.respond = lambda data: None
    d.do('/call KE8PLM cf')
    d.host.client.send_text('hello')
    d.session.command('/qsy 1')
    for _ in range(5):
        d.tick()
    d.receive('<QSYR>')
    d.hear('BUFFER 0')
    for _ in range(5):
        d.tick()
    d.hear('BUFFER 0')
    for _ in range(5):
        d.tick()
    assert d.tunes == [v.slot_frequency(20, 1)]
    assert d.flow.qsy_phase == 'waiting for peer acceptance (<QSYR>)'
    assert not any('de W9SSJ' in text for text in d.texts)
