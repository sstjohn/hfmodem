# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Documented LHR request through the operator command; fake host only."""
import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac_bbs import linked


@pytest.mark.parametrize('command', ['/inquire heard', '/tag LHR'])
def test_one_request_no_automatic_retry_or_ack_success(command):
    desk = linked()
    desk.session.command(command)
    assert desk.texts == ['5 <LHR>']
    assert desk.session.steps is None
    assert any('request queued' in line and 'not a list response' in line for line in desk.said)
    desk.hear('BUFFER 0')
    desk.host.now += 120
    desk.tick()
    assert desk.texts == ['5 <LHR>']
    assert desk.kinds('inquiry')[-1]['outcome'] == 'queued'
    assert not any('list received' in line for line in desk.said)


@pytest.mark.parametrize('reply', ['K1ABC\r\nW9XYZ', '<LHE>', '<LHJ>'])
def test_reply_stays_contact_text_and_never_local_heard(reply):
    desk = linked()
    desk.session.command('/inquire heard')
    before = list(desk.session.heard)
    desk.receive(reply)
    assert desk.session.heard == before
    assert any(record['text'] == reply for record in desk.kinds('rx'))
    assert desk.texts == ['5 <LHR>']


@pytest.mark.parametrize('frequency', list(v.CALLING_HZ.values()))
def test_calling_frequencies_rejected(frequency):
    desk = linked()
    desk.flow.dial = frequency
    desk.session.command('/inquire heard')
    assert not desk.texts
    assert any('off the calling frequency' in line for line in desk.said)


@pytest.mark.parametrize('guard', ['disconnected', 'closed', 'opening', 'bbs', 'vmail', 'ptt', 'pending', 'buffer', 'unreported'])
def test_only_current_settled_link_may_request(guard):
    desk = linked()
    client, session = desk.host.client, desk.session
    if guard == 'disconnected':
        desk.hear('DISCONNECTED')
    elif guard == 'closed':
        client.closed = True
    elif guard == 'opening':
        session._opening = True
    elif guard == 'bbs':
        session._bbs_waiting = True
    elif guard == 'vmail':
        session.mail_ack.reserve('direct', 1, lambda ack, reason: None)
    elif guard == 'buffer':
        client.buffer = 5
    elif guard == 'unreported':
        client._sent_after = client.sequence
    else:
        setattr(client, guard, True)
    session.command('/inquire heard')
    assert not desk.texts
    assert not desk.kinds('inquiry')


def test_local_heard_and_bad_syntax_never_query_peer():
    desk = linked()
    for command in ('/heard', '/inquire', '/inquire heard extra', '/tag LHR bad'):
        desk.session.command(command)
    assert not desk.texts
    assert v.emit_tag('LHR') == '<LHR>'
