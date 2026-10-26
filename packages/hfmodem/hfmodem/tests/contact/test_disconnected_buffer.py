# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk, Host


def test_disconnected_retires_stale_queue_and_idle_bye_does_not_strand_broadcast():
    d = Desk(**{'BROADCAST CAPS': ['BROADCAST CAPS 0 1']})
    d.session.set_beacon(15)
    schedule = d.session._beacon_due, d.session._beacon_expires
    d.hear('CONNECTED W9SSJ K6HN 500', 'BUFFER 26')
    d.hear('DISCONNECTED')
    assert d.host.client.buffer == 0
    assert d.flow.state == v.State.MONITORING
    sent, texts = list(d.sent), list(d.texts)
    d.do('/bye')
    d.do('/bye')
    assert not d.session._closing and d.session.steps is None
    assert d.sent == sent and d.texts == texts
    assert (d.session._beacon_due, d.session._beacon_expires) == schedule
    d.do('/all slot 1')
    assert d.session.broadcast.selected == 14_104_250
    d.session.command('/all hello')
    d.tick()
    assert any(line.startswith('BROADCAST ') and line != 'BROADCAST CAPS' for line in d.sent)
    assert not any('acknowledged by peer' in line for line in d.said)


@pytest.mark.parametrize('reply', ['CLEANTXBUFFER OK', 'CLEANTXBUFFER BUFFEREMPTY'])
def test_clear_reply_updates_queue_without_inventing_delivery(reply):
    h = Host()
    h.client.connected = True
    h.client.send_text('pending')
    h.client._line('BUFFER 26')
    h.client._line(reply)
    assert h.client.buffer == 0
    assert h.client.buffer_sequence == h.client.sequence
    with pytest.raises(v.HostError, match='cleared without delivery'):
        h.client.drain(timeout=.1)


def test_failed_clear_does_not_retire_queue():
    h = Host()
    h.client._line('BUFFER 26')
    h.client._line('CLEANTXBUFFER FAILED ')
    assert h.client.buffer == 26


def test_disconnected_is_not_successful_drain():
    h = Host()
    h.client.connected = True
    h.client.send_text('pending')
    h.client._line('BUFFER 26')
    h.client._line('DISCONNECTED')
    with pytest.raises(v.HostError, match='Link ended'):
        h.client.drain(timeout=.1)


@pytest.mark.parametrize('selected', [False, True])
def test_bye_from_monitoring_waits_for_unkey_then_resumes_same_beacon_schedule(selected):
    d = Desk(**{'DISCONNECT': ['OK', 'DISCONNECTED']})
    d.session.set_beacon(15)
    schedule = d.session._beacon_due, d.session._beacon_expires
    if selected:
        d.do('/all slot 1')
    d.hear('PTT ON')
    d.session.command('/bye')
    d.tick()
    assert d.session._closing and d.session.steps is not None
    assert not d.texts
    d.hear('PTT OFF')
    assert d.session.steps is None and not d.session._closing
    assert d.flow.state == v.State.MONITORING
    assert (d.session._beacon_due, d.session._beacon_expires) == schedule


def test_idle_bye_does_not_bypass_physical_high():
    d = Desk()
    d.line = True
    d.session.command('/bye')
    assert 'keying line remains HIGH' in str(d.session._close_failed)
    assert d.session._closing and d.session._close_failed is not None
    assert not d.texts


def test_confirmed_abort_retires_queue_without_delivery_ack():
    d = Desk(**{'CLEANTXBUFFER': ['CLEANTXBUFFER FAILED '],
                'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    d.hear('CONNECTED W9SSJ K6HN 500', 'BUFFER 26')
    after = d.host.client.sequence
    d.host.client.finish(d.flow.close_steps(timeout=.2))
    assert d.host.client.buffer == 0
    assert not d.host.client.connected
    assert 'BUFFER 0' not in d.host.client.since(after)


@pytest.mark.parametrize('reply', ['CLEANTXBUFFER OK', 'CLEANTXBUFFER BUFFEREMPTY'])
def test_clear_cannot_revalidate_an_old_empty_buffer_report(reply):
    h = Host()
    h.client.connected = True
    h.client.send_text('pending')
    h.client._line('BUFFER 0')
    h.client._line('BUFFER 9')
    h.client._line(reply)
    with pytest.raises(v.HostError, match='cleared without delivery'):
        h.client.drain(timeout=.1)
    h.client.send_text('new text')
    h.client._line('BUFFER 0')
    h.client.drain(timeout=.1)
