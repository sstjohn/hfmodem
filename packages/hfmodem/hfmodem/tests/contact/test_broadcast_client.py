# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest
from hfmodem.ax25 import kiss
from hfmodem.contact.broadcast import decode_ui, encode_ui
from hfmodem.tests.contact.test_varac import Desk


def desk():
    return Desk(kiss=True, **{'BROADCAST CAPS': ['BROADCAST CAPS 0 1']})


def select(d):
    d.do('/all slot 11')
    assert d.flow.dial == 14105750
    assert d.session.broadcast.selected == d.flow.dial


def wire(d):
    return [x for x in d.sent if x.startswith('BROADCAST ') and x != 'BROADCAST CAPS']


def answer(d, *states):
    token = d.session.broadcast.token
    d.hear(*(f'BROADCAST {token} {state}' for state in states))


def test_explicit_selection_and_send_has_no_contact_bytes_or_delivery_claim():
    d = desk()
    d.do('/all hello')
    assert not wire(d)
    before = list(d.texts)
    select(d)
    d.session.command('/all hello everyone')
    d.tick()
    assert len(wire(d)) == 1
    message = decode_ui(bytes.fromhex(wire(d)[0].split()[3]), command=1)
    assert (message.source, message.destination, message.info) == ('W9SSJ', 'ALL', b'hello everyone')
    assert d.texts == before
    answer(d, 'QUEUED')
    assert d.session.broadcast.active
    assert any('not yet counted as transmitted' in line for line in d.said)
    d.hear('PTT ON')
    d.hear('PTT OFF')
    answer(d, 'SENT')
    assert not d.session.broadcast.active
    assert 'remote reception unconfirmed' in d.session.broadcast.result
    assert d.flow.dial == 14105750
    d.do('/all off')
    assert d.flow.dial == 14105000 and d.session.broadcast.selected is None


def test_unknown_host_never_gets_raw_kiss_fallback():
    d = Desk(**{'BROADCAST CAPS': ['WRONG']})
    select(d)
    d.do('/all hello')
    assert not wire(d) and not d.session.broadcast.active
    assert any('Host rejected broadcast' in line for line in d.said)


@pytest.mark.parametrize('channel', ['slot 11', 'cf'])
@pytest.mark.parametrize('guard', ['linked', 'pending', 'ptt', 'physical', 'busy', 'steps'])
def test_unsafe_state_refuses_without_sending(guard, channel):
    d = desk()
    d.do(f'/all {channel}')
    if guard == 'linked':
        d.host.client.connected = True
    elif guard == 'pending':
        d.host.client.pending = True
    elif guard == 'ptt':
        d.host.client.ptt = True
    elif guard == 'physical':
        d.line = True
    elif guard == 'busy':
        d.host.client.busy = True
    else:
        d.session.steps = iter([None])
    d.session.command('/all hello')
    assert not wire(d)


def test_stop_after_queue_settles_abort_and_preserves_beacon_schedule():
    d = desk()
    d.session.set_beacon(15)
    due, expiry = d.session._beacon_due, d.session._beacon_expires
    select(d)
    d.session.command('/all hello')
    d.tick()
    answer(d, 'QUEUED')
    d.session.command('/stop')
    d.drive()
    assert 'ABORT' in d.sent and not d.session.broadcast.active
    assert not d.session._closing
    assert (d.session._beacon_due, d.session._beacon_expires) == (due, expiry)
    assert d.flow.dial == 14105750
    assert len(wire(d)) == 1


def test_failed_and_unmatched_tokens_are_not_success():
    d = desk()
    select(d)
    d.session.command('/all hello')
    d.tick()
    d.hear('BROADCAST other SENT')
    assert d.session.broadcast.active
    answer(d, 'QUEUED', 'FAILED channel busy')
    assert d.session.broadcast.result == 'FAILED channel busy'
    assert not d.session.broadcast.active and len(wire(d)) == 1
    assert not any('Broadcast transmitted on' in x for x in d.said)


def test_broadcast_rx_is_separate_and_control_bytes_are_escaped():
    d = desk()
    before = list(d.texts)
    raw = encode_ui('K1ABC', 'ALL', b'hi\x1b[31m<DISC>', command=1)
    d.host.kiss.incoming.append(kiss.encode(raw, 1))
    d.tick()
    assert any('[broadcast ' in x and '\\x1b' in x for x in d.said)
    assert not any('\x1b' in x for x in d.said)
    assert d.texts == before and not d.kinds('rx') and not d.kinds('heard')
    assert d.kinds('broadcast_rx')[-1]['payload_hex'] == b'hi\x1b[31m<DISC>'.hex()


def test_quit_cancels_admitted_frame_before_finishing():
    d = desk()
    select(d)
    d.session.command('/all hello')
    d.tick()
    answer(d, 'QUEUED')
    d.session.quit()
    assert 'ABORT' in d.sent and not d.session.broadcast.active


def test_capability_timeout_does_not_queue_or_abort():
    d = Desk(**{'BROADCAST CAPS': []})
    select(d)
    before = d.sent.count('ABORT')
    d.do('/all hello')
    assert not wire(d) and d.sent.count('ABORT') == before
    assert not d.session.broadcast.active


def test_completion_timeout_aborts_without_false_success_or_retry():
    d = desk()
    select(d)
    d.session.command('/all hello')
    d.tick()
    answer(d, 'QUEUED')
    d.drive()
    assert 'ABORT' in d.sent
    assert 'timed out' in d.session.broadcast.result
    assert len(wire(d)) == 1 and d.flow.dial == 14105750
    assert not d.session._closing


def test_new_incoming_link_invalidates_broadcast_without_aborting_new_peer():
    d = desk()
    select(d)
    d.session.command('/all hello')
    d.tick()
    answer(d, 'QUEUED')
    before = d.sent.count('ABORT')
    d.hear('PENDING', 'CONNECTED W9SSJ K1ABC 500')
    assert d.host.client.connected and d.session.peer == 'K1ABC'
    assert not d.session.broadcast.active and d.session.broadcast.selected is None
    assert d.sent.count('ABORT') == before


def test_cancel_with_physical_high_fails_closed():
    from hfmodem.contact.varac import HostError
    d = desk()
    select(d)
    d.session.command('/all hello')
    d.tick()
    answer(d, 'QUEUED')
    d.line = True
    d.session.command('/stop')
    with pytest.raises(HostError, match='unconfirmed; terminal must end'):
        d.drive()
    assert d.session._closing and d.session._close_failed is not None


@pytest.mark.parametrize('refused', [False, True])
def test_operator_command_crosses_actual_host_queue_without_rf(monkeypatch, refused):
    import numpy as np
    from hfmodem.kestrel.tx import broadcast
    from hfmodem.tests.kestrel.test_cq import session_mock
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda _: np.zeros(8))
    server, replies, modem = session_mock()
    server._dispatch('MYCALL W9SSJ')
    modem.poll()
    sent = []
    modem._transport.tx = sent.append
    modem._transport.tx_went_out = lambda: not refused
    d = desk()
    d.host.command.respond = lambda raw: server._dispatch(raw.decode().rstrip('\r'))
    server._send_cmd = lambda line: d.host.lines(line)
    ready = d.host.client.ready
    def owner_ready(sockets, timeout):
        modem.poll()
        return ready(sockets, timeout)
    d.host.client.ready = owner_ready
    select(d)
    d.do('/all hello real host queue')
    assert len(sent) == 1 and len(wire(d)) == 1
    if refused:
        assert d.session.broadcast.result == 'FAILED transport refused'
    else:
        assert d.session.broadcast.result == 'transmitted; remote reception unconfirmed'
    assert not d.texts and not d.session.broadcast.active


def test_failed_completion_with_high_line_settles_once_then_fails_fatally():
    from hfmodem.contact.varac import HostError
    d = desk()
    select(d)
    d.session.command('/all hello')
    d.tick()
    answer(d, 'QUEUED')
    d.line = True
    start = d.host.now
    answer(d, 'FAILED unkey failed')
    with pytest.raises(HostError, match='unconfirmed; terminal must end'):
        d.drive()
    assert d.sent.count('ABORT') == 1
    assert d.host.now - start < 11
    assert d.session._close_failed is not None and d.session._closing


def test_explicit_calling_frequency_selection_sends_and_restores_monitoring():
    d = desk()
    d.session.set_beacon(15)
    schedule = d.session._beacon_due, d.session._beacon_expires
    d.do('/all cf')
    assert d.flow.dial == 14105000
    assert d.session.broadcast.selected == 14105000
    assert not wire(d) and not d.texts
    assert d.sent[-1] == 'LISTEN OFF'
    d.session.command('/all hello on CF')
    d.tick()
    assert len(wire(d)) == 1
    answer(d, 'QUEUED', 'SENT')
    assert not d.session.broadcast.active
    assert 'remote reception unconfirmed' in d.session.broadcast.result
    d.do('/all off')
    assert d.session.broadcast.selected is None and d.flow.dial == 14105000
    assert (d.session._beacon_due, d.session._beacon_expires) == schedule
