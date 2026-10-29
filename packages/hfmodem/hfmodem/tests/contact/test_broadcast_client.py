# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest
from hfmodem.ax25 import kiss
from hfmodem.contact.broadcast import decode_ui, encode_ui
from hfmodem.contact.varac import State
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


def rig_line(d):
    """The bridge's Rig.line_state: no keyer, so no reading, until the session arms."""
    d.session.line_state = lambda: d.line if d.armed and d.armed[-1] else None


@pytest.mark.parametrize('channel', ['cf', 'slot 11'])
def test_selection_after_operator_close_arms_and_verifies_the_line(channel):
    d = desk()
    rig_line(d)
    d.hear('CONNECTED W9SSJ K7QB 500')
    d.do('/bye')
    d.do('/auto off')
    assert d.session.flow.state is State.MONITORING and not d.armed[-1]
    d.do(f'/all {channel}')
    assert d.session.broadcast.selected == d.flow.dial and d.armed[-1]
    d.session.command('/all hello')
    d.tick()
    assert len(wire(d)) == 1


def test_unreadable_line_after_arming_refuses_and_restores_the_arm():
    d = desk()
    d.hear('CONNECTED W9SSJ K7QB 500')
    d.do('/bye')
    d.do('/auto off')
    d.session.line_state = lambda: None
    d.do('/all cf')
    assert d.session.broadcast.selected is None and not d.armed[-1]
    assert any('keying line is unverified' in line for line in d.said)


@pytest.mark.parametrize('guard, reason', [
    ('linked', 'a link is up'), ('pending', 'connection pending'),
    ('ptt', 'PTT on'), ('physical', 'keying line is HIGH'), ('steps', 'in progress')])
def test_refusal_names_the_blocking_condition(guard, reason):
    d = desk()
    if guard == 'linked':
        d.host.client.connected = True
    elif guard == 'pending':
        d.host.client.pending = True
    elif guard == 'ptt':
        d.host.client.ptt = True
    elif guard == 'physical':
        d.line = True
    else:
        d.session.steps = iter([None])
    d.session.command('/all cf')
    assert d.session.broadcast.selected is None
    assert any(line.startswith('error: Broadcast refused: ') and reason in line for line in d.said)


def test_directed_notice_reuses_admission_completion_without_selection_or_retune():
    d = desk()
    d.hear('BUSY OFF')
    order = []
    before = list(d.tunes), list(d.texts)
    d.session.broadcast.directed('W9SSJ-1', before_admit=lambda: order.append('reserved'),
                                 sent=lambda: order.append('sent'))
    assert order == []
    d.tick()
    assert order == ['reserved']
    assert len(wire(d)) == 1
    assert bytes.fromhex(wire(d)[0].split()[3]) == encode_ui('W9SSJ', 'W9SSJ-1', b'<SMP>')
    assert d.session.broadcast.selected is None
    answer(d, 'QUEUED')
    assert order == ['reserved']
    answer(d, 'SENT')
    assert order == ['reserved', 'sent']
    assert (d.tunes, d.texts) == before


@pytest.mark.parametrize('invalid', ['eligibility', 'busy', 'unknown', 'ptt', 'high', 'unknown-line',
                                     'dial', 'generation', 'writer', 'buffer'])
def test_directed_notice_rechecks_before_admission(invalid):
    d = Desk(**{'BROADCAST CAPS': []})
    d.hear('BUSY OFF')
    allowed, reserved = [True], []
    d.session.broadcast.directed('K1ABC', valid=lambda: allowed[0],
                                 before_admit=lambda: reserved.append(True))
    if invalid == 'eligibility':
        allowed[0] = False
    elif invalid in ('busy', 'unknown'):
        d.host.client.busy = True if invalid == 'busy' else None
    elif invalid == 'ptt':
        d.host.client.ptt = True
    elif invalid in ('high', 'unknown-line'):
        d.line = True if invalid == 'high' else None
    elif invalid == 'dial':
        d.flow.dial += 750
    elif invalid == 'generation':
        d.host.client.link_generation += 1
    elif invalid == 'writer':
        d.host.client._writes[1] = (0, b'pending')
    else:
        d.host.client.buffer = 1
    d.hear('BROADCAST CAPS 0 1')
    assert not wire(d) and not reserved and not d.session.broadcast.active


def test_directed_notice_failed_cooldown_commit_never_submits():
    d = desk()
    d.hear('BUSY OFF')
    def fail():
        raise OSError('disk full')
    before = d.sent.count('ABORT')
    d.session.broadcast.directed('K1ABC', before_admit=fail)
    d.drive()
    assert not wire(d) and d.sent.count('ABORT') == before
    assert 'disk full' in d.session.broadcast.result


def test_directed_notice_cancel_after_admission_keeps_reservation_without_sent():
    d = desk()
    d.hear('BUSY OFF')
    events = []
    d.session.broadcast.directed('K1ABC', before_admit=lambda: events.append('reserved'),
                                 sent=lambda: events.append('sent'))
    d.tick()
    answer(d, 'QUEUED')
    d.session.broadcast.cancel()
    d.drive()
    assert events == ['reserved'] and len(wire(d)) == 1
    assert 'ABORT' in d.sent and not d.session.broadcast.active


def test_directed_notice_ambiguous_admission_retains_reservation():
    d = desk()
    d.hear('BUSY OFF')
    events = []
    d.session.broadcast.directed('K1ABC', before_admit=lambda: events.append('reserved'),
                                 sent=lambda: events.append('sent'))
    d.drive()
    assert events == ['reserved'] and len(wire(d)) == 1
    assert 'ABORT' in d.sent and 'timed out' in d.session.broadcast.result


def test_directed_notice_sent_does_not_commit_while_physical_line_is_high():
    from hfmodem.contact.varac import HostError
    d = desk()
    d.hear('BUSY OFF')
    sent = []
    d.session.broadcast.directed('K1ABC', sent=lambda: sent.append(True))
    d.tick()
    answer(d, 'QUEUED')
    d.line = True
    answer(d, 'SENT')
    with pytest.raises(HostError, match='unconfirmed; terminal must end'):
        d.drive()
    assert not sent


@pytest.mark.parametrize('change', ['high', 'unknown-line', 'busy', 'unknown-busy', 'dial',
                                   'generation', 'connected', 'pending', 'ptt', 'writer', 'buffer'])
def test_directed_notice_rechecks_state_after_durable_reservation(change):
    d = desk()
    d.hear('BUSY OFF')
    reserved = []
    def reserve():
        reserved.append(True)
        c = d.host.client
        if change in ('high', 'unknown-line'):
            d.line = True if change == 'high' else None
        elif change in ('busy', 'unknown-busy'):
            c.busy = True if change == 'busy' else None
        elif change == 'dial':
            d.flow.dial += 750
        elif change == 'generation':
            c.link_generation += 1
        elif change in ('connected', 'pending', 'ptt'):
            setattr(c, change, True)
        elif change == 'writer':
            c._writes[1] = (c.link_generation, b'pending')
        else:
            c.buffer = 1
    d.session.broadcast.directed('K1ABC', before_admit=reserve)
    d.drive()
    assert reserved == [True] and not wire(d)
    assert not d.session.broadcast.active


def test_directed_notice_restores_auto_off_arm_after_completion():
    d = desk()
    d.do('/auto off')
    rig_line(d)
    d.hear('BUSY OFF')
    assert not d.armed[-1]
    d.session.broadcast.directed('K1ABC')
    assert d.armed[-1]
    d.tick()
    answer(d, 'QUEUED', 'SENT')
    assert not d.armed[-1] and not d.session.broadcast.active


@pytest.mark.parametrize('change', ['connected', 'pending'])
def test_directed_notice_does_not_disarm_new_contact(change):
    d = Desk(**{'BROADCAST CAPS': []})
    d.do('/auto off')
    rig_line(d)
    d.hear('BUSY OFF')
    d.session.broadcast.directed('K1ABC')
    assert d.armed[-1]
    setattr(d.host.client, change, True)
    d.tick()
    assert d.armed[-1] and not wire(d)


def test_directed_notice_preflight_failure_restores_auto_off_arm():
    from hfmodem.contact.varac import HostError
    d = desk()
    d.do('/auto off')
    rig_line(d)
    d.host.client.busy = None
    with pytest.raises(HostError, match='deferred'):
        d.session.broadcast.directed('K1ABC')
    assert not d.armed[-1] and not wire(d)
