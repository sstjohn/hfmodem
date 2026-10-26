# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Close regressions from KE8PLM, 20261024T022751257601Z, host.jsonl.

At 02:47:21.478Z 73<DISC> queued BUFFER 10, never BUFFER 0; at
02:48:02.120Z /quit queued a second farewell, BUFFER 20. No DISCONNECT
was issued. Model the missing ACK, then vary the host's close response.
"""
import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk, Host


def stuck_desk(**replies):
    desk = Desk(**replies)
    desk.hear('CONNECTED W9SSJ KE8PLM 500')
    desk.host.data.respond = lambda data: desk.host.lines('BUFFER 10', 'PTT ON')
    return desk


@pytest.mark.parametrize('clean', ['CLEANTXBUFFER OK', 'CLEANTXBUFFER FAILED ', 'WRONG'])
def test_missing_farewell_ack_always_disconnects_within_15_seconds(clean):
    desk = stuck_desk(**{'CLEANTXBUFFER': [clean],
                         'DISCONNECT': ['OK', 'DISCONNECTED', 'PTT OFF']})
    start = desk.host.now
    assert desk.session.command('/bye')
    assert desk.host.now - start < .2 and desk.session.steps is not None
    desk.drive()
    assert desk.host.now - start < 16
    assert desk.sent.index('CLEANTXBUFFER') < desk.sent.index('DISCONNECT')
    assert not desk.host.client.connected and not desk.host.client.ptt
    assert desk.texts.count('8 73<DISC>') == 1


@pytest.mark.parametrize('late_quit', [False, True])
def test_quit_during_stuck_bye_never_queues_second_farewell_and_waits_for_unkey(late_quit):
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    desk.session.command('/bye')
    desk.tick()
    if late_quit:
        desk.host.now += 15
        desk.tick()
    assert desk.session.command('/bye again') and desk.said[-1].startswith('closing; /stop')
    assert desk.session.command('do not queue this') and desk.said[-1].startswith('closing; /stop')
    start = desk.host.now
    ready = desk.host.ready
    def unkey_after_abort(sockets, timeout):
        if 'ABORT' in desk.sent and desk.host.now - start >= 1:
            desk.host.lines('PTT OFF')
        return ready(sockets, timeout)
    desk.host.client.ready = unkey_after_abort
    desk.session.quit()
    assert desk.host.now - start < 2
    assert desk.texts.count('8 73<DISC>') == 1
    assert desk.sent.count('DISCONNECT') == desk.sent.count('ABORT') == 1
    assert not desk.host.client.ptt and desk.session.steps is None


def test_bye_cannot_restart_beacon_or_answer_qsy_while_closing():
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    desk.session.set_beacon(15)
    desk.session.command('/bye')
    texts = desk.texts[:]
    desk.receive('<Q:375>')
    desk.host.now += 16
    desk.tick()
    assert desk.texts == texts and not desk.tunes
    assert not any(s.startswith('CQFRAME') for s in desk.sent)
    desk.session.quit()


def test_quit_drops_a_parked_application_exchange_instead_of_finishing_it():
    desk = stuck_desk(**{'DISCONNECT': ['OK', 'DISCONNECTED', 'PTT OFF']})
    def application():
        yield
        desk.host.client.send_text('must not send after quit')
    desk.session._run(application())
    desk.session.quit()
    assert not any('must not send' in text for text in desk.texts)
    assert desk.sent[-4:] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert not desk.host.client.connected and not desk.host.client.ptt


def test_abort_without_ptt_off_does_not_report_success():
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    desk.session.command('/bye')
    desk.tick()
    with pytest.raises(TimeoutError, match='confirm ABORT and become unkeyed'):
        desk.session.quit()
    assert desk.sent[-1] == 'ABORT' and desk.host.client.ptt


def test_disconnect_before_ptt_off_still_waits_for_last_frame():
    desk = stuck_desk(**{'DISCONNECT': ['OK', 'DISCONNECTED']})
    desk.session.command('/bye')
    desk.tick()
    desk.host.now += 15
    desk.tick()
    assert desk.session.steps is not None and desk.host.client.ptt
    desk.hear('PTT OFF')
    assert desk.session.steps is None


def test_drain_names_the_missing_peer_ack_and_has_one_total_deadline():
    host = Host()
    host.client.connected = True
    host.client.send_text('hello')
    steps = host.client.drain_steps(timeout=1)
    next(steps)
    host.now = .9
    host.client._line('BUFFER 0')
    host.client._line('PTT ON')
    next(steps)
    host.now = 1.1
    with pytest.raises(TimeoutError, match='unkeyed'):
        next(steps)
    host.client.send_text('hello again')
    with pytest.raises(TimeoutError, match='Peer has not acknowledged'):
        host.client.drain(timeout=.1)


@pytest.mark.parametrize('missing', ['CLEANTXBUFFER', 'DISCONNECT', 'both'])
def test_lost_close_command_replies_cannot_delay_disconnect_or_abort(missing):
    replies = {'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']}
    if missing in ('CLEANTXBUFFER', 'both'):
        replies['CLEANTXBUFFER'] = []
    if missing in ('DISCONNECT', 'both'):
        replies['DISCONNECT'] = []
    desk = stuck_desk(**replies)
    desk.session.command('/bye')
    desk.tick()
    start = desk.host.now
    desk.session.quit()
    assert desk.sent[-4:] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert desk.host.now - start < 1
    assert not desk.host.client.ptt and not desk.host.client.connected


def test_missing_abort_ack_is_not_success_even_with_key_up():
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['PTT OFF']})
    desk.session.command('/bye')
    desk.tick()
    with pytest.raises(TimeoutError, match='confirm ABORT'):
        desk.session.quit()


@pytest.mark.parametrize('failed', ['CLEANTXBUFFER', 'DISCONNECT'])
def test_close_write_failure_still_attempts_remaining_stop_commands(failed):
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    original = desk.host.command.sendall
    attempts = []
    def send(data):
        command = data.decode().rstrip('\r')
        attempts.append(command)
        if command == failed:
            raise OSError('injected write failure')
        original(data)
    desk.host.command.sendall = send
    desk.session.quit()
    assert attempts == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert not desk.host.client.ptt


@pytest.mark.parametrize('state', ['PTT ON', 'PENDING'])
def test_quit_without_a_link_still_stops_unfinished_keying(state):
    desk = Desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    desk.hear(state)
    desk.session.quit()
    assert desk.sent[-4:] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert not desk.host.client.ptt and not desk.host.client.pending
    assert not desk.texts


@pytest.mark.parametrize('interrupt_at', ['farewell', 'clear'])
def test_quit_closes_even_when_interrupt_exhausted_the_farewell_generator(interrupt_at):
    desk = stuck_desk(**{'DISCONNECT': ['OK', 'DISCONNECTED', 'PTT OFF']})
    if interrupt_at == 'farewell':
        def interrupt(text):
            raise KeyboardInterrupt
        desk.host.client.send_text = interrupt
    else:
        desk.host.data.respond = lambda data: desk.host.lines('BUFFER 0')
        original = desk.host.client.send_command
        def interrupt(command):
            if command == 'CLEANTXBUFFER':
                desk.host.client.send_command = original
                raise KeyboardInterrupt
            return original(command)
        desk.host.client.send_command = interrupt
    with pytest.raises(KeyboardInterrupt):
        desk.session.command('/bye')
        desk.drive()
    desk.session.quit()
    assert desk.sent[-4:] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert not desk.host.client.connected and not desk.host.client.ptt


def test_failed_bye_escapes_tick_and_finally_quit_does_not_delay_teardown():
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    desk.session.command('/bye')
    with pytest.raises(TimeoutError, match='confirm ABORT'):
        desk.drive()
    assert desk.host.client.ptt and desk.session.steps is None
    commands, failed_at = desk.sent[:], desk.host.now
    with pytest.raises(TimeoutError, match='confirm ABORT'):
        desk.session.quit()
    assert desk.sent == commands and desk.host.now == failed_at


@pytest.mark.parametrize('fence,abort,match', [([], ['OK', 'PTT OFF'], 'confirm ABORT'),
                                             (['VERSION test-host'], ['WRONG'], 'rejected ABORT')])
def test_abort_requires_its_ordered_fence_and_a_non_rejected_ack(fence, abort, match):
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': abort})
    desk.host.replies['VERSION'] = fence
    desk.session.command('/bye')
    desk.tick()
    with pytest.raises((TimeoutError, v.HostError), match=match):
        desk.session.quit()
    assert desk.host.client.connected


@pytest.mark.parametrize('during_bye', [False, True])
def test_stop_aborts_stuck_link_without_drain_or_second_farewell(during_bye):
    desk = stuck_desk(**{'CLEANTXBUFFER': ['CLEANTXBUFFER FAILED '],
                         'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    if during_bye:
        desk.session.command('/bye')
        desk.tick()
    else:
        desk.host.client.send_text('stuck data')
        desk.tick()
    texts = desk.texts[:]
    start = desk.host.now
    desk.session.command('/stop')
    desk.tick()
    assert desk.host.now - start < 1
    assert desk.sent[-2:] == ['VERSION', 'ABORT']
    assert desk.texts == texts
    assert desk.session.steps is not None and desk.host.client.ptt
    desk.hear('PTT OFF')
    desk.drive()
    assert not desk.host.client.connected and not desk.host.client.ptt
    assert desk.host.client.buffer == 0
    assert desk.session.steps is None
    assert not desk.session._force_close


def test_stop_escalates_an_already_waiting_disconnect_without_new_deadline():
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    desk.session.command('/bye')
    desk.tick()
    desk.host.now += 15
    desk.tick()
    assert 'DISCONNECT' in desk.sent and 'ABORT' not in desk.sent
    texts = desk.texts[:]
    start = desk.host.now
    desk.session.command('/stop')
    desk.drive()
    assert desk.host.now - start < 1
    assert desk.sent.count('DISCONNECT') == desk.sent.count('ABORT') == 1
    assert desk.texts == texts
    assert desk.texts.count('8 73<DISC>') == 1
    assert not desk.host.client.connected and not desk.host.client.ptt


def test_help_explains_graceful_and_immediate_close_without_transmitting():
    desk = Desk()
    texts = desk.texts[:]
    desk.session.command('/help')
    assert any('/bye sends a graceful goodbye; /stop aborts' in line for line in desk.said)
    assert any('/quit closes and exits without waiting for delivery' in line for line in desk.said)
    assert desk.texts == texts
