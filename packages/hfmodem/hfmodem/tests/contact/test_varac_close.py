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


@pytest.mark.parametrize('answer', [False, True])
@pytest.mark.parametrize('quiet', [8, 11])
def test_connect_timeout_preserves_final_listen_window_before_closing_id(answer, quiet):
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK']})
    keyed = []
    desk.session.keyer = lambda *args: keyed.append(desk.host.now)
    desk.session.command('/call K7QB cf')
    start = desk.host.now
    desk.host.now = start + 88 - quiet
    desk.hear('PTT ON', 'PTT OFF')
    previous_end = desk.host.now
    desk.host.now = previous_end + quiet - .1
    desk.hear('PTT ON')
    desk.host.now = start + 90
    desk.tick()
    assert 'ABORT' not in desk.sent
    desk.hear('PTT OFF')
    last_end = desk.host.now
    assert 'ABORT' not in desk.sent
    assert not keyed
    desk.host.now = last_end + 8 - .2
    desk.tick()
    assert 'ABORT' not in desk.sent
    assert not keyed
    if answer:
        desk.hear('SN 3.0', 'CONNECTED W9SSJ K7QB 500')
        desk.drive()
        assert desk.host.client.connected
        assert 'ABORT' not in desk.sent
        assert not keyed
    else:
        desk.drive()
        assert 'ABORT' in desk.sent
        assert len(keyed) == 1
        assert 8 <= keyed[0] - last_end <= 8.5


def test_connect_without_keying_still_times_out():
    host = Host()
    with pytest.raises(v.HostError, match='timed out'):
        host.client.finish(host.client.connect_steps('K7QB', timeout=.5))
    assert 'ABORT' in host.sent
    assert host.now < 1


@pytest.mark.parametrize('phase', [0, 8.3])
def test_connect_retries_after_deadline_cannot_renew_final_window(phase):
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK']})
    keyed, aborted = [], []
    desk.session.keyer = lambda *args: keyed.append(desk.host.now)
    def abort():
        aborted.append(desk.host.now)
        return ['OK']
    desk.host.replies['ABORT'] = abort
    desk.session.command('/call K7QB cf')
    start = desk.host.now
    events = iter((start + phase + retry * 9.8 + offset, line)
                  for retry in range(14)
                  for offset, line in [(0, 'PTT ON'), (2, 'PTT OFF')])
    at, line = next(events)
    while desk.session.steps is not None and desk.host.now < start + 120:
        if desk.host.now >= at:
            desk.host.lines(line)
            at, line = next(events)
        desk.tick()
    assert len(aborted) == 1
    assert start + 90 <= aborted[0] <= start + 100.2
    assert desk.session.steps is None
    assert len(keyed) == 1
    assert keyed[0] <= start + 102.2


def test_connect_long_idle_gap_cannot_inflate_final_window():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK']})
    keyed = []
    desk.session.keyer = lambda *args: keyed.append(desk.host.now)
    desk.session.command('/call K7QB cf')
    start = desk.host.now
    desk.hear('PTT ON')
    desk.host.now = start + 2
    desk.hear('PTT OFF')
    desk.host.now = start + 88
    desk.hear('PTT ON')
    desk.host.now = start + 90
    desk.hear('PTT OFF')
    last_end = desk.host.now
    desk.drive(until=lambda: desk.host.now >= last_end + 8.5)
    assert 'ABORT' in desk.sent
    assert desk.session.steps is None
    assert len(keyed) == 1
    assert 8 <= keyed[0] - last_end <= 8.5


def test_a_key_that_never_drops_cannot_hold_the_attempt_open():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK']})
    desk.session.command('/call K7QB cf')
    start = desk.host.now
    desk.host.now = start + 89
    desk.hear('PTT ON')
    desk.drive(until=lambda: 'ABORT' in desk.sent or desk.host.now > start + 200)
    assert 'ABORT' in desk.sent
    assert desk.host.now <= start + 90 + v.CONNECT_GRACE_S + .5


def test_operator_stop_during_connect_listen_window_is_immediate():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'PTT ON', 'PTT OFF']})
    desk.session.command('/call K7QB cf')
    start = desk.host.now
    desk.session.command('/stop')
    desk.drive()
    assert 'ABORT' in desk.sent
    assert desk.host.now - start < 1


def stuck_desk(bw=500, **replies):
    desk = Desk(bw=bw, **replies)
    desk.hear(f'CONNECTED W9SSJ KE8PLM {bw}')
    desk.host.data.respond = lambda data: desk.host.lines('BUFFER 10', 'PTT ON')
    return desk


@pytest.mark.parametrize('ack_at', [.5, 9, 18.447])
def test_healthy_bye_drains_before_disconnect_without_clearing_or_aborting(ack_at):
    desk = stuck_desk(**{'DISCONNECT': ['OK']})
    desk.host.client.send_text('queued chat')
    desk.session.command('/bye')
    before = desk.sent[:]
    desk.drive(until=lambda: desk.host.now >= ack_at)
    assert desk.sent == before
    desk.host.lines('BUFFER 0', 'PTT OFF')
    desk.drive(until=lambda: 'DISCONNECT' in desk.sent)
    assert desk.sent[len(before):] == ['LISTEN OFF', 'DISCONNECT']
    assert desk.texts[-2:] == ['11 queued chat', '8 73<DISC>']
    desk.hear('DISCONNECTED', 'PTT ON')
    assert desk.session.steps is not None
    desk.hear('PTT OFF')
    desk.drive()
    assert 'CLEANTXBUFFER' not in desk.sent and 'ABORT' not in desk.sent
    assert not desk.host.client.connected and not desk.host.client.ptt


def test_progress_extends_farewell_drain_and_close_until_graceful_completion():
    desk = stuck_desk(**{'DISCONNECT': ['OK']})
    start = desk.host.now
    desk.session.command('/bye')
    for at, remaining in [(8, 8), (16, 4), (24, 0)]:
        desk.drive(until=lambda: desk.host.now >= start + at)
        desk.hear(f'BUFFER {remaining}')
        assert 'DISCONNECT' not in desk.sent
    desk.hear('PTT OFF')
    desk.drive(until=lambda: 'DISCONNECT' in desk.sent)
    desk.drive(until=lambda: desk.host.now >= start + 32)
    desk.receive('73<DISC>')
    desk.drive(until=lambda: desk.host.now >= start + 40)
    desk.hear('DISCONNECTED', 'PTT ON')
    desk.drive(until=lambda: desk.host.now >= start + 48)
    desk.hear('PTT OFF')
    desk.drive()
    assert desk.session.steps is None
    assert desk.sent.count('DISCONNECT') == 1
    assert 'CLEANTXBUFFER' not in desk.sent and 'ABORT' not in desk.sent


@pytest.mark.parametrize('bw', [500, 2300, 2750])
@pytest.mark.parametrize('drained', [False, True])
def test_farewell_hard_cap_aborts_despite_continuing_drain_progress(bw, drained):
    desk = stuck_desk(bw=bw, **{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    start = desk.host.now
    desk.session.command('/bye')
    for at in range(8, 57, 8):
        desk.drive(until=lambda: desk.host.now >= start + at)
        desk.hear(f'BUFFER {10 - at // 8}')
        assert 'DISCONNECT' not in desk.sent
    if drained:
        desk.hear('BUFFER 0', 'PTT OFF')
        desk.drive(until=lambda: 'DISCONNECT' in desk.sent)
    desk.drive(until=lambda: 'ABORT' in desk.sent)
    assert 60 <= desk.host.now - start <= 60.1 + 1e-9
    desk.drive()
    at = desk.sent.index('CLEANTXBUFFER')
    assert desk.sent[at:at + 4] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert not desk.host.client.connected and not desk.host.client.ptt


@pytest.mark.parametrize('bw,window', [(500, 25), (2300, 10), (2750, 10)])
def test_idle_host_activity_does_not_extend_the_farewell_deadline(bw, window):
    desk = stuck_desk(bw=bw, **{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    start = desk.host.now
    desk.session.command('/bye')
    for at in range(1, window):
        desk.drive(until=lambda: desk.host.now >= start + at)
        desk.hear('PTT OFF', 'PTT ON', 'BUSY ON', 'BUSY OFF',
                  'BUFFER 10', 'BUFFER 10', 'BUFFER 0 junk', 'BUFFER nope', 'OK')
    desk.drive(until=lambda: 'ABORT' in desk.sent)
    assert window <= desk.host.now - start <= window + .1 + 1e-9
    desk.drive()


@pytest.mark.parametrize('bw,window', [(500, 25), (2300, 10), (2750, 10)])
def test_late_progress_cannot_revive_an_expired_farewell(bw, window):
    desk = stuck_desk(bw=bw, **{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    desk.session.command('/bye')
    desk.tick()
    desk.host.now += window
    desk.hear('BUFFER 5')
    desk.drive()
    assert 'ABORT' in desk.sent


@pytest.mark.parametrize('bw,window', [(500, 25), (2300, 10), (2750, 10)])
def test_close_completed_after_deadline_still_requires_abort_confirmation(bw, window):
    desk = stuck_desk(bw=bw, **{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    desk.session.command('/bye')
    desk.hear('BUFFER 0', 'PTT OFF')
    desk.drive(until=lambda: 'DISCONNECT' in desk.sent)
    desk.host.now += window
    desk.hear('DISCONNECTED')
    desk.drive()
    at = desk.sent.index('CLEANTXBUFFER')
    assert desk.sent[at:at + 4] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']


@pytest.mark.parametrize('bw,window', [(500, 25), (2300, 10), (2750, 10)])
def test_fresh_buffer_zero_extends_the_wait_for_unkey_without_a_positive_report(bw, window):
    desk = stuck_desk(bw=bw, **{'DISCONNECT': ['OK', 'DISCONNECTED']})
    desk.host.data.respond = lambda data: desk.host.lines('PTT ON')
    start = desk.host.now
    desk.session.command('/bye')
    desk.drive(until=lambda: desk.host.now >= start + window - 2)
    desk.hear('BUFFER 0')
    desk.drive(until=lambda: desk.host.now >= start + window + 6)
    desk.hear('PTT OFF')
    desk.drive()
    assert desk.sent.count('DISCONNECT') == 1
    assert 'CLEANTXBUFFER' not in desk.sent and 'ABORT' not in desk.sent


@pytest.mark.parametrize('drained', [False, True])
def test_quit_immediately_aborts_a_progress_extended_farewell(drained):
    desk = stuck_desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    start = desk.host.now
    desk.session.command('/bye')
    for at, remaining in [(8, 8), (16, 0 if drained else 4)]:
        desk.drive(until=lambda: desk.host.now >= start + at)
        desk.hear(f'BUFFER {remaining}', 'PTT OFF')
    if drained:
        desk.drive(until=lambda: 'DISCONNECT' in desk.sent)
    before, start = len(desk.sent), desk.host.now
    pending = len(desk.host.command.incoming)
    commands = []
    respond = desk.host.command.respond
    def record(data):
        commands.append(desk.host.now)
        respond(data)
    desk.host.command.respond = record
    assert desk.session.command('/quit') is False
    desk.session.quit()
    assert commands == [start] * 5
    assert desk.host.now - start <= .1 * (pending + 5) + 1e-9
    assert desk.sent[before:] == ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert desk.texts.count('8 73<DISC>') == 1
    assert not desk.host.client.connected and not desk.host.client.ptt


@pytest.mark.parametrize('clean', ['CLEANTXBUFFER OK', 'CLEANTXBUFFER FAILED ', 'WRONG'])
def test_missing_farewell_ack_always_disconnects_within_25_seconds(clean):
    desk = stuck_desk(**{'CLEANTXBUFFER': [clean],
                         'DISCONNECT': ['OK', 'DISCONNECTED', 'PTT OFF']})
    start = desk.host.now
    assert desk.session.command('/bye')
    assert desk.host.now - start < .2 and desk.session.steps is not None
    desk.drive()
    assert 25 <= desk.host.now - start < 26
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
    desk.host.now += 25
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


@pytest.mark.parametrize('failed', ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT'])
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
    assert attempts == ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert not desk.host.client.ptt


@pytest.mark.parametrize('state', ['PTT ON', 'PENDING'])
def test_quit_without_a_link_still_stops_unfinished_keying(state):
    desk = Desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK', 'PTT OFF']})
    desk.hear(state)
    desk.session.quit()
    assert desk.sent[-4:] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert not desk.host.client.ptt and not desk.host.client.pending
    assert not desk.texts


def test_quit_cannot_succeed_when_host_is_idle_but_physical_key_is_high():
    desk = Desk()
    desk.line = True
    with pytest.raises(v.HostError, match='keying line remains HIGH'):
        desk.session.quit()


@pytest.mark.parametrize('interrupt_at', ['farewell', 'disconnect', 'clear'])
def test_quit_closes_even_when_interrupt_exhausted_the_farewell_generator(interrupt_at):
    desk = stuck_desk(**{'DISCONNECT': ['OK', 'DISCONNECTED', 'PTT OFF']})
    if interrupt_at == 'farewell':
        def interrupt(text):
            raise KeyboardInterrupt
        desk.host.client.send_text = interrupt
    else:
        if interrupt_at == 'disconnect':
            desk.host.data.respond = lambda data: desk.host.lines('BUFFER 0')
        original = desk.host.client.send_command
        def interrupt(command):
            if command == ('DISCONNECT' if interrupt_at == 'disconnect' else 'CLEANTXBUFFER'):
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
    desk.host.lines('BUFFER 0', 'PTT OFF')
    desk.tick()
    desk.tick()
    assert 'DISCONNECT' in desk.sent and 'ABORT' not in desk.sent
    texts = desk.texts[:]
    start = desk.host.now
    desk.session.command('/stop')
    desk.drive()
    assert desk.host.now - start < 1
    assert desk.sent.count('DISCONNECT') == 2 and desk.sent.count('ABORT') == 1
    at = desk.sent.index('CLEANTXBUFFER')
    assert desk.sent[at:at + 4] == ['CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
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
