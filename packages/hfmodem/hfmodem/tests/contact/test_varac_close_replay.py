# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from collections import deque
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.tests.contact.test_varac_launcher import launch as launch  # noqa: F401


RECORDING = json.loads((Path(__file__).parent / 'fixtures' / 'varac-ke8plm-close.json').read_text())
BYE = datetime.fromisoformat(RECORDING['events'][0]['utc'])


def seconds(row):
    return (datetime.fromisoformat(row['utc']) - BYE).total_seconds()


def replay():
    desk = Desk(**{'CONNECT W9SSJ KE8PLM': [], 'CLEANTXBUFFER': ['CLEANTXBUFFER FAILED '],
                   'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    host, client = desk.host, desk.host.client
    host.data.respond = lambda data: None
    desk.session.intent = 'call'
    desk.flow.band, desk.flow.dial = 80, 3_595_000
    desk.flow.state = v.State.CONNECTED
    rows = RECORDING['host']
    for row in rows:
        if seconds(row) >= 0:
            break
        host.now = seconds(row)
        if row['kind'] == 'command':
            client.send_command(row['raw'])
        elif row['kind'] == 'sent':
            client.send_text(row['raw'])
        elif row['kind'] == 'host':
            host.lines(row['raw'])
        elif row['kind'] == 'message':
            host.data.incoming.append(v.frame_message(row['raw']))
        client.poll(0)
    assert client.connected and client.buffer == 0 and not client.ptt
    assert desk.texts == ['5 <AWQ>', '2 ?\n']
    host.now = 0
    pending = deque(row for row in rows if seconds(row) >= 0 and row['kind'] == 'host')
    commands = []
    respond = host.command.respond
    unkey_at = None

    def command(data):
        nonlocal unkey_at
        raw = data.decode().rstrip('\r')
        commands.append((host.now, raw))
        if raw == 'ABORT':
            pending.clear()
            unkey_at = host.now + .7
            host.lines('PTT ON')
        respond(data)

    def ready(sockets, timeout):
        nonlocal unkey_at
        host.now += timeout
        while pending and seconds(pending[0]) <= host.now:
            host.lines(pending.popleft()['raw'])
        if unkey_at is not None and host.now >= unkey_at:
            host.lines('PTT OFF')
            unkey_at = None
        return [stream for stream in sockets if stream.incoming]

    host.command.respond = command
    client.ready = ready
    desk.session.line_state = lambda: client.ptt
    return desk, commands


def test_recording_pins_missing_ack_duplicate_farewell_and_eighteen_keyings():
    rows = RECORDING['host']
    assert [seconds(row) for row in rows if row['kind'] == 'sent' and row['raw'] == '73<DISC>'] == [0, 40.642]
    assert [row['raw'] for row in rows if seconds(row) >= 0 and row['raw'].startswith('BUFFER ')] == ['BUFFER 10', 'BUFFER 20']
    assert not any(row['raw'] in ('DISCONNECT', 'ABORT') for row in rows)
    assert sum(seconds(row) >= 0 and row['raw'] == 'PTT ON' for row in rows) == 18
    assert sum(seconds(row) >= 40.642 and row['raw'] == 'PTT ON' for row in rows) == 6
    assert [row['packet_kind'] for row in RECORDING['frames']] == [
        'session-turn-request-short', 'data', 'session-answer-query-short-a',
        *(['session-intermediate-answer-query-a'] * 15)]


def test_ke8plm_unacknowledged_farewell_sends_all_stop_commands_at_twenty_five_seconds():
    desk, commands = replay()
    assert desk.session.command('/bye')
    assert desk.host.now == 0 and desk.session.steps is not None
    desk.drive()
    assert [raw for _, raw in commands[:5]] == ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert all(25 <= at <= 25.1 + 1e-9 for at, _ in commands[:5])
    assert desk.host.now <= 26
    assert desk.texts == ['5 <AWQ>', '2 ?\n', '8 73<DISC>']
    assert not desk.host.client.connected and not desk.host.client.ptt


@pytest.mark.parametrize('quit_at', [.5, 8.2, 40.642])
def test_terminal_reads_quit_during_recorded_farewell_and_waits_for_unkey(launch, quit_at):
    desk, commands = replay()
    typed = []

    class Tty:
        def readline(self):
            if not typed:
                typed.append(desk.host.now)
                return '/bye\n'
            if desk.host.now >= quit_at:
                typed.append(desk.host.now)
                return '/quit\n'
            return None

    launch.terminal(desk.session, Tty())
    assert typed[1] <= quit_at + .1 + 1e-9
    assert desk.texts.count('8 73<DISC>') == 1
    assert [raw for _, raw in commands if raw in ('CLEANTXBUFFER', 'DISCONNECT', 'ABORT')] == [
        'CLEANTXBUFFER', 'DISCONNECT', 'ABORT']
    abort_at = next(at for at, raw in commands if raw == 'ABORT')
    assert abort_at <= min(quit_at, typed[0] + 25) + .1 + 1e-9
    assert not desk.host.client.ptt and not desk.host.client.connected


def test_ack_at_nine_seconds_restarts_the_close_deadline():
    desk, commands = replay()
    desk.session.command('/bye')
    desk.drive(until=lambda: desk.host.now >= 9)
    desk.host.lines('BUFFER 0', 'PTT OFF')
    desk.drive()
    assert next(at for at, raw in commands if raw == 'DISCONNECT') < 9.5
    disconnect = next(at for at, raw in commands if raw == 'DISCONNECT')
    abort = next(at for at, raw in commands if raw == 'ABORT')
    assert 25 <= abort - disconnect <= 25.1 + 1e-9


def test_operator_close_cancels_beacon_schedule_after_return_to_monitoring():
    desk = Desk(**{'DISCONNECT': ['OK', 'DISCONNECTED', 'PTT OFF']})
    desk.hear('CONNECTED W9SSJ KE8PLM 500')
    desk.session.set_beacon(15)
    desk.session._beacon_due = desk.host.now + 900
    due = desk.session._beacon_due
    desk.session.command('/bye')
    desk.drive()
    assert desk.flow.state is v.State.MONITORING
    assert desk.session.beacon is None
    assert desk.session._beacon_due is desk.session._beacon_expires is None
    sent = desk.sent[:]
    desk.host.now += v.BEACON_CLEAR_S + 1
    desk.tick()
    assert desk.sent == sent
    desk.host.now = due
    desk.tick()
    desk.drive()
    assert desk.sent == sent
    assert 'CQFRAME W9SSJ-9 500' not in desk.sent


def test_operator_close_returns_to_answering_after_the_close():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KE8PLM 500')
    desk.session.command('/bye')
    assert not desk.session._auto_answer
    desk.drive()
    assert desk.flow.state is v.State.MONITORING
    assert desk.session._auto_answer and desk.armed[-1] is True
    assert desk.sent[-1] == 'LISTEN CQ'


def test_late_bbs_entry_cannot_queue_even_a_refusal_during_close():
    desk, _ = replay()
    desk.session.bbs.core = SimpleNamespace(tick=lambda: None)
    desk.session.command('/bye')
    texts = desk.texts[:]
    desk.receive('BBS')
    assert desk.texts == texts
    desk.session.quit()
