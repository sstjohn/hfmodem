# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS K calendar and TheFirmware-derived event timestamp presentation."""
import re

import pytest

from hfmodem.ax25.connected import Connection, State
from hfmodem.brubru.host.tracker import TrackerHost
from hfmodem.shrike import hostmode as hm
from .test_tracker import request


def setup():
    now = [0.]
    engine = Connection('N0CALL', lambda _: True, clock=lambda: now[0])
    host = TrackerHost(engine, clock=lambda: now[0])
    host.feed(b'\x1bJHOST1\r')
    return host, engine, now


def test_k_query_calendar_formats_and_leap_day_advance_on_the_owner_clock():
    host, _, now = setup()
    assert re.fullmatch(rb'0 \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}', request(host, 'K').data)
    for command in ['K 28.02.28', 'K 23:59:59', 'K1']:
        assert request(host, command).code == hm.OK
    assert request(host, 'K').data == b'1 28.02.28 23:59:59'
    now[0] += 1
    assert request(host, 'K').data == b'1 29.02.28 00:00:00'
    assert request(host, 'K 02/29/28').code == hm.OK
    assert request(host, 'K2').code == hm.OK
    assert request(host, 'K').data == b'2 02/29/28 00:00:00'


def test_event_time_is_captured_at_transition_and_terminal_event_still_waits_for_binary_rx():
    host, engine, now = setup()
    for command in ['K 28.02.28', 'K 23:59:59', 'K1']:
        assert request(host, command).code == hm.OK
    engine.remote = 'N0DX'
    engine._state(State.CONNECTED)
    now[0] = 2
    host._receive(b'\x00\xff final')
    engine.abort('peer disconnected')
    now[0] = 100
    first = request(host, 'G')
    assert first.code == hm.LINK
    assert first.data == b'(1) CONNECTED to N0DX - 28.02.28 23:59:59'
    assert request(host, 'G1').code == hm.OK
    assert request(host, 'G').data == b'\x00\xff final'
    last = request(host, 'G')
    assert last.code == hm.LINK
    assert last.data == b'(1) DISCONNECTED to N0DX - 29.02.28 00:00:01'
    assert request(host, 'G').code == hm.OK


def test_timestamp_enable_is_applied_when_polled_without_restamping_the_event():
    host, engine, now = setup()
    for command in ['K 02/20/28', 'K 17:36:00']:
        assert request(host, command).code == hm.OK
    engine.remote = 'N0DX'
    engine._state(State.CONNECTED)
    now[0] = 10
    assert request(host, 'K1').code == hm.OK
    assert request(host, 'G').data == b'(1) CONNECTED to N0DX - 02/20/28 17:36:00'
    engine.abort('peer disconnected')
    assert request(host, 'K0').code == hm.OK
    assert request(host, 'G').data == b'(1) DISCONNECTED to N0DX'


@pytest.mark.parametrize('invalid', ['3', '-1', '02/30/28', '31.04.28',
                                   '25:00:00', '12:60:00', '12:00:60', '2028-02-20'])
def test_invalid_calendar_requests_leave_clock_policy_and_ax25_timers_intact(invalid):
    host, engine, _ = setup()
    for command in ['K 20.02.28', 'K 17:36:00', 'K1']:
        assert request(host, command).code == hm.OK
    engine.connect('N0DX')
    engine.tick()
    assert engine._deadline == 2 * engine.rtt
    before = (dict(host.values), engine.rtt, engine.t2, engine.t3, engine._deadline)
    assert request(host, 'K '+invalid).code == hm.FAIL
    assert request(host, 'K').data == b'1 20.02.28 17:36:00'
    assert (dict(host.values), engine.rtt, engine.t2, engine.t3, engine._deadline) == before


def test_calendar_changes_do_not_move_an_active_ax25_deadline():
    host, engine, now = setup()
    engine.connect('N0DX')
    engine.tick()
    deadline = engine._deadline
    assert deadline == 2 * engine.rtt
    for command in ['K 31.12.99', 'K 23:59:59', 'K1']:
        assert request(host, command).code == hm.OK
    assert engine.clock() == 0 and engine._deadline == deadline
    now[0] = deadline
    engine.tick()
    assert engine.retries == 1 and engine._deadline > deadline
