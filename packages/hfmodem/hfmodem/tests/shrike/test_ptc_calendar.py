# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Required DATE/TIME initialization changes an advancing private calendar."""
import time

import pytest

from hfhost import ptc
from hfmodem.shrike.ptc import PtcHost
from .test_ptc_compatibility import Client


@pytest.mark.parametrize('date', ['28.02.28', '280228', '28:02:28'])
@pytest.mark.parametrize('clock', ['23:59:59', '235959'])
def test_required_calendar_formats_advance_across_leap_midnight(date, clock):
    now = [0.]
    host = PtcHost(calendar_clock=lambda: now[0])
    assert b'ERROR' not in host.feed(f'DA {date}\rTI {clock}\r'.encode())
    now[0] = 2
    assert b'DATE: 29.02.28' in host.feed(b'DATE\r')
    assert b'TIME: 00:00:01' in host.feed(b'TIME\r')


@pytest.mark.parametrize('year, full', [('90', 1990), ('89', 2089)])
def test_date_uses_the_manuals_1990_to_2089_era(year, full):
    host = PtcHost()
    assert b'ERROR' not in host.feed(f'DA 01.01.{year}\r'.encode())
    assert host._calendar_now().year == full


@pytest.mark.parametrize('command', ['DA 30.02.28', 'DA 31.04.28', 'DA 1.01.28',
    'DA 28.02:28', 'TI 24:00:00', 'TI 23:60:00', 'TI 23:00:60', 'TI 1:02:03'])
def test_bad_requests_preserve_calendar_readback_and_protocol_timers(command):
    now = [0.]
    host = PtcHost(calendar_clock=lambda: now[0])
    host.feed(b'DA 20.10.26\rTI 12:34:56\r')
    future = time.monotonic() + 1000
    host.wait_until, host.busy_until = future, future + 1
    before = dict(host.settings), host._calendar_now(), host.wait_until, host.busy_until
    assert b'ERROR' in host.feed((command+'\r').encode())
    assert (host.settings, host._calendar_now(), host.wait_until, host.busy_until) == before


def test_forwarded_crc_retry_does_not_reset_the_clock_twice():
    now = [0.]
    host = PtcHost(calendar_clock=lambda: now[0])
    client = Client(host)
    assert client.cmd('#DATE 28:02:28').code == ptc.OK
    result = client.cmd('#TIME 23:59:59')
    assert result.code == ptc.OK
    now[0] = 2
    assert client.retry() == result
    assert client.cmd('JHOST0').code == ptc.OK
    assert b'DATE: 29.02.28' in host.feed(b'DA\r')
    assert b'TIME: 00:00:01' in host.feed(b'TI\r')
