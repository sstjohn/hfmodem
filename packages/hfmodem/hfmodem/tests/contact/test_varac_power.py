# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""`/power` sets the rig's RF power through the session's CAT, held to the station clamp."""
import pytest

from hfmodem.tests.contact.test_varac import Desk
from hfmodem.tests.contact.test_varac_local_commands import closing


class Radio:
    """The session's CAT: RFPOWER in watts, read back after any write."""

    def __init__(self, watts=100.0):
        self.watts, self.writes = watts, []

    def __call__(self, watts=None):
        if watts is not None:
            self.writes.append(watts)
            self.watts = watts
        return self.watts


def powered(desk=None, clamp=25.0, watts=100.0):
    desk = desk or Desk()
    radio = desk.session.power = Radio(watts)
    desk.session.power_clamp_w = clamp
    return desk, radio


def test_power_alone_reads_the_rig_and_names_the_clamp():
    desk, radio = powered()
    desk.session.command('/power')
    assert desk.said[-1] == 'power 100 W (clamp 25 W)' and radio.writes == []


def test_power_under_the_clamp_is_set_read_back_and_logged():
    desk, radio = powered()
    desk.session.command('/power 20')
    assert radio.writes == [20] and desk.said[-1] == 'power set to 20 W (clamp 25 W)'
    assert desk.events[-1] == dict(kind='power', requested_w=20, set_w=20, read_w=20, clamp_w=25)


def test_power_above_the_clamp_sets_the_clamp_and_says_both():
    desk, radio = powered()
    desk.session.command('/power 80')
    assert radio.writes == [25]
    assert desk.said[-2:] == ['warning: /power asks 80 W, above the 25 W station clamp '
                              '([rig] power_clamp_w); setting 25 W',
                              'power set to 25 W (clamp 25 W)']
    assert desk.events[-1]['requested_w'] == 80 and desk.events[-1]['set_w'] == 25


def test_without_a_clamp_the_asked_power_is_set():
    desk, radio = powered(clamp=None)
    desk.session.command('/power 80')
    assert radio.writes == [80] and desk.said[-1] == 'power set to 80 W (no clamp)'


def test_power_asked_while_keyed_waits_for_the_key_to_come_up():
    desk, radio = powered()
    desk.hear('PTT ON')
    desk.session.command('/power 15')
    assert radio.writes == []
    assert desk.said[-1] == 'transmitting; power goes to 15 W as soon as the key is up'
    desk.tick()
    assert radio.writes == []
    desk.hear('PTT OFF')
    assert radio.writes == [15] and 'power set to 15 W (clamp 25 W)' in desk.said


def test_power_waits_for_the_keying_line_as_well_as_the_host():
    desk, radio = powered()
    desk.line = True
    desk.session.command('/power 15')
    desk.tick()
    assert radio.writes == []
    desk.line = False
    desk.tick()
    assert radio.writes == [15]


def test_power_is_a_local_setting_that_answers_while_closing():
    desk, radio = powered(closing())
    desk.session.command('/power 10')
    assert desk.said[-1] == 'transmitting; power goes to 10 W as soon as the key is up'
    desk.session.quit()
    desk, radio = powered(closing())
    desk.hear('PTT OFF')
    assert desk.session._closing and not desk.host.client.ptt
    desk.session.command('/power 10')
    assert radio.writes == [10] and desk.said[-1] == 'power set to 10 W (clamp 25 W)'
    desk.session.quit()


@pytest.mark.parametrize('line', ['/power abc', '/power 0', '/power 101', '/power 10 20'])
def test_a_power_that_is_not_watts_is_a_usage_line(line):
    desk, radio = powered()
    desk.session.command(line)
    assert desk.said[-1] == 'usage: /power [WATTS], 0 < WATTS <= 100' and radio.writes == []


def test_a_session_without_cat_says_so():
    desk = Desk()
    desk.session.command('/power 10')
    assert desk.said[-1] == 'power control is not available in this session'


def test_a_cat_failure_is_said_and_the_session_goes_on():
    desk, _ = powered()
    def refused(watts=None):
        raise ConnectionRefusedError('rigctld is gone')
    desk.session.power = refused
    assert desk.session.command('/power 10')
    assert desk.said[-1] == 'power not set: rigctld is gone'


@pytest.mark.parametrize('reads, said', [
    (100.0, '!! power asked 20 W, the rig reads 100 W (clamp 25 W)'),
    (None, '!! power asked 20 W, RFPOWER did not read back (clamp 25 W)'),
])
def test_a_readback_off_the_asked_power_is_said_with_both_figures(reads, said):
    desk, _ = powered()
    desk.session.power = lambda watts=None: reads
    desk.session.command('/power 20')
    assert desk.said[-1] == said
    assert desk.events[-1]['set_w'] == 20 and desk.events[-1]['read_w'] == reads
