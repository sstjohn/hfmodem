# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Unsupported wire commands cannot masquerade as effective settings or polls."""
import pytest

from hfhost import ptc
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from .test_ptc_compatibility import Client


def test_unknown_readable_terminal_commands_refuse_without_creating_settings():
    host = PtcHost()
    before = dict(host.settings)
    for command in (b'XENON 1\r', b'XENON\r', b'PTCC 31\r'):
        assert b'*** ERROR:' in host.feed(command)
        assert host.settings == before
    assert host._command('\xaa\x00garbage') == b''


@pytest.mark.parametrize('command', ['Z 1', 'Z', '%Z1', '#XENON 1', '#XENON'])
def test_unknown_host_commands_and_cached_retries_refuse_without_mutation(command):
    host = PtcHost()
    client = Client(host)
    before = dict(host.settings)
    response = client.cmd(command)
    assert response.code == ptc.FAIL
    assert client.retry() == response
    assert host.settings == before


def test_manual_minimum_listen_abbreviation_updates_the_actual_monitor_state():
    host = PtcHost()
    host.set_listen(True)
    assert host.listening
    assert b'ERROR' not in host.feed(b'L 0\r')
    assert not host.listening and host.settings['LISTEN'] == '0'
    assert 'L' not in host.settings
    assert host.arq.state == State.LISTENING  # L is monitoring, not call admission.
    assert host._status_bytes()[0] == 0x87
    assert b'LISTEN: 0' in host.feed(b'L\r')


@pytest.mark.parametrize('channel,command,is_command', [
    (254, b'Z3', True), (254, b'G4', True), (254, b'G-1', True),
    (254, b'G3 extra', True), (254, b'G3', False),
    (255, b'Z', True), (255, b'G1', True), (255, b'G', False),
])
def test_special_channels_require_their_actual_poll_grammar(channel, command, is_command):
    host = PtcHost()
    client = Client(host)
    for _ in range(8):  # The startup banner can span multiple host packets.
        response = client.cmd('G', channel=0)
        if response.code == ptc.OK:
            break
        assert response.code == ptc.DATA
    else:
        pytest.fail('startup banner did not drain within its bound')
    host.channel(31).rx.extend(b'pending\x00\xff')
    before = host._laststatus
    response = client.exchange(command, command=is_command, channel=channel)
    assert response.code == ptc.FAIL
    assert client.retry() == response
    assert host._laststatus == before
    assert host.channel(31).rx == b'pending\x00\xff'
    assert client.cmd('G', channel=255).data == bytes([32])
    assert client.cmd('G3', channel=254).data == host._status_bytes()
    assert client.cmd('G').data == b'pending\x00\xff'


@pytest.mark.parametrize('command', [b'JHOST4junk\r', b'JHOST4 1\r', b'JHOST9\r'])
def test_invalid_terminal_hostmode_entry_refuses_without_entering(command):
    host = PtcHost()
    assert b'*** ERROR:' in host.feed(command)
    assert not host.hostmode


def test_hostmode_exit_requires_the_complete_command():
    host = PtcHost()
    client = Client(host)
    response = client.cmd('JHOST0junk')
    assert response.code == ptc.FAIL and host.hostmode
    assert client.retry() == response and host.hostmode
    assert client.cmd('JHOST0').code == ptc.OK and not host.hostmode
