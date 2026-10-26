# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An unsupported encryption request cannot succeed on a plaintext modem."""
import pytest

from hfhost import ptc
from hfmodem.shrike.ptc import PtcHost
from .test_ptc_compatibility import Client, connected


@pytest.mark.parametrize('value', ('1', '2', '3'))
def test_terminal_and_forwarded_enable_refuse_before_changing_state(value):
    host = PtcHost()
    assert b'ERROR' in host.feed(f'AES {value}\r'.encode())
    assert host.settings.get('AES', '0') == '0'
    client = Client(host)
    assert client.cmd(f'#AES {value}').code == ptc.FAIL
    assert host.settings.get('AES', '0') == '0'


@pytest.mark.parametrize('channel', (0, 31))
@pytest.mark.parametrize('command', ('AES 1', 'AES2', 'AES 3'))
def test_direct_hostmode_enable_is_not_parsed_as_unrelated_a_setting(channel, command):
    host = PtcHost()
    client = Client(host)
    host.settings['A'] = 'preserved'
    assert client.cmd(command, channel=channel).code == ptc.FAIL
    assert host.settings['A'] == 'preserved'
    assert host.settings.get('AES', '0') == '0'


def test_explicit_disable_and_query_keep_pending_plaintext_and_accounting():
    host, client = connected()
    original = b'AES 1\x00\xff' + bytes(range(256))
    for offset in range(0, len(original), 128):
        assert client.exchange(original[offset:offset+128], False).code == ptc.OK
    host.settings['A'] = 'preserved'
    assert client.cmd('AES 0').code == ptc.OK
    assert client.cmd('AES').text == '0'
    assert host.settings['AES'] == '0' and host.settings['A'] == 'preserved'
    assert client.cmd('AES 1').code == ptc.FAIL
    assert client.retry().code == ptc.FAIL
    assert host.arq.payload_ledger.accepted == len(original)
    assert host.arq.payload_ledger.pending == len(original)
    assert host.arq.payload_ledger.confirmed == host.sent_total == 0
