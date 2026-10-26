# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fixed PACTOR profiles must refuse auxiliary behavior they cannot perform."""
import pytest

from hfhost import ptc
from hfmodem.shrike.ptc import PtcHost
from .test_ptc_compatibility import Client, connected


@pytest.mark.parametrize('command', ['ARX', 'CHOBELL', 'CMSG', 'REMOTE',
                                     'LFIGNORE', 'ADDLF', 'PDUPLEX', 'HCR',
                                     'UMLAUTS'])
@pytest.mark.parametrize('route', ['terminal', 'forwarded'])
def test_auxiliary_enables_refuse_and_preserve_the_effective_disabled_profile(command, route):
    host = PtcHost()
    client = Client(host) if route == 'forwarded' else None
    if client:
        assert client.cmd('#'+command+' 0').code == ptc.OK
    else:
        assert b'ERROR' not in host.feed((command+' 0\r').encode())
    before = dict(host.settings)
    if client:
        refused = client.cmd('#'+command+' 1')
        assert refused.code == ptc.FAIL
        assert client.retry() == refused
    else:
        assert b'*** ERROR:' in host.feed((command+' 1\r').encode())
        assert (command+': 0').encode() in host.feed((command+'\r').encode())
    assert host.settings == before and host.settings[command] == '0'


@pytest.mark.parametrize('command', ['CWID 1 0', 'CWID 4 2', 'CWID 0 4',
                                     'TONES 0', 'TONES 2', 'TONES 5',
                                     'MARK 299', 'SPACE 2701', 'BR 8'])
def test_unsupported_identification_or_tone_controls_cannot_change_pending_originals(command):
    host, client = connected()
    assert client.exchange(b'original\x00\xff\x1c', False).code == ptc.OK
    before = dict(host.settings)
    ledger = (host.arq.payload_ledger.accepted, host.arq.payload_ledger.confirmed,
              host.arq.payload_ledger.pending)
    response = client.cmd('#'+command)
    assert response.code == ptc.FAIL and client.retry() == response
    assert host.settings == before
    assert (host.arq.payload_ledger.accepted, host.arq.payload_ledger.confirmed,
            host.arq.payload_ledger.pending) == ledger


def test_fixed_tones_and_display_profiles_have_explicit_readback():
    host = PtcHost()
    assert b'TONES: 4' in host.feed(b'TO\r')
    assert b'BRIGHTN: 6' in host.feed(b'BR\r')
    for command in [b'TONES 4', b'MARK 1600', b'SPACE 1400', b'BR 0', b'CWID 0']:
        assert b'ERROR' not in host.feed(command+b'\r')
    assert host.settings['CWID'] == '0 0'
    assert (host.settings['MARK'],host.settings['SPACE'],host.settings['TONES']) == ('1600','1400','4')
    assert b'BRIGHTN: 0' in host.feed(b'BR\r')


@pytest.mark.parametrize('arg', ['S 1', 'SCAN 1', 'DU 1', 'DUMP 1', 'TY I', 'S C 5', ''])
def test_unimplemented_trx_operation_refuses_without_mutating_settings(arg):
    host = PtcHost()
    before = dict(host.settings)
    assert b'*** ERROR:' in host.feed(('TRX '+arg+'\r').encode())
    assert host.settings == before


def test_required_trx_disable_commands_preserve_two_distinct_disabled_states():
    host = PtcHost()
    for command in [b'TRX S0', b'TRX SCAN 0', b'TRX DU0', b'TRX DUMP 0']:
        assert b'ERROR' not in host.feed(command+b'\r')
    assert host.settings['TRX_SCAN'] == host.settings['TRX_DUMP'] == '0'
    assert 'TRX' not in host.settings


@pytest.mark.parametrize('command', [b'PR\r', b'PAC\r'])
def test_selecting_an_unimplemented_packet_personality_cannot_succeed(command):
    host = PtcHost()
    before = dict(host.settings)
    assert b'*** ERROR:' in host.feed(command)
    assert host.settings == before
    assert b'ERROR' not in host.feed(b'PT\r')
