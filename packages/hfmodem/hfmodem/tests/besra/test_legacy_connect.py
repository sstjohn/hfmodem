# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Legacy CONNECT reaches native ARQ; its reply is not an ARQCALL echo."""
import numpy as np
import pytest

from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.host.protocol import ArdopState
from hfmodem.besra.host.server import HostServer
from hfmodem.besra.phy.demodulator import decode


def setup():
    modem = BesraModem()
    host = HostServer(modem, control_port=0, data_port=0, quiet=True)
    events, samples = [], []
    host._send_cmd = events.append
    modem.audio_out = samples.append
    modem.start(host, threaded=False)
    host._dispatch('MYCALL W9SSJ')
    return modem, host, events, samples


def test_trimode_connect_uses_configured_bandwidth_and_real_default_retry_budget():
    modem, host, events, samples = setup()
    try:
        host._dispatch('ARQBW 200MAX')
        host._dispatch('Connect k7abc')
        assert events[-1] == 'CONNECT IN PROCESS'
        assert modem.state == ArdopState.ISS and modem._session._conreq_budget == 5
        assert modem._session._session_bw == 200
        assert not modem.connected and not any(s.startswith('CONNECTED ') for s in events)
        assert len(samples) == 1
        frames = decode(np.pad(samples[0], (2400, 4800)))
        assert len(frames) == 1 and frames[0].ok
        assert frames[0].caller == 'W9SSJ' and frames[0].target == 'K7ABC'
        assert 'ConReq200' in frames[0].name
        assert modem._session.accepted_total == modem._session.confirmed_total == 0
    finally:
        modem.stop()


@pytest.mark.parametrize('command', ['CONNECT', 'CONNECT N0CALL 5', 'CONNECT @BAD',
                                   'CONNECT K7ABC,500', 'CONNECT K7ABC-999'])
def test_unsupported_connect_syntax_never_starts_a_contact(command):
    modem, host, events, samples = setup()
    try:
        host._dispatch(command)
        assert events[-1] == 'FAULT Connect Failure'
        assert modem.state == ArdopState.DISC and not samples
    finally:
        modem.stop()


@pytest.mark.parametrize('refusal', ['mycall', 'fec', 'codec', 'busy', 'owner-race'])
def test_rejected_connect_is_a_failure_and_preserves_the_preceding_contact(refusal):
    modem, host, events, samples = setup()
    try:
        if refusal == 'mycall':
            host._mycall = ''
        elif refusal == 'fec':
            host._dispatch('PROTOCOLMODE FEC')
        elif refusal == 'codec':
            host._dispatch('CODEC FALSE')
        elif refusal == 'busy':
            host._dispatch('ARQCALL N0OLD 3')
        else:
            # The owner accepts RX between the frontend's state check and the
            # queued connect operation. Its second check must refuse, not echo.
            original = modem.connect
            def raced(target):
                modem._session._set_state(ArdopState.IRS)
                original(target)
            modem.connect = raced
        count = len(samples)
        host._dispatch('CONNECT K7ABC')
        assert events[-1] == 'FAULT Connect Failure' and len(samples) == count
        if refusal == 'busy':
            assert modem._session._remote == 'N0OLD' and modem._session._conreq_budget == 3
    finally:
        modem.stop()


def test_arqcall_keeps_its_explicit_budget_and_original_echo():
    modem, host, events, samples = setup()
    try:
        host._dispatch('arqcall k7abc 9')
        assert events[-1] == 'arqcall k7abc 9'
        assert modem._session._conreq_budget == 9 and len(samples) == 1
        assert not any(s == 'CONNECT IN PROCESS' for s in events)
    finally:
        modem.stop()


def test_legacy_connect_over_real_host_sockets_keeps_binary_data_framing():
    from hfmodem.tests.besra.test_host_loopback import Client
    host = HostServer(control_port=0, data_port=0, quiet=True)
    control, data = host.start()
    client = Client('127.0.0.1', control, data)
    try:
        client.send('MYCALL W9SSJ')
        assert client.wait_present('MYCALL now W9SSJ')
        client.send('CONNECT K7ABC')
        assert client.wait_present('CONNECT IN PROCESS') == 'CONNECT IN PROCESS'
        assert client.wait_present('CONNECTED K7ABC')
        original = bytes(range(256))*2
        client.send_data(original)
        kind, received = client.recv_data_block()
        assert kind == 'ARQ' and received == original
        assert not any(line.startswith('ARQCALL') for line in client.lines)
    finally:
        client.close()
        host.stop()
