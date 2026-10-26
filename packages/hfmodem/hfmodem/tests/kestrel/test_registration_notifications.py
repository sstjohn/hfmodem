# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Registration grammar and received link capability have different sources."""
import socket

import pytest

from hfmodem.kestrel.host.modem_core import LoopbackModem, ModemCore
from hfmodem.kestrel.host.server import HostSession
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F
from hfmodem.tests.kestrel.test_compression import IO


@pytest.fixture
def endpoint():
    pairs = [socket.socketpair(), socket.socketpair()]
    modem = VaraHostModem(IO(), bandwidth='500')
    session = HostSession(pairs[0][0], pairs[1][0], modem)
    lines = []
    session._send_cmd = lines.append
    modem.start(session)
    yield session, modem, lines
    session.close()
    modem.poll()
    session.ensure_retired()
    for _, peer in pairs:
        peer.close()


@pytest.mark.parametrize('full', [False, True])
def test_local_capability_is_once_per_attachment_without_a_license_callsign(endpoint, full):
    _, modem, lines = endpoint
    modem._high_speed = full
    assert not lines
    modem.poll()
    modem.poll()
    assert [line for line in lines if line.startswith('REGISTERED')] == [
        'REGISTERED HFMODEM' if full else 'REGISTERED']


@pytest.mark.parametrize('caller_high_speed', [False, True])
@pytest.mark.parametrize('local_high_speed', [False, True])
def test_link_notification_uses_received_caller_marker_after_actual_answer(endpoint, monkeypatch,
                                                                         caller_high_speed, local_high_speed):
    _, modem, lines = endpoint
    modem.poll()
    lines.clear()
    hs = modem.station
    hs.high_speed = local_high_speed
    hs.role, hs.called, hs.state = 'responder', 'N0CALL', A.VaraState.CONNECTING
    frame = F.link_setup_frame('N0DX-10', bw='500', high_speed=caller_high_speed, compression='OFF')
    monkeypatch.setattr(A.OF, 'link_setup_read', lambda *a, **kw: frame)
    monkeypatch.setattr(hs, '_tx_connected_ack', lambda: False)
    assert not hs._rx_link_setup([])
    assert not lines
    monkeypatch.setattr(hs, '_tx_connected_ack', lambda: True)
    assert hs._rx_link_setup([])
    assert lines == ['CONNECTED N0DX-10 N0CALL 500',
                     'LINK REGISTERED' if caller_high_speed else 'LINK UNREGISTERED']
    assert hs._link_high_speed == (caller_high_speed and local_high_speed)


def test_outgoing_connect_cannot_invent_peer_registration_from_its_own_flag(endpoint):
    _, modem, lines = endpoint
    modem.poll()
    lines.clear()
    hs = modem.station
    hs.role, hs.caller, hs.called = 'initiator', 'N0CALL', 'N0DX'
    hs._link_high_speed = True
    hs._connected(confirm=False)
    assert lines == ['CONNECTED N0CALL N0DX 500']


def test_detached_unserviced_attachment_emits_no_registration(endpoint):
    session, modem, lines = endpoint
    session.close()
    modem.poll()
    assert not lines


@pytest.mark.parametrize('command', ['CHAT ON', 'CQFRAME N0CALL 500'])
def test_modems_without_chat_or_cq_refuse_without_effect(command):
    class FixedProfile(LoopbackModem):
        # Borrow the local transport, retaining the conservative core profile.
        set_chat = ModemCore.set_chat

    modem = FixedProfile()
    lines = []
    pairs = [socket.socketpair(), socket.socketpair()]
    try:
        session = HostSession(pairs[0][0], pairs[1][0], modem)
        session._send_cmd = lines.append
        session._dispatch(command)
    finally:
        for pair in pairs:
            for end in pair:
                end.close()
    assert lines == ['WRONG']
    assert not session.chat and not session.listening
    assert not modem._listen


@pytest.mark.parametrize('command', ['CHAT ON', 'CQFRAME N0CALL 500'])
def test_vara_host_admits_chat_and_cq(endpoint, command):
    session, modem, lines = endpoint
    session._dispatch(command)
    assert lines == ['OK']
    assert not modem._events.empty()


def test_fixed_chat_off_profile_is_admitted(endpoint):
    session, _, lines = endpoint
    session._dispatch('CHAT OFF')
    assert lines == ['OK']
    assert not session.chat
