# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS 4.1 TERM 0/1 output and passive serial drain; no RF qualification."""
import os

from hfmodem.core import hostmode as hm
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.shrike.serial_endpoint import PtcSerialEndpoint
from .test_ptc_serial_endpoint import read_until


def connected():
    peer = SimPeer()
    host = PtcHost(peer, mycall='W9SSJ')
    host.feed(b'C REMOTE\r')
    peer.pump()
    assert host._link_up
    return host, peer


def drain(host):
    result = bytearray()
    for _ in range(64):
        part = host.drain_terminal_output()
        if not part:
            return bytes(result)
        result += part
    raise AssertionError('terminal output did not retire')


def test_terminal_connect_data_and_close_are_delivered_once_in_order():
    host, _ = connected()
    payload = bytes(range(256))
    host.deliver(payload)
    host.arq.on_host_abort()
    result = drain(host)
    assert result.index(b'CONNECTED to REMOTE') < result.index(payload) < result.index(b'DISCONNECTED fm REMOTE')
    assert result.endswith(b'cmd: ')
    assert not drain(host) and host.contact_ready() and host.host_output_deadline is None


def test_next_terminal_command_receives_final_event_before_redial():
    host = PtcHost()
    host.feed(b'C %REMOTE\r')
    host.arq.on_host_abort()
    result = host.feed(b'C !NEXT\r')
    assert result.index(b'LINK FAILURE with REMOTE') < result.index(b'NOW CALLING NEXT')
    assert host.arq.dxcall == 'NEXT' and host.arq.connect_variant == 'longpath'


def test_term0_holds_receive_text_during_command_input_then_releases_it():
    host, _ = connected()
    drain(host)
    host.feed(b'\x1bVER')
    host.deliver(b'held incoming text')
    assert not host.drain_terminal_output()
    result = host.feed(b'\r')
    assert b'PTC-IIIusb' in result
    assert b'held incoming text' in result + drain(host)
    assert host.sent_total == 0


def test_term0_documented_command_hold_is_bounded_and_fails_visibly():
    host, _ = connected()
    drain(host)
    host.feed(b'\x1bVER')
    host.deliver(b'x'*2000)
    host.deliver(b'overflow')
    assert len(host.channel(host.ptchn).rx) == 2000
    assert host.host_output_failure == 'HOST OUTPUT OVERFLOW'
    host.service_host_output()
    assert not host._link_up


def test_term1_delayed_echo_requires_confirmation_and_hostmode_stays_opaque():
    host, _ = connected()
    drain(host)
    host.feed(b'\x1bTERM 1\r')
    original = bytes(range(256))
    assert host.arq.on_host_data(original)
    assert not drain(host) and host.sent_total == 0
    host.payload_confirmed(original)
    assert drain(host) == original and host.sent_total == 256
    host.hostmode = True
    host.deliver(original)
    assert not host.drain_terminal_output()
    response = tuple(hm.Decoder('master').feed(host._get(host.ptchn, 'G')))[0]
    assert response.code == hm.DATA and response.data == original


def test_unimplemented_split_screen_modes_refuse_before_setting_changes():
    host = PtcHost()
    assert b'ERROR' not in host.feed(b'TERM 1\r')
    for value in ('-1', '2', '3', '4', '5', 'nonsense'):
        assert b'*** ERROR:' in host.feed(('TERM '+value+'\r').encode())
        assert host.settings['TERM'] == '1'


def test_passive_real_pty_emits_received_bytes_and_close_without_another_host_write():
    peer = SimPeer(greeting=bytes(range(256)))
    host = PtcHost(peer, mycall='W9SSJ')
    endpoint = PtcSerialEndpoint(host)
    fd = os.open(endpoint.open(), os.O_RDWR | os.O_NOCTTY)
    try:
        endpoint.service()
        read_until(fd, b'cmd: ')
        os.write(fd, b'C REMOTE\r')
        endpoint.service(timeout=.1)
        peer.pump()
        for _ in range(6):
            endpoint.service()
        result = read_until(fd, bytes(range(256)))
        assert b'CONNECTED to REMOTE' in result and not host.channel(host.ptchn).rx
        host.arq.on_host_abort()
        for _ in range(4):
            endpoint.service()
        read_until(fd, b'DISCONNECTED fm REMOTE')
        assert host.contact_ready()
    finally:
        os.close(fd)
        endpoint.close()
