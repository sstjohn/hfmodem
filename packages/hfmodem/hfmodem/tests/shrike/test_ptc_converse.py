# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS 4.1 converse input, one-command ESC and direction/close controls."""
from hfmodem.shrike.arq import IRS, ISS, State
from hfmodem.shrike.ptc import PtcHost, SimPeer


def setup():
    peer = SimPeer()
    host = PtcHost(peer, mycall='W9SSJ')
    assert b'cmd:' not in host.feed(b'C REMOTE\r')
    peer.pump()
    host.feed(b'')
    assert host.arq.state == State.CONNECTED
    return host, peer


def test_converse_data_is_opaque_except_control_characters_and_has_no_local_echo():
    host, _ = setup()
    original = b'plain\r\n\x00\x1c\x1e\x80\xff'
    assert host.feed(original) == b''
    assert host.arq.payload_ledger.unconfirmed == original
    assert host.sent_total == 0 and not host._line


def test_escape_takes_one_command_across_splits_then_returns_to_converse():
    host, _ = setup()
    result = b''.join(host.feed(bytes([b])) for b in b'\x1bVERSION\r')
    assert b'cmd:' in result and b'PTC-IIIusb' in result
    host.feed(b'payload\n')
    assert host.arq.payload_ledger.unconfirmed == b'payload\n'
    assert b'VERSION' not in host.arq.payload_ledger.unconfirmed


def test_typeahead_survives_setup_then_enters_the_original_payload_ledger():
    peer = SimPeer()
    host = PtcHost(peer, mycall='W9SSJ')
    host.feed(b'C REMOTE\rtyped ahead\r\n')
    assert bytes(host._terminal_pending) == b'typed ahead\r\n' and host.sent_total == 0
    peer.pump()
    host.service_host_output()
    assert not host._terminal_pending
    assert host.arq.payload_ledger.unconfirmed == b'typed ahead\r\n'


def test_failed_setup_and_attachment_loss_retain_typeahead_without_delivery_credit():
    host = PtcHost()
    host.feed(b'C REMOTE\rnot transmitted')
    host.arq.on_host_abort()
    assert host.terminal_retired[-1]['original'] == b'not transmitted'
    assert not host._terminal_pending and host.sent_total == 0
    host.feed(b'')
    host.feed(b'C NEXT\rsecond')
    host.attachment_lost()
    assert host.terminal_retired[-1]['original'] == b'second'
    assert not host._terminal_pending and not host._terminal_converse


def test_control_y_changes_the_correct_direction_and_control_d_closes_after_data():
    host, _ = setup()
    host.arq.role = ISS
    host.feed(b'outgoing\x19')
    assert host.arq._over_pending and not host.arq._breakin_pending
    host.arq._over_pending = False
    host.arq.role = IRS
    host.feed(b'\x19')
    assert host.arq._breakin_pending
    host.arq.role = ISS
    host.feed(b'\x04later')
    assert host.arq._qrt_pending
    assert host.arq.payload_ledger.unconfirmed == b'outgoing'
    assert host.sent_total == 0


def test_typeahead_is_bounded_and_invalid_commands_remain_in_command_input():
    host = PtcHost()
    host.feed(b'C REMOTE\r')
    host.feed(b'x'*65536)
    assert b'*** ERROR: TRANSMIT BUFFER FULL' in host.feed(b'overflow')
    assert len(host._terminal_pending) == 65536 and host.sent_total == 0
    response = host.feed(b'\x1bTERM 5\r')
    assert b'*** ERROR:' in response
    response = host.feed(b'VERSION\r')
    assert b'PTC-IIIusb' in response
    assert len(host._terminal_pending) == 65536


def test_configured_control_characters_apply_before_readback_and_reject_conflicts():
    host = PtcHost(SimPeer())
    for command in (b'BK 2\r', b'CHO 3\r', b'ESC 29\r', b'QRTC 6\r'):
        assert b'*** ERROR:' not in host.feed(command)
    host.feed(b'C REMOTE\r')
    host.peer.pump()
    host.feed(b'')
    host.arq.role = IRS
    host.feed(b'\x02')
    assert host.arq._breakin_pending
    response = host.feed(b'\x1dESC 6\r')
    assert b'*** ERROR:' in response and host.settings['ESCCHR'] == '29'
    assert b'PTC-IIIusb' in host.feed(b'VERSION\r')
    host.arq.role = ISS
    host.feed(b'\x03')
    assert host.arq._over_pending
    host.feed(b'\x06')
    assert host.arq._qrt_pending


def test_terminal_tx_and_delayed_echo_follow_real_logical_peer_acknowledgements():
    host, peer = setup()
    host.feed(b'\x1bTERM 1\r')
    original = b'terminal payload\r\nwith \x00\x1c\x1e\x80\xff'
    assert not host.feed(original) and host.sent_total == 0
    output = bytearray()
    for _ in range(100):
        host.tick()
        output += host.drain_terminal_output()
        if host.sent_total == len(original):
            break
    assert bytes(peer.received) == bytes(output) == original
    assert host.arq.payload_ledger.confirmed == host.sent_total == len(original)


def test_owner_typeahead_flush_follows_protocol_announcement_and_preserves_its_credit():
    host = PtcHost()
    host.enable_owner()
    host.feed(b'C REMOTE\rpretyped')
    host.arq._enter_connected()
    host._setup_bytes = host.p1_setup_payload()
    host.arq.queue_protocol_data(host._setup_bytes)
    host.service_host_output()
    assert bytes(host.arq._outbuf).startswith(host._setup_bytes)
    assert host.arq.payload_ledger.unconfirmed == b'pretyped'
    assert host.arq.payload_ledger.accepted == 8 and host.sent_total == 0
