# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS 1.7 terminal echo and the unchanged Trimode recovery/handoff."""
from hfmodem.shrike import hostmode as hm
from .test_tracker import setup, connected


def test_power_on_echo_matches_manual_and_masks_nonprinting_input():
    host, _, _, _ = setup()
    assert host.values['E'] == 1
    assert host.feed(b'***\r\x00\x07\x09\x7f\x80') == b'***..\x07\x09..'


def test_actual_recovery_probe_and_explicit_echo_disable_across_byte_splits():
    host, engine, tx, _ = setup()
    host.command('E0')
    assert b'INVALID' not in b''.join(host.feed(bytes([byte])) for byte in b'\x1bE1\r')
    assert host.feed(b'***\r') == b'***.'
    host.feed(b'\x1bE0\r')
    assert host.feed(b'***\r') == b''
    assert not tx and not engine.active and engine.accepted == 0


def test_hostmode_handoff_has_no_terminal_reply_to_poison_the_first_binary_frame():
    host, _, _, _ = setup()
    host.feed(b'\x1bE0\r')
    assert host.feed(b'\x1bJHOST1\r') == b''
    assert host.feed(hm.unwrap(hm.command(1, 'G'))) == b'\x01\x00'


def test_enabled_terminal_echo_never_changes_hostmode_binary_payload_or_reply():
    host, engine, _, _ = setup()
    host.command('E1')
    host.feed(b'\x1bJHOST1\r')
    connected(host, engine)
    original = bytes(range(256))
    wire = hm.unwrap(hm.data(1, original))
    reply = b''.join(host.feed(bytes([byte])) for byte in wire)
    assert reply == b'\x01\x00'
    assert b''.join(engine.pending) == original and engine.delivered == 0
