# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Duplicate receive windows belong to one peer turn, not the entire contact."""
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as A
from .test_data_over_gate import _connected


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_identical_message_after_completed_handover_is_new_data(role, bw):
    h, io = _connected(bw=bw, role=role)
    message = b'same message\r'
    body = phy.vara_body(message, h.caller, body_len=phy.body_size(bw, phy.base_level(bw)))
    h._deliver([body])
    h._deliver([body])  # Lost ACK: still the same peer turn.
    assert io.host == [message]
    if role == 'initiator':
        h._took_turn_grant()
    else:
        h._take_caller_release()
    assert h.turn == A._TURN_OURS
    assert h._release_turn()
    assert h.turn == A._TURN_PEER
    h._deliver([body])
    assert io.host == [message, message]
    h._deliver([body])
    assert io.host == [message, message]


def test_asking_for_turn_does_not_forget_unconfirmed_peer_delivery():
    h, io = _connected(bw='500')
    body = phy.vara_body(b'repeat', h.caller, body_len=phy.body_size('500', 4))
    h._deliver([body])
    h._tx_turn_request()
    assert h.turn == A._TURN_ASKED
    h._deliver([body])
    assert io.host == [b'repeat']


def test_completed_handover_retires_repacketized_replay_obligation():
    h, io = _connected(bw='500')
    body = phy.vara_body(b'abcdef', h.caller, body_len=phy.body_size('500', 4))
    h._deliver([body])
    h._expect_resend()
    h._took_turn_grant()
    h._release_turn()
    # A new message can begin with the previous message without replaying it.
    h._deliver([phy.vara_body(b'abcdef new', h.caller, body_len=phy.body_size('500', 4))])
    assert io.host == [b'abcdef', b'abcdef new']


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_equal_payload_with_advanced_frame_field_is_not_a_replay(bw):
    h, io = _connected(bw=bw)
    size = phy.body_size(bw, phy.base_level(bw))
    payload = b'A' * (size - 1)
    first = phy.vara_body(payload, h.caller, body_len=size, tail=0x95)
    second = phy.vara_body(payload, h.caller, body_len=size, tail=0x91)
    h._deliver([first])
    h._deliver([second])
    assert b''.join(io.host) == payload * 2
    h._deliver([second])
    assert b''.join(io.host) == payload * 2


def test_explicit_resend_still_suppresses_payload_with_rewritten_field():
    h, io = _connected(bw='500')
    first = phy.vara_body(b'A' * 9, h.caller, body_len=10, tail=0x95)
    retry = phy.vara_body(b'A' * 9, h.caller, body_len=10, tail=0x89)
    h._deliver([first])
    h._expect_resend()
    h._deliver([retry])
    assert io.host == [b'A' * 9]


def test_lower_record_retry_retains_payload_based_duplicate_check():
    h, io = _connected(bw='500')
    h._deliver([phy.vara_body(b'hello', h.caller, body_len=44)])
    h._deliver([phy.vara_body(b'hello', h.caller, body_len=10)])
    assert io.host == [b'hello']
