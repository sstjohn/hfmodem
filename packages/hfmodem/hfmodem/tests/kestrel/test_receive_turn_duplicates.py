# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Duplicate receive windows belong to one peer turn, not the entire contact."""
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as A
from .test_data_over_gate import _connected
from .test_compression import commit_bodies


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_identical_message_after_completed_handover_is_new_data(role, bw):
    h, io = _connected(bw=bw, role=role)
    message = b'same message\r'
    body = phy.vara_body(message, h.caller, body_len=phy.body_size(bw, phy.base_level(bw)))
    commit_bodies(h, [body])
    commit_bodies(h, [body])
    assert io.host == [message] * 2
    if role == 'initiator':
        h._took_turn_grant()
    else:
        h._take_caller_release()
    assert h.turn == A._TURN_OURS
    assert h._release_turn()
    assert h.turn == A._TURN_PEER
    commit_bodies(h, [body])
    assert io.host == [message] * 3
    commit_bodies(h, [body])
    assert io.host == [message] * 4


def test_asking_for_turn_does_not_forget_unconfirmed_peer_delivery():
    h, io = _connected(bw='500')
    body = phy.vara_body(b'repeat', h.caller, body_len=phy.body_size('500', 4))
    commit_bodies(h, [body])
    h._tx_turn_request()
    assert h.turn == A._TURN_ASKED
    commit_bodies(h, [body])
    assert io.host == [b'repeat', b'repeat']


def test_completed_handover_retires_repacketized_replay_obligation():
    h, io = _connected(bw='500')
    body = phy.vara_body(b'abcdef', h.caller, body_len=7, tail=0x81)
    commit_bodies(h, [body])
    h._rx_query(query_start=h._unkeyed_at + 60)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert (h._rx_window.T.start, h._rx_window.T.frontier) == (0, h._rx_window.S)
    h._took_turn_grant()
    h._release_turn()
    # A new message can begin with the previous message without replaying it.
    commit_bodies(h, [phy.vara_body(b'abcdef new', h.caller, body_len=phy.body_size('500', 4))])
    assert io.host == [b'abcdef', b'abcdef new']


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_equal_payload_with_advanced_frame_field_is_not_a_replay(bw):
    h, io = _connected(bw=bw)
    size = phy.body_size(bw, phy.base_level(bw))
    payload = b'A' * (size - 1)
    first = phy.vara_body(payload, h.caller, body_len=size, tail=0x95)
    second = phy.vara_body(payload, h.caller, body_len=size, tail=0x91)
    commit_bodies(h, [first])
    commit_bodies(h, [second])
    assert b''.join(io.host) == payload * 2
    commit_bodies(h, [second])
    assert b''.join(io.host) == payload * 3


def test_explicit_resend_still_suppresses_payload_with_rewritten_field():
    h, io = _connected(bw='500')
    first = phy.vara_body(b'A' * 9, h.caller, body_len=10, tail=0x95)
    retry = phy.vara_body(b'A' * 9, h.caller, body_len=10, tail=0x89)
    commit_bodies(h, [first])
    h._rx_query(query_start=h._unkeyed_at + 60)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert (h._rx_window.T.start, h._rx_window.T.frontier) == (0, h._rx_window.S)
    commit_bodies(h, [retry])
    assert io.host == [b'A' * 9]


def test_lower_record_retry_retains_payload_based_duplicate_check():
    h, io = _connected(bw='500')
    commit_bodies(h, [phy.vara_body(b'hello', h.caller, body_len=44)])
    commit_bodies(h, [phy.vara_body(b'hello', h.caller, body_len=10)])
    assert io.host == [b'hello', b'hello']


def test_lower_record_resend_after_aired_nak_matches_payload():
    h, io = _connected(bw='500')
    commit_bodies(h, [phy.vara_body(b'hello', h.caller, body_len=6, tail=0x81)])
    h._rx_query(query_start=h._unkeyed_at + 60)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert (h._rx_window.T.start, h._rx_window.T.frontier) == (0, 5)
    commit_bodies(h, [phy.vara_body(b'hello', h.caller, body_len=10)])
    assert io.host == [b'hello']
