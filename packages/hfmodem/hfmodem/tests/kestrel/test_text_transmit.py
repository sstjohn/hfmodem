# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Effective TEXT encoding, original-byte settlement and NAK reframing."""
import random

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import compression as C, vara_arq as A
from hfmodem.kestrel.vara.ledger import PayloadLedger
from hfmodem.tests.kestrel.test_data_over_gate import _connected


def sender(monkeypatch, bw='2300'):
    hs, io = _connected(bw=bw)
    hs._start_compression('TEXT')
    hs.turn = A._TURN_OURS
    bodies = []
    def render(blocks, *args):
        bodies.append(tuple(blocks))
        return np.zeros(10)
    monkeypatch.setattr(hs, '_render_over', render)
    return hs, io, bodies


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_coded_partial_overs_keep_the_delivery_open_and_ack_originals(monkeypatch, bw):
    hs, _, emitted = sender(monkeypatch, bw)
    original = b'the radio carries original data and has a VARA encoder.\r\n'*7
    hs.send(original)
    receiver = C.ReceiveCompression('TEXT')
    received = bytearray()
    coded, partial = 0, 0
    for _ in range(100):
        assert hs._tx_pending is not None
        for body in emitted[-1]:
            wire = phy.vara_payload(body, caller=hs.caller, body_len=len(body))
            last = phy.over_is_last(body, hs.caller)
            data, error = receiver.feed(wire, body[-1], last)
            assert error is None
            received += data
            coded += bool(body[-1] & C.TEXT_FLAG)
            partial += bool(body[-1] & C.TEXT_FLAG and not last and len(wire) < len(body)-1)
        assert hs.payload_ledger.confirmed < len(received)
        before = hs.payload_ledger.confirmed
        hs._took_control_burst(continue_reply=True)
        assert hs.payload_ledger.confirmed == len(received) > before
        assert hs.pending_bytes == len(original)-len(received)
        if hs._tx_pending is None:
            break
    assert bytes(received) == original and coded and partial
    assert hs.payload_ledger.confirmed == hs.payload_ledger.accepted == len(original)
    assert hs.payload_ledger.wire_confirmed < len(original)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_incompressible_binary_falls_back_without_losing_any_octet(monkeypatch, bw):
    hs, _, emitted = sender(monkeypatch, bw)
    original = bytes(range(256))*2
    hs.send(original)
    received = bytearray()
    receiver = C.ReceiveCompression('TEXT')
    for _ in range(100):
        for body in emitted[-1]:
            wire = phy.vara_payload(body, caller=hs.caller, body_len=len(body))
            data, error = receiver.feed(wire, body[-1], phy.over_is_last(body, hs.caller))
            assert error is None
            received += data
        hs._took_control_burst(continue_reply=True)
        if hs._tx_pending is None:
            break
    assert bytes(received) == original
    assert hs.payload_ledger.confirmed == len(original)


def test_refused_text_output_never_binds_or_credits_the_attempt(monkeypatch):
    hs, io, _ = sender(monkeypatch)
    io.tx_went_out = lambda: False
    original = b'a'*200
    hs.send(original)
    assert hs._tx_pending is None and b''.join(hs._txq) == original
    assert hs.payload_ledger.wire_pending == hs.payload_ledger.pending == len(original)
    assert hs.payload_ledger.confirmed == 0
    io.tx_went_out = lambda: True
    hs._tx_data_over()
    assert hs._tx_pending[0][-1] & C.TEXT_FLAG
    assert hs.payload_ledger.confirmed == 0 and hs.payload_ledger.unconfirmed == original


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_terminal_text_uses_the_plain_fallback_layout(monkeypatch, bw):
    hs, _, _ = sender(monkeypatch, bw)
    size = phy.body_size(bw, hs._tx_level)
    original = b'a'*(size-4)
    assert len(C.encode_text(original)) < len(original)
    body = hs._tx_body(original, tail=0x89, body_len=size, level=3)
    assert body == phy.vara_body(original, hs.caller, body_len=size, level=3)
    assert not body[-1] & C.TEXT_FLAG and phy.over_is_last(body, hs.caller)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_peer_nak_reencodes_originals_at_the_lower_record(monkeypatch, bw):
    hs, _, emitted = sender(monkeypatch, bw)
    original = b'original text carries data '*6+bytes(range(256))+b'a'*150
    hs.send(original)
    assert hs._tx_pending[0][-1] & C.TEXT_FLAG
    previous_level = hs._tx_pending[2]
    hs._retry_data_over()  # Solicit the measured peer-NAK recovery window.
    hs.send(b'next write')
    hs._took_responder_nak()
    pending = hs._tx_pending
    assert pending[2] < previous_level
    assert hs.payload_ledger.confirmed == 0
    assert hs._tx_payload(pending[0])+b''.join(hs._txq) == original+b'next write'
    assert hs.payload_ledger.unconfirmed == original+b'next write'
    assert hs.payload_ledger.accepted == len(original)+10
    assert hs.pending_bytes == len(original)+10
    assert hs.retire_payload() == original+b'next write'
    assert hs.payload_ledger.confirmed == 0 and hs.payload_ledger.discarded == len(original)+10


@pytest.mark.parametrize('turn', [A._TURN_OURS, A._TURN_PEER])
def test_invalid_text_ends_the_contact_before_a_positive_rf_ack(monkeypatch, turn):
    from hfmodem.tests.kestrel.test_compression import IO
    io = IO()
    io.sent, io.keyed = [], []
    io.tx = io.sent.append
    io.key = io.keyed.append
    hs, _ = _connected(io=io)
    hs.turn = turn
    hs._start_compression('TEXT')
    good = hs._tx_body(b'known original text '*2, tail=0x95, body_len=90)
    bad = bytearray(phy.vara_body(b'\x00', hs.caller))
    bad[-1] |= C.TEXT_FLAG
    monkeypatch.setattr(hs, '_peer_data_over', lambda _: [good, bytes(bad)])
    assert hs._answer_data_over(np.zeros(10))
    assert not io.sent and not io.keyed and hs.state == A.VaraState.DISCONNECTED
    assert b''.join(io.data_received) == b'known original text '*2
    assert len(io.warnings) == 1
    assert hs.payload_ledger.confirmed == 0


def test_wire_binding_is_atomic_and_reframing_never_admits_again():
    ledger = PayloadLedger()
    ledger.queue(b'a'*40+b'b'*20, b'a'*40+b'b'*20)
    ledger.bind_prefix(b'packed', b'a'*40)
    assert ledger.settle(2) == b'' and ledger.confirmed == 0
    with pytest.raises(ValueError, match='partially settled'):
        ledger.bind_prefix(b'new', b'a'*20)
    assert ledger.settle(4) == b'a'*40
    ledger.bind_prefix(b'small', b'b'*20)
    assert ledger.accepted == 60 and ledger.confirmed == 40
    assert ledger.settle(5) == b'b'*20 and ledger.confirmed == 60
    with pytest.raises(ValueError, match='unconfirmed prefix'):
        ledger.bind_prefix(b'bad', b'new')


def test_lost_ack_and_lower_reencoded_frames_deliver_each_original_once(monkeypatch):
    hs, _, emitted = sender(monkeypatch)
    receiver, io = _connected(role='responder')
    receiver._start_compression('TEXT')
    original = b'the original text crosses the link.\r\n'*7+bytes(range(256))
    hs.send(original)
    receiver._deliver(list(emitted[-1]))
    first = b''.join(io.host)
    assert first and first == original[:len(first)]
    assert hs.payload_ledger.confirmed == 0  # The first answer never reached us.
    receiver._expect_resend()
    hs._retry_data_over()
    hs._took_responder_nak()
    for _ in range(100):
        receiver._deliver(list(emitted[-1]))
        hs._took_control_burst(continue_reply=True)
        if hs._tx_pending is None:
            break
    assert b''.join(io.host) == original
    assert hs.payload_ledger.confirmed == len(original)
    assert hs.payload_ledger.pending == 0 and not receiver._rx_compression.damaged


def test_refused_lower_text_retry_keeps_the_preceding_binding(monkeypatch):
    hs, io, _ = sender(monkeypatch)
    original = b'the original text remains pending '*10
    hs.send(original)
    hs._retry_data_over()
    pending, queue = hs._tx_pending, tuple(hs._txq)
    wire = hs.payload_ledger.wire_pending
    io.tx_went_out = lambda: False
    hs._took_responder_nak()
    assert hs._tx_pending == pending and tuple(hs._txq) == queue
    assert hs.payload_ledger.wire_pending == wire
    assert hs.payload_ledger.unconfirmed == original and hs.payload_ledger.confirmed == 0


def test_paired_text_overs_settle_originals_together_and_keep_the_next_prefix(monkeypatch):
    hs, _, emitted = sender(monkeypatch, '500')
    hs._link_high_speed = False
    hs._singles = A._SINGLES_BEFORE_PAIR
    original = b'original text with repeated words '*12
    hs.send(original)
    assert len(emitted[0]) == 2 and hs._tx_pair is not None
    first = b''.join(hs._tx_payload(body) for body in emitted[0])
    assert first == original[:len(first)] and len(first) == 86
    assert all(body[-1] & C.TEXT_FLAG for body in emitted[0])
    assert hs.pending_bytes == len(original) and hs.payload_ledger.confirmed == 0
    hs._took_control_burst(continue_reply=True)
    assert hs.payload_ledger.confirmed == 86
    assert hs.payload_ledger.unconfirmed == original[86:]
    assert hs.pending_bytes == len(original)-86


def test_refused_paired_text_nak_repacking_preserves_every_unconfirmed_original(monkeypatch):
    hs, io, _ = sender(monkeypatch, '500')
    hs._link_high_speed = False
    hs._singles = A._SINGLES_BEFORE_PAIR
    original = b'original text with repeated words '*12
    hs.send(original)
    assert hs._tx_pair is not None
    wire = hs.payload_ledger.wire_pending
    hs._retry_data_over()
    io.tx_went_out = lambda: False
    hs._took_responder_nak()
    assert hs._tx_pair is not None, 'refused retry must retain the original pair'
    bodies = hs._over_bodies(hs._tx_pending[0])
    assert b''.join(hs._tx_payload(body) for body in bodies)+b''.join(hs._txq) == original
    assert hs.payload_ledger.wire_pending == wire
    assert hs.payload_ledger.unconfirmed == original and hs.payload_ledger.confirmed == 0


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_generated_text_and_binary_body_boundaries_roundtrip(monkeypatch, bw):
    hs, _, _ = sender(monkeypatch, bw)
    rng = random.Random(901)
    for level in range(1, 5):
        size = phy.body_size(bw, phy.level_key(bw, level))
        for n in (0, 1, size-3, size-2, size-1):
            for original in (b'a'*n, (b'line\r\n'+rng.randbytes(n))[:n]):
                body = hs._tx_body(original, tail=0x95, body_len=size)
                wire = phy.vara_payload(body, caller=hs.caller, body_len=size)
                received, error = C.ReceiveCompression('TEXT').feed(
                    wire, body[-1], phy.over_is_last(body, hs.caller))
                assert error is None and received == original
                assert phy.over_is_last(body, hs.caller) == (n < size-1)
