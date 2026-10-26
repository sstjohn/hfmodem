# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""L flow control follows actual RF offers, not merely accepted host writes."""
import math

import pytest
from hfhost import ptc

from hfmodem.shrike import arq, hostmode
from hfmodem.tests.shrike.test_ptc_compatibility import connected


def status(client):
    reply = client.cmd('L')
    assert reply.code == ptc.MSG
    return tuple(map(int, reply.text.split()))


def prepare(monkeypatch, payload=b'queued payload '*20, *, refuse=False):
    host, client = connected()
    for start in range(0, len(payload), 128):
        assert client.exchange(payload[start:start+128], False).code == ptc.OK
    sent = []
    def transmit(sl, data, control, breakin=False):
        if refuse:
            return arq.REFUSED
        sent.append(bytes(data))
        return len(data)
    monkeypatch.setattr(host, 'send_packet', transmit)
    host.arq.role = arq.ISS
    return host, client, sent


def test_accepted_but_untransmitted_bytes_are_not_also_reported_unacknowledged(monkeypatch):
    host, client, sent = prepare(monkeypatch)
    assert status(client)[2:5] == (math.ceil(host._txbuf/hostmode.MAX_DATA), 0, 0)
    assert not sent and host.sent_total == 0


@pytest.mark.parametrize('payload', [b'queued payload '*20, bytes(range(256))*2, b'\x1c'*300])
def test_successful_offer_moves_only_that_field_from_queued_to_unacknowledged(monkeypatch, payload):
    host, client, sent = prepare(monkeypatch, payload)
    host.arq._start_next_packet()
    assert sent and host.arq._inflight.sent_at_level == 1
    expected = math.ceil((host._txbuf-len(sent[0]))/hostmode.MAX_DATA)
    assert status(client)[2:5] == (expected, 1, 0)
    assert host.sent_total == host.arq.payload_ledger.confirmed == 0


def test_crc_retry_keeps_the_reply_snapshot_when_rf_state_advances(monkeypatch):
    host, client, _ = prepare(monkeypatch)
    queued = client.cmd('L')
    assert tuple(map(int, queued.text.split()))[3] == 0
    host.arq._start_next_packet()
    assert client.retry().data == queued.data
    assert status(client)[3] == 1


def test_refused_offer_stays_unsent_and_has_no_rf_retry_count(monkeypatch):
    host, client, sent = prepare(monkeypatch, refuse=True)
    initial = status(client)
    host.arq._start_next_packet()
    assert host.arq._inflight is not None and not host.arq._inflight.sent_at_level
    assert status(client) == initial and not sent
    assert host.arq.payload_ledger.confirmed == 0


def test_retry_count_changes_without_moving_or_crediting_the_offered_originals(monkeypatch):
    host, client, sent = prepare(monkeypatch)
    host.arq._start_next_packet()
    original = host.arq.payload_ledger.unconfirmed
    before = status(client)
    host.arq.on_cycle()
    assert len(sent) == 2 and sent[0] == sent[1]
    after = status(client)
    assert after[2:4] == before[2:4] and after[4] == 1
    assert host.arq.payload_ledger.unconfirmed == original and host.sent_total == 0


def test_refused_retransmission_does_not_erase_the_previous_successful_offer(monkeypatch):
    host, client, _ = prepare(monkeypatch)
    host.arq._start_next_packet()
    before = status(client)
    monkeypatch.setattr(host, 'send_packet', lambda *a, **kw: arq.REFUSED)
    host.arq.on_cycle()
    assert host.arq._refused_burst and status(client)[2:4] == before[2:4]
    assert host.sent_total == 0


@pytest.mark.parametrize('refuse', [False, True])
def test_purge_preserves_an_offered_field_with_its_actual_transmit_disposition(monkeypatch, refuse):
    host, client, sent = prepare(monkeypatch, refuse=refuse)
    host.arq._start_next_packet()
    field = host.arq._inflight.payload
    assert client.cmd('#CL').code == ptc.OK
    assert not host.arq._outbuf and host.sent_total == 0
    assert status(client)[2:4] == ((math.ceil(len(field)/hostmode.MAX_DATA), 0)
                                  if refuse else (0, 1))
    assert host.arq.payload_ledger.unconfirmed


def test_complete_peer_settlement_clears_both_data_fields_before_reuse(monkeypatch):
    host, client, sent = prepare(monkeypatch, b'short')
    host.arq._start_next_packet()
    assert sent == [b'short'] and status(client)[2:4] == (0, 1)
    host.arq._settle_inflight()
    assert status(client)[2:5] == (0, 0, 0)
    assert host.sent_total == 5 and not host.arq.payload_ledger.pending
    host.arq._start_next_packet()  # An idle protocol packet is not host data.
    assert host.arq._inflight is not None and not host.arq._inflight.payload
    assert status(client)[2:5] == (0, 0, 0)


def test_partial_escape_settlement_cannot_make_a_pending_original_disappear(monkeypatch):
    host, client, sent = prepare(monkeypatch, b'\x1c')
    host.arq.payload_bytes_override = 1
    host.arq._start_next_packet()
    assert len(sent[0]) == 1 and status(client)[2:4] == (1, 1)
    host.arq._settle_inflight()
    assert host.sent_total == 0 and host.arq.payload_ledger.pending == 1
    assert status(client)[2:4] == (1, 0)
    host.arq._start_next_packet()
    assert status(client)[2:4] == (0, 1)
    host.arq._settle_inflight()
    assert host.sent_total == 1 and status(client)[2:4] == (0, 0)
