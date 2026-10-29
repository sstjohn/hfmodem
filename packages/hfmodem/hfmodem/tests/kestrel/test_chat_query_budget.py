# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Interactive chat has a smaller unanswered-query budget in either role."""
import hashlib
import json
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A
from .test_bw500_query_recovery import _k5fit_close, _drive, clock as clock  # noqa: F401
from .test_va7rmm_nak_recovery import pending as responder


def reply():
    p = Path(__file__).parent / 'fixtures/w1hpc-query-1025'
    meta = json.loads((p / 'provenance.json').read_text())
    assert hashlib.sha256((p / 'reply.wav').read_bytes()).hexdigest() == meta['sha256']
    fs, x = wavfile.read(p / 'reply.wav')
    assert fs == 48000
    return x.astype(float) / 32768


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_the_chat_budget_of_unanswered_queries_closes_without_retransmitting_or_crediting(role, clock):
    h, io = _k5fit_close(clock) if role == 'initiator' else responder(query=False)
    h.chat = True
    before = len(io.sent)
    _drive(h, clock, clock.now + 45)
    assert h.state == A.VaraState.DISCONNECTED, io.msgs
    queries = (h._intermediate_query_attempts if role == 'initiator' else h._answer_queries)
    assert queries == A._CHAT_QUERY_MAX
    assert len(io.sent) == before + A._CHAT_QUERY_MAX + 1  # The queries and the close, no blind DATA.
    assert h.payload_ledger.confirmed == 0 and h._tx_retries == 0


def test_insufficient_reply_and_host_writes_cannot_restart_budget(clock):
    h, io = _k5fit_close(clock)
    h.chat, h.called = True, 'W1HPC'
    x = reply()
    # Erase more than the strong-prefix recovery permits.
    x[48000:] = 0
    original = h._tx_pending
    for n in range(A._CHAT_QUERY_MAX):
        clock.now += 5
        h._heard += 5 * 48000
        assert h._probe_intermediate_answer()
        assert h._intermediate_query_attempts == n + 1
        for a in range(0, len(x), 960):
            chunk = x[a:a+960]
            clock.now += len(chunk) / 48000
            h.on_rx_stream(chunk)
        h.send(b'queued during recovery')
        assert h._tx_pending is original and h.payload_ledger.confirmed == 0
        assert h._tx_retries == 0
    _drive(h, clock, clock.now + 10)
    assert h.state == A.VaraState.DISCONNECTED, io.msgs
    assert h._intermediate_query_attempts == A._CHAT_QUERY_MAX


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_refused_queries_do_not_consume_budget(role, clock):
    h, io = _k5fit_close(clock) if role == 'initiator' else responder(query=False)
    h.chat = True
    io.tx_went_out = lambda: False
    clock.now += 5
    h._heard += 5 * 48000
    send = h._probe_intermediate_answer if role == 'initiator' else h._query_answerer_over
    assert not send()
    assert (h._intermediate_query_attempts if role == 'initiator' else h._answer_queries) == 0
    io.tx_went_out = lambda: True
    assert send()
    assert (h._intermediate_query_attempts if role == 'initiator' else h._answer_queries) == 1


def test_mail_retains_its_existing_recovery_allowance(clock):
    h, _ = _k5fit_close(clock)
    assert not h.chat and h._query_budget() == 18
