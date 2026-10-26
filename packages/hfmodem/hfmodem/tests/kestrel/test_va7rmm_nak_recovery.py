# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A recorded caller NAK must recover lost DATA without crediting delivery."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_responder_control_stream import station, feed
from .test_idle_poll import clock, _Clock  # noqa: F401


def pending(query=True, final=True):
    h, io = station('W9SSJ')
    h.caller, h.chat, h.turn = 'VA7RMM', True, A._TURN_OURS
    h._link_high_speed, h._rec500 = True, 5
    h._lead = A.VB6Rnd(F.lead_state(h.caller, True, '500'))
    h._start_compression('OFF')
    h.send(b'how are you today\n' if final else b'x' * 200)
    if query:
        assert h._query_answerer_over()
    return h, io


def recorded():
    p = Path(__file__).parent / 'fixtures/va7rmm-nak-1025'
    meta = json.loads((p / 'provenance.json').read_text())
    assert hashlib.sha256((p / 'reply.wav').read_bytes()).hexdigest() == meta['sha256']
    rate, x = wavfile.read(p / 'reply.wav')
    assert rate == M.FS
    return x.astype(float) / 32768


def generated(call='W9SSJ', kind=F.SESSION_OVER_NAK):
    return np.r_[np.zeros(4800), M.synth_burst(call, F.for_bw(kind, '500')), np.zeros(4800)]


@pytest.mark.parametrize('block', [960, 4096, 24000])
@pytest.mark.parametrize('final', [True, False])
def test_recorded_nak_resends_lower_without_settling_and_cannot_replay(block, final):
    h, io = pending(final=final)
    count, old, ledger = len(io.sent), h._tx_pending, h.payload_ledger.pending
    x = recorded()
    feed(h, x, block)
    assert len(io.sent) == count + 1, io.msgs
    assert h._tx_retries == 1 and h._tx_pending[2] < old[2]
    assert h._tx_pending[1] == old[1]
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == ledger
    assert h._answer_query_for is None
    feed(h, x, block)
    assert len(io.sent) == count + 1 and h._tx_retries == 1
    assert h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('case', ['no-query', 'refused-query', 'stale-wall', 'stale-samples',
    'different-pending', 'wrong-call', 'wrong-role', 'wrong-turn', 'disconnected',
    'truncated', 'corrupt-tail', 'wrong-family'])
def test_no_retry_outside_complete_solicited_current_reply(case):
    h, io = pending(query=case not in ('no-query', 'refused-query'))
    if case == 'refused-query':
        io.tx_went_out = lambda: False
        assert not h._query_answerer_over()
        io.tx_went_out = lambda: True
    if case == 'stale-wall': _Clock.t += 3
    if case == 'stale-samples': h._answer_query_samples = 3 * M.FS
    if case == 'different-pending': h._tx_pending = tuple(list(h._tx_pending))
    if case == 'wrong-role': h.role = 'initiator'
    if case == 'wrong-turn': h.turn = A._TURN_PEER
    if case == 'disconnected': h.state = A.VaraState.DISCONNECTED
    x = generated('N0ZZZ' if case == 'wrong-call' else 'W9SSJ',
                  F.SESSION_OVER_NAK_RESPONDER if case == 'wrong-family' else F.SESSION_OVER_NAK)
    if case == 'truncated': x = x[:4800 + 29 * M.HOP]
    if case == 'corrupt-tail':
        tones = F.handshake_tones('W9SSJ', F.for_bw(F.SESSION_OVER_NAK, '500'))
        tones[-1] = (tones[-1] + 3) % 80
        x = np.r_[np.zeros(4800), M.synth_tones(tones), np.zeros(4800)]
    old, count = h._tx_pending, len(io.sent)
    feed(h, x)
    assert h._tx_retries == 0 and h._tx_pending is old, io.msgs
    assert len(io.sent) == count and h.payload_ledger.confirmed == 0


def test_recorded_reply_waits_for_complete_tail():
    h, io = pending()
    count = len(io.sent)
    x = recorded()
    feed(h, x[:48000])
    assert len(io.sent) == count and h._tx_retries == 0
    feed(h, x[48000:])
    assert len(io.sent) == count + 1 and h._tx_retries == 1


def test_refused_retry_preserves_data_and_requires_another_query():
    h, io = pending()
    old, ledger = h._tx_pending, h.payload_ledger.pending
    io.tx_went_out = lambda: False
    feed(h, recorded())
    assert h._tx_pending is old and h._tx_retries == 0
    assert h.payload_ledger.pending == ledger and h.payload_ledger.confirmed == 0
    count = len(io.sent)
    io.tx_went_out = lambda: True
    feed(h, recorded())
    assert len(io.sent) == count
    assert h._query_answerer_over()
    feed(h, recorded())
    assert h._tx_retries == 1 and h.payload_ledger.confirmed == 0


def test_repeated_missing_data_remains_bounded():
    h, io = pending()
    for n in range(A._OVER_RETRY_MAX):
        if n:
            assert h._query_answerer_over()
        feed(h, generated())
        assert h._tx_retries == n + 1, io.msgs
        assert h.payload_ledger.confirmed == 0
    assert h._query_answerer_over()
    feed(h, generated())
    assert h._tx_retries == A._OVER_RETRY_MAX
    assert any('retry budget exhausted' in m for m in io.msgs)
    assert h.payload_ledger.confirmed == 0


def test_retried_message_is_settled_only_when_caller_acknowledges():
    h, io = pending()
    total = h.payload_ledger.pending
    feed(h, recorded())
    assert h._tx_retries == 1 and h.payload_ledger.confirmed == 0
    assert A._phy.over_is_last(h._tx_pending[0], h.caller)
    ack = M.synth_tone_pairs(F.control_bursts(h.caller, '500', 1)[0])
    feed(h, np.r_[np.zeros(4800), ack, np.zeros(12000)])
    assert h._tx_pending is None, io.msgs
    assert h.payload_ledger.confirmed == total and h.payload_ledger.pending == 0
    assert h.turn == A._TURN_PEER
