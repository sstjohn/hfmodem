# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive obligations survive erased DATA and expire only on protocol events."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF
from .test_setup_query_recovery import setup, padded
from .test_responder_control_stream import feed
from .test_idle_poll import clock  # noqa: F401


def recorded(name):
    p = Path(__file__).parent / 'fixtures/w5jab-recovery-1025'
    meta = json.loads((p / 'provenance.json').read_text())
    entry = next(row for row in meta['slices'] if row['file'] == name + '.wav')
    path = p / entry['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    fs, x = wavfile.read(path)
    assert fs == M.FS
    return x.astype(float) / 32768


def burst(call, kind):
    return padded(M.synth_burst(call, F.for_bw(kind, '500')))


def granted(*, refused=False):
    h, io = setup(caller='W5JAB', marked=True, ask=3)
    h.send(b'local report and hello')
    if refused:
        io.tx_went_out = lambda: False
    feed(h, burst(h.caller, F.SESSION_TURN_REQUEST))
    io.tx_went_out = lambda: True
    return h, io


def is_nak(x, h, short=True):
    kind = F.SESSION_OVER_NAK_RESPONDER_SHORT if short else F.SESSION_OVER_NAK_RESPONDER
    return np.array_equal(x, M.synth_burst(h.called, F.for_bw(kind, '500')))


@pytest.mark.parametrize('block', [960, 4096, 24000])
@pytest.mark.parametrize('loss', ['recorded', 'erased'])
def test_grant_then_lost_data_then_recorded_query_requests_resend(block, loss):
    h, io = granted()
    before, progress, ledger = len(io.sent), h.progress, h.payload_ledger.pending
    feed(h, recorded('post-grant') if loss == 'recorded' else np.zeros(5*M.FS), block)
    assert io.host == []
    # The recorded post-grant window already contains a caller query.
    assert all(is_nak(x, h) for x in io.sent[before:]), io.msgs
    count, retries = len(io.sent), h._reacks
    feed(h, recorded('query'), block)
    assert len(io.sent) == count + 1 and is_nak(io.sent[-1], h, short=retries == 0), io.msgs
    assert h._reacks == retries + 1 and h.progress == progress
    assert h.turn == A._TURN_PEER and h._tx_pending is None
    assert h.payload_ledger.pending == ledger and h.payload_ledger.confirmed == 0
    assert not io.host


def test_good_resend_delivers_once_then_queries_repeat_ack_not_nak():
    h, io = granted()
    feed(h, recorded('query'))
    assert is_nak(io.sent[-1], h)
    payload = b'hi'
    level = A._phy.level_key('500', h._expect_level)
    body = A._phy.vara_body(payload, h.caller, body_len=A._phy.body_size('500', level))
    data = padded(OF.data_over_tx(body, bw='500', level=level))
    feed(h, data)
    assert io.host == [payload], io.msgs
    answer = io.sent[-1].copy()
    assert not is_nak(answer, h) and h._answer_owed == A._OWED_RELEASE
    progress = h.progress
    feed(h, recorded('query'))
    np.testing.assert_array_equal(io.sent[-1], answer)
    assert io.host == [payload] and h.progress == progress
    feed(h, data)
    assert io.host == [payload]
    assert h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('case', ['wrong-call', 'partial', 'our-turn', 'disconnected',
                                  'closing', 'held-answer', 'refused-grant'])
def test_queries_cannot_escape_receive_context(case):
    h, io = granted(refused=case == 'refused-grant')
    if case == 'our-turn': h.turn = A._TURN_OURS
    if case == 'disconnected': h.state = A.VaraState.DISCONNECTED
    if case == 'closing': h._close_asked_at = 1000.
    if case == 'held-answer': h._held_answer = (False, False)
    x = (burst('N0ZZZ', F.SESSION_FINAL_ANSWER_QUERY) if case == 'wrong-call'
         else recorded('query'))
    if case == 'partial': x = x[:45000]
    before = len(io.sent)
    feed(h, x)
    assert len(io.sent) == before and not io.host, io.msgs
    assert h.payload_ledger.confirmed == 0


def test_refused_nak_preserves_missing_data_and_budget():
    h, io = granted()
    io.tx_went_out = lambda: False
    feed(h, recorded('query'))
    assert h._owed_block and h._reacks == 0 and h._rx_resend is None
    io.tx_went_out = lambda: True
    feed(h, recorded('query'))
    assert is_nak(io.sent[-1], h) and h._reacks == 1


def test_missing_data_queries_have_bounded_responses_without_progress():
    h, io = granted()
    progress = h.progress
    for n in range(A._REACK_MAX):
        feed(h, recorded('query'))
        assert h._reacks == n + 1 and h.progress == progress
        assert is_nak(io.sent[-1], h, short=n == 0)
    feed(h, recorded('query'))
    assert h.state != A.VaraState.CONNECTED or h._close_asked_at is not None
    assert not io.host and h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('kind', [None, *F.SESSION_ANSWER_QUERIES_SHORT,
                                  *F.SESSION_INTERMEDIATE_ANSWER_QUERIES])
def test_all_query_encodings_obey_same_missing_data_state(kind):
    h, io = granted()
    x = (padded(M.synth_tone_pairs(F.over_poll(h.called, '500')[0])) if kind is None
         else burst(h.called, kind))
    before = len(io.sent)
    feed(h, x)
    assert len(io.sent) == before + 1 and is_nak(io.sent[-1], h), io.msgs
    assert not io.host and h.payload_ledger.confirmed == 0


def test_new_grant_cannot_replay_previous_delivery_ack():
    h, io = granted()
    h._answer_owed, h._over_answer = A._OWED_RELEASE, ('control', 1)
    h._owed_block = False
    feed(h, burst(h.caller, F.SESSION_TURN_REQUEST))
    assert h._over_answer is None and h._owed_block
    feed(h, recorded('query'))
    assert is_nak(io.sent[-1], h) and not io.host


def test_repeated_ack_queries_are_bounded_and_refusal_costs_no_attempt():
    h, io = granted()
    h._owed_block = False
    h._answer_owed, h._over_answer = A._OWED_RELEASE, ('control', 1)
    progress = h.progress
    io.tx_went_out = lambda: False
    feed(h, recorded('query'))
    assert h._reacks == 0 and h.progress == progress
    io.tx_went_out = lambda: True
    for n in range(A._REACK_MAX):
        feed(h, recorded('query'))
        assert h._reacks == n + 1 and h.progress == progress
        assert not is_nak(io.sent[-1], h)
    feed(h, recorded('query'))
    assert h.state != A.VaraState.CONNECTED or h._close_asked_at is not None
    assert not io.host and h.payload_ledger.confirmed == 0
