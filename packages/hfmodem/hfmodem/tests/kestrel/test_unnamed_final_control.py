# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fake detector-result reproducer; no waveform synthesis or DSP.

Pins the production stream dispatch after a shared control preamble was found
but neither complete named tail was identified. Does not claim a particular
recording produced these detector results.
"""
import numpy as np
import pytest
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.tests.kestrel.test_bw500_query_recovery import _IO, clock as clock  # noqa: F401


@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('named', [None, 0, 1])
@pytest.mark.parametrize('chat', [True, False])
@pytest.mark.parametrize('record', [2, 4])
def test_final_control_delivery_requires_named_chat_tail(monkeypatch, clock, record, route, named, chat):
    hs = VA.VaraStationHandshake(['W9SSJ'], _IO(), bw='500', chat=chat)
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'K6HN'
    hs.state, hs.turn = VA.VaraState.CONNECTED, VA._TURN_OURS
    hs._link_high_speed = True
    payload = b'7 <Q:-75>'
    body = VA._phy.vara_body(payload, hs.caller,
                             body_len=VA._phy.body_size('500', record))
    hs._tx_pending = (body, 1, record)
    hs.payload_ledger.queue(payload, payload)
    assert VA._phy.over_is_last(body, hs.caller)
    if record == 2:
        hs._intermediate_query_for = hs._tx_pending
    hs._pending_answer_at = clock.now
    pending = hs._tx_pending
    monkeypatch.setattr(hs, '_cached_control', lambda *a, **k: None)
    monkeypatch.setattr(VA, '_ack_plateau', lambda *a, **k: VA._ACK_PLATEAU)
    # No competing identified frame. Keep production _peer_control_burst,
    # _chat_tail_due, dispatch, and host settlement intact.
    for name in ('_peer_receiver_close', '_peer_short_close', '_peer_data_nak',
                 '_peer_query_turn_idle', '_peer_intermediate_query_answer',
                 '_peer_low_record_answer', '_peer_over_continue', '_held_continue',
                 '_peer_responder_nak', '_peer_idle_response', '_peer_ident',
                 '_peer_responder_idle', '_peer_wants_turn'):
        monkeypatch.setattr(hs, name, lambda *a, **k: False)
    # All hard tail scores fail (including a wrong-call tail); a named
    # soft result remains acceptable, preserving weak known controls.
    monkeypatch.setattr(VA, '_burst_match', lambda *a: 0.0)
    monkeypatch.setattr(hs, '_soft_control_frame', lambda *a: named)
    monkeypatch.setattr(hs, '_ask_if_owed', lambda: None)
    monkeypatch.setattr(hs, '_answer_early', lambda: 0)
    monkeypatch.setattr(hs, '_a_gap', lambda: True)
    releases = []
    monkeypatch.setattr(hs, '_release_turn', lambda *a: releases.append(a) or True)
    samples = np.zeros(VA._SESSION_NEED)
    query = hs._intermediate_query_for
    deadline = hs.intermediate_query_due_in()
    assert deadline is not None
    progress = (hs.progress, hs._progress_at, hs._since_progress)
    if route == 'stream':
        hs._ans_due = 0
        hs._stream_answer(samples)
        if chat and named is None:
            assert hs._tx_pending == pending  # first scan waits for the tail
            hs._ans_due = 0
            assert not hs._stream_answer(np.zeros(8 * VA.MK.HOP))
            assert len(hs._ans_buf) > 0  # no reset/consumption on rejection
    else:
        hs.on_rx_audio(samples)
    if chat and named is None:
        assert hs._tx_pending == pending
        assert hs.payload_ledger.pending == len(payload)
        assert hs._intermediate_query_for == query
        assert hs._tx_retries == 0
        assert hs.intermediate_query_due_in() == deadline
        assert (hs.progress, hs._progress_at, hs._since_progress) == progress
        assert not releases
        # An unresolved scan must not poison a subsequent named weak ACK.
        monkeypatch.setattr(hs, '_soft_control_frame', lambda *a: 0)
        hs._ans_buf = np.empty(0)
        hs._ans_due = 0
        hs._stream_answer(samples)
    assert hs._tx_pending is None
    assert hs.payload_ledger.pending == 0
    assert len(releases) == 1
