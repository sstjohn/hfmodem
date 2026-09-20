# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A diagnostic no-repeat policy leaves unconfirmed bytes pending on closure."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from .test_data_over_gate import _IO
from .test_final_ack_recovery import is_burst, stock_reply, stream


def pending(*, allow=True, payload=None, role='initiator', probe=True):
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='2750',
                                  allow_data_retries=allow, probe_intermediate_query=probe)
    hs.role, hs.caller, hs.called = role, 'W9SSJ', 'KC9GHZ'
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_OURS
    hs.send(payload if payload is not None else b'A' * 89 + b'next')
    assert len(io.sent) == 1 and hs._tx_pending is not None
    return hs, io


def test_default_queries_without_replaying_pending_data():
    assert VA.VaraStationHandshake(['W9SSJ'], _IO()).allow_data_retries is True
    hs, io = pending()
    before = hs._tx_pending, list(hs._txq), hs._over
    hs._retry_data_over()
    assert not np.array_equal(io.sent[0], io.sent[1])
    assert hs._intermediate_query_for == hs._tx_pending
    assert hs._tx_retries == 0 and hs.state == VA.VaraState.CONNECTED
    assert (hs._tx_pending, hs._txq, hs._over) == before


@pytest.mark.parametrize('route', [
    'central', 'central_final_query', 'cadence', 'pending_send', 'idle_response',
])
def test_disabled_retries_cannot_key_data_from_any_entry_route(monkeypatch, route):
    hs, io = pending(allow=False, probe=False)
    before = hs._tx_pending, list(hs._txq), hs._over

    def forbidden(*a, **kw):
        pytest.fail('disabled retransmission reached DATA synthesis')

    monkeypatch.setattr(VA.OF, 'data_over_tx', forbidden)
    actions = {
        'central': hs._retry_data_over,
        'central_final_query': lambda: hs._retry_data_over(final_query=True),
        'cadence': hs.idle_keepalive,
        'pending_send': hs._tx_data_over,
        'nak': hs._took_nak,
        'idle_response': hs._took_idle_response,
    }
    actions[route]()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert (hs._tx_pending, hs._txq, hs._over) == before
    assert hs._tx_retries == 0 and hs._final_query_attempts == 0
    assert len(io.sent) == 2  # Original DATA and the normal close only.
    assert is_burst(io.sent[-1], VF.SESSION_DISCONNECT_REQ)
    assert any('unconfirmed' in msg for msg in io.msgs)
    assert not hs._released and not hs._handed_over and not io.host
    # Re-entering after closure cannot turn this into a retry or a close loop.
    hs._retry_data_over()
    assert len(io.sent) == 2 and hs._tx_pending == before[0]


def test_disabled_retries_preserve_short_final_query_and_stock_handover():
    hs, io = pending(allow=False, payload=b'F' * 79)
    before = hs._tx_pending
    hs._retry_data_over(final_query=True)
    assert hs.state == VA.VaraState.CONNECTED and hs._tx_pending == before
    assert hs._tx_retries == 0 and hs._final_query_attempts == 1
    assert is_burst(io.sent[-1], VF.SESSION_FINAL_ANSWER_QUERY)
    stream(hs, stock_reply())
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert len(io.sent) == 3 and is_burst(io.sent[-1], VF.SESSION_DRAINED)
    assert hs._released and hs._handed_over


def test_short_data_timeout_uses_query_without_a_separate_flag():
    hs, io = pending(allow=False, payload=b'F' * 79)
    before = hs._tx_pending
    hs._retry_data_over()
    assert hs.state == VA.VaraState.CONNECTED and hs._tx_pending == before
    assert hs._final_query_attempts == 1 and hs._tx_retries == 0
    assert len(io.sent) == 2


def test_refused_close_retains_unconfirmed_bytes_and_never_retries_data():
    hs, io = pending(allow=False, probe=False)
    before = hs._tx_pending, list(hs._txq), hs._over
    io.tx_went_out = lambda: False
    hs._retry_data_over()
    assert hs.state == VA.VaraState.DISCONNECTING
    assert (hs._tx_pending, hs._txq, hs._over) == before
    assert hs._tx_retries == 0 and not hs._released and not hs._handed_over
    assert is_burst(io.sent[-1], VF.SESSION_DISCONNECT_REQ)
    sent = len(io.sent)
    hs.idle_keepalive()
    assert len(io.sent) == sent  # Teardown belongs to the driver, no DATA loop.
    io.tx_went_out = lambda: True
    hs.disconnect()
    assert hs.state == VA.VaraState.DISCONNECTED and hs._tx_pending == before[0]


def test_responder_policy_stops_locally_without_inventing_close_waveform():
    hs, io = pending(allow=False, role='responder')
    before = hs._tx_pending
    hs._retry_data_over()
    assert hs.state == VA.VaraState.DISCONNECTED and hs._tx_pending == before
    assert len(io.sent) == 1 and hs._tx_retries == 0


def test_valid_continue_still_keys_next_distinct_body_with_retries_disabled():
    hs, io = pending(allow=False)
    first = hs._tx_pending
    hs._took_control_burst()
    assert hs.state == VA.VaraState.CONNECTED and hs._tx_pending[1] == first[1] + 1
    assert hs._tx_pending[0] != first[0] and len(io.sent) == 2
    assert hs._tx_retries == 0 and not hs._txq
