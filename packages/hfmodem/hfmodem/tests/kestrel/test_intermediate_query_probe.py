# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An opt-in, one-shot query cannot turn uncertain feedback into a DATA ACK."""
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from .test_data_over_gate import _IO
from .test_final_ack_recovery import is_burst, stock_reply, stream
from .test_head_cut_continue import recorded


def pending(*, probe=True, bw='2750', payload=None):
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw=bw,
                                  allow_data_retries=False,
                                  probe_intermediate_query=probe)
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'KC9GHZ'
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    hs.send(payload if payload is not None else b'A' * 89 + b'B' * 89 + b'last')
    return hs, io


def test_queries_are_enabled_by_default_independently_of_legacy_retry_flag():
    for allow in (False, True):
        hs = VA.VaraStationHandshake(['W9SSJ'], _IO(), allow_data_retries=allow)
        assert hs.probe_intermediate_query


def test_no_probe_without_explicit_opt_in():
    hs, io = pending(probe=False)
    hs._retry_data_over(final_query=True)
    assert hs.state == VA.VaraState.DISCONNECTED and not hs._intermediate_query_attempted
    assert len(io.sent) == 2 and is_burst(io.sent[-1], VF.SESSION_DISCONNECT_REQ)


def test_one_query_has_exact_known_waveform_and_retains_all_pending_state():
    hs, io = pending()
    before = hs._tx_pending, list(hs._txq), hs._over, hs.progress
    hs._retry_data_over(final_query=True)
    assert hs.state == VA.VaraState.CONNECTED and hs._intermediate_query_attempted
    assert hs._intermediate_query_for == before[0]
    assert (hs._tx_pending, hs._txq, hs._over, hs.progress) == before
    assert hs._tx_retries == 0 and hs._final_query_attempts == 0
    assert len(io.sent) == 2 and is_burst(io.sent[-1], VF.SESSION_FINAL_ANSWER_QUERY)
    assert not hs._released and not hs._handed_over


@pytest.mark.parametrize('route', ['central', 'cadence', 'pending_send', 'idle_response'])
def test_every_later_retry_closes_without_requery_or_data(monkeypatch, route):
    hs, io = pending()
    hs._retry_data_over(final_query=True)
    before = hs._tx_pending, list(hs._txq), hs._over
    monkeypatch.setattr(VA.OF, 'data_over_tx', lambda *a, **kw: pytest.fail('repeated DATA'))
    action = {'central': hs._retry_data_over, 'cadence': hs.idle_keepalive,
              'nak': hs._took_nak, 'pending_send': hs._tx_data_over,
              'idle_response': hs._took_idle_response}[route]
    for _ in range(VA._FINAL_QUERY_MAX):
        hs._intermediate_query_at -= 4.6
        action()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert (hs._tx_pending, hs._txq, hs._over) == before
    assert len(io.sent) == 2 + VA._FINAL_QUERY_MAX
    assert is_burst(io.sent[-1], VF.SESSION_DISCONNECT_REQ)
    hs._retry_data_over()
    assert len(io.sent) == 2 + VA._FINAL_QUERY_MAX and hs._tx_retries == 0


@pytest.mark.parametrize('via_stream', [False, True])
def test_only_validated_continue_advances_and_new_body_gets_own_query(via_stream):
    hs, io = pending()
    first = hs._tx_pending
    hs._retry_data_over(final_query=True)
    x = recorded('clean-caller-over2-stock-continue')
    if via_stream:
        stream(hs, x)
    else:
        hs.on_rx_audio(x)
    assert hs.state == VA.VaraState.CONNECTED and len(io.sent) == 3
    assert hs._tx_pending[1] == first[1] + 1 and hs._tx_pending[0] != first[0]
    assert hs._intermediate_query_for is None and not hs._intermediate_query_attempted
    second = hs._tx_pending
    hs.idle_keepalive()
    assert hs.state == VA.VaraState.CONNECTED and len(io.sent) == 4
    assert hs._intermediate_query_for == second and hs._tx_pending == second
    assert is_burst(io.sent[-1], VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[1])
    for _ in range(VA._FINAL_QUERY_MAX):
        hs._intermediate_query_at -= 4.6
        hs.idle_keepalive()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert is_burst(io.sent[-1], VF.SESSION_DISCONNECT_REQ)
    assert hs._tx_retries == 0


def test_turn_requests_and_generic_control_do_not_retire_or_send_drained():
    hs, io = pending()
    before = hs._tx_pending, list(hs._txq), hs._over, hs.progress
    assert not hs._took_turn_request(raw=True)  # Unsolicited before probe.
    assert len(io.sent) == 1
    hs._retry_data_over(final_query=True)
    stream(hs, stock_reply())  # Native request that settles short-final recovery.
    assert len(io.sent) == 2
    hs._took_control_burst()  # Generic final ACK / over-answer transition.
    assert (hs._tx_pending, hs._txq, hs._over, hs.progress) == before
    assert hs.state == VA.VaraState.CONNECTED and len(io.sent) == 2
    assert hs._final_ack_confirmed is None and hs._final_query_attempts == 0
    assert not hs._released and not hs._handed_over
    hs.idle_keepalive()
    assert hs.state == VA.VaraState.CONNECTED and hs._tx_pending == before[0]


def test_refused_query_retains_pending_and_does_not_spend_transmit_budget():
    hs, io = pending()
    before = hs._tx_pending
    io.tx_went_out = lambda: False
    hs._retry_data_over(final_query=True)
    assert hs._intermediate_query_attempted and hs._tx_pending == before
    assert hs.state == VA.VaraState.CONNECTED and len(io.sent) == 2
    assert hs._intermediate_query_attempts == 0
    assert hs._intermediate_query_for is None
    io.tx_went_out = lambda: True
    hs._retry_data_over()
    assert len(io.sent) == 3 and is_burst(io.sent[-1], VF.SESSION_FINAL_ANSWER_QUERY)
    assert hs.state == VA.VaraState.CONNECTED and hs._tx_retries == 0


@pytest.mark.parametrize('field,value', [('bw', '2300'), ('role', 'responder'),
                                        ('turn', VA._TURN_PEER), ('_txq', [])])
def test_query_requires_the_measured_shape_and_sending_state(field, value):
    hs, io = pending()
    setattr(hs, field, value)
    assert not hs._probe_intermediate_answer()
    assert not hs._intermediate_query_attempted and len(io.sent) == 1


def test_short_frame_with_more_queue_does_not_qualify():
    hs, io = pending(payload=b'short')
    hs._txq = [b'next']
    hs._retry_data_over(final_query=True)
    assert hs.state == VA.VaraState.DISCONNECTED and not hs._intermediate_query_attempted
    assert len(io.sent) == 2


def test_eligible_short_final_recovery_remains_the_validated_exchange():
    hs, io = pending(payload=b'F' * 79)
    hs._retry_data_over(final_query=True)
    assert hs._final_query_attempts == 1 and not hs._intermediate_query_attempted
    stream(hs, stock_reply())
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert len(io.sent) == 3 and is_burst(io.sent[-1], VF.SESSION_DRAINED)
