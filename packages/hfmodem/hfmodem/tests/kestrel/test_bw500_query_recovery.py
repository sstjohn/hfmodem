# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The end of the K5FIT arm of 2026-09-27 04:12Z, generated: our level-5 close
drew an answer that faded, and we disconnected 1.9 s after keying it.

A stock BW500 caller in that place queries instead: 61/331 1.2 s after its
over, 60/683 every 1.75 s after each query's unkey, the close after eighteen,
and a responder that read the over answers the query with its turn-request
[vara_frames.SESSION_ANSWER_QUERIES_SHORT]."""
import numpy as np
import pytest

from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

FS = MK.FS


class _IO(VA.VaraIO):
    def __init__(self):
        self.sent, self.msgs = [], []

    def key(self, on): ...
    def tx(self, samples): self.sent.append(np.asarray(samples, float))
    def tx_went_out(self): return True
    def log(self, msg): self.msgs.append(msg)
    def data(self, payload): ...


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(VA.time, "monotonic", c)
    return c


def _k5fit_close(clock, overs_before=0):
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="500")
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "K5FIT"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    hs._link_high_speed, hs._rec500 = True, 5
    hs._lead = TX.VB6Rnd(VF.lead_state("W9SSJ", False, "500"))
    capacity = VA._phy.body_size("500", 5) - 1
    hs.send(bytes(capacity * overs_before + 79))
    for _ in range(overs_before):
        hs._continue_frame = (hs._tx_pending, 0)
        hs._took_over_continue()
    assert io.msgs[-1].startswith(f"tx DATA over #{overs_before + 1} at record 5 (79 payload")
    return hs, io


def _drive(hs, clock, until):
    """kestrel_connect.mail_session's recovery branches, on the test's clock."""
    while clock.now < until and hs.state is VA.VaraState.CONNECTED:
        clock.now += 0.05
        query, retry = hs.intermediate_query_due_in(), hs.data_retry_due_in()
        if query is not None and query <= 0:
            hs.query_intermediate_answer()
        elif retry is not None and retry <= 0:
            hs.retry_data_over()


def _keyed(io, kind):
    return np.array_equal(io.sent[-1], MK.synth_burst("K5FIT", VF.for_bw(kind, "500")))


def test_an_unanswered_level5_close_is_queried_not_closed(clock):
    hs, io = _k5fit_close(clock)
    keyed = clock.now
    _drive(hs, clock, keyed + 1.9)
    assert hs.state is VA.VaraState.CONNECTED, io.msgs
    assert _keyed(io, VF.SESSION_ANSWER_QUERIES_SHORT[0])
    assert any("intermediate query 1/18 (61/331)" in m for m in io.msgs)
    assert hs._tx_pending[2] == 5


def test_the_queries_follow_stock_and_close_after_eighteen(clock, monkeypatch):
    hs, io = _k5fit_close(clock)
    monkeypatch.setattr(VA.OF, "data_over_tx", lambda *a, **k: pytest.fail("repeated DATA"))
    keyed_at = []
    send = hs._send_burst

    def timed(kind):
        keyed_at.append((clock.now, kind.name))
        return send(kind)
    hs._send_burst = timed
    start = clock.now
    _drive(hs, clock, start + 90)
    queries = [(t - start, n) for t, n in keyed_at if "query" in n]
    assert len(queries) == VA._BW500_QUERY_MAX
    assert queries[0][1] == "session-answer-query-short-a"
    assert {n for _, n in queries[1:]} == {"session-intermediate-answer-query-a"}
    assert queries[0][0] == pytest.approx(1.2, abs=0.06)
    assert np.allclose(np.diff([t for t, _ in queries]), 1.75, atol=0.06)
    assert hs.state is VA.VaraState.DISCONNECTED
    assert keyed_at[-1][1] == VF.SESSION_DISCONNECT_REQ.name


def test_the_second_over_of_a_delivery_asks_on_the_other_phase(clock):
    hs, io = _k5fit_close(clock, overs_before=1)
    _drive(hs, clock, clock.now + 1.3)
    assert _keyed(io, VF.SESSION_ANSWER_QUERIES_SHORT[1])
    _drive(hs, clock, clock.now + 2.6)
    assert _keyed(io, VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[1])


def test_the_turn_request_behind_our_query_says_the_close_was_read(clock):
    hs, io = _k5fit_close(clock)
    _drive(hs, clock, clock.now + 1.3)
    assert _keyed(io, VF.SESSION_ANSWER_QUERIES_SHORT[0])
    ask = MK.synth_burst("W9SSJ", VF.for_bw(VF.SESSION_TURN_REQUEST_RESPONDER, "500"))
    x = np.concatenate([np.zeros(FS // 10), ask / np.abs(ask).max() / 2, np.zeros(FS // 2)])
    for at in range(0, len(x), 960):
        clock.now += 960 / FS
        hs.on_rx_stream(x[at:at + 960])
    assert any("behind our final over" in m for m in io.msgs), io.msgs
    assert hs._tx_pending is None and hs.state is VA.VaraState.CONNECTED
