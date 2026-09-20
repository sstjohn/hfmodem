# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Missing replies solicit state on a bounded clock; they never replay DATA."""
import numpy as np
import pytest
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_no_data_retries import pending


class _Clock:
    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(VA.time, 'monotonic', lambda: self.now)

    def __iadd__(self, dt):
        self.now += dt
        return self


@pytest.fixture
def clock(monkeypatch):
    return _Clock(monkeypatch)


def test_a_fresh_over_is_queried_on_the_measured_interval(clock):
    hs, io = pending()
    old = hs._tx_pending
    assert hs.data_retry_due_in() is None
    assert hs.intermediate_query_due_in() == pytest.approx(VA._OVER_RETRY_S)
    clock += VA._OVER_RETRY_S - .2
    assert not hs.query_intermediate_answer()
    clock += .21
    assert hs.query_intermediate_answer()
    assert hs._tx_pending == old and hs._tx_retries == 0
    np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_FINAL_ANSWER_QUERY, hs.bw)))


def test_lost_query_answers_repeat_the_query_to_a_finite_budget(clock):
    hs, io = pending()
    old = hs._tx_pending
    clock += VA._OVER_RETRY_S + .01
    for n in range(1, VA._FINAL_QUERY_MAX + 1):
        assert hs.query_intermediate_answer()
        assert hs._intermediate_query_attempts == n
        assert hs._tx_pending == old and hs._tx_retries == 0
        assert not hs.query_intermediate_answer()
        clock += 4.51
    assert not hs.query_intermediate_answer()
    assert hs.state is VA.VaraState.DISCONNECTED and hs._tx_pending == old
    assert len(io.sent) == 2 + VA._FINAL_QUERY_MAX
    for query in io.sent[1:-1]:
        np.testing.assert_array_equal(query, io.sent[1])


def test_refused_query_preserves_budget_and_retries_after_its_window(clock):
    hs, io = pending()
    io.tx_went_out = lambda: False
    clock += VA._OVER_RETRY_S + .01
    assert not hs.query_intermediate_answer()
    assert hs._intermediate_query_attempts == 0 and hs._intermediate_query_for is None
    assert hs.intermediate_query_due_in() == pytest.approx(4.5)
    io.tx_went_out = lambda: True
    clock += 4.51
    assert hs.query_intermediate_answer() and hs._intermediate_query_attempts == 1


@pytest.mark.parametrize('field,value', [
    ('_tx_pending', None), ('turn', VA._TURN_PEER), ('_release_owed', True),
    ('_held_answer', (b'', 0)), ('state', VA.VaraState.DISCONNECTED),
    ('role', 'responder'),
])
def test_other_obligations_disable_the_query_clock(clock, field, value):
    hs, _ = pending()
    setattr(hs, field, value)
    assert hs.intermediate_query_due_in() is None


def test_nak_requested_data_gets_a_new_reply_window_without_renewing_budgets(clock):
    hs, io = pending()
    clock += 1.81
    assert hs.query_intermediate_answer()
    attempts = hs._intermediate_query_attempts
    clock += 6.0  # Includes the peer's reply and our DATA playback.
    hs._took_nak()
    assert hs._tx_retries == 1 and hs._intermediate_query_attempts == attempts
    assert hs.intermediate_query_due_in() == pytest.approx(VA._OVER_RETRY_S)
    assert not hs.query_intermediate_answer()
    np.testing.assert_array_equal(io.sent[0], io.sent[-1])
