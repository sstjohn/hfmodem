# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recovery events preserve ownership, obligations and monotonic budgets."""
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F
from .test_w5jab_receive_recovery import granted, recorded, feed, burst, is_nak
from .test_va7rmm_nak_recovery import pending
from .test_idle_poll import clock, _Clock  # noqa: F401


def test_repeated_grant_preserves_missing_data_recovery():
    h, io = granted()
    feed(h, recorded('query'))
    before = (h.progress, h._reacks, h._rx_resend, h._expect_level)
    for _ in range(3):
        feed(h, burst(h.caller, F.SESSION_TURN_REQUEST))
        assert (h.progress, h._reacks, h._rx_resend, h._expect_level) == before
        assert h._answer_owed == A._OWED_OVER and h._owed_block
    feed(h, recorded('query'))
    assert h._reacks == before[1] + 1


@pytest.mark.parametrize('kind', [F.SESSION_TURN_REQUEST, F.SESSION_TURN_REQUEST_SHORT])
def test_turn_request_cannot_steal_unconfirmed_data(kind):
    h, io = pending(query=False)
    old, count, progress = h._tx_pending, len(io.sent), h.progress
    feed(h, burst(h.caller, kind))
    assert h.turn == A._TURN_OURS and h._tx_pending is old
    assert len(io.sent) == count and h.progress == progress
    assert h.payload_ledger.confirmed == 0


def test_failed_data_still_owes_recovery_after_direct_nak():
    h, io = granted()
    h._irs_nak(A._phy.LevelRead(h._expect_level, (), None, 1))
    count = len(io.sent)
    assert h._rx_failed
    feed(h, recorded('query'))
    assert len(io.sent) == count + 1 and is_nak(io.sent[-1], h, short=False)
    assert h._owed_block and not io.host


def test_query_retries_refused_intermediate_ack_without_old_ack_or_nak():
    h, io = granted()
    h._over_answer = ('control', 1)  # Previous delivery must not leak through.
    io.tx_went_out = lambda: False
    h._key_over_answer(last=False, owes_release=False)
    io.tx_went_out = lambda: True
    progress, count = h.progress, len(io.sent)
    feed(h, recorded('query'))
    assert len(io.sent) == count + 1
    assert h._over_answer[0] == 'continue'
    assert not is_nak(io.sent[-1], h)
    assert h.progress == progress and not io.host


def test_refused_direct_nak_cannot_leave_previous_positive_answer_live():
    h, io = granted()
    h._owed_block = False
    h._over_answer = ('control', 1)
    io.tx_went_out = lambda: False
    level = h._expect_level
    h._irs_nak(A._phy.LevelRead(level, (), None, 1))
    assert h._expect_level == level and h._rx_resend is None
    io.tx_went_out = lambda: True
    feed(h, recorded('query'))
    assert is_nak(io.sent[-1], h) and not io.host


@pytest.mark.parametrize('case', ['held-window', 'closing', 'received-data'])
def test_late_turn_request_preserves_receive_transaction(case):
    h, io = granted()
    if case == 'held-window': h._held_answer = (False, False)
    if case == 'closing': h._close_asked_at = 1000.
    if case == 'received-data':
        h._owed_block = False
        h._over_answer = ('continue', A.OVER_CONTINUE_GENERATED)
    before = (len(io.sent), h.progress, h._held_answer, h._over_answer, h._owed_block)
    feed(h, burst(h.caller, F.SESSION_TURN_REQUEST))
    assert (len(io.sent), h.progress, h._held_answer, h._over_answer, h._owed_block) == before


def test_repeated_grants_between_queries_cannot_restart_recovery_budget():
    h, io = granted()
    for _ in range(A._REACK_MAX):
        feed(h, recorded('query'))
        feed(h, burst(h.caller, F.SESSION_TURN_REQUEST))
    feed(h, recorded('query'))
    assert h.state != A.VaraState.CONNECTED or h._close_asked_at is not None
    assert not io.host and h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('case', ['stale', 'refused-query', 'different-pending', 'replayed'])
def test_short_caller_nak_has_same_query_ownership_as_long_nak(case):
    h, io = pending(query=case != 'refused-query')
    if case == 'stale': _Clock.t += 3
    if case == 'refused-query':
        io.tx_went_out = lambda: False
        h._query_answerer_over()
        io.tx_went_out = lambda: True
    if case == 'different-pending': h._tx_pending = tuple(list(h._tx_pending))
    x = burst(h.called, F.SESSION_OVER_NAK_SHORT)
    if case == 'replayed':
        feed(h, x)
        assert h._tx_retries == 1
    count, retries, progress = len(io.sent), h._tx_retries, h.progress
    feed(h, x)
    assert h._tx_retries == retries and len(io.sent) == count
    assert h.progress == progress and h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('event', ['poll', 'turn-idle', 'unexpected-drained'])
def test_peer_activity_does_not_restart_queued_work_deadline(event):
    h, io = granted()
    h._answer_owed = None
    h._owed_block = False
    h._released_form = 'f20'
    progress = h.progress
    if event == 'poll':
        h._caller_holds_control(1)
    else:
        h._answer_responder_control(F.SESSION_TURN_IDLE if event == 'turn-idle'
                                   else F.SESSION_DRAINED)
    assert h.progress == progress
    assert h.payload_ledger.pending > 0 and h.payload_ledger.confirmed == 0


def test_healthy_idle_contact_remains_progress():
    h, io = granted()
    h._txq.clear()
    h._answer_owed = None
    h._owed_block = False
    before = h.progress
    h._answer_responder_control(F.SESSION_TURN_IDLE)
    assert h.progress == before + 1


def test_repeated_polls_cannot_keep_queued_message_waiting_forever():
    h, io = granted()
    h._answer_owed = None
    h._owed_block = False
    h._released_form = 'f20'
    for _ in range(A._MAX_WITHOUT_PROGRESS + 2):
        _Clock.t += A._IDLE_POLL_S
        h._caller_holds_control(1)
        h._answerer_tick()
        if h._close_asked_at is not None or h.state != A.VaraState.CONNECTED:
            break
    assert h._close_asked_at is not None or h.state != A.VaraState.CONNECTED
    assert h.payload_ledger.confirmed == 0
