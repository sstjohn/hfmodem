# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Distinct pending frames get distinct bounded recovery opportunities."""
import pytest
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from hfmodem.tests.kestrel.test_intermediate_query_probe import pending
from hfmodem.tests.kestrel.test_intermediate_query_answer import native, timely
from hfmodem.tests.kestrel.test_final_ack_recovery import stream, is_burst


def test_three_lost_answers_can_recover_without_repeating_data(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    hs, io = pending(payload=b'A'*89 + b'B'*89 + b'C'*89 + b'end')
    acknowledged = []
    for index in range(3):
        old = hs._tx_pending
        assert not hs._intermediate_query_attempted
        assert hs.intermediate_query_due_in() == pytest.approx(1.8)
        clock[0] += 1.81
        assert hs.query_intermediate_answer()
        assert is_burst(io.sent[-1], VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[index % 2])
        assert not hs.query_intermediate_answer()
        clock[0] += 1.6
        stream(hs, native('stock'))
        assert hs._tx_pending[1] == old[1]+1
        assert hs._tx_pending[0] != old[0]
        assert hs._intermediate_query_for is None
        acknowledged.append(old)
        assert hs.turn == VA._TURN_OURS and not hs._handed_over
    assert len(io.sent) == 7  # four distinctDATA andthreequeries, noDRAINED/repeats
    assert not hs._txq and hs._tx_retries == 0
    assert hs._tx_pending not in acknowledged


def test_first_query_after_intact_first_answer_uses_second_data_phase():
    hs, io = pending()
    hs._took_over_continue()
    assert hs._full_keyed == 2 and not hs._intermediate_query_attempted
    assert hs._probe_intermediate_answer()
    assert is_burst(io.sent[-1], VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[1])


def test_refused_data_and_query_cannot_advance_phase():
    hs, io = pending()
    io.tx_went_out = lambda: False
    hs._took_over_continue()
    assert hs._full_keyed == 1 and hs._tx_pending is None
    io.tx_went_out = lambda: True
    hs._tx_data_over()
    assert hs._full_keyed == 2
    io.tx_went_out = lambda: False
    assert not hs._probe_intermediate_answer()
    assert hs._full_keyed == 2 and hs._intermediate_query_attempted
    assert is_burst(io.sent[-1], VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[1])


def test_new_frame_rejects_old_query_marker_and_unsolicited_replay(monkeypatch):
    hs, io = pending()
    assert hs._probe_intermediate_answer()
    old = hs._tx_pending
    bracket = timely(hs, native('stock'))
    hs.on_rx_audio(bracket)
    before = hs._tx_pending, list(hs._txq), hs.progress, len(io.sent)
    assert hs._intermediate_query_for is None
    hs.on_rx_audio(bracket)
    assert (hs._tx_pending, hs._txq, hs.progress, len(io.sent)) == before
    hs._intermediate_query_for = old
    assert not hs._intermediate_query_fresh()
    hs._took_intermediate_query_answer()
    assert (hs._tx_pending, hs._txq, hs.progress, len(io.sent)) == before


def test_refused_next_data_does_not_replenish_attempt_or_start_deadline():
    hs, io = pending()
    assert hs._probe_intermediate_answer()
    io.tx_went_out = lambda: False
    hs.on_rx_audio(timely(hs, native('stock')))
    assert hs._tx_pending is None  # delivered priorbody mayretire despitefailednextkey
    assert hs._intermediate_query_attempted
    assert hs.intermediate_query_due_in() is None
    io.tx_went_out = lambda: True
    assert hs._tx_data_over()
    assert not hs._intermediate_query_attempted
    assert 0 < hs.intermediate_query_due_in() <= 1.8


def test_second_query_negative_draws_one_bounded_resend_and_keeps_it_pending():
    """The negative answer to a query IS the responder over-NAK: one frame, two
    names. `vara_frames` records the negative capture as the controlled experiment
    where "Missing DATA instead produced 288/1117", and 288/1117 is
    `SESSION_OVER_NAK_RESPONDER` — the peer saying it could not read the over.

    A NAK the peer keyed is a positive ask, so it licenses exactly one resend of
    the outstanding over even here, where host retries are disabled: the silence
    `allow_data_retries=False` guards against is the opposite of a keyed NAK
    [see _took_responder_nak]. What bounds it is the over's own retry budget, and
    the block stays pending throughout — nothing but a validated continue or a
    solicited intermediate answer retires it.
    """
    hs, io = pending()
    assert hs._probe_intermediate_answer()
    hs.on_rx_audio(timely(hs, native('stock')))
    assert hs._probe_intermediate_answer()
    before = hs._tx_pending, list(hs._txq), hs._over
    assert len(io.sent) == 4 and hs._tx_retries == 0   # two DATA, two queries
    stream(hs, native('negative'))
    assert hs._tx_pending[1:] == (before[0][1], 101)
    assert len(hs._tx_pending[0]) == 23
    assert len(io.sent) == 5, "the peer's NAK drew no resend, or more than one"
    assert 0 < hs._tx_retries <= VA._OVER_RETRY_MAX, "the resend is unbounded"
    # Repacketization changes the pending frame but does not acknowledge any
    # bytes. Its own positive continue is required before the next base frame.
    pending_retry = hs._tx_pending
    hs._took_over_continue()
    assert hs.state == VA.VaraState.CONNECTED
    assert hs._tx_pending[1] == pending_retry[1] + 1
    assert hs._tx_pending[2] == 101
    assert len(io.sent) == 6
    assert hs._tx_retries == 0
