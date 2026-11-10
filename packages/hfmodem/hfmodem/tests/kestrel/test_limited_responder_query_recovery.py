# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Limited BW500 queries name keyed overs, including NAK retries."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock, _IO  # noqa: F401
from .test_responder_control_stream import feed


def pending(*, final=False, payload=None, bw='500'):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    h.role, h.called, h.state = 'responder', 'W9SSJ', A.VaraState.CONNECTED
    h.caller, h.turn = 'W1AW', A._TURN_OURS
    h.chat = h._chat_profile = h._link_high_speed = False
    h._start_compression('OFF')
    h.send(payload if payload is not None else
           b'last over' if final else bytes(range(256)) * 3)
    assert A._phy.over_is_last(h._tx_pending[0], h.caller) == final
    return h, io


def burst(h, kind):
    return np.r_[np.zeros(4800), M.synth_burst(h.called, F.for_bw(kind, '500')),
                 np.zeros(4800)]


def short_ack(h):
    pairs = F.over_continue_frames(h.called, '500', (0,))[0][0]
    return np.r_[np.zeros(4800), M.synth_tone_pairs(pairs), np.zeros(4800)]


def query(h, io, frame):
    assert h._query_answerer_over(), io.msgs
    pairs = F.over_continue_frames(h.called, h.bw, (frame,))[frame][1]
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(pairs))


@pytest.mark.parametrize('block', [960, 4096, 24000])
def test_limited_nonchat_reack_credits_only_the_queried_over(block):
    h, io = pending()
    first = h._tx_pending
    query(h, io, 11)
    sent = len(io.sent)
    x = burst(h, F.SESSION_OVER_RESPONSE)
    feed(h, x, block)
    assert h.payload_ledger.confirmed == len(first[0]) - 1, io.msgs
    assert h._tx_pending[1] == 2 and h._tx_retries == 0
    assert len(io.sent) == sent + 1
    assert h._answer_query_for is None
    second = h._tx_pending
    feed(h, x, block)
    assert h._tx_pending is second and len(io.sent) == sent + 1
    assert h.payload_ledger.confirmed == len(first[0]) - 1


def test_query_phase_follows_keyed_overs_and_repeated_queries_retain_it():
    h, io = pending()
    for over, frame in [(1, 11), (2, 12), (3, 11)]:
        assert h._tx_pending[1] == over
        query(h, io, frame)
        query(h, io, frame)
        feed(h, short_ack(h))


def test_lost_over_nak_resends_and_advances_query_phase_without_credit():
    h, io = pending()
    first, debt = h._tx_pending, h.payload_ledger.pending
    query(h, io, 11)
    feed(h, burst(h, F.SESSION_OVER_NAK))
    assert h._tx_retries == 1 and h._tx_pending[1] == first[1], io.msgs
    assert h.payload_ledger.confirmed == 0 and h.payload_ledger.pending == debt
    query(h, io, 12)
    retry = h._tx_pending
    feed(h, burst(h, F.SESSION_OVER_RESPONSE))
    assert h.payload_ledger.confirmed == len(retry[0]) - 1, io.msgs
    assert h._tx_pending[1] == 2 and h._tx_retries == 0
    query(h, io, 11)


@pytest.mark.parametrize('case', ['no-query', 'refused-query', 'refused-renewal',
    'stale-wall', 'stale-samples', 'different-pending'])
def test_reack_requires_one_fresh_successful_query_for_the_exact_over(case):
    h, io = pending()
    if case not in ('no-query', 'refused-query'):
        query(h, io, 11)
    if case in ('refused-query', 'refused-renewal'):
        io.tx_went_out = lambda: False
        assert not h._query_answerer_over()
        io.tx_went_out = lambda: True
    if case == 'stale-wall': _Clock.t += A._FINAL_QUERY_REPLY_S + 1
    if case == 'stale-samples': h._answer_query_samples = (A._FINAL_QUERY_REPLY_S + 1) * M.FS
    if case == 'different-pending': h._tx_pending = tuple(list(h._tx_pending))
    old, sent = h._tx_pending, len(io.sent)
    feed(h, burst(h, F.SESSION_OVER_RESPONSE))
    assert h._tx_pending is old and h.payload_ledger.confirmed == 0, io.msgs
    assert len(io.sent) == sent


@pytest.mark.parametrize('frame', [0, 1])
def test_final_over_query_repeats_its_control_ack_once(frame):
    h, io = pending(final=True)
    total = h.payload_ledger.pending
    query(h, io, 11)
    ack = M.synth_tone_pairs(F.control_bursts(h.caller, '500', frame)[0])
    x = np.r_[np.zeros(4800), ack, np.zeros(4800)]
    feed(h, x)
    assert h._tx_pending is None and h.payload_ledger.confirmed == total, io.msgs
    feed(h, x)
    assert h._tx_pending is None and h.payload_ledger.confirmed == total


def test_final_over_keeps_its_control_ack_family():
    h, io = pending(final=True)
    query(h, io, 11)
    old = h._tx_pending
    feed(h, burst(h, F.SESSION_OVER_RESPONSE))
    assert h._tx_pending is old and h.payload_ledger.confirmed == 0, io.msgs


def test_final_over_uses_its_delivery_phase_and_accepts_long_idle_ack_once():
    h, io = pending(payload=bytes(range(52)))
    feed(h, short_ack(h))
    assert h._tx_pending[1] == 2 and A._phy.over_is_last(h._tx_pending[0], h.caller)
    query(h, io, 12)
    x = np.r_[np.zeros(4800), M.synth_burst(h.caller, F.for_bw(F.SESSION_TURN_IDLE, '500')),
              np.zeros(4800)]
    feed(h, x)
    assert h._tx_pending is None and h.payload_ledger.confirmed == 52, io.msgs
    feed(h, x)
    assert h.payload_ledger.confirmed == 52
    h.send(b'next delivery')
    h.turn = A._TURN_OURS
    h._tx_data_over()
    query(h, io, 11)


def test_refused_nak_retry_does_not_advance_query_phase():
    h, io = pending()
    query(h, io, 11)
    io.tx_went_out = lambda: False
    feed(h, burst(h, F.SESSION_OVER_NAK))
    assert h._tx_retries == 0 and h.payload_ledger.confirmed == 0, io.msgs
    io.tx_went_out = lambda: True
    query(h, io, 11)


@pytest.mark.parametrize('role', ['responder', 'initiator'])
def test_final_repacketization_preserves_delivery_phase(role):
    h, io = pending(payload=bytes(range(75)))
    h.role = role

    def ack():
        h._continue_frame = (h._tx_pending, 0)
        h._took_over_continue()

    def ask(frame):
        if role == 'responder':
            query(h, io, frame)
        else:
            _Clock.t += A._FINAL_QUERY_REPLY_S + 1
            full = h._intermediate_query_attempted
            assert h._probe_intermediate_answer()
            kind = (F.SESSION_INTERMEDIATE_ANSWER_QUERIES if full
                    else F.SESSION_ANSWER_QUERIES_SHORT)[frame - 11]
            assert np.array_equal(io.sent[-1], M.synth_burst(h.called, F.for_bw(kind, '500')))

    ack()
    assert h.payload_ledger.confirmed == 43
    assert A._phy.over_is_last(h._tx_pending[0], h.caller)
    # The lost final becomes a nonfinal nine-byte retry at record 0.
    h._retry_data_over(peer_asked=True, adapt=True)
    assert h._tx_retries == 1 and not A._phy.over_is_last(h._tx_pending[0], h.caller)
    ack()
    assert h.payload_ledger.confirmed == 52
    # The fourth transmission is also lost. Xara must hear phase 1 and NAK;
    # phase 0 would replay the previous ACK and falsely credit nine bytes.
    ask(12)
    ask(12)
    old = h._tx_pending
    io.tx_went_out = lambda: False
    if role == 'responder':
        assert not h._query_answerer_over()
    else:
        _Clock.t += A._FINAL_QUERY_REPLY_S + 1
        assert not h._probe_intermediate_answer()
    io.tx_went_out = lambda: True
    ask(12)
    if role == 'responder':
        feed(h, burst(h, F.SESSION_OVER_NAK))
        assert h._tx_retries == 1
    else:
        h._retry_data_over(peer_asked=True, adapt=True)
    assert h.payload_ledger.confirmed == 52 and h._tx_pending[1] == old[1]
    assert h._delivery_keyed == 5
    while not A._phy.over_is_last(h._tx_pending[0], h.caller):
        ack()
    h._took_control_burst()
    assert h.payload_ledger.confirmed == 75 and h._delivery_keyed == 0
    h.send(b'next delivery')
    h.turn = A._TURN_OURS
    h._tx_data_over()
    assert h._delivery_keyed == 1
    ask(11)


def receiver(h):
    from .test_idle_poll import _IO
    io = _IO()
    c = A.VaraStationHandshake([h.caller], io, bw=h.bw)
    c.role, c.called, c.caller = 'initiator', h.called, h.caller
    c.state, c.step, c.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_PEER
    c._start_compression('OFF')
    return c, io


def on_air(x):
    return np.r_[np.zeros(4800), x, np.zeros(4800)]


def test_two_kestrels_recover_lost_second_ack_once_without_duplicate_delivery():
    payload = bytes(range(110))
    h, io = pending(payload=payload)
    c, ci = receiver(h)
    feed(c, on_air(io.sent[-1]))
    assert b''.join(ci.host) == payload[:43], ci.msgs
    feed(h, on_air(ci.sent[-1]))
    assert h._tx_pending[1] == 2
    feed(c, on_air(io.sent[-1]))
    assert b''.join(ci.host) == payload[:86], ci.msgs
    # Erase the second ACK. Pass the actual query and caller reply, never a
    # synthesized expected response, through both public receive streams.
    query(h, io, 12)
    sent = len(ci.sent)
    # The query deadline starts at unkey. This IO keys instantly, so replay
    # its waveform from before that timestamp, as a real peer receives it.
    _Clock.t -= len(io.sent[-1]) / M.FS
    feed(c, on_air(io.sent[-1]))
    assert len(ci.sent) == sent + 1, ci.msgs
    reply = on_air(ci.sent[-1])
    feed(h, reply)
    assert h.payload_ledger.confirmed == 86 and h._tx_pending[1] == 3, io.msgs
    third, count = h._tx_pending, len(io.sent)
    feed(h, reply)
    assert h._tx_pending is third and len(io.sent) == count
    feed(c, np.r_[on_air(io.sent[-1]), np.zeros(M.FS // 2)])
    feed(h, on_air(ci.sent[-1]))
    assert b''.join(ci.host) == payload, '\n'.join(ci.msgs)
    assert h.payload_ledger.confirmed == len(payload) and h._tx_pending is None


@pytest.mark.parametrize('read', [0, 1, 2, 3])
@pytest.mark.parametrize('missing', [False, True])
def test_limited_caller_queries_repeat_saved_answer_or_request_missing_over(read, missing):
    h, hi = pending(payload=bytes(range(256)))
    c, ci = receiver(h)
    for _ in range(read):
        feed(c, on_air(hi.sent[-1]))
        _Clock.t -= len(on_air(ci.sent[-1])) / M.FS
        feed(h, on_air(ci.sent[-1]))
    phase = read % 2 if missing else (read - 1) % 2
    count = len(ci.sent)
    feed(c, on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', phase))))
    assert len(ci.sent) == count + 1, ci.msgs
    kind = (F.SESSION_OVER_NAK_SHORT if read else F.SESSION_OVER_NAK) if missing or not read else F.SESSION_OVER_RESPONSE
    assert np.array_equal(ci.sent[-1], M.synth_burst(c.called, F.for_bw(kind, '500'))), ci.msgs
    assert len(b''.join(ci.host)) == 43 * read


@pytest.mark.parametrize('phase', [0, 1])
def test_high_speed_caller_retains_missing_over_queries_after_direct_nak(monkeypatch, phase):
    from .test_answer_query_recovery import caller_owing, keyed
    c, ci = caller_owing(monkeypatch)
    monkeypatch.setattr(c, '_peer_data_over', A.VaraStationHandshake._peer_data_over.__get__(c))
    c._delivery_read = True
    c._irs_nak(A._phy.LevelRead(c._expect_level, (), None, 1))
    assert c._answer_owed == A._OWED_OVER and c._rx_failed
    assert c._peer_answered == 1
    sent = len(ci.sent)
    feed(c, on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', phase))))
    assert len(ci.sent) == sent + 1 and keyed(ci, c.called, F.SESSION_OVER_NAK_SHORT), ci.msgs
    # Explicit missing-DATA evidence retains the VARA HF NAK even for the saved
    # ACK's phase; Xara's phase-only receiver would replay that ACK instead.
    assert c._peer_answered == 1
    assert not c._rx_window.trusted


def test_nak_requested_duplicate_data_owns_a_new_query_phase(monkeypatch):
    """DESIGN §5: a NAK of an unread attempt after K leaves the peer at S, so
    the DATA it draws is new even when equal to K byte for byte. DESIGN §6:
    the NAK's first keying flips the phase, and that DATA opens the next."""
    from .test_answer_query_recovery import caller_owing, keyed
    c, ci = caller_owing(monkeypatch, high_speed=False)
    delivered = ci.host.copy()
    c._owe_unread_over()
    feed(c, on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', 0))))
    assert keyed(ci, c.called, F.SESSION_OVER_NAK), ci.msgs
    assert c._peer_answered == 1 and c._rx_window.T is None
    # caller_owing's DATA stub returns K's CRC-clean body again.
    assert c._answer_data_over([])
    assert ci.host == delivered * 2 and c._peer_received_phase == 0
    feed(c, on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', 1))))
    assert keyed(ci, c.called, F.SESSION_OVER_RESPONSE), ci.msgs
    assert ci.host == delivered * 2


def test_two_kestrels_ack_recovery_nak_keeps_saved_over_phase():
    h, hi = pending(payload=bytes(range(110)))
    c, ci = receiver(h)
    h._link_high_speed = c._link_high_speed = True
    h._lead = A.VB6Rnd(F.lead_state(h.caller, True, '500'))
    # Both transports consume airtime; replay it from before their unkey.
    for io in (hi, ci):
        tx = io.tx

        def transmit(samples, tx=tx):
            tx(samples)
            _Clock.t += len(samples) / M.FS

        io.tx = transmit

    def hear(peer, io, *, data=False):
        _Clock.t -= len(io.sent[-1]) / M.FS
        feed(peer, np.r_[on_air(io.sent[-1]), np.zeros(M.FS // 2 if data else 0)])

    # Erase the initial DATA. Its query requests a nine-byte retry.
    query(h, hi, 11)
    hear(c, hi)
    hear(h, ci)
    assert h._tx_retries == 1
    hear(c, hi, data=True)
    assert b''.join(ci.host) == bytes(range(9)), ci.msgs
    assert c._peer_received_phase == 1
    # Erase that ACK and both ACK rungs, then deliver the ladder's NAK.
    for _ in range(3):
        query(h, hi, 12)
        assert c._reack_answer(query_start=c._unkeyed_at + .1)
    _Clock.t += 60
    query(h, hi, 12)
    hear(c, hi)
    assert c._rx_window.A.kind == 'nak' and c._rx_window.A.sent
    assert c._rx_window.T is not None
    hear(h, ci)
    assert h._tx_retries == 2 and h.payload_ledger.confirmed == 0, hi.msgs
    hear(c, hi, data=True)
    assert len(b''.join(ci.host)) == 9, ci.msgs
    query(h, hi, 11)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.zeros(len(hi.sent[-1])))
    hear(h, ci)
    assert h.payload_ledger.confirmed == 9, hi.msgs
    # Erase the following DATA and exchange its actual phase-1 query/reply.
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, 12)
    sent = len(ci.sent)
    hear(c, hi, data=True)
    assert len(ci.sent) == sent + 1, ci.msgs
    hear(h, ci)
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == 9, hi.msgs
    assert c._peer_received_phase == 0


def missing_high_over(*, high_speed):
    h, _ = pending()
    c, ci = receiver(h)
    c._link_high_speed = high_speed
    c._expect_level, c._lv_owed = 13, True
    c._answer_owed, c._rx_failed = A._OWED_OVER, True
    c._delivery_read, c._peer_delivery_open = True, True
    c._peer_received_phase = c._peer_answered = 0
    c._rx_window.alt = 1
    c._rx_window.event('G')
    c._key(False)
    return c, ci


@pytest.mark.parametrize('high_speed', [True, False])
def test_repeated_eight_symbol_missing_over_query_retains_unheard_nak_rate(high_speed):
    c, ci = missing_high_over(high_speed=high_speed)
    x = np.r_[on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', 1))),
              np.zeros(M.FS // 2)]
    for count in (1, 2):
        if count == 2:
            feed(c, np.zeros(M.FS))
        feed(c, x)
        assert len(ci.sent) == count, ci.msgs
        assert c._expect_level == 9 and c._resend_matches(9), ci.msgs
    assert not ci.host


@pytest.mark.parametrize('high_speed', [True, False])
def test_delayed_frame12_missing_over_query_waits_for_next_poll(high_speed):
    c, ci = missing_high_over(high_speed=high_speed)
    c._callback_lag = M.FS
    feed(c, on_air(M.synth_tone_pairs(F.answer_query(c.called, '500', 1))))
    assert not ci.sent, ci.msgs
    assert c._expect_level == 13 and c._reacks == 0
    assert any('listening gap has gone' in m for m in ci.msgs)


@pytest.mark.parametrize('attempts', [(True,), (True, True), (False, True),
                                   (True, False, True)])
def test_peer_idle_unread_naks_do_not_credit_the_following_lost_over(attempts):
    from .test_bw500_recovery import _unreadable
    h, hi = pending(payload=bytes(range(110)))
    c, ci = receiver(h)
    assert c._answer_data_over(_unreadable(
        np.r_[np.zeros(M.FS), hi.sent[-1], np.zeros(M.FS)]))
    assert c._owed_block and not ci.sent and not ci.host
    idle = on_air(M.synth_burst(c.called, F.for_bw(F.SESSION_RESPONDER_IDLE, '500')))
    for went in attempts:
        previous = c._peer_answered
        ci.tx_went_out = lambda: went
        c.on_rx_audio(idle)
        assert np.array_equal(ci.sent[-1], M.synth_burst(
            c.called, F.for_bw(F.SESSION_OVER_NAK, '500')))
        assert c._peer_answered == (0 if went else previous)
    ci.tx_went_out = lambda: True
    query(h, hi, 11)
    feed(h, on_air(ci.sent[-1]))
    assert h._tx_retries == 1 and h.payload_ledger.confirmed == 0, hi.msgs
    assert c._answer_data_over(on_air(hi.sent[-1]))
    assert b''.join(ci.host) == bytes(range(9)), ci.msgs
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 9 and h._tx_pending[1] == 2
    # Erase the next DATA and exchange its actual phase-0 query and reply.
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, 11)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == 9, hi.msgs
    assert h._tx_retries == 1


@pytest.mark.parametrize('attempts', [(True,), (True, True), (False, True),
                                   (True, False, True)])
def test_consecutive_unread_retries_do_not_credit_the_following_lost_over(attempts):
    from .test_bw500_recovery import _unreadable
    h, hi = pending(payload=bytes(range(110)))
    c, ci = receiver(h)
    idle = on_air(M.synth_burst(c.called, F.for_bw(F.SESSION_RESPONDER_IDLE, '500')))
    for frame in (11, 12):
        assert c._answer_data_over(_unreadable(
            np.r_[np.zeros(M.FS), hi.sent[-1], np.zeros(M.FS)]))
        assert c._owed_block and not ci.host
        for went in attempts:
            ci.tx_went_out = lambda: went
            c.on_rx_audio(idle)
        ci.tx_went_out = lambda: True
        query(h, hi, frame)
        feed(h, on_air(ci.sent[-1]))
        assert h.payload_ledger.confirmed == 0, hi.msgs
    assert h._tx_retries == 2
    assert c._answer_data_over(on_air(hi.sent[-1]))
    assert b''.join(ci.host) == bytes(range(9)), ci.msgs
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 9 and h._tx_pending[1] == 2
    # Erase the next DATA; its actual phase-1 query must draw a NAK.
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, 12)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == 9, hi.msgs
    assert h._tx_retries == 1


def test_query_recovered_refused_final_ack_resets_phase_through_handover():
    h, hi = pending(final=True)
    c, ci = receiver(h)
    c.over_continue_after = c.over_continue
    ci.tx_went_out = lambda: False
    assert c._answer_data_over(on_air(hi.sent[-1]))
    assert b''.join(ci.host) == b'last over'
    ci.tx_went_out = lambda: True
    query(h, hi, 11)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 9 and h._tx_pending is None, hi.msgs
    # Exchange the actual release and empty-turn handback before the next DATA.
    h.send(bytes(range(110)))
    feed(c, on_air(hi.sent[-1]))
    assert c.turn == A._TURN_PEER and c._released, ci.msgs
    feed(h, on_air(ci.sent[-1]))
    assert h.turn == A._TURN_OURS, hi.msgs
    feed(c, on_air(hi.sent[-1]))
    assert len(b''.join(ci.host)) == 52, ci.msgs
    _Clock.t -= len(ci.sent[-1]) / M.FS
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == 52, '\n'.join(hi.msgs + ci.msgs)
    # Erase the second DATA of the new delivery and exchange its query/reply.
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, 12)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, on_air(hi.sent[-1]))
    assert len(ci.sent) == sent + 1, '\n'.join(ci.msgs)
    feed(h, on_air(ci.sent[-1]))
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == 52, hi.msgs
    assert h._tx_retries == 1


@pytest.mark.parametrize('answer', A.OVER_CONTINUE_ANSWERS)
@pytest.mark.parametrize('refused', [False, True])
def test_all_continue_waveforms_and_query_repeats_record_the_received_phase(answer, refused):
    h, hi = pending()
    c, ci = receiver(h)
    c.over_continue = answer
    ci.tx_went_out = lambda: not refused
    assert c._answer_data_over(on_air(hi.sent[-1]))
    assert c._peer_received_phase == 0
    assert c._peer_answered == (None if refused else 0)
    ci.tx_went_out = lambda: True
    c._reanswer_over(query_start=c._unkeyed_at + .1)
    assert c._peer_answered == 0
    c._reanswer_over(query_start=c._unkeyed_at + .1)
    assert c._peer_answered == 0
    for went in (False, True):
        ci.tx_went_out = lambda: went
        assert c._reack_over(query_start=c._unkeyed_at + .1) == went
        assert c._peer_answered == 0
    query(h, hi, 11)
    c._reanswer_over(query_start=c._unkeyed_at + .1)
    feed(h, on_air(ci.sent[-1]))
    assert h._tx_pending[1] == 2, hi.msgs
    assert c._answer_data_over(on_air(hi.sent[-1]))
    assert c._peer_received_phase == 1 and c._peer_answered == 1


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('route', ['query', 'release-ladder', 'long-final'])
@pytest.mark.parametrize('phase', [0, 1])
def test_final_ack_routes_close_the_delivery_and_ignore_refused_keyings(chat, route, phase):
    h, hi = pending(final=not phase, payload=bytes(range(52)) if phase else None)
    c, ci = receiver(h)
    c.chat = c._chat_profile = chat
    if phase:
        assert c._answer_data_over(on_air(hi.sent[-1]))
        feed(h, on_air(ci.sent[-1]))
        assert h.payload_ledger.confirmed == 43
    ci.tx_went_out = lambda: False
    assert c._answer_data_over(on_air(hi.sent[-1]))
    assert c._peer_received_phase == phase and c._peer_fresh
    assert c._peer_answered == (0 if phase else None)
    if route == 'long-final':
        c.chat = True
        c.send(b'waiting for the turn')
    answer = {'query': c._reanswer_over, 'release-ladder': c._reack_release,
              'long-final': c._repeat_final_ack}[route]
    ci.tx_went_out = lambda: True
    answer()
    assert c._peer_answered is None and not c._peer_fresh
    assert not c._peer_delivery_open
    waveform = ci.sent[-1].copy()
    for went in (False, True):
        ci.tx_went_out = lambda: went
        answer()
        assert c._peer_answered is None and not c._peer_fresh
        np.testing.assert_array_equal(ci.sent[-1], waveform)
    c._received_data_phase()
    assert c._peer_received_phase == 0


def test_data_and_query_events_survive_repeat_requests_and_refused_answers():
    h, hi = pending()
    c, ci = receiver(h)
    assert c._answer_data_over(on_air(hi.sent[-1]))
    delivered = ci.host.copy()
    assert c._rx_window.alt == 1 and c._peer_answered == 0
    for phase in (1, 0):
        c._owe_unread_over()
        assert c._rx_window.A.phase == phase and not c._rx_window.A.sent
        for went in (False, True, True):
            ci.tx_went_out = lambda: went
            already_sent = c._rx_window.A.sent
            c._rx_query(phase=phase ^ 1, query_start=c._unkeyed_at + .1)
            assert c._rx_window.A.phase == phase
            assert c._rx_window.A.sent == (already_sent or went)
        assert c._peer_answered == phase and not c._peer_fresh
        assert not c._rx_window.trusted
    assert ci.host == delivered


@pytest.mark.parametrize('route', ['band-envelope', 'base-locator'])
def test_failed_data_detected_at_idle_names_a_new_answer_once(monkeypatch, route):
    from types import SimpleNamespace
    from .test_answer_query_recovery import caller_owing
    c, ci = caller_owing(monkeypatch)
    if route == 'band-envelope':
        c._owe_nothing()
        monkeypatch.setattr(c, '_answer_unheard', lambda *args: False)
        # A successful keying clears the envelope; the next idle has no DATA.
        monkeypatch.setattr(c, '_over_heard', lambda: c._peer_answered == 0)
        answer = c._reack_on_idle
    else:
        over = SimpleNamespace(refined_score=A._UNREAD_LOCK, candidate=SimpleNamespace(level=4),
                               result=SimpleNamespace(complete=False), pair=False, end=0)
        c._base_rx = SimpleNamespace(history=[over], pending=[])
        c._front_heard = 0
        assert c._unread_over_500()
        c._answer_owed, c._owed_block = A._OWED_OVER, True
        c._owed_recovery, c._reacks = True, 0
        assert not c._unread_over_500()
        answer = c._reack_over
    ci.tx_went_out = lambda: False
    assert not answer()
    assert c._peer_answered == 0 and c._peer_fresh
    ci.tx_went_out = lambda: True
    assert answer()
    assert c._peer_answered == 1 and not c._peer_fresh
    assert answer()
    assert c._peer_answered == 1 and not c._peer_fresh


def high_speed_pair(payload, bw='500'):
    h, hi = pending(payload=payload, bw=bw)
    c, ci = receiver(h)
    h._link_high_speed = c._link_high_speed = True
    h._lead = A.VB6Rnd(F.lead_state(h.caller, True, bw))
    h._key(False)
    return h, hi, c, ci


def hear_with_tail(peer, io):
    feed(peer, np.r_[on_air(io.sent[-1]), np.zeros(M.FS // 2)])


@pytest.mark.parametrize('attempts', [(True,), (True, True), (False, True),
                                   (True, False, True)])
def test_idle_detected_over_with_existing_debt_does_not_credit_later_lost_data(monkeypatch, attempts):
    from .test_vara_levels_arq import _band_noise
    h, hi, c, ci = high_speed_pair(bytes(range(256)) * 8)
    for _ in range(2):
        hear_with_tail(c, hi)
        hear_with_tail(h, ci)
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == 175
    assert c._answer_owed == A._OWED_OVER and c._peer_answered == 1
    # Hold the retry's rate so the following missing over has the same load.
    monkeypatch.setitem(A._phy.RX_CEILING, '500', 8)
    # The next DATA has no decoder lock, but its band envelope survives.
    # Preserve the obligation from the previous ACK, as on the live idle route.
    rms = np.sqrt(np.mean(hi.sent[-1] ** 2)) / 2
    n = max(len(hi.sent[-1]), int(4.3 * M.FS))
    x = np.r_[_band_noise(8 * M.FS, rms, 1),
              _band_noise(n, rms, 2, 1150, 1850) + _band_noise(n, rms, 3)]
    c._track_band(x)
    assert c._over_heard()
    _Clock.t += len(x) / M.FS
    c._heard += len(x)
    c._idle_end = _Clock.t
    assert not c._answer_unheard() and not c._rx_failed
    for went in attempts:
        ci.tx_went_out = lambda: went
        c._peer_polled()
        assert c._reack_on_idle() == went
    ci.tx_went_out = lambda: True
    query(h, hi, 11)
    feed(h, on_air(ci.sent[-1]))
    assert h._tx_retries == 1 and h.payload_ledger.confirmed == 175, hi.msgs
    hear_with_tail(c, hi)
    feed(h, on_air(ci.sent[-1]))
    assert len(b''.join(ci.host)) == 435, ci.msgs
    # Erase the following DATA; exchange its real frame-11 query and reply.
    feed(c, np.zeros(len(hi.sent[-1])))
    query(h, hi, 11)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    hear_with_tail(c, hi)
    assert len(ci.sent) == sent + 1, ci.msgs
    hear_with_tail(h, ci)
    assert h.payload_ledger.confirmed == len(b''.join(ci.host)) == 435, hi.msgs
    assert h._tx_retries == 1
    assert c._peer_received_phase == 0
    assert c._rx_window.A.kind == 'nak'


def test_high_speed_even_final_ack_query_recovers_once_without_duplicate_delivery():
    payload = bytes(range(52))
    h, hi, c, ci = high_speed_pair(payload)
    hear_with_tail(c, hi)
    hear_with_tail(h, ci)
    assert h.payload_ledger.confirmed == 43 and h._tx_pending[1] == 2
    hear_with_tail(c, hi)
    assert b''.join(ci.host) == payload and not c._lv_owed, ci.msgs
    assert c._answer_owed == A._OWED_RELEASE and c._peer_received_phase == 1
    # Erase the final ACK. Frame 12 must recover it despite no next DATA due.
    query(h, hi, 12)
    sent = len(ci.sent)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    hear_with_tail(c, hi)
    assert len(ci.sent) == sent + 1, ci.msgs
    reply = on_air(ci.sent[-1])
    feed(h, reply)
    assert h.payload_ledger.confirmed == 52 and h._tx_pending is None, hi.msgs
    sent = len(hi.sent)
    feed(h, reply)
    assert h.payload_ledger.confirmed == 52 and len(hi.sent) == sent
    assert b''.join(ci.host) == payload


def test_new_idle_envelopes_after_naks_each_name_a_fresh_over():
    from .test_vara_levels_arq import _band_noise
    c, ci = missing_high_over(high_speed=True)
    c._idle_kind = F.SESSION_RESPONDER_OVER_IDLE
    x = np.r_[_band_noise(8 * M.FS, .1, 1),
              _band_noise(5 * M.FS, .1, 2, 1150, 1850)
              + _band_noise(5 * M.FS, .1, 3)]
    for phase in (1, 0):
        assert c._rx_failed and not c._peer_fresh
        c._track_band(x)
        assert c._over_heard()
        _Clock.t += len(x) / M.FS
        c._heard += len(x)
        c._idle_end = _Clock.t
        assert not c._answer_unheard()
        for went in (False, True, False, True):
            ci.tx_went_out = lambda: went
            previous = c._peer_answered
            c._peer_polled()
            assert c._reack_on_idle() == went
            assert c._peer_answered == (phase if went else previous)
            assert c._peer_fresh == (not went and previous != phase)
    assert not ci.host


@pytest.mark.parametrize('route', ['idle', 'over-poll', 'query', 'quiet-timeout', 'saved-ack-ladder'])
def test_caller_nak_routes_name_the_missing_or_saved_over_through_refusals(monkeypatch, route):
    from .test_answer_query_recovery import caller_owing
    c, ci = caller_owing(monkeypatch)
    assert c._peer_answered == 0
    if route == 'saved-ack-ladder':
        for _ in range(2):
            assert c._reack_answer(query_start=c._unkeyed_at + .1)
            assert c._peer_answered == 0
        answer, phase = lambda: c._reack_answer(query_start=c._unkeyed_at + .1), 0
    else:
        c._deliver([None], hold=True)
        c._finish_delivery_window(complete=False)
        answer, phase = {'idle': c._reack_on_idle, 'over-poll': c._took_over_poll,
                         'quiet-timeout': lambda: c._nak_owed_over(quiet_retry=True),
                         'query': lambda: c._took_answer_query(1)}[route], 1
    for went in (False, True, False, True):
        previous = c._peer_answered
        ci.tx_went_out = lambda: went
        answer()
        assert c._peer_answered == (phase if went else previous), ci.msgs


    delivered = ci.host.copy()
    c._rx_failed = False
    assert c._answer_data_over([])
    assert c._peer_received_phase == phase ^ 1
    assert c._peer_answered == phase ^ 1 and ci.host == delivered * 2


@pytest.mark.parametrize('read', [False, True])
def test_direct_crc_naks_count_received_failed_overs_but_not_refused_keyings(read):
    h, _ = pending()
    c, ci = receiver(h)
    c._link_high_speed = True
    c._delivery_read = read
    nak = lambda: c._irs_nak(A._phy.LevelRead(c._expect_level, (), None, 1))
    for went, expected in [(False, None), (True, 0),
                           (False, 0), (True, 1)]:
        ci.tx_went_out = lambda: went
        nak()
        assert c._peer_answered == expected
    if read:
        np.testing.assert_array_equal(ci.sent[0], M.synth_tone_pairs(F.nak(c.called, '500')[0]))
    else:
        np.testing.assert_array_equal(ci.sent[0], M.synth_burst(
            c.called, F.for_bw(F.SESSION_DATA_NAK_CALLER_SHORT, '500')))


def test_partial_pair_and_its_reack_record_the_same_phase_after_refused_answer():
    from .test_bw500_pairing import _receiver, _build, rx
    c, ci = _receiver('initiator', 3, 6)
    body = _build(3)
    frame = rx.FrameResult(0, (0,), 0, 1., True, body, level=3, frames_in_burst=2)
    ci.tx_went_out = lambda: False
    c._answer_pair_loss_500(frame, body[:-2])
    assert c._peer_received_phase == 0 and c._peer_answered is None
    ci.tx_went_out = lambda: True
    assert c._reack_over(query_start=c._unkeyed_at + .1)
    assert c._peer_answered == 0
    assert c._reack_over(query_start=c._unkeyed_at + .1)
    assert c._peer_answered == 0
    assert len(ci.host) == 1
    for x in ci.sent[1:]:
        np.testing.assert_array_equal(x, ci.sent[0])


def test_peer_idle_answer_nak_records_its_over_even_without_recovery_debt():
    h, _ = pending()
    c, ci = receiver(h)
    c.caller = 'W9SSJ'
    c._idle_kind = F.SESSION_RESPONDER_IDLE
    assert c._answer_owed is None and c._peer_answered is None
    for went, expected in [(False, None), (True, 0),
                           (False, 0), (True, 0)]:
        ci.tx_went_out = lambda: went
        assert c._answer_peer_idle() == went
        assert c._peer_answered == expected
        np.testing.assert_array_equal(ci.sent[-1], M.synth_burst(
            c.called, F.for_bw(F.SESSION_OVER_NAK, '500')))


@pytest.mark.parametrize('bw,credits', [('500', (43, 175)),
                                        ('2300', (89, 262)), ('2750', (89, 262))])
def test_high_speed_lost_acks_frame11_then_frame12_queries_repeat_them_once(bw, credits):
    # W9SSJ-1 to W9SSJ, weak path chat 2026-10-30: at high speed the caller
    # took the phase-1 query behind a second over it had read for the VARA HF
    # over-poll, refused it as late seven times, and the answerer closed.
    payload = bytes(range(256)) * 8
    h, hi, c, ci = high_speed_pair(payload, bw)
    for frame, confirmed in zip((11, 12), credits):
        hear_with_tail(c, hi)
        assert len(b''.join(ci.host)) == confirmed and c._lv_owed, ci.msgs
        assert c._peer_answered == (frame - 11), ci.msgs
        # Erase the ACK; exchange the answerer's real query and the reply.
        query(h, hi, frame)
        sent = len(ci.sent)
        _Clock.t -= len(hi.sent[-1]) / M.FS
        hear_with_tail(c, hi)
        assert len(ci.sent) == sent + 1, ci.msgs
        assert any('answering again' in m for m in ci.msgs[-4:]), ci.msgs
        # Replay the answer from key-down; wide saved answers are 32 symbols.
        _Clock.t = ci.at[-1] - 4800 / M.FS
        hear_with_tail(h, ci)
        assert h.payload_ledger.confirmed == confirmed, hi.msgs
        assert h._tx_pending[1] == frame - 9 and h._tx_retries == 0, hi.msgs
    assert not any('over-poll' in m for m in ci.msgs), ci.msgs
    hear_with_tail(c, hi)
    hear_with_tail(h, ci)
    assert b''.join(ci.host) == payload[:h.payload_ledger.confirmed], ci.msgs
    assert h.payload_ledger.confirmed > confirmed
