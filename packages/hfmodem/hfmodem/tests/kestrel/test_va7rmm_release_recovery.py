# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A lost caller handover must not strand the responder's queued DATA."""
import numpy as np
import pytest
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from hfmodem.tests.kestrel.test_bw500_query_recovery import _IO, clock as clock  # noqa: F401


def pending(clock):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True)
    h.role, h.caller, h.called = 'responder', 'VA7RMM', 'W9SSJ'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_PEER
    h._start_compression('OFF')
    h._answer_owed, h._over_answer = A._OWED_RELEASE, ('control', 1)
    h._progressed()
    h.send(b'5 hello')
    return h, io


def test_lost_release_solicits_fresh_grant_without_crediting_or_sending_data(clock):
    h, io = pending(clock)
    h.idle_keepalive()
    assert io.sent == []
    clock.now += A._IDLE_POLL_S + .01
    h.idle_keepalive()
    assert len(io.sent) == 1
    np.testing.assert_array_equal(io.sent[0], M.synth_burst('VA7RMM', F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER_SHORT, '500')))
    assert h.turn == A._TURN_ASKED and h._answer_owed == A._OWED_RELEASE
    assert h._tx_pending is None and h.payload_ledger.pending == 7
    # No grant, no DATA, even with repeated host writes and timer ticks.
    h.send(b'!')
    h.idle_keepalive()
    assert len(io.sent) == 1 and h.payload_ledger.pending == 8


@pytest.mark.parametrize('guard', ['receiving','busy','held','unread','no-reply','no-data','not-chat','other-band','closing'])
def test_no_recovery_across_receive_or_an_unfinished_over(clock, guard):
    h, io = pending(clock)
    clock.now += A._IDLE_POLL_S + .01
    if guard == 'receiving': io.receiving = True
    if guard == 'busy': h._band_quiet_s = lambda: 0.0
    if guard == 'held': h._held_answer = (True, True)
    if guard == 'unread': h._answer_owed = A._OWED_OVER
    if guard == 'no-reply': h._over_answer = None
    if guard == 'no-data': h._txq.clear()
    if guard == 'not-chat': h.chat = False
    if guard == 'other-band': h.bw = '2300'
    if guard == 'closing': h._close_asked_at = clock.now
    h.idle_keepalive()
    assert not io.sent


def test_request_budget_and_refused_transmission_preserve_debt(clock):
    h, io = pending(clock)
    clock.now += A._IDLE_POLL_S + .01
    io.tx_went_out = lambda: False
    h.idle_keepalive()
    assert h.turn == A._TURN_PEER and h._release_asks == 0
    io.tx_went_out = lambda: True
    for _ in range(A._TURN_MAX_ASK + 2):
        clock.now += A._IDLE_POLL_S + .01
        h.idle_keepalive()
    assert h._release_asks == A._TURN_MAX_ASK
    assert h.payload_ledger.pending == 7 and h._tx_pending is None


@pytest.mark.parametrize('reply', ['grant', 'wrong-call', 'truncated'])
def test_only_complete_addressed_grant_starts_queued_data(clock, reply):
    h, io = pending(clock)
    h._link_high_speed, h._rec500 = True, 4
    h._lead = A.VB6Rnd(F.lead_state(h.caller, True, '500'))
    clock.now += A._IDLE_POLL_S + .01
    h.idle_keepalive()
    wave = M.synth_burst('K7DPD' if reply == 'wrong-call' else 'W9SSJ',
                         F.for_bw(F.SESSION_DRAINED, '500'))
    if reply == 'truncated': wave = wave[:len(wave)//2]
    wave = np.r_[np.zeros(4800), wave, np.zeros(12000)]
    for at in range(0, len(wave), 960):
        clock.now += .02
        h._stream_responder_control(wave[at:at+960])
    if reply == 'grant':
        assert h._tx_pending is not None and h._answer_owed is None
        assert h._release_asks == 0
    else:
        assert h._tx_pending is None and h._answer_owed == A._OWED_RELEASE
    # A grant is not a DATA acknowledgment.
    assert h.payload_ledger.pending == 7


def caller(clock, *, turn=A._TURN_OURS):
    io = _IO()
    c = A.VaraStationHandshake(['VA7RMM'], io, bw='500', chat=True)
    c.role, c.caller, c.called = 'initiator', 'VA7RMM', 'W9SSJ'
    c.state, c.step, c.turn = A.VaraState.CONNECTED, A._I_CONNECTED, turn
    return c, io


def heard(c, clock, burst):
    wave = np.r_[np.zeros(4800), burst, np.zeros(9600)]
    for at in range(0, len(wave), 960):
        clock.now += 960 / M.FS
        c.on_rx_stream(wave[at:at+960])


@pytest.mark.parametrize('idle', [False, True], ids=['release', 'f20'])
def test_the_answerers_ask_behind_our_unheard_release_draws_it_again(clock, idle):
    """Kestrel answering Kestrel, OpenSlot air loop at 3 dB moderate fading,
    2026-11-10: our release faded, the answerer asked three times and the
    caller keyed nothing. A VARA HF caller keys its release again (VA7RMM)."""
    h, _ = pending(clock)
    h._link_high_speed, h._rec500 = True, 4
    h._lead = A.VB6Rnd(F.lead_state(h.caller, True, '500'))
    c, cio = caller(clock)
    assert c._release_turn(idle=idle)
    release = cio.sent[-1]
    clock.now += A._IDLE_POLL_S + .01
    h.idle_keepalive()
    heard(c, clock, h.io.sent[-1])
    assert len(cio.sent) == 2, cio.msgs[-3:]
    np.testing.assert_array_equal(cio.sent[-1], release)
    assert c.turn == A._TURN_PEER and c._tx_pending is None
    wave = np.r_[np.zeros(4800), release, np.zeros(12000)]
    for at in range(0, len(wave), 960):
        clock.now += .02
        h._stream_responder_control(wave[at:at+960])
    # The answerer took the handover it was asking after.
    assert h._answer_owed is None, h.io.msgs[-3:]
    assert (h._tx_pending is not None) is not idle, h.io.msgs[-3:]


def test_the_ask_is_left_alone_while_we_hold_the_turn(clock):
    h, _ = pending(clock)
    clock.now += A._IDLE_POLL_S + .01
    h.idle_keepalive()
    c, cio = caller(clock)
    c._released = True
    heard(c, clock, h.io.sent[-1])
    assert cio.sent == []
