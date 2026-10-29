# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CHAT answerer with text queued asks for the turn, and asks again in one
cadence when the caller's handover is lost.

W9SSJ answering W9SSJ-1 on 7105 kHz, 2026-10-28: six bytes were queued behind
the caller's confirm, and the answerer answered three keepalives with its idle
control frame. Under CHAT a stock answerer says idle with frame 1 and traffic
with frame 0, and the caller hands over on frame 0.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock as _Clock
from .test_responder_control_stream import feed
from .test_responder_initial_greeting import confirm
from .test_connect_half_open import answerer
from .test_va7rmm_release_recovery import pending


def keepalive(hs):
    feed(hs, np.concatenate([np.zeros(4800), M.synth_burst(hs.called, F.for_bw(F.SESSION_KEEPALIVE_B, '500')),
                             np.zeros(24000)]))


def control(hs, frame):
    return M.synth_tone_pairs(F.control_bursts(hs.caller, '500', frame)[1])


def set_up():
    hs, io = answerer()
    confirm(hs)
    confirm(hs, long=True)
    assert hs.turn != A._TURN_OURS and hs._released_form is None
    return hs, io


def test_queued_text_answers_the_callers_keepalive_with_traffic():
    hs, io = set_up()
    hs.send(b'6 hi')
    sent = len(io.sent)
    keepalive(hs)
    assert len(io.sent) == sent + 1, io.msgs
    assert np.array_equal(io.sent[-1], control(hs, 0))
    assert hs.turn == A._TURN_ASKED
    feed(hs, np.concatenate([np.zeros(4800), M.synth_burst(hs.called, F.for_bw(F.SESSION_DRAINED, '500')),
                             np.zeros(24000)]))
    assert hs._tx_pending is not None, io.msgs


def test_the_ask_stands_while_the_callers_handover_is_lost():
    """A stock answerer holding with traffic keeps asking with frame 0 until
    the caller's handover comes (2026-10-18 stock pairs); answered idle the second
    time, the queued text had nothing left to ask for it."""
    hs, io = set_up()
    hs.send(b'6 hi')
    keepalive(hs)
    assert hs.turn == A._TURN_ASKED
    keepalive(hs)
    assert np.array_equal(io.sent[-1], control(hs, 0)), io.msgs
    assert hs.turn == A._TURN_ASKED


def test_with_nothing_queued_the_keepalive_still_draws_idle():
    hs, io = set_up()
    keepalive(hs)
    assert np.array_equal(io.sent[-1], control(hs, 1))
    assert hs.turn != A._TURN_ASKED


@pytest.mark.parametrize('case', ['initiator', 'band-busy', 'receiving'])
def test_no_release_ask_is_reported_due_where_none_can_go(case):
    """Reported due where the ask cannot go, the driver ticked on every pass:
    433 ticks in 120 s instead of 12 for a CHAT caller owing a release."""
    h, io = pending(None)
    if case == 'initiator':
        h.role = 'initiator'
    if case == 'band-busy':
        h._band_quiet_s = lambda: 0.0
    if case == 'receiving':
        io.receiving = True
    _Clock.t += A._IDLE_POLL_S + 1
    assert h.idle_poll_due_in() is None


def test_a_lost_release_is_asked_after_in_one_cadence():
    h, io = pending(None)
    due = h.idle_poll_due_in()
    assert due is not None and due <= A._IDLE_POLL_S + .01
    _Clock.t += due
    h.idle_keepalive()
    assert len(io.sent) == 1 and h.turn == A._TURN_ASKED, io.msgs
