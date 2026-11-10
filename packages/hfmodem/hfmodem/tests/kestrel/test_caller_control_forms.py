# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A caller's idle and break, read in whichever of their forms it keys.

Xara calling Kestrel over the OpenSlot air loop, BW500 CHAT, clean channel,
2026-11-09. Behind our answer to its last over Xara asked for the turn with
the 11-symbol break, unread, and 6 s later with the 32-symbol one. Holding the
empty turn we handed it, it polled with the 32-symbol idle every 3.16 s,
unread, until its watchdog closed the link. Each form says what the others do.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_answerer_traffic_handback import _answered, _over
from .test_idle_poll import clock as clock
from .test_responder_control_stream import feed, station


def _held():
    """Our last over taken by the caller's idle, and the empty turn handed over."""
    hs, io = station()
    hs.chat = hs._chat_profile = True
    hs.turn = A._TURN_OURS
    hs.send(b'K9TST last over')
    feed(hs, np.r_[np.zeros(4800), M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 1)[0]),
                   np.zeros(4800)])
    assert hs.turn == A._TURN_PEER and hs._released_form == 'f20', io.msgs[-3:]
    feed(hs, np.zeros(10 * M.FS))
    return hs, io


def _behind_our_idle_answer():
    """The caller's last over answered with our frame 1."""
    hs, io = _answered(1)
    hs._txq.clear()
    feed(hs, np.r_[np.zeros(1200), _over(hs, b' ok\r', 4), np.zeros(36000)])
    assert hs._answer_owed == A._OWED_RELEASE and hs._over_answer == ('control', 1)
    return hs, io


def _heard(hs, burst):
    sent = len(hs.io.sent)
    feed(hs, np.r_[np.zeros(2400), burst, np.zeros(M.FS)])
    return hs.io.sent[sent:]


def _idle(hs, form):
    return (M.synth_burst(hs.caller, F.for_bw(F.SESSION_TURN_IDLE, '500')) if form == 32
            else M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 1)[0]))


def _break(hs, form):
    if form == 11:
        return M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 0)[0])
    kind = F.SESSION_TURN_REQUEST if form == 32 else F.SESSION_TURN_REQUEST_SHORT
    return M.synth_burst(hs.caller, F.for_bw(kind, '500'))


@pytest.mark.parametrize('form', [11, 32])
def test_the_callers_idle_in_a_turn_we_handed_it_draws_the_idle_response(form):
    hs, io = _held()
    for _ in range(2):
        keyed = _heard(hs, _idle(hs, form))
        assert len(keyed) == 1, io.msgs[-3:]
        np.testing.assert_array_equal(keyed[0], M.synth_burst(
            hs.called, F.for_bw(F.SESSION_IDLE_RESPONSE, '500')))
    assert hs.state is A.VaraState.CONNECTED and hs.turn == A._TURN_PEER


@pytest.mark.parametrize('form', [11, 32])
def test_the_callers_idle_in_a_turn_we_handed_it_draws_our_ask(form):
    hs, io = _held()
    hs._ask_not_before = 0.0
    hs._txq.append(b'answerer text')
    keyed = _heard(hs, _idle(hs, form))
    assert hs.turn == A._TURN_ASKED and keyed, io.msgs[-3:]
    np.testing.assert_array_equal(keyed[-1], M.synth_burst(
        hs.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, '500')))


@pytest.mark.parametrize('form', [11, 17, 32])
def test_the_callers_break_behind_our_idle_answer_draws_the_grant(form):
    hs, io = _behind_our_idle_answer()
    keyed = _heard(hs, _break(hs, form))
    assert len(keyed) == 1, io.msgs[-3:]
    np.testing.assert_array_equal(keyed[0], M.synth_tone_pairs(F.turn_grant(hs.called, '500')))
    assert hs._answer_owed == A._OWED_OVER and hs.turn == A._TURN_PEER
