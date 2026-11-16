# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CHAT caller closing behind the empty turn it just handed back.

Kestrel answering Kestrel, OpenSlot air loop, 3 dB moderate fading, seeds
20261109 and 20261110 (2026-11-10). The caller's host wrote DISCONNECT 0.16 s
into its own f20, and its frame-0 ask followed the f20 with no gap, as a VARA
HF caller's ask follows its release 0.016 s on (gate4/stock 45.93 s). The
answerer named the f20 with the ask's head already in its buffer and dropped
it. The third ask was granted, but the caller had closed at once behind it and
the grant keyed across the close.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_answerer_traffic_handback import _answered
from .test_chat_caller_idle_close import CALLED, caller
from .test_idle_poll import clock as clock, _Clock as _Clock
from .test_responder_control_stream import feed


def _hand_back(hs, form):
    if form == 'f20':
        return M.synth_burst(hs.called, F.for_bw(F.SESSION_TURN_IDLE_SHORT, '500'))
    return M.synth_tone_pairs(A._caller_idle_forms(hs.called, '500')[0])


@pytest.mark.parametrize('form', ['f20', 'idle-pair'])
@pytest.mark.parametrize('block', [960, 4096])
def test_an_ask_keyed_straight_behind_the_hand_back_is_granted(form, block):
    hs, io = _answered(1)
    hs._txq.clear()
    ask = M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 0)[0])
    sent = len(io.sent)
    feed(hs, np.r_[np.zeros(2400), _hand_back(hs, form), ask, np.zeros(9600)], block)
    assert len(io.sent) == sent + 1 and hs.turn == A._TURN_PEER, io.msgs[-4:]


def test_a_hand_back_keeps_no_audio_from_before_our_own_keying():
    hs, io = _answered(1)
    hs._txq.clear()
    feed(hs, np.r_[np.zeros(2400), _hand_back(hs, 'f20'), np.zeros(4096)], 8192)
    assert hs.turn == A._TURN_OURS, io.msgs[-4:]
    assert len(hs._ans_buf) <= hs._heard
    hs._key(True)
    hs._key(False)
    hs._keep_audio_behind_hand_back(np.zeros(8192), 0)
    assert len(hs._ans_buf) == 0


@pytest.mark.parametrize('then', ['granted', 'silent', 'refused-ask'])
def test_the_last_ask_is_answered_before_the_close_goes_unasked(then):
    hs, io = caller('500', turn=A._TURN_OURS)
    assert hs._release_turn(idle=True)
    for ask in range(A._TURN_MAX_ASK):
        if ask:
            _Clock.t += A._CLOSE_ASK_REPEAT_S + .01
        hs.disconnect(acquire=True)
    assert hs._close_asks == A._TURN_MAX_ASK
    _Clock.t += .1
    hs.disconnect(acquire=True)
    assert hs.state is A.VaraState.CONNECTED, io.msgs[-3:]
    if then == 'granted':
        grant = M.synth_burst(CALLED, F.for_bw(F.SESSION_TURN_RELEASE_RESPONDER, '500'))
        feed(hs, np.r_[np.zeros(2400), grant, np.zeros(9600)])
    else:
        if then == 'refused-ask':
            # An ask the transport refused, off the peer's poll, is not one owed an answer.
            _Clock.t += A._CLOSE_ASK_REPEAT_S - .2
            io.tx_went_out = lambda: False
            hs._ask_to_close()
            io.tx_went_out = lambda: True
        _Clock.t += .2 if then == 'refused-ask' else A._CLOSE_ASK_REPEAT_S
        hs.disconnect(acquire=True)
    assert hs.state is A.VaraState.DISCONNECTED, io.msgs[-3:]
    assert np.array_equal(io.sent[-1], M.synth_burst(CALLED, F.for_bw(F.SESSION_DISCONNECT_REQ, '500')))
