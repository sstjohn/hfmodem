# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A caller recovering a lost final ACK, with traffic queued, and the answerer.

Kestrel calling Kestrel over the OpenSlot air loop, 3 dB moderate fading,
2026-11-10: the answerer's level-6 last over was read and its ACK lost. The
caller answered a query with 850/63, the answerer took that as the ACK and
keyed pair 136, which the caller's recovery does not read, then asked for the
turn with its own traffic. The caller held the ask as unproven, and each end
waited on the other until the answerer closed 78 s later.

VARA HF confirms the repeated ACK at pair 135, and the caller's 850/1 and the
grant follow (N5YYF lost-ACK bench, 2026-10-25).

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock
from .test_limited_responder_query_recovery import on_air, pending, query, receiver
from .test_responder_control_stream import feed


def lost_final_ack():
    """The caller's 850/63, its answer to a query behind its lost final ACK."""
    h, hi = pending(final=True)
    h.chat = h._chat_profile = True
    c, ci = receiver(h)
    c.chat, c._released = True, True
    c._ask_not_before = _Clock.t + 100
    c.send(b'hello there')
    c.send(b'x' * 40)
    feed(c, on_air(hi.sent[-1]))
    assert ci.host == [b'last over'] and len(ci.sent) == 1, ci.msgs[-3:]
    # Above the hand-back level, as the cell's level-6 over was.
    h._tx_pending = h._tx_pending[:2] + (6,)
    _Clock.t += 2
    query(h, hi, 11)
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS)])
    assert np.array_equal(ci.sent[-1], M.synth_burst(c.caller, F.for_bw(F.SESSION_TURN_IDLE, '500')))
    assert c._receive_ack_recovery == 'ack' and c.turn == A._TURN_PEER
    _Clock.t -= len(ci.sent[-1]) / M.FS
    feed(h, np.r_[on_air(ci.sent[-1]), np.zeros(M.FS)])
    assert h._tx_pending is None and h.payload_ledger.confirmed == 9, hi.msgs[-3:]
    return h, hi, c, ci


def test_the_answerer_confirms_the_repeated_ack_as_varahf_does(clock):
    h, hi, c, ci = lost_final_ack()
    confirmation = M.synth_tone_pairs(F.responder_recovery_confirmation(h.called, '500'))
    assert np.array_equal(hi.sent[-1], confirmation), hi.msgs[-3:]
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS)])
    assert np.array_equal(ci.sent[-1], M.synth_burst(c.caller, F.for_bw(F.SESSION_TURN_REQUEST, '500')))
    assert c._receive_ack_recovery == 'request', ci.msgs[-3:]
    _Clock.t -= len(ci.sent[-1]) / M.FS
    feed(h, np.r_[on_air(ci.sent[-1]), np.zeros(M.FS)])
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS)])
    assert c.turn == A._TURN_OURS and c._tx_pending is not None, ci.msgs[-3:]


def test_the_answerers_ask_behind_an_unread_confirmation_draws_our_request(clock):
    h, hi, c, ci = lost_final_ack()
    # The confirmation is lost; the answerer, having released, asks with its own.
    h.send(b'answerer traffic ' * 5)
    ask = M.synth_burst(c.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, '500'))
    assert np.array_equal(hi.sent[-1], ask), hi.msgs[-3:]
    progress = c.progress
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS)])
    assert np.array_equal(ci.sent[-1], M.synth_burst(c.caller, F.for_bw(F.SESSION_TURN_REQUEST, '500')))
    assert c._receive_ack_recovery == 'request' and c.progress > progress, ci.msgs[-3:]
    _Clock.t -= len(ci.sent[-1]) / M.FS
    feed(h, np.r_[on_air(ci.sent[-1]), np.zeros(M.FS)])
    _Clock.t -= len(hi.sent[-1]) / M.FS
    feed(c, np.r_[on_air(hi.sent[-1]), np.zeros(M.FS)])
    assert c.turn == A._TURN_OURS and c._tx_pending is not None, ci.msgs[-3:]
    assert c._receive_ack_recovery is None
    feed(h, np.r_[on_air(ci.sent[-1]), np.zeros(M.FS)])
    assert b''.join(hi.host) == c._tx_pending[0][:43], hi.msgs[-3:]
