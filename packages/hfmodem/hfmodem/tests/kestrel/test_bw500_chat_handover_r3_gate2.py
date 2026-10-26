# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Kestrel calls stock at BW500, chat on: the turn after our high-speed close.

Bench 3 r3 gate 2, 2026-10-21 (working/varac-on-air/bench3/r3/gate2): after
12.5 min of idle our 24-byte line went out at level 6 (825.4-829.5 s). Stock
answered with its control frame 1 (829.6), we keyed the caller idle answer
(830.3) and went on holding the turn, polling. Stock answered the first of its
own polls itself (840.9), then polled every 2.3 s (843.2-889.8), unanswered,
with its 22-byte line queued from 853.3; it closed at 883.3 without ever having
the turn, so no close frame went out (890.9 is its CW ID), and our host
answered the next CONNECT with WRONG.

Behind a close at level 6 or above the idle answer is the 11-symbol form of
the f20 release (2026-10-18 data runs): the turn is stock's after it, its polls
are answered, and its frame 0 draws the caller grant.
"""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_idle_poll import (CALLED, CALLER, _feed, _keepalives, _noise,
                                                  _peer_polls, _station, clock)  # noqa: F401


def _stock(frame):
    return np.concatenate([MK.synth_tone_pairs(VF.control_bursts(CALLER, "500", frame)[1]),
                           _noise(0.6)])


def _closed_at_level_6():
    hs, io = _station("500")
    hs.chat = True
    hs.turn = VA._TURN_OURS
    hs._tx_pending = (phy.vara_body(b"after idle from kestrel\r", CALLER), 2, 6)
    _feed(hs, _stock(1))
    return hs, io


def _keyed(io, name):
    return sum(m.startswith(f"tx {name}") for m in io.msgs)


def test_the_idle_answer_hands_the_turn_to_stock_and_its_polls_are_answered():
    hs, io = _closed_at_level_6()
    assert _keyed(io, "session-turn-idle-short") == 1
    assert hs.turn == VA._TURN_PEER and not hs._holding_idle(), "held the turn behind the idle answer"

    _peer_polls(hs, io, 12)
    assert _keepalives(io, VF.SESSION_KEEPALIVE_B) == 12, io.msgs[-6:]
    assert hs.state is VA.VaraState.CONNECTED

    _feed(hs, _stock(0))
    assert _keyed(io, "session-turn-release") == 1 and hs.turn == VA._TURN_PEER

    _feed(hs, MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_DISCONNECT_REQ_RESPONDER, "500")))
    _feed(hs, _noise(0.5))
    assert hs.state is VA.VaraState.DISCONNECTED


def test_stocks_poll_while_we_hold_the_turn_concedes_it_and_is_answered():
    hs, io = _station("500")
    hs.chat = True
    hs.turn = VA._TURN_OURS
    hs._hold_idle()
    sent = len(io.sent)
    _feed(hs, _stock(1))
    assert hs.turn == VA._TURN_PEER and not hs._holding_idle()
    assert _keepalives(io, VF.SESSION_KEEPALIVE_B) == 1 and len(io.sent) == sent + 1
