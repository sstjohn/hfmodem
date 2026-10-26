# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Kestrel receiving stock at BW500: an over outstanding is never asked across.

Bench 2, ks-rebased-0.6-r1 and -r2, 2026-10-21: kestrel called stock with 300 bytes queued and stock sent
first. Its opening L6 over drew our 8-symbol NAK, its L5 resend our 16-symbol
one, keyed 0.9 s behind the over, and stock's 289/361 query (1.38 s behind its
over) went out across it: stock never heard the second NAK. We named the query
correctly and answered it with the turn-request; stock took that as the L5
over's acknowledgement and its 90 bytes were lost.

Stock receivers in the same spot (19 stock pairs): no turn request while an over is outstanding; an unreadable
retry draws the same NAK again; a missed one draws nothing until the sender's
query, which draws the missing-DATA NAK. The forms follow whether any over of
the delivery has been read: 61/31 and 60/1117 before, the 8-symbol NAK and
61/541 after (th, th2, th3; tg, tk, ta, tc).
"""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_idle_poll import (CALLED, CALLER, _feed, _noise, _station,
                                                  clock)  # noqa: F401


def _receiving(opening=6):
    hs, io = _station("500")
    hs.high_speed = hs._link_high_speed = True
    hs.turn, hs._released = VA._TURN_PEER, True
    hs._expect_level = opening
    hs._lv_owed = hs._lv_prompt = True
    io.receiving = True
    hs.send(bytes(300))
    io.receiving = False
    assert hs._ask_owed and not _asks(io)
    return hs, io


def _asks(io):
    return sum(m.startswith("tx session-turn-request") for m in io.msgs)


def _unread(hs, io, level):
    keyed = len(io.sent)
    hs._irs_nak(phy.LevelRead(level, (), None, 1))
    assert len(io.sent) == keyed + 1
    return io.sent[-1]


def _query(kind=VF.SESSION_RESPONDER_OVER_IDLE_SHORT, lead=0.3):
    return np.concatenate([_noise(lead), MK.synth_burst(CALLED, VF.for_bw(kind, "500")),
                           _noise(0.6)])


def _ticks(hs, n=3):
    for _ in range(n):
        hs.idle_keepalive()


def _is(burst, kind):
    return len(burst) == len(MK.synth_burst(CALLED, VF.for_bw(kind, "500")))


def test_the_bench_2_sequence_draws_stocks_naks_and_never_the_ask():
    hs, io = _receiving(6)
    assert _is(_unread(hs, io, 6), VF.SESSION_DATA_NAK_CALLER_SHORT), "not stock's 61/31"
    assert _is(_unread(hs, io, 5), VF.SESSION_DATA_NAK_CALLER_SHORT)
    assert hs._expect_level == 4

    _feed(hs, _query())
    assert not _asks(io), "asked across the outstanding over"
    assert any(m.startswith("tx session-over-nak keyed") for m in io.msgs), io.msgs[-3:]
    assert hs._expect_level == 3, "stock resends L5 two down once it reads missing DATA"
    _ticks(hs)
    assert not _asks(io)


def test_an_unreadable_retry_draws_the_same_nak_again():
    """th, th2: L7, L6 and L5 unreadable, three 61/31 in a row; tc, ta: after a
    read over, the 8-symbol NAK twice."""
    hs, io = _receiving(7)
    for level in (7, 6, 5):
        assert _is(_unread(hs, io, level), VF.SESSION_DATA_NAK_CALLER_SHORT)
    _ticks(hs)
    assert not _asks(io)

    hs, io = _receiving(7)
    hs._note_close_level([phy.vara_body(b"x" * 89, CALLER)])
    nak = MK.synth_tone_pairs(VF.nak(CALLED, "500")[0])
    hs._expect_level = 6
    for level in (6, 5):
        assert len(_unread(hs, io, level)) == len(nak)
    assert not _asks(io)


def test_a_missed_retry_waits_for_the_query_and_names_missing_data():
    """th3: two 61/31, the L5 resend unheard, 289/331, 60/1117, B resends at L3.
    tk: an over read, then the query draws 61/541."""
    hs, io = _receiving(7)
    _unread(hs, io, 7)
    _unread(hs, io, 6)
    _feed(hs, _noise(4.0))
    _ticks(hs)
    assert not _asks(io) and len(io.sent) == 2, "keyed into the missed retry"
    _feed(hs, _query(VF.SESSION_RESPONDER_IDLE_SHORT, lead=0.1))
    assert any(m.startswith("tx session-over-nak keyed") for m in io.msgs), io.msgs[-3:]
    assert hs._expect_level == 3 and not _asks(io)

    hs, io = _receiving(7)
    _unread(hs, io, 7)
    hs._note_close_level([phy.vara_body(b"x" * 89, CALLER)])
    hs._expect_level, hs._rx_failed = 5, True
    hs._rx_resend = (5, ())
    _feed(hs, _noise(4.0))
    _feed(hs, _query(VF.SESSION_RESPONDER_IDLE_SHORT, lead=0.1))
    assert any(m.startswith("tx session-over-nak-short keyed") for m in io.msgs), io.msgs[-3:]
    assert not _asks(io)


def test_the_ask_is_owed_again_once_the_resend_is_read():
    hs, io = _receiving(6)
    _unread(hs, io, 6)
    assert not hs._owes_turn()
    hs._note_close_level([phy.vara_body(b"x" * 40, CALLER)])
    assert hs._rx_resend is None and hs._owes_turn()
