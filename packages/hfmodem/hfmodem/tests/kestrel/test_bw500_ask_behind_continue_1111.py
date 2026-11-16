# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A caller whose continue asked for the next over does not ask for the turn
before that over has been answered.

Kestrel to Kestrel, BW500 CHAT, 3 dB moderate fading, seed 20261111: the
caller answered the answerer's full over #3 with a continue and re-keyed it on
the query. Over #4 was under way, faded below the band gate, when the caller's
host queued a line; the caller keyed its turn-request 3.6 s into the 4.4 s over
and missed its tail. The turn change ended the receive window, so each of the
answerer's seven queries that did read here found no answer to repeat, and the
answerer closed with the line unsent.

A VARA HF receiver never asks while an over of the peer's delivery is
outstanding: in 19 VARA HF pairs with traffic queued both ways, the turn moved
only behind the sender's last over (bench, 2026-10-21).
"""
import numpy as np

from hfmodem.kestrel.vara import vara_arq as A
from .test_answer_query_recovery import answer_query, caller_owing
from .test_idle_poll import clock as clock  # noqa: F401
from .test_responder_control_stream import feed


def _asks(io):
    return sum(m.startswith('tx session-turn-request') for m in io.msgs)


def test_a_line_queued_behind_a_continue_waits_for_the_over(monkeypatch):
    hs, io = caller_owing(monkeypatch)
    hs.chat = True
    hs.send(b'28 C2 W5TST to K9TST chat 2 ok\n')
    feed(hs, np.zeros(A.MK.FS))
    for _ in range(3):
        hs.idle_keepalive()
    assert not _asks(io), 'asked across the over its continue asked for'
    assert hs._ask_owed and hs._answer_owed == A._OWED_OVER

    feed(hs, answer_query())
    assert any('answering again' in m for m in io.msgs), io.msgs[-4:]
    assert not _asks(io)


def test_a_continue_inside_our_ask_holds_the_retries(monkeypatch):
    """The peer answered our ask with a delivery and we continued its over: the
    request is not repeated across the next one."""
    hs, io = caller_owing(monkeypatch)
    window = hs._rx_window.A
    hs.turn = A._TURN_ASKED
    hs._rx_window.A = window
    assert hs._rx_window.A.kind == 'continue' and hs._answer_owed == A._OWED_OVER
    hs._asked, hs._txq = 1, [b'x']
    for _ in range(4):
        A.time.t += 3.0
        hs.idle_keepalive()
    assert not _asks(io), 'asked again across the over its continue asked for'


def test_a_debt_the_turn_change_dropped_does_not_hold_the_queue(monkeypatch):
    hs, io = caller_owing(monkeypatch)
    hs.turn = A._TURN_ASKED
    hs.turn = A._TURN_PEER
    assert hs._rx_window.A is None and hs._answer_owed == A._OWED_OVER
    hs.send(b'28 C2 W5TST to K9TST chat 2 ok\n')
    feed(hs, np.zeros(A.MK.FS))
    for _ in range(3):
        A.time.t += 3.0
        hs.idle_keepalive()
    assert _asks(io), 'an orphaned continue held the queue'


def _asked_behind_continue(monkeypatch):
    hs, io = caller_owing(monkeypatch)
    window = hs._rx_window.A
    hs.turn = A._TURN_ASKED
    hs._rx_window.A = window
    hs._asked, hs._txq = 1, [b'x']
    return hs, io


def test_a_held_ask_still_spends_the_silence_budget(monkeypatch):
    """The over our continue asked for never comes, nor any query: the NAK
    behind it waits on a resend, and the silence is charged as in the peer's
    turn rather than holding the link open with the line queued."""
    hs, io = _asked_behind_continue(monkeypatch)
    for _ in range(20):
        feed(hs, np.zeros(12 * A.MK.FS))
        hs.idle_keepalive()
        if hs.state is not A.VaraState.CONNECTED:
            break
    assert hs.state is not A.VaraState.CONNECTED, (hs.turn, hs._since_progress)
    assert not _asks(io)


def test_a_held_ask_waits_out_the_over_it_expects(monkeypatch):
    """Six seconds behind our continue the over it asked for may still be on
    the air: nothing is keyed across it, as in the peer's turn."""
    hs, io = _asked_behind_continue(monkeypatch)
    window = hs._expected_window_samples() / A.MK.FS
    assert window > 6.0
    sent = len(io.sent)
    feed(hs, np.zeros(6 * A.MK.FS))
    A.time.t = hs._unkeyed_at + 6.0
    hs.idle_keepalive()
    assert len(io.sent) == sent, io.msgs[-2:]
