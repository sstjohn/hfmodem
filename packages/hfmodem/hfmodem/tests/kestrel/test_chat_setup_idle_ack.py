# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CHAT ON caller answers the answerer's idle connected-ack as VARA HF does."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from . import test_varac_answer_1020 as R

PEER, KESTREL = 'WI9YZA', 'W9SSJ'


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    R.Clock.t = 1000.
    monkeypatch.setattr(A, 'time', R.Clock)


def calling(bw='500', chat=True):
    io = R.IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw=bw, high_speed=True,
                                compression='TEXT', chat=chat)
    hs.originate(PEER, KESTREL)
    kind = F.connect_response(bw, 1)
    hs.on_rx_tones(F.handshake_tones(PEER, kind), F.connect_response(bw))
    assert hs.step == A._I_LINKSETUP_SENT, io.logs
    io.sent.clear()
    return hs, io


def ack(frame, bw='500', lead=0.02):
    return np.r_[np.zeros(int(lead * M.FS)),
                 M.synth_tone_pairs(F.control_bursts(KESTREL, bw, frame)[1]), np.zeros(M.FS)]


def stream(hs, x):
    for at in range(0, len(x), 1024):
        R.Clock.t += 1024 / M.FS
        hs.on_rx_stream(x[at:at + 1024])


def keyed(io, kind, bw='500', call=PEER):
    return bool(io.sent) and np.array_equal(io.sent[-1], M.synth_burst(call, F.for_bw(kind, bw)))


@pytest.mark.skipif(not R.CHAT.is_dir(), reason=f'recordings not present ({R.CHAT})')
def test_recorded_idle_ack_draws_the_f20_the_stock_caller_keyed():
    io = R.IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw='500', high_speed=True,
                                compression='TEXT', chat=True)
    hs.originate(PEER, KESTREL)
    R.feed(hs, R.audio(R.CHAT, 'chat-w9-on-callerclose-01-b2a'))
    assert hs.step == A._I_LINKSETUP_SENT, io.logs
    sent = len(io.sent)
    R.feed(hs, R.audio(R.CHAT, 'chat-w9-on-callerclose-03-b2a'))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.sent) == sent + 1 and keyed(io, F.SESSION_TURN_IDLE_SHORT), io.logs
    assert hs.turn == A._TURN_PEER
    assert R.recorded(R.audio(R.CHAT, 'chat-w9-on-callerclose-04-a2b'),
                      F.handshake_tones(PEER, F.for_bw(F.SESSION_TURN_IDLE_SHORT, '500')))


@pytest.mark.parametrize('bw', ['500', '2300'])
def test_idle_ack_draws_the_f20(bw):
    hs, io = calling(bw)
    hs.on_rx_audio(ack(1, bw))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.sent) == 1 and keyed(io, F.SESSION_TURN_IDLE_SHORT, bw), io.logs
    assert hs.turn == A._TURN_PEER


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_stream_waits_for_the_tail_of_a_late_idle_ack(bw):
    """Late in the answer window the preamble is read before the tail is in."""
    hs, io = calling(bw)
    stream(hs, ack(1, bw, lead=0.7))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.sent) == 1 and keyed(io, F.SESSION_TURN_IDLE_SHORT, bw), io.logs


def test_idle_ack_owes_no_greeting():
    hs, io = calling()
    hs.on_rx_audio(ack(1))
    assert hs._setup_greeting_at is None and not hs._setup_grant_pending


def test_refused_f20_is_retried_as_the_f20():
    hs, io = calling()
    io.tx_went_out = lambda: False
    hs.on_rx_audio(ack(1))
    assert hs._release_owed, io.logs
    io.tx_went_out = lambda: True
    io.sent.clear()
    R.Clock.t += 30
    hs.idle_keepalive()
    assert io.sent and keyed(io, F.SESSION_TURN_IDLE_SHORT), io.logs


def test_idle_ack_with_host_data_queued_asks_for_the_turn():
    hs, io = calling()
    hs.send(b'hello')
    hs.on_rx_audio(ack(1))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.sent) == 1 and keyed(io, F.SESSION_TURN_REQUEST_SHORT, call=KESTREL), io.logs
    assert hs.turn == A._TURN_ASKED and hs._txq


@pytest.mark.parametrize('frame,chat', [(0, True), (1, False)])
def test_traffic_ack_or_chat_off_still_confirms(frame, chat):
    hs, io = calling(chat=chat)
    hs.on_rx_audio(ack(frame))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert io.sent and keyed(io, F.SESSION_CONFIRM), io.logs


def test_32_symbol_idle_poll_behind_our_f20_is_answered_as_a_poll():
    """Xara polls the empty turn with 1078/63, which VARA HF holds a BW500 turn with."""
    hs, io = calling()
    hs.on_rx_audio(ack(1))
    io.sent.clear()
    R.Clock.t += 10
    poll = M.synth_burst(KESTREL, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, '500'))
    stream(hs, np.r_[np.zeros(4800), poll, np.zeros(M.FS)])
    assert len(io.sent) == 1 and keyed(io, F.SESSION_KEEPALIVE_B), io.logs
    assert hs.turn == A._TURN_PEER
