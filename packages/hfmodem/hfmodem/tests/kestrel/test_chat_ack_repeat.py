# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A BW500 CHAT link whose caller's first hand-back is lost.

Kestrel answering Kestrel, OpenSlot air loop, 3 dB moderate fading, seed
20261111 (2026-11-10): the caller read our connected-ack and answered it with
its f20, which faded (-3.9 dB); we keyed nothing more and closed the link 30 s
on while the caller waited for our poll. A VARA HF CHAT answerer repeats its
connected-ack every 3.0 s start-to-start until it is answered (stock silence
controls, working/varac-on-air/ve6dnl-1024, 2026-10-24: 11.30, 14.29, 17.30,
20.30 s and 11.31, 14.30, 17.28, 20.27, 23.26 s), and a caller answers the
repeat as it answered the first.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.contact import varac as v
from hfmodem.contact.bbs import HostCatalog
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from hfmodem.tests.contact.test_kestrel_pair import Channel, Peer
from .test_chat_caller_idle_close import CALLED as CALLED_CHAT, CALLER as CALLER_CHAT, caller
from .test_connect_half_open import answerer
from .test_idle_poll import clock as clock, _Clock as _Clock
from .test_responder_control_stream import feed


def test_an_unanswered_chat_ack_is_repeated_on_the_stock_cadence():
    hs, io = answerer('500', chat=True, greeting=b'<Q>')
    ack, acked_at = io.sent[0], hs._acked_at[1]
    due = acked_at + A._CHAT_ACK_REPEAT_S - len(ack) / M.FS
    assert hs.idle_poll_due_in() == pytest.approx(due - acked_at)
    for n in range(1, 4):
        _Clock.t = due - .01
        hs.idle_keepalive()
        assert len(io.sent) == n, io.msgs
        _Clock.t = due
        hs.idle_keepalive()
        assert len(io.sent) == n + 1 and np.array_equal(io.sent[-1], ack), io.msgs
        due += A._CHAT_ACK_REPEAT_S
    f20 = M.synth_burst(hs.called, F.for_bw(F.SESSION_TURN_IDLE_SHORT, '500'))
    feed(hs, np.r_[np.zeros(2400), f20, np.zeros(9600)])
    assert not hs._ack_unheard() and hs._poll_at is not None, io.msgs[-3:]
    _Clock.t += A._CHAT_ACK_REPEAT_S
    hs.idle_keepalive()
    assert not any(np.array_equal(s, ack) for s in io.sent[4:]), io.msgs[-3:]


def _connected_caller():
    hs, io = caller('500', turn=A._TURN_PEER)
    hs.state, hs.step = A.VaraState.CONNECTING, A._I_LINKSETUP_SENT
    hs._connected_by_ack(1)
    f20 = M.synth_burst(CALLED_CHAT, F.for_bw(F.SESSION_TURN_IDLE_SHORT, '500'))
    assert np.array_equal(io.sent[-1], f20), io.msgs[-3:]
    return hs, io, f20


def test_the_caller_answers_a_repeated_ack_with_its_hand_back_again():
    hs, io, f20 = _connected_caller()
    ack = M.synth_tone_pairs(F.control_bursts(CALLER_CHAT, '500', 1)[1])
    sent = len(io.sent)
    feed(hs, np.r_[np.zeros(int(1.6 * M.FS)), ack, np.zeros(9600)])
    assert len(io.sent) == sent + 1 and np.array_equal(io.sent[-1], f20), io.msgs[-3:]


def test_the_holders_poll_and_its_repeat_are_answered_as_polls():
    hs, io, f20 = _connected_caller()
    poll = M.synth_tone_pairs(F.control_bursts(CALLER_CHAT, '500', 1)[1])
    for wait in (A._HOLD_FIRST_POLL_S, A._CONTROL_REPEAT_S):
        sent = len(io.sent)
        feed(hs, np.r_[np.zeros(int(wait * M.FS)), poll, np.zeros(9600)])
        assert len(io.sent) == sent + 1 and not np.array_equal(io.sent[-1], f20), io.msgs[-3:]


class _LoseHandBack(Channel):
    """Drops the caller's first burst once it is connected: its f20."""
    def step(self):
        if self.air and not self.dropped:
            sender = self.air[0][0]
            if sender == 1 and self.peers[1].modem.station.state is A.VaraState.CONNECTED:
                self.air.popleft()
                self.dropped += 1
        super().step()


def test_a_lost_first_hand_back_still_draws_the_answerers_greeting(tmp_path, monkeypatch):
    channel = _LoseHandBack(None, 65536)
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=channel.clock))
    monkeypatch.setitem(v.CALLING_HZ, 20, 14_074_500)
    callee = Peer(channel, 'W9SSJ', 500, tmp_path / 'a', HostCatalog())
    calling = Peer(channel, 'W9SSJ-1', 500, tmp_path / 'b', HostCatalog())
    calling.command('/call W9SSJ cf')
    channel.until(lambda: channel.dropped, "the caller's hand-back sent")
    station = callee.modem.station
    channel.until(lambda: station.pending_bytes == 0 and callee.modem.confirmed_total > 0,
                  "the answerer's greeting delivered", seconds=20)
    assert station.state is A.VaraState.CONNECTED
