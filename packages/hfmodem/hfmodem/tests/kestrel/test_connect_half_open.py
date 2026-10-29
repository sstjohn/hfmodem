# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Link establishment between W9SSJ-1 and W9SSJ, 7105 kHz BW500 CHAT, 2026-10-28.

Three of four calls came up at the answerer only: the caller missed the
connected-ack, re-keyed its link-setup, and the answerer, already CONNECTED,
refused the repeat as a burst with no data frame in it. One caller came up on a
stranger's burst 5.8 s behind its link-setup. And an answerer's greeting queued
at CONNECTED was never keyed under CHAT.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF
from .test_idle_poll import clock as clock, _Clock as _Clock, _IO as _IO
from .test_responder_control_stream import feed
from .test_responder_initial_greeting import confirm

CALLER, CALLED = 'W9SSJ-1', 'W9SSJ'


def answerer(bw='500', *, chat=True, marked=False, greeting=b''):
    io = _IO()
    hs = A.VaraStationHandshake([CALLED], io, bw=bw)
    hs.role, hs.called, hs.caller = 'responder', CALLED, CALLER
    hs.state = A.VaraState.CONNECTING
    hs.chat = hs._chat_profile = chat
    hs.high_speed = True
    if greeting:
        io.connected = lambda *args: hs.send(greeting)
    assert hs._take_link_setup(F.link_setup_frame(CALLER, bw=bw, ask=1, high_speed=marked))
    assert len(io.sent) == 1
    return hs, io


def link_setup(bw='500', caller=CALLER):
    return np.concatenate([np.zeros(4800), OF.link_setup_tx(caller, bw=bw), np.zeros(48000)])


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
def test_repeated_link_setup_behind_our_ack_is_acked_again(bw, chat):
    hs, io = answerer(bw, chat=chat)
    progress = hs.progress
    feed(hs, link_setup(bw))
    assert len(io.sent) == 2, io.msgs
    assert np.array_equal(io.sent[1], io.sent[0])
    assert hs.state == A.VaraState.CONNECTED and hs.progress == progress
    feed(hs, link_setup(bw))
    assert len(io.sent) == 3 and np.array_equal(io.sent[2], io.sent[0])


def test_link_setup_from_another_caller_is_not_acked():
    hs, io = answerer()
    feed(hs, link_setup(caller='N0ZZZ'))
    assert len(io.sent) == 1, io.msgs


def test_link_setup_after_the_caller_was_heard_is_not_acked():
    hs, io = answerer(chat=False)
    confirm(hs)
    sent = len(io.sent)
    feed(hs, link_setup())
    assert len(io.sent) == sent, io.msgs


@pytest.mark.parametrize('chat', [True, False])
def test_an_unconfirmed_link_is_closed_not_left_connected(chat):
    hs, io = answerer(chat=chat, greeting=b'<Q>')
    _Clock.t += A._ACK_UNHEARD_S - 1
    hs.idle_keepalive()
    assert hs.state == A.VaraState.CONNECTED
    _Clock.t += 2
    hs.idle_keepalive()
    assert hs.state == A.VaraState.DISCONNECTED, io.msgs
    assert any('never confirmed the link' in m for m in io.msgs)


def test_each_re_ack_restarts_the_unconfirmed_bound():
    hs, io = answerer()
    _Clock.t += A._ACK_UNHEARD_S - 5
    feed(hs, link_setup())
    assert len(io.sent) == 2
    _Clock.t += 10
    hs.idle_keepalive()
    assert hs.state == A.VaraState.CONNECTED


def test_a_confirmed_link_is_not_held_to_the_unconfirmed_bound():
    hs, io = answerer(chat=False)
    confirm(hs)
    _Clock.t += A._ACK_UNHEARD_S + 1
    hs.idle_keepalive()
    assert hs.state == A.VaraState.CONNECTED


def test_an_over_behind_an_acknowledged_one_confirms_the_link(monkeypatch):
    hs, io = answerer()
    monkeypatch.setattr(hs, '_peer_data_over', lambda samples: [A._UNREAD_OVER])
    assert hs._answer_data_over([])
    _Clock.t += A._ACK_UNHEARD_S + 1
    hs.idle_keepalive()
    assert hs.state == A.VaraState.CONNECTED, io.msgs
    assert not any('never confirmed the link' in m for m in io.msgs)


@pytest.mark.parametrize('marked', [False, True])
def test_chat_greeting_queued_at_connected_is_delivered_on_the_handing_confirm(marked):
    greeting = b'<AWQ>Away, back soon'
    hs, io = answerer(marked=marked, greeting=greeting)
    assert hs._tx_pending is None and hs.payload_ledger.pending == len(greeting)
    confirm(hs)
    assert len(io.sent) == 2 and hs._tx_pending is None
    assert np.array_equal(io.sent[1], M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 0)[1]))
    confirm(hs, long=True)
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None, io.msgs
    assert len(io.sent) == 3 and hs.payload_ledger.confirmed == 0


def test_chat_greeting_reaches_a_caller_with_nothing_to_say(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hfmodem.contact import varac as v
    from hfmodem.contact.bbs import HostCatalog
    from hfmodem.tests.contact.test_kestrel_pair import Channel, Peer

    channel = Channel(None, 65536)
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=channel.clock))
    monkeypatch.setitem(v.CALLING_HZ, 20, 14_074_500)
    callee = Peer(channel, CALLED, 500, tmp_path / 'a', HostCatalog())
    caller = Peer(channel, CALLER, 500, tmp_path / 'b', HostCatalog())
    caller.command(f'/call {CALLED} cf')
    channel.until(lambda: callee.modem.station.state == A.VaraState.CONNECTED
                  and callee.modem.station.pending_bytes, 'callee greeting queued')
    channel.until(lambda: callee.modem.station.pending_bytes == 0
                  and callee.modem.confirmed_total > 0, 'callee greeting delivered')


def test_chat_confirm_with_nothing_queued_is_still_answered_with_control():
    hs, io = answerer()
    confirm(hs)
    assert len(io.sent) == 2 and hs._tx_pending is None and hs.turn != A._TURN_OURS
    pairs = F.control_bursts(hs.caller, '500', 0)[1]
    assert np.array_equal(io.sent[1], M.synth_tone_pairs(pairs))


def awaiting_ack(bw):
    io = _IO()
    hs = A.VaraStationHandshake([CALLED], io, bw=bw)
    hs.originate('NS0A', CALLED)
    hs.step = A._I_LINKSETUP_SENT
    return hs, io


@pytest.mark.parametrize('bw', ['500', '2300'])
@pytest.mark.parametrize('gap,up', [(0.15, True), (5.8, False)])
def test_bracketed_ack_is_held_to_the_answer_window(bw, gap, up):
    hs, io = awaiting_ack(bw)
    ack = M.synth_tone_pairs(F.control_bursts(CALLED, bw, 0)[1])
    # The stream heard the turnaround but not the ack in it; the bracket did.
    feed(hs, np.zeros(int(gap * M.FS) + len(ack) + 6144))
    assert hs.state == A.VaraState.CONNECTING
    hs.on_rx_audio(np.concatenate([np.zeros(4096), ack, np.zeros(6144)]))
    assert (hs.state == A.VaraState.CONNECTED) is up, io.msgs


def test_a_link_setup_that_will_not_decode_leaves_the_ack_bound_standing():
    """At the wide bandwidths a reference-locked frame that fails its CRC is
    reported as an over that will not decode, and the caller's re-keyed
    link-setup on a weak path is one: it must leave the next clean repeat to
    be acked again rather than reopen the link at one end."""
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    hs, io = answerer('2300')
    frame = bytearray(F.link_setup_frame(CALLER, bw='2300'))
    frame[-1] ^= 0xFF
    bad = np.concatenate([np.zeros(4800), tx.synth_frame(bytes(frame), over=0), np.zeros(48000)])
    feed(hs, bad)
    hs.on_rx_audio(bad)
    assert any('will not decode' in m for m in io.msgs), io.msgs
    sent = len(io.sent)
    feed(hs, link_setup('2300'))
    assert len(io.sent) == sent + 1 and np.array_equal(io.sent[-1], io.sent[0]), io.msgs
