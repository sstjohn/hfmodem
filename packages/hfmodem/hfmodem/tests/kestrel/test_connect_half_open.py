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


def answerer(bw='500', *, chat=True, marked=False, greeting=b'', io=None):
    io = _IO() if io is None else io
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


@pytest.mark.parametrize('chat', [True, False])
def test_repeated_link_setup_behind_our_ack_is_acked_again(chat):
    hs, io = answerer('500', chat=chat)
    progress = hs.progress
    feed(hs, link_setup('500'))
    assert len(io.sent) == 2, io.msgs
    assert np.array_equal(io.sent[1], io.sent[0])
    assert hs.state == A.VaraState.CONNECTED and hs.progress == progress
    feed(hs, link_setup('500'))
    assert len(io.sent) == 3 and np.array_equal(io.sent[2], io.sent[0])


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
def test_wide_repeated_link_setup_is_not_acked_again(bw, chat):
    hs, io = answerer(bw, chat=chat)
    progress, acked_at = hs.progress, hs._acked_at
    for delta in (12.6, 25.2):
        _Clock.t = acked_at[1] + delta
        feed(hs, link_setup(bw))
        assert len(io.sent) == 1, io.msgs
        assert hs.state == A.VaraState.CONNECTED and hs.progress == progress
        assert hs._acked_at == acked_at
        assert hs.idle_poll_due_in() == 0
    assert any('not re-acked' in m for m in io.msgs)
    assert hs._repeated_link_setup(F.link_setup_frame(CALLER, bw=bw))
    assert len(io.sent) == 1 and hs._acked_at == acked_at


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
def test_link_setup_from_another_caller_is_not_acked(bw, chat):
    hs, io = answerer(bw, chat=chat)
    acked_at, progress = hs._acked_at, hs.progress
    feed(hs, link_setup(bw, caller='N0ZZZ'))
    assert len(io.sent) == 1, io.msgs
    assert hs._acked_at == acked_at and hs.progress == progress
    assert hs.state == A.VaraState.CONNECTED


def test_link_setup_after_the_caller_was_heard_is_not_acked():
    hs, io = answerer(chat=False)
    confirm(hs)
    sent = len(io.sent)
    feed(hs, link_setup())
    assert len(io.sent) == sent, io.msgs


@pytest.mark.parametrize('chat', [True, False])
@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_an_unconfirmed_link_is_closed_not_left_connected(bw, chat):
    hs, io = answerer(bw, chat=chat, greeting=b'<Q>')
    _Clock.t += A._ACK_UNHEARD_S - 1
    hs.idle_keepalive()
    assert hs.state == A.VaraState.CONNECTED
    _Clock.t += 2
    hs.idle_keepalive()
    assert hs.state == A.VaraState.DISCONNECTED, io.msgs
    assert any('never confirmed the link' in m for m in io.msgs)


def padded(x):
    return np.concatenate([np.zeros(4800), x, np.zeros(24000)])


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
def test_wide_unheard_ack_turn_request_train(bw, chat):
    hs, io = answerer(bw, chat=chat)
    acked_at, progress = hs._acked_at, hs.progress
    ask = M.synth_burst(CALLER, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, bw))
    assert hs.idle_poll_due_in() == pytest.approx(5.7)
    for n, delta in enumerate((5.7, 9.34, 12.98), 1):
        _Clock.t = acked_at[1] + delta - .001
        assert hs.idle_poll_due_in() == pytest.approx(.001)
        hs.idle_keepalive()
        assert len(io.sent) == n
        _Clock.t = acked_at[1] + delta
        assert hs.idle_poll_due_in() == 0
        hs.idle_keepalive()
        assert len(io.sent) == n + 1, io.msgs
        assert np.array_equal(io.sent[-1], ask)
        assert hs.idle_poll_due_in() == pytest.approx(3.64)
        assert hs.turn == A._TURN_PEER
        assert hs.progress == progress and hs._acked_at == acked_at
    _Clock.t = acked_at[1] + 29.9
    hs.idle_keepalive()
    assert hs.idle_poll_due_in() == pytest.approx(.1)
    _Clock.t = acked_at[1] + 30
    hs.idle_keepalive()
    assert hs.state == A.VaraState.DISCONNECTED


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('sent', [True, False])
def test_delayed_wide_ack_train_leaves_a_listening_gap_after_unkey(bw, sent):
    class TimedIO(_IO):
        def tx(self, samples):
            super().tx(samples)
            _Clock.t += len(samples) / M.FS

    hs, io = answerer(bw, io=TimedIO())
    acked_at = hs._acked_at[1]
    ask = M.synth_burst(CALLER, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, bw))
    duration = len(ask) / M.FS
    listening_gap = 3.64 - duration
    _Clock.t = acked_at + 5.7
    io.receiving = True
    hs.idle_keepalive()
    assert len(io.sent) == 1 and hs.idle_poll_due_in() is None
    _Clock.t = acked_at + 9.20
    hs.idle_keepalive()
    assert len(io.sent) == 1
    io.receiving = False
    io.tx_went_out = lambda: sent

    for n in (2, 3):
        hs.idle_keepalive()
        assert len(io.sent) == n and np.array_equal(io.sent[-1], ask)
        assert hs._ack_turn_asked is sent
        unkeyed_at = _Clock.t
        due = hs.idle_poll_due_in()
        assert due == pytest.approx(listening_gap)
        hs.idle_keepalive()
        assert len(io.sent) == n
        _Clock.t = unkeyed_at + due - .001
        hs.idle_keepalive()
        assert len(io.sent) == n
        _Clock.t = unkeyed_at + due
    assert io.at[-1] - (io.at[-2] + duration) >= listening_gap - 1e-9



@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_slightly_late_ticks_keep_the_wide_ack_train_cadence(bw):
    class TimedIO(_IO):
        def tx(self, samples):
            super().tx(samples)
            _Clock.t += len(samples) / M.FS

    hs, io = answerer(bw, io=TimedIO())
    acked_at = hs._acked_at[1]
    starts = []
    _Clock.t = acked_at + 5.7
    for _ in range(4):
        _Clock.t += hs.idle_poll_due_in() + .001
        starts.append(_Clock.t)
        hs.idle_keepalive()
    assert len(io.sent) == 5
    assert np.diff(starts) == pytest.approx([3.641] * 3, abs=.002)

@pytest.mark.parametrize('chat', [True, False])
def test_bw500_unheard_ack_has_no_turn_request_train(chat):
    hs, io = answerer('500', chat=chat)
    acked_at = hs._acked_at
    for delta in (5.7, 9.34, 12.98):
        _Clock.t = acked_at[1] + delta
        assert hs.idle_poll_due_in() is None
        hs.idle_keepalive()
    assert len(io.sent) == 1, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
def test_wide_ack_train_waits_for_receive_and_refused_ask_waits_one_slot(bw, chat):
    hs, io = answerer(bw, chat=chat)
    acked_at = hs._acked_at
    _Clock.t = acked_at[1] + 5.7
    io.receiving = True
    assert hs.idle_poll_due_in() is None
    hs.idle_keepalive()
    assert len(io.sent) == 1
    io.receiving = False
    assert hs.idle_poll_due_in() == 0
    io.tx_went_out = lambda: False
    hs.idle_keepalive()
    assert len(io.sent) == 2 and not hs._ack_turn_asked
    assert hs.idle_poll_due_in() == pytest.approx(3.64)
    hs.idle_keepalive()
    assert len(io.sent) == 2
    io.tx_went_out = lambda: True
    _Clock.t = acked_at[1] + 9.34
    hs.idle_keepalive()
    assert len(io.sent) == 3 and hs._ack_turn_asked
    assert hs.idle_poll_due_in() == pytest.approx(3.64)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
@pytest.mark.parametrize('heard', ['progress', 'acked-cleared', 'setup-query'])
def test_hearing_the_caller_stops_the_wide_ack_train(bw, chat, heard):
    hs, io = answerer(bw, chat=chat, greeting=b'queued greeting')
    _Clock.t = hs._acked_at[1] + 5.7
    hs.idle_keepalive()
    assert len(io.sent) == 2, io.msgs
    if heard == 'setup-query':
        assert hs._setup_query_pending() and hs.turn == A._TURN_PEER
        query = M.synth_burst(CALLED, F.for_bw(F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0], bw))
        feed(hs, padded(query))
        assert len(io.sent) == 3, io.msgs
        assert np.array_equal(io.sent[-1], M.synth_burst(
            CALLER, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, bw)))
        assert hs._tx_pending is None and hs.payload_ledger.pending == len(b'queued greeting')
    elif heard == 'progress':
        hs._progressed()
    else:
        hs._acked_at = None
    assert not hs._ack_unheard()
    sent = len(io.sent)
    if chat:
        assert hs.idle_poll_due_in() is None
        for delta in (3.64, 3.64):
            _Clock.t += delta
            hs.idle_keepalive()
        assert len(io.sent) == sent, io.msgs
    else:
        # Hearing the caller ends the ack train; queued WINLINK traffic waits 5.7 s behind our last unkey.
        unkeyed = hs._unkeyed_at
        assert hs.idle_poll_due_in() == pytest.approx(unkeyed + 5.7 - _Clock.t)
        _Clock.t = unkeyed + 3.64
        hs.idle_keepalive()
        assert len(io.sent) == sent, io.msgs
        _Clock.t = unkeyed + 5.7 - .001
        assert hs.idle_poll_due_in() == pytest.approx(.001)
        hs.idle_keepalive()
        assert len(io.sent) == sent, io.msgs
        _Clock.t = unkeyed + 5.7
        assert hs.idle_poll_due_in() == 0
        hs.idle_keepalive()
        assert len(io.sent) == sent + 1, io.msgs
        assert np.array_equal(io.sent[-1], M.synth_burst(
            CALLER, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, bw)))
        assert hs.turn == A._TURN_ASKED and hs._tx_pending is None
        assert hs.payload_ledger.pending == len(b'queued greeting')


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
@pytest.mark.parametrize('greeting', [b'', b'responder greeting'])
def test_wide_ack_erasure_recovers_on_stream_train_ask(bw, chat, greeting):
    caller_io, responder_io = _IO(), _IO()
    caller = A.VaraStationHandshake([CALLER], caller_io, bw=bw)
    caller.role, caller.called, caller.caller = 'initiator', CALLED, CALLER
    caller.state, caller.step = A.VaraState.CONNECTING, A._I_LINKSETUP_SENT
    caller.chat = caller._chat_profile = chat
    assert caller._tx_link_setup()
    responder = A.VaraStationHandshake([CALLED], responder_io, bw=bw)
    responder.role, responder.called = 'responder', CALLED
    responder.state, responder.step = A.VaraState.CONNECTING, A._R_RESP_SENT
    responder.chat = responder._chat_profile = chat
    if greeting:
        responder_io.connected = lambda *args: responder.send(greeting)
    feed(responder, padded(caller_io.sent[0]))
    assert responder.state == A.VaraState.CONNECTED and len(responder_io.sent) == 1
    # Erase the ack on the caller's stream.
    feed(caller, padded(np.zeros_like(responder_io.sent[0])))
    assert caller.state == A.VaraState.CONNECTING and len(caller_io.sent) == 1
    _Clock.t = responder._acked_at[1] + 5.7
    responder.idle_keepalive()
    assert len(responder_io.sent) == 2, responder_io.msgs
    ask = M.synth_burst(CALLER, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, bw))
    assert np.array_equal(responder_io.sent[-1], ask)
    feed(caller, padded(ask))
    assert caller.state == A.VaraState.CONNECTED, caller_io.msgs
    assert len(caller_io.sent) == 2
    release = M.synth_burst(CALLED, F.for_bw(F.SESSION_TURN_RELEASE, bw))
    assert np.array_equal(caller_io.sent[-1], release), caller_io.msgs
    feed(responder, padded(caller_io.sent[-1]))
    assert responder.turn == A._TURN_OURS, responder_io.msgs
    assert not responder._awaiting_caller_confirm and not responder._ack_unheard()
    if greeting:
        assert responder._tx_pending is not None and len(responder_io.sent) == 3
        feed(caller, padded(responder_io.sent[-1]))
        assert caller_io.host == [greeting], caller_io.msgs
    else:
        assert responder._holding_idle()
    train_count = sum(np.array_equal(x, ask) for x in responder_io.sent)
    for _ in range(2):
        _Clock.t += 3.64
        responder.idle_keepalive()
    assert sum(np.array_equal(x, ask) for x in responder_io.sent) == train_count


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('chat', [True, False])
@pytest.mark.parametrize('asked', [False, True])
def test_wide_setup_caller_control_grant_needs_a_keyed_train_ask(bw, chat, asked):
    hs, io = answerer(bw, chat=chat)
    if asked:
        _Clock.t = hs._acked_at[1] + 5.7
        hs.idle_keepalive()
        assert len(io.sent) == 2, io.msgs
    grant = M.synth_tone_pairs(F.caller_turn_grant(CALLED, bw))
    feed(hs, padded(grant))
    assert (hs.turn == A._TURN_OURS) is asked, io.msgs
    assert hs._ack_unheard() is (not asked)
    if asked:
        assert not hs._awaiting_caller_confirm and hs._holding_idle()


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
    link-setup on a weak path is one: it must preserve the unconfirmed bound
    across both the damaged setup and the next clean repeat."""
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    hs, io = answerer('2300')
    acked_at, progress = hs._acked_at, hs.progress
    frame = bytearray(F.link_setup_frame(CALLER, bw='2300'))
    frame[-1] ^= 0xFF
    bad = np.concatenate([np.zeros(4800), tx.synth_frame(bytes(frame), over=0), np.zeros(48000)])
    feed(hs, bad)
    hs.on_rx_audio(bad)
    assert any('will not decode' in m for m in io.msgs), io.msgs
    sent = len(io.sent)
    feed(hs, link_setup('2300'))
    assert len(io.sent) == sent, io.msgs
    assert hs._acked_at == acked_at and hs.progress == progress
