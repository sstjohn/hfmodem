# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bench 3 gates 3 and 4, 2026-10-20: CLEANTXBUFFER, the answerer's final ACK
behind its own over, and the caller's close after a release."""

import numpy as np
import pytest

from hfmodem.kestrel.host.server import HostSession
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_varac_answer_1020 import (GATE as GATE, Clock as Clock, IO as IO, Transport as Transport, Wire as Wire, audio as audio, clock as clock, feed as feed,
                                      recorded_plain_session)

LONG = (b''.join(b'line %04d the quick brown fox\r' % i for i in range(96)))[:2880]
STOCK, KESTREL = 'WI9YZA', 'W9SSJ'


def connected_caller_host():
    cmd, data, transport = Wire(), Wire(), Transport()
    modem = VaraHostModem(transport, bandwidth='500')
    session = HostSession(cmd, data, modem)
    modem.start(session)
    modem.poll()
    hs = modem.station
    hs.role, hs.caller, hs.called = 'initiator', KESTREL, STOCK
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_PEER
    hs._start_compression('TEXT')
    modem._connected = True
    cmd.data.clear()
    return session, modem, cmd, transport


# -- gate 3, item 1 and 4: a clear while our turn request is on the air -----

def test_clear_after_the_turn_request_keyed_fails_and_keeps_every_byte():
    """19.3 s: 2880 bytes queued, the request keyed, CLEANTXBUFFER at 20.3 s."""
    session, modem, cmd, transport = connected_caller_host()
    modem.transmit(LONG)
    modem.poll()
    assert np.array_equal(transport.sent[-1], M.synth_burst(
        KESTREL, F.for_bw(F.SESSION_TURN_REQUEST, '500')))
    cmd.data.clear()
    session._dispatch('CLEANTXBUFFER')
    modem.poll()
    assert cmd.data.startswith(b'CLEANTXBUFFER FAILED \r'), cmd.data
    assert b'BUFFER 0' not in cmd.data
    assert modem.station.payload_ledger.pending == len(LONG)


def test_clear_during_our_burst_is_answered_before_the_burst_ends():
    session, modem, cmd, transport = connected_caller_host()
    during = []

    def tx(samples):
        session._dispatch('CLEANTXBUFFER')
        during.append(bytes(cmd.data))
        transport.sent.append(samples)

    transport.tx = tx
    modem.transmit(b'queued at kestrel while stock still sends\r')
    modem.poll()
    assert during and b'CLEANTXBUFFER FAILED \r' in during[0], during


# -- gate 4 (a): the answerer's final ACK behind its own over ---------------

def answerer_with_final_over():
    """Gate 1's recorded stock session up to our reply's level-6 over, as in
    gate 4, where stock's release family is its 11-symbol form."""
    hs, io = recorded_plain_session()
    hs.send(b'ack from ours\n')
    x, sent = audio(GATE, 'a2-release'), len(io.sent)
    for at in range(0, len(x), 1024):
        Clock.t += 1024 / M.FS
        hs.on_rx_stream(x[at:at + 1024])
        if len(io.sent) > sent:
            break   # the transport skips what arrives under our keying
    assert hs._tx_pending is not None and A._phy.over_is_last(hs._tx_pending[0], STOCK)
    return hs, io


@pytest.mark.parametrize('cut', [0, 1, 2, 3])
def test_stocks_final_ack_with_its_head_under_the_echo_guard_settles_the_over(cut):
    """Session 2: stock acked 0.085 s behind our over; the guard ate its head."""
    hs, io = answerer_with_final_over()
    ack = M.synth_tone_pairs(F.control_bursts(STOCK, '500', 0)[0])
    feed(hs, ack[cut * M.HOP:])
    assert hs._tx_pending is None and hs.turn == A._TURN_PEER, io.logs
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.turn_grant(KESTREL, '500')))


def test_stocks_repeated_polls_behind_our_over_draw_the_release():
    """Session 2: then 25 polls every 2.3 s, none answered."""
    hs, io = answerer_with_final_over()
    poll = M.synth_tone_pairs(F.control_bursts(STOCK, '500', 1)[0])
    feed(hs, np.r_[np.zeros(int(1.83 * M.FS)), poll])
    assert hs._tx_pending is None and hs.turn == A._TURN_PEER, io.logs
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(
        F.responder_post_delivery_idle(KESTREL, '500')))


# -- gate 4 (b): the caller's host disconnects right after its release ------

def released_caller():
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw='500', high_speed=True)
    hs.role, hs.caller, hs.called = 'initiator', KESTREL, STOCK
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_OURS
    assert hs._release_turn()
    return hs, io


def test_close_after_release_asks_for_the_turn_then_closes_on_the_grant():
    """Stock pair, 45.92 s: release, DISCONNECT, control frame 0, the answerer's
    grant 0.57 s later, then the close."""
    hs, io = released_caller()
    hs.disconnect(acquire=True)
    assert hs.state == A.VaraState.CONNECTED
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(KESTREL, '500', 0)[0]))
    sent = len(io.sent)
    hs.disconnect(acquire=True)
    assert len(io.sent) == sent
    feed(hs, np.r_[np.zeros(int(.07 * M.FS)), M.synth_tone_pairs(F.turn_grant(STOCK, '500'))])
    assert hs.state == A.VaraState.DISCONNECTED, io.logs
    assert np.array_equal(io.sent[-1], M.synth_burst(STOCK, F.for_bw(F.SESSION_DISCONNECT_REQ, '500')))


def test_unanswered_close_asks_are_bounded():
    hs, io = released_caller()
    for _ in range(A._TURN_MAX_ASK + 1):
        hs.disconnect(acquire=True)
        Clock.t += A._CLOSE_ASK_REPEAT_S + .01
    assert hs.state == A.VaraState.DISCONNECTED
    assert np.array_equal(io.sent[-1], M.synth_burst(STOCK, F.for_bw(F.SESSION_DISCONNECT_REQ, '500')))


def driven(hs, x):
    """Feed ``x`` with the host driver offering the close on every block, as
    VaraHostModem.poll does while closing; returns how many bursts we keyed."""
    sent = len(hs.io.sent)
    x = np.r_[x, np.zeros(M.FS // 4)]
    for at in range(0, len(x), 1024):
        Clock.t += 1024 / M.FS
        hs.on_rx_stream(x[at:at + 1024])
        hs.disconnect(acquire=True)
    hs.on_rx_audio(x)
    return len(hs.io.sent) - sent


def test_a_close_ask_lost_under_stocks_poll_goes_into_its_repeat():
    """rx2b gate 2 (bench 3, 2026-10-21), 1105.29: DISCONNECT as stock keyed
    its idle poll; our ask went out 0.04 s behind it and neither heard the
    other. Stock polled again every 2.36 s and our ask repeated every 2.35 s,
    0.03 and 0.01 s behind each poll, until the third ask spent the budget
    and the close keyed behind its own ask: +6.7 s. The holder listens behind
    its poll, so the ask goes there and its grant carries the close."""
    hs, io = released_caller()
    Clock.t += 10.
    hs.disconnect(acquire=True)
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(KESTREL, '500', 0)[0]))
    poll = M.synth_tone_pairs(F.control_bursts(KESTREL, '500', 1)[1])
    assert driven(hs, np.r_[np.zeros(int(2.314 * M.FS)), poll]) == 1, io.logs
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(KESTREL, '500', 0)[0]))
    assert driven(hs, np.r_[np.zeros(int(.07 * M.FS)),
                            M.synth_tone_pairs(F.turn_grant(STOCK, '500'))]) == 1, io.logs
    assert hs.state == A.VaraState.DISCONNECTED, io.logs
    assert np.array_equal(io.sent[-1], M.synth_burst(STOCK, F.for_bw(F.SESSION_DISCONNECT_REQ, '500')))


# -- gate 3, item 2: stock's overs above level 11 -----------------------------

def high_speed_caller(expect):
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw='500', high_speed=True, compression='TEXT')
    hs.role, hs.caller, hs.called = 'initiator', KESTREL, STOCK
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_PEER
    hs._link_high_speed = True
    hs._lead = A.VB6Rnd(F.lead_state(KESTREL, True, '500'))
    hs._start_compression('TEXT')
    hs._expect_level, hs._peer_over = expect, 3
    return hs, io


def stream(hs, x, t0, start=0., stop=None, trail=0.):
    """Recorded audio from ``start`` s into the cut, the clock at stream time
    plus ``trail``; returns when we keyed, on the recording's clock."""
    io, keyed = hs.io, []
    end = len(x) if stop is None else int(stop * M.FS)
    for at in range(int(start * M.FS), end, 1024):
        Clock.t = t0 + (at + 1024) / M.FS + trail
        sent = len(io.sent)
        hs.on_rx_stream(x[at:at + 1024])
        if len(io.sent) > sent:
            keyed.append(Clock.t - trail)
    return keyed


def test_a_single_level_12_frame_is_answered_in_its_turnaround():
    """474.935-478.755: one L12 frame; stock polls 1.285 s after it, and our
    answer went out at 479.985, on top of that poll."""
    hs, io = high_speed_caller(12)
    Clock.t = 474.84
    hs._key(False)
    hs._lv_armed = hs._lv_owed = hs._lv_prompt = True
    keyed = stream(hs, audio(GATE, 'g3-level12'), 474.84)
    assert keyed and keyed[0] - 478.755 < .25, (keyed, io.logs)
    assert len(io.host[0]) == 867
    assert not any('will not decode' in line for line in io.logs), io.logs


def owed_over_caller():
    hs, io = high_speed_caller(13)
    Clock.t = 484.4
    hs._key(False)
    hs._answer_owed, hs._lv_owed, hs._rx_failed = A._OWED_OVER, True, True
    return hs, io


def test_no_nak_into_a_poll_the_stream_names_late_in_wall_time():
    """484.495: stock's poll behind the unread over. Named fresh on the stream
    but a second behind the clock, the 1.37 s NAK would land on its next poll."""
    hs, io = owed_over_caller()
    assert stream(hs, audio(GATE, 'g3-resend'), 484.4, stop=2.2) == [pytest.approx(485.85, abs=.05)]
    hs, io = owed_over_caller()
    assert stream(hs, audio(GATE, 'g3-resend'), 484.4, stop=2.2, trail=1.0) == [], io.logs


def test_a_nak_the_peer_polled_over_does_not_step_its_resend():
    """486.92: our NAK, lost under stock's poll; we name that poll's tail behind
    our unkey and NAK again at 489.09, which stock hears and answers with its
    L9 resend at 490.62. Stock steps once per NAK it hears."""
    hs, io = owed_over_caller()
    Clock.t = 486.92
    assert hs._nak_owed_over() and hs._expect_level == 9
    Clock.t = 488.29
    hs._key(False)
    keyed = stream(hs, audio(GATE, 'g3-resend'), 484.4, start=488.29 - 484.4)
    assert keyed and keyed[0] < 489.3, (keyed, io.logs)
    assert [len(h) for h in io.host] == [458], io.logs


def test_clear_just_before_our_next_over_keys_is_answered_at_once():
    """r4 caller mid-send: CLEANTXBUFFER 13 ms before our over keyed, answered
    3.7 s later at its end; stock answers at once."""
    session, modem, cmd, transport = connected_caller_host()
    hs = modem.station
    hs.turn = A._TURN_OURS
    hs.send(LONG)
    assert hs._tx_pending is not None
    cmd.data.clear()
    session._dispatch('CLEANTXBUFFER')
    assert cmd.data == b'CLEANTXBUFFER FAILED \r', cmd.data


# -- af39 gate 2: a close asked as the holder's poll falls due ---------------

def polled_caller():
    """The caller in the holder's turn, its last poll answered; the clock
    stands at that poll's end."""
    hs, io = released_caller()
    poll = M.synth_tone_pairs(F.control_bursts(KESTREL, '500', 1)[1])
    feed(hs, np.r_[np.zeros(int(.5 * M.FS)), poll])
    assert np.array_equal(io.sent[-1], M.synth_burst(STOCK, F.for_bw(F.SESSION_KEEPALIVE_B, '500'))), io.logs
    Clock.t -= .25
    return hs, io, poll


def ask_keyed(io):
    return np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(KESTREL, '500', 0)[0]))


def test_a_close_asked_as_the_holders_poll_falls_due_goes_into_the_poll():
    """af39 gate 2 (bench 3, 2026-10-22), 1105.46: DISCONNECT 11.6 s behind
    stock's last poll; our frame 0 keyed 0.02 s behind its next poll (1105.49)
    and 0.44 s under it, stock polled again at 1107.91, the ask went into that
    turnaround and the close took +5.6 s against 33fd0b9c9's +4.8. The holder
    polls every 12.06 s and listens behind each: an ask due with the poll
    waits for it."""
    hs, io, poll = polled_caller()
    Clock.t += 11.6
    sent = len(io.sent)
    hs.disconnect(acquire=True)
    assert len(io.sent) == sent and any('poll is due' in m for m in io.logs), io.logs[-3:]
    assert driven(hs, np.r_[np.zeros(int(.03 * M.FS)), poll]) == 1 and ask_keyed(io), io.logs[-4:]
    assert driven(hs, np.r_[np.zeros(int(.07 * M.FS)),
                            M.synth_tone_pairs(F.turn_grant(STOCK, '500'))]) == 1, io.logs[-4:]
    assert hs.state == A.VaraState.DISCONNECTED, io.logs[-4:]


def test_a_close_asked_mid_cycle_asks_at_once():
    hs, io, _ = polled_caller()
    Clock.t += 5.
    hs.disconnect(acquire=True)
    assert ask_keyed(io), io.logs[-3:]


def test_a_close_held_for_a_poll_that_never_comes_asks_once_the_poll_is_late():
    hs, io, _ = polled_caller()
    Clock.t += 11.6
    hs.disconnect(acquire=True)
    assert driven(hs, np.zeros(int(1.5 * M.FS))) == 0, io.logs[-3:]
    assert driven(hs, np.zeros(int(2. * M.FS))) == 1 and ask_keyed(io), io.logs[-3:]


def test_a_poll_before_the_turn_last_changed_hands_does_not_hold_the_close():
    hs, io, _ = polled_caller()
    Clock.t += 2.
    hs.turn = A._TURN_OURS
    assert hs._release_turn()
    Clock.t += 9.6
    hs.disconnect(acquire=True)
    assert ask_keyed(io), io.logs[-3:]


def test_a_poll_before_the_holders_last_over_does_not_hold_the_close():
    hs, io = high_speed_caller(12)
    hs._key(False)
    hs._lv_armed = hs._lv_owed = hs._lv_prompt = True
    hs._peer_poll_at = 472.
    assert stream(hs, audio(GATE, 'g3-level12'), 474.84), io.logs
    Clock.t = 472. + 11.6
    hs.disconnect(acquire=True)
    assert ask_keyed(io), io.logs[-3:]
