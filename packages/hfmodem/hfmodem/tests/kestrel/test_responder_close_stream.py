# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The measured responder close survives every caller turn state, unacknowledged."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock as _Clock, _feed as _feed, _station as _station


@pytest.mark.parametrize('turn', [A._TURN_OURS, A._TURN_PEER])
@pytest.mark.parametrize('calls', [('W9SSJ', 'KD7UHR'), ('W9SSJ-14', 'W9SSJ-15')])
@pytest.mark.parametrize('high_speed', [False, True])
def test_stream_close_cannot_be_consumed_as_an_unowed_data_ack_or_poll(turn, calls, high_speed):
    hs, io = _station('500')
    hs.caller, hs.called = calls
    hs.turn = turn
    hs.high_speed = hs._link_high_speed = high_speed
    hs._released = True
    if turn == A._TURN_OURS:
        hs._poll_at = _Clock.t
    frame = M.synth_burst(hs.called, F.for_bw(F.SESSION_DISCONNECT_REQ_RESPONDER, '500'))
    _feed(hs, np.concatenate([np.zeros(4800), frame, np.zeros(24000)]))
    assert hs.state == A.VaraState.DISCONNECTED, io.msgs
    assert not io.sent, 'a peer close is not answered or interrupted by another carrier'
    assert not io.host


@pytest.mark.parametrize('block', [4096, 8192, 24000])
@pytest.mark.parametrize('head_cut', [0, 2400, 4800, 9600])
def test_close_before_generated_morse_with_callback_batching_and_a_missing_head(block, head_cut):
    from hfmodem.core import cwid
    hs, io = _station('500')
    hs.caller, hs.called = 'W9SSJ-14', 'W9SSJ-15'
    hs.turn = A._TURN_OURS
    hs.high_speed = hs._link_high_speed = True
    hs._poll_at = _Clock.t
    frame = M.synth_burst(hs.called, F.for_bw(F.SESSION_DISCONNECT_REQ_RESPONDER, '500'))
    envelope = np.concatenate([np.full(round(seconds * 5/7 * M.FS), int(on))
                               for on, seconds in cwid.keying('W9SSJ')])
    morse = .5 * envelope * np.sin(2*np.pi*1500*np.arange(len(envelope))/M.FS)
    samples = np.concatenate([frame[head_cut:], np.zeros(2400), morse, np.zeros(24000)])
    for at in range(0, len(samples), block):
        chunk = samples[at:at+block]
        _Clock.t += len(chunk) / M.FS
        hs.on_rx_stream(chunk)
    assert hs.state == A.VaraState.DISCONNECTED, io.msgs
    assert not io.sent, io.msgs


def test_data_continue_without_an_outstanding_over_cannot_refresh_the_idle_contact():
    hs, io = _station('500')
    hs.caller, hs.called = 'W9SSJ-14', 'W9SSJ-15'
    hs.turn = A._TURN_OURS
    hs.high_speed = hs._link_high_speed = True
    hs._poll_at = _Clock.t
    progress = hs.progress
    pairs = F.over_continue_frames(hs.called, '500', (0,))[0][0]
    _feed(hs, np.concatenate([np.zeros(4800), M.synth_tone_pairs(pairs), np.zeros(24000)]))
    assert hs._tx_pending is None
    assert hs.progress == progress, io.msgs
    assert not io.sent and not io.host


@pytest.mark.parametrize('called', ['W9SSJ-15', 'W9SSJ-13'])
@pytest.mark.parametrize('block', [960, 4096, 24000])
def test_measured_eight_symbol_receiver_close_ends_the_contact_without_an_answer(called, block):
    hs, io = _station('500')
    hs.caller, hs.called = 'W9SSJ-14', called
    hs.turn = A._TURN_OURS
    hs.high_speed = hs._link_high_speed = True
    hs._poll_at = _Clock.t
    pairs = F.over_continue_frames(called, '500', (2,))[2][1]
    samples = np.concatenate([np.zeros(4800), M.synth_tone_pairs(pairs), np.zeros(24000)])
    for at in range(0, len(samples), block):
        chunk = samples[at:at+block]
        _Clock.t += len(chunk) / M.FS
        hs.on_rx_stream(chunk)
    assert hs.state == A.VaraState.DISCONNECTED, io.msgs
    assert not io.sent and not io.host


@pytest.mark.parametrize('frame', [0, 1, 3, 4, 5, 6, 7, 8, 10, 12, 13, 15])
def test_other_answer_indices_are_not_receiver_closes(frame):
    hs, io = _station('500')
    hs.caller, hs.called = 'W9SSJ-14', 'W9SSJ-15'
    hs.turn = A._TURN_OURS
    hs._poll_at = _Clock.t
    pairs = F.over_continue_frames(hs.called, '500', (frame,))[frame][1]
    _feed(hs, np.concatenate([M.synth_tone_pairs(pairs), np.zeros(24000)]))
    assert hs.state == A.VaraState.CONNECTED, io.msgs


@pytest.mark.parametrize('damage', ['other-call', 'interior-symbol'])
def test_unknown_or_damaged_receiver_close_keeps_the_contact(damage):
    hs, io = _station('500')
    hs.caller, hs.called = 'W9SSJ-14', 'W9SSJ-15'
    hs.turn = A._TURN_OURS
    pairs = list(F.over_continue_frames('N0ZZZ' if damage == 'other-call' else hs.called, '500', (2,))[2][1])
    if damage == 'interior-symbol':
        pairs[4] = pairs[0]
    _feed(hs, np.concatenate([M.synth_tone_pairs(pairs), np.zeros(24000)]))
    assert hs.state == A.VaraState.CONNECTED, io.msgs


def test_receiver_close_retires_outstanding_originals_without_data_credit():
    from hfmodem.kestrel.host.vara_modem import VaraHostModem
    from .test_compression import IO
    from .test_vara_host_modem import Observer
    modem = VaraHostModem(IO(), bandwidth='500', clock=_Clock.monotonic)
    observer = Observer()
    modem.start(observer)
    hs = modem.station
    hs.caller, hs.called, hs.role = 'W9SSJ-14', 'W9SSJ-15', 'initiator'
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_OURS
    hs._start_compression('OFF')
    modem._connected = True
    original = b'unconfirmed\x00\xff'
    modem.transmit(original)
    modem.poll()
    assert hs._tx_pending is not None
    assert modem.confirmed_total == 0
    pairs = F.over_continue_frames(hs.called, '500', (2,))[2][1]
    _feed(hs, np.concatenate([M.synth_tone_pairs(pairs), np.zeros(24000)]))
    modem.poll()
    assert observer.closes == 1 and not modem.connected
    assert modem.confirmed_total == 0 and modem.discarded_total == len(original)
    assert modem.retired[-1]['unconfirmed'] == original
