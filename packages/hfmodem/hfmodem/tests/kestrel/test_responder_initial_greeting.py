# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CONNECTED-time greeting waits for the caller's actual confirmation."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock
from .test_responder_control_stream import station, feed


def acquired(monkeypatch, *, immediate=b''):
    hs, io = station()
    hs.high_speed = True
    hs.state = A.VaraState.CONNECTING
    frame = F.link_setup_frame(hs.caller, bw='500', high_speed=True, compression='OFF')
    monkeypatch.setattr(A.OF, 'link_setup_read', lambda *a, **kw: frame)
    if immediate:
        io.connected = lambda *args: hs.send(immediate)
    assert hs._rx_link_setup([])
    return hs, io


def confirm(hs, *, long=False):
    kind = (replace(F.SESSION_KEEPALIVE_A, preadv=373) if long else
            replace(F.SESSION_CONFIRM, preadv=181))
    feed(hs, np.concatenate([np.zeros(4800), M.synth_burst(hs.called, F.for_bw(kind, '500')),
                             np.zeros(24000)]))


def test_immediate_connected_callback_cannot_key_a_turn_request_across_the_confirm(monkeypatch):
    original = bytes(range(256))*2
    hs, io = acquired(monkeypatch, immediate=original)
    assert len(io.sent) == 1, io.msgs  # Only the physically completed connected ACK.
    assert hs._tx_pending is None and hs.payload_ledger.pending == len(original)
    confirm(hs)
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None
    assert len(io.sent) == 2 and hs.payload_ledger.confirmed == 0
    assert hs._rec500 == A._phy.level_key('500', 7)


def test_greeting_posted_between_short_and_long_confirmation_still_waits(monkeypatch):
    hs, io = acquired(monkeypatch)
    confirm(hs)
    assert len(io.sent) == 2 and hs._tx_pending is None
    hs.send(bytes(range(256))*2)
    assert len(io.sent) == 2 and hs._tx_pending is None, io.msgs
    confirm(hs, long=True)
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None
    assert len(io.sent) == 3 and hs.payload_ledger.confirmed == 0


def test_refused_initial_data_keeps_its_originals_and_confirmation_wait(monkeypatch):
    original = bytes(range(256))*2
    hs, io = acquired(monkeypatch, immediate=original)
    io.tx_went_out = lambda: False
    confirm(hs)
    assert hs._awaiting_caller_confirm and hs._tx_pending is None
    assert hs.payload_ledger.pending == len(original) and hs.payload_ledger.confirmed == 0
    io.tx_went_out = lambda: True
    confirm(hs)
    assert not hs._awaiting_caller_confirm and hs._tx_pending is not None
    assert hs.payload_ledger.accepted == len(original) and hs.payload_ledger.confirmed == 0
    assert np.array_equal(io.sent[-1], io.sent[-2])


def final_ack(hs, *, pairs=None):
    if pairs is None:
        pairs = F.control_bursts(hs.caller, '500')[0]
    feed(hs, np.concatenate([np.zeros(4800), M.synth_tone_pairs(pairs), np.zeros(24000)]))


def greeting_final(monkeypatch):
    original = bytes(range(256))*2
    hs, io = acquired(monkeypatch, immediate=original)
    confirm(hs)
    assert hs._initial_answer_turn
    for _ in range(10):
        assert hs._tx_pending is not None
        if A._phy.over_is_last(hs._tx_pending[0], hs.caller):
            return hs, io, original
        pairs = F.over_continue_frames(hs.called, '500', (0,))[0][0]
        final_ack(hs, pairs=pairs)
    raise AssertionError('greeting never reached its final over')


def test_initial_greeting_hands_over_with_the_stock_called_keyed_frame_and_regrants(monkeypatch):
    hs, io, original = greeting_final(monkeypatch)
    final_ack(hs)
    grant = M.synth_tone_pairs(F.turn_grant(hs.called, '500'))
    assert hs.payload_ledger.confirmed == len(original) and hs.payload_ledger.pending == 0
    assert hs._tx_pending is None and hs.turn == A._TURN_PEER and hs._released
    assert np.array_equal(io.sent[-1], grant)
    sent, progress = len(io.sent), hs.progress
    for _ in range(2):
        final_ack(hs)             # Caller missed the preceding handover.
    assert len(io.sent) == sent+2
    assert all(np.array_equal(audio, grant) for audio in io.sent[-2:])
    assert hs.payload_ledger.confirmed == len(original) and hs.progress == progress
    assert not io.host


def test_declined_initial_handover_can_retry_on_the_same_ack_without_new_credit(monkeypatch):
    hs, io, original = greeting_final(monkeypatch)
    io.tx_went_out = lambda: False
    final_ack(hs)
    assert hs.payload_ledger.confirmed == len(original)
    assert hs.turn == A._TURN_OURS and hs._release_owed
    progress = hs.progress
    io.tx_went_out = lambda: True
    final_ack(hs)
    assert hs.turn == A._TURN_PEER and not hs._release_owed
    assert hs.payload_ledger.confirmed == len(original) and hs.progress == progress
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.turn_grant(hs.called, '500')))


@pytest.mark.parametrize('case', ['wrong-role', 'wrong-caller', 'wrong-frame', 'damaged', 'partial'])
def test_initial_handover_retry_requires_the_complete_final_caller_ack(monkeypatch, case):
    hs, io, original = greeting_final(monkeypatch)
    final_ack(hs)
    pairs = list(F.control_bursts('N0ZZZ' if case == 'wrong-caller' else hs.caller,
                                '500', 2 if case == 'wrong-frame' else 0)[case == 'wrong-role'])
    if case == 'damaged':
        pairs[6] = pairs[0]
    samples = M.synth_tone_pairs(pairs)
    if case == 'partial':
        samples = samples[:-3*M.HOP]
    sent, progress = len(io.sent), hs.progress
    feed(hs, np.concatenate([np.zeros(4800), samples, np.zeros(24000)]))
    assert len(io.sent) == sent and hs.progress == progress
    assert hs.payload_ledger.confirmed == len(original)


def test_caller_release_ends_the_initial_greeting_handover_profile(monkeypatch):
    hs, io, _ = greeting_final(monkeypatch)
    final_ack(hs)
    hs._take_caller_release()
    assert not hs._initial_answer_turn
    assert hs._release_turn()
    # Stock's ordinary release is the 11-symbol grant behind a level-6+ over.
    ordinary = (M.synth_tone_pairs(F.turn_grant(hs.called, '500')) if hs._last_tx_level >= 6
                else M.synth_burst(hs.called, F.for_bw(F.SESSION_TURN_RELEASE_RESPONDER, '500')))
    assert np.array_equal(io.sent[-1], ordinary)
    assert not hs._initial_answer_turn and hs._released_form != 'f20'
    sent = len(io.sent)
    final_ack(hs)
    assert len(io.sent) == sent


def test_initial_turn_answers_the_callers_idle_poll_then_its_data_request(monkeypatch):
    hs, io, original = greeting_final(monkeypatch)
    final_ack(hs)
    sent, progress, keyed = len(io.sent), hs.progress, hs.idle_keyed
    final_ack(hs, pairs=F.control_bursts(hs.caller, '500', 1)[0])
    assert len(io.sent) == sent+1 and hs.turn == A._TURN_PEER
    expected = M.synth_tone_pairs(F.responder_idle_answer(hs.called, '500'))
    assert np.array_equal(io.sent[-1], expected)
    assert hs.idle_answers == 1 and hs.idle_keyed == keyed and hs.progress == progress
    final_ack(hs)           # The queued caller now asks with control frame 0.
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.turn_grant(hs.called, '500')))
    assert hs.payload_ledger.confirmed == len(original) and hs.progress == progress
    assert hs._tx_pending is None and not io.host


def test_declined_idle_poll_answer_retries_without_advancing_a_clock_or_delivery(monkeypatch):
    hs, io, original = greeting_final(monkeypatch)
    final_ack(hs)
    progress, keyed = hs.progress, hs.idle_keyed
    io.tx_went_out = lambda: False
    final_ack(hs, pairs=F.control_bursts(hs.caller, '500', 1)[0])
    assert hs.idle_answers == 0 and hs.turn == A._TURN_PEER
    io.tx_went_out = lambda: True
    final_ack(hs, pairs=F.control_bursts(hs.caller, '500', 1)[0])
    assert hs.idle_answers == 1
    assert np.array_equal(io.sent[-1], io.sent[-2])
    assert hs.payload_ledger.confirmed == len(original)
    assert hs.progress == progress and hs.idle_keyed == keyed
