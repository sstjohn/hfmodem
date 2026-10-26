# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Wide callers confirm the initial answering turn before the greeting keys."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock, _IO
from .test_responder_control_stream import feed


def acquired(bw, *, immediate=b'', chat=False):
    io = _IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, bw=bw, chat=chat)
    hs.role, hs.called, hs.caller = 'responder', 'W9SSJ', 'W1AW'
    hs.state = A.VaraState.CONNECTING
    hs._chat_profile = chat
    if immediate:
        io.connected = lambda *args: hs.send(immediate)
    frame = F.link_setup_frame(hs.caller, bw=bw, high_speed=False, compression='OFF')
    assert hs._take_link_setup(frame)
    return hs, io


def confirm(hs, *, called=None, truncated=False):
    # Stock's level-4 short confirmation: also Xara c54/F1, seed 1551,
    # code index 3 (91 generator draws), keyed by the called station.
    samples = M.synth_burst(called or hs.called, F.for_bw(F.SESSION_CONFIRM, hs.bw))
    if truncated:
        samples = samples[:-4*M.HOP]
    feed(hs, np.concatenate([np.zeros(4800), samples, np.zeros(24000)]))


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_wide_greeting_waits_for_confirmation_and_keys_data_in_the_granted_turn(bw):
    original = bytes(range(256))
    hs, io = acquired(bw, immediate=original)
    assert len(io.sent) == 1 and hs._tx_pending is None
    assert hs._awaiting_caller_confirm
    confirm(hs)
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None, io.msgs
    assert len(io.sent) == 2 and not hs._awaiting_caller_confirm
    assert hs.payload_ledger.accepted == len(original) and hs.payload_ledger.confirmed == 0
    assert not hs._initial_answer_turn  # The special frame-13 greeting release is BW500-only.
    assert not any('tx session-turn-request' in m for m in io.msgs)


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('case', ['wrong-call', 'truncated'])
def test_wide_confirmation_requires_the_called_identity_and_complete_tail(bw, case):
    hs, io = acquired(bw, immediate=b'greeting')
    confirm(hs, called='N0ZZZ' if case == 'wrong-call' else None,
            truncated=case == 'truncated')
    assert len(io.sent) == 1 and hs._tx_pending is None
    assert hs._awaiting_caller_confirm and hs.payload_ledger.confirmed == 0


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_wide_greeting_queued_after_first_confirmation_waits_for_repeated_confirmation(bw):
    hs, io = acquired(bw)
    confirm(hs)
    assert hs._tx_pending is None and len(io.sent) == 2
    hs.send(b'delayed greeting')
    assert hs._tx_pending is None and len(io.sent) == 2
    confirm(hs)
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None
    assert len(io.sent) == 3 and hs.payload_ledger.confirmed == 0


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_wide_chat_setup_keeps_its_existing_no_confirmation_profile(bw):
    hs, io = acquired(bw, immediate=b'chat greeting', chat=True)
    assert hs._chat_profile and not hs._awaiting_caller_confirm
    assert hs._tx_pending is None and len(io.sent) == 1


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('queue_before_response', [True, False])
def test_empty_initial_turn_waits_for_idle_response_before_sending_new_host_data(bw, queue_before_response):
    hs, io = acquired(bw)
    confirm(hs)
    long = M.synth_burst(hs.called, F.for_bw(replace(F.SESSION_KEEPALIVE_A, preadv=187), bw))
    feed(hs, np.concatenate([np.zeros(4800), long, np.zeros(24000)]))
    assert hs.turn == A._TURN_OURS and hs._holding_idle()
    assert not hs._awaiting_caller_confirm and hs._tx_pending is None
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(hs.caller, bw, 1)[1]))
    sent = len(io.sent)
    original = b'posted around the initial idle response'
    if queue_before_response:
        hs.send(original)
        assert len(io.sent) == sent and hs._tx_pending is None
        assert hs.turn == A._TURN_ASKED
    # NO_TRAFFIC is the caller's reply to the outstanding idle poll. Only
    # after hearing it can we ask for a fresh sending turn with frame 0.
    response = M.synth_burst(hs.called, F.for_bw(F.SESSION_KEEPALIVE_B, bw))
    feed(hs, np.concatenate([np.zeros(4800), response, np.zeros(24000)]))
    if not queue_before_response:
        assert len(io.sent) == sent and hs._tx_pending is None
        hs.send(original)
    assert len(io.sent) == sent+1 and hs._tx_pending is None
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(hs.caller, bw, 0)[1]))
    grant = M.synth_tone_pairs(F.caller_turn_grant(hs.called, bw))
    feed(hs, np.concatenate([np.zeros(4800), grant, np.zeros(24000)]))
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None
    assert len(io.sent) == sent+2 and hs.payload_ledger.confirmed == 0


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_empty_initial_wide_turn_keeps_polling_on_the_responder_cadence(bw):
    hs, io = acquired(bw)
    hs._answer_responder_control(F.SESSION_KEEPALIVE_A)
    assert hs._holding_idle() and hs.idle_poll_due_in() is not None
    sent = len(io.sent)
    _Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert len(io.sent) == sent+1
    assert np.array_equal(io.sent[-1], M.synth_tone_pairs(F.control_bursts(hs.caller, bw, 1)[1]))
