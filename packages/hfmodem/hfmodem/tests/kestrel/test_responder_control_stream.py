# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An answering station must hear a complete control without an energy bracket."""
from dataclasses import replace
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock as _Clock, _IO as _IO


def station(called='W9SSJ-15'):
    io = _IO()
    hs = A.VaraStationHandshake([called], io, bw='500')
    hs.role, hs.called, hs.caller = 'responder', called, 'W9SSJ-14'
    hs.state = A.VaraState.CONNECTED
    return hs, io


def feed(hs, samples, block=960):
    for at in range(0, len(samples), block):
        chunk = samples[at:at+block]
        _Clock.t += len(chunk) / M.FS
        hs.on_rx_stream(chunk)


@pytest.mark.parametrize('kind,frame', [(F.SESSION_CONFIRM, 0), (F.SESSION_KEEPALIVE_A, 1),
                                      (F.SESSION_KEEPALIVE_B, 1)])
@pytest.mark.parametrize('block', [960, 4096, 24000])
def test_stock_profile_confirm_and_keepalive_draw_the_measured_role_control(kind, frame, block):
    hs, io = station()
    samples = np.concatenate([np.zeros(4800), M.synth_burst(hs.called, F.for_bw(kind, '500')),
                              np.zeros(24000)])
    feed(hs, samples, block)
    assert len(io.sent) == 1, io.msgs
    pairs = F.control_bursts(hs.caller, '500', frame)[1]
    assert np.array_equal(io.sent[0], M.synth_tone_pairs(pairs))
    assert not io.host and hs.state == A.VaraState.CONNECTED
    # The same bracket is subordinate once the raw stream owns these controls.
    hs.on_rx_audio(samples)
    assert len(io.sent) == 1


@pytest.mark.parametrize('case', ['wrong-call', 'damaged', 'partial', 'noise'])
def test_unrecognized_control_never_keys(case):
    hs, io = station()
    tones = F.handshake_tones('N0ZZZ' if case == 'wrong-call' else hs.called,
                             F.for_bw(F.SESSION_CONFIRM, '500'))
    if case == 'damaged':
        tones[5:8] = tones[9:12]
    samples = M.synth_tones(tones)
    if case == 'partial':
        samples = samples[:-4*M.HOP]
    if case == 'noise':
        samples = np.random.default_rng(0).normal(0, .001, len(samples))
    feed(hs, np.concatenate([samples, np.zeros(24000)]))
    assert not io.sent and not io.host, io.msgs


def test_control_retention_stays_bounded_during_a_silent_connected_peer():
    hs, io = station()
    for _ in range(100):
        feed(hs, np.zeros(4800))
        assert len(hs._ans_buf) < A._SESSION_NEED + A._STREAM_BLOCK
    assert not io.sent


def test_complete_caller_turn_request_gets_called_keyed_grant():
    hs, io = station()
    request = M.synth_burst(hs.caller, F.for_bw(F.SESSION_TURN_REQUEST, '500'))
    feed(hs, np.concatenate([np.zeros(4800), request, np.zeros(24000)]))
    assert len(io.sent) == 1, io.msgs
    assert np.array_equal(io.sent[0], M.synth_tone_pairs(F.turn_grant(hs.called, '500')))
    assert hs.turn == A._TURN_PEER


def test_refused_turn_grant_never_hands_over_the_turn():
    hs, io = station()
    hs.turn = A._TURN_OURS
    io.tx_went_out = lambda: False
    request = M.synth_burst(hs.caller, F.for_bw(F.SESSION_TURN_REQUEST, '500'))
    feed(hs, np.concatenate([np.zeros(4800), request, np.zeros(24000)]))
    assert len(io.sent) == 1
    assert hs.turn == A._TURN_OURS


@pytest.mark.parametrize('kind', [F.SESSION_CONFIRM, F.SESSION_KEEPALIVE_A, F.SESSION_KEEPALIVE_B])
@pytest.mark.parametrize('cut', [2400, 4800])
def test_transport_echo_guard_can_cut_the_head_but_not_the_complete_control_tail(kind, cut):
    hs, io = station()
    samples = M.synth_burst(hs.called, F.for_bw(kind, '500'))
    feed(hs, np.concatenate([samples[cut:], np.zeros(24000)]))
    assert len(io.sent) == 1, io.msgs


@pytest.mark.parametrize('short,start', list(A._CONFIRM_START.items()) +
                         [(replace(F.SESSION_CONFIRM, preadv=181), 7)])
def test_each_selected_start_has_a_complete_long_followup(short, start):
    hs, io = station()
    hs._link_high_speed = True
    for kind in (short, replace(F.SESSION_KEEPALIVE_A,
                                preadv=1+62*((short.preadv-1)//30))):
        feed(hs, np.concatenate([np.zeros(4800), M.synth_burst(hs.called, F.for_bw(kind, '500')),
                                 np.zeros(24000)]))
    assert len(io.sent) == 2, io.msgs
    assert hs._rec500 == A._phy.level_key('500', start)
    assert np.array_equal(io.sent[1], M.synth_tone_pairs(F.control_bursts(hs.caller, '500', 1)[1]))


def test_complete_peer_close_ends_the_responder_without_reply_or_delivery_credit():
    hs, io = station()
    progress = hs.progress
    close = M.synth_burst(hs.called, F.for_bw(F.SESSION_DISCONNECT_REQ, '500'))
    feed(hs, np.concatenate([np.zeros(4800), close, np.zeros(24000)]))
    assert hs.state == A.VaraState.DISCONNECTED
    assert not io.sent and not io.host and hs.progress == progress


@pytest.mark.parametrize('form', ['single-tone', 'control-grant'])
def test_host_reply_waits_for_the_owed_caller_release_even_when_the_energy_gate_is_quiet(form):
    hs, io = station()
    hs._link_high_speed = True
    hs._start_compression('OFF')
    hs.turn = A._TURN_PEER
    hs._answer_owed = A._OWED_RELEASE
    original = b'answerer binary\x00\xff'
    hs.send(original)
    assert not io.sent and hs._tx_pending is None
    assert hs.payload_ledger.pending == len(original)
    release = (M.synth_burst(hs.called, F.for_bw(F.SESSION_TURN_RELEASE, '500'))
               if form == 'single-tone' else M.synth_tone_pairs(F.caller_turn_grant(hs.called, '500')))
    feed(hs, np.concatenate([np.zeros(4800), release, np.zeros(24000)]))
    assert hs.turn == A._TURN_OURS and hs._tx_pending is not None
    assert len(io.sent) == 1 and hs.payload_ledger.confirmed == 0


@pytest.mark.parametrize('case', ['wrong-role', 'wrong-called', 'damaged', 'partial'])
def test_caller_handover_requires_its_complete_role_and_called_identity(case):
    hs, io = station()
    hs._answer_owed = A._OWED_RELEASE
    pairs = list(F.turn_grant(hs.called, '500') if case == 'wrong-role' else
                 F.caller_turn_grant('N0ZZZ' if case == 'wrong-called' else hs.called, '500'))
    if case == 'damaged': pairs[7] = pairs[0]
    samples = M.synth_tone_pairs(pairs)
    if case == 'partial': samples = samples[:-3*M.HOP]
    feed(hs, np.concatenate([np.zeros(4800), samples, np.zeros(24000)]))
    assert hs.turn == A._TURN_PEER and hs._answer_owed == A._OWED_RELEASE
    assert not io.sent and not io.host


@pytest.mark.parametrize('role', [0, 1])
def test_only_the_peer_role_ack_settles_an_outstanding_final_original(role):
    hs, io = station()
    hs.turn = A._TURN_OURS
    hs._link_high_speed = True
    hs._start_compression('OFF')
    original = b'answerer binary\x00\xff'
    hs.send(original)
    assert hs._tx_pending is not None and hs.payload_ledger.confirmed == 0
    sent = len(io.sent)
    ack = M.synth_tone_pairs(F.control_bursts(hs.caller, '500')[role])
    feed(hs, np.concatenate([np.zeros(4800), ack, np.zeros(24000)]))
    if role == 0:
        assert hs.payload_ledger.confirmed == len(original)
        assert hs._tx_pending is None and hs.turn == A._TURN_PEER
        release = M.synth_burst(hs.called, F.for_bw(F.SESSION_TURN_RELEASE_RESPONDER, '500'))
        assert len(io.sent) == sent+1 and np.array_equal(io.sent[-1], release)
    else:
        assert hs.payload_ledger.confirmed == 0 and hs._tx_pending is not None
        assert len(io.sent) == sent


def sending_long_reply():
    hs, io = station()
    hs.turn = A._TURN_OURS
    hs._link_high_speed = True
    hs._lead = A.VB6Rnd(F.lead_state(hs.caller, True, '500'))
    hs._rec500 = A._phy.level_key('500', 7)
    hs._start_compression('OFF')
    hs.send(bytes(range(256))*2)
    assert hs._tx_pending is not None
    assert not A._phy.over_is_last(hs._tx_pending[0], hs.caller)
    return hs, io


def test_long_answering_write_settles_each_original_span_and_closes_its_final_over():
    hs, io = sending_long_reply()
    for _ in range(10):
        before = hs.payload_ledger.confirmed
        pending = hs._tx_pending
        assert pending is not None
        last = A._phy.over_is_last(pending[0], hs.caller)
        pairs = (F.control_bursts(hs.caller, '500')[0] if last else
                 F.over_continue_frames(hs.called, '500', (0,))[0][0])
        feed(hs, np.concatenate([np.zeros(4800), M.synth_tone_pairs(pairs), np.zeros(24000)]))
        assert before < hs.payload_ledger.confirmed <= 512, io.msgs
        if last: break
    assert hs.payload_ledger.confirmed == 512 and hs.payload_ledger.pending == 0
    assert hs._tx_pending is None and hs.turn == A._TURN_PEER and not io.host


@pytest.mark.parametrize('case', ['wrong-role', 'wrong-called', 'damaged', 'partial', 1, 2, 3, 9, 10, 12, 13, 15])
def test_unqualified_or_damaged_continue_cannot_settle_an_intermediate_answering_over(case):
    hs, io = sending_long_reply()
    index = case if type(case) is int else 0
    call = 'N0ZZZ' if case == 'wrong-called' else hs.called
    pairs = list(F.over_continue_frames(call, '500', (index,))[index][case == 'wrong-role'])
    if case == 'damaged': pairs[4] = pairs[0]
    samples = M.synth_tone_pairs(pairs)
    if case == 'partial': samples = samples[:-3*M.HOP]
    sent, pending, progress = len(io.sent), hs._tx_pending, hs.progress
    feed(hs, np.concatenate([np.zeros(4800), samples, np.zeros(24000)]))
    assert hs.payload_ledger.confirmed == 0 and hs._tx_pending == pending
    assert len(io.sent) == sent and hs.progress == progress
