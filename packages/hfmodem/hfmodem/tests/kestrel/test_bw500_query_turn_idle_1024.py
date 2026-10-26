# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock bench 3 corrected the KE8PLM postmortem hypothesis: queried final
1078/63 acknowledges the over, then draws 60/1241, not a DATA retry.
"""
import numpy as np
import pytest

from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_bw500_query_recovery import _IO, clock as clock  # noqa: F401


def _frame(hs, kind, call=None):
    wave = MK.synth_burst(call or hs.caller, VF.for_bw(kind, "500"))
    return np.r_[np.zeros(int(.1 * MK.FS)), wave, np.zeros(int(.3 * MK.FS))]


def _pending(clock, payload=b'8 73<DISC>', compression='OFF'):
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True,
                                 compression=compression)
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'KE8PLM'
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    hs._link_high_speed, hs._rec500 = True, 4
    hs._start_compression(compression)
    hs._lead = TX.VB6Rnd(VF.lead_state('W9SSJ', False, '500'))
    hs.send(payload)
    assert hs._tx_pending[2] == 4
    return hs, io


def _queried(clock, *, payload=b'8 73<DISC>', compression='OFF', phase=0, repeat=False):
    hs, io = _pending(clock, payload, compression)
    hs._delivery_keyed = phase + 1
    clock.now += 1.3
    assert hs.query_intermediate_answer()
    if repeat:
        clock.now += 2
        assert hs.query_intermediate_answer()
    return hs, io


def _stream(hs, clock, wave):
    for at in range(0, len(wave), 960):
        block = wave[at:at + 960]
        clock.now += len(block) / MK.FS
        hs.on_rx_stream(block)


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('phase,repeat', [(0,False),(1,False),(0,True),(1,True)])
def test_final_idle_settles_once_and_answers_with_keepalive(clock, stream, phase, repeat):
    hs, io = _queried(clock, phase=phase, repeat=repeat)
    assert np.array_equal(io.sent[-1], MK.synth_burst(hs.called, VF.for_bw(
        VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[phase], '500')))
    idle = _frame(hs, VF.SESSION_TURN_IDLE_RESPONDER)
    count = len(io.sent)
    (_stream(hs, clock, idle) if stream else hs.on_rx_audio(idle))
    assert hs.turn == VA._TURN_PEER and hs._tx_pending is None, io.msgs
    assert hs.payload_ledger.pending == 0 and hs._tx_retries == 0
    assert hs.intermediate_query_due_in() is None
    assert len(io.sent) == count + 1
    assert np.array_equal(io.sent[-1], MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_KEEPALIVE_B,'500')))
    count = len(io.sent)
    hs.on_rx_audio(idle)
    hs._took_query_turn_idle()
    assert hs.payload_ledger.pending == 0 and len(io.sent) == count
    assert not any('retry DATA' in m for m in io.msgs)


@pytest.mark.parametrize('negative', ['unsolicited','nonfinal','wall-age','sample-age',
                                     'wrong-caller','truncated','not-chat','other-record'])
def test_idle_outside_measured_final_query_state_never_credits_or_keys(clock, negative):
    hs, io = (_pending(clock) if negative=='unsolicited' else
              _queried(clock,payload=b'x'*200 if negative=='nonfinal' else b'8 73<DISC>'))
    idle = _frame(hs, VF.SESSION_TURN_IDLE_RESPONDER,
                  'K5FIT' if negative=='wrong-caller' else hs.caller)
    if negative=='wall-age': clock.now += 5
    if negative=='sample-age': hs._intermediate_query_samples = 5*MK.FS
    if negative=='truncated': idle = idle[:int(1.0*MK.FS)]
    if negative=='not-chat': hs.chat = False
    if negative=='other-record': hs._tx_pending = (*hs._tx_pending[:2],3)
    pending, queued, ledger, count = hs._tx_pending, list(hs._txq), hs.payload_ledger.pending, len(io.sent)
    assert not hs._peer_query_turn_idle(idle)
    if negative in ('unsolicited', 'nonfinal'):
        hs.on_rx_audio(idle)
    assert hs._tx_pending == pending and hs._txq == queued
    assert hs.payload_ledger.pending == ledger and len(io.sent) == count


def test_text_final_idle_settles_original_host_bytes(clock):
    payload = b'hello from W9SSJ '*2
    hs, io = _queried(clock, payload=payload, compression='TEXT')
    assert hs.payload_ledger.pending == len(payload)
    assert hs._final_short_over()
    # Production TEXT currently uses measured plain fallback for terminal overs.
    assert hs.session_compression == 'TEXT'
    hs.on_rx_audio(_frame(hs,VF.SESSION_TURN_IDLE_RESPONDER))
    assert hs._tx_pending is None and hs.payload_ledger.pending == 0


def test_refused_keepalive_does_not_retransmit_acknowledged_data(clock):
    hs, io = _queried(clock)
    io.tx_went_out = lambda: False
    hs.on_rx_audio(_frame(hs,VF.SESSION_TURN_IDLE_RESPONDER))
    assert hs.turn == VA._TURN_PEER and hs._tx_pending is None
    assert hs.payload_ledger.pending == 0
    io.tx_went_out = lambda: True
    count = len(io.sent)
    clock.now += 3.48
    hs.query_intermediate_answer()
    hs.idle_keepalive()
    assert len(io.sent) == count and not any('retry DATA' in m for m in io.msgs)


# Read the original tape without altering it. Portable generated gates above
# remain available when the operator's local recordings are not installed.
def _ke8plm_clip(utc, seconds):
    from datetime import datetime
    from hfmodem.tests import evidence
    import shutil
    import subprocess

    source = (evidence.RECORD / 'operator-data/working/vara-chat/'
                  'sessions/20261024T022751257601Z/receive.flac')
    ffmpeg = shutil.which('ffmpeg')
    if not source.exists() or not ffmpeg:
        pytest.skip('KE8PLM native recording and ffmpeg required')
    start = datetime.fromisoformat('2026-10-24T02:28:09.689')
    offset = (datetime.fromisoformat('2026-10-24T' + utc) - start).total_seconds()
    raw = subprocess.check_output([ffmpeg, '-v', 'error', '-threads', '1',
                                   '-ss', str(offset), '-i', str(source),
                                   '-t', str(seconds), '-ac', '1', '-ar', str(MK.FS),
                                   '-f', 'f64le', '-'])
    return np.frombuffer(raw, dtype='<f8')


def test_native_ke8plm_first_idle_reply_stops_the_query_train(clock):
    hs, io = _queried(clock)
    hs.called = 'KE8PLM'
    pending = hs.payload_ledger.pending
    _stream(hs, clock, _ke8plm_clip('02:47:30.5', 1.8))
    assert hs.turn == VA._TURN_PEER, io.msgs
    assert hs._tx_pending is None and hs.payload_ledger.pending == 0
    assert pending == 10 and hs._tx_retries == 0
    assert hs.intermediate_query_due_in() is None
    assert io.msgs[-1].startswith('tx session-keepalive-b'), io.msgs


@pytest.mark.parametrize('utc,accepted', [('02:46:43.7',True), ('02:47:22.8',False)])
def test_native_grants_match_stock_acceptance(clock, utc, accepted):
    hs, io = _queried(clock)
    hs.called = 'KE8PLM'
    assert hs._peer_turn_grant(_ke8plm_clip(utc, .85)) is accepted


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('utc,accepted', [('02:46:43.7',True), ('02:47:22.8',False)])
def test_native_grant_caller_routes_agree_with_stock(clock, stream, utc, accepted):
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'],io,bw='500',chat=True)
    hs.role, hs.caller, hs.called = 'initiator','W9SSJ','KE8PLM'
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn, hs._released = VA._TURN_PEER, True
    hs._link_high_speed, hs._rec500 = True, 4
    hs._lead = TX.VB6Rnd(VF.lead_state('W9SSJ',False,'500'))
    hs.send(b'8 73<DISC>')
    wave = _ke8plm_clip(utc,.85)
    started, keyed_at = clock.now, []
    tx = io.tx
    def record_tx(samples):
        keyed_at.append(clock.now - started)
        tx(samples)
    io.tx = record_tx
    (_stream(hs,clock,wave) if stream else hs.on_rx_audio(wave))
    data_keys = sum(m.startswith('tx DATA over') for m in io.msgs)
    assert data_keys == int(accepted), io.msgs
    assert (hs._tx_pending is not None) is accepted
    assert hs.payload_ledger.pending == 10
    if stream and accepted:
        # The native1 fitted grant starts at .095 s and lasts .47 s.
        assert keyed_at and .565 <= keyed_at[0] <= .85
