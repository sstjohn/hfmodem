# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""W0TRM's record-5 final reply and a farewell queued behind that delivery."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.contact.slots import Mixer
from hfmodem.contact.varac import frame_message
from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from hfmodem.tests.kestrel.test_bw500_query_recovery import _IO, clock as clock  # noqa: F401


def pending(clock, record=5, payload=None):
    io = _IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True)
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'W0TRM'
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_OURS
    hs._link_high_speed, hs._rec500 = True, record
    hs._lead = TX.VB6Rnd(F.lead_state('W9SSJ', False, '500'))
    hs.send(frame_message('<QF:14104250>') if payload is None else payload)
    assert hs._tx_pending[2] == record
    clock.now += 1.3
    assert hs.query_intermediate_answer()
    return hs, io


def reply(native=True):
    if native:
        fs, x = wavfile.read(Path(__file__).with_name('fixtures') / 'w0trm-1025-query-idle.wav')
        assert fs == M.FS
        return Mixer(1500)(x)
    return np.r_[np.zeros(4800), M.synth_burst('W9SSJ', F.for_bw(
        F.SESSION_TURN_IDLE_RESPONDER, '500')), np.zeros(14400)]


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('block_size', [960, 4096, 24000])
@pytest.mark.parametrize('record', [2, 4, 5])
@pytest.mark.parametrize('queued', [False, True])
def test_final_reply_settles_only_queried_delivery(clock, native, block_size, record, queued):
    hs, io = pending(clock, record)
    original = hs.payload_ledger.pending
    assert original == 16
    goodbye = frame_message('73<DISC>')
    if queued:
        hs.send(goodbye)
    queue = list(hs._txq)
    count = len(io.sent)
    x = reply(native)
    for at in range(0, len(x), block_size):
        block = x[at:at + block_size]
        clock.now += len(block) / M.FS
        hs.on_rx_stream(block)
    assert hs._tx_pending is None and hs.turn == A._TURN_PEER, io.msgs
    assert hs.payload_ledger.confirmed == original
    assert hs.payload_ledger.pending == (len(goodbye) if queued else 0)
    assert hs.payload_ledger.unconfirmed == (goodbye if queued else b'')
    assert hs._txq == queue and hs.intermediate_query_due_in() is None
    assert len(io.sent) == count + 1
    assert np.array_equal(io.sent[-1], M.synth_burst('W0TRM', F.for_bw(F.SESSION_KEEPALIVE_B, '500')))
    count = len(io.sent)
    hs.on_rx_audio(x)
    hs._took_query_turn_idle()
    assert hs.payload_ledger.confirmed == original and hs._txq == queue
    assert len(io.sent) == count


@pytest.mark.parametrize('negative', ['nonfinal', 'wrong-caller', 'truncated', 'stale',
                                      'stale-samples', 'refused', 'replaced', 'not-chat'])
def test_record5_does_not_credit_unqualified_reply(clock, negative):
    hs, io = pending(clock, payload=b'x'*500 if negative == 'nonfinal' else None)
    hs.send(frame_message('73<DISC>'))
    x = reply()
    if negative == 'wrong-caller': hs.caller = 'K5FIT'
    if negative == 'truncated': x = x[:48000]
    if negative == 'stale': clock.now += 5
    if negative == 'stale-samples': hs._intermediate_query_samples = 5*M.FS
    if negative == 'refused': hs._intermediate_query_for = None
    if negative == 'replaced': hs._intermediate_query_for = (b'other', 0, 5)
    if negative == 'not-chat': hs.chat = False
    before = hs._tx_pending, list(hs._txq), hs.payload_ledger.pending, len(io.sent)
    assert not hs._peer_query_turn_idle(x)
    if negative not in ('wrong-caller', 'truncated'):
        hs._took_query_turn_idle()
    assert (hs._tx_pending, hs._txq, hs.payload_ledger.pending, len(io.sent)) == before
