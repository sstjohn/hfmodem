# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""N3MEL answered a queried record-2 close with caller-keyed 1078/63."""
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.contact.slots import Mixer
from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_bw500_query_recovery import _IO, clock as clock  # noqa: F401


def pending(clock, payload=b'hello?\r'):
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True)
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'N3MEL'
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    hs._link_high_speed, hs._rec500 = True, 2
    hs._lead = TX.VB6Rnd(VF.lead_state('W9SSJ', False, '500'))
    hs.send(payload)
    assert hs._tx_pending[2] == 2
    clock.now += 1.3
    assert hs.query_intermediate_answer()
    return hs, io


def wave(native):
    if native:
        fs, x = wavfile.read(Path(__file__).parent / 'fixtures/n3mel-1024-query-idle.wav')
        assert fs == MK.FS
        return Mixer(1500)(x)
    return np.r_[np.zeros(4800), MK.synth_burst('W9SSJ', VF.for_bw(
        VF.SESSION_TURN_IDLE_RESPONDER, '500')), np.zeros(14400)]


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('block_size', [960, 4096, 24000])
def test_record2_final_query_idle_retires_once(clock, native, block_size):
    hs, io = pending(clock)
    assert hs.payload_ledger.pending == 7
    count = len(io.sent)
    x = wave(native)
    for at in range(0, len(x), block_size):
        block = x[at:at + block_size]
        clock.now += len(block) / MK.FS
        hs.on_rx_stream(block)
    assert hs._tx_pending is None, io.msgs
    assert hs.payload_ledger.pending == 0 and hs.turn == VA._TURN_PEER
    assert len(io.sent) == count + 1
    assert np.array_equal(io.sent[-1], MK.synth_burst('N3MEL', VF.for_bw(
        VF.SESSION_KEEPALIVE_B, '500')))
    assert hs.intermediate_query_due_in() is None
    count = len(io.sent)
    hs.on_rx_audio(x)
    hs._took_query_turn_idle()
    assert hs.payload_ledger.pending == 0 and len(io.sent) == count


@pytest.mark.parametrize('negative', ['wrong-caller', 'truncated', 'nonfinal',
                                      'stale-wall', 'stale-samples', 'refused'])
def test_record2_query_idle_requires_solicited_final_complete_reply(clock, negative):
    hs, io = pending(clock, b'x'*200 if negative == 'nonfinal' else b'hello?\r')
    x = wave(False)
    if negative == 'wrong-caller':
        x = np.r_[np.zeros(4800), MK.synth_burst('K5FIT', VF.for_bw(
            VF.SESSION_TURN_IDLE_RESPONDER, '500')), np.zeros(14400)]
    if negative == 'truncated': x = x[:int(1.0 * MK.FS)]
    if negative == 'stale-wall': clock.now += 5
    if negative == 'stale-samples': hs._intermediate_query_samples = 5 * MK.FS
    if negative == 'refused':
        hs._intermediate_query_for = None
        io.tx_went_out = lambda: False
        clock.now += 2
        assert not hs.query_intermediate_answer()
    before = hs._tx_pending, list(hs._txq), hs.payload_ledger.pending, len(io.sent)
    assert not hs._peer_query_turn_idle(x)
    hs.on_rx_audio(x)
    assert (hs._tx_pending, hs._txq, hs.payload_ledger.pending, len(io.sent)) == before
