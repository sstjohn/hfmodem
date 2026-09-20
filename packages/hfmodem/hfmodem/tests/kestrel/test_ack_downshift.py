# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock caller replay proves 288/311 retires DATA and lowers the next record."""
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from hfmodem.kestrel.arq import phy
from .test_data_over_gate import _connected


def sender(requested=None, total=527):
    hs, io = _connected(bw='2750')
    hs.tx_level = requested
    hs.turn = A._TURN_OURS
    payload = bytes(33+i%80 for i in range(total))
    hs.send(payload)
    hs._took_control_burst(continue_reply=True)
    return hs, io, payload


def native():
    path = Path(__file__).with_name('fixtures') / 'kc9ghz-downshift-0920/query-answer.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == M.FS
    return x.astype(float)/32768


@pytest.mark.parametrize('requested', [None, 4])
@pytest.mark.parametrize('chunk', [960, 4096])
def test_native_query_reply_retires_only_pending_and_repacks_unsent(requested, chunk):
    hs, io, payload = sender(requested)
    hs.send(b'later host write')
    old = hs._tx_pending
    hs._probe_intermediate_answer()
    hs._peer_offset = 3.1/(M.FS/M.NFFT)
    x = native()
    for i in range(0,len(x),chunk):hs.on_rx_stream(x[i:i+chunk])
    assert hs._tx_pending[1] == old[1]+1 and hs._tx_pending[2] == 102, io.msgs
    body = hs._tx_pending[0]
    assert body[:-1] == payload[178:225]
    assert body[-1] == 0x89
    assert body[:-1] + b''.join(hs._txq[:-1]) == payload[178:]
    assert hs._txq[-1] == b'later host write'
    assert hs._tx_recovery_levels == [102]*7
    before = hs._tx_pending, list(hs._txq), len(io.sent)
    hs.on_rx_audio(x)
    hs._took_intermediate_query_answer()
    assert (hs._tx_pending, hs._txq, len(io.sent)) == before
    for _ in range(8):hs._took_control_burst(continue_reply=True)
    assert hs._tx_pending[2] == 103
    assert phy.vara_payload(hs._tx_pending[0],caller=hs.caller) == b'later host write'


def test_direct_downshift_matches_stock_and_preserves_exact_multiple_close():
    hs, io, payload = sender(total=178+94)
    x = np.pad(M.synth_tone_pairs(F.OVER_CONTINUE_RESPONDER_DOWN_BY_LINK['W9SSJ','KC9GHZ','2750']), (4800,4800))
    assert hs._peer_over_continue(x)
    hs._took_over_continue()
    assert hs._tx_pending[2] == 102
    assert hs._tx_pending[0][:-1] == payload[178:225]
    assert hs._txq == [payload[225:],b'']
    assert hs._tx_recovery_levels == [102,102]


def test_refused_next_transmission_keeps_repacked_bytes_and_level():
    hs, io, payload = sender()
    hs._probe_intermediate_answer()
    hs._peer_offset = 3.1/(M.FS/M.NFFT)
    x = native()[:round(1.5*M.FS)]
    assert hs._peer_intermediate_query_answer(x)
    io.tx_went_out = lambda:False
    hs._took_intermediate_query_answer()
    assert hs._tx_pending is None and b''.join(hs._txq) == payload[178:]
    assert hs._tx_recovery_levels == [102]*8
    io.tx_went_out = lambda:True
    hs._tx_data_over()
    assert hs._tx_pending[2] == 102 and hs._tx_pending[0][:-1] == payload[178:225]


@pytest.mark.parametrize('bw', ['2300','2750'])
def test_repeated_downshift_retires_each_pending_block_without_losing_bytes(bw):
    hs, io = _connected(bw=bw)
    hs.turn = A._TURN_OURS
    payload = bytes(33+i%80 for i in range(527))
    hs.send(payload)
    accepted = 0
    for expected_level in range(phy.base_level(bw)-1, phy.base_level(bw)-4, -1):
        accepted += len(hs._tx_pending[0])-1
        hs._probe_intermediate_answer()
        kind = F.for_bw(F.SESSION_RESPONDER_OVER_ANSWER,bw)
        x = np.pad(M.synth_burst(hs.called,kind), (0,4800))
        assert hs._peer_intermediate_query_answer(x)
        hs._took_intermediate_query_answer()
        assert hs._tx_pending[2] == expected_level
        remaining = hs._tx_pending[0][:-1] + b''.join(hs._txq)
        assert remaining == payload[accepted:]
        assert hs._tx_retries == 0
