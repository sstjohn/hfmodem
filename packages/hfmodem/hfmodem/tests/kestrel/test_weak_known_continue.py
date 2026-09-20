# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Host-confirmed K0SI ACKs whose pairs failed the generic shape threshold."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name('fixtures') / 'k0si-weak-continue-0920'


def pending(bw='2300', called='K0SI'):
    hs, io = _connected(bw=bw)
    hs.called = called
    hs.turn = A._TURN_OURS
    hs.send(b'A'*89 + b'B'*89 + b'last')
    return hs, io


@pytest.mark.parametrize('over', [4, 6])
@pytest.mark.parametrize('chunk', [None, 960, 4096])
def test_native_ack_advances_without_query(over, chunk):
    path = FIXTURES / f'over-{over}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == M.FS
    x = x.astype(float) / 32768
    hs, io = pending()
    hs._peer_offset = -8.4 / (M.FS / M.NFFT)
    assert A._cont_plateau(x, band=hs._band, bin_offset=hs._grid) == 0
    before = hs._tx_pending[1]
    if chunk is None:
        hs.on_rx_audio(x)
    else:
        for i in range(0, len(x), chunk):
            hs.on_rx_stream(x[i:i+chunk])
    assert hs._tx_pending[1] == before + 1, io.msgs
    assert len(io.sent) == 2 and hs._txq == [b'last'], io.msgs
    assert not hs._intermediate_query_attempted
    if chunk is not None:
        hs.on_rx_audio(x)
        assert len(io.sent) == 2


@pytest.mark.parametrize('over', [4, 6])
def test_native_wrong_link_does_not_advance(over):
    path = FIXTURES / f'over-{over}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    hs, io = pending(called='KC9GHZ')
    hs._peer_offset = -8.4 / (M.FS / M.NFFT)
    old = hs._tx_pending
    hs.on_rx_audio(x.astype(float)/32768)
    assert hs._tx_pending == old and len(io.sent) == 1


@pytest.mark.parametrize('link,pairs', list(F.DATA_NAK_RESPONDER_BY_LINK.items()))
@pytest.mark.parametrize('weak', [(), (0,), (3, 5)])
def test_known_nak_with_weak_pairs_never_becomes_ack(link, pairs, weak):
    caller, called, bw = link
    hs, io = pending(bw=bw, called=called)
    hs.caller = caller
    x = M.synth_tone_pairs(pairs)
    for i in weak:
        x[i*M.HOP:(i+1)*M.HOP] *= .02
    x = np.pad(x, (4800, 4800))
    assert not hs._peer_over_continue(x)
