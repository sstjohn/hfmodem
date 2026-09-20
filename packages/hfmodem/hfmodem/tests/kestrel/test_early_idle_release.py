# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A complete idle can disprove a falsely held second-block window."""
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from hfmodem.kestrel.arq import phy
from .test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name('fixtures') / 'kc9ghz-early-idle-0920'


def held():
    hs, io = _connected(bw='2750')
    hs.turn = A._TURN_PEER
    hs._deliver([phy.vara_body(b'A'*89, hs.caller)], hold=True)
    hs._answer_over(False, False, True)
    return hs, io


@pytest.mark.parametrize('over', [1, 2])
@pytest.mark.parametrize('chunk', [960, 1500, 4096])
def test_first_native_idle_releases_once(over, chunk):
    path = FIXTURES / f'after-data-{over}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == M.FS
    hs, io = held()
    x = x.astype(float)/32768
    for i in range(0, len(x), chunk):
        hs.on_rx_stream(x[i:i+chunk])
    assert io.keys == 1, io.msgs
    assert hs._held_answer is None and not hs._window_bodies
    assert io.host == [b'A'*89]
    assert any('complete early idle' in m for m in io.msgs)


@pytest.mark.parametrize('elapsed', [0, 1.5, 4.0, 7.0])
def test_complete_idle_outside_early_cadence_cannot_release_half_window(elapsed):
    hs, io = held()
    hs._held_samples = int(elapsed*M.FS)
    kind = F.for_bw(F.SESSION_RESPONDER_IDLE, hs.bw)
    x = np.pad(M.synth_burst(hs.called, kind), (0, 4800))
    assert hs._peer_responder_idle(x)
    assert hs._idle_complete_exact
    assert not hs._a_gap()
    hs._release_held_answer(0)
    assert not io.sent and hs._held_answer is not None


def test_partial_idle_name_is_not_enough_even_at_the_right_time():
    hs, io = held()
    hs._held_samples = int(3.5*M.FS)
    hs._idle_kind = F.SESSION_RESPONDER_IDLE
    hs._idle_complete_exact = False
    assert not hs._a_gap()
    hs._release_held_answer(0)
    assert not io.sent and hs._held_answer is not None
