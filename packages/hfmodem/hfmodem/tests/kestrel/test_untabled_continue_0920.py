# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Three gateways no table ever held, answering a full intermediate over.

Each reply was logged on 2026-09-20 as an unclassified eight-symbol DATA reply
and cost a query round trip before a called-keyed 288/1 confirmed it. Each is
the called station's generated continue.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name('fixtures') / 'untabled-continue-0920'
NAMES = ['n5tw', 'kb2pcn-8', 'kd0sfy']


def case(name):
    path = FIXTURES / f'{name}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    row = json.loads((FIXTURES / 'provenance.json').read_text())[f'{name}.wav']
    fs, x = wavfile.read(path)
    assert fs == MK.FS and x.dtype == np.int16
    return row, x.astype(float) / 32768


def pending(row):
    hs, io = _connected(bw=row['bw'])
    hs.called = row['called']
    hs._peer_offset = row['offset_hz'] / (MK.FS / MK.NFFT)
    hs.turn = VA._TURN_OURS
    hs._over = row['answers_data_over'] - 1
    hs.send(b'A' * 89 + b'B' * 89 + b'end')
    return hs, io


@pytest.mark.parametrize('name', NAMES)
def test_the_reply_is_the_called_stations_generated_continue(name):
    row, x = case(name)
    hs, _ = pending(row)
    track = VA._top3_track(x, band=hs._band, bin_offset=hs._grid)
    at, width = VA._widest(VA._cont_held(track[0], track[1], VA._live_track(x),
                                         hs._peer_shift))
    assert width >= VA._CONT_PLATEAU
    read = tuple(tuple(sorted(int(v - hs._peer_shift) for v in p))
                 for p in track[0][at + width // 2 + VA._CONT_OFF, :2])
    assert read == VF.over_continue(row['called'], row['bw'])[1]


@pytest.mark.parametrize('chunk', [512, 4096, 4800])
@pytest.mark.parametrize('name', NAMES)
def test_the_reply_advances_once_without_a_query(name, chunk):
    row, x = case(name)
    hs, io = pending(row)
    old = hs._tx_pending
    for at in range(0, len(x), chunk):
        hs.on_rx_stream(x[at:at + chunk])
    assert hs._tx_pending[1] == old[1] + 1 and len(io.sent) == 2
    assert hs._intermediate_query_attempts == 0
    assert not [m for m in io.msgs if 'unclassified' in m], io.msgs
