# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock-confirmed positive tails are distinct from the common NACK shape."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_data_over_gate import _connected

HERE = Path(__file__).with_name('fixtures') / 'data-ack-pairs-0919'
# Recording corpora are optional in the public distribution.
ROWS = (json.loads((HERE / 'manifest.json').read_text())
        if (HERE / 'manifest.json').is_file() else [])


@pytest.mark.parametrize('row', ROWS, ids=[r['name'] for r in ROWS])
def test_native_positive_tail_advances_one_pending_frame(row):
    hs, io = _connected(bw=row['bw'])
    hs.turn = VA._TURN_OURS
    size = phy.body_size(row['bw'], row['record'])
    hs._tx_pending = (phy.vara_body(b'A' * (size - 1), hs.caller,
                                  tail=0x89, body_len=size), 1, row['record'])
    hs._over = 1
    hs._txq = [b'last']
    hs._pending_answer_at = time.monotonic()
    fs, audio = wavfile.read(HERE / row['name'])
    assert fs == MK.FS
    audio = audio.astype(float) / (32768 if audio.dtype == np.int16 else 1)
    table = (VF.OVER_CONTINUE_RESPONDER_DOWN_BY_LINK if row['kind'] == 'down'
             else VF.OVER_CONTINUE_RESPONDER_BY_LINK)
    assert tuple(map(tuple, row['pairs'])) == table[hs.caller, hs.called, hs.bw]
    assert not hs._peer_data_nak(audio)
    assert hs._peer_over_continue(audio)
    if hs.bw == '500':
        hs.on_rx_audio(audio)
    else:
        for at in range(0, len(audio), 4096):
            hs.on_rx_stream(audio[at:at + 4096])
    assert hs._tx_pending[1] == 2 and len(io.sent) == 1
    assert not hs._txq
