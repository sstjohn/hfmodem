# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A repeated caller setup must never enter the connected DATA byte stream."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.rx import varahf500 as R
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_ofdm as O
from .test_idle_poll import clock as clock, _IO as _IO
from .test_responder_control_stream import feed

FIXTURE = Path(__file__).with_name('fixtures') / 'kr7v-1025-setup.wav'
FRAME = bytes.fromhex('2d28960000000083149d0000000000000000000000034204d6')


def recorded():
    fs, pcm = wavfile.read(FIXTURE)
    assert fs == 48000 and pcm.dtype == np.int16
    provenance = json.loads(FIXTURE.with_suffix('.json').read_text())
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == provenance['pcm_sha256']
    return pcm.astype(float) / 32768


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('marked,compression', [(False, 'OFF'), (True, 'OFF'),
                                             (True, 'TEXT'), (True, 'FILES')])
def test_all_negotiated_setup_formats_are_excluded_from_data(level, marked, compression):
    audio = O.link_setup_tx('KR7V', bw='500', level=level,
                            high_speed=marked, compression=compression)
    result = R.decode_stream(np.r_[audio, np.zeros(24000)])
    assert len(result.frames) == 1 and result.frames[0].crc_ok
    assert bytes(result.frames[0].frame_bytes) == F.link_setup_frame(
        'KR7V', '500', level, high_speed=marked, compression=compression)
    assert result.frames[0].is_connect
    assert not result.data_frames and not result.payload


def test_recorded_kr7v_setup_is_not_chat_data():
    result = R.decode_stream(recorded())
    assert len(result.frames) == 1 and result.frames[0].crc_ok
    assert bytes(result.frames[0].frame_bytes) == FRAME
    assert result.frames[0].is_connect
    assert not result.data_frames and not result.payload


@pytest.mark.parametrize('block', [960, 4096])
def test_recorded_setup_repeat_cannot_deliver_identity_bytes_or_advance_progress(block):
    io = _IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True, high_speed=True)
    hs.role, hs.called, hs.caller = 'responder', 'W9SSJ', 'KR7V'
    hs.state = A.VaraState.CONNECTING
    hs._chat_profile = True
    assert hs._take_link_setup(FRAME)
    hs.send(b'queued greeting')
    before = (hs.progress, hs.payload_ledger.pending, hs.payload_ledger.confirmed)
    feed(hs, np.r_[np.zeros(4800), recorded(), np.zeros(24000)], block)
    assert not io.host
    assert (hs.progress, hs.payload_ledger.pending, hs.payload_ledger.confirmed) == before
    assert hs._answer_owed is None and hs._peer_over == 0
