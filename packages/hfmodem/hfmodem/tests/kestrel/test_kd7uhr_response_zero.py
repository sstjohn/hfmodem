# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KD7UHR recordings are level-1 offers, independently accepted by stock VARA."""
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as va
from .test_bw2750_peer_answer import awaiting

FIXTURES = Path(__file__).with_name('fixtures') / 'kd7uhr-response-zero-0920'


@pytest.mark.parametrize('seq', [4, 5, 6])
@pytest.mark.parametrize('chunk', [960, 4096, 4800])
def test_native_position_zero_sends_level_one_setup(seq, chunk):
    path = FIXTURES / f'after-{seq}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == 48000
    hs = awaiting('KD7UHR')
    for at in range(0, len(x), chunk):
        hs.on_rx_stream(x[at:at + chunk].astype(float) / 32768)
    assert [a.position for a in hs.answers] == [0]
    assert hs.answers[0].tones == 15
    assert hs.state == va.VaraState.CONNECTING and hs.step == va._I_LINKSETUP_SENT
    assert not hs.answer_retry and hs.io.keyed == [True, False]
    assert hs._setup_level == 1
    assert any('level 1' in m for m in hs.io.logs)
    assert not any('could not read' in m for m in hs.io.logs)


@pytest.mark.parametrize('call,bw', [('N0CALL', '2750'), ('KD7UHR', '2300'),
                                     ('KD7UHR', '500')])
def test_native_position_zero_cannot_cross_call_or_bandwidth(call, bw):
    path = FIXTURES / 'after-4.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    hs = awaiting(call, bw)
    for at in range(0, len(x), 4800):
        hs.on_rx_stream(x[at:at + 4800].astype(float) / 32768)
    assert not hs.answers and not hs.io.keyed
    assert hs.state == va.VaraState.CONNECTING
