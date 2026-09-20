# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CRC-clean low-speed greeting discarded by the reference-only gate."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.vara import vara_arq as va
from .test_bw2750_low_levels import _IO

FIXTURES = Path(__file__).with_name('fixtures') / 'weak-data-0920'
PAYLOAD = b'RMS Trimode 1.4.3.0 Wa'


def audio(name='kc9ghz-record101'):
    path = FIXTURES / (name + '.wav')
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == 48000
    return x.astype(float) / 32768


def connected():
    io = _IO()
    hs = va.VaraStationHandshake(['W9SSJ'], io, bw='2750')
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'KC9GHZ'
    hs.state, hs.step = va.VaraState.CONNECTED, va._I_CONNECTED
    hs.turn = va._TURN_PEER
    return hs, io


def feed(hs, io, x, chunk):
    for at in range(0, len(x), chunk):
        io.received = min(at + chunk, len(x))
        hs.on_rx_stream(x[at:at + chunk])


@pytest.mark.parametrize('chunk', [960, 4096, 4800])
def test_recorded_greeting_reaches_host_and_retains_its_answer(chunk):
    hs, io = connected()
    feed(hs, io, audio(), chunk)
    assert io.payloads == [PAYLOAD], io.logs
    assert len(io.answers) == 1, io.logs
    # Column zero is +0.0912 s; 228 columns of 1024 samples finish at +4.9552 s.
    # Noise holds the window open; the existing bounded hold eventually answers.
    assert 4.955 * 48000 <= io.answers[0] <= 17 * 48000
    assert any('12/24' in line and 'CRC clean' in line for line in io.logs)


def test_weak_crc_failure_is_silent_and_does_not_create_a_nak_debt():
    x = audio()[:6 * 48000]
    # Destroy the coded payload while keeping the native reference columns.
    refs = set(rx._ROLES[101][0])
    rng = np.random.default_rng(920)
    for col in range(rx.RECORDS[101].ncols):
        if col in refs:
            continue
        start = 4376 + col * 1024
        x[start:start + 1024] = rng.normal(0, .18, 1024)
    assert not rx.decode_over(x, 0, len(x), level=101).crc_ok
    hs, io = connected()
    feed(hs, io, x, 4800)
    assert not io.payloads and not io.answers, io.logs
    assert hs._answer_owed is None and not hs._owed_block


def test_same_session_band_noise_does_not_key_or_deliver():
    hs, io = connected()
    feed(hs, io, audio('same-session-noise'), 4800)
    assert not io.payloads and not io.answers, io.logs
    assert hs._answer_owed is None


def test_weak_crc_clean_echo_still_cannot_reach_the_host():
    hs, io = connected()
    fr = rx.decode_over(audio(), 0, len(audio()), level=101)
    assert fr.crc_ok
    hs._keyed_bodies.add(bytes(fr.payload))
    feed(hs, io, audio()[:6 * 48000], 4800)
    assert not io.payloads and not io.answers, io.logs
