# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert, resample_poly

from hfmodem.shrike import p4chirp, p4chirprx, p4frame
from hfmodem.tests.shrike.test_p4chirp import REAL_FIELDS


def test_recorded_dragon_chirps():
    path = Path(__file__).parent / 'fixtures/p4-chirps/audio.npz'
    with np.load(path, allow_pickle=False) as fixture:
        rows = fixture['audio']
        fs = int(fixture['fs'])
    for index, audio in enumerate(rows):
        result = p4chirprx.decode_packet(audio, fs=fs, frequency_offset=-242.4)
        assert result is not None
        assert result.marker == (9, 9, 8, 8)[index]
        assert result.rank0_upper == (False, False, True, True)[index]
        assert p4frame.encode_field(result.frame.data, result.frame.status) == REAL_FIELDS[index]
        assert p4chirprx.decode_packet(audio[:-1], fs=fs, frequency_offset=-242.4) is None
        assert p4chirprx.decode_packet(audio, fs=fs, frequency_offset=0) is None


@pytest.mark.parametrize('fs', [8000, 12000, 48000])
@pytest.mark.parametrize('offset', [-23., 0., 19.])
def test_shifted_render_with_noise(fs, offset):
    data = b'P4 field test'.ljust(22, b'!')
    x = p4chirp.entry_packet(data, 0x62)
    x = (hilbert(x) * np.exp(2j * np.pi * offset * np.arange(len(x)) / 48000)).real
    if fs != 48000:
        x = resample_poly(x, 1, 48000 // fs)
    x += np.random.default_rng(42).normal(0, .01, len(x))
    x = np.r_[np.zeros(123), x]
    result = p4chirprx.decode_packet(x, fs=fs, start_sample=123, frequency_offset=offset)
    assert result is not None
    assert result.frame.data == data and result.frame.status.byte == 0x62


def test_silence_noise_and_wrong_header():
    n = round(48000 * p4chirp.PACKET_S)
    assert p4chirprx.decode_packet(np.zeros(n)) is None
    assert p4chirprx.decode_packet(np.random.default_rng(9).normal(size=n)) is None
    audio = p4chirp.entry_packet(b'field', 0)
    audio[:round(.19 * 48000)] = 0
    assert p4chirprx.decode_packet(audio) is None


@pytest.mark.parametrize('kwargs', [dict(fs=0), dict(start_sample=-1),
    dict(frequency_offset=float('nan'))])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        p4chirprx.decode_packet(np.zeros(4), **kwargs)


@pytest.mark.parametrize('audio', [np.array([float('nan')]), np.zeros((2, 2)),
                                   np.array([1j])])
def test_invalid_audio(audio):
    with pytest.raises(ValueError):
        p4chirprx.decode_packet(audio)
