# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from scipy.signal import hilbert, resample_poly

from hfmodem.shrike import p4chirp, p4chirprx
from hfmodem.tests.shrike.corpora import n9vc_0802

FS = 48000


@pytest.fixture(scope='module')
def stock():
    fs, pcm = n9vc_0802()
    assert fs == FS
    audio = pcm[round(6*fs):round(10.3*fs)]/32768
    audio = np.real(hilbert(audio)*np.exp(-2j*np.pi*400*np.arange(len(audio))/fs))
    return audio, round(.660*fs)


@pytest.mark.parametrize('error_ms', [-35, -20, -10, -1, 0, 1, 10, 20, 35])
def test_stock_header_places_one_crc_attempt_before_the_body_arrives(stock, error_ms, monkeypatch):
    audio, at = stock
    nominal = at+round(error_ms*FS/1000)
    early = nominal+round(.235*FS)
    with monkeypatch.context() as patch:
        patch.setattr(p4chirprx.pactor2, 'decode_frame', lambda *a: pytest.fail('header acquisition tried CRC'))
        assert p4chirprx.acquire_header(audio[:early-1], nominal_sample=nominal) is None
        started = perf_counter()
        header = p4chirprx.acquire_header(audio[:early], nominal_sample=nominal)
        elapsed = 1000*(perf_counter()-started)
    assert header is not None and header.marker == 8 and header.score >= .85
    assert abs(header.start_sample-nominal) <= round(.035*FS)
    assert p4chirprx.decode_packet(audio[:early], start_sample=header.start_sample) is None
    result = p4chirprx.decode_packet(audio, start_sample=header.start_sample)
    assert result is not None and result.frame.data == b'[BPQ-6.0.25.1-B2FWIHJM'
    assert result.frame.status.byte == 0x61
    print(dict(error_ms=error_ms, selected_ms=1000*(header.start_sample-at)/FS,
               score=header.score, acquire_ms=elapsed))


@pytest.mark.parametrize('fs', [8000, 12000, 48000])
@pytest.mark.parametrize('cfo', [-23., 0., 19.])
@pytest.mark.parametrize('error_ms', [-35, 35])
def test_rendered_header_acquires_with_sample_rate_and_carrier(fs, cfo, error_ms):
    signal = p4chirp.entry_packet(b'bounded header'.ljust(22, b'!'), 0x62)
    signal = np.real(hilbert(signal)*np.exp(2j*np.pi*cfo*np.arange(len(signal))/FS))
    if fs != FS:
        signal = resample_poly(signal, 1, FS//fs)
    at = round(.2*fs)
    audio = np.r_[np.zeros(at), signal, np.zeros(round(.05*fs))]
    nominal = at+round(error_ms*fs/1000)
    header = p4chirprx.acquire_header(audio[:nominal+round(.235*fs)], fs=fs,
                                    nominal_sample=nominal, frequency_offset=cfo)
    assert header is not None and abs(header.start_sample-at) < round(.003*fs)
    result = p4chirprx.decode_packet(audio, fs=fs, start_sample=header.start_sample, frequency_offset=cfo)
    assert result is not None and result.frame.data == b'bounded header'.ljust(22, b'!')


def test_header_never_depends_on_the_payload_or_expands_its_bracket():
    signal = p4chirp.entry_packet(b'payload', 0)
    audio = np.r_[np.zeros(9600), signal, np.zeros(4800)]
    header = p4chirprx.acquire_header(audio, nominal_sample=9600)
    changed = audio.copy()
    changed[9600+round(.24*FS):] = 0
    assert p4chirprx.acquire_header(changed, nominal_sample=9600) == header
    assert p4chirprx.decode_packet(changed, start_sample=header.start_sample) is None
    assert p4chirprx.acquire_header(audio, nominal_sample=round(.35*FS)) is None
    assert p4chirprx.acquire_header(np.zeros(FS), nominal_sample=9600) is None


def test_both_recorded_header_markers_choose_a_crc_valid_placement():
    path = Path(__file__).parent/'fixtures/sigidwiki/p4-chirps/audio.npz'
    with np.load(path, allow_pickle=False) as fixture:
        rows, fs = fixture['audio'], int(fixture['fs'])
    for row, marker in zip(rows, (9, 9, 8, 8)):
        at = round(.1*fs)
        audio = np.r_[np.zeros(at), row, np.zeros(round(.05*fs))]
        header = p4chirprx.acquire_header(audio[:at+round(.260*fs)], fs=fs,
            nominal_sample=at+round(.025*fs), frequency_offset=-242.4)
        assert header is not None and header.marker == marker
        result = p4chirprx.decode_packet(audio, fs=fs, start_sample=header.start_sample,
                                       frequency_offset=-242.4)
        assert result is not None and result.marker == marker


@pytest.mark.parametrize('kwargs', [dict(fs=0), dict(nominal_sample=-1),
    dict(frequency_offset=float('nan')), dict(slack_s=.1)])
def test_header_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        p4chirprx.acquire_header(np.zeros(FS), **{'nominal_sample': 9600, **kwargs})
