# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hashlib import sha256
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pytest

from hfmodem.shrike import p4frame, p4robustrx, p4rx, p4sf8rx, session

ROOT = Path(__file__).resolve().parents[5]
STREAM = ROOT/'captures/onair-1002-2154/stream.flac'
WAV_SHA = '6ad2b2cb24ca129a822ccf7cdb0f39e8ef7d088184daf582b20edf01d9e4374b'
DATA_SHA = 'e8acf31f32a128b6d89a3bb2780cf3fa513014358934a4ba3f8966a922704a92'


@pytest.fixture(scope='module')
def copies():
    if not STREAM.is_file():
        pytest.skip('onair-1002-2154 recording absent')
    decoded = subprocess.run(['flac', '-d', '-c', str(STREAM)], capture_output=True,
                             check=True).stdout
    assert sha256(decoded).hexdigest() == WAV_SHA
    audio = session.load_wav(str(STREAM), 48000)
    result = []
    for n in range(12):
        onset = 3528586 + 180000*n
        chunk = audio[onset-4800:onset-4800+round(3.48*48000)]
        start = perf_counter()
        baseband = p4rx.audio_to_baseband(chunk, 48000)
        converted = perf_counter()
        observed = p4rx.recover_robust(baseband, reference_start=8000,
                                       spread_factor=8, data_symbols=696,
                                       timing_offsets=range(-48, 49), cfo_hz=(.69,))
        placed = perf_counter()
        field = p4sf8rx.decode_symbols(observed.symbols)
        decoded_at = perf_counter()
        result.append((baseband, observed, field,
                       (converted-start, placed-converted, decoded_at-placed)))
    return result


def test_recorded_sf8_fields_and_budget(copies):
    assert [None if row[2] is None else row[2].correction for row in copies] == [
        'systematic', None, 'amplitude-c1', 'unit-c1', *(['systematic']*8)]
    for n, (_, observed, reception, times) in enumerate(copies):
        assert .50 <= observed.coherence <= .90
        if n == 1:
            assert reception is None
            continue
        assert reception.quarter_turn == 0
        assert reception.frame.status.byte == 0x03
        assert len(reception.frame.data) == 141
        assert sha256(reception.frame.data).hexdigest() == DATA_SHA
        assert sum(times) < .110


def test_wrong_layout_carrier_and_incomplete_body(copies):
    baseband, observed, reception, _ = copies[0]
    assert reception is not None
    assert p4sf8rx.decode_symbols(observed.symbols, stride=26) is None
    for turn in (1, 2, 3):
        assert p4sf8rx.decode_symbols(observed.symbols, turns=(turn,)) is None
    assert p4sf8rx.decode_symbols(observed.symbols[:300]) is None
    assert p4sf8rx.decode_symbols(np.zeros(697, complex)) is None
    for reference, carrier in ((8500, .69), (8000, 100.69)):
        wrong = p4rx.recover_robust(baseband, reference_start=reference,
                                    spread_factor=8, data_symbols=696,
                                    timing_offsets=(0,), cfo_hz=(carrier,))
        assert p4sf8rx.decode_symbols(wrong.symbols, c1=False) is None
    sf16 = p4rx.recover_robust(baseband, reference_start=8000,
                               spread_factor=16, data_symbols=348,
                               timing_offsets=(0,), cfo_hz=(.69,))
    assert p4robustrx.decode_symbols(sf16.symbols, iterations=0) is None


def test_crc_rejects_corruption_after_correction(copies):
    reception = copies[0][2]
    bits = reception.fields < 0
    bits[:64] = ~bits[:64]
    field = np.packbits(bits[:1152], bitorder='little').tobytes()
    assert p4frame.decode_field(field, data_bytes=141) is None
