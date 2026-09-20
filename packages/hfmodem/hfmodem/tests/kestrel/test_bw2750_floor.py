# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Native BW2750 host L1: independent captures, wire bytes and live RX timing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as va
from hfmodem.tests.kestrel.test_bw2750_low_levels import _IO

FIX = Path(__file__).with_name('fixtures') / 'bw2750-floor'

pytestmark = pytest.mark.skipif(
    not FIX.is_dir(),
    reason='the native BW2750 floor captures require the source checkout '
           '(fixtures/bw2750-floor)')


def native(name):
    meta = json.loads((FIX / 'provenance.json').read_text())[name]
    path = FIX / (name + '.wav')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['fixture_sha256']
    rate, x = wavfile.read(path)
    assert rate == 48000
    return meta, np.asarray(x, float)


@pytest.mark.parametrize('name', ['record100-v0', 'record100-v1'])
def test_native_floor_crc_payload_and_bandwidth_discrimination(name):
    meta, x = native(name)
    assert rx.RECORDS[100].il_col == 0
    frame = rx.decode_over(x, 0, len(x), level=100)
    assert frame.crc_ok and frame.frame_bytes.hex() == meta['decoded_frame_hex']
    assert phy.vara_payload(frame.payload, caller='W9SSJ', body_len=10).hex() == meta['known_payload_hex']
    scores = {lv: (hits, count) for lv, hits, count, _ in rx.index_guard(x, (100, 0, 101, 102, 103))}
    assert scores[100] == (24, 24)
    assert all(hits < 16 for lv, (hits, _) in scores.items() if lv != 100)
    assert not rx.decode_over(x, 0, len(x), level=0).crc_ok


@pytest.mark.parametrize('name', ['record100-v0', 'record100-v1'])
def test_floor_encoder_matches_independent_native_body(name):
    meta, x = native(name)
    coded = tx.synth_frame(bytes.fromhex(meta['decoded_frame_hex']), level=100, over=None)
    g = meta['geometry']
    start = g['phase'] + g['live_indices'][2] * 2048
    errors = []
    for shift in range(-4, 5):
        # Native key-off tapers the final64 samples of the2048 grid.
        # Independent localization: level1/analysis/body-alignment.json.
        actual = x[start + shift:start + shift + len(coded)][:-64]
        expected = coded[:-64]
        gain = np.dot(actual, expected) / np.dot(expected, expected)
        errors.append(np.linalg.norm(actual - gain * expected) / np.linalg.norm(actual))
    assert min(errors) < .0001


@pytest.mark.parametrize('name', ['record100-v0', 'record100-v1'])
def test_native_floor_stream_delivers_once_after_peer_finishes(name):
    meta, x = native(name)
    io = _IO()
    hs = va.VaraStationHandshake(['W9SSJ'], io, bw='2750')
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'KC9GHZ'
    hs.state, hs.step, hs.turn = va.VaraState.CONNECTED, va._I_CONNECTED, va._TURN_PEER
    samples = np.concatenate((x, np.zeros(24000)))
    for at in range(0, len(samples), 4800):
        io.received = min(at + 4800, len(samples))
        hs.on_rx_stream(samples[at:at + 4800])
    assert io.payloads == [bytes.fromhex(meta['known_payload_hex'])], io.logs
    assert len(io.answers) == 1, io.logs
    end = int(np.flatnonzero(np.abs(x) > 1e-8)[-1] + 1)
    assert end <= io.answers[0] <= end + 24000, io.logs


def test_native_three_record_training_stream():
    meta = json.loads((FIX / 'training.json').read_text())
    state = meta['hypotheses']['3']['first_states'][0]
    before = ((state - 0xC39EC3) * pow(0x43FD43FD, -1, 1 << 24)) % (1 << 24)
    rnd = tx.VB6Rnd(before)
    for frame in meta['frames']:
        assert tx.preamble_bins(rnd, frame['record']) == frame['training_bins']
    assert rnd.state == meta['hypotheses']['3']['last_states'][0]
    assert meta['hypotheses']['4']['count'] == 0
