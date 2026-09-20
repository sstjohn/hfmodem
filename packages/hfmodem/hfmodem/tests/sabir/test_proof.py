# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Independent-recording decode and negative controls; no radio or network."""
import json
from hashlib import sha256

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.sabir import offair, onair, proof
from hfmodem.sabir.arq.fsm import ArqConfig
from hfmodem.sabir.arq.modem import LinkModem
from hfmodem.sabir.frame.datagram import fragment_object
from hfmodem.sabir.phy.rate import FS


@pytest.fixture
def prepared(tmp_path):
    manifest = proof.prepare(tmp_path / 'reference', callsign='W9SSJ',
                             profiles=['workhorse'], payload_bytes=128, seed=19)
    data = json.loads(manifest.read_text())
    return manifest, data


def test_all_ofdm_profiles_use_blind_normal_decode(tmp_path):
    manifest = proof.prepare(tmp_path / 'all', callsign='W9SSJ',
                             profiles=proof.PROOF_PROFILES, payload_bytes=64, seed=20)
    data = json.loads(manifest.read_text())
    report = proof.verify(manifest, manifest.parent / data['waveform'])
    assert report['ok'] and report['byte_exact']
    assert len(report['trials']) == len(proof.PROOF_PROFILES)
    assert all(t['received_copies'] == 1 for t in report['trials'])
    assert report['capture_is_reference']
    assert report['radio_proof'] is False


def test_kiwi_rate_int16_offset_drift_noise_capture(prepared, tmp_path):
    manifest, data = prepared
    clean = offair.wav_read(str(manifest.parent / data['waveform']))
    # An independent recording: unknown delay, gain, CFO, drift, noise, and a
    # 12 kHz integer sound file. The verifier gets no packet timing or CFO hint.
    rng = np.random.default_rng(119)
    t = np.arange(len(clean)) / FS
    audio = (offair.to_analytic(clean) * np.exp(2j * np.pi * (17 * t + 0.04 * t * t))).real
    audio = np.concatenate([np.zeros(int(0.731 * FS)), audio, np.zeros(FS)])
    audio = .5 * audio + rng.normal(0, .002, len(audio))
    recording = resample_poly(audio, 1, 4)
    path = tmp_path / 'receiver.wav'
    wavfile.write(path, 12000, np.rint(np.clip(recording, -1, 1) * 32767).astype(np.int16))
    result = proof.verify(manifest, path)
    assert result['ok'] and result['capture_sample_rate_hz'] == 12000
    assert result['capture_sha256'] == sha256(path.read_bytes()).hexdigest()
    assert not result['capture_is_reference'] and not result['radio_proof']
    assert result['observed'][0]['profile_id'] == 4


@pytest.mark.parametrize('kind', ['noise', 'missing_header', 'truncated_body'])
def test_negative_recordings_do_not_pass(prepared, tmp_path, kind):
    manifest, data = prepared
    clean = offair.wav_read(str(manifest.parent / data['waveform']))
    if kind == 'noise':
        audio = np.random.default_rng(5).normal(0, .05, len(clean))
    elif kind == 'missing_header':
        audio = np.concatenate([np.zeros(FS), clean[int(6.5 * FS):]])
    else:
        audio = clean[:int(6.2 * FS)]
    capture = tmp_path / f'{kind}.wav'
    wavfile.write(capture, FS, audio.astype(np.float32))
    assert not proof.verify(manifest, capture)['ok']


def test_same_message_id_with_wrong_decoded_payload_fails(prepared, tmp_path):
    manifest, data = prepared
    mid = bytes.fromhex(data['trials'][0]['message_id'])
    fragment = fragment_object(b'wrong payload' * 8, 'W9SSJ', message_id=mid,
                               fragment_bytes=256, parity=False)[0]
    modem = LinkModem(ArqConfig())
    modem.send_datagram(fragment.pack(), 'workhorse')
    capture = tmp_path / 'wrong-payload.wav'
    offair.wav_write(str(capture), offair._lead_silence(offair.to_real(modem.outbox[0]), 1.0))
    report = proof.verify(manifest, capture)
    assert not report['ok']
    assert len(report['observed']) == 1
    assert report['observed'][0]['message_id'] == mid.hex()


def test_repeats_have_fresh_distinct_ids(tmp_path):
    path = proof.prepare(tmp_path / 'repeat', profiles=['workhorse'] * 2,
                         callsign='W9SSJ', payload_bytes=64, seed=3)
    data = json.loads(path.read_text())
    assert data['trials'][0]['message_id'] != data['trials'][1]['message_id']
    assert proof.verify(path, path.parent / data['waveform'])['ok']


def test_manifest_checks_station_source_and_waveform(prepared, monkeypatch):
    path, data = prepared
    with pytest.raises(ValueError, match='callsign'):
        proof.load_burst(path, callsign='W1AW')
    original = proof.source_hashes()
    monkeypatch.setattr(proof, 'source_hashes', lambda: {})
    with pytest.raises(ValueError, match='sources changed'):
        proof.load_burst(path)
    monkeypatch.setattr(proof, 'source_hashes', lambda: original)
    waveform = path.parent / data['waveform']
    waveform.write_bytes(waveform.read_bytes() + b'x')
    with pytest.raises(ValueError, match='waveform hash'):
        proof.load_burst(path)


def test_manifest_expected_profile_must_match_decoded_header(prepared):
    path, data = prepared
    data['trials'][0]['profile'] = 'fast'
    data['trials'][0]['profile_id'] = 6
    path.write_text(json.dumps(data))
    assert not proof.verify(path, path.parent / data['waveform'])['ok']


def test_onair_rehearsal_uses_identified_burst_without_hardware(prepared, monkeypatch, capsys):
    path, data = prepared
    def forbidden(*args, **kwargs):
        raise AssertionError('rehearsal opened a radio')
    monkeypatch.setattr(onair.radio, 'Rig', forbidden)
    assert onair.main(['--proof-manifest', str(path), '--mycall', 'W9SSJ',
                       '--channel', '14100000']) == 0
    assert 'NOT ARMED' in capsys.readouterr().out
    burst, checked = proof.load_burst(path, 'W9SSJ')
    assert burst.seconds == checked['transmit_seconds']
    assert burst.span_hz[0] <= 1500 <= burst.span_hz[1]


def test_cannot_reuse_manifest_directory(prepared):
    path, _ = prepared
    with pytest.raises(ValueError, match='already exists'):
        proof.prepare(path.parent, callsign='W9SSJ')



def test_updated_decoder_can_revisit_a_recording(prepared, monkeypatch):
    path, data = prepared
    monkeypatch.setattr(proof, 'source_hashes', lambda: {'changed.py': 'changed'})
    report = proof.verify(path, path.parent / data['waveform'])
    assert report['ok'] and report['source_matches'] is False
    assert report['transmit_source_hashes'] == data['source_hashes']
    assert report['decode_source_hashes'] == {'changed.py': 'changed'}


def test_empty_manifest_trials_rejected(prepared):
    path, data = prepared
    data['trials'] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='1..8 trials'):
        proof.verify(path, path.parent / data['waveform'])
