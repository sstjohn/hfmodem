# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Owned recording paths use synthetic samples and a fake FLAC executable."""
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.core import capture_identity as ci, wav


def recording(tmp_path):
    capture = ci.allocate({'capture': tmp_path / 'capture'}, {'modem': 'capture'})
    writer = capture.claim('modem')
    return capture, ci.Recording.reserve(writer, 'stream.wav', clock_id='local-48000')


def fake_flac(monkeypatch, *, corrupt=False):
    def convert(*args):
        destination = Path(args[args.index('-o') + 1])
        source = Path(args[-1])
        content = source.read_bytes()
        if '-d' in args:
            assert content.startswith(b'fake-container')
            destination.write_bytes(content[len(b'fake-container'):] + (b'wrong' if corrupt else b''))
        else:
            destination.write_bytes(b'fake-container' + content)
    monkeypatch.setattr(wav, '_flac', convert)


def test_finite_owned_wav_preserves_levels_and_final_hash(tmp_path):
    _, owner = recording(tmp_path)
    samples = np.array([0, 0.5, -0.5, 1.0, -1.0])
    side = wav.write(owner.audio.path, samples, 48000, ownership=owner,
                     sample_start=123, sample_end=128)
    with wave.open(str(owner.audio.path)) as source:
        assert (source.getframerate(), source.getnframes(), source.getnchannels()) == (48000, 5, 1)
        assert np.array_equal(np.frombuffer(source.readframes(5), dtype='<i2'), [0, 16384, -16384, 32767, -32768])
    assert side['normalised_on_write'] is False
    assert (side['sample_start'], side['sample_end']) == (123, 128)
    assert side['capture_state'] == 'finalized'
    assert ci.validate_artifact(owner.audio.path, clock_id='local-48000')['status'] == 'bound'
    journal = owner.sidecar.artifact.path.parent / '.stream.json.versions'
    assert json.loads((journal / '00000000.json').read_text())['capture_state'] == 'partial'
    assert json.loads((journal / '00000001.json').read_text())['capture_state'] == 'finalized'


def test_owned_and_legacy_finite_pcm_bytes_and_measurements_are_exact(tmp_path):
    _, owner = recording(tmp_path)
    samples = np.array([0, 0.5, -0.5, 1, -1, 0.1234567])
    legacy = tmp_path / 'legacy.wav'
    old = wav.write(legacy, samples, 12000, channels=2, sample_end=777)
    new = wav.write(owner.audio.path, samples, 12000, channels=2, sample_end=777, ownership=owner)
    assert legacy.read_bytes() == owner.audio.path.read_bytes()
    assert {key: new[key] for key in old if key != 'wav'} == {key: value for key, value in old.items() if key != 'wav'}
    assert 'capture_id' not in json.loads(legacy.with_suffix('.json').read_text())


def test_foreign_finite_destination_never_replaced(tmp_path):
    _, owner = recording(tmp_path)
    owner.audio.path.write_bytes(b'foreign')
    with pytest.raises(FileExistsError):
        wav.write(owner.audio.path, np.zeros(4), 48000, ownership=owner)
    assert owner.audio.path.read_bytes() == b'foreign'
    assert not list(owner.audio.path.parent.glob('.stream.wav.*.partial'))
    assert not owner.sidecar.artifact.path.exists()


def test_stream_is_ongoing_then_finalized_without_new_reservations_after_end(tmp_path, monkeypatch):
    capture, owner = recording(tmp_path)
    fake_flac(monkeypatch)
    recorder = wav.CaptureRecorder(owner.audio.path, rate=48000, source='fake', ownership=owner)
    assert json.loads(owner.sidecar.artifact.path.read_text())['capture_state'] == 'ongoing'
    recorder.push(np.array([0.5, -0.5], dtype=np.float32))
    recorder.drain()
    capture.end()
    recorder.push(np.array([1, -1], dtype=np.float32))
    recorder.close()
    recorder.close()
    side = json.loads(owner.sidecar.artifact.path.read_text())
    assert side['samples'] == 4 and side['capture_state'] == 'finalized'
    assert side['normalised_on_write'] is False
    assert side['wav'] == 'stream.flac'
    assert not owner.audio.path.exists()
    assert ci.validate_artifact(owner.container.path, clock_id='local-48000')['status'] == 'bound'
    assert capture.lifecycle()['capture_complete']


def test_failed_stream_drain_leaves_partial_sidecar_and_original_wav(tmp_path, monkeypatch):
    _, owner = recording(tmp_path)
    recorder = wav.CaptureRecorder(owner.audio.path, rate=48000, ownership=owner)
    def fail():
        raise OSError('fake disk failure')
    monkeypatch.setattr(recorder, 'drain', fail)
    with pytest.raises(OSError):
        recorder.close()
    assert owner.audio.path.exists() and not owner.container.path.exists()
    assert json.loads(owner.sidecar.artifact.path.read_text())['capture_state'] == 'partial'
    state=owner.audio.writer.capture.lifecycle()['writers']['modem']
    assert state['state']=='partial' and state['partial']==1


@pytest.mark.parametrize('failure', ['foreign-flac', 'roundtrip'])
def test_failed_conversion_preserves_original_and_partial_container(tmp_path, monkeypatch, failure):
    _, owner = recording(tmp_path)
    wav.write(owner.audio.path, np.zeros(5), 48000, ownership=owner)
    original = owner.audio.path.read_bytes()
    fake_flac(monkeypatch, corrupt=failure == 'roundtrip')
    if failure == 'foreign-flac':
        owner.container.path.write_bytes(b'foreign')
    result = wav.store(owner.audio.path, ownership=owner)
    assert result['kept']
    assert owner.audio.path.read_bytes() == original
    if failure == 'foreign-flac':
        assert owner.container.path.read_bytes() == b'foreign'
    assert list(owner.audio.path.parent.glob('.stream.flac.*.partial'))


def test_owned_encoding_keeps_original_wav_hash_and_validates_container(tmp_path, monkeypatch):
    _, owner = recording(tmp_path)
    side = wav.write(owner.audio.path, np.array([0.25, -0.25]), 48000, ownership=owner)
    original_sha = side['sha256']
    fake_flac(monkeypatch)
    result = wav.store(owner.audio.path, ownership=owner)
    owner.sidecar.update({key: value for key, value in side.items()
                          if key not in owner.audio.record and key not in ['metadata_version', 'previous_sha256']}
                         | result)
    assert result['sha256'] == original_sha
    assert result['container_sha256'] != original_sha
    assert ci.validate_artifact(owner.container.path)['status'] == 'bound'
    owner.container.path.write_bytes(b'foreign')
    with pytest.raises(ci.IdentityError):
        ci.validate_artifact(owner.container.path)


def test_failed_drain_and_journal_preserve_original_failure(tmp_path, monkeypatch):
    capture, owner = recording(tmp_path)
    recorder = wav.CaptureRecorder(owner.audio.path, rate=48000, ownership=owner)
    original = OSError('synthetic drain failure')
    def fail_drain():
        raise original
    def fail_journal(self, **kwargs):
        raise RuntimeError('synthetic journal failure')
    monkeypatch.setattr(recorder, 'drain', fail_drain)
    monkeypatch.setattr(ci.Writer, 'update_status', fail_journal)
    with pytest.raises(OSError) as caught:
        recorder.close()
    assert caught.value is original
    assert owner.audio.writer._status['failed']
    assert owner.audio.writer._status['journal_error'] == 'RuntimeError'
    capture.end()
    assert not capture.lifecycle()['capture_complete']
