# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Owned recording paths use synthetic samples and a fake FLAC executable."""
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.core import capture_identity as ci, wav


def recording(tmp_path, *, continuous=False):
    capture = ci.allocate({'capture': tmp_path / 'capture'}, {'modem': 'capture'})
    writer = capture.claim('modem')
    return capture, ci.Recording.reserve(writer, 'stream.wav', clock_id='local-48000', continuous=continuous)


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
    from hfhost import recordings
    original = recordings.audio_info
    def info(path):
        path = recordings.stored(path)
        if path.read_bytes().startswith(b'fake-container'):
            import io
            with wave.open(io.BytesIO(path.read_bytes()[len(b'fake-container'):])) as src:
                return src.getframerate(), src.getnframes()
        return original(path)
    monkeypatch.setattr(recordings, 'audio_info', info)


@pytest.mark.parametrize('operation', ['capture', 'write', 'store'])
@pytest.mark.parametrize('matches', [False, True])
def test_owned_recording_accepts_plain_path_and_refuses_a_different_path(tmp_path, monkeypatch,
                                                                     operation, matches):
    _, owner = recording(tmp_path, continuous=operation == 'capture')
    fake_flac(monkeypatch)
    path = tmp_path / 'capture' / 'stream.wav'
    assert ci._absolute(path) == (owner.base or owner.audio.path)
    if not matches:
        path = path.with_name('different.wav')
    if operation == 'store':
        wav.write(owner.audio.path, np.zeros(2), 48000, ownership=owner)

    def perform():
        if operation == 'capture':
            recorder = wav.CaptureRecorder(path, rate=48000, ownership=owner)
            recorder.close()
        elif operation == 'write':
            wav.write(path, np.zeros(2), 48000, ownership=owner)
        else:
            assert 'kept' not in wav.store(path, ownership=owner)

    if matches:
        perform()
    else:
        with pytest.raises(ValueError, match='Recording ownership path differs'):
            perform()
        assert not path.exists()


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
    capture, owner = recording(tmp_path, continuous=True)
    fake_flac(monkeypatch)
    recorder = wav.CaptureRecorder(owner.base, rate=48000, source='fake', ownership=owner)
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
    [segment] = wav.segments(owner.base)
    assert side['wav'] == segment.name and segment.suffix == '.flac'
    assert not owner.base.exists()
    assert ci.validate_artifact(segment, clock_id='local-48000')['status'] == 'bound'
    assert capture.lifecycle()['capture_complete']


def test_failed_stream_drain_leaves_partial_sidecar_and_original_wav(tmp_path, monkeypatch):
    _, owner = recording(tmp_path, continuous=True)
    recorder = wav.CaptureRecorder(owner.base, rate=48000, ownership=owner)
    def fail():
        raise OSError('fake disk failure')
    monkeypatch.setattr(recorder, 'drain', fail)
    with pytest.raises(OSError):
        recorder.close()
    [segment] = wav.segments(owner.base)
    assert segment.exists() and not segment.with_suffix('.flac').exists()
    assert json.loads(owner.sidecar.artifact.path.read_text())['capture_state'] == 'partial'
    state=owner.audio.writer.capture.lifecycle()['writers']['modem']
    assert state['state']=='partial' and state['partial']==1


@pytest.mark.parametrize('failure', ['foreign-flac', 'roundtrip', 'no-encoder'])
def test_failed_conversion_keeps_the_original_and_leaves_no_partial(tmp_path, monkeypatch, failure):
    """Rig proof 11-05: with no `flac` on PATH each kept WAV had a 0-byte
    `.flac.<uuid>.partial` beside it."""
    _, owner = recording(tmp_path)
    wav.write(owner.audio.path, np.zeros(5), 48000, ownership=owner)
    original = owner.audio.path.read_bytes()
    if failure == 'no-encoder':
        monkeypatch.setattr(wav, '_flac_exe', lambda: None)
    else:
        fake_flac(monkeypatch, corrupt=failure == 'roundtrip')
    if failure == 'foreign-flac':
        owner.container.path.write_bytes(b'foreign')
    result = wav.store(owner.audio.path, ownership=owner)
    assert result['kept']
    assert owner.audio.path.read_bytes() == original
    if failure == 'foreign-flac':
        assert owner.container.path.read_bytes() == b'foreign'
    assert not list(owner.audio.path.parent.glob('.stream.flac.*.partial'))


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
    capture, owner = recording(tmp_path, continuous=True)
    recorder = wav.CaptureRecorder(owner.base, rate=48000, ownership=owner)
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


def test_owned_segments_keep_the_writer_clock_and_independent_hashes(tmp_path, monkeypatch):
    import threading
    capture, owner = recording(tmp_path, continuous=True)
    fake_flac(monkeypatch)
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    threads = []
    store = wav.store
    def tracked(path, **kwargs):
        threads.append(threading.get_ident())
        return store(path, **kwargs)
    monkeypatch.setattr(wav, 'store', tracked)
    samples = np.tile(np.array([0, .5, -.5, 1, -1], np.float32), 250)
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    recorder.push(samples[:1000])
    recorder.drain()
    current = recorder._out._wav
    assert len(recorder._out.paths) == 3
    assert current is not None
    capture.end()
    recorder.push(samples[1000:])
    recorder.close()
    recorder.close()
    parts = wav.segments(owner.base)
    assert len(parts) == 3 and all(p.suffix == '.flac' for p in parts)
    assert not owner.base.exists()
    assert threads == [threads[0]] * 3 and threads[0] != threading.get_ident()
    side = json.loads(owner.sidecar.artifact.path.read_text())
    assert side['segments'] == [p.name for p in parts]
    assert side['wav'] == parts[0].name
    assert side['samples'] == samples.size and side['segment_samples'] == 480
    assert side['capture_state'] == 'finalized'
    counts = []
    ids = []
    for part in parts:
        binding = ci.validate_artifact(part, clock_id='local-48000')
        assert binding['capture_id'] == capture.capture_id
        facts = ci.audio_metadata(part)
        assert facts['writer_id'] == owner.audio.writer.writer_id
        assert facts['capture_state'] == 'finalized'
        counts.append(facts['samples'])
        ids.append(binding['artifact_id'])
        with wave.open(wav.source(part)) as src:
            assert src.getnframes() == facts['samples']
    assert counts == [480, 480, 290] and len(set(ids)) == 3
    expected = np.clip(np.round(samples * 32768), -32768, 32767) / 32768
    assert np.array_equal(wav.read(owner.base), expected)
    assert capture.lifecycle()['capture_complete']


def test_a_foreign_owned_segment_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(wav.time, 'time', lambda: 0)
    _, owner = recording(tmp_path, continuous=True)
    occupied = owner.audio.path
    occupied.write_bytes(b'foreign')
    with pytest.raises(FileExistsError):
        wav.CaptureRecorder(owner.base, rate=48000, ownership=owner)
    assert occupied.read_bytes() == b'foreign'


@pytest.mark.parametrize('compress', [False, True])
def test_stream_reserves_real_segments_and_one_recording_sidecar(tmp_path, monkeypatch, compress):
    capture, owner = recording(tmp_path, continuous=True)
    fake_flac(monkeypatch)
    if not compress:
        monkeypatch.setattr(wav, 'store', lambda *args, **kwargs: {})
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    recorder.push(np.array([.25, -.5], np.float32))
    recorder.drain()
    capture.end()
    assert not capture.lifecycle()['capture_complete']
    recorder.close()
    state = capture.lifecycle()
    assert state['capture_complete'] and state['artifacts_complete']
    assert state['writers']['modem']['written'] == 1
    assert 'segmented' not in state['artifact_states'].values()
    reservations = [json.loads(p.read_text())['artifact_path']
                    for p in owner.base.parent.joinpath('.capture-artifacts').glob('*.json')]
    assert 'stream.wav' not in reservations and 'stream.flac' not in reservations
    assert list(owner.base.parent.glob('stream*.json')) == [owner.base.with_suffix('.json')]
    [segment] = wav.segments(owner.base)
    assert ci.validate_artifact(segment)['status'] == 'bound'
    facts = ci.audio_metadata(segment)
    assert facts['recording_artifact_id'] == owner.audio.record['artifact_id']
    assert facts['sha256']
    segment.unlink()
    assert not capture.lifecycle()['capture_complete']


@pytest.mark.parametrize('fault', ['partial', 'bytes', 'parent', 'clock', 'container', 'samples'])
def test_finalized_writer_cannot_certify_invalid_segment_metadata(tmp_path, monkeypatch, fault):
    capture, owner = recording(tmp_path, continuous=True)
    fake_flac(monkeypatch)
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    recorder.push(np.array([.25, -.5], np.float32))
    recorder.close()
    capture.end()
    assert capture.lifecycle()['capture_complete']
    [segment] = wav.segments(owner.base)
    side = json.loads(owner.sidecar.artifact.path.read_text())
    facts = side['segment_artifacts'][segment.with_suffix('.wav').name]
    if fault == 'bytes':
        segment.write_bytes(b'foreign')
    elif fault == 'samples':
        side['samples'] = 3
    else:
        key, value = {'partial': ('capture_state', 'partial'),
                      'parent': ('recording_artifact_id', 'foreign'),
                      'clock': ('clock_id', 'foreign'),
                      'container': ('container_artifact_id', 'foreign')}[fault]
        facts[key] = value
    owner.sidecar.artifact.path.write_text(json.dumps(side))
    if fault in ('partial', 'samples'):
        state = capture.lifecycle()
        assert state['claimed_writers_complete'] and not state['artifacts_complete']
        assert not state['capture_complete']
    else:
        with pytest.raises(ci.IdentityError):
            capture.lifecycle()


def test_owned_audio_queued_before_end_rolls_over_after_end(tmp_path, monkeypatch):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    capture, owner = recording(tmp_path, continuous=True)
    fake_flac(monkeypatch)
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    samples = np.arange(1000, dtype=np.float32) / 2000
    recorder.push(samples)
    capture.end()
    recorder.close()
    assert len(wav.segments(owner.base)) == 3
    assert np.array_equal(wav.read(owner.base), np.round(samples * 32768) / 32768)
    assert capture.lifecycle()['capture_complete']


def test_push_during_owned_rollover_keeps_the_segment_admitted(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    capture, owner = recording(tmp_path, continuous=True)
    fake_flac(monkeypatch)
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    entered, release = threading.Event(), threading.Event()
    reserve = ci.QueuedSegment.reserve
    failures = []
    def paused(ticket):
        entered.set()
        assert release.wait(3)
        return reserve(ticket)
    monkeypatch.setattr(ci.QueuedSegment, 'reserve', paused)
    recorder.push(np.zeros(481, np.float32))
    def drain():
        try:
            recorder.drain()
        except Exception as error:
            failures.append(error)
    thread = threading.Thread(target=drain)
    thread.start()
    try:
        assert entered.wait(3)
        recorder.push(np.ones(2, np.float32) / 2)
        assert recorder._push_error is None
        assert recorder._queued_samples == 483
        capture.end()
    finally:
        release.set()
        thread.join(3)
    recorder.close()
    assert not thread.is_alive() and not failures
    assert len(wav.read(owner.base)) == 483
    assert capture.lifecycle()['capture_complete']


def test_push_admission_failure_is_reported_by_drain_not_callback(tmp_path, monkeypatch):
    _, owner = recording(tmp_path, continuous=True)
    fake_flac(monkeypatch)
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    def fail(_):
        raise ci.IdentityError('synthetic admission failure')
    monkeypatch.setattr(recorder._out, 'queue_through', fail)
    recorder.push(np.zeros(3, np.float32))
    with pytest.raises(ci.IdentityError, match='synthetic admission failure'):
        recorder.close()
    assert recorder._out.samples == recorder._queued_samples == 3


def test_owned_store_publishes_valid_identity_before_removing_wav(tmp_path, monkeypatch):
    _, owner = recording(tmp_path)
    wav.write(owner.audio.path, np.array([.25, -.5]), 48000, ownership=owner)
    fake_flac(monkeypatch)
    unlink = Path.unlink
    observed = []
    def check(path, *args, **kwargs):
        if path == owner.audio.path:
            observed.append(ci.validate_artifact(owner.container.path)['status'])
            assert ci.validate_artifact(path)['status'] == 'bound'
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', check)
    assert 'kept' not in wav.store(owner.audio.path, ownership=owner)
    assert observed == ['bound']


def test_active_owned_recording_keeps_its_epoch_when_segment_zero_disappears(tmp_path, monkeypatch):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav, 'store', lambda path, **kwargs: {'wav': path.name})
    _, owner = recording(tmp_path, continuous=True)
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    try:
        recorder.push(np.zeros(481, np.float32))
        recorder.drain()
        recorder._out.paths[0].unlink()
        with pytest.raises(ValueError, match='missing segment'):
            wav.segments(owner.base)
    finally:
        recorder.close()


def test_owned_uncompressed_segment_keeps_the_storage_failure_in_its_journal(tmp_path, monkeypatch):
    capture, owner = recording(tmp_path, continuous=True)
    def unavailable(*args):
        raise OSError('synthetic missing compressor')
    monkeypatch.setattr(wav, '_flac', unavailable)
    recorder = wav.CaptureRecorder(owner.base, rate=8, ownership=owner)
    recorder.push(np.zeros(3, np.float32))
    recorder.close()
    capture.end()
    [part] = wav.segments(owner.base)
    assert part.suffix == '.wav'
    assert ci.audio_metadata(part)['kept'] == 'synthetic missing compressor'
    assert capture.lifecycle()['capture_complete']


def test_owned_validation_retries_when_storage_replaces_the_wav(tmp_path, monkeypatch):
    _, owner = recording(tmp_path)
    wav.write(owner.audio.path, np.array([.25, -.5]), 48000, ownership=owner)
    fake_flac(monkeypatch)
    opening = Path.open
    replaced = False
    def open_replaced(path, *args, **kwargs):
        nonlocal replaced
        if path == owner.audio.path and args == ('rb',) and not replaced:
            replaced = True
            assert 'kept' not in wav.store(path, ownership=owner)
        return opening(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_replaced)
    binding = ci.validate_artifact(owner.audio.path)
    assert replaced and binding['artifact_id'] == owner.container.record['artifact_id']
