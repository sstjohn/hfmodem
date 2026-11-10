# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Selected replay allocates PCM only for available bounded source intervals."""
import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_stage5_support import (
    RATE, START, STOP, entry, reader, save_selection, wave_file, write_json,
)


@pytest.mark.parametrize('parts', [3, 31])
def test_wav_family_read_bounds_before_float_allocation(tmp_path, monkeypatch, parts):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    path = tmp_path / 'receive.wav'
    writer = wav.SegmentedWriter(path, rate=RATE, closed=lambda _: None)
    values = np.arange(parts * 60 * RATE, dtype='<i2')
    writer.writeframes(values.tobytes())
    writer.close()
    original = wav._read_one
    def bounded_read(part, *args, **kwargs):
        result = original(part, *args, **kwargs)
        assert len(result[1]) <= 16, 'family reader converted a whole segment before selecting samples'
        return result
    monkeypatch.setattr(wav, '_read_one', bounded_read)
    lo, hi = 60 * RATE - 8, 60 * RATE + 8
    selected = wav.read(path, start=lo, stop=hi)
    np.testing.assert_array_equal(selected, values[lo:hi] / 32768)


def test_wav_bounded_read_preserves_absolute_origin(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    from hfmodem.tests.station.recording_support import push, bounded
    push(rec, lane, START, START + 48)
    bounded(rec.close)
    selected = wav.read(root / lane.row['source'], start=START + 4, stop=START + 12)
    np.testing.assert_array_equal(selected, np.arange(START + 4, START + 12) / 32768)


def test_wav_read_across_gap_is_explicit_instead_of_compressed(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    from hfmodem.tests.station.recording_support import push, bounded
    push(rec, lane, START, START + 8)
    push(rec, lane, START + 16, START + 24)
    bounded(rec.close)
    with pytest.raises(ValueError, match='gap|missing|unavailable'):
        wav.read(root / lane.row['source'], start=START, stop=START + 24)
    got = wav.read(root / lane.row['source'], start=START + 16, stop=START + 24)
    np.testing.assert_array_equal(got, np.arange(START + 16, START + 24) / 32768)


@pytest.mark.parametrize('extension', ['wav', 'float'])
def test_legacy_selected_reads_keep_pcm_and_float_scale(tmp_path, extension):
    path = wave_file(tmp_path / 'legacy.wav', count=24)
    expected = np.arange(24) / 32768
    if extension == 'float':
        from scipy.io import wavfile
        wavfile.write(path, RATE, expected.astype(np.float32))
    np.testing.assert_allclose(wav.read(path, start=7, stop=12), expected[7:12], rtol=1e-6)


def test_vara_limit_is_checked_before_any_pcm_read(tmp_path, monkeypatch):
    view = reader('logbook_view.vara_view')
    path = wave_file(tmp_path / 'long.wav', count=1801, rate=1)
    monkeypatch.setattr(wav, 'read', lambda *a, **k: pytest.fail('duration limit followed PCM allocation'))
    with pytest.raises(ValueError, match='limit|minutes'):
        view.load_audio(path)


@pytest.mark.parametrize('total_seconds', [4, 40])
def test_armsource_detection_and_measurement_windows_are_bounded(tmp_path, monkeypatch, total_seconds):
    arm = reader('consolelib.armsource')
    root = tmp_path / 'arms/vara-c2'
    root.mkdir(parents=True)
    write_json(root / 'provenance.json', {'arm': root.name, 'dial_hz': 7105000})
    (root / 'arm.log').write_text('')
    wave_file(root / 'tap.wav', count=total_seconds * 48000, rate=48000)
    monkeypatch.setattr(arm.rigsource, 'station_config', lambda *a: None)
    def detect(pcm):
        assert len(pcm) <= 2 * 48000 * 2, 'ArmSource joined duration-sized detection PCM'
        return []
    def measure(audio, keyed, *args):
        assert len(audio) <= 2 * 48000, 'ArmSource allocated duration-sized measurement floats'
        return []
    monkeypatch.setattr(arm, 'detect', detect)
    monkeypatch.setattr(arm, 'measure', measure)
    with arm.ArmSource(root, realtime=False) as source:
        assert sum(len(pcm) // 2 for _, pcm in source.frames()) == total_seconds * 48000


def test_armsource_uses_selected_tap_log_mapping_and_provenance(tmp_path, monkeypatch):
    arm = reader('consolelib.armsource')
    selected = entry(tmp_path, protocol='vara', source='working/run/tap.wav')
    path = save_selection(tmp_path, selected)
    write_json(path.parent / 'provenance.json', {'arm': 'vara-c2', 'dial_hz': 7105000,
        'recording': selected['recording'], 'local_role': 'called'})
    monkeypatch.setattr(arm.rigsource, 'station_config', lambda *a: None)
    def fitting(*args):
        pytest.fail('selected tap clocks must use context anchors rather than a whole-arm fit')
    monkeypatch.setattr(arm, 'cross_check', fitting)
    with arm.ArmSource(path.parent, realtime=False) as source:
        assert source.mycall == 'N0CALL'
        assert source.logged == [(0., 1.)]
        frames = list(source.frames())
        assert frames[0][0] == START * 48000 // RATE
        assert sum(len(pcm) // 2 for _, pcm in frames) == (STOP - START) * 48000 // RATE


def test_absent_tap_does_not_substitute_session_rx(tmp_path):
    arm = reader('consolelib.armsource')
    from hfhost.audio import AudioError
    root = tmp_path / 'arms/vara-c2'
    write_json(root / 'provenance.json', {'arm': root.name, 'dial_hz': 7105000})
    wave_file(root / 'stream.wav', rate=48000)
    with pytest.raises(AudioError, match='tap|unavailable'):
        arm.ArmSource(root, realtime=False)


def test_armsource_measurement_never_allocates_a_duration_sized_float_array(tmp_path, monkeypatch):
    arm = reader('consolelib.armsource')
    root = tmp_path / 'arms/vara-c2'
    write_json(root / 'provenance.json', {'arm': root.name, 'dial_hz': 7105000})
    (root / 'arm.log').write_text('')
    wave_file(root / 'tap.wav', count=8 * 48000, rate=48000)
    monkeypatch.setattr(arm.rigsource, 'station_config', lambda *a: None)
    monkeypatch.setattr(arm, 'detect', lambda *a: [])
    def measure(audio, *args):
        assert len(audio) <= 2 * 48000, 'ArmSource materialized whole-tap float measurement audio'
        return []
    monkeypatch.setattr(arm, 'measure', measure)
    with arm.ArmSource(root, realtime=False):
        pass


def test_armsource_keeps_detector_boundary_across_tiny_segments(tmp_path, monkeypatch):
    arm = reader('consolelib.armsource')
    root = tmp_path / 'arms/vara-c2'
    write_json(root / 'provenance.json', {'arm': root.name, 'dial_hz': 7105000})
    (root / 'arm.log').write_text('')
    rate = 48000
    rng = np.random.default_rng(5)
    samples = rng.normal(0, .02, 4 * rate)
    t = np.arange(len(samples)) / rate
    muted = (t >= 1.8) & (t < 2.4)
    samples[muted] *= .01
    samples[muted] += .2 * np.sin(2 * np.pi * 1500 * t[muted])
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', rate)
    writer = wav.SegmentedWriter(root / 'tap.wav', rate=rate, closed=lambda _: None, sample_clock=True)
    writer.set_position(0)
    writer.writeframes(np.round(samples * 32768).astype('<i2').tobytes())
    writer.close()
    monkeypatch.setattr(arm.rigsource, 'station_config', lambda *a: None)
    detector = arm.detect
    sizes = []
    def detect(pcm):
        sizes.append(len(pcm))
        assert len(pcm) <= 2 * rate * 2, 'detection must keep bounded overlap across segment boundaries'
        return detector(pcm)
    monkeypatch.setattr(arm, 'detect', detect)
    with arm.ArmSource(root, realtime=False) as source:
        assert len(source.spans) == 1
        lo, hi = source.spans[0]
        assert lo == pytest.approx(1.8, abs=2 * arm.STEP_S)
        assert hi == pytest.approx(2.4, abs=2 * arm.STEP_S)
        [record] = source.records()
        assert record['centre_hz'] == pytest.approx(1500, abs=2)
    assert sizes
