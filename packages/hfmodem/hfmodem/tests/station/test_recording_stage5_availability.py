# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Policy indexes preserve expected coverage independently of present files."""
from pathlib import Path

import pytest

from hfhost import recordings
from hfmodem.core import wav
from hfmodem.core.capture_identity import IdentityError
from hfmodem.tests.station.recording_stage5_support import (
    RATE, reader, recorded, wave_file, write_json,
)


@pytest.mark.parametrize('missing', ['first', 'middle', 'last', 'all'])
def test_indexed_expected_segments_survive_moves_and_restore(sessions, tmp_path, monkeypatch, missing):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4 * 48000)
    _, root, saved = recorded(sessions)
    lane = saved['lanes']['rx']
    logical = root / lane['source']
    parts = [root / d['source'] for d in saved['segments']]
    assert len(parts) >= 2
    chosen = {'first': parts[:1], 'middle': parts[len(parts)//2:len(parts)//2+1],
              'last': parts[-1:], 'all': parts}[missing]
    moved = []
    original_pairs = {item: item.read_bytes() for part in chosen
                      for item in (part, part.with_suffix('.json')) if item.is_file()}
    for part in original_pairs:
        target = tmp_path / ('away-' + part.name)
        part.rename(target)
        moved.append((part, target))
    try:
        indexed, capacity = recordings.indexed(logical)
        assert len(indexed) == len(parts)
        assert list(indexed.values()) == parts, 'moves must retain original paths and indices'
        status = getattr(recordings, 'status', None)
        assert callable(status), 'hfhost.recordings.status must expose policy coverage and unavailable intervals'
        result = status(logical)
        assert result['state'] == 'unavailable'
        assert result['actual'] == [0, 96]
        assert result['expected_samples'] == 96
        assert result['available_samples'] < 96
        assert {item['source'] for item in result['segments']} == {p.relative_to(root).as_posix() for p in parts}
    finally:
        for part, target in moved:
            target.rename(part)
    assert recordings.available(logical)
    assert {part: part.read_bytes() for part in original_pairs} == original_pairs


@pytest.mark.parametrize('policy,lane_state,expected', [
    ('off', 'ongoing', 'disabled'), ('contacts', 'ongoing', 'contact-only'),
    ('everything', 'ongoing', 'pending'), ('everything', 'storing', 'pending'), ('everything', 'finalized', 'available'),
    ('everything', 'disabled', 'unavailable'), ('everything', 'failed', 'unavailable'),
])
def test_policy_states_do_not_treat_disabled_or_failed_lanes_as_empty_success(tmp_path, policy, lane_state, expected):
    path = wave_file(tmp_path / 'receive.wav')
    write_json(tmp_path / 'recording.json', {
        'policy': policy, 'lanes': {'rx': {'source': 'receive.wav', 'rate': RATE,
            'clock': 'card', 'source_id': 's', 'index_id': 'i', 'state': lane_state,
            'origin': 0, 'position': 96, 'gaps': []}}, 'contacts': {},
        'segments': [] if policy != 'everything' else [{'source': 'receive.wav',
            'lane': 'rx', 'rate': RATE, 'source_id': 's', 'index_id': 'i',
            'actual': [0, 96], 'samples': 96, 'state': lane_state}]})
    status = getattr(recordings, 'status', None)
    assert callable(status), 'hfhost.recordings.status must read recording.json lane policy and state'
    assert status(path)['state'] == expected


def test_audio_duration_is_unknown_for_missing_legacy_length(tmp_path):
    audio = reader('logbook_view.audio')
    assert audio.duration_s(tmp_path / 'missing.wav', {}) is None


def test_audio_on_disk_isolates_identity_failure(tmp_path, monkeypatch):
    audio = reader('logbook_view.audio')
    identity = reader('hfmodem.core.capture_identity')
    bad, healthy = wave_file(tmp_path / 'bad.wav'), wave_file(tmp_path / 'healthy.wav')
    def validate(path):
        if path == bad:
            raise IdentityError('different source identity')
        return {'status': 'legacy'}
    monkeypatch.setattr(identity, 'validate_artifact', validate)
    assert audio.on_disk(bad) is None
    assert audio.on_disk(healthy) == healthy


@pytest.mark.parametrize('point', ['discovery', 'stat', 'open'])
def test_cli_multi_root_discovery_keeps_healthy_roots(tmp_path, monkeypatch, point):
    api = reader('tools.rehear')
    broken, healthy = tmp_path / 'broken', tmp_path / 'healthy'
    wave_file(broken / 'missing.wav')
    wanted = wave_file(healthy / 'good.wav')
    if point == 'discovery':
        original = Path.rglob
        def scan(path, pattern):
            if path == broken:
                raise FileNotFoundError(str(path))
            return original(path, pattern)
        monkeypatch.setattr(Path, 'rglob', scan)
    elif point == 'stat':
        original = wav.recording_base
        def base(path):
            if Path(path).is_relative_to(broken):
                raise IdentityError('moved during validation')
            return original(path)
        monkeypatch.setattr(wav, 'recording_base', base)
    else:
        (broken / 'missing.wav').unlink()
    found = api.recordings([broken, healthy])
    assert wanted in found


def test_explicit_missing_family_does_not_abort_next_cli_root(sessions, tmp_path, monkeypatch):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4 * 48000)
    _, root, saved = recorded(sessions)
    for d in saved['segments']:
        (root / d['source']).unlink()
    healthy = wave_file(tmp_path / 'healthy.wav')
    api = reader('tools.rehear')
    assert healthy in api.recordings([root / saved['lanes']['rx']['source'], healthy])


def test_open_retry_preserves_wav_to_flac_replacement(tmp_path, monkeypatch):
    path = wave_file(tmp_path / 'receive.wav')
    replacement = path.with_suffix('.flac')
    original = Path.open
    races = []
    def opening(candidate, *args, **kwargs):
        if candidate == path and args == ('rb',) and not races:
            races.append(candidate)
            path.rename(replacement)
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opening)
    with recordings.open_segment(path) as stream:
        assert stream.read(4) == b'RIFF'
        assert Path(stream.name) == replacement
    assert races == [path]


def test_writer_construction_failure_is_unavailable_through_host_index(sessions, monkeypatch):
    from hfmodem.core import recording
    def fail(*args):
        raise OSError('synthetic writer construction failure')
    monkeypatch.setattr(recording.RecordingSession, '_writer', fail)
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    assert lane.row['state'] == 'failed'
    status = getattr(recordings, 'status', None)
    assert callable(status), 'failed lanes require unavailable policy status even without a first segment'
    assert status(root / lane.row['source'])['state'] == 'unavailable'
