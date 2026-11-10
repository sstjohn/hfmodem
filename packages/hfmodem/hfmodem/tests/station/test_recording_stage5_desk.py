# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Logbook text survives unavailable audio and selected references remain bounded."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.core.capture_identity import IdentityError
from hfmodem.tests.station.recording_stage5_support import (
    START, STOP, entry, reader, save_selection, wave_file, write_json,
)


def book_and_arm(root, monkeypatch, *, policy='contacts'):
    face = reader('desk.face')
    selected = entry(root, protocol='vara')
    selected['recording']['policy'] = policy
    path = root / 'working/run/arms/c2/provenance.json'
    facts = dict(arm='c2', call='W0QXR', protocol='vara', band_m=40, dial_hz=7105000,
        started_utc=selected['started_utc'], ended_utc=selected['ended_utc'],
        verdict={'reached': ['keyed', 'connect']}, adjustments={'bw_hz': 2750},
        recording=selected['recording'], local_role='called')
    write_json(path, facts)
    (path.parent / 'arm.log').write_text('Contact with W0QXR\n')
    book = face.Logbook(root, station={'mycall': 'N0CALL', 'grid': 'AA00'})
    monkeypatch.setattr(book, 'unconverted', lambda: {})
    return face, book, path, selected


@pytest.mark.parametrize('policy,state', [('off', 'disabled'), ('contacts', 'contact-only')])
def test_desk_keeps_contact_text_and_adif_with_policy_state(tmp_path, monkeypatch, policy, state):
    face, book, path, selected = book_and_arm(tmp_path, monkeypatch, policy=policy)
    [row] = book.entries()['entries']
    assert row['call'] == 'W0QXR' and row['transcript']
    assert face.read_adif(face.adif([row], book.station))[0]['CALL'] == 'W0QXR'
    assert row.get('audio_state') == state, 'desk must expose intentional policy absence'
    assert row['audio'] == [] and row['viewer'] is None


def test_desk_uses_contact_reference_instead_of_transcript_filename(tmp_path, monkeypatch):
    _, book, path, selected = book_and_arm(tmp_path, monkeypatch)
    save_selection(tmp_path, selected)
    [row] = book.entries()['entries']
    assert len(row['audio']) == 1, 'contact lane references must appear without a filename mention'
    audio = row['audio'][0]
    assert audio['selection']['actual'] == [START, STOP]
    assert audio['duration_s'] == 4
    assert row['recording']['context']['id'] == 'context-c2'


@pytest.mark.parametrize('failure', ['missing', 'identity'])
def test_log_json_keeps_healthy_sources_and_visuals_when_one_audio_is_unavailable(tmp_path, monkeypatch, failure):
    face, book, path, selected = book_and_arm(tmp_path, monkeypatch)
    healthy = wave_file(path.parent / 'tap.wav')
    tap = dict(selected['recording']['lanes']['rx'], lane='tap', source=str(healthy.relative_to(tmp_path)))
    selected['recording']['lanes']['tap'] = tap
    write_json(healthy.with_suffix('.json'), dict(start_stream_sample=START,
        end_stream_sample=START+96, samples=96, source_id=tap['source_id'], index_id=tap['index_id']))
    bad = wave_file(tmp_path / 'logs/onair/other.wav')
    ref = selected['recording']['lanes']['rx']
    ref.update(source=str(bad.relative_to(tmp_path)), source_id='bad-source', index_id='bad-index')
    write_json(bad.with_suffix('.json'), dict(start_stream_sample=START,
        end_stream_sample=START+96, samples=96, source_id='bad-source', index_id='bad-index'))
    facts = json.loads(path.read_text())
    facts['recording'] = selected['recording']
    write_json(path, facts)
    (path.parent / 'arm.log').write_text(f'session stream: {bad}\n')
    identity = reader('hfmodem.core.capture_identity')
    original = identity.validate_artifact
    reached = []
    def validate(candidate, *args, **kwargs):
        if candidate == bad:
            reached.append('identity')
            raise IdentityError('wrong source')
        return original(candidate, *args, **kwargs)
    if failure == 'identity':
        monkeypatch.setattr(identity, 'validate_artifact', validate)
    else:
        bad.unlink()
        recordings = reader('hfhost.recordings')
        original_info = recordings.audio_info
        def audio_info(candidate, *args, **kwargs):
            if candidate == bad:
                reached.append('missing')
                assert not candidate.exists()
            return original_info(candidate, *args, **kwargs)
        monkeypatch.setattr(recordings, 'audio_info', audio_info)
    request = SimpleNamespace(path='/log.json', wfile=io.BytesIO(), command='GET',
        send_response=lambda *a: None, send_header=lambda *a: None, end_headers=lambda: None)
    face.Handler(request, book).do_GET()
    assert failure in reached
    [row] = json.loads(request.wfile.getvalue())['entries']
    assert row['call'] == 'W0QXR' and row['viewer']
    assert any(a['kind'] == 'rig tap' and a['url'] == '/file/' + healthy.relative_to(tmp_path).as_posix() for a in row['audio'])
    assert any(a.get('state') == 'unavailable' for a in row['audio']), 'named missing sources must remain visible'


def test_served_path_rejects_identity_before_http_headers(tmp_path, monkeypatch):
    face, book, path, _ = book_and_arm(tmp_path, monkeypatch)
    audio = wave_file(path.parent / 'tap.wav')
    identity = reader('hfmodem.core.capture_identity')
    monkeypatch.setattr(identity, 'validate_artifact',
        lambda *a, **k: (_ for _ in ()).throw(IdentityError('mismatched source')))
    responses = []
    request = SimpleNamespace(path='/file/' + str(audio.relative_to(tmp_path)),
        wfile=io.BytesIO(), command='GET', headers={},
        send_response=lambda code: responses.append(code), send_header=lambda *a: None,
        end_headers=lambda: None, send_error=lambda code, *a: responses.append(code))
    face.Handler(request, book).do_GET()
    assert 200 not in responses and responses == [404]


@pytest.mark.parametrize('operation', ['listen', 'run', 'pending'])
@pytest.mark.parametrize('failure', ['missing', 'identity'])
def test_legacy_converter_isolates_missing_named_reference_without_substitution(tmp_path, monkeypatch, operation, failure):
    legacy = reader('logbook_view.legacy')
    face = reader('desk.face')
    audio = legacy.audio
    book = face.Logbook(tmp_path, station={'mycall': 'N0CALL', 'grid': 'AA00'})
    converter = legacy.Converter(tmp_path, book=book)
    named = tmp_path / 'logs/onair/named.wav'
    unrelated = wave_file(tmp_path / 'logs/onair/20261101T121640Z-N0CALL-W0QXR.wav')
    log = tmp_path / 'working/session.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f'session stream: {named}\n')
    from datetime import datetime, UTC
    attempt = legacy.Attempt('vara', 'W0QXR', 7106500, 7105000, 2750, log.read_text(),
        source=log, family='loose', started=datetime(2026, 11, 1, 12, 16, 40, tzinfo=UTC),
        ended=datetime(2026, 11, 1, 12, 16, 44, tzinfo=UTC))
    original = audio.on_disk
    def on_disk(path):
        if path == named and failure == 'identity':
            raise IdentityError('named source changed')
        return original(path)
    monkeypatch.setattr(audio, 'on_disk', on_disk)
    tapes = {'N0CALL-W0QXR': [unrelated]}
    if operation == 'listen':
        converter.listen(attempt, set(), tapes)
        sources = attempt.audio
    else:
        monkeypatch.setattr(converter, 'read', lambda: [attempt])
        monkeypatch.setattr(converter, 'dedupe', lambda found: found)
        monkeypatch.setattr(converter, 'recordings', lambda: tapes)
        monkeypatch.setattr(converter, 'grids', lambda found: {})
        records = converter.pending() if operation == 'pending' else converter.run(write=False)['records']
        sources = next(iter(records.values()))['audio']
    assert any(a['path'] == str(named.relative_to(tmp_path)) and a['state'] == 'unavailable' for a in sources)
    assert all(a['path'] != str(unrelated.relative_to(tmp_path)) for a in sources)


def test_legacy_without_named_audio_keeps_current_time_discovery(tmp_path):
    legacy = reader('logbook_view.legacy')
    from datetime import datetime, UTC
    converter = legacy.Converter(tmp_path, book=SimpleNamespace(station={'call': 'N0CALL'}))
    path = wave_file(tmp_path / 'logs/onair/20261101T121640Z-N0CALL-W0QXR.wav')
    source = tmp_path / 'working/session.log'
    source.parent.mkdir(exist_ok=True)
    source.write_text('')
    attempt = legacy.Attempt('vara', 'W0QXR', 7106500, 7105000, 2750, '', source=source,
        started=datetime(2026, 11, 1, 12, 16, 40, tzinfo=UTC))
    converter.listen(attempt, set(), {'N0CALL-W0QXR': [path]})
    assert attempt.audio == [{'path': str(path.relative_to(tmp_path)), 'kind': 'rig rx'}]


@pytest.mark.parametrize('policy', ['off', 'contacts', 'everything'])
def test_varac_published_rollover_contact_is_read_through_real_desk(sessions, tmp_path, monkeypatch, policy):
    from hfmodem.core import wav
    from hfmodem.core.contactlog import ContactLog
    from hfmodem.tests.station.recording_support import RATE, push, bounded
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4 * 48000)
    root = tmp_path / 'working/vara-chat/sessions/run'
    rec, _ = sessions(policy, directory=root)
    lane = rec.stream('rx', rate=RATE, clock='card')
    log = ContactLog(root, recording=rec)
    facts = dict(n=2, peer='W0QXR', dial_hz=7105000, bandwidth_hz=2750,
        started_utc='2026-11-01T12:00:00Z', how='incoming')
    push(rec, lane, 0, 24)
    log.update(facts, sample=24, sample_rate=RATE)
    native = log.update(facts | {'ended_utc': '2026-11-01T12:00:02Z'}, sample=40, sample_rate=RATE)
    push(rec, lane, 24, 96)
    bounded(rec.close)
    published = json.loads((Path(native['directory']) / 'provenance.json').read_text())
    assert published['recording']['policy'] == policy
    face = reader('desk.face')
    book = face.Logbook(tmp_path, station={'mycall': 'N0CALL', 'grid': 'AA00'})
    monkeypatch.setattr(book, 'unconverted', lambda: {})
    [row] = book.entries()['entries']
    assert row['call'] == 'W0QXR'
    assert face.read_adif(face.adif([row], book.station))[0]['CALL'] == 'W0QXR'
    if policy == 'off':
        assert row.get('audio_state') == 'disabled' and not row['audio']
    else:
        assert published['recording']['lanes']['rx']['actual'] == [0, 80]
        assert row['audio'], 'published contact references must be playable across tiny segments'
        assert row['audio'][0]['duration_s'] == 10
        parts = row['audio'][0]['segments']
        assert sum(part['duration_s'] for part in parts) == 10
        assert max(part['start_s'] + part['duration_s'] for part in parts) == 10
