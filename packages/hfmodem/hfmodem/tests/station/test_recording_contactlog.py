# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Contact publication preserves selected lanes and intersecting TX evidence."""
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.core.tx_events import TxEvents, read_line
from hfmodem.station import contactlog
from hfmodem.tests.station.recording_support import (
    RATE, audio_files, bounded, flush, index, pcm,
)


def contact(**changes):
    return dict(n=1, peer='K9ZZT', dial_hz=7105000, bandwidth_hz=500,
                started_utc='2026-11-01T12:00:00+00:00',
                ended_utc='2026-11-01T12:00:10+00:00', how='outgoing', **changes)


def wave_file(path, count, rate=RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as dst:
        dst.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        dst.writeframes(np.arange(count, dtype='<i2').tobytes())


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_contactlog_publishes_adapter_result_without_cutting(sessions, monkeypatch, tmp_path, policy):
    root = tmp_path / 'sessions' / 'run'
    root.mkdir(parents=True)
    log = contactlog.ContactLog(root)
    native = log.update(contact(), sample=16, sample_rate=RATE)
    rows = []
    rec, _ = sessions(policy, directory=root, publish=rows.append)
    lane = rec.stream('rx', rate=RATE, clock='card')
    for first in (0, 8):
        lane.push(first, pcm(first, first + 8))
        flush(rec)
    rec.begin('c1', {'card': 16}, peer='K9ZZT')
    rec.end('c1', {'card': 24})
    for first in range(16, 64, 8):
        lane.push(first, pcm(first, first + 8))
        flush(rec)
    bounded(rec.close)
    selected = rows[0] if rows else {'id': 'c1', 'lanes': {}, 'state': 'disabled'}
    native['recording'].update(selected, policy=policy)
    before = {path: path.read_bytes() for path in audio_files(root)}

    def forbidden(*args):
        pytest.fail('new policy contact made an automatic whole-contact copy')

    monkeypatch.setattr(contactlog, 'cut_audio', forbidden)
    log.finish(native)
    published = json.loads((Path(native['directory']) / 'provenance.json').read_text())
    assert published['call'] == 'K9ZZT'
    assert published['recording']['lanes'] == selected['lanes']
    assert published['recording']['policy'] == policy
    assert published['recording']['state'] == selected['state']
    assert {path: path.read_bytes() for path in audio_files(root)} == before
    assert audio_files(Path(native['directory'])) == []
    if policy != 'off':
        indexed = dict(index(root)['contacts']['c1']['lanes']['rx'])
        assert selected['lanes']['rx'].pop('state') in ('pending', indexed.pop('state'))
        assert selected['lanes']['rx'] == indexed


@pytest.mark.parametrize('lo,hi,expected', [
    (5, 15, (0, 5)), (15, 25, (5, 15)), (25, 35, (15, 20)),
    (5, 35, (0, 20)), (0, 10, None), (30, 40, None),
])
def test_contactlog_clips_intersecting_tx_and_preserves_original_bounds(tmp_path, lo, hi, expected):
    root = tmp_path / 'sessions' / 'run'
    source = root / 'receive.wav'
    wave_file(source, 48)
    journal = TxEvents('vara', session_id='synthetic-session')
    original = journal.record('completed', 1, outcome='sent', recording=str(source), sample_rate=RATE,
                              ptt_start_sample=lo, ptt_end_sample=hi,
                              audio_start_sample=lo, audio_end_sample=hi)
    (root / 'arm.log').write_text('\n'.join(journal.lines()) + '\n')
    log = contactlog.ContactLog(root)
    opened = contact()
    opened.pop('ended_utc')
    log.update(opened, sample=10, sample_rate=RATE)
    native = log.update(contact(), sample=30, sample_rate=RATE)
    log.finish(native)
    events = [event for line in (Path(native['directory']) / 'arm.log').read_text().splitlines()
              if (event := read_line(line)) is not None]
    assert len(events) == (0 if expected is None else 1)
    if expected is not None:
        [event] = events
        assert (event['ptt_start_sample'], event['ptt_end_sample']) == expected
        assert (event['audio_start_sample'], event['audio_end_sample']) == expected
        assert event['original_bounds'] == {field: original[field] for field in
            ('ptt_start_sample', 'ptt_end_sample', 'audio_start_sample', 'audio_end_sample')}
        assert event['tx_id'] == original['tx_id']


@pytest.mark.parametrize('segmented', (False, True))
def test_legacy_cutout_output_never_exceeds_native_segment_capacity(tmp_path, monkeypatch, segmented):
    capacity = 60 * RATE
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    source, target = tmp_path / 'source.wav', tmp_path / 'contact.wav'
    values = np.arange(3 * capacity, dtype='<i2')
    if segmented:
        writer = wav.SegmentedWriter(source, rate=RATE, closed=lambda path: None)
        writer.writeframes(values.tobytes())
        writer.close()
    else:
        wave_file(source, len(values))
    contactlog.cut_audio(source, target, 7, 3 * capacity - 3)
    files = [path for path in audio_files(tmp_path) if path != source and not path.name.startswith('source-')]
    assert files
    blocks = []
    for path in files:
        with wave.open(str(path)) as src:
            assert src.getnframes() <= capacity, 'automatic legacy cutout exceeded native segment capacity'
            assert src.getframerate() == RATE
            blocks.append(np.frombuffer(src.readframes(src.getnframes()), '<i2'))
    np.testing.assert_array_equal(np.concatenate(blocks), values[7:-3])


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_contactlog_receives_postroll_on_storage_worker(sessions, monkeypatch, tmp_path, policy):
    rec, root = sessions(policy, directory=tmp_path / 'sessions' / 'run')
    lane = rec.stream('rx', rate=RATE, clock='card')
    log = contactlog.ContactLog(root, recording=rec)
    published = []
    finish = log.finish
    def completed(record):
        import threading
        published.append(threading.current_thread())
        finish(record)
    monkeypatch.setattr(log, 'finish', completed)
    opened = contact()
    opened.pop('ended_utc')
    for first in (0, 8):
        lane.push(first, pcm(first, first + 8))
        flush(rec)
    log.update(opened, sample=16, sample_rate=RATE)
    lane.push(16, pcm(16, 24))
    flush(rec)
    native = log.update(contact(), sample=24, sample_rate=RATE)
    for first in range(24, 64, 8):
        lane.push(first, pcm(first, first + 8))
        flush(rec)
    bounded(rec.close)
    saved = json.loads((Path(native['directory']) / 'provenance.json').read_text())
    assert saved['recording']['policy'] == policy
    assert published == [rec._storage_thread]
    assert audio_files(Path(native['directory'])) == []
    if policy != 'off':
        assert saved['recording']['lanes']['rx']['actual'] == [0, 64]
    else:
        assert saved['recording']['state'] == 'finalized'
        assert saved['recording']['lanes'] == {}


def test_contact_edges_map_submitted_tx_from_source_anchor(sessions, tmp_path):
    rec, root = sessions('contacts', directory=tmp_path / 'sessions' / 'run')
    rx = rec.stream('rx', rate=RATE, clock='card')
    tx = rec.stream('submitted-tx', rate=RATE, clock='submitted-tx')
    tx.row['source_anchor'] = {'clock': 'card', 'sample': 1000, 'position': 2000}
    log = contactlog.ContactLog(root, recording=rec)
    opened = contact()
    opened.pop('ended_utc')
    rx.push(1000, pcm(1000, 1008))
    tx.push(2000, pcm(2000, 2008))
    flush(rec)
    log.update(opened, sample=1000, sample_rate=RATE)
    log.update(contact(), sample=1008, sample_rate=RATE)
    flush(rec)
    second = opened | {'n': 2}
    log.update(second, sample=2000, sample_rate=RATE)
    tx.push(3000, pcm(3000, 3008))
    flush(rec)
    log.update(second | {'ended_utc': contact()['ended_utc']}, sample=2008, sample_rate=RATE)
    for first in range(3008, 3048, 8):
        tx.push(first, pcm(first, first + 8))
        flush(rec)
    bounded(rec.close)
    selected = index(root)['contacts']['vara-2']
    assert selected['positions']['submitted-tx'] == 3000
    assert selected['end_positions']['submitted-tx'] == 3008
    assert selected['lanes']['submitted-tx']['requested'] == [2520, 3048]
    assert selected['lanes']['submitted-tx']['actual'] == [3000, 3048]
