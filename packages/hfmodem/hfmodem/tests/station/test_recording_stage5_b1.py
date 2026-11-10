# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Producer schedules and bounded readers preserve source-clock evidence."""
import hashlib

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_stage5_support import (
    RATE, START, STOP, bounded, context, entry, index, push, reader, save_selection, select,
    write_json,
)


def test_publication_hashes_only_supplied_context_ranges(sessions, tmp_path):
    supplied = context(tmp_path)
    expected = {name: supplied[name].pop('sha256') for name in ('transcript', 'probe')}
    for name in expected:
        supplied[name]['path'] = supplied[name]['path'].removeprefix('working/run/')
    rec, root = sessions('everything', directory=tmp_path / 'working/run')
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, START - 16, START)
    rec.begin('c2', {'card': START}, context=supplied)
    rec.end('c2', {'card': START + 8})
    push(rec, lane, START, STOP + 16)
    bounded(rec.close)
    ctx = index(root)['contacts']['c2']['context']
    assert {name: ctx[name]['sha256'] for name in expected} == expected


def test_state_changes_keep_preroll_predecessor_and_lane_generation(sessions):
    rec, root = sessions('contacts')
    rx = rec.stream('rx', rate=RATE, clock='card')
    rec.stream('witness', rate=RATE, clock='remote')
    rec.state('rx', START - 600, clock_generation=2, expected_session=0x22, arq_epoch=4)
    rec.state('rx', START - 20, clock_generation=2, muted=True)
    push(rec, rx, START - 16, START)
    rec.begin('c2', {'card': START})
    rec.state('witness', START + 1, clock_generation=9, expected_session=0x99)
    rec.state('rx', START + 2, clock_generation=2, muted=False)
    rec.end('c2', {'card': START + 8})
    push(rec, rx, START, STOP + 16)
    bounded(rec.close)
    ctx = index(root)['contacts']['c2']['context']
    assert ctx['lanes']['rx']['clock_generation'] == 2
    assert ctx['initial']['expected_session'] == 0x22
    assert ctx['initial']['muted'] is True
    assert ctx['changes'] == [dict(sample=START + 2, clock_generation=2, muted=False)]


def test_selection_requires_clock_uncertainty(tmp_path):
    selected = entry(tmp_path)
    del selected['recording']['context']['lanes']['rx']['anchor']['uncertainty_s']
    result = select(selected, save_selection(tmp_path, selected), tmp_path)
    assert not result.complete and result.reason
    assert result.tx == () and result.grading is None


def test_selection_requires_transcript_range_clock_origin(tmp_path):
    selected = entry(tmp_path)
    del selected['recording']['context']['transcript']['first_s']
    result = select(selected, save_selection(tmp_path, selected), tmp_path)
    assert not result.complete and result.reason
    assert result.tx == () and result.grading is None


def test_selection_grades_only_participation_with_known_expected_session(tmp_path):
    selected = entry(tmp_path)
    result = select(selected, save_selection(tmp_path, selected), tmp_path)
    assert [frame.at for frame in result.grading] == [1001., 1002.]
    result = select(selected, tmp_path / selected['recording']['lanes']['rx']['source'],
                    tmp_path, start=8024, stop=STOP)
    assert result.grading == ()


def test_selection_transcript_clock_has_explicit_range_origin(tmp_path):
    selected = entry(tmp_path)
    ctx = selected['recording']['context']
    ctx['transcript']['first_s'] = 998.
    ctx['lanes']['rx']['anchor'].update(sample=START, transcript_s=1000.)
    result = select(selected, save_selection(tmp_path, selected), tmp_path)
    assert result.complete
    assert [f.at for f in result.log.frames] == [1000.]


def test_range_validation_reads_no_transcript_tail(tmp_path):
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    ctx = selected['recording']['context']
    log = tmp_path / ctx['transcript']['path']
    with log.open('ab') as stream:
        stream.write(b'x' * 100000)
    assert select(selected, path, tmp_path).complete
    a, b = ctx['transcript']['range']
    assert hashlib.sha256(log.read_bytes()[a:b]).hexdigest() == ctx['transcript']['sha256']


def test_bounded_family_read_isolates_moved_earlier_segment(sessions, monkeypatch):
    from hfhost import recordings
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 4 * 48000)
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, START, STOP + 32)
    bounded(rec.close)
    rows = index(root)['segments']
    (root / rows[0]['source']).unlink()
    _, capacity = recordings.indexed(root / lane.row['source'])
    assert capacity == 4 * RATE
    got = wav.read(root / lane.row['source'], start=STOP, stop=STOP + 8)
    np.testing.assert_array_equal(got, np.arange(STOP, STOP + 8) / 32768.)


@pytest.mark.parametrize('state', ['failed', 'disabled'])
def test_failed_lane_has_no_available_action_even_with_a_retained_file(tmp_path, state):
    from hfhost import recordings
    selected = entry(tmp_path)
    path = save_selection(tmp_path, selected)
    ref = selected['recording']['lanes']['rx'] | {'state': state}
    write_json(tmp_path / 'recording.json', dict(policy='everything', lanes={'rx': ref},
        contacts={}, segments=[ref | {'lane': 'rx', 'samples': STOP - START}]))
    assert recordings.status(path)['state'] == 'unavailable'
    assert not recordings.available(path)


def test_armsource_keeps_long_keying_with_bounded_pcm(tmp_path, monkeypatch):
    arm = reader('consolelib.armsource')
    root = tmp_path / 'arms/vara-c2'
    write_json(root / 'provenance.json', {'arm': root.name, 'dial_hz': 7105000})
    (root / 'arm.log').write_text('')
    rate = 48000
    rng = np.random.default_rng(7)
    samples = rng.normal(0, .02, 12 * rate)
    t = np.arange(len(samples)) / rate
    muted = (t >= 1.8) & (t < 10.4)
    samples[muted] *= .01
    samples[muted] += .2 * np.sin(2 * np.pi * 1500 * t[muted])
    wav.write(root / 'tap.wav', samples, rate)
    monkeypatch.setattr(arm.rigsource, 'station_config', lambda *a: None)
    with arm.ArmSource(root, realtime=False) as source:
        assert len(source.spans) == 1
        lo, hi = source.spans[0]
        assert lo == pytest.approx(1.8, abs=2 * arm.STEP_S)
        assert hi == pytest.approx(10.4, abs=2 * arm.STEP_S)
