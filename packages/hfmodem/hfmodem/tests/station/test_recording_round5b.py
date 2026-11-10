# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Crash tails are unknown beyond the last retained frame."""
import json
import subprocess
import sys

import pytest

from hfmodem.tests.station.recording_support import (
    RATE, bounded, flush, index, module, pcm, push,
)


@pytest.mark.parametrize('owned', [False, True])
def test_crash_tail_and_contact_post_roll_follow_retained_frames(tmp_path, owned):
    root = tmp_path / 'crashed'
    script = '''
import json, os, sys, threading
from hfmodem.core import capture_identity as ci, wav
from hfmodem.core.recording import RecordingSession
from hfmodem.tests.station.recording_support import RATE, TIMEOUT, flush, pcm
wav.SEGMENT_SAMPLES = 2 * 48000
rec = RecordingSession('everything', sys.argv[1], report=lambda msg: None)
owner = None
if sys.argv[2] == 'True':
    capture = ci.allocate({'capture': rec.directory / 'owned'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav', clock_id='card', continuous=True)
lane = rec.stream('rx', rate=RATE, clock='card', ownership=owner)
for cid, end in [('c1', 1120), ('c2', 1160)]:
    rec.begin(cid, {'card': 1000})
    rec.end(cid, {'card': end})
flush(rec)
entered, release, consumed, advanced = (threading.Event() for _ in range(4))
def blocked():
    entered.set()
    assert release.wait(30)
rec._enqueue(blocked)
assert entered.wait(TIMEOUT)
consume = rec._consume
def observed(*args):
    consume(*args)
    consumed.set()
rec._consume = observed
update = rec._update_contacts
def updated(*args, **kwargs):
    update(*args, **kwargs)
    if lane.consumed == 1172:
        advanced.set()
rec._update_contacts = updated
def feed(first, last):
    for start in range(first, last, RATE):
        consumed.clear()
        lane.push(start, pcm(start, min(start + RATE, last)))
        assert consumed.wait(TIMEOUT)
feed(1000, 1032)
feed(1040, 1050)
feed(1052, 1084)
lane.push(1084, pcm(1084, 1108))
feed(1116, 1148)
lane.push(1148, pcm(1148, 1172))
assert advanced.wait(TIMEOUT)
with rec._state_lock:
    facts = {'origin': lane.origin, 'position': lane.position, 'gaps': lane.gaps,
             'segments': [{'actual': c['actual'], 'spans': c['spans']} for c in lane.writer.coverage.values()]}
    print(json.dumps(facts), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), str(owned)],
                            check=True, timeout=30, capture_output=True, text=True)
    expected = json.loads(result.stdout)
    assert index(root)['segments'] == []
    assert expected['position'] == 1172
    assert expected['gaps'] == [[1032, 1040], [1050, 1052], [1084, 1116], [1148, 1172]]
    loaded = module().load_recording_index(root)
    row = loaded['lanes']['rx']
    retained_end = expected['segments'][-1]['actual'][1]
    assert row['origin'] == expected['origin'] == 1000
    assert row['position'] == retained_end == 1148
    assert row['gaps'] == [g for g in expected['gaps'] if g[1] <= retained_end]
    actual = sorted(({'actual': d['actual'], 'spans': d['spans']} for d in loaded['segments']),
                    key=lambda d: d['actual'][0])
    assert actual == expected['segments']
    assert all(0 < d['samples'] <= 2 * RATE and d['state'] == 'pending' for d in loaded['segments'])
    assert row.get('unknown_from') == retained_end
    for cid, missing in [('c1', 12), ('c2', 5 * RATE)]:
        contact = loaded['contacts'][cid]
        ref = contact['lanes']['rx']
        assert ref['actual'] == [1000, retained_end]
        assert ref['gaps'] == row['gaps']
        assert ref['missing_post_roll'] == missing
        assert ref['unknown_from'] == retained_end
        assert ref['state'] == contact['state'] == 'partial'
    (root / 'recording.json').write_text(json.dumps(loaded))
    assert module().load_recording_index(root) == loaded


def test_clean_close_preserves_known_trailing_gap_on_load(sessions):
    rec, root = sessions('everything')
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 1000})
    push(rec, lane, 1000, 1008)
    rec.end('c1', {'card': 1008})
    lane.push(1008, pcm(1008, 1032))
    flush(rec)
    bounded(rec.close)
    saved = index(root)
    assert saved['lanes']['rx']['position'] == 1032
    assert saved['lanes']['rx']['gaps'] == [[1008, 1032]]
    loaded = module().load_recording_index(root)
    for key in ('origin', 'position', 'gaps'):
        assert loaded['lanes']['rx'][key] == saved['lanes']['rx'][key]
    assert 'unknown_from' not in loaded['lanes']['rx']
    assert loaded['contacts'] == saved['contacts']


def test_recovered_cutout_marks_short_post_roll_unknown(sessions):
    rec, root = sessions('contacts')
    lane = rec.stream('rx', rate=RATE, clock='card')
    rec.begin('c1', {'card': 1000})
    push(rec, lane, 1000, 1008)
    rec.end('c1', {'card': 1008})
    flush(rec)
    cached = index(root)
    bounded(rec.close)
    (root / 'recording.json').write_text(json.dumps(cached))
    loaded = module().load_recording_index(root)
    contact = loaded['contacts']['c1']
    ref = contact['lanes']['rx']
    assert ref['actual'] == [1000, 1008]
    assert ref['missing_post_roll'] == 5 * RATE
    assert ref.get('unknown_from') == 1008
    assert ref['gaps'] == []
    assert ref['state'] == contact['state'] == 'partial'
