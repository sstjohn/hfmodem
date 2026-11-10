# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recovered contact coverage belongs to one lane and recording family."""
import json
from pathlib import Path
import subprocess
import sys
import wave

import pytest

from hfmodem.tests.station.recording_support import RATE, module


@pytest.mark.parametrize('published', [False, True])
def test_recovered_owned_contacts_keep_independent_lane_clocks(tmp_path, published):
    root = tmp_path / 'crashed'
    script = '''
import os, sys, threading
from hfmodem.core import capture_identity as ci
from hfmodem.core.recording import RecordingSession
from hfmodem.tests.station.recording_support import RATE, TIMEOUT, flush, pcm
rec = RecordingSession('everything', sys.argv[1], report=lambda message: None)
capture = ci.allocate({'capture': rec.directory / 'owned'}, {'modem': 'capture'})
container = capture.claim('modem')
lanes = []
for name, source in [('rx', 'receive.wav'), ('tx', 'transmit.wav')]:
    owner = ci.Recording.reserve(container, source, clock_id=name, continuous=True)
    lanes.append(rec.stream(name, rate=RATE, clock=name, ownership=owner))
rec.begin('c1', {'rx': 1000, 'tx': 1020})
rec.end('c1', {'rx': 1008, 'tx': 1028})
flush(rec)
entered, release, consumed = (threading.Event() for _ in range(3))
def blocked():
    entered.set()
    assert release.wait(30)
if sys.argv[2] == 'False':
    rec._enqueue(blocked)
    assert entered.wait(TIMEOUT)
consume = rec._consume
def observed(*args):
    consume(*args)
    consumed.set()
rec._consume = observed
for lane, first in zip(lanes, [1000, 1020]):
    consumed.clear()
    lane.push(first, pcm(first, first + RATE))
    assert consumed.wait(TIMEOUT)
    lane.writer.flush()
if sys.argv[2] == 'True':
    flush(rec)
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(root), str(published)],
                   check=True, timeout=30)
    loaded = module().load_recording_index(root)
    assert len({Path(d['source']).parent for d in loaded['segments']}) == 1
    for name, first in [('rx', 1000), ('tx', 1020)]:
        ref = loaded['contacts']['c1']['lanes'][name]
        [descriptor] = [d for d in loaded['segments'] if d['lane'] == name]
        assert ref['actual'] == [first, first + RATE]
        assert ref['unknown_from'] == first + RATE
        assert ref['missing_post_roll'] == 5 * RATE
        assert ref['gaps'] == []
        assert ref['segments'] == [descriptor['source']]
        assert ref['coverage'] == [{
            'source': descriptor['source'], 'actual': [first, first + RATE],
            'spans': [{'absolute': [first, first + RATE], 'offset': 0}],
        }]


def test_recovered_contact_excludes_other_family_on_same_lane(tmp_path):
    directory = tmp_path / ('a' * 32)
    directory.mkdir()
    descriptors = []
    for stem, first in [('receive-l7278', 1000), ('other-l7278', 1020)]:
        path = directory / f'{stem}-20261101T0000Z-s{first:020d}.wav'
        with wave.open(str(path), 'wb') as audio:
            audio.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
            audio.writeframes(bytes(2 * RATE))
        descriptors.append({'source': str(path.relative_to(tmp_path)), 'lane': 'rx',
                            'state': 'pending', 'rate': RATE})
    source = str((directory / 'receive-l7278.wav').relative_to(tmp_path))
    cached = {'lanes': {'rx': {'source': source, 'origin': 1000}},
              'segments': descriptors, 'contacts': {'c1': {'lanes': {'rx': {
                  'source': source, 'lane': 'rx', 'clock': 'card', 'rate': RATE,
                  'requested': [1000, 1048], 'kind': 'interval',
              }}, 'end_positions': {'card': 1008}}}}
    (tmp_path / 'recording.json').write_text(json.dumps(cached))
    loaded = module().load_recording_index(tmp_path)
    ref = loaded['contacts']['c1']['lanes']['rx']
    assert ref['actual'] == [1000, 1008]
    assert ref['unknown_from'] == 1008
    assert ref['gaps'] == []
    assert ref['missing_post_roll'] == 5 * RATE
    assert ref['segments'] == [descriptors[0]['source']]
