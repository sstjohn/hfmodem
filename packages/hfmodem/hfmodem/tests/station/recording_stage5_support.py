# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Selected contact fixtures use absolute samples and explicit clock generations."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import wave

import numpy as np

from hfmodem.tests.station.recording_support import bounded, index, push

ROOT = Path(__file__).resolve().parents[5]
RATE = 8
START, STOP = 8000, 8032


def readers():
    for path in (ROOT, ROOT / 'tools', ROOT / 'tools/logbook_view'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def reader(name):
    readers()
    module = importlib.import_module(name)
    if name.endswith('vara_view'):
        from hfmodem.tests import hermetic
        # A built view records the viewer checkout's revision.
        hermetic.guard.admit(['git', '-C', str(module.TREE), 'rev-parse', 'HEAD'], os.environ)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def wave_file(path, count=96, rate=RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as dst:
        dst.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        dst.writeframes((np.arange(count, dtype='<i2') % 30000).tobytes())
    return path


def transcript(root):
    lines = [
        'originating: N0CALL -> K9ZZT (BW500)',
        '2026-11-01 12:00:00,000 INFO RX IDLE sess=0x11 ok=True',
        '[DISC]',
        'originating: W0QXR -> N0CALL (BW2750)',
        '2026-11-01 12:16:40,000 INFO RX DATAACK sess=0x22 ok=UNVERIFIED',
        '[IRS]',
        '2026-11-01 12:16:41,000 INFO RX IDLE sess=0x22 ok=True',
        '2026-11-01 12:16:42,000 INFO RX END sess=0x22 ok=UNVERIFIED',
        '[DISC]',
    ]
    path = root / 'working/run/arm.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    return path


def context(root, *, protocol='ARDOP', source_id='source-rx', clock_generation=2):
    path = transcript(root)
    data = path.read_bytes()
    first = data.index(b'originating: W0QXR')
    probe = root / 'working/run/selected-probe.json'
    write_json(probe, {'emissions': [{'tx_id': 2, 'audio_start': 7998, 'audio_end': 8006}]})
    return {
        'version': 1, 'id': 'context-c2', 'contact_id': 'c2', 'endpoint_generation': 7,
        'protocol': protocol, 'calls': {'local': 'N0CALL', 'peer': 'W0QXR'},
        'local_role': 'called', 'transcript': {
            'path': path.relative_to(root).as_posix(), 'source_id': 'transcript-run',
            'first_s': 1000.,
            'range': [first, len(data)], 'sha256': hashlib.sha256(data[first:]).hexdigest()},
        'lanes': {'rx': {'source_id': source_id, 'index_id': 'index-rx', 'clock': 'card',
                         'clock_generation': clock_generation, 'rate': RATE,
                         'anchor': {'sample': START, 'transcript_s': 1000., 'uncertainty_s': .002}}},
        'initial': {'expected_session': 0x22, 'arq_epoch': 4, 'bandwidth_hz': {'VARA': 2750, 'ARDOP': 500, 'PACTOR': 2400}[protocol],
                    'mode': 'normal', 'muted': True, 'grid': {'anchor': 7992, 'cycle': 8}},
        'changes': [
            {'sample': 8008, 'clock_generation': 2, 'muted': False},
            {'sample': 8016, 'clock_generation': 2, 'arq_epoch': 5, 'mode': 'P4'},
            {'sample': 8024, 'clock_generation': 2, 'expected_session': None},
        ],
        'tx': [{'tx_id': 2, 'source_id': source_id, 'clock_generation': 2,
                'ptt_start_sample': 7996, 'ptt_end_sample': 8008,
                'audio_start_sample': 7998, 'audio_end_sample': 8006,
                'sample_rate': RATE, 'protocol': protocol, 'label': 'selected TX'}],
        'grid': [{'sample': 7992, 'clock_generation': 2, 'cycle': 8},
                 {'sample': 8020, 'clock_generation': 2, 'word': 'CS1'}],
        'probe': {'path': probe.relative_to(root).as_posix(), 'source_id': 'probe-c2',
                  'range': [0, probe.stat().st_size], 'sha256': hashlib.sha256(probe.read_bytes()).hexdigest(),
                  'emissions': [{'tx_id': 2, 'audio_start': 7998, 'audio_end': 8006}]},
    }


def reference(source='working/run/renamed.wav', *, actual=None, gaps=()):
    return {'lane': 'rx', 'source': source, 'source_id': 'source-rx', 'index_id': 'index-rx',
            'clock': 'card', 'clock_generation': 2, 'rate': RATE,
            'requested': [START - 8, STOP + 8], 'actual': actual or [START, STOP],
            'selection': actual or [START, STOP], 'participation': [8008, 8024],
            'gaps': list(gaps), 'state': 'finalized'}


def entry(root, *, protocol='ardop', source='working/run/renamed.wav', gaps=()):
    ctx = context(root, protocol=protocol.upper())
    ref = reference(source, gaps=gaps)
    return {'id': 'c2', 'protocol': protocol, 'call': 'W0QXR', 'local_role': 'called',
            'bw_hz': 2750 if protocol == 'vara' else 500,
            'started_utc': '2026-11-01T12:16:41Z', 'ended_utc': '2026-11-01T12:16:43Z',
            'reached': ['keyed', 'connect'], 'speed': None,
            'audio': [{'kind': 'rig rx', 'url': '/file/' + source, 'selection': ref}],
            'transcript': '/file/working/run/arm.log',
            'recording': {'policy': 'contacts', 'state': 'finalized', 'id': 'c2',
                          'context': ctx, 'lanes': {'rx': ref}}}


def save_selection(root, selected):
    ref = selected['recording']['lanes']['rx']
    path = root / ref['source']
    wave_file(path, STOP - START)
    write_json(path.with_suffix('.json'), {'recording': selected['recording'],
        'source_id': ref['source_id'], 'index_id': ref['index_id'], 'clock': ref['clock'],
        'clock_generation': ref['clock_generation'], 'start_stream_sample': START,
        'end_stream_sample': STOP, 'samplerate': RATE, 'samples': STOP - START})
    return path


def select(selected, path, root, **kwargs):
    module = reader('hfmodem.station.sessionlog')
    helper = getattr(module, 'select', None)
    assert callable(helper), 'sessionlog.select must evaluate one contact selection and context'
    return helper(selected['recording'], path, root=root, **kwargs)


def recorded(sessions, *, policy='everything', directory=None, publish=None):
    options = {'directory': directory} if directory is not None else {}
    if publish is not None:
        options['publish'] = publish
    rec, root = sessions(policy, **options)
    lane = rec.stream('rx', rate=RATE, clock='card')
    push(rec, lane, 0, 24)
    rec.begin('c2', {'card': 24}, peer='W0QXR')
    rec.end('c2', {'card': 40})
    push(rec, lane, 24, 96)
    bounded(rec.close)
    return rec, root, index(root)


def clone(value):
    return copy.deepcopy(value)
