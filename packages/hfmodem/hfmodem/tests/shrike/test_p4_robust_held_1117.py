# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VE3KPG 11-16 22:17Z: the gateway held robust entry 0 through the whole opening."""
import contextlib
from functools import lru_cache
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from hfmodem.shrike import onair, p4probe, session

FS = 48000
ROOT = Path(__file__).resolve().parents[5]


@lru_cache(maxsize=1)
def _tape():
    stream = next((ROOT / 'operator-data/captures').glob(
        'onair-20261116T221745091177Z-e6ab9e8a-*/stream-l7278-*.flac'), None)
    if stream is None:
        pytest.skip('VE3KPG 11-16 22:17Z recording is not available')
    submitted = next(stream.parent.glob('submitted-tx-*.flac'), None)
    if submitted is None:
        pytest.skip('VE3KPG 11-16 22:17Z submitted TX recording is not available')
    audio = session.load_wav(str(stream.resolve()), FS).astype(np.float32)
    own = session.load_wav(str(submitted.resolve()), FS)
    idx = np.flatnonzero(np.abs(own) > 1e-4)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 480) + 1):
        if run.size:
            audio[max(0, run[0]+84-240):run[-1]+84+240] = 0
    return audio[5633:]


def _replay(tmp_path, monkeypatch, parity=None):
    wav = tmp_path / 'peer.wav'
    session.write_wav(str(wav), _tape())
    keyed, tx = [], onair.RadioTx._tx

    def key(self, audio, what, *args, **kwargs):
        before = len(self.slots_used)
        result = tx(self, audio, what, *args, **kwargs)
        if not self.refused and (len(self.slots_used) > before or self.raster is None):
            keyed.append((self.slot if self.raster else 0, what))
        return result

    monkeypatch.setattr(onair.RadioTx, '_tx', key)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(p4probe, 'code_revision', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', [
        'onair', '--replay', str(wav), '--mycall', 'W9SSJ', '--dxcall', 'VE3KPG',
        '--dial', '7089500', '--hold', '0', '--connect', 'robust',
        '--after-robust', 'p4-robust', '--rig', 'ftdx10',
        '--serial', '/dev/null', '--ptt-port', '/dev/null',
        '--retries', '20', '--max-cycles', '32', '--outdir', str(tmp_path / 'out'),
        *(['--p4-robust-parity', parity] if parity is not None else [])])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    (tmp_path / 'replay.log').write_text(log.getvalue())
    result = json.loads((tmp_path / 'out/p4-probe.json').read_text())
    return keyed, log.getvalue(), result


def _answer(log):
    line = next(l for l in log.splitlines() if l.startswith('[robust result] '))
    return json.loads(line.split(' ', 2)[2])['answer']


def test_held_gateway_gets_v6a_in_every_slot(tmp_path, monkeypatch):
    keyed, log, result = _replay(tmp_path, monkeypatch)
    answer = _answer(log)
    assert answer['entry'] == 0, log
    shorts = result['emissions']
    assert [(e['value'], e['header']) for e in shorts] == [(6, 'A')] * 10, log
    starts = [e['audio_start'] for e in shorts]
    assert starts == [(answer['slot'] + 1 + i) * 60000 for i in range(10)], log
    assert sum(what.startswith('P4 LINK short') for _, what in keyed) == 7, keyed
    repeats = [e['slot'] for e in result['link_events'] if e['event'] == 'robust entry repeated']
    assert repeats == list(range(answer['slot'] + 1, answer['slot'] + 11)), log
    assert result['reason'] == 'retry budget exhausted'


def test_stock_parity_keeps_its_opening_and_stops(tmp_path, monkeypatch):
    _, log, result = _replay(tmp_path, monkeypatch, 'stock')
    assert [(e['value'], e['header']) for e in result['emissions']][:3] == [
        (6, 'A'), (9, 'A'), (8, 'A')], log
    assert 'whole opening plus 2 link slots' in result['reason']
