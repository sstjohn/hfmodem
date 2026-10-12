# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline drive.py replay, starting at the recorded probe's last sample."""
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_p4_link_0927 import FS

ROOT = Path(__file__).resolve().parents[5]


def probe(arm):
    return json.loads(require_input(ROOT / f'captures/onair-0930-{arm}/p4-probe.json').read_text())


def replay(arm, *, ladder='chirp', end=180., queue_at=None):
    require_input(ROOT / f'captures/onair-0930-{arm}/stream.flac')
    require_input(ROOT / 'working/p4-plan-0930/stall-bisect/drive.py')
    recorded = probe(arm)
    opening = recorded['emissions'][0]
    requests = [e for e in recorded['emissions'] if e.get('status') == 0x40]
    payload = b''.join(bytes.fromhex(e['payload']) for e in recorded['emissions']
                       if e.get('payload')) or b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;'
    if queue_at is None:
        queue_at = requests[0]['audio_start']/FS-1.3 if requests else (
            recorded['emissions'][-1]['audio_start']/FS+.25)
    results = []
    run = onair._run_p4_link

    def drive(tx, stream, raster):
        tx.settle = .044
        tx.p4_link_options.p4_ladder = ladder
        tx.p4_link_options.p4_data = 'chirp' if ladder == 'chirp' else 'body'
        take = stream.take_until
        queued = False

        def feed(end):
            nonlocal queued
            if not queued and end >= round(queue_at*FS):
                tx.p4_link_host.arq.on_host_data(payload)
                queued = True
            return take(end)

        stream.take_until = feed
        run(tx, stream, raster)
        results.append(tx)

    argv = ['drive.py', str(ROOT/f'captures/onair-0930-{arm}/stream.flac'),
            str(opening['audio_start'] % 60000), str(opening['audio_start']), '1',
            str(opening['audio_end']/FS), str(end)]
    with patch.object(onair, '_run_p4_link', drive), patch.object(sys, 'argv', argv):
        runpy.run_path(str(ROOT/'working/p4-plan-0930/stall-bisect/drive.py'), run_name='__main__')
    return results[0]
