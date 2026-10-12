# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Authenticated, bounded fixed-tape driver replay with charged work."""
import hashlib
import json
import re
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair, p4chirp, p4breakintx, p4shorttx, p4host
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx, FS
from hfmodem.tests.evidence import CAPTURES, WORKING, checked_record


def prerender(records):
    """Project keyed records onto the schema pinned before `--p4-word-render`,
    asserting each carried the default off placement."""
    projected = []
    for record in records:
        record = dict(record)
        if 'word_render' in record:
            assert (record.pop('word_render'), record.pop('word_offset_hz')) == ('off', 0.)
        projected.append(record)
    return projected


def replay(arm, stop=None, *, measured_work=False, work_costs=None):
    capture = CAPTURES/arm
    manifests = Path(__file__).with_name('fixtures')/'chirp-deadline'
    require_input(capture/'stream.flac')
    manifest = json.loads((manifests/arm/'external.json').read_text())
    for row in manifest['files']:
        base = WORKING/arm if row['role'] == 'log' else capture
        checked_record(base/row['file'], manifests/arm/'external.json')
    metadata = json.loads((capture/'stream.json').read_text())
    assert metadata['samplerate'] == FS and metadata['first_stream_sample'] == 0
    assert Path(metadata['wav']).with_suffix('.flac').name == 'stream.flac'
    recorded = json.loads((capture/'p4-probe.json').read_text())
    opening = recorded['emissions'][0]
    audio = onair.session.load_wav(str(capture/'stream.flac'), FS).astype(np.float32)
    assert len(audio) == metadata['samples']
    audio = audio[:stop]
    intervals = [(e['audio_start'], e['audio_end']) for e in recorded['emissions']]
    for start, end in intervals:
        audio[start:end] = 0
    stream = Stream(audio)
    stream.pos = opening['audio_end']
    stream.wall = stream.pos
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.transmit = True
    stream.key_notice, stream._blk, stream.tx_latency_n = 1796, 128, 144
    stream._callback_timing = [(0, 128, 0., 0), (128, 128, .0001, 0), (256, 128, .0002, 0)]
    stream.clamp_late = lambda at: max(0, stream.sample_now()+stream.key_notice-at)
    tx = Tx(stream)
    tx.settle = .037
    tx.p4_probe.emissions = [opening.copy()]
    tx.p4_probe.values = recorded.get('values') or (7, 38)
    args = tx.p4_link_options
    for name, value in manifest['options'].items():
        setattr(args, 'p4_'+name, value)
    args.p4_link_timeout = len(audio)/FS-stream.pos/FS
    request = next((e for e in recorded['emissions'] if e.get('status') == 0x40), None)
    payload = bytes.fromhex(manifest['source'])
    assert hashlib.sha256(payload).hexdigest() == manifest['source_sha256']
    log_row, = [row for row in manifest['files'] if row['role'] == 'log']
    log = (WORKING/arm/log_row['file']).read_text()
    resolved, = re.findall(r'^  resolved args: (.*)$', log, re.M)
    for name, value in manifest['options'].items():
        assert f'--p4-{name.replace("_", "-")}={value}' in resolved
    lines = re.findall(r'^mail: we (?:said|queued): (.*)$', log, re.M)
    assert lines or not payload
    source = ('\r'.join(lines)+'\r').encode('latin-1') if payload else b''
    responses = {match[1] for emission in recorded['emissions'] if emission.get('payload')
                 for match in re.finditer(rb';?PR: ([0-9]{8})', bytes.fromhex(emission['payload']))}
    assert len(responses) <= 1
    if responses:
        source = source.replace(b'########', responses.pop())
    assert source == payload
    take = stream.take_until
    queued = False
    def feed(end):
        nonlocal queued
        source_ready = manifest.get('source_ready_sample',
                                    request['audio_start']-round(1.3*FS) if request else 0)
        if request and not queued and end >= source_ready:
            tx.p4_link_host.arq.on_host_data(payload)
            queued = True
        start = stream.pos
        part = take(end)
        for left, right in intervals:
            if start < right and stream.pos > left:
                rx.note_gap(max(left, start), min(right, stream.pos))
        return part
    stream.take_until = feed
    lead = onair._prekey_lead(stream, round(tx.settle*FS))
    key = tx._tx
    def emit(audio, what):
        admission_lead = stream.key_notice
        assert stream.sample_now() <= tx.raster.boundary(tx.slot)-admission_lead
        key(audio, what)
        stream.pos = max(stream.pos, tx.tx_end)
        stream.wall = max(stream.wall, stream.pos)
    tx._tx = emit
    misses = []
    def charge(seconds):
        stream.wall = stream.sample_now()+round(seconds*FS)
    def builder(build, seconds):
        def timed(*a, **kw):
            wave = build(*a, **kw)
            if a[0]:
                misses.append(dict(at=stream.sample_now(), builder=build.__name__, payload=a[0].hex()))
                charge(seconds)
            return wave
        return timed
    decisions = [e for e in recorded.get('link_events', []) if e.get('event') == 'Decision']
    rx = tx.sessrx.p4
    cycles = []
    finish_cycle = rx.finish_cycle
    def counted(onset):
        count = finish_cycle(onset)
        cycles.append((onset, count))
        return count
    rx.finish_cycle = counted
    for method in ('read_partial', 'read_window'):
        read = getattr(rx, method)
        def timed(*a, _read=read, _method=method, **kw):
            onset = rx.clock.next_due if rx.clock else stream.pos
            started = time.perf_counter()
            result = _read(*a, **kw)
            row = min(decisions, key=lambda e: abs(e['at']-onset)) if decisions else {}
            seconds = (work_costs[_method](onset, a) if work_costs is not None else
                       time.perf_counter()-started if measured_work else
                       row.get('work_ms', {}).get(_method, 3 if _method == 'read_partial' else 4)/1000)
            charge(seconds)
            return result
        setattr(rx, method, timed)
    adapter = p4host.P4Host
    with patch.object(p4host, 'P4Host', lambda host, **kw: adapter(host, terminal=True)), \
         patch.object(p4chirp, 'entry_packet', builder(p4chirp.entry_packet, .106/16)), \
         patch.object(p4breakintx, 'changeover_packet', builder(p4breakintx.changeover_packet, .0085/5)), \
         patch.object(p4shorttx, 'short_packet', builder(p4shorttx.short_packet, .002)):
        onair._run_p4_link(tx, stream, onair._MasterGrid(opening['audio_start'] % 60000,
            60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return dict(capture=arm, lead=lead, source=payload.hex(),
                source_basis=manifest['source_basis'], reason=tx.p4_probe.reason,
                emissions=tx.p4_probe.emissions, heard=tx.p4_probe.heard,
                events=tx.p4_probe.link_events, misses=misses, cycles=cycles,
                buffered=tx.p4_link_host.arq._buffer_raw,
                recorded_emissions=recorded['emissions'])
