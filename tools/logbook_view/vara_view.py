# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Build an offline VARA inspection artifact for one existing logbook QSO.

No radio, network, multidecode service or transmitter is used. The decoder is
loaded from this checkout; its revision and actual imported source hashes are
recorded, including uncommitted instrumentation. Artifacts never change arms.
"""
from __future__ import annotations

import argparse
import hashlib
import errno
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(TREE / 'packages/hfmodem'))
sys.path.insert(0, str(TREE / 'tools'))
sys.path.insert(0, str(TREE))

import vara_timing
from vara_signal import signal_view, align_audio, align_packets, align_handshakes
from vara_packets import consolidate, packet_bytes, vara_payloads

import numpy as np
from scipy.signal import resample_poly

from hfmodem.core import wav
from hfmodem.kestrel import monitor
from hfmodem.kestrel.rx import varaofdm, varatone
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.arq.phy import _tone_burst

SCHEMA = 1
FS = 48000
NFFT, HOP, BINS, TILE_ROWS = 4096, 2400, 513, 256
MAX_SECONDS = 1800
MAX_EVENTS = 10000


def sha(path, *, bounds=None):
    h = hashlib.sha256()
    path = Path(path)
    if bounds is not None:
        for first, last in bounds:
            for start in range(first, last, 65536):
                h.update(wav.read(path, start=start, stop=min(last,start+65536)).tobytes())
        return h.hexdigest()
    for part in wav.segments(path) if path.suffix in ('.wav', '.flac') else [path]:
        from hfhost.recordings import open_segment
        with (open_segment(part) if part.suffix in ('.wav', '.flac') else part.open('rb')) as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False, separators=(',', ':')))


def load_audio(path, *, start=None, stop=None):
    from audio import duration_s, sidecar
    from hfhost.recordings import audio_info
    from hfhost.recordings import intervals
    parts = [part for (a,b),part in intervals(path, start=start, stop=stop)
             if (start is None or b>start) and (stop is None or a<stop)] if start is not None or stop is not None else wav.segments(path)
    rate = audio_info(parts[0])[0]
    seconds = (stop-start)/rate if start is not None and stop is not None else duration_s(path, sidecar(path))
    if seconds and seconds > MAX_SECONDS:
        raise ValueError(f'Replay is limited to {MAX_SECONDS // 60} minutes per recording.')
    x = wav.read(path, start=start, stop=stop)
    if len(x) / rate > MAX_SECONDS:
        raise ValueError('Recording exceeds the replay duration limit.')
    if not np.isfinite(x).all():
        raise ValueError('Recording contains non-finite samples.')
    if rate != FS:
        g = math.gcd(rate, FS)
        x = resample_poly(x, FS // g, rate // g)
    return x, rate


def waterfall(x, out):
    """Fixed dBFS scale, bounded FFT batches; first row is centered at t=0."""
    window = np.hanning(NFFT)
    padded = np.pad(x, (NFFT // 2, NFFT // 2))
    count = max(1, math.ceil(len(x) / HOP))
    tiles = []
    for first in range(0, count, TILE_ROWS):
        positions = np.arange(first, min(first + TILE_ROWS, count)) * HOP
        blocks = padded[positions[:, None] + np.arange(NFFT)]
        amplitude = np.abs(np.fft.rfft(blocks * window, axis=1))[:, :BINS] * 2 / window.sum()
        db = 20 * np.log10(np.maximum(amplitude, 1e-12))
        pixels = np.clip((db + 100) * 255 / 100, 0, 255).astype(np.uint8)
        name = f'waterfall-{first // TILE_ROWS:05d}.bin'
        (out / name).write_bytes(pixels.tobytes())
        tiles.append({'file': name, 'first': first, 'rows': len(pixels)})
    return {'bins': BINS, 'hz_per_bin': FS / NFFT, 'hop_s': HOP / FS,
            'rows': count, 'db_min': -100, 'db_max': 0, 'tiles': tiles}


def symbol_event(frame, base, number, out, duration):
    points = frame.points
    times = (base + frame.sample_indices) / FS
    valid = np.isfinite(points.real) & np.isfinite(points.imag) & (times >= 0) & (times <= duration)
    name = f'symbols-{number:05d}.json'
    write_json(out / name, {
        'i': points.real[valid].round(6).tolist(), 'q': points.imag[valid].round(6).tolist(),
        't': times[valid].round(6).tolist(), 'carrier': frame.carrier_hz[valid].tolist(),
        'ideal': [[float(z.real), float(z.imag)] for z in frame.ideal],
        'stage': frame.stage, 'normalization': 'receiver decision-input gain; fixed I/Q axes',
        'omitted_nonfinite_or_outside': int((~valid).sum()),
    })
    return {'id': f'frame-{number:05d}', 't0': max(0, (base + frame.start_sample) / FS),
            't1': min(duration, (base + frame.end_sample) / FS),
            'kind': f'{"Tone" if "tone" in frame.stage else "OFDM"} L{frame.level}', 'direction': 'observed',
            'status': 'CRC pass' if frame.crc_ok else 'CRC failed · level hypothesis',
            'detail': f'{frame.stage}; {len(points)} payload symbols', 'level': frame.level,
            'anatomy': [{'start_s': 0, 'end_s': (frame.end_sample-frame.start_sample)/FS, 'label': 'Coded payload symbols', 'basis': 'Decoder payload-cell span; lead-in and control are outside this interval.'}],
            'frame_type': 'data', 'stage': frame.stage, 'symbols': name, 'carrier_hz': sorted(set(frame.carrier_hz.tolist())),
            'snr_db': float(frame.snr_db) if math.isfinite(frame.snr_db) else None,
            'offset_hz': float(frame.offset_hz), 'origin': 'instrumented VARA replay'}


def replay(x, bw, calls, out, progress=print, tx=(), decode_tx=False, selection=None):
    """One VARA gate/scanner and decoder, no multimode dispatcher."""
    gate = monitor.MonitorGate()
    scanner = monitor.HandshakeScanner(calls)
    events = []
    duration = len(x) / FS

    def append(event):
        if len(events) >= MAX_EVENTS:
            raise ValueError('Replay event limit exceeded; use a shorter recording.')
        events.append(event)

    def base_event(result, start, end, origin, timing):
        return {'id': f'base-{len(events):05d}', 't0': start, 't1': end,
                'kind': f'Base data L{result.level}', 'direction': 'observed',
                'status': 'CRC pass', 'frame_type': 'data', 'symbols': None,
                'frame_count': len(result.frames),
                'payloads': [packet_bytes(f, 'VARA decoded body') for f in result.frames],
                'packet_sha256': hashlib.sha256(b''.join(result.frames)).hexdigest(),
                'origin': origin, 'timing_precision': timing,
                'detail': f'{len(result.frames)} CRC-valid frame(s); '
                          f'{sum(len(f) for f in result.frames)} decoded bytes including protocol fields.',
                'unavailable': ('Base index modulation; a PSK/QAM constellation does not describe these symbols.'
                                if bw != 500 else 'Base BW500 symbol telemetry is not implemented.')}

    def burst(base, segment):
        # Gate intervals are candidates, not assertions that a frame was sent.
        frames = []
        if len(segment) >= 3 * FS:
            # Pad the cropped gate interval so the locator can see the lead/tail.
            lo, hi = max(0, base - FS // 4), min(len(x), base + len(segment) + FS // 4)
            decoded = varaofdm.decode_burst(x[lo:hi], bw, on_symbols=frames.append)
            if not any(f.crc_ok for f in frames):
                tone_frames = []
                result = varatone.decode_burst(x[lo:hi], bw, on_symbols=tone_frames.append)
                if result.ok or (not frames and _tone_burst(x[lo:hi], bw, result.level)):
                    frames = tone_frames
                    decoded = result
            clean = iter(decoded.frames)
            for frame in frames:
                event = symbol_event(frame, lo, len(events), out, duration)
                if frame.crc_ok:
                    raw = next(clean)
                    event['packet_sha256'] = hashlib.sha256(raw).hexdigest()
                    event['payloads'] = [packet_bytes(raw, 'VARA decoded frame · includes protocol fields')]
                append(event)
        if not frames:
            result = monitor.classify(segment, calls, decode_wideband=True,
                                      reference=gate.reference())
            # The classifier reports a CRC verdict, but intentionally discards
            # bytes. Materialize base decoder frames before publishing that verdict.
            base_reads = []
            if result.kind == 'DATA over' and result.quality == 'CRC ok':
                spans = (monitor.RX.detect_overs(segment) or [(0, len(segment))]
                         if bw != 500 else [(0, len(segment))])
                for first, last in spans:
                    decoded = phy.decode_level(segment[first:last], str(bw), 4)
                    if decoded is not None and decoded.frames:
                        base_reads.append((first, last, decoded))
            if base_reads:
                for first, last, decoded in base_reads:
                    append(base_event(decoded, (base + first) / FS,
                                      min(duration, (base + last) / FS),
                                      'VARA base decoder in blind burst',
                                      'energy-gate window; exact frame boundaries not exported'))
            else:
                append({'id': f'event-{len(events):05d}', 't0': base / FS,
                    't1': min(duration, (base + len(segment)) / FS),
                    'kind': result.kind, 'direction': 'observed',
                    'role': ('caller' if calls and result.kind == 'link-setup' and result.caller == calls[0]
                             else ('caller' if result.kind == 'CR' else 'called')
                             if calls and result.kind in ('CR', 'connect-response') and result.gateway == calls[-1]
                             else 'unknown'),
                    'status': 'CRC pass' if result.kind == 'DATA over' and result.quality == 'CRC ok' else 'classified burst',
                    'frame_type': 'data' if result.kind == 'DATA over' else 'observation', 'detail': ' · '.join(filter(None, [result.info, result.quality])),
                    'symbols': None, 'origin': 'VARA burst classifier',
                    'unavailable': 'No supported decision symbols for this burst; index/MFSK/control symbol adapters are not implemented.'})
        progress(f'  {base / FS:.1f}s: {len(frames)} symbol frame(s), {len(events)} events', flush=True)

    for base in range(0, len(x), 4800):
        chunk = x[base:base + 4800]
        for at, r in scanner.push(chunk):
            # Scanner gives an onset, not a measured frame duration.
            t = at / FS
            if 0 <= t <= duration:
                append({'id': f'scan-{len(events):05d}', 't0': t, 't1': t,
                        'kind': r.kind, 'direction': 'observed',
                        'handshake_key': f'{r.kind} {r.gateway or r.info}',
                        'role': ('caller' if r.kind == 'CR' else 'called') if calls and r.gateway == calls[-1] else 'unknown',
                        'status': 'handshake detection',
                        'detail': ' · '.join(filter(None, [r.info, r.quality])), 'symbols': None,
                        'origin': 'VARA handshake scanner',
                        'unavailable': 'MFSK handshake; no I/Q constellation adapter.'})
        for at, segment in gate.push(chunk):
            burst(at, segment)
    for at, segment in gate.flush():
        burst(at, segment)
    # The energy gate can miss a whole over when its floor was learned during
    # traffic. Original PTT timestamps independently identify receive windows.
    # Re-decode short gaps in full, including the unmute edge; only CRC-clean
    # results supplement the blind sweep. No transcript payload is used.
    recovered = []
    windows = [(a['t1'], b['t0']) for a,b in zip(tx, tx[1:])]
    if decode_tx:
        windows += [(e['t0'],e['t1']) for e in tx]
    for start, end in windows:
        if not 3 <= end-start <= 12:
            continue
        if any(e.get('frame_type')=='data' and e['status']=='CRC pass' and e.get('payloads') and
               start <= (e['t0']+e['t1'])/2 <= end for e in events):
            continue
        lo, hi = max(0,int((start-.05)*FS)), min(len(x),int((end+.05)*FS))
        frames = []
        result = varatone.decode_burst(x[lo:hi], bw, on_symbols=frames.append)
        if not result.ok:
            frames = []
            result = varaofdm.decode_burst(x[lo:hi], bw, on_symbols=frames.append)
        if not any(f.crc_ok for f in frames):
            base_result = phy.decode_level(x[lo:hi], str(bw), 4)
            if base_result is not None and base_result.frames:
                event = base_event(base_result, start, end,
                                   'VARA base decoder in logged '+('transmit/receive window' if decode_tx else 'receive window'),
                                   'logged window; exact frame boundaries not exported')
                append(event)
                recovered.append(event)
                progress(f'  {start:.1f}s: recovered {event["kind"]} in logged RX window', flush=True)
        clean = iter(result.frames)
        for frame in frames:
            if frame.crc_ok:
                event = symbol_event(frame, lo, len(events), out, duration)
                raw = next(clean)
                event['packet_sha256'] = hashlib.sha256(raw).hexdigest()
                event['payloads'] = [packet_bytes(raw, 'VARA decoded frame · includes protocol fields')]
                event['origin'] = 'VARA replay in logged '+('transmit/receive window' if decode_tx else 'receive window')
                append(event)
                recovered.append(event)
                progress(f'  {event["t0"]:.1f}s: recovered {event["kind"]} in logged RX window', flush=True)
    # A recovered clean frame supersedes a failed same-burst hypothesis. Keep
    # independent handshake observations and actual original PTT evidence.
    events = [e for e in events if not (e.get('symbols') and e['status']!='CRC pass'
              and any(max(0,min(e['t1'],r['t1'])-max(e['t0'],r['t0'])) >
                      .6 * max(.01,e['t1']-e['t0']) for r in recovered))]
    return sorted(events, key=lambda e: (e['t0'], e['id']))


def source_fingerprint():
    # Hash the complete kestrel receive/coding source, not just modules which
    # happened to import before a lazy decoder branch ran.
    files = list((TREE / 'packages/hfmodem/hfmodem/kestrel').rglob('*.py'))
    files += list((TREE / 'packages/hfmodem/hfmodem/besra').rglob('*.py'))
    files += [HERE / 'ardop_view.py',TREE / 'tools/rehear/besra.py',TREE / 'packages/hfmodem/hfmodem/station/sessionlog.py']
    files += list((TREE / 'packages/hfmodem/hfmodem/core').glob('*.py'))
    files += list((TREE / 'packages/hfmodem/hfmodem/shrike').glob('*.py'))
    files += [Path(__file__), HERE / 'audio.py', HERE / 'vara_timing.py', HERE / 'vara_signal.py', HERE / 'vara_packets.py', HERE / 'pactor_view.py', HERE / 'pactor_timing.py', HERE / 'tx_timing.py', TREE / 'tools/consolelib/armsource.py']
    hashes = {str(p.relative_to(TREE)): sha(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def publish(out, final, manifest):
    """An identical concurrent build may have published this fingerprint first."""
    try:
        os.rename(out, final)
    except OSError as exc:
        if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        try:
            existing = json.loads((final / 'manifest.json').read_text())
        except (OSError, ValueError):
            raise exc
        def identity(m):
            return (m.get('schema'), m.get('arm'), m.get('recording'), m.get('bw_hz'),
                    m.get('selection'), m.get('context'), m.get('transcript_sha256'), m.get('decoder', {}).get('source_sha256'),
                    [(s.get('url'), s.get('sha256')) for s in m.get('sources', [])])
        if identity(existing) != identity(manifest):
            raise exc
        print('An identical replay is already published; using that revision.', flush=True)
    return final


def build(book, entry, out_root, recording=None, bw=None):
    protocol=entry['protocol']
    if protocol not in ('vara','pactor','ardop'):
        raise ValueError('The viewer supports VARA, PACTOR and ARDOP QSOs.')
    replay_fn=replay
    timing_fn=vara_timing.intervals
    if protocol=='pactor':
        import pactor_view, pactor_timing
        replay_fn=pactor_view.replay
        timing_fn=pactor_timing.intervals
    rec = entry.get('recording')
    if rec and not (rec.get('policy') or rec.get('lanes') or 'context' in rec):
        rec = None
    available = [r for kind in ('rig rx', 'rig tap', 'sdr witness') for r in entry['audio'] if r['kind'] == kind and r.get('url')]
    if recording:
        available = [r for r in available if r['url'] == recording]
    if not available:
        raise ValueError('This QSO has no available recording for VARA replay.')
    selected = available[0]
    from urllib.parse import unquote
    logical = lambda item: item.get('recording') or item['url']
    def resolve(item):
        rel = unquote(logical(item).removeprefix('/file/'))
        if rec:
            for root in book.roots:
                candidate = (root / rel).resolve()
                bound = any((Path(rec.get('run_directory', root))/ref['source']).resolve().with_suffix('.wav')
                            == candidate.with_suffix('.wav') for ref in rec.get('lanes', {}).values())
                if bound and candidate.is_relative_to(root.resolve()) and book.recordings(item['kind'], candidate, selection=rec):
                    return candidate
        return book.served_path(rel)
    path = resolve(selected)
    if path is None:
        raise ValueError('The selected recording is unavailable.')
    from hfmodem.station import sessionlog
    selection = sessionlog.select(rec, path) if rec else None
    ident = sessionlog.identity(rec, selection.lane) if rec else None
    root = sessionlog.source_root(rec, path) if rec else None
    ref = next((r for r in (rec or {}).get('lanes', {}).values()
                if (root / r['source']).with_suffix('.wav') == path.with_suffix('.wav')), {})
    selected_bounds = selection.bounds if selection else None
    bandwidth = bw or (selection.state.get('bandwidth_hz') if selection else None) or entry.get('bw_hz') or (2400 if protocol=='pactor' else 2000 if protocol=='ardop' else None)
    if protocol=='vara' and bandwidth not in (500, 2300, 2750):
        raise ValueError('Recording bandwidth is unknown; supply --bw 500, 2300 or 2750.')
    digest, files = source_fingerprint()
    def fingerprint(candidate):
        if not rec:
            return sha(candidate)
        chosen = sessionlog.select(rec, candidate)
        if chosen.bounds:
            return sha(candidate, bounds=[span.bounds for span in chosen.spans])
        return sha(candidate)
    source_hash = fingerprint(path)
    transcript_url = entry.get('transcript')
    transcript = book.served_path(unquote(transcript_url.removeprefix('/file/'))) if transcript_url else None
    transcript_hash = sha(transcript) if transcript and not rec else None
    transcript_text = transcript.read_text(errors='replace') if transcript and not rec else ''
    if rec:
        transcript_hash = rec.get('context', {}).get('transcript', {}).get('sha256')
        transcript = None
        if selection.complete:
            link = rec['context']['transcript']
            if 'text' in link:
                transcript_text = link['text']
            else:
                with (root / link['path']).open('rb') as stream:
                    stream.seek(link['range'][0])
                    transcript_text = stream.read(link['range'][1]-link['range'][0]).decode(errors='replace')
    from functools import partial
    if protocol=='ardop':
        import ardop_view
        log=selection.log if selection else ardop_view.sessionlog.read(transcript) if transcript else None
        replay_fn=partial(ardop_view.replay,log=log)
        timing_fn=partial(ardop_view.intervals,log=log)
    if protocol=='pactor':
        replay_fn=partial(replay_fn,transcript=transcript_text)
    p4_path=path.with_name('p4-probe.json') if not rec and protocol=='pactor' and selected['kind']=='rig rx' else None
    p4_hash=sha(p4_path) if p4_path is not None and p4_path.is_file() else None
    p4_record=json.loads(p4_path.read_text()) if p4_hash else (selection.probe if selection and selection.complete else None)
    if protocol=='pactor':
        timing_fn=partial(timing_fn,p4_record=p4_record)
    other_paths, unavailable_sources = [], []
    seen = {logical(selected)}
    for item in entry['audio']:
        if item['kind'] not in ('rig rx', 'rig tap', 'sdr witness') or not item.get('url') or logical(item) in seen:
            continue
        seen.add(logical(item))
        try:
            candidate = resolve(item)
            if candidate is None:
                raise FileNotFoundError(logical(item))
            other_paths.append((item, candidate, fingerprint(candidate)))
        except (OSError, ValueError):
            unavailable_sources.append(dict(item, state='unavailable', offset_s=None))
    options = {'sources': [(r['url'],h) for r,p,h in other_paths], 'bw': bandwidth, 'calls': [entry['call'], book.station['call']] if entry.get('local_role') == 'called' else [book.station['call'], entry['call']],
               'transcript_sha256': transcript_hash, 'session': {k: entry.get(k) for k in ('dial_hz', 'centre_hz', 'band_m', 'tx_w', 'tx_w_max', 'reached', 'mail', 'speed')},
                'local_role': entry.get('local_role', 'caller'), 'recording_window': entry.get('recording'),
                'p4_record_sha256':p4_hash, 'schema': SCHEMA}
    if rec:
        calls = selection.calls if selection.complete else {}
        local, peer = calls.get('local', ''), calls.get('peer', '')
        options['calls'] = [peer, local] if selection.local_role == 'called' else [local, peer]
        options['selection'] = ident
        options['context_complete'] = selection.complete
        options['local_role'] = selection.local_role
        options['unavailable'] = [s.get('url') for s in unavailable_sources]
    key = hashlib.sha256(json.dumps([source_hash, digest, options], sort_keys=True).encode()).hexdigest()[:20]
    arm = entry['id']
    if Path(arm).name != arm or arm in ('.', '..'):
        raise ValueError('Invalid QSO ID.')
    final = out_root / arm / key
    if (final / 'manifest.json').is_file():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging directory: a failed build cannot look complete or replace a
    # previous good revision. Retain it for diagnosis on failure.
    import tempfile
    out = Path(tempfile.mkdtemp(prefix='.building-', dir=final.parent))
    print(f'Rehearing {arm}\n  {path}\n  BW{bandwidth}; decoder source {digest[:12]}', flush=True)
    if not rec:
        x, source_rate = load_audio(path)
        tx, timing_note = timing_fn(transcript_text, path, selected['kind'], len(x)/FS)
        participating = 'connect' in (entry.get('reached') or [])
        events = replay_fn(x, bandwidth, options['calls'], out, tx=tx, decode_tx=True)
    else:
        if not selected_bounds or not ref:
            raise ValueError('Selected audio bounds are unavailable')
        lo, hi = selected_bounds
        if (hi-lo)/ref['rate'] > MAX_SECONDS:
            raise ValueError('Recording exceeds the replay duration limit.')
        spans = [(lo, hi)]
        for a, b in ref.get('gaps', []):
            spans = [(c,d) for x0,y in spans for c,d in ((x0,min(y,a)),(max(x0,b),y)) if c<d]
        x = np.zeros(round((hi-lo)*FS/ref['rate']))
        events, source_rate = [], ref['rate']
        tx, timing_note = timing_fn(transcript_text, path, selected['kind'], len(x)/FS, selection=rec)
        for a, b in spans:
            part, source_rate = load_audio(path, start=a, stop=b)
            offset = (a-lo)/ref['rate']
            first = round(offset*FS)
            x[first:first+len(part)] = part
            chosen = sessionlog.select(rec, path, start=a, stop=b)
            payload = dict(rec, bounds=[a,b], lane=chosen.lane, complete=chosen.complete,
                           state=chosen.state, changes=[c for span in chosen.spans for c in span.changes])
            if not chosen.complete:
                payload['context'] = {}
            local_tx, _ = timing_fn(transcript_text, path, selected['kind'], len(part)/FS,
                                   selection={**rec, 'lanes': {name: dict(r, selection=[a,b]) if r is ref else r
                                                            for name,r in rec['lanes'].items()}})
            kwargs = dict(tx=local_tx, decode_tx=True, selection=payload)
            with tempfile.TemporaryDirectory(dir=out, prefix='.span-') as scratch:
                found = replay_fn(part, chosen.state.get('bandwidth_hz', bandwidth),
                                  options['calls'] if chosen.complete else [], Path(scratch), **kwargs)
                for e in found:
                    e['t0'] += offset
                    e['t1'] += offset
                    e['id'] = f"{a}-" + e['id'] if len(spans)>1 else e['id']
                    if e.get('symbols'):
                        old = e['symbols']
                        name = f'symbols-{a}-' + old
                        points = json.loads((Path(scratch)/old).read_text())
                        points['t'] = [round(t+offset,6) for t in points['t']]
                        write_json(out/name, points)
                        e['symbols'] = name
            events.extend(found)
        participating = selection.complete and 'connect' in (entry.get('reached') or [])
    wf = waterfall(x, out)
    attribution = dict(qso=participating, local_role=options.get('local_role') or 'caller',
                       bounded=(entry.get('recording') or {}).get('state') == 'complete')
    vara_timing.attribute(events, tx, **attribution)
    if rec:
        a, b = ref.get('participation', selected_bounds)
        for event in events:
            sample = selected_bounds[0] + event['t0']*ref['rate']
            if not selection.complete or a is None or b is None or not a <= sample < b:
                event.update(direction='observed', role='unknown', in_qso=False,
                             attribution='Outside selected participation or context incomplete')
    events = sorted(events + tx, key=lambda e: (e['t0'], e['id']))
    sources = [{'id': 'rx', 'kind': selected['kind'], 'url': selected['url'],
                'segments': selected.get('segments'),
                'sha256': source_hash, 'offset_s': 0, 'alignment': 'Primary recording clock',
                'duration_s': len(x)/FS}]
    if rec:
        sources[0]['context'] = dict(state='complete' if selection.complete else 'incomplete', reason=selection.reason)
    sources.extend(unavailable_sources)
    audio_sources = [('rx', x, 0)]
    source_events = {'rx': list(events)}
    for number, (item, candidate, candidate_hash) in enumerate(other_paths):
        source_id = f'source{number}'
        secondary = sessionlog.select(rec, candidate) if rec else None
        try:
            if secondary is not None and secondary.bounds:
                a, b = secondary.bounds
                if (b-a)/secondary.rate > MAX_SECONDS:
                    raise ValueError('Recording exceeds the replay duration limit.')
                other = np.zeros(round((b-a)*FS/secondary.rate))
                for span in secondary.spans:
                    part, _ = load_audio(candidate, start=span.bounds[0], stop=span.bounds[1])
                    first = round((span.bounds[0]-a)*FS/secondary.rate)
                    other[first:first+len(part)] = part
            else:
                other, _ = load_audio(candidate)
        except (OSError, ValueError):
            sources.append(dict(item, state='unavailable', offset_s=None))
            continue
        secondary_payload = dict(rec, complete=secondary.complete, state=secondary.state) if rec else None
        if secondary_payload is not None and not secondary.complete:
            secondary_payload['context'] = {}
        secondary_kwargs = {'selection': secondary_payload} if rec else {}
        if selection is not None and selection.complete and secondary.complete:
            shift = selection.log.selection_offset-secondary.log.selection_offset
            alignment = 'Bound source-clock anchors'
        elif rec:
            shift, alignment = None, 'Context incomplete; unhinted, unattributed observations only.'
        else:
            shift, alignment = align_audio(x, other, events)
        if shift is None:
            with tempfile.TemporaryDirectory(dir=out, prefix='.alignment-') as scratch:
                independent = replay_fn(other, bandwidth, [] if rec else options['calls'], Path(scratch), **secondary_kwargs)
            if rec:
                for observation in independent:
                    observation.update(direction='observed', role='unknown', in_qso=False)
                sources.append(dict(id=source_id, kind=item['kind'], url=item['url'], sha256=candidate_hash,
                    offset_s=None, alignment=alignment, duration_s=len(other)/FS,
                    context=dict(state='incomplete', reason=secondary.reason), observations=independent))
                continue
            packet_shift, packet_note = align_packets(events, independent)
            if packet_shift is not None:
                shift, alignment = packet_shift, packet_note
            else:
                handshake_shift, handshake_note = align_handshakes(events, independent)
                if handshake_shift is not None:
                    shift, alignment = handshake_shift, handshake_note
                else:
                    alignment += ' '+packet_note+' '+handshake_note
        source = {'id': source_id, 'kind': item['kind'], 'url': item['url'],
                  'segments': item.get('segments'),
                  'sha256': candidate_hash, 'offset_s': shift, 'alignment': alignment,
                  'duration_s': len(other)/FS}
        if rec:
            source['context'] = dict(state='complete' if secondary.complete else 'incomplete', reason=secondary.reason)
        sources.append(source)
        print(f'  {item["kind"]}: {alignment}', flush=True)
        if shift is None:
            # Never insert an unmatched source on a guessed QSO timeline.
            continue
        keyed = [dict(e, t0=e['t0']+shift, t1=e['t1']+shift) for e in tx] if not rec else []
        import tempfile
        with tempfile.TemporaryDirectory(dir=out, prefix='.source-') as scratch:
            if secondary is not None and secondary.complete:
                from copy import deepcopy
                local = []
                for span in secondary.spans:
                    subrec = deepcopy(rec)
                    subrec['lanes'][secondary.lane]['selection'] = list(span.bounds)
                    sub = sessionlog.select(subrec, candidate)
                    payload = dict(subrec, complete=sub.complete, lane=sub.lane, bounds=list(span.bounds), state=sub.state,
                                   changes=[c for s in sub.spans for c in s.changes])
                    start = (span.bounds[0]-secondary.bounds[0])/secondary.rate
                    end = (span.bounds[1]-secondary.bounds[0])/secondary.rate
                    subtx, _ = timing_fn(transcript_text, candidate, item['kind'], end-start, selection=subrec)
                    with tempfile.TemporaryDirectory(dir=scratch, prefix='.span-') as piece:
                        found = replay_fn(other[round(start*FS):round(end*FS)],
                            sub.state.get('bandwidth_hz', bandwidth), options['calls'], Path(piece),
                            tx=subtx, decode_tx=True, selection=payload)
                        for event in found:
                            event['t0'] += start
                            event['t1'] += start
                            if event.get('symbols'):
                                old = event['symbols']
                                new = f'symbols-{span.bounds[0]}-' + old
                                points = json.loads((Path(piece)/old).read_text())
                                points['t'] = [round(t+start,6) for t in points['t']]
                                write_json(Path(scratch)/new, points)
                                event['symbols'] = new
                        local.extend(found)
            else:
                local = replay_fn(other, bandwidth, options['calls'], Path(scratch), tx=keyed, decode_tx=True, **secondary_kwargs)
            vara_timing.attribute(local, keyed, **attribution)
            for event in local:
                if secondary is not None:
                    a, b = secondary.participation or (None, None)
                    sample = secondary.bounds[0]+event['t0']*secondary.rate
                    if not secondary.complete or a is None or b is None or not a <= sample < b:
                        event.update(direction='observed', role='unknown', in_qso=False,
                                     attribution='Outside selected source participation or context incomplete')
                event['id'] = source_id+'-'+event['id']
                event['t0'] -= shift; event['t1'] -= shift
                event['source_id'] = source_id
                if event.get('symbols'):
                    old = event['symbols']; new = 'symbols-'+source_id+'-'+old
                    payload = json.loads((Path(scratch)/old).read_text())
                    payload['t'] = [round(t-shift,6) for t in payload['t']]
                    write_json(out/new,payload); event['symbols'] = new
            source_events[source_id] = local
            original_hashes = {e.get('packet_sha256') for e in events if e.get('packet_sha256') and e['direction']=='rx'}
            matching_hashes = original_hashes & {e.get('packet_sha256') for e in local}
            source['matching_packets'] = len(matching_hashes)
            source['alignment'] += f' {len(matching_hashes)} distinct CRC-valid packet payloads also match.'
        audio_sources.append((source_id, other, shift))
        source['decoded_frames'] = sum(e.get('frame_count',1) for e in local if e.get('frame_type')=='data' and e['status']=='CRC pass')
        # Default TX packet inspection comes from the local monitor, with its
        # own CRC/decision evidence. The host PTT envelope remains selectable.
        if item['kind']=='rig tap':
            for event in local:
                if event['direction']=='tx' and event.get('frame_type')=='data':
                    event['origin'] += ' · local rig monitor tap'
                    events = [old for old in events if not (old.get('frame_type')=='data' and old['direction']=='tx' and not old.get('source_id') and (event['status']=='CRC pass' or old['status']!='CRC pass') and
                        min(old['t1'],event['t1'])-max(old['t0'],event['t0']) > .5*max(.01,old['t1']-old['t0']))]
                    events.append(event)
    unassigned = [e for e in events if e.get('in_qso') is False]
    events = consolidate([e for e in events if e.get('in_qso') is not False]) + unassigned
    if p4_record and p4_record.get('sample_rate')==FS:
        for event in events:
            prior=[h for h in p4_record.get('heard',[]) if isinstance(h.get('onset_sample'),(int,float))
                   and abs(h['onset_sample']/FS-event['t0'])<.06]
            event['original_observations']=list(dict.fromkeys(
                ' '.join(str(h.get(k) or '') for k in ('kind','word','outcome')).strip()
                for h in prior))
    for index, event in enumerate(events):
        event['views'] = {}
        for source_id, audio, shift in audio_sources:
            # Point detections have no exported duration: show explicit context.
            lo, hi = event['t0'], event['t1']
            if hi-lo < .02:
                lo, hi = lo-event.get('context_before_s',.15), hi+event.get('context_after_s',1.0)
            name = f'symbols-view-{index:05d}-{source_id}.json'
            file = signal_view(audio, lo+shift, hi+shift, out, name, source_id)
            if not file:
                continue
            observed = [e for e in source_events[source_id] if e.get('frame_type')=='data' and
                        min(e['t1'],hi)-max(e['t0'],lo) > .5*max(.01,e['t1']-e['t0'])]
            matched = [e for e in source_events[source_id] if e.get('symbols') and
                       min(e['t1'],hi)-max(e['t0'],lo) > .5*max(.01,e['t1']-e['t0'])]
            event['views'][source_id] = {'signal': file, 'context': bool(event['t1']-event['t0']<.02),
                'decoded': [{k:e.get(k) for k in ('id','kind','status','t0','t1','available_at_s','payloads','combined')} for e in observed],
                'frames': [{k:e.get(k) for k in ('symbols','kind','status','carrier_hz','stage','level','snr_db','protocol','modulation','carrier_count','symbol_rate','quality','offset_hz','combined')} for e in matched]}
        event.setdefault('source_id', 'rx')
        if event['direction']=='tx':
            event['preferred_source'] = next((r['id'] for r in sources if r['kind']=='rig tap' and r['id'] in event['views']), event['source_id'])
        else:
            event['preferred_source'] = event['source_id']
    events.sort(key=lambda e:(e['t0'],e['id']))
    if protocol == 'vara':
        vara_payloads(events, entry['call'] if entry.get('local_role') == 'called' else book.station['call'])
    write_json(out / 'events.json', events)
    loaded = {name: {'file': str(Path(m.__file__).resolve()), 'sha256': sha(m.__file__)}
              for name, m in list(sys.modules.items()) if name.startswith('hfmodem.')
              and getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'}
    if any(not Path(m['file']).is_relative_to(TREE) for m in loaded.values()):
        raise ValueError('A decoder module loaded outside the viewer worktree.')
    try:
        head = subprocess.run(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    except FileNotFoundError:
        head = None
    commit = head.stdout.strip() if head and head.returncode == 0 else None
    manifest = {'schema': SCHEMA, 'arm': arm, 'call': entry['call'], 'protocol': protocol.upper(),
                'started_utc': entry['started_utc'], 'created_utc': datetime.now(timezone.utc).isoformat(),
                'recording': {'segments': selected.get('segments'), 'url': selected['url'], 'kind': selected['kind'], 'sha256': source_hash,
                              'source_rate': source_rate, 'channel': 0,
                              'source_window': entry.get('recording')},
                'sample_rate': FS, 'duration_s': len(x) / FS, 'bw_hz': bandwidth,
                'clock': 'Seconds from the first sample of this recording; not the arm start.',
                'caller': options['calls'][0],
                'called': options['calls'][1],
                'local_role': entry.get('local_role', 'caller'), 'timing': timing_note,
                'session': {k: entry.get(k) for k in ('dial_hz', 'centre_hz', 'band_m', 'tx_w', 'tx_w_max', 'reached', 'mail', 'speed')},
                'p4_record_sha256':p4_hash,
                'transcript_sha256': transcript_hash,
                'comparison': {
                    'original_summary': entry.get('summary'),
                    'logged_rx_overs': len(re.findall(r'^\s*rx DATA over #\d+', transcript_text, re.M)) if protocol=='vara' else None,
                    'replayed_rx_frames': sum(e.get('frame_count', 1) for e in events
                                             if e.get('frame_type')=='data' and e['status']=='CRC pass'
                                             and e['direction']=='rx'),
                    'note': 'Original-session overs and replayed CRC-valid data frames are different counts; replay may be incomplete.'},
                'decoder': {'commit': commit, 'source_sha256': digest, 'files': files, 'loaded': loaded},
                'byte_schema': 1, 'events': 'events.json', 'waterfall': wf, 'sources': sources,
                'limits': [('ARDOP: measured PSK/QAM symbols; FSK tone maps. Header candidates and bodyless controls are not CRC-verified data. Combining is labelled; symbols describe this transmission.' if protocol=='ardop' else 'Constellations: tone QPSK and OFDM; base index/MFSK/control packets have measured tone maps, spectra and envelopes.' if protocol=='vara' else 'PACTOR: P1 tone/control views and P3 phase constellations; P2 single-copy CRC decodes have differential PSK constellations. P4 spread control-word and robust-field constellations and normal-short BPSK bodies are decoded from audio; other P4 families retain measured signal views. P4 normal-short body geometry does not resolve SL5 versus SL6. P3 combining is explicitly labelled; plotted points describe only one transmission.'),
                           timing_note,
                           'A connected QSO has two participants. Its recovered local transmit schedule identifies local and peer frames; recording prelude and unaligned observations remain outside that assignment.',
                           'PTT, scanner and classifier evidence is grouped under physical packets; individual decoded frames within an over remain selectable.',
                           'No inferred ACK/retry relationships; replay findings do not replace the original QSO verdict.']}
    if rec:
        manifest['selection'] = ident
        manifest['context'] = dict(state='complete' if selection.complete else 'incomplete', reason=selection.reason)
        manifest['gaps'] = [[(a-selected_bounds[0])/ref['rate'], (b-selected_bounds[0])/ref['rate']]
                            for a,b in ref.get('gaps', [])]
        if not selection.complete:
            manifest['comparison']['logged_rx_overs'] = None
    stable_others = True
    for item, candidate, candidate_hash in other_paths:
        try:
            stable_others &= fingerprint(candidate) == candidate_hash
        except (OSError, ValueError):
            for source in sources:
                if source.get('url') == item.get('url'):
                    source.update(state='unavailable', offset_s=None)
    if (source_fingerprint()[0] != digest or fingerprint(path) != source_hash
            or (transcript is not None and sha(transcript) != transcript_hash)
            or (p4_hash is not None and sha(p4_path)!=p4_hash)
            or not stable_others):
        raise ValueError('Decoder source or recording changed during replay; rebuild from a stable checkout.')
    write_json(out / 'manifest.json', manifest)
    publish(out, final, manifest)
    print(f'{len(events)} events; {sum(bool(e.get("symbols")) for e in events)} constellation frames\n{final}', flush=True)
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', type=Path, action='append', required=True,
                    help='logbook data root; repeat for older trees, the first holding an arm wins')
    ap.add_argument('--arm', required=True, help='QSO ID in the logbook')
    ap.add_argument('--out', type=Path, help='artifact root, also passed to desk/face.py --viewer-root')
    ap.add_argument('--bw', type=int, choices=[500, 2300, 2750])
    ap.add_argument('--recording', help='exact /file/ URL of a recording listed for this QSO')
    ap.add_argument('--selection', type=Path)
    args = ap.parse_args(argv)
    from desk.face import Logbook
    root, *also = args.root
    book = Logbook(root, also=also)
    entry = next((e for e in book.entries()['entries'] if e['id'] == args.arm), None)
    if entry is None:
        ap.error('QSO ID is not in this logbook.')
    if args.selection:
        entry = dict(entry, recording=json.loads(args.selection.read_text()))
    try:
        build(book, entry, args.out or root / 'working' / 'rehear-viewer', args.recording, args.bw)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
