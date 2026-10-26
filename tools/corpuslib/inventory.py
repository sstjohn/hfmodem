# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Rebuild a capture inventory from durable visit metadata and event logs."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import wave

from .policy import READABLE


def build(directory):
    directory = Path(directory)
    try:
        run_status = json.loads((directory / 'run-status.json').read_text())
    except (OSError, ValueError):
        run_status = {}
    from hfmodem.core import capture_identity as ci
    from hfmodem.core import wav as audio_files
    visits = []
    paths = set(directory.glob('[0-9]*-*/session.json'))
    # New identity roots are discovered by their manifests, never inferred from a minute label.
    paths.update(path for path in directory.glob('*/session.json')
                 if not path.parent.is_symlink() and (path.parent / ci.MANIFEST).is_file())
    if (directory / ci.MANIFEST).is_file() and (directory / 'session.json').is_file():
        paths.add(directory / 'session.json')
    for path in sorted(paths):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        problems = []
        try:
            reference = ci.evidence_reference(path)
            metadata = ci.metadata_record(path) if reference.get('capture_id') else json.loads(path.read_text())
            if not isinstance(metadata, dict):
                raise ValueError('session metadata must be an object')
        except (OSError, ValueError) as exc:
            metadata = {}
            problems.append(f'unreadable session metadata: {type(exc).__name__}')
        counts = Counter()
        first_p4 = None
        events = path.parent / 'events.jsonl'
        if events.exists():
            with events.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                        if event.get('event') != 'detection':
                            continue
                        kind = event['kind']
                        counts[kind] += 1
                        if kind.startswith('P4-') and first_p4 is None:
                            first_p4 = event.get('t')
                    except (ValueError, KeyError, TypeError, AttributeError):
                        if 'incomplete event log' not in problems:
                            problems.append('incomplete event log')
        wavs = []
        seconds = 0.
        size = 0
        recordings = {audio.stem: audio for audio in sorted(path.parent.glob('rx-*.flac'))}
        recordings.update({audio.stem: audio for audio in sorted(path.parent.glob('rx-*.wav'))})
        recording_references = []
        for wav in sorted(recordings.values()):
            if wav.is_symlink():
                problems.append('symlink recording refused')
                continue
            try:
                stored_path = audio_files.stored(wav)
                source = audio_files.source(wav)
                recording_references.append(ci.evidence_reference(stored_path))
            except (OSError, ValueError) as exc:
                problems.append('recording identity: ' + type(exc).__name__)
                continue
            size += wav.stat().st_size
            wavs.append(str(wav.relative_to(directory)))
            try:
                with wave.open(source, 'rb') as audio:
                    seconds += audio.getnframes() / audio.getframerate()
            except (OSError, EOFError, wave.Error, ZeroDivisionError):
                problems.append(f'unreadable WAV header: {wav.name}')
        if not metadata.get('ended_utc'):
            problems.append('visit unfinished or still running')
        if metadata.get('continuous') is False:
            problems.append('audio continuity check failed')
        if metadata.get('detector_drained') is False:
            problems.append('detector did not finish its queued audio')
        if metadata.get('detector_dropped_samples', 0):
            problems.append('detector dropped samples')
        analyzed = metadata.get('analyzed_through_s')
        if analyzed is not None and analyzed + .1 < seconds:
            problems.append('detector did not analyze all saved audio')
        if metadata.get('queue_errors'):
            problems.append('recording queue errors')
        reason = metadata.get('reason', 'unfinished')
        if reason.startswith(('error:', 'maximum-dwell')):
            problems.append(reason)
        visits.append(dict(directory=path.parent.name,
            center_hz=metadata.get('channel', {}).get('center_hz'),
            started_utc=metadata.get('started_utc'), ended_utc=metadata.get('ended_utc'),
            audio_seconds=round(seconds, 3), audio_bytes=size, files=wavs,
            detections=dict(counts), readable_detections=sum(counts[k] for k in READABLE),
            p4_crc_chirps=counts['P4-CHIRP-CRC'],
            p4_candidates=counts['P4-HEADER'] + counts['P4-CHIRP-HEADER'],
            first_p4_seconds=first_p4, reason=reason, problems=problems))
        if metadata.get('capture_id'):
            visits[-1]['capture_identity'] = reference
            visits[-1]['recording_identities'] = recording_references
    return dict(schema=2 if any('capture_identity' in visit for visit in visits) else 1, generated_utc=datetime.now(timezone.utc).isoformat(), run_status=run_status,
        directory=str(directory.resolve()), visits=visits,
        totals=dict(visits=len(visits), channels=len({v['center_hz'] for v in visits if v['center_hz']}),
            audio_seconds=round(sum(v['audio_seconds'] for v in visits), 3),
            audio_bytes=sum(v['audio_bytes'] for v in visits),
            wav_files=sum(len(v['files']) for v in visits),
            readable_visits=sum(v['readable_detections'] > 0 for v in visits),
            p4_crc_chirps=sum(v['p4_crc_chirps'] for v in visits),
            p4_candidates=sum(v['p4_candidates'] for v in visits),
            problem_visits=sum(bool(v['problems']) for v in visits)))


def render(report, limit=None):
    totals = report['totals']
    lines = ['Capture inventory',
        f"{totals['visits']} visits on {totals['channels']} channels; "
        f"{totals['audio_seconds']/3600:.2f} hours recorded in {totals['wav_files']} WAVs "
        f"({totals['audio_bytes']/1e9:.3f} GB).",
        f"{totals['readable_visits']} visits with readable PACTOR; "
        f"{totals['p4_crc_chirps']} CRC-valid P4 entry chirps; "
        f"{totals['p4_candidates']} tentative P4 header detections.",
        'Header detections may include a chirp later validated by CRC; these are not session counts.',
        f"{totals['problem_visits']} visits with capture/analysis issues or a maximum-dwell ending.",
        '', 'Visit / center Hz / audio seconds / P4 CRC / P4 tentative / notes']
    if report.get('run_status', {}).get('status') == 'failed':
        lines[0] = 'Capture inventory — collection FAILED'
        lines.insert(1, report['run_status'].get('error', 'Unknown collection error'))
    if not totals['visits']:
        lines.insert(1, 'No channel visits were recorded.')
    ordered = sorted(report['visits'], key=lambda v: (
        -v['p4_crc_chirps'], -v['p4_candidates'], -v['readable_detections'], v['directory']))
    for visit in ordered[:limit]:
        note = '; '.join(visit['problems']) or visit['reason']
        lines.append(f"{visit['directory']} / {visit['center_hz']} / {visit['audio_seconds']:.1f} / "
                     f"{visit['p4_crc_chirps']} / {visit['p4_candidates']} / {note}")
        if limit is None:
            lines.extend(f'  {name}' for name in visit['files'])
    if limit is not None and len(ordered) > limit:
        lines.append(f'... {len(ordered)-limit} further visits in inventory.txt and inventory.json')
    lines.append(f"Recordings and full inventory: {report['directory']}")
    return '\n'.join(lines) + '\n'


def write(directory, *, display=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'capture directory does not exist: {directory}')
    report = build(directory)
    for name, content in [('inventory.json', json.dumps(report, indent=2, allow_nan=False)+'\n'),
                          ('inventory.txt', render(report))]:
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.inventory-',
                                         delete=False) as temporary:
            temporary.write(content)
        Path(temporary.name).replace(directory / name)
    if display:
        print('\n' + render(report, limit=20), flush=True)
    return report
