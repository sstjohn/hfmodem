# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Save attended contacts as ordinary arms, with their own recording and transcript.

Session journals remain diagnostic evidence. The per-contact provenance is the
same record the station logbook already reads for VARA, ARDOP and PACTOR arms.
No devices or decoders are opened here.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import shutil
import tempfile
import threading
import wave

from hfmodem.core.band import AMATEUR_HF_BANDS, DIAL_OFFSET_HZ
from hfmodem.core.capture_identity import _absolute
from hfmodem.core.tx_events import read_line


def atomic_json(path, record):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cut_audio(source, target, first, last):
    """Copy [first, last) into outputs bounded by native segment capacity."""
    from hfmodem.core import wav
    from hfhost.recordings import audio_info, indexed, offset_segments
    source, target = Path(source), Path(target)
    first, last = int(first), int(last)
    if first < 0 or last <= first:
        raise ValueError('Invalid contact recording interval')
    parts = offset_segments(source)
    if parts is None:
        parts, _ = indexed(source, wav.SEGMENT_SAMPLES)
    rate = None
    for part in parts.values():
        try:
            rate, _ = audio_info(part)
            break
        except FileNotFoundError:
            continue
    if rate is None:
        raise ValueError('Contact recording is not yet completely written')
    capacity = max(1, wav.SEGMENT_SAMPLES * rate // 48000)
    if last - first <= capacity:
        return _cut_audio_interval(source, target, first, last)
    writer = wav.SegmentedWriter(target, rate=rate, closed=lambda path: None)
    try:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            for start in range(first, last, capacity):
                clip = Path(tmp) / 'clip.wav'
                _cut_audio_interval(source, clip, start, min(last, start + capacity))
                with wave.open(str(clip), 'rb') as src:
                    if (src.getnchannels(), src.getsampwidth(), src.getframerate()) != (1, 2, rate):
                        raise ValueError('Contact recording must be mono int16 PCM at its native rate')
                    while block := src.readframes(32768):
                        writer.writeframes(block)
    finally:
        writer.close()


def _cut_audio_interval(source, target, first, last):
    from hfmodem.core import wav
    source, target = Path(source), Path(target)
    first, last = int(first), int(last)
    if first < 0 or last <= first:
        raise ValueError('Invalid contact recording interval')
    from hfhost.recordings import audio_info, indexed, offset_segments
    absolute = offset_segments(source)
    if absolute is not None:
        parts, capacity = absolute, 1
    else:
        parts, capacity = indexed(source, wav.SEGMENT_SAMPLES)
    if capacity is None:
        return _cut_audio_file(parts[0], target, first, last)
    temporary = target.with_suffix('.wav.part')
    try:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp, wave.open(str(temporary), 'wb') as dst:
            dst.setnchannels(1)
            dst.setsampwidth(2)
            dst.setframerate(48000)
            position = first
            params = None
            while position < last:
                if absolute is not None:
                    index = next((start for start in reversed(parts) if start <= position), None)
                    if index is None:
                        raise ValueError('Contact recording is not yet completely written')
                    offset = position - index
                    count = min(last - position, audio_info(parts[index])[1] - offset)
                    if count <= 0:
                        raise ValueError('Contact recording is not yet completely written (gap)')
                else:
                    index, offset = divmod(position, capacity)
                    count = min(last - position, capacity - offset)
                if index not in parts:
                    raise ValueError('Contact recording is not yet completely written')
                clip = Path(tmp) / 'clip.wav'
                try:
                    _cut_audio_file(parts[index], clip, offset, offset + count)
                except FileNotFoundError:
                    raise ValueError('Contact recording is not yet completely written') from None
                with wave.open(str(clip), 'rb') as src:
                    current = (src.getnchannels(), src.getsampwidth(), src.getframerate())
                    if params is None:
                        dst.setparams(src.getparams()._replace(nframes=0))
                        params = current
                    elif params != current:
                        raise ValueError('Contact recording segments have different formats')
                    remaining = count
                    while remaining:
                        chunk = min(remaining, 32768)
                        pcm = src.readframes(chunk)
                        if len(pcm) != chunk * src.getnchannels() * src.getsampwidth():
                            raise ValueError('Contact recording ended early')
                        dst.writeframes(pcm)
                        remaining -= chunk
                position += count
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _cut_audio_file(source, target, first, last):
    from hfhost.recordings import open_segment
    with open_segment(source) as stream:
        return _cut_audio_resolved(stream, target, first, last)


def _cut_audio_resolved(source, target, first, last):
    target = Path(target)
    first, last = int(first), int(last)
    if first < 0 or last <= first:
        raise ValueError('Invalid contact recording interval')
    temporary = target.with_suffix('.wav.part')
    try:
        if Path(source.name).suffix == '.flac':
            with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
                encoded = Path(tmp) / 'source.flac'
                with encoded.open('wb') as out:
                    shutil.copyfileobj(source, out)
                subprocess.run(['flac', '-d', '--silent', '--force', f'--skip={first}',
                                f'--until=+{last-first}', '-o', str(temporary), str(encoded)],
                               check=True, capture_output=True)
            with wave.open(str(temporary), 'rb') as clip:
                if clip.getnframes() != last - first:
                    raise ValueError('Contact recording is incomplete')
        else:
            with wave.open(source, 'rb') as src:
                if last > src.getnframes():
                    raise ValueError('Contact recording is not yet completely written')
                src.setpos(first)
                with wave.open(str(temporary), 'wb') as dst:
                    dst.setparams(src.getparams()._replace(nframes=0))
                    remaining = last - first
                    while remaining:
                        count = min(remaining, 32768)
                        pcm = src.readframes(count)
                        if len(pcm) != count * src.getnchannels() * src.getsampwidth():
                            raise ValueError('Contact recording ended early')
                        dst.writeframes(pcm)
                        remaining -= count
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _audio_frames(path):
    from hfhost.recordings import audio_info
    return audio_info(path)


def contact_record(session, contact, *, protocol='vara', application='VarAC'):
    """The common arm schema, independent of the attended application's journal."""
    session = Path(session).resolve()
    name = f'{protocol}-chat-{session.name}-{int(contact["n"])}'
    directory = session.parent.parent / 'arms' / name
    dial = contact['dial_hz']
    inherited = {}
    try:
        launch = json.loads((session / 'provenance.json').read_text())
        inherited = {k: launch[k] for k in ('git', 'env', 'residency', 'readings', 'console_argv',
                                           'command', 'locator') if k in launch}
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return inherited | dict(arm=name, directory=str(directory), protocol=protocol, application=application,
                call=contact['peer'], mode='chat',
                band_m=next((m for lo, hi, m in AMATEUR_HF_BANDS if lo * 1e6 <= dial <= hi * 1e6), None),
                dial_hz=dial, centre_hz=dial + DIAL_OFFSET_HZ,
                started_utc=contact['started_utc'], ended_utc=contact.get('ended_utc'),
                local_role='called' if contact['how'] == 'incoming' else 'caller',
                level={'name': 'attended', 'note': 'per-contact recording from an attended session'},
                peer={'call': contact['peer']},
                adjustments={'bw_hz': contact['bandwidth_hz']},
                verdict={'reached': ['connect'], 'line': contact.get('line')},
                source={'session': str(session), 'contact_number': contact['n']})


def _index(recording, directory):
    return Path(os.path.relpath(_absolute(recording.directory / 'recording.json'), _absolute(directory))).as_posix()


class ContactLog:
    """Publish contact metadata, selected audio and clipped transcript events."""
    def __init__(self, session, *, protocol='vara', application='VarAC', recording=None, mycall=None):
        self.session = Path(session).resolve()
        self.protocol, self.application = protocol, application
        self.starts = {}
        self.source_ids = {}
        self.recording = recording
        self.mycall = mycall
        self.records = {}
        self.ended = set()
        self._lock = threading.RLock()
        if recording is not None:
            recording.publish = self.publish

    def update(self, contact, *, sample=None, sample_rate=48000, offset_hz=0, recording_id=None):
        with self._lock:
            return self._update(contact, sample=sample, sample_rate=sample_rate, offset_hz=offset_hz,
                                recording_id=recording_id)

    def _update(self, contact, *, sample, sample_rate, offset_hz, recording_id):
        if recording_id is not None:
            self.source_ids.setdefault(contact["n"], recording_id)
        recording_id = self.source_ids.get(contact["n"])
        positions = ({lane.clock: lane.position for lane in self.recording.lanes.values()}
                     if self.recording is not None else {})
        if self.recording is not None and sample is not None and 'rx' in self.recording.lanes:
            positions[self.recording.lanes['rx'].clock] = sample
        if self.recording is not None:
            from hfmodem.core.recording import lane_positions
            positions = lane_positions(self.recording, positions)
        record = contact_record(self.session, contact, protocol=self.protocol, application=self.application)
        if recording_id is not None:
            record.update(self._source_location(recording_id))
        path = _absolute(record['directory']) / 'provenance.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            previous = json.loads(path.read_text())
            record['recording'] = previous.get('recording', {})
        else:
            record['recording'] = {}
        published = (dict(record['recording']) if recording_id is not None
                     and 'end_positions' in record['recording'] else {})
        if sample is not None:
            if contact['n'] not in self.starts:
                self.starts[contact['n']] = sample
            record['recording'].update(source=str(self.session / 'receive.wav'),
                first_sample=self.starts[contact['n']], last_sample=sample if contact.get('ended_utc') else None,
                sample_rate=sample_rate, offset_hz=offset_hz, timing_basis='recorded callback sample positions')
        record['recording']['state'] = 'pending' if contact.get('ended_utc') else 'ongoing'
        if self.recording is not None:
            cid = recording_id or f'{self.protocol}-{contact["n"]}'
            local = contact.get('mycall', self.mycall)
            for name, lane in self.recording.lanes.items():
                if lane.clock in positions:
                    self.recording.state(name, positions[lane.clock],
                        protocol=self.protocol.upper(), calls=dict(local=local, peer=contact['peer']),
                        local_role=record['local_role'], bandwidth_hz=contact['bandwidth_hz'],
                        **{field: contact[field] for field in ('expected_session', 'arq_epoch') if field in contact})
            record['recording'].update(policy=self.recording.policy, id=cid,
                                       run_directory=str(self.recording.directory),
                                       index=_index(self.recording, record['directory']))
            rx = self.recording.lanes.get('rx')
            if rx is not None and rx.row.get('source'):
                record['recording']['source'] = str(self.recording.directory / rx.row['source'])
            first = cid not in self.records
            record['recording'].update(published)
            self.records[cid] = record
            atomic_json(path, record)
            if first and recording_id is None:
                self.recording.begin(cid, positions, peer=contact['peer'],
                                     **({'context': contact['context']} if 'context' in contact else {}))
            elif first and contact.get('context'):
                self.recording.enrich(cid, context=contact['context'])
            if recording_id is None and contact.get('ended_utc') and cid not in self.ended:
                self.ended.add(cid)
                self.recording.end(cid, positions)
        else:
            atomic_json(path, record)
        return record

    def _source_location(self, cid, protocol=None):
        name = f'{protocol or self.protocol}-link-{self.session.name}-{cid}'
        return dict(arm=name, directory=str(self.session.parent.parent / 'arms' / name))

    def publish(self, selected):
        with self._lock:
            record = self.records.get(selected['id'])
            if record is None:
                rx = self.recording.lanes.get('rx')
                key = selected.get('key')
                protocol = key[1] if key is not None else self.protocol
                record = dict(self._source_location(selected['id'], protocol), protocol=protocol,
                              application=self.application, mode='link', call=selected.get('peer', ''),
                              peer={'call': selected.get('peer', '')}, started_utc=None,
                              ended_utc=None, source={'session': str(self.session)},
                              recording=dict(policy=self.recording.policy,
                                  run_directory=str(self.recording.directory),
                                  sample_rate=rx.rate if rx is not None else None))
                self.records[selected['id']] = record
                _absolute(record['directory']).mkdir(parents=True, exist_ok=True)
                record['recording']['index'] = _index(self.recording, record['directory'])
            if record.get('ended_utc') is None:
                record['ended_utc'] = datetime.now(timezone.utc).isoformat()
            if record['recording'].get('last_sample') is None:
                rx = self.recording.lanes.get('rx')
                if rx is not None:
                    record['recording']['last_sample'] = selected.get('end_positions', {}).get(rx.clock)
            record['recording'].update(selected)
            self.finish(record)

    def finish(self, record):
        """Run after the session recorder has flushed the requested samples."""
        directory = Path(record['directory'])
        recording = record['recording']
        try:
            legacy = 'policy' not in recording
            if legacy:
                cut_audio(recording['source'], _absolute(directory / 'receive.wav'),
                          recording['first_sample'], recording['last_sample'])
            target = directory / 'receive.wav' if legacy else None
            # Contact display positions retain the original source bounds.
            lines = ([f'recording receive audio to {target}'] if legacy else
                     [f'recording policy: {recording["policy"]}'])
            if recording.get('timing_basis'):
                lines.append(f'contact recording timing: {recording["timing_basis"]}')
            parent_log = self.session / 'arm.log'
            if parent_log.is_file():
                for line in parent_log.read_text(errors='replace').splitlines():
                    event = read_line(line)
                    if event is None or not event.get('recording'):
                        continue
                    if not recording.get('source') or Path(event['recording']).with_suffix('') != Path(recording['source']).with_suffix(''):
                        continue
                    if event.get('sample_rate') != recording['sample_rate']:
                        continue
                    lo = event.get('ptt_start_sample')
                    hi = event.get('ptt_end_sample')
                    if lo is None:
                        lo = event.get('audio_start_sample')
                    if hi is None:
                        hi = event.get('audio_end_sample')
                    first, last = recording.get('first_sample'), recording.get('last_sample')
                    if (lo is None or hi is None or first is None or last is None
                            or max(first, lo) >= min(last, hi)):
                        continue
                    event['original_bounds'] = {field: event[field] for field in
                        ('ptt_start_sample', 'ptt_end_sample', 'audio_start_sample', 'audio_end_sample')
                        if event.get(field) is not None}
                    for field in ('ptt_start_sample', 'ptt_end_sample', 'audio_start_sample', 'audio_end_sample'):
                        if event.get(field) is not None:
                            event[field] = min(last, max(first, event[field])) - first
                    if target is not None:
                        event['recording'] = str(target)
                    else:
                        event['source_first_sample'] = first
                    lines.append('tx_event ' + json.dumps(event))
            journal = self.session / 'events.jsonl'
            if journal.is_file() and record.get('started_utc'):
                for raw in journal.read_text().splitlines():
                    try:
                        event = json.loads(raw)
                        at = datetime.fromisoformat(event['utc'])
                    except (ValueError, KeyError):
                        continue
                    if not (datetime.fromisoformat(record['started_utc']) <= at
                            <= datetime.fromisoformat(record['ended_utc'])):
                        continue
                    lines.append('application_event ' + json.dumps(event, ensure_ascii=False))
            _absolute(directory / 'arm.log').write_text('\n'.join(lines) + '\n')
            if legacy:
                recording['state'] = 'complete'
            recording.pop('error', None)
        except Exception as error:
            recording.update(state='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            atomic_json(_absolute(directory / 'provenance.json'), record)


def migrate_varac(sessions):
    """Backfill standard arms once, without modifying any original session evidence.

    ``sessions`` is the explicit operator-selected session directory.
    Old journals have UTC timestamps but no callback sample positions. Their
    cutouts use an explicitly approximate host-start anchor. New captures always
    record sample positions instead. Incomplete links remain without a cutout.
    """
    results = []
    for journal in sorted(Path(sessions).resolve().glob('*/contacts.jsonl')):
        latest = {}
        for line in journal.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get('kind') == 'contact':
                latest[record['n']] = record
        host = journal.parent / 'host.jsonl'
        anchor = None
        if host.is_file():
            for line in host.read_text().splitlines():
                try:
                    anchor = datetime.fromisoformat(json.loads(line)['utc'])
                    break
                except (ValueError, KeyError):
                    continue
        writer = ContactLog(journal.parent)
        for contact in latest.values():
            native = contact_record(journal.parent, contact)
            directory = Path(native['directory'])
            existing = directory / 'provenance.json'
            if existing.exists():
                previous = json.loads(existing.read_text())
                if (not previous.get('source', {}).get('backfilled')
                        or (previous.get('recording', {}).get('state') == 'complete'
                            and Path(previous['recording']['source']).is_absolute())):
                    continue
            native = writer.update(contact)
            native['source']['backfilled'] = True
            results.append(native)
            if not contact.get('ended_utc') or anchor is None:
                native['recording'].update(state='unavailable', error='No closed contact or recording clock anchor')
                atomic_json(directory / 'provenance.json', native)
                continue
            from hfmodem.core.wav import segments
            source = journal.parent / 'receive.wav'
            parts = segments(source)
            from hfhost.recordings import open_segment
            try:
                for part in parts:
                    with open_segment(part):
                        pass
            except FileNotFoundError:
                native['recording'].update(state='unavailable', error='Session recording is absent')
                atomic_json(directory / 'provenance.json', native)
                continue
            available = 0
            rate = None
            for part in parts:
                current_rate, frames = _audio_frames(part)
                if rate is not None and rate != current_rate:
                    raise ValueError('Recording segments have different sample rates')
                rate = current_rate
                available += frames
            first = max(0, round((datetime.fromisoformat(contact['started_utc']) - anchor).total_seconds() * rate))
            last = round((datetime.fromisoformat(contact['ended_utc']) - anchor).total_seconds() * rate)
            native['recording'].update(source=str(source), first_sample=first, last_sample=min(last, available),
                estimated_last_sample=last,
                sample_rate=rate, offset_hz=0,
                timing_basis='approximate: UTC relative to first host event; audio startup latency unmeasured')
            try:
                writer.finish(native)
            except (OSError, ValueError, subprocess.CalledProcessError):
                # A missing/bad source must not prevent other contacts being
                # recovered. finish persisted the failure beside this QSO.
                continue
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backfill VarAC per-contact arms and recording cutouts.')
    parser.add_argument('--sessions', type=Path, required=True, help='directory containing the VarAC session journals')
    args = parser.parse_args()
    results = migrate_varac(args.sessions)
    for record in results:
        print(record['arm'], record['recording']['state'], record['recording'].get('error', ''))
    return int(any(r['recording']['state'] == 'failed' for r in results))
