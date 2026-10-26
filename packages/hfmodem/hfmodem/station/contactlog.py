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
from datetime import datetime
from pathlib import Path
import subprocess
import tempfile
import wave

from hfmodem.core.band import AMATEUR_HF_BANDS, DIAL_OFFSET_HZ
from hfmodem.core.tx_events import read_line


def atomic_json(path, record):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def cut_audio(source, target, first, last):
    """Copy PCM frames in [first, last), with bounded memory and no resampling."""
    from hfmodem.core.wav import stored
    source, target = stored(source), Path(target)
    first, last = int(first), int(last)
    if first < 0 or last <= first:
        raise ValueError('Invalid contact recording interval')
    temporary = target.with_suffix('.wav.part')
    try:
        if source.suffix == '.flac':
            subprocess.run(['flac', '-d', '--silent', '--force', f'--skip={first}',
                            f'--until=+{last-first}', '-o', str(temporary), str(source)],
                           check=True, capture_output=True)
            with wave.open(str(temporary), 'rb') as clip:
                if clip.getnframes() != last - first:
                    raise ValueError('Contact recording is incomplete')
        else:
            with wave.open(str(source), 'rb') as src:
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


class ContactLog:
    """Publish a contact at link-up and finish it with an independent audio cutout."""
    def __init__(self, session, *, protocol='vara', application='VarAC'):
        self.session = Path(session).resolve()
        self.protocol, self.application = protocol, application
        self.starts = {}

    def update(self, contact, *, sample=None, sample_rate=48000, offset_hz=0):
        record = contact_record(self.session, contact, protocol=self.protocol, application=self.application)
        path = Path(record['directory']) / 'provenance.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            previous = json.loads(path.read_text())
            record['recording'] = previous.get('recording', {})
        else:
            record['recording'] = {}
        if sample is not None:
            if contact['n'] not in self.starts:
                self.starts[contact['n']] = sample
            record['recording'].update(source=str(self.session / 'receive.wav'),
                first_sample=self.starts[contact['n']], last_sample=sample if contact.get('ended_utc') else None,
                sample_rate=sample_rate, offset_hz=offset_hz, timing_basis='recorded callback sample positions')
        record['recording']['state'] = 'pending' if contact.get('ended_utc') else 'ongoing'
        atomic_json(path, record)
        return record

    def finish(self, record):
        """Run after the session recorder has flushed the requested samples."""
        directory = Path(record['directory'])
        recording = record['recording']
        try:
            cut_audio(recording['source'], directory / 'receive.wav',
                      recording['first_sample'], recording['last_sample'])
            target = directory / 'receive.wav'
            # Keep only this recording's TX events, shifted to the cutout clock.
            lines = [f'recording receive audio to {target}',
                     f'contact recording timing: {recording["timing_basis"]}']
            parent_log = self.session / 'arm.log'
            if parent_log.is_file():
                for line in parent_log.read_text(errors='replace').splitlines():
                    event = read_line(line)
                    if event is None or not event.get('recording'):
                        continue
                    if Path(event['recording']).with_suffix('') != Path(recording['source']).with_suffix(''):
                        continue
                    if event.get('sample_rate') != recording['sample_rate']:
                        continue
                    lo = event.get('ptt_start_sample')
                    hi = event.get('ptt_end_sample') or event.get('audio_end_sample')
                    if lo is None or hi is None or not (
                            recording['first_sample'] <= lo < hi <= recording['last_sample']):
                        continue
                    for field in ('ptt_start_sample', 'ptt_end_sample', 'audio_start_sample', 'audio_end_sample'):
                        if event.get(field) is not None:
                            event[field] -= recording['first_sample']
                    event['recording'] = str(target)
                    lines.append('tx_event ' + json.dumps(event))
            journal = self.session / 'events.jsonl'
            if journal.is_file():
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
            (directory / 'arm.log').write_text('\n'.join(lines) + '\n')
            recording['state'] = 'complete'
            recording.pop('error', None)
        except Exception as error:
            recording.update(state='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            atomic_json(directory / 'provenance.json', record)


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
            from hfmodem.core.wav import stored
            source = stored(journal.parent / 'receive.wav')
            if not source.is_file():
                native['recording'].update(state='unavailable', error='Session recording is absent')
                atomic_json(directory / 'provenance.json', native)
                continue
            if source.suffix == '.wav':
                with wave.open(str(source), 'rb') as audio:
                    rate = audio.getframerate()
                    available = audio.getnframes()
            else:
                rate, available = map(int, subprocess.run(
                    ['metaflac', '--show-sample-rate', '--show-total-samples', str(source)],
                    capture_output=True, check=True, text=True).stdout.split())
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backfill VarAC per-contact arms and recording cutouts.')
    parser.add_argument('--sessions', type=Path, required=True, help='directory containing the VarAC session journals')
    args = parser.parse_args()
    results = migrate_varac(args.sessions)
    for record in results:
        print(record['arm'], record['recording']['state'], record['recording'].get('error', ''))
    raise SystemExit(int(any(r['recording']['state'] == 'failed' for r in results)))
