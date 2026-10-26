# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Read bounded existing VARA arm evidence; no radio, audio, DSP or network.

BITRATE TX names the next over, not proof of keying. Native 'tx DATA over'
is logged only after transport success. Level coverage is not mail delivery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

MAX_FILES = 32
MAX_BYTES = 32 * 1024 * 1024
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
BW = re.compile(r'\bBW\s*(500|2300|2750)\b', re.I)
TX = re.compile(r'\btx DATA over #(\d+)(?: at record (\d+))? \((\d+) payload bytes,')
WIDE = re.compile(r'rx wideband frame:.*?reference columns(?: at record (\d+))?, CRC clean')
NARROW = re.compile(r'rx BW500 burst: (\d+) frame\(s\) at level (\d+), CRC clean')
BITRATE = re.compile(r'\bBITRATE\s*\((\d+)\)\s+\d+\s+bps\s+(TX|RX)\b')


def host_level(bw, record):
    if record is None:
        return 4
    if bw == 2750:
        return record - 99 if record in (100, 101, 102, 103) else None
    if bw == 2300:
        return record + 1
    if bw == 500:
        return 3 if record == 2 else (4 if record == 4 else None)
    return None


def analyze_lines(lines, *, bw=None, requested_level=None, direction='either', metadata=None):
    """Lines are (source, one-based line, text); metadata carries explicit facts."""
    metadata = metadata or {}
    current_bw = int(bw) if bw is not None else None
    observations, accepted, failures, ambiguities = [], [], [], []
    pending = None
    connected = False
    mail_complete = False
    previous_source = None
    for source, number, original in lines:
        if source != previous_source:
            pending = None
            current_bw = int(bw) if bw is not None else None
            previous_source = source
        line = ANSI.sub('', original)
        if match := BW.search(line):
            current_bw = int(match[1])
        evidence = dict(source=str(source), line=number, text=line.strip()[:1200])
        if ('— CONNECTED' in line or re.search(r'\bCONNECTED\s+[A-Z0-9]', line)
                or '— the link is up' in line):
            connected = True
        if '— session closed' in line or '— CONNECTED' in line or '— the link is up' in line:
            pending = None
        # `far end FF received` is `winlink/client.py`'s own report of
        # `session.remote_ff`, true only once the far end has confirmed our
        # proposal in full -- printed whether the summary calls it "accepted"
        # or "transmitted". `mail: received <MID>` is its receive-direction
        # counterpart, printed only for a message that reached `session.inbox`
        # complete. Either is a completed exchange no matter what else the
        # transcript says; `mail: out of time` names the session's overall
        # clock, not this message, and must not veto proof the message landed.
        if ('mail: exchange complete' in line or 'far end FF received' in line
                or 'mail: received ' in line):
            mail_complete = True
        if any(s in line for s in ('mail: exchange failed', 'mail: out of time',
                  'mail: the link closed before the exchange finished',
                  'DATA retransmission disabled', 'DATA over retry budget exhausted')):
            failures.append(evidence)
        if 'DATA over was not transmitted' in line:
            ambiguities.append(dict(evidence, reason='transport declined; no keyed DATA'))
        if match := TX.search(line):
            record = int(match[2]) if match[2] else None
            pending = dict(evidence, kind='tx_keyed', bw=current_bw,
                           host_level=host_level(current_bw, record), record=record,
                           over=int(match[1]), payload_bytes=int(match[3]))
            observations.append(pending)
        elif match := WIDE.search(line):
            record = int(match[1]) if match[1] else None
            observations.append(dict(evidence, kind='rx_decoded', bw=current_bw,
                                     record=record, host_level=host_level(current_bw, record)))
        elif match := NARROW.search(line):
            observations.append(dict(evidence, kind='rx_decoded', bw=500,
                                     host_level=int(match[2]), frames=int(match[1])))
        elif match := BITRATE.search(line):
            observations.append(dict(evidence, kind='tx_reported_next' if match[2] == 'TX'
                                     else 'rx_reported_decoded', bw=current_bw,
                                     host_level=int(match[1])))
        explicit_ack = re.search(r'solicited pending over #(\d+) acknowledged', line)
        ordinary_ack = 'rx over-continue from ' in line and 'the over was heard' in line
        if explicit_ack or ordinary_ack:
            if pending and (not explicit_ack or int(explicit_ack[1]) == pending['over']):
                accepted.append(dict(evidence, bw=pending['bw'], host_level=pending['host_level'],
                                     over=pending['over'], basis='local decoder recognized peer acknowledgment'))
                pending = None
            else:
                ambiguities.append(dict(evidence, reason='acknowledgment has no matching keyed DATA in this evidence'))
    by_bw = {}
    for obs in observations:
        key = str(obs['bw']) if obs['bw'] else 'unknown'
        group = by_bw.setdefault(key, dict(tx_keyed=[], tx_reported_next=[], rx_decoded=[],
                                          rx_reported_decoded=[], peer_acknowledged=[]))
        if obs['host_level'] is not None and obs['host_level'] not in group[obs['kind']]:
            group[obs['kind']].append(obs['host_level'])
    for ack in accepted:
        key = str(ack['bw']) if ack['bw'] else 'unknown'
        if ack['host_level'] is not None and ack['host_level'] not in by_bw[key]['peer_acknowledged']:
            by_bw[key]['peer_acknowledged'].append(ack['host_level'])
    for group in by_bw.values():
        for values in group.values():
            values.sort()
    eligible = {'tx_keyed'} if direction == 'tx' else {'rx_decoded', 'rx_reported_decoded'}
    if direction == 'either':
        eligible.add('tx_keyed')
    exercised = [o for o in observations if o['kind'] in eligible
                 and (bw is None or o['bw'] == int(bw)) and o['bw'] is not None
                 and o['host_level'] == requested_level]
    cleanup = metadata.get('cleanup_complete')
    if not isinstance(cleanup, bool):
        cleanup = None
        if metadata.get('cleanup_returncode') == 0 and str(metadata.get('final_ptt')) == '0':
            cleanup = True
        elif metadata.get('cleanup_returncode') not in (None, 0):
            cleanup = False
    # A confirmed completion outranks a bare failure phrase: `mail: out of time`
    # is the session's own clock and says nothing about a message the far end
    # already confirmed receiving in full.
    mail_status = ('complete' if mail_complete
                    else 'failed_or_incomplete' if failures else 'unknown')
    return dict(schema_version=1, requested_bw=bw, requested_level=requested_level,
                direction=direction, requested_level_status=('not_requested' if requested_level is None
                else 'exercised' if exercised else 'not_exercised'), by_bw=by_bw,
                observations=observations, peer_acknowledgments=accepted, connected=connected,
                mail_status=mail_status, cleanup_complete=cleanup,
                completed_with_cleanup=mail_status == 'complete' and cleanup is True,
                failures=failures, ambiguities=ambiguities,
                limits=['TX BITRATE reports the next over and alone never proves keying.',
                        'Peer acceptance is local acknowledgment recognition, not independent peer host proof.',
                        'Connected, keyed, or level exercised never means mail completed.',
                        'No captured waveform or RF signal is analyzed.'])


def analyze(path, *, bw=None, requested_level=None, direction=None):
    path = Path(path).resolve()
    pending = [path] if path.is_file() else sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix in ('.json', '.log', '.txt'))
    if len(pending) > MAX_FILES:
        raise ValueError('Too many input files; select one arm manifest or log')
    seen, lines, metadata, size = set(), [], {}, 0
    while pending:
        p = pending.pop(0).resolve()
        if p in seen:
            continue
        seen.add(p)
        if len(seen) > MAX_FILES:
            raise ValueError('Input exceeds file-count bound')
        size += p.stat().st_size
        if size > MAX_BYTES:
            raise ValueError('Input exceeds32MiB bound')
        text = p.read_text(errors='replace')
        if p.suffix != '.json':
            lines.extend((p, n, s) for n, s in enumerate(text.splitlines(), 1))
            continue
        data = json.loads(text)
        if not isinstance(data, dict):
            continue
        data = dict(data.get('plan', {}), **data)
        for key in ('bw_hz', 'bw', 'requested_level', 'direction', 'cleanup_complete', 'cleanup_returncode', 'final_ptt'):
            if key in data:
                if key in metadata and metadata[key] != data[key]:
                    raise ValueError(f'Conflicting manifest {key}; select one arm')
                metadata[key] = data[key]
        for key in ('log_path', 'transcript', 'transcript_path', 'log'):
            if isinstance(data.get(key), str):
                linked = Path(data[key])
                if not linked.is_absolute():
                    local = p.parent / linked
                    linked = local if local.exists() else Path.cwd() / linked
                pending.append(linked)
        # Stock result timelines contain [time, endpoint, text]. Do not infer
        # endpoint identity or parse observations from unrelated JSON strings.
        for n, row in enumerate(data.get('timeline', []), 1):
            if isinstance(row, list) and len(row) >= 3 and row[1] == 'caller':
                lines.append((str(p) + ':timeline', n, str(row[2])))
    if bw is None:
        bw = metadata.get('bw_hz', metadata.get('bw'))
    if requested_level is None:
        requested_level = metadata.get('requested_level')
    if direction is None:
        direction = metadata.get('direction', 'either')
    if direction not in ('tx', 'rx', 'either'):
        raise ValueError(f'Invalid direction in evidence: {direction}')
    result = analyze_lines(lines, bw=int(bw) if bw else None,
                           requested_level=int(requested_level) if requested_level is not None else None,
                           direction=direction, metadata=metadata)
    result['input_files'] = sorted(str(p) for p in seen)
    result['input_bytes'] = size
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--bw', type=int, choices=(500, 2300, 2750))
    parser.add_argument('--requested-level', type=int, choices=range(1, 18))
    parser.add_argument('--direction', choices=('tx', 'rx', 'either'),
                        help='Default: manifest direction, otherwise either')
    args = parser.parse_args(argv)
    try:
        report = analyze(args.path, bw=args.bw, requested_level=args.requested_level, direction=args.direction)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
