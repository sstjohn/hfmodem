# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Read shared modem transmit events on the recording's own sample clock."""
from pathlib import Path

from hfmodem.core.tx_events import read_line

# Outcomes whose carrier went up; a late or cut burst is still where it was keyed.
KEYED = frozenset(('sent', 'late', 'interrupted'))


def intervals(text, recording, kind, duration, protocol, *, selection=None):
    if selection is not None:
        from hfmodem.station import sessionlog
        selected = sessionlog.select(selection, recording)
        if not selected.complete:
            return []
        out = []
        for tx in selected.tx:
            if not tx.get('protocol', protocol).startswith(protocol):
                continue
            event = dict(tx, id=f"ptt-{tx.get('tx_id')}",
                source_offset_sample=selected.bounds[0], role=selected.local_role,
                kind=tx.get('label') or tx.get('packet_kind') or 'Local transmission',
                evidence_type='ptt', direction='tx', tx_confirmed=True,
                status='logged PTT interval', symbols=None, origin=f'original {protocol} transmit event',
                detail=tx.get('label', ''), attribution='confirmed local transmission')
            out.append(event)
        return out
    if kind != 'rig rx':
        return []
    records = [r for line in text.splitlines() if (r := read_line(line)) is not None
               and r.get('outcome') in KEYED and r.get('protocol', '').startswith(protocol)]
    key = lambda r: (r.get('session_id'), r.get('tx_id'))
    metadata = {key(r): r for r in records if r['event'] == 'waveform'}
    out = []
    for r in records:
        if r['event'] != 'completed' or not r.get('recording'):
            continue
        if Path(r['recording']).with_suffix('') != Path(recording).with_suffix(''):
            continue
        a = r.get('ptt_start_sample')
        b = r.get('ptt_end_sample') or r.get('audio_end_sample')
        rate = r.get('sample_rate')
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (a, b, rate)):
            continue
        if rate <= 0 or b <= a or a/rate >= duration or b <= 0:
            continue
        meta = r | metadata.get(key(r), {})
        event = {'id': f"ptt-{r.get('session_id') or 'legacy'}-{r['tx_id']}", 'tx_id': r['tx_id'], 'session_id': r.get('session_id'),
                 't0': max(0, a/rate), 't1': min(duration, b/rate),
                 'kind': meta.get('label') or meta.get('packet_kind') or 'Local transmission',
                 'evidence_type': 'ptt', 'direction': 'tx', 'role': 'caller', 'tx_confirmed': True,
                 'status': 'logged PTT interval' + ('' if r['outcome'] == 'sent' else f" ({r['outcome']})"),
                 'symbols': None, 'origin': f'original {protocol} transmit event',
                 'detail': meta.get('label', ''), 'attribution': 'confirmed local transmission',
                 'timing_precision': r.get('timing_basis', 'host sample positions; not measured RF')}
        event.update({k: meta[k] for k in ('protocol', 'modulation', 'symbol_rate', 'carrier_count',
                                         'packet_kind', 'bandwidth_hz', 'link_state', 'retry', 'over_id') if k in meta})
        if r.get('timestamp_limitation') or r.get('driver_offset_provenance') == 'nominal':
            event.update({k: r[k] for k in ('timestamp_limitation', 'driver_offset_provenance',
                                            'driver_offset_samples') if k in r})
            event['timing_precision'] += '; estimated sample timing'
            if r.get('driver_offset_provenance') == 'nominal':
                event['timing_precision'] += '; nominal driver offset'
            if r.get('timestamp_limitation'):
                event['timing_precision'] += '; ' + r['timestamp_limitation']
        if meta.get('speed_level') is not None:
            event['level'] = meta['speed_level']
        out.append(event)
    return out


def merge(explicit, legacy):
    # A confirmed explicit key interval replaces its older prose reconstruction.
    # PACTOR's prose grid can describe the slot boundary rather than the later
    # key edge. Match emission identity before comparing estimated timestamps.
    ids = {t['tx_id'] for t in explicit if t.get('tx_id') is not None}
    return sorted(explicit + [e for e in legacy if not any(
        abs(e['t0']-t['t0']) < .15 and abs(e['t1']-t['t1']) < .3 for t in explicit)
        and not (e.get('origin') == 'original PACTOR session grid'
                 and e['id'].removeprefix('ptt-').isdigit()
                 and int(e['id'][4:]) in ids)],
        key=lambda e: e['t0'])
