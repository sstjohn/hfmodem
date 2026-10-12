"""Map VARA host PTT events onto the exact receiver recording they counted.

Capture samples are counted at the same callback which queues the rig RX WAV.
Use nearby sample/monotonic anchors, never buffer cursors or wall-clock filename
subtraction. These are software timings, not measured acoustic/PTT edges.
"""
import re
import tx_timing
from pathlib import Path

ANSI = re.compile(r'\x1b\[[0-9;]*m')
RECORDING = re.compile(r'recording receive audio to (\S+)')
FIELDS = re.compile(r'(\w+)=([^\s]+)')
EDGE_MARGIN = .12  # boundary guard, not a claim of measured timing accuracy


def intervals(text, recording, kind, duration, fs=48000):
    if kind != 'rig rx':
        return [], 'Host capture counters do not identify the clock of this tap/witness recording.'
    structured = tx_timing.intervals(text,recording,kind,duration,'VARA')
    text = ANSI.sub('', text)
    starts = list(RECORDING.finditer(text))
    wanted = [i for i, m in enumerate(starts) if Path(m[1]).stem == Path(recording).stem]
    if len(wanted) != 1:
        return structured, 'Explicit local transmit events on the recording clock.' if structured else 'Transcript does not uniquely identify this receiver recording.'
    i = wanted[0]
    text = text[starts[i].end():starts[i+1].start() if i+1 < len(starts) else len(text)]
    events, descriptions, last_key = [], {}, None
    for line in text.splitlines():
        if line.startswith('timing event='):
            fields = dict(FIELDS.findall(line))
            events.append(fields)
            if fields.get('event') == 'key_request' and fields.get('on') == '1':
                last_key = None
            if fields.get('event') == 'key_return' and fields.get('on') == '0' and fields.get('ok') == '1':
                last_key = fields.get('key_seq')
        elif last_key is not None and re.match(r'^\s*(?:tx |retry DATA over)', line):
            descriptions[last_key] = line.strip()
            last_key = None
    anchors, keys = [], {}
    for e in events:
        try:
            mono = float(e['mono_s'])
            if 'capture_samples' in e:
                anchors.append((mono, int(e['capture_samples'])/fs))
            if e['event'] == 'key_return' and e.get('ok') == '1':
                keys.setdefault(e['key_seq'], {})[e['on']] = mono
        except (KeyError, ValueError):
            continue
    def at(mono):
        if not anchors:
            return None
        anchor, sample = min(anchors, key=lambda a: abs(a[0]-mono))
        return sample+mono-anchor if abs(anchor-mono) <= .5 else None
    out = []
    for key, edges in keys.items():
        if set(edges) != {'0','1'} or edges['0'] <= edges['1']:
            continue
        lo, hi = at(edges['1']), at(edges['0'])
        if lo is None or hi is None or hi <= lo or lo >= duration or hi <= 0:
            continue
        out.append({'id': f'tx-{key}', 't0': max(0,lo), 't1': min(duration,hi),
                    'kind': descriptions.get(key, 'PTT / transmit').split(' keyed-by=')[0].split(' (')[0],
                    'evidence_type': 'ptt', 'direction': 'tx', 'role': 'caller',
                    'status': 'logged PTT interval', 'symbols': None,
                    'origin': 'original session timing',
                    'detail': descriptions.get(key, '') + ' · Successful key/unkey returns mapped by nearby capture sample counters.',
                    'attribution': 'local station is the caller; host PTT timing',
                    'unavailable': 'Transmit interval; no remote-receiver constellation.'})
    out = tx_timing.merge(structured,out)
    return out, (f'{len(out)} caller TX intervals recovered from host PTT timing; '
           f'{EDGE_MARGIN:.2f}s boundary guard. Software estimates, not measured RF edges.' if out else
           'No complete successful PTT pairs with nearby capture counters in this recording.')


def attribute(events, tx, *, qso=False):
    """Known protocol roles first; otherwise use the recorded local key schedule."""
    first = min((e['t0'] for e in tx), default=float('inf'))
    for e in events:
        if e.get('evidence_type') == 'ptt' or e.get('tx_confirmed') or e.get('transmit_label'):
            e.update(direction='tx', role='caller', tx_confirmed=True)
            e.setdefault('attribution', 'original local transmit log')
            continue
        if e.get('role') in ('caller','called'):
            e['direction'] = 'tx' if e['role'] == 'caller' else 'rx'
            e['attribution'] = ('QSO sender identified by protocol role' if qso else
                                'handshake role; local station assumed caller')
            continue
        lo, hi = e['t0'], e['t1']
        mid = (lo+hi)/2
        if qso and tx:
            # In a participating QSO there are two senders. Guard bands are
            # timing tolerances, not a third sender. Prelude signals are outside
            # this locally initiated exchange, rather than attributed to its peer.
            if hi < first:
                e.update(in_qso=False, direction='observed', role='unknown',
                         attribution='Recording prelude; before this QSO began')
                continue
            overlap = sum(max(0, min(hi,t['t1'])-max(lo,t['t0'])) for t in tx)
            local = (overlap > (hi-lo)/2 if hi>lo else
                     any(t['t0'] <= mid < t['t1'] for t in tx))
            e.update(direction='tx' if local else 'rx', role='caller' if local else 'called',
                     attribution=('Local transmission on the QSO transmit schedule' if local else
                                  'Peer transmission between local QSO transmissions'))
            continue
        if any(t['t0']+EDGE_MARGIN <= mid <= t['t1']-EDGE_MARGIN and
               (hi-lo < .001 or max(0,min(hi,t['t1'])-max(lo,t['t0']))/(hi-lo) >= .8) for t in tx):
            e.update(direction='tx', role='caller', attribution='overlaps logged local transmit interval')
        elif (tx and lo >= first
              and not any(t['t0']-EDGE_MARGIN <= mid <= t['t1']+EDGE_MARGIN for t in tx)
              and sum(max(0,min(hi,t['t1'])-max(lo,t['t0'])) for t in tx) <= .2 * max(.001,hi-lo)):
            e.update(direction='rx', role='called', attribution='outside local TX; called station assumed for this QSO')
        else:
            e.update(direction='observed', role='unknown', attribution='unmapped recording or transmit boundary')
