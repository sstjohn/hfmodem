"""Rank exact centers from the Winlink completed-session propagation feed.

This is session evidence, not the RMS channel heartbeat. Missing client grids
and nonreporting gateways make it incomplete; distance is a heuristic, not an
HF propagation prediction.
"""
from datetime import datetime, timezone
import json
import math
import os
import re
import urllib.parse
import urllib.request

from hfcapture import distance_km, grid_to_latlon


def fetch(hours=24):
    key = os.environ.get('WINLINK_API_KEY')
    if not key:
        raise RuntimeError('set WINLINK_API_KEY or supply --feed-file')
    query = urllib.parse.urlencode(dict(format='json', windowStart=hours,
                                       windowLength=hours, key=key))
    # Do not include the URL/key in error messages or artifacts.
    try:
        with urllib.request.urlopen('https://cms.winlink.org/propagation/list?' + query,
                                    timeout=20) as response:
            data = json.load(response)
    except Exception as exc:
        raise RuntimeError(f'Winlink feed failed ({type(exc).__name__})') from None
    if not isinstance(data, dict) or not isinstance(data.get('PropagationRecords'), list):
        raise ValueError('Winlink feed lacks PropagationRecords')
    return data


def timestamp(value):
    if isinstance(value, str) and value.startswith('/Date('):
        match = re.match(r'/Date\((-?\d+)', value)
        if not match:
            raise ValueError('invalid Winlink timestamp')
        return int(match[1]) / 1000
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def rank(data, grid, *, now=None, hours=24, reach_km=4200,
         min_hz=3000000, max_hz=30000000):
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    home = grid_to_latlon(grid) if grid else None
    channels = {}
    tally = dict(records=0, pactor=0, eligible=0, malformed=0)
    for row in data['PropagationRecords']:
        tally['records'] += 1
        if 'PACTOR' not in str(row.get('Mode', '')).upper():
            continue
        tally['pactor'] += 1
        try:
            hz = int(row['Frequency'])
            age = (now - timestamp(row['Timestamp'])) / 3600
            if not min_hz <= hz <= max_hz or not 0 <= age <= hours:
                continue
            positions = [(float(row[f'{side}Lat']), float(row[f'{side}Lon']))
                         for side in ('Server', 'Client')]
            if any(not (-90 <= lat <= 90 and -180 <= lon <= 180)
                   or (lat == 0 and lon == 0) for lat, lon in positions):
                continue
            far = max(distance_km(home, pos) for pos in positions) if home else 0
            if far > reach_km:
                continue
            duration = float(row.get('HoldingSeconds', 0))
            if not math.isfinite(duration) or duration <= 0:
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            tally['malformed'] += 1
            continue
        tally['eligible'] += 1
        entry = channels.setdefault(hz, dict(center_hz=hz, dial_hz=hz-1500,
            score=0., sessions=0, p4_sessions=0, freshest_hours=age, gateways=[], modes=[], recent_sessions=[]))
        p4 = bool(re.search(r'PACTOR\s*[- ]?4', row['Mode'], re.I))
        entry['score'] += (2 if p4 else 1) * math.exp(-age / 2) / (1 + (far / 2000)**2)
        entry['sessions'] += 1
        entry['p4_sessions'] += int(p4)
        entry['freshest_hours'] = min(entry['freshest_hours'], age)
        entry['recent_sessions'].append(dict(server=row.get('Server'), client=row.get('Client'),
            timestamp=row['Timestamp'], mode=row['Mode'], farther_km=round(far), age_hours=age))
        entry['recent_sessions'].sort(key=lambda s: s['age_hours'])
        del entry['recent_sessions'][10:]
        for field, value in [('gateways', row.get('Server', '')), ('modes', row['Mode'])]:
            if value not in entry[field]:
                entry[field].append(value)
    return sorted(channels.values(), key=lambda c: (-c['score'], c['center_hz'])), tally
