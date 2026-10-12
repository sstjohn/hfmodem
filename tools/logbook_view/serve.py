"""The logbook, read back from the arms themselves, on a map and in a table.

    .venv/bin/python tools/logbook_view/serve.py [--root PATH] [--port 8737] [--adif PATH ...]
    .venv/bin/python tools/logbook_view/serve.py --export DIR [--audio-base URL]

Every arm directory's provenance.json is a line in the log, the attempts from
before the arms among them as legacy.py converted them (and, until it writes
them, as it would: a transcript shows on the next reload), with what the newest
rehear of its audio (the operator data replay directory, rehear.py) heard added, and the mail its
recipient's inbox shows it delivered (the optional delivery-evidence file,
delivered.py). Additional contacts can be imported from explicitly supplied --adif files. Completed contacts export as ADIF for LoTW at /api/log.adi. Nothing
under the data root is written.

--export writes the public snapshot for static hosting: the page, log.json and
a CNAME, without transcripts, local paths or the LoTW export. Audio stays out of
DIR; DIR/../audio-manifest.txt maps each recording to the object key the page
expects under --audio-base.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import mimetypes
import re
import shlex
import sys
import subprocess
import threading
import time
import tomllib
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import parse_qs, quote, unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "packages" / package)
               for package in ("hfmodem", "hfhost")]
from hfmodem.core.band import AMATEUR_HF_BANDS as BANDS
from hfmodem.station.qsolog import read_adif

import audio
import link
import viewer
from viewer_jobs import Replays

HERE = Path(__file__).resolve().parent
mimetypes.add_type('audio/flac', '.flac')  # the stdlib table still says audio/x-flac

# Our own sense and keying; anything past these means the far end was heard.
OURS = {'cleared', 'keyed'}
# A contact LoTW can confirm: an arm that reached connect, or a QSO another program logged.
COMPLETE = {'connect', 'qso'}

# One QSO logged by two programs can differ by a decode cycle or two.
SAME_QSO_S = 120

# Optional location history is supplied with operator data, never baked into code.
AWAY = ()

# rehear.py's LIMITS, the fixed half of each `limits` line, in words the row detail can use.
LIMIT_WORDS = {
    'VARA high-SL overs undecoded': "Couldn't read {n} VARA data transmission{s} at the higher speed levels.",
    'VARA overs undecoded at the base rate': "Couldn't read {n} VARA data transmission{s}, even at the base rate.",
    'VARA BW2750 handshakes unattributed': "Couldn't tie {n} 2750 Hz VARA handshake{s} to a station.",
    'VARA handshakes off the candidate list': "Couldn't name the station behind {n} VARA handshake{s}; none on the gateway list fit.",
    'ARDOP peer payload text not emitted': "Couldn't read the text in {n} ARDOP data frame{s}.",
    'PACTOR-2 shape only': "Couldn't read {n} PACTOR-2 transmission{s}, only recognise the shape.",
    'PACTOR-4 entry chirp only': "Couldn't read {n} PACTOR-4 opening chirp{s}, only recognise the shape.",
    'PACTOR capture windows only, no continuous audio': "Couldn't hear the whole session: only short snippets after each of our calls were saved.",
    'PACTOR capture windows replayed deaf': "Couldn't fully hear {n} snippet{s}, lacking the session details the live decoder had.",
    'our keyings not separable on a witness recording': "Couldn't tell our transmissions from the gateway's on the remote receiver's recording.",
}
# rehear.py withholds ARDOP levels: the frames it decodes cannot be told apart by sender.
REHEARD_SPEED = {'vara', 'pactor'}

def grid_to_latlon(grid: str) -> tuple[float, float]:
    g = grid.strip().upper()
    lon = (ord(g[0]) - 65) * 20 - 180 + int(g[2]) * 2
    lat = (ord(g[1]) - 65) * 10 - 90 + int(g[3])
    if len(g) >= 6:
        lon += (ord(g[4]) - 65) * 5 / 60
        lat += (ord(g[5]) - 65) * 2.5 / 60
        return lat + 1.25 / 60, lon + 2.5 / 60
    return lat + 0.5, lon + 1.0


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def band_m(hz: int) -> int | None:
    return next((m for lo, hi, m in BANDS if lo <= hz / 1e6 <= hi), None)


def stamp(t: datetime) -> str:
    return t.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def duration(s: int) -> str:
    return f'{s // 60}m{s % 60:02d}s' if s >= 60 else f'{s}s'


def messages(n: int) -> str:
    return f"{n} message{'s' * (n != 1)}"


def mail_words(sent: int, received: int, delivered: int = 0) -> str:
    """Which way mail went, "mail sent (1 message)", "mail received (2 messages)" or both, from
    the messages the transcript shows moving and the ones the recipient's inbox confirms."""
    said = []
    if sent := max(sent, delivered):
        confirmed = '' if not delivered else ' delivered' if delivered == sent else f', {delivered} delivered'
        said.append(f'mail sent ({messages(sent)}{confirmed})')
    if received:
        said.append(f'mail received ({messages(received)})')
    return '; '.join(said)


def verdict_levels(levels: dict | None) -> dict:
    """A VARA verdict's own ladder: what we decoded of the peer's overs, and the fastest of ours it took."""
    level = lambda v: v if v is None or str(v).startswith('L') else f'L{v}'
    return {k: v for k, v in (('decoded', level(((levels or {}).get('rx') or {}).get('decoded_max'))),
                              ('acknowledged', level(((levels or {}).get('tx') or {}).get('acknowledged_max')))) if v}


def summary(e: dict) -> str:
    """How far the contact got, in plain words: what came back, how fast, how long."""
    reached, call, p4 = set(e['reached']), e['call'], e.get('p4')
    if 'connect' in reached:
        how = ' '.join(filter(None, (e['speed'] and f"at up to {e['speed']['top']}", e['bw_hz'] and f"({e['bw_hz']} Hz)")))
        if p4 and p4['reached'] != 'probed':
            how = f"at {p4['opening']}, then " + P4_WORDS[p4['reached']].format(**p4_evidence(p4))
        elif p4:
            how += '; ' + P4_WORDS['probed'].format(**p4_evidence(p4))
        if moved := mail_words(**e['mail']):
            mail = 'logged in; ' * ('login' in reached) + moved
        else:
            banner = p4 and any(x.startswith('banner ') for x in p4['peer'])
            mail = ('mail exchanged' if 'mail' in reached else 'logged in; no mail moved' if 'login' in reached
                    else 'greeting received; no mail' if 'greeting' in reached
                    else 'no SID after the banner' if banner else 'no greeting received')
        said = f'Connected to {call} {how}'.rstrip() + '; ' + mail
    elif 'answer' in reached:
        said = f'Called {call}; it answered, but no link came up'
    elif 'keyed' in reached:
        said = f'Called {call}; no answer'
    else:
        return 'Did not transmit.'
    return said + (f". Lasted {duration(e['duration_s'])}." if e['duration_s'] else '.')


# How far a link went in PACTOR-4, from link.pactor's `p4`.
P4_WORDS = {'exchange': 'PACTOR-4 both ways (the peer: {peer}; ours: {ours})',
            'answered': 'PACTOR-4 words from the peer answering ours (the peer: {peer}; ours: {ours})',
            'heard': 'PACTOR-4 heard from the peer only ({peer}); nothing of ours in PACTOR-4',
            'probed': 'our PACTOR-4 went unanswered ({ours})'}


def p4_evidence(p4: dict) -> dict:
    return {side: ', '.join(p4[side]) for side in ('peer', 'ours')}


@functools.lru_cache(maxsize=None)
def reading(log: Path, version: int, protocol: str) -> dict:
    """What an arm's own transcript says, read once for each version of it."""
    text = link.ANSI.sub('', log.read_text(errors='replace'))
    got = {'named': audio.named(text), 'reached': link.milestones(text, protocol), 'mail': link.mail(text)} | link.tx_power(text)
    if protocol == 'pactor':
        got |= link.pactor(text) | {'submode': next((sub for line, sub in link.PACTOR_ENTRY.items() if line in text), None)}
    elif protocol == 'ardop':
        got |= link.ardop(text)
    return got


@functools.lru_cache(maxsize=None)
def rehearing(path: Path, version: int) -> dict:
    """One rehear of an arm, read once for each version of it."""
    r = json.loads(path.read_text())
    return {'commit': r['rehearer']['commit'][:8], 'run_utc': r['rehearer']['run_utc'],
            'heard': r['heard'] or {}, 'limits': r['limits']}


def limit_words(line: str) -> str:
    what, _, n = line.rpartition(': ')
    if what not in LIMIT_WORDS:
        return line
    return LIMIT_WORDS[what].format(n=n, s='' if n == '1' else 's')


def adif_field(name: str, value) -> str:
    value = str(value)
    return f'<{name}:{len(value)}>{value}'


def adif_logs(extra: Iterable[Path] = ()) -> list[Path]:
    return list(dict.fromkeys(p.resolve() for p in extra))


def qso_time(r: dict, date: str, time: str) -> datetime | None:
    if not r.get(date) or not r.get(time):
        return None
    return datetime.strptime(r[date] + r[time].ljust(6, '0')[:6], '%Y%m%d%H%M%S').replace(tzinfo=UTC)


def adif_band(r: dict) -> int | None:
    if m := re.fullmatch(r'(\d+)m', r.get('BAND', '').strip().lower()):
        return int(m[1])
    return band_m(float(r['FREQ']) * 1e6) if r.get('FREQ') else None


def imported(paths: Iterable[Path]) -> tuple[list[tuple[dict, str]], Counter]:
    """Every QSO in the logs once, each with the program that logged it. Programs
    that share or copy a log record the same QSO more than once; of those the
    fullest record stays. Returns the survivors and the collapses per pair of files."""
    seen: dict[tuple, list] = {}
    collapsed = Counter()
    for path in paths:
        text = path.read_bytes().decode('utf-8', errors='replace')
        program = (re.search(r'<programid:\d+(?::[^>]*)?>([^<\r\n]*)', text, re.I) or [None, path.stem])[1].strip()
        for r in read_adif(text):
            if not (r.get('CALL') and (on := qso_time(r, 'QSO_DATE', 'TIME_ON'))):
                continue
            key = (r['CALL'].upper().split('-')[0], adif_band(r), (r.get('SUBMODE') or r.get('MODE', '')).lower())
            kept = seen.setdefault(key, [])
            twin = next((k for k in kept if abs((k[0] - on).total_seconds()) <= SAME_QSO_S), None)
            if twin is None:
                kept.append([on, r, program, path])
                continue
            collapsed[tuple(sorted((str(twin[3]), str(path))))] += 1
            if sum(map(bool, r.values())) > sum(map(bool, twin[1].values())):
                twin[:] = [on, r, program, path]
    return [(r, program) for kept in seen.values() for _, r, program, _ in kept], collapsed


def qso_adif(e: dict, station: dict) -> str:
    """One contact as ADIF. A gateway's own export must agree on CALL, BAND,
    mode group and a TIME_ON within 30 minutes for LoTW to match the pair.
    A QSO read from another program's log goes back out as that program wrote it."""
    if 'adif' in e:
        return ' '.join(adif_field(k, v) for k, v in e['adif'].items() if v) + ' <EOR>\n'
    started = datetime.fromisoformat(e['started_utc'])
    fields = {
        'CALL': e['call'].split('-')[0],  # a Winlink SSID is not part of the callsign
        'QSO_DATE': f'{started:%Y%m%d}', 'TIME_ON': f'{started:%H%M%S}',
        'BAND': f"{e['band_m']}m", 'FREQ': f"{(e['centre_hz'] or e['dial_hz']) / 1e6:.6f}",
        # ADIF 3.1.7 carries ARDOP as its own MODE with no listed submodes.
        'MODE': {'pactor': 'PAC', 'vara': 'DYNAMIC', 'ardop': 'ARDOP'}[e['protocol']], 'SUBMODE': e['submode'],
        'GRIDSQUARE': e['grid'],
        'STATION_CALLSIGN': station['call'], 'MY_GRIDSQUARE': e['my_grid'],
    }
    if e.get('tx_w'):
        fields['TX_PWR'] = f"{e['tx_w']:g}"
    if e['ended_utc']:
        ended = datetime.fromisoformat(e['ended_utc'])
        fields |= {'QSO_DATE_OFF': f'{ended:%Y%m%d}', 'TIME_OFF': f'{ended:%H%M%S}'}
    return ' '.join(adif_field(k, v) for k, v in fields.items() if v) + ' <EOR>\n'


def matches_filters(e: dict, only: dict) -> bool:
    """Keep ADIF exports aligned with the page's date and hour filters."""
    if any(str(e[field]) != only[key] for key, field in
           (('protocol', 'protocol'), ('band', 'band_m'), ('bw', 'bw_hz'), ('call', 'call')) if key in only):
        return False
    if only.get('q', '').upper() not in e['call'].upper():
        return False
    date = e['started_utc'][:10]
    if only.get('since') and date < only['since'] or only.get('until') and date > only['until']:
        return False
    if only.get('hourFrom', '') != '' or only.get('hourThrough', '') != '':
        hour = datetime.fromisoformat(e['started_utc']).astimezone(ZoneInfo(only.get('timeZone') or 'UTC')).hour
        start, end = int(only.get('hourFrom') or 0), int(only.get('hourThrough') or 23)
        if not (start <= hour <= end if start <= end else hour >= start or hour <= end):
            return False
    return True


def adif(entries: Iterable[dict], station: dict) -> str:
    now = datetime.now(UTC)
    head = (f"{station['call']} logbook, completed contacts, exported {stamp(now)}\n"
            + adif_field('ADIF_VER', '3.1.7') + '\n' + adif_field('PROGRAMID', 'hfmodem-logbook') + '\n'
            + adif_field('CREATED_TIMESTAMP', f'{now:%Y%m%d %H%M%S}') + '\n<EOH>\n\n')
    return head + ''.join(qso_adif(e, station) for e in entries if COMPLETE & set(e['reached']))


def default_root() -> Path:
    from consolelib.operator import data_root
    return data_root()


class Logbook:
    def __init__(self, root: Path, adif_paths: Iterable[Path] = (), viewer_root: Path | None = None):
        self.root = root.resolve()
        self.viewer_root = (viewer_root or self.root / 'working' / 'viewer-artifacts').resolve()
        self.adif_paths = list(adif_paths)
        self.working = (self.root / 'working').resolve()
        # What /file/ serves: the arms and transcripts, and the recordings they name.
        self.served = tuple((self.root / d).resolve() for d in ('working', 'captures', 'logs/onair', 'logs/sense'))
        self.receivers = audio.kiwis(self.root)
        self.converter, self.converted, self.converting = None, (None, {}), threading.Lock()
        from consolelib.operator import station_file
        config_path = self.root / 'station.toml'
        if os.environ.get('HFMODEM_STATION') or not config_path.is_file():
            config_path = station_file()
        station = tomllib.loads(config_path.read_text())['station']
        history = self.root / 'location-history.json'
        self.away = [(row['grid'], datetime.fromisoformat(row['since']), datetime.fromisoformat(row['until']))
                     for row in json.loads(history.read_text())] if history.is_file() else list(AWAY)
        self.station = {'call': station['mycall'], 'grid': station['grid']}
        self.replays = Replays(self)

    def my_grid(self, started_utc: str) -> str:
        t = datetime.fromisoformat(started_utc)
        return next((grid for grid, since, until in self.away if since <= t < until), self.station['grid'])

    def gateways(self) -> dict[str, tuple[str, str]]:
        found = {}
        for name in ('winlink-vara-gateways.csv', 'winlink-pactor-gateways.csv'):
            path = self.root / name
            if path.is_file():
                with path.open(newline='') as f:
                    for row in csv.DictReader(f):
                        if row['GridSquare']:
                            found.setdefault(row['Callsign'].upper(), (row['GridSquare'], row['QTH']))
        return found

    def served_path(self, rel: str | Path) -> Path | None:
        path = (self.root / rel).resolve()
        return path if path.is_file() and any(path.is_relative_to(d) for d in self.served) else None

    def url(self, path: Path) -> str | None:
        return self.served_path(path) and '/file/' + quote(str(path.relative_to(self.root)))

    def recording(self, kind: str, rel: str | Path) -> dict | None:
        if not (local := self.served_path(rel) or self.served_path(Path(rel).with_suffix('.flac'))):
            return None
        side = audio.sidecar(local)
        return {'url': self.url(local), 'kind': kind, 'label': audio.label(kind, side, self.receivers),
                'duration_s': audio.duration_s(local, side)}

    def rehears(self, arms: set[str]) -> dict[str, dict]:
        """The newest rehear of each arm, whichever decoder commit made it."""
        newest = {}
        for p in (self.working / 'rehear').glob('*/*.json'):
            if p.stem in arms:
                r = rehearing(p, p.stat().st_mtime_ns)
                if p.stem not in newest or r['run_utc'] > newest[p.stem]['run_utc']:
                    newest[p.stem] = r
        return newest

    def delivered(self) -> Counter:
        """How many messages the recipient's inbox shows each arm delivered."""
        path = self.working / 'winlink-delivered.json'
        return Counter(d['arm'] for d in json.loads(path.read_text())) if path.is_file() else Counter()

    def unconverted(self) -> dict[str, dict]:
        """The attempts legacy.py has not written yet, converted here the way it would write them,
        and again only once a transcript, arm or recording it reads has changed."""
        import legacy  # it reads the gateway listings through this module
        with self.converting:
            if self.converter is None:
                self.converter = legacy.Converter(self.root)
            seen = self.converter.fingerprint()
            if seen != self.converted[0]:
                self.converted = seen, self.converter.pending()
            return self.converted[1]

    def arm(self, path: Path, rehear: dict | None = None, delivered: int = 0, p: dict | None = None) -> dict:
        """One arm's line, from its provenance.json or from a record legacy.py has yet to write there."""
        p = p or json.loads(path.read_text())
        verdict = p.get('verdict') or {}
        peer = p.get('peer') or {}
        said = p.get('adjustments') or {}
        started = datetime.fromisoformat(p['started_utc'])
        ended = p.get('ended_utc') and datetime.fromisoformat(p['ended_utc'])
        protocol = p['arm'].split('-', 1)[0]
        reached = verdict.get('reached', [])
        # A converted arm keeps its transcript and recordings where they were made, and says what they held.
        converted = (p.get('source') or {}).get('converted_from')
        log = self.root / converted if converted else path.parent / 'arm.log'
        read, rederived = {}, []
        # The TX power a converted arm flew is read from its transcript here too,
        # not from what legacy.py stored: that field only catches up when it is re-run.
        if log.is_file():
            read = reading(log, log.stat().st_mtime_ns, protocol)
        if converted:
            got = p.get('link') or {}
            found = [(r['kind'], r['path']) for r in p.get('audio') or []]
        else:
            got = ({'bw_hz': said.get('bw_hz')} | (verdict_levels(verdict.get('levels'))
                                                     or link.vara_levels(verdict.get('line'))) if protocol == 'vara'
                   else dict(read, bw_hz=read.get('bw_hz') or said.get('bw_hz')))
            tap = next((q for q in (path.parent / 'tap.wav', path.parent / 'tap.flac') if q.is_file()), None)
            found = [('rig tap', tap)] * bool(tap) + read.get('named', [])
            # The live verdicts miss what their anchored patterns cannot see, and PACTOR's starts at
            # connect; the transcript is read the way a converted arm's was, adding and never removing.
            rederived = [m for m in read.get('reached', []) if m not in reached]
            reached = link.merged(reached, rederived, protocol)
        # A converted transcript can hold several attempts, so the converter kept each one's own mail.
        moved = verdict.get('mail', {}) if converted else read.get('mail', {})
        reheard, faster = [], {}
        # Only an arm that called can be answered: a rehear also hears whatever else shares its
        # capture folder, and an arm stopped before keying would take a later arm's link.
        if rehear and 'keyed' in reached:
            heard, order = rehear['heard'], [name for name, _ in link.MILESTONES[protocol]]
            reheard = [m for m in heard.get('reached', []) if m in order and m not in reached]
            reached = link.merged(reached, reheard, protocol)
            got = dict(got)
            for way in ('decoded', 'acknowledged') if protocol in REHEARD_SPEED else ():
                level = heard.get(f'{way}_max')
                if (link.rank(level) or 0) > (link.rank(got.get(way)) or 0):
                    got[way] = faster[way] = level
        # A message in its recipient's inbox is mail the gateway took and passed on, whatever the transcript saw.
        confirmed = ['mail'] if delivered else []
        reached = link.merged(reached, confirmed, protocol)
        recordings = {r['url']: r for kind, rel in found if (r := self.recording(kind, rel))}
        top = link.fastest(got.get('decoded'), got.get('acknowledged'))
        e = {
            'id': p['arm'], 'protocol': protocol, 'call': p['call'],
            'viewer': '/inspect?arm=' + quote(p['arm']) if protocol in ('vara','pactor','ardop') else None,
            'band_m': p['band_m'], 'dial_hz': p['dial_hz'], 'centre_hz': p.get('centre_hz'),
            'started_utc': stamp(started), 'ended_utc': ended and stamp(ended),
            'duration_s': ended and round((ended - started).total_seconds()),
            'level': (p.get('level') or {}).get('name'), 'flymode': p.get('mode'),
            'reached': reached, 'rederived': rederived, 'reheard': reheard, 'confirmed': confirmed,
            'rehear': rehear and {'commit': rehear['commit'], 'date': rehear['run_utc'][:10], 'faster': faster,
                                  'limits': [limit_words(line) for line in rehear['limits']]},
            'bw_hz': got.get('bw_hz'), 'tx_w': read.get('tx_w'), 'tx_w_max': read.get('tx_w_max'),
            'mail': {'sent': max(len(moved.get('sent', ())), delivered), 'received': len(moved.get('received', ())),
                     'delivered': delivered},
            'speed': top and {'top': top, 'rank': link.rank(top),
                              'decoded': got.get('decoded'), 'acknowledged': got.get('acknowledged')},
            'grid': peer.get('grid'), 'qth': peer.get('qth'),
            'audio': sorted(recordings.values(), key=lambda r: audio.KINDS.index(r['kind'])),
            'transcript': self.url(log),
            'p4': got.get('p4'),
            # PACTOR connects at level 1, which ADIF gives no submode; only a read entry packet, or
            # PACTOR-4 both ways, moved the link up.
            'submode': 'VARA HF' if protocol == 'vara' else 'PAC4' if (got.get('p4') or {}).get('reached') == 'exchange'
                       else read.get('submode') if 'entry read' in reached else None,
        }
        return e | {'summary': summary(e)}

    @staticmethod
    def qso(r: dict, program: str) -> dict:
        started = qso_time(r, 'QSO_DATE', 'TIME_ON')
        ended = qso_time(r, 'QSO_DATE_OFF', 'TIME_OFF')
        protocol = (r.get('SUBMODE') or r['MODE']).lower()  # FT4 and JS8 are MFSK submodes
        band = adif_band(r)
        freq = r.get('FREQ') and round(float(r['FREQ']) * 1e6)
        reports = ', '.join(f'{k} {r[f]}' for k, f in (('sent', 'RST_SENT'), ('received', 'RST_RCVD')) if r.get(f))
        return {
            'id': f"{protocol}-{r['CALL'].lower()}-{band}-{started:%Y%m%dT%H%M%SZ}",
            'protocol': protocol, 'submode': r.get('SUBMODE'), 'call': r['CALL'],
            'band_m': band, 'dial_hz': None, 'centre_hz': freq,
            'started_utc': stamp(started), 'ended_utc': ended and stamp(ended),
            # WSJT-X stamps TIME_OFF at logging and often TIME_ON with it, which says nothing about length.
            'duration_s': round((ended - started).total_seconds()) if ended and ended > started else None,
            'level': None, 'flymode': None,
            'reached': ['qso'], 'summary': f"Logged by {program}" + (f"; reports {reports}." if reports else '.'),
            'bw_hz': None, 'tx_w': float(r['TX_PWR']) if r.get('TX_PWR') else None, 'tx_w_max': None, 'speed': None,
            'grid': r.get('GRIDSQUARE') or None, 'qth': r.get('QTH') or None, 'my_grid': r.get('MY_GRIDSQUARE') or None,
            'audio': [], 'transcript': None, 'adif': r,
        }

    def entries(self) -> dict:
        grids = self.gateways()
        arms = list(self.working.glob('*/arms/*/provenance.json'))
        pending = self.unconverted()
        heard, sent = self.rehears({p.parent.name for p in arms} | pending.keys()), self.delivered()
        entries = [self.arm(p, heard.get(p.parent.name), sent[p.parent.name]) for p in arms]
        entries += [self.arm(Path(r['directory']) / 'provenance.json', heard.get(name), sent[name], r)
                    for name, r in pending.items()]
        entries += (self.qso(r, program) for r, program in imported(self.adif_paths)[0])
        out = []
        for e in entries:
            e = dict(e, answered=bool(set(e['reached']) - OURS))
            if not e['grid']:
                e['grid'], qth = grids.get(e['call'].upper()) or grids.get(e['call'].split('-')[0].upper()) or (None, None)
                e['qth'] = e['qth'] or qth
            e['my_grid'] = e.get('my_grid') or self.my_grid(e['started_utc'])
            e['lat'], e['lon'] = grid_to_latlon(e['grid']) if e['grid'] else (None, None)
            # peer.distance_km was measured from whatever station.toml said at the time, so measure again.
            e['distance_km'] = e['grid'] and round(distance_km(grid_to_latlon(e['my_grid']), (e['lat'], e['lon'])), 1)
            out.append(e)
        out.sort(key=lambda e: e['started_utc'])
        homes = {}
        for e in out:
            lat, lon = grid_to_latlon(e['my_grid'])
            home = homes.setdefault(e['my_grid'], {'grid': e['my_grid'], 'lat': lat, 'lon': lon, 'first_utc': e['started_utc']})
            home['last_utc'] = e['started_utc']
        out.reverse()
        return {'station': {'call': self.station['call'], 'homes': list(homes.values())}, 'entries': out}


class Handler(BaseHTTPRequestHandler):
    book: Logbook

    def do_GET(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        if path == '/':
            self.send_bytes((HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif path == '/summary.js':
            self.send_bytes((HERE / 'summary.js').read_bytes(), 'text/javascript; charset=utf-8')
        elif path in ('/inspect', '/viewer.js', '/viewer.css', '/link_state.mjs', '/waveforms.json'):
            name = {'/inspect': 'viewer.html', '/viewer.js': 'viewer.js', '/viewer.css': 'viewer.css', '/link_state.mjs': 'link_state.mjs', '/waveforms.json': 'waveforms.json'}[path]
            kind = {'/inspect': 'text/html', '/viewer.js': 'text/javascript', '/viewer.css': 'text/css', '/link_state.mjs': 'text/javascript', '/waveforms.json': 'application/json'}[path]
            self.send_bytes((HERE / name).read_bytes(), kind + '; charset=utf-8')
        elif path == '/api/inspect/job':
            key = parse_qs(url.query).get('id', [''])[0]
            job = self.book.replays.status(key)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(json.dumps(job).encode(), 'application/json')
        elif path == '/api/inspect':
            arm = parse_qs(url.query).get('arm', [''])[0]
            command = shlex.join([sys.executable, str(HERE / 'vara_view.py'),
                                  '--root', str(self.book.root), '--arm', arm,
                                  '--out', str(self.book.viewer_root)])
            self.send_bytes(json.dumps({'artifacts': viewer.artifacts(self.book.viewer_root, arm),
                                        'build_command': command}).encode(), 'application/json')
        elif path.startswith('/viewer-data/'):
            local = viewer.data_path(self.book.viewer_root, path.removeprefix('/viewer-data/'))
            if local is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(local.read_bytes(), 'application/json' if local.suffix == '.json'
                                else 'application/octet-stream')
        elif path == '/log.json':
            self.send_bytes(json.dumps(self.book.entries() | {'lotw': '/api/log.adi'}).encode(), 'application/json')
        elif path == '/api/log.adi':
            self.send_adif({k: v[0] for k, v in parse_qs(url.query).items()})
        elif path.startswith('/file/'):
            self.send_file(path.removeprefix('/file/'))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlsplit(self.path).path != '/api/inspect/rehear':
            return self.send_error(HTTPStatus.NOT_FOUND)
        # Only same-origin JSON requests from this local viewer can start work.
        # No shell interpolation or caller-supplied paths/commands.
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.send_error(HTTPStatus.FORBIDDEN)
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Invalid request size.')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or not isinstance(request.get('arm'), str):
                raise ValueError('A QSO ID is required.')
            job = self.book.replays.start(request['arm'])
        except (ValueError, OSError) as exc:
            return self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
        self.send_bytes(json.dumps(job).encode(), 'application/json')

    def send_adif(self, only: dict):
        book = self.book.entries()
        try:
            entries = [e for e in book['entries'] if matches_filters(e, only)]
        except (ValueError, KeyError):
            self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid logbook filter')
            return
        entries.reverse()
        name = f"{book['station']['call']}-{datetime.now(UTC):%Y%m%d}.adi"
        self.send_bytes(adif(entries, book['station']).encode(), 'text/plain; charset=utf-8',
                        ('Content-Disposition', f'attachment; filename="{name}"'))

    def send_bytes(self, body: bytes, kind: str, *headers: tuple[str, str]):
        self.send_response(HTTPStatus.OK)
        for header in headers:
            self.send_header(*header)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def send_file(self, rel: str):
        if not (path := self.book.served_path(rel)):
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = path.stat().st_size
        first, last = 0, size - 1
        ranged = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', '').strip())
        if ranged and any(ranged.groups()):
            a, b = ranged.groups()
            first, last = (max(0, size - int(b)), last) if not a else (int(a), min(int(b), last) if b else last)
            if first < 0 or first > last:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
        else:
            self.send_response(HTTPStatus.OK)
        kind = 'text/plain; charset=utf-8' if path.suffix == '.log' else mimetypes.guess_type(path)[0]
        self.send_header('Content-Type', kind or 'application/octet-stream')
        self.send_header('Content-Length', str(last - first + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        if self.command == 'HEAD':
            return
        with path.open('rb') as f:
            f.seek(first)
            left = last - first + 1
            try:
                while left and (chunk := f.read(min(left, 1 << 16))):
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except ConnectionError:
                pass  # a seeking <audio> drops the old request mid-stream

    def do_HEAD(self):
        self.do_GET()

    def log_message(self, fmt, *args):
        pass


def export(book: Logbook, out: Path, audio_base: str | None, hostname: str | None = None):
    """The public snapshot. Each recording is named by object key, <arm id>/<file name>,
    and left for a separate upload; transcripts carry Winlink login responses,
    message subjects and addresses, so they stay home."""
    log = book.entries()
    manifest = []
    for e in log['entries']:
        e.pop('viewer', None)  # replay artifacts are local, not part of the public snapshot
        e.pop('adif', None)
        e.pop('transcript')
        keys = set()
        for r in e['audio']:
            local = book.served_path(unquote(r['url'].removeprefix('/file/')))
            # Two witness.wav of one arm come from two session folders; the folder tells them apart.
            key = f"{e['id']}/{local.name}"
            if key in keys:
                key = f"{e['id']}/{local.parent.name}-{local.name}"
            keys.add(key)
            manifest.append(f'{local}\t{key}\n')
            r['url'] = audio_base and f"{audio_base.rstrip('/')}/{quote(key)}"
    body = json.dumps(log)
    if str(Path.home()) in body:
        raise SystemExit(f'refusing to publish: log.json names a path under {Path.home()}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'index.html').write_bytes((HERE / 'index.html').read_bytes())
    (out / 'summary.js').write_bytes((HERE / 'summary.js').read_bytes())
    (out / 'log.json').write_text(body)
    if hostname:
        (out / 'CNAME').write_text(hostname + '\n')
    (out.parent / 'audio-manifest.txt').write_text(''.join(manifest))
    print(f"{len(log['entries'])} entries to {out}; {len(manifest)} recordings listed in {out.parent / 'audio-manifest.txt'}")


def git_email() -> str | None:
    found = subprocess.run(['git', '-C', str(HERE), 'config', 'user.email'], capture_output=True, text=True)
    return found.stdout.strip() or None


def watch_inbox(root: Path, account: str):
    """Keep the optional delivery-evidence file current, so a delivered message shows on the next reload."""
    import delivered  # it imports this module
    while True:
        try:
            delivered.refresh(root, imap=account, say=lambda line: None)
        except Exception as failed:  # a dropped connection waits for the next round, and the page keeps serving
            print(f'inbox: {failed}')
        time.sleep(600)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', type=Path, default=None, help='repository holding working/ and station.toml')
    ap.add_argument('--viewer-root', type=Path, help='derived replay artifact directory')
    ap.add_argument('--port', type=int, default=8737)
    ap.add_argument('--adif', type=Path, action='append', default=[], help='another ADIF log to import (repeatable)')
    ap.add_argument('--export', type=Path, metavar='DIR', help='write the public snapshot to DIR and exit')
    ap.add_argument('--audio-base', metavar='URL', help='where the exported page finds the uploaded taps')
    ap.add_argument('--hostname', help='optional hostname for the static export CNAME')
    ap.add_argument('--imap', metavar='ACCOUNT', default=None,
                    help="Gmail account whose mail confirms deliveries, read every ten minutes "
                         "when explicitly enabled and ~/.imap_password exists")
    args = ap.parse_args()
    logs = adif_logs(args.adif)
    qsos, collapsed = imported(logs)
    print(f"{len(qsos)} QSOs from {', '.join(map(str, logs)) or 'no ADIF logs'}")
    for pair, n in collapsed.items():
        print(f"  {n} duplicate{'s' * (n != 1)} collapsed: {' + '.join(dict.fromkeys(pair))}")
    book = Logbook((args.root or default_root()).resolve(), logs, viewer_root=args.viewer_root)
    if args.export:
        return export(book, args.export.resolve(), args.audio_base, args.hostname)
    Handler.book = book
    # The first conversion reads every transcript; have it done before the first page asks.
    threading.Thread(target=book.unconverted, daemon=True).start()
    if args.imap and (Path.home() / '.imap_password').is_file():
        threading.Thread(target=watch_inbox, args=(book.root, args.imap), daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'logbook at http://127.0.0.1:{args.port}/  (data from {Handler.book.root})')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
