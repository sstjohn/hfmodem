# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The on-air history from before the arms, converted into arm records.

    .venv/bin/python tools/logbook_view/legacy.py [--root PATH] [--dry-run]

Before the console wrote provenance.json, an attempt left only its console
transcript: optional historical on-air transcripts from tools/onair.sh --log (deeper where the log
name carried a path of its own), the rig sessions of August, and a few dozen loose
logs from the weeks before those; harnesses that fly without the console still
leave them. Each attempt in them becomes the converted legacy-arm directory<arm>/
provenance.json in the arms' own spelling, its milestones read with
link.MILESTONES, the patterns the logbook reads a native arm's transcript with,
so the logbook reads one format. The transcript stays where it is, and so does
the audio; the record names both by their paths under the repository.

Folder names are local time (CDT, UTC-5); a stamp ending in Z is UTC and wins.
An attempt already recorded as an arm, or by another transcript within two
minutes, is converted once. Re-running rewrites the converted legacy-arm directory and
nothing else, and leaves a record whose content has not changed untouched. The
viewer runs the same conversion in memory for transcripts not yet written, so an
arm keeps its name from the moment it shows until long after it is written.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import audio
import link
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.face import Logbook, band_m, default_root, mail_words

CONVERTER = 'tools/logbook_view/legacy.py'
LOCAL = timezone(timedelta(hours=-5))  # every folder name was written in CDT
SAME_ATTEMPT_S = 120

HEADER = re.compile(
    r'^== (?:(?P<pactor>PACTOR)-\d|(?P<vara>VARA)(?: mail)?|(?P<ardop>ARDOP) mail) \S+ <?-> '
    r'(?P<call>[A-Z0-9/-]+)[^,\n]*, centre (?P<centre>\d+) \(dial (?P<dial>\d+)\)(?:, BW(?P<bw>\d+))?(?P<rest>.*)$', re.M)
WAITING = re.compile(r'^== waiting for ([\d.]+) kHz to clear, then (pactor|vara|ardop)[\w-]* -> ([A-Z0-9/-]+)', re.M)
NOT_ATTEMPTS = re.compile(r'^(?:== hardware|== stopping everything|!! THE RIG IS ALREADY CLAIMED|== listening .*NO TX)', re.M)
SKIP_NAMES = re.compile(r'^(?:witness|rehear|tap-|vara-monitor|sabir-proof-|txwitness|scan-|chase)')
LOOSE_DIRS = ('ardop-day-08*', 'ardop-night-0826', 'pactor-day-0828', 'vara-day-0828', 'pactor3-handoff')

# What refused an attempt before anything went out: the launcher's `!!` lines, less the ones that report on a run.
REFUSAL = re.compile(r'^\s*!! ?(?!.*(?:SATURATED|TRANSMIT ARMED|WITNESS|ANSWER SLOT|PTT LEAD|RX \d))(.+(?:\n {3}\S.*)*)', re.M)
NOT_TRANSMITTING = re.compile(r'channel in use — NOT transmitting|channel never cleared|-> OCCUPIED')
# run_server says "the link went down mid-exchange" of a call that never connected; besra's own line is the truth there.
FAILED = re.compile(r'^\s*(CONNECT TO \S+ FAILED!)', re.M)
STOPPED = re.compile(r'^\((SIG[A-Z]+)\)$', re.M)
OWN_WORD = re.compile(r'^(?:verdict: |session ended: )(.+)$|^(NOT connected.*|DISCONNECTED: .+|mail: exchange (?:complete|failed).*)$', re.M)

UTC_STAMP = re.compile(r'(\d{8}T\d{6})(?:\.\d+)?Z')
LOG_STAMP = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3}(Z?) ', re.M)
CAPTURES = re.compile(r'captures/onair-(\d{4})-(\d{4})')
STREAM_S = re.compile(r'^session stream: \S+ -- ([\d.]+) s', re.M)
GRID = r'([A-R]{2}\d\d(?:[A-Xa-x]{2})?)\b'


def local(month_day: str, hour_minute: str, year: int = 2026, second: int = 0) -> datetime:
    return datetime(year, int(month_day[:2]), int(month_day[2:]), int(hour_minute[:2]), int(hour_minute[2:]),
                    second, tzinfo=LOCAL).astimezone(UTC)


def utc_stamp(s: str) -> datetime:
    return datetime.strptime(s, '%Y%m%dT%H%M%S').replace(tzinfo=UTC)


def onair_folder(path: Path) -> Path | None:
    """The onair.sh session folder a transcript sits in, however deep."""
    return next((p for p in path.parents if p.name.startswith('onair-')), None)


def folder_time(path: Path) -> datetime | None:
    """The local minute a folder name records, as UTC."""
    from hfmodem.core.capture_identity import evidence_reference
    identity = evidence_reference(path)
    if identity['status'] == 'bound':
        return datetime.fromisoformat(identity['allocated_utc'])
    for part in reversed(path.parts):
        if m := re.fullmatch(r'onair-(\d{4})-(\d{4})', part):
            return local(*m.groups())
        if m := re.fullmatch(r'rig-session-2026(\d{4})-(\d{4})(\d\d)', part):
            return local(m[1], m[2], second=int(m[3]))
    return None


@functools.lru_cache(maxsize=8192)
def text_time(text: str) -> datetime | None:
    """The first instant a transcript states to the second: a UTC stamp, or a log line's own stamp."""
    found = [(m.start(), utc_stamp(m[1])) for m in UTC_STAMP.finditer(text)]
    for m in LOG_STAMP.finditer(text):
        t = datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S')
        found.append((m.start(), t.replace(tzinfo=UTC) if m[2] else t.replace(tzinfo=LOCAL).astimezone(UTC)))
    return min(found, default=(0, None))[1]


@dataclass
class Attempt:
    protocol: str
    call: str
    centre_hz: int | None
    dial_hz: int | None
    bw_hz: int | None
    text: str
    mode: str = 'sense'
    started: datetime | None = None
    ended: datetime | None = None
    source: Path | None = None
    family: str = ''
    joined: dict = field(default_factory=dict)
    audio: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    capture_reference: dict = field(default_factory=dict)

    @functools.cached_property
    def reached(self) -> list[str]:
        return link.milestones(self.text, self.protocol)

    @functools.cached_property
    def mail(self) -> dict[str, list[str]]:
        return link.mail(self.text)

    @functools.cached_property
    def line(self) -> str:
        return summary(self)

    @property
    def base(self) -> str:
        return self.call.upper().split('-')[0]


def attempts(text: str) -> list[Attempt]:
    """The attempts one transcript holds: one per gateway it called, retries folded in."""
    heads = list(HEADER.finditer(text))
    found: list[Attempt] = []
    for i, h in enumerate(heads):
        protocol = next(p for p in ('pactor', 'vara', 'ardop') if h[p])
        start = h.start() if i else 0
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        if found and (found[-1].protocol, found[-1].call) == (protocol, h['call']):
            found[-1].text += text[start:end]
            continue
        forced = 'FORCED' in h['rest'] or '--force: calling anyway' in text[start:end]
        found.append(Attempt(protocol, h['call'], int(h['centre']), int(h['dial']), h['bw'] and int(h['bw']),
                             text[start:end], mode='force' if forced else 'sense'))
    if not found and (w := WAITING.search(text)):
        centre = round(float(w[1]) * 1000)
        found.append(Attempt(w[2], w[3], centre, centre - 1500, None, text))
    if WAITING.search(text):
        for a in found:
            a.mode = 'pounce'
    return found


def summary(a: Attempt) -> str:
    """One line of what the transcript shows, in its own words where it has them."""
    reached = a.reached
    own = [m[1] or m[2] for m in OWN_WORD.finditer(a.text)]
    if 'connect' not in reached and (failed := FAILED.search(a.text)):
        own = [failed[1]]
    if not own and (stopped := STOPPED.search(a.text)):
        own = [f'Stopped by {stopped[1]}.']
    if 'keyed' not in reached:
        why = REFUSAL.findall(a.text) or NOT_TRANSMITTING.findall(a.text)
        said = why[0].strip() if why else 'the transcript ends before anything was keyed'
        return clip(f"Not keyed{' after the channel cleared' if 'cleared' in reached else ''}: {said}")
    if moved := mail_words(len(a.mail['sent']), len(a.mail['received'])):
        what = f'Connected; {moved}'
    elif 'mail' in reached:
        what = 'Connected and moved mail'
    elif 'login' in reached:
        what = 'Connected, logged in, no message moved'
    elif 'greeting' in reached:
        what = "Connected; the gateway's greeting arrived"
    elif 'connect' in reached:
        what = 'Connected' + ('; the peer granted the PACTOR-3 entry' if 'grant' in reached else '')
    elif 'answer' in reached:
        what = 'Keyed and the gateway answered; no link came up'
    else:
        what = 'Keyed; nothing decoded back'
    return clip(f'{what}. {own[-1].strip()}' if own else f'{what}.')


def clip(line: str, n: int = 280) -> str:
    line = ' '.join(line.split())
    return line if len(line) <= n else line[:n - 1].rstrip() + '…'


def version(path: Path) -> tuple[Path, int, int]:
    st = path.stat()
    return path, st.st_mtime_ns, st.st_size


@functools.lru_cache(maxsize=8192)
def transcript(path: Path, mtime_ns: int, size: int) -> tuple[str, str]:
    """A transcript's digest and its text without colour, read once for each version of it."""
    raw = path.read_bytes()
    return hashlib.sha1(raw).hexdigest(), link.ANSI.sub('', raw.decode(errors='replace'))


@functools.lru_cache(maxsize=8192)
def recorded(path: Path, mtime_ns: int, size: int) -> dict:
    from hfmodem.core.capture_identity import metadata_record
    return metadata_record(path)


@functools.lru_cache(maxsize=8192)
def speed(protocol: str, text: str) -> tuple:
    return tuple(getattr(link, protocol)(text).items())


@functools.lru_cache(maxsize=8192)
def vara_speed(analyzer, text: str, bw: int | None) -> tuple[str | None, str | None]:
    """The fastest decoded and acknowledged, as the analyzer reads records and continues and as
    the transcript states the level of each of the peer's overs it read."""
    decoded, acknowledged = link.vara_overs(text), None
    if analyzer:
        report = analyzer.analyze_lines([(None, n, line) for n, line in enumerate(text.splitlines(), 1)], bw=bw)

        def fastest(*kinds):
            return link.fastest(*(f'L{n}' for group in report['by_bw'].values() for k in kinds for n in group[k]))
        decoded = link.fastest(decoded, fastest('rx_decoded', 'rx_reported_decoded'))
        acknowledged = fastest('peer_acknowledged')
    return decoded, acknowledged


class Converter:
    def __init__(self, root: Path, *, book=None, stop=None):
        self.root = root.resolve()
        self.working = self.root / 'working'
        self.out = self.working / 'legacy-air' / 'arms'
        self.book = book or Logbook(self.root)
        self.stop = stop
        self.mycall = self.book.station['call']
        self.skipped: Counter = Counter()
        self.skipped_names: dict[str, list[str]] = defaultdict(list)
        self.deduped: Counter = Counter()
        self.parsed: dict[tuple, list[Attempt]] = {}
        from consolelib import vara_evidence
        self.analyzer = vara_evidence

    def check_cancelled(self):
        if self.stop is not None and self.stop.is_set():
            raise InterruptedError("logbook conversion stopped")

    def rel(self, path: Path | str) -> str:
        return str(Path(path).resolve().relative_to(self.root))

    def skip(self, why: str, path: Path):
        self.skipped[why] += 1
        self.skipped_names[why].append(self.rel(path))

    def native(self) -> list[dict]:
        found = []
        for p in self.working.glob('*/arms/*/provenance.json'):
            self.check_cancelled()
            if p.is_relative_to(self.out):
                continue
            r = recorded(*version(p))
            found.append(dict(protocol=r['arm'].split('-', 1)[0], base=r['call'].upper().split('-')[0],
                              centre=r.get('centre_hz'), started=datetime.fromisoformat(r['started_utc']),
                              arm=r['arm'], dir=p.parent, capture_id=r.get('capture_id'), registry_sha256=r.get('registry_sha256')))
        return found

    def supervisor(self) -> dict[str, dict]:
        """The supervisor's flat records by label, which is the transcript's name."""
        return {r['label']: dict(r, record=self.rel(p)) for p in self.working.glob('*/arms/*.json')
                if (r := recorded(*version(p))).get('label')}

    def sources(self) -> list[tuple[str, Path]]:
        arms = {p.parent.name for p in self.working.glob('*/arms/*/provenance.json')}
        found = []
        for p in sorted(self.working.glob('onair-*/**/*.log')):
            self.check_cancelled()
            if p.stem in arms:
                self.skip('already an arm', p)
            elif SKIP_NAMES.match(p.name):
                self.skip('witness, rehearsal, tap, monitor or sabir log', p)
            else:
                found.append(('onair', p))
        found += [('rig-session', p) for p in sorted(self.working.glob('rig-session-202608*/session.log'))]
        loose = sorted(self.working.glob('*.log')) + sorted(q for d in LOOSE_DIRS for q in self.working.glob(f'{d}/*.log'))
        for p in loose:
            self.check_cancelled()
            if SKIP_NAMES.match(p.name):
                self.skip('witness, rehearsal, tap, monitor or sabir log', p)
            else:
                found.append(('loose', p))
        return found

    def fingerprint(self) -> int:
        """Every input a run reads, as it stands: the transcripts, the arms and the supervisor's
        records, and the folders the recordings land in."""
        paths = [*self.working.glob('onair-*/**/*.log'), *self.working.glob('rig-session-202608*/session.log'),
                 *self.working.glob('*.log'), *(q for d in LOOSE_DIRS for q in self.working.glob(f'{d}/*.log')),
                 *self.working.glob('*/arms/*/provenance.json'), *self.working.glob('*/arms/*.json'),
                 self.root / 'logs' / 'onair', *(self.root / 'captures').glob('onair-*')]
        return hash(tuple(sorted(version(p) for p in paths if p.exists())))

    def read(self) -> list[Attempt]:
        seen: dict[str, Path] = {}
        for n in self.native():
            self.check_cancelled()
            if (log := n['dir'] / 'arm.log').is_file():
                identity = n.get('capture_id')
                key = ('bound', identity, n.get('registry_sha256')) if identity else transcript(*version(log))[0]
                seen[key] = log
        supervised = self.supervisor()
        found = []
        parsed = {}
        for family, path in self.sources():
            self.check_cancelled()
            key = version(path)
            digest, text = transcript(*key)
            from hfmodem.core.capture_identity import evidence_reference
            identity = evidence_reference(path)
            if identity['status'] == 'bound':
                if identity['kind'] != 'arm':
                    self.skip('registered non-arm evidence', path)
                    continue
                digest = ('bound', identity['capture_id'], identity['registry_sha256'])
            if digest in seen:
                self.skip('identical to another transcript', path)
                continue
            seen[digest] = path
            s = supervised.get(path.stem)
            key += (family, s and json.dumps(s, sort_keys=True))
            if key in self.parsed:
                # Read before, and nothing it was read from has changed since.
                parsed[key] = got = self.parsed[key]
                for a in got:
                    a.duplicates, a.audio = [], []
                    self.place(a, text, len(got) == 1)
                found += got
                continue
            if family == 'loose' and re.search(r'recording to \S*rig-session-', text):
                self.skip('copy of a rig session', path)
                continue
            got = attempts(text)
            # A claimed rig refuses before anything is ours; a later run appended to the same log may still key.
            claimed = 'THE RIG IS ALREADY CLAIMED' in text and not any('keyed' in a.reached for a in got)
            if NOT_ATTEMPTS.search(text) and (not got or claimed):
                self.skip('hardware check, teardown, claimed rig or listen-only', path)
                continue
            if not got and (s := supervised.get(path.stem)):
                # The launcher refused its arguments before the header; the supervisor knew the channel.
                got = [Attempt(s['command'][1].split('-')[0], s['gateway'], s['centre_hz'], s['usb_dial_hz'],
                               s.get('bw_hz'), text)]
            if not got:
                self.skip('no attempt header', path)
                continue
            for a in got:
                a.source, a.family = path, family
                if a.mode == 'sense' and re.search(r'-force\b', path.stem):
                    a.mode = 'force'
                a.joined = s or {}
                self.place(a, text, len(got) == 1)
                found.append(a)
            parsed[key] = got
        self.parsed = parsed
        return found

    def place(self, a: Attempt, whole: str, only: bool):
        """When the attempt ran: a UTC stamp in the name, the supervisor's, the transcript's own, or the folder's."""
        from hfmodem.core.capture_identity import evidence_reference
        a.capture_reference = evidence_reference(a.source)
        if a.capture_reference['status'] == 'bound':
            a.started = datetime.fromisoformat(a.capture_reference['allocated_utc'])
            ended = a.capture_reference.get('arm_ended_utc')
            a.ended = datetime.fromisoformat(ended) if ended else None
            return
        s = a.joined
        a.started = a.ended = None
        named = UTC_STAMP.search(a.source.name)
        folder = folder_time(a.source)
        stated = text_time(a.text) or (text_time(whole) if only else None)
        if named and not (folder and utc_stamp(named[1]) < folder - timedelta(minutes=1)):
            a.started = utc_stamp(named[1])
        elif s.get('utc'):
            a.started = datetime.strptime(s['utc'], '%Y%m%dT%H%M%S.%fZ').replace(tzinfo=UTC)
        elif folder and stated and timedelta(minutes=-2) <= stated - folder <= timedelta(hours=1):
            a.started = stated
        elif folder or stated:
            a.started = folder or stated
        elif m := CAPTURES.search(whole):
            a.started = local(*m.groups())
        length = STREAM_S.search(a.text)
        elapsed = (s.get('supervision') or {}).get('elapsed_s')
        mtime = datetime.fromtimestamp(a.source.stat().st_mtime, UTC)
        if a.started is None:
            a.started = mtime - timedelta(seconds=float(length[1])) if length else mtime
        if length:
            a.ended = a.started + timedelta(seconds=float(length[1]))
        elif elapsed:
            a.ended = a.started + timedelta(seconds=elapsed)
        elif only and timedelta(0) <= mtime - a.started <= timedelta(hours=2):
            a.ended = mtime
        a.started = a.started.replace(microsecond=0)
        a.ended = a.ended and a.ended.replace(microsecond=0)

    def dedupe(self, found: list[Attempt]) -> list[Attempt]:
        """One record per attempt: an arm wins, then an onair transcript, a rig session, a loose log."""
        native = defaultdict(list)
        for n in self.native():
            self.check_cancelled()
            native[n['protocol'], n['base']].append(n)
        order = {'onair': 0, 'rig-session': 1, 'loose': 2}
        kept: list[Attempt] = []
        same = defaultdict(list)
        for a in sorted(found, key=lambda a: (order[a.family], a.started)):
            self.check_cancelled()
            identity = a.capture_reference.get('capture_id')
            if identity:
                matching = [n for n in native[a.protocol, a.base] if n.get('capture_id') == identity]
                if any(n.get('registry_sha256') != a.capture_reference['registry_sha256'] for n in matching):
                    raise ValueError('Capture registries disagree')
                if matching:
                    self.deduped[a.family, 'arm'] += 1
                    continue
                twin = next((k for k in kept if k.capture_reference.get('capture_id') == identity
                             and (k.protocol, k.base) == (a.protocol, a.base)), None)
                if twin and twin.capture_reference['registry_sha256'] != a.capture_reference['registry_sha256']:
                    raise ValueError('Capture registries disagree')
                if twin:
                    twin.duplicates.append(self.rel(a.source))
                    self.deduped[a.family, twin.family] += 1
                    continue
                kept.append(a)
                continue
            def near(protocol, base, centre, t):
                return ((protocol, base) == (a.protocol, a.base) and centre in (a.centre_hz, None)
                        and abs((t - a.started).total_seconds()) <= SAME_ATTEMPT_S)
            if any(near(n['protocol'], n['base'], n['centre'], n['started']) for n in native[a.protocol, a.base] if not n.get('capture_id')):
                self.deduped[(a.family, 'arm')] += 1
                continue
            twin = next((k for k in same[a.protocol, a.base] if not k.capture_reference.get('capture_id') and k.family != a.family
                         and near(k.protocol, k.base, k.centre_hz, k.started)), None)
            if twin:
                twin.duplicates.append(self.rel(a.source))
                self.deduped[(a.family, twin.family)] += 1
                continue
            kept.append(a)
            same[a.protocol, a.base].append(a)
        return kept

    def recordings(self) -> dict[str, list[Path]]:
        """Our receiver's logs/onair recordings by what follows the stamp: MYCALL-PEER or besra-DIAL."""
        found = defaultdict(list)
        for p in sorted((self.root / 'logs' / 'onair').glob('*')):
            self.check_cancelled()
            if p.suffix in ('.wav', '.flac') and UTC_STAMP.match(p.name):
                found[p.stem.split('-', 1)[1]].append(p)
        return found

    def listen(self, a: Attempt, shared: set[Path], tapes: dict[str, list[Path]]):
        """Every recording of the attempt already on disk: the ones the transcript names, then
        its folder's for a kind it named none of, then our receiver's by the time it started.
        A recording compressed to FLAC since is found under that name."""
        allowed = [self.root / d for d in ('captures', 'logs/onair', 'logs/sense', 'working')]

        def ok(p: Path) -> Path | None:
            p = audio.on_disk(p.resolve())
            return p if p and any(p.is_relative_to(d) for d in allowed) else None

        found: dict[str, str] = {}
        for kind, path in audio.named(a.text):
            if p := ok(Path(path) if path.startswith('/') else self.root / path):
                found.setdefault(self.rel(p), kind)
        if (folder := onair_folder(a.source)) and folder not in shared:
            for kind, name in (('sdr witness', 'witness.wav'), ('rig rx', 'stream.wav')):
                if kind not in found.values() and (p := ok(self.root / 'captures' / folder.name / name)):
                    found[self.rel(p)] = kind
        if 'rig rx' not in found.values():
            tag = f'{self.mycall}-{a.call}' if a.protocol == 'vara' else f'besra-{a.dial_hz}'
            until = a.ended or a.started + timedelta(minutes=10)
            for p in tapes.get(tag, ()):
                if a.started - timedelta(seconds=90) <= utc_stamp(UTC_STAMP.match(p.name)[1]) <= until:
                    found[self.rel(p)] = 'rig rx'
                    break
        a.audio = [{'path': path, 'kind': kind} for path, kind in found.items()]

    def link(self, a: Attempt) -> dict:
        """How wide the link ran and how fast it got, as the arms' own readers would have it."""
        if a.protocol in ('pactor', 'ardop'):
            got = dict(speed(a.protocol, a.text))
        else:
            bw = a.joined.get('bw_hz') or a.bw_hz or link.vara_bw(a.text)
            decoded, acknowledged = vara_speed(self.analyzer, a.text, bw)
            got = {'bw_hz': bw, 'decoded': decoded, 'acknowledged': acknowledged}
        return {k: v for k, v in got.items() if v}

    def record(self, a: Attempt, now: str, grids: dict[str, str]) -> tuple[str, dict]:
        band = band_m(a.centre_hz) if a.centre_hz else None
        name = f"{a.protocol}-legacy-{re.sub(r'[^a-z0-9-]', '-', a.call.lower())}-{band}m-{a.started:%Y%m%dT%H%M%S}Z"
        if a.capture_reference.get('capture_id'):
            name += '-source-' + a.capture_reference['capture_id']
        s = a.joined
        # The TX power itself is not stored here: the viewer reads it straight off
        # the transcript, the one place link.tx_power ever has to look for it.
        level = {'name': s['row']} if s.get('row') else {}
        peer = {'call': a.call, 'band_m': band, 'centre_hz': a.centre_hz, 'dial_hz': a.dial_hz}
        if grid := grids.get(a.base):
            peer['grid'] = grid
        mode = 'force' if s.get('force_once') else 'pounce' if s.get('pounce_s') else a.mode
        source = {'converted_from': self.rel(a.source), 'converter': CONVERTER, 'converted_utc': now}
        if a.capture_reference.get('capture_id'):
            source['capture_reference'] = a.capture_reference
        if a.duplicates:
            source['duplicates'] = sorted(a.duplicates)
        if s:
            source['supervisor'] = s['record']
        record = {
            'arm': name, 'directory': str(self.out / name), 'mode': mode,
            'call': a.call, 'band_m': band, 'centre_hz': a.centre_hz, 'dial_hz': a.dial_hz,
            'peer': peer, 'level': level,
            'started_utc': a.started.isoformat(), 'ended_utc': a.ended and a.ended.isoformat(),
            'returncode': s.get('returncode'),
            'verdict': {'reached': a.reached, 'line': a.line} | ({'mail': a.mail} if any(a.mail.values()) else {}),
            'link': self.link(a), 'audio': a.audio, 'source': source,
        }
        return name, record

    def grids(self, found: list[Attempt]) -> dict[str, str]:
        """Grids for the gateways the listings do not carry, as the transcripts and the ARDOP roster give them."""
        located = self.book.gateways().keys()
        missing = {a.base for a in found if a.base not in located and a.call.upper() not in located}
        said: dict[str, Counter] = defaultdict(Counter)
        peers = self.working / 'ardop-air-0922' / 'peers.json'
        if peers.is_file():
            for p in json.loads(peers.read_text()).get('peers', []):
                if p.get('grid'):
                    said[p['call'].upper().split('-')[0]][p['grid']] += 1
        for a in found:
            self.check_cancelled()
            if a.base in missing:
                for m in re.finditer(rf'\b{re.escape(a.call)}\b[ ,]+\(?{GRID}', a.text):
                    said[a.base][m[1]] += 1
        return {call: said[call].most_common(1)[0][0] for call in missing if said[call]}

    def run(self, write: bool = True) -> dict:
        self.skipped, self.skipped_names, self.deduped = Counter(), defaultdict(list), Counter()
        self.check_cancelled()
        found = self.dedupe(self.read())
        self.check_cancelled()
        shared = {p for p, n in Counter(onair_folder(a.source) for a in found if a.family == 'onair').items() if n > 1}
        tapes = self.recordings()
        for a in found:
            self.check_cancelled()
            self.listen(a, shared, tapes)
        grids = self.grids(found)
        now = datetime.now(UTC).isoformat()
        records = {}
        for a in found:
            self.check_cancelled()
            name, record = self.record(a, now, grids)
            # Two runs a minute apart in folders named to the minute can land on one stamp.
            n = 1
            while name in records:
                n += 1
                name = record['arm'] = f"{record['arm'].rsplit('Z', 1)[0]}Z-{n}"
                record['directory'] = str(self.out / name)
            records[name] = record
        if write:
            self.write(records)
        return {'attempts': found, 'records': records, 'grids': grids}

    def pending(self) -> dict[str, dict]:
        """What a run would write that is not written yet, under the names it would be written with."""
        return {name: r for name, r in self.run(write=False)['records'].items()
                if not (self.out / name / 'provenance.json').is_file()}

    def write(self, records: dict[str, dict]):
        self.out.mkdir(parents=True, exist_ok=True)
        for old in self.out.iterdir():
            if old.name not in records:
                shutil.rmtree(old)
        for name, record in records.items():
            path = self.out / name / 'provenance.json'
            if path.is_file():
                before = json.loads(path.read_text())
                record['source']['converted_utc'] = before.get('source', {}).get('converted_utc', record['source']['converted_utc'])
                if before == record:
                    continue
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', type=Path, default=None, help='repository holding working/ and station.toml')
    ap.add_argument('--dry-run', action='store_true', help='read and count, write nothing')
    args = ap.parse_args()
    conv = Converter(args.root or default_root())
    got = conv.run(write=not args.dry_run)
    records = got['records'].values()
    print(f"{len(records)} attempts converted to {conv.out}")
    for (family, why), n in sorted(conv.deduped.items()):
        print(f'  {n} {family} attempt{"s" * (n != 1)} already recorded by {"an" if why == "arm" else "a"} {why}')
    for why, n in conv.skipped.most_common():
        print(f"  {n} skipped: {why}, e.g. {', '.join(conv.skipped_names[why][:2])}")


if __name__ == '__main__':
    main()
