# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Every arm in the logbook, heard again through a pinned decoder.

    .venv/bin/python tools/logbook_view/rehear_arms.py run [--commit REV] [--arm ID ...] [--workers 12]
    .venv/bin/python tools/logbook_view/rehear_arms.py census [--commit REV]

The decoders run out of a detached worktree of this repository at REV
(../hfmodem-rehear-REV), never out of the checkout: the checkout carries
uncommitted work, and a rehear is only worth keeping if the decoder that made it
can be named and run again. PYTHONPATH puts the pinned packages ahead of the
venv's editable installs (their finder appends to sys.meta_path, so the path
finder is asked first), bytecode goes to a cache outside the tree, and every
process's candidate modules are checked to have resolved inside it.

Output, beside the arms and never inside them:

    replay/REV/ARM.json          what the audio proves, per arm
    replay/REV/ARM/              each tool's raw output: events.jsonl,
                                         monitor.json, rehear.json, stdout/stderr
    replay/REV/residency.json    what the pinned interpreter loaded
    replay/REV/run.log           starts, pauses, failures

A later decoder writes its own REV directory beside this one. An arm is skipped
when its JSON exists for REV with the same audio (by sha256), the same READER and
no failure, so a killed run picks up where it stopped.

ARM.json:

    rehearer  commit, argv of each tool, residency, run_utc, host, and the READER
              version and sha256 of this file, which does the reading below
    audio     [{path, kind, sha256, seconds}] for every file a decoder read
    identity  the station callsign and the recordings handed to the decoder, which the audio's
              bytes alone do not fix
    heard     reached         milestones proven by decoded frames, in link.py's names
              decoded_max     fastest level the peer sent that decoded with a clean CRC
              acknowledged_max  fastest level of ours the audio shows the peer acknowledging
              bw_hz, callsigns, peer_frames
              unseen_by_live  what tools/rehear decoded that the live path did not
    limits    "WHAT: N" lines, one per thing this decoder saw and could not read; the
              text before the colon is fixed (LIMITS), so a later rehear's JSON can
              be matched against this one
    failed    [{tool, why}] for a tool that failed twice; heard is then partial

Milestones are claimed only on frames. `answer` needs a frame from the peer (its
callsign, or a reply timed to our keying); `connect` needs a connect-ack after
it, or the link carrying the peer's frames on. Nothing here claims login or mail.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

from hfmodem.core.capture_identity import _absolute

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
COMMIT = '56c44eb0'
# The reading of decoder output into `heard` and `limits`, and which recordings plan()
# hands them. Bump it when either changes, and every arm is read again; a change
# anywhere else leaves them be.
READER = 5
WORKERS = 12
# A decoder gets ten times the audio it is given, and a minute to start.
TIMEOUT_X, TIMEOUT_S = 10, 60
# What a tool may print before the middle of it is cut.
KEEP_BYTES = 1 << 20
# An arm is flying if its provenance has no end and was written this recently;
# older than this it is an arm that died without writing one.
FLYING_S = 2 * 3600
PAUSE_S = 60
MIN_FREE_DISK = 20 << 30
# This machine shares its memory with a resident model, so a job starts only with this
# much free, and never within ADMIT_S of the last one: a decoder takes seconds to
# reach its size, and the reading before then says nothing about it.
MIN_FREE_RAM = 3 << 30
ADMIT_S = 3
# Processes that exist only while an arm holds the rig.
KEYERS = re.compile(r'onair\.sh|onair_session\.py|kestrel_connect\.py|ardop_call\.py|tools/lib/rig\.sh|'
                    r'console(?:\.py)? (?:pactor|vara|ardop) (?:fly|campaign)|console(?:\.py)? campaign')
# consolelib/monitor.py: kestrel's `t` is where its window put the burst, up to this
# much ahead of where the burst began. shrike's control signals are placed to the
# slot: on the K0NTS tap of 2026-09-17 they sit in the 0.24 s between our keyings.
EARLY_S = {'kestrel': 0.4}
# A PACTOR control signal answers the keying that ended at most this long before it.
PACTOR_REPLY_S = 1.2
# VARA's connected-ack follows the connect-response within the handshake, and the
# first over follows the link; later than these they may be anybody's on the channel.
VARA_ACK_S, VARA_OVER_S = 10.0, 30.0
# A gateway's connect-response sits in the reply slot after one of our CRs: from just
# inside our mute (kestrel places it early) to a few seconds after it lifts.
VARA_REPLY_S = (-0.4, 3.0)

# The fixed half of every `limits` line.
LIMITS = {
    'high-sl': 'VARA high-SL overs undecoded',
    'vara-over': 'VARA overs undecoded at the base rate',
    'bw2750': 'VARA BW2750 handshakes unattributed',
    'handshake': 'VARA handshakes off the candidate list',
    'ardop-text': 'ARDOP peer payload text not emitted',
    'p2': 'PACTOR-2 shape only',
    'p4': 'PACTOR-4 entry chirp only',
    'windows': 'PACTOR capture windows only, no continuous audio',
    'deaf': 'PACTOR capture windows replayed deaf',
    'witness': 'our keyings not separable on a witness recording',
}

P_HEADER_P1 = re.compile(r'\bP1 DATA (\d+)Bd\b.*\[CRC-VALID')
P_HEADER_P3 = re.compile(r'\bSL(\d) DATA\b.*\[CRC-VALID')
P_TEXT = re.compile(r"b'((?:[^'\\]|\\.)*)'")
P_BYTES = re.compile(r"\b(\d+)B (?:status|b')")
P_TRIALS = re.compile(r'\s*\[CRC-VALID[^\]]*\]')


def signature(detail: str) -> str:
    """A PACTOR packet as decoded, less how many tries the decoder took over it."""
    return P_TRIALS.sub('', detail).strip()
SID = re.compile(r'\[[A-Za-z0-9]+-[^\]\s]*\$\]')
A_FRAME = re.compile(r'^((?:4FSK|4PSK|8PSK|16QAM)\.\d+\.\d+S?)\.[EO]$')
A_BW = re.compile(r'^Con(?:Req|Ack)(\d+)')
V_BW = re.compile(r'\bBW(500|2300|2750)\b')
V_SHAPES = {'unknown', 'one-hot burst', 'DBPSK burst', 'long burst', 'wideband', 'high-SL over'}


# -- what the audio proves ----------------------------------------------------

def base(call: str) -> str:
    return call.split('-')[0].upper()


def fastest(levels) -> str | None:
    import link
    return link.fastest(*levels)


def inside(e: dict, spans) -> bool:
    t = e['t'] + EARLY_S.get(e['modem'], 0.0)
    return any(t0 <= t < t1 for t0, t1 in spans)


def heard(protocol: str, call: str, mycall: str, events: list[dict], spans: list, rehearing: dict | None,
          *, source: str | None, bw_hz: int | None = None) -> tuple[dict, Counter]:
    """(heard, limits) from one arm's decoder output.

    `events` are the monitor's detections on `source`, the kind of recording the
    monitor heard (None when only capture windows were reheard), `spans` our keyings
    read off its receiver mute, and `rehearing` tools/rehear's JSON over the
    recordings it claims. A witness has no mute to read, so on one our own frames
    cannot be set aside and no level is claimed.
    """
    separable = source != 'sdr witness'
    spans = spans if separable else []
    peer = [e for e in events if not inside(e, spans)]
    return {'vara': vara, 'ardop': ardop, 'pactor': pactor}[protocol](
        call, mycall, peer, events, spans, rehearing or {}, separable, bw_hz, source)


def _callsigns(events, mycall, *more) -> list[str]:
    found = {e['station'].upper() for e in events if e.get('station')} | {c.upper() for c in more if c}
    return sorted(c for c in found if base(c) != base(mycall))


def _unseen(rehearing: dict) -> dict | None:
    runs = rehearing.get('rehearings')
    if not runs:
        return None
    lost = []
    for r in runs:
        if r['verdicts'] is not None:
            lost += [[at, what] for at, what, verdict, _ in r['verdicts'] if verdict == 'unseen']
        else:
            lost += [[at, what] for at, what, _ in r['discarded']]
    return {'count': len(lost), 'of': sum(len(r['ungated']) for r in runs),
            'adapter': runs[0]['protocol'], 'first': lost[:20]}


def vara(call, mycall, peer, events, spans, rehearing, separable, bw_hz, source):
    limits = Counter()
    lo, hi = VARA_REPLY_S

    def replied(t):
        return not spans or any(lo <= t - t1 <= hi for _, t1 in spans)
    answers = [e['t'] for e in peer if e['kind'] == 'connect-response' and e.get('station')
               and base(e['station']) == base(call) and replied(e['t'])]
    answers += [at for r in rehearing.get('rehearings') or () for at, what, _ in r['ungated']
                if what.startswith('connect-response ') and base(what.split()[1]) == base(call) and replied(at)]
    reached = []
    if answers:
        reached.append('answer')
        soon = {'connected-ack': VARA_ACK_S, 'DATA over': VARA_OVER_S}
        if any(0 < e['t'] - at <= soon[e['kind']] for e in peer if e['kind'] in soon for at in answers):
            reached.append('connect')
    linked = [e for e in peer if 'connect' in reached and e['t'] > min(answers)]
    overs = [e for e in linked if e['kind'] == 'DATA over']
    # A CR is the caller's frame, whoever's it is; only what answers or follows the link is the peer's.
    heard_from = [e for e in peer if e['kind'] == 'connect-response' and e['t'] in answers] + [
        e for e in linked if e['kind'] not in V_SHAPES | {'CR', 'connect-response'}]
    bands = Counter(int(m[1]) for e in heard_from + [e for e in linked if e['kind'] == 'wideband']
                    for m in [V_BW.search(e['detail'])] if m)
    for e in peer:
        if e['kind'] == 'high-SL over':
            limits[LIMITS['high-sl']] += 1
        elif e['kind'] in ('long burst', 'wideband'):
            limits[LIMITS['vara-over']] += 1
        elif e['kind'] == 'connect-response' and not e.get('station') and replied(e['t']):
            limits[LIMITS['bw2750' if bw_hz == 2750 else 'handshake']] += 1
    if not separable:
        limits[LIMITS['witness']] = 1
    return {
        'reached': reached,
        'decoded_max': 'L4' if overs and separable else None,
        'acknowledged_max': None,
        'bw_hz': bands.most_common(1)[0][0] if bands else None,
        'callsigns': _callsigns(heard_from, mycall, *([call] if answers else [])),
        'peer_frames': len(heard_from),
        'unseen_by_live': _unseen(rehearing),
    }, limits


def ardop(call, mycall, peer, events, spans, rehearing, separable, bw_hz, source):
    """ConAck and ConRej are the answering station's alone. Data frames and DATAACKs
    are either side's, and the adapter's split of them from the session log does not
    hold (KN4LQN 2026-08-29: frames it calls ours are the peer's and the reverse), so
    no level is claimed from them."""
    limits = Counter()
    runs = rehearing.get('rehearings') or []
    if runs:
        frames = sorted((at, what) for r in runs for key in ('ungated', 'ours') for at, what, _ in r[key])
    else:
        frames = sorted((e['t'], e['kind']) for e in peer if e['protocol'] == 'ARDOP')
    answers = [(at, w) for at, w in frames if w.startswith(('ConAck', 'ConRej'))]
    reached = ['answer'] * bool(answers) + ['connect'] * any(w.startswith('ConAck') for _, w in answers)
    data = [at for at, w in frames if A_FRAME.match(w) and 'connect' in reached]
    bands = Counter(int(m[1]) for _, w in answers for m in [A_BW.match(w)] if m)
    if data:
        limits[LIMITS['ardop-text']] = len(data)
    if not separable:
        limits[LIMITS['witness']] = 1
    return {
        'reached': reached,
        'decoded_max': None,
        'acknowledged_max': None,
        'bw_hz': bands.most_common(1)[0][0] if bands else None,
        'callsigns': [],
        'peer_frames': len(answers),
        'unseen_by_live': _unseen(rehearing),
    }, limits


def pactor(call, mycall, peer, events, spans, rehearing, separable, bw_hz, source):
    limits = Counter()
    ends = [t1 for _, t1 in spans]
    # A control signal is ours to count when it sits in the answer slot of one of our
    # keyings; on a capture window every sample is already the slot after our carrier.
    replies = sorted({round(t1, 2) for e in peer if e['kind'] == 'CS'
                      for t1 in ends if 0 <= e['t'] - t1 <= PACTOR_REPLY_S})
    windows = [r for r in rehearing.get('rehearings') or () if r['protocol'] == 'shrike']
    answered = [(r['recording'], at) for r in windows for at, what, _ in r['ungated'] if what.startswith('cs ')]
    reached = []
    first = min((e['t'] for e in peer if e['kind'] == 'CS'
                 and any(0 <= e['t'] - t1 <= PACTOR_REPLY_S for t1 in ends)), default=None)
    # Our own packets leak wherever the mute reading misses a keying, so a packet is the
    # peer's only if it carries data and no copy of it decoded under our keying.
    ours = {signature(e['detail']) for e in events if e['kind'] == 'HEADER' and inside(e, spans)}
    headers = [e for e in peer if e['kind'] == 'HEADER' and first is not None and e['t'] > first
               and signature(e['detail']) not in ours and (m := P_BYTES.search(e['detail'])) and int(m[1])]
    levels = [f'P1 {m[1]}Bd' for e in headers if (m := P_HEADER_P1.search(e['detail']))]
    levels += [f'P3 SL{m[1]}' for e in headers if e['protocol'] == 'PACTOR-3'
               and (m := P_HEADER_P3.search(e['detail']))]
    if replies or answered:
        reached.append('answer')
        # The ARQ cycle went round more than once, or the peer's own packets decoded.
        if len(replies) >= 2 or levels or len({rec for rec, _ in answered}) >= 2:
            reached.append('connect')
    grants = [e for e in peer if e['kind'] == 'UNASSIGNED' and '0x59A' in e['detail']
              and '(0 bit errors' in e['detail'] and first is not None and e['t'] > first]
    if 'connect' in reached and grants:
        reached.append('grant')
    text = ''.join(m[1] for e in headers if (m := P_TEXT.search(e['detail'])))
    if 'connect' in reached and SID.search(text):
        reached.append('greeting')
    frames = [e for e in peer if e['kind'] in ('CS', 'HEADER', 'UNASSIGNED')]
    limits[LIMITS['p2']] = sum(e['kind'] == 'P2-PAIR' for e in peer)
    limits[LIMITS['p4']] = sum(e['protocol'] == 'PACTOR-4' for e in peer)
    if windows and source is None:
        limits[LIMITS['windows']] = 1
    limits[LIMITS['deaf']] = sum(bool(r['deaf']) for r in windows)
    if not separable:
        limits[LIMITS['witness']] = 1
    return {
        'reached': reached,
        'decoded_max': fastest(levels) if separable else None,
        'acknowledged_max': 'P1 100Bd' if 'connect' in reached else None,
        'bw_hz': None,
        # A CONNECT frame is our own call to the peer as often as not, and a garbled one
        # names UK0NTS for K0NTS; only a Free Signal or Robust Call names its sender.
        'callsigns': _callsigns([e for e in peer if e['kind'] == 'CALLB'], mycall),
        'peer_frames': len(frames) + len(answered),
        'unseen_by_live': _unseen(rehearing),
    }, +limits


# -- which recordings, and which decoders ----------------------------------------

@dataclasses.dataclass
class Job:
    arm: str
    protocol: str
    call: str
    bw_hz: int | None
    transcript: Path | None
    primary: tuple[str, Path] | None           # (kind, path) the monitor hears
    claimed: list[Path]                        # the recordings tools/rehear is handed
    windows: list[Path]                        # which of `claimed` are capture windows
    selection: dict | None = None
    bounds: tuple[int, int] | None = None
    clock_generation: int | None = None
    context_id: str | None = None
    context_version: int | None = None

    def audio(self) -> list[tuple[str, Path]]:
        out = [self.primary] if self.primary else []
        out += [('rig rx', p) for p in self.claimed
                if p not in self.windows and (not self.primary or p != self.primary[1])]
        out += [('capture window', p) for p in self.windows]
        return out


PRIMARY = {'vara': ('rig rx', 'rig tap', 'sdr witness'),
           'ardop': ('rig rx', 'rig tap', 'sdr witness'),
           'pactor': ('rig tap', 'rig rx', 'sdr witness')}
CAPTURES = re.compile(r'^\s*captures: (\S+)\s*$|^session stream: (\S+)/stream\.wav', re.M)
WINDOW = re.compile(r'(?:rx|hold)_\d+')
# A converted arm's ended_utc, when its own transcript states no duration, is legacy.py's
# fallback: the transcript file's own mtime, which can trail the session's real last
# capture window by minutes (one arm's tail ran 883 s past it -- an outlier; 120 s alone
# already covers 99.4% of solo legacy arms measured against every window their own
# folder holds). A live arm's ended_utc comes from the run itself and is never short.
LEGACY_SLACK_S = 120.0


def local(root: Path, url: str | None) -> Path | None:
    return url and root / unquote(url.removeprefix('/file/'))


def recorded_path(path: Path, root: Path) -> str:
    path, root = _absolute(path), _absolute(root)
    return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()


def window_time(wav: Path) -> float:
    """A capture window's own timestamp. Its sidecar carries a sample count on the
    capture stream's clock, wall-anchored only where a stream.json sits beside it -- true
    for under 80% of capture folders, and never for the oldest sidecar schema -- so the
    one timestamp every window has is the filesystem's record of when the sidecar itself
    was written, moments after the window was."""
    return wav.with_suffix('.json').stat().st_mtime


def capture_starts(entries: list[dict], root: Path) -> dict[Path, list[float]]:
    """Every pactor arm's own started_utc, keyed by each captures folder its transcript
    names -- computed once over the whole logbook so plan() can bound one arm's window
    span at the next arm's own start, not just its own slack-extended end, when a
    folder is shared."""
    out: dict[Path, list[float]] = {}
    for e in entries:
        if e['protocol'] != 'pactor' or e.get('recording'):
            continue
        t = local(root, e.get('transcript'))
        if not t or not t.is_file():
            continue
        started = datetime.fromisoformat(e['started_utc']).timestamp()
        for m in CAPTURES.finditer(t.read_text(errors='replace', encoding='utf-8')):
            d = Path(m[1] or m[2])
            d = d if d.is_absolute() else root / d
            out.setdefault(d, []).append(started)
    for starts in out.values():
        starts.sort()
    return out


def own_windows(entry: dict, d: Path, found: list[Path], starts: dict[Path, list[float]] | None) -> list[Path]:
    """Which of a captures folder's windows are this arm's: those whose own timestamp
    falls in its [started_utc, ended_utc) span, with a legacy arm's slack on the end --
    capped at the next arm's own start, by start time, if `starts` (from
    capture_starts()) says another arm's transcript also names this folder. Two arms
    end up sharing a folder when an onair session was retried into it, or a legacy
    transcript covers more than one attempt; each then takes only its own."""
    from hfmodem.core.capture_identity import recording_windows
    reference = (entry.get('source') or {}).get('capture_reference') or {}
    identity = entry.get('capture_id') or reference.get('capture_id')
    if identity:
        originals = {_absolute(p): p for p in found}
        found = [originals[_absolute(p)] for p in recording_windows(
            found, capture_id=identity, clock_id=entry.get('window_clock_id'))]
        rec = entry.get('recording')
        if rec:
            refs = list(rec.get('lanes', {}).values())
            selected = []
            for path in found:
                try:
                    side = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                    a, b = side['actual']
                    if any(side.get('source_id') == r.get('source_id')
                           and side.get('clock_generation') == r.get('clock_generation')
                           and a < r.get('selection', r['actual'])[1]
                           and b > r.get('selection', r['actual'])[0] for r in refs):
                        selected.append(path)
                except (OSError, ValueError, KeyError):
                    continue
            return selected
        return found
    lo = datetime.fromisoformat(entry['started_utc']).timestamp()
    hi = datetime.fromisoformat(entry['ended_utc']).timestamp() if entry.get('ended_utc') else float('inf')
    if '-legacy-' in entry['id']:
        hi += LEGACY_SLACK_S
    nexts = [t for t in (starts or {}).get(d, ()) if t > lo]
    if nexts:
        hi = min(hi, min(nexts))
    return sorted(w for w in found if lo <= window_time(w) < hi)


def plan(entry: dict, root: Path, starts: dict[Path, list[float]] | None = None, *,
         mycall: str = '') -> Job | None:
    """The recordings an entry is reheard from, or None if it has none worth a decoder.

    `starts` (capture_starts() over the whole logbook) bounds a shared captures folder's
    windows at the next arm's own start; without it, only this arm's own span and slack
    apply, which is all a lone entry (as in most of the tests) can see anyway."""
    protocol = entry['protocol']
    if protocol not in PRIMARY:
        return None
    found = list(dict.fromkeys((a['kind'], local(root, a.get('recording') or a.get('url')))
                               for a in entry['audio'] if a['kind'] in ('rig rx', 'rig tap', 'sdr witness')))
    from hfmodem.core.wav import segments
    from hfhost.recordings import available
    rec = entry.get('recording')
    if rec and (rec.get('policy') or rec.get('lanes') or 'context' in rec):
        from hfhost.audio import WavReplaySource, AudioError
        def available(path):
            try:
                with WavReplaySource(path, realtime=False, selection=rec) as source:
                    source.selection_reference()
                return True
            except (OSError, ValueError, AudioError):
                return False
    found = [(k, p) for k, p in found if p and p.suffix in ('.wav', '.flac')
             and available(p)]
    primary = next(((k, p) for want in PRIMARY[protocol] for k, p in found if k == want), None)
    transcript = local(root, entry.get('transcript'))
    transcript = transcript if transcript and transcript.is_file() else None
    if (entry.get('recording') or {}).get('policy') or (entry.get('recording') or {}).get('lanes') or 'context' in (entry.get('recording') or {}):
        from hfmodem.station import sessionlog
        rec = entry['recording']
        selected = sessionlog.select(rec, primary[1], root=root) if primary else None
        ident = sessionlog.identity(rec, selected.lane if selected else None)
        return Job(entry['id'], protocol, entry['call'], entry.get('bw_hz'), transcript,
                   primary, [p for _, p in found], [], rec,
                   tuple(ident['bounds']) if ident['bounds'] else None,
                   ident['clock_generation'], ident['context_id'], ident['context_version']) if primary else None
    claimed, windows, folders = [], [], set()
    if protocol == 'vara':
        mine = mycall.upper()
        claimed = [p for k, p in found if k == 'rig rx' and mine and f'-{mine}-' in p.name.upper()]
    elif protocol == 'ardop':
        claimed = [p for k, p in found if k == 'rig rx' and '-besra-' in p.name]
    elif transcript:
        from hfmodem.core.capture_identity import evidence_reference, IdentityError
        reference = evidence_reference(transcript)
        if reference['status'] == 'bound':
            if entry.get('capture_id') not in (None, reference['capture_id']):
                raise IdentityError('Transcript belongs to a different selected capture')
            from hfmodem.core.capture_identity import validate_artifact
            for _, audio_path in found:
                for part in segments(audio_path):
                    binding = validate_artifact(part)
                    if binding.get('capture_id') != reference['capture_id']:
                        raise IdentityError('Audio belongs to a different selected capture')
            d = Path(reference['primary_root'])
            if _absolute(d).is_relative_to(_absolute(root)):
                d = root / _absolute(d).relative_to(_absolute(root))
            here = [p for p in d.iterdir() if p.is_file() and p.suffix in {'.wav', '.flac'}
                    and WINDOW.fullmatch(p.stem) and p.with_suffix('.json').is_file()]
            bound_entry = dict(entry, capture_id=reference['capture_id'])
            windows = own_windows(bound_entry, d, here, starts)
            claimed = windows
            if not primary and not windows:
                return None
            return Job(entry['id'], protocol, entry['call'], entry.get('bw_hz'),
                       transcript, primary, claimed, windows)
        text = transcript.read_text(errors='replace', encoding='utf-8')
        for m in CAPTURES.finditer(text):
            d = Path(m[1] or m[2])
            d = d if d.is_absolute() else root / d
            here = [p for p in d.glob('*.wav') if WINDOW.fullmatch(p.stem) and p.with_suffix('.json').is_file()]
            if d not in folders:
                folders.add(d)
                windows += own_windows(entry, d, here, starts)
        claimed = windows
    if not primary and not windows:
        return None
    return Job(entry['id'], protocol, entry['call'], entry.get('bw_hz'), transcript, primary, claimed, windows)


# -- the pinned decoder -----------------------------------------------------------

@dataclasses.dataclass
class Pinned:
    commit: str
    root: Path            # the checkout, whose data and venv this reads
    tree: Path            # the detached worktree the decoders run from
    out: Path             # the per-revision replay output directory

    @classmethod
    def at(cls, root: Path, commit: str) -> 'Pinned':
        return cls(commit, root, root.parent / f'{root.name}-rehear-{commit}', root / 'working' / 'rehear' / commit)

    @property
    def python(self) -> Path:
        return self.root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

    @property
    def stage(self) -> Path:
        """Where the decoders start, so nothing they write lands in the tree."""
        return self.out / 'stage'

    @property
    def lists(self) -> Path:
        """The gateway list, where the pinned tools/hfcapture.py reads it: the tree's
        root, a path the tree's own .gitignore keeps out of git. It is the one file
        put in the tree, read-only, and its sha256 goes in every arm's JSON."""
        return self.tree / 'winlink-vara-gateways.csv'

    def env(self, tmp: Path) -> dict:
        drop = {'PYTHONPATH', 'PYTHONHOME', 'CREANCE_MODEM_ROOT', 'CREANCE_MODEM_PYTHON', 'VIRTUAL_ENV'}
        env = {k: v for k, v in os.environ.items() if k not in drop}
        path = [self.tree / 'packages' / p for p in ('hfmodem', 'hfhost', 'creance')] + [self.tree / 'tools']
        return env | {
            'PYTHONPATH': os.pathsep.join(map(str, path)),
            'PYTHONPYCACHEPREFIX': str(self.out / 'pycache'),
            'PYTHONNOUSERSITE': '1', 'TMPDIR': str(tmp),
            # Twelve decoders at once, each on one core.
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
        }

    def prepare(self) -> dict:
        """Check the tree is the commit and untouched, stage the runners, take the residency."""
        if not self.tree.is_dir():
            subprocess.run(['git', '-C', str(self.root), 'worktree', 'add', '--detach', str(self.tree), self.commit],
                           check=True, capture_output=True)
        head = git(self.tree, 'rev-parse', 'HEAD')
        if not head.startswith(git(self.root, 'rev-parse', self.commit)[:len(head)]):
            raise SystemExit(f'{self.tree} is at {head}, not {self.commit}')
        self.untouched()
        self.stage.mkdir(parents=True, exist_ok=True)
        if not self.lists.exists():
            shutil.copyfile(self.root / 'winlink-vara-gateways.csv', self.lists)
            self.lists.chmod(0o444)
        probe = self.out / 'residency.json'
        if not probe.exists():
            tmp = self.out / 'tmp'
            tmp.mkdir(parents=True, exist_ok=True)
            done = subprocess.run([str(self.python), '-c', PROBE, *PROBED], cwd=self.stage, env=self.env(tmp),
                                  capture_output=True, text=True, timeout=300)
            if done.returncode:
                raise SystemExit(f'the pinned interpreter could not import its decoders: {done.stderr[-2000:]}')
            loaded = json.loads(done.stdout)
            self.check(loaded['modules'], 'probe')
            atomic(probe, dict(loaded, commit=head, tree=str(self.tree), modules_probed=PROBED))
        return json.loads(probe.read_text(encoding='utf-8')) | {'commit': head, 'lists_sha256': sha256(self.lists)}

    def untouched(self):
        lists = f'!! {self.lists.name}'
        dirty = '\n'.join(line for line in git(self.tree, 'status', '--porcelain', '--ignored').splitlines()
                          if line != lists)
        if dirty:
            raise SystemExit(f'{self.tree} has been written to:\n{dirty[:2000]}')

    def check(self, modules: dict, who: str):
        stray = [f"{name} <- {m['file']}" for name, m in modules.items()
                 if not m['file'] or not _absolute(Path(m['file']).resolve()).is_relative_to(
                     _absolute(self.tree.resolve()))]
        if stray:
            raise RuntimeError(f'{who} loaded {len(stray)} module(s) outside {self.tree}: {", ".join(stray[:4])}')


# The modules the runners and tools/rehear import, loaded once to take the residency
# the decoders run under: the runners are subprocesses of their own and report none.
PROBED = ['creance.monitor', 'hfmodem.kestrel.vara.vara_frames', 'hfmodem.besra.phy.demodulator',
          'hfmodem.shrike.rxfront', 'hfmodem.shrike.p4chirp', 'hfmodem.shrike.p4chirprx', 'hfmodem.shrike.live',
          'hfmodem.shrike.onair', 'hfmodem.besra.radio', 'vara_monitor', 'kestrel_connect', 'txwitness',
          'corpuslib.detector', 'rehear', 'rehear.kestrel', 'rehear.besra', 'rehear.shrike']
PROBE = """
import importlib, json, sys
from consolelib import residency
for name in sys.argv[1:]:
    importlib.import_module(name)
loaded = residency.imported()
json.dump(dict(modules=loaded, set_sha256=residency.set_sha256(loaded), label=residency.label(loaded)), sys.stdout)
"""


def git(where: Path, *argv) -> str:
    return subprocess.run(['git', '-C', str(where), *argv], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path: Path, *, selection=None) -> str:
    h = hashlib.sha256()
    if selection is not None and path.suffix in ('.wav', '.flac'):
        from hfmodem.station import sessionlog
        from hfmodem.core import wav
        chosen = sessionlog.select(selection, path)
        if not chosen.bounds:
            raise ValueError('Selected source interval unavailable')
        for span in chosen.spans:
            first, last = span.bounds
            for start in range(first, last, 65536):
                h.update(wav.read(path, start=start, stop=min(last,start+65536)).tobytes())
        return h.hexdigest()
    from hfmodem.core.wav import segments
    for part in segments(path) if path.suffix in ('.wav', '.flac') else [path]:
        from hfhost.recordings import open_segment
        with (open_segment(part) if part.suffix in ('.wav', '.flac') else part.open('rb')) as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def seconds(path: Path, *, selection=None) -> float | None:
    if selection is not None:
        from hfmodem.station import sessionlog
        chosen = sessionlog.select(selection, path)
        return (chosen.bounds[1]-chosen.bounds[0])/chosen.rate if chosen.bounds else None
    from audio import _file_duration
    from hfmodem.core.wav import segments
    lengths = [_file_duration(part) for part in segments(path)]
    return round(sum(lengths), 2) if all(s is not None for s in lengths) else None


def atomic(path: Path, value) -> None:
    part = path.with_name(path.name + '.part')
    part.write_text(json.dumps(value, indent=1, default=str) + '\n', encoding='utf-8')
    os.replace(part, path)


# -- the decoder process ------------------------------------------------------------

def decode(args) -> int:
    """Runs under the pinned tree: the monitor's runners over one recording.

    The runners are creance's, fed the whole file without dropping any of it. Our
    keyings are read off the receiver's mute the way txwitness reads them. The
    PACTOR-4 entry chirp has no runner at every commit, so it is searched here with
    the corpus detector's own `chirp_events`.
    """
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE]
    import queue

    import numpy as np
    import txwitness
    from consolelib import residency
    from creance.monitor.driver import ModemMonitor, default_specs
    from hfhost.audio import FS, WavReplaySource
    from hfmodem.shrike.p4chirp import PACKET_S

    selection = getattr(args, 'selection', None)
    if isinstance(selection, Path):
        selection = json.loads(selection.read_text(encoding='utf-8'))
    specs = [s for s in default_specs(args.base, deep=args.deep) if s.name in args.runners]
    missing = {s.name: why for s in specs if (why := s.unavailable())}
    if missing or len(specs) != len(args.runners):
        raise SystemExit(f'runners not available: {missing or set(args.runners) - {s.name for s in specs}}')
    found: queue.Queue = queue.Queue()
    monitors, short, events = [], [], []
    spans, gaps, current = [], [], []
    origin = end = None
    duration = 0.
    pending = b''
    window_start = 0
    hop = FS
    window = int(np.ceil(PACKET_S * FS)) + hop
    baseline, mute_start, mute_deep, live_start = None, None, False, None
    def finish_mute(at):
        nonlocal mute_start, mute_deep, live_start
        at = live_start if live_start is not None else at
        if mute_start is not None and mute_deep and at-mute_start >= txwitness.MIN_KEYING_S:
            spans.append((mute_start, at))
        mute_start, mute_deep, live_start = None, False, None
    def analyze(raw, first, count=None):
        nonlocal baseline, mute_start, mute_deep, live_start
        audio = np.frombuffer(raw, '<i2').astype(np.float32)/32768.
        offset = (first-origin)/FS
        mute_audio = audio if count is None else audio[:count]
        spectra, freqs, win = txwitness._spectra(mute_audio, FS, txwitness.STEP_S)
        if spectra.size:
            levels = np.array([spectra[:, (freqs > lo) & (freqs < hi)].mean(axis=1)/(win.sum()/2)
                               for lo, hi in zip(txwitness.MUTE_BANDS, txwitness.MUTE_BANDS[1:])])
            if baseline is None:
                baseline = np.maximum([txwitness.receiver_live_level(band) for band in levels], 1e-12)
            for i, level in enumerate(levels.T):
                ratio = np.median(level/baseline)
                at = offset + i*txwitness.STEP_S
                if ratio < txwitness.LIVE_AGAIN:
                    live_start = None
                    if mute_start is None:
                        mute_start = at
                    mute_deep |= ratio < txwitness.MUTED
                else:
                    if mute_start is not None:
                        if live_start is None:
                            live_start = at
                        if at-live_start > txwitness.MUTE_GAP_S:
                            finish_mute(at)
                    if mute_start is None:
                        baseline += (level-baseline)*min(1., txwitness.STEP_S/txwitness.LIVE_TRACK_S)
        if args.p4:
            for event in chirps(audio.astype(float), FS):
                event = dict(event, t=event['t']+offset)
                duplicate = next((i for i,e in enumerate(events) if e['modem'] == 'p4'
                                  and abs(e['t']-event['t']) < 1.), None)
                if duplicate is None:
                    events.append(event)
                elif event['kind'] == 'P4-CHIRP-CRC':
                    events[duplicate] = event
    for first, pcm in WavReplaySource(args.wav, realtime=False, selection=selection).frames():
        if origin is None:
            origin = first
        if end != first:
            if pending:
                analyze(pending, window_start)
            if end is not None:
                finish_mute((end-origin)/FS)
            baseline = None
            if end is not None:
                gaps.append([(end-origin)/FS, (first-origin)/FS])
            short.extend(m.spec.name for m in current if not m.stop(args.grace))
            current = [ModemMonitor(s, found, backlog_s=60, lossy=False) for s in specs]
            monitors.extend(current)
            for m in current:
                m.start()
            pending, window_start = b'', first
        for m in current:
            m.feed(first, pcm)
        pending += pcm
        while len(pending) >= window*2:
            analyze(pending[:window*2], window_start, hop)
            pending = pending[hop*2:]
            window_start += hop
        end = first+len(pcm)//2
    if pending:
        analyze(pending, window_start)
    if end is not None:
        finish_mute((end-origin)/FS)
    short.extend(m.spec.name for m in current if not m.stop(args.grace))
    if origin is None:
        origin = 0
    duration = (end-origin)/FS if end is not None else 0.
    merged = []
    for a,b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a,b))
    spans = merged
    while not found.empty():
        event = dataclasses.asdict(found.get())
        event['t'] -= origin/FS
        events.append(event)
    events.sort(key=lambda e: e['t'])
    with open(args.out / 'events.jsonl', 'w') as f:
        for e in events:
            f.write(json.dumps(e) + '\n')
    loaded = residency.imported()
    summary = dict(wav=str(args.wav), seconds=round(duration, 2), spans=[[round(a, 2), round(b, 2)] for a, b in spans],
                   runners=args.runners, p4=args.p4, events=len(events),
                   dropped_s={s.name: sum(m.dropped_samples for m in monitors if m.spec.name == s.name)/FS for s in specs},
                   killed_at_drain=short, residency=dict(modules=loaded, set_sha256=residency.set_sha256(loaded)))
    if selection is not None:
        from hfmodem.station import sessionlog
        summary.update(selection=sessionlog.identity(selection), gaps=gaps, source_offset_sample=origin)
    (args.out / 'monitor.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
    return 1 if short or any(m.dropped_samples for m in monitors) else 0


def chirps(audio, fs: int, block_s: float = 60.0, lookback_s: float = 4.0) -> list[dict]:
    """The PACTOR-4 entry chirp, block by block with the detector's own lookback."""
    from corpuslib import detector
    out, block, back = [], int(block_s * fs), int(lookback_s * fs)
    for lo in range(0, max(len(audio) - back, 1), block):
        for ev in detector.chirp_events(audio[lo:lo + block + back], 60.0):
            t = round(lo / fs + ev['t'], 3)
            if not any(o['kind'] == ev['kind'] and abs(o['t'] - t) < 1.0 for o in out):
                out.append(dict(ev, t=t, modem='p4', station='', role='', wall=0.0))
    return out


# -- running one arm ---------------------------------------------------------------

def run_tool(argv: list[str], *, cwd: Path, env: dict, timeout: float, out: Path, name: str) -> str | None:
    """Run one decoder at low priority, its output capped; None, or why it failed."""
    def pump(stream, path):
        head, tail, total = b'', b'', 0
        for chunk in iter(lambda: stream.read(65536), b''):
            total += len(chunk)
            room = KEEP_BYTES - len(head)
            head += chunk[:room]
            if len(chunk) > room:
                tail = (tail + chunk[room:])[-KEEP_BYTES // 4:]
        cut = total - len(head) - len(tail)
        path.write_bytes(head + (f'\n[... {cut} bytes cut ...]\n'.encode() if cut else b'') + tail)

    job = None
    if os.name == 'nt':
        from hfmodem.core import winproc
        job = winproc.Job()
        command = argv
        options = {'creationflags': subprocess.BELOW_NORMAL_PRIORITY_CLASS |
                   subprocess.CREATE_NEW_PROCESS_GROUP | 4}
    else:
        command = ['nice', '-n', '19', *argv]
        options = {'start_new_session': True}
    try:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
        if job is not None:
            try:
                job.assign(proc)
                with winproc.open(proc.pid, winproc.END) as entry:
                    winproc.resume(entry)
            except BaseException:
                proc.kill()
                proc.wait()
                proc.stdout.close()
                proc.stderr.close()
                raise
    except BaseException:
        if job is not None:
            job.close()
        raise
    pumps = [threading.Thread(target=pump, args=(s, out / f'{name}.{n}'), daemon=True)
             for s, n in ((proc.stdout, 'stdout'), (proc.stderr, 'stderr'))]
    for p in pumps:
        p.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if job is not None:
            job.terminate()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        code = None
    finally:
        if job is not None:
            job.close()
    for p in pumps:
        p.join(30)
    proc.stdout.close()
    proc.stderr.close()
    if code is None:
        return f'timed out after {timeout:.0f} s'
    if code:
        said = (out / f'{name}.stderr').read_text(errors='replace', encoding='utf-8').strip().splitlines()
        return f'exit {code}: ' + (said[-1][:300] if said else 'no diagnostic')
    return None


def rehear_arm(job: Job, pin: Pinned, book: dict, entry: dict, *, mycall: str, audit=print) -> dict:
    """Run the decoders over one arm and write its JSON; the JSON is returned."""
    audio = [{'path': recorded_path(p, pin.root),
              'kind': kind, 'sha256': sha256(p, selection=job.selection),
              'seconds': seconds(p, selection=job.selection)} for kind, p in job.audio()]
    work = pin.out / f'{job.arm}.partial'
    shutil.rmtree(work, ignore_errors=True)
    (work / 'tmp').mkdir(parents=True)
    env = pin.env(work / 'tmp')
    selection_args = []
    if job.selection is not None:
        atomic(work / 'selection.json', job.selection)
        selection_args = ['--selection', str(work / 'selection.json')]
    argv, failed = {}, []
    events, spans, rehearing, summary = [], [], None, None

    def attempt(name, cmd, timeout, result, modules):
        """The tool's JSON, once it has run clean from the pinned tree; None after two failures."""
        argv[name] = cmd
        for n in range(2):
            why = run_tool(cmd, cwd=pin.stage, env=env, timeout=timeout, out=work, name=name)
            if why is None:
                try:
                    got = json.loads((work / result).read_text(encoding='utf-8'))
                    loaded = got[modules]
                    pin.check(loaded.get('modules', loaded), name)
                    return got
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    why = f'{type(exc).__name__}: {exc}'[:300]
            audit(f'{job.arm}: {name} failed ({why})' + ('; retrying' if n == 0 else ''))
        failed.append({'tool': name, 'why': why})
        return None

    if job.primary:
        kind, wav = job.primary
        length = seconds(wav) or 0
        runners = {'vara': ['kestrel'], 'ardop': ['besra'], 'pactor': ['shrike']}[job.protocol]
        cmd = [str(pin.python), str(HERE / 'rehear_arms.py'), 'decode', str(wav), '--out', str(work),
               '--base', str(pin.stage), '--runners', *runners, '--grace', str(TIMEOUT_X * length + TIMEOUT_S)]
        cmd += ['--deep'] * (job.protocol == 'vara') + ['--p4'] * (job.protocol == 'pactor')
        cmd += selection_args
        if summary := attempt('monitor', cmd, 2 * TIMEOUT_X * length + 2 * TIMEOUT_S, 'monitor.json', 'residency'):
            events = [json.loads(line) for line in (work / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
            spans = summary['spans']
    if job.claimed:
        length = sum(seconds(p) or 0 for p in job.claimed)
        cmd = [str(pin.python), str(pin.tree / 'tools' / 'rehear'), *map(str, job.claimed),
               '--json', str(work / 'rehear.json'), '--ours']
        cmd += ['--log', str(job.transcript)] if job.transcript else []
        cmd += selection_args
        rehearing = attempt('rehear', cmd, TIMEOUT_X * length + TIMEOUT_S + 30 * len(job.windows), 'rehear.json', 'loaded')
    shutil.rmtree(work / 'tmp', ignore_errors=True)
    ran = summary is not None or rehearing is not None
    source = job.primary[0] if job.primary and summary is not None else None
    selected = None
    if job.selection is not None and job.primary:
        from hfmodem.station import sessionlog
        selected = sessionlog.select(job.selection, job.primary[1], root=pin.root)
    got, limits = heard(job.protocol, job.call, mycall, events, spans, rehearing,
                        source=source, bw_hz=job.bw_hz) if ran and (selected is None or selected.complete) else (None, Counter())
    out = {
        'arm': job.arm, 'protocol': job.protocol, 'call': job.call,
        'started_utc': entry['started_utc'], 'ended_utc': entry['ended_utc'],
        'rehearer': {
            'commit': book['commit'], 'tree': str(pin.tree), 'argv': argv,
            'env': {k: env[k] for k in ('PYTHONPATH', 'PYTHONPYCACHEPREFIX')}, 'cwd': str(pin.stage),
            'residency': {'set_sha256': book['set_sha256'], 'label': book['label'],
                          'monitor': summary and summary['residency']['set_sha256'],
                          'rehear': rehearing and _set_sha(rehearing['loaded'])},
            'gateway_list_sha256': book['lists_sha256'], 'reader': READER, 'reader_sha256': sha256(Path(__file__)),
            'transcript': job.transcript and {'path': recorded_path(job.transcript, pin.root),
                                              'sha256': sha256(job.transcript)},
            'run_utc': datetime.now(UTC).isoformat(timespec='seconds'), 'host': socket.gethostname(),
        },
        'audio': audio,
        'identity': identity(job, pin, mycall),
        'heard': got,
        'limits': [f'{what}: {n}' for what, n in sorted(limits.items())],
        'recorded': {'reached': entry['reached'], 'speed': entry['speed'], 'bw_hz': entry['bw_hz']},
        'failed': failed,
    }
    if selected is not None:
        out['context'] = dict(state='complete' if selected.complete else 'incomplete', reason=selected.reason)
    final = pin.out / job.arm
    out = json.loads(json.dumps(out).replace(json.dumps(str(work))[1:-1], json.dumps(str(final))[1:-1]))
    atomic(work / 'arm.json', out)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(work, final)
    (final / 'arm.json').unlink()
    atomic(pin.out / f'{job.arm}.json', out)
    return out


def _set_sha(modules: dict) -> str:
    return hashlib.sha256(json.dumps({n: m['sha256'] for n, m in modules.items()}, sort_keys=True).encode()).hexdigest()


def identity(job: Job, pin: Pinned, mycall: str) -> dict:
    """What the reading depends on besides the audio's bytes: whose station this is, and which
    recordings it was handed to decode. A callsign change can claim a file that was always there."""
    selected = {}
    if job.selection:
        from hfmodem.station import sessionlog
        selected = sessionlog.identity(job.selection)
    return {'bounds': list(job.bounds) if job.bounds else None,
            'clock_generation': job.clock_generation, 'context_id': job.context_id,
            'context_version': job.context_version, 'mycall': mycall.upper(),
            'claimed': sorted(recorded_path(p, pin.root) for p in job.claimed), **selected}


def done_already(pin: Pinned, job: Job, *, mycall: str) -> bool:
    """Whether this arm's JSON already stands for this audio; one from an older
    reading is read again from its raw output rather than decoded again."""
    if job.selection is not None and job.primary:
        from hfmodem.station import sessionlog
        if not sessionlog.select(job.selection, job.primary[1], root=pin.root).complete:
            return False
    path = pin.out / f'{job.arm}.json'
    try:
        old = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if old.get('failed') or not old.get('heard'):
        return False
    if ({(a['path'], a['sha256']) for a in old['audio']}
            != {(recorded_path(p, pin.root), sha256(p, selection=job.selection))
                for _, p in job.audio()}):
        return False
    # Records predating identity cannot establish which recordings were claimed
    # or which callsign the tools decoded against. Decode them again once.
    if old.get('identity') != identity(job, pin, mycall):
        return False
    if old['rehearer'].get('reader') != READER:
        raw = pin.out / job.arm
        load = lambda name: json.loads((raw / name).read_text(encoding='utf-8')) if (raw / name).is_file() else None
        summary, rehearing = load('monitor.json'), load('rehear.json')
        if rehearing:
            # A reader-4 decode was handed the whole captures folder, a shared one's
            # neighbours included; each window is replayed on its own, so this arm's are kept.
            rehearing['rehearings'] = [r for r in rehearing['rehearings']
                                       if _absolute(Path(r['recording']).with_suffix('.wav'))
                                       in {_absolute(p) for p in job.claimed}]
        events = [json.loads(line) for line in (raw / 'events.jsonl').read_text(encoding='utf-8').splitlines()] if summary else []
        source = job.primary[0] if job.primary and summary else None
        got, limits = heard(job.protocol, job.call, mycall, events, summary['spans'] if summary else [],
                            rehearing, source=source, bw_hz=job.bw_hz)
        old['heard'], old['limits'] = got, [f'{what}: {n}' for what, n in sorted(limits.items())]
        old['rehearer'] |= {'reader': READER, 'reader_sha256': sha256(Path(__file__)),
                            'reread_utc': datetime.now(UTC).isoformat(timespec='seconds')}
        atomic(path, old)
    return True


# -- living beside the station ----------------------------------------------------------

def flying(root: Path) -> str | None:
    """An arm holding the rig right now, or the process that means one is."""
    now = time.time()
    for prov in (root / 'working').glob('*/arms/*/provenance.json'):
        try:
            if now - prov.stat().st_mtime < FLYING_S:
                p = json.loads(prov.read_text(encoding='utf-8'))
                # legacy.py rewrites its conversions whenever it runs; they never flew.
                if 'ended_utc' not in p and not (p.get('source') or {}).get('converted_from'):
                    return prov.parent.name
        except (OSError, ValueError):
            continue
    return next((line.strip()[:120] for line in process_commands() if KEYERS.search(line)), None)


def process_commands() -> list[str]:
    if os.name != 'nt':
        return subprocess.run(['ps', '-axo', 'command='], capture_output=True, text=True,
                              check=True).stdout.splitlines()
    from hfmodem.core import winproc
    entries, _ = winproc.pinned()
    commands = []
    try:
        for entry in entries:
            try:
                commands.append(winproc.command_line(entry))
            except (OSError, winproc.ClaimError):
                continue
    finally:
        for entry in entries:
            entry.close()
    return commands


def free_ram() -> int:
    if sys.platform == 'win32':
        import ctypes as C
        from ctypes import wintypes as W
        class MemoryStatus(C.Structure):
            _fields_ = [('length', W.DWORD), ('load', W.DWORD),
                        ('total_physical', C.c_uint64), ('available_physical', C.c_uint64),
                        ('total_pagefile', C.c_uint64), ('available_pagefile', C.c_uint64),
                        ('total_virtual', C.c_uint64), ('available_virtual', C.c_uint64),
                        ('available_extended', C.c_uint64)]
        status = MemoryStatus(length=C.sizeof(MemoryStatus))
        query = C.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
        query.restype, query.argtypes = W.BOOL, [C.POINTER(MemoryStatus)]
        if not query(C.byref(status)):
            raise C.WinError(C.get_last_error())
        return status.available_physical
    if sys.platform.startswith('linux'):
        data = Path('/proc/meminfo').read_text(encoding='ascii')
        return int(re.search(r'^MemAvailable:\s+(\d+) kB$', data, re.M)[1]) * 1024
    said = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    page = int(re.search(r'page size of (\d+)', said)[1])
    pages = sum(int(m[1]) for m in re.finditer(r'Pages (?:free|inactive|speculative|purgeable):\s+(\d+)', said))
    return page * pages


class Gate:
    """Held shut while an arm flies, the disk runs low or memory is short."""

    def __init__(self, root: Path, out: Path, audit):
        self.root, self.out, self.audit = root, out, audit
        self.lock = threading.Lock()
        self.pauses, self.paused_s, self.admitted = [], 0.0, 0.0

    def wait(self):
        with self.lock:
            time.sleep(max(0.0, self.admitted + ADMIT_S - time.time()))
            since = None
            while why := self.blocked():
                if since is None:
                    since = time.time()
                    self.audit(f'paused: {why}')
                time.sleep(PAUSE_S)
            if since is not None:
                took = time.time() - since
                self.pauses.append(took)
                self.paused_s += took
                self.audit(f'resumed after {took:.0f} s')
            self.admitted = time.time()

    def blocked(self) -> str | None:
        if arm := flying(self.root):
            return f'{arm} is flying'
        if (free := shutil.disk_usage(self.out).free) < MIN_FREE_DISK:
            return f'{free >> 30} GB free on disk'
        if (ram := free_ram()) < MIN_FREE_RAM:
            return f'{ram >> 30} GB of memory free'
        return None


# -- the whole logbook -----------------------------------------------------------------

def logbook(root: Path, out: Path) -> tuple[dict, list[dict]]:
    """The logbook's entries, read by desk/face.py; its last good reading if it cannot be imported now."""
    snapshot = out / 'entries.json'
    try:
        sys.path.insert(0, str(HERE))
        sys.path.insert(0, str(HERE.parent))
        from desk import face
        book = face.Logbook(root)
        got = {'station': book.station, 'entries': book.entries()['entries']}
        atomic(snapshot, got)
    except Exception as exc:                                        # noqa: BLE001
        if not snapshot.is_file():
            raise
        print(f'desk/face.py could not be read ({type(exc).__name__}: {exc}); using {snapshot}', file=sys.stderr)
        got = json.loads(snapshot.read_text(encoding='utf-8'))
    return got['station'], got['entries']


def run(args) -> int:
    sys.path.insert(0, str(HERE))
    root = Path(args.root).resolve() if args.root else _default_root()
    pin = Pinned.at(root, args.commit)
    pin.out.mkdir(parents=True, exist_ok=True)
    log = (pin.out / 'run.log').open('a')

    def audit(line):
        stamped = f'{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {line}'
        print(stamped, file=log, flush=True)
        print(stamped, flush=True)

    book = pin.prepare()
    station, entries = logbook(root, pin.out)
    by_id = {e['id']: e for e in entries}
    wanted = [by_id[a] for a in args.arm] if args.arm else entries
    starts = capture_starts(entries, root)
    jobs = []
    for entry in wanted:
        try:
            job = plan(entry, root, starts, mycall=station['call'])
            if job:
                length = sum(seconds(p) or 0 for _, p in job.audio())
                jobs.append((length, job, entry))
        except (OSError, ValueError) as exc:
            audit(f"{entry['id']}: unavailable here (possibly moved): {exc}")
    jobs = [(job, entry) for _, job, entry in sorted(jobs, key=lambda item: -item[0])]
    audit(f'start: {len(jobs)} arms with audio of {len(wanted)} entries, decoders at {book["commit"]}, '
          f'{args.workers} workers, pid {os.getpid()}')
    gate = Gate(root, pin.out, audit)
    counts, cpu0, t0 = Counter(), os.times(), time.time()

    def one(job, entry):
        if done_already(pin, job, mycall=station['call']):
            return 'skipped'
        gate.wait()
        started = time.time()
        got = rehear_arm(job, pin, book, entry, mycall=station['call'], audit=audit)
        audit(f'{job.arm}: {time.time() - started:.0f} s, reached {(got["heard"] or {}).get("reached")}'
              + (f', FAILED {[f["tool"] for f in got["failed"]]}' if got['failed'] else ''))
        return 'failed' if got['failed'] else 'done'

    with ThreadPoolExecutor(max_workers=min(args.workers, WORKERS)) as pool:
        futures = {pool.submit(one, j, e): j for j, e in jobs}
        for f in as_completed(futures):
            try:
                counts[f.result()] += 1
            except Exception as exc:                                # noqa: BLE001
                counts['crashed'] += 1
                audit(f'{futures[f].arm}: crashed: {type(exc).__name__}: {exc}')
    cpu = os.times()
    pin.untouched()
    audit(f'end: {dict(counts)}; {time.time() - t0:.0f} s wall, '
          f'{cpu.children_user + cpu.children_system - cpu0.children_user - cpu0.children_system:.0f} s CPU in decoders, '
          f'{len(gate.pauses)} pause(s) for {gate.paused_s:.0f} s; tree still clean')
    return 0


def _default_root() -> Path:
    common = git(HERE, 'rev-parse', '--path-format=absolute', '--git-common-dir')
    return Path(common).parent


# -- the census --------------------------------------------------------------------------

def era(started: str) -> str:
    return 'before Aug 20' if started < '2026-08-20' else 'Aug 20 - Sep 14' if started < '2026-09-15' else 'Sep 15 on'


def census(args) -> int:
    """What this rehear found against the record, and what a better decoder would reach."""
    sys.path.insert(0, str(HERE))
    import link
    root = Path(args.root).resolve() if args.root else _default_root()
    pin = Pinned.at(root, args.commit)
    arms = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(pin.out.glob('*.json'))
            if p.name not in ('residency.json', 'entries.json')]
    rows, gained, limits, failed = Counter(), Counter(), Counter(), []
    for a in arms:
        key = (a['protocol'], era(a['started_utc']))
        rows[key] += 1
        if a['failed']:
            failed.append((a['arm'], a['failed']))
        got = a['heard']
        if not got:
            continue
        new = set(got['reached']) - set(a['recorded']['reached'])
        if new:
            gained[key + ('milestones',)] += 1
            for m in new:
                gained[(a['protocol'], 'milestone', m)] += 1
        was = (a['recorded']['speed'] or {}).get('decoded')
        if got['decoded_max'] and (link.rank(got['decoded_max']) or 0) > (link.rank(was) or 0):
            gained[key + ('decoded level',)] += 1
        for line in a['limits']:
            what, _, n = line.rpartition(': ')
            limits[(what, 'arms')] += 1
            limits[(what, 'count')] += int(n)
    print(json.dumps({
        'arms': {f'{p} / {e}': n for (p, e), n in sorted(rows.items())},
        'gained': {' / '.join(k): n for k, n in sorted(gained.items())},
        'limits': {w: {'arms': limits[(w, 'arms')], 'count': limits[(w, 'count')]}
                   for w in sorted({w for w, _ in limits})},
        'failed': failed,
    }, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='rehear_arms.py', description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='rehear every arm with audio, or those named')
    p.add_argument('--root', help='the checkout whose working/ holds the arms')
    p.add_argument('--commit', default=COMMIT)
    p.add_argument('--arm', nargs='*', default=[])
    p.add_argument('--workers', type=int, default=WORKERS)
    p.set_defaults(func=run)
    p = sub.add_parser('census', help="summarise one commit's rehear against the record")
    p.add_argument('--root')
    p.add_argument('--commit', default=COMMIT)
    p.set_defaults(func=census)
    p = sub.add_parser('decode', help=argparse.SUPPRESS)
    p.add_argument('wav', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--runners', nargs='+', required=True)
    p.add_argument('--deep', action='store_true')
    p.add_argument('--p4', action='store_true')
    p.add_argument('--selection', type=Path)
    p.add_argument('--grace', type=float, default=600.0)
    p.set_defaults(func=decode)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
