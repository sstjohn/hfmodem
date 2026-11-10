# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The station's own log of a session: the three things a replay cannot read off audio.

A recording holds what arrived at the codec. It does not hold the session id this
end was party to, it does not hold when the receiver was muted, and it does not
hold when memory ARQ was reset — and without those a replay of one of *our*
sessions is not a replay of it.

**The session id**, because `Demodulator` grades a bodyless control on eight bits
of corroboration it can only get from the ARQ layer: a frame bearing an id we are
in session with is admitted at crisp 8.0 where an unsolicited one faces 9.5
(`_crisp_floor`), and `_scan_header`'s fallback — the path that mints every
header-only sighting — admits nothing at all below 9.5 without the id. Replayed
with no hint, `20260826T133925Z` reproduces 28 of the 51 frames that session
reported: no ConAck, no END, eight of the DATAACKs and three of the BREAKs.

**The mute**, because `RadioLink._transmit` sets `muted` one statement after the
`TX <frame> <dur>s` line is written (`arq/modem.py:233`) and clears it one
statement after `PTT OFF -> …` (`core/ptt.py:572`), so those two lines are the
interval, to within the length of a Python statement. Grown off the capture's own
silence brackets instead, the same session withholds 64.5 s against the 50.6 s
those edges give, misses five keyings outright, and loses three more frames.

**`rx_epoch`**, the memory-ARQ reset, because it moves where `ArqSession` enters
IRS or DISC and nothing about that reaches the audio. The runner prints each state
as the session announces it, so the transitions are on the record — see `EPOCH`.

The log's clock is the wall clock of the machine that ran the session and the
recording's zero is its own first sample; the sidecar carries that only to the
whole second, which is two slack widths of error. So the offset between the two is
measured against the capture's own key-downs rather than assumed — see `align`. Two
slack widths is nowhere near a whole transmission, though, so where the key-downs
vote to a tie the clocks still say which of the tied offsets is the station's own.
"""
from __future__ import annotations

import bisect
import datetime as dt
from copy import deepcopy
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterable, Sequence

from hfmodem.besra import crc
from hfmodem.station.narrate import TX_FRAME

#: The logging module's own prefix, with the milliseconds `narrate.STAMPED` drops.
#: A mute edge is worth 0.1 s — one whole block of the decoder's input — so the
#: three digits it discards are three digits this reads.
STAMPED = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) \w+ (.*)$")

#: `arq/modem.py:278` writes `ok=` as `UNVERIFIED` wherever there is no body to
#: check, so `narrate.RX_FRAME`'s `ok=(True|False)` reads none of the bodyless
#: controls — twelve of this recording's 51 frames, which is the ConAck that opened
#: the session, the END that closed it, and every DATAACK and BREAK between them.
RX_FRAME = re.compile(r"^RX (\S+) sess=(0x[0-9a-f]+) ok=(True|False|UNVERIFIED)"
                      r"( HEADER-ONLY)?")

#: Who this station called. `ArqSession.connect` sets `expected_session` from the
#: two callsigns before the first ConReq leaves, so a call nobody answered still had
#: a hint on the air — and that is the case the lower floor was written for: KE8LVA
#: answered six of eleven ConReqs at crisp 8.3-9.4 and the strict floor rejected
#: every one. A replay of such a call has no RX line to read an id off, and this is
#: the only other place it is written down.
CALLING = re.compile(r"^mail: ([A-Z0-9-]+) -> ([A-Z0-9-]+) ")

#: The VARA bandwidth an arm ran, from `kestrel_connect.py:1749`'s own unstamped
#: line — the one place a session log says which alphabet its handshake used.
ORIGINATING = re.compile(r"^originating: \S+ -> \S+ \(BW(\d+)\)")

#: The unstamped line the runner prints for each state the session announces
#: through `Observer.newstate`. `ArqSession._set_state` moves `rx_epoch` on exactly
#: two of them, IRS and DISC, and this is the only record of when.
EPOCH = re.compile(r"^\s*\[(IRS|DISC)\]\s*$")

#: Where the session named its recording: `run_server.py:429` as the capture opens,
#: `kestrel_connect.py:1340` likewise, and `radio.py` again in the closing summary.
#: Matching on the WAV's own name rather than on a path keeps a moved corpus
#: readable.
_NAMES_WAV = ("recording receive audio to", "session stream:")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _roots() -> list[Path]:
    """Where session logs are, absent a `--log`. The runner writes one per attempt
    to the operator's home directory and nothing has ever moved them into the tree,
    so home is the default and `HFMODEM_SESSION_LOGS` is for a machine where they
    live somewhere else."""
    named = os.environ.get("HFMODEM_SESSION_LOGS", "")
    return [Path(p).expanduser() for p in named.split(os.pathsep) if p] or [Path.home()]


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame the live receive path reported, on the log's clock."""
    at: float
    name: str
    session: int
    ok: bool
    header_only: bool


@dataclass(slots=True)
class SessionLog:
    """One session's log, with every time in seconds from its first stamped line."""
    path: Path
    session: int | None
    #: The VARA bandwidth this arm originated at, from its own `originating:` line —
    #: None where the log holds no such line, which is every non-VARA arm.
    bw: str | None = None
    #: `[TX line, PTT OFF line]` for each transmission: the mute, as the station set it.
    keyed: list[tuple[float, float]] = field(default_factory=list)
    #: Every `RX …` line, which is what the live path reported and the only frame
    #: list a replay of one of our sessions can be graded against.
    frames: list[Frame] = field(default_factory=list)
    #: When the session was in progress at all — `expected_session` is None outside it.
    session_span: tuple[float, float] = (0.0, float("inf"))
    #: Times `rx_epoch` moved, in order.
    epochs: list[float] = field(default_factory=list)
    #: The wall clock this log's second zero is. The only thing that relates it to a
    #: recording, whose own zero is stamped in UTC by the recorder that opened it.
    zero: dt.datetime | None = None
    selection_offset: float | None = None
    selection: dict | None = None
    context: dict | None = None
    grading: tuple[Frame, ...] | None = None

    def predicted(self, started_utc: dt.datetime) -> float | None:
        """Where the two clocks put a capture's first sample, in this log's seconds.

        This log stamps the local wall time of the machine that ran the session and a
        capture's sidecar stamps UTC, and neither writes down the zone. Neither has
        to: every civil offset is a whole number of quarter hours, and the recorder
        opens within seconds of the log's first stamped line — 0.35 to 1.30 s across
        fifty-one of this station's arms, 14.08 s at the worst — so the zone divides
        out.

        A floor and not a reading. `CaptureRecorder` stamps `time.time()` truncated
        to the second and the first block reaches it after that, so the truth is
        above this by the truncation and the card's own start.
        """
        if self.selection_offset is not None:
            return self.selection_offset
        if self.zero is None:
            return None
        gap = (started_utc - self.zero).total_seconds()
        return gap - _QUARTER_HOUR * round(gap / _QUARTER_HOUR)


def read(path: Path, *, text=None, selection=None) -> SessionLog:
    """Parse one session log. Times are seconds from its first stamped line."""
    if selection is not None:
        root = source_root(selection, path, transcript=True)
        ref = next(iter(selection.get("lanes", {}).values()), {})
        result = select(selection, root / ref.get("source", ""), root=root)
        return result.log or SessionLog(path, None, selection=selection, context={})
    stamped: list[tuple[float, str]] = []
    epochs: list[tuple[float, str]] = []
    called: tuple[str, str] | None = None
    bw: str | None = None
    base: dt.datetime | None = None
    for raw in (path.read_text(errors="replace") if text is None else text).splitlines():
        line = _ANSI.sub("", raw)
        m = STAMPED.match(line)
        if m is None:
            # A state marker carries no stamp of its own. `_set_state` runs inside
            # the handling of the frame the PRECEDING stamped line reports — the
            # `[IRS]` transitions of 20260826T133925Z each follow their own
            # `PTT OFF` by a statement and lead the next `RX` by four seconds — so
            # the line above it is the tight bound and the line below is not.
            if (e := EPOCH.match(line)) and stamped:
                epochs.append((stamped[-1][0], e.group(1)))
            elif (c := CALLING.match(line)) and called is None:
                called = (c.group(1), c.group(2))
            elif (o := ORIGINATING.match(line)) and bw is None:
                bw = o.group(1)
            continue
        when = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
        base = base or when
        stamped.append(((when - base).total_seconds(), m.group(2)))

    keyed: list[tuple[float, float]] = []
    frames: list[Frame] = []
    opened: float | None = None
    for at, text in stamped:
        if TX_FRAME.match(text):
            opened = at if opened is None else opened
        elif text.startswith("PTT OFF"):
            if opened is not None:
                keyed.append((opened, at))
                opened = None
        elif (m := RX_FRAME.match(text)) and "funnel" not in text:
            frames.append(Frame(at, m.group(1), int(m.group(2), 16),
                                m.group(3) != "False", m.group(4) is not None))

    ids = Counter(f.session for f in frames if f.session != 0xFF)
    session = (ids.most_common(1)[0][0] if ids
               else crc.session_id(*called) if called else None)
    down = next((at for at, state in epochs if state == "DISC"), float("inf"))
    return SessionLog(path=path, session=session, bw=bw, keyed=keyed, frames=frames,
                      session_span=(keyed[0][0] if keyed else 0.0, down),
                      epochs=[at for at, _ in epochs], zero=base)


@dataclass(frozen=True)
class SelectedSpan:
    bounds: tuple[int, int]
    state: dict
    changes: tuple[dict, ...] = ()
    reset_decoder: bool = True


@dataclass
class Selection:
    bounds: tuple[int, int] | None = None
    state: dict = field(default_factory=lambda: {'expected_session': None})
    spans: tuple[SelectedSpan, ...] = ()
    gaps: tuple[tuple[int, int], ...] = ()
    tx: tuple[dict, ...] = ()
    participation: tuple[int, int] | None = None
    clock_generation: int | None = None
    context_id: str | None = None
    context_version: int | None = None
    log: SessionLog | None = None
    grading: tuple[Frame, ...] | None = None
    complete: bool = False
    reason: str | None = None
    legacy: bool = False
    protocol: str | None = None
    calls: dict = field(default_factory=dict)
    local_role: str | None = None
    rate: int | None = None
    lane: str | None = None
    anchor: dict = field(default_factory=dict)
    grid: tuple[dict, ...] = ()
    probe: dict | None = None


def identity(recording, lane=None):
    """Cache identity from declared bounds, including unsupported context versions."""
    recording = (recording or {}).get('recording', recording or {})
    refs = recording.get('lanes', {})
    ref = refs.get('rx', next(iter(refs.values()), {}))
    ctx = recording.get('context') or {}
    def bounds(ref):
        selected, actual = ref.get('selection', ref.get('actual')), ref.get('actual')
        if selected is None:
            return None
        return [max(selected[0], actual[0]), min(selected[1], actual[1])] if actual else list(selected)
    return dict(bounds=bounds(ref),
                clock_generation=ref.get('clock_generation'),
                context_id=ctx.get('id'), context_version=ctx.get('version'),
                sources={name: deepcopy({**{k: r.get(k) for k in
                    ('source', 'source_id', 'index_id', 'clock', 'clock_generation', 'rate',
                     'actual', 'participation', 'gaps')},
                    'bounds': bounds(r),
                    'mapping': ctx.get('lanes', {}).get(name),
                    'state': ctx.get('lane_states', {}).get(name)}) for name, r in sorted(refs.items())})


def source_root(recording, path, *, transcript=False):
    recording = recording.get('recording', recording)
    if recording.get('run_directory'):
        return Path(recording['run_directory'])
    links = ([recording.get('context', {}).get('transcript', {})] if transcript
             else recording.get('lanes', {}).values())
    for ref in links:
        relative = Path(ref.get('path' if transcript else 'source', ''))
        if not relative.is_absolute() and relative.parts:
            candidate = Path(path).absolute()
            if candidate.with_suffix('.wav').parts[-len(relative.parts):] == relative.with_suffix('.wav').parts:
                for _ in relative.parts:
                    candidate = candidate.parent
                return candidate
    return Path(path).parent


def recording_for(path):
    """Read policy metadata attached to this exact source; no sibling discovery."""
    try:
        value = json.loads(Path(path).with_suffix('.json').read_text()).get('recording')
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def select(recording, path, *, root=None, start=None, stop=None, lane=None):
    """Validate and evaluate one contact on its absolute source sample clock.

    Policy records require bound context. Only legacy records discover logs.
    Every returned span starts with independent decoder state after a gap.
    """
    path = Path(path)
    recording = recording.get('recording', recording)
    if not recording.get('policy') and 'context' not in recording and not recording.get('lanes'):
        return Selection(legacy=True, complete=True, log=find(path))
    result = Selection()
    try:
        root = Path(root) if root is not None else source_root(recording, path)
        refs = recording['lanes']
        if lane is None:
            candidates = [name for name, ref in refs.items()
                          if (root / ref['source']).absolute().with_suffix('.wav') == path.absolute().with_suffix('.wav')]
            if len(candidates) != 1:
                raise ValueError('Context has no unique source lane')
            lane = candidates[0]
        ref = refs[lane]
        if (root / ref['source']).absolute().with_suffix('.wav') != path.absolute().with_suffix('.wav'):
            raise ValueError('Context source path mismatch')
        try:
            side = json.loads(path.with_suffix('.json').read_text())
        except FileNotFoundError:
            side = {}
        for key in ('source_id', 'index_id', 'clock', 'clock_generation'):
            if key in side and side[key] != ref[key]:
                raise ValueError(f'Audio {key} mismatch')
        lo, hi = ref.get('selection', ref['actual'])
        lo, hi = max(lo, ref['actual'][0]), min(hi, ref['actual'][1])
        lo = max(lo, start) if start is not None else lo
        hi = min(hi, stop) if stop is not None else hi
        if lo >= hi:
            raise ValueError('Empty contact selection')
        rate, generation = ref['rate'], ref['clock_generation']
        result.bounds, result.rate, result.lane = (lo, hi), rate, lane
        available = [(lo, hi)]
        for a, b in ref.get('gaps', []):
            available = [(c,d) for x,y in available for c,d in ((x,min(y,a)),(max(x,b),y)) if c<d]
        result.spans = tuple(SelectedSpan(bounds, {'expected_session': None}) for bounds in available)
        ctx = recording['context']
        result.protocol, result.calls, result.local_role = ctx.get('protocol'), deepcopy(ctx.get('calls', {})), ctx.get('local_role')
        schedule = (ctx.get('lane_states', {}).get(lane, {}) if lane != 'rx' and 'rx' in refs else ctx)
        initial = schedule.get('initial', {})
        changes = sorted((deepcopy(change) for change in schedule.get('changes', [])
                          if change.get('clock_generation') == generation), key=lambda c: c['sample'])
        def state_at(at):
            state = deepcopy(initial)
            for change in changes:
                if change['sample'] > at:
                    break
                state.update({k: v for k, v in change.items() if k not in ('sample', 'clock_generation')})
            return state
        states = [state_at(a) for a, b in available]
        missing = []
        if not ctx.get('transcript'):
            missing.append('transcript range')
        if any(s.get('arq_epoch') is None for s in states):
            missing.append('ARQ epoch')
        if 'tx' not in ctx:
            missing.append('TX intervals')
        if any(not isinstance(s.get('muted'), bool) for s in states):
            missing.append('mute state')
        if str(ctx.get('protocol', '')).upper().startswith('PACTOR'):
            if not ctx.get('probe'):
                missing.append('probe binding')
            if not any(g.get('clock_generation') == generation and g.get('sample', hi) < hi
                       for g in ctx.get('grid', [])):
                missing.append('grid binding')
        if missing:
            raise ValueError('Missing replay context: ' + ', '.join(missing))
        if ctx.get('state') == 'incomplete':
            raise ValueError(ctx.get('reason', 'Context incomplete'))
        if ctx['version'] != 1 or ctx['contact_id'] != recording['id']:
            raise ValueError('Context version or contact identity mismatch')
        mapping = ctx['lanes'][lane]
        for key in ('source_id', 'index_id', 'clock', 'clock_generation', 'rate'):
            if ref.get(key) is None or ref[key] != mapping.get(key):
                raise ValueError(f'Context lane {key} mismatch')
        result.clock_generation = generation
        result.context_id, result.context_version = ctx['id'], ctx['version']
        result.participation = tuple(ref.get('participation', (lo, hi)))
        result.anchor = deepcopy(mapping['anchor'])
        if (not math.isfinite(result.anchor['uncertainty_s']) or result.anchor['uncertainty_s'] < 0
                or not math.isfinite(result.anchor['transcript_s'])
                or not isinstance(result.anchor['sample'], int)):
            raise ValueError('Invalid source clock anchor')
        def bound_bytes(link):
            source = root / link['path'] if 'path' in link else path.with_suffix('.context')
            a, b = link['range']
            if source.is_absolute() and not source.resolve().is_relative_to(root.resolve()):
                raise ValueError('Context path is outside its root')
            if a < 0 or b < a or not link.get('source_id'):
                raise ValueError('Invalid context source range')
            if 'text' in link:
                data = link['text'].encode()[a:b]
            else:
                with source.open('rb') as stream:
                    stream.seek(a)
                    data = stream.read(b - a)
            if len(data) != b - a or hashlib.sha256(data).hexdigest() != link['sha256']:
                raise ValueError('Context range digest mismatch')
            return source, data
        transcript, data = bound_bytes(ctx['transcript'])
        parsed = read(transcript, text=data.decode(errors='replace'))
        if parsed.zero is None:
            raise ValueError('Missing replay context: real transcript range')
        if ctx.get('probe'):
            bound_bytes(ctx['probe'])
        gaps = []
        for a, b in sorted(ref.get('gaps', [])):
            a, b = max(lo, a), min(hi, b)
            if a < b:
                if gaps and a <= gaps[-1][1]:
                    gaps[-1] = (gaps[-1][0], max(b, gaps[-1][1]))
                else:
                    gaps.append((a, b))
        spans, cursor = [], lo
        for a, b in [*gaps, (hi, hi)]:
            if cursor < a:
                spans.append(SelectedSpan((cursor, a), state_at(cursor),
                    tuple(c for c in changes if cursor < c['sample'] < a)))
            cursor = b
        txs = []
        for tx in ctx.get('tx', []):
            if tx.get('source_id') != ref['source_id'] or tx.get('clock_generation') != generation:
                continue
            a, b = tx.get('ptt_start_sample'), tx.get('ptt_end_sample')
            if a is None or b is None:
                continue
            for span in spans:
                first, last = max(a, span.bounds[0]), min(b, span.bounds[1])
                if first < last:
                    row = deepcopy(tx)
                    row['original_bounds'] = {k: v for k, v in tx.items() if k.endswith('_sample')}
                    for k in row['original_bounds']:
                        row[k] = min(last, max(first, row[k]))
                    row.update(t0=(first - lo) / rate, t1=(last - lo) / rate)
                    txs.append(row)
        log = parsed
        anchor = result.anchor
        offset = anchor['transcript_s'] + (lo - anchor['sample']) / rate
        base_s = ctx['transcript']['first_s']
        if not math.isfinite(base_s):
            raise ValueError('Invalid transcript range clock origin')
        frames = [Frame(f.at + base_s, f.name, f.session, f.ok, f.header_only) for f in log.frames]
        def available(at):
            sample = anchor['sample'] + (at - anchor['transcript_s']) * rate
            return any(a <= sample < b for a, b in (s.bounds for s in spans))
        log.frames = [f for f in frames if available(f.at)]
        log.selection_offset = offset
        selected_state = state_at(lo)
        log.session = selected_state.get('expected_session')
        log.bw = str(selected_state['bandwidth_hz']) if selected_state.get('bandwidth_hz') is not None else None
        log.keyed = [(offset + t['t0'], offset + t['t1']) for t in txs]
        log.epochs = [anchor['transcript_s'] + (c['sample'] - anchor['sample']) / rate
                      for c in changes if 'arq_epoch' in c and lo <= c['sample'] < hi]
        log.session_span = (offset, offset + (hi - lo) / rate)
        result.state, result.spans, result.gaps = selected_state, tuple(spans), tuple(gaps)
        def grade(frame):
            sample = anchor['sample'] + (frame.at - anchor['transcript_s']) * rate
            a, b = result.participation
            return (a is not None and b is not None and a <= sample < b
                    and state_at(sample).get('expected_session') == frame.session)
        log.selection, log.context = dict(recording, lane=lane), ctx
        result.tx, result.log, result.grading = tuple(txs), log, tuple(f for f in log.frames if grade(f))
        log.grading = result.grading
        result.protocol, result.calls, result.local_role = ctx.get('protocol'), deepcopy(ctx.get('calls', {})), ctx.get('local_role')
        result.grid = tuple(deepcopy(ctx.get('grid', [])))
        result.probe = deepcopy(ctx.get('probe'))
        result.complete = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.reason = str(exc)
        result.state, result.tx, result.grading = {'expected_session': None}, (), None
        result.log = None
    return result


def find(wav: Path, extra: Iterable[Path] = ()) -> SessionLog | None:
    """The session log that names this recording, or None.

    A path given on the command line may be the log itself; a directory is
    searched, as are `_roots`, for a `*.log` naming this WAV.

    BY THE DIRECTORY WHERE THE ARM WROTE NO NAME, which is every shrike arm: a
    session writes 140-odd `rx_NN`/`hold_NN` windows and names none of them,
    only the `stream.wav` they are all indexed on. Matching the wav's own name
    alone therefore handed the shrike adapter `None` for every capture ever
    passed to it with `--log`, and it is the adapter that most needs one — the
    grid's answer instant is the whole of what a capture cannot hold.

    An arm older than `stream.wav` names only the directory, with `captures:`.

    A log names a recording as the WAV it was while it was being written, so a
    stored one is looked for under that name.
    """
    named_as = wav.with_suffix(".wav").name
    for path in _candidates(wav, extra):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        named = [line for line in text.splitlines()
                 if any(f"{key} " in line for key in _NAMES_WAV)]
        stream = re.compile(rf"[\\/]{re.escape(wav.parent.name)}[\\/]{re.escape(_STREAM)}")
        if (any(named_as in line or stream.search(line) for line in named)
                # A log names paths in its own host's form; this reads either.
                or any(PureWindowsPath(d).name == wav.parent.name
                       for d in _CAPTURES.findall(text))):
            return read(path)
    return None


#: What a shrike arm calls the one recording it does name, and which sits in the
#: same directory as every window it wrote. Matching the directory through this
#: rather than on its bare name is what keeps `logs/onair/` — a besra directory
#: whose name really is a word — from claiming somebody else's log.
_STREAM = "stream.wav"

#: How a shrike arm named its capture directory before it wrote a `stream.wav`
#: there. Most of August's PACTOR transcripts print only `  captures: DIR`, and
#: every window of theirs replayed deaf until this was read. The whole last
#: component is matched, for the reason `_STREAM` is.
_CAPTURES = re.compile(r"^\s*captures: (\S+)\s*$", re.M)


def _candidates(wav: Path, extra: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in [*extra, *_roots()]:
        if root.is_file():
            out.append(root)
        elif root.is_dir():
            out += sorted(root.glob("*.log"))
    return out


class Ambiguous(ValueError):
    """The capture cannot say which of the alignments on offer is the station's own.

    Raised rather than returned, and rather than picking. Every offset that would
    pair a log key-down with a silence bracket in the capture gets the same vote,
    and where the keying is regular the bracket a transmission *ends* on votes as
    often as the one it starts on — the two combs are one transmission apart and
    the tie went to whichever came first in a sorted list. On the 2026-08-29 15:15
    W4UC call that put the offset at -4.058 s where the truth is +1.681, and an
    inverted offset withholds the gaps and passes our own six ConReqs off as the
    gateway's. An instrument that inverts its own sign in silence is worse than one
    that declines.

    Reached only where the recorder's own stamp cannot separate the offsets either,
    which for the key-up comb it always can — see `align`'s `predicted`.
    """


@dataclass(frozen=True, slots=True)
class Alignment:
    """Where the log's clock sits against the recording's first sample."""
    offset: float
    anchors: int
    residual: float
    #: How many of the station's own keyings the capture is quieter inside than
    #: either side of, at this offset — see `align`.
    corroborated: int
    keyings: int

    def __str__(self) -> str:
        return (f"aligned to the capture's own key-downs, {self.anchors} anchors, "
                f"median {self.residual * 1000:.0f} ms, "
                f"{self.corroborated} of {self.keyings} keyings quieter in the capture")


#: A key-down the log and the capture agree on within this is the same key-down.
#: The pairing is a vote over every offset the two lists admit, so this only has to
#: be tighter than the gap between neighbouring transmissions (2.5 s at worst here)
#: and looser than the true residual, which on `20260826T133925Z` runs to 31 ms.
_ANCHOR_SLACK = 0.15

#: Every civil UTC offset is a whole number of these, which is what lets a log's
#: local stamps be subtracted from a capture's UTC one without knowing the zone.
_QUARTER_HOUR = 900.0

#: How far above `SessionLog.predicted` the capture's first sample can be. The
#: prediction is a floor — a truncated second plus the card's own start — and across
#: forty-three aligned besra captures the truth ran 0.40 to 1.34 s above it. Wide enough
#: for that, and narrower than the 1.87 s that separates the closest pair of offsets
#: the key-down vote has ever returned.
_PREDICTED_WINDOW = 1.75


def align(log: SessionLog, keydowns: Sequence[float],
          quieter: Callable[[float], int],
          predicted: float | None = None) -> Alignment | None:
    """Seconds to subtract from a log time to put it on the recording's timebase.

    The sidecar's `started_utc` is a whole second and the log's stamps are local
    wall time, so neither end of that subtraction is worth 0.1 s — and 0.1 s is one
    block of the decoder's input. What both files do hold to the millisecond is
    when this station keyed: the log writes `TX` as `muted` goes up, and the
    capture goes to digital silence in the same instant wherever the rig mutes its
    own receiver. So every offset that would pair a log key-down with a capture
    key-down gets a vote, and the offset with the most wins.

    The vote alone does not settle it. A transmission is bracketed by two runs of
    silence, not one, and `keydowns` is every bracket the capture holds — so on an
    arm whose keyings are regular the comb built on the key-*ups* pairs exactly as
    often as the one built on the key-downs. Both reach the top of the poll and the
    difference between them is one transmission. `quieter` breaks that: given a
    candidate offset it returns how many of the station's own keyings the capture's
    level agrees with, the receiver being muted while we transmit and the band
    audible either side of it. A winner has to be alone; `Ambiguous` where it is not.

    That leaves the shape this station produces most: six identical ConReqs on a
    fixed cadence and nothing back. Both combs score, and a capture whose rig mutes
    shallowly scores them alike — eight of the thirty-four calls of 2026-08-29 and
    08-30 tie that way and were refused. `predicted` settles those, because the two
    clocks are wrong by a truncated second where the tie is a whole transmission
    apart: a candidate outside `_PREDICTED_WINDOW` of it is not the offset the
    recorder and the log describe, whatever the vote made of it.

    None where too few key-downs agree to be a coincidence — a recording whose rig
    does not mute the capture at all, or a log that belongs to some other session —
    and None where the vote's winners all lie outside the window, which is the same
    thing said by the clocks: this capture's silences are not our keyings.
    """
    if not log.keyed or not keydowns:
        return None
    theirs = [a for a, _ in log.keyed]
    # Every offset the two lists admit, and then the densest clusters of them. Two
    # capture key-downs never sit inside one window — the shortest gap between
    # transmissions in these sessions is 2.5 s — so a window's size is the number
    # of key-downs the offset it stands for explains.
    offsets = sorted(t - k for t in theirs for k in keydowns)

    def past(i: int) -> int:
        return bisect.bisect_right(offsets, offsets[i] + 2 * _ANCHOR_SLACK)

    sizes = [past(i) - i for i in range(len(offsets))]
    best = max(sizes)
    if best * 2 < len(theirs):
        return None
    tops: list[float] = []
    for i, size in enumerate(sizes):
        if size == best:
            at = _median(offsets[i:past(i)])
            if not tops or at - tops[-1] > 2 * _ANCHOR_SLACK:
                tops.append(at)
    if predicted is not None:
        tops = [at for at in tops if 0.0 <= at - predicted <= _PREDICTED_WINDOW]
        if not tops:
            return None
    offset = tops[0] if len(tops) == 1 else _settle(tops, quieter, log.path.name)
    nearest = [min(abs(t - offset - k) for k in keydowns) for t in theirs]
    kept = [r for r in nearest if r < _ANCHOR_SLACK]
    return Alignment(offset, len(kept), _median(kept), quieter(offset), len(theirs))


def _settle(tops: Sequence[float], quieter: Callable[[float], int], name: str) -> float:
    """Which of the equally-voted offsets the capture's own level corroborates."""
    ranked = sorted(((quieter(at), at) for at in tops), reverse=True)
    if ranked[0][0] == ranked[1][0]:
        raise Ambiguous(
            f"{name}: {len(tops)} offsets pair the log's key-downs with this capture's "
            f"silence equally well and its level tells none of them apart — "
            + ", ".join(f"{at:+.3f} s corroborated by {n} keyings" for n, at in ranked)
            + ". Refusing rather than picking: what ties here is the key-down comb "
            "against the key-up comb, one transmission apart, and an alignment off "
            "by one transmission passes our own audio off as the peer's")
    return ranked[0][1]


def _median(xs: Sequence[float]) -> float:
    ordered = sorted(xs)
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2
            else 0.5 * (ordered[mid - 1] + ordered[mid]))


#: How far a frame's report can trail the sample its header sits on. `RollingDecoder`
#: reports a frame when the window holding it is decoded, so the lag is the rest of
#: the frame plus the pump's own cadence: on `20260826T133925Z` it runs 0.28-0.35 s
#: for a bodyless control and 3.88-4.05 s for a 4PSK.200.100 data frame, over all 45
#: frames that session decoded. Wider than this pairs a frame with a repeat of
#: itself — 4PSK.200.100 repeats on a ~5 s grid here — and the pairing slides by one
#: arrival while still counting right, which makes the lags it reports a fiction.
_REPORT_LAG = 8.0


def reproduced(logged: Sequence[Frame], got: Sequence[tuple[float, str]],
               offset: float = 0.0) -> list[tuple[int, int]]:
    """Which of the frames the session logged this replay found, pairwise.

    A frame list is a sequence, not a set: the same DATAACK goes out a dozen times
    and only the order tells one from the next. So the frames both ends decoded are
    matched as a longest common subsequence over names, constrained to pairs whose
    times can be the same arrival.

    A header-only sighting is graded as a type and not as a time. It is a type minted
    by a window that found nothing better, and where its header sits is that window's
    guess: of the six on `20260826T133925Z`, one is logged 10.3 s after the sample the
    replay pins it to — behind two DATAACKs the live path reported in between, so out
    of sequence as well — and one 7.5 s before it. Neither end knows a sighting's
    position well enough to hold the other to it, so they are paired by name within
    `_SIGHTING_SLACK` either way and in any order, once the decodes have taken
    theirs.
    """
    def fits(f: Frame, at: float, name: str) -> bool:
        return f.name == name and 0.0 <= f.at - offset - at <= _REPORT_LAG

    def sighted(f: Frame, at: float, name: str) -> bool:
        return f.name == name and abs(f.at - offset - at) <= _SIGHTING_SLACK

    firm = [i for i, f in enumerate(logged) if not f.header_only]
    out = [(firm[a], b) for a, b in
           _subsequence([logged[i] for i in firm], got, fits)]
    taken = {b for _, b in out}
    for i, f in enumerate(logged):
        if not f.header_only:
            continue
        j = next((j for j, (at, name) in enumerate(got)
                  if j not in taken and sighted(f, at, name)), None)
        if j is not None:
            out.append((i, j))
            taken.add(j)
    return sorted(out)


#: How far a header-only sighting may sit from the live path's report of it, either
#: way. Measured on `20260826T133925Z`: 10.3 s late and 7.5 s early are the extremes
#: over six sightings, against 4.05 s for the worst of the 45 decodes.
_SIGHTING_SLACK = 15.0


def _subsequence(logged, got, fits) -> list[tuple[int, int]]:
    """Longest common subsequence of the two frame lists under ``fits``."""
    ok = [[fits(f, at, name) for at, name in got] for f in logged]
    n, m = len(logged), len(got)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best[i][j] = max(best[i + 1][j], best[i][j + 1],
                             best[i + 1][j + 1] + 1 if ok[i][j] else 0)
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if ok[i][j] and best[i][j] == best[i + 1][j + 1] + 1:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


# -- a shrike ARQ arm, which is a different log ------------------------------

#: The line `onair._SessionRx._on` prints for every control signal its readers
#: delivered, tagged `RX` while calling and `HOLD RX` once the link is up.
#:
#: ITS TIME IS THE SESSION'S STREAM CLOCK, which is the clock a capture sidecar's
#: `end_stream_sample` counts, so a window and this line can be laid beside each
#: other to the millisecond. That was not always so and `rehear.shrike` said as
#: much for as long: `control_signal` used to time its event inside the window it
#: was handed, and now re-times it by `seg_start` before printing
#: (`tests.shrike.test_session_clock`). Two decimals is all it prints, so a match
#: has to be looser than 5 ms and is; one codeword is 120.
_WORD = re.compile(r"^(?:HOLD )?RX\s+(\d+\.\d+)\s+(cs|unassigned)\s+(.+)$")

#: The connect-phase reader's own line, from the codeword search that runs before
#: a grid exists. It reports an offset into the answer band and no instant — the
#: sample is on the `[grid]` line that follows it, which is why these are paired
#: rather than read alone.
_CANDIDATE = re.compile(r"^RX candidate CS(\d) at d = \d+ ms")

#: Every instant the grid put a control signal at, in stream samples. `CS due` is
#: where the next one is predicted and the other three are where one was measured;
#: a slot boundary is neither, and neither is the grid's own origin.
_AIMED = re.compile(r"(?:CS due @ |turnaround CANDIDATE @ sample "
                    r"|TURNAROUND ACQUIRED @ sample |ms @ sample )(\d+)")

_CS_TEXT = re.compile(r"^CS(\d)/")


@dataclass(frozen=True, slots=True, order=True)
class Word:
    """One control signal a shrike arm's readers delivered, on the stream clock."""
    at: float
    what: str
    text: str


@dataclass(frozen=True, slots=True)
class Grid:
    """Where a shrike arm's own readers were aimed, and what they came back with.

    Neither is in the audio. `aimed` is a property of when WE keyed — the transmit
    grid's answer instant, which is what `_SessionRx.control_signal` is called
    with — and `words` is the station's record of what it read there. A replay
    given a capture and no log has neither, which is most of the difference
    between what such a replay hears and what the session did.
    """
    aimed: tuple[int, ...]
    words: tuple[Word, ...]
    predecessor: dict | None = None
    clock_generation: int | None = None

    def within(self, lo: int, hi: int, fs: int) -> tuple[list[int], list[Word]]:
        """The instants and the decodes that fall inside one capture's stream span."""
        return ([a for a in self.aimed if lo <= a <= hi],
                [w for w in self.words if lo / fs <= w.at <= hi / fs])


def shrike_grid(path: Path, fs: int, *, selection=None, lane=None) -> Grid:
    """Read one shrike arm's log for the two things its captures cannot hold."""
    if selection is not None:
        root = source_root(selection, path, transcript=True)
        lane = lane or selection.get('lane', 'rx')
        ref = selection.get('lanes', {}).get(lane, {})
        selected = select(selection, root / ref.get('source', ''), root=root, lane=lane)
        if not selected.complete:
            return Grid((), ())
        lo, hi = selected.bounds
        rows = [r for r in selected.grid if r.get('clock_generation') == selected.clock_generation]
        predecessor = next((r for r in reversed(rows) if r['sample'] <= lo), None)
        return Grid(tuple(r['sample'] for r in rows if lo <= r['sample'] < hi),
                    tuple(Word(r['sample']/selected.rate, 'cs '+r['word'], 'selected grid')
                          for r in rows if lo <= r['sample'] < hi and r.get('word')),
                    predecessor, selected.clock_generation)
    return _legacy_grid(path, fs)


@lru_cache(maxsize=8)
def _legacy_grid(path, fs):
    aimed: list[int] = []
    words: list[Word] = []
    pending: str | None = None
    for raw in path.read_text(errors="replace").splitlines():
        line = _ANSI.sub("", raw).strip()
        for at in _AIMED.findall(line):
            aimed.append(int(at))
            # A candidate answers at the instant the SAME cycle's `[grid]` line
            # reports, and the runner prints the two in that order. Cleared on the
            # first instant either way, so a cycle that read a codeword and then
            # printed no sample contributes no word rather than the next cycle's.
            if pending is not None:
                words.append(Word(int(at) / fs, pending, "codeword search"))
                pending = None
        if (m := _CANDIDATE.match(line)):
            pending = f"cs CS{m.group(1)}"
        elif (m := _WORD.match(line)):
            kind, text = m.group(2), m.group(3)
            cs = _CS_TEXT.match(text)
            words.append(Word(float(m.group(1)),
                              f"cs CS{cs.group(1)}" if cs else kind, text))
    return Grid(tuple(sorted(set(aimed))), tuple(sorted(words)))
