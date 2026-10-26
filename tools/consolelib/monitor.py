# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""What the rig is hearing, as it happens, and the same stream on disk.

Receive-only. Every record the runners raise goes to two sinks, the terminal and
`events.jsonl`, and the terminal is the one that is thinned: a held track prints
when it moves or changes shape, the file gets every update.

THE RIG SOURCE TAKES THE SHARED HARDWARE CLAIM. Every local mode contends for
the same claim, so no launcher can fly into a card this is already reading.
The Kiwi, a recording and an arm's tap touch no card and take no claim,
which is what lets a Kiwi monitor stand witness beside a flying arm.

AN ARM IS REHEARD, NOT RE-RECORDED. Its tap is already the audio, and a
recording replayed is already on disk; only a live source writes `rx-NNN.wav`.

INSIDE OUR OWN KEYING NOTHING IS A PEER. Whatever a runner raises in a span the
arm keyed is written as ours, with what it looked like in the detail, because
our signal leaks through the mute and a decoder will name it.
"""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import math
import re
from pathlib import Path
import signal
import threading
import time
import wave

from creance.monitor import OURS, TRACKER, aggregate, default_specs
from creance.monitor.activity import CONFIRMED, Detection
from hfhost.audio import FS, SAMPLE_WIDTH, AudioError, AudioSource, KiwiSource
from hfmodem.core import wav
import numpy as np

from . import arm, claim, rigsource

INTERRUPTED = 130
CHANNEL_EVERY_S = 10.0
SEGMENT_S = 600.0
# The tracker's centroid is sub-bin on an 11.7 Hz grid and wanders a few Hz on a
# held link; a move under half a bin is the same emission, and printing it would
# turn one session into a line every half second.
CENTRE_STEP_HZ = 6.0
# How much earlier than its burst a runner stamps it, where the runner does not
# say when the burst began. Kestrel's `t` is where its window put the burst: on
# the 2026-09-18 KB8AY arm it placed five of our own connect requests 0.25-0.30 s
# before the key-up the log and the mute agree on. Shrike's control-signal times
# are true, and on the WS8EOC-80 arm a shared lead called WS8EOC's CS ours.
LEAD_S = {'kestrel': 0.4}
# A per-track offset is measured from the modem's nominal centre, so past any
# tuning error it is where the track sits, and printing it signed reads as a
# station 10 kHz off frequency.
TUNING_HZ = 300.0
# The tracker relabels an emission as its shape fills in; a label that has not
# held this long is still being decided.
LABEL_HOLD_S = 2.0
DURING = re.compile(r'during our keying \d+$')
EDGES = re.compile(r'^(-?\d+)-(-?\d+) Hz')
DIRECTION = {'called': 'for {}', 'caller': 'from {}', 'answering': '{} answering',
             'sender': 'from {}', 'gateway': 'gateway {}'}


def capture_root(cfg=None):
    cfg = cfg or rigsource.station_config()
    where = cfg.audio.capture_dir if cfg else ''
    if not where:
        raise ValueError('station.toml has no [audio] capture_dir, and a monitor keeps other '
                         "people's traffic outside the tree. Set it.")
    return Path(where).expanduser()


def holder(mode):
    """The arm that is flying under `mode`'s lock: the newest one with no end yet."""
    for directory in sorted(Path(mode.arms).glob('*/provenance.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True):
        if 'ended_utc' not in json.loads(directory.read_text()):
            return directory.parent.name
    return None


@contextmanager
def every_lock(*, env=None):
    try:
        with claim.lease('monitor', env=env) as held:
            yield held
    except claim.hardware.ClaimBusy as error:
        # Contention names the holder; a remote receiver can still listen while
        # that owner uses the local radio and sound card.
        raise claim.hardware.ClaimBusy(
            f'{error}\nUse --kiwi HOST:PORT to monitor a remote receiver.') from None


def open_source(args):
    """``(source, dial, device, described)``."""
    if args.wav:
        path = wav.stored(args.wav)
        if not path.is_file():
            raise AudioError(f'no such recording: {path}')
        source = rigsource.wav_source(path, dial_hz=args.dial, realtime=False,
                                      span=tuple(args.span) if args.span else None)
        return source, args.dial, None, dict(kind='wav', path=str(path.resolve()))
    if args.arm:
        from . import armsource
        source = armsource.ArmSource(args.arm)
        return source, source.dial_hz, None, dict(
            kind='arm', arm=source.name, directory=str(source.directory),
            tap=str(source.tap), keyings=source.check.line)
    if args.kiwi:
        host, _, port = args.kiwi.partition(':')
        [span] = args.span
        source = KiwiSource(host, args.freq, port=int(port or 8073), span=span)
        return source, round(args.freq * 1e3), None, dict(kind='kiwi', host=host,
                                                          port=int(port or 8073), span_hz=span)
    freq_hz = round(args.freq * 1e3) if args.freq is not None else None
    cfg = rigsource.station_config()
    device = args.device or rigsource.input_device(cfg)
    source, dial = rigsource.source(device, freq_hz=freq_hz, dial_hz=args.dial)
    return source, dial.hz, device, dict(kind='rig', dial_source=dial.source,
                                         rig_mode=dial.mode, tuned_hz=freq_hz)


def validate(args):
    for flag, value in (('--freq', args.freq), ('--minutes', args.minutes)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{flag} needs a finite positive number.')
    if args.cap < 1:
        raise ValueError('--cap is how many decoders may run at once, and none reads nothing.')
    if args.minutes is not None and args.minutes <= 0:
        raise ValueError('--minutes is how long to listen, and it wants a positive number.')
    if args.kiwi and args.freq is None:
        raise ValueError('--kiwi needs --freq, the dial in kHz; a Kiwi is tuned by the request.')
    if args.span is not None and not (args.kiwi or args.wav):
        raise ValueError('--span is the width asked of a Kiwi, or the band of a recording '
                         'to hear; no other source has one to set.')
    if args.kiwi and args.span is not None and len(args.span) != 1:
        raise ValueError('--span for a Kiwi is one width in Hz.')
    if args.wav and args.span is not None and len(args.span) != 2:
        raise ValueError("--span for a recording is LO HI in Hz, the band to hear in place "
                         "of the station's filter.")
    if args.freq is not None and (args.wav or args.arm):
        raise ValueError('--freq tunes a receiver; a recording is already where it was. '
                         'Label one with --dial.')
    if args.dial is not None and (args.kiwi or args.arm):
        raise ValueError('--dial labels a run with no dial of its own; a Kiwi is tuned by '
                         '--freq and an arm carries its dial in its provenance.')
    if args.dial is not None and args.freq is not None:
        raise ValueError('--freq sets the dial and --dial names it; give one.')
    if args.device and (args.wav or args.arm or args.kiwi):
        raise ValueError("--device picks the rig's input, and this source is not the rig.")
    if args.kiwi and args.span is None:
        args.span = [12000]
    return args


def stamp(t, wall):
    if wall is None:
        return f'{t:7.1f}s'
    return time.strftime('%H:%M:%S', time.localtime(wall))


def dial_khz(hz):
    return f'{hz / 1e3:.3f}' if hz else 'unknown'


class Lines:
    """The operator's view: one line a record, tracks only when they move."""

    def __init__(self, clock):
        self.clock = clock
        self.shown, self.opened, self.pending = {}, {}, {}

    def say(self, t, wall, label, body, tentative=False):
        print(f'{stamp(t, wall)} {"?" if tentative else " "}{label:<9} {body}', flush=True)

    def __call__(self, d):
        line = self.spell(d)
        if line is not None:
            self.say(d.t, self.clock(d.t, d.wall), *line)

    def spell(self, d):
        tentative = d.grade != CONFIRMED
        if d.modem == OURS:
            return 'ours', self.relabelled(d), tentative
        if d.modem == TRACKER:
            return self.track(d)
        if d.text is not None or d.protocol == 'CW':
            return d.protocol.lower(), self.decoded(d), tentative
        role = d.role or ('called' if d.kind == 'CONNECT' else '')
        who = DIRECTION.get(role, '{}').format(d.station) if d.station else ''
        parts = [d.kind, who, _offset(d.offset_hz, d.centre_hz), _rf(d.rf_hz),
                 '' if tentative else d.grade, d.detail]
        return d.protocol.lower(), '  '.join(p for p in parts if p), tentative

    def ended(self, d):
        first = self.opened.setdefault(d.track, d.t)
        if d.kind != 'track-end':
            return None
        self.shown.pop(d.track, None)
        self.pending.pop(d.track, None)
        return f'ended  {d.t - first:.0f} s'

    def moved(self, d):
        """Whether a live track's update is worth a line: a first sight, a centre
        that moved half a bin, or a label that has held."""
        last = self.shown.get(d.track)
        if last and last[1] != d.protocol:
            since = self.pending.setdefault(d.track, (d.protocol, d.t))
            if since[0] != d.protocol:
                self.pending[d.track] = (d.protocol, d.t)
                return False
            if d.t - since[1] < LABEL_HOLD_S:
                return False
        elif last and (d.centre_hz is None or last[0] is not None
                       and abs(d.centre_hz - last[0]) < CENTRE_STEP_HZ):
            self.pending.pop(d.track, None)
            return False
        self.pending.pop(d.track, None)
        self.shown[d.track] = (d.centre_hz, d.protocol)
        return True

    def track(self, d):
        label = f'track#{d.track}'
        ended = self.ended(d)
        if ended is not None:
            return label, ended, True
        if not self.moved(d):
            return None
        parts = []
        edges = EDGES.match(d.detail)
        if edges:
            parts.append(f'{edges[1]}-{edges[2]} Hz')
        if d.centre_hz is not None:
            parts.append(f'ctr {d.centre_hz:.0f}'
                         + (f' ({d.offset_hz:+.0f})' if _tuning(d.offset_hz) else ''))
        if d.level_dbfs is not None:
            parts.append(f'{d.level_dbfs:.0f} dBFS')
        parts.append(d.protocol + (f', {d.cadence_s:.2f} s' if d.cadence_s else ''))
        parts.append(_rf(d.rf_hz))
        keyed = DURING.search(d.detail)
        if keyed:
            parts.append(keyed[0])
        return label, '  '.join(p for p in parts if p), True

    @staticmethod
    def decoded(d):
        text = (d.text or '') if d.protocol == 'CW' else (
            ' / '.join((d.text or '').strip('\r\n').splitlines()))
        who = DIRECTION.get(d.role, '{}').format(d.station) if d.station else ''
        parts = [f'{d.centre_hz:.0f} Hz' if d.centre_hz is not None else '']
        # Keep CW's established field order; message headers precede their text.
        parts += [d.detail, text, who] if d.protocol == 'CW' else [who, d.detail, text]
        parts.append(_rf(d.rf_hz))
        return '   '.join(p for p in parts if p)

    @staticmethod
    def relabelled(d):
        parts = [d.kind, f'{d.centre_hz:.0f} Hz' if d.centre_hz is not None else '',
                 f'{d.level_dbfs:.0f} dBFS' if d.level_dbfs is not None else '',
                 f'({d.detail})']
        return '  '.join(p for p in parts if p)

    def keying(self, record, keyed, measured):
        n = keyed.logged + 1 if keyed.logged is not None else '?'
        self.say(record['t'], self.clock(record['t'], record['wall']), 'ours',
                 f'TX[{n}] {record["duration_s"]:.2f} s   '
                 f'{measured.level_dbfs:.0f} dBFS in the keyed window, '
                 f'mute {measured.mute_db:.0f} dB',
                 record['grade'] != CONFIRMED)


def _tuning(offset_hz):
    return offset_hz is not None and abs(offset_hz) <= TUNING_HZ


def _offset(offset_hz, centre_hz=None):
    if _tuning(offset_hz):
        return f'{offset_hz:+.0f} Hz'
    return f'at {centre_hz:.0f} Hz' if offset_hz is not None and centre_hz is not None else ''


def _rf(hz):
    return f'dial {hz / 1e3:.3f}' if hz is not None else ''


class Events:
    """`events.jsonl`: every record, one a line, flushed as it comes."""

    def __init__(self, path, *, ownership=None):
        self.file = io.TextIOWrapper(ownership.open()) if ownership else open(path, "w")
        self.ownership = ownership

    def __call__(self, record):
        row = record.to_json() if isinstance(record, Detection) else dict(record)
        if self.ownership is not None:
            at = row.get('t')
            if isinstance(at, (int, float)) and not isinstance(at, bool) and np.isfinite(at):
                row['event_stream_sample'] = round(at * FS)
            row = self.ownership.metadata(row)
        self.file.write(json.dumps(row) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


class Recorder:
    """PCM16 segments on the frame-reader thread, with source sample placement."""

    def __init__(self, directory, segment_s=SEGMENT_S, *, writer=None, clock_id=None):
        self.directory, self.limit = Path(directory), int(segment_s * FS)
        if self.limit <= 0:
            raise ValueError('Recording segment must contain samples')
        self.writer, self.clock_id = writer, clock_id
        if writer is not None and (self.directory.absolute() != writer.root or not clock_id):
            raise ValueError('Recording writer root and clock must agree')
        self.index, self.written, self.file = 0, 0, None
        self.paths, self.segments = [], []
        self._handle, self._artifact, self._sidecar = None, None, None
        self.start, self.next_sample = None, None

    def _facts(self, state):
        return dict(wav=self.paths[-1].name, samplerate=FS, channels=1,
                    samples=self.written, start_stream_sample=self.start,
                    end_stream_sample=self.start + self.written,
                    capture_state=state, normalised_on_write=False)

    def _open(self, start):
        path = self.directory / f'rx-{self.index:03d}.wav'
        self.index += 1
        self.start, self.written = start, 0
        self.paths.append(path)
        if self.writer is not None:
            self._artifact = self.writer.reserve(path.name, clock_id=self.clock_id)
            self._sidecar = self._artifact.versions(path.with_suffix('.json').name)
            self._sidecar.update(self._facts('ongoing'))
            self._handle = self._artifact.open()
        self.file = wave.open(str(path) if self._handle is None else self._handle, 'wb')
        self.file.setnchannels(1)
        self.file.setsampwidth(SAMPLE_WIDTH)
        self.file.setframerate(FS)

    def __call__(self, t0, pcm):
        if len(pcm) % SAMPLE_WIDTH:
            raise ValueError('Recording frame has an incomplete sample')
        if self.writer is not None:
            if not isinstance(t0, (int, np.integer)) or isinstance(t0, bool) or t0 < 0:
                raise ValueError('Recording frame requires a source sample index')
            if self.next_sample is not None and t0 != self.next_sample:
                self.close()
        at = int(t0)
        try:
            while pcm:
                if self.file is None or self.written >= self.limit:
                    self.close()
                    self._open(at)
                room = (self.limit - self.written) * SAMPLE_WIDTH
                part = pcm[:room]
                self.file.writeframes(part)
                count = len(part) // SAMPLE_WIDTH
                self.written += count
                at += count
                self.next_sample = at
                pcm = pcm[room:]
        except BaseException:
            self.close(state='partial')
            raise

    def close(self, *, state='finalized'):
        if self.file is None:
            return
        completed = False
        try:
            self.file.close()
            completed = True
        finally:
            self.file = None
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            if self._sidecar is not None:
                import hashlib
                path = self.paths[-1]
                with path.open('rb') as audio:
                    digest = hashlib.file_digest(audio, 'sha256').hexdigest()
                row = self._sidecar.update(self._facts(state if completed else 'partial') | {'sha256': digest})
                self.segments.append({k: row[k] for k in ('artifact_id', 'clock_id', 'start_stream_sample', 'end_stream_sample', 'samples', 'capture_state', 'sha256')})
                self._sidecar, self._artifact = None, None


class Channel:
    """Every ten seconds of audio, what `core.busy` makes of the last eight."""

    def __init__(self, lines, dial, every_s=CHANNEL_EVERY_S, *, classify=True):
        from hfmodem.core import busy
        self.busy, self.lines, self.dial = busy, lines, dial
        self.every = int(every_s * FS)
        self.keep = int(busy.WINDOW_S * FS)
        self.frames, self.held, self.due = deque(), 0, self.every
        self.classify = classify

    def __call__(self, t0, pcm):
        self.frames.append(pcm)
        self.held += len(pcm) // SAMPLE_WIDTH
        while self.held - len(self.frames[0]) // SAMPLE_WIDTH >= self.keep:
            self.held -= len(self.frames.popleft()) // SAMPLE_WIDTH
        end = t0 + len(pcm) // SAMPLE_WIDTH
        if end >= self.due:
            self.due = end + self.every
            t = end / FS
            self.lines.say(t, self.lines.clock(t, time.time()), 'channel', self.verdict())

    def verdict(self):
        from hfmodem.core import levels
        a = np.frombuffer(b''.join(self.frames), '<i2').astype(float) / 32768.0
        seconds = len(a) / FS
        loud, quiet = self.busy.level_range(a)
        fault = self.busy.receiver_fault(loud, quiet, seconds=seconds, device='')
        s = self.busy.scores(a) if self.classify else None
        db = self.busy.occupancy_db(a)
        level = f'{np.median(db):.0f} dBFS' if db.size else '-inf dBFS'
        if s is None:
            said = f'{"unjudged" if self.classify else "receiving"}  {level}'
        else:
            name, margin = self.busy.dominant(s)
            said = f'{"busy " if margin >= 0 else "clear"}   {level}   {name} {margin:+.1f}'
        said += f'   dial {dial_khz(self.dial())}'
        if levels.railed_ppm(a) > 0:
            said += '   RAILED'
        elif fault is not None:
            said += '   ' + ('DEAF' if fault.kind == 'deaf' else 'MUTED')
        return said


class Cat:
    """The rig's dial, re-read off the audio loop: a CAT that has died under the
    tuner's RFI takes seconds to time out, and the card does not wait."""

    def __init__(self, hz):
        self.hz, self.reading = hz, False

    def __call__(self):
        if not self.reading:
            self.reading = True
            threading.Thread(target=self._read, daemon=True).start()
        return self.hz

    def _read(self):
        try:
            self.hz = rigsource.read_dial().hz
        finally:
            self.reading = False


class Ours:
    """The arm's keyings, handed out as the audio passes them, and the relabelling."""

    def __init__(self, source, lines, events):
        self.spans = list(source.spans)
        self.called = source.provenance.get('call', '')
        self.pending = deque(zip(source.records(), source.keyed, source.measured))
        self.lines, self.events = lines, events

    def __call__(self, t0, pcm):
        now = (t0 + len(pcm) // SAMPLE_WIDTH) / FS
        while self.pending and self.pending[0][0]['t'] <= now:
            self.emit(*self.pending.popleft())

    def emit(self, record, keyed, measured):
        self.events(record)
        self.lines.keying(record, keyed, measured)

    def relabel(self, d):
        """Ours when it began inside a keyed span, or when it is a PACTOR-1 connect
        to the station this arm called: only the caller sends one, and shrike stamps
        it with the start of the 2 s window that decoded it, not with the burst."""
        if d.modem == TRACKER:
            return self.during(d)
        found = ' '.join(p for p in (d.protocol, d.kind, d.station) if p)
        for i, (t0, t1) in enumerate(self.spans, 1):
            if t0 <= onset(d) < t1:
                return self.ours(d, f'{found} inside our keying {i} at {t0:.2f}-{t1:.2f} s')
        if d.kind == 'CONNECT' and d.station and d.station == self.called:
            return self.ours(d, f'{found}: this arm called {self.called}, and only the '
                                'caller sends a connect (so a third station calling '
                                f'{self.called} would read as ours too)')
        return d

    def during(self, d):
        """A track is not a burst: it may open on our keying and carry the peer's
        answers after it, so it is never ours. An update says when its own instant
        fell in a keyed span; its start and its end say nothing of the kind."""
        if d.kind != 'track':
            return d
        for i, (t0, t1) in enumerate(self.spans, 1):
            if t0 <= d.t < t1:
                return replace(d, detail=f'{d.detail}; during our keying {i}')
        return d

    @staticmethod
    def ours(d, why):
        return replace(d, modem=OURS, station='', role='',
                       detail=f'{why}, not a peer' + (f'; {d.detail}' if d.detail else ''))


def onset(d):
    return d.t_start if d.t_start is not None else d.t + LEAD_S.get(d.modem, 0.0)


class Tapped(AudioSource):
    """The source, with each frame handed to the audio sinks on its way through."""

    def __init__(self, inner, taps, stop, limit_s=None):
        self.inner, self.taps, self.stop, self.limit_s = inner, taps, stop, limit_s
        self.window = inner.window
        self.heard_s = 0.0

    def frames(self):
        for t0, pcm in self.inner.frames():
            for tap in self.taps:
                tap(t0, pcm)
            self.heard_s = (t0 + len(pcm) // SAMPLE_WIDTH) / FS
            if self.limit_s and self.heard_s >= self.limit_s:
                self.stop.set()
            yield t0, pcm

    def close(self):
        self.inner.close()


def run_directory(args, source):
    if args.arm:
        return Path(source.directory) / 'monitor-runs' / f'monitor-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    return capture_root() / f'monitor-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'


def make_run_directory(args, source):
    directory = run_directory(args, source)
    directory.parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 100):
        candidate = directory if n == 1 else directory.with_name(f'{directory.name}-{n}')
        try:
            from hfmodem.core import capture_identity
            capture_identity.allocate({'analysis': candidate}, {'analysis': 'analysis'},
                                      primary_root='analysis', kind='monitor-analysis')
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(directory)


def window_record(window):
    return dict(label=window.label, lo_hz=window.lo_hz, hi_hz=window.hi_hz,
                rf_origin_hz=window.rf_origin_hz, inverted=window.inverted)


@contextmanager
def stopping():
    stop, caught = threading.Event(), []

    def handler(signum, frame):
        # The first interrupt drains queued detections. That drain has no time
        # bound, so a second interrupt must let the operator stop waiting.
        if caught:
            raise KeyboardInterrupt
        caught.append(signum)
        stop.set()

    previous = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield stop, caught
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


def report(args):
    """`console monitor`: listen until the source ends, the minutes run out or ^C."""
    from . import monitor_modes
    if args.list_modes:
        for name in monitor_modes.NAMES:
            setting = monitor_modes.WAVEFORMS.get(name)
            hint = f'  --mode {setting[1]} (default)' if setting and setting[1] else ''
            if name == 'stanag':
                hint = "  --mode '4285 1200L' (default); choose another standard explicitly"
            print(f'tools/console {name} monitor{hint}')
        print('Aliases: ' + ', '.join(f'{a}={n}' for a, n in monitor_modes.ALIASES.items()))
        return 0
    chosen = getattr(args, 'monitor_mode', None)
    if chosen:
        monitor_modes.peer_frequency(args)
    validate(args)
    specs = default_specs(no_rtty_usos=args.no_rtty_usos)
    if chosen:
        if args.dispatch != 'whole-band':
            raise ValueError('Single-mode monitoring runs one receiver set over the rig audio; '
                             'omit --per-track. Use `console monitor` for track discovery.')
        specs = monitor_modes.selected(args, specs)
        for spec in specs:
            why = spec.unavailable()
            if why:
                raise RuntimeError(f'{chosen} monitor cannot start: {spec.name}: {why}')
    from hfmodem.core import capture_identity
    from . import armsource
    with ExitStack() as held:
        if not (args.wav or args.arm or args.kiwi):
            held.enter_context(every_lock())
        attached = capture_identity.allocate_analysis(armsource.find(args.arm)) if args.arm else capture_identity.load(make_run_directory(args, None))
        attached_writer = attached.claim('analysis')
        attached_events = attached_writer.reserve('events.jsonl', clock_id=attached.capture_id + ':monitor-samples')
        attached_provenance = attached_writer.reserve('provenance.json').versions()
        attached_provenance.update({'capture_state': 'opening'})
        held.callback(attached.end)
        outcome = {'state': 'partial', 'written': 0, 'partial': 1}
        held.callback(lambda: attached_writer.update_status(**outcome))
        try:
            source, dial_hz, device, described = open_source(args)
            held.enter_context(source)
        except BaseException as exc:
            attached_provenance.update({'capture_state': 'partial', 'error_type': type(exc).__name__})
            raise
        live = not (args.wav or args.arm)
        directory = attached.root('analysis')
        origin = source.started() if args.arm else None
        clock = ((lambda t, wall: wall) if live else
                 (lambda t, wall: origin + t) if origin is not None else (lambda t, wall: None))
        lines = Lines(clock)
        events = Events(directory / 'events.jsonl', **({'ownership': attached_events} if attached_events else {}))
        def close_events():
            try:
                events.close()
            except BaseException:
                outcome.update(state='partial', partial=outcome['partial'] + 1)
                raise
        held.callback(close_events)
        ours = Ours(source, lines, events) if args.arm else None
        dial = (Cat(dial_hz) if described['kind'] == 'rig' and args.dial is None
                else (lambda: dial_hz))
        taps = [Channel(lines, dial, classify=not chosen)]
        recorder = None
        if live and not args.no_record:
            recorder = Recorder(directory, writer=attached_writer, clock_id=attached_events.record['clock_id'])
            held.callback(recorder.close)
            taps.append(recorder)
        if ours:
            taps.append(ours)
        provenance = dict(
            source=described, window=window_record(source.window),
            dispatch=args.dispatch if args.dispatch == 'whole-band'
            else f'{args.dispatch}, cap {args.cap}',
            dial_hz=dial_hz, device=device, argv=list(args.console_argv),
            git=arm.git_head(), started_utc=datetime.now(timezone.utc).isoformat())
        if chosen:
            provenance.update(monitor_mode=chosen, peer=getattr(args, 'monitor_peer', None),
                              receivers=[dict(name=s.name, args=list(s.args)) for s in specs])
        written = directory / 'provenance.json'
        if attached_provenance:
            attached_provenance.update(provenance | {'capture_state': 'ongoing'})
        else:
            written.write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'# monitor {chosen or "all modes"}  {source.window.label}  dial {dial_khz(dial_hz)}  -> {directory}',
              flush=True)

        def sink(d):
            if chosen and described['kind'] == 'rig':
                # The operator can move the radio while buffered audio decodes.
                # The opening dial cannot identify those later detections' RF.
                d = replace(d, rf_hz=None)
            d = ours.relabel(d) if ours else d
            events(d)
            lines(d)

        with stopping() as (stop, caught):
            tapped = Tapped(source, taps, stop, args.minutes and args.minutes * 60)
            completed = False
            try:
                aggregate.run(tapped, specs,
                              live=live, stop=stop,
                              summary_every_s=float('inf'), out=io.StringIO(), sinks=[sink],
                              dispatch=args.dispatch, cap=args.cap)
                completed = True
            finally:
                state = 'finalized' if completed and not caught else 'partial'
                if recorder:
                    recorder.close(state=state)
                provenance.update(
                    ended_utc=datetime.now(timezone.utc).isoformat(),
                    audio_s=round(tapped.heard_s, 2), interrupted=bool(caught),
                    segments=[p.name for p in recorder.paths] if recorder else [], segment_records=recorder.segments if recorder else [])
                if attached_provenance:
                    attached_provenance.update(provenance | {'capture_state': state})
                    outcome.update(state=state, written=1 + len(recorder.segments) if recorder else 1, partial=0 if state == 'finalized' else 1)
                else:
                    written.write_text(json.dumps(provenance, indent=2) + '\n')
    return INTERRUPTED if caught else 0
