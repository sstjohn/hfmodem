# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every track its own decoders: the tracker's emissions, dispatched.

The modems expect their signal at a nominal centre and keep expecting it; in a
12 kHz Kiwi window nothing is within any modem's acquisition, and even in a rig
window two BW500 sessions read as one to a decoder written for one signal near
1500 Hz. So the tracker hears the whole window, and each track it raises gets an
instance of every modem its width admits, started with ``--centre`` and ``--bw``
so the runner cuts the track out and moves it to where that modem listens.
Each instance is first fed the audio from a little before the track began, out of
a ring, so it hears the burst that opened the track; it retires when the track
ends. Lanner can continue an unfinished emission until its decoder closes it,
with a 600-second bound and an EOF flush. The shape only prunes: no classifier
picks a modem, and the modem's own acquisition finishes the job.

Instances are processes, so a busy wide window is a lot of them. The number
running at once is capped, loudest tracks first, and a track that ends without
the decoders its width asked for is named at the end of the pass beside the audio
the pass dropped: a track nobody decoded is not a track that held nothing.
"""

from __future__ import annotations

import math
import queue
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher

from hfhost.audio import FS, SAMPLE_WIDTH, Window

from .activity import TRACKER, UNKNOWN, Detection
from .driver import RUNNERS, ModemMonitor, MonitorSpec

#: What a detection's ``via`` says produced it.
WHOLE_BAND, PER_TRACK = "whole-band", "per-track"
#: Decoders running at once, the tracker not counted.
CAP = 12
#: A track this much louder than the quietest one holding decoders takes that
#: track's decoders when the cap is full. A replay of the synthesized DL6MAA
#: fixture raises tracks at -94 to -103 dBFS off its synthesis residue, and they
#: held every slot when the -23 dBFS link widened and asked for its PACTOR-3
#: search.
PREEMPT_DB = 20.0
#: A track is offered decoders only when its strongest bin stands this far over
#: the tracker's floor there, the tracker's own burst seed, and it is at least
#: this loud. On a replay of the synthesized DL6MAA fixture, which leaves a floor
#: near -125 dBFS, residue at -88 to -106 dBFS rose 10 dB over it in two dozen tracks
#: and took the cap from the -30 dBFS link; a real receiver's band is louder
#: than the absolute floor everywhere.
ADMIT_OVER_DB = 10.0
ADMIT_DBFS = -80.0
#: Tracks kept after they end, so a whole-band detection that arrives late can
#: still be placed on the one it sat in.
HISTORY = 1000
#: How far before a track's first burst its instances start hearing.
PREROLL_S = 2.0
#: Audio held for instances that start late: the tracker reports a track up to
#: five seconds after its first burst (2.4 s is typical), over a replay the
#: source runs as much as 16 s ahead of the tracker (a backlog and a pipe), and
#: a track the cap held back starts when a slot frees.
RING_S = 45.0
#: Where every runner moves its track to, and so what its offsets are from.
NOMINAL_HZ = 1500.0
#: Where an instance's channel is centred: the track's centroid, to this grid.
#: A centroid is not a tuning. Over a PACTOR-1 link it carries the mark/space
#: duty and both stations' errors at once: KE5YTA's link, 5 Hz low on both
#: tones, reads 1475.6 Hz, and moved by that much its connect frames stopped
#: decoding. On the grid, a signal already within its modem's reach stays where
#: its own acquisition finds it, and one at 700 Hz or 8.2 kHz moves home exactly.
GRID_HZ = 100.0
#: Detections kept to recognise the same frame from a second instance, and how
#: close two must sit: DEDUP_S is shrike's own, OFFSET_HZ a 5 Hz fit grid twice.
SAID = 2000
DEDUP_S = 0.8
OFFSET_HZ = 10.0
#: How far outside a track's reported span a whole-band detection may sit and
#: still be placed on it: a decoder stamps a burst's start, and the tracker
#: dates a track from its first burst.
PLACE_S = 0.5
#: What a retiring instance is given to decode the audio it still holds. It is
#: retired off the loop, so the wait costs the stream nothing.
RETIRE_S = 120.0
LANNER_HOLD_S = 600.0


@dataclass(frozen=True, slots=True)
class Admits:
    """The track widths a modem is started for, and the channel it hears.

    ``bw_hz`` is the width cut around the track's centre, ``None`` for the
    track's own width plus ``margin_hz``: VARA holds its bandwidth for the whole
    link, so a BW500 instance can be kept off its neighbour. PACTOR and ARDOP
    links open narrow and widen, so theirs is the modem's widest waveform.

    ``wide_hz`` splits a modem whose hypotheses divide by width into a ``pair``
    and a ``wide`` instance; a track that widens gets the second beside the first.
    """
    lo_hz: float
    hi_hz: float
    bw_hz: float | None = None
    margin_hz: float = 0.0
    wide_hz: float | None = None

    def admits(self, bw: float) -> str | None:
        """The instance a track this wide gets, ``""`` for the only one, or
        None."""
        if not self.lo_hz <= bw <= self.hi_hz:
            return None
        if self.wide_hz is None:
            return ""
        return "wide" if bw >= self.wide_hz else "pair"

    def channel(self, bw: float) -> float:
        return self.bw_hz if self.bw_hz is not None else max(bw, 600.0) + self.margin_hz


#: Measured widths, -26 dB: a VARA 500 connect request 550-715 Hz, VARA 2300
#: overs 2.3-2.5 kHz, a PACTOR-1 link 190-490 Hz, a PACTOR-4 chirp 2.4 kHz. On a
#: crowded channel the tracker can take a VARA 500 connect request in fragments
#: 130-220 Hz wide, one hop at a time, and kestrel's channel is wide enough to
#: hear the whole request from any of them. Under 100 Hz is the CW reader's,
#: which the tracker runs itself. A PACTOR link past the 500-wide class is
#: searched for PACTOR-3: DL6MAA's upgraded link reads 785-1090 Hz, the median
#: of its PACTOR-1 and PACTOR-3 overs' edges, where the tones alone are 840 apart.
#: A PACTOR-3 link heard on a Kiwi measured 3035-3059 Hz once the tracker took
#: its neighbours in, so the search admits the width PACTOR-4's does.
ADMITS = {
    "kestrel": Admits(100.0, 3300.0, None, 240.0),
    "shrike": Admits(150.0, 3300.0, 2800.0, wide_hz=700.0),
    "besra": Admits(150.0, 2700.0, 2600.0),
    "p4": Admits(1500.0, 3300.0, 2800.0),
    #: Measured -26 dB widths: synthetic 170 Hz shift RTTY 328 Hz, NAVTEX 352 Hz.
    "lanner": Admits(150.0, 600.0, 700.0),
}


def tracker_spec(base: MonitorSpec, window: Window) -> MonitorSpec:
    """The tracker runner beside the modems ``base`` names, over ``window``."""
    return replace(base, name=TRACKER, runner=str(RUNNERS / "tracker_runner.py"),
                   args=("--span", f"{window.lo_hz:g}", f"{window.hi_hz:g}", "--heard"))


@dataclass
class _Track:
    id: int
    t_start: float
    centre_hz: float
    bw_hz: float
    level_dbfs: float
    label: str
    last: float = 0.0
    over_db: float | None = None
    wanted: set[tuple[str, str]] = field(default_factory=set)
    started: dict[tuple[str, str], ModemMonitor] = field(default_factory=dict)
    #: decoders stopped for a louder track, not to be started again
    cut: set[tuple[str, str]] = field(default_factory=set)
    #: where the track's first instance was centred, and every later one is: a
    #: link that upgrades keeps its dial while its centroid follows its new
    #: edges, and DL6MAA's moved 55 Hz, past what PACTOR-3 will decode through.
    #: A centroid that has walked more than a grid step away is another
    #: emission the tracker took into the track, and the next instance is
    #: homed on it: a KiwiSDR copy of WS8EOC's PACTOR-3 link opened as a
    #: PACTOR-1 neighbour 406 Hz below it, and its wide search heard nothing.
    home_hz: float | None = None
    ended_at: float | None = None
    covered: set[tuple[str, str]] = field(default_factory=set)

    def stamp(self, d: Detection, window: Window) -> Detection:
        """The detection on this track, and where its own decoder put it.

        An instance's offset is its modem's own measurement from where it was
        moved to, and that is a finer place for the emission than the track's
        centroid: the same VARA connect request reads -46.9 Hz at 1500 Hz and
        +6653.1 Hz at 8.2 kHz, one dial frequency, where the tracker's
        centroids of the two land 18 Hz apart.
        """
        centre = self.centre_hz if d.offset_hz is None else NOMINAL_HZ + d.offset_hz
        return replace(d, track=self.id, centre_hz=centre, bw_hz=self.bw_hz,
                       rf_hz=window.rf_hz(centre))


class _Stamped:
    """What an instance reports, placed on its track before anyone sees it."""

    def __init__(self, owner: "Channelizer", track: _Track) -> None:
        self.owner, self.track = owner, track

    def put(self, d: Detection) -> None:
        self.owner.report(self.track.stamp(d, self.owner.window))


class Channelizer:
    """The tracker and every track's instances, driven as one monitor.

    ``feed``, ``alive``, ``stop`` and the loss accounting are
    :class:`ModemMonitor`'s, so the aggregate loop runs this where it would run
    the whole-band runners.
    """

    def __init__(self, window: Window, tracker: MonitorSpec, specs: list[MonitorSpec],
                 out: "queue.Queue[Detection]", *, cap: int = CAP,
                 backlog_s: float = ModemMonitor.BACKLOG_S, lossy: bool = True) -> None:
        self.window, self.out, self.cap, self.lossy = window, out, cap, lossy
        self.backlog_s = backlog_s
        self.modems = {s.name: s for s in specs if s.name in ADMITS}
        self._records: "queue.Queue[Detection]" = queue.Queue()
        self.tracker = ModemMonitor(tracker, self._records, backlog_s=backlog_s,
                                    lossy=lossy)
        self.started: list[MonitorSpec] = []
        self._lost: list[ModemMonitor] = []
        self.live: dict[int, _Track] = {}
        self.short: list[_Track] = []
        self._held: dict[int, _Track] = {}
        self._lanner_state: dict[int, tuple[float, bool]] = {}
        self.seen = self.peak = 0
        self._ring: deque[tuple[int, bytes]] = deque()
        self._ring_n = 0
        self._released = 0
        self._sent: dict[int, int] = {}
        self._tick = 0.0
        self._retiring: list[tuple[threading.Thread, ModemMonitor, list[bool]]] = []
        self._ended: set[int] = set()
        self._said: deque[Detection] = deque(maxlen=SAID)
        self._lock = threading.Lock()
        self._since: dict[int, float] = {}
        self.history: deque[_Track] = deque(maxlen=HISTORY)

    @property
    def alive(self) -> bool:
        return self.tracker.alive

    @property
    def running(self) -> list[ModemMonitor]:
        return [m for tr in [*self.live.values(), *self._held.values()] for m in tr.started.values()]

    @property
    def decoded_s(self) -> float:
        """Where on the stream everything the tracker and every instance will
        say about the audio before has been said, each by its own account: the
        tracker has reported every emission that began earlier, and every
        running or retiring instance has decoded that far. An instance that has
        said nothing yet has decoded nothing past where its audio began."""
        told = self.tracker.decoded_s
        at = [-math.inf if told is None else told]
        for m in self.running + [m for t, m, _ in self._retiring if t.is_alive()]:
            at.append(self._since.get(id(m), 0.0) if m.decoded_s is None else m.decoded_s)
        return min(at)

    def place(self, d: Detection) -> Detection:
        """A whole-band detection on the track it sat in, if one held its time
        and the place its decoder listens: 1500 Hz and the offset it read."""
        hz = NOMINAL_HZ + (d.offset_hz or 0.0)
        held = [tr for tr in [*self.history, *self.live.values()]
                if tr.t_start - PLACE_S <= d.t <= tr.last + PLACE_S
                and abs(hz - tr.centre_hz) <= tr.bw_hz / 2]
        if not held:
            return d
        tr = max(held, key=_level)
        centre = hz if d.offset_hz is not None else tr.centre_hz
        return replace(d, track=tr.id, centre_hz=centre, bw_hz=tr.bw_hz,
                       rf_hz=self.window.rf_hz(centre))

    @property
    def monitors(self) -> list[ModemMonitor]:
        """The tracker, every running instance, and every retired one that
        dropped audio; the rest are let go, since a live pass starts thousands."""
        return [self.tracker, *self.running, *self._lost]

    def start(self) -> None:
        self.tracker.start()

    def feed(self, t0: int, pcm: bytes) -> None:
        self._ring.append((t0, pcm))
        self._ring_n += len(pcm) // SAMPLE_WIDTH
        while self._ring_n - len(self._ring[0][1]) // SAMPLE_WIDTH >= RING_S * FS:
            self._ring_n -= len(self._ring.popleft()[1]) // SAMPLE_WIDTH
        self.tracker.feed(t0, pcm)
        self._dispatch()

    def _release(self, t: float, *, held_only=False) -> None:
        """Hand the running instances the audio up to where the tracker is.

        An instance hears no further than the tracker has reported, so it stops
        where its track ended, however far ahead of the tracker the source has
        run. Over a replay that is seconds: the source waits only for a backlog,
        and a pipe holds 8 s of audio past it, and a PACTOR-1 track that ended at
        13.6 s had its instance name a PACTOR-3 header at 21.2 s. Live, it costs
        the tracker's half-second tick. A held lanner instance is the exception:
        it hears past its track's end until its emission closes or the hold
        limit expires.
        """
        upto = round(t * FS)
        if not held_only and upto <= self._released:
            return
        running = ([m for tr in self._held.values() for m in tr.started.values()]
                   if held_only else self.running)
        for t0, pcm in self._ring:
            if t0 >= upto or (not held_only and t0 < self._released):
                continue
            end = t0 + len(pcm) // SAMPLE_WIDTH
            for m in running:
                if t0 >= self._sent.get(id(m), self._released):
                    m.feed(t0, pcm)
                    self._sent[id(m)] = end
            if not held_only:
                self._released = end

    def report(self, d: Detection) -> None:
        """An instance's detection, unless another instance already said it.

        A detection that names no protocol is energy with a shape, and the
        track already says that. Channels overlap, and a PACTOR link the
        tracker splits in two, or an upgrade that starts a wide instance
        beside a narrow one, puts the same frame in two instances; the same
        modem, protocol, kind and station at the same moment and offset is one
        frame. Lanner uses on-air identity or similar text from different
        decoder instances, so repeated lines from one instance survive.
        Records already emitted cannot be replaced: the first copy wins.
        """
        if d.modem == "lanner" and d.kind == "emission-status":
            if d.track in self._ended and d.track not in self._held:
                return
            self._lanner_state[d.track] = (d.t, d.detail == "open")
            return
        if d.protocol == UNKNOWN:
            return
        with self._lock:
            if any(same_frame(d, e) for e in self._said):
                return
            self._said.append(d)
        self.out.put(replace(d, via=PER_TRACK))

    def _dispatch(self) -> None:
        while True:
            try:
                d = self._records.get_nowait()
            except queue.Empty:
                break
            if d.centre_hz is not None and d.rf_hz is None:
                d = replace(d, rf_hz=self.window.rf_hz(d.centre_hz))
            if d.modem == TRACKER and d.track is not None:
                if d.t > self._tick:
                    self._grant()
                    self._tick = d.t
                self._release(d.t)
                self._follow(d)
            self.out.put(d)
        # Tracker watermarks advance even with no occupied bins or track rows.
        progress = self.tracker.decoded_s
        if progress is not None:
            if math.isinf(progress) and self._ring:
                t0, pcm = self._ring[-1]
                progress = (t0 + len(pcm) // SAMPLE_WIDTH) / FS
            self._release(progress, held_only=True)
        for tid, tr in list(self._held.items()):
            state = self._lanner_state.get(tid)
            # Status distinguishes unfinished traffic from ordinary lookback.
            complete = (state[0] >= tr.ended_at and not state[1] if state is not None else
                        all(m.decoded_s is not None and m.decoded_s >= tr.ended_at
                            for m in tr.started.values()))
            if (complete or all(m.decoded_s == math.inf for m in tr.started.values())
                    or (progress is not None and progress >= tr.ended_at + LANNER_HOLD_S)):
                self._held.pop(tid)
                self._lanner_state.pop(tid, None)
                self._stop_all(tr.started.values(), RETIRE_S)

    def _follow(self, d: Detection) -> None:
        if d.track in self._ended:
            return
        tr = self.live.get(d.track)
        if tr is None:
            self.seen += 1
            tr = self.live[d.track] = _Track(
                d.track, d.t_start if d.t_start is not None else d.t,
                d.centre_hz, d.bw_hz, d.level_dbfs, d.protocol)
        tr.centre_hz, tr.bw_hz = d.centre_hz, d.bw_hz
        tr.level_dbfs, tr.label, tr.last, tr.over_db = d.level_dbfs, d.protocol, d.t, d.over_db
        if self._admitted(tr):
            tr.wanted |= {(name, shape) for name in self.modems
                          if (shape := ADMITS[name].admits(d.bw_hz)) is not None}
        if d.kind == "track-end":
            self._grant()
            self._end(self.live.pop(d.track), RETIRE_S)

    @staticmethod
    def _admitted(tr: _Track) -> bool:
        return (_level(tr) >= ADMIT_DBFS
                and (tr.over_db is None or tr.over_db >= ADMIT_OVER_DB))

    def _end(self, tr: _Track, grace: float, *, defer=True) -> None:
        self._ended.add(tr.id)
        self.history.append(tr)
        self._retiring = [(t, m, done) for t, m, done in self._retiring
                          if t.is_alive() or not done[0]]
        if tr.wanted - tr.started.keys() - tr.covered:
            self.short.append(tr)
        held = {key: m for key, m in tr.started.items()
                if defer and key[0] == "lanner"}
        self._stop_all([m for key, m in tr.started.items() if key not in held], grace)
        if held:
            self._held[tr.id] = replace(tr, started=held, ended_at=tr.last)
        else:
            self._lanner_state.pop(tr.id, None)

    def _stop_all(self, monitors, grace: float) -> None:
        for m in monitors:
            done: list[bool] = []
            t = threading.Thread(target=self._retire, args=(m, grace, done), daemon=True)
            t.start()
            self._retiring.append((t, m, done))

    def _retire(self, m: ModemMonitor, grace: float, done: list[bool]) -> None:
        done.append(m.stop(grace))
        if m.dropped_samples:
            self._lost.append(m)
        self._since.pop(id(m), None)
        self._sent.pop(id(m), None)

    def _grant(self) -> None:
        """Loudest track first, a decoder at a time, while the cap has room.

        Asked between the tracker's reporting instants and never inside one, so
        what gets a slot does not turn on how the records of one instant were
        split across reads.
        """
        room = self.cap - len(self.running)
        for tr in sorted(self.live.values(), key=lambda t: -_level(t)):
            for want in sorted(tr.wanted - tr.started.keys() - tr.cut,
                               key=lambda want: (bool(self._held) and want[0] == "lanner", want)):
                # A lanner instance already searching this centre, live or held,
                # decodes the track; a second one would re-read its predecessor's
                # tail out of the new instance's preroll.
                if want[0] == "lanner" and any(
                        abs((old.home_hz or old.centre_hz) - tr.centre_hz)
                        <= ADMITS["lanner"].channel(old.bw_hz) / 2 - 85
                        for old in [*self.live.values(), *self._held.values()]
                        if old is not tr and any(key[0] == "lanner" for key in old.started)):
                    tr.covered.add(want)
                    continue
                if room <= 0:
                    room += self._preempt(_level(tr) - PREEMPT_DB)
                if room <= 0:
                    return
                tr.started[want] = self._spawn(*want, tr)
                room -= 1
                self.peak = max(self.peak, len(self.running))

    def _preempt(self, below: float) -> int:
        """Stop the decoders of the quietest track under ``below`` dBFS; how
        many that frees."""
        holding = [t for t in self.live.values() if t.started and _level(t) < below]
        if not holding:
            return 0
        victim = min(holding, key=_level)
        freed = victim.started
        victim.cut |= freed.keys()
        victim.started = {}
        self._stop_all(freed.values(), RETIRE_S)
        return len(freed)

    def _spawn(self, name: str, shape: str, tr: _Track) -> ModemMonitor:
        base = self.modems[name]
        if tr.home_hz is None or abs(tr.centre_hz - tr.home_hz) > GRID_HZ:
            tr.home_hz = round(tr.centre_hz / GRID_HZ) * GRID_HZ
        args = ("--centre", f"{tr.home_hz:g}",
                "--bw", f"{ADMITS[name].channel(tr.bw_hz):.0f}")
        if shape:
            name, args = f"{name}#{tr.id}/{shape}", args + ("--shape", shape)
        else:
            name = f"{name}#{tr.id}"
        spec = replace(base, name=name, args=base.args + args)
        m = ModemMonitor(spec, _Stamped(self, tr),
                         backlog_s=self.backlog_s + RING_S, lossy=self.lossy)
        m.start()
        self.started.append(spec)
        since = (tr.t_start - PREROLL_S) * FS
        self._since[id(m)] = max(0.0, since / FS)
        for t0, pcm in self._ring:
            if t0 + len(pcm) // SAMPLE_WIDTH > since and t0 < self._released:
                m.feed(t0, pcm)
                self._sent[id(m)] = t0 + len(pcm) // SAMPLE_WIDTH
        return m

    def stop(self, grace: float = 0.0) -> bool:
        """The tracker first, so every track it still holds ends and every
        instance it owes is started and fed; then every instance, in parallel,
        inside the same ``grace``."""
        clean = self.tracker.stop(grace)
        self._dispatch()
        self._grant()
        for tr in list(self.live.values()):
            self._end(self.live.pop(tr.id), grace, defer=False)
        if self._ring:
            t0, pcm = self._ring[-1]
            self._release((t0 + len(pcm) // SAMPLE_WIDTH) / FS, held_only=True)
        for tr in self._held.values():
            self._stop_all(tr.started.values(), grace)
        self._held.clear()
        for t, _, _ in self._retiring:
            t.join()
        return clean and all(done and done[0] for _, _, done in self._retiring)

    @property
    def dropped_samples(self) -> int:
        return sum(m.dropped_samples for m in self.monitors)

    def denied(self) -> list[str]:
        """Every track that ended short of a decoder its width asked for."""
        lines = []
        for tr in self.short:
            short = sorted({name for name, _ in tr.wanted - tr.started.keys() - tr.covered})
            if short:
                lo, hi = tr.centre_hz - tr.bw_hz / 2, tr.centre_hz + tr.bw_hz / 2
                lines.append(f"# track#{tr.id} {lo:.0f}-{hi:.0f} Hz {tr.label} "
                             f"{tr.level_dbfs:.0f} dBFS from {tr.t_start:.1f} s: no "
                             f"{', '.join(short)} -- " + (
                                 "stopped for a louder track" if tr.cut else
                                 f"{self.cap} decoders were already running")
                             + ", so nothing read it to its end")
        return lines


def same_frame(a: Detection, b: Detection) -> bool:
    """Match modem, protocol, kind and station within time and offset tolerances.

    Lanner NAVTEX copies share station/subject/serial, header times within
    three seconds, frequencies within 40 Hz, and matching body wording outside
    locally garbled regions.
    Other lanner text allows small decoding errors within three seconds and the usual offset
    tolerance. Both require different instances (via, track), preserving
    repeated lines from one decoder.
    """
    same = ((a.modem, a.protocol, a.kind, a.station)
            == (b.modem, b.protocol, b.kind, b.station))
    if a.modem == b.modem == "lanner" and (a.text is not None or b.text is not None):
        if not same or (a.via, a.track) == (b.via, b.track):
            return False
        if a.protocol == "NAVTEX":
            if (not a.station or not a.detail or a.detail != b.detail
                    or a.text is None or b.text is None or abs(a.t - b.t) > 3.0):
                return False
            # Station-Q's default dispatch differed by up to 24 Hz between
            # whole-band and per-track fits; 40 Hz leaves tuning margin while
            # preserving separate occurrences on different carriers.
            for left_hz, right_hz in ((a.rf_hz, b.rf_hz),
                                      (a.centre_hz, b.centre_hz),
                                      (a.offset_hz, b.offset_hz)):
                if left_hz is not None and right_hz is not None:
                    if abs(left_hz - right_hz) > 40.0:
                        return False
                    break
            # Whole-band and per-track paths acquire and fit word clocks in
            # different windows. Allow a few seconds of header dating error,
            # independent of the minutes spent receiving the message body.
            # Garbles can include lost symbols or shift codes, corrupting nearby
            # wording as well as the marked character. Require strong overall
            # agreement and confine differences to those damaged regions.
            left, right = " ".join(a.text.split()), " ".join(b.text.split())
            match = SequenceMatcher(None, left, right, autojunk=False)
            return (bool(left) and bool(right) and match.ratio() >= 0.9
                    and all(tag == "equal"
                            or "_" in left[max(0, i - 28):j + 28]
                            or "_" in right[max(0, k - 28):l + 28]
                            for tag, i, j, k, l in match.get_opcodes()))

        if a.protocol in ("RTTY", "SITOR-B"):
            if a.text is None or b.text is None or abs(a.t - b.t) > 3.0:
                return False
            # Disable the popular-character heuristic because radio text can
            # be highly repetitive.
            left, right = " ".join(a.text.split()), " ".join(b.text.split())
            return (a.detail == b.detail
                    and SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.9
                    and (a.offset_hz is None or b.offset_hz is None
                         or abs(a.offset_hz - b.offset_hz) <= OFFSET_HZ))
        same = a.text == b.text and a.detail == b.detail
    return (same and abs(a.t - b.t) < DEDUP_S
            and (a.offset_hz is None or b.offset_hz is None
                 or abs(a.offset_hz - b.offset_hz) <= OFFSET_HZ))


def _level(tr: _Track) -> float:
    return tr.level_dbfs if tr.level_dbfs is not None else -200.0
