"""One of this station's own arms, reheard through the monitor, our keyings labelled.

The input is the arm's tap: the rig's speaker feed, continuous and on one clock,
carrying both directions when the TX monitor was on. The session capture is
sliced into listen windows and blanked across our keyings, and the witness is
another receiver's view, so neither stands in for a missing tap.

Where we were keyed is read off the tap itself, by `txwitness`'s own-transmission
detector (the receiver's mute), and checked against the keyings `arm.log`
recorded. The audio inside those spans passes through untouched: our own signal
sometimes leaks through the mute, and a monitor that blanked the span could not
measure the leak. Instead each span becomes a `kind="ours"` record carrying what
is in it, and `spans` lets the console call anything raised inside one ours
rather than a peer's.

The dial is provenance's top-level `dial_hz`, never `readings`: those can name
another band (the WS8EOC-80 arm reads 3595000 at the top and 7100000 in its
readings).
"""
from __future__ import annotations

from . import operator

from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re

from hfhost.audio import FS, AudioError, AudioSource, WavReplaySource, Window
from hfmodem.core import wav
import numpy as np

from . import modes, rigsource, station

_spec = importlib.util.spec_from_file_location('txwitness', station.TXWITNESS)
txwitness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(txwitness)

# `onair.sh` sets PKTUSB on every arm it tunes.
ARM_MODE = 'PKTUSB'
STEP_S = txwitness.STEP_S
# The receiver's unmuted floor is read over this much audio before each key.
FLOOR_S = 0.2
# The tracker's grid, so an `ours` record's centre and width read like a track's.
NFFT, HOP = 4096, 960
OVER_REFERENCE_DB, EDGE_DB, BRIDGE_BINS = 6.0, 26.0, 2
# The reference bin is the span's lower quartile, not its median: a VARA data
# over fills 800-2200 Hz of a 300-2900 Hz window, more than half of it, so the
# median bin lies inside the emission and nothing stands over it.
REFERENCE_PERCENTILE = 25
# A logged keying agrees with a detected one when its middle lies inside it.
MATCH_SLACK_S = 0.25


@dataclass(frozen=True)
class Keyed:
    t0: float
    t1: float
    logged: int | None = None
    joins: tuple[int, ...] = ()     # later logged keyings the tap shows as this one mute


@dataclass(frozen=True)
class CrossCheck:
    """How the tap's keyings and the log's agree. `offset_s` places log time on the tap."""
    detected: int
    logged: int
    matched: int
    offset_s: float | None
    unlogged: tuple[Keyed, ...]
    unseen: tuple[int, ...]
    shared: int = 0

    @property
    def line(self):
        where = f', log clock {self.offset_s:+.3f} s onto the tap' if self.offset_s is not None else ''
        return (f'{self.detected} keyings on the tap, {self.logged} in arm.log, '
                f'{self.matched} of those found{where}'
                + (f'; {self.shared} of the tap\'s {"hold" if self.shared > 1 else "holds"} '
                   'logged keyings made back to back' if self.shared else '')
                + (f'; {len(self.unseen)} logged and not on the tap' if self.unseen else '')
                + (f'; {len(self.unlogged)} on the tap and not logged' if self.unlogged else ''))


def arm_roots():
    roots = [modes.resolve(name).arms for name in modes.KNOWN]
    roots += sorted((operator.data_root() / 'working').glob('*/arms'))
    return list(dict.fromkeys(Path(root).resolve() for root in roots))


def find(name, roots=None):
    """Find a legacy arm or resolve one exact registered capture root."""
    from hfmodem.core import capture_identity
    supplied = Path(name).expanduser()
    if supplied.is_absolute() or supplied.parent != Path('.'):
        if not supplied.is_dir():
            raise AudioError('Explicit arm directory is absent')
        found = [supplied.absolute()]
    else:
        found = [root / supplied.name for root in roots or arm_roots()
                 if (root / supplied.name).is_dir()]
    if not found:
        raise AudioError('No arm under the requested arms directories')
    try:
        return capture_identity.select_roots(found, preferred_role='native')
    except capture_identity.IdentityError as exc:
        raise AudioError('Arm identity is incomplete or ambiguous') from exc


_KEY = re.compile(r'key_request mono_s=([0-9.]+)\b.*\bon=([01])\b')
_PTT = re.compile(r'PTT keyed ([0-9.]+) s')
_ANCHOR = re.compile(r'master grid: anchor @ sample (\d+), cycle ([0-9.]+) s')
_SLOT = re.compile(r'\[grid\] (?:hold \d+ )?slot (\d+)(?: boundary @ sample (\d+))?')
_RF = re.compile(r'RF started (?:([-+][0-9.]+) ms from the slot boundary|with no slot)')


def logged_keyings(log_text):
    """`(start or None, duration)` per keying the arm logged, start in seconds.

    VARA and ARDOP write both key requests on the monotonic clock. Shrike writes
    how long each keying held and, on its capture's sample clock, the grid it
    keyed on: an anchor, a cycle, and each slot's boundary, with RF placed so
    many ms from it. The first call is keyed before the grid exists and becomes
    its anchor, so it is placed there. Either clock is foreign to the tap; the
    cross-check fits the one offset between them.
    """
    out, up = [], None
    anchor = cycle = slot = boundary = rf_ms = None
    unplaced = []
    for line in log_text.splitlines():
        timed = _KEY.search(line)
        if timed:
            at = float(timed[1])
            if timed[2] == '1':
                up = at
            elif up is not None:
                out.append((up, at - up))
                up = None
            continue
        grid = _ANCHOR.search(line)
        if grid:
            anchor, cycle = int(grid[1]), float(grid[2])
            for i in unplaced:
                out[i] = (anchor / FS, out[i][1])
            unplaced, slot, boundary = [], None, None
            continue
        placed = _SLOT.search(line)
        if placed:
            slot = int(placed[1])
            boundary = int(placed[2]) if placed[2] else None
            continue
        rf = _RF.search(line)
        if rf:
            rf_ms = float(rf[1]) if rf[1] else None
        held = _PTT.search(line)
        if not held:
            continue
        duration = float(held[1])
        if rf_ms is None or slot is None:
            unplaced.append(len(out))
            out.append((None, duration))
        else:
            at = boundary if boundary is not None else anchor + round(slot * cycle * FS)
            out.append((at / FS + rf_ms / 1e3, duration))
        rf_ms = slot = boundary = None
    return out


def detect(pcm):
    """Keyed spans on the tap, in seconds, off the receiver's mute."""
    a = np.frombuffer(pcm, dtype='<i2').astype(float) / 32768.0
    return [(u * STEP_S, d * STEP_S) for u, d in txwitness._runs(txwitness.muted(a))]


def cross_check(detected, logged):
    """Match the log's keyings to the tap's, and place the log's clock on the tap."""
    inside = lambda mid, span: span[0] - MATCH_SLACK_S <= mid <= span[1] + MATCH_SLACK_S
    timed = [(i, t, d) for i, (t, d) in enumerate(logged) if t is not None]
    pairs, offset = {}, None
    if timed:
        best, fit = {}, (0, 0.0)
        for span in detected:
            for _, t, _ in timed:
                shift = span[0] - t
                hits = {i: j for i, t, d in timed
                        for j, s in enumerate(detected) if inside(t + d / 2 + shift, s)}
                miss = sum(abs(detected[j][0] - (t + shift)) for i, t, _ in timed
                           if (j := hits.get(i)) is not None)
                if (len(hits), -miss) > fit:
                    best, offset, fit = hits, shift, (len(hits), -miss)
        pairs = dict(best)
    else:
        j = 0
        for i, (_, d) in enumerate(logged):
            while j < len(detected) and abs((detected[j][1] - detected[j][0]) - d) > MATCH_SLACK_S:
                j += 1
            if j < len(detected):
                pairs[i] = j
                j += 1
    claimed = set(pairs.values())
    shared = sum(1 for j in claimed if list(pairs.values()).count(j) > 1)
    return CrossCheck(shared=shared,
        detected=len(detected), logged=len(logged), matched=len(pairs), offset_s=offset,
        unlogged=tuple(Keyed(*span) for j, span in enumerate(detected) if j not in claimed),
        unseen=tuple(i for i in range(len(logged)) if i not in pairs)), pairs


def keyed_spans(detected, logged, check, pairs, duration_s):
    """The union of what the tap shows and what the log placed.

    Nothing is padded. The mute opens before our audio (the PTT lead) and closes
    after it (the rig's recovery), so a mute run already holds the emission, and
    padding would call the head of a peer answering at +0.1 s ours.
    """
    by_detection = {}
    for i, j in sorted(pairs.items()):
        by_detection.setdefault(j, []).append(i)
    spans = [Keyed(t0, t1, *_first_and_rest(by_detection.get(j, [])))
             for j, (t0, t1) in enumerate(detected)]
    if check.offset_s is not None:
        spans += [Keyed(t + check.offset_s, t + check.offset_s + d, i)
                  for i, (t, d) in enumerate(logged) if i in check.unseen and t is not None]
    merged = []
    for k in sorted(spans, key=lambda k: k.t0):
        lo, hi = max(0.0, k.t0), min(duration_s, k.t1)
        if merged and lo <= merged[-1].t1:
            last = merged[-1]
            indices = sorted(filter(lambda i: i is not None,
                                    (last.logged, *last.joins, k.logged, *k.joins)))
            merged[-1] = Keyed(last.t0, max(last.t1, hi), *_first_and_rest(indices))
        else:
            merged.append(Keyed(lo, hi, k.logged, k.joins))
    return merged


def _first_and_rest(indices):
    return (indices[0], tuple(indices[1:])) if indices else (None, ())


def _spectrum(a):
    """Mean power per bin, scaled so a sine of amplitude A sums to A squared.

    Half that sum is the mean square, which is how every level here is quoted:
    RMS dBFS, as `core.busy` reads it.
    """
    n = min(len(a), NFFT)
    win = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(a, n)[::HOP]
    power = (np.abs(np.fft.rfft(frames * win, n=NFFT, axis=1)) ** 2).mean(axis=0)
    return 4 * power / (NFFT * (win ** 2).sum())


def _dbfs(power):
    return float(10 * np.log10(max(float(power), 1e-20)))


def occupied(span, freqs, lo_hz, hi_hz):
    """``(centre_hz, bw_hz)`` of what stands over the span's own quiet bins, or Nones.

    Referred to the span and not to the receiver before the key: our monitor
    audio and a leak through the mute both sit under the unmuted floor, and it
    is their shape inside the muted window that says where they are. The width
    is the -26 dB width about the strongest bin that clears the reference by
    6 dB, gaps of two bins bridged; the centre is the power-weighted centroid
    inside it.
    """
    band = np.flatnonzero((freqs >= lo_hz) & (freqs <= hi_hz))
    reference = np.percentile(span[band], REFERENCE_PERCENTILE)
    over = band[span[band] > reference * 10 ** (OVER_REFERENCE_DB / 10)]
    if not over.size:
        return None, None
    peak = over[np.argmax(span[over])]
    strong = band[span[band] >= span[peak] * 10 ** (-EDGE_DB / 10)]
    runs = np.split(strong, np.flatnonzero(np.diff(strong) > BRIDGE_BINS + 1) + 1)
    run = next(r for r in runs if peak in r)
    lo, hi = run[0], run[-1]
    sel = slice(lo, hi + 1)
    centre = float((freqs[sel] * span[sel]).sum() / span[sel].sum())
    return centre, float(freqs[hi] - freqs[lo] + freqs[1])


@dataclass(frozen=True)
class Measured:
    level_dbfs: float
    floor_dbfs: float | None
    mute_db: float
    centre_hz: float | None
    bw_hz: float | None


def measure(a, keyed, lo_hz, hi_hz):
    """What each keyed span holds, set against the 200 ms of receiver before it."""
    freqs = np.fft.rfftfreq(NFFT, 1 / FS)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    hi = txwitness.out_of_band_level(a)
    out, previous_end = [], 0.0
    for k in keyed:
        i0, i1 = round(k.t0 * FS), round(k.t1 * FS)
        f0 = round(max(k.t0 - FLOOR_S, previous_end) * FS)
        span = _spectrum(a[i0:i1])
        floor = _spectrum(a[f0:i0]) if i0 - f0 >= FS // 100 else None
        steps = hi[int(k.t0 / STEP_S):max(int(k.t1 / STEP_S), int(k.t0 / STEP_S) + 1)]
        centre, width = occupied(span, freqs, lo_hz, hi_hz)
        out.append(Measured(
            level_dbfs=_dbfs(span[band].sum() / 2),
            floor_dbfs=_dbfs(floor[band].sum() / 2) if floor is not None else None,
            mute_db=float(-20 * np.log10(max(float(np.median(steps)), 1e-10))),
            centre_hz=centre, bw_hz=width))
        previous_end = k.t1
    return out


def tx_monitor(provenance):
    """What the rig read back about its TX monitor, at the arm's start and end."""
    said = []
    for when, key in (('start', 'readings'), ('end', 'readings_at_end')):
        rig = (provenance.get(key) or {}).get('rig') or {}
        on = (rig.get('funcs') or {}).get('MON')
        if on is not None:
            gain = (rig.get('levels') or {}).get('MONITOR_GAIN')
            said.append((when, on == '1', gain))
    return said


class ArmSource(AudioSource):
    """The arm's tap, untouched. `spans` and `records()` say where we were keyed."""

    def __init__(self, name, *, roots=None, realtime=False, station_path=None):
        self.directory = find(name, roots)
        self.name = self.directory.name
        self.tap = wav.stored(self.directory / 'tap.wav')
        if not self.tap.is_file():
            raise AudioError(
                f'{self.name} has no tap.wav, and nothing else will do: the session '
                'capture is sliced into listen windows and blanked across our keyings, '
                'and the witness is another receiver. Rehear this arm from its captures '
                'with --wav if that is what is wanted.')
        from hfmodem.core import capture_identity
        self.capture_identity = capture_identity.provenance(self.directory)
        self.tap_identity = capture_identity.validate_artifact(self.tap)
        self.provenance = json.loads((self.directory / 'provenance.json').read_text())
        self.dial_hz = self.provenance.get('dial_hz')
        cfg = rigsource.station_config(station_path)
        self.mycall = cfg.station.mycall if cfg else ''
        lo, hi = cfg.rig.filter_hz if cfg else rigsource.DEFAULT_FILTER_HZ
        self.window = Window.sideband(float(lo), float(hi), f'arm {self.name} tap',
                                      self.dial_hz, ARM_MODE)
        self.protocol = self.name.split('-')[0].upper()
        self.monitor = tx_monitor(self.provenance)
        self._replay = WavReplaySource(self.tap, realtime=realtime)
        pcm = b''.join(pcm for _, pcm in WavReplaySource(self.tap, realtime=False).frames())
        a = np.frombuffer(pcm, dtype='<i2').astype(float) / 32768.0
        log = self.directory / 'arm.log'
        self.logged = logged_keyings(log.read_text(errors='replace') if log.is_file() else '')
        detected = detect(pcm)
        self.check, pairs = cross_check(detected, self.logged)
        self.keyed = keyed_spans(detected, self.logged, self.check, pairs, len(a) / FS)
        self.spans = [(k.t0, k.t1) for k in self.keyed]
        self.measured = measure(a, self.keyed, lo, hi)

    def started(self):
        tap = self.directory / 'tap.json'
        if not tap.is_file():
            return None
        stamp = json.loads(tap.read_text()).get('started_utc')
        return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp() if stamp else None

    def _monitor_note(self):
        if not self.monitor:
            return 'TX monitor state not recorded'
        if all(on for _, on, _ in self.monitor):
            gains = sorted({g for _, _, g in self.monitor if g is not None})
            return ('TX monitor on' + (f' at {"/".join(gains)}' if gains else '')
                    + ', so the keyed window is our monitor audio, not a leak')
        return 'TX monitor ' + ', '.join(f'{"on" if on else "off"} at {when}'
                                         for when, on, _ in self.monitor)

    def _logged_as(self, k):
        if k.logged is None:
            return 'not in arm.log'
        if not k.joins:
            return f'arm.log keying {k.logged + 1}'
        numbers = [k.logged + 1] + [i + 1 for i in k.joins]
        said = f'arm.log keyings {", ".join(map(str, numbers[:-1]))} and {numbers[-1]}'
        (t0, d0), (t1, _) = self.logged[k.logged], self.logged[k.joins[-1]]
        if t0 is None or t1 is None:
            return f'{said}, one mute on the tap'
        gap_ms = 1e3 * (t1 - (t0 + d0))
        return f'{said}, {gap_ms:.0f} ms apart in the log and one mute on the tap'

    def records(self):
        start, note = self.started(), self._monitor_note()
        out = []
        for k, m in zip(self.keyed, self.measured):
            floor = (f'floor {m.floor_dbfs:.1f} dBFS in the {FLOOR_S * 1e3:.0f} ms before'
                     if m.floor_dbfs is not None else 'no unkeyed audio before it')
            out.append(dict(
                t=k.t0, duration_s=k.t1 - k.t0, modem='arm', protocol=self.protocol,
                kind='ours', grade='confirmed' if k.logged is not None else 'tentative',
                station=self.mycall, role='caller',
                detail=(f'keyed {k.t0:.2f}-{k.t1:.2f} s, {self._logged_as(k)}'
                        + f'; {floor}, mute {m.mute_db:.1f} dB deep; {note}'),
                wall=(start + k.t0) if start is not None else 0.0,
                centre_hz=m.centre_hz, bw_hz=m.bw_hz,
                rf_hz=self.window.rf_hz(m.centre_hz) if m.centre_hz is not None else None,
                offset_hz=None, level_dbfs=m.level_dbfs, cadence_s=None, track=None,
                text=None))
        return out

    def frames(self):
        return self._replay.frames()

    def close(self):
        self._replay.close()
