#!/usr/bin/env python3
"""Receive-only VARA HF channel monitor — who is calling whom, and which gateways
are busy or idle.

It listens to receiver audio (a live rig audio input, or a recorded WAV),
segments every keyed burst, classifies it against the VARA waveforms kestrel
already decodes, and prints a live traffic log:

    [t] CR calling <gateway>            (initiator dialing that gateway)
    [t] connect-response to <call>      (a gateway answering)
    [t] link-setup from <caller>        (the caller's own callsign, blind-decoded)
    [t] connected-ack                   (the responder's step-5 burst)
    [t] data-ACK / NAK                  (short DBPSK control tokens)
    [t] DATA over (<n> payload bytes)

The CR and connect-response are keyed to the CALLED callsign, so they're named by
matching against the Winlink VARA gateway list (fetched once and cached). The
link-setup over carries the CALLER's callsign 6-bit-packed in the clear, so that
one is decoded outright, whoever it is. The connected-ack names nobody.

WHOSE TRANSMISSION IT WAS is not among the things this can answer. It has one
audio input and no knowledge of the station's own transmit schedule, so our own
calls come back through the receiver and are logged like anybody else's — on a
connect attempt against NS0A the eight `CR calling NS0A` lines in the log ARE the
eight connect requests this station sent. Nothing here should be read as a third
party unless a decoded callsign says so.

Those two named bursts are found by :class:`HandshakeScanner`, which reads them off
the raw stream rather than out of a bracketed keying — an energy gate cannot see a
burst that arrives through our own muted receiver, or one that sits level with the
band noise, and both happen on every session this has been measured against. The
gate still brackets everything else: wideband overs, session frames, control
tokens. Between bursts the channel line reports occupancy from
``core.busy.is_busy``, so traffic nobody here can decode still reads as busy.

    # live off the rig's receive audio (never transmits):
    python3 tools/vara_monitor.py --device "USB Audio Device"

    # offline decode of a recording:
    python3 tools/vara_monitor.py --wav capture.wav

    # scope the gateway candidate list to a region, or add extra known calls:
    python3 tools/vara_monitor.py --device "USB Audio Device" --grid EN54id
    python3 tools/vara_monitor.py --wav capture.wav --calls AAAA1,BBBB2

THIS TOOL NEVER TRANSMITS. It only ever opens an audio INPUT (or reads a WAV) —
no PTT, no rigctld, no audio output.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "packages" / "hfmodem"))
from hfmodem.core.busy import dominant, is_busy, scores
from hfcapture import (GATEWAY_LIST, distance_km, fetch_gateways, ffmpeg,
                       grid_to_latlon, input_spec)

from hfmodem.kestrel.monitor import (  # noqa: F401 -- read as vara_monitor.* by tools and suites
    FS, MONITOR_HELD_ENTER, SCANNED, SEG_FLOOR_Q, SEG_FRAME, SEG_HANG, SEG_MAX_S,
    SEG_MUTE_RATIO, SEG_PAD, HandshakeScanner, MonitorGate, Result, Segmenter,
    _COLLAPSE_MIN, _CTRL_CARRIERS, _TOKEN_COL_HI, _TOKEN_COL_LO, _dbpsk_collapse,
    _unnamed_burst, band_label, classify, occupied_band_hz, vara_gateways)

CHUNK = 4800                      # live read granularity (~0.1 s)
STATUS_EVERY = 10.0               # seconds of audio-time between status lines
IDLE_AFTER = 15.0                 # channel goes "idle" this long after last burst


# --------------------------------------------------------------------------- #
# Gateway candidate list (names the CALLED station in handshake bursts).
def load_gateways(cache: Path, grid: str | None, refresh: bool) -> list[str]:
    """Every HF VARA gateway callsign from Winlink (cached CSV), optionally the
    nearest to ``grid``. VARA FM is excluded — it's a different waveform."""
    if cache.exists() and not refresh:
        text = cache.read_text()
    else:
        print("Fetching the Winlink VARA gateway list…", file=sys.stderr)
        try:
            text = fetch_gateways()
        except Exception as exc:                       # offline is survivable
            print(f"  (gateway fetch failed: {exc} — continuing without it)",
                  file=sys.stderr)
            return []
        # A MISS WRITES TO THE ONE PATH, never to wherever it was pointed. This
        # line used to be `cache.write_text(text)`, and it is how the tree came to
        # hold three copies of this list: a caller with a relative `--cache` and a
        # working directory of its own missed, fetched, and left a new copy behind
        # in that directory. Reading from any path is fine and sometimes wanted --
        # an audit against a preserved snapshot -- but a fetch has exactly one
        # place to land, and the file it lands in is the one the guard reads.
        GATEWAY_LIST.parent.mkdir(parents=True, exist_ok=True)
        GATEWAY_LIST.write_text(text)

    home = grid_to_latlon(grid) if grid else None
    rows = []
    for call, sq in vara_gateways(text):
        km = 1e9
        if home:
            try:
                km = distance_km(home, grid_to_latlon(sq)) if len(sq) >= 4 else 1e9
            except (ValueError, IndexError):
                km = 1e9
        rows.append((km, call))
    rows.sort()
    return list(dict.fromkeys(call for _, call in rows))   # dedupe, nearest first



# --------------------------------------------------------------------------- #
# Rolling session view: which gateways have been heard, and channel activity.
def _dominant_score(audio: np.ndarray) -> str:
    """Which of core.busy's deciding scores sits furthest above its own threshold.

    ``is_busy`` returns that distance and not its name, and the name is what tells a
    reader what kind of occupant was found — see :meth:`Session._state`. Recomputing
    the scores costs one more pass over a 10 s window, against changing a return
    signature eight callers share, one of them the listen-before-transmit guard.
    A window too muted or too short for a verdict yields no name either.
    """
    s = scores(audio)
    return "no verdict" if s is None else dominant(s)[0]


@dataclass
class Session:
    seen: dict = field(default_factory=dict)     # call -> (last_t, role)
    last_burst_t: float = -1e9
    # (busy, margin dB, which score carried it), from core.busy
    occupancy: tuple[bool, float, str] | None = None
    next_status: float = 0.0
    # The transmit mute this stream measured, said once and then cleared; see
    # :class:`MuteFloor`. It is a fact about the receiver rather than about the
    # channel, so it goes out on a channel line and never as a bracket.
    mute_line: str | None = None

    def note(self, t: float, r: Result) -> None:
        self.last_burst_t = t
        if r.gateway:
            self.seen[r.gateway] = (t, "called")
        if r.caller:
            self.seen[r.caller] = (t, "caller")

    def _state(self, t: float) -> str:
        """Is anyone using the channel? From the occupancy score where one has been
        measured, because a monitor that reports only what it can name calls a
        channel idle while somebody's un-decodable traffic is all over it — which
        is exactly the 37 s of another station's narrowband session at the end of
        the 2026-07-26 recording. Falls back to burst recency for a caller that
        never sets ``occupancy`` (kestrel_connect, which is watching one QSO rather
        than the channel).

        The fallback says what it rests on, because it is the weaker of the two by
        a long way and read identically: ``last_burst_t`` is armed by EVERY
        classification, an unidentified one included, so a bare "ACTIVE" was a
        restatement of "the gate opened recently" in the vocabulary of a station
        being heard. A night of logs was read as evidence of activity on that
        basis. Now it prints the age of the burst it is standing on, and a reader
        can see there is no measurement behind it.

        The line names WHICH of core.busy's three scores carried the verdict,
        because without it the reader supplies their own occupant and is often
        wrong. On 2026-08-14 the slot's negative control at 6800 kHz read
        `BUSY (+2.9 dB)` to `(+4.3 dB)` on 21 of 21 status lines and was written up
        as firing on an empty channel; the channel in fact carried two steady
        carriers at RF 6799.633 and 6799.730 kHz, 30.4 and 25.1 dB over the local
        spectral baseline, and `tone` — the carrier detector — was what found them,
        at 7.64 to 9.57 against its 5.0 threshold, while `burst` stayed at 0.49-0.99
        and `shape` at 3.03-4.61. `BUSY (tone +4.3 dB)` reads as "there is a
        carrier"; a bare `BUSY (+4.3 dB)` reads as "somebody is working this
        channel", which is what a modem monitor's reader is primed to expect and is
        the one thing tone cannot tell them.
        """
        if self.occupancy is not None:
            busy, margin, which = self.occupancy
            sign = "+" if busy else ""
            return f"{'BUSY' if busy else 'clear'} ({which} {sign}{margin:.1f} dB)"
        if self.last_burst_t < -1e8:
            return "idle (nothing bracketed yet)"
        age = t - self.last_burst_t
        if age >= IDLE_AFTER:
            return f"idle (last burst {age:.0f}s ago)"
        return f"ACTIVE (a burst {age:.0f}s ago, not an occupancy measurement)"

    def maybe_status(self, t: float) -> str | None:
        if self.mute_line is not None:
            line, self.mute_line = self.mute_line, None
            return line
        if t < self.next_status:
            return None
        self.next_status = t + STATUS_EVERY
        state = self._state(t)
        if not self.seen:
            return f"  -- channel {state}; no stations identified yet --"
        parts = []
        for call, (lt, role) in sorted(self.seen.items(), key=lambda kv: -kv[1][0]):
            age = t - lt
            tag = "busy" if age < IDLE_AFTER else f"{age:.0f}s ago"
            parts.append(f"{call}({role}, {tag})")
        return f"  -- channel {state}: " + ", ".join(parts) + " --"


# --------------------------------------------------------------------------- #
def emit(t: float, r: Result, session: Session) -> None:
    session.note(t, r)
    q = f"  ({r.quality})" if r.quality else ""
    print(f"[{t:8.2f}] {r.kind:<16} {r.info}{q}".rstrip(), flush=True)
    line = session.maybe_status(t)
    if line:
        print(line, flush=True)


# Audio kept for the occupancy score. core.busy wants a window it can find both
# a duty cycle and a spectrum in, and refuses one holding less than 4 s of live
# audio; 10 s survives a receiver muted for half of it.
BUSY_WINDOW_S = 10.0


def monitor(source, calls: list[str], session: Session,
            decode_wideband: bool = True):
    """Yield ``(t, Result)`` for everything heard on SOURCE, and ``(t, None)`` each
    time the clock passes with nothing to report — which is when SESSION's channel
    line is due. Two detectors run over the same audio: the gate brackets keyed
    regions for :func:`classify`, and :class:`HandshakeScanner` reads the handshake
    bursts straight off the stream, most of which no gate would ever bracket."""
    seg, scan = MonitorGate(), HandshakeScanner(calls)
    window: deque[np.ndarray] = deque()
    held = 0
    said_mute = False
    for t0_samples, chunk in source:
        if not said_mute and (depth := seg.mute.depth) is not None:
            said_mute = True
            session.mute_line = (
                f"  -- mute {depth:.1f} dB below the band on this tape --")
        events = [(start, classify(burst, calls, decode_wideband,
                                   level_db=seg.level_db(burst),
                                   reference=seg.reference()))
                  for start, burst in seg.push(chunk)]
        events = [e for e in events if e[1].kind not in SCANNED]
        events += scan.push(chunk)
        yield from ((start / FS, r) for start, r in sorted(events, key=lambda e: e[0]))
        window.append(chunk)
        held += len(chunk)
        while held - len(window[0]) >= BUSY_WINDOW_S * FS:
            held -= len(window.popleft())
        t = (t0_samples + len(chunk)) / FS
        if t >= session.next_status and held >= BUSY_WINDOW_S * FS:
            audio = np.concatenate(window)
            busy, _, margin = is_busy(audio)
            session.occupancy = (busy, margin, _dominant_score(audio))
        yield t, None
    for start, burst in seg.flush():
        r = classify(burst, calls, decode_wideband, level_db=seg.level_db(burst),
                     reference=seg.reference())
        if r.kind not in SCANNED:
            yield start / FS, r


def run(source, calls: list[str], decode_wideband: bool = True) -> None:
    session = Session()
    for t, r in monitor(source, calls, session, decode_wideband):
        if r is not None:
            emit(t, r, session)
        elif (line := session.maybe_status(t)) is not None:
            print(line, flush=True)


# --------------------------------------------------------------------------- #
# Sources: a WAV file, or a live audio input via ffmpeg (INPUT only — no PTT,
# no output stream).
def wav_source(path: str):
    from scipy.io import wavfile

    from hfmodem.core import wav
    _, a = wavfile.read(wav.source(path))
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a[:, 0]
    a = a / (np.abs(a).max() or 1.0)                    # normalise like RX.decode_wav
    for i in range(0, len(a), CHUNK):
        yield i, a[i:i + CHUNK]


def device_source(device: str):
    """Read live audio from an INPUT device via ffmpeg (s16le mono @48k). This
    opens a capture only — it never keys PTT or plays audio."""
    proc = subprocess.Popen(
        [ffmpeg(), "-hide_banner", "-loglevel", "error",
         *input_spec(device), "-ac", "1", "-ar", str(FS),
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print(f"Listening on {device!r} (receive-only). Ctrl-C to stop.\n", flush=True)
    consumed = 0
    try:
        while True:
            raw = proc.stdout.read(CHUNK * 2)
            if not raw:
                break
            a = np.frombuffer(raw, "<i2").astype(float) / 32768.0
            yield consumed, a
            consumed += len(a)
    finally:
        proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--device", help="audio INPUT device carrying receiver audio")
    src.add_argument("--wav", help="decode a recording instead of a live device")
    ap.add_argument("--grid", help="scope the gateway list to nearest this grid")
    ap.add_argument("--calls", help="extra candidate callsigns (comma-separated)")
    ap.add_argument("--cache", type=Path, default=GATEWAY_LIST)
    ap.add_argument("--refresh", action="store_true", help="re-fetch the gateway list")
    # The wideband decode used to be skipped on a live device, on the belief that
    # turbo-decoding an over could not keep up with one. Measured, the whole 150 s
    # 2026-07-26 recording — eight of our overs, two gateway answers and another
    # station's session — processes end to end in 4.1 s, 37x faster than real time,
    # and the two off-air gateway sessions in 61 s and 68 s of 109 s and 108 s. So it
    # is on everywhere now, and the flag is there for a channel busy enough, or a
    # machine slow enough, to make an operator want the log cheap.
    ap.add_argument("--no-wideband", action="store_true",
                    help="skip the wideband decode (no caller callsign, no byte counts)")
    args = ap.parse_args()

    calls = load_gateways(args.cache, args.grid, args.refresh)
    if args.calls:
        extra = [c.strip().upper() for c in args.calls.split(",") if c.strip()]
        calls = extra + [c for c in calls if c not in extra]
    print(f"{len(calls)} candidate callsign(s) for handshake attribution.\n",
          file=sys.stderr)

    source = wav_source(args.wav) if args.wav else device_source(args.device)
    try:
        run(source, calls, decode_wideband=not args.no_wideband)
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
