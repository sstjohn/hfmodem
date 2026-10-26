#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One command for a supervised on-air connect session.

Operator time on a shared band is the scarcest thing in this project, and the two
sessions that lost it were lost to improvisation at the radio: a frequency change
without re-sensing, a power command whose failure went unread, and a detector
argued with mid-session. Everything here is decided beforehand so the session is
execution only.

Per target, in order, stopping at the first connect:

    QSY to the dial  ->  verify it reads back  ->  verify the data mode reads back
         ->  verify PTT is down  ->  listen  ->  call, or skip and move on

Winlink publishes the signal CENTRE; the dial is 1500 Hz below it (:data:`DIAL_OFFSET_HZ`).
Both are printed everywhere a frequency appears, because at the radio they are two
different numbers and only one of them goes into the VFO.

Nothing is keyed without ``--arm``, and ``--arm`` is refused unless the rig
identifies as ``--expect-model`` and confirms the mode it was told to use — in a
voice mode an FT-891 routes the microphone rather than the USB codec, and the
attempt would key an unmodulated carrier for the length of every burst. The dial
and the mode are both re-sensed after every QSY, never carried over from the
previous frequency. A dry run does all of it except arm, which is the rehearsal
this wants before the slot opens.

Each attempt keys the serial line named by ``--ptt-device`` and leaves rigctld to
tune. On 2026-08-13 this station's daemon answered every command until the first
key-down and then answered nothing — twelve unkeys out of twelve, ~8.4 s of
unintended carrier apiece — while the line read low instantly, eight of eight.

The callsign goes out once, in Morse, by :mod:`hfmodem.core.cwid`, at the end of the
session and nowhere else — nothing in a connect attempt names this station, since
the connect-request burst is keyed to the *called* gateway's callsign and our own
call rides the link-setup, which only goes out after a gateway answers. It used to
be sent after every attempt, which is Morse in the middle of a run of connect
requests and, on a session that connected, Morse inside somebody's ARQ link: a
transmission out of turn that corrupts the exchange for the peer. So it is sent at
the end of a communication, it is not sent at all while a link is still open, and
the ten-minute identification reminders are printed rather than keyed. The operator
is at the control point and identifies by hand, as he has throughout.

Everything the session hears is kept: each target's listen window and each
attempt's receive audio as timestamped WAVs, each attempt's full output as a
transcript, all under ``logs/onair/<session>``. Only ``--no-record`` turns that
off. The one on-air session that could be diagnosed afterwards was the one that
happened to be recorded — it held two answers from a gateway this software walked
straight past — and the ones that were not are simply gone.

The channel sense can be overruled with ``--force``, which is what the operator's
own ears are for; a receiver that is deaf or being muted by another transmitter
cannot be overruled, because it is not a judgement about the band. ``--calls``
names the stations to work and the order to work them in, over the nearest-first
ranking.

    # rehearsal, transmits nothing
    python3 tools/onair_session.py --mycall MYCALL --band 40 --dry-run

    # the real thing
    python3 tools/onair_session.py --mycall MYCALL --band 40 \
        --calls KB9MMT --rigctld 127.0.0.1:4532 --expect-model FT-891 --arm
"""
from __future__ import annotations

import argparse
import atexit
import csv
import datetime as _dt
import math
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfcapture import GATEWAY_LIST, grid_to_latlon, device_claim, device_opened  # noqa: E402
from hfmodem.core.audio import play_drained  # noqa: E402
from hfmodem.core.busy import FULL_BAND, is_busy, level_range, receiver_fault, scores
from hfmodem.core.occupied import occupied_hz
from kestrel_connect import TX_IDLE_HOLD_S  # noqa: E402
from vara_rig_bridge import Rig  # noqa: E402
from hfmodem.core import band, cwid
from hfmodem.core import config, claim, coexist
from hfmodem.core.ptt import arming_refusal

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
FS = 48000
_BANDS = {"80": (3500, 4000), "40": (7000, 7300), "30": (10100, 10150),
          "20": (14000, 14350), "17": (18068, 18168)}

# Selected data segments, which are narrower than the bands above.
# The published channel can sit inside the band and still put the emission in a
# region outside the selected data segment, so the edges are checked against these.
_DATA_SEGMENTS = {"80": (3500, 3600), "40": (7000, 7125), "30": (10100, 10150),
                  "20": (14000, 14150), "17": (18068, 18110)}

# Winlink publishes the signal CENTRE frequency; on USB the dial is 1500 Hz below
# it. This tool used to send the published number straight to `F`, tuning every
# attempt 1500 Hz high — our bursts above where the gateway listens, its answer at
# 0 Hz audio under the SSB filter, dead in both directions. Measured, not assumed:
# the two connects that did work ran dials of 7102.000 and 7100.500 against
# published 7103.500 and 7102.000; a connect-response demodulated through
# `lock_preamble` spans 703-2273 Hz audio, putting the true centre 12 Hz from the
# published figure; and 191 KiwiSDR captures give a median audio centre of
# 1511.7 Hz. `tools/hfcapture.py` had the convention right all along.
DIAL_OFFSET_HZ = band.DIAL_OFFSET_HZ

# What a VARA over actually occupies in the audio passband, so the emission can be
# checked against a segment edge rather than the channel number.
_EMISSION_AUDIO_HZ = (750, 2300)

_CHILD_GRACE_S = 12.0       # for the child's own unkey handlers, before SIGKILL

RECORD_DIR = _ROOT / "logs" / "onair"

_child: subprocess.Popen | None = None      # the running attempt, for the signal path


def _miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 3958.8
    dlat, dlon = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def _open_now(hours: str, utc_hour: int) -> bool:
    hours = (hours or "").strip()
    if "-" not in hours:
        return True
    try:
        a, b = (int(v) for v in hours.split("-")[:2])
    except ValueError:
        return True
    return a <= utc_hour <= b if a <= b else (utc_hour >= a or utc_hour <= b)


def dial_hz(published_hz: int) -> int:
    """The USB dial for a published channel centre."""
    return published_hz - DIAL_OFFSET_HZ


def emission_khz(published_hz: int) -> tuple[float, float]:
    """The occupied band in kHz — the dial plus the audio the modem fills."""
    lo, hi = _EMISSION_AUDIO_HZ
    return (dial_hz(published_hz) + lo) / 1000, (dial_hz(published_hz) + hi) / 1000


def pick_targets(csv_path: Path, grid: str, band: str, limit: int,
                 max_miles: float,
                 only: list[str] | None = None) -> list[tuple[str, int, float, str]]:
    """Gateways on `band`, in service now, nearest first, by published centre.

    ``only`` names the stations to call, in the order to call them, and is how the
    operator overrules the ranking. Distance and published hours stop applying —
    both are proxies for "might hear us", and a station the operator has chosen has
    better evidence behind it than either. What still applies is what the band plan
    says: the frequency has to be on this band and our emission inside its data
    segment. KB9MMT is the reason this exists: it is the one station that has
    demonstrably heard this one and answered, and it ranks sixth by distance.

    Plain `VARA` sorts first. kestrel generates ONE MFSK tone set, which
    spec/04 §4.2.1 records as the BW2300 set and states differs for BW500 and
    BW2750. A plain-`VARA` gateway is therefore the only kind whose tone set we
    expect to match; 500 and 2750 stations are tried afterwards, and a connect to
    one of those would itself be a finding.

    A target whose *emission* would fall outside the band's data segment is
    dropped here rather than at the radio — the band table is the whole
    allocation, so a distant channel can be inside the band and still put our
    signal outside the selected data segment.
    """
    lo, hi = _BANDS[band]
    me = grid_to_latlon(grid) if grid else None
    utc_hour = _dt.datetime.now(_dt.UTC).hour
    want = [c.strip().upper() for c in (only or []) if c.strip()]
    rows: list[tuple[str, int, float, str]] = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                khz = float(r["Frequency"].replace(",", "").split()[0])
            except (KeyError, ValueError, IndexError):
                continue
            if not lo <= khz <= hi:
                continue
            if want and r["Callsign"].strip().upper() not in want:
                continue
            g = (r.get("GridSquare") or "").strip()
            if not want and (len(g) < 4 or not _open_now(r.get("Hours", ""), utc_hour)):
                continue
            d = _miles(me, grid_to_latlon(g)) if me and len(g) >= 4 else float("nan")
            if not want and d > max_miles:
                continue
            published = int(khz * 1000)
            lo_edge, hi_edge = emission_khz(published)
            seg_lo, seg_hi = _DATA_SEGMENTS[band]
            if not (seg_lo <= lo_edge and hi_edge <= seg_hi):
                print(f"  skipping {r['Callsign']} @ {khz:.3f}: emission "
                      f"{lo_edge:.1f}-{hi_edge:.1f} kHz is outside the {band} m data "
                      f"segment {seg_lo}-{seg_hi} kHz")
                continue
            rows.append((r["Callsign"], published, d, (r.get("Mode") or "").strip()))
    seen: set[str] = set()
    uniq = []
    def rank(t):
        if want:
            return (want.index(t[0].strip().upper()), 0.0)
        mode = t[3]
        return (0 if mode.strip().upper() == "VARA" else 1, 0.0 if math.isnan(t[2]) else t[2])

    for call, hz, d, mode in sorted(rows, key=rank):
        if call in seen:
            continue
        seen.add(call)
        uniq.append((call, hz, d, mode))
    return uniq[:limit]


class Cat:
    """The session's rigctld: drained reads for tuning, and :class:`Rig` for the key.

    The drained read matters and was learned the hard way — an undrained socket
    returns the previous command's reply, so a readback can silently describe
    something else — and it is bought with a fixed settling wait after every command.
    That wait is affordable for frequency and mode, which happen between
    transmissions. It is not affordable anywhere near PTT: ``key`` used to command
    the transmitter and then read it back through two of those waits, which put
    2.21 s of unmodulated carrier on a shared band before the first dit of every
    identification. Nothing about the drain discipline required that, so PTT is not
    on this socket at all.

    It keys because the session must be able to *stop* keying. The attempt runs as a
    child process which owns PTT while it lives; if it dies at the wrong moment —
    killed on timeout, say — its own unkey handlers die with it, and only the parent
    is left to drop the transmitter. That job already has an implementation with a
    bounded key-up, an unkey confirmed against the PTT line and repeated to a
    deadline, and an independent watchdog, so this borrows :class:`Rig` rather than
    keeping a second one in step with it. The second one had drifted: its unkey spent
    7.5 s on three attempts with no deadline at all, which is the shape of the bug
    that held this station's FT-891 on the air on 2026-07-28.
    """

    def __init__(self, hostport: str, wait: float = 1.1, armed: bool = False,
                 ptt_device: str | None = None) -> None:
        if armed and not ptt_device:
            raise ValueError("transmit unavailable: no keying line")
        import socket
        host, _, port = hostport.partition(":")
        if not port.isdigit():
            # A silent fallback picks which daemon you key through, so say which
            # one you mean. The station's own is 4532: `tools/onair.sh` starts
            # rigctld with no -t, which is Hamlib's default, and hands every modem
            # 127.0.0.1:4532. This tool once launched its own on 4533 and its
            # defaults still said so; 4533 is Hamlib's rotctld port.
            raise ValueError(f"--rigctld {hostport!r} has no port — give one explicitly "
                             "(the station's rigctld is 127.0.0.1:4532)")
        self._s = socket.create_connection((host, int(port)), timeout=8)
        self._s.setblocking(False)
        self._wait = wait
        try:
            # Rig.armed owns the line's lifetime; constructing it must not open
            # and hand back a line that the model/mode gate has yet to admit.
            self.rig = Rig(hostport, False, lambda m: print(f"  {m}", flush=True),
                           ptt_device=ptt_device, line_ptt=True)
            self.rig.armed = armed
        except BaseException:
            self._s.close()
            raise
        self._stop = threading.Event()

    # -- tuning, on this session's own connection ------------------------------
    def _drain(self) -> None:
        try:
            while self._s.recv(4096):
                pass
        except BlockingIOError:
            pass

    def cmd(self, text: str) -> str:
        self._drain()
        self._s.setblocking(True)
        self._s.sendall(text.encode())
        time.sleep(self._wait)
        self._s.setblocking(False)
        out = b""
        try:
            while chunk := self._s.recv(4096):
                out += chunk
        except BlockingIOError:
            pass
        return out.decode(errors="replace").strip()

    def qsy(self, hz: int) -> None:
        reply = self.cmd(f"F {hz}\n")
        if not reply.startswith("RPRT 0"):
            raise RuntimeError(f"QSY to {hz} refused: {reply!r}")
        time.sleep(0.4)
        back = self.cmd("f\n").strip()
        # Against the station's one tolerance rather than by string equality.
        # `!=` on the reply text is roughly twenty times stricter than the figure
        # every other transmit path here is held to, and stricter by accident: it
        # refuses a rig that reads back a hertz off, which this one measurably
        # does not, while saying nothing about how far off is too far.
        try:
            off = abs(int(back) - hz)
        except ValueError:
            raise RuntimeError(f"dial reads {back!r}, expected {hz}") from None
        if off > band.QSY_TOLERANCE_HZ:
            raise RuntimeError(f"dial reads {back}, expected {hz} "
                               f"({off} Hz off, over {band.QSY_TOLERANCE_HZ})")

    def model(self) -> str:
        """The rig's own name for itself — ``vara_rig_bridge.Rig.model``'s
        reading, not a second parser kept in step with it. Not ``identify()``:
        that also reads the dial, and the arm path has no use for a second
        round trip on the CAT bus."""
        return self.rig.model()

    def mode(self) -> tuple[str, int]:
        """``(mode, passband Hz)``. Bare ``m`` answers on two lines; the drained
        read takes both, so neither becomes the next command's reply."""
        lines = self.cmd("m\n").splitlines()
        got = lines[0].strip() if lines else ""
        pb = lines[1].strip() if len(lines) > 1 else ""
        return got, (int(pb) if pb.isdigit() else 0)

    def set_mode(self, mode: str, passband: int, min_passband: int) -> tuple[str, int]:
        """Put the rig in a data mode and prove it took.

        Nothing else in the session did this, so the emission went out in
        whatever mode the radio was last left in — and on an FT-891 a voice USB
        mode routes the microphone instead of the USB codec, keying an
        unmodulated carrier for the full length of every burst.

        The passband is asked for and then only required to be wide enough:
        a rig offers a discrete ladder of filters (the FT-891's widest PKTUSB is
        2.4 kHz) and answers with the one it actually selected, so demanding the
        requested number back would fail on a perfectly correct set. What has to
        be true is that the filter is not narrower than the emission.
        """
        reply = self.cmd(f"M {mode} {int(passband)}\n")
        if not reply.startswith("RPRT 0"):
            raise RuntimeError(f"mode {mode}/{passband} Hz refused: {reply!r}")
        time.sleep(0.4)
        got, got_pb = self.mode()
        if got != mode:
            raise RuntimeError(f"mode reads {got!r}, expected {mode!r}")
        if got_pb < min_passband:
            raise RuntimeError(f"passband reads {got_pb} Hz, narrower than the "
                               f"{min_passband} Hz emission")
        return got, got_pb

    # -- PTT, on the Rig's own per-command connections --------------------------
    @property
    def armed(self) -> bool:
        return self.rig.armed

    @armed.setter
    def armed(self, on: bool) -> None:
        self.rig.armed = on

    @property
    def line_held(self) -> bool:
        return self.rig.line_held

    def ptt_state(self) -> bool | None:
        """True keyed, False down, None without a confirmed line reading."""
        if not self.line_held:
            low = self.rig.released_low
            return None if low is None else not low
        return self.rig.line_state()

    def ptt_is_down(self) -> bool:
        return self.ptt_state() is False

    def key(self, on: bool, why: str = "") -> bool:
        if on and not self.armed:
            print(f"  (DISARMED — not keying for {why or 'transmission'})")
            return False
        return self.rig.key(True, why) if on else self.unkey(why)

    def unkey(self, why: str) -> bool:
        return self.line_held and self.rig.drop(why)

    def ensure_down(self, why: str) -> bool:
        """Read PTT back, and drop it if it is up. Never takes anyone's word."""
        state = self.ptt_state()
        if state is False:
            return True
        print(f"  PTT reads {'UP' if state else 'UNKNOWN'} ({why})")
        return self.unkey(why)

    def shutdown(self) -> None:
        """Idempotent: registered with atexit, and called from the signal path."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            if self.line_held:
                self.ensure_down("shutdown")
        finally:
            try:
                self.rig.shutdown()
            finally:
                self._s.close()


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


class Sense(NamedTuple):
    """What one listen window says — about the channel, and about the receiver.

    ``judged`` is the distinction that cost the last rehearsal: an occupancy check
    with too little live audio to score returns no verdict, which the bias resolves
    as busy with a margin of exactly +0.0 dB. On the screen that is the same line a
    genuinely occupied frequency prints, so a session with a dead input skipped
    every target reporting a busy band.
    """
    busy: bool
    margin: float
    loud: float
    quiet: float
    judged: bool


def listen(secs: float, device: str, record: Path | None = None,
           band: tuple[float, float] = FULL_BAND) -> Sense:
    with device_claim("onair_session listen", devices=claim.Devices(capture=device)):
        audio = sd.rec(int(FS * secs), samplerate=FS, channels=1, device=device,
                       dtype="float32")
        device_opened()
        try:
            sd.wait()
        finally:
            sd.stop()
        a = audio[:, 0]
        if record is not None:
            write_wav(record, a)
        busy, _per_sec, margin = is_busy(a, FS, band)
        loud, quiet = level_range(a)
        return Sense(busy, margin, loud, quiet, scores(a, FS, band) is not None)


def play(samples: np.ndarray, device: str) -> None:
    """Into the transmit device, filling its channel count — a mono stream is
    silent on a multi-channel codec, and a silent identification is no
    identification."""
    with device_claim("onair_session output", devices=claim.Devices(playback=device)):
        channels = int(sd.query_devices(device, "output")["max_output_channels"])
        block = np.zeros((len(samples), channels), np.float32)
        block[:, 0] = samples
        if channels > 1:
            block[:, 1] = block[:, 0]
        play_drained(block, FS, device)
        device_opened()


def stop_child() -> None:
    """SIGTERM first, so the child's own unkey handlers and key watchdog run.

    ``subprocess.run(timeout=)`` sends SIGKILL, the one signal a child cannot
    unkey through: its atexit/SIGINT/SIGTERM handlers and its watchdog thread all
    die with it, and if it was keyed at that instant the radio stays keyed.
    """
    p = _child
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=_CHILD_GRACE_S)
    except subprocess.TimeoutExpired:
        print("  attempt ignored SIGTERM — SIGKILL")
        p.kill()
        p.wait(timeout=5)


class NotAdmitted(claim.ClaimError):
    """The attempt ended, or ran out its admission window, without acknowledging."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__("CLAIM NOT ADMITTED")
        self._lines = lines       # complete once run_attempt has joined its reader

    @property
    def out(self) -> str:
        return "".join(self._lines)


def run_attempt(cmd: list[str], window: float, cat: Cat,
                transcript: Path | None = None, *,
                on_admitted: Callable[[], None] | None = None) -> tuple[int, str]:
    """Run the connect tool, and never let it take the transmitter with it.

    The parent drops PTT afterwards whatever happened, because it cannot know
    where in an over the child stopped — and because a `TimeoutExpired` escaping
    from here used to end the session with a traceback while the transmitter was
    still on the air. Its stderr is merged into its stdout: the caller reads one
    stream, and a child traceback is diagnosis rather than a captured string
    nobody looks at.

    Everything it says goes to ``transcript`` as it says it, because the session
    prints only the last ten lines and the rest — every burst logged, every
    preamble locked, every score that did not quite reach — used to be dropped on
    the floor. Line by line rather than at the end, so a Ctrl-C mid-attempt still
    leaves the file behind: the interrupt path exits from inside this call.
    """
    global _child
    lines: list[str] = []
    with device_claim("onair_session attempt") as held:
        fd = held.claim.fd
        trampoline = ("import os,sys; fd=int(sys.argv[1]); os.dup2(fd,9); "
                      "os.set_inheritable(9,True); "
                      "os.close(fd) if fd != 9 else None; os.execv(sys.argv[2],sys.argv[2:])")
        _child = subprocess.Popen([sys.executable, "-c", trampoline, str(fd), *cmd],
                                  pass_fds=(fd,), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            started = claim.process_started(_child.pid)
        except BaseException:
            stop_child()
            raise
        coexist.event("tool_spawn", child_pid=_child.pid, child_started=started, argv=cmd)
        admitted = threading.Event()
        process = _child
        fh = transcript.open("w") if transcript is not None else None

        def tee() -> None:
            for line in process.stdout:
                if line.strip() == "CLAIM ADMITTED":
                    admitted.set()
                lines.append(line)
                # An attempt's output is summarised when it ends, which is far too late
                # for an alarm: on 2026-07-28 the child shouted TRANSMITTER MAY BE STUCK
                # into this pipe for eight seconds while the terminal showed nothing and
                # the operator found it at the radio. Alarms are marked `***`, and they
                # go through now.
                if "***" in line:
                    print(f"    {line.rstrip()}", flush=True)
                if fh is not None:
                    fh.write(line)
                    fh.flush()

        reader = threading.Thread(target=tee, daemon=True)
        reader.start()
        try:
            start = time.monotonic()
            deadline = start + min(window, claim.ADMISSION_TIMEOUT)
            while not admitted.wait(0.05):
                if _child.poll() is not None or time.monotonic() >= deadline:
                    stop_child()
                    raise NotAdmitted(lines)
            device_opened()
            if on_admitted is not None:
                on_admitted()
            try:
                elapsed = time.monotonic() - start
                _child.wait(timeout=max(0.01, window - elapsed))
            except subprocess.TimeoutExpired:
                print(f"  attempt ran past {window:.0f}s — terminating "
                      f"({_CHILD_GRACE_S:.0f}s to unkey itself)", flush=True)
                stop_child()
            rc = _child.returncode
        finally:
            stop_child()
            _child = None
            try:
                cat.unkey("after attempt")
            finally:
                reader.join(timeout=5)
                if reader.is_alive():
                    raise claim.ClaimError("attempt output reader did not stop")
                process.stdout.close()
                if fh is not None:
                    fh.close()
        out = "".join(lines)
        if rc < 0:
            print(f"  attempt was killed by signal {-rc}")
        elif rc > 1:
            print(f"  attempt exited {rc} (it never got as far as calling)")
        return rc, out


def attempt_reached_air(out: str) -> bool:
    """Read off the child's transcript whether anything reached the air.

    ``originating:`` prints immediately before the first CR — before anything is
    keyed — so on a rig that refuses every key-up the whole attempt is silence
    and the marker still shows. The marker cannot answer this where it stands,
    and it is the connect tool's launch line rather than this session's to move,
    so the verdict is read the way the disconnect verdict already is: off the
    transcript. A burst logs ``keyed-by=`` only once the transport has confirmed
    it went out, and the first burst of any originate is one of those (the CR),
    so its absence alongside ``NOT TRANSMITTING`` is an attempt that never
    reached the air. Identification is owed for a transmission that happened, so
    uncertainty resolves toward having transmitted: with no refusal on record
    the launch marker keeps its old meaning.
    """
    if "keyed-by=" in out:
        return True
    if "NOT TRANSMITTING" in out:
        return False
    return "originating:" in out


def attempt_lost_transmitter(out: str) -> bool:
    """Whether the attempt ended because its rig retired, not because the gateway
    stayed silent — read the same way :func:`attempt_reached_air` is, off the
    transcript rather than off the exit code, which is 1 either way.

    ``kestrel_connect`` prints this exact wording — mid-attempt when it abandons
    the loop, and again in its final verdict — for one reason only: an unkey that
    never confirmed, which is a transmitter that may never key again.
    """
    return "the transmitter has been taken down" in out


def main() -> int:
    try:
        return session()
    except claim.ClaimError as exc:
        print(f"\nclaim problem: {exc}", flush=True)
        return 5 if isinstance(exc, claim.ClaimBusy) else 2


def session() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mycall", required=True, help="your callsign (you are the licensee)")
    ap.add_argument("--band", default="40", choices=sorted(_BANDS))
    ap.add_argument("--grid", default=None,
                    help="default: [station] grid; without one, gateways are not ranked by distance")
    ap.add_argument("--gateways", type=Path, default=GATEWAY_LIST)
    ap.add_argument("--rigctld", default="127.0.0.1:4532")
    ap.add_argument("--expect-model", default="FT-891")
    ap.add_argument("--device", default="USB Audio Device")
    ap.add_argument("--arm", action="store_true", help="permit keying; without it, nothing transmits")
    ap.add_argument("--ptt-device", metavar="DEV",
                    help="serial line whose RTS is the PTT. Required with --arm, "
                         "and passed to every attempt; CAT tunes only")
    # THE ATTEMPT KEYS THE LINE, AND THAT IS THE DEFAULT RATHER THAN A FLAG. On
    # 2026-08-13 every armed attempt at this station held ~8.4 s of unintended
    # carrier, one shape twelve times over: the socket connects in under a
    # millisecond, rigctld accepts `T 0`, and nothing comes back — while the
    # keying line, tried afterwards, read LOW instantly, eight of eight. The
    # daemon answers everything until it keys the rig, then answers nothing. So
    # the attempt drives --ptt-device itself and rigctld is left to tune, which
    # it does before there is any RF about; it is the shape onair.sh's vara and
    # vara-mail verbs hold.
    ap.add_argument("--dry-run", action="store_true", help="rehearse: sense and report, never call")
    ap.add_argument("--force", action="store_true",
                    help="call a channel the occupancy check reports busy — the "
                         "operator's ears outrank the detector")
    ap.add_argument("--calls", default="",
                    help="comma-separated callsigns to call, in this order, instead "
                         "of the nearest-first ranking (distance and published "
                         "hours stop applying; the band plan does not)")
    # Two targets rather than four: an attempt gets further with time than the
    # session gets with breadth, and the station that has answered us is one
    # station, not four.
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--max-miles", type=float, default=600.0)
    # An ARQ turnaround can be silent for longer than a short listen: the NS0A
    # recording holds a 12 s gap. A window that fits inside one hears nothing and
    # says clear, correctly on the evidence and wrongly about the channel.
    ap.add_argument("--listen", type=float, default=20.0)
    ap.add_argument("--per-target", type=float, default=150.0, help="seconds per attempt")
    ap.add_argument("--record", type=Path, default=RECORD_DIR,
                    help="directory for this session's audio and transcripts "
                         f"(default {RECORD_DIR}/<timestamp>)")
    ap.add_argument("--no-record", dest="record", action="store_const", const=None,
                    help="record nothing — a session nobody recorded cannot be "
                         "diagnosed afterwards")
    ap.add_argument("--bw", default="2300", choices=["500", "2300"])
    ap.add_argument("--mode", default="PKTUSB", help="data mode to set and verify")
    ap.add_argument("--passband", type=int, default=2400,
                    help="filter width to ask for, Hz (a rig picks its nearest)")
    ap.add_argument("--id-wpm", type=float, default=cwid.WPM)
    ap.add_argument("--id-tone", type=float, default=cwid.TONE_HZ,
                    help="Hz — keep it inside the data filter")
    args = ap.parse_args()

    if args.arm and (refusal := arming_refusal(args.ptt_device)):
        print(refusal)
        return 2

    try:
        id_audio = cwid.audio(args.mycall, wpm=args.id_wpm, tone_hz=args.id_tone)
    except ValueError as e:
        print(f"--mycall: {e}")
        return 2

    named = [c.strip().upper() for c in args.calls.split(",") if c.strip()]
    targets = pick_targets(args.gateways, args.grid if args.grid is not None else config.grid(), args.band,
                           len(named) or args.targets, args.max_miles, named)
    for call in named:
        if call not in {t[0].upper() for t in targets}:
            print(f"  {call} is not a {args.band} m gateway in {args.gateways.name} "
                  "(or its emission falls outside the data segment) — not calling it")
    if not targets:
        print("no gateways match — widen --max-miles, change --band, or check --calls")
        return 2
    print(f"{args.band} m, {len(targets)} target(s), "
          f"{'as named' if named else 'nearest first'} "
          f"(dial = published channel − {DIAL_OFFSET_HZ} Hz):")
    print(f"    {'call':11s} {'channel':>9}   {'dial USB':>9}     mi  mode")
    for call, hz, d, mode in targets:
        print(f"    {call:11s} {hz / 1000:9.3f}   {dial_hz(hz) / 1000:9.3f}  {d:5.0f}  {mode}")

    session_dir = None
    if args.record is not None:
        session_dir = Path(args.record) / _dt.datetime.now(_dt.UTC).strftime(
            "%Y%m%dT%H%M%SZ")
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nrecording this session to {session_dir} — the listen window and the "
              "receive audio of every attempt, plus each attempt's full output")
    else:
        print("\nNOT RECORDING (--no-record): nothing about tonight will be "
              "diagnosable afterwards")
    if args.force:
        print("--force IS IN EFFECT: a channel the occupancy check calls busy will "
              "be called anyway")

    with device_claim("onair_session", devices=claim.Devices(
            ptt_port=args.ptt_device, capture=args.device, playback=args.device, cat="rigctld")):
        try:
            cat = Cat(args.rigctld, ptt_device=args.ptt_device)
        except ValueError as e:
            print(f"\n{e}")
            return 2
        except OSError as e:
            print(f"\ncannot reach rigctld at {args.rigctld}: {e}")
            print("start it with:")
            print(f"  {_ROOT.parent}/hamlib/_install/bin/rigctld -m 1036 "
                  "-r /dev/cu.usbserial-XXXXB0 -s 38400 &")
            return 2
        try:
            device_opened()
            model = cat.model()
            print(f"\nrig: {model[:60]} (rigctld {args.rigctld})")
            if args.arm and args.expect_model.lower() not in model.lower():
                print(f"REFUSING TO ARM: rig identifies as {model!r}, not {args.expect_model!r}")
                return 2

            # The emission has to fit the filter the rig actually selected.
            min_passband = int(args.bw)
            try:
                got, got_pb = cat.set_mode(args.mode, args.passband, min_passband)
                print(f"mode: {got} / {got_pb} Hz")
            except RuntimeError as e:
                print(f"mode not confirmed: {e}")
                if args.arm:
                    print(f"REFUSING TO ARM: the rig will not confirm {args.mode} — in a voice "
                          "mode it transmits the microphone, not the codec")
                    return 2

            # Armed only once the rig has named itself and confirmed its mode.
            cat.armed = args.arm and not args.dry_run
            if not cat.armed:
                print("NOT ARMED — this run will not key the transmitter")

            owes_id = False
            link_open = ""        # the gateway still holding a session open, if any

            def identify(why: str) -> None:
                """MYCALL in Morse, keyed through the same checked, watchdogged path.

                Sent once, at the end of the session — see the module docstring for why it
                is not sent between attempts. It is a transmission like any other, so it
                obeys --arm; unarmed it says what it would have sent, which is what makes a
                rehearsal worth running. It is also never the reason a session stops: if it
                cannot key or the audio device refuses, it reports that and the session ends
                anyway — the operator is at the control point and sends the call by hand as
                before.
                """
                nonlocal owes_id
                if not cat.armed:
                    print(f"  (would identify: {args.mycall.upper()} in Morse, "
                          f"{len(id_audio) / FS:.1f}s — {why})")
                    owes_id = False
                    return
                print(f"  identifying {args.mycall.upper()} ({why})", flush=True)
                try:
                    if not cat.key(True, "station ID"):
                        print("  *** COULD NOT KEY TO IDENTIFY — send it by hand ***", flush=True)
                        return                              # still owed, and said so
                    play(id_audio, args.device)
                    # `play` returns only once the device has emitted the audio, so the
                    # callsign is whole before the key is even considered; this is the same
                    # idle hold a burst gets, and for the same reason. It used to be a
                    # quarter second covering the drain as well, because nothing else did:
                    # a dah at 20 WPM is 0.18 s, and an unkey on the old `sd.play` return
                    # sent W9SSJ as W9SSW. A callsign missing its last element identifies
                    # somebody else.
                    time.sleep(TX_IDLE_HOLD_S)
                    owes_id = False
                except Exception as e:                      # noqa: BLE001 - never fatal
                    print(f"  *** IDENTIFICATION FAILED ({e!r}) — send it by hand ***", flush=True)
                finally:
                    cat.unkey("station ID sent")

            def panic(*_) -> None:
                """An interrupt is an emergency stop, and it stops at the transmitter.

                A `finally:` does not survive a Ctrl-C mid-over, and an interrupt that
                left the attempt running would leave it keying with nobody watching. It
                deliberately does not identify on the way out: the operator reaching for
                Ctrl-C is not asking for one more transmission, and he is at the control
                point.
                """
                nonlocal owes_id
                stop_child()
                owes_id = False
                cat.shutdown()
                print("*** interrupted — send your call by hand if the attempt transmitted ***",
                      flush=True)
                sys.exit(1)

            atexit.register(cat.shutdown)
            for _sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(_sig, panic)

            result = 1

            def attempt_admitted() -> None:
                nonlocal owes_id
                owes_id = True

            try:
                for n, (call, hz, dist, mode) in enumerate(targets, 1):
                    dial = dial_hz(hz)
                    stem = f"{n:02d}-{call.upper()}" if session_dir else None
                    print(f"\n=== {call}  channel {hz / 1000:.3f} -> dial {dial / 1000:.3f} kHz USB "
                          f"({dist:.0f} mi, {mode}) ===", flush=True)
                    try:
                        cat.qsy(dial)
                        # Re-asserted after every QSY, like the dial: a Yaesu restores the
                        # mode last used on the band it is tuned to, so a mode confirmed on
                        # 40 m says nothing about 30 m.
                        cat.set_mode(args.mode, args.passband, min_passband)
                    except RuntimeError as e:
                        print(f"  {e} — skipping")
                        continue
                    if not cat.line_held:
                        print("  PTT UNKNOWN (no line held)")
                    elif not cat.ptt_is_down():
                        print("  PTT IS UP — dropping it and stopping the session")
                        cat.unkey("stuck PTT")
                        result = 3
                        break
                    try:
                        sense = listen(args.listen, args.device,
                                       session_dir / f"{stem}-listen.wav" if stem else None,
                                       occupied_hz("vara", args.bw))
                    except (sd.PortAudioError, OSError, ValueError) as e:
                        # The station runs a second modem against the same codec, so losing
                        # the device is a thing that happens rather than a thing that
                        # cannot. It costs this target, not the slot.
                        print(f"  cannot listen on {args.device!r} ({e}) — skipping")
                        continue
                    # Whether we can hear at all comes before what we heard. A receiver that
                    # is not attached scores as a busy channel on every target in turn, so
                    # the session that has this wrong reads as an unlucky band and spends the
                    # whole window on it. Neither fault is `--force`'s to overrule and the
                    # sentence itself says so; this end used to let --force past the
                    # intermittent one, which is a rig somebody else is keying.
                    fault = receiver_fault(sense.loud, sense.quiet, seconds=args.listen,
                                           device=args.device)
                    if fault is not None:
                        print(f"\n*** {fault.reason}\n    Stopping the session: no target "
                              "can work through this. ***", flush=True)
                        result = 5
                        break
                    if sense.judged:
                        verdict = (f"{sense.margin:+.1f} dB against the occupancy thresholds "
                                   f"-> {'BUSY' if sense.busy else 'clear'}")
                    else:
                        verdict = ("NO VERDICT — too little live audio in this window to "
                                   "score, which is not the same as a clear channel")
                    if sense.busy:
                        verdict += (", but --force is set — calling anyway" if args.force
                                    else ", skipping")
                    print(f"  channel sense: {verdict}", flush=True)
                    if sense.busy and not args.force:
                        continue
                    if not cat.armed:
                        print("  (would call here)")
                        owes_id = True
                        continue
                    cmd = [sys.executable, str(_HERE / "kestrel_connect.py"),
                           "--gateway", call, "--mycall", args.mycall, "--bw", args.bw,
                           "--tx-device", args.device, "--rx-device", args.device,
                           "--rigctld", args.rigctld, "--expect-model", args.expect_model,
                           "--arm", "--ptt-device", args.ptt_device,
                           "--line-ptt",
                           "--listen-first", "0",              # already sensed, above
                           "--timeout", str(args.per_target)]
                    cmd += ["--record", str(session_dir)] if session_dir else ["--no-record"]
                    if args.force:
                        cmd.append("--force")
                    already_owes_id = owes_id
                    try:
                        rc, out = run_attempt(cmd, args.per_target + 90, cat,
                                              session_dir / f"{stem}.log" if stem else None,
                                              on_admitted=attempt_admitted)
                    except NotAdmitted as exc:
                        for line in exc.out.strip().splitlines()[-30:]:
                            print("   ", line)
                        if attempt_reached_air(exc.out):
                            owes_id = True      # keyed before acknowledging: still a transmission
                            print(f"\n*** attempt claim problem: {exc}, after it transmitted. "
                                  "Stopping the session. ***", flush=True)
                        else:
                            print(f"\n*** the attempt never reached the air ({exc}). That is "
                                  "this software failing, not a gateway staying silent — the "
                                  "lines above are the diagnosis. Stopping with the slot "
                                  "intact. ***", flush=True)
                        result = 4
                        break
                    except claim.ClaimError as exc:
                        print(f"\n*** attempt claim problem: {exc}. "
                              "Stopping the session. ***", flush=True)
                        result = 4
                        break
                    # "The gateway stayed silent" and "nothing we did reached the air" are
                    # opposite conclusions, and only the first is worth carrying to the
                    # next target: the second — a child that never started, or a rig that
                    # refused every key-up — would repeat identically on every one of them
                    # and take the slot with it.
                    attempted = attempt_reached_air(out)
                    for line in out.strip().splitlines()[-(10 if attempted else 30):]:
                        print("   ", line)
                    if not attempted:
                        owes_id = already_owes_id       # this attempt added no transmission
                        print(f"\n*** the attempt never reached the air (exit {rc}). That is this "
                              "software failing, not a gateway staying silent — the lines above "
                              "are the diagnosis. Stopping with the slot intact. ***")
                        result = 4
                        break
                    if rc == 0 and "CONNECTED" in out:
                        print(f"\n*** CONNECTED to {call} ***")
                        # The connect tool drives the close and says which way it
                        # went; the session's duties differ with the answer — a closed
                        # link owes the identification, an open one forbids it. Read the
                        # child's verdict, never assume one. A 4.9.0 keys its own close
                        # once and nothing answers it (spec 05 §5.6, measured
                        # 2026-08-26), so an unanswered close is not by itself a peer
                        # that ignored us.
                        if "DISCONNECTED: clean close" in out:
                            print(f"    Local session closed; disconnect sent to {call}. "
                                  "Peer receipt is unconfirmed.")
                        else:
                            link_open = call
                            print(f"    Local close to {call} did not complete. "
                                  "Peer state is unknown; allow its inactivity timeout.")
                        result = 0
                        break
                    if attempt_lost_transmitter(out):
                        # Not a per-target condition: a rig that cannot key does not get
                        # better at the next channel, and every target called over it
                        # would log the same false silence in turn.
                        print(f"\n*** the transmitter has been taken down — not {call} "
                              "staying silent. Stopping the session with the remaining "
                              "target(s) uncalled. ***")
                        result = 6
                        break
                    print(f"  {call} did not answer")
            finally:
                if owes_id and link_open:
                    # Morse into a live ARQ session is a transmission out of turn: the peer
                    # is still holding the link and still expecting VARA overs on it.
                    print(f"\n  NOT identifying: {link_open} still holds the link open, and "
                          "keying the callsign into somebody's session corrupts it. Send "
                          f"{args.mycall.upper()} by hand.", flush=True)
                elif owes_id:
                    identify("end of session")
                cat.shutdown()

            print(f"\n{'connected' if result == 0 else 'no connect'}. "
                  f"PTT down: {cat.ptt_is_down()}")
            if session_dir is not None:
                print(f"recordings and transcripts: {session_dir}")
            return result
        finally:
            stop_child()
            cat.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
