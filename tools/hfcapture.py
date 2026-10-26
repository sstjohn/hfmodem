#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Field capture kit for kestrel: record a VARA HF session off the rig.

    hfcapture.py devices
    hfcapture.py check --device 1
    hfcapture.py record --device 1 --peer W1AW
    hfcapture.py proxy                    # optional, see README
    hfcapture.py package cap-20260721-1432

Capture goes through PortAudio, which is what every other audio leg at this
station opens a card with; `devices` still shells out to ffmpeg to list the
host's inputs. Rig control additionally needs rigctld, and borrows this tree's
one rigctld client from vara_rig_bridge.
"""

from __future__ import annotations

import argparse
import array
from contextlib import ExitStack, contextmanager
import json
import math
import os
import platform
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from hfmodem.core import band, claim
from hfmodem.core.native import NativeChild, spawn_native

RATE = 48000
SEGMENT_SECONDS = 600
SEGMENT_FRAMES = SEGMENT_SECONDS * RATE

# Why PortAudio and not ffmpeg, measured 2026-08-14 on this station's own codec
# and on a virtual device with no hardware in it at all: ffmpeg's avfoundation
# input delivered 0.904 +/- 0.005 of real time in samples, unchanged by
# -drop_late_frames, -thread_queue_size, -c:a copy or the choice of card, while
# stamping the buffers that did arrive with true host time. Its own accounting
# says so -- 2507 packets demuxed across a 29.80 s span is 26.74 s of audio -- so
# `-segment_time 600` faithfully cut a segment holding 545 s, and nothing was
# logged at -loglevel error. Nine windows on disk are short by 9% in the middle
# because of it. PortAudio on the same card delivers 0.9979 over 120 s with zero
# overflows -- which is better than ffmpeg by two orders of magnitude and is not
# a clean bill: the overflow count is the driver's own, and a starved interpreter
# drops blocks that never reach it, so 0.9979 bounds the loss on that bench run
# and does not attribute it. `tools/clock_shortfall.py` reads the same shortfall
# off the session record, where it runs to 47.7 s of one 86.8 s arm.

# A step past this is a lost buffer rather than clock jitter: PortAudio's ADC
# timestamps tracked the sample count to 0.1 ms over 20 s here, with no
# single-buffer step past 3 us, and one 512-frame buffer is 10.7 ms.
GAP_S = 0.001

# What the wall clock may exceed the audio by before the run is not a timeline.
# Stream-open latency measured 0.19-0.28 s across 20, 40 and 120 s runs and does
# not grow, so this is a wide margin on a fixed cost -- and 9% of a 900 s window
# is 81 s, which it is meant to be impossible to sit under.
SLIP_S = 1.0

# The X6100 speaks Icom CI-V; hamlib drives it as an IC-705 (3085) at 19200 baud
# — verified live on this station 2026-07-22. Its own native backend (3087) exists
# but the IC-705 path is the one confirmed to key PTT over CAT. g90/ft891 use the
# hamlib backend default baud unless --rig-baud overrides.
RIG_MODELS = {"g90": 3088, "x6100": 3085, "ft891": 1036}
RIG_BAUD = {"x6100": 19200}
RIGCTLD = ("127.0.0.1", 4532)
MIN_PASSBAND = 2700

DEVICE_HELP = "capture input: a name as `devices` prints it, or its index there"

# Winlink publishes the signal CENTRE frequency; on USB the dial you tune to is
# 1500 Hz below it (VARA's audio passband is centred on 1500 Hz). Tune the dial
# to the centre by mistake and the whole signal lands near 0 Hz audio, outside
# the passband — a silently dead capture.
DIAL_OFFSET_HZ = band.DIAL_OFFSET_HZ


_tool_claims = threading.local()


def inherited_claim():
    # Only a launcher that exported RIG_CLAIMED passed the claim; any other fd 9
    # is whatever the parent left open, never a claim to verify.
    if not os.environ.get("RIG_CLAIMED"):
        return None
    try:
        os.fstat(9)
    except OSError as exc:
        raise claim.ClaimError("missing inherited claim descriptor") from exc
    return claim.inherited(9)


@contextmanager
def device_claim(operation, *, devices=None):
    stack = getattr(_tool_claims, "stack", [])
    source = stack[-1].claim if stack else None
    acquired = False
    if source is None:
        source = inherited_claim()
    if source is None:
        source = claim.acquire("tool", "-".join(operation.split()), devices=devices,
                               prove_line=True,
                               ptt_factory=claim.line_factory(devices.ptt_port) if devices else None)
        acquired = True
    descriptor = source if acquired else claim.ClaimDescriptor(os.dup(source.fd))
    admission = None
    complete = False
    try:
        admission = claim.inherited(descriptor, launch=True)
        _tool_claims.stack = [*stack, admission]
        yield admission
        complete = True
    finally:
        _tool_claims.stack = stack
        if admission is not None:
            admission.failed()
        if acquired and complete:
            claim.release(descriptor)
        else:
            claim.close(descriptor)


def device_opened():
    stack = getattr(_tool_claims, "stack", [])
    for admission in reversed(stack):
        if admission.fd is not None:
            if admission is stack[0]:
                admission.opened()
            else:
                # Each open closes its admission barrier; only the outer scope
                # acknowledges this process to its parent.
                admission.failed()


def stop_native(child, timeout=3.0):
    child.terminate()
    if not child.wait_gone(timeout):
        child.kill()
        if not child.wait_gone(5.0):
            raise claim.ClaimError(f"termination unresolved for pid {child.pid}")


def close_native(child, *, stopped=False):
    if not stopped:
        stop_native(child)
    for stream in (child.stdin, child.stdout, child.stderr):
        if stream is not None:
            stream.close()


def native_command(argv, *, input=None, timeout=4):
    if not 0 < timeout <= claim.ADMISSION_TIMEOUT:
        raise ValueError("native timeout must be in (0, 30] seconds")
    with ExitStack() as stack:
        try:
            held = stack.enter_context(device_claim("CAT", devices=claim.Devices(
                cat_port=argv[argv.index("-r") + 1] if "-r" in argv else None,
                cat="rigctld" if "-m" in argv and argv[argv.index("-m") + 1] == "2" else "serial")))
        except OSError as exc:
            raise claim.ClaimError(f"CLAIM REFUSED: CAT admission: {exc}") from exc
        fd = held.claim.fd
        result = subprocess.run([sys.executable, "-m", "hfmodem.core.claim", "spawn",
                                 "--fd", str(fd), "--timeout", str(timeout), "--", *argv],
                                input=input, capture_output=True, text=True, pass_fds=(fd,))
        # claim spawn and rigctl share exit codes 1/2. The diagnostic, rather
        # than the code, distinguishes a refusal from a radio command failure.
        if "CLAIM REFUSED" in result.stderr:
            raise claim.ClaimError(result.stderr.strip())
        return result


def drain_native(child, log, *, stdout=True):
    child.stdin.close()
    threads = []
    def drain(stream):
        for block in iter(lambda: stream.read(4096), b""):
            log.write(block.decode(errors="replace"))
            log.flush()
    for stream in ((child.stdout, child.stderr) if stdout else (child.stderr,)):
        thread = threading.Thread(target=drain, args=(stream,), daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def die(msg: str) -> None:
    print(f"\n!! {msg}\n", file=sys.stderr)
    sys.exit(1)


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        die("ffmpeg not found. macOS: brew install ffmpeg. "
            "Windows: winget install Gyan.FFmpeg. Linux: apt install ffmpeg.")
    return exe


def input_spec(device: str) -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return ["-f", "avfoundation", "-i", f":{device}"]
    if system == "Windows":
        return ["-f", "dshow", "-i", f"audio={device}"]
    return ["-f", "pulse", "-i", device]


# --- devices ---------------------------------------------------------------

def avfoundation_inputs() -> list[tuple[str, str]]:
    """``(index, name)`` for every macOS capture input, as ffmpeg numbers them."""
    out = subprocess.run(
        [ffmpeg(), "-hide_banner", "-f", "avfoundation", "-list_devices", "true",
         "-i", ""], capture_output=True, text=True)
    listing = ((out.stderr or "") + (out.stdout or "")).split(
        "AVFoundation audio devices:")[-1]
    found = []
    for line in listing.splitlines():
        m = re.search(r"\[(\d+)\]\s+(.*)", line)
        if m:
            found.append((m.group(1), m.group(2).strip()))
    return found


def cmd_devices(args: argparse.Namespace) -> None:
    system = platform.system()
    print("Audio inputs on this machine:\n")
    if system == "Darwin":
        for index, name in avfoundation_inputs():
            print(f"  {index:>3}  {name}")
    else:
        probe = ([ffmpeg(), "-hide_banner", "-f", "dshow", "-list_devices",
                  "true", "-i", "dummy"] if system == "Windows"
                 else ["pactl", "list", "short", "sources"])
        out = subprocess.run(probe, capture_output=True, text=True)
        listing = (out.stderr or "") + (out.stdout or "")
        print("\n".join("  " + ln for ln in listing.splitlines() if ln.strip()))
    print("\nPick the one carrying receiver audio (the rig interface — "
          "SignaLink, IC-7300, Digirig, BlackHole…), not the built-in mic.")
    print("Then:  hfcapture.py check --device <n>")


def portaudio():
    try:
        import sounddevice
    except ImportError:
        die("sounddevice not installed — `pip install sounddevice`, which is the "
            "`radio` extra, and is what every audio leg at this station uses")
    return sounddevice


def sound_input(sd, device: str) -> tuple[int, str]:
    """The capture input ``--device`` names, as ``(PortAudio index, card name)``.

    A name matches as a substring, which is how the rest of the station addresses
    this codec — `onair.sh` says "USB Audio" for a card that enumerates as "USB
    Audio Device". A bare index keeps meaning the one `devices` prints, resolved
    through that listing first, because PortAudio orders the same cards
    differently and `rx_levels.py` sends operators to `hfcapture.py devices` for
    the number. Two matches is an error rather than a pick: guessing between two
    inputs is how a session records the wrong one.
    """
    name = device
    if device.isdigit() and platform.system() == "Darwin":
        name = next((n for i, n in avfoundation_inputs() if i == device), device)
    hits = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and name.lower() in d["name"].lower()]
    if len(hits) > 1:
        die(f"{device!r} matches more than one audio input: "
            + ", ".join(repr(n) for _, n in hits))
    if not hits:
        die(f"no audio input matching {device!r} — run `hfcapture.py devices`")
    return hits[0]


# --- level check -----------------------------------------------------------

def repair_wav(path: Path) -> float:
    """Patch RIFF/data sizes from the file on disk and return duration.

    A capture killed mid-write leaves placeholder sizes in the header, which
    makes the file undecodable by anything that trusts them.
    """
    size = path.stat().st_size
    with open(path, "r+b") as fh:
        if fh.read(4) != b"RIFF":
            return 0.0
        fh.seek(12)
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return 0.0
            chunk, length = head[:4], int.from_bytes(head[4:], "little")
            if chunk == b"data":
                break
            fh.seek(length + (length & 1), os.SEEK_CUR)
        data_at = fh.tell()
        data_bytes = size - data_at
        if length != data_bytes:
            fh.seek(data_at - 4)
            fh.write(data_bytes.to_bytes(4, "little"))
            fh.seek(4)
            fh.write((size - 8).to_bytes(4, "little"))
    return data_bytes / (RATE * 2)


def levels(samples: array.array) -> tuple[float, float, int]:
    if not samples:
        return -999.0, -999.0, 0
    peak = max(abs(s) for s in samples)
    energy = sum(float(s) * s for s in samples) / len(samples)
    clipped = sum(1 for s in samples if abs(s) >= 32700)
    to_db = lambda v: 20 * math.log10(v / 32768.0) if v > 0 else -999.0
    return to_db(peak), to_db(energy ** 0.5), clipped


def measure(path: Path) -> tuple[float, float, int]:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        width = w.getsampwidth()
    if width != 2:
        die(f"expected 16-bit wav, got {width * 8}-bit")
    samples = array.array("h")
    samples.frombytes(frames)
    return levels(samples)


def cmd_check(args: argparse.Namespace) -> None:
    with device_claim("hfcapture check", devices=claim.Devices(capture=args.device)):
        sd = portaudio()
        index, name = sound_input(sd, args.device)
        print(f"Recording {args.seconds}s from [{index}] {name} — "
              "have the receiver hearing the band you'll work.")
        heard = sd.rec(int(args.seconds * RATE), samplerate=RATE, channels=1,
                       device=index, dtype="int16", blocking=False)
        device_opened()
        try:
            sd.wait()
        finally:
            sd.stop()
        samples = array.array("h")
        samples.frombytes(heard.tobytes())
        peak, rms, clipped = levels(samples)
        print(f"\n  peak {peak:6.1f} dBFS")
        print(f"  rms  {rms:6.1f} dBFS")
        print(f"  clipped samples: {clipped}")

        if clipped or peak > -1.0:
            print("\n  TOO HOT — clipping. Turn the soundcard input level down.")
        elif peak < -30.0:
            print("\n  TOO QUIET — turn the input level up. Aim for peaks near -10 dBFS.")
        else:
            print("\n  Good. Aim for peaks around -10 dBFS on a strong signal.")


# --- rig (via rigctld only) ------------------------------------------------
#
# The client itself is vara_rig_bridge.Rig — one rigctld client in this tree.
# Capture opens it disarmed: it reads and tunes, and cannot key the transmitter.


def start_rigctld(model: int, device: str, baud: int | None = None, *, claim_fd, log, children) -> NativeChild:
    exe = shutil.which("rigctld")
    if exe is None:
        # The launchers' contract (tools/lib/ports.sh): HAMLIB_BIN names the bin
        # directory of a hamlib built from source, defaulting to the station's
        # own build. One answer to "where is hamlib", not a guess per file.
        candidate = Path(os.environ.get(
            "HAMLIB_BIN", str(Path.home() / "src/radio/hamlib/_install/bin")
        )).expanduser() / "rigctld"
        if os.access(candidate, os.X_OK):
            exe = str(candidate)
        else:
            die("rigctld not found — install hamlib, export HAMLIB_BIN=<bin dir of "
                "your build>, or start it yourself and pass --rigctld host:port")
    cmd = [exe, "-m", str(model), "-r", device, "-t", str(RIGCTLD[1]),
           "--set-conf=ptt_type=None"]
    if baud:
        cmd += ["-s", str(baud)]
    proc = spawn_native(cmd, claim_fd=claim_fd, long_lived=True, device=device)
    children.append((proc, drain_native(proc, log)))
    for _ in range(20):
        time.sleep(0.25)
        try:
            with socket.create_connection(RIGCTLD, timeout=1):
                return proc
        except OSError:
            continue
    stop_native(proc)
    die(f"rigctld would not answer on {device} (model {model})")


def watch_rig(rig, log: Path, period: float, stop: threading.Event) -> None:
    first = None
    while not stop.wait(period):
        try:
            now = rig.state()
        except Exception as exc:
            with open(log, "a") as fh:
                fh.write(f"{time.time():.3f} unreachable {exc}\n")
            continue
        first = first or now
        changed = "" if now == first else "  CHANGED"
        with open(log, "a") as fh:
            fh.write(f"{time.time():.3f} {now['frequency_hz']} {now['mode']} "
                     f"{now['passband_hz']}{changed}\n")


def cmd_rigs(args: argparse.Namespace) -> None:
    print("Known rigs (pass as --rig-model):\n")
    for name, model in RIG_MODELS.items():
        baud = f" @ {RIG_BAUD[name]}" if name in RIG_BAUD else ""
        print(f"  {name:<8} hamlib model {model}{baud}")
    print("\n  (x6100 speaks Icom CI-V, so it's driven as an IC-705 at 19200.)")
    print("\nAnything else: `rigctl --list` and pass the number directly.")
    print("\nThe serial port is single-owner. If your Winlink client drives PTT over")
    print("CAT, start rigctld once and point both it and this script at rigctld —")
    print("do not let both open the port.")


# --- winlink gateways ------------------------------------------------------

GATEWAYS_URL = "https://cms.winlink.org:444/GatewayChannels.aspx"
_REPO = Path(__file__).resolve().parents[1]

# Where the fetched export lands, for every tool that reads it. One path per mode
# and no other, because for a month there were three VARA copies: the one named
# below, the archived Winlink channel listings, fetched on 30 July, 4 August and 29
# July. The `--cache` defaults here and in the monitor were bare relative names,
# so the copy a run read or wrote was decided by the directory it was started in,
# and each launcher then hardcoded the one its own working directory had produced.
# Nothing was wrong in any of them; they were three answers to one question.
#
# It reached the air. Asked at BW2300 whether a channel could answer, the three
# disagreed on 58 of the 1255 channels they name between them, and on 129 at
# BW500. WP4OH-13 on 7104.7 is the shape of it: listed VARA 500 only, so a BW2300
# call there cannot be answered -- but the copy the launcher passed did not list
# the station at all, and an unlisted station reads as unknown, which passes. The
# stalest copy did not contradict the guard; it was silent, and silence is a pass.
#
# The keys are the page's own `rblModes` values, so nothing translates between
# the roster a run reads and the button the fetch presses.
_GATEWAY_CACHE = Path(os.environ.get("HFMODEM_DATA_DIR", _REPO / "operator-data")) / "cache"
GATEWAY_LISTS = {"VARA": _GATEWAY_CACHE / "winlink-vara-gateways.csv",
                 "Pactor": _GATEWAY_CACHE / "winlink-pactor-gateways.csv"}
GATEWAY_LIST = GATEWAY_LISTS["VARA"]
BANDS = {"80m": (3.5, 4.0), "60m": (5.3, 5.5), "40m": (7.0, 7.3),
         "30m": (10.1, 10.15), "20m": (14.0, 14.35), "17m": (18.068, 18.168),
         "15m": (21.0, 21.45), "12m": (24.89, 24.99), "10m": (28.0, 29.7)}


def grid_to_latlon(grid: str) -> tuple[float, float]:
    g = grid.strip().upper()
    lon = (ord(g[0]) - 65) * 20 - 180 + (int(g[2]) * 2)
    lat = (ord(g[1]) - 65) * 10 - 90 + int(g[3])
    if len(g) >= 6:
        lon += (ord(g[4]) - 65) * 5 / 60
        lat += (ord(g[5]) - 65) * 2.5 / 60
        return lat + 1.25 / 60, lon + 2.5 / 60
    return lat + 0.5, lon + 1.0


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def fetch_gateways(mode: str = "VARA") -> str:
    """Pull the public gateway CSV that winlink.org's own RMS Channels page
    renders. Two ASP.NET postbacks: select the mode, then export."""
    import http.cookiejar
    import urllib.request
    from urllib.parse import urlencode

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", "kestrel-field-capture")]

    def viewstate(html: str) -> dict:
        fields = {}
        for name in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"):
            m = re.search(rf'id="{name}" value="([^"]*)"', html)
            if m:
                fields[name] = m.group(1)
        return fields

    page = opener.open(GATEWAYS_URL, timeout=30).read().decode("utf-8", "replace")
    radio = re.search(rf'id="rblModes_(\d+)"[^>]*value="{re.escape(mode)}"', page)
    if not radio:
        raise ValueError(f"{GATEWAYS_URL} offers no {mode!r} mode button")
    form = viewstate(page)
    form.update({"__EVENTTARGET": f"rblModes${radio.group(1)}", "__EVENTARGUMENT": "",
                 "rblModes": mode, "txtServiceCodes": "PUBLIC"})
    page = opener.open(GATEWAYS_URL, urlencode(form).encode(),
                       timeout=60).read().decode("utf-8", "replace")

    form = viewstate(page)
    form.update({"__EVENTTARGET": "btnCsvFile", "__EVENTARGUMENT": "",
                 "rblModes": mode, "txtServiceCodes": "PUBLIC"})
    return opener.open(GATEWAYS_URL, urlencode(form).encode(),
                       timeout=60).read().decode("utf-8", "replace")


def cmd_gateways(args: argparse.Namespace) -> None:
    import csv
    import io

    print(f"Fetching the Winlink {args.mode} gateway list…")
    text = fetch_gateways(args.mode)
    path = GATEWAY_LISTS[args.mode]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

    rows = list(csv.DictReader(io.StringIO(text)))
    calls = {(r.get("Callsign") or "").strip() for r in rows}
    posted = rows[0].get("Timestamp", "?") if rows else "?"
    print(f"  {len(rows)} channels, {len(calls)} stations, posted {posted}")
    print(f"  {path}")


def cmd_stations(args: argparse.Namespace) -> None:
    import csv
    import io

    cache = args.cache
    if cache.exists() and not args.refresh:
        text = cache.read_text()
        print(f"(cached {cache} — --refresh to re-fetch)\n")
    else:
        print("Fetching the Winlink VARA gateway list…")
        text = fetch_gateways()
        GATEWAY_LIST.parent.mkdir(parents=True, exist_ok=True)
        GATEWAY_LIST.write_text(text)          # the one path; see load_gateways

    lo, hi = BANDS[args.band] if args.band else (0.0, 1e9)
    home = grid_to_latlon(args.grid)

    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        mode = (row.get("Mode") or "").strip()
        if not mode.startswith("VARA") or mode.startswith("VARA FM"):
            continue
        raw = (row.get("Frequency") or "").replace(",", "").replace('"', "")
        m = re.match(r"([\d.]+)\s*(KHz|MHz)", raw, re.I)
        if not m:
            continue
        mhz = float(m.group(1)) / (1000 if m.group(2).lower() == "khz" else 1)
        if not lo <= mhz <= hi:
            continue
        grid = (row.get("GridSquare") or "").strip()
        if len(grid) < 4:
            continue
        try:
            km = distance_km(home, grid_to_latlon(grid))
        except (ValueError, IndexError):
            continue
        rows.append((mode, km, mhz, row.get("Callsign", "").strip(),
                     grid, (row.get("Hours") or "").strip(),
                     (row.get("QTH") or "").strip()))

    # VARA 2750 first: those sysops run a licensed VARA, which is what can
    # originate the speed levels we still need.
    rank = {"VARA 2750": 0, "VARA": 1}
    rows.sort(key=lambda r: (rank.get(r[0], 9), r[1]))
    rows = [r for r in rows if r[0] != "VARA 500"][:args.limit]

    if not rows:
        die(f"no VARA HF gateways found on {args.band or 'any band'}")

    band = args.band or "all bands"
    off = DIAL_OFFSET_HZ / 1e6
    print(f"VARA HF gateways on {band}, nearest first (wide-capable first):\n")
    print(f"  {'call':<9} {'dial kHz':>9} {'centre':>9}  {'mode':<10} {'km':>6}  "
          f"{'UTC':<6} QTH")
    for mode, km, mhz, call, grid, hours, qth in rows:
        print(f"  {call:<9} {(mhz - off) * 1000:>9.1f} {mhz:>9.3f}  {mode:<10} "
              f"{km:>6.0f}  {hours:<6} {qth[:32]}")
    print(f"\nWinlink lists the CENTRE frequency; the dial column is centre "
          f"− {DIAL_OFFSET_HZ} Hz (USB), which")
    print("is what you tune and what --set-freq expects. VARA 2750 gateways are the")
    print("ones worth travelling for — a plain VARA entry tops out at 2300. Pick a")
    print("close one whose hours cover you now, then:")
    print("  hfcapture.py record --device N --peer <call> --set-freq <dial-kHz>")


# --- record ----------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    """Ask the operator, or take the default when there is nobody to ask.

    A prompt with no terminal behind it is not a question, it is a stall: the
    queue runner starts this in the background, the process stays alive, its
    liveness check passes, and the window records nothing at all. Metadata is
    never worth that.
    """
    if not sys.stdin.isatty():
        return default
    suffix = f" [{default}]" if default else ""
    got = input(f"  {prompt}{suffix}: ").strip()
    return got or default


class Capture:
    """Segmented capture for one session: fixed-length wavs, and where the gaps are.

    Segments are cut on the sample count, so a segment holds exactly the audio it
    claims to hold and a wav second is a wav second. ffmpeg's segment muxer cuts
    on input timestamps instead, and against a device that under-delivers those
    are different clocks — nine windows recorded before 2026-08-14 hold 536-550 s
    where 600 was asked for, missing from the middle rather than the tail.

    What cannot be recovered is reported: `gaps` carries every discontinuity the
    device's own ADC clock admits to, as ``(seconds into the capture, seconds
    lost)``, so an analysis downstream can refuse to read across one instead of
    silently sliding by it.
    """

    def __init__(self, session: Path, segment_frames: int = SEGMENT_FRAMES) -> None:
        self.session = session
        self.segment_frames = segment_frames
        self.frames = 0
        self.gaps: list[tuple[float, float]] = []
        self.parts: list[Path] = []
        self._wav: wave.Wave_write | None = None
        self._written = 0
        self._due: float | None = None

    def feed(self, pcm: bytes, adc_time: float | None = None) -> float:
        """Write one device block; return the audio lost immediately before it."""
        gap = 0.0
        if adc_time is not None:
            if self._due is not None and adc_time - self._due > GAP_S:
                gap = adc_time - self._due
                self.gaps.append((self.frames / RATE, gap))
            self._due = adc_time + (len(pcm) // 2) / RATE
        while pcm:
            if self._wav is None:
                self._open()
            room = (self.segment_frames - self._written) * 2
            self._wav.writeframes(pcm[:room])
            self._written += len(pcm[:room]) // 2
            self.frames += len(pcm[:room]) // 2
            pcm = pcm[room:]
            if self._written >= self.segment_frames:
                self.close()
        return gap

    def _open(self) -> None:
        path = self.session / f"rx-{len(self.parts):03d}.wav"
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(RATE)
        self._written = 0
        self.parts.append(path)

    def close(self) -> None:
        wav, self._wav = self._wav, None
        if wav is not None:
            wav.close()

    @property
    def seconds(self) -> float:
        return self.frames / RATE

    @property
    def lost(self) -> float:
        return sum(g for _, g in self.gaps)

    def continuous(self, elapsed: float) -> bool:
        """Whether the run is one timeline, so a wav offset is an elapsed time.

        Two ways it is not, and the second is the one that went unnoticed for two
        days: audio the device admitted losing, and audio that simply never
        arrived while the clock ran on.
        """
        return not self.gaps and elapsed - self.frames / RATE <= SLIP_S


def cmd_record(args: argparse.Namespace) -> None:
    args.outdir = args.outdir or f"cap-{datetime.now():%Y%m%d-%H%M}"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with device_claim("hfcapture record", devices=claim.Devices(
            cat_port=args.rig_device, capture=args.device,
            cat="rigctld" if args.rigctld else "serial")) as held:
        with (Path(args.outdir or ".") / "rigctld.log").open("a") as log:
            children = []
            try:
                with ExitStack() as cleanup:
                    _record(args, held.claim, children, log, cleanup)
            finally:
                for child, readers in children:
                    stop_native(child)
                    for reader in readers:
                        reader.join(timeout=2)
                        if reader.is_alive():
                            raise claim.ClaimError("native diagnostic reader did not stop")
                    close_native(child, stopped=True)


def _record(args, descriptor, children, log, cleanup):
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    session = Path(args.outdir or f"cap-{stamp}")
    session.mkdir(parents=True, exist_ok=True)

    print(f"\nSession: {session}\n")

    sd = portaudio()
    index, card = sound_input(sd, args.device)
    print(f"  input: [{index}] {card}\n")

    rig, state = None, None
    if args.rig_model or args.rigctld:
        from vara_rig_bridge import Rig

        if args.rigctld:
            host, _, port = args.rigctld.partition(":")
            hostport = f"{host}:{int(port or RIGCTLD[1])}"
        else:
            name = args.rig_model.lower()
            model = RIG_MODELS.get(name, args.rig_model)
            if not args.rig_device:
                die("--rig-model needs --rig-device (e.g. /dev/cu.usbserial-1420)")
            baud = args.rig_baud or RIG_BAUD.get(name)
            start_rigctld(int(model), args.rig_device, baud, claim_fd=descriptor,
                        log=log, children=children)
            hostport = f"{RIGCTLD[0]}:{RIGCTLD[1]}"
        rig = Rig(hostport, armed=False, log=print)
        cleanup.callback(rig.shutdown)
        if args.set_freq:
            rig.set_freq(float(args.set_freq) * 1000)
        if args.set_mode:
            rig.set_mode(args.set_mode, args.set_passband or 0)
        state = rig.state()
        print(f"  rig: {state['frequency_hz']} Hz  {state['mode']}  "
              f"passband {state['passband_hz']} Hz")
        if 0 < (state["passband_hz"] or 0) < MIN_PASSBAND:
            print(f"  WARNING: passband under {MIN_PASSBAND} Hz clips the top of the")
            print("           signal — widen the receive filter before transmitting.")
        print()

    device_opened()
    meta = {
        "utc_start": datetime.now(UTC).isoformat(timespec="seconds"),
        "my_call": args.call or ask("your callsign"),
        "peer_call": args.peer or ask("peer / gateway callsign"),
        "band": args.band or ask("band (e.g. 20m)", ""),
        "frequency": args.freq or (state and str(state["frequency_hz"]))
                     or ask("dial frequency (e.g. 14109.0 USB)"),
        "rig_state": state,
        "bandwidth": args.bandwidth or ask("VARA bandwidth", "2300"),
        "rig": args.rig or ask("rig + interface", ""),
        "notes": args.notes or ask("anything else worth knowing", ""),
        "sample_rate": RATE,
        "format": "pcm_s16le mono",
        "device": card,
        "host": platform.platform(),
    }
    (session / "session.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\nRecording. Run the Winlink session now.")
    print("Press Ctrl-C once when the session is finished and the rig is quiet.\n")

    cap = Capture(session)
    blocks: queue.Queue = queue.Queue()

    def on_block(pcm, frames, stamp, status) -> None:
        blocks.put((bytes(pcm), stamp.inputBufferAdcTime))

    stop = threading.Event()
    # SIGINT alone does not end this run. bash starts a background child with
    # SIGINT ignored and python inherits SIG_IGN, so the queue runner had to kill
    # the ffmpeg child instead and hope the parent noticed; both signals land here
    # now, and the wav is closed with its header written either way.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    watcher = None
    if rig:
        watcher = threading.Thread(target=watch_rig,
                                   args=(rig, session / "rig.log", args.rig_poll, stop),
                                   daemon=True)
        watcher.start()

    try:
        started = shown = time.time()
        with device_claim("hfcapture input", devices=claim.Devices(capture=args.device)):
            with sd.RawInputStream(device=index, channels=1, samplerate=RATE,
                                   dtype="int16", callback=on_block):
                device_opened()
                while not stop.is_set():
                    try:
                        pcm, adc = blocks.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if cap.feed(pcm, adc):
                        at, secs = cap.gaps[-1]
                        print(f"\n  !! GAP {secs:.3f} s lost at {at:.2f} s into the capture")
                    if time.time() - shown >= 1.0:
                        shown = time.time()
                        mins, sec = divmod(int(shown - started), 60)
                        # Both clocks, side by side: the failure this replaced was one of
                        # them running away from the other for two days unnoticed.
                        print(f"\r  recording {mins:02d}:{sec:02d}  "
                              f"audio {cap.seconds:.1f} s", end="", flush=True)
        while not blocks.empty():
            cap.feed(*blocks.get())
    finally:
        stop.set()
        cap.close()
        if watcher is not None:
            watcher.join(timeout=5)
            if watcher.is_alive():
                raise claim.ClaimError("rig watcher did not stop")
    elapsed = time.time() - started
    print("\n")

    parts = cap.parts
    if not parts:
        die("no audio was captured — check the device with `devices`")
    total = sum(repair_wav(p) for p in parts)
    peak, rms, clipped = measure(parts[0])
    meta["duration_seconds"] = round(total, 1)
    meta["capture_wall_seconds"] = round(elapsed, 1)
    meta["gaps"] = [{"at_seconds": round(at, 3), "seconds": round(secs, 3)}
                    for at, secs in cap.gaps]
    meta["gap_seconds"] = round(cap.lost, 3)
    meta["continuous"] = cap.continuous(elapsed)
    meta["utc_end"] = datetime.now(UTC).isoformat(timespec="seconds")
    (session / "session.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"  {len(parts)} file(s), {total / 60:.1f} min, "
          f"first segment peak {peak:.1f} dBFS, rms {rms:.1f} dBFS")
    if meta["continuous"]:
        print(f"  continuous: {total:.1f} s of audio over {elapsed:.1f} s of clock")
    else:
        print(f"  !! NOT CONTINUOUS — {len(cap.gaps)} gap(s), {cap.lost:.2f} s lost, "
              f"{elapsed - total:.2f} s short of {elapsed:.1f} s of clock")
        print("     A wav offset is not a clock time across a gap. Read each "
              "stretch between gaps on its own.")
    if clipped:
        print("  WARNING: clipping present — the decode may suffer.")
    print("\nDrop the Winlink session log into the session directory if you have it,")
    print(f"then:  {sys.argv[0]} package {session}")


# --- optional host-port tee ------------------------------------------------

def pump(src: socket.socket, dst: socket.socket, label: str,
         log: Path, binary: bool, lock: threading.Lock) -> None:
    raw = open(log.with_name(f"data-{label}.bin"), "ab") if binary else None
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
            with lock, open(log, "a") as fh:
                if raw:
                    raw.write(chunk)
                    raw.flush()
                    fh.write(f"{time.time():.3f} {label} {len(chunk)}\n")
                else:
                    for line in chunk.decode("ascii", "replace").splitlines():
                        if line.strip():
                            fh.write(f"{time.time():.3f} {label} {line.strip()}\n")
    except OSError:
        pass
    finally:
        if raw:
            raw.close()
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def serve(listen_port: int, upstream: tuple[str, int], log: Path,
          textual: bool) -> None:
    lock = threading.Lock()
    srv = socket.create_server(("127.0.0.1", listen_port), reuse_port=False)
    while True:
        client, _ = srv.accept()
        try:
            modem = socket.create_connection(upstream, timeout=10)
        except OSError as exc:
            print(f"  port {listen_port}: upstream {upstream} refused ({exc})")
            client.close()
            continue
        for a, b, label in ((client, modem, "to-modem"), (modem, client, "from-modem")):
            threading.Thread(target=pump,
                             args=(a, b, label, log, textual, lock),
                             daemon=True).start()


def cmd_proxy(args: argparse.Namespace) -> None:
    session = Path(args.outdir)
    session.mkdir(parents=True, exist_ok=True)
    host = args.upstream_host
    print(f"Teeing host ports into {session}/")
    print(f"  Winlink -> 127.0.0.1:{args.cmd_port}/{args.data_port} "
          f"-> VARA {host}:{args.upstream_cmd}/{args.upstream_data}\n")
    threading.Thread(
        target=serve,
        args=(args.cmd_port, (host, args.upstream_cmd),
              session / "cmd.log", False), daemon=True).start()
    threading.Thread(
        target=serve,
        args=(args.data_port, (host, args.upstream_data),
              session / "data.log", True), daemon=True).start()
    print("Running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print()


# --- package ---------------------------------------------------------------

def cmd_package(args: argparse.Namespace) -> None:
    session = Path(args.session).resolve()
    if not session.is_dir():
        die(f"{session} is not a directory")
    bundle = session.with_suffix(".tar.gz")
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(session, arcname=session.name)
    size = bundle.stat().st_size / 1e6
    print(f"\n  {bundle}  ({size:.1f} MB)\n")
    print("Hand that file back whenever it's convenient.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="list audio inputs").set_defaults(fn=cmd_devices)

    p = sub.add_parser("check", help="5s level check before you commit")
    p.add_argument("--device", required=True, help=DEVICE_HELP)
    p.add_argument("--seconds", type=int, default=5)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("record", help="record a session")
    p.add_argument("--device", required=True, help=DEVICE_HELP)
    p.add_argument("--outdir")
    p.add_argument("--call")
    p.add_argument("--peer")
    p.add_argument("--band")
    p.add_argument("--freq")
    p.add_argument("--bandwidth")
    p.add_argument("--rig")
    p.add_argument("--notes")
    p.add_argument("--rig-model", help="g90 | x6100 | ft891 | a hamlib model number")
    p.add_argument("--rig-device", help="serial device for rigctld")
    p.add_argument("--rig-baud", type=int,
                   help="CAT baud (default: 19200 for x6100, hamlib default otherwise)")
    p.add_argument("--rigctld", help="host[:port] of an already-running rigctld")
    p.add_argument("--set-freq", help="dial frequency in kHz, e.g. 14109.0")
    p.add_argument("--set-mode", help="e.g. PKTUSB")
    p.add_argument("--set-passband", type=int)
    p.add_argument("--rig-poll", type=float, default=5.0)
    p.set_defaults(fn=cmd_record)

    sub.add_parser("rigs", help="rig models and the serial-port caveat").set_defaults(fn=cmd_rigs)

    p = sub.add_parser("gateways", help="refresh one mode's Winlink gateway roster")
    p.add_argument("--mode", default="VARA", choices=sorted(GATEWAY_LISTS))
    p.set_defaults(fn=cmd_gateways)

    p = sub.add_parser("stations", help="find VARA HF gateways worth working")
    p.add_argument("--grid", required=True, help="your Maidenhead grid, e.g. EM12jt")
    p.add_argument("--band", choices=sorted(BANDS))
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--cache", type=Path, default=GATEWAY_LIST)
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_stations)

    p = sub.add_parser("proxy", help="optional: tee the VARA host ports")
    p.add_argument("--outdir", required=True)
    p.add_argument("--cmd-port", type=int, default=8300)
    p.add_argument("--data-port", type=int, default=8301)
    p.add_argument("--upstream-host", default="127.0.0.1")
    p.add_argument("--upstream-cmd", type=int, default=8320)
    p.add_argument("--upstream-data", type=int, default=8321)
    p.set_defaults(fn=cmd_proxy)

    p = sub.add_parser("package", help="tar a session for handoff")
    p.add_argument("session")
    p.set_defaults(fn=cmd_package)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
