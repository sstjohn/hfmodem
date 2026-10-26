# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Record one whole on-air exchange from a third-party receiver, unwindowed.

Everything this project has measured about its own transmissions has come from
its own capture stream, which is sliced into 240 ms listen windows and blanked
across the keyed interval. So our transmitted waveform AS THE BAND CARRIES IT has
never been observed -- not the audio we render, but what leaves the rig and
arrives somewhere else, with the linear amplifier, the ALC and the antenna in the
path.

A KiwiSDR a few hundred miles away hears both stations on one clock with no
windowing by us. That is the only vantage from which "did our packet land on the
raster the gateway set" and "what did the gateway send, and when" are the same
measurement.

AGC is off and ADPCM is off (`kiwirx` defaults): a re-levelling receiver puts a
per-symbol amplitude error on the very thing being examined.

`--peer` says which of the two stations the receiver is there for. Without it the
witness is placed around us, which is right when our own emission is the thing
being read. The arms that key over the peer need the opposite (see `WHICH
STATION` below), and a witness placed for us cannot serve them. `--prefer` names
one outright, for when a particular receiver is known to be the only vantage that
resolves both stations and no distance can say so.

`--out` takes the arm's directory as readily as a file path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import time
import wave
from pathlib import Path
from typing import NamedTuple

import numpy as np

# WHICH STATION THE WITNESS IS FOR. Every measurement this tool was built for was
# of our own emission, so "a few hundred miles away" meant a few hundred from us.
# The arms that transmit over the peer invert that: our receiver is muted for two
# of every three of the peer's answer slots, so only a third party can say
# whether the peer answered during them -- and a Kiwi inside our ground wave
# hears our carrier and not a gateway 662 mi behind it. On 2026-08-30 the only
# way to reach toward Arkansas was a distance window, which cannot express a
# direction: `--near 500 --far 900` selected a receiver 516 mi away in Virginia.
# `--peer` names the station to be heard, which is the thing a distance from us
# can never say.
#
# HOW FAR A WITNESS MAY BE, AND WHY THAT IS MEASURED FROM US. A witness is read
# by registering it against our own keying, so one that cannot hear our key-down
# holds no answer slot of ours whatever it heard of the peer. The pairs in this
# station's captures that registered sit at 103 and 329 mi from us; every one
# that did not is 581 mi or further -- 581, 609, 652, 805, 830 and 1074, the last
# two 717 and 685 mi from their gateways and neither any use for it. Inside
# NEAR_MI our own carrier is the only thing a Kiwi hears.
NEAR_MI, FAR_MI = 50.0, 500.0

# HOW MANY GET TRIED. The walk stops at the first receiver that answers, so this
# is not a cost, it is how many full or unreachable Kiwis a night may hold before
# an arm goes unwitnessed. The attempt that ended in `all 2 client slots taken`
# is the reason it is not one.
TRIES = 8

# HOW OFTEN ONE IS ASKED BEFORE THE WALK MOVES ON. A Kiwi that drops the socket
# or sends no audio has said nothing about the next minute: on 09-14 Empire
# closed before sending audio once, and the arm spent its 600 s on Dayton, 408 mi
# out, whose recording does not carry this station's 236 ms controls over its own
# noise. A FULL receiver is a different answer -- that one is a fact about the
# receiver and is taken the first time, as `open_first_available` always has.
ATTEMPTS, BACKOFF_S = 3, 1.5

# WHEN THE TWO STATIONS ARE CLOSE ENOUGH TO SIT EITHER SIDE OF ONE RECEIVER.
# Detour off the great circle ranks a receiver by an ellipse, which on a long
# path admits one sitting beside either end; under this separation the midpoint
# is reachable and a receiver at it hears both stations at the most similar
# range, which is what "one clock" asks for.
PAIR_MI = 300.0

# HOW OFTEN THE SIDECAR IS RE-WRITTEN WHILE THE RECORDING RUNS. The audio is
# normalised per file and so cannot be written a block at a time, but a witness
# that is killed outright should still leave which receiver it got and how long
# it had been listening.
FLUSH_S = 30.0

# Where a gateway callsign becomes a grid. All three are live downloads and none
# of them is in the repository, so a callsign that is not in them is said out
# loud rather than guessed at -- a witness placed at an invented peer is worse
# than one placed at us and known to be.
ROSTERS = ("winlink-pactor-gateways.csv", "winlink-channels-from-pat.csv",
           "winlink-vara-gateways.csv")
from consolelib.operator import data_root
REPO = data_root()
_GRID = re.compile(r"[A-R]{2}[0-9]{2}([A-X]{2})?$", re.I)


class Cand(NamedTuple):
    host: str
    port: int
    name: str
    mi_us: float
    mi_peer: float | None        # None when no peer was named
    mi_mid: float | None = None  # from the midpoint of us and the peer


class Take(NamedTuple):
    """The receiver the walk settled on, with the audio its probe already holds."""
    session: object
    cand: Cand
    probe: list
    t0: float
    blocks: object


# WHAT THE PROBE IS ALLOWED TO CONCLUDE. A peak is a level and a level cannot
# tell band noise from a receiver's own noise, so `peak >= DEAF_PEAK` passed
# three receivers on 2026-09-03 and 09-04 whose whole recording was noise. What
# separates them is whether anything ever stood over that noise: all 17 probes in
# this station's captures from a receiver hearing nothing are flat to 1.0-1.4 dB
# over 20 s, and none of them ever exceeds it. Flatness alone is not deafness --
# 12% of working receivers' first 20 s are that flat too, on a quiet channel --
# so a candidate is dropped only when it is BOTH flat and under the level a
# working receiver's band noise sits at.
PROBE_S, DEAF_S = 4.0, 20.0
DEAF_PEAK, SIGNAL_DB = 5e-3, 6.0

# HOW LONG PLACEMENT MAY TAKE, COUNTED FROM THE FIRST CANDIDATE. The walk is no
# longer paid out of the sense: the launcher places the witness once the gate has
# passed the channel, so every second spent here is a second of the arm that the
# recording does not hold. `witness_align` searches either way round but needs
# half of ours to lie inside the witness, and the short connect arm this station
# flies is about 48 s of `stream.wav` -- so a placement later than 24 s takes the
# pair under that half and prints `no overlap to measure`, which reads as two
# recordings of different air rather than as a receiver that came up late.
#
# GIVING UP IS THE ANSWER, NOT WAITING LONGER. A witness late enough to break the
# registration is worth less than no witness at all: the first leaves a verdict
# line to be misread, the second leaves an honest absence. The 165 s walk in this
# station's record got its recording, and nothing could have been read against it.
#
# IT BOUNDS A BLOCKING CONNECT AS WELL AS THE LOOP. `kiwirx` waits
# `socket_timeout + 15` -- 45 s -- for a first block and that is not this tool's
# to set, so the budget arrives as SIGALRM rather than only as a check between
# tries. One deaf candidate can spend the whole of it; that is the trade, and the
# alternative is arriving too late to be read.
PLACE_S = 20.0


class Late(Exception):
    """The placement budget ran out, wherever the walk had got to."""


def _late(*_):
    raise Late


def great_circle(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 3959.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = (math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(b[1] - a[1]) / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def latlon(lat_lon: str) -> tuple[float, float]:
    la, lo = (float(v) for v in lat_lon.strip('"').split(","))
    return la, lo


def peer_latlon(spec: str, rosters=None) -> tuple[tuple[float, float], str]:
    """Where `--peer` is, from a callsign, a grid, or a bare `lat,lon`.

    A callsign is what an arm actually holds -- `onair.sh` knows which gateway it
    is calling and nothing else about it -- so the rosters it is already calling
    from are where the grid comes from. Matched on the base call, because a
    station is listed as `K5DAT` and called as `K5DAT-13`.
    """
    spec = spec.strip()
    if "," in spec:
        return latlon(spec), spec
    if _GRID.fullmatch(spec):
        return grid_to_latlon(spec), spec.upper()
    call = spec.upper().split("-")[0]
    for path in (Path(p) for p in (rosters or [REPO / r for r in ROSTERS])):
        if not path.exists():
            continue
        for row in csv.DictReader(path.open()):
            r = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
            listed = (r.get("basecallsign") or r.get("callsign") or "").strip().upper()
            if listed.split("-")[0] != call:
                continue
            grid = (r.get("gridsquare") or r.get("grid") or "").strip()
            if _GRID.fullmatch(grid):
                return grid_to_latlon(grid), f"{call} {grid.upper()}"
    raise LookupError(f"{spec}: no grid for it in " + ", ".join(ROSTERS))


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dlo = math.radians(b[1] - a[1])
    bx, by = math.cos(la2) * math.cos(dlo), math.cos(la2) * math.sin(dlo)
    la = math.atan2(math.sin(la1) + math.sin(la2),
                    math.hypot(math.cos(la1) + bx, by))
    return math.degrees(la), a[1] + math.degrees(math.atan2(by, math.cos(la1) + bx))


def preferred(c: Cand, prefer: str | None) -> bool:
    """Whether `--prefer` names this receiver: host, `host:port`, or its first label.

    `22059.proxy.kiwisdr.com` is one receiver an operator knows by the number in
    front, so that is enough to name it with.
    """
    if not prefer:
        return False
    host = c.host.lower()
    return prefer.strip().split(":")[0].lower() in (host, host.split(".")[0])


def candidates(path: Path, want: int, near: float = NEAR_MI, far: float = FAR_MI,
               peer: tuple[float, float] | None = None,
               prefer: str | None = None,
               station: tuple[float, float] | None = None) -> list[Cand]:
    """Receivers with a free slot, ranked by whether they can hear BOTH stations.

    Three tests in that order, because they fail differently. Outside `near`-`far`
    OF US the recording cannot be registered against our keying and holds no
    answer slot to read, which is the failure that costs the whole arm. Nearer us
    than the peer, our carrier sits over the peer's answer. Between those, the
    receiver closest to the great-circle path hears the two of them on the most
    similar footing.

    The ones that fail a test are kept as a tail rather than dropped -- a witness
    from the wrong place is still a second clock, and `placement` says which.

    Under `PAIR_MI` apart the third test becomes the distance from the MIDPOINT of
    the two stations: detour ranks by an ellipse, which on a short path is wide
    enough to seat a receiver beside one end of it, and one end is where a witness
    hears one station. `--prefer` is the operator saying which receiver this is,
    and it outranks all three -- the geometry describes candidates, and a named
    one has already been chosen.
    """
    mid = midpoint(station, peer) if peer and station else None
    out = []
    for row in csv.DictReader(path.open()):
        try:
            if int(row["free_channels"]) < 1:
                continue
            here = latlon(row["lat_lon"])
            out.append(Cand(row["hostname"], int(row["port"]), row["name_location"],
                            great_circle(station, here) if station else math.nan,
                            great_circle(peer, here) if peer else None,
                            great_circle(mid, here) if mid else None))
        except Exception:
            continue
    if station is None:
        out.sort(key=lambda c: (not preferred(c, prefer), c.mi_peer or 0.0))
        return out[:want]
    apart = great_circle(station, peer) if peer else 0.0
    close = peer is not None and apart <= PAIR_MI
    out.sort(key=lambda c: (not preferred(c, prefer), not near <= c.mi_us <= far,
                            blind(c), c.mi_mid if close else off_path(c, apart),
                            c.mi_us))
    return out[:want]


def blind(c: Cand) -> bool:
    """Nearer us than the peer: our own carrier is over whatever the peer sent."""
    return c.mi_peer is not None and c.mi_peer > c.mi_us


def off_path(c: Cand, apart: float) -> float:
    """Miles of detour off the great circle between the two stations."""
    return 0.0 if c.mi_peer is None else c.mi_us + c.mi_peer - apart


def over_noise(x: np.ndarray, fs: float) -> float:
    """How far the loudest 40 ms of a probe stands over the probe's own median."""
    n = int(fs * 0.04)
    m = len(x) // n
    if m < 20:
        return 0.0
    lvl = 10 * np.log10(np.mean(x[:m * n].reshape(m, n).astype(np.float64) ** 2,
                                axis=1) + 1e-20)
    return float(lvl.max() - np.median(lvl))


def usable(x: np.ndarray, fs: float) -> bool:
    """Whether a probe proves the receiver is hearing the band and not itself."""
    return over_noise(x, fs) >= SIGNAL_DB or float(np.abs(x).max()) >= DEAF_PEAK


def placement(c: Cand, label: str | None, apart: float) -> str:
    """Why this receiver was taken, in one line, where the choice is made.

    IT IS A PLACEMENT AND NOT A LICENCE. Distances say where the receiver stands;
    they cannot say that a slot in its recording is the slot we keyed over, and
    reading a silence needs both. On 2026-08-31 a receiver 717 mi from the gateway
    and 830 mi from us passed the old geometry gate and the deaf probe, and could
    not be registered against our own recording at all. So the sentence stops at
    what geometry knows and hands the rest to `witness_align`, which the arm runs
    at teardown.
    """
    if math.isnan(c.mi_us):
        where = "" if c.mi_peer is None else f", {c.mi_peer:.0f} mi from {label}"
        return (f"WITNESS PLACED, STATION LOCATION UNKNOWN: {c.host}{where} -- no [station] grid, "
                "so nothing is claimed about its distance from us, whether it hears our key-down, "
                "or the path between the stations. The alignment verdict at teardown says "
                "whether it reads that way.")
    if label is None or c.mi_peer is None:
        return (f"WITNESS PLACED FOR US: {c.host}, {c.mi_us:.0f} mi from us and no "
                "peer named -- our own emission is all this placement covers.")
    why = []
    if not NEAR_MI <= c.mi_us <= FAR_MI:
        why.append(f"outside the {NEAR_MI:.0f}-{FAR_MI:.0f} mi from us our key-down "
                   "has ever registered from, so the pair may not be one measurement")
    if blind(c):
        why.append(f"nearer us than {label}, so our carrier is over the peer's answer")
    return (f"WITNESS PLACED{' SHORT' if why else ''}: {c.host}, {c.mi_us:.0f} mi "
            f"from us and {c.mi_peer:.0f} mi from {label}, "
            f"{off_path(c, apart):.0f} mi off the {apart:.0f} mi path between them"
            + (" -- " + "; ".join(why) if why else
               " -- near enough to us to register and nearer the peer than we are, "
               "so both stations are on one clock")
            + ". The alignment verdict at teardown says whether it reads that way.")


def resolve_out(spec: str) -> Path:
    """`--out` as a file, or as the arm's directory to write `witness.wav` into.

    Twelve arms on 2026-09-13 lost the recording at teardown to an
    IsADirectoryError: the launcher handed over the session directory, which is
    the obvious thing to hand over, and the tool found out at the write -- after
    the audio it had spent the arm collecting was the only copy.
    """
    p = Path(spec)
    if p.is_dir() or spec.endswith(os.sep) or not p.suffix:
        return p / "witness.wav"
    return p


def listen(session, blocks, stopping: list, deadline: float) -> tuple[list, float]:
    """Audio until the receiver has proved it hears the band, `DEAF_S` passes, or
    the placement budget runs out -- which returns what it has rather than
    throwing away a receiver that had already proved itself."""
    heard, t0 = [], time.time()
    for blk in blocks:
        heard.append(np.asarray(blk.samples, dtype=np.float32))
        waited = time.time() - t0
        if stopping or waited >= DEAF_S or time.time() >= deadline:
            break
        if waited >= PROBE_S and len(heard) % 16 == 0 and over_noise(
                np.concatenate(heard), session.sample_rate) >= SIGNAL_DB:
            break
    return heard, t0


def acquire(client, cands: list[Cand], dial_hz: float, stopping: list,
            current: dict, receiver_options: dict | None = None) -> Take | None:
    """A session on the best receiver that answers AND hears the band.

    ASKED AGAIN BEFORE THE WALK MOVES ON. A dropped socket or a stream that opens
    and sends nothing is a moment, not a property: on 09-14 the only receiver that
    resolves both this station's controls and WS8EOC closed once before sending
    audio, and the arm spent its whole 600 s on one 408 mi away that does not
    carry a 236 ms control over its own noise. A FULL receiver is answering about
    itself and is taken at its word the first time.

    A candidate that answers and is deaf is not retried either -- the probe has
    already listened for up to `DEAF_S`, which is the measurement, not a mishap.

    THE WHOLE WALK IS BOUNDED BY `PLACE_S`, wherever it has got to when the
    budget runs out -- mid-handshake on a Kiwi that will never answer as readily
    as between candidates.
    """
    busy = getattr(client, "BUSY_ERRORS", ())
    deadline = time.time() + PLACE_S
    signal.signal(signal.SIGALRM, _late)
    signal.setitimer(signal.ITIMER_REAL, PLACE_S)
    try:
        return walk(client, cands, dial_hz, stopping, current, busy, deadline,
                    receiver_options)
    except Late:
        print(f"WITNESS LATE: nothing was recording {PLACE_S:.0f}s after the walk "
              "began, so no receiver is placed. One this far into the arm cannot "
              "be registered against our own keying, and an absent witness is a "
              "cleaner answer than an unreadable one.", flush=True)
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def walk(client, cands: list[Cand], dial_hz: float, stopping: list, current: dict,
         busy, deadline: float, receiver_options: dict | None = None) -> Take | None:
    """The candidate list itself, under `acquire`'s budget."""
    for c in cands:
        url = f"{c.host}:{c.port}"
        for attempt in range(1, ATTEMPTS + 1):
            if stopping:
                return None
            try:
                current["s"] = session = client.KiwiSession(
                    c.host, c.port, dial_hz, **(receiver_options or {})).start()
                heard, t0 = listen(session, blocks := session.blocks(), stopping,
                                   deadline)
            except Late:
                raise
            except busy as e:
                print(f"  {url:38s} full ({type(e).__name__}) -- taking the next "
                      "candidate rather than polling it", flush=True)
                break
            except Exception as e:
                print(f"  {url:38s} failed on try {attempt} of {ATTEMPTS} "
                      f"({type(e).__name__}: {e})", flush=True)
                shut(current)
                if attempt < ATTEMPTS:
                    time.sleep(BACKOFF_S * attempt)
                continue
            if not heard:
                print(f"  {url:38s} opened and sent no audio on try {attempt} of "
                      f"{ATTEMPTS}", flush=True)
                shut(current)
                if attempt < ATTEMPTS:
                    time.sleep(BACKOFF_S * attempt)
                continue
            x = np.concatenate(heard)
            if stopping or usable(x, session.sample_rate):
                return Take(session, c, heard, t0, blocks)
            print(f"WITNESS DEAF: {url} delivered noise and nothing in it over "
                  f"{time.time() - t0:.0f}s -- peak {float(np.abs(x).max()):.5f}, "
                  f"flat to {over_noise(x, session.sample_rate):.1f} dB; dropping "
                  "it and taking the next candidate.", flush=True)
            shut(current)
            break
    return None


def shut(current: dict) -> None:
    session = current.pop("s", None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass


from hfmodem.core.capture_producer import restore_signals
from hfcapture import grid_to_latlon
from hfmodem.core import config


@restore_signals
def main(argv: list[str] | None = None, client=None) -> int:
    global PLACE_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", type=float, required=True, help="receiver dial in kHz")
    ap.add_argument("--mode", choices=('usb', 'lsb'), default='usb', help='receiver sideband (default USB)')
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--out", required=True)
    from hfmodem.core import capture_producer
    capture_producer.arguments(ap)
    from consolelib.operator import witness_inventory
    ap.add_argument("--candidates", type=Path, default=witness_inventory())
    ap.add_argument("--grid", help="station grid; defaults to configured station grid")
    ap.add_argument("--near", type=float, default=NEAR_MI)
    ap.add_argument("--far", type=float, default=FAR_MI)
    ap.add_argument("--peer", help="the station this witness is FOR: gateway "
                                   "callsign, grid, or lat,lon. Without it the "
                                   "receiver is placed around us.")
    ap.add_argument("--prefer", help="take this receiver first, ahead of the "
                                     "geometry: hostname, host:port, or the "
                                     "number in front of a proxy name.")
    ap.add_argument("--passband", nargs=2, type=int, metavar=("LOW", "HIGH"),
                    help="positive audio passband in Hz for either sideband; omitted uses client defaults")
    ap.add_argument("--agc-gain", type=int,
                    help="explicit fixed receiver gain with AGC off; omitted uses client defaults")
    ap.add_argument("--place-seconds", type=float, default=PLACE_S,
                    help="how long the walk may take to start recording; a camp "
                         "has no arm to keep up with and can wait for a busy Kiwi")
    args = ap.parse_args(argv)
    PLACE_S = args.place_seconds
    if args.passband is not None and not 0 <= args.passband[0] < args.passband[1] <= 6000:
        ap.error("--passband requires 0 <= LOW < HIGH <= 6000 Hz")
    if args.agc_gain is not None and not 0 <= args.agc_gain <= 120:
        ap.error("--agc-gain requires 0..120")
    if not args.candidates.is_file():
        print(f'WITNESS NONE: candidate inventory absent: {args.candidates}', flush=True)
        return 1
    out = resolve_out(args.out)
    ownership = capture_producer.prepare(out, capture_id=args.capture_id,
        capture_root=args.capture_root, capture_role=args.capture_role,
        roles=("launcher-witness", "console-witness"))
    with capture_producer.Lifecycle(ownership) as lifecycle:

        # Imported here rather than at the top so the receiver choice above can be
        # exercised without the vendored Kiwi client on the path.
        if client is None:
            from consolelib import kiwi_client as client

        # AN UNRESOLVABLE PEER DOES NOT COST THE RECORDING. The roster may be stale
        # or absent and the callsign may be one nobody publishes; a witness placed
        # for us is still a second clock, and `placement` below says which it is.
        peer = label = None
        if args.peer:
            try:
                peer, label = peer_latlon(args.peer)
            except (LookupError, ValueError) as e:
                print(f"WITNESS: --peer {e} -- placing the receiver for us instead.",
                      flush=True)

        grid = args.grid if args.grid is not None else config.grid()
        station = grid_to_latlon(grid) if grid else None
        cands = candidates(Path(args.candidates), TRIES, args.near, args.far, peer,
                           args.prefer, station)
        if not cands:
            print("no receiver with a free slot", flush=True)
            return 1
        if args.prefer and not preferred(cands[0], args.prefer):
            print(f"WITNESS: --prefer {args.prefer} names no receiver with a free slot "
                  "in the candidate list -- the geometry chooses instead.", flush=True)
        apart = great_circle(station, peer) if peer and station else 0.0
        if station is None:
            print(f"{len(cands)} candidates, station location unknown (no [station] grid): "
                  + (f"nearest {label} first, with no distance window or path test:" if peer else
                     "in the candidate list's order, with no distance window:"), flush=True)
        else:
            print(f"{len(cands)} candidates, {args.near:.0f}-{args.far:.0f} mi from us and "
                  f"nearest the {apart:.0f} mi path to {label} first:" if peer else
                  f"{len(cands)} candidates, {args.near:.0f}-{args.far:.0f} mi from us "
                  "first, nearest inside that:", flush=True)
        for c in cands:
            near_peer = "" if c.mi_peer is None else f"{c.mi_peer:5.0f} mi peer  "
            if c.mi_peer is not None and station is not None:
                near_peer += f"{off_path(c, apart):5.0f} mi off  "
            us = "    ? mi us" if math.isnan(c.mi_us) else f"{c.mi_us:5.0f} mi us"
            print(f"  {us}  {near_peer}{c.host}:{c.port}  {c.name[:40]}",
                  flush=True)

        # A DEAF RECEIVER IS WORSE THAN NO RECEIVER, because a placement line will
        # otherwise license "what it did not hear, the peer did not send" from a
        # receiver that heard nothing at all -- three of six arms on 2026-08-30 came
        # back at peak 0.0006 while the geometry verdict read `nearer the peer`.
        current: dict = {}
        stopping: list = []

        # The session is closed from inside the handler and not merely flagged. The
        # record loop blocks in a queue read with no timeout, so a flag is only seen
        # when the NEXT block arrives -- and the receiver going away is both the
        # commonest reason to be stopping and the one case where no next block is
        # coming. Closing puts the sentinel that ends the iteration. It reads the
        # session indirectly because the probe below may go through several.
        def stop(*_):
            stopping.append(True)
            s = current.get("s")
            # NOTHING TO SAVE YET MEANS GO NOW. Between candidates there is no open
            # session and the search is blocked in a Kiwi handshake that no flag
            # reaches, so a handler that only sets one leaves the launcher waiting out
            # its SIGTERM grace before SIGKILL -- which is how two arms lost a witness
            # that had already chosen its receiver correctly.
            if s is None:
                os._exit(143)
            s.close()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)

        walked = time.time()
        receiver_options = {}
        if args.mode != 'usb':
            receiver_options['mode'] = args.mode
        if args.passband is not None:
            receiver_options["passband"] = tuple(args.passband)
        if args.agc_gain is not None:
            receiver_options["agc_gain"] = args.agc_gain
        take = acquire(client, cands, args.dial * 1e3, stopping, current, receiver_options)
        if take is None:
            print("WITNESS NONE: "
                  + ("the placement budget above ran out"
                     if time.time() - walked >= PLACE_S else
                     "no candidate answered and heard the band")
                  + ". This arm has no usable third-party cover, and silence in any "
                    "recording is not the peer's.", flush=True)
            return 1
        session, got, probe, t0, blocks = take

        fs = session.sample_rate
        url = f"{got.host}:{got.port}"
        print(f"recording {args.seconds:.0f}s from {url} at dial {args.dial:.1f} kHz, "
              f"{fs:.0f} Hz", flush=True)
        print(placement(got, label, apart), flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        side = out.with_suffix(".json")
        client_path = Path(client.__file__) if getattr(client, "__file__", None) else None
        client_source = ({"path": str(client_path),
                          "sha256": hashlib.sha256(client_path.read_bytes()).hexdigest()}
                         if client_path and client_path.is_file() else None)

        # The dial belongs WITH the audio. `rf-corpus/offair/kc9ghz_answer_20260806/` is a
        # witness that reads 41 of 41 tones of our own connect request and no longer has a
        # frequency attached to it, because the dial was printed and nothing kept the
        # print. A witness that cannot say what it was tuned to corroborates a waveform
        # and nothing else.
        # `utc_start` is the FIRST sample, probe included -- `acc` starts with the
        # probe audio, so a t0 taken after it dated every witness that many seconds
        # late, and a frame anchor is read against this.
        def save(x: np.ndarray, complete: bool) -> float:
            """The sidecar, WHILE IT RECORDS as well as at the end.
    
                The audio is normalised per file, so it cannot be written a block at a
                time; this can, and a witness that is killed outright then still says
                which receiver it had and how long it had been listening, which is the
                difference between "we did not witness this arm" and "the recording is
                gone". `complete` is the recording, not the file.
                """
            peak = float(np.abs(x).max()) or 1.0
            placed_blind = math.isnan(got.mi_us)
            facts = {'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0)), 'dial_khz': args.dial, 'mode': args.mode, 'centre_khz': args.dial + (1.5 if args.mode == 'usb' else -1.5), 'receiver': url, 'requested_settings': receiver_options, 'client_source': client_source, 'seconds': len(x) / fs, 'sample_rate': fs, 'peak': peak, 'peak_dbfs': 20 * math.log10(max(peak, 1e-12)), 'rms': float(x.std()), 'normalised_to': 0.9, 'complete': complete, 'peer': label, 'peer_miles': got.mi_peer, 'station_miles': None if placed_blind else got.mi_us, 'midpoint_miles': got.mi_mid, 'path_miles': apart or None, 'off_path_miles': None if got.mi_peer is None or placed_blind else off_path(got, apart), 'hears_us': None if placed_blind else args.near <= got.mi_us <= args.far, 'resolves_peer': None if got.mi_peer is None or placed_blind else got.mi_peer <= got.mi_us}
            if ownership is None:
                side.write_text(json.dumps(facts, indent=2) + '\n')
            else:
                facts['capture_state'] = ('finalized' if wrote else 'partial') if complete else 'ongoing'
                if complete and wrote:
                    facts['sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
                if complete and (not wrote):
                    facts['complete'] = False
                ownership.sidecar.update(facts)
            return peak

        # STOPPED IS NOT KILLED. `--seconds` is set long, because an arm's length is
        # not known when it starts and a witness that stops early loses the tail --
        # the changeover and the acknowledgement, which is the part worth having. What
        # ends the recording is the arm ending, and that arrives as a SIGTERM. Under
        # the default disposition it took every sample with it: the audio accumulates
        # in memory and is written once at the end, so the arm we came to witness
        # would have left no file at all.
        #
        # THE CLOCK STARTS AT THE RECEIVER WE KEPT. `t0` is the first sample of the
        # session being recorded, not of the process, so a walk that spent a minute on
        # receivers that refused still records the full `--seconds` from the one that
        # answered, rather than a remainder of it.
        stream_error = False
        acc, flushed = list(probe), t0
        save(np.concatenate(acc), complete=False)
        try:
            for blk in blocks:
                acc.append(np.asarray(blk.samples, dtype=np.float32))
                now = time.time()
                if stopping or now - t0 >= args.seconds:
                    break
                if now - flushed >= FLUSH_S:
                    flushed = now
                    save(np.concatenate(acc), complete=False)
        except Exception as e:
            stream_error = True
            print(f"WITNESS SHORT: {url} stopped sending after {time.time() - t0:.0f}s "
                  f"({type(e).__name__}: {e}) -- writing what it did send.", flush=True)
        finally:
            # NOT IN THE WAY OF THE WRITE. A receiver that goes away can throw out of
            # its own close, and the audio is still in hand at that point; letting
            # that reach the caller would discard a recording that had already run to
            # its end.
            try:
                session.close()
            except Exception as e:
                stream_error = True
                print(f"WITNESS: {url} did not close cleanly ({type(e).__name__}: {e})"
                      " -- the recording below is unaffected.", flush=True)

        if not acc:
            print("no audio", flush=True)
            return 1
        x = np.concatenate(acc)
        # 16-bit, PER-FILE NORMALISED to 0.9 -- every witness peaks at 0.89996 on disk,
        # whatever the receiver was doing. The printed peak/rms are the levels BEFORE
        # that, so an absolute level can be recovered, which shrike's own captures
        # cannot offer.
        #
        # Those two facts are easy to combine wrongly, and were: two witnesses of the
        # same transmitter printed peaks 16 dB apart and that was read as our signal
        # having weakened, when it describes the KiwiSDR's own gain and AGC state.
        # Measured relative to the band inside each file -- the only comparison a
        # normalised file supports -- the quieter-printing one was 4.9 dB BETTER.
        # Compare across witnesses only against something in the same recording.
        #
        # A WAV header holds a whole number of hertz and this receiver's clock is
        # 11998.874629, so the header is a rounded copy and the sidecar below is the
        # measurement. `witness_align.witness_rate` reads the sidecar for that
        # reason; resampling to make the header true would put an interpolation on
        # the very waveform this file exists to hold.
        peak = float(np.abs(x).max()) or 1.0
        wrote = True
        try:
            audio_path = out
            sink = None
            if ownership:
                audio_path, sink = ownership.audio.temporary()
            try:
                with wave.open(sink if sink is not None else str(audio_path), 'wb') as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(int(round(fs)))
                    w.writeframes((x / max(peak, 1e-09) * 0.9 * 32767).astype(np.int16).tobytes())
            finally:
                if sink is not None:
                    sink.close()
            if ownership:
                ownership.audio.publish_file(audio_path)
        except Exception as e:
            wrote = False
            print(f'WITNESS: the audio could not be written to {out} ({type(e).__name__}: {e}) -- the sidecar still says what was heard.', flush=True)
        save(x, complete=True)
        if wrote:
            print(f"wrote {out}  {len(x)/fs:.1f}s at {fs:.6g} Hz "
                  f"({'stopped with the arm' if stopping else 'ran its full --seconds'})",
                  flush=True)
        print(f"wrote {side}", flush=True)
        # THE FILE'S OWN LEVEL SAYS NOTHING, so the true one is said here. Every
        # witness peaks at 0.89996 on disk whatever the receiver was doing, and four
        # captures at peak 0.05 read as healthy until someone opened the sidecar.
        print(f"WITNESS LEVEL: peak {peak:.5f} ({20 * math.log10(max(peak, 1e-12)):+.1f} "
              f"dBFS) rms {x.std():.5f} at the receiver -- the .wav is normalised to "
              "0.9, so nothing about the file's own level is a measurement."
              + ("\nWITNESS LEVEL: that is near-silence; this receiver heard nothing."
                 if peak < 1e-3 else ""), flush=True)
        if wrote and not stream_error:
            lifecycle.complete()
        return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main())
