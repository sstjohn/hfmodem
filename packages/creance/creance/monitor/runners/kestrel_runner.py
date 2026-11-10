# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel monitor runner: live s16le on stdin -> normalised detection JSON out.

This is the whole kestrel adapter. It reuses the streaming pieces of
``hfmodem.kestrel.monitor`` that ``tools/vara_monitor``'s traffic log runs — the
``MonitorGate`` that brackets keyed bursts, ``classify`` that names each one, and
the ``HandshakeScanner`` that reads connect-request and connect-response off the
raw stream — so the detections are the same set the vara_monitor log produces.
The scanner's tone shift is the sender's tuning error, and it is the offset.
Wideband overs are not decoded here (that is a ~20 s job that would back the live
stream up), matching what vara_monitor does on ``--device``.

``--centre HZ`` makes this a track's instance: the audio is cut to ``--bw`` around
that centre and moved to 1500 Hz before anything reads it, and the offset is
reported from where the track sits.

``normalize`` is pure and carries the confidence policy: a resolved caller or a
gateway matched against the Winlink list, or a CRC-valid DATA over, is CONFIRMED;
a recognised VARA waveform whose destination was not in the list, a bare control
token, or an undecoded wideband over is TENTATIVE — a real signal, unproven
identity. An unrecognised burst names no protocol at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hfmodem.monitor import FS

#: How often a track's instance says how far it has decoded.
HEARD_EVERY_S = 0.5
#: One handshake read by two tiles is one burst.
DUP_S = 0.2
#: A track's instance scans its channel in tiles this far apart: the handshake
#: scanner's own reach is +/-3 tone bins, so seven bins of 23.4375 Hz butt one
#: tile's reach against the next's and keep every tile on the tone grid.
TONE_HZ = 23.4375
TILE_HZ = 7 * TONE_HZ
#: How near the channel's edge the last tile's centre may sit: a BW500 request
#: is 610 Hz wide.
TILE_EDGE_HZ = 250.0
#: VARA's widest waveform, BW2750, when no width is asked for
BW_HZ = 3000.0
READ_BYTES = 9600
CONFIRMED, TENTATIVE = "confirmed", "tentative"
GATEWAY_LIST = Path("winlink-vara-gateways.csv")

_HANDSHAKE = {"CR", "connect-response", "connected-ack"}
# Kinds that name no protocol. `one-hot burst` is here because its test is a
# spectral shape shared by every keyed narrowband carrier: graded VARA by the
# fallback below, it put 30 VARA lines on five corpus recordings that contain no
# VARA at all, the PACTOR-2 reference among them. A shape is not a protocol.
#
# `DBPSK burst` is here on the same measurement and a larger one. It is a
# differential-BPSK collapse on VARA's control sub-bands with no token pattern
# resolved, and over the shared regression corpus it fires 45 times across seven
# recordings holding no VARA — 16 on 80 m band noise, 19 across the three
# PACTOR-1 captures, 8 across two PACTOR-3 recordings, 2 on the ARDOP suspect —
# against 14 firings on the three VARA fixtures. A decoded control token still
# grades VARA below; this one decoded nothing.
_UNNAMED = {"unknown", "one-hot burst", "DBPSK burst"}


def normalize(t: float, kind: str, info: str = "", quality: str = "",
              gateway: str = "", caller: str = "",
              offset_hz: float | None = None) -> dict:
    """A kestrel ``classify`` Result -> a normalised detection dict."""
    detail = "  ".join(p for p in (info, f"({quality})" if quality else "") if p)
    if caller:
        grade, proto, station, role = CONFIRMED, "VARA", caller, "caller"
    elif gateway:
        grade, proto, station, role = CONFIRMED, "VARA", gateway, "gateway"
    elif kind == "DATA over":
        grade, proto, station, role = CONFIRMED, "VARA", "", ""
    elif kind in _HANDSHAKE or kind in ("wideband", "long burst", "high-SL over"):
        grade, proto, station, role = TENTATIVE, "VARA", "", ""
    elif kind in _UNNAMED:
        grade, proto, station, role = TENTATIVE, "UNKNOWN", "", ""
    else:                              # decoded DBPSK control token: link is live
        grade, proto, station, role = TENTATIVE, "VARA", "", ""
    return {"t": round(t, 3), "modem": "kestrel", "protocol": proto,
            "kind": kind, "grade": grade, "station": station, "role": role,
            "detail": detail,
            "offset_hz": None if offset_hz is None else round(offset_hz, 1)}


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _load_calls(vm) -> list:
    """Gateway callsigns for handshake attribution, from the tree's cached export
    only — a monitor must not stall on a network fetch. No cache means no
    attribution, so handshake bursts land as TENTATIVE, which is the honest grade
    for them."""
    try:
        export = GATEWAY_LIST.read_text()
    except OSError:
        return []
    return list(dict.fromkeys(call for call, _ in vm.vara_gateways(export)))


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from hfmodem.kestrel import monitor as vm

    from hfmodem.monitor.channel import tuned

    ap = argparse.ArgumentParser()
    # A wideband over is a grid-lock plus up to ~26 turbo passes — seconds of CPU
    # per burst. On live audio that stalls the loop and backs the stream up, so it
    # is off by default and the over is reported as an undecoded TENTATIVE signal.
    # Offline over a recording there is no such pressure, and skipping it means
    # never exercising the decode that actually reads payload.
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--centre", type=float)
    ap.add_argument("--bw", type=float, default=BW_HZ)
    args = ap.parse_args(argv)
    deep = args.deep
    if args.centre is not None:
        # Moved by whole tone bins, a request on the grid stays on it and its
        # offset reads exactly; moved by the aggregator's 100 Hz, it lands a few
        # Hz off every bin and reads a bin either side.
        args.centre = 1500.0 + round((args.centre - 1500.0) / TONE_HZ) * TONE_HZ
    chan = tuned(args.centre, args.bw)
    shift = chan.offset if chan else (lambda hz: hz)

    calls = _load_calls(vm)
    # The traffic log's gate, not the connect tool's. `Segmenter`'s 4.0x enter
    # threshold is calibrated for MFSK handshake bursts, which put all their
    # power in one tone and stand ~12 dB proud of the broadband floor; a
    # wideband over spreads the same power across the whole passband and lifts
    # frame RMS far less. Measured 2026-08-10 on this station's receiver at
    # 7101.5 kHz, a channel carrying a live VARA session a KiwiSDR had just
    # decoded 14 frames on: `Segmenter` bracketed 1 burst in 30 s and the
    # monitor called the first two windows quiet; `MonitorGate` (2.5x, and
    # mute-aware) bracketed 13 and heard the session's DBPSK control bursts.
    seg = vm.MonitorGate()
    # The gate is not the instrument for connect-request and connect-response, and
    # this runner ran without the scanner until 2026-08-14. Measured that day on a
    # 546 s recording of a VARA 500 exchange whose two callsigns an independent
    # source names: the gate path produced 0 kestrel detections over the whole
    # recording, and the scanner read two connect requests off the same audio, at
    # 4.35 s and 286.18 s. The morning slot had already published "no connect
    # handshake captured" from a gate-only pass that could not have seen one. So
    # the scanner runs unconditionally rather than under --deep: it costs ~9 s per
    # 546 s of audio against 340 candidate callsigns, which is 60x real time and
    # nowhere near backing a live stream up.
    scan = vm.HandshakeScanner(calls)
    # A track's centroid is the modem's centre only when the track holds that one
    # emission. A BW500 request 900 Hz up the 2026-07-26 tape shares its track
    # with a neighbour's overs, the centroid lands 153 Hz under it, and a scan
    # of +/-70 Hz from there misses it; the tiles cover the channel instead.
    tiles = [(0.0, None, scan)]
    if chan:
        from hfmodem.monitor.channel import Channel, NOMINAL_HZ

        reach = int(max(0.0, args.bw / 2 - TILE_EDGE_HZ) // TILE_HZ)
        tiles += [(k * TILE_HZ, Channel(args.centre + k * TILE_HZ, NOMINAL_HZ,
                                        TILE_EDGE_HZ + TILE_HZ / 2),
                   vm.HandshakeScanner(calls))
                  for k in sorted(range(-reach, reach + 1), key=abs) if k]

    def _gate(bracketed) -> list:
        """Classified gate brackets, less the two kinds the scanner owns — so a
        bracket that happens to land on a handshake is not reported twice, once
        from a worse alignment."""
        named = ((start, vm.classify(burst, calls, decode_wideband=deep,
                                     level_db=seg.level_db(burst)))
                 for start, burst in bracketed)
        return [(start, r) for start, r in named if r.kind not in vm.SCANNED]

    def _emit_all(events: list) -> None:
        for start, r in sorted(events, key=lambda e: e[0]):
            _emit(normalize(start / FS, r.kind, r.info, r.quality,
                            r.gateway, r.caller, shift(r.offset_hz)))

    def _scan(x, own, flush: bool = False) -> list:
        """Every tile's handshakes, each once, the nearest tile's reading kept.
        ``x`` is the input, ``own`` the channel's output of it."""
        out = []
        for moved, tile, scanner in tiles:
            heard = own if tile is None else tile.flush() if flush else tile.push(x)
            for start, r in scanner.push(heard):
                if moved and r.offset_hz is not None:
                    r.offset_hz += moved
                if not any(q.kind == r.kind and abs(s - start) < DUP_S * FS
                           for s, q in out):
                    out.append((start, r))
        return out

    def heard() -> float:
        """Where everything before has been said: the scanners' scored windows,
        and the pre-roll of a burst the gate still holds open."""
        at = min(scanner._t0 for _, _, scanner in tiles)
        if seg.inb:
            at = min(at, seg.start - seg.pad * vm.SEG_FRAME)
        return at / FS

    said = 0.0
    stdin = sys.stdin.buffer
    while True:
        raw = stdin.read(READ_BYTES)
        if not raw:
            break
        raw_x = np.frombuffer(raw, "<i2").astype(float) / 32768.0
        x = chan.push(raw_x) if chan else raw_x
        _emit_all(_gate(seg.push(x)) + _scan(raw_x, x))
        if chan and heard() >= said + HEARD_EVERY_S:
            said = heard()
            _emit({"heard": round(said, 3)})
    if chan:
        x = chan.flush()
        _emit_all(_gate(seg.push(x)) + _scan(None, x, flush=True))
    _emit_all(_gate(seg.flush()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
