# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel monitor runner: live s16le on stdin -> normalised detection JSON out.

This is the whole kestrel adapter. It reuses ``tools/vara_monitor``'s own
streaming pieces — the ``MonitorGate`` that brackets keyed bursts exactly as
that tool's own traffic log does, ``classify`` that names each one, and the
``HandshakeScanner`` that reads connect-request and connect-response off the raw
stream — so the detections are the same set the vara_monitor log path produces.
Wideband overs are not decoded here (that is a ~20 s job that would back the live
stream up), matching what vara_monitor does on ``--device``.

``normalize`` is pure and carries the confidence policy: a resolved caller or a
gateway matched against the Winlink list, or a CRC-valid DATA over, is CONFIRMED;
a recognised VARA waveform whose destination was not in the list, a bare control
token, or an undecoded wideband over is TENTATIVE — a real signal, unproven
identity. An unrecognised burst names no protocol at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FS = 48000
READ_BYTES = 9600
CONFIRMED, TENTATIVE = "confirmed", "tentative"

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
              gateway: str = "", caller: str = "") -> dict:
    """A kestrel ``classify`` Result -> a normalised detection dict."""
    detail = "  ".join(p for p in (info, f"({quality})" if quality else "") if p)
    if caller:
        grade, proto, station, role = CONFIRMED, "VARA", caller, "caller"
    elif gateway:
        grade, proto, station, role = CONFIRMED, "VARA", gateway, "gateway"
    elif kind == "DATA over":
        grade, proto, station, role = CONFIRMED, "VARA", "", ""
    elif kind in _HANDSHAKE or kind in ("wideband", "long burst"):
        grade, proto, station, role = TENTATIVE, "VARA", "", ""
    elif kind in _UNNAMED:
        grade, proto, station, role = TENTATIVE, "UNKNOWN", "", ""
    else:                              # decoded DBPSK control token: link is live
        grade, proto, station, role = TENTATIVE, "VARA", "", ""
    return {"t": round(t, 3), "modem": "kestrel", "protocol": proto,
            "kind": kind, "grade": grade, "station": station, "role": role,
            "detail": detail}


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _load_calls(vm) -> list:
    """Gateway callsigns for handshake attribution, from a cached CSV only —
    a monitor must not stall on a network fetch. No cache means no attribution,
    so handshake bursts land as TENTATIVE, which is the honest grade for them."""
    if not vm.GATEWAY_LIST.exists():
        return []
    try:
        return vm.load_gateways(vm.GATEWAY_LIST, None, False)
    except Exception:
        return []


def main() -> int:
    # `from tools import vara_monitor` does not work: tools/ has no __init__.py
    # and vara_monitor imports its siblings flat, so the package form raises
    # ModuleNotFoundError on hfcapture. Putting tools/ on the path directly is
    # what resolves both it and its siblings.
    sys.path.insert(0, str(Path.cwd() / "tools"))
    import numpy as np
    import vara_monitor as vm

    # A wideband over is a grid-lock plus up to ~26 turbo passes — seconds of CPU
    # per burst. On live audio that stalls the loop and backs the stream up, so it
    # is off by default and the over is reported as an undecoded TENTATIVE signal.
    # Offline over a recording there is no such pressure, and skipping it means
    # never exercising the decode that actually reads payload.
    deep = "--deep" in sys.argv

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
                            r.gateway, r.caller))

    stdin = sys.stdin.buffer
    while True:
        raw = stdin.read(READ_BYTES)
        if not raw:
            break
        x = np.frombuffer(raw, "<i2").astype(float) / 32768.0
        _emit_all(_gate(seg.push(x)) + scan.push(x))
    _emit_all(_gate(seg.flush()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
