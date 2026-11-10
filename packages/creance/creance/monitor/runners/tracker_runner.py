# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""tracker runner: live s16le on stdin -> track records as JSON lines out.

Every emission in the window, whether or not a modem claims it, from
``hfmodem.monitor.tracker``: a ``track`` record for each live track every half
second and a ``track-end`` as it ends, graded TENTATIVE always, because a shape
is a description and never an identification.

Every track narrower than 60 Hz is also handed to the CW reader, from the audio
of the burst that opened it, and the reader's text records go out beside the
tracker's.

``--span LO HI`` sets the window in audio Hz; the rig's filter, 300 to 2900,
when it is not given. ``--heard`` adds, each second, ``{"heard": s}``: every
emission that began before ``s`` has been reported.
"""

from __future__ import annotations

import argparse
import json
import sys

from hfmodem.monitor import FS

READ_BYTES = 9600
TENTATIVE = "tentative"
CW_BELOW_HZ = 60.0
#: Audio kept for a CW reader's first push. A track is reported up to a
#: couple of seconds after its first burst began, and the reader needs that
#: burst.
RING_S = 5.0
#: How long after its first burst a track may first be reported: 2.4 s is
#: typical, and five the most measured across the labelled corpus.
REPORT_LAG_S = 6.0


def normalize(t: float, kind: str, track: int, t_start: float, centre_hz: float,
              lo_hz: float, hi_hz: float, level_dbfs: float, over_db: float,
              crest_db: float | None,
              cadence_s: float | None, label: str, bursts: int, active_s: float) -> dict:
    """A tracker ``TrackReport``'s fields -> a normalised detection dict."""
    bw = hi_hz - lo_hz + FS / 4096
    crest = "" if crest_db is None else f", crest {crest_db:.1f} dB"
    out = {"t": round(t, 3), "modem": "tracker", "protocol": label, "kind": kind,
           "grade": TENTATIVE,
           "detail": (f"{lo_hz:.0f}-{hi_hz:.0f} Hz since {t_start:.2f} s, "
                      f"{bursts} bursts over {active_s:.1f} s{crest}"),
           "centre_hz": round(centre_hz, 2), "bw_hz": round(bw, 1),
           "level_dbfs": level_dbfs, "over_db": over_db, "track": track,
           "t_start": t_start}
    if cadence_s is not None:
        out["cadence_s"] = cadence_s
    return out


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    from dataclasses import asdict

    import numpy as np
    from hfmodem.monitor.cw import CwReader
    from hfmodem.monitor.tracker import Tracker

    ap = argparse.ArgumentParser()
    ap.add_argument("--span", nargs=2, type=float, default=(300.0, 2900.0),
                    metavar=("LO", "HI"))
    ap.add_argument("--heard", action="store_true")
    args = ap.parse_args(argv)
    tracker = Tracker(*args.span)
    readers: dict[int, tuple[object, float]] = {}
    ring = np.zeros(0)
    n = 0

    def cw(rep) -> None:
        if rep.track in readers:
            reader, t0 = readers[rep.track]
            out = []
        elif rep.bw_hz < CW_BELOW_HZ:
            start = n - len(ring)
            t0 = max(rep.t_start - 0.2, start / FS)
            reader = CwReader(FS, rep.centre_hz)
            readers[rep.track] = (reader, t0)
            out = reader.push(ring[int(round(t0 * FS)) - start:])
        else:
            return
        if rep.kind == "track-end":
            out += reader.flush()
            del readers[rep.track]
        for rec in out:
            _emit(rec | {"t": round(rec["t"] + t0, 3), "track": rep.track})

    def reports(reps) -> None:
        for rep in reps:
            _emit(normalize(**asdict(rep)))
            cw(rep)

    stdin = sys.stdin.buffer
    carry = b""
    while True:
        raw = stdin.read(READ_BYTES)
        if not raw:
            break
        raw, carry = carry + raw, b""
        if len(raw) % 2:
            raw, carry = raw[:-1], raw[-1:]
        x = np.frombuffer(raw, "<i2").astype(float) / 32768.0
        ring = np.concatenate([ring, x])[-int(RING_S * FS):]
        for track, (reader, t0) in list(readers.items()):
            for rec in reader.push(x):
                _emit(rec | {"t": round(rec["t"] + t0, 3), "track": track})
        reps = tracker.push(n, x)
        n += len(x)
        reports(reps)
        if args.heard and n // FS > (n - len(x)) // FS:
            _emit({"heard": round(n / FS - REPORT_LAG_S, 3)})
    reports(tracker.flush())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
