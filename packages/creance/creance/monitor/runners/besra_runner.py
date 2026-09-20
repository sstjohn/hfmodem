# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra monitor runner: live s16le on stdin -> normalised detection JSON out.

This is the whole besra adapter. It drives the *real* decode path —
``hfmodem.besra.phy.demodulator.Demodulator``, the same demodulator the live
receiver and ``python -m hfmodem.besra.monitor`` run — so nothing here
reimplements DSP. ARDOP is a 12 kHz waveform and the stream arrives at 48 kHz,
so each window is decimated 4:1 on its way in.

The demodulator locates every frame in a buffer by its leader, so the stream is
carried as consecutive blocks with a lookback wide enough for the longest ARDOP
frame (5.5 s, ``4FSK.2000.600``) to fall wholly inside one window, frames deduped
across the overlap.

``normalize`` is pure and carries the confidence policy, and it is deliberately
strict. An ARDOP frame is named by ten 4FSK symbols, and VARA and PACTOR energy
reaches that bar often enough that besra needs a per-frame-class quality floor to
hold them off. So only a frame whose *content* validated behind the header — a
data frame's RS and payload CRC, or a ConReq/ID/Ping's Packed6 callsign — is
CONFIRMED. A bare control or ack frame has nothing behind the header to check, so
it is TENTATIVE, and a frame that failed its integrity check is not a detection
at all.
"""

from __future__ import annotations

import json
import sys

FS = 48000
BESRA_FS = 12000
READ_BYTES = 9600               # ~0.1 s of s16le
BLOCK_S = 6.0                   # decode this much fresh audio at a time
LOOKBACK_S = 6.0                # > the longest ARDOP frame, so one straddling a
                                # block boundary is whole in the next window
DEDUP_S = 0.5                   # same frame within this of an emitted one -> dup
MAX_LAG_S = 30.0                # un-decoded backlog past this is skipped to stay current

CONFIRMED, TENTATIVE = "confirmed", "tentative"

#: Frames with nothing behind the frame-type header to corroborate it: no
#: callsign, no payload CRC, at most three body bytes carrying a majority vote.
#: The header is effectively their whole proof, so they never grade better than
#: tentative. ConAck* joins them by prefix.
_BARE = {"BREAK", "IDLE", "DISC", "END", "ConRejBusy", "ConRejBW",
         "DATAACK", "DATANAK", "PingAck"}


def normalize(t: float, name: str, ok: bool, detail: str = "",
              station: str = "") -> dict | None:
    """A besra ``DecodedFrame`` -> a normalised detection dict, or None for a
    frame that proved nothing."""
    if not ok:
        return None                    # integrity said no; that is not a detection
    bare = name in _BARE or name.startswith("ConAck")
    grade = TENTATIVE if bare else CONFIRMED
    role = "caller" if name.startswith("ConReq") or name == "Ping" else ""
    return {"t": round(t, 3), "modem": "besra", "protocol": "ARDOP",
            "kind": name, "grade": grade, "station": station, "role": role,
            "detail": detail}


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _detail(f) -> str:
    """The same one-line summary ``besra.monitor`` prints for each frame class."""
    if f.name.startswith("ConReq") or f.name == "Ping":
        return f"> {f.target}"
    if f.name == "IDFrame":
        return f.grid or ""
    if f.name.startswith("ConAck"):
        return f"leader {f.conack_timing_ms} ms"
    if f.name == "PingAck":
        return f"S/N {f.pingack_sn_db} dB  Q {f.pingack_quality}"
    return f"{len(f.payload)} B" if f.payload else ""


def main() -> int:
    import numpy as np
    from scipy.signal import resample_poly
    from hfmodem.besra.phy.demodulator import Demodulator

    demod = Demodulator()
    buf = np.zeros(0, dtype=float)
    base = 0                       # global sample index of buf[0], at FS
    done = 0                       # samples from base already decoded
    emitted: list[tuple[float, str]] = []
    block, lookback = int(BLOCK_S * FS), int(LOOKBACK_S * FS)

    def decode(lo: int, hi: int) -> None:
        nonlocal emitted
        if hi - lo < FS // 2:
            return
        t0 = (base + lo) / FS
        for f in demod.decode(resample_poly(buf[lo:hi], 1, FS // BESRA_FS)):
            t = t0 + f.offset / BESRA_FS
            if any(n == f.name and abs(t - pt) < DEDUP_S for pt, n in emitted):
                continue
            det = normalize(t, f.name, f.ok, _detail(f), f.caller or "")
            if det:
                # Only a detection takes a slot. The lookback is there so a frame
                # cut by a block boundary is whole in the next window, and the cut
                # copy fails integrity -- so recording it here let the truncated
                # sighting suppress the whole one and the frame was never reported
                # at all, `normalize` having refused the only copy that reached it.
                emitted.append((t, f.name))
                _emit(det)
        cutoff = (base + hi) / FS - (BLOCK_S + LOOKBACK_S)
        emitted = [(t, n) for t, n in emitted if t > cutoff]

    def compact() -> None:
        nonlocal buf, base, done
        keep = max(0, done - lookback)
        if keep:
            buf = buf[keep:]
            base += keep
            done -= keep

    stdin = sys.stdin.buffer
    while True:
        raw = stdin.read(READ_BYTES)
        if not raw:
            break
        buf = np.concatenate([buf, np.frombuffer(raw, "<i2").astype(float)])
        if len(buf) - done > MAX_LAG_S * FS:      # slower than realtime: skip stale
            skipped = (len(buf) - block) - done
            done = len(buf) - block
            print(f"falling behind; skipped {skipped / FS:.1f}s to stay current",
                  file=sys.stderr, flush=True)
        while len(buf) - done >= block:
            lo = max(0, done - lookback)
            hi = done + block
            decode(lo, hi)
            done = hi
            compact()
    if len(buf) > done:
        decode(max(0, done - lookback), len(buf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
