# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""shrike monitor runner: live s16le on stdin -> normalised detection JSON out.

This is the whole shrike adapter. It drives the *real* decode path —
``hfmodem.shrike.rxfront.decode_events``, the same front end the live modem
receiver uses — so nothing here reimplements DSP.

rxfront decodes a whole buffer at once, so the stream is carried as a run of
consecutive blocks: each stretch of fresh audio is decoded exactly once (with a
short lookback so a burst on a block boundary still resolves), events deduped
across that lookback. This avoids the redundant re-decode a sliding window would
do — which matters because the front end runs an expensive header/CS scan at
every hop a pilot is present, so on a *continuously busy* channel it decodes
slower than real time. When the live source outruns it, the runner skips the
oldest un-decoded audio to stay current: a monitor answering "what is on now"
would rather be current than complete. (This front end is being unified with
shrike's softmodem receiver; a faster entry point drops straight in here, and
the block loop stays as is.)

``normalize`` is pure and carries the confidence policy: a PACTOR-3 header or a
control signal or a PACTOR-1 callsign decoded to certainty is CONFIRMED; a shape
that was measured but not decoded is TENTATIVE, and carries shrike's own wording
for what it did and did not establish rather than a summary of it.
"""

from __future__ import annotations

import json
import sys

FS = 48000
BLOCK_S = 6.0                   # decode this much fresh audio at a time
LOOKBACK_S = 1.5               # re-cover the prior boundary so a straddling burst decodes
HOP_S = 0.5                     # rxfront detection hop; matches the offline monitor
                                # (finer catches the link-setup callsign, which is
                                # the point of a monitor — at higher per-block cost)
MAX_LAG_S = 30.0                # un-decoded backlog past this is skipped to stay current
DEDUP_S = 0.8                   # same kind within this of an emitted event -> dup
READ_BYTES = 9600               # ~0.1 s of s16le

CONFIRMED, TENTATIVE = "confirmed", "tentative"

# `detect` is absent on purpose. Its only emitter is shrike's PACTOR-2 carrier
# pair, which this table used to grade PACTOR-3 -- a fallback default reading as a
# confident wrong answer on every line. rxfront now sets `Event.protocol` there and
# the decoder's own answer wins below; anything that reaches this table without one
# lands as UNKNOWN, which is what a kind with no protocol behind it means.
_PROTO = {"packet": "PACTOR-3", "cs": "PACTOR-3",
          "connect": "PACTOR-1", "fsk": "PACTOR-1", "p1reply": "PACTOR-1"}
_GRADE = {"packet": CONFIRMED, "cs": CONFIRMED, "connect": CONFIRMED,
          "detect": TENTATIVE, "fsk": TENTATIVE, "p1reply": TENTATIVE}
# P1-BURST and P2-PAIR, not P1-REPLY and DETECT: neither test establishes a reply
# or a mode, and the tag is the part an operator reads. See `rxfront.EVENT_KINDS`.
_TAG = {"packet": "HEADER", "detect": "P2-PAIR", "cs": "CS",
        "connect": "CONNECT", "fsk": "P1-FSK", "p1reply": "P1-BURST"}


def normalize(t: float, kind: str, text: str, station: str = "",
              protocol: str | None = None) -> dict:
    """A shrike rxfront event -> a normalised detection dict.

    An unrecognised kind is reported, not fatal. The modems gain event kinds as
    their receivers grow (``p1reply`` arrived this way), and a KeyError here used
    to kill the whole runner mid-capture — losing every later detection, including
    the ones that prompted the new kind in the first place. An unknown event is
    surfaced as TENTATIVE with its own name so it is visible rather than silently
    dropped."""
    # A control signal is not evidence of PACTOR-3. Both protocols have one and
    # they are different waveforms — PACTOR-1's is four 12-bit codewords in FSK,
    # PACTOR-3's is six 20-bit codewords in DBPSK on tones 5 and 12 — so mapping
    # the KIND to a protocol reported "PACTOR-3 active (link decoded)" for an
    # exchange that never left PACTOR-1.
    #
    # The decoder's own answer wins. `Event.protocol` carries it for the kinds
    # where the kind alone cannot; the text sniff behind it is a fallback for an
    # older shrike that predates the field, and is the reason this was ever
    # readable at all -- it was the only place the truth appeared.
    proto = protocol or _PROTO.get(kind, "UNKNOWN")
    if protocol is None and kind == "cs" and "PACTOR-1" in text:
        proto = "PACTOR-1"
    return {"t": round(t, 3), "modem": "shrike",
            "protocol": proto,
            "kind": _TAG.get(kind, kind.upper()),
            "grade": _GRADE.get(kind, TENTATIVE),
            "station": station, "detail": text}


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> int:
    import numpy as np
    from hfmodem.shrike import rxfront

    buf = np.zeros(0, dtype=float)
    base = 0                       # global sample index of buf[0]
    done = 0                       # samples from base already decoded
    emitted: list[tuple[float, str]] = []
    block, lookback = int(BLOCK_S * FS), int(LOOKBACK_S * FS)

    def decode(lo: int, hi: int) -> None:
        nonlocal emitted
        if hi - lo < FS // 2:
            return
        t0 = (base + lo) / FS
        for ev in rxfront.decode_events(buf[lo:hi], hop_s=HOP_S):
            t = t0 + ev.t
            if any(k == ev.kind and abs(t - pt) < DEDUP_S for pt, k in emitted):
                continue
            emitted.append((t, ev.kind))
            station = getattr(ev.connect, "callsign", "") if ev.connect else ""
            _emit(normalize(t, ev.kind, ev.text, station,
                            getattr(ev, "protocol", None)))
        cutoff = (base + hi) / FS - (BLOCK_S + LOOKBACK_S)
        emitted = [(t, k) for t, k in emitted if t > cutoff]

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
        x = np.frombuffer(raw, "<i2").astype(float) / 32768.0
        buf = np.concatenate([buf, x])
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
