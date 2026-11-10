# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-4 monitor runner: live s16le on stdin -> normalised detection JSON out.

This is the whole PACTOR-4 adapter. The entry chirp is acquired blind by the
corpus collector's detector (``tools/corpuslib/detector.chirp_events``: a
dual-lane header search whose timing and carrier offset must agree, then the
chirp's own CRC reader), so this runner needs the source tree beside it and the
startup probe says so where it is not.

The stream is carried as 4 s blocks with 4 s of lookback, which is longer than
the 3.3 s chirp, so a chirp that straddles a boundary is read whole in the next
window and not reported cut in this one.

``normalize`` carries the confidence policy: a chirp whose field passed its
CRC, and a burst the word bank names, are CONFIRMED. A header that matched
without a readable field, the SF-16 spreading statistic over its knee, and a
burst `p4oracle` can only put in a family, are TENTATIVE.

A block whose spread clears the knee is read burst by burst with `p4oracle`, the
arm's own reply reader: `P4-WORD` for a bank word (CALLER20, GATEWAY,
BREAKIN), `P4-ROBUST-SHORT` / `P4-ROBUST-CS` / `P4-ROBUST+BLOCK` /
`P4-HEADER` for robust bursts by length, `P4-NORMAL` for a training lock. Over
PACTORIVaudio1 that names the caller's three CALLER20 headers and the
answerer's three answer words, where the spread alone said PACTOR-4 twice. It costs 1.4-5.7 s a four-second block, and only where the
spread already fired.

``--centre HZ`` makes this a track's instance: the audio is cut to ``--bw`` around
that centre and moved to 1500 Hz, and the offset is reported from where the
track sits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hfmodem.monitor import FS

#: The chirp's sweep and its skirts, when no width is asked for.
BW_HZ = 2800.0
READ_BYTES = 9600               # ~0.1 s of s16le
BLOCK_S = 4.0
LOOKBACK_S = 4.0
MAX_OFFSET_HZ = 60.0            # the collector's acceptance: past this is another channel
#: A track's centroid is not the chirp's centre until the chirp has run: over the
#: four recorded Dragon chirps the first track report sat 180-218 Hz off it, and
#: the channel has already cut every other signal away, so the search widens.
TRACK_OFFSET_HZ = 320.0
SPREAD_WIN_S = 0.5              # the window `p4sig.SPREAD_KNEE` was measured on
SPREAD_HOP_S = 0.25
DEDUP_S = 0.75
MAX_LAG_S = 30.0

CONFIRMED, TENTATIVE = "confirmed", "tentative"


def normalize(t: float, kind: str, detail: str = "",
              offset_hz: float | None = None, grade: str | None = None) -> dict:
    """A PACTOR-4 sighting -> a normalised detection dict."""
    return {"t": round(t, 3), "modem": "p4", "protocol": "PACTOR-4",
            "kind": kind,
            "grade": grade or (CONFIRMED if kind == "P4-CHIRP-CRC" else TENTATIVE),
            "station": "", "detail": detail,
            "offset_hz": None if offset_hz is None else round(offset_hz, 1)}


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def bursts(audio, t0: float, final: bool) -> tuple[list[dict], list[str]]:
    """Each PACTOR-4 burst in ``audio`` named by `p4oracle`, and the classes of
    the rest. A burst running into the block's end waits for the next block,
    whose lookback holds it whole. No offset: the Chu fit's carrier trades
    against its shift (`p4sig.header_score`), so it places nothing."""
    from hfmodem.shrike import p4oracle

    named, other = [], []
    for row in p4oracle.read_window(audio, FS):
        cls = row["reply_class"]
        onset, dur = row["onset_ms"] / 1e3, row["duration_ms"] / 1e3
        if not final and (onset + dur) * FS >= len(audio) - FS // 20:
            continue
        if not cls.startswith("p4-") or cls.endswith("?"):
            other.append(cls)
            continue
        kind = cls.partition(":")[0].upper()
        note = p4oracle.word_note(row)
        named.append(normalize(
            t0 + onset, kind,
            f"{cls}  {row['duration_ms']:.0f} ms, spread {row['spread_score']:.2f}"
            + (f", {note}" if note else ""),
            grade=CONFIRMED if kind == "P4-WORD" and row["word"] else None))
    return named, other


def spread(audio, t0: float) -> dict | None:
    """The strongest SF-16 window in ``audio`` over the knee, as a tentative
    sighting with no offset: the statistic aliases every 112.5 Hz, and the Chu
    header search that would refine it pins the root, not the carrier."""
    from hfmodem.shrike import p4sig

    win, hop = int(SPREAD_WIN_S * FS), int(SPREAD_HOP_S * FS)
    score, at = max(((p4sig.spread_score(audio[i:i + win]), i)
                     for i in range(0, len(audio) - win + 1, hop)), default=(0.0, 0))
    if score < p4sig.SPREAD_KNEE:
        return None
    return normalize(t0 + at / FS, "P4-SPREAD", f"SF-16 spread {score:.3f}")


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path.cwd() / "tools"))
    import numpy as np
    from corpuslib import detector
    from hfmodem.shrike import p4chirp

    from hfmodem.monitor.channel import tuned

    ap = argparse.ArgumentParser()
    ap.add_argument("--centre", type=float)
    ap.add_argument("--bw", type=float, default=BW_HZ)
    args = ap.parse_args(argv)
    chan = tuned(args.centre, args.bw)
    shift = chan.offset if chan else (lambda hz: hz)
    search = TRACK_OFFSET_HZ if chan else MAX_OFFSET_HZ

    buf = np.zeros(0)
    base = 0                       # global sample index of buf[0]
    done = 0                       # samples from base already decoded
    emitted: list[tuple[float, str]] = []
    block, lookback = int(BLOCK_S * FS), int(LOOKBACK_S * FS)
    whole = round(p4chirp.PACKET_S * FS)

    def family(kind: str) -> str:
        return "P4-CHIRP" if kind.startswith("P4-CHIRP") else kind

    def decode(lo: int, hi: int, final: bool) -> None:
        nonlocal emitted
        audio = buf[lo:hi]
        t0 = (base + lo) / FS
        found = [normalize(t0 + ev["t"], ev["kind"], ev["detail"],
                           shift(ev["offset_hz"]))
                 for ev in detector.chirp_events(audio, search)
                 if final or ev["t"] * FS + whole <= len(audio)]
        if (sp := spread(buf[max(lo, done):hi], (base + max(lo, done)) / FS)):
            named, other = bursts(buf[lo:hi], t0, final)
            found += named
            if not named:
                found.append(dict(sp, detail=sp["detail"] + (
                    f"; its bursts read {', '.join(sorted(set(other)))}"
                    if other else "")))
        for det in sorted(found, key=lambda d: d["t"]):
            t, k = det["t"], family(det["kind"])
            if not any(k == pk and abs(t - pt) < DEDUP_S for pt, pk in emitted):
                emitted.append((t, k))
                _emit(det)
        if chan:
            _emit({"heard": round((base + hi) / FS - LOOKBACK_S, 3)})
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
        buf = np.concatenate([buf, chan.push(x) if chan else x])
        if len(buf) - done > MAX_LAG_S * FS:
            skipped = (len(buf) - block) - done
            done = len(buf) - block
            print(f"falling behind; skipped {skipped / FS:.1f}s to stay current",
                  file=sys.stderr, flush=True)
        while len(buf) - done >= block:
            lo = max(0, done - lookback)
            hi = done + block
            decode(lo, hi, final=False)
            done = hi
            compact()
    if chan:
        buf = np.concatenate([buf, chan.flush()])
    if len(buf) > done:
        decode(max(0, done - lookback), len(buf), final=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
