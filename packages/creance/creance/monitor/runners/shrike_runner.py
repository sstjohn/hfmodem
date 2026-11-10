# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""shrike monitor runner: live s16le on stdin -> normalised detection JSON out.

This is the whole shrike adapter. It drives the *real* decode path —
``hfmodem.shrike.rxfront.decode_events``, the same front end the live modem
receiver uses — so nothing here reimplements a decoder. What it adds is where
each event sits: the PACTOR-1 pair fit (also the corpus collector's, which
imports it from here) and the PACTOR-2 marker's carrier pair.

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

``--centre HZ`` makes this a track's instance: the audio is cut to ``--bw`` around
that centre and moved to 1500 Hz, offsets are reported from where the track
sits, the P3 envelope pass is off as every stream caller has it, and
``--shape`` names the families the track's width can hold (see ``SHAPES``).

``normalize`` is pure and carries the confidence policy: a PACTOR-3 header or a
control signal or a PACTOR-1 callsign decoded to certainty is CONFIRMED; a shape
that was measured but not decoded is TENTATIVE, and carries shrike's own wording
for what it did and did not establish rather than a summary of it.
"""

from __future__ import annotations

import argparse
import json
import sys

from hfmodem.monitor import CALLSIGN, FS

BLOCK_S = 6.0                   # decode this much fresh audio at a time
LOOKBACK_S = 1.5               # re-cover the prior boundary so a straddling burst decodes
HOP_S = 0.5                     # rxfront detection hop; matches the offline monitor
                                # (finer catches the link-setup callsign, which is
                                # the point of a monitor — at higher per-block cost)
MAX_LAG_S = 30.0                # un-decoded backlog past this is skipped to stay current
COMPENSATE_HZ = 15.0            # a heard PACTOR-3 link this far off is moved home first
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

#: Both PACTOR-1 tones and both PACTOR-2 carriers sit either side of this.
CENTRE_HZ = 1500.0
#: PACTOR-3's widest level, tones 480-2520 Hz, and the skirts: a link that opens
#: in PACTOR-1 may be here a few cycles later.
BW_HZ = 2800.0
#: How much audio the FSK pair is fitted over: one control signal, or a second of
#: anything longer.
FSK_SPAN_S = {"cs": 0.18, "unassigned": 0.18}
#: A P2 marker's bin names its carrier pair: centre = 1200 + 25 * bin Hz.
P2_BIN_HZ, P2_BIN0_HZ = 25.0, 1200.0

#: What a track's instance searches, by the width the tracker measured (the
#: aggregator names the shape). The PACTOR-3 header and control signals ride
#: tones 5 and 12, 840 Hz apart, so a PACTOR-1 or PACTOR-2 wide track cannot
#: hold them; a wide one holds no PACTOR-1 control signal or PACTOR-2 marker. A
#: link that upgrades widens its track, and the aggregator then starts the wide
#: instance beside the narrow one, fed from before the track began. Measured
#: over `watch_pactor3_maryland.wav`, the P1 control-signal search, the P3 packet
#: pass and the P3 header search are 45 %, 39 % and 6 % of what an instance
#: spends with the P3 envelope pass off; that pass alone was 89 % of the
#: whole-band runner's 68 s over the tape's 25 s, and found nothing there or on
#: any labelled PACTOR fixture that the other passes did not.
SHAPES = {"pair": dict(p3_packets=False, p3_headers=False),
          "wide": dict(p1_controls=False, p2_markers=False)}


def protocol_of(kind: str, text: str, protocol: str | None = None) -> str:
    """The protocol an event reads as.

    A control signal is not evidence of PACTOR-3. Both protocols have one and
    they are different waveforms — PACTOR-1's is four 12-bit codewords in FSK,
    PACTOR-3's is six 20-bit codewords in DBPSK on tones 5 and 12 — so mapping
    the KIND to a protocol reported "PACTOR-3 active (link decoded)" for an
    exchange that never left PACTOR-1.

    The decoder's own answer wins. `Event.protocol` carries it for the kinds
    where the kind alone cannot; the text sniff behind it is a fallback for an
    older shrike that predates the field, and is the reason this was ever
    readable at all -- it was the only place the truth appeared."""
    proto = protocol or _PROTO.get(kind, "UNKNOWN")
    if protocol is None and kind == "cs" and "PACTOR-1" in text:
        proto = "PACTOR-1"
    return proto


def normalize(t: float, kind: str, text: str, station: str = "",
              protocol: str | None = None,
              offset_hz: float | None = None, cs: int | None = None,
              grade: str | None = None) -> dict:
    """A shrike rxfront event -> a normalised detection dict.

    An unrecognised kind is reported, not fatal. The modems gain event kinds as
    their receivers grow (``p1reply`` arrived this way), and a KeyError here used
    to kill the whole runner mid-capture — losing every later detection, including
    the ones that prompted the new kind in the first place. An unknown event is
    surfaced as TENTATIVE with its own name so it is visible rather than silently
    dropped.

    A PACTOR-1 connect frame carries no CRC; the structural lock behind it
    accepts 1,080 times ungated over the corpus, so it is CONFIRMED only when
    what it names is a callsign. A track's instance read ``KO`` off a VARA
    session."""
    grade = grade or _GRADE.get(kind, TENTATIVE)
    if kind == "connect" and not _callsign(station):
        grade = TENTATIVE
    proto = protocol_of(kind, text, protocol)
    tag = (f"CS{cs + 1}" if kind == "cs" and proto == "PACTOR-3" and cs is not None
           else _TAG.get(kind, kind.upper()))
    return {"t": round(t, 3), "modem": "shrike",
            "protocol": proto,
            "kind": tag,
            "grade": grade,
            "station": station, "detail": text, "offset_hz": offset_hz}


def _callsign(station: str) -> bool:
    return len(station) >= 3 and CALLSIGN.fullmatch(station) is not None


def fsk_offset(audio) -> float | None:
    """Where a PACTOR-1 pair sits against 1400/1600 Hz, fitted on both tones, not
    the centroid, which unequal mark/space duty biases. A coarse +/-400 Hz search
    tells an adjacent FSK station from one on the channel. None means no credible
    paired peak, never 'zero offset'."""
    import numpy as np
    from scipy.signal import welch

    if len(audio) < FS // 20:
        return None
    f, power = welch(audio, FS, nperseg=min(4096, len(audio)))
    offsets = np.arange(-400., 401., 5.)
    paired = np.minimum(np.interp(CENTRE_HZ - 100 + offsets, f, power),
                        np.interp(CENTRE_HZ + 100 + offsets, f, power))
    k = int(np.argmax(paired))
    floor = np.median(power[(f > 800) & (f < 2200)])
    if paired[k] < max(floor * 6.3, 1e-18):
        return None
    return float(offsets[k])


def p2_offset(audio) -> float | None:
    """The strongest armed PACTOR-2 marker's carrier pair against 1500 Hz."""
    from hfmodem.shrike import p2rx

    armed = [(score, b) for _, b, _, _, score, _ in p2rx.find_markers(audio, FS, 0.90)
             if score >= p2rx.MARKER_ARM]
    return P2_BIN0_HZ + P2_BIN_HZ * max(armed)[1] - CENTRE_HZ if armed else None


def offset_of(audio, at: int, kind: str, protocol: str) -> float | None:
    """The event's tuning error, read off the audio at sample ``at``. PACTOR-3 has
    no search for one and reports None."""
    if protocol == "PACTOR-1":
        return fsk_offset(audio[at:at + int(FSK_SPAN_S.get(kind, 1.0) * FS)])
    if protocol == "PACTOR-2":
        return p2_offset(audio[max(0, at - FS):at + FS])
    return None


class P3Controls:
    """PACTOR-3 control signals acquired blind, and graded by the cycle they keep.

    rxfront reads a control signal only where the header pilot points, which on
    a monitor is almost never: over `captures/onair-0923-1933`, a VE3KPG link in
    PACTOR-3 both ways with 54 control signals in the arm's log, it named one.
    `p3acquire.control_signal` is the arm's own entry-time search -- every
    codeword over +/-75 Hz, two adjacent timings agreeing, the burst's own
    extent -- and over the same audio it names 101 at 5 % of real time.

    A bare codeword has no CRC, and VARA's PSK can spell one: two VARA fixtures
    gave nine at +31 and +52 Hz, where a 600 s stretch of a busy 40 m band read
    at ten centres gave none. What they lack is PACTOR-3's shape and its
    cycle. `rxfront.p3_comb_db` over the codeword reads VARA at -2.1..+3.2 dB
    and real control signals at +6.8..+38 on a link heard here, but only
    -1.3..+11.5 on KB5LZK heard 726 miles off, whose codewords keep one
    1.25 s cycle to 30 ms thirty times running. So a codeword is said when its
    comb stands over `COMB_DB`, or when two earlier ones at its frequency sit
    whole cycles before it; that second test, or a CRC-valid changeover body,
    is what confirms one. VARA's nine made one such pair each and no three.
    """

    WIN_S, HOP_S = 2.0, 1.0      # windows the search is given; it refuses > 4 s
    CYCLE_S = 1.25               # a long cycle is three of these
    RASTER_S = 0.03
    SAME_HZ = 3.0
    CHAIN_S = 15.0
    COMB_DB = 5.0
    #: Two codewords this far apart on the cycle are two answer instants.
    PHASES_S = 0.06
    SESSION_EVERY_S = 60.0

    def __init__(self) -> None:
        self.heard: list[tuple[float, float]] = []      # (t, hz), every acquisition
        self.chained: list[tuple[float, float]] = []    # the ones that kept a cycle
        self.paired_at = -1e9

    def _on_cycle(self, dt: float) -> bool:
        return abs((dt + self.CYCLE_S / 2) % self.CYCLE_S - self.CYCLE_S / 2) <= self.RASTER_S

    def _earlier(self, t: float, hz: float) -> int:
        return sum(1 for pt, phz in self.heard
                   if 0.3 < t - pt <= self.CHAIN_S and abs(hz - phz) <= self.SAME_HZ
                   and self._on_cycle(t - pt))

    def read(self, audio, t0: float) -> list[tuple[float, object, float, str, str]]:
        """``(t, event, hz, grade, note)`` for each codeword said in ``audio``,
        which starts at stream second ``t0``."""
        from hfmodem.shrike import p3acquire, rxfront

        win, hop = int(self.WIN_S * FS), int(self.HOP_S * FS)
        out = []
        for lo in range(0, max(1, len(audio) - win + hop), hop):
            got = p3acquire.control_signal(audio[lo:lo + win])
            if got is None:
                continue
            at = lo + got.event.start
            t = t0 + at / FS
            if any(abs(t - pt) < 0.3 for pt, _ in self.heard):
                continue
            hz = got.offset_hz
            earlier = self._earlier(t, hz)
            self.heard.append((t, hz))
            comb = rxfront.p3_comb_db(audio[at:at + int(0.2 * FS)], FS, hz)
            body = got.event.packet is not None
            if not (body or earlier >= 2 or comb >= self.COMB_DB):
                continue
            if earlier:
                self.chained.append((t, hz))
            grade = CONFIRMED if body or earlier >= 2 else TENTATIVE
            note = (f"  comb {comb:+.1f} dB, {earlier} earlier on its "
                    f"{self.CYCLE_S:g} s cycle")
            out.append((t, got.event, hz, grade, note))
        self.heard = [(t, hz) for t, hz in self.heard if t > t0 - self.CHAIN_S]
        return out

    def offset_hz(self, t: float) -> float | None:
        """Where the chained codewords put this link, for the header search."""
        hz = sorted(h for pt, h in self.chained if t - pt <= self.CHAIN_S)
        return hz[len(hz) // 2] if hz else None

    def session(self, t: float) -> dict | None:
        """Two answer instants on one cycle inside `CHAIN_S`: the two stations
        of a session, heard as one. Said once a minute at most."""
        recent = [(pt, hz) for pt, hz in self.chained if t - pt <= self.CHAIN_S]
        self.chained = recent
        if t - self.paired_at < self.SESSION_EVERY_S:
            return None
        phase = lambda pt: pt % self.CYCLE_S
        for i, (ta, ha) in enumerate(recent):
            for tb, hb in recent[i + 1:]:
                gap = (phase(tb) - phase(ta)) % self.CYCLE_S
                gap = min(gap, self.CYCLE_S - gap)
                if gap > self.PHASES_S or abs(ha - hb) > 2 * self.SAME_HZ:
                    self.paired_at = t
                    return normalize(t, "session",
                                     f"two answer instants {gap * 1e3:.0f} ms apart "
                                     f"on one {self.CYCLE_S:g} s cycle, at {ha:+.1f} "
                                     f"and {hb:+.1f} Hz: both ends of one link",
                                     protocol="PACTOR-3", grade=TENTATIVE)
        return None


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from hfmodem.shrike import p3acquire, rxfront

    from hfmodem.monitor.channel import tuned

    ap = argparse.ArgumentParser()
    ap.add_argument("--centre", type=float)
    ap.add_argument("--bw", type=float, default=BW_HZ)
    ap.add_argument("--shape", choices=sorted(SHAPES))
    args = ap.parse_args(argv)
    chan = tuned(args.centre, args.bw)
    shift = chan.offset if chan else (lambda hz: hz)
    prune = dict(SHAPES.get(args.shape, {}), p3_envelope=False) if chan else {}
    p3 = P3Controls() if prune.get("p3_headers", True) else None
    buf = np.zeros(0, dtype=float)
    base = 0                       # global sample index of buf[0]
    done = 0                       # samples from base already decoded
    emitted: list[tuple[float, tuple[str, str]]] = []
    block, lookback = int(BLOCK_S * FS), int(LOOKBACK_S * FS)

    def decode(lo: int, hi: int) -> None:
        nonlocal emitted
        if hi - lo < FS // 2:
            return
        t0 = (base + lo) / FS

        def said(t: float, kind: str, proto: str) -> bool:
            if any(k == (kind, proto) and abs(t - pt) < DEDUP_S for pt, k in emitted):
                return True
            emitted.append((t, (kind, proto)))
            return False

        audio = buf[lo:hi]
        heard_hz = None
        if p3:
            for t, ev, hz, grade, note in p3.read(audio, t0):
                if not said(t, ev.kind, "PACTOR-3"):
                    _emit(normalize(t, ev.kind, ev.text + note, protocol="PACTOR-3",
                                    offset_hz=shift(hz), cs=ev.cs, grade=grade))
            if (session := p3.session((base + hi) / FS)):
                _emit(session)
            heard_hz = p3.offset_hz((base + hi) / FS)
        # A track's channel is centred to a 100 Hz grid on a centroid, and the
        # header search gives out 30 Hz off: measured over the VE3KPG link, 16
        # headers at -15..+15 Hz, 14 at +30, 7 at +45.
        if chan and heard_hz is not None and abs(heard_hz) >= COMPENSATE_HZ:
            audio = p3acquire.compensate(audio, heard_hz)
        for ev in rxfront.decode_events(audio, hop_s=HOP_S, **prune):
            t = t0 + ev.t
            proto = protocol_of(ev.kind, ev.text, getattr(ev, "protocol", None))
            if said(t, ev.kind, proto):
                continue
            station = getattr(ev.connect, "callsign", "") if ev.connect else ""
            offset = shift(offset_of(buf, lo + max(0, int(ev.t * FS)), ev.kind, proto))
            _emit(normalize(t, ev.kind, ev.text, station, proto, offset, cs=ev.cs))
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
    if chan:
        buf = np.concatenate([buf, chan.flush()])
    if len(buf) > done:
        decode(max(0, done - lookback), len(buf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
