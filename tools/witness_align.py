"""Measure a session recording's timebase against an independent receiver's.

A dropped block leaves no trace in the file it is missing from. The cycle grid,
the window indices and `stream.wav` all count DELIVERED samples, so they stay
perfectly consistent with each other while the whole arithmetic slides against
the air, and `capture clock: ... ppm` reads the shortfall as a crystal. Nothing
inside one recording can separate the two.

A second receiver on a second clock can. Cross-correlating a key-down mask in a
sliding window gives lag(t) against the witness: a rate error is a straight
line, and lost capture is a STAIRCASE -- flat runs at correlation near 1 with
steps between them, each step the samples that went missing.

Key-down, not amplitude. Two receivers a hundred miles apart on two dials share
which tone is up and when, and share nothing about level or fading; the envelope
of the same pair correlates at 0.3 and picks a different lag in every segment.

That mask needs OUR key-down in the witness, and a witness placed to hear the
peer often carries it too faintly to correlate. So when it fails the pair is
registered instead on the bursts both recordings DECODE -- a matched filter over
a known 31-tone connect request reads one at an SNR the energy mask cannot see.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "hfmodem"))
from hfmodem.core import wav  # noqa: E402

HELD = 0.5
"""Correlation below which a window is reported but does not move the track."""

NO_OVERLAP = (
    "no overlap to measure: not one window of ours sits inside the witness at "
    "this lag. Either the two recordings do not cover the same air -- compare "
    "the durations above -- or one of them is never keyed, which the key-down "
    "fractions say.")


def read(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(wav.source(path)) as w:
        sw, ch = w.getsampwidth(), w.getnchannels()
        d = np.frombuffer(w.readframes(w.getnframes()),
                          dtype={1: "u1", 2: "<i2", 4: "<i4"}[sw]).astype(np.float64)
        d /= float(2 ** (8 * sw - 1))
        return d.reshape(-1, ch)[:, 0], float(w.getframerate())


def _owned_recording_side(path: Path):
    from hfmodem.core import capture_identity as ci
    binding = ci.artifact_binding(path)
    if not binding.get('capture_id'):
        return None
    ci.validate_artifact(wav.stored(path))
    return json.loads(Path(path).with_suffix('.json').read_text())


def witness_rate(path: Path, header: float) -> float:
    """The rate the RECEIVER measured, which a WAV header cannot hold.

    A KiwiSDR's audio clock is not an integer -- 11998.874629 Hz on the one arm
    in the record -- and `kiwi_witness` rounds it to 11999 for the header and
    keeps the measurement in the sidecar. 10.4 ppm is a millisecond every 96 s:
    under the 10 ms this tool quantises a lag to, but there is no reason to hand
    the correlator a number when the right one is on disk beside the audio.

    Asked of the witness alone, and deliberately. `stream.wav`'s sidecar says
    `samplerate`, and that 48000 is nominal -- the recorder's own clock reading
    is the very thing being tested here, so correcting the file by it would
    answer the question with itself.
    """
    _owned_recording_side(path)
    side = path.with_suffix(".json")
    if not side.exists():
        return header
    try:
        return float(json.loads(side.read_text())["sample_rate"])
    except (ValueError, KeyError, OSError):
        return header


def keydown(x: np.ndarray, fs: float, tones: tuple[float, float], rate: float,
            margin: float = 12.0) -> np.ndarray:
    """Per-frame: is one of the two FSK tones up, against the band beside them.

    FRAMES LAND ON THE SECONDS THEY CLAIM. The hop was `int(fs / rate)`, which
    is exact at 48000 Hz and 119 samples where 119.99 were meant at a KiwiSDR's
    11999 -- so the witness series ran 0.83% fast, 8300 ppm, and every lag read
    off it grew 8.3 ms per second of session. That is the shape of a recording
    losing air, and it was read as one: the staircase on the one witnessed arm
    measured +760 ms over 60 s and is +260 ms placed exactly, against the 295 ms
    that arm's own counter claimed.
    """
    n_fft = int(fs * 0.04)
    step = fs / rate
    if len(x) <= n_fft:
        return np.zeros(0)
    starts = np.round(np.arange(int((len(x) - n_fft) / step) + 1) * step).astype(int)
    starts = starts[starts + n_fft <= len(x)]
    w = np.hanning(n_fft)
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    sel = (np.abs(f - tones[0]) < 30) | (np.abs(f - tones[1]) < 30)
    ref = (f > 2000) & (f < 2400)
    # A CHUNK AT A TIME, because the arm's teardown now asks for this on every
    # witnessed session rather than an operator asking once on a 60 s pair. The
    # windowed frames are a copy, not a view: a hold-length arm's own recording
    # is minutes of 48 kHz audio, and 600 s of it materialises 920 MB in one
    # multiply. Chunking changes no frame and no threshold -- the median below is
    # still taken over the whole series.
    view = np.lib.stride_tricks.sliding_window_view(x, n_fft)
    lvl = np.concatenate([
        10 * np.log10(p[:, sel].max(axis=1)
                      / (np.median(p[:, ref], axis=1) + 1e-20) + 1e-20)
        for p in (np.abs(np.fft.rfft(view[s] * w, axis=1)) ** 2
                  for s in np.array_split(starts, max(len(starts) // 4096, 1)))])
    return (lvl > np.median(lvl) + margin).astype(float)


def lag_curve(a: np.ndarray, b: np.ndarray, rate: float, guess: float,
              win: float, hop: float, search: float) -> np.ndarray:
    """(t, lag, r) for each window of `a` found inside `b`, all in seconds.

    Each window is searched around the lag the LAST one settled on, not around
    the caller's guess. An ARQ link is periodic -- both stations key on the same
    1.25 s raster all session -- so a key-down mask correlates with itself one
    whole cycle away almost as well as at zero, and a window searched +-2 s has
    three peaks to choose from. The 0826 PACTOR arm picked the +1.25 s alias at
    r=0.814, printed it beside its neighbours as a 1240 ms step, and passed it
    through the `r > 0.5` filter into the drift figure. Tracking pins the search
    to a radius a capture can plausibly lose between windows, where the alias is
    not on the menu; a window whose correlation is poor holds the track rather
    than steering it.

    Empty and still (0, 3) when no window can be placed, because that is an
    answer and not a fault: the witness has no span to search at this lag, or
    ours is never keyed across a whole window, and an operator points this tool
    at a second receiver precisely to find out which. The caller slices the
    columns by index either way, and a bare `np.array([])` is one-dimensional.
    """
    w, h, s = int(win * rate), int(hop * rate), int(search * rate)
    out, tracked = [], guess
    # `range(0, len(a) - w, h)` is empty when the window IS the whole of ours,
    # which is exactly how the caller asks for a first guess -- so every run
    # without `--guess` printed NO_OVERLAP, and NO_OVERLAP reads as two
    # recordings of different air. One window still fits; place it.
    for i in range(0, max(len(a) - w, 1), h):
        if len(a) - i < w:
            break
        seg = a[i:i + w] - a[i:i + w].mean()
        lo = int(i + tracked * rate) - s
        ref = b[max(lo, 0):max(lo, 0) + w + 2 * s]
        if len(ref) < w + 2 * s or not seg.any():
            continue
        ref = ref - ref.mean()
        c = signal.correlate(ref, seg, "valid")
        c /= (np.sqrt(np.convolve(ref * ref, np.ones(w), "valid"))
              * np.linalg.norm(seg) + 1e-30)
        k = int(c.argmax())
        lag = (max(lo, 0) + k) / rate - i / rate
        out.append(((i / rate), lag, float(c[k])))
        if c[k] > HELD:
            tracked = lag
    return np.array(out) if out else np.empty((0, 3))


OVERLAP = 0.5
"""Least of our own recording that has to lie inside the witness to be searched.

Past that the correlation is taken on a handful of frames against a run of zeros
and reads whatever it likes."""


def whole_file_lag(ka: np.ndarray, kb: np.ndarray,
                   rate: float) -> tuple[float, float] | None:
    """One correlation of the whole of each, for the windowed track to start at.

    PARTIAL OVERLAP IS THE ORDINARY CASE, and `lag_curve` cannot express it: it
    slides a window of `a` around inside `b`, so it needs the witness to contain
    the whole of our recording. The witness contains neither end of it reliably.
    It opens on a Kiwi handshake of its own after the gate and stops with the
    arm, so it can start behind our recording as readily as ahead of it, and it
    loses blocks of its own, which pulls it short at the end -- on
    24 of the 29 peer-placed pairs in this station's captures the witness file is
    the SHORTER of the two. Every one of those printed NO_OVERLAP, whose text
    reads as two recordings of different air; it was the tool declining to look.

    So the guess is one normalised correlation over every offset with at least
    `OVERLAP` of ours inside the witness, either way round, computed on the
    running sums rather than by sliding a window.
    """
    n = len(ka)
    if n == 0 or len(kb) == 0 or not ka.any():
        return None
    pad = int(n * (1 - OVERLAP))
    b = np.concatenate([np.zeros(pad), kb, np.zeros(pad)])
    if len(b) < n:
        return None
    a = ka - ka.mean()
    cs, cs2 = np.cumsum(b), np.cumsum(b * b)
    s1 = np.concatenate([cs[n - 1:n], cs[n:] - cs[:-n]])
    s2 = np.concatenate([cs2[n - 1:n], cs2[n:] - cs2[:-n]])
    r = signal.correlate(b, a, "valid") / (
        np.sqrt(np.maximum(s2 - s1 * s1 / n, 0.0)) * np.linalg.norm(a) + 1e-30)
    k = int(np.argmax(r))
    return (k - pad) / rate, float(r[k])


def too_short(ka: np.ndarray, kb: np.ndarray, rate: float) -> str | None:
    """The one reason a lag cannot be searched for that is NOT different air.

    `whole_file_lag` needs `OVERLAP` of ours to fit inside the witness, and a
    witness shorter than that fails whatever it heard. Since the launcher places
    the witness after the gate, the commonest way to land here is a receiver that
    came up several seconds into the arm -- and NO_OVERLAP's text sends the
    reader off to look for two recordings of different air instead. How far the
    file falls short is the measurement that separates the two, and
    `kiwi_witness` bounds its walk so this stays rare: its log beside the .wav
    says what the placement cost.
    """
    if not len(ka) or not ka.any():
        return None
    need = len(ka) * OVERLAP
    if len(kb) >= need:
        return None
    return (f"the witness holds {len(kb) / rate:.0f} s against our "
            f"{len(ka) / rate:.0f} s, short of the {need / rate:.0f} s of ours a "
            "lag is searched over -- a receiver that came up late or dropped "
            "early, not a recording of different air. Both files are of this "
            "arm; there is not enough of the witness to register it against.")


REGISTERS = 0.5
"""Fraction of windows that must correlate for the pair to be ONE measurement.

Measured over the 123 session/witness pairs in this station's captures: the arms
whose lag track is a readable staircase sit at 0.50 to 1.00 of windows above
`HELD`, the ones whose per-window lag jumps hundreds of milliseconds at 0.00 to
0.42, and 0.42 to 0.50 is empty. It is the trough of a two-humped distribution
rather than a tolerance, so a pair that lands near it is a pair to read the
table for."""

MIN_WINDOWS = 3
"""Windows below which a fraction of them is not a reading of anything."""


def verdict(ka: np.ndarray, kb: np.ndarray, lc: np.ndarray) -> str:
    """Whether these two recordings are one measurement, said where it is known.

    THE GEOMETRY GATE ASKS WHICH STATION THE RECEIVER IS NEARER AND THE PROBE
    ASKS WHETHER IT HEARS ANYTHING; neither asks whether it hears US. Our own
    keying is the only thing both files carry, so it is the time base they are
    registered on, and a witness that does not hold it cannot be placed in our
    cycle grid at all -- no answer slot can be read out of it, and its silence is
    not the peer's silence, which is exactly what `nearer the peer` used to
    license a reader to conclude. On 2026-08-31 a receiver that passed both gates carried
    our key-down in 0.04 of its frames against our own 0.37 and drew a whole-file
    lag of 189.69 s at r=0.177, with the per-window lag jumping +-400 ms; the two
    gates said nothing about it and the arm was written up before anyone ran this.
    """
    if len(lc) < MIN_WINDOWS:
        return (f"WITNESS UNALIGNED: {len(lc)} window(s) of this session can be "
                f"placed in the witness -- our key-down is {ka.mean():.2f} of "
                f"frames and the witness's {kb.mean():.2f}. The pair is not one "
                "measurement: no answer slot can be read out of this recording, "
                "and silence in it is not the peer's silence.")
    held = lc[:, 2] > HELD
    r = float(np.median(lc[:, 2]))
    if held.mean() >= REGISTERS:
        # WHAT ALIGNMENT KNOWS AND NO MORE. That a slot in this recording is the
        # slot we keyed over is registration's to say; that what is missing from
        # it was never sent is the geometry line's, and only when the receiver
        # sits nearer the peer than us. Both are printed, and neither is the
        # other.
        return (f"WITNESS ALIGNED: {held.sum()} of {len(lc)} windows register "
                f"against our own keying (median r {r:.2f}), so this recording "
                "can be placed in our cycle grid and an answer slot in it is the "
                "slot we keyed over.")
    return (f"WITNESS UNALIGNED: {held.sum()} of {len(lc)} windows register "
            f"against our own keying (median r {r:.2f}) -- it carries our "
            f"key-down in {kb.mean():.2f} of its frames against our own "
            f"{ka.mean():.2f}. The pair cannot be registered, so no answer slot "
            "can be read out of this recording and silence in it is not the "
            "peer's silence.")


ANCHOR_WINDOW = 0.25
"""Lags this far apart are the same anchor.

Wider than the 25 ms the six anchors of the 09-03 VE3KPG arm agree to and far
under the 12.6 s a connect request repeats at, which is the alias a vote could
otherwise land on.
"""

MIN_ANCHORS = 3
"""Anchors below which an offset is a coincidence rather than a registration."""


def _at_48k(path: Path, rate: float | None) -> np.ndarray:
    x, fs = read(path)
    fs = rate or fs
    if abs(fs - 48000.0) > 1.0:
        x = signal.resample(x, int(round(len(x) * 48000.0 / fs)))
    return x / (np.abs(x).max() or 1.0)


def frames(path: Path, rate: float | None, calls: list[str]) -> list[tuple[float, str]]:
    """Every named handshake burst in a recording, as (seconds, what it says).

    `vara_monitor`'s scanner and not its segmenter, which also names CRC-clean
    record-3 overs: a segmenter's burst start moves with the level, and the two
    data overs common to the 09-03 pair place it 1.2 s from where its six
    handshake bursts agree. A scanner reports the alignment its matched filter
    locked, which is the same instant in both files.
    """
    for d in (REPO / "tools", REPO / "packages" / "hfmodem", REPO / "working" / "vara"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import vara_monitor as VM

    x, scan, out = _at_48k(path, rate), VM.HandshakeScanner(calls), []
    for i in range(0, len(x), VM.CHUNK):
        out += [(n / 48000.0, f"{r.kind} {r.gateway or r.info}")
                for n, r in scan.push(x[i:i + VM.CHUNK])]
    return out


def _group(votes: list[tuple]) -> list[tuple]:
    best: list[tuple] = []
    for v in votes:
        g = [w for w in votes if v[0] <= w[0] <= v[0] + ANCHOR_WINDOW]
        if len(g) > len(best) or (len(g) == len(best) and g
                                  and g[-1][0] - g[0][0] < best[-1][0] - best[0][0]):
            best = g
    return best


def anchors(ours: list[tuple[float, str]], witness: list[tuple[float, str]]
            ) -> tuple[list[tuple], int]:
    """The bursts both recordings decoded, and how big the runner-up offset is.

    Every pair that says the same thing votes for an offset; the true one is
    voted for once per burst and the raster's aliases pick up the rest. A cluster
    that does not beat the best offset disjoint from it is not a registration --
    on the 09-03 arm six anchors sit inside 25 ms against four for the alias one
    connect-request interval away.
    """
    votes = sorted((tw - to, lab, to, tw)
                   for to, lab in ours for tw, lb in witness if lab == lb)
    best = _group(votes)
    return best, len(_group([v for v in votes if v not in best]))


def frame_verdict(best: list[tuple], rival: int) -> list[str]:
    if len(best) < MIN_ANCHORS or len(best) <= rival:
        return [f"WITNESS UNREGISTERED: {len(best)} burst(s) decode in both "
                f"recordings at one offset against {rival} at another, which is "
                "not enough to place this pair. No answer slot can be read out of "
                "the witness, and silence in it is not the peer's silence."]
    lag = float(np.median([v[0] for v in best]))
    spread = (best[-1][0] - best[0][0]) * 1e3
    return [f"WITNESS ALIGNED ON FRAMES: {len(best)} bursts decode in both "
            f"recordings and place them at one offset -- the witness begins "
            f"{-lag:.2f} s into this recording, the anchors agreeing to "
            f"{spread:.0f} ms. Our key-down could not carry the registration; "
            "these could."] + [
        f"  anchor {lab:<48s} ours {to:8.3f}  witness {tw:8.3f}"
        for _, lab, to, tw in best]


def register_on_frames(ours: Path, witness: Path, calls: list[str],
                       rate: float) -> list[str]:
    if not calls:
        return ["no frame anchor: no callsign to name a burst by -- the witness "
                "sidecar carries no `peer` and --calls was not given."]
    try:
        a, b = frames(ours, None, calls), frames(witness, rate, calls)
    except ImportError as e:
        return [f"no frame anchor: {e}"]
    print(f"decoded bursts: ours {len(a)}, witness {len(b)}")
    return frame_verdict(*anchors(a, b))


def peer_calls(witness: Path) -> list[str]:
    """The station the witness was placed for, from the sidecar `kiwi_witness` wrote."""
    _owned_recording_side(witness)
    side = witness.with_suffix(".json")
    if not side.exists():
        return []
    try:
        peer = json.loads(side.read_text()).get("peer") or ""
    except (ValueError, OSError):
        return []
    return [peer.split()[0].upper()] if peer.split() else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ours")
    p.add_argument("witness")
    p.add_argument("--guess", type=float,
                   help="seconds ours starts after the witness; searched if absent")
    p.add_argument("--tones", type=float, nargs=2, default=(1400.0, 1600.0))
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--win", type=float, default=8.0)
    p.add_argument("--hop", type=float, default=2.0)
    p.add_argument("--search", type=float, default=0.25,
                   help="radius each window searches around the tracked lag; "
                        "keep it under half the link's cycle or the correlator "
                        "can pick a whole-cycle alias of the raster")
    p.add_argument("--calls", help="callsigns a handshake burst may be addressed "
                                   "to, for the frame anchor; defaults to the "
                                   "peer the witness sidecar names")
    a = p.parse_args()

    x, fx = read(Path(a.ours))
    y, fy = read(Path(a.witness))
    fy = witness_rate(Path(a.witness), fy)
    print(f"ours    {len(x)/fx:8.3f} s @ {fx:.0f} Hz")
    print(f"witness {len(y)/fy:8.3f} s @ {fy:.6g} Hz")
    ka = keydown(x, fx, a.tones, a.rate)
    kb = keydown(y, fy, a.tones, a.rate)
    print(f"key-down: ours {ka.mean():.2f} of frames, witness {kb.mean():.2f}")

    calls = ([c.strip().upper() for c in a.calls.split(",") if c.strip()] if a.calls
             else peer_calls(Path(a.witness)))

    def fall_back() -> None:
        for line in register_on_frames(Path(a.ours), Path(a.witness), calls, fy):
            print(line)

    guess = a.guess
    lc = np.empty((0, 3))
    if guess is None:
        whole = whole_file_lag(ka, kb, a.rate)
        if whole is None:
            print(too_short(ka, kb, a.rate) or NO_OVERLAP)
            print(verdict(ka, kb, lc))
            fall_back()
            return
        guess = whole[0]
        print(f"whole-file lag {guess:.2f} s at r={whole[1]:.3f}")

    lc = lag_curve(ka, kb, a.rate, guess, a.win, a.hop, a.search)
    if not len(lc):
        opened = "opened after we did" if guess < 0 else "was already recording"
        print(f"at a lag of {guess:+.1f} s -- the witness {opened} -- "
              + NO_OVERLAP)
        print(verdict(ka, kb, lc))
        fall_back()
        return
    print(f"\n{'t (s)':>7} {'lag (s)':>9} {'step ms':>9} {'r':>6}")
    prev = None
    for t, lag, r in lc:
        step = "" if prev is None else f"{(lag - prev) * 1e3:+9.0f}"
        print(f"{t:7.1f} {lag:9.3f} {step:>9} {r:6.3f}")
        prev = lag
    said = verdict(ka, kb, lc)
    print(f"\n{said}")
    good = lc[lc[:, 2] > HELD]
    if len(good) > 1 and not said.startswith("WITNESS UNALIGNED"):
        drift = (good[-1, 1] - good[0, 1]) * 1e3
        print(f"\nover {good[-1,0]-good[0,0]:.0f} s of session the witness runs "
              f"{drift:+.0f} ms further ahead of this recording.")
        print("A recording short of the air drifts POSITIVE; the witness losing "
              "samples of its own would drift the other way.")
    if said.startswith("WITNESS UNALIGNED"):
        fall_back()


if __name__ == "__main__":
    main()
