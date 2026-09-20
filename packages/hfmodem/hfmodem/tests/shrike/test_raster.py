# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate analysis/raster.py against a synthetic link whose timing is known.

The raster tool exists to answer a question no recording can answer twice: what
cadence was the far end actually running, and where in it were we transmitting.
On real audio there is nothing to check the answer against -- which is precisely
how a period estimator can return a THIRD of the truth and look convincing doing
it. That failure is real and it is the reason this file exists: fed the witness
recording, an earlier version returned 0.41667 s with a tighter confidence
interval and a higher coherence score than the correct 1.25 s, because a finer
grid has more nodes and the few off-raster strays each sat near one.

So the timing here is built, not measured, and the tool has to recover it:

  * a peer on a 1.2499 s raster (its clock, near the specified 1.25),
  * us on a 1.27 s raster (shrike's measured overrun) -- two metronomes,
  * our transmissions stopping partway, the peer carrying on unperturbed,
  * a Morse ID at the end, whose dashes are 0.17 s and would be read as control
    signals by any classifier working on duration alone.

Run:  python -m hfmodem.tests.shrike.test_raster
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests.shrike import archive

# The tool under test lives in the working record, which never crosses the
# publication boundary. This was a bare `import raster` at module scope, so in the
# distribution it aborted collection of the WHOLE suite rather than skipping one
# file. The module still has to import cleanly either way: `test_static.py` walks
# every test module here and imports it.
_ANALYSIS = archive.ARCHIVE / "analysis"
sys.path.insert(0, str(_ANALYSIS))
try:
    import raster  # noqa: E402
except ModuleNotFoundError:
    raster = None

requires_raster = pytest.mark.skipif(
    raster is None, reason=f"the raster tool is not present at {_ANALYSIS}/raster.py")

FS = 12000
PEER_PERIOD = 1.2499        # the far end's clock
OUR_PERIOD = 1.27           # ours, long -- the drift measured on the air
TURNAROUND = 0.085          # our carrier drops, the peer answers this much later
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _fsk(dur: float, rng: np.random.Generator, amp: float) -> np.ndarray:
    """A burst of 100 Bd 1400/1600 Hz FSK, phase-continuous, raised-cosine edges."""
    n = int(dur * FS)
    sym = FS // 100
    bits = rng.integers(0, 2, n // sym + 1)
    f = np.repeat(np.where(bits, 1600.0, 1400.0), sym)[:n]
    x = np.sin(2 * np.pi * np.cumsum(f) / FS)
    edge = int(0.004 * FS)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    x[:edge] *= ramp
    x[-edge:] *= ramp[::-1]
    return amp * x


def build(path: Path, txlog: Path) -> dict:
    """A 55 s recording of one link setup: our bursts, the peer's, then its ID."""
    rng = np.random.default_rng(7)
    n = int(55 * FS)
    x = rng.normal(0, 0.02, n).astype(np.float32)

    def put(t: float, seg: np.ndarray) -> None:
        i = int(round(t * FS))
        x[i:i + seg.size] += seg[:max(0, n - i)]

    # Our sync packets: 0.96 s, every other slot of OUR long cycle, six of them.
    tx = []
    for k in range(6):
        start = 3.0 + k * 2 * OUR_PERIOD
        put(start, _fsk(0.96, rng, 0.5))
        tx.append({"n": k + 1, "what": "connect->TEST",
                   "start_sample": int(round(start * FS)),
                   "end_sample": int(round((start + 0.96) * FS))})
    last_end = 3.0 + 5 * 2 * OUR_PERIOD + 0.96

    # The peer's control signals: 0.12 s on ITS raster, phased so the first one
    # lands a turnaround after our first burst and free-running from there --
    # which is exactly the claim the tool has to be able to reject or confirm.
    peer0 = 3.0 + 0.96 + TURNAROUND
    peer_t = [peer0 + k * PEER_PERIOD for k in range(34)]
    for t in peer_t:
        put(t, _fsk(0.12, rng, 0.25))
    # Morse: dits, dahs and spaces at 20 WPM, deliberately unclassifiable by
    # duration (a dah is 0.18 s against a 0.12 s control signal).
    at = peer_t[-1] + 2.0
    for e in ".-- ... ---.. . --- -.-.":
        if e == " ":
            at += 0.42
            continue
        d = 0.06 if e == "." else 0.18
        put(at, _fsk(d, rng, 0.3))
        at += d + 0.06

    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    txlog.write_text(json.dumps({
        "fs": FS, "recording_start_sample": 0, "transmissions": tx,
        "silent_from_sample": int(round(last_end * FS))}) + "\n")
    return {"peer_t": peer_t, "last_end": last_end}


def main() -> int:
    tmp = Path(__file__).resolve().parent / "__pycache__"
    tmp.mkdir(exist_ok=True)
    wav, txlog = tmp / "raster_synth.wav", tmp / "raster_synth.json"
    truth = build(wav, txlog)

    x, fs = raster.load_wav(str(wav))
    check("recording loads at its own rate", fs == FS, f"{fs}")
    mark, space = raster.find_tones(x, fs)
    check("tone pair located without being told",
          abs(mark - 1400) < 15 and abs(space - 1600) < 15, f"{mark:.0f}/{space:.0f}")

    env = raster.band_envelope(x, fs, (mark + space) / 2, 200.0, 0.008)
    bursts = raster.detect_bursts(env, fs)
    raster.classify(bursts, 0.96)
    kinds = [b["kind"] for b in bursts]
    # Not six. Three of our packets have a control signal inside them, because
    # two rasters 20 ms apart per cycle walk into each other -- which is the
    # whole complaint the experiment exists to diagnose, reproduced here. One
    # receiver cannot separate two stations transmitting at once, and a tool that
    # claimed to would be lying. It is why the measurement is taken in the
    # SILENT tail, where nothing of ours is on the channel.
    check("our packets are labelled where they are not collided with",
          kinds.count("packet") >= 3, f"{kinds.count('packet')} clear of 6 sent")
    check("the Morse ID is not mistaken for control signals",
          kinds.count("cw") >= 15 and
          not any(b["kind"] == "cs" for b in bursts
                  if b["onset"] > truth["peer_t"][-1] + 1.0),
          f"{kinds.count('cw')} cw, "
          f"{sum(b['kind'] == 'cs' for b in bursts if b['onset'] > truth['peer_t'][-1] + 1)}"
          f" late cs")

    peer = np.array([b["onset"] for b in bursts if b["kind"] in ("cs", "cs3")])
    tail = np.array([t for t in truth["peer_t"] if t >= truth["last_end"]])
    check("every control signal in the silent tail is detected",
          sum(t >= truth["last_end"] for t in peer) == tail.size,
          f"{sum(t >= truth['last_end'] for t in peer)} of {tail.size}")
    fit = raster.fit_raster(peer[peer >= truth["last_end"]])
    err = fit["period"] - PEER_PERIOD
    check("period recovered", abs(err) < 0.002, f"{fit['period']:.5f} ({err * 1e3:+.2f} ms)")
    whole = raster.fit_raster(peer)
    check("NOT a harmonic of the true period, even over the contaminated whole",
          abs(whole["period"] - PEER_PERIOD) < 0.005, f"{whole['period']:.5f}")
    check("confidence interval covers the truth",
          abs(err) < 1.96 * fit["period_sigma"] + 5e-4,
          f"+/-{1.96 * fit['period_sigma'] * 1e3:.2f} ms")
    check("onset residuals are at the millisecond",
          fit["resid_rms"] < 0.004, f"{fit['resid_rms'] * 1e3:.1f} ms")

    # Onsets 4 s past the last fitted burst, predicted from the fit alone.
    t, sig = raster.predict(fit, truth["peer_t"][-1] + 4.0)
    nearest = min((peer0 for peer0 in
                   (truth["peer_t"][-1] + k * PEER_PERIOD for k in range(10))),
                  key=lambda v: abs(v - t))
    check("extrapolates 4 s past the data", abs(t - nearest) < 0.01,
          f"{(t - nearest) * 1e3:+.1f} ms, quoted +/-{sig * 1e3:.1f} ms")

    # Phase: our carrier dropped TURNAROUND before one of the peer's slots by
    # construction only for the FIRST cycle; the two rasters walk apart after.
    ph = raster.phase_of(3.0 + 0.96, fit)
    check("phase of our first burst end recovered",
          abs(-ph - TURNAROUND) < 0.01, f"{-ph * 1e3:.1f} ms before the slot")

    # Lock: the peer's cadence must be measured the same either side of our
    # going silent, because by construction it never listened to us.
    before = peer[peer < truth["last_end"]]
    after = peer[peer >= truth["last_end"]]
    fa, fb = raster.fit_raster(before), raster.fit_raster(after)
    check("free-running peer reads the same either side of our silence",
          abs(fb["period"] - fa["period"]) < 0.004,
          f"{fa['period']:.4f} -> {fb['period']:.4f}")
    ours = raster.fit_raster(np.array([w["end_sample"] / FS
                                       for w in json.loads(txlog.read_text())
                                       ["transmissions"]]))
    check("our own cycle measured from the transmission log",
          abs(ours["period"] - 2 * OUR_PERIOD) < 0.002, f"{ours['period']:.4f}")
    check("the two cycles are told apart",
          abs(ours["period"] / 2 - fit["period"]) >
          2 * (ours["period_sigma"] / 2 + fit["period_sigma"]),
          f"{ours['period'] / 2 - fit['period']:+.4f} s per cycle")

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


@requires_raster
def test_raster() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
