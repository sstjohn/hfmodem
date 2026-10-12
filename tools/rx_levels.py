#!/usr/bin/env python3
"""What the receive working point actually is: band noise, dynamic range, and the floor.

Every level criterion at this station is a peak criterion. `hfcapture check` asks for
peaks near -10 dBFS, `codec_gain` exists to stop a capture railing, and the certified
working point -- rig DATA OUT 45/100 with the codec input scalar at 0.040 -- was
accepted on peak headroom and nothing railed. None of that says where the *noise floor*
sits, and the floor is what a detector has to lift a signal out of.

So nobody knows what this chain's floor and dynamic range are. Not "they are bad": not
known. 767 session sidecars, none with a noise-floor key; no tool in the flock reports
one. This tool reports one. It measures:

    band noise      in-band RMS over the windows that are live, in dBFS
    quiet floor     the very quietest windows, which on a session recording are
                    usually our own receiver muted by our own transmitter, and are
                    not the band at all -- these two get told apart, not averaged
    dynamic range   p99 of noise-only windows to the peak of the strongest burst
    rail            fraction of samples against full scale

and recommends a change only when one of those numbers is out of bounds.

Deliberately NOT a level-versus-distortion test. That hypothesis is closed: captures
in this same chain with 8.8% of their samples railed top out at a third-harmonic image
of -16.7 dBc, while the KB9MMT response sits at +4.9 dBc in a file with zero railed
samples, and image level anti-correlates with fundamental level at -0.37 dB/dB. No
static transfer curve does that, so sweeping the gain and watching the images would
prove nothing.

What did settle the images was a frequency question, and that is the shape the
diagnostics here take. A harmonic made at the far end arrives with the same dial offset
as its fundamental; one made after our own frequency translation arrives with that
offset multiplied by the harmonic number. `origin` measures both offsets and their
standard errors and reports which hypothesis survives. `sweep` writes a fixed-level
tone sweep to play into the dongle through a loopback cable with the radio
disconnected, and measures the images per frequency from the recording -- level held
constant, frequency varied, which is the axis the effect actually lives on.

    measure     at the radio: the working point that exists right now
    state       the rig settings nothing else in the flock writes down
    check       the same measurements on recordings -- no hardware
    origin      harmonic offsets: made here, or made at the far end
    sweep       write / analyse a fixed-level loopback tone sweep

Receive-only throughout. It never keys, never opens a rig-control socket, never writes
the codec's output scope, and `sweep` only ever writes a file for the operator to play
themselves.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/hfmodem"))
from hfmodem.core.busy import DEAF_DBFS, MUTE_DROP_DB, occupancy_db  # noqa: E402
# The peak figures and the rail come from the same module as the working point
# this tool is here to find, and they were retyped beside it -- equal to core's
# by luck rather than by wiring, on the one tool an operator sets the input with.
from hfmodem.core.levels import (PEAK_AIM_DBFS, PEAK_LIMIT_DBFS,  # noqa: E402
                                 WORKING_POINT, peak_dbfs, railed_ppm)

FS = 48000

# The known-good reference: offair/NS0A_2300 decoded cleanly, and is the yardstick any
# number here is read against. The acceptance window is wide because the target is a
# range the modem can work in, not a set point -- too high and the floor eats the
# headroom, too low and we are listening to the codec rather than to the band.
REFERENCE = "offair/NS0A_2300"
NOISE_WINDOW_DBFS = (-48.0, -30.0)
MIN_RANGE_DB = 25.0

# shrike calls the receiver deaf below an RMS of 0.01; core.busy calls it deaf below
# DEAF_DBFS. Both are absolute floors, neither is this tool's to move, and a working
# point has to be read against them.
SHRIKE_DEAF_DBFS = -40.0

# A window sitting within this much of the band-noise estimate is treated as noise-only.
# Wide enough to hold the noise's own p99 (band noise reached +6.7 dB over its tracked
# floor on the 2026-07-26 recording), narrow enough to exclude a burst.
NOISE_SPREAD_DB = 8.0

# One spectrum serves every job here: 2.93 Hz bins place a fifth harmonic inside a
# +/-15 Hz window, and 4 s of audio still averages eleven segments.
_NFFT = 16384
_HARM_HW_HZ = 15.0
_BASE_HW_HZ = 300.0
_CARRIER_BAND = (300.0, 2900.0)
_HARMONICS = (3, 5)            # odd: a symmetric compressor makes these and not the even
_MIN_TONE_DB = 10.0            # peak over local baseline, below which there is no carrier
# A harmonic window has about eleven bins, so its noise sum scatters ~1.2 dB; six is five
# sigma of that. Below it the reading is the local noise and the honest answer is the
# floor, not a number -- a two-decibel bump over the baseline is not an image.
_IMAGE_MARGIN_DB = 6.0

_SWEEP_HZ = tuple(range(300, 2800, 200))
_SWEEP_DWELL_S = 6.0
_SWEEP_LEVEL_DBFS = -20.0
_SWEEP_EDGE_S = 1.0            # trimmed off each dwell: the ramps are not steady tone


# --------------------------------------------------------------------- spectral core

def _welch(audio: np.ndarray, fs: int, nfft: int = _NFFT):
    """Averaged one-sided power spectrum, scaled so ``P.sum()`` is the mean square.

    Same convention as ``core.busy.occupancy_db``: these are RMS figures, so a
    full-scale sine reads -3 dBFS.
    """
    audio = np.asarray(audio, float)
    n = len(audio) // nfft
    if n < 2:
        return None, None
    win = np.hanning(nfft)
    x = (audio[:n * nfft].reshape(n, nfft) - audio.mean()) * win
    P = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=0) * (2 / (nfft ** 2 * (win ** 2).mean()))
    return P, np.fft.rfftfreq(nfft, 1 / fs)


def _peak_hz(P: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    """Frequency of the strongest bin in ``[lo, hi]``, interpolated between bins.

    Parabolic on the log magnitude. Without it a tone landing between bins is placed up
    to half a bin out, and `origin` multiplies that error by the harmonic number.
    """
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return float("nan")
    i = int(np.flatnonzero(band)[np.argmax(P[band])])
    if i == 0 or i == len(P) - 1:
        return float(f[i])
    a, b, c = (float(np.log(P[i + k] + 1e-30)) for k in (-1, 0, 1))
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom else 0.0
    return float(f[i] + np.clip(shift, -0.5, 0.5) * (f[1] - f[0]))


def _win_power(P: np.ndarray, f: np.ndarray, centre: float, half: float = _HARM_HW_HZ) -> float:
    return float(P[(f >= centre - half) & (f <= centre + half)].sum())


def _base_power(P: np.ndarray, f: np.ndarray, centre: float, half: float = _HARM_HW_HZ) -> float:
    """Power the window would hold if only the local noise were in it.

    Median density either side of the window, times the window's bin count, and the
    louder side wins. The median is what stops a second signal sitting nearby from
    inflating the estimate and hiding a real image underneath it; taking the louder side
    rather than pooling both is what stops a *step* in the noise floor from reading as an
    image. Third harmonics of 900-1000 Hz land on the SSB filter's own skirt, and a
    baseline pooled across that edge is low by half the step at exactly the frequencies
    where a real image would be most interesting.
    """
    win = (f >= centre - half) & (f <= centre + half)
    sides = [(f >= centre - _BASE_HW_HZ) & (f < centre - half),
             (f > centre + half) & (f <= centre + _BASE_HW_HZ)]
    dens = [np.median(P[s]) for s in sides if s.any()]
    if not dens:
        return float("nan")
    # Floored rather than allowed to reach zero: a synthetic tone with no noise under it
    # has a literally empty neighbourhood, and a zero baseline makes every ratio infinite.
    return max(float(max(dens) * win.sum()), 1e-30)


def carrier(audio: np.ndarray, fs: int = FS, band: tuple = _CARRIER_BAND) -> dict | None:
    """Strongest narrow feature in ``band``: ``{hz, dbfs, excess_db}``.

    ``excess_db`` is how far it stands over its own local baseline, and is the honest
    answer to "is there a carrier here at all": band noise and wideband data read a few
    dB, a real carrier reads tens.
    """
    P, f = _welch(audio, fs)
    if P is None:
        return None
    hz = _peak_hz(P, f, *band)
    peak, base = _win_power(P, f, hz), _base_power(P, f, hz)
    return dict(hz=hz,
                dbfs=float(10 * np.log10(max(peak, 1e-30))),
                excess_db=float(10 * np.log10(max(peak, 1e-30) / max(base, 1e-30))))


def image_dbc(audio: np.ndarray, fs: int = FS) -> dict:
    """Odd-harmonic image power relative to the fundamental, in dBc.

    ``image_dbc`` is ``None`` when every harmonic window sits at its local noise: the
    images are then below what this capture can see, and ``floor_dbc`` is that limit.
    The distinction between "clean" and "too quiet to tell" is the whole value of the
    measurement, and a comparison between two floor readings compares nothing.
    """
    P, f = _welch(audio, fs)
    tone = carrier(audio, fs)
    out = dict(hz=None, tone_db=None, fund_dbfs=None,
               image_dbc=None, floor_dbc=None, harmonics={})
    if P is None or tone is None:
        return out
    out.update(hz=tone["hz"], tone_db=tone["excess_db"])
    if tone["excess_db"] < _MIN_TONE_DB:
        return out

    f0 = tone["hz"]
    fund = _win_power(P, f, f0)
    out["fund_dbfs"] = float(10 * np.log10(max(fund, 1e-30)))
    excess = floor = 0.0
    for k in _HARMONICS:
        if k * f0 > 0.45 * fs:
            continue
        total, base = _win_power(P, f, k * f0), _base_power(P, f, k * f0)
        floor += base
        seen = total - base
        shows = total > base * 10 ** (_IMAGE_MARGIN_DB / 10)
        out["harmonics"][k] = float(10 * np.log10(seen / fund)) if shows else None
        excess += max(seen, 0.0)
    if out["harmonics"]:
        out["floor_dbc"] = float(10 * np.log10(floor / fund))
        if excess > floor * 10 ** (_IMAGE_MARGIN_DB / 10):
            out["image_dbc"] = float(10 * np.log10(excess / fund))
    return out


def harmonic_origin(audio: np.ndarray, fs: int = FS, tone_hz: float = 1500.0,
                    harmonic: int = 3, search_hz: float = 40.0) -> dict:
    """Was the harmonic made before our frequency translation, or after it?

    A tone transmitted at ``tone_hz`` arrives at ``tone_hz + a``, where ``a`` is our own
    dial offset. A harmonic generated at the far end -- in the transmitter, or in
    anything ahead of our product detector -- rides that same offset unchanged, so it
    lands at ``harmonic * tone_hz + a``. One generated after our translation triples (or
    quintuples) the offset along with the tone, landing at ``harmonic * (tone_hz + a)``.
    Measuring both offsets separates the two, at fixed level, with no knobs touched.

    Offsets come from per-segment peak estimates, so their scatter gives a real standard
    error and the two hypotheses can be separated in sigma rather than by eye.
    """
    audio = np.asarray(audio, float)
    n = len(audio) // _NFFT
    out = dict(tone_hz=tone_hz, harmonic=harmonic, segments=0,
               a_hz=None, a_err=None, b_hz=None, b_err=None,
               sigma_far=None, sigma_near=None, verdict="")
    if n < 2:
        out["verdict"] = "too short to measure: needs at least two spectral segments"
        return out
    win = np.hanning(_NFFT)
    f = np.fft.rfftfreq(_NFFT, 1 / fs)
    fk_nom = harmonic * tone_hz
    a_s, b_s = [], []
    for i in range(n):
        P = np.abs(np.fft.rfft((audio[i * _NFFT:(i + 1) * _NFFT] - audio.mean()) * win)) ** 2
        f0 = _peak_hz(P, f, tone_hz - search_hz, tone_hz + search_hz)
        fk = _peak_hz(P, f, fk_nom - harmonic * search_hz, fk_nom + harmonic * search_hz)
        # A single segment has no averaging behind it, so a noise peak inside the
        # search window clears a small margin easily. Both ends must look like tones.
        need = 10 ** (_MIN_TONE_DB / 10)
        if _win_power(P, f, f0) < need * _base_power(P, f, f0):
            continue
        if _win_power(P, f, fk) < need * _base_power(P, f, fk):
            continue
        a_s.append(f0 - tone_hz)
        b_s.append(fk - fk_nom)
    out["segments"] = len(a_s)
    if len(a_s) < 2:
        out["verdict"] = ("no harmonic stands above its local noise in at least two "
                          "segments -- nothing to attribute")
        return out

    a, b = float(np.mean(a_s)), float(np.mean(b_s))
    # Standard error of the mean, floored at a tenth of a bin: a perfectly steady
    # synthetic tone has zero scatter, and a zero denominator would report infinite
    # confidence in whichever hypothesis it happened to land on.
    grain = 0.1 * (f[1] - f[0])
    sa = max(float(np.std(a_s, ddof=1)) / np.sqrt(len(a_s)), grain)
    sb = max(float(np.std(b_s, ddof=1)) / np.sqrt(len(b_s)), grain)
    far = abs(b - a) / np.hypot(sb, sa)
    near = abs(b - harmonic * a) / np.hypot(sb, harmonic * sa)
    out.update(a_hz=a, a_err=sa, b_hz=b, b_err=sb,
               sigma_far=float(far), sigma_near=float(near))

    if abs(a) < 3 * sa or abs((harmonic - 1) * a) < 3 * np.hypot(sb, harmonic * sa):
        out["verdict"] = (f"inconclusive: the tone is only {a:+.2f} Hz off nominal, so both "
                          "hypotheses predict the same harmonic frequency. Retune a few Hz "
                          "off and repeat.")
    elif near < 3 <= far:
        out["verdict"] = (f"made HERE, after our frequency translation: the offset scaled "
                          f"with the harmonic ({b:+.2f} Hz against {harmonic * a:+.2f} "
                          f"predicted), far-end origin excluded at {far:.1f} sigma.")
    elif far < 3 <= near:
        out["verdict"] = (f"made AT THE FAR END, ahead of our product detector: the offset "
                          f"did not scale ({b:+.2f} Hz against {a:+.2f} predicted), local "
                          f"origin excluded at {near:.1f} sigma.")
    else:
        out["verdict"] = (f"neither hypothesis fits ({far:.1f} sigma from far-end, "
                          f"{near:.1f} from local) -- the tone is drifting, or the peak "
                          "found is not the harmonic.")
    return out


# --------------------------------------------------------------------- level measures

def levels(audio: np.ndarray, fs: int = FS) -> dict:
    """Band noise, quiet floor, peak and rail occupancy for one capture.

    In-band level comes from ``core.busy.occupancy_db``, which integrates inside the
    passband. A reference band past the SSB filter skirt would measure the codec, not
    the band, and that mistake has already been made twice in this tree.

    ``band_noise`` and ``quiet_floor`` are reported separately and never averaged. On a
    session recording the very quietest windows are usually our own receiver muted by
    our own transmitter -- 16 of NS0A_2300's 108 seconds sit at a flat -45.0 dBFS, 22 dB
    below anything else in the file -- and reading that as the band's noise floor
    flatters the chain by exactly the depth of the mute.
    """
    audio = np.asarray(audio, float)
    db = occupancy_db(audio, fs)
    out = dict(seconds=len(audio) / fs,
               windows=len(db),
               peak_dbfs=peak_dbfs(audio),
               railed_ppm=railed_ppm(audio))
    if not len(db):
        return out | dict(band_noise=float("nan"), quiet_floor=float("nan"),
                          noise_p99=float("nan"), muted_windows=0, inband_p99=float("nan"))

    # Same rule core.busy uses to drop frames the receiver was not hearing through,
    # applied to the per-second series: a window this far under the loud end of the
    # capture is our own transmitter, or a codec dropout, and is not band audio.
    dead = db < np.percentile(db, 95) - MUTE_DROP_DB
    live = db[~dead] if (~dead).any() else db
    band_noise = float(np.percentile(live, 10))
    noise_only = live[live <= band_noise + NOISE_SPREAD_DB]
    return out | dict(band_noise=band_noise,
                      quiet_floor=float(np.percentile(db, 1)),
                      noise_p99=float(np.percentile(noise_only, 99)),
                      muted_windows=int(dead.sum()),
                      inband_p99=float(np.percentile(db, 99)))


def dynamic_range(noise: np.ndarray, signal: np.ndarray, fs: int = FS) -> dict:
    """Room between the loudest noise-only window and the loudest sample of a burst.

    p99 of the noise-only windows, not their median: what a detector has to separate a
    signal from is the noise at its worst. On the 2026-07-26 recording band noise
    reached +6.7 dB over its own tracked floor while the gateway's reply sat +6.2 --
    against the median the reply looked detectable, against the p99 it never was.
    """
    n, s = levels(noise, fs), levels(signal, fs)
    span = s["peak_dbfs"] - n["noise_p99"]
    return dict(noise=n, signal=s, range_db=float(span),
                ok=bool(span >= MIN_RANGE_DB
                        and s["peak_dbfs"] <= PEAK_LIMIT_DBFS
                        and s["railed_ppm"] == 0.0))


def assess(dr: dict) -> list[str]:
    """Findings, in the order they matter. Silence about a number is a pass.

    A change is recommended only where a measurement asks for one. Short dynamic range
    with the floor in its window is a weak burst, not a chain fault, and telling the
    operator to turn something is the wrong answer to it.
    """
    n, s, out = dr["noise"], dr["signal"], []
    lo, hi = NOISE_WINDOW_DBFS
    noted = 0                    # observations that are not faults, and must not mask a pass
    if n["band_noise"] > hi:
        out.append(f"Band noise {n['band_noise']:.1f} dBFS is above the {hi:.0f} dBFS window: "
                   f"the chain is amplifying the band's own noise into the headroom. Reduce "
                   f"gain ahead of the codec -- DATA OUT first, AF gain second.")
    elif n["band_noise"] < lo:
        out.append(f"Band noise {n['band_noise']:.1f} dBFS is below the {lo:.0f} dBFS window. "
                   f"Either the antenna is not connected or this is the codec's own floor "
                   f"rather than the band's -- check that the receiver is hearing anything.")
    if n["muted_windows"] and n["quiet_floor"] < n["band_noise"] - MUTE_DROP_DB:
        out.append(f"{n['muted_windows']} of {n['windows']} quiet windows sit at "
                   f"{n['quiet_floor']:.1f} dBFS, {n['band_noise'] - n['quiet_floor']:.0f} dB "
                   f"under band noise: that is a muted or dropped-out receiver, not the band. "
                   f"Read the dynamic range against band noise, not against those.")
    if s["railed_ppm"]:
        out.append(f"{s['railed_ppm']:.0f} ppm of the strong capture is against the rail. "
                   f"Nothing above that is recoverable; take DATA OUT down.")
    elif s["peak_dbfs"] > PEAK_LIMIT_DBFS:
        out.append(f"Strongest peak {s['peak_dbfs']:.1f} dBFS, only "
                   f"{-s['peak_dbfs']:.1f} dB from the rail. Take DATA OUT down.")
    elif s["peak_dbfs"] > PEAK_AIM_DBFS:
        noted += 1
        out.append(f"Strongest peak {s['peak_dbfs']:.1f} dBFS is above the "
                   f"{PEAK_AIM_DBFS:.0f} dBFS aim but nothing railed, which is inside the "
                   f"measured spread of this working point. Noted, not a fault.")
    if dr["range_db"] < MIN_RANGE_DB:
        if lo <= n["band_noise"] <= hi and s["peak_dbfs"] <= PEAK_LIMIT_DBFS:
            out.append(f"Dynamic range {dr['range_db']:.1f} dB is under the "
                       f"{MIN_RANGE_DB:.0f} dB the modem needs, but the floor is where it "
                       f"should be and nothing is railing: this burst was simply weak. "
                       f"Re-measure on the strongest signal you actually work.")
        else:
            out.append(f"Dynamic range {dr['range_db']:.1f} dB, under the "
                       f"{MIN_RANGE_DB:.0f} dB the modem needs.")
    if len(out) == noted:
        out.append(f"Working point is good: band noise {n['band_noise']:.1f} dBFS, "
                   f"{dr['range_db']:.1f} dB of range, nothing railed. Leave it alone and "
                   f"write it down.")
    return out


def guard_implications(band_noise_dbfs: float) -> list[str]:
    """What this floor does to the two absolute "receiver deaf" guards.

    Neither is this tool's to move. In-band RMS is a lower bound on the broadband RMS
    they actually test, so these margins are the pessimistic side.
    """
    out = []
    for name, thresh in (("core.busy.DEAF_DBFS", DEAF_DBFS),
                         ("shrike RX_DEAF_RMS", SHRIKE_DEAF_DBFS)):
        margin = band_noise_dbfs - thresh
        verdict = ("FIRES at this band noise" if margin <= 0 else
                   "thin -- a quiet band or a QSB null will trip it" if margin < 6 else
                   "clear")
        out.append(f"{name} = {thresh:.0f} dBFS: {margin:+.1f} dB of margin, {verdict}")
    return out


# ---------------------------------------------------------------------------- sweep

def sweep_plan(hz=_SWEEP_HZ, dwell: float = _SWEEP_DWELL_S,
               level_dbfs: float = _SWEEP_LEVEL_DBFS) -> dict:
    return dict(hz=list(hz), dwell_s=dwell, level_dbfs=level_dbfs)


def sweep_signal(plan: dict, fs: int = FS) -> np.ndarray:
    """The fixed-level tone sweep, as samples. Cosine ramps so no dwell starts on a step."""
    amp = np.sqrt(2) * 10 ** (plan["level_dbfs"] / 20)
    n = int(plan["dwell_s"] * fs)
    ramp = np.hanning(2 * int(0.05 * fs))
    env = np.ones(n)
    env[:len(ramp) // 2], env[-(len(ramp) // 2):] = ramp[:len(ramp) // 2], ramp[len(ramp) // 2:]
    t = np.arange(n) / fs
    return np.concatenate([amp * env * np.sin(2 * np.pi * hz * t) for hz in plan["hz"]])


def sweep_report(audio: np.ndarray, plan: dict, fs: int = FS) -> list[dict]:
    """Per-frequency images from a loopback recording of ``sweep_signal``.

    Dwells are located by what they contain rather than by when they should start, so a
    recording that began late, or ran long, still analyses. Each dwell is trimmed at
    both ends: the ramps are not steady tone and would smear the harmonic windows.
    """
    audio = np.asarray(audio, float)
    sec = int(fs)
    P_all = [_welch(audio[i * sec:(i + 1) * sec], fs, nfft=4096) for i in range(len(audio) // sec)]
    dom = [_peak_hz(P, f, *_CARRIER_BAND) if P is not None else float("nan") for P, f in P_all]

    out = []
    for hz in plan["hz"]:
        hit = np.flatnonzero([abs(d - hz) < 30.0 for d in dom])
        if len(hit) == 0:
            out.append(dict(hz=hz, seconds=0, image=None))
            continue
        # The longest run of consecutive seconds on this tone, minus its edges.
        breaks = np.flatnonzero(np.diff(hit) > 1) + 1
        best = max(np.split(hit, breaks), key=len)
        edge = int(_SWEEP_EDGE_S) if len(best) > 2 * _SWEEP_EDGE_S + 1 else 0
        seg = audio[(best[0] + edge) * sec:(best[-1] + 1 - edge) * sec]
        out.append(dict(hz=hz, seconds=len(seg) / fs, image=image_dbc(seg, fs)))
    return out


def sweep_verdict(report: list[dict]) -> str:
    seen = [(r["hz"], r["image"]["image_dbc"]) for r in report
            if r["image"] and r["image"]["image_dbc"] is not None]
    if not seen:
        floors = [r["image"]["floor_dbc"] for r in report
                  if r["image"] and r["image"]["floor_dbc"] is not None]
        if not floors:
            return "No dwells found. Was the sweep played into the loopback while recording?"
        return (f"No image above the measurement floor at any frequency (best floor "
                f"{min(floors):.1f} dBc). The dongle alone is linear at this level.")
    span = max(v for _, v in seen) - min(v for _, v in seen)
    worst = max(seen, key=lambda r: r[1])
    if span < 6.0:
        return (f"Images present and flat with frequency ({span:.1f} dB across the sweep, "
                f"worst {worst[1]:.1f} dBc at {worst[0]:.0f} Hz) -- a plain static "
                f"nonlinearity in the dongle.")
    return (f"Images vary {span:.1f} dB across the sweep, worst {worst[1]:.1f} dBc at "
            f"{worst[0]:.0f} Hz. Frequency-dependent, so not a plain transfer curve.")


# -------------------------------------------------------------------- audio sources

class Source:
    """Where captures come from, and what an operator action does to the chain.

    The guided flow only asks for audio and announces changes; keeping both behind this
    seam is what lets the whole procedure run, and be tested, with no radio.
    """

    interactive = True

    def capture(self, seconds: float, why: str) -> np.ndarray:
        raise NotImplementedError

    def codec_scalar(self, scope: str = "input") -> float | None:
        return None

    def act(self, key: str, instruction: str) -> None:
        print(f"\n  >> {instruction}")
        input("     press Enter when done: ")


class DeviceSource(Source):
    """ffmpeg against the rig's audio input; CoreAudio for the codec scalars, read-only."""

    def __init__(self, device: str, name: str = "USB Audio Device"):
        self.device, self.name = device, name

    def capture(self, seconds: float, why: str) -> np.ndarray:
        import hfcapture

        print(f"  recording {seconds:.0f}s -- {why}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cap.wav"
            subprocess.run([hfcapture.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                            *hfcapture.input_spec(self.capture_device()), "-t", str(seconds),
                            "-ac", "1", "-ar", str(FS), "-c:a", "pcm_s16le", str(path)],
                           check=False)
            if not path.exists():
                sys.exit(f"nothing recorded from device {self.device!r}")
            return read_wav(path)[0]

    def capture_device(self):
        """Resolve a configured name to ffmpeg's index only at live capture."""
        import hfcapture
        if sys.platform == "darwin" and not self.device.isdigit():
            matches = [index for index, name in hfcapture.avfoundation_inputs()
                       if name == self.device]
            if len(matches) != 1:
                raise ValueError(f"capture input {self.device!r}: expected one exact match")
            return matches[0]
        return self.device

    def codec_scalar(self, scope: str = "input") -> float | None:
        import codec_gain

        scope = codec_gain.SCOPE_INPUT if scope == "input" else codec_gain.SCOPE_OUTPUT
        return codec_gain.get_gain(codec_gain.find(self.name, scope), scope)[1]


@dataclass
class SimulatedSource(Source):
    """A modelled receive chain, so the whole procedure can be rehearsed without a radio.

    Band noise at a set level, optionally with a signal on top, through a soft limiter.
    The limiter is odd-symmetric, so it makes the harmonics the measurements look for
    and no even ones.
    """

    noise_dbfs: float = -35.0
    signal_dbfs: float = -12.0
    carrier_hz: float = 1500.0
    drive: float = 0.0
    quiet: bool = True
    seed: int = 0
    interactive: bool = False
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def capture(self, seconds: float, why: str) -> np.ndarray:
        n = int(seconds * FS)
        # Passband-shaped noise over a small flat floor; the flat part is the codec's own
        # noise, and it is what sets the measurement floor the images are read against.
        F = np.fft.rfft(self._rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / FS)
        band = np.fft.irfft(F * (((f > 350) & (f < 2750)) + 0.02), n)
        band *= 10 ** (self.noise_dbfs / 20) / np.sqrt((band ** 2).mean())
        x = band
        if not self.quiet:
            t = np.arange(n) / FS
            x = x + np.sqrt(2) * 10 ** (self.signal_dbfs / 20) * np.cos(
                2 * np.pi * self.carrier_hz * t)
        if self.drive:
            x = np.tanh(self.drive * x) / self.drive
        print(f"  simulated {seconds:.0f}s -- {why}")
        return np.clip(x, -1.0, 1.0)

    def codec_scalar(self, scope: str = "input") -> float | None:
        return WORKING_POINT if scope == "input" else None

    def act(self, key: str, instruction: str) -> None:
        print(f"\n  >> {instruction}")
        self.quiet = key != "signal"
        print(f"     (simulated: {'quiet channel' if self.quiet else 'signal present'})")


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """A capture as mono float in +/-1, whatever the file's sample format."""
    from scipy.io import wavfile

    from hfmodem.core import wav
    fs, raw = wavfile.read(wav.source(path))
    a = np.asarray(raw, float)
    if a.ndim > 1:
        a = a[:, 0]
    if np.issubdtype(raw.dtype, np.integer):
        a = a / float(np.iinfo(raw.dtype).max + 1)
    return a, fs


def write_wav(path: Path | str, audio: np.ndarray, fs: int = FS) -> None:
    from scipy.io import wavfile

    wavfile.write(str(path), fs, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


# ------------------------------------------------------------------------- reporting

def _fmt_levels(lv: dict) -> str:
    return (f"    band noise   {lv['band_noise']:7.1f} dBFS in-band\n"
            f"    noise p99    {lv['noise_p99']:7.1f} dBFS   (what a detector competes with)\n"
            f"    quiet floor  {lv['quiet_floor']:7.1f} dBFS   "
            f"({lv['muted_windows']} of {lv['windows']} windows gated as not-the-band)\n"
            f"    loudest      {lv['inband_p99']:7.1f} dBFS in-band, peak "
            f"{lv['peak_dbfs']:.1f} dBFS, railed {lv['railed_ppm']:.0f} ppm")


def _fmt_image(m: dict) -> str:
    if m["tone_db"] is None:
        return "    no carrier: capture too short to measure"
    if m["tone_db"] < _MIN_TONE_DB:
        return (f"    no carrier: strongest feature at {m['hz']:.0f} Hz stands only "
                f"{m['tone_db']:.0f} dB over its baseline")
    head = (f"    carrier {m['hz']:.1f} Hz at {m['fund_dbfs']:.1f} dBFS, "
            f"{m['tone_db']:.0f} dB over baseline\n")
    if m["image_dbc"] is None:
        return head + f"    images below the measurement floor of {m['floor_dbc']:.1f} dBc"
    per = ", ".join(f"{k}f0 {v:.1f}" if v is not None else f"{k}f0 <floor"
                    for k, v in sorted(m["harmonics"].items()))
    return head + (f"    images {m['image_dbc']:.1f} dBc ({per}), "
                   f"floor {m['floor_dbc']:.1f} dBc")


def _report(dr: dict) -> None:
    print("\n  quiet channel")
    print(_fmt_levels(dr["noise"]))
    print("\n  strongest signal")
    print(_fmt_levels(dr["signal"]))
    print(f"\n  DYNAMIC RANGE  {dr['range_db']:.1f} dB   "
          f"(quiet-channel p99 {dr['noise']['noise_p99']:.1f} to peak "
          f"{dr['signal']['peak_dbfs']:.1f}), need {MIN_RANGE_DB:.0f}"
          f"   {'PASS' if dr['ok'] else 'FAIL'}")
    print(f"\n  yardstick: {REFERENCE} is the session that decoded cleanly at this station.")
    print("\n  findings")
    for line in assess(dr):
        print(f"    - {line}")
    print("\n  deafness guards at this band noise")
    for line in guard_implications(dr["noise"]["band_noise"]):
        print(f"    {line}")


# ---------------------------------------------------------------------------- steps

def step_state(src: Source) -> dict:
    print("\n== state ==\n")
    rec = {"utc": datetime.now(UTC).isoformat(timespec="seconds")}
    for scope in ("input", "output"):
        v = src.codec_scalar(scope)
        rec[f"codec_{scope}_scalar"] = v
        print(f"  codec {scope} scalar: {v if v is None else f'{v:.4f}'}"
              f"{'  (transmit drive -- read only)' if scope == 'output' else ''}")
    print("\n  Nothing in this flock records the rig's own settings, which is why no past\n"
          "  level result can be reproduced. Read them off the radio now.\n")
    for key, prompt in (("data_out", "DATA OUT / USB AF level (0-100)"),
                        ("af_gain", "AF gain"),
                        ("agc", "AGC mode (off / fast / mid / slow)"),
                        ("rf_gain", "RF gain"),
                        ("preamp", "preamp / attenuator"),
                        ("band", "band and dial frequency")):
        rec[key] = _ask(src, prompt)
    return rec


def _ask(src: Source, prompt: str, default: str = "") -> str:
    if not src.interactive or not sys.stdin.isatty():
        return default
    return input(f"  {prompt}: ").strip() or default


def cmd_state(args):
    _save(args, {"state": step_state(_source(args))})


def cmd_measure(args):
    src = _source(args)
    rec = {} if args.skip_state else {"state": step_state(src)}
    print("\n== measure ==\n")
    src.act("quiet", "Tune to a QUIET frequency in the band you work, antenna connected, "
                     "nothing else keying.")
    noise = src.capture(args.seconds, "quiet channel")
    src.act("signal", "Now tune to the strongest signal you expect to work.")
    signal = src.capture(args.seconds, "strongest expected signal")
    dr = dynamic_range(noise, signal)
    _report(dr)
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        write_wav(Path(args.save) / "quiet.wav", noise)
        write_wav(Path(args.save) / "signal.wav", signal)
        print(f"\n  captures written to {args.save}/")
    _save(args, rec | {"measure": dr})


def cmd_check(args):
    out = {}
    quiet = signal = None
    for label, path in (("quiet", args.quiet_wav), ("signal", args.signal_wav)):
        if not path:
            continue
        audio, fs = read_wav(path)
        print(f"\n{label}: {path}  ({len(audio) / fs:.1f}s @ {fs} Hz)")
        lv = levels(audio, fs)
        print(_fmt_levels(lv))
        print(_fmt_image(image_dbc(audio, fs)))
        out[label] = dict(levels=lv, image=image_dbc(audio, fs))
        if label == "quiet":
            quiet = (audio, fs)
            print("\n  deafness guards at this band noise")
            for line in guard_implications(lv["band_noise"]):
                print(f"    {line}")
        else:
            signal = (audio, fs)
    if quiet and signal:
        dr = dynamic_range(quiet[0], signal[0], quiet[1])
        print()
        _report(dr)
        out["dynamic_range"] = dr
    _save(args, {"check": out})


def cmd_origin(args):
    audio, fs = read_wav(args.wav)
    r = harmonic_origin(audio, fs, args.tone, args.harmonic)
    print(f"\n{args.wav}  ({len(audio) / fs:.1f}s @ {fs} Hz)\n")
    print(f"  nominal tone {args.tone:.2f} Hz, harmonic {args.harmonic}, "
          f"{r['segments']} usable segments")
    if r["a_hz"] is not None:
        print(f"  fundamental offset  a = {r['a_hz']:+.2f} +/- {r['a_err']:.2f} Hz")
        print(f"  harmonic offset     b = {r['b_hz']:+.2f} +/- {r['b_err']:.2f} Hz")
        print(f"  predictions: far-end b = {r['a_hz']:+.2f}  ({r['sigma_far']:.1f} sigma), "
              f"local b = {args.harmonic * r['a_hz']:+.2f}  ({r['sigma_near']:.1f} sigma)")
    print(f"\n  {r['verdict']}")
    _save(args, {"origin": r})


def cmd_sweep(args):
    plan = sweep_plan(dwell=args.dwell, level_dbfs=args.level)
    if args.write:
        write_wav(args.write, sweep_signal(plan))
        total = len(plan["hz"]) * plan["dwell_s"]
        print(f"\n  {args.write}: {len(plan['hz'])} tones "
              f"{plan['hz'][0]}-{plan['hz'][-1]} Hz at {plan['level_dbfs']:.0f} dBFS, "
              f"{total:.0f}s\n")
        print("  Unplug the dongle from the radio. Loop its output back to its input with\n"
              "  a cable -- nothing else connected, the radio out of the picture entirely.\n"
              "  Play this file into the dongle's output while recording its input, then:\n")
        print(f"    python3 {sys.argv[0]} sweep --analyse recording.wav")
        return
    audio, fs = read_wav(args.analyse)
    report = sweep_report(audio, plan, fs)
    print(f"\n{args.analyse}  ({len(audio) / fs:.1f}s @ {fs} Hz)\n")
    print(f"  {'tone':>6}  {'dwell':>6}  {'3f0':>8}  {'5f0':>8}  {'floor':>8}")
    for r in report:
        m = r["image"] or {}
        cell = lambda v: f"{v:8.1f}" if isinstance(v, float) else f"{'-':>8}"  # noqa: E731
        print(f"  {r['hz']:6.0f}  {r['seconds']:5.1f}s  "
              f"{cell(m.get('harmonics', {}).get(3))}  {cell(m.get('harmonics', {}).get(5))}  "
              f"{cell(m.get('floor_dbc'))}")
    print(f"\n  {sweep_verdict(report)}")
    _save(args, {"sweep": report})


# ------------------------------------------------------------------------------ cli

def _source(args) -> Source:
    if args.simulate:
        return SimulatedSource(noise_dbfs=args.sim_noise, signal_dbfs=args.sim_signal,
                               drive=args.sim_drive, seed=args.seed)
    if not args.device:
        sys.exit("need --device N (see `hfcapture.py devices`) or --simulate")
    if not sys.stdin.isatty():
        sys.exit("the guided steps need an operator at the radio; run this on a terminal")
    return DeviceSource(args.device, args.codec_name)


def _save(args, record: dict) -> None:
    if not args.record:
        return
    path = Path(args.record)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(record)
    path.write_text(json.dumps(old, indent=2, default=str) + "\n")
    print(f"\n  written to {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", help="JSON file to accumulate results into")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def guided(name, fn, help):
        p = sub.add_parser(name, help=help)
        p.add_argument("--device", help="ffmpeg audio input index (hfcapture.py devices)")
        p.add_argument("--codec-name", default=None,
                       help="CoreAudio device whose scalars to read")
        p.add_argument("--seconds", type=float, default=20.0, help="capture length")
        p.add_argument("--simulate", action="store_true",
                       help="no hardware: run against a modelled receive chain")
        p.add_argument("--sim-noise", type=float, default=-35.0)
        p.add_argument("--sim-signal", type=float, default=-12.0)
        p.add_argument("--sim-drive", type=float, default=0.0)
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)
        return p

    guided("state", cmd_state, "record the rig settings nothing else writes down")
    p = guided("measure", cmd_measure, "the working point that exists right now")
    p.add_argument("--skip-state", action="store_true", help="do not ask for rig settings")
    p.add_argument("--save", help="directory to keep the two captures in")

    p = sub.add_parser("check", help="measure recordings instead of a radio")
    p.add_argument("quiet_wav", help="a quiet-channel recording")
    p.add_argument("signal_wav", nargs="?", help="a recording with the strongest signal")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("origin", help="harmonic offsets: made here, or made at the far end")
    p.add_argument("wav")
    p.add_argument("--tone", type=float, required=True,
                   help="nominal audio frequency of the tone as transmitted, in Hz")
    p.add_argument("--harmonic", type=int, default=3)
    p.set_defaults(fn=cmd_origin)

    p = sub.add_parser("sweep", help="fixed-level loopback tone sweep, dongle alone")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", help="write the sweep to this WAV for the operator to play")
    g.add_argument("--analyse", help="a loopback recording of the sweep")
    p.add_argument("--dwell", type=float, default=_SWEEP_DWELL_S)
    p.add_argument("--level", type=float, default=_SWEEP_LEVEL_DBFS)
    p.set_defaults(fn=cmd_sweep)

    args = ap.parse_args()
    if hasattr(args, "codec_name"):
        from hfmodem.core.rigs import launcher_defaults
        launcher_defaults(args, audio_fields={"codec_name": "input"},
                          legacy={"codec_name": "USB Audio Device"})
        if args.rig != "ft891" and args.device is None:
            args.device = args.codec_name
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
