# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Is the channel in use? Receive-only, and deliberately biased toward "yes".

Two earlier versions of this check were each calibrated on one signal class and
went blind to the others. The first scored ``peak / median`` inside the passband,
which a 2300 Hz occupant defeats by lifting both. The second normalised in-band
power by the spectrum above 3200 Hz on the theory that an occupant cannot reach
there — but that band is past the receiver's SSB filter skirt, so the denominator
was codec noise, and its jitter went straight into the verdict. Both scored
``max - min`` across the window, which reads a *steady* occupant as clear (0.9 dB
on a steady wideband signal, 0.9 dB on a steady carrier) and reads digital silence
as clear too (0.0 dB) — vacuous exactly where it must not be.

The third failed the other way, and cost more. ``burst`` was the 90th-to-10th
percentile spread of in-band level, calibrated on a single 30 s quiet capture that
gave 0.96 dB, and 40 m on a summer evening gives 3.1 to 3.8 dB of the same figure
with nothing on the frequency at all. On 2026-08-09 every channel of the slot came
back occupied by +0.6 to +0.7 dB, including one the operator then listened to for
45 s and heard nothing on — so ``--force`` became reflex, and the guard was
carrying no information on the two channels that really were carrying somebody
else's session. A guard that fires always is a guard that gets switched off.

What replaces them is grounded in the signal's structure rather than in a ratio
against a reference the receiver does not really provide. Three scores, all in dB,
because no one of them sees every occupant:

``burst``  The strongest lift of in-band power that is *held*, meant to read an
           over rather than a crack. It does not: across 36 kept windows it is
           2.2 to 4.0 times the standard deviation of the in-band level, and
           shuffling the frame order — which leaves the window no durations at
           all — costs only 1.5 to 2.3 dB on the seven that refused a call. It is
           measured and reported on every verdict and decides none: ``DECIDING``.
``shape``  Temporal contrast of the *level-normalised* spectrum, per sub-band.
           This is what catches an occupant that never stops keying, whose total
           power is therefore steady: its spectral shape still moves with the
           modulation, while noise keeps the receiver's fixed response. Being
           level-blind it also ignores AGC pumping, QSB and receiver muting, none
           of which change the shape.
``tone``   Excess of the strongest narrow feature over a locally smoothed
           baseline. This is the carrier/CW detector, and it is the one that sees
           a steady unmodulated carrier, which by definition moves neither of the
           other two.

The caller says what passband it is about to fill, and each score is measured over
as much of it as that score can be measured over: `tone` all the way down, `shape`
no narrower than ``MIN_SHAPE_HZ``, `burst` never — it is a total-power score, the
only occupant it sees is one wide enough to be in every band at once, and
narrowing it moves its two populations together rather than apart.

Measured in-band SNR at which each occupant class first trips a threshold, mixed
into a verified-clean off-air noise capture (40 m, FT-891, nothing else keying):

    steady wideband data   -6.0 dB   (shape)
    bursty ARQ wideband    -7.0 dB   (shape)
    steady carrier         -8.0 dB   (tone)
    keyed CW              -10.5 dB   (shape)

Every one of those is a `shape` or a `tone`. There is no occupant class `burst`
is the floor for.

Liveness comes first. A receiver that is muted, deaf or dropping out delivers
beautifully steady samples and every occupancy score reads clear — the most
dangerous failure this module can have. Frames whose in-band power collapses
``MUTE_DROP_DB`` below the window are dropped before anything is scored (that is
this rig transmitting, or the codec glitching, not a quiet band), and a window
left with less than ``MIN_LIVE_S`` of real audio yields no verdict at all, which
the bias resolves as busy. `receiver_fault` is the operator's side of that same
finding, and the tools that listen before keying all print it: a channel verdict
read off a receiver this module would not trust is not a channel verdict.

Bias: a false "busy" costs a wait, a false "clear" transmits over somebody.

Who asks. The operator's tools — `tools/onair_session.py`, which refuses to call
a busy channel unless the operator forces it, and the monitors and surveys beside
it. Listening before transmitting is the operator's duty and stays with the
operator: the station's TX arbiter does not consult this, and the one that
claimed to never could — the receive window it judged was a stub returning no
samples, so its gate read every channel as clear behind comments promising
otherwise. Anyone wiring this into a transmit path anyway: it must **never** run
before answering an ARQ interrogation. The peer that is interrogating us is
itself the occupant the score will find, so a reply gated on this is a reply
suppressed by the very signal it answers, and the link fails in the one way that
looks like a receive fault at both ends.

Nor is a refusal waived by naming the occupant, which would be the safe shape: a
channel occupied solely by the station we are answering is not a channel we would
transmit over. That needs the station KEYING identified, and an occupant holding
an ARQ link puts no such thing on the air — an ARDOP ConAck or data frame and a
PACTOR-1 control signal carry no callsign at all, and PACTOR's one addressed
frame carries the callsign of the station being CALLED. Measured over 240.1 s of
exactly that occupant (the gateway of the 2026-08-18 refusal at +12.0 dB on
7101500 kHz, and two continuous minutes of another calling with our transmitter
off): 201 detections and 41,108 sync alignments name no station, and the two
callsigns the connect decoder does mint out of it are not stations. A waiver has
nothing here to key on, and `tests/core/test_occupant_identity.py` is where that
stops being true if it ever does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

FS = 48000
_FRAME_S = 4096 / 48000        # 85 ms frames, half-overlapped

# The frame is a DURATION, and the FFT size follows the rate it is handed. Fixed
# at 4096 samples it was 85 ms only on a 48 kHz stream: a KiwiSDR's 11999 Hz gave
# a 341 ms frame, which averages across exactly the keying gaps `burst` exists to
# find. `tools/gwsurvey.py` read every occupancy figure it ever printed that way,
# and correcting it moved the reading on three of the five channels in one
# survey -- one of them from 2 of 7 windows busy to 7 of 7.
#: The widest band judged, and it is the passband the rig is asked for --
#: `core.occupied.FILTER_HZ` is this number, not one of its own. `_widen` clips
#: every declared emission to it, so a top edge under the filter leaves air
#: inside the receiver that nothing here scores: at 2700 it took 265 Hz off
#: sabir's own 140-2860 Hz row, which is the top of what that gear is about to
#: key, and a carrier at 2707 Hz on 7103.5 kHz read 12 Hz below the edge because
#: the tone search could not reach its peak. The corpus puts the receiver's -6 dB
#: point at 2906-3018 Hz across seven captures and flat to within 2 dB at 2800,
#: and nothing is bought by stopping short of it: false-busy holds at 2 of 78
#: verified-empty windows and false-clear at 1 of 2 for every top edge from 2700
#: to 3200. `shape` is fenced separately by `_SKIRT_FREE` and `tone` is an excess
#: over a LOCAL median, which subtracts a filter skirt rather than importing it.
FULL_BAND = (400.0, 3000.0)
# `shape` weighs sub-bands against each other, so it must stay off the SSB filter
# skirts: that is where the receiver's own response is steepest, and there a small
# AGC or tuning movement reads as a change of shape rather than of level.
_SKIRT_FREE = (500.0, 2600.0)
# The narrowest band `shape` is measured over, whatever the emission. Over the
# seven windows this sweep was run on its worst climbs 4.73 -> 5.74 -> 7.54 dB as
# the band narrows 2300 -> 1000 -> 400 Hz, against a threshold of 6.0 — so 1000 Hz
# is a floor and not a margin: 0.26 dB of one, and less than that against the
# 78-window clear side at SHAPE_DB, which reaches 8.47 at the widest of the three.
# Move it up against a population measured narrower, never down.
MIN_SHAPE_HZ = 1000.0
#: The shortest window these thresholds are measured over, and the figure is set
#: from the occupied side because that is where the cost falls. A caller keys on
#: two consecutive clear windows: over a whole live gateway session on 7103.5 kHz
#: it never gets them at 8 s, and at 5 s it gets them in the turnarounds.
WINDOW_S = 8.0
#: The window a pounce waits in, and it wants three clear ones in a row. Over the
#: same live gateway session 3 s windows refuse 29 of 35 and the longest run of
#: clears is two, so three never keys into it; on the empty control 10 of 10 read
#: clear, so nothing here holds an operator off a free channel.
POUNCE_WINDOW_S = 3.0
# ... and it reads single frames. Averaging pairs of them first looked like free
# variance reduction and was not: what it averages away is the modulation, which
# is the whole of the signal here, while the noise it suppresses is common to
# every sub-band and has already gone with the level. Dropping it lifted the
# quiet windows of a real gateway session from 5.1-5.5 dB to 7.0-7.6 against a
# 6.0 dB threshold, at a cost of 0.7 dB on the clear side.
_NSUB = 6                      # sub-bands across the passband, for `shape`

# How long a lift has to hold before `burst` will count it, and READ THE FIGURE
# AS A QUARTER OF WHAT IT SAYS: the filter below is a *median*, exceeded once half
# the frames in the hold are up, so 0.5 s rejects what is shorter than about
# 0.25 s. That is the 20-80 ms crack of a summer 40 m evening and nothing longer.
#
# No value of it does the job the score was named for. The lift on 7103.5 that
# `burst` alone catches measures 0.30 s and the 80 m crashes measure 0.30-0.34,
# so the median reaches both together or neither: swept from one frame to half
# the window the occupant stays inside the clear population at every length, and
# under the whole of it at most of them. `tests/kestrel/test_channel_busy.py`
# holds the sweep, and `DECIDING` below is what follows from it.
_BURST_HOLD_S = 0.5
_SMOOTH_HZ = 58.0              # spectral smoothing before the tone baseline
_BASE_HZ = 700.0               # width of the tone score's local baseline

# A receiver that is not hearing the radio still delivers samples — the codec's own
# noise floor — so silence has to be measured off the signal. Measured at the bench:
# a muted input sits at -75 dBFS, live receiver audio at -10. Windows where we had
# the radio to ourselves spanned 2-10 dB; windows where another modem was keying it
# spanned 38-63 dB. The threshold sits in the empty gap.
DEAF_DBFS = -50.0
MUTE_DROP_DB = 25.0
# Below this the tone estimate is too noisy to trust. It was 4.0 until the pounce
# window went to 3 s, which no 4 s floor can ever judge: at 2.5 the empty control
# reads clear on 10 of 10 three-second windows, and at 2.0 the live gateway
# session on 7103.5 opens a run of three clears -- a pounce into a held link.
MIN_LIVE_S = 2.5

# Thresholds. Each sits between the highest value measured on verified-clear
# receiver audio and the lowest measured on the real occupancy that only that
# score sees. The figures below are the two numbers either side of each gap, over
# 8 s windows of the recordings named in tests/kestrel/test_channel_busy.py; that
# module asserts both sides, so the calibration moves with the corpus rather than
# with an opinion.
#
# BURST_DB was 5.5 until 2026-08-10, when a control recording at 6950 kHz —
# outside the band, verified by readback, nothing there — read busy 3 of 3 over
# the 10 s windows the survey listens in, at margins +0.5 to +0.9, and so did
# every amateur channel of the slot. That night's static holds its lifts past
# the 0.5 s that separated crashes from overs on 2026-08-09: the control's
# `burst` reaches 5.93 over 8 s windows and 6.38 over 10 s, against 3.76 for
# every clear capture the old figure was set on, while the weakest occupant only
# `burst` catches still reads 7.86. The threshold sits between those two.
# Overlap: over the whole 30 s the same control accumulates 6.87,
# so at windows much past the 8-20 s the tools listen in, `burst` alone cannot
# clear this control with real margin — the occupied verdicts there ride on
# `shape`, which read 9.1-29.9 on the occupied channels against 3.0 on the
# control.
# WHAT THAT GAP IS WORTH ON 80 M, WHICH IS NOWHERE IN THE POPULATION ABOVE. Every
# clear-side figure here was taken on 40 m or out of band. Through this station's
# own receiver on 3585 and 3595 kHz, in the window after our own transmission
# stops, a single static crash reads 9.98 and 9.06 dB of `burst` with `shape` and
# `tone` both at their noise floors — above the 7.86 of the occupant `burst` alone
# catches, and the same 0.30-0.34 s long. The two populations are not close, they
# are inverted, and no value of BURST_DB separates them. It is a reason to keep
# the audio of a refusal, which `tools/lib/rig.sh` does, and not a reason to move
# this number.
# NEITHER DOES `shape`, and the figure beside it used to claim otherwise. The 4.73
# was the whole of `CLEAR_CHANNEL` -- three windows of one 30 s capture -- while
# the verified-empty population is the 78 windows counted above, and
# `MONITORED_QUIET`, the ten minutes of 7103.5 kHz the feed and every classifier
# agree were empty, puts two of them at 6.61 and 8.47. The clear side runs 2.5 dB
# PAST this threshold rather than stopping 1.3 dB short of it, so the 2 of 78
# false-busy recorded above is not a rounding of a clean separation, it is the
# separation.
#
# 6.0 stays anyway: clearing that tail costs half of what `shape` catches -- 17.4%
# of the windows of the off-air gateway sessions reach 6.0 and 9.4% reach 8.47 --
# and on the bias above a wait is the cheaper error. What re-reads a refusal on the
# tail is WHICH sub-bands moved, which the max below drops. Both empty windows are
# one or two adjacent bands of the six, third-highest 3.95 and 4.02; the occupied
# recordings reach 12.02 there, and the two 40 m refusals of 2026-08-29 that were
# read as false read 6.57 and 5.43 -- outside the empty population on the measure
# the verdict does not carry. It is recovered from the audio `tools/lib/rig.sh`
# keeps, which is the whole reason that audio is kept.
BURST_DB = 6.6                 # clear max 9.99, the occupant only `burst` sees 7.86
SHAPE_DB = 6.0                 # clear reaches 8.47, the occupant only `shape` sees 6.96
TONE_DB = 5.0                  # clear max 3.57, carrier at -8 dB SNR 5.07
THRESHOLD_DB = dict(burst=BURST_DB, shape=SHAPE_DB, tone=TONE_DB)

#: The scores a refusal may rest on. Every window still carries all three, and
#: `burst` is on every printed verdict and in every kept sidecar, because it is
#: what an inverted population looks like and the operator re-reads refusals by it.
#:
#: It cannot be one of these. Its clear side reaches 9.98 dB and the strongest lift
#: it alone catches on a channel known to be occupied reads 7.86, and the two are
#: alike in every other way that can be measured: that lift holds 0.30 s against the
#: crashes' 0.30-0.34, and spread across the passband it is 1.37 dB against their
#: 0.78-3.17 — flat, where the sessions this gate exists for run 4.48 to 40.37. It is
#: a crash: 11.3 dB crest and kurtosis 3.29 over its length, against 8.9-10.6 and
#: 2.45-2.93 for the six 80 m windows — past their edge in the impulsive direction,
#: which is the wrong way round for a station. Retiring it costs two isolated windows
#: over seven occupied recordings (74 read busy, 72 without it) and buys back six
#: refusals of eight on verified-empty band audio, four of them a channel that was
#: keyed over immediately afterwards and held nothing.
#:
#: The defence of that trade is not that no caller gets the two or three consecutive
#: clear windows it keys on — `tools/onair_session.py` listens once per target and
#: calls on that single window. It is what the window holds. The one that has been
#: read is the eight seconds of 7103.5 kHz before this station keyed on 2026-08-09,
#: and it does hold a station: a carrier at 806.8 Hz through its first 2.2 s, coherent
#: to under 1 Hz across them, at -37.5 dBFS — 17.2 dB UNDER the in-band noise. `tone`
#: finds it, at 808.6 Hz, and scores it 2.26 dB; the weakest carrier that threshold is
#: set for is -8 dB SNR at 5.07 and the clear side already reaches 3.33, so there is
#: nowhere left to put a number that catches this one. What refused the window was
#: `burst` 7.86, and `burst` was not looking at the carrier: 0.25 s of dense
#: atmospheric impulses, lifting the passband 6.9 dB and 3250-4000 Hz by 21 — above
#: the receiver's own SSB filter, where nothing on the channel can put energy at all.
DECIDING = ("shape", "tone")


def _sizes(fs: float) -> tuple[int, int]:
    """``(nfft, hop)`` for a ``_FRAME_S`` frame at FS, half-overlapped. Exactly
    ``(4096, 2048)`` at 48 kHz, so the calibrated rate is untouched."""
    hop = max(64, int(round(fs * _FRAME_S / 2)))
    return 2 * hop, hop


def _frames(audio: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-overlapped power spectra and their bin frequencies."""
    nfft, hop = _sizes(fs)
    n = (len(audio) - nfft) // hop + 1
    if n < 8:
        return np.zeros((0, nfft // 2 + 1)), np.fft.rfftfreq(nfft, 1 / fs)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    P = np.abs(np.fft.rfft((audio - audio.mean())[idx] * np.hanning(nfft), axis=1)) ** 2
    return P, np.fft.rfftfreq(nfft, 1 / fs)


def _widen(band: tuple[float, float], least_hz: float) -> tuple[float, float]:
    """``band`` clipped to what the receiver passes, at least ``least_hz`` wide."""
    lo = max(float(band[0]), FULL_BAND[0])
    hi = min(float(band[1]), FULL_BAND[1])
    if hi - lo >= least_hz:
        return lo, hi
    half = least_hz / 2
    mid = min(max((lo + hi) / 2, FULL_BAND[0] + half), FULL_BAND[1] - half)
    return mid - half, mid + half


def occupancy_db(audio: np.ndarray, fs: int = FS,
                 band: tuple[float, float] = FULL_BAND) -> np.ndarray:
    """Per-second in-band level in dBFS, over the audio that is actually live.

    An absolute level, not a ratio against an out-of-band reference: the spectrum
    past the SSB filter skirt is the codec's own noise, and dividing by it only
    imports its jitter.
    """
    audio = np.asarray(audio, float)
    n = int(fs)
    secs = len(audio) // n
    if secs == 0:
        return np.zeros(0)
    lo, hi = band
    f = np.fft.rfftfreq(n, 1 / fs)
    inb = (f > lo) & (f < hi)
    win = np.hanning(n)
    # Parseval, with the window's power gain divided back out, so the figure is the
    # in-band RMS of the audio in dBFS and comparable between captures.
    norm = 2.0 / (n ** 2 * (win ** 2).mean())
    out = np.empty(secs)
    for i in range(secs):
        p = np.abs(np.fft.rfft(audio[i * n:(i + 1) * n] * win)) ** 2
        out[i] = 10 * np.log10(max(p[inb].sum() * norm, 1e-30))
    return out


def scores(audio: np.ndarray, fs: int = FS,
           band: tuple[float, float] = FULL_BAND) -> dict | None:
    """``{burst, shape, tone, tone_hz, live_s}`` in dB, or ``None`` when the input
    is not live enough to judge — a muted or dead receiver, which must never read
    clear.

    ``band`` is the audio passband the caller is about to occupy, and only the two
    scores that are about *where* the energy sits narrow to it. ``burst`` is a
    total-power score and stays on :data:`FULL_BAND`, because narrowing it closes
    the gap rather than opening one: the clear-side worst climbs 3.76 -> 5.38 dB
    as the band narrows to 400 Hz while the occupant only ``burst`` catches falls
    7.86 -> 7.47, and the one occupant it can see at all is wide enough to be in
    both bands anyway.
    """
    audio = np.asarray(audio, float)
    nfft, hop = _sizes(fs)
    P, f = _frames(audio, fs)
    if len(P) == 0:
        return None
    df = f[1] - f[0]
    tone_lo, tone_hi = _widen(band, 0.0)
    shape_lo, shape_hi = _widen(band, MIN_SHAPE_HZ)
    inb = (f >= FULL_BAND[0]) & (f <= FULL_BAND[1])

    # Deafness is absolute and has to be tested first: when the input is *uniformly*
    # dead there is no collapse for a relative gate to find, every score reads 0,
    # and silence would be reported as the clearest channel on the band.
    frame_dbfs = 10 * np.log10(P.sum(axis=1) * 2 / (nfft ** 2 * 0.375) + 1e-30)
    if frame_dbfs.max() < DEAF_DBFS:
        return None

    level = 10 * np.log10(P[:, inb].mean(axis=1) + 1e-30)
    dead = level < np.percentile(level, 95) - MUTE_DROP_DB
    # a frame beside a dead one straddles the mute edge, so it is not band audio either
    live = ~(dead | np.r_[dead[1:], False] | np.r_[False, dead[:-1]])
    live_s = live.sum() * hop / fs
    if live_s < MIN_LIVE_S:
        return None

    lv = level[live]
    hold = int(round(_BURST_HOLD_S * fs / hop)) | 1
    burst = float(median_filter(lv, size=hold, mode="nearest").max()
                  - np.percentile(lv, 10))

    Q = P[live]
    edges = np.linspace(max(shape_lo, _SKIRT_FREE[0]),
                        min(shape_hi, _SKIRT_FREE[1]), _NSUB + 1)
    sub = np.array([10 * np.log10(Q[:, (f >= a) & (f < b)].mean(axis=1) + 1e-30)
                    for a, b in zip(edges[:-1], edges[1:])])
    sub -= sub.mean(axis=0)                       # drop the common mode: level, not shape
    shape = float(max(np.percentile(d, 90) - np.percentile(d, 10) for d in sub))

    # The 75th percentile across frames, not the mean: an occupant present for a
    # quarter of the window still shows, and the baseline is drawn from the same
    # statistic so its bias cancels.
    s = np.percentile(10 * np.log10(P[live] + 1e-30), 75, axis=0)
    nb = max(3, int(round(_SMOOTH_HZ / df)) | 1)
    s = np.convolve(s, np.ones(nb) / nb, "same")
    base = median_filter(s, size=int(round(_BASE_HZ / df)) | 1, mode="nearest")
    inb_tone = (f >= tone_lo) & (f <= tone_hi)
    excess = (s - base)[inb_tone]
    tone = float(excess.max())
    return dict(burst=burst, shape=shape, tone=tone,
                tone_hz=float(f[inb_tone][excess.argmax()]), live_s=live_s)


def dominant(s: dict) -> tuple[str, float]:
    """Which of :data:`DECIDING` decided, and how far above its threshold it sits.

    :func:`is_busy` returns that distance without the name, and the name is the
    difference between an occupant lying across the passband and a carrier sitting
    beside it — the second is the only refusal a narrower judged band can ever
    waive, and ``tone_hz`` says whether it lies inside what we are about to fill.
    A record carrying the margin alone cannot be re-read for either question.
    """
    return max(((k, s[k] - THRESHOLD_DB[k]) for k in DECIDING),
               key=lambda kv: kv[1])


def is_busy(audio: np.ndarray, fs: int = FS,
            band: tuple[float, float] = FULL_BAND):
    """``(busy, per-second in-band dBFS, margin)`` over ``band``.

    ``margin`` is how far the strongest score sits above its own threshold, in dB:
    positive means occupied, negative is headroom on a clear channel. A window too
    muted or too short to judge returns busy with a margin of 0.0 — no evidence is
    not evidence of a clear channel.
    """
    db = occupancy_db(audio, fs, band)
    s = scores(audio, fs, band)
    if s is None:
        return True, db, 0.0
    margin = dominant(s)[1]
    return bool(margin >= 0.0), db, float(margin)


# -- the receiver itself, which is a different question ----------------------

def level_range(audio: np.ndarray, fs: int = FS) -> tuple[float, float]:
    """Loudest and quietest 100 ms of a window, in dBFS.

    The two numbers `receiver_fault` reads, and both ends of the range matter: a
    floor held across the whole window is an input that is not hearing the radio
    at all, while a floor reached only in places is another transmitter muting
    it — which is a difference a mean over the window erases.

    A window with no samples in it at all is the device that opened and delivered
    nothing, and its level is the floor: the mean of an empty slice is nan, and
    nan is below no threshold and above none either, so `receiver_fault` reads it
    as a receiver worth believing.
    """
    audio = np.asarray(audio, float)
    if audio.size == 0:
        return -np.inf, -np.inf
    win = max(1, fs // 10)
    rms = [np.sqrt((audio[i * win:(i + 1) * win] ** 2).mean())
           for i in range(max(len(audio) // win, 1))]
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    return float(db.max()), float(db.min())


@dataclass(frozen=True, slots=True)
class ReceiverFault:
    """Why this input may not be believed, in one word and in one sentence."""

    kind: str               #: "deaf" | "intermittent"
    reason: str             #: the operator's line, and the only wording for it


def receiver_fault(loud_dbfs: float, quiet_dbfs: float, *, seconds: float,
                   device) -> ReceiverFault | None:
    """Whether this receiver can be trusted at all, or None if it can.

    Asked BEFORE any occupancy verdict is read, and by everything that listens
    before keying. A receiver that is not hearing the radio delivers the codec's
    own noise floor, which scores as the clearest channel on the band; a receiver
    something else is keying delivers real band audio interrupted by its own
    mute, and a gateway answering inside one of those gaps is simply not heard.
    This was three private copies of one judgement — the same two constants,
    three wordings, and three different answers on whether ``--force`` got past
    it.

    It does not. ``--force`` is the operator saying the band sounds workable to
    *him*, which is a judgement about the channel; neither of these is about the
    channel, and nothing he can hear says our input is attached. A station that
    cannot hear a channel may not call on it, so the sentence says so and no
    caller offers a way past it.
    """
    if loud_dbfs < DEAF_DBFS:
        return ReceiverFault("deaf", (
            f"RECEIVER DEAF: nothing above {loud_dbfs:.0f} dBFS in {seconds:.0f} s "
            f"— that is the codec noise floor, not a quiet band. Every channel will "
            f"read busy and no gateway could be heard answering one of them. Check "
            f"the rig is on, its volume up, and its audio reaching {device!r}. "
            f"--force does not make a deaf receiver hear."))
    if loud_dbfs - quiet_dbfs >= MUTE_DROP_DB:
        return ReceiverFault("intermittent", (
            f"RECEIVER INTERMITTENT: the input collapses "
            f"{loud_dbfs - quiet_dbfs:.0f} dB (to {quiet_dbfs:.0f} dBFS) inside the "
            f"{seconds:.0f} s window — something else is keying this rig. A "
            f"transmitting radio mutes its own receive audio, so a gateway "
            f"answering during one of those gaps is simply not heard. --force "
            f"weighs a judgement about the band, and this is not one."))
    return None
