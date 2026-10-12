#!/usr/bin/env python3
"""Record and read this station's own transmission, off its own monitor.

WHY THIS EXISTS. Until 2026-08-14 the only way to hear what this station put on
the air was a KiwiSDR 103 miles off, which answers the question only for a path
that has propagation in it and never answers it at all for a band that is shut.
The rig's speaker output now feeds a capture interface the modems do not own, and
the rig's MONITOR puts the transmit audio on it, so the emission can be measured
in the room it is made in.

WHAT IT MEASURES, and it is the part no log can report: the anatomy of a keying.
The tap sees the receiver mute when PTT goes up and recover when it drops, and it
sees the transmit audio in between, so all four instants come off one recording —

    PTT up ........ audio starts ........ audio ends ........ PTT down
           \\__LEAD__/                             \\__TAIL__/

Both matter and for the same reason. A peer answers inside its turnaround, 70 to
130 ms for PACTOR-1, so a station still keyed a second after its audio stopped is
transmitting across the window its answer arrives in. `shrike.onair` keyed 0.988 s
for 0.960 s of audio; `shrike.beacon` was measured on 2026-08-14 keying 2.02 and
2.76 s for 0.82 and 0.92 s, with a 1.12 s tail on both. Nothing reported it,
because nothing was listening.

Those four figures are the record of what was found, not a current reading. On
2026-08-15 every transmit path moved onto a drained playback: until then the
device's last buffers were discarded, so the audio each of them was measured
against ended 3.42 to 12.75 ms early. The keyed spans stand; the audio spans, and
therefore the tails, have to be taken again.

    ./tools/txwitness.py record 30 out.wav      # tap for 30 s
    ./tools/txwitness.py measure out.wav        # the anatomy above
    ./tools/txwitness.py edges out.wav          # the same, at one millisecond

THE SECOND TAP NEEDS NO INTERFACE AT ALL, and it was there before this file was.
A shrike session's `stream.wav` is written from `_LiveInput`'s capture callback in
front of the half-duplex discard, so the receiver's mute, our own emission back
through the rig, and the peer's answer are all in it on one sample clock — the
same clock every window sidecar's `end_stream_sample` indexes into. Point either
command at one. On the 2026-08-19 sessions our own bursts sit at -42 dBFS against
a -88 dBFS mute and correlate 0.97 against the very samples that made them — but
what they arrive by is the codec's own DAC-to-ADC crosstalk, an electrical path
whose body moves 0.46 dB across three bands, two days and 22 dB of receive chain.
Levels read off one are the codec's, not the band's, and the edges below carry
the caveat every edge on this rig carries.

`measure` reads its instants on `STEP_S`, which resolves them to two PACTOR-1
bits — the right grain for a 1.12 s tail and useless for the question of where
the body of a twelve-bit codeword arrives. `edges` reads the same instants on a
4 ms window every millisecond.

NO RECORDING MADE THROUGH THIS STATION'S OWN RECEIVER YIELDS AN EMISSION LENGTH,
and `edges` no longer offers one. Both ends of the audio are moved by that
receiver. The rig flushes its muted receive pipeline when PTT drops, so the last
two bits of every burst never reach the file. And while the receiver sits muted a
low-frequency transient lands in the passband ahead of the burst, which the
leading edge finds instead on some sessions and not others.

What survives both is KEY-UP TO THE END OF THE AUDIO, and the corpus is what says
so. All 20 sessions of 2026-08-19/20 sent the same 960 ms connect: the end of the
audio comes back 986.5 to 990.0 ms after key-up on every one of them, 3.5 ms of
spread over three bands and two days, while the leading edge inside that span
swings 36.0 to 51.0 ms in two clusters 13 ms apart. Subtracting the two therefore
produces 17 ms of emission length that no recording here contains — 937 to 954 ms
of the same 960 ms burst, which is the figure this printed until 2026-08-25 and
the reason it does not now.

The swing is the detector, and two threshold-free readings say so. Correlating
each burst against the render that made it puts the head a flat 20.8 to 23.9 ms
behind the DAC on all 20 and finds exactly two bits gone off every tail
(`tests/shrike/test_txhead.py`). Correlating the render across the whole keying
finds it ONCE, at 0.97, 46 to 51 ms after key-up — including on the sessions this
reads at 36 — so what those readings land on is not an early copy of the burst.
Set the invariant against the keyed span the session logged; an emission length
needs a clock that is not ours, and `tools/kiwi_witness.py` records one.

`measure` reports an audio span rather than the two instants, at a grain coarser
than the error, and it keeps it: the tails it exists for run to 1.12 s. Read that
span as the envelope of a keying and not as a length on the air.

RECORDING WHILE A MODEM WORKS is the point: this reads a different device from
the one the modems transmit through, so it costs them nothing and needs no
cooperation from them.

THE FILE IS ON DISK WHILE IT IS BEING MADE. `record` writes through
`core.wav.CaptureRecorder`, which patches the wav header behind every block, so
the file is valid and holds everything captured up to the last drain at any
instant a reader or a crash arrives. A kill, an unplug or a segfault costs
`DRAIN_S` and no more, and `measure` reads a tap that is still running. It
cannot do better: a USB interface pulled mid-recording takes the process down
with a SIGSEGV raised inside CoreAudio, where no Python handler runs and no
PortAudio error is ever returned, so writing through is the whole mitigation.
So `seconds` need not be guessed: give the tap the length of the slot and stop
it with SIGTERM when the slot ends.

NOTHING HERE SERIALISES A SECOND TAP, and nothing should: this card has taken a
second and a third reader while the operator's own monitor held it. What made a
second recorder wait for the first was a `pgrep` loop in front of it, and there
is nothing left to queue for now that the running recorder's file can be read
while it is still being written.

WHERE THE AUDIO IS, is read off the level in the whole transmit passband against the
unkeyed baseline, so it reads the same for shrike's 200 Hz FSK pair, for ARDOP, and
for VARA's 2300 Hz of OFDM. Nothing here may key off a particular modem's tones: a
test for a tone pair cannot see OFDM at all, and cannot see FSK either, which sends
one tone at a time.

THE MONITOR HAS TO BE LOUD ENOUGH OR THE ANSWER IS WRONG, and wrong quietly.
Both 2026-08-14 recordings hold the same two keyings; at `MONITOR_GAIN` 0.2 the
monitor sat 7.6 dB over the tap's noise floor and this read the audio as 0.20 s
starting 700 ms in, where at 0.5 — 39 dB over the floor — the same burst reads
0.82 s starting 80 ms in. The head of a connect comes up softest, so it is the
LEAD that a quiet monitor eats; the tail came out 1120 ms either way. Set the
monitor first — measure the selected converter gain before transmission. `measure` prints what the
monitor read before it prints a single keying and checks it against
:func:`monitor_headroom`; failing that it says so and exits 1, because a slot that
ran 7x hot published figures that had to be withdrawn.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import wave
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:      # noqa: F401 — the message is the point
    raise SystemExit(
        f"{exc.name!r} is not on this interpreter's path. Run this through the "
        "project venv:\n"
        f"    ./.venv/bin/python tools/txwitness.py {' '.join(sys.argv[1:]) or '...'}"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/hfmodem"))

FS = 48000
#: The transmit passband, wide enough for every modem this station hosts: shrike's
#: 1400/1600 FSK pair, ARDOP centred on 1500, VARA BW2300 spread over 680-2297.
BAND = (300.0, 2800.0)
#: Edges of the sub-bands the mute is read across, tiling the receiver's audio
#: output from the foot of the transmit passband to well above it. No single band
#: decides: see :func:`out_of_band_level`.
MUTE_BANDS = np.linspace(300.0, 4000.0, 8)
#: A step is inside the mute once the level has fallen this far below the receiver
#: listening, and stays inside it until the level comes back up to
#: :data:`LIVE_AGAIN`. One threshold cannot do both: the 2026-08-15 captures cross
#: any single one of them repeatedly inside one continuous keying, which is what
#: turned 21 logged keyings on `20260815-191006-mail80.wav` into 224 fragments.
MUTED, LIVE_AGAIN = 0.05, 0.40
#: A transient does not end a keying, and two steps of collapse are not one.
#: 0.10 s clears the shortest keying this station makes — a PACTOR-1 control
#: signal, 0.12 s of audio inside a 0.15 s key — by half a step.
MUTE_GAP_S, MIN_KEYING_S = 0.06, 0.10
#: A PACTOR-1 bit at 100 Bd, which is the shortest thing this station keys and
#: the unit every lead and tail here is worth reading in.
BIT_S = 0.010
#: 20 ms: a PACTOR-1 bit is 10 ms at 100 Bd, so this resolves a lead or tail to
#: two bits. Finer than the quantity is worth and coarse enough to be stable.
STEP_S = 0.02
#: How long a stretch the listening level is estimated over when it is tracked
#: rather than taken across the whole file: see :func:`receiver_live_track`. Long
#: enough to hold several overs and the listening between them for every modem
#: here — PACTOR-1 cycles at 1.25 s and the ARDOP and VARA overs in
#: the recorded transmitter-monitor trials run 1 to 2.2 s — and short enough that an hour-long tap is
#: 240 estimates of the receiver rather than one.
LIVE_TRACK_S = 15.0
#: Transmit audio is present when the in-band level is within this of the
#: receiver-live level. Across the 2026-08-15 captures the keyed-and-modulating
#: steps sit 12 to 31 dB above it and the keyed-and-silent steps 21 to 27 dB below.
AUDIO_RATIO = 0.05
#: What the monitor has to read for anything measured off it to mean anything, and
#: neither figure is a level, so neither has to be calibrated: see
#: :func:`monitor_headroom`. The 2026-08-15 captures peak at the rail with 1 to 157
#: railed samples apiece, where the one made at a working gain reaches 0.70 and
#: rails nothing; at `MONITOR_GAIN` 0.200 the transmit audio sat 7.6 dB over the
#: tap's noise and no audio was found in any keying, against 23.1 dB at 0.330.
MONITOR_HEADROOM_DB, MONITOR_OVER_FLOOR_DB = 2.0, 12.0
#: The grain the EDGES are read at. :data:`STEP_S` resolves them to two bits at
#: 100 Bd, and two bits is the whole of what `edges` exists to report, so it reads
#: a 4 ms window every millisecond instead.
EDGE_STEP_S, EDGE_HOP_S = 0.004, 0.001
#: How far either side of a coarse mute edge the fine pass looks for the key: three
#: coarse steps, so an edge :func:`muted` placed a step out is still inside it.
EDGE_SPAN_S = 3 * STEP_S
#: A run this long has to agree before an edge is called. One 4 ms window of a fade
#: is not a key-up, and one window of the codec's own crosstalk is not the emission
#: -- the DAC's output couples straight back into the ADC on this interface, 28 ms
#: and 14 dB under the burst that follows it through the rig.
EDGE_HOLD_S = 0.010
#: The emission is present where it is within this of the level the same keying
#: settles at, which is read off that keying and so needs no calibration.
#:
#: It has to stand over the codec's crosstalk copy of the burst — 14 dB under, so
#: 0.20 of the body — and it may stand no higher, because everything above that is
#: the waveform's own crest. 0.71 did, and read the opening ramp of a multi-carrier
#: burst as no emission: a PACTOR-3 packet is 30 to 38 dB below its body for its
#: first symbols, so through a clean channel with a 40 ms lead it came back with
#: bit 0 at 75-85 ms and 780 ms of its 862. At 0.40 the same packets read 60 ms and
#: 813, and PACTOR-1 — which snaps on — reads 38 ms and its whole length at either.
#: Nothing further is had by going lower: the last 17 ms of a PACTOR-3 head lies
#: under every level a recording of it can resolve.
EDGE_AT_BODY = 0.40
#: An edge is only worth reading where the span's own level stands this far over
#: the tap's floor. A ratio against levels read off the same keying, so neither
#: needs calibrating.
EDGE_OVER_FLOOR = 4.0
#: How often `record` writes the queued blocks through. A crash loses this much.
DRAIN_S = 0.25
#: No sample delivered for this long and the card is gone rather than quiet: a
#: muted receiver still delivers silence, and only a dead stream delivers nothing.
STALL_S = 2.0


def _read(path: str) -> np.ndarray:
    from hfmodem.core import wav

    with wave.open(wav.source(path), "rb") as w:
        if w.getframerate() != FS:
            raise SystemExit(f"{path} is {w.getframerate()} Hz, not {FS}")
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            a = a[::w.getnchannels()]
    return a.astype(float) / 32768.0


def _steps(a: np.ndarray, fs: int, step_s: float, hop_s: float | None = None):
    """Successive `step_s` windows of `a`, `hop_s` apart. Abutted where hop is None."""
    n = max(1, int(round(step_s * fs)))
    hop = n if hop_s is None else max(1, int(round(hop_s * fs)))
    a = np.asarray(a, float)
    if len(a) < n:
        return np.zeros((0, n)), n, hop
    return np.lib.stride_tricks.sliding_window_view(a, n)[::hop], n, hop


def _spectra(a: np.ndarray, fs: int, step_s: float, hop_s: float | None = None):
    """``(per-step magnitude spectra, bin frequencies, window)``, empty if a is shorter."""
    frames, n, _ = _steps(a, fs, step_s, hop_s)
    if not frames.size:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
    win = np.hanning(n)
    return (np.abs(np.fft.rfft(frames * win, axis=1)),
            np.fft.rfftfreq(n, 1 / fs), win)


def out_of_band_level(a: np.ndarray, fs: int = FS, step_s: float = STEP_S) -> np.ndarray:
    """Per-step level of the receiver's own noise, as a fraction of what it reads
    with the receiver listening: 1 while it listens, and collapsing when it mutes.

    ONE BAND CANNOT DO THIS, and 2000-2800 Hz was the one that could not. It is
    inside what this station transmits — VARA BW2300 occupies 680-2297, ARDOP
    BW2000 500-2500, PACTOR-3 SL6 380-2620 — so a wide burst RAISES it, and on
    `20260815-0201-vara-n0lcr1-force.wav` the keyed steps came out 10.5 dB ABOVE
    the listening ones. The populations were the wrong way round, and 15 keyings
    of 2.04 s read as 31 of 0.24 to 1.64.

    So every band is measured against its own listening level and the MEDIAN taken
    across them. Half the tiling has to fall, which VARA BW2300 and everything
    narrower cannot arrange, and which one band's bad luck cannot either: a receiver
    whose own hiss dies above 2.9 kHz — as it does on `20260815-205842-cal033.wav`,
    where the tap's floor is louder than the receiver is up there — takes the top of
    the tiling with it and is outvoted.

    A quantile low enough to also survive a full-passband emission (ARDOP BW2000,
    PACTOR-3 SL5 and SL6, none of which this corpus holds) is not available at this
    price. The bands above 2.9 kHz are the only ones such an emission leaves clear,
    and they are the same bands a quiet receiver empties on its own, so every
    quantile that reaches them turns the pre-session minutes of
    `20260815-205842-cal033.wav` into a 12 to 16 s keying.

    Each band is measured against its own listening level TRACKED OVER TIME rather
    than one figure taken across the whole file — :func:`receiver_live_track`. One
    figure per recording is a claim that the receiver read the same at minute 55 as
    at minute 2, and the recordings this is pointed at are whole sessions: the band
    comes up at dusk, the operator moves AF gain, a loud station arrives and leaves.
    Any of those shifts the average, and a threshold hung off the average moves with
    it — in either direction, and the direction that matters is the one where a real
    keying stops reading as one.

    Mean of the bin magnitudes within each band rather than their RMS: RMS follows
    the loudest bin, and a railed monitor spilling into the bottom of a band pulls
    it back over the threshold for a step or two, which reads as the transmitter
    unkeying mid-over.

    `tools/ptt_tail_check.py` reads its keyed interval off this too, and that is why
    it is a function rather than four lines inside `measure`. It used to threshold
    TOTAL level instead, and on 2026-08-15 the two tools were pointed at the same
    100 s recording: this found eight keyings of about 1.9 s, the tail check found a
    single 0.28 s region and passed it. With the monitor at a working gain the
    transmit audio is the loudest thing in the recording, so the only stretch that
    still looks quiet is the unmodulated carrier between PTT and the first sample —
    a check for a tail that had, in effect, measured the lead.
    """
    s, freqs, win = _spectra(a, fs, step_s)
    if not s.size:
        return np.zeros(0)
    bands = []
    for lo, hi in zip(MUTE_BANDS, MUTE_BANDS[1:]):
        level = s[:, (freqs > lo) & (freqs < hi)].mean(axis=1) / (win.sum() / 2)
        bands.append(level / receiver_live_track(level, step_s))
    return np.median(bands, axis=0)


def muted(a: np.ndarray, fs: int = FS, step_s: float = STEP_S) -> np.ndarray:
    """Per-step: is the receiver muted, i.e. are we keyed?

    A keying is a stretch that the level is down for and deep inside somewhere,
    which is hysteresis written as a run rather than as a state machine.
    """
    hi = out_of_band_level(a, fs, step_s)
    deep = hi < MUTED
    m = np.zeros(len(hi), bool)
    for u, d in _runs(hi < LIVE_AGAIN):
        m[u:d] = deep[u:d].any()
    for u, d in _runs(~m):
        if 0 < u and d < len(m) and (d - u) * step_s <= MUTE_GAP_S:
            m[u:d] = True
    for u, d in _runs(m):
        if (d - u) * step_s < MIN_KEYING_S:
            m[u:d] = False
    return m


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edge = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edge[::2].tolist(), edge[1::2].tolist()))


def in_band_level(a: np.ndarray, fs: int = FS, step_s: float = STEP_S,
                  hop_s: float | None = None) -> np.ndarray:
    """Per-step RMS bin magnitude across :data:`BAND` — the level transmit audio moves.

    RMS where :func:`out_of_band_level` takes the mean, and for the opposite reason:
    that one wants a level one loud bin cannot lift, this one wants a level one loud
    bin is enough for. A step of PACTOR-1 holds one tone in fifty bins of band, and
    the mean divides it by the empty forty-nine — 1.2 dB under the threshold on the
    calibrated capture, where the RMS clears it by 12.3.
    """
    s, freqs, win = _spectra(a, fs, step_s, hop_s)
    if not s.size:
        return np.zeros(0)
    sel = (freqs > BAND[0]) & (freqs < BAND[1])
    return np.sqrt((s[:, sel] ** 2).mean(axis=1)) / (win.sum() / 2)


def transmit_audio(a: np.ndarray, fs: int = FS, step_s: float = STEP_S) -> np.ndarray:
    """Per-step: is the transmit passband carrying audio rather than a dead carrier?"""
    ib = in_band_level(a, fs, step_s)
    if not ib.size:
        return np.zeros(0, bool)
    return ib > receiver_live_level(ib) * AUDIO_RATIO


def monitor_level(a: np.ndarray, steps: np.ndarray) -> tuple[float, float, int]:
    """``(rms, peak, railed samples)`` of the audio in ``steps``."""
    n = int(round(STEP_S * FS))
    sel = np.repeat(np.asarray(steps, bool), n)[:len(a)]
    seg = a[:len(sel)][sel]
    if not seg.size:
        return 0.0, 0.0, 0
    return (float(np.sqrt(np.mean(seg ** 2))), float(np.abs(seg).max()),
            int(np.sum(np.abs(seg) >= 0.999)))


def receiver_live_level(hi: np.ndarray) -> float:
    """The out-of-band level with the receiver listening, taken across the whole
    of a tap that may be keyed for most of its length.

    This is the SEED that :func:`receiver_live_track` tracks from, and the level
    :func:`transmit_audio` still compares against. Tracking that one would be a
    mistake of a different shape: the in-band level RISES when the station keys
    and modulates, so a baseline built from the loud steps of a window would be
    built partly from the transmit audio itself, and every dB it gained would be a
    dB of margin taken off the test for whether there was any audio at all. That
    is the direction that reports a real emission as a quiet one.

    The 80th percentile of the WHOLE recording stood here, and it inverts the
    moment the keyings fill more than 80% of it: the percentile lands inside the
    mute, the mute's own level becomes the baseline, nothing then sits 16 dB
    below it, and a 30 s tap holding two 13 s keyings reads as no keying at all.
    A long transmission is exactly what this tool is pointed at, so the baseline
    cannot be a fixed quantile of a population whose composition is the thing
    being measured.

    The percentile was right; the population was wrong. The two populations are
    far apart — on `20260815-0133-ardop.wav` the live frames sit at -35 to -28 dB
    and the muted ones at -81 to -73 — so the live ones are found rather than
    assumed: two means over log level, seeded at the extremes, and the 80th
    percentile taken inside the upper cluster. Which side holds the majority then
    stops mattering, and on all fourteen of the 2026-08-15 recordings the figure
    moves at most 0.95 dB from what the old one read. A recording with no keying
    in it still has one population, which splits into halves a fraction of a
    decibel apart, so nothing falls the 16 dB below it that counts as a mute.
    """
    x = np.log10(np.maximum(np.asarray(hi, float), 1e-12))
    lo, up = float(x.min()), float(x.max())
    for _ in range(20):
        mid = (lo + up) / 2
        below, above = x[x <= mid], x[x > mid]
        if not below.size or not above.size:
            return float(10 ** np.percentile(x, 80))
        lo, up = float(below.mean()), float(above.mean())
    return float(10 ** np.percentile(x[x > (lo + up) / 2], 80))


def receiver_live_track(hi: np.ndarray, step_s: float = STEP_S,
                        window_s: float = LIVE_TRACK_S) -> np.ndarray:
    """The listening level per step: the same quantity as
    :func:`receiver_live_level`, followed over the recording instead of collapsed
    to one number for it.

    A mute is read as a fall BELOW the listening level, so that level is the whole
    test, and one figure for a session-long tap asserts a receiver that never
    moved. This station's move every night: the band comes up at dusk, a loud
    station arrives, the operator touches AF gain between overs. `20260815-164250`
    is an hour long and `20260815-0210` half of one, and every keying in them was
    judged against the same number.

    Estimated only from steps the receiver was audibly listening in, which is what
    the whole-file figure above is for: it splits the two populations, and a step
    at :data:`LIVE_AGAIN` of it or better is one of the live ones. A window with no
    live step in it contributes nothing and is spanned by the interpolation, which
    is what a long over is meant to look like here.

    AND IT NEVER READS BELOW THE WHOLE-FILE FIGURE. That is the asymmetry the tool
    is owed, and it is not a fudge: the way a window's estimate goes wrong is by
    being built from too few live steps in a densely keyed stretch, and those steps
    are the ones nearest the mute, so the error is always downward. A tracked level
    that came out below the whole-file one has been dragged there by the keyings it
    exists to find, and a keying compared against it stops reading as one. Floored,
    the change can only ever find MORE mute than the level taken across the file —
    which is what it does. Across the 25 recordings in the recorded transmitter-monitor trials — 2.65
    hours of tap, in which the level taken across each file finds 794 keyings — not
    one of those is lost or shortened, and 15 recordings gain 62 s of keyed time
    between them. **98% of the added steps carry
    transmit audio**, so what the whole-file level was doing was cutting real
    emission short: 40 s of it on `20260815-185350-ws8eoc.wav` alone, which read as
    34 s of carrier in 300 and was 74.

    The other direction is not fixed here and is not claimed to be. Where the level
    taken across the file sits ABOVE what the receiver was reading in some stretch,
    that stretch reads as a shallow mute and neighbouring keyings run together
    through it. The floor keeps that behaviour exactly as it was.
    """
    hi = np.asarray(hi, float)
    seed = receiver_live_level(hi)
    live = hi >= seed * LIVE_AGAIN
    n = max(1, int(round(window_s / step_s)))
    at, level = [], []
    for u in range(0, len(hi), n):
        window = hi[u:u + n][live[u:u + n]]
        if window.size:
            at.append(u + n / 2)
            level.append(float(np.percentile(window, 80)))
    if not level:
        return np.full(len(hi), seed)
    return np.maximum(np.interp(np.arange(len(hi)), at, level), seed)


def monitor_headroom(a: np.ndarray, audio: np.ndarray, keyed: np.ndarray) -> tuple[float, float, int]:
    """``(dB below the rail, dB above the tap's own noise, railed samples)``.

    Both figures come out of the recording being measured, which is the point.
    :data:`MONITOR_NOMINAL` used to stand here — a level read off one 2026-08-14
    capture by the detector this file has since replaced, so the tool was checking
    its input against a number its own broken output had produced. The tap's noise
    is not assumed either: the unmodulated carrier between PTT and the first sample
    is the tap with nothing at all on it, and every keying carries one.

    A quartile of those steps rather than all of them, because the step at each edge
    of a keying straddles it and holds live receiver audio as well as the carrier,
    and an RMS over the lot follows those: 8.2 dB on the calibrated capture where
    the carrier itself sits 23.1 below the audio.
    """
    n = int(round(STEP_S * FS))
    step = np.asarray(a, float)[:len(a) // n * n].reshape(-1, n)
    dead = np.sqrt((step ** 2).mean(axis=1))[(keyed & ~audio)[:len(step)]]
    rms, peak, railed = monitor_level(a, audio)
    floor = float(np.percentile(dead, 25)) if dead.size else 0.0
    return (-20 * np.log10(peak) if peak else 0.0,
            20 * np.log10(rms / floor) if floor and rms else 0.0, railed)


def key_edge(a: np.ndarray, at: int, fs: int = FS, down: bool = False) -> int | None:
    """The sample the receiver muted on, refined off a coarse :func:`muted` edge.

    :func:`out_of_band_level` reads the mute across seven bands and cannot be asked
    for a finer grain — at 4 ms a band of that tiling holds one FFT bin — so the
    coarse pass decides THAT there is a keying and this one sharpens the instant.
    Total level is enough over this span and nowhere else: at the key the receiver
    is on its way down or back up and the transmit audio is not there yet, which is
    the very thing the span is being measured to find out.
    """
    lo = max(0, at - int(round(EDGE_SPAN_S * fs)))
    frames, n, hop = _steps(a[lo:at + int(round(EDGE_SPAN_S * fs))],
                            fs, EDGE_STEP_S, EDGE_HOP_S)
    if len(frames) < 2:
        return None
    lvl = np.sqrt((frames ** 2).mean(axis=1))
    live, floor = float(np.percentile(lvl, 90)), float(lvl.min())
    if floor <= 0.0 or live < floor * EDGE_OVER_FLOOR:
        return None
    quiet = lvl < np.sqrt(live * floor)
    k = _held(quiet[::-1] if down else quiet)
    if k is None:
        return None
    k = len(quiet) - k if down else k
    return lo + k * hop + n // 2


def emission(a: np.ndarray, up: int, end: int, fs: int = FS) -> tuple[int, int] | None:
    """``(first sample, last sample)`` of the transmit audio inside one keying.

    Read against the level that keying settles at rather than against a level of
    its own, so what comes back is where the emission is and not a claim about how
    loud it should have been.
    """
    ib = in_band_level(a[up:end], fs, EDGE_STEP_S, EDGE_HOP_S)
    k = max(1, int(round(EDGE_HOLD_S / EDGE_HOP_S)))
    if len(ib) < 2 * k:
        return None
    on = ib >= float(np.percentile(ib, 80)) * EDGE_AT_BODY
    first, last = _held(on), _held(on[::-1])
    if first is None or last is None:
        return None
    off = up + int(round(EDGE_STEP_S * fs)) // 2
    hop = max(1, int(round(EDGE_HOP_S * fs)))
    return off + first * hop, off + (len(on) - last) * hop


def _held(mask: np.ndarray) -> int | None:
    """The index a run of :data:`EDGE_HOLD_S` first starts at, or None."""
    k = max(1, int(round(EDGE_HOLD_S / EDGE_HOP_S)))
    if len(mask) < k:
        return None
    at = np.flatnonzero(np.convolve(mask, np.ones(k), "valid") == k)
    return int(at[0]) if at.size else None


def edges(path: str) -> int:
    """`measure`'s anatomy at one millisecond instead of twenty.

    Same detectors; what changes is that the instants come back resolved to a
    fifth of a PACTOR-1 bit rather than to two of them.

    KEY-UP IS THE MUTE ENGAGING and is honest to a millisecond or two: on the
    0819 sessions it lands within 2 ms of the PTT lead those runs logged. THE
    OTHER EDGE IS NOT A KEY-DOWN. Receive audio comes back tens of milliseconds
    after the key drops — 33 ms on this rig, and `shrike.onair.TR_SWITCH_S` calls
    it 55 — so a tail measured off it is that recovery and not dead carrier, and
    it is reported as the recovery it is. `measure` reads a tail off the same
    edge and can afford to: the tails it was written for run to 1.12 s.

    IN BETWEEN, THIS PRINTS TWO INSTANTS AND REFUSES THEIR DIFFERENCE. Both are
    facts about the recording — where the in-band level first reaches the body of
    this keying and where it last does, each against key-up. The difference is not
    a fact about the transmission, and until 2026-08-25 it was printed as one.

    The far instant is the sound one. Across all 20 sessions of 2026-08-19/20 the
    end of the audio sits 986.5 to 990.0 ms after key-up for a 960 ms burst — 3.5
    ms of spread over three bands and two days — so THAT is the figure to set
    against the keyed span the session logged. The near instant is not: it swings
    36.0 to 51.0 ms over the same 20, and the swing is this detector rather than
    the rig. `tests/shrike/test_txhead.py` correlates those bursts against the
    render that made each one, which needs no level threshold, and finds the head
    a flat 20.8 to 23.9 ms behind the DAC on all 20 and exactly two bits missing
    off every tail.

    WHAT MOVES THE NEAR INSTANT is not the burst, and it is measured twice. The
    render correlates against the keying ONCE, at 0.97, 46 to 51 ms after key-up
    on all 20 — including the five this reads at 36 — so there is no early copy of
    the burst for the edge to have found. What is there instead is low frequency:
    the 10 ms at +30 ms carries 0.39 to 0.40 of full scale between 300 and 1000 Hz
    against 0.43 to 0.74 in the 1250-1750 Hz tone pair on those five, where the
    burst body carries 0.03 to 0.07 against 1.25. PACTOR-1 puts nothing below 1400
    Hz. :func:`in_band_level` takes its RMS across the whole 300-2800 Hz passband,
    which is what it must do to see OFDM at all, so that transient alone clears
    :data:`EDGE_AT_BODY` of the body and the edge lands on it. No threshold read
    off the same keying rejects it, which is why this reports the instant and not
    a length.

    An emission length needs a second clock. `tools/kiwi_witness.py` records one,
    and a KiwiSDR reads these same bursts at 959.40 +/- 2.09 ms.

    A tap on the rig's speaker output reads a real T/R here. So does a shrike
    session's own `stream.wav`: it is written from the capture callback ahead of
    the half-duplex discard, so the receiver's mute, our own emission back through
    the rig, and the peer's answer are all in it on one sample clock. Neither is a
    second clock: both come through this rig's own receive path, so both carry the
    flush and the thump, and the refusal above is for both.
    """
    a = _read(path)
    keyed = muted(a)
    n = int(round(STEP_S * FS))
    keyings = _runs(keyed)
    print(f"  {path}: {len(a) / FS:.1f} s, {len(keyings)} keyings")
    rows = []
    for i, (u, d) in enumerate(keyings, 1):
        up, back = key_edge(a, u * n), key_edge(a, min(d * n, len(a) - 1), down=True)
        em = None if up is None or back is None else emission(a, up, back)
        if em is None:
            why = ("the recording does not hold both edges of this keying"
                   if up is None or back is None else "no emission inside the keying")
            print(f"  key {i:2d}: {u * n / FS:7.3f} s — nothing to read: {why}")
            continue
        rows.append((1e3 * (em[0] - up) / FS, 1e3 * (em[1] - up) / FS,
                     1e3 * (back - em[1]) / FS))
        print(f"  key {i:2d}: key-up {up / FS:7.3f} s   body from {rows[-1][0]:+6.1f} ms   "
              f"to {rows[-1][1]:+7.1f} ms   receiver back {rows[-1][2]:+6.1f} ms after it")
    if not rows:
        print("  nothing measurable — either nothing transmitted, or the tap is "
              "not on the rig's speaker output.")
        return 1
    r = np.array(rows)
    print(f"  median of {len(r)}: body from {np.median(r[:, 0]):.1f} ms after key-up "
          f"({np.median(r[:, 0]) / (1e3 * BIT_S):.1f} bits at 100 Bd) "
          f"to {np.median(r[:, 1]):.1f} ms after it, "
          f"receiver back {np.median(r[:, 2]):.1f} ms after that")
    print(f"  no emission length: set {np.median(r[:, 1]):.1f} ms against the keyed "
          "span the session logged. Their difference is not a reading — this "
          "recording came through our own receiver.")
    return 0


def measure(path: str) -> int:
    a = _read(path)
    keyed = muted(a)
    t = np.arange(len(keyed)) * int(round(STEP_S * FS)) / FS
    # `& keyed`: a keying is bounded by the edges of the mute, so the step either side
    # of it is the live receiver, and in-band level cannot tell that from a monitor.
    audio = transmit_audio(a) & keyed
    keyings = _runs(keyed)
    print(f"  {path}: {len(a) / FS:.1f} s, {len(keyings)} keyings")
    if not keyings:
        print("  no keying found — the receiver never muted. Either nothing "
              "transmitted, or the tap is not on the rig's speaker output.")
        return 1
    ends_keyed = keyings[-1][1] == len(keyed)
    if ends_keyed:
        # Dropped from the anatomy below and still reported: a tap that stops
        # inside a keying has one unterminated keying and every complete one
        # before it, and a session's own recording ends that way as a matter of
        # course. Returning here made the whole file unreadable for the sake of
        # its last burst.
        print(f"  PTT up {t[keyings[-1][0]]:6.2f} s and never seen to drop — "
              "*** the recording ends still keyed ***")
        keyings = keyings[:-1]
    if not keyings:
        return 1
    head, over, railed = monitor_headroom(a, audio, keyed)
    rms, peak, _ = monitor_level(a, audio if audio.any() else keyed)
    in_range = (audio.any() and not railed and head >= MONITOR_HEADROOM_DB
                and over >= MONITOR_OVER_FLOOR_DB)
    print(f"  monitor rms {rms:.4f}  peak {peak:.3f}  railed {railed}  "
          f"headroom {head:.1f} dB  over the tap's noise {over:.1f} dB"
          f"{'' if audio.any() else '  — NO AUDIO IN ANY KEYING, this is the keyed level'}")
    if not in_range:
        print(f"  *** MONITOR UNUSABLE — it has to sit {MONITOR_HEADROOM_DB:.0f} dB "
              f"clear of the rail with nothing on it, and {MONITOR_OVER_FLOOR_DB:.0f} dB "
              "over the tap's own noise "
              "— EVERY AUDIO, LEAD AND TAIL FIGURE BELOW IS SUSPECT ***")
    n = 0
    for u, d in keyings:
        on = np.flatnonzero(audio[u:d])
        if not on.size:
            print(f"  PTT up {t[u]:6.2f}-{t[d]:6.2f} s ({t[d] - t[u]:.2f} s) "
                  "with NO audio detected — a keyed carrier and nothing on it")
            continue
        a0, a1 = t[u + on[0]], t[u + on[-1]] + STEP_S
        seg = a[int(a0 * FS):int(a1 * FS)]
        n += 1
        print(f"  key {n}: PTT {t[u]:6.2f}-{t[d]:6.2f} ({t[d] - t[u]:5.2f} s)  "
              f"audio {a0:6.2f}-{a1:6.2f} ({a1 - a0:.2f} s)  "
              f"LEAD {1000 * (a0 - t[u]):5.0f} ms  TAIL {1000 * (t[d] - a1):5.0f} ms")
        print(f"          monitor rms {np.sqrt(np.mean(seg ** 2)):.4f}  "
              f"peak {np.abs(seg).max():.4f}  "
              f"railed {int(np.sum(np.abs(seg) >= 0.999))}  "
              f"dead carrier {100 * (1 - (a1 - a0) / (t[d] - t[u])):.0f}% of the keying")
    return 0 if in_range and not ends_keyed else 1


from hfmodem.core.capture_producer import restore_signals, owned_record


@restore_signals
@owned_record
def record(seconds: float, path: str, device: str, *, ownership=None) -> int:
    # `measure` needs only numpy, so `./tools/txwitness.py measure` runs under the
    # shebang's python and `record` does not — it wants `sounddevice` and this
    # tree, which live in the venv. That asymmetry cost a slot its first witness
    # on 2026-08-15, because the traceback said `No module named 'hfmodem'` and
    # not what to run instead.
    try:
        import sounddevice as sd

        from hfmodem.core import wav
        from hfmodem.core.devices import find_device
        from hfmodem.core.rig import FATAL_SIGNALS
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"{exc.name!r} is not on this interpreter's path. `record` needs the "
            "project venv where `measure` does not:\n"
            f"    ./.venv/bin/python tools/txwitness.py record {seconds:g} {path}"
        ) from exc

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in FATAL_SIGNALS:
        signal.signal(sig, stop)

    dev = find_device(device, "in")
    name = sd.query_devices()[dev]["name"]
    print(f"  recording {seconds:g} s from {name} — {path} is readable while it runs")
    tap = wav.CaptureRecorder(path, rate=FS, source=name, **({"ownership": ownership} if ownership else {}))
    lost = 0

    def heard(block, _frames, _time, status):
        nonlocal lost
        lost += bool(status.input_overflow)
        tap.push(block[:, 0].copy())

    stalled = False
    try:
        with sd.InputStream(samplerate=FS, channels=1, device=dev, callback=heard):
            last = time.monotonic()
            while tap.seconds < seconds:
                time.sleep(DRAIN_S)
                before = tap.seconds
                tap.drain()
                if tap.seconds > before:
                    last = time.monotonic()
                elif time.monotonic() - last > STALL_S:
                    stalled = True
                    break
    finally:
        tap.facts["xruns"] = lost
        tap.close()
        print(f"  saved {path}: {tap.seconds:.1f} s"
              + (f", {lost} overflows" if lost else ""))
    if stalled:
        print(f"  *** the card delivered nothing for {STALL_S:g} s and the tap "
              f"stopped there — the interface went away ***")
    return 1 if stalled else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("seconds", type=float,
                   help="SIGTERM stops the tap early and keeps what it has")
    r.add_argument("path")
    r.add_argument("--device", default=None)
    from hfmodem.core import capture_producer
    capture_producer.arguments(r)
    for name in ("measure", "edges"):
        sub.add_parser(name).add_argument("path")
    args = p.parse_args()
    if args.cmd == "record":
        ownership = capture_producer.prepare(args.path, capture_id=args.capture_id,
            capture_root=args.capture_root, capture_role=args.capture_role, roles=("tap",))
        from hfmodem.core.rigs import resolve_rig
        if args.device is None and resolve_rig() != "ft891":
            p.error("--device must explicitly name the independent witness input")
        args.device = args.device or "KT USB"
        return record(args.seconds, args.path, args.device, **({"ownership": ownership} if ownership else {}))
    return (edges if args.cmd == "edges" else measure)(args.path)


if __name__ == "__main__":
    sys.exit(main())
