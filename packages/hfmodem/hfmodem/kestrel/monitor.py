# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Receive-only VARA traffic reading: the burst gate, the classifier that names
what it brackets, and the handshake scanner that reads connect requests and
responses off the raw stream.

This is the streaming half of ``tools/vara_monitor.py``, which is the command
line around it. It lives in the package because the creance monitor's kestrel
runner imports it, and a runner in an installed distribution has nothing else to
import it from.
"""

from __future__ import annotations

import csv
import io
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve, find_peaks, firwin

from hfmodem.kestrel.rx import varahf2300 as RX
from hfmodem.kestrel.vara import vara_control as VC
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import _ACK_PLATEAU, _ack_plateau, _tone_track

FS = 48000
ACCEPT = 0.8                      # gateway match acceptance (matched/total)

# Streaming segmenter (all fixed — the class is only ever built with defaults).
SEG_FRAME = 1024                  # energy-gate analysis frame
SEG_HANG = 6                      # trailing quiet frames that close a burst
SEG_PAD = 4                       # idle frames of pre-roll kept before a burst
SEG_LO_S = 0.30                   # min burst length handed to find_bursts
SEG_MAX_S = 12.0                  # force-close a burst that never falls silent

# The noise floor: the SEG_FLOOR_Q quantile of the last SEG_FLOOR_S of frame RMS.
# The quantile has to sit above the fraction of time the input spends in a
# dropout, or the dropouts become the floor; measured over the three off-air
# sessions the receiver is muted for 11.5%, 19.7% and 20.6% of the recording
# (frames >15 dB below the median), so a quarter clears all three. Any window
# that does exceed it corrects itself within SEG_FLOOR_S — which is the whole
# point, and what a running minimum could never do. 30 s rather than 10 s
# because a shorter window is more often filled by traffic alone: 10 s finds 11
# bursts in NS0A_2300 against 13, and 10 against 11 in KC9GHZ_2300.
SEG_FLOOR_S = 30.0
SEG_FLOOR_Q = 0.25

# A frame this far below the median of the live history is not band audio: the
# receiver is muted, deaf, or dropping out, and what it delivers is a monitor bleed
# or the codec's own noise. Only :class:`MonitorGate` applies it.
#
# 15 dB, from the corpus. On the 2026-07-26 recording this rig's own transmissions
# come back through its muted receiver 20.1 dB below the band; the deep mute either
# side of them is 63 dB down. Across fourteen recordings made with no local
# transmitter keying — the verified clear channel, the shared regression corpus's
# ARDOP, PACTOR-1/2/3, FT8, WSPR, narrowband and band-noise fixtures — the deepest
# any frame falls below its own running median is 13.6 dB. The one exception is
# 20 s of a very strong local FT8 station, where the median is the signal and the
# gaps between its overs are the band: 8.6% of frames read dead there, which costs
# a slightly higher floor and never a wrong line.
SEG_MUTE_DROP_DB = 15.0
SEG_MUTE_RATIO = 10 ** (-SEG_MUTE_DROP_DB / 20)
SEG_LIVE_MIN = 48                 # frames of live history before the mute test arms

# One number for every rig is a guess, and on this one it is the wrong guess. The
# rule above wants a mute 15 dB deep; the mute this station's receiver delivers under
# its own keying is 12.0 to 13.4 dB deep on the 2026-09-04 recordings, because the
# muted receiver still leaks codec crosstalk. So 88-94% of our own transmit frames
# read as band audio, the floor settled on the mute rather than on the band, and the
# band RETURNING at unmute stood 13 dB over that floor and held a bracket open until
# the next key-down: the eight "ten-second wideband overs" on
# 20260904T195626Z-W9SSJ-W0LON are the gaps between our own transmissions, and the
# quietest ten seconds of the tape is among them. The same shallowness silences the
# gate the other way on a quiet channel — five of the six tapes of that afternoon
# bracket nothing at all, our keyings arriving as 11-12 dB dips the floor swallows.
#
# So the depth is measured, per stream, off the stream. Frame level in dB is bimodal
# wherever a local transmitter keys — the band, and the receiver under it — and
# two-medians separates the two in a handful of passes. Medians rather than means
# because the PTT transition frames sit 45-60 dB down and would drag a mean onto
# them. Over the whole of each tape: 13.4 and 13.1 dB (the two W0LON tapes of
# 2026-09-04), 12.0 dB (W6IDS, the empty channel of the same afternoon), 18.9 dB
# (VE3KPG, 2026-09-03). What the monitor prints is the first window's, which is the
# moment the gate starts acting on it: 12.4, 14.4, 12.0 and 18.7 dB.
MUTE_WINDOW_S = 60.0              # frames the split is measured over
MUTE_ARM_S = 20.0                 # ...and the least of it that will answer
MUTE_SEED_DB = 8.0                # where the low cluster is first looked for
MUTE_ITERS = 8
MUTE_MIN_DEPTH_DB = 10.0
MUTE_FRAC = (0.02, 0.45)          # what fraction of the window a mute may hold

# What separates a transmitter from a fade, and the guard that keeps a strong
# station's gaps between overs from being read as a mute: a mute REACHES THE CODEC.
# Nothing a receiver HEARS falls 40 dB below its own median — over the fourteen
# recordings made with no local keying the deepest frame is 13.6 dB down, and the
# worst case on record is 20 s of a very strong local FT8 station whose median is
# the signal, 25 dB over the band in its gaps. The PTT transition takes the
# receiver to nothing at all: 55 to 250 frames a tape, and every 60 s window of the
# four tapes above that holds a keying holds 7 to 83 of them.
#
# A handful is all it takes, so this counts frames rather than reading a percentile
# of the mute cluster — a percentile is a description of the whole keying and does
# not reach the transition until several have gone by, which on 20260904T195626Z
# left the split unarmed until 38 s and three gap brackets already open.
MUTE_DEEP_DB = 40.0
MUTE_DEEP_FRAMES = 4

# Enter and exit, as multiples of that floor. The gate's sensitivity is just the
# enter threshold: a synthetic connected-ack in white noise is bracketed whole
# 5/5 from 12 dB SNR at 4.0x, and only from 16 dB at the 6.0x this replaces.
# 30 s of verified clear channel and 54 s of quiet receive window between our own
# overs give their first false open at 2.0x, so 4.0x keeps 6 dB of margin over
# measured band noise. Exit is half of enter, i.e. 6 dB of hysteresis.
#
# A consumer that demodulates every burst and throws away what does not match can
# afford to sit right down on that knee, and kestrel_connect does — see
# _BracketSegmenter. A traffic log cannot: every false open there is a wrong line,
# and it reads the floor differently as well — see MonitorGate.
SEG_ENTER = 4.0
SEG_EXIT = 2.0

# The second way into a burst, and the one that exists because the first missed two
# confirmed VARA sessions in a single slot.
#
# The enter test above is a PEAK: one 21 ms frame's broadband RMS against the floor.
# That is the right instrument for a keyed MFSK burst, which puts all its power in
# one tone and stands ~12 dB proud, and the wrong one for a wideband over on a noisy
# band, which spreads the same power across the passband. On 2026-08-14 the gate
# handed `classify` ZERO bursts across 815.6 s of `rx-20260814-123414-7103500`, a
# 25-minute VARA session between W9SEM-10 and KD9OSV that the Winlink feed names and
# whose spectrum is on channel: the loudest frame in either segment stood +5.98 dB
# over the floor against the +7.96 dB that 2.5x asks for. The same slot's confirmed
# VARA 500 exchange on 7096.5 kHz, both callsigns attested by two independent
# sources, peaked at +4.99 dB and was missed the same way. The corpus already held a
# third instance — see `ONAIR_GATEWAY_GREETING`, a CRC-clean BW2300 over "4.1 dB over
# the band noise" that no energy gate opens on.
#
# Lowering the peak threshold cannot fix it: on 40 m at midday a static crash reaches
# as far as the over does. What separates them is that an over is HELD — this is the
# same distinction `core.busy._BURST_HOLD_S` is built on, measured there as 20-80 ms
# for the crashes against seconds for a keying. So the second test is the MEDIAN of
# the last MONITOR_HOLD_S of in-band level, and a crash cannot move a median.
#
# In-band rather than broadband because the SSB skirt and everything below 700 Hz can
# hold no VARA carrier, and excluding them widens the gap this threshold has to sit
# in from 1.15 dB to 1.55 dB. Measured over every full capture held, as the streaming
# gate computes it — causal, no padding, floor from live frames only:
#
#     QUIET, must not open                          held in-band, dB over floor
#       6800.0 kHz control (two steady carriers)          +1.05
#       clear_7101800, 30 s verified clear                +0.83
#       7108.5 kHz, listened 45 s and silent              +1.51
#       7103.5 window C, feed and deep pass agree empty   +3.74   <- worst
#     OCCUPIED, must open
#       7102.7 window A, feed: a VARA 2750 exchange       +5.29   <- weakest
#       7096.5 VARA 500, both callsigns named             +5.31
#       7103.5 W9SEM-10 <- KD9OSV, segments 0 and 1       +6.41, +7.40
#       14103.2 kHz, 20 m                                +14.82
#       kc9ghz greeting, the 4.1 dB over                 +25.67
#       NS0A_2300 / KC9GHZ_2300 sessions                 +36.29, +36.47
#
# 1.68x is +4.50 dB, which sits 0.76 dB above the loudest quiet capture and 0.79 dB
# below the weakest occupied one. That is the whole budget and it is what there is;
# `core.busy` ships `shape` on about 1 dB of the same.
#
# This is ADDITIVE. The peak test still runs and still opens first where it can, so
# no bracket the gate found before is lost — only bursts it never saw are added.
MONITOR_HELD_ENTER = 1.68
MONITOR_HOLD_S = 0.5
MONITOR_BAND = (700.0, 2700.0)
# The held test needs a floor before it can mean anything, and a quantile over a
# quarter of its 30 s window is where the calibration above was measured. The peak
# test is unaffected and still arms on the first frame.
MONITOR_HELD_MIN_S = SEG_FLOOR_S / 4
# Pre-roll. A median over MONITOR_HOLD_S crosses about half a hold after the lift
# begins, so SEG_PAD's 4 frames (85 ms) would hand `find_bursts` a region that starts
# inside the over. One second covers the hold twice over, and costs nothing: the
# region is re-cropped on its own envelope.
MONITOR_PAD = int(round(1.0 * FS / SEG_FRAME))


def vara_gateways(export: str) -> list[tuple[str, str]]:
    """``(callsign, grid)`` for every HF VARA row of a Winlink GatewayChannels
    export, in its own order. VARA FM is excluded: it is a different waveform."""
    rows = []
    for row in csv.DictReader(io.StringIO(export)):
        mode = (row.get("Mode") or "").strip()
        call = (row.get("Callsign") or "").strip().upper()
        if mode.startswith("VARA") and not mode.startswith("VARA FM") and call:
            rows.append((call, (row.get("GridSquare") or "").strip()))
    return rows


# Off air a fade drops one transmission below the energy gate for a few tens of
# milliseconds, and the gate then reports the pieces as separate bursts — one
# gateway over came back as four. Pieces closer than GAP_S are rejoined.
#
# The ceiling is how close two DIFFERENT transmissions ever get: 0.90 s in the
# staged BW500 control captures, and 1.1 s already welds two consecutive off-air
# DATA overs into one segment (losing the second). Below that, 178 of the 242
# inter-burst gaps measured across three off-air gateway sessions fall under 0.5 s
# — the fragmentation mass — so that is where this sits.
GAP_S = 0.5


def find_bursts(a: np.ndarray, lo_s=0.15, thr=0.12,
                gap_s=GAP_S) -> list[tuple[int, int]]:
    win = FS // 50
    # Boxcar mean by cumulative sums: O(N) instead of the O(N*win) direct-form
    # convolution, which dominated burst detection on long captures.
    p = np.empty(len(a) + 1)
    p[0] = 0.0
    np.cumsum(a * a, dtype=float, out=p[1:])
    lo = np.arange(len(a)) - win // 2      # np.convolve 'same' alignment
    hi = lo + win
    np.clip(lo, 0, len(a), out=lo)
    np.clip(hi, 0, len(a), out=hi)
    env = np.sqrt((p[hi] - p[lo]) / win)
    on = env > env.max() * thr
    edges = np.diff(on.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        stops.append(len(a))
    # Length filter first: the quiet between two overs is peppered with sub-lo_s
    # noise specks, and coalescing through those would weld the overs together.
    out = []
    for s, e in zip(starts, stops):
        if (e - s) / FS < lo_s:
            continue
        if out and s - out[-1][1] <= gap_s * FS:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


class MuteFloor:
    """The two levels a receiver delivers while a local transmitter keys — the band,
    and the mute under our own carrier — told apart by two-medians over the last
    :data:`MUTE_WINDOW_S`. See it for what the split is worth and what it is
    measured on.

    Fed every frame, bracketed ones included, because this describes the INPUT and
    not the channel: a gate that has swallowed the band into a ten-second bracket
    still has to be able to find the mute, and a window fed only from outside the
    brackets is exactly the feedback that put the floor on the mute in the first
    place.

    It answers only where the split is real — see :data:`MUTE_DEEP_DB` — so a
    recording with no local transmitter on it gets ``None`` and the older rule.
    """

    def __init__(self) -> None:
        n = int(MUTE_WINDOW_S * FS / SEG_FRAME)
        self.db: deque[float] = deque(maxlen=n)
        self.idb: deque[float] = deque(maxlen=n)
        self.threshold: float | None = None    # dBFS; under it the receiver is deaf
        self.depth: float | None = None        # band - mute, dB
        self.floor: float | None = None        # dBFS; the band, as the gate reads one
        self.iband_floor: float | None = None  # ...and the same in MONITOR_BAND
        self._due = 0

    def push(self, rms: float, iband: float) -> None:
        self.db.append(20.0 * float(np.log10(rms)))
        self.idb.append(20.0 * float(np.log10(iband)))
        self._due -= 1
        if self._due <= 0 and len(self.db) * SEG_FRAME >= MUTE_ARM_S * FS:
            self._due = int(FS / SEG_FRAME)             # re-measure once a second
            split = self._split()
            (self.threshold, self.depth,
             self.floor, self.iband_floor) = split or (None, None, None, None)

    def _split(self) -> tuple[float, float, float, float] | None:
        d = np.fromiter(self.db, float, len(self.db))
        band = float(np.median(d))
        low = d[d < band - MUTE_SEED_DB]
        if low.size < MUTE_FRAC[0] * d.size:
            return None
        mute = float(np.median(low))
        for _ in range(MUTE_ITERS):
            lo, hi = d[d < (cut := 0.5 * (band + mute))], d[d >= cut]
            if not lo.size or not hi.size:
                return None
            band, mute = float(np.median(hi)), float(np.median(lo))
        heard = d >= (cut := 0.5 * (band + mute))
        n_mute = int((~heard).sum())
        if not MUTE_FRAC[0] * d.size <= n_mute <= MUTE_FRAC[1] * d.size:
            return None
        if band - mute < MUTE_MIN_DEPTH_DB:
            return None
        if int((d < band - MUTE_DEEP_DB).sum()) < MUTE_DEEP_FRAMES:
            return None
        # Both floors are the band side's own SEG_FLOOR_Q, so that what comes out is
        # in the gate's currency — broadband for the peak test, MONITOR_BAND for the
        # held one. Traffic is in them, which the gate's `nf` and `ihist` are careful
        # to keep out; but a quarter-quantile of a minute does not notice a burst,
        # and these have to answer for a window a bracket may be holding open, which
        # is the one question those two cannot be asked.
        i = np.fromiter(self.idb, float, len(self.idb))
        return (cut, band - mute,
                float(np.quantile(d[heard], SEG_FLOOR_Q)),
                float(np.quantile(i[heard], SEG_FLOOR_Q)))

    def muted(self, rms: float) -> bool:
        """Is this frame the mute rather than the band?

        Against the split's own decision boundary, which scales with the depth it
        measured: 6.0 dB over the mute on the shallowest of the four tapes and
        9.4 dB on the deepest, and on every one of them above the mute cluster's
        95th percentile and below the band cluster's 1st. A fixed margin cannot do
        both ends of that range.
        """
        return (self.threshold is not None
                and 20.0 * float(np.log10(rms)) <= self.threshold)


# --------------------------------------------------------------------------- #
# Streaming burst segmenter. An adaptive-floor energy gate brackets each keyed
# region (with a little silence padding on both ends); the precise boundaries
# come from `find_bursts` run on that region — the same
# proven segmenter the offline tools use, so the boundaries the decoders see are
# identical whether the source is a WAV or a live device. The decoders are
# start-alignment sensitive, so getting this exactly right matters.
class Segmenter:
    """Bracket a live stream into keyed regions, then hand each to find_bursts
    for exact boundaries (it thresholds on a global max and can't run
    incrementally). An energy gate opens above ``enter`` × the noise floor and
    closes on SEG_HANG quiet frames below ``exit`` × it — or after ``max_s``, so a
    continuous carrier can't grow the buffer without bound.

    The floor is the SEG_FLOOR_Q quantile of the last SEG_FLOOR_S of frame RMS
    *measured between bursts*. It used to be a running minimum, which can only
    ever go down. One dropout sets it for good — and every off-air session here
    holds them, 1.4 to 5 s of the receiver muted while a transmitter keys: on
    offair/NS0A_2300 the running minimum ends at -78.8 dBFS against a band noise
    of -43.2, 35 dB below the channel it is supposed to describe. Nothing is ever
    quiet enough to close the gate after that, so every bracket runs to
    SEG_MAX_S; measured, 7 of 10. A quantile comes back up, within SEG_FLOOR_S of
    whatever knocked it down.

    The between-bursts part is the other half, and it is what makes the floor a
    description of the *band* rather than of the traffic on it. A window fed from
    every frame is fed from the bursts too, so each burst raises the threshold the
    next one has to clear: on 44 s of real VARA session traffic
    (rf-corpus pos_vara_session, whose median frame sits 13 dB over its band) the
    floor walked from 0.054 to 0.444 — 18.3 dB — and two bursts came out of the
    file. Off that same recording with burst frames withheld the floor holds 0.054
    to 0.068 and fourteen do. Nothing a station transmits can raise the bar for the
    next station, because nothing a station transmits is in the window.

    The exception is a signal that never stops being one. A burst force-closed at
    ``max_s`` fell silent nowhere in ``max_s`` seconds, which makes it channel
    rather than keying, so its frames do go into the window — see :meth:`_close`.

    A quantile is still diluted by a dropout rather than immune to it, and the
    traffic log cannot afford that — see :class:`MonitorGate`, which is the gate
    this module's own log runs."""

    enter = SEG_ENTER          # subclassed where the cost of a false open differs
    exit = SEG_EXIT
    mute_ratio = 0.0           # 0 -> every frame is band audio; see MonitorGate
    pad = SEG_PAD              # idle frames of pre-roll; see MonitorGate
    held = 0.0                 # 0 -> peak test only; see MONITOR_HELD_ENTER
    hold_s = MONITOR_HOLD_S    # the window the held median spans
    ref_s = 0.0                # idle audio kept as a spectral reference; see MonitorGate

    def __init__(self, max_s: float = SEG_MAX_S):
        self.max_s = max_s              # force-close; a live handshake wants it shorter
        self.buf = np.zeros(0)
        self.t0 = 0                     # global sample index of buf[0]
        self.pos = 0                    # frame cursor within buf
        self.hist: deque[float] = deque(maxlen=int(SEG_FLOOR_S * FS / SEG_FRAME))
        self.nf: float | None = None    # noise-floor RMS (frozen during a burst)
        self.inb = False
        self.acc: list[np.ndarray] = []
        self.start = 0
        self.quiet = 0
        self.ring: list[np.ndarray] = []    # recent idle frames, for pre-roll
        self.brms: list[float] = []         # frame RMS inside the open burst
        self.ihist: deque[float] = deque(maxlen=int(SEG_FLOOR_S * FS / SEG_FRAME))
        self.iheld: deque[float] = deque(
            maxlen=max(1, int(round(self.hold_s * FS / SEG_FRAME))))
        self.ref: deque[np.ndarray] = deque(maxlen=int(self.ref_s * FS / SEG_FRAME))
        self.mute = MuteFloor() if self.mute_ratio else None
        self.knows_mute = False
        f = np.fft.rfftfreq(SEG_FRAME, 1 / FS)
        self._iband = (f > MONITOR_BAND[0]) & (f < MONITOR_BAND[1])
        self._iwin = np.hanning(SEG_FRAME)

    def _crop(self, chunk: np.ndarray, base: int) -> list[tuple[int, np.ndarray]]:
        return [(base + s, chunk[s:e])
                for s, e in find_bursts(chunk, lo_s=SEG_LO_S)]

    def _in_band(self, fr: np.ndarray) -> np.ndarray:
        """Level in :data:`MONITOR_BAND`, one value per frame of FR."""
        p = np.abs(np.fft.rfft(fr * self._iwin)) ** 2
        return np.sqrt(p[..., self._iband].mean(axis=-1)) + 1e-12

    def _close(self, out: list, *, sustained: bool = False) -> None:
        """Hand the bracketed region over. ``sustained`` means it force-closed at
        ``max_s`` without ever falling below the exit threshold, and then its
        frames are the only description of the channel there is: withholding them
        deadlocks the floor at whatever preceded the traffic. On the corpus's
        PACTOR-3 reference recording that is 2 s of digital silence at 0.0011
        against an inter-packet band of 0.02-0.04, and every bracket after it runs
        the full cap — 8 regions, 5 of them force-closed, where the boundaries the
        P3 decoder needs are 11."""
        if sustained:
            self.hist.extend(self.brms)
        # The held window is fed on idle frames only, so it FREEZES across a
        # bracket holding exactly the frames that opened it -- and the first idle
        # frame after this close would find a median still over the threshold and
        # reopen on nothing. Dropped for the same reason :meth:`_mute` drops it
        # across a dropout: a median must not span a gap in what it is a median
        # of. Measured before this, on a bench tape at +4 dB SNR: 75 brackets
        # where nineteen keyings exist.
        self.iheld.clear()
        out += self._regions()
        self.inb, self.acc, self.brms, self.ring, self.quiet = False, [], [], [], 0

    def _regions(self) -> list[tuple[int, np.ndarray]]:
        """The bracket, cut down to the parts of it that are a burst.

        One frame opens a bracket and another closes it, and between the two the
        gate asks nothing. On a receiver whose transmit mute is shallow the frame
        that opens it can be the band RETURNING at our own unkey, and then the
        bracket runs to the next key-down and what it holds is the turnaround: eight
        of those on 20260904T195626Z were reported as ten-second wideband overs at
        +13 dB over the floor, and the quietest ten seconds of the tape was one of
        them.

        So the region is run past the gate a second time — the same two entry tests
        and the same exit hysteresis, frame by frame — against the two floors
        :class:`MuteFloor` measures, which owe nothing to this bracket. That is the
        one thing the gate's own frozen ``nf`` cannot be asked, and nothing weaker
        will do: W0LON's idle frames stand +5.5 to +6.7 dB over the band floor on
        that tape while the band's own frames reach +6.65, so broadband level alone
        does not separate them and :data:`MONITOR_HELD_ENTER` is what does.

        Both tests answer the exit as well as the entry, because the floor this
        reads against is not the one the streaming gate closes on. ``nf`` is
        measured between bursts; :class:`MuteFloor`'s is a quarter-quantile of a
        window traffic is in, and on offair/KC9GHZ_2300 it stands 4.9 dB over the
        gate's own. An over whose broadband RMS breathes under it for SEG_HANG
        frames then ends a region the held median never stopped calling open, and
        the piece that reopens is padded back one second — 0.43 s into an over
        whose lead-in the wideband decoder needs. That tape's second DATA over came
        out 25.19 s + 1.44 s and 25.62 s + 4.39 s and decoded as neither; whole, it
        is CRC clean.

        What comes back is the burst and not the turnaround it sat in: the
        16.43-23.71 s bracket comes out as W0LON's turn-request at 22.10.

        Without a measured floor — every stream with no local transmitter on it —
        the region goes over whole, as it always did.
        """
        if self.mute is None or self.mute.floor is None or not self.held:
            return self._crop(np.concatenate(self.acc), self.start)
        fr = np.stack(self.acc)
        rms = np.sqrt((fr * fr).mean(axis=1)) + 1e-12
        ratio = self._in_band(fr) / 10.0 ** (self.mute.iband_floor / 20.0)
        k = self.iheld.maxlen
        held = np.full(len(fr), 0.0)
        if len(fr) >= k:
            held[k - 1:] = np.median(sliding_window_view(ratio, k), axis=-1)
        floor = 10.0 ** (self.mute.floor / 20.0)
        lift = held > self.held
        open_ = (rms > floor * self.enter) | lift
        shut = (rms < floor * self.exit) & ~lift
        out: list[tuple[int, np.ndarray]] = []
        start, quiet = None, 0
        for i, closing in enumerate(shut):
            if start is None:
                if open_[i]:
                    start, quiet = i, 0
                continue
            quiet = quiet + 1 if closing else 0
            if quiet >= SEG_HANG or i == len(rms) - 1:
                out += self._crop(
                    np.concatenate(self.acc[max(0, start - self.pad):i + 1]),
                    self.start + max(0, start - self.pad) * SEG_FRAME)
                start = None
        return out

    def _enter(self, rms: float, ib: float) -> bool:
        """Does this frame open a burst? A peak over the floor, or — where ``held``
        is set — a lift that has been HELD for ``hold_s``.

        See :data:`MONITOR_HELD_ENTER` for what the second test is worth and what
        it is calibrated against. It is ADDITIVE: the peak test still runs and
        still opens first where it can, so no bracket a gate found without it is
        lost. It is evaluated on every idle frame, so the in-band history obeys
        the same discipline as the broadband one — nothing inside a bracketed
        burst enters it, and no station's traffic raises the bar for the next
        station's.
        """
        if rms > self.nf * self.enter:
            return True
        if not self.held:
            return False
        self.ihist.append(ib)
        if len(self.ihist) * SEG_FRAME < MONITOR_HELD_MIN_S * FS:
            return False                 # no floor worth measuring against yet
        self.iheld.append(ib / float(np.quantile(self.ihist, SEG_FLOOR_Q)))
        return (len(self.iheld) == self.iheld.maxlen
                and float(np.median(self.iheld)) > self.held)

    def _disown_mute(self, threshold: float) -> None:
        """The mute has just been measured for the first time; everything the floor
        learned before it was learned under the wrong rule.

        It is not enough to stop feeding the floor mute frames from here on. The
        first keying of a session lands before there is enough stream to measure a
        split over — on 20260904T195626Z the mute test arms at 38 s and our
        link-setup keying is at 11.8-16.4 s — and those 4.6 s of mute put the floor
        10.7 dB under the band, where the band standing over it opened a bracket
        that then held the floor frozen, because nothing inside a bracket feeds it.
        So the history is re-read against the threshold and the frames the receiver
        was deaf through come out of it. The in-band history and the spectral
        reference carry no level this can re-test, so they are dropped whole and
        refill within SEG_FLOOR_S.
        """
        band = [r for r in self.hist if 20.0 * np.log10(r) > threshold]
        self.hist.clear()
        self.hist.extend(band)
        self.ihist.clear()
        self.iheld.clear()
        self.ref.clear()

    def _mute(self) -> None:
        """The receiver stopped hearing the band. Everything the gate was carrying
        toward an open describes audio on the far side of a hole, so it is dropped
        rather than compared across one.

        A median must not span a dropout either: what it would be averaging across
        is a hole, and the band RESUMING on the far side reads as a lift that was
        held through it. So the held window goes with the pre-roll ring.

        It is not suppressed for a guard interval afterwards, which was tried on
        2026-08-14 and cost more than it bought. A gateway answers inside the ARQ
        turnaround, so its first over begins where our own transmission stops:
        measured on `kc9ghz_connect_20260806`, the mute ends at 58.05 s and the
        CRC-clean BW2300 greeting starts at 58.08 s, 30 ms later. On
        `kb9mmt_reply_20260726` a mute ends at 14.31 s and 14.34 s is a documented
        phantom. The two are 30 ms apart from their mute in both cases, so no
        timing rule separates them — only what is in the audio does, and that is
        what the held ratio measures. A guard wide enough to suppress the phantom
        takes the head off the greeting and the CRC with it.
        """
        self.ring = []
        self.iheld.clear()

    def _live(self, rms: float) -> bool:
        """Was the receiver hearing the band in this frame?

        The measured mute first, where this stream holds one: :class:`MuteFloor`
        knows how deep the mute on THIS receiver is, and the fixed rule below only
        knows how deep one was on another. Nothing it calls muted reaches
        :attr:`hist`, so no transmission of ours can describe the band.

        The fixed rule stays under it, unchanged and additive, because it is what
        every stream with no local transmitter on it gets. Against the median of the
        live history rather than the floor quantile: a mute has to be told from the
        *channel*, not from its quietest quarter. The test arms only once there is a
        second of history to compare with, so a stream that opens mid-mute is read as
        band audio until the band arrives — the alternative is to guess, and the
        guess that a stream begins muted silences the gate for good.
        """
        if not self.mute_ratio:
            return True
        if self.mute is not None and self.mute.muted(rms):
            return False
        if len(self.hist) < SEG_LIVE_MIN:
            return True
        return rms > float(np.median(self.hist)) * self.mute_ratio

    def push(self, x: np.ndarray) -> list[tuple[int, np.ndarray]]:
        self.buf = np.concatenate([self.buf, np.asarray(x, float)])
        out = []
        while len(self.buf) - self.pos >= SEG_FRAME:
            fr = self.buf[self.pos:self.pos + SEG_FRAME]
            rms = float(np.sqrt(np.mean(fr * fr))) + 1e-12
            gs = self.t0 + self.pos
            ib = float(self._in_band(fr)) if self.held or self.mute else 0.0
            if self.mute is not None:
                self.mute.push(rms, ib)
                if self.mute.threshold is not None and not self.knows_mute:
                    self.knows_mute = True
                    self._disown_mute(self.mute.threshold)
            if not self._live(rms):
                if self.inb:                     # a mute interrupts a burst; it is
                    self._close(out)             # not the burst falling silent
                self._mute()
                self.pos += SEG_FRAME
                continue
            if not self.inb:
                self.hist.append(rms)                # the band, and only the band
                self.nf = float(np.quantile(self.hist, SEG_FLOOR_Q))
                if self._enter(rms, ib):
                    self.inb = True
                    self.acc = list(self.ring)       # silence pre-roll for find_bursts
                    self.start = gs - len(self.ring) * SEG_FRAME
                    self.acc.append(fr)
                    self.brms = [rms]
                    self.quiet = 0
                else:
                    self.ring = (self.ring + [fr])[-self.pad:]
                    if self.ref.maxlen:
                        self.ref.append(fr.copy())   # a view would pin every old buf
            else:
                self.acc.append(fr)
                self.brms.append(rms)
                if rms < self.nf * self.exit:
                    self.quiet += 1
                    if self.quiet >= SEG_HANG:       # trailing silence closes it
                        self._close(out)
                else:
                    self.quiet = 0
                if self.inb and len(self.acc) * SEG_FRAME >= self.max_s * FS:
                    self._close(out, sustained=True)  # never fell silent: it is the band
            self.pos += SEG_FRAME
        if not self.inb and self.pos > FS * 5:         # compact idle backlog
            self.buf = self.buf[self.pos:]
            self.t0 += self.pos
            self.pos = 0
        return out

    def flush(self) -> list[tuple[int, np.ndarray]]:
        return self._regions() if self.inb and self.acc else []

    def reference(self) -> np.ndarray | None:
        """The band as it was between bursts: the last ``ref_s`` of idle, live
        frames, fed on the same discipline as the floor — nothing bracketed and
        nothing the receiver was deaf through. None until there is enough of it
        to hold a spectrum against; see :func:`occupied_band_hz`."""
        if len(self.ref) * SEG_FRAME < REF_MIN_S * FS:
            return None
        return np.concatenate(self.ref)

    def level_db(self, burst: np.ndarray) -> float | None:
        """How far BURST stands over the floor this gate is tracking, in dB.

        Against ``nf`` as it reads now rather than as it read when the region
        opened, which is the same number in every case that matters: the floor is
        frozen for the whole of a burst and is a 30 s quantile either side of it,
        so it moves by a fraction of a dB across the handful of frames between a
        close and the caller asking. None before the first floor estimate exists —
        a burst in the opening second of a stream has nothing to be measured
        against, and 0.0 dB would read as a measurement.
        """
        if not self.nf or burst.size == 0:
            return None
        rms = float(np.sqrt(np.mean(burst * burst)))
        return 20.0 * float(np.log10(max(rms, 1e-12) / self.nf))


class MonitorGate(Segmenter):
    """The traffic log's gate: a floor that reads the band, and a threshold over it.

    A dropout is not a quiet channel, and the quantile only dilutes it. Frames the
    receiver was deaf through are dropped from the history outright and neither open
    nor close a burst, because the gate's whole question — is this louder than the
    band — is unanswerable while the band is not being heard. Without that, the 2.1 s
    receiver mute under each of our own transmissions pulls the floor 10 dB below the
    band noise, and the noise *resuming* on unmute clears the enter threshold: on the
    2026-07-26 recording that is eight brackets, one per over, each opening within
    10 ms of the unmute and reported as a 12 s wideband over. Eight lines, none of
    them a signal, and none of the eight real connect-requests inside those mutes.

    The enter threshold is re-derived over the honest floor, because 4.0x was
    calibrated against the diluted one and does not survive it: at 4.0x over live
    frames only, NS0A_2300 brackets 10 regions against 13 and loses all three of its
    decodable DATA overs. Measured over the four off-air recordings, 30 s of verified
    clear channel opens no gate at any threshold down to 1.5x, and neither do the
    65 s of quiet receive window on 2026-07-26 until 2.0x, where five open in the
    band noise. 2.5x is the knee: 0 false opens, against 19 bracketed regions in
    NS0A_2300 and 22 in KC9GHZ_2300 — both more than 4.0x found over the old floor.

    :class:`Segmenter` keeps the diluted floor and its 4.0x, because the connect
    tool's gate is tuned against it and a bracket it opens in error costs a
    demodulation rather than a wrong line.

    How deep a mute has to be before a frame is dropped is not a constant here —
    see :class:`MuteFloor`, which measures it off the stream — and a bracket that
    survived a shallow one anyway is re-read at close against the floors it
    measures; see :meth:`Segmenter._regions`.
    """

    enter = 2.5
    exit = 1.25
    mute_ratio = SEG_MUTE_RATIO
    pad = MONITOR_PAD
    held = MONITOR_HELD_ENTER
    hold_s = MONITOR_HOLD_S
    ref_s = SEG_FLOOR_S


# --------------------------------------------------------------------------- #
# Classification. Duration picks the family; the matching decoder names it.
@dataclass
class Result:
    kind: str                 # short type label
    info: str = ""            # callsign / detail
    quality: str = ""         # match / confidence note
    gateway: str = ""         # named CALLED station, if any
    caller: str = ""          # named CALLER (link-setup), if any
    offset_hz: float | None = None   # sender's tuning error, where a search measured it


# --------------------------------------------------------------------------- #
# Handshake scanner: the MFSK handshake bursts, found on the raw stream.
#
# An energy gate is the wrong instrument for these and the 2026-07-26 recording
# says so twice over. Our own eight connect-requests are *quieter* than the band,
# not louder: the rig mutes its receiver while it keys, so they arrive as a
# -33 dBFS monitor bleed inside a 2.1 s hole in a -13 dBFS channel, and no
# threshold over a floor read from the band will ever bracket one. KB9MMT's two
# answers are barely above the band it does hear, and the second of them begins
# 0.34 s before our transmission ends — its whole preamble is under our own mute.
#
# So these bursts are found by what they say. Every tone of a handshake burst is
# fixed: the preamble by the burst kind, the payload by the callsign being called
# [spec 04 §4.2.2, §4.2.3]. Transform the stream once on a 32-sample lattice
# (:func:`vara_arq._tone_track`, band-limited to the tone alphabet), read the tones
# each candidate burst start would see, and score. That is the search kestrel's own
# handshake runs to hear a gateway answer; the only difference here is that the
# monitor does not know who was called, so the payload is scored against every
# candidate callsign instead of one.
SCAN_GRID = 32                    # alignment lattice, in samples
SCAN_BLOCK = FS // 2              # audio gathered before the search runs again
SCAN_PRE_SLACK = 2                # preamble tones a lock may lose to fading

# Frequency offsets searched, in tone bins of 23.4375 Hz. A station tuned off
# frequency shifts every tone by the same amount, so its preamble still matches —
# against a shifted copy of itself. The 2026-07-26 recording carries a third
# station calling 2 bins low, which is how it gets reported at all; ±3 bins is
# ±70 Hz, past which two operators are not on the same channel.
SCAN_SHIFTS = tuple(range(-3, 4))

# The BW500 handshake, which shares the preamble and does not share the alphabet.
#
# [spec 04 §4.2.1] says the payload tone VALUES above are the BW2300 set and that
# BW500 keeps the counts and changes the values; [spec 04 §4.2C] says the
# connected-ack preamble is the same at every bandwidth and only its state symbols
# narrow, "at BW500 they stay inside 52..74 where the wide modes spread 31..98".
# The connect-request is the same: the same ten fixed preamble tones, then a
# payload drawn from 14 carriers on a 46.9 Hz lattice inside 1172-1781 Hz,
# against the 70-carrier 23.4 Hz alphabet spanning 680-2297 Hz that BW2300 uses.
# A BW500 request read against the BW2300 plan locks its preamble and scores its
# payload at chance.
#
# So the payload score is not evidence of anything until you know which alphabet
# the sender used, and the lattice says which — which is what `_ScanPlan.kind_for`
# uses it for now that :data:`VF.CR500` can regenerate a callsign from the other
# alphabet, and the lattice itself comes from that alphabet rather than from the
# readings below. Those readings are what established it, and are kept for that.
#
# Measured 2026-08-14, on a 546 s receive-only recording of 7096.5 kHz carrying a
# VARA 500 exchange whose two callsigns the Winlink RMS feed names independently
# of the audio, and whose spectrum is a 600 Hz square-shouldered signal centred on
# the high speed capable channel: two bursts, at wav 4.348 s and 286.176 s, each lock 9 of
# the 10 preamble tones exactly at zero frequency offset with those tones standing
# 12-30 dB over the median of their own symbol window, then put 24 of their next
# 28 tones on this lattice. The four that miss are each one bin off a lattice
# carrier (65, 71, 53, 53), which is what a fading MFSK symbol reads as.
#
# Corroborated twice, off other recordings and other bursts. A real VARA BW500
# connect-response, captured 2026-07-13 over a loopback where the PTT ledger says
# which station keyed it: preamble 8/8 exact, and all 15 payload tones even and
# inside 50..72, scoring 1/15 against the BW2300 generator. And the third station
# on the 2026-07-26 off-air recording, calling somebody 47 Hz low at 100.81 s,
# whose payload had never matched any candidate: 28 of 28 on this lattice.
SCAN_BW500_LATTICE = VF.BW500_TONES.carriers

# 28 rather than the 31 of [spec 04 §4.2.1], because 28 is what this recording can
# support and the first 28 are the first 28 under either count. The 4.348 s burst
# runs 38 symbol slots and stops: in-band power falls 13 dB between slot 37 and
# slot 38, and 1.57 s of keying is 0.18 s short of the 1.75 s a 41-symbol request
# takes.
#
# The count itself is settled and it is 31, so the doubt this comment used to
# record is closed. Measured 2026-08-15 on the 2026-07-13 BW500 loopback
# captures: keyed audio spans 83967 samples, 41.000 symbol advances to three
# decimals, in every one of the four sessions whose burst start the recorder did
# not clip. The on-air burst lost its last three slots to that recorder's
# independently measured ~9% sample loss (1.75 s x 0.906 = 1.59 s), and the count
# of three follows from that arithmetic alone: 41 keyed against 38 read. Where
# inside the payload they fell is NOT established — locating them means reading
# the burst against an assumed callsign, and no fit in this tree does that. This
# stays 28 because a check on a fading path should not be widened past what the
# path delivers.
SCAN_BW500_PAYLOAD = 28

# Both bursts score 24/28 = 0.857, and the 2026-07-26 one 28/28. Over 1154
# alignments sampled across two negative controls recorded the same afternoon —
# 6800 kHz, which has no amateur allocation, and ten minutes of a 40 m channel
# every classifier and the feed agree was empty — band noise means 0.23 on this
# lattice and never once exceeds 0.58. 0.7 has margin on both sides, and it
# decides a LABEL rather than a detection: nothing here can make the scanner
# report a burst it would not otherwise have reported.
SCAN_BW500_ACCEPT = 0.7


@dataclass(frozen=True)
class _ScanPlan:
    """One burst, pre-arranged for the lattice search — at every bandwidth it can
    arrive in. The bandwidths share a preamble, a symbol count and an alignment
    and differ only in the payload alphabet, so they are one search and one
    detection, not two."""
    kind: VF.BurstKind            # the burst at BW2300
    alt: VF.BurstKind | None      # the same burst at BW500, when one exists
    off: np.ndarray               # lattice index of each symbol, from the burst start
    lsb: np.ndarray               # parity the BW2300 alphabet fixes per payload tone
    need: int                     # payload tones an acceptance takes

    @classmethod
    def build(cls, kind: VF.BurstKind, alt: VF.BurstKind | None = None) -> _ScanPlan:
        n = len(kind.preamble) + kind.n_payload
        off = np.rint((np.arange(n) * MK.HOP + MK._WOFF) / SCAN_GRID).astype(np.int64)
        # carrier = base + parity + 14P + 2D, so the symbol index alone fixes the
        # low bit of every payload tone  [spec 04 §4.2.3].
        #
        # BW2300's only. BW500's alphabet has no parity term — its low bit is 0
        # throughout — so the same rule there is the test "are these tones all
        # even", which is not a filter. Measured over 120 s of the 2026-08-14
        # recording against 343 callsigns: adding it as a second pattern takes the
        # scanner from 59x real time to 19x with the CR plan alone and to 9.9x with
        # both, because it hands `VF.best_match` thousands of alignments band noise
        # happens to put on even bins. It buys nothing that is not already had —
        # the preamble route finds BW500 bursts, and it found both of the two held
        # at 9/10. Left out, both bandwidths together run at 59.4x, which is the
        # baseline: `kind_for` decides the alphabet off a lattice test the result
        # already needed.
        lsb = np.array([(kind.tones.base + kind.tones.parity(i, kind.par0)) & 1
                        for i in range(kind.n_payload)])
        return cls(kind, alt, off, lsb,
                   int(np.ceil(ACCEPT * kind.n_payload)))

    def kind_for(self, heard: np.ndarray, npre: int) -> VF.BurstKind:
        """Which bandwidth's alphabet to regenerate a callsign against.

        Decided by the tone lattice rather than by trying both, because trying
        both doubles the one step this scanner's cost is dominated by. The
        lattice is 14 of the 70 carriers and separates them cleanly: band noise
        reaches 0.23 of the payload on it and a BW2300 burst cannot beat chance
        on an alphabet it did not use."""
        if self.alt is not None and _bw500_payload(heard, npre):
            return self.alt
        return self.kind


class HandshakeScanner:
    """Every MFSK handshake burst on the stream, named where it can be.

    Connect-request and connect-response only — the two bursts a callsign can be
    read out of, and the two whose ground truth is settled. Nothing here rests on
    the connected-ack, whose model is under review at the time of writing; it stays
    in ``classify``'s duration bins, where a change to it costs one label rather
    than the attribution of a whole exchange.

    Two routes to a candidate alignment, because either can be the only one open:

      * the **preamble**, matched against each of ``SCAN_SHIFTS`` — the route that
        finds a burst addressed to a station not on the candidate list, and the
        only route that survives a frequency offset.
      * the **payload's parity**, at zero offset. The tone alphabet fixes the low
        bit of every payload tone from its symbol index alone, so an alignment that
        cannot match ``need`` tones exactly cannot match ``need`` parities either:
        the filter discards no acceptance it would have made, and it cuts 223,000
        alignments to about 8,000 before any callsign is regenerated. This is the
        route that reads a burst whose preamble was never received. It is BW2300's
        parity and only BW2300's — see :meth:`_ScanPlan.build` for why BW500 has
        no usable one and reaches this scanner by its preamble alone.

    Cost is dominated by that last step — one ``VF.best_match`` over the candidate
    list per surviving alignment. Against 343 callsigns it runs at 21x real time on
    the 2026-07-26 recording and 9x on the busiest session in the corpus, so the
    recogniser stays the project's one recogniser rather than a faster copy of it.
    """

    def __init__(self, calls: Sequence[str]):
        self.calls = list(calls)
        self._plan = [_ScanPlan.build(VF.CR, VF.CR500),
                      _ScanPlan.build(VF.CONNECT_RESPONSE, VF.CONNECT_RESPONSE_500)]
        self._span = max(int(p.off[-1]) for p in self._plan)
        self._buf = np.zeros(0)
        self._track = np.zeros(0, dtype=np.int32)
        self._t0 = 0              # sample index of the window self._track[0] read
        self._last: dict[str, int] = {}

    def push(self, x: np.ndarray) -> list[tuple[int, Result]]:
        """Detections in the audio so far, as ``(burst start sample, Result)``."""
        self._buf = np.concatenate([self._buf, np.asarray(x, dtype=np.float64)])
        if len(self._buf) < MK.NFFT + SCAN_BLOCK:
            return []
        # Windows start at buf[0], which stays lattice-aligned because exactly the
        # samples consumed by the windows just taken are the ones dropped.
        nwin = (len(self._buf) - MK.NFFT) // SCAN_GRID + 1
        self._track = np.concatenate(
            [self._track, _tone_track(self._buf[:MK.NFFT + (nwin - 1) * SCAN_GRID],
                                      SCAN_GRID)])
        self._buf = self._buf[nwin * SCAN_GRID:]
        n = len(self._track) - self._span
        if n <= 0:
            return []
        out = []
        for plan in self._plan:
            out += self._detect(plan, self._track[np.arange(n)[:, None] + plan.off])
        self._track = self._track[n:]        # keep only the unscored tail
        self._t0 += n * SCAN_GRID
        return sorted(out)

    def _candidates(self, plan: _ScanPlan, heard: np.ndarray) -> dict[int, int]:
        """``{alignment: tone shift}`` worth regenerating a callsign for."""
        npre = len(plan.kind.preamble)
        pre = np.asarray(plan.kind.preamble)
        score = np.stack([(heard[:, :npre] == pre + s).sum(1) for s in SCAN_SHIFTS])
        best = score.argmax(0)
        found = {int(i): SCAN_SHIFTS[int(best[i])]
                 for i in np.flatnonzero(score.max(0) >= npre - SCAN_PRE_SLACK)}
        for i in np.flatnonzero(((heard[:, npre:] & 1) == plan.lsb).sum(1) >= plan.need):
            found.setdefault(int(i), 0)       # preamble unheard, payload parity fits
        return found

    def _detect(self, plan: _ScanPlan, heard: np.ndarray) -> list[tuple[int, Result]]:
        kind, npre = plan.kind, len(plan.kind.preamble)
        pre = np.asarray(kind.preamble)
        found = self._candidates(plan, heard)

        def lock(a: int) -> int:
            return int((heard[a][:npre] == pre + found[a]).sum())

        out = []
        for group in _runs(sorted(found), MK.HOP // SCAN_GRID):
            # Best callsign match across the plateau, and the preamble lock to break
            # its ties. A burst whose payload alphabet is not the one being scored
            # has no callsign match to rank on, so ranking on that alone picks an
            # arbitrary member of the plateau: on the 2026-08-14 VARA 500 recording
            # that reported 8/10 for two bursts that lock 9/10 one lattice step away.
            #
            # Which is why a score below `need` ranks as nothing at all rather than
            # as itself. Chance on the BW500 alphabet is 1 tone in 14, three times
            # what it is on BW2300's 70, and it varies enough across a plateau to
            # outrank the true alignment's preamble — a synthesised BW500 request
            # addressed to a station not on the candidate list then went unreported
            # entirely, the lock at the alignment chance had picked falling under
            # SCAN_PRE_SLACK.
            scored = []
            for a in group:
                shifted = heard[a] - found[a]
                at = plan.kind_for(shifted, npre)
                scored.append((VF.best_match(list(shifted), self.calls, at) + (at,),
                               lock(a), a))
            (cs, m, tot, at), locked, align = max(
                scored, key=lambda s: (s[0][1] if s[0][1] >= plan.need else 0, s[1]))
            m = max(m, 0)                 # no candidate list scores nothing, not -1
            shift = found[align]
            if m < plan.need and locked < npre - SCAN_PRE_SLACK:
                continue                      # parity alone is not a detection
            start = self._t0 + align * SCAN_GRID - MK._WOFF
            if start - self._last.get(kind.name, -1 << 40) < len(plan.off) * MK.HOP:
                continue                      # still inside the burst just reported
            self._last[kind.name] = start
            out.append((start, _scan_result(at, cs, m, tot, locked, npre, shift,
                                            _bw500_payload(heard[align] - shift, npre))))
        return out


def _runs(idx: Sequence[int], gap: int) -> list[list[int]]:
    """Split sorted alignments into groups no more than ``gap`` apart. One burst
    matches across its whole plateau — about 60 lattice points — and is one event."""
    out: list[list[int]] = []
    for i in idx:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _bw500_payload(heard: np.ndarray, npre: int) -> tuple[int, int] | None:
    """How much of HEARD's payload sits on the BW500 tone lattice, once enough of
    it does to say the sender was transmitting at BW500. None otherwise."""
    pay = heard[npre:npre + SCAN_BW500_PAYLOAD]
    on = sum(1 for v in pay if int(v) in SCAN_BW500_LATTICE)
    return (on, len(pay)) if on >= SCAN_BW500_ACCEPT * len(pay) else None


def _scan_result(kind: VF.BurstKind, cs: str, m: int, tot: int,
                 pre: int, npre: int, shift: int,
                 bw500: tuple[int, int] | None = None) -> Result:
    name, verb = (("CR", "calling") if kind in (VF.CR, VF.CR500)
                  else ("connect-response", "to"))
    hz = MK.carrier_to_hz(shift)
    off = f", {hz:+.0f} Hz off frequency" if shift else ""
    at = " at BW500" if kind.tones is VF.BW500_TONES else ""
    if m >= ACCEPT * tot:
        return Result(name, f"{verb} {cs}{at}{off}",
                      f"{m}/{tot} payload, {pre}/{npre} preamble", gateway=cs,
                      offset_hz=hz)
    if bw500:
        # The burst is BW500 and no candidate callsign regenerates its payload, so
        # the bandwidth is the whole of what can be said. The alphabet exists now
        # (VF.CR500) — what is missing is the station, which is a candidate list
        # that does not hold it rather than a waveform that cannot be read.
        on, n = bw500
        return Result(name, f"{verb} a station not on the candidate list at BW500{off}",
                      f"{on}/{n} payload on the BW500 tone lattice, "
                      f"{m}/{tot} against the best candidate, {pre}/{npre} preamble",
                      offset_hz=hz)
    return Result(name, f"{verb} a station not on the candidate list{off}",
                  f"{m}/{tot} payload, {pre}/{npre} preamble", offset_hz=hz)


def _best_token(seg: np.ndarray) -> tuple[VC.TokenMatch | None, str]:
    """Best DBPSK control-token match. One vocabulary, because one bandwidth has
    one: BW2300's short bursts are index-modulated, not DBPSK  [see vara_control]."""
    m = VC.detect_token(seg)
    return m, "500" if m else ""


def _mfsk_match(seg: np.ndarray, kind: VF.BurstKind,
                calls: list[str]) -> tuple[str, int, int]:
    n_sym = len(kind.preamble) + kind.n_payload
    tones = MK.demod_tones(seg, n_sym)
    return VF.best_match(tones, calls, kind)


_LP = firwin(801, 120 / (FS / 2))
_CTRL_CARRIERS = (675.0, 1125.0, 1350.0, 1575.0, 1875.0)   # control sub-bands, both BWs


def _dbpsk_collapse(seg: np.ndarray) -> float:
    """Raw-differential-BPSK collapse over the control sub-bands, **in units of its
    own noise floor** — how far a VARA session/control burst (spec 02 §2.6) stands
    out of what the statistic reads on band noise of the same length.

    The collapse itself is a phase mean over the burst's ~ncol differentials,
    maximised over five carriers and thirty-two onsets, so on noise it is not a
    number near zero: it is the largest of 160 tries at averaging ncol unit vectors,
    and it falls as 1/sqrt(ncol). Measured over 3120 windows of the clear-channel
    capture and the two 2026-08-06 calls nobody answered, from 16 to 320 columns,
    the raw collapse's 95th percentile runs 0.92 down to 0.25 — and multiplied by
    sqrt(ncol) it is 3.66 to 4.68, flat across a twentyfold span. That flat number
    is what a threshold can be set against; the raw one cannot be, and the fixed
    0.35 it used to be compared against passed 100% of 16-column noise and 0.4% of
    320-column noise for exactly that reason.
    """
    n, nn = len(seg), np.arange(len(seg))
    best = 0.0
    for f0 in _CTRL_CARRIERS:
        bb = fftconvolve(seg * np.exp(-2j * np.pi * f0 * nn / FS), _LP, "same")
        for onset in range(0, 512, 16):
            c = onset + np.arange(n // 512 - 1) * 512
            c = c[c < len(bb)]
            if len(c) < 10:
                continue
            d = bb[c][1:] * np.conj(bb[c][:-1])
            m = np.abs(d) > 0.3 * np.median(np.abs(d))
            if m.sum() > 6:
                best = max(best, abs(np.mean(np.exp(2j * np.angle(d[m])))))
    return best * np.sqrt(max(n // 512, 1))


def _simul_tones(seg: np.ndarray, blk: int = 1024, lo: int = 300,
                 hi: int = 2600) -> float:
    """Mean count of simultaneous spectral peaks per block (≥0.35×block-max, >100 Hz
    apart). The hallmark that separates single-tone MFSK from multi-carrier DBPSK:
    an MFSK handshake burst carries **one** tone at any instant (≈1.0); a VARA DBPSK
    session/token burst lights several sub-carriers at once (≥1.4, multi-carrier ≥2).
    Measured over labeled ground truth: MFSK CR/response = 1.00 exactly."""
    b0, b1 = int(lo * blk / FS), int(hi * blk / FS)
    dist = max(1, int(100 * blk / FS))
    counts = []
    for i in range(0, len(seg) - blk, blk // 2):
        sp = np.abs(np.fft.rfft(seg[i:i + blk] * np.hanning(blk)))[b0:b1]
        if sp.max() < 1e-4:
            continue
        pk, _ = find_peaks(sp, height=0.35 * sp.max(), distance=dist)
        counts.append(len(pk))
    return float(np.mean(counts)) if counts else 0.0


def _is_mfsk_handshake(seg: np.ndarray) -> bool:
    """Structural gate for the MFSK handshake bins (CR / connect-response): the burst
    carries one tone at a time. MFSK measures 1.0; every DBPSK burst measures ≥1.4 and
    off-air traffic 2.0-10.9. Stops an in-session burst that merely lands in a handshake
    duration bin from being scored against the wrong template and reported as a
    near-miss connect-response.

    A dominant-bin-hop test used to gate this too. Measured over 47 real off-air bursts
    it passed 47 of them, band noise included — it separated only on the loopback corpus
    it was tuned against — so it is gone rather than left in looking meaningful."""
    return _simul_tones(seg) < 1.5


def _index_mod_frame(seg: np.ndarray) -> bool:
    """Does SEG's spectrum sit in ONE bin per 512-sample column?

    Mean peak-to-mean over the rec3 span above 4 — index-mod rather than noise or
    weak DBPSK. That is the whole test, and it is a test of shape, not of VARA:
    measured over the regression corpus it fires 30 times across five recordings
    that contain no VARA at all (8 on the PACTOR-2 reference, 9 on the KB8AY PACTOR-1
    capture, 6 on the two-sided PACTOR-1 recording, 5 on the local PACTOR-1 one,
    2 on HB9AK), because a keyed narrowband carrier is one-hot per column whoever
    is sending it. It says nothing about who, about role, or about position in a
    session, and the line it produces no longer says any of those either.

    An argmax-spread clause used to be ANDed here; off air it passed 41 of 47
    bursts against 11 for this one, so it added no separation and is dropped.

    **What it fires on off the air, and what it does not.** Sampled as random
    windows at thirteen widths from 16 to 320 columns, this reads 16% over the
    three recordings the DBPSK gate's floor was measured on — flat in length, so
    it is not that gate's defect wearing a different label. It decomposes: 1.0%
    on 30 s of verified-clear 40 m, 0.0% on the two 80 m windows holding one
    static crash each, and 25.3% on the two 2026-08-06 calls nobody answered.
    Inside those two the split is sharper still — 2% of the windows that hold
    live band, and 100% of the ones that hold this station's own transmit mute,
    which is two thirds of both recordings and sits 16-21 dB under their own
    in-band median.

    A dead input is one-hot, and no threshold recovers that. Measured per column:
    the mute's peak stands 12.7x over its own median, a real gateway's MFSK
    session frame 7.2x and live band noise 2.2x — so the mute is the MORE one-hot
    of the two signals, and a cut that removed it would remove every real frame
    first. What separates them is level and not shape, and level is what the gate
    ahead of this already reads: over the whole of `silent_kd9usw.wav`, this tool
    brackets nine bursts and names none of them here, because a segmenter tracking
    the floor never opens on audio 16 dB beneath it."""
    r = RX.RECORDS[RX.BASE_LEVEL]
    dw, fb, span = r.dw50, r.first_bin, r.span
    ncol = len(seg) // dw
    if ncol < 16:
        return False
    peak = []
    for col in range(ncol):
        sp = np.abs(np.fft.fft(seg[col * dw:(col + 1) * dw])[fb:fb + span])
        peak.append(sp.max() / (sp.mean() + 1e-9))
    return float(np.mean(peak)) > 4.0


# The control-token vocabulary's own span, in 512-sample columns: `detect_token`
# demodulates len(pattern)+1 of them. A burst outside this span is not a token this
# code could have resolved at any SNR, and saying so is the cheapest structure
# available once the patterns have missed.
_TOKEN_COLS = sorted(len(bits) + 1 for bits in VC.TOKENS.values())
_TOKEN_COL_LO, _TOKEN_COL_HI = _TOKEN_COLS[0], _TOKEN_COLS[-1]

# Roughly the 95th percentile of :func:`_dbpsk_collapse` on band noise, and it is
# one number rather than thirteen because that function is normalised: measured at
# 16, 24, 32, 36, 40, 44, 48, 64, 96, 128, 192, 256 and 320 columns over the
# clear-channel capture and the two 2026-08-06 calls nobody answered, the noise
# 95th percentile is 3.66-4.68 with no trend in it, and this passes 0.0-5.8% of
# those windows at every one of the thirteen widths.
_COLLAPSE_MIN = 4.5


def _unnamed_burst(seg: np.ndarray, dur: float, n: int,
                   level_db: float | None) -> Result:
    """A burst that matched no handshake kind and no named token.

    VARA's short rec3 index-mod session-setup frames land here, and so does
    everything else with a one-hot column spectrum — see :func:`_index_mod_frame`,
    which cannot tell them apart. This line used to read `session-setup ·
    initiator rec3 short frame`, three claims deep: that the burst was VARA, that
    it was the caller's, and that it sat before the first data over. The test
    supports none of them, and one night's log carried five of these against a
    gateway that never answered, three of them on our own connect-request cadence.

    The DBPSK branch below carried the same defect, one line further down.
    `session · DBPSK control/keepalive (token vocabulary unknown)` was printed for
    every multi-carrier burst whose patterns missed, whatever its size: on
    2026-08-18 against KC9GHZ it named eleven pre-connect bursts running from 22 to
    278 columns, a 12x spread, one of them 2.97 s long. A reader of that log would
    conclude the peer sent nothing but keepalives, and seven candidate
    connect-responses went unexamined on the strength of it.

    So the label claims what was measured — a DBPSK collapse on the control
    sub-bands, and no pattern within Hamming 2 — and the size decides the rest,
    against the vocabulary's own column span. `session` is gone from it too. The
    collapse test is a shape and shapes are not protocols: measured over the shared
    regression corpus it fires 29 times across five recordings that hold no VARA at
    all — 8 on 80 m band noise, 14 across two PACTOR-1 captures, 5 on the PACTOR-3
    reference,
    2 on the ARDOP suspect — against 9 firings on the three VARA fixtures.

    What it no longer fires on is band noise, and until 2026-08-23 that was most of
    what it fired on. The threshold was a fixed 0.35 against an un-normalised
    collapse, so it was thirteen different thresholds at thirteen burst lengths:
    100% of 16-column noise windows passed it and 0.4% of 320-column ones, and at
    the 32-45 columns the token vocabulary occupies, 84%. On the four 2026-08-23
    arms that reads as 26 `DBPSK burst` lines carrying no burst — the whole of what
    KE8LVA, KB3AC-10 and KD0PYG were reported as sending after their overs — all
    but one of them 2.0-2.3 kHz wide, which is the channel rather than a signal in
    it, and all of them at -0.1 to +2.6 dB. The 26 are gone; the one burst among
    them with structure, 108 columns at +3.3 dB in the KE8LVA session, stays.

    The level is how far the burst stood over the floor the segmenter was tracking.
    NOT a level gate: measured over 34 recorded connect attempts these bursts run
    +4.4 to +19.0 dB, and the wideband overs and control tokens in the same
    recordings reach down to +1.6 and +2.1 — so a threshold that removed the weak
    ones here would cost real detections there, and the number belongs in the
    reader's hands instead of in a gate.
    """
    cols = n // 512
    # Signed, because a burst AT the floor is the case that misled: `+-0.3 dB`
    # would read as a typo where `-0.3 dB` reads as the measurement it is.
    lvl = f"{dur:.2f}s" + (f", {level_db:+.1f} dB" if level_db is not None else "")
    if _index_mod_frame(seg):
        return Result("one-hot burst", "one peak per column — a rec3 short frame "
                      "or any other keyed carrier; not decoded, no sender",
                      f"{lvl}, {cols}col")
    if _dbpsk_collapse(seg) > _COLLAPSE_MIN:
        fit = ("shorter than" if cols < _TOKEN_COL_LO else
               "longer than" if cols > _TOKEN_COL_HI else "within")
        return Result("DBPSK burst", f"tokens unresolved — {fit} the "
                      f"{_TOKEN_COL_LO}-{_TOKEN_COL_HI} col token vocabulary",
                      f"{lvl}, ~{cols}col")
    return Result("unknown", "", f"{lvl}, ~{_nsym(n)}sym")


def _is_link_setup(fr: RX.Frame2300) -> bool:
    return fr.crc_ok and VF.is_link_setup(fr.frame_bytes)


# A VARA over fills its channel, and where an undecoded burst sits against the
# channel is the cheapest thing about it that can be measured without a decode.
# It used to be measured as a 90%-of-power span over the burst's own 10th
# percentile, and on a bracket whose in-band energy is mostly noise that measures
# the noise: the seven KE8LVA bursts of 2026-09-03 came out `~1423-1994 Hz wide`
# at centres of 1601-1780 Hz, a band they hold nothing in, and were written up as
# VARA traffic this receiver could not read. Against a reference window of the
# band between bursts, their excess runs -2 to +2 dB across 375-1875 Hz and steps
# to +8...+13 dB across 2000-2510 Hz: a 510 Hz occupant just above the passband.
#
# So the band is the -10 dB edges of the burst's excess power over that
# reference, and the line prints the edges against the bands the modem keys:
#
#     BW2300 rec3   843-2250 Hz     BW2750 base   562-2344 Hz
#     BW2300 rec2   797-2297 Hz     BW500        1250-1750 Hz
#
# Control, measured the same way: VE3KPG's record-3 over at 94.8 s of the same
# night's second tape reads 797-2297 Hz.
VARA_BANDS = (("BW2300 rec3", 843.0, 2250.0), ("BW2300 rec2", 797.0, 2297.0),
              ("BW2750 base", 562.0, 2344.0), ("BW500", 1250.0, 1750.0))
BAND_FFT = 4096                   # 11.7 Hz bins
BAND_EDGE_DB = 10.0               # edges are this far under the peak excess
BAND_TOL_HZ = 60.0                # an edge this close to a band's is at it
BAND_GAP_HZ = 200.0               # a fade this narrow inside the band is bridged
BAND_MIN_SNR_DB = 5.0             # under this the edges would be noise
REF_MIN_S = 5.0                   # reference audio the spectrum is held against


def _psd(x: np.ndarray) -> np.ndarray:
    n, w = BAND_FFT, np.hanning(BAND_FFT)
    starts = range(0, len(x) - n + 1, n // 2)
    acc = np.zeros(n // 2 + 1)
    for i in starts:
        acc += np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2
    return acc / max(len(starts), 1)


def occupied_band_hz(seg: np.ndarray,
                     ref: np.ndarray) -> tuple[float, float, float] | None:
    """``(lo, hi, snr_db)``: the -10 dB edges of SEG's excess power over REF, and
    how far the peak bin stands over the reference. None when nothing stands out
    far enough for edges to mean anything."""
    f = np.fft.rfftfreq(BAND_FFT, 1 / FS)
    band = (f >= 300) & (f <= 2900)
    ps, pr = _psd(seg)[band], _psd(ref)[band] + 1e-20
    excess = np.clip(ps - pr, 0, None)
    k = int(np.argmax(excess))
    snr = 10 * np.log10(ps[k] / pr[k])
    if snr < BAND_MIN_SNR_DB:
        return None
    idx = np.flatnonzero(excess >= excess[k] * 10 ** (-BAND_EDGE_DB / 10))
    gap = int(BAND_GAP_HZ / (FS / BAND_FFT)) + 1
    lo = hi = int(np.searchsorted(idx, k))
    while lo > 0 and idx[lo] - idx[lo - 1] <= gap:
        lo -= 1
    while hi < len(idx) - 1 and idx[hi + 1] - idx[hi] <= gap:
        hi += 1
    ff = f[band]
    return float(ff[idx[lo]]), float(ff[idx[hi]]), float(snr)


def band_label(lo: float, hi: float) -> tuple[bool, str]:
    """``(fills, text)``: whether the edges sit at both edges of a band the modem
    keys, and the words for where they sit."""
    fills = [name for name, b_lo, b_hi in VARA_BANDS
             if abs(lo - b_lo) <= BAND_TOL_HZ and abs(hi - b_hi) <= BAND_TOL_HZ]
    if fills:
        return True, "fills " + " / ".join(fills)
    inside = [(b_hi - b_lo, f"{name} ({b_lo:.0f}-{b_hi:.0f})") for name, b_lo, b_hi in VARA_BANDS
              if lo >= b_lo - BAND_TOL_HZ and hi <= b_hi + BAND_TOL_HZ]
    if inside:
        return False, f"inside {min(inside)[1]} and narrower than its over"
    widest = max(VARA_BANDS, key=lambda b: b[2] - b[1])
    return False, f"outside every VARA band (the widest, {widest[0]}, is {widest[1]:.0f}-{widest[2]:.0f})"


def classify(seg: np.ndarray, calls: list[str],
             decode_wideband: bool = True,
             level_db: float | None = None,
             reference: np.ndarray | None = None) -> Result:
    """Name one bracketed burst, as specifically as the evidence allows.

    ``level_db`` is how far the burst stood over the noise floor the segmenter was
    tracking when it opened, and it is carried into the lines that rest on shape
    rather than on a decode. A "session-setup" was reported at the noise floor on
    a night when nothing answered, and the line gave a reader no way to see that:
    the identification and the strength of the thing identified were the same
    sentence. Callers that have no floor to compare against pass None, and then
    the line simply omits the figure rather than inventing one.

    ``reference`` is the band between bursts, and it is what an undecoded
    wideband bracket's occupied band is measured against; without one the line
    says so rather than measuring the bracket against itself.
    """
    n = len(seg)
    dur = n / FS

    if n >= 180000:                                    # ~4.4 s wideband OFDM over
        # The wideband decode ranks the frame alignments and turbo-decodes the best
        # few — around a second per burst when one locks, five when none does. Still
        # enough to back up the audio on a real-time path (a live connect or
        # --device), so callers there pass decode_wideband=False.
        lvl = f"{dur:.1f}s" + (f", {level_db:+.1f} dB" if level_db is not None else "")
        if not decode_wideband:
            if cp := _cp_over(seg, lvl):
                return cp
            # "over" is a VARA frame, and nothing here decoded one. On a channel
            # the gate never stops hearing, every bracket force-closes at the
            # segmenter's cap and arrives at this branch: calling W8MW on 3595.0
            # on 2026-08-06 produced five of these lines off 71 s of 80 m band
            # noise carrying a narrowband signal near the channel centre, and the
            # best rec3 reference-column population anywhere in that recording is
            # 9 of 24 against a gate of 16. What is measured is the duration and
            # the level, so that is what the line says.
            return Result("long burst", "longer than any handshake burst; not "
                          "decoded on the live path", lvl)
        # A segment can hold more than one over — fades split a keying and the
        # segmenter coalesces the pieces back, and gateways send overs back to back.
        # decode_over returns the FIRST clean frame, so decode each sub-over or the
        # rest are dropped without a trace.
        subs = RX.detect_overs(seg) or [(0, n)]
        frames = [RX.decode_over(seg, s0, e0) for s0, e0 in subs]
        good = [f for f in frames if f.crc_ok]
        link = next((f for f in good if _is_link_setup(f)), None)
        if link is not None:
            caller = VF.caller_from_link_setup(link.frame_bytes) or "?"
            return Result("link-setup", f"from {caller}", "CRC ok", caller=caller)
        if good:
            nb = sum(RX.payload_bytes(f.level) for f in good)
            more = f" in {len(good)}/{len(subs)} overs" if len(subs) > 1 else ""
            return Result("DATA over", f"{nb} payload bytes{more}", "CRC ok")
        if cp := _cp_over(seg, lvl):
            return cp
        # A bracket that ran to the segmenter's cap never fell silent, which makes
        # it channel rather than one keying: the 2026-08-14 pair were 11.8 and
        # 10.5 s against the 4.3-4.4 s a BW2300 over takes.
        held = " (never fell silent; a run of overs or the channel)" if dur >= SEG_MAX_S - 0.5 else ""
        if reference is None:
            return Result("long burst", "undecoded; no window of the band between "
                          f"bursts yet to measure its edges against{held}", lvl)
        edges = occupied_band_hz(seg, reference)
        if edges is None:
            return Result("long burst", "undecoded; stands too little over the band "
                          f"between bursts for its edges to be read{held}", lvl)
        lo, hi, snr = edges
        fills, where = band_label(lo, hi)
        # `unknown` rather than a new kind name for a burst that fills no band:
        # creance grades any kind it does not recognise as VARA, and a burst
        # outside the passband has no protocol to name.
        return Result("wideband" if fills else "unknown",
                      f"undecoded, {lo:.0f}-{hi:.0f} Hz {where}{held}",
                      f"{lvl}, peak {snr:+.1f} dB over the band; no CRC lock")

    # The connected-ack, ahead of the duration bins. Its four fixed two-tone symbols
    # [spec 04 §4.2C] identify it on their own — measured, no false match in 1227 s of
    # off-air HF — so it needs no duration gate, and having one only meant a bracket
    # with a little silence on it landed in the connect-response bin instead.
    if dur < 1.5:
        wide = _ack_plateau(seg)
        if wide >= _ACK_PLATEAU:                       # names nobody: no `gateway`
            return Result("connected-ack", "(carries no callsign)", f"preamble x{wide}")

    # Handshake bins: a DEFINITIVE match (≥ACCEPT against a known callsign) always
    # reports — no DBPSK burst scores that high against an MFSK template, so it holds
    # even at low SNR where the structural gate would trip. The gate only decides
    # whether a *near-miss* is a genuine weak handshake worth surfacing as "best N/tot"
    # or an in-session DBPSK burst that merely landed in this duration bin (→ session).
    if 1.55 <= dur <= 2.0:                             # ~41-symbol connect-request
        cs, m, tot = _mfsk_match(seg, VF.CR, calls)
        if tot and m >= ACCEPT * tot:
            return Result("CR", f"calling {cs}", f"{m}/{tot}", gateway=cs)
        if _is_mfsk_handshake(seg):
            return Result("CR", "calling (dest not in gateway list)", f"best {m}/{tot}")
        return _unnamed_burst(seg, dur, n, level_db)

    if 0.85 <= dur < 1.25:                             # ~23-symbol connect-response
        cs, m, tot = _mfsk_match(seg, VF.CONNECT_RESPONSE, calls)
        if tot and m >= ACCEPT * tot:
            return Result("connect-response", f"to {cs}", f"{m}/{tot}", gateway=cs)
        if _is_mfsk_handshake(seg):
            return Result("connect-response", "(dest not in gateway list)", f"best {m}/{tot}")
        return _unnamed_burst(seg, dur, n, level_db)

    if 0.30 <= dur < 0.85:                             # short DBPSK control token
        tok, band = _best_token(seg)
        if tok:
            return Result(tok.name, "", f"BW{band} h{tok.hamming} q{tok.quality:.2f}")
        return _unnamed_burst(seg, dur, n, level_db)

    return _unnamed_burst(seg, dur, n, level_db)


def _cp_over(seg: np.ndarray, lvl: str) -> Result | None:
    """A const-law over nothing here decodes, named by its cyclic prefix alone."""
    useful, metric = RX.cp_repetition(seg)
    if metric < RX.CP_FLOOR:
        return None
    return Result("high-SL over", f"undecoded CP-OFDM, {useful}+{RX.CP_SAMPLES}-sample "
                  "symbol, record unknown", f"{lvl}, CP {metric:.2f}")


def _nsym(n: int) -> int:
    return round((n - MK.STRIDE) / MK.HOP) + 1


# The scanner owns these outright, so a bracket that happens to land on one is not
# reported twice — under a duration bin that would name it from a worse alignment.
SCANNED = {"CR", "connect-response"}
