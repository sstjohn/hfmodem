# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-2 receiver (audio -> symbols -> control signals / payload).

PACTOR-2 is a two-carrier differential-PSK waveform: two tones ~200 Hz apart at
100 Bd, each carrying its own DPSK stream (DBPSK at the low speed levels, DQPSK
higher). This is a different physical layer from PACTOR-3's 18-tone raster, but it
shares the upper layers with PACTOR-3 -- the same control-signal codewords
(`spec.CONTROL_SIGNALS`), the same convolutional-code + interleaver family, and
the same CRC-16/X-25 -- which is why the decoder reuses `shrike.coding`.

Status of what is proven here:
  * front end (carrier lock, per-tone differential symbols and modulation-order
    detection) -- checked on the published FEC recording against the reference
    vectors in `tests/shrike/test_p2rx.py`;
  * symbol period -- MEASURED per recording rather than assumed to be
    `fs / SYMBOL_RATE`, because a capture carries its own clock: both tones of
    the published FEC sample return 100.77 Bd on its 48 kHz grid, and that 0.77 %
    walks a 72-symbol frame nearly half a symbol end to end;
  * control-signal hunt -- reuses the shared distance-12 codeword set, and
    `control_signal_at` reads one at the instant a cycle grid predicts, which is
    the form a live link needs. What it reads is a HYPOTHESIS about how PACTOR-2
    keys those codewords -- see `pactor2.control_signal`;
  * the LONG CYCLE -- `burst_grid` and `decode_bursts` fit the 3.75 s grid and the
    320-pulse frame as of 2026-09-02. Data mode is what a mailbox listing arrives
    in, and until then every long-frame marker was filtered out at acquisition;
  * payload FEC -- DECODING REAL OFF-AIR AUDIO at speed level 3 as of 2026-08-02.
    `decode_bursts` returns 31 CRC-valid fields from the 32 bursts of
    `hb9ak_055246_c1500.wav`, and all 31 are byte-exact against the fields an
    independent decoder's CRC accepted for the same bursts, unioned over three
    of its runs -- 0 of 280 graded bits wrong, thirty-one times.
    `rf-corpus/regress` carries it as the `pos_p2_sl3_hb9ak` fixture.
    The two PUBLISHED recordings still do not decode: blind and anchored, the scan
    sits at chance both ways, and a plant demodulated no better than the FEC clip
    decodes outright -- pactor2.md §7.6 and §7.9 carry those trial counts.

The two virtual carriers EXCHANGE TONES every ARQ cycle, and take their channel
rank and their symbol clock with them (`lane_tones`). Reading every cycle in the
home arrangement decodes half a link: 14 of those 32 bursts, every one of them on
an even cycle. pactor2.md §7.11.

Demodulate the data field with `bin_phasors`, not by integrating one symbol of
baseband at the carrier. The transmitted pulse is shaped and spans some three
symbols, and the shipped kernel is close to matched to it: on real speed-level-3
material the boxcar returns 0.33-0.55 concentration under the 8th power and this
front end returns 0.90-0.94, the difference between a 3 dB frame and a 10 dB one.
pactor2.md §7.10.

Provenance for every constant is in docs/protocols/pactor/pactor2.md.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import spec, tablegen

FS_DEFAULT = spec.SAMPLE_RATE
SYMBOL_RATE = 100.0                     # Bd; one symbol = 10 ms (M.1798: 100 Bd)
TONE_SPACING_HZ = 200.0                 # measured 194-204 Hz off-air; nominal 200
NOMINAL_TONES_HZ = (1400.0, 1600.0)     # center 1500, +-100 Hz: what the transmitter
                                        # emits. Received signals arrive de-tuned, so
                                        # the demod re-measures (`measure_carriers`).

CS_BITS = spec.CS_BITS_PER_TONE         # 20


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    import wave
    w = wave.open(path)
    n, fs, ch = w.getnframes(), w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch)[:, 0].astype(float)
    return x / (np.abs(x).max() + 1e-9), fs


def measure_carriers(audio: np.ndarray, fs: int = FS_DEFAULT) -> tuple[float, float]:
    """Locate the two DPSK carriers (strongest, >150 Hz apart) by averaged PSD.

    A 100 Bd DPSK carrier has no line to peak-pick: it is a ~200 Hz-wide hump,
    and its tallest bin wanders with the data. Each peak is therefore only a
    bracket, and the returned frequency is the energy CENTROID of the hump around
    it. Measured on a synthesized pair whose true centres are 1400 and 1600:
    tallest-bin gives 1429.7 and 1599.6, a spacing of 169.9 where the emitted
    spacing is exactly 200; the centroid gives 1398.9 and 1600.9.
    """
    seg = audio[len(audio) // 4: 3 * len(audio) // 4]
    win = 8192
    acc = np.zeros(win // 2 + 1)
    for i in range(0, len(seg) - win, win // 4):
        acc += np.abs(np.fft.rfft(seg[i:i + win] * np.hanning(win))) ** 2
    f = np.fft.rfftfreq(win, 1 / fs)
    a = np.where((f > 500) & (f < 2400), acc, 0.0)
    peaks: list[float] = []
    for i in np.argsort(a)[::-1]:
        if all(abs(f[i] - p) > 150 for p in peaks):
            peaks.append(float(f[i]))
        if len(peaks) == 2:
            break
    out = []
    for p0 in peaks:
        m = (f > p0 - 75) & (f < p0 + 75)
        out.append(float((f[m] * a[m]).sum() / (a[m].sum() + 1e-30)))
    return tuple(sorted(out))                        # type: ignore[return-value]


def symbol_stream(audio: np.ndarray, fs: int, f0: float,
                  period: float, off: int) -> np.ndarray:
    """Integrate one symbol per `period` samples from `off`, at carrier `f0`.

    The period is a float because a recording's symbol clock is its own, not
    ours; the integration boundaries are rounded per symbol so the error never
    accumulates.
    """
    n = np.arange(audio.size)
    bb = audio * np.exp(-2j * np.pi * f0 * n / fs)
    count = int((bb.size - off) / period) - 1
    edges = off + np.round(np.arange(count + 1) * period).astype(int)
    return np.add.reduceat(bb, edges[:-1]) / np.diff(edges)


DPSK_ORDERS = (2, 4, 8, 16)
"""Alphabet sizes the four speed levels use: DBPSK, DQPSK, 8-DPSK, 16-DPSK."""


def _concentration(sym: np.ndarray) -> float:
    """How tightly the differential phases cluster, whatever the DPSK order.

    An M-ary differential concentrates under the Mth power and under no lower
    one, so this has to span every order the ladder uses. It tried only the 2nd
    and 4th, which is blind to speed levels 3 and 4 by construction: on a
    synthesized 8-DPSK carrier the old statistic reads 0.06 where this reads
    0.99, and `measure_symbol_period` on top of it returned a symbol period fitted
    to nothing at all.
    """
    d = sym[1:] * np.conj(sym[:-1])
    u = d / (np.abs(d) + 1e-12)
    return max(float(abs(np.mean(u ** m))) for m in DPSK_ORDERS)


def measure_symbol_period(audio: np.ndarray, fs: int = FS_DEFAULT,
                          f0: float = NOMINAL_TONES_HZ[0]) -> tuple[float, int]:
    """Samples per symbol and the symbol phase, measured rather than assumed.

    A recording carries its capture chain's clock, not the transmitter's, and
    `fs / SYMBOL_RATE` is only the nominal. On the published FEC sample both
    tones independently return 476.3 samples -- 100.77 Bd on that file's 48 kHz
    grid -- where the nominal 480 scores 0.45 against 0.73 on the concentration
    below. Over a 72-symbol frame that 0.77 % walks the integrator nearly half a
    symbol from one end to the other, which is enough on its own to stop any
    frame decoding, and it cannot be recovered by a per-symbol phase search
    because it is a rate error and not an offset.

    Call it per tone. The phase is NOT shared between the carriers: on that same
    sample the two tones want 60 and 324 of 476 samples, 0.554 of a symbol apart
    against half a symbol at 238, with sharply peaked phase curves either side.
    Integrating both carriers on one grid puts one of them across its own symbol
    boundaries for the whole frame. See pactor2.md §1.4.

    Returns (period_samples, phase_samples).
    """
    span = 2.0                                   # per cent either side of nominal
    nominal = fs / SYMBOL_RATE
    best = (0.0, nominal, 0)
    for coarse in (2.0, 0.2):
        lo, hi = best[1] * (1 - span / 100), best[1] * (1 + span / 100)
        for period in np.arange(lo, hi, coarse * nominal / 1000):
            for off in range(0, int(period), max(1, int(period) // 16)):
                sym = symbol_stream(audio, fs, f0, period, off)
                if sym.size < 64:
                    continue
                v = _concentration(sym)
                if v > best[0]:
                    best = (v, float(period), off)
        span = 0.1
    return best[1], best[2]


def _tone_symbols(audio, fs, f0, *, fine=True):
    """Carrier-lock tone `f0` and return per-symbol complex samples + timing offset.

    The symbol period comes from `measure_symbol_period` on this tone, not from
    `fs / SYMBOL_RATE`; a recording's clock is its own and the difference walks a
    frame apart (see that function).

    The carrier search resolves integration loss only. A constant frequency error
    turns every differential by the same angle, and the concentration score is
    `|mean(u**M)|`, which that common turn leaves unchanged -- so the score cannot
    see a carrier offset at all, and the residual rotation has to come off the
    constellation later rather than out of `flock` here.

    Returns (symbols, locked_freq, offset)."""
    period, _ = measure_symbol_period(audio, fs, f0)
    step = max(1, int(period) // 16)
    grid = np.arange(-30, 30.01, 0.25) if fine else np.array([0.0])
    best = None
    for df in grid:
        for off in range(0, int(period), step):
            sym = symbol_stream(audio, fs, f0 + df, period, off)
            d = sym[1:] * np.conj(sym[:-1])
            amp = np.minimum(np.abs(sym[1:]), np.abs(sym[:-1]))
            keep = amp > np.percentile(amp, 55)
            dd = d[keep]
            if len(dd) < 32:
                continue
            u = dd / np.abs(dd)
            score = max(abs(np.mean(u ** 2)), abs(np.mean(u ** 4)))
            if best is None or score > best[0]:
                best = (score, sym, f0 + df, off)
    _, sym, flock, off = best
    return sym, flock, off


def modulation_order(symbols: np.ndarray) -> tuple[str, float, float]:
    """DBPSK vs DQPSK from the Mth-power concentration of the differential phase."""
    d = symbols[1:] * np.conj(symbols[:-1])
    amp = np.minimum(np.abs(symbols[1:]), np.abs(symbols[:-1]))
    d = d[amp > np.percentile(amp, 55)]
    u = d / (np.abs(d) + 1e-12)
    c2, c4 = abs(np.mean(u ** 2)), abs(np.mean(u ** 4))
    return ("DQPSK" if c4 > c2 else "DBPSK"), float(c2), float(c4)


def dbpsk_bits(symbols: np.ndarray) -> np.ndarray:
    """Differential BPSK hard bits: bit = 1 when the phase reverses (~pi)."""
    d = symbols[1:] * np.conj(symbols[:-1])
    return (np.cos(np.angle(d)) < 0).astype(np.uint8)


MARKER_STAGGER = 0.5
"""Symbols the lower carrier leads the upper by inside a frame marker."""

ACQ_RATE = 800.0
"""Analysis rate of the acquisition front end: 25 Hz bins on a 32-point transform,
and one symbol at 100 Bd is exactly eight frames."""

ACQ_FS = 9600
"""Input rate of the acquisition front end -- a receiver's internal rate."""

ACQ_DECIM = 12
"""9600 Hz down to the 800 Hz analysis rate."""

ACQ_LO_HZ = 1500.0
"""Mixing frequency: the signal centre."""

FRAMES_PER_SYMBOL = 8
"""Analysis frames in one symbol: 800 Hz against 100 Bd."""

LANE_LEAD_FRAMES = 4
"""Frames by which one virtual carrier leads the other -- the T/2 stagger of s1.4.

The lead belongs to the carrier and not to the tone, which is what `lane_tones`
is about."""


def bin_phasors(audio: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    """Audio -> `(frames, 32)` complex phasor per 25 Hz bin, at 800 Hz.

    Mix to baseband at 1500 Hz, a 96-tap lowpass, decimate to 800 Hz, then a
    32-point transform through the symbol pulse as its window, once per decimated
    sample. Bin `m` is `1500 + 25*m` Hz folded about the transform, so the two
    carriers of a frame land eight bins apart.

    **The window is not a taper you can substitute, and it is not the symbol
    pulse either.** 800 Hz is the pulse's own 8 samples per symbol, which makes
    reuse tempting and wrong: `tablegen.acq_window` is a wider kernel, and
    swapping the symbol pulse in here still scores 0.997 on an off-air marker
    while costing this path every dB of its single-copy margin. Off
    `PACTOR-II_FEC.wav` the window recovers all three of an independent decoder's
    markers at 0.997-0.998 where a Hann of the same support finds none; s5.1
    measured a symbol-length boxcar at the carrier -- the textbook DPSK detector
    -- recovering 0.36 of this correlator's output.

    That 0.36 is about the DATA as much as the header. Demodulating the
    HB9AK captures' 8-DPSK field with a boxcar integrator, on a clock and a grid
    phase fitted per window, concentrates at 0.33-0.55 under the 8th power; the
    same field through this kernel concentrates at 0.90-0.94, which is what a
    plant at 10 dB reads. The whole difference is the pulse shape.
    """
    fs_i, decim = ACQ_FS, ACQ_DECIM
    window = tablegen.acq_window()
    nfft = window.size
    x = np.asarray(audio, dtype=float)
    if fs != fs_i:
        # Whole ratios decimate; interpolating instead costs a third of the
        # correlator's output on real audio, because linear interpolation is a
        # poor anti-alias filter and the codeword lives in the phase.
        if fs % fs_i == 0:
            x = x[::fs // fs_i]
        else:
            m = int(round(len(x) * fs_i / fs))
            x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x)
    n = np.arange(len(x))
    bb = x * np.exp(-2j * np.pi * ACQ_LO_HZ * n / fs_i)
    dec = np.convolve(bb, tablegen.acq_lowpass())[:len(x)][decim - 1::decim]
    if len(dec) < nfft:
        return np.zeros((0, nfft), dtype=complex)
    blocks = np.lib.stride_tricks.sliding_window_view(dec, nfft)[:, ::-1]
    # ifft, not fft: block index 0 is the NEWEST sample, so the positive exponent
    # is what puts bin m at 1500 + 25m Hz rather than mirroring the band.
    return np.fft.ifft(blocks * window, axis=1)


def bin_differentials(audio: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    """Audio -> `(frames, 32)` one-symbol differential phase per 25 Hz bin.

    `bin_phasors` against itself one symbol -- eight frames -- earlier. This is
    what the frame-marker correlator scores its codewords on.
    """
    z = bin_phasors(audio, fs)
    diff = FRAMES_PER_SYMBOL
    if len(z) < diff + 1:
        return np.zeros((0, z.shape[1] if z.size else 32))
    ph = np.angle(z)
    return (ph[diff:] - ph[:-diff] + np.pi) % (2 * np.pi) - np.pi


def carrier_bins(bin_pair: int) -> tuple[int, int]:
    """Frame-descriptor bin pair -> transform bins of the (lower, upper) carrier.

    `find_markers` reports the PAIR: the frame's centre is `1200 + 25*b` Hz and
    its carriers sit 100 Hz either side of it, eight bins apart."""
    return (bin_pair - 16) % 32, (bin_pair - 8) % 32


def lane_tones(bin_pair: int, swapped: bool) -> tuple[int, int]:
    """Transform bins the two virtual carriers occupy, in CHANNEL-RANK order.

    Rank 0 is `channel_buffer`'s first lane and is read on the late clock; rank 1
    leads it by `LANE_LEAD_FRAMES`. In the HOME arrangement rank 0 is the upper
    tone (pactor2.md s5.12) and so the lower tone is the one that leads.

    Every ARQ cycle the two virtual carriers exchange tones, taking their rank --
    and therefore their clock -- with them, so on a swapped cycle rank 0 is the
    LOWER tone and it is the UPPER that leads. PUBLISHED, and for PACTOR-2 rather
    than by analogy: [SCS] s5, "In the PACTOR-2 system, the transferred
    information is swapped from one channel (tone) to the other in every cycle",
    which it gives as the reason narrow-band QRM on one tone costs the link speed
    instead of the link. `spec.CARRIER_SWAP` is the same involution on PACTOR-3's
    eighteen carriers, and the same reason `spec.SUBBAND_LEAD` is indexed by rank
    rather than by channel. What is measured here is that the swap takes the RANK
    and the STAGGER with it, which the sentence does not say.

    Both halves are one fact and neither works alone. On the 32 bursts of
    `hb9ak_055246_c1500.wav`: home arrangement 14 fields, stagger reversed by
    itself 0, lanes swapped by themselves 1, both together the other 15.
    """
    lo_bin, hi_bin = carrier_bins(bin_pair)
    return (lo_bin, hi_bin) if swapped else (hi_bin, lo_bin)


def find_markers(audio: np.ndarray, fs: int = FS_DEFAULT, threshold: float = 0.80
                 ) -> list[tuple[float, int, int, int, float, bool]]:
    """Correlate for PACTOR-2 frame markers.

    Returns `(seconds, bin, k, sense, score, swapped)`, one entry per detected
    marker, `seconds` being where the marker ends.

    The marker is eight chips of a complex codeword carried on the DIFFERENTIAL
    phase of each carrier, one of them leading by half a symbol (`lane_tones`,
    `pactor2.frame_marker`). Scoring it is the matched filter for exactly that:
    sum `e^{i d_t} * c[t]` over the eight one-symbol differentials of each carrier
    against each of the sixteen codewords, add the two carriers, and normalise by
    the sixteen-tap ceiling. `score` is 1 for a noiseless marker; a conforming
    receiver arms at 0.94 of it, which is where `threshold` sits by default less a
    margin.

    `bin` names the two-carrier PAIR, not a tone: it scores transform bins
    `b - 16` and `b - 8`, which are 200 Hz apart, so the frame's centre frequency
    is `1200 + 25*b` Hz and its carriers sit 100 Hz either side. `sense` is +1 or
    -1 for the codeword and its conjugate, and both occur on air.

    BOTH ARRANGEMENTS are scored, and `swapped` reports which one the marker is
    in. A marker rides the same two virtual carriers the data field does, so the
    carrier swap moves the codebooks and the stagger together and a correlator
    pinned to the home arrangement is deaf on every other ARQ cycle -- which is
    what put the markers of these recordings on a 2.5 s spacing rather than the
    1.25 s the link actually runs at. It separates cleanly: across the 32 anchors
    of `hb9ak_055246_c1500.wav` the arrangement a burst is in scores 0.966-0.999
    and the other one 0.52-0.66, on all 30 that read above the arming threshold
    either way.

    This is what gives receive an anchor. Without it the demodulator has only a
    blind scan over every symbol offset, and the frame's speed level and length
    have to be guessed rather than read: `k` IS the descriptor's parameter byte,
    with `sl = (k >> 1) & 3` and `long = (k >> 3) & 1`.

    THE CODEWORD LOOP IS A MATRIX PRODUCT, and it is written as one because this
    is now a per-cycle cost rather than an analysis run: a station holding a link
    acquires once per 1.25 s, on its own audio, inside the slot it is transmitting
    in. Each chip's differentials are a contiguous slice of one bin, so the eight
    taps stack once per lane and all sixteen codewords come off that stack in one
    product instead of the stack being re-gathered sixteen times. Measured on a
    1.30 s window, 98.8 ms becomes 14.3 ms; on the 40.7 s of
    `hb9ak_055246_c1500.wav`, 3.36 s becomes 0.55 s, and the hits agree to 3.3e-16
    in the score with every discrete field identical.
    """
    from . import pactor2

    codes = pactor2.marker_codewords()
    d = bin_differentials(audio, fs)
    if len(d) == 0:
        return []
    u = np.exp(1j * d)
    n_chips = pactor2.MARKER_CHIPS
    lead = int(round(MARKER_STAGGER * ACQ_RATE / SYMBOL_RATE))
    span = 8 * (n_chips - 1) + lead
    idx = np.arange(span, len(u))
    ceiling = 2 * n_chips * np.sqrt(2.0)

    def taps(shift: int, b: int) -> np.ndarray:
        """The eight chip differentials of bin `b`, `(positions, chips)`."""
        return np.stack([u[span - shift - 8 * t: len(u) - shift - 8 * t, b]
                         for t in range(n_chips)], axis=1)

    hits: list[tuple[float, int, int, int, float, bool]] = []
    for b in range(24):
        for swapped in (False, True):
            # codebook 0 is the leading carrier's, which is rank 1
            late_bin, lead_bin = lane_tones(b, swapped)
            lead_taps, late_taps = taps(lead, lead_bin), taps(0, late_bin)
            for sense in (1, -1):
                a_lead = lead_taps if sense > 0 else np.conj(lead_taps)
                a_late = late_taps if sense > 0 else np.conj(late_taps)
                s = np.abs(a_lead @ codes[0].T + a_late @ codes[1].T) / ceiling
                for p, k in zip(*np.nonzero(s > threshold)):
                    hits.append((float((idx[p] + 8) / ACQ_RATE), b, int(k), sense,
                                 float(s[p, k]), swapped))
    hits.sort(key=lambda h: -h[4])
    kept: list[tuple[float, int, int, int, float, bool]] = []
    for h in hits:
        if all(abs(h[0] - g[0]) > 4 / SYMBOL_RATE for g in kept):
            kept.append(h)
    return sorted(kept)


CYCLE_S = 1.25
"""Seconds between data bursts on the short cycle."""

CYCLE_LONG_S = 3.75
"""Seconds between data bursts in DATA MODE, where the frame is 320 pulses.

[SCS] s2: an ISS whose buffer holds more than a standard packet carries "sets the
long cycle flag in the status word", the IRS accepts with CS6, "the length of
these data packets is 3.28 seconds, which leads to an entire cycle duration of
3.75 seconds in this so-called data mode" -- the mode a mailbox listing arrives
in. `spec.CYCLE_LONG_S` is the same number for PACTOR-3, which runs the same two
cycle lengths."""

MARKER_ARM = 0.94
"""Score a marker must reach before the field behind it is demodulated.

`find_markers` is asked for a margin below this and the accept applied here, so
that the correlator sees the near misses its own de-duplication needs to keep the
right hit of an overlapping pair."""


def burst_grid(audio: np.ndarray, fs: int = FS_DEFAULT, level: int = 2,
               long_frame: bool = False) -> tuple[list[float], int, list[bool]]:
    """Data-burst start times, carrier bin pair and arrangement, from acquisition.

    The markers land on an exact cycle grid -- on 40 s of real off-air material
    their common residue holds to better than 5e-5 -- so one residue fixes every
    burst, including the ones whose own marker scores too low to be offered as an
    anchor. Returns ([], -1, []) when nothing acquires.

    `long_frame` is data mode: 320-pulse frames on a `CYCLE_LONG_S` grid rather
    than 72-pulse ones on `CYCLE_S`. The marker carries the flag as bit 3 of its
    codeword index, so the two are separated at acquisition and a grid is fitted
    over one of them -- a recording that changes cycle length mid-link is two
    grids, and asking for the one that is not there returns nothing rather than a
    grid of the wrong pitch.

    The arrangement is carried the same way. It alternates strictly with the
    cycle, so it is one bit for the whole recording rather than a reading per
    burst: every acquired marker votes on whether grid index 0 is at home, and the
    rest follows from the index's parity. On `hb9ak_055246_c1500.wav` that vote is
    unanimous over 30 markers, and it covers the two anchors that faded below the
    arming threshold on both arrangements.
    """
    cycle = CYCLE_LONG_S if long_frame else CYCLE_S
    marks = [(t, b, sw) for t, b, k, _s, sc, sw in find_markers(audio, fs, 0.90)
             if sc >= MARKER_ARM and (k >> 1) & 3 == level
             and bool((k >> 3) & 1) == long_frame]
    if not marks:
        return [], -1, []
    origin = float(np.median([t % cycle for t, _, _ in marks]))
    bins = [b for _, b, _ in marks]
    n = int((audio.size / fs - origin) / cycle)
    votes = [sw ^ (int(round((t - origin) / cycle)) & 1) for t, _, sw in marks]
    first = 2 * sum(votes) > len(votes)
    return ([origin + cycle * k for k in range(n)],
            max(set(bins), key=bins.count),
            [first ^ bool(k & 1) for k in range(n)])


def burst_window(phasors: np.ndarray, bin_pair: int, t: float, n_symbols: int,
                 order: int = 8, *, swapped: bool = False, slack: int = 0
                 ) -> list[np.ndarray] | None:
    """One burst's two differential windows, in channel-rank order.

    The comb phase within the symbol is READ OFF THE ANCHOR, not searched: both
    carriers ride one transmitter clock, so the anchor's frame index fixes the
    phase of each lane's comb (the T/2 stagger apart, per `lane_tones`), and on
    every confidently-decoded burst of `hb9ak_055246_c1500.wav` that is exactly
    the phase the old per-lane concentration search converged to. The search is
    what failed on the bursts that mattered: concentration is a fine judge of a
    strong lane and a misleading one of a faded lane -- at 33.15 s it preferred
    a comb three frames off the clock on the weak carrier, a 3/8-symbol timing
    error that read as unexplained phase noise, and the burst was lost. `slack`
    admits +-that many frames of anchor error for callers whose anchor is a
    single marker rather than a whole recording's grid, chosen by the Mth-power
    concentration as before; it never sees a CRC either way.

    The returned phasors are NOT unit: each symbol carries its differential
    magnitude, normalised to the burst's median, and `pactor2.soft_bits` reads
    that magnitude as confidence. On this waveform selective fades take one
    carrier at a time, and the whole difference between a lost burst and a
    byte-exact one can be whether the faded stretch is allowed to shout.

    `swapped` is which tone each rank is on for this ARQ cycle, and the T/2
    stagger of s1.4 goes with the rank -- `lane_tones`.
    """
    f0 = int(round(t * ACQ_RATE))
    step = FRAMES_PER_SYMBOL
    lanes = []
    for b, lead in zip(lane_tones(bin_pair, swapped), (0, LANE_LEAD_FRAMES)):
        best = (-1.0, None, None)
        pin = (f0 - lead) % step
        for dp in range(-slack, slack + 1):
            p = (pin + dp) % step
            s = phasors[p::step, b]
            d = s[1:] * np.conj(s[:-1])
            u = d / (np.abs(d) + 1e-12)
            i0 = int(round((f0 - lead - p) / step))
            if i0 < 0 or i0 + n_symbols > u.size:
                continue
            w = u[i0:i0 + n_symbols]
            c = np.mean(w ** order)
            if abs(c) > best[0]:
                best = (float(abs(c)), w * np.exp(-1j * np.angle(c) / order),
                        np.abs(d[i0:i0 + n_symbols]))
        if best[1] is None:
            return None
        lanes.append(best[1:])
    med = np.median(np.concatenate([a for _, a in lanes]))
    return [w * a / (med + 1e-30) for w, a in lanes]


def decode_burst(wins: list[np.ndarray], path) -> bytes | None:
    """A burst's two windows -> its CRC-valid field, or None.

    The differential is read in the CONJUGATE sense, which is the sense
    `pactor2.soft_bits` labels the constellation in."""
    from . import coding, pactor2

    src = pactor2.channel_of_code(path)
    m = 1 << path.bits_per_cell
    # Each carrier's constellation phase belongs to the channel, not to the
    # protocol -- the two arrive independently -- so it is the one thing the
    # decoder fits per frame, in half-sector steps of THIS level's constellation:
    # the sweep exp(2*pi*j*r/(2m)) has period 2m. A constant 16 here was right
    # only for 8-DPSK -- DBPSK and DQPSK repeated themselves 4x and 2x per lane,
    # and 16-DPSK reached only half its circle, leaving half the possible channel
    # rotations undecodable at SL4.
    rotations = 2 * m
    for r0 in range(rotations):
        for r1 in range(rotations):
            lanes = [pactor2.soft_bits(
                np.conj(w) * np.exp(2j * np.pi * r / (2 * m)), path.bits_per_cell)
                for w, r in zip(wins, (r0, r1))]
            field = pactor2.decode_frame(
                pactor2.channel_buffer(lanes, 0, path)[src], path)
            if coding.crc16(field[:-2]) == int.from_bytes(field[-2:], "little"):
                return field
    return None


MEMORY_COPIES = 4
"""Burst copies `BurstMemory` holds at most, oldest dropped first.

A bound, not a measurement. It caps what combining offers the CRC -- one
combined decode per failed burst, over at most this many copies -- and it is
how a mis-grouped copy leaves: a field boundary the memory missed poisons the
sum, and the sliding window ages the stale copy out within this many cycles
instead of carrying it for the rest of the session. A fade that outlasts four
cycles has the link changing speed level anyway, which reframes the field and
resets the memory through `add`'s path check."""


class BurstMemory:
    """Soft combining across repeats of one unacknowledged field -- memory ARQ.

    A field the peer holds no acknowledgement for is transmitted AGAIN, the same
    bytes under the same mod-4 counter (`spec.STATUS_SEQ`, read at
    `field[crc_bytes - 3]`), until it is acked. A receiver that decodes each
    copy alone throws the earlier copies away; summing their soft values first
    is what recovers a field whose every individual copy is marginal, and it is
    a standard fixture of commercial PACTOR hardware.

    What makes the sum meaningful:

      * copies are grouped by CONSECUTIVE FAILURE, not by reading the counter --
        an undecoded burst's counter is exactly what cannot be read. Any burst
        that decodes single-shot clears the memory (`clear`), so the copies here
        are the bursts between decodes, which under ARQ repetition are copies of
        one field whenever the grouping is right. When it is wrong -- the real
        receiver acked a burst this one lost, and the accumulator straddles a
        field boundary -- the sum is two codewords' softs superposed, the
        trellis decodes neither and the CRC refuses it: the failure is a burst
        reported undecoded, the same as with no memory at all, never a wrong
        field delivered. `MEMORY_COPIES` then ages the stale copies out.
      * each copy arrives through `burst_window` in CHANNEL-RANK order, so the
        carrier swap (`lane_tones`) is already unwound: copies from alternate
        ARQ cycles ride opposite tones but sum lane for lane.
      * the waveform is differential, so a constant carrier offset turns every
        one-symbol differential of a copy by one constant angle. `add` measures
        that angle per lane -- the phase of the copy's correlation against the
        accumulated sum, over the field's symbols -- and takes it off before
        summing. No per-copy carrier phase alignment exists to need.

    Trials the CRC is offered: one `decode_burst` scan -- `(2m)^2` alignments --
    per `add` that holds two copies or more. A lone failure costs nothing: its
    only copy was already scanned single-shot, and scanning the same softs again
    would be a pure doubling of exposure for an answer already known.
    """

    def __init__(self) -> None:
        self._path = None
        self._copies: list[list[np.ndarray]] = []

    def clear(self) -> None:
        self._copies.clear()

    def add(self, wins: list[np.ndarray], path) -> bytes | None:
        """Absorb one undecoded burst's windows; return the combined field, if any.

        A CRC-valid combined decode clears the memory -- the field is delivered
        and whatever follows is either its next repeat or its successor, both of
        which stand alone again."""
        if path != self._path:
            self._path = path
            self._copies.clear()
        self._copies.append(self._aligned(wins))
        del self._copies[:-MEMORY_COPIES]
        if len(self._copies) < 2:
            return None
        combined = [np.sum([c[lane] for c in self._copies], axis=0)
                    for lane in range(len(wins))]
        field = decode_burst(combined, path)
        if field is not None:
            self.clear()
        return field

    def _aligned(self, wins: list[np.ndarray]) -> list[np.ndarray]:
        """`wins` de-rotated per lane onto the accumulated copies' constellation.

        The correlation phase against the running sum IS the inter-copy
        rotation when the copies carry the same field; on a stranger it is
        noise, the sum part-cancels, and the CRC above stays shut."""
        if not self._copies:
            return [np.asarray(w) for w in wins]
        out = []
        for lane, w in enumerate(wins):
            ref = np.sum([c[lane] for c in self._copies], axis=0)
            out.append(w * np.exp(-1j * np.angle(np.vdot(ref, w))))
        return out


def decode_bursts(audio: np.ndarray, fs: int = FS_DEFAULT, level: int = 2,
                  long_frame: bool = False) -> list[tuple[float, bytes]]:
    """Every PACTOR-2 data burst a recording carries, as CRC-valid fields.

    This is the whole receive path: acquire, put the burst grid and the carrier
    arrangement on the markers, demodulate on the shipped kernel, and decode. On
    `hb9ak_055246_c1500.wav` it returns 31 fields from 32 bursts, all 31
    byte-exact against the fields an independent decoder's CRC accepted for the
    same bursts across three of its runs.
    The one that stays lost, at 38.15 s, is a selective fade that sweeps the
    pair mid-field and takes each carrier in turn; the reference decoder fails
    it too. pactor2.md s7.11 and s7.13.

    Which arrangement a burst is in comes from ACQUISITION, not from a second
    trial through the CRC. Trying both would be the simpler code and this is not a
    hot path, but it would double the alignments the scan offers the CRC, and the
    scan is charged per level: `decode_burst` tries (2m)^2 rotation pairs per
    scan -- 16 at SL1, 64 at SL2, 256 at SL3, 1,024 at SL4. Every burst gets one
    single-shot scan, and `BurstMemory` adds one combined scan per failed burst
    that follows another failure, so a recording of B bursts with F such failures
    offers (B + F) * (2m)^2 alignments -- at most double the memoryless figure,
    and only when every burst but the first fails. This recording, 32 SL3 bursts
    with its one failure isolated between decodes, offers exactly the memoryless
    8,192 alignments and 0.125 expected false accepts at CRC-16's 2^-16. Both
    arrangements would double that to buy nothing: the
    marker rides the same two virtual carriers the data field does,
    it separates the arrangements 0.97-1.00 against 0.52-0.66 (`find_markers`),
    and the arrangement it names is the arrangement that decodes at all 29
    anchors that decode at all. Neither decodes at the other three. That is also
    how `p3rx.header_of` reads PACTOR-3's swap: off the header block, against a
    fit that never sees a CRC.
    """
    from . import pactor2

    grid, bin_pair, arms = burst_grid(audio, fs, level, long_frame)
    if not grid:
        return []
    phasors = bin_phasors(audio, fs)
    path = (pactor2.PATHS_LONG if long_frame else pactor2.PATHS)[level]
    memory = BurstMemory()
    out = []
    for t, swapped in zip(grid, arms):
        wins = burst_window(phasors, bin_pair, t, path.n_symbols,
                            1 << path.bits_per_cell, swapped=swapped)
        if wins is None:
            continue
        field = decode_burst(wins, path)
        if field is not None:
            memory.clear()
        else:
            field = memory.add(wins, path)
        if field is not None:
            out.append((t, field))
    return out


def decode_expected_burst(audio: np.ndarray, fs: int = FS_DEFAULT,
                          memory: BurstMemory | None = None) -> tuple | None:
    """The burst a single ARQ cycle carries: `(seconds, path, field, bin_pair)`.

    `path` is the `pactor2.Path` the marker named, unannotated here only because
    `pactor2` imports this module and the arrow cannot name it. `bin_pair` is the
    carrier pair the marker armed on -- the frame's centre is `1200 + 25*b` Hz --
    and it is returned because a station holding the link needs it for the OTHER
    read of the same cycle: `control_signal_at` takes a bin pair and has nothing
    but `CENTRE_BIN_PAIR` to guess with, and a peer 50 Hz off centre would have
    its data read and its codeword missed.

    `decode_bursts` reads a RECORDING and this reads a CYCLE, and the whole
    difference is where the burst's placement comes from. A recording holds many
    bursts on one residue, so `burst_grid` fits that residue across all of them
    and thereby covers the anchors whose own marker faded below the arming
    threshold; a cycle holds ONE, there is no residue to fit, and the marker it
    arrived with is the whole of what places it. The speed level and the frame
    length come off that same marker, so this READS the geometry rather than
    being told it -- which is what lets a station follow a peer onto a ladder it
    is not itself on.

    THE BEST-SCORING MARKER AND NO OTHER, and that bound is what makes this
    affordable in a cycle. Acquisition is 14.5 ms on a 1.30 s window, averaged
    over 862 such windows of real off-air audio; the rotation search behind it is
    25.5 ms when the field decodes and 334 ms when it does not, because a miss
    runs every alignment -- (2m)^2 of them, 256 at the SL3 those figures were
    measured on, 1,024 at SL4. Trying every armed marker would multiply that miss
    by whatever the channel offered. One cycle carries one transmission from the
    peer, so the strongest marker is the one that is there -- across those 862
    windows exactly one armed on audio that is not PACTOR-2, and it decoded
    nothing.

    `memory` is the session's `BurstMemory`, one per link, held by the caller
    because a cycle cannot remember anything: a station on the receiving side of
    ARQ KNOWS the peer repeats an unacked field, so the copies it failed to
    read singly are exactly what the memory sums. A miss with a memory costs a
    second rotation scan on the combined softs -- the same 334 ms ceiling
    again -- which still fits the 1.25 s cycle beside acquisition."""
    from . import pactor2

    marks = [m for m in find_markers(audio, fs, 0.90) if m[4] >= MARKER_ARM]
    if not marks:
        return None
    t, bin_pair, k, _sense, _score, swapped = max(marks, key=lambda m: m[4])
    paths = pactor2.PATHS_LONG if (k >> 3) & 1 else pactor2.PATHS
    path = paths[(k >> 1) & 3]
    # slack 1: the marker rides the audio, so anchor error reaches the comb only
    # as frame quantisation. More freedom is worse, not safer -- at 2 the faded
    # lane's concentration walks off the clock again and 33.15 s is lost.
    wins = burst_window(bin_phasors(audio, fs), bin_pair, t, path.n_symbols,
                        1 << path.bits_per_cell, swapped=swapped, slack=1)
    if wins is None:
        return None                     # the window does not hold the whole field
    field = decode_burst(wins, path)
    if field is not None:
        if memory is not None:
            memory.clear()
    elif memory is not None:
        field = memory.add(wins, path)
    return None if field is None else (t, path, field, bin_pair)


@lru_cache
def cs_table() -> np.ndarray:
    """The six codewords as bits, LEAST SIGNIFICANT BIT OF THE WORD FIRST.

    One orientation for the whole modem: `rx._cs_bits` reads `spec.CONTROL_SIGNALS`
    this way for PACTOR-3 and `placement.control_signal` keys them this way, so a
    PACTOR-2 renderer and reader that read them the other way round would agree
    with each other and with nothing else. Reversing the words preserves every
    pairwise distance -- the set stays equidistant either way -- so nothing about
    the table looks wrong when the orientation is; only a link does.

    This module transcribed them most-significant-first until 2026-09-02, which
    reached nothing but `analyze`."""
    return np.array([[(w >> i) & 1 for i in range(CS_BITS)]
                     for w in spec.CONTROL_SIGNALS], np.uint8)


CS_NAMES = ("CS1/ACK", "CS2/req", "CS3/break", "CS4/speedup", "CS5/NAK", "CS6/cyc")

CS_SYMBOLS = CS_BITS + 1
"""Pulses a control signal occupies: Figure 1's 20, and the reference pulse."""

CENTRE_BIN_PAIR = 12
"""The bin pair of a link centred at 1500 Hz -- `carrier_bins` reads 1400/1600.

An answer arrives with no marker in front of it, so nothing in the burst names
the pair the way a data frame does. A station holding a link has it from the
frames it is already reading; this is what a caller that has not got one uses."""

CS_MAX_ERRORS = 0
"""Bit errors of twenty a codeword may carry and still be reported.

Zero, which is `rxfront.CS_MAX_ERRORS` and the same rule PACTOR-1 and PACTOR-3
are read under. The set is distance 12 and five errors are correctable, so
spending that radius looks free and is not: the six words inside radius 4 cover
3.5 per cent of the 20-bit space, and at radius 4 this reader accepted a codeword
in 49 of 400 windows of noise. At radius 0 it accepted none, and every clean
codeword still reads -- the accept costs the link nothing and the radius costs it
the difference between an answer and a hallucination."""

CS_SLACK_FRAMES = 2
"""Analysis frames either side of the predicted instant that are also read.

2.5 ms at 800 Hz. The anchor is the caller's cycle grid rather than a detection,
and a quarter-symbol of grid error is what it is worth defending against; wider
buys nothing and costs false accepts, which are charged per alignment."""


def cs_softs(phasors: np.ndarray, bin_pair: int, frame: int,
             swapped: bool = False) -> np.ndarray | None:
    """Summed DBPSK soft bits of the codeword whose reference pulse is at `frame`.

    Both carriers carry the SAME twenty bits -- [SCS] s2, "All CS are always sent
    in DBPSK in order to obtain a maximum of robustness" -- so they are summed
    before anything is sliced, which is what makes the 40-bit block a distance-24
    code rather than two distance-12 ones. `lane_tones` supplies the arrangement
    and the T/2 stagger, exactly as it does for a data field: the codeword rides
    the same two virtual carriers.

    Positive is a zero bit, `pactor2.soft_bits`' convention, read in the conjugate
    sense `decode_burst` reads a field in."""
    from . import pactor2

    acc = np.zeros(CS_BITS)
    for b, lead in zip(lane_tones(bin_pair, swapped), (0, LANE_LEAD_FRAMES)):
        idx = frame - lead + np.arange(CS_SYMBOLS) * FRAMES_PER_SYMBOL
        if idx[0] < 0 or idx[-1] >= len(phasors):
            return None
        y = phasors[idx, b]
        acc += pactor2.soft_bits(np.conj(y[1:] * np.conj(y[:-1])), 1)[:, 0]
    return acc


CS_FRAME_LAG = -12
"""Analysis frames between a pulse's own sample index and the frame that reads it.

`bin_phasors` scores a 32-frame ring against a window that peaks thirteen frames
back from the block's newest sample, so a pulse is read by a frame ahead of its
own index rather than at it. MEASURED rather than counted off that window: all
six codewords rendered into silence at a known instant, in both arrangements,
read at ZERO bit errors over offsets -14 to -9 and at four or more outside, so
this is the centre of a six-frame plateau and `CS_SLACK_FRAMES` stays inside
it."""


def control_signal_at(audio: np.ndarray, at: int, fs: int = FS_DEFAULT, *,
                      bin_pair: int = CENTRE_BIN_PAIR, swapped: bool = False,
                      max_errors: int = CS_MAX_ERRORS
                      ) -> tuple[int, int] | None:
    """The codeword whose reference pulse the caller predicts at sample `at`.

    Returns `(index into spec.CONTROL_SIGNALS, bit errors)`, or None when nothing
    inside `max_errors` is there. The PACTOR-2 counterpart of
    `rxfront.SyncedRx.control_signal_at` and of `p1rx.cs_anchored`, and it exists
    for their reason: a station holding a cycle grid knows where the turnaround is
    from the grid, and reading twenty bits at a point is the only decode a keyed
    cycle can afford. `pactor2.cs_slot` is the offset from its own packet.

    ANCHORED, NOT HUNTED. `hunt_control_signals` slides the codewords over a whole
    bitstream and is an analysis tool; sliding a distance-12 code over noise
    manufactures accepts in proportion to the alignments offered, and a cycle
    offers one instant.
    """
    phasors = bin_phasors(audio, fs)
    frame = int(round(at / fs * ACQ_RATE)) + CS_FRAME_LAG
    table = cs_table()
    best: tuple[int, int] | None = None
    for d in range(-CS_SLACK_FRAMES, CS_SLACK_FRAMES + 1):
        soft = cs_softs(phasors, bin_pair, frame + d, swapped)
        if soft is None:
            continue
        bits = (soft < 0).astype(np.uint8)
        dist = (table != bits).sum(axis=1)
        j = int(np.argmin(dist))
        if dist[j] <= max_errors and (best is None or dist[j] < best[1]):
            best = (j, int(dist[j]))
    return best


def hunt_control_signals(bitstream: np.ndarray, max_dist: int = 3):
    """Slide the six 20-bit CS codewords (and their bit-inverses, since the DBPSK
    reference polarity is a free choice) across a hard bitstream. Returns hits as
    (offset, cs_index, distance, inverted)."""
    table = cs_table()
    hits = []
    b = bitstream.astype(np.int16)
    for off in range(0, len(b) - CS_BITS):
        window = b[off:off + CS_BITS]
        for inv in (0, 1):
            w = window ^ inv
            dists = (table != w).sum(axis=1)
            j = int(np.argmin(dists))
            if dists[j] <= max_dist:
                hits.append((off, j, int(dists[j]), inv))
    return hits


def analyze(audio: np.ndarray, fs: int = FS_DEFAULT) -> dict:
    """One-shot front-end characterisation of a P2 segment."""
    f_lo, f_hi = measure_carriers(audio, fs)
    out: dict = {"carriers_measured": (f_lo, f_hi),
                 "spacing": f_hi - f_lo, "tones": {}}
    all_bits = []
    for f0 in (f_lo, f_hi):
        sym, flock, off = _tone_symbols(audio, fs, f0)
        order, c2, c4 = modulation_order(sym)
        bits = dbpsk_bits(sym)
        all_bits.append(bits)
        out["tones"][round(f0, 1)] = {
            "locked_hz": round(flock, 2), "offset": off,
            "order": order, "C2": round(c2, 3), "C4": round(c4, 3),
            "n_symbols": len(sym)}
    # CS hunt on each tone independently and on the two combined (diversity model)
    out["cs_hits"] = {}
    for name, bits in (("tone_lo", all_bits[0]), ("tone_hi", all_bits[1])):
        out["cs_hits"][name] = hunt_control_signals(bits)
    return out


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--fields":
        # one CRC-valid field per line, `seconds hex` -- what the regression
        # fixture in rf-corpus grades against known ground truth
        for wav in args[1:]:
            audio, fs = _read_wav_mono(wav)
            for t, field in decode_bursts(audio, fs):
                print(f"{t:.3f} {field.hex()}")
        raise SystemExit(0)
    for path in args:
        audio, fs = _read_wav_mono(path)
        r = analyze(audio, fs)
        print(f"\n== {path} ==")
        print(f"  carriers {tuple(round(x,1) for x in r['carriers_measured'])} "
              f"spacing {r['spacing']:.1f} Hz")
        for f0, t in r["tones"].items():
            print(f"  tone {f0}: locked {t['locked_hz']} order {t['order']} "
                  f"(C2={t['C2']} C4={t['C4']}) nsym={t['n_symbols']}")
        for tone, hits in r["cs_hits"].items():
            best = sorted(hits, key=lambda h: h[2])[:4]
            print(f"  CS hits {tone}: {len(hits)} <=3;  best: "
                  + ", ".join(f"{CS_NAMES[j]}@{o}(d{d},inv{iv})" for o, j, d, iv in best))
