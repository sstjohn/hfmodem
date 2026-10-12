# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Each chirp is a query and the window behind it is the answer.

A chirp carries three things a peer could be reading -- a status byte, the
counter in its low two bits, and whatever is in the field -- and whatever comes
back in the receive window is the only thing that says what any of them meant.
So every probe window is read the same way whatever the arm was for: segment it,
name each burst with the instruments that already exist (`p1rx` for a PACTOR-1
codeword, the published spreading sequence and the published trainings for the
two PACTOR-4 families), and write the pair down. Across arms the
(query -> reply) pairs accumulate into a table somebody can read.

ONE FAMILY IS NO LONGER JUST NAMED. A robust short packet's 64 datablock
symbols are `p4robusttx`'s [64,6,32] Walsh code, so the six bits it spreads
come back as `p4_value` with the Walsh peak that earned them, and its header
comes back as A or B. That is a field -- still no CRC and no counter, and the
value is pinned only up to a fixed XOR, but a reply whose value MOVES with the
value we keyed is the code's meaning being handed over. The answerer's 27-symbol
control burst carries a field of the same width, six symbols, but unspread and
with no mask known for it, so that one is reported raw or not at all.

WHAT SEPARATES THE TWO PACTOR-4 FAMILIES IS THE SPREADING AND NOT THE HEADER.
Both families open on a header and the Chu19 template reaches only one of them,
so asking the header first names a robust burst and a normal one alike. The
question that divides them is whether every 16 chips is a scaled copy of the
published SF-16 sequence: measured over every PACTOR-4 recording on file, an
off-air robust burst spends 0.29-0.34 of its energy that way and a clean one
0.81, where the two CRC-valid normal-short packets spend 0.079-0.085 and the
quiet channel beside them 0.086. Below the knee the question becomes the
normal-mode one -- a coherent lock on the six published CAZAC trainings at
their own lattice, 0.39-0.42 on those two packets against 0.10-0.13 for robust
bursts and 0.07-0.09 for noise and for this station's own chirp.

WHAT THE CHU ROOT IS WORTH OFF THE AIR: very little on its own. audio1's three
caller bursts carry one 320-chip header -- 0.957 correlated between them --
and `p4sig.header_score` answers roots 5, 11 and 7 on the three. The
root is recorded because a CONSISTENT root across many cycles at one station
would mean something; one burst's root means nothing, and nothing here is
classed by it.
"""
from __future__ import annotations

from functools import lru_cache
from math import gcd
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import fftconvolve, get_window

from . import (p1rx, p4normal, p4robusttx, p4rx, p4sig, p4words, pactor1,
               rxfront, spec)

#: The passband and power-frame grid `rxfront` measures occupancy on. A burst is
#: a run of frames standing over the window's own quietest tenth, which is the
#: one segmenter that does not presume the waveform it is about to name.
BURST_KNEE_DB = 6.0
BURST_MIN_S = 0.05

#: Which family a burst belongs to, on the SF-16 despread energy fraction.
ROBUST_FRACTION_KNEE = 0.25

# The 2031 v53 copies have zero Walsh residual at coherence 0.245/0.244.
FULL_PEAK_COHERENCE = 0.20

#: A lock strong enough that a word it cannot name is a word we do not hold,
#: rather than a burst the despreader never got hold of. Measured: this
#: station's gateways and the clean recordings sit at 0.52-0.91 and the
#: weak-signal tapes at 0.30-0.35.
WORD_LOCK_KNEE = 0.50

#: How long a burst has to be to BE a control word rather than to contain one:
#: the shortest word in the bank, and a little past the longest.
WORD_SHAPED_S = (0.160, 0.270)

#: What a robust block's Walsh peak has to reach before it is a value. The seven
#: known packets and every clean render come back at 64. The null, through this
#: same reader with every origin and quarter turn searched: 200 random DQPSK
#: robust bursts of the same length peak at 38 at worst, mean 29.3, and 60 noise
#: windows at 34, mean 26.2. Not one of the 260 was named, and not one was given
#: a header. The knee sits ten points clear of the best of them.
ROBUST_BLOCK_KNEE = 48

#: Symbols of slack the field reader gets at each end of a burst. The onset a
#: power grid hands it can be a spread period out either way -- taunus's two
#: header-B packets are one symbol late -- and a field read at the wrong origin
#: is not a weaker reading, it is a different packet.
_FIELD_SLACK = 4

#: How much of a header has to agree before the burst is called A or B, out of
#: `p4robusttx.HEADER_WORD`'s eighteen steps. The two words agree with each
#: other in seven of them, and a relabelled carrier branch agrees by chance.
HEADER_WORD_KNEE = 14

#: A WORD UNDER THE SEGMENTER'S KNEE IS STILL A WORD. VE3KPG's answers on
#: onair-0926-1218 stood 2-7 dB over the floor and the 6 dB segmenter found
#: none of them, while its SF16 symbols sit 12 dB higher again. So every
#: window is also scanned for the bank's words as they are spoken: at every
#: sample, the phase steps between despread symbols one spread period apart,
#: taken to unit magnitude, against each word's own steps. A step is blind to
#: the carrier, so the grid only has to keep the despreading whole. The
#: measured words scan 0.94-1.00, and 360 s of other arms' noise, QRM and
#: PACTOR-1 traffic plus white noise at most 0.79 against the 27-symbol words.
#: The scan only nominates: `p4words.match` names, on the symbols read at the
#: scan's own lock, against its own knee, and those same 360 s reach 0.79 on it.
#: CALLER20 and GATEWAY are too short to scan: seventeen and nineteen steps
#: put their scan null at 0.84-0.86 and the match behind it at 0.87-0.90.
WORD_SCAN_KNEE = 0.85
WORD_SCAN_POWER_DB = 3.0
PLACED_MATCH_KNEE = 0.75
PLACED_FRACTION_KNEE = 0.12
WORD_SCAN_HZ = 144.0
WORD_SCAN_STEP_HZ = 48.0

#: A WORD BEHIND OUR OWN CARRIER IS READ BY ITS TAIL. Still keying packet 1, we
#: hear the gateway's answer word only past our mute: KB5LZK's Y27 on
#: onair-0926-1342 sat at our boundary + 0.918 s and the window after our
#: carrier held its last 16-22 symbols, too few for the scan's 26-step kernels.
#: Those tails read r 0.996-0.998 at SF16 fraction 0.72-0.79. Sliding 0.3 s
#: windows over seven whole arms (1230, 1308, 1322, 1339, 1342 off its words,
#: 1409, 1415: CS1 repeat requests, our own packets, QRN and silence) reach
#: r 0.75 at fraction 0.10-0.14 at most, so a tail is named only past both knees.
WORD_TAIL_SYMBOLS = 16
#: A symbol despread that well is where the word's tail is taken to begin,
#: past whatever our own carrier and its mute left at the window's head.
WORD_TAIL_BLOCK = 0.5

#: A normal-short packet, on the coherent CAZAC training-lattice lock.
NORMAL_LATTICE_KNEE = 0.33

#: A CLASS NEEDS AT LEAST ITS OWN LENGTH. Every one of these is a published
#: dimension on the 16-chip grid at 1800 chips/s, 8.889 ms a spread symbol:
#: 19-21 symbols of header, 27 for the control signal audio1's answering
#: station keys three of, 85 for the robust short packet its caller keys three
#: of, and `p4shorttx.EMISSION_S` for a normal short. A burst shorter than the
#: thing it would be named is a fragment, and fragments are energy.
HEADER_MIN_S = 0.150
ROBUST_CS_S = (0.230, 0.260)
ROBUST_SHORT_S = (0.740, 0.780)
NORMAL_MIN_S = 0.690

#: The width a normal-short has to have before its lattice is worth a lock.
NORMAL_BW_HZ = 1500.0

#: The span the width is measured across, and the peak-relative floor it is
#: measured to. `rxfront.occupied_bw` takes the same band at -14 dB, for a
#: question about PACTOR-1 rather than about a 2 kHz normal-mode packet.
BW_BAND_HZ = (300.0, 2700.0)
BW_FLOOR_DB = -20.0

_SPREAD_CONJ = p4sig.SPREAD16.conj()
_SPREAD_ENERGY = float(np.vdot(p4sig.SPREAD16, p4sig.SPREAD16).real)
#: THE CARRIER IS SEARCHED WHERE A LINK CAN SIT, not where most of them do. The
#: N9VC link on the 08-02 K5DAT-13 recording ran 400 Hz up the passband, P1
#: tones and robust shorts alike, and a +/-120 Hz search locked its shorts on an
#: alias 450 Hz off: the Walsh values survived it and every header read '?'.
#: A block's despread holds its lock over about 100 Hz, so a quarter of that is
#: the step and the refinement below finishes it. The grid spans the audio
#: passband: N9VC on the 09-25 Kiwi recording sits about +900 Hz, and a
#: +/-600 Hz grid read its headers on an alias.
_DESPREAD_CFOS = np.arange(-1488.0, 1489.0, 24.0)
_DESPREAD_MIN_BLOCKS = 4
#: The longest word in the bank plus the slide `p4words.match` is allowed.
_WORD_SYMBOLS = max(map(len, p4words.BANK.values())) + p4words.MATCH_SHIFT

_LATTICE = p4normal.layout('short', terminal_training=False)
_LATTICE_GUARD = np.arange(6, 26)
_LATTICE_POS = ((_LATTICE.training_offsets[:, None] + _LATTICE_GUARD).ravel()
                * p4rx.SAMPLES_PER_CHIP)
_LATTICE_EXPECTED = np.concatenate(
    [p4normal.training_sequence(k + 1)[_LATTICE_GUARD] for k in range(6)])
_LATTICE_SPAN = ((_LATTICE.total_symbols - 1) * p4rx.SAMPLES_PER_CHIP
                 + len(p4rx.RRC_TAPS))

_CLASS_RANK = ('none', 'filled', 'energy', 'p4-robust?', 'p4-normal?', 'p1-cs',
               'p4-header', 'p4-robust-cs', 'p4-normal', 'p4-robust-short',
               'p4-robust+block', 'p4-word')


def _frames(seg: np.ndarray, fs: int) -> tuple[np.ndarray, int]:
    """Per-frame in-band power in dB, on `rxfront`'s occupancy grid."""
    x = np.asarray(seg, dtype=np.float64)
    w = int(round(rxfront.OCCUPANCY_FRAME_S * fs))
    if x.size < w * rxfront.OCCUPANCY_MIN_FRAMES or not np.all(np.isfinite(x)):
        return np.empty(0), w
    lo, hi = rxfront.OCCUPANCY_BAND_HZ
    n = x.size // w
    # Band-limited frame by frame, not over the window: a brick wall across the
    # whole four seconds rings, and the ringing walks a codeword's onset.
    power = np.abs(np.fft.rfft(x[:n * w].reshape(n, w), axis=1)) ** 2
    f = np.fft.rfftfreq(w, 1 / fs)
    return 10 * np.log10(power[:, (f >= lo) & (f <= hi)].sum(1) / w ** 2 + 1e-20), w


def inband_db(seg: np.ndarray, fs: int = 48000) -> tuple[float, float]:
    """(mean, quietest tenth) of this window's in-band level, both in dB."""
    e, _ = _frames(seg, fs)
    if not e.size:
        return 0.0, 0.0
    return (float(10 * np.log10(np.mean(10 ** (e / 10)) + 1e-20)),
            float(np.percentile(e, 10)))


def bursts(seg: np.ndarray, fs: int = 48000, *, knee_db=BURST_KNEE_DB, bridge_s=0., gaps=(), floor_tail_s=0., floor_db=None) -> list[tuple[float, float]]:
    """(onset seconds, duration seconds) of every run of passband energy.

    THE CONTRAST IS WITHIN THE WINDOW and that is the one thing this cannot see
    past: an emission filling the whole receive window raises the tenth it is
    measured against and leaves no edge, so it reads as no burst at all. That
    case is not left looking like silence -- `query` carries the window's own
    in-band level and the floor the session measured beside it, and names it
    `filled` -- but it is not a burst and nothing here will find one.
    """
    e, w = _frames(seg, fs)
    if not e.size:
        return []
    valid = np.ones(len(e), dtype=bool)
    for begin, end in gaps:
        valid[max(0, begin // w):max(0, min(len(e), (end + w - 1) // w))] = False
    if not np.any(valid):
        return []
    if floor_db is not None:
        floor = floor_db
    elif floor_tail_s:
        tail = valid & (np.arange(len(e)) >= len(e) - max(1, round(floor_tail_s * fs / w)))
        if not np.any(tail):
            return []
        floor = float(np.median(e[tail]))
    else:
        floor = float(np.percentile(e[valid], 10))
    active = (e > floor + knee_db) & valid
    if bridge_s:
        # Only close short interior fades, never extend an end into silence.
        runs=np.flatnonzero(np.diff(np.r_[False,~active,False].astype(np.int8)))
        for begin,end in zip(runs[::2],runs[1::2]):
            if begin>0 and end<len(active) and (end-begin)*w/fs<=bridge_s:
                active[begin:end]=True
    on = np.r_[False, active, False]
    edges = np.flatnonzero(np.diff(on.astype(np.int8)))
    out = []
    for i, j in zip(edges[0::2], edges[1::2]):
        duration = (j - i) * w / fs
        if duration >= BURST_MIN_S:
            out.append((i * w / fs, duration))
    return out


def occupied_bw(seg: np.ndarray, fs: int = 48000) -> float:
    """Width in Hz of the passband bins within `BW_FLOOR_DB` of the peak."""
    x = np.asarray(seg, np.float64)
    if x.size < int(0.02 * fs):
        return 0.0
    mag = np.abs(np.fft.rfft(x * get_window('hann', x.size)))
    f = np.fft.rfftfreq(x.size, 1 / fs)
    lo, hi = BW_BAND_HZ
    band = (f >= lo) & (f <= hi)
    over = mag[band] > mag[band].max() * 10 ** (BW_FLOOR_DB / 20)
    return float(f[band][over].max() - f[band][over].min()) if over.sum() > 1 else 0.0


def despread(seg: np.ndarray, fs: int = 48000,
             count: int | None = None) -> tuple[float, np.ndarray, float]:
    """(SF-16 energy fraction, the symbols it despreads to, carrier) at the best lock.

    Block by block along the whole burst, so a header on one family's template
    cannot carry the reading: each 16 chips is correlated against `SPREAD16`
    and charged against its own energy, and the fraction is the mean over every
    block at the best chip phase and carrier offset. This is what says WHICH
    PACTOR-4 FAMILY a burst belongs to, and it is asked before any header is.
    The symbols that fall out of the same lock are what `p4words` matches.
    """
    fraction, symbols, cfo, _, _ = _lock(seg, fs)
    return fraction, symbols[:count or _WORD_SYMBOLS], cfo


def _lock(seg: np.ndarray, fs: int):
    """`despread`'s reading with every symbol, the sample its first one starts
    at, and each symbol's own fraction."""
    z = p4rx.audio_to_baseband(np.asarray(seg, np.float64), fs)
    matched = p4rx.matched_filter(z)[p4rx.MATCHED_DELAY:]
    grids = [matched[phase::p4rx.SAMPLES_PER_CHIP]
             for phase in range(p4rx.SAMPLES_PER_CHIP)]
    # THE SPREADING ALONE DOES NOT PLACE THE CARRIER. At chip rate a carrier
    # and the same carrier 1800 Hz away despread identically, and 450 Hz away
    # nearly so: N9VC's weakest 08-02 short despreads better at +824 Hz than
    # at its own +384. Only one of them sits where the burst's energy is.
    captured = _captured(z)
    best = (0.0, 0, 0.0, 0)
    for phase, chips in enumerate(grids):
        if chips.size < _DESPREAD_MIN_BLOCKS * p4sig.SPREAD_FACTOR:
            continue
        scores = _coarse_fractions(chips) * captured[:, None]
        cfo, start = np.unravel_index(int(scores.argmax()), scores.shape)
        if scores[cfo, start] > best[0]:
            best = (float(scores[cfo, start]), phase,
                    float(_DESPREAD_CFOS[cfo]), int(start))
    if not grids[best[1]].size:
        return 0.0, np.empty(0, complex), 0.0, 0, np.empty(0)
    # THE SYMBOLS NEED A FINER CARRIER THAN THE FAMILY DOES. A block spans
    # 8.9 ms, so the coarse grid's 12 Hz residual is little to it and three
    # turns across a 27-symbol word.
    chips = grids[best[1]]
    best = (_despread_at(chips, best[2], best[3])[0],) + best[1:]
    for cfo in np.arange(best[2] - _DESPREAD_CFOS[1] + _DESPREAD_CFOS[0],
                         best[2] - _DESPREAD_CFOS[0] + _DESPREAD_CFOS[1] + 1e-9, 0.5):
        fraction = _despread_at(chips, cfo, best[3])[0]
        if fraction > best[0]:
            best = (fraction, best[1], float(cfo), best[3])
    fraction, symbols, ratios = _despread_at(chips, best[2], best[3])
    first = (best[1] + best[3] * p4rx.SAMPLES_PER_CHIP) * fs / p4rx.SAMPLE_RATE
    return fraction, symbols, best[2], first, ratios


def _captured(z: np.ndarray) -> np.ndarray:
    """The share of the burst's energy the matched filter takes in with the
    carrier at each coarse offset, relative to the best of them."""
    n = z.size + p4rx.RRC_TAPS.size - 1
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1 / p4rx.SAMPLE_RATE))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(z, n)) ** 2)
    response = np.fft.fftshift(np.abs(np.fft.fft(p4rx.RRC_TAPS, n)) ** 2)
    energy = np.array([spectrum @ np.interp(freqs - cfo, freqs, response,
                                            left=0.0, right=0.0)
                       for cfo in _DESPREAD_CFOS])
    return energy / max(float(energy.max()), 1e-30)


def _coarse_fractions(chips: np.ndarray) -> np.ndarray:
    """`_despread_at`'s fraction over the whole coarse grid, as [carrier, start].

    Every 16-chip window is correlated once per carrier, and each start takes
    the mean over the windows on its own block grid.
    """
    turns = np.arange(chips.size) / p4rx.CHIP_RATE
    tuned = chips * np.exp(-2j * np.pi * _DESPREAD_CFOS[:, None] * turns)
    windows = np.lib.stride_tricks.sliding_window_view(
        tuned, p4sig.SPREAD_FACTOR, axis=1)
    power = np.maximum(np.lib.stride_tricks.sliding_window_view(
        np.abs(chips) ** 2, p4sig.SPREAD_FACTOR).sum(1), 1e-30) * _SPREAD_ENERGY
    ratio = np.abs(windows @ _SPREAD_CONJ) ** 2 / power
    fractions = np.zeros((_DESPREAD_CFOS.size, p4sig.SPREAD_FACTOR))
    for start in range(p4sig.SPREAD_FACTOR):
        usable = (chips.size - start) // p4sig.SPREAD_FACTOR
        if usable >= _DESPREAD_MIN_BLOCKS:
            fractions[:, start] = ratio[:, start::p4sig.SPREAD_FACTOR][:, :usable].mean(1)
    return fractions


def _despread_at(chips: np.ndarray, cfo: float, start: int):
    usable = (chips.size - start) // p4sig.SPREAD_FACTOR
    if usable < _DESPREAD_MIN_BLOCKS:
        return 0.0, np.empty(0, complex), np.empty(0)
    turns = np.arange(chips.size) / p4rx.CHIP_RATE
    tuned = chips * np.exp(-2j * np.pi * cfo * turns)
    blocks = tuned[start:start + usable * p4sig.SPREAD_FACTOR]
    blocks = blocks.reshape(usable, p4sig.SPREAD_FACTOR)
    power = np.maximum((np.abs(blocks) ** 2).sum(1), 1e-30) * _SPREAD_ENERGY
    symbols = blocks @ _SPREAD_CONJ
    ratios = np.abs(symbols) ** 2 / power
    return float(ratios.mean()), symbols / _SPREAD_ENERGY, ratios


def despread_fraction(seg: np.ndarray, fs: int = 48000) -> float:
    """`despread`'s fraction alone, for a caller that wants only the family."""
    return despread(seg, fs)[0]


def despread_word(seg: np.ndarray, fs: int = 48000) -> np.ndarray:
    """The burst's opening symbols, off a lock taken on THE HEAD ALONE.

    A packet's own best grid is the one its 85 data symbols vote for, and on a
    weak recording that is not the grid its 20-symbol header sits on: read that
    way, PACTORIVaudio1's three caller headers correlate against each other at
    0.48-1.00 where the head-locked reading puts them at 0.99. What is matched
    against the bank is a word, so the lock that finds it is a word's lock.
    """
    head = int(_WORD_SYMBOLS * p4sig.SPREAD_FACTOR / p4rx.CHIP_RATE * fs)
    return despread(np.asarray(seg)[:head + fs // 100], fs)[1]


def coherence_knee(walsh_peak, ordinary):
    """Only a complete 64-of-64 codeword earns the lower spreading floor."""
    return FULL_PEAK_COHERENCE if walsh_peak == 64 else ordinary


def robust_dibits(symbols: np.ndarray) -> np.ndarray:
    """Hard quarter-turn decisions, with the residual common rotation removed.

    One rotation for the whole burst, estimated off the fourth power, because
    the header is quaternary even where the field behind it is not.
    """
    step = symbols[1:] * symbols[:-1].conj()
    step = step / np.maximum(abs(step), 1e-30)
    rotation = np.angle(np.mean(step ** 4)) / 4
    return (np.round(np.angle(step * np.exp(-1j * rotation)) / (np.pi / 2))
            % 4).astype(int)


def _raw_field(field: np.ndarray) -> int | None:
    """Six unspread steps as six bits, or None where one of them is not a bit."""
    if not np.all((field == 0) | (field == 2)):
        return None
    return int(''.join('1' if step == 2 else '0' for step in field), 2)


def _header_match(steps: np.ndarray) -> tuple[str, int]:
    """('A' | 'B' | '?', how many of the eighteen steps the winner agreed on)."""
    word = np.asarray(steps)[p4robusttx.HEADER_WORD]
    scores = {name: int(np.count_nonzero(
        word == np.array(dibits)[p4robusttx.HEADER_WORD]))
        for name, dibits in p4robusttx.HEADER_DIBITS.items()}
    best, runner = sorted(scores, key=scores.get, reverse=True)
    named = scores[best] >= HEADER_WORD_KNEE > scores[runner]
    return (best if named else '?'), scores[best]


def header_name(steps: np.ndarray) -> str:
    """Which of `p4robusttx`'s two headers a burst opened on: 'A', 'B' or '?'.

    Matched on `HEADER_WORD` alone. The nineteenth step is the one the two
    recordings disagree on at the same header, so it is not evidence of either.
    """
    return _header_match(steps)[0]


def robust_field(seg: np.ndarray, fs: int, packet_symbols: int) -> dict:
    """What a robust short packet or control burst says, over its own extent.

    THE PACKET IS ITS OWN ORIGIN AND ITS OWN CARRIER BRANCH. `despread`
    maximises a magnitude, so it is blind both to the whole spread period the
    burst grid can be out by and to the 28.125 Hz offset that relabels every
    dibit. Neither has to be guessed: a window of the right length either
    reaches a Walsh codeword and a known header or it does not, so every origin
    within `_FIELD_SLACK` and all four quarter turns are tried and the block
    and the header are maximised SEPARATELY over them. Read off the
    despreader's first offering instead, the two taunus packets on header B
    come back a symbol late at peak 26.

    A short's six bits come back with the peak that earned them, and a peak
    under `ROBUST_BLOCK_KNEE` returns no value at all. The control burst's six
    steps have no code to vote with -- only its header does -- so they are read
    raw, 0 -> 0 and 2 -> 1; a step that is neither is not a bit, and that field
    goes unread rather than guessed.
    """
    _, symbols, _ = despread(seg, fs, packet_symbols + 2 * _FIELD_SLACK)
    return robust_symbols(symbols, packet_symbols)


def robust_symbols(symbols, packet_symbols):
    """Read the header and Walsh field from already located spread symbols."""
    spread = packet_symbols == p4robusttx.SHORT_SYMBOLS
    block = header = None
    for origin in range(symbols.size - packet_symbols + 1):
        steps = robust_dibits(symbols[origin:origin + packet_symbols])
        for turn in range(4):
            turned = (steps + turn) % 4
            field = turned[p4robusttx.HEADER_SYMBOLS:]
            if spread:
                value, peak = p4robusttx.block_value((field == 2).astype(int))
            else:
                value, peak = _raw_field(field), 0
            name, agreed = _header_match(turned)
            read = (peak, agreed, name, value)
            if block is None or (peak, agreed) > (block[0], block[1]):
                block = read
                block_steps = turned
            if header is None or (agreed, peak) > (header[1], header[0]):
                header = read
    if block is None:
        return dict(p4_value=None, p4_walsh_peak=None, p4_hdr=None)
    peak, _, block_hdr, value = block
    _, _, header_hdr, header_field = header
    if not spread:
        # An unnamed header leaves the origin and the branch unknown, and six
        # steps read at an unknown origin are not a field.
        return dict(p4_value=header_field if header_hdr != '?' else None,
                    p4_walsh_peak=None, p4_hdr=header_hdr)
    if peak >= ROBUST_BLOCK_KNEE:
        if block_hdr == '?':
            # Walsh correlation is unsigned: the other binary polarity can
            # carry the header even when errors chose this block branch.
            block_hdr, _ = _header_match((block_steps + 2) % 4)
        return dict(p4_value=value, p4_walsh_peak=peak, p4_hdr=block_hdr)
    # NO CODEWORD IS NOT NO PACKET. Where the block reaches nothing the peak is
    # noise picking an origin, so the header is asked on its own -- a Dragon
    # answering with a short we cannot decode still opens on a header we know,
    # and losing that would lose the one thing the burst did say.
    return dict(p4_value=None, p4_walsh_peak=peak, p4_hdr=header_hdr)


_SYMBOL_N = p4sig.SPREAD_FACTOR * p4rx.SAMPLES_PER_CHIP
_SPREAD_TAPS = np.zeros(_SYMBOL_N, complex)
_SPREAD_TAPS[::p4rx.SAMPLES_PER_CHIP] = p4sig.SPREAD16
_SPREAD_KERNEL = _SPREAD_TAPS[::-1].conj() / _SPREAD_ENERGY


def _step_kernel(word: np.ndarray) -> np.ndarray:
    kernel = np.zeros(_SYMBOL_N * (word.size - 2) + 1, complex)
    kernel[::_SYMBOL_N] = (word[1:] * word[:-1].conj()).conj()
    return kernel[::-1]


_STEP_KERNELS = {name: _step_kernel(word) for name, word in p4words.BANK.items()
                 if word.size >= p4robusttx.CS_SYMBOLS}


@lru_cache(maxsize=8)
def _step_spectra(n):
    """Fixed word kernels at a bounded set of padded transform lengths."""
    return tuple((kernel.size, fft(kernel, n)) for kernel in _STEP_KERNELS.values())


def _word_scan(audio, fs: int = 48000, cfo_hz: float | None = None, *, work=None) -> list[dict]:
    """Every 27-symbol bank word in the window, found without the segmenter.

    One dict per word named, in time order: onset and duration in seconds, the
    word, its `p4words.match` score and the step coherence that nominated it.
    See `WORD_SCAN_KNEE`.
    """
    z = p4rx.audio_to_baseband(np.asarray(audio, np.float64), fs)
    chips = p4rx.matched_filter(z)[p4rx.MATCHED_DELAY:z.size + p4rx.MATCHED_DELAY]
    longest = p4robusttx.CS_SYMBOLS + p4words.MATCH_SHIFT
    if chips.size < longest * _SYMBOL_N:
        return []
    turns = np.arange(chips.size) / p4rx.SAMPLE_RATE
    centre = cfo_hz or 0.0
    best = np.zeros(chips.size)
    lock = np.zeros(chips.size)
    step_size = chips.size - _SYMBOL_N
    n = next_fast_len(step_size + max(k.size for k in _STEP_KERNELS.values()) - 1)
    spectra = _step_spectra(n)
    carriers = centre + np.arange(-WORD_SCAN_HZ, WORD_SCAN_HZ + 1e-9, WORD_SCAN_STEP_HZ)
    for cfo in carriers:
        if work is not None and not work.checkpoint('detect'):
            return []
        symbols = fftconvolve(chips * np.exp(-2j * np.pi * cfo * turns),
                              _SPREAD_KERNEL)[_SYMBOL_N - 1:]
        steps = symbols[_SYMBOL_N:] * symbols[:-_SYMBOL_N].conj()
        steps = np.divide(steps, np.abs(steps), out=np.zeros_like(steps),
                          where=np.abs(steps) > 0)
        # Each carrier has one step input shared by every word template.
        transformed = fft(steps, n)
        for kernel_size, spectrum in spectra:
            summed = ifft(transformed * spectrum)[kernel_size - 1:step_size]
            coherence = np.abs(summed) / (p4robusttx.CS_SYMBOLS - 1)
            better = coherence > best[:coherence.size]
            best[:coherence.size][better] = coherence[better]
            # What the steps turn by is the carrier the grid left over.
            lock[:coherence.size][better] = (
                cfo + np.angle(summed[better]) / (2 * np.pi) * p4words.SYMBOL_RATE)
    levels, frame_n = _frames(audio, fs)
    floor = np.percentile(levels, 10) if levels.size else np.inf
    found, taken = [], np.zeros(chips.size, bool)
    for at in sorted(np.flatnonzero(best >= WORD_SCAN_KNEE), key=lambda i: -best[i]):
        if taken[at]:
            continue
        if work is not None and not work.checkpoint('detect'):
            return []
        taken[max(0, at - _SYMBOL_N // 2):at + _SYMBOL_N // 2] = True
        cfo = lock[at]
        spots = at + _SYMBOL_N * np.arange(longest)
        spots = spots[spots + _SYMBOL_N <= chips.size]
        grid = spots[:, None] + p4rx.SAMPLES_PER_CHIP * np.arange(p4sig.SPREAD_FACTOR)
        heard = (chips[grid] * np.exp(-2j * np.pi * cfo * turns[grid])) @ _SPREAD_CONJ
        name, score = p4words.match(heard)
        if name not in _STEP_KERNELS:
            continue
        _, slide, residual = p4words.correlate(p4words.BANK[name], heard)
        cfo += residual
        at -= slide * _SYMBOL_N
        span = len(p4words.BANK[name]) * _SYMBOL_N
        if name in ("Z27", "V27", "U27", "T27", "W27"):
            first = max(0, round(at * fs / p4rx.SAMPLE_RATE / frame_n))
            stop = min(len(levels), round((at + span) * fs / p4rx.SAMPLE_RATE / frame_n))
            if stop <= first or np.mean(10 ** ((levels[first:stop] - floor) / 10)) < 10 ** (WORD_SCAN_POWER_DB / 10):
                continue
        taken[max(0, at - span):at + span] = True
        found.append(dict(onset=at / p4rx.SAMPLE_RATE,
                          duration=span / p4rx.SAMPLE_RATE + p4robusttx.PULSE_S,
                          word=name, score=score, coherence=float(best[at]), cfo_hz=float(cfo)))
    return sorted(found, key=lambda hit: hit['onset'])


def _word_tail(audio, fs: int = 48000, *, refine=False) -> list[dict]:
    """The X27 or Y27 whose head our carrier censored, as `word_scan` reports one.

    The window opens where the word is already under way, so its first symbols
    are some tail of the word: every tail of at least `WORD_TAIL_SYMBOLS` is
    tried from the first symbol that despreads as one, and the onset is where
    the named word's first symbol would sit. `coherence` is the SF16 fraction
    of the symbols matched. See `WORD_TAIL_SYMBOLS`.
    """
    symbol_s = p4sig.SPREAD_FACTOR / p4rx.CHIP_RATE
    if np.asarray(audio).size < WORD_TAIL_SYMBOLS * symbol_s * fs:
        return []
    _, symbols, cfo, first, ratios = _lock(audio, fs)
    lit = np.flatnonzero(ratios >= WORD_TAIL_BLOCK)
    if not lit.size or symbols.size - lit[0] < WORD_TAIL_SYMBOLS:
        return []
    start = int(lit[0])
    fraction = float(ratios[start:start + WORD_TAIL_SYMBOLS].mean())
    if fraction < ROBUST_FRACTION_KNEE:
        return []
    heard = symbols[start:] / np.maximum(np.abs(symbols[start:]), 1e-30)
    best = (0.0, None, 0, 0.0)
    for name, word in p4words.BANK.items():
        if len(word) != 27:
            continue
        word = p4words.BANK[name]
        for k in range(word.size - WORD_TAIL_SYMBOLS + 1):
            n = min(word.size - k, heard.size)
            r, shift, hz = p4words.correlate(word[k:k + n], heard[:n])
            if r > best[0] and p4words.interior(shift, hz):
                best = (r, name, -shift - k, hz)
    score, name, origin, residual = best
    if score < p4words.MATCH_KNEE:
        return []
    return [dict(onset=first / fs + (start + origin) * symbol_s,
                 duration=len(p4words.BANK[name]) * symbol_s + p4robusttx.PULSE_S,
                 word=name, score=score, coherence=fraction, cfo_hz=cfo - residual if refine else cfo)]


def training_origin(seg: np.ndarray, fs: int = 48000):
    """Coherent lock on the six published CAZAC trainings at their own lattice.

    `p4normal`'s short layout puts a 32-symbol training in front of each of six
    176-symbol data blocks. Correlating the interior of all six at once, over a
    start and carrier grid and then a refinement of it, asks whether this burst
    is built on that lattice -- which the Chu19 template cannot ask, its 304
    chips being longer than a normal-mode header.
    """
    z = p4rx.audio_to_baseband(np.asarray(seg, np.float64), fs)
    if z.size < _LATTICE_SPAN:
        return 0.0, 0, 0.0
    coarse = _lattice_grid(z, np.arange(0, z.size - _LATTICE_SPAN + 1,
                                        p4rx.SAMPLES_PER_CHIP),
                           np.arange(-120.0, 121.0, 5.0))
    fine = _lattice_grid(z, np.arange(max(0, coarse[1] - p4rx.SAMPLES_PER_CHIP),
                                      coarse[1] + p4rx.SAMPLES_PER_CHIP + 1, 2),
                         np.arange(coarse[2] - 6, coarse[2] + 6.01, 1.0))
    return max((coarse, fine), key=lambda fit: fit[0])


def training_lock(seg: np.ndarray, fs: int = 48000) -> float:
    return training_origin(seg,fs)[0]


def _lattice_grid(z, starts, cfos) -> tuple[float, int, float]:
    turns = np.arange(z.size) / p4rx.SAMPLE_RATE
    fits = (starts >= 0) & (starts + _LATTICE_SPAN <= z.size)
    best = (0.0, 0, 0.0)
    for cfo in cfos:
        matched = p4rx.matched_filter(z * np.exp(-2j * np.pi * cfo * turns))
        index = starts[:, None] + p4rx.MATCHED_DELAY + _LATTICE_POS[None, :]
        taken = matched[np.clip(index, 0, matched.size - 1)]
        power = np.sum(np.abs(taken) ** 2, 1) * _LATTICE_EXPECTED.size
        scored = np.where(fits, np.abs(taken @ _LATTICE_EXPECTED.conj()) ** 2
                          / np.maximum(power, 1e-30), 0.0)
        peak = int(scored.argmax())
        if scored[peak] > best[0]:
            best = (float(scored[peak]), int(starts[peak]), float(cfo))
    return best


def read_burst(seg: np.ndarray, onset: float, duration: float,
               fs: int = 48000, cfo_hz: float | None = None,
               headers: bool = True) -> dict:
    """Name one burst, over its own extent and nothing beyond it.

    `cfo_hz` is the session's own measured peer carrier offset where the arm has
    one. Locking the Chu search to it narrows a tie that is otherwise nine to
    thirteen templates wide, because a cyclic shift and a carrier offset are the
    same thing to this correlator.

    `headers` IS THE COST SWITCH AND IT BELONGS TO THE CALLER'S CLOCK. Three
    things here are expensive: the Chu family is 342 templates over up to
    sixteen offsets, the word lock re-despreads the head over its own grid, and
    the training lattice runs a start-and-carrier search -- together 1.13 s for
    one burst and 1.95 s for a ten-burst window, out of a sixty-second
    post-grant deadline. Off, all three are skipped and a burst is named by its
    FAMILY alone, `p4-robust?` or `p4-normal?`, for a pass that runs with the
    transmitter already down to say which word it was.
    """
    row = dict(onset_ms=round(onset * 1e3, 1), duration_ms=round(duration * 1e3, 1),
               p1_cs=None, spread_score=0.0, header_score=None,
               root=None, shift=None, cfo_hz=None, training_lock=0.0,
               word=None, word_score=0.0, cfo_locked=cfo_hz is not None,
               p4_value=None, p4_walsh_peak=None, p4_hdr=None)
    if 0.07 <= duration <= 0.26:
        control = p1rx.decode_control_signal(seg, onset, duration, fs)
        if control is not None and control.errors == 0:
            row['p1_cs'] = spec.P1_CS_NAMES[control.index]
            row['p1_word'] = f'0x{pactor1.CS_WORDS[control.index]:03X}'
            row['p1_sense'] = control.sense
    # OVER THE BURST AND NOT A CHIP BEYOND IT. Scoring burst-plus-a-header let
    # a 50 ms fragment borrow the emission behind it and be named for it.
    body = np.asarray(seg)[int(onset * fs):int((onset + duration) * fs)]
    row['spread_score'] = round(float(p4sig.spread_score(body, fs)), 4)
    row['occupied_bw_hz'] = round(occupied_bw(body, fs), 1)
    fraction, _, carrier = despread(body, fs)
    row['despread_fraction'] = round(fraction, 4)
    robust = row['despread_fraction'] >= ROBUST_FRACTION_KNEE
    if headers and robust:
        word, score = p4words.match(despread_word(body, fs))
        row.update(word=word, word_score=round(score, 4))
    if (headers and not robust and duration >= NORMAL_MIN_S
            and row['occupied_bw_hz'] > NORMAL_BW_HZ):
        row['training_lock'] = round(training_lock(body, fs), 4)
    row['reply_class'] = _classify(row, duration, headers)
    # AFTER the family, and on the LENGTH rather than the class: a clean
    # control burst is named `p4-word:?` by the bank long before it is named
    # `p4-robust-cs`, and its six steps are the same six steps either way. A
    # header-A short is named CALLER20 and carries its 64-symbol block all the same.
    packet = None
    if robust and ROBUST_SHORT_S[0] <= duration <= ROBUST_SHORT_S[1]:
        packet = p4robusttx.SHORT_SYMBOLS
    elif robust and ROBUST_CS_S[0] <= duration <= ROBUST_CS_S[1]:
        packet = p4robusttx.CS_SYMBOLS
    if packet is not None:
        slack = round(_FIELD_SLACK * p4sig.SPREAD_FACTOR / p4rx.CHIP_RATE * fs)
        row.update(robust_field(
            np.asarray(seg)[max(0, int(onset * fs) - slack):
                            int((onset + duration) * fs) + slack], fs, packet))
    # The root is informational and the class never reads it: see the module.
    if headers and robust and row['spread_score'] >= p4sig.SPREAD_KNEE:
        score, root, shift, cfo = p4sig.header_score(body, fs, cfo_hz, carrier)
        row.update(header_score=round(float(score), 4), root=int(root),
                   shift=int(shift), cfo_hz=round(float(cfo), 2))
    return row


def _classify(row: dict, duration: float, headers: bool = True) -> str:
    if row['p1_cs'] is not None:
        return f"p1-cs:{row['p1_cs']}"
    if not headers:
        if row['despread_fraction'] >= ROBUST_FRACTION_KNEE:
            return 'p4-robust?'
        return ('p4-normal?' if duration >= NORMAL_MIN_S
                and row['occupied_bw_hz'] > NORMAL_BW_HZ else 'energy')
    if row['despread_fraction'] >= ROBUST_FRACTION_KNEE:
        # A MEASURED WORD OUTRANKS A LENGTH. `p4words` names what was said;
        # the duration below only says how long the saying took.
        if row['word'] is not None:
            return f"p4-word:{row['word']}"
        if (row['despread_fraction'] >= WORD_LOCK_KNEE
                and WORD_SHAPED_S[0] <= duration <= WORD_SHAPED_S[1]):
            return 'p4-word:?'
        if duration > ROBUST_SHORT_S[1]:
            return 'p4-robust+block'
        if ROBUST_SHORT_S[0] <= duration <= ROBUST_SHORT_S[1]:
            return 'p4-robust-short'
        if ROBUST_CS_S[0] <= duration <= ROBUST_CS_S[1]:
            return 'p4-robust-cs'
        return 'p4-header' if duration >= HEADER_MIN_S else 'energy'
    if row['training_lock'] >= NORMAL_LATTICE_KNEE:
        return 'p4-normal'
    return 'energy'


def class_key(reply_class: str) -> str:
    """What a reply is counted under: the bare class.

    The root is deliberately NOT part of it -- one burst's root is not evidence
    (see the module header) -- so a row prepared elsewhere that spells its root
    inside parentheses counts with the bare class it belongs to.
    """
    return reply_class.partition('(')[0].strip()


def _rank(reply_class: str) -> int:
    return _CLASS_RANK.index(reply_class.partition('(')[0].partition(':')[0])


def word_note(reply: dict) -> str:
    """What the bank and the block reader made of a robust burst, named or not."""
    notes = []
    if reply['word_score']:
        notes.append(f"word {reply['word'] or 'unmatched'} r {reply['word_score']:.2f}")
    if reply.get('p4_hdr'):
        value = 'unread' if reply['p4_value'] is None else reply['p4_value']
        peak = reply['p4_walsh_peak']
        notes.append(f"header {reply['p4_hdr']} value {value}"
                     + (f" peak {peak}/{p4robusttx.BLOCK_SYMBOLS}"
                        if peak is not None else ''))
    return ', '.join(notes)


def strongest(replies: list[dict]) -> dict | None:
    """The one burst a window is named by, when it carried several."""
    return max(replies, key=lambda r: _rank(r['reply_class']), default=None)


def read_window(audio, fs: int = 48000, cfo_hz: float | None = None,
                headers: bool = True) -> list[dict]:
    """Every burst in one receive window, in the order they were heard."""
    x = np.asarray(audio, dtype=np.float32)
    rows = [read_burst(x, onset, duration, fs, cfo_hz, headers)
            for onset, duration in bursts(x, fs)]
    if not headers:
        return rows
    for hit in word_scan(x, fs, cfo_hz):
        onset = hit.first_chip / fs
        end = onset + hit.duration_s
        rows = [row for row in rows if not (row['onset_ms'] < end * 1e3 and onset * 1e3
                                            < row['onset_ms'] + row['duration_ms'])]
        row = read_burst(x, onset, hit.duration_s, fs, cfo_hz, headers=False)
        row.update(word=hit.word, word_score=round(hit.score, 4),
                   word_coherence=round(hit.coherence, 4),
                   reply_class=f"p4-word:{hit.word}")
        rows.append(row)
    return sorted(rows, key=lambda row: row['onset_ms'])


def query(index: int, status: int | None, payload_len: int, chirp_end: int,
          audio, fs: int = 48000, cfo_hz: float | None = None,
          headers: bool = True, floor_db: float | None = None) -> dict:
    """One (query -> reply) pair: what we keyed, and what came back.

    `floor_db` is the quietest the session heard the channel beside this window
    -- the probe's own pre-key intervals. A window with no edges in it is
    `filled` rather than `none` when its level stands `BURST_KNEE_DB` over that
    floor, which is the one reading that separates a peer holding the channel
    from a peer saying nothing.
    """
    replies = read_window(audio, fs, cfo_hz, headers)
    best = strongest(replies)
    level, tenth = inband_db(audio, fs)
    against = tenth if floor_db is None else floor_db
    if best is None and level >= against + BURST_KNEE_DB:
        best_class = 'filled'
    else:
        best_class = best['reply_class'] if best else 'none'
    return dict(index=index, status=status, status_hex=status_name(status),
                counter=None if status is None else status & 3,
                payload_len=payload_len,
                chirp_end_sample=int(chirp_end),
                chirp_end_s=round(chirp_end / fs, 3),
                window_samples=int(np.asarray(audio).size),
                cfo_locked=cfo_hz is not None, headers_read=headers,
                window_rms_db=round(level, 1), window_floor_db=round(against, 1),
                reply_class=best_class, replies=replies)


def status_name(status: int | None) -> str:
    """How a query's status is spelled: the byte, or `clone` where it is not ours."""
    return 'clone' if status is None else f'0x{status:02X}'


def lines(row: dict) -> list[str]:
    """The arm log's own reading of one query, one line per burst."""
    counter = '-' if row['counter'] is None else row['counter']
    head = (f"    [p4 query] {row['index']} status={row['status_hex']} "
            f"counter={counter} field={row['payload_len']}B")
    if not row['replies']:
        return [f"{head} -> {row['reply_class']}, window {row['window_rms_db']:.1f} dB "
                f"over a {row['window_floor_db']:.1f} dB floor"]
    out = []
    for reply in row['replies']:
        named = reply['reply_class']
        if reply['header_score'] is not None:
            named += (f" (root {reply['root']} r {reply['header_score']:.2f} at "
                      f"cfo {reply['cfo_hz']:+.0f} Hz"
                      f"{', locked' if reply['cfo_locked'] else ''})")
        elif reply['word_score']:
            named += (f" ({reply['word'] or 'best'} r {reply['word_score']:.2f}, "
                      f"SF16 lock {reply['despread_fraction']:.2f})")
        elif named == 'p4-normal':
            named += (f" ({reply['occupied_bw_hz']:.0f} Hz wide, training "
                      f"lattice {reply['training_lock']:.2f})")
        out.append(f"{head} -> {named} at +{reply['onset_ms']:.0f} ms, "
                   f"{reply['duration_ms']:.0f} ms, SF16 "
                   f"{reply['despread_fraction']:.2f}")
    return out


# Records live separately so the decoder and stateful receiver share them without
# a circular import. The diagnostic query API above deliberately stays JSON.
from .p4observation import Kind, Observation


def _extent(audio, fs, onset_s, *, word=None, gaps=()):
    """Closed emission extent; fragmented or censored Y27 tails stay unknown."""
    x = np.asarray(audio)
    spans = bursts(x, fs, gaps=gaps)
    for start, duration in spans:
        stop = start + duration
        if abs(start - onset_s) > 0.05 or stop >= len(x) / fs - 0.005:
            continue
        extent = stop - onset_s
        if extent < ROBUST_CS_S[0]:
            return None
        if word == "Y27" and extent <= .50:
            horizon = onset_s + .805
            grid_slack = 2 * round(rxfront.OCCUPANCY_FRAME_S * fs) / fs
            if extent > WORD_SHAPED_S[1] + grid_slack or len(x) / fs < horizon:
                return None
            if any(a < horizon * fs and b > onset_s * fs for a, b in gaps):
                return None
            if any(stop <= tail < horizon for tail, _ in spans):
                return None
        return extent
    return None


def _word_record(hit, fs, origin, audio, whole=False, gaps=()):
    first = round(hit["onset"] * fs)
    return Observation(
        Kind.WORD,
        fs,
        origin,
        first,
        origin + first,
        0.240,
        hit["cfo_hz"],
        hit["coherence"],
        word=hit["word"],
        score=hit["score"],
        symbols_read=27,
        extent_s=_extent(audio, fs, first / fs, word=hit["word"], gaps=gaps) if whole else None,
    )


def word_scan(audio, fs=48000, cfo_hz=None, *, origin=0, whole=False, gaps=(), work=None):
    if work is not None and len(audio) > round(.680*fs):
        work.deferred.append('detect')
        return []
    if work is not None and not work.admits('detect'):
        return []
    if len(audio) < round(0.24 * fs):
        return []
    return [
        _word_record(hit, fs, origin, audio, whole, gaps)
        for hit in _word_scan(audio, fs, cfo_hz, **({'work': work} if work else {}))
    ]


def word_tail(audio, fs=48000, *, origin=0, refine=False, work=None):
    if work is not None and len(audio) > round(.680*fs):
        work.deferred.append('refine')
        return []
    if work is not None and not work.admits('refine'):
        return []
    return [_word_record(hit, fs, origin, audio)
            for hit in _word_tail(audio, fs, refine=refine)]


def _despread_window(audio, fs, cfo_hz):
    z = p4rx.audio_to_baseband(np.asarray(audio), fs)
    tuned = z * np.exp(-2j * np.pi * cfo_hz * np.arange(len(z)) / p4rx.SAMPLE_RATE)
    chips = p4rx.matched_filter(tuned)[p4rx.MATCHED_DELAY : len(z) + p4rx.MATCHED_DELAY]
    spread = fftconvolve(chips, _SPREAD_KERNEL)[_SYMBOL_N - 1 :]
    power = fftconvolve(abs(chips) ** 2, abs(_SPREAD_TAPS[::-1]) ** 2)[_SYMBOL_N - 1 :]
    return spread, power, len(chips)


def _despread_fraction(symbols, power, floor):
    return np.divide(
        abs(symbols) ** 2 * _SPREAD_ENERGY,
        np.maximum(power, 1e-30),
        out=np.zeros(power.shape),
        where=power > floor)


def placed_despread(audio, fs, onset_s, cfo_hz, *, count=20, slack_s=0.035):
    """One carrier and a bounded timing bracket; no acquisition search."""
    lo = max(0, round((onset_s - slack_s) * fs))
    hi = min(len(audio), round((onset_s + count / p4words.SYMBOL_RATE + slack_s) * fs))
    if hi - lo < count * fs / p4words.SYMBOL_RATE:
        return 0.0, np.empty(0, complex), None
    spread, power, size = _despread_window(audio[lo:hi], fs, cfo_hz)
    if size < count * _SYMBOL_N:
        return 0.0, np.empty(0, complex), None
    starts = np.arange(
        min(
            size - count * _SYMBOL_N + 1,
            round(2 * slack_s * p4rx.SAMPLE_RATE) + 1,
        )
    )
    indices = starts[:, None] + _SYMBOL_N * np.arange(count)
    fraction = np.divide(
        abs(spread[indices]) ** 2 * _SPREAD_ENERGY,
        np.maximum(power[indices], 1e-30),
        out=np.zeros(indices.shape),
        where=power[indices] > max(float(power.max()) * 1e-9, 1e-30),
    ).mean(1)
    best = int(fraction.argmax())
    return (
        float(fraction[best]),
        spread[indices[best]],
        lo + round(starts[best] * fs / p4rx.SAMPLE_RATE),
    )


LONG_OVERLAP_CORRELATION_MIN = .90
"""Guard against misaligned or foreign prefix buffers; not an SNR threshold."""


def prepare_despread_long(audio, fs, onset_s, cfo_hz):
    lo = round((onset_s-.035)*fs)
    hi = round((onset_s+3.230)*fs)
    if lo < 0 or hi > len(audio):
        return None
    spread, power, size = _despread_window(audio[lo:hi], fs, cfo_hz)
    starts = np.arange(round(.070*p4rx.SAMPLE_RATE)+1)
    indices = starts[:, None]+_SYMBOL_N*np.arange(352)
    if indices[-1, -1]+_SYMBOL_N > size:
        return None
    return dict(lo=lo, starts=starts, symbols=spread[indices], power=power[indices],
                power_floor=max(float(power.max())*1e-9, 1e-30),
                cfo=cfo_hz, onset=round(onset_s*fs))


def finish_despread_long(audio, fs, prepared, *, diagnostics=None):
    onset, lo, cfo = prepared['onset'], prepared['lo'], prepared['cfo']
    begin = onset+round(2.950*fs)
    end = min(len(audio), onset+round((373/112.5+.035)*fs))
    if end <= begin:
        return 0., np.empty(0, complex), None, 'incomplete'
    spread, power, size = _despread_window(audio[begin:end], fs, cfo)
    offset = round((begin-lo)*p4rx.SAMPLE_RATE/fs)
    starts = prepared['starts']
    indices = starts[:, None]+_SYMBOL_N*np.arange(344, 373)-offset
    valid = (indices[:, 0] >= 0) & (indices[:, -1]+_SYMBOL_N <= size)
    if not np.any(valid):
        return 0., np.empty(0, complex), None, 'incomplete'
    indices, starts = indices[valid], starts[valid]
    tail = spread[indices]*np.exp(-2j*np.pi*(p4rx.CENTRE_HZ+cfo)*(begin-lo)/fs)
    prefix = prepared['symbols'][valid]
    overlap = np.sum(tail[:, :8].conjugate()*prefix[:, 344:352], axis=1)
    energy = np.sqrt(np.sum(abs(tail[:, :8])**2, axis=1)
                     * np.sum(abs(prefix[:, 344:352])**2, axis=1))
    correlation = abs(overlap)/np.maximum(energy, 1e-30)
    usable = (abs(overlap) >= 1e-30) & (correlation >= LONG_OVERLAP_CORRELATION_MIN)
    if not np.any(usable):
        if diagnostics is not None:
            diagnostics['overlap_correlation'] = float(correlation.max())
        return 0., np.empty(0, complex), None, 'overlap'
    phase = overlap[usable]/abs(overlap[usable])
    symbols = np.concatenate((prefix[usable], tail[usable, 8:]*phase[:, None]), axis=1)
    powers = np.concatenate((prepared['power'][valid][usable], power[indices][usable, 8:]), axis=1)
    floor = max(prepared['power_floor'], float(power.max())*1e-9, 1e-30)
    fraction = _despread_fraction(symbols, powers, floor).mean(1)
    best = int(fraction.argmax())
    if diagnostics is not None:
        diagnostics['overlap_correlation'] = float(correlation[usable][best])
    return (float(fraction[best]), symbols[best],
            lo+round(starts[usable][best]*fs/p4rx.SAMPLE_RATE), 'prepared')


def robust_at(audio, fs, onset_s, cfo_hz):
    """Placed full robust short, including timing lock and field decode."""
    fraction, symbols, first = placed_despread(audio, fs, onset_s, cfo_hz, count=85)
    field = robust_symbols(symbols, 85)
    return dict(field, coherence=fraction, first_chip=first, cfo_hz=cfo_hz)


def word_at(
    audio, fs, origin, onset_s, *, cfo_hz, slack_s=0.035, whole=False, provisional=False, expected_words=None, diagnostics=None, gaps=(), carrier_offsets=None, reacquire=False, work=None
):
    """Locate near the predicted onset, at body CFO + each word's carrier offset.

    carrier_offsets can preserve an uncalibrated word reference during entry.
    """
    if work is not None and not work.admits('refine' if cfo_hz is None else 'head'):
        return None
    x = np.asarray(audio)
    lo = max(0, round((onset_s - slack_s) * fs))
    # Keep the resampler phase independent of the predicted response centre.
    phase = fs // gcd(fs, p4rx.SAMPLE_RATE)
    lo = max(0, lo - (origin + lo) % phase)
    hi = min(len(x), round((onset_s + 0.240 + slack_s) * fs))
    if hi - lo < 0.10 * fs:
        return None
    seg = x[lo:hi]
    acquisition = cfo_hz is None and provisional and expected_words is not None
    if acquisition:
        cfo_hz = 0.
    if cfo_hz is None:
        fraction, symbols, carrier, first, ratios = _lock(seg, fs)
        name, score = p4words.match(symbols)
        if name is None or len(p4words.BANK[name]) != 27:
            return None
        _, shift, residual = p4words.correlate(p4words.BANK[name], symbols)
        carrier += residual
        first = lo + round(first - shift * fs / p4words.SYMBOL_RATE)
    else:
        carrier = float(cfo_hz)
        z = p4rx.audio_to_baseband(seg, fs)
        tuned = z * np.exp(-2j * np.pi * carrier * np.arange(len(z)) / p4rx.SAMPLE_RATE)
        chips = p4rx.matched_filter(tuned)[
            p4rx.MATCHED_DELAY : len(z) + p4rx.MATCHED_DELAY
        ]
        count = 20 if provisional else 27
        span = count * _SYMBOL_N
        if len(chips) < span:
            return None
        # All sample phases in the small onset bracket, with one carrier.
        spread = fftconvolve(chips, _SPREAD_KERNEL)[_SYMBOL_N - 1 :]
        power = fftconvolve(abs(chips) ** 2, abs(_SPREAD_TAPS[::-1]) ** 2)[
            _SYMBOL_N - 1 :
        ]
        starts = np.arange(
            min(len(chips) - span + 1, round(2 * slack_s * p4rx.SAMPLE_RATE) + 1)
        )
        indices = starts[:, None] + _SYMBOL_N * np.arange(count)
        fractions = np.divide(
            abs(spread[indices]) ** 2 * _SPREAD_ENERGY,
            np.maximum(power[indices], 1e-30),
            out=np.zeros(indices.shape),
            where=power[indices] > max(float(power.max()) * 1e-9, 1e-30),
        )
        ranks = fractions.mean(axis=1)
        recovery_ranks = ranks.copy() if reacquire else None

        def recover():
            if not reacquire:
                return None
            if work is not None and not work.admits('refine'):
                return None
            # Reuse the failed despread at six separated timing peaks. The
            # wide bank reports word residual Hz, not a body-carrier offset.
            knee = PLACED_FRACTION_KNEE if expected_words is not None else ROBUST_FRACTION_KNEE
            peaks = []
            for _ in range(6):
                at = int(recovery_ranks.argmax())
                if recovery_ranks[at] < knee:
                    break
                peaks.append(at)
                recovery_ranks[max(0, at-8):at+9] = 0
            if not peaks:
                return None
            fitted, scores = p4words.placed_scores(spread[indices[peaks]], acquisition=True)
            offsets = p4words.CARRIER_HZ if carrier_offsets is None else carrier_offsets
            for candidate in np.argsort(scores)[::-1]:
                name, shift, hz = fitted[candidate]
                if scores[candidate] < p4words.MATCH_KNEE:
                    break
                if abs(hz) >= p4words.MATCH_HZ-p4words.MATCH_STEP_HZ/2:
                    continue
                found = word_at(x, fs, origin, onset_s,
                    cfo_hz=carrier+hz-offsets.get(name, 0.),
                    slack_s=slack_s, whole=whole, provisional=provisional,
                    expected_words=expected_words, diagnostics=diagnostics,
                    gaps=gaps, carrier_offsets=carrier_offsets, work=work)
                return found if (found is not None and found.word == name
                                 and found.score >= p4words.MATCH_KNEE) else None
            return None

        best = None
        if expected_words is not None:
            eligible = np.flatnonzero(ranks >= PLACED_FRACTION_KNEE)
            if not len(eligible):
                return None
            candidates = spread[indices[eligible]]
            fitted, scores = (p4words.placed_scores(candidates, acquisition=True) if acquisition
                              else p4words.placed_scores(candidates, carrier_offsets=carrier_offsets))
            for candidate in np.argsort(scores)[::-1]:
                at = eligible[candidate]
                name, shift, hz = fitted[candidate]
                score = float(scores[candidate])
                if score < (p4words.MATCH_KNEE if acquisition else PLACED_MATCH_KNEE):
                    break
                limit = p4words.MATCH_HZ-p4words.MATCH_STEP_HZ/2 if acquisition else 2.75
                if ranks[at] < PLACED_FRACTION_KNEE or abs(hz) >= limit:
                    continue
                if acquisition:
                    found = word_at(x, fs, origin, onset_s, cfo_hz=hz-p4words.CARRIER_HZ.get(name, 0.), slack_s=slack_s,
                                    whole=whole, provisional=True, expected_words=expected_words,
                                    diagnostics=diagnostics, gaps=gaps, work=work)
                    return found if (found is not None and found.word == name
                                     and found.score >= p4words.MATCH_KNEE) else None
                if name not in expected_words and score < p4words.MATCH_KNEE:
                    if diagnostics is not None:
                        diagnostics.append(dict(event='unexpected weak word', word=name,
                                                score=score, at=origin+round(onset_s*fs)))
                    return recover()
                best = score, name, shift, int(starts[at]), float(ranks[at])
                break
        else:
            for _ in range(6):
                if work is not None and not work.checkpoint('detect'):
                    return None
                at = int(ranks.argmax())
                fraction = float(ranks[at])
                if fraction < ROBUST_FRACTION_KNEE:
                    break
                symbols = spread[indices[at]]
                name, score, shift = p4words.match_placed(symbols, cfo_hz=carrier, carrier_offsets=carrier_offsets)
                if name and (best is None or score > best[0]):
                    best = score, name, shift, int(starts[at]), fraction
                ranks[max(0, at - 8) : at + 9] = 0
        if best is None:
            return recover()
        score, name, shift, at, fraction = best
        aligned = spread[at + _SYMBOL_N * np.arange(count)]
        reference = p4words.BANK[name][max(0, shift) :]
        aligned = aligned[max(0, -shift) :]
        n = min(len(reference), len(aligned))
        residual_phase = aligned[:n] * reference[:n].conj()
        steps = residual_phase[1:] * residual_phase[:-1].conj()
        steps /= np.maximum(abs(steps), 1e-30)
        carrier += float(np.angle(steps.sum()) * p4words.SYMBOL_RATE / (2 * np.pi))
        first = lo + round(
            at * fs / p4rx.SAMPLE_RATE - shift * fs / p4words.SYMBOL_RATE
        )
    if fraction < (PLACED_FRACTION_KNEE if expected_words is not None else ROBUST_FRACTION_KNEE):
        return None
    return Observation(
        Kind.WORD,
        fs,
        origin,
        first,
        origin + first,
        0.240,
        carrier,
        fraction,
        placed=True,
        weak=score < p4words.MATCH_KNEE,
        unexpected=expected_words is not None and name not in expected_words,
        provisional=provisional,
        symbols_read=20 if provisional else 27,
        word=name,
        score=score,
        shift=shift,
        extent_s=_extent(x, fs, first / fs, word=name, gaps=gaps) if whole and not provisional else None,
    )


def body_after(audio, fs, origin, obs, *, work=None):
    """Decode only behind a full Y27, rebasing its absolute onset into this window."""
    from . import p4breakin

    if work is not None and not work.admits('body'):
        return None
    if obs.word != "Y27" or obs.provisional:
        return None
    start = round((obs.stream_sample - origin) * p4rx.SAMPLE_RATE / fs) + 432 * 16
    if start < 0:
        return None
    z = p4rx.audio_to_baseband(np.asarray(audio), fs)
    return p4breakin.decode_body(
        z, first_training_start=start, frequency_offset=obs.cfo_hz
    )


def body_carrier(audio, fs, origin, head):
    """Compare Y27 tone aliases using body training, without a trellis."""
    from . import p4breakin

    lo = max(0, head.stream_sample-origin-round(.050*fs))
    hi = min(len(audio), head.stream_sample-origin+round(.820*fs))
    x = np.asarray(audio)[lo:hi]
    origin += lo
    z = p4rx.audio_to_baseband(x, fs)

    def quality(obs):
        start = round((obs.stream_sample-origin)*p4rx.SAMPLE_RATE/fs) + 432*16
        if start < 0:
            return 0.
        # Keep the decoder's existing absolute Y27 seed. Only the placed
        # word reader takes the body reference introduced by placed-carrier.
        scores = p4breakin.body_scores(z, first_training_start=start,
                                       frequency_offset=obs.cfo_hz)
        return 0. if scores is None else float(np.mean(abs(scores)))

    baseline = quality(head)
    best, best_q = head, baseline
    carrier = head.cfo_hz-p4words.CARRIER_HZ['Y27']
    for delta in (-p4words.SYMBOL_RATE, p4words.SYMBOL_RATE):
        candidate = word_at(x, fs, origin, (head.stream_sample-origin)/fs,
                            cfo_hz=carrier+delta, expected_words={'Y27'})
        if candidate is None or candidate.word != 'Y27' or candidate.weak:
            continue
        q = quality(candidate)
        if q > max(best_q, 1.5*baseline):
            best, best_q = candidate, q
    return best
