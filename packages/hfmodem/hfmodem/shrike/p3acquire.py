# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Acquire P3 entry controls and changeovers across answer phase and RX offset.

A short head is a candidate, not permission to change roles: the caller must
corroborate it on a separate cycle. A complete CRC-confirmed body stands alone.

A `Candidate` reports TWO frequencies and they are not the same quantity.
`coarse_hz` is the hypothesis whose control and extent were qualified, drawn
from the list the caller swept, and the one a changeover body is read at first;
a body that will not decode there is tried once more at `offset_hz`, with the
CRC as the acceptor. `offset_hz` is the peer's carrier offset MEASURED off that
hypothesis, to about a tenth of a hertz -- the number `onair` stores as the
session's receive offset, corrects its tracked reads by, and keys our own
PACTOR-3 on. Nothing here decides that last part; since 2026-09-13 the
transmitter follows the session offset (`onair.RadioTx._p3_offset`), which is
what makes the difference between a hypothesis and a measurement matter.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fft import fft, ifft, irfft, next_fast_len, rfft
from scipy.signal import fftconvolve, hilbert, resample_poly

from . import placement, rx, rxfront, spec

FS = spec.SAMPLE_RATE
SPS = FS // 100
OFFSETS_HZ = (0.0, -25.0, 25.0, -50.0, 50.0, -75.0, 75.0)
CONTROL_OFFSETS_HZ = tuple(float(hz) for hz in range(-75, 76, 5))
ENTRY_CONTROL_OFFSETS_HZ = tuple(float(hz) for hz in range(-100, 101, 5))
SEARCH_FS = 8000
DECIMATION = FS // SEARCH_FS
STEP = SEARCH_FS // 800
"""The search's alignment step: an eighth of a symbol, 10 samples at 8 kHz."""

P1_CROSS_CHECK_HZ = 60.0
"""How far off nominal a station can be and still deliver a PACTOR-1 codeword.

The PACTOR-1 reader searches time, never frequency: `p1rx.decode_control_signal`
correlates against MARK 1400 and SPACE 1600 Hz and nothing else. MEASURED by
shifting a rendered control signal through `compensate` in 2.5 Hz steps: all four
codewords read at zero of twelve bit errors out to +/-57.5 Hz, the first one
fails at 60, and none reads past 77.5. The span is asymmetric and which way
depends on the word's own mark/space balance -- CS0 and CS2 give out going up,
CS1 and CS3 going down -- so the bound is the first failure and not the last
success. So a zero-error PACTOR-1 codeword IS a frequency measurement of
the station that sent it, to this bound, taken by an instrument with no
differential ambiguity at all -- which is what the PACTOR-3 acquisition has
(`_refined`). Audited against a synthetic control at a known offset with a
deliberate per-symbol phase convention: the estimator reports a phi-per-symbol
convention as phi/3.6 Hz added to the true offset, so the codeword survives and
only the frequency is wrong.

Noise narrows the real span and cannot widen it, so a follow further out than
this contradicts the PACTOR-1 read outright rather than merely disagreeing with
it. `onair._SessionRx._follow_p3_offset` is the only caller.
"""

FINE_LIMIT_HZ = 12.5
"""How far `_refined` may move a hypothesis off the grid it was found on.

Half `OFFSETS_HZ`'s step: past that, a different grid point is the better
hypothesis and a residual that large is a bad measurement rather than a distant
peer. The refinement is a correction to the hypothesis that decoded, not a
search of its own.
"""


def compensate(audio: np.ndarray, hz: float) -> np.ndarray:
    """Shift `audio`'s spectrum DOWN by `hz`, through the analytic signal.

    ON A FAST TRANSFORM LENGTH, because the caller is a receive window whose
    length is whatever the cycle collected and `hilbert` is one FFT pair over
    all of it. `onair-0913-1629`'s own windows: 106646 samples = 2 x 41 x 1301
    costs 4.8-7.1 ms and 45974 = 2 x 127 x 181 costs 1.8 ms, against 2.0 and
    0.8 ms padded to `next_fast_len` and trimmed straight back -- 2.2x to 3.6x,
    and it is spent TWICE a cycle, in front of a key that had about 8 ms of
    notice. Fifteen of that arm's slots went to overruns of 4.2-10.2 ms.

    NEITHER LENGTH IS THE TRUTH and that is why the pad is admissible: a finite
    window's Hilbert transform is circular either way, so both forms wrap, and
    they differ by 3-8% of peak in the first and last 200 samples and 0.1-0.8%
    between. What has to be unchanged is what the readers RETURN, and that is
    measured rather than assumed -- 768 brackets of the fixture corpus through
    `control_signal` and `changeover` at 36 detections, and every hold window of
    the three 2026-09-13 WS8EOC arms through `rxfront.decode_expected_packet` and
    `changeover`, 920 reads. Zero disagreements, quality bit-identical.
    `tests.shrike.test_offset_fine_estimate` keeps the corpus half in the tree.

    HALF THE TRANSFORMS, AND NO LONG EXPONENTIAL. The analytic signal's real
    part is the audio itself, so the shift is `x cos(wn) + H{x} sin(wn)`: only
    the Hilbert transform `H{x}` wants a transform pair, a real one at half the
    cost of `hilbert`'s complex pair, and the carrier is the outer product of
    two short ones. 0.4 ms against 0.8 on a 0.87 s pre-key window, 1.7 against
    3.3 on a 3.75 s one, and the same samples to 2e-13 of a 0.4 peak.
    """
    if not hz:
        return audio
    x = np.asarray(audio, dtype=float)
    n = next_fast_len(len(x))
    spectrum = rfft(x, n)
    spectrum[0] = 0
    if n % 2 == 0:
        spectrum[-1] = 0
    quadrature = irfft(spectrum * -1j, n)[:len(x)]
    carrier = _carrier(hz, len(x))
    return x * carrier.real - quadrature * carrier.imag


def _carrier(hz: float, n: int, block: int = 1024) -> np.ndarray:
    """`exp(-2j pi hz m / FS)` for m < n, as blocks of one short table."""
    w = -2j * np.pi * hz / FS
    blocks = np.exp(w * block * np.arange(-(-n // block)))
    return np.outer(blocks, np.exp(w * np.arange(block))).ravel()[:n]


def packet_offset(audio: np.ndarray, ev: rxfront.Event) -> float | None:
    """Qualify an SL2 carrier from the CRC-known field's coherent phase slope."""
    if (ev.packet is None or ev.packet[0] != 2 or not ev.packet[3]
            or ev.cycle_long or ev.information is None
            or ev.carrier_swapped is None):
        return None
    path = placement.SPEED_PATHS[2]
    steps = placement.grid_steps(placement.build_grid(
        ev.information, path, flush=placement.DATA_FLUSH), path)
    tones = tuple(spec.CARRIER_SWAP[c] if ev.carrier_swapped else c
                  for c in path.tones)
    offsets = path.clock_offsets(SPS)
    idx = ev.start + np.arange(path.n_symbols) * SPS + rxfront.MATCHED_DELAY_N
    if idx[0] < 0 or idx[-1] + max(offsets) >= audio.size:
        return None
    z = rxfront._sampled_baseband(
        audio, tones, np.unique(np.add.outer(idx, offsets).ravel()))
    wiped = np.stack([z[c][idx + off] * np.exp(-1j * np.cumsum(steps[home]))
                      for c, home, off in zip(tones, path.tones, offsets)])
    hz = np.arange(-40., 40.01, .1)
    corr = wiped @ np.exp(-2j * np.pi * np.arange(path.n_symbols)[:, None]
                         / 100 * hz)
    score = np.sum(np.abs(corr) ** 2, axis=0)
    best = int(score.argmax())
    energy = path.n_symbols * np.sum(np.abs(wiped) ** 2)
    if energy == 0 or score[best] / energy < .55 or best in (0, hz.size - 1):
        return None
    return float(hz[best])


@dataclass
class Candidate:
    event: rxfront.Event
    offset_hz: float
    """The peer's carrier offset, refined off the grid to about 0.1 Hz."""
    quality: float
    coarse_hz: float
    """The coarse hypothesis at which the control and extent were qualified."""


def changeover(audio: np.ndarray, *, offsets=OFFSETS_HZ) -> Candidate | None:
    """Search complete CS3 heads only; report body CRC when enough audio exists.

    The 2026-09-08 WS8EOC head precedes the old P1 anchor by ~100 ms and needs
    RX frequency compensation. A ±40-ms tracked read at that old anchor cannot
    acquire it. Search the supplied receive window, with zero bit errors at
    adjacent alignments, a whole-burst extent check and coherent soft symbols.
    """
    return _acquire(audio, offsets, (placement.BREAKIN_CS,))


def control_signal(audio: np.ndarray, *, offsets=CONTROL_OFFSETS_HZ,
                   preferred_hz: float | None = None,
                   preferred_at: int | None = None) -> Candidate | None:
    """Acquire any P3 answer during entry, including a bare acknowledgement.

    VE3KPG and KB5LZK on 2026-09-10 answered entry with CS1, not a
    changeover. KB5LZK needs a finer frequency hypothesis near -65 Hz.
    The caller must corroborate body-less candidates on distinct cycles.
    A preferred frequency only ranks independently qualified candidates; it
    cannot lower the coherent, adjacent-alignment or whole-burst requirements.
    """
    return _acquire(audio, offsets, range(len(rx._CS_TABLE)),
                    preferred_hz=preferred_hz, preferred_at=preferred_at)


AIM_BITS = 12
"""Differential bits an aim reads: the head's first thirteen symbols, and the
main lobe of the last (`AIM_S` of audio behind the head).

Our first bridge stops where these bits reach every head whose fourteen
(`CUT_WORD_BITS`) the read at the key holds, and a gateway's answer to our
entry starts 120-138 ms after the entry ends. On the FTDX10 that bridge ends
1171.6 ms past our boundary, and the heads of 1003-0000, -0008, -0010, -0012 and
the 10-10 changeover start 968-986 ms past it, so they show 18-21 of their 21
symbols. Twelve bits plus the last symbol's main lobe reach a head that starts
as late as 1046 ms.
"""


AIM_S = (AIM_BITS * (SEARCH_FS // 100) + SEARCH_FS // 200 + 1) / SEARCH_FS


def head_aims(audio: np.ndarray, *, offsets=ENTRY_CONTROL_OFFSETS_HZ,
              bits: int = AIM_BITS, limit: int = 2) -> list[tuple[int, float]]:
    """Where a P3 control starts in `audio` and at what frequency, read on the
    first `bits` of its word: an aim for one tracked read of the whole word,
    not a decode.

    The search `control_signal` runs, on every frequency, with the same accept:
    zero errors on two neighbouring alignments, coherent on both carriers, all
    of it inside one burst. The word the bits belong to is not reported. Up to
    `limit` (start, offset_hz) pairs at the capture rate, best first.
    """
    if len(audio) > 4*FS or not np.any(audio):
        return []
    sps = SEARCH_FS // 100
    found = _search(audio, offsets, range(len(rx._CS_TABLE)), bits + 1,
                    bits*sps + sps//2 + 1)
    if found is None:
        return []
    hits = sorted(((_ranked(found.qualities[trial, k]), int(trial), int(k))
                   for trial, k in zip(*np.nonzero(found.accepted))), reverse=True)
    aims, envelopes = [], {}
    slack = rxfront.CS_EXTENT_SLACK * sps
    for _, trial, k in hits:
        if trial not in envelopes:
            # `rxfront._tone_envelope` at the search rate: one symbol's mean
            # magnitude, centred on the symbol it is indexed by.
            mag = sum(np.abs(band) for band in found.band(trial).values())
            c = np.concatenate([[0.0], np.cumsum(mag[found.delay - sps//2:])])
            envelopes[trial] = (c[sps:] - c[:-sps]) / sps
        env, a = envelopes[trial], int(found.starts[k])
        b = a + bits*sps
        body = env[a + slack:b + 1 - slack]
        if not body.min() > rxfront.CS_EXTENT_FLOOR * env[a:b + 1].max():
            continue
        start = a*DECIMATION
        hz = _refined(found.diffs, found.trial_signs, trial, k, offsets[trial])
        if any(abs(start - s) < SPS and abs(hz - h) <= 10 for s, h in aims):
            continue
        aims.append((start, hz))
        if len(aims) == limit:
            break
    return aims


CUT_WORD_BITS = 14
"""The fewest bits a word cut short by the window's end is read on: every two
codewords differ in at least six of their first fourteen. A head starting as
late as 1045 ms past our boundary shows that many to the FTDX10's read."""


def cut_word(audio: np.ndarray, at: int, hz: float, *,
             bits_min: int = CUT_WORD_BITS) -> int | None:
    """The codeword starting within a symbol of `at`, at `hz`, read on as many of
    its bits as `audio` holds, and no fewer than `bits_min`: for a word whose
    last symbols fall behind our key. The search's accept, on one frequency."""
    sps = SEARCH_FS // 100
    lo = max(0, at - SPS)
    bits = min(spec.CS_BITS_PER_TONE,
               ((len(audio) - at) // DECIMATION - sps//2 - 1) // sps)
    if bits < bits_min or not np.any(audio[lo:]):
        return None
    found = _search(audio[lo:], (float(hz),), range(len(rx._CS_TABLE)), bits + 1,
                    bits*sps + sps//2 + 1)
    if found is None:
        return None
    near = [k for k in np.flatnonzero(found.accepted[0])
            if abs(lo + int(found.starts[k])*DECIMATION - at) <= SPS]
    if not near:
        return None
    return int(found.cis[0, max(near, key=lambda k: _ranked(found.qualities[0, k]))])


def _bands(analytic: np.ndarray, clock: np.ndarray, offsets,
           pulse: np.ndarray) -> dict[int, np.ndarray]:
    """`rx._baseband` for every frequency hypothesis, one call per channel.

    The same matched filter on the same samples, stacked along one axis: the
    per-hypothesis cost was the call and not the arithmetic, 0.32 ms of the
    0.40 ms a trial took over 3680 samples and a 310-tap pulse. Thirty-one of
    them measured 13.9 ms that way against the 8.0 ms of notice a 40 ms settle
    leaves past `key_notice`, which is what the 2026-09-10 entry arms lost
    every other slot to.

    One FFT rather than `rx._baseband`'s overlap-add, for the reason that
    function gives for not using `np.convolve`: at this length and tap count
    the block machinery is the cost. It is not bit-identical and the accept
    below is a sign test, so the difference is measured rather than assumed --
    1.7e-15 of a sample and 5.4e-15 of peak, worst of 127 brackets across the
    fixture corpus, which is the order `rx._baseband` claims against the
    direct form. What the search RETURNS is unchanged over that corpus:
    codeword, position, instant, offset and body, 678 comparisons and 28
    detections, in `tests.shrike.test_slot_deadline`.
    """
    shifted = (analytic[None, :]
               * np.exp(-2j*np.pi*np.asarray(offsets, float)[:, None]
                        * clock[None, :])).real
    return {cn: fftconvolve(shifted * rx.carrier(cn, SEARCH_FS, 0, shifted.shape[1]),
                            pulse[None, :], axes=1)
            for cn in spec.VH_CHANNELS}


BIN_HZ = 5.0
"""The frequency step `_grid` places on a transform bin: every list a caller
sweeps is on it, and a hypothesis off it costs one transform of its own."""


def _fold_length(n: int) -> int:
    """The fast transform length of at least `n` whose bins hold `BIN_HZ`."""
    unit = round(SEARCH_FS / BIN_HZ)
    q = -(-n // unit)
    while next_fast_len(unit*q) != unit*q:
        q += 1
    return unit*q


@lru_cache(maxsize=16)
def _fold_weight(nfft: int, phase: int) -> np.ndarray:
    """The matched filter's spectrum, turned to read sample `phase` of every
    `STEP` and scaled for the real part's half and the fold's `STEP` terms."""
    m = np.arange(nfft)
    w = (fft(rx._pulse(SEARCH_FS // 100), nfft)
         * np.exp(2j*np.pi*m*phase/nfft) / (2*STEP)).reshape(STEP, -1)
    w.flags.writeable = False
    return w


def _grid(analytic: np.ndarray, offsets, first: int,
          count: int) -> dict[int, np.ndarray]:
    """`_bands(...)[cn][:, first::STEP][:, :count]`, without the rest of it.

    The search reads its bands at one sample of every `STEP` and nowhere else,
    so the nine between are never needed. In the transform domain both the
    hypothesis and the carrier are bin shifts when they sit on bins -- a 5 Hz
    grid and the 120 Hz tone raster on a length `_fold_length` makes them so
    -- the real part `_bands` takes is the spectrum plus its mirrored conjugate,
    and sampling one in `STEP` is the spectrum folded `STEP` times onto a
    transform `STEP` times shorter. So one forward transform serves every
    hypothesis on the grid, each row is two shifted reads of it, a multiply and
    a fold, and the inverse is a tenth of the length.

    Exact rather than approximate: the same circular convolution `fftconvolve`
    takes, the transform length covering the full linear output. Against
    `_bands` it agrees to 4e-12 of peak over a 4 s window, the residue being the
    `np.exp` that `_bands` evaluates at up to 2500 radians, not the transforms;
    `tests.shrike.test_p3_acquisition_cost` holds it.
    """
    n = len(analytic)
    nfft = _fold_length(n + len(rx._pulse(SEARCH_FS // 100)) - 1)
    weight = _fold_weight(nfft, first % STEP)
    width = nfft // STEP
    spectrum = fft(analytic, nfft)
    twice, mirror = (np.concatenate([spectrum, spectrum]),
                     np.tile(np.conj(np.roll(spectrum[::-1], 1)), 2))
    rows = {cn: np.empty((len(offsets), width), complex)
            for cn in spec.VH_CHANNELS}
    for i, hz in enumerate(offsets):
        shift = hz * nfft / SEARCH_FS
        if abs(shift - round(shift)) < 1e-9:
            line, image, shift = twice, mirror, round(shift)
        else:
            moved = fft(analytic * np.exp(-2j*np.pi*hz*np.arange(n)/SEARCH_FS),
                        nfft)
            line, image, shift = (np.concatenate([moved, moved]),
                                  np.tile(np.conj(np.roll(moved[::-1], 1)), 2),
                                  0)
        for cn in spec.VH_CHANNELS:
            tone = round(spec.channel_freq_hz(cn) * nfft / SEARCH_FS)
            a, b = (shift + tone) % nfft, (tone - shift) % nfft
            z = line[a:a + nfft] + image[b:b + nfft]
            rows[cn][i] = np.sum(z.reshape(STEP, width) * weight, axis=0)
    at = first // STEP
    return {cn: ifft(v, axis=1)[:, at:at + count] for cn, v in rows.items()}


def _refined(diffs: np.ndarray, trial_signs: np.ndarray, trial: int, k: int,
             hz: float) -> float:
    """`hz` corrected by the residual carrier rotation the word itself carries.

    The grid is the coarse list's 25 Hz or the control list's 5, and the follow
    keys our own controls at whatever it returns, so a hypothesis is not a
    measurement. The measurement is already sitting in `diffs`: one symbol apart,
    a differential product's phase is the data's 0 or pi plus 2.pi.df.Tsym, and
    `trial_signs` is what takes the data back off. Both carriers of a word are
    displaced by the same df, so the twenty pairs and the two channels sum
    coherently into one angle -- forty complex terms out of an array the accept
    tensors already built, which is why this costs nothing a cycle can feel.

    Tsym is 10 ms, so the residual is unambiguous over +/-50 Hz and the grid it
    corrects is finer than that either way. Before the rounding below it reads
    0.04 Hz off our own transmitter's DAC reference at a declared offset, over
    all 38 controls of `captures/onair-0913-1629`; on WS8EOC's eleven CS3 heads
    in the same capture it reads -26.0 Hz with a 0.56 Hz spread, where the coarse
    grid said -25.
    """
    residual = (np.angle(np.sum(diffs[:, trial, k] * trial_signs[trial, k]))
                * SEARCH_FS / (2*np.pi*(SEARCH_FS//100)))
    # To a tenth of a hertz: finer than the estimator's own spread, and it keeps
    # the offset a number the TX and RX lines can print and a scene can name.
    return round(hz + float(np.clip(residual, -FINE_LIMIT_HZ, FINE_LIMIT_HZ)), 1)


RANK_DIGITS = 9
"""The decimal places a coherent quality is ranked to.

Past them two scores are one: identical copies of a word a cycle apart score
alike to 1e-15, and which wins was rounding -- that of whatever transform
length and summation order the search happened to use. So ranking stops at
1e-9, far below any difference a channel makes, and a tie keeps the order
each caller gave an exact one: the later alignment and higher hypothesis
first, `cut_word`'s earliest."""


def _ranked(quality: float) -> float:
    return round(float(quality), RANK_DIGITS)


class _Signs:
    """`signs[cis]` read one alignment at a time, which is all `_refined` asks
    of it: the dense [offsets, starts, bits] table was a 20 MB copy a cycle."""

    def __init__(self, signs: np.ndarray, cis: np.ndarray):
        self.signs, self.cis = signs, cis

    def __getitem__(self, key):
        return self.signs[self.cis[key]]


_BYTE_BITS = np.array([bin(b).count("1") for b in range(256)], np.uint8)


def _popcount(words: np.ndarray) -> np.ndarray:
    """Set bits of each word below 2**24, a byte at a time: `np.bitwise_count`
    is NumPy 2 and the package still installs on 1.24."""
    return (_BYTE_BITS[words & 0xFF] + _BYTE_BITS[(words >> 8) & 0xFF]
            + _BYTE_BITS[words >> 16])


@dataclass
class _Search:
    """What the vectorised search accepted, on the 8 kHz alignment grid.

    `qualities` holds the coherent score where an alignment passed the bit
    test on two neighbours and zero elsewhere: nothing reads it anywhere else.
    """
    starts: np.ndarray
    accepted: np.ndarray
    cis: np.ndarray
    qualities: np.ndarray
    diffs: np.ndarray
    trial_signs: _Signs
    analytic: np.ndarray
    offsets: tuple
    delay: int

    def band(self, trial: int) -> dict[int, np.ndarray]:
        """`_bands` of one hypothesis at the full 8 kHz, for an extent check."""
        clock = np.arange(len(self.analytic)) / SEARCH_FS
        got = _bands(self.analytic, clock, (self.offsets[trial],),
                     rx._pulse(SEARCH_FS // 100))
        return {cn: v[0] for cn, v in got.items()}


def _search(audio: np.ndarray, offsets, codes, symbols: int,
            reach: int) -> _Search | None:
    """Every frequency and eighth-symbol alignment holding `symbols` symbols
    and `reach` samples of the 8 kHz window behind the first, scored at once.

    Search at 8 kHz (both carriers remain below Nyquist); a caller validates
    extent and body at the original rate. The search runs after the live bridge
    and must fit the few milliseconds remaining before key notice.

    EVERY ALIGNMENT IS A WHOLE NUMBER OF STEPS FROM EVERY OTHER, and a symbol is
    eight of them, so the products one symbol apart are one line per hypothesis
    and carrier (`_grid`), and the [starts, bits] tensors are views of it.
    The bit test is a popcount of the packed sign word against each codeword's,
    in integers, so on the same products it decides what the dense tensors
    did; the coherent scores are taken only where the bit test passed on two
    neighbours, which is the only place they decide anything. Held against
    the dense form over every fixture in the tree by
    `tests.shrike.test_p3_acquisition_cost`.
    """
    small = resample_poly(np.asarray(audio, dtype=float), 1, DECIMATION)
    analytic = hilbert(small)
    search_sps = SEARCH_FS // 100
    delay = (len(rx._pulse(search_sps))-1)//2
    starts = np.arange(0, len(small)-reach, STEP)
    if not len(starts):
        return None
    hop, bits, n = search_sps // STEP, symbols - 1, len(starts)
    table = rx._CS_TABLE[list(codes)][:, :bits]
    signs = 1.0-2.0*table
    grid = _grid(analytic, offsets, delay, n + hop*bits)
    line = np.asarray([grid[cn][:, hop:]*np.conj(grid[cn][:, :-hop])
                       for cn in spec.VH_CHANNELS])
    s0, s1, s2 = line.strides
    diffs = as_strided(line, (len(spec.VH_CHANNELS), len(offsets), n, bits),
                       (s0, s1, s2, hop*s2), writeable=False)
    negative = (line[0].real + line[1].real) < 0
    word = np.zeros((len(offsets), n), np.uint32)
    for j in range(bits):
        word |= negative[:, j*hop:j*hop + n].astype(np.uint32) << j
    codewords = np.sum(table.astype(np.uint32) << np.arange(bits, dtype=np.uint32),
                       axis=1, dtype=np.uint32)
    errors = _popcount(word[:, None, :] ^ codewords[None, :, None])
    cis = np.argmin(errors, axis=1)
    exact = np.take_along_axis(errors, cis[:, None], axis=1)[:, 0] == 0
    # At least two neighbouring timing hypotheses must agree; an isolated
    # perfect word among thousands of trials is not a decoded head.
    paired = exact[:, :-1] & exact[:, 1:] & (cis[:, :-1] == cis[:, 1:])
    adjacent = np.zeros_like(exact)
    adjacent[:, 1:] |= paired
    adjacent[:, :-1] |= paired
    adjacent &= exact
    qualities = np.zeros(exact.shape)
    accepted = np.zeros_like(exact)
    trial, k = np.nonzero(adjacent)
    if len(trial):
        held = diffs[:, trial, k]
        wiped = signs[cis[trial, k]]
        magnitude = np.abs(held)
        quality = np.sum(held.real.sum(axis=0)*wiped, axis=1) / np.maximum(
            np.sum(magnitude, axis=(0, 2)), 1e-30)
        carrier_quality = np.sum(held.real*wiped, axis=2) / np.maximum(
            np.sum(magnitude, axis=2), 1e-30)
        qualities[trial, k] = quality
        accepted[trial, k] = ((quality >= .65)
                              & (carrier_quality.max(axis=0) >= .85))
    return _Search(starts, accepted, cis, qualities, diffs,
                   _Signs(signs, cis), analytic, tuple(offsets), delay)


def _inside_burst(shifted: np.ndarray, at: int) -> bool:
    """`rxfront._inside_burst` on the whole window's envelope, at `at`.

    Filtered over only the span that envelope reads there: the full-rate
    matched filter is 1860 taps and the test reads 21 symbols of it, so
    filtering the whole window per hypothesis was half of what a changeover
    spent once the search itself was cheap -- 52 of 104 ms over five 2.5 s
    reads. The samples it reads are the whole window's to rounding: the slice
    starts a filter length ahead of the first and runs one sample past the
    last, and a magnitude does not see the carrier's starting phase.
    """
    n, sps = len(shifted), SPS
    if at < 0 or at + 20*sps >= n:
        return False
    pulse = rx._pulse(sps)
    delay = (len(pulse)-1)//2
    first = at + delay - sps//2
    last = first + 21*sps
    lo = max(0, first - (len(pulse)-1))
    z = {cn: rx._baseband(shifted[lo:min(n, last + 1)], cn, FS, pulse)
         for cn in spec.VH_CHANNELS}
    env = rxfront._tone_envelope(z, delay, at - lo, at - lo + 20*sps + 1)
    return rxfront._inside_burst(env, 0, 0)


def _acquire(audio: np.ndarray, offsets, codes, *,
             preferred_hz: float | None = None,
             preferred_at: int | None = None) -> Candidate | None:
    if len(audio) < round(.23*FS) or not np.any(audio):
        return None
    # A caller passing a long recording must slice it into receive windows.
    if len(audio) > 4*FS:
        return None
    found = _search(audio, offsets, codes, 21, round(.22*SEARCH_FS))
    if found is None:
        return None
    starts, accepted, cis, qualities, diffs, trial_signs = (
        found.starts, found.accepted, found.cis, found.qualities,
        found.diffs, found.trial_signs)
    best = None
    ranked = []
    for trial, hz in enumerate(offsets):
        good = np.flatnonzero(accepted[trial])
        if not len(good):
            continue
        ci, quality = cis[trial], qualities[trial]
        if len(codes) > 1:
            # Rank the cheap 8-kHz hypotheses before doing any full-rate
            # extent/body work. Fine frequency trials often find the same
            # physical word; validating each one wastes the live key reserve.
            ranked.extend((float(quality[k]), float(hz),
                           int(starts[k])*DECIMATION, codes[int(ci[k])],
                           _refined(diffs, trial_signs, trial, int(k), hz))
                          for k in good)
            continue
        # Highest coherent candidate first. A head has to occupy the measured
        # two-tone envelope, not straddle quiet padding or a different burst.
        shifted = compensate(audio, hz)
        for k in good[np.argsort(np.round(quality[good], RANK_DIGITS),
                                 kind="stable")[::-1]]:
            at = int(starts[k])*DECIMATION
            if not _inside_burst(shifted, at):
                continue
            fine = _refined(diffs, trial_signs, trial, int(k), hz)
            code, note = codes[int(ci[k])], f", acquisition RX {fine:+g} Hz"
            # The body is read at the hypothesis that was qualified, because
            # that is where the extent above was measured; the refinement is a
            # correction with no CRC behind it, so it gets a second trial only
            # when the first does not decode, and the CRC is the acceptor.
            ev = rxfront._cs_event(shifted, code, 0, at, at/FS, note)
            if ev.packet is None and fine != hz:
                refined = rxfront._cs_event(compensate(audio, fine), code, 0,
                                            at, at/FS, note)
                if refined.packet is not None:
                    ev = refined
            got = Candidate(ev, fine, float(quality[k]), float(hz))
            if ev.packet is not None:
                return got
            if best is None or _ranked(got.quality) > _ranked(best.quality):
                best = got
            break
    full_cache = {}
    # A 100-Hz alternative can score higher on an isolated bare word. Prefer
    # continuity only among hypotheses which already passed every soft/bit
    # gate, and still validate each one's physical extent below. Without a
    # retained frequency, keep the original quality ordering.
    ordering = ((lambda row: (_ranked(row[0]), *row[1:]))
                if preferred_hz is None and preferred_at is None else
                lambda row: (preferred_hz is not None and abs(row[4]-preferred_hz) <= 25.0,
                             preferred_at is not None and abs(row[2]-preferred_at) <= round(.020*FS),
                             _ranked(row[0]), *row[1:]))
    for quality, hz, at, ci, fine in sorted(ranked, key=ordering, reverse=True):
        if hz not in full_cache:
            full_cache[hz] = compensate(audio, hz)
        shifted = full_cache[hz]
        if _inside_burst(shifted, at):
            ev = rxfront._cs_event(shifted, ci, 0, at, at/FS,
                                   f", acquisition RX {fine:+g} Hz")
            return Candidate(ev, fine, quality, hz)
    return best
