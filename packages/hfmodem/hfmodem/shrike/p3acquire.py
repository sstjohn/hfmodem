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

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve, hilbert, resample_poly

from . import placement, rx, rxfront, spec

FS = spec.SAMPLE_RATE
SPS = FS // 100
OFFSETS_HZ = (0.0, -25.0, 25.0, -50.0, 50.0, -75.0, 75.0)
CONTROL_OFFSETS_HZ = tuple(float(hz) for hz in range(-75, 76, 5))
ENTRY_CONTROL_OFFSETS_HZ = tuple(float(hz) for hz in range(-100, 101, 5))
SEARCH_FS = 8000
DECIMATION = FS // SEARCH_FS

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
    """
    if not hz:
        return audio
    x = np.asarray(audio, dtype=float)
    analytic = hilbert(x, next_fast_len(len(x)))[:len(x)]
    return (analytic * np.exp(-2j*np.pi*hz*np.arange(len(x))/FS)).real


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
                   preferred_hz: float | None = None) -> Candidate | None:
    """Acquire any P3 answer during entry, including a bare acknowledgement.

    VE3KPG and KB5LZK on 2026-09-10 answered entry with CS1, not a
    changeover. KB5LZK needs a finer frequency hypothesis near -65 Hz.
    The caller must corroborate body-less candidates on distinct cycles.
    A preferred frequency only ranks independently qualified candidates; it
    cannot lower the coherent, adjacent-alignment or whole-burst requirements.
    """
    return _acquire(audio, offsets, range(len(rx._CS_TABLE)),
                    preferred_hz=preferred_hz)


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


def _acquire(audio: np.ndarray, offsets, codes, *,
             preferred_hz: float | None = None) -> Candidate | None:
    if len(audio) < round(.23*FS) or not np.any(audio):
        return None
    # A caller passing a long recording must slice it into receive windows.
    if len(audio) > 4*FS:
        return None
    # Search at 8 kHz (both carriers remain below Nyquist), then validate the
    # extent and body at the original rate. The search runs after the live
    # bridge and must fit the few milliseconds remaining before key notice.
    small = resample_poly(np.asarray(audio, dtype=float), 1, DECIMATION)
    analytic = hilbert(small)
    clock = np.arange(len(small)) / SEARCH_FS
    search_sps = SEARCH_FS // 100
    pulse = rx._pulse(search_sps)
    delay = (len(pulse)-1)//2
    starts = np.arange(0, len(small)-round(.22*SEARCH_FS), search_sps//8)
    if not len(starts):
        return None
    idx = starts[:, None] + np.arange(21)[None, :]*search_sps + delay
    table = rx._CS_TABLE[list(codes)]
    signs = 1.0-2.0*table
    best = None
    ranked = []
    # Every hypothesis at once; see `_bands` for what that costs and what
    # it is held to. The accept tensors stack with it.
    bands = _bands(analytic, clock, offsets, pulse)
    y = np.asarray([bands[cn][:, idx] for cn in spec.VH_CHANNELS])
    diffs = y[..., 1:]*np.conj(y[..., :-1])
    combined = diffs.real.sum(axis=0)
    errors = np.sum((combined[:, None] < 0) != table[None, :, None], axis=3)
    cis = np.argmin(errors, axis=1)
    trials, rows = np.arange(len(offsets))[:, None], np.arange(len(starts))
    exact = errors[trials, cis, rows[None, :]] == 0
    # At least two neighbouring timing hypotheses must agree; an isolated
    # perfect word among thousands of trials is not a decoded head.
    paired = exact[:, :-1] & exact[:, 1:] & (cis[:, :-1] == cis[:, 1:])
    adjacent = np.zeros_like(exact)
    adjacent[:, 1:] |= paired
    adjacent[:, :-1] |= paired
    adjacent &= exact
    trial_signs = signs[cis]
    qualities = np.sum(combined*trial_signs, axis=2) / np.maximum(
        np.sum(np.abs(diffs), axis=(0, 3)), 1e-30)
    carrier_quality = np.sum(diffs.real*trial_signs, axis=3) / np.maximum(
        np.sum(np.abs(diffs), axis=3), 1e-30)
    accepted = (adjacent & (qualities >= .65)
                & (carrier_quality.max(axis=0) >= .85))
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
        full_pulse = rx._pulse(SPS)
        full_z = {cn: rx._baseband(shifted, cn, FS, full_pulse)
                  for cn in spec.VH_CHANNELS}
        env = rxfront._tone_envelope(full_z, (len(full_pulse)-1)//2,
                                     0, len(audio))
        for k in good[np.argsort(quality[good])[::-1]]:
            at = int(starts[k])*DECIMATION
            if not rxfront._inside_burst(env, 0, at):
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
            if best is None or got.quality > best.quality:
                best = got
            break
    full_cache = {}
    # A 100-Hz alternative can score higher on an isolated bare word. Prefer
    # continuity only among hypotheses which already passed every soft/bit
    # gate, and still validate each one's physical extent below. Without a
    # retained frequency, keep the original quality ordering.
    ordering = (None if preferred_hz is None else
                lambda row: (abs(row[4] - preferred_hz) <= 25.0, *row))
    for quality, hz, at, ci, fine in sorted(ranked, key=ordering, reverse=True):
        if hz not in full_cache:
            shifted = compensate(audio, hz)
            full_pulse = rx._pulse(SPS)
            full_z = {cn: rx._baseband(shifted, cn, FS, full_pulse)
                      for cn in spec.VH_CHANNELS}
            env = rxfront._tone_envelope(full_z, (len(full_pulse)-1)//2,
                                         0, len(audio))
            full_cache[hz] = shifted, env
        shifted, env = full_cache[hz]
        if rxfront._inside_burst(env, 0, at):
            ev = rxfront._cs_event(shifted, ci, 0, at, at/FS,
                                   f", acquisition RX {fine:+g} Hz")
            return Candidate(ev, fine, quality, hz)
    return best
