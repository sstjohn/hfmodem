# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-III multitone DPSK modulator and matched demodulator.

IMPORTANT: PACTOR-III is NOT OFDM. There is no cyclic prefix, and no transform is
part of the waveform. It is a set of up to 18 independent 100 Bd carriers on a
120 Hz grid, each carrying its own differentially-encoded PSK stream, pulse-shaped
for spectral containment. (The spec makes a point of the signal's "very high
spectral steepness" and of its low crest factor.) So we synthesise it directly as
a sum of shaped carriers. That the sum is *computed* through an FFT is an
arithmetic convenience -- see `modulate_tones` -- and changes nothing about what
the signal is.

The demodulator here is OURS -- it closes a loopback, which exercises the
modulator against this file's own conventions and nothing else. Only an
independent PACTOR-III receiver can say whether the signal really is PACTOR-III.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft

from . import spec, tablegen
from scipy.signal import fftconvolve


DEFAULT_SAMPLE_RATE = spec.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Differential PSK mapping
# ---------------------------------------------------------------------------

# DQPSK phase increments, Gray-coded: adjacent phase steps differ in one bit, so
# a phase slip costs one bit rather than two. Gray coding here is an ASSUMPTION
# (the spec does not state the dibit->phase map); it is the conventional choice
# and is cheap to flip if a real decoder disagrees.
DQPSK_GRAY = {
    (0, 0): 0.0,
    (0, 1): np.pi / 2,
    (1, 1): np.pi,
    (1, 0): 3 * np.pi / 2,
}
DQPSK_GRAY_INV = {v: k for k, v in DQPSK_GRAY.items()}


def differential_encode(bits: np.ndarray, bits_per_symbol: int,
                        ref_phase: float = 0.0) -> np.ndarray:
    """Bits -> unit-modulus symbols, differentially encoded.

    The returned array is length N+1: element 0 is the PHASE REFERENCE symbol
    (the spec's "single phase reference pulse" that precedes every packet and
    every control signal), followed by one symbol per input symbol period.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % bits_per_symbol:
        raise ValueError(f"{bits.size} bits is not a multiple of "
                         f"{bits_per_symbol} bits/symbol")
    n = bits.size // bits_per_symbol

    if bits_per_symbol == 1:
        deltas = np.where(bits == 1, np.pi, 0.0)          # DBPSK
    elif bits_per_symbol == 2:
        pairs = bits.reshape(n, 2)
        deltas = np.array([DQPSK_GRAY[(int(a), int(b))] for a, b in pairs])
    else:
        raise ValueError("PACTOR-III uses only DBPSK and DQPSK")

    phases = np.concatenate([[ref_phase], ref_phase + np.cumsum(deltas)])
    return np.exp(1j * phases)


def differential_decode(symbols: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Soft bits from differentially-encoded symbols (inverse of the above).

    Returns one soft value per bit: positive = 0, negative = 1, magnitude =
    confidence -- the form `coding.viterbi_decode` wants.
    """
    symbols = np.asarray(symbols)
    d = symbols[1:] * np.conj(symbols[:-1])       # phase difference

    if bits_per_symbol == 1:
        # DBPSK: 0 -> phase step 0 (Re>0), 1 -> phase step pi (Re<0)
        return d.real / (np.abs(d) + 1e-12)

    # DQPSK, Gray (see DQPSK_GRAY): bit0 = 0 for phase steps {0, pi/2} and 1 for
    # {pi, 3pi/2}; bit1 = 0 for {0, 3pi/2} and 1 for {pi/2, pi}. Rotating by
    # -pi/4 moves the four points to the diagonals, so the decision boundaries
    # become the I and Q axes and the soft metrics are simply:
    #     bit0 ~  Re   (positive => 0)
    #     bit1 ~ -Im   (positive => 0)
    r = d * np.exp(-1j * np.pi / 4)
    mag = np.abs(r) + 1e-12
    soft = np.empty(d.size * 2)
    soft[0::2] = r.real / mag
    soft[1::2] = -r.imag / mag
    return soft


# ---------------------------------------------------------------------------
# Pulse shaping
# ---------------------------------------------------------------------------

def raised_cosine(sps: int, rolloff: float, span_symbols: int = 8) -> np.ndarray:
    """Raised-cosine pulse, normalised to unit peak.

    Rolloff is an UNKNOWN in the sense that the spec only says the signal has
    "very high spectral steepness"; 0.25-0.35 keeps each 100 Bd tone comfortably
    inside its 120 Hz slot.
    """
    n = span_symbols * sps
    t = (np.arange(n + 1) - n / 2) / sps          # in symbol periods
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.sinc(t)
        denom = 1 - (2 * rolloff * t) ** 2
        cos = np.cos(np.pi * rolloff * t) / denom
        h = sinc * cos
    # remove the removable singularities at t = +-1/(2*rolloff)
    if rolloff > 0:
        bad = np.isclose(np.abs(denom), 0.0)
        h[bad] = np.pi / 4 * np.sinc(1 / (2 * rolloff))
    h[np.isnan(h)] = 0.0
    return h / h.max()


# ---------------------------------------------------------------------------
# Modulator
# ---------------------------------------------------------------------------

@dataclass
class ModConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    rolloff: float = 0.3
    span_symbols: int = 8
    amplitude: float = 0.5
    matched_pulse: bool = False
    """Shape with `tablegen.symbol_pulse` instead of a raised cosine.

    The receiver runs that 31-tap symmetric FIR at 8 samples/symbol per tone;
    using it as the transmit pulse is what a matched pair means, and it measures
    far better across all 18 tones than a raised cosine wide enough to spill into
    the neighbouring 120 Hz slot. The taps are symmetric and sum to 1, so the
    pulse adds no group delay of its own and leaves the level alone."""

    @property
    def sps(self) -> int:
        s = self.sample_rate / spec.SYMBOL_RATE_BD
        if s != int(s):
            raise ValueError("sample rate must be an integer multiple of 100 Bd")
        return int(s)

    def pulse(self) -> np.ndarray:
        if not self.matched_pulse:
            return raised_cosine(self.sps, self.rolloff, self.span_symbols)
        return matched_pulse(self.sps)

    @property
    def pulse_key(self) -> tuple:
        """Everything `pulse()` reads, so that a cache can be keyed on the taps
        without holding them. Tap COUNT is not enough to tell two pulses apart --
        rolloff 0.30 and rolloff 0.15 are both 3841 taps at 48 kHz -- so anything
        new that reaches `pulse()` has to appear here too."""
        return (self.sps, self.matched_pulse, self.rolloff, self.span_symbols)


def matched_pulse(sps: int) -> np.ndarray:
    """The 31-tap symbol pulse, resampled from 8 to `sps` samples/symbol."""
    if sps % 8:
        raise ValueError("matched pulse needs a multiple of 8 samples/symbol")
    from scipy.signal import resample_poly
    return resample_poly(tablegen.symbol_pulse(), sps // 8, 1)


def modulate_tones(tone_symbols: dict[int, np.ndarray],
                   cfg: ModConfig | None = None,
                   delay: dict[int, int] | None = None) -> np.ndarray:
    """Sum a set of DPSK-modulated tones into a real passband waveform.

    `tone_symbols` maps channel number -> complex symbol sequence (as produced by
    `differential_encode`, i.e. INCLUDING the leading phase-reference symbol).
    All sequences must be the same length.

    `delay` maps channel number -> samples that tone's symbol train starts into
    the waveform, for the one speed level whose carriers do not share a symbol
    clock (`spec.SUBBAND_LEAD`). It is expressed in samples rather than symbols
    because the offset is sub-symbol and `placement.Path.clock_offsets` is the one
    place that turns the measurement into a count; the waveform grows at the end
    by the largest of them. Omitted or all-zero, this is the single-clock
    modulator it has always been, sample for sample.

    The tones are summed in frequency and brought back with ONE inverse transform
    (`_sum_in_frequency`), which is worth 3-8x per frame; a sample rate that will
    not carry that construction gets the plain per-tone sum instead. The two agree
    to ~1e-12 on a full-length cycle, and `tests/shrike/test_modem.py` renders both
    ways and checks it.
    """
    cfg = cfg or ModConfig()
    lengths = {len(v) for v in tone_symbols.values()}
    if len(lengths) != 1:
        raise ValueError(f"all tones must carry the same symbol count, got {lengths}")
    n_sym = lengths.pop()
    delay = {cn: int(delay.get(cn, 0)) for cn in tone_symbols} if delay else {}

    pulse = cfg.pulse()
    n_out = n_sym * cfg.sps + pulse.size - 1 + max(delay.values(), default=0)
    n = _transform_length(cfg, n_out)
    signal = (_sum_in_time(tone_symbols, cfg, pulse, n_sym, n_out, delay) if n is None
              else _sum_in_frequency(tone_symbols, cfg, pulse, n_sym, n_out, n, delay))

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * cfg.amplitude
    return signal


def _transform_length(cfg: ModConfig, n_out: int) -> int | None:
    """Transform length for the frequency-domain sum, or None if the rate cannot
    carry it.

    Choosing N rather than inheriting it is what makes the construction exact.
    Three demands: at least `n_out`, so the convolution stays linear and nothing
    wraps; a multiple of `sps`, so the upsampled symbol train's spectrum is the
    symbols' own spectrum tiled; and a whole number of 120 Hz periods, so one tone
    spacing is an integer number of bins and mixing becomes a rotation.

    The last is the one that can quietly go wrong. Bin spacing is `fs/N`, so a
    120 Hz step is a whole number of bins only when N is a multiple of `fs/120` --
    and no N can be, unless `fs/120` is itself a whole number of samples. It is at
    48 kHz (400) and at 12 kHz (100); it is not at 8 kHz (66.67) or at 44.1 kHz
    (367.5), where rounding the step would put every carrier tens of Hz off with
    nothing raised. `ModConfig` accepts all four rates, so the property is tested
    rather than assumed.
    """
    periods = cfg.sample_rate / spec.TONE_SPACING_HZ      # samples per 120 Hz cycle
    if periods != int(periods):
        return None
    grid = int(np.lcm(cfg.sps, int(periods)))
    n = -(-n_out // grid) * grid
    # The OFFSET half of the same integrality condition. The test above fixes the
    # tone SPACING in bins; this fixes where the comb starts, since a rotation by a
    # whole number of bins is only the right carrier if the first carrier is on a
    # bin too. Both are needed to make the mix exact, and neither implies the other
    # -- a hypothetical 510 or 550 Hz first tone would pass the spacing test and
    # fail here. At the rates shrike actually uses it never fires, because 480 is
    # four spacings; it is free insurance against a constant changing, not a
    # statement that channel 0 is special.
    return n if spec.TONE0_HZ * n % cfg.sample_rate == 0 else None


_PULSE_SPECTRUM: dict[tuple, np.ndarray] = {}


def _sum_in_frequency(tone_symbols: dict[int, np.ndarray], cfg: ModConfig,
                      pulse: np.ndarray, n_sym: int, n_out: int,
                      n: int, delay: dict[int, int]) -> np.ndarray:
    """Sum the shaped tones as spectra, one inverse transform for the lot.

    Three facts about the waveform collapse the work. Every tone is shaped by the
    SAME pulse, so its spectrum is computed once and cached instead of being
    re-transformed inside a convolution per tone. The upsampled symbol train is an
    impulse train at stride `sps`, so its length-N spectrum is the length-N/sps
    spectrum of the symbols, TILED -- 90 points instead of 43200 on a short cycle.
    And a tone's carrier is a whole number of bins away (see `_transform_length`),
    so mixing it up is a circular rotation of its spectrum rather than a
    length-N complex exponential.

    A delayed tone keeps all three: a shift in time is a linear phase ramp, and it
    goes on the BASEBAND spectrum, before the rotation that mixes the tone up. The
    envelope moves and the carrier does not, which is what a transmitter whose
    oscillator keeps running does and what `_sum_in_time` computes; ramping after
    the rotation delays the carrier too and leaves that tone rotated by
    exp(2i.pi.f.d/fs) against the other construction. N covers `n_out`, which
    already includes the longest delay, so nothing wraps.
    """
    n_blk = n // cfg.sps
    key = (n, cfg.pulse_key)
    ps = _PULSE_SPECTRUM.get(key)
    if ps is None:
        ps = _PULSE_SPECTRUM[key] = sfft.fft(pulse, n)

    tile = np.arange(n) % n_blk
    ramps: dict[int, np.ndarray] = {}
    acc = np.zeros(n, complex)
    for cn, syms in tone_symbols.items():
        # channel_freq_hz range-checks the channel; N makes the quotient whole.
        carrier_bin = round(spec.channel_freq_hz(cn) * n / cfg.sample_rate)
        blk = np.zeros(n_blk, complex)
        blk[:n_sym] = syms
        x = sfft.fft(blk)[tile] * ps
        d = delay.get(cn, 0)
        if d:
            if d not in ramps:
                ramps[d] = np.exp(-2j * np.pi * d * np.arange(n) / n)
            x = x * ramps[d]
        acc += np.roll(x, carrier_bin)

    return sfft.ifft(acc).real[:n_out]


def _sum_in_time(tone_symbols: dict[int, np.ndarray], cfg: ModConfig,
                 pulse: np.ndarray, n_sym: int, n_out: int,
                 delay: dict[int, int]) -> np.ndarray:
    """Shape and mix each tone on its own. The definition of the waveform, and
    what any rate off the 120 Hz grid gets."""
    sps = cfg.sps
    t = np.arange(n_out) / cfg.sample_rate
    signal = np.zeros(n_out)

    for cn, syms in tone_symbols.items():
        d = delay.get(cn, 0)
        up = np.zeros(n_sym * sps + d, dtype=complex)
        up[d::sps] = np.asarray(syms)
        # fftconvolve, not np.convolve: the direct form is O(n_sym*sps*taps) and at
        # SL6 that is 18 tones x a 1860-tap pulse, which put the MODULATOR at real
        # time (RTF 1.04) on the fastest machine here -- pure synthesis, no search.
        # Numerically identical (max abs diff ~5e-15), 15x faster.
        shaped = fftconvolve(up, pulse)                  # complex baseband
        f = spec.channel_freq_hz(cn)
        m = shaped.size
        signal[:m] += (shaped * np.exp(2j * np.pi * f * t[:m])).real

    return signal


def demodulate_tones(signal: np.ndarray, channels: tuple[int, ...],
                     n_symbols: int, cfg: ModConfig | None = None,
                     delay: dict[int, int] | None = None,
                     ) -> dict[int, np.ndarray]:
    """Recover per-tone complex symbols (matched filter + symbol sampling).

    This is our own receiver, used to close the loopback. It assumes the signal
    starts at sample 0 (no timing search) -- real synchronisation is an RX
    problem and out of scope for the transmitter. `delay` is `modulate_tones`'s,
    read back: a tone that went out late is sampled late.
    """
    cfg = cfg or ModConfig()
    sps = cfg.sps
    pulse = cfg.pulse()
    group = (pulse.size - 1) // 2
    t = np.arange(signal.size) / cfg.sample_rate

    out: dict[int, np.ndarray] = {}
    for cn in channels:
        f = spec.channel_freq_hz(cn)
        # Mix to baseband. The real passband signal has both +f and -f images;
        # the matched filter is narrow enough to kill the -2f term.
        bb = signal * np.exp(-2j * np.pi * f * t)
        # Same reasoning as the modulator above, same 1860 taps: an identical
        # 'full' result to 2e-15, and hard bit decisions bit-identical from 30 dB
        # down to -6 dB SNR.
        #
        # Deliberately NOT quoting a speedup factor here. The figure that used to
        # sit in this comment was measured on the station board while three other
        # benchmarks were resident, and contention does not cancel between the two
        # arms: the direct form is arithmetic-bound over a tiny working set while
        # the transform streams ~600 kB arrays, so a busy board compresses the
        # ratio by 20-25% in one direction. Any number worth pinning has to come
        # from a run that held the board's lock.
        #
        # Two semantic differences from np.convolve, latent today because nothing
        # in shrike/ reaches this function -- it has exactly two callers and both
        # are tests -- but real if that changes. An empty input RAISES under
        # np.convolve and returns an empty array here; and a single non-finite
        # sample contaminates only the filter's support under the direct form but
        # the ENTIRE output under a transform, so one bad sample would take out
        # every symbol on every tone rather than a few.
        mf = fftconvolve(bb, pulse)
        # Peak of a pulse at symbol k sits at k*sps + 2*group after the two
        # convolutions (shaping + matched filter).
        idx = np.arange(n_symbols) * sps + 2 * group + (delay.get(cn, 0) if delay else 0)
        idx = idx[idx < mf.size]
        out[cn] = mf[idx]
    return out
