# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP receive-side DSP: tone detection and acquisition primitives.

This is the sample-domain front end of the receiver — the mirror of
:mod:`besra.dsp.templates` on transmit. It provides the tools the demodulator
(:mod:`besra.phy.demodulator`) drives:

- the **½N-offset sliding DFT** (:class:`SlidingDFT`) that lands exactly on
  ARDOP's off-grid tones — a 240-sample transform hits 1425/1475/1525/1575 Hz, a
  120-sample one hits 1350/1450/1550/1650 Hz, bins a plain 50 Hz DFT grid cannot
  produce — plus a plain :func:`goertzel` for single-block reference measurement;
- vectorised sliding tone bins (:func:`sliding_bin`, :func:`tone_mag_series`)
  for scanning symbol timing across a whole capture at once;
- two-tone **leader detection** (:func:`leader_start`) by envelope correlation of
  the 1475/1525 Hz leader, which rejects single-tone carriers.

All facts (tone sets, 12 kHz rate, the SDFT recurrence
``S(n)=e^{j2π f/SR}·(S(n−1)+x(n)+x(n−N))``) are from ``docs/protocols/ardop/11-WAVEFORM.md`` §3, §4,
§7 and the MIT reference ``sdft.c``. Written as original numpy.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 12000

# The two-tone leader (1500 ± 25 Hz) and the 4FSK tone sets per baud (spec §3, §4).
LEADER_TONES = (1475, 1525)
FSK_TONES: dict[int, tuple[int, ...]] = {
    50: (1425, 1475, 1525, 1575),
    100: (1350, 1450, 1550, 1650),
    600: (600, 1200, 1800, 2400),
}


class SlidingDFT:
    """ARDOP's ½N-offset sliding DFT (``sdft.c``; spec §7).

    A conventional length-N DFT resolves multiples of ``SR/N`` (50 Hz for N=240);
    ARDOP's FSK tones sit halfway between those bins. Offsetting every bin by ½
    slides the transform onto 1425/1475/1525/1575 Hz (N=240) or
    1350/1450/1550/1650 Hz (N=120). The recurrence, fed one sample at a time, is
    ``S(n) = e^{j2π f/SR}·(S(n−1) + x(n) + x(n−N))`` — the reference's derivation
    folds the ½-bin offset straight into the per-frequency coefficient
    ``e^{j2π f/SR}``, so the bins are named by frequency, not integer index.

    Streaming a whole symbol yields the tone vector at every sample, which is what
    lets the reference refine symbol timing within a symbol period. For a plain
    per-symbol magnitude, :func:`tone_mag_series` is the vectorised equivalent.
    """

    def __init__(self, n: int, freqs: tuple[int, ...]):
        self.n = n
        self.coeff = np.exp(2j * np.pi * np.asarray(freqs, float) / SAMPLE_RATE)
        self.reset()

    def reset(self) -> None:
        self.s = np.zeros(self.coeff.size, dtype=complex)
        self._buf = np.zeros(self.n)
        self._i = 0

    def push(self, x: float) -> np.ndarray:
        """Feed one sample; return the complex tone vector at this sample."""
        old = self._buf[self._i]
        self.s = self.coeff * (self.s + x + old)
        self._buf[self._i] = x
        self._i = (self._i + 1) % self.n
        return self.s

    def block(self, seg: np.ndarray) -> np.ndarray:
        """The tone vector after streaming ``seg`` from a cleared state — the
        offset-DFT of one symbol (magnitude matches :func:`goertzel`)."""
        self.reset()
        out = self.s
        for x in seg:
            out = self.push(x)
        return out


def goertzel(samples: np.ndarray, start: int, n: int, freq: float) -> complex:
    """Single-bin DFT at ``freq`` over ``samples[start:start+n]`` by the Goertzel
    recurrence — retained for single-block reference tone measurement (``sdft.c``)."""
    seg = samples[start:start + n]
    w = 2 * np.pi * freq / SAMPLE_RATE
    coeff = 2 * np.cos(w)
    z1 = z2 = 0.0
    for x in seg:
        z1, z2 = x + coeff * z1 - z2, z1
    return complex(z1 - z2 * np.cos(w), z2 * np.sin(w))


def boxcar(x: np.ndarray, n: int) -> np.ndarray:
    """Sum of every length-``n`` window of ``x``, by window start — the O(N) face
    of ``np.convolve(x, np.ones(n), "valid")``, which is O(N·n). The rows of a 2-D
    ``x`` are summed independently, which is how a measurement made at several
    carrier offsets at once holds each offset to its own run."""
    acc = np.cumsum(x, axis=-1)
    acc = np.concatenate([np.zeros_like(acc, shape=acc.shape[:-1] + (1,)), acc], axis=-1)
    return acc[..., n:] - acc[..., :-n]


def _phasor(freq: float, size: int) -> np.ndarray:
    """``exp(-2j π · freq · k / SAMPLE_RATE)`` for k in ``[0, size)``.

    An integer frequency repeats every ``SAMPLE_RATE / gcd(freq, SAMPLE_RATE)``
    samples and every tone besra detects is one, so the capture-length phasor is a
    tile of at most one period rather than a capture's worth of complex
    exponentials. The tile is also the more accurate of the two: the direct form's
    phase argument reaches ~1e9 radians over a minute of audio and loses a digit
    of the angle for every decade of k."""
    period = size
    if freq == int(freq):
        period = min(size, SAMPLE_RATE // gcd(int(freq) % SAMPLE_RATE, SAMPLE_RATE))
    base = np.exp(-2j * np.pi * freq / SAMPLE_RATE * np.arange(period))
    return base if period == size else np.resize(base, size)


def sliding_bin(samples: np.ndarray, n: int, freq: float) -> np.ndarray:
    """The complex DFT bin at ``freq`` for *every* length-``n`` window, indexed by
    window start. A running complex-heterodyne sum, so the whole capture's symbol
    timing can be probed at once (the vectorised face of the sliding DFT).

    Cost is linear in ``samples``, not in the number of windows read, so slice the
    region of interest before calling rather than transforming a whole capture to
    index a few hundred columns of it."""
    return boxcar(samples * _phasor(freq, samples.size), n)


def tone_mag_series(samples: np.ndarray, n: int, freqs: tuple[int, ...]) -> np.ndarray:
    """Magnitude² at each of ``freqs`` for every length-``n`` window — shape
    ``(len(freqs), len(samples)-n+1)``. The tone detector's raw material."""
    return np.stack([np.abs(sliding_bin(samples, n, f)) ** 2 for f in freqs])


def carrier_bin_series(samples: np.ndarray, freq: float, n: int = 120) -> np.ndarray:
    """The complex 100-baud PSK/QAM carrier bin at every window start. A rectangular
    120-sample window makes ARDOP's 200 Hz-spaced carriers mutually orthogonal
    (200 Hz = two 100 Hz DFT bins), so no inter-carrier window is needed."""
    return sliding_bin(samples, n, freq)


def leader_start(samples: np.ndarray, threshold: float = 0.3,
                 shift: float = 0.0) -> int:
    """Locate the start of the two-tone leader by the rising edge of the
    1475·1525 Hz envelope product (a single tone leaves one factor near zero, so
    the product rejects it). Silence precedes the leader, so the first crossing of
    ``threshold`` × the capture's peak product is the frame's leader.

    ``shift`` moves both bins by a carrier frequency offset. Detecting a mistuned
    leader that way is the same measurement as de-rotating the samples and looking
    at the nominal bins — the de-rotation is a constant phase inside each window,
    which the magnitudes drop — and it does not cost a capture-length complex
    array per candidate offset."""
    n = SAMPLE_RATE // 50  # one 20 ms leader symbol
    lo = np.abs(sliding_bin(samples, n, LEADER_TONES[0] + shift))
    hi = np.abs(sliding_bin(samples, n, LEADER_TONES[1] + shift))
    product = lo * hi
    peak = product.max()
    if peak <= 0:
        return -1
    crossings = np.flatnonzero(product > threshold * peak)
    return int(crossings[0]) if crossings.size else -1


def leader_presence(samples: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """How much like the two-tone leader each 20 ms window looks, on an absolute
    scale — ``2·|1475|·|1525| / (window power)``, indexed by window start.

    :func:`leader_start` measures the leader against the slice's own peak, which
    answers "where in this burst does it begin" and cannot answer "is there a
    leader here at all". This is the second question: the ratio is 1 for a
    balanced pair of tones and nothing else, ~0 for one tone alone (a body symbol,
    a carrier) and ~0 for noise, with no reference to any peak. That independence
    is the point — a busy HF channel has no quiet stretch to normalise against.

    Measured on the 2026-08-05 KE8LVA session: real off-air leaders hold ≥ 0.25 for
    62–90% of their 240 ms, while the recording as a whole reads 0.034 at the
    median and 0.179 at the 90th percentile.

    The denominator is the one place in this module that squares the samples
    without a complex phasor to promote them, so an int16 capture is widened here
    rather than left to wrap. Left to wrap it collapses: the ``1e-9`` clamp turns
    what survives into division by nothing and the ratio comes back around 1e21 —
    maximal confidence, for a capture of noise. `Demodulator.decode` casts before
    it calls this, which is not a property this function should have to borrow."""
    au = np.asarray(samples, dtype=np.float64)
    n = SAMPLE_RATE // 50
    lo = np.abs(sliding_bin(au, n, LEADER_TONES[0] + shift))
    hi = np.abs(sliding_bin(au, n, LEADER_TONES[1] + shift))
    power = boxcar(au * au, n)[:lo.size]
    return 2 * lo * hi / np.maximum(power * n / 2, 1e-9)


#: The DFT grid :func:`leader_presence_grid` reads its carrier offsets off. 25 Hz
#: is half the 20 ms detection bin's own width, so the worst-placed leader in the
#: grid still lands within a quarter bin of a trial offset and keeps 81% of its
#: presence; it is also the coarsest grid on which 1475 and 1525 Hz are both whole
#: bins of a 480-point transform.
LEADER_SHIFT_HZ = 25


def leader_presence_grid(samples: np.ndarray, shifts: np.ndarray,
                         hop: int) -> np.ndarray:
    """:func:`leader_presence` at a whole grid of carrier offsets at once — shape
    ``(len(shifts), windows)``, one column per ``hop`` samples. ``shifts`` are
    multiples of :data:`LEADER_SHIFT_HZ`.

    A fixed-bin leader detector is deaf to a mistuned channel, and the way out is
    to look at the offsets too. Doing that through :func:`sliding_bin` costs a
    capture-length heterodyne per bin — thirty-four of them for the ±200 Hz a
    receiver must tolerate — where zero-padding each 20 ms window to 480 samples
    puts the DFT grid on 25 Hz, lands 1475 and 1525 Hz on bins 59 and 61, and
    makes every offset in the search the same transform read one bin further
    along. Measured on a 6.25 s window: 0.6 ms for seventeen offsets against 1.7
    ms for the one nominal offset at sample resolution.

    The price is position resolution — ``hop`` samples, where
    :func:`leader_presence` answers for every sample. A search this wide is worth
    that trade only where positions are not the answer wanted, so this is a
    measurement of the channel and not a source of acquisition candidates; the
    reference's own leader search hops 240 samples and never looks between
    (``SearchFor2ToneLeader3``)."""
    au = np.asarray(samples, dtype=np.float64)
    n = SAMPLE_RATE // 50
    if au.size < n:
        return np.zeros((np.size(shifts), 0))
    pad = SAMPLE_RATE // LEADER_SHIFT_HZ
    spec = np.abs(np.fft.rfft(sliding_window_view(au, n)[::hop], pad, axis=-1)).T
    power = boxcar(au * au, n)[::hop]
    lo = spec[(LEADER_TONES[0] + shifts) // LEADER_SHIFT_HZ]
    hi = spec[(LEADER_TONES[1] + shifts) // LEADER_SHIFT_HZ]
    return 2 * lo * hi / np.maximum(power * n / 2, 1e-9)


def _energy_onset(au: np.ndarray, frac: float = 0.15) -> int:
    """First sample where one-symbol energy rises past ``frac`` of the capture's
    peak — the burst onset, measured on |x|² so it is independent of any tuning
    offset (unlike a fixed-bin detector)."""
    n = SAMPLE_RATE // 50
    e = boxcar(au * au, n)
    if e.size == 0 or e.max() <= 0:
        return 0
    return int(np.argmax(e > frac * e.max()))


#: How much of the burst the offset estimate reads. The spec's shortest legal
#: leader is five symbols and the last of them is the phase-reversed sync
#: (§App. B, 5–50 symbols; `ARDOPC.c:98` defaults to 12), so four symbols is all
#: any legal leader guarantees. Past that the window runs into the 4FSK header,
#: whose tones are also 50 Hz apart and ~3 dB hotter, and the two-tone product
#: scores a sequential pair as readily as a simultaneous one: on rendered frames
#: in an off-air noise floor, a 200 ms window misread a 120 ms leader by more
#: than the deadband on 46% of offsets and a 100 ms leader on 90%, at every SNR
#: from +20 to 0 dB — a straddle, not a sensitivity limit.
_CFO_WINDOW = 4 * SAMPLE_RATE // 50


def estimate_leader_cfo(samples: np.ndarray, max_hz: float = 220.0) -> float:
    """Carrier frequency offset of the two-tone leader, in Hz, to ~1 Hz.

    A ±200 Hz mistuned leader lands off the 1475/1525 Hz bins — 200 Hz is exactly
    four 50 Hz bins, a sinc null, so the fixed detector goes deaf (spec §4.1/§7
    require ±200 Hz tolerance). Anchor a window on the burst onset and, over
    :data:`_CFO_WINDOW` (fine enough to resolve the 50 Hz-spaced pair, which the
    20 ms detection bin cannot), sweep a trial shift and keep the one whose two
    tone bins ring together. Coarse 5 Hz sweep then a 1 Hz refine gives the
    docstring's "frequency tuning to ~1 Hz"; the estimate de-rotates the frame
    onto its nominal tones for the rest of the receive chain.

    The estimate is only as good as the segment is leader: a leader whose front
    was clipped (an ARQ reply landing in the receiving station's own post-TX
    recovery) fills the window with header tones instead, and two sequential 4FSK
    tones 50 Hz apart score as a leader pair — a −50 Hz alias (measured −43 Hz
    against a +5 Hz truth on KE8LVA's off-air ConAck2000). The demodulator
    therefore retries a failed acquisition at zero shift rather than trusting
    this estimate unconditionally."""
    au = np.real(np.asarray(samples, float))
    seg = au[_energy_onset(au):][:_CFO_WINDOW]
    if seg.size < SAMPLE_RATE // 25:  # under two leader symbols: nothing to lock
        return 0.0

    # Every trial shift is a whole number of Hz, so the whole sweep is one
    # transform: zero-padding the segment to SAMPLE_RATE makes the DFT grid
    # exactly 1 Hz and the bin index the frequency in Hz (negative shifts index
    # from the top, which is the same bin). The direct form recomputed a
    # windowful of complex heterodyne per trial — ~98 of them per call.
    spec = np.abs(np.fft.fft(seg, SAMPLE_RATE))

    def product(trials: np.ndarray) -> np.ndarray:
        return spec[trials + LEADER_TONES[0]] * spec[trials + LEADER_TONES[1]]

    grid = np.arange(-int(max_hz), int(max_hz) + 1, 5)
    coarse = grid[int(np.argmax(product(grid)))]
    fine = np.arange(coarse - 4, coarse + 5)
    return float(fine[int(np.argmax(product(fine)))])
