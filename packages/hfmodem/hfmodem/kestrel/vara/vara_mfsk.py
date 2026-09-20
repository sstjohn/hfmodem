# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""MFSK burst synthesis and per-symbol tone demodulation.

The waveform is exactly the shared MFSK burst of [spec 04 §4.2.1];
every physical constant below is cited to that section. The DSP code
(overlap-add loop, FFT tone detector) is our own  [ours].

Physical facts  [spec 04 §4.2.1]:
  * single tone per symbol; each symbol is one tone = sin(2*pi*carrier*n/2048).
  * tone -> frequency:  f = carrier * 48000/2048  Hz  (the emitted 2048-pt @48 kHz
    FFT bin equals the tone/carrier index).
  * sample rate 48000 Hz.
  * per-symbol advance 2048 samples @48 kHz — measured off air, and exactly the
    reciprocal of the 23.4375 Hz tone spacing, so the tones are orthogonal over
    one symbol and the burst carries no clock of its own.
  * inter-symbol shaping: N=32 raised-cosine WOLA cross-fade, so a symbol extends
    2048+32 samples and adjacent symbols overlap by N. Rise
    w[i]=(cos((i-0.5)*pi/32+pi)+1)/2, fall w[i]=(cos((i-0.5)*pi/32)+1)/2,
    i=0..31; flat 1.0 between. N itself is *not* resolved by any recording we
    hold (see spec) — only the advance is.

The connected-ack shares this grid but lights two carriers per symbol rather than
one  [spec 04 §4.2C]; :func:`synth_tone_pairs` / :func:`demod_tone_pairs` are its
half of the module.

Because the advance equals NFFT, every symbol begins at the same phase whichever
convention is used: a global oscillator and a per-symbol phase reset emit the
same samples. That equivalence is a property of the correct advance, and its
absence is a sharp test — see ``tests/kestrel/test_mfsk_timing.py``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import fftconvolve as _fftconvolve

from . import vara_frames as VF

# Physical constants  [spec 04 §4.2.1].
FS = 48000
NFFT = 2048               # tone<->bin identity: f = carrier*FS/NFFT
HOP = 2048                # per-symbol advance, measured off air
WOLA_N = 32               # raised-cosine cross-fade length (unresolved by our audio)
STRIDE = HOP + WOLA_N     # symbol extent: the advance plus the overlap it cross-fades


# Tone-search band: the carriers the waveform can actually use, and nothing else —
# the session's own payload alphabet [spec 04 §4.2.3], the BW2300 alphabet every
# station has to read a connect-request on whatever it is armed at
# [vara_frames, CR2750], every fixed preamble, and the captured BW2300 session
# response [spec 04 §4.2.2, spec 05 §5.3.3].
#
# Searching outside it is not merely wasted work, it is wrong: a received signal
# carries strong odd-harmonic images of its own tones, and on a recording whose
# in-band-to-out-of-band ratio is only 9 dB the 3f image beats the fundamental for
# every tone above ~1.1 kHz. An unbounded argmax then reads 3x the true carrier for
# most symbols, the preamble never locks, and a gateway that answered on the air
# presents as silence.
#
# THE BAND IS THE SESSION'S, and that is measured rather than tidy. BW2300 and
# BW500 share bins 29..98 (679.7-2296.9 Hz) — BW500's fourteen carriers sit inside
# BW2300's seventy — and BW2750 reaches seven bins past them at each end, 22..105
# (515.6-2460.9 Hz). One band wide enough for all three costs the narrower ones
# real readings: at 22..105 the KB3AC-10 session of 2026-08-26 drops answers below
# ten confirmed tones and the N3HYM-10 stranger of the greeting corpus stops being
# found at all, both of them BW2300 audio whose 3f images now land on BW2750
# carriers. So a station searches the band its own bandwidth can emit, and a
# BW2750 session pays for its reach on its own audio only.
_FIXED_TONES = frozenset().union(
    *(k.preamble for k in VF.BURSTS.values()), VF.SESSION_RESPONSE_2300,
    *VF.CONNECTED_ACK_2300)


def band_for(bw: str) -> tuple[int, int]:
    """Lowest and highest bin a station armed at ``bw`` may read a tone in."""
    t = (_FIXED_TONES | VF.BW2300_TONES.carriers
         | VF.ALPHABETS.get(str(bw), VF.BW2300_TONES).carriers)
    return min(t), max(t)


BIN_LO, BIN_HI = band_for("2300")


def _symbol_window() -> np.ndarray:
    """Per-symbol amplitude envelope: 32-sample raised-cosine rise, flat 1.0,
    32-sample raised-cosine fall  [spec 04 §4.2.1]."""
    w = np.ones(STRIDE)
    i = np.arange(WOLA_N)
    rise = (np.cos((i - 0.5) * np.pi / WOLA_N + np.pi) + 1) / 2.0
    fall = (np.cos((i - 0.5) * np.pi / WOLA_N) + 1) / 2.0
    w[:WOLA_N] = rise
    w[STRIDE - WOLA_N:] = fall
    return w


_W = _symbol_window()


def carrier_to_hz(carrier: int) -> float:
    """f = carrier * 48000/2048 Hz  [spec 04 §4.2.1]."""
    return carrier * FS / NFFT


def synth_tones(carriers: Sequence[int], amplitude: float = 0.5) -> np.ndarray:
    """Render a tone/carrier-index sequence to real 48 kHz audio.

    Each symbol is a single sine tone (global-phase for continuity), shaped by
    the raised-cosine window and overlap-added at HOP=2048  [spec 04 §4.2.1].
    Adjacent symbols cross-fade over the 32-sample overlap (rise+fall = 1)."""
    n_sym = len(carriers)
    if n_sym == 0:
        return np.zeros(0)
    total = (n_sym - 1) * HOP + STRIDE
    out = np.zeros(total)
    for k, c in enumerate(carriers):
        a = k * HOP
        n = np.arange(STRIDE)
        tone = np.sin(2 * np.pi * c * (a + n) / NFFT)   # global sample index
        out[a:a + STRIDE] += amplitude * _W * tone
    return out


def demod_tones(samples: np.ndarray, n_sym: int,
                band: tuple[int, int] | None = None) -> list[int]:
    """Recover the ``n_sym`` tone/carrier indices from a burst.

    Per-symbol tone detection: for symbol k, take a NFFT-sample window centred
    in the symbol's flat region, apply a Hann taper, FFT, and take the peak bin
    of the tone band — which equals the carrier index  [spec 04 §4.2.1]. ``band``
    is the session's  [see band_for]; omitted, it is BW2300's.
    """
    lo, hi = band or (BIN_LO, BIN_HI)
    s = np.asarray(samples, dtype=np.float64)
    win = np.hanning(NFFT)
    offset = (STRIDE - NFFT) // 2                 # centre the analysis window
    carriers: list[int] = []
    for k in range(n_sym):
        a = k * HOP + offset
        seg = s[a:a + NFFT]
        if seg.size < NFFT:
            seg = np.concatenate([seg, np.zeros(NFFT - seg.size)])
        mag = np.abs(np.fft.rfft(seg * win))
        carriers.append(lo + int(np.argmax(mag[lo:hi + 1])))
    return carriers


# --------------------------------------------------------------------------- #
# Burst-level convenience (frames <-> audio).
def synth_burst(callsign: str, kind: VF.BurstKind,
                amplitude: float = 0.5) -> np.ndarray:
    """Full handshake burst for ``callsign`` as audio  [spec 04 §4.2]."""
    return synth_tones(VF.handshake_tones(callsign, kind), amplitude=amplitude)


def demod_burst(samples: np.ndarray, kind: VF.BurstKind,
                band: tuple[int, int] | None = None) -> list[int]:
    """Demodulate a burst of ``kind`` to its full tone sequence
    (preamble + payload)  [spec 04 §4.2.1]."""
    n_sym = len(kind.preamble) + kind.n_payload
    return demod_tones(samples, n_sym, band)


# --------------------------------------------------------------------------- #
# Two-tone symbols. The connected-ack rides the same symbol grid as the bursts
# above but lights TWO equal-amplitude carriers per symbol  [spec 04 §4.2C], so
# it needs its own synth and its own reader — a single-tone reader returns one
# of the two, and which one depends on where the analysis window happens to sit.
# Bins masked around the first carrier before the second is read. A tone that sits
# exactly on a bin — which these do, the grid being the reciprocal of the advance —
# leaks into its two neighbours and nowhere else under a Hann window, so one bin
# either side is the whole main lobe. Two would swallow the closest pairs the
# waveform actually emits (symbols three bins apart occur in the payload).
_PAIR_SKIRT = 1


def synth_tone_pairs(pairs: Sequence[tuple[int, int]],
                     amplitude: float = 0.5) -> np.ndarray:
    """Render a sequence of two-tone symbols to real 48 kHz audio  [spec 04 §4.2C].

    Same symbol grid, window and overlap-add as :func:`synth_tones`; each symbol
    is the sum of its two carriers at equal amplitude.
    """
    if not len(pairs):
        return np.zeros(0)
    total = (len(pairs) - 1) * HOP + STRIDE
    out = np.zeros(total)
    n = np.arange(STRIDE)
    for k, pair in enumerate(pairs):
        a = k * HOP
        for c in pair:
            out[a:a + STRIDE] += amplitude * _W * np.sin(2 * np.pi * c * (a + n) / NFFT)
    return out


def demod_tone_pairs(samples: np.ndarray, n_sym: int,
                     band: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Recover ``n_sym`` two-tone symbols as sorted ``(lo, hi)`` carrier pairs.

    The strongest bin of the tone band, then the strongest outside that bin's
    Hann main lobe — which is the second carrier, because the two are emitted at
    equal amplitude and the lobe of the first is all that could outrank it.
    """
    lo, hi = band or (BIN_LO, BIN_HI)
    s = np.asarray(samples, dtype=np.float64)
    win = np.hanning(NFFT)
    offset = (STRIDE - NFFT) // 2
    out: list[tuple[int, int]] = []
    for k in range(n_sym):
        a = k * HOP + offset
        seg = s[a:a + NFFT]
        if seg.size < NFFT:
            seg = np.concatenate([seg, np.zeros(NFFT - seg.size)])
        mag = np.abs(np.fft.rfft(seg * win))[lo:hi + 1]
        i = int(np.argmax(mag))
        mag[max(0, i - _PAIR_SKIRT):i + _PAIR_SKIRT + 1] = 0
        j = int(np.argmax(mag))
        out.append((lo + min(i, j), lo + max(i, j)))
    return out


_WOFF = (STRIDE - NFFT) // 2          # analysis window offset within a symbol
_LOCK_CANDIDATES = 24                 # exact checks the prefilter is allowed to cost
_LOCK_SLACK = 1                       # preamble tones a lock may lose to fading


def _bin_energy(x: np.ndarray, bins) -> dict:
    """Sliding |DFT| at each distinct bin — one FFT-convolution each."""
    w = np.hanning(NFFT)
    out = {}
    for b in sorted(set(bins)):
        h = w * np.exp(-2j * np.pi * b * np.arange(NFFT) / NFFT)
        out[b] = np.abs(_fftconvolve(x, h[::-1], mode="valid"))
    return out


def _window_energy(x: np.ndarray) -> np.ndarray:
    """Sliding window energy, so the prefilter ranks by concentration rather than
    by loudness — otherwise it just finds the strongest part of the recording."""
    w2 = np.hanning(NFFT) ** 2
    return np.sqrt(np.maximum(_fftconvolve(x * x, w2[::-1], mode="valid"), 1e-30))


def lock_preamble(samples: np.ndarray, kind: VF.BurstKind,
                  coarse: int = 32, fine: int = 4,
                  band: tuple[int, int] | None = None) -> int | None:
    """Sample offset at which ``kind``'s fixed preamble sits, or None.

    Burst segmentation by audio envelope is not symbol-accurate and, on a real
    channel, is not even reliably burst-accurate: a fading edge or an amplitude
    dip splits one keyed transmission into fragments, so the piece handed to the
    demodulator can begin ~200 ms from where the burst actually starts, or miss it
    entirely. Demodulating such a piece from sample 0 yields tones that match
    nothing, which presents as "the gateway never answered".

    Every handshake burst opens with a fixed, callsign-independent preamble
    [spec 04 §4.2.2], so it can be *found* instead of assumed. Coarse-then-fine
    because a full-resolution search costs an FFT per candidate offset: the
    analysis window tolerates a few percent of a symbol, so 32 samples is fine to
    localise and 4 to settle.
    """
    pre = list(kind.preamble)
    k = len(pre)
    if k < 2:                       # a 1-tone preamble identifies nothing
        return None
    # demod_tones reads symbol i at a + i*HOP + _WOFF .. +NFFT, so the last usable
    # offset has to account for _WOFF too — it zero-pads rather than failing, which
    # is why omitting it went unnoticed.
    span = (k - 1) * HOP + _WOFF + NFFT
    if len(samples) < span:
        return None
    limit = len(samples) - span
    x = np.asarray(samples, dtype=np.float64)

    def match(a: int) -> int:
        t = demod_tones(x[a:], k, band)
        return sum(1 for u, v in zip(t, pre) if u == v)

    # Prefilter, so the exact check runs a handful of times instead of thousands.
    # Each preamble tone is one FFT bin, and a sliding DFT at a fixed bin is a
    # convolution — computed once over the whole record per DISTINCT bin, rather
    # than a full 2048-point transform at every candidate offset. Ranking by how
    # much of each window's energy lands in the expected bin is not the same test
    # as "argmax == expected bin", so it only orders the candidates; `match` still
    # decides, and the returned offset is identical to an exhaustive scan.
    denom = _window_energy(x)
    energy = _bin_energy(x, pre)
    grid = np.arange(0, limit + 1, coarse)
    score = np.zeros(len(grid), dtype=np.float64)
    for i, b in enumerate(pre):
        idx = grid + i * HOP + _WOFF
        score += energy[b][idx] / denom[idx]
    order = grid[np.argsort(score)[::-1]]

    # Require k - _LOCK_SLACK rather than a perfect match. A single deep fade on one
    # preamble symbol otherwise discards the whole burst — measured, on every one of
    # the 8 and 10 preamble positions — which is the wrong failure mode for a channel
    # whose defining behaviour is fading, and backwards from tolerating fading in the
    # payload. The payload check behind this gate is what actually identifies the
    # station, so the preamble only has to be discriminating enough to find a burst.
    need = k - _LOCK_SLACK
    hit = None
    for a in order[:_LOCK_CANDIDATES]:
        if match(int(a)) >= need:
            hit = int(a)
            break
    if hit is None:
        return None

    # Exact preamble match is a PLATEAU, not a point — roughly 1400 samples wide,
    # because the analysis window tolerates a large fraction of a symbol. The
    # payload, read further into the burst, is only correct over a sub-interval of
    # it, so an offset at the early edge costs payload tones: measured on real
    # gateway audio, the edge gives 13/15 where the centre gives 15/15.
    lo = hi = hit
    while lo - fine >= 0 and match(lo - fine) >= need:
        lo -= fine
    while hi + fine <= limit and match(hi + fine) >= need:
        hi += fine
    return (lo + hi) // 2
