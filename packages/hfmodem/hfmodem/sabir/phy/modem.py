# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Scattered-pilot OFDM at a common 1500 Hz audio center.

Contiguous even-sized carrier sets use signed FFT bins -N/2 .. N/2-1
and a mixer at CENTER_HZ + half a bin. Both the arithmetic carrier midpoint
and the acquisition preamble are therefore centered exactly at CENTER_HZ.
Windowed cyclic prefixes, envelope clipping and a linear bandpass filter
control spectral skirts. The receiver estimates a time/frequency channel
surface from scattered pilots, with optional decoder feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import hilbert, firwin2

from hfmodem.sabir.dsp import fading
from hfmodem.sabir.frame.codec import pn9
from hfmodem.sabir.dsp.constellation import Constellation
from hfmodem.sabir.dsp.sync import remove_cfo

from .preamble import ZCPreambleDetector, preamble_waveform
from .rate import CENTER_HZ, FS

N_FFT = 1024
SPACING = FS / N_FFT                # 46.875 Hz
CP = 256
WINDOW = 64                         # raised-cosine CP edge taper, samples
GUARD_HEAD = 512
GUARD_TAIL = 256

# scattered pilot lattice: pilot at carrier r in symbol s iff (r - 2s) mod 6 == 0
LATTICE_DF = 6
LATTICE_STAG = 2
LATTICE_PERIOD = 3                  # symbols between pilots on one carrier

# selective-ACK / loading granularity: every gear's active carriers split into
# N_GROUPS contiguous groups (3 carriers each at 24, 7 at 56)
N_GROUPS = 8
CONST_BPC = {"qpsk": 2, "16qam": 4, "64qam": 6, "256qam": 8}
_BPC_CONST = {v: k for k, v in CONST_BPC.items()}
_CONSTS: dict[int, Constellation] = {}


def carrier_groups(n_carriers: int) -> np.ndarray:
    """Relative carrier index -> its selective-ACK / loading group."""
    return np.arange(n_carriers) * N_GROUPS // n_carriers


def const_for(bpc: int) -> Constellation:
    if bpc not in _CONSTS:
        _CONSTS[bpc] = Constellation.create(_BPC_CONST[bpc])
    return _CONSTS[bpc]


def analytic(real: np.ndarray) -> np.ndarray:
    """Real audio -> unit-scaled analytic signal, at an FFT-friendly length.

    The one place sabir turns audio into a complex baseband stream. Both the
    PHY ingress (``Phy.from_audio``) and the off-air tools route through here,
    because they were the same expression written twice and the edge treatment
    below has to be stated once rather than diverge.

    **Edge treatment, stated because the textbook answer is just "the analytic
    signal".** ``scipy.signal.hilbert`` transforms at exactly ``len(x)`` and
    never pads, so a length with a large prime factor costs ~5x the next fast
    length -- and once audio comes off a sound card, length is not something
    anyone chose. Padding to a fast length is *not* the usual free FFT pad: the
    analytic transform is non-local, so ``hilbert(x, m)[:n]`` differs from
    ``hilbert(x)`` by order 1 on a unit-variance signal.

    That difference is an edge effect, and there is no privileged answer to
    prefer -- for a finite capture, period-n and period-m are both arbitrary
    conventions and scipy's choice of n is not special. So this pads, and the
    contract is: **the interior is what sabir demodulates, and bursts must sit
    inside guard or pre-roll rather than against the ends.** Every waveform here
    already does. Decodes are byte-exact under either convention, which is the
    property ``tests/sabir/test_offair.py`` pins.
    """
    x = np.asarray(real, dtype=np.float64)
    if x.ndim != 1:
        # A sound card hands you (frames, channels), and `hilbert` transforms
        # the LAST axis -- so a 2-D array would be transformed across the two
        # channels rather than along time, and the pad/slice below would return
        # a differently shaped array with no error at all. Downmix first.
        raise ValueError(f"analytic() takes a 1-D real signal, got {x.shape}; "
                         "select or mix down to one channel first")
    m = next_fast_len(x.size)
    z = hilbert(x) if m == x.size else hilbert(x, m)[: x.size]
    return z / np.sqrt(2.0)


@dataclass(frozen=True)
class Gear:
    name: str
    n_carriers: int
    clip_papr_db: float             # per-gear envelope clip depth
    constellation: str = "qpsk"
    code: str = "r12"               # fec.CODES rung
    repeat: int = 1                 # all-carrier/time spreading factor
    lattice: tuple[int, int, int] = (LATTICE_DF, LATTICE_STAG, LATTICE_PERIOD)
    est_kernel: tuple[float, float] = (2.5, 1.5)   # smoother (sigma_t, sigma_f)
    ace: bool = False               # active constellation extension on TX
    n_fft: int = N_FFT              # per-gear numerology: FFT size ...
    cp: int = CP                    # ... and cyclic prefix, samples at 48 kHz

    def __post_init__(self):
        if self.n_carriers <= 0 or self.n_carriers % 2:
            raise ValueError("OFDM gears require a positive even carrier count")
        if self.n_carriers >= self.n_fft or not 0 < self.cp <= self.n_fft:
            raise ValueError("invalid OFDM carrier count or cyclic prefix")
        df, stag, period = self.lattice
        if df <= 1 or stag <= 0 or period != df // gcd(df, stag):
            raise ValueError("pilot period must match lattice spacing and stagger")
        if self.transition_hz <= 0:
            raise ValueError("carrier allocation leaves no filter transition band")

    @property
    def first_carrier(self) -> int:
        """First signed baseband FFT bin; use carrier_hz for audio frequency."""
        return -(self.n_carriers // 2)

    @property
    def mixer_hz(self) -> float:
        # The midpoint of -N/2 .. N/2-1 is -1/2, not zero.
        return CENTER_HZ + FS / self.n_fft / 2

    @property
    def carrier_hz(self) -> np.ndarray:
        return CENTER_HZ + (np.arange(self.n_carriers)
                            - (self.n_carriers - 1) / 2) * FS / self.n_fft

    @property
    def passband_hz(self) -> tuple[float, float]:
        # Preserve the shared 1200-chip/s preamble as well as the data grid.
        half = max(600.0, self.n_carriers * FS / self.n_fft / 2)
        return CENTER_HZ - half, CENTER_HZ + half

    @property
    def transition_hz(self) -> float:
        # The widest gear tapers to zero at 100 and 2900 Hz. Finite-burst
        # spectral leakage is measured separately; these are filter edges.
        return min(100.0, 1400.0 - (self.passband_hz[1] - CENTER_HZ))


# Standard pilot lattice: 1/6 overhead; narrower QPSK and wider QAM bands.
# QAM uses shallower clipping to protect its smaller decision distances.
GEARS = {
    "robust": Gear("24c-qpsk-r13x2", 24, 4.0,
                   "qpsk", "r13", repeat=2),
    "workhorse": Gear("24c-qpsk-r12", 24, 4.0,
                      "qpsk", "r12"),
    "workhorse34": Gear("24c-qpsk-r34", 24, 4.0,
                        "qpsk", "r34"),
    "fast": Gear("56c-16qam-r34", 56, 6.0,
                 "16qam", "r34"),
    "max": Gear("56c-64qam-r56", 56, 8.0,
                "64qam", "r56"),
    # Doppler profile: 512 FFT / 93.75 Hz / 57.7 symbols/s. Every carrier
    # receives pilots at 28.8 Hz, costing half the resource elements. The
    # narrow frequency smoother resolves the ~3.5-carrier notch spacing of a
    # 3 ms echo. Sparse34 spends 1/12 of resource elements on pilots for slow
    # channels and supports a decoder-aided second pass.
    "doppler": Gear("24c-qpsk-r12-512", 24, 4.0,
                    "qpsk", "r12", lattice=(2, 1, 2), est_kernel=(0.6, 0.3),
                    n_fft=512, cp=320),
    "sparse34": Gear("24c-qpsk-r34-sp", 24, 4.0,
                     "qpsk", "r34", lattice=(12, 2, 6), est_kernel=(4.0, 1.5)),
    # Explicitly negotiated short-delay, stable-channel profile. It is never
    # implied by a ladder ceiling; a receiver must advertise this exact ID.
    "wide256": Gear("56c-256qam-r56-sp", 56, 12.0,
                    "256qam", "r56", lattice=(12, 2, 6),
                    est_kernel=(2.5, 1.0), cp=128),
}
# The protocol gear index spans both waveform families:
# the noncoherent floor rows live in floor.mfsk.FLOOR_GEARS, the OFDM rows
# in GEARS above.
LADDER = ("floor4", "floor2", "floor",
          "robust", "workhorse", "workhorse34", "fast", "max")


@dataclass
class FadingResult:
    frame_start: int
    cfo_hz: float                   # coarse + integer correction applied
    n_symbols: int
    noise_var: float
    carrier_snr: np.ndarray         # per-carrier time-averaged SNR (linear)
    masked: np.ndarray              # dead carriers erased by the equaliser
    metric: float
    pilot_snr: np.ndarray | None = None  # interpolated pilot-channel power / noise


class Phy:
    def __init__(self, gear: Gear = GEARS["workhorse"], clip: bool = True):
        self.gear = gear
        self.clip = clip
        self._masks: dict[int, np.ndarray] = {}
        self.mixer_hz = gear.mixer_hz
        self.n_fft = gear.n_fft
        self.cp = gear.cp
        self.window = WINDOW * gear.n_fft // N_FFT
        self.spacing = FS / gear.n_fft
        self.const = Constellation.create(gear.constellation)
        self.pilot_value = 1 + 0j
        # Half the untapered CP slack tolerates timing error on either side.
        self.detector = ZCPreambleDetector(
            early_bias=(self.cp - self.window) // 2)
        self._carriers = np.arange(gear.first_carrier,
                                   gear.first_carrier + gear.n_carriers)
        self._bins = self._carriers % self.n_fft
        self.preamble = preamble_waveform()
        half = gear.passband_hz[1] - CENTER_HZ
        transition = np.linspace(half, half + gear.transition_hz, 65)
        gain = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 65)))
        taps = firwin2(8193, np.r_[0, transition, FS / 2],
                      np.r_[1, gain, 0], fs=FS, window="hann")
        time = (np.arange(taps.size) - taps.size // 2) / FS
        self._filter = taps * np.exp(2j * np.pi * CENTER_HZ * time)

    # -- scattered pilot lattice ----------------------------------------------
    def lattice_mask(self, n_syms: int) -> np.ndarray:
        """(n_syms, n_carriers) bool: True where the cell carries a pilot."""
        df, stag, _ = self.gear.lattice
        s = np.arange(n_syms)[:, None]
        r = np.arange(self.gear.n_carriers)[None, :]
        return (r - stag * s) % df == 0

    def _data_positions(self, n_syms: int,
                        tx_mask: np.ndarray | None = None) -> np.ndarray:
        pos = ~self.lattice_mask(n_syms)
        if tx_mask is not None:
            pos &= np.asarray(tx_mask, dtype=bool)[None, :]
        return pos

    def _norm_loading(self, tx_mask, loading):
        """-> (effective carrier mask | None, per-carrier bits-per-cell | None).

        ``loading`` is the bit-loading actuator: per-carrier bits per data
        cell (0 = carrier transmit-masked, 2/4/6/8 = QPSK/16-QAM/64-QAM/256-QAM),
        normally uniform within each carrier group. None = the gear's own
        constellation on every carrier.
        """
        if loading is None:
            mask = None if tx_mask is None else np.asarray(tx_mask, dtype=bool)
            if mask is not None and mask.shape != (self.gear.n_carriers,):
                raise ValueError("carrier mask has the wrong shape")
            return mask, None
        loading = np.asarray(loading, dtype=np.int64)
        if (loading.shape != (self.gear.n_carriers,)
                or not np.isin(loading, (0, 2, 4, 6, 8)).all()):
            raise ValueError("loading requires 0, 2, 4, 6 or 8 bits per carrier")
        mask = loading > 0
        if tx_mask is not None:
            tx_mask = np.asarray(tx_mask, dtype=bool)
            if tx_mask.shape != mask.shape:
                raise ValueError("carrier mask has the wrong shape")
            mask = mask & tx_mask
        return mask, loading

    def cell_map(self, n_syms: int, tx_mask: np.ndarray | None = None,
                 loading: np.ndarray | None = None):
        """Row-major data cells: (symbol idx, carrier idx, bits/cell, offsets).

        ``offsets`` has one trailing entry: the total bit capacity. The flat
        bit stream fed to ``transmit`` (and returned as LLRs by ``receive``)
        is ordered by these cells, ``bpc[i]`` bits each at ``offsets[i]``.
        """
        mask, bpc = self._norm_loading(tx_mask, loading)
        s_idx, r_idx = np.nonzero(self._data_positions(n_syms, mask))
        cb = (np.full(s_idx.size, self.const.bits_per_symbol, dtype=np.int64)
              if bpc is None else bpc[r_idx])
        off = np.concatenate([[0], np.cumsum(cb)])
        return s_idx, r_idx, cb, off

    def capacity(self, n_syms: int, tx_mask: np.ndarray | None = None,
                 loading: np.ndarray | None = None) -> int:
        """Payload bits carried by ``n_syms`` scattered-lattice symbols."""
        mask, bpc = self._norm_loading(tx_mask, loading)
        pos = self._data_positions(n_syms, mask)
        if bpc is None:
            return int(pos.sum()) * self.const.bits_per_symbol
        return int((pos * bpc[None, :]).sum())

    def n_symbols_for(self, n_bits: int, tx_mask: np.ndarray | None = None,
                      loading: np.ndarray | None = None) -> int:
        if n_bits <= 0:
            raise ValueError("an OFDM burst must contain coded bits")
        mask, bpc = self._norm_loading(tx_mask, loading)
        vec = (np.full(self.gear.n_carriers, self.const.bits_per_symbol,
                       dtype=np.int64) if bpc is None else bpc)
        period = self.gear.lattice[2]
        per_row = self.capacity(3 * period, tx_mask, loading) / (3 * period)
        if per_row == 0:
            raise ValueError("carrier allocation has no data capacity")
        rows = int(n_bits / max(per_row, 1.0)) + period + 2
        while True:
            counts = (self._data_positions(rows, mask)
                      * vec[None, :]).sum(axis=1)
            cum = np.cumsum(counts)
            if cum[-1] >= n_bits:
                return int(np.searchsorted(cum, n_bits)) + 1
            rows *= 2

    # -- transmit -------------------------------------------------------------
    def _body(self, grids: np.ndarray) -> np.ndarray:
        """Frequency grids (n_syms, n_fft) -> windowed-CP OFDM sample stream."""
        n_syms = grids.shape[0]
        cp, win = self.cp, self.window
        x = np.fft.ifft(grids, norm="ortho", axis=1)
        ext = np.hstack([x[:, -cp:], x, x[:, :win]])
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(win) + 0.5) / win))
        ext[:, :win] *= ramp
        ext[:, -win:] *= ramp[::-1]
        stride = self.n_fft + cp
        out = np.zeros(n_syms * stride + win, dtype=np.complex128)
        for s in range(n_syms):
            out[s * stride : s * stride + stride + win] += ext[s]
        return out

    def transmit(self, bits: np.ndarray, tx_mask: np.ndarray | None = None,
                 loading: np.ndarray | None = None) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int64).ravel()
        n_syms = self.n_symbols_for(bits.size, tx_mask, loading)
        s_idx, r_idx, cb, off = self.cell_map(n_syms, tx_mask, loading)
        n_pad = int(off[-1]) - bits.size
        if n_pad:       # spectrum filler; the frame codec ignores the tail
            bits = np.concatenate([bits, pn9(n_pad)])
        cells = np.empty(s_idx.size, dtype=np.complex128)
        for b in np.unique(cb):
            sel = np.nonzero(cb == b)[0]
            idx = off[sel][:, None] + np.arange(b)
            cells[sel] = const_for(int(b)).modulate(bits[idx].ravel())
        grids = np.zeros((n_syms, self.n_fft), dtype=np.complex128)
        grids[s_idx, self._bins[r_idx]] = cells
        pil_s, pil_r = np.nonzero(self.lattice_mask(n_syms))
        grids[pil_s, self._bins[pil_r]] = self.pilot_value
        if self.gear.ace:
            grids = self._ace(grids, (s_idx, self._bins[r_idx]),
                              cells, cb)
        body = self._body(grids)
        body *= np.exp(2j * np.pi * self.mixer_hz * np.arange(body.size) / FS)
        rms = np.sqrt(np.mean(np.abs(body[:-self.window]) ** 2))
        burst = np.concatenate([
            np.zeros(GUARD_HEAD, dtype=np.complex128),
            self.preamble * rms,
            body,
            np.zeros(GUARD_TAIL, dtype=np.complex128)])
        if self.clip:
            burst = self._clip_filter(burst)
        return burst

    def _ace(self, grids: np.ndarray, pos, cells: np.ndarray,
             cb: np.ndarray, n_iter: int = 8) -> np.ndarray:
        """Active constellation extension (Krongold-Jones clip-and-project).

        Iteratively clip each naked OFDM symbol to the gear's PAPR target and
        project the clipping distortion onto the *allowed* set: data-cell
        perturbations that push outer constellation points outward (never
        toward a decision boundary), everything else -- pilots, inner points,
        inactive bins -- restored exactly. The extension preserves decision
        regions before the subsequent burst clipping and filtering passes.
        """
        lmax = np.empty(cb.size)
        for b in np.unique(cb):
            lmax[cb == b] = const_for(int(b)).points.real.max()
        out_re = np.abs(cells.real) > lmax - 1e-9
        out_im = np.abs(cells.imag) > lmax - 1e-9
        sr, si = np.sign(cells.real), np.sign(cells.imag)
        # ortho FFT: mean time-sample power == mean grid power
        limit = np.sqrt(np.mean(np.abs(grids) ** 2)) * 10 ** (
            self.gear.clip_papr_db / 20)
        X = grids
        for _ in range(n_iter):
            x = np.fft.ifft(X, norm="ortho", axis=1)
            mag = np.abs(x)
            x = np.where(mag > limit, x * (limit / np.maximum(mag, 1e-12)), x)
            d = np.fft.fft(x, norm="ortho", axis=1)[pos] - cells
            ext = (np.where((d.real * sr > 0) & out_re, d.real, 0)
                   + 1j * np.where((d.imag * si > 0) & out_im, d.imag, 0))
            X = grids.copy()
            X[pos] = cells + ext
        return X

    def _clip_filter(self, x: np.ndarray) -> np.ndarray:
        active = x[GUARD_HEAD : x.size - GUARD_TAIL]
        limit = np.sqrt(np.mean(np.abs(active) ** 2)) * 10 ** (
            self.gear.clip_papr_db / 20)
        for _ in range(2):     # second pass re-clips filter-induced regrowth
            mag = np.abs(x)
            x = np.where(mag > limit, x * (limit / np.maximum(mag, 1e-12)), x)
            x = self._bandpass(x)
        return x

    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        """Centered linear convolution with the 8193-tap bandpass FIR.

        Padding covers the entire convolution. Discarding the group delay
        preserves acquisition timing; the burst guards contain the retained
        filter tails. Finite-length truncation is included in spectrum tests.
        """
        n = x.size
        m = next_fast_len(n + self._filter.size - 1)
        if m not in self._masks:
            self._masks[m] = fft(self._filter, m)
        y = ifft(fft(x, m) * self._masks[m])
        delay = self._filter.size // 2
        return y[delay:delay + n]

    # -- receive --------------------------------------------------------------
    def receive(self, samples: np.ndarray, n_symbols: int | None = None,
                tx_mask: np.ndarray | None = None,
                loading: np.ndarray | None = None, dd: int = 0,
                blank: float = 0.0, feedback=None, fb_iters: int = 1):
        """Raw analytic samples -> (hard bits, LLRs, result).

        ``n_symbols`` bounds the body length (the signaling field carries it
        in the full protocol); ``tx_mask``/``loading`` describe carriers the
        transmitter skipped or loaded differently; ``dd`` > 0 runs the
        decision-directed re-estimation pass. ``blank`` > 0 enables the
        impulse blanker (threshold in units of the local median envelope)
        plus the robust noise estimate and LLR weight cap. ``feedback`` is
        the decoder-aided loop: a callable LLRs -> (coded stream bits, known
        bool mask) or None; codeword-exact cells become virtual pilots for
        ``fb_iters`` re-estimation passes. All scattered-gear only.
        """
        return self._receive_scattered(samples, n_symbols, tx_mask,
                                       loading, dd, blank, feedback, fb_iters)

    @staticmethod
    def blank_impulses(samples: np.ndarray, factor: float,
                       block: int = 2048) -> np.ndarray:
        """Zero samples whose envelope exceeds ``factor`` x the local median.

        The per-block median tracks the burst-vs-silence power steps while a
        sub-ms impulse (a few dozen samples) cannot move it; for the OFDM
        signal's Rayleigh envelope P(|x| > 3.5 median) ~ 2e-4, so the blanker
        is inert on clean captures.
        """
        mag = np.abs(samples)
        n_blk = -(-mag.size // block)
        pad = np.pad(mag, (0, n_blk * block - mag.size), mode="edge")
        med = np.repeat(np.median(pad.reshape(n_blk, block), axis=1),
                        block)[: mag.size]
        return np.where(mag > factor * med, 0, samples)

    def _receive_scattered(self, samples: np.ndarray,
                           n_symbols: int | None = None,
                           tx_mask: np.ndarray | None = None,
                           loading: np.ndarray | None = None, dd: int = 0,
                           blank: float = 0.0, feedback=None,
                           fb_iters: int = 1):
        if blank:
            samples = self.blank_impulses(samples, blank)
        det = self.detector.detect(samples)
        if det is None:
            raise ValueError("preamble detector found no burst")
        aligned = remove_cfo(samples, det.coarse_cfo_hz + self.mixer_hz, FS)[
            max(0, det.frame_start):]
        sym = self.n_fft + self.cp
        avail = aligned.size // sym
        n_syms = avail if n_symbols is None else min(n_symbols, avail)
        if n_syms == 0:
            raise ValueError("no complete OFDM symbols after alignment")

        def grids_of(al):
            useful = al[: n_syms * sym].reshape(n_syms, sym)[:, self.cp:]
            return np.fft.fft(useful, norm="ortho", axis=1)

        G = grids_of(aligned)
        mask = self.lattice_mask(n_syms)
        df, stag, period = self.gear.lattice
        # shifts congruent to 0 mod df re-read pilot cells and tie the true
        # offset: keep the search inside the lattice's unambiguous range
        d = fading.integer_cfo(G, self._carriers, mask, self.n_fft,
                               search=min(2, df - 1), step=period)
        if d:
            aligned = remove_cfo(aligned, d * self.spacing, FS)
            G = grids_of(aligned)
        Y = G[:, self._bins]

        sigma_t, sigma_f = self.gear.est_kernel
        est = fading.estimate(Y, mask, pilot_value=self.pilot_value,
                              step=period, stag=stag, df=df, robust=blank > 0,
                              sigma_t=sigma_t, sigma_f=sigma_f)
        s_idx, r_idx, cb, off = self.cell_map(n_syms, tx_mask, loading)

        def demap(est_):
            # dead carriers have H ~ 0 *and* weight 0: keep the 0-weight LLRs
            # finite (a NaN would survive HARQ combining forever)
            H = np.where(np.abs(est_.H) < 1e-9, 1.0, est_.H)
            Z = Y * np.exp(-1j * est_.common_phase)[:, None] / H
            z = Z[s_idx, r_idx]
            w = est_.weights[s_idx, r_idx]
            if blank and w.any():
                # residual impulse energy inflates |H|^2 on hit cells: cap
                # the confidence any one cell can claim
                w = np.minimum(w, 8.0 * np.median(w[w > 0]))
            llr = np.empty(int(off[-1]))
            hard = np.empty(int(off[-1]), dtype=np.int64)
            for b in np.unique(cb):
                sel = np.nonzero(cb == b)[0]
                c = const_for(int(b))
                idx = (off[sel][:, None] + np.arange(b)).ravel()
                llr[idx] = c.llr(z[sel], w[sel])
                hard[idx] = c.demodulate(z[sel])
            return z, llr, hard

        def refine(z, bits=None, known=None):
            # every decided data cell becomes a virtual pilot, densifying the
            # lattice so the surface can be re-smoothed with narrower kernels
            # (less bias, better tracking); cells from CRC-clean codewords
            # (``bits``/``known``, decoder-aided pass) enter *exactly*
            ref = np.zeros((n_syms, self.gear.n_carriers),
                           dtype=np.complex128)
            ref[mask] = self.pilot_value
            for b in np.unique(cb):
                sel = np.nonzero(cb == b)[0]
                c = const_for(int(b))
                vals = c.modulate(c.demodulate(z[sel]))
                if known is not None:
                    ks = np.nonzero(known[sel])[0]
                    if ks.size:
                        idx = (off[sel[ks]][:, None] + np.arange(b)).ravel()
                        vals[ks] = c.modulate(bits[idx])
                ref[s_idx[sel], r_idx[sel]] = vals
            dense = mask.copy()
            dense[s_idx, r_idx] = True
            dense[:, est.masked] = False
            return fading.estimate(
                Y, dense, pilot_value=self.pilot_value, step=period,
                stag=stag, df=df, robust=blank > 0, ref=ref,
                sigma_t=min(0.4 * period, sigma_t),
                sigma_f=min(1.0, sigma_f))

        z, llr, hard = demap(est)
        for _ in range(dd):
            est = refine(z)
            z, llr, hard = demap(est)
        for _ in range(fb_iters if feedback is not None else 0):
            out = feedback(llr)
            if out is None:
                break
            fb_bits, fb_known = out
            stream = int(off[-1])
            bits = np.zeros(stream, dtype=np.int64)
            good = np.zeros(stream, dtype=bool)
            n = min(fb_bits.size, stream)
            bits[:n], good[:n] = fb_bits[:n], fb_known[:n]
            if stream > n:
                # the tail is transmit()'s PN9 filler: free virtual pilots
                bits[n:] = pn9(stream - n)
                good[n:] = True
            cell_known = np.add.reduceat(good, off[:-1]) == cb
            est = refine(z, bits, cell_known)
            z, llr, hard = demap(est)
        res = FadingResult(
            frame_start=max(0, det.frame_start),
            cfo_hz=det.coarse_cfo_hz + d * self.spacing,
            n_symbols=n_syms, noise_var=est.noise_var,
            carrier_snr=est.carrier_snr, masked=est.masked,
            metric=det.metric,
            pilot_snr=np.mean(np.abs(est.H) ** 2, axis=0) / est.noise_var)
        return hard, llr, res

    # -- audio ----------------------------------------------------------------
    @staticmethod
    def to_audio(x: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0) * x.real

    @staticmethod
    def from_audio(audio: np.ndarray) -> np.ndarray:
        return analytic(audio)
