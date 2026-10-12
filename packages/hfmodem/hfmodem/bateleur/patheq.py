# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An equalizer that models the channel as a few paths, and tracks their gains.

The Watterson channels MIL-STD-188-110B table XX is stated on are two paths a
fixed delay apart, each with its own fading gain. `phy.Equalizer` learns the
inverse of that channel directly: twenty-odd taps adapted by RLS, all of which
have to follow the fading. At 5 Hz of Doppler spread no forgetting factor
serves both the noise and the fades: the taps are either too noisy or too late,
and 2400 bit/s on the 2 ms / 5 Hz channel lost 0.29 of its bits at 30 dB.

This one estimates what the channel is made of instead:

- **Paths**, once, from the known symbols ahead of the data: the lags at which
  they correlate, found by CLEAN (take the strongest, subtract its pattern,
  repeat) over short chunks so that a path fading in one chunk still counts
  from the others; then to a sixteenth of a sample by least squares.
- **Gains**, continually: two or three complex numbers, fitted by RLS to the
  received samples through the pulse the transmitter used, from symbols
  already decided. Two numbers track 5 Hz where twenty taps could not.
- **Frequency and timing**, as the gains' common rotation and as the fit's
  error against the pulse's slope; each is a second-order loop, so a steady
  offset or a sound card's clock error leaves no standing lag.
- **The equalizer**, computed from that model every few symbols: the MMSE
  decision-feedback filter for the channel the gains describe.

The MMSE noise covariance includes a white floor: the pulse leaves part of
the twice-symbol-rate input space empty, where an unloaded inverse amplifies
small channel-fit errors. Frequency and timing loops remove residual drift;
their bounds and the periodic path search are set below.

The interface is `phy.Equalizer`'s, so `phy.Selector` and the stream receivers
drive either.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, resample_poly

from . import phy
from .spec import KNOWN_POINTS, SPS

_PTS = KNOWN_POINTS                     # 8-PSK, then the fifth roots

#: Sixteenths of a sample: the resolution of the pulse table and of the delays.
_FINE = 16
#: Pulse samples either side of its centre that the equalizer model keeps.
_SPAN = 44
#: Pulse samples either side of its centre the table holds: room for the
#: transmit trim's ringing (``phy.TRIM_HZ``), which runs out past 16 symbols.
_HALF = 64
#: The main lobe the gain tracker fits: symbols further out are undecided.
_TRACK = 12
#: Lags searched for paths either side of acquisition's timing, in samples.
_REACH = 16 * SPS
#: Symbols of known content the path search uses at most.
_FIND_SYMBOLS = 2880
#: Paths are searched for again this often, over this many decided symbols:
#: on a 1 Hz channel a path can sit in a fade for the whole of the first
#: search, and on the air paths come and go.
_RESEARCH_EVERY = 1200
_RESEARCH_SYMBOLS = 1200
#: The loops' limits. A sound card is not off by more than a few hundred ppm,
#: and acquisition leaves a few hertz; a loop past either is chasing a model
#: that is wrong, not the signal.
_MAX_PPM = 300e-6
_MAX_HZ = 6.0


class _Pulse:
    span = _SPAN
    """What one transmitted symbol looks like after the receiver's matched
    filter, at a sixteenth of a sample; peak 1 at the symbol's own time."""

    def __init__(self, trim: bool, half: int = _HALF, freq_hz: float = 0.0) -> None:
        self.half = half
        lead = 400
        a = np.concatenate([np.zeros(lead), phy.modulate(np.zeros(1, dtype=int), trim=trim),
                            np.zeros(lead)])
        n = np.arange(a.size)
        if freq_hz:
            # Arriving off frequency, the pulse meets the matched filter off
            # centre: one edge of its spectrum is cut and the other passed
            # into the filter's skirt. Derotated, what is left is no longer
            # the pulse sent (C.6.3's 75 Hz: 64-QAM on the poor channel lost
            # one bit in a hundred to it).
            a = np.real(hilbert(a) * np.exp(2j * np.pi * freq_hz * n / phy.FS))
        at = lead + phy.leadin_samples()
        bb = phy.downconvert(a) * np.exp(-2j * np.pi * freq_hz * (n - at) / phy.FS)
        q = bb[at - half: at + half + 1] / bb[at]
        self.q = resample_poly(q, _FINE, 1)
        self.dq = np.gradient(self.q) * _FINE           # per sample
        self.centre = half * _FINE
        self.trim, self.freq_hz = trim, freq_hz

    def _idx(self, t, span):
        """Table index below ``t`` and the fraction past it: the pulse is read
        between its sixteenth-sample points, not at the nearest. The nearest
        point is up to a thirty-second of a sample out, which on the pulse's
        slope produces an error near -30 dB."""
        t = np.asarray(t, dtype=np.float64)
        x = t * _FINE + self.centre
        idx = np.floor(x).astype(np.int64)
        frac = x - idx
        ok = (np.abs(t) <= span) & (idx >= 0) & (idx + 1 < self.q.size)
        return idx, frac, ok

    def _read(self, table, t, span):
        idx, frac, ok = self._idx(t, span)
        out = np.zeros(np.shape(t), dtype=np.complex128)
        i, f = idx[ok], frac[ok]
        out[ok] = table[i] * (1 - f) + table[i + 1] * f
        return out

    def at(self, t, span: float | None = None) -> np.ndarray:
        return self._read(self.q, t, self.span if span is None else span)

    def slope(self, t, span: float | None = None) -> np.ndarray:
        return self._read(self.dq, t, self.span if span is None else span)


#: Built once: the far end sends either shape (``phy.modulate``'s trim), and
#: the path fit says which.
_PULSES: dict[tuple, _Pulse] = {}


#: The carrier offset below which the pulse is taken as sent, and the step
#: between the tables for larger offsets.
_PULSE_HZ = 5.0


def _pulse(trim: bool, half: int = _HALF, span: int = _SPAN, freq_hz: float = 0.0) -> _Pulse:
    """The pulse table, ``half`` samples either side of its centre, read out
    to ``span`` of them by default; as it looks arriving ``freq_hz`` off."""
    f = _PULSE_HZ * round(freq_hz / _PULSE_HZ)
    key = (trim, half, span, f)
    if key not in _PULSES:
        _PULSES[key] = _Pulse(trim, half, f)
        _PULSES[key].span = span
    return _PULSES[key]


# --------------------------------------------------------------------------
# Finding the paths
# --------------------------------------------------------------------------

def _chunk_wave(sym: np.ndarray, pulse: _Pulse, delay: float, n0: int, n1: int,
                span: float) -> np.ndarray:
    """Samples ``n0 .. n1`` (relative to symbol 0) that ``sym`` makes through
    one path at ``delay``; ``sym`` is zero where the content is unknown."""
    n = np.arange(n0, n1)
    reach = int(span + abs(delay)) // SPS + 2
    m = np.arange(max(0, n0 // SPS - reach), min(sym.size, n1 // SPS + reach + 1))
    t = n[:, None] - SPS * m[None, :] - delay
    return pulse.at(t, span) @ sym[m]


def find_paths(bb: np.ndarray, s0: int, sym: np.ndarray, pulse: _Pulse, *,
               chunk: int = 64, most: int = 3, rel: float = 0.1,
               sep: int = 3, full: float = 0.0,
               chunks: list[tuple[int, int]] | None = None) -> list[float]:
    """Whole-sample path delays relative to ``s0``, in time order. ``sym`` is
    each symbol's value, zero where the receiver does not know it; a chunk
    less than ``full`` known is passed over, its unknown symbols being so
    much noise to the correlation. ``chunks``, symbol ranges [c0, c1) of
    ``sym``, replaces the regular grid: a waveform's known runs, say."""
    lo, hi = -_REACH, _REACH
    power: dict[int, float] = {}
    if chunks is None:
        chunks = [(c0, c0 + chunk) for c0 in range(0, sym.size - chunk + 1, chunk)]
    for c0, c1 in chunks:
        s = sym[c0:c1]
        if not np.any(s) or np.mean(s != 0) < full:
            continue
        core = _chunk_wave(s, pulse, 0.0, -_TRACK, SPS * s.size + _TRACK, _TRACK)
        a = s0 + SPS * c0 - _TRACK + lo
        if a < 0 or a + (hi - lo) + core.size > bb.size:
            continue
        seg = bb[a: a + (hi - lo) + core.size]
        corr = np.correlate(seg, core, "valid")          # lag lo .. hi
        ac = np.correlate(core, core, "full")            # lag -(N-1) .. N-1
        zero = core.size - 1
        e0 = ac[zero].real
        if e0 <= 0:
            continue
        resid = corr.copy()
        lags = np.arange(corr.size)
        for _ in range(most + 2):
            i = int(np.argmax(np.abs(resid)))
            if abs(resid[i]) < 1e-12:
                break
            amp = resid[i] / e0
            power[i + lo] = power.get(i + lo, 0.0) + abs(amp) ** 2
            d = lags - i + zero
            ok = (d >= 0) & (d < ac.size)
            resid[ok] -= amp * ac[d[ok]]
    if not power:
        return [0.0]
    ranked = sorted(power.items(), key=lambda kv: -kv[1])
    top = ranked[0][1]
    out: list[int] = []
    for lag, pw in ranked:
        if pw < rel * top or len(out) >= most:
            break
        if all(abs(lag - o) >= sep for o in out):
            out.append(lag)
    return sorted(float(o) for o in out)


def refine_paths(bb: np.ndarray, s0: int, sym: np.ndarray, delays: list[float],
                 pulse: _Pulse, *, chunk: int = 32, step: float = 1 / _FINE,
                 reach: float = 0.75, rounds: int = 2, most_chunks: int = 40,
                 full: float = 0.0):
    """Delays to a sixteenth of a sample, by coordinate descent on the residual
    of a least-squares fit of the gains in each chunk (the gains may fade from
    chunk to chunk; the delays may not). Returns the delays, the residual
    power per sample, and the first chunk's gains."""
    n_ch = min(sym.size // chunk, most_chunks + 2)
    spans = [(c * chunk * SPS, (c + 1) * chunk * SPS) for c in range(1, n_ch - 1)
             if np.any(sym[c * chunk:(c + 1) * chunk])
             and np.mean(sym[c * chunk:(c + 1) * chunk] != 0) >= full]
    if not spans:
        return list(delays), 0.05, np.full(len(delays), 0.5 + 0j)
    obs = [bb[s0 + n0: s0 + n1] for n0, n1 in spans]
    total = sum(o.size for o in obs)

    def wave(d):
        return [_chunk_wave(sym, pulse, d, n0, n1, pulse.span) for n0, n1 in spans]

    def fit(cols):
        cost, first = 0.0, None
        for ci, y in enumerate(obs):
            X = np.stack([col[ci] for col in cols], axis=1)
            g, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ g
            cost += float(np.vdot(r, r).real)
            if first is None:
                first = g
        return cost, first

    delays = list(delays)
    cols = [wave(d) for d in delays]
    best_cost, gains = fit(cols)
    for _ in range(rounds):
        for p in range(len(delays)):
            for grid in (np.arange(-reach, reach + 1e-9, 4 * step),
                         np.arange(-3 * step, 3 * step + 1e-9, step)):
                centre = delays[p]
                for dd in grid:
                    if dd == 0:
                        continue
                    trial = cols[:p] + [wave(centre + dd)] + cols[p + 1:]
                    cost, g = fit(trial)
                    if cost < best_cost:
                        best_cost, gains = cost, g
                        delays[p], cols[p] = centre + dd, trial[p]
    return delays, best_cost / total, gains


def _known_values(plan: phy.Plan, n: int) -> np.ndarray:
    """The first ``n`` symbols' values where the plan knows them, else 0."""
    n = min(n, len(plan))
    out = np.zeros(n, dtype=np.complex128)
    known = plan.kind[:n] == phy.KNOWN
    out[known] = _PTS[plan.value[:n][known]]
    return out


# --------------------------------------------------------------------------
# The equalizer
# --------------------------------------------------------------------------

class PathEqualizer:
    """``phy.Equalizer``'s interface over a path model of the channel.

    ``lam`` is the gain tracker's forgetting factor per symbol; ``load`` the
    white floor added to the noise model, relative to the noise; ``f_loop``,
    ``t_kp`` and ``t_ki`` the frequency and timing loops' gains. ``n_pre`` and
    ``n_post`` are the forward filter's reach in symbols beyond the paths.
    """

    lam = 0.9 ** (1 / SPS)
    lag, load = 4, 0.3
    f_loop, t_kp, t_ki = 0.02, 0.01, 5e-5
    find_symbols, track_every, track_span = _FIND_SYMBOLS, 2, _TRACK

    def __init__(self, *, n_pre=6, n_post=10, every=4, full=0.0,
                 span=_SPAN, pulse_half=_HALF) -> None:
        #: The pulse the channel model is built of: its table's half-length
        #: and how far of it the model reads. 188-110 needs no more than the
        #: defaults; a trimmed pulse under 64-QAM needs its ringing, which
        #: the defaults cut off at -33 dB (STANAG 4539 uses 160 and 128).
        self.span, self.pulse_half = span, pulse_half
        #: The share of a chunk's symbols the first path search needs known
        #: to use it. 188-110's frames are a third known and more, and every
        #: chunk helps; STANAG 4539's a ninth, and only its preamble does.
        self.full = full
        self.ff_pre, self.ff_post = n_pre, n_post
        self.every = every
        #: How far behind ``pos`` the equalizer still reads, in symbols; the
        #: stream receivers keep that much of the buffer.
        self.n_pre = _REACH // SPS + 4

    # -- streaming ----------------------------------------------------------

    def start(self, start: int, freq_hz: float, gain: complex,
              constellation: tuple[int, ...]) -> None:
        self.s0 = start
        self.pos = start
        self.k = 0
        self.stopped = False
        self._const = np.array(constellation)
        self._qam = np.iscomplexobj(self._const)       # points, decided nearest
        self._w0 = 2 * np.pi * freq_hz / phy.FS         # rad/sample, acquisition's
        self._ready = False
        self.soft = phy.History()
        self.sig = phy.History()
        self.kerr = phy.History()
        self.perr = phy.History()
        self.walsh = phy.History()
        self.walsh_sig = phy.History()
        self.delays: list[float] = []

    def needs(self) -> int:
        if not self._ready:
            return self.s0 + SPS * self.find_symbols + _REACH + self.span + SPS
        return self.pos + self._J[-1] + 1

    def advance(self, buf: np.ndarray, base: int, plan: phy.Plan, stop=None) -> int:
        if self.stopped:
            return 0
        if not self._ready:
            # The paths are found from the first ``find_symbols`` symbols, so
            # nothing is equalized until they have arrived — unless this is a
            # whole recording, which is searched as far as it goes.
            n = min(self.find_symbols, len(plan))
            have = (base + buf.size - self.s0 - _REACH - self.span - SPS) // SPS
            if have < n:
                return 0
            self._setup(buf, base, plan, n)
        return self._run(buf, base, plan, stop)

    # -- set-up ---------------------------------------------------------------

    def _setup(self, buf, base, plan, n) -> None:
        sym = _known_values(plan, n)
        rel = self.s0 - base
        x = buf * np.exp(-1j * self._w0 * (base + np.arange(buf.size) - self.s0))
        best = None
        for trim in (True, False):
            pulse = _pulse(trim, self.pulse_half, self.span, self._w0 * phy.FS / (2 * np.pi))
            if best is None:
                delays = find_paths(x, rel, sym, pulse, full=self.full)
            else:
                delays = best[1]
            d, s2, g = refine_paths(x, rel, sym, delays, pulse, full=self.full)
            if best is None or s2 < best[2]:
                best = (pulse, d, s2, g)
        self._pulse, self.delays, s2, g = best
        self._sigma2 = max(float(s2), 1e-4)
        P = len(self.delays)
        self._a = np.asarray(g, dtype=np.complex128).copy()
        self._Pm = np.eye(P, dtype=np.complex128) * 0.1
        self._a_prev = self._a.copy()
        self._power = np.abs(self._a) ** 2               # each path's, smoothed
        # Loops.
        self._om = 0.0                                   # rad/sample, the loop's
        self._ph0, self._n0 = 0.0, self.s0
        self._tau = 0.0                                  # samples, common delay
        self._t_rate = 0.0
        self._dec = phy.SymbolWindow(np.zeros(max(len(plan), 1), dtype=np.complex128))
        self._final = -1                                 # decisions settled up to here
        self._fed = 0                                    # next sample (rel s0) to track
        self._set: list = []
        self._shift = 0
        self._next_search = _RESEARCH_EVERY + _RESEARCH_SYMBOLS
        self._geometry()
        self._ready = True

    def _geometry(self) -> None:
        """Everything that follows from the set of paths: the forward window,
        the noise model, which symbols reach the window."""
        # The forward window: T/2 taps from before the first path's lobe to
        # after the last one's, around each symbol's nominal time.
        dmin, dmax = min(self.delays), max(self.delays)
        j0 = int(np.floor(dmin / 2)) * 2 - SPS * self.ff_pre - 2
        j1 = int(np.ceil(dmax / 2)) * 2 + SPS * self.ff_post + 2
        self._J = np.arange(j0, j1 + 1, 2)
        J = self._J
        # Noise through the matched filter, correlated as the pulse is, plus
        # the floor that keeps the filter out of the pulse's empty space.
        Rq = self._pulse.at((J[:, None] - J[None, :]).astype(float)).real
        self._Rn = Rq + self.load * np.eye(J.size)
        reach = self.span + 2 * SPS                       # room for the timing to move
        self._ivals = np.arange(int(np.floor((dmin - reach - J[-1]) / SPS)),
                                int(np.ceil((dmax + reach - J[0]) / SPS)) + 1)
        self._i0 = int(np.flatnonzero(self._ivals == 0)[0])
        self._fb = (self._ivals >= 1)
        self._fb_i = self._ivals[self._fb]
        self._layout_key = None
        self._layout()
        self._w = None
        # What is still read behind ``pos``: the tracker's lag, and the
        # re-search's window.
        self.n_pre = max(-J[0] // SPS, 0) + self.lag + _RESEARCH_SYMBOLS \
            + (_REACH + self.track_span + int(abs(self._tau))) // SPS + 8

    def _research(self, buf, base, k) -> None:
        """Search the decided symbols for a path the model does not have."""
        last = min(self._final, k - self.lag)
        first = last - _RESEARCH_SYMBOLS
        if first < 0:
            return
        sym = self._dec[first:last].copy()
        n0 = self.s0 + SPS * first
        rel = n0 - base
        if rel - _REACH - _TRACK < 0:
            return
        lo = max(0, rel - _REACH - _TRACK - SPS)
        hi = min(buf.size, rel + SPS * sym.size + _REACH + _TRACK + SPS)
        seg = buf[lo:hi] * self._derot(base + lo + np.arange(hi - lo))
        found = find_paths(seg, rel - lo, sym, self._pulse, most=3, rel=0.15)
        have = [d + self._tau for d in self.delays]
        for lag in found:
            if all(abs(lag - h) >= 3 for h in have):
                self._add_path(lag - self._tau)
                have.append(lag)

    def _add_path(self, delay: float) -> None:
        if len(self.delays) >= 3:
            weakest = int(np.argmin(self._power))
            if self._power[weakest] > 0.05 * float(np.max(self._power)):
                return                                   # all three are real
            keep = [i for i in range(len(self.delays)) if i != weakest]
            self.delays = [self.delays[i] for i in keep]
            self._a = self._a[keep]
            self._a_prev = self._a_prev[keep]
            self._power = self._power[keep]
            self._Pm = self._Pm[np.ix_(keep, keep)]
        P = len(self.delays)
        self.delays = self.delays + [float(delay)]
        self._a = np.append(self._a, 0j)
        self._a_prev = np.append(self._a_prev, 0j)
        self._power = np.append(self._power, 0.0)
        Pm = np.zeros((P + 1, P + 1), dtype=np.complex128)
        Pm[:P, :P] = self._Pm
        Pm[P, P] = 0.1
        self._Pm = Pm
        self._geometry()

    def _layout(self) -> None:
        """Tables for the current common delay: the model's columns and the
        tracker's pulse, both at a sixteenth of a sample."""
        tau = self._tau
        shift = 2 * int(np.round(tau / 2))               # the window, in whole T/2
        key = (shift, int(np.round((tau - shift) * _FINE)))
        if key == self._layout_key:
            return
        self._layout_key = key
        self._shift = shift
        J, iv = self._J, self._ivals
        t = (J[None, :] + SPS * iv[:, None]).astype(float)
        d = np.array(self.delays) + tau - shift
        self._Q = np.stack([self._pulse.at(t - dp) for dp in d])          # (P, I, J)
        dd = np.array(self.delays) + tau
        span = self.track_span
        self._o_lo = int(np.floor(dd.min())) - span - 1
        o = np.arange(self._o_lo, int(np.ceil(dd.max())) + span + 2).astype(float)
        self._qt = np.stack([self._pulse.at(o - dp, span) for dp in dd])   # (P, O)
        self._dqt = np.stack([self._pulse.slope(o - dp, span) for dp in dd])
        self._w = None                                   # recompute the filter

    def _filter(self) -> None:
        H = np.tensordot(self._a, self._Q, axes=1)       # rows: symbol k - i
        A = H[~self._fb]
        R = self._sigma2 * self._Rn + A.T @ A.conj()
        c0 = H[self._i0]
        try:
            w = np.linalg.solve(R, c0)
        except np.linalg.LinAlgError:
            w = np.zeros_like(c0)
        g = float(np.real(np.vdot(w, c0)))
        self._w = w
        self._g = g if g > 1e-6 else 1e-6
        self._b = H[self._fb] @ np.conj(w)

    # -- running --------------------------------------------------------------

    def _derot(self, n):
        return np.exp(-1j * (self._w0 * (n - self.s0) + self._ph0 + self._om * (n - self._n0)))

    def _run(self, buf, base, plan, stop) -> int:
        end = base + buf.size
        kinds, values, rots = plan.kind, plan.value, plan.rot
        self._kinds = kinds
        self._dec.grow(len(plan))
        self._dec.discard_before(plan.kind.base)
        dec = self._dec
        done = 0
        k = self.k
        while k < len(plan):
            J = self._J                                  # a new path moves it
            nk = self.s0 + SPS * k + self._shift
            if nk + J[-1] >= end:
                break
            if self._w is None or k % self.every == 0:
                self._filter()
            idx = nk + J
            if idx[0] >= base:
                u = buf[idx - base]
            else:
                u = np.zeros(J.size, dtype=np.complex128)
                ok = idx >= base
                u[ok] = buf[idx[ok] - base]
            u = u * self._derot(idx)
            y = np.vdot(self._w, u)
            back = k - self._fb_i
            ok = back >= 0
            y -= np.dot(self._b[ok], dec[back[ok]])
            y /= self._g
            var = max((1 - self._g) / self._g, 1e-4)
            kind, r = kinds[k], rots[k]
            e2 = None
            if kind == phy.SET:
                z = y * _PTS[(-r) % 8]
                dec[k] = _PTS[((0 if z.real >= 0 else 4) + r) % 8]
                self._set.append((k, y, r, values[k]))
                if len(self._set) == 32:
                    e2 = self._decide_set(plan.patterns, var)
            else:
                if kind == phy.KNOWN:
                    d = _PTS[values[k]]
                elif kind == phy.DATA and self._qam:
                    d = self._const[int(np.argmin(np.abs(self._const - y)))]
                    self.soft.append(y)
                    self.sig.append(var)
                else:
                    z = y * _PTS[(-r) % 8]
                    cand = self._const if kind == phy.DATA else np.array((0, 4))
                    j = cand[int(np.argmin(np.abs(_PTS[cand] - z)))]
                    d = _PTS[(j + r) % 8]
                    if kind == phy.DATA:
                        self.soft.append(z)
                        self.sig.append(var)
                dec[k] = d
                if kind != phy.DATA:
                    e = d - y
                    e2 = float(e.real * e.real + e.imag * e.imag)
                    self.perr.append(e2)
                    if kind == phy.KNOWN:
                        self.kerr.append(e2)
            if not self._set:
                self._final = k
            if k % self.track_every == 0:
                self._track(buf, base, k)
            if k >= self._next_search:
                self._next_search = k + _RESEARCH_EVERY
                self._research(buf, base, k)
            k += 1
            done += 1
            self.k = k
            self.pos = self.s0 + SPS * k + self._shift
            if stop is not None and e2 is not None and (kind == phy.KNOWN or kind == phy.SET) \
                    and stop(k - 1, e2):
                self.stopped = True
                break
        return done

    def _decide_set(self, patterns, var) -> float:
        ks = np.array([s[0] for s in self._set])
        ys = np.array([s[1] for s in self._set])
        rots = np.array([s[2] for s in self._set])
        base = int(self._set[0][3])
        z = ys * _PTS[(-rots) % 8]
        rows = patterns[base:base + 4]
        corr = np.real(np.conj(_PTS[rows]) @ z)
        best = int(np.argmax(corr))
        self.walsh.append(corr)
        self.walsh_sig.append(var)
        d = _PTS[(rows[best] + rots) % 8]
        self._dec[ks] = d
        e = d - ys
        self._set = []
        return float(np.mean(e.real ** 2 + e.imag ** 2))

    def _track(self, buf, base, k) -> None:
        """Fit the gains to every sample whose symbols are all decided, then
        steer the frequency and timing loops by what the fit saw."""
        last = min(k - self.lag, self._final)
        if last < 0:
            return
        # The samples every contributing symbol of which is at or before
        # ``last`` (a symbol reaches ``o_lo`` samples ahead of its own time),
        # fitted as one block: usually the four of one symbol.
        stop_at = SPS * last + self._o_lo
        if self._fed > stop_at:
            return
        rel = np.arange(self._fed, stop_at + 1)
        self._fed = stop_at + 1
        n = self.s0 + rel
        inside = (n >= base) & (n < base + buf.size)
        rel, n = rel[inside], n[inside]
        a, Pm = self._a, self._Pm
        num = den = 0.0
        if rel.size:
            qt, dqt, o_lo = self._qt, self._dqt, self._o_lo
            width = qt.shape[1]
            m_lo = max(0, -(-(int(rel[0]) - o_lo - width + 1) // SPS))
            m_hi = min(last, (int(rel[-1]) - o_lo) // SPS)
            ms = np.arange(m_lo, m_hi + 1)
            col = rel[:, None] - SPS * ms[None, :] - o_lo            # (L, M)
            ok = (col >= 0) & (col < width)
            col = np.where(ok, col, 0)
            dv = np.where(ok, self._dec[ms][None, :], 0)
            X = np.einsum("plm,lm->lp", qt[:, col], dv)               # (L, P)
            Xd = np.einsum("plm,lm->lp", dqt[:, col], dv)
            obs = buf[n - base] * self._derot(n)
            e = obs - X @ a
            # Block RLS: the block's samples weighted alike, forgotten as one.
            L = rel.size
            lam = self.lam ** L
            PXh = Pm @ X.conj().T                                     # (P, L)
            G = np.linalg.solve(lam * np.eye(L) + X @ PXh, np.eye(L))
            K = PXh @ G                                               # (P, L)
            a = a + K @ e
            Pm = (Pm - K @ X @ Pm) / lam
            Pm = 0.5 * (Pm + Pm.conj().T)
            f = 0.999 ** L
            e2 = np.abs(e) ** 2
            self._sigma2 = max(f * self._sigma2 + (1 - f) * float(np.mean(e2)), 1e-5)
            self._power = f * self._power + (1 - f) * np.abs(a) ** 2
            g = Xd @ a
            num = float(np.real(np.vdot(g, e)))
            den = float(np.real(np.vdot(g, g)))
        self._a, self._Pm = a, Pm
        # Frequency: the gains' common rotation since the last symbol.
        if self.f_loop:
            z = np.vdot(self._a_prev, a)
            if abs(z) > 1e-12:
                n = self.s0 + SPS * k
                self._ph0 += self._om * (n - self._n0)
                self._n0 = n
                lim = 2 * np.pi * _MAX_HZ / phy.FS
                self._om = max(-lim, min(lim, self._om + self.f_loop * float(np.angle(z)) / SPS))
        self._a_prev = a.copy()
        # Timing: the residual against the pulse's slope says how far late.
        if self.t_kp and den > 1e-9:
            te = -num / den
            te = max(-0.5, min(0.5, te))
            lim = _MAX_PPM * SPS
            self._t_rate = max(-lim, min(lim, self._t_rate + self.t_ki * te))
            self._tau += self._t_rate + self.t_kp * te
            self._layout()

