# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Serial waveform pulse, acquisition and equalizer selection.

The pulse is root-raised cosine with roll-off 0.1. The default transmitter
also applies a 2950 Hz low-pass filter. The declared audio spans and the
Welch measurement used to check them are in ``core.occupied`` and
``tests/bateleur/test_phy.py``.

The receiver mixes down and matched-filters at four samples per symbol.
``Selector`` compares a symbol-spaced RLS decision-feedback equalizer with
``patheq.PathEqualizer``, using probe error to select each frame. At 75 bit/s
only the path equalizer runs; 4539 uses ``blockeq.BlockEqualizer``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, firwin

from .spec import BAUD, CARRIER_HZ, FS, KNOWN_POINTS, SPS, psk8

ROLLOFF = 0.1
PULSE_SPAN = 24                                     # symbols either side


def rrc() -> np.ndarray:
    """Unit-energy root-raised-cosine taps."""
    t = np.arange(-PULSE_SPAN * SPS, PULSE_SPAN * SPS + 1) / SPS
    a = ROLLOFF
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - a + 4 * a / np.pi
        elif abs(abs(ti) - 1 / (4 * a)) < 1e-9:
            h[i] = (a / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
                                       + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - a)) + 4 * a * ti * np.cos(np.pi * ti * (1 + a))) \
                / (np.pi * ti * (1 - (4 * a * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


_PULSE = rrc()

#: The top-edge trim: 401 taps, Kaiser beta 8, so ~100 Hz from pass to stop.
TRIM_HZ = 2950.0
_TRIM = firwin(401, TRIM_HZ, fs=FS, window=("kaiser", 8.0))
_TRIM_DELAY = (_TRIM.size - 1) // 2


def modulate(indices: np.ndarray, trim: bool = True) -> np.ndarray:
    """8-PSK indices -> real audio at ``FS``, peak 1.0, carrier at 1800 Hz;
    as `modulate_symbols`."""
    return modulate_symbols(psk8(indices), trim)


def modulate_symbols(sym: np.ndarray, trim: bool = True) -> np.ndarray:
    """Complex symbols (8-PSK points, or QAM ones) -> real audio at ``FS``,
    peak 1.0, carrier at 1800 Hz.

    The filters' tails are kept, so the audio runs longer at each end than the
    symbols do; ``leadin_samples`` says by how much.

    ``trim`` cuts the top edge at `TRIM_HZ`, so that the emission fits a rig
    whose transmit filter stops at 3000 Hz. The cut costs the receiver: it
    bends the pulse the matched filter expects, leaving self-interference near
    -20 dB where the untrimmed pulse leaves -51. A rig that passes 3.2 kHz
    should be sent the whole pulse. Untrimmed audio is padded by the trim's
    delay so that either one has the same timing.
    """
    sym = np.asarray(sym, dtype=np.complex128)
    up = np.zeros(sym.size * SPS, dtype=np.complex128)
    up[::SPS] = sym
    bb = np.convolve(up, _PULSE)
    # The carrier's time is the audio's own sample index, which the trim
    # delays by _TRIM_DELAY: then ``downconvert`` sees each symbol at its
    # transmitted phase rather than half a cycle round.
    t = (np.arange(bb.size) + _TRIM_DELAY) / FS
    audio = np.real(bb * np.exp(2j * np.pi * CARRIER_HZ * t))
    if trim:
        audio = fftconvolve(audio, _TRIM)
    else:
        audio = np.concatenate([np.zeros(_TRIM_DELAY), audio, np.zeros(_TRIM_DELAY)])
    return audio / np.max(np.abs(audio))


def sample_count(symbols: int) -> int:
    """Rendered samples including both pulse and trim tails."""
    return symbols * SPS + _PULSE.size - 1 + _TRIM.size - 1


def leadin_samples() -> int:
    """Audio samples ahead of the first symbol's peak: the pulse's half-length
    and the trim's delay."""
    return PULSE_SPAN * SPS + _TRIM_DELAY


def downconvert(audio: np.ndarray) -> np.ndarray:
    """Real audio at ``FS`` -> matched-filtered complex baseband, four samples a
    symbol, aligned so that a symbol modulated at sample ``n`` peaks at ``n``."""
    x = np.asarray(audio, dtype=np.float64)
    t = np.arange(x.size) / FS
    bb = x * np.exp(-2j * np.pi * CARRIER_HZ * t)
    y = fftconvolve(bb, _PULSE)
    return y[PULSE_SPAN * SPS: PULSE_SPAN * SPS + x.size] * np.sqrt(2)


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------

@dataclass
class Hit:
    """A known sequence found in the baseband."""

    sample: int             # baseband index of the sequence's first symbol
    score: float            # normalised differential-correlation peak, 0..1
    freq_hz: float          # carrier offset estimate
    gain: complex           # complex channel gain at the first symbol


def correlate(bb: np.ndarray, ref_indices, chunk: int = 16, freq_hz: float = 0.0):
    """Per-chunk coherent correlation of ``bb`` against a known symbol run, at
    every sample offset. Returns ``(chunks, energy)``: ``chunks[k, n]`` is chunk
    ``k`` correlated with the run starting at sample ``n``, and ``energy[n]``
    the received energy those symbols cover."""
    ref = psk8(ref_indices)
    n_chunks = ref.size // chunk
    x = bb
    if freq_hz:
        x = bb * np.exp(-2j * np.pi * freq_hz * np.arange(bb.size) / FS)
    span = ref.size * SPS
    out_len = x.size - span + 1
    if out_len <= 0:
        return np.zeros((n_chunks, 0), complex), np.zeros(0)
    chunks = np.empty((n_chunks, out_len), dtype=np.complex128)
    for k in range(n_chunks):
        h = np.zeros(chunk * SPS - SPS + 1, dtype=np.complex128)
        h[::SPS] = np.conj(ref[k * chunk:(k + 1) * chunk])
        c = fftconvolve(x, h[::-1], mode="valid")      # c[m] = sum x[m+SPS i] h_i
        off = k * chunk * SPS
        chunks[k] = c[off: off + out_len]
    p = np.abs(x) ** 2
    cs = np.concatenate([[0.0], np.cumsum(p)])
    energy = (cs[span: span + out_len] - cs[:out_len]) / SPS
    return chunks, energy


def differential_metric(chunks: np.ndarray, energy: np.ndarray, chunk: int):
    """``|sum c_{k+1} c_k*|`` normalised to 1 for a clean match, and the
    carrier offset its phase implies."""
    d = np.sum(chunks[1:] * np.conj(chunks[:-1]), axis=0)
    n = chunks.shape[0]
    norm = (energy / n) * chunk * (n - 1) + 1e-30
    score = np.abs(d) / norm
    freq = np.angle(d) * BAUD / (2 * np.pi * chunk)
    return score, freq


#: How much baseband acquisition correlates at once: a few seconds, so a long
#: recording costs time in proportion and memory not at all.
_SCAN_SAMPLES = 8 * BAUD * SPS


def _balance(seg: np.ndarray, run: int, n: int) -> np.ndarray:
    """1 where a run's energy is spread across it, falling to 0 where it
    all lies in one half. The metric is normalised by the whole run's
    energy, so where noise fills only its last few chunks — a receiver
    unmuted after its own transmission, a burst of static after quiet — it
    averages over a few chunk pairs rather than all of them and clears the
    threshold by chance; a real preamble fills the run."""
    cs = np.concatenate([[0.0], np.cumsum(np.abs(seg) ** 2)])
    half = run // 2
    i = np.arange(n)
    e1 = cs[i + half] - cs[i]
    e2 = cs[i + run] - cs[i + half]
    bal = 2 * np.minimum(e1, e2) / (e1 + e2 + 1e-30)
    return np.clip(2 * bal, 0.0, 1.0)             # uneven past 3:1 costs score


def find(bb: np.ndarray, ref_indices, *, chunk: int = 16, threshold: float = 0.35,
         start: int = 0, freq_hz: float = 0.0) -> Hit | None:
    """Locate the first ``ref_indices`` in ``bb`` at or after ``start``.

    The earliest peak that clears ``threshold`` wins, refined to the best
    sample within one run's length after it: a run that begins and ends alike
    (188-110's sync does) scores half marks where only its tail overlaps the
    signal's head, a full run earlier than the real one, and the first
    crossing is that ghost. The frequency estimate is the differential one,
    refined by the phase slope across chunks once corrected for it.

    ``freq_hz`` shifts the search: a chunk turned half round by the offset
    scores nothing, so an offset beyond some tens of hertz is searched for
    from nearer it.
    """
    run = len(ref_indices) * SPS
    reach = run + 8 * SPS                       # the refinement's window
    at = start
    while at < bb.size - run:
        seg = bb[at: at + _SCAN_SAMPLES + run + reach]
        chunks, energy = correlate(seg, ref_indices, chunk, freq_hz=freq_hz)
        if chunks.shape[1] == 0:
            return None
        score, freq = differential_metric(chunks, energy, chunk)
        score = score * _balance(seg, run, score.size)
        over = np.flatnonzero(score[:_SCAN_SAMPLES] > threshold)
        if over.size:
            lo = over[0]
            n = lo + int(np.argmax(score[lo: lo + reach]))
            return _refine(bb, ref_indices, chunk, at + n, float(score[n]),
                           freq_hz + float(freq[n]))
        at += _SCAN_SAMPLES
    return None


def _refine(bb, ref_indices, chunk, n, score, f1) -> Hit:
    """Derotate by the estimate and fit the residual phase slope. The slice
    starts at the hit, so once derotated the chunks' common phase is the
    channel's phase at the first symbol."""
    seg = bb[n: n + len(ref_indices) * SPS + SPS]
    for _ in range(2):
        ch, _ = correlate(seg, ref_indices, chunk, freq_hz=f1)
        c = ch[:, 0]
        if c.size > 2:
            ph = np.unwrap(np.angle(c))
            f1 += np.polyfit(np.arange(ph.size), ph, 1)[0] * BAUD / (2 * np.pi * chunk)
    ch, _ = correlate(seg, ref_indices, chunk, freq_hz=f1)
    g = np.sum(ch[:, 0]) / len(ref_indices)
    return Hit(n, score, f1, complex(g))


# --------------------------------------------------------------------------
# The equalizer
# --------------------------------------------------------------------------

KNOWN, DATA, PROBE_BPSK, SET = 0, 1, 2, 3


class History:
    """A retained window indexed by its original stream positions."""

    def __init__(self, values=(), base=0):
        self.values = list(values)
        self.base = base

    def __len__(self):
        return self.base + len(self.values)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def _index(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return slice(max(self.base, start) - self.base, max(self.base, stop) - self.base, step)
        if index < 0:
            index += len(self)
        if index < self.base:
            raise IndexError("history before retained window")
        return index - self.base

    def __getitem__(self, index):
        return self.values[self._index(index)]

    def __setitem__(self, index, value):
        self.values[self._index(index)] = value

    def __delitem__(self, index):
        del self.values[self._index(index)]

    def append(self, value):
        self.values.append(value)

    def extend(self, values):
        self.values.extend(values)

    def discard_before(self, index):
        cut = max(0, min(index - self.base, len(self.values)))
        del self.values[:cut]
        self.base += cut


class SymbolWindow:
    """An array whose retained symbols keep their absolute symbol indices."""

    def __init__(self, values, base=0):
        self.values = np.asarray(values)
        self.base = base

    @property
    def size(self):
        return self.base + self.values.size

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __eq__(self, value):
        return self.values == value

    def _index(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.size)
            return slice(max(self.base, start) - self.base, max(self.base, stop) - self.base, step)
        return np.asarray(index) - self.base

    def __getitem__(self, index):
        return self.values[self._index(index)]

    def __setitem__(self, index, value):
        self.values[self._index(index)] = value

    def grow(self, end):
        if end > self.size:
            self.values = np.concatenate([self.values, np.zeros(end - self.size, dtype=self.values.dtype)])

    def discard_before(self, index):
        cut = max(0, min(index - self.base, self.values.size))
        self.values = self.values[cut:].copy()
        self.base += cut


@dataclass
class Plan:
    """What the equalizer should expect, symbol by symbol.

    ``kind`` is KNOWN (``value`` is the transmitted 8-PSK index, randomiser
    included), DATA (one of ``constellation`` rotated by ``rot``), PROBE_BPSK
    (a probe whose content the receiver is unsure of; decided among ``rot``
    and ``rot + 4``), or SET: one symbol of a run of 32 that is decided whole
    as one of four ``patterns`` rows — ``value`` + 0..3 — which is 188-110's
    75 bit/s.
    """

    kind: np.ndarray
    value: np.ndarray
    rot: np.ndarray
    constellation: tuple[int, ...]
    patterns: np.ndarray | None = None

    def __post_init__(self):
        for name in ("kind", "value", "rot"):
            if not isinstance(getattr(self, name), SymbolWindow):
                setattr(self, name, SymbolWindow(getattr(self, name)))

    def discard_before(self, index):
        for values in (self.kind, self.value, self.rot):
            values.discard_before(index)

    def __len__(self) -> int:
        return self.kind.size

    @staticmethod
    def concat(plans: list["Plan"]) -> "Plan":
        return Plan(SymbolWindow(np.concatenate([p.kind for p in plans]), plans[0].kind.base),
                    SymbolWindow(np.concatenate([p.value for p in plans]), plans[0].value.base),
                    SymbolWindow(np.concatenate([p.rot for p in plans]), plans[0].rot.base),
                    plans[0].constellation,
                    next((p.patterns for p in plans if p.patterns is not None), None))


class Equalizer:
    """Symbol-spaced DFE adapted by exponentially-weighted RLS, with a phase
    loop ahead of it and a slow timing loop.

    What each choice replaced, measured on the channels in ``sim`` — the
    two-path Watterson channels MIL-STD-188-110B table XX is stated on:

    - RLS, not LMS: on the 2 ms / 1 Hz channel no LMS step size both followed
      the fades and settled quietly enough for 8-PSK.
    - Trained on every known symbol, and on a data symbol only when its
      decision is confident (squared error under ``dd``). Fed its own
      decisions without that gate, RLS at this forgetting factor learns a
      burst of wrong ones as fast as right ones; trained on the probes alone,
      16 updates in 48 symbols left it chasing the channel.
    - A forward section reaching 12 symbols past the one being decided and
      8 before it. When the later of two equal paths is the stronger the
      channel is non-minimum-phase, and the forward taps have to reach past
      the second path's echo to undo it; with 6 symbols of reach, 2400 bit/s
      on the 2 ms channel made a fifth of its symbol decisions wrong at 40 dB.
    - Symbol-spaced, not T/2: at a roll-off of 0.1 half of a T/2 input's
      space holds nothing but noise, where RLS's inverse grows without bound.
    - ``P`` kept Hermitian and its trace capped, because without that it is
      the arithmetic, not the channel, that decides when the equalizer fails.

    Together these took 2400 bit/s, long interleave, on the 2 ms / 1 Hz
    channel at 18 dB and 100 ppm of clock error from a coded bit error rate
    of 1e-3 to none in 112 kbit; table XX asks for 1e-5.

    ``n_pre`` and ``n_post`` are the forward taps' reach, in symbols, either
    side of the symbol being decided; the window is placed by acquisition on
    the strongest path.
    """

    n_pre, n_post, n_fb = 8, 12, 7
    lam, delta = 0.98, 0.02
    kp, ki = 0.03, 0.0004
    t_kp, t_ki = 0.002, 1e-6
    dd = 0.3

    # -- streaming ----------------------------------------------------------

    def start(self, start: int, freq_hz: float, gain: complex,
              constellation: tuple[int, ...]) -> None:
        """Begin at baseband sample ``start`` (an absolute index in whatever
        stream ``advance`` will be shown), with acquisition's estimates."""
        nf = self.n_pre + self.n_post + 1
        self._nf, self._centre = nf, self.n_pre
        n = nf + self.n_fb
        self._w = np.zeros(n, dtype=np.complex128)          # y = w^H v
        g = gain if abs(gain) > 1e-12 else 1.0
        self._w[self._centre] = np.conj(1.0 / g)
        self._P = np.eye(n, dtype=np.complex128) / self.delta
        self._p_cap = n / self.delta
        self._v = np.zeros(n, dtype=np.complex128)
        self._theta = 0.0
        self._omega = 2 * np.pi * freq_hz / BAUD
        self._const = np.array(constellation)
        #: A QAM constellation is given as its points: decided nearest,
        #: unrotated (its scrambling permutes the points, it does not turn
        #: them).
        self._qam = np.iscomplexobj(self._const)
        #: A decision trains the taps only if nearer its point than this; a
        #: QAM's points are closer than 8-PSK's, so the gate closes with them
        #: (half the squared least distance), or a fade's wrong decisions
        #: train it as readily as right ones.
        self._dd = self.dd
        if self._qam and self.dd:
            c = self._const
            dmin2 = np.min(np.abs(c[:, None] - c[None, :])[~np.eye(c.size, dtype=bool)]) ** 2
            self._dd = min(self.dd, dmin2 / 2)
        self._sigma2 = 0.1
        self._t_rate = self._t_acc = 0.0
        self.pos = start
        self.k = 0
        self.stopped = False
        self.soft = History()           # derotated output, DATA symbols
        self.sig = History()              # noise variance at each of them
        self.kerr = History()             # |error|^2 at each KNOWN symbol
        self.perr = History()             # ... and at every probe, decided too

    def needs(self) -> int:
        """The absolute sample index one past what the next symbol reads."""
        return self.pos + (self._nf - self._centre) * SPS

    def advance(self, buf: np.ndarray, base: int, plan: Plan, stop=None) -> int:
        """Equalize every symbol of ``plan`` from ``self.k`` on that ``buf``
        (whose first sample is absolute index ``base``) holds the samples for.
        Samples before ``base`` read as zero. Returns the symbols done.

        ``stop(k, err2)`` may end the run; it is called at every symbol whose
        kind is KNOWN, and once it says so ``stopped`` stays set.
        """
        if self.stopped:
            return 0
        nf, centre = self._nf, self._centre
        w, P, v = self._w, self._P, self._v
        theta, omega = self._theta, self._omega
        sigma2 = self._sigma2
        t_rate, t_acc = self._t_rate, self._t_acc
        pos = self.pos
        pts = _PTS
        const, bpsk = self._const, _BPSK
        kinds, values, rots = plan.kind, plan.value, plan.rot
        soft, sig, kerr = self.soft, self.sig, self.kerr
        lead = centre * SPS
        end = base + buf.size
        done = 0
        k = self.k
        while k < len(plan):
            lo = pos - lead
            hi = lo + (nf - 1) * SPS + 1
            if hi > end:
                break
            if lo >= base:
                u = buf[lo - base: hi - base: SPS]
            else:
                u = np.zeros(nf, dtype=np.complex128)
                idx = lo + SPS * np.arange(nf)
                ok = idx >= base
                u[ok] = buf[idx[ok] - base]
            v[:nf] = u * np.exp(-1j * theta)
            y = np.vdot(w, v)
            kind = kinds[k]
            r = rots[k]
            if kind == KNOWN:
                d = pts[values[k]]
            elif kind == DATA and self._qam:
                d = const[int(np.argmin(np.abs(const - y)))]
                soft.append(y)
                sig.append(sigma2)
            else:
                z = y * pts[(-r) % 8]
                cand = const if kind == DATA else bpsk
                j = cand[int(np.argmin(np.abs(pts[cand] - z)))]
                d = pts[(j + r) % 8]
                if kind == DATA:
                    soft.append(z)
                    sig.append(sigma2)
            e = d - y
            e2 = float(e.real * e.real + e.imag * e.imag)
            if kind != DATA:
                self.perr.append(e2)
            if kind == KNOWN:
                kerr.append(e2)
                sigma2 = 0.95 * sigma2 + 0.05 * e2
            else:
                sigma2 = 0.99 * sigma2 + 0.01 * e2
            if kind != DATA or (self._dd and e2 < self._dd):
                P = self._rls(w, P, v, e)
            # Phase loop, on the decision.
            phi = float(np.angle(y * np.conj(d))) if abs(y) > 1e-9 else 0.0
            omega += self.ki * phi
            theta += omega + self.kp * phi
            # The feedback section sees past decisions, newest first.
            v[nf + 1:] = v[nf:-1]
            v[nf] = d
            pos += SPS
            # Timing: follow the sound cards' clocks, not the fades. Clock
            # error is a steady drift, so a slow second-order loop estimates
            # its *rate* from the forward taps' centre of energy, once a
            # 48-symbol frame. The window then moves a whole symbol at a time,
            # and the taps and the RLS state move with it, which to a
            # symbol-spaced equalizer is no change at all; the drift in between
            # is the equalizer's to follow, and at 100 ppm it is a quarter of a
            # symbol a second. Moving a quarter symbol at a time instead made
            # the channel change under the taps at every step, which on the
            # 2 ms channel tripled the symbol error rate.
            if k % 48 == 47:
                p = np.abs(w[:nf]) ** 2
                cog = np.sum(p * np.arange(nf)) / (np.sum(p) + 1e-12)
                te = (cog - centre) * SPS                       # samples late
                t_rate += self.t_ki * te
                t_acc += t_rate + self.t_kp * te
                while abs(t_acc) >= SPS:
                    step = 1 if t_acc > 0 else -1
                    pos += step * SPS
                    t_acc -= step * SPS
                    P = self._shift_taps(w, P, nf, step)
            k += 1
            done += 1
            if stop is not None and kind == KNOWN and stop(k - 1, e2):
                self.stopped = True
                break
        self._P = P
        self._theta, self._omega = theta, omega
        self._sigma2 = sigma2
        self._t_rate, self._t_acc = t_rate, t_acc
        self.pos, self.k = pos, k
        return done

    def _rls(self, w, P, v, error):
        pv = P @ v
        gain = pv / (self.lam + np.real(np.vdot(v, pv)))
        w += gain * np.conj(error)
        P = (P - np.outer(gain, np.conj(pv))) / self.lam
        P = 0.5 * (P + P.conj().T)
        trace = np.real(np.trace(P))
        if trace > self._p_cap:
            P *= self._p_cap / trace
        return P

    def _shift_taps(self, w, P, nf, step):
        """Slide the forward taps one symbol against a window that moved by
        ``step``: tap i now sees what tap i + step saw. The tap that falls off
        is dropped and the one that arrives starts at zero, knowing nothing."""
        idx = np.arange(P.shape[0])
        ff = np.arange(nf)
        src = ff + step
        keep = (src >= 0) & (src < nf)
        new_w = np.zeros(nf, dtype=w.dtype)
        new_w[ff[keep]] = w[src[keep]]
        w[:nf] = new_w
        perm = idx.copy()
        perm[:nf] = np.clip(src, 0, nf - 1)
        Q = P[np.ix_(perm, perm)]
        fresh = ff[~keep]
        Q[fresh, :] = 0
        Q[:, fresh] = 0
        Q[fresh, fresh] = 1.0 / self.delta
        return Q

class Selector:
    """Two equalizers on the same symbols, and each frame's soft decisions
    taken from the one whose probes around that frame came out nearer right.

    The first is `patheq.PathEqualizer`, which models the channel as a few
    paths and tracks their gains; it is used for the simulated MIL-STD-188-110B table XX cases
    on the 5 Hz channels, where this module's RLS equalizer lost 0.29 of
    2400 bit/s's bits. The second is that RLS equalizer, kept because it
    assumes nothing about the channel's shape: a channel that is not a few
    discrete paths is one it can still follow. It has to be clearly better
    (``bias``) over ``window`` frames either side before it is believed,
    because where both work they score alike and a choice made on noise is a
    coin toss. With no probes (75 bit/s) there is nothing to choose on, and
    the path model is used alone.

    ``unknown`` and ``known`` are one frame's data and probe symbol counts;
    ``pre_known`` the known symbols before the first frame. ``known`` may
    instead be a function of the frame number, for a waveform whose frames
    do not all end alike (STANAG 4539 reinserts its preamble every 72).
    """

    bias, window = 0.7, 2

    def __init__(self, unknown: int, known, pre_known: int, *, path_cls=None) -> None:
        from .patheq import PathEqualizer
        self.unknown, self.pre = unknown, pre_known
        self._known_of = known if callable(known) else (lambda f, n=known: n)
        self.known = known if not callable(known) else known(0)
        self._probe_counts = np.r_[0, np.cumsum([self._known_of(f) for f in range(72)])]
        self.a = (path_cls or PathEqualizer)()
        self.b = Equalizer() if known else None
        self._soft = History()
        self._sig = History()
        self._frames = 0
        self._final = False
        self._stopped = False
        self._checked = 0
        #: The symbol the signal was judged lost at. Over a whole recording
        #: both equalizers run to its end before the probes are judged, so
        #: what they produced past this is noise.
        self.stop_k: int | None = None

    def start(self, start, freq_hz, gain, constellation) -> None:
        self.a.start(start, freq_hz, gain, constellation)
        if self.b is not None:
            self.b.start(start, freq_hz, gain, constellation)

    def advance(self, buf, base, plan, stop=None) -> int:
        """Both, over the same samples. The signal is judged lost on the
        better of the two at each probe, so neither one's failure ends a
        reception the other is following."""
        if self.b is None:
            return self.a.advance(buf, base, plan, stop=stop)
        if self._stopped:
            return 0
        done = self.a.advance(buf, base, plan)
        self.b.advance(buf, base, plan)
        if stop is not None:
            n = min(len(self.a.perr), len(self.b.perr))
            for i in range(self._checked, n):
                if stop(self._k_of(i), min(self.a.perr[i], self.b.perr[i])):
                    self._stopped = True
                    self.stop_k = self._k_of(i)
                    break
            self._checked = n
        return done

    def _kstart(self, f: int) -> int:
        cycles, within = divmod(f, 72)
        return int(self.pre + cycles * self._probe_counts[-1] + self._probe_counts[within])

    def _frame_of(self, probe: int) -> int:
        if probe < self.pre:
            return -1
        cycles, within = divmod(probe - self.pre, int(self._probe_counts[-1]))
        return cycles * 72 + int(np.searchsorted(self._probe_counts, within, side="right")) - 1

    def discard(self, data: int, known: int = 0, walsh: int = 0) -> None:
        self._soft.discard_before(data)
        self._sig.discard_before(data)
        for eq in (self.a, self.b):
            if eq is None:
                continue
            cut = data if self.b is None else min(data, self._frames * self.unknown)
            eq.soft.discard_before(cut)
            eq.sig.discard_before(cut)
            eq.kerr.discard_before(min(known, len(eq.kerr) - 8192))
            eq.perr.discard_before(min(self._checked, max(0, self._kstart(max(0, self._frames - 2 * self.window - 2)))))
        self.a.walsh.discard_before(walsh)
        self.a.walsh_sig.discard_before(walsh)

    def _k_of(self, probe: int) -> int:
        """The symbol index of probe number ``probe``, for ``stop``'s gate."""
        if probe < self.pre:
            return probe
        f = self._frame_of(probe)
        # Frame f's probes follow its data: the pre-known symbols, f frames
        # of data and probes, and frame f's data.
        return probe + (f + 1) * self.unknown

    # What callers read, as from one equalizer.
    pos = property(lambda self: self.a.pos)
    k = property(lambda self: self.a.k)
    stopped = property(lambda self: self._stopped if self.b is not None else self.a.stopped)
    kerr = property(lambda self: self.a.kerr)
    walsh = property(lambda self: self.a.walsh)
    walsh_sig = property(lambda self: self.a.walsh_sig)
    n_pre = property(lambda self: max(self.a.n_pre, self.b.n_pre if self.b else 0))
    delays = property(lambda self: self.a.delays)

    @property
    def soft(self) -> list:
        self._merge()
        return self._soft

    @property
    def sig(self) -> list:
        self._merge()
        return self._sig

    def _merge(self) -> None:
        if self.b is None:
            self._soft, self._sig = self.a.soft, self.a.sig
            return
        u, w = self.unknown, self.window
        ready = min(len(self.a.soft), len(self.b.soft)) // u
        probes = min(len(self.a.perr), len(self.b.perr))
        if not (self.stopped or self._final):
            # A frame's score reaches ``window`` frames ahead; wait for them.
            ready = min(ready, self._frame_of(probes) - w)
        for f in range(self._frames, max(ready, self._frames)):
            lo = self._kstart(max(0, f - w + 1))
            hi = self._kstart(f + w)
            sa = np.mean(self.a.perr[lo:hi]) if hi > lo else 0.0
            sb = np.mean(self.b.perr[lo:hi]) if hi > lo else 0.0
            src = self.b if sb < self.bias * sa else self.a
            self._soft.extend(src.soft[f * u:(f + 1) * u])
            self._sig.extend(src.sig[f * u:(f + 1) * u])
        self._frames = max(ready, self._frames)



_PTS = KNOWN_POINTS                     # 8-PSK, then the fifth roots
_BPSK = np.array((0, 4))


# --------------------------------------------------------------------------
# Soft demapping
# --------------------------------------------------------------------------

def demap(z: np.ndarray, sigma2: np.ndarray, gray: tuple[int, ...]) -> np.ndarray:
    """Max-log LLRs, first bit most significant, positive for 0.

    ``gray[v]`` is the 8-PSK index that carries bit group ``v``."""
    m = len(gray)
    nb = m.bit_length() - 1
    pts = psk8(np.array(gray))
    d2 = np.abs(z[:, None] - pts[None, :]) ** 2          # (n, m)
    s2 = np.maximum(np.asarray(sigma2), 1e-3)[:, None]
    out = np.empty((z.size, nb))
    v = np.arange(m)
    for j in range(nb):
        bit = (v >> (nb - 1 - j)) & 1
        d1 = np.min(d2[:, bit == 1], axis=1)
        d0 = np.min(d2[:, bit == 0], axis=1)
        out[:, j] = (d1 - d0) / s2[:, 0]
    return out.ravel()


def loss_detector(after: int, rate: float, level: float = 0.8):
    """End after sustained probe error; short fades remain inside the message."""
    run = 0.0
    def stop(k, e2):
        nonlocal run
        if k < after:
            return False
        run = (1 - rate) * run + rate * min(e2, 2.0)
        return run > level
    return stop


def snr_db(known_error) -> float:
    k = np.asarray(known_error)
    if not k.size:
        return float("nan")
    error = float(np.median(k[:max(1, k.size // 2)]))
    return float(10 * np.log10(1 / max(error, 1e-6)))
