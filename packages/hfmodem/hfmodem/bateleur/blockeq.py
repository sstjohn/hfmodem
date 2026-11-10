# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Path equalizer with gains fitted on known probes.

Probe fits avoid training the channel on bursts of wrong QAM decisions.
Gains are interpolated between probes, then refined inside each data block.
Each pass equalizes forwards and backwards; the nearer decisions are kept.
Soft variance comes from local decision error and the adjacent probe errors,
since variance from the channel fit alone misses a lost block.

Timing follows the probe fit against the pulse slope. Frequency follows the
common gain rotation. The loops and search bounds are fixed production values.
"""
from __future__ import annotations

import numpy as np

from . import phy
from .patheq import _PULSE_HZ, _REACH, _TRACK, PathEqualizer, _pulse, find_paths
from .spec import KNOWN_POINTS, SPS

#: How far off the frequency loop may go, all told (C.6.3 and C.6.4 ask for
#: 75 Hz).
_MAX_HZ = 100.0
#: The most of one known run its gains are fitted on. A preamble can be
#: seconds of known symbols (appendix D's tests send twenty superframes,
#: 11520 of them): fitted whole, the model would not fit in memory, and the
#: gains would be an average over seconds of fading.
_LONG_RUN = 576

_PTS = KNOWN_POINTS                     # 8-PSK, then the fifth roots


class BlockEqualizer(PathEqualizer):
    """``PathEqualizer``'s interface and set-up; gains from the known runs.

    ``fit_span`` is how much of the pulse (samples either side of its peak)
    the probe fit models: the lobe, so that a 31-symbol probe keeps enough
    samples that no unknown symbol reaches. ``f_gain`` and the timing gains
    are per probe; ``f_rate`` is the frequency loop's second order, which
    follows a Doppler sweep (C.6.4's 3.5 Hz/s) without lagging it, out to
    ``_MAX_HZ`` either side of nothing.
    """

    fit_span, f_gain, f_rate = 20, 0.2, 0.05
    bt_kp, bt_ki = 0.3, 0.02
    passes, dd_every, dd_span = 2, 64, 48
    var_window, var_gain = 32, 2.0

    def __init__(self):
        super().__init__(full=1.0, pulse_half=160, span=128, n_pre=8, n_post=16, every=8)

    def _setup(self, buf, base, plan, n) -> None:
        super()._setup(buf, base, plan, n)
        #: Each known run's gains, by the run's first symbol: (the symbol
        #: time they belong to, the gains).
        self._anch: dict[int, tuple[float, np.ndarray]] = {}
        self._last_fit: int | None = None       # the run fitted most recently
        #: Each known run's mean squared error as equalized.
        self._run_acc: dict[int, tuple[int, float]] = {}
        self._run_err: dict[int, float] = {}
        self._pending_measure = None
        self._om_rate = 0.0                     # rad/sample per sample
        self._left: int | None = None           # the known run behind the data

    # -- the known runs ---------------------------------------------------------

    def _runs(self, kinds) -> None:
        """Where each run of one kind starts and ends, for the plan as it is."""
        if getattr(self, "_runs_n", None) == (kinds.base, kinds.size):
            return
        change = np.flatnonzero(np.diff(kinds) != 0) + 1 + kinds.base
        starts = np.concatenate([[kinds.base], change])
        ends = np.concatenate([change, [kinds.size]])
        self._run_of = np.repeat(np.arange(starts.size), ends - starts)
        self._starts, self._ends = starts, ends
        self._runs_n = (kinds.base, kinds.size)

    def _run_bounds(self, kinds, k):
        """The run of one kind that ``k`` is in: [lo, hi)."""
        self._runs(kinds)
        i = self._run_of[k - kinds.base]
        return int(self._starts[i]), int(self._ends[i])

    def _reach(self) -> tuple[int, int]:
        """Samples before and after a symbol's time that it reaches, through
        the lobe the fit models, over all paths."""
        d = np.array(self.delays) + self._tau
        return int(np.ceil(self.fit_span - d.min())), int(np.ceil(self.fit_span + d.max()))

    def _fit_run(self, buf, base, r0: int, r1: int) -> bool:
        """Fit the gains on known run [r0, r1): the samples every symbol
        reaching which lies inside it. False if they have not all arrived.
        A run longer than ``_LONG_RUN`` is fitted on its last that many, the
        ones nearest the data after it."""
        before, after = self._reach()
        key, r0 = r0, max(r0, r1 - _LONG_RUN)
        n_lo = self.s0 + SPS * r0 + after + 1           # the first symbol's reach is inside
        n_hi = self.s0 + SPS * (r1 - 1) - before - 1
        if n_hi + 1 > base + buf.size:
            return False
        if n_hi <= n_lo + 8:
            # Too short to leave any sample clear of the unknown: take its
            # middle, what reaches it from outside being so much noise.
            mid = self.s0 + SPS * (r0 + r1) // 2
            n_lo, n_hi = mid - 2 * SPS, mid + 2 * SPS
        n = np.arange(max(n_lo, base), n_hi + 1)
        if n.size < 4:
            return True
        rel = n - self.s0
        dd = np.array(self.delays) + self._tau
        span = self.fit_span
        ms = np.arange(max(0, r0 - 2), min(r1 + 2, self._dec.size))
        t = rel[:, None] - SPS * ms[None, :]                          # (L, M)
        sym = self._dec[ms]
        X = np.stack([self._pulse.at(t - d, span) @ sym for d in dd], axis=1)    # (L, P)
        Xd = np.stack([self._pulse.slope(t - d, span) @ sym for d in dd], axis=1)
        obs = buf[n - base] * self._derot(n)
        a, *_ = np.linalg.lstsq(X, obs, rcond=None)
        e = obs - X @ a
        s2 = float(np.mean(np.abs(e) ** 2))
        self._sigma2 = max(0.7 * self._sigma2 + 0.3 * s2, 1e-5)
        centre = (float(rel[0]) + float(rel[-1])) / 2 / SPS
        # Frequency: the gains' common rotation since the last probe.
        if self._last_fit is not None and self.f_gain:
            c_prev, a_prev = self._anch[self._last_fit]
            z = np.vdot(a_prev, a)
            dn = (centre - c_prev) * SPS
            if abs(z) > 1e-12 and dn > 0:
                n0 = self.s0 + SPS * int(centre)
                self._ph0 += self._om * (n0 - self._n0)
                self._n0 = n0
                w = float(np.angle(z)) / dn
                om = self._om + self._om_rate * dn + self.f_gain * w
                self._om_rate += self.f_rate * w / dn
                lim = 2 * np.pi * _MAX_HZ / phy.FS
                om = max(-lim - self._w0, min(lim - self._w0, om))
                # The gains so far were fitted under the old rotation, which
                # the new one meets at n0: turned to match, a block between
                # two of them is interpolated between like and like.
                for r, (c, g) in self._anch.items():
                    self._anch[r] = (c, g * np.exp(-1j * (om - self._om)
                                                   * (self.s0 + SPS * c - n0)))
                self._om = om
                f = (self._w0 + self._om) * phy.FS / (2 * np.pi)
                if abs(f - self._pulse.freq_hz) > _PULSE_HZ / 2:
                    self._pulse = _pulse(self._pulse.trim, self.pulse_half, self.span, f)
                    self._layout_key = None
                    self._layout()
        # Timing: the residual against the pulse's slope says how far late.
        g = Xd @ a
        den = float(np.real(np.vdot(g, g)))
        if den > 1e-9 and (self.bt_kp or self.bt_ki):
            te = max(-0.5, min(0.5, -float(np.real(np.vdot(g, e))) / den))
            lim = 300e-6 * SPS * (r1 - r0 + 256)
            self._t_rate = max(-lim, min(lim, self._t_rate + self.bt_ki * te))
            self._tau += self._t_rate + self.bt_kp * te
            self._layout()
        self._anch[key] = (centre, a)
        self._last_fit = key
        self._power = 0.9 * self._power + 0.1 * np.abs(a) ** 2
        for old in [r for r in self._anch if r < key - 4096]:
            del self._anch[old]
        for history in (self._run_acc, self._run_err):
            for old in [key for key in history if key < r0 - 4096]:
                del history[old]
        return True

    def _gains_at(self, k: int, right: int) -> np.ndarray:
        """The gains at data symbol ``k``: from the known run behind it to the
        one ahead (``right``), linearly."""
        c1, a1 = self._anch[right]
        if self._left is None or self._left not in self._anch:
            return a1
        c0, a0 = self._anch[self._left]
        if c1 <= c0:
            return a1
        f = min(max((k - c0) / (c1 - c0), 0.0), 1.0)
        return _between(a0, a1, f)

    def _fitted(self, buf, base, kinds, r0) -> bool:
        if r0 in self._anch:
            return True
        _, r1 = self._run_bounds(kinds, r0)
        return self._fit_run(buf, base, r0, r1)

    # -- running ----------------------------------------------------------------

    def _run(self, buf, base, plan, stop) -> int:
        end = base + buf.size
        kinds, values, rots = plan.kind, plan.value, plan.rot
        self._kinds = kinds
        self._dec.grow(len(plan))
        self._dec.discard_before(plan.kind.base)
        known = np.flatnonzero(kinds == phy.KNOWN) + kinds.base
        self._dec[known] = _PTS[values[known]]
        done = 0
        k = self.k
        while k < len(plan):
            if kinds[k] == phy.KNOWN:
                r0, _ = self._run_bounds(kinds, k)
                if not self._fitted(buf, base, kinds, r0):
                    break
                self._left = r0
                a = self._anch[r0][1]
            else:
                d0, d1 = self._run_bounds(kinds, k)
                if d1 >= len(plan) or not self._fitted(buf, base, kinds, d1):
                    break                   # the probe after it has not come yet
                if k == d0:
                    if not self._block(buf, base, end, d0, d1, kinds, values, rots):
                        break
                    n = d1 - d0
                    k, done = d1, done + n
                    self._final = d1 - 1
                    self.k = k
                    self.pos = self.s0 + SPS * k + self._shift
                    continue
                a = self._gains_at(k, d1)
            if not self._equalize(buf, base, end, k, a, kinds, values, rots):
                break
            e2 = self._last_e2
            if e2 is not None:
                r0, r1 = self._run_bounds(kinds, k)
                n_, s_ = self._run_acc.get(r0, (0, 0.0))
                self._run_acc[r0] = (n_ + 1, s_ + e2)
                self._run_err[r0] = (s_ + e2) / (n_ + 1)
                if k == r1 - 1 and self._pending_measure is not None:
                    self._measure(*self._pending_measure)
                    self._pending_measure = None
            self._final = k
            if kinds[k] == phy.KNOWN and k + 1 < len(plan) and kinds[k + 1] != phy.KNOWN:
                self._probe_search(buf, base, k + 1)
            k += 1
            done += 1
            self.k = k
            self.pos = self.s0 + SPS * k + self._shift
            if stop is not None and e2 is not None and stop(k - 1, e2):
                self.stopped = True
                break
        return done

    def _add_path(self, delay: float) -> None:
        """A path found: the anchors so far are gains of the old set, so
        they go, and the next known run's alone serves until there are two."""
        super()._add_path(delay)
        self._anch = {}
        self._last_fit = self._left = None

    #: Known runs a path search reads, the latest last: about a second's.
    SEARCH_RUNS = 8

    def _probe_search(self, buf, base, k) -> None:
        """Search the latest known runs (up to ``k``) for a path the model
        lacks. The decided data is no help: if a path is missing the
        decisions are what is wrong."""
        self._runs(self._kinds)
        i = int(np.searchsorted(self._starts, k)) - 1
        runs = []
        while i >= 0 and len(runs) < self.SEARCH_RUNS:
            r0, r1 = int(self._starts[i]), int(self._ends[i])
            if self._kinds[r0] == phy.KNOWN and r1 <= k:
                runs.append((max(r0, r1 - _LONG_RUN), r1))
            i -= 1
        if len(runs) < 2:
            return
        first = runs[-1][0]
        sym = np.where(self._kinds[first:k] == phy.KNOWN, self._dec[first:k], 0)
        rel = self.s0 + SPS * first - base
        if rel - _REACH - _TRACK < 0:
            return
        lo = max(0, rel - _REACH - _TRACK - SPS)
        hi = min(buf.size, rel + SPS * sym.size + _REACH + _TRACK + SPS)
        seg = buf[lo:hi] * self._derot(base + lo + np.arange(hi - lo))
        found = find_paths(seg, rel - lo, sym, self._pulse, most=3, rel=0.15,
                           chunks=[(r0 - first, r1 - first) for r0, r1 in runs])
        have = [d + self._tau for d in self.delays]
        for lag in found:
            if all(abs(lag - h) >= 3 for h in have):
                self._add_path(lag - self._tau)
                have.append(lag)

    def _block(self, buf, base, end, d0, d1, kinds, values, rots) -> bool:
        """A whole data block, ``passes`` times: the gains between the probes
        at first, then refitted inside the block to what the last pass
        decided."""
        last = self.s0 + SPS * (d1 - 1) + self._shift + self._J[-1]
        if last >= end:
            return False
        mark = len(self.soft)
        pts = None
        self._block_rots = rots
        for p in range(self.passes):
            del self.soft[mark:], self.sig[mark:]
            self._w = None
            for k in range(d0, d1):
                a = self._gains_at(k, d1) if pts is None else _interp(pts, k)
                self._equalize(buf, base, end, k, a, kinds, values, rots)
            if p + 1 < self.passes:
                self._backward(buf, base, end, d0, d1, kinds, values, rots, pts, mark)
                pts = self._dd_points(buf, base, d0, d1)
        self._backward(buf, base, end, d0, d1, kinds, values, rots, pts, mark)
        # The probe after this block must be equalized before its variance is used.
        self._pending_measure = (mark, d0, d1, rots, self._left)
        return True

    def _measure(self, mark: int, d0: int, d1: int, rots, left=None) -> None:
        z = np.array(self.soft[mark:])
        d = self._dec[d0:d1]
        if not self._qam:
            d = d * _PTS[(-rots[d0:d1]) % 8]
        e2 = np.abs(z - d) ** 2
        w = self.var_window
        kern = np.ones(w) / w
        local = np.convolve(np.pad(e2, (w // 2, w - 1 - w // 2), mode="edge"), kern, "valid")
        pl = self._run_err.get(left, None) if left is not None else None
        pr = self._run_err.get(d1, None)
        if pl is None and pr is None:
            probe = 0.0
        elif pl is None or pr is None:
            probe = pl if pr is None else pr
        else:
            probe = pl + (pr - pl) * (np.arange(d1 - d0) + 0.5) / (d1 - d0)
        self.sig[mark:] = list(np.maximum(self.var_gain * local, probe))

    def _backward(self, buf, base, end, d0, d1, kinds, values, rots, pts, mark) -> None:
        zf = np.array(self.soft[mark:])
        sf = np.array(self.sig[mark:])
        df = self._dec[d0:d1].copy()
        fb, fbi = self._fb, self._fb_i
        self._fb = self._ivals <= -1
        self._fb_i = self._ivals[self._fb]
        self._w = None
        del self.soft[mark:], self.sig[mark:]
        try:
            for k in range(d1 - 1, d0 - 1, -1):
                a = self._gains_at(k, d1) if pts is None else _interp(pts, k)
                self._equalize(buf, base, end, k, a, kinds, values, rots)
        finally:
            self._fb, self._fb_i = fb, fbi
            self._w = None
        zb = np.array(self.soft[mark:])[::-1]
        sb = np.array(self.sig[mark:])[::-1]
        db = self._dec[d0:d1].copy()
        rot = 1 if self._qam else _PTS[(-rots[d0:d1]) % 8]
        w = self.var_window
        kern = np.ones(w) / w

        def local(z, d):
            e2 = np.abs(z - d * rot) ** 2
            return np.convolve(np.pad(e2, (w // 2, w - 1 - w // 2), mode="edge"), kern, "valid")
        take_b = local(zb, db) < local(zf, df)
        del self.soft[mark:], self.sig[mark:]
        self.soft.extend(np.where(take_b, zb, zf))
        self.sig.extend(np.where(take_b, sb, sf))
        self._dec[d0:d1] = np.where(take_b, db, df)

    def _dd_points(self, buf, base, d0, d1) -> list[tuple[float, np.ndarray]]:
        """The gains through the block: the probes' either side, and between
        them fits to the block's own decisions every ``dd_every`` symbols."""
        c1, a1 = self._anch[d1]
        pts = []
        if self._left is not None and self._left in self._anch:
            pts.append(self._anch[self._left])
        step = self.dd_every
        dd = np.array(self.delays) + self._tau
        span = self.dd_span
        for c0 in range(d0, d1, step):
            ce = min(c0 + step, d1)
            rel = np.arange(SPS * c0, SPS * ce) + self._shift
            n = self.s0 + rel
            if n[0] < base or n[-1] >= base + buf.size:
                continue
            reach = (span + int(np.max(np.abs(dd)))) // SPS + 2
            ms = np.arange(max(0, c0 - reach), min(self._dec.size, ce + reach))
            t = rel[:, None] - SPS * ms[None, :]
            X = np.stack([self._pulse.at(t - d, span) @ self._dec[ms] for d in dd], axis=1)
            obs = buf[n - base] * self._derot(n)
            g, *_ = np.linalg.lstsq(X, obs, rcond=None)
            pts.append(((c0 + ce - 1) / 2, g))
        pts.append((c1, a1))
        return pts

    def _equalize(self, buf, base, end, k, a, kinds, values, rots) -> bool:
        J = self._J
        nk = self.s0 + SPS * k + self._shift
        if nk + J[-1] >= end:
            return False
        if self._w is None or k % self.every == 0 or kinds[k] == phy.KNOWN \
                and (k == 0 or kinds[k - 1] != phy.KNOWN):
            self._a = np.asarray(a, dtype=np.complex128)
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
        dec = self._dec
        back = k - self._fb_i
        ok = (back >= 0) & (back < dec.size)
        y -= np.dot(self._b[ok], dec[back[ok]])
        y /= self._g
        var = max((1 - self._g) / self._g, 1e-4)
        kind, r = kinds[k], rots[k]
        self._last_e2 = None
        if kind == phy.KNOWN:
            d = _PTS[values[k]]
        elif self._qam:
            d = self._const[int(np.argmin(np.abs(self._const - y)))]
            self.soft.append(y)
            self.sig.append(var)
        else:
            z = y * _PTS[(-r) % 8]
            j = self._const[int(np.argmin(np.abs(_PTS[self._const] - z)))]
            d = _PTS[(j + r) % 8]
            self.soft.append(z)
            self.sig.append(var)
        dec[k] = d
        if kind != phy.DATA:
            e = d - y
            e2 = float(e.real * e.real + e.imag * e.imag)
            self.perr.append(e2)
            self.kerr.append(e2)
            self._last_e2 = e2
        return True


def _interp(pts: list[tuple[float, np.ndarray]], k: float) -> np.ndarray:
    """Piecewise-linear gains through ``pts`` (in order), held at the ends."""
    if k <= pts[0][0]:
        return pts[0][1]
    for (c0, a0), (c1, a1) in zip(pts, pts[1:]):
        if k <= c1:
            f = (k - c0) / (c1 - c0) if c1 > c0 else 1.0
            return _between(a0, a1, f)
    return pts[-1][1]


def _between(a0: np.ndarray, a1: np.ndarray, f: float) -> np.ndarray:
    """The gains ``f`` of the way from ``a0`` to ``a1``: their common turn
    taken evenly, the rest linearly. What the frequency loop has not yet
    caught turns every path alike, and straight across, a turn shrinks the
    gains between (a quarter turn by 3 dB midway: the C.6.4 sweep's corners,
    where the loop runs 2 Hz behind for a second)."""
    z = np.vdot(a0, a1)
    if abs(z) < 1e-12:
        return a0 + f * (a1 - a0)
    th = float(np.angle(z))
    return (a0 + f * (a1 * np.exp(-1j * th) - a0)) * np.exp(1j * th * f)
