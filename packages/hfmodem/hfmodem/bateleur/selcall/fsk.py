# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The selcall modem: two tones, 1700 and 1870 Hz, 100 bits a second, the
phase continuous across bits.

**Which tone is which bit.** ITU-R M.493 gives the maritime service's
higher tone to the B (0) state; the UN and HFLINK diagrams of the land
mobile selcall label 1870 Hz "1", and two recordings of a Vertex VX1210
(published by HFLINK) bear them out: their known characters' 0 bits are at
1700 Hz and 1 bits at 1870 Hz, and so does a third HFLINK publishes (40 Hz
off, its clock a part in a hundred fast). That is what we send. An inverted
sideband somewhere on the way swaps them, so the receiver tries both and
says which it found.

**Receiving.** Each tone's energy over a bit (a 10 ms window, at a tenth of
a bit's steps, over a few offsets either side of the nominal tones, the
best kept) gives a soft bit. The phasing sequence is looked for at every
tenth of a bit, both ways round; found (M.493 3.3: two DX characters and
one RX, or two RX and one DX, exact), the characters are read from there.
Each DX character is read from its own copy, or its RX copy four
characters later, or the two copies' soft bits added, whichever has its
check bits right; the call ends at its third end character, or when the
signal goes. The bit rate is taken from the phasing (each half timed on
its own, to 3 %), and the timing followed a tenth of a bit at a time as the
call is read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import code as C

FS = 8000
BAUD = 100
SPB = FS // BAUD                  # 80 samples a bit
ZERO_HZ, ONE_HZ = 1700.0, 1870.0  # as land mobile radios send; ``invert`` swaps them


def modulate(chars: list[int], *, dots_s: float = 2.0, invert: bool = False,
             freq_hz: float = 0.0, amplitude: float = 0.9) -> np.ndarray:
    """Characters (``code.air_chars``) -> audio at ``FS``, after
    ``dots_s`` of dot pattern."""
    n_dots = int(round(dots_s * BAUD))
    bits = [i & 1 for i in range(n_dots)]
    for c in chars:
        bits += C.char_bits(c)
    f0, f1 = (ONE_HZ, ZERO_HZ) if invert else (ZERO_HZ, ONE_HZ)
    f = np.repeat(np.where(np.array(bits) == 1, f1, f0), SPB) + freq_hz
    ph = 2 * np.pi * np.cumsum(f) / FS
    out = amplitude * np.sin(ph)
    ramp = min(24, out.size // 2)
    if ramp:
        edge = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
        out[:ramp] *= edge
        out[-ramp:] *= edge[::-1]
    return out


def call_audio(call: C.Call, **kw) -> np.ndarray:
    return modulate(C.air_chars(C.encode(call)), **kw)


@dataclass
class Heard:
    call: C.Call | None
    chars: list                       # the DX characters read (None: unread)
    start: int                        # sample of the phasing's first bit
    end: int                          # sample after the last character read
    inverted: bool                    # 1870 Hz read as 1
    freq_hz: float                    # the tones' offset found


_PHASE = C.air_chars([0, 0, 0])[:12]          # the twelve phasing characters
_PHASE_BITS = np.array([b for c in _PHASE for b in C.char_bits(c)])
_OFFSETS = (0.0, -20.0, 20.0, -40.0, 40.0)
_STEP = SPB // 10                             # soft bits at a tenth of a bit
_LEVEL = 20 * 10                              # each tone's level over 20 bits (in steps)
_MAX_CHARS = 64


class Receiver:
    """Audio in (at ``FS``, any chunk size), calls out.

    ``push`` returns the calls completed in what it was given. Only calls
    with their phasing are read; calls are bounded to 64 characters."""

    def __init__(self) -> None:
        self._buf = np.zeros(0)
        self._base = 0                         # sample number of _buf[0]
        self._searched = 0                     # samples searched for phasing
        self._powers = {df: np.zeros((2, 0)) for df in _OFFSETS}
        self._power_end = 0

    # -- soft bits ------------------------------------------------------------

    def _soft(self, x: np.ndarray, df: float) -> np.ndarray:
        """(E0 - E1) / (E0 + E1) for a bit window starting at every
        ``_STEP``-th sample of ``x`` (+ is 0)."""
        n = np.arange(x.size)
        k = np.ones(SPB)
        e = []
        for f in (ZERO_HZ + df, ONE_HZ + df):
            z = x * np.exp(-2j * np.pi * f * n / FS)
            c = np.convolve(z, k, "valid")[::_STEP]
            p = np.abs(c) ** 2
            # Each tone by its own level over some twenty bits: two paths
            # put notches 1/delay apart, and a fade can take one tone
            # and leave the other.
            if p.size >= _LEVEL:
                lvl = np.convolve(p, np.ones(_LEVEL), "same") / _LEVEL
            else:
                lvl = np.full(p.size, np.mean(p) if p.size else 0.0)
            e.append(p / (lvl + 1e-9 * (np.max(lvl) + 1e-30)))
        return (e[0] - e[1]) / (e[0] + e[1] + 1e-12)

    def _extend_powers(self):
        start = self._power_end - self._base
        x = self._buf[start:]
        if x.size < SPB:
            return
        n = np.arange(x.size)
        for df in _OFFSETS:
            rows = []
            for f in (ZERO_HZ + df, ONE_HZ + df):
                z = x * np.exp(-2j * np.pi * f * n / FS)
                sums = np.concatenate(([0j], np.cumsum(z)))
                rows.append(np.abs((sums[SPB:] - sums[:-SPB])[::_STEP]) ** 2)
            self._powers[df] = np.concatenate((self._powers[df], rows), axis=1)
        self._power_end += len(rows[0]) * _STEP

    def _scan_soft(self, start, df):
        e = []
        for p in self._powers[df][:, start // _STEP:]:
            if p.size >= _LEVEL:
                lvl = np.convolve(p, np.ones(_LEVEL), "same") / _LEVEL
            else:
                lvl = np.full(p.size, np.mean(p) if p.size else 0.0)
            e.append(p / (lvl + 1e-9 * (np.max(lvl) + 1e-30)))
        return (e[0] - e[1]) / (e[0] + e[1] + 1e-12)

    # -- the stream -------------------------------------------------------------

    def push(self, audio: np.ndarray) -> list[Heard]:
        self._buf = np.concatenate([self._buf, np.asarray(audio, dtype=np.float64)])
        self._extend_powers()
        out = []
        while True:
            h = self._scan()
            if h is None:
                break
            out.append(h)
        # Keep what a phasing search still needs.
        keep = 20 * SPB * 12 + 2 * SPB
        cut = max(0, min(self._searched, self._buf.size - keep))
        cut -= cut % _STEP
        if cut:
            self._buf = self._buf[cut:]
            self._base += cut
            self._searched -= cut
            for df in _OFFSETS:
                self._powers[df] = self._powers[df][:, cut // _STEP:]
        return out

    def flush(self) -> list[Heard]:
        """The end of the audio: a call in progress is read as far as it went."""
        out = self.push(np.zeros(4 * SPB))
        h = self._scan(final=True)
        return out + ([h] if h is not None else [])

    def _scan(self, final: bool = False) -> Heard | None:
        while True:
            before = self._searched
            heard = self._scan_once(final)
            if heard is not None or self._searched == before:
                return heard

    def _scan_once(self, final: bool = False) -> Heard | None:
        x = self._buf
        need = _PHASE_BITS.size * SPB
        if x.size < self._searched + need + SPB:
            return None
        seg_from = self._searched
        ref = 1.0 - 2.0 * _PHASE_BITS                     # + where a 0 is sent
        found = []                                        # per (df, inv)
        first_good = None
        for df in _OFFSETS:
            s = self._scan_soft(seg_from, df)
            n_can = s.size - (_PHASE_BITS.size - 1) * 10
            if n_can <= 0:
                continue
            idx = np.arange(n_can)[:, None] + 10 * np.arange(_PHASE_BITS.size)[None, :]
            for inv in (False, True):
                sg = -s if inv else s
                bits = (sg[idx] < 0).astype(np.int8)
                per_char = (bits == _PHASE_BITS[None, :]).reshape(n_can, 12, 10).all(axis=2)
                n_dx = per_char[:, 0::2].sum(axis=1)
                n_rx = per_char[:, 1::2].sum(axis=1)
                ok = ((n_dx >= 2) & (n_rx >= 1)) | ((n_rx >= 2) & (n_dx >= 1))
                good = np.flatnonzero(ok)
                if good.size:
                    first_good = good[0] if first_good is None else min(first_good, good[0])
                found.append((df, inv, sg, idx, n_dx + n_rx, ok))
        if first_good is None:
            self._searched = max(self._searched, x.size - need - SPB)
            self._searched -= self._searched % _STEP
            return None
        # Every candidate within a pair and a half of the first, every
        # offset, both ways round: the most characters exact, then the
        # surest. A start a pair early still finds its DX characters (all
        # 125), and at a low SNR the odd RX one too.
        best = None
        hi = first_good + 30 * 10
        for df, inv, sg, idx, exact, ok in found:
            for k in np.flatnonzero(ok[first_good:hi]) + first_good:
                key = (int(exact[k]), float(sg[idx[k]] @ ref))
                if best is None or key > best[0]:
                    best = (key, k, df, inv, sg, idx)
        _, i, df, inv, sg, idx = best
        # The bit timing: the soft correlation at its best within a bit.
        lo, top = max(0, i - 10), min(idx.shape[0], i + 10)
        j = lo + int(np.argmax([float(sg[idx[k]] @ ref) for k in range(lo, top)]))
        # The bit rate: each half of the phasing timed on its own. The far
        # end's clock, or a resampling on the way, can be out by a part in a
        # hundred, which is a bit's slip in a second.
        ref = 1.0 - 2.0 * _PHASE_BITS
        def half(h):
            b = np.arange(60 * h, 60 * h + 60)
            ks = range(max(0, j - 15), min(sg.size - 10 * b[-1] - 1, j + 16))
            c = [float(sg[k + 10 * b] @ ref[b]) for k in ks]
            return list(ks)[int(np.argmax(c))] if c else j
        k1, k2 = half(0), half(1)
        spb = SPB * (1.0 + (k2 - k1) * _STEP / (60 * SPB))
        spb = min(max(spb, 0.97 * SPB), 1.03 * SPB)
        start = seg_from + j * _STEP
        # k2 times the second half about its middle (bit 90); the first DX
        # character follows the phasing's 120 bits.
        first = seg_from + k2 * _STEP + 90 * SPB + 30 * spb
        for back in (4, 2, 0):
            candidate = first - back * 10 * spb
            if candidate < 0:
                continue
            soft = self._soft(x[int(candidate):], df)
            indices = np.round(np.arange(10) * spb / _STEP).astype(int)
            if indices[-1] >= soft.size:
                continue
            value, good = C.char_value(((-soft if inv else soft)[indices]) < 0)
            if good and value in (C.INDIVIDUAL, C.NULL):
                first = candidate
                break
        h = self._read(start, first, spb, df, inv, final)
        if h is None:
            return None                                   # wait for more audio
        if h.call is None:
            # Nothing a call makes: a false phasing (noise on the dot pattern
            # can make an RX phasing character exact), or a call lost. Look
            # again from just after it, not after all it read.
            self._searched = start + 10 * SPB
            self._searched -= self._searched % _STEP
            return None
        self._searched = h.end - self._base
        return h

    def _read(self, start: int, first: float, spb: float, df: float, inv: bool,
              final: bool) -> Heard | None:
        """The call after the phasing starting at ``start`` (in ``_buf``).

        The bit timing is followed as it is read: after each character pair,
        a step (a tenth of a bit) earlier or later if the bits are surer
        there. A sound card's clock 100 ppm out is a bit adrift in 10 s."""
        x = self._buf
        f0 = int(first)
        s = self._soft(x[f0:], df)                        # a soft bit every _STEP
        if inv:
            s = -s
        off = [0]                                         # timing, in steps

        def at(p, o=None):                                # character p's soft bits
            o = off[0] if o is None else o
            pos = (first - f0) + (10 * p + np.arange(10)) * spb
            i = np.round(pos / _STEP).astype(int) + o
            if i[0] < 0 or i[-1] >= s.size:
                return None
            return s[i]

        def value(v):
            return C.char_value(v < 0)

        dx: list = []
        ended = False
        for k in range(_MAX_CHARS):
            d, r = at(2 * k), at(2 * k + 5)               # the DX copy and its RX copy
            if d is None or r is None:
                if not final:
                    return None                           # wait for more audio
                if d is None and r is None:
                    break
            cands = [value(c) for c in (d, r) if c is not None]
            ok = [c[0] for c in cands if c[1]]
            v = ok[0] if ok else None
            if v is None and d is not None and r is not None:
                vv, good = value(d + r)
                v = vv if good else None
            dx.append(v)
            if v in C.ENDS and dx[-3:] == [v] * 3:
                ended = True
                break
            if len(dx) >= 3 and dx[-1] is None and dx[-2] in C.ENDS and dx[-3] == dx[-2]:
                dx[-1] = dx[-2]                           # the third lost with the signal
                ended = True
                break
            if len(dx) >= 2 and dx[-1] is None and dx[-2] is None and \
                    self._quiet(f0 + int(20 * spb * (k - 1)), f0 + int(20 * spb * (k + 1))):
                dx = dx[:-2]
                break
            # Follow the timing on this pair.
            pair = lambda o: sum(float(np.sum(np.abs(c))) for c in
                                 (at(2 * k, o), at(2 * k + 1, o)) if c is not None)
            here, early, late = pair(off[0]), pair(off[0] - 1), pair(off[0] + 1)
            if early > here and early >= late:
                off[0] -= 1
            elif late > here:
                off[0] += 1
        n_read = 2 * len(dx) + (5 if ended else 0)
        end = self._base + f0 + int(n_read * 10 * spb) + off[0] * _STEP
        call = C.decode(dx) if dx else None
        return Heard(call, dx, self._base + start, min(end, self._base + x.size), inv, df)

    def _quiet(self, a: int, b: int) -> bool:
        x = self._buf
        seg = x[max(0, a):min(b, x.size)]
        if seg.size < SPB:
            return True
        ref = np.sqrt(np.mean(x[max(0, a - 40 * SPB):max(1, a)] ** 2)) + 1e-12
        return np.sqrt(np.mean(seg ** 2)) < 0.3 * ref


def receive(audio: np.ndarray) -> list[Heard]:
    r = Receiver()
    return r.push(audio) + r.flush()
