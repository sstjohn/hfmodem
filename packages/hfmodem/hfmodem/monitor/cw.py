# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Morse on one narrow track, read into text.

Gateways identify in CW, and it is the loudest thing they send. The reader is
given a track's centre and nothing else, so everything it knows about the
keying it learns from the envelope: where key-down sits against the floor, how
long a dit is, and whether the emission is narrow enough to be a keyed carrier.
"""
from __future__ import annotations

import numpy as np

from hfmodem.core.cwid import MORSE
from hfmodem.monitor import CALLSIGN

HOP_S = 0.010
BAND_HZ = 40.0
#: Where the neighbour bands sit. A keyed carrier's own sidebands are more than
#: 20 dB down here at 40 WPM; a keyed FSK pair or a data burst fills them.
SIDE_HZ = 120.0
MIN_NARROW_DB = 10.0

HYSTERESIS_DB = 3.0
#: A key state held for one hop is a noise spike or a fade's flicker.
MIN_RUN_HOPS = 2
STATS_WINDOW_S = 15.0
STATS_EVERY_S = 0.5
#: Key decisions trail the audio by this much, so the first element of a
#: sending is judged against a ceiling that already includes it.
LOOKAHEAD_S = 0.5
#: A hop this far over the floor counts toward the ceiling. Noise in a 40 Hz
#: band at 10 ms reaches it about once in ten thousand hops.
SIGNAL_OVER_FLOOR_DB = 15.0
MIN_CEILING_HOPS = 5
EDGE_HOPS = 3

MIN_WPM, MAX_WPM = 8.0, 40.0
#: Longer than a dah at the slowest accepted speed, with margin: not Morse.
MAX_ON_S = 3 * 1.2 / MIN_WPM * 1.4
#: A silence this long ends a sending, and the timing learned from it.
END_OF_SENDING_S = 3.0
HISTORY_RUNS = 80

CONFIRM_CHARS = 4
RATIO_RANGE = (2.5, 3.5)
MAX_CV = 0.3
#: How far a dit's keyed width (``CwReader._width``) may stray from the word's
#: median dit. Measured: at most 0.09 over 240 rendered callsigns at 12 to 30 WPM
#: under 15 dB fades, 0.08 on the WS8EOC ID, 0.41 on the clipped head of an ID
#: in rf-corpus pos_vara_session that reads D5YTA.
MAX_DIT_SPREAD = 0.25

DECODE = {code: ch for ch, code in MORSE.items()} | {
    ".-.-.-": ".", "--..--": ",", "..--..": "?", "-...-": "=", ".-.-.": "+",
    "...-.-": "<SK>", "-.--.": "<KN>",
}


class CwReader:
    """Push audio as it arrives; each finished word comes back as a record.

    A run is ``(key down, first hop, end hop, band powers)``, the powers summed
    over the run's hops in the centre band and the two neighbours, or None for
    key-up.
    """

    def __init__(self, fs: int, centre_hz: float) -> None:
        self.fs = fs
        self.centre_hz = float(centre_hz)
        self.hop = int(round(HOP_S * fs))
        self.box = int(round(fs / BAND_HZ))
        self._freqs = self.centre_hz + np.array([0.0, -SIDE_HZ, SIDE_HZ])
        self._zprev = np.zeros((3, self.box), complex)
        self._n = 0
        self._tail = np.zeros(0)
        self._env = np.zeros((0, 3))
        self._base = 0
        self._decided = 0
        self._stats_at = 0
        self._mid: float | None = None
        self._down = False
        self._run_start = 0
        self._acc = np.zeros(3)
        self._runs: list[tuple] = []
        self._history: list[tuple] = []

    def push(self, samples) -> list[dict]:
        x = np.concatenate([self._tail, np.asarray(samples, float).ravel()])
        whole = len(x) // self.hop * self.hop
        self._tail = x[whole:]
        if whole:
            self._env = np.concatenate([self._env, self._envelope(x[:whole])])
        return self._advance(final=False)

    def flush(self) -> list[dict]:
        return self._advance(final=True)

    def _envelope(self, x: np.ndarray) -> np.ndarray:
        """Band power at the end of each hop, through a boxcar of 40 Hz noise bandwidth.

        A boxcar settles in exactly its own length, which is what lets a 40 ms
        element gap at 30 WPM reach the floor; a recursive filter of the same
        bandwidth rings across it.
        """
        n = self._n + np.arange(len(x))
        self._n += len(x)
        z = x * np.exp(-2j * np.pi * np.outer(self._freqs, n) / self.fs)
        z = np.concatenate([self._zprev, z], axis=1)
        self._zprev = z[:, -self.box:]
        c = np.cumsum(z, axis=1)
        ends = np.arange(self.box + self.hop - 1, z.shape[1], self.hop)
        y = (c[:, ends] - c[:, ends - self.box]) / self.box
        return 2 * np.abs(y.T) ** 2 + 1e-20

    def _advance(self, final: bool) -> list[dict]:
        hops = self._base + len(self._env)
        look, window = int(LOOKAHEAD_S / HOP_S), int(STATS_WINDOW_S / HOP_S)
        upto = hops if final else hops - look
        for k in range(self._decided, upto):
            if k >= self._stats_at:
                self._mid = self._midpoint(k + look - window, k + look)
                self._stats_at = k + int(STATS_EVERY_S / HOP_S)
            self._key(k, self._env[k - self._base])
        self._decided = max(self._decided, upto)
        if final:
            self._close_run(self._decided)
        out = self._emit(final)
        pending = min((a for _, a, _, _ in self._runs), default=self._run_start)
        drop = min(self._decided + look - window, pending - EDGE_HOPS) - self._base
        if drop > 0:
            self._env = self._env[drop:]
            self._base += drop
        return out

    def _midpoint(self, lo: int, hi: int) -> float | None:
        """Halfway, in dB, between the two-medians floor and ceiling."""
        c = 10 * np.log10(self._env[max(lo - self._base, 0):hi - self._base, 0])
        if not len(c):
            return None
        floor = float(np.median(c[c <= np.median(c)]))
        up = c[c > floor + SIGNAL_OVER_FLOOR_DB]
        if len(up) < MIN_CEILING_HOPS:
            return None
        return (floor + float(np.median(up))) / 2

    def _key(self, k: int, power: np.ndarray) -> None:
        db = 10 * np.log10(power[0])
        if self._mid is None:
            flip = self._down
        elif self._down:
            flip = db < self._mid - HYSTERESIS_DB / 2
        else:
            flip = db > self._mid + HYSTERESIS_DB / 2
        if flip:
            self._close_run(k)
        if self._down:
            self._acc = self._acc + power

    def _close_run(self, k: int) -> None:
        if k - self._run_start < MIN_RUN_HOPS and self._runs:
            self._down, self._run_start, _, acc = self._runs.pop()
            self._acc = acc if self._down else np.zeros(3)
            return
        if k > self._run_start:
            self._runs.append((self._down, self._run_start, k, self._acc if self._down else None))
        self._down = not self._down
        self._run_start = k
        self._acc = np.zeros(3)

    def _emit(self, final: bool) -> list[dict]:
        out = []
        while (end := self._word_end(final)) is not None:
            cut, resume, over = end
            word, self._runs = self._runs[:cut], self._runs[resume:]
            rec = self._read(word)
            if rec:
                out.append(rec)
            if over or not rec:
                self._history = []
            elif rec["grade"] == "confirmed":
                self._history = (self._history + word)[-HISTORY_RUNS:]
        return out

    def _word_end(self, final: bool) -> tuple[int, int, bool] | None:
        """``(runs in the word, runs consumed, sending over)`` once a word is finished."""
        if not self._runs:
            return None
        unit, shift = _timing(self._history + self._runs)[:2]

        def word_gap(secs: float) -> bool:
            return secs > END_OF_SENDING_S or 0 < unit and secs + shift > 5 * unit

        for i, (down, a, b, _) in enumerate(self._runs):
            secs = (b - a) * HOP_S
            if down and secs > MAX_ON_S:
                return i, i + 1, True
            if not down and i and word_gap(secs):
                return i, i, secs > END_OF_SENDING_S
        open_s = (self._decided - self._run_start) * HOP_S
        if final or not self._down and word_gap(open_s):
            return len(self._runs), len(self._runs), final or open_s > END_OF_SENDING_S
        return None

    def _read(self, word: list[tuple]) -> dict | None:
        runs = list(word)
        while runs and not runs[0][0]:
            runs.pop(0)
        if not runs:
            return None
        unit, shift, ratio, dits, dahs = _timing(self._history + runs)
        if not unit or not MIN_WPM <= 1.2 / unit <= MAX_WPM:
            return None
        bands = 10 * np.log10(sum(p for d, _, _, p in runs if d)
                              / sum(b - a for d, a, b, _ in runs if d))
        if bands[0] - max(bands[1:]) < MIN_NARROW_DB:
            return None
        chars, code = [], ""
        for down, a, b, _ in runs:
            secs = (b - a) * HOP_S
            if down:
                code += "-" if secs - shift > 2 * unit else "."
            elif secs + shift > 2 * unit:
                chars.append(DECODE.get(code, "?"))
                code = ""
        text = "".join(chars + [DECODE.get(code, "?")] * bool(code))
        stable = all(len(g) < 2 or np.std(g) / np.mean(g) <= MAX_CV for g in (dits, dahs))
        widths = np.array([self._width(a, b) for d, a, b, _ in runs
                           if d and (b - a) * HOP_S - shift <= 2 * unit])
        alike = len(widths) < 2 or np.max(np.abs(widths / np.median(widths) - 1)) <= MAX_DIT_SPREAD
        confirmed = (len(text) >= CONFIRM_CHARS and "?" not in text and stable and alike
                     and RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1])
        rec = {
            "t": runs[0][1] * HOP_S, "modem": "cw", "protocol": "CW", "kind": "text",
            "text": text, "role": "sender",
            "grade": "confirmed" if confirmed else "tentative",
            "centre_hz": self.centre_hz, "level_dbfs": round(float(bands[0]), 1),
            "detail": f"{1.2 / unit:.0f} WPM",
        }
        if CALLSIGN.fullmatch(text):
            rec["station"] = text
        return rec

    def _width(self, a: int, b: int) -> float:
        """A key-down's width in hops, each edge taken at half the level just inside it.

        The threshold is set by the whole window, so a fade moves where it cuts
        each element. Half of the level the boxcar settles to on either side of
        an edge does not move, and through the boxcar it marks the keyed instant
        exactly; dits that disagree on it were not all keyed as dits.
        """
        lo = a - EDGE_HOPS - self._base
        seg = np.sqrt(self._env[max(lo, 0):max(b + EDGE_HOPS - self._base, 0), 0])
        if lo < 0 or len(seg) < 2 * EDGE_HOPS + 1:
            return np.nan
        inside = seg[EDGE_HOPS:-EDGE_HOPS]
        up = EDGE_HOPS + int(np.argmax(inside[:EDGE_HOPS]))
        dn = len(seg) - EDGE_HOPS - 1 - int(np.argmax(inside[::-1][:EDGE_HOPS]))
        return _crossing(seg[dn:], seg[dn] / 2) + dn - (up - _crossing(seg[up::-1], seg[up] / 2))


def _timing(runs: list[tuple]) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """``(unit, shift, dah/dit ratio, dit runs, dah runs)``, in seconds.

    The threshold sits down the envelope's skirts, so every key-down reads long
    by some amount and every key-up short by the same. The mean dit and the mean
    element gap average it out of the unit and give the shift; a fade moves the
    shift and leaves the unit alone.
    """
    on = np.array([(b - a) * HOP_S for d, a, b, _ in runs if d])
    off = np.array([(b - a) * HOP_S for d, a, b, _ in runs[1:] if not d])
    if not len(on):
        return 0.0, 0.0, 0.0, on, on
    split = _split(on)
    dits, dahs = (on[on < split], on[on >= split]) if split else (on, on[:0])
    d = float(np.mean(dits))
    guess = (float(np.mean(dahs)) - d) / 2 if len(dahs) else d
    gaps = off[off < 2 * guess]
    g = float(np.mean(gaps)) if len(gaps) else d
    unit, shift = (d + g) / 2, (d - g) / 2
    ratio = (float(np.mean(dahs)) - shift) / unit if len(dahs) else 0.0
    return unit, shift, ratio, dits, dahs


def _crossing(amp: np.ndarray, level: float) -> float:
    """Hops from ``amp[0]`` to where it first falls through ``level``, interpolated."""
    below = amp < level
    if not below.any():
        return float(len(amp) - 1)
    k = int(np.argmax(below))
    return k - 1 + (amp[k - 1] - level) / (amp[k - 1] - amp[k]) if k else 0.0


def _split(on: np.ndarray) -> float | None:
    """The duration dividing dits from dahs, or None when the runs are all one kind."""
    s = np.sort(on)
    if len(s) < 2 or s[-1] < 2 * s[0]:
        return None
    ls = np.log(s)
    cost = [np.var(ls[:i]) * i + np.var(ls[i:]) * (len(s) - i) for i in range(1, len(s))]
    i = int(np.argmin(cost)) + 1
    return float(s[i - 1] + s[i]) / 2
