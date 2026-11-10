# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Baudot RTTY: 45.45 Bd, 170 Hz shift, ITA2 over async framing.

Start bit space, units 1..5 with unit 1 first and 1 = mark, 1.5 stop units of
mark; mark is the higher tone (USB audio convention).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from hfmodem.lanner import fsk, ita2

BAUD = 45.45
SHIFT = 170.0
MARK_HZ = 1585.0
SPACE_HZ = 1415.0

LEAD_TAIL_UNITS = 10.0
#: The width of a tone change, as a fraction of a unit, and the key-up and
#: key-down envelope ramp.
SHAPE_UNITS = 0.2
RAMP_S = 0.005

_LTRS = 0b11111
_FIGS = 0b11011
#: A decided unit clears this soft value, about a twentieth of a unit from a
#: tone change. With 1.5 stop units the next character's units sit half a unit
#: off this one's, so a frame hunted from a data edge samples that character
#: exactly on its tone changes, where only the sign of a near-zero value would
#: accept the frame or name its code. Every frame needs a decided stop; a shift,
#: which recases all that follows, needs every unit decided.
_DECISIVE = 0.2


class Rate(NamedTuple):
    baud: float
    shift: float
    stop: float = 1.5

    @classmethod
    def parse(cls, text: str) -> "Rate":
        """``BAUD/SHIFT[/STOP]``, as in ``45.45/170`` or ``50/850/1``."""
        try:
            fields = [float(field) for field in text.split("/")]
            rate = cls(*fields)
        except (TypeError, ValueError):
            raise ValueError(f"{text!r} is not BAUD/SHIFT[/STOP], e.g. 45.45/170") from None
        if len(fields) < 2 or min(rate) <= 0:
            raise ValueError(f"{text!r} is not BAUD/SHIFT[/STOP], e.g. 45.45/170")
        return rate

    def __str__(self) -> str:
        return f"{self.baud:g}/{self.shift:g}" + ("" if self.stop == 1.5 else f"/{self.stop:g}")


AMATEUR = Rate(BAUD, SHIFT)


class Transmitter:
    """Keyed RTTY rendered as it is asked for, with the text pulled as it goes out.

    Unit boundaries sit at round(units * fs / baud) on a cumulative count, so a
    fractional rate never drifts, and the phase carries across every `render`.
    The instantaneous frequency moves between the tones on a raised cosine
    `SHAPE_UNITS` wide, and the envelope ramps over `RAMP_S` at key-up and at
    the end of the tail.

    Rendering opens with `LEAD_TAIL_UNITS` of mark. At each character boundary
    with nothing queued, ``pull()`` is asked for the next text; an empty answer
    sends LTRS (diddle), after which the encoder opens with LTRS again. Once
    `finish()` has been called an empty answer sends the mark tail instead, and
    `end` becomes the sample index where the emission stops; past it `render`
    returns silence.
    """

    def __init__(self, rate: Rate, mark_hz: float, space_hz: float, *, pull=lambda: "",
                 variant: str = "us", fs: int = fsk.FS, amplitude: float = 0.5):
        self.rate = rate
        self.amplitude = amplitude
        self.end: int | None = None
        self._tones = (space_hz, mark_hz)
        self._pull = pull
        self._encoder = ita2.Encoder(variant)
        self._codes: list[int] = []
        self._fs = fs
        self._half = max(1, round(SHAPE_UNITS / 2 * fs / rate.baud))
        self._kernel = np.hanning(2 * self._half + 3)[1:-1]
        self._kernel /= self._kernel.sum()
        self._ramp = max(1, round(RAMP_S * fs))
        self._finishing = False
        self._units = 0.0
        self._scheduled = 0
        self._origin = -self._half
        self._freq = np.full(self._half, float(mark_hz))
        self._pos = 0
        self._phase = 0.0
        self._schedule(LEAD_TAIL_UNITS, True)

    @property
    def position(self) -> int:
        return self._pos

    @property
    def scheduled(self) -> int:
        """The sample where what has been pulled so far ends."""
        return self._scheduled

    def finish(self) -> None:
        self._finishing = True

    def _schedule(self, units: float, mark: bool) -> None:
        self._units += units
        end = round(self._units * self._fs / self.rate.baud)
        self._freq = np.concatenate((self._freq, np.full(end - self._scheduled, self._tones[mark])))
        self._scheduled = end

    def _character(self, code: int) -> None:
        self._schedule(1.0, False)
        for k in range(5):
            self._schedule(1.0, bool(code >> k & 1))
        self._schedule(self.rate.stop, True)

    def _extend(self) -> None:
        if self.end is not None:
            self._schedule(LEAD_TAIL_UNITS, True)
            return
        if not self._codes:
            self._codes = self._encoder.feed(self._pull())[0]
        if self._codes:
            self._character(self._codes.pop(0))
        elif self._finishing:
            self._schedule(LEAD_TAIL_UNITS, True)
            self.end = self._scheduled
        else:
            self._character(_LTRS)
            self._encoder.reset()

    def render(self, n: int) -> np.ndarray:
        while self._scheduled < self._pos + n + self._half:
            self._extend()
        lo = self._pos - self._origin
        shaped = np.convolve(self._freq[lo - self._half:lo + n + self._half], self._kernel, "valid")
        phase = self._phase + 2 * np.pi * np.cumsum(shaped) / self._fs
        if n:
            self._phase = float(phase[-1] % (2 * np.pi))
        index = self._pos + np.arange(n)
        envelope = np.clip(index / self._ramp, 0, 1)
        if self.end is not None:
            envelope = np.minimum(envelope, np.clip((self.end - index) / self._ramp, 0, 1))
        envelope = 0.5 - 0.5 * np.cos(np.pi * envelope)
        self._pos += n
        keep = self._pos - self._half - self._origin
        self._freq = self._freq[keep:]
        self._origin += keep
        return (self.amplitude * envelope * np.sin(phase)).astype(np.float32)


def modulate(text, *, baud=BAUD, mark_hz=MARK_HZ, space_hz=SPACE_HZ,
             stop_units=1.5, variant="us", fs=fsk.FS):
    pending = [text]
    tx = Transmitter(Rate(baud, abs(mark_hz - space_hz), stop_units), mark_hz, space_hz,
                     pull=lambda: pending.pop() if pending else "", variant=variant, fs=fs)
    tx.finish()
    chunks = []
    while tx.end is None or tx.position < tx.end:
        chunks.append(tx.render(int(fs // 4)))
    return np.concatenate(chunks)[:tx.end]


def _frames(soft, unit, start, limit, *, confidence=_DECISIVE):
    """Return accepted frames, rejected frames, and the next hunting sample.

    Zero confidence retains sign-only stop framing for the carrier classifier,
    which counts frames without interpreting their characters or shifts.
    """
    out = []
    rejected = []
    i = max(1, start)
    # Locate edges in NumPy so long stretches of idle mark remain cheap.
    edges = np.flatnonzero((soft[i - 1:limit - 1] > 0) &
                           (soft[i:limit] <= 0)) + i if i < limit else ()
    for edge in edges:
        if edge < i:
            continue
        edge = int(edge)
        units = soft[edge + np.round((np.arange(7) + 0.5) * unit).astype(int)]
        if units[0] >= 0:
            continue
        code = sum(int(units[k] > 0) << (k - 1) for k in range(1, 6))
        stop = edge + int(round(6.5 * unit))
        decided = np.abs(units) > confidence
        if units[6] > confidence and (confidence == 0 or decided.all()
                                     or code not in (_LTRS, _FIGS)):
            out.append((edge, code))
        else:
            rejected.append((edge, code))
        i = stop
    return out, rejected, max(i, limit)


def characters(audio, *, baud=BAUD, mark_hz=MARK_HZ, space_hz=SPACE_HZ,
               fs=fsk.FS, start=0) -> list[tuple[int, int]]:
    """Return (start-bit edge sample, ITA2 code), hunting at ``start``.

    Positions are relative to this buffer. A frame is accepted when its stop
    unit is mark; shift codes are returned without interpreting them.
    """
    audio = np.asarray(audio)
    unit = fs / baud
    limit = audio.size - int(round(6.5 * unit))
    if max(1, start) >= limit:
        return []
    soft = fsk.discriminate(audio, f_hi=mark_hz, f_lo=space_hz, baud=baud, fs=fs)
    return _frames(soft, unit, start, limit)[0]


def demodulate(audio, *, baud=BAUD, mark_hz=MARK_HZ, space_hz=SPACE_HZ,
              variant="us", fs=fsk.FS, unshift_on_space=True):
    """Decode finished audio with the selected ITA2 receiving space policy."""
    decoder = ita2.Decoder(variant, unshift_on_space=unshift_on_space)
    return "".join(decoder.feed(code) for _, code in
                   characters(audio, baud=baud, mark_hz=mark_hz,
                              space_hz=space_hz, fs=fs))


class FrameEvent(NamedTuple):
    """A committed frame: absolute start-bit sample, ITA2 code, text, acceptance."""

    sample: int
    code: int
    text: str
    accepted: bool
    shift_known: bool = True


class Stream:
    """Decode contiguous audio blocks with an internally overlapping tail.

    ``feed`` commits only frames whose discriminator windows are complete.
    ``flush`` also uses the held buffer's endpoint decisions, as ``demodulate``
    does for a finished recording. Neither method resets the ITA2 shift state.
    ``events`` lists FrameEvents committed by the latest feed/flush; rejected text is empty.
    ``accepted`` and ``rejected`` count all committed frames since construction.
    ``shift_known`` becomes true on an accepted LTRS or FIGS, and false after
    erasures or rejected frames that could hide a shift. Text follows the decoder's current case, including
    the selected ``unshift_on_space`` policy (default on), and remains provisional
    until another accepted shift establishes certainty; each event records certainty.
    """

    def __init__(self, *, baud=BAUD, mark_hz=MARK_HZ, space_hz=SPACE_HZ,
                 variant="us", fs=fsk.FS, unshift_on_space=True):
        self._baud = baud
        self._mark_hz = mark_hz
        self._space_hz = space_hz
        self._fs = fs
        self._unit = fs / baud
        window = max(1, int(round(self._unit)))
        self._future = window - 1 - (window - 1) // 2
        self._tail = int(np.ceil(7.5 * self._unit)) + window + 2
        self._buffer = np.empty(0, dtype=np.float64)
        self._valid = np.empty(0, dtype=bool)
        self._last_erased = -float("inf")
        self._origin = 0
        self._next = 1  # Absolute hunting position, including rejected frames.
        self._decoder = ita2.Decoder(variant, unshift_on_space=unshift_on_space)
        self.shift_known = False
        self.events: list[FrameEvent] = []
        self.accepted = self.rejected = 0

    @property
    def unshift_on_space(self) -> bool:
        return self._decoder.unshift_on_space

    @unshift_on_space.setter
    def unshift_on_space(self, value: bool) -> None:
        self._decoder.unshift_on_space = value

    def retune(self, mark_hz, space_hz) -> None:
        """Change the discriminator tones, preserving held audio and decoder state."""
        self._mark_hz = mark_hz
        self._space_hz = space_hz

    def feed(self, block, *, valid=None) -> str:
        """Append contiguous samples; optional validity mask marks erased audio.

        Frames overlapping erasures or starting within one frame after them
        are rejected without changing ITA2 shift; this excludes partial frames
        first heard when the carrier returns. Erasures revoke certainty until
        the next accepted LTRS or FIGS, preserving the last interpreted case.
        The clock and framing search still advance over every input sample.
        """
        self._buffer = np.concatenate((self._buffer, np.asarray(block)))
        self._valid = np.concatenate((self._valid, np.ones(len(block), dtype=bool)
                                      if valid is None else np.asarray(valid, dtype=bool)))
        return self._decode(final=False)

    def flush(self) -> str:
        """Decode the remaining held audio as the end of a recording."""
        return self._decode(final=True)

    def _decode(self, *, final):
        self.events = []
        limit = self._buffer.size - int(round(6.5 * self._unit))
        if not final:
            limit -= self._future
        start = max(1, self._next - self._origin)
        out = []
        if start < limit:
            soft = fsk.discriminate(self._buffer, f_hi=self._mark_hz,
                                    f_lo=self._space_hz, baud=self._baud, fs=self._fs)
            frames, rejected, next_sample = _frames(soft, self._unit, start, limit)
            reliable = []
            for edge, code in frames:
                before = edge - round(7.5 * self._unit)
                after = edge + round(6.5 * self._unit)
                erased = (self._origin + before <= self._last_erased
                          or not np.all(self._valid[max(0, before):after]))
                (rejected if erased else reliable).append((edge, code))
            frames = reliable
            self._next = self._origin + next_sample
            self.accepted += len(frames)
            self.rejected += len(rejected)
            checked = start
            for edge, code, accepted in sorted(
                    [(edge, code, True) for edge, code in frames]
                    + [(edge, code, False) for edge, code in rejected]):
                stop = edge + round(6.5 * self._unit)
                if not accepted or not np.all(self._valid[checked:stop]):
                    self.shift_known = False
                checked = stop
                if accepted and code in (0b11111, 0b11011):  # ITA2 LTRS / FIGS
                    self.shift_known = True
                text = self._decoder.feed(code) if accepted else ""
                self.events.append(FrameEvent(self._origin + edge, code, text,
                                              accepted, self.shift_known))
            # A fade can hide every edge, leaving no rejected frame to report.
            if not np.all(self._valid[checked:next_sample]):
                self.shift_known = False
            out = [event.text for event in self.events]
        drop = max(0, self._buffer.size - self._tail)
        if drop:
            # Copy to release the allocation containing the preceding block.
            erased = np.flatnonzero(~self._valid[:drop])
            if len(erased):
                self._last_erased = self._origin + int(erased[-1])
            self._buffer = self._buffer[drop:].copy()
            self._valid = self._valid[drop:].copy()
            self._origin += drop
        return "".join(out)
