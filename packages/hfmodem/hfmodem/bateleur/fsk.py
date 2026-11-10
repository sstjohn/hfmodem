# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Binary FSK: the naval broadcast's (STANAG 4481) and 188-110's (5.1).

**The texts.** STANAG 4481 annex C gives the broadcast's baseband:

- single channel (C.2): a shift of 425 Hz either side of the assigned
  frequency, space the higher, the mark frequency ending in 75 Hz so that
  the carrier sits on a multiple of 100 Hz; equipment puts the pair about
  1700 or about 2000 Hz in the audio, either serving;
- multi-channel (C.3): sub-carriers at 255 + 170n Hz, n from 1 to 16,
  each shifted 42.5 Hz either way; in diversity, channel n is twinned with
  n + 8; channel 3 is the one a ship takes as a single channel;
- PSK (C.4): STANAG 4285 with its annex E, which ``s4285`` is.

MIL-STD-188-110B 5.1 gives the HF characteristic frequencies the same way
(table IV: mark 1575, centre 2000, space 2425 Hz for the 850 Hz shift, and
the same centre for the 170 Hz narrow shift), mark the lower frequency and
binary one (4.2.2, table II), and the rates: 50 Bd, or 75 times a power of
two (4.2.1).

**What neither says**, and what is chosen here. The keying rate of the FSK
broadcast is not in 4481; 75 Bd, the rate its annex D keys its FSK test
interferer at, is the default. Neither text frames bytes: a broadcast
carries a synchronous stream. Here a message is either that (``S``: the
bits as they come, which a receiver joining the stream cannot align to
bytes), or, by default, start-stop characters: a space start bit, eight
data bits, a mark stop bit, as teleprinters frame them, so that the far
end finds the bytes. Before a message the transmitter sends half a second
of mark (idle), and after it two characters' worth. The keying is
phase-continuous.

**The receiver** measures each tone's energy over a bit-long window at
every sample: the soft bit is how the mark's share differs from the
space's. Where the windows lie across a bit's edge the two mix, so the
bit timing is where they mix least; the carrier offset comes from where
the tones stand in a spectrum of the first half second; the signal is on
while the tones stand over places nothing is sent. Twinned channels'
soft bits are added.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .spec import FS

RATES = (50, 75, 150, 300, 600, 1200)
#: 110B table IV, HF radio.
HF_CENTRE_HZ = 2000.0
WIDE_HZ = 850.0
NARROW_HZ = 170.0
#: 4481 C.3: the multi-channel sub-carriers and their shift.
VFT_SHIFT_HZ = 85.0


def vft_centre(n: int) -> float:
    if not 1 <= n <= 16:
        raise ValueError("the multi-channel sub-carriers are numbered 1 to 16")
    return 255.0 + 170.0 * n


def vft_twin(n: int) -> int:
    """C.3: channels 1 and 9, ..., 8 and 16 go together in diversity."""
    return n + 8 if n <= 8 else n - 8


@dataclass(frozen=True)
class FskMode:
    baud: int
    shift: float
    centre: float
    sync: bool = False
    #: Multi-channel: the channel and whether its twin carries it too.
    channel: int = 0
    diversity: bool = False

    def __post_init__(self):
        if self.baud not in RATES:
            raise ValueError(f"FSK runs at {', '.join(map(str, RATES))} Bd")
        if self.baud > self.shift * 1.5 and not self.channel:
            raise ValueError(f"{self.baud} Bd is too fast for a {self.shift:g} Hz shift")

    @property
    def name(self) -> str:
        if self.channel:
            kind = f"V{self.channel}" + ("D" if self.diversity else "")
        else:
            kind = "W" if self.shift == WIDE_HZ else "N"
            if self.centre != HF_CENTRE_HZ:
                kind += f"/{self.centre:g}"
        return f"{self.baud}{kind}" + ("S" if self.sync else "")

    @property
    def n(self) -> int:
        """Samples a bit."""
        return FS // self.baud

    def pairs(self) -> list[tuple[float, float]]:
        """(mark, space) of each pair of tones that carries the bits."""
        centres = [self.centre]
        if self.channel and self.diversity:
            centres.append(vft_centre(vft_twin(self.channel)))
        return [(c - self.shift / 2, c + self.shift / 2) for c in centres]

    def quiet(self) -> list[float]:
        """Places nothing is sent, beside the tones, for the noise."""
        out = []
        for m, s in self.pairs():
            g = max(self.shift / 2, 2.0 * self.baud)
            out += [m - g, s + g]
        return [f for f in out if 150 < f < 3300]


_NAME = re.compile(r"^(\d+)(W|N|V(\d+)(D?))(?:/(\d+(?:\.\d+)?))?(S?)$")


def mode(name: str | FskMode) -> FskMode:
    """"75W" (850 Hz shift about 2000 Hz), "75W/1700" (about 1700 Hz),
    "50N" (170 Hz), "75V3" (multi-channel sub-carrier 3), "75V3D" (with its
    twin), and an ``S`` after any for the bare synchronous stream."""
    if isinstance(name, FskMode):
        return name
    m = _NAME.match(name.strip().upper())
    if m is None:
        raise ValueError(f"no FSK mode {name!r}: e.g. 75W, 75W/1700, 50N, 75V3, 75V3D, 75WS")
    baud, kind, ch, div, centre, sync = m.groups()
    baud = int(baud)
    if ch:
        if centre:
            raise ValueError("a multi-channel sub-carrier's centre is its own")
        n = int(ch)
        return FskMode(baud, VFT_SHIFT_HZ, vft_centre(n), bool(sync), n, bool(div))
    shift = WIDE_HZ if kind == "W" else NARROW_HZ
    c = float(centre) if centre else HF_CENTRE_HZ
    if (c - shift / 2) < 300 or (c + shift / 2) > 3000:
        raise ValueError(f"tones about {c:g} Hz fall outside the radio's audio")
    return FskMode(baud, shift, c, bool(sync))


# --------------------------------------------------------------------------
# The modulator
# --------------------------------------------------------------------------

LEAD_S = 0.5
TAIL_CHARS = 2


def _bits(data: bytes, order: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8),
                         bitorder="little" if order == "lsb" else "big")


def frame(m: FskMode, data: bytes, order: str = "lsb") -> np.ndarray:
    """The bits keyed for a message: the idle mark ahead, the data (framed
    as characters unless synchronous), the idle mark after."""
    lead = np.ones(int(LEAD_S * m.baud), dtype=np.uint8)
    b = _bits(data, order)
    if not m.sync:
        chars = b.reshape(-1, 8)
        b = np.concatenate([np.zeros((chars.shape[0], 1), np.uint8), chars,
                            np.ones((chars.shape[0], 1), np.uint8)], axis=1).ravel()
    tail = np.ones(TAIL_CHARS * 10, dtype=np.uint8)
    return np.concatenate([lead, b, tail])


def bit_count(m: FskMode | str, n_bytes: int) -> int:
    m = mode(m)
    return int(LEAD_S * m.baud) + n_bytes * (8 if m.sync else 10) + TAIL_CHARS * 10


def transmit(data: bytes, m: FskMode | str = "75W", *, scale: float = 0.9,
             order: str = "lsb") -> np.ndarray:
    """A message -> real audio at ``spec.FS``, peak ``scale``; the phase
    runs on unbroken from bit to bit. Mark (1) is the lower tone."""
    m = mode(m)
    bits = frame(m, data, order)
    x = np.zeros(bits.size * m.n)
    for mark, space in m.pairs():
        f = np.repeat(np.where(bits == 1, mark, space), m.n)
        x += np.cos(2 * np.pi * np.cumsum(f) / FS)
    return scale * x / np.max(np.abs(x))


# --------------------------------------------------------------------------
# The receiver
# --------------------------------------------------------------------------

def tone_energy(x: np.ndarray, f: float, n: int, t0: int = 0) -> np.ndarray:
    """|sum of x against f over the n samples from each sample|^2, for every
    start at which n samples are there."""
    t = t0 + np.arange(x.size)
    c = np.concatenate([[0], np.cumsum(x * np.exp(-2j * np.pi * f * t / FS))])
    return np.abs(c[n:] - c[:-n]) ** 2


def offset(x: np.ndarray, m: FskMode) -> float:
    """The carrier offset: where the tones stand in a spectrum of ``x``,
    looked for within a quarter of the shift (and 60 Hz) either way."""
    n = 1 << int(np.ceil(np.log2(max(x.size, 1024))))
    s = np.abs(np.fft.rfft(x * np.hanning(x.size), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    reach = min(60.0, m.shift / 4)
    grid = np.arange(-reach, reach + 0.5, 1.0)
    score = np.zeros(grid.size)
    for mark, space in m.pairs():
        for tone in (mark, space):
            score += np.interp(tone + grid, f, s)
    return float(grid[int(np.argmax(score))])


@dataclass
class Reception:
    mode: FskMode
    data: bytes
    start_sample: int
    end_sample: int
    freq_hz: float
    snr_db: float = float("nan")


def receive(x: np.ndarray, m: FskMode | str = "75W", *, start: int = 0,
            order: str = "lsb") -> Reception | None:
    """A whole recording at ``spec.FS`` -> the first transmission in it at
    or after ``start``, through the streaming receiver."""
    from .stream import Data, Ended, ReceiverFsk, Started
    m = mode(m)
    r = ReceiverFsk(m, bit_order=order)
    x = np.asarray(x, dtype=np.float64)[start:]
    events = []
    for i in range(0, x.size, FS):
        events += r.push(x[i:i + FS])
        if any(isinstance(e, Ended) for e in events):
            break
    else:
        events += r.push(np.zeros(2 * FS)) + r.flush()
    got, data = None, b""
    for e in events:
        if isinstance(e, Started) and got is None:
            got = Reception(m, b"", start + e.at, 0, e.freq_hz)
        elif isinstance(e, Data) and got is not None:
            data += e.data
        elif isinstance(e, Ended) and got is not None:
            got.data, got.snr_db = data, e.snr_db
            got.end_sample = start + r._scanned_to - r._LOOK
            return got
    if got is not None:
        got.data, got.end_sample = data, start + x.size
    return got
