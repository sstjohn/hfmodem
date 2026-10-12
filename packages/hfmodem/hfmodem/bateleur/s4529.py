# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 4529: STANAG 4285 at half speed, in 1240 Hz, for maritime links.

**The text.** Written from the Joint Interoperability Test Command's STANAG
4529 Conformance Test Procedures (April 2004), which restate the STANAG's
requirements one by one (its paragraph 1.2 and appendix B, reference
numbers 1 to 17). No implementation was read. What they give:

- 8-PSK at 1200 Bd on a sub-carrier selectable from 800 to 2400 Hz in
  100 Hz steps, 1700 Hz by default; 99% of the power within 1240 Hz;
- the frame of 256 symbols, 213.3 ms: 80 symbols of 2-PSK sync from
  x^5 + x^2 + 1 loaded with 11010, then three blocks of 32 data and 16
  reference symbols and a last of 32 data, all 176 scrambled by an 8-phase
  sequence from x^9 + x^4 + 1 loaded with ones;
- one, two or three bits to a symbol at channel rates of 600, 1200 and
  1800 bit/s, linked to symbols by its tables 1.2-1.4;
- coding, interleaving, interleaver synchronization and the message
  protocol as STANAG 4285 annex E gives them for the next higher data rate:
  1200 bit/s as 4285's 2400, 600 as 1200, 300, 150 and 75 as 600, 300 and
  150; the interleaver of 32 rows, each delayed 48, 24 or 12 more (long)
  or 4, 2 or 1 (short), and the flush given for each;
- uncoded channel rates of 600, 1200 and 1800 bit/s;
- an offset of up to 37.5 Hz and 3.5 Hz/s of drift to be tolerated.

Every one of those is ``s4285``'s, symbol for symbol: the frame, the sync
and scrambling sequences (the 80-bit sync pattern the procedures print is
4285's), the bit-to-symbol tables, and the coding, interleaving and flush
of the 4285 mode of twice the rate (the procedures' flush counts are what
4285's give). So a 4529 transmission here is a 4285 transmission played at
half speed and moved to the 4529 sub-carrier, and a 4529 receiver is the
4285 receiver behind the reverse: ``stretch`` and ``Squeeze``.

**Modes** are named by the user rate and interleaver as for 4285 (``600L``,
``75S``), uncoded by the channel rate with ``U`` (``1800U``), and a
sub-carrier other than 1700 Hz after a slash (``600L/2000``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, hilbert, lfilter, lfilter_zi, resample_poly

from . import s4285
from .spec import CARRIER_HZ, FS, MODES4285

DEFAULT_CARRIER_HZ = 1700.0
CARRIERS_HZ = tuple(float(f) for f in range(800, 2401, 100))
BAUD = 1200

#: 4529 mode (user rate and interleaver) -> the 4285 mode it is at half speed.
_AS_4285 = {
    "1200L": "2400L", "1200S": "2400S", "600L": "1200L", "600S": "1200S",
    "300L": "600L", "300S": "600S", "150L": "300L", "150S": "300S",
    "75L": "150L", "75S": "150S",
    "1800U": "3600U", "1200U": "2400U", "600U": "1200U",
}


@dataclass(frozen=True)
class Mode4529:
    name: str                       # "600L", "1800U"
    carrier: float = DEFAULT_CARRIER_HZ

    @property
    def as_4285(self) -> str:
        return _AS_4285[self.name]

    @property
    def full_name(self) -> str:
        return self.name if self.carrier == DEFAULT_CARRIER_HZ else \
            f"{self.name}/{self.carrier:g}"

    @property
    def coded(self) -> bool:
        return MODES4285[self.as_4285].coded


MODES4529 = tuple(_AS_4285)

_NAME = re.compile(r"^(\d+[LSU])(?:/(\d+))?$")


def mode(name: str | Mode4529) -> Mode4529:
    if isinstance(name, Mode4529):
        return name
    m = _NAME.match(name.strip().upper())
    if m is None or m.group(1) not in _AS_4285:
        raise ValueError(f"no 4529 mode {name!r}; one of {', '.join(MODES4529)}, "
                         f"with /<sub-carrier Hz> after it if not 1700")
    carrier = float(m.group(2)) if m.group(2) else DEFAULT_CARRIER_HZ
    if carrier not in CARRIERS_HZ:
        raise ValueError("the sub-carrier is 800 to 2400 Hz in 100 Hz steps")
    return Mode4529(m.group(1), carrier)


# --------------------------------------------------------------------------
# Half speed and back
# --------------------------------------------------------------------------

def stretch(x4285: np.ndarray, carrier: float) -> np.ndarray:
    """4285 audio (2400 Bd about 1800 Hz, at FS) -> the same symbols at
    1200 Bd about ``carrier``, at FS."""
    n = np.arange(x4285.size)
    bb = hilbert(x4285) * np.exp(-2j * np.pi * CARRIER_HZ * n / FS)
    bb = resample_poly(bb, 2, 1)                     # each sample now lasts twice as long
    m = np.arange(bb.size)
    return np.real(bb * np.exp(2j * np.pi * carrier * m / FS))


class Squeeze:
    """The reverse, a block at a time: audio at FS holding 1200 Bd about
    ``carrier`` -> what 4285's receiver expects, 2400 Bd about 1800 Hz at
    FS. Mixed down, low-passed (which also takes away the mix's image) and
    every other sample kept, then mixed up again: every filter's state and
    every oscillator's phase carried from block to block."""

    def __init__(self, carrier: float) -> None:
        self.carrier = carrier
        self._h = firwin(129, 760.0, fs=FS)
        self._zi = lfilter_zi(self._h, 1.0).astype(complex) * 0.0
        self._n = 0                                  # input samples taken
        self._m = 0                                  # output samples made

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = self._n + np.arange(x.size)
        bb = 2 * x * np.exp(-2j * np.pi * self.carrier * n / FS)
        bb, self._zi = lfilter(self._h, 1.0, bb, zi=self._zi)
        keep = bb[(-self._n) % 2::2]                 # even input samples only
        self._n += x.size
        m = self._m + np.arange(keep.size)
        self._m += keep.size
        return np.real(keep * np.exp(2j * np.pi * CARRIER_HZ * m / FS))


def squeeze(x: np.ndarray, carrier: float) -> np.ndarray:
    return Squeeze(carrier).push(x)


# --------------------------------------------------------------------------
# Sending and receiving
# --------------------------------------------------------------------------

def transmit(data: bytes, m: Mode4529 | str = "600L", *, order: str = "lsb",
             scale: float = 1.0) -> np.ndarray:
    m = mode(m)
    x = stretch(s4285.transmit(data, m.as_4285, order=order, trim=False), m.carrier)
    return scale * x / np.max(np.abs(x))


def seconds(m: Mode4529 | str, n_bytes: int) -> float:
    """Twice 4285's, by construction."""
    m = mode(m)
    return 2 * s4285.symbol_count(MODES4285[m.as_4285], n_bytes) / 2400 + 0.1


@dataclass
class Reception:
    mode: Mode4529
    data: bytes
    start_sample: int
    end_sample: int
    freq_hz: float
    frames: int
    snr_db: float


def receive(audio: np.ndarray, m: Mode4529 | str = "600L", *, start: int = 0,
            order: str = "lsb") -> Reception | None:
    """The first 4529 transmission in ``audio`` at or after ``start``."""
    m = mode(m)
    y = squeeze(np.asarray(audio, dtype=np.float64)[start - start % 2:], m.carrier)
    r = s4285.receive(y, m.as_4285, order=order)
    if r is None:
        return None
    base = start - start % 2
    return Reception(m, r.data, base + 2 * r.start_sample, base + 2 * r.end_sample,
                     r.freq_hz / 2, r.frames, r.snr_db)
