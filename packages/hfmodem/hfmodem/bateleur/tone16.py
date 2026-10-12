# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 16-tone DPSK mode (MIL-STD-188-110B appendix A).

Sixteen data tones 110 Hz apart, 935 to 2585 Hz, each keyed in differential
phase once a 13 1/3 ms signal element (75 baud); a 17th, unmodulated, at
605 Hz for the Doppler correction, 7 dB over a data tone; nothing at 825 Hz,
the synchronization slot. 75 to 2400 bit/s: four-phase at 2400 and 1200,
two-phase below, the same bits on 2, 4, 8 or 16 tones below 2400 (table
A-I). No coding, no interleaving, no end-of-message: the bits go out as
they come and the message ends when the signal does.

**The text.** Written from appendix A. Where the text leaves a choice open:

- each tone's phase is reckoned from the onset of its own signal element,
  as the published recordings showed appendix B's to be (``tone39``); no
  recording of this mode has been found to confirm it here;
- of a dibit (table A-II), the earlier bit is the odd bit location, as in
  appendix B;
- the element that follows the preamble (A.5.4) sets each tone's phase
  reference; the appendix gives no phases for it, and any serve, since the
  data is in the changes. Newman phases spread the peaks across the element;
- the preamble runs 12 elements (160 ms), inside A.5.4's 5 to 32;
- the Doppler tone runs on unbroken, preamble to end.

**Bytes.** The appendix carries a bit stream. Bytes are sent least
significant bit first by default, and the last element is filled out with
zero bits, which arrive as NULs.

**The receiver** finds the preamble's two tones, the Doppler tone giving
the carrier offset and the 1705 Hz tone's reversals the element timing.
Each element is read over 87 samples (about 1/110 s) centred in it. Timing
tracks leakage into the empty places; the Doppler tone's own change tracks
the offset through the message, and the copies of a bit under diversity are added before it is
decided. The message ends where the data tones sink into what the empty
synchronization slot holds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import FS

T = 128                                     # samples a 13 1/3 ms element
W = 87                                      # the window read: ~1/110 s
GUARD = (T - W) // 2                        # where in an element it starts
DOPPLER_HZ = 605.0
SYNC_HZ = 1705.0                            # data tone 8, reversing in the preamble
SLOT_HZ = 825.0                             # the synchronization slot: empty
TONE_HZ = 935.0 + 110.0 * np.arange(16)
#: Where nothing is sent: the slot, and the places a tone would have below
#: the Doppler tone and above the last. Orthogonal to every tone over a
#: window inside an element, so what is there is noise; a window across
#: an element's edge spills the tones' energy into them, which is what
#: the receiver's timing looks for.
QUIET_HZ = np.array([SLOT_HZ, DOPPLER_HZ - 110.0, TONE_HZ[-1] + 110.0, TONE_HZ[-1] + 220.0])
DOPPLER_AMP = 10 ** (7 / 20)                # 7 dB over a data tone (A.5.5)
PREAMBLE_ELEMENTS = 12

RATES = (75, 150, 300, 600, 1200, 2400)
#: Distinct bits an element carries at each rate (table A-I).
BITS = {2400: 32, 1200: 16, 600: 8, 300: 4, 150: 2, 75: 1}
#: Table A-II: (odd bit, even bit) -> degrees; MARK is 1.
_DPHI4 = {(0, 1): 45.0, (0, 0): 135.0, (1, 0): 225.0, (1, 1): 315.0}
_DPHI2 = {1: 315.0, 0: 135.0}


def mode(name: str | int) -> int:
    """"600", "2400" or 600 -> the rate."""
    try:
        r = int(str(name).strip())
    except ValueError:
        r = -1
    if r not in RATES:
        raise ValueError(f"the 16-tone mode runs at {', '.join(map(str, RATES))} bit/s")
    return r


def quadrature(rate: int) -> bool:
    return rate >= 1200


def carried(rate: int) -> np.ndarray:
    """For each of the 16 tones, which symbol of the element it carries: a
    dibit at 2400 and 1200, a bit below (table A-I)."""
    per = BITS[rate] // 2 if quadrature(rate) else BITS[rate]
    return np.arange(16) % per


# --------------------------------------------------------------------------
# The modulator
# --------------------------------------------------------------------------

def _bits(data: bytes, order: str) -> np.ndarray:
    b = np.unpackbits(np.frombuffer(data, dtype=np.uint8),
                      bitorder="little" if order == "lsb" else "big")
    return b.astype(np.int64)


def element_count(rate: int, n_bytes: int) -> int:
    """Elements a message of ``n_bytes`` keys for, preamble and reference
    element among them."""
    rate = mode(rate)
    return PREAMBLE_ELEMENTS + 1 + -(-8 * n_bytes // BITS[rate])


def changes(rate: int, data: bytes, order: str = "lsb") -> np.ndarray:
    """The phase change of each tone in each data element, degrees."""
    rate = mode(rate)
    n = BITS[rate]
    bits = _bits(data, order)
    bits = np.r_[bits, np.zeros((-bits.size) % n, dtype=np.int64)]
    if bits.size == 0:
        return np.zeros((0, 16))
    rows = bits.reshape(-1, n)
    sym = carried(rate)
    if quadrature(rate):
        pairs = rows.reshape(rows.shape[0], -1, 2)
        table = np.array([[_DPHI4[(a, b)] for b in (0, 1)] for a in (0, 1)])
        per = table[pairs[:, :, 0], pairs[:, :, 1]]
    else:
        per = np.where(rows == 1, _DPHI2[1], _DPHI2[0])
    return per[:, sym]


def _element(freqs, amps, phases) -> np.ndarray:
    t = np.arange(T)
    return np.sum(np.asarray(amps)[:, None] *
                  np.cos(2 * np.pi * np.asarray(freqs)[:, None] * t[None, :] / FS
                         + np.asarray(phases)[:, None]), axis=0)


def transmit(data: bytes, rate: int | str = 2400, *,
             order: str = "lsb") -> np.ndarray:
    """A message -> real audio at ``spec.FS``, peak 1.0."""
    rate = mode(rate)
    dphi = np.radians(changes(rate, data, order))
    # A.5.4: the preamble's composite rms that of the data's (16 tones and
    # the Doppler tone), the Doppler tone 7 dB over the 1705 Hz one.
    p_data = 16 + DOPPLER_AMP ** 2
    a_sync = np.sqrt(p_data / (1 + DOPPLER_AMP ** 2))
    els = []
    k = 0

    def doppler_phase():
        return 2 * np.pi * DOPPLER_HZ * k * T / FS

    for i in range(PREAMBLE_ELEMENTS):
        els.append(_element([DOPPLER_HZ, SYNC_HZ], [a_sync * DOPPLER_AMP, a_sync],
                            [doppler_phase(), np.pi * i]))
        k += 1
    freqs = np.r_[DOPPLER_HZ, TONE_HZ]
    amps = np.r_[DOPPLER_AMP, np.ones(16)]
    phase = np.pi * np.arange(16) ** 2 / 16
    els.append(_element(freqs, amps, np.r_[doppler_phase(), phase]))
    k += 1
    for row in dphi:
        phase = phase + row
        els.append(_element(freqs, amps, np.r_[doppler_phase(), phase]))
        k += 1
    x = np.concatenate(els)
    return x / np.max(np.abs(x))


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read(x: np.ndarray, at: np.ndarray, df: float, freqs) -> np.ndarray:
    """Each tone's complex amplitude over the windows starting at samples
    ``at`` (each element's ``GUARD`` in): (windows, tones). Reckoned from
    the element's start, so that a tone of constant phase reads the same
    in every element."""
    at = np.asarray(at, dtype=np.int64)
    f = np.asarray(freqs, dtype=float) + df
    t = np.arange(W)
    z = x[at[:, None] + t[None, :]] @ np.exp(-2j * np.pi * t[:, None] * f[None, :] / FS)
    return z * np.exp(-2j * np.pi * GUARD * f / FS)[None, :]


ALL_HZ = np.r_[DOPPLER_HZ, TONE_HZ]


def _doppler_turn(zd: np.ndarray, zd_prev: np.ndarray) -> np.ndarray:
    """What an offset turns every tone by in an element, from the Doppler
    tone's change: it runs on unbroken, so it turns by its own frequency's
    worth besides, which is known."""
    return zd * np.conj(zd_prev) * np.exp(-2j * np.pi * DOPPLER_HZ * T / FS)


def decide(rate: int, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One element's 16 changes, turned back for the offset -> its bits and
    their soft values (positive: 1). The copies under diversity are added
    first, each weighed by its own strength."""
    rate = mode(rate)
    sym = carried(rate)
    n_sym = sym.max() + 1
    acc = np.zeros(n_sym, dtype=complex)
    np.add.at(acc, sym, w)
    if quadrature(rate):
        # Table A-II: the odd bit is MARK below the real axis (225, 315
        # degrees), the even bit right of the imaginary (45, 315).
        odd, even = -np.imag(acc), np.real(acc)
        soft = np.stack([odd, even], axis=1).ravel()
    else:
        soft = np.real(acc * np.exp(-1j * np.radians(315.0)))
    return (soft > 0).astype(np.uint8), soft


@dataclass
class Reception:
    rate: int
    data: bytes
    start_sample: int
    end_sample: int
    freq_hz: float
    snr_db: float = float("nan")


def receive(x: np.ndarray, rate: int | str = 2400, *, start: int = 0,
            order: str = "lsb") -> Reception | None:
    """A whole recording at ``spec.FS`` -> the first message in it at or
    after sample ``start``."""
    from .stream import Receiver16, receive_whole
    got = receive_whole(x, Receiver16(rate, bit_order=order), start=start, parallel=True)
    if got is None:
        return None
    first, data, end = got
    return Reception(mode(rate), data, first.at, end.at, first.freq_hz, end.snr_db)
