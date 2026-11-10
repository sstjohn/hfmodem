# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP transmitter: turn a frame's bytes into 12 kHz int16 audio.

An ARDOP frame on the air is

    two-tone leader → phase-reversed sync symbol → 10-symbol 4FSK frame-type
    header → data (4FSK tone stream, or summed differential PSK/QAM carriers)
    → 1500 Hz trailer

and the whole of it is band-limited by a frequency-sampling filter: a 120-sample
comb feeding resonators on the 100 Hz grid around 1500 Hz, as many as the
bandwidth spans, whose first 60 output samples are group delay and dropped.

The byte layer lives in :mod:`besra.frame`; this module is pure DSP. The sample
templates are :mod:`besra.dsp.templates`.

Exactness: what other ARDOP stations transmit is the yardstick. Template values,
signs, differential phase, carrier scaling and the soft clip agree exactly; the
leader and the 100/600-baud tones carry the ±2 template residual described in
:mod:`besra.dsp.templates`. The filter runs in float64 where the stations we
measure against run it in single precision, so the int16 output agrees to within
a small per-sample bound rather than bit for bit (``tests/besra/test_modulator.py``).
The filter's weights and the carrier scaling are fitted values, kept as found
because they set the spectrum a peer's receiver was built for.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..frame import callsign
from ..frame import frame as F
from ..dsp.templates import (
    FSK_50BD,
    FSK_100BD,
    FSK_600BD,
    LEADER_50BD,
    PSK_100BD,
    TRAILER_1500HZ,
)

SAMPLE_RATE = 12000
DEFAULT_LEADER_MS = 240
DEFAULT_TRAILER_MS = 20

# The filter bandwidth each mode is sent through, by carrier count for PSK/QAM
# and by baud for 4FSK, and the first of the nine carriers a PSK/QAM mode rides.
_FILTER_WIDTH = {1: 200, 2: 500, 4: 1000, 8: 2000}
_FILTER_WIDTH_FOR_FSK = {50: 200, 100: 500, 600: 2000}
_CAR_START = {1: 4, 2: 3, 4: 2, 8: 0}

#: Gain on the summed PSK/QAM carriers, by carrier count: (PSK, 16QAM). More
#: carriers sum to a higher crest, so each is sent quieter.
_CARRIER_GAIN = {1: (1.2, 1.2), 2: (0.65, 0.67), 4: (0.4, 0.4), 8: (0.25, 0.27)}


def _psk_scale(carriers: int, mod: F.Mod) -> float:
    return _CARRIER_GAIN[carriers][mod is F.Mod.QAM16]


# --------------------------------------------------------------------------- #
# The transmit filter.
# --------------------------------------------------------------------------- #

#: Per filter width: how many 100 Hz bins it spans either side of the centre, and
#: the weights of its outer bins, edge first. The bins between take ±1 by the
#: parity of their index and are cut to whole numbers before they are summed,
#: which is part of what fixes the shape; the 200 Hz filter has no such bins.
_FILTER_SHAPE = {
    200: (1, (0.7389, -1.0), False),
    500: (3, (0.10601, -0.59383), True),
    1000: (5, (0.377,), True),
    2000: (10, (0.371,), True),
}


class _TxFilter:
    """The transmit filter over one whole frame.

    A comb of length 120 at radius R = 0.9995 cancels the poles of a bank of
    resonators on the 100 Hz bins it spans, which together make a near-FIR
    passband about ``width`` Hz wide around ``centre``. The first 60 output
    samples, half the comb, are its delay and are dropped."""

    _R = 0.9995
    _N = 120

    def __init__(self, width: int, centre: int = 1500, drive: int = 100):
        self._drive = drive
        self._rn = self._R**self._N
        self._r2 = self._R**2
        span, edges, whole_inner = _FILTER_SHAPE[width]
        bins = np.arange(centre // 100 - span, centre // 100 + span + 1)
        self._coef = 2 * self._R * np.cos(2 * np.pi * bins / self._N)

        self._weight = np.where(bins % 2 == 0, 1.0, -1.0)
        self._whole = np.full(bins.size, whole_inner)
        for depth, w in enumerate(edges):
            self._weight[[depth, -1 - depth]] = w
            self._whole[[depth, -1 - depth]] = False

    def render(self, src: np.ndarray) -> np.ndarray:
        # Drive is applied to integer samples with floor division, so a negative
        # sample rounds away from zero, and the result wraps into int16; scaling
        # in float would differ on exactly those samples.
        s = (src * self._drive // 100).astype(np.int16).astype(np.float64)

        # The comb, x[n] - R^120·x[n-120], and the R² two-sample notch that
        # pairs with every resonator's pole radius; both start from silence.
        n = self._N
        combed = s.copy()
        combed[n:] -= self._rn * s[:-n]
        notched = combed.copy()
        notched[2:] -= self._r2 * combed[:-2]

        # Each resonator y[k] = x[k] + c·y[k-1] - R²·y[k-2] is written into its
        # own column, so the sum across bins runs along contiguous memory: numpy
        # orders a pairwise sum by layout, and the 11- and 21-bin filters round
        # differently if the bins are strided.
        bank = np.empty((notched.size - n // 2, self._weight.size))
        for i, coef in enumerate(self._coef):
            y = signal.lfilter([1.0], [1.0, -coef, self._r2], notched)[n // 2:]
            if self._whole[i]:
                np.trunc(y, out=y)
            bank[:, i] = y * self._weight[i]

        out = bank.sum(axis=1) * 0.00833333333       # ≈ 1/120, the comb's gain
        np.clip(out, -32700.0, 32700.0, out=out)
        return out.astype("<i2")


# --------------------------------------------------------------------------- #
# Leader, sync, and the 4FSK frame-type header.
# --------------------------------------------------------------------------- #

def _leader_and_sync(frame_type: int, session_id: int, leader_ms: int) -> np.ndarray:
    nsym = leader_ms // 20
    # Every symbol flips phase; the final (sync) symbol repeats — the phase
    # reversal the receiver locks symbol timing on.
    signs = (-1 if nsym & 1 else 1) * (-1) ** np.arange(nsym)
    signs[-1:] *= -1
    leader = signs[:, None] * LEADER_50BD

    # 10 4FSK symbols: [type dibits][parity][type^session dibits][parity]. The
    # template sign alternates by absolute symbol position (5j+k) to hold phase
    # continuity across boundaries.
    syms = F.header_symbols(frame_type, session_id)
    header = ((-1) ** np.arange(len(syms)))[:, None] * FSK_50BD[syms]
    return np.concatenate([leader.ravel(), header.ravel()])


def _trailer(trailer_ms: int) -> np.ndarray:
    return np.tile(TRAILER_1500HZ.astype(np.int64), 1 + trailer_ms // 10)


# --------------------------------------------------------------------------- #
# 4FSK data.
# --------------------------------------------------------------------------- #

def _fsk_data(data: bytes, baud: int) -> np.ndarray:
    table = {50: FSK_50BD, 100: FSK_100BD, 600: FSK_600BD}[baud]
    b = np.frombuffer(data, dtype=np.uint8)
    dibits = np.stack([(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3], axis=1)
    tones = table[dibits].astype(np.int64)
    if baud != 600:  # 600 baud plays templates unflipped
        tones *= np.array([1, -1, 1, -1])[:, None]  # +,-,+,- within each byte
    return tones.ravel()


# --------------------------------------------------------------------------- #
# PSK / QAM data.
# --------------------------------------------------------------------------- #

_BITS_PER_SYMBOL = {F.Mod.PSK4: 2, F.Mod.PSK8: 3, F.Mod.QAM16: 4}


def _carrier_symbols(block: bytes, mod: F.Mod) -> list[int]:
    """One carrier's symbols: a phase index 0-7, in 45° steps, in the low three
    bits, and for 16QAM the inner-ring flag in bit 3.

    The block is a bit stream, most significant bit first, cut into symbols of
    the mode's width; bits too few to fill a last symbol are not sent. Phase is
    differential from the phase-0 reference symbol: each symbol turns it on by its
    value, in 90° steps for 4PSK and 45° otherwise. The ring flag is the 16QAM
    value's top bit, sent as it stands rather than accumulated."""
    width = _BITS_PER_SYMBOL[mod]
    bits = np.unpackbits(np.frombuffer(block, dtype=np.uint8))
    count = bits.size // width
    values = bits[:count * width].reshape(count, width) @ (1 << np.arange(width)[::-1])
    turn = np.cumsum(values * (2 if mod is F.Mod.PSK4 else 1)) & 7
    return (turn | values & 8).tolist()


def _soft_clip(x: np.ndarray) -> np.ndarray:
    """Compress the summed waveform above ±30000, with a square-root knee that
    tops out at ±32700."""
    hi, lo = x > 30000, x < -30000
    out = x.copy()
    out[hi] = np.minimum(32700.0, 30000 + 20 * np.sqrt(x[hi] - 30000)).astype(np.int64)
    out[lo] = np.maximum(-32700.0, -30000 - 20 * np.sqrt(-(x[lo] + 30000))).astype(np.int64)
    return out


def carrier_indices(carriers: int) -> list[int]:
    """Which of the nine `templates.PSK_CARRIERS_HZ` a PSK/QAM mode rides, in the
    order its per-carrier blocks are assigned."""
    car, out = _CAR_START[carriers], []
    for _ in range(carriers):
        out.append(car)
        car += 2 if car == 3 else 1      # multi-carrier modes skip 1500 Hz
    return out


def _psk_data(per_carrier: list[list[int]], carriers: int, scale: float) -> np.ndarray:
    syms = np.array(per_carrier, dtype=np.int64)
    phase, shift = syms & 0x07, syms >> 3
    # Phases 4-7 are the first four templates negated, and 16QAM's half-magnitude
    # shift lands on the positive template — `-(v >> 1)` != `(-v) >> 1` for odd v.
    rows = np.where(phase < 4, phase, phase - 4)
    signs = np.where(phase < 4, 1, -1)

    acc = np.zeros((syms.shape[1], 120), dtype=np.int64)
    for c, car in enumerate(carrier_indices(carriers)):
        acc += signs[c, :, None] * (PSK_100BD[car][rows[c]] >> shift[c, :, None])

    return _soft_clip((acc.astype(np.float32) * np.float32(scale)).astype(np.int64)).ravel()


# --------------------------------------------------------------------------- #
# Frame assembly.
# --------------------------------------------------------------------------- #

def _encoded_bytes(frame_type: int, payload: bytes, session_id: int,
                   meta: dict) -> bytes:
    """Every byte the frame sends: the frame type, the type XOR the session ID,
    then the per-carrier blocks (or the control-frame body). Built on
    :mod:`besra.frame`."""
    fd = F.FRAMES[frame_type]
    head = bytes([frame_type, frame_type ^ (session_id & 0xFF)])
    name = fd.name

    # Payload-free control frames (BREAK/IDLE/DISC/END/ConRej*/DATAACK/DATANAK).
    if name in {"BREAK", "IDLE", "DISC", "END", "ConRejBusy", "ConRejBW",
                "DATAACK", "DATANAK"}:
        return head

    # For the structured control frames the ARQ layer supplies the already-built
    # body as ``payload``; the fixture tests instead pass ``meta`` and we build
    # it here. Prefer a supplied payload so one seam serves both callers.

    if name.startswith("ConAck"):
        if payload:
            return head + payload[:3]
        ms = int(meta["timing"])
        timing = 0 if not 0 <= ms <= 2550 else min(255, ms // 10)
        return head + bytes([timing, timing, timing])

    if name == "PingAck":
        if payload:
            return head + payload[:3]
        sn = int(meta["sn"])
        quality = int(meta["quality"])
        value = 0xF8 if sn >= 21 else (((sn + 10) & 0x1F) << 3)
        value += max(0, (quality - 30) // 10) & 7
        return head + bytes([value, value, value])

    if name == "IDFrame":
        body = payload[:12] if payload else (
            callsign.pack_callsign(meta["caller"]) + callsign.pack_grid(meta["grid"]))
        return head + F.carrier_block(body, 12, 4, frame_type, with_crc=False)

    if name.startswith("ConReq") or name == "Ping":
        body = payload[:12] if payload else (
            callsign.pack_callsign(meta["caller"]) + callsign.pack_callsign(meta["target"]))
        return head + F.carrier_block(body, 12, 4, frame_type, with_crc=False)

    # Data frames — one block per carrier, or the 600-baud full frame's three
    # sub-packets; `build_data_frame` owns that geometry.
    _, blocks = F.build_data_frame(frame_type, payload, session_id)
    return head + b"".join(blocks)


def render_frame(frame_type: int, payload: bytes = b"", session_id: int = 0xFF,
                 leader_ms: int = DEFAULT_LEADER_MS,
                 trailer_ms: int = DEFAULT_TRAILER_MS, drive: int = 100,
                 **meta) -> np.ndarray:
    """Render one ARDOP frame to 12 kHz int16 samples.

    ``payload`` is the raw data bytes for data frames; control frames take their
    content via keyword ``meta`` (``timing=`` for ConAck, ``sn=``/``quality=`` for
    PingAck, ``caller=``/``target=``/``grid=`` for ConReq/Ping/ID)."""
    fd = F.FRAMES[frame_type]
    encoded = _encoded_bytes(frame_type, payload, session_id, meta)
    lead = _leader_and_sync(frame_type, session_id, leader_ms)

    if fd.mod is F.Mod.FSK4:
        filt = _TxFilter(_FILTER_WIDTH_FOR_FSK[fd.baud], drive=drive)
        return filt.render(np.concatenate(
            [lead, _fsk_data(encoded[2:], fd.baud), _trailer(trailer_ms)]))

    # PSK / QAM.
    per_block = fd.k + fd.r + 3
    blocks = [encoded[2 + c * per_block:2 + (c + 1) * per_block]
              for c in range(fd.carriers)]
    per_carrier = [_carrier_symbols(b, fd.mod) for b in blocks]
    scale = _psk_scale(fd.carriers, fd.mod)

    # One phase-0 full-scale reference symbol per carrier, then the data symbols.
    filt = _TxFilter(_FILTER_WIDTH[fd.carriers], drive=drive)
    return filt.render(np.concatenate([
        lead,
        _psk_data([[0]] * fd.carriers, fd.carriers, scale),
        _psk_data(per_carrier, fd.carriers, scale),
        _trailer(trailer_ms)]))
