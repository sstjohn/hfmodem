# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP transmitter: turn a frame's bytes into 12 kHz int16 audio.

Assembles the wire waveform exactly as ardopcf does (`Modulate.c`, MIT):

    two-tone leader → phase-reversed sync symbol → 10-symbol 4FSK frame-type
    header → data (4FSK tone stream, or summed differential PSK/QAM carriers)
    → 1500 Hz trailer

then runs the whole stream through ardopcf's frequency-sampling TX filter (a
120-tap comb feeding a bank of 100 Hz resonators around 1500 Hz) which shapes the
occupied bandwidth and drops the first 60 samples of group delay.

The byte layer lives in :mod:`besra.frame`; this module is pure DSP. The sample
templates are :mod:`besra.dsp.templates`.

Exactness: the templates reproduce ardopcf's arrays bit-exact for PSK/QAM and
50-baud 4FSK, within ±2 for the leader / 100-/600-baud 4FSK (a historical
rounding artifact). The resonator filter is evaluated in float64; ardopcf's
shipped binary evaluates it in C ``float``, so the int16 output tracks the
reference to within a small, documented per-sample bound rather than bit-exactly
(see ``tests/besra/test_modulator.py``). Everything upstream of the filter — template
values, sign conventions, differential phase, carrier scaling, soft clip — is
reproduced exactly.

The parts of this module ``NOTICE`` names as ardopcf's are under that project's
MIT licence, Copyright (c) 2014-2024 Rick Muething, John Wiseman, Peter LaRue;
the copyright and permission notice it requires ship in ``NOTICE``.
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

# Per-carrier-count filter bandwidth and the crest-factor scaling ardopcf applies
# to the summed PSK/QAM carriers (`Modulate.c:505-526`, empirical).
_FILTER_WIDTH = {1: 200, 2: 500, 4: 1000, 8: 2000}
_FILTER_WIDTH_FOR_FSK = {50: 200, 100: 500, 600: 2000}
_CAR_START = {1: 4, 2: 3, 4: 2, 8: 0}


def _psk_scale(carriers: int, mod: F.Mod) -> float:
    if carriers == 1:
        return 1.2
    if carriers == 2:
        return 0.67 if mod is F.Mod.QAM16 else 0.65
    if carriers == 4:
        return 0.4
    return 0.27 if mod is F.Mod.QAM16 else 0.25


# --------------------------------------------------------------------------- #
# The TX filter — comb + resonator bank (`Modulate.c:676-914`).
# --------------------------------------------------------------------------- #

class _TxFilter:
    """ardopcf's frequency-sampling transmit filter, applied to a frame in one
    pass; the first 60 samples (half the 120-tap comb) are group delay and dropped.

    The resonator bank runs at pole radius R=0.9995 with a comb that notches the
    poles, giving a near-FIR passband ~`width` Hz wide around 1500 Hz. Middle
    resonators of the wider filters have their outputs truncated to integers
    before summing — an ardopcf quirk preserved here for fidelity."""

    _R = 0.9995
    _N = 120

    def __init__(self, width: int, centre: int = 1500, drive: int = 100):
        self._drive = drive
        self._rn = self._R**self._N
        self._r2 = self._R**2
        cs = centre // 100
        first, last = {
            200: (cs - 1, cs + 1),
            500: (cs - 3, cs + 3),
            1000: (cs - 5, cs + 5),
            2000: (cs - 10, cs + 10),
        }[width]
        bins = np.arange(first, last + 1)
        self._coef = 2 * self._R * np.cos(2 * np.pi * bins / self._N)

        # Per-bin combine rule: edge/inner bins scale by a float transition
        # coefficient; the deep-middle bins add ±trunc(output). `_tcoef` folds
        # the sign in; `_trunc` marks which bins truncate first.
        tcoef = np.empty(bins.size)
        trunc = np.zeros(bins.size, dtype=bool)
        for i, j in enumerate(bins):
            if width == 200:
                tcoef[i] = 0.7389 if j in (first, last) else -1.0
            elif width == 500:
                if j in (first, last):
                    tcoef[i] = 0.10601
                elif j in (first + 1, last - 1):
                    tcoef[i] = -0.59383
                else:
                    tcoef[i], trunc[i] = (1.0 if j % 2 == 0 else -1.0), True
            else:  # 1000 / 2000
                edge = 0.377 if width == 1000 else 0.371
                if j in (first, last):
                    tcoef[i] = edge
                else:
                    tcoef[i], trunc[i] = (1.0 if j % 2 == 0 else -1.0), True
        self._tcoef = tcoef
        self._trunc = trunc

    def render(self, src: np.ndarray) -> np.ndarray:
        # Drive scaling is ardopcf's `(short)(sample * drive / 100)`: integer
        # floor division (which diverges from float scaling on negative samples)
        # followed by an int16 wrap.
        s = (src * self._drive // 100).astype(np.int16).astype(np.float64)

        # Comb: zin[n] = s[n] - rn·s[n-120], then zcomb[n] = zin[n] - r2·zin[n-2],
        # with s and zin zero before the frame.
        n = self._N
        zin = s.copy()
        zin[n:] -= self._rn * s[:-n]
        zcomb = zin.copy()
        zcomb[2:] -= self._r2 * zin[:-2]

        # One resonator per bin, z0[k] = zcomb[k] + coef·z0[k-1] - r2·z0[k-2],
        # weighted straight into a sample-major column so the per-sample sum
        # reduces over the contiguous bin axis — numpy's pairwise summation order
        # depends on the reduced axis being contiguous, and the 11- and 21-bin
        # filters land in a different order if it is not.
        shaped = np.empty((zcomb.size - n // 2, self._tcoef.size))
        for i, coef in enumerate(self._coef):
            z0 = signal.lfilter([1.0], [1.0, -coef, self._r2], zcomb)[n // 2:]
            if self._trunc[i]:
                np.trunc(z0, out=z0)
            shaped[:, i] = z0 * self._tcoef[i]

        filt = shaped.sum(axis=1) * 0.00833333333
        np.clip(filt, -32700.0, 32700.0, out=filt)
        return filt.astype("<i2")


# --------------------------------------------------------------------------- #
# Leader, sync, and the 4FSK frame-type header (`Modulate.c:39-116`).
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
# 4FSK data (`Mod4FSKDataAndPlay` / `Mod4FSK600BdDataAndPlay`).
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
# PSK / QAM data (`Calc1CarPSKSymbols` / `PlayPSKSymbols`).
# --------------------------------------------------------------------------- #

_BITS_PER_SYMBOL = {F.Mod.PSK4: 2, F.Mod.PSK8: 3, F.Mod.QAM16: 4}


def _carrier_symbols(block: bytes, mod: F.Mod) -> list[int]:
    """Differential phase/magnitude symbols for one carrier. Low 3 bits are the
    absolute phase index (0-7 = 0..315 deg), accumulated from a phase-0 reference;
    4PSK steps in units of 2 (SymSet) so the full circle is still reachable. Bit 3
    is 16QAM's half-magnitude flag, taken absolute from the raw symbol."""
    bps = _BITS_PER_SYMBOL[mod]
    sym_set = 2 if mod is F.Mod.PSK4 else 1
    count = len(block) * 8 // bps
    out: list[int] = []
    databuf = 0
    buffered = 0
    ptr = 0
    for i in range(count):
        if buffered < bps:
            databuf = (databuf + (block[ptr] << (8 - buffered))) & 0xFFFF
            ptr += 1
            buffered += 8
        raw = databuf >> (16 - bps)
        databuf = (databuf << bps) & 0xFFFF
        buffered -= bps
        prior = 0 if i == 0 else out[i - 1]
        out.append(((prior + raw * sym_set) & 7) + (raw & 0x08))
    return out


def _soft_clip(x: np.ndarray) -> np.ndarray:
    """Compress the summed waveform above ±30000 (`Modulate.c:337`)."""
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
    """The flat ardopcf ``bytEncodedBytes``: two type bytes then the per-carrier
    blocks (or the control-frame body). Built on :mod:`besra.frame`."""
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
