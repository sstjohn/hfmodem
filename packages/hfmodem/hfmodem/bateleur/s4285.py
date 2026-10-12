# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 4285: 256-symbol frames, 75 to 3600 bit/s.

Every frame opens with the same 80-symbol sync, and its other 176 symbols —
four runs of 32 data symbols with 16 reference symbols between them — are
randomised by a sequence that restarts each frame (Annex A). There is no
preamble and nothing on the air names the rate: both ends are set to the same
mode, as the standard intends.

The coded modes follow Annex E: the K=7 code, repeated in pairs below 600
bit/s, a 32-row convolutional interleaver, and a message framed as a 32-bit
start-of-message pattern, the message, the 32-bit end-of-message pattern and
enough zeros to flush the interleaver and the decoder. The receiver looks for
the start pattern in what it decodes and stops at the end pattern, so it hands
back the message and nothing around it. The uncoded modes carry a bare bit
stream, as Annex A does. Bytes go least-significant bit first by default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fec, interleave, phy
from .mil110 import bytes_to_bits
from .spec import (
    find_eom as find_eom,
    DATA4285_PER_FRAME, EOM110_BITS, FRAME4285, LAYOUT4285, MODES4285,
    SOM4285_BITS, SPS, Mode4285, gray4285, scramble4285, sync4285,
)

EOM_BITS = EOM110_BITS                  # Annex E uses 188-110's pattern

SYNC = sync4285()
SCR = scramble4285()


def _slots_per_frame(mode: Mode4285) -> int:
    return DATA4285_PER_FRAME * mode.slots_per_symbol


def _per_frame(mode: Mode4285) -> int:
    """Message bits a frame carries."""
    if mode.coded:
        return _slots_per_frame(mode) // (2 * mode.repeat)
    return DATA4285_PER_FRAME * mode.bits_per_symbol


def _info_len(mode: Mode4285, n_bytes: int) -> int:
    """SOM, message, EOM and flush, padded to whole frames."""
    if mode.coded:
        n = len(SOM4285_BITS) + 8 * n_bytes + len(EOM_BITS) + mode.flush_bits
    else:
        n = max(8 * n_bytes, 1)                     # an empty message is one frame
    return n + (-n) % _per_frame(mode)


def symbol_count(mode: Mode4285, n_bytes: int) -> int:
    """How many symbols ``frames`` makes of an ``n_bytes`` message."""
    return _info_len(mode, n_bytes) // _per_frame(mode) * FRAME4285


def _info_bits(mode: Mode4285, data: bytes, order: str = "lsb") -> np.ndarray:
    """What goes into the encoder: framed and flushed if coded."""
    body = bytes_to_bits(data, order)
    if mode.coded:
        body = np.concatenate([SOM4285_BITS, body, EOM_BITS]).astype(np.uint8)
    n = _info_len(mode, len(data))
    return np.concatenate([body, np.zeros(n - body.size, dtype=np.uint8)])


def data_indices(mode: Mode4285, data: bytes, order: str = "lsb") -> np.ndarray:
    """Pre-randomiser 8-PSK indices for the data symbols, 128 a frame."""
    bits = _info_bits(mode, data, order)
    if mode.coded:
        slots, _ = fec.encode(bits)
        if mode.repeat > 1:                         # in pairs (Annex E 2(b))
            slots = np.repeat(slots.reshape(-1, 2), mode.repeat, axis=0).ravel()
        slots = interleave.interleave4285(slots, mode.increment)
        if mode.slots_per_symbol == 4:
            slots = slots.reshape(-1, 4)[:, :3].ravel()      # skip every fourth row
    else:
        slots = bits
    nb = mode.bits_per_symbol
    vals = slots.reshape(-1, nb) @ (1 << np.arange(nb - 1, -1, -1))
    return np.asarray(gray4285(nb))[vals]


def frames(mode: Mode4285, data: bytes, order: str = "lsb") -> np.ndarray:
    """The whole transmission as 8-PSK indices."""
    d = data_indices(mode, data, order).reshape(-1, DATA4285_PER_FRAME)
    out = []
    for f in d:
        body = np.zeros(176, dtype=np.int64)
        body[~LAYOUT4285] = f
        out.append(SYNC)
        out.append((body + SCR) % 8)
    return np.concatenate(out)


def transmit(data: bytes, mode: Mode4285 | str = "1200L", *, order: str = "lsb",
             trim: bool = True) -> np.ndarray:
    """Message -> real audio at ``spec.FS``, peak 1.0. ``trim`` as `phy.modulate`."""
    if isinstance(mode, str):
        mode = MODES4285[mode]
    return phy.modulate(frames(mode, data, order), trim)


# --------------------------------------------------------------------------
# Receive
# --------------------------------------------------------------------------

@dataclass
class Reception:
    mode: Mode4285
    data: bytes
    frames: int
    start_sample: int
    end_sample: int
    freq_hz: float
    snr_db: float
    eom: bool = False           # coded: the end-of-message pattern arrived


_SOM_TAIL = np.array(SOM4285_BITS[6:], dtype=np.uint8)     # Annex E App 1 5(b)


def find_som(bits: np.ndarray, start: int = 0) -> int:
    """Index just past the SOM, or -1. The receiver matches the SOM's last 26
    bits exactly, as Annex E Appendix 1 asks; its first six are what the
    decoder's start-up may still have wrong."""
    b = np.asarray(bits, dtype=np.uint8)
    n = _SOM_TAIL.size
    if b.size - start < n:
        return -1
    w = np.lib.stride_tricks.sliding_window_view(b[start:], n)
    hit = np.flatnonzero(np.all(w == _SOM_TAIL, axis=1))
    return int(start + hit[0] + n) if hit.size else -1




def _plan(mode: Mode4285, n_frames: int) -> phy.Plan:
    kind = np.empty(FRAME4285, dtype=np.int64)
    kind[:80] = phy.KNOWN
    kind[80:] = np.where(LAYOUT4285, phy.KNOWN, phy.DATA)
    val = np.concatenate([SYNC, SCR])                 # references carry 0
    rot = np.concatenate([np.zeros(80, dtype=np.int64), SCR])
    return phy.Plan(np.tile(kind, n_frames), np.tile(val, n_frames),
                    np.tile(rot, n_frames), tuple(gray4285(mode.bits_per_symbol)))


def confirmed(bb: np.ndarray, hit: phy.Hit, threshold: float) -> bool:
    """Is there a second sync a frame after ``hit``?

    Eighty symbols is a short thing to find in noise, and one run of them
    clears the threshold by chance often enough to start a phantom reception
    every few minutes of empty band. The frame repeats every 256 symbols, so
    the next sync is where the first one says it will be, or the first was
    not one.
    """
    at = hit.sample + FRAME4285 * SPS
    seg = bb[at - 2: at + len(SYNC) * SPS + SPS + 2]
    if seg.size < len(SYNC) * SPS + SPS + 4:
        return False
    chunks, energy = phy.correlate(seg, SYNC, 16, freq_hz=hit.freq_hz)
    score, _ = phy.differential_metric(chunks, energy, 16)
    return bool(np.max(score[:5]) > threshold)


#: Baseband a hit needs behind it before ``confirmed`` can answer.
CONFIRM_SAMPLES = (FRAME4285 + len(SYNC) + 2) * SPS


def receive(audio: np.ndarray, mode: Mode4285 | str = "1200L", *, start: int = 0,
            threshold: float = 0.35, order: str = "lsb") -> Reception | None:
    """Find the first 4285 transmission in ``audio`` (real, at ``spec.FS``) at
    or after sample ``start`` and decode it in ``mode``."""
    if isinstance(mode, str):
        mode = MODES4285[mode]
    from .stream import Receiver4285, receive_whole
    got = receive_whole(audio, Receiver4285(mode, bit_order=order, threshold=threshold), start=start)
    if got is None:
        return None
    first, data, end = got
    frames = max(0, (end.at - first.at) // (FRAME4285 * SPS))
    return Reception(mode, data, frames, first.at, end.at, first.freq_hz, end.snr_db, end.complete)


#: How many frames late acquisition may be (a sync or more lost in a fade)
#: and the message still be read: each is a decode more when it is.
LATE_FRAMES = 4


#: Known symbols a frame holds: its sync and its three reference runs.
_KNOWN_PER_FRAME = 80 + 48

