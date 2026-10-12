# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 188-110A "mod" waveform: received to its raw channel bits.

No text describes this waveform. What is here was measured from seven
published off-air recordings, with no other implementation consulted:

- **A tone header.** Four tones (900, 1350, 1800 and 2250 Hz) or seven (450
  to 3150 Hz, 450 Hz apart) for about 1.6 s, ending 3.8 s before the data
  phase. The 1800 Hz tone is the serial carrier, whatever the receiver's
  tuning.
- **A parallel section.** About eighteen tones on a 75 Hz grid at 75 Bd,
  filling the 3.8 s. Not read here.
- **A data phase.** The serial waveform of 188-110A at 150 to 1200 bit/s: 2400
  symbols a second on 1800 Hz, frames of 20 data and 20 known symbols, all
  added to the 160-symbol randomiser, which starts at the first frame. The
  data symbols are BPSK (0 or 4 before randomising). Its first 16 frames
  (640 symbols, 267 ms) carry zeros, which makes them a 640-symbol known
  sequence: the synchronisation this receiver looks for.
- **An end.** Some bursts end with a run of ones, every data symbol turned
  half round, some 500 of them; the others simply stop.
- **Two formats.** In some bursts every channel bit is sent twice in a row;
  in others no bit is. Which one is not read from the header: the receiver
  sees it in the bits.
- **A matrix.** The bits of a burst are one matrix of six columns, sent a
  row at a time; the message fills it a column at a time, and what it does
  not fill stays zero. Each column ends with a byte of its own, repeated four
  times (``COLUMN_MARKERS``), and 16 bits more; the payload is above.

How the payload is coded is not known. It is not 188-110's code and
interleaver at any of its BPSK rates, nor (read down the columns) a rate-1/2,
constraint-length-7 convolutional code; it looks random, though unused space
stays zero. So this receiver stops at the columns' bits.

The sync phase alone is a 188-110 data phase sending zeros, which an
ordinary 188-110 transmission also does (its flush, for one). So a sync
phase is taken as this waveform's only with the tone header before it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import phy
from .spec import FS, SPS, scramble110

_SCR = scramble110()

UNKNOWN, KNOWN = 20, 20
FRAME = UNKNOWN + KNOWN
SYNC_FRAMES = 16
#: The synchronisation phase as it is sent: 16 frames of zero data, so every
#: symbol is the randomiser's.
SYNC = _SCR[np.arange(SYNC_FRAMES * FRAME) % _SCR.size]

#: The header tones every variant has, from the carrier (Hz).
HEADER_TONES = (-900.0, -450.0, 0.0, 450.0)
#: Where the header lies before the sync phase (s): it ended 3.8 s ahead in
#: every recording, and lasted some 1.6 s.
HEADER_SPAN = (5.2, 4.0)
#: The share of the band's power in the header tones taken as a header: the
#: recordings' headers made 0.6 to 0.98, anything else under 0.1.
HEADER_SHARE = 0.5
#: The search's threshold. Noise alone scores under 0.05 against the 640
#: symbols; the weakest burst in the recordings scored 0.17.
THRESHOLD = 0.12
#: Ones in a row taken as the end marker (it ran 507).
END_RUN = 200

#: A burst carries whole blocks of 6432 bits. The blocks of a burst are one
#: matrix of 6 columns, sent a row at a time; the message fills it a column at
#: a time, zeros where it runs short (``Reception.columns``).
BLOCK_BITS = 6432
COLUMNS = 6
#: The byte each column carries four times near its end, after its payload,
#: whatever the message: a marker, one per column. In a burst of k blocks it
#: is in the last block's rows, 1056 - 32k of them down (the doubled format
#: puts it at 1040), so a burst's length is read from where they are.
COLUMN_MARKERS = bytes((0x4F, 0x2F, 0x8E, 0x6C, 0x4A, 0x20))
#: Bits of the six markers' 192 that may be wrong and the block count still
#: taken. On the recordings the right count's markers had 0 to 25 wrong, every
#: other count's 58 or more.
MARKER_ERRORS = 40
MAX_BLOCKS = 5
_MARKER_BITS = np.array([np.unpackbits(np.full(4, m, dtype=np.uint8)) for m in COLUMN_MARKERS])


@dataclass
class Reception:
    bits: np.ndarray            # channel bits after the sync phase, to the end
    repeat: int                 # 2 where every bit came twice, else 1
    payload: np.ndarray         # the bits once each (the pairs combined)
    frames: int                 # data frames after the sync phase
    end_marker: bool            # the burst ended with its run of ones
    start_sample: int           # the sync phase's first symbol
    end_sample: int
    freq_hz: float
    snr_db: float
    framed: int = 0             # blocks, as the column markers counted them

    @property
    def data(self) -> bytes:
        """The payload packed first bit high, as a hex dump shows it."""
        b = self.payload[: self.payload.size - self.payload.size % 8]
        return np.packbits(b).tobytes()

    @property
    def blocks(self) -> int:
        """Whole 6432-bit blocks, counting one nearly all there (a burst
        without an end marker can lose its last few frames)."""
        if self.framed:
            return self.framed
        return int(round(self.payload.size / BLOCK_BITS)) if self.payload.size else 0

    @property
    def columns(self) -> list[bytes]:
        """The six columns, each the bytes written down it, first bit high:
        the message in its order. Bits short of whole blocks count as zero."""
        n = self.blocks * BLOCK_BITS
        if n == 0:
            return [b""] * COLUMNS
        b = np.zeros(n, dtype=np.uint8)
        m = min(n, self.payload.size)
        b[:m] = self.payload[:m]
        return [np.packbits(c).tobytes() for c in b.reshape(-1, COLUMNS).T]


def plan(n_frames: int) -> phy.Plan:
    """The sync phase, then ``n_frames`` frames of BPSK data and probes; the
    randomiser runs on from the sync phase's first symbol."""
    n = SYNC.size + n_frames * FRAME
    k = np.arange(n)
    kind = np.full(n, phy.KNOWN)
    data = (k >= SYNC.size) & (k % FRAME < UNKNOWN)
    kind[data] = phy.DATA
    scr = _SCR[k % _SCR.size]
    return phy.Plan(kind, scr, scr, (0, 4))


def loss_stop(after: int, rate: float = 0.005):
    """The signal has gone when the probes have been as wrong as noise makes
    them for a while: past 0.8 (as the 188-110 receiver judges it, which a
    strong signal's fade is not) and three times what the sync phase made
    them. On the recordings the probes' error ran at 0.2 to 0.4 with the
    signal there and 1.2 with it gone."""
    st = [0.0, 0.0, 0.0, 0]

    def stop(k, e2):
        if k < after:
            st[2] += e2
            st[3] += 1
            return False
        base = max(st[2] / max(st[3], 1), 1e-3)
        st[0] = (1 - rate) * st[0] + rate * min(e2, 3.0)
        st[1] = (1 - rate) * st[1] + rate * min(e2 / base, 20.0)
        return st[0] > 0.8 and st[1] > 3.0
    return stop


def header_share(bb: np.ndarray, at: int, freq_hz: float) -> float:
    """The share of the band's power in the header tones, over the span
    before a sync phase starting at baseband sample ``at``."""
    lo, hi = at - int(HEADER_SPAN[0] * FS), at - int(HEADER_SPAN[1] * FS)
    if lo < 0:
        return 0.0
    seg = bb[lo:hi] * np.hanning(hi - lo)
    S = np.abs(np.fft.fft(seg)) ** 2
    f = np.fft.fftfreq(seg.size, 1 / FS)
    tones = sum(S[np.abs(f - (t + freq_hz)) < 12].sum() for t in HEADER_TONES)
    band = S[np.abs(f) < 1500].sum()
    return float(tones / band) if band > 0 else 0.0


def find(bb: np.ndarray, start: int = 0, threshold: float = THRESHOLD) -> phy.Hit | None:
    """The next sync phase at or after baseband sample ``start`` with the
    tone header before it."""
    at = start
    while True:
        hit = phy.find(bb, SYNC, chunk=20, threshold=threshold, start=at)
        if hit is None:
            return None
        if header_share(bb, hit.sample, hit.freq_hz) >= HEADER_SHARE:
            return hit
        at = hit.sample + SYNC.size * SPS


def marker_errors(payload: np.ndarray, k: int) -> int | None:
    """The fewest of the six markers' bits wrong, at any one row of the last
    block, reading ``payload`` as ``k`` blocks; None if it is too short."""
    if payload.size < k * BLOCK_BITS * 0.97:
        return None
    b = np.zeros(k * BLOCK_BITS, dtype=np.uint8)
    m = min(payload.size, b.size)
    b[:m] = payload[:m]
    last = b.reshape(-1, COLUMNS).T[:, -(BLOCK_BITS // COLUMNS):]     # (6, 1072)
    errs = 0
    for c in range(COLUMNS):
        w = np.lib.stride_tricks.sliding_window_view(last[c], 32)
        errs = errs + np.sum(w != _MARKER_BITS[c], axis=1)
    return int(np.min(errs))


def block_count(payload: np.ndarray) -> int:
    """How many blocks the burst is, from its column markers; 0 if no count
    shows them."""
    best, best_k = None, 0
    for k in range(1, MAX_BLOCKS + 1):
        e = marker_errors(payload, k)
        if e is None:
            break
        if e <= MARKER_ERRORS and (best is None or e < best):
            best, best_k = e, k
    return best_k


def _end_marker(bits: np.ndarray) -> int:
    """Where the end marker starts in the data bits, or -1."""
    b = np.r_[0, bits.astype(np.int64), 0]
    d = np.diff(b)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    long = np.flatnonzero(ends - starts >= END_RUN)
    return int(starts[long[0]]) if long.size else -1


def receive(audio: np.ndarray, *, start: int = 0, threshold: float = THRESHOLD,
            invert: bool = False) -> Reception | None:
    """The first burst's data phase at or after sample ``start`` of ``audio``
    (real, at ``spec.FS``), to the channel bits. ``invert`` for a recording
    made on the other sideband."""
    bb = phy.downconvert(audio)
    if invert:
        bb = np.conj(bb)
    hit = find(bb, start, threshold)
    if hit is None:
        return None
    n_frames = max(0, (bb.size - hit.sample) // SPS - SYNC.size) // FRAME
    # Enough for the longest burst the markers are looked for in, doubled.
    n_frames = min(n_frames, 2 * MAX_BLOCKS * BLOCK_BITS // UNKNOWN + 40)
    sel = phy.Selector(UNKNOWN, KNOWN, SYNC.size)
    p = plan(n_frames)
    sel.start(hit.sample, hit.freq_hz, hit.gain, p.constellation)
    sel.advance(bb, 0, p, stop=loss_stop(SYNC.size))
    # A finite recording has no future probes to wait for.
    sel._final = True
    soft_all = np.real(np.asarray(sel.soft))
    bits_all = (soft_all < 0).astype(np.uint8)
    head = bits_all[: min(bits_all.size, 6000) // 2 * 2].reshape(-1, 2)
    repeat = 2 if head.size and float(np.mean(head[:, 0] == head[:, 1])) > 0.9 else 1
    if repeat == 2:
        s = soft_all[: soft_all.size // 2 * 2].reshape(-1, 2).sum(axis=1)
        payload_all = (s < 0).astype(np.uint8)
    else:
        payload_all = bits_all
    blocks = block_count(payload_all)
    if blocks:
        # The framing says where the burst ends.
        n = blocks * BLOCK_BITS
        payload = payload_all[:n]
        bits = bits_all[: n * repeat]
        got = -(-bits.size // UNKNOWN)
        end = _end_marker(bits_all[bits.size: bits.size + 2 * END_RUN])
        marker = 0 <= end < 40                  # right after, give or take an error
    else:
        # No markers: the end marker, or the signal going.
        got = soft_all.size // UNKNOWN
        if sel.stop_k is not None:
            got = min(got, max(0, (sel.stop_k - SYNC.size) // FRAME))
        bits = bits_all[: got * UNKNOWN]
        end = _end_marker(bits)
        marker = end >= 0
        if marker:
            bits = bits[:end]
            got = -(-end // UNKNOWN)
        payload = payload_all[: bits.size // repeat]
    snr = float("nan")
    k = np.asarray(sel.kerr[: SYNC.size + got * KNOWN])
    if k.size:
        snr = 10 * np.log10(1.0 / max(float(np.median(k)), 1e-6))
    stop = hit.sample + SPS * (SYNC.size + got * FRAME)
    return Reception(bits, repeat, payload, got, marker, hit.sample, stop,
                     hit.freq_hz, snr, blocks)


def synthesize(bits: np.ndarray, *, repeat: int = 1, end_marker: bool = True,
               seven: bool = False, seed: int = 0, trim: bool = True) -> np.ndarray:
    """A burst carrying ``bits`` as this receiver reads them, for testing it:
    the tone header, a stand-in for the parallel section (random QPSK on a
    75 Hz grid at 75 Bd, which nothing here reads), the sync phase, the data
    and the end marker."""
    rng = np.random.default_rng(seed)
    b = np.repeat(np.asarray(bits, dtype=np.int64), repeat)
    if end_marker:
        b = np.r_[b, np.ones(500, dtype=np.int64)]
    n_frames = -(-b.size // UNKNOWN)
    pad = np.zeros(n_frames * UNKNOWN, dtype=np.int64)
    pad[: b.size] = b
    p = plan(n_frames)
    sym = np.array(p.value, copy=True)
    data = np.asarray(p.kind) == phy.DATA
    sym[data] = (sym[data] + 4 * pad) % 8
    serial = phy.modulate(sym, trim)
    t = np.arange(int(1.6 * FS)) / FS
    tones = (450.0 * np.arange(1, 8)) if seven else (900.0, 1350.0, 1800.0, 2250.0)
    header = sum(np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in tones)
    n_par = int(3.8 * FS)
    t = np.arange(n_par) / FS
    sym_idx = (t * 75).astype(int)
    par = np.zeros(n_par)
    for f in 600.0 + 75.0 * np.arange(18) * 2:
        ph = rng.integers(0, 4, sym_idx[-1] + 1) * np.pi / 2
        par += np.cos(2 * np.pi * f * t + ph[sym_idx])
    parts = [header / np.abs(header).max(), par / np.abs(par).max() * 0.7, serial]
    return np.concatenate(parts)
