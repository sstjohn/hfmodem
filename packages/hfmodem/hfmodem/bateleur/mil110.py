# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""MIL-STD-188-110 serial (single-tone), fixed frequency: 75 to 4800 bit/s.

Written to MIL-STD-188-110B 5.3.2, which carries 110A's serial waveform
unchanged, with 110A's own text consulted where the two differ (below).

A transmission is a preamble — three 0.2 s segments, or twenty-four ahead of
the long interleaver — and then frames of data symbols followed by known probe
symbols, all randomised by the 160-symbol data sequence, until the message,
its end-of-message pattern and the coder flush have filled a whole number of
interleaver blocks. The preamble names the rate and interleave (D1, D2) and
counts down to the data (C1-C3). At 75 bit/s there are no probes: each coded
dibit is sent as one of four 32-symbol Walsh sets.

**Zero interleave** carries the short interleave's D1/D2 (5.3.2.3.7.2.1): the
standard leaves the choice to be coordinated with the far end, so a receiver
has to be told, and ``receive`` takes ``zero_interleave``.

**Count reading, unverified against a foreign implementation.** The transmitter
encodes each two-bit count field as 4 + field, sending (4, 4, 6), (4, 4, 5),
(4, 4, 4) for the short countdown. The receiver also accepts the alternate
01/10 row order if successive segments count down. This is an implemented
reading, not a confirmed erratum. A foreign short and long preamble with
known mode and count would settle it.

Message bits are the bytes least-significant bit first by default, the order an
asynchronous serial port puts them on the wire; the standard carries a bit
stream and says nothing about bytes. The two probe periods before each
interleaver block carry D1 and D2 (5.3.2.3.7.1.2); this receiver decides them
as BPSK rather than training on them, so it does not depend on their content.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from . import fec, interleave, phy
from .spec import (
    find_eom as find_eom,
    CHANNEL_PATTERNS, EOM110_BITS, FLUSH110_BITS, MODES110, MODES110_BY_D,
    PREAMBLE_BLOCK_SYMBOLS, PREAMBLE_SCRAMBLE, PREAMBLE_SYNC, SCRAMBLE110_PERIOD,
    SPS, Mode110, gray110, scramble110, zero_of,
)

_SCR = scramble110()
_PATTERNS = np.array(CHANNEL_PATTERNS, dtype=np.int64)
#: Table XIII as numbers: the two input bits (first bit high) to the
#: modified-Gray value that picks a 75 bit/s Walsh set.
_MGD4 = (0, 1, 3, 2)


def bytes_to_bits(data: bytes, order: str = "lsb") -> np.ndarray:
    a = np.frombuffer(bytes(data), dtype=np.uint8)
    return np.unpackbits(a[:, None], axis=1,
                         bitorder="little" if order == "lsb" else "big").ravel()


def bits_to_bytes(bits: np.ndarray, order: str = "lsb") -> bytes:
    b = np.asarray(bits, dtype=np.uint8)
    b = b[: b.size - b.size % 8]
    return np.packbits(b.reshape(-1, 8), axis=1,
                       bitorder="little" if order == "lsb" else "big").ravel().tobytes()


# --------------------------------------------------------------------------
# Preamble
# --------------------------------------------------------------------------

def _expand(channel_symbols) -> np.ndarray:
    """Channel symbols -> 32 8-PSK indices each, preamble randomiser added."""
    out = np.concatenate([CHANNEL_PATTERNS[t] for t in channel_symbols])
    scr = np.tile(PREAMBLE_SCRAMBLE, len(channel_symbols))
    return (out + scr) % 8


def count_symbols(count: int, table_b: bool = False) -> tuple[int, int, int]:
    """C1-C3: 4 + each two-bit field; ``table_b`` swaps the 01/10 readings."""
    f = ((count >> 4) & 3, (count >> 2) & 3, count & 3)
    if table_b:
        f = tuple((v >> 1) | ((v & 1) << 1) for v in f)
    return tuple(4 + v for v in f)


def preamble_block(mode: Mode110, count: int) -> np.ndarray:
    return _expand(PREAMBLE_SYNC + (mode.d1, mode.d2) + count_symbols(count) + (0,))


def preamble(mode: Mode110) -> np.ndarray:
    n = mode.preamble_blocks
    return np.concatenate([preamble_block(mode, n - 1 - b) for b in range(n)])


SYNC_REF = _expand(PREAMBLE_SYNC)            # what acquisition looks for


# --------------------------------------------------------------------------
# Data phase
# --------------------------------------------------------------------------

def coded_bits_per_frame(mode: Mode110) -> int:
    """Channel bits a frame carries: one Walsh set holds a dibit."""
    return 2 if mode.walsh else mode.bits_per_symbol * mode.unknown


def _granule(mode: Mode110) -> tuple[int, int]:
    """(message bits, frames) a transmission is padded to a multiple of: an
    interleaver block, or — with none — the fewest whole frames the coded
    bits can fill."""
    per_frame = coded_bits_per_frame(mode)
    if not mode.coded:
        return per_frame, 1
    if mode.rows:
        return mode.block_bits // (2 * mode.repeat), mode.frames_per_block
    step = 2 * mode.repeat
    info = per_frame // gcd(per_frame, step)
    return info, info * step // per_frame


def _flush(mode: Mode110) -> int:
    """5.3.2.3.2: 144 coder-flush zeros, and then whatever completes the
    block; with no interleaver, only the 144 (none when there is no coder)."""
    return FLUSH110_BITS if mode.coded else 0


def _padded_bits(mode: Mode110, n_bits: int) -> int:
    info, _ = _granule(mode)
    n = n_bits + len(EOM110_BITS) + _flush(mode)
    return n + (-n) % info


def data_symbol_count(mode: Mode110, n_bytes: int) -> int:
    """How many symbols ``data_symbols`` makes of an ``n_bytes`` message."""
    return _symbols_for_bits(mode, 8 * n_bytes)



def symbol_count(mode: Mode110, n_bytes: int) -> int:
    return mode.preamble_blocks * PREAMBLE_BLOCK_SYMBOLS + data_symbol_count(mode, n_bytes)


def _symbols_for_bits(mode: Mode110, n_bits: int) -> int:
    info, frames = _granule(mode)
    units = _padded_bits(mode, n_bits) // info
    return units * frames * (mode.unknown + mode.known)


def _probe(mode: Mode110, frame: int) -> np.ndarray:
    """Pre-randomiser probe symbols of data frame ``frame``: zeros, but D1
    and D2 in the two probes before each new block (5.3.2.3.7.1.2)."""
    fpb = mode.frames_per_block
    at = frame % fpb
    p = np.zeros(mode.known, dtype=np.int64)
    if at >= fpb - 2:
        pat = CHANNEL_PATTERNS[mode.d1 if at == fpb - 2 else mode.d2]
        p[:16] = pat[:16]
    return p


def _message_bits(mode: Mode110, data: bytes, order: str = "lsb") -> np.ndarray:
    bits = np.concatenate([bytes_to_bits(data, order), EOM110_BITS]).astype(np.uint8)
    n = _padded_bits(mode, 8 * len(data))
    return np.concatenate([bits, np.zeros(n - bits.size, dtype=np.uint8)])


def _is_exceptional(mode: Mode110, set_index: int) -> bool:
    """75 bit/s: every 45th set (short or zero) or 360th (long) after the sync
    uses table XIVb's sets (5.3.2.3.7.1.1)."""
    return (set_index + 1) % mode.frames_per_block == 0


def data_symbols(mode: Mode110, data: bytes, order: str = "lsb") -> np.ndarray:
    """The whole data phase as randomised 8-PSK indices."""
    bits = _message_bits(mode, data, order)
    if mode.coded:
        coded, _ = fec.encode(bits)
        if mode.repeat > 1:
            coded = np.repeat(coded.reshape(-1, 2), mode.repeat, axis=0).ravel()
        if mode.rows:
            coded = interleave.interleave110(coded, mode.rows, mode.cols,
                                             mode.row_step, mode.col_step)
        chan = coded
    else:
        chan = bits
    if mode.walsh:
        dibits = chan.reshape(-1, 2) @ np.array([2, 1])
        mgd = np.asarray(_MGD4)[dibits]
        sets = [_PATTERNS[v + (4 if _is_exceptional(mode, s) else 0)]
                for s, v in enumerate(mgd)]
        sym = np.concatenate(sets)
    else:
        nb = mode.bits_per_symbol
        vals = chan.reshape(-1, nb) @ (1 << np.arange(nb - 1, -1, -1))
        unknown = np.asarray(gray110(nb))[vals].reshape(-1, mode.unknown)
        sym = np.concatenate([np.concatenate([u, _probe(mode, f)
                              if f < len(unknown) - 2 else np.zeros(mode.known, dtype=int)])
                              for f, u in enumerate(unknown)])
    return (sym + np.resize(_SCR, sym.size)) % 8


def transmit(data: bytes, mode: Mode110 | str = "2400S", *, order: str = "lsb",
             trim: bool = True) -> np.ndarray:
    """Message -> real audio at ``spec.FS``, peak 1.0. ``trim`` as `phy.modulate`."""
    if isinstance(mode, str):
        mode = MODES110[mode]
    return phy.modulate(np.concatenate([preamble(mode), data_symbols(mode, data, order)]),
                        trim)


# --------------------------------------------------------------------------
# Receive: the parts both receivers share
# --------------------------------------------------------------------------

def walsh_llr(corr: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """75 bit/s: per set, the four set correlations (indexed by modified-Gray
    value) -> LLRs for its two channel bits, first bit first, positive for 0."""
    c = np.asarray(corr, dtype=np.float64).reshape(-1, 4)
    s2 = np.maximum(np.asarray(sigma2, dtype=np.float64).reshape(-1), 1e-3)
    by_input = c[:, list(_MGD4)]                     # column u = input bits u
    first = np.max(by_input[:, [0, 1]], 1) - np.max(by_input[:, [2, 3]], 1)
    second = np.max(by_input[:, [0, 2]], 1) - np.max(by_input[:, [1, 3]], 1)
    return (np.stack([first, second], axis=1) * (2.0 / s2)[:, None]).ravel()


class Decoder:
    """Channel-bit LLRs in, message bits out, a piece at a time."""

    def __init__(self, mode: Mode110) -> None:
        self.mode = mode
        self._llr = np.zeros(0)
        self._vit = fec.ViterbiStream() if mode.coded else None

    def push(self, llr) -> np.ndarray:
        m = self.mode
        self._llr = np.concatenate([self._llr, np.asarray(llr, dtype=np.float64)])
        if not m.coded:
            out = (self._llr < 0).astype(np.uint8)
            self._llr = np.zeros(0)
            return out
        unit = m.block_bits if m.rows else 2 * m.repeat
        n = self._llr.size - self._llr.size % unit
        if n == 0:
            return np.zeros(0, dtype=np.uint8)
        soft, self._llr = self._llr[:n], self._llr[n:]
        if m.rows:
            soft = interleave.deinterleave110(soft, m.rows, m.cols, m.row_step, m.col_step)
        if m.repeat > 1:
            soft = soft.reshape(-1, m.repeat, 2).sum(axis=1).ravel()
        return self._vit.push(soft)

    def finish(self) -> np.ndarray:
        return self._vit.finish() if self._vit is not None else np.zeros(0, np.uint8)




def data_symbols_to_end(mode: Mode110, eom_at: int) -> int:
    """Data-phase symbols of a transmission whose EOM starts at message bit
    ``eom_at``: the block (or frames) the EOM and flush land in, whole."""
    return _symbols_for_bits(mode, eom_at)


# --------------------------------------------------------------------------
# Receive: the preamble
# --------------------------------------------------------------------------

@dataclass
class Reception:
    mode: Mode110
    data: bytes
    eom: bool                   # the end-of-message marker was found
    start_sample: int           # at ``spec.FS``, first preamble symbol found
    end_sample: int
    freq_hz: float
    snr_db: float
    detail: str = ""


#: Preamble segments read together: every one repeats the mode, and the
#: count drops by one from each to the next, so three are three looks at
#: the same answer. A short preamble has exactly three.
READ_BLOCKS = 3
#: Lags either side of the hit searched for the channel's paths.
_READ_REACH = 16 * SPS

#: Baseband a hit needs behind it before ``read_preamble`` can answer.
READ_SAMPLES = READ_BLOCKS * PREAMBLE_BLOCK_SYMBOLS * SPS + _READ_REACH + SPS

_PATTERN_PTS = phy.psk8(np.array(CHANNEL_PATTERNS))              # (8, 32)
_DESCRAMBLE = phy.psk8(-np.asarray(PREAMBLE_SCRAMBLE))


def _paths(bb, at, n_blocks, freq_hz) -> np.ndarray:
    """The lags, around ``at``, at which the segments' sync symbols arrive:
    the strongest dozen within a third of the strongest's energy, summed
    over the segments and taken 32 symbols at a time, so a fade or a fast
    channel does not hide a path."""
    ref = phy.psk8(SYNC_REF)
    lags = np.arange(-_READ_REACH, _READ_REACH + 1)
    energy = np.zeros(lags.size)
    for b in range(n_blocks):
        base = at + b * PREAMBLE_BLOCK_SYMBOLS * SPS
        for c in range(len(PREAMBLE_SYNC)):
            n = base + (32 * c + np.arange(32)) * SPS
            if n[0] + lags[0] < 0 or n[-1] + lags[-1] >= bb.size:
                continue
            idx = n[None, :] + lags[:, None]
            x = bb[idx] * np.exp(-2j * np.pi * freq_hz * idx / phy.FS)
            energy += np.abs(x @ np.conj(ref[32 * c: 32 * c + 32])) ** 2
    if not np.any(energy):
        return np.array([0])
    strong = np.flatnonzero(energy >= energy.max() / 3)
    return lags[strong[np.argsort(-energy[strong])][:12]]


def _pattern_energy(bb, at, freq_hz, lags) -> np.ndarray:
    """How much of each of the eight channel patterns is at ``at``, summed
    over the paths. On the size of the correlations, not their phase: the
    phase at the sync is stale by the D and C symbols, and on a 5 Hz channel
    that is most of a cycle. No pattern is another's negation, so size alone
    tells all eight apart."""
    n = at + np.arange(32) * SPS
    idx = n[None, :] + lags[:, None]
    if idx.min() < 0 or idx.max() >= bb.size:
        return np.zeros(8)
    x = bb[idx] * np.exp(-2j * np.pi * freq_hz * idx / phy.FS) * _DESCRAMBLE
    return np.sum(np.abs(x @ np.conj(_PATTERN_PTS).T) ** 2, axis=0)


def read_preamble(bb, hit: phy.Hit, *, zero_interleave: bool = False):
    """(mode, count) from the segment whose sync starts at ``hit``, or None.

    Every reading is scored at once against up to ``READ_BLOCKS`` segments:
    the mode, the count, and which of the two C1-C3 conventions the far end
    uses (they differ for some counts; the one that counts down across the
    segments is the one in use). A reading is scored on each channel symbol
    by how far its pattern stands above the eight patterns' mean, so a
    segment that is not preamble — a count that ran out — adds nothing
    either way. Where the conventions still tie, the text's reading stands.
    """
    at = hit.sample
    step = PREAMBLE_BLOCK_SYMBOLS * SPS
    n_blocks = 0
    while n_blocks < READ_BLOCKS and at + (n_blocks + 1) * step + _READ_REACH <= bb.size:
        n_blocks += 1
    if n_blocks == 0:
        return None
    lags = _paths(bb, at, n_blocks, hit.freq_hz)
    # score[b, i, t]: channel symbol i of segment b, pattern t, above the mean.
    score = np.zeros((n_blocks, 15, 8))
    for b in range(n_blocks):
        for i in range(15):
            e = _pattern_energy(bb, at + b * step + 32 * SPS * i, hit.freq_hz, lags)
            score[b, i] = e - e.mean()
    sync = [sum(score[b, i, t] for i, t in enumerate(PREAMBLE_SYNC)) + score[b, 14, 0]
            for b in range(n_blocks)]
    d_score = score[:, 9, :, None] + score[:, 10, None, :]           # (b, d1, d2)
    best, answer = -np.inf, None
    # The mode first, on every segment: it is the same in all of them.
    d_total = d_score.sum(axis=0)
    d1, d2 = np.unravel_index(int(np.argmax(d_total)), d_total.shape)
    mode = MODES110_BY_D.get((int(d1), int(d2)))
    if mode is None:
        return None             # a D1/D2 not implemented: keep searching (5.3.2.3.7.2.1)
    for table_b in (False, True):
        for count in range(mode.preamble_blocks):
            total = 0.0
            for b in range(min(n_blocks, count + 1)):
                cs = count_symbols(count - b, table_b)
                total += sync[b] + d_score[b, d1, d2] \
                    + sum(score[b, 11 + i, c] for i, c in enumerate(cs))
            if total > best + 1e-9:
                best, answer = total, count
    if answer is None or best <= 0:
        return None
    if zero_interleave:
        mode = zero_of(mode)
    return mode, answer


def preamble_plan(mode: Mode110, count: int) -> phy.Plan:
    """The rest of the preamble, from the segment counting ``count`` down."""
    pre = np.concatenate([preamble_block(mode, c) for c in range(count, -1, -1)])
    return phy.Plan(np.full(pre.size, phy.KNOWN), pre, np.zeros(pre.size, dtype=np.int64),
                    tuple(gray110(mode.bits_per_symbol)), _PATTERNS)


def frames_plan(mode: Mode110, first: int, n_frames: int) -> phy.Plan:
    """Data frames ``first`` to ``first + n_frames``, counted from the data
    phase's start, which is where the randomiser's period is counted from."""
    fl = mode.unknown + mode.known
    fpb = mode.frames_per_block
    n = n_frames * fl
    scr = _SCR[(first * fl + np.arange(n)) % SCRAMBLE110_PERIOD]
    kind = np.empty(n, dtype=np.int64)
    val = np.zeros(n, dtype=np.int64)
    for i in range(n_frames):
        a = i * fl
        if mode.walsh:
            kind[a:a + fl] = phy.SET
            val[a:a + fl] = 4 if _is_exceptional(mode, first + i) else 0
            continue
        kind[a:a + mode.unknown] = phy.DATA
        at = (first + i) % fpb
        kind[a + mode.unknown:a + fl] = phy.PROBE_BPSK if at >= fpb - 2 else phy.KNOWN
        val[a + mode.unknown:a + fl] = scr[a + mode.unknown:a + fl]
    return phy.Plan(kind, val, scr, tuple(gray110(mode.bits_per_symbol)), _PATTERNS)


def _plan(mode: Mode110, count: int, n_frames: int) -> phy.Plan:
    """The rest of the preamble from the found segment, then ``n_frames``."""
    return phy.Plan.concat([preamble_plan(mode, count), frames_plan(mode, 0, n_frames)])


# --------------------------------------------------------------------------
# Receive: a whole recording
# --------------------------------------------------------------------------

def receive(audio: np.ndarray, *, start: int = 0, threshold: float = 0.35,
            zero_interleave: bool = False, order: str = "lsb") -> Reception | None:
    """Find the first 188-110 transmission in ``audio`` (real, at ``spec.FS``)
    at or after sample ``start`` and decode it."""
    from .stream import Receiver110, receive_whole
    got = receive_whole(audio, Receiver110(bit_order=order, threshold=threshold,
                        zero_interleave=zero_interleave), start=start)
    if got is None:
        return None
    first, data, end = got
    return Reception(MODES110[first.mode], data, end.complete, first.at, end.at,
                     first.freq_hz, end.snr_db)


__all__ = ["transmit", "receive", "Reception", "MODES110", "SCRAMBLE110_PERIOD"]
