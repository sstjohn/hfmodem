# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Incremental receivers for 110 serial, 4285, 4539, 39-tone and 16-tone.

Acquisition, equalization and decoding retain state between chunks. Events
report acquisition, decoded bytes and the reason reception ended. Coded
110/4285 hold the EOM and its confirming flush before releasing bytes.
The waveform modules' batch entry points feed these same receivers through
``receive_whole``; offline 39-tone can also correct gross clock error first.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import firwin, lfilter

from . import fec, fsk, interleave, mil110, phy, s4285, s4529, s4539, tone16, tone39, wbhf
from .spec import (
    CARRIER_HZ, DATA4285_PER_FRAME, EOM110_BITS, FRAME4285, FS,
    MODES4285, SPS, Mode110, Mode4285, gray110,
    gray4285,
)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Started:
    standard: str               # "110" | "4285" | "4539" | "39" | "16"
    mode: str                   # the mode's name, e.g. "600L"
    freq_hz: float              # carrier offset found
    at: int                     # baseband sample of the first symbol found


@dataclass(frozen=True)
class Data:
    data: bytes


@dataclass(frozen=True)
class Ended:
    complete: bool              # EOM received; absent in uncoded 4285, 16-tone and synchronous 39-tone
    snr_db: float
    why: str
    at: int | None = None


# --------------------------------------------------------------------------
# Front end
# --------------------------------------------------------------------------

class FrontEnd:
    """Card-rate audio in, matched-filtered baseband at four samples a symbol
    out, with every filter's state carried across calls — so chunk edges are
    invisible, which a chunk-wise resample would not be."""

    def __init__(self, rate: int) -> None:
        if rate % FS:
            raise ValueError(f"the card rate {rate} is not a multiple of {FS}")
        self.decim = rate // FS
        self._aa = firwin(20 * self.decim + 1, 0.42 * FS, fs=rate)
        self._aa_zi = np.zeros(self._aa.size - 1)
        self._phase = 0                          # card samples until the next kept one
        self._n = 0                              # baseband samples produced
        self._mf = phy._PULSE
        self._mf_zi = np.zeros(self._mf.size - 1, dtype=np.complex128)
        period = FS // np.gcd(FS, CARRIER_HZ)
        self._lo = np.exp(-2j * np.pi * CARRIER_HZ * np.arange(period) / FS)

    def push(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64)
        if not x.size:
            self.audio = x
            return np.zeros(0, dtype=np.complex128)
        if self.decim > 1:
            y, self._aa_zi = lfilter(self._aa, 1.0, x, zi=self._aa_zi)
            keep = y[self._phase::self.decim]
            self._phase = (self._phase - x.size) % self.decim
        else:
            keep = x
        #: The same audio at FS, real, before the matched filter: what the
        #: 39-tone receiver reads.
        self.audio = keep
        if not keep.size:
            return np.zeros(0, dtype=np.complex128)
        lo = self._lo[(self._n + np.arange(keep.size)) % self._lo.size]
        self._n += keep.size
        bb, self._mf_zi = lfilter(self._mf, 1.0, keep * lo, zi=self._mf_zi)
        return bb * np.sqrt(2)


# --------------------------------------------------------------------------
# Bits to bytes
# --------------------------------------------------------------------------

class Bytes:
    """Bits in, whole bytes out, in either bit order."""

    def __init__(self, order: str = "lsb") -> None:
        if order not in ("lsb", "msb"):
            raise ValueError(f"bit order is lsb or msb, not {order!r}")
        self.order = "little" if order == "lsb" else "big"
        self._held = np.zeros(0, dtype=np.uint8)

    def push(self, bits) -> bytes:
        b = np.concatenate([self._held, np.asarray(bits, dtype=np.uint8)])
        n = b.size - b.size % 8
        self._held = b[n:]
        if not n:
            return b""
        return np.packbits(b[:n].reshape(-1, 8), axis=1,
                           bitorder=self.order).ravel().tobytes()


# --------------------------------------------------------------------------
# The common part: a baseband buffer, acquisition, the equalizer
# --------------------------------------------------------------------------

_SCAN_EVERY = 2400 * SPS // 2                  # half a second of baseband


class _Buffer:
    def __init__(self, dtype=float):
        self.buf = np.zeros(0, dtype=dtype)
        self.base = self.search_from = self._scanned_to = 0

    @property
    def end(self):
        return self.base + self.buf.size

    def push(self, samples) -> list:
        samples = np.asarray(samples, dtype=self.buf.dtype)
        events = []
        for begin in range(0, samples.size, FS):
            self.buf = np.concatenate([self.buf, samples[begin:begin + FS]])
            while True:
                if not self.receiving:
                    if not self._acquire(events):
                        break
                elif not self._receive(events):
                    break
            self._trim()
        return events


class _Receiver(_Buffer):
    """Holds the baseband since the oldest sample anything still needs."""

    ref: np.ndarray                                     # what acquisition looks for
    #: Samples past a hit that must have arrived before it is trusted: the
    #: refinement window, and whatever the standard reads to describe itself.
    after_hit: int

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold
        super().__init__(complex)
        self.eq: phy.Selector | None = None
        self.plan: phy.Plan | None = None

    @property
    def receiving(self) -> bool:
        return self.eq is not None

    def flush(self) -> list:
        events = []
        if self.eq is None:
            return events
        if isinstance(self, Receiver110):
            if self._decoded(self._dec.finish(), events):
                return events
        elif isinstance(self, Receiver4285):
            if self._chains is not None:
                c = self._chains[0]
                self._chains = None
                self._vit = c.vit
                self._decoded_n = 0
                self._bits = c.bits
                self._msg_at = 0
            if self._vit is not None and self._place(self._vit.finish(), events):
                return events
        self._release(self._bits, events)
        events.append(Ended(False, self._snr_db(), "recording ended before EOM"))
        self._finish(events)
        return events

    # -- acquisition -------------------------------------------------------

    def _acquire(self, events) -> bool:
        run = len(self.ref) * SPS
        if self.end - self._scanned_to < _SCAN_EVERY and self._scanned_to:
            return False
        if self.end - self.search_from < run + self.after_hit:
            return False
        self._scanned_to = self.end
        hit = phy.find(self.buf, self.ref, threshold=self.threshold,
                       start=self.search_from - self.base)
        if hit is None:
            # Keep enough behind to find a run that straddles this scan's end.
            self.search_from = max(self.search_from, self.end - run - self.after_hit)
            return False
        at = hit.sample + self.base
        if at + self.after_hit > self.end:
            self.search_from = at - SPS                 # look again when it is whole
            self._scanned_to = 0
            return False
        if not self._begin(hit, events):
            self.search_from = at + len(self.ref) * SPS // 2
            self._scanned_to = 0
            return True
        return True

    def _begin(self, hit: phy.Hit, events) -> bool:
        raise NotImplementedError

    # -- receiving ---------------------------------------------------------

    def _receive(self, events) -> bool:
        raise NotImplementedError

    def _finish(self, events, end: int | None = None) -> None:
        """Back to listening from ``end``, where the transmission just decoded
        stopped, or from where the equalizer gave up on it."""
        self.last_end = self.eq.pos if end is None else end
        events[-1] = replace(events[-1], at=self.last_end)
        self.search_from = max(self.search_from, self.last_end)
        self._scanned_to = 0
        self.eq = None
        self.plan = None

    def _trim(self) -> None:
        if self.eq is not None:
            eq = self.eq
            self.plan.discard_before(max(0, min(eq.k, eq.b.k if eq.b else eq.k) - 8192))
            eq.discard(self._taken if not getattr(self.mode, "walsh", False) else 0,
                       getattr(self, "_kerr_taken", len(eq.kerr)),
                       self._taken if getattr(self.mode, "walsh", False) else 0)
            keep_from = self.eq.pos - (self.eq.n_pre + 8) * SPS
        else:
            keep_from = self.search_from
        # Listening resumes where a decoded transmission ended, which can be
        # ahead of what has arrived: its last frames are still on the way
        # when the interleaver delivers its EOM. Never cut past the buffer.
        cut = min(keep_from - self.base, self.buf.size)
        if cut > 4 * _SCAN_EVERY:
            self.buf = self.buf[cut:]
            self.base += cut

    def _release(self, bits, events) -> None:
        self._sent += len(bits)
        out = self._bytes.push(bits)
        if out:
            events.append(Data(out))

    def _hold(self, events):
        safe = max(0, self._bits.size - self._HOLD_BITS)
        safe -= safe % 8
        if safe:
            self._release(self._bits[:safe], events)
            self._bits = self._bits[safe:]

    def _snr_db(self) -> float:
        return phy.snr_db(self.eq.kerr)


# --------------------------------------------------------------------------
# MIL-STD-188-110
# --------------------------------------------------------------------------

class Receiver110(_Receiver):
    """Listens for any 188-110 serial mode; the preamble says which.

    ``zero_interleave`` is the coordination the standard asks for: a short
    preamble then stands for the same rate with its interleaver bypassed.
    """

    ref = mil110.SYNC_REF
    after_hit = mil110.READ_SAMPLES + 16 * SPS

    #: Bits held back until they cannot be the start of the EOM.
    _HOLD_BITS = len(EOM110_BITS) + 64 + 8

    def __init__(self, *, bit_order: str = "lsb", zero_interleave: bool = False,
                 **kw) -> None:
        super().__init__(**kw)
        self.bit_order = bit_order
        self.zero_interleave = zero_interleave

    def _begin(self, hit, events) -> bool:
        got = mil110.read_preamble(self.buf, hit, zero_interleave=self.zero_interleave)
        if got is None:
            return False
        mode, count = got
        self.mode: Mode110 = mode
        self.fl = mode.unknown + mode.known
        self.frames = 0
        self.plan = mil110.preamble_plan(mode, count)
        self._n_pre = len(self.plan)
        self._data_at = hit.sample + self.base + self._n_pre * SPS
        self._extend()
        self.eq = phy.Selector(mode.unknown, mode.known, self._n_pre)
        self.eq.start(hit.sample + self.base, hit.freq_hz, hit.gain, self.plan.constellation)
        self._stop = phy.loss_detector(self._n_pre + 4 * self.fl,
                                       0.007 if mode.walsh else 0.0007,
                                       1.4 if mode.walsh else 0.8)
        self._taken = 0
        self._dec = mil110.Decoder(mode)
        self._bits = np.zeros(0, dtype=np.uint8)    # decoded, not yet released
        self._sent = 0                         # released, from message start
        self._bytes = Bytes(self.bit_order)
        self._eom_from = 0
        events.append(Started("110", mode.name, hit.freq_hz, hit.sample + self.base))
        return True

    def _extend(self) -> None:
        more = mil110.frames_plan(self.mode, self.frames, 200)
        self.frames += 200
        self.plan = phy.Plan.concat([self.plan, more])

    def _receive(self, events) -> bool:
        eq = self.eq
        if len(self.plan) - eq.k < 50 * self.fl:
            self._extend()
        eq.advance(self.buf, self.base, self.plan, stop=self._stop)
        m = self.mode
        # Everything the equalizer has decided since last time, as LLRs.
        if m.walsh:
            n = len(eq.walsh)
            llr = mil110.walsh_llr(np.array(eq.walsh[self._taken:n]),
                                   np.array(eq.walsh_sig[self._taken:n])) \
                if n > self._taken else np.zeros(0)
        else:
            n = len(eq.soft)
            llr = phy.demap(np.array(eq.soft[self._taken:n]),
                            np.array(eq.sig[self._taken:n]), gray110(m.bits_per_symbol)) \
                if n > self._taken else np.zeros(0)
        self._taken = n
        if llr.size and self._decoded(self._dec.push(llr), events):
            return True
        if eq.stopped:
            if self._decoded(self._dec.finish(), events):
                return True
            self._release(self._bits, events)
            events.append(Ended(False, self._snr_db(), "signal lost before EOM"))
            self._finish(events)
            return True
        return False

    def _decoded(self, bits, events) -> bool:
        """New decoded bits: look for the EOM, release what cannot be it."""
        self._bits = np.concatenate([self._bits, bits])
        at = mil110.find_eom(self._bits, self._eom_from,
                             flush=64 if self.mode.coded else 0,
                             errors=3 if self.mode.coded else 0)
        if at >= 0:
            self._release(self._bits[:at], events)
            end = self._data_at + mil110.data_symbols_to_end(self.mode, self._sent) * SPS
            events.append(Ended(True, self._snr_db(), "EOM"))
            self._finish(events, end=end - 16 * SPS)
            return True
        self._hold(events)
        self._eom_from = max(0, self._bits.size - self._HOLD_BITS)
        self._eom_from -= self._eom_from % 8
        return False





# --------------------------------------------------------------------------
# STANAG 4285
# --------------------------------------------------------------------------

class _Deinterleave4285:
    """The 32-row deinterleaver fed a frame at a time."""

    def __init__(self, increment: int) -> None:
        self.J = increment
        self._groups: list[np.ndarray] = []
        self._first = 0                        # index of _groups[0]
        self._next = 0                         # next output group

    def push(self, slots: np.ndarray) -> np.ndarray:
        g = np.asarray(slots, dtype=np.float64).reshape(-1, 32)
        self._groups.extend(g)
        out = []
        have = self._first + len(self._groups)
        pos = interleave._POS
        while self._next + 31 * self.J < have:
            h = self._next
            row = np.empty(32)
            for o in range(32):
                c = pos[o]
                row[o] = self._groups[h + self.J * c - self._first][c]
            out.append(row)
            self._next += 1
        drop = self._next - self._first
        if drop > 0:
            del self._groups[:drop]
            self._first += drop
        return np.concatenate(out) if out else np.zeros(0)


@dataclass
class _Chain:
    deint: _Deinterleave4285
    vit: fec.ViterbiStream
    pairs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    n: int = 0
    late: int = 0


class Receiver4285(_Receiver):
    """Listens for 4285 in one mode, which both ends must share.

    Coded modes are framed as Annex E frames them: nothing is released until
    the decoder has produced the start-of-message pattern, and the message
    ends at the end-of-message pattern. If no start pattern has come by the
    time the flush alone would have ended the message, the decoder's output is
    released as it stands, which is what Annex E asks of a receiver too.
    """

    ref = s4285.SYNC
    after_hit = s4285.CONFIRM_SAMPLES + 16 * SPS
    _HOLD_BITS = len(s4285.EOM_BITS) + 64 + 8

    def __init__(self, mode: Mode4285 | str, *, bit_order: str = "lsb", **kw) -> None:
        super().__init__(**kw)
        self.mode = MODES4285[mode] if isinstance(mode, str) else mode
        self.bit_order = bit_order

    def _begin(self, hit, events) -> bool:
        if not s4285.confirmed(self.buf, hit, self.threshold):
            return False
        self.frames = 0
        self.plan = s4285._plan(self.mode, 0)
        self._extend()
        self.eq = phy.Selector(DATA4285_PER_FRAME, FRAME4285 - DATA4285_PER_FRAME, 0)
        self.eq.start(hit.sample + self.base, hit.freq_hz, hit.gain, self.plan.constellation)
        self._start_at = hit.sample + self.base
        self._stop = phy.loss_detector(2 * FRAME4285, 0.0005)
        self._taken = 0
        self._kerr_taken = 0
        self._suspect: list[np.ndarray] = []
        coded = self.mode.coded
        self._deint = _Deinterleave4285(self.mode.increment) if coded else None
        self._vit = fec.ViterbiStream() if coded else None
        self._pairs = np.zeros(0)                   # soft slots awaiting a repeat group
        #: Until the SOM is found, a decoder for each number of frames the
        #: acquisition may have come late (s4285.LATE_FRAMES), each started
        #: with that many frames of erasures: the interleaver is convolutional,
        #: and a deinterleaver started a frame late decodes only noise.
        self._chains = [self._chain(k) for k in range(s4285.LATE_FRAMES + 1)] \
            if coded else None
        self._late = 0
        self._bytes = Bytes(self.bit_order)
        self._bits = np.zeros(0, dtype=np.uint8)    # decoded, not yet placed
        self._decoded_n = 0                         # decoder output so far
        self._msg_at = None if coded else 0         # decoder index of message start
        self._sent = 0                              # message bits released
        events.append(Started("4285", _name4285(self.mode), hit.freq_hz, self._start_at))
        return True

    def _extend(self) -> None:
        self.plan = phy.Plan.concat([self.plan, s4285._plan(self.mode, 200)])
        self.frames += 200

    def _receive(self, events) -> bool:
        eq = self.eq
        if len(self.plan) - eq.k < 50 * FRAME4285:
            self._extend()
        eq.advance(self.buf, self.base, self.plan, stop=self._stop)
        known = s4285._KNOWN_PER_FRAME
        while (len(eq.soft) - self._taken >= DATA4285_PER_FRAME
               and len(eq.kerr) - self._kerr_taken >= known):
            z = np.array(eq.soft[self._taken:self._taken + DATA4285_PER_FRAME])
            s2 = np.array(eq.sig[self._taken:self._taken + DATA4285_PER_FRAME])
            ke = np.array(eq.kerr[self._kerr_taken:self._kerr_taken + known])
            self._taken += DATA4285_PER_FRAME
            self._kerr_taken += known
            llr = phy.demap(z, s2, gray4285(self.mode.bits_per_symbol))
            if np.median(ke[:80]) < 0.5:
                # A good sync vouches for the frames held since the last one:
                # they were a fade inside the message, not its end.
                for held in self._suspect:
                    if self._frame(held, events):
                        return True
                self._suspect.clear()
                if self._frame(llr, events):
                    return True
            else:
                self._suspect.append(llr)
                if len(self._suspect) > 64 and self._frame(self._suspect.pop(0), events):
                    return True
        if eq.stopped:
            self._suspect.clear()
            if self._chains is not None:
                # Lost before any chain found a SOM: the first chain's
                # output, as the batch receiver hands on.
                c = self._chains[0]
                self._chains = None
                self._vit = c.vit
                self._decoded_n = c.n - c.bits.size
                self._bits = np.zeros(0, dtype=np.uint8)
                if self._place(c.bits, events):
                    return True
            if self._vit is not None:
                if self._place(self._vit.finish(), events):
                    return True
            self._release(self._bits, events)
            events.append(Ended(False, self._snr_db(), "signal lost"))
            self._finish(events)
            return True
        return False

    def _chain(self, late: int) -> _Chain:
        c = _Chain(_Deinterleave4285(self.mode.increment), fec.ViterbiStream(), late=late)
        per = DATA4285_PER_FRAME * self.mode.slots_per_symbol
        if late:
            self._decode(c, np.zeros(late * per))
        return c

    def _decode(self, c: _Chain, slots: np.ndarray) -> np.ndarray:
        """Soft slots through a chain's deinterleaver and decoder."""
        m = self.mode
        soft = np.concatenate([c.pairs, c.deint.push(slots)])
        group = 2 * m.repeat
        n = soft.size - soft.size % group
        c.pairs = soft[n:]
        soft = soft[:n]
        if m.repeat > 1:
            soft = soft.reshape(-1, m.repeat, 2).sum(axis=1).ravel()
        return c.vit.push(soft)

    def _frame(self, llr, events) -> bool:
        m = self.mode
        if not m.coded:
            return self._place((llr < 0).astype(np.uint8), events)
        if m.slots_per_symbol == 4:
            llr = np.concatenate([llr.reshape(-1, 3), np.zeros((llr.size // 3, 1))],
                                 axis=1).ravel()
        if self._chains is None:
            c = _Chain(self._deint, self._vit, self._pairs)
            bits = self._decode(c, llr)
            self._pairs = c.pairs
            return self._place(bits, events)
        # Every chain on; the first to find the SOM is the one kept, or the
        # first chain when the flush alone would have ended a message.
        for c in self._chains:
            b = self._decode(c, llr)
            c.n += len(b)
            c.bits = np.concatenate([c.bits, b])
        pick = next((c for c in self._chains if s4285.find_som(c.bits) >= 0), None)
        first = self._chains[0]
        if pick is None and first.n > m.flush_bits + 32:
            pick = first
        if pick is None:
            return False
        self._chains = None
        self._deint, self._vit, self._pairs = pick.deint, pick.vit, pick.pairs
        self._late = pick.late
        self._decoded_n = pick.n - pick.bits.size
        self._bits = np.zeros(0, dtype=np.uint8)
        return self._place(pick.bits, events)

    def _place(self, bits, events) -> bool:
        """Decoded bits in: find the frame's edges, release what is message."""
        self._bits = np.concatenate([self._bits, bits])
        self._decoded_n += len(bits)
        m = self.mode
        if self._msg_at is None:
            at = s4285.find_som(self._bits)
            if at >= 0:
                self._msg_at = self._decoded_n - self._bits.size + at
                self._bits = self._bits[at:]
            elif self._decoded_n > m.flush_bits + 32:
                self._msg_at = self._decoded_n - self._bits.size   # none: all of it
            else:
                return False
        if not m.coded:
            out = self._bits
            self._bits = np.zeros(0, dtype=np.uint8)
            self._release(out, events)
            return False
        candidate = np.r_[self._bits, self._vit.pending()]
        e = s4285.find_eom(candidate)
        if e + 32 > self._bits.size:
            e = -1
        if e >= 0:
            self._release(self._bits[:e], events)
            events.append(Ended(True, self._snr_db(), "EOM"))
            self._finish(events)
            return True
        self._hold(events)
        return False



# --------------------------------------------------------------------------
# STANAG 4539 (MIL-STD-188-110B appendix C)
# --------------------------------------------------------------------------

class Receiver4539(_Receiver):
    """Listens for the high-rate waveform at any rate; the preamble says
    which. Bytes are released a whole interleaver block at a time, which is
    as soon as the tail-biting decoder can have them, less the last few that
    might be the start of the end-of-message pattern."""

    ref = s4539.SYNC
    after_hit = 287 * SPS + 16 * SPS
    _HOLD_BITS = len(s4539.EOM_BITS) + 8

    def __init__(self, *, bit_order: str = "lsb", **kw) -> None:
        super().__init__(**kw)
        self.bit_order = bit_order

    def _acquire(self, events) -> bool:
        """The synchronisation preamble, or (late entry) a reinserted one."""
        if self.end - self._scanned_to < _SCAN_EVERY and self._scanned_to:
            return False
        span = (287 + 71 + 32) * SPS + self.after_hit
        if self.end - self.search_from < span:
            return False
        self._scanned_to = self.end
        got = s4539._acquire(self.buf, self.search_from - self.base, self.threshold)
        if got is None:
            self.search_from = max(self.search_from, self.end - span)
            return False
        hit, mode, pre = got
        at = hit.sample + self.base
        if at + (pre + 16) * SPS > self.end:
            self.search_from = at - SPS                 # look again when it is whole
            self._scanned_to = 0
            return False
        self._begin_mode(hit, mode, pre, events)
        return True

    def _begin_mode(self, hit, mode, pre, events) -> None:
        self.mode = mode
        self.frames = 0
        self._pre = pre
        self.plan = s4539.preamble_plan(mode) if pre == 287 else s4539.late_plan(mode, 0)
        self._extend()
        self._start_at = hit.sample + self.base
        self.eq = s4539.selector(pre)
        self.eq.start(self._start_at, hit.freq_hz, hit.gain, self.plan.constellation)
        self._stop = phy.loss_detector(pre + 4 * s4539.FRAME, 0.005)
        self._taken = 0
        self._blocks = 0
        self._sent = 0
        self._bits = np.zeros(0, dtype=np.uint8)
        self._bytes = Bytes(self.bit_order)
        events.append(Started("4539", mode.name, hit.freq_hz, self._start_at))

    def _extend(self) -> None:
        n = max(72, self.mode.frames)
        self.plan = phy.Plan.concat([self.plan, s4539.frames_plan(self.mode, self.frames, n)])
        self.frames += n

    def _receive(self, events) -> bool:
        eq, m = self.eq, self.mode
        if len(self.plan) - eq.k < 36 * s4539.FRAME:
            self._extend()
        eq.advance(self.buf, self.base, self.plan, stop=self._stop)
        per = m.frames * s4539.DATA_PER_FRAME
        while len(eq.soft) - self._taken >= per:
            z = np.array(eq.soft[self._taken:self._taken + per])
            s2 = np.array(eq.sig[self._taken:self._taken + per])
            self._taken += per
            self._blocks += 1
            bits = s4539.decode_block(s4539.demap(z, s2, m), m)
            if self._decoded(bits, events):
                return True
        if eq.stopped:
            self._release(self._bits, events)
            events.append(Ended(False, self._snr_db(), "signal lost before EOM"))
            self._finish(events)
            return True
        return False

    def _decoded(self, bits, events) -> bool:
        self._bits = np.concatenate([self._bits, bits])
        e = s4539.find_eom(self._bits)
        if e >= 0:
            self._release(self._bits[:e], events)
            f = self._blocks * self.mode.frames
            end = self._start_at + SPS * (self._pre + f * s4539.FRAME
                                          + (f // s4539.SUPERFRAME) * s4539.REINSERT)
            events.append(Ended(True, self._snr_db(), "EOM"))
            self._finish(events, end=end - 16 * SPS)
            return True
        self._hold(events)
        return False



# --------------------------------------------------------------------------
# MIL-STD-188-110C appendix D, 3 kHz
# --------------------------------------------------------------------------

class ReceiverWbhf(_Receiver):
    """Listens for appendix D's 3 kHz waveforms; the preamble says which.
    Bytes are released a whole interleaver block at a time, less the last
    few that might be the start of the end-of-message pattern."""

    ref = wbhf.FIXED
    after_hit = wbhf.NEED * SPS
    _HOLD_BITS = len(wbhf.EOM) + 8

    def __init__(self, *, bit_order: str = "lsb", threshold: float = wbhf.THRESHOLD) -> None:
        super().__init__(threshold)
        self.bit_order = bit_order

    def _acquire(self, events) -> bool:
        if self.end - self._scanned_to < _SCAN_EVERY and self._scanned_to:
            return False
        span = wbhf.FIXED.size * SPS + self.after_hit
        if self.end - self.search_from < span:
            return False
        self._scanned_to = self.end
        got = wbhf._acquire(self.buf, self.search_from - self.base, self.threshold)
        if got is None:
            self.search_from = max(self.search_from, self.end - span)
            return False
        hit, count, mode = got
        self.mode = mode
        self.frames = 0
        self._pre = wbhf.known_ahead(mode, count)
        self.plan = wbhf.head_plan(mode, count)
        self._extend()
        self._start_at = hit.sample + self.base
        self.eq = wbhf.selector(mode, self._pre)
        self.eq.start(self._start_at, hit.freq_hz, hit.gain, self.plan.constellation)
        self._fs = mode.frame_symbols
        self._stop = wbhf.loss_stop(mode, self._pre + 4 * self._fs)
        self._taken = 0
        self._blocks = 0
        self._bits = np.zeros(0, dtype=np.uint8)
        self._bytes = Bytes(self.bit_order)
        events.append(Started("WBHF", mode.name, hit.freq_hz, self._start_at))
        return True

    def _extend(self) -> None:
        n = max(64, self.mode.frames)
        more = wbhf.frames_plan(self.mode, self.frames, n)
        constellation = self.plan.constellation
        self.plan = phy.Plan.concat([self.plan, more])
        self.plan.constellation = constellation
        self.frames += n

    def _receive(self, events) -> bool:
        eq, m = self.eq, self.mode
        if len(self.plan) - eq.k < max(32, m.frames // 2) * self._fs:
            self._extend()
        eq.advance(self.buf, self.base, self.plan, stop=self._stop)
        while True:
            if m.walsh:
                if len(eq.walsh) - self._taken < m.frames:
                    break
                sl = slice(self._taken, self._taken + m.frames)
                llr = wbhf.walsh_llr(np.array(eq.walsh[sl]), np.array(eq.walsh_sig[sl]))
                self._taken += m.frames
            else:
                per = m.frames * m.w.unknown
                if len(eq.soft) - self._taken < per:
                    break
                sl = slice(self._taken, self._taken + per)
                llr = wbhf.demap(np.array(eq.soft[sl]), np.array(eq.sig[sl]), m)
                self._taken += per
            self._blocks += 1
            if self._decoded(wbhf.decode_block(llr, m), events):
                return True
        if eq.stopped:
            self._release(self._bits, events)
            events.append(Ended(False, self._snr_db(), "signal lost before EOM"))
            self._finish(events)
            return True
        return False

    def _decoded(self, bits, events) -> bool:
        self._bits = np.concatenate([self._bits, bits])
        e = s4539.find_eom(self._bits)
        if e >= 0:
            self._release(self._bits[:e], events)
            f = self._blocks * self.mode.frames
            end = self._start_at + SPS * (self._pre + f * self._fs)
            events.append(Ended(True, self._snr_db(), "EOM"))
            self._finish(events, end=end - 16 * SPS)
            return True
        safe = self._bits.size - self._HOLD_BITS
        safe -= safe % 8
        if safe > 0:
            self._release(self._bits[:safe], events)
            self._bits = self._bits[safe:]
        return False

    def _release(self, bits, events) -> None:
        out = self._bytes.push(bits)
        if out:
            events.append(Data(out))


# --------------------------------------------------------------------------
# The 39-tone parallel mode (MIL-STD-188-110 appendix B)
# --------------------------------------------------------------------------

@dataclass
class _Rx39:
    m: tone39.Mode39
    df: float
    pos: float
    prev: np.ndarray
    decoder: tone39.Decoder
    data_at: int
    back: int
    step: float = float(tone39.T)
    ppm: float = 0.0
    z: list = field(default_factory=list)
    since: int = 0
    clock_skip: int = 0
    soft: np.ndarray = field(default_factory=lambda: np.zeros((0, 39, 2)))
    signal: float = 0.0
    noise: float = 0.0
    elements: int = 0
    pending_z: np.ndarray = field(default_factory=lambda: np.zeros((0, 40), dtype=complex))
    pending_q: np.ndarray = field(default_factory=lambda: np.zeros((0, len(tone39.QUIET_HZ)), dtype=complex))


class Receiver39(_Buffer):
    """Listens for the 39-tone mode, on real audio at ``FS`` (the front
    end's ``audio``, taken before the 1800 Hz matched filter, which would
    cut the Doppler tone at 393.75 Hz).

    The preamble does not say the mode. Given one (both ends agreeing, as
    for 4285), the receiver starts decoding as soon as it has the timing;
    given none, it waits until it has seen enough of the data for
    ``tone39.detect`` to weigh every mode, about eight seconds, and takes
    what that finds.

    Elements are read as they arrive, each where the sender's clock puts
    it: the clock is measured from the data every few seconds and the
    reading follows it, so a long message does not walk out of the guard.
    A sender whose clock is off by more than ``tone39.CLOCK_PPM`` (a
    recording played at the wrong speed, not a sound card) is for the
    offline receiver, which resamples.

    Bytes are released a super block at a time. A synchronous message has
    no end but the signal's: it ends when a super block's code words fail
    as noise makes them."""

    #: Elements past part three needed before decoding starts, when the
    #: mode is known: ``settle`` reads 40, ``align`` a few more either side.
    _START = 48
    _DRIFT_EVERY = 128                  # elements between clock measurements
    _DRIFT_OVER = 384                   # elements each measurement weighs

    def __init__(self, mode: str | tone39.Mode39 | None = None, *, bit_order: str = "lsb",
                 extended: bool = False) -> None:
        self.mode = tone39.mode(mode) if mode is not None else None
        self.bit_order = bit_order
        self.extended = self.mode.extended if self.mode is not None else extended
        super().__init__()
        self._rx = None
        self.last_end = 0

    @property
    def receiving(self) -> bool:
        return self._rx is not None

    # -- acquisition -------------------------------------------------------

    def _acquire(self, events) -> bool:
        T = tone39.T
        n_pre = 97
        if self._scanned_to and self.end - self._scanned_to < FS // 2:
            return False
        if self.end - self.search_from < (n_pre + 4) * T:
            return False
        self._scanned_to = self.end
        got = tone39.find_any_preamble(self.buf, max(0, self.search_from - self.base))
        if got is None:
            # Keep enough behind to find a preamble that straddles the end.
            self.search_from = max(self.search_from, self.end - (n_pre + 4) * T)
            return False
        p3, df, extended = got
        onset = p3 - (85 if extended else 22) * T
        if onset + 101 * T > self.buf.size:
            return False
        n_pre = 97 if extended else 23
        n3 = 12 if extended else 1
        wait = self._START if self.mode is not None else tone39.detect_elements(extended)
        if p3 + (n3 + wait + 2) * T > self.buf.size:
            # Found, but not yet enough after it: look again when there is.
            self.search_from = self.base + p3 - n_pre * T - T
            return False
        x = self.buf
        p3 = tone39.settle(x, p3, df, extended=extended)
        p3 = tone39.align(x, p3, df, m=self.mode, extended=extended)
        m = replace(self.mode, extended=extended) if self.mode is not None else tone39.detect(x, p3, df, extended=extended)
        if m is None:
            self.search_from = self.base + p3 + T
            self._scanned_to = 0
            return True
        n3 = 12 if m.extended else 1
        ref = self.base + p3 + (n3 - 1) * T
        z = tone39.read_elements(x, np.array([ref - self.base], dtype=float), df)
        self._rx = _Rx39(m, df, float(ref + T), z[0], tone39.Decoder(m, self.bit_order),
                         ref + T, (m.d - 1) * (16 // m.d)
                         if m.rate < 1200 and m.diversity == "tf" else 0)
        events.append(Started("39", m.name, df, self.base + p3 - (85 if extended else 22) * T))
        return True

    # -- receiving ---------------------------------------------------------

    def _receive(self, events) -> bool:
        r, m, T = self._rx, self._rx.m, tone39.T
        k = int((self.end - T - 1 - r.pos) // r.step) + 1
        if k <= 0:
            return False
        at = r.pos + np.arange(k) * r.step
        z = tone39.read_elements(self.buf, at - self.base, r.df)
        q = tone39.read_elements(self.buf, at - self.base, r.df, tone39.QUIET_HZ)
        r.pos = float(at[-1] + r.step)
        r.pending_z = np.vstack([r.pending_z, z])
        r.pending_q = np.vstack([r.pending_q, q])
        z, q = r.pending_z, r.pending_q
        extent = tone39.extent(z, q)
        lost = extent < len(z)
        take = extent if lost else max(0, len(z) - 64)
        if not take:
            return False
        z, q = z[:take], q[:take]
        r.pending_z, r.pending_q = r.pending_z[take:], r.pending_q[take:]
        soft, _ = tone39.soft_dibits(np.vstack([r.prev[None], z]), r.df)
        r.prev = z[-1]
        r.signal += float(np.sum(np.abs(z) ** 2))
        r.noise += float(np.sum(np.mean(np.abs(q) ** 2, axis=1)))
        r.elements += len(z)
        self._track(r, z)
        r.soft = np.concatenate([r.soft, soft])
        n = r.soft.shape[0] - r.back
        if n > 0:
            words = tone39._combine(m, r.soft)[:n]
            r.soft = r.soft[n:]
            out = r.decoder.push(words.ravel())
            if out:
                events.append(Data(out))
            if r.decoder.eom:
                self._end(events, True, "EOM")
                return True
        if lost:
            self._end(events, False, "signal lost before EOM" if m.char else "signal lost")
            return True
        return False

    def _track(self, r, z) -> None:
        """Follow the sender's clock: measure it on the last few hundred
        elements read at the current timing."""
        skip = min(r.clock_skip, len(z))
        r.clock_skip -= skip
        z = z[skip:]
        r.z.append(z)
        r.since += z.shape[0]
        if r.since < self._DRIFT_EVERY:
            return
        zz = np.vstack(r.z)[-self._DRIFT_OVER:]
        r.z, r.since = [zz], 0
        if zz.shape[0] < self._DRIFT_OVER // 2:
            return
        d = tone39.drift_of(zz, r.df)
        if abs(d) > 20:
            # drift_of already requires a significant slope. Halving it can
            # leave a residual too small to measure but large enough to walk
            # the elements out of the guard over a long message.
            r.step /= 1 + d * 1e-6
            r.ppm = (1 + r.ppm * 1e-6) * (1 + d * 1e-6) * 1e6 - 1e6
            r.z = []
            # These elements were read ahead at the old step. Decode them,
            # but do not measure the corrected clock from their old timing.
            r.clock_skip = len(r.pending_z)

    def _end(self, events, complete: bool, why: str) -> None:
        r = self._rx
        noise = r.noise / max(1, r.elements)
        signal = r.signal / max(1, r.elements) - 40 * noise
        snr = 10 * np.log10(max(signal, 1e-12) /
                            max(noise * 3000 * tone39.W / FS, 1e-12))
        dec = r.decoder
        self.last_end = int(r.data_at + -(-dec.used // r.m.word_bits) * r.step)
        self.last_corrected, self.last_failed = dec.corrected, dec.failed
        self.last_ppm = r.ppm
        self.last_mode = r.m
        events.append(Ended(complete, float(snr), why, self.last_end))
        self.search_from = self.last_end
        self._scanned_to = 0
        self._rx = None

    def flush(self) -> list:
        events = []
        if self._rx is not None:
            self._end(events, False, "recording ended before EOM")
        return events

    def _trim(self) -> None:
        keep = self.search_from if self._rx is None else int(self._rx.pos) - 2 * tone39.T
        cut = min(keep - self.base, self.buf.size)
        if cut > 8 * FS:
            self.buf = self.buf[cut:]
            self.base += cut


def _name4285(mode: Mode4285) -> str:
    for name, m in MODES4285.items():
        if m == mode:
            return name
    return mode.name





# --------------------------------------------------------------------------
# The 16-tone DPSK mode (appendix A)
# --------------------------------------------------------------------------

@dataclass
class _Rx16:
    df: float
    pos: int
    prev: np.ndarray
    start: int
    bytes: Bytes
    turn: deque = field(default_factory=lambda: deque(maxlen=8))
    ratios: deque = field(default_factory=lambda: deque(maxlen=8))
    held: deque = field(default_factory=deque)
    off: int = 0
    sig: float = 0.0
    noise: float = 0.0
    n: int = 0
    floor: float | None = None


class Receiver16(_Buffer):
    """Listens for the 16-tone mode at one rate, on real audio at ``FS``
    (the front end's ``audio``: the Doppler tone at 605 Hz is below the
    matched filter). Nothing in the signal says the rate, so both ends
    agree it beforehand, as for 4285.

    **Finding it.** The preamble is two tones: the Doppler tone, a pure
    line, and 1705 Hz reversing each element. A spectrum of a few elements
    finds the one and checks for the other (with each element's phase
    reckoned from its own start, the reversals make one line 17.5 Hz below
    1705 Hz, not two 37.5 Hz either side of it); the reversals, read the
    same way, give a first timing. The reference element is where the data
    tones' power steps up; the timing is then settled on the data after it
    (``_lock`` says how), and the reference taken again on that timing.

    **Reading it.** Every element's 17 tones over the window centred in it;
    each data tone's change, turned back by what the Doppler tone says the
    offset turned it; the copies of a bit added. Every ``_TRACK`` elements
    the windows are tried a few samples early and late, and move away from
    the side that spills more into the empty places, so a sender's clock
    does not walk them out of the element.

    **The end** is where the data tones fall to what the empty places
    hold, which is noise, over a run of elements; decisions are held back
    ``_HOLD`` elements until the signal is known to have been on for them,
    and the message ends at the last element that still read as signal."""

    _HOP = 256
    _FFT = 1024
    _HOLD = 12
    #: Data tones over the empty places, in power, over eight elements: the
    #: signal at 0 dB in 3 kHz reads about 2.3, noise 1 within about 0.1.
    _ON = 1.6
    _TRACK = 32
    _SETTLE = 24                        # data elements the timing is settled on
    MAX_OFFSET_HZ = 60.0

    def __init__(self, rate: int | str = 2400, *, bit_order: str = "lsb") -> None:
        self.rate = tone16.mode(rate)
        self.mode = str(self.rate)
        self.bit_order = bit_order
        super().__init__()
        self._rx = None
        self.last_end = 0

    @property
    def receiving(self) -> bool:
        return self._rx is not None

    def flush(self) -> list:
        """The audio has ended: whatever is being received ends with it."""
        events: list = []
        if self._rx is not None:
            self._finish(events)
        return events

    # -- acquisition -------------------------------------------------------

    def _acquire(self, events) -> bool:
        T = tone16.T
        need = self._FFT + 8 * T
        if self._scanned_to and self.end - self._scanned_to < self._HOP * 4:
            return False
        if self.end - self.search_from < need:
            return False
        self._scanned_to = self.end
        x = self.buf
        win = np.hanning(self._FFT)
        freqs = np.fft.rfftfreq(self._FFT, 1 / FS)
        band = (freqs > 400) & (freqs < 2800)
        near = np.abs(freqs - tone16.DOPPLER_HZ) <= self.MAX_OFFSET_HZ
        p = max(0, self.search_from - self.base)
        while p + need <= x.size:
            s = np.abs(np.fft.rfft(x[p:p + self._FFT] * win)) ** 2
            floor = np.median(s[band]) + 1e-20
            k = int(np.argmax(np.where(near, s, 0)))
            if s[k] > 40 * floor:
                df = float(freqs[k] - tone16.DOPPLER_HZ)
                sync = np.abs(freqs - (tone16.SYNC_HZ + df)) <= 40
                if np.max(s[sync]) > 8 * floor:
                    got = self._lock(p, df)
                    if got == "wait":
                        self.search_from = self.base + p
                        return False
                    if got is not None:
                        self._rx = got
                        events.append(Started("16", self.mode, got.df, got.start))
                        return True
                    if self._rx is None and self.end - (self.base + p) < need + 40 * T:
                        # Perhaps its reference element is not here yet.
                        self.search_from = self.base + p
                        return False
            p += self._HOP
        self.search_from = self.base + max(0, x.size - need)
        return False

    def _lock(self, p: int, df: float) -> _Rx16 | str | None:
        """Timing and the reference element, from a preamble found at
        ``p`` (a buffer index), or None."""
        T, x = tone16.T, self.buf
        n = min(6, (x.size - p - tone16.W) // T - 1)
        if n < 4:
            return None
        score = []
        for o in range(0, T, 2):
            z = tone16.read(x, p + o + np.arange(n) * T, df, [tone16.SYNC_HZ])[:, 0]
            c = -np.sum(z[1:] * np.conj(z[:-1]))
            score.append((abs(c), c, o, abs(c) / (np.sum(np.abs(z) ** 2) + 1e-20)))
        top = max(score, key=lambda s: s[0])
        if top[3] < 0.5:
            return None                         # no reversals: not the preamble
        # And 1705 Hz alone of the data tones: in the data they are all alike.
        zo = tone16.read(x, p + top[2] + np.arange(n) * T, df, tone16.TONE_HZ)
        p8 = np.mean(np.abs(zo[:, 7]) ** 2)
        if p8 < 3 * np.mean(np.abs(np.delete(zo, 7, axis=1)) ** 2):
            return None
        # A window across two elements still sees a tone reversing, less of
        # it: the ones that read the most lie wholly inside an element.
        good = [s for s in score if s[0] >= 0.95 * top[0]]
        # The windows that read the reversals whole: the middle of the run
        # of them (taken round the element, which it may straddle).
        offs = np.array([s[2] for s in good])
        ang = np.angle(np.mean(np.exp(2j * np.pi * offs / T)))
        at = p + int(round((ang % (2 * np.pi)) * T / (2 * np.pi)))
        # Each element read from its own start, the reversals turn by what
        # the offset itself turns them, whatever frequency they are read
        # at: the offset, to a multiple of 75 Hz that the spectrum settles.
        c = sum(s[1] for s in good)
        fine = float(np.angle(c)) * FS / (2 * np.pi * T)
        df = fine + 75.0 * round((df - fine) / 75.0)
        # The reference element: where the data tones' power steps up. The
        # step is taken over four elements each side (at 0 dB one element's
        # power swings as far as the step itself), from a little before the
        # lock to a preamble's length after it, on the reversals' timing: the
        # windows may lie across edges, which moves power between tones but
        # not out of them. Not 1705 Hz's neighbours either: its reversals,
        # read across an edge, spill into them.
        others = np.r_[tone16.TONE_HZ[:6], tone16.TONE_HZ[9:]]
        w = 4
        js = np.arange(-w, tone16.PREAMBLE_ELEMENTS + 2 + w)
        if at + (js[-1] + self._SETTLE + 3) * T > x.size:
            return "wait"
        at_j = at + js * T
        js = js[at_j >= 0]
        at_j = at_j[at_j >= 0]
        if js.size < 2 * w + 1:
            return None
        pw = np.mean(np.abs(tone16.read(x, at_j, df, others)) ** 2, axis=1)
        pw = pw + 0.01 * pw.max() + 1e-20        # silence to leakage is no step
        cs = np.r_[0.0, np.cumsum(pw)]
        i = np.arange(w, js.size - w + 1)
        step = (cs[i + w] - cs[i]) / (cs[i] - cs[i - w])
        if step.max() < 1.3:
            return None                         # no data after it
        best = int(i[int(np.argmax(step))])
        # The step says where to within an element or so (a window half in
        # the one and half in the other reads between); the first near it
        # standing over the levels' geometric middle is the one, for now.
        mid = np.sqrt(np.median(pw[best - w:best]) * np.median(pw[best:best + w]))
        for j in range(max(0, best - 2), min(js.size, best + 3)):
            if pw[j] > mid:
                best = j
                break
        k = at + int(js[best]) * T
        # The timing, settled on the data after it. A window across an
        # element's edge mixes two elements' phases: how tightly the changes
        # (their diversity copies added) sit on the constellation says where
        # the windows lie wholly inside. Where that has no contrast (2400
        # bit/s at a few dB, which does not decode anyway), what spills into
        # the empty places says it instead.
        offs = np.arange(-T // 2, T // 2, 2)
        ats = [k + T + o + np.arange(self._SETTLE) * T for o in offs]
        coh = np.array([self._coherence(x, a, df) for a in ats])
        if coh.max() - coh.min() > 0.05:
            bad = coh.max() - coh
        else:
            bad = np.array([self._spill(x, a, df) for a in ats])
        # Fit the run of windows inside an element, of known width (the
        # element less the window, less a little for the channel's spread):
        # the place round the element where what lies inside the run beats
        # what lies outside by the most. A box fits through noise that a
        # threshold on the profile does not.
        half = (T - tone16.W) // 2 - 4
        d = (offs[None, :] - offs[:, None] + T // 2) % T - T // 2
        inside = np.abs(d) <= half
        good = -bad
        fit = [good[m].mean() - good[~m].mean() for m in inside]
        centre = float(offs[int(np.argmax(fit))])
        centre = (centre + T // 2) % T - T // 2
        k += int(round(centre))
        # The windows inside elements now, which element is the reference
        # is plain: of this one and its neighbours, the first over the
        # middle level. (The step's window may have been mostly the last of
        # the preamble with an edge of the reference in it.)
        near = k + np.arange(-1, 2) * T
        pn = np.mean(np.abs(tone16.read(x, near, df, others)) ** 2, axis=1)
        over = np.flatnonzero(pn > mid)
        k = int(near[over[0]]) if over.size else k
        # And the offset again, from the Doppler tone over the same elements:
        # 7 dB up and unbroken, it says it better than a few reversals.
        zd = tone16.read(x, k + np.arange(self._SETTLE + 1) * T, df, [tone16.DOPPLER_HZ])[:, 0]
        turn = np.sum(tone16._doppler_turn(zd[1:], zd[:-1]))
        fine = float(np.angle(turn)) * FS / (2 * np.pi * T)
        df = fine + 75.0 * round((df - fine) / 75.0)
        ref = tone16.read(x, np.array([k]), df, tone16.ALL_HZ)[0]
        return _Rx16(df, self.base + k + T, ref, self.base + k - tone16.GUARD,
                     Bytes(self.bit_order))

    # -- reading -------------------------------------------------------------

    def _receive(self, events) -> bool:
        rx, T = self._rx, tone16.T
        at = rx.pos - self.base
        if at + tone16.W + 8 > self.buf.size:
            return False
        z = tone16.read(self.buf, np.array([at]), rx.df, tone16.ALL_HZ)[0]
        q = tone16.read(self.buf, np.array([at]), rx.df, tone16.QUIET_HZ)[0]
        turn = tone16._doppler_turn(z[0], rx.prev[0])
        rx.turn.append(turn)
        t = np.sum(rx.turn)
        w = z[1:] * np.conj(rx.prev[1:]) * np.exp(-1j * np.angle(t))
        bits, _ = tone16.decide(self.rate, w)
        sig, noise = float(np.mean(np.abs(z[1:]) ** 2)), float(np.mean(np.abs(q) ** 2))
        # The noise from the empty places over many elements: four places
        # in one element are too few to divide by (the ratio's tail would
        # turn noise into signal now and then).
        rx.floor = noise if rx.floor is None else 0.97 * rx.floor + 0.03 * noise
        ratio = sig / (rx.floor + 1e-20)
        # One element's ratio at a few dB is too rough to end a message on:
        # the last eight's.
        rx.ratios.append(ratio)
        on = float(np.mean(rx.ratios)) > self._ON
        rx.prev = z
        rx.pos += T
        rx.n += 1
        if (rx.n % self._TRACK) == 0:
            self._track(at)
        # An element that reads weak is held with the rest: a fade that
        # passes keeps its bits (dropping them would slip every bit after),
        # and only a run of them ends the message, before its first.
        rx.held.append((bits, ratio))
        if on:
            rx.off = 0
            rx.sig += sig
            rx.noise += noise
            while len(rx.held) > self._HOLD:
                self._release(events, rx.held.popleft()[0])
            return True
        rx.off += 1
        if rx.off >= self._HOLD:
            self._finish(events)
        return True

    def _coherence(self, x, at, df) -> float:
        """How tightly the changes over the windows at ``at`` (the copies
        of each bit added) sit on the constellation: 1 exactly, near 0 for
        noise. Every point of both constellations is 45 degrees plus a
        multiple of 90, so the fourth power sends them all to one place."""
        z = tone16.read(x, np.concatenate([[at[0] - tone16.T], at]), df, tone16.TONE_HZ)
        w0 = z[1:] * np.conj(z[:-1])
        sym = tone16.carried(self.rate)
        w = np.zeros((w0.shape[0], sym.max() + 1), dtype=complex)
        for t in range(16):
            w[:, sym[t]] += w0[:, t]
        q = w ** 4 / (np.abs(w) ** 2 + 1e-20)
        return float(np.abs(np.sum(q)) / (np.sum(np.abs(w) ** 2) + 1e-20))

    @staticmethod
    def _spill(x, at, df) -> float:
        q = tone16.read(x, at, df, tone16.QUIET_HZ)
        z = tone16.read(x, at, df, tone16.TONE_HZ)
        return float(np.sum(np.abs(q) ** 2) / (np.sum(np.abs(z) ** 2) + 1e-20))

    def _track(self, at: int) -> None:
        """Try the windows a few samples either way over the last elements
        and move one sample away from the side that spills more."""
        rx, T = self._rx, tone16.T
        d = tone16.GUARD - 6
        ks = at - np.arange(self._TRACK) * T
        ks = ks[(ks - d >= 0) & (ks + d + tone16.W <= self.buf.size)]                    # none from before the buffer
        if ks.size < 4:
            return
        early, late = (self._spill(self.buf, ks + s, rx.df) for s in (-d, d))
        step = -1 if late > 1.3 * early else 1 if early > 1.3 * late else 0
        if step:
            rx.pos += step
            # The next change is against the element just read: read it
            # again where the next will be read, or the move turns it.
            rx.prev = tone16.read(self.buf, np.array([rx.pos - T - self.base]),
                                     rx.df, tone16.ALL_HZ)[0]

    def _release(self, events, bits) -> None:
        out = self._rx.bytes.push(bits)
        if out:
            events.append(Data(out))

    def _finish(self, events) -> None:
        """The signal has gone: what was held, up to the last element that
        still read as signal, is the message's end."""
        rx = self._rx
        held = list(rx.held)
        # Halfway (in decibels) between the noise and what the message read
        # while it was on: noise does not reach it by chance, as it may
        # reach the fixed threshold that only has to tell a run of them.
        level = rx.sig / (rx.noise + 1e-20) if rx.noise else self._ON ** 2
        cut = max(self._ON, float(np.sqrt(level)))
        keep = len(held)
        while keep and held[keep - 1][1] <= cut:
            keep -= 1
        for bits, _ in held[:keep]:
            self._release(events, bits)
        rx.off = len(held) - keep
        # Sixteen tones' worth of signal over the noise in 3 kHz of bins.
        p_s = 16 * max(rx.sig - rx.noise, 1e-20)
        snr = 10 * np.log10(p_s / max(rx.noise * 3000 / (FS / tone16.W), 1e-20))
        # No end-of-message: the signal's going is the only end there is.
        self.last_end = rx.pos - rx.off * tone16.T
        events.append(Ended(False, float(snr), "signal ended", self.last_end))
        self.search_from = self.last_end
        self._scanned_to = 0
        self._rx = None

    def _trim(self) -> None:
        keep_from = (self._rx.pos - (self._TRACK + 8) * tone16.T) if self._rx is not None \
            else self.search_from
        cut = min(keep_from - self.base, self.buf.size)
        if cut > 8 * FS:
            self.buf = self.buf[cut:]
            self.base += cut


# --------------------------------------------------------------------------
# Binary FSK (STANAG 4481 C.2-C.3, MIL-STD-188-110B 5.1)
# --------------------------------------------------------------------------

class ReceiverFsk:
    """Listens for one FSK mode on real audio at ``FS``.

    Every ``_HOP`` it looks at the last half second: where the tones stand
    (the carrier offset) and whether they stand over the places beside them
    nothing is sent in. Once they do, each sample from the half second back
    gets a soft bit: the mark's share of the two tones' energy over the bit
    from there, less the space's. Bits are decided ``_AHEAD`` bits behind
    what has arrived, at the timing the last ``_OVER`` bits say: where the
    windows lie across an edge the tones mix and their difference shrinks,
    so the bit is read half a bit from where it shrinks most; once edges
    have been seen, the timing moves only a little at a time. Character
    framing is a space start bit, eight data bits and a mark stop bit; a
    character whose stop bit is not mark is dropped, and the next looked
    for. Nothing is taken until eight marks in a row, the idle line, have
    been read. The signal has gone when the tones have stood no higher than
    the noise for ``_GONE_S``; the bits after the last that stood clearly
    over it are dropped."""

    _HOP = FS // 4
    _LOOK = FS // 2
    _AHEAD = 16
    _OVER = 96
    _ON = 4.0                            # the tones over the noise, in power
    _GONE_S = 1.0

    def __init__(self, mode: str | fsk.FskMode = "75W", *, bit_order: str = "lsb") -> None:
        self.m = fsk.mode(mode)
        self.mode = self.m.name
        self.bit_order = bit_order
        self.buf = np.zeros(0)
        self.base = 0
        self._scanned_to = 0
        self._rx: dict | None = None

    @property
    def end(self) -> int:
        return self.base + self.buf.size

    @property
    def receiving(self) -> bool:
        return self._rx is not None

    def push(self, audio: np.ndarray) -> list:
        self.buf = np.concatenate([self.buf, np.asarray(audio, dtype=np.float64)])
        events: list = []
        if self._rx is None:
            self._acquire(events)
        if self._rx is not None:
            self._receive(events)
        self._trim()
        return events

    def flush(self) -> list:
        events: list = []
        if self._rx is not None:
            self._receive(events, final=True)
            if self._rx is not None:
                self._finish(events)
        return events

    # -- energies --------------------------------------------------------------

    def _energies(self, lo: int, hi: int, df: float):
        """For the bit-long windows starting at absolute samples [lo, hi):
        the soft bits (mark positive) and the tones' power over the noise."""
        n = self.m.n
        x = self.buf[lo - self.base:hi - self.base + n - 1]
        em = es = 0.0
        for mark, space in self.m.pairs():
            em = em + fsk.tone_energy(x, mark + df, n, lo)
            es = es + fsk.tone_energy(x, space + df, n, lo)
        q = np.mean([fsk.tone_energy(x, f + df, n, lo) for f in self.m.quiet()], axis=0)
        # The noise over half a second about each window: a few places over
        # one bit are too few to divide by (the ratio's tail would make
        # signal of noise now and then).
        h = min(q.size, FS // 2) // 2
        c = np.concatenate([[0.0], np.cumsum(q)])
        i = np.arange(q.size)
        a, b = np.maximum(i - h, 0), np.minimum(i + h + 1, q.size)
        q = (c[b] - c[a]) / (b - a)
        k = len(self.m.pairs())
        return (em - es) / (em + es + 1e-20), (em + es) / (2 * k * q + 1e-20), np.abs(em - es)

    # -- acquisition -----------------------------------------------------------

    def _acquire(self, events) -> None:
        while self.end - max(self._scanned_to, self.base + self._LOOK) >= self._HOP or \
                (self._scanned_to == 0 and self.buf.size >= self._LOOK):
            at = max(self._scanned_to + self._HOP, self.base + self._LOOK)
            if at > self.end:
                return
            self._scanned_to = at
            seg = self.buf[at - self._LOOK - self.base:at - self.base]
            df = fsk.offset(seg, self.m)
            lo = at - self._LOOK
            _, ratio, _ = self._energies(lo, at - self.m.n + 1, df)
            if np.median(ratio) > self._ON:
                self._rx = {"df": df, "start": lo, "next": None, "bits": [],
                            "last_on": lo, "last_strong": lo, "char": None, "out": [],
                            "sig": [], "seen": False, "idle": 0, "hunt": None}
                events.append(Started("FSK", self.mode, df, lo))
                return

    # -- reading -------------------------------------------------------------

    def _receive(self, events, final: bool = False) -> None:
        rx, n = self._rx, self.m.n
        top = self.end - n + 1                   # windows with all their samples
        stop = top if final else top - self._AHEAD * n
        nxt = rx["next"]
        if (nxt if nxt is not None else rx["start"]) >= stop:
            return
        lo = max(self.base, (nxt if nxt is not None else rx["start"]) - self._OVER * n)
        soft, ratio, diff = self._energies(lo, top, rx["df"])
        # Where the tones' difference shrinks most, round the bit, are the
        # edges. The difference itself, not its share: a stretch where one
        # tone has faded says nothing of the edges, and counts for little.
        idx = lo + np.arange(soft.size)
        prof = np.bincount(idx % n, weights=diff, minlength=n) / \
            np.maximum(np.bincount(idx % n, minlength=n), 1)
        prof = np.convolve(np.r_[prof[-2:], prof, prof[:2]], np.ones(5) / 5, "valid")
        phase = (int(np.argmin(prof)) + n // 2) % n
        # Edges are seen once the bits change; on the idle mark there are
        # none, and the timing may go where it likes. After that it moves a
        # little at a time: in fading the edges read noisily, and a jump of
        # a large part of a bit would drop or repeat one.
        if prof.max() - prof.min() > 0.2 * prof.max():
            rx["edges"] = True
        if nxt is None:
            nxt = rx["start"] + (phase - rx["start"]) % n
        else:
            step = (phase - nxt + n // 2) % n - n // 2
            if rx.get("edges_used"):
                step = int(np.clip(step, -max(1, n // 32), max(1, n // 32)))
            nxt += step
            rx["edges_used"] = rx.get("edges", False)
        while nxt < stop:
            self._bit(events, int(soft[nxt - lo] > 0), float(ratio[nxt - lo]), nxt)
            if self._rx is None:
                return
            nxt += n
        rx["next"] = nxt

    def _bit(self, events, bit: int, ratio: float, at: int) -> None:
        rx = self._rx
        # Each bit's own reading says where the signal last was; it has gone
        # when that is long enough ago (a fade does not end a message).
        # Against halfway (in decibels) between the noise and what the signal
        # has read so far: noise reaches the fixed threshold now and then,
        # most at the higher rates, where a bit's window is short.
        # Two tests. The signal is still there while bits stand over the
        # noise now and then (a fade does not end it); a bit is released
        # once one after it has stood clearly over, halfway (in decibels)
        # between the noise and what the signal has read so far, which
        # noise does not reach by chance as it reaches the first.
        level = float(np.median(rx["sig"])) if len(rx["sig"]) >= 8 else self._ON ** 2
        if ratio > self._ON:
            rx["last_on"] = at
            rx["sig"].append(ratio)
        if ratio > max(self._ON, float(np.sqrt(level))):
            rx["last_strong"] = at
        rx["bits"].append((bit, at))
        while rx["bits"] and rx["bits"][0][1] <= rx["last_strong"]:
            self._take(events, rx["bits"].pop(0)[0])
        if at - rx["last_on"] > self._GONE_S * FS:
            self._finish(events)

    def _take(self, events, b: int) -> None:
        rx = self._rx
        # Nothing is taken until a run of marks, the idle line, has been
        # seen: the bits read while the signal was coming up are not bits.
        if rx["idle"] < 8:
            rx["idle"] = rx["idle"] + 1 if b == 1 else 0
            return
        if self.m.sync:
            if not rx["seen"]:
                if b == 1:
                    return                       # still the idle mark
                rx["seen"] = True
            rx["out"].append(b)
            if len(rx["out"]) == 8:
                self._emit(events, rx["out"])
                rx["out"] = []
            return
        if rx["hunt"] is not None:
            self._hunt(events, b)
            return
        c = rx["char"]
        if c is None:
            if b == 0:
                rx["char"] = []                  # a start bit
            return
        if len(c) < 8:
            c.append(b)
            return
        rx["char"] = None
        if b == 1:                               # its stop bit
            self._emit(events, c)
        else:
            rx["hunt"] = []

    _HUNT = 6

    def _hunt(self, events, b: int) -> None:
        """After a stop bit that was not mark: characters sent back to back
        give a wrong framing that passes now and then by chance, and the
        next space is no surer a start bit than any other. Take instead the
        framing under which the next ``_HUNT`` characters all start with a
        space and stop with a mark; the line going idle ends the hunt too."""
        h = self._rx["hunt"]
        h.append(b)
        if len(h) >= 10 and all(h[-10:]):
            self._rx["hunt"] = None              # idle: the next space starts a character
            return
        need = 10 * self._HUNT
        if len(h) < need + 9:
            return
        for o in range(10):
            w = np.array(h[o:o + need]).reshape(self._HUNT, 10)
            if not w[:, 0].any() and w[:, 9].all():
                for ch in w:
                    self._emit(events, list(ch[1:9]))
                rest = h[o + need:]
                self._rx["hunt"] = None
                for x in rest:
                    self._take(events, x)
                return
        del h[:10]                               # none fits yet: slide on

    def _emit(self, events, bits) -> None:
        byte = np.packbits(np.array(bits, dtype=np.uint8),
                           bitorder="little" if self.bit_order == "lsb" else "big")
        events.append(Data(byte.tobytes()))

    def _finish(self, events) -> None:
        rx = self._rx
        sig = float(np.mean(rx["sig"])) if rx["sig"] else 1.0
        # The tones' power over the noise in a bit's bandwidth, put in 3 kHz.
        snr = 10 * np.log10(max(sig - 1, 1e-6) * self.m.baud / 3000)
        events.append(Ended(False, float(snr), "signal ended"))
        # Look again only past it: a window over its tail would find it again.
        self._scanned_to = rx["last_on"] + self.m.n + self._LOOK
        self._rx = None

    def _trim(self) -> None:
        keep = ((self._rx["next"] or self._rx["start"]) - (self._OVER + 4) * self.m.n) \
            if self._rx is not None \
            else self._scanned_to - self._LOOK
        cut = min(keep - self.base, self.buf.size)
        if cut > 8 * FS:
            self.buf = self.buf[cut:]
            self.base += cut


# --------------------------------------------------------------------------
# STANAG 4529: 4285 at half speed
# --------------------------------------------------------------------------

class Receiver4529:
    """Listens for 4529 in one mode and on one sub-carrier, both set
    beforehand: the audio is brought to what 4285's receiver expects
    (``s4529.Squeeze``) and that receiver does the rest. What it reports is
    put back on the real time scale: offsets halved, samples doubled."""

    def __init__(self, mode: str | s4529.Mode4529 = "600L", *, bit_order: str = "lsb") -> None:
        self.m = s4529.mode(mode)
        self.mode = self.m.full_name
        self._sq = s4529.Squeeze(self.m.carrier)
        self._fe = FrontEnd(FS)
        self._rx = Receiver4285(self.m.as_4285, bit_order=bit_order)

    @property
    def receiving(self) -> bool:
        return self._rx.receiving

    def push(self, audio: np.ndarray) -> list:
        out = []
        for e in self._rx.push(self._fe.push(self._sq.push(audio))):
            if isinstance(e, Started):
                e = Started("4529", self.mode, e.freq_hz / 2, 2 * e.at)
            out.append(e)
        return out


def receive_whole(audio, receiver, *, start=0, parallel=False):
    """Feed a recording through the live receiver and collect its first burst."""
    fe = None if parallel else FrontEnd(FS)
    first, body = None, bytearray()
    x = np.asarray(audio, dtype=float)
    receiver.search_from = start
    for chunk in (x, np.zeros(12 * FS if isinstance(receiver, Receiver39) else 4 * FS)):
        for i in range(0, chunk.size, FS // 4):
            part = chunk[i:i + FS // 4]
            events = receiver.push(part if parallel else fe.push(part))
            for event in events:
                if isinstance(event, Started) and first is None:
                    first = event
                elif isinstance(event, Data) and first is not None:
                    body.extend(event.data)
                elif isinstance(event, Ended) and first is not None:
                    return first, bytes(body), event
    for event in receiver.flush():
        if isinstance(event, Data) and first is not None:
            body.extend(event.data)
        elif isinstance(event, Ended) and first is not None:
            return first, bytes(body), event
    return None


__all__ = ["FrontEnd", "Receiver110", "Receiver4285", "Receiver4539", "ReceiverWbhf", "Receiver39", "Receiver16", "ReceiverFsk", "Receiver4529",
           "Started", "Data", "Ended", "receive_whole"]
