# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ALE modem: 8-ary FSK at 125 tones a second (MIL-STD-188-141B A.5.1),
and a receiver that reads words and frames from it (A.5.2.6).

**Transmit.** Each tone carries three bits, first bit sent the most
significant, mapped by the table in A.5.1.2 (a Gray code: neighbouring tones
differ in one bit). Tones are 8 ms, and every tone frequency is a multiple of
125 Hz, so each tone is a whole number of cycles: a cosine started at its
peak ends at its peak, which is the phase-continuous, zero-slope transition
the text asks for. 8000 samples a second makes a tone 64 samples.

**Receive.** Tone energies are taken over 8 ms windows every millisecond;
the symbol timing is the millisecond offset at which the energy stands most
in one tone. A.4.2.3 asks for no more than a decibel lost to a 60 Hz Doppler
shift, which at 125 Hz tone bins is most of a bin, so the tones' frequency
offset is estimated from a quarter-second spectrum (the eight tones as a comb)
and the correlators move with it.

Where the text describes a hard 2-of-3 majority vote over each bit's three
copies (table A-VI), this receiver sums the three copies' soft values and
decodes the Golay halves with their reliabilities (`golay.decode_soft`). The
unanimous-vote count A.5.4.1.1 builds the BER report on is still counted,
from the copies' hard decisions, because the far end reads that report.

Words are acquired by the criteria of A.5.2.6.3 — both halves decode, the
preamble is a routing one, the characters are in the Basic 38 set, and
enough of the votes were unanimous — and then read at that word phase,
every 392 ms, until four go unread or the frame's conclusion ends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from . import words as W

FS = 8000
BAUD = 125
SPS = FS // BAUD                                        # 64 samples a tone

#: A.5.1.2: tribit (first bit most significant) -> tone.
TONE_HZ = {0b000: 750, 0b001: 1000, 0b011: 1250, 0b010: 1500,
           0b110: 1750, 0b111: 2000, 0b101: 2250, 0b100: 2500}
#: Tone index 0..7 (750 .. 2500 Hz) -> tribit.
BITS_OF_TONE = np.array([b for f, b in sorted((f, b) for b, f in TONE_HZ.items())])
TONES = np.array(sorted(TONE_HZ.values()), dtype=np.float64)


def modulate(ws: list, *, freq_hz: float = 0.0) -> np.ndarray:
    """Words -> audio at ``FS``, each word sent three times, peak 1. An array
    among the words is bits sent as they are: a DBM data block (A.5.7.4)."""
    bits = np.concatenate([np.asarray(w, dtype=np.int64) if isinstance(w, np.ndarray)
                           else np.tile(W.to_air(w), 3) for w in ws]) if len(ws) else np.zeros(0)
    tri = bits.reshape(-1, 3) @ np.array([4, 2, 1])
    t = np.arange(SPS) / FS
    out = np.empty(tri.size * SPS)
    phase = 0.0
    for i, b in enumerate(tri):
        f = TONE_HZ[int(b)] + freq_hz
        out[i * SPS:(i + 1) * SPS] = np.cos(phase + 2 * np.pi * f * t)
        phase += 2 * np.pi * f * SPS / FS
    ramp = min(24, out.size // 2)
    if ramp:
        edge = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
        out[:ramp] *= edge
        out[-ramp:] *= edge[::-1]
    return out


def air_seconds(ws: list) -> float:
    """How long a frame (words, and DBM blocks' bits) is on the air."""
    return sum(w.size / 3 if isinstance(w, np.ndarray) else W.TRIPLE_BITS / 3
               for w in ws) / BAUD




# --------------------------------------------------------------------------
# Receive
# --------------------------------------------------------------------------

_HOP = 8                                                  # 1 ms
_PHASES = SPS // _HOP
#: Tribit -> which tones have each of its bits at 0 (for soft bits).
_ZERO = np.array([[((BITS_OF_TONE[k] >> (2 - j)) & 1) == 0 for k in range(8)]
                  for j in range(3)])


@dataclass
class Heard:
    """One word, read."""

    word: int
    end: int                     # sample index (at FS) where its third copy ends
    errors: int                  # bits the Golay decoders corrected
    unanimous: int               # of 48 votes (A.5.4.1.1)
    score: float

    @property
    def preamble(self) -> int:
        return W.preamble_of(self.word)

    def __str__(self) -> str:
        return W.describe(self.word)


@dataclass
class Received:
    """A frame, read to its end."""

    frame: W.Frame
    words: list[Heard]
    start: int
    end: int                     # sample index where the frame ended
    ber: float                   # A.5.4.1.1's average of non-unanimous votes
    sinad_db: float
    freq_hz: float
    #: Under linking protection, the seed the frame was read under: the
    #: station's own ("fine"), a coarse one, or none ("clear").
    seed: str | None = None


class _Candidate(NamedTuple):
    key: tuple
    offset: int
    heard: Heard
    seeds: list
    end: int


class Receiver:
    """Audio at ``FS`` in, words and frames out. Every sample once, in order."""

    #: Acquisition. A scanning call repeats one word, so three copies of a
    #: *rotated* word sit in every window at every bit offset: all unanimous,
    #: and the Golay decoders find some codeword near most of them. Only the
    #: true offset decodes cleanly, so acquisition asks for a near-perfect
    #: decode and takes the best over a word's span of offsets.
    ACQUIRE_ERRORS = 2
    ACQUIRE_SCORE = 0.8
    #: At a known phase a word need only decode within the code's power.
    TRACK_ERRORS = 4
    TRACK_SCORE = 0.3
    #: How much better (unanimous votes, plus 20 x score) another alignment
    #: must read to be believed over the one being tracked: fading moves the
    #: count from word to word by as much as noise does.
    SLIP_MARGIN = 4
    SHIFT_MARGIN = 10
    MISSES = 4                                           # A.5.5.2.4

    def __init__(self, window: int = SPS, *, max_dtm_words=60, max_dbm_depth=196) -> None:
        #: Samples of each tone integrated, from its end back.
        self.max_dtm_words = max_dtm_words
        self.max_dbm_depth = max_dbm_depth
        self._block_deadline = 0
        self._tones = np.zeros(8)
        self.window = window
        self.buf = np.zeros(0)
        self.base = 0                                    # sample index of buf[0]
        self.next_window = 0                             # next window start (abs)
        self._freq = 0.0
        self._templates = self._make_templates(0.0, window)
        self._metric = np.zeros(_PHASES)
        self._phase = 0
        self._last_symbol = -SPS                         # abs start of last symbol used
        self._noise = 1e-6
        self._spec = np.zeros(0)
        self._spec_at = 0
        # Soft bits and where each ended.
        self.llr: list[float] = []
        self.hard: list[int] = []
        self.bit_end: list[int] = []
        self._first_bit = 0                              # index of llr[0] in the stream
        self._n = 0                                      # bits seen
        self._sinad: list[float] = []
        # Word reading.
        self._lock: int | None = None                    # bit index of a word end, mod 147
        self._misses = 0
        self._frame: list[Heard] = []
        self._concluded = False
        self._conclusion_words = 0
        #: Words left in a DTM data block (and its CMD CRC): read whatever
        #: they hold, never taken for the frame's end (A.5.7.3).
        self._block = 0
        #: A DBM data block being received: its first and last bits, and
        #: the blocks decoded in this frame (A.5.7.4).
        self._dbm: tuple[int, int, W.Dbm] | None = None
        self._dbm_done: list[W.Dbm] = []
        self._dbm_next: W.Dbm | None = None
        self._pending: tuple[int, Heard] | None = None   # acquisition, awaiting a better phase
        self._pending_since: int | None = None
        #: Linking protection (``lp.View``), and the seeds the frame being
        #: read may be under (B.5.3.2); None before word sync.
        self.lp = None
        self._lp_states: list | None = None
        self._lp_last = 0                                # bit where that word ended
        self._lp_hold: list | None = None                # words waiting on the next
        self._pending_states: list | None = None
        # Words the protocol expects next, as +-1 per on-air bit (0 -> +1).
        self._expect_words: list[int] = []
        self._expect = np.zeros((0, 48))

    # -- tones -------------------------------------------------------------

    @staticmethod
    def _make_templates(freq_hz: float, window: int = SPS) -> np.ndarray:
        t = np.arange(SPS) / FS
        tpl = np.exp(-2j * np.pi * (TONES[None, :] + freq_hz) * t[:, None])  # (64, 8)
        tpl[: SPS - window] = 0
        return tpl

    @property
    def freq_hz(self) -> float:
        return self._freq

    @property
    def locked(self) -> bool:
        return self._lock is not None

    #: How much of a symbol's energy stands in one tone, averaged over about
    #: 130 ms at the best timing, above which ALE signalling is taken to be
    #: present. Noise alone puts about a third there (the largest of eight
    #: like energies); a tone at 0 dB in 3 kHz, over three quarters.
    SIGNAL = 0.5

    @property
    def samples(self) -> int:
        """Samples pushed so far: an event's ``end`` is this many samples
        into the stream, so ``samples - end`` is how late it was reported."""
        return self.base + self.buf.size

    @property
    def signal(self) -> bool:
        """ALE tones and timing present: the fast detection a scanning
        station needs within a dwell (A.5.5.3.2, Tds), well before a word
        can be read."""
        return self.locked or self._pending is not None \
            or (float(self._metric.max()) > self.SIGNAL and np.count_nonzero(self._tones > 0.04) >= 3)

    def push(self, audio: np.ndarray) -> list:
        """Returns the ``Heard`` words and ``Received`` frames this audio
        completed, in order."""
        self.buf = np.concatenate([self.buf, np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)])
        out: list = []
        self._track_frequency()
        end = self.base + self.buf.size
        starts = []
        s = self.next_window
        while s + SPS <= end:
            starts.append(s)
            s += _HOP
        if starts:
            idx = np.array(starts)[:, None] - self.base + np.arange(SPS)[None, :]
            e = np.abs(self.buf[idx] @ self._templates) ** 2          # (n, 8)
            for s0, row in zip(starts, e):
                self._window(s0, row, out)
            self.next_window = starts[-1] + _HOP
        keep = min(self.next_window, self._spec_at) - 4096
        cut = keep - self.base
        if cut > 16384:
            self.buf = self.buf[cut:]
            self.base += cut
        return out

    #: Offsets searched, and how far the comb must stand above its own
    #: valley — the offsets that put it between the tones — to be believed.
    _OFFSETS = np.arange(-124.0, 125.0, 2.0)
    _SEARCH = np.abs(_OFFSETS) <= 80
    _CONTRAST = 2.0

    def _track_frequency(self) -> None:
        """The comb of eight tones in a quarter-second spectrum, smoothed;
        held while a frame is being read."""
        n = 2048
        df = FS / n
        while self._spec_at + n <= self.base + self.buf.size:
            if self._spec_at < self.base:
                self._spec_at = self.base
                continue
            seg = self.buf[self._spec_at - self.base: self._spec_at - self.base + n]
            self._spec_at += n // 2
            if self._lock is not None:
                continue
            p = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
            self._spec = p if self._spec.size == 0 else 0.7 * self._spec + 0.3 * p
            centre = np.rint((TONES[None, :] + self._OFFSETS[:, None]) / df).astype(int)
            comb = sum(self._spec[centre + d].sum(axis=1) for d in (-1, 0, 1))
            j = int(np.argmax(np.where(self._SEARCH, comb, -np.inf)))
            if comb[j] < self._CONTRAST * float(np.min(comb)):
                continue
            if abs(self._OFFSETS[j] - self._freq) >= 6:
                self._freq = float(self._OFFSETS[j])
                self._templates = self._make_templates(self._freq, self.window)

    def _window(self, s0: int, e: np.ndarray, out: list) -> None:
        """One 8 ms window's tone energies; a symbol when it is at the phase."""
        p = (s0 // _HOP) % _PHASES
        tot = float(e.sum()) + 1e-12
        self._metric[p] = 0.94 * self._metric[p] + 0.06 * (float(e.max()) / tot)
        free = self._lock is None and self._pending is None
        if free:
            best = int(np.argmax(self._metric))
            if self._metric[best] > 1.15 * self._metric[self._phase]:
                self._phase = best
        # A symbol starts at least a symbol less a hop after the last one:
        # a timing jump then loses a symbol rather than reading one twice.
        if p != self._phase or s0 < self._last_symbol + SPS - _HOP:
            return
        self._last_symbol = s0
        self._symbol(s0 + SPS, e, out)
        if not free:
            # Reading a word: follow the clocks a millisecond at a time, and
            # only just after a symbol, so the next one comes a symbol less or
            # more a hop later and none is lost. Stepping at the window a
            # symbol was due, the earlier window had already gone by: a symbol
            # was lost, and every later bit read three places out.
            for step in (-1, 1):
                q = (self._phase + step) % _PHASES
                if self._metric[q] > 1.08 * self._metric[self._phase]:
                    self._phase = q
                    break

    def _symbol(self, end: int, e: np.ndarray, out: list) -> None:
        self._tones *= 0.96
        self._tones[int(np.argmax(e))] += 0.04
        if not np.isfinite(self._noise):
            self._noise = 1e-6
        srt = np.sort(e)
        self._noise = 0.995 * self._noise + 0.005 * float(np.mean(srt[:6]))
        n = max(self._noise, 1e-12)
        self._sinad.append(float(srt[-1] / n))
        if len(self._sinad) > 400:
            self._sinad = self._sinad[-400:]
        k = int(np.argmax(e))
        bits = int(BITS_OF_TONE[k])
        for j in range(3):
            z = e[_ZERO[j]].max()
            o = e[~_ZERO[j]].max()
            self.llr.append(float(np.clip((z - o) / n, -30, 30)))
            self.hard.append((bits >> (2 - j)) & 1)
            self.bit_end.append(end)
            self._n += 1
            self._bit(out)
        if len(self.llr) > 4 * W.TRIPLE_BITS:
            drop = len(self.llr) - 2 * W.TRIPLE_BITS
            if self._dbm is not None:                    # keep a DBM block whole
                drop = min(drop, self._dbm[0] - self._first_bit)
            del self.llr[:drop], self.hard[:drop], self.bit_end[:drop]
            self._first_bit += drop

    # -- words ---------------------------------------------------------------

    #: How closely three copies must match an expected word's 48 bits to be
    #: taken for it: at most this many of the combined hard decisions wrong.
    #: Random bits get about 24 wrong, and 6 or fewer about once in a million
    #: tries; a normalised soft correlation was tried first and let noise
    #: through, a few large values carrying it.
    EXPECT_ERRORS = 6
    EXPECT_SCORE = 0.6

    def expect(self, ws: list[int]) -> None:
        """Words the protocol expects next — a reply to a call just sent, an
        acknowledgment to a reply. A.5.2.6.3 lists "history, status,
        expectations, and protocol" among a word's acceptance criteria: a
        word that fails to decode at the frame's own word phase, or is heard
        while waiting to acquire, is compared with these, and taken for the
        one it clearly is. Empty to expect nothing."""
        self._expect_words = list(dict.fromkeys(ws))
        self._expect = np.array([1.0 - 2.0 * W.to_air(w)[:48] for w in self._expect_words]) \
            if self._expect_words else np.zeros((0, 48))

    def _match(self, n: int) -> Heard | None:
        if not self._expect_words or self.lp is not None:
            return None
        i = n - self._first_bit + 1
        if i < W.TRIPLE_BITS:
            return None
        llr = np.array(self.llr[i - W.TRIPLE_BITS:i]).reshape(3, W.WORD_BITS)[:, :48].sum(axis=0)
        tot = float(np.sum(np.abs(llr))) or 1.0
        c = self._expect @ llr / tot
        k = int(np.argmax(c))
        errors = int(np.count_nonzero((llr < 0) != (self._expect[k] < 0)))
        if c[k] < self.EXPECT_SCORE or errors > self.EXPECT_ERRORS:
            return None
        hard = np.array(self.hard[i - W.TRIPLE_BITS:i]).reshape(3, W.WORD_BITS)[:, :48]
        unanimous = int(np.count_nonzero((hard[0] == hard[1]) & (hard[1] == hard[2])))
        return Heard(self._expect_words[k], self.bit_end[i - 1], errors, unanimous, float(c[k]))

    def _read(self, n: int) -> Heard | None:
        """The triple word whose last bit is bit ``n``."""
        i = n - self._first_bit + 1
        if i < W.TRIPLE_BITS:
            return None
        llr = np.array(self.llr[i - W.TRIPLE_BITS:i]).reshape(3, W.WORD_BITS)
        hard = np.array(self.hard[i - W.TRIPLE_BITS:i]).reshape(3, W.WORD_BITS)
        unanimous = int(np.count_nonzero((hard[0] == hard[1]) & (hard[1] == hard[2])
                                         & (np.arange(W.WORD_BITS) < 48)))
        w, ea, eb, score = W.from_air_soft(llr.sum(axis=0))
        if ea < 0 or eb < 0:
            return None
        return Heard(w, self.bit_end[i - 1], ea + eb, unanimous, score)

    @staticmethod
    def _plausible(h: Heard) -> bool:
        """A word that could start a frame someone should read: a routing
        preamble with Basic 38 characters (A.5.2.6.3)."""
        return h.preamble in (W.TO, W.THRU, W.TIS, W.TWAS) \
            and all(c in W.BASIC38 for c in W.chars_of(h.word))

    def _bit(self, out: list) -> None:
        n = self._n - 1
        if self._lock is None:
            self._acquire(n, out)
        elif (n - 1 - self._lock) % W.TRIPLE_BITS == 0:
            self._track(n - 1, out)

    def _acquire(self, n: int, out: list) -> None:
        """A word's first candidate can be read from one copy and two of
        noise; the offset where all three copies are there is the one they
        vote alike at. So a candidate opens a redundant word's wait, and the
        best of it — most unanimous, then best decoded — is locked on."""
        if self._pending is not None and n - self._pending_since >= W.TRIPLE_BITS:
            at, h = self._pending
            self._pending = self._pending_since = None
            self._lock = at % W.TRIPLE_BITS
            self._misses = 0
            cands = self._pending_states
            if cands is not None:
                cands = [c._replace(end=at) for c in cands]
                top = [c for c in cands if c.key[0] == cands[0].key[0]]
                if len(top) > 1:
                    # Two seeds, two plausible words: the next word decides.
                    self._lp_hold = top
                    self._lp_states, self._lp_last = [st for c in top for st in c.seeds], at
                    return
                self._lp_states, self._lp_last = cands[0].seeds, at
                h = cands[0].heard
            self._accept(h, out)
            return
        h = self._read(n)
        states = None
        if self.lp is not None:
            # Protected: the word is plausible under one of the seeds the
            # protocol allows at this time, or it is not a word.
            if h is None or h.errors > self.ACQUIRE_ERRORS or h.score < self.ACQUIRE_SCORE:
                return
            groups: dict[int, _Candidate] = {}
            for st, p in self._lp_open(h, self.lp.initial()):
                if not self._plausible(p):
                    continue
                rank = 4 if p.word in self._expect_words else 2
                g = groups.setdefault(p.word, _Candidate((rank, 0.0), 0, p, [], n))
                g.seeds.append(st)
            if not groups:
                return
            states = sorted(groups.values(), key=lambda g: g.key, reverse=True)
            h = states[0].heard
        elif h is None or not self._plausible(h) \
                or h.errors > self.ACQUIRE_ERRORS or h.score < self.ACQUIRE_SCORE:
            h = self._match(n)
            if h is None:
                return
        if self._pending is None:
            self._pending, self._pending_since = (n, h), n
            self._pending_states = states
        elif (h.unanimous, h.score) > (self._pending[1].unanimous, self._pending[1].score):
            self._pending = (n, h)
            self._pending_states = states

    def _block_word(self, n: int, out: list) -> None:
        """A DTM data block's word: the best decode within a bit's slip, or,
        if none decodes, the bits as read, marked damaged (errors -1)."""
        best = None
        for d in (0, -1, 1):
            h = self._read(n + d)
            if h is not None and (best is None or h.score > best[1].score + 0.05 * (d != 0)):
                best = (d, h)
        if best is None:
            i = n - self._first_bit + 1
            llr = np.array(self.llr[i - W.TRIPLE_BITS:i]).reshape(3, W.WORD_BITS)
            hard = np.array(self.hard[i - W.TRIPLE_BITS:i]).reshape(3, W.WORD_BITS)
            unanimous = int(np.count_nonzero((hard[0] == hard[1]) & (hard[1] == hard[2])
                                             & (np.arange(W.WORD_BITS) < 48)))
            best = (0, Heard(W.from_air_hard(llr.sum(axis=0)), self.bit_end[i - 1], -1,
                             unanimous, 0.0))
        self._lock = (n + best[0]) % W.TRIPLE_BITS
        self._block -= 1
        self._misses = 0
        h = best[1]
        if self.lp is not None and self._lp_states:
            st = self.lp.next(self._lp_states[0])
            self._lp_states, self._lp_last = [st], n + best[0]
            h = self._lp_open(h, [st])[0][1]
        self._frame.append(h)
        out.append(h)

    def _track(self, n: int, out: list) -> None:
        """Read the word expected to end at bit ``n``, allowing a bit's slip
        either way for the sound cards' clocks. While the frame is still in
        its calling cycle the word phase may also be a word out: a scanning
        call repeats one word, which reads cleanly at any of the three
        single-word offsets, and only where the words change does the true
        one (all three copies alike) stand out."""
        if (self._dbm is not None or self._block) and n > self._block_deadline:
            self._end(out)
            return
        if self._dbm is not None:
            start, end, blk = self._dbm
            if n < end:
                return                                   # the block's bits go by
            llr = np.array(self.llr[start - self._first_bit:end - self._first_bit + 1])
            blk.decode(llr)
            if self.lp is not None and self._lp_states:
                # B.5.3.4: the field's 12-bit words were encrypted in pairs.
                states, last = self.lp.dbm_states(self._lp_states[0], blk.depth)
                blk.data_field = list(self.lp.decrypt_field(blk.data_field, states))
                blk.rows = None              # copies under other seeds do not add
                self._lp_states, self._lp_last = [last], end
            self._dbm_done.append(blk)
            self._dbm = None
            self._misses = 0
            return
        if self._block:
            self._block_word(n, out)
            return
        if self.lp is not None:
            self._track_protected(n, out)
            return
        calling = all(h.preamble in (W.TO, W.THRU, W.DATA, W.REP) for h in self._frame)
        shifts = (0, -W.WORD_BITS, -2 * W.WORD_BITS) if calling else (0,)
        found = []
        for base in shifts:
            cand = None
            for d in (0, -1, 1):
                h = self._read(n + base + d)
                if h is None or h.score < self.TRACK_SCORE or h.errors > self.TRACK_ERRORS:
                    continue
                key = h.unanimous + 20 * h.score - (0 if d == 0 else self.SLIP_MARGIN)
                if cand is None or key > cand[0]:
                    cand = (key, base + d, h)
            if cand is not None:
                found.append(cand)
        # A word unread at the tracked phase is a miss, not a reason to move:
        # the other alignments overlap the same faded stretch. Move a word
        # only where the alignments read different words (the true one reads
        # its three copies alike), and clearly.
        best = next((c for c in found if -1 <= c[1] <= 1), None)
        other = max(found, key=lambda c: c[0]) if found else None
        if best is None and other is not None:
            # In the calling cycle a word repeats: another alignment reads it
            # from other copies, which is more redundancy, not a new phase.
            self._misses = 0
            self._accept(other[2], out)
            self._dbm_begin(n + other[1])
            return
        if best is not None and other is not None \
                and other[2].word != best[2].word and other[0] >= best[0] + self.SHIFT_MARGIN:
            best = other
        elif best is not None and other is not None and other is not best \
                and self._frame and best[2].word != self._frame[-1].word \
                and other[0] >= best[0] + self.SHIFT_MARGIN:
            # Where the words change, a phase a copy out still reads the new
            # word, from two of its copies and one of its neighbour's; only
            # the true phase reads all three alike. A repeated word in the
            # calling cycle cannot tell them apart, so this is where the
            # phase is settled.
            best = other
        if best is None:
            for d in (0, -1, 1):
                h = self._match(n + d)
                if h is not None:
                    best = (0.0, d, h)
                    break
        if best is None:
            self._misses += 1
            if self._concluded or self._misses >= self.MISSES:
                self._end(out)
            return
        self._misses = 0
        self._lock = (n + best[1]) % W.TRIPLE_BITS
        self._accept(best[2], out)
        self._dbm_begin(n + best[1])

    def _dbm_begin(self, word_end: int) -> None:
        """A CMD DBM just read, ending at bit ``word_end``: its data block
        starts with the next bit, 24 x depth of them (A.5.7.4)."""
        if self._dbm_next is not None:
            blk, self._dbm_next = self._dbm_next, None
            self._dbm = (word_end + 1, word_end + 24 * blk.depth, blk)

    def _lp_open(self, h: Heard, states: list) -> list:
        """``h`` decrypted under each of ``states``: [(state, plaintext)]."""
        out = []
        for st in states:
            w = self.lp.decrypt(h.word, st)
            out.append((st, Heard(w, h.end, h.errors, h.unanimous, h.score)))
        return out

    def _lp_candidates(self, n: int, states: list, last: int, extra: Heard | None = None):
        """The words that fit as the one ending near bit ``n``, under the
        seeds that follow ``states`` (the word ending at bit ``last``), one
        per plaintext, best first: [key, bit offset, plaintext, seeds, end].
        ``extra``: a word taken as read before it, on trial. Also returns
        the seeds tried."""
        nxt = list(states or [])
        for _ in range(max(1, round((n - last) / W.TRIPLE_BITS))):
            nxt = list(dict.fromkeys(s2 for st in nxt for s2 in self.lp.successors(st)))
        if extra is not None:
            self._frame.append(extra)
        try:
            groups: dict[int, _Candidate] = {}
            for d in (0, -1, 1):
                h = self._read(n + d)
                if h is None or h.score < self.TRACK_SCORE or h.errors > self.TRACK_ERRORS:
                    continue
                prev = self._frame[-1].word if self._frame else None
                for st, p in self._lp_open(h, nxt):
                    if not self._fits(p):
                        continue
                    # A word the protocol expects next fits best (A.5.2.6.3),
                    # then a scanning call's word repeated, then a routing
                    # word or a command table A-XVI knows, then data or an
                    # unknown command (which a wrong seed's garbage makes
                    # more often).
                    rank = 4 if p.word in self._expect_words else 3 if p.word == prev \
                        else 2 if (p.preamble in W.ROUTING or W.is_known_cmd(p.word)) else 1
                    key = (rank, h.unanimous + 20 * h.score - (0 if d == 0 else self.SLIP_MARGIN))
                    g = groups.get(p.word)
                    if g is None or key > g.key:
                        groups[p.word] = _Candidate(key, d, p, [st], n + d)
                    elif key == g.key:
                        g.seeds.append(st)
        finally:
            if extra is not None:
                self._frame.pop()
        return sorted(groups.values(), key=lambda g: g.key, reverse=True), nxt

    def _lp_take(self, c: _Candidate, out: list) -> None:
        self._misses = 0
        self._lock = c.end % W.TRIPLE_BITS
        self._lp_states, self._lp_last = c.seeds, c.end
        self._accept(c.heard, out)
        self._dbm_begin(c.end)

    def _track_protected(self, n: int, out: list) -> None:
        """A protected frame's next word: read at the word phase (within a
        bit's slip), decrypted under each seed the last one's allow, and taken
        under the one it fits the frame under. Unread, the seeds move on
        regardless, as the sender's did.

        Where two seeds give two words that fit equally well, the choice
        waits a word (B.4.4): the one whose seeds' successors read the next
        word is the one. If nothing follows, a conclusion, since that is how
        frames end."""
        if self._lp_hold is not None:
            held, self._lp_hold = self._lp_hold, None
            tried = [(c, self._lp_candidates(n, c.seeds, c.end, extra=c.heard)[0]) for c in held]
            live = [(c, nxt) for c, nxt in tried if nxt]
            if live:
                c = max(live, key=lambda x: (x[1][0].key, x[0].key))[0]
            else:
                c = max(held, key=lambda c: (c.heard.preamble in (W.TIS, W.TWAS), c.key))
            self._lp_take(c, out)
            if not self._frame:                          # the frame ended on it
                return
        cands, nxt = self._lp_candidates(n, self._lp_states, self._lp_last)
        if not cands:
            self._lp_states, self._lp_last = nxt, n
            self._misses += 1
            if self._concluded or self._misses >= self.MISSES:
                self._end(out)
            return
        blocks = any(W.is_dtm(c.heard.word) or W.is_dbm(c.heard.word) for c in cands)
        if len(cands) > 1 and cands[1].key[0] == cands[0].key[0] and not blocks:
            self._lp_hold = [c for c in cands if c.key[0] == cands[0].key[0]]
            self._lp_states, self._lp_last = nxt, n
            return
        self._lp_take(cands[0], out)

    def _fits(self, h: Heard) -> bool:
        """Whether a word can be what comes next (A.5.2.6.3's acceptable
        characters, read in context): address words carry Basic 38
        characters, and an AMD's words the expanded 64 set."""
        p, chars = h.preamble, W.chars_of(h.word)
        basic = all(c in W.BASIC38 for c in chars)
        if p in W.ROUTING:
            return basic
        if p == W.CMD:
            return True
        # DATA and REP continue whatever came before them.
        prev = next((x for x in reversed(self._frame) if x.preamble not in (W.DATA, W.REP)), None)
        if prev is None:
            return False
        if prev.preamble == W.CMD:
            return not W.is_amd(prev.word) or W.is_amd(h.word)
        return basic

    def _accept(self, h: Heard, out: list) -> None:
        p = h.preamble
        if not self._fits(h):
            if self._concluded:
                self._end(out)
            else:
                self._misses += 1
                if self._misses >= self.MISSES:
                    self._end(out)
            return
        if self._concluded:
            first = self._frame[-self._conclusion_words]
            if p == first.preamble and h.word == first.word:
                self._conclusion_words = 1          # a sound starting over
            elif p in (W.DATA, W.REP) and self._conclusion_words < 5:
                self._conclusion_words += 1
            else:
                self._end(out, leftover=h)
                return
        elif p in (W.TIS, W.TWAS):
            self._concluded = True
            self._conclusion_words = 1
        elif W.is_dtm(h.word):
            d = W.dtm_fields(h.word)
            if d["words"] > self.max_dtm_words:
                self._end(out)
                return
            if d["words"]:
                self._block_deadline = self._n + (d["words"] + 2) * W.TRIPLE_BITS
                self._block = d["words"] + 1          # and its CMD CRC
        elif W.is_dbm(h.word):
            d = W.dbm_fields(h.word)
            if d["depth"] > self.max_dbm_depth:
                self._end(out)
                return
            if d["depth"]:
                self._block_deadline = self._n + d["depth"] * 24 + W.TRIPLE_BITS
                self._dbm_next = W.Dbm(d["kind"], d["kd4"], d["kd3"], d["kd2"], d["kd1"],
                                       d["depth"], d["bits"])
        self._frame.append(h)
        out.append(h)

    def _end(self, out: list, leftover: Heard | None = None) -> None:
        if self.lp is not None and len(self._frame) < 2:
            # Protected, one word is no evidence: some seed or other makes
            # a plausible word of one in thirteen. Word sync is the second
            # word checking under that seed's successors (B.4.4).
            self._frame = []
        if self._frame:
            f = W.parse([h.word for h in self._frame],
                        {i for i, h in enumerate(self._frame) if h.errors < 0})
            done = iter(self._dbm_done)
            f.dbm = [next(done, d) if d.depth else d for d in f.dbm]
            ber = float(np.mean([48 - h.unanimous for h in self._frame]))
            s = np.mean(self._sinad[-3 * 49 * len(self._frame):]) if self._sinad else 1.0
            sinad = 10 * np.log10(max(s, 1.0))
            first = self._frame[0].end - int(W.TRIPLE_BITS / 3 * SPS)
            seed = None
            if self.lp is not None and self._lp_states:
                seed = ("fine", "coarse", "clear")[self._lp_states[0].kind]
            out.append(Received(f, self._frame, first, self._frame[-1].end, ber,
                                float(sinad), self._freq, seed))
        self._frame = []
        self._concluded = False
        self._conclusion_words = 0
        self._block = 0
        self._dbm = self._dbm_next = None
        self._dbm_done = []
        self._lp_states = None
        self._lp_hold = None
        self._misses = 0
        if leftover is not None and self._plausible(leftover):
            # A new frame at the same phase (a collision or a quick reply).
            self._accept(leftover, out)
        else:
            self._lock = None


class DualReceiver:
    """Two receivers on the same audio, integrating a whole tone and its last
    three quarters, and the frames of both, once each.

    The whole tone is the better in white noise, by about 2 dB where frames
    begin to be lost. On table A-II's poor channel, 2.2 ms of echo is a
    quarter of a tone: integrated over the whole of it, the previous tone
    arrives by the late path, and where the paths cancel at the present
    tone's frequency the previous one is read in its place, confidently
    (70-78% of frames read at any SNR). Over the last three quarters,
    88-95%.
    """

    WINDOWS = (SPS, 3 * SPS // 4)

    def __init__(self, **limits) -> None:
        self.parts = [Receiver(window=w, **limits) for w in self.WINDOWS]
        self._recent: list[tuple[tuple, int]] = []
        self._held: list[list] = []

    @property
    def freq_hz(self) -> float:
        return self.parts[0].freq_hz

    @property
    def locked(self) -> bool:
        return any(p.locked for p in self.parts)

    @property
    def signal(self) -> bool:
        return any(p.signal for p in self.parts)

    @property
    def samples(self) -> int:
        return self.parts[0].samples

    def expect(self, ws: list[int]) -> None:
        for p in self.parts:
            p.expect(ws)

    def set_lp(self, view) -> None:
        """Linking protection (``lp.View``), or None."""
        for p in self.parts:
            p.lp = view

    #: A frame carrying a DTM block waits this long for the other window's
    #: copy of it, to be merged word by word.
    HOLD = 2 * (W.TRIPLE_BITS // 3) * SPS

    def push(self, audio: np.ndarray) -> list:
        out: list = []
        for i, part in enumerate(self.parts):
            for o in part.push(audio):
                if isinstance(o, Received):
                    key = (tuple(o.frame.to), o.frame.caller, o.frame.terminator,
                           tuple(o.frame.amd), len(o.frame.dtm), len(o.frame.dbm))
                    near = [(k, e) for k, e in self._recent
                            if k == key and abs(e - o.end) < 3 * W.TRIPLE_BITS // 3 * SPS]
                    held = next((h for h in self._held if h[0] == key
                                 and abs(h[1].end - o.end) < 3 * W.TRIPLE_BITS // 3 * SPS), None)
                    if held is not None:
                        held[1] = _merged(held[1], o)
                        continue
                    if near:
                        continue
                    self._recent = [(k, e) for k, e in self._recent if o.end - e < FS * 10]
                    self._recent.append((key, o.end))
                    if any(d.words for d in o.frame.dtm) or any(d.depth for d in o.frame.dbm):
                        self._held.append([key, o])
                    else:
                        out.append(o)
                elif i == 0 or not self.parts[0].locked:
                    out.append(o)
        now = self.samples
        for h in [h for h in self._held if now - h[1].end >= self.HOLD]:
            self._held.remove(h)
            out.append(h[1])
        out.sort(key=lambda o: o.end)
        return out


def _merged(a: Received, b: Received) -> Received:
    """Two windows' readings of one frame, word by word: where one could not
    decode a DTM data word and the other could, the other's. Aligned on the
    first CMD DTM word each read. A DBM block: the one that passed its CRC."""
    if len(a.frame.dbm) == len(b.frame.dbm):
        a.frame.dbm = [y if (not x.ok and y.ok) else x for x, y in zip(a.frame.dbm, b.frame.dbm)]

    def first_dtm(r):
        return next((i for i, h in enumerate(r.words) if h.errors >= 0 and W.is_dtm(h.word)), None)
    ia, ib = first_dtm(a), first_dtm(b)
    if ia is None or ib is None:
        return a
    words = list(a.words)
    for k in range(ia, len(words)):
        j = k - ia + ib
        if words[k].errors < 0 and 0 <= j < len(b.words) and b.words[j].errors >= 0:
            words[k] = b.words[j]
    f = W.parse([h.word for h in words], {i for i, h in enumerate(words) if h.errors < 0})
    f.dbm = a.frame.dbm
    return Received(f, words, a.start, a.end, a.ber, a.sinad_db, a.freq_hz, a.seed)
