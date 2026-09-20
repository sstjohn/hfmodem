# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-1 receiver: FSK audio in -> connect callsign / data text out.

This inverts shrike's own PACTOR-1 connect encoder (`shrike.pactor1`), and with it
the connect parser any receiver runs on the same burst. Two layers:

  * `decode_connect(audio)` -- recover the link-setup callsign and its variant
    (Normal / Longpath) from a connect burst. This is the burst a Winlink gateway
    sends when it answers a connect, so it directly answers "did the gateway reply,
    and as whom". The Normal path is anchored to ground truth: byte-exact on the
    real off-air DL6MAA capture, and agreeing with what an independent decoder
    makes of the same audio (###CONNECT: [Normal Call: DL6MAA]). The Longpath
    discrimination is validated against shrike's own encoder round-trip only --
    no off-air Longpath capture yet -- so trust it less.

  * `decode_call_b(audio)` -- the other connect frame, SCS's Robust Call and the
    two Free Signals. 11 bytes at 100 Bd with a CRC-16 and a 6-bit packed address,
    sharing nothing with the frame above but its tones. It carries its own tone
    search, because a caller keying one is on its own dial, not ours.

  * `decode_p1(audio)` -- PACTOR-1 ARQ data frames -> text. The FSK front end and
    timing recovery are the same as the connect path; the frame layer (packet
    length, status byte, CRC-16 gate) follows ITU-R M.1798. See the note on
    `decode_p1` for its validation status.

FSK, per the encoder and the real capture: MARK(bit1)=1400 Hz, SPACE(bit0)=1600 Hz
(inverted polarity), LSB-first, continuous phase. The connect is dual-rate: a
9-byte address section at 100 Bd (primary copy) followed by a 6-byte redundancy
section at 200 Bd (the Memory-ARQ secondary copy).

The parser rotates each on-air byte  T[i] = (raw[i]>>1) | (bit0(raw[i+1])<<7),
recovering the address image T = [0x55 sync] + ASCII callsign + 0x0F fill; the
callsign is T[1] up to the 0x0F terminator, each char validated in (0x2C, 0x5A].
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import NamedTuple

import numpy as np

from . import coding, pactor1, session, spec

FS = pactor1.FS
MARK = pactor1.MARK          # 1400 Hz, bit 1
SPACE = pactor1.SPACE        # 1600 Hz, bit 0
SYNC = (0x55, 0xAA)          # 0xAA = the whole frame inverted (FSK polarity ambiguity)
TERM = 0x0F
ADDR_LEN = pactor1.ADDR_LEN  # 9 bytes @ 100 Bd
RED_LEN = pactor1.RED_LEN    # 6 bytes @ 200 Bd


class CS(NamedTuple):
    """A control signal as read off the air: which word, how well, in which shift.

    `sense` is 0 where the twelve bits matched the codeword as tabulated and 1
    where they matched its complement -- the peer's Shiftlage for the cycle the
    burst belongs to, which by docs/protocols/pactor/pactor1-data-packets.md §7 is also the shift it
    expects OUR transmission in that same cycle to arrive in.
    """
    index: int
    errors: int
    sense: int

    @property
    def unassigned(self) -> bool:
        """This is one of the words PACTOR-1 gives no meaning (`pactor1.CS_WORDS`).

        Every consumer of an index reads it as an acknowledgement, a break-in or a
        speed change, and none of those is what arrived. A caller that hands one on
        anyway is acting on a word it has only recognised.
        """
        return self.index >= len(pactor1.CONTROL_SIGNALS)


# --- FSK front end ---------------------------------------------------------

def _sliding_bin(samples: np.ndarray, n: int, freq: float) -> np.ndarray:
    """The complex DFT bin at `freq` for every length-`n` window, by window start.

    A running complex heterodyne sum, so one pass answers every symbol alignment
    at once. Same construction as besra's dsp.detect.sliding_bin -- the flock has
    one implementation of this idea and it belongs in the shared hfmodem layer if
    it is ever hoisted, not forked a second time here.
    """
    het = samples * _lo(samples.size, freq)
    acc = np.concatenate(([0.0], np.cumsum(het)))
    return acc[n:] - acc[:-n]


def _lo(n: int, freq: float) -> np.ndarray:
    """The heterodyne exp(-2j*pi*freq*k/FS) for k in [0, n).

    An integer-Hz tone at an integer sample rate repeats exactly every FS/gcd
    samples -- 240 for MARK, 30 for SPACE -- so one period is built and tiled. The
    full-length `np.exp` this replaces was 89% of the sliding DFT's cost and was
    recomputed for every buffer the front end was handed.
    """
    if freq != int(freq):
        return np.exp(-2j * np.pi * freq / FS * np.arange(n))
    period = FS // gcd(int(freq), FS)
    one = np.exp(-2j * np.pi * freq / FS * np.arange(period))
    return np.tile(one, -(-n // period))[:n]


class _Tones:
    """MARK/SPACE magnitudes for every symbol-aligned offset in one buffer.

    Dotting each candidate symbol against the two references as the search walks it
    correlates the same audio over and over, millions of times per capture. Both
    series are sliding single-bin DFTs, so they are computed ONCE here, in two O(N)
    passes, and a bit decision becomes an array lookup.

    Held per buffer and passed down, never cached globally: keying on array
    identity across a streaming front end is how a later block gets served stale
    magnitudes.
    """

    __slots__ = ("mark", "space", "env", "n")

    def __init__(self, audio: np.ndarray, sps: int,
                 mark: float = MARK, space: float = SPACE):
        m = np.abs(_sliding_bin(audio, sps, mark))
        s = np.abs(_sliding_bin(audio, sps, space))
        self.mark, self.space, self.env, self.n = m, s, m + s, m.size

    def fits(self, starts: np.ndarray, sps: int, nbits: int) -> np.ndarray:
        """Which of `starts` have all `nbits` of their symbol windows in range."""
        return (starts >= 0) & (starts + (nbits - 1) * sps < self.n)

    def read(self, starts: np.ndarray, sps: int, n: int,
             invert: bool) -> np.ndarray:
        """`n` LSB-first bytes at each of `starts` -> (k, n) uint8.

        The searches visit tens of thousands of candidate alignments between them,
        and one gather answers all of them. `starts` must already have passed
        `fits`; nothing here is bounds-checked.
        """
        idx = starts[:, None] + np.arange(n * 8) * sps
        b = (self.mark[idx] > self.space[idx]) ^ invert
        return np.packbits(b.reshape(-1, n, 8), axis=2, bitorder="little")[:, :, 0]


def tone_series(audio: np.ndarray, fs: int = FS) -> "_Tones":
    """The 100 Bd MARK/SPACE magnitudes for `audio`, for a caller about to hand the
    same buffer to both PACTOR-1 decoders.

    `decode_connect` and `decode_p1_packets` each build this when it is not
    supplied, and a front end that tries the connect decode and then the packet
    scan on one hop of audio would otherwise pay for the pass twice. The sharing is
    at the call site rather than in a cache inside `_Tones`, because keying on array
    identity across a streaming front end is how a later block gets served stale
    magnitudes.
    """
    return _Tones(_to_fs(audio, fs), FS // 100)


def _rotate(raw: np.ndarray) -> np.ndarray:
    """The per-byte cross-rotate T[i] = (raw[i]>>1) | (bit0(raw[i+1])<<7), applied
    to every row of a (k, n) uint8 array."""
    nxt = np.zeros_like(raw)
    nxt[:, :-1] = raw[:, 1:] & 1
    return (raw >> 1) | (nxt << 7)


def _locked(T: np.ndarray) -> np.ndarray:
    """Which rows of a (k, n) uint8 address image are a lock: T[0] is the sync and
    T[2:] is a valid character run ending in the 0x0F terminator. T[1] is left free
    so a Longpath (bit-inverted first character) still locks."""
    body = T[:, 2:]
    stop = (body <= 0x2C) | (body > 0x5A)      # the terminator is itself a stop
    first = body[np.arange(len(T)), np.argmax(stop, axis=1)]
    # PRECONDITION, not an observation: this finds the first stop and asks whether
    # it IS the terminator, so it is correct only while TERM lies outside the valid
    # character range -- 0x0F <= 0x2C, so it does. The scalar form this replaced
    # tested for TERM *before* validity and held for any TERM. If TERM ever moves
    # inside the range, this silently stops locking rather than failing loudly.
    return ((T[:, 0] == SYNC[0]) | (T[:, 0] == SYNC[1])) & (first == TERM)


def _redundancy_first(audio: np.ndarray, start: int, sps_r: int,
                      invert: bool) -> int | None:
    """The true first callsign character, from the 200 Bd redundancy section.

    Only 48 symbol windows are read, and the sliding DFT costs what it transforms
    rather than what is read out of it -- so it is taken over that slice alone.
    Running it across the whole buffer, to index the tail of whichever lock the
    address section found, was most of the connect decoder's arithmetic.
    """
    span = RED_LEN * 8 * sps_r
    seg = audio[start:start + span]
    if seg.size < span:
        return None
    raw = _Tones(seg, sps_r).read(np.zeros(1, np.int64), sps_r, RED_LEN, invert)
    return int(_rotate(raw)[0, 0]) >> 1


def _onsets(tones: "_Tones", sps: int, limit: int = 8) -> list[int]:
    """Candidate burst starts: every rising edge of the FSK-band envelope.

    Taking only the FIRST crossing (the earlier behaviour) locks onto whatever
    energy happens to open the window -- noise, a neighbouring signal, the tail of
    a previous burst -- and since the caller only searches a couple of symbols
    either side of it, a real connect a moment later is never found. Measured on a
    known-real capture that failed in 27 of 30 window alignments. Every rising edge
    is offered instead, strongest first; a wrong one merely fails to lock.
    """
    idx = np.arange(0, max(1, tones.n), sps // 4)
    env = tones.env[idx]                    # already computed for this buffer
    if env.max() <= 0:
        return [0]
    hot = env > 0.3 * env.max()
    edges = np.flatnonzero(hot & ~np.concatenate(([False], hot[:-1])))
    edges = edges[np.argsort(-env[edges], kind="stable")]
    return [int(idx[k]) for k in edges[:limit]] or [0]


def _sync_starts(tones: "_Tones", lo: int, hi: int, sps: int,
                 invert: bool) -> np.ndarray:
    """Of every start in [lo, hi), those whose rotated first byte is the sync.

    T[0] is built from only the first 9 on-air bits, so this rejects almost every
    candidate for a fraction of the work of reading the whole address section --
    and it is done for all candidates at once, because the alternative is a Python
    call per position and the search visits a quarter-million of them.
    `_locked` demands the same sync, so this changes nothing but speed.
    """
    starts = np.arange(max(0, lo), max(0, hi), dtype=np.int64)
    starts = starts[tones.fits(starts, sps, 9)]
    if starts.size == 0:
        return starts
    idx = starts[:, None] + np.arange(9) * sps
    b = (tones.mark[idx] > tones.space[idx]) ^ invert
    v = np.packbits(b[:, :8], axis=1, bitorder="little")[:, 0]
    t0 = (v >> 1) | (b[:, 8].astype(np.uint8) << 7)
    return starts[(t0 == SYNC[0]) | (t0 == SYNC[1])]


def _valid_char(b: int) -> bool:
    return 0x2C < b <= 0x5A


# --- connect decode --------------------------------------------------------

@dataclass
class Connect:
    variant: str          # "Normal" | "Longpath" | "Robust" | "FreeSignal..."
    callsign: str
    inverted: bool        # global FSK polarity was inverted (sync decoded as 0xAA)

    def report(self) -> str:
        return f"###CONNECT: [{_LABEL.get(self.variant, self.variant + ' Call')}:" \
               f" {self.callsign}]"


_LABEL = {"FreeSignalNormal": "Free Signal Normal",
          "FreeSignalEncrypted": "Free Signal Encrypted"}
"""How a monitor spells the two branch-B kinds that are not `<variant> Call`."""


CS_BIT_S = 0.010
"""One control-signal bit at 100 Bd. Confirmed off-air rather than assumed: the
transition spacing inside a real burst measures 9.8-10.2 ms and 19.8-20.2 ms,
i.e. exactly one and two bits, read by instantaneous frequency."""


def _cs_trace(audio: np.ndarray, lo: int, hi: int, fs: int
              ) -> tuple[np.ndarray, np.ndarray] | None:
    """(tone decisions, envelope) over `audio[lo:hi]`, sample by sample.

    Instantaneous frequency, so a transition can be located to a sample. That is
    what a timing-error estimator wants and what a burst-position search wants;
    it is NOT what the bit reader wants, and `cs_bits` no longer uses it. A
    symbol-window magnitude comparison smears across transitions, which is
    exactly why it beats a point sample of this once the window is placed
    correctly: it integrates the whole bit instead of betting on one instant.
    """
    from scipy.signal import butter, filtfilt, hilbert

    if lo < 0 or hi > audio.size or hi - lo < int(0.004 * fs):
        return None
    b, a = butter(4, [1200 / (fs / 2), 1800 / (fs / 2)], btype="band")
    z = hilbert(filtfilt(b, a, audio[lo:hi].astype(float)))
    fi = np.diff(np.unwrap(np.angle(z))) / (2 * np.pi) * fs
    k = int(0.0015 * fs)
    fi = np.convolve(fi, np.ones(k) / k, "same")
    return (fi > (MARK + SPACE) / 2).astype(int), np.abs(z)[:len(fi)]


def cs_time_dev(audio: np.ndarray, t0: float, fs: int = FS) -> float | None:
    """How far a control signal's bit transitions sit from a grid starting at `t0`.

    Seconds, signed, positive meaning the burst is LATE against the prediction.
    This is the error signal of the master's receive-window loop, and it is the
    only thing in the link a master is allowed to steer on
    (docs/protocols/pactor/pactor1-timing.md §1). It is fed on every receive attempt,
    decoded or not, so the loop keeps trimming through a fade instead of
    freezing (§4).

    `t0` is a PREDICTION -- where the grid says the burst begins -- and that is
    what separates this from the transition-derived bit phase that was tried as a
    decode origin and measured WORSE than the envelope onset on the corpus
    anchors. That one had to be right on its own, first time, for twelve bits.
    This one is a residual against an anchor that already exists, and an eighth
    of it is applied per cycle, so a noisy measurement costs a millisecond rather
    than a codeword.

    Bounded to +/- half a bit by the wrap, where the reference implementation's
    estimator searches +/- a whole bit around each expected transition. The
    difference only bites past a half-bit error, and by then the codeword is
    already unreadable at the fixed offset a link is read at -- "+/-10 ms is a
    whole bit slip and is unrecoverable" (§5). A caller that wants to know about
    a slip that big must get it from the burst detector, not from here.
    """
    step = CS_BIT_S * fs
    span = pactor1.CS_BITS * step
    # CLIPPED to the buffer rather than refused. Three bits of margin is what
    # keeps the whole burst inside the bracket whatever the prediction, but a
    # burst near the end of a per-cycle capture cannot have it -- and half the
    # bursts in captures/eoc_fix and captures/eoc_even are that burst. Losing the
    # margin costs a transition or two off one end; refusing costs the cycle's
    # whole correction.
    lo, hi = max(0, int(t0 * fs - 3 * step)), min(audio.size,
                                                  int(t0 * fs + span + 3 * step))
    if hi - lo < span:
        return None
    trace = _cs_trace(audio, lo, hi, fs)
    if trace is None:
        return None
    sym, env = trace
    origin = int(t0 * fs) - lo
    edges = np.flatnonzero(np.diff(sym)) + 1
    # Gated on the ENVELOPE, not on the predicted span. The tone decision is a
    # threshold on instantaneous frequency, which in silence reads whatever the
    # noise reads, so some gate there must be -- but gating on the PREDICTION
    # makes the set of transitions move with the prediction, and a window edge
    # that admits or drops one transition then shifts the answer by up to 1.3 ms.
    # Three bits of margin either side, so the whole burst is inside the bracket
    # whatever the prediction, and only the reference moves.
    #
    # It does not come out exactly shift-invariant even so -- the bracket moves,
    # so the filter's edge effects and `env.max()` move with it, and a transition
    # sitting on the 0.4 threshold changes sides. Measured over five prediction
    # shifts on captures/provoke2 and captures/w6ids_silent, the worst residual is
    # 1.2 ms and it is the noisier recording that produces it. An eighth of that
    # per cycle is 0.15 ms against a +/-2.5 ms clean band, so it is left where it
    # is; a gate at 0.6, and trimming the first and last transition, were both
    # tried and neither improves both recordings.
    edges = edges[env[edges] > 0.4 * env.max()]
    if edges.size == 0:
        return None
    # CIRCULAR mean, because the quantity is circular: a transition 4.9 ms late
    # and one 4.9 ms early are 0.2 ms apart on a bit, not 9.8, and averaging the
    # wrapped values arithmetically puts the answer between them at zero -- which
    # made this read 4.5 ms off when the prediction moved by 3.
    phase = np.exp(2j * np.pi * (edges - origin) / step).mean()
    return float(np.angle(phase)) / (2 * np.pi) * step / fs


CS_SEARCH_S = 0.045
"""How far outside the burst detector's run the twelve bits are looked for.

`rxfront._p1_cs_bursts` reports run edges from a spectral profile against an
energy threshold, and it is not a bit-grid estimator. Measured against the origins
that decode, over 41 bursts from four stations, its run START sat 5 to 36 ms early
-- up to three and a half bits. That was on the old profile, whose edges were
stamped at the leading sample of a 20 ms window; centre-stamping moved every start
about 10 ms later and so shrank the early bias by that much. Four and a half bits
of guard covers either with
margin; the search is bounded rather than open because a wider one is free to walk
onto a neighbouring transmission. 30 and 60 ms were measured alongside and moved
the decode count by two bursts in 142 either way, so this is not a tuned constant
and should not be treated as one.
"""

CS_TAIL_BITS = 2
"""Bit slots past the word whose energy is subtracted from the alignment score.

The station keys a lead-in BEFORE the data bits and drops the carrier at the END
of them (docs/protocols/pactor/pactor1-timing.md sec 3). A steady lead-in tone is
indistinguishable from a data bit under an eye-opening score -- both are one pure
tone for a whole bit -- so on a station that sends one, a window placed two bits
early scores the same as the right one and the tie goes to noise. W6IDS does send
one: its runs measure 145 ms against WS8EOC's 115, and its true origin sits a
consistent 19 ms into the run.

The unkey is the asymmetry that breaks the tie. Requiring the two slots after the
word to be QUIET costs a wrongly-early window two bits of real signal and costs
the right one nothing. Measured over the same 41 bursts, against the alignment
that decodes: no tail term, mean -5.3 ms and sd 9.4; one slot, -1.1 and 5.7; two,
+0.5 and 2.9, with 41 of 42 inside half a bit. A third slot was measured too and
gained one decode in 142 -- inside the noise, and 30 ms past the word starts to
reach into the turnaround, so two it is.
"""


CS_ANCHOR_S = 0.005
"""How far a codeword may sit from the instant A PEER reads at and still be read.

Half a bit at 100 Bd, and this is a claim about the far end rather than about
this receiver -- `onair.RadioTx._ack_gap_line` is its only consumer, and what it
asks is whether an acknowledgement we are about to key will be inside the twelve
bit periods the station receiving it latches. The one written record of that read
is `receive_cs` at hfkernel/fsk/pactor.c:745, which reads at a point with no
search, and replay against it is a cliff and not a margin: 100% inside 3 ms, 59%
at 4 ms, nothing at 6, and 14 dB more signal buys none of it back
(`tests.shrike.test_ackplace`).

IT IS NOT THIS READER'S SEARCH WIDTH, and was until 2026-08-28. Nothing we hold
tests the far end's tolerance -- 149 acknowledgements in the whole record, the
largest 2.1 ms out, so no codeword of ours has ever been placed outside it
(`test_ackoff`) -- while our own reach is measurable and was measured, at
`CS_SEARCH_HALF_S`. One constant cannot be both an assumption about somebody
else's receiver and a property of ours."""

CS_SEARCH_HALF_S = 0.015
"""How far either side of a PREDICTED instant this receiver looks for the word.

Three half-bits, and the width is the operating point of a curve rather than a
tolerance. The curve was swept from 5 to 145 ms against 20 measured answers of a
live gateway, the same answers read from a displaced anchor, and 16 201 twelve-bit
reads of quiet audio from nine recordings; `tests.shrike.test_cssearch` holds the
point it landed on.

WHAT THE PREDICTION IS ACTUALLY WORTH. Across 755 tracked cycles in 45 sessions
the burst the grid was steering on landed further than half a bit from the
predicted instant in 23% of them, further than 10 ms in 8.7%, and never further
than 21.4 ms -- an independent instrument, the burst detector, whose own onset
sits within 5 ms of the codeword in 56 of 62 corpus bursts. Half a bit was never
the error budget; it was the budget the reader could position within.

WIDENING WITHOUT SEARCHING IS NOT WIDENING. The reader used to take the
best-scoring alignment and decode it once, and over the 20 measured answers that
holds 14 at 5 ms, 13 at 7.5 and 3 at 10: past half a bit the eye score cannot say
which bit period the word is in and it wanders off. So the window widened only
once the reader decoded every alignment in it and took the best-scoring one that
IS a codeword, which is what the 1990 description has its master do over the
0.29 s `Fenster fuer Kontrollsignalempfang` (`pactor.txt:159`, `:272-274`).

WHY IT STOPS HERE AND NOT AT THE DESCRIPTION'S WINDOW. The searched reader holds
every real answer at every width tried, and adds six the point read missed --
20 of 20, all six on the cycle parity and inside the session's own CS1-then-CS4
partition. What widening costs is false locks, strictly linearly: 1.54 per 1000
quiet reads at 5 ms, 3.58 here, 4.44 at 20, 23.95 at 145, against 1.11 for the
point read. And at 20 ms two of the twenty answers stop being themselves -- the
CS3 that every CS4 casts two bits along wins the score at exactly +20 ms. So the
knee is a WRONG WORD and not a rate, it sits at 20 ms, and 15 leaves half a bit
in front of it. The description's window is not free and was not taken.

The cost is stated rather than argued: a phantom every 280 silent cycles where
the point read manufactures one every 900, and 26 of those 58 phantoms are the
break-in. Nothing here is a recording of a peer answering off-anchor, which the
record does not hold and which is the only thing that would settle what a real
station does with one."""

CS_SEARCH_EYE_MIN = 0.35
"""The eye every searched codeword must clear, assigned or not.

A search is many trials where the point read was one, and `CS_SPLIT_MIN`'s margin
was measured at one. This is the frame decoder's own statistic asked of the same
twelve symbols -- `_eye`, as `cs_head` asks it -- and it is what the extra trials
are paid for with: over the 16 201 quiet reads it takes a 15 ms search from 11.91
false locks per 1000 to 3.58.

IT DOES NOT SEPARATE THE TWO POPULATIONS AND IS NOT SET AS IF IT DID. Real
codewords in the corpus read 0.368 to 0.853 over 62 bursts of seven sessions,
and the quiet reads that manufacture one reach 0.910. 0.35 is under the weakest
real word measured and no further under it than that; 0.40 would drop two of the
62. `EYE_MIN` still applies on top for an unassigned word, which drives nothing
and may spend none of this budget."""

CS_SPLIT_MIN = 0.70
"""How far apart the two halves of the word must come out to be believed.

`_cs_decide` splits twelve slots into six and six because every control signal has
weight 6, and a split always exists -- noise splits too. This is the eye of that
decision: the gap between the sixth and seventh sorted decision statistics over
their own mean magnitude. A real word separates its halves by the burst's whole
tone amplitude; noise separates them by the width of its own scatter.

Measured over 2169 twelve-bit reads of quiet audio from seven sessions of the same
station, phantom codewords -- noise landing on one of the four in one of the two
senses, which happens once in 512 reads -- score at most 0.64 and 0.20 at the
median. WS8EOC's 24 real answers score 0.97 or better in 16 of the 20 that carry a
codeword. The gate takes the phantoms to NONE in 2169 reads and costs four of the
twenty.

That trade is the right way round on an established link. A control signal that is
not decoded costs one retry and the peer sends it again; a control signal invented
from noise advances the sequence past a packet the peer never received, or takes a
changeover that was never offered, and nothing on the air breaks it."""


def _cs_decide(mark: np.ndarray, space: np.ndarray
               ) -> tuple[np.ndarray, float]:
    """Twelve bit decisions from the two tone magnitudes, and how open the split is.

    A SPLIT, not `mark > space`. A comparison assumes the two tones arrive at the
    same amplitude, and on the air they do not: over WS8EOC's answers of 2026-07-30 the
    1400 Hz tone measured 2 to 19 dB above the 1600 Hz one, median 9, varying cycle
    to cycle within one 30 s session and inverting in one of them. The noise floor
    across the same 200 Hz is flat to within a couple of dB, so this is the signal
    and not the receiver, and it is what a selective fade at HF looks like. Under
    it, a plain comparison reads ten or eleven of the twelve slots as mark and no
    alignment repairs that.

    What survives the imbalance is the CODE's own structure: all four control
    signals have Hamming weight 6, so exactly six slots carry each tone and the
    decision is a SPLIT rather than a threshold. Sorting the twelve differences and
    cutting at the median needs no knowledge of either tone's level -- it reads a
    burst whose space tone is buried under the noise floor as easily as a balanced
    one, because the mark tone's presence and absence alone carries the pattern.

    The split is not free: it can only ever return a weight-6 word, and 8 of the
    924 weight-6 words are a control signal in one of the two senses, against 8 in
    4096 for an unconstrained read. `CS_SPLIT_MIN` is what pays for that, and the
    returned score is what it gates on.
    """
    d = np.asarray(mark, dtype=float) - np.asarray(space, dtype=float)
    order = np.sort(d)
    n = d.size // 2
    scale = max(float(np.abs(d).mean()), 1e-12)
    return (d > np.median(d)).astype(int), float((order[n] - order[n - 1]) / scale)


def cs_bits(audio: np.ndarray, t0: float, dur: float, fs: int = FS,
            *, slip: int = 0, extra: int = 0) -> list[int] | None:
    """The on-air bits of a control-signal burst at `t0`, in TIME order.

    The twelve-bit window is POSITIONED, by sliding it over the detector's run and
    keeping the alignment where the eye opens widest and the two slots after the
    word are quietest. This is not the codeword search `acquire_control_signal`
    does and must not be confused with it: the score never looks at the codeword
    table, so it cannot manufacture a match, and exactly ONE alignment reaches the
    decoder. The distance-8 margin is therefore still spent on one trial.

    THIS IS THE READER FOR A BURST A DETECTOR FOUND, and `CS_SEARCH_S` is the
    guard that costs. A station holding a link already knows roughly where the
    word is and wants `cs_anchored`, which searches three half-bits either side of
    that instead of four bit periods either side of a run start.

    What it replaced read twelve bits at the detector's own burst start, and a
    burst start is not a bit grid. That cost a systematic 3-5 bit errors on strong,
    clean, correctly detected bursts -- sampling near transitions rather than at
    bit centres -- and every acknowledgement on an established link with it.

    `slip` moves the grid by whole bits and `extra` asks for bits past the
    codeword; neither is used by the decoder. They exist for the alignment studies
    in `analysis/p1cs/`.
    """
    sps = FS // 100
    lo = max(0, int((t0 - CS_SEARCH_S) * fs))
    hi = min(audio.size, int((t0 + dur + CS_SEARCH_S) * fs))
    x = _to_fs(np.asarray(audio[lo:hi], dtype=float), fs)
    if x.size < pactor1.CS_BITS * sps:
        return None
    # Audio we do not have counts as silence, which is what the end of a
    # transmission sounds like. Without the padding the last few alignments would
    # have no tail slot to be charged for and would win by default; with it, a
    # capture that stops mid-burst simply scores badly and fails, which is the
    # honest answer. It is also what leaves at least one candidate alignment
    # whenever the word itself fits.
    T = _Tones(np.concatenate([x, np.zeros(CS_TAIL_BITS * sps)]), sps)
    starts = np.arange(T.n - (pactor1.CS_BITS + CS_TAIL_BITS - 1) * sps)
    eye = starts[:, None] + np.arange(pactor1.CS_BITS) * sps
    tail = starts[:, None] + (pactor1.CS_BITS + np.arange(CS_TAIL_BITS)) * sps
    score = (np.abs(T.mark[eye] - T.space[eye]).sum(axis=1)
             - T.env[tail].sum(axis=1))
    origin = int(np.argmax(score)) + slip * sps
    bits = []
    for j in range(pactor1.CS_BITS + extra):
        i = origin + j * sps
        if i < 0 or i >= T.n:
            break
        bits.append(int(T.mark[i] > T.space[i]))     # MARK is a one, as in a frame
    return bits if len(bits) >= pactor1.CS_BITS else None


def decode_control_signal(audio: np.ndarray, t0: float, dur: float,
                          fs: int = FS, *, msb_first: bool = False
                          ) -> CS | None:
    """A PACTOR-1 control signal from a burst at `t0`.

    The four control signals are at MUTUAL DISTANCE 8 -- a perfect equidistant
    code, and by the Plotkin bound no fifth word can exist at that distance in 12
    bits. So a genuine decode lands at 0 errors with the runner-up at 8, and
    anything else is not one of these signals. Callers get the error count and
    should insist on 0; the wide margin is the whole point of the code.

    The two unassigned words of `pactor1.CS_WORDS` are matched too, and bring the
    runner-up in to 6 for the words that mean something. That is still a decode at
    zero and nothing else, and `CS.unassigned` says which kind of word it was.
    """
    bits = cs_bits(audio, t0, dur, fs)
    if bits is None:
        return None
    # LSB FIRST: the first bit on the air is bit 0 of the value as tabulated.
    #
    # Bit order is NOT determined by the decode, and cannot be. The four words
    # are closed under reversal -- CS1 <-> CS2 and CS3 <-> CS4, the description's
    # own "symmetrical pairs (bit reverse patterns)" -- so reading them backwards
    # returns ZERO errors on the wrong codeword. The distance-8 argument that
    # rules out a wrong TONE SENSE says nothing here, and no clean decode ever
    # will. What settles it is protocol semantics read across whole sessions --
    # which opening codeword is a legal answer to a connect, and which of the four
    # is a bare 12-bit burst rather than the head of a packet -- together with
    # every other PACTOR-1 field going LSB-first. docs/protocols/pactor/pactor1-control-signals.md carries
    # the adjudication and is explicit that it is an argument, not a measurement.
    # `msb_first` keeps the other reading one argument away for that analysis.
    #
    # The window is taken BEFORE the reversal, and that ordering is load-bearing.
    # Reversing a longer read and then slicing twelve off the front yields the
    # reverse of a DIFFERENT twelve bits -- the same window shifted by one, an
    # origin slip smuggled in under the bit order's name. The two orders must read
    # the SAME twelve bits or the comparison measures the slip instead.
    window = bits[:pactor1.CS_BITS]
    if not msb_first:
        window = window[::-1]
    # BOTH shift positions, and nothing else.
    #
    # "Mit jedem neuen Paket oder Kontrollsignal wird die Shiftlage invertiert" --
    # the tone sense flips on every transmission, so half of a peer's control
    # signals arrive as the complement of the codeword. The packet decoder has
    # always tried both senses; this one read a single fixed polarity and so
    # could only ever decode every second answer.
    #
    # A one-bit origin slip was tried here too, back when the window started at
    # the detector's burst start, and it was the right diagnosis of the wrong
    # layer: the origin was indeed wrong, by a fraction of a bit as often as by a
    # whole one, and searching two guesses at it inside the codeword comparison
    # bought four decodes and one fictional acknowledgement across the negative
    # corpus. `cs_bits` now positions the window on the SIGNAL, blind to the
    # codewords, so the decoder is back to one trial against a distance-8 code and
    # has no business searching anything.
    return nearest_cs(window)


def nearest_cs(window: list[int]) -> CS:
    """Nearest of `pactor1.CS_WORDS` to twelve bits, over both shift senses.

    MSB-first against the table as tabulated; the caller reverses `window` first
    for the LSB-first reading the air uses.

    THE FOUR CONTROL SIGNALS COME FIRST AND THE TWO UNASSIGNED WORDS AFTER, and
    the order is the whole of what keeps the wider table from costing anything.
    An unassigned word is 6 from every control signal, so a read within 2 of a
    control signal is at least 4 from both of them and cannot be taken; a read at
    3 is at best equidistant, and `e < best` keeps the first, which is the
    published word. Every caller here insists on zero errors, so what the two
    extra words add to the FSM's phantom rate is nothing: the words they
    manufacture out of noise are unassigned words, which drive nothing. What they
    cost is a report -- 12 of the 4096 twelve-bit windows now name a word rather
    than 8 of them.

    THE SENSE IS THE FRAME'S. `sense` is `Packet.inverted` for the same cycle and
    the `invert` every renderer in `pactor1` takes: a one on SPACE rather than on
    MARK. It reads that way because the readers above decide a one on MARK, as
    `_Tones.read` does for a data field, and because a codeword's one and a data
    byte's one ride the SAME tone -- measured, `pactor1.control_signal`.

    That identity is what makes the value usable. `onair._MasterGrid.align` takes
    the peer's sense and hands it straight to the packet renderer, so a `sense` in
    a convention of its own puts every packet we transmit one cycle out of phase
    with the station that just told us its phase.

    THE SENSE IS RETURNED AND NOT DISCARDED. Trying both and keeping only the
    codeword is the failure docs/protocols/pactor/pactor1-data-packets.md §7 names by its symptom:
    "a control-signal decoder must report which sense matched ... or the station
    cannot observe the peer's phase even in principle -- the failure mode is
    twenty packets, CS4 every cycle, no advance." That is WS8EOC on 2026-07-30,
    to the cycle. The peer's phase is the only thing on the air that says which
    shift our own next transmission owes it, so it is half the decode.
    """
    best, which, matched = 99, -1, 0
    for sense in (0, 1):
        read = [b ^ sense for b in window]
        for idx, word in enumerate(pactor1.CS_WORDS):
            e = sum(x != ((word >> (pactor1.CS_BITS - 1 - j)) & 1)
                    for j, x in enumerate(read))
            if e < best:
                best, which, matched = e, idx, sense
    return CS(which, best, matched)


CS_ALIAS_BITS = 2
"""The slip at which one control signal casts another, and the only one that does.

CS4 read two bit periods late IS CS3, bit for bit, in the same shift sense. Ten
slots hold the tail of the codeword and the last two hold the silence after the
burst, and the fill is not free to go either way: the ten real bits already carry
weight 6 or 4, `_cs_decide` splits twelve slots six and six, so the two silent
ones are FORCED to the value that completes the other word. Searching every slip
in +/-3 bits with every fill over all four signals in both senses, this pair is
the whole of it -- CS1 and CS2 cast nothing at any slip.

WHAT IT COSTS IS THE CHANNEL. CS3 is the break-in, so a CS4 read two bit periods
late hands the link to a station that asked for nothing:
`captures/onair-0828-1838` at 22.35 s, where the same 120 ms reads CS4 for any
anchor at or below 70 ms into the window and CS3 for any at or above 72, the
session's own anchor sat at 84, and sixteen cycles then passed with both ends
receiving. Across the corpus 436 of the 568 windows that carry a CS3 at all carry
a CS4 two bit periods in front of it in the same sense, and the eye score does not
separate the pair.

THE MIRROR IS EXACT AND IS WHAT MAKES THE SLIP ALONE USELESS: a GENUINE CS3
head read two bits early, with the turnaround silence in front of it, is a
zero-error CS4 in the same sense. The codewords therefore cannot say which of
the two is on the air, and `cs_anchored` asks the silence instead."""


def cs_anchored(audio: np.ndarray, t0: float, fs: int = FS) -> CS | None:
    """The control signal the GRID says is at `t0`, or None.

    `cs_bits` reads a burst a detector found and spends `CS_SEARCH_S` looking for
    it, because a run start is not a bit grid. An anchor is not that kind of
    input: it is already a bit grid, carried a cycle at a time and trimmed
    sub-bit by `cs_time_dev`. What it is not is exact. Over 755 tracked cycles the
    burst the grid was steering on sat further than half a bit from the predicted
    instant in 23% of them, so this searches `CS_SEARCH_HALF_S` either side --
    three half-bits -- and the constant carries the curve that width came off.

    A SEARCH AND NOT A WIDER BRACKET. Every alignment is decoded and the
    best-scoring one that IS a codeword wins, which is what the 1990 description
    has its master do across the receive window (`pactor.txt:272-274`). Taking
    the best-scoring alignment and decoding it once -- what this did until
    2026-08-28 -- cannot be widened at all: over the 20 measured answers of
    2026-07-30 it holds 14 at half a bit, 13 at 7.5 ms and 3 at 10, because past
    one bit period the eye score cannot say which period the word is in. Searched
    at the same instants it reads all 20, the six it used to miss all landing on
    the cycle's own shift parity.

    Three things separate it from `cs_bits`, and they are one claim about the
    input:

      * the window is `CS_SEARCH_HALF_S` rather than the detector's guard, and
        the alignment is chosen by what decodes rather than by the eye alone;
      * the bits are DECIDED by `_cs_decide`'s split rather than by comparing the
        two tones, which is what a peer whose tones arrive at different levels
        needs and what `CS_SPLIT_MIN` pays for;
      * every accepted alignment clears `CS_SEARCH_EYE_MIN`, which is what the
        extra trials are paid for: the split's weight-6 prior costs four times
        the phantom rate per trial, and a search spends it once per bit period of
        window.

    The tail term stays in the score, so a changeover packet -- 840 ms of data
    where a bare codeword has silence -- scores badly here. IT IS NOT REJECTED BY
    IT: the term orders the alignments and the loop takes the best-scoring one
    that decodes, so a head reaches this reader whenever it is the only zero-error
    alignment in the bracket, which the rendered packet and K4MSU's real one both
    are. A break-in is still `cs_head`'s to read, and `onair._SessionRx._p1_cs`
    is where that is enforced rather than assumed.
    """
    sps = FS // 100
    # The tail slots are inside the slice, not past its end. Cutting the audio at
    # the word and letting the padding stand in for the unkey charges the LATE
    # alignments nothing and the early ones the real thing, which is a half-bit
    # bias dressed as a guard -- it moved the whole capture range one bit early.
    span = CS_BIT_S * (pactor1.CS_BITS + CS_TAIL_BITS)
    want = int((t0 - CS_SEARCH_HALF_S) * fs)
    # CS_ALIAS_BITS of HISTORY, and not one alignment more of SEARCH. The shadow
    # test needs the two bit periods in front of the earliest candidate, which the
    # bracket alone does not hold; `base` puts the first candidate back exactly
    # where it was, so the set of alignments that can be ACCEPTED is unchanged.
    lo = max(0, want - int(CS_ALIAS_BITS * CS_BIT_S * fs))
    hi = min(audio.size, int((t0 + span + CS_SEARCH_HALF_S) * fs))
    x = _to_fs(np.asarray(audio[lo:hi], dtype=float), fs)
    base = max(0, int(round((want - lo) * FS / fs)))
    if x.size - base < pactor1.CS_BITS * sps:
        return None
    # Audio past the capture counts as silence, for `cs_bits`' reason: a window
    # that stops inside the burst scores badly and fails rather than winning by
    # having nothing to be charged for.
    T = _Tones(np.concatenate([x, np.zeros(CS_TAIL_BITS * sps)]), sps)
    # The searched window and not one sample more. The slice is longer than that
    # -- it has to hold the tail slots -- so the width is stated here rather than
    # inferred from what happens to fit.
    top = min(2 * int(CS_SEARCH_HALF_S * FS),
              T.n - base - (pactor1.CS_BITS + CS_TAIL_BITS - 1) * sps - 1)
    if top < 0:
        return None

    def read(start: int) -> CS | None:
        cols = start + np.arange(pactor1.CS_BITS) * sps
        bits, split = _cs_decide(T.mark[cols], T.space[cols])
        if split < CS_SPLIT_MIN:
            return None
        got = nearest_cs(bits.tolist()[::-1])       # LSB first on the air
        if got.errors:
            return None
        # AN UNASSIGNED WORD PAYS FOR ITS OWN TRIAL, and pays more than a control
        # signal does, because the consequences are not symmetric. Two more words
        # in the table are 1.5x the chance of landing on one, and a word that
        # drives nothing may not spend any of a budget the acknowledgements need
        # -- every count this station has ever taken of a peer's answers is read
        # through it. Measured over the 560 quiet reads of the WS8EOC windows: the
        # one read that manufactures an unassigned word scores 0.19 where the real
        # ones score 0.56 to 0.93, the DL6MAA grant at 0.93.
        floor = EYE_MIN if got.unassigned else CS_SEARCH_EYE_MIN
        if _eye(T, np.array([start]), sps, pactor1.CS_BITS)[0] < floor:
            return None
        return got

    # `ACQUIRE_HOP_S`, which is the resolution the description's own acquisition
    # search works at and the one `acquire_control_signal` already uses. An eighth
    # of a bit: finer buys nothing a matched filter can tell apart and costs
    # trials the eye gate has to pay for.
    starts = base + np.arange(0, top + 1, max(1, int(ACQUIRE_HOP_S * FS)))
    eye = starts[:, None] + np.arange(pactor1.CS_BITS) * sps
    tail = starts[:, None] + (pactor1.CS_BITS + np.arange(CS_TAIL_BITS)) * sps
    score = (np.abs(T.mark[eye] - T.space[eye]).sum(axis=1)
             - T.env[tail].sum(axis=1))
    ahead = starts[:, None] + (np.arange(CS_ALIAS_BITS) - CS_ALIAS_BITS) * sps
    for k in np.argsort(-score):
        got = read(int(starts[k]))
        if got is None:
            continue
        # WHICH SIDE THE SILENCE IS ON, and nothing else, tells the two apart.
        # The mirror is exact: a GENUINE changeover head read two bit periods
        # early -- its turnaround gap in front of it -- is a zero-error CS4 too,
        # so the codewords alone would refuse every real break-in as readily as
        # the alias. What differs is the arithmetic either side of the fourteen
        # slots. A bare burst casting a false CS3 has its own first two bits
        # ahead of the word and the unkey behind it; a real head has the
        # turnaround ahead and 840 ms of the new sender's packet behind. So the
        # two energies are compared to each other -- no level, no threshold, and
        # scale-free -- and only the loud-ahead-quiet-behind case is refused.
        if (got.index == pactor1.CS_CHANGEOVER and starts[k] >= CS_ALIAS_BITS * sps
                and T.env[ahead[k]].sum() > T.env[tail[k]].sum()):
            cast = read(int(starts[k]) - CS_ALIAS_BITS * sps)
            if cast is not None and cast.index == pactor1.CS_SPEED \
                    and cast.sense == got.sense:
                continue
        return got
    return None


HEAD_LEAD_BITS = 2
"""Bit slots BEFORE the word whose energy is subtracted from the head's score.

The exact mirror of CS_TAIL_BITS, and it has to be the mirror: a changeover
packet's codeword is the first 120 ms of a 960 ms transmission, so what is quiet
around it is the turnaround gap in FRONT of it and never the slots behind."""

HEAD_BRACKET_S = 0.015
"""How far either side of the predicted instant the word is positioned.

A point read would do if the anchor it is aimed at were exact, and it is not.
Measured against our own encoder over turnarounds from 80 to 120 ms, `rx_due`
predicts the burst a flat +10.0 ms late at every one of them: `p1_burst_onsets`
stamps a clean rendered burst 5.0-5.5 ms late, and `cs_time_dev` -- asked to
correct that from an anchor sitting exactly on its own half-bit wrap -- adds
another five instead of subtracting them. One whole bit, systematically, which no
read at a point survives. Fifteen milliseconds covers it with half a bit to
spare and stays inside the 20 ms the grid will pull for at all.

OFF THE AIR THE SPREAD IS WIDER THAN THE ENCODER SHOWED AND SITS THE SAME WAY.
Over the 23 real changeover packets measured on 2026-09-01 (WS8EOC at both
rates, KB5LZK, KC0TPS)
the instant each session's own grid called for -- its `rx_due`, off its own
measured `d` -- ran from 2.6 ms EARLY to 35.4 ms late against the first bit the
packet reader gives, with a median of +3.1. One reading is outside this bracket
and no widening fixes it: it belongs to a cycle whose grid had just been
re-placed."""

HEAD_SLIP_BITS = 2
"""Whole bit periods the score may slip by, and so the alignments also tried.

THE SCORE IS A LEVEL: twelve slots of |MARK - SPACE| less two of envelope. It
positions the word only while the packet's own level is flat across the bracket,
and off the air it is not. Selective fading puts one FSK tone several times below
the other, which can make a head the weakest part of its own packet -- on
`captures/onair-0821-2114` at 76.9244 s the twelve head slots run 20 to 126 in
envelope against 28 to 147 for the twelve behind them -- and the argmax then
walks INTO the packet, where the eye term gains more than the two lead slots
cost. That is 7 of the 23 real changeover packets, read at the instant their own
session called for, and every one of the seven misses by a WHOLE bit period:
within the packet only symbol-aligned offsets score at all.

So the extra trials are exactly those slips, and two bits is what the corpus
asks for -- +-1 recovers 2 of the 7 and +-2 recovers 6. The seventh is not a
positioning failure: its head reads CS3 at zero errors and scores 0.49 against
EYE_MIN's 0.50. It is the same distance `CS_ALIAS_BITS` names from the other
end, which is not a coincidence -- a codeword slipped two bits is where the
twelve-bit code's other member lives."""


def cs_head(audio: np.ndarray, t0: float, fs: int = FS) -> CS | None:
    """The control signal at the HEAD of a transmission at `t0`.

    `cs_bits` cannot read one and never will: it positions its window by
    requiring the two slots after the word to be QUIET, which is true of a bare
    burst and false of a packet head by construction. So a changeover packet --
    CS3 as its first 120 ms and 840 ms of the new sender's data behind it -- is
    invisible to every path that goes through it, and the packet scan that CAN
    see one needs all 960 ms, which no sending station's listening window holds.
    This is the twelve bits, read where the grid says they are, so the reversal
    can be recognised inside the 170 - d - settle the cycle leaves for it.

    The asymmetry that places the window is the other end of the one `cs_bits`
    uses. A changeover packet BEGINS with the word, so the slots ahead of it are
    the turnaround's silence: an alignment a bit late is charged with a bit of
    real signal, and one a bit early loses a bit of eye to that same silence.
    Within the packet itself every symbol-aligned offset opens the eye equally
    wide -- so without that term nothing here could say which twelve bits are the
    word, and the bracket would be free to slip a whole one.

    FIVE TRIALS against the distance-8 code -- the argmax and the whole bit slips
    of `HEAD_SLIP_BITS` either side, in score order -- and the score still never
    looks at the codeword table, so it cannot manufacture a match. What the four
    extra alignments cost is stated where they are asked for.

    A trial is not free, and reading at an instant means reading whatever is
    there: twelve bits of noise land on one of the four words in one of two senses
    once in 512 -- 8 of the 4096 twelve-bit words, as `_cs_split` states from the
    other side -- measured at 11 accepts in 20000 reads of white noise and at 3
    CS3s in 800 quiet instants of two real off-air recordings. EYE_MIN is what
    answers that -- the frame decoder's own question, asked of these twelve
    symbols. Over those same 800 quiet instants the three that read a codeword
    score 0.35 to 0.42 and every real burst that reads one scores 0.78 or better,
    with a rendered head at 0.90 at 3 dB SNR; the gate takes the false accepts to
    none while costing no burst in either recording.

    AND THE ACCEPTED ALIGNMENT HAS TO LOOK LIKE A HEAD, which is `cs_anchored`'s
    alias test read in the mirror: the two slots ahead of the word must not be the
    loudest thing in the read, no threshold anywhere, where what they are weighed
    against is the two slots behind the word OR the word's own median slot,
    whichever is louder. What it refuses is the CS4 read two bit periods late,
    which is the 22.35 s cycle of `captures/onair-0828-1838` and the one thing
    this reader could never tell from a break-in: those two slots are then the
    burst's own first bits, above both terms at every anchor of that cycle.

    THE WORD'S OWN LEVEL IS THE TERM THAT CANNOT GO MISSING, and comparing to the
    material behind alone was wrong for the case the corpus does not hold and the
    link does: A BARE CS3. A changeover packet has 840 ms of the new sender's data
    behind its codeword -- over the 75 real heads on disk that ratio has a median
    of 0.31 -- but a peer whose packet lands outside the window we opened, and
    every rendered codeword a test hands this reader, has SILENCE there. Both
    terms are then zero, the comparison decides on no evidence, and the reader
    walked two bits early instead and named the CS4 the code puts there
    (`CS_ALIAS_BITS`), which `onair._SessionRx._p1_cs` will not act on. That is a
    changeover ignored and the channel kept, and it is what
    `tests.shrike.test_holdbudget`'s replayed K4MSU hold caught.

    THE MARGIN IS NOT LARGE EVERYWHERE: on the two WS8EOC 80 m arms, where our own
    emission leaks into the capture, the lead runs 0.74 and 1.08 of the word's own
    median and it is the material behind that carries the second one.

    VALIDATED OFF THE AIR, and it no longer stands on the encoder. The corpus
    holds 23 real changeover packets from four gateways at both speeds
    (the 23 packets of 2026-09-01, plus VE1YZ's 51 repeats on
    `captures/onair-0902-2342` and KB5LZK's on `captures/onair-0903-1051`), every
    one CRC-valid with a zero-error CS3 head. Read at the instant each session's
    own grid called for, this takes 22 of the 23 and positions the word within
    -2.4 to +2.7 ms of the first bit the whole-packet reader gives; the misses are
    the EYE_MIN cases named in `HEAD_SLIP_BITS`. Over the 3965 such instants 91
    sessions actually called at, it corroborates 68 break-ins against 35 accepts
    no packet reader can find -- where the argmax alone corroborated 59 against 30.

    THE LEAD-IN TONE THE OLD DISCLAIMER FEARED still makes it fail rather than
    lie, and now it is the mirror test that says so rather than the codeword: a
    tone at the packet's own level prepended to four real heads, 20 to 100 ms of
    it, is read as the break-in in 5 of the 15 cases, refused outright in 9, and
    in the last named as the CS4 the head itself casts two bits early, which
    `onair._SessionRx._p1_cs` does not act on. No station in the corpus keys one
    -- the whole-packet reader, which the session falls back to, loses 4 of the
    same 15.
    """
    sps = FS // 100
    # An EARLY slip reads two bit periods further back than the score does, and a
    # caller whose buffer starts inside that is served with the slips it can hold
    # rather than refused: the anchor is `rx_due`, which sits a turnaround inside
    # the window, and a station that opened its window late still has its head.
    pre = (HEAD_LEAD_BITS + HEAD_SLIP_BITS) * fs // 100
    lo = int((t0 - HEAD_BRACKET_S) * fs) - pre
    if lo < 0:
        pre, lo = pre + lo, 0
    if pre < HEAD_LEAD_BITS * fs // 100:
        return None
    span = (pactor1.CS_BITS + HEAD_SLIP_BITS + HEAD_LEAD_BITS) * fs // 100
    x = _to_fs(np.asarray(audio[lo:int((t0 + HEAD_BRACKET_S) * fs) + span],
                          dtype=float), fs)
    br = int(HEAD_BRACKET_S * FS)
    base = int(round(pre * FS / fs))
    want = base + 2 * br + (pactor1.CS_BITS + HEAD_SLIP_BITS) * sps
    # ...but the buffer is asked for the WORD's reach and never the slips': a
    # caller holding one listening window is the whole point of this reader, and a
    # late slip that runs off the end scores badly and is refused, which is the
    # right answer. Demanding its samples refuses the read instead -- which it
    # did, on every cycle of `test_grid`'s changeover scene.
    reach = base + 2 * br + pactor1.CS_BITS * sps
    if x.size < reach - br:
        return None
    # Audio past the window counts as silence, for `cs_bits`' reason: a station
    # whose capture stops inside the head scores badly and fails, rather than
    # having its last alignments win by having nothing to be charged for.
    x = np.concatenate([x, np.zeros(max(0, want - x.size))])
    T = _Tones(x, sps)
    starts = np.arange(2 * br + 1) + base
    origin = int(starts[int(np.argmax(_head_score(T, starts, sps)))])
    trials = np.array([origin + j * sps
                       for j in range(-HEAD_SLIP_BITS, HEAD_SLIP_BITS + 1)])
    trials = trials[(trials >= HEAD_LEAD_BITS * sps)
                    & (trials + (pactor1.CS_BITS + HEAD_LEAD_BITS - 1) * sps < T.n)]
    for k in trials[np.argsort(-_head_score(T, trials, sps))]:
        got = _head_read(T, int(k), sps)
        if got is not None:
            return got
    return None


def _head_score(T: "_Tones", starts: np.ndarray, sps: int) -> np.ndarray:
    """How much like the start of a transmission each alignment looks."""
    eye = starts[:, None] + np.arange(pactor1.CS_BITS) * sps
    lead = starts[:, None] + (np.arange(HEAD_LEAD_BITS) - HEAD_LEAD_BITS) * sps
    return (np.abs(T.mark[eye] - T.space[eye]).sum(axis=1)
            - T.env[lead].sum(axis=1))


def _head_read(T: "_Tones", start: int, sps: int) -> CS | None:
    """One alignment: the eye, the mirror of `CS_ALIAS_BITS`, then the codeword.

    The lead is weighed against the loudest of what surrounds the word, so a
    burst with nothing behind it is still a head.
    """
    if _eye(T, np.array([start]), sps, pactor1.CS_BITS)[0] < EYE_MIN:
        return None
    ahead = T.env[start - HEAD_LEAD_BITS * sps:start:sps]
    behind = T.env[start + pactor1.CS_BITS * sps::sps][:HEAD_LEAD_BITS]
    word = T.env[start:start + pactor1.CS_BITS * sps:sps]
    loudest = max(behind.mean() if behind.size else 0.0, float(np.median(word)))
    if ahead.mean() >= loudest:
        return None
    idx = start + np.arange(pactor1.CS_BITS) * sps
    bits = (T.mark[idx] > T.space[idx]).astype(int).tolist()
    got = nearest_cs(bits[::-1])             # LSB first on the air
    return None if got.errors else got


ACQUIRE_HOP_S = 0.00125
"""The step between the alignments `acquire_control_signal` tries."""

ACQUIRE_LEAD_S = 0.004
"""Audio `acquire_control_signal` takes in front of the `t0` it is handed, and
the origin the offsets it tries are counted from.

IT LOOKS LIKE A DOUBLE COUNT AND IT IS NOT ONE, which earns the paragraph
because the arithmetic invites a correction that is wrong. Alignment zero sits
at `t0 - ACQUIRE_LEAD_S`, so `t0 + offset` should name a codeword this much late
-- and measured, it does not: the scan returns the FIRST alignment that reads
twelve bits correctly, and bit centres sampled a few milliseconds early still
land inside their own 10 ms bits, so the first match runs early by about the same
4 ms. Over a codeword planted at a known offset in quiet audio, `t0 + offset`
recovers the planting to 0-1 ms while the lead-corrected reading is 4-5 ms early;
near threshold the first match slips late instead and the two swap places.

The air agrees with the bench. On `pactor-current-ws8eoc-80-force-20260916T135856Z`
the energy detector -- a different instrument, no alignments in it -- put the
answer of cycle 17 at 71.4 ms after our data ended, against 71.3 from `t0 +
offset` and 67.2 lead-corrected.

So the constant is named and used where the slice is cut, and NOT subtracted
from what the search returns. What a caller is owed by that is 4 ms of absolute
uncertainty either way, which is why the corroboration rule is a spread and not
a position."""

ACQUIRE_ALPHABET = (pactor1.CS_ACK_A, pactor1.CS_SPEED)
"""The only two codewords that are a legal answer to a call."""

ACQUIRE_P_ALIGN = 2 * len(ACQUIRE_ALPHABET) / 2 ** pactor1.CS_BITS
"""The chance ONE alignment of the search reads an accept out of nothing: two
codewords in two shift senses, against the 4096 twelve-bit words it could read."""


def _alignments(span: float, fs: int, hop: float) -> range:
    """Every offset the search tries across `span`."""
    return range(0, int(span * fs), max(1, int(hop * fs)))


def acquire_null(span: float, fs: int = FS,
                 hop: float = ACQUIRE_HOP_S) -> float:
    """How often `acquire_control_signal` accepts across `span` with NOTHING there.

    The search reads twelve bits at each alignment and takes any of two codewords
    in either shift sense, so four of the 4096 words it could read are an accept
    -- and there is no energy test anywhere in it. Feed it noise and it accepts
    at 4/4096 per alignment, which over the alignments a span holds is the figure
    below. The detector's own arithmetic, not a fitted rate: the only inputs are
    the alphabet, the word length and the hop.

    It is an APPROXIMATION, because adjacent alignments overlap by nine tenths of
    a bit and are not independent. Measured against what the search actually does:
    on white noise at 100 alignments it accepts in 8.8-10.8% of trials against
    9.3% predicted, and over 3867 real off-air receive segments the ratio of
    observed accepts to this figure runs 1.00, 1.36, 1.09 and 1.05 as the span
    they are searched across grows from 15 ms to 1.77 s. So it is good to a small
    factor and no better, and everything that prints it says "about".
    """
    return 1 - (1 - ACQUIRE_P_ALIGN) ** len(_alignments(span, fs, hop))


def acquire_control_signal(audio: np.ndarray, t0: float, dur: float,
                           fs: int = FS, span: float = 0.139,
                           hop: float = ACQUIRE_HOP_S) -> tuple[int, float, int] | None:
    """A connect ANSWER, found by sliding a correlator. Acquisition only.

    Returns (codeword index, offset seconds, shift sense) on an exact match, else
    None. `t0 + offset` is where the codeword is, to a millisecond on a clean
    one, though the alignments are counted from `ACQUIRE_LEAD_S` in front of `t0`
    -- that constant carries the measurement. The sense is the answering
    station's Shiftlage for that cycle and so the phase our first data packet
    owes it -- see `nearest_cs`. This is the earliest point in a link at which it
    can be known, and the only one before the first data packet goes out.

    The established-link decoder reads ONE alignment and demands the grid already
    be right. That is correct once a link is running -- a working implementation
    does exactly that, with no timing search at all -- but it is the wrong tool for
    acquisition, where the grid is what we are trying to find.

    The gap this used to open has closed. It was
    measured at 10% of W6IDS's bursts for the single read against 36% for the
    slide, and read as the cost of not searching; it was really the cost of
    reading twelve bits off a burst START. With `cs_bits` positioning its own
    window the two are level -- 16 of 37 against 15 of 37 on W6IDS, 14 and 14 on
    WS8EOC. So this earns its place on the architecture and not on a score: at
    acquisition the receive anchor does not exist yet, and the two-codeword
    alphabet is a constraint the established link cannot use.

    Three things keep the search from manufacturing an answer, and all three are
    the reference implementation's, not inventions:

      * only CS1 and CS4 are legal answers to a call, so the alphabet is two
        codewords rather than four;
      * the match must be EXACT -- zero bit errors in twelve;
      * it runs once per cycle over the window where an answer is due, not over
        whatever the detector offers.

    It is not free even so: 1 foreign candidate in 32 matched in a sweep that
    ignored the third constraint. That is the price of acquiring at all, and it is
    bounded by a wrong acquisition producing a link that immediately fails and
    retries -- not a reported contact. Do NOT reach for this once connected.

    THE SMOOTHING TAIL IS A CONVENIENCE AND THE CODEWORD IS NOT. The slice runs
    20 ms past the last alignment so the 1.5 ms smoother has audio to run out
    into, and refusing the whole search when a recording ended inside that tail
    charged those 20 ms to the top of every band the loop could hand it. A keyed
    1.25 s cycle leaves 235-241 ms; VE3KPG answers at 96-107. Over the fifteen
    calls to it of 2026-09-13 -- 357 listen windows -- that refusal cost 53 of
    the 95 zero-error codewords those windows hold.
    So the slice is clipped to the audio, and what is actually required is
    stated instead: one codeword has to fit. Alignments past the end read short
    and stop, so nothing partial can match.
    """
    from scipy.signal import butter, filtfilt, hilbert

    lo = int((t0 - ACQUIRE_LEAD_S) * fs)
    hi = min(int((t0 + dur + span + 0.02) * fs), audio.size)
    if lo < 0 or hi - lo < int((dur + ACQUIRE_LEAD_S) * fs):
        return None
    b, a = butter(4, [1200 / (fs / 2), 1800 / (fs / 2)], btype="band")
    z = hilbert(filtfilt(b, a, audio[lo:hi].astype(float)))
    fi = np.diff(np.unwrap(np.angle(z))) / (2 * np.pi) * fs
    k = int(0.0015 * fs)
    fi = np.convolve(fi, np.ones(k) / k, "same")
    sym = (fi < (MARK + SPACE) / 2).astype(int)     # MARK is a one, as in a frame
    step = int(CS_BIT_S * fs)
    for off in _alignments(span, fs, hop):
        bits = []
        for j in range(pactor1.CS_BITS):
            i = off + step // 2 + j * step
            if i >= len(sym):
                break
            bits.append(int(sym[i]))
        if len(bits) < pactor1.CS_BITS:
            break
        window = bits[::-1]                     # the air is LSB-first
        for sense in (0, 1):
            read = [x ^ sense for x in window]
            for idx in ACQUIRE_ALPHABET:
                word = pactor1.CONTROL_SIGNALS[idx]
                if all(v == ((word >> (pactor1.CS_BITS - 1 - j)) & 1)
                       for j, v in enumerate(read)):
                    return idx, off / fs, sense
    return None


def decode_connect(audio: np.ndarray, fs: int = FS,
                   tones: "_Tones | None" = None) -> Connect | None:
    """Recover (variant, callsign) from a PACTOR-1 connect burst, or None.

    Locks the 9-byte 100-Bd address section by sync + terminator, refines the
    baud phase to the centre of the lock plateau (the coarse onset runs early,
    which 100-Bd integration tolerates but the 200-Bd redundancy section does
    not), then reads the redundancy section to fix the first character and split
    Normal from Longpath.

    `tones` is this audio's 100 Bd `tone_series` when the caller already has one.
    """
    audio = _to_fs(audio, fs)
    sps = FS // 100
    if len(audio) < sps:                          # nothing to lock onto
        return None

    sps_r = FS // 200                             # the 200 Bd redundancy section
    if tones is None:
        tones = _Tones(audio, sps)

    for on in _onsets(tones, sps):
        lo, hi = max(0, on - sps), on + 2 * sps
        for invert in (False, True):
            cand = _sync_starts(tones, lo, hi, sps, invert)
            cand = cand[tones.fits(cand, sps, ADDR_LEN * 8)]
            if cand.size == 0:
                continue
            good = cand[_locked(_rotate(tones.read(cand, sps, ADDR_LEN, invert)))]
            if good.size == 0:
                continue
            start = int(good[0] + good[-1]) // 2
            T = _rotate(tones.read(np.array([start], np.int64), sps,
                                   ADDR_LEN, invert))[0].tobytes()

            red_first = _redundancy_first(audio, start + ADDR_LEN * 8 * sps,
                                          sps_r, invert)

            variant, first = _classify(T[1], red_first)
            chars = [first]
            for i in range(2, len(T)):
                if T[i] == TERM:
                    break
                chars.append(T[i])
            if not all(_valid_char(c) for c in chars):
                continue
            return Connect(variant, "".join(chr(c) for c in chars), invert)
    return None


def _classify(primary_first: int, red_first: int | None) -> tuple[str, int]:
    """Normal vs Longpath from the address byte and the redundancy's first char.

    Longpath transmits the first callsign character bit-inverted in the 100-Bd
    address section while the 200-Bd redundancy copy carries the true character
    (docs/protocols/pactor/pactor-connect-frames.md sec 3, PROVED on air). The redundancy image byte 0 is
    (char << 1), so the true first character is red_first = T6[0] >> 1.
    """
    if red_first is not None and _valid_char(red_first):
        if primary_first == red_first:
            return "Normal", primary_first
        if primary_first == (~red_first) & 0xFF:
            return "Longpath", red_first
    return "Normal", primary_first


# --- branch B: Robust Call and Free Signal ---------------------------------

CALL_B_LEN = pactor1.CALL_B_LEN
CALL_B_SHIFT = 200.0
CALL_B_SWEEP = np.arange(450.0, 2551.0, 50.0)
"""Tone-pair centres the branch-B scan tries when the caller names none.

`decode_connect` reads at the fixed 1400/1600 this station keys, and branch B
cannot: the four Robust Calls in our own corpus sit at 707, 2006, 1013 and
1500 Hz, wherever each caller's dial and audio chain put them. A 50 Hz grid is
fine either side -- a 100 Bd symbol window is 100 Hz wide, so a 25 Hz centre
error costs about a tenth of the tone magnitude -- and all seven known trains
decode off this grid, none of them on it.
"""

CALL_B_VARIANTS = {pactor1.CALL_B_MASKS["robust"]: "Robust",
                   pactor1.CALL_B_MASKS["fs_normal"]: "FreeSignalNormal",
                   pactor1.CALL_B_MASKS["fs_encrypted"]: "FreeSignalEncrypted"}


def _call_b_gate(bits: np.ndarray, sps: int, n: int) -> np.ndarray:
    """Which of the first `n` bit offsets satisfy a[6]==~a[8], a[7]==~a[9], a[6]==a[10].

    Twenty-four bit relations inside the frame, and they hold whatever whitening
    mask the sender used and whichever shift it keyed -- a mask XORs both sides of
    a comparison and cancels. So this runs once per offset, before the frame is
    read as bytes and before the kind is known, and it is what makes an
    every-sample search affordable: it costs 24 shifted boolean ops over the
    buffer and leaves a handful of candidates for the CRC.
    """
    ok = np.ones(n, bool)
    for i, j, complement in ((6, 8, True), (7, 9, True), (6, 10, False)):
        for k in range(8):
            x = bits[(i * 8 + k) * sps:][:n]
            y = bits[(j * 8 + k) * sps:][:n]
            ok &= (x ^ y) == complement
    return ok


def _call_b_slot(raw: bytes) -> Connect | None:
    """One 11-byte slot -> the connect it carries, by whichever mask closes the CRC."""
    for mask, variant in CALL_B_VARIANTS.items():
        for inverted in (False, True):
            a = bytes(x ^ mask ^ (0xFF if inverted else 0) for x in raw)
            b = bytes([a[0] ^ a[6], a[1] ^ a[7], a[2] ^ a[8], a[3] ^ a[9],
                       a[4] ^ a[6], a[5] ^ a[7], ~a[6] & 0xFF, ~a[7] & 0xFF])
            if coding.crc16(b, pactor1.CALL_B_CRC) ^ 0xFFFF != pactor1.CALL_B_RESIDUE:
                continue
            call = _call_b_address(b)
            if call:
                return Connect(variant, call, inverted)
    return None


def _call_b_address(b: bytes) -> str | None:
    """The callsign out of the six payload bytes: eight 6-bit chars, cut at the space.

    The space is the field's separator and the monitor's terminator
    (`pactor1.call_b_address`), so what a receiver reports is the run before it.
    A frame whose characters are not callsign characters is refused here rather
    than reported: a single CRC-valid burst is not enough to trust an ident, and
    the Free Signal reference tape carries one garbage ident that closes the CRC
    among five good ones.
    """
    v = int.from_bytes(b[:6], "little")
    chars = []
    for k in range(pactor1.CALL_B_CHARS):
        c = ((v >> (6 * k)) & 0x3F) + 0x20
        if c == 0x20:
            break
        if not _valid_char(c):
            return None
        chars.append(chr(c))
    return "".join(chars) or None


def _call_b_frames(tones: "_Tones", sps: int):
    """Every branch-B frame one tone pair's magnitudes carry.

    (start sample, connect, how much of the pair's energy the frame sat in) --
    the third is what picks a winner between sweep centres. A centre a whole
    shift off reads the frame complemented and byte-perfect, because one bin
    lands on a real tone and the other lands in a null, so half the symbols
    decode on nothing at all; it is the CAPTURED ENERGY that separates that from
    the centre the station is actually keying, not the bytes, which are equally
    valid under the complement mask.
    """
    bits = tones.mark > tones.space
    n = bits.size - (CALL_B_LEN * 8 - 1) * sps
    if n <= 0:
        return
    starts = np.flatnonzero(_call_b_gate(bits, sps, n))
    if starts.size == 0:
        return
    energy = tones.env[starts[:, None] + np.arange(CALL_B_LEN * 8) * sps].mean(axis=1)
    for start, raw, e in zip(starts, tones.read(starts, sps, CALL_B_LEN, False),
                             energy):
        cn = _call_b_slot(raw.tobytes())
        if cn is not None:
            yield int(start), cn, float(e)


def _call_b_scan(audio: np.ndarray, fs: int, tones: "_Tones | None",
                 centre: float | None) -> list[tuple[float, Connect]]:
    """Branch-B frames in one buffer, one entry per key-down, earliest first.

    The same frame decodes at a span of adjacent sample offsets and at two or
    three sweep centres; bursts are 1.25 s apart and the frame is 0.88 s, so hits
    within a frame length of each other are one key-down and the strongest of
    them is the reading kept.
    """
    sps = FS // 100
    if tones is not None:
        hits = list(_call_b_frames(tones, sps))
    else:
        audio = _to_fs(audio, fs)
        hits = [h for fc in ([centre] if centre else CALL_B_SWEEP)
                for h in _call_b_frames(
                    _Tones(audio, sps, fc - CALL_B_SHIFT / 2,
                           fc + CALL_B_SHIFT / 2), sps)]
    out: list[tuple[float, Connect, float]] = []
    for start, cn, energy in sorted(hits, key=lambda h: h[0]):
        t = start / FS
        if out and t - out[-1][0] < CALL_B_LEN * 8 / 100:
            if energy > out[-1][2]:
                out[-1] = (t, cn, energy)
            continue
        out.append((t, cn, energy))
    return [(t, cn) for t, cn, _ in out]


def decode_call_b(audio: np.ndarray, fs: int = FS, tones: "_Tones | None" = None,
                  centre: float | None = None) -> Connect | None:
    """Recover a Robust Call or a Free Signal from branch-B audio, or None.

    The tone pair is searched over `CALL_B_SWEEP` unless the caller pins it:
    `centre` names one pair centre, and `tones` hands over magnitudes already
    taken at this station's own 1400/1600 (`tone_series`), the way
    `decode_connect` accepts them.

    Anchored on the air: every branch-B train in the corpus reads back with the
    ident and kind an independent monitor gives it, and neither of the two real
    Normal connects in the corpus produces a frame here.
    """
    found = _call_b_scan(audio, fs, tones, centre)
    return found[0][1] if found else None


def decode_call_b_all(audio: np.ndarray, fs: int = FS,
                      centre: float | None = None) -> list[tuple[float, Connect]]:
    """Every branch-B frame in a recording, as (seconds into the audio, connect)."""
    return _call_b_scan(audio, fs, None, centre)


# --- data decode -----------------------------------------------------------

@dataclass
class Packet:
    payload: bytes        # the data field, less IDLE (`pactor1.field_bytes`)
    status: int
    header: int
    baud: int             # 100 or 200
    inverted: bool
    breakin: bool = False
    """A changeover packet: its first bytes are the CS3 head, not a header byte,
    and the station that decodes one is being told to give up the channel."""
    start: int | None = None
    """Sample index of the frame's first bit within the audio it was read from.

    Where the frame IS, which is not where the search that found it began: an
    onset span is twelve symbols wide and a sweeping caller's window is two
    seconds. None from `PacketMemory`, which decodes a sum of copies and has no
    one buffer to point into.
    """

    @property
    def packet_count(self) -> int:
        return self.status & 0b11

    @property
    def data_type(self) -> int:
        """Status-byte bits 2-**3** (`spec.DataType`), the whole field in PACTOR-1.

        Widened to three bits to match `spec`, `rxfront` and `arq`, on consistency
        with the later layout rather than on the recordings, and the recordings
        refuse it. The 1990 description gives the Datenmodus as bits 2-3 and calls
        bit 4 `noch nicht belegt`; bit 4 joins the field only from PACTOR-2 on,
        where the PMC modes need it. Both PACTOR-1 stations the corpus can check
        agree: DL6MAA's 200 Bd announcement, status 0x35, is plain Huffman under
        two bits and decompresses to `1dl6maa` -- the nine characters an
        independent decoder reads off the same audio -- while three bits call it
        PMC German swapped and give `1DIT)   WS DUNZ0 AN`; W4DNA's 0x31 is ASCII
        `1w4dna` under two and PMC German under three. What bit 4 carries here is
        the capability declaration of bits 4-5, which is what a modern gateway
        answers with a grant, and it is legible in the rendered status byte.

        AND THE INDEPENDENT DECODER DOES BOTH, which the monitor line
        deliberately does not follow. It prints the WIDE value -- `TYPE 5` for
        DL6MAA, `TYPE: 4` for W4DNA and for our own 0x31 render -- and then
        decodes all three under the narrow one. A number rendered here that
        contradicted the characters printed beside it is what let the three-bit
        read stand.
        """
        return (self.status >> 2) & 0b11

    @property
    def changeover_request(self) -> bool:
        """Bit 6: the sending station is asking us to take the channel.

        An ISS with nothing left to say holds this up on idle packets until the
        other end's changeover packet arrives, so its repeats are one standing
        request and not a station claiming the channel -- which is the reading
        `arq.PactorArq.on_rx_packet` grades them against.
        """
        return bool(self.status & spec.STATUS_CHANGEOVER)

    @property
    def qrt(self) -> bool:
        return bool(self.status & spec.STATUS_QRT)


def decode_p1(audio: np.ndarray, fs: int = FS) -> str:
    """PACTOR-1 ARQ data packets -> text (8-bit ASCII, data type 0).

    Inverts pactor1.data_packet: each ARQ cycle is one FSK burst carrying
    [header][field][status][CRC-16], 12 bytes at 100 Bd (8-byte field) or 24 at
    200 Bd (20-byte field). Duplicate memory-ARQ copies are emitted once.

    "Only CRC-valid packets contribute text, so this never fabricates a decode"
    stood here, and it was wrong by a factor of the trial count: the scan reads
    thousands of alignments a second and a 16-bit checksum passed one of them every
    1.8 s of pure noise. What makes the claim true is the header and eye gates on
    `decode_p1_packets`; on audio with no clean P1 data packet this now does return
    "", measured over the 902-capture corpus, 10.36 hours.

    VALIDATION: three stations that are not ours, byte-exact off the recordings --
    W4DNA's 100 Bd announcement, the JN36lf station's two fields, and DL6MAA's
    200 Bd announcement, whose field decompresses to the callsign an independent
    monitor reads off the same audio. The TX<->RX round-trip against the frame
    structure of ITU-R M.1798 (tests/shrike/test_p1rx.py) stands behind those and
    not in front of them.
    """
    return "".join(p.payload.decode("latin-1") for p in decode_p1_packets(audio, fs))


# A PACTOR-1 data frame carries its header byte OUTSIDE the CRC, and the
# description allows it exactly two values: "1) Header : Bitmuster 55(HEX) ... Bei
# jedem Paket, das neue Information enthaelt, wird das Bitmuster invertiert."
# Eight bits the CRC does not cover -- and they are needed, because a CRC-16 alone
# is not a gate for a scan that reads thousands of candidate alignments per second.
# The two values are complements, so the test is blind to the shift sense, which a
# gate on a polarity-ambiguous mode has to be.
#
# It is confirmed on both stations the corpus has: W4DNA's link-setup announcement
# reads 0xAA, the JN36lf 14110 kHz station's two frames read 0x55 then 0xAA, and our
# own transmitter alternates the pair across the packet counter.
P1_DATA_HEADERS = (pactor1.SYNC_HEADER, pactor1.DATA_HEADER)

RISE_SYMBOLS = 6
"""How far EITHER SIDE of a burst's envelope onset a frame's first symbol may lie.

`_burst_onsets` fires where the envelope crosses a fraction of the buffer peak,
which is a point somewhere inside the transmitter's keying rise, and the frame can
begin on either side of it. Both cases are measured, and each is a whole station's
worth of packets:

  * the onset LATE -- a peer keys up into its own first byte, so the carrier
    reaches the threshold after the frame has started. Over the 200 Bd hold cycles
    of `captures/onair-0818-2235`, the eight whose copy sits within two bits of the
    frame it is a copy of lead their nearest onset by 0.24, 0.40, 0.58, 0.95, 1.00,
    2.01, 3.01 and 6.08 symbols.
  * the onset EARLY -- a peer brings its carrier up and starts the frame once it is
    up. DL6MAA's 200 Bd announcement begins 16.9 ms, 3.4 symbols, past its own
    onset; the 100 Bd frames of W4DNA and of the JN36lf station begin within 0.9.

Six symbols is 30 ms at 200 Bd and 60 at 100, and a keying rise is a property of the
two stations rather than of the symbol rate -- so a bound in SYMBOLS is generous at
the slower rate and just covers both measurements at the faster one, which is where
both were taken.
"""

PREFIX_FLOOR = 0.25
"""Envelope, against the frame's own median, below which a head symbol is charged
for nothing.

The CRC covers [field][status] and NOT the head, so a frame whose opening symbols
were keyed into the rise verifies completely while its head reads noise -- and the
head is what keeps a 16-bit checksum from accepting a coincidence. Charging it only
where the burst is up keeps both: on noise every symbol is present and the gate is
the whole word, and it opens by exactly the symbols the rise ate. Over the same
cycles a head symbol inside the rise reads 0.00 to 0.06 of the frame's median
envelope and the first one clear of it 0.32 or better, so the two populations do
not touch and the floor sits between them.

What a rise eats is a LEADING RUN of head symbols and nothing else, so that is all
the excuse covers. A head with a hole in the middle of it is a fade, not a keying
edge, and reading it as one is what let a low-level flutter on 14105 kHz through
with five head symbols present and an eye of exactly 0.50.
"""

PREFIX_MIN_SYMBOLS = 2
"""How much of the head has to be present for the gate to stand at all.

Below this the accept rests on the CRC and the eye alone, which `decode_p1_packets`
argues is not enough on its own. Two of the eight head symbols of a 200 Bd data
packet is what the deepest rise in `captures/onair-0818-2235` leaves standing.
"""

EYE_MIN = 0.50
"""How open the FSK eye must be, median over the frame's symbols, for a CRC-valid
candidate to be a frame rather than a coincidence.

Per symbol the score is |MARK - SPACE| / (MARK + SPACE) off the magnitudes the bit
decision already used, so it costs one median per surviving candidate and asks the
only question the CRC cannot: is this alignment reading two-tone keying at all, or
noise that happened to checksum?

The threshold sits between two MEASURED populations, and both margins are
measurements rather than guesses:

  * the worst CRC-valid non-frame candidate a narrower search reached, over
    419,740,806 alignments, scores 0.445, and it is a 100 Bd signal read at 200 Bd
    (its payload is the giveaway alternating 55/AA). Over 2.9M alignments of pure
    white noise the ceiling is 0.416 at 100 Bd and 0.362 at 200 Bd. Searching both
    sides of the onset reaches further: a flutter on 14105 kHz passes the CRC at
    0.5056, ABOVE this threshold, so on that one the eye is not what refuses it and
    the head is. The null rests on two gates and never on this one alone.
  * the weakest CORRECT decode, at an SNR where the frame still decodes 9 times in
    10, scores 0.561. A synthetic packet in real off-air noise decodes reliably to
    +3 dB (100 Bd) and +4 dB (200 Bd) in 400 Hz, and the CRC gate itself collapses
    a decibel or two below that -- 3 of 24 at +2 dB -- so what this rejects is a
    regime that was already failing, not working frames.

Real off-air frames measure far above it: 0.885 for W4DNA, 0.760 and 0.791 for the
two JN36lf frames, 0.97 for DL6MAA's 200 Bd announcement.
"""


def decode_p1_packets(audio: np.ndarray, fs: int = FS,
                      tones100: "_Tones | None" = None, *,
                      breakin: bool = False) -> list[Packet]:
    """`tones100` is this audio's 100 Bd `tone_series` when the caller has one --
    the 200 Bd series is this decoder's alone and is always built here.

    `breakin` looks for the CHANGEOVER packet instead: same length, same CRC
    region, but the header byte is replaced by the CS3 head (2 bytes at 100 Bd,
    3 at 200), so the gate is 16 or 24 known bits outside the CRC rather than 8.
    It is a separate pass because the two frames cannot both be present at one
    alignment and a station knows which it is expecting -- an ISS is expecting a
    break-in, an IRS is expecting data.

    This is the ONLY way a real CS3 can reach the state machine. The
    control-signal path cannot carry it: `_p1_cs_bursts` accepts a run of 60-260 ms
    and a break-in packet is one continuous 960 ms run, and `cs_bits` positions its
    window by requiring the two bit slots after the word to be QUIET, which is true
    of a bare burst and false of a packet head by construction.

    THREE gates, and the CRC is the weakest of them. This scan reads two bauds x two
    polarities x 97 sub-symbol offsets per envelope rising edge -- about 59000
    candidate alignments per second of corpus audio and 279000 per second of noise,
    which crosses the envelope threshold far more often -- so a 16-bit checksum alone
    accepts constantly. That is not a coding bug and no amount of care in the CRC
    fixes it: swept in 30 s chunks over the 902 captures of the corpus, 10.36 hours
    and 2,185,085,562 alignments, the CRC alone accepts 33161 times, which is
    combinatorics working exactly as it must.

    The head and eye gates are what make the answer mean something. Over that same
    sweep they leave 56 accepts, and every one is in one of the three recordings that
    carry a real PACTOR-1 data frame -- W4DNA's five announcement cycles, DL6MAA's
    200 Bd announcement and the JN36lf station's two fields -- with nothing anywhere
    else in the corpus and nothing at all on four minutes of white noise.
    """
    audio = _to_fs(audio, fs)
    out: list[Packet] = []
    seen: set[tuple[int, int, bytes]] = set()
    for baud, nbytes in ((100, 12), (200, 24)):
        wants = (_prefix_words(pactor1.breakin_head(baud)) if breakin
                 else _prefix_words(*(bytes([h]) for h in P1_DATA_HEADERS)))
        nskip = wants[0].size // 8
        sps = FS // baud
        step = max(1, sps // 8)
        tones = tones100 if baud == 100 and tones100 is not None else _Tones(audio, sps)
        # Every onset's candidate alignments are read and CRC-gated in one pass --
        # a busy 2 s buffer offers a hundred-odd onsets and the per-onset call was
        # too small a batch to pay for itself. `edge` puts the surviving frames back
        # under their own onset, so the scan order is unchanged.
        spans = [np.arange(max(0, o - RISE_SYMBOLS * sps),
                           o + RISE_SYMBOLS * sps + 1, step)
                 for o in _burst_onsets(tones)]
        if not spans:
            continue
        starts = np.concatenate(spans)
        onset_of = np.repeat(np.arange(len(spans)), [s.size for s in spans])
        keep = tones.fits(starts, sps, nbytes * 8)
        starts, onset_of = starts[keep], onset_of[keep]
        if starts.size == 0:
            continue
        edge = np.searchsorted(onset_of, np.arange(len(spans) + 1))
        reads = []
        for invert in (False, True):
            f = tones.read(starts, sps, nbytes, invert)
            ok = _crc_pass(f, nskip)
            # The head and the eye are measured only on what the CRC already
            # passed -- a median over 96 or 192 symbol windows, and a second read
            # of the head's magnitudes, are far too costly to run on every
            # alignment, and by construction neither can ever admit.
            hit = np.flatnonzero(ok)
            if hit.size:
                ok[hit] = (_prefix_ok(tones, starts[hit], sps, nbytes * 8,
                                      wants, invert)
                           & (_eye(tones, starts[hit], sps, nbytes * 8) > EYE_MIN))
            reads.append((invert, f, ok))
        for lo, hi in zip(edge, edge[1:]):
            for invert, frames, ok in reads:
                for i in np.flatnonzero(ok[lo:hi]) + lo:
                    pkt = frames[i].tobytes()
                    header, field, status = pkt[0], pkt[nskip:-3], pkt[-3]
                    key = (baud, status, field)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Packet(pactor1.field_bytes(field, (status >> 2) & 3),
                                      status, header,
                                      baud, invert, breakin, int(starts[i])))
    return out


def _crc_pass(frames: np.ndarray, skip: int = 1) -> np.ndarray:
    """Which candidate frames carry a valid CRC-16 over their protected region
    [field][status] -- the two CRC bytes LOW BYTE FIRST.

    `skip` is what precedes the protected region: one header byte on a data packet,
    and the CS3 head on a changeover packet.

    Both the variant and the byte order are measured off a real off-air packet,
    not read off the description, which says only "nach CCITT-Norm". This read
    them big-endian and CCITT-FALSE, which agreed perfectly with our transmitter
    and with nothing else on the air -- decode_p1_packets returned nothing at all
    on a real capture."""
    want = (frames[:, -1].astype(np.uint32) << 8) | frames[:, -2]
    return coding.crc16_rows(frames[:, skip:-2], pactor1.DATA_CRC) == want


def _eye(tones: "_Tones", starts: np.ndarray, sps: int,
         nbits: int) -> np.ndarray:
    """Median |MARK - SPACE| / (MARK + SPACE) over each candidate frame's symbols.

    Normalised per symbol, so it measures the eye and not the signal level, and a
    frame that fades across its own 0.96 s is scored on how it read rather than on
    how loud it was. See EYE_MIN for the two populations it separates.
    """
    idx = starts[:, None] + np.arange(nbits) * sps
    m, s = tones.mark[idx], tones.space[idx]
    return np.median(np.abs(m - s) / np.maximum(m + s, 1e-12), axis=1)


def _prefix_words(*words: bytes) -> list[np.ndarray]:
    """Each head a frame may open with, as the bits in the order they are keyed."""
    return [np.unpackbits(np.frombuffer(w, np.uint8),
                          bitorder="little").astype(bool) for w in words]


def _prefix_ok(tones: "_Tones", starts: np.ndarray, sps: int, nbits: int,
               wants: list[np.ndarray], invert: bool) -> np.ndarray:
    """Which candidates open with one of `wants`, charged only where the burst is up.

    The head is the frame's unprotected bytes -- the 0x55/0xAA alternation of a data
    packet, the CS3 word of a changeover -- so this is the whole of what stands
    between a CRC-valid alignment and a coincidence. It is also the part a rise eats
    (PREFIX_FLOOR), and a symbol the receiver never heard says nothing either way --
    but only where the rise is what silenced it, which is a LEADING run and never a
    hole further in.
    """
    idx = starts[:, None] + np.arange(wants[0].size) * sps
    floor = PREFIX_FLOOR * np.median(
        tones.env[starts[:, None] + np.arange(nbits) * sps], axis=1)
    up = tones.env[idx] > floor[:, None]
    eaten = np.cumprod(~up, axis=1, dtype=bool)
    bits = (tones.mark[idx] > tones.space[idx]) ^ invert
    ok = np.zeros(starts.size, bool)
    for w in wants:
        ok |= ((bits == w) | eaten).all(axis=1)
    return (ok & (up | eaten).all(axis=1)
            & (up.sum(axis=1) >= PREFIX_MIN_SYMBOLS))


def _burst_onsets(tones: "_Tones", thresh: float = 0.3) -> list[int]:
    """Rising edges of the FSK envelope -- one candidate onset per packet burst."""
    env = tones.env
    if env.size == 0 or env.max() <= 0:
        return []
    hot = env > thresh * env.max()
    rises = (np.where(hot[1:] & ~hot[:-1])[0] + 1).tolist()
    return ([0] if hot[0] else []) + rises


# --- memory ARQ -------------------------------------------------------------

def packet_softs(audio: np.ndarray, fs: int = FS,
                 baud: int = 100) -> np.ndarray | None:
    """Per-bit soft tone difference of the data packet this window holds.

    The soft value is MARK minus SPACE at each bit instant, in time order,
    normalised to the frame's own mean magnitude -- the same quantity whose SIGN
    `_Tones.read` hardens and whose SPLIT `_cs_decide` reasons about. It is what
    a frame with no FEC has instead of a channel decoder: a copy too noisy to
    read alone still carries the field in the sign bias of these values, and
    summing copies is the only error correction the mode offers
    (`PacketMemory`).

    The window is POSITIONED the way `cs_bits` positions a control signal --
    widest eye over the frame's bits, quiet in the two bit slots after the unkey
    -- and by nothing else: the score never sees the header bytes or the CRC,
    so exactly one alignment's softs leave here, and what the combined decode
    later spends on origins is charged where it is spent (`PacketMemory`).
    Audio past the capture counts as silence for `cs_bits`' reason: an
    alignment that runs off the end scores badly instead of winning by having
    no tail to be charged for.

    The returned vector carries `MEMORY_SLIP_BITS` extra bit slots either side
    of the positioned frame, because near the decode cliff the positioner slips
    by whole bits (see that constant) and a copy trimmed to exactly the frame
    it guessed has thrown away the bits a slipped guess needs back. The frame
    the positioner chose starts at index `MEMORY_SLIP_BITS`; where the margin
    runs off the capture it reads as zero, a soft value that says nothing.
    """
    x = _to_fs(np.asarray(audio, dtype=float), fs)
    sps = FS // baud
    nbits = {100: 12, 200: 24}[baud] * 8
    if x.size < nbits * sps:
        return None
    pad = MEMORY_SLIP_BITS * sps
    T = _Tones(np.concatenate([np.zeros(pad), x,
                               np.zeros(pad + CS_TAIL_BITS * sps)]), sps)
    spans = [np.arange(max(pad, o - sps), max(pad, o - sps) + 2 * sps + 1,
                       max(1, sps // 8))
             for o in _burst_onsets(T)]
    if not spans:
        return None
    starts = np.unique(np.concatenate(spans))
    starts = starts[(starts >= pad)
                    & (starts + (nbits + CS_TAIL_BITS - 1) * sps < T.n - pad)]
    if starts.size == 0:
        return None
    eye = starts[:, None] + np.arange(nbits) * sps
    tail = starts[:, None] + (nbits + np.arange(CS_TAIL_BITS)) * sps
    score = (np.abs(T.mark[eye] - T.space[eye]).sum(axis=1)
             - T.env[tail].sum(axis=1))
    best = int(starts[int(np.argmax(score))])
    idx = (best + (np.arange(nbits + 2 * MEMORY_SLIP_BITS)
                   - MEMORY_SLIP_BITS) * sps)
    d = T.mark[idx] - T.space[idx]
    frame = d[MEMORY_SLIP_BITS:MEMORY_SLIP_BITS + nbits]
    scale = float(np.abs(frame).mean())
    return d / scale if scale > 0 else None


MEMORY_COPIES = 4
"""Packet copies `PacketMemory` holds per geometry at most, oldest dropped first.

A bound, not a measurement, and `p2rx.MEMORY_COPIES` is its precedent: it caps
what combining offers the CRC and it is how a mis-grouped copy leaves. A sum
straddling a field boundary decodes to nothing -- the header and CRC gates
refuse it -- and the sliding window ages the stale copy out within this many
cycles instead of carrying it for the rest of the session."""

MEMORY_SLIP_BITS = 8
"""Whole-bit slips either side over which copies and the frame origin can move.

`packet_softs` positions each copy blind to the codeword, and near the decode
cliff its eye score slips by whole bits: measured on rendered copies in seeded
noise at the sigma where single-shot decoding fails, every mis-positioned copy
sat a clean -2 to +6 bits off, reading 0.96-0.99 correlation against the frame
once shifted and 0.06-0.3 where it lay; the real off-air jn36lf copy degraded
the same way slipped -1 to -2. One copy in the deepest sweep lay 25 bits out
at 0.8 correlation -- past any sane bound, and the copy window ages it out
rather than the bound chasing it. The bound covers the rest with margin, twice:
`add` aligns each new copy to the accumulator over these slips, and `_decode`
tries each origin inside the same margin -- because when the FIRST copy is the
slipped one, every later copy aligns onto its frame and only the decode can
put the origin back. The alignment search never sees the header bytes or the
CRC; the origin search does, and the class docstring charges for it."""


def _slid(d: np.ndarray, s: int) -> np.ndarray:
    """`d` moved `s` whole bits later, zero-filled -- silence, as elsewhere."""
    out = np.zeros_like(d)
    if s >= 0:
        out[s:] = d[:d.size - s]
    elif -s < d.size:
        out[:s] = d[-s:]
    return out


class _Lane:
    """One geometry's accumulator: aligned copies, and the last copy's flip."""

    __slots__ = ("copies", "flipped")

    def __init__(self) -> None:
        self.copies: list[np.ndarray] = []
        self.flipped = False


class PacketMemory:
    """Soft combining across repeats of one unacknowledged PACTOR-1 packet.

    A PACTOR-1 packet is `[header][field][status][CRC-16]` with no FEC at all:
    one bit error destroys it and nothing downstream can repair it. Repetition
    is the mode's whole error-correction budget -- an unacked packet is sent
    again, identical, every 1.25 s cycle until acknowledged -- so a receiver
    that reads each copy alone and throws the failures away is discarding the
    only redundancy the protocol transmits. This sums the per-bit soft values
    (`packet_softs`) of consecutive failed copies and offers the CRC one decode
    of the sum, which is `p2rx.BurstMemory` rebuilt on a mark/space split
    instead of complex phasors.

    What makes the sum meaningful:

      * copies are grouped by CONSECUTIVE FAILURE within one geometry, not by
        reading the header or counter -- an undecoded copy's bytes are exactly
        what cannot be read. Any single-shot decode clears the memory
        (`clear`), so the copies here are the bursts between decodes, which
        under ARQ repetition are copies of one packet whenever the grouping is
        right. When it is wrong the sum is two frames' softs superposed and the
        header and CRC gates refuse it: the failure is a cycle reported
        undecoded, never a wrong field delivered.
      * geometry is the BAUD. A peer can drop 200 -> 100 Bd on a
        retransmission (CS4 after a faulty packet means exactly that), and the
        two layouts are not combinable, so each baud accumulates in its own
        lane and a decode in either clears both.
      * THE SHIFT INVERTS EVERY CYCLE -- "mit jedem neuen Paket oder
        Kontrollsignal wird die Shiftlage invertiert", corpus 85/85, 34/34,
        6/6, both directions -- so consecutive copies arrive with MARK and
        SPACE exchanged and summing them raw adds mark to space and cancels.
        The sense is established FROM THE DATA, per copy: the sign of the
        copy's correlation against the accumulated sum, which is `_aligned`'s
        construction in `p2rx` collapsed from a phase to a sign. A cycle
        counter is never consulted, because a missed cycle would silently
        invert every copy after it.

    Trials the gates are offered: one decode of the sum per `add` that holds
    two copies or more -- `2 * MEMORY_SLIP_BITS + 1` frame origins by two
    polarities, 34 alignments, each against an 8-bit header alternation (two
    accepted values) and the CRC-16. That is about 34 * 2^-23 = 4e-6 expected
    false accepts per add, against the same 23-bit gate the single-shot scan
    `decode_p1_packets` already offers thousands of alignments per second; a
    session whose every cycle fails for an hour accumulates ~0.01 expected
    false accepts from combining, the size of `p3rx.TRIAL_BUDGET`. A lone
    failure costs nothing: its only copy was already scanned single-shot.
    """

    def __init__(self) -> None:
        self._lanes: dict[int, _Lane] = {}

    def clear(self) -> None:
        self._lanes.clear()

    def add(self, softs: np.ndarray, baud: int = 100) -> Packet | None:
        """Absorb one undecoded copy's softs; return the combined packet, if any.

        A delivered packet clears the memory: whatever follows is either its
        successor or a fresh repeat, and both stand alone again. The returned
        `Packet.inverted` is the LAST copy's on-air shift -- the current
        cycle's, which is the one a grid taking its phase from the packet needs.
        """
        lane = self._lanes.setdefault(baud, _Lane())
        d = np.asarray(softs, dtype=float)
        lane.flipped = False
        if lane.copies:
            # Aligned to the accumulator over sign AND whole-bit slips, both
            # read from the data. The sense is per copy, not cumulative -- the
            # accumulator already sits in the first copy's sense, so each new
            # copy's flip is its own correlation sign, carrying nothing from
            # the copy before it. On a stranger the best of these correlations
            # is still noise, the sum part-cancels, and the gates in `_decode`
            # stay shut.
            ref = np.sum(lane.copies, axis=0)
            score, d = max(
                ((float(ref @ shifted), shifted)
                 for s in range(-MEMORY_SLIP_BITS, MEMORY_SLIP_BITS + 1)
                 for shifted in (_slid(d, s),)),
                key=lambda t: abs(t[0]))
            lane.flipped = score < 0
            if lane.flipped:
                d = -d
        lane.copies.append(d)
        del lane.copies[:-MEMORY_COPIES]
        if len(lane.copies) < 2:
            return None
        pkt = self._decode(np.sum(lane.copies, axis=0), baud, lane.flipped)
        if pkt is not None:
            self.clear()
        return pkt

    @staticmethod
    def _decode(d: np.ndarray, baud: int, flipped: bool) -> Packet | None:
        """One combined decode, behind the same gates a single copy faces.

        The frame origin is searched inside the margin the copies carry,
        because the accumulator sits in its FIRST copy's frame and that copy
        can be the slipped one -- every later copy then aligns onto the slip,
        and no per-copy correction can see it. `flipped` is whether the last
        copy was complemented onto the accumulator's reference, so
        `invert ^ flipped` is that copy's own on-air sense -- see `add`."""
        nbytes = {100: 12, 200: 24}[baud]
        nbits = nbytes * 8
        for o in range(2 * MEMORY_SLIP_BITS + 1):
            frame = d[o:o + nbits]
            for invert in (False, True):
                row = np.packbits(((frame > 0) ^ invert).reshape(nbytes, 8),
                                  axis=1, bitorder="little")[:, 0]
                if int(row[0]) not in P1_DATA_HEADERS:
                    continue
                if not _crc_pass(row[None, :])[0]:
                    continue
                pkt = row.tobytes()
                return Packet(pactor1.field_bytes(pkt[1:-3], (pkt[-3] >> 2) & 3),
                              pkt[-3], pkt[0], baud, invert ^ flipped)
        return None


# --- io --------------------------------------------------------------------

def _to_fs(audio: np.ndarray, fs: int) -> np.ndarray:
    if fs == FS:
        return np.asarray(audio, dtype=float)
    from scipy.signal import resample_poly
    g = gcd(int(fs), FS)
    return resample_poly(np.asarray(audio, dtype=float), FS // g, int(fs) // g)


def load_wav(path: str) -> np.ndarray:
    return session.load_wav(path, FS)
