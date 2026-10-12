# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 4539's high-rate waveform: 3200 to 9600 bit/s, and 12800 uncoded.

Written from MIL-STD-188-110B appendix C, which specifies it. 2400 symbols a
second on 1800 Hz, as 188-110's serial tone, but the data symbols are QPSK,
8-PSK, 16-, 32- or 64-QAM (table C-II), 256 of them to a frame followed by a
31-symbol mini-probe; a preamble names the data rate and interleaver, and
after every 72 frames the last 72 symbols of it are sent again (C.5.2).

**Coding.** The K=7 code of 188-110, made a block code the length of the
interleaver by full tail-biting and punctured to rate 3/4 (C.5.3.2). The block
interleaver loads bit n at n x increment modulo its size and is read out in
order (C.5.3.3); its blocks line up with the reinserted preamble.

**Scrambling.** The data symbols only: 8-PSK symbol numbers by adding, QAM
ones by exclusive-or, the value from a 9-stage register (x^9 + x^4 + 1) set to
1 at every frame's start and stepped once for each bit used (figure C-5).

**Autobaud.** Rate and interleaver are three bits each, sent twice over: as
the phases of three 13-chip Barker codes in the preamble (table C-IX), and as
the signs of the mini-probes, eighteen to a set (tables C-XII, C-XIII).

Messages are framed by the optional 32-bit end of message, 4B65A5B2, and
padded with zeros to whole input blocks (C.5.4.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fec, phy, spec
from .mil110 import bytes_to_bits
from .spec import BAUD, SPS, psk8

# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

#: Table C-II, C-X: data rate -> (bits a symbol, the 3-bit code).
RATES = {3200: (2, 0b001), 4800: (3, 0b010), 6400: (4, 0b011), 8000: (5, 0b100),
         9600: (6, 0b101), 12800: (6, 0b110)}
#: Table C-X: interleaver frames -> 3-bit code, and its name.
INTERLEAVERS = {1: (0b001, "US"), 3: (0b010, "VS"), 9: (0b011, "S"), 18: (0b100, "M"),
                36: (0b101, "L"), 72: (0b110, "VL")}
_RATE_OF = {c: r for r, (_, c) in RATES.items()}
_IL_OF = {c: f for f, (c, _) in INTERLEAVERS.items()}

#: Table C-XVI: the interleaver increment, by rate and frames.
INCREMENT = {
    3200: {1: 97, 3: 229, 9: 805, 18: 1393, 36: 3281, 72: 6985},
    4800: {1: 145, 3: 361, 9: 1045, 18: 2089, 36: 5137, 72: 10273},
    6400: {1: 189, 3: 481, 9: 1393, 18: 3281, 36: 6985, 72: 11141},
    8000: {1: 201, 3: 601, 9: 1741, 18: 3481, 36: 8561, 72: 14441},
    9600: {1: 229, 3: 805, 9: 2089, 18: 5137, 36: 10273, 72: 17329},
}

#: Table C-III and C-IV: bit group -> 8-PSK symbol number.
QPSK_MAP = (0, 2, 6, 4)                 # 00, 01, 10, 11
PSK8_MAP = (1, 0, 2, 3, 6, 7, 5, 4)     # 000 ... 111

#: Tables C-V, C-VI, C-VII.
QAM16 = np.array([
    0.866025 + 0.500000j, 0.500000 + 0.866025j, 1.000000 + 0.000000j, 0.258819 + 0.258819j,
    -0.500000 + 0.866025j, 0.000000 + 1.000000j, -0.866025 + 0.500000j, -0.258819 + 0.258819j,
    0.500000 - 0.866025j, 0.000000 - 1.000000j, 0.866025 - 0.500000j, 0.258819 - 0.258819j,
    -0.866025 - 0.500000j, -0.500000 - 0.866025j, -1.000000 + 0.000000j, -0.258819 - 0.258819j])
_Q32 = [(0.866380, 0.499386), (0.984849, 0.173415), (0.499386, 0.866380), (0.173415, 0.984849),
        (0.520246, 0.520246), (0.520246, 0.173415), (0.173415, 0.520246), (0.173415, 0.173415)]
QAM32 = np.array([complex(sx * i, q) for sx in (1, -1) for i, q in _Q32]
                 + [complex(sx * i, -q) for sx in (1, -1) for i, q in _Q32])
_Q64 = [(1.000000, 0.000000), (0.822878, 0.568218), (0.821137, 0.152996), (0.932897, 0.360142),
        (0.000000, -1.000000), (0.822878, -0.568218), (0.821137, -0.152996), (0.932897, -0.360142),
        (0.568218, 0.822878), (0.588429, 0.588429), (0.588429, 0.117686), (0.588429, 0.353057),
        (0.568218, -0.822878), (0.588429, -0.588429), (0.588429, -0.117686), (0.588429, -0.353057),
        (0.152996, 0.821137), (0.117686, 0.588429), (0.117686, 0.117686), (0.117686, 0.353057),
        (0.152996, -0.821137), (0.117686, -0.588429), (0.117686, -0.117686), (0.117686, -0.353057),
        (0.360142, 0.932897), (0.353057, 0.588429), (0.353057, 0.117686), (0.353057, 0.353057),
        (0.360142, -0.932897), (0.353057, -0.588429), (0.353057, -0.117686), (0.353057, -0.353057)]
#: Symbols 32-63 mirror 0-31 in the in-phase axis (table C-VII), except that
#: 32 and 36 are the points 0 and 4 turned a quarter: (0, 1) and (-1, 0).
QAM64 = np.array([complex(i, q) for i, q in _Q64]
                 + [complex(-i, q) for i, q in _Q64])
QAM64[32], QAM64[36] = 0 + 1j, -1 + 0j

TABLES = {4: QAM16, 5: QAM32, 6: QAM64}

#: Table C-VIII, the synchronisation preamble's fixed 184 symbols.
PREAMBLE184 = np.array([
    1, 5, 1, 3, 6, 1, 3, 1, 1, 6, 3, 7, 7, 3, 5, 4, 3, 6, 6, 4, 5, 4, 0,
    2, 2, 2, 6, 0, 7, 5, 7, 4, 0, 7, 5, 7, 1, 6, 1, 0, 5, 2, 2, 6, 2, 3,
    6, 0, 0, 5, 1, 4, 2, 2, 2, 3, 4, 0, 6, 2, 7, 4, 3, 3, 7, 2, 0, 2, 6,
    4, 4, 1, 7, 6, 2, 0, 6, 2, 3, 6, 7, 4, 3, 6, 1, 3, 7, 4, 6, 5, 7, 2,
    0, 1, 1, 1, 4, 4, 0, 0, 5, 7, 7, 4, 7, 3, 5, 4, 1, 6, 5, 6, 6, 4, 6,
    3, 4, 3, 0, 7, 1, 3, 4, 7, 0, 1, 4, 3, 3, 3, 5, 1, 1, 1, 4, 6, 1, 0,
    6, 0, 1, 3, 1, 4, 1, 7, 7, 6, 3, 0, 0, 7, 2, 7, 2, 0, 2, 6, 1, 1, 1,
    2, 7, 7, 5, 3, 3, 6, 0, 5, 3, 3, 1, 0, 7, 1, 1, 0, 3, 0, 4, 0, 7, 3])

#: C.5.2.2: the mini-probe, "+", and its inverse, "-".
PROBE_PLUS = np.array([0, 0, 0, 0, 0, 2, 4, 6, 0, 4, 0, 4, 0, 6, 4, 2,
                       0, 0, 0, 0, 0, 2, 4, 6, 0, 4, 0, 4, 0, 6, 4])
PROBE_MINUS = (PROBE_PLUS + 4) % 8
BARKER13 = np.array([0, 4, 0, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0])

EOM_BITS = spec.EOM110_BITS

DATA_PER_FRAME = 256
PROBE_LEN = 31
FRAME = DATA_PER_FRAME + PROBE_LEN
SUPERFRAME = 72                          # frames between reinserted preambles
REINSERT = 72                            # the reinserted preamble's symbols


@dataclass(frozen=True)
class Mode4539:
    rate: int                            # bit/s
    frames: int                          # interleaver length, frames

    @property
    def name(self) -> str:
        return f"{self.rate}{INTERLEAVERS[self.frames][1]}"

    @property
    def bits_per_symbol(self) -> int:
        return RATES[self.rate][0]

    @property
    def coded(self) -> bool:
        return self.rate != 12800

    @property
    def block_bits(self) -> int:
        """Table C-XIV: input data bits a block (C-XV's size less a quarter)."""
        n = self.interleaver_bits
        return n * 3 // 4 if self.coded else n

    @property
    def interleaver_bits(self) -> int:
        return DATA_PER_FRAME * self.frames * self.bits_per_symbol

    @property
    def increment(self) -> int:
        return INCREMENT[self.rate][self.frames] if self.coded else 1


def _all_modes() -> dict[str, Mode4539]:
    out = {}
    for r in RATES:
        for f in INTERLEAVERS:
            if r == 12800 and f != 1:
                continue                 # table C-IX: 12800 has no interleaver
            m = Mode4539(r, f)
            out[m.name] = m
    return out


MODES4539 = _all_modes()


def mode_of(rate_code: int, il_code: int) -> Mode4539 | None:
    r, f = _RATE_OF.get(rate_code), _IL_OF.get(il_code)
    if r is None or f is None or (r == 12800 and f != 1):
        return None
    return Mode4539(r, f)


# --------------------------------------------------------------------------
# The known symbols
# --------------------------------------------------------------------------

def d_values(mode: Mode4539) -> tuple[int, int, int]:
    """Table C-IX: three dibits, each a rate bit (the more significant) and an
    interleaver bit, as 8-PSK symbols by table C-III."""
    rc, ic = RATES[mode.rate][1], INTERLEAVERS[mode.frames][0]
    return tuple(QPSK_MAP[(((rc >> (2 - i)) & 1) << 1) | ((ic >> (2 - i)) & 1)]
                 for i in range(3))


def known103(mode: Mode4539) -> np.ndarray:
    """Table C-XI: the preamble's last 103 symbols, which are also a mini-probe
    and the reinserted preamble: "+" and a 32nd symbol, the three Barker
    codes shifted by D0, D1, D2, and "-" with a symbol ahead of it."""
    barkers = [(d + BARKER13) % 8 for d in d_values(mode)]
    return np.concatenate([PROBE_PLUS, [2], *barkers, [6], PROBE_MINUS])


def preamble(mode: Mode4539) -> np.ndarray:
    """C.5.2.1.1: 287-symbol synchronisation preamble, no AGC blocks."""
    return np.concatenate([PREAMBLE184, known103(mode)])


def probe_signs(mode: Mode4539) -> np.ndarray:
    """Mini-probes 1 to 72's signs, True for "+" (C.5.2.2): each set of 18 is
    seven "-", a "+", the six rate and interleaver bits (table C-XII: "-" for
    a 1), three bits counting the set (table C-XIII), and a "+"."""
    rc, ic = RATES[mode.rate][1], INTERLEAVERS[mode.frames][0]
    six = [(rc >> 2) & 1, (rc >> 1) & 1, rc & 1, (ic >> 2) & 1, (ic >> 1) & 1, ic & 1]
    out = []
    for s in range(4):
        cnt = s + 1
        bits = six + [(cnt >> 2) & 1, (cnt >> 1) & 1, cnt & 1]
        out += [False] * 7 + [True] + [b == 0 for b in bits] + [True]
    return np.array(out)


def probe(mode: Mode4539, number: int) -> np.ndarray:
    """Mini-probe ``number``, 1 to 72, after the data block of that number."""
    return PROBE_PLUS if probe_signs(mode)[number - 1] else PROBE_MINUS


# --------------------------------------------------------------------------
# Scrambling (C.5.1.3)
# --------------------------------------------------------------------------

def scramble_values(bits_per_symbol: int, n: int = DATA_PER_FRAME) -> np.ndarray:
    """A frame's scrambling values, one a data symbol: the register's last
    3 (8-PSK), 4, 5 or 6 stages (the leftmost the most significant), the
    register then shifted that many times (figure C-5: the stages move
    right, and the leftmost takes the fifth-from-left stage's output (index 4)
    exclusive-or'd with the last's: s[n-5] XOR s[n-9]). This reading
    remains unverified against a foreign 4539 implementation."""
    nb = 3 if bits_per_symbol <= 3 else bits_per_symbol
    r = [0] * 8 + [1]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        v = 0
        for b in r[9 - nb:]:
            v = (v << 1) | b
        out[i] = v
        for _ in range(nb):
            fb = r[4] ^ r[8]
            r = [fb] + r[:8]
    return out


_SCR = {nb: scramble_values(nb) for nb in (2, 3, 4, 5, 6)}


# --------------------------------------------------------------------------
# Coding (C.5.3)
# --------------------------------------------------------------------------

_KEEP = np.array([True, True, True, False, False, True])      # the mask 111001


def encode_block(bits: np.ndarray, mode: Mode4539) -> np.ndarray:
    """One input block -> its interleaved bits, in the order they are sent."""
    if not mode.coded:
        return np.asarray(bits, dtype=np.uint8)
    coded = fec.encode_tailbiting(bits)
    punct = coded.reshape(-1, 6)[:, _KEEP].ravel()
    size = mode.interleaver_bits
    inter = np.empty(size, dtype=np.uint8)
    inter[(np.arange(size) * mode.increment) % size] = punct
    return inter


def decode_block(llr: np.ndarray, mode: Mode4539) -> np.ndarray:
    """An interleaver block's LLRs (positive for 0), in the order received ->
    the input bits."""
    if not mode.coded:
        return (np.asarray(llr) < 0).astype(np.uint8)
    size = mode.interleaver_bits
    punct = np.asarray(llr)[(np.arange(size) * mode.increment) % size]
    full = np.zeros((size // 4, 6))
    full[:, _KEEP] = punct.reshape(-1, 4)
    return fec.viterbi_tailbiting(full.ravel())


def symbol_numbers(bits: np.ndarray, mode: Mode4539) -> np.ndarray:
    """Interleaved bits -> symbol numbers before scrambling (the earliest bit
    the most significant): 8-PSK numbers through tables C-III and C-IV, QAM
    ones the bits themselves."""
    nb = mode.bits_per_symbol
    v = np.asarray(bits, dtype=np.int64).reshape(-1, nb) @ (1 << np.arange(nb - 1, -1, -1))
    if nb == 2:
        return np.asarray(QPSK_MAP)[v]
    if nb == 3:
        return np.asarray(PSK8_MAP)[v]
    return v


def data_frame(numbers: np.ndarray, mode: Mode4539) -> np.ndarray:
    """256 symbol numbers -> the frame's data symbols, scrambled, as points."""
    nb = mode.bits_per_symbol
    scr = _SCR[nb]
    if nb <= 3:
        return psk8((numbers + scr) % 8)
    return TABLES[nb][numbers ^ scr]


# --------------------------------------------------------------------------
# Transmit
# --------------------------------------------------------------------------

def _message_bits(data: bytes, mode: Mode4539, order: str) -> np.ndarray:
    bits = np.concatenate([bytes_to_bits(data, order), EOM_BITS])
    n = max(bits.size, 1)
    n += (-n) % mode.block_bits
    return np.concatenate([bits, np.zeros(n - bits.size, dtype=np.uint8)])


def symbols(data: bytes, mode: Mode4539 | str = "9600L", *,
            order: str = "lsb") -> np.ndarray:
    """A message as the transmission's complex symbols, preamble first."""
    if isinstance(mode, str):
        mode = MODES4539[mode]
    bits = _message_bits(data, mode, order)
    sent = np.concatenate([encode_block(b, mode)
                           for b in bits.reshape(-1, mode.block_bits)])
    nums = symbol_numbers(sent, mode).reshape(-1, DATA_PER_FRAME)
    rein = psk8(known103(mode)[-REINSERT:])
    out = [psk8(preamble(mode))]
    for f, frame in enumerate(nums):
        n = f % SUPERFRAME + 1
        if f and n == 1:
            out.append(rein)
        out.append(data_frame(frame, mode))
        out.append(psk8(probe(mode, n)))
    return np.concatenate(out)


def symbol_count(mode: Mode4539, n_bytes: int) -> int:
    """How many symbols ``symbols`` makes of an ``n_bytes`` message."""
    n = 8 * n_bytes + 32
    n = max(n, 1)
    frames = -(-n // mode.block_bits) * mode.frames
    return 287 + frames * FRAME + ((frames - 1) // SUPERFRAME) * REINSERT


def transmit(data: bytes, mode: Mode4539 | str = "9600L", *,
             order: str = "lsb", trim: bool = True) -> np.ndarray:
    """Message -> real audio at ``spec.FS``, peak 1.0. ``trim`` as
    `phy.modulate`."""
    return phy.modulate_symbols(symbols(data, mode, order=order), trim)


# --------------------------------------------------------------------------
# Receive
# --------------------------------------------------------------------------

@dataclass
class Reception:
    mode: Mode4539
    data: bytes
    frames: int
    start_sample: int
    end_sample: int
    freq_hz: float
    snr_db: float
    eom: bool = False


#: The channel read_barkers fits: symbol-spaced taps this many symbols
#: before and after the main one (5 ms of echo is 12 symbols).
_BARKER_TAPS = (2, 13)


def _all_d() -> list[tuple[int, int, int]]:
    return [(a, b, c) for a in (0, 2, 4, 6) for b in (0, 2, 4, 6) for c in (0, 2, 4, 6)]


def read_barkers(bb: np.ndarray, at: int, freq_hz: float,
                 before: np.ndarray | None = None) -> Mode4539 | None:
    """The rate and interleaver from the three Barker codes of the 103 known
    symbols starting at baseband sample ``at`` (C.5.2.1): a symbol-spaced
    channel fitted to the known symbols around them (``before`` more of them
    ahead, the synchronisation preamble's 184), then each of the 64 sets of
    three phases tried against what arrived."""
    pre = np.asarray(before if before is not None else [], dtype=np.int64)
    n0 = pre.size
    L1, L2 = _BARKER_TAPS
    total = n0 + 103
    idx = at - SPS * n0 + SPS * np.arange(-L1, total + L2)
    if idx[0] < 0 or idx[-1] >= bb.size:
        return None
    y = bb[idx] * np.exp(-2j * np.pi * freq_hz * (idx - at) / phy.FS)
    head = np.concatenate([pre, PROBE_PLUS, [2]])
    tail = np.concatenate([[6], PROBE_MINUS])

    def sym(ds):
        bark = np.concatenate([(d + BARKER13) % 8 for d in ds])
        return psk8(np.concatenate([head, bark, tail]))
    # The channel, from the known symbols: rows whose every tap sees one.
    known = np.ones(total, dtype=bool)
    known[n0 + 32: n0 + 71] = False
    s0 = sym((0, 0, 0))
    # y[k + L1] is symbol k's time; it hears symbols k - j, j from -L1 to L2.
    taps = np.arange(-L1, L2 + 1)
    rows, obs = [], []
    for k in range(L2, total - L1):
        js = k - taps
        if not known[js].all():
            continue
        rows.append(s0[js])
        obs.append(y[k + L1])
    if len(rows) < taps.size + 8:
        return None
    h, *_ = np.linalg.lstsq(np.array(rows), np.array(obs), rcond=None)
    # The symbol times the codes reach.
    ks = np.arange(max(L2, n0 + 32 - L1), min(total - L1, n0 + 71 + L2))
    best, bd = None, None
    for ds in _all_d():
        s_ = sym(ds)
        pred = s_[ks[:, None] - taps[None, :]] @ h
        r = float(np.sum(np.abs(y[ks + L1] - pred) ** 2))
        if best is None or r < best:
            best, bd = r, ds
    dib = [QPSK_MAP.index(d) for d in bd]
    rc = sum(((b >> 1) & 1) << (2 - i) for i, b in enumerate(dib))
    ic = sum((b & 1) << (2 - i) for i, b in enumerate(dib))
    return mode_of(rc, ic)


def read_mode(bb: np.ndarray, hit: phy.Hit) -> Mode4539 | None:
    """The rate and interleaver from a synchronisation preamble found at
    ``hit`` (its 184 fixed symbols)."""
    return read_barkers(bb, hit.sample + SPS * 184, hit.freq_hz, before=PREAMBLE184)


#: What ends a reinserted preamble, and what begins it 71 symbols earlier:
#: late entry looks for the one and checks for the other.
_LATE_TAIL = np.concatenate([[6], PROBE_MINUS])
_LATE_HEAD = np.concatenate([PROBE_PLUS, [2]])


def _refine_late(bb: np.ndarray, first: int, f: float) -> float:
    """The carrier offset from a reinserted preamble's two known ends, 71
    symbols apart: the phase between them, which the 32-symbol search leaves
    several hertz uncertain, and which is unambiguous to 17 Hz."""
    def corr(ref, at):
        n = first + SPS * (at + np.arange(len(ref)))
        y = bb[n] * np.exp(-2j * np.pi * f * (n - first) / phy.FS)
        return np.vdot(psk8(ref), y)
    for _ in range(2):
        c1, c2 = corr(_LATE_HEAD, 0), corr(_LATE_TAIL, 71)
        if abs(c1) < 1e-12 or abs(c2) < 1e-12:
            return f
        f += float(np.angle(c2 * np.conj(c1))) / (2 * np.pi * 71 / BAUD)
    return f


def find_late(bb: np.ndarray, *, start: int = 0, threshold: float = 0.35,
              stop: int | None = None, freq_hz: float = 0.0):
    """Acquisition on data (C.4): the first reinserted preamble at or after
    ``start`` -> (sample of its first symbol, the carrier offset, the mode),
    or None. Ordinary "-" mini-probes match its tail too; the "+" 71 symbols
    before, where an ordinary probe has data, tells them apart."""
    stop = bb.size if stop is None else stop
    at = start + 71 * SPS
    while at < stop:
        hit = phy.find(bb[:stop], _LATE_TAIL, chunk=16, threshold=threshold, start=at,
                       freq_hz=freq_hz)
        if hit is None:
            return None
        first = hit.sample - 71 * SPS
        if first >= 0:
            seg = bb[first - 2: first + 32 * SPS + 2]
            ch, en = phy.correlate(seg, _LATE_HEAD, 16, freq_hz=hit.freq_hz)
            score, _ = phy.differential_metric(ch, en, 16)
            if score.size and float(np.max(score[:5])) > threshold:
                f = _refine_late(bb, first, hit.freq_hz)
                mode = read_barkers(bb, first, f)
                if mode is not None:
                    return first, f, mode
        at = hit.sample + 16 * SPS
    return None


def _known_plan(v) -> phy.Plan:
    v = np.asarray(v)
    return phy.Plan(np.full(v.size, phy.KNOWN), v, np.zeros(v.size, dtype=np.int64), ())


def _constellation(mode: Mode4539):
    nb = mode.bits_per_symbol
    return tuple(QPSK_MAP) if nb == 2 else tuple(range(8)) if nb == 3 else TABLES[nb]


def preamble_plan(mode: Mode4539) -> phy.Plan:
    """The 287-symbol synchronisation preamble, all known."""
    p = _known_plan(np.concatenate([PREAMBLE184, known103(mode)]))
    p.constellation = _constellation(mode)
    return p


def frames_plan(mode: Mode4539, first: int, n: int) -> phy.Plan:
    """Frames ``first`` to ``first + n - 1`` (from 0): each its data and its
    probe, the reinserted preamble ahead of every 72nd after the first."""
    nb = mode.bits_per_symbol
    scr = _SCR[nb] if nb <= 3 else np.zeros(DATA_PER_FRAME, dtype=np.int64)
    rein = known103(mode)[-REINSERT:]
    parts = []
    for f in range(first, first + n):
        k = f % SUPERFRAME + 1
        if f and k == 1:
            parts.append(_known_plan(rein))
        parts.append(phy.Plan(np.full(DATA_PER_FRAME, phy.DATA),
                              np.zeros(DATA_PER_FRAME, dtype=np.int64), scr, ()))
        parts.append(_known_plan(probe(mode, k)))
    p = phy.Plan.concat(parts)
    p.constellation = _constellation(mode)
    return p


def plan(mode: Mode4539, n_frames: int) -> phy.Plan:
    """What the equalizer expects: the preamble, then ``n_frames`` frames."""
    return phy.Plan.concat([preamble_plan(mode), frames_plan(mode, 0, n_frames)]) \
        if n_frames else preamble_plan(mode)


def known_after(f: int) -> int:
    """Known symbols after frame ``f`` (from 0): its probe, and the reinserted
    preamble after every 72nd."""
    return PROBE_LEN + (REINSERT if (f + 1) % SUPERFRAME == 0 else 0)


def demap(z: np.ndarray, sigma2: np.ndarray, mode: Mode4539) -> np.ndarray:
    """Max-log LLRs (positive for 0) of the data bits, the earliest first,
    for whole frames of equalized data symbols. PSK arrives descrambled (the
    equalizer turns each symbol back); QAM is descrambled here, the points'
    labels exclusive-or'd with the frame's scrambling values."""
    nb = mode.bits_per_symbol
    if nb == 2:
        return phy.demap(z, sigma2, QPSK_MAP)
    if nb == 3:
        return phy.demap(z, sigma2, PSK8_MAP)
    table = TABLES[nb]
    n = z.size
    scr = np.tile(_SCR[nb], -(-n // DATA_PER_FRAME))[:n]
    d2 = np.abs(z[:, None] - table[None, :]) ** 2                 # (n, M)
    s2 = np.maximum(np.asarray(sigma2), 1e-4)[:, None]
    labels = np.arange(table.size)[None, :] ^ scr[:, None]        # data bits of each point
    out = np.empty((n, nb))
    big = 1e9
    for j in range(nb):
        one = ((labels >> (nb - 1 - j)) & 1).astype(bool)
        d1 = np.min(np.where(one, d2, big), axis=1)
        d0 = np.min(np.where(one, big, d2), axis=1)
        out[:, j] = (d1 - d0) / s2[:, 0]
    return out.ravel()


def find_eom(bits: np.ndarray) -> int:
    return spec.find_eom(bits, flush=0, errors=0)


SYNC = PREAMBLE184


#: The path model as this waveform needs it (see ``blockeq``): gains from
#: the probes, not from the decisions; the first path search confined to
#: chunks wholly known, since a ninth of the symbols are and only the preamble
#: is long enough to be a chunk; and the whole of the pulse, ringing and all,
#: to include the transmit trim ringing.


def selector(pre: int = 287) -> phy.Selector:
    """The block path equalizer beside the RLS one;
    ``pre`` known symbols ahead of the first frame (the synchronisation
    preamble's 287, or a reinserted preamble's 103 on late entry)."""
    from .blockeq import BlockEqualizer
    return phy.Selector(DATA_PER_FRAME, known_after, pre, path_cls=BlockEqualizer)


def late_plan(mode: Mode4539, n_frames: int) -> phy.Plan:
    """Late entry: the reinserted preamble's 103 known symbols, then frames
    numbered from the first after it."""
    p = _known_plan(known103(mode))
    p.constellation = _constellation(mode)
    return phy.Plan.concat([p, frames_plan(mode, 0, n_frames)]) if n_frames else p


#: A preamble found this far under the threshold is taken if the mini-probes
#: of the frames after it are where it says: a fade can take a preamble down
#: that far, and the probes are sixty-odd more known symbols.
MARGINAL = 0.8


def _probes_confirm(bb: np.ndarray, hit: phy.Hit, mode: Mode4539, frames: int = 6,
                    need: float = 0.5) -> bool:
    """Do the first ``frames`` mini-probes correlate where they should, each
    against its own sign and against the other's: the right sign stronger in
    at least ``need`` of those that stand out at all?"""
    right = seen = 0
    for f in range(frames):
        n0 = hit.sample + SPS * (287 + f * FRAME + DATA_PER_FRAME)
        n = n0 + SPS * np.arange(PROBE_LEN)
        if n[-1] >= bb.size:
            break
        y = bb[n] * np.exp(-2j * np.pi * hit.freq_hz * (n - hit.sample) / phy.FS)
        c = np.vdot(psk8(probe(mode, f + 1)), y)
        c_other = np.vdot(psk8((probe(mode, f + 1) + 4) % 8), y)
        e = float(np.sqrt(np.sum(np.abs(y) ** 2) * PROBE_LEN)) or 1.0
        if max(abs(c), abs(c_other)) / e > 0.5:
            seen += 1
            right += abs(c) > abs(c_other) and np.real(c * np.conj(hit.gain)) > 0
    return seen >= 2 and right >= need * seen


#: The carrier offsets acquisition searches from (C.6.3's 75 Hz): a 23-symbol
#: chunk 75 Hz out turns 260 degrees and scores a third; 25 Hz out, nine
#: tenths.
SHIFTS = (0.0, 50.0, -50.0)


def _better(a: phy.Hit, sa: float, b: phy.Hit, sb: float) -> bool:
    """``a`` (found searching from ``sa``) earlier than ``b`` (from ``sb``)
    by more than a symbol; or the same preamble, found from nearer its
    offset. A search from farther off may have locked to a later path of
    it, and a start on a later path leaves the path search a ghost to find."""
    if abs(a.sample - b.sample) > 8 * SPS:
        return a.sample < b.sample
    return abs(a.freq_hz - sa) < abs(b.freq_hz - sb)


def _find_sync(bb, start, threshold, shift, stop):
    at = start
    while True:
        hit = phy.find(bb if stop is None else bb[:stop], SYNC, chunk=23,
                       threshold=MARGINAL * threshold, start=at, freq_hz=shift)
        if hit is None:
            return None
        mode = read_mode(bb, hit)
        if mode is not None and (hit.score >= threshold or _probes_confirm(bb, hit, mode)):
            return hit, mode, 287
        at = hit.sample + len(SYNC) * SPS // 2


def _acquire(bb: np.ndarray, start: int, threshold: float):
    """The first synchronisation preamble or reinserted preamble at or after
    ``start``: (hit, mode, known symbols ahead of the first frame)."""
    sync, s_from = None, 0.0
    for shift in SHIFTS:
        got = _find_sync(bb, start, threshold, shift,
                         None if sync is None else sync[0].sample + 3 * len(SYNC) * SPS)
        if got is not None and (sync is None or _better(got[0], shift, sync[0], s_from)):
            sync, s_from = got, shift
    # A reinserted one earlier than that (or instead of none).
    late = None
    for shift in SHIFTS:
        stop = sync[0].sample + 287 * SPS if sync is not None else None
        if late is not None:
            stop = late[0] + 103 * SPS if stop is None else min(stop, late[0] + 103 * SPS)
        got = find_late(bb, start=start, threshold=threshold, stop=stop, freq_hz=shift)
        if got is not None and (late is None or got[0] < late[0] - 8 * SPS):
            late = got
    if late is not None and (sync is None or late[0] < sync[0].sample):
        first, f, mode = late
        n = first + SPS * np.arange(32)
        y = bb[n] * np.exp(-2j * np.pi * f * (n - first) / phy.FS)
        g = complex(np.vdot(psk8(_LATE_HEAD), y) / 32)
        return phy.Hit(first, threshold, f, g), mode, 103
    return sync


def receive(audio: np.ndarray, *, start: int = 0, threshold: float = 0.35,
            order: str = "lsb") -> Reception | None:
    """Find the first transmission in ``audio`` (real, at ``spec.FS``) at or
    after sample ``start``, read its mode from the preamble, and decode it.
    One whose preamble was lost is joined at its next reinserted preamble,
    and decoded from there."""
    from .stream import Receiver4539, receive_whole
    got = receive_whole(audio, Receiver4539(bit_order=order, threshold=threshold), start=start)
    if got is None:
        return None
    first, data, end = got
    mode = MODES4539[first.mode]
    frames = max(0, ((end.at - first.at) // SPS - 287) // FRAME)
    return Reception(mode, data, frames, first.at, end.at, first.freq_hz, end.snr_db, end.complete)
