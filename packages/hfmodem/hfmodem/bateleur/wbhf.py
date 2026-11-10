# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""MIL-STD-188-110C appendix D in 3 kHz: waveforms 0 to 13, 75 to 16000 bit/s.

Written from MIL-STD-188-110C with Change 1, appendix D, and no other source.
Only the 3 kHz bandwidth is here, and only the constraint length 7 code. 2400
symbols a second on 1800 Hz (table D-I), as 188-110's serial tone.

**Frames.** After the preamble a mini-probe, then frames of U data symbols and
a K-symbol mini-probe (tables D-XI, D-XII). The mini-probe is a base sequence
cyclically extended (table D-XXI); the one after the second-to-last frame of
each long interleaver block is the same sequence started further in, which
marks where the blocks lie (D.5.2.2). Waveform 0 has no frames: each pair of
coded bits is a 32-chip Walsh symbol, sent back to back (D.5.1.2.3).

**Data symbols.** BPSK, QPSK and 8-PSK numbers (tables D-III to D-V) are
scrambled by adding, modulo 8, the last three stages of an x^9 + x^4 + 1
register; QAM numbers by exclusive-or with its last 4, 5, 6 or 8 stages. The
register is set to 1 at every frame (D.5.1.3). Waveform 0's chips have the
values of a (159, 31) trinomial register added, 2048 of them repeating, from
its initial state at each interleaver block (D.5.1.4).

**Coding.** The K=7 code of 188-110 made a block code by full tail-biting,
repeated and then punctured to the rate of table D-XLI with the patterns of
table D-XLII, the pattern aligned to the block's start (D.5.3.2). The block
interleaver loads bit n at n x increment modulo its size and is read in order
(D.5.3.3).

**Preamble.** N transmit level control blocks, the first 32 symbols of table
D-XVIII conjugated; then M superframes of 18 Walsh-modulated symbols: nine
fixed, four counting the superframes down and five naming the waveform,
interleaver and constraint length (D.5.2.1). Each symbol is a 4-chip Walsh
sequence repeated to 32 and added to the first 32 values of its section's
table (D-XVIII, D-XIX, D-XX); the appendix gives one table per section, 256
long for the widest bandwidth, and does not say whether a 3 kHz symbol takes
the next 32 of it, so each takes its start, as the TLC blocks do.

The appendix asks for d2, the waveform identifier's checksum bit, and also
says w1's least significant bit (d2) shall be 0. The checksum is sent here;
either is accepted.

Messages end with the optional end of message, 4B65A5B2, and are filled out
with zeros to whole input blocks (D.5.4.3). The optional end of transmission
marker is not sent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fec, phy
from .mil110 import bits_to_bytes as bits_to_bytes, bytes_to_bits
from .s4539 import EOM_BITS, PSK8_MAP, QPSK_MAP, TABLES as _QAM, find_eom as find_eom, scramble_values
from .spec import KNOWN_POINTS, SPS, psk8

# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

#: Interleaver lengths, in the order tables D-XXXIII and D-XLIII list them,
#: and their codes in the waveform identifier (table D-XVI).
INTERLEAVERS = ("US", "S", "M", "L")


@dataclass(frozen=True)
class Waveform:
    """One column of the 3 kHz tables."""
    number: int
    rate: int                            # bit/s, table D-II
    bits: int                            # bits a data symbol; 0 for Walsh
    unknown: int                         # data symbols a frame, table D-XI
    known: int                           # mini-probe, table D-XII
    repeat: int                          # table D-XLII's repetitions
    t1: str                              # its puncture pattern, "" for none
    t2: str
    frames: tuple                        # frames an interleaver, US S M L (D-XXXIII)
    increment: tuple                     # table D-XLIII


_F1, _F3, _F5 = (4, 16, 64, 256), (2, 8, 32, 128), (1, 4, 16, 64)
WAVEFORMS = {w.number: w for w in (
    Waveform(0, 75, 0, 32, 0, 1, "", "", (None, 40, 144, 576), (None, 11, 37, 145)),
    Waveform(1, 150, 1, 48, 48, 4, "", "", _F1, (25, 97, 385, 1543)),
    Waveform(2, 300, 1, 48, 48, 2, "", "", _F1, (25, 97, 385, 1543)),
    Waveform(3, 600, 1, 96, 32, 2, "11", "10", _F3, (25, 97, 385, 1549)),
    Waveform(4, 1200, 1, 96, 32, 1, "11", "10", _F3, (25, 97, 385, 1549)),
    Waveform(5, 1600, 1, 256, 32, 1, "110", "101", _F5, (33, 129, 513, 2081)),
    Waveform(6, 3200, 2, 256, 32, 1, "110", "101", _F5, (65, 257, 1025, 4161)),
    Waveform(7, 4800, 3, 256, 32, 1, "110", "101", _F5, (97, 385, 1537, 6241)),
    Waveform(8, 6400, 4, 256, 32, 1, "110", "101", _F5, (129, 513, 2049, 8321)),
    Waveform(9, 8000, 5, 256, 32, 1, "110", "101", _F5, (161, 641, 2561, 10403)),
    Waveform(10, 9600, 6, 256, 32, 1, "110", "101", _F5, (193, 769, 3073, 12481)),
    Waveform(11, 12000, 6, 360, 24, 1, "11110100", "10001011", _F5, (271, 1081, 4321, 17551)),
    Waveform(12, 16000, 8, 360, 24, 1, "11110100", "10001011", _F5, (361, 1441, 5761, 23401)),
    Waveform(13, 2400, 2, 256, 32, 1, "111101111", "111111011", _F5, (65, 257, 1025, 4161)),
)}

#: Table D-X, the 256-QAM constellation.
QAM256 = np.array([complex(i, q) for i, q in (
    (0.959366, 0.056433), (0.959366, 0.1693), (0.846499, 0.507899), (0.959366, 0.282166),
    (0.846499, 0.056433), (0.846499, 0.1693), (0.846499, 0.395033), (0.846499, 0.282166),
    (0.959366, -0.056433), (0.959366, -0.1693), (0.846499, -0.507899), (0.959366, -0.282166),
    (0.846499, -0.056433), (0.846499, -0.1693), (0.846499, -0.395033), (0.846499, -0.282166),
    (0.1693, 0.959366), (0.056433, 0.998304), (0.733632, 0.507899), (0.733632, 0.620766),
    (0.733632, 0.056433), (0.733632, 0.1693), (0.733632, 0.395033), (0.733632, 0.282166),
    (0.1693, -0.959366), (0.056433, -0.998304), (0.733632, -0.507899), (0.733632, -0.620766),
    (0.733632, -0.056433), (0.733632, -0.1693), (0.733632, -0.395033), (0.733632, -0.282166),
    (0.507899, 0.846499), (0.507899, 0.733632), (0.507899, 0.507899), (0.507899, 0.620766),
    (0.507899, 0.056433), (0.507899, 0.1693), (0.507899, 0.395033), (0.507899, 0.282166),
    (0.507899, -0.846499), (0.507899, -0.733632), (0.507899, -0.507899), (0.507899, -0.620766),
    (0.507899, -0.056433), (0.507899, -0.1693), (0.507899, -0.395033), (0.507899, -0.282166),
    (0.282166, 0.959366), (0.620766, 0.733632), (0.620766, 0.507899), (0.620766, 0.620766),
    (0.620766, 0.056433), (0.620766, 0.1693), (0.620766, 0.395033), (0.620766, 0.282166),
    (0.282166, -0.959366), (0.620766, -0.733632), (0.620766, -0.507899), (0.620766, -0.620766),
    (0.620766, -0.056433), (0.620766, -0.1693), (0.620766, -0.395033), (0.620766, -0.282166),
    (0.056433, 0.846499), (0.056433, 0.733632), (0.056433, 0.507899), (0.056433, 0.620766),
    (0.056433, 0.056433), (0.056433, 0.1693), (0.056433, 0.395033), (0.056433, 0.282166),
    (0.056433, -0.846499), (0.056433, -0.733632), (0.056433, -0.507899), (0.056433, -0.620766),
    (0.056433, -0.056433), (0.056433, -0.1693), (0.056433, -0.395033), (0.056433, -0.282166),
    (0.1693, 0.846499), (0.1693, 0.733632), (0.1693, 0.507899), (0.1693, 0.620766),
    (0.1693, 0.056433), (0.1693, 0.1693), (0.1693, 0.395033), (0.1693, 0.282166),
    (0.1693, -0.846499), (0.1693, -0.733632), (0.1693, -0.507899), (0.1693, -0.620766),
    (0.1693, -0.056433), (0.1693, -0.1693), (0.1693, -0.395033), (0.1693, -0.282166),
    (0.395033, 0.846499), (0.395033, 0.733632), (0.395033, 0.507899), (0.395033, 0.620766),
    (0.395033, 0.056433), (0.395033, 0.1693), (0.395033, 0.395033), (0.395033, 0.282166),
    (0.395033, -0.846499), (0.395033, -0.733632), (0.395033, -0.507899), (0.395033, -0.620766),
    (0.395033, -0.056433), (0.395033, -0.1693), (0.395033, -0.395033), (0.395033, -0.282166),
    (0.282166, 0.846499), (0.282166, 0.733632), (0.282166, 0.507899), (0.282166, 0.620766),
    (0.282166, 0.056433), (0.282166, 0.1693), (0.282166, 0.395033), (0.282166, 0.282166),
    (0.282166, -0.846499), (0.282166, -0.733632), (0.282166, -0.507899), (0.282166, -0.620766),
    (0.282166, -0.056433), (0.282166, -0.1693), (0.282166, -0.395033), (0.282166, -0.282166),
    (-0.959366, 0.056433), (-0.959366, 0.1693), (-0.846499, 0.507899), (-0.959366, 0.282166),
    (-0.846499, 0.056433), (-0.846499, 0.1693), (-0.846499, 0.395033), (-0.846499, 0.282166),
    (-0.959366, -0.056433), (-0.959366, -0.1693), (-0.846499, -0.507899), (-0.959366, -0.282166),
    (-0.846499, -0.056433), (-0.846499, -0.1693), (-0.846499, -0.395033), (-0.846499, -0.282166),
    (-0.1693, 0.959366), (-0.056433, 0.998304), (-0.733632, 0.507899), (-0.733632, 0.620766),
    (-0.733632, 0.056433), (-0.733632, 0.1693), (-0.733632, 0.395033), (-0.733632, 0.282166),
    (-0.1693, -0.959366), (-0.056433, -0.998304), (-0.733632, -0.507899), (-0.733632, -0.620766),
    (-0.733632, -0.056433), (-0.733632, -0.1693), (-0.733632, -0.395033), (-0.733632, -0.282166),
    (-0.507899, 0.846499), (-0.507899, 0.733632), (-0.507899, 0.507899), (-0.507899, 0.620766),
    (-0.507899, 0.056433), (-0.507899, 0.1693), (-0.507899, 0.395033), (-0.507899, 0.282166),
    (-0.507899, -0.846499), (-0.507899, -0.733632), (-0.507899, -0.507899), (-0.507899, -0.620766),
    (-0.507899, -0.056433), (-0.507899, -0.1693), (-0.507899, -0.395033), (-0.507899, -0.282166),
    (-0.282166, 0.959366), (-0.620766, 0.733632), (-0.620766, 0.507899), (-0.620766, 0.620766),
    (-0.620766, 0.056433), (-0.620766, 0.1693), (-0.620766, 0.395033), (-0.620766, 0.282166),
    (-0.282166, -0.959366), (-0.620766, -0.733632), (-0.620766, -0.507899), (-0.620766, -0.620766),
    (-0.620766, -0.056433), (-0.620766, -0.1693), (-0.620766, -0.395033), (-0.620766, -0.282166),
    (-0.056433, 0.846499), (-0.056433, 0.733632), (-0.056433, 0.507899), (-0.056433, 0.620766),
    (-0.056433, 0.056433), (-0.056433, 0.1693), (-0.056433, 0.395033), (-0.056433, 0.282166),
    (-0.056433, -0.846499), (-0.056433, -0.733632), (-0.056433, -0.507899), (-0.056433, -0.620766),
    (-0.056433, -0.056433), (-0.056433, -0.1693), (-0.056433, -0.395033), (-0.056433, -0.282166),
    (-0.1693, 0.846499), (-0.1693, 0.733632), (-0.1693, 0.507899), (-0.1693, 0.620766),
    (-0.1693, 0.056433), (-0.1693, 0.1693), (-0.1693, 0.395033), (-0.1693, 0.282166),
    (-0.1693, -0.846499), (-0.1693, -0.733632), (-0.1693, -0.507899), (-0.1693, -0.620766),
    (-0.1693, -0.056433), (-0.1693, -0.1693), (-0.1693, -0.395033), (-0.1693, -0.282166),
    (-0.395033, 0.846499), (-0.395033, 0.733632), (-0.395033, 0.507899), (-0.395033, 0.620766),
    (-0.395033, 0.056433), (-0.395033, 0.1693), (-0.395033, 0.395033), (-0.395033, 0.282166),
    (-0.395033, -0.846499), (-0.395033, -0.733632), (-0.395033, -0.507899), (-0.395033, -0.620766),
    (-0.395033, -0.056433), (-0.395033, -0.1693), (-0.395033, -0.395033), (-0.395033, -0.282166),
    (-0.282166, 0.846499), (-0.282166, 0.733632), (-0.282166, 0.507899), (-0.282166, 0.620766),
    (-0.282166, 0.056433), (-0.282166, 0.1693), (-0.282166, 0.395033), (-0.282166, 0.282166),
    (-0.282166, -0.846499), (-0.282166, -0.733632), (-0.282166, -0.507899), (-0.282166, -0.620766),
    (-0.282166, -0.056433), (-0.282166, -0.1693), (-0.282166, -0.395033), (-0.282166, -0.282166)
)])
QAM = {**_QAM, 8: QAM256}

#: Tables D-XVIII, D-XIX, D-XX: the preamble sections' scrambling values.
FIXED_PN = np.array([int(c) for c in (
    "2400621461057341261707322232463663754756740261530424645254531545"
    "6510710105352245406414033003373427444034764262035322452003503266"
    "1423613033242265536765665254233357553704704162355626463407005215"
    "4345705376664560204234407662003305242245466632103260624064133536")])
COUNT_PN = np.array([int(c) for c in (
    "5522025671351565370403321303162606412563537426730201750615032252"
    "5234276115215403550314050306003161447705701510131507122271250332"
    "2045131353175271315624606100362732476413660300754512150310466105"
    "2632742401707051457204435277451446330515543203047451557762435224")])
WID_PN = np.array([int(c) for c in (
    "2303733014465545620566535522123611431051033030446256172620047235"
    "2716504162154350341321615704766047666623507031512053245667735160"
    "1445606724403720014071745455533205131534154144224307415714725566"
    "1656302577443446072200210036640243452637757307007262261437650654")])
#: D.5.1.4: the trinomial register's initial state, stage 0 first.
BITSHIFT = [int(c) for c in (
    "0001001101100101111101001000000010110011001011101110001100010000"
    "1000001000001101011110101000111111001101011111011110001100011010"
    "1110011100011000100100011010011")]

#: Table D-XIV (and D.5.1.2.3): dibit -> four Walsh chips; the alternates
#: that mark an interleaver block's last dibit in waveform 0.
WALSH = np.array([[0, 0, 0, 0], [0, 4, 0, 4], [0, 0, 4, 4], [0, 4, 4, 0]])
WALSH_ALT = np.array([[0, 0, 0, 0, 4, 4, 4, 4], [0, 4, 0, 4, 4, 0, 4, 0],
                      [0, 0, 4, 4, 4, 4, 0, 0], [0, 4, 4, 0, 4, 0, 0, 4]])
#: The 32 chips of each, before scrambling: rows 0-3 by dibit, 4-7 the
#: alternates (the rows a SET plan decides among).
PATTERNS = np.vstack([np.tile(WALSH, 8), np.tile(WALSH_ALT, 4)])
CHIPS = 32                               # table D-XIII, 3 kHz

FIXED_DIBITS = (0, 0, 2, 1, 2, 1, 0, 2, 3)  # D.5.2.1.3
SUPERFRAME = CHIPS * (len(FIXED_DIBITS) + 4 + 5)

#: Tables D-XXII(a), D-XXIII, D-XXV: the mini-probes' base sequences, as
#: indices into ``spec.KNOWN_POINTS`` (8-PSK numbers, and 8 + k for the
#: fifth root of unity exp(2 pi j k / 5)); and table D-XXI's shift.
BASES = {
    24: (np.array([0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 0, 4, 0]), 6),
    32: (np.array([0, 0, 0, 0, 0, 6, 4, 2, 0, 4, 0, 4, 0, 2, 4, 6]), 8),
    48: (np.array([0, 0, 0, 0, 0, 0, 12, 11, 10, 9, 0, 11, 9, 12, 10, 0, 10, 12, 9, 11,
                   0, 9, 10, 11, 12]), 12),
}

EOM = EOM_BITS


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeWbhf:
    waveform: int
    interleaver: str                     # "US", "S", "M" or "L"

    @property
    def w(self) -> Waveform:
        return WAVEFORMS[self.waveform]

    @property
    def name(self) -> str:
        return f"{self.w.rate}{self.interleaver}"

    @property
    def walsh(self) -> bool:
        return self.waveform == 0

    @property
    def frames(self) -> int:
        """Frames (Walsh symbols, for waveform 0) an interleaver block."""
        return self.w.frames[INTERLEAVERS.index(self.interleaver)]

    @property
    def long_frames(self) -> int:
        return self.w.frames[3]

    @property
    def interleaver_bits(self) -> int:
        return self.frames * (2 if self.walsh else self.w.unknown * self.w.bits)

    @property
    def increment(self) -> int:
        return self.w.increment[INTERLEAVERS.index(self.interleaver)]

    @property
    def block_bits(self) -> int:
        """Input data bits a block: the interleaver less the code's rate."""
        w = self.w
        if not w.t1:
            return self.interleaver_bits // (2 * w.repeat)
        p = len(w.t1)
        kept = (w.t1 + w.t2).count("1")
        return self.interleaver_bits * p // (w.repeat * kept)

    @property
    def frame_symbols(self) -> int:
        return CHIPS if self.walsh else self.w.unknown + self.w.known


def _all_modes() -> dict[str, ModeWbhf]:
    out = {}
    for n, w in WAVEFORMS.items():
        for i, il in enumerate(INTERLEAVERS):
            if w.frames[i] is not None:
                m = ModeWbhf(n, il)
                out[m.name] = m
    return out


MODES_WBHF = _all_modes()


# --------------------------------------------------------------------------
# Coding (D.5.3)
# --------------------------------------------------------------------------

def _mask(w: Waveform, n: int) -> np.ndarray | None:
    """Which of ``n`` repeated (T1, T2) pairs' bits are sent, (n, 2)."""
    if not w.t1:
        return None
    p = len(w.t1)
    m = np.array([[a == "1", b == "1"] for a, b in zip(w.t1, w.t2)])
    return np.tile(m, (n // p, 1))


def encode_block(bits: np.ndarray, mode: ModeWbhf) -> np.ndarray:
    """One input block -> its interleaved bits, in the order they are sent."""
    w = mode.w
    pairs = np.repeat(fec.encode_tailbiting(bits).reshape(-1, 2), w.repeat, axis=0)
    m = _mask(w, len(pairs))
    punct = pairs[m] if m is not None else pairs.ravel()
    size = mode.interleaver_bits
    inter = np.empty(size, dtype=np.uint8)
    inter[(np.arange(size) * mode.increment) % size] = punct
    return inter


def decode_block(llr: np.ndarray, mode: ModeWbhf) -> np.ndarray:
    """An interleaver block's LLRs (positive for 0), in the order received ->
    the input bits."""
    w = mode.w
    size = mode.interleaver_bits
    punct = np.asarray(llr)[(np.arange(size) * mode.increment) % size]
    n = mode.block_bits * w.repeat
    m = _mask(w, n)
    full = np.zeros((n, 2))
    if m is not None:
        full[m] = punct
    else:
        full[:] = punct.reshape(-1, 2)
    return fec.viterbi_tailbiting(full.reshape(-1, w.repeat, 2).sum(axis=1).ravel())


# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------

_PSK_MAPS = {1: (0, 4), 2: QPSK_MAP, 3: PSK8_MAP}
_SCR = {}


def _scr(mode: ModeWbhf) -> np.ndarray:
    """A frame's scrambling values (D.5.1.3)."""
    key = (mode.w.bits, mode.w.unknown)
    if key not in _SCR:
        _SCR[key] = scramble_values(mode.w.bits, mode.w.unknown)
    return _SCR[key]


def tri_values(n: int = 2048) -> np.ndarray:
    """D.5.1.4's ``tri()``: the register stepped 16 times, then its stages 2,
    1, 0 read as a number, ``n`` times."""
    r = list(BITSHIFT)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        for _ in range(16):
            r = [r[158] ^ r[31]] + r[:158]
        out[i] = (r[2] << 2) | (r[1] << 1) | r[0]
    return out


TRI = tri_values()


def _walsh_scrambling(n: int) -> np.ndarray:
    """The chips' scrambling values for the first ``n`` Walsh symbols of an
    interleaver block, (n, 32)."""
    return TRI[np.arange(n * CHIPS) % TRI.size].reshape(n, CHIPS)


def symbol_numbers(bits: np.ndarray, mode: ModeWbhf) -> np.ndarray:
    """Interleaved bits -> data symbol numbers before scrambling (the earliest
    bit the most significant): PSK through tables D-III to D-V, QAM the bits
    themselves, Walsh the dibits."""
    nb = 2 if mode.walsh else mode.w.bits
    v = np.asarray(bits, dtype=np.int64).reshape(-1, nb) @ (1 << np.arange(nb - 1, -1, -1))
    if not mode.walsh and nb <= 3:
        return np.asarray(_PSK_MAPS[nb])[v]
    return v


def data_frame(numbers: np.ndarray, mode: ModeWbhf) -> np.ndarray:
    """One frame's symbol numbers -> its data symbols, scrambled, as points."""
    nb = mode.w.bits
    scr = _scr(mode)
    if nb <= 3:
        return psk8((numbers + scr) % 8)
    return QAM[nb][numbers ^ scr]


def walsh_block(dibits: np.ndarray) -> np.ndarray:
    """One interleaver block of waveform 0 -> its chips as 8-PSK numbers: a
    pattern a dibit, the last one's alternate, scrambled from the block's
    start."""
    rows = np.asarray(dibits, dtype=np.int64).copy()
    rows[-1] += 4
    return ((PATTERNS[rows] + _walsh_scrambling(rows.size)) % 8).ravel()


def probe(mode: ModeWbhf, shifted: bool = False) -> np.ndarray:
    """A mini-probe as ``KNOWN_POINTS`` indices: the base sequence extended,
    from its start or (marking a long interleaver block) from table D-XXI's
    shift."""
    base, shift = BASES[mode.w.known]
    return base[(np.arange(mode.w.known) + (shift if shifted else 0)) % base.size]


def shifted_after(mode: ModeWbhf, f: int) -> bool:
    """Is the mini-probe after frame ``f`` (from 0) the shifted one?"""
    return f % mode.long_frames == mode.long_frames - 2


def count_dibits(count: int) -> tuple[int, ...]:
    """D.5.2.1.3.1: the 5-bit count and its 3 parity bits as c3, c2, c1, c0."""
    b = [(count >> i) & 1 for i in range(5)]
    b5 = b[0] ^ b[1] ^ b[2]
    b6 = b[2] ^ b[3] ^ b[4]
    b7 = b[1] ^ b[2] ^ b[3]
    return ((b7 << 1) | b6, (b5 << 1) | b[4], (b[3] << 1) | b[2], (b[1] << 1) | b[0])


def wid_dibits(mode: ModeWbhf, checksum: bool = True) -> tuple[int, ...]:
    """D.5.2.1.3.2: w4 ... w0. ``checksum`` False sends d2 as 0, as the
    sentence after table D-XVI says w1's least significant bit shall be."""
    d = [0] * 10                         # d[i] is d_i
    for i in range(4):
        d[6 + i] = (mode.waveform >> i) & 1
    il = INTERLEAVERS.index(mode.interleaver)
    d[5], d[4] = (il >> 1) & 1, il & 1
    d[3] = 0                             # constraint length 7 (table D-XVII)
    d[2] = (d[9] ^ d[8] ^ d[7]) if checksum else 0
    d[1] = d[7] ^ d[6] ^ d[5]
    d[0] = d[5] ^ d[4] ^ d[3]
    return tuple((d[2 * k + 1] << 1) | d[2 * k] for k in range(4, -1, -1))


def walsh_symbol(dibit: int, pn: np.ndarray) -> np.ndarray:
    """A preamble channel symbol: the dibit's Walsh chips, repeated, added to
    the section's table (D.5.2.1.1)."""
    return (np.tile(WALSH[dibit], CHIPS // 4) + pn[:CHIPS]) % 8


FIXED = np.concatenate([walsh_symbol(d, FIXED_PN) for d in FIXED_DIBITS])


def superframe(mode: ModeWbhf, count: int) -> np.ndarray:
    """One preamble superframe: fixed, count, waveform identifier."""
    return np.concatenate([FIXED]
                          + [walsh_symbol(d, COUNT_PN) for d in count_dibits(count)]
                          + [walsh_symbol(d, WID_PN) for d in wid_dibits(mode)])


def preamble(mode: ModeWbhf, superframes: int = 1, tlc: int = 0) -> np.ndarray:
    """D.5.2.1: ``tlc`` (0-255) level control blocks, then ``superframes``
    (1-32) superframes counting down to 0."""
    head = np.tile((-FIXED_PN[:CHIPS]) % 8, tlc)
    return np.concatenate([head] + [superframe(mode, c)
                                    for c in range(superframes - 1, -1, -1)])


# --------------------------------------------------------------------------
# Transmit
# --------------------------------------------------------------------------

def _message_bits(data: bytes, mode: ModeWbhf, eom: bool, order: str) -> np.ndarray:
    bits = bytes_to_bits(data, order)
    if eom:
        bits = np.concatenate([bits, EOM])
    n = max(bits.size, 1)
    n += (-n) % mode.block_bits
    return np.concatenate([bits, np.zeros(n - bits.size, dtype=np.uint8)])


def symbols(data: bytes, mode: ModeWbhf | str = "9600L", *, eom: bool = True,
            order: str = "lsb", superframes: int = 1, tlc: int = 0) -> np.ndarray:
    """A message as the transmission's complex symbols, preamble first."""
    if isinstance(mode, str):
        mode = MODES_WBHF[mode]
    bits = _message_bits(data, mode, eom, order)
    blocks = [encode_block(b, mode) for b in bits.reshape(-1, mode.block_bits)]
    out = [psk8(preamble(mode, superframes, tlc))]
    if mode.walsh:
        out += [psk8(walsh_block(symbol_numbers(b, mode))) for b in blocks]
        return np.concatenate(out)
    pts = KNOWN_POINTS
    nums = symbol_numbers(np.concatenate(blocks), mode).reshape(-1, mode.w.unknown)
    out.append(pts[probe(mode)])
    for f, frame in enumerate(nums):
        out.append(data_frame(frame, mode))
        out.append(pts[probe(mode, shifted_after(mode, f))])
    return np.concatenate(out)


def symbol_count(mode: ModeWbhf, n_bytes: int, *, eom: bool = True,
                 superframes: int = 1, tlc: int = 0) -> int:
    """How many symbols ``symbols`` makes of an ``n_bytes`` message."""
    n = max(8 * n_bytes + (32 if eom else 0), 1)
    frames = -(-n // mode.block_bits) * mode.frames
    head = CHIPS * tlc + SUPERFRAME * superframes
    if mode.walsh:
        return head + frames * CHIPS
    return head + mode.w.known + frames * mode.frame_symbols


def transmit(data: bytes, mode: ModeWbhf | str = "9600L", *, eom: bool = True,
             order: str = "lsb", superframes: int = 1, tlc: int = 0,
             trim: bool = True) -> np.ndarray:
    """Message -> real audio at ``spec.FS``, peak 1.0. ``trim`` as
    `phy.modulate`."""
    return phy.modulate_symbols(symbols(data, mode, eom=eom, order=order,
                                        superframes=superframes, tlc=tlc), trim)


# --------------------------------------------------------------------------
# Receive
# --------------------------------------------------------------------------

@dataclass
class Reception:
    mode: ModeWbhf
    data: bytes
    frames: int
    start_sample: int
    end_sample: int
    freq_hz: float
    snr_db: float
    eom: bool = False


#: The channel ``read_header`` fits: symbol-spaced taps this many symbols
#: before and after the main one.
_TAPS = (2, 13)


def _valid_counts() -> list[tuple[int, tuple]]:
    return [(c, count_dibits(c)) for c in range(32)]


def _valid_wids() -> list[tuple[ModeWbhf, tuple]]:
    out = []
    for m in MODES_WBHF.values():
        for d in {wid_dibits(m, True), wid_dibits(m, False)}:
            out.append((m, d))
    return out


_COUNTS = _valid_counts()
_WIDS = _valid_wids()


def read_header(bb: np.ndarray, at: int, freq_hz: float):
    """The count and waveform identifier of the superframe whose fixed
    section starts at baseband sample ``at``: a symbol-spaced channel fitted to
    the fixed section, then each Walsh symbol's four values scored against
    what arrived, in turn, the chosen ones taken as known for the next; the
    count and identifier those scores favour most among the valid ones.
    Returns (count, mode) or None."""
    L1, L2 = _TAPS
    idx = at + SPS * np.arange(-L1, SUPERFRAME + L1)
    if idx[0] < 0 or idx[-1] >= bb.size:
        return None
    y = bb[idx] * np.exp(-2j * np.pi * freq_hz * (idx - at) / phy.FS)
    nf = FIXED.size
    taps = np.arange(-L1, L2 + 1)
    s = np.zeros(SUPERFRAME, dtype=np.complex128)
    s[:nf] = psk8(FIXED)
    ks = np.arange(L2, nf - L1)
    A = s[ks[:, None] - taps[None, :]]
    h, *_ = np.linalg.lstsq(A, y[ks + L1], rcond=None)
    resid = y[ks + L1] - A @ h
    s2 = max(float(np.mean(np.abs(resid) ** 2)), 1e-6)

    def section(first: int, pn: np.ndarray, valid):
        score = np.zeros((len(valid[0][1]), 4))
        for i in range(score.shape[0]):
            a = nf + CHIPS * (first + i)
            kk = np.arange(a, a + CHIPS - L1)
            for d in range(4):
                s[a:a + CHIPS] = psk8(walsh_symbol(d, pn))
                pred = s[kk[:, None] - taps[None, :]] @ h
                score[i, d] = -np.sum(np.abs(y[kk + L1] - pred) ** 2) / s2
            s[a:a + CHIPS] = psk8(walsh_symbol(int(np.argmax(score[i])), pn))
        best = max(valid, key=lambda v: sum(score[i, d] for i, d in enumerate(v[1])))
        for i, d in enumerate(best[1]):
            a = nf + CHIPS * (first + i)
            s[a:a + CHIPS] = psk8(walsh_symbol(d, pn))
        return best[0]

    count = section(0, COUNT_PN, _COUNTS)
    mode = section(4, WID_PN, _WIDS)
    return count, mode


def _known_plan(v) -> phy.Plan:
    v = np.asarray(v)
    return phy.Plan(np.full(v.size, phy.KNOWN), v, np.zeros(v.size, dtype=np.int64), ())


def _constellation(mode: ModeWbhf):
    nb = mode.w.bits
    if mode.walsh:
        return (0, 4)
    return _PSK_MAPS[nb] if nb <= 3 else QAM[nb]


def frames_plan(mode: ModeWbhf, first: int, n: int) -> phy.Plan:
    """Frames ``first`` to ``first + n - 1`` (from 0; Walsh symbols, for
    waveform 0), each with the mini-probe after it."""
    if mode.walsh:
        F = mode.frames
        i = (first + np.arange(n)) % F
        kind = np.full(n * CHIPS, phy.SET)
        value = np.repeat(np.where(i == F - 1, 4, 0), CHIPS)
        rot = TRI[(i[:, None] * CHIPS + np.arange(CHIPS)[None, :]) % TRI.size].ravel()
        return phy.Plan(kind, value, rot, (0, 4), PATTERNS)
    nb = mode.w.bits
    U = mode.w.unknown
    scr = _scr(mode) if nb <= 3 else np.zeros(U, dtype=np.int64)
    parts = []
    for f in range(first, first + n):
        parts.append(phy.Plan(np.full(U, phy.DATA), np.zeros(U, dtype=np.int64), scr, ()))
        parts.append(_known_plan(probe(mode, shifted_after(mode, f))))
    p = phy.Plan.concat(parts)
    p.constellation = _constellation(mode)
    return p


def head_plan(mode: ModeWbhf, count: int) -> phy.Plan:
    """From a superframe counting ``count`` to the first frame: the rest of
    the preamble, and the mini-probe that ends it."""
    v = [superframe(mode, c) for c in range(count, -1, -1)]
    p = _known_plan(np.concatenate(v))
    if not mode.walsh:
        p = phy.Plan.concat([p, _known_plan(probe(mode))])
    p.constellation = _constellation(mode)
    if mode.walsh:
        p.patterns = PATTERNS
    return p


def plan(mode: ModeWbhf, count: int, n_frames: int) -> phy.Plan:
    """What the equalizer expects from a superframe counting ``count``: it
    and the rest of the preamble, then ``n_frames`` frames."""
    head = head_plan(mode, count)
    if not n_frames:
        return head
    p = phy.Plan.concat([head, frames_plan(mode, 0, n_frames)])
    p.constellation = head.constellation
    return p


def known_ahead(mode: ModeWbhf, count: int) -> int:
    """Known symbols from the found superframe to the first frame's data."""
    return SUPERFRAME * (count + 1) + (0 if mode.walsh else mode.w.known)


def demap(z: np.ndarray, sigma2: np.ndarray, mode: ModeWbhf) -> np.ndarray:
    """Max-log LLRs (positive for 0) of the data bits, the earliest first,
    for whole frames of equalized data symbols. PSK arrives descrambled; QAM
    is descrambled here, each point's label exclusive-or'd with its value."""
    nb = mode.w.bits
    if nb <= 3:
        return phy.demap(z, sigma2, _PSK_MAPS[nb])
    table = QAM[nb]
    n = z.size
    scr = np.tile(_scr(mode), -(-n // mode.w.unknown))[:n]
    d2 = np.abs(z[:, None] - table[None, :]) ** 2
    s2 = np.maximum(np.asarray(sigma2), 1e-4)[:, None]
    labels = np.arange(table.size)[None, :] ^ scr[:, None]
    out = np.empty((n, nb))
    big = 1e9
    for j in range(nb):
        one = ((labels >> (nb - 1 - j)) & 1).astype(bool)
        out[:, j] = (np.min(np.where(one, d2, big), axis=1)
                     - np.min(np.where(one, big, d2), axis=1)) / s2[:, 0]
    return out.ravel()


def walsh_llr(corr: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Waveform 0: each Walsh symbol's four correlations, by dibit -> LLRs
    of its two bits, the first first, positive for 0."""
    c = np.asarray(corr, dtype=np.float64).reshape(-1, 4)
    s2 = np.maximum(np.asarray(sigma2, dtype=np.float64).reshape(-1), 1e-3)
    first = np.max(c[:, [0, 1]], 1) - np.max(c[:, [2, 3]], 1)
    second = np.max(c[:, [0, 2]], 1) - np.max(c[:, [1, 3]], 1)
    return (np.stack([first, second], axis=1) * (2.0 / s2)[:, None]).ravel()


def selector(mode: ModeWbhf, pre: int) -> phy.Selector:
    """The equalizers for ``mode``, ``pre`` known symbols ahead of the first
    frame: waveform 0's Walsh symbols decided whole by the path model alone,
    the others' frames by the block path equalizer beside the RLS one."""
    if mode.walsh:
        return phy.Selector(CHIPS, 0, pre)
    from .blockeq import BlockEqualizer
    return phy.Selector(mode.w.unknown, mode.w.known, pre, path_cls=BlockEqualizer)


def loss_stop(mode: ModeWbhf, after: int):
    """The signal has gone when the known symbols (waveform 0: the decided
    Walsh symbols) have been as wrong as noise makes them for a while: both
    wrong outright (as the other receivers judge it, which a strong signal's
    fade is not) and well past what they were in the preamble. These
    waveforms work below 0 dB, where the known symbols' error is past the
    first test throughout (at -6 dB it is about 4 with the signal there and
    30 with it gone); the second is what tells the two apart there. Noise
    alone scores about 1 at the probes, more than half of it under 0.8, so
    each symbol counts to 3 before it is clipped, not 2 as elsewhere: at 2
    the run sat at the line for seconds after the signal had gone."""
    rate = 0.007 if mode.walsh else 0.005
    st = [0.0, 0.0, 0.0, 0]                     # run, relative run, sum, count

    def stop(k, e2):
        if k < after:
            st[2] += e2
            st[3] += 1
            return False
        base = max(st[2] / max(st[3], 1), 1e-3)
        st[0] = (1 - rate) * st[0] + rate * min(e2, 3.0)
        st[1] = (1 - rate) * st[1] + rate * min(e2 / base, 20.0)
        return st[0] > 0.8 and st[1] > 6.0
    return stop


def llrs(mode: ModeWbhf, soft, sigma2, walsh, walsh_sig) -> np.ndarray:
    if mode.walsh:
        return walsh_llr(walsh, walsh_sig)
    return demap(np.asarray(soft), np.asarray(sigma2), mode)


#: The carrier offsets acquisition searches from.
SHIFTS = (0.0, 50.0, -50.0)

#: The preamble's score that is taken as one. Waveform 0 is to work at -6 dB
#: (table D-LII), where the fixed section scores about 0.2 (it would be 0.24
#: on a clean channel at that SNR: S/(S+N) in the symbol band); noise alone,
#: searched over minutes, has not gone above 0.075, nor have the other
#: serial waveforms.
THRESHOLD = 0.13


#: Symbols past a preamble's start that must have arrived before it is taken:
#: what ``phy.find`` refines over (a run more than the fixed section, which has
#: sidelobes a run of 32 either side that clear the threshold), which also
#: covers the header ``read_header`` reads.
NEED = 2 * len(FIXED) + 8 + 2 * _TAPS[0]


def _acquire(bb: np.ndarray, start: int, threshold: float = THRESHOLD):
    """The first superframe at or after ``start`` with ``NEED`` symbols of it
    in ``bb``: (hit, count, mode), or None."""
    best = None
    for shift in SHIFTS:
        stop = bb.size if best is None else min(bb.size, best[0].sample + 2 * SUPERFRAME * SPS)
        hit = phy.find(bb[:stop], FIXED, chunk=16, threshold=threshold, start=start,
                       freq_hz=shift)
        if hit is None or hit.sample + NEED * SPS > bb.size:
            continue
        if best is not None and abs(hit.sample - best[0].sample) <= 8 * SPS:
            # The same preamble from another search. A chunk of 16 cannot
            # tell offsets 150 Hz apart, so a search far from the offset can
            # land on its alias, which scores far less: 75 Hz off, the
            # search from 0 read -75.8 Hz at 0.39, the one from 50 75.1 at
            # 0.87.
            if hit.score <= best[0].score:
                continue
        elif best is not None and hit.sample > best[0].sample:
            continue
        got = read_header(bb, hit.sample, hit.freq_hz)
        if got is not None:
            best = (hit, *got)
    return best


def receive(audio: np.ndarray, *, start: int = 0, threshold: float = THRESHOLD,
            order: str = "lsb") -> Reception | None:
    """Find the first transmission in ``audio`` (real, at ``spec.FS``) at or
    after sample ``start``, read its waveform from the preamble, and decode
    it."""
    from .stream import ReceiverWbhf, receive_whole
    receiver = ReceiverWbhf(bit_order=order, threshold=threshold)
    got = receive_whole(audio, receiver, start=start)
    if got is None:
        return None
    first, data, end = got
    mode = MODES_WBHF[first.mode]
    frames = receiver._blocks * mode.frames
    return Reception(mode, data, frames, first.at, end.at,
                     first.freq_hz, end.snr_db, end.complete)
