# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-III channel coding: CRC, convolutional encode/decode, puncturing, interleaving.

Order on the wire, per the spec: the information packet (user data + status byte
+ 2 CRC bytes) is convolutionally encoded, punctured, and then "full-frame
bit-interleaved" across the whole packet.

Everything the spec leaves open is a PARAMETER here, never a baked-in constant --
so a search over the candidate combinations is a loop over parameter sets, not a
code rewrite. See `unknowns.py` for what is genuinely unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import spec, unknowns

# ---------------------------------------------------------------------------
# CRC-16
# ---------------------------------------------------------------------------

# The spec says "CCITT-CRC16" and gives the polynomial by name only. The three
# real-world variants that go by that name differ in init/reflection/xorout, and
# guessing wrong is invisible until a receiver rejects the packet -- so all three
# are here and the choice is an explicit, tagged parameter.
CRC_VARIANTS: dict[str, dict] = {
    "ccitt-false": dict(init=0xFFFF, refin=False, refout=False, xorout=0x0000),
    "kermit":      dict(init=0x0000, refin=True,  refout=True,  xorout=0x0000),
    "x25":         dict(init=0xFFFF, refin=True,  refout=True,  xorout=0xFFFF),
}


def _reflect(x: int, width: int) -> int:
    r = 0
    for _ in range(width):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def reflected_crc_ccitt_table() -> list[int]:
    """The 256-entry reflected CRC-16/CCITT table (poly reversed = 0x8408).

    Used for the CRC and data whitening. First entries: 0x0000, 0x1189, 0x2312, ...
    """
    tbl = []
    for b in range(256):
        crc = b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
        tbl.append(crc & 0xFFFF)
    return tbl


_WHITEN = reflected_crc_ccitt_table()
_CRC_TABLE_LSB = _WHITEN                    # shared CRC and whitening table


def _msb_crc_ccitt_table() -> list[int]:
    """The un-reflected companion to the above, for the ccitt-false variant."""
    tbl = []
    for b in range(256):
        crc = b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ spec.CRC_POLY) & 0xFFFF if crc & 0x8000 \
                else (crc << 1) & 0xFFFF
        tbl.append(crc)
    return tbl


_CRC_TABLE_MSB = _msb_crc_ccitt_table()


def whiten(data: bytes) -> bytes:
    """PACTOR-3 data whitening (its own inverse).

    The buffer is XORed, as BIG-endian 16-bit words indexed by word position i,
    with the CRC table entry crctable[i]; the position counter resets per packet.

    The byte order is a measurement, not a convention: an all-zero 9-byte case-0
    field is reconstructed at the far end as `00 00 11 89 23 12 32 9b`, which is
    crctable[0..3] = 0x0000, 0x1189, 0x2312, 0x329B laid down HIGH BYTE FIRST.
    Little-endian words are the natural reading and they are wrong.

    Not documented in ITU-R M.1798, and not optional: a raw, unwhitened coding
    chain produces a frame that no PACTOR-3 receiver decodes.
    """
    b = bytearray(data)
    odd = len(b) & 1
    if odd:
        b.append(0)
    out = bytearray(len(b))
    for i in range(0, len(b), 2):
        w = _WHITEN[(i // 2) % 256]
        out[i] = b[i] ^ (w >> 8)
        out[i + 1] = b[i + 1] ^ (w & 0xFF)
    return bytes(out[:len(data)]) if odd else bytes(out)


def crc16(data: bytes, variant: str = "x25") -> int:
    """CRC-16 with polynomial 0x1021, in the named variant.

    Default is CRC-16/X-25: init 0xFFFF, reflected in and out, xorout 0xFFFF. The
    variant is pinned by its residue, 0xF0B8 -- the value a correct message plus
    its own CRC leaves behind -- which separates it from the other 0x1021
    variants, so a receiver's verify step doubles as identification of the
    variant.
    """
    if variant not in CRC_VARIANTS:
        raise ValueError(f"unknown CRC variant {variant!r}; "
                         f"choose from {sorted(CRC_VARIANTS)}")
    p = CRC_VARIANTS[variant]
    # Byte-at-a-time, which is how PACTOR receivers do it -- the reflected
    # variants run on `reflected_crc_ccitt_table` above. The bitwise form this
    # replaces was 8x the work, and the P1 packet scan CRCs tens of thousands of
    # candidate frames per capture.
    if p["refin"]:                          # refin and refout always agree here
        reg = _reflect(p["init"], 16)
        for byte in data:
            reg = (reg >> 8) ^ _CRC_TABLE_LSB[(reg ^ byte) & 0xFF]
    else:
        reg = p["init"]
        for byte in data:
            reg = ((reg << 8) & 0xFFFF) ^ _CRC_TABLE_MSB[((reg >> 8) ^ byte) & 0xFF]
    return reg ^ p["xorout"]


def crc16_rows(data: np.ndarray, variant: str = "x25") -> np.ndarray:
    """`crc16` of every row of a (k, n) uint8 array, as a (k,) uint32 array.

    The register recurrence is sequential in the byte but independent across rows,
    so a scan of k candidate frames is n table lookups wide rather than k*n
    lookups long. The PACTOR-1 packet search gates tens of thousands of candidate
    alignments per capture, which is a call per frame the narrow way.
    """
    if variant not in CRC_VARIANTS:
        raise ValueError(f"unknown CRC variant {variant!r}; "
                         f"choose from {sorted(CRC_VARIANTS)}")
    p = CRC_VARIANTS[variant]
    cols = np.asarray(data, np.uint32).T
    if p["refin"]:
        reg = np.full(cols.shape[1], _reflect(p["init"], 16), np.uint32)
        for col in cols:
            reg = (reg >> 8) ^ _TABLE_LSB[(reg ^ col) & 0xFF]
    else:
        reg = np.full(cols.shape[1], p["init"], np.uint32)
        for col in cols:
            reg = ((reg << 8) & 0xFFFF) ^ _TABLE_MSB[((reg >> 8) ^ col) & 0xFF]
    return reg ^ p["xorout"]


_TABLE_LSB = np.array(_CRC_TABLE_LSB, np.uint32)
_TABLE_MSB = np.array(_CRC_TABLE_MSB, np.uint32)


# ---------------------------------------------------------------------------
# bit <-> byte helpers  (MSB-first; bit order is itself a candidate -- see U8)
# ---------------------------------------------------------------------------

def bytes_to_bits(data: bytes, msb_first: bool = True) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    if not msb_first:
        bits = bits.reshape(-1, 8)[:, ::-1].ravel()
    return bits.astype(np.uint8)


def bits_to_bytes(bits: np.ndarray, msb_first: bool = True) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).copy()
    if bits.size % 8:
        raise ValueError(f"bit count {bits.size} is not a multiple of 8")
    if not msb_first:
        bits = bits.reshape(-1, 8)[:, ::-1].ravel()
    return np.packbits(bits).tobytes()


# ---------------------------------------------------------------------------
# Convolutional code
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvCode:
    """Rate-1/2 feed-forward convolutional code.

    `generators` are octal, e.g. (0o171, 0o133) for K=7. Which generator produces
    the FIRST output bit is not stated by the spec -- swap them to test the other
    ordering (both orderings are in unknowns.U1_GENERATORS.candidates).
    """
    constraint_length: int
    generators: tuple[int, int]

    @property
    def n_states(self) -> int:
        return 1 << (self.constraint_length - 1)

    def _taps(self) -> tuple[np.ndarray, np.ndarray]:
        K = self.constraint_length
        out = []
        for g in self.generators:
            out.append(np.array([(g >> (K - 1 - i)) & 1 for i in range(K)],
                                dtype=np.uint8))
        return out[0], out[1]

    def encode(self, bits: np.ndarray, terminate: bool = True) -> np.ndarray:
        """Encode, MSB-first, optionally flushing the register with K-1 zeros."""
        K = self.constraint_length
        g0, g1 = self._taps()
        bits = np.asarray(bits, dtype=np.uint8)
        if terminate:
            bits = np.concatenate([bits, np.zeros(K - 1, dtype=np.uint8)])
        # shift register, newest bit first
        reg = np.zeros(K, dtype=np.uint8)
        out = np.empty(bits.size * 2, dtype=np.uint8)
        for i, b in enumerate(bits):
            reg[1:] = reg[:-1]
            reg[0] = b
            out[2 * i] = np.bitwise_xor.reduce(reg & g0)
            out[2 * i + 1] = np.bitwise_xor.reduce(reg & g1)
        return out


CODE_K9 = ConvCode(9, (0o657, 0o435))
"""The K=9, rate-1/2 mother code PACTOR-2 and PACTOR-III case 0 share.

Published for PACTOR-2: SCS, *The PACTOR-2 Protocol -- A Technical Description*
(1996), Annex I p. 6 -- "PACTOR-2 always uses a Convolutional Code with k=9 and
R=1/2 [...] The polynomials are G1=111101011; G2=101110001" -- and reprinted in
SCS, *The PACTOR-4 Protocol*, section 11.5.

WHICH END OF A PRINTED STRING IS THE NEWEST TAP is a convention the document does
not state, and the two readings give different codes. Read newest-bit-first the
strings are (0o753, 0o561), the classic optimum K=9 rate-1/2 pair; read
oldest-bit-first they are the (0o657, 0o435) here, which is that pair reciprocal.
`ConvCode._taps` indexes the register newest bit first, so this object is the
SECOND reading -- the printed strings taken as oldest-bit-first -- and saying it
is the first was a straight contradiction until 2026-09-02.

The choice is settled from outside twice over, which is why it can be stated as a
choice: `test_p2.py::test_accepted_vector` pins the 144 channel bits an
independent decoder's own trellis returned byte-exact for a frame built this way,
and the HB9AK fixtures decode byte-exact off tape. Both codes have d_free = 12,
so nothing local could have separated them.

PACTOR-III's slowest path independently reaches the same taps by measurement
(`placement.CASE0_CODE`), so one object owns them rather than two protocols each
declaring the pair."""


@lru_cache(maxsize=None)
def _trellis(code: ConvCode) -> tuple[np.ndarray, ...]:
    """Branch amplitudes and predecessors for `code` -- a pure function of the code.

    State convention (must match encode()): the state holds the K-1 PREVIOUS input
    bits, with bit 0 = most recent. Feeding input b therefore yields
        next = ((state << 1) | b) & mask
    and so, going backwards, the input that produced a given state is simply
    `state & 1` -- which is what makes viterbi_decode's traceback unambiguous.

    Cached because the timing search calls the decoder once per candidate symbol
    position and this was rebuilt every time -- 512 ufunc reduces per call at K=9,
    which cost more than the trellis sweep it was setting up for.
    """
    K = code.constraint_length
    g0, g1 = code._taps()
    states = np.arange(code.n_states)
    out = np.zeros((code.n_states, 2, 2), dtype=np.int8)
    for s in range(code.n_states):
        for b in (0, 1):
            reg = np.empty(K, dtype=np.uint8)
            reg[0] = b                                  # newest bit
            for i in range(1, K):
                reg[i] = (s >> (i - 1)) & 1             # older bits
            out[s, b, 0] = np.bitwise_xor.reduce(reg & g0)
            out[s, b, 1] = np.bitwise_xor.reduce(reg & g1)

    # +1 for coded bit 0, -1 for coded bit 1: the branch metric is a correlation,
    # which we MAXIMISE. Passing 0.0 for a punctured bit contributes nothing --
    # exactly the "exactly intermediate value" the spec asks for.
    amp = np.where(out == 0, 1.0, -1.0)

    # For every next-state, its two predecessors and the input bit that got there.
    b_of = (states & 1).astype(np.int64)
    pred0 = (states >> 1).astype(np.int64)
    pred1 = pred0 | (1 << (K - 2))
    return (amp[pred0, b_of, 0], amp[pred0, b_of, 1],
            amp[pred1, b_of, 0], amp[pred1, b_of, 1], pred0, pred1)


def viterbi_decode(soft: np.ndarray, code: ConvCode, terminated: bool = True,
                   pinned: bool = False) -> np.ndarray:
    """Soft-decision Viterbi.

    `soft` holds one value per CODED bit, where +1 means "confidently 0" and -1
    means "confidently 1" (i.e. it is the expected BPSK amplitude). A punctured
    bit is passed in as 0.0 -- exactly the "neither a 1 nor a 0, but an exactly
    intermediate value" the spec describes.

    `terminated` says the last K-1 steps carry the flush and their bits are not
    information. It does NOT say the register was at zero there, and by default
    neither boundary state is assumed, because on the air neither one holds for
    PACTOR-3. Measured on the DL6MAA session: a decoder pinned to state 0 at
    both ends recovers a real speed level 5 or 6 field everywhere except its
    first byte and its last few -- which is where the CRC lives -- so all
    fourteen of that session's punctured packets failed the CRC while their
    payload sat in plain sight, the same repeating test pattern the speed level
    3 packets carry, one byte out of place. Reading the ends free returns every
    one of them, and the fields it returns are byte-for-byte what the pinned
    decoder already had in the middle. The lower speed levels never showed it:
    at rate 1/2 the trellis recovers inside the field's own slack, so the CRC
    bytes are clean either way.

    `pinned` holds both boundary states at zero, and it is the honest model for
    PACTOR-2, whose encoder measurably starts at zero and flushes to zero
    (`pactor2.encode_frame`, and the flush is inside `Path.n_buf`). The freedom
    is not free at the decode edge: on the faded 33.15 s burst of
    `hb9ak_055246_c1500.wav` the free-boundary trellis prefers an impostor path
    whose re-encoded-from-zero metric is 36 below the true field's, and pinning
    the ends is one of the three changes that turned that burst from a miss
    into a byte-exact decode. Pinning also shrinks the space a chance CRC
    accept can come from.
    """
    K = code.constraint_length
    n_states = code.n_states
    soft = np.asarray(soft, dtype=np.float64)
    if soft.size % 2:
        raise ValueError("expected an even number of coded soft values")
    n_steps = soft.size // 2
    a0_0, a0_1, a1_0, a1_1, pred0, pred1 = _trellis(code)

    if pinned:
        metric = np.full(n_states, -1e12)
        metric[0] = 0.0
    else:
        metric = np.zeros(n_states)
    back = np.zeros((n_steps, n_states), dtype=np.int32)

    for t in range(n_steps):
        r0, r1 = soft[2 * t], soft[2 * t + 1]
        cand0 = metric[pred0] + (a0_0 * r0 + a0_1 * r1)
        cand1 = metric[pred1] + (a1_0 * r0 + a1_1 * r1)
        take1 = cand1 > cand0
        metric = np.where(take1, cand1, cand0)
        back[t] = np.where(take1, pred1, pred0)

    state = 0 if pinned else int(np.argmax(metric))
    bits = np.zeros(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        bits[t] = state & 1          # the input bit is the low bit of the state
        state = int(back[t, state])
    if terminated:
        bits = bits[: n_steps - (K - 1)]
    return bits


# ---------------------------------------------------------------------------
# Puncturing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Puncture:
    """A puncture pattern over the rate-1/2 output stream.

    `mask` is a 2 x period array of 0/1; 1 = transmit, 0 = delete, row 0 for the
    G0 output and row 1 for G1, with the period counted in trellis steps. Rate 1/2
    is the all-ones mask, which makes the stage a no-op rather than a branch at
    every caller that does not puncture.
    """
    mask: tuple[tuple[int, ...], ...]

    @property
    def period(self) -> int:
        return len(self.mask[0])

    @property
    def rate(self) -> tuple[int, int]:
        kept = sum(sum(row) for row in self.mask)
        return (self.period, kept)

    def apply(self, coded: np.ndarray) -> np.ndarray:
        m = np.array(self.mask, dtype=bool)          # shape (2, period)
        pairs = coded.reshape(-1, 2).T               # shape (2, n)
        n = pairs.shape[1]
        keep = np.tile(m, (1, int(np.ceil(n / self.period))))[:, :n]
        return pairs.T[keep.T]                       # column-major, time-ordered

    def depuncture(self, soft: np.ndarray, n_pairs: int) -> np.ndarray:
        """Reinsert punctured positions as 0.0 (the neutral value)."""
        m = np.array(self.mask, dtype=bool)
        keep = np.tile(m, (1, int(np.ceil(n_pairs / self.period))))[:, :n_pairs]
        full = np.zeros((2, n_pairs), dtype=np.float64)
        full.T[keep.T] = soft
        return full.T.ravel()


RATE_1_2 = Puncture(((1,), (1,)))

# PACTOR-III's two puncture patterns, not guessed: row 0 is the keep-mask for
# output G0 and row 1 for G1, over one period. Rate 3/4 keeps 4 of 6 and rate 8/9
# keeps 9 of 16, which is what SL5's and SL6's published net data rates require.
PUNCTURE_3_4 = Puncture(((1, 1, 0), (0, 1, 1)))                       # SL5
PUNCTURE_8_9 = Puncture(((0, 1, 0, 1, 1, 0, 1, 0),                    # SL6
                         (1, 0, 1, 0, 1, 0, 1, 1)))

# PACTOR-2's two punctured levels, published as serial vectors over the rate-1/2
# stream: SCS, *The PACTOR-2 Protocol -- A Technical Description* (1996), Annex I
# p. 6 -- SL3 at R=2/3 with vector 1011, SL4 at R=7/8 with 10100110101011: 3 of 4
# kept, then 8 of 14.
#
# ANNEX I DOES NOT SAY HOW TO FOLD A SERIAL VECTOR into the two per-generator
# rows this class takes, and two readings are available. ROW-MAJOR: the first
# half of the vector is G0's whole mask, the second half is G1's. INTERLEAVED:
# even positions are G0 and odd positions G1, the order the punctured stream
# leaves in.
#
# SL3 IS THE ROW-MAJOR READING WITH NO ROTATION. Twelve frames of
# `hb9ak_055246_c1500.wav` come out byte-exact under ((1,0),(1,1)) and no other
# rotation returns any of them; `1011` split down the middle is exactly that
# pair, and it is also the textbook rate-2/3 matrix over a rate-1/2 mother code.
# The interleaved reading reaches the same mask only after rotating the printed
# vector one place, so the two agree at period 2 and part at period 7.
#
# AT SL4 THEY DIVERGE AND NOTHING HERE ARBITRATES. Free distance cannot: over
# this K=9 code, minimised across puncture phase, the mother code is 12, both
# rate-2/3 candidates are 7, and every rate-7/8 candidate -- both readings and
# all their rotations -- is 3. Only level-4 material can, and the corpus holds
# none. So both are named and the row-major one is shipped, because it is the
# reading SL3 confirms without a rotation.
PUNCTURE_2_3 = Puncture(((1, 0), (1, 1)))                             # P2 SL3
PUNCTURE_7_8 = Puncture(((1, 0, 1, 0, 0, 1, 1),                       # P2 SL4
                         (0, 1, 0, 1, 0, 1, 1)))
PUNCTURE_7_8_INTERLEAVED = Puncture(((1, 0, 0, 1, 0, 0, 0),
                                     (1, 1, 0, 1, 1, 1, 1)))
"""SL4's other reading: the printed vector rotated one place, taken interleaved.

`PUNCTURE_7_8` is what ships; this is the candidate it was chosen over, kept so
that a level-4 recording -- or a monitor that reads one and not the other --
settles the question without the pair being derived again. Both were put to SCS's
own monitor at speed level 4, idle and text fields, two runs each, inside a link
the monitor was already tracking at speed level 3. It read the level-3 lock in
every segment of both runs and not one level-4 frame of 128."""

# kept under the old name for callers; now the confirmed decoder patterns
CANDIDATE_PUNCTURES: dict[str, Puncture] = {
    "3/4-yasuda": PUNCTURE_3_4,
    "8/9-yasuda": PUNCTURE_8_9,
}


# ---------------------------------------------------------------------------
# Interleaving
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockInterleaver:
    """Rectangular block interleaver: write by rows, read by columns.

    "Full-frame bit-interleaving of the entire data packet" -- the geometry is
    UNKNOWN (unknowns.U3_INTERLEAVER); this class makes it a parameter.
    """
    n_rows: int
    n_cols: int

    def _perm(self, n: int) -> np.ndarray:
        if self.n_rows * self.n_cols < n:
            raise ValueError(
                f"interleaver {self.n_rows}x{self.n_cols} too small for {n} bits")
        idx = np.arange(self.n_rows * self.n_cols)
        perm = idx.reshape(self.n_rows, self.n_cols).T.ravel()
        return perm[perm < n]

    def interleave(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits)[self._perm(bits.size)]

    def deinterleave(self, bits: np.ndarray) -> np.ndarray:
        perm = self._perm(bits.size)
        out = np.empty_like(np.asarray(bits))
        out[perm] = bits
        return out


def conv_code_for(sl: int, generators: dict[int, tuple[int, int]] | None = None) -> ConvCode:
    """The convolutional code for a speed level, using the ASSUMED generators."""
    gens = generators or unknowns.U1_GENERATORS.value
    K = spec.SPEED_LEVELS[sl].constraint_length
    return ConvCode(constraint_length=K, generators=gens[K])
