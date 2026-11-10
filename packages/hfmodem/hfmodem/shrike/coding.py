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

import numpy as np

# M.1798 names only "CCITT-CRC16". Init, reflection and xorout distinguish
# three variants, so callers retain an explicit variant choice. The X.25
# choice is pinned by the 0xF0B8 residue of a message including its CRC.
from hfmodem.core.crc import (
    CRC_VARIANTS as CRC_VARIANTS, crc16 as crc16, crc16_rows as crc16_rows,
    reflected_crc_ccitt_table as reflected_crc_ccitt_table,
)
from hfmodem.core.fec import (
    ConvCode as ConvCode, Puncture as Puncture, viterbi_decode as viterbi_decode,
)

from . import spec, unknowns

_WHITEN = reflected_crc_ccitt_table()


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


# ---------------------------------------------------------------------------
# Puncturing
# ---------------------------------------------------------------------------


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
    """The convolutional code for a speed level, using the ASSUMED generators.

    The spec does not state which generator emits the first bit; both orders
    remain in unknowns.U1_GENERATORS.candidates.

    PACTOR-3 uses free Viterbi boundary states, even when the last K-1 steps
    are omitted as flush bits. Pinning both ends to zero spoiled the first
    byte and the CRC end of all fourteen punctured SL5/6 fields in the DL6MAA
    session; free ends recovered every field, agreeing with the pinned
    decoder in the middle. At rate 1/2 the trellis recovered within the
    field's slack, hiding the boundary mistake.

    PACTOR-2 instead starts and flushes at zero (pactor2.encode_frame, with
    flush inside Path.n_buf). On the faded 33.15 s hb9ak_055246_c1500.wav
    burst, free ends preferred an impostor whose re-encoded-from-zero metric
    was 36 below the true field's. Pinning was one of three changes that
    recovered it byte-exact, and also narrows the chance-CRC search space.
    """
    gens = generators or unknowns.U1_GENERATORS.value
    K = spec.SPEED_LEVELS[sl].constraint_length
    return ConvCode(constraint_length=K, generators=gens[K])
