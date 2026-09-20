# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-1 FSK connect generation.

PACTOR-3 link setup is done in PACTOR-1 FSK: a receiver reports the connect and
starts tracking the QSO only once it sees a valid one. This module synthesises the
connect audio for an arbitrary callsign, and it is verified end to end -- an
independent PACTOR monitor reads a fully self-generated connect back as
`###CONNECT: [Normal Call: W1AW]`.

FRAME STRUCTURE, as it goes on air and as the real DL6MAA connect in
captures/real_connect_only.wav carries it:

A connect burst is ONE dual-rate FSK frame, continuous phase, 0.960 s:
  * Address section    -- 9 bytes at 100 Bd  (the primary copy)      0.72 s
  * Redundancy section -- 6 bytes at 200 Bd  (the Memory-ARQ 6-byte secondary)  0.24 s

Both halves are 96 bit periods, and the two add to 0.960 s exactly. Nothing
follows them: this file used to append a four-bit 200 Bd postamble on the
grounds that a reference demod needed a trailing edge to finalise the last
symbol, and it made the frame 0.980 s. Three sources say 0.960 -- the protocol
description's `/Header/SLAVCALL@100Bd/SLAVCA@200Bd/`, a reference receiver's
96-bit and 192-bit link-setup registers, and real off-air connect frames that
measure 0.964 and 0.965 s of key-down including their own envelope edges. The
20 ms also ate the changeover budget, which is 1.250 - 0.960 = 0.290 s and has
to hold the far end's answer.

FSK: MARK(bit1)=1400 Hz, SPACE(bit0)=1600 Hz (polarity INVERTED vs the naive guess;
confirmed by correlation against the real signal), each byte LSB-first.

Encoding -- both sections go on air exactly as written, no transform:
  1. Address image T = [0x55 sync] + uppercase-ASCII callsign + 0x0F fill to 9 bytes.
  2. Redundancy section (200 Bd) = T[1:7], the callsign again. The parser cross-
     checks secondary[i] against primary[i+1], which this satisfies by construction.

The receiving parser does rotate each byte it locks
(T[i] = (raw[i]>>1) | (bit0(raw[i+1])<<7)), and this file used to pre-compensate
for that with an inverse rotate. That was wrong: the rotation is the receiver's
byte lock sitting one bit early, not something the transmitter undoes. See
`address_bytes` and `redundancy_bytes` for what the air actually carries, and
docs/protocols/pactor/pactor1-link-request.md for what settled it: four independent lines, why
burst envelope onset (the obvious discriminator) gives the wrong answer, and why
a reference decoder refusing a half-changed frame was evidence for this form
rather than against it.

The whole burst is one continuous-phase waveform (100 Bd part then 200 Bd part).

The 0.960 s frame is checked in both directions. `tests/shrike/test_p1rx.py`
recovers the callsign byte-exact from self-generated bursts for six callsigns and
both variants, and from the real off-air connect -- and measures the three wrong
encodings alongside, so the gate is known to discriminate.
`tests/shrike/test_p1_oracle.py` renders a whole link setup out of this module
alone and requires an independent PACTOR-1 monitor to read the payload back out
of it, with a corrupted-field arm as the negative.
"""

from __future__ import annotations

import numpy as np

from . import coding, compress, spec

FS = spec.SAMPLE_RATE
BAUD_ADDR = 100.0       # address section symbol rate
BAUD_RED = 200.0        # redundancy section symbol rate
MARK = 1400.0           # bit 1  (inverted polarity, confirmed on-air)
SPACE = 1600.0          # bit 0
SYNC = 0x55
PAD = 0x0F
ADDR_LEN = 9            # address section length in bytes (9 @ 100 Bd)
RED_LEN = 6             # redundancy section length in bytes (6 @ 200 Bd)


def address_image(callsign: str) -> bytes:
    """T: [0x55 sync] + uppercase ASCII callsign + 0x0F fill to 9 bytes."""
    c = callsign.upper().encode("ascii")
    if not 1 <= len(c) <= ADDR_LEN - 1:
        raise ValueError("callsign must be 1..8 characters")
    for ch in c:
        if not 0x2C < ch <= 0x5A:
            raise ValueError(f"callsign char {ch:#x} outside parser-accepted range")
    return bytes([SYNC]) + c + bytes([PAD]) * (ADDR_LEN - 1 - len(c))


def _inverse_rotate(t: bytes) -> bytes:
    """On-air bytes whose per-byte cross-rotate yields image `t`.

    Parser: T[i] = (raw[i]>>1) | (bit0(raw[i+1])<<7); this inverts it.
    """
    raw = bytearray(len(t))
    raw[0] = ((t[0] & 0x7F) << 1) & 0xFF
    for i in range(1, len(t)):
        raw[i] = (((t[i] & 0x7F) << 1) | (t[i - 1] >> 7)) & 0xFF
    return bytes(raw)


def address_bytes(callsign: str) -> bytes:
    """The 9 on-air address bytes, sent at 100 Bd: the image, as-is.

    No transform. `_inverse_rotate` used to be applied here, which shifts the
    whole byte stream one bit left, and it is not what a real station sends.
    Demodulated from three real sync bursts in the corpus, from first principles
    and independently of this code, the 100 Bd section reads

        55 4B 45 35 59 54 41 0F 0F      = [sync]"KE5YTA"[0x0F][0x0F]

    which is `address_image` exactly. A receiver hunting for the sync byte can
    still lock onto a stream that is one bit late, which is why the shifted form
    was accepted by a decoder and got answers on the air -- but it is a bit out
    from every other station on the band.
    """
    return address_image(callsign)


def redundancy_bytes(callsign: str) -> bytes:
    """The 6 on-air redundancy bytes, sent at 200 Bd: the callsign, again.

    Measured from the same real bursts, the 200 Bd tail reads

        4B 45 35 59 54 41               = "KE5YTA", no sync byte, no fill

    i.e. `address_image(callsign)[1:7]` -- which is what the description's
    packet diagram says it is, /Header/S L A V C A L L/SLAVCA/. It exists
    "lediglich zur Ueberpruefung der Kanalqualitaet": the called station answers
    CS1 if it arrives clean and CS4 if it does not.

    This once applied the rotation TWICE. Correcting only this half -- leaving
    the address section shifted -- made the two disagree and a reference decoder
    stopped reading the frame, which read as evidence for the double rotation.
    It was evidence that the two halves must agree.
    """
    return address_image(callsign)[1 : 1 + RED_LEN]


def _bits_lsb_first(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")


def _fsk_freqs(bits, sps: int, invert: bool = False) -> np.ndarray:
    """Per-sample carrier frequency for a bit sequence at `sps` samples per baud.

    Kept separate from the phase integral because a connect burst runs two baud
    rates under ONE continuous phase, so the rates concatenate before the cumsum.
    """
    hi, lo = (SPACE, MARK) if invert else (MARK, SPACE)
    return np.repeat(np.where(np.asarray(bits), hi, lo), sps)


CONNECT_VARIANTS = ("normal", "longpath", "robust")
"""Connect variants shrike can transmit, by the call type each announces.

`normal` is variant code 1, a Normal Call. `longpath` is code 5, a Longpath Call:
the same frame with byte 1 of the address IMAGE complemented before the inverse
rotate, the redundancy section carrying the true byte so the far end can recover
it. Long path is how a station signals it is being heard the long way round.

`robust` is code 9, and it is not this frame at all: it is the branch-B family
below, SCS's Robust Connect, 11 bytes at one rate with a CRC-16 and no sync byte.
"""


# --- branch B: Robust Call and Free Signal ---------------------------------
#
# A different connect frame, sharing only the tones and the 1.25 s raster:
# 11 bytes, 88 symbols, 0.88 s, all at 100 Bd, no 0x55 sync and no 200 Bd
# redundancy section. Its structure is in docs/protocols/pactor/pactor-connect-frames.md
# sec 4; the framing is measured, from seven trains of real SCS modems read
# byte-clean by an independent monitor (K6SDR, AJ7C, W5STX, KY4RY, WM4RB, KD4JWF,
# and the Free Signal ident DAO8).
#
# The 11 bytes `a[]` are a fold-inverse of an 8-byte block `b[]` -- six payload
# bytes carrying eight 6-bit characters, then the X.25 FCS of those six. The
# receiver folds a[] back to b[] and checks the CRC; the three complement
# relations among a[6..10] are its cheap pre-gate, and they survive both the
# whitening mask and a global polarity flip, so a decoder can apply them before
# it knows either.
#
# The whitening mask IS the kind: identity says Robust Call, 0x55 says Free
# Signal Normal, 0x0F says Free Signal Encrypted, and each has an inverse
# (0xFF / 0xAA / 0xF0) that is the same frame sent in the other shift.
#
# An independent monitor reads a twelfth ring byte that no gate touches; nothing
# keys it.

CALL_B_LEN = 11
CALL_B_CHARS = 8
CALL_B_MASKS = {"robust": 0x00, "fs_normal": 0x55, "fs_encrypted": 0x0F}
CALL_B_CRC = "x25"
CALL_B_RESIDUE = 0xF0B8


def call_b_address(callsign: str) -> str:
    """The 8-character address field: the callsign repeated on a space, cyclically.

    Derived from the tapes rather than from any published text, and all seven
    known-good idents agree: `K6SDR` goes out as `K6SDR K6`, `AJ7C` as `AJ7C AJ7`,
    the 6-character `KD4JWF` as `KD4JWF K`. The field is always full; the space is
    a separator, and it doubles as the receiver's terminator, which is why a
    monitor prints only the part before it. An 8-character callsign fills the
    field exactly and carries no space at all.
    """
    c = callsign.upper()
    if not 1 <= len(c) <= CALL_B_CHARS:
        raise ValueError("callsign must be 1..8 characters")
    if any(not 0x20 < ord(ch) <= 0x5F for ch in c):
        raise ValueError(f"{callsign!r} has a character outside the 6-bit set")
    return ((c + " ") * CALL_B_CHARS)[:CALL_B_CHARS]


def call_b_bytes(callsign: str, kind: str = "robust") -> bytes:
    """The 11 on-air bytes of a branch-B connect frame, sent LSB-first at 100 Bd.

    Byte-exact against the air: this returns
    `1c 5d db a5 07 32 b7 68 48 97 b7` for `("K6SDR", "robust")`, which is what
    K6SDR's modem keyed ten times on 10.145 MHz on 2026-07-24.
    """
    if kind not in CALL_B_MASKS:
        raise ValueError(f"unknown branch-B kind {kind!r}; "
                         f"choose from {list(CALL_B_MASKS)}")
    v = 0
    for k, ch in enumerate(call_b_address(callsign)):
        v |= (ord(ch) - 0x20) << (6 * k)
    b = v.to_bytes(6, "little")
    fcs = coding.crc16(b, CALL_B_CRC)
    b += bytes([fcs & 0xFF, fcs >> 8])
    c6, c7 = ~b[6] & 0xFF, ~b[7] & 0xFF
    a = [b[0] ^ c6, b[1] ^ c7, b[2] ^ b[6], b[3] ^ b[7], b[4] ^ c6, b[5] ^ c7,
         c6, c7, b[6], b[7], c6]
    return bytes(x ^ CALL_B_MASKS[kind] for x in a)


def build_call_b(callsign: str, kind: str, invert: bool = False,
                 amp: float = 0.11, phase0: float = 1.22) -> np.ndarray:
    """One 0.88 s branch-B burst: Robust Call or either Free Signal, for `callsign`.

    Same tones, bit order and drive as the connect burst, one rate throughout.
    `invert` flips the shift, which a receiver reads as the mask's inverse and
    reports as the same kind.
    """
    bits = _bits_lsb_first(call_b_bytes(callsign, kind))
    freqs = _fsk_freqs(bits, int(FS / BAUD_ADDR), invert)
    return amp * np.cos(phase0 + 2 * np.pi * np.cumsum(freqs) / FS)


def connect_frame_bytes(callsign: str, variant: str = "normal") -> tuple[bytes, bytes]:
    """(address section, redundancy section) on-air bytes for a connect variant."""
    if variant == "normal":
        return address_bytes(callsign), redundancy_bytes(callsign)
    if variant == "longpath":
        image = bytearray(address_image(callsign))
        image[1] ^= 0xFF
        return bytes(image), redundancy_bytes(callsign)
    if variant == "robust":
        raise ValueError("the robust connect is one 11-byte section at one rate; "
                         "call `call_b_bytes`")
    raise ValueError(f"unknown connect variant {variant!r}; "
                     f"choose from {list(CONNECT_VARIANTS)}")


def _dualrate_frame(callsign: str, amp: float, phase0: float,
                    variant: str = "normal", invert: bool = False) -> np.ndarray:
    """One continuous-phase dual-rate connect burst."""
    addr, red = connect_frame_bytes(callsign, variant)
    sps_a = int(FS / BAUD_ADDR)
    sps_r = int(FS / BAUD_RED)
    freqs = np.concatenate([_fsk_freqs(_bits_lsb_first(addr), sps_a, invert),
                            _fsk_freqs(_bits_lsb_first(red), sps_r, invert)])
    phase = phase0 + 2 * np.pi * np.cumsum(freqs) / FS
    return amp * np.cos(phase)


def connect_signal(
    callsign: str,
    amp: float = 0.11,
    lead_s: float = 0.5,
    tail_s: float = 0.5,
    phase0: float = 1.22,
    variant: str = "normal",
    invert: bool = False,
) -> np.ndarray:
    """Full PACTOR-1 connect audio for `callsign` (float32 in [-1,1], FS=48 kHz).

    A single dual-rate burst between short silences; an independent monitor reads
    it back as a connect to `callsign`.
    `variant` selects which call it announces -- see CONNECT_VARIANTS. `robust`
    keys the shorter branch-B frame instead, which is 0.88 s rather than 0.960.
    """
    burst = (build_call_b(callsign, "robust", invert, amp, phase0)
             if variant == "robust"
             else _dualrate_frame(callsign, amp, phase0, variant, invert))
    return np.concatenate(
        [np.zeros(int(lead_s * FS)), burst, np.zeros(int(tail_s * FS))]
    ).astype(np.float32)


# --- PACTOR-1 ARQ data frames ----------------------------------------------
#
# After the FSK connect, an ARQ QSO transfers user data one packet per 1.25 s
# cycle. The on-air packet is one continuous-phase FSK burst of 96 bauds (100 Bd,
# 12 bytes) or 192 bauds (200 Bd, 24 bytes) = 0.96 s either way:
#
#   [header][field][status][CRC-16]
#     * 100 Bd (SL1):  1 +  8 + 1 + 2 = 12 bytes   (CRC-protected region 11, LEN 8)
#     * 200 Bd (SL2):  1 + 20 + 1 + 2 = 24 bytes   (CRC-protected region 23, LEN 20)
#
# The leading header byte sits OUTSIDE the CRC-protected region: the checksum runs
# from the field through the status byte and stops, which is why the header can
# alternate per packet without recomputing anything, and why a decoder that folds
# the header in never verifies a real frame. For the status byte's real layout see
# `status_byte` below -- the counter is PACTOR-1's own, and the data mode is read
# at bits 2-4, the way every later PACTOR reads it.
# The CRC-16 trailer goes out LOW BYTE FIRST; see DATA_CRC, settled against a real
# off-air packet.
#
# Three earlier readings of this block were wrong, and each one looked settled: the
# CRC was not big-endian, the status byte was not PACTOR-3's, and a packet does not
# need to be sent twice to be accepted. A single copy is read end to end. Memory-ARQ
# repetition is faithful and costs nothing, but nothing depends on it.
# See docs/protocols/pactor/pactor1-control-signals.md for what settled each.
#
# This layout is anchored on the air rather than on our own encoder:
# `tests/shrike/test_p1data.py` decodes packets two real stations transmitted,
# byte-exact, and requires zero decodes across corpus recordings that are provably
# not PACTOR-1.

DATA_FIELD = {100: 8, 200: 20}   # payload bytes per speed level
# The header byte, and it is not a constant. From the protocol description:
#
#   "1) Header : Bitmuster 55(HEX) ... Bei jedem Paket, das neue Information
#    enthaelt, wird das Bitmuster invertiert."
#
# An ALTERNATION, not a two-valued flag. The pattern starts at 0x55 and is
# INVERTED by each packet carrying new information, so a retransmission goes out
# with whatever the packet it repeats went out with. Reading it as "0xAA means
# fresh, 0x55 means a repeat" is what this said, and the air says otherwise:
# W4DNA's link-setup announcement is retransmitted on five consecutive cycles in
# captures/offair_KE5YTA_p3.wav and every copy carries header 0xAA. (The same
# five bursts settle the shift question -- they alternate polarity strictly.)
#
# Link setup pins the phase: after the first valid control signal the caller
# sends "das erste normale Datenpaket mit Head=AA (HEX) und Paketzaehler=1". So
# the header follows the counter's low bit during the initial sending turn.
# A break-in starts a new turn whose first ordinary packet is count 1/head55;
# a speed-down also resets the retained packet's header to55. The host must
# therefore preserve the current header phase separately and pass it below.
#
# This was 0x00, which is neither.
SYNC_HEADER = 0x55               # sync packet; even counter in the setup phase
DATA_HEADER = 0xAA               # odd counter in the setup phase
IDLE = spec.IDLE                 # RS -- "dient als IDLE-Symbol", not 0x00
# CRC-16/X-25 (0x1021 reflected, init and xorout 0xFFFF), transmitted LOW BYTE
# FIRST -- the HDLC/X.25 FCS convention. The description says only "nach
# CCITT-Norm", which is not specific enough to implement from, and the obvious
# reading (CCITT-FALSE, big-endian) is wrong in both halves.
#
# Settled against a real off-air packet rather than against our own decoder:
#   on air   aa 31 77 34 64 6e 61 0d 1e | 31 | 41 5b
#   x25 over bytes 1..9 = 0x5b41, low byte first = 41 5b. Exact.
# Corroborated by the canonical HDLC residue: crc16(field..crc, x25) ^ 0xFFFF
# = 0xF0B8, which a wrong variant reproduces with probability 2**-16.
DATA_CRC = "x25"


def status_byte(packet_count: int, data_type: int = 0, *,
                changeover_request: bool = False, qrt: bool = False,
                bits45: int = 0) -> int:
    """The PACTOR-1 status byte. NOT PACTOR-3's -- the layouts differ.

        bit 0-1  packet counter, modulo 4
        bit 2-3  Datenmodus (`spec.DataType`), and two bits is the whole of it:
                 the 1990 description calls bit 4 "noch nicht belegt". PACTOR-2
                 onward widens the field to bits 2-4 for the PMC modes, so a
                 modern receiver reads our bit 4 as data type and a PACTOR-1 one
                 does not -- which is the asymmetry the grants answer
        bit 5    long-cycle request
        bit 6    BK-Anforderung
        bit 7    QRT

    shrike was building this with `spec.status_byte`, which is PACTOR-3's: a
    THREE-bit data type spanning bits 2-4, and bit 5 as a long-cycle request.
    PACTOR-1 has no long or short cycle -- speed is signalled by CS4 and by
    packet length -- so every 200 Bd packet went out with a bit set that means
    nothing here, and any data type above 1 corrupted bit 4.

    BITS 4-5 GO OUT CLEAR, and `bits45` sets either or both. Which is right is
    UNSETTLED, and the two readings disagree about what the two bits even are.

    THEY ARE NOT ONE FIELD, and the arms that read them apart are the experiment.
    Under the modern three-bit reading bit 4 is the top of the data type and bit 5
    is "suggests switching to data mode" -- a request, not a declaration. So of the
    four values only two say anything coherent about a plain-ASCII payload:

        0x01  type 0, no request      -- what we send today
        0x11  type 4 (PMC German!)    -- misdeclares the payload, requests nothing
        0x21  type 0, REQUEST         -- the only coherent way to ask
        0x31  type 4, request         -- what W4DNA sent, and what 2026-08-22 tested

    Both on-air stations that were granted PACTOR-3 set BIT 5: W4DNA 0x31 and the
    DL6MAA session 0x35. Neither can be told apart from bit 4 without an arm at
    0x21, because both of them set both.

    THE DATES DECIDE IT. Every session in this station's record that was granted
    PACTOR-3 -- codeword 0x59A in the answer slot, eight cycles running at zero bit
    errors, shift sense alternating -- is dated 2026-08-02 or earlier, when this
    function returned `| 0x30`. Eight sessions of 269. Since the bits were cleared,
    across every session on every band, not one grant has arrived: on 2026-08-22
    K0NTS ran a full ARQ exchange, alternated its acknowledgement, advanced our
    counter and broke in, with four control signals at zero errors and not one word
    that could be a mis-windowed grant.

    The reading that cleared them was that W4DNA's 0x31 announcement "WAS NEVER
    ACKNOWLEDGED ... a station being refused six times". An independent PACTOR
    monitor decoding the same capture reads it as the corpus's first ground-truth
    positive: FRNR 1-5 repeat `1w4dna` at level 1, FRNR 6 comes back at level 3,
    and FRNR 9 carries `Welcome W4DNA  QTC` from the gateway. The refusals are its
    first five frames; the resolution is in the sixth.

    What they mean to a station built after 1990 is measured. An independent PACTOR
    monitor, given one field of `1W9SSJ`, reports `TYPE: 4` under status 0x31 and
    `TYPE: 0` under 0x01 -- so it takes bits 2-**4** as the data type, the way every
    later PACTOR does, and 4 in that table is PMC German compression. Bit 5 is the
    long-cycle request (published, not measured here). So 0x31 tells a modern
    gateway that eight bytes of plain ASCII are Markov-compressed German and asks
    for a 3.75 s cycle we do not implement. The CRC is valid, the frame is well
    formed, and there is nothing in it to acknowledge.

    AND THE CORPUS SPLITS ON PACKET ROLE, which is the reading that survives both.
    The two on-air ANNOUNCEMENTS -- W4DNA 0x31, the DL6MAA session 0x35 -- carry
    them set. The two on-air MID-LINK data packets we reproduce byte for byte,
    `pos_p1_data_jn36lf.wav` counts 3 and 0, carry 0x03 and 0x04 with them clear.
    So they plausibly declare capability on the first packet of a link rather than
    on every packet, and setting them everywhere breaks a byte-exact gate against
    real traffic (`tests/shrike/test_p1.py`).

    The on-air arms have since said otherwise, and the shipped arm default moved
    to 3 (`onair.P1_STATUS_ANNOUNCE`): 0x31 drew grants on 08-22, 08-26 and
    08-29 while 0x21 and clear drew none, and the same gateway that grants a
    0x31 announcement walks a clear-bits link up to the mail layer and refuses
    it there as PACTOR-1. THIS function's
    default stays 0: it is the renderer's neutral value, the mid-link value both
    JN36lf packets carry, and the byte-exact gates in `test_p1.py` stand on it.
    AND THE TWO BITS ARE NOT A PAIR. Bit 4 is the top of the data type a PACTOR-2
    or later receiver reads (`spec.DataType` spans bits 2-4 there, and bits 2-3
    alone in PACTOR-1); bit 5 is not. The four type values that require bit 4 are
    the PMC modes -- published in the PACTOR-III table, absent from PACTOR-1,
    whose own compression is Huffman. So 0x31 carries no unassigned capability
    flag: it declares a compression a PACTOR-1 modem cannot produce, and that is
    what the grant answers. 0x31 and 0x21 both set bit 5 and differ only in bit 4.
    Across 27 announced arms, bit 4 set drew 15 grants of 22 and bit 4 clear drew
    0 of 5 -- including an interleaved
    0x21/0x31/0x21 triple inside four minutes on one gateway and centre, where
    only the middle arm was granted and the other two were answered at 2 and 4
    control signals. Bit 5 alone has never been flown.

    What we declare is not what we send. The transmit path holds no compressor,
    so 0x31 announces PMC German over plain ASCII.
    """
    if not 0 <= packet_count <= 3:
        raise ValueError("packet counter is modulo-4")
    # TWO BITS HERE because the third is written through `bits45`, which is the
    # knob the arms turn and the only writer of bit 4 -- a type of 4 or more
    # would set that bit from two parameters at once. The values this one
    # carries, 0 and 1, mean the same under either reading.
    if not 0 <= data_type <= 3:
        raise ValueError("PACTOR-1 data mode is 2 bits")
    return (packet_count | (data_type << 2) | ((bits45 & 3) << 4)
            | (int(changeover_request) << 6) | (int(qrt) << 7))


def data_packet(payload: bytes, baud: int = 100, packet_count: int = 0,
                data_type: int = spec.DataType.ASCII_8BIT,
                header: int | None = None, *,
                changeover_request: bool = False, qrt: bool = False,
                bits45: int = 0) -> bytes:
    """One PACTOR-1 data packet: [header][field][status][CRC-16 low byte first].

    `payload` is 8-bit data, right-padded with `IDLE` (0x1E, not 0x00) or
    truncated to the speed level's field size. `baud` is 100 (8-byte field) or 200 (20-byte field).
    `header` defaults to the initial sending turn's counter/header phase.
    A live host supplies its current phase explicitly after a break-in or
    speed-down; retransmissions retain that header with the same information.

    The payload has to be a character stream already -- `compress.transparent`
    is what makes one out of arbitrary bytes. Padding is indistinguishable from
    data here and at every receiver: a field ending in a data 0x1E arrives one
    byte short.
    """
    n = DATA_FIELD[baud]
    field = payload[:n] + bytes([IDLE]) * max(0, n - len(payload))
    count = packet_count & 3
    status = status_byte(count, data_type, changeover_request=changeover_request,
                         qrt=qrt, bits45=bits45)
    if header is None:
        header = DATA_HEADER if count & 1 else SYNC_HEADER
    protected = field + bytes([status])
    crc = coding.crc16(protected, DATA_CRC)
    return bytes([header]) + protected + bytes([crc & 0xFF, crc >> 8])


def field_bytes(field: bytes, data_type: int) -> bytes:
    """The characters a data field carries: everything in it that is not IDLE.

    Not `rstrip`. Under a byte-oriented mode IDLE is a character rather than a
    trailer, and a receiver takes it out wherever it sits -- Sailer's hfkernel
    does (`if (*pp != PACTOR_IDLE)`), and so does SCS's own monitor: MEASURED
    2026-09-01 over 0x1E at each of the 8 positions of a 100 Bd field and each
    of the 20 of a 200 Bd one, 28 of 28 reported one byte short, and a full
    field whose last byte was a data 0x1E lost it exactly as a padded one did.

    A coded field is a BIT stream and is handed on whole: its IDLE is a code
    word, and `compress.Decoder` drops it at the symbol level where it lives.

    So this is lossy by construction and the loss is the protocol's: what makes
    a byte stream survive it is `compress.transparent` at the far end of the
    link, not a rule here.
    """
    return (field if compress.bit_stream(data_type)
            else field.translate(None, bytes([IDLE])))


# PACTOR-1's four control signals, 12 bits each. These are the link-layer
# acknowledgements -- a separate, older set from the six 20-bit codewords PACTOR-2
# and -3 share (spec.CONTROL_SIGNALS), which is why matching a P1 reply against
# those found nothing.
#
# The PACTOR-1 protocol description states them normatively -- "CS1: 4D5 CS2: AB2
# CS3: 34B CS4: D2C (all hex numbers, LSB right)" -- and an independent
# implementation's codeword table holds the same four, stored little-endian. Two
# sources agreeing, one of them normative, and the four are mutually Hamming
# distance 8. See docs/protocols/pactor/pactor1-control-signals.md.
#
# TABULATED IN THE SAME BIT SENSE AS A DATA FIELD, which is a fact about the air
# and not a choice about the table. A control signal's one and a data byte's one
# ride the SAME tone, so `_fsk_freqs` maps both the same way and nothing here is
# complemented on its way to the modulator. See `control_signal`.
CONTROL_SIGNALS: tuple[int, ...] = (0x4D5, 0xAB2, 0x34B, 0xD2C)

# What they MEAN, which the binary table could never have told us:
#
#   "CS1..3 have the same function as their AMTOR counterparts; CS4 serves as
#    the speed change control. In contrast to AMTOR, CS3 is transmitted as head
#    portion of a special changeover packet."
#
# The AMTOR reference is the whole content of CS1 and CS2, and it is NOT a pair of
# distinct meanings. In AMTOR/SITOR-A the receiving station acknowledges by
# ALTERNATING between two control signals; the acknowledgement is carried by the
# TOGGLE, not by the codeword. Repeating the same one is how a repeat is
# requested. So there is no separate "NAK" codeword to look for, and shrike
# looking for one is why answering a real station changed nothing whichever of the
# four it sent -- a constant CS2 reads as "send that again", forever.
#
#   CS1/CS2  acknowledge, alternating. Toggle = the packet was good, send the
#            next one. Repeat the previous one = send that packet again.
#   CS3      break-in, and NOT a bare burst: it is the first two bytes (three at
#            200 Bd) of the new sending station's own packet, and the station
#            yielding reads the rest of that packet before answering.
#   CS4      speed change, and context-dependent rather than one meaning. After a
#            faulty packet it is a REJECT -- discard it and send the information
#            again at 100 Bd. After a correctly received 100 Bd packet it is an
#            acknowledgement that forces 200 Bd. Consecutive CS4s with no valid CS
#            between them are a plain repeat request. It is also one of the two
#            legal answers to a connect (the other is CS1): CS4 says the 200 Bd
#            redundancy section did not arrive cleanly.
CS_ACK_A, CS_ACK_B, CS_CHANGEOVER, CS_SPEED = 0, 1, 2, 3
CS_BITS = 12

# Two more twelve-bit words are tabulated alongside the four in implementation
# codeword tables, and PACTOR-1 assigns them no meaning
# (docs/protocols/pactor/pactor1-control-signals.md sec 3). They are shaped like
# control signals -- weight 6, so DC-free on the air -- and sit at distance 6 from
# every published word and from each other.
#
# They are matched so that a receiver can SAY one arrived. Read against the four
# alone, 0x59A is six errors from all of them in every reading, so a real one
# reaches the log as "CS1, 6 bit errors" and is discarded as noise: the burst at
# 4.412 s of the DL6MAA capture, whose sender answers it by keying PACTOR-3.
#
# WHICH IS WHAT IT ASKS FOR. 0x59A in the answer slot is the PACTOR-3 upgrade
# grant, measured at two gateways and in eight of this station's own sessions,
# and `ptc.PtcHost._take_grant` keys the entry packet on one -- by default, since
# a link goes as far as both ends allow and this word is the far end's half of
# that (`--decline-grant` is the way to refuse it).
#
# 0x6A9 IS THE OTHER HALF OF THE SAME ANSWER and still drives nothing. A gateway
# returned it fifteen times in fifteen consecutive cycles, zero errors, to an
# announcement setting status bit 4 with bit 5 clear, and returned 0x59A to the
# same announcement with bit 5 set -- same slot, same 120 ms burst, same
# repetition. Bit 5 picks the word. What 0x6A9 asks for is unknown: no recording
# of two commercial stations holds one, so acting on it would be validated
# against nothing but our own convention.
#
# THE TABLE'S THIRD EXTRA, 0xB2A, IS DELIBERATELY ABSENT. It is the bit complement
# of CS1, and every reader here matches both shift senses, so 0xB2A is already
# what a CS1 in the inverted shift looks like on the air. Adding it would take
# half of every peer's acknowledgements -- the half the Shiftlage rule inverts --
# and report them as an unassigned word instead of as the CS1 they are.
UNASSIGNED_SIGNALS: tuple[int, ...] = (0x6A9, 0x59A)

# The index space of a decoded control signal: the four that mean something, then
# the two that are only recognised.
CS_WORDS: tuple[int, ...] = CONTROL_SIGNALS + UNASSIGNED_SIGNALS
CS_6A9, CS_59A = 4, 5


def control_signal(index: int, *, repeats: int = 1, amp: float = 0.11,
                   invert: bool = False,
                   msb_first: bool = False) -> np.ndarray:
    """One PACTOR-1 control signal as FSK audio, sent `repeats` times.

    ONE copy, and the tone mapping is the inverse of what this used to send.
    Both were established off-air rather than reasoned about.

    A real station's control signal measures 115-135 ms -- twelve bits at 100 Bd,
    a SINGLE copy. The earlier reading of ~350 ms as "three copies, as memory-ARQ
    expects" came from a burst measured with a magnitude detector that smears
    across transitions; every clean example in the corpus is one copy.

    ONE TONE SENSE FOR THE WHOLE PROTOCOL. `invert` here means exactly what it
    means for `_fsk_burst`: a one sits on SPACE instead of MARK. The codeword goes
    to the modulator as tabulated, and this used to complement it first.

    The complement came from reading two real bursts 2.5 s apart -- two cycles, so
    the SAME shift position -- and concluding that CONTROL_SIGNALS was tabulated in
    the opposite bit sense from a data field. Two samples of one parity cannot
    answer that question, and a round trip never could: our decoder read 1600 Hz as
    a one to match, so encoder and decoder agreed at zero errors on a signal one
    cycle out of phase with the band.

    What answers it is a frame and the control signal that ANSWERS IT, in the same
    cycle, where both directions share the shift (pactor1-data-packets.md sec 7):

        W4DNA packet t=7.616 + CS4 at +110 ms   frame 1 = SPACE   codeword 1 = SPACE
        W4DNA packet t=8.865 + CS4 at +109 ms   frame 1 = MARK    codeword 1 = MARK
        W4DNA packet t=10.114 + CS4 at +115 ms  frame 1 = SPACE   codeword 1 = SPACE

    read off the raw tone magnitudes with no decoder involved, 96 frame bits and 12
    codeword bits agreeing at 0% or 100% and never in between, over three cycles of
    alternating polarity. A DL6MAA connect answered by CS1 at +42 ms gives the same
    answer through the two decoders. Same tone, no exception.

    It cost the acknowledgements we send as the receiving station, which went out
    one parity off. It cost more than that as the SENDING station: the grid takes
    its shift from the peer's control signal and hands the same number to both
    renderers, so a control signal out of step with a frame puts every DATA PACKET
    in the wrong shift, from the first answer onward. Those do not arrive as noise
    -- the peer reads the inverted-FCS constant, repeats its codeword rather than
    alternating, and the link never advances.

    BIT ORDER: least-significant bit first, which is what every other PACTOR-1
    field does. shrike sent these MSB-first for its whole on-air history, and
    because the four words are closed under reversal that is not noise on the air
    -- it is the OTHER codeword of the pair. Every acknowledgement left as its
    partner and every break-in left as the speed-change signal. The acknowledgement
    alternation happened to survive the swap, CS1 and CS2 both being
    acknowledgements, which is why no round trip in the suite could notice: encoder
    and decoder shared the mistake and agreed at zero errors on the wrong name.
    `msb_first` keeps the other reading available to the analysis scripts.

    `index` runs over `CS_WORDS`, so the two unassigned words render as well as
    the four control signals -- a decoder that claims to recognise a word has to
    be measurable against one. Nothing in the transmitter asks for those two.
    """
    w = CS_WORDS[index]
    bits = [(w >> (CS_BITS - 1 - i)) & 1 for i in range(CS_BITS)] if msb_first \
        else [(w >> i) & 1 for i in range(CS_BITS)]
    phase = 2 * np.pi * np.cumsum(_fsk_freqs(bits * repeats, FS // 100, invert)) / FS
    return amp * np.cos(phase)


def _fsk_burst(data: bytes, baud: int, phase0: float, amp: float,
               invert: bool = False) -> np.ndarray:
    """Continuous-phase FSK for `data`, LSB-first, MARK=1/SPACE=0 (inverted)."""
    freqs = _fsk_freqs(_bits_lsb_first(data), int(FS / baud), invert)
    phase = phase0 + 2 * np.pi * np.cumsum(freqs) / FS
    return amp * np.cos(phase)


def packet_signal(payload: bytes = b"", baud: int = 100, *, packet_count: int = 1,
                  header: int | None = None, amp: float = 0.11,
                  invert: bool = False, changeover_request: bool = False,
                  qrt: bool = False, lead_s: float = 0.05, tail_s: float = 0.05,
                  phase0: float = 1.22, bits45: int = 0) -> np.ndarray:
    """ONE PACTOR-1 data packet as audio (float32, FS=48 kHz).

    The caller's half of link setup. The description is explicit about what
    follows a successful sync: "Sobald das erste gueltige CS empfangen und
    synchronisiert ist, wird das erste normale Datenpaket mit Head=AA (HEX) und
    Paketzaehler=1 ausgesendet." The station that called sends a PACKET once the
    called station answers -- control signals travel the other way, from the
    receiving station. Answering a peer's control signal with another control
    signal, which is what shrike did, is not a move this protocol has.

    `changeover_request` is the ISS asking to hand the channel over -- the move
    that lets a Winlink RMS send its banner after our `1<MYCALL><CR>` -- and
    `qrt` ends the link. Both were unreachable from here, so shrike could open a
    PACTOR-1 session and had no way to give the far end a turn in it.
    """
    pkt = data_packet(payload, baud, packet_count, header=header, bits45=bits45,
                      changeover_request=changeover_request, qrt=qrt)
    burst = _fsk_burst(pkt, baud, phase0, amp, invert)
    return np.concatenate([np.zeros(int(lead_s * FS)), burst,
                           np.zeros(int(tail_s * FS))]).astype(np.float32)


# --- the changeover packet -------------------------------------------------
#
# "In contrast to AMTOR, CS3 is transmitted as head portion of a special
# changeover packet." A break-in is therefore not a burst at all: the station
# taking the link sends its OWN first packet, whose first 120 ms happen to be the
# CS3 codeword, and the station yielding switches to receive on hearing that and
# reads the remaining 840 ms of the same transmission.
#
# The packet is 960 ms like any other -- that is what makes the grid arithmetic
# work, both stations rotating by 960 - 120 = 840 ms in the same cycle -- so the
# head is taken OUT of the data field rather than added in front of it. At 100 Bd
# the twelve codeword bits fill twelve of sixteen and the field loses one byte; at
# 200 Bd they are sent DOUBLED, which fills three bytes exactly and keeps the head
# at 120 ms, and the field loses two.
#
# There is no header byte: the head is where it would have been. The CRC covers
# the same [field][status] region it always does, which here begins after the head.
#
# Sourced and adjudicated in docs/protocols/pactor/pactor1-control-signals.md §3 (CS3 as head portion, and
# that nothing on the air was ever a bare CS3) and docs/protocols/pactor/pactor1-timing.md §7
# (the bit-doubling, and the 120/160 ms a receiver skips before the data).
BREAKIN_HEAD = {100: 2, 200: 3}    # bytes the CS3 head occupies
BREAKIN_FIELD = {100: 7, 200: 18}  # ...and what is left for data


def breakin_head(baud: int = 100) -> bytes:
    """The CS3 head as on-air bytes: the same tones a bare CS3 carries.

    The codeword goes in as tabulated. These bytes reach the air through
    `_fsk_burst` and a bare CS3 reaches it through `control_signal`, and the two
    map a one to the same tone -- which is the whole requirement on these bytes:
    the first 120 ms of the packet has to read as CS3 to a station whose receiver
    is looking for a control signal there and for nothing else.

    The four spare bits at 100 Bd are sent as zeros. Nothing reads them: a station
    that has recognised the head skips 160 ms before the data, and one that has not
    is reading twelve bits.
    """
    w = CONTROL_SIGNALS[CS_CHANGEOVER]
    bits = [(w >> i) & 1 for i in range(CS_BITS)]
    if baud == 200:
        bits = [b for b in bits for _ in (0, 1)]
    n = BREAKIN_HEAD[baud]
    bits += [0] * (8 * n - len(bits))
    return bytes(sum(b << i for i, b in enumerate(bits[j:j + 8]))
                 for j in range(0, 8 * n, 8))


def breakin_packet(payload: bytes = b"", baud: int = 100, *,
                   packet_count: int = 0, data_type: int = spec.DataType.ASCII_8BIT,
                   qrt: bool = False) -> bytes:
    """One changeover packet: [CS3 head][field][status][CRC-16], 960 ms.

    The counter resets to 0 across a direction change, so `packet_count` defaults
    to it; the packet after this one is the first of the new direction and carries
    header 0x55.
    """
    head = breakin_head(baud)
    n = BREAKIN_FIELD[baud]
    field = payload[:n] + bytes([IDLE]) * max(0, n - len(payload))
    status = status_byte(packet_count & 3, data_type, qrt=qrt)
    protected = field + bytes([status])
    crc = coding.crc16(protected, DATA_CRC)
    return head + protected + bytes([crc & 0xFF, crc >> 8])


def breakin_signal(payload: bytes = b"", baud: int = 100, *, packet_count: int = 0,
                   amp: float = 0.11, invert: bool = False, qrt: bool = False,
                   lead_s: float = 0.05, tail_s: float = 0.05,
                   phase0: float = 1.22) -> np.ndarray:
    """One changeover packet as audio -- the IRS taking the link by force."""
    pkt = breakin_packet(payload, baud, packet_count=packet_count, qrt=qrt)
    burst = _fsk_burst(pkt, baud, phase0, amp, invert)
    return np.concatenate([np.zeros(int(lead_s * FS)), burst,
                           np.zeros(int(tail_s * FS))]).astype(np.float32)


def data_signal(payload: bytes, baud: int = 100, *, repeats: int = 1,
                packet_count: int = 0, data_type: int = spec.DataType.ASCII_8BIT,
                cycle_s: float = spec.CYCLE_SHORT_S, amp: float = 0.11,
                lead_s: float = 0.5, tail_s: float = 0.5,
                phase0: float = 1.22) -> np.ndarray:
    """PACTOR-1 ARQ data audio: `repeats` copies of one packet on the ARQ grid.

    Memory-ARQ sends the identical packet each cycle until acknowledged; each copy
    is one FSK burst placed at the start of a `cycle_s` slot (float32, FS=48 kHz).
    """
    pkt = data_packet(payload, baud, packet_count, data_type)
    burst = _fsk_burst(pkt, baud, phase0, amp)
    slot = int(round(cycle_s * FS))
    out = np.zeros(int(lead_s * FS) + max(repeats * slot, len(burst))
                   + int(tail_s * FS), dtype=np.float32)
    for i in range(repeats):
        s = int(lead_s * FS) + i * slot
        out[s:s + len(burst)] += burst.astype(np.float32)
    return out
