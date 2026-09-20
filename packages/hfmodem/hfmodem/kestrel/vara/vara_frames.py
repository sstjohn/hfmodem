# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VARA-compatible MFSK handshake-tone generators and recognizers.

EVERY value and step below traces to a fact in ``spec/`` (cited
inline as ``[spec NN §x]``) or is our own original design (``[ours]``). This
module contains NO VARA-derived identifiers, addresses, or code — only the
functional facts published in the spec.

The VARA handshake bursts that carry a callsign (connect-request,
connect-response, and the post-connect session bursts) share one MFSK waveform
family and one payload-tone generator, differing only by three per-burst scalars
``(preamble, N_payload, SEED_OFF, PREADV)``  [spec 04 §4.2, §4.2.2]. Each is keyed
to the CALLED/destination callsign only (caller-invariant)  [spec 04 §4.2]. The
connected-ack is not one of them — it is a two-tone-per-symbol burst carrying no
callsign, and lives at the bottom of this module  [spec 04 §4.2C].

Payload-tone generator (closed form)  [spec 04 §4.2.3]::

    crc  = CRC-16/GENIBUS(callsign_ascii)            # spec 03 §3.4
    seed = (crc + SEED_OFF) & 0x7FFF
    mult = G(callsign) + ((crc + 50) >> 15)
    s = start(seed);  repeat (mult + PREADV) times: s = LCG(s)
    for k in 0..N-1:
        s = LCG(s);  P = floor((s / 2**24) * n_p)    # n_p = 5 at BW2300, 6 at BW2750, 2 at BW500
        s = LCG(s);  D = floor((s / 2**24) * 7)      # D in {0..6}
        parity = 1 if k even else 0                  # BW500 has no parity term
        bin[k] = (base + parity + 14*P + 2*D) & 0xFF # base = 29 at BW2300, 22 at BW2750, 50 at BW500

The bandwidth changes ``(base, n_p, parity)`` and nothing else — see
:class:`ToneAlphabet`, which each :class:`BurstKind` carries.

The VB6 ``Rnd`` LCG core and the ``Randomize`` seeding map (fold/start) are both
per [spec 04 §4.2.3]. The tone carrier index equals the emitted 2048-pt @48 kHz FFT bin
[spec 04 §4.2.1].
"""
from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ..coding.crc import crc16_genibus

# 6-bit packed callsign: A-Z -> 1..26, 0-9 -> 27..36, 0 terminates  [spec 04 §4.2A].
_CS6 = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
_CS6.update({c: i + 27 for i, c in enumerate("0123456789")})


def _pack_callsign6(call: str) -> bytes:
    bits = "".join(f"{_CS6[c]:06b}" for c in call.upper() if c in _CS6)
    bits = bits.ljust(-(-len(bits) // 8) * 8, "0")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _split_ssid(call: str) -> tuple[str, int]:
    """Split "W1AW-10" into ("W1AW", 10); a bare call yields SSID 0. VARA packs
    the base call into the 6-bit field and carries the SSID as an integer in
    body[5]  [spec 04 §4.2A]."""
    base, _, ssid = call.upper().partition("-")
    return base, int(ssid) if ssid.isdigit() else 0


_CS6_INV = {v: k for k, v in _CS6.items()}


def unpack_callsign6(data: bytes) -> str:
    """Inverse of :func:`_pack_callsign6`: read the 6-bit-packed callsign that
    opens a link-setup body  [spec 04 §4.2A].

    Codes are MSB-first from bit 0; 1-26 -> A-Z, 27-36 -> 0-9, and 0 terminates
    (as does any out-of-alphabet code, which is where the fixed structure bytes
    begin). Pass the frame/body — decoding stops at the terminator on its own.

    Returns the BASE call only — the 6-bit field never carries an SSID (VARA
    stores that as an integer in body[5]; use :func:`caller_from_link_setup` to
    recover the full "CALL-SSID"). The 6-bit alphabet has no '-', which is why
    the split is needed  [spec 04 §4.2A].
    """
    bits = "".join(f"{b:08b}" for b in data)
    out = []
    for i in range(0, len(bits) - 5, 6):
        ch = _CS6_INV.get(int(bits[i:i + 6], 2))
        if ch is None:
            break
        out.append(ch)
    return "".join(out)


#: Body bytes a link-setup fills at each bandwidth, ahead of its CRC-16 trailer:
#: one wideband DATA frame's worth  [spec 04 §4.2A, §3.6.4]. BW2750 carries the
#: identical 90-byte body on its own 20-bin base burst — the bandwidth changes the
#: comb, not the frame — read byte-exact off the 2026-07-21 loopback link-setup
#: over, caller AAAA1  [see rx.varahf2300.BASE_LEVELS].
LINK_SETUP_BODY = {"2300": 90, "2750": 90, "500": 44}


def link_setup_frame(caller: str, bw: str = "2300", level: int = 4) -> bytes:
    """Caller-ID setup at the negotiated host speed level (1..4).

    Eight identity bytes are followed by the short-frame trailer. At levels
    2..4 this is ``14 <call CRC high>``, zero fill, and ``04 82``. Level 1
    has only ten body bytes and ends directly in ``04 8b``. Append the body's
    CRC-16/GENIBUS in both cases. Stock-caller output independently fixes these
    formats (2026-09-20; docs/protocols/vara/30-connect-speed-negotiation.md).
    """
    if type(level) is not int or level not in (1, 2, 3, 4):
        raise ValueError("link-setup level must be a host speed level 1..4")
    sizes = (10, 23, 35 if str(bw) == "500" else 48, LINK_SETUP_BODY[str(bw)])
    body = bytearray(sizes[level - 1])
    base, ssid = _split_ssid(caller)
    packed = _pack_callsign6(base)
    body[:len(packed)] = packed
    body[5] = ssid                                            # SSID as integer, 0 if bare
    body[7] = 0x80
    if level > 1:
        body[8] = 0x14
        body[9] = crc16_genibus(caller.upper().encode()) >> 8
    body[-2] = 0x04
    body[-1] = 0x8b if level == 1 else 0x82                                            # block/trailer byte
    return bytes(body) + crc16_genibus(bytes(body)).to_bytes(2, "big")


def caller_from_link_setup(frame_bytes: bytes) -> str:
    """The full caller "CALL-SSID" from a decoded link-setup frame: base call
    from the 6-bit field, SSID from body[5] (dropped when 0)  [spec 04 §4.2A]."""
    base = unpack_callsign6(frame_bytes)
    ssid = frame_bytes[5] if len(frame_bytes) > 5 else 0
    return f"{base}-{ssid}" if ssid else base


def is_link_setup(frame_bytes: bytes) -> bool:
    """Recognize the setup structure at host levels 1..4; also require CRC.

    The compact level-1 frame omits the callsign-CRC delimiter used at higher
    levels. Caller identity is still at the same offsets in every format.
    """
    b = frame_bytes
    return ((len(b) == 12 and b[7] == 0x80 and b[-4:-2] == b"\x04\x8b")
            or (len(b) >= 25 and b[7] == 0x80 and b[8] == 0x14
                and b[-4] == 0x04))


# --------------------------------------------------------------------------- #
# Public VB6 Rnd LCG (msvbvm60), period 2**24  [spec 04 §4.2.3].
_LCG_MULT = 0x43FD43FD
_LCG_ADD = 0xC39EC3
_LCG_MASK = 0xFFFFFF


def _lcg(s: int) -> int:
    """One VB6 Rnd draw; Rnd value = returned_state / 2**24  [spec 04 §4.2.3]."""
    return (s * _LCG_MULT + _LCG_ADD) & _LCG_MASK


def _lcg_jump_coeffs(n: int) -> tuple[int, int]:
    """(a, c) with ``lcg^n(s) == (a*s + c) & MASK``.

    The draw is the affine map ``x -> MULT*x + ADD``, and affine maps compose as
    ``(a1,c1)∘(a2,c2) = (a1*a2, a1*c2 + c1)``, so n steps fold by repeated squaring
    in O(log n) instead of n Python calls. The pre-advance ahead of every burst is
    hundreds of draws, which is where the 3.2 M calls came from.
    """
    if n < 0:
        raise ValueError(f"cannot advance the generator backwards ({n})")
    a, c = 1, 0
    ba, bc = _LCG_MULT & _LCG_MASK, _LCG_ADD
    while n:
        if n & 1:
            a, c = (a * ba) & _LCG_MASK, (a * bc + c) & _LCG_MASK
        ba, bc = (ba * ba) & _LCG_MASK, (ba * bc + bc) & _LCG_MASK
        n >>= 1
    return a, c


def _lcg_advance(s: int, n: int) -> int:
    """State after ``n`` draws from ``s``  (exactly ``_lcg`` applied n times)."""
    a, c = _lcg_jump_coeffs(n)
    return (a * s + c) & _LCG_MASK


# Public VB6 Randomize(seed) seeding map  [spec 04 §4.2.3].
def _hi32(x: int) -> int:
    """High 32 bits of the little-endian IEEE-754 double of ``x``."""
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0] >> 32


def _fold(hi: int) -> int:
    return (((hi & 0xFFFF) << 8) ^ ((hi >> 8) & 0xFFFF00)) & 0xFFFFFF


def _start(seed: int) -> int:
    """24-bit LCG start state for a given Randomize seed  [spec 04 §4.2.3]."""
    return (_fold(_hi32(seed)) & 0xFFFF00) | 0x86


# --------------------------------------------------------------------------- #
# GF(2)-linear callsign hash G  [spec 04 §4.2.3].
# Only the LAST THREE characters matter; earlier characters affect ``mult`` only
# through the CRC carry.
_GBASIS = {
    -3: [110, 220, 440, 338, 134, 310],
    -2: [102, 204, 408, 274, 6, 22],
    -1: [32, 64, 128, 258, 36, 216],
}


def _g_hash(callsign: str) -> int:
    """G(cs) = 254 XOR T[-3] XOR T[-2] XOR T[-1], skipping offsets past the
    string start  [spec 04 §4.2.3]."""
    v = 254
    for off in (-3, -2, -1):
        if len(callsign) >= -off:                 # offset lands inside the string
            code = ord(callsign[off]) ^ 0x41      # ord(ch) XOR 0x41
            t = 0
            for i in range(6):                    # bits 0..5 (A-Z / 0-9)
                if (code >> i) & 1:
                    t ^= _GBASIS[off][i]
            v ^= t
    return v


# --------------------------------------------------------------------------- #
# Tone alphabets. The carrier a payload symbol lands on is
# ``base + parity + 14*P + 2*D``, where P is the symbol's first generator draw
# and D its second  [spec 04 §4.2.3]. What the bandwidth changes is this map and
# nothing else about the burst  [spec 04 §4.2.1].
@dataclass(frozen=True)
class ToneAlphabet:
    base: int                     # lowest carrier the alphabet can emit
    n_p: int                      # values the first draw takes; the second takes 7
    parities: int = 2             # parity values the symbol index alternates over

    def parity(self, k: int, par0: int) -> int:
        """The parity bit symbol ``k`` carries. One-parity alphabets have none —
        their carriers are all even and the symbol index says nothing."""
        return 0 if self.parities == 1 else (par0 if k % 2 == 0 else 1 - par0)

    @property
    def carriers(self) -> frozenset[int]:
        """Every carrier index this alphabet can emit  [spec 04 §4.2.3]."""
        return frozenset(self.base + p + 14 * P + 2 * D
                         for p in range(self.parities)
                         for P in range(self.n_p) for D in range(7))


#: 70 carriers on the 23.4375 Hz grid, bins 29..98 = 679.7..2296.9 Hz.
BW2300_TONES = ToneAlphabet(29, 5)

#: 14 carriers on a 46.9 Hz lattice, bins 50..76 = 1171.9..1781.3 Hz — the even
#: bins only, the alphabet a BW500 station's payload is drawn from.
#:
#: Solved 2026-08-15 against the 2026-07-13 BW500 loopback corpus, whose PTT
#: ledger says which station keyed each burst: nine connect requests across eight
#: sessions (seven keyed to BBBB2, one to AAAA1) and eight connect responses, all
#: locking their preamble exactly and putting every payload tone on an even
#: carrier in 50..76. An exhaustive search of all 2**24 VB6 ``Rnd`` states returns
#: exactly ONE state per burst under this map and NONE under any of the four
#: other alphabets tried — a single 14-way draw, P and D transposed, the BW2300
#: draws folded mod 14, and the BW2300 draws with the tone taken from the second
#: alone. 31 tones pin a state to one in 2**24, so the identification is not in
#: doubt.
#:
#: Derived a second time and independently the same day, off a different tape: the
#: five real VARA HF 4.9.0 -> VARA HF 4.9.0 BW500 connects driven on the Wine bench
#: (called W1AW, N0DX, W2XY, KC2OUR) reach ``base 50, n_p 2, no parity`` from four
#: called callsigns none of which appears in the loopback corpus. Two solutions
#: from two bodies of evidence agreeing byte for byte is what stands behind this
#: alphabet, rather than either one alone —
#: ``tests/kestrel/test_bw500_handshake`` holds the bench tones.
BW500_TONES = ToneAlphabet(50, 2, parities=1)

#: 84 carriers on the 23.4375 Hz grid, bins 22..105 = 515.6..2460.9 Hz: BW2300's
#: alphabet with a sixth value for the first draw and the base seven bins lower.
#:
#: Solved off the 2026-07-21 loopback's BW2750 session (AAAA1 -> BBBB2): its
#: connect-response, session-confirm and both keepalives each pin exactly one of
#: the 2**24 generator states under this map and none under BW2300's, and every
#: one of the 92 payload carriers is the BW2300 carrier moved seven bins up or
#: down — up exactly when ``Int(Rnd*6)`` exceeds ``Int(Rnd*5)`` on the same draw,
#: which is this alphabet said the other way round. The BW2300 session 50 s
#: later on the same tape is the control: on this alphabet it scores chance.
#: One session and one callsign pair  [``tests/kestrel/test_bw2750``].
BW2750_TONES = ToneAlphabet(22, 6)


# --------------------------------------------------------------------------- #
# Per-burst descriptors  [spec 04 §4.2.2].
@dataclass(frozen=True)
class BurstKind:
    name: str
    preamble: tuple[int, ...]     # fixed, callsign-independent preamble tones
    n_payload: int                # N: number of callsign-keyed payload tones
    seed_off: int                 # SEED_OFF
    preadv: int                   # PREADV pre-advance draws
    par0: int = 1                 # parity of the first payload tone (spec 04 §4.2.3)
    keyed_by: str = "called"      # "called" or "caller": whose callsign keys it
    tones: ToneAlphabet = BW2300_TONES     # the bandwidth's payload alphabet


# preamble / N / (SEED_OFF, PREADV) exactly per [spec 04 §4.2.2].
CR = BurstKind(
    "connect-request",
    (74, 68, 70, 60, 60, 77, 50, 76, 78, 74), 31, 50, 1)
CONNECT_RESPONSE = BurstKind(
    "connect-response",
    (62, 67, 55, 66, 59, 72, 68, 55), 15, 289, 511)

# A Trimode responder's connect confirmation: a 16-symbol frame on the
# connect-response's own stream (SEED_OFF 289) at lattice position 1, keyed to
# the called station — the connect-response is position 17, so PREADV is
# 1 + 2*15 * 1 = 31. K0SI on 80 m answered our FIRST link-setup with it at +0.12 s
# on both 2026-09-16 tapes (034218Z at 43.86 s, 034443Z at 23.53 and 79.24 s),
# 14-15 of 15 payload tones exact, then ignored our retries. Eight link-setup
# bench runs against a stock 4.9.0 on 2026-09-16 — both wide bandwidths, clean,
# retried, mismatched and with no link-setup at all — draw it in none, so it
# never relaxes the connect gate the two-tone ack and turn-request routes hold —
# it is taken only where a link-setup of ours is outstanding, the same window the
# turn-request-before-connected route lives in  [see
# vara_arq._connect_confirmed]. The single-tone preamble is nominal: recognition
# reads the payload alone [see recognize]. The DATA NACK use measured later is
# documented below; it does not assign the DATA action to a link-setup reply.
#
# ITS PAYLOAD IS BIT-IDENTICAL TO CONNECT-RESPONSE LATTICE POSITION 1 AT BW2300,
# because that is exactly what it is drawn from. `for_bw` swaps the alphabet and
# not the SEED_OFF, so this frame stays on stream 289 at every bandwidth while
# the connect-response's own stream does not: BW2750 answers on 579 (its level-4 offer is
# position 3 of that stream, PREADV 91). So the two readings collide at BW2300
# and nowhere else, and each bandwidth is safe for its own reason:
#
#   BW2300  same fifteen tones. `vara_arq._ANSWER_LATTICE_BY_BW["2300"]` does not
#           carry 1 and must never be widened down to it, or a Trimode station's
#           connect confirmation reads as a new connection offer.
#   BW2750  different streams, so different tones. Positions 1 and 2 of stream
#           579 are level-2 and level-3 offers and this frame is not on that stream
#           at all; both can be named in one session without ambiguity.
#
# Route precedence settles the rest: the confirmation is read only by
# `vara_arq._connect_confirmed`, only while CONNECTING at `_I_LINKSETUP_SENT`,
# and `_stream_connect_ask` runs ahead of the connect-response search — so at
# that one step the confirmation claims the payload and the offer search
# never sees it  [tests/kestrel/test_connect_confirm_stream.py].
#
# A GATED 16-SYMBOL BRACKET CARRYING IT would be named `session-confirm` by the
# bracket reader — the two share a symbol count and `_CANDIDATES` resolves that
# count to the confirm. This frame is read on the stream only, so nothing in the
# bracket route is asked to tell them apart today.
SESSION_CONNECT_CONFIRM = BurstKind(
    "session-connect-confirm", (62,), 15, 289, 31)

# In a pending DATA turnaround this same payload is a NAK. Independently
# reproduced with stock 4.9.0 by encoding a wrong CRC at BW2300: no host bytes,
# short 289/31, then 288/63 after each state query. K0SI's 2026-09-19 recording
# carries all 15/31 payload tones. The DATA action needs its own state gate;
# the existing link-setup interpretation must not leak into a connected link.
SESSION_DATA_NAK_SHORT = replace(SESSION_CONNECT_CONFIRM, name="session-data-nak-short")
SESSION_DATA_NAK_QUERY = BurstKind(
    "session-data-nak-query", (74,), 31, 288, 63, 1)

# The same burst from a station calling at BW500: same ten preamble tones — the
# preamble is bandwidth-independent [spec 04 §4.2C] — and the same 31 payload
# tones [spec 04 §4.2.1], over the fourteen-carrier alphabet and at its own
# SEED_OFF. (57, 1) is the ONLY (SEED_OFF, PREADV) pair that reaches the solved
# generator states of both the BBBB2-keyed and the AAAA1-keyed loopback requests
# from their two different callsigns' seeds; the seeding map is many-to-one, so
# the cross-callsign intersection is the whole test, as for SESSION_IDLE_RESPONSE
# below. BW2300's request sits at (50, 1) — the pre-advance is what the two share.
#
# 31 payload tones, not the 28 a lattice check reads or the 28 the 2026-08-14
# on-air burst supports: keyed audio in these captures measures 83967 samples,
# 41.000 symbol advances to three decimals, in every one of the four sessions
# whose burst start the recorder did not clip. So [spec 04 §4.2.1] is right that
# BW500 keeps BW2300's counts, and the 38 slots read off air were that recorder's
# independently measured ~9% sample loss (1.75 s x 0.906 = 1.59 s).
#
# Corroborated off air, on a fading path, as far as that recording reaches: the
# two BW500 requests on the 546 s 2026-08-14 recording of 7096.5 kHz, at wav
# 4.348 s and 286.176 s. Both lock 9/10 preamble at zero offset and put 24 of the
# next 28 tones on this alphabet's lattice, each of the four misses one bin off a
# carrier; band noise means 0.23 there over 1154 alignments of two negative
# controls and never reaches 0.58. The feed and the gateway registry attest the
# channel and the mode independently of the audio — they do not say which station
# keyed which burst and cannot, every feed timestamp being truncated to the whole
# minute. So what this recording corroborates is the alphabet, and no callsign.
#
# Nor could it: the recorder dropped ~9% of its samples, so three of the 41 keyed
# symbols never arrived and every tone behind each gap reads a slot early, so
# regenerating tone by tone against any candidate scores at chance. The scanner
# duly names nobody: 7/31 and 6/31, the right answer here and not a defect. Closing
# the gap would take resynchronising inside a burst, which is a receiver inventing
# symbols the channel did not deliver; creance's `test_monitor_handshake_scan`
# holds the scanner to naming these two by bandwidth alone.
#
# And confirmed against a second, disjoint body of evidence: the five Wine-bench
# connects of 2026-08-15, four distinct called callsigns (W1AW, N0DX, W2XY,
# KC2OUR), none of them in the loopback corpus. Intersecting their solved states
# also returns (57, 1) alone, where any one callsign leaves 9 pairs standing.
CR500 = BurstKind(
    "connect-request-500",
    (74, 68, 70, 60, 60, 77, 50, 76, 78, 74), 31, 57, 1, tones=BW500_TONES)


# And what a BW500 responder answers it with. Same eight preamble tones and same
# 15 payload tones as CONNECT_RESPONSE, over the fourteen-carrier alphabet, and
# at BW2300's SEED_OFF with a pre-advance 210 draws further on.
#
# Solved off the loopback corpus like CR500 above and then confirmed on the
# bench: a real VARA HF v4.9.0 armed at BW500 answered kestrel's request on
# 2026-08-15 and its response regenerates 15/15 for W1AW at this pair, where the
# only other (SEED_OFF, PREADV) the loopback burst admitted scores 2/15 and
# BW2300's own (289, 511) scores 0/15. The bench went on to supply three more
# called callsigns (N0DX, W2XY, KC2OUR); intersecting all four leaves (289, 721)
# alone, where W1AW by itself leaves 9 pairs standing. Two sources and four
# callsigns is the standard the rest of this file's descriptors are held to.
CONNECT_RESPONSE_500 = BurstKind(
    "connect-response-500",
    (62, 67, 55, 66, 59, 72, 68, 55), 15, 289, 721, tones=BW500_TONES)

# The BW2750 pair, off the 2026-07-21 loopback. The request is keyed on BW2300's
# alphabet — a station listening at any bandwidth reads it — and only its state
# names the bandwidth asked for: 50 at BW2300, 57 at BW500, 58 here. The answer
# is already on the session's own alphabet.
CR2750 = BurstKind(
    "connect-request-2750",
    (74, 68, 70, 60, 60, 77, 50, 76, 78, 74), 31, 58, 1)
CONNECT_RESPONSE_2750 = BurstKind(
    "connect-response-2750",
    (62, 67, 55, 66, 59, 72, 68, 55), 15, 579, 91, tones=BW2750_TONES)

#: The handshake pair each bandwidth keys: the one burst family that carries its
#: own descriptor per bandwidth rather than merely its own alphabet
#: [spec 04 §4.2.2]. Every post-connect frame is one descriptor at all three.
HANDSHAKE_BY_BW = {
    "2300": {CR: CR, CONNECT_RESPONSE: CONNECT_RESPONSE},
    "2750": {CR: CR2750, CONNECT_RESPONSE: CONNECT_RESPONSE_2750},
    "500": {CR: CR500, CONNECT_RESPONSE: CONNECT_RESPONSE_500}}
#: What a caller re-keys after its first request: one preamble tone in front of
#: the same 31 payload tones, 32 symbols against 41  [connect_request_retry].
CR_RETRY_BY_BW = {bw: replace(pair[CR], name=f"{pair[CR].name}-retry",
                              preamble=pair[CR].preamble[:1])
                  for bw, pair in HANDSHAKE_BY_BW.items()}
#: The handshake bursts themselves, retry forms included: such a burst's alphabet
#: is its own, never the session's — the BW2750 request rides the wide alphabet
#: into a 2750 session.
_HANDSHAKE = frozenset(
    [*(k for pair in HANDSHAKE_BY_BW.values() for k in pair.values()),
     *CR_RETRY_BY_BW.values()])


def connect_request(bw: str = "2300") -> BurstKind:
    """The connect request a station calling at ``bw`` keys  [spec 04 §4.2.1].

    A real VARA responder in BW2750 answers :data:`CR` too and brings the session
    up as BW2300: the request's state is the bandwidth asked for."""
    return HANDSHAKE_BY_BW[str(bw)][CR]


def connect_request_retry(bw: str = "2300") -> BurstKind:
    """The shorter request a stock caller re-keys after its first: one preamble
    tone in front of the same 31 payload tones, 32 symbols against 41.

    Read off the 2026-09-09 VARA HF 4.9.0 lattice bench — a stock 4.9.0 caller
    keys the full form once and this one every 2.8 s behind it, twelve keyings
    in one arm, payload 31/31 against ours. Nothing about the answer moved with
    it, so it buys 0.38 s of air and no more.
    """
    return CR_RETRY_BY_BW[str(bw)]


def connect_response(bw: str = "2300", level: int = 4) -> BurstKind:
    """Connect offer selecting host setup speed ``level`` at ``bw``.

    Each step below level 4 moves back one 15-tone payload in the response
    generator's stream. Keep the canonical level-4 object for existing callers.
    """
    if type(level) is not int or level not in (1, 2, 3, 4):
        raise ValueError("connect-response level must be a host speed level 1..4")
    kind = HANDSHAKE_BY_BW[str(bw)][CONNECT_RESPONSE]
    return kind if level == 4 else replace(
        kind, preadv=kind.preadv - (4 - level) * lattice_step(kind))


#: The alphabet a station running at BW draws its payload tones from. The
#: bandwidth changes the alphabet and nothing else about a session burst
#: [spec 04 §4.2.1], which is what lets one table of descriptors serve them all.
ALPHABETS = {"500": BW500_TONES, "2300": BW2300_TONES, "2750": BW2750_TONES}


@lru_cache(maxsize=None)
def for_bw(kind: BurstKind, bw: str) -> BurstKind:
    """``kind`` as a station running at ``bw`` keys and reads it.

    A session frame's ``(SEED_OFF, PREADV)`` pair is the state and its alphabet
    is the bandwidth's, so the BW500 frame is the BW2300 descriptor over
    :data:`BW500_TONES` — same preamble, same payload count, same ``keyed_by``.
    Measured on four stock BW500 sessions with two callsign pairs: twelve frame
    kinds, 41 keyings, 1233 of 1233 carriers, against 0 to 3 for the wide
    alphabet on the same audio  [spec 05 §5.3.5,
    ``tests/kestrel/test_bw500_session_frames``]. BW2750 the same way over
    :data:`BW2750_TONES`, on one loopback session  [spec 05 §5.3.6]. Only the
    handshake pair is its own descriptor per bandwidth  [HANDSHAKE_BY_BW].

    Memoised on the canonical constant, so callers go on passing
    :data:`SESSION_KEEPALIVE_A` and the ``is`` comparisons a state machine runs
    against the kind it received still hold.
    """
    bw = str(bw) if str(bw) in ALPHABETS else "2300"
    if kind in HANDSHAKE_BY_BW[bw]:
        return HANDSHAKE_BY_BW[bw][kind]
    if kind in _HANDSHAKE:
        return kind
    a = ALPHABETS[bw]
    return kind if kind.tones is a else replace(kind, tones=a)

# --------------------------------------------------------------------------- #
# The connected-ack (step 5) is NOT a member of the family above  [spec 04 §4.2C].
# It rides the same 2048-sample symbol grid, but every symbol lights TWO
# equal-amplitude carriers, it is 11 symbols long, and it carries no callsign:
# its first four symbols are a fixed preamble of tone pairs and the remaining
# seven vary with session state.
CONNECTED_ACK_NSYM = 11
CONNECTED_ACK_PREAMBLE = ((64, 67), (56, 74), (64, 69), (68, 78))

# One captured BW2300 ack, kept whole: its seven state symbols are one reading of
# one burst on a channel 10 dB above its noise, and the BW500 table below is the
# clean measurement of what those symbols are for  [spec 04 §4.2C]. Nothing keys
# it now — see the two measured bursts below, which are what a 4.9.0 reads.
CONNECTED_ACK_2300 = (
    (64, 67), (56, 74), (64, 69), (68, 78),
    (52, 86), (83, 87), (46, 82), (59, 77), (56, 58), (69, 75), (52, 66))

# THE TWO CONTROL BURSTS A BW2300 SESSION ACTUALLY CARRIES, off two stock VARA HF
# 4.9.0 instances over a fake cable on 2026-08-26 — one cable per direction, so
# which station keyed a burst is a property of the tape and not of a scorer.
#
# Each station keys ONE burst and keys it at everything: the caller's answers both
# of the responder's DATA overs in every session, the responder's answers the
# link-setup over, both of the caller's DATA overs, and the close. Twelve
# occurrences across three sessions, preamble 4 of 4 at every one, and the seven
# symbols behind it identical at every one — so they carry no per-over state, no
# ACK/NAK and no block count, unlike the three BW500 tails below.
#
# THE TAIL IS READ, and by the peer rather than by us. Same bench, same responder,
# the same 65-byte block, three arms differing only in the seven symbols this
# station keys back at the responder's over:
#
#   CONNECTED_ACK_2300's tail   the responder idles at 3.4 s to its own timeout
#   the RESPONDER's tail        the same
#   the CALLER's tail           the responder releases the turn, our block is
#                               delivered byte-exact, and its queued reply comes
#                               back to our host in the same phase
#
# THE SEVEN SYMBOLS ARE KEYED TO THE CALLER. Eight two-VARA sessions on
# 2026-08-26 — same bench, same shape, same payloads, the callsign assignment the
# only thing that moves — separate the three candidates outright, with both cables
# recorded so a burst is attributed by which file holds it:
#
#   W9SSJ -> W1AW / W1AX / KI7QQQ   the same two tails, tone for tone
#   W9SSK -> W1AW                   both tails change, sharing no symbol with these
#   W1AW  -> W9SSJ                  both change, and neither is either of these
#   N0XYZ -> KI7QQQ, VE3ABC -> G0XYZ    a distinct pair each
#
# Three callees against one caller move nothing and one changed caller moves
# everything, which is the OPPOSITE key from every other burst in this file
# [see payload_bins, keyed to the CALLED station]. Role is real as well — the two
# tails of one link differ in all seven — but role alone does not name a burst,
# and it was the only thing the three recordings of one pair could distinguish.
#
# The pair below is therefore a link W9SSJ CALLED, which is every session this
# station has put on the air. What is not derivable is the rest: the generator
# behind these seven symbols is not reversed, so a link somebody else called has
# no burst here  [see CONTROL_BURSTS_BY_CALLER].
CONTROL_BURST_CALLER_2300 = (
    (64, 67), (56, 74), (64, 69), (68, 78),
    (60, 96), (39, 49), (56, 88), (29, 61), (61, 63), (49, 95), (66, 90))
CONTROL_BURST_RESPONDER_2300 = (
    (64, 67), (56, 74), (64, 69), (68, 78),
    (34, 79), (80, 82), (50, 66), (59, 95), (64, 96), (59, 77), (57, 72))

# Two stock VARA HF4.9.0 sessions, 2026-09-04 continue-2750-{1,2},
# W9SSJ caller/W1AW responder, separate cables. Four caller final-DATA ACKs
# followed by responder release; four responder connected/final-DATA ACKs.
# All seven tail pairs differ from BW2300. The former BW2300 fallback delivers
# the data but leaves a stock BW2750 sender idling after its empty final block
# (2026-09-10 clean baseline). Responder's final pair is63/65, confirmed by
# rectangular FFT: a Hann window's overlapping skirts can falsely suggest62/64.
CONTROL_BURST_CALLER_2750 = (
    (64, 67), (56, 74), (64, 69), (68, 78),
    (53, 89), (32, 42), (49, 95), (22, 68), (61, 71), (68, 102), (63, 71))
CONTROL_BURST_RESPONDER_2750 = (
    (64, 67), (56, 74), (64, 69), (68, 78),
    (27, 87), (90, 104), (43, 73), (66, 102), (71, 103), (66, 84), (63, 65))

# THE BURST THAT ANSWERS AN OVER WITH MORE BEHIND IT. The 11-symbol burst above
# is not the answer to every over: at every INTERMEDIATE over of a multi-over
# delivery a station keys this shorter one instead, and only the last over draws
# the 11-symbol one. Three bench sessions of 2026-08-30, one cable per direction:
# a 600-byte delivery answered 6 x this + 1, a 200-byte BW500 delivery 4 x this +
# 1, and a 178-byte delivery each way 2 + 1 in both directions — the responder
# keys the same shape at the caller's intermediate overs.
#
# Two-tone throughout, like the 11-symbol burst, and its FIRST symbol is that
# burst's own opening pair — in every one of the fourteen copies on these tapes,
# and no other symbol is shared by all of them. The eight below are one form of
# it: tone for tone identical at eight of the ten a caller keyed across those
# sessions, including all six of the 600-byte run. The other two are a second
# form differing in five symbols, and nothing varied here separates the two — so
# the seven symbols behind the lead carry something, and this rig has not pinned
# what. Keyed here for the same reason the 11-symbol burst's tail is: a peer that
# reads past the lead sees stale state, and one answered with nothing at all has
# never been measured.
#
# READ ONE SYMBOL EARLY UNTIL 2026-09-02, and the correction is mechanical rather
# than a rereading: `burst_tones` padded 0.06 s ahead of the onset and capped its
# offset search at 2400 samples, so the true start lay outside the search and the
# best reachable alignment began a whole symbol in front of the burst. Uncapped,
# all three BW2300 caller copies land at +0.003 to +0.004 s of the segmenter's own
# onset and read identically; what the constant used to open with was pre-burst
# audio, and its true last symbol (52, 70) was missing.
#
# Whether the seven are keyed to the caller, as the 11-symbol tail is, is not
# established — every copy held comes from a link W9SSJ called.
OVER_CONTINUE_NSYM = 8
OVER_CONTINUE_CALLER_2300 = (
    (64, 67), (30, 74), (39, 51), (32, 62), (29, 89), (70, 82), (45, 83), (52, 70))

# THE SAME THREE BURSTS AT BW500, off the stock 4.9.0 pair of 2026-08-30 — a link
# W9SSJ CALLED, one cable per direction, every keying named by the PTT ledger.
# Preamble and lead pair are the bandwidth-neutral ones above; only the seven
# state symbols move, and they move onto the narrow alphabet's carriers.
#
# The tails of the AAAA1-called session in CONTROL_TAILS_500 below are the
# control: same role, same occasion, and the two responders share ONE of seven.
# A caller and a responder of one link share none. So these are a link's tails
# and not a bandwidth's state code, which is the key the BW2300 pair was already
# shown to carry  [spec 04 §4.2C].
#
# The caller keys a SECOND 11-symbol tail on its idle cadence, eight times at
# 12.06 s — `(58,76) (58,70) (52,76) (50,68) (54,64) (52,74) (54,66)`. It is the
# poll a drained caller owes its peer, and nothing here keys it: `idle_keepalive`
# sends the MFSK frames, and what the poll draws has not been measured from this
# end.
CONTROL_BURST_CALLER_500 = CONNECTED_ACK_PREAMBLE + (
    (52, 74), (56, 76), (52, 64), (56, 68), (56, 74), (58, 68), (60, 68))
CONTROL_BURST_RESPONDER_500 = CONNECTED_ACK_PREAMBLE + (
    (54, 72), (56, 70), (56, 76), (54, 66), (58, 72), (54, 76), (50, 72))
OVER_CONTINUE_CALLER_500 = (
    (64, 67), (50, 66), (60, 64), (50, 68), (58, 76), (52, 76), (58, 76), (60, 68))

# THE CONTINUE BURST AT BW2750, off two stock 4.9.0s on the cables, 2026-09-04 —
# a link W9SSJ CALLED, one recording per direction, each delivery padded past one
# over so both ends had to answer an intermediate one. Ten copies: the caller
# keyed this one at all three of its turns in each session, six for six identical,
# and the responder keyed two further sets of seven behind the same lead. Three of
# the responder's symbols reach bins 22-23 and 104-105, where no BW2300 burst goes
# — the wide alphabet is what these are drawn on.
#
# The lead pair is the bandwidth-neutral one the other two bandwidths open on, and
# it is the whole of what the peer is known to read: across all copies at all
# three bandwidths symbol 0 is the same and no other symbol is shared.
OVER_CONTINUE_CALLER_2750 = (
    (64, 67), (37, 81), (36, 58), (29, 81), (36, 96), (31, 77), (76, 80), (43, 59))

# The responder's intermediate-DATA answer, measured on the full W9SSJ ->
# KC9GHZ BW2750 link. Four native stock replies (two full-mail sessions,
# 2026-09-11 22:58/23:00 UTC) match all eight pairs; both sessions sent235 and
# fetched645 bytes exactly. The seven after the lead also match the receiver-
# muted response in the 2026-09-12 01:04 UTC KC9GHZ tape. This is not a NAK
# table or a general callsign law. Keep both endpoints in the lookup until
# independent links establish which endpoint/session fields select this tail.
# Provenance: tests/kestrel/fixtures/head-cut-continue/provenance.json.
OVER_CONTINUE_RESPONDER_BY_LINK = {
    # 2026-09-20 01:36 UTC: full eight-pair K0SI reply, followed by exact
    # called-keyed 288/1 confirming the same pending DATA (no retransmission).
    # See fixtures/k0si-data-ack-0920/provenance.json.
    ("W9SSJ", "K0SI", "2300"): (
        (64, 67), (58, 92), (77, 91), (48, 64),
        (65, 71), (38, 92), (81, 85), (58, 98)),
    # Three stock replies on 2026-09-19 and independent 2026-09-18 RF
    # sessions agree, including replies whose first symbol the RX mute cut.
    # See fixtures/data-replies-0919/provenance.json.
    ("W9SSJ", "KC9GHZ", "2300"): (
        (64, 67), (32, 82), (37, 81), (54, 66),
        (49, 81), (64, 84), (63, 79), (66, 96)),
    ("W9SSJ", "KC9GHZ", "2750"): (
        (64, 67), (25, 89), (30, 88), (47, 73),
        (42, 88), (71, 103), (56, 76), (73, 79)),
    ("W9SSJ", "KC9GHZ", "500"): (
        (64, 67), (52, 74), (58, 74), (60, 72),
        (56, 74), (56, 72), (62, 66), (50, 74))}

# Successful explicit descents at all four low records, paired with exact host
# delivery, qualify this positive answer to the sender's 0x81 next-record field.
OVER_CONTINUE_RESPONDER_DOWN_BY_LINK = {
    ("W9SSJ", "KC9GHZ", "2300"): (
        (64, 67), (44, 96), (33, 53), (56, 68),
        (45, 93), (68, 94), (81, 93), (50, 98)),
    ("W9SSJ", "KC9GHZ", "2750"): (
        (64, 67), (77, 91), (36, 82), (27, 83),
        (74, 94), (43, 101), (88, 100), (83, 101)),
    ("W9SSJ", "KC9GHZ", "500"): (
        (64, 67), (58, 68), (62, 64), (52, 76),
        (52, 74), (50, 72), (56, 76), (58, 64)),
}

#: The two bursts of a link, by the callsign that CALLED it and the bandwidth it
#: came up at — ours when we originated, the peer's when we answered. One caller
#: is measured and nothing here can generate a second.
#: The one caller whose bursts are measured — at both bandwidths, and the only
#: link this station has ever originated.
CONTROL_BURST_CALLER = "W9SSJ"

CONTROL_BURSTS_BY_CALLER = {
    (CONTROL_BURST_CALLER, "2300"): (CONTROL_BURST_CALLER_2300,
                                     CONTROL_BURST_RESPONDER_2300),
    (CONTROL_BURST_CALLER, "2750"): (CONTROL_BURST_CALLER_2750,
                                     CONTROL_BURST_RESPONDER_2750),
    (CONTROL_BURST_CALLER, "500"): (CONTROL_BURST_CALLER_500,
                                    CONTROL_BURST_RESPONDER_500)}

#: The answer to an over with another behind it, by the same key.
OVER_CONTINUE_BY_CALLER = {
    (CONTROL_BURST_CALLER, "2300"): OVER_CONTINUE_CALLER_2300,
    (CONTROL_BURST_CALLER, "2750"): OVER_CONTINUE_CALLER_2750,
    (CONTROL_BURST_CALLER, "500"): OVER_CONTINUE_CALLER_500}

# THE NAK, off two stock 4.9.0s on the cables, 2026-09-07 -- a link W9SSJ CALLED,
# one recording per direction, noise injected into the RECEIVER's input for the
# body of selected overs so it registered a burst and failed its CRC. The station
# that failed an over keys this 8-symbol burst in the turnaround, and the sender
# answers it by dropping a speed level and re-sending the over: over 3 of the
# give-up probe was rejected and every later over survived, and both reversed and
# forward deliveries closed byte-exact. It refutes the standing assumption that
# BW2300 keys nothing in that turnaround  [see vara_arq._undecoded_over].
#
# Eight two-tone symbols on the bandwidth-neutral lead pair the continue and
# control bursts open on, and DISTINCT BY ROLE: the caller's and the responder's
# share only that lead (1 of 8), the way a link's two control tails do. The
# receiver that keys it is the caller in RESPONDER, the responder in CALLER --
# named for the station that CALLED the link, as the rest of the family is. Four
# copies of each, identical across overs, so it is a fixed NAK and not a state
# report [spec 04 Sec 4.2C].
#
# Whether the seven behind the lead are keyed to the caller's callsign, as the
# 11-symbol tail is assumed to be, is not established: every copy comes from a
# link W9SSJ called, and this station always originates as W9SSJ.
NAK_CALLER_2300 = (
    (64, 67), (40, 62), (43, 83), (80, 98), (67, 95), (74, 88), (67, 83), (66, 74))
NAK_RESPONDER_2300 = (
    (64, 67), (66, 78), (61, 95), (70, 98), (41, 71), (34, 98), (35, 51), (46, 92))

#: ``(ours as caller, ours as responder)`` NAK for a link ``caller`` called at
#: ``bw``, by the same key as the control bursts. The RECEIVER of a failed over
#: keys its role's burst; the sender reads the other.
NAK_BY_CALLER = {
    (CONTROL_BURST_CALLER, "2300"): (NAK_CALLER_2300, NAK_RESPONDER_2300)}


# Native CRC-rejection replies from the 2026-09-19 all-bandwidth campaign.
# The legacy caller-only NAK table above belongs to a different link. These
# seven-symbol tails require both endpoints; a shared lead is not an ACK.
DATA_NAK_RESPONDER_BY_LINK = {
    ("W9SSJ", "KC9GHZ", "2300"): (
        (64, 67), (42, 46), (43, 67), (74, 96),
        (53, 93), (70, 80), (35, 43), (42, 76)),
    ("W9SSJ", "KC9GHZ", "2750"): (
        (64, 67), (39, 97), (34, 74), (67, 91),
        (46, 74), (77, 87), (28, 36), (35, 69)),
    ("W9SSJ", "KC9GHZ", "500"): (
        (64, 67), (54, 76), (56, 68), (62, 68),
        (58, 74), (60, 76), (58, 72), (62, 64)),
}


def nak(caller: str, bw: str) -> tuple | None:
    """``(caller NAK, responder NAK)`` for a link ``caller`` called at ``bw``, or
    None when no NAK of that link is measured at that bandwidth. Only BW2300 has
    one measured, off a W9SSJ-called link  [see NAK_BY_CALLER]."""
    return NAK_BY_CALLER.get((caller.upper(), str(bw)))


def _tail_bw(bw: str) -> str:
    return str(bw)


def control_bursts(caller: str, bw: str) -> tuple | None:
    """``(ours as caller, ours as responder)`` for a link ``caller`` called at
    ``bw``, or None when no link of that caller is measured at that bandwidth."""
    return CONTROL_BURSTS_BY_CALLER.get((caller.upper(), _tail_bw(bw)))


def over_continue(caller: str, bw: str) -> tuple | None:
    """The 8-symbol intermediate-over answer of that link, or None.

    THIS ONE DOES NOT FALL BACK, where the 11-symbol tails above do. Three BW2750
    fetches on 2026-09-04 keyed BW2300's copy at every changeover: each took the
    gateway's first over, answered it, and drew the responder's own idle cadence
    until the timeout, `gateway_rx_bytes` 0 in all three (053431Z, 053613Z,
    053755Z). A bandwidth whose copy is not measured keys the generated frame
    instead  [vara_arq._tx_over_response]; BW2750's has been measured since.
    """
    return OVER_CONTINUE_BY_CALLER.get((caller.upper(), str(bw)))


# THE SEVEN STATE SYMBOLS AT BW500 ARE A REPORT ON THE OVER JUST ANSWERED, and
# the table below is the only way to key one: they are measured per link and the
# generator that draws them is unknown.
#
# Ninety continues off nine BW500 tapes, both cables of each, every keying
# attributed by the cable that holds it: the seven symbols behind the lead move
# with exactly three things — the link's two callsigns, the keying station's
# role, and the state of the over just answered, where that state is the over's
# speed level together with whether its per-frame field is 0x81. Nothing else
# moves them. Not the cumulative over index (three, four and forty-nine
# identical tails in a row), not bytes acknowledged (43 to 4085), not the
# field's countdown value (0x99, 0x95, 0x91, 0x8d and 0x89 all draw one tail),
# not the host's SN reading (11.3 and 0.1 dB drew the same level-1 tail), and
# not a session nonce (three sessions on two dates agree symbol for symbol).
# 90 of 90 obey it, and a tail read from one arm predicts the other arm 8/8.
#
# The callsign pair is the key. Same caller, same role, same state: W1AW's link
# and KC9GHZ's share 0 of 7 — which is why OVER_CONTINUE_CALLER_500 above, the
# W1AW link's mid-level-4 tail, is ignored by a stock station at KC9GHZ, twice
# per over (stock500-chain, step 4). And no index law regenerates the
# symbols: every VB6 Rnd draw schedule of the family every single-tone session
# frame here is built on returns zero of 2**24 states for every tail, seed
# agnostic (the bw500-continue analysis). So a table it is, for the one
# link whose caller tails are on tape end to end, and None for every other.
#
# Sample offsets, capture hashes and the over each one answered:
# tests/kestrel/fixtures/bw500-continue/caller-tails.json.
OVER_CONTINUE_LEAD = (64, 67)
OVER_CONTINUE_CALLER_TAILS_500 = {
    ("W9SSJ", "KC9GHZ", 1, False): (
        (58, 76), (58, 72), (54, 74), (50, 72), (54, 68), (60, 66), (62, 74)),
    ("W9SSJ", "KC9GHZ", 2, False): (
        (54, 74), (60, 64), (52, 76), (54, 76), (62, 70), (62, 76), (60, 74)),
    ("W9SSJ", "KC9GHZ", 4, False): (
        (62, 68), (58, 66), (50, 70), (52, 66), (52, 64), (60, 66), (56, 74)),
    ("W9SSJ", "KC9GHZ", 4, True): (
        (60, 68), (58, 70), (54, 64), (56, 76), (62, 72), (58, 74), (50, 70))}


def over_continue_state(caller: str, called: str, level: int,
                        field: int) -> tuple | None:
    """The whole 8-symbol BW500 continue a CALLER keys after an over at ``level``
    carrying ``field``, or None where that link and state are not on tape.

    Only the caller's side of W9SSJ -> KC9GHZ is populated, and a responder's
    tails of the same link share none of its seven symbols, so this is not the
    answer for a station that answered somebody's call
    [see OVER_CONTINUE_CALLER_TAILS_500].
    """
    tail = OVER_CONTINUE_CALLER_TAILS_500.get(
        (caller.upper(), called.upper(), int(level), field == 0x81))
    return None if tail is None else (OVER_CONTINUE_LEAD,) + tail

# What the seven state symbols say, read off a responder's own transmit path
# through a whole logged BW500 session — 59 keyings, 478 s, every one of them
# labelled by the PTT ledger, so which burst answered which is not inferred.
# Eight of those keyings are this 11-symbol burst; all eight lock the preamble
# 4/4, and between them they carry exactly three tails, pairwise distinct in all
# seven symbols:
#
#   ACK    answering the link-setup, the session-confirm, and the LAST data over
#   POLL   answering each idle keepalive, and the disconnect request
#   TURN   answering the initiator's turn-request — and nothing else in 478 s
#
# Read from the sample :func:`vara_arq._ack_lock` puts the burst's first symbol
# at. That is what makes them a measurement rather than a reading of one offset:
# each burst returns these same eleven symbols at every alignment across ±800
# samples of the lock, and re-synthesised from them it correlates 0.999 with the
# audio it was read from. A symbol taken from the adjacent bin its Hann skirt
# occupies costs that outright — four such symbols score 0.73, six score 0.64.
#
# The initiator keyed its first DATA over 0.085 s after the TURN burst ended.
# That is one occurrence, which is why nothing here gates a transmission on
# reading it: the tail is reported, and a turn is taken on the answer arriving at
# all. No recording held reads the same symbols at BW2300 — the one control burst
# recoverable from either off-air gateway session is the connected-ack.
CONTROL_TAIL_ACK = (
    (58, 64), (52, 72), (54, 74), (56, 68), (58, 72), (54, 68), (58, 68))
CONTROL_TAIL_POLL = (
    (58, 72), (56, 76), (54, 68), (56, 70), (58, 68), (58, 70), (52, 68))
CONTROL_TAIL_TURN = (
    (56, 68), (60, 68), (50, 72), (54, 68), (60, 76), (60, 68), (62, 76))
CONTROL_TAILS_500 = {"ack": CONTROL_TAIL_ACK, "poll": CONTROL_TAIL_POLL,
                     "turn": CONTROL_TAIL_TURN}


def control_state(pairs: Sequence[tuple[int, int]], tolerate: int = 1) -> str | None:
    """Name the state a responder's 11-symbol control burst reports, or None.

    ``pairs`` is the whole preamble-locked burst; the four preamble symbols are
    skipped and the seven behind them matched against :data:`CONTROL_TAILS_500`.
    ``tolerate`` symbols may be lost to the channel — the three tails differ in
    all seven, so one is a wide margin and two would still not cross them.
    """
    tail = [tuple(sorted(p)) for p in pairs[len(CONNECTED_ACK_PREAMBLE):]]
    if len(tail) < len(CONTROL_TAIL_ACK):
        return None
    for name, ref in CONTROL_TAILS_500.items():
        miss = sum(1 for a, b in zip(tail, ref) if a != tuple(sorted(b)))
        if miss <= tolerate:
            return name
    return None

# --------------------------------------------------------------------------- #
# The post-CONNECTED session frames an initiator keys. Same PRNG-MFSK family as
# the bursts above, one tone per 2048-sample symbol, and every one of them a pure
# state report: the *callsign* it is keyed to says only which end of the link it
# speaks for, and the (SEED_OFF, PREADV) pair is the state  [spec 05 §5.3.3].
#
# Recovered, not captured. Each frame's 31 payload tones pin one 24-bit generator
# state out of 2**24 (30 tones would leave 2**24 / 35**30 candidates), and a
# discrete log against the seeding map gives (SEED_OFF, PREADV) exactly. Every
# pair below is confirmed on two independent recordings with *different*
# callsigns, which is what separates the state from the identity:
#
#   frame                 SEED_OFF/PREADV   seen keyed to
#   session-confirm       1551 /   91      the called station, both corpora
#   keepalive-a           1550 /  187      the called station, both corpora
#   keepalive-b             60 / 1241      the called station, loopback corpus
#   over-response           60 /    1      the called station, two gateways
#   over-response (short)   61 /    1      the called station, one gateway
#   turn-request           850 /    1      THE CALLER, loopback + two gateways
#   turn-idle              850 /   63      THE CALLER, two gateways (12 frames)
#   idle-response          288 / 1241      the called station, two gateways —
#                                          and the one frame here read off a
#                                          gateway's transmitter, not ours
#
# Two regularities fall out and are worth naming: the 16-symbol form of a state
# carries SEED_OFF one greater than its 32-symbol form (1551/1550, 61/60), and
# turn-idle sits exactly 62 generator draws — one whole frame — behind
# turn-request on the same stream.
SESSION_CONFIRM = BurstKind(
    "session-confirm", (62,), 15, 1551, 91, 1)
SESSION_KEEPALIVE_A = BurstKind(
    "session-keepalive-a", (74,), 31, 1550, 187, 1)
SESSION_KEEPALIVE_B = BurstKind(
    "session-keepalive-b", (74,), 31, 60, 1241, 1)

# The answer to a peer DATA over that leaves the peer transmitting: keyed to the
# CALLED station, like every frame above it. Measured as the answer to the
# second-to-last gateway over in both off-air sessions held.
SESSION_OVER_RESPONSE = BurstKind(
    "session-over-response", (74,), 31, 60, 1, 1)
SESSION_OVER_RESPONSE_SHORT = BurstKind(
    "session-over-response-short", (62,), 15, 61, 1, 1)

# What a stock caller keys at BW500 when it could not read the responder's owed
# DATA over: nothing into that over's turnaround, then this frame 0.24 s behind
# the responder's idle — and the responder answers it by re-sending the over from
# speed level 1 and climbing. Two arms on the cables of 2026-09-11, 31/31 tones
# each, keyed to the CALLED station: the over-response's SEED_OFF two lattice
# positions before keepalive-b, a coordinate nothing else here holds
# [tests/kestrel/fixtures/bw500-recovery, tests/kestrel/test_bw500_recovery].
# Also measured from stock W9SSJ receiving BW2750 on 2026-09-16: after an unread
# responder DATA frame and its subsequent idle, the caller keys this descriptor
# on the BW2750 alphabet (all32 symbols match). The responder resends from host
# level2 and climbs through3/4, with exact host delivery. This is idle recovery,
# not evidence for an eight-symbol BW2750 NAK in NAK_BY_CALLER.
SESSION_OVER_NAK = BurstKind(
    "session-over-nak", (74,), 31, 60, 1117, 1)

# The responder's counterpart, keyed to the CALLED station like SESSION_OVER_NAK
# and at the responder SEED_OFF the whole family shifts to — 288 is to 60 what
# 1078 is to 850. KC9GHZ answered every final-answer query of ours with it on the
# BW2750 tape of 2026-09-16, three copies at 69.312/81.093/92.837 s, 30-31 of 32
# payload tones each. It is the responder saying it could not read a DATA over of
# ours, and a keyed ask licenses one bounded resend where our own timeout would
# not  [see vara_arq._took_responder_nak]. Same experiment as
# SESSION_RESPONDER_OVER_ANSWER above, whose note records that a missing DATA
# produced 288/1117 rather than the 288/311 a delivered one draws.
SESSION_OVER_NAK_RESPONDER = BurstKind(
    "session-over-nak-responder", (74,), 31, 288, 1117, 1)

# The turn frames, and the only two in the family keyed to the CALLER — to the
# station that transmits them rather than to its peer. That is what makes the
# same frame come back identical from three different gateways: it names us.
#
# turn-request is what a station keys to become the sender. In the loopback
# corpus the initiator emitted it 0.17 s after its host handed it a payload,
# breaking a 12 s keepalive cadence to do so — idle, with its peer nine seconds
# from transmitting — and began its DATA overs 0.74 s later. In both off-air
# gateway sessions THIS station keyed the same frame — byte-identical across two
# different gateways, which is what a frame keyed to the caller does — 0.405 s
# and 0.411 s after the gateway's LAST DATA over, alone in that turnaround, and
# no wideband over ever came back from us afterwards.
#
# turn-idle is what that station then keys on the idle cadence while it holds the
# turn: 13 of them across the two sessions (7 to KC9GHZ, 6 to NS0A), 13.2 s
# apart, unchanging. Neither session timed out — the gateway transmitted into
# every one of the 13 turnarounds, 0.11-0.15 s after our burst ended, to the last
# sample of both recordings. Ten of those answers are SESSION_IDLE_RESPONSE
# below. The other three are the first turnaround of the NS0A session and the
# first two of the KC9GHZ one: gateway-strength bursts on the same symbol grid
# that are not members of this family at any alignment, are not the two-tone
# control burst either, and occur once each, so they are recorded here as
# unidentified rather than named.
SESSION_TURN_REQUEST = BurstKind(
    "session-turn-request", (74,), 31, 850, 1, 1, keyed_by="caller")
SESSION_TURN_IDLE = BurstKind(
    "session-turn-idle", (74,), 31, 850, 63, 1, keyed_by="caller")

# Stock BW2750 caller, 2026-09-11: after its final short DATA was delivered but
# its ACK was corrupted, this called-keyed solicitation retained BUFFER 79.
# A subsequent responder turn request cleared that buffer and drew SESSION_DRAINED.
# The same solicitation after corrupted DATA drew no request and retired nothing.
# Its 31 recorded payload tones uniquely fit 60/683; see test_final_ack_recovery.
SESSION_FINAL_ANSWER_QUERY = BurstKind(
    "session-final-answer-query", (74,), 31, 60, 683, 1)

# Stock repeated intermediate-ACK erasures, 2026-09-12: the four native
# queries in stock-repeated-all/124008 are 683,745,683,745, each32/32.
# Distinct diagnostic names expose the two phases without changing their tones.
# Selection belongs to the session, not the
# synthesis generator, and must not be extrapolated as an increasing preadv.
SESSION_INTERMEDIATE_ANSWER_QUERIES = (
    replace(SESSION_FINAL_ANSWER_QUERY, name="session-intermediate-answer-query-a"),
    replace(SESSION_FINAL_ANSWER_QUERY, name="session-intermediate-answer-query-b", preadv=745),
)

# Stock BW2750 intermediate-ACK erasure, 2026-09-12: this called-keyed
# reply to SESSION_FINAL_ANSWER_QUERY retired exactly one host-delivered 89-byte
# frame, then the caller sent its next distinct DATA. KC9GHZ also keyed this
# reply on air after our query (26/26 clear payload tones). Recognition is
# limited to a fresh solicited full intermediate caller boundary; not a grant.
SESSION_INTERMEDIATE_QUERY_ANSWER = BurstKind(
    "session-intermediate-query-answer", (74,), 31, 288, 1, 1)

# Stock BW2750 double loss, 2026-09-19: after missing DATA and then the retry's
# ACK, the caller queries 60/745. This 288/249 response confirms the 22-byte
# retry and returns the short remainder to base. All31 payload tones match.
SESSION_RETRY_QUERY_ANSWER = BurstKind(
    "session-retry-query-answer", (74,), 31, 288, 249, 1)

# The responder's turn-request: what an answering station keys to become the
# sender, and the frame this tree spent eleven unanswered askings a session not
# holding. Same lattice position as the caller's request above (PREADV 1) and
# keyed to the CALLER exactly as it is; the SEED_OFF is the caller's plus 228,
# which is the offset that separates every responder frame here from its
# initiator counterpart (60/288, 850/1078).
#
# Measured off eleven bench sessions against a stock VARA HF 4.9.0 on
# 2026-08-26: it is keyed after the responder's host queues a payload and before
# it transmits, and again after its peer's over is acknowledged and it has a
# reply waiting — three keyings then a transmission when the ask is answered, and
# twelve keyings and no transmission at all in the three sessions where it was
# not. Its 31 payload tones pin one 24-bit state (0x6f753c, unique over an
# exhaustive search); of the seven (SEED_OFF, PREADV) pairs under 4096 that
# reach it from either callsign, this is the only one on the family's 62-draw
# lattice.
#
# A GATEWAY KEYED IT TOO, which is what makes this a frame rather than a bench
# artefact: KC9GHZ, off air on 40 m, at 15.88 s of the 2300 session — 30 of the
# 30 payload tones its own alignment finds comparable. The recording carries its
# own negative population, swept whole every half symbol: nothing else in it
# reaches 20 of 31 against this frame, and this station answered the one
# occurrence with silence.
SESSION_TURN_REQUEST_RESPONDER = BurstKind(
    "session-turn-request-responder", (74,), 31, 1078, 1, 1, keyed_by="caller")

# What the responder keys on its own cadence between finishing a DATA over and
# its peer's answer to it — the same slot SESSION_RESPONDER_IDLE was read in,
# one lattice position further on. Seventeen keyings across eleven bench
# sessions, 3.33-3.52 s apart, always following the responder's last over and
# always ending in SESSION_DRAINED_RESPONDER or the close.
#
# One callsign pair, so the lattice carries it as it does the frames above: the
# 31 tones pin 0xc0c7ba, and 288/745 is the only pair under 4096 on the 62-draw
# lattice — position 12, which no other member occupies. Nothing keys it here and
# nothing needs to: reading it is what keeps a responder's cadence out of the
# "not an MFSK handshake burst" log.
SESSION_RESPONDER_OVER_IDLE = BurstKind(
    "session-responder-over-idle", (74,), 31, 288, 745, 1)

# What a Winlink gateway keys back at an idling station that holds the turn, and
# the only frame in this file read off a GATEWAY's transmitter rather than an
# initiator's. It is what answered our turn-idle ten times across the two off-air
# sessions — 0.112-0.150 s after our burst ended, five times in each — and it
# answered nothing else in either recording.
#
# That it is the gateway transmitting is measured, not assumed. These recordings
# hold our own transmissions through the receiver's own mute, 20-27 dB down on
# everything else in them; this burst sits level with the gateway's connect-response
# at the top of that range, which is the one transmission in either session whose
# origin was never in doubt.
#
# Its 31 payload tones pin one 24-bit generator state out of 2**24 in each
# session, both regenerate 31/31, and (288, 1241) is the ONLY (SEED_OFF, PREADV)
# pair that produces both from their two different callsigns — the seeding map is
# many-to-one, so a pair derived from a single recording proves nothing and the
# cross-callsign intersection is the whole test. Keyed to the CALLED station, like
# every frame above it but the two turn frames.
#
# What it MEANS is not established, and the name says only where it was heard.
# What the recordings do settle: it is bit-identical at every one of the ten
# occurrences, so it carries no reason, count or field beyond its own identity;
# and it is not the answer to a turn-request, which is the frame a grant would
# answer. Each session's turn-request drew a gateway burst of its own, and
# neither is this frame at any alignment. The only responder transmit path we
# hold that runs long enough to test is a logged BW500 session — 479 s, 59
# keyings, answering keepalives, turn-requests and overs — and it answers all of
# them with the 11-symbol two-tone control burst and never with this.
SESSION_IDLE_RESPONSE = BurstKind(
    "session-idle-response", (74,), 31, 288, 1241, 1)

# What a gateway keys on its own cadence while IT holds the turn — the responder's
# counterpart of SESSION_TURN_IDLE, and the second frame here read off a gateway's
# transmitter rather than an initiator's.
#
# KE8LVA keyed it thirteen times in the 2026-08-23 04:57z session, 3.406 s apart to
# the 5 ms census resolution, from 53.371 s to 101.086 s of the recording: after its
# greeting over and our per-over response — which leaves the gateway the sender —
# and through our own keepalive cadence, which it neither answered nor waited for.
# Its slots fall inside and outside our turnarounds alike, so it runs on its own
# clock. Seven of the thirteen read all 32 symbols clear of the band; the other six
# lose a tone or two to it.
#
# Fitted, not captured, and from ONE recording and ONE callsign — which is below the
# two-corpus standard the frames above meet, so what stands in for the second
# callsign is stated here rather than assumed. The 31 payload tones pin one 24-bit
# generator state out of 2**24 (0xcb89c9, unique over an exhaustive search). Taking
# the four SEED_OFFs this family is already known to use — 60, 288, 850, 1550 — and
# discrete-logging that state from each of the two callsigns in the session gives
# eight PREADVs with no freedom left to fit: seven land in the millions, and the
# eighth is KE8LVA at 288, which lands on 683. Every 32-symbol frame above sits at
# PREADV = 1 (mod 62) — 62 draws is one frame's payload, so they are all positions
# on one generator stream — and 683 is position 11 on that lattice, which none of
# them occupies. A state falling on it by chance is 21 admissible values in 2**24.
# Read as keyed to the CALLER instead, none of KE8LVA's fourteen candidate pairs
# under PREADV 4096 is on the lattice at all, which is what settles `keyed_by` here
# without a second callsign.
#
# What it MEANS is not established and the name says only when it was heard. Nothing
# reads it: a gateway keying it thirteen times drew nothing from this station and
# the session ended in our own disconnect 39 s later, so no recording says what an
# answer to it would be.
SESSION_RESPONDER_IDLE = BurstKind(
    "session-responder-idle", (74,), 31, 288, 683, 1)

# What a station keys when its own send queue has drained — the frame the turn
# changes hands on. Measured, not inferred: two full bidirectional real-VARA <->
# real-VARA BW2300 sessions, 126 bytes each way, with different callsign pairs
# (W9SSJ/W1AW on 2026-08-14, K5ABC/N0DX on 2026-08-15), both cables recorded
# separately so which station keyed a burst is a property of the tape.
#
# In both sessions the initiator keys ITS frame exactly twice: once on the
# turnaround after its host reports BUFFER 0, and once more on the turnaround
# immediately before the peer starts transmitting. The two keyings are tone-
# identical, 31/31 — so this is not a grant answering a request, because the same
# frame goes out when nobody has asked for anything. What the recordings show
# around it is the peer's control burst arriving LATE: 0.811 s after the holder's
# burst ended, against 0.02-0.12 s on every other turnaround in the session, and
# the peer's first DATA over follows 0.029 s after the holder's next burst.
#
# The responder keys the counterpart when its own queue drains. The two are
# distinct frames, not one frame keyed to whoever sends it: read as sender-keyed
# or peer-keyed, no (SEED_OFF, PREADV) is consistent across the two sessions at
# all. Read as keyed to the CALLED station — as every session frame here is
# except the two turn frames — each is a singleton over both, where either
# session alone leaves 11 pairs standing. They share PREADV 807 and differ only
# in SEED_OFF, and it is the same 60/288 pair that separates the over-response
# from the idle-response above.
#
# What they MEAN beyond "this station has nothing more queued" is not
# established, and the names say only when they are keyed.
SESSION_DRAINED = BurstKind(
    "session-drained", (74,), 31, 60, 807, 1)
SESSION_DRAINED_RESPONDER = BurstKind(
    "session-drained-responder", (74,), 31, 288, 807, 1)

# THE RESPONDER'S TURN RELEASE: the answering station's counterpart of
# SESSION_TURN_RELEASE below, at that frame's own lattice position with the
# responder's SEED_OFF — 289 is to 61 what 288 is to 60 across this whole family.
#
# Read off two stock VARA HF 4.9.0 instances passing the turn to each other over
# a fake cable, 2026-08-26, one cable per direction so every burst is attributed
# by which recording holds it. The answering station keyed it four times across
# three sessions — 15.045, 15.080 and 26.955 s — always in the turnaround of the
# caller's answer to its own DATA over, and the caller's next over followed
# 0.07-0.10 s later. 17 of 17 tones at every occurrence, preamble included;
# nothing else in this file reaches 3 of 17 against it.
#
# It was called session-turn-refused until that capture, off the first request of
# two gateway sessions, and the name was wrong in the way a name can cost a
# session: what KE8LVA and KB3AC-10 keyed 0.18 s and 0.21 s after our first
# turn-request was the turn being handed over, and this station logged a refusal
# and asked again. Both gateways then answered the second request with
# SESSION_DRAINED_RESPONDER, which is the same state at 32 symbols — so the
# first ask was granted in both sessions and neither grant was acted on.
#
# The opening pair is measured here and was not before: our own receiver was
# muting our turn-request through it in both gateway recordings, and the tone
# that stood in its place was the one every other 16-symbol frame opens on. It is
# (62, 67), the caller's release's preamble, which is what makes this 17 symbols
# rather than 16.
#
# AND IT ARRIVES UNASKED, which the two gateway sessions above could not show
# because in both of them we had asked. Three arms of 2026-08-29 across two
# gateways keyed it 0.13-0.15 s behind our per-over control burst with no request
# of ours anywhere in the session: KB5LZK at 72.711 s and 98.702 s, N5TW at
# 110.730 s. So the release answers an over's acknowledgement, not a request, and
# a station that reads it only inside its own asking window reads none of these.
# All three greetings broke off mid-word at 89 bytes, which is the gateway
# handing over mid-message rather than a message cut short  [see
# tests/kestrel/test_unasked_handover.py].
SESSION_TURN_RELEASE_RESPONDER = BurstKind(
    "session-turn-release-responder", (62, 67), 15, 289, 391, 1)

# What a gateway keys back at an initiator's DATA over. ONE occurrence and one
# callsign, which is below the two-corpus standard the frames above meet, and the
# reason is that there is only one over to answer: KE8LVA keyed this 0.17 s after
# the last sample of the first payload over this station has ever put on the air
# (69.16 s of the 2026-08-26 12:59z recording), at 32 of 32 tones.
#
# The negative population stands in for the second occurrence. Swept whole against
# both callsigns, the four gateway sessions on disk — 750 s, two gateways, and no
# over of ours in three of them — reach 5 of 32 everywhere else, so the one window
# that takes it is the one turnaround that could hold it.
#
# Its 31 payload tones pin one 24-bit state (0x2fe3c5, unique over an exhaustive
# search), and what stands in for the second callsign is the same argument
# SESSION_RESPONDER_IDLE rests on. The four SEED_OFFs the 32-symbol frames use —
# 60, 288, 850, 1550 — discrete-logged from each of the session's two callsigns
# give eight pre-advances: seven land in the millions and the eighth is KE8LVA at
# 288, which lands on 311. Position 5 on the 62-draw lattice every 32-symbol frame
# here occupies, and a state falling on it by chance is 21 admissible values in
# 2**24. That is what settles `keyed_by`.
#
# What it MEANS is not established and the name says only when it was heard.
# Recorded here because the session that holds it is also the one it explains: the
# gateway keyed this once and then transmitted nothing for the remaining 110 s,
# through six keepalives and four disconnect-requests of ours, and the log called
# the whole of that "nothing back from KE8LVA".
SESSION_RESPONDER_OVER_ANSWER = BurstKind(
    "session-responder-over-answer", (74,), 31, 288, 311, 1)
# Independent stock evidence, 2026-09-12: after the fourth full89 reached its
# host but its short reply was erased, KC9GHZ-keyed288/311 answered the caller
# query, retired exactly89 and preceded final37 at record3. Both original and
# independently recorded caller input match32/32. Our BW2750 sender currently
# announces and keys a base-record close, so this lower-record query transition
# is not enabled here. Missing DATA instead produced288/1117 in the separate
# controlled negative experiment.

# THE TURN RELEASE: what a station keys once the peer has answered its DATA over
# and it has nothing more queued. Read off two stock VARA HF 4.9.0 instances
# passing the turn to each other over a fake cable, 2026-08-26, one cable per
# direction so every burst is attributed by which recording holds it. Neither end
# is ours, which is what makes this the release and not our own convention.
#
# The data phase is one shape repeated: a 4.34 s wideband over, the peer's 0.47 s
# two-tone control burst answering it, THIS BURST 0.13-0.14 s later, and the
# peer's own over 0.07-0.10 s after it ends. The caller keyed it at 21.050 s of
# the first session and 21.015 s of the third, 17 of 17 tones exact against the
# called station at these parameters, and in both the peer answered by
# transmitting the reply that had been sitting on its data port.
#
# THE RELEASE IS A TRANSMISSION, which is the whole of why it is here: the peer
# may key a DATA over only while it holds the turn [spec 05 §5.4], so a release
# that goes no further than a variable in the sending process leaves it listening
# until its inactivity timeout. That is what KE8LVA did on 2026-08-26 and what a
# bench VARA reproduced seven times.
#
# It was called session-disconnect-request until this capture, off a loopback
# session's close, and the name was wrong: both bench sessions ran on for two
# more overs after it. What a 4.9.0 keys at a host DISCONNECT is a 32-symbol
# frame this file does not hold — one burst at 32.943 s of the third session,
# followed by the CW ident, with no acknowledgement and no final answering it.
SESSION_TURN_RELEASE = BurstKind(
    "session-turn-release", (62, 67), 15, 61, 391, 1)

# The responder's counterpart is SESSION_TURN_RELEASE_RESPONDER above: the same
# preamble and the same lattice position, keyed with the responder's SEED_OFF.

# THE CLOSE. A host ``DISCONNECT`` at the caller put this on the cable once, at
# 32.947 s of the third bench session, immediately ahead of the CW ident — and
# nothing answered it but the peer's own ident. 31 of 31 payload tones at these
# parameters, keyed to the called station.
#
# One occurrence and one callsign, which is the standard SESSION_RESPONDER_OVER_ANSWER
# rests on, and the lattice is what carries it: every 32-symbol frame in this family
# sits at PREADV = 1 (mod 62) — 62 draws is one frame's payload — and 125 is position
# 2, which no other member occupies. SEED_OFF 60 is one the family already uses. A
# state landing on the lattice by chance is 21 admissible values in 2**24.
#
# It is 32 symbols, which is what settles the older reading being wrong: the close
# read off a 2026 loopback tape as a 17-symbol request, an acknowledgement and a
# 16-symbol final is three bursts this session does not hold, and the 17-symbol
# burst that reading was built on is the release above.
SESSION_DISCONNECT_REQ = BurstKind(
    "session-disconnect-request", (74,), 31, 60, 125, 1)

# The 16-symbol burst that same loopback close was read as ending on. Nothing
# answered the close above — on the 2026-08-26 bench, or against a stock 4.9.0 on
# 2026-09-02, where four re-keyed closes drew no key-up at all from the responder —
# so this station neither keys it nor waits for it. Kept as the reading it is, and
# as one of the names the bracket reader can put to a 16-symbol burst.
SESSION_DISCONNECT_FINAL = BurstKind(
    "session-disconnect-final", (62,), 15, 351, 151, 1)

# The turn-idle frame exactly as recorded off air on 2026-07-24, keyed to the
# recording station's own callsign. Kept as the arbiter the generator is graded
# against: a real VARA's own symbols, not a round trip through our own encoder.
SESSION_RESPONSE_2300 = (
    74, 86, 39, 80, 51, 54, 33, 42, 51, 98, 31, 50, 97, 64, 71, 96,
    39, 90, 41, 48, 59, 44, 81, 82, 51, 30, 63, 52, 61, 44, 69, 86)
SESSION_RESPONSE_2300_CALL = "W9SSJ"
#: Tones of :data:`SESSION_RESPONSE_2300` that the recordings actually resolve.
#: The station's receiver unmutes across the tail of its own transmission, so the
#: last symbol reads a different carrier in every one of the twelve captures and
#: is not evidence of anything.
SESSION_RESPONSE_2300_RESOLVED = 31

BURSTS = {b.name: b for b in (CR, CR500, CR2750, CONNECT_RESPONSE,
                              CONNECT_RESPONSE_500, CONNECT_RESPONSE_2750,
                              SESSION_CONFIRM, SESSION_KEEPALIVE_A,
                              SESSION_KEEPALIVE_B, SESSION_OVER_RESPONSE,
                              SESSION_OVER_RESPONSE_SHORT, SESSION_OVER_NAK,
                              SESSION_TURN_REQUEST, SESSION_TURN_IDLE,
                              SESSION_FINAL_ANSWER_QUERY,
                              *SESSION_INTERMEDIATE_ANSWER_QUERIES,
                              SESSION_INTERMEDIATE_QUERY_ANSWER, SESSION_RETRY_QUERY_ANSWER,
                              SESSION_IDLE_RESPONSE, SESSION_RESPONDER_IDLE,
                              SESSION_DRAINED, SESSION_DRAINED_RESPONDER,
                              SESSION_TURN_RELEASE_RESPONDER,
                              SESSION_TURN_REQUEST_RESPONDER,
                              SESSION_RESPONDER_OVER_IDLE,
                              SESSION_RESPONDER_OVER_ANSWER,
                              SESSION_OVER_NAK_RESPONDER,
                              SESSION_CONNECT_CONFIRM,
                              SESSION_DATA_NAK_SHORT, SESSION_DATA_NAK_QUERY)}


def normalize_callsign(callsign: str) -> str:
    """ASCII, uppercase, including any SSID characters  [spec 04 §4.2.3].

    The generator is keyed to the destination callsign string exactly as
    dialled (e.g. ``"W1AW-7"`` keeps the ``-7``). Only ASCII uppercasing is
    applied here; the spec does not define further canonicalisation, so none is
    performed  [ours: minimal normalization, spec silent on '-'/SSID splitting
    for the tone key].
    """
    return callsign.upper()


# --------------------------------------------------------------------------- #
# Payload-tone generator (closed form)  [spec 04 §4.2.3].
def payload_bins(callsign: str, kind: BurstKind) -> list[int]:
    """The ``N`` callsign-keyed payload carrier indices for ``kind``."""
    return list(_payload_bins_cached(normalize_callsign(callsign), kind))


@lru_cache(maxsize=4096)
def _payload_bins_cached(cs: str, kind: BurstKind) -> tuple[int, ...]:
    """Memoised core: a recogniser scores every candidate callsign against every
    burst, so the same (callsign, kind) recurs constantly."""
    ascii_bytes = cs.encode("ascii")
    crc = crc16_genibus(ascii_bytes)
    seed = (crc + kind.seed_off) & 0x7FFF
    # NB: the "+50" carry is fixed for ALL three bursts  [spec 04 §4.2.3].
    mult = _g_hash(cs) + ((crc + 50) >> 15)

    s = _lcg_advance(_start(seed), mult + kind.preadv)

    alpha = kind.tones
    bins: list[int] = []
    for k in range(kind.n_payload):
        s = _lcg(s)
        P = (s * alpha.n_p) >> 24                  # floor((s/2**24)*n_p)
        s = _lcg(s)
        D = (s * 7) >> 24                          # floor((s/2**24)*7), D in {0..6}
        tone = alpha.base + alpha.parity(k, kind.par0) + 14 * P + 2 * D
        bins.append(tone & 0xFF)
    return tuple(bins)


TONE_ALPHABET = BW2300_TONES.carriers
"""Every carrier index the BW2300 payload generator can emit  [spec 04 §4.2.3]."""


# The draw is ``x -> MULT*x + ADD`` on 24 bits and MULT is odd, so it is a
# bijection and one step runs backwards as cheaply as it runs forwards.
_LCG_MULT_INV = pow(_LCG_MULT, -1, 1 << 24)


def _lcg_back(s: int, n: int) -> int:
    """State ``n`` draws BEFORE ``s``  (exactly the inverse of _lcg_advance)."""
    a, c = _lcg_jump_coeffs(n)
    return (pow(a, -1, 1 << 24) * (s - c)) & _LCG_MASK


def payload_states(received: Sequence[int], kind: BurstKind) -> tuple[int, ...]:
    """Every generator state that emits ``received`` — :func:`payload_bins` run
    backwards, over the whole callsign space at once.

    ``received`` is a demodulated PAYLOAD tone sequence with ``-1`` where the
    receiver delivered nothing comparable, exactly as :func:`recognize` takes it.
    Each returned value is the state the generator stands at when the payload's
    first draw is taken — ``_lcg_advance(_start(seed), mult + PREADV)`` — so it is
    directly comparable with what :func:`_payload_bins_cached` starts from.

    THE CALLSIGN ENTERS ONLY THROUGH THAT STATE. Every other step of the
    generator is fixed by the burst kind, so a state that emits the tones says
    the burst IS this family's payload — addressed to somebody — where matching
    a callsign says which somebody. That is the difference between reading a
    connect-response and naming the station that keyed it, and off air they come
    apart: on 2026-08-26 a burst answered this station's second call to KB3AC-10
    with fourteen payload tones that pin exactly one of the 2**24 states and
    reproduce no callsign on the published panel.

    It is also the rejection the tones alone do not carry. A payload symbol draws
    from 35 of the alphabet's 70 carriers — parity is fixed by the symbol index —
    so eight comparable tones are one of 35**8 sequences against 2**24 states,
    and an arbitrary eight land on a state 7e-6 of the time. Fewer than that and
    the answer stops meaning anything: two comparable tones admit thousands.

    Empty means the tones are not this generator's output at all, which includes
    every tone that is off the alphabet's own parity lattice.
    """
    alpha = kind.tones
    draws = []
    for k, tone in enumerate(received[:kind.n_payload]):
        if tone < 0:
            draws.append(None)
            continue
        v = tone - alpha.base - alpha.parity(k, kind.par0)
        if v < 0 or v % 2 or v // 14 >= alpha.n_p or (v % 14) // 2 > 6:
            return ()
        draws.append((v // 14, (v % 14) // 2))
    first = next((k for k, d in enumerate(draws) if d is not None), None)
    if first is None:
        return ()

    # The first comparable symbol bounds its own P draw to one n_p-th of the state
    # space, which is what makes this a filter over a few million candidates
    # instead of a sweep of 2**24. Everything behind it cuts by 35 a symbol.
    P0 = draws[first][0]
    entry = np.arange(-(-(P0 << 24) // alpha.n_p),
                      -(-((P0 + 1) << 24) // alpha.n_p), dtype=np.int64)
    cur = entry.copy()
    for k in range(first, len(draws)):
        if k > first:
            cur = (cur * _LCG_MULT + _LCG_ADD) & _LCG_MASK
        d = draws[k]
        if d is not None:
            keep = ((cur * alpha.n_p) >> 24) == d[0]
            cur, entry = cur[keep], entry[keep]
        cur = (cur * _LCG_MULT + _LCG_ADD) & _LCG_MASK
        if d is not None:
            keep = ((cur * 7) >> 24) == d[1]
            cur, entry = cur[keep], entry[keep]
        if not len(cur):
            return ()
    # `entry` holds the P draw of symbol `first`, which is that many draws into
    # the payload; wind back to the state the payload itself starts from.
    return tuple(sorted(_lcg_back(int(s), 2 * first + 1) for s in entry))


def lattice_step(kind: BurstKind) -> int:
    """Draws one payload of ``kind`` consumes: two per symbol  [spec 04 §4.2.3].

    Every ``PREADV`` in this module is ``1`` mod this number for its own ``N``,
    because the pre-advance is a position on the family's own lattice and 1 is
    where the lattice starts."""
    return 2 * kind.n_payload


def payload_position(received: Sequence[int], callsign: str, kind: BurstKind,
                     span: int = 64) -> int | None:
    """Which frame of ``callsign``'s own stream ``received`` is, or ``None``.

    A family's frames all run off one stream: the callsign seeds it, and each frame
    starts one whole payload further along than the one before, which is why every
    ``PREADV`` here is ``1`` mod :func:`lattice_step`. ``kind.preadv`` names one
    position on that lattice, and this asks which position the tones are at.

    THE DIFFERENCE IT MAKES IS WHETHER A BURST NAMES A STATION. Regenerating the
    tones for a callsign asks whether the peer sent THIS frame, so a peer that
    answered with a neighbouring one scores at chance and its answer reads as an
    answer from nobody. That is how every burst in ``ONAIR_PEER_ANSWERS`` was read
    before this existed, on four attempts across two bands and three gateways.
    """
    states = payload_states(received, kind)
    if not states:
        return None
    cs = normalize_callsign(callsign)
    crc = crc16_genibus(cs.encode("ascii"))
    s = _lcg_advance(_start((crc + kind.seed_off) & 0x7FFF),
                     _g_hash(cs) + ((crc + 50) >> 15) + 1)
    step = lattice_step(kind)
    for position in range(span):
        if s in states:
            return position
        s = _lcg_advance(s, step)
    return None


def handshake_tones(callsign: str, kind: BurstKind) -> list[int]:
    """Full burst carrier sequence: ``preamble ++ payload_bins``  [spec 04
    §4.2.3, last line]. Rendered to audio by :mod:`vara_mfsk` per §4.2.1."""
    return list(kind.preamble) + payload_bins(callsign, kind)


# --------------------------------------------------------------------------- #
# Recognizers  [spec 04 §4.2.4].
# "an initiator that dialed <called> regenerates the expected payload bins (same
#  closed form) and matches them against the demodulated response to confirm
#  step 2." — RX use, spec 04 §4.2.4.

# Acceptance threshold. Clean recognisers score correct = N/N and wrong <= 2/N
# [spec 04 §4.2.4], and 0.8 sits between the two.
#
# This was briefly lowered to 0.4 on the belief that a real HF channel costs a
# genuine answer several tones. It does not, and the evidence said so: measured
# against the off-air recordings, a real gateway connect-response scores 15/15 and
# a connect-request 29/31 once the burst is correctly located. The 11/15 that
# justified the change came from a timing search that returned the earliest
# acceptable alignment rather than the best — a defect in the locator, not fading.
#
# Restored, because lowering it hid the symptom that would have exposed that
# defect: at 0.8 a misaligned lock is rejected and visible, at 0.4 it is accepted
# and silent. The false-accept rate matters less than that: a wrong callsign
# reaching 0.4 is ~2.4e-6 for the 15-tone response and ~1.2e-5 for the 12-tone ack,
# against ~2.4e-7 at 0.8 — the per-tone collision is 1/35, not 1/70, because parity
# is fixed by symbol index and splits the alphabet into two disjoint halves.
#
# The BW500 handshake draws from 14 carriers, so its per-tone collision is 1/14 and
# its wrong-callsign population sits at 7/15 where BW2300's sits at 5/15 (measured
# over 36,315 callsign pairs, see tests/kestrel/test_handshake_offair). 0.8 still
# clears that by more than it clears the worst genuine answer, which is the test
# the cut has to pass — but the margin at BW500 is half what it is wide.
_ACCEPT_FRAC = 0.8                                 # [ours — spec gives the gap, not the cut]

#: Payload tones that have to be comparable at all before a fraction of them means
#: anything. A negative entry in a received tone sequence is a symbol the receiver
#: never delivered — its analysis window fell outside the audio, or the rig was
#: still muting its own transmission — and scoring one as a miss throws away real
#: answers: on 2026-08-06 a gateway's connect-response arrived with six of its
#: fifteen payload tones inside this station's post-transmit mute and the other
#: nine all correct. Eight, because a clean sweep of eight is 35**-8 per alignment,
#: stricter per tone than the two misses 12-of-15 already tolerates.
MIN_COMPARABLE = 8                                 # [ours]


def payload_match(received_payload: Sequence[int], callsign: str,
                  kind: BurstKind) -> tuple[int, int]:
    """Return (matched_bins, N) comparing a demodulated PAYLOAD tone sequence
    against the expected bins for ``callsign``  [spec 04 §4.2.4]."""
    expected = payload_bins(callsign, kind)
    n = len(expected)
    m = sum(1 for a, b in zip(received_payload, expected) if a == b)
    return m, n


def _strip_preamble(received_tones: Sequence[int], kind: BurstKind) -> Sequence[int]:
    """Given a full received tone sequence, return the payload region."""
    if len(received_tones) >= len(kind.preamble) + kind.n_payload:
        return received_tones[len(kind.preamble):len(kind.preamble) + kind.n_payload]
    # Fall back: assume the sequence is already payload-only.
    return received_tones


def recognize(received_tones: Sequence[int], callsign: str, kind: BurstKind,
              accept_frac: float = _ACCEPT_FRAC) -> bool:
    """True iff a received burst's tones match ``handshake_tones(callsign,kind)``.

    Accepts either a full (preamble+payload) tone sequence or a payload-only
    sequence. Uses the payload region for the match  [spec 04 §4.2.4]. A negative
    tone is a symbol the receiver never delivered and is not comparable; the
    fraction is taken over the tones that were, and at least
    :data:`MIN_COMPARABLE` of them must be."""
    payload = _strip_preamble(received_tones, kind)
    expected = payload_bins(callsign, kind)
    pairs = [(a, b) for a, b in zip(payload, expected) if a >= 0]
    if len(pairs) < min(kind.n_payload, MIN_COMPARABLE):
        return False
    m = sum(1 for a, b in pairs if a == b)
    return m >= accept_frac * len(pairs)


def best_match(received_tones: Sequence[int], callsigns: Sequence[str],
               kind: BurstKind) -> tuple[str, int, int]:
    """Over a set of candidate callsigns, return (best_call, matched, N).

    Used e.g. by a listening responder that knows its own MYCALL(s) to decide
    whether an inbound CR is addressed to it  [spec 04 §4.2.4]."""
    payload = _strip_preamble(received_tones, kind)
    best = ("", -1, kind.n_payload)
    for cs in callsigns:
        m, n = payload_match(payload, cs, kind)
        if m > best[1]:
            best = (cs, m, n)
    return best
