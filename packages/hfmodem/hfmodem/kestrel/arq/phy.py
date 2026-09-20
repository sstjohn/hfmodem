# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PHY adapter: one ARQ frame <-> one synthesised VARA-HF data burst.

The ARQ layer is transport-agnostic; this module is the single place it touches
the physical codecs. It dispatches on the connection **bandwidth**:

    * ``bw="500"``  -> the proven BW500 Level-4 codec (``kestrel/tx/varahf500_tx``,
      ``kestrel/rx/varahf500``). Unchanged from the validated BW500 milestone.
    * ``bw="2300"`` -> the BW2300 wideband OFDM codec (``kestrel/tx/varahf2300_tx``,
      ``kestrel/rx/varahf2300``), with a selectable **speed level** for the
      per-over gear-shift (base index-mod rec3, or the high-throughput ladder).

Frame container: the ARQ marker byte (control-type / seq / last-of-over) is byte 0
of the codec payload, uniform across bandwidths and speed levels. BW500 keeps its
native marker-as-frame-byte layout (46-byte frame); BW2300 carries the marker as
payload[0] of the record's payload (zero-padded up to the level's capacity), so
one wiring covers every level. The receiver, which is not told the speed level
out of band, CRC-confirms a trial decode over the length-compatible levels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coding.crc import crc16_genibus
from ..rx import varahf500 as _rx500
from ..rx import varahf2300 as _rx2300
from ..tx import varahf500_tx as _tx500
from ..tx import varahf2300_tx as _tx2300
from ..vara import vara_control as _vc
from ..vara import vara_frames as _vf
from ..vara import vara_mfsk as _mfsk

ONSET = 3000  # BW500 column-0 centre sample index (head-room for the demod pad)

# VARA's real BW500 control channel is a set of short fixed DBPSK tokens
# (spec 02 §2.6). A native kestrel control frame rides the ~4.3 s DATA waveform,
# so burst length alone separates the two on receive.
CONTROL_TOKEN_MAX_SAMPLES = 40000       # ~0.83 s; DATA/native-control overs are ~4.3 s

#: The one short control burst BW2300 has. At BW500 the vocabulary is four fixed
#: DBPSK tokens; at BW2300 it is a single index-modulated waveform — eleven
#: two-tone MFSK symbols on a fixed four-symbol preamble, 0.489 s
#: [spec 04 §4.2C] — and the responder's CONNECT-time burst and its first per-over
#: answer are that one waveform to correlation 1.00. So the bandwidth has one name
#: here where BW500 has four, and which member of the family it is, is the
#: receiver's state and not the audio.
ANSWER_2300 = "answer"


def render_token(name: str, bw: str = "500") -> np.ndarray:
    """Render a VARA control token to audio.

    BW2300 has one, whatever it is called: the seven state symbols behind the
    fixed preamble are not reversed, so what goes out is one captured frame —
    the same one ``vara_arq._tx_connected_ack`` keys.
    """
    if _wideband(bw):
        return _mfsk.synth_tone_pairs(_vf.CONNECTED_ACK_2300)
    return _vc.synth_token(name)


def detect_token(audio: np.ndarray, bw: str = "500"):
    """Classify a short received burst as a VARA control token, or None.

    At BW2300 the classifier is the preamble reader, not the DBPSK table: no real
    VARA has been recorded keying that table, and on the one BW2300 recording held
    with a real VARA at both ends it matches none of the thirteen short control
    bursts while the preamble reader takes all eleven acknowledgements among them
    at 55-60 alignments against a threshold of 3 [see vara_control].
    """
    x = np.asarray(audio, float)
    if _wideband(bw):
        from ..vara import vara_arq as _va       # session engine; imports this module
        hold = _va._ack_plateau(x)
        return (_vc.TokenMatch(ANSWER_2300, 0, float(hold))
                if hold >= _va._ACK_PLATEAU else None)
    return _vc.detect_token(x)


def _wideband(bw) -> bool:
    """Whether ``bw``'s DATA over is a record of the BW2300 family.

    BW2750 is: its base level is record 3's index law on a 20-bin comb, same frame
    length, same coding chain, and the MFSK control frames are bandwidth-neutral."""
    return str(bw) in _rx2300.BASE_LEVELS


def base_level(bw) -> int:
    """The robust base speed level for ``bw`` (the ARQ handshake/keepalive level)."""
    return _rx2300.BASE_LEVELS[str(bw)] if _wideband(bw) else 4


def speed_ladder(bw) -> list:
    """Ascending-throughput speed levels available at ``bw`` (for the gear-shift).

    The gear-shift can only reach levels this station can key, which need not be
    every level it can read, and it may not go BELOW the base one: the ARQ block
    is sized at the base level's capacity  [see :func:`payload_size`], so record 2
    — keyable since its training preamble came off the tape, and the step a real
    VARA pair drops to — cannot carry one. Its frame holds 48 bytes where a block
    is 89, and rendering one into it loses 42 bytes an over, on an over whose CRC
    is clean at the far end.

    BW2750 has one rung: no record of that bandwidth but its base level is
    reversed, so there is nowhere to shift to."""
    if str(bw) == "2300":
        return [lv for lv in _rx2300.KEYABLE_LEVELS if lv >= _rx2300.BASE_LEVEL]
    return [base_level(bw)] if _wideband(bw) else [4]


def payload_size(bw) -> int:
    """ARQ block payload bytes per frame at ``bw`` (marker byte reserved separately
    for the BW2300 family; BW500 keeps its native 43-byte payload + separate
    marker byte)."""
    if _wideband(bw):
        return _rx2300.payload_bytes(base_level(bw)) - 1         # 90 - 1 marker = 89
    return _rx500.FRAME_PAYLOAD                                   # 43


def body_size(bw, level: int | None = None) -> int:
    """Whole DATA body bytes at ``bw``: the payload plus the per-frame field
    behind it, and what :func:`vara_body` and :func:`vara_payload` lay out over.

    90 at BW2300 and 44 at BW500, where the field is the byte
    ``rx.varahf500.FrameResult.marker`` reads. The frame's own CRC sits outside
    it at both  [see :func:`vara.vara_ofdm.data_over_tx`]. ``level`` names a
    record other than the bandwidth's base one — 48 at BW2300's record 2 and 35
    at BW500's, which is where :func:`close_level` sends the over that ends a
    delivery."""
    if level is None or level == base_level(bw):
        return payload_size(bw) + 1
    if _wideband(bw):
        return _rx2300.payload_bytes(level)
    return {3: 44, 2: 35, 1: 23, 0: 10}[level]


def lower_level(bw, level: int, steps: int = 1) -> int:
    """Drop supported DATA records, stopping at that bandwidth's lowest record.

    BW500's public base is host level 4 (3 is its old alias), but lower ARQ
    records are numbered 2, 1, 0. The PHY codec itself uses host levels 3, 2, 1.
    """
    if str(bw) == "500":
        levels = (4, 2, 1, 0)
        return levels[min(levels.index(4 if level == 3 else level) + steps, 3)]
    return max(100 if str(bw) == "2750" else 0, level - steps)


def close_level(bw) -> int:
    """The record the over that CLOSES a delivery is keyed at.

    A stock 4.9.0 drops a speed level for it and goes straight back up for the
    next delivery's first over, so the record is the sender's statement about
    the block and not about the channel. It goes with the per-frame field the
    full over in front announced  [see ``vara_arq._frame_field``]: across eight
    deliveries on tape — six sessions, both stations, both directions — the six
    behind a full over carrying ``0x81`` close at record 2 and the two behind
    ``0x89`` close at the base level. ``0x89`` is not a value this station keys,
    so its own close has one record to go to. The byte is not the whole of it: a
    stock responder closed at the base level behind ``0x81`` three times in a row
    to this station, and was read 93 of 93 each time — what the record buys is
    being read, and record 2 is the one a responder reads from us.

    That leaves the pair we keyed until this was read — ``0x81`` and a base-level
    close — one no recording holds, and a responder handed it answers with the
    continue burst and waits for a message that has already ended. The body is
    not the difference: our close is byte-identical to a stock station's at both
    records.

    Record 2's body is 48 bytes against the base level's 90, so a close longer
    than 47 payload bytes does not fit one and stays where it is; no recording
    holds a stock station in that case  [see ``_tx_data_over``].

    BW500 closes the same way: a stock pair's 200-byte delivery is four level-4
    bodies behind ``95 91 8d 81`` and then a record-2 body of 35, the record the
    narrow renderer keys as ``level=2``  [see rx.varahf500.ROBUST_LEVEL]. BW2750
    has no record below its base one and stays there."""
    return 2 if str(bw) in ("2300", "500") else base_level(bw)


# --------------------------------------------------------------------------- #
# VARA-compatible DATA over framing  [spec 04 §4.2B, confirmed off air].
# VARA puts payload at body[0] and uses all 90 bytes; kestrel's own framing
# reserves body[0] for an in-band control marker. The marker is unnecessary when
# speaking VARA, whose control rides the separate DBPSK tokens, and a gateway will
# not parse an over that carries it.
_VARA_BODY = 90
#: What ONE base BW2300 over carries, not what a delivery may total: a stock pair
#: handed 600 bytes steps its host reads 89/178/267/356/445/534/600, and a BW500
#: pair steps by 43. Nothing anywhere in the modem caps a session.
_VARA_OVER_PAYLOAD = 89        # body[89] is a per-frame field, never payload
_VARA_TRAILER = 0x14

#: What a VARA HF 4.9.0 writes at ``body[88:90]`` to close a SHORT over. Measured,
#: not fitted: the two-VARA bench of 2026-08-26 rendered the same 65-byte login
#: block this station sends, and its over is byte-identical to ours through
#: body[87] and carries ``04 82`` where we carried ``00 00``. Every short block in
#: those three sessions holds it — 2, 3, 50 and 65 payload bytes, both stations,
#: both directions — as does an RMS Trimode 1.4.2.0 short block off 80 m, and as
#: does the link-setup over, which ``vara_frames.link_setup_frame`` has always
#: ended this way. It is the whole difference the burst carries: rendered against
#: the recorded one our over correlates 0.973 ending ``04 82`` and 0.597 ending
#: ``00 00``, which is why every check on envelope, duration, band and symbol
#: count passed while the peer waited for an over that had not ended.
_VARA_SHORT_END = (0x04, 0x82)


def vara_body(payload: bytes, callsign: str, tail: int = 0,
              body_len: int = _VARA_BODY) -> bytes:
    """One VARA DATA body: payload at offset 0, trailer only if short.

    ``body_len`` is the level's whole body  [see :func:`body_size`], and the
    default is the BW2300 base level's 90. Payload is capped one short of it —
    the last byte is a per-frame field, not payload. Reading it as payload
    corrupts the join between overs (observed off air: a full over ending "…5.0"
    carries 0x09 there, and another carries 0x51, neither of which appear in the
    delivered text).

    The layout is one law at every bandwidth and level, measured on a stock
    4.9.0 BW500 pair (2026-08-30, W9SSJ → W1AW, 200 bytes responder to caller):
    four full 44-byte bodies of 43 payload plus the field, then a record-2 body
    of 35 carrying 28 payload bytes and ``14 f6 00 00 00 04 82`` — 0x14, the
    caller's callsign CRC high byte, zero fill, and :data:`_VARA_SHORT_END` in
    the last two, which is the 90-byte form laid out over 35.

    A short over ends on :data:`_VARA_SHORT_END`, which is what closes it: a real
    VARA answers an over that carries it with one 0.470 s control burst and takes
    the turn, and answers the same over ending ``00 00`` by acknowledging twice and
    then waiting for the rest of a message that never comes.

    The full form has no such constant — the field moves per over on every real
    station held, 0x95/0x91/0x8d/0x81 across those four BW500 overs — so ``tail``
    is still passed through rather than synthesised. The two ends meet two bytes
    short of the payload cap, where the trailer runs into the field and takes it:
    the delimiter is what the peer needs to find the payload's end, and no real
    station has been recorded keying a short block that long.
    """
    cap = body_len - 1
    pl = bytes(payload[:cap])
    if len(pl) == cap:
        return pl + bytes([tail & 0xFF])
    body = bytearray(body_len)
    body[-2:] = _VARA_SHORT_END
    body[:len(pl)] = pl
    body[len(pl)] = _VARA_TRAILER
    body[len(pl) + 1] = crc16_genibus(callsign.upper().encode()) >> 8
    return bytes(body)


def vara_payload(body: bytes, length: int | None = None,
                 caller: str | None = None, body_len: int = _VARA_BODY) -> bytes:
    """Payload carried by a VARA DATA body (inverse of :func:`vara_body`).

    Pass ``length`` when the ARQ layer knows it. With ``caller``, the payload
    end is found by the whole trailer rather than by its first byte: 0x14, then
    the high byte of the caller callsign's CRC-16/GENIBUS, then zeros up to the
    per-frame field at body[88]. Both recorded gateway sessions carry exactly
    that pattern on their short blocks — 0xF6 after the 0x14, the CRC byte of
    W9SSJ, the *caller* — and a full block carries no trailer at all, so a body
    with no such pattern is a full 89 bytes of payload. A stray 0x14 inside
    compressed payload fails the pattern unless the payload itself ends in the
    whole trailer, which is where the ambiguity genuinely lives — the real
    length field (the body's last byte but one is the unreversed candidate) would
    close it.

    ``body_len`` is the speed level's whole body, and the default is the base
    level's 90. The trailer is laid out from the END of it, so a peer that drops
    to record 2 sends the same pattern inside 48 bytes — 0x14, 0xF6, zeros, then
    ``\\x04\\x82`` in the last two — and reading that body against 90 finds no
    trailer at all and hands eleven bytes of it to the host as mail. Anything
    shorter than its record's body is a truncation rather than a level, which is
    why the default pads rather than measures.

    Without ``caller`` the end is inferred from the first 0x14 alone, which is
    wrong for any binary payload containing one: kept for the call sites that
    predate the pattern and read text-only material.
    """
    b = bytes(body[:body_len - 1])
    if length is not None:
        return b[:length]
    if caller is None:
        i = b.find(bytes([_VARA_TRAILER]))
        return b if i < 0 else b[:i]
    full = bytes(body).ljust(body_len, b"\0")
    key = crc16_genibus(caller.upper().encode()) >> 8
    # Full fields end in binary 01. For callers such as W9SSJ (key=f6),
    # neither the canonical short end 82 nor an overlapping short CRC can
    # have that form. A binary payload ending 14 f6 is still full DATA.
    # Do not extend this discriminator to a caller whose CRC shares the
    # full-field form: its historical overlapping short form is ambiguous.
    if full[-1] & 3 == 1 and key & 3 != 1:
        return b
    for i in range(body_len - 1):
        if (full[i] == _VARA_TRAILER and full[i + 1] == key
                and not any(full[i + 2:body_len - 2])):
            return b[:i]
    return b


def over_is_last(body: bytes, caller: str) -> bool:
    """Does this DATA over end the delivery, or is another one behind it?

    The over says so itself, in the same trailer :func:`vara_body` writes and
    :func:`vara_payload` reads: a body filled to capacity carries no trailer and
    another over follows it, and a short body carries the trailer that closes the
    delivery. A sender with an exact multiple of a body to move keys one more over
    carrying no payload at all rather than ending on a full one — measured at the
    bench, where a station handed exactly one over's worth keyed two overs, the
    second delivering nothing to its peer's host.

    Which is why this is the over's own property and not the receiver's: at that
    second over's predecessor the receiving station already held every byte of the
    message and still answered *keep sending*  [see
    ``VaraStationHandshake._answer_data_over``].
    """
    return len(vara_payload(body, caller=caller,
                            body_len=len(body))) < len(body) - 1


#: Where :func:`overs_after` saturates. A sender with more than this still to
#: come holds the field at ``0x99`` and does not move it: the four-message bench
#: arms of 2026-09-09 carried ``0x99`` on overs with 23 and with 18 to go, as did
#: all seven post-`FS` overs of the 2026-09-08 KE8LVA delivery.
OVERS_AFTER_MAX = 4


def over_field(body: bytes) -> int:
    """The per-frame field this DATA body carries, which is its last byte
    [see :func:`vara_body`]."""
    return body[-1]


def overs_after(field: int, first: bool = False) -> int | None:
    """Overs the sender says are still to come behind the one carrying ``field``,
    or ``None`` where the byte cannot be placed.

    Measured off a stock VARA HF 4.9.0 responder on the cables, 2026-09-09,
    against the ``BUFFER n`` it reports at each key-up: a 524-byte delivery keyed
    as five full overs and one short read ``0x95 0x99 0x95 0x91 0x89 0x82``, and
    the countdown behind the first is ``0x99 0x95 0x91 0x89``. It is monotone
    non-increasing and it saturates.

    TWO BYTES SIT AT ZERO and the difference between them is the record the
    closing over comes at, not a count: ``0x81`` announces a close at record 2
    and ``0x89`` one at the base level, across eight deliveries on tape
    [see :func:`close_level`].

    TWO THINGS DO NOT FIT, and both return ``None`` rather than a number. The
    FIRST over of a delivery is not on the ladder at all — three arms gave
    ``0x95``, ``0x99`` and ``0x19`` for it, the last with bit 7 clear on a
    CRC-clean over — which is what ``first`` says. And the step at the bottom is
    8 rather than 4: ``0x8d`` appears on no capture, so 1 is an interpolation
    between two measured neighbours rather than a reading.

    Advisory. Nothing decides on this. It is read for the log, so that a delivery which
    stops early leaves in the transcript whether its end was near.
    """
    if first:
        return None
    if field in (0x81, 0x89):
        return 0                    # the last full one; the two differ in the
    n, rem = divmod(field - 0x89, 4)    # record the close behind it comes at
    return n if not rem and 1 <= n <= OVERS_AFTER_MAX else None


@dataclass
class DecodedFrame:
    """Bandwidth-neutral decode result (duck-typed like the BW500 FrameResult)."""
    crc_ok: bool
    marker: int
    payload: bytes


# --------------------------------------------------------------------------- #
def render(payload: bytes, marker: int, bw="500", level=None) -> np.ndarray:
    """(block payload, marker) -> one real 48 kHz transmit burst at ``bw``/``level``."""
    if _wideband(bw):
        lv = base_level(bw) if level is None else level
        cap = _rx2300.payload_bytes(lv)
        block = bytes([marker & 0xFF]) + bytes(payload)
        block = block + bytes(max(0, cap - len(block)))          # pad to capacity
        return _tx2300.synth_burst(block[:cap], lv)
    frame = _tx500.build_frame(payload, marker)
    return _tx500.synth_burst(frame, onset=ONSET)


def decode(audio: np.ndarray, bw="500") -> DecodedFrame:
    """Decode one burst rendered by :func:`render` into a bandwidth-neutral frame."""
    if _wideband(bw):
        # BW2300 does not signal the level, so it is trialled; BW2750 holds one
        # reversed record and there is nothing to trial.
        lv = None if str(bw) == "2300" else base_level(bw)
        fr = _rx2300.decode_burst(np.asarray(audio, float), lv)
        pl = fr.payload
        block_bytes = payload_size(bw)
        marker = pl[0] if pl else 0
        data = bytes(pl[1:1 + block_bytes])
        data = data + bytes(max(0, block_bytes - len(data)))
        return DecodedFrame(crc_ok=fr.crc_ok, marker=marker, payload=data)
    fr = _rx500.decode_burst(np.asarray(audio, float), start=ONSET)
    return DecodedFrame(crc_ok=fr.crc_ok, marker=fr.marker, payload=fr.payload)


# ---- legacy BW500 helpers (kept for the existing BW500 tests) ---------------
def render_frame(frame_bytes: bytes) -> np.ndarray:
    """46-byte proven BW500 frame -> transmit audio (one burst)."""
    return _tx500.synth_burst(frame_bytes, onset=ONSET)


def decode_frame(audio: np.ndarray):
    """Decode one BW500 burst rendered by :func:`render_frame` -> proven FrameResult."""
    return _rx500.decode_burst(np.asarray(audio, float), start=ONSET)


def build_frame(payload43: bytes, marker: int) -> bytes:
    """43 payload bytes + marker + CRC-16/GENIBUS -> 46-byte proven BW500 frame."""
    return _tx500.build_frame(payload43, marker)
