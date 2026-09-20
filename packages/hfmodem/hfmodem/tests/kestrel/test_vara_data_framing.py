# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VARA DATA over framing, against frames decoded off air  [spec 04 §4.2B].

The bodies below are real Winlink RMS gateway overs (NS0A, KC9GHZ, 40 m BW2300),
decoded CRC-clean by kestrel and cross-checked against the plaintext VARA delivered
on its host port. They pin the layout kestrel must emit to be parsed by a gateway:
payload at offset 0 using all 90 bytes, and a 0x14 trailer only when short.
"""
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara.vara_frames import crc16_genibus

# 90-byte bodies (the 2-byte CRC-16/GENIBUS trailer is stripped).
_NS0A_SHORT = bytes.fromhex(
    "35206d696e757465732e0d5b574c324b2d352e302d4232465749484a4d245d0d3b50513a"
    "2031373730323231380d434d5320766961204e533041203e0d14f6" + "00" * 25 + "0102")
_KC9_FULL = bytes.fromhex(
    "524d53205472696d6f646520312e342e322e300d0a573953534a20686173203131302064"
    "61696c79206d696e757465732072656d61696e696e672077697468204b433947485a2028"
    "454e3632424b290d5b574c324b2d352e3009")
_KC9_SHORT = bytes.fromhex(
    "2d4232465749484a4d245d0d3b50513a2035353733343731330d434d5320766961204b43"
    "3947485a203e0d14f6" + "00" * 43 + "0302")
# KB3AC-10, 2026-08-23 04:52z (`corpora.ONAIR_TURN_GRANTS`): stock RMS Trimode
# 1.4.2.0 on 80 m, the deepest session this station has held. Overs 1 and 3 of its
# greeting, decoded off the recording at 23 and 22 of 24 reference columns.
_KB3AC_FULL = bytes.fromhex(
    "524d53205472696d6f646520312e342e322e30204b423341432d48462d474154455741590d"
    "0a573953534a206861732031343430206461696c79206d696e757465732072656d61696e69"
    "6e672077697468204b423341432d314d")
_KB3AC_SHORT = bytes.fromhex(
    "61204b423341432d3130203e0d14f6" + "00" * 73 + "0482")
# The one BW2300 session held with a real VARA at both ends (`corpora.BW2300_REFERENCE`),
# initiator side, first over: 89 of the 126 payload bytes, VARA HF 4.9.0's own framing.
_BENCH_FULL = bytes.fromhex(
    "4b45535452454c205245464552454e43452053455353494f4e20412d3e42203031323334"
    "3536373839204b45535452454c205245464552454e43452053455353494f4e20412d3e42"
    "2030313233343536373839204b4553545281")


@pytest.mark.parametrize("body,nbytes,head", [
    (_NS0A_SHORT, 61, b"5 minutes.\r[WL2K-5.0"),
    (_KC9_FULL, 89, b"RMS Trimode 1.4.2.0\r\n"),
    (_KC9_SHORT, 43, b"-B2FWIHJM$]\r;PQ: 5573"),
    (_KB3AC_FULL, 89, b"RMS Trimode 1.4.2.0 K"),
    (_KB3AC_SHORT, 13, b"a KB3AC-10 >\r"),
    (_BENCH_FULL, 89, b"KESTREL REFERENCE SES"),
])
def test_payload_extracted_from_real_gateway_frames(body, nbytes, head):
    pl = phy.vara_payload(body)
    assert len(pl) == nbytes
    assert pl.startswith(head)


@pytest.mark.parametrize("body", [_NS0A_SHORT, _KC9_FULL, _KC9_SHORT,
                                 _KB3AC_FULL, _KB3AC_SHORT, _BENCH_FULL])
def test_builder_reproduces_the_real_bodies(body):
    """Rebuild each captured body from its own payload. body[88:90] carries two
    session fields we have not reversed, so the comparison stops there."""
    rebuilt = phy.vara_body(phy.vara_payload(body), "W9SSJ")
    assert rebuilt[:88] == body[:88]


_REAL_BODIES = ((_NS0A_SHORT, False), (_KC9_FULL, True), (_KC9_SHORT, False),
                (_KB3AC_FULL, True), (_KB3AC_SHORT, False), (_BENCH_FULL, True))


@pytest.mark.parametrize("body,full", _REAL_BODIES)
def test_the_per_frame_field_is_never_zero_and_its_low_bit_says_full(body, full):
    """What closes a real over, measured rather than assumed.

    body[88:90] is unreversed, and the population says two things about it. It is
    never zero: the short forms here read ``01 02``, ``03 02`` and ``04 82``, and
    the full ones end 0x09, 0x4d and 0x81 — with 0x41, 0x49, 0x51, 0x89 and 0x91
    on the other overs of the same recordings. And bit 0 separates the two forms
    over every real body held, four stations and two VARA builds.

    :func:`phy.vara_body` wrote 0x00 there for both forms until 2026-08-26 — the
    one value the population does not hold and, under that rule, *short* on an
    over carrying no 0x14 trailer to end it. The short form is now measured (see
    below); the full form still passes ``tail`` through, because the field moves
    from over to over on every real station held.
    """
    assert body[89] != 0x00
    assert bool(body[89] & 1) is full


# Two stock VARA HF 4.9.0 instances on the Wine bench's fake cable, 2026-08-26,
# one recording per direction — so each body is attributable to the station that
# keyed it without a scorer in between. Every one decoded CRC-clean off the cable
# that carried it by `rx.varahf2300.decode_overs`, and every payload is known
# exactly because the bench handed it to that station's data port. Neither
# modulator is ours, which is the whole value of these four: they are what a real
# VARA renders, not a round trip through our own builder.
#
# The recordings run to tens of megabytes and are git-ignored, so the bodies are
# inlined the way `test_turn_law`'s tone sequences are  [see that module's
# preamble]. The B2F traffic is of the right SHAPE and nothing more — the
# challenge, its response and the far-end call are invented, because a real
# challenge/response pair recorded byte-exact is what `mask_pr` exists to keep out
# of a shipping file, and a body carries it past any masking of the transcript.
_HANDOVER_SHORT = [
    # A -> B: a 65-byte B2F login block, and the over this station's own rendering
    # is diffed against.
    (b";FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: 10000001\r; N0CALL DE W9SSJ\rFF\r",
     bytes.fromhex(
         "3b46573a20573953534a0d5b5061742d312e302e302d423246484d245d0d3b50523a20"
         "31303030303030310d3b204e3043414c4c20444520573953534a0d46460d14f6"
         + "00" * 21 + "0482")),
    # B -> A: the responder's 50-byte CMS greeting.
    (b"[WL2K-5.0-B2FWIHJM$]\r;PQ: 20000002\rCMS via W1AW >\r",
     bytes.fromhex(
         "5b574c324b2d352e302d4232465749484a4d245d0d3b50513a2032303030303030320d"
         "434d53207669612057314157203e0d14f6" + "00" * 36 + "0482")),
    # B -> A: four bytes.
    (b";OK\r", bytes.fromhex("3b4f4b0d14f6" + "00" * 82 + "0482")),
    # B -> A: three.
    (b"FQ\r", bytes.fromhex("46510d14f6" + "00" * 83 + "0482")),
]


@pytest.mark.parametrize("payload,body", _HANDOVER_SHORT)
def test_a_short_over_is_built_exactly_as_a_real_4_9_0_builds_it(payload, body):
    """The whole 90 bytes, not the first 88.

    The arbiter is somebody else's modem: two 4.9.0s keying at each other with the
    payload known from the host port, so nothing here is a round trip through this
    tree's own encoder. Payload length spans 3 to 65 bytes and both stations and
    both directions are represented, which is what says ``04 82`` is the form's
    terminator rather than one session's number.
    """
    assert phy.vara_body(payload, "W9SSJ") == body


def test_the_terminator_is_the_only_thing_that_ever_differed():
    """What the 110 s silences cost, in bytes.

    Against the same 65-byte block a real VARA rendered, this station's over was
    already byte-identical through body[87] — payload, trailer, callsign CRC and
    zero fill — and carried ``00 00`` where the real one carries ``04 82``. The
    peer read the payload out of both and answered only the second one by taking
    the turn; the first it acknowledged twice and then waited out.
    """
    payload, real = _HANDOVER_SHORT[0]
    ours = phy.vara_body(payload, "W9SSJ")
    assert ours[:88] == real[:88]
    assert real[88:90] == b"\x04\x82"


def test_trailer_carries_the_callers_callsign_crc():
    """The byte after 0x14 is the high byte of CRC-16/GENIBUS(own callsign) — it
    identifies the session, not the sender (both gateways sent our own 0xf6)."""
    body = phy.vara_body(b"hello", "W9SSJ")
    assert body[5] == 0x14
    assert body[6] == crc16_genibus(b"W9SSJ") >> 8 == 0xF6


def test_full_over_is_89_payload_bytes_plus_a_field():
    """body[89] is a per-frame field, never payload — treating it as payload corrupts
    the join between overs (observed off air)."""
    pl = bytes(range(89))
    body = phy.vara_body(pl, "W9SSJ", tail=0x09)
    assert len(body) == 90 and body[:89] == pl and body[89] == 0x09
    assert phy.vara_payload(body, length=89) == pl


def test_binary_payload_containing_0x14_is_ambiguous_without_a_length():
    """KNOWN LIMITATION, not a passing behaviour to rely on: a payload that itself
    contains 0x14 cannot be delimited by scanning for the trailer. Winlink traffic
    is compressed, so this WILL occur on air. The length has to come from the ARQ
    layer until body[88:89] is reversed  [spec 04 §4.2B]."""
    pl = bytes(range(90))                      # contains 0x14 at index 20
    assert phy.vara_payload(pl) == pl[:20]     # truncates: the ambiguity, demonstrated
    assert phy.vara_payload(pl, length=89) == pl[:89]


def test_kestrel_marker_is_not_present():
    """kestrel's own framing reserves body[0] as a control marker; VARA does not,
    and a gateway will not parse an over that carries one."""
    assert phy.vara_body(b"ABC", "W9SSJ")[0:3] == b"ABC"


def test_a_body_short_of_90_bytes_reads_short():
    """The trailer scan reads the padded copy, not the caller's own bytes. A
    partial body is not something a gateway sends, but a decoder that raises on
    one turns a truncated frame into a crash instead of a NAK."""
    assert phy.vara_payload(b"hi", caller="W9SSJ") == b"hi"
