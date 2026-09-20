# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The B2F session against the bytes real gateways sent this station.

The first two greeting blocks below are byte-exact reconstructions of the
plaintext two live Winlink RMS gateways delivered over VARA in recorded
sessions (the same material the kestrel receive chain is held to; the
reconstruction is asserted against the recorded bytes whenever the corpus is
present). The third is KY4RY over ARDOP, delivered in two payload frames and
in the shape a real gateway actually sent it: a greeting cut across the frame
seam, and behind it the same greeting starting again. They are what a session
must answer. The scripted exchanges then walk both roles through proposals,
transfer and close-out, with a planted corruption at every checkpoint —
proposal checksum, EOT checksum, proposed size, compressed CRC, a line where
a command was owed — each of which must go red and stop the session.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hfmodem.winlink import (B2FSession, Message, compose, compress, mask_pr,
                             secure_login_response, summarize)
from hfmodem.winlink.session import EOT, SOH, STX, _is_forward

KC9GHZ = ("RMS Trimode 1.4.2.0\r\n"
          "W9SSJ has 110 daily minutes remaining with KC9GHZ (EN62BK)\r"
          "[WL2K-5.0-B2FWIHJM$]\r"
          ";PQ: 55734713\r"
          "CMS via KC9GHZ >\r").encode()
NS0A = ("RMS Trimode 1.4.2.0 Welcome to the NS0A Gateway\r\n"
        "W9SSJ has 118 daily minutes remaining with NS0A (EN41WK)\r"
        "Sessions for users running unregistered versions of Vara are limited "
        "to 5 minutes.\r"
        "[WL2K-5.0-B2FWIHJM$]\r"
        ";PQ: 17702218\r"
        "CMS via NS0A >\r").encode()

# KY4RY over ARDOP on 14103.0, 2026-08-14 — the two 4PSK.500.100 payload frames
# besra took off a second station, its first, in the order the link delivered
# them. Kestrel had read KC9GHZ's greeting off the air on 2026-08-06 and shrike
# had delivered `RMS Trimode 1.4.2.0\r\nTh` to its host on 2026-08-09, but both
# stopped short of the SID line; these two are the first SID and `;PQ:`
# challenge a modem here decoded for itself. The second is byte-exact off
# `logs/onair/20260814T151101Z-besra-14101500.wav` (frame at 149.63 s, q=63).
# The first is the session report's transcription, which starts mid-line at
# " (Starlink)" and so runs 126 of the frame's 128 bytes; the two it is short
# are the "NC" before it, which the second frame's own copy of that banner
# line spells out.
#
# Neither frame is a greeting. The first ends mid-SID-line, the second opens
# with the CR that closes it, and past the prompt the second carries the *head
# of the greeting over again* — the gateway repeated itself while the link
# below was stalled, so what reached the parser was one greeting's tail, its
# challenge and prompt, and then the next greeting starting up.
KY4RY = (
    b"NC (Starlink)\r\nW9SSJ has 120 daily minutes remaining with KY4RY "
    b"(FM26DC){SFI = 108 On 2026-08-14 13:00 UTC}\r[WL2K-5.0-B2FWIHJM$]",
    b"\r;PQ: 99446591\rCMS via KY4RY >\rRMS Trimode 1.4.2.0 Welcome to KY4RY "
    b"Hybrid RMS - Outer Banks NC (Starlink)\r\nW9SSJ has 119 daily ",
)
#: The second frame stops mid-line and the frame that would have finished it
#: never came — the link below stalled and the session was torn down. Anything
#: the gateway said next would have arrived behind this.
KY4RY_TAIL = b"minutes remaining with KY4RY (FM26DC)\r"


def test_the_embedded_greetings_are_the_recorded_bytes():
    """The reconstructions above against the recorded host-port output of the
    live sessions, when the corpus is on this machine."""
    from hfmodem.tests.kestrel.corpora import OFFAIR as base
    if not base.is_dir():
        pytest.skip(f"off-air gateway corpus not present at {base}")
    assert KC9GHZ == (base / "KC9GHZ_2300" / "payload.bin").read_bytes()
    assert NS0A == (base / "NS0A_2300" / "payload.bin").read_bytes()


# --------------------------------------------------------------------------- #
# Handshake against the real greetings.
def test_missing_sid_is_an_incomplete_greeting_not_a_capability_verdict():
    # K9KDJ, 2026-09-20 13:56: live deliveries skipped the SID and PQ marker.
    chunks = [
        b"RMS Trimode 1.4.3.1 Welcome to Newburgh, IN - K9KDJ Hybrid RMS\r\n"
        b"W9SSJ has 1400 daily minu",
        b"tes remaining with K9KDJ (EM67HW)\rSessions for users running "
        b"unregistered versions of Var",
        b": 35418710\rCMS via K9KDJ >\r",
    ]
    session = B2FSession("W9SSJ", target="K9KDJ")
    for chunk in chunks:
        assert session.feed(chunk) == b""
    assert session.done and not session.remote_sid
    assert "incomplete greeting" in session.failure
    assert "no B2 support" not in session.failure


def test_explicit_non_b2_sid_still_fails_capability_negotiation():
    session = B2FSession("W9SSJ")
    assert session.feed(b"[Example-1.0-B1FHM$]\rCMS >\r") == b""
    assert "no B2 support" in session.failure


@pytest.mark.parametrize("banner,gateway,challenge", [
    (KC9GHZ, "KC9GHZ", "55734713"),
    (NS0A, "NS0A", "17702218"),
])
def test_a_real_greeting_is_answered(banner, gateway, challenge):
    s = B2FSession("W9SSJ", role="calling", target=gateway, grid="EN63")
    assert s.start() == b""                     # the gateway speaks first
    out = s.feed(banner)
    assert s.remote_sid == "[WL2K-5.0-B2FWIHJM$]"
    assert s.challenge == challenge
    lines = out.decode().split("\r")
    assert lines[0] == ";FW: W9SSJ"
    assert lines[1] == "[HFM-0.1-B2FHM$]"
    assert lines[2] == f"; {gateway} DE W9SSJ (EN63)"
    assert lines[3] == "FF"                     # empty outbox: nothing to offer
    assert not s.done and not s.failure


def test_a_greeting_split_across_overs_is_reassembled():
    """VARA delivered these bytes over several 90-byte overs; the session must
    not care where the seams fall."""
    s = B2FSession("W9SSJ", role="calling", target="NS0A")
    out = b""
    for i in range(0, len(NS0A), 90):
        out += s.feed(NS0A[i:i + 90])
    assert b";FW: W9SSJ\r" in out and out.endswith(b"FF\r")


def test_a_password_answers_the_challenge():
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", password="hunter2")
    out = s.feed(KC9GHZ)
    expect = secure_login_response("55734713", "hunter2")
    assert f";PR: {expect}\r".encode() in out


def test_no_password_sends_no_pr():
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ")
    assert b";PR:" not in s.feed(KC9GHZ)


def test_the_courtesy_line_is_what_puts_a_proposing_first_turn_over_one_over():
    """A VARA DATA over carries 89 payload bytes  [kestrel.arq.phy.payload_size].

    The caller's `; TARGET DE MYCALL (GRID)` line is a courtesy the forward
    protocol never asked for, and with it a first turn that proposes a message
    needs two overs — the shape no gateway this station has called has ever
    answered, 0 of 5, against 6 of 6 for the turns that fitted one
    [vara-evening-en63bc-0910/analysis/kc9]. Without it the same turn is one
    over, and it is the only line that changes.
    """
    msg = compose("KC9GHZ", "W9SSJ", "wx bulletin", "high pressure inbound\r\n",
                  mid="ABCDEF123456")
    kw = dict(role="calling", target="KC9GHZ", password="hunter2", grid="EN63bc",
              outbox=[msg])
    spoken = B2FSession("W9SSJ", **kw).feed(KC9GHZ)
    quiet = B2FSession("W9SSJ", comment=False, **kw).feed(KC9GHZ)
    courtesy = b"; KC9GHZ DE W9SSJ (EN63bc)\r"
    assert courtesy in spoken and len(spoken) > 89
    assert len(quiet) <= 89
    assert quiet == spoken.replace(courtesy, b"")


def test_secure_login_published_vectors():
    """The two vectors published with the response algorithm the B2F
    specification points to. The password's case matters."""
    assert secure_login_response("23753528", "FOOBAR") == "72768415"
    assert secure_login_response("23753528", "FooBar") == "95074758"


# --------------------------------------------------------------------------- #
# WW2MI's CMS over ARDOP on 7103600, 2026-08-18 21:48 local: the one connect
# whose login was refused. Both blocks are byte-exact, decoded back off the
# session's own receive recording — the greeting as three 4PSK.200.100 frames
# of 64, 64 and 31 bytes, the refusal as two of 64 and 35.
#
# The refusal is one transmission: the gateway's very next after our handshake
# was acknowledged, carrying the prompt and its verdict on our `;PR:` together.
# There was never a turn in which `Login [931]:` was a question waiting on an
# answer.
WW2MI_CHALLENGE = ("RMS Trimode 1.4.2.1 W8OAK Oakland County ARPSC.\r\n"
                   "W9SSJ has 1437 daily minutes remaining with WW2MI (EN82KR)\r"
                   "[WL2K-5.0-B2FWIHJM$]\r"
                   ";PQ: 90189792\r"
                   "CMS via WW2MI >\r").encode()
WW2MI_REFUSAL = ("Login [931]:\r"
                 "CMS via WW2MI >\r"
                 "Invalid login challenge response -- 2 attempts remaining "
                 "(H for Help)\r").encode()


def _ww2mi_refused() -> B2FSession:
    s = B2FSession("W9SSJ", role="calling", target="WW2MI", password="hunter2")
    s.feed(WW2MI_CHALLENGE)
    return s


def test_one_connect_spends_one_login_attempt():
    """The account's challenge allowance is finite, so how much of it a session
    spent is a number the record has to carry. A prompt is a station talking
    once the handshake is answered, and only the handshake answers a `;PQ:`:
    the gateway's second `CMS via WW2MI >` draws no second `;PR:`, and nothing
    at all goes back on the air."""
    s = _ww2mi_refused()
    assert s.sent_text.count(";PR: ########") == 1
    assert s.feed(WW2MI_REFUSAL) == b""
    assert s.sent_text.count(";PR: ########") == 1


def test_what_we_transmitted_is_in_the_record_beside_what_they_said():
    """One side of an exchange is half a record. What this station put on the
    air on 2026-08-18 was recoverable only by decoding its own transmissions
    back out of the receive capture, frame by frame — four 4FSK.200.50S of
    sixteen bytes each, and these four lines are what they carried."""
    s = _ww2mi_refused()
    assert s.sent_text == [";FW: W9SSJ", "[HFM-0.1-B2FHM$]", ";PR: ########",
                           "; WW2MI DE W9SSJ"]


def test_the_transcript_keeps_both_sides_in_the_order_they_spoke():
    s = _ww2mi_refused()
    assert s.transcript[-6:] == [
        (False, ";PQ: 90189792"),
        (False, "CMS via WW2MI >"),
        (True, ";FW: W9SSJ"),
        (True, "[HFM-0.1-B2FHM$]"),
        (True, ";PR: ########"),
        (True, "; WW2MI DE W9SSJ")]


def test_the_challenge_response_is_recorded_as_sent_and_never_as_digits():
    """`;PR:` is 30 bits of MD5 over the challenge, the password and a
    published salt, and the challenge sits two lines above it in the same
    transcript. The pair searches a password offline, and the digits answer no
    question that the line's presence does not."""
    s = _ww2mi_refused()
    answer = secure_login_response("90189792", "hunter2")
    said = "\n".join(t for _, t in s.transcript)
    assert answer not in said and "hunter2" not in said
    assert ";PR: ########" in said


def test_the_mask_holds_on_shapes_a_well_formed_answer_never_takes():
    """`rehear` masks this station's own transmissions rejoined from the frames a
    receiver read back, where the digits can arrive a byte short or a frame short,
    so the rule is the `;PR:` prefix and not the shape. Character for character,
    too: that caller cuts the joined stream back into frames by their lengths.
    """
    assert {said: mask_pr(said) for said in (
        ";PR: 12345678", ";PR:12345678", ";PR: 1234567", ";PR: 123456789",
        ";PR: 1234", ";PR: 1234x678")} == {
        ";PR: 12345678": ";PR: ########",
        ";PR:12345678": ";PR:########",
        ";PR: 1234567": ";PR: #######",
        ";PR: 123456789": ";PR: #########",
        ";PR: 1234": ";PR: ####",
        ";PR: 1234x678": ";PR: ########"}
    assert (mask_pr(";FW: W9SSJ\r;PR: 12345678\r; WW2MI DE W9SSJ\r")
            == ";FW: W9SSJ\r;PR: ########\r; WW2MI DE W9SSJ\r")


def test_our_commands_are_no_more_transcript_than_theirs():
    """The same rule both ways: FF, FS and F> are this layer's punctuation."""
    s = _ww2mi_refused()
    assert not any(t.startswith("F") for t in s.sent_text)


# --------------------------------------------------------------------------- #
# KY4RY, 2026-08-14: the greeting that killed a live session.
def test_the_ky4ry_frames_are_whole_payloads():
    """Both deliveries are one 4PSK.500.100 payload each — the check that says
    the reconstructed first frame is the whole frame and not a fragment."""
    assert [len(f) for f in KY4RY] == [128, 128]


def _ky4ry(**kw) -> tuple[B2FSession, bytes]:
    s = B2FSession("W9SSJ", role="calling", target="KY4RY", **kw)
    out = b"".join(s.feed(frame) for frame in KY4RY)
    return s, out


def test_the_ky4ry_greeting_is_answered():
    """The live bytes, fed as the two link deliveries that carried them.

    The banner arrives *after* the prompt here — the gateway had started
    repeating itself — so it lands in our turn, where it used to be fatal. The
    session must read the SID and challenge out of the seam-split greeting,
    answer it, and be waiting on the gateway with mail still able to move.
    """
    s, out = _ky4ry()
    assert not s.failure, s.log_lines
    assert s.remote_sid == "[WL2K-5.0-B2FWIHJM$]"
    assert s.challenge == "99446591"
    lines = out.decode().split("\r")
    assert lines[:4] == [";FW: W9SSJ", "[HFM-0.1-B2FHM$]",
                         "; KY4RY DE W9SSJ", "FF"]
    assert s.stage == "their turn" and not s.done
    assert ("banner: RMS Trimode 1.4.2.0 Welcome to KY4RY Hybrid RMS - "
            "Outer Banks NC (Starlink)") in s.log_lines


def test_a_repeated_greeting_does_not_unseat_our_proposals():
    """With mail queued the handshake goes straight to FS-waiting, so a gateway
    repeating itself lands there instead of in our turn. Same rule, same
    answer: it is a station talking, not the FS we are owed being malformed."""
    msg = compose("W9SSJ", "KY4RY", "hello", "test\r\n", mid="MIDMIDMID123")
    s, out = _ky4ry(outbox=[msg])
    assert not s.failure and b"F> " in out
    assert s.stage == "awaiting proposal answer"
    assert s.feed(KY4RY_TAIL + b"FS Y\r").startswith(bytes((SOH,)))
    assert s.sent_mids == ["MIDMIDMID123"]


# --------------------------------------------------------------------------- #
# KX8U over ARDOP: the sentence that cost a rig slot.
#
# besra decoded both payload frames of it — 4FSK.200.50S even and odd, sess=0x78,
# CRC valid, q=90 and q=77 — and the session record said `mail: nothing moved`
# and nothing else. The reason was read back off the recording six hours later.
# The text below is that reassembly, not a byte-exact frame corpus like KY4RY's
# above, so it is fed here as the one line the gateway meant it to be.
KX8U_REFUSAL = ("*** Unknown client types are not allowed on production servers "
                "-- use cms-z.winlink.org - Disconnecting (208.102.176.56)")


def _refused(tail: bytes = b"\r") -> B2FSession:
    """A fetch-only session, greeted and answered, then refused by the CMS.

    The greeting is KC9GHZ's recorded one — every RMS Trimode gateway opens
    that way and what the refusal lands behind is not the point — and the
    refusal is KX8U's own.
    """
    s = B2FSession("W9SSJ", role="calling", target="KX8U")
    s.feed(KC9GHZ)
    s.feed(KX8U_REFUSAL.encode() + tail)
    return s


def test_the_refusal_is_in_the_record_and_not_only_in_the_audio():
    assert any("Unknown client types are not allowed" in said
               for said in _refused().remote_text)


def test_a_refusal_the_gateway_hung_up_mid_sentence_is_still_kept():
    """A station that refuses drops the link, so the last line it says is the
    one least likely to arrive terminated — and the most worth having."""
    s = _refused(tail=b"")
    assert s.remote_text[-1].endswith("(208.102.176.56)")


def test_the_greeting_a_gateway_spoke_is_the_record_too():
    s = _refused()
    assert s.remote_text[:3] == [
        "RMS Trimode 1.4.2.0",
        "W9SSJ has 110 daily minutes remaining with KC9GHZ (EN62BK)",
        "[WL2K-5.0-B2FWIHJM$]"]
    assert ";PQ: 55734713" in s.remote_text


def test_the_transcript_is_what_was_said_not_what_was_commanded():
    """FF, FS and F> are this layer's own vocabulary. A person reading back a
    dead session wants the gateway's sentences, not its punctuation."""
    s = _connected()
    s.feed(b"FF\r")
    assert s.done and "FF" not in s.remote_text


def test_the_commands_are_kept_beside_the_words_and_marked_apart_from_them():
    """Dropped, they were unrecoverable. KN4LQN's CMS refused a proposal on
    2026-08-29 — `*** [1] Unexpected response to proposal - Disconnecting` —
    and establishing that our answer had gone out at all, never mind what it
    said, took byte-accounting against the frame log, because the `FS` line was
    the one line the record did not hold. `exchange` holds every line in the
    order it was spoken and says which are the protocol's; `transcript` is that
    less the protocol, so it still reads as a conversation."""
    s = _connected()
    s.feed(b"FF\r")
    assert (False, "FF", True) in s.exchange
    assert [(o, t) for o, t, protocol in s.exchange if not protocol] == s.transcript
    assert all(_is_forward(t) for o, t, protocol in s.exchange if protocol)


def test_bytes_that_were_never_text_do_not_corrupt_the_record():
    """VARA and PACTOR carry trailer bytes inside the frame CRC and a delivery
    can be partial, so a line can arrive with bytes in it that are not
    characters. They are shown as unreadable rather than as plausible text."""
    s = B2FSession("W9SSJ", role="calling", target="KX8U")
    s.feed(b"Welcome to \xff\xfe KX8U\r\x00\x81\x92\r")
    assert s.remote_text == ["Welcome to \ufffd\ufffd KX8U"]


def test_a_line_longer_than_a_gateway_would_speak_is_cut_and_says_so():
    s = B2FSession("W9SSJ", role="calling", target="KX8U")
    s.feed(b"x" * 260 + b"\r")
    assert s.remote_text == ["x" * 200 + " [+60 bytes]"]


def test_a_gateway_without_b2_is_refused():
    s = B2FSession("W9SSJ", role="calling")
    out = s.feed(b"[OLDBBS-1.0-AF$]\rHello >\r")
    assert out == b"" and s.failure and s.done


# --------------------------------------------------------------------------- #
# Receiving mail: the scripted gateway turn.
def _connected(**kw) -> B2FSession:
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", **kw)
    s.feed(KC9GHZ)
    return s


def _proposal_block(*props: str, checksum: int | None = None) -> bytes:
    total = sum(sum(p.encode()) + 0x0D for p in props)
    checksum = ((-total) & 0xFF) if checksum is None else checksum
    return "".join(p + "\r" for p in props).encode() + f"F> {checksum:02X}\r".encode()


def _body_stream(blob: bytes, title: str = "t", offset: bytes = b"0") -> bytes:
    head = title.encode() + b"\0" + offset + b"\0"
    out = bytes((SOH, len(head))) + head
    for i in range(0, len(blob), 250):
        chunk = blob[i:i + 250]
        out += bytes((STX, len(chunk) & 0xFF)) + chunk
    return out + bytes((EOT, (-sum(blob)) & 0xFF))


def _mail() -> tuple[Message, bytes]:
    msg = compose("KC9GHZ", "W9SSJ", "wx bulletin", "high pressure inbound\r\n",
                  mid="ABCDEF123456")
    return msg, compress(msg.render())


def test_mail_is_received_end_to_end():
    msg, blob = _mail()
    s = _connected()
    out = s.feed(_proposal_block(
        f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    assert out == b"FS +\r"
    out = s.feed(_body_stream(blob))
    assert [m.render() for m in s.inbox] == [msg.render()]
    assert out == b"FF\r"                       # our turn again, nothing to offer
    out = s.feed(b"FF\r")                       # gateway is empty too
    assert out == b"FQ\r" and s.done and not s.failure


def test_mail_moves_through_a_session_the_ky4ry_greeting_opened():
    """The 2026-08-14 bytes all the way to a message in the inbox — what that
    link carried the bytes for and the state machine refused."""
    msg, blob = _mail()
    s, _ = _ky4ry()
    out = s.feed(KY4RY_TAIL + _proposal_block(
        f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    assert out == b"FS +\r"
    s.feed(_body_stream(blob))
    assert [m.render() for m in s.inbox] == [msg.render()]
    assert not s.failure


def test_a_message_arriving_in_dribbles_is_reassembled():
    msg, blob = _mail()
    s = _connected()
    s.feed(_proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    stream = _body_stream(blob)
    out = b""
    for i in range(0, len(stream), 7):
        out += s.feed(stream[i:i + 7])
    assert len(s.inbox) == 1 and out == b"FF\r"


def test_two_proposals_one_duplicate_mid():
    """A duplicated MID inside one block is deferred, not accepted twice."""
    msg, blob = _mail()
    line = f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"
    s = _connected()
    assert s.feed(_proposal_block(line, line)) == b"FS +=\r"


def test_an_unreadable_proposal_form_is_deferred():
    """FA is the uncompressed ascii form this station does not read; every
    proposal still gets an answer."""
    msg, blob = _mail()
    s = _connected()
    out = s.feed(_proposal_block(
        "FA P FC1CDC F6ABJ F6AXV 24754_F6FBB 345",
        f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    assert out == b"FS =+\r"


def test_a_real_compressed_message_rides_the_whole_receive_path():
    """The externally produced message pair through the session itself: FC
    proposal with the real sizes, the real compressed bytes in STX blocks, and
    the parsed mail — jpeg attachment included — in the inbox."""
    from hfmodem.tests.winlink import corpora
    plain, packed = corpora.real_pair()
    s = _connected()
    out = s.feed(_proposal_block(f"FC EM LPE5NXDVLVSQ {len(plain)} {len(packed)} 0"))
    assert out == b"FS +\r"
    out = s.feed(_body_stream(packed, title="73 fra Brekke"))
    assert out == b"FF\r" and not s.failure
    (msg,) = s.inbox
    assert msg.mid == "LPE5NXDVLVSQ" and msg.render() == plain
    assert msg.attachments[0].data[:2] == b"\xff\xd8"


def test_the_ww2mi_turn_is_received_as_mail():
    """Everything WW2MI said on 2026-08-16, fed to a session as one stream:
    banner, SID, `;PQ:` challenge, prompt, `;PM:`, the `FC` proposal and the
    545-byte forward block. The link, the login and the proposal all worked
    that night and the run still ended `LJ2AJE2IHO9B: message ends inside the
    body`, with nothing kept. The same bytes must end in the inbox."""
    from hfmodem.tests.winlink import corpora
    stream, plain = corpora.ww2mi()
    s = B2FSession("W9SSJ", role="calling", target="WW2MI")
    out = s.feed(stream)
    assert not s.failure
    assert out.endswith(b"FS +\rFF\r")
    (msg,) = s.inbox
    assert msg.render() == plain
    assert msg.subject == "testing without the magic subject line"
    assert s.remote_sid == "[WL2K-5.0-B2FWIHJM$]"
    assert s.challenge == "31206858"
    assert [mid for mid, _ in s.received_blocks] == ["LJ2AJE2IHO9B"]


# KN4LQN over ARDOP at 3592008, 2026-08-28 21:11-21:13Z, off
# `working/onair-0828-2111/ardop-night-05-kn4lqn.log`: every line the gateway
# spoke, in the order it spoke them. The login was accepted — `;PQ:` drew a
# `;PR:` and the CMS answered with its prompt and an offer rather than English
# — and the run still ended `*** [1] Unexpected response to proposal -
# Disconnecting (100.7.81.58)` with JWKY65C2OZES left at the far end. The `FC`
# line is the one thing here the record does not hold: the transcript keeps a
# gateway's words and not its commands, so it carries the form and the
# compressed size the `;PM:` states, and an uncompressed size that is a guess.
# (That is the 2026-08-28 log, not this layer: `exchange` keeps the commands.)
KN4LQN = ("RMS Trimode 1.4.2.3\r"
          "W9SSJ has 155 daily minutes remaining with KN4LQN (FM17EI)\r"
          "[WL2K-5.0-B2FWIHJM$]\r"
          ";PQ: 30340143\r"
          "CMS via KN4LQN >\r").encode()
KN4LQN_PM = (";PM: W9SSJ JWKY65C2OZES 524 saul.stjohn@gmail.com "
             "testing 8/26 this is a test\r").encode()


def test_the_kn4lqn_offer_draws_the_answer_the_document_defines():
    """`FS +`, not `FS Y`: the FBB document names +/-/= and wl2k-go — whose SID
    this station wears — sends them. This exchange is what `FS Y` drew."""
    s = B2FSession("W9SSJ", role="calling", target="KN4LQN",
                   password="pw", client_sid="Pat-1.0.0")
    assert s.feed(KN4LQN) == (
        b";FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: "
        + secure_login_response("30340143", "pw").encode()
        + b"\r; KN4LQN DE W9SSJ\rFF\r")
    out = s.feed(KN4LQN_PM + _proposal_block("FC EM JWKY65C2OZES 640 524 0"))
    assert out == b"FS +\r"
    assert not s.failure and s.stage == "receiving"
    assert s.unfetched == ["JWKY65C2OZES"]


def test_the_answer_a_refusal_lands_on_is_in_the_record_the_operator_reads():
    """The refusal above names our answer to the proposal, and the record has to
    hold the line it names. `sent` against `said` keeps the gateway's English
    reading as English while the protocol's own lines stand in their place in it."""
    s = B2FSession("W9SSJ", role="calling", target="KN4LQN",
                   password="pw", client_sid="Pat-1.0.0")
    s.feed(KN4LQN)
    s.feed(KN4LQN_PM + _proposal_block("FC EM JWKY65C2OZES 640 524 0"))
    said = summarize(s).splitlines()
    assert "mail: we queued: FS +" in said
    assert "mail: peer sent: FC EM JWKY65C2OZES 640 524 0" in said
    assert "mail: peer said: CMS via KN4LQN >" in said
    assert "mail: we said: ;PR: ########" in said
    assert secure_login_response("30340143", "pw") not in "\n".join(said)


def test_a_doubled_frame_in_the_ww2mi_turn_never_reaches_the_inbox():
    """What an ARQ layer that delivers a repeat twice costs, measured rather than
    reasoned about. The alternation an ARDOP session deduplicates data frames on
    is cleared at a role reversal — it has to be, or a new frame is taken for a
    repeat — so a frame the peer replays across one can be appended twice.

    Every frame boundary of the recovered WW2MI turn is doubled here at 16, 64
    and 128 bytes, and both of its STX blocks are doubled whole, which is the
    only alignment that leaves the framing in step. Not one of them produces
    mail: the block reader refuses the unaligned ones on its own framing and the
    aligned ones on the EOT checksum, and behind that the proposed compressed
    size is exact. So a doubled append is a session lost, not a message
    corrupted, and the receiver cannot be quietly wrong about what a gateway
    sent."""
    from hfmodem.tests.winlink import corpora
    stream, plain = corpora.ww2mi()
    start = stream.index(bytes([SOH]))

    def fed(data: bytes) -> B2FSession:
        s = B2FSession("W9SSJ", role="calling", target="WW2MI")
        s.feed(data)
        return s

    assert fed(stream).inbox[0].render() == plain, "the clean turn still parses"

    doubles = [(off, n) for n in (16, 64, 128)
               for off in range(start, len(stream) - n + 1, n)]
    at = start + 2 + stream[start + 1]                  # past the SOH header block
    while stream[at] == STX:                            # every whole STX block
        n = 2 + (stream[at + 1] or 256)
        doubles.append((at, n))
        at += n
    assert stream[at] == EOT and len(doubles) > 40

    for off, n in doubles:
        s = fed(stream[:off + n] + stream[off:off + n] + stream[off + n:])
        assert s.failure and not s.inbox, f"doubled {n} B at {off} was taken as mail"


# --------------------------------------------------------------------------- #
# 2026-08-23, WW2MI over ARDOP BW200: the ten payload frames of the CMS's half
# of this station's first real send, recovered from
# `logs/onair/20260823T031121Z-besra-7102100.wav` and given here in the order
# the link delivered them. The exchange is whole and correct up to the last
# one: greeting, challenge, three `;PM:` offers, `FS Y` for our message, three
# `FC` proposals whose `F> 88` checksums exactly, our `FS +++` — and then a
# forward block that opens on 0xFD.
#
# It opens there because the frame before it never arrived. The ARQ layer
# acknowledged the CMS's first data frame without decoding it, the ISS moved
# on, and 64 bytes — the SOH header block and the head of the first STX block
# — were spliced out of a stream this layer is entitled to read as continuous.
# 0xFD is the 65th byte of the message and was on the air at quality 80.
WW2MI_0823 = (
    b"RMS Trimode 1.4.2.2 W8OAK Oakland County ARPSC.\r\nW9SSJ has 1437 ",
    b"daily minutes remaining with WW2MI (EN82KR)\r[WL2K-5.0-B2FWIHJM$]",
    b"\r;PQ: 74048083\rCMS via WW2MI >\r",
    b";PM: W9SSJ WWTD6QMC61TV 464 saul.stjohn@gmail.com test3\r;PM: W9S",
    b"SJ V0DAF1MQ2Y1J 504 saul.stjohn@gmail.com this is a test of a me",
    b"ssage\r;PM: W9SSJ BJNKZAK5PF5K 530 saul.stjohn@gmail.com test ema",
    b"il another one\rFS Y\r",
    b"FC EM WWTD6QMC61TV 560 464 0\rFC EM V0DAF1MQ2Y1J 622 504 0\rFC EM ",
    b"BJNKZAK5PF5K 657 530 0\rF> 88\r",
    bytes.fromhex("fdfbfcd3abdf9527b917fed407f1ba97b4fb4ffd3f5ccf9cf44fac"
                  "aca7bdfbc7cf66c5b5117c5995fd6870a04b7ba4ece0dbbcbbe66d"
                  "d1821e4e6b5a1bb58ed0"),
)
WW2MI_0823_OFFERED = ["WWTD6QMC61TV", "V0DAF1MQ2Y1J", "BJNKZAK5PF5K"]


def _ww2mi_0823() -> B2FSession:
    """The 2026-08-23 session, frame by frame as the link delivered it."""
    ours = compose("W9SSJ", "saul.stjohn@gmail.com", "W9SSJ first send over ARDOP",
                   "testing\r\n", mid="2KX2HWM919HR")
    s = B2FSession("W9SSJ", role="calling", target="WW2MI", outbox=[ours])
    sent = [s.feed(frame) for frame in WW2MI_0823]
    assert sent[6].startswith(bytes((SOH,))), "our message went out on their FS Y"
    assert sent[8] == b"FS +++\r", "all three offers were accepted"
    return s


def test_a_forward_block_that_lost_its_head_says_so():
    """`expected SOH, got byte 0xfd` was true and read as the CMS refusing the
    message we had just sent it. The byte is not the fault: the bytes in front
    of it are, and nothing above the link can put them back."""
    s = _ww2mi_0823()
    assert s.failure.startswith("WWTD6QMC61TV began 0xfd, not SOH")
    assert "short of its head" in s.failure
    assert not s.inbox and not s.received_blocks


def test_the_offers_a_broken_transfer_leaves_behind_are_named():
    """Three messages were accepted and the session died on the first. They
    were never in danger — an `FS +` the transfer never honoured leaves them
    at the CMS — but the run reported a send and nothing else."""
    s = _ww2mi_0823()
    assert s.unfetched == WW2MI_0823_OFFERED


def test_a_receive_failure_is_not_a_verdict_on_what_we_sent():
    """2KX2HWM919HR was proposed, accepted, transmitted whole and delivered;
    the CMS confirmed it. The failure four turns later belongs to the CMS's
    own mail, and printing it against the mid read as a rejection."""
    s = _ww2mi_0823()
    assert s.sent_mids == ["2KX2HWM919HR"] and s.failure
    assert not s.remote_refusal
    report = summarize(s)
    assert ("mail: prepared 2KX2HWM919HR — accepted FS Y, body prepared"
            in report), report
    assert "delivery unconfirmed by the CMS" in report
    assert "remote reported:" not in report
    for mid in WW2MI_0823_OFFERED:
        assert f"mail: accepted {mid} and never got it" in report


def test_a_refusal_is_still_reported_against_the_message_it_refused():
    """The other half: WW2MI answered a completed EOT with an error line on
    2026-08-21, and that one is a verdict on the mid beside it."""
    msg, blob = _mail()
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    s.feed(b"FS Y\r")
    s.feed(b"*** Error check failed on receiving B2 message\r")
    assert s.sent_mids == [msg.mid]
    assert s.remote_refusal == "*** Error check failed on receiving B2 message"
    assert "*** Error check failed" in summarize(s).split("mail: prepared ")[1]


def _unreadable() -> tuple[B2FSession, str, bytes]:
    """A session handed a whole block it cannot read: the compressed body
    decodes, and the message inside it lies about its body size."""
    msg, _ = _mail()
    blob = compress(msg.render().replace(b"Body: 23", b"Body: 99"))
    s = _connected()
    s.feed(_proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    s.feed(_body_stream(blob))
    return s, msg.mid, blob


def test_a_message_the_parser_refuses_is_still_kept_as_bytes():
    """The specimen outlives the parse. A block that arrived whole is the only
    copy of what the gateway sent, and the run it came from does not repeat."""
    s, mid, blob = _unreadable()
    assert s.failure and not s.inbox
    assert s.received_blocks == [(mid, blob)]
    assert s.in_flight is None, "the EOT closed it, whatever the verdict said"


def _part_way_through() -> tuple[B2FSession, str, bytes]:
    """A session the link stopped inside a message: the header and 38 bytes of
    the first block, which is where KE8LVA's second message stopped."""
    msg, blob = _mail()
    s = _connected()
    s.feed(_proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    stream = _body_stream(blob)
    head = 2 + stream[1] + 2                    # the SOH block, then STX and its length
    s.feed(stream[:head + 38])
    return s, msg.mid, blob[:38]


def test_a_transfer_the_link_stopped_in_the_middle_of_keeps_what_arrived():
    """Tonight's shape: KE8LVA's second message opened, 38 bytes of it crossed,
    and the link went quiet. The run said three were never got, which was true
    of the mailbox and not of the air."""
    s, mid, head = _part_way_through()
    _, blob = _mail()
    assert s.in_flight == (mid, 38, len(blob))
    assert s.partial_block == head == blob[:38]
    assert s.unfetched == [mid]


def test_a_message_that_lost_its_head_is_not_a_partial():
    """No SOH, no message: bytes that arrive with nothing to attach them to
    cannot be resumed and are not a specimen of anything."""
    assert _ww2mi_0823().in_flight is None


def test_remote_fq_ends_the_session():
    s = _connected()
    s.feed(b"FQ\r")
    assert s.done and not s.failure


def test_a_block_of_mail_we_already_hold_is_not_an_empty_mailbox():
    """`FQ` says both mailboxes are empty, and it is final. A gateway offering
    mail this station already holds is not empty — the offer is answered
    `FS -`, nothing transfers, and quitting there hangs up on a far end that
    has just said what it is holding."""
    msg, blob = _mail()
    offer = f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"
    mine = [compose("W9SSJ", "KC9GHZ", f"m{i}", f"body {i}\r\n") for i in range(11)]
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=mine)
    s.feed(KC9GHZ)                              # five proposals go out
    s.feed(b"FS -----\r")
    assert s.feed(_proposal_block(offer)) == b"FS +\r"
    s.feed(_body_stream(blob))                  # received, and five more offered
    s.feed(b"FS -----\r")
    s.feed(b"FF\r")                             # empty, so the eleventh goes out
    s.feed(b"FS -\r")
    assert s.feed(_proposal_block(offer)) == b"FS -\rFF\r"
    assert not s.done and not s.failure


# --------------------------------------------------------------------------- #
# Planted corruptions on the receive path. Every one must go red, and red
# means: failure set, nothing further transmitted, inbox untouched.
def test_a_bad_proposal_checksum_stops_the_session():
    msg, blob = _mail()
    line = f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"
    good = _proposal_block(line)
    bad_sum = (int(good[-3:-1], 16) + 1) & 0xFF
    s = _connected()
    out = s.feed(_proposal_block(line, checksum=bad_sum))
    assert out == b"" and "checksum" in s.failure and not s.inbox


def test_a_bad_eot_checksum_stops_the_session():
    msg, blob = _mail()
    s = _connected()
    s.feed(_proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    stream = bytearray(_body_stream(blob))
    stream[-1] = (stream[-1] + 1) & 0xFF
    out = s.feed(bytes(stream))
    assert out == b"" and "EOT" in s.failure and not s.inbox


def test_a_short_body_contradicts_the_proposal():
    msg, blob = _mail()
    s = _connected()
    s.feed(_proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    out = s.feed(_body_stream(blob[:-1]))       # framing valid, one byte missing
    assert out == b"" and "proposed" in s.failure and not s.inbox


def test_a_corrupt_compressed_body_fails_its_own_crc():
    msg, blob = _mail()
    tampered = bytearray(blob)
    tampered[8] ^= 0x20
    s = _connected()
    s.feed(_proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    out = s.feed(_body_stream(bytes(tampered)))
    assert out == b"" and "CRC" in s.failure and not s.inbox


def test_a_stray_line_inside_a_proposal_block_stops_the_session():
    """The other half of the banner rule. A proposal block is one checksummed
    run of commands, so a line that is not one of them arriving between the FC
    and its F> is framing lost, not a gateway being chatty — and it stops the
    session before the checksum it would have broken."""
    msg, blob = _mail()
    s = _connected()
    out = s.feed(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0\r"
                 "RMS Trimode 1.4.2.0 Welcome to KY4RY\r".encode())
    assert out == b"" and "proposal block" in s.failure and s.done


def test_free_text_without_end_stops_the_session():
    """Tolerance with a floor under it: a far end that never gets back to the
    protocol is a desynced stream, and this must go red rather than absorb it
    until the run's own clock gives up."""
    s = _connected()
    out = s.feed(b"not a command\r" * 25)
    assert out == b"" and "free text" in s.failure and s.done
    assert s.log_lines.count("banner: not a command") == 20


def test_the_free_text_budget_is_per_turn():
    """Fifteen lines of banner in each of two turns is a gateway repeating
    itself, not a desynced stream, and thirty lines that a single turn would
    refuse must pass when the exchange moves between them."""
    msgs = [compose("W9SSJ", "KC9GHZ", f"m{i}", f"body {i}\r\n") for i in range(7)]
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=msgs)
    s.feed(KC9GHZ)                              # five proposals go out
    banner = b"RMS Trimode 1.4.2.0 Welcome to KC9GHZ\r" * 15
    s.feed(banner + b"FS NNNNN\r")
    assert not s.failure
    s.feed(b"FF\r")                             # our turn again: the last two
    assert s.stage == "awaiting proposal answer"
    s.feed(banner + b"FS NN\r")
    assert not s.failure and s.stage == "their turn"


def test_an_unknown_command_still_stops_the_session():
    """`F` is the protocol's own namespace: a line inside it that names no
    command is malformed, whatever the banner rule allows beside it."""
    s = _connected()
    assert s.feed(b"FX 12\r") == b"" and "unknown command" in s.failure


def test_a_command_where_fs_was_owed_still_stops_the_session():
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    assert s.feed(b"FF\r") == b"" and "expected FS" in s.failure


def test_an_unrequested_offset_is_refused():
    msg, blob = _mail()
    s = _connected()
    s.feed(_proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"))
    out = s.feed(_body_stream(blob, offset=b"42"))
    assert out == b"" and "offset" in s.failure and not s.inbox


# --------------------------------------------------------------------------- #
# Sending mail: the scripted far end answers our proposals.
def test_our_proposal_carries_the_message_and_its_sizes():
    """With mail queued, the handshake reply carries the proposal block where
    the empty session said FF."""
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    out = s.feed(KC9GHZ).decode()
    rendered, blob = msg.render(), compress(msg.render())
    line = f"FC EM MIDMIDMID123 {len(rendered)} {len(blob)} 0"
    assert line + "\r" in out
    assert "FF\r" not in out
    expect = (-(sum(line.encode()) + 0x0D)) & 0xFF
    assert f"F> {expect:02X}\r" in out


def test_an_accepted_proposal_is_followed_by_the_body():
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    out = s.feed(b"FS Y\r")
    blob = compress(msg.render())
    assert out.startswith(bytes((SOH,)))
    assert blob in out
    assert out.endswith(bytes((EOT, (-sum(blob)) & 0xFF)))
    assert s.sent_mids == ["MIDMIDMID123"]


# WW2MI's CMS over ARDOP, 2026-08-21: FS Y, the body went out whole, and the
# next line back closed the link rather than opening the far end's own turn.
BODY_REFUSAL = ("*** Error check failed on receiving B2 message. [Received "
                "data stream not a correct format] - Disconnecting")


def _body_refused() -> B2FSession:
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    s.feed(b"FS Y\r")
    s.feed(BODY_REFUSAL.encode() + b"\r")
    return s


def test_a_gateway_that_refuses_the_body_still_leaves_the_mid_recorded():
    """`sent_mids` is what this station transmitted, not what the gateway
    kept — the refusal is what says the transfer did not land."""
    s = _body_refused()
    assert s.sent_mids == ["MIDMIDMID123"]
    assert s.failure and "Error check failed" in s.failure
    assert s.stage == "failed" and s.done


def test_a_rejected_proposal_sends_no_body():
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    assert s.feed(b"FS N\r") == b""
    assert s.sent_mids == [] and not s.failure


def test_an_offset_answer_resumes_mid_blob():
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    out = s.feed(b"FS A10\r")
    blob = compress(msg.render())
    assert blob[10:] in out and blob[:10] not in out.replace(blob[10:], b"")
    assert b"\x00" + b"10" + b"\x00" in out     # the SOH header names the offset


def test_an_offset_at_the_end_of_the_blob_sends_the_framing_and_no_body():
    """`FS A<csize>` is the far end saying it already holds every byte, and the
    transfer it is owed is the header and an EOT over nothing. Refusing it
    ended the session over a message that had already arrived."""
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    blob = compress(msg.render())
    out = s.feed(f"FS A{len(blob)}\r".encode())
    off = str(len(blob)).encode()
    assert out == (bytes((SOH, len(off) + 7)) + b"hello\0" + off + b"\0"
                   + bytes((EOT, 0)))
    assert s.sent_mids == [msg.mid] and not s.failure


def test_a_turn_that_answered_our_proposals_starts_a_fresh_banner_budget():
    """The budget is per turn and the far end's answer ends ours: a gateway
    repeating its greeting while our proposals are on the air must not spend
    the allowance of the turn the mail arrives in."""
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    banner = b"RMS Trimode 1.4.2.0 Welcome to KC9GHZ\r" * 15
    out = s.feed(banner + b"FS +\r")
    assert out.startswith(bytes((SOH,))) and not s.failure
    s.feed(banner)
    assert not s.failure and s.stage == "their turn"


def test_an_fs_answer_count_mismatch_stops_the_session():
    msg = compose("W9SSJ", "KC9GHZ", "hello", "test\r\n", mid="MIDMIDMID123")
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    assert s.feed(b"FS YY\r") == b""
    assert "answers" in s.failure


# --------------------------------------------------------------------------- #
# The whole exchange, both roles, both directions.
def _deliver(s: B2FSession, data: bytes, chunk: int) -> bytes:
    """One transmission as the link hands it up: whole, or in `chunk`-byte
    deliveries the way an ARQ layer drains an over."""
    if not chunk:
        return s.feed(data)
    return b"".join(s.feed(data[i:i + chunk]) for i in range(0, len(data), chunk))


def _pump(a: B2FSession, b: B2FSession, *, chunk: int = 0) -> None:
    to_a, to_b = b.start(), a.start()
    for _ in range(200):
        if to_a:
            data, to_a = to_a, b""
            to_b += _deliver(a, data, chunk)
        elif to_b:
            data, to_b = to_b, b""
            to_a += _deliver(b, data, chunk)
        else:
            return
    raise AssertionError("the exchange did not settle")


def test_full_exchange_delivers_mail_both_ways():
    ours = compose("W9SSJ", "N0DX", "outbound", "from the caller\r\n" * 30)
    theirs = compose("N0DX", "W9SSJ", "inbound", "from the answering side\r\n")
    a = B2FSession("W9SSJ", role="calling", target="N0DX", outbox=[ours])
    b = B2FSession("N0DX", role="answering", target="W9SSJ", outbox=[theirs])
    _pump(a, b)
    assert a.done and b.done and not a.failure and not b.failure
    assert [m.render() for m in b.inbox] == [ours.render()]
    assert [m.render() for m in a.inbox] == [theirs.render()]
    assert a.sent_mids == [ours.mid] and b.sent_mids == [theirs.mid]


def test_empty_exchange_closes_cleanly():
    a = B2FSession("W9SSJ", role="calling", target="N0DX")
    b = B2FSession("N0DX", role="answering", target="W9SSJ")
    _pump(a, b)
    assert a.done and b.done and not a.failure and not b.failure
    assert not a.inbox and not b.inbox


def test_seven_messages_take_two_proposal_blocks():
    """The FBB block is at most five proposals; a fuller outbox must arrive
    across turns, not overflow one block."""
    msgs = [compose("W9SSJ", "N0DX", f"m{i}", f"body {i}\r\n") for i in range(7)]
    a = B2FSession("W9SSJ", role="calling", target="N0DX", outbox=msgs)
    b = B2FSession("N0DX", role="answering", target="W9SSJ")
    _pump(a, b)
    assert a.done and b.done and not a.failure and not b.failure
    assert [m.render() for m in b.inbox] == [m.render() for m in msgs]


# --------------------------------------------------------------------------- #
# A whole mailbox in one turn, the shape of the one that stalled.
#
# KE8LVA proposed four messages over VARA on 2026-09-08 and this station
# accepted all four; one arrived — 643 bytes, checksum and lzhuf verified —
# and then the gateway stopped sending and the link timed out with three still
# at the far end. Three never crossed, so the bodies below are ours. What is
# tonight's is the shape: four messages rendering to the four sizes the `FC`
# lines proposed, compressing to within 32 bytes of the four compressed sizes
# proposed beside them, offered in one block and delivered back to back.
_KE8LVA_DATE = datetime(2026, 9, 8, 19, 53, tzinfo=timezone.utc)
KE8LVA_0908 = [
    ("K2VD8PWQ1XLM", "Net report 0148Z and the new feedline",
     "31 check-ins on 3.945, best since 12 Mar. W7QJ/DM43 549 both ways 0148Z, "
     "KD9YRM/EN61 579, VE3XPQ/FN25 449 with QSB, N5TBM/EM12 339, K0WHY/DN70 "
     "559. Noise S1 after 2340Z, preamp off for the western half. New feed: "
     "34 m RG-213, SWR 1.31 at 3.750, 1.42 at 3.985, was 2.9 above 3.90 on the "
     "old RG-8X. Tuner bypassed, first time since 2022. Mast to 17 m on the "
     "19th, remeasure DM43 and FN25 then. Lost KB8QRX and AC9UFO at 0205Z to "
     "QRN off the lake, both back next week. Log is with W8ARC."),
    ("QH3ZP1L8MTRB", "Field day 27-28 Jun, kit list",
     "Mine: 3.5 kW genset, 20gal fuel, 2x K3S, AA-55 Zoom, 120 m LMR-400. "
     "Yours: 40/80 wires, 2x 12 m masts, 16 guy stakes, mallet, spare 30 A "
     "supply under the bench. Unowned: 2 logging laptops plus the LAN between "
     "them, settled in the field last year at 90 min. Awning, coffee, trailer "
     "booked Fri 0800. Site is the north field, gate code 4417, gravel only. "
     "Setup 1400 Fri, teardown 1600 Sun, 2 wanted for GOTA Sat AM. Class 3A, "
     "sections MI and OH, 5 W QRP bonus on 20 m. Reply with what you hold by "
     "the 14th. KE8LVA."),
    ("5XN7DKVA2WGC", "147.180 controller, RTC failure",
     "W8ARC 147.180 fault log, 6 Sep. ID drops 4x/hr, clock +6 s/day, tail "
     "beep intermittent since 22 Aug. Cell VL2330 leaked, 2 traces open at "
     "U9-3 and R41, RTC dead. Fit the RC-210 from the cabinet: 443.550 link, "
     "agreed 18 Jan, 2 U shelf, PL-259 tails 18 in, DB-25 breakout, 5/16 "
     "nutdriver, 25 ft ladder. PL 100.0, TX 147.780, RX 147.180, ID 9.5 min. "
     "Trip up Beecher Rd, 0730 at the gate, 2 ops, 90 min down. Free 19, 20, "
     "26, 27 Sep. W8PQR has key. Old board to N8QWD for the bench stock. "
     "KE8LVA EN82dj"),
    ("T9BFR4EJUMYD", "Traffic practice, Tue 2000 local",
     "W8ARC 147.180 traffic drill, Tue 2000-2055 local, 9, 16, 23, 30 Sep. "
     "1 radiogram each, passed and receipted, preamble in full, no shortcuts. "
     "Msg 4412-4431 ready, HXG for 8RN: 6 MI, 5 OH, 4 IN, 3 KY, 2 WV. Same 20 "
     "in the mailbox as B2F, pull any time, no sked. 4412, 4417, 4425, 4431 "
     "are book traffic, 6 lines ea; 4419 and 4428 carry ARL FIFTY SIX, check "
     "ARL 6. NCS 9 Sep KD8VYZ, 16 Sep W8PQR, 23 Sep AC8JTL, 30 Sep KE8LVA. "
     "Liaison to 8RN: N8XCV Mon, W8UKQ Thu. Bkup 443.550 -5, PL 110.9. "
     "Reports by the 1st to KE8LVA EN82dj, W8ARC NM since 2019."),
]

#: The `FC` sizes KE8LVA proposed, uncompressed and compressed.
KE8LVA_SIZES = [(643, 524), (662, 535), (651, 536), (699, 566)]

#: One VARA over of payload, the unit the four were delivered in.
OVER = 89


def _ke8lva_mailbox() -> list[Message]:
    return [compose("KE8LVA", "W9SSJ", subject, body + "\r\n", mid=mid,
                    date=_KE8LVA_DATE) for mid, subject, body in KE8LVA_0908]


def _offer_four(s: B2FSession, msgs: list[Message],
                chunk: int = OVER) -> list[bytes]:
    """The four proposed in one block and their bodies delivered back to back
    in `chunk`-byte deliveries. Returns what the session answered, silence
    dropped: a station that keys between two messages of one run is keying
    into the far end's transmission."""
    blobs = [compress(m.render()) for m in msgs]
    props = [f"FC EM {m.mid} {len(m.render())} {len(b)} 0"
             for m, b in zip(msgs, blobs)]
    out = [s.feed(_proposal_block(*props))]
    stream = b"".join(_body_stream(b, title=m.subject)
                      for m, b in zip(msgs, blobs))
    out += [s.feed(stream[i:i + chunk]) for i in range(0, len(stream), chunk)]
    return [answer for answer in out if answer]


def test_the_mailbox_fixture_is_the_shape_the_gateway_proposed():
    sizes = [(len(m.render()), len(compress(m.render())))
             for m in _ke8lva_mailbox()]
    assert [u for u, _ in sizes] == [u for u, _ in KE8LVA_SIZES]
    for (_, ours), (_, theirs) in zip(sizes, KE8LVA_SIZES):
        assert abs(ours - theirs) <= 32, "lzhuf takes these somewhere else"


def test_four_messages_cross_on_one_fs_with_nothing_keyed_between_them():
    """The whole mailbox in one turn: one `FS`, four bodies with no answer
    owed between them, and one line at the end. Anything this station sent
    between two of those messages would be keyed into the gateway's own
    transmission."""
    msgs = _ke8lva_mailbox()
    s = _connected()
    assert _offer_four(s, msgs) == [b"FS ++++\r", b"FF\r"]
    assert [m.render() for m in s.inbox] == [m.render() for m in msgs]
    assert not s.failure and not s.unfetched and s.in_flight is None


def test_the_line_after_four_is_our_own_block_when_we_hold_mail():
    """The other ending: with mail of ours still pending the turn closes on a
    proposal block rather than `FF`, and still on nothing in between."""
    mine = [compose("W9SSJ", "KC9GHZ", f"m{i}", f"body {i}\r\n") for i in range(6)]
    s = _connected(outbox=mine)                 # five proposals go out
    s.feed(b"FS -----\r")
    answers = _offer_four(s, _ke8lva_mailbox())
    assert answers[0] == b"FS ++++\r" and len(answers) == 2
    assert answers[1].startswith(b"FC EM ") and b"\rF> " in answers[1]


def test_the_send_leg_rides_the_same_session_and_closes_it():
    """`--send-with-fetch`, end to end: ours goes out on their `FS +`, the four
    come back, and the session closes on the `FQ` an empty far end draws."""
    ours = compose("W9SSJ", "KC9GHZ", "one out", "the send leg\r\n",
                   mid="OUTBOUND1234")
    s = _connected(outbox=[ours])
    assert compress(ours.render()) in s.feed(b"FS +\r")
    assert _offer_four(s, _ke8lva_mailbox()) == [b"FS ++++\r", b"FF\r"]
    assert s.feed(b"FF\r") == b"FQ\r"
    assert s.done and not s.failure and s.sent_mids == [ours.mid]


def test_a_mailbox_and_one_of_ours_cross_in_over_sized_deliveries():
    """Both sides of the same exchange against each other, with every
    transmission handed up an over at a time: our own back-to-back
    `_send_body` calls are what the far end is reading here."""
    theirs = _ke8lva_mailbox()
    ours = compose("W9SSJ", "KE8LVA", "one out", "the send leg\r\n" * 4)
    a = B2FSession("W9SSJ", role="calling", target="KE8LVA", outbox=[ours])
    b = B2FSession("KE8LVA", role="answering", target="W9SSJ", outbox=theirs)
    _pump(a, b, chunk=OVER)
    assert a.done and b.done and not a.failure and not b.failure
    assert [m.render() for m in a.inbox] == [m.render() for m in theirs]
    assert [m.render() for m in b.inbox] == [ours.render()]
