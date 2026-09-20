# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The Winlink message structure, against a real message.

The external fixture is a genuine Winlink message — eleven headers, an
ISO-8859-1 body, a jpeg attachment. Parsing it and rendering it back must be
byte-exact: the `Body:`/`File:` sizes are the delimiters and each section's
trailing CRLF sits outside its count, which is exactly the layout measured off
the file. The planted cases then lie about a size and must go red.
"""
from __future__ import annotations

import pytest

from hfmodem.tests.winlink import corpora
from hfmodem.winlink import Attachment, Message, MessageError, compose


def test_real_message_parses():
    plain, _ = corpora.real_pair()
    msg = Message.parse(plain)
    assert msg.mid == "LPE5NXDVLVSQ"
    assert msg.subject == "73 fra Brekke"
    assert msg.sender == "LA5NTA"
    assert msg.recipients == ["LA4TTA"]
    assert len(msg.body) == 104
    (att,) = msg.attachments
    assert att.name == "1469042410710.jpg"
    assert len(att.data) == 31028
    assert att.data[:2] == b"\xff\xd8"          # a jpeg, as the name says


def test_real_message_renders_byte_exact():
    plain, _ = corpora.real_pair()
    assert Message.parse(plain).render() == plain


# --------------------------------------------------------------------------- #
def _wire(body: bytes, files: list[tuple[int, str, bytes]] = ()) -> bytes:
    head = f"Mid: TESTTESTTEST\r\nBody: {len(body)}\r\n"
    for size, name, _ in files:
        head += f"File: {size} {name}\r\n"
    out = head.encode() + b"\r\n" + body + b"\r\n"
    for _, _, data in files:
        out += data + b"\r\n"
    return out


def test_compose_and_parse_round_trip():
    msg = compose("W9SSJ", "KC9GHZ", "test traffic", "hello from the bench\r\n",
                  attachments=[Attachment("blob.bin", bytes(range(64)))])
    back = Message.parse(msg.render())
    assert back == msg
    assert len(back.mid) == 12 and back.mid.isalnum() and back.mid.isupper()
    assert back.get("Type") == "Private"
    assert back.get("Body") == "22"


def test_multiple_recipients_and_attachment_names_with_spaces():
    wire = _wire(b"hi", [(3, "two words.txt", b"abc")])
    msg = Message.parse(wire)
    assert msg.attachments[0].name == "two words.txt"
    assert msg.render() == wire


def test_iso_8859_1_headers_survive_the_round_trip():
    """Headers are US-ASCII by specification, but real clients put raw
    ISO-8859-1 in subjects; what parses must render back byte-exact."""
    wire = b"Mid: X\r\nSubject: hyggelig kveldstur p\xe5 Hausdal\r\nBody: 4\r\n\r\nhi\r\n"
    assert Message.parse(wire).render() == wire


def test_the_ww2mi_message_parses():
    """The message WW2MI's CMS put on the air on 2026-08-16 and this parser
    refused: no attachments, and the stream stops where `Body:` stops."""
    _, plain = corpora.ww2mi()
    msg = Message.parse(plain)
    assert msg.mid == "LJ2AJE2IHO9B"
    assert msg.subject == "testing without the magic subject line"
    # The sender is a live address and the recipient a live callsign, so both are
    # read back out of the specimen rather than written here: this file ships and
    # the specimen does not.
    headers = dict(Message.parse(plain).headers)
    assert msg.sender == headers["From"]
    assert msg.recipients == [headers["To"]]
    assert msg.body == b"<no message body>"
    assert not msg.attachments


def test_the_ww2mi_message_renders_byte_exact():
    _, plain = corpora.ww2mi()
    assert Message.parse(plain).render() == plain


def test_a_body_that_ends_the_message_needs_no_crlf():
    wire = b"Mid: LJ2AJE2IHO9B\r\nBody: 7\r\n\r\nhello\r\n"
    assert Message.parse(wire).body == b"hello\r\n"


def test_a_last_attachment_that_ends_the_message_needs_no_crlf_either():
    wire = _wire(b"hi", [(3, "a.bin", b"abc")]).removesuffix(b"\r\n")
    msg = Message.parse(wire)
    assert msg.body == b"hi" and msg.attachments[0].data == b"abc"


def test_a_section_with_another_behind_it_still_needs_its_crlf():
    """The CRLF separates sections; only the one with nothing behind it goes."""
    wire = b"Mid: X\r\nBody: 2\r\nFile: 3 a.bin\r\n\r\nhiabc\r\n"
    with pytest.raises(MessageError, match="CRLF"):
        Message.parse(wire)


def test_body_size_is_the_delimiter():
    """A body that contains what looks like an attachment boundary is carried
    intact: only the declared count delimits it."""
    body = b"1: first\r\n\r\n2: second"
    wire = _wire(body)
    assert Message.parse(wire).body == body


# --------------------------------------------------------------------------- #
# Planted lies, each of which must go red.
def test_a_body_size_lie_goes_red():
    wire = _wire(b"twelve bytes")
    wrong = wire.replace(b"Body: 12", b"Body: 13")
    with pytest.raises(MessageError):
        Message.parse(wrong)


def test_an_attachment_size_lie_goes_red():
    wire = _wire(b"hi", [(3, "a.bin", b"abc")])
    wrong = wire.replace(b"File: 3 ", b"File: 2 ")
    with pytest.raises(MessageError):
        Message.parse(wrong)


def test_trailing_garbage_goes_red():
    with pytest.raises(MessageError, match="after the last"):
        Message.parse(_wire(b"hi") + b"stray")


def test_missing_blank_line_goes_red():
    with pytest.raises(MessageError, match="blank line"):
        Message.parse(b"Mid: X\r\nBody: 2\r\nhi\r\n")


def test_a_headerless_blob_goes_red():
    with pytest.raises(MessageError):
        Message.parse(b"\xff\xd8\xff\xdb\r\n\r\n\r\n")


# --------------------------------------------------------------------------- #
# Line endings. The structure is CRLF throughout, headers and body alike, and
# `compose` took a file's bytes as the body untouched: the first message this
# station ever transmitted (2026-08-21, WW2MI over ARDOP) carried a Unix body
# and came back "*** Error check failed on receiving B2 message. [Received data
# stream not a correct format] - Disconnecting". Every real message in the
# corpus is CRLF; that one was the only one that was not.
def test_a_composed_body_is_crlf():
    msg = compose("W9SSJ", "SMTP:op@example.net", "unix body",
                  b"first line\nsecond line\n")
    assert msg.body == b"first line\r\nsecond line\r\n"
    assert msg.get("Body") == str(len(msg.body))


def test_a_body_that_is_already_crlf_is_left_alone():
    msg = compose("W9SSJ", "SMTP:op@example.net", "dos body",
                  b"first line\r\nsecond line\r\n")
    assert msg.body == b"first line\r\nsecond line\r\n"


def test_no_real_message_body_carries_a_bare_lf():
    for plain in (corpora.real_pair()[0], corpora.ww2mi()[1]):
        body = Message.parse(plain).body
        assert b"\n" not in body.replace(b"\r\n", b"")
