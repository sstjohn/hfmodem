# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The Winlink message: what a B2 forwarding transfer actually carries.

A message is US-ASCII headers, a body, and zero or more binary attachments,
in one byte string  [B2F spec, message structure]:

    Mid: LPE5NXDVLVSQ\r\n
    Date: 2016/07/20 19:21\r\n
    ...
    Body: 104\r\n
    File: 31028 1469042410710.jpg\r\n
    \r\n
    <body, exactly 104 bytes>\r\n
    <attachment, exactly 31028 bytes>\r\n

The sizes in `Body:` and `File:` headers are the delimiters — the body and
each attachment end where their declared count ends, followed by one CRLF
that is not part of the count. That layout is measured off a real Winlink
message and round-trips byte-exact (tests/winlink/test_message.py).

A body with no attachments behind it is the exception: the message stops
where `Body:` stops, with no CRLF at all. `la5nta/wl2k-go` writes exactly
that and takes end of stream in place of any section's CRLF on the way back
in; WW2MI's CMS wrote exactly that on 2026-08-16, a 603-byte message whose
body is the seventeen characters `<no message body>`, and this parser
demanded the CRLF and refused it — `message ends inside the body`, the first
inbound message from a real gateway. Five the parser had already accepted sit
in the mail log, the newest 2026-08-13, and it only writes what parsed. Both real messages this tree
holds now round-trip byte-exact, one of each shape.

Header order and key case vary by author (gateways write `Mid`, some clients
alphabetise), so parsing keeps the header list as it arrived and matches keys
case-insensitively; `render` writes back exactly what was parsed.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone

_MID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class MessageError(ValueError):
    """The bytes are not a well-formed Winlink message."""


def new_mid() -> str:
    """A fresh message ID: 12 characters, the B2F proposal field's maximum."""
    return "".join(secrets.choice(_MID_ALPHABET) for _ in range(12))


@dataclass
class Attachment:
    name: str
    data: bytes


@dataclass
class Message:
    headers: list[tuple[str, str]] = field(default_factory=list)
    body: bytes = b""
    attachments: list[Attachment] = field(default_factory=list)

    # -- header access ----------------------------------------------------
    def get(self, key: str, default: str = "") -> str:
        for k, v in self.headers:
            if k.lower() == key.lower():
                return v
        return default

    def get_all(self, key: str) -> list[str]:
        return [v for k, v in self.headers if k.lower() == key.lower()]

    @property
    def mid(self) -> str:
        return self.get("Mid")

    @property
    def subject(self) -> str:
        return self.get("Subject")

    @property
    def sender(self) -> str:
        return self.get("From")

    @property
    def recipients(self) -> list[str]:
        return self.get_all("To") + self.get_all("Cc")

    # -- wire form --------------------------------------------------------
    @classmethod
    def parse(cls, data: bytes) -> "Message":
        end = data.find(b"\r\n\r\n")
        if end < 0:
            raise MessageError("no blank line after the headers")
        headers: list[tuple[str, str]] = []
        for raw in data[:end].split(b"\r\n"):
            key, sep, value = raw.partition(b": ")
            if not sep or not key:
                raise MessageError(f"unparseable header line {raw[:40]!r}")
            # Headers are US-ASCII by specification, but real clients put raw
            # ISO-8859-1 in subjects; latin-1 is lossless in both directions,
            # so what parses renders back byte-exact.
            headers.append((key.decode("latin-1"), value.decode("latin-1")))
        msg = cls(headers=headers)

        body_len = _size(msg.get("Body"), "Body")
        files = [(_size(v.split(" ", 1)[0], "File"),
                  v.split(" ", 1)[1] if " " in v else "")
                 for v in msg.get_all("File")]

        pos = end + 4
        msg.body, pos = _take(data, pos, body_len, "body")
        for size, name in files:
            blob, pos = _take(data, pos, size, f"attachment {name!r}")
            msg.attachments.append(Attachment(name, blob))
        if pos != len(data):
            raise MessageError(f"{len(data) - pos} bytes after the last attachment")
        return msg

    def render(self) -> bytes:
        out = bytearray()
        for k, v in self.headers:
            out += f"{k}: {v}\r\n".encode("latin-1", "replace")
        out += b"\r\n"
        out += self.body
        if self.attachments:
            out += b"\r\n"
        for att in self.attachments:
            out += att.data + b"\r\n"
        return bytes(out)


def _size(text: str, what: str) -> int:
    if not text.isdigit():
        raise MessageError(f"{what} header does not carry a size: {text!r}")
    return int(text)


def _take(data: bytes, pos: int, n: int, what: str) -> tuple[bytes, int]:
    end = pos + n
    if end > len(data):
        raise MessageError(f"message ends inside the {what}")
    if end == len(data):
        return data[pos:end], end
    if data[end:end + 2] != b"\r\n":
        raise MessageError(f"the {what} is not followed by CRLF where its "
                           "declared size ends")
    return data[pos:end], end + 2


def _address(addr: str) -> str:
    """An address the far end can route. A bare `user@host` is ambiguous where
    hierarchical addressing is in the SID: it reads as a callsign at a BBS
    rather than as internet mail, so the proto that says which one goes in."""
    return addr if ":" in addr or "@" not in addr else f"SMTP:{addr}"


def compose(sender: str, to: str | list[str], subject: str, body: bytes | str,
            *, attachments: list[Attachment] | None = None, mbo: str = "",
            mid: str = "", date: datetime | None = None) -> Message:
    """A new outbound message in the header order gateways themselves write."""
    if isinstance(body, str):
        body = body.encode("latin-1", "replace")
    # CRLF THROUGHOUT, AND THE COUNT IS TAKEN AFTER. A body composed from a Unix
    # file goes out with CRLF headers and bare-LF lines, and on 2026-08-22 a CMS
    # answered one with "Error check failed on receiving B2 message. [Received
    # data stream not a correct format]" and disconnected -- after ACKing every
    # frame, with the EOT checksum and the lzhuf CRC both verifying on the wire.
    # The two messages this station has received off the air carry no bare LF.
    body = body.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    date = date or datetime.now(timezone.utc)
    headers = [("Mid", mid or new_mid()),
               ("Date", date.strftime("%Y/%m/%d %H:%M")),
               ("Type", "Private"),
               ("From", _address(sender))]
    headers += [("To", _address(t)) for t in ([to] if isinstance(to, str) else to)]
    headers += [("Subject", subject),
                ("Mbo", mbo or sender),
                ("Body", str(len(body)))]
    attachments = list(attachments or [])
    headers += [("File", f"{len(a.data)} {a.name}") for a in attachments]
    return Message(headers=headers, body=body, attachments=attachments)
