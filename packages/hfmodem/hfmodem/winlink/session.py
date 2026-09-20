# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The B2F forwarding session: link bytes in, link bytes out, mail either way.

One state machine for the exchange every Winlink gateway speaks above every
ARQ link this project builds — the modem is the transport, so the modem is not
in this file. Feed it what the link delivers and transmit what it returns:

    session = B2FSession("W9SSJ", role="calling", target="KC9GHZ")
    tx = session.start()
    ...
    tx = session.feed(rx_bytes)     # on every delivery from the link

The exchange, as the FBB forward protocol defines it and two live gateway
captures confirm it  [FBB doc, forward protocol; B2F spec]:

  * The **answering** station opens with its SID — `[WL2K-5.0-B2FWIHJM$]` off
    a real RMS — optionally a `;PQ:` secure-login challenge, and a prompt line
    ending `>`. The **calling** station replies with `;FW:`, its own SID, a
    `;PR:` response when it holds a password, a comment, and takes the first
    turn.
  * A turn is up to five `FC` proposals closed by `F> XX` — XX the two's
    complement of the summed proposal characters, CRs included — or `FF` for
    nothing to send, or `FQ` to end the session once the other side has also
    reported empty. The receiving side answers `FS` with one character per
    proposal — `+` `-` `=`, the set the FBB document requires, with Y/N/L,
    H/R/E and the A/! offset forms read as the same three — and each
    accepted message follows as binary: SOH header block, STX data blocks
    carrying the compressed body (lzhuf), EOT and a checksum making the data
    bytes sum to zero mod 256  [FBB doc, compressed forward].
  * Between those blocks a station may say whatever it likes: the free text of
    a welcome banner is not protocol and the protocol never said it was. So
    free text is logged and passed over wherever no block is open — before the
    first `FC` of a turn as readily as during the greeting — and refused
    inside one, where the `FC`…`F>` run is a checksummed unit and a line that
    is not part of it is framing lost rather than a gateway talking. Whatever
    the far end says in words, protocol or not, is kept in `transcript`, and
    so is every line this station speaks back: a gateway explains itself in
    English, a refusal is always in English, and a refusal is always a verdict
    on something we sent.

Everything the far end sends is checked — proposal-block checksum, block
framing, EOT checksum, the compressed body's own CRC and declared size, the
message structure — and the session declines to continue past a failure
(`failure` is set, nothing more is transmitted): these transports already
guarantee an error-free byte stream, so damage at this layer is structural,
not noise to press on through. Error-free is not gapless, though — an ARQ layer
that acknowledges a frame it never delivered splices the stream and says
nothing — and B2F has no sync word to hunt for, so a transfer that opens
mid-message is named and abandoned rather than guessed at.
"""
from __future__ import annotations

import hashlib
import re

from .lzhuf import LzhufError, compress, decompress
from .message import Message, MessageError

SOH, STX, EOT = 0x01, 0x02, 0x04

#: The name and version half of our SID. The operator may replace it — see
#: `[station] client_sid` — because a gateway may refuse a client type it does
#: not know, and only the operator can decide what their station announces.
CLIENT_SID = "HFM-0.1"

#: The other half, and NOT an operator setting. B2 forwarding, F compression,
#: H hierarchical addresses, M message IDs: each letter is a promise this file
#: keeps, so it is derived from the code and changes with it. Announcing a
#: capability we do not implement fails a session further in and much more
#: obscurely than being refused at the greeting.
_CAPABILITIES = "B2FHM$"

_MAX_PROPOSALS = 5              # per block  [FBB doc, forward protocol]
_BLOCK = 250                    # STX payload size; the length field is one byte

#: The second letter of every command the forward protocol defines. A line that
#: begins with one is the protocol talking; anything else is the station. The
#: test is `line[:2]`, the same two characters `_turn_line` dispatches on, so
#: the transcript reads a line exactly as the parser did.
_F_COMMANDS = "ABCDFQS>"

#: The proposal answers this station sends. The FBB document is normative on
#: the set — "FS line MUST have as many +,-,= signs as lines in the proposal"
#: [FBB doc, forward protocol] — and wl2k-go, whose SID this station wears,
#: sends exactly these three. Y/N/L mean the same to every reader that has been
#: seen, but they are not the form the specification defines.
_ACCEPT, _REJECT, _DEFER = "+", "-", "="

#: A `;PR:` answer and the rest of the line it is on.
_PR_ANSWER = re.compile(r"(?<=;PR:)[^\r\n]*")

#: What a byte that was never a character is shown as, U+FFFD.
_UNREADABLE = "\ufffd"

#: Longest line kept verbatim in the transcript. A greeting line is under 100
#: characters; something far longer is framing lost, not a station talking, and
#: the record stays readable either way.
_SAID_LIMIT = 200

# How much free text one turn of ours may absorb before the far end is talking
# nonsense rather than talking. A gateway greeting is four such lines at most —
# KY4RY's on 2026-08-14 was banner, minutes remaining, SID and prompt — and it
# arrived twice in the one turn, the gateway repeating itself while the link
# below stalled. Five greetings' worth is room for that and then some; past it
# the far end is not a station being chatty, and a session that says so beats
# one that quietly absorbs a desynced stream until the run's clock runs out.
_MAX_FREE_LINES = 20

# States. GREETING and ANSWER_HS are the two handshake sides; the rest is
# common to both roles.
_GREETING, _ANSWER_HS, _WAIT_FS, _THEIR_TURN, _RECEIVING, _DONE = range(6)

# The same states, named for a post-mortem: which stage a dead session was in
# separates a link that never carried the far end's greeting from a handshake
# that stalled from a transfer that broke.
_STAGES = {
    _GREETING: "awaiting greeting",
    _ANSWER_HS: "awaiting caller",
    _WAIT_FS: "awaiting proposal answer",
    _THEIR_TURN: "their turn",
    _RECEIVING: "receiving",
    _DONE: "closed",
}

# The Winlink secure-login salt, published through the implementations the
# B2F specification itself points to for this computation.
_SECURE_SALT = bytes((
    77, 197, 101, 206, 190, 249, 93, 200, 51, 243, 93, 237, 71, 94, 239, 138,
    68, 108, 70, 185, 225, 137, 217, 16, 51, 122, 193, 48, 194, 195, 198, 175,
    172, 169, 70, 84, 61, 62, 104, 186, 114, 52, 61, 168, 66, 129, 192, 208,
    187, 249, 232, 193, 41, 113, 41, 45, 240, 16, 29, 228, 208, 228, 61, 20))


def secure_login_response(challenge: str, password: str) -> str:
    """The 8-digit `;PR:` answer to an 8-digit `;PQ:` challenge  [B2F spec].

    MD5 over challenge + password + salt; the low 30 bits of the first four
    digest bytes, little-endian, printed in decimal and cut to the last eight
    digits. The password's case matters.

    NOT A CONVENTION ONLY WE READ, which is the failure class this project has
    lost days to. It is a line-for-line port of LA5NTA's wl2k-go `fbb/secure.go`
    — the library under Pat, itself ported from paclink-unix, where the salt was
    found — same 64 salt bytes in the same order, same concatenation, same 30
    bits, same little-endian assembly, same `%08d` and same last eight. Held to
    that implementation's OWN published known-answer vectors, somebody else's
    challenge against somebody else's password, by
    `tests/winlink/test_session.py::test_secure_login_published_vectors`, the
    case-sensitivity pair included. Five gateways have accepted an answer of
    ours, the most recent on 2026-08-23 on this code and this client SID.

    So silence after a `;PR:` is not evidence against the computation. Every
    refusal this station has drawn arrived as English in the gateway's very next
    transmission — a refusal is loud, and silence is not its signature.
    """
    digest = hashlib.md5(
        challenge.encode("ascii") + password.encode("ascii") + _SECURE_SALT
    ).digest()
    pr = digest[3] & 0x3F
    for i in (2, 1, 0):
        pr = (pr << 8) | digest[i]
    return f"{pr:08d}"[-8:]


def mask_pr(text: str) -> str:
    """`text` with every `;PR:` answer in it struck out, character for character.

    The response is thirty bits of MD5 over the challenge, the password and a salt
    this file publishes, and the `;PQ:` it answers is in the same record: kept
    whole, the pair is an offline search for the password. That an answer went out,
    and to which challenge, is what the record is for; the digits add nothing to it.

    The rule is the `;PR:` prefix and not the eight digits a well-formed answer
    carries, because not every record this masks is well formed — `rehear` masks a
    stream rejoined from the frames a receiver read, where a frame it never read
    leaves `;PR: 1234` behind and a shape rule prints what it did read. And `#` for
    character rather than one fixed line, because that same caller slices the
    joined stream back into frames by their own lengths.
    """
    return _PR_ANSWER.sub(lambda m: re.sub(r"\S", "#", m.group()), text)


def sid_line(client_sid: str = CLIENT_SID) -> str:
    """The SID a station with this name and version announces itself with."""
    return f"[{client_sid}-{_CAPABILITIES}]"


def _is_forward(line: str) -> bool:
    """Whether a line is one of the forward protocol's own commands rather than
    something a station said. `line[:2]`, the two characters `_turn_line`
    dispatches on, so the record classifies a line exactly as the parser did."""
    return line[:1] == "F" and line[1:2] in _F_COMMANDS


def _readable(line: str) -> str:
    """One line as a person can read it, or "" if it was never words.

    B2F above the link is ASCII, and the link below is not: VARA and PACTOR
    carry trailer bytes inside the frame CRC, a payload can be delivered
    partial, and a desynced stream is binary read as text. Every byte outside
    printable ASCII therefore becomes U+FFFD rather than a plausible accented
    letter — a record that shows where it could not read beats one a reader
    cannot tell from what the gateway typed — and a line of nothing but those
    is not speech and is dropped.
    """
    out = "".join(c if " " <= c <= "~" else _UNREADABLE for c in line).strip()
    if not out.strip(_UNREADABLE):
        return ""
    if len(out) > _SAID_LIMIT:
        out = f"{out[:_SAID_LIMIT]} [+{len(out) - _SAID_LIMIT} bytes]"
    return out


class _Outbound:
    """One message of ours, compressed once and carried through the turns."""

    def __init__(self, msg: Message):
        rendered = msg.render()
        self.mid = msg.mid
        self.title = msg.subject or "No title"
        self.blob = compress(rendered)
        self.usize = len(rendered)
        self.csize = len(self.blob)
        self.status = "pending"        # pending / sent / rejected / deferred
        self.offered = False           # its FC line went into a proposal block
        self.answer = ""               # the FS character the far end answered with
        self.blocks = 0                # STX blocks this session framed for it
        self.offset = 0


class _Inbound:
    """One proposal of theirs, from its FC line to its decompressed message."""

    def __init__(self, line: str):
        self.line = line
        self.ok = False
        self.msg_type = self.mid = ""
        self.usize = self.csize = 0
        parts = line[3:].split()
        if (line[:2] == "FC" and len(parts) >= 4
                and parts[0] in ("EM", "CM") and len(parts[1]) <= 12
                and parts[2].isdigit() and parts[3].isdigit()):
            self.ok = True
            self.msg_type, self.mid = parts[0], parts[1]
            self.usize, self.csize = int(parts[2]), int(parts[3])
        self.answer = ""


class B2FSession:
    """Both sides of a B2F mail exchange, sans I/O.

    `role` is "calling" (we brought the link up; the far end greets first and
    we take the first turn) or "answering" (the mirror — `start()` returns our
    greeting). `outbox` is the mail to offer; received mail lands in `inbox`
    as parsed :class:`Message` objects.

    The session is finished when `done` is true: either a clean `FF`/`FQ`
    close-out, or `failure` names what went wrong. Either way the link itself
    is still up — dropping it is the station's decision, not this layer's.

    `comment` sends the caller's `; <target> DE <mycall>` courtesy line, which
    the protocol never asked for. It costs 17 bytes and up, and a first turn
    carrying a proposal crosses a 89-byte VARA over with it and fits inside one
    without: every on-air first turn of this station that needed two overs drew
    silence from the gateway, 0 of 5, and every one that fitted one over was
    answered  [vara-evening-en63bc-0910/analysis/kc9]. A transport whose over
    is that tight may turn it off.
    """

    def __init__(self, mycall: str, *, role: str = "calling", target: str = "",
                 password: str = "", grid: str = "",
                 client_sid: str = CLIENT_SID, comment: bool = True,
                 outbox: list[Message] | None = None):
        assert role in ("calling", "answering")
        self.mycall = mycall.upper()
        self.role = role
        self.target = target.upper()
        self.password = password
        self.grid = grid
        self.client_sid = client_sid
        self.comment = comment
        self.inbox: list[Message] = []
        #: Every compressed block the far end delivered whole, MID and bytes,
        #: kept before anything is checked or decoded. The block is the only
        #: copy of what a gateway sent, and on 2026-08-16 WW2MI's CMS offered
        #: this station its first real message, the parser refused it, and the
        #: bytes went down with the link. What arrives is kept whether or not
        #: it can be read.
        self.received_blocks: list[tuple[str, bytes]] = []
        # Prepared bodies; transport acknowledgement and delivery are unknown.
        self.sent_mids: list[str] = []
        #: The far end said it had nothing to offer, in its own `FF`, rather
        #: than by proposing an empty block  [see _turn_line].
        self.remote_ff = False
        #: This station queued the close. Whether it reached the air is the
        #: transport's to know and it does not report back  [see summarize].
        self.our_fq = False
        self.remote_sid = ""
        self.challenge = ""
        self.failure = ""
        #: The far end's own words, when the failure was its verdict rather
        #: than this station's machinery giving up. Only a refusal says
        #: anything about mail we sent; every other failure is about what
        #: reached us afterwards.
        self.remote_refusal = ""
        self.log_lines: list[str] = []
        #: Every line either end spoke, in order: `(ours, text, protocol)`.
        self._script: list[tuple[bool, str, bool]] = []

        self._out = [_Outbound(m) for m in (outbox or [])]
        self._buf = bytearray()
        self._state = _GREETING if role == "calling" else _ANSWER_HS
        self._remote_no_msgs = False
        self._quit = False
        self._proposed: list[_Outbound] = []
        self._free_lines = 0
        self._in_props: list[_Inbound] = []
        self._in_sum = 0
        self._rxq: list[_Inbound] = []
        self._rx_data = bytearray()
        self._rx_sum = 0
        self._rx_head: bytes | None = None

    # ------------------------------------------------------------------ #
    @property
    def done(self) -> bool:
        return self._state == _DONE

    @property
    def stage(self) -> str:
        """Where the exchange has got to, in post-mortem terms."""
        return "failed" if self.failure else _STAGES[self._state]

    @property
    def sid(self) -> str:
        return sid_line(self.client_sid)

    @property
    def proposals(self) -> list["_Outbound"]:
        """Every message this session offered, with what the far end said about
        it. A refused or deferred one never reaches `sent_mids`, and an operator
        reading a summary needs to see it  [see client.summarize]."""
        return [m for m in self._out if m.offered]

    @property
    def unfetched(self) -> list[str]:
        """Mail this station answered `FS +` to and never received.

        A dead transfer takes the rest of its `FS` run with it, and the far
        end still holds every one: on 2026-08-23 WW2MI offered three and the
        session died on the first, so nothing was lost and nothing said so.
        """
        return [p.mid for p in self._rxq]

    @property
    def in_flight(self) -> tuple[str, int, int] | None:
        """The message a stopped transfer was in the middle of — `(mid, bytes
        in hand, bytes proposed)` — or None if none was open.

        A transfer that stops mid-message is the ordinary way an HF link dies,
        and the bytes already delivered are not nothing: on 2026-09-08 KE8LVA
        sent this station the head of its second message and 38 bytes of the
        535 that follow it, and the run reported four accepted, one received
        and three never got — true of the mailbox, not of the air. The SOH
        header is what separates this from a message the link never began —
        that one has no head, cannot be resumed and cannot be named as
        partial, which is what `unfetched` says instead.
        """
        if self._rx_head is None or not self._rxq:
            return None
        return (self._rxq[0].mid, len(self._delivered()), self._rxq[0].csize)

    @property
    def partial_block(self) -> bytes:
        """The compressed bytes of `in_flight`, as far as they arrived."""
        return self._delivered() if self.in_flight else b""

    def _delivered(self) -> bytes:
        """The open message's body so far: the STX blocks that closed, and the
        head of the one that did not. The last block is where a transfer stops
        — KE8LVA's second message got 38 bytes into a 250-byte block — and the
        bytes waiting on the rest of their block are as delivered as any."""
        buf, tail = self._buf, b""
        if len(buf) > 2 and buf[0] == STX:
            tail = bytes(buf[2:2 + (buf[1] or 256)])
        return bytes(self._rx_data) + tail

    @property
    def exchange(self) -> list[tuple[bool, str, bool]]:
        """Every line either end spoke, in order — `(ours, line, protocol)`, where
        `protocol` marks the forward-protocol commands `transcript` leaves out.

        The record has to hold what this station answered, and a session dies at
        exactly the lines `transcript` is not keeping. On 2026-08-29 KN4LQN's CMS
        answered a `;PM:` proposal with `*** [1] Unexpected response to proposal -
        Disconnecting`; establishing that our `FS` had gone out at all, never mind
        what it said, took byte-accounting against the frame log, because the one
        line that would have answered it was the one line dropped. Evidence
        collected and then discarded is the same defect the mail stage had.
        """
        tail = ("" if self._state in (_RECEIVING, _DONE)
                else _readable(self._buf.decode("latin-1")))
        return self._script + ([(False, tail, _is_forward(tail))] if tail else [])

    @property
    def transcript(self) -> list[tuple[bool, str]]:
        """The exchange in words, both sides, in the order they were spoken —
        `(ours, line)` per entry. `exchange` less the protocol's own punctuation,
        which is what makes this one read as a conversation.

        A gateway explains itself in English and the protocol has no field for
        it: the greeting, the minutes remaining, the `;PQ:` challenge, and —
        the one that costs a run — a refusal. KX8U's CMS answered our SID with
        `*** Unknown client types are not allowed on production servers`, every
        frame of it decoded and acknowledged, and the session record said
        `nothing moved` and nothing else. So the words are kept, including a
        last line the far end never got to terminate: a station
        that refuses and hangs up drops the link mid-sentence, which is exactly
        when what it was saying matters most.

        Ours stand beside them, one turn further on and for the same reason.
        WW2MI's CMS refused a login challenge on 2026-08-18 and said two
        attempts of a finite allowance remained; whether the session had spent
        one of them or two took a frame-level replay of the receive capture to
        answer, and the answer was one.
        """
        return [(ours, said) for ours, said, protocol in self.exchange
                if not protocol]

    @property
    def remote_text(self) -> list[str]:
        """The far end's half of `transcript`."""
        return [said for ours, said in self.transcript if not ours]

    @property
    def sent_text(self) -> list[str]:
        """This station's half."""
        return [said for ours, said in self.transcript if ours]

    def start(self) -> bytes:
        """What to transmit before anything has been heard."""
        if self.role != "answering":
            return b""
        who = " ".join(p for p in (self.target, "DE", self.mycall) if p)
        grid = f" ({self.grid})" if self.grid else ""
        return self._encode([self.sid, f"; {who}{grid}>"])

    def feed(self, data: bytes) -> bytes:
        """Consume bytes the link delivered; return bytes to transmit."""
        if self._state == _DONE:
            return b""
        self._buf += data
        out = bytearray()
        while self._state != _DONE:
            if self._state == _RECEIVING:
                made = self._binary_step(out)
            else:
                made = self._line_step(out)
            if not made:
                break
        return bytes(out)

    # -- plumbing ------------------------------------------------------- #
    def log(self, msg: str) -> None:
        """One event of the exchange's own progress. `MailClient` emits these as
        they are appended, so this is what a running log shows of a live session
        and the masking is not optional: the answering side puts a caller's whole
        handshake through here, `;PR:` answer and all."""
        self.log_lines.append(mask_pr(msg))

    def _fail(self, why: str) -> None:
        self.failure = why
        self._state = _DONE
        self.log(f"session failed: {why}")

    def _refused(self, line: str) -> None:
        self.remote_refusal = line
        self._fail(f"remote reported: {line}")

    def _encode(self, lines: list[str]) -> bytes:
        for line in lines:
            self._remember(line, ours=True)
        return "".join(line + "\r" for line in lines).encode("latin-1")

    def _remember(self, line: str, *, ours: bool = False) -> None:
        said = _readable(mask_pr(line))
        if said:
            self._script.append((ours, said, _is_forward(line)))

    def _free_text(self, line: str) -> None:
        """A line that is not protocol, where protocol was expected.

        The greeting has always read these as banner and carried on; a turn
        called them fatal, and on 2026-08-14 that cost besra its first payload
        frames off a second station — and this parser the first SID and `;PQ:`
        challenge any modem here had decoded for itself. KY4RY's
        `CMS via KY4RY >` prompt left the greeting state, and the very next
        line — `RMS Trimode 1.4.2.0 Welcome to KY4RY Hybrid RMS - Outer Banks
        NC (Starlink)`, the head of the greeting the gateway had begun
        repeating — killed a session whose bytes were perfect. Every RMS
        Trimode gateway opens this way, and a gateway may always say it twice.

        So: tolerated, up to `_MAX_FREE_LINES` in a turn, and never inside an
        open proposal block. Both halves matter. A greeting is a handful of
        lines and then the far end gets on with it, so a stream that never
        stops being free text is not a banner — it is the line framing lost,
        and it stops the session instead of stalling it. And a stray line
        between an `FC` and its `F>` is not a gateway talking: that run is one
        checksummed unit  [FBB doc, forward protocol], so a line inside it is
        damage, which this layer refuses on sight rather than waiting for the
        checksum to disagree.
        """
        if self._in_props:
            self._fail(f"unexpected line inside a proposal block: {line!r}")
            return
        self._free_lines += 1
        if self._free_lines > _MAX_FREE_LINES:
            self._fail(f"{self._free_lines} lines of free text where the "
                       f"protocol was owed, last {line!r}")
            return
        self.log(f"banner: {line}")

    def _line_step(self, out: bytearray) -> bool:
        """Extract one line from the buffer and act on it. False = need more."""
        for i, b in enumerate(self._buf):
            if b in (0x0D, 0x0A):
                line = self._buf[:i].decode("latin-1")
                del self._buf[:i + 1]
                if line.strip():
                    self._on_line(line.strip(), out)
                return True
        return False

    def _on_line(self, line: str, out: bytearray) -> None:
        self._remember(line)
        if self._state == _GREETING:
            self._greeting_line(line, out)
        elif self._state == _ANSWER_HS:
            self._answer_hs_line(line, out)
        elif self._state == _WAIT_FS:
            self._wait_fs_line(line, out)
        elif self._state == _THEIR_TURN:
            self._turn_line(line, out)

    # -- handshake, calling side ---------------------------------------- #
    def _greeting_line(self, line: str, out: bytearray) -> None:
        if line.startswith("[") and line.endswith("]"):
            self.remote_sid = line
            self.log(f"remote SID {line}")
        elif line.startswith(";PQ:"):
            self.challenge = line[4:].strip()
        elif line.endswith(">"):
            self._hs_reply(out)
        else:
            self._free_text(line)

    def _hs_reply(self, out: bytearray) -> None:
        if not self.remote_sid:
            self._fail("incomplete greeting: prompt received without a remote SID "
                       "— greeting bytes may be missing from the link")
            return
        if "B2" not in self.remote_sid.upper():
            self._fail(f"no B2 support in remote SID {self.remote_sid!r}")
            return
        lines = [f";FW: {self.mycall}", self.sid]
        if self.challenge and self.password:
            lines.append(
                f";PR: {secure_login_response(self.challenge, self.password)}")
        if self.comment:
            who = " ".join(p for p in (self.target, "DE", self.mycall) if p)
            grid = f" ({self.grid})" if self.grid else ""
            lines.append(f"; {who}{grid}")
        out += self._encode(lines)
        self._our_turn(out)

    # -- handshake, answering side --------------------------------------- #
    def _answer_hs_line(self, line: str, out: bytearray) -> None:
        if line.startswith("[") and line.endswith("]"):
            self.remote_sid = line
            self.log(f"remote SID {line}")
        elif line.startswith(";") or line.startswith("*"):
            self.log(f"handshake: {line}")
        elif line[:1] == "F":
            if "B2" not in self.remote_sid.upper():
                self._fail("caller sent a command before a B2 SID")
                return
            self._state = _THEIR_TURN
            self._turn_line(line, out)
        else:
            self._free_text(line)

    # -- our turn --------------------------------------------------------- #
    def _our_turn(self, out: bytearray) -> None:
        self._free_lines = 0            # the budget is per turn, not per session
        pending = [m for m in self._out if m.status == "pending"]
        if not pending:
            if self._remote_no_msgs:
                out += self._encode(["FQ"])
                self._quit = self.our_fq = True
                self._state = _DONE
                self.log("session over (FQ)")
            else:
                out += self._encode(["FF"])
                self._state = _THEIR_TURN
            return
        self._proposed = pending[:_MAX_PROPOSALS]
        for m in self._proposed:
            m.offered = True
        lines = [f"FC EM {m.mid} {m.usize} {m.csize} 0" for m in self._proposed]
        checksum = (-sum(sum(line.encode("ascii")) + 0x0D for line in lines)) & 0xFF
        out += self._encode(lines + [f"F> {checksum:02X}"])
        self._state = _WAIT_FS

    def _wait_fs_line(self, line: str, out: bytearray) -> None:
        if line.startswith(";"):
            return
        if line.startswith("*"):
            self._refused(line)
            return
        if not line.startswith("FS "):
            # An F command here is the far end answering the wrong question and
            # is held to it; anything else is a station talking, and a gateway
            # that repeats its greeting does so whether or not our proposals
            # are already on the air.
            if line[:1] == "F":
                self._fail(f"expected FS, got {line!r}")
            else:
                self._free_text(line)
            return
        answers = self._parse_fs(line[3:].strip())
        if answers is None:
            return
        for m, (verdict, offset, said) in zip(self._proposed, answers):
            m.answer = said
            if verdict == "accept":
                if offset > len(m.blob):
                    self._fail(f"offset {offset} beyond {m.mid}'s {len(m.blob)} bytes")
                    return
                self._send_body(m, offset, out)
                m.status = "sent"
                self.sent_mids.append(m.mid)
            elif verdict == "reject":
                m.status = "rejected"
            else:
                m.status = "deferred"
        self._proposed = []
        self._free_lines = 0
        self._state = _THEIR_TURN

    def _parse_fs(self, text: str) -> list[tuple[str, int, str]] | None:
        """Each proposal's verdict, its resume offset, and the character the far
        end answered with — which is what an operator reading the summary is
        told, rather than this file's name for it  [see client.summarize]."""
        answers: list[tuple[str, int, str]] = []
        i = 0
        while i < len(text):
            c = text[i]
            i += 1
            if c in "Yy+":
                answers.append(("accept", 0, c))
            elif c in "NnRr-":
                answers.append(("reject", 0, c))
            elif c in "Ll=Hh":
                answers.append(("defer", 0, c))
            elif c in "Aa!":
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                if j == i:
                    self._fail("offset answer with no offset")
                    return None
                answers.append(("accept", int(text[i:j]), c + text[i:j]))
                i = j
            elif c == " ":
                continue
            else:
                self._fail(f"unreadable FS answer {text!r}")
                return None
        if len(answers) != len(self._proposed):
            self._fail(f"{len(answers)} FS answers to {len(self._proposed)} proposals")
            return None
        return answers

    def _send_body(self, m: _Outbound, offset: int, out: bytearray) -> None:
        """SOH header block, STX data blocks, EOT checksum  [FBB doc].

        An offset at the end of the blob is the far end saying it already holds
        every byte, and what it is owed is the framing with nothing inside it:
        the header and an EOT over no data. Refusing that used to kill a
        session over a message that had already got there."""
        title = m.title.encode("ascii", "replace")[:80]
        off = str(offset).encode("ascii")
        out += bytes((SOH, len(title) + len(off) + 2)) + title + b"\0" + off + b"\0"
        data = m.blob[offset:]
        for i in range(0, len(data), _BLOCK):
            chunk = data[i:i + _BLOCK]
            out += bytes((STX, len(chunk))) + chunk
        m.blocks = -(-len(data) // _BLOCK)
        m.offset = offset
        out += bytes((EOT, (-sum(data)) & 0xFF))

    # -- their turn ------------------------------------------------------- #
    def _turn_line(self, line: str, out: bytearray) -> None:
        if line.startswith(";"):
            return
        if line.startswith("*"):
            # `_wait_fs_line` already treats a `*` here as fatal; a gateway
            # that rejects the body we just sent says so the same way, one
            # line later. On 2026-08-21 a CMS answered a completed EOT with
            # "Error check failed on receiving B2 message" and disconnected,
            # and this used to log it and keep waiting — the mid already
            # recorded in `sent_mids` stayed there with nothing to say it had
            # been refused.
            self._refused(line)
            return
        if line[:1] != "F":
            self._free_text(line)
            return
        cmd = line[:2]                  # a bare "F" is nobody's command below
        if cmd in ("FA", "FB", "FC", "FD"):
            self._in_sum += sum(line.encode("latin-1")) + 0x0D
            self._in_props.append(_Inbound(line))
        elif cmd == "FF":
            self._in_props, self._in_sum = [], 0
            self._remote_no_msgs = self.remote_ff = True
            self._our_turn(out)
        elif cmd == "FQ":
            self._quit = True
            self._state = _DONE
            self.log("remote closed the session (FQ)")
        elif cmd == "F>":
            self._end_of_proposals(line, out)
        else:
            self._fail(f"unknown command {line!r}")

    def _end_of_proposals(self, line: str, out: bytearray) -> None:
        ours = (-self._in_sum) & 0xFF
        self._in_sum = 0
        stated = line[2:].strip()
        try:
            theirs = int(stated, 16)
        except ValueError:
            self._fail(f"F> without a readable checksum: {line!r}")
            return
        if theirs != ours:
            self._fail(f"proposal checksum {stated} != computed {ours:02X}")
            return
        props, self._in_props = self._in_props, []
        # `FQ` claims both mailboxes are empty, so this tracks what the far end
        # HAS rather than what it sent: a block we answer `FS -` to, every mid
        # of it already in the inbox, leaves the far end as full as it was.
        self._remote_no_msgs = not props
        if not props:
            self._our_turn(out)
            return
        seen_now = set()
        received = {m.mid for m in self.inbox}
        for p in props:
            if not p.ok:
                p.answer = _DEFER        # a form we do not read; let it wait
            elif p.mid in received:
                p.answer = _REJECT
            elif p.mid in seen_now:
                p.answer = _DEFER        # duplicate within the block
            else:
                p.answer = _ACCEPT
                seen_now.add(p.mid)
        out += self._encode(["FS " + "".join(p.answer for p in props)])
        self._rxq = [p for p in props if p.answer == _ACCEPT]
        if self._rxq:
            self._begin_message()
            self._free_lines = 0
            self._state = _RECEIVING
        else:
            self._our_turn(out)

    # -- receiving accepted messages -------------------------------------- #
    def _begin_message(self) -> None:
        self._rx_data = bytearray()
        self._rx_sum = 0
        self._rx_head = None

    def _binary_step(self, out: bytearray) -> bool:
        """One framing element off the buffer. False = need more bytes."""
        buf = self._buf
        if not buf:
            return False
        lead = buf[0]
        if self._rx_head is None:
            if lead == 0x2A:                    # '*': an error line, not a block
                return self._error_line()
            if lead != SOH:
                # Every byte the link hands up is CRC-checked, so a message
                # that opens on anything but SOH is one whose opening bytes
                # never arrived. B2F carries no sync word to hunt for — SOH,
                # STX and EOT all occur freely inside a compressed body — so
                # naming the loss is the whole of what this layer can do.
                self._fail(f"{self._rxq[0].mid} began {lead:#04x}, not SOH — "
                           "the link delivered this message short of its head")
                return True
            if len(buf) < 2 or len(buf) < 2 + buf[1]:
                return False
            header = bytes(buf[2:2 + buf[1]])
            del buf[:2 + buf[1]]
            fields = header.split(b"\0")
            if len(fields) != 3 or fields[2] != b"" or not fields[1].isdigit():
                self._fail(f"unreadable SOH header {header!r}")
                return True
            if int(fields[1]) != 0:
                self._fail(f"message offered at offset {int(fields[1])}, "
                           "none was requested")
                return True
            self._rx_head = header
            self.log(f"receiving [{fields[0].decode('ascii', 'replace')}]")
            return True
        if lead == STX:
            if len(buf) < 2:
                return False
            n = buf[1] or 256
            if len(buf) < 2 + n:
                return False
            chunk = buf[2:2 + n]
            del buf[:2 + n]
            self._rx_data += chunk
            self._rx_sum = (self._rx_sum + sum(chunk)) & 0xFF
            return True
        if lead == EOT:
            if len(buf) < 2:
                return False
            cks = buf[1]
            del buf[:2]
            self._finish_message(cks, out)
            return True
        self._fail(f"expected STX or EOT, got byte {lead:#04x}")
        return True

    def _error_line(self) -> bool:
        for i, b in enumerate(self._buf):
            if b in (0x0D, 0x0A):
                line = self._buf[:i].decode("latin-1")
                self._remember(line)
                self._refused(line)
                return True
        return False

    def _finish_message(self, cks: int, out: bytearray) -> None:
        prop = self._rxq[0]
        self.received_blocks.append((prop.mid, bytes(self._rx_data)))
        self._rx_head = None            # the EOT closed it, whatever the verdict
        if (self._rx_sum + cks) & 0xFF != 0:
            self._fail(f"EOT checksum failed for {prop.mid}")
            return
        if len(self._rx_data) != prop.csize:
            self._fail(f"{prop.mid}: {len(self._rx_data)} bytes against a "
                       f"proposed {prop.csize}")
            return
        try:
            msg = Message.parse(
                decompress(bytes(self._rx_data), expected_size=prop.usize))
        except (LzhufError, MessageError) as e:
            self._fail(f"{prop.mid}: {e}")
            return
        self.inbox.append(msg)
        self.log(f"received {prop.mid}: {msg.subject!r}")
        self._rxq.pop(0)
        if self._rxq:
            self._begin_message()
        else:
            self._our_turn(out)
