# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The other direction: a message of ours out to the gateway, over PACTOR-3.

`test_b2f_loopback_ws8eoc` drives the whole exchange with the gateway's mailbox
full and ours empty. Its harness is this file's too -- the same `PtcHost`, the
same `PactorArq`, the same real `MailClient` around a real `B2FSession`, the
same scripted far end keying the packet shapes `working/pactor3-header-0913`
watched WS8EOC key. What changes here is whose mailbox is full: we open the
turn with `FC EM`, the gateway answers `FS`, and the SOH/STX/EOT body goes out
over a changeover this end asked for.

The far end's words are KB5LZK's own, 2026-09-12 13:38Z, the one complete send
this station has on record and the one the operator's own inbox confirmed:

    we:   FC EM Q2J6KMZG9FEX 355 283 0
    we:   F> 76
    peer: FS Y
    peer: FF
    we:   FQ

`FS Y`, not `FS +`: the form a real RMS answered with, which the parser reads
as the same acceptance the FBB document spells `+`.

The gateway's `FF` is triggered by the body itself, byte for byte, so a body
that crossed wrong draws no answer at all and the scene stalls instead of
passing.

    python -m pytest hfmodem/tests/shrike/test_b2f_loopback_send_ws8eoc.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import numpy as np

from hfmodem.shrike import arq, spec
from hfmodem.shrike.compress import SB, transparent
from hfmodem.shrike.spec import IDLE
from hfmodem.tests.shrike.test_b2f_loopback_ws8eoc import (Ws8eoc, body_stream,
                                                           linked, run)
from hfmodem.winlink import Message, compose, compress, decompress
from hfmodem.winlink.session import EOT, SOH, STX

#: KB5LZK's answer to our proposal, and the form the parser has to read as an
#: acceptance. Every other live gateway on file has proposed rather than
#: answered, so this is the only shape taken off the air.
ACCEPT = b"FS Y\r"

#: The turn back, after the body. WW2MI's shape: the answer in one over and the
#: far end's own empty mailbox in the next, not both at once.
EMPTY = b"FF\r"

SEND_CYCLES = 2000


def outbound(body: bytes = b"one message, out over PACTOR-3.\r\n",
             subject: str = "EN54 send check",
             mid: str = "B2FSEND0001") -> Message:
    """The message we hold for the CMS, addressed where E1 was addressed."""
    return compose("W9SSJ", "SMTP:op@example.net", subject, body, mid=mid)


def binary_outbound() -> Message:
    """A body whose compressed block carries both bytes the character stream
    keeps for itself, so the escape has to survive the three-byte changeover
    field as well as every field behind it.

    THE DATE IS PART OF THE SPECIMEN. `compose` stamps the current minute into
    the header, so the message this returns -- and the compressed block the
    scene asserts on -- changed every sixty seconds, and 76 minutes in 600 carry
    no 0x1E through the compressor at all. Those are the minutes the scene
    failed in, on its own first line, before any of the modem ran. The seed and
    the MID were already pinned for this reason; the clock was the one input
    left loose.
    """
    body = np.random.default_rng(23).integers(0, 256, 400,
                                              dtype=np.uint8).tobytes()
    return compose("W9SSJ", "SMTP:op@example.net", "outbound binary", body,
                   mid="B2FSENDBIN1",
                   date=datetime(2026, 9, 10, 18, 32, tzinfo=timezone.utc))


def send_script(blob: bytes, title: bytes, *, answer: bytes = ACCEPT,
                after: bytes = EMPTY):
    """What the gateway says, and what of ours makes it say it.

    Two triggers: the `F>` that closes our proposal block draws the answer, and
    the body -- the whole of it, byte for byte -- draws the turn back. A body
    delivered wrong never matches, so the exchange stalls where a weaker
    trigger would have let it close green.
    """
    return [(b"F> ", answer), (body_stream(blob, title), after)]


class Volunteering(Ws8eoc):
    """The gateway with a turn of its own to take.

    A declined proposal leaves this end with nothing to say, so whatever comes
    next is not an answer to anything of ours: the far end takes the channel
    back and reports its own mailbox with nothing in our stream to prompt it.
    """

    def __init__(self, script, *, unprompted: tuple[bytes, ...] = (), **kw):
        super().__init__(script, **kw)
        self.unprompted = list(unprompted)

    def _take(self) -> None:
        if not self.out and self.unprompted:
            self.out += transparent(self.unprompted.pop(0))
        super()._take()


def gateway_read(rx: bytes) -> tuple[bytes, int, bytes]:
    """The body as the far end's own reader takes it apart: title, offset and
    the compressed block, with the EOT checksum held to zero mod 256."""
    i = rx.index(bytes((SOH,)))
    head = rx[i + 2:i + 2 + rx[i + 1]]
    title, offset, tail = head.split(b"\0")
    assert tail == b"", head
    i += 2 + rx[i + 1]
    blob = bytearray()
    while rx[i] == STX:
        n = rx[i + 1] or 256
        blob += rx[i + 2:i + 2 + n]
        i += 2 + n
    assert rx[i] == EOT, rx[i]
    assert (sum(blob) + rx[i + 1]) & 0xFF == 0
    return title, int(offset), bytes(blob)


def sending(*, msg: Message | None = None, answer: bytes = ACCEPT,
            after: bytes = EMPTY, gateway=Ws8eoc, cycles: int = SEND_CYCLES,
            **kw):
    msg = msg or outbound()
    blob = compress(msg.render())
    gw = gateway(send_script(blob, msg.subject.encode("ascii"), answer=answer,
                             after=after), **kw)
    host, session = linked(gw, outbox=[msg])
    return msg, blob, host, session, gw, run(host, gw, cycles=cycles)


# --------------------------------------------------------------------------- #
# Scene 7 -- the send KB5LZK answered, end to end.

def test_scene7_a_message_of_ours_reaches_the_gateway_byte_exact():
    msg, blob, host, session, gw, cycles = sending()

    assert cycles, "\n".join(gw.tape[-20:])
    assert not session.failure, session.failure
    assert session.done and session.stage == "closed"
    assert session.sent_mids == [msg.mid]

    title, offset, delivered = gateway_read(bytes(gw.rx))
    assert (title, offset) == (b"EN54 send check", 0)
    assert delivered == blob
    assert Message.parse(decompress(delivered, expected_size=len(msg.render()))
                         ).render() == msg.render()


def test_scene7_the_proposal_is_the_one_the_far_end_checksums():
    msg, blob, host, session, gw, cycles = sending()
    line = f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"
    checksum = (-(sum(line.encode()) + 0x0D)) & 0xFF
    rx = bytes(gw.rx)
    assert rx.startswith(b";FW: W9SSJ")
    assert f"{line}\rF> {checksum:02X}\r".encode() in rx


def test_scene7_the_transcript_is_the_one_kb5lzk_answered():
    """The five lines of KB5LZK's completed send, in order, with the free text
    of the greeting taken out."""
    msg, blob, host, session, gw, cycles = sending()
    line = f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0"
    checksum = (-(sum(line.encode()) + 0x0D)) & 0xFF
    assert [(ours, said) for ours, said, protocol in session.exchange
            if protocol] == [(True, line), (True, f"F> {checksum:02X}"),
                             (False, "FS Y"), (False, "FF"), (True, "FQ")]


def test_scene7_the_body_goes_out_behind_a_changeover_of_ours():
    """Two stints of ours and two changeover packets, not three: the proposal
    is written in the same `feed` as the login and rides the same turn, which
    is the shape the session's `comment` note is about."""
    msg, blob, host, session, gw, cycles = sending()
    ours = [b for b in gw.bursts if b[3]]
    assert [b[1] for b in ours] == [b";FW", bytes((SOH,)) + b"\x12E"]
    assert bytes(gw.rx).index(b"FC EM") < bytes(gw.rx).index(bytes((SOH,)))
    counters = [b[2] & spec.STATUS_SEQ for b in gw.bursts]
    assert counters == [i % 4 for i in range(len(gw.bursts))], \
        "the peer accepted every packet of the body first time"


def test_scene7_a_compressed_body_crosses_the_changeover_field_escaped():
    """The three-byte changeover field is the head of the SOH block, and 0x1C
    and 0x1E inside the block go out as the supervisor sequence."""
    msg = binary_outbound()
    blob = compress(msg.render())
    assert blob.count(SB) and blob.count(IDLE), "the specimen carries both"

    msg, blob, host, session, gw, cycles = sending(msg=msg)
    assert cycles, "\n".join(gw.tape[-20:])
    assert not session.failure, session.failure
    assert gateway_read(bytes(gw.rx))[2] == blob


# --------------------------------------------------------------------------- #
# Scene 8 -- the proposal the gateway does not want.

@pytest.mark.parametrize("answer", [b"FS N\r", b"FS -\r", b"FS =\r"])
def test_scene8_a_declined_proposal_closes_the_session_clean(answer):
    """`-` is the FBB document's form, `N` the one a live RMS writes, `=` the
    defer wl2k-go's own tests expect. None of them is a body, and none of them
    is a failure."""
    msg = outbound()
    gw = Volunteering([(b"F> ", answer)], unprompted=(EMPTY,))
    host, session = linked(gw, outbox=[msg])
    cycles = run(host, gw, cycles=SEND_CYCLES)

    assert cycles, "\n".join(gw.tape[-20:])
    assert not session.failure, session.failure
    assert session.done and session.stage == "closed"
    assert session.sent_mids == []
    assert bytes((SOH,)) not in bytes(gw.rx), "no body followed a refusal"
    assert session.exchange[-1] == (True, "FQ", True)


def test_scene8_a_declined_proposal_is_not_a_refusal_of_ours():
    msg = outbound()
    gw = Volunteering([(b"F> ", b"FS N\r")], unprompted=(EMPTY,))
    host, session = linked(gw, outbox=[msg])
    run(host, gw, cycles=SEND_CYCLES)
    assert not session.remote_refusal
    assert not any("failed" in line for line in session.log_lines)


# --------------------------------------------------------------------------- #
# Scene 9 -- the body across a turn this end did not choose.

def breaks_in_partway_through(gw: Ws8eoc, after: int) -> list[int]:
    """The far end taking the channel back in the middle of our body.

    An IRS may ask for the turn at any packet boundary, and what this end owes
    it is the channel with the rest of the message still in the link's buffer.
    The gateway has nothing to say when it gets there; the scene is the role
    flip and what survives it.
    """
    taken, acknowledge, take = [], gw._acknowledge, gw._take

    def interrupt():
        i = bytes(gw.rx).find(bytes((SOH,)))
        if not taken and 0 <= i and len(gw.rx) - i > after:
            taken.append(len(gw.rx) - i)
            take()
        else:
            acknowledge()

    gw._acknowledge = interrupt
    return taken


def test_scene9_a_body_that_spans_a_break_in_of_the_gateways_crosses_whole():
    msg = binary_outbound()
    blob = compress(msg.render())
    gw = Ws8eoc(send_script(blob, b"outbound binary"))
    host, session = linked(gw, outbox=[msg])
    taken = breaks_in_partway_through(gw, 120)
    cycles = run(host, gw, cycles=SEND_CYCLES)

    assert taken, "the gateway never reached the middle of the body"
    assert cycles, "\n".join(gw.tape[-20:])
    assert not session.failure, session.failure
    assert session.sent_mids == [msg.mid]
    assert gateway_read(bytes(gw.rx))[2] == blob


# --------------------------------------------------------------------------- #
# Scene 10 -- the answer this end did not read.

def the_answer_is_not_decoded(gw: Ws8eoc, cycles: int) -> None:
    """The gateway's proposal answer keyed and not read.

    The changeover packet carries `FS ` -- three bytes and the link with it --
    and what goes missing is the rest of the line behind it. The gateway holds
    that field under the counter it was keyed with and keys it again, and this
    end sits in `awaiting proposal answer` spending the budget it keeps for a
    peer that has stopped saying anything it can read.
    """
    read = gw._read

    def deaf_behind_the_changeover(payload, status):
        step = gw._step
        read(payload, status)
        if step == 0 and gw._step:
            gw.deaf.update(range(1, cycles + 1))

    gw._read = deaf_behind_the_changeover


def test_scene10_an_answer_read_late_costs_cycles_and_not_the_body():
    msg = outbound()
    blob = compress(msg.render())
    gw = Ws8eoc(send_script(blob, b"EN54 send check"))
    host, session = linked(gw, outbox=[msg])
    the_answer_is_not_decoded(gw, 4)
    cycles = run(host, gw, cycles=SEND_CYCLES)

    assert cycles, "\n".join(gw.tape[-20:])
    assert any("[not decoded]" in line for line in gw.tape)
    assert not session.failure, session.failure
    assert session.sent_mids == [msg.mid]
    assert gateway_read(bytes(gw.rx))[2] == blob


def test_scene10_an_answer_that_never_arrives_spends_the_budget_and_says_goodbye():
    """NEGATIVE CONTROL. Past `max_retries` the far end is not slow, and a link
    that says goodbye beats one holding a channel for an answer to a proposal
    the gateway has stopped being able to deliver."""
    msg = outbound()
    blob = compress(msg.render())
    gw = Ws8eoc(send_script(blob, b"EN54 send check"))
    host, session = linked(gw, outbox=[msg])
    the_answer_is_not_decoded(gw, 40)

    assert run(host, gw, cycles=SEND_CYCLES) == 0
    assert session.stage == "awaiting proposal answer"
    assert session.sent_mids == []
    assert bytes((SOH,)) not in bytes(gw.rx)
    assert host.arq._qrt_pending or host.arq.state is not arq.State.CONNECTED


# --------------------------------------------------------------------------- #
# Scene 11 -- the answer and the far end's own empty mailbox together.

def test_scene11_an_answer_and_an_empty_mailbox_in_one_over_leave_the_body_unsent():
    """`FS Y` and `FF` in one stint of the gateway's, and the whole message
    still in the link's buffer at the instant the session calls itself done.

    The changeover packet takes `FS ` and one five-byte field carries the rest,
    so both lines reach `feed` together: it writes the body and then `FQ`, and
    sets `done` on the same call. `sent_mids` says the message was prepared,
    `MailClient.done` says the exchange is over, and the gateway has not had a
    byte of it. `onair` closes the hold on that flag alone
    (`shrike/onair.py`, `[mail] exchange complete -- closing`), so what is
    riding on the teardown here is the message, not the three bytes of `FQ`
    the fetch direction leaves behind.
    """
    msg, blob, host, session, gw, cycles = sending(answer=ACCEPT + EMPTY)

    assert cycles, "\n".join(gw.tape[-20:])
    assert session.done and not session.failure, session.failure
    assert session.sent_mids == [msg.mid], "the client says it prepared it"
    assert bytes((SOH,)) not in bytes(gw.rx), "and not a byte of it crossed"
    assert bytes(host.arq._outbuf).endswith(b"FQ\r")
    assert len(host.arq._outbuf) > len(blob)


def test_scene11_the_teardown_carries_the_body_the_session_left_behind():
    msg, blob, host, session, gw, cycles = sending(answer=ACCEPT + EMPTY)
    assert cycles

    host.arq.on_host_disconnect()
    for _ in range(SEND_CYCLES):
        gw.key()
        host.tick()
        if host.arq.state is not arq.State.CONNECTED:
            break
    assert gateway_read(bytes(gw.rx))[2] == blob
    assert bytes(gw.rx).endswith(b"FQ\r")


# --------------------------------------------------------------------------- #
# Scene 12 -- more than one message in the turn.

def test_scene12_two_proposals_one_answer_and_only_the_accepted_body_crosses():
    """`--mail-send` is repeatable, so a turn may carry more than one `FC`, and
    the answer is one character per proposal in the order they were offered."""
    first = outbound(subject="EN54 send check", mid="B2FSEND0001")
    second = outbound(body=b"and a second one.\r\n", subject="second",
                      mid="B2FSEND0002")
    blob = compress(first.render())
    gw = Ws8eoc([(b"F> ", b"FS YN\r"),
                 (body_stream(blob, b"EN54 send check"), EMPTY)])
    host, session = linked(gw, outbox=[first, second])
    cycles = run(host, gw, cycles=SEND_CYCLES)

    assert cycles, "\n".join(gw.tape[-20:])
    assert not session.failure, session.failure
    assert session.sent_mids == [first.mid]
    rx = bytes(gw.rx)
    assert rx.index(f"FC EM {first.mid}".encode()) \
        < rx.index(f"FC EM {second.mid}".encode()) < rx.index(bytes((SOH,)))
    assert gateway_read(rx)[2] == blob
    assert rx.count(body_stream(blob, b"EN54 send check")) == 1
    assert b"second\x000\x00" not in rx, "the declined message has no header"


# --------------------------------------------------------------------------- #
# Scene 13 -- the verdict a CMS gave a completed body.

def test_scene13_a_refusal_after_the_body_is_recorded_against_the_message():
    """2026-08-21: a CMS answered a completed EOT with `Error check failed on
    receiving B2 message` and disconnected, every frame acknowledged and both
    checksums verifying on the wire. The mid stays in `sent_mids`, and what the
    far end said about it stays beside it."""
    refusal = b"*** Error check failed on receiving B2 message\r"
    msg, blob, host, session, gw, cycles = sending(after=refusal)

    assert cycles, "\n".join(gw.tape[-20:])
    assert session.done and session.failure
    assert session.remote_refusal == refusal.decode().strip()
    assert session.sent_mids == [msg.mid]
    assert gateway_read(bytes(gw.rx))[2] == blob
    assert b"FQ\r" not in bytes(gw.rx)
