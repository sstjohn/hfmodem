# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The mail client and exchange engine, and the seams a wire can break.

The round-trips over real modem byte paths live with their modems
(tests/shrike/test_mail.py, tests/besra/test_mail.py). This file proves the
engine itself — and, more importantly, that the session's own checks catch a
seam that corrupts, reorders, or swallows link bytes, so a broken wire fails
loudly rather than delivering quietly wrong mail.
"""
from __future__ import annotations

import re

import pytest

from hfmodem.winlink import (B2FSession, MailClient, MailExchange, compose,
                             load_outbound, summarize, write_inbox)
from hfmodem.winlink.message import MessageError

SOH, STX = 0x01, 0x02


def _incompressible(n: int) -> bytes:
    """A deterministic body lzhuf cannot shrink much, so it spans several
    250-byte transfer blocks."""
    import hashlib
    out = bytearray()
    seed = b"wire"
    while len(out) < n:
        seed = hashlib.sha256(seed).digest()
        out += seed
    return bytes(out[:n])


def _sessions(with_mail: bool = True):
    out = compose("W9SSJ", "SMTP:op@example.net", "wire test",
                  _incompressible(600))
    calling = B2FSession("W9SSJ", role="calling", target="K7ABC",
                         outbox=[out] if with_mail else [])
    answering = B2FSession("K7ABC", role="answering", target="W9SSJ")
    return out, calling, answering


def _pump(calling, answering, tamper=None, drop=None):
    """Ping-pong two sessions over a perfect wire, with optional sabotage.

    `tamper` maps each calling-side blob to what the answerer is fed;
    `drop` names blob indices (calling side) the wire swallows outright.
    """
    a_out, b_out = [], []
    cal = MailClient(calling, a_out.append)
    ans = MailClient(answering, b_out.append)
    cal.link_up()
    ans.link_up()
    sent = 0
    for _ in range(200):
        if not a_out and not b_out:
            break
        if b_out:
            cal.on_link_data(b_out.pop(0))
        elif a_out:
            blob = a_out.pop(0)
            idx, sent = sent, sent + 1
            if drop is not None and idx in drop:
                continue
            if tamper is not None:
                blob = tamper(blob)
            ans.on_link_data(blob)
    return cal, ans


def test_clean_exchange_lands_the_message_byte_exact():
    out, calling, answering = _sessions()
    _pump(calling, answering)
    assert calling.done and answering.done
    assert not calling.failure and not answering.failure
    assert [m.render() for m in answering.inbox] == [out.render()]
    assert calling.sent_mids == [out.mid]


def test_a_wire_that_corrupts_the_body_is_caught():
    out, calling, answering = _sessions()

    def corrupt(blob: bytes) -> bytes:
        i = blob.find(STX)
        if i < 0 or len(blob) < i + 6:
            return blob
        b = bytearray(blob)
        b[i + 4] = (b[i + 4] + 1) & 0xFF          # inside the block payload
        return bytes(b)

    _pump(calling, answering, tamper=corrupt)
    assert answering.failure, "a corrupted body must not pass"
    assert not answering.inbox, "and the message must not be delivered"


def test_a_wire_that_reorders_the_body_blocks_is_caught():
    """Two STX blocks swapped in transit. The EOT checksum is a byte sum and
    cannot see order, so this is the compressed body's own CRC earning its
    keep — the last guard between a reordering seam and silently wrong mail."""
    out, calling, answering = _sessions()

    def swap_blocks(blob: bytes) -> bytes:
        if not blob or blob[0] != SOH:
            return blob
        head = 2 + blob[1]
        p, blocks = head, []
        while p < len(blob) and blob[p] == STX:
            n = blob[p + 1] or 256
            blocks.append(blob[p:p + 2 + n])
            p += 2 + n
        if len(blocks) < 2:
            return blob
        blocks[0], blocks[1] = blocks[1], blocks[0]
        return blob[:head] + b"".join(blocks) + blob[p:]

    _pump(calling, answering, tamper=swap_blocks)
    assert answering.failure, "reordered body blocks must not pass"
    assert not answering.inbox


def test_a_wire_that_swallows_a_delivery_stalls_rather_than_lies():
    out, calling, answering = _sessions()
    _pump(calling, answering, drop={1})
    assert not answering.inbox
    assert not (calling.done and answering.done), \
        "a swallowed delivery must not read as a completed exchange"


def test_exchange_engine_reports_the_stage_a_dead_link_stalls_in():
    class DeafTransport:
        """Connects, then never carries a byte."""
        def __init__(self):
            self.sent = []
            self.send = self.sent.append
            self.torn_down = False
            self.connected = True
        def attach(self, client): self.client = client
        def connect(self): return True
        def step(self): pass
        def disconnect(self): self.torn_down = True

    _, calling, _ = _sessions()
    t = DeafTransport()
    report = MailExchange(calling, t, max_steps=5).run()
    assert "stalled in: awaiting greeting" in report
    assert t.torn_down, "teardown must run whatever happened"


def test_exchange_engine_tears_down_after_no_link():
    class NoLink:
        send = staticmethod(lambda blob: None)
        connected = False
        def attach(self, client): pass
        def connect(self): return False
        def disconnect(self): self.torn_down = True

    _, calling, _ = _sessions()
    t = NoLink()
    report = MailExchange(calling, t).run()
    assert "NO LINK" in report and t.torn_down


def test_summarize_names_stage_failure_and_traffic():
    _, calling, answering = _sessions()
    _pump(calling, answering)
    text = summarize(answering)
    assert "stage closed" in text
    assert "received" in text and "wire test" in text


@pytest.mark.parametrize("delivery", ["withheld", "failed", "received"])
def test_outbound_report_requires_delivery_evidence(delivery):
    out, calling, answering = _sessions()
    calling.start()
    proposal = calling.feed(answering.start())
    body = calling.feed(answering.feed(proposal))
    assert body and not answering.inbox

    if delivery == "failed":
        calling.feed(b"*** Transfer failed\r")
    elif delivery == "received":
        calling.feed(answering.feed(body))
        assert [m.render() for m in answering.inbox] == [out.render()]
        assert f"mail: received {out.mid}" in summarize(answering)

    report = summarize(calling)
    # What the record holds, and no more: the gateway's verdict on the proposal,
    # the body this station framed for it, and both ends' closes. B2F has no
    # positive acknowledgement above that, so the line never says delivered.
    assert re.search(rf"mail: prepared {out.mid} — accepted FS \+, "
                     r"body prepared \(4 block\(s\), \d+ B\)", report), report
    assert "delivered" not in report
    assert "mail: sent" not in report
    if delivery == "failed":
        assert "remote reported: *** Transfer failed" in report
    else:
        assert "remote reported:" not in report


def test_summarize_carries_the_refusal_that_ended_the_session():
    """The guard that would have saved a rig slot. besra took KX8U's refusal
    off the air with a valid CRC and acknowledged every frame of it, and the
    record it printed was `mail: nothing moved` with no reason — the reason was
    recovered from the recording hours later. Whatever else this summary says,
    it says what the gateway said."""
    from hfmodem.tests.winlink.test_session import _refused

    text = summarize(_refused())
    assert "mail: nothing moved" in text
    assert ("mail: peer said: *** Unknown client types are not allowed on "
            "production servers -- use cms-z.winlink.org - Disconnecting "
            "(208.102.176.56)") in text


def test_summarize_never_claims_a_send_the_gateway_refused():
    from hfmodem.tests.winlink.test_session import BODY_REFUSAL, _body_refused

    text = summarize(_body_refused())
    assert "mail: sent" not in text and "delivered" not in text
    line, = [ln for ln in text.splitlines() if ln.startswith("mail: prepared")]
    assert line.startswith("mail: prepared MIDMIDMID123 — accepted FS Y, "
                           "body prepared (")
    assert line.endswith(f"no FF from the far end, no close of ours; "
                         f"remote reported: {BODY_REFUSAL}"), line


def test_summarize_reports_our_own_close_the_way_it_reports_the_far_end_s():
    """B2F knows FQ was queued; only the transport can prove it was sent."""
    from hfmodem.tests.winlink.test_session import _body_refused

    s = _body_refused()
    s.remote_ff = s.our_fq = True
    line, = [ln for ln in summarize(s).splitlines()
             if ln.startswith("mail: prepared")]
    assert "far end FF received" in line and "our FQ queued" in line
    assert "not sent" not in line

    s.our_fq = False
    line, = [ln for ln in summarize(s).splitlines()
             if ln.startswith("mail: prepared")]
    assert "no close of ours" in line


def test_summarize_says_what_this_station_transmitted_too():
    """An exchange only one side of which is on disk is half a record. On
    2026-08-18 WW2MI refused a login challenge and told this station two
    attempts remained; whether the session had spent one attempt or two could
    not be read back from anything it wrote, only from the modem's frame
    count. The `;PR:` digits stay out — they are an offline oracle on the
    password beside the `;PQ:` they answer — but that one went out does not."""
    from hfmodem.tests.winlink.test_session import _ww2mi_refused

    text = summarize(_ww2mi_refused())
    assert "mail: peer said: ;PQ: 90189792" in text
    assert text.count("mail: we said: ;PR: ########") == 1
    assert "hunter2" not in text


def test_load_outbound_round_trips_a_rendered_message(tmp_path):
    out, _, _ = _sessions()
    p = tmp_path / "m.b2f"
    p.write_bytes(out.render())
    assert load_outbound(p, "W9SSJ").render() == out.render()


def test_load_outbound_composes_a_body_file_and_requires_to(tmp_path):
    p = tmp_path / "note.txt"
    p.write_bytes(b"plain body\r\n")
    msg = load_outbound(p, "W9SSJ", to="SMTP:x@example.net", subject="hi")
    assert msg.body == b"plain body\r\n" and msg.subject == "hi"
    with pytest.raises(MessageError):
        load_outbound(p, "W9SSJ")


def test_write_inbox_writes_each_message_once(tmp_path):
    out, calling, answering = _sessions()
    _pump(calling, answering)
    paths = write_inbox(answering, tmp_path / "mail")
    assert [p.read_bytes() for p in paths] == [out.render()]


def test_write_inbox_keeps_a_block_it_could_not_read(tmp_path):
    """The lesson of 2026-08-16: a message the parser refuses is still the only
    copy of what a gateway sent, and the session it arrived in does not come
    back. It goes to disk as the compressed block it was."""
    from hfmodem.tests.winlink.test_session import _unreadable

    session, mid, blob = _unreadable()
    (path,) = write_inbox(session, tmp_path / "mail")
    assert path.name == f"{mid}.lzh" and path.read_bytes() == blob


def test_the_stage_is_readable_while_the_session_is_still_running(caplog):
    """The 2026-08-28 defect: `summarize` runs at teardown and nowhere else, so an
    operator watching a live log for `mail:` lines saw none while a W6IDS greeting
    was already in hand, read that as payload not reaching the application, and
    stopped the arm at the login with 120 gateway minutes left.

    So the pump emits what the session knows about its own progress as it changes:
    the stage, and the events the session was already recording into `log_lines`
    where nothing ever read them. Both are in the log before the exchange ends —
    here at the point the greeting has landed and the session has answered it.
    """
    caplog.set_level("INFO", logger="hfmodem.winlink.client")
    calling = B2FSession("W9SSJ", role="calling", target="W6IDS")
    cal = MailClient(calling, lambda _b: None)
    cal.link_up()
    cal.on_link_data(b"RMS Trimode 1.4.2.3 Welcome to the W6IDS RMS HYBRID "
                     b"Winlink gateway, Richmond IN.\r"
                     b"W9SSJ has 120 daily minutes remaining with W6IDS (EM79NV)\r")
    lines = [r.getMessage() for r in caplog.records]
    assert "mail: stage awaiting greeting" in lines
    assert any("120 daily minutes" in ln for ln in lines)
    assert not calling.done, "the reading is available before the exchange ends"


def test_the_live_lines_carry_no_challenge_answer(caplog):
    """A `;PR:` answer and the `;PQ:` it answers are, together, an offline search
    for the password — `mask_pr` exists for that and the answering side puts a
    caller's whole handshake through `log`. The digits must not reach a log line
    that ships, so the masking is in `log` itself rather than at each caller."""
    caplog.set_level("INFO", logger="hfmodem.winlink.client")
    answering = B2FSession("K7ABC", role="answering", target="W9SSJ")
    ans = MailClient(answering, lambda _b: None)
    ans.link_up()
    ans.on_link_data(b"[Pat-1.0.0-B2FHM$]\r;PR: 12345678\r")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert ";PR: ########" in text and "12345678" not in text
    assert "12345678" not in "\n".join(answering.log_lines)


def test_write_inbox_keeps_the_block_a_stalled_transfer_got_part_of(tmp_path):
    """A message the link stopped in the middle of is written as far as it
    came. It will not decompress — lzhuf reads a stream and not a prefix — and
    it is still the only copy of what crossed and the only measure of how far
    the transfer got."""
    from hfmodem.tests.winlink.test_session import _part_way_through

    session, mid, head = _part_way_through()
    (path,) = write_inbox(session, tmp_path / "mail")
    assert path.name == f"{mid}.partial.lzh" and path.read_bytes() == head


def test_summarize_counts_the_bytes_a_stalled_transfer_did_deliver():
    """`accepted ... and never got it` is a claim about the air, and it was
    wrong about the message the link died inside of on 2026-09-08."""
    from hfmodem.tests.winlink.test_session import _part_way_through

    session, mid, head = _part_way_through()
    report = summarize(session)
    assert f"mail: accepted {mid} and got {len(head)} of " in report
    assert "never got it" not in report


def test_progress_to_stdout_stamps_the_lines_and_gives_the_logger_back(capsys):
    """The arm log is the only record a session leaves, and `rehear` aligns it
    against the tape by wall clock: a mail line without a time on it cannot be
    put beside the burst it belongs to. And the handler comes off again — the
    tool that keys the transmitter is not the only thing in the process."""
    import logging
    from datetime import datetime, timezone

    from hfmodem.winlink import progress_to_stdout

    logger = logging.getLogger("hfmodem.winlink.client")
    before = (logger.level, list(logger.handlers))
    with progress_to_stdout():
        MailClient(B2FSession("W9SSJ", role="calling", target="K7ABC"),
                   lambda _b: None)
    line = capsys.readouterr().out.strip()
    assert line.endswith("Z mail: stage awaiting greeting")
    stamped = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((stamped - now).total_seconds()) < 5, "the stamp is not UTC"
    assert (logger.level, logger.handlers) == before


@pytest.mark.parametrize("answer,said", [("N", "declined"), ("=", "deferred")])
def test_a_refused_proposal_does_not_read_like_a_carried_one(answer, said):
    """The distinction the old line could not make. `prepared <MID> — delivery
    unconfirmed` printed the same for a gateway that took the message and for
    one that never let it on the air, and an operator reading the summary of a
    session had no way to tell which had happened."""
    from hfmodem.tests.winlink.test_session import KC9GHZ, _mail

    msg, _ = _mail()
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    s.feed(f"FS {answer}\r".encode())
    text = summarize(s)
    assert f"mail: not transmitted {msg.mid} — proposal {said} FS {answer}" in text
    assert "body prepared" not in text
    assert "delivery unconfirmed" not in text


def test_a_proposal_the_far_end_never_answered_says_so():
    from hfmodem.tests.winlink.test_session import KC9GHZ, _mail

    msg, _ = _mail()
    s = B2FSession("W9SSJ", role="calling", target="KC9GHZ", outbox=[msg])
    s.feed(KC9GHZ)
    assert (f"mail: not transmitted {msg.mid} — the proposal drew no answer"
            in summarize(s))
