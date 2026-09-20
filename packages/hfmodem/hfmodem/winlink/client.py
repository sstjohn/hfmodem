# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The station as its own mail application: one client, whichever link is up.

`B2FSession` answers bytes with bytes and holds no opinion about time, turns,
or transports. A live exchange needs two more things, and they are the same
two for every modem in this package, so they are written here once:

  * :class:`MailClient` — the pump binding one session to one link's byte
    pipe. A transport calls `link_up()` when its link connects and
    `on_link_data()` with each payload delivery, in arrival order, and queues
    whatever `send` is handed back. That is the whole seam; the modems'
    host layers already end in exactly this shape.
  * :class:`MailExchange` — drives a session from connect to teardown against
    a transport and keeps the stage-by-stage record. When a session dies,
    WHICH stage it died in is the finding: no link, link but no greeting,
    greeting but no proposal answer, transfer begun but broken — each points
    at a different layer.

A transport is duck-typed, because the modems' drivers differ more than any
base class would admit:

    send            callable(bytes) — queue payload on the link
    attach(client)  route delivered payload to `client.on_link_data` and the
                    link-up edge to `client.link_up`
    connect()       bring the link up; True once it is, False when it will
                    never be
    step()          advance the modem by one increment (an ARQ cycle, a
                    pumped air, a slice of wall clock)
    connected       property: the link is still up
    disconnect()    tear the link down; called whatever happened
    stall_note      optional property: one line of link-layer state, recorded
                    when the exchange stalls or the link dies mid-session

Turn-taking stays in the transport, where the link's own law lives — PACTOR
hands a channel over by changeover packet, ARDOP by BREAK, and the session
neither knows nor cares. The exchange's turn order past the greeting is taken
from the published FBB protocol, and a live session has now run the whole of
it: WW2MI's CMS over ARDOP BW200 on 2026-08-19 proposed a message, was
answered FS, sent the SOH/STX/EOT blocks and closed clean, and the message
parsed.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Callable

from .message import Message
from .session import B2FSession

log = logging.getLogger(__name__)


@contextlib.contextmanager
def progress_to_stdout():
    """Put :class:`MailClient`'s progress on stdout for the length of a run,
    stamped UTC, and give the logger back exactly as it was found.

    Every caller that keys a transmitter wants this and each was writing its
    own: the arm log is the only record a session leaves, and until it carries
    these lines an operator watching a live run cannot tell a stage from a
    stall. The stamp is what makes the log an instrument rather than a story —
    `rehear` aligns a session against the tape by wall clock, and on 2026-09-08
    it could not align a mail arm at all because not one mail line had a time
    on it.
    """
    logger = logging.getLogger(__name__)
    previous = logger.level
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)sZ %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous)


class MailClient:
    """One B2F session pumped over one link's byte pipe, and the exchange's
    only live instrument.

    `summarize` runs at teardown and nowhere else. On 2026-08-28 an operator
    watched a running W6IDS arm for `mail:` lines, saw none, and stopped it
    after two minutes of clean decodes on the reading that payload was not
    reaching the application — while the gateway's greeting was already in
    hand, the link had 120 minutes left on it and the login was the next thing
    due. Absence in a running log meant nothing, and a rule to remember that is
    weaker than a reading that shows the stage.

    So the two things the session knows about its own progress are emitted as
    they change: `B2FSession.stage`, and the events the session was already
    recording into `log_lines` where nothing ever read them. Every byte the
    link delivers passes through here and nowhere else, which is why the watch
    is here; it costs one comparison per delivery and a log record only when
    something moved, because this runs on the modem's receive thread. The
    first line goes out at construction, so an attempt that never connects
    still names the stage it was waiting in.
    """

    def __init__(self, session: B2FSession, send: Callable[[bytes], None]):
        self.session = session
        self._send = send
        self._stage = ""
        self._logged = 0
        self._watch()

    def link_up(self) -> None:
        """The link connected: transmit whatever the session opens with."""
        self._push(self.session.start())
        self._watch()

    def on_link_data(self, blob: bytes) -> None:
        """The link delivered payload: feed it, transmit the answer."""
        self._push(self.session.feed(blob))
        self._watch()

    def _watch(self) -> None:
        for line in self.session.log_lines[self._logged:]:
            log.info("mail: %s", line)
        self._logged = len(self.session.log_lines)
        if self.session.stage != self._stage:
            self._stage = self.session.stage
            log.info("mail: stage %s", self._stage)

    def _push(self, tx: bytes) -> None:
        if tx:
            self._send(tx)

    @property
    def done(self) -> bool:
        return self.session.done


class MailExchange:
    """One mail exchange, connect to teardown, with the record kept.

    `run()` never leaves the link up: the teardown is in a finally, because a
    session that failed is exactly the one whose link must not be left holding
    a channel.
    """

    def __init__(self, session: B2FSession, transport, *, max_steps: int = 600):
        self.session = session
        self.transport = transport
        self.max_steps = max_steps
        self.stages: list[str] = []

    def _mark(self, line: str) -> None:
        self.stages.append(line)

    def _note(self) -> None:
        """A transport may name its own link-layer state for the post-mortem:
        which layer a dead exchange died in is the finding."""
        note = getattr(self.transport, "stall_note", "")
        if note:
            self._mark(note)

    def run(self) -> str:
        client = MailClient(self.session, self.transport.send)
        self.transport.attach(client)
        try:
            self._mark("connect: calling")
            if not self.transport.connect():
                self._mark("connect: NO LINK")
                return self.report()
            self._mark("connect: link up")
            stage = self.session.stage
            for step in range(self.max_steps):
                if self.session.done:
                    break
                self.transport.step()
                if self.session.stage != stage:
                    self._mark(f"step {step}: {stage} -> {self.session.stage}")
                    stage = self.session.stage
                if not self.transport.connected and not self.session.done:
                    self._mark(f"link lost during: {stage}")
                    self._note()
                    return self.report()
            if not self.session.done:
                self._mark(f"stalled in: {self.session.stage} "
                           f"after {self.max_steps} steps")
                self._note()
        finally:
            self.transport.disconnect()
        return self.report()

    def report(self) -> str:
        return "\n".join(self.stages) + "\n" + summarize(self.session)


def _delivery(session: B2FSession, m) -> str:
    """What the record holds about one outbound message, and nothing beyond it.

    `prepared <MID> — delivery unconfirmed` read the same on a message a gateway
    accepted and carried and on one it declined at the proposal, which is the
    one distinction an operator reading the line needs. Keep the proposal answer
    and subsequent peer commands, but label our bodies as prepared and our FQ
    as queued. This layer receives no transport drain/ACK notification and
    cannot establish transmission or final mailbox delivery.
    """
    said = f" FS {m.answer}" if m.answer else ""
    if m.status == "rejected":
        return f"not transmitted {m.mid} — proposal declined{said}"
    if m.status == "deferred":
        return f"not transmitted {m.mid} — proposal deferred{said}"
    if m.status != "sent":
        return f"not transmitted {m.mid} — the proposal drew no answer"
    body = (f"body prepared ({m.blocks} block(s), {m.csize} B)" if not m.offset
            else f"body prepared from byte {m.offset} of {m.csize} "
                 f"({m.blocks} block(s))")
    facts = [f"accepted{said}", body,
             "far end FF received" if session.remote_ff else "no FF from the far end",
             "our FQ queued" if session.our_fq else "no close of ours"]
    return (f"prepared {m.mid} — " + ", ".join(facts)
            + (f"; remote reported: {session.remote_refusal}"
               if session.remote_refusal else "; delivery unconfirmed by the CMS"))


def summarize(session: B2FSession) -> str:
    """Summarize progress, prepared outbound messages, and verified inbound mail."""
    lines = [f"mail: stage {session.stage}"
             + (f" — {session.failure}" if session.failure else "")]
    if session.remote_sid:
        lines.append(f"mail: remote SID {session.remote_sid}")
    for ours, said, protocol in session.exchange:
        who = "we" if ours else "peer"
        action = "queued" if ours and protocol else "sent" if protocol else "said"
        lines.append(f"mail: {who} {action}: {said}")
    for m in session.proposals:
        lines.append("mail: " + _delivery(session, m))
    for msg in session.inbox:
        lines.append(f"mail: received {msg.mid} {msg.subject!r}")
    for mid in session.unfetched:
        held = session.in_flight
        if held and held[0] == mid:
            lines.append(f"mail: accepted {mid} and got {held[1]} of {held[2]} "
                         "bytes before the link stopped — the rest is at the "
                         "far end")
        else:
            lines.append(f"mail: accepted {mid} and never got it — "
                         "it is still at the far end")
    if not session.proposals and not session.inbox:
        lines.append("mail: nothing moved")
    return "\n".join(lines)


def load_outbound(path: str | Path, mycall: str, *, to: str = "",
                  subject: str = "") -> Message:
    """A file as an outbound message: a rendered B2F message as it stands, or
    anything else as the body of a fresh one addressed by the flags."""
    from .message import MessageError, compose
    data = Path(path).read_bytes()
    try:
        return Message.parse(data)
    except MessageError:
        if not to:
            raise MessageError(
                f"{path} is not a rendered message, so --to must say where "
                "its contents are going") from None
        return compose(mycall, to, subject or Path(path).name, data)


def write_inbox(session: B2FSession, outdir: str | Path) -> list[Path]:
    """Received mail to disk: a rendered `.b2f` per message, and beside it the
    compressed block, `.lzh`, for anything the gateway delivered that did not
    become one.

    A block that fails its checksum, its decompression, or its parse is the
    only copy of what the far end sent, and the run it came from is not
    repeatable — WW2MI's CMS offered this station a real message on
    2026-08-16, the parser refused it, and the specimen went down with the
    link. So an unreadable block is written too, and the operator has
    something to take apart.

    A message the link stopped in the middle of gets `<mid>.partial.lzh`: the
    same compressed bytes, as far as they came. It will not decompress — lzhuf
    reads a stream, not a prefix — and it is still the only copy of what
    crossed, and the only thing that says how far a stalled transfer got.
    """
    out = Path(outdir)
    read = {m.mid for m in session.inbox}
    unread = [(mid, blob) for mid, blob in session.received_blocks
              if mid not in read]
    partial = session.partial_block
    written = []
    if session.inbox or unread or partial:
        out.mkdir(parents=True, exist_ok=True)
    for msg in session.inbox:
        p = out / f"{msg.mid or 'NOMID'}.b2f"
        p.write_bytes(msg.render())
        written.append(p)
    for mid, blob in unread:
        p = out / f"{mid or 'NOMID'}.lzh"
        p.write_bytes(blob)
        written.append(p)
    if partial:
        mid, _, _ = session.in_flight
        p = out / f"{mid}.partial.lzh"
        p.write_bytes(partial)
        written.append(p)
    return written
