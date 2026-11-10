# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The data transfer sublayer (annex C): C_PDUs across the radio.

**Services.** Reliable delivery by selective-repeat ARQ over a connection
(DATA-ONLY, ACK-ONLY and DATA-ACK D_PDUs, C.3.3-C.3.5, C.6.2), expedited
reliable delivery by a C_PDU-at-a-time stop-and-wait (C.3.7, C.3.8,
C.6.1.3), and best-effort non-ARQ delivery with repeated copies, to single
nodes or groups, connected or not (C.3.10, C.3.11, C.4.2). MANAGEMENT
D_PDUs carry EOW messages that need acknowledging, one at a time (C.3.9),
and with them the data rate change procedure (C.6.4).

**One radio, half duplex.** The sublayer composes a transmission when the
node may key, and learns what came of it from what the far end sends back.
It does not keep time itself: every call is told ``now``, so two nodes can
run against each other in a simulation as fast as it computes, or on the
air in real time.

**Composing a transmission** takes, in order, what C.3.9, C.3.7 and C.6.1.3
put first: management (repeated to fill the interleaver, table C-26),
expedited acknowledgements and data, the acknowledgement owed for regular
data (folded into the first DATA D_PDU as a DATA-ACK when there is data to
send the same node), regular data (the window's retransmissions first, then
new segments), and non-ARQ data with its repeated copies.

**Retransmission** is by the far end's acknowledgements: every frame sent
and not acknowledged when an acknowledgement arrives goes again, and so
does everything outstanding if the far end stays silent past the time it
had to answer. A connection that stays silent for ``max_retries`` of those
is lost (D_CONNECTION_LOST).
"""
from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field, replace
from enum import StrEnum

from . import dpdu as D

MAX_CPDU = 2054                            # A.2.1.3 MTU plus S/C headers
MAX_REASSEMBLY = 64
MAX_PER_SOURCE = 8
MAX_HOLD_S = 600.0


class Reject(StrEnum):
    TTL = "TTL expired"
    RESET = "reset"
    LOST = "destination node not responding"
    TERMINATED = "connection terminated"
    UNCONNECTED = "not connected"
    SIZE = "C_PDU too large"


MAX_WINDOW = 128                            # C.6.2 (g)


@dataclass(eq=False)
class CPDU:
    """A C_PDU handed down for delivery, with how it is to be delivered."""
    data: bytes
    dest: int
    arq: bool = True
    expedited: bool = False
    in_order: bool = True
    #: Non-ARQ: copies beyond the first (the client's minimum number of
    #: retransmissions, A.2.2.3).
    repeats: int = 0
    group: bool = False
    #: Absolute time after which it is not worth sending (TTD), or None.
    ttd: float | None = None
    priority: int = 0
    #: Anything: handed back with its confirmation or rejection.
    handle: object = None


class Upper:
    """What the sublayer reports to the channel access sublayer."""

    def cpdu_received(self, src: int, dest: int, data: bytes, *, expedited: bool,
                      arq: bool, errors=None) -> None: ...
    def cpdu_confirmed(self, handle) -> None: ...
    def cpdu_rejected(self, handle, reason: str) -> None: ...
    def connection_lost(self, remote: int) -> None: ...
    def warning_received(self, src: int, rx_type: int, reason: int) -> None: ...
    def rate_changed(self, remote: int, rate: int, interleave: str) -> None: ...


@dataclass
class _Frame:
    seq: int
    data: bytes
    start: bool
    end: bool
    in_order: bool
    cpdu: CPDU
    drop: bool = False
    state: str = "new"                      # new | sent | resend
    sent: int = 0


@dataclass
class _Arq:
    """One direction pair of the selective-repeat machine (C.6.2)."""
    tx_lwe: int = 0
    tx_uwe: int = 255
    rx_lwe: int = 0
    rx_uwe: int = 255
    next_seq: int = 0
    frames: dict = field(default_factory=dict)          # seq -> _Frame, outstanding
    pending: deque = field(default_factory=deque)       # CPDUs not yet segmented
    cursor: tuple | None = None                         # (CPDU, offset) mid-segmentation
    unacked: dict = field(default_factory=dict)         # CPDU -> [CPDU, frames left]
    got: set = field(default_factory=set)               # received seqs at/after rx_lwe
    held: dict = field(default_factory=dict)            # seq -> received segment info
    deliver_at: int = 0                                 # first seq of the next C_PDU
    owe_ack: bool = False
    partial: bytearray = field(default_factory=bytearray)
    partial_started: bool = False
    partial_drop: bool = False


@dataclass
class _Expedited:
    """C.6.1.3: one expedited C_PDU at a time, its frames numbered from 0."""
    queue: deque = field(default_factory=deque)
    cpdu: CPDU | None = None
    tx_count: int = 0
    cpdu_id: int = 15                                   # incremented before use
    frames: dict = field(default_factory=dict)          # seq -> _Frame
    rx_id: int | None = None
    rx: dict = field(default_factory=dict)              # seq -> (start, end, data)
    rx_uwe: int = -1
    rx_done: int | None = None                          # id of the last delivered
    owe_ack: bool = False


@dataclass
class _Management:
    """C.3.9: at most one MANAGEMENT D_PDU unacknowledged."""
    queue: deque = field(default_factory=deque)         # (eow, extended, repeats)
    out: tuple | None = None                            # (frame_id, eow, extended, repeats)
    sent: int = 0
    unknown: set = field(default_factory=set)
    tx_id: int = 0
    rx_id: int = 256                                    # out of range: none yet
    owe_ack: bool = False


@dataclass
class Link:
    """What the sublayer keeps for one connected remote node."""
    remote: int
    arq: _Arq = field(default_factory=_Arq)
    exp: _Expedited = field(default_factory=_Expedited)
    mgmt: _Management = field(default_factory=_Management)
    #: Waiting for the far end to answer what was sent, until this time.
    deadline: float | None = None
    silences: int = 0
    #: The data rate change procedure (C.6.4), when one is running.
    drc: dict | None = None
    #: RESET/WIN-RESYNC (C.3.6): a FULL RESET we sent, awaiting its ACK;
    #: an ACK or a resynchronising command we owe.
    reset_out: dict | None = None
    reset_ack_due: bool = False
    reset_ack_id: int = 0
    reset_rx_id: int | None = None
    zero_due: bool = False
    resync_cmd_due: bool = False
    #: When a D_PDU last came from the far end on this connection.
    heard_at: float | None = None


def _between(lo: int, x: int, hi: int) -> bool:
    """lo <= x <= hi, modulo 256, the span from lo to hi taken forward."""
    return (x - lo) & 0xFF <= (hi - lo) & 0xFF


def _repeats_for(rate: int, interleave: str) -> int:
    """Table C-26: copies of a DRC request to fill a 4285 interleaver."""
    short = {75: 1, 150: 1, 300: 1, 600: 3, 1200: 7, 2400: 15}
    long_ = {75: 9, 150: 18, 300: 37, 600: 75, 1200: 150, 2400: 300}
    table = long_ if interleave == "long" else short
    return table.get(rate, 1)


class DataTransfer:
    """The sublayer for one node.

    ``segment`` is the largest segment of a C_PDU a D_PDU carries (C.4.1
    a: at most 1023; annex H.7's 200 is the default). ``window`` is the
    maximum allowable window (at most 128)."""

    def __init__(self, address: int, upper: Upper | None = None, *, segment: int = 200,
                 window: int = MAX_WINDOW, max_retries: int = 6, groups=(),
                 deliver_with_errors: bool = False) -> None:
        if not 1 <= segment <= D.MAX_SEGMENT:
            raise ValueError(f"a segment is 1 to {D.MAX_SEGMENT} bytes")
        if not 1 <= window <= MAX_WINDOW:
            raise ValueError(f"the window is 1 to {MAX_WINDOW}")
        self.address = address
        self.upper = upper or Upper()
        self.segment = segment
        self.window = window
        self.max_retries = max_retries
        self.drc_retries = 3
        self.drc_timeout_s = 120.0
        self._now = 0.0
        self._room = segment
        self._unconnected_ack: dict[int, int] = {}
        self.nonarq_deadlines: dict[tuple, float] = {}
        self.groups = set(groups)
        self.deliver_with_errors = deliver_with_errors
        self.links: dict[int, Link] = {}
        # Non-ARQ (C.3.10): an ID space for each of the two types.
        self._nonarq_id = {D.NON_ARQ: 0, D.EXP_NON_ARQ: 0}
        self._nonarq_out: deque = deque()               # D_PDUs still to send
        self._reassembly: dict = {}                     # (src, type, id) -> dict
        self._delivered: dict = {}                      # (src, type, id) -> until
        self.warnings_out: deque = deque()              # (dest, rx_type, reason)
        self._reset_ids = -1                            # RESET FRAME ID NUMBER
        self.counts = {"sent": 0, "resent": 0, "received": 0, "bad_data": 0}
        #: The modem's current transmit rate and interleave, as the data
        #: rate change procedure leaves them: (rate, "short"|"long").
        self.tx_rate: dict[int, tuple] = {}
        #: What the modem sends at until a rate change says otherwise.
        self.rate: tuple[int, str] = (300, "short")

    # -- from the channel access sublayer ----------------------------------

    def connection_made(self, remote: int) -> None:
        """D_CONNECTION_MADE: a fresh ARQ machine (C.6.3.1)."""
        if remote not in self.links:
            self.links[remote] = Link(remote)

    def connection_terminated(self, remote: int) -> None:
        link = self.links.pop(remote, None)
        if link:
            for cp in self._outstanding(link):
                self.upper.cpdu_rejected(cp.handle, Reject.TERMINATED)

    def connected(self, remote: int) -> bool:
        return remote in self.links

    def submit(self, cp: CPDU) -> bool:
        """D_UNIDATA_REQUEST or D_EXPEDITED_UNIDATA_REQUEST. False (and a
        rejection) if reliable delivery is asked of an unconnected node."""
        if len(cp.data) > MAX_CPDU:
            self.upper.cpdu_rejected(cp.handle, Reject.SIZE)
            return False
        if cp.arq:
            link = self.links.get(cp.dest)
            if link is None:
                self.upper.cpdu_rejected(cp.handle, Reject.UNCONNECTED)
                return False
            if cp.expedited:
                link.exp.queue.append(cp)
            else:
                q = link.arq.pending
                # Higher priority first; equal priority in order of arrival.
                i = len(q)
                while i > 0 and q[i - 1].priority < cp.priority:
                    i -= 1
                q.insert(i, cp)
            return True
        self._segment_nonarq(cp)
        return True

    def request_rate(self, remote: int, rate: int, interleave: str, *, equal: bool = False,
                     advisory: bool = False) -> None:
        """Start a data rate change (C.6.4.2) as DRC master, or advise the
        far end (``advisory``) what it should send at."""
        link = self.links.get(remote)
        if link is None:
            raise ValueError("no connection to that node")
        if not self.can_receive(rate, interleave):
            raise ValueError("requested rate cannot carry a D_PDU")
        other = (2 if advisory else 0) + (1 if equal else 0)
        eow = D.eow_drc_request(rate, interleave, other)
        if advisory:
            link.mgmt.queue.append((eow, b"", 1))
            return
        link.drc = {"role": "master", "phase": "request", "rate": rate,
                    "interleave": interleave, "equal": equal,
                    "until": None}
        link.mgmt.queue.append((eow, b"", _repeats_for(*self._rate(remote))))

    # -- non-ARQ segmentation (C.4.2) ----------------------------------------

    def _segment_nonarq(self, cp: CPDU) -> None:
        t = D.EXP_NON_ARQ if cp.expedited else D.NON_ARQ
        cid = self._nonarq_id[t]
        self._nonarq_id[t] = (cid + 1) % 4096
        pieces = [(o, cp.data[o:o + self.segment])
                  for o in range(0, max(len(cp.data), 1), self.segment)] or [(0, b"")]
        for copy in range(cp.repeats + 1):
            for off, seg in pieces:
                self._nonarq_out.append((cp, D.DPDU(t, cp.dest, self.address, cpdu_id=cid,
                                                    in_order=cp.in_order, group=cp.group,
                                                    cpdu_size=len(cp.data), offset=off,
                                                    data=seg), copy == cp.repeats and
                                         off == pieces[-1][0]))

    # -- composing a transmission -----------------------------------------

    def pending(self, now: float) -> bool:
        """Whether there is anything the sublayer would send now."""
        if self._nonarq_out or self.warnings_out or self._unconnected_ack:
            return True
        for link in self.links.values():
            if link.mgmt.owe_ack or link.exp.owe_ack or link.arq.owe_ack \
                    or link.reset_ack_due \
                    or link.resync_cmd_due or link.zero_due:
                return True
            if link.deadline is not None and now < link.deadline:
                continue                        # the far end has its turn
            if link.mgmt.out or link.mgmt.queue or link.reset_out:
                return True
            if self._exp_sendable(link) or self._arq_sendable(link):
                return True
        return False

    def waiting(self, now: float) -> bool:
        """Whether a reply is owed us by any connected node."""
        return any(link.deadline is not None and now < link.deadline
                   for link in self.links.values())

    def _warn(self, dest: int, rx_type: int, reason: int) -> None:
        """One WARNING answers every D_PDU of its kind in a transmission."""
        if (dest, rx_type, reason) not in self.warnings_out:
            self.warnings_out.append((dest, rx_type, reason))

    def compose(self, now: float, budget: int) -> list[D.DPDU]:
        """The D_PDUs for the next transmission, at most ``budget`` bytes;
        their EOT fields are the caller's to fill in."""
        self._now = now
        self._room = max(1, min(self.segment, budget - 40))
        out: list[D.DPDU] = []
        used = 0

        def fits(p: D.DPDU) -> bool:
            nonlocal used
            n = D.size(p)
            if used + n > budget:
                return False
            out.append(p)
            used += n
            return True

        for remote, fid in list(self._unconnected_ack.items()):
            if fits(D.DPDU(D.MANAGEMENT, remote, self.address, ack=True, frame_id=fid)):
                del self._unconnected_ack[remote]
        for link in list(self.links.values()):
            self._compose_management(link, now, fits)
        while self.warnings_out:
            dest, rx_type, reason = self.warnings_out[0]
            if not fits(D.DPDU(D.WARNING, dest, self.address, rx_type=rx_type, reason=reason)):
                break
            self.warnings_out.popleft()
        self._compose_nonarq(now, fits, expedited=True)
        for link in list(self.links.values()):
            self._compose_reset(link, now, fits)
        for link in list(self.links.values()):
            self._compose_expedited(link, now, fits)
        for link in list(self.links.values()):
            if link.zero_due and not (self._arq_sendable(link) or self._exp_sendable(link)):
                link.exp.queue.append(CPDU(b"", link.remote, expedited=True))
                self._compose_expedited(link, now, fits)
            link.zero_due = False
            if link.mgmt.out is not None or link.drc is not None \
                    or link.reset_out is not None:
                continue                        # C.6.1.2 and C.3.6: these hold the rest
            self._compose_arq(link, now, fits)
        self._compose_nonarq(now, fits)
        return out

    def _compose_nonarq(self, now: float, fits, expedited: bool = False) -> None:
        """The non-ARQ queue, in order; ``expedited``: only its expedited
        D_PDUs (C.3.10, C.6.4)."""
        kept: deque = deque()
        while self._nonarq_out:
            cp, p, last = self._nonarq_out[0]
            deadline = self.nonarq_deadlines.get((p.dest, p.type, p.cpdu_id))
            if (cp.ttd is not None and now > cp.ttd) or (deadline is not None and now >= deadline):
                self._nonarq_out.popleft()
                if last:
                    self.upper.cpdu_rejected(cp.handle, Reject.TTL)
                continue
            if expedited and p.type != D.EXP_NON_ARQ:
                kept.append(self._nonarq_out.popleft())
                continue
            if len(p.data) > self._room:
                self._nonarq_out.popleft()
                tail = replace(p, data=p.data[self._room:], offset=p.offset + self._room)
                p = replace(p, data=p.data[:self._room])
                self._nonarq_out.appendleft((cp, tail, last))
                self._nonarq_out.appendleft((cp, p, False))
                last = False
            if not fits(p):
                break
            self._nonarq_out.popleft()
            if last:
                self.upper.cpdu_confirmed(cp.handle)
        kept.extend(self._nonarq_out)
        self._nonarq_out = kept

    def _compose_management(self, link: Link, now, fits) -> None:
        m = link.mgmt
        if m.out is None and m.queue and (link.deadline is None or now >= link.deadline):
            eow, ext, reps = m.queue.popleft()
            m.out = (m.tx_id, eow, ext, reps)
            m.sent = 0
            m.tx_id = (m.tx_id + 1) % 256
        if m.owe_ack:
            if fits(D.DPDU(D.MANAGEMENT, link.remote, self.address, ack=True,
                           frame_id=m.tx_id)):
                m.owe_ack = False
                if link.drc and link.drc["role"] == "master" and \
                        link.drc["phase"] == "accepted":
                    self._apply_rate(link, link.drc)
                    link.drc["phase"] = "confirm"
        if m.out is None or (link.deadline is not None and now < link.deadline):
            return
        limit = self.drc_retries if link.drc else self.max_retries
        if m.sent > limit:
            self._lose(link)
            return
        fid, eow, ext, reps = m.out
        sent = False
        for _ in range(max(1, reps)):
            if not fits(D.DPDU(D.MANAGEMENT, link.remote, self.address, eow=eow, valid=True,
                               frame_id=fid, extended=ext)):
                break
            sent = True
        if sent:
            if link.drc and link.drc["until"] is None:
                link.drc["until"] = now + self.drc_timeout_s
            m.sent += 1
            self._await(link, now)

    def _exp_sendable(self, link: Link) -> bool:
        e = link.exp
        return bool(e.queue) or any(f.state != "sent" for f in e.frames.values())

    def _compose_expedited(self, link: Link, now, fits) -> None:
        e = link.exp
        if e.owe_ack:
            got = {s for s in e.rx}
            lwe = 0
            while lwe in got:
                lwe += 1
            edge = lwe & 0xFF
            uwe = max(got) if got else (edge - 1) & 0xFF
            if fits(D.DPDU(D.EXP_ACK_ONLY, link.remote, self.address, rx_lwe=edge,
                           acks=D.ack_bitmap(edge, uwe, got))):
                e.owe_ack = False
        if link.deadline is not None and now < link.deadline:
            return
        if e.cpdu is None and e.queue:
            cp = e.queue.popleft()
            e.cpdu = cp
            e.cpdu_id = (e.cpdu_id + 1) % 16
            e.tx_count = 0
            data = cp.data
            pieces = [data[o:o + max(self._room, (len(data) + 254) // 255)]
                      for o in range(0, max(len(data), 1),
                                     max(self._room, (len(data) + 254) // 255))] or [b""]
            e.frames = {i: _Frame(i, seg, i == 0, i == len(pieces) - 1, True, cp)
                        for i, seg in enumerate(pieces)}
        e.tx_count = max(e.tx_count, len(e.frames)) if e.frames else 0
        sent_any = False
        for f in sorted(e.frames.values(), key=lambda f: f.seq):
            if f.state == "sent":
                continue
            if f.sent > self.max_retries:
                self._lose(link)
                return
            p = D.DPDU(D.EXP_DATA_ONLY, link.remote, self.address, start=f.start, end=f.end,
                       cpdu_id=e.cpdu_id, seq=f.seq, data=f.data)
            if not fits(p):
                break
            self.counts["resent" if f.sent else "sent"] += 1
            f.state, f.sent = "sent", f.sent + 1
            sent_any = True
        if sent_any:
            self._await(link, now)

    def _arq_sendable(self, link: Link) -> bool:
        a = link.arq
        if any(f.state != "sent" for f in a.frames.values()):
            return True
        return bool(a.pending or a.cursor) and self._window_room(a)

    def _window_room(self, a: _Arq) -> bool:
        return ((a.next_seq - a.tx_lwe) & 0xFF) < self.window if a.frames else True

    def _next_frame(self, a: _Arq) -> _Frame | None:
        """Segment the next piece of the queue into a numbered frame."""
        if not self._window_room(a):
            return None
        while a.cursor is None:
            if not a.pending:
                return None
            cp = a.pending.popleft()
            a.cursor = (cp, 0)
            a.unacked[cp] = [cp, 0]
        cp, off = a.cursor
        seg = cp.data[off:off + self._room]
        end = off + len(seg) >= len(cp.data)
        f = _Frame(a.next_seq, seg, off == 0, end, cp.in_order, cp)
        a.cursor = None if end else (cp, off + len(seg))
        a.frames[f.seq] = f
        a.unacked[cp][1] += 1
        a.next_seq = (a.next_seq + 1) & 0xFF
        return f

    def _compose_arq(self, link: Link, now, fits) -> None:
        a = link.arq
        ack_due = a.owe_ack
        can_send = link.deadline is None or now >= link.deadline
        frames: list[_Frame] = []
        if can_send:
            frames = [f for f in sorted(a.frames.values(), key=lambda f: (f.seq - a.tx_lwe) & 0xFF)
                      if f.state != "sent"]
        sent_any = False

        def ack_fields():
            acks = D.ack_bitmap(a.rx_lwe, a.rx_uwe, a.got)
            return a.rx_lwe, acks

        i = 0
        while can_send:
            if i < len(frames):
                f = frames[i]
                i += 1
            else:
                f = self._next_frame(a)
                if f is None:
                    break
            if f.cpdu.ttd is not None and now > f.cpdu.ttd:
                f.drop = True
                f.data = b""
            if f.sent > self.max_retries:
                self._lose(link)
                return
            ptype = D.DATA_ACK if ack_due else D.DATA_ONLY
            p = D.DPDU(ptype, link.remote, self.address, start=f.start, end=f.end,
                       in_order=f.in_order, drop=f.drop, seq=f.seq, data=f.data,
                       tx_lwe=f.seq == a.tx_lwe)
            if ack_due:
                p.rx_lwe, p.acks = ack_fields()
            if not fits(p):
                break                           # a new frame stays in the window, unsent
            self.counts["resent" if f.sent else "sent"] += 1
            f.state, f.sent = "sent", f.sent + 1
            if ((f.seq - a.tx_uwe) & 0xFF) < 128 and f.seq != a.tx_uwe:
                a.tx_uwe = f.seq
            ack_due = False
            sent_any = True
        if sent_any:
            a.owe_ack = False
            self._await(link, now)
        elif ack_due:
            lwe, acks = ack_fields()
            if fits(D.DPDU(D.ACK_ONLY, link.remote, self.address, rx_lwe=lwe, acks=acks)):
                a.owe_ack = False

    def _await(self, link: Link, now: float) -> None:
        link.deadline = float("inf")            # set properly by ``transmitted``

    def transmitted(self, end: float, answer_by: float) -> None:
        """The transmission just composed went out, ending at ``end``; the
        far end has until ``answer_by`` to begin its reply."""
        for link in self.links.values():
            if link.deadline == float("inf"):
                link.deadline = answer_by
        self._end = end

    def mark_uwe(self, pdus: list[D.DPDU]) -> None:
        """Set TX WIN UWE on each connection's last data frame (C.3.3)."""
        seen = set()
        for p in reversed(pdus):
            if p.type in (D.DATA_ONLY, D.DATA_ACK) and p.dest not in seen:
                seen.add(p.dest)
                link = self.links.get(p.dest)
                if link and p.seq == link.arq.tx_uwe:
                    p.tx_uwe = True

    # -- timers -----------------------------------------------------------

    def hearing(self, remote: int | None, until: float) -> None:
        """The far end is transmitting (a reception is in progress), so its
        answer is on the way: wait at least until ``until``."""
        for link in self.links.values():
            if remote is not None and link.remote != remote:
                continue
            if link.deadline is not None and link.deadline < until:
                link.deadline = until

    def tick(self, now: float) -> None:
        """Deadlines passed: what was sent goes again; too many silences in a
        row and the connection is lost."""
        self._now = now
        for link in list(self.links.values()):
            until = link.drc.get("until") if link.drc else None
            if until is not None and now >= until:
                self._lose(link)
                continue
            if link.deadline is None or now < link.deadline:
                continue
            link.deadline = None
            outstanding = (any(f.state == "sent" for f in link.arq.frames.values())
                           or any(f.state == "sent" for f in link.exp.frames.values())
                           or link.mgmt.out is not None
                           or link.reset_out is not None)
            if not outstanding:
                link.silences = 0
                continue
            link.silences += 1
            if link.silences > self.max_retries:
                self._lose(link)
                continue
            for f in itertools.chain(link.arq.frames.values(), link.exp.frames.values()):
                if f.state == "sent":
                    f.state = "resend"
        self._release_nonarq(now)
        expired = {key for key, until in self.nonarq_deadlines.items() if now >= until}
        if expired:
            kept = deque()
            rejected = set()
            for cp, p, last in self._nonarq_out:
                if (p.dest, p.type, p.cpdu_id) in expired:
                    rejected.add(cp)
                else:
                    kept.append((cp, p, last))
            self._nonarq_out = kept
            for cp in rejected:
                self.upper.cpdu_rejected(cp.handle, Reject.TTL)
        for table in (self._delivered, self.nonarq_deadlines):
            for key, until in list(table.items()):
                if now >= until:
                    del table[key]

    def _lose(self, link: Link) -> None:
        if self.links.pop(link.remote, None) is None:
            return
        for cp in self._outstanding(link):
            self.upper.cpdu_rejected(cp.handle, Reject.LOST)
        self.upper.connection_lost(link.remote)

    def _outstanding(self, link: Link) -> list[CPDU]:
        out = [v[0] for v in link.arq.unacked.values()]
        out += list(link.arq.pending)
        if link.exp.cpdu is not None:
            out.append(link.exp.cpdu)
        out += list(link.exp.queue)
        return out

    # -- reception --------------------------------------------------------

    def for_me(self, p: D.DPDU) -> bool:
        if p.dest == self.address:
            return True
        return p.type in (D.NON_ARQ, D.EXP_NON_ARQ) and p.group and p.dest in self.groups

    def received(self, p: D.DPDU, now: float) -> None:
        """A D_PDU with a good header, from the air."""
        if not self.for_me(p):
            return
        self._now = now
        self.counts["received"] += 1
        if not p.data_ok and p.type in D.DATA_TYPES:
            self.counts["bad_data"] += 1
        link = self.links.get(p.src)
        if link is not None:
            link.heard_at = now
            if link.deadline is not None and link.deadline != float("inf"):
                link.deadline = min(link.deadline, now)
        t = p.type
        if t in (D.NON_ARQ, D.EXP_NON_ARQ):
            self._rx_nonarq(p, now)
            return
        if t == D.WARNING:
            self.upper.warning_received(p.src, p.rx_type, p.reason)
            return
        if t == D.MANAGEMENT:
            self._rx_management(p, link, now)
            return
        if t not in D.NAMES:
            self._warn(p.src, t, D.WARN_UNRECOGNISED)
            return
        if link is None:
            # C.6.1.1.1: connection-related, but there is no connection.
            self._warn(p.src, t, D.WARN_UNCONNECTED)
            return
        if link.reset_out is not None and t != D.RESET_WIN_RESYNC:
            return
        if t in (D.DATA_ONLY, D.DATA_ACK):
            self._rx_data(link, p)
        if t in (D.ACK_ONLY, D.DATA_ACK):
            self._rx_ack(link, p)
        elif t == D.EXP_DATA_ONLY:
            self._rx_exp_data(link, p)
        elif t == D.EXP_ACK_ONLY:
            self._rx_exp_ack(link, p)
        elif t == D.RESET_WIN_RESYNC:
            self._rx_reset(link, p)

    # regular data

    def _rx_data(self, link: Link, p: D.DPDU) -> None:
        a = link.arq
        a.owe_ack = True
        s = p.seq
        low = (a.rx_uwe - (MAX_WINDOW - 1)) & 0xFF
        high = (a.rx_lwe + MAX_WINDOW - 1) & 0xFF
        if (p.tx_uwe and not _between(a.rx_uwe, s, high)) or \
                (p.tx_lwe and not _between(low, s, a.rx_lwe)) or \
                not _between(low, s, high):
            self.full_reset(link.remote)
            return
        behind = (a.rx_lwe - s) & 0xFF
        if 0 < behind <= MAX_WINDOW:
            return
        if not (p.data_ok or p.drop):
            return                              # C.3.4 (6): acknowledged only if good
        if s in a.got:
            return
        a.got.add(s)
        a.held[s] = (p.start, p.end, p.in_order, p.drop, p.data)
        if ((s - a.rx_lwe) & 0xFF) > ((a.rx_uwe - a.rx_lwe) & 0xFF) or a.rx_uwe == \
                ((a.rx_lwe - 1) & 0xFF):
            a.rx_uwe = s
        while a.rx_lwe in a.got:
            a.got.discard(a.rx_lwe)
            a.rx_lwe = (a.rx_lwe + 1) & 0xFF
        if a.rx_uwe == ((a.rx_lwe - 1) & 0xFF) or \
                ((a.rx_uwe - a.rx_lwe) & 0xFF) >= MAX_WINDOW:
            a.rx_uwe = (a.rx_lwe - 1) & 0xFF
        self._assemble(link)

    def _assemble(self, link: Link) -> None:
        """Deliver every C_PDU whose frames are all in: those sent in order
        in the order sent, the others as soon as they are whole."""
        a = link.arq
        held = a.held

        def run(first):
            """The frames of the C_PDU starting at ``first``, if all here."""
            seqs = [first]
            s = first
            while True:
                if held.get(s) is None:
                    return None
                if held[s][1]:                  # end
                    return seqs
                s = (s + 1) & 0xFF
                seqs.append(s)
                if len(seqs) > MAX_WINDOW:
                    return None

        # Out-of-order C_PDUs: whole ones anywhere.
        for s in list(held):
            if held.get(s) is not None and held[s][0] and not held[s][2]:
                seqs = run(s)
                if seqs:
                    self._deliver_run(link, seqs)
        # In order, from where delivery stands.
        while True:
            s = a.deliver_at
            if s not in held:
                break
            if held[s] is None:                 # delivered out of order already
                del held[s]
                a.deliver_at = (s + 1) & 0xFF
                continue
            start, end, _order, drop, data = held.pop(s)
            a.deliver_at = (s + 1) & 0xFF
            if start:
                a.partial.clear()
                a.partial_started = True
                a.partial_drop = False
            if not a.partial_started:
                continue
            a.partial_drop |= drop or len(a.partial) + len(data) > MAX_CPDU
            if not a.partial_drop:
                a.partial.extend(data)
            if end:
                if not a.partial_drop:
                    self.upper.cpdu_received(link.remote, self.address, bytes(a.partial),
                                             expedited=False, arq=True)
                a.partial.clear()
                a.partial_started = False

    def _deliver_run(self, link: Link, seqs) -> None:
        held = link.arq.held
        parts = [held[s] for s in seqs]
        drop = any(x[3] for x in parts)
        data = b"".join(x[4] for x in parts)
        for s in seqs:
            held[s] = None                      # a placeholder for the in-order walk
        if not drop and len(data) <= MAX_CPDU:
            self.upper.cpdu_received(link.remote, self.address, data, expedited=False,
                                     arq=True)

    def _rx_ack(self, link: Link, p: D.DPDU) -> None:
        a = link.arq
        lwe = p.rx_lwe
        # Test 4 (C.6.3.2.2): the far end's lower edge within ours.
        if not _between(a.tx_lwe, lwe, (a.tx_uwe + 1) & 0xFF) or any(
                not 0 < ((s - a.tx_lwe) & 0xFF) <= ((a.tx_uwe - a.tx_lwe) & 0xFF)
                or ((a.tx_uwe - a.tx_lwe) & 0xFF) >= MAX_WINDOW
                or s in a.frames and not a.frames[s].sent
                for s in D.acked(lwe, p.acks)):
            self.full_reset(link.remote)
            return
        acked = {s for s in a.frames if ((s - a.tx_lwe) & 0xFF) < ((lwe - a.tx_lwe) & 0xFF)}
        acked |= {s for s in D.acked(lwe, p.acks) if s in a.frames}
        if acked:
            link.silences = 0
        for s in acked:
            f = a.frames.pop(s)
            entry = a.unacked.get(f.cpdu)
            if entry is not None:
                entry[1] -= 1
                if entry[1] == 0 and (a.cursor is None or a.cursor[0] is not f.cpdu):
                    del a.unacked[f.cpdu]
                    if f.drop:
                        self.upper.cpdu_rejected(f.cpdu.handle, Reject.TTL)
                    else:
                        self.upper.cpdu_confirmed(f.cpdu.handle)
        # Everything else sent before this acknowledgement goes again.
        for f in a.frames.values():
            if f.state == "sent":
                f.state = "resend"
        if a.frames:
            a.tx_lwe = min(a.frames, key=lambda s: (s - a.tx_lwe) & 0xFF)
        else:
            a.tx_lwe = a.next_seq
        link.deadline = None

    # expedited data

    def _rx_exp_data(self, link: Link, p: D.DPDU) -> None:
        e = link.exp
        e.owe_ack = True
        if p.cpdu_id == e.rx_done:
            return                              # a copy of one delivered
        if e.rx_id != p.cpdu_id:
            e.rx_id, e.rx = p.cpdu_id, {}
        if not p.data_ok:
            return
        if sum(len(v[2]) for s, v in e.rx.items() if s != p.seq) + len(p.data) > MAX_CPDU:
            self.full_reset(link.remote)
            return
        e.rx[p.seq] = (p.start, p.end, p.data)
        ends = [s for s, v in e.rx.items() if v[1]]
        if ends and 0 in e.rx and e.rx[0][0] and all(s in e.rx for s in range(ends[0] + 1)):
            data = b"".join(e.rx[s][2] for s in range(ends[0] + 1))
            e.rx_done = p.cpdu_id
            self.upper.cpdu_received(link.remote, self.address, data, expedited=True, arq=True)

    def _rx_exp_ack(self, link: Link, p: D.DPDU) -> None:
        e = link.exp
        if e.cpdu is None:
            return
        if p.rx_lwe > e.tx_count or any(s >= e.tx_count for s in D.acked(p.rx_lwe, p.acks)):
            self.full_reset(link.remote)
            return
        got = set(range(p.rx_lwe))
        got |= set(D.acked(p.rx_lwe, p.acks))
        if any(not f.sent for s, f in e.frames.items() if s in got):
            self.full_reset(link.remote)
            return
        if got.intersection(e.frames):
            link.silences = 0
        for s in list(e.frames):
            if s in got:
                del e.frames[s]
        for f in e.frames.values():
            if f.state == "sent":
                f.state = "resend"
        if not e.frames:
            self.upper.cpdu_confirmed(e.cpdu.handle)
            e.cpdu = None
        link.deadline = None

    # management

    def _rx_management(self, p: D.DPDU, link: Link | None, now: float) -> None:
        if link is None:
            if p.valid and len(self._unconnected_ack) < 64:
                self._unconnected_ack[p.src] = 0
            return
        m = link.mgmt
        if p.ack and m.out is not None:
            acknowledged = m.out[1]
            m.out = None
            link.silences = 0
            if not p.valid:
                self._management_acked(link, acknowledged)
            link.deadline = None
        if not p.valid:
            return
        m.owe_ack = True
        if p.frame_id == m.rx_id:
            return                              # a repeat of one already had
        m.rx_id = p.frame_id
        self._management_message(link, p.eow)

    def _management_message(self, link: Link, eow: int) -> None:
        kind, body = D.eow_fields(eow)
        if kind == D.EOW_DRC_REQUEST:
            rate = D.DRC_RATE_OF.get(body >> 4)
            interleave = D.INTERLEAVE_OF.get((body >> 2) & 3)
            other = body & 3
            if other >= 2:
                return                          # advisory: the decision is ours
            ok = rate is not None and interleave is not None and \
                self.can_receive(rate, interleave)
            if not ok:
                link.drc = {"role": "slave", "phase": "refused",
                            "until": None}
                link.mgmt.queue.append((D.eow_drc_response(D.RESP_REFUSE, D.REASON_NO_RATE),
                                        b"", 1))
                return
            link.drc = {"role": "slave", "phase": "accepted", "rate": rate,
                        "interleave": interleave, "equal": bool(other & 1),
                        "until": None}
            link.mgmt.queue.append((D.eow_drc_response(D.RESP_ACCEPT, D.REASON_NONE), b"", 1))
        elif kind == D.EOW_DRC_RESPONSE:
            resp = body >> 5
            drc = link.drc
            if drc is None:
                return
            if drc["role"] == "master":
                if resp == D.RESP_ACCEPT:
                    if body & 31 == 1:
                        if not self.can_receive(drc["rate"], drc["interleave"]):
                            link.mgmt.queue.append((D.eow_drc_response(D.RESP_CANCEL), b"", 1))
                            drc["phase"] = "cancel"
                            return
                        drc["equal"] = True
                    # The management acknowledgement already on its way is
                    # the DT_ACK; the change follows it (C.6.4.2 (28)).
                    drc["phase"] = "accepted"
                elif resp == D.RESP_CONFIRM:
                    link.drc = None
                else:
                    link.mgmt.queue.append((D.eow_drc_response(D.RESP_CANCEL), b"", 1))
                    drc["phase"] = "cancel"
            elif resp == D.RESP_CANCEL:
                link.drc = None
                link.zero_due = True
        elif kind in (D.EOW_CAPABILITY, D.EOW_UNRECOGNISED, 0):
            pass
        elif kind not in link.mgmt.unknown:
            link.mgmt.unknown.add(kind)
            link.mgmt.queue.append(((D.EOW_UNRECOGNISED << 8) | (kind & 0xF), b"", 1))

    def _management_acked(self, link: Link, eow: int) -> None:
        kind, body = D.eow_fields(eow)
        drc = link.drc
        if drc is None:
            return
        if kind == D.EOW_DRC_RESPONSE and drc["role"] == "master" and \
                body >> 5 == D.RESP_CANCEL:
            link.drc = None
        elif kind == D.EOW_DRC_RESPONSE and drc["role"] == "slave":
            resp = body >> 5
            if resp == D.RESP_ACCEPT:
                # The master's DT_ACK: change, then confirm (C.6.4.2 (29)).
                self._apply_rate(link, drc)
                link.mgmt.queue.append((D.eow_drc_response(D.RESP_CONFIRM), b"", 1))
                drc["phase"] = "confirming"
            elif resp == D.RESP_CONFIRM:
                link.drc = None
                link.zero_due = True

    def _apply_rate(self, link: Link, drc: dict) -> None:
        """Only the master's sending changes, unless both were asked to; the
        receivers of the self-describing waveforms follow by themselves."""
        if drc["role"] == "master" or drc["equal"]:
            self.tx_rate[link.remote] = (drc["rate"], drc["interleave"])
            self.upper.rate_changed(link.remote, drc["rate"], drc["interleave"])

    def can_receive(self, rate: int, interleave: str) -> bool:
        """Whether this node's receiver takes ``rate`` with ``interleave``:
        the self-describing waveforms take anything in their set."""
        return rate in D.DRC_RATES

    def _rate(self, remote: int) -> tuple[int, str]:
        return self.tx_rate.get(remote, self.rate)

    # reset

    def full_reset(self, remote: int) -> None:
        """Start a FULL RESET (C.6.5), e.g. on finding the ARQ machines out
        of step."""
        link = self.links.get(remote)
        if link is not None:
            if link.reset_out is None:
                link.reset_out = {"id": self._reset_id(), "full": True, "sent": 0}
                link.deadline = None

    def _reset_id(self) -> int:
        self._reset_ids = (self._reset_ids + 1) % 256
        return self._reset_ids

    def _rx_reset(self, link: Link, p: D.DPDU) -> None:
        if p.full_reset:
            if p.reset_id != link.reset_rx_id:
                self._reset(link)
                link.reset_rx_id = p.reset_id
            link.reset_ack_due = True
            link.reset_ack_id = p.reset_id
        elif p.reset_ack:
            out = link.reset_out
            if out is not None and p.reset_id == out["id"]:
                if out["full"]:
                    self._reset(link)
                link.reset_out = None
                link.deadline = None
                link.silences = 0
        elif p.reset_tx_win_rqst:
            link.resync_cmd_due = True
            link.deadline = None
        elif p.reset_win_cmnd:
            a = link.arq
            if not _between(a.tx_lwe, p.new_rx_lwe, (a.tx_uwe + 1) & 0xFF):
                self.full_reset(link.remote)
                return
            discarded = set()
            for s in list(a.frames):
                if ((s - a.tx_lwe) & 0xFF) < ((p.new_rx_lwe - a.tx_lwe) & 0xFF):
                    discarded.add(a.frames.pop(s).cpdu)
            for cp in discarded:
                a.unacked.pop(cp, None)
                if a.cursor is not None and a.cursor[0] is cp:
                    a.cursor = None
                self.upper.cpdu_rejected(cp.handle, Reject.RESET)
                for f in a.frames.values():
                    if f.cpdu is cp:
                        f.drop, f.data = True, b""
            a.tx_lwe = p.new_rx_lwe
            link.deadline = None
            for f in a.frames.values():
                f.state = "resend"
            link.reset_ack_due = True
            link.reset_ack_id = p.reset_id

    def _compose_reset(self, link: Link, now, fits) -> None:
        if link.reset_ack_due:
            if fits(D.DPDU(D.RESET_WIN_RESYNC, link.remote, self.address, reset_ack=True,
                           reset_id=link.reset_ack_id)):
                link.reset_ack_due = False
        if link.resync_cmd_due and link.reset_out is None:
            link.reset_out = {"id": self._reset_id(), "full": False, "sent": 0,
                              "edge": link.arq.rx_lwe}
            link.resync_cmd_due = False
        out = link.reset_out
        if out is not None and (link.deadline is None or now >= link.deadline):
            if out["sent"] > self.max_retries:
                self._lose(link)
                return
            if fits(D.DPDU(D.RESET_WIN_RESYNC, link.remote, self.address,
                           full_reset=out["full"], reset_win_cmnd=not out["full"],
                           new_rx_lwe=out.get("edge", 0), reset_id=out["id"])):
                out["sent"] += 1
                self._await(link, now)

    def _reset(self, link: Link) -> None:
        """C.6.5: window edges to their first values, partial C_PDUs out."""
        old = link.arq
        link.arq = _Arq(pending=old.pending)
        for cp in old.unacked:
            self.upper.cpdu_rejected(cp.handle, Reject.RESET)
        if old.cursor and old.cursor[0] not in old.unacked:
            self.upper.cpdu_rejected(old.cursor[0].handle, Reject.RESET)
        old_exp = link.exp
        link.exp = _Expedited(queue=old_exp.queue)
        if old_exp.cpdu is not None:
            self.upper.cpdu_rejected(old_exp.cpdu.handle, Reject.RESET)

    # non-ARQ reception

    def _rx_nonarq(self, p: D.DPDU, now: float) -> None:
        key = (p.src, p.type, p.cpdu_id)
        if key in self._delivered or not 0 <= p.cpdu_size <= MAX_CPDU or \
                p.offset + len(p.data) > p.cpdu_size:
            return
        if not p.data_ok and not self.deliver_with_errors:
            return
        r = self._reassembly.get(key)
        if r is None:
            if len(self._reassembly) >= MAX_REASSEMBLY or \
                    sum(k[0] == p.src for k in self._reassembly) >= MAX_PER_SOURCE:
                return
            r = {"size": p.cpdu_size, "buf": bytearray(p.cpdu_size),
                 "status": bytearray(p.cpdu_size), "dest": p.dest,
                 "until": now + min(p.window / 2, MAX_HOLD_S),
                 "limit": now + MAX_HOLD_S, "in_order": p.in_order}
            self._reassembly[key] = r
        elif r["size"] != p.cpdu_size or r["dest"] != p.dest:
            return
        r["until"] = min(r["limit"], max(r["until"], now + p.window / 2))
        for i, value in enumerate(p.data, p.offset):
            if p.data_ok or r["status"][i] != 1:
                r["buf"][i] = value
                r["status"][i] = 1 if p.data_ok else 2
        self._release_nonarq(now)

    def _release_nonarq(self, now: float) -> None:
        blocked = set()
        for key, r in list(self._reassembly.items()):
            stream = key[:2]
            complete = all(v == 1 for v in r["status"])
            expired = now >= r["until"]
            if not expired and (not complete or r["in_order"] and stream in blocked):
                blocked.add(stream)
                continue
            del self._reassembly[key]
            self._delivered[key] = max(r["until"], now + 1)
            while len(self._delivered) > 256:
                del self._delivered[next(iter(self._delivered))]
            if complete or self.deliver_with_errors:
                self._deliver_nonarq(key, r, complete=complete)

    def _deliver_nonarq(self, key, r, *, complete: bool) -> None:
        src, t, _cid = key
        errors = None
        if not complete:
            bad = [i for i, v in enumerate(r["status"]) if v == 2]
            missing = [i for i, v in enumerate(r["status"]) if v == 0]
            errors = {"errored": _blocks(bad), "missing": _blocks(missing)}
        self.upper.cpdu_received(src, r["dest"], bytes(r["buf"]),
                                 expedited=t == D.EXP_NON_ARQ, arq=False, errors=errors)


def _blocks(positions: list[int]) -> list[tuple[int, int]]:
    """Sorted byte positions -> (first, length) runs."""
    out: list[tuple[int, int]] = []
    for i in positions:
        if out and out[-1][0] + out[-1][1] == i:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((i, 1))
    return out


def shift_errors(errors: dict | None, k: int) -> dict | None:
    """Error runs in a PDU -> the same runs in what follows its first
    ``k`` bytes (its header, taken off)."""
    if errors is None:
        return None
    out = {}
    for name, runs in errors.items():
        out[name] = [(max(0, p - k), n - max(0, k - p)) for p, n in runs if p + n > k]
    return out


def header_damaged(errors: dict | None, size: int) -> bool:
    """Do not route a C/S_PDU using missing or CRC-failed header bytes."""
    return errors is not None and any(
        pos < size and length for runs in errors.values() for pos, length in runs)
