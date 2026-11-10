# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The subnetwork interface sublayer (annex A.1, A.3): the clients' side.

Clients bind to SAPs 0-15 (annex F assigns them: 3 for HMTP mail, 6 for
RCOP, 7 for UDOP, 9 for IP) with a rank and a default service, and hand
down U_PDUs of at most the MTU (2048 bytes, A.2.1.3). The sublayer:

- sends non-ARQ U_PDUs at once, as often as the client's minimum number of
  retransmissions says, to a node or a group, link or none;
- queues ARQ U_PDUs by destination and priority, makes a soft link to the
  destination of the most pressing (A.3.2.1), sends them over it, and
  breaks it when the queue for that node has been empty ``idle_s``, or when
  it has held it ``session_s`` while others wait;
- makes hard links on request (A.3.2.2: one at a time, precedence by rank,
  then priority, then who was first), with the S_PDU handshake;
- confirms delivery to the node (the ARQ acknowledgement below) or to the
  client (the far sublayer's DATA DELIVERY CONFIRM S_PDU), when asked;
- discards what outlives its time to die (TTL from submission; the S_PDU
  carries it as Julian day modulo 16 and seconds of the day, A.3.1.1).

S_PDUs (A.3.1): the type in the top four bits of the first byte, then as
figure A-31 has them.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from . import cas as C
from . import primitives as P
from . import dts as T

DATA = 0
DELIVERY_CONFIRM = 1
DELIVERY_FAIL = 2
HL_REQUEST = 3
HL_CONFIRM = 4
HL_REJECTED = 5
HL_TERMINATE = 6
HL_TERMINATE_CONFIRM = 7

#: DATA DELIVERY FAIL reason (A.3.1.3).
FAIL_SAP_UNBOUND = 1


@dataclass
class Client:
    sap: int
    rank: int
    service: P.ServiceType
    #: Takes a Primitive to the client.
    send: object
    expedited: int = 0
    attachment: object = None


@dataclass(eq=False)
class Outgoing:
    u_pdu: bytes
    src_sap: int
    dest: int
    dest_sap: int
    priority: int
    mode: P.ServiceType
    expedited: bool
    #: On the node's clock; None until the node's next tick, which is when
    #: the time to live starts counting (the clock is only known then).
    ttd: float | None
    group: bool = False
    ttl: float = 0.0
    #: Waiting on: "node" (the ARQ acknowledgement), "client" (that and the
    #: far sublayer's confirmation); what has come of each.
    node_ok: bool = False
    client_ok: bool = False
    deadline: float | None = None
    owner: Client | None = None
    finished: bool = False


@dataclass
class HardLink:
    remote: int
    link_type: int
    priority: int
    rank: int
    requesting_sap: int
    remote_sap: int
    caller: bool
    state: str = "making"       # making | requested | established | terminating
    deadline: float | None = None


def _ttd_field(ttd: float) -> tuple[int, int]:
    t = time.gmtime(ttd)
    secs = t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
    return t.tm_yday % 16, secs >> 1


def _ttd_passed(julian16: int, half_secs: int, now: float) -> bool:
    """Read a day-of-year residue within the next 15 days (A.2.2.20).

    The field has no epoch. This forward-horizon reading is unverified
    against a foreign implementation, including across New Year.
    """
    if half_secs >= 43200:
        return True
    midnight = now - (now % 86400)
    for delta in range(16):
        day = midnight + delta * 86400
        candidate = day + half_secs * 2
        if time.gmtime(day).tm_yday % 16 == julian16 and now <= candidate <= now + P.MAX_TTL:
            return False
    return True


class Subnetwork(C.Upper):
    mtu = 2048                              # A.2.1.3
    default_ttl_s = 3600.0                   # A.2.1.6: local default
    idle_s = 10.0                           # A.3.2.1: local timing
    session_s = 600.0
    confirm_s = 600.0                       # A.3.2.4: local timing
    max_expedited = 10_000                  # A.2.1.5: local resource limit
    broadcast_only = False
    confirm_copy = 64                       # A.2.1.7: returned U_PDU prefix
    max_links = 1
    hard_link_s = 120.0                     # A.3.2.2: local timing
    max_queued = 256
    max_client_queued = 64

    def __init__(self, cas: C.ChannelAccess) -> None:
        self.cas = cas
        cas.upper = self
        self.clients: dict[int, Client] = {}
        self.queue: list[Outgoing] = []
        self.awaiting: list[Outgoing] = []
        self.hard: HardLink | None = None
        self.sessions: dict[int, dict] = {}
        self.now = 0.0
        self.wall = time.time
        self.log = deque(maxlen=200)

    @property
    def address(self) -> int:
        return self.cas.address

    # -- clients ----------------------------------------------------------

    def attach(self, send) -> "Attachment":
        """A client connection: the primitives it sends go to
        ``Attachment.handle``; the sublayer's go to ``send``."""
        return Attachment(self, send)

    def _bind(self, att: "Attachment", p: P.Primitive) -> None:
        sap, rank, service = p["sap"], p["rank"], p["service"]
        if not 0 <= sap <= 15:
            att.send(P.prim(P.BIND_REJECTED, reason=P.BIND_INVALID_SAP))
            return
        if service.mode == P.MODE_DEFAULT:
            # A.2.1.4 has no invalid-service reason; this is a local fallback.
            att.send(P.prim(P.BIND_REJECTED, reason=P.BIND_NO_RESOURCES))
            return
        if self.broadcast_only and service.mode == P.MODE_ARQ:
            att.send(P.prim(P.BIND_REJECTED, reason=P.BIND_BROADCAST_ONLY))
            return
        old = self.clients.get(sap)
        if old is not None:
            if rank <= old.rank:
                att.send(P.prim(P.BIND_REJECTED, reason=P.BIND_SAP_TAKEN))
                return
            old.send(P.prim(P.UNBIND_INDICATION, reason=P.UNBIND_PREEMPTED))
            self._unbind(old.attachment)
        client = Client(sap, rank, service, att.send, attachment=att)
        self.clients[sap] = client
        att.client = client
        self._error_service()
        att.send(P.prim(P.BIND_ACCEPTED, sap=sap, mtu=self.mtu))

    def _error_service(self) -> None:
        self.cas.dts.deliver_with_errors = any(
            c.service.mode == P.MODE_NON_ARQ_ERRORS for c in self.clients.values())

    def _unbind(self, att: "Attachment") -> None:
        client = att.client
        if client is not None and self.clients.get(client.sap) is client:
            del self.clients[client.sap]
            self.queue[:] = [o for o in self.queue if o.owner is not client]
            client.expedited = 0
            dts = self.cas.dts
            cancelled = set()

            def belongs(cp):
                handle = cp.handle
                owned = isinstance(handle, tuple) and len(handle) == 2 and \
                    isinstance(handle[1], Outgoing) and handle[1].owner is client
                if owned:
                    cancelled.add(handle[1])
                return owned

            dts._nonarq_out = deque(item for item in dts._nonarq_out if not belongs(item[0]))
            for link in dts.links.values():
                link.arq.pending = deque(cp for cp in link.arq.pending if not belongs(cp))
                link.exp.queue = deque(cp for cp in link.exp.queue if not belongs(cp))
            # Work already in a transmit window still counts until it completes.
            self.awaiting[:] = [o for o in self.awaiting if o not in cancelled]
            if self.hard is not None and self.hard.requesting_sap == client.sap:
                self._hard_terminate(P.HL_TERM_SAP_UNBOUND)
        att.client = None
        self._error_service()

    def _submit(self, client: Client, p: P.Primitive) -> None:
        expedited = p.type == P.EXPEDITED_UNIDATA_REQUEST
        u = p["u_pdu"]
        mode = p["mode"] if p["mode"].mode != P.MODE_DEFAULT else client.service
        rejected = P.EXPEDITED_UNIDATA_REQUEST_REJECTED if expedited \
            else P.UNIDATA_REQUEST_REJECTED
        if self.clients.get(client.sap) is not client:
            return
        if expedited and client.expedited >= self.max_expedited:
            client.send(P.prim(P.UNBIND_INDICATION, reason=P.UNBIND_TOO_MANY_EXPEDITED))
            self._unbind(client.attachment)
            return
        pending = self.queue + self.awaiting
        if len(pending) >= self.max_queued or \
                sum(o.owner is client for o in pending) >= self.max_client_queued:
            client.send(P.prim(rejected, reason=P.REJECT_NO_MODE, dest=p["dest"],
                               dest_sap=p["dest_sap"], u_pdu=u[:self.confirm_copy]))
            return
        if len(u) > self.mtu:
            client.send(P.prim(rejected, reason=P.REJECT_MTU, dest=p["dest"],
                               dest_sap=p["dest_sap"], u_pdu=u[:self.confirm_copy]))
            return
        if mode.mode == P.MODE_ARQ and self.broadcast_only:
            client.send(P.prim(rejected, reason=P.REJECT_NO_MODE, dest=p["dest"],
                               dest_sap=p["dest_sap"], u_pdu=u[:self.confirm_copy]))
            return
        ttl = min(max(0, p["ttl"] or self.default_ttl_s), P.MAX_TTL)
        if expedited:
            client.expedited += 1
        self.queue.append(Outgoing(u, client.sap, p["dest"], p["dest_sap"],
                                   p.get("priority", 15 if expedited else 0), mode, expedited,
                                   None, group=p.get("group", False), ttl=ttl, owner=client))
        self.queue.sort(key=lambda o: (not o.expedited, -o.priority))

    # -- S_PDUs -----------------------------------------------------------

    def _data_spdu(self, o: Outgoing) -> bytes:
        julian, half = _ttd_field(self.wall() + (o.ttd - self.now))
        confirm = o.mode.confirm == P.CONFIRM_CLIENT and o.mode.mode == P.MODE_ARQ and not o.group
        return bytes([(DATA << 4) | (o.priority & 0xF), (o.src_sap << 4) | o.dest_sap,
                      (confirm << 7) | (1 << 6) | julian]) + half.to_bytes(2, "big") + o.u_pdu

    def _control(self, remote: int, spdu: bytes) -> None:
        """Hard-link and delivery-report S_PDUs: expedited, reliable if
        there is a link, else best effort."""
        arq = self.cas.made(remote)
        self.cas.send(remote, spdu, arq=arq, expedited=True, handle=None)

    # -- scheduling -------------------------------------------------------

    def tick(self, now: float) -> None:
        self.now = now
        for o in self.queue:
            if o.ttd is None:
                o.ttd = now + o.ttl
        # Out of time.
        for o in [o for o in self.queue if now > o.ttd]:
            self.queue.remove(o)
            self._report(o, False, P.REJECT_TTL)
        for o in [o for o in self.awaiting if o.deadline is not None and now > o.deadline]:
            self.awaiting.remove(o)
            self._report(o, False, P.REJECT_NOT_RESPONDING)
        self._tick_hard(now)
        # Non-ARQ: out at once, link or none.
        for o in [o for o in self.queue if o.mode.mode != P.MODE_ARQ]:
            self.queue.remove(o)
            self.awaiting.append(o)
            self.cas.send(o.dest, self._data_spdu(o), arq=False, expedited=o.expedited,
                          in_order=o.mode.in_order, repeats=o.mode.retransmissions,
                          group=o.group, priority=o.priority, ttd=o.ttd, handle=o)
        # ARQ over links made.
        for o in [o for o in self.queue if self.cas.made(o.dest) and self._allowed(o)]:
            self.queue.remove(o)
            if o.mode.confirm != P.CONFIRM_NONE:
                o.deadline = now + self.confirm_s
            self.awaiting.append(o)
            self.cas.send(o.dest, self._data_spdu(o), arq=True, expedited=o.expedited,
                          in_order=o.mode.in_order, priority=o.priority, ttd=o.ttd, handle=o)
            s = self.sessions.setdefault(o.dest, {"since": now, "last": now})
            s["last"] = now
        # Soft links: make one to the most pressing destination waiting.
        waiting = [o for o in self.queue if o.mode.mode == P.MODE_ARQ]
        linked = [r for r, l in self.cas.links.items() if l.state in ("made", "making", "breaking")]
        if waiting and len(linked) < self.max_links:
            dest = waiting[0].dest
            if dest not in self.cas.links:
                self.cas.make(dest, exclusive=False)
        # And break the ones idle, or held too long while others wait.
        for remote, s in list(self.sessions.items()):
            link = self.cas.links.get(remote)
            if link is None or link.state != "made":
                continue
            if self.hard is not None and self.hard.remote == remote:
                s["last"] = now
                continue
            busy = any(o.dest == remote for o in self.queue) or \
                self.cas.dts.links.get(remote) is not None and self._dts_busy(remote)
            if busy:
                s["last"] = now
            others = any(o.dest != remote for o in waiting)
            if not link.caller:
                continue
            if (not busy and now - s["last"] > self.idle_s) or \
                    (others and now - s["since"] > self.session_s):
                self.cas.break_link(remote)

    def _dts_busy(self, remote: int) -> bool:
        link = self.cas.dts.links.get(remote)
        if link is None:
            return False
        a = link.arq
        return bool(a.frames or a.pending or a.cursor or link.exp.cpdu or link.exp.queue
                    or link.mgmt.out or link.mgmt.queue)

    def _allowed(self, o: Outgoing) -> bool:
        """Types 1 and 2 hard links carry only their own clients (A.1.1.2)."""
        h = self.hard
        if h is None or h.remote != o.dest or h.link_type == 0:
            return True
        if h.link_type == 1:
            return o.src_sap == h.requesting_sap
        return o.src_sap == h.requesting_sap and o.dest_sap == h.remote_sap

    def _report(self, o: Outgoing, ok: bool, reason: int = 0) -> None:
        if o.finished:
            return
        o.finished = True
        if o.expedited and o.owner is not None:
            o.owner.expedited = max(0, o.owner.expedited - 1)
        client = self.clients.get(o.src_sap)
        if client is not o.owner or client is None or o.mode.confirm == P.CONFIRM_NONE:
            return                              # A.2.1.7, A.2.1.8: only when asked
        if ok:
            t = P.EXPEDITED_UNIDATA_REQUEST_CONFIRM if o.expedited else P.UNIDATA_REQUEST_CONFIRM
            client.send(P.prim(t, dest=o.dest, dest_sap=o.dest_sap,
                               u_pdu=o.u_pdu[:self.confirm_copy]))
        else:
            t = P.EXPEDITED_UNIDATA_REQUEST_REJECTED if o.expedited \
                else P.UNIDATA_REQUEST_REJECTED
            client.send(P.prim(t, reason=reason, dest=o.dest, dest_sap=o.dest_sap,
                               u_pdu=o.u_pdu[:self.confirm_copy]))

    # -- from the channel access sublayer ----------------------------------

    def spdu_confirmed(self, handle) -> None:
        o = handle
        if not isinstance(o, Outgoing) or o not in self.awaiting:
            return
        o.node_ok = True
        if o.mode.confirm != P.CONFIRM_CLIENT or o.mode.mode != P.MODE_ARQ or o.client_ok:
            self.awaiting.remove(o)
            self._report(o, True)

    def spdu_rejected(self, handle, reason: str) -> None:
        o = handle
        if not isinstance(o, Outgoing):
            return
        if o in self.awaiting:
            self.awaiting.remove(o)
        why = P.REJECT_TTL if reason == T.Reject.TTL else P.REJECT_NOT_RESPONDING
        self._report(o, False, why)

    def link_made(self, remote, exclusive, caller) -> None:
        self.sessions[remote] = {"since": self.now, "last": self.now}
        h = self.hard
        if h is not None and h.remote == remote and h.caller and h.state == "making":
            h.state = "requested"
            h.deadline = self.now + self.hard_link_s
            self._control(remote, bytes([(HL_REQUEST << 4) | ((h.link_type & 3) << 2)
                                         | (h.priority & 3),
                                         (h.requesting_sap << 4) | h.remote_sap]))

    def link_failed(self, remote, reason) -> None:
        for o in [o for o in self.queue if o.dest == remote and o.mode.mode == P.MODE_ARQ]:
            self.queue.remove(o)
            self._report(o, False, P.REJECT_NOT_RESPONDING)
        h = self.hard
        if h is not None and h.remote == remote:
            self._hard_rejected(P.HL_NOT_RESPONDING)

    def link_broken(self, remote, reason) -> None:
        self.sessions.pop(remote, None)
        h = self.hard
        if h is not None and h.remote == remote:
            self._hard_terminated(P.HL_TERM_PHYSICAL)

    def spdu_received(self, src, dest, spdu, *, expedited, arq, errors=None) -> None:
        if not spdu or T.header_damaged(errors, 1):
            return
        kind = spdu[0] >> 4
        if kind != DATA and errors is not None:
            return
        if kind == DATA:
            self._rx_data(src, dest, spdu, expedited, arq, errors)
        elif kind in (DELIVERY_CONFIRM, DELIVERY_FAIL) and len(spdu) >= 2:
            self._rx_report(src, spdu)
        elif kind in (HL_REQUEST, HL_CONFIRM, HL_REJECTED, HL_TERMINATE, HL_TERMINATE_CONFIRM):
            if self.cas.made(src):
                self._rx_hard(src, spdu)
        s = self.sessions.get(src)
        if s is not None:
            s["last"] = self.now

    def _rx_data(self, src, dest, spdu, expedited, arq, errors) -> None:
        if len(spdu) < 3:
            return
        prio = spdu[0] & 0xF
        src_sap, dest_sap = spdu[1] >> 4, spdu[1] & 0xF
        confirm, valid_ttd = bool(spdu[2] & 0x80), bool(spdu[2] & 0x40)
        at = 3
        if valid_ttd:
            if len(spdu) < 5:
                return
            if _ttd_passed(spdu[2] & 0xF, int.from_bytes(spdu[3:5], "big"), self.wall()):
                return
            at = 5
        if T.header_damaged(errors, at):
            return
        u = spdu[at:]
        client = self.clients.get(dest_sap)
        if client is not None:
            if errors is not None and client.service.mode != P.MODE_NON_ARQ_ERRORS:
                return
            if errors is not None:
                mode = P.MODE_NON_ARQ_ERRORS
            else:
                mode = P.MODE_ARQ if arq else P.MODE_NON_ARQ
            t = P.EXPEDITED_UNIDATA_INDICATION if expedited else P.UNIDATA_INDICATION
            args = dict(priority=prio, dest_sap=dest_sap, dest=dest, mode=mode,
                        src_sap=src_sap, src=src, u_pdu=u)
            if errors is not None:
                errors = T.shift_errors(errors, at)
                args.update(errored=errors["errored"], missing=errors["missing"])
            client.send(P.Primitive(t, args))
        if confirm and arq and dest == self.address and errors is None:
            head = spdu[1:2]
            if client is not None:
                self._control(src, bytes([DELIVERY_CONFIRM << 4]) + head + u[:self.confirm_copy])
            else:
                self._control(src, bytes([(DELIVERY_FAIL << 4) | FAIL_SAP_UNBOUND]) + head
                              + u[:self.confirm_copy])

    def _rx_report(self, src, spdu) -> None:
        kind = spdu[0] >> 4
        src_sap, dest_sap = spdu[1] >> 4, spdu[1] & 0xF
        copy = spdu[2:]
        for o in self.awaiting:
            if o.dest == src and o.src_sap == src_sap and o.dest_sap == dest_sap \
                    and o.u_pdu.startswith(copy):
                if kind == DELIVERY_CONFIRM:
                    o.client_ok = True
                    if o.node_ok:
                        self.awaiting.remove(o)
                        self._report(o, True)
                else:
                    self.awaiting.remove(o)
                    self._report(o, False, P.REJECT_SAP_UNBOUND)
                return

    # -- hard links (A.3.2.2) ---------------------------------------------

    def _precedes(self, rank, prio, ltype, h: HardLink) -> bool:
        """Whether a request (rank, priority, type) takes precedence over
        the existing hard link ``h`` (A.3.2.2.1)."""
        if rank != h.rank:
            return rank > h.rank
        if prio != h.priority:
            return prio > h.priority
        return ltype > h.link_type

    def _hard_establish(self, client: Client, p: P.Primitive) -> None:
        remote, ltype, prio, rsap = p["remote"], p["link_type"], p["link_priority"], \
            p["remote_sap"]
        h = self.hard
        rej = dict(link_type=ltype, link_priority=prio, remote_sap=rsap, remote=remote)
        if h is not None:
            if h.remote == remote and ltype == 0 and h.link_type == 0 \
                    and client.rank == h.rank and prio == h.priority:
                client.send(P.prim(P.HARD_LINK_REJECTED, reason=P.HL_TYPE0_EXISTS, **rej))
                return
            if not self._precedes(client.rank, prio, ltype, h):
                client.send(P.prim(P.HARD_LINK_REJECTED, reason=P.HL_HIGHER_PRIORITY, **rej))
                return
            self._hard_terminate(P.HL_TERM_HIGHER)
        self.hard = HardLink(remote, ltype, prio, client.rank, client.sap, rsap, True)
        self.cas.make(remote, exclusive=True)

    def _hard_terminate(self, reason: int) -> None:
        h = self.hard
        if h is None:
            return
        h.state = "terminating"
        h.deadline = self.now + self.hard_link_s
        self._control(h.remote, bytes([(HL_TERMINATE << 4) | (reason & 0xF)]))

    def _hard_terminated(self, reason: int) -> None:
        h, self.hard = self.hard, None
        if h is None:
            return
        for c in self.clients.values():
            if c.sap == h.requesting_sap or h.link_type == 0 or not h.caller:
                c.send(P.prim(P.HARD_LINK_TERMINATED, reason=reason, link_type=h.link_type,
                              link_priority=h.priority, remote_sap=h.remote_sap,
                              remote=h.remote))
        if h.caller and self.cas.made(h.remote):
            self.cas.break_link(h.remote)

    def _hard_rejected(self, reason: int) -> None:
        h, self.hard = self.hard, None
        if h is None:
            return
        client = self.clients.get(h.requesting_sap)
        if client is not None:
            client.send(P.prim(P.HARD_LINK_REJECTED, reason=reason, link_type=h.link_type,
                               link_priority=h.priority, remote_sap=h.remote_sap,
                               remote=h.remote))

    def _tick_hard(self, now: float) -> None:
        h = self.hard
        if h is None or h.deadline is None or now < h.deadline:
            return
        if h.state == "requested":
            self._hard_rejected(P.HL_NOT_RESPONDING)
        elif h.state == "indicated":
            self._control(h.remote, bytes([(HL_REJECTED << 4) | P.HL_REMOTE_BUSY]))
            self.hard = None
        elif h.state == "terminating":
            self._hard_terminated(P.HL_TERM_NOT_RESPONDING)

    def _rx_hard(self, src, spdu) -> None:
        kind, low = spdu[0] >> 4, spdu[0] & 0xF
        h = self.hard
        if kind == HL_REQUEST and len(spdu) >= 2:
            ltype, prio = (low >> 2) & 3, low & 3
            req_sap, rsap = spdu[1] >> 4, spdu[1] & 0xF
            if ltype == 2 and rsap not in self.clients:
                self._control(src, bytes([(HL_REJECTED << 4) | P.HL_SAP_UNBOUND]))
                return
            rank = self.clients[rsap].rank if ltype == 2 else 0
            if h is not None and h.remote != src and not self._precedes(rank, prio, ltype, h):
                reason = P.HL_REMOTE_BUSY if (rank, prio) == (h.rank, h.priority) \
                    else P.HL_HIGHER_PRIORITY
                self._control(src, bytes([(HL_REJECTED << 4) | reason]))
                return
            if h is not None and h.remote != src:
                self._hard_terminate(P.HL_TERM_HIGHER)
            self.hard = HardLink(src, ltype, prio, rank, rsap, req_sap, False)
            if ltype == 2:
                self.hard.state = "indicated"
                self.hard.deadline = self.now + self.hard_link_s
                self.clients[rsap].send(P.prim(P.HARD_LINK_INDICATION, status=0,
                                               link_type=ltype, link_priority=prio,
                                               remote_sap=req_sap, remote=src))
                return
            self.hard.state = "established"
            self._control(src, bytes([HL_CONFIRM << 4]))
        elif kind == HL_CONFIRM and h is not None and h.remote == src and h.caller:
            h.state, h.deadline = "established", None
            client = self.clients.get(h.requesting_sap)
            if client is not None:
                client.send(P.prim(P.HARD_LINK_ESTABLISHED, status=0, link_type=h.link_type,
                                   link_priority=h.priority, remote_sap=h.remote_sap,
                                   remote=h.remote))
        elif kind == HL_REJECTED and h is not None and h.remote == src and h.caller:
            self._hard_rejected(low)
            self.cas.break_link(src)
        elif kind == HL_TERMINATE:
            self._control(src, bytes([HL_TERMINATE_CONFIRM << 4]))
            if h is not None and h.remote == src:
                self._hard_terminated(low)
        elif kind == HL_TERMINATE_CONFIRM and h is not None and h.remote == src:
            self._hard_terminated(P.HL_TERM_CLIENT)

    def _hard_answer(self, client: Client, p: P.Primitive, accept: bool) -> None:
        h = self.hard
        if h is None or h.state != "indicated" or h.remote != p["remote"]:
            return
        if accept:
            h.state, h.deadline = "established", None
            self._control(h.remote, bytes([HL_CONFIRM << 4]))
        else:
            self._control(h.remote, bytes([(HL_REJECTED << 4) | (p.get("reason", 0) & 0xF)]))
            self.hard = None


class Attachment:
    """One client's connection to the sublayer."""

    def __init__(self, sis: Subnetwork, send) -> None:
        self.sis = sis
        self.send = send
        self.client: Client | None = None

    def handle(self, p: P.Primitive) -> None:
        s, c = self.sis, self.client
        t = p.type
        if t == P.BIND_REQUEST:
            if c is not None:
                self.send(P.prim(P.BIND_REJECTED, reason=P.BIND_SAP_TAKEN))
                return
            s._bind(self, p)
        elif t == P.UNBIND_REQUEST:
            s._unbind(self)
        elif t == P.KEEP_ALIVE:
            self.send(P.prim(P.KEEP_ALIVE))
        elif c is None:
            return                              # nothing before a bind
        elif t in (P.UNIDATA_REQUEST, P.EXPEDITED_UNIDATA_REQUEST):
            s._submit(c, p)
        elif t == P.HARD_LINK_ESTABLISH:
            s._hard_establish(c, p)
        elif t == P.HARD_LINK_TERMINATE:
            h = s.hard
            if h is not None and h.remote == p["remote"] and (h.requesting_sap == c.sap
                                                              or h.remote_sap == c.sap):
                s._hard_terminate(P.HL_TERM_CLIENT)
        elif t == P.HARD_LINK_ACCEPT:
            s._hard_answer(c, p, True)
        elif t == P.HARD_LINK_REJECT:
            s._hard_answer(c, p, False)

    def close(self) -> None:
        self.sis._unbind(self)
