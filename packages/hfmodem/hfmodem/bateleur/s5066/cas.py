# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The channel access sublayer (annex B): physical links, and C_PDUs.

A C_PDU is a byte of protocol control, its type in the top four bits, and
for DATA the S_PDU after it (B.3.1). Links are made and broken by the
exchange of B.3.2, every C_PDU of it sent by expedited non-ARQ delivery
(type 8 D_PDUs), there being no connection yet (or soon none) for ARQ:

- the caller sends PHYSICAL LINK REQUEST and waits; ACCEPTED makes the link,
  REJECTED fails it, silence repeats the request, up to a limit;
- the called node accepts a nonexclusive link unless it holds an exclusive
  one or already its most nonexclusive, and an exclusive link while it has
  at most one; it declares the link made as it accepts;
- either end breaks a link with PHYSICAL LINK BREAK, which the other
  declares broken at once and answers with BREAK CONFIRM.

A link made is a connection for the data transfer sublayer below
(D_CONNECTION_MADE), and a physical link for the subnetwork interface
sublayer above.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from . import dts as T

DATA = 0
LINK_REQUEST = 1
LINK_ACCEPTED = 2
LINK_REJECTED = 3
LINK_BREAK = 4
LINK_BREAK_CONFIRM = 5

# Reasons (B.3.1.4, B.3.1.5).
REJECT_UNKNOWN, REJECT_BROADCAST_ONLY, REJECT_HIGHER_PRIORITY, REJECT_EXCLUSIVE = 0, 1, 2, 3
BREAK_UNKNOWN, BREAK_HIGHER_LAYER, BREAK_TO_BROADCAST, BREAK_HIGHER_PRIORITY = 0, 1, 2, 3


class Upper:
    """What the sublayer reports to the subnetwork interface sublayer."""

    def link_made(self, remote: int, exclusive: bool, caller: bool) -> None: ...
    def link_failed(self, remote: int, reason: str) -> None: ...
    def link_broken(self, remote: int, reason: str) -> None: ...
    def spdu_received(self, src: int, dest: int, spdu: bytes, *, expedited: bool, arq: bool,
                      errors=None) -> None: ...
    def spdu_confirmed(self, handle) -> None: ...
    def spdu_rejected(self, handle, reason: str) -> None: ...


@dataclass
class PhysicalLink:
    remote: int
    exclusive: bool
    caller: bool
    state: str                          # making | made | breaking
    tries: int = 0
    deadline: float | None = None       # waiting for an answer until
    first_data: bool = False            # a DATA C_PDU has come (B.3.2.1 d)
    made_at: float = 0.0


class ChannelAccess(T.Upper):
    """``answer_s``: how long to wait, after a request has gone out, for the
    answer; ``tries``: how many requests before giving up (B.3.2.1 (4),
    (23)); ``backoff_s``: up to how much longer, drawn at random, to wait
    before asking again, so that two nodes whose requests collided do not
    collide again; ``first_data_s``: how long a called node keeps a link made that
    carries no data (B.3.2.1 (17)); ``max_nonexclusive``: how many
    nonexclusive links at once."""

    tries = 3                              # B.3.2.1: local retry limit
    first_data_s = 120.0                    # B.3.2.1 (17): local timeout
    max_nonexclusive = 4
    broadcast_only = False

    def __init__(self, dts: T.DataTransfer, upper: Upper | None = None, *,
                 answer_s: float = 20.0) -> None:
        self.dts = dts
        dts.upper = self
        self.upper = upper or Upper()
        self.answer_s = answer_s
        self.backoff_s = answer_s
        self.rng = random.Random(dts.address)
        self.links: dict[int, PhysicalLink] = {}
        self.now = 0.0
        #: Set by the node: when the transmission now going out will end.
        self.tx_end = 0.0

    @property
    def address(self) -> int:
        return self.dts.address

    # -- requests from above ----------------------------------------------

    def make(self, remote: int, exclusive: bool = False) -> None:
        """C_PHYSICAL_LINK_MAKE."""
        link = self.links.get(remote)
        if link is not None and link.state in ("made", "making"):
            if link.state == "made":
                self.upper.link_made(remote, link.exclusive, link.caller)
            return
        self.links[remote] = PhysicalLink(remote, exclusive, True, "making")
        self._request(self.links[remote])

    def break_link(self, remote: int, reason: int = BREAK_HIGHER_LAYER) -> None:
        """C_PHYSICAL_LINK_BREAK."""
        link = self.links.get(remote)
        if link is None:
            return
        link.state, link.tries, link.deadline = "breaking", 1, None
        self._send_control(remote, bytes([(LINK_BREAK << 4) | (reason & 0xF)]),
                           ("break", remote))

    def send(self, remote: int, spdu: bytes, *, arq: bool, expedited: bool = False,
             in_order: bool = True, repeats: int = 0, group: bool = False,
             priority: int = 0, ttd: float | None = None, handle=None) -> bool:
        """C_UNIDATA_REQUEST / C_EXPEDITED_UNIDATA_REQUEST: a DATA C_PDU."""
        return self.dts.submit(T.CPDU(bytes([DATA << 4]) + spdu, remote, arq=arq,
                                      expedited=expedited, in_order=in_order, repeats=repeats,
                                      group=group, priority=priority, ttd=ttd,
                                      handle=("data", handle)))

    def made(self, remote: int) -> bool:
        link = self.links.get(remote)
        return link is not None and link.state == "made"

    # -- the protocol -----------------------------------------------------

    def _send_control(self, remote: int, cpdu: bytes, handle) -> None:
        self.dts.submit(T.CPDU(cpdu, remote, arq=False, expedited=True, handle=handle))

    def _request(self, link: PhysicalLink) -> None:
        link.tries += 1
        link.deadline = None
        self._send_control(link.remote, bytes([(LINK_REQUEST << 4) | int(link.exclusive)]),
                           ("request", link.remote))

    def tick(self, now: float) -> None:
        self.now = now
        for link in list(self.links.values()):
            conn = self.dts.links.get(link.remote)
            if conn is not None and conn.heard_at is not None and conn.heard_at > link.made_at:
                # D_PDUs of the connection coming in: its data is on the way,
                # a C_PDU of it whole or not.
                link.first_data = True
            if link.state == "made" and not link.caller and not link.first_data \
                    and now - link.made_at > self.first_data_s:
                # B.3.2.1 (16): made, but nothing came on it.
                self._broken(link.remote, "no data on a new link")
                continue
            if link.deadline is None or now < link.deadline:
                continue
            if link.state == "making":
                if link.tries >= self.tries:
                    del self.links[link.remote]
                    self.upper.link_failed(link.remote, "no answer")
                else:
                    self._request(link)
            elif link.state == "breaking":
                if link.tries >= self.tries:
                    self._broken(link.remote, "no answer to break")
                else:
                    link.tries += 1
                    link.deadline = None
                    self._send_control(link.remote, bytes([(LINK_BREAK << 4)]),
                                       ("break", link.remote))

    def hearing(self, until: float) -> None:
        """The far end is transmitting: an answer may be in it."""
        for link in self.links.values():
            if link.deadline is not None and link.deadline < until:
                link.deadline = until

    def _made(self, link: PhysicalLink) -> None:
        link.state, link.deadline, link.made_at = "made", None, self.now
        if link.exclusive and link.caller:
            # B.3.2.1 (19): an exclusive link breaks the nonexclusive ones.
            for other in list(self.links.values()):
                if other is not link and not other.exclusive and other.state == "made":
                    self.break_link(other.remote, BREAK_HIGHER_PRIORITY)
        self.dts.connection_made(link.remote)
        self.upper.link_made(link.remote, link.exclusive, link.caller)

    def _broken(self, remote: int, reason: str) -> None:
        link = self.links.pop(remote, None)
        self.dts.connection_terminated(remote)
        if link is not None:
            self.upper.link_broken(remote, reason)

    def _accepts(self, exclusive: bool) -> int | None:
        """None to accept; else the reason to reject (B.3.2.1 step 2)."""
        if self.broadcast_only:
            return REJECT_BROADCAST_ONLY
        made = [l for l in self.links.values() if l.state == "made"]
        n_excl = sum(l.exclusive for l in made)
        if exclusive:
            return None if n_excl <= 1 else REJECT_EXCLUSIVE
        if n_excl:
            return REJECT_EXCLUSIVE
        if sum(not l.exclusive for l in made) >= self.max_nonexclusive:
            return REJECT_UNKNOWN
        return None

    # -- from the data transfer sublayer ----------------------------------

    def cpdu_received(self, src, dest, data, *, expedited, arq, errors=None) -> None:
        if not data:
            return
        if T.header_damaged(errors, 1):
            return
        kind, low = data[0] >> 4, data[0] & 0xF
        if kind == DATA:
            link = self.links.get(src)
            if link is not None:
                link.first_data = True
            self.upper.spdu_received(src, dest, data[1:], expedited=expedited, arq=arq,
                                     errors=T.shift_errors(errors, 1))
            return
        if dest != self.address:
            return                              # link control is never to a group
        if kind == LINK_REQUEST:
            exclusive = bool(low & 1)
            link = self.links.get(src)
            if link is not None and link.state == "made":
                if link.first_data:
                    self.dts.connection_terminated(src)
                    self.dts.connection_made(src)
                    link.first_data = False
                    link.made_at = self.now
                self._send_control(src, bytes([LINK_ACCEPTED << 4]), ("accept", src))
                return
            reason = self._accepts(exclusive)
            if reason is not None:
                self._send_control(src, bytes([(LINK_REJECTED << 4) | reason]),
                                   ("reject", src))
                return
            # A request crossing our own to the same node: answer it; the
            # link is made from both ends at once.
            link = PhysicalLink(src, exclusive, False, "making")
            self.links[src] = link
            self._send_control(src, bytes([LINK_ACCEPTED << 4]), ("accept", src))
            self._made(link)
        elif kind == LINK_ACCEPTED:
            link = self.links.get(src)
            if link is not None and link.state == "making" and link.caller:
                self._made(link)
        elif kind == LINK_REJECTED:
            link = self.links.get(src)
            if link is not None and link.state == "making":
                del self.links[src]
                self.upper.link_failed(src, f"rejected ({low})")
        elif kind == LINK_BREAK:
            self._send_control(src, bytes([LINK_BREAK_CONFIRM << 4]), ("confirm", src))
            if src in self.links:
                self._broken(src, f"broken by the far end ({low})")
        elif kind == LINK_BREAK_CONFIRM:
            link = self.links.get(src)
            if link is not None and link.state == "breaking":
                self._broken(src, "broken")

    def cpdu_confirmed(self, handle) -> None:
        kind, what = handle if isinstance(handle, tuple) else (None, None)
        if kind == "data":
            self.upper.spdu_confirmed(what)
        elif kind in ("request", "break"):
            # The request has gone out (non-ARQ: sent, not delivered); its
            # answer is due within ``answer_s`` of the transmission's end.
            link = self.links.get(what)
            if link is not None and link.state in ("making", "breaking"):
                link.deadline = max(self.tx_end, self.now) + self.answer_s \
                    + self.rng.uniform(0, self.backoff_s)

    def cpdu_rejected(self, handle, reason: str) -> None:
        kind, what = handle if isinstance(handle, tuple) else (None, None)
        if kind == "data":
            self.upper.spdu_rejected(what, reason)

    def connection_lost(self, remote: int) -> None:
        self._broken(remote, "connection lost")

    def warning_received(self, src, rx_type, reason) -> None:
        if reason == 1 and src in self.links and self.links[src].state == "made":
            # The far end has no connection with us: the link is gone.
            self._broken(src, "the far end holds no link")

    def rate_changed(self, remote, rate, interleave) -> None:
        self.on_rate(remote, rate, interleave)

    def on_rate(self, remote: int, rate: int, interleave: str) -> None:
        """Replaced by the node: the modem's data rate is to change."""
