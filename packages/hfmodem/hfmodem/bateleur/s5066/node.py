# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A STANAG 5066 node: the three sublayers, on a half-duplex modem.

The node is driven from outside, as the modem is: it is told what the
receiver hears (a transmission beginning, its bytes, its end) and asked,
whenever the station could key, whether it has a transmission to make. It
keeps no clock of its own; every call brings ``now``.

**Whose turn.** A node keys only when the channel is free: not while it is
hearing a transmission, and not before ``turnaround_s`` after it ends. The
receiver says when a transmission ends; if it lost it before its end (a
fade), the far end's EOT says when it will (C.3.2.3 (3)): the EOT counts
from the start of the D_PDU that carries it, which began at least that
D_PDU's own airtime before its last byte was heard. Having
sent what needs an answer, it waits ``answer_s`` past the end of its
transmission for the answer to begin before it sends again.

**EOT.** Each D_PDU carries the half seconds left in the transmission from
its own start, rounded up (C.3.2.3), reckoned from the modem's timing: a
lead (the preamble) before the first byte, the data rate, and a tail (the
interleaver's flush and the end-of-message) after the last.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, replace

from . import cas as C
from . import dpdu as D
from . import dts as T
from . import sis as S


@dataclass(frozen=True)
class Timing:
    """How long the modem takes to send bytes."""
    rate: int                               # bit/s
    lead: float = 0.0                       # before the first data byte
    tail: float = 0.0                       # after the last

    def airtime(self, n: int) -> float:
        return self.lead + 8 * n / self.rate + self.tail

    def at(self, k: int) -> float:
        """When byte ``k`` of the transmission starts."""
        return self.lead + 8 * k / self.rate

    def most_bytes(self, max_s: float) -> int:
        """The most bytes a transmission of at most ``max_s`` carries."""
        return max(0, int((max_s - self.lead - self.tail) * self.rate / 8))

    def at_rate(self, rate: int, interleave: str) -> "Timing":
        """The modem's timing once it sends at ``rate``."""
        return replace(self, rate=rate)


class Node:
    turnaround_s = 1.0

    def __init__(self, address: int | str, timing: Timing, *, segment: int = 200,
                 max_tx_s: float = 60.0, groups=(), max_retries: int = 6) -> None:
        self.address = D.address(address)
        self.timing = timing
        self.max_tx_s = min(max_tx_s, 127.5)
        self.answer_s = 2 * self.turnaround_s + 2 * timing.lead + 3.0
        self.dts = T.DataTransfer(self.address, segment=segment, groups=groups,
                                  max_retries=max_retries)
        self.dts.rate = (timing.rate, "short")
        self.cas = C.ChannelAccess(self.dts, answer_s=self.answer_s)
        self.cas.on_rate = self._rate_changed
        self.sis = S.Subnetwork(self.cas)
        self.parser = D.Parser()
        self.receiving = False
        self.free_at = 0.0
        self.eot_end = 0.0
        self.busy_until = 0.0
        self.transmissions = 0
        self.heard: deque[D.DPDU] = deque(maxlen=256)
        self._prepared = None
        self.dts.can_receive = self._can_receive

    def attach(self, send) -> S.Attachment:
        """A client: ``send`` takes the primitives the node has for it."""
        return self.sis.attach(send)

    # -- time -------------------------------------------------------------

    def tick(self, now: float) -> None:
        self.cas.now = now
        self.sis.tick(now)
        if not self.receiving:
            self.cas.tick(now)
            self.dts.tick(now)

    def may_key(self, now: float) -> bool:
        return not self.receiving and now >= self.free_at and now >= self.busy_until

    def poll(self, now: float, *, defer: bool = False) -> tuple[bytes, float] | None:
        """(the bytes of a transmission to make now, how long it keys), or
        None."""
        if self._prepared is not None:
            return None
        self.tick(now)
        if not self.may_key(now) or not self.dts.pending(now):
            return None
        budget = self.timing.most_bytes(self.max_tx_s)
        if budget <= 0:
            return None
        memo = {id(self.cas): self.cas, id(self): self}
        for link in self.dts.links.values():
            for cp in self.dts._outstanding(link):
                memo[id(cp)] = cp
        for cp, _p, _last in self.dts._nonarq_out:
            memo[id(cp)] = cp
        saved = {k: v if callable(v) or k == "upper" else copy.deepcopy(v, memo)
                 for k, v in self.dts.__dict__.items()}
        events = []
        upper = self.dts.upper
        self.dts.upper = _Deferred(events)
        try:
            pdus = self.dts.compose(now, budget)
        finally:
            self.dts.upper = upper
        if not pdus:
            for name, args in events:
                getattr(upper, name)(*args)
            return None
        self.dts.mark_uwe(pdus)
        sizes = [D.size(p) for p in pdus]
        total = self.timing.airtime(sum(sizes))
        before = 0
        for p, n in zip(pdus, sizes):
            p.eot = min(255, math.ceil(2 * (total - self.timing.at(before)) - 1e-9))
            before += n
        self._windows(now, pdus, sizes, total, budget)
        data = b"".join(D.encode(p) for p in pdus)
        self._prepared = (saved, events, now, total)
        if not defer:
            self.sent(True)
        return data, total

    def sent(self, accepted: bool, at: float | None = None) -> None:
        """Commit a composed turn only after the modem accepts it."""
        prepared, self._prepared = self._prepared, None
        if prepared is None:
            return
        saved, events, now, total = prepared
        if not accepted:
            self.dts.__dict__.clear()
            self.dts.__dict__.update(saved)
            self.busy_until = now + self.turnaround_s
            return
        end = max(now + total, at if at is not None else now)
        self.busy_until = end
        self.cas.tx_end = end
        for name, args in events:
            getattr(self.cas, name)(*args)
        self.cas.tick(now)
        self.dts.transmitted(end, end + self.answer_s)
        self.transmissions += 1

    def _windows(self, now, pdus, sizes, total, budget) -> None:
        remaining = list(self.dts._nonarq_out)
        before = 0
        for p, size in zip(pdus, sizes):
            if p.type in (D.NON_ARQ, D.EXP_NON_ARQ):
                key = (p.dest, p.type, p.cpdu_id)
                deadline = self.dts.nonarq_deadlines.get(key)
                if deadline is None:
                    last = [i for i, (_, q, _) in enumerate(remaining)
                            if (q.dest, q.type, q.cpdu_id) == key]
                    duration = total
                    if last:
                        used = 0
                        for _, q, _ in remaining[:last[-1] + 1]:
                            # Future non-ARQ segments may need splitting too.
                            room = self.dts._room
                            for off in range(0, max(len(q.data), 1), room):
                                n = D.size(q) - len(q.data) + len(q.data[off:off + room])
                                if used and used + n > budget:
                                    duration += self.turnaround_s + self.timing.airtime(used)
                                    used = 0
                                used += n
                        if used:
                            duration += self.turnaround_s + self.timing.airtime(used)
                    deadline = now + duration
                    self.dts.nonarq_deadlines[key] = deadline
                p.window = min(65535, max(0, math.ceil(
                    2 * (deadline - now - self.timing.at(before)) - 1e-9)))
            before += size

    # -- the receiver -----------------------------------------------------

    def rx_start(self, now: float) -> None:
        """A transmission is being heard."""
        self.receiving = True

    def rx_bytes(self, now: float, data: bytes) -> None:
        self.cas.now = now
        for p in self.parser.push(data):
            self.heard.append(p)
            if p.eot:
                began = now - 8 * D.size(p) / self.timing.rate
                self.eot_end = max(self.eot_end, began + p.eot / 2)
            self.dts.received(p, now)

    def rx_end(self, now: float, complete: bool = True) -> None:
        """The transmission heard has ended: at its end (``complete``), or
        lost before it."""
        self.receiving = False
        self.parser.flush()
        end = now if complete else max(now, self.eot_end)
        self.free_at = max(self.free_at, end + self.turnaround_s)
        self.eot_end = 0.0
        # Whatever it held, its sender's turn is over: an answer owed us that
        # was not in it is late from now, not from before we heard it.
        grace = now + self.turnaround_s
        for link in self.dts.links.values():
            if link.deadline is not None and link.deadline < grace:
                link.deadline = grace
        for link in self.cas.links.values():
            if link.deadline is not None and link.deadline < grace:
                link.deadline = grace

    # -- modem control ----------------------------------------------------

    def _can_receive(self, rate: int, interleave: str) -> bool:
        timing = self.timing.at_rate(rate, interleave)
        room = timing.most_bytes(self.max_tx_s)
        largest = max((len(f.data) + 40 for link in self.dts.links.values()
                       for f in list(link.arq.frames.values()) + list(link.exp.frames.values())),
                      default=41)
        return rate in D.DRC_RATES and room >= largest

    def _rate_changed(self, remote: int, rate: int, interleave: str) -> None:
        # One modem, one rate: what goes to ``remote`` now goes at it.
        self.timing = self.timing.at_rate(rate, interleave)
        self.on_rate(remote, rate, interleave)

    def on_rate(self, remote: int, rate: int, interleave: str) -> None:
        """Replaced by whoever drives the modem: send at ``rate``."""


class _Deferred:
    def __init__(self, events):
        self.events = events

    def __getattr__(self, name):
        return lambda *args: self.events.append((name, args))
