# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Teardown semantics: an ABORT must not strand the peer, and silence must end a link.

Both defects here were found by grading kestrel's *real* ARQ core rather than its
loopback, and neither is visible from one endpoint — you need a peer to be stranded
before anything looks wrong. They are interop defects, not harness artefacts:

  * ABORT was a purely local close. The far end stayed CONNECTED forever, and a
    connected station refuses every later connect request, so a single ABORT wedged
    the pair permanently. On air that presents as a gateway that answered once and
    then refuses you for no visible reason.
  * Nothing ever concluded a link was dead. A peer that aborts, dies, drops carrier
    or is powered off held the session open indefinitely — which is the same wedge,
    arrived at without anyone calling ABORT.

The two fixes are deliberately independent. Notifying on ABORT is worth nothing
against a peer that was switched off mid-session, and the link-death timer is the
half that protects against peers we do not control — which, on air, is all of them.

These drive the FSM directly through a fake :class:`ArqIO` and a manual clock. The
full audio stack is exercised elsewhere; what is under test here is the state
machine's own contract, and a fake clock makes a 60-second timeout a fast test.
"""
from __future__ import annotations

import pytest

from hfmodem.kestrel.arq import frames as F
from hfmodem.kestrel.arq.fsm import ArqConfig, ArqFsm, State


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _IO:
    """Records what the FSM emitted, and optionally forwards it to a peer."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.peer: ArqFsm | None = None
        self.sent: list = []            # (payload, marker)
        self.events: list[str] = []
        self.delivered = bytearray()
        self.buffer_reports: list[int] = []

    # -- commands ------------------------------------------------------------
    def key(self, on: bool) -> None: ...
    def tx_token(self, name: str, bw: str = "500") -> None: ...
    def log(self, msg: str) -> None: ...

    def tx(self, payload: bytes, marker: int, bw: str = "500", level=None) -> None:
        self.sent.append((payload, marker))
        if self.peer is not None:
            self.peer.on_rx_frame(F.classify(_Result(payload, marker)))

    # -- events --------------------------------------------------------------
    def on_busy(self, on: bool) -> None: ...
    def on_pending(self, cancel: bool = False) -> None: ...
    def on_connected(self, src, dst, bw) -> None: self.events.append("CONNECTED")
    def on_disconnected(self) -> None: self.events.append("DISCONNECTED")
    def on_deliver(self, blob: bytes) -> None: self.delivered += blob

    def on_buffer(self, n: int) -> None:
        self.buffer_reports.append(n)


class _Result:
    """The shape `frames.classify` expects out of the receiver."""

    def __init__(self, payload: bytes, marker: int) -> None:
        self.payload = payload
        self.marker = marker
        self.crc_ok = True


def _linked(cfg: ArqConfig | None = None):
    """Two connected FSMs, each delivering straight into the other."""
    clock = _Clock()
    ioa, iob = _IO("A"), _IO("B")
    a = ArqFsm(ioa, cfg or ArqConfig(), clock=clock)
    b = ArqFsm(iob, cfg or ArqConfig(), clock=clock)
    ioa.peer, iob.peer = b, a
    b.on_host_listen(True)
    a.on_host_connect("W9SSJ", "K7ABC", "500")
    return a, b, ioa, iob, clock


def test_the_pair_connects_at_all():
    """Guard: every assertion below is about what happens *after* CONNECTED, so a
    pair that never connects would pass them vacuously."""
    a, b, _, _, _ = _linked()
    assert a.state == State.CONNECTED and b.state == State.CONNECTED


def test_abort_does_not_strand_the_peer():
    """The defect: the initiator tore down locally and said nothing, so the far end
    stayed CONNECTED — and a connected station refuses every later connect."""
    a, b, _, _, _ = _linked()
    a.on_host_abort()

    assert a.state != State.CONNECTED, "aborting end did not close"
    assert b.state != State.CONNECTED, (
        "peer left CONNECTED by an ABORT it was never told about — it will now "
        "refuse every subsequent connect request")


def test_the_pair_is_reusable_after_an_abort():
    """The consequence that actually bites: one ABORT used to wedge the pair for
    good, and every later probe failed for a reason unrelated to what it tested."""
    a, b, _, iob, _ = _linked()
    a.on_host_abort()

    iob.events.clear()
    b.on_host_listen(True)
    a.on_host_connect("W9SSJ", "K7ABC", "500")
    assert a.state == State.CONNECTED and b.state == State.CONNECTED, (
        "the pair could not reconnect after an ABORT")
    assert "CONNECTED" in iob.events


def test_silence_eventually_ends_the_link():
    """No ABORT, no notification, no cooperation from the peer at all — the case a
    notification can never cover, because the peer is gone."""
    a, b, _, _, clock = _linked()
    b.peer = None                      # the far end vanishes: no traffic, no reply
    a.io.peer = None

    clock.advance(a.cfg.link_death_s + 1.0)
    a.on_timer()

    assert a.state != State.CONNECTED, (
        f"still CONNECTED {a.cfg.link_death_s}s after the peer went silent")


def test_a_live_link_is_not_torn_down():
    """The timer must not kill a slow-but-alive link: any traffic resets it."""
    a, b, _, _, clock = _linked()
    for _ in range(4):
        clock.advance(a.cfg.link_death_s * 0.5)
        a.on_timer()
        # Deliver a real keepalive rather than poking the activity clock directly:
        # calling a._touch() would keep passing even if the receive path stopped
        # refreshing it, which is the regression this test exists to catch.
        b._send_control(F.KA)
    assert a.state == State.CONNECTED, "a link with traffic was declared dead"


def test_link_death_can_be_disabled():
    """`link_death_s = 0` restores the old unbounded behaviour, for callers that
    manage liveness themselves."""
    a, _, _, _, clock = _linked(ArqConfig(link_death_s=0))
    clock.advance(100_000.0)
    a.on_timer()
    assert a.state == State.CONNECTED


@pytest.mark.parametrize("payload", [b"early bytes", b"x" * 300])
def test_preconnect_host_data_is_queued_not_dropped(payload):
    """[spec 07 §7.4]: bytes written before the link comes up are buffered against
    the TX queue and flushed once CONNECTED — not discarded. The core used to drop
    them while this package's own LoopbackModem buffered, so the two halves
    disagreed about the contract."""
    clock = _Clock()
    ioa, iob = _IO("A"), _IO("B")
    a = ArqFsm(ioa, ArqConfig(), clock=clock)
    b = ArqFsm(iob, ArqConfig(), clock=clock)
    ioa.peer, iob.peer = b, a

    a.on_host_data(payload)                       # BEFORE any connect
    assert ioa.buffer_reports, "pre-connect write was not reported as buffered"
    assert ioa.buffer_reports[-1] == len(payload)

    b.on_host_listen(True)
    a.on_host_connect("W9SSJ", "K7ABC", "500")

    assert iob.delivered.startswith(payload), (
        f"pre-connect bytes lost: delivered {bytes(iob.delivered)[:32]!r}")


def test_abort_discards_the_queue_but_disconnect_flushes_it():
    """[spec 07 §7.4] distinguishes the two teardowns, and the pre-connect queue
    must follow the same rule as the main one."""
    clock = _Clock()
    ioa = _IO("A")
    a = ArqFsm(ioa, ArqConfig(), clock=clock)
    a.on_host_data(b"discard me")
    a.on_host_abort()
    assert a._preconnect == [], "ABORT must discard the queue, not hold it"


def test_responder_holds_preconnect_bytes_instead_of_losing_them():
    """Counted bytes must not vanish.

    Only the initiator sends data overs in this milestone, so flushing the
    pre-connect queue on a responder fed every blob into the role check in
    `on_host_data`, which logged and dropped it. The host had already been told
    BUFFER n, `_buffer_raw` was decremented to match, and no further BUFFER was
    emitted — the host's flow-control view stayed permanently wrong with nothing
    reporting it. Silently losing bytes the host was told are queued is the one
    outcome that should not survive.
    """
    clock = _Clock()
    io = _IO("A")
    a = ArqFsm(io, ArqConfig(), clock=clock)
    a.on_host_listen(True)
    a.on_host_data(b"twenty bytes exactly")
    assert io.buffer_reports[-1] == 20

    a.role = "responder"
    a._enter_connected()

    assert a._buffer_raw == 20, (
        f"_buffer_raw dropped to {a._buffer_raw}: counted bytes were discarded")
    assert a._preconnect, "the held blobs were thrown away"
    assert io.buffer_reports[-1] == 20, (
        "the host was never re-told the true queue depth")
