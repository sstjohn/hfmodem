# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Robustness regressions for besra's ARQ engine — the failure edges besra↔besra
loopback never exercises: keepalive under a live idle link, lost-frame retries for
both DATA and the connect handshake, queue survival across a BREAK turnover, host
verbs racing modem lifecycle, a malformed ID frame, and stale-ACK dedup.

Frame-level (`Pair`) tests drive two `ArqSession`s over a relay that can drop or
corrupt selected frames and a hand-cranked clock; the modem-level tests poke
`BesraModem` directly.
"""

from __future__ import annotations

import ast
import pathlib
import time
from collections import deque

import numpy as np
import pytest

from hfmodem.besra import crc
from hfmodem.besra.arq import session as S
from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.arq.session import (
    CONREQ_MAX, DISC, IDLE, _UNDELIVERED_BUDGET, _UNREPAIRED_BUDGET, ArqSession,
    _datanak_for, _is_data, _is_dataack, _is_datanak, _type_of,
)
from hfmodem.besra.host import protocol as P
from hfmodem.besra.host.modem_core import ModemObserver
from hfmodem.besra.phy.demodulator import DecodedFrame


# -- frame-level harness -----------------------------------------------------

class Recorder:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.connects: list[tuple[str, int]] = []
        self.disconnects = 0
        self.received = bytearray()
        self.buffers: list[int] = []
        self.statuses: list[str] = []

    def newstate(self, s): self.states.append(s)
    def connected(self, r, bw): self.connects.append((r, bw))
    def disconnected(self): self.disconnects += 1
    def data_received(self, k, b): self.received += b
    def buffer(self, n): self.buffers.append(n)
    def pending(self, cancel=False): pass
    def target(self, call): pass
    def status(self, t): self.statuses.append(t)


class Relay:
    """A half-duplex ether that logs every send and can drop or corrupt frames by
    a caller-supplied predicate ``fn(seq, sender, frame_type) -> bool``.

    ``guess`` is the third thing a channel does to a frame and the one the other
    two cannot express: ``fn(seq, sender, frame_type) -> int | None`` returns the
    type the receiver read off a header when nothing else validated, so the frame
    is delivered under that type, ``ok=False`` and flagged as the guess it is.

    ``mint`` is the fourth, and it is a receiver defect rather than a channel one:
    ``fn(seq, sender, frame_type) -> int | None`` returns a bare control type this
    receiver minted out of the frame's leader, which the half-duplex mute splice
    does when it lands there. The emission is then reported twice, as the recorded
    2026-08-23 splice reported it — once as that control, whole and ``ok=True``
    because a bodyless frame has no body to fail, and once as a header-only guess
    at its own type."""

    def __init__(self) -> None:
        self.q: deque = deque()
        self.log: list[tuple[str, int]] = []
        self.seq = 0
        self.drop = None
        self.corrupt = None
        self.guess = None
        self.mint = None

    def submit(self, sender, dest, ft, payload, sid) -> None:
        self.seq += 1
        self.log.append((sender, ft))
        if self.drop and self.drop(self.seq, sender, ft):
            return
        minted = self.mint(self.seq, sender, ft) if self.mint else None
        if minted is not None:
            self.q.append((dest, minted, b"", sid, True, False))
            self.q.append((dest, ft, b"", sid, False, True))
            return
        read = self.guess(self.seq, sender, ft) if self.guess else None
        if read is not None:
            self.q.append((dest, read, b"", sid, False, True))
            return
        ok = not (self.corrupt and self.corrupt(self.seq, sender, ft))
        self.q.append((dest, ft, payload, sid, ok, False))

    def pump(self, limit: int = 4000) -> None:
        steps = 0
        while self.q and steps < limit:
            dest, ft, payload, sid, ok, guessed = self.q.popleft()
            dest.on_receive(ft, payload, sid, ok, header_only=guessed)
            steps += 1
        assert steps < limit, "relay did not settle — probable frame loop"


class Wire:
    def __init__(self, relay: Relay, call: str) -> None:
        self._relay = relay
        self._call = call
        self.dest: ArqSession | None = None

    def send(self, ft: int, payload: bytes, sid: int) -> float:
        self._relay.submit(self._call, self.dest, ft, bytes(payload), sid)
        return 0.0                     # synchronous relay — no real transmit time


class Pair:
    def __init__(self, *, bandwidth: int = 500, timeout_s: float = 90.0) -> None:
        self.relay = Relay()
        self.obs_a = Recorder()
        self.obs_b = Recorder()
        wa = Wire(self.relay, "W9SSJ")
        wb = Wire(self.relay, "K7ABC")
        self.caller = ArqSession("W9SSJ", wa, self.obs_a,
                                 bandwidth=bandwidth, timeout_s=timeout_s)
        self.responder = ArqSession("K7ABC", wb, self.obs_b,
                                    bandwidth=bandwidth, timeout_s=timeout_s,
                                    listen=True)
        wa.dest = self.responder
        wb.dest = self.caller
        self.now = 0.0

    def tick(self, dt: float = 0.0) -> None:
        self.now += dt
        self.caller.tick(self.now)
        self.responder.tick(self.now)

    def pump(self) -> None:
        self.relay.pump()

    def connect(self) -> None:
        self.caller.connect("K7ABC")
        self.pump()

    def run(self, seconds: float, dt: float = 0.5) -> None:
        for _ in range(int(round(seconds / dt))):
            self.tick(dt)
            self.pump()


# -- fix 1: the session deadline counts progress, not our own answers --------

def test_an_exchange_that_is_only_keepalives_ends_on_its_own_deadline():
    """25 cycles of `RX IDLE` drew 25 `TX DATAACK` over 71 s after a gateway had
    refused our login, and the link ended on the operator's SIGTERM.

    Nothing was misbehaving. An ISS holding the link with nothing to send chirps;
    an IRS answers, because an unanswered chirp is a lost frame. What was wrong is
    that answering refreshed the liveness clock — so the only thing that could
    have ended the exchange was also the thing postponing it, and the deadline was
    unreachable by construction. The peer is alive the whole time here, and the
    exchange is over regardless: an ARQ layer cannot be told the difference by
    anything except whether it is still getting somewhere.
    """
    pair = Pair(timeout_s=20.0)
    pair.connect()
    assert pair.caller.connected and pair.responder.connected

    pair.run(40.0)  # twice the deadline, and every chirp answered

    assert not pair.caller.connected, "an exchange of keepalives held the link"
    assert any("Timeout" in s for s in pair.obs_a.statuses + pair.obs_b.statuses)


def test_a_link_that_is_getting_somewhere_outlives_the_deadline():
    """The other half, and the one that costs a session if it is got wrong: the
    deadline may not fire on a link that is working. Payload moving is progress,
    and progress is what holds it — over four times the deadline here."""
    pair = Pair(timeout_s=20.0)
    pair.connect()

    for i in range(6):
        pair.caller.queue_data(b"block %d" % i)
        pair.run(14.0)
        assert pair.caller.connected, f"dropped while block {i} was moving"

    assert bytes(pair.obs_b.received) == b"".join(b"block %d" % i for i in range(6))


# -- fix 2: an outstanding DATA frame is retried -----------------------------

def test_lost_data_frame_retransmits_and_delivers():
    pair = Pair()
    pair.connect()

    dropped: list[int] = []

    def drop(seq, sender, ft):
        if _is_data(ft) and not dropped:
            dropped.append(seq)
            return True
        return False

    pair.relay.drop = drop
    pair.caller.queue_data(b"resend please")
    pair.pump()
    assert bytes(pair.obs_b.received) == b"", "first DATA was dropped"

    pair.run(6.0)  # the ~2 s repeat timer resends it

    assert dropped, "a DATA frame must have been dropped"
    assert bytes(pair.obs_b.received) == b"resend please"
    assert pair.obs_a.buffers[-1] == 0


def test_lost_dataack_retransmits_and_delivers_once():
    pair = Pair()
    pair.connect()

    dropped: list[int] = []

    def drop(seq, sender, ft):
        if sender == "K7ABC" and _is_dataack(ft) and not dropped:
            dropped.append(seq)
            return True
        return False

    pair.relay.drop = drop
    pair.caller.queue_data(b"ack me")
    pair.pump()  # data delivered to IRS, but its ACK is dropped
    assert bytes(pair.obs_b.received) == b"ack me"
    assert pair.obs_a.buffers[-1] != 0, "ISS must still hold the unacked frame"

    pair.run(6.0)

    assert dropped, "a DATAACK must have been dropped"
    assert bytes(pair.obs_b.received) == b"ack me", "no duplicate delivery"
    assert pair.obs_a.buffers[-1] == 0


# -- fix 3: the confirming ConAck leg is retried -----------------------------

def test_lost_confirming_conack_still_connects():
    pair = Pair()
    pair.relay.drop = lambda seq, sender, ft: seq == 3  # caller's confirming ConAck

    pair.caller.connect("K7ABC")
    pair.pump()
    assert not pair.obs_a.connects and not pair.obs_b.connects, "handshake wedged"

    pair.run(8.0)

    assert pair.obs_a.connects == [("K7ABC", 500)], f"caller stuck: {pair.caller.state}"
    assert pair.obs_b.connects == [("W9SSJ", 500)], f"responder stuck: {pair.responder.state}"


# -- fix 4: a deposed ISS keeps its queued data ------------------------------

def test_deposed_iss_replays_queued_data_after_regaining_iss():
    pair = Pair()
    pair.connect()

    # Caller has data outstanding when the responder seizes the link with a BREAK.
    pair.caller.queue_data(b"CALLER-DATA")
    pair.responder.queue_data(b"RESP-DATA")
    pair.pump()

    assert pair.caller.state == P.ArdopState.IRS, "caller conceded to the BREAK"
    assert bytes(pair.obs_a.received) == b"RESP-DATA"
    assert pair.caller._outbound, "conceded ISS must keep its unsent queue"

    # Caller reclaims the link; the preserved bytes plus new ones both go out.
    pair.caller.queue_data(b"-AND-MORE")
    pair.pump()

    assert bytes(pair.obs_b.received) == b"CALLER-DATA-AND-MORE"


# -- fix 8: a stale duplicate ACK does not drop an unacked frame -------------

def test_stale_duplicate_ack_does_not_drop_unacked_frame():
    pair = Pair()
    pair.connect()

    # Drop the first DATAACK so the ISS must repeat its frame; the IRS then
    # re-ACKs. That advance arms the half-duplex guard, since the frame was
    # repeated and a delayed duplicate ACK is now possible.
    dropped: list[int] = []

    def drop(seq, sender, ft):
        if sender == "K7ABC" and _is_dataack(ft) and not dropped:
            dropped.append(seq)
            return True
        return False

    pair.relay.drop = drop
    payload = bytes(range(40))          # 40 > 32 → two 500-Hz frames
    pair.caller.queue_data(payload)
    pair.pump()                         # frame 1 delivered; its ACK dropped

    pair.tick(2.5)                      # one repeat interval
    pair.pump()                         # resend → re-ACK → advance to frame 2

    assert dropped, "a DATAACK must have been dropped"
    assert pair.caller._last_data_type >= 0, "frame 2 is now outstanding"

    # A stale duplicate ACK for frame 1 lands right after the advance. Without the
    # guard it would delete frame 2's still-unacked bytes.
    held = bytes(pair.caller._outbound)
    pair.caller.on_receive(0xF0, b"", pair.caller._session, True)
    assert bytes(pair.caller._outbound) == held, "stale ACK dropped an unacked frame"

    pair.run(8.0)

    assert bytes(pair.obs_b.received) == payload, "all bytes delivered"
    assert pair.obs_a.buffers[-1] == 0


# -- fix 9: the IRS dedup does not outlive the stint it was measured in ------

def test_a_header_only_drop_after_a_turnaround_is_not_acked():
    """WW2MI, 2026-08-18, 21:45:27: `RX 4PSK.200.100.E sess=0xf3 ok=False
    HEADER-ONLY q=64` answered with `TX DATAACK`. That frame carried the first
    64 bytes of the gateway's forward block; the next one carried the second
    64, and the message came out starting at `0xb1` — `expected SOH, got byte
    0xb1`, six seconds later.

    That frame was even, and so was the last one this end ACKed before the
    turnover — 21:45:11, q=86. What a station sends first on taking the link
    is its own business, so graded against a `_last_acked_type` held over from
    the previous stint, an undecodable frame is ACKed as already delivered and
    the ISS moves on with the payload.
    """
    pair = Pair()
    pair.connect()

    theirs = b"one whole frame."
    forward = b"the first frame."
    assert len(theirs) == len(forward) == 16

    pair.responder.queue_data(theirs)
    pair.pump()
    assert bytes(pair.obs_a.received) == theirs
    acked = pair.caller._last_acked_type

    pair.caller.queue_data(b"our answer")
    pair.run(6.0)
    assert not pair.caller._outbound, "our answer went out and the link fell idle"
    assert pair.caller.state == P.ArdopState.IDLE

    corrupted: list[int] = []

    def corrupt(seq, sender, ft):
        if sender == "K7ABC" and _is_data(ft) and not corrupted:
            corrupted.append(ft)
            return True
        return False

    pair.relay.corrupt = corrupt
    pair.responder.queue_data(forward)
    pair.pump()

    assert corrupted == [acked], "the peer's first frame back repeats the ACKed type"
    assert any(_is_datanak(ft) for who, ft in pair.relay.log if who == "W9SSJ"), \
        "an undelivered frame was ACKed as a repeat"

    pair.run(6.0)

    assert bytes(pair.obs_a.received) == theirs + forward


def test_a_guessed_type_may_ask_for_a_repeat_but_may_never_acknowledge_one():
    """WW2MI, 2026-08-19 23:01:13: `RX 4PSK.200.100.O sess=0xf3 ok=False
    HEADER-ONLY q=64` answered with `TX DATAACK`, inside one stint as IRS and with
    no turnover to blame.

    A header-only frame's type is read off ten tones nothing corroborated, and the
    repeat rule decides on that type alone: guessed as the one last ACKed, it tells
    the peer we already hold a frame we never read and the peer advances past it.
    Replayed over the ARDOP captures in `logs/onair`, 26 of 161 header-only data
    frames read as the type just ACKed, and six drew a DATAACK on the air.

    A repeat rule is still right for a frame that was read — that is what tells a
    peer whose ACK was lost to move on. What a guess may authorise is the narrower
    thing: a retransmit, never a delivery.
    """
    pair = Pair()
    pair.connect()

    first, second = b"the forward block, frame one....", b"the forward block, frame two..."
    pair.responder.queue_data(first)
    pair.pump()
    assert bytes(pair.obs_a.received) == first
    acked = pair.caller._last_acked_type

    guessed: list[int] = []

    def guess(seq, sender, ft):
        if sender == "K7ABC" and _is_data(ft) and not guessed:
            guessed.append(ft)
            return acked
        return None

    pair.relay.guess = guess
    pair.responder.queue_data(second)
    pair.pump()

    assert guessed == [acked ^ 1], "the peer's next frame alternates off the ACKed type"
    assert any(_is_datanak(ft) for who, ft in pair.relay.log if who == "W9SSJ"), \
        "a type nothing corroborated was answered as a frame already delivered"

    pair.run(6.0)
    assert bytes(pair.obs_a.received) == first + second


def test_a_guess_that_reads_as_a_control_frame_still_asks_for_the_frame_again():
    """The same guess, landing on a type that is not data at all.

    Every control branch in `_rx_irs` needs `ok`, which a guess never has, so a
    frame that arrived and would not read fell through the whole handler and drew
    neither an acknowledgement nor a repeat request — the silence the operator
    heard on the monitor. 2 of the 163 header-only frames in the ARDOP corpus read
    as a control type; the other 161 read as data and were answered.
    """
    pair = Pair()
    pair.connect()
    pair.responder.queue_data(b"the forward block, frame one....")
    pair.pump()
    marks = len(pair.relay.log)

    once: list[int] = []

    def guess(seq, sender, ft):
        if sender == "K7ABC" and _is_data(ft) and not once:
            once.append(ft)
            return IDLE
        return None

    pair.relay.guess = guess
    pair.responder.queue_data(b"the forward block, frame two...")
    pair.pump()

    assert once, "the peer never sent the frame the guess was made of"
    assert any(_is_datanak(ft) for who, ft in pair.relay.log[marks:] if who == "W9SSJ"), \
        "a frame that arrived and would not read was answered with nothing at all"


# -- fix 9b: and only a frame addressed to this session may draw one ---------

def test_a_guess_off_a_strangers_conreq_draws_nothing():
    """Three third-party ConReqs, read as headers and nothing else, drew
    `DATANAK DATANAK DISC` — stamped with this session's id and reported against a
    station that had sent none of them.

    ConReq/Ping/ID clear the address filter on the forced 0xFF wire id and are
    addressed by callsign in their own handlers. A guessed one has no payload to
    read a callsign from, so it is not evidence about anybody: not that our peer
    is unreadable, and not that this link is worth ending.
    """
    pair = Pair()
    pair.connect()
    pair.run(2.0)
    irs = pair.responder
    assert irs.state == P.ArdopState.IRS

    keyed = len(pair.relay.log)
    for _ in range(_UNREPAIRED_BUDGET + 1):
        irs.on_receive(CONREQ_MAX[500], b"", 0xFF, False, quality=19, header_only=True)

    assert pair.relay.log[keyed:] == [], "a stranger's guessed ConReq was answered"
    assert irs.connected, "a stranger's guessed ConReq ended the link"


def test_a_guessed_frame_is_not_a_grade_of_the_path():
    """The NAK a guess draws carries the reading measured on it, and the session's
    standing grade — what the next ACK reports — is left to the frames that read.
    """
    pair = Pair()
    pair.connect()
    pair.responder.queue_data(b"the forward block, frame one....")
    pair.pump()
    irs, graded = pair.caller, pair.caller._rx_quality

    keyed = len(pair.relay.log)
    irs.on_receive(irs._last_acked_type ^ 1, b"", irs._session, False,
                   quality=19, header_only=True)

    assert [ft for who, ft in pair.relay.log[keyed:]] == [_datanak_for(19)], \
        "the guess's own reading is what its NAK asks the peer to shift on"
    assert irs._rx_quality == graded, "a frame that never read graded the path"


# -- fix 10: and a frame replayed across one is not delivered twice ----------

def test_identical_frames_after_a_replay_follow_type_alternation():
    pair = Pair()
    pair.connect()
    payload = b"\x00\x1c\x1e\xffsame"
    pair.caller.queue_data(payload)
    pair.pump()
    irs = pair.responder
    ft = irs._last_acked_type
    irs._reset_stint()

    for frame_type in (ft, ft, ft ^ 1, ft ^ 1, ft):
        irs.on_receive(frame_type, payload, irs._session, True)

    assert bytes(pair.obs_b.received) == payload * 3


@pytest.mark.parametrize("idle_count", [0, 1])
def test_unconfirmed_idle_preserves_replay_suppression(idle_count):
    pair = Pair()
    pair.connect()
    payload = b"pending"
    pair.caller.queue_data(payload)
    pair.pump()
    irs = pair.responder
    ft = irs._last_acked_type
    for _ in range(idle_count):
        irs.on_receive(IDLE, b"", irs._session, True)
    irs._reset_stint()
    irs.on_receive(ft, payload, irs._session, True)
    assert bytes(pair.obs_b.received) == payload


def test_identical_deliveries_survive_an_idle_turnover():
    pair = Pair()
    pair.connect()
    payload = b"\x00\x1c\x1e\xffsame"
    pair.caller.queue_data(payload)
    pair.run(10.0)
    assert bytes(pair.obs_b.received) == payload
    assert sum(who == "W9SSJ" and ft == IDLE
               for who, ft in pair.relay.log) >= 2

    pair.responder.queue_data(b"reply")
    pair.run(10.0)
    assert bytes(pair.obs_a.received) == b"reply"
    pair.caller.queue_data(payload)
    pair.run(10.0)

    assert bytes(pair.obs_b.received) == payload * 2


def test_a_frame_replayed_across_a_reversal_is_not_appended_twice():
    """The other side of fix 9, and the same shortage of identity. Clearing the
    alternation at a reversal is what stops a NEW frame being taken for a repeat;
    it also stops a REPEAT being taken for one, and a deposed ISS unwinds the
    frame it never heard an ACK for and puts it back on the air the moment it
    regains the link (ardopcf SaveQueueOnBreak).

    An application that answers from inside `data_received` — which the mail
    client does — is what makes the two meet: the break-in it queues overtakes
    this end's DATAACK, so the frame is delivered here and unacknowledged there
    at the same instant. The bytes are then the only identity left, and they are
    the same bytes, so the replay is acknowledged again and delivered once.

    What it costs when it is missed is not a corrupt message: fed a doubled
    frame at every frame boundary of the recovered WW2MI turn, and a doubled STX
    block besides, `B2FSession` refuses all of them — on its own framing, or on
    the EOT checksum, or on the proposed compressed size. It costs the session.
    """
    theirs, ours = b"the gateway's forward block, frame one", b"FS Y\r"
    pair = Pair()

    class _Answering(Recorder):
        def data_received(self, k, b):
            super().data_received(k, b)
            pair.caller.queue_data(ours)

    pair.obs_a.__class__ = _Answering
    pair.connect()
    pair.run(3.0)
    assert pair.caller.connected and pair.responder.connected

    lost: list[int] = []

    def drop(seq, sender, ft):
        if sender == "W9SSJ" and _is_dataack(ft) and len(lost) < 3:
            lost.append(seq)
            return True
        return False

    pair.relay.drop = drop
    pair.responder.queue_data(theirs)
    pair.run(12.0)
    assert len(lost) == 3, "the ACKs this end owed were not the ones dropped"

    pair.relay.drop = None
    pair.run(40.0)

    assert bytes(pair.obs_a.received) == theirs
    assert bytes(pair.obs_b.received) == ours, "the answer was appended twice"
# -- fix 11: an unreadable frame is answered, and the answering is bounded ---

def test_frames_that_never_decode_draw_naks_and_then_a_deliberate_close():
    """A frame that arrived and would not read is neither silence nor a repeat.

    The NAK is right and ARQ does its job. What it asks a peer for is a shift
    DOWN, not a resend — the reference gearshifts on it and reaches `SendData()`
    only when the shift is non-zero (`ARQ.c:2210-2212`), and the frame comes
    round again on the ISS's own repeat timer either way. So the repair a NAK
    buys is a more robust mode over the frame we were handed.

    This test used to close on the third such frame, on the argument that a
    repeat which fails the same way is this receiver's fault and no amount of the
    peer transmitting can mend it. That premise does not survive 2026-08-29.
    W6IDS's greeting failed three frames running on two of four connects and the
    peer was demonstrably still there — heard from the speaker, and read by the
    channel sense as `+4.8 dB tone at 1512 Hz -> OCCUPIED` a minute after we had
    hung up — while on the connect that worked the copy after a failure decoded,
    on a path whose quality was climbing. A run of failures at a peer this end has
    never read is a fade to sit out, not a verdict on the receiver.

    So what is pinned here is what has not changed: the train is finite, it ends
    on a count rather than by waiting for the clock, and the peer is told. Only
    the count moved, and to `_UNDELIVERED_BUDGET` because nothing has been
    delivered over this link at all.
    """
    pair = Pair()
    pair.connect()

    pair.relay.corrupt = lambda seq, sender, ft: sender == "K7ABC" and _is_data(ft)
    pair.responder.queue_data(b"a block that never arrives intact.")
    pair.run(20.0)

    naks = [ft for who, ft in pair.relay.log if who == "W9SSJ" and _is_datanak(ft)]
    assert len(naks) == _UNDELIVERED_BUDGET, \
        f"{len(naks)} repeat requests, expected the budget of {_UNDELIVERED_BUDGET}"
    assert not pair.caller.connected
    assert not [t for t in pair.obs_a.statuses if "Timeout" in t], \
        "the count stopped bounding it and the session deadline did the work"
    assert any("UNREADABLE" in s for s in pair.obs_a.statuses), pair.obs_a.statuses
    assert any(ft == DISC for who, ft in pair.relay.log if who == "W9SSJ"), \
        "the link was dropped without telling the peer"


def test_a_deduped_replay_does_not_license_the_standing_nak_train():
    """What licenses the standing NAK train is payload reaching the application,
    and a replay across a reversal is the one frame that reads and delivers none.

    `_last_rx_type` records it all the same, because the frames after it have to
    follow type alternation to be told apart. Read as delivery, that record hands
    a stint which has passed the application nothing the train
    `_UNREPAIRED_BUDGET` reserves for one that has — and takes with it the
    `_UNDELIVERED_BUDGET` close, the only thing that ends an ask at a peer this
    end has never read, spending a gateway's transmitter as well as ours."""
    pair = Pair()
    pair.connect()
    iss, irs = pair.responder, pair.caller
    block = b"unsettled"          # one frame at the base rung
    # No clock between the delivery and the turnover: a peer that chirps IDLE
    # first has settled the bytes, and a later copy of them is a new delivery.
    iss.queue_data(block)
    pair.pump()
    assert bytes(pair.obs_a.received) == block
    ft = irs._last_rx_type

    irs.queue_data(b"and this end answers")
    pair.pump()
    irs.on_receive(S.BREAK, b"", irs._session, True)

    irs.on_receive(ft, block, irs._session, True)
    assert bytes(pair.obs_a.received) == block, "the replay was appended twice"

    for _ in range(_UNDELIVERED_BUDGET + 1):
        irs.on_receive(ft ^ 1, b"", irs._session, False)
    naks = [t for who, t in pair.relay.log if who == "W9SSJ" and _is_datanak(t)]
    assert len(naks) == _UNDELIVERED_BUDGET, \
        f"{len(naks)} repeat requests, expected the budget of {_UNDELIVERED_BUDGET}"
    assert irs._disc_repeating, \
        "a stint that delivered nothing went on asking past its budget"
    assert any(t == DISC for who, t in pair.relay.log if who == "W9SSJ"), \
        "the link was dropped without telling the peer"


def test_a_frame_that_reads_clears_the_repeat_budget():
    """The budget counts CONSECUTIVE unreadable frames: a link that loses one
    frame an hour to a fade is a working link, and must not accumulate its way to
    a disconnect."""
    pair = Pair()
    pair.connect()

    nth = [0]

    def corrupt(seq, sender, ft):
        if sender != "K7ABC" or not _is_data(ft):
            return False
        nth[0] += 1
        return nth[0] % 2 == 1              # every other frame is unreadable

    pair.relay.corrupt = corrupt
    for i in range(4):
        pair.responder.queue_data(b"block %d" % i)
        pair.run(10.0)
        assert pair.caller.connected, f"dropped on block {i} with repairs in between"


# -- modem-level fixes -------------------------------------------------------

class _Obs(ModemObserver):
    def __init__(self) -> None:
        self.faults: list[str] = []

    def modem_newstate(self, state): pass
    def modem_connected(self, remote, bw): pass
    def modem_disconnected(self): pass
    def modem_ptt(self, on): pass
    def modem_buffer(self, n): pass
    def modem_data_received(self, kind, blob): pass
    def modem_fault(self, text): self.faults.append(text)


# -- fix 5: host verbs guard a null session ----------------------------------

def test_verbs_safe_before_start_and_after_stop():
    m = BesraModem(bandwidth=500)
    m.set_mycall("W9SSJ")
    m.connect("K7ABC")           # pre-start: _session is None
    m.transmit(b"x")
    m.disconnect()
    m.abort()

    m.start(_Obs())
    m.stop()                     # nulls _session
    m.connect("K7ABC")           # post-stop
    m.transmit(b"x")
    m.disconnect()
    m.abort()


# -- fix 6: a malformed ID frame must not raise / kill the pump --------------

def test_empty_grid_id_frame_payload_is_tolerated():
    f = DecodedFrame(type=0x30, session_id=0xFF, ok=True, name="IDFrame",
                     caller="W9SSJ", grid=None)
    payload = BesraModem._frame_payload(f)  # must not raise
    assert len(payload) == 12


def test_pump_survives_a_decode_exception():
    m = BesraModem(threaded=True)
    m.set_mycall("W9SSJ")
    obs = _Obs()
    m.start(obs)
    try:
        calls = {"n": 0}

        def boom(samples):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("bad burst")
            return []

        m._demod.decode = boom
        m.deliver_rx(np.zeros(100, dtype="<i2"))
        _spin(lambda: calls["n"] >= 1)
        assert m._running and m._pump.is_alive(), "pump died on a decode error"
        assert obs.faults, "the fault was reported to the host"

        m.deliver_rx(np.zeros(100, dtype="<i2"))
        _spin(lambda: calls["n"] >= 2)
        assert m._running and m._pump.is_alive(), "pump kept decoding after the fault"
    finally:
        m.stop()


def _spin(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)


# -- fix 7: the session ID is canonical (interop with real ardopcf) ----------

class _NullTx:
    def send(self, ft, payload, sid): return 0.0


def _session_for(mycall: str, target: str) -> int:
    s = ArqSession(mycall, _NullTx(), Recorder())
    s.connect(target)
    return s._session


def test_session_id_is_stable_across_callsign_spellings():
    canon = crc.session_id("W9SSJ", "K7ABC")
    for mycall in ("w9ssj", "W9SSJ", "W9SSJ-0"):
        assert _session_for(mycall, "K7ABC") == canon
    for target in ("k7abc", "K7ABC", "K7ABC-0"):
        assert _session_for("W9SSJ", target) == canon


# -- fix 11: and the rule those two fixes were reaching for, written once ----

@pytest.mark.parametrize("control", ["BREAK", "IDLE", "ConAck500"])
def test_a_data_frame_minted_into_a_control_frame_is_never_acknowledged(control):
    """`WWTD6QMC61TV`, 2026-08-23: `expected SOH, got byte 0xfd` — the 65th byte of
    the message, because the 64 in front of it were acknowledged and never
    delivered.

    A 4.4 s `4PSK.200.100` from the CMS was reported twice, once as a bodyless
    BREAK the splice minted from its leader and once as a header-only guess at its
    real type. `_rx_irs` answered the BREAK with `_concede_again`'s DATAACK, the
    ISS read that as the acknowledgement of the frame it was sending, and the head
    of the message went with it. There is no BREAK on the air anywhere in that
    span: a whole-capture pass, a stepped-window pass and a `RollingDecoder`
    replay all agree, and the only bodyless controls there are our own.

    Parametrised because the two recorded losses arrived by two routes and a fix
    aimed at one route is not the rule. While the peer holds the link, no bare
    control it appears to have sent may draw an acknowledgement off a single
    report — the frame it was minted from is what that acknowledgement would
    clear.
    """
    forward = b"SOH, and the head of the first STX block, 64 bytes of it......."
    pair = Pair()
    pair.connect()

    marks: list[int] = []

    def mint(seq, sender, ft):
        if sender == "K7ABC" and _is_data(ft) and not marks:
            marks.append(len(pair.relay.log) - 1)
            return _type_of(control)
        return None

    pair.relay.mint = mint
    pair.responder.queue_data(forward)
    pair.pump()

    assert marks, "the peer's data frame was never minted into a control frame"
    answer = next(ft for who, ft in pair.relay.log[marks[0] + 1:] if who == "W9SSJ")
    assert _is_datanak(answer), "a frame the application never saw was acknowledged"

    pair.run(6.0)
    assert bytes(pair.obs_a.received) == forward


def test_a_data_frame_minted_into_a_disc_does_not_tear_the_link_down():
    """The same receiver defect wearing the frame type that costs the most.

    A minted BREAK spends one frame; a minted DISC ends the session and takes the
    whole queue with it, and it drew no DATAACK so it sat outside the invariant
    the BREAK case was scoped to. It is the same single uncorroborated report of a
    bare control while the peer holds the link, so it waits for the same repeat —
    which a real DISC always sends and a minted one never does.
    """
    forward = b"the CMS's first forward frame, and the whole queue behind it"
    pair = Pair()
    pair.connect()

    marks: list[int] = []

    def mint(seq, sender, ft):
        if sender == "K7ABC" and _is_data(ft) and not marks:
            marks.append(len(pair.relay.log) - 1)
            return DISC
        return None

    pair.relay.mint = mint
    pair.responder.queue_data(forward)
    pair.pump()

    assert marks, "the peer's data frame was never minted into a DISC"
    assert not pair.obs_a.disconnects, "a frame nobody sent ended the session"
    pair.run(6.0)
    assert bytes(pair.obs_a.received) == forward


def test_a_disc_that_repeats_still_ends_the_session():
    """The corroboration rule costs a real disconnect one repeat interval and no
    more. The reference repeats DISC until an END comes back or the count runs out
    (`ARQ.c` `blnDISCRepeating`), which is what makes the second copy something to
    wait for rather than something to hope for."""
    pair = Pair()
    pair.connect()
    pair.responder.queue_data(b"a frame, so this end is IRS with the peer sending")
    pair.pump()

    for _ in range(2):
        pair.caller.on_receive(DISC, b"", pair.caller._session, True)
        pair.pump()

    assert pair.obs_a.disconnects, "a repeated DISC left the link up"
    assert any(ft == S.END for who, ft in pair.relay.log if who == "W9SSJ"), \
        "the peer's DISC was never answered with an END"


def test_every_acknowledgement_on_the_air_leaves_through_one_gate():
    """The two losses were one defect reached by two routes, and what let the
    second one happen is that the first was mended where it was found rather than
    stated where it belonged. There are eight sites in this module that can key a
    DATAACK; the invariant they all have to satisfy is written at one of them."""
    tree = ast.parse(pathlib.Path(S.__file__).read_text())
    keyed = {fn.name for fn in ast.walk(tree)
             if isinstance(fn, ast.FunctionDef)
             for call in ast.walk(fn)
             if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
             and call.func.id == "_dataack_for"}

    assert keyed == {"_dataack"}, "a DATAACK is keyed outside the gate that states the rule"
    assert "no sequence number" in S.ArqSession._dataack.__doc__
