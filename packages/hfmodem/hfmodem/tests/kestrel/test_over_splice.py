# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What a station hears while it is transmitting, and what it does with the rest.

On 2026-08-20 this station connected to KC9GHZ, read the first half of the
gateway's Winlink greeting, answered it, and 3.7 s into the second half keyed a
keepalive on top of it. The log's last protocol line is `tx NAK (1/2)`; nothing
follows it, and the recording runs on for another 102 s.

Three separate faults meet there, and this module holds the recording that shows
all of them.

**The over was good.** `ONAIR_NAKED_OVER` carries it at 70.174-74.388 s and it
decodes off that recording with a clean CRC, 19 of 24 reference columns, into the
half of the SID banner the session needed most — the gateway's `[WL2K-...]`
identifier and the `;PQ:` secure-login challenge. What the live modem scored was
not that audio. `AudioVaraIO.tx` drops every sample received under one of our own
transmissions, so the stream the over search is fed has a hole in it at every
keying, and `_ov_buf` used to carry straight across that hole: the 3.9 s in front
of the keepalive concatenated to the band behind it. Spliced there the frame
still passes `_OVER_GUARD_MIN` on reference columns and fails its CRC, which is
`_UNDECODED` — an over identified and not readable — and the NAK ladder answers
it. A recogniser that splices is a recogniser that can invent that verdict out of
a frame it broke itself, so the buffer is dropped whenever we key.

**And nothing ended the session.** The gateway never repeated the over, so the
ladder was never asked a second question: the best any alignment in the remaining
102 s of that recording scores is 10 of 24, against a guard of 16.
`_MAX_WITHOUT_PROGRESS` is the bound that should have closed the link instead, and
it could not. `_since_progress` only ever rises inside `idle_keepalive`, and
`mail_session` only called `idle_keepalive` after ten seconds with no bracket.
Replaying that recording's remaining audio through the gate hands over 42 brackets
at a longest gap of 8.50 s, none of them anything the handshake could name — so
the cadence never came due, the budget never spent a tick, and the bound sat armed
behind a branch nothing could reach. The clock is
paced on progress now, which is what its own name has always claimed.

**AND THE GATEWAY WAS NOT SILENT.** Read with the responder's second idle frame
in hand, those 102 s hold thirteen keyings of `session-responder-over-idle`
(288/745) from KC9GHZ on a 3.6 s cadence, each named 0.03-0.44 s from its own
last symbol — the peer waiting between finishing a DATA over and its peer's
answer to it, which is a second on-air instance of the stall the 2026-09-08
KE8LVA mailbox ended in. So the assertions here are on `tx NAK`, the one this
module is about: the re-acknowledgement ladder keys its own last rung into that
cadence, and it is not the verdict on a frame we broke  [see
`VaraStationHandshake._reack`, test_reack_over].

**And a clock is not a channel.** `next_rx_burst` hands back keyed regions only
once they have CLOSED, so a loop reading nothing but its own arrivals is blind
for exactly as long as the peer is transmitting: the cadence falls due into an
open carrier and keys. The gate's open region is the only reading of "somebody is
keying right now" this station has — the over search wants a whole frame before
it will say anything — so a due keepalive waits on it, for one bracket's worth
and no longer. `ONAIR_STANDING_LINK` is the same evening's earlier call to the
same gateway, which ended the same way with no NAK on it to say so.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_mfsk as MK

FS = MK.FS
MYCALL, GATEWAY = "W9SSJ", "KC9GHZ"

kc = corpora.harness("kestrel_connect")

#: The two overs of the greeting, as `vara_payload` delivers them to the host.
#: They join at "has 11" + "8 daily minutes" — one banner, sent in two.
GREETING = (b"RMS Trimode 1.4.2.0 Wadsworth IL - NVIS inverted V at 25 feet "
            b"with 50 watts\r\nW9SSJ has 11")
CHALLENGE = (b"8 daily minutes remaining with KC9GHZ (EN62BK)\r"
             b"[WL2K-5.0-B2FWIHJM$]\r;PQ: 74237263\rCMS via")

#: Where each over sits in the recording, off the frame alignment that decodes it.
_GREETING_AT = (63.693, 67.907)
_CHALLENGE_AT = (70.174, 74.388)

#: Where the mail loop takes over: the far side of our session-confirm's receiver
#: blackout, 0.07 s before the gateway starts the greeting.
_RESUME = 63.62

#: What `AudioVaraIO.tx` discards beyond the transmission itself — the keyed idle
#: hold and the codec-latency margin behind it.
_GUARD_S = kc.TX_IDLE_HOLD_S + kc.RX_ECHO_GUARD_S

#: The longest the 102 s after the NAK ever went without handing the gate a
#: bracket, rounded down to the second — the cadence's worst case on that channel,
#: and still inside the ten it waits.
_WORST_GAP_S = 8

#: Feed granularity. The live transport polls at 0.05 s and hands over whatever
#: arrived; nothing here depends on the figure beyond it being well under a block.
_FEED = int(0.1 * FS)

#: Band audio the gate's floor is primed on before a replay starts, which is what
#: the live transport always has behind it: `AudioVaraIO.prime_floor` runs before
#: the first key-up and every transmission after it keeps the floor across the
#: restart. Started cold instead, the gate's first frames ARE the gateway's over,
#: the quantile lands on the signal, and nothing opens.
_PRIME_S = 2.5


class _Transport(VA.VaraIO):
    """`AudioVaraIO` with a recording in place of the codec.

    The four behaviours that decide this case are the real ones: the receive
    cursor jumps past everything that arrived under a transmission, the raw
    stream goes to the handshake ahead of the gate, the gate's brackets are what
    `next_rx_burst` hands back, and `receiving` reads the gate the way the live
    transport does. Nothing here opens a device or reaches a radio, and the
    recording's own timeline is the clock.
    """

    def __init__(self, x: np.ndarray, start: float):
        self.x = x
        self.cursor = int(start * FS)
        self.seg = kc._BracketSegmenter()
        self.seg.push(x[max(0, int((start - _PRIME_S) * FS)):self.cursor])
        self.seg.restart()
        self.bursts: list[np.ndarray] = []
        self.keyed: list[float] = []
        self.msgs: list[str] = []
        self.payloads: list[bytes] = []

    @property
    def clock(self) -> float:
        return self.cursor / FS

    @property
    def receiving(self) -> bool:
        return self.seg.inb

    def key(self, on):
        if on:
            self.keyed.append(self.clock)

    def tx(self, samples):
        self.cursor += len(samples) + int(_GUARD_S * FS)
        self.seg.restart()
        self.bursts.clear()

    def pending(self): ...

    def connected(self, *a): ...

    def data(self, payload):
        self.payloads.append(bytes(payload))

    def log(self, msg):
        self.msgs.append(msg)

    def next_rx_burst(self, timeout, hs=None):
        end = self.cursor + int(timeout * FS)
        while self.cursor < end:
            n = min(_FEED, end - self.cursor, len(self.x) - self.cursor)
            if n <= 0:
                self.cursor = end       # the recording ran out; the clock did not
                break
            chunk = self.x[self.cursor:self.cursor + n]
            self.cursor += n
            if hs is not None:
                hs.on_rx_stream(chunk)
            self.bursts += [b for _, b in self.seg.push(chunk)]
            if self.bursts:
                return self.bursts.pop(0)
        return None


class _Waiting:
    """A mail client with nothing to say and nothing to end on — the session as
    it stood, waiting for a greeting it had half of."""
    done = False


def _connected(io):
    hs = VA.VaraStationHandshake([MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", GATEWAY, MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs


def _naked_over():
    return corpora.wav_mono(corpora.ONAIR_NAKED_OVER) / 32768.0


@corpora.requires_onair_naked_over
def test_the_over_that_drew_the_nak_decodes_off_the_recording():
    """The measurement the verdict rests on: fed the stream unbroken, both halves
    of the greeting come out, and the second is the one the modem said it could
    not read."""
    io = _Transport(_naked_over(), _RESUME)
    hs = _connected(io)
    while io.cursor < len(io.x):
        chunk = io.x[io.cursor:io.cursor + _FEED]
        io.cursor += len(chunk)
        hs.on_rx_stream(chunk)

    assert io.payloads == [GREETING, CHALLENGE], (
        f"{len(io.payloads)} payload(s) off a recording holding two overs, at "
        f"{_GREETING_AT[0]}-{_GREETING_AT[1]} s and {_CHALLENGE_AT[0]}-"
        f"{_CHALLENGE_AT[1]} s", io.msgs)
    assert not [m for m in io.msgs if "tx NAK" in m], io.msgs


@corpora.requires_onair_naked_over
def test_a_keying_across_an_arriving_over_is_never_scored_as_an_unreadable_one():
    """The splice, isolated: key in the middle of the second over, exactly where
    the keepalive went, and the search must come back with nothing.

    Missing an over costs a retry. Claiming one and NAKing it costs the peer a
    repeat of a frame it already sent correctly, and spends a budget whose whole
    purpose is to bound a decoder that cannot read what the channel delivered.
    """
    io = _Transport(_naked_over(), _RESUME)
    hs = _connected(io)
    interrupted = False
    while io.cursor < len(io.x):
        if not interrupted and io.clock >= _CHALLENGE_AT[0] + 3.7:
            interrupted = True
            hs.idle_keepalive()
        chunk = io.x[io.cursor:io.cursor + _FEED]
        io.cursor += len(chunk)
        hs.on_rx_stream(chunk)

    assert interrupted, "the replay never reached the keying under test"
    assert not [m for m in io.msgs if "tx NAK" in m], (
        "an over spliced across our own transmission was claimed and NAKed",
        io.msgs)
    assert hs._undecoded == 0, "the NAK budget was spent on a frame we broke"


@corpora.requires_onair_naked_over
def test_the_session_reads_the_whole_greeting_and_then_closes_itself(monkeypatch):
    """The whole failure, end to end, through the real mail loop.

    Live this delivered 89 bytes, NAKed the next over, and then sat for 102 s
    without keying or logging anything while the gate handed it 42 brackets it
    could not name. Both halves of that are here: the greeting arrives whole, and
    when the gateway stops saying anything the session recognises, the link is
    closed rather than left standing.
    """
    x = _naked_over()
    io = _Transport(x, _RESUME)
    hs = _connected(io)
    monkeypatch.setattr(kc.time, "time", lambda: io.clock)

    # Past the recording's own end, because what is asserted below is that the
    # LINK LAYER closes and not that the tape runs out first: the cadence is
    # ~10-12 s and the recording ends one tick short of the budget, so a timeout
    # cut to its length would pass or fail on the length of the capture.
    kc.mail_session(hs, io, _Waiting(), timeout=len(x) / FS - _RESUME + 30)

    assert io.payloads == [GREETING, CHALLENGE], (
        f"{len(io.payloads)} payload(s) of the two the gateway sent", io.msgs)
    assert not [m for m in io.msgs if "tx NAK" in m], io.msgs
    assert hs._undecoded == 0, "the NAK budget was spent on a frame we broke"
    assert hs.state is not VA.VaraState.CONNECTED, (
        "the link was still held after the gateway stopped answering", io.msgs)
    assert any("closing the link" in m for m in io.msgs), io.msgs


def test_the_idle_budget_is_spent_by_time_and_not_by_whatever_the_gate_brackets(
        monkeypatch):
    """A channel that brackets faster than the cadence must not hold the link open.

    Measured on the 102 s after that NAK: 42 brackets, longest gap 8.50 s, and not
    one of them anything the handshake could name. Against a cadence reset by
    traffic that is a link with no way out — the budget below spends only while
    the modem is keying, and it was never given the chance to key.
    """
    class _Io(VA.VaraIO):
        def __init__(self):
            self.now = 0.0
            self.msgs: list[str] = []
            self.keyed = 0

        def key(self, on):
            self.keyed += on

        def tx(self, samples): ...

        def log(self, msg):
            self.msgs.append(msg)

        def next_rx_burst(self, timeout, hs=None):
            self.now += timeout
            return np.zeros(4096) if int(self.now) % _WORST_GAP_S == 0 else None

    io = _Io()
    hs = _connected(io)
    monkeypatch.setattr(kc.time, "time", lambda: io.now)
    kc.mail_session(hs, io, _Waiting(), timeout=600.0, keepalive_s=10.0)

    assert hs.state is not VA.VaraState.CONNECTED, (
        f"{io.keyed} keyings and the link is still up after 600 s of a channel "
        f"nobody was answering on", io.msgs[-6:])
    assert io.now < 600.0, "the loop sat out the whole mail timeout"


#: Where the mail loop takes over in `ONAIR_STANDING_LINK` — the far side of that
#: session's session-confirm, 0.3 s before its gateway starts the one over it sent.
_STANDING_RESUME = 17.60

#: That over, as the gate brackets it.
_STANDING_OVER = (17.90, 22.85)

#: A cadence short enough to fall due while a gateway over is still on the air,
#: which is where the live one fell: the loop resumed at `_RESUME` and its first
#: keepalive was due at 73.62 s, 3.4 s into the over at `_CHALLENGE_AT`. Pacing
#: the clock on `hs.progress` moved that phase off this recording — the answered
#: greeting restarts it — so the case is put back by shortening the cadence
#: rather than by editing the recording.
_CADENCE_INTO_AN_OVER = 2.0


def _keepalives(hs, io) -> list[float]:
    """Where in the recording the loop keys each idle-cadence burst.

    The cadence and nothing else: an answer to the peer's own over is keyed on
    the turnaround it is owed, and gating a reply on the channel would suppress
    it with the very signal it answers.
    """
    fired: list[float] = []
    keyed = hs.idle_keepalive

    def watched():
        fired.append(io.clock)
        keyed()

    hs.idle_keepalive = watched
    return fired


def _run(io, hs, keepalive_s, monkeypatch):
    monkeypatch.setattr(kc.time, "time", lambda: io.clock)
    kc.mail_session(hs, io, _Waiting(), timeout=len(io.x) / FS - io.clock,
                    keepalive_s=keepalive_s)


@corpora.requires_onair_naked_over
def test_no_keepalive_is_keyed_into_an_over_the_gateway_is_still_sending(monkeypatch):
    """The collision itself, on the recording that produced it.

    Live this station keyed 3.7 s into `_CHALLENGE_AT` and lost the half of the
    greeting it needed most. `next_rx_burst` hands back only regions that have
    CLOSED, so a loop reading nothing but its own arrivals is deaf for exactly as
    long as the peer is transmitting — the cadence comes due into an open carrier
    and keys.
    """
    io = _Transport(_naked_over(), _RESUME)
    hs = _connected(io)
    fired = _keepalives(hs, io)
    _run(io, hs, _CADENCE_INTO_AN_OVER, monkeypatch)

    assert fired, "the cadence never came due; the case was not exercised"
    into = [t for t in fired
            if _GREETING_AT[0] <= t <= _GREETING_AT[1]
            or _CHALLENGE_AT[0] <= t <= _CHALLENGE_AT[1]]
    assert not into, (f"keepalive keyed at {into} s, on top of an over the "
                      f"gateway was still sending", io.msgs)
    assert io.payloads == [GREETING, CHALLENGE], (
        f"{len(io.payloads)} payload(s) of the two the gateway sent", io.msgs)


@corpora.requires_onair_standing_link
@pytest.mark.parametrize("keepalive_s", (10.0, _CADENCE_INTO_AN_OVER))
def test_the_session_that_stood_answers_its_over_and_then_closes(
        keepalive_s, monkeypatch):
    """The same ending on the same evening's earlier call, with no NAK on it.

    That log stops at `tx per-over response` while its recording runs on for
    another 213 s: one over answered, and then a held link nobody was talking on.
    At the cadence it ran at nothing of ours falls anywhere near that over; at one
    that falls due inside it, the over survives anyway.
    """
    x = corpora.wav_mono(corpora.ONAIR_STANDING_LINK) / 32768.0
    io = _Transport(x, _STANDING_RESUME)
    hs = _connected(io)
    fired = _keepalives(hs, io)
    _run(io, hs, keepalive_s, monkeypatch)

    assert len(io.payloads) == 1, (
        f"{len(io.payloads)} payload(s) off the one over in this recording",
        io.msgs)
    assert not [t for t in fired
                if _STANDING_OVER[0] <= t <= _STANDING_OVER[1]], (fired, io.msgs)
    assert hs.state is not VA.VaraState.CONNECTED, (
        f"{len(fired)} keepalives and the link is still standing", io.msgs)
    assert any("closing the link" in m for m in io.msgs), io.msgs


def test_the_channel_clearing_produces_no_keepalive_in_the_peers_turn(monkeypatch):
    """In the peer's turn a stock caller keys nothing on its own clock.

    The cadence that used to key a keepalive here is gone: the channel clearing
    draws no burst, because there is no unprompted cadence in the peer's turn to
    keep a deadline for. The peer's own over-idle draws the single reactive
    keepalive instead, and a peer gone silent is closed on the give-up budget
    [see vara_arq.idle_keepalive, vara_arq._stream_answer].
    """
    class _Io(VA.VaraIO):
        def __init__(self, busy_until: float):
            self.now, self.busy_until = 0.0, busy_until
            self.keyed: list[float] = []
            self.msgs: list[str] = []

        @property
        def receiving(self) -> bool:
            return self.now < self.busy_until

        def key(self, on):
            if on:
                self.keyed.append(self.now)

        def tx(self, samples): ...

        def log(self, msg):
            self.msgs.append(msg)

        def next_rx_burst(self, timeout, hs=None):
            self.now += timeout
            return None

    io = _Io(busy_until=13.0)
    hs = _connected(io)
    monkeypatch.setattr(kc.time, "time", lambda: io.now)
    kc.mail_session(hs, io, _Waiting(), timeout=26.0, keepalive_s=10.0)

    assert not io.keyed, (
        f"keyed on our own clock in the peer's turn: {io.keyed}", io.msgs)
    assert hs.state is VA.VaraState.CONNECTED


def test_a_channel_that_never_clears_still_tears_down_the_link(monkeypatch):
    """The other way to become the station talking to nobody.

    A bracket that never closes — a carrier, a busy channel, a gate held open by
    the band — must not hold the teardown budget off for good. The caller keys no
    keepalive on its own clock in the peer's turn, but the give-up budget still
    spends on the silence timer, so the link is torn down rather than held open
    for the whole mail timeout  [see vara_arq.idle_keepalive].
    """
    class _Io(VA.VaraIO):
        receiving = True

        def __init__(self):
            self.now = 0.0
            self.keyed: list[float] = []
            self.msgs: list[str] = []

        def key(self, on):
            if on:
                self.keyed.append(self.now)

        def tx(self, samples): ...

        def log(self, msg):
            self.msgs.append(msg)

        def next_rx_burst(self, timeout, hs=None):
            self.now += timeout
            return None

    io = _Io()
    hs = _connected(io)
    monkeypatch.setattr(kc.time, "time", lambda: io.now)
    kc.mail_session(hs, io, _Waiting(), timeout=600.0, keepalive_s=10.0)

    assert not any("keepalive" in m for m in io.msgs), (
        "keyed a keepalive on our own clock into a busy peer's turn", io.msgs)
    assert hs.state is not VA.VaraState.CONNECTED, (
        f"{len(io.keyed)} keyings into a channel that never cleared, and the "
        f"link is still up", io.msgs[-4:])
    assert io.now < 600.0, "the loop sat out the whole mail timeout"
