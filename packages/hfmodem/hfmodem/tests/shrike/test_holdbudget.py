# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What ends a hold, which is a condition and not a duration.

The `--hold` loop counted cycles. Its deadline was `--hold` and nothing a session
did could move it, so the one thing that decides whether hanging up is right --
whether the exchange is still moving -- was never consulted. Twelve cycles is the
correct answer for a peer that has gone and the wrong one for a peer that is
mid-sentence, and the loop could not tell the two apart.

MEASURED, K4MSU on 3595 kHz, 2026-08-19 22:10, `captures/onair-0819-2210`. The
session held twelve cycles hearing codewords only -- CS4 CS4 . CS1 CS1 CS1 . CS2
. . . . -- and in cycle 13 the gateway took the channel with a CS3-headed packet
and put `RMS Tri` in it, the opening of "RMS Trimode". Cycle 13 is also the cycle
the constant deadline expired in. The station queued its goodbye in the same
cycle the exchange first moved, and the mail stage was still `awaiting greeting`
when it did.

`RMS Tri` reads like a truncated greeting and is not one: `BREAKIN_FIELD[100]` is
7 bytes, so it is a full field, and `hold_13` and `hold_15` both decode it under
packet counter 0. Two windows two cycles apart carrying one packet is an ISS
repeating what its IRS never acknowledged -- the gateway was waiting on an answer
this station spent its cycle 13 declining to send. Nothing was lost to the
channel and nothing was cut off mid-transfer; the exchange stopped because the
deadline landed on it.

So the deadline here is an IDLE timeout: it sits `--hold` cycles past the last
cycle in which payload crossed the link, and `HOLD_MAX_CYCLES` is the ceiling it
cannot be pushed past. Against a peer that has gone, the first cycle is the last
one that moved anything and the hold ends at exactly `--hold` -- what the loop
did before. What buys a cycle is BYTES: a codeword buys nothing, and neither does
the idle fill an ISS with an empty buffer transmits, which decodes to nothing and
is never delivered. Both of those arrive every cycle of a link that is doing
nothing at all, and either one read as "the peer is talking" would hold a shared
channel open on a station with nothing to say.

Run: python -m hfmodem.tests.shrike.test_holdbudget
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1
from hfmodem.shrike.arq import (GOODBYE_CYCLES, IRS, ISS, LONG_TICKS,
                                PactorArq, State)
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_qrtack import (CS1, CS2, CS3, CS4,
                                              calling_station, cs_audio)

#: The windows that session wrote, on this station's disk and no other clone's.
CAPTURES = evidence.CAPTURES / "onair-0819-2210"

#: The 22:10 session's hold, cycle by cycle, off its own log. A codeword index is
#: the control signal that cycle decoded, `None` a cycle that decoded nothing,
#: and `bytes` the payload of a CS3-headed break-in packet.
TRAIN = (CS4, CS4, None, CS1, CS1, CS1, None, CS2, None, None, None, None,
         b"RMS Tri")
HOLD = 12

# A cycle that moved nothing, for the ceiling arm: it is the SLOTS that bind.
_LINK = onair._Link(0, 0, ISS)
GREETING = 13

#: The same session as `TRAIN`, as replaying its own `stream.wav` decodes it:
#: window boundaries a cycle out from the live ones, so the gateway's break-in
#: reaches the receiver as a bare CS3 and the packet behind it is never
#: delivered. `GRAB` is the cycle it lands in, which is the cycle the byte rule
#: alone puts the deadline on.
REPLAYED = (CS4, CS1, CS1, CS2, CS2, CS2, CS2, None, None, None, None, None,
            None, CS3, None, None, None, None)
GRAB = 14

#: A line of our own, and how many cycles put one on the link. `ANNOUNCE` is the
#: callsign announcement the link layer queues for itself when the connect is
#: answered, which is the one thing every hold here sends without being asked.
LINE = b"the quick brown fox "
TALKING = 8
ANNOUNCE = 7

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def hold(train, *, base: int = HOLD, blind: bool = False,
         ceiling: int = 1 << 30, saying: int = 0) -> dict:
    """One hold, through the real receive path and the real budget.

    `blind` is the station that shipped: the same budget, told nothing about what
    crossed the link, whose deadline therefore never moves off `base`.

    `saying` is how many of the first cycles queue a line of our own -- a station
    mid-message rather than one holding an open link with nothing on it.
    """
    host = calling_station()
    rx = onair._SessionRx(host)
    budget = onair._HoldBudget(base, ceiling)
    saved, onair.LINKED = onair.LINKED, (State.CONNECTED, State.DISCONNECTING)
    goodbye, deadlines = None, []
    try:
        for h, step in enumerate(train, 1):
            if host.arq.state not in onair.LINKED:
                break
            if h <= saying:
                host.arq.on_host_data(LINE)
            rx.new_cycle()
            was = onair._Link.of(host)
            if isinstance(step, bytes):
                rx.deep_scan(np.asarray(
                    pactor1.breakin_signal(step, 100, packet_count=0),
                    np.float32))
            else:
                rx.control_signal(cs_audio(step), 0, round(0.0938 * onair.FS))
            if not blind:
                budget.cycle(h, was, onair._Link.of(host))
            if h > budget.deadline and goodbye is None:
                host.arq.on_host_disconnect()
                goodbye = h
            host.tick()
            deadlines.append(budget.deadline)
    finally:
        onair.LINKED = saved
    return {"host": host, "goodbye": goodbye, "deadlines": deadlines,
            "budget": budget}


def acking(cycles: int) -> tuple[int, ...]:
    """A peer that acknowledges every cycle, which is not one codeword repeated.

    In PACTOR-1 the acknowledgement IS the alternation -- CS1 and CS2 are one
    signal in two phases -- so a peer sending CS1 forever acknowledges the packet
    in flight once and nothing after it. The two alternating are what a healthy
    link sounds like, and what `_on_ack` runs on.
    """
    return tuple(CS1 if h % 2 else CS2 for h in range(cycles))


def acknowledged(*args, **kwargs) -> tuple[dict, list[int]]:
    """One hold, plus the payload that was under each acknowledgement in it."""
    seen, real = [], PactorArq._on_ack

    def counted(self):
        if self._inflight is not None:
            seen.append(len(self._inflight.payload))
        return real(self)

    PactorArq._on_ack = counted
    try:
        return hold(*args, **kwargs), seen
    finally:
        PactorArq._on_ack = real


# --------------------------------------------------------------------------
def the_cycle_the_greeting_arrived_in() -> None:
    """The 22:10 hold, replayed against both deadlines."""
    print("\nThe K4MSU hold of 2026-08-19 22:10, cycle by cycle")
    shipped = hold(TRAIN, blind=True)
    fixed = hold(TRAIN)
    print(f"    deadline held at {shipped['deadlines'][-1]} blind, "
          f"pushed to {fixed['deadlines'][-1]} on the bytes")

    check("the gateway's seven bytes reach the host in cycle 13 -- the greeting "
          "is on the link before that cycle's key",
          fixed["host"].rcvd_total == len(TRAIN[-1]),
          f"{fixed['host'].rcvd_total}B")
    check("WHAT SHIPPED: the goodbye is queued in cycle 13, the cycle the "
          "exchange first moved", shipped["goodbye"] == GREETING,
          f"cycle {shipped['goodbye']}")
    check("THE FIX: seven bytes buy another `--hold` cycles and no goodbye is "
          "queued at all", fixed["goodbye"] is None
          and fixed["deadlines"][-1] == GREETING + HOLD,
          f"goodbye {fixed['goodbye']}, deadline {fixed['deadlines'][-1]}")

    # OUR OWN BYTES COUNT TOO, and this hold is where that shows: the link layer
    # queues a 7-byte callsign announcement of its own when the connect is
    # answered, and the peer's first alternation -- CS4 to CS1 in cycle 4 -- is
    # what confirms it. The recorded session advanced its packet counter in the
    # same place, `#1 x3` then `#2`.
    moved_in = [h for h, (a, b) in enumerate(
        zip(fixed["deadlines"], (HOLD, *fixed["deadlines"])), 1) if a != b]
    check("two cycles moved payload, and the first is ours: the announcement "
          "confirmed in cycle 4, the greeting in cycle 13",
          moved_in == [4, GREETING] and fixed["host"].sent_total == 7,
          f"cycles {moved_in}, {fixed['host'].sent_total}B sent")


def the_channel_the_gateway_took() -> None:
    """A break-in whose packet does not reach the host, which is most of them."""
    print("\nThe cycle the gateway took the transmit direction")
    run = hold(REPLAYED)
    grabbed = run["host"]

    check("the changeover reaches the receiver with nothing behind it: the CS3 "
          "hands us the receive role and no payload is delivered",
          grabbed.arq.role == IRS and grabbed.rcvd_total == 0,
          f"role {grabbed.arq.role}, {grabbed.rcvd_total}B received")
    check(f"bytes alone put the deadline on cycle {GRAB}, the cycle the "
          f"break-in lands in", run["deadlines"][GRAB - 2] == GRAB,
          f"deadline {run['deadlines'][GRAB - 2]} entering cycle {GRAB}")

    silent = hold(REPLAYED[:GRAB - 1] + (None,) * 5)
    check("NEGATIVE CONTROL, bytes and only bytes: with that cycle silent "
          f"instead the goodbye is queued in cycle {GRAB + 1}",
          silent["goodbye"] == GRAB + 1, f"cycle {silent['goodbye']}")
    check("THE FIX: the peer asking for the channel buys it the same "
          "`--hold` cycles a payload does, and no goodbye breaks in over the "
          "packet it asked to send",
          run["goodbye"] is None and run["deadlines"][-1] == GRAB + HOLD,
          f"goodbye {run['goodbye']}, deadline {run['deadlines'][-1]}")

    done = onair._HoldBudget(HOLD)
    done.close(3)
    done.cycle(4, onair._Link(0, 0, ISS), onair._Link(0, 7, IRS))
    check("...and a finished exchange is not reopened by one: `close` is final, "
          "so neither a changeover nor a byte after it holds the channel past "
          "the goodbye it owes", done.deadline == 3, f"deadline {done.deadline}")


def a_peer_that_has_gone() -> None:
    """Nothing on the channel at all, which is what has to terminate."""
    print("\nA hold nothing answers")
    quiet = hold((None,) * (HOLD * 4))
    retries = quiet["host"].arq.cfg.max_retries
    check("the deadline never moves off `--hold`: silence buys nothing, so a "
          "hold conditioned on bytes is no longer than the count it replaces",
          set(quiet["deadlines"]) == {HOLD},
          f"deadlines {sorted(set(quiet['deadlines']))}")
    check("...and the deadline is not even what stops it. The ARQ gives up on "
          f"an unanswered packet after {retries} retries and spends "
          f"{GOODBYE_CYCLES} more putting a QRT on the air, so the link is "
          f"down in cycle {retries + GOODBYE_CYCLES + 3} -- inside the budget, "
          "on a QRT the state machine owed and sent for itself",
          len(quiet["deadlines"]) == retries + GOODBYE_CYCLES + 3
          and quiet["host"].arq.state not in onair.LINKED
          and quiet["host"].arq.said_goodbye,
          f"{len(quiet['deadlines'])} cycles, {quiet['host'].arq.state}, "
          f"goodbye {quiet['host'].arq.said_goodbye}")


def what_does_not_buy_a_cycle() -> None:
    """A link that is doing nothing still fills every cycle."""
    print("\nCodewords and idle fill, every cycle, for longer than the budget")
    acks = hold((CS1,) * (HOLD * 2))
    check("an acknowledgement every cycle buys nothing: the peer is answering, "
          "not saying anything. The one move is our own announcement, "
          "confirmed by the first alternation, and the deadline never shifts "
          "again over 24 cycles of codewords",
          acks["deadlines"] == [1 + HOLD] * len(acks["deadlines"])
          and acks["goodbye"] == HOLD + 2,
          f"goodbye {acks['goodbye']}, deadlines "
          f"{sorted(set(acks['deadlines']))}")

    host = calling_station()
    rx = onair._SessionRx(host)
    idle = np.asarray(pactor1.packet_signal(b"", 100, packet_count=1),
                      np.float32)
    saved, onair.LINKED = onair.LINKED, (State.CONNECTED, State.DISCONNECTING)
    try:
        rx.new_cycle()
        rx.deep_scan(idle)
    finally:
        onair.LINKED = saved
    check("...and neither does the idle fill an ISS with an empty buffer sends: "
          "it decodes to nothing and is never delivered", host.rcvd_total == 0,
          f"{host.rcvd_total}B delivered")


def a_peer_that_answers_every_cycle() -> None:
    """The clean link, where every codeword is an acknowledgement.

    The case above repeats one codeword, and a repeat is not an acknowledgement:
    the state machine takes the first and ignores the rest, so its `_on_ack` runs
    once in the whole hold. A peer that alternates runs it in every cycle of
    the hold -- and `_on_ack` is where `sent_total` moves, which is half of
    what the budget reads. Before `p1rx.CS_SEARCH_HALF_S` those reads were lost
    often enough that a hold hit its ceiling on the missing ones; recovering them
    makes this link the ordinary one, so what the budget does under it is worth
    an assertion of its own rather than an inference.

    The answer is that the acknowledgement is not what moves the counter. What
    leaves the buffer does: an ISS with nothing to send fills its packet with
    0x1E, the field is empty, and an acknowledged empty field subtracts nothing.
    """
    print("\nA peer that acknowledges every cycle, over an empty buffer")
    quiet, acks = acknowledged(acking(HOLD * 2))
    check("every cycle is acknowledged, and only the first has anything under "
          "it: the announcement, and then the idle fill",
          len(acks) > HOLD and acks[0] == ANNOUNCE and set(acks[1:]) == {0},
          f"{len(acks)} acknowledgements, payloads {sorted(set(acks))}")
    check("the hold still expires on N: the deadline stays in the cycle our "
          f"announcement was confirmed in, the goodbye is queued in cycle "
          f"{HOLD + 2}, and the link closes on it",
          set(quiet["deadlines"]) == {1 + HOLD}
          and quiet["goodbye"] == HOLD + 2 and quiet["host"].arq.said_goodbye,
          f"goodbye {quiet['goodbye']}, deadlines "
          f"{sorted(set(quiet['deadlines']))}")

    # THE OTHER HALF, and it is the half a fix that stopped reading `sent_total`
    # would break: the same peer, against a station that HAS something to say.
    # A 100 Bd field is 8 bytes and a line is 20, so the buffer fills faster than
    # the link drains it and the payload outlasts the cycles that queued it.
    moving, moving_acks = acknowledged(acking(HOLD * 4), saying=TALKING)
    moved_in = [h for h, (a, b) in enumerate(
        zip(moving["deadlines"], (HOLD, *moving["deadlines"])), 1) if a != b]
    last = moved_in[-1]
    check("a station mid-message is not cut short: every cycle that moved "
          "payload pushed the deadline, and the run of them outlasts the cycles "
          "that queued the bytes",
          moved_in == list(range(1, last + 1)) and last > TALKING,
          f"{len(moved_in)} moving cycles, the last in {last}")
    check("...and the hold ends `--hold` cycles after the last of them, with "
          "every byte acknowledged",
          moving["goodbye"] == last + HOLD + 1
          and moving["host"].sent_total == ANNOUNCE + TALKING * len(LINE),
          f"goodbye {moving['goodbye']}, deadline {last + HOLD}, "
          f"{moving['host'].sent_total}B sent over "
          f"{len(moving_acks)} acknowledgements")


def a_peer_that_never_stops() -> None:
    """The bound. A grant with no ceiling is a transmitter held on a channel."""
    print("\nBytes every cycle, against the ceiling")
    budget = onair._HoldBudget(HOLD, 40)
    for h in range(1, 200):
        budget.moved(h)
    check("the deadline stops at the ceiling however long the peer talks",
          budget.deadline == 40, str(budget.deadline))
    check(f"and the shipped ceiling is {onair.HOLD_MAX_CYCLES} slots, "
          f"{onair.HOLD_MAX_CYCLES * 1.25 / 60:.0f} minutes of channel",
          onair.HOLD_MAX_CYCLES * 1.25 <= 600)

    # THE CEILING IS TIME AND THE HOLD IS TURNS. A long cycle is three grid
    # slots, so 480 of them would be half an hour of a shared channel rather
    # than the ten minutes the constant promises. The slots a cycle spent are
    # charged, and the ceiling binds on those.
    long = onair._HoldBudget(1 << 20, 40)
    for h in range(1, 200):
        long.spend(LONG_TICKS)
        long.moved(h)
        if h > long.deadline:
            break
    check("a long-cycle hold reaches the ceiling in a third of the turns, "
          "which is the same span of channel",
          long.deadline == -(-40 // LONG_TICKS) and h == long.deadline + 1,
          f"stopped in hold cycle {h} at deadline {long.deadline}, "
          f"{long.slots} slots spent")
    check("...and it says which bound stopped it, in the unit it is kept in",
          "40-slot ceiling" in long.ran_out, long.ran_out)


def a_hold_longer_than_the_ceiling() -> None:
    """`--hold` past the ceiling, where the ceiling used to run backwards.

    The clamp lived in `moved` alone, so `--hold 1000` started with a deadline of
    1000 and the peer's first byte pulled it IN to 480: silence bought the full
    grant and talking spent it. That is the inverse of the rule this class exists
    for, and the inverse of what the ceiling means -- the furthest out a hold can
    ever be pushed is not a place a hold can start beyond.

    Latent when it was found: the longest `--hold` anywhere on disk is
    `rigtest.sh`'s 150, and every value under the ceiling behaves identically
    either way. It is one flag away from not being latent.
    """
    print("\nA --hold past the ceiling")
    over = onair._HoldBudget(1000, 480)
    check("the deadline is the ceiling's from cycle 1, before the peer has said "
          "anything at all", over.deadline == 480, str(over.deadline))
    was = over.deadline
    over.moved(5)
    check("...and a byte in cycle 5 does not pull it IN: bytes buy cycles, they "
          "have never spent them", over.deadline == was == 480,
          f"{was} -> {over.deadline}")
    check("the hold that ends names the bound that ended it -- the ceiling, and "
          "not an idle timeout that never came near expiring",
          "480-slot ceiling" in over.ran_out and "idle" not in over.ran_out,
          over.ran_out)
    ordinary = onair._HoldBudget(HOLD, 480)
    ordinary.moved(3)
    check("...while a hold the ceiling never touched still reports its idle "
          f"budget, which is the {HOLD} cycles the operator asked for",
          ordinary.ran_out == f"the hold's {HOLD} idle cycles ran out",
          ordinary.ran_out)


def what_the_gateway_actually_sent() -> None:
    """The windows themselves, read off disk. Our own receiver, and no other.

    `mail: peer said: RMS Tri` reads like a truncated greeting -- seven bytes of
    "RMS Trimode" -- and it is not. `pactor1.BREAKIN_FIELD[100]` is 7, so those
    seven bytes ARE the field: a whole changeover packet, filled, with nothing
    lost off either end. What follows it is the gateway's next packet, and in
    PACTOR-1 memory ARQ an ISS does not reach its next packet until the IRS
    acknowledges this one -- which this station answered with a goodbye instead.
    """
    print(f"\nThe windows {CAPTURES.name} wrote")
    heard = {}
    for name in ("hold_13", "hold_15"):
        audio = onair.session.load_wav(str(CAPTURES / f"{name}.wav"), onair.FS)
        heard[name] = [p for p in p1rx.decode_p1_packets(audio, breakin=True)]
        print(f"    {name}.wav {audio.size / onair.FS:.3f} s: {heard[name]}")

    got = [p for pkts in heard.values() for p in pkts]
    check("the greeting is a FULL break-in field, not a fragment of one",
          all(p.payload == b"RMS Tri" for p in got)
          and len(b"RMS Tri") == pactor1.BREAKIN_FIELD[100], f"{got[0]}")
    check("and both windows carry it under the same packet counter, two cycles "
          "apart: the gateway was repeating an unacknowledged packet, not "
          "sending a second one", len(got) == 2
          and {p.status & 3 for p in got} == {0},
          f"{len(got)} packets, counters {[p.status & 3 for p in got]}")


STAGES = (the_cycle_the_greeting_arrived_in, the_channel_the_gateway_took,
          a_peer_that_has_gone, what_does_not_buy_a_cycle,
          a_peer_that_answers_every_cycle, a_peer_that_never_stops,
          a_hold_longer_than_the_ceiling)
RECORDED = (what_the_gateway_actually_sent,)


def _run(stages) -> int:
    global ok
    ok = True
    for stage in stages:
        stage()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def test_main() -> None:
    assert _run(STAGES) == 0


@pytest.mark.skipif(
    not CAPTURES.is_dir(),
    reason=f"{CAPTURES} is not on this machine -- on-air captures are gitignored "
           "and live only in the checkout that recorded them")
def test_what_the_gateway_actually_sent() -> None:
    assert _run(RECORDED) == 0


if __name__ == "__main__":
    sys.exit(_run(STAGES + (RECORDED if CAPTURES.is_dir() else ())))
