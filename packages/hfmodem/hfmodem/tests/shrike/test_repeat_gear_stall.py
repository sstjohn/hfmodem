# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The way out of a run of identical packets answered identically.

WS8EOC, 40 m, 2026-09-13 (`working/pactor3-header-0913/cycle-ledger`). Three
arms broke in on the same gateway within the hour. The two that delivered the
whole greeting keyed a CS4 speed-up request inside the first three accepted
packets; the one that stalled keyed none at all, and its transcript is 36
consecutive copies of `SL1 DATA status=0x21 seq=1 b' Trim'` answered CS2 every
time -- the correct alternation word for an odd counter, and the same word the
two delivering arms keyed against the same byte. The gateway never advanced,
then stopped transmitting.

`_gear_cs`'s climb cannot reach that link: `speed_up_after` spends ACCEPTED
traffic packets and a repeat is not one, so a repeating peer produces no clean
run and the link never leaves the level it stalled on. `cfg.repeat_gear` counts
the ANSWERS instead -- N identical packets answered with one identical codeword,
and the next repeat draws CS4, which substitutes in the acknowledgement's own
slot (pactor3.md 7) and so acknowledges the counter it replaces.

Everything below drives the real `PtcHost` seam, so the codewords asserted on are
the physical ones the counter alternation produces, not the FSM's logical
alphabet. The default is off and the last two scenes pin what off means.

Run:  python -m pytest hfmodem/tests/shrike/test_repeat_gear_stall.py
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import arq, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

#: The byte the gateway repeated: counter 1, ASCII, status bit 5 up -- the
#: long-cycle ask declined in the historical short-cycle fixture below. The
#: current modem supports long SL1; these speed tests disable it explicitly.
STALL_STATUS = 0x21

#: The level the whole stall ran at, and the level the repeated packet arrived
#: at on all three arms.
STALL_SL = 1

FIELD = b" Trim"


class Seam:
    """A transmit seam that keys nothing and remembers every codeword.

    `send_p1_cs` exists so `PtcHost.send_cs` routes through `_counter_cs_for`:
    a seam without it takes the FSM's own alphabet unchanged, and the
    alternation is exactly what these scenes are reading.
    """

    # Recording a word is the synchronous emission in this fixture, not a
    # deferred queue requiring a later driver on_cs_emitted callback.
    defer_p3_cs = False

    def __init__(self) -> None:
        self.words: list[int] = []
        self.packets: list[tuple[int, int]] = []

    def attach(self, host) -> None: ...

    def send_cs(self, index: int) -> None:
        self.words.append(index)

    def send_p1_cs(self, index: int) -> None:
        self.words.append(index)

    def send_packet(self, sl, payload, status, breakin=False) -> int:
        self.packets.append((sl, status))
        return len(payload)

    def __getattr__(self, name):
        return lambda *a, **k: None


def linked(*, repeat_gear: int = 0,
           speed_up_after: int = 1000, long_cycle: bool = False) -> tuple[PtcHost, Seam]:
    """A PACTOR-3 link in the state the break-in left: connected, receiving.

    `speed_up_after` is parked out of reach by default so a run of answers is
    the only thing that can produce a CS4; the two scenes that are about the
    ordinary climb pass the real threshold back in.
    Long-cycle negotiation is disabled to reproduce the historical reply train
    and isolate CS4 behavior. Tests of its interaction with speed hold opt in.
    """
    seam = Seam()
    host = PtcHost(peer=seam, mycall="W9SSJ")
    host.arq.cfg.repeat_gear = repeat_gear
    host.arq.cfg.speed_up_after = speed_up_after
    host.arq.cfg.long_cycle = long_cycle
    host.arq.role, host.arq.dxcall = arq.IRS, "WS8EOC"
    host.arq._enter_connected()
    host.protocol = Protocol.PACTOR3
    return host, seam


def arrives(host: PtcHost, *, status: int = STALL_STATUS, sl: int = STALL_SL,
            field: bytes = FIELD, crc: bool = True, cycle_long: bool = False) -> None:
    """One cycle of the peer's, through the receiver's own event."""
    host.on_rx_event(rxfront.Event(0.1, "packet", "p3",
                                   protocol=Protocol.PACTOR3,
                                   packet=(sl, status, field, crc),
                                   cycle_long=cycle_long))


def status_at(seq: int, *, long_cycle: bool = True, qrt: bool = False) -> int:
    return spec.status_byte(seq, data_type=spec.DataType.ASCII_8BIT,
                            long_cycle_request=long_cycle, qrt=qrt)


def test_the_stall_transcript_reproduces_word_for_word():
    """Today's behaviour, and the arm that died in it."""
    host, seam = linked()
    for _ in range(36):
        arrives(host)
    assert seam.words == [arq.CS_REQUEST] * 36
    assert arq.CS_SPEED_UP not in seam.words
    assert host.arq.rx_progress == 1


def test_three_identical_answers_and_the_fourth_repeat_asks_for_a_gear():
    host, seam = linked(repeat_gear=3)
    for _ in range(4):
        arrives(host)
    assert seam.words == [arq.CS_REQUEST] * 3 + [arq.CS_SPEED_UP]


def test_the_flag_at_zero_keeps_the_fourth_answer_an_alternation_word():
    host, seam = linked(repeat_gear=0)
    for _ in range(4):
        arrives(host)
    assert seam.words == [arq.CS_REQUEST] * 4


def test_the_gear_request_acknowledges_the_counter_it_replaced():
    """CS4 substitutes in the acknowledgement's slot and takes nothing with it."""
    host, seam = linked(repeat_gear=3)
    for _ in range(4):
        arrives(host)
    assert seam.words[-1] == arq.CS_SPEED_UP
    assert host.arq.rx_seq == 1
    assert host.arq.rx_progress == 1
    assert bytes(host.channel(host.ptchn).rx) == FIELD


def test_a_peer_that_advances_is_answered_from_the_new_counter():
    host, seam = linked(repeat_gear=3)
    for _ in range(4):
        arrives(host)
    arrives(host, status=status_at(2), field=b"ode")
    arrives(host, status=status_at(3), field=b" more")
    assert seam.words == ([arq.CS_REQUEST] * 3 + [arq.CS_SPEED_UP]
                          + [arq.CS_ACK, arq.CS_REQUEST])
    assert bytes(host.channel(host.ptchn).rx) == FIELD + b"ode more"
    assert host.arq.rx_progress == 3


def test_the_run_restarts_behind_its_own_request():
    """A peer that goes on repeating is asked again N later, not every cycle."""
    host, seam = linked(repeat_gear=3)
    for _ in range(8):
        arrives(host)
    assert seam.words == ([arq.CS_REQUEST] * 3 + [arq.CS_SPEED_UP]) * 2


def test_an_advance_between_repeats_starts_a_new_run():
    host, seam = linked(repeat_gear=3)
    arrives(host)
    arrives(host)
    arrives(host, status=status_at(2), field=b"ode")
    for _ in range(2):
        arrives(host, status=status_at(2), field=b"ode")
    assert arq.CS_SPEED_UP not in seam.words
    assert seam.words == [arq.CS_REQUEST] * 2 + [arq.CS_ACK] * 3


def test_a_goodbye_is_acknowledged_and_never_geared():
    host, seam = linked(repeat_gear=1)
    qrt = status_at(1, qrt=True)
    arrives(host, status=qrt, field=b"")
    arrives(host, status=qrt, field=b"")
    assert arq.CS_SPEED_UP not in seam.words


def test_the_top_rung_has_nothing_to_ask_for():
    host, seam = linked(repeat_gear=2)
    top = host.arq._top
    for _ in range(4):
        arrives(host, sl=top, status=status_at(1, long_cycle=False))
    assert seam.words == [arq.CS_REQUEST] * 4


def test_the_pactor1_seam_is_never_asked_to_speed_up():
    """CS4's index is Speedchange there -- a repeat request at 100 Bd and a
    reject at 200 -- so the run may not reach it (`arq._gear_cs`)."""
    seam = Seam()
    host = PtcHost(peer=seam, mycall="W9SSJ")
    host.arq.cfg.repeat_gear = 2
    host.arq.role, host.arq.dxcall = arq.IRS, "W9SSJ"
    host.arq._enter_connected()
    status = status_at(1, long_cycle=False)
    for _ in range(4):
        host.on_rx_event(rxfront.Event(0.1, "packet", "p1",
                                       protocol=Protocol.PACTOR1,
                                       packet=(arq.P1_SPEED_LEVEL, status,
                                               FIELD, True)))
    assert arq.CS_SPEED_UP not in seam.words


@pytest.mark.parametrize("repeat_gear", [0, 3])
def test_the_old_climb_never_fires_on_repeats_alone(repeat_gear):
    """The negative control: `speed_up_after` spends ACCEPTED packets, and a
    peer that repeats supplies one. Whatever the new flag is set to, the
    clean-run count cannot climb -- and since round 24 the repeat behind the
    single acceptance ends the run outright (`arq._gear_cs`)."""
    host, seam = linked(repeat_gear=repeat_gear, speed_up_after=3)
    for _ in range(9):
        arrives(host)
    assert host.arq.rx_progress == 1
    assert host.arq._clean_run == 0


def test_three_accepted_packets_are_what_the_old_climb_spends():
    host, seam = linked(speed_up_after=3)
    for seq in (1, 2, 3):
        arrives(host, status=status_at(seq, long_cycle=False),
                field=bytes([0x41 + seq]))
    assert seam.words[-1] == arq.CS_SPEED_UP
    assert host.arq.rx_progress == 3


def test_the_silence_budget_already_caps_a_run_keyed_into_an_empty_channel():
    """No second cap is wired, because this one is already the bound.

    A repeat that decodes is a peer that is there, and the run is only worth
    ending once the peer stops: `_on_nak` charges `_silent_cycles` for every
    cycle nothing decoded in and gives up past `max_retries`, which is what
    ended the stalled arm after its 36th repeat. A `--p3-repeat-cap` counting
    the repeats themselves would end links a peer is still serving.
    """
    host, seam = linked(repeat_gear=3)
    for _ in range(36):
        arrives(host)
    assert host.arq.state is arq.State.CONNECTED
    cycles = 0
    while host.arq.state is arq.State.CONNECTED and cycles < 40:
        host.arq.on_cycle()
        cycles += 1
    assert host.arq.cfg.max_retries < cycles <= host.arq.cfg.max_retries + 3
