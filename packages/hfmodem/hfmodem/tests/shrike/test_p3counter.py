# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The IRS's PACTOR-3 codeword across successive acknowledgements.

`test_p1counter` pins the same rule at PACTOR-1 and this is its other half, because
the rule is one rule: `ptc.PtcHost._counter_cs_for` maps the FSM's logical ACK onto
CS1 for an even accepted counter and CS2 for an odd one, and `_p1_cs_for` returns
through it. Nothing below the mapping knows which renderer the codeword reaches.

WHAT THIS ASSERTS IS THE SEQUENCE, never a single word. A mapping that returned CS1
unconditionally satisfies every existing PACTOR-3 assertion in the suite:
`test_control_offset_follow` reads one `CS1 ACK` off one emission, and
`test_changeover_progression` reads `air.controls` at the ARQ, which is the LOGICAL
alphabet the parity mapping has not been applied to yet. Neither can see a constant.
So every case here walks a peer's counter and reads the physical words back through
`PtcHost`, where the mapping is in the loop.

The record it is cut from, all three at KB5LZK and WS8EOC on 40 and 80 m: 25 CS1 and
58 CS2 against counters 0,1,2,3 in `working/night-17-kb5lzk-pactor-40m-2026-08-22.log`,
15 and 10 in `working/night-01b-ws8eoc-pactor-40m-2026-08-22.log`, and 28 CS1 with no
CS2 at all in `working/pactor-level-air-0916/arms/pactor-p3-100-kb5lzk-80-sense-
20260918T011351Z/arm.log`, where all sixteen of the peer's decoded packets carry
`seq=0`. A held codeword against a held counter is the rule working, not failing --
which is why the count of each word proves nothing on its own and the walk does.

Run: python -m pytest packages/hfmodem/hfmodem/tests/shrike/test_p3counter.py
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import pactor1, rxfront, spec
from hfmodem.shrike.arq import CS_BREAKIN, IRS, ISS
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

P3_NAMES = {0: "CS1", 1: "CS2", 2: "CS3", 3: "CS4", 4: "CS5", 5: "CS6"}
P1_NAMES = {pactor1.CS_ACK_A: "CS1", pactor1.CS_ACK_B: "CS2",
            pactor1.CS_CHANGEOVER: "CS3", pactor1.CS_SPEED: "CS4"}

BANNER = ((0, b"RMS"), (1, b" Tri"), (2, b"mode"), (3, b" 1.4"), (0, b" KB5"))


class _Keyed:
    """A peer that records the codewords, and nothing else."""

    defer_p3_cs = False

    def __init__(self) -> None:
        self.cs: list[str] = []

    def send_cs(self, index: int) -> None:
        self.cs.append(P3_NAMES.get(index, str(index)))

    def send_p1_cs(self, index: int) -> None:
        self.cs.append(P1_NAMES.get(index, str(index)))

    def __getattr__(self, _name):
        return lambda *a, **kw: None


def _packet(counter: int, field: bytes, protocol: str, *, breakin: bool = False,
            sl: int = 1, **status) -> rxfront.Event:
    return rxfront.Event(0.0, "packet", "", protocol=protocol, breakin=breakin,
                         packet=(sl, spec.status_byte(counter, **status), field, True))


def _yielded_to(protocol: Protocol, *, speed_up: str = "hold") -> tuple[PtcHost, _Keyed]:
    """A call answered, and the gateway taking the channel to greet us.

    The gear seams are held off by default: `_gear_cs` and `_repeat_gear_cs` both
    substitute CS4 into the acknowledgement's own slot, which is a rung and not an
    alternation, and a test that let them fire would be reading the ladder.
    """
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.cfg.speed_up = speed_up
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(rxfront.Event(0.0, "cs", "CS1", protocol="PACTOR-1",
                                   cs=pactor1.CS_ACK_A))
    host.protocol = protocol
    host.stay_in_pactor1 = protocol is Protocol.PACTOR1
    host.arq.on_rx_cs(CS_BREAKIN)
    assert host.arq.role is IRS
    return host, peer


def _walk(host: PtcHost, protocol: str, packets, *, quiet: int = 0) -> None:
    for counter, field in packets:
        host.on_rx_event(_packet(counter, field, protocol,
                                 breakin=counter == 0 and not host.rcvd_total))
        host.arq.on_cycle()
        for _ in range(quiet):
            host.arq.on_cycle()


def test_every_new_counter_flips_the_codeword():
    """The peer's banner, and the word changes on each packet that advances it."""
    host, peer = _yielded_to(Protocol.PACTOR3)
    host.arq.on_cycle()
    _walk(host, "PACTOR-3", BANNER)
    assert peer.cs == ["CS2", "CS1", "CS2", "CS1", "CS2", "CS1"]
    assert host.rcvd_total == 19


def test_a_cycle_that_decoded_nothing_holds_the_word_and_the_next_packet_flips_it():
    """Silence is not an acknowledgement, and it does not spend one either."""
    host, peer = _yielded_to(Protocol.PACTOR3)
    host.arq.on_cycle()
    _walk(host, "PACTOR-3", BANNER, quiet=1)
    assert peer.cs == ["CS2", "CS1", "CS1", "CS2", "CS2", "CS1", "CS1",
                       "CS2", "CS2", "CS1", "CS1"]
    assert host.rcvd_total == 19


def test_a_held_counter_holds_the_codeword_and_the_link_delivers_once():
    """KB5LZK 80 m, 2026-09-18: sixteen decoded packets, every one of them seq=0.

    The arm keyed 28 CS1 and no CS2, which is the mapping answering the counter it
    was given. `_counter_cs_for` is unchanged since before the 2026-08-22 controls
    -- 443c3c10 renamed `_p3_cs_for` and gave PACTOR-1 the same body -- so a run of
    one word here dates nothing about this station. The peer's counter does.
    """
    host, peer = _yielded_to(Protocol.PACTOR3)
    host.arq.on_cycle()
    _walk(host, "PACTOR-3", [(0, b"RMS")] * 12)
    assert peer.cs == ["CS2"] + ["CS1"] * 12
    assert host.rcvd_total == 3


def test_a_second_changeover_restarts_the_alternation_at_cs1():
    """A mail session's whole shape: the peer greets, we answer, it takes it back.

    Both of the gateway's stints open at counter 0, so both open CS1 behind the CS2
    the waiting cycle carries, whatever word the stint before ended on.
    """
    host, peer = _yielded_to(Protocol.PACTOR3)
    _walk(host, "PACTOR-3", BANNER[:3])
    assert peer.cs == ["CS1", "CS2", "CS1"]

    host.arq.on_host_data(b"login please")
    host.on_rx_event(_packet(3, b" at t", "PACTOR-3", changeover_request=True))
    for _ in range(3):
        host.arq.on_cycle()
    assert host.arq.role is ISS

    mark = len(peer.cs)
    host.arq.on_rx_cs(CS_BREAKIN)
    assert host.arq.role is IRS
    host.arq.on_cycle()
    for counter, field in ((0, b"he Ark"), (1, b"ansas ")):
        host.on_rx_event(_packet(counter, field, "PACTOR-3", breakin=counter == 0))
        host.arq.on_cycle()
    assert peer.cs[mark:] == ["CS2", "CS1", "CS2"]


def test_pactor1_and_pactor3_key_the_same_alternation():
    """One rule, two renderers -- so PACTOR-1 cannot drift from PACTOR-3 unseen."""
    words = []
    for protocol, name in ((Protocol.PACTOR1, "PACTOR-1"),
                           (Protocol.PACTOR3, "PACTOR-3")):
        host, peer = _yielded_to(protocol)
        host.arq.on_cycle()
        _walk(host, name, BANNER)
        _walk(host, name, [(0, b" KB5")] * 3)
        words.append(peer.cs)
        assert host.rcvd_total == 19
    assert words[0] == words[1]


@pytest.mark.parametrize("protocol,name", [(Protocol.PACTOR1, "PACTOR-1"),
                                           (Protocol.PACTOR3, "PACTOR-3")])
def test_the_waiting_cycles_before_a_changeover_packet_ask_for_it(protocol, name):
    """CS2 until the counter-0 packet decodes: keying CS1 first confirms nothing."""
    host, peer = _yielded_to(protocol)
    for _ in range(4):
        host.arq.on_cycle()
    assert peer.cs == ["CS2"] * 4
    _walk(host, name, ((0, b"RMS"),))
    assert peer.cs[4:] == ["CS1"]
