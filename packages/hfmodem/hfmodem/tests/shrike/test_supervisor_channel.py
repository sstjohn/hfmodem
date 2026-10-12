# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 1990 supervisor channel, both halves: the announcement's level digit and
the function block, SB + code + parameters + SPACE, carried as text.

These serve the on-air sounding at KB5LZK-30 and VE3KPG-40: emit the digit and
the `1C A n SPACE` block and read for a codeword change, a CS3 refusal, or a
returned supervisor block; `docs/protocols/pactor/pactor-capability.md`
section 1 gives the mechanism.
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import compress, pactor1, rxfront, spec
from hfmodem.shrike.arq import CS_BREAKIN
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.tests.shrike.test_p3_offer import Keyed, acknowledge, cs_event

BLOCK = b"\x1c\x41\x33\x20"


def calling(*, digit: int = 1, schedule: tuple[str, ...] = (),
            every: int = 1) -> tuple[PtcHost, Keyed]:
    keyed = Keyed()
    host = PtcHost(peer=keyed, mycall="W9SSJ")
    host.level_digit, host.supervisor_schedule = digit, schedule
    host.supervisor_every = every
    host.p1_grant_only = True
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    return host, keyed


def acked(host: PtcHost, n: int) -> None:
    """`n` cycles, each acknowledging the packet the one before it keyed."""
    for i in range(n):
        acknowledge(host, (pactor1.CS_ACK_A, pactor1.CS_ACK_B)[i & 1])
        host.tick()


def test_digit_1_is_todays_announcement():
    _, keyed = calling()
    assert keyed.packets[0][1] == b"1w9ssj\r"


def test_digit_3_changes_exactly_one_byte():
    _, keyed = calling(digit=3)
    sent = keyed.packets[0][1]
    assert sent == b"3w9ssj\r"
    assert [i for i, (a, b) in enumerate(zip(sent, b"1w9ssj\r")) if a != b] == [0]


def test_the_block_is_sb_code_parameters_space():
    assert compress.supervisor_block("A 3") == BLOCK
    assert compress.supervisor_block("A3") == BLOCK
    assert compress.supervisor_block("B") == b"\x1cB "
    assert compress.transparent(b"\x1c") == b"\x1c\x7c"


def test_the_highest_function_code_is_underscore():
    assert compress.supervisor_block("_ 3") == b"\x1c_3 "


@pytest.mark.parametrize("text", ["", " 3", "A 3 4", "\x1c", "\x7f", "A\x1e",
                                  "`", "a", "a 3", "~"])
def test_what_is_not_a_function_block_is_refused(text):
    with pytest.raises(ValueError):
        compress.supervisor_block(text)


def test_the_block_rides_the_packet_after_the_announcement_unescaped():
    host, keyed = calling(schedule=("A 3",))
    acked(host, 1)
    assert keyed.packets[1][1] == BLOCK
    assert "supervisor tx: A 3 (cycle 1)" in host.log_lines


def test_one_block_every_n_acknowledgements():
    host, keyed = calling(schedule=("A 3", "B"), every=3)
    acked(host, 7)
    fields = [field for _, field in keyed.packets]
    assert fields.index(BLOCK) == 3 and fields.index(b"\x1cB ") == 6
    assert [line for line in host.log_lines if line.startswith("supervisor tx")] == [
        "supervisor tx: A 3 (cycle 3)", "supervisor tx: B (cycle 6)"]


def test_a_block_split_across_two_packets_still_parses():
    host, keyed = calling(schedule=("A 3",))
    host.arq.on_host_data(b"123456")
    acked(host, 3)
    fields = [field for _, field in keyed.packets[1:3]]
    assert fields == [b"123456\x1cA", b"3 "]
    si = compress.Supervisor()
    assert b"".join(si.feed(f) for f in fields) == b"123456"
    assert si.blocks == [("A", b"3")]


def test_a_three_block_schedule_crosses_the_loopback_as_supervisor_events():
    peer = SimPeer()
    host = PtcHost(peer, mycall="W9SSJ")
    host.supervisor_schedule, host.supervisor_every = ("A 3", "B", "C"), 2
    events, wire = [], bytearray()
    peer._far.io.supervisor = lambda code, params: events.append((code, params))
    send = peer.send_packet

    def keyed(sl, payload, status, breakin=False):
        wire.extend(payload)
        return send(sl, payload, status, breakin)
    peer.send_packet = keyed
    host.arq.on_host_connect("W9SSJ", "K7ABC")
    for _ in range(12):
        host.tick()
    assert events == [("A", b"3"), ("B", b""), ("C", b"")]
    assert BLOCK in wire and b"\x1cB " in wire and b"\x1cC " in wire
    assert bytes(peer.received) == b""


def _packet(counter: int, field: bytes):
    return rxfront.Event(0.0, "packet", "", protocol="PACTOR-1",
                         packet=(0, spec.status_byte(counter), field, True))


def test_a_peers_block_is_logged_where_rx_data_is_and_kept_out_of_the_stream():
    peer = Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(cs_event(pactor1.CS_ACK_A))
    host.arq.on_rx_cs(CS_BREAKIN)
    for counter, field in enumerate((b"hi\x1cA", b"3 there", b"\x1c\x7e!")):
        host.on_rx_event(_packet(counter, field))
        host.arq.on_cycle()
    lines = [line for line in host.log_lines
             if line.startswith(("rx data", "supervisor"))]
    assert lines == ["rx data 2B: 'hi'", "rx data 5B: 'there'",
                     "supervisor: A 3", "rx data 2B: '\\x1e!'"]
    assert bytes(host.channel(host.ptchn).rx) == b"hithere\x1e!"


def idling(every: int, mycall: str = "W9SSJ", baud_cs: int = pactor1.CS_SPEED):
    """A linked ISS with an empty buffer after the announcement is acknowledged."""
    keyed = Keyed()
    host = PtcHost(peer=keyed, mycall=mycall)
    host.idle_id_every, host.p1_grant_only = every, True
    host.arq.on_host_connect(mycall, "KB5LZK")
    host.on_rx_event(cs_event(baud_cs))
    host.tick()
    return host, keyed


def test_the_idle_id_is_function_b_in_every_nth_idle_packet():
    host, keyed = idling(3)
    acked(host, 6)
    fields = [field for _, field in keyed.packets[1:]]
    assert fields == [b"", b"", b"\x1cBW9SSJ ", b"", b"", b"\x1cBW9SSJ "]


def test_off_leaves_idle_packets_as_they_were():
    assert PtcHost(peer=Keyed()).idle_id_every == 0
    host, keyed = idling(0)
    acked(host, 6)
    assert [f for _, f in keyed.packets[1:]] == [b""] * 6


def test_a_repeat_carries_the_same_id():
    host, keyed = idling(1)
    acked(host, 1)
    host.tick()
    host.tick()
    assert [f for _, f in keyed.packets[1:]] == [b"\x1cBW9SSJ "] * 3


def test_a_call_too_long_for_the_field_is_not_sent():
    host, keyed = idling(1, mycall="VE3KPG")
    acked(host, 3)
    assert host.p1_baud == 100
    assert [f for _, f in keyed.packets[1:]] == [b""] * 3


def test_the_loopback_hears_our_call_and_delivers_no_data():
    peer = SimPeer()
    host = PtcHost(peer, mycall="W9SSJ")
    host.idle_id_every, host.p1_grant_only = 2, True
    heard = []
    peer._far.io.supervisor = lambda code, params: heard.append((code, params))
    host.arq.on_host_connect("W9SSJ", "K7ABC")
    for _ in range(8):
        host.tick()
    assert heard and set(heard) == {("B", b"W9SSJ")}
    assert bytes(peer.received) == b""


def test_a_peers_b_is_shown_as_the_ptc_shows_it():
    host = PtcHost(peer=Keyed(), mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(cs_event(pactor1.CS_ACK_A))
    host.arq.on_rx_cs(CS_BREAKIN)
    host.on_rx_event(_packet(0, b"\x1cBKB5LZK "))
    assert host.log_lines[-2:] == ["supervisor: B KB5LZK", "<<ID: KB5LZK>>"]
    assert not host.channel(host.ptchn).rx
