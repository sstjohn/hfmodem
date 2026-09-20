# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The two halves of the PTC dialect, driven against each other.

`hfhost.PtcClient` is the HOST of manual 10.9.3 and `hfmodem.shrike.PtcHost`
the MODEM of 10.9.4. Each has its own suite and neither had ever been pointed at
the other, which is how they came to sit in opposite counter regimes and still
pass. This runs one whole session through the pair -- JHOST4, `I`, a `G` poll on
channel 0 and on the PACTOR channel, `L`, `C` against the SimPeer, data both
ways, `D`, `JHOST0` -- with a CRC error injected in each direction and the
header byte carried inside the data.

The assertions are the manual's, not today's emulation's, so they stand whatever
the server half does next.
"""
from __future__ import annotations

import pytest

from hfhost.ptc import (LINK, MSG, OK, LinkStatus, Packet, PtcClient, PtcError,
                        unstuff)
from hfmodem.shrike.ptc import PtcHost, SimPeer

CONNECTED, DISCONNECTED = 4, 0          # the L vector's state field [ptc.py:179]

GREETING = b"HI \xaa\xaa THERE"         # the far end's first words, stuffed on the wire
PAYLOAD = b"to the modem \xaa\xaa and out"


class Wire:
    """The pty `serve()` opens, minus the pty.

    `PtcHost.feed` is synchronous, so the modem's one reaction is on the wire
    before the client looks for it -- which is the protocol's own shape, not a
    shortcut. `tick` is what the select loop's timer does; nothing else drives
    the link, because a modem never speaks unasked (10.9.2).
    """

    def __init__(self, host: PtcHost) -> None:
        self.host = host
        self.sent: list[bytes] = []
        self.flip_out: int | None = None
        self.flip_in: int | None = None
        self._rx = bytearray(host.open())

    def write(self, data: bytes) -> None:
        self.sent.append(bytes(data))
        self._rx += self._corrupt(self.host.feed(self._corrupt(data, "flip_out")),
                                  "flip_in")

    def read(self, n: int) -> bytes:
        chunk, self._rx[:] = bytes(self._rx[:n]), self._rx[n:]
        return chunk

    def close(self) -> None:
        pass

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.host.tick()

    def _corrupt(self, data: bytes, which: str) -> bytes:
        at = getattr(self, which)
        if at is None or not data:
            return data
        setattr(self, which, None)
        spoiled = bytearray(data)
        spoiled[at] ^= 0x01             # a low bit: never makes or unmakes an AA
        return bytes(spoiled)


class Bench:
    def __init__(self, **peer_kw) -> None:
        self.peer = SimPeer(**peer_kw)
        self.host = PtcHost(self.peer, mycall="N0CALL")
        self.peer.attach(self.host)
        self.wire = Wire(self.host)
        self.client = PtcClient(self.wire, channel=self.host.ptchn)

    def open_hostmode(self) -> None:
        self.client.command("MYcall N0CALL", timeout=0.2)
        self.client.enter_hostmode(timeout=0.2)
        assert self.host.hostmode

    def until(self, done, ticks: int = 60) -> None:
        for _ in range(ticks):
            if done():
                return
            self.wire.tick()
        raise AssertionError("the link never got there")

    def drain(self, ticks: int = 60) -> tuple[list[str], bytes]:
        """G-poll the PACTOR channel until it goes quiet [10.4.4]."""
        events, data = [], bytearray()
        for _ in range(ticks):
            resp = self.client.ask("G")
            assert resp.channel == self.host.ptchn
            if resp.code == OK:
                self.wire.tick()
                if not self.peer.sending:
                    return events, bytes(data)
                continue
            (events.append(resp.text) if resp.code == LINK
             else data.extend(resp.data))
        raise AssertionError("the channel never went quiet")


@pytest.fixture
def bench() -> Bench:
    return Bench(greeting=GREETING, reply=lambda blob: blob.upper())


def flags_of(frame: bytes) -> int:
    return unstuff(frame[2:])[1]


def test_a_whole_session_runs_through_both_halves(bench):
    client, host = bench.client, bench.host
    bench.open_hostmode()

    assert client.ask("I W1AW").code == OK
    assert client.ask("I").text == "W1AW"               # read-back [10.4.5]
    client.ask("I N0CALL")

    assert client.ask("G", channel=0).channel == 0      # the poll is answered...
    assert client.ask("G").code == OK                   # ...and the PACTOR channel is idle

    assert client.status().state == DISCONNECTED        # [10.4.8]

    assert client.ask("C N0DX").code == OK
    bench.until(lambda: client.status().state == CONNECTED)

    events, greeting = bench.drain()
    assert any("N0DX" in line for line in events), events
    assert greeting == GREETING                         # 10.9.2 stuffing, modem -> host

    assert client.exchange(Packet(host.ptchn, False, PAYLOAD)).code == OK
    bench.until(lambda: bytes(bench.peer.received) == PAYLOAD)
    assert b"\xaa\x00\xaa\x00" in bench.wire.sent[-1], "the data frame was not stuffed"

    _, echoed = bench.drain()
    assert echoed.startswith(PAYLOAD.upper())

    assert client.ask("D").code == OK
    bench.until(lambda: client.status().state == DISCONNECTED)
    events, _ = bench.drain()
    assert any("DISCONNECTED" in line for line in events), events

    client.leave_hostmode()
    assert not host.hostmode
    assert "PTC-IIIusb" in client.command("VERsion", timeout=0.2)


def test_only_the_first_packet_ignores_the_counter(bench):
    """Bit 6 makes every packet an ACK-condition, so a repeat would be executed
    twice; the manual asks for it on the first packet after a hostmode start and
    on no other [10.9.3, 10.9.6]."""
    bench.open_hostmode()
    for text in ("I", "G", "L 4", "I"):
        bench.client.ask(text)

    flags = [flags_of(frame) for frame in bench.wire.sent if frame[:2] == b"\xaa\xaa"]
    assert [bool(f & 0x40) for f in flags] == [True, False, False, False]
    assert [(f >> 7) & 1 for f in flags] == [1, 0, 1, 0], "the counter must invert"


def test_a_corrupt_packet_draws_the_request_packet_and_a_repeat(bench):
    """The modem's NACK reaction is `AA AA AA 55`; the host's is the buffered
    packet again, request bit unchanged [10.9.3, 10.9.4]."""
    bench.open_hostmode()
    bench.client.ask("I")
    before = bench.client.repeats

    bench.wire.flip_out = 6                             # inside the command text
    assert bench.client.ask("I").text == "N0CALL"
    assert bench.client.deframer.requests == 1
    assert bench.client.repeats == before + 1
    assert bench.wire.sent[-1] == bench.wire.sent[-2], "the repeat was not verbatim"


def test_a_corrupt_reply_is_recovered_by_the_watchdog(bench):
    """No reaction within the timeout is a NACK too, and the modem answers the
    repeat from its buffer rather than running the command again [10.9.3/4]."""
    bench.open_hostmode()
    bench.client.ask("I W1AW")
    before = bench.client.repeats

    bench.wire.flip_in = 5
    assert bench.client.ask("I").text == "W1AW"
    assert bench.client.repeats == before + 1
    assert bench.client.deframer.crc_errors == 1


def test_an_empty_body_never_reaches_the_wire(bench):
    """The length byte is len-1 [10.6], so a zero-byte body claims one byte and
    sends none: the modem then waits for a byte that never comes."""
    bench.open_hostmode()
    with pytest.raises(PtcError):
        Packet(bench.host.ptchn, True, b"").encode()
    assert bench.client.ask("I").code == MSG            # the link still runs


def test_hostmode_re_entry_runs_a_repeated_command(bench):
    """After a hostmode start the modem's request bit is undefined, so the first
    packet is an ACK-condition whatever its counter -- even a byte-identical
    repeat of the one that closed the last session [10.9.6]."""
    bench.open_hostmode()
    jhost0 = Packet(0, True, b"JHOST0", counter=1).encode()
    bench.wire.write(Packet(bench.host.ptchn, True, b"I", counter=0).encode())
    bench.wire.write(jhost0)
    assert not bench.host.hostmode

    bench.client.command("JHOST4", timeout=0.2)
    assert bench.host.hostmode
    bench.wire.write(jhost0)
    assert not bench.host.hostmode, "the buffered reply was replayed instead"


def test_the_l_argument_names_the_channel(bench):
    """"Parameter: X 0...31, channel" [10.4.8]: `L` reports the channel its
    argument names, whichever channel the packet asking arrives on."""
    client, host = bench.client, bench.host
    bench.open_hostmode()
    client.ask("C N0DX")
    bench.until(lambda: client.status().state == CONNECTED)

    asked_on_zero = client.ask(f"L {host.ptchn}", channel=0)
    assert LinkStatus.parse(asked_on_zero.text) == client.status()


def test_d_twice_breaks_the_link_immediately(bench):
    """"If the Disconnect command is given twice, one after the other, then the
    link is broken immediately (corresponding to DD in PACTOR)" [10.4.2]."""
    client = bench.client
    bench.open_hostmode()
    client.ask("C N0DX")
    bench.until(lambda: client.status().state == CONNECTED)

    client.ask("D")
    client.ask("D")
    assert client.status().state == DISCONNECTED


def test_data_and_status_survive_a_full_length_block(bench):
    """256 bytes is the dialect's limit and every byte value has to make it
    through the stuffing [10.9.2, 10.9.3]."""
    client, host = bench.client, bench.host
    bench.open_hostmode()
    client.ask("C N0DX")
    bench.until(lambda: client.status().state == CONNECTED)
    bench.drain()

    block = bytes(range(256))
    assert client.exchange(Packet(host.ptchn, False, block)).code == OK
    bench.until(lambda: bytes(bench.peer.received) == block, ticks=400)
    assert client.status().unsent == 0

    _, echoed = bench.drain(ticks=400)
    assert echoed == block.upper()
