# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the link hands the host has to be the bytes the peer sent, once each.

`winlink.B2FSession.feed` is a pure byte-stream consumer with no resync: every
framing failure is fatal, and a run of bytes appearing twice inside a proposal
block fails `F>`'s checksum exactly as a run appearing once too few does. So the
two ways this layer can break a mail exchange without breaking a link are a
duplicate and a silent drop, and both were reachable.

THE DUPLICATE IS THE CHANGEOVER'S. `_reset_rx_seq` clears `_rx_seen` with the
counter on every path that hands the link over, deliberately: until something
has been delivered under the new numbering there is nothing for a repeat to be a
repeat OF, and demanding a particular counter is a guess that costs payload when
it is wrong. What that leaves is the ISS deposed before it heard its last packet
settled -- it keeps that packet (ardopcf's SaveQueueOnBreak is the same
behaviour on another modem) and puts it back on the air as the first packet of
its next stint, where the counter now proves nothing. Reproduced at the ARQ seam
on 2026-09-01, `FC EM ABC 100 50 0` delivered twice.

`besra.arq.session._is_replay` is the guard, and its reasoning carries across
whole: only ever asked about the first packet of a stint, and answered by the
payload bytes, because across a reversal the bytes are the only identity left.
Everything that is not byte-exact keeps the trade the reset was made for.

THE DROP IS THE HOST'S. `on_host_data` refuses a blob during teardown -- the
goodbye rides the packet that drains the buffer, so a host still writing
postpones it forever -- and refused it saying nothing at all. Two paths reach it
with a B2F session still owing a line: the hold budget expiring, and
`UNREPAIRED_BUDGET` calling `on_host_disconnect` from inside the receive that
will hand those very bytes up. The discard stays; what it now leaves is a line
in the log, so a stage that stopped advancing has a cause on the page.

Run: python -m pytest hfmodem/tests/shrike/test_bytestream.py
"""
from __future__ import annotations

import pytest

from hfmodem.shrike.spec import Protocol
from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN, IRS, ISS, P1_SPEED_LEVEL,
                                ArqConfig, ArqIO, PactorArq, State)

CFG = ArqConfig(max_retries=4)

#: One B2F proposal line, the shape the duplicate was reproduced on.
PROPOSAL = b"FC EM ABC 100 50 0\r"
GREETING = b"[RMS Trimode-1.3.]\r"


class _Air(ArqIO):
    def __init__(self):
        self.packets: list[tuple[int, bool]] = []
        self.cs: list[int] = []
        self.data = bytearray()
        self.lines: list[str] = []

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((status, breakin))

    def send_cs(self, cs_index):
        self.cs.append(cs_index)

    def send_p1_cs(self, index):
        self.cs.append(index)

    def upgrade(self, payload_waiting):
        return False

    def deliver(self, blob):
        self.data += blob

    def log(self, msg):
        self.lines.append(msg)


def _receiving() -> tuple[_Air, PactorArq]:
    """A linked IRS, reached the way a gateway puts us there: it broke in."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KB5LZK")
    a.on_rx_cs(CS_ACK)
    a.on_cycle()
    a.on_rx_cs(CS_BREAKIN)
    assert a.role == IRS
    return air, a


def _deposed(air: _Air, a: PactorArq) -> None:
    """Take the link off the peer mid-stint, the way `app_turns` does when the
    B2F layer has a line to send."""
    a.on_host_data(b"FS Y\r")
    a.on_host_breakin()
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 2, True)
    a.on_cycle()
    assert a.role == ISS and air.packets[-1][1]


def test_the_packet_the_peer_never_heard_settled_is_delivered_once():
    air, a = _receiving()
    a.on_rx_packet(P1_SPEED_LEVEL, GREETING, 1, True)
    a.on_cycle()
    _deposed(air, a)
    before = a.rx_progress
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 1, True, breakin=True)
    assert a.role == IRS
    assert bytes(air.data) == GREETING + PROPOSAL, bytes(air.data)
    assert a.rx_progress > before, "the replay was not acknowledged as progress"
    assert air.cs and air.cs[-1] == CS_ACK, air.cs


def test_new_bytes_across_the_same_changeover_are_delivered():
    """The trade `_reset_rx_seq` was made for is untouched: anything that is not
    byte-exact is taken at whatever counter it carries."""
    air, a = _receiving()
    a.on_rx_packet(P1_SPEED_LEVEL, GREETING, 1, True)
    a.on_cycle()
    _deposed(air, a)
    a.on_rx_packet(P1_SPEED_LEVEL, b"FS Y\r", 1, True, breakin=True)
    assert bytes(air.data) == GREETING + PROPOSAL + b"FS Y\r", bytes(air.data)


def test_the_peer_may_send_the_same_line_twice_in_one_stint():
    """The guard is the first packet of a stint and nothing else. Inside one the
    counter is the identity, so a peer with two identical lines to send gets both
    of them delivered."""
    air, a = _receiving()
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 1, True)
    a.on_cycle()
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 2, True)
    a.on_cycle()
    assert bytes(air.data) == PROPOSAL * 2, bytes(air.data)


def test_a_repeat_inside_a_stint_is_still_the_counters_job():
    air, a = _receiving()
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 1, True)
    a.on_cycle()
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 1, True)
    a.on_cycle()
    assert bytes(air.data) == PROPOSAL, bytes(air.data)


@pytest.mark.parametrize("sl,protocol", [
    (P1_SPEED_LEVEL, Protocol.PACTOR1),
    (P1_SPEED_LEVEL, Protocol.PACTOR2),
    (3, Protocol.PACTOR3),
])
def test_a_changed_coding_mode_is_not_a_replay(sl, protocol):
    air, a = _receiving()
    field = bytes.fromhex("1438842e2d04f3f0")
    a.on_rx_packet(sl, field, (1 << 2) | 1, True, protocol=protocol)
    assert bytes(air.data) == b"1dl6maa\r"
    a._reset_rx_seq()
    a.on_rx_packet(sl, field, (2 << 2) | 1, True, protocol=protocol)
    assert bytes(air.data) == b"1dl6maa\r1DL6MAA\r"


def test_a_replayed_first_packet_of_the_next_link_is_new_bytes():
    """Replay history belongs to one link."""
    air, a = _receiving()
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 1, True)
    a.on_cycle()
    a.on_host_abort()
    a.on_host_connect("W9SSJ", "KB5LZK")
    a.on_rx_cs(CS_ACK)
    a.on_cycle()
    a.on_rx_cs(CS_BREAKIN)
    a.on_rx_packet(P1_SPEED_LEVEL, PROPOSAL, 1, True)
    assert bytes(air.data) == PROPOSAL * 2, bytes(air.data)


def test_a_blob_discarded_during_the_teardown_says_so():
    air, a = _receiving()
    a.on_host_disconnect()
    assert a._qrt_pending
    a.on_host_data(b"FS +\r")
    assert not a._outbuf, "the discard was the point of the branch"
    assert any("FS +" not in m and "discard" in m for m in air.lines), air.lines[-3:]


def test_a_blob_discarded_before_the_link_is_up_says_so():
    air = _Air()
    a = PactorArq(air, CFG)
    assert a.state == State.DISCONNECTED
    a.on_host_data(b"FF\r")
    assert not a._outbuf
    assert any("discard" in m for m in air.lines), air.lines


def test_the_bytes_themselves_stay_out_of_the_log():
    """The line names what was lost and why. It does not put the host's traffic
    on the operator's screen, which is a separate decision made elsewhere."""
    air, a = _receiving()
    a.on_host_disconnect()
    a.on_host_data(b"; a passphrase would be here\r")
    assert not any("passphrase" in m for m in air.lines), air.lines[-3:]
