# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A CS4 train behind a CS4 connect answer asks for the first 100 Bd block again.

"CS4 dient als 'REQUEST'-CS fuer den ersten 100-Bd-Block, Bestaetigung erfolgt
mittels CS1 oder CS3" (pactor1-control-signals.md §4.3). WS8EOC-80 shows both
exits on file: 2026-09-20 22:59 held CS4 seven cycles and granted 0x59A, 19:42
took pkt#1 with CS1. On 2026-09-23 20:45 it held CS4 for 28 cycles and never
read pkt#1. Rising to 200 Bd on that train would send a packet it could not
read at 100 at twice the rate.
"""
import pytest

from hfmodem.shrike import pactor1, ptc, rxfront
from hfmodem.shrike.arq import State


class Peer:
    def __init__(self):
        self.frames = []

    def send_p1_packet(self, payload, baud, count, **kwargs):
        self.frames.append((count, baud, payload))

    def __getattr__(self, name):
        return lambda *a, **kw: None


def _cs(index):
    return rxfront.Event(0, "cs", "", protocol="PACTOR-1", cs=index)


def _grant():
    return rxfront.Event(0, "unassigned", "grant", protocol="PACTOR-1",
                         spare=pactor1.CS_59A)


def _held_cs4(cycles):
    peer = Peer()
    host = ptc.PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(_cs(pactor1.CS_SPEED))
    host.arq.on_cycle()
    assert host.arq.state is State.CONNECTED and host.p1_baud == 100
    first = peer.frames[-1]
    assert first[:2] == (1, 100)
    for _ in range(cycles):
        host.on_rx_event(_cs(pactor1.CS_SPEED))
        host.arq.on_cycle()
    return host, peer, first


@pytest.mark.parametrize("cycles", [7, 28])
def test_a_held_cs4_repeats_the_first_block_at_100(cycles):
    host, peer, first = _held_cs4(cycles)
    assert host.p1_baud == 100 and not host._p1_speedup_pending
    assert set(peer.frames) == {first}
    assert not host._grant_taken


def test_the_grant_behind_the_train_takes_the_link_to_pactor3():
    host, peer, _ = _held_cs4(7)
    host.on_rx_event(_grant())
    assert host._grant_taken and host.arq.entry_pending


def test_cs1_behind_the_train_acknowledges_the_first_block():
    host, peer, _ = _held_cs4(7)
    host.stay_in_pactor1 = True
    host.arq.on_host_data(b"x")
    host.on_rx_event(_cs(pactor1.CS_ACK_A))
    host.arq.on_cycle()
    assert host.p1_baud == 100
    assert peer.frames[-1] == (2, 100, b"x")
