# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""KB5LZK-30, 2026-09-23 13:40: three faults behind one dead changeover.

As IRS we keyed CS4 twice and the peer obeyed, its next packets at 100 Bd. This
end never learned the rate, so our changeover packet went out at 200 thirty
times to a station holding CS2 for it, and the four-request fallback that would
have geared it down was armed only by the connect answer and the speed-up.
Before that, a bit-6 invitation whose placement was refused stayed pending, and
we keyed an empty break-in into the greeting that followed.
"""
from hfmodem.shrike import pactor1, ptc, rxfront
from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN, IRS, ISS, REFUSED,
                                ArqConfig, ArqIO, P1_SPEED_LEVEL, PactorArq)


class Peer:
    def __init__(self):
        self.cs, self.frames = [], []

    def send_p1_cs(self, index):
        self.cs.append(index)

    def send_p1_packet(self, payload, baud, count, **kwargs):
        self.frames.append((count, baud, payload))

    def send_p1_breakin(self, payload, baud, count, **kwargs):
        self.frames.append((count, baud, "BK", payload))

    def __getattr__(self, name):
        return lambda *a, **kw: None


def _cs(index):
    return rxfront.Event(0, "cs", "", protocol="PACTOR-1", cs=index)


def _packet(status, baud, payload=b""):
    return rxfront.Event(0, "packet", "", protocol="PACTOR-1",
                         packet=(0, status, payload, True), baud=baud)


def irs_at_200():
    peer = Peer()
    host = ptc.PtcHost(peer=peer, mycall="W9SSJ")
    host.stay_in_pactor1 = True
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.arq.on_cycle()
    host.on_rx_event(_cs(pactor1.CS_ACK_A))
    host.arq.on_cycle()
    host.on_rx_event(_cs(pactor1.CS_CHANGEOVER))
    host.on_rx_event(_packet(0x00, 200))
    host.arq.on_cycle()
    assert host.arq.role is IRS and host.p1_baud == 200
    return host, peer


def test_a_peer_that_obeyed_cs4_gets_our_changeover_at_100():
    host, peer = irs_at_200()
    for _ in range(2):
        host.arq.on_rx_packet(0, b"", 0x01, False, protocol="PACTOR-1")
        host.arq.on_cycle()
    assert pactor1.CS_SPEED in peer.cs and host.p1_baud == 200
    host.arq.on_host_data(b"FF\r" * 5)
    host.arq.on_host_breakin()
    host.on_rx_event(_packet(0x01, 100, b"hello"))
    assert host.p1_baud == 100
    host.arq.on_cycle()
    assert host.arq.role is ISS
    assert peer.frames[-1][1:3] == (100, "BK")
    assert len(peer.frames[-1][3]) == pactor1.BREAKIN_FIELD[100]


def test_a_changeover_held_on_cs2_at_200_falls_back_to_100():
    host, peer = irs_at_200()
    host.arq.on_host_data(bytes(range(65, 95)))
    host.arq.on_host_breakin()
    host.on_rx_event(_packet(0x01, 200, b"hello"))
    host.arq.on_cycle()
    assert peer.frames[-1][1:3] == (200, "BK")
    for _ in range(ptc.P1_HISPEED_RETRIES):
        host.on_rx_event(_cs(pactor1.CS_ACK_B))
        host.arq.on_cycle()
    assert host.p1_baud == 100
    count, baud, kind, field = peer.frames[-1]
    assert (count, baud, kind) == (0, 100, "BK")
    assert field == bytes(range(65, 65 + pactor1.BREAKIN_FIELD[100]))


class _Air(ArqIO):
    def __init__(self):
        self.packets, self.cs, self.lines = [], [], []
        self.refuse = False

    def send_packet(self, sl, payload, status, breakin=False):
        if self.refuse and breakin:
            return REFUSED
        self.packets.append((status, breakin))

    def send_cs(self, cs_index):
        self.cs.append(cs_index)

    def upgrade(self, payload_waiting):
        return False

    def log(self, msg):
        self.lines.append(msg)


def _invited_and_refused():
    air = _Air()
    a = PactorArq(air, ArqConfig(max_retries=4))
    a.on_host_connect("W9SSJ", "KB5LZK")
    a.on_rx_cs(CS_ACK)
    a.on_cycle()
    a.on_rx_cs(CS_BREAKIN)
    assert a.role == IRS
    air.refuse = True
    a.on_rx_packet(P1_SPEED_LEVEL, b"", 0x41, True)
    a.on_cycle()
    air.refuse = False
    assert any("unkeyed changeover refused" in m for m in air.lines)
    assert a.role == IRS and not a._breakin_pending
    return air, a


def test_a_refused_invitation_keys_no_break_in_into_the_greeting():
    air, a = _invited_and_refused()
    a.on_rx_packet(P1_SPEED_LEVEL, b"[RMS Trimode", 0x02, True)
    a.on_cycle()
    assert a.role == IRS
    assert not any(brk for _, brk in air.packets), air.packets
    assert air.cs[-1] == CS_ACK


def test_a_refused_invitation_asked_again_is_answered():
    air, a = _invited_and_refused()
    a.on_rx_packet(P1_SPEED_LEVEL, b"", 0x42, True)
    a.on_cycle()
    assert a.role == ISS
    assert air.packets and air.packets[-1][1], air.packets
