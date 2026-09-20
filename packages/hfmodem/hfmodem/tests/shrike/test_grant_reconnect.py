# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A grant belongs to one contact, including when a hostmode modem is reused."""
from hfmodem.shrike import pactor1, rxfront
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p3_offer import Keyed, cs_event


def test_a_second_contact_can_accept_its_own_grant():
    host = PtcHost(peer=Keyed(), mycall="W9SSJ")
    host.p1_grant_only = True
    grant = rxfront.Event(0.2, "unassigned", "0x59A",
                          protocol=Protocol.PACTOR1, spare=pactor1.CS_59A)
    for gateway in ("KB5LZK", "WS8EOC"):
        host.arq.on_host_connect(host.mycall, gateway)
        host.on_rx_event(cs_event(pactor1.CS_SPEED))
        host.tick()
        host.on_rx_event(grant)
        assert host.protocol == Protocol.PACTOR3, gateway
        assert host.arq.entry_pending, gateway
        host.arq.on_host_abort()
        assert host.protocol == Protocol.PACTOR1
