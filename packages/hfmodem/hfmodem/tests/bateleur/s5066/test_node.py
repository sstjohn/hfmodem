# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 5066 nodes against each other, through ``s5066.ether``."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur.s5066 import dpdu as D
from hfmodem.bateleur.s5066 import primitives as P
from hfmodem.tests.bateleur.s5066.ether import Client, Ether
from hfmodem.bateleur.s5066.node import Node, Timing

TIMING = Timing(1200, lead=0.6, tail=0.6)
A, B, C = D.address("0.0.0.1"), D.address("0.0.0.2"), D.address("0.0.0.3")


def _data(n, seed=1):
    return bytes(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8))


def _pair(**kw):
    a, b = Node(A, TIMING, **kw), Node(B, TIMING, **kw)
    return a, b, Client(a), Client(b)


def test_a_soft_link_is_made_used_and_broken():
    """A.3.2.1: queued ARQ data makes the link, goes over it, and the
    link is broken once the queue has stood empty."""
    a, b, ca, cb = _pair()
    assert ca.bind(6, confirm=P.CONFIRM_NODE).type == P.BIND_ACCEPTED
    cb.bind(6)
    msg = _data(5000)
    for i in range(0, len(msg), 1000):
        ca.unidata(B, 6, msg[i:i + 1000])
    et = Ether([a, b])
    et.run(600, until=lambda: len(ca.of(P.UNIDATA_REQUEST_CONFIRM)) == 5)
    assert cb.received() == msg
    assert all(p["src"] == A and p["src_sap"] == 6 and p["mode"] == P.MODE_ARQ
               for p in cb.of(P.UNIDATA_INDICATION))
    et.run(60)
    assert not a.cas.links and not b.cas.links


@pytest.mark.parametrize("ber", [1e-4, 1e-3])
def test_selective_repeat_through_bit_errors(ber):
    a, b, ca, cb = _pair(max_retries=24)
    ca.bind(6, confirm=P.CONFIRM_NODE)
    cb.bind(6)
    msg = _data(10000)
    for i in range(0, len(msg), 2000):
        ca.unidata(B, 6, msg[i:i + 2000])
    et = Ether([a, b], ber=ber, seed=3)
    et.run(3000, until=lambda: len(ca.of(P.UNIDATA_REQUEST_CONFIRM)) == 5)
    assert cb.received() == msg
    assert a.dts.counts["resent"] > 0


def test_u_pdus_arrive_in_the_order_sent_across_a_lossy_link():
    a, b, ca, cb = _pair(segment=100)
    ca.bind(6)
    cb.bind(6)
    parts = [bytes([i]) * 300 for i in range(20)]
    for p in parts:
        ca.unidata(B, 6, p)
    et = Ether([a, b], ber=1e-3, seed=5)
    et.run(3000, until=lambda: len(cb.of(P.UNIDATA_INDICATION)) == 20)
    assert [p["u_pdu"] for p in cb.of(P.UNIDATA_INDICATION)] == parts


def test_both_ways_at_once():
    a, b, ca, cb = _pair()
    ca.bind(6)
    cb.bind(6)
    ma, mb = _data(3000, 1), _data(3000, 2)
    ca.unidata(B, 6, ma[:1500])
    ca.unidata(B, 6, ma[1500:])
    cb.unidata(A, 6, mb[:1500])
    cb.unidata(A, 6, mb[1500:])
    et = Ether([a, b], ber=2e-4, seed=7)
    et.run(1500, until=lambda: len(cb.received()) == 3000 and len(ca.received()) == 3000)
    assert cb.received() == ma and ca.received() == mb


def test_broadcast_to_a_group_with_repeated_copies():
    """Non-ARQ (C.3.10): no link, no acknowledgement, the copies carrying
    the U_PDU through errors that would lose any one of them."""
    a = Node(A, TIMING)
    b = Node(B, TIMING, groups={D.address("0.0.9.9")})
    c = Node(C, TIMING)                          # not in the group
    ca, cb, cc = Client(a), Client(b), Client(c)
    ca.bind(2, mode=P.MODE_NON_ARQ, retransmissions=4)
    cb.bind(2, mode=P.MODE_NON_ARQ)
    cc.bind(2, mode=P.MODE_NON_ARQ)
    msg = _data(1500)
    ca.unidata(D.address("0.0.9.9"), 2, msg, group=True)
    et = Ether([a, b, c], ber=5e-4, seed=11)
    et.run(120)
    assert cb.received() == msg and cc.received() == b""
    assert cb.of(P.UNIDATA_INDICATION)[0]["mode"] == P.MODE_NON_ARQ
    assert not a.cas.links and a.transmissions == 1


def test_broadcast_with_errors_is_delivered_marked():
    """A.2.1.9: a client of non-ARQ with errors gets what came, and where
    it was bad or missing."""
    a = Node(A, TIMING)
    b = Node(B, TIMING)
    ca, cb = Client(a), Client(b)
    ca.bind(2, mode=P.MODE_NON_ARQ)
    cb.bind(2, mode=P.MODE_NON_ARQ_ERRORS)
    msg = _data(2000)
    ca.unidata(B, 2, msg)
    et = Ether([a, b])

    def corrupt_payload(data):
        pdus = D.Parser().push(data)
        at = D.size(pdus[0]) + D.size(pdus[1]) - 5
        raw = bytearray(data)
        raw[at] ^= 1
        return bytes(raw)

    et._corrupt = corrupt_payload
    et.run(200)
    got = cb.of(P.UNIDATA_INDICATION)
    assert len(got) == 1 and got[0]["mode"] == P.MODE_NON_ARQ_ERRORS
    bad = got[0]["errored"] + got[0]["missing"]
    assert bad
    good = np.ones(len(msg), bool)
    for pos, size in bad:
        good[pos:pos + size] = False
    u = np.frombuffer(got[0]["u_pdu"], np.uint8)
    assert np.array_equal(u[good], np.frombuffer(msg, np.uint8)[good])


def test_expedited_data_overtakes_the_queue():
    a, b, ca, cb = _pair()
    ca.bind(6)
    cb.bind(6)
    bulk = _data(8000)
    for i in range(0, len(bulk), 2000):
        ca.unidata(B, 6, bulk[i:i + 2000])
    et = Ether([a, b])
    et.run(3)                                    # the link is made, the bulk queued
    ca.unidata(B, 6, b"FLASH", expedited=True)
    et.run(600, until=lambda: len(cb.received()) >= 8005)
    order = [(p.type, p["u_pdu"][:5]) for p in cb.got if p.type in
             (P.UNIDATA_INDICATION, P.EXPEDITED_UNIDATA_INDICATION)]
    assert (P.EXPEDITED_UNIDATA_INDICATION, b"FLASH") in order
    assert order.index((P.EXPEDITED_UNIDATA_INDICATION, b"FLASH")) < len(order) - 1


def test_client_delivery_confirmation_and_an_unbound_sap():
    """A.3.1.2-3: the far sublayer confirms delivery to the client, or says
    no client is bound."""
    a, b, ca, cb = _pair()
    ca.bind(6, confirm=P.CONFIRM_CLIENT)
    cb.bind(6)
    ca.unidata(B, 6, b"to a client")
    ca.unidata(B, 9, b"to no one")
    et = Ether([a, b])
    et.run(300, until=lambda: len(ca.of(P.UNIDATA_REQUEST_CONFIRM))
           + len(ca.of(P.UNIDATA_REQUEST_REJECTED)) == 2)
    assert [p["u_pdu"] for p in ca.of(P.UNIDATA_REQUEST_CONFIRM)] == [b"to a client"]
    rej = ca.of(P.UNIDATA_REQUEST_REJECTED)
    assert rej[0]["u_pdu"] == b"to no one" and rej[0]["reason"] == P.REJECT_SAP_UNBOUND


def test_binding_rules():
    a = Node(A, TIMING)
    c1, c2 = Client(a), Client(a)
    assert c1.bind(6).type == P.BIND_ACCEPTED
    r = c2.bind(6)
    assert r.type == P.BIND_REJECTED and r["reason"] == P.BIND_SAP_TAKEN
    assert c1.of(P.BIND_ACCEPTED)[0]["mtu"] == 2048
    c1.unidata(B, 6, bytes(3000))
    assert c1.got[-1].type == P.UNIDATA_REQUEST_REJECTED
    assert c1.got[-1]["reason"] == P.REJECT_MTU


def test_ttl_expires_undelivered_data():
    a, b, ca, cb = _pair()
    ca.bind(6, confirm=P.CONFIRM_NODE)
    cb.bind(6)
    ca.unidata(B, 6, b"stale", ttl=10)
    et = Ether([a, b], deaf=[(A, B)])            # B never hears A
    et.run(60)
    rej = ca.of(P.UNIDATA_REQUEST_REJECTED)
    assert rej and rej[0]["reason"] in (P.REJECT_TTL, P.REJECT_NOT_RESPONDING)


def test_an_unanswered_link_request_gives_up():
    a, b, ca, cb = _pair()
    a.cas.tries = b.cas.tries = 2
    ca.bind(6, confirm=P.CONFIRM_NODE)
    ca.unidata(B, 6, b"anyone?")
    et = Ether([a, b], deaf=[(A, B)])
    et.run(300)
    assert not a.cas.links
    assert ca.of(P.UNIDATA_REQUEST_REJECTED)
    assert [x for x in et.log if x[1] == A][:2] and len([x for x in et.log if x[1] == A]) == 2


def test_a_link_lost_to_silence_is_declared_lost():
    a, b, ca, cb = _pair(max_retries=2)
    ca.bind(6, confirm=P.CONFIRM_NODE)
    cb.bind(6)
    ca.unidata(B, 6, _data(3000))
    et = Ether([a, b])
    et.run(20, until=lambda: a.cas.made(B))
    et.deaf = {frozenset((A, B))}                # the path closes
    et.run(600, until=lambda: not a.cas.links)
    assert not a.cas.links
    assert ca.of(P.UNIDATA_REQUEST_REJECTED)


def test_hard_link_type_2_needs_the_far_client():
    a, b, ca, cb = _pair()
    ca.bind(6, rank=5)
    cb.bind(9)
    ca.send(P.prim(P.HARD_LINK_ESTABLISH, link_type=2, link_priority=1, remote_sap=9, remote=B))
    et = Ether([a, b])
    et.run(60, until=lambda: cb.of(P.HARD_LINK_INDICATION))
    ind = cb.of(P.HARD_LINK_INDICATION)[0]
    assert ind["remote"] == A and ind["link_type"] == 2
    cb.send(P.prim(P.HARD_LINK_ACCEPT, link_type=2, link_priority=1, remote_sap=6, remote=A))
    et.run(60, until=lambda: ca.of(P.HARD_LINK_ESTABLISHED))
    assert ca.of(P.HARD_LINK_ESTABLISHED)[0]["remote"] == B
    et.run(60)
    assert a.cas.made(B)                         # held, though idle
    ca.send(P.prim(P.HARD_LINK_TERMINATE, remote=B))
    et.run(120, until=lambda: not a.cas.links and not b.cas.links)
    assert ca.of(P.HARD_LINK_TERMINATED) and cb.of(P.HARD_LINK_TERMINATED)
    assert not a.cas.links


def test_hard_link_type_2_to_an_unbound_sap_is_rejected():
    a, b, ca, cb = _pair()
    ca.bind(6)
    ca.send(P.prim(P.HARD_LINK_ESTABLISH, link_type=2, link_priority=0, remote_sap=12,
                   remote=B))
    et = Ether([a, b])
    et.run(120, until=lambda: ca.of(P.HARD_LINK_REJECTED))
    assert ca.of(P.HARD_LINK_REJECTED)[0]["reason"] == P.HL_SAP_UNBOUND


def test_full_reset_resynchronises():
    """C.6.5: a FULL RESET puts both ARQ machines back to their start, and
    data flows again after it."""
    a, b, ca, cb = _pair()
    ca.bind(6)
    cb.bind(6)
    ca.unidata(B, 6, b"first")
    et = Ether([a, b])
    et.run(60, until=lambda: cb.received() == b"first")
    a.dts.full_reset(B)
    et.run(30, until=lambda: a.dts.links[B].reset_out is None)
    assert a.dts.links[B].arq.tx_lwe == 0 and b.dts.links[A].arq.rx_lwe == 0
    ca.unidata(B, 6, b"second")
    et.run(120, until=lambda: cb.received() == b"firstsecond")
    assert cb.received() == b"firstsecond"


def test_link_requests_that_cross():
    """Both ends call at once: the link is made either way."""
    a, b, ca, cb = _pair()
    ca.bind(6)
    cb.bind(6)
    ca.unidata(B, 6, b"from a")
    cb.unidata(A, 6, b"from b")
    et = Ether([a, b], seed=4)
    et.run(600, until=lambda: cb.received() == b"from a" and ca.received() == b"from b")
    assert cb.received() == b"from a" and ca.received() == b"from b"


def test_three_nodes_take_turns():
    a, b, c = Node(A, TIMING), Node(B, TIMING), Node(C, TIMING)
    ca, cb, cc = Client(a), Client(b), Client(c)
    for cl in (ca, cb, cc):
        cl.bind(6)
    ca.unidata(B, 6, b"a to b")
    ca.unidata(C, 6, b"a to c")
    et = Ether([a, b, c])
    et.run(900, until=lambda: cb.received() == b"a to b" and cc.received() == b"a to c")
    assert cb.received() == b"a to b" and cc.received() == b"a to c"


def test_warning_for_data_on_no_connection():
    """C.3.12: DATA to a node that holds no connection draws a WARNING."""
    a, b = Node(A, TIMING), Node(B, TIMING)
    a.dts.connection_made(B)                     # A believes a link exists
    a.dts.submit(__import__("hfmodem.bateleur.s5066.dts", fromlist=["CPDU"]).CPDU(
        b"\x00stray", B))
    et = Ether([a, b])
    et.run(30, until=lambda: any(p.type == D.WARNING for p in a.heard))
    w = [p for p in a.heard if p.type == D.WARNING][0]
    assert w.reason == D.WARN_UNCONNECTED and w.rx_type == D.DATA_ONLY


def test_a_data_rate_change_on_a_made_link():
    """C.6.4: the master asks, the slave accepts, the master changes on the
    slave's acknowledged accept, and the exchange ends in a confirm; data
    goes on at the new rate."""
    a, b, ca, cb = _pair()
    ca.bind(6)
    cb.bind(6)
    changed = []
    a.on_rate = lambda remote, rate, il: changed.append(("a", remote, rate, il))
    b.on_rate = lambda remote, rate, il: changed.append(("b", remote, rate, il))
    ca.unidata(B, 6, b"first")
    et = Ether([a, b], ber=1e-4, seed=5)
    et.run(200, until=lambda: cb.received() == b"first")
    a.dts.request_rate(B, 2400, "long")
    et.run(300, until=lambda: changed and a.dts.links[B].drc is None
           and b.dts.links[A].drc is None)
    assert changed == [("a", B, 2400, "long")]
    assert a.timing.rate == 2400 and b.timing.rate == 1200
    msg = _data(2000, 4)
    ca.unidata(B, 6, msg)
    et.run(300, until=lambda: len(cb.received()) == 5 + len(msg))
    assert cb.received() == b"first" + msg


def test_a_rate_the_far_end_cannot_take_is_refused():
    a, b, ca, cb = _pair()
    ca.bind(6)
    cb.bind(6)
    b.dts.can_receive = lambda rate, il: rate <= 1200
    ca.unidata(B, 6, b"first")
    et = Ether([a, b])
    et.run(200, until=lambda: cb.received() == b"first")
    a.dts.request_rate(B, 2400, "short")
    et.run(300, until=lambda: a.dts.links[B].drc is None)
    assert a.timing.rate == 1200 and a.dts.links[B].drc is None
    ca.unidata(B, 6, b"second")
    et.run(200, until=lambda: cb.received() == b"firstsecond")
    assert cb.received() == b"firstsecond"


def test_a_clock_far_from_zero():
    """The node's clock may be monotonic, hours into the machine's uptime:
    a U_PDU submitted before its first tick lives its time to live from
    then, not from zero; the TTD on the air is UTC, whatever that clock."""
    a, b, ca, cb = _pair()
    ca.bind(6)
    cb.bind(6)
    ca.unidata(B, 6, b"late in the day")
    et = Ether([a, b])
    et.now = 5.0e6
    et.run(200, until=lambda: cb.received() == b"late in the day")
    assert cb.received() == b"late in the day"
