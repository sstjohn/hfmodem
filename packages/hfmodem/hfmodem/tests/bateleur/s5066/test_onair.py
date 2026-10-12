# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 5066 nodes on the bateleur modem, through ``AudioEther``: the
waveform, the channel and the receivers in the loop."""
from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from hfmodem.bateleur.modem import Mode
from hfmodem.bateleur.s5066 import primitives as P
from hfmodem.tests.bateleur.s5066.ether import AudioEther, Client
from hfmodem.bateleur.s5066.node import Node, Timing
from hfmodem.bateleur.s5066.onair import ModeTiming, mode_for, rate_of
from hfmodem.bateleur.s5066.server import SisServer


def _data(n, seed=1):
    return bytes(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8))


def test_modes_and_the_rates_a_drc_names():
    assert rate_of(Mode("110", "1200L")) == (1200, "long")
    assert rate_of(Mode("110", "600Z")) == (600, "none")
    assert rate_of(Mode("4285", "2400U")) == (2400, "none")
    assert rate_of(Mode("4539", "6400S")) == (6400, "short")
    assert mode_for(2400, "short", "110") == Mode("110", "2400S")
    assert mode_for(4800, "none", "110") == Mode("110", "4800N")
    # Past 188-110's rates, its receivers' neighbour takes over.
    assert mode_for(6400, "long", "110") == Mode("4539", "6400L")
    # 4285's receiver is told its mode: no crossing to another standard.
    assert mode_for(6400, "long", "4285") is None


def test_a_mode_timing_is_the_modem_s():
    m = Mode("110", "1200L")
    t = ModeTiming(m)
    for n in (0, 100, 1000):
        assert t.airtime(n) == pytest.approx(m.seconds(n))
    assert t.airtime(t.most_bytes(30.0)) <= 30.0
    assert t.lead == pytest.approx(4.8)
    assert t.at_rate(2400, "short").mode == Mode("110", "2400S")


def test_arq_over_the_waveform():
    et = AudioEther(snr_db=15, seed=1)
    a = et.station("0.0.0.1", "110 1200S")
    b = et.station("0.0.0.2", "110 1200S")
    ca, cb = Client(a.node), Client(b.node)
    ca.bind(6, confirm=P.CONFIRM_NODE)
    cb.bind(6)
    msg = _data(2000)
    ca.unidata(2, 6, msg)
    et.run(120, until=lambda: ca.of(P.UNIDATA_REQUEST_CONFIRM))
    assert cb.received() == msg
    assert a.node.dts.counts["resent"] == 0
    assert len(et.log) == 4            # request, accept, the data, its ACK


def test_a_rate_change_retunes_both_modems():
    et = AudioEther(snr_db=20, seed=2)
    a = et.station("0.0.0.1", "110 600S")
    b = et.station("0.0.0.2", "110 600S")
    ca, cb = Client(a.node), Client(b.node)
    ca.bind(6)
    cb.bind(6)
    ca.unidata(2, 6, b"hello")
    et.run(60, until=lambda: cb.received() == b"hello")
    a.node.dts.request_rate(2, 2400, "short", equal=True)
    et.run(120, until=lambda: a.node.dts.links[2].drc is None
           and b.node.dts.links[1].drc is None and a.modem.mode.name == "2400S"
           and b.modem.mode.name == "2400S")
    assert a.modem.mode == b.modem.mode == Mode("110", "2400S")
    msg = _data(2000, 2)
    ca.unidata(2, 6, msg)
    assert not ca.of(P.UNIDATA_REQUEST_REJECTED)
    et.run(120, until=lambda: cb.received() == b"hello" + msg)
    assert cb.received() == b"hello" + msg


def test_the_client_port_takes_s_primitives():
    """Annex A.2.2 over TCP: bind, send, and what comes back."""
    node = Node("0.0.0.1", Timing(1200, 0.6, 0.6))
    srv = SisServer(node, lambda fn: fn(), port=0)
    srv.start_background()
    try:
        c = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
        c.sendall(P.encode(P.prim(P.BIND_REQUEST, sap=6, rank=0,
                                  service=P.ServiceType(P.MODE_ARQ, P.CONFIRM_NONE, True))))
        f = P.Framer()
        got: list = []
        deadline = time.monotonic() + 5
        while not got and time.monotonic() < deadline:
            got = f.push(c.recv(4096))
        assert got[0].type == P.BIND_ACCEPTED and got[0]["sap"] == 6
        c.sendall(P.encode(P.prim(P.UNIDATA_REQUEST, priority=0, dest_sap=6, dest=2,
                                  group=False, mode=P.ServiceType(P.MODE_DEFAULT), ttl=0,
                                  u_pdu=b"x")))
        deadline = time.monotonic() + 5
        while not (node.sis.queue or node.cas.links) and time.monotonic() < deadline:
            time.sleep(0.01)
        node.tick(1.0)
        assert node.cas.links[2].state == "making"
        c.close()
    finally:
        srv.stop()
