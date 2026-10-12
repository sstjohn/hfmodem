# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Three behaviours of SCS's modem layer, as SCS's own copy of it shows them.

Three facts read off SCS's own protocol engine: the longpath marker
stretches the cycle to 1.4 s, the called station names its caller from the
first text line, and the dF status byte is the receive offset from 1500 Hz.

Run: pytest hfmodem/tests/shrike/test_pmon_modem_layer.py
"""
from __future__ import annotations

import select
from types import SimpleNamespace

import pytest

from hfmodem.shrike import ptc, spec
from hfmodem.shrike.arq import ArqConfig, PactorArq, State
from hfmodem.shrike.ptc import PtcHost, SimPeer, serve


def _serve_ticks(host: PtcHost, ticks: int, monkeypatch) -> list[float]:
    """The instants `serve` clocks `host` at, on a simulated clock."""
    now = [0.0]

    def elapse(_r, _w, _x, timeout):
        now[0] += timeout
        return [], [], []

    at: list[float] = []
    tick = host.tick

    def clocked(**kw):
        at.append(now[0])
        tick(**kw)
        if len(at) == ticks:
            raise KeyboardInterrupt

    monkeypatch.setattr(ptc.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(select, "select", elapse)
    monkeypatch.setattr(host, "tick", clocked)
    serve(host, verbose=False)
    return at


@pytest.mark.parametrize("dial, slot", [("!N0DX", 1.4), ("N0DX", 1.25)])
def test_a_longpath_link_runs_its_schedule_on_the_longpath_cycle(dial, slot,
                                                                 monkeypatch):
    peer = SimPeer()
    host = PtcHost(peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", dial)
    peer.pump()
    assert (host.arq.state, peer._far.state) == (State.CONNECTED, State.CONNECTED)
    assert peer._far.answering and host.arq.slot_s == peer._far.slot_s == slot
    sent: list[float] = []
    send = peer.send_packet
    monkeypatch.setattr(peer, "send_packet", lambda *a, **k: (
        sent.append(ptc.time.monotonic()), send(*a, **k))[1])
    at = _serve_ticks(host, 8, monkeypatch)
    assert [round(b - a, 6) for a, b in zip(at, at[1:])] == [slot] * 7
    assert len(sent) >= 7
    assert {round(b - a, 6) for a, b in zip(sent, sent[1:])} == {slot}


def test_the_long_cycle_on_a_longpath_link_is_three_longpath_slots():
    arq = PactorArq(SimpleNamespace(log=print), ArqConfig())
    arq.connect_variant = "longpath"
    assert arq.cycle_s == spec.CYCLE_SHORT_LONGPATH_S
    arq.cycle_long = True
    assert arq.cycle_s == spec.CYCLE_LONG_LONGPATH_S
    arq.connect_variant = "normal"
    assert arq.cycle_s == spec.CYCLE_LONG_S


class _End:
    """One station's antenna: a connect burst names the called party only."""

    def __init__(self, air):
        self.air, self.host = air, None

    def connect_burst(self, mycall, dxcall):
        self.air.send(self, "on_rx_connect",
                      ("", dxcall, self.host.arq.connect_variant))

    def send_packet(self, sl, payload, status, breakin=False):
        self.air.send(self, "on_rx_packet", (sl, payload, status, True, breakin,
                                             None, self.host.arq.cycle_long))

    def send_cs(self, cs_index):
        self.air.send(self, "on_rx_cs", (cs_index,))

    def attach(self, host):
        self.host = host

    def pump(self):
        self.air.pump()

    def cycle(self):
        pass


class _Air:
    def __init__(self):
        self.q: list = []
        self.a, self.b = _End(self), _End(self)

    def send(self, src, method, args):
        self.q.append((self.b if src is self.a else self.a, method, args))

    def pump(self):
        while self.q:
            dst, method, args = self.q.pop(0)
            getattr(dst.host.arq, method)(*args)


def _call(first_line: bytes):
    air = _Air()
    PtcHost(air.a, mycall="W9SSJ")
    PtcHost(air.b, mycall="N0DX")
    caller, called = air.a.host, air.b.host
    called.set_listen(True)
    caller.arq.on_host_connect("W9SSJ", "N0DX")
    air.pump()
    assert called.arq.answering and called.arq.dxcall == ""
    caller.arq.on_host_data(first_line)
    for _ in range(12):
        caller.tick()
        called.tick()
    return caller, called


def test_the_called_station_names_its_caller_from_the_announcement():
    _, called = _call(b"1w9ssj\rHELLO\r")
    link = called.channel(called.ptchn).link
    assert called.arq.dxcall == "W9SSJ"
    assert link == [f"({called.ptchn}) CONNECTED to W9SSJ"]
    assert bytes(called.channel(called.ptchn).rx) == b"HELLO\r"


def test_a_long_first_line_keeps_eight_characters():
    _, called = _call(b"1dl6maa/p-10\r")
    assert called.arq.dxcall == "DL6MAA/P"


def test_a_first_line_naming_nobody_drops_the_link():
    _, called = _call(b"1K\rHELLO\r")
    assert not any("CONNECTED" in m for m in called.channel(called.ptchn).link)
    assert called.arq.state is not State.CONNECTED
    assert bytes(called.channel(called.ptchn).rx) == b""


@pytest.mark.parametrize("centre, df", [
    (None, 0x80), (1500.0, 0), (1475.5, 0xE8), (1627.0, 127), (1373.0, 0x81),
    (1627.5, 0x80), (1372.9, 0x80),
])
def test_df_is_the_receive_offset_from_1500_hz(centre, df):
    peer = SimpleNamespace(attach=lambda host: None,
                           sessrx=SimpleNamespace(receive_centre_hz=centre))
    assert PtcHost(peer)._status_bytes()[3] == df
