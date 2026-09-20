# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The SCS connect operators, and the two acceptance settings behind them.

`C %CALL` is a Robust Connect and `C !CALL` (or `;CALL`) a longpath call, on the
terminal and in WA8DED hostmode alike [PTC-IIIusb 4.1 §6.22]. Both together is
refused, a bare `C` re-dials the last remote with the operator it was dialled
with, and `CONType` decides which incoming calls this station answers at all.

Until the branch-B encoder lands, "robust" is a request the keying seam has not
been taught to render, so what is checked here is the boundary the seam reads:
the callsign and variant `PactorArq` holds while it calls `io.connect_burst`.

Run: pytest hfmodem/tests/shrike/test_connect_prefix.py
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import hostmode, p1rx, rxfront
from hfmodem.shrike.arq import ArqConfig, ArqIO, PactorArq, State, parse_connect_arg
from hfmodem.shrike.ptc import PtcHost, SimPeer

PACTOR_CH = 4


class _Keyed(ArqIO):
    """Records what the FSM asked for, and which call it asked for it as."""

    def __init__(self, arq=None):
        self.arq = arq
        self.calls: list[tuple[str, str]] = []

    def connect_burst(self, mycall, dxcall):
        self.calls.append((dxcall, self.arq.connect_variant))

    def log(self, msg):
        pass


def _calling(cfg: ArqConfig | None = None) -> tuple[PactorArq, _Keyed]:
    io = _Keyed()
    io.arq = arq = PactorArq(io, cfg or ArqConfig())
    return arq, io


class _Master:
    """A host program on the wire: terminal init, then CRC hostmode."""

    def __init__(self):
        self.host = PtcHost(mycall="W9SSJ")
        self.host.feed(b"JHOST4\r")
        self.decoder = hostmode.Decoder("master")
        self.counter = 0

    def cmd(self, text: str) -> hostmode.Response:
        self.counter ^= 1
        (resp,) = self.decoder.feed(self.host.feed(
            hostmode.command(PACTOR_CH, text, counter=self.counter)))
        return resp


@pytest.mark.parametrize("arg, parsed", [
    ("W1AW", ("W1AW", "normal")),
    ("%w1aw", ("W1AW", "robust")),
    ("!W1AW", ("W1AW", "longpath")),
    (";W1AW", ("W1AW", "longpath")),
    ("  %W1AW ", ("W1AW", "robust")),
])
def test_an_operator_is_stripped_and_named(arg, parsed):
    assert parse_connect_arg(arg) == parsed


@pytest.mark.parametrize("arg", ["!%W1AW", "%!W1AW", "%%W1AW", "%", ""])
def test_two_operators_and_no_callsign_are_refused(arg):
    with pytest.raises(ValueError):
        parse_connect_arg(arg)


def test_the_variant_stands_while_the_seam_keys_it():
    arq, io = _calling(ArqConfig(max_connect_retries=4))
    arq.on_host_connect("W9SSJ", "%ws8eoc")
    for _ in range(3):
        arq.on_cycle()
    assert arq.dxcall == "WS8EOC"
    assert len(io.calls) > 1, "the call was never resent"
    assert set(io.calls) == {("WS8EOC", "robust")}


def test_longpath_takes_the_existing_variant():
    arq, io = _calling()
    arq.on_host_connect("W9SSJ", "!WS8EOC")
    assert io.calls == [("WS8EOC", "longpath")]
    assert arq.connect_variant == "longpath"


def test_a_plain_call_is_unchanged():
    arq, io = _calling()
    arq.on_host_connect("W9SSJ", "ws8eoc")
    assert io.calls == [("WS8EOC", "normal")]


def test_a_bare_connect_re_dials_with_the_same_operator():
    arq, io = _calling()
    arq.on_host_connect("W9SSJ", "%WS8EOC")
    arq.on_host_abort()
    arq.on_host_connect("W9SSJ", "")
    assert io.calls == [("WS8EOC", "robust"), ("WS8EOC", "robust")]


def test_an_incoming_call_does_not_become_the_last_remote():
    arq, io = _calling()
    arq.on_host_connect("W9SSJ", "!WS8EOC")
    arq.on_host_abort()
    arq.on_host_listen(True)
    arq.on_rx_connect("KI0BK", "W9SSJ")
    arq.on_host_abort()
    arq.on_host_connect("W9SSJ", "")
    assert io.calls[-1] == ("WS8EOC", "longpath")


def test_nothing_to_re_dial_is_refused():
    arq, _ = _calling()
    with pytest.raises(ValueError):
        arq.on_host_connect("W9SSJ", "")


def test_hostmode_takes_the_operator_and_refuses_both():
    m = _Master()
    assert m.cmd("C %WS8EOC").code == hostmode.OK
    assert m.host.arq.state == State.CONNECTING
    assert (m.host.arq.dxcall, m.host.arq.connect_variant) == ("WS8EOC", "robust")

    m.host.arq.on_host_abort()
    assert m.cmd("C %!WS8EOC").code == hostmode.FAIL
    assert m.host.arq.state == State.DISCONNECTED

    assert m.cmd("C").code == hostmode.OK
    assert (m.host.arq.dxcall, m.host.arq.connect_variant) == ("WS8EOC", "robust")


def test_the_terminal_says_which_call_it_is_making():
    host = PtcHost(mycall="W9SSJ")
    assert "*** NOW CALLING WS8EOC (ROBUST CONNECT)" in \
        host.feed(b"C %WS8EOC\r").decode("latin-1")
    host.arq.on_host_abort()
    assert "*** NOW CALLING KI0BK (LONGPATH)" in \
        host.feed(b"C !KI0BK\r").decode("latin-1")
    host.arq.on_host_abort()
    out = host.feed(b"C W1AW\r").decode("latin-1")
    assert "*** NOW CALLING W1AW\r\n" in out
    host.arq.on_host_abort()
    assert "*** ERROR" in host.feed(b"C !%W1AW\r").decode("latin-1")


@pytest.mark.parametrize("contype, accepted", [
    (0, ()),
    (1, ("Normal", "Longpath")),
    (2, ("Robust",)),
    (3, ("Normal", "Longpath", "Robust")),
])
def test_contype_decides_which_calls_are_answered(contype, accepted):
    for variant in ("Normal", "Longpath", "Robust"):
        arq, _ = _calling(ArqConfig(contype=contype))
        arq.on_host_listen(True)
        arq.on_rx_connect("", "W9SSJ", variant)
        answered = arq.state == State.CONNECTED
        assert answered == (variant in accepted), \
            f"CONType {contype} {'took' if answered else 'refused'} {variant}"


def test_contype_gates_the_decoded_burst_the_receiver_hands_over():
    for contype, answered in ((3, True), (2, False)):
        host = PtcHost(mycall="W9SSJ")
        host.arq.cfg.contype = contype
        host.arq.on_host_listen(True)
        host.on_rx_event(rxfront.Event(
            t=0.0, kind="connect", text="###CONNECT",
            connect=p1rx.Connect("Normal", "W9SSJ", False)))
        assert (host.arq.state == State.CONNECTED) == answered


def test_the_acceptance_settings_are_the_manuals_defaults():
    cfg = ArqConfig()
    assert (cfg.contype, cfg.conintegrity) == (3, 0)


@pytest.mark.parametrize("field, value", [
    ("contype", 4), ("contype", -1), ("conintegrity", 2),
])
def test_a_setting_outside_its_range_is_refused(field, value):
    with pytest.raises(ValueError):
        ArqConfig(**{field: value})


def test_the_host_can_set_both():
    host = PtcHost(mycall="W9SSJ")
    host.feed(b"CONTYPE 2\r")
    host.feed(b"CONINTEGRITY 1\r")
    assert (host.arq.cfg.contype, host.arq.cfg.conintegrity) == (2, 1)
    assert "*** ERROR" in host.feed(b"CONTYPE 4\r").decode("latin-1")
    assert host.arq.cfg.contype == 2
    assert "*** CONTYPE: 2" in host.feed(b"CONTYPE\r").decode("latin-1")


def test_a_simulated_far_end_sees_the_call_it_was_made_with():
    for contype, answered in ((2, True), (1, False)):
        peer = SimPeer()
        host = PtcHost(peer=peer, mycall="W9SSJ")
        peer._far.cfg.contype = contype
        host.arq.on_host_connect("W9SSJ", "%K7ABC")
        peer.pump()
        assert (peer._far.state == State.CONNECTED) == answered, \
            f"CONType {contype} against a robust call"
