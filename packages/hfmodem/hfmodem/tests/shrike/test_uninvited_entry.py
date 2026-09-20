# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The door a peer did not open, and what this station can say about it.

An uninvited upgrade opens the PACTOR-3 phase at `arq.ArqConfig.entry_sl` and puts
the host's own bytes in the field -- not the template entry packet a grant draws,
which `5aa3f1bc` is the reason for: the peer's first look at a waveform it must
acquire cold should be a full field rather than fill. What it asks is the same
question, and until something comes back nothing has answered it.

`ptc.PtcHost._entry_answered` was gated on `arq.entry_pending`, which `upgrade`
sets only on a grant, so the milestone -- and the `entry read` column the console
scores off its line -- could not fire on this door at all. All 27 uninvited
entries in this station's record scored 0 there by construction rather than by
measurement.

The seam here is the transmitter's: `Keyed` is what `ptc.PtcHost.send_packet`
hands a packet to, so the level asserted is the level that would be rendered.

Run:  python -m pytest hfmodem/tests/shrike/test_uninvited_entry.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from unittest import mock

import pytest

from hfmodem.shrike import arq, onair, pactor1, rxfront, spec
from hfmodem.shrike.ptc import GRANT_ENTRY_SL, PtcHost

MESSAGE = b"the bytes the uninvited upgrade was taken for"
ANSWERED = "the peer answered the entry packet"


class Keyed:
    """Every PACTOR-3 packet this station puts on the air, with its level."""

    def __init__(self) -> None:
        self.p3: list[tuple[str, int]] = []

    def attach(self, host) -> None: ...
    def pump(self) -> None: ...
    def cycle(self) -> None: ...
    def connect_burst(self, mycall: str, dxcall: str) -> None: ...
    def send_p1_packet(self, payload, baud, seq, **kw) -> int: return len(payload)
    def send_p1_breakin(self, payload, baud, seq, **kw) -> int: return len(payload)
    def send_p1_cs(self, index: int) -> None: ...
    def send_cs(self, index: int) -> None: ...

    def send_packet(self, sl, payload, status, breakin=False) -> int:
        self.p3.append(("packet", sl))
        return len(payload)

    def send_entry_packet(self, sl, payload, status, acquire=False) -> int:
        self.p3.append(("entry", sl))
        return len(payload)


def _cs(index: int, protocol=spec.Protocol.PACTOR3):
    return rxfront.Event(0.1, "cs", "control", protocol=protocol, cs=index,
                         sense=0)


class _Link:
    """A PACTOR-1 link with bytes queued, driven to whichever door is asked for."""

    def __init__(self) -> None:
        self.tx = Keyed()
        self.host = PtcHost(peer=self.tx, mycall="W9SSJ")
        self.said: list[str] = []
        self.host.log = self.said.append

    def open(self, dxcall: str) -> None:
        self.host.arq.on_host_connect("W9SSJ", dxcall)
        self.host.on_rx_event(_cs(pactor1.CS_SPEED, spec.Protocol.PACTOR1))
        self.host.tick()
        self.host.arq.on_host_data(MESSAGE)
        self.host.tick()

    def answer(self, protocol=spec.Protocol.PACTOR3) -> None:
        """The peer acknowledges what is in flight; its codeword follows our seq."""
        seq = self.host.arq.tx_seq
        assert seq is not None
        self.host.on_rx_event(
            _cs(arq.CS_REQUEST if seq & 1 else arq.CS_ACK, protocol))
        self.host.tick()

    def grant(self) -> None:
        self.host.on_rx_event(rxfront.Event(0.1, "unassigned", "grant",
                                            protocol=spec.Protocol.PACTOR1,
                                            spare=pactor1.CS_59A, sense=0))
        self.host.tick()


def _uninvited(entry_sl: int | None = None) -> _Link:
    link = _Link()
    if entry_sl is not None:
        link.host.arq.cfg.entry_sl = entry_sl
    link.open("WS8EOC")
    link.answer(spec.Protocol.PACTOR1)
    assert link.host.protocol is spec.Protocol.PACTOR3
    assert f"link upgraded to PACTOR-3 at SL{link.host.arq.entry_level} " \
        "uninvited" in link.said
    return link


def _granted(entry_sl: int | None = None) -> _Link:
    link = _Link()
    if entry_sl is not None:
        link.host.arq.cfg.entry_sl = entry_sl
    link.host.p1_grant_only = True
    link.open("VE3KPG")
    link.grant()
    assert link.host.protocol is spec.Protocol.PACTOR3
    return link


def test_an_uninvited_upgrade_keys_its_traffic_and_not_a_template():
    link = _uninvited()
    assert link.host._uninvited_entry and not link.host.arq.entry_pending
    assert link.tx.p3 == [("packet", arq.ArqConfig.entry_sl)]


def test_the_milestone_fires_on_the_uninvited_door():
    link = _uninvited()
    link.answer()
    assert not link.host._uninvited_entry
    assert ANSWERED in " ".join(link.said)
    assert link.host.arq.speed_level == link.host.arq.traffic_level


def test_the_granted_door_is_unchanged():
    link = _granted()
    assert link.host.arq.entry_pending and not link.host._uninvited_entry
    assert link.tx.p3 == [("entry", GRANT_ENTRY_SL)]
    link.answer()
    assert ANSWERED in " ".join(link.said)
    assert link.tx.p3 == [("entry", GRANT_ENTRY_SL),
                          ("packet", link.host.arq.traffic_level)]


@pytest.mark.parametrize("sl", [1, 2, 3, 4, 5, 6])
def test_every_level_the_flag_accepts_is_the_level_that_flies(sl: int):
    """`Keyed.send_packet` is the seam `ptc.PtcHost.send_packet` hands the
    opening packet to, so this is the level the renderer would be given."""
    link = _uninvited(entry_sl=sl)
    assert link.tx.p3 == [("packet", sl)]
    assert link.host.arq.speed_level == sl


def test_the_granted_door_keeps_its_own_level():
    """`GRANT_ENTRY_SL` is not `entry_sl` and the flag does not reach it."""
    assert _granted(entry_sl=6).tx.p3 == [("entry", GRANT_ENTRY_SL)]


def test_the_flag_reaches_the_link_the_arm_builds():
    """The other half of the chain: `onair.run` resolves the namespace and hands
    the level to the host it builds. `entry_sl` had no flag at all, so all 27
    uninvited entries on record went out at the default and no hand-flown arm
    could have moved it."""
    built, real = [], onair.PtcHost
    argv = ["onair", "--dxcall", "WS8EOC", "--mycall", "W9SSJ",
            "--p3-entry-sl", "1"]
    parsed = []
    with mock.patch.object(onair, "run", lambda a: parsed.append(a) or 0), \
            mock.patch.object(sys, "argv", argv):
        onair.main()
    with mock.patch.object(onair, "PtcHost",
                           lambda **kw: built.append(real(**kw)) or built[-1]), \
            contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()), \
            contextlib.suppress(SystemExit):
        onair.run(parsed[0])
    assert built, "run never built a host"
    assert built[0].arq.cfg.entry_sl == 1
    assert built[0].arq.entry_level == 1
