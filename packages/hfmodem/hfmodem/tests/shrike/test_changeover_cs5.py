# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A reference-CS5 pilot must not change unrelated answers or clock policy."""
import sys

import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_turn_recovery import Sink, pending, packet


@pytest.mark.parametrize("enabled", [False, True])
def test_only_good_p3_changeovers_select_experimental_cs5(enabled):
    io=Sink()
    io.p3_changeover_cs5=enabled
    a=arq.PactorArq(io)
    a.role=arq.IRS
    a._enter_connected()
    for _ in range(3):
        packet(a)
        assert io.sent[-1] == ("cs", arq.CS_NAK if enabled else arq.CS_ACK)
        a.on_cycle()
    assert io.delivered == [b"RMS"]
    a.on_rx_packet(1,b"hello",1,True,protocol=spec.Protocol.PACTOR3)
    assert io.sent[-1] == ("cs",arq.CS_ACK)
    a.on_cycle()
    a.on_cycle()
    assert io.sent[-1] == ("cs",arq.CS_REQUEST)


@pytest.mark.parametrize("protocol", [spec.Protocol.PACTOR1,spec.Protocol.PACTOR2])
def test_other_protocols_keep_their_changeover_answer(protocol):
    io=Sink()
    io.p3_changeover_cs5=True
    a=arq.PactorArq(io)
    a.role=arq.IRS
    a._enter_connected()
    a.on_rx_packet(1,b"RMS",0,True,breakin=True,protocol=protocol)
    assert io.sent[-1] == ("cs",arq.CS_ACK)


def test_recovery_plain_ack_barrier_survives_experiment():
    a,io=pending()
    io.p3_changeover_cs5=True
    packet(a,repeated_stint=True)
    assert io.sent[-1] == ("cs",arq.CS_ACK)
    assert a._turn_ack_owed and a._buffer_raw == 6


def test_qrt_still_gets_final_ack():
    io=Sink()
    io.p3_changeover_cs5=True
    a=arq.PactorArq(io)
    a.role=arq.IRS
    a._enter_connected()
    a.on_rx_packet(1,b"",spec.STATUS_QRT,True,breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    assert io.sent[-1] == ("cs",arq.CS_ACK) and a._rx_close_pending
    a.on_cs_emitted(arq.CS_ACK)
    assert a.state == arq.State.DISCONNECTED


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_requires_explicit_opt_in(monkeypatch, enabled):
    monkeypatch.setattr(sys,"argv",["onair","--dxcall","WS8EOC"]
                        + (["--p3-changeover-cs5"] if enabled else []))
    monkeypatch.setattr(onair.config,"mail_password",lambda *args: None)
    observed=[]
    monkeypatch.setattr(onair,"run",lambda args: observed.append(args.p3_changeover_cs5) or 0)
    assert onair.main() == 0
    assert observed == [enabled]
