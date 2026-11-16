# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.p3_arq_support import Seam, calling_station, cs_event, grant
from hfmodem.tests.shrike.p3_reply_support import arm_reply, emit_reply, receiving


def test_radio_tx_reports_current_slot_keying(tmp_path):
    assert onair.RadioTx(transmit=False, outdir=tmp_path).p3_slot_keyed() is None
    s, tx, live, grid = receiving(tmp_path)
    assert tx.p3_slot_keyed() is True               # slot 1's ACK went out
    arm_reply(tx, live, grid, 2)
    assert tx.p3_slot_keyed() is False              # armed, not keyed
    assert emit_reply(tx, live) == arq.CS_ACK
    assert tx.p3_slot_keyed() is True
    # A decode after the tick can answer the slot it left behind, which the
    # key of the later one has already spent.
    tx.aim(grid, 1)
    assert tx.p3_slot_keyed() is True


@pytest.mark.parametrize('emitted', [False, True, None])
def test_tick_reports_peer_slot_emission(monkeypatch, emitted):
    peer = Seam()
    monkeypatch.setattr(peer, 'p3_slot_keyed', lambda: emitted)
    host = PtcHost(peer=peer, mycall='W9SSJ')
    host.protocol = spec.Protocol.PACTOR3
    cycles = []

    def on_cycle(*, elapsed_ticks, cycle_ticks, response_emitted):
        cycles.append((elapsed_ticks, cycle_ticks, response_emitted))

    monkeypatch.setattr(host.arq, 'on_cycle', on_cycle)
    host.tick(elapsed_ticks=2, cycle_ticks=3)
    assert cycles == [(2, 3, emitted)]


@pytest.mark.parametrize('receive_owned', [False, True])
def test_grant_turn_marker_respects_receive_opportunity_ownership(receive_owned):
    host, keyed = calling_station()
    host.on_rx_event(grant())
    token = host.arq.begin_receive_opportunity() if receive_owned else None
    before = len(keyed.cs)
    host.on_rx_event(cs_event(arq.CS_BREAKIN, spec.Protocol.PACTOR3))
    assert host.arq.role == arq.IRS
    host.tick()
    assert len(keyed.cs) == before + int(receive_owned)
    if token is not None:
        host.arq.finish_receive_opportunity(token)
    host.tick()
    assert len(keyed.cs) == before + int(receive_owned) + 1
