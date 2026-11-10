# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Native ARQ establishment is independent of PTC caller identification and gear."""
from types import SimpleNamespace as NS

import pytest

from hfmodem.shrike import arq
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.p4host import P4Host, P4Outbox
from hfmodem.shrike.p4turn import P4Turn
from hfmodem.shrike.station_runtime import StationPtcRuntime
from hfmodem.tests.station.recording_live_support import LOCAL, PEER, Clock, assert_edges


@pytest.mark.parametrize('answering', [False, True])
def test_native_establishment_precedes_deferred_ptc_notification(live, answering):
    owner = live()
    owner.advance(65)
    host = owner.bind(PtcHost(mycall=LOCAL, calendar_clock=Clock()))
    machine = owner.bind(host.arq)
    machine.mycall, machine.dxcall = LOCAL, '' if answering else PEER
    machine.answering = answering
    machine.role = 'IRS' if answering else 'ISS'
    delivered = []
    callback = host.connected

    def connected(*args):
        delivered.append(len(owner.starts))
        callback(*args)

    host.connected = connected
    machine._enter_connected()
    row = owner.one()
    cid = row['id']
    assert row['key'][1] == 'pactor'
    assert row['positions'] == {'card': 520, 'dac': 325}
    assert delivered == [1], 'source boundary precedes PTC application notification'
    assert row['peer'] == ('' if answering else PEER)
    if answering:
        assert not host._link_up
        host._link_established(PEER)
        row = owner.one()
        assert row['id'] == cid and row['peer'] == PEER
    owner.advance(67)
    machine._finish_disconnected()
    machine._finish_disconnected()
    assert_edges(owner, start={'card': 520, 'dac': 325}, end={'card': 536, 'dac': 335})


@pytest.mark.parametrize('close', ['native', 'p4-link-down'])
def test_p1_to_p4_handoff_preserves_id_and_closes_once(live, close):
    owner = live()
    machine = owner.bind(arq.PactorArq(arq.ArqIO()))
    machine.mycall, machine.dxcall, machine.role = LOCAL, PEER, 'ISS'
    owner.advance(65)
    machine._enter_connected()
    turn = owner.bind(P4Turn(P4Outbox(machine), P4Host()))
    machine.p4_admitted = True
    owner.advance(67)
    if close == 'p4-link-down':
        turn.link_down('peer timeout')
    else:
        machine._finish_disconnected()
    row = assert_edges(owner, start={'card': 520, 'dac': 325}, end={'card': 536, 'dac': 335})
    assert row['key'][1] == 'pactor'
    if close == 'p4-link-down':
        assert row['state'] == 'partial'


def test_entry_probe_does_not_establish_but_interrupted_native_link_is_kept(live):
    owner = live('off')
    machine = owner.bind(arq.PactorArq(arq.ArqIO()))
    machine.mycall, machine.dxcall, machine.role = LOCAL, PEER, 'ISS'
    machine.state = arq.State.CONNECTING
    machine.p4_expected = True
    machine._finish_disconnected()
    assert not owner.rows()
    machine._enter_connected()
    machine.on_host_abort()
    row = owner.one()
    assert row['state'] == 'partial'
    assert len(owner.starts) == len(owner.ends) == 1


@pytest.mark.parametrize('managed', [False, True])
def test_station_ptc_runtime_retirement_closes_established_unknown_caller(live, managed):
    owner = live('off')
    station = NS(rig=NS(), arbiter=NS(identity=NS(mode='cw')), capture=owner.capture)
    runtime = owner.bind(StationPtcRuntime(NS(station=station)))
    runtime._managed = managed
    runtime.host = owner.bind(PtcHost(mycall=LOCAL, calendar_clock=Clock()))
    machine = owner.bind(runtime.host.arq)
    machine.mycall, machine.dxcall, machine.role, machine.answering = LOCAL, '', 'IRS', True
    machine._enter_connected()
    runtime.stop()
    runtime._run()
    row = owner.one()
    assert row['peer'] == '' and row['state'] == 'partial'
    assert len(owner.starts) == len(owner.ends) == 1
