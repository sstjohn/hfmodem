# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""ARDOP adapters and dormant Sabir HostLink preserve source boundary order."""
from types import SimpleNamespace as NS

import pytest

from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.arq import session as S
from hfmodem.besra.host.protocol import ArdopState
from hfmodem.besra.station_runtime import StationArdopModem, _Observer
from hfmodem.sabir.arq.fsm import ArqConfig
from hfmodem.sabir.host import SabirModem
from hfmodem.tests.station.recording_live_support import (
    LOCAL, PEER, Clock, Observer, assert_edges,
)


def establish(modem, monkeypatch, *, incoming=False):
    session = modem._session
    session._remote, session._session_bw, session._session = PEER, 500, 23
    session._identify = None
    monkeypatch.setattr(modem, 'send', lambda *a: .01)
    session._set_state(ArdopState.IRS if incoming else ArdopState.ISS)
    session._sub = S._Sub.IRS_CONACK if incoming else S._Sub.ISS_CONACK
    if incoming:
        session._rx_irs(S.CONACK[500], b'', True)
    else:
        session._rx_iss(S.DATAACK_MIN, b'', True)


@pytest.mark.parametrize('incoming', [False, True])
@pytest.mark.parametrize('end', ['disconnect', 'abort', 'stop', 'timeout'])
def test_ardop_observer_adapter_stamps_before_host_and_teardown(live, monkeypatch, end, incoming):
    owner = live('off')
    modem = owner.bind(BesraModem(threaded=False))
    modem.set_mycall(LOCAL)
    observer = Observer(owner)
    modem.start(observer, threaded=False)
    try:
        establish(modem, monkeypatch, incoming=incoming)
        if end == 'disconnect':
            modem._session._end_contact('peer_end')
        elif end == 'abort':
            modem.abort()
        elif end == 'timeout':
            modem._session.tick(modem._session._now + 1000)
        else:
            modem.stop()
        row = owner.one()
        connected = [e for e in observer.events if e[0] == 'modem_connected']
        assert connected and connected[0][2] == 1
        assert len(owner.starts) == len(owner.ends) == 1
        assert row['key'][1] == 'ardop'
        if end != 'disconnect':
            assert row['state'] == 'partial'
    finally:
        modem.stop()


def test_station_ardop_forwarding_uses_same_core_hook_and_snapshot(live, monkeypatch):
    owner = live()
    runtime = owner.bind(StationArdopModem(lambda pcm: None, clock=Clock(),
                                         capture_position=lambda: owner.rx.position))
    owner.bind(runtime.core)
    first, replacement = Observer(owner), Observer(owner)
    runtime._observer = first
    runtime.core.set_mycall(LOCAL)
    runtime.core.start(_Observer(runtime), threaded=False)
    owner.advance(65)
    establish(runtime.core, monkeypatch)
    runtime._observer = replacement
    owner.advance(67)
    runtime.core._session._end_contact('peer_end')
    runtime.core.stop()
    assert_edges(owner, start={'card': 520, 'dac': 325}, end={'card': 536, 'dac': 335})
    assert len([e for e in first.events if e[0] == 'modem_connected']) == 1
    assert len([e for e in replacement.events if e[0] == 'modem_disconnected']) == 1
    assert replacement.events[-1][3] == 1


@pytest.mark.parametrize('iss', [False, True])
@pytest.mark.parametrize('end', ['abort', 'timeout'])
def test_mounted_sabir_hostlink_qualifies_without_enabling_station(live, monkeypatch, iss, end):
    owner = live('off')
    clock = Clock()
    air = NS(clock=clock, register=lambda modem: None)
    modem = owner.bind(SabirModem(air, ArqConfig(callsign=LOCAL)))
    monkeypatch.setattr(modem.link, 'send_control', lambda control: .01)
    owner.bind(modem.link)
    owner.bind(modem.fsm)
    observer = Observer(owner)
    modem.start(observer)
    modem.fsm.peer, modem.fsm.session = PEER, 19
    modem.fsm._enter_connected(iss=iss)
    if end == 'abort':
        modem.fsm.on_host_abort()
    else:
        clock.now += modem.fsm.cfg.inactivity_timeout_s + 1
        modem.fsm.on_timer(clock.now)
    row = owner.one()
    assert row['key'][1] == 'sabir'
    assert row['state'] == 'partial'
    connected = [e for e in observer.events if e[0] == 'modem_connected']
    assert connected and connected[0][2] == 1
    assert len(owner.starts) == len(owner.ends) == 1


@pytest.mark.parametrize('retirement', ['air-stop', 'replacement', 'host-stop'])
def test_sabir_air_owner_retirement_preserves_partial_contact(live, retirement):
    from hfmodem.core.audio import StreamLane
    from hfmodem.station.air import StationAir

    owner = live('off')
    air = StationAir(NS(), StreamLane(48000))
    modem = owner.bind(SabirModem(air, ArqConfig(callsign=LOCAL)))
    modem.start(Observer(owner))
    modem.fsm.peer, modem.fsm.session = PEER, 19
    modem.fsm._enter_connected(iss=True)
    if retirement == 'air-stop':
        air.stop()
    elif retirement == 'replacement':
        SabirModem(air, ArqConfig(callsign=LOCAL))
    else:
        modem.stop()
    row = owner.one()
    assert row['state'] == 'partial' and row.get('end_positions') is not None
    assert len(owner.starts) == len(owner.ends) == 1


def test_ardop_runtime_stop_retires_a_passive_core(live, monkeypatch):
    owner = live('off')
    runtime = owner.bind(StationArdopModem(lambda pcm: None, clock=Clock()))
    owner.bind(runtime.core)
    runtime.core.set_mycall(LOCAL)
    runtime.core.start(_Observer(runtime), threaded=False)
    establish(runtime.core, monkeypatch)
    runtime.stop()
    row = owner.one()
    assert row['state'] == 'partial' and row.get('end_positions') is not None
    assert len(owner.starts) == len(owner.ends) == 1
