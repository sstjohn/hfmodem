# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Concurrent receive admission must precede notifications and RF state changes."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from hfmodem.station.arbiter import TxArbiter, Refused
from hfmodem.station.contact import ContactOwner
from hfmodem.tests.core.test_arbiter import arb as arb, rig as rig, req


def test_four_simultaneous_receivers_have_one_opaque_owner_without_active_snapshot():
    arbiter = TxArbiter(None, None)
    barrier = threading.Barrier(4)
    def claim(protocol):
        barrier.wait(timeout=2)
        return protocol, arbiter.claim_contact(protocol)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(claim, ('vara','pactor','ardop','rpr')))
    winners = [(p,t) for p,t in results if t is not None]
    assert len(winners) == 1
    protocol, token = winners[0]
    assert arbiter.contact_protocol == protocol
    assert arbiter.claim_contact(protocol) is None   # another attachment cannot steal it
    with arbiter._admit:                            # an existing keyed owner can renew
        assert arbiter.claim_contact(protocol, token) is token
        assert arbiter.claim_contact('other') is None
    with pytest.raises(ValueError, match='stale'):
        arbiter.release_contact('other', token)
    arbiter.release_contact(protocol, token)
    with pytest.raises(ValueError, match='stale'):
        arbiter.claim_contact(protocol, token)
    assert arbiter.claim_contact('other') is not None


def test_contact_blocks_other_keying_and_scanning_before_snapshot_changes(arb):
    token = arb.claim_contact('pactor')
    assert token is not None and tuple(arb.active_protocols()) == ()
    with pytest.raises(Refused, match='pactor'):
        arb.submit(req(protocol='vara'))
    assert not arb.audio.transmitted and arb.sent == 0
    assert arb.acquire_scan_hold('vara', eligible=lambda: True) is None
    with arb.try_hold() as held:
        assert not held                           # direct ALE retuning cannot cross acquisition
    arb.release_contact('pactor', token)
    scan = arb.acquire_scan_hold('vara', eligible=lambda: True)
    assert scan is not None and arb.claim_contact('pactor') is None
    arb.release_scan_hold(scan)
    assert arb.claim_contact('pactor') is not None


def test_nonblocking_admission_and_owner_retirement_do_not_retain_a_key_mutex():
    arbiter = TxArbiter(None, None)
    owner = ContactOwner(arbiter, 'vara')
    with arbiter._admit:
        assert not owner.admit()
    assert owner.admit()
    owner.release_if(lambda: False)
    assert arbiter.contact_protocol == 'vara'
    assert arbiter._admit.acquire(blocking=False)
    arbiter._admit.release()
    owner.close()
    assert arbiter.contact_protocol is None
    owner.close()                                  # normal repeated retirement is harmless
    assert not ContactOwner(None, 'vara').admit()    # a read-only station cannot acquire


def test_real_vara_request_and_ax25_sabm_compete_before_notifications():
    from hfmodem.kestrel.vara import vara_arq as va, vara_frames as vf
    from hfmodem.ax25.connected import Connection, State, SABM
    from hfmodem.ax25.frame import Frame, address
    arbiter = TxArbiter(None, None)
    barrier = threading.Barrier(2)
    vara_owner, packet_owner = ContactOwner(arbiter,'vara'), ContactOwner(arbiter,'rpr')
    notices = []
    class IO(va.VaraIO):
        def admit_contact(self):
            barrier.wait(timeout=2)
            return vara_owner.admit()
        def pending(self): notices.append('pending')
    vara = va.VaraStationHandshake(['W9SSJ'], IO(), bw='500')
    vara.listen()
    vara._send_burst = lambda *a, **kw: True         # no synthetic waveform needed here
    packet = Connection('W9SSJ', lambda raw: False)
    def admit_packet():
        barrier.wait(timeout=2)
        return packet_owner.admit()
    packet.admit_contact = admit_packet
    raw = Frame((address('W9SSJ',flag=True),address('N0CALL',last=True)),SABM|0x10,None,b'').encode()
    kind = vf.connect_request('500')
    tones = vf.handshake_tones('W9SSJ', vf.for_bw(kind,'500'))
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(vara.on_rx_tones, tones, kind)
        b = pool.submit(packet.on_frame, raw)
        a.result(timeout=3);b.result(timeout=3)
    assert (vara.state == va.VaraState.CONNECTING) + (packet.state is State.CONNECTED) == 1
    assert notices == (['pending'] if arbiter.contact_protocol=='vara' else [])
    assert bool(packet.controls) == (arbiter.contact_protocol=='rpr')
    assert packet.accepted == packet.delivered == 0
    vara_owner.close();packet_owner.close()


def test_old_vara_adapter_cannot_release_or_reuse_the_next_adapters_reservation():
    from types import SimpleNamespace
    from hfmodem.kestrel.host.station_runtime import StationTransport
    arbiter = TxArbiter(None, None)
    transport = StationTransport(SimpleNamespace(station=SimpleNamespace(arbiter=arbiter)))
    old, new = object(), object()
    assert transport.admit_contact(old)
    assert not transport.admit_contact(new)
    transport.retire_contact(new, idle=True)
    assert arbiter.contact_protocol == 'vara'
    transport.retire_contact(old, idle=True)
    assert transport.admit_contact(new)
    transport.retire_contact(old, idle=True)
    assert arbiter.contact_protocol == 'vara'
    transport.retire_contact(new, idle=True)
    assert arbiter.contact_protocol is None
    transport.stop.set()
    assert not transport.admit_contact(new)


def test_packet_runtime_defers_a_competing_sabm_and_refuses_outgoing_without_credit():
    from types import SimpleNamespace
    from hfmodem.brubru.host.runtime import PacketRuntime
    from hfmodem.tests.brubru.test_connected_hosts import Link
    from hfmodem.ax25.connected import State, SABM
    from hfmodem.ax25.frame import Frame, address
    arbiter = TxArbiter(None, None)
    token = arbiter.claim_contact('vara')
    link = Link()
    link.station = SimpleNamespace(arbiter=arbiter)
    runtime = PacketRuntime(link, mycall='W9SSJ', contact=ContactOwner(arbiter, 'rpr'))
    runtime.engine.listen = True
    raw = Frame((address('W9SSJ',flag=True),address('N0CALL',last=True)),SABM|0x10,None,b'').encode()
    try:
        with pytest.raises(ValueError, match='OWNER BUSY'):
            runtime.engine.connect('N0CALL')
        runtime.send_frame(raw);runtime._step()
        assert runtime.engine.state is State.DISCONNECTED
        assert not link.requests and not runtime.engine.controls
        assert runtime.engine.accepted == runtime.engine.delivered == 0
        arbiter.release_contact('vara', token)
        runtime.send_frame(raw);runtime._step()
        assert runtime.engine.state is State.CONNECTED and arbiter.contact_protocol == 'rpr'
        assert len(link.requests) == 1
        link.complete();runtime._step()
        runtime.engine.abort('test peer retired');runtime._step()
        assert arbiter.contact_protocol is None
    finally:
        runtime.stop()


def test_native_ardop_owner_refuses_competing_connect_and_retires_old_receive_epoch():
    from hfmodem.besra.station_runtime import StationArdopModem
    from hfmodem.tests.besra.test_station_runtime import Observer, wait_for
    arbiter = TxArbiter(None,None)
    token = arbiter.claim_contact('vara')
    modem = StationArdopModem(lambda samples: None, contact=ContactOwner(arbiter,'ardop'))
    received = []
    modem.core.receive_frames = received.extend
    modem.start(Observer())
    try:
        modem.set_mycall('W9SSJ')
        with pytest.raises(ValueError, match='OWNER BUSY'):
            modem.connect('N0CALL')
        generation = modem.generation
        modem.deliver_frames([object()], generation=generation)
        wait_for(lambda: modem.generation > generation)
        assert not received and arbiter.contact_protocol == 'vara'
        arbiter.release_contact('vara',token)
        modem.deliver_frames([object()], generation=generation) # cannot revive blocked epoch
        modem.set_listen(True)                                # wait for an owner turn
        assert not received
    finally:
        modem.stop()
    assert arbiter.contact_protocol is None


def test_ptc_deferred_admission_reserves_the_station_before_arq_connects(tmp_path):
    from types import SimpleNamespace
    from hfmodem.shrike.station_runtime import StationPtcRuntime
    from hfmodem.shrike.ptc import PtcHost
    from hfmodem.shrike.onair import RadioTx
    from hfmodem.station.arbiter import Identity
    arbiter = TxArbiter(None,None,identity=Identity('N0CALL'))
    station = SimpleNamespace(arbiter=arbiter,rig=object(),lanes={})
    runtime = StationPtcRuntime(SimpleNamespace(station=station))
    runtime.host = PtcHost()
    runtime.tx = RadioTx(transmit=False, outdir=tmp_path, settle=.020)
    runtime.host.enable_owner()
    token = arbiter.claim_contact('vara')
    with pytest.raises(ValueError, match='OWNER BUSY'):
        runtime._validate_connect('N0CALL','N0DX')
    arbiter.release_contact('vara',token)
    runtime._validate_connect('N0CALL','N0DX')
    assert arbiter.contact_protocol == 'pactor'
    assert not runtime.active                              # reservation precedes RF state
    assert arbiter.claim_contact('vara') is None
    runtime.contact.close()
