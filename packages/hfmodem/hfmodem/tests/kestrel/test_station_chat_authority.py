# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Session authority and paused contact moves, using no radio or sockets."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.host.station_runtime import StationVaraRuntime
from hfmodem.station.arbiter import Refused, TxArbiter, TxRequest
from hfmodem.station.contact import ContactOwner
from hfmodem.station.link import KestrelLink
from .test_vara_host_modem import Observer


def bench(*, due=True, attended=True):
    arb = TxArbiter(SimpleNamespace(samples=0), SimpleNamespace(line_state='low'),
                    identity=SimpleNamespace(due=due),
                    attended=lambda sid: attended and sid == 'chat')
    return arb


def test_attended_contact_due_exemption_requires_captured_live_authority():
    arb = bench()
    authority = arb.authority('chat')
    live = [True]
    fence = lambda: live[0]
    assert arb.claim_contact('vara') is None
    assert arb.claim_contact('vara', authority=arb.authority('other')) is None
    token = arb.claim_contact('vara', authority=authority, fence=fence)
    assert token is not None
    assert arb.claim_contact('vara', token, authority=authority, fence=fence) is token
    assert arb.claim_contact('vara', token) is None
    live[0] = False
    assert arb.claim_contact('vara', token, authority=authority, fence=fence) is None
    live[0] = True
    arb._revocations['chat'] = (authority[1] + 1, 0)
    assert arb.claim_contact('vara', token, authority=authority, fence=fence) is None
    arb.release_contact('vara', token)


def test_unattended_session_keeps_due_id_refusal():
    arb = bench(attended=False)
    assert arb.claim_contact('vara', authority=arb.authority('chat'), fence=lambda: True) is None


def test_existing_contact_token_can_be_reused_inside_own_transmission():
    arb = bench(due=False)
    token = arb.claim_contact('vara')
    arb._owner = object()
    assert arb.claim_contact('vara', token) is token
    arb.disabled.add('vara')
    assert arb.claim_contact('vara', token) is token


@pytest.mark.parametrize('blocked', ['fence', 'paused', 'line', 'lane', 'scan', 'owner', 'authority'])
def test_contact_hold_requires_all_live_guards(blocked):
    arb = bench()
    authority = arb.authority('chat')
    live = [True]
    fence = lambda: live[0]
    token = arb.claim_contact('vara', authority=authority, fence=fence)
    paused = lambda: blocked != 'paused'
    if blocked == 'fence': live[0] = False
    if blocked == 'line': arb.rig.line_state = 'unconfirmed'
    if blocked == 'lane': arb.active_protocols = lambda: ('ardop',)
    if blocked == 'scan': arb._scan_hold = ('vara', object())
    if blocked == 'owner': arb._owner = object()
    if blocked == 'authority': authority = ('other', 0)
    with arb.try_contact_hold('vara', token, authority=authority, fence=fence, paused=paused) as held:
        assert not held


def test_contact_hold_retains_token_and_retires_receive_boundaries():
    arb = bench()
    authority = arb.authority('chat')
    fence = lambda: True
    owner = ContactOwner(arb, 'vara')
    assert owner.admit(authority=authority, fence=fence)
    before = arb.receive_boundary
    with arb.try_contact_hold('vara', owner.token, authority=authority, fence=fence,
                              paused=lambda: True) as held:
        assert held and arb.contact_protocol == 'vara'
        assert arb.receive_boundary[0] == before[0] + 1
        with arb.try_hold() as idle_hold:
            assert not idle_hold
    assert arb.receive_boundary[0] == before[0] + 2
    assert owner.admit(authority=authority, fence=fence)
    owner.close()


def test_tx_fence_checked_again_for_committed_request():
    arb = bench(due=False)
    live = [True]
    request = TxRequest(0, 0, protocol='vara', authority=arb.authority('chat'),
                        control_fence=lambda: live[0])
    arb._validate(request)
    live[0] = False
    with pytest.raises(Refused, match='control fence'):
        arb._validate(request, committed=True)


def test_fence_expiry_during_last_admission_callback_cannot_key():
    from hfmodem.station.arbiter import _Flight
    arb = bench(due=False)
    arb.rig.key = lambda *args, **kwargs: pytest.fail('revoked request keyed')
    live = [True]
    request = TxRequest(0, 0, protocol='vara', authority=arb.authority('chat'),
                        control_fence=lambda: live[0],
                        admit=lambda: live.__setitem__(0, False))
    arb._validate(request)
    with pytest.raises(Refused, match='control fence'):
        arb._key(request, _Flight(request))


def test_station_burst_uses_modem_authority_and_fence():
    sent = []
    arb = bench()
    arb.submit = lambda req: sent.append(req) or 1
    station = SimpleNamespace(arbiter=arb, audio=SimpleNamespace(samples=0),
        rig=SimpleNamespace(dial=lambda: 7100000),
        cfg=SimpleNamespace(declared_power_w=10,
            rig=SimpleNamespace(power_w=80, power_clamp_w=10, ptt=SimpleNamespace(settle_s=0))))
    link = KestrelLink(station)
    authority = arb.authority('chat')
    fence = lambda: True
    modem = link.runtime.modem_factory(authority=authority, fence=fence)
    modem.start(Observer())
    transport = link.runtime.transport
    transport.hs = modem.station
    transport.tx_metadata(bandwidth_hz=500)
    transport.tx(np.zeros(8))
    assert len(sent) == 1
    assert sent[0].emission.power_w == 10
    assert sent[0].authority == authority
    assert sent[0].control_fence is fence
    assert modem.authority == authority and modem.control_fence is fence
    transport.feed(0, np.ones(12))
    old_epoch = transport.receive_epoch
    transport.retuned()
    assert transport.receive_epoch != old_epoch
    delivered = []
    receiver = SimpleNamespace(on_rx_stream=lambda *a, **kw: delivered.append(a))
    assert transport.next_rx_burst(.01, receiver) is None
    assert transport.frames.empty()
    assert delivered == []


def test_owner_retune_runs_at_paused_boundary_and_rechecks_validity():
    runtime = StationVaraRuntime(SimpleNamespace(station=SimpleNamespace(
        arbiter=bench(), audio=SimpleNamespace(samples=0))))
    runtime.owner.step = lambda timeout: runtime.transport.stop.wait(.001)
    runtime.start()
    try:
        result = runtime.retune(lambda: (threading.current_thread().name, runtime.paused),
                                lambda: True, timeout=2)
        assert result == ('station-vara-owner', True)
        assert not runtime.paused
        with pytest.raises(ConnectionAbortedError):
            runtime.retune(lambda: pytest.fail('stale callback ran'), lambda: False, timeout=2)
    finally:
        runtime.stop()


@pytest.mark.parametrize('blocked', ['disabled', 'closed'])
def test_bound_contact_renewal_keeps_admission_checks(blocked):
    arb = bench()
    authority, fence = arb.authority('chat'), lambda: True
    token = arb.claim_contact('vara', authority=authority, fence=fence)
    assert token is not None
    try:
        if blocked == 'disabled':
            arb.disabled.add('vara')
        else:
            arb.close()
        assert arb.claim_contact('vara', token, authority=authority, fence=fence) is None
        assert arb.claim_contact('vara', token) is None
    finally:
        arb.release_contact('vara', token)
        arb.close()
