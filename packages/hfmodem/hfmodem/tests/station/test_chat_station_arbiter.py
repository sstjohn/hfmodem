# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Production arbiter construction observes the real attended Registry owner."""
from types import SimpleNamespace

import pytest

from hfmodem.kestrel.host.station_runtime import StationVaraRuntime
from hfmodem.station.process import Station
from hfmodem.station.sessions import Registry


@pytest.fixture(params=[True, False])
def ownership_preflight(request):
    return request.param


def station(*, ownership_preflight):
    # Deliberately bypass Station's device/recording constructor and ownership
    # acquisition. Only the production arbiter-construction method runs.
    item = Station.__new__(Station)
    item.receive = False
    item.ownership_preflight = ownership_preflight
    item.rig = SimpleNamespace(line_state='low')
    item.audio = SimpleNamespace(samples=0)
    item.cw_identity = None
    item.cfg = SimpleNamespace(station=SimpleNamespace(mycall='W9SSJ', id_interval_s=10,
                                                      id_mode='cw'),
                               audio=SimpleNamespace(tx_drive=1.),
                               rig=SimpleNamespace(ptt=SimpleNamespace(settle_s=0)))
    item.sessions = Registry(authority=lambda sid: item.arbiter.authority(sid))
    item.sessions.admit()
    item.build_arbiter()
    assert item.arbiter.defer_identification is (not ownership_preflight)
    item.arbiter.identity._last -= 11
    return item


@pytest.mark.parametrize('incoming', [False, True])
@pytest.mark.parametrize('idle', [False, True])
def test_station_built_arbiter_admits_due_id_attended_chat_incoming_and_outgoing(incoming, idle, ownership_preflight):
    item = station(ownership_preflight=ownership_preflight)
    assert item.arbiter._attended.__self__ is item.sessions
    session = item.sessions.start(kind='chat', lane='vara', attended=True,
                                  holds_rig=True, directory='unused', client='operator')
    item.sessions.running(session.id)
    runtime = StationVaraRuntime(SimpleNamespace(station=item))
    live = [True]
    fence = lambda: live[0]
    modem = runtime.modem_factory(authority=session.authority, fence=fence)
    try:
        if idle:
            item.arbiter.identity.note_sent()
            assert not item.arbiter.identity.due
            item.arbiter.identity._last -= item.cfg.station.id_interval_s + 1
        assert item.arbiter.identity.due
        assert item.arbiter.claim_contact('vara') is None
        if incoming:
            assert modem.station.io.admit_contact()
        else:
            modem.connect('W9SSJ', 'K7ABC')
        assert runtime.transport.contact.token is not None
        # Background ID is suppressed before an identification thread or rig
        # action can be started. The due debt itself remains untouched.
        assert not item.arbiter.identify_if_free(None)
        assert item.arbiter.identity.due
        live[0] = False
        assert not runtime.transport.admit_contact(modem)
        live[0] = True
        item.arbiter.stop(session.id)
        assert not runtime.transport.admit_contact(modem)
    finally:
        runtime.transport.contact.close()
        item.arbiter.close()


def test_station_built_arbiter_keeps_unattended_due_id_restriction(ownership_preflight):
    item = station(ownership_preflight=ownership_preflight)
    session = item.sessions.start(kind='test', lane='vara', attended=False,
                                  holds_rig=True, directory='unused', client='operator')
    runtime = StationVaraRuntime(SimpleNamespace(station=item))
    modem = runtime.modem_factory(authority=session.authority, fence=lambda: True)
    try:
        assert item.arbiter.identity.due
        assert not modem.station.io.admit_contact()
        assert runtime.transport.contact.token is None
    finally:
        item.arbiter.close()


def test_attended_nonchat_keeps_due_identification_policy_and_contact_refusal(ownership_preflight):
    from hfmodem.station.arbiter import Refused
    from hfmodem.tests.core.test_arbiter import req
    item = station(ownership_preflight=ownership_preflight)
    session = item.sessions.start(kind='monitor', lane='vara', attended=True,
                                  holds_rig=True, directory='unused', client='operator')
    try:
        assert item.sessions.attended(session.id)
        with pytest.raises(Refused, match='identification is due'):
            item.arbiter._policy(req(authority=session.authority))
        assert item.arbiter.claim_contact('vara', authority=session.authority,
                                          fence=lambda: True) is None
        assert not item.arbiter._attended(None)
    finally:
        item.arbiter.close()


@pytest.mark.parametrize('blocked', ['disabled', 'closed'])
def test_legacy_contact_renewal_survives_midcontact_admission_change(blocked, ownership_preflight):
    item = station(ownership_preflight=ownership_preflight)
    arb = item.arbiter
    arb.identity = None
    token = arb.claim_contact('vara')
    assert token is not None
    try:
        if blocked == 'disabled':
            arb.disabled.add('vara')
        else:
            arb.close()
        assert arb.claim_contact('vara', token) is token
        assert arb.claim_contact('vara', token, receive_epoch=arb.receive_boundary[0] + 1) is None
    finally:
        arb.release_contact('vara', token)
        arb.close()
