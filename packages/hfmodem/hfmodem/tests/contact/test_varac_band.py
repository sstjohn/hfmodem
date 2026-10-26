# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk


def test_band_change_keeps_beacon_schedule_and_scopes_heard_history():
    desk = Desk()
    desk.hear('SN 2.0', 'CQFRAME K7QB-1 2300')
    desk.do('/beacon 20')
    due = desk.session._beacon_due
    setup = []
    desk.session.band_tuner = lambda dial: setup.append((dial, desk.armed[-1], desk.sent[-1]))
    watch = SimpleNamespace(retune=lambda *args: setup.append(args), table=lambda: ([], 0), poll=lambda: [])
    desk.session.slots = watch
    desk.session.following = 1
    desk.session._transcripts = {1: ['old']}
    desk.do('/band 40')
    assert setup == [(7_105_000, False, 'LISTEN OFF'), (7_105_000, 7_105_000)]
    assert desk.flow.band == 40 and desk.flow.dial == 7_105_000
    assert desk.session.beacon == 20 and desk.session._beacon_due == due
    assert desk.session.following is None and desk.session._transcripts == {}
    assert desk.session.free_slot() == 1
    assert '14104.250 kHz' in desk.session.listing(desk.session.heard[0])
    assert desk.session.listing(desk.session.heard[0]).endswith('20 m')
    desk.do('/answer 1')
    assert desk.said[-1] == 'error: K7QB was heard on 20 m; /band 20 first'
    desk.session.command('/call K7QB 1')
    assert desk.host.client.bandwidth == 500, 'old-band bandwidth must not follow the call'


@pytest.mark.parametrize('busy', ['connected', 'ptt', 'pending', 'physical', 'steps'])
def test_band_change_refuses_a_link_or_any_keying(busy):
    desk = Desk()
    if busy == 'physical':
        desk.line = True
    elif busy == 'steps':
        desk.session.steps = iter([None])
    else:
        setattr(desk.host.client, busy, True)
    setup = []
    desk.session.band_tuner = setup.append
    desk.session.command('/band 40')
    assert setup == [] and desk.flow.band == 20
    assert desk.said[-1].startswith('error:')


def test_failed_band_setup_stays_disarmed_until_complete_retry():
    desk = Desk()
    def fail(dial):
        raise v.HostError('filters unverified')
    desk.session.band_tuner = fail
    desk.do('/band 40')
    assert desk.armed[-1] is False and desk.flow.dial is None
    desk.tick()
    assert desk.tunes == [] and desk.armed[-1] is False
    for command in ['/auto on', '/call K7QB', '/cq 1']:
        desk.session.command(command)
        assert desk.said[-1].startswith('error: band setup unverified')
        assert desk.armed[-1] is False
    setup = []
    desk.session.band_tuner = setup.append
    desk.do('/band 20')
    assert setup == [14_105_000] and desk.armed[-1] is True
    assert desk.flow.state is v.State.MONITORING


@pytest.fixture
def band_rig(tools, monkeypatch):
    import varac_chat
    monkeypatch.setenv('HFMODEM_RIG', 'ftdx10')
    path = Path(varac_chat.__file__).parent / 'tests' / 'fake_rigctld.py'
    spec = importlib.util.spec_from_file_location('band_fake_rigctld', path)
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    state, requests = dict(fake.RIG), []
    rig = SimpleNamespace(keyed=False)
    def batch(text):
        return fake.answer(text.splitlines(), state, rig.ignores, requests.append)
    rig.batch, rig.ignores = batch, ()
    return varac_chat, rig, state, requests


def test_band_stack_filters_are_restored_in_order_and_atu_is_advisory(band_rig):
    launch, rig, state, requests = band_rig
    said = []
    assert launch.change_band(rig, 14_105_000, 500, said.append, settle=0) == dict(
        verified=True, tuner_on=True, atu_match_known=False)
    assert state['MD0;'] == 'MD0C;' and state['SH0;'] == 'SH0021;'
    assert state['RF0;'] == 'RF06;' and state['IS0;'] == 'IS00+0500;'
    assert requests.index('+\\send_cmd FA014105000;') < requests.index('+\\send_cmd MD0C;')
    assert requests.index('+\\send_cmd RF01;') < requests.index('+\\send_cmd SH0021;')
    assert requests.index('+\\send_cmd SH0021;') < requests.index('+\\send_cmd SH0;')
    assert 'press TUNE on the rig' in said[-1]
    assert '+\\send_cmd AC;' in requests
    assert not any('AC00' in request or 'set_ptt' in request for request in requests)


@pytest.mark.parametrize('ignored', ['FA', 'MD0C', 'RF01', 'SH0021', 'IS00+0500'])
def test_failed_band_readback_is_fatal(band_rig, ignored):
    launch, rig, state, requests = band_rig
    rig.ignores = (ignored,)
    with pytest.raises(v.HostError, match='band change'):
        launch.change_band(rig, 14_105_000, 500, lambda line: None, settle=0)
    assert '+\\send_cmd AC;' not in requests


def test_pending_link_received_when_disabling_listen_cancels_band_change():
    desk = Desk()
    desk.host.replies['LISTEN OFF'] = ['OK', 'PENDING']
    setup = []
    desk.session.band_tuner = setup.append
    desk.session.command('/band 40')
    assert setup == [] and desk.armed[-1] is False
    assert desk.said[-1] == 'error: link or keying began while disabling listen; band change cancelled'


@pytest.mark.parametrize('reply,on', [('AC000;', False), ('AC001;', True), ('invalid', None)])
def test_atu_readback_never_claims_a_known_match(band_rig, reply, on):
    launch, rig, state, requests = band_rig
    state['AC;'] = reply
    result = launch.change_band(rig, 14_105_000, 500, lambda line: None, settle=0)
    assert result == dict(verified=True, tuner_on=on, atu_match_known=False)


@pytest.mark.parametrize('reply', [[], ['WRONG']])
def test_listen_disable_failure_leaves_band_change_disarmed(reply):
    desk = Desk()
    desk.flow.state = v.State.CONNECTED
    desk.host.replies['LISTEN OFF'] = reply
    setup = []
    desk.session.band_tuner = setup.append
    desk.session.command('/band 40')
    assert setup == [] and desk.armed[-1] is False
    assert desk.flow.state is v.State.CALLING and desk.session._band_fault
    if not reply:
        # A command timeout invalidates the whole host connection; it cannot be
        # reused for a retry or a frequency-only automatic return.
        assert desk.host.client.closed and desk.tunes == []
        assert desk.said[-1] == 'error: host connection closed during band setup; exit and restart the session'
        return
    desk.tick()
    assert desk.tunes == [] and desk.armed[-1] is False
    desk.host.replies['LISTEN OFF'] = ['OK']
    desk.do('/band 40')
    assert setup == [7_105_000] and desk.armed[-1] is True


def test_rig_batch_reads_short_tcp_fragments_until_peer_closes(tools, monkeypatch):
    from collections import deque
    from vara_rig_bridge import Rig
    import vara_rig_bridge
    fragments = deque([b'first\n', b'second\n', b'RPRT 0\n', b''])
    sent, closed = [], []
    sock = SimpleNamespace(sendall=sent.append, settimeout=lambda timeout: None,
                           recv=lambda count: fragments.popleft(), close=lambda: closed.append(True))
    monkeypatch.setattr(vara_rig_bridge.socket, 'create_connection', lambda *args, **kwargs: sock)
    rig = Rig.__new__(Rig)
    rig.addr = ('fixture-only', 0)
    assert rig.batch('+\\get_vfo\nq\n') == 'first\nsecond\nRPRT 0\n'
    assert sent == [b'+\\get_vfo\nq\n'] and closed == [True]


@pytest.mark.parametrize('level', ['VFO', 'AGC', 'PREAMP'])
def test_wrong_session_setup_readback_cannot_pass_on_good_filter_readback(band_rig, level):
    launch, rig, state, requests = band_rig
    batch = rig.batch
    def wrong(text):
        reply = batch(text)
        if level == 'VFO':
            return reply.replace('VFO: VFOA', 'VFO: VFOB')
        return reply.replace(f'get_level: {level}\n0\n', f'get_level: {level}\n6\n')
    rig.batch = wrong
    with pytest.raises(v.HostError, match='setup unverified'):
        launch.change_band(rig, 14_105_000, 500, lambda line: None, settle=0)
