# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Station chat tune and identification adapters use only fake CAT and TX."""
from contextlib import contextmanager
from types import SimpleNamespace
import time
import threading

import numpy as np
import pytest

from hfmodem.contact.varac import HostError
from hfmodem.core import band, cwid
from hfmodem.station.arbiter import Identity, TxArbiter, TxResult
from hfmodem.station.chat import Attachment
from hfmodem.station.contact import ContactOwner


class Fence:
    live = True
    def __call__(self): return self.live
    def check(self):
        if not self.live: raise HostError('revoked')


def attachment():
    events = []
    fence = Fence()
    cat = SimpleNamespace(clock=lambda: 0, frequency=7100000,
        set_mode=lambda *args: events.append(('mode', args)) or True,
        receiver_filters=lambda: (SimpleNamespace(mode='PKTUSB', passband_hz=3000), None))
    cat.freq = lambda: cat.frequency
    @contextmanager
    def lease():
        events.append('lease')
        yield True
    rig = SimpleNamespace(model='generic', line_state='low', cat=cat,
                          operations=SimpleNamespace(lease=lease),
                          hold=lambda deadline: lease(), dial=lambda: cat.frequency)
    def tune(centre):
        events.append(('tune', centre))
        cat.frequency = band.dial_hz(centre)
    rig.tune = tune
    arb = TxArbiter(SimpleNamespace(samples=0), rig, attended=lambda sid: sid == 'chat')
    item = Attachment.__new__(Attachment)
    item.station = SimpleNamespace(arbiter=arb, rig=rig,
        cfg=SimpleNamespace(declared_power_w=17,
            rig=SimpleNamespace(power_w=80, power_clamp_w=17, ptt=SimpleNamespace(settle_s=.04))))
    item._authority = arb.authority('chat')
    contact = ContactOwner(arb, 'vara')
    item.fence = fence
    item.host = SimpleNamespace(_modem=SimpleNamespace(control_fence=fence))
    item.client = SimpleNamespace(bandwidth='2300')
    item.runtime = SimpleNamespace(paused=True,
        transport=SimpleNamespace(contact=contact, receiving=False,
                                  retuned=lambda: events.append('retuned')))
    item.deadline = None
    item.chat = SimpleNamespace(session=SimpleNamespace(id='chat'))
    item._say = events.append
    return item, events


def test_idle_tune_converts_dial_and_verifies_mode_under_resource_leases():
    item, events = attachment()
    result = item._tune_boundary(14074000)
    assert result == dict(verified=True, atu_match_known=False)
    assert ('tune', band.centre_hz(14074000)) in events
    assert ('mode', ('PKTUSB', 3000)) in events
    assert events.count('lease') == 2
    assert events[-1] == 'retuned'


def test_contact_tune_keeps_same_authority_fence_and_requires_paused_owner():
    item, events = attachment()
    contact = item.runtime.transport.contact
    assert contact.admit(authority=item._authority, fence=item.host._modem.control_fence)
    token = contact.token
    item.runtime.paused = False
    with pytest.raises(HostError, match='admission'):
        item._tune_boundary(7105000)
    assert events == []
    item.runtime.paused = True
    item._tune_boundary(7105000)
    assert contact.token is token
    assert contact.admit(authority=item._authority, fence=item.host._modem.control_fence)


def test_failed_readback_still_retires_old_frequency_capture():
    item, events = attachment()
    item.station.rig.cat.receiver_filters = lambda: (SimpleNamespace(mode='USB', passband_hz=3000), None)
    with pytest.raises(HostError, match='readback'):
        item._tune_boundary(7105000)
    assert events[-1] == 'retuned'


@pytest.mark.parametrize('bad_readback', [None, 'roof', 'shift'])
def test_band_stack_restores_and_verifies_ftdx10_profile_with_fake_cat(monkeypatch, bad_readback):
    from hfmodem.station import chat
    from hfmodem.core.ftdx10 import filter_commands
    from hfmodem.core.rigs import RIGS
    item, events = attachment()
    clock = SimpleNamespace(now=0.)
    def sleep(seconds): clock.now += seconds
    monkeypatch.setattr(chat, 'time', SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    rig = item.station.rig
    rig.model = 'ftdx10'
    rig.cat.write = lambda command: events.append(('raw', command))
    rig.cat.read = lambda command, prefix: '0+0000' if bad_readback == 'shift' else '0+0500'
    rig.cat.receiver_filters = lambda: (SimpleNamespace(mode='PKTUSB', passband_hz=4000),
                                        3000 if bad_readback == 'roof' else 12000)
    if bad_readback is not None:
        with pytest.raises(HostError, match='roofing/shift'):
            item._tune_boundary(14074000)
    else:
        assert item._tune_boundary(14074000)['verified']
    assert ('mode', ('PKTUSB', 0)) in events
    assert [e[1] for e in events if isinstance(e, tuple) and e[0] == 'raw'] == list(
        filter_commands(RIGS['ftdx10']))
    assert clock.now >= 1.
    assert events[-1] == 'retuned'


def test_revoked_tune_never_moves_fake_cat():
    item, events = attachment()
    item.fence.live = False
    with pytest.raises(HostError, match='revoked'):
        item._tune_boundary(7105000)
    assert events == []


def test_ident_uses_station_audio_geometry_power_and_captured_authority():
    item, events = attachment()
    audio = np.ones(4)
    item.station.arbiter.identity = SimpleNamespace(burst=lambda: audio,
        note_sent=lambda: pytest.fail('adapter bypassed arbiter completion accounting'))
    sent = []
    item.station.arbiter.submit_result = lambda req: sent.append(req) or TxResult('sent', 'committed', end=123)
    for delay in item.identify('500', False):
        time.sleep(delay)
    assert len(sent) == 1
    req = sent[0]
    assert req.protocol == 'ident' and req.authority == item._authority
    assert req.control_fence is item.host._modem.control_fence
    assert req.audio is audio
    assert req.emission.power_w == 17
    assert req.emission.bandwidth_hz == cwid.BANDWIDTH_HZ
    assert (req.emission.audio_hi_hz + req.emission.audio_lo_hz) / 2 == cwid.TONE_HZ


@pytest.mark.parametrize('result', [TxResult('refused', 'waiting', reason='test refusal'),
                                  TxResult('sent', 'committed'), TxResult('sent', 'committed', end=0),
                                  TxResult('invalidated', 'committed', end=123)])
def test_ident_without_positive_completion_cannot_claim_success(result):
    item, events = attachment()
    item.station.arbiter.identity = SimpleNamespace(burst=lambda: np.ones(4))
    item.station.arbiter.submit_result = lambda req: result
    with pytest.raises(HostError, match='completion evidence'):
        for delay in item.identify('500', False):
            time.sleep(delay)


@pytest.mark.parametrize('receiving,expected', [(False, .5), (True, 8.)])
def test_ident_preserves_peer_gap_and_bounded_channel_observation(monkeypatch, receiving, expected):
    from hfmodem.station import chat
    item, events = attachment()
    clock = SimpleNamespace(now=0.)
    class Event:
        flag = False
        def set(self): self.flag = True
        def is_set(self): return self.flag
        def wait(self, seconds):
            clock.now += seconds
            return self.flag
    monkeypatch.setattr(chat, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(chat, 'threading', SimpleNamespace(Event=Event, Thread=threading.Thread))
    item.runtime.transport.receiving = receiving
    item.station.arbiter.identity = SimpleNamespace(burst=lambda: np.ones(4))
    sent_at = []
    item.station.arbiter.submit_result = lambda req: sent_at.append(clock.now) or TxResult('sent', 'committed', end=123)
    for delay in item.identify('500', True):
        time.sleep(.001)
    assert len(sent_at) == 1
    assert expected <= sent_at[0] < expected + .1


def test_ident_runs_real_stateful_arbiter_api_with_synthetic_card_and_line():
    from hfmodem.tests.core.test_arbiter import Card, LineRig
    item, events = attachment()
    card = Card()
    rig = LineRig(card)
    rig.dial = lambda: 7100000
    identity = Identity('W9SSJ')
    arbiter = TxArbiter(card, rig, identity=identity, attended=lambda sid: sid == 'chat')
    item.station.arbiter, item.station.rig = arbiter, rig
    item._authority = arbiter.authority('chat')
    previous_identification = identity._last
    try:
        for delay in item.identify('500', False):
            time.sleep(delay)
        assert arbiter.sent == 1
        assert identity._last > previous_identification
        assert rig.line_state == 'low'
        assert [call[0] for call in rig.calls] == ['key', 'unkey']
        assert rig.calls[0][2]['attended'] is True
        records = []
        while not arbiter.records.empty():
            records.append(arbiter.records.get_nowait())
        assert any(record['authority'] == item._authority and record['protocol'] == 'ident'
                   and record['outcome'] == 'released' for record in records)
    finally:
        arbiter.close()


@pytest.mark.parametrize('backend', ['yaesu', 'rigctld'])
@pytest.mark.parametrize('clamp,initial', [(25, 100), (150, 200)])
def test_station_power_callback_scales_reads_clamps_writes_and_checks_fence(backend, clamp, initial):
    from hfmodem.core.config import Rig, StationConfig
    from hfmodem.tests.contact.test_varac import Desk
    item, events = attachment()
    item.station.cfg = StationConfig(station=None, audio=None, protocols={}, rig=Rig(cat=backend, max_watts=200, power_clamp_w=clamp))
    watts = [float(initial)]
    cat = item.station.rig.cat
    cat.set_power = lambda value: (events.append(('power', value)), watts.__setitem__(0, value))
    cat.read = lambda command, prefix: f'{int(watts[0]):03d}'
    cat.command = lambda command: f'get_level: RFPOWER\n{watts[0] / 200:g}\nRPRT 0\n'
    desk = Desk()
    item.session = desk.session
    desk.session.power = item.power
    desk.session.power_clamp_w = item.station.cfg.rig.power_clamp_w
    item.client.ptt = False
    item._hold_power()
    assert watts == [clamp] and ('power', clamp) in events
    assert item.power(None) == clamp
    desk.session.command('/power 80')
    assert watts == [min(80, clamp)] and desk.events[-1]['set_w'] == min(80, clamp)
    item.fence.live = False
    with pytest.raises(HostError, match='revoked'):
        item.power(10)
    assert ('power', 10) not in events


def test_station_power_start_without_cat_leaves_power_alone():
    from hfmodem.core.config import Rig, StationConfig
    item, _ = attachment()
    item.station.cfg = StationConfig(station=None, audio=None, protocols={}, rig=Rig(cat='none', power_clamp_w=25))
    item.station.rig.cat = None
    item.session = SimpleNamespace(set_power=lambda *args: pytest.fail('no CAT power write'))
    item._hold_power()


def power_attachment(backend='yaesu'):
    from hfmodem.core.config import Rig, StationConfig
    from hfmodem.tests.contact.test_varac import Desk
    item, events = attachment()
    item.station.cfg = StationConfig(station=None, audio=None, protocols={},
        rig=Rig(cat=backend, max_watts=200, power_clamp_w=25))
    item.client.ptt = False
    cat = item.station.rig.cat
    cat.set_power = lambda watts: events.append(('power', watts))
    cat.read = lambda *args: events.append('read') or '025'
    cat.command = lambda *args: events.append('read') or '0.125\n'
    desk = Desk()
    item.session = desk.session
    desk.session.power = item.power
    desk.session.power_clamp_w = 25
    return item, events, desk


@pytest.mark.parametrize('watts', [None, 10])
def test_station_power_refused_cat_lease_never_reads_or_writes(watts):
    item, events, _ = power_attachment()
    @contextmanager
    def refused(deadline):
        yield False
    item.station.rig.hold = refused
    with pytest.raises(HostError, match='CAT lease refused'):
        item.power(watts)
    assert events == ['lease']


@pytest.mark.parametrize('watts', [None, 10])
def test_station_power_closed_operation_gate_never_acquires_cat_lease(watts):
    from hfmodem.core.operations import OperationGate, ResourceClosed
    item, events, _ = power_attachment()
    item.station.rig.operations = OperationGate()
    item.station.rig.operations.close(0)
    with pytest.raises(ResourceClosed):
        item.power(watts)
    assert events == []


@pytest.mark.parametrize('line,ptt', [('high', False), ('unknown', False), ('low', True)])
def test_station_power_write_requires_physical_low_and_host_unkeyed(line, ptt):
    item, events, _ = power_attachment()
    item.station.rig.line_state = line
    item.client.ptt = ptt
    with pytest.raises(HostError, match='verified low line'):
        item.power(10)
    assert events == ['lease', 'lease']


@pytest.mark.parametrize('lease', ['operations', 'cat'])
@pytest.mark.parametrize('watts', [None, 10])
def test_station_power_revocation_during_lease_acquisition_never_reads_or_writes(lease, watts):
    item, events, _ = power_attachment()
    @contextmanager
    def revoke(*args):
        item.fence.live = False
        yield True
    if lease == 'operations':
        item.station.rig.operations.lease = revoke
    else:
        item.station.rig.hold = revoke
    with pytest.raises(HostError, match='revoked'):
        item.power(watts)
    assert not any(isinstance(event, tuple) or event == 'read' for event in events)


def test_station_power_write_failure_is_reported_without_claiming_success():
    from hfmodem.core.rig import RigError
    item, events, desk = power_attachment()
    def fail(watts):
        raise RigError('synthetic write failure')
    item.station.rig.cat.set_power = fail
    desk.session.command('/power 10')
    assert desk.said[-1] == 'power not set: synthetic write failure'
    assert events == ['lease', 'lease']
    assert not any(event['kind'] == 'power' for event in desk.events)


@pytest.mark.parametrize('backend', ['yaesu', 'rigctld'])
@pytest.mark.parametrize('failure', ['error', 'malformed', 'negative', 'nonfinite'])
def test_station_power_readback_failure_reports_unknown_after_write(backend, failure):
    from hfmodem.core.rig import RigError
    item, events, desk = power_attachment(backend)
    def read(*args):
        if failure == 'error':
            raise RigError('synthetic read failure')
        return {'malformed': '?', 'negative': '-1', 'nonfinite': 'nan'}[failure]
    item.station.rig.cat.read = item.station.rig.cat.command = read
    desk.session.command('/power 10')
    assert ('power', 10) in events
    assert desk.events[-1] == dict(kind='power', requested_w=10, set_w=10, read_w=None, clamp_w=25)
    assert 'RFPOWER did not read back' in desk.said[-1]
    events.clear()
    item._hold_power()
    assert not any(isinstance(event, tuple) for event in events)
    assert 'station clamp is unchecked' in events[-1]


def test_retune_queue_full_is_a_reportable_session_error():
    import queue
    from hfmodem.kestrel.host.station_runtime import StationVaraRuntime
    from hfmodem.contact.chat_session import SessionAdapter
    from hfmodem.contact import chat_state as s
    from hfmodem.tests.contact.test_chat_controller import execute
    from hfmodem.tests.contact.test_varac import Desk
    runtime = StationVaraRuntime.__new__(StationVaraRuntime)
    runtime.transport = SimpleNamespace(stop=threading.Event())
    runtime._thread = SimpleNamespace(is_alive=lambda: True)
    runtime._calls = queue.Queue(maxsize=1)
    with pytest.raises(TimeoutError):
        runtime.retune(lambda: pytest.fail('cancelled retune ran'), lambda: True, timeout=0)
    desk = Desk()
    desk.session.band_tuner = lambda hz: runtime.retune(lambda: None, lambda: True, timeout=0)
    port = SessionAdapter(desk.session)
    _, result = execute(port, s.BandSet(40))
    assert result.state == 'failed' and 'queue' in result.detail
    runtime._service_calls()
    assert runtime._calls.empty()
