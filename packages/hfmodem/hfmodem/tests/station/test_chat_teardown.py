# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace

import pytest

from hfmodem.core.operations import LeaseTimeout
from hfmodem.station import hosts
from hfmodem.station.chat_transport import Events
from hfmodem.tests.station import test_process

fake_claim = test_process.fake_claim
station = test_process.station


@pytest.mark.parametrize('failed', [False, True])
def test_chat_retires_before_hosts_and_lanes_and_failure_reaches_fatal(station, monkeypatch, failed):
    events = station.calls
    station.audio = test_process.FakeAudio(events)
    station.rig = test_process.FakeRig(events)
    station.lanes = {'fake': SimpleNamespace(stop=lambda: events.append('lane-stop'))}

    def close_chat():
        assert not station._poll_stop.is_set()
        assert not station.sessions.accepting
        assert events == ['chat-revoke', 'arbiter-stop']
        events.append('chat-close')
        if failed:
            raise RuntimeError('chat worker deadline; attachment retained')

    def stop_hosts(_):
        assert station._poll_stop.is_set()
        assert events[-1] == 'chat-close'
        events.append('hosts-stop')

    station.service = SimpleNamespace(events=Events(), chat=SimpleNamespace(
        revoke=lambda: events.append('chat-revoke'), close=close_chat))
    station.arbiter = SimpleNamespace(stop=lambda *args: events.append('arbiter-stop'),
                                     close=lambda **kwargs: events.append('arbiter-close'))
    monkeypatch.setattr(hosts, 'stop_all', stop_hosts)
    if failed:
        with pytest.raises(LeaseTimeout, match='chat worker deadline'):
            station._teardown()
        assert station.state == 'failed' and events[-1][0] == 'fatal'
    else:
        station._teardown()
        assert station.state == 'observing'
    assert events[:8] == ['chat-revoke', 'arbiter-stop', 'chat-close', 'hosts-stop',
                          'lane-stop', 'arbiter-close', 'rig-close', 'audio-close']
