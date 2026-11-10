# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Owner-settled closing admission; fake transport, no DSP or RF."""
import queue

import pytest

from hfmodem.kestrel.vara.vara_arq import VaraState
from .test_host_frame_force import rig as rig, BEACON as BEACON


def queued_abort_listen(rig):
    host, replies, modem, sent = rig
    modem.station.state = VaraState.CONNECTED
    modem.station.role = 'initiator'
    modem._connected = True
    host.modem_connected('W9SSJ', 'KL7QR', '500')
    host._dispatch('ABORT')
    host._dispatch('LISTEN OFF')
    host._dispatch('LISTEN CQ')
    assert modem._admission_closing
    return host, replies, modem, sent


def test_force_after_owner_consumes_abort_and_listen(rig):
    host, replies, modem, sent = queued_abort_listen(rig)
    host._dispatch(f'BEACONFORCE {BEACON}')
    assert replies[-1] == 'WRONG' and not sent
    modem.poll()
    assert replies.count('DISCONNECTED') == 1
    assert not modem._admission_closing
    modem._channel_busy = True
    host._dispatch(f'BEACONFORCE {BEACON}')
    assert replies[-1] == 'OK'
    modem.poll()
    assert len(sent) == 1 and modem._channel_busy


@pytest.mark.parametrize('guard', ['ptt', 'connect'])
def test_settled_abort_does_not_bypass_new_active_guard(rig, guard, monkeypatch):
    host, replies, modem, sent = queued_abort_listen(rig)
    modem.poll()
    if guard == 'ptt':
        modem.station.io.report_ptt(True)
    else:
        host._dispatch('CONNECT W9SSJ K0EPI')
    host._dispatch(f'BEACONFORCE {BEACON}')
    assert replies[-1] == 'WRONG' and not sent


def test_pending_owner_commands_keep_closing_latch(rig):
    host, replies, modem, sent = rig
    # Production queue is 64. A larger fake queue pins the owner's 64-command
    # budget independently of capacity and protects future capacity changes.
    modem._events = queue.Queue()
    modem._admission_closing = True
    for _ in range(64):
        modem.set_chat(True)
    modem._events.put(('disconnect',))
    modem.poll()
    assert modem._admission_closing and not modem._events.empty()
    host._dispatch(f'BEACONFORCE {BEACON}')
    assert replies[-1] == 'WRONG' and not sent
    modem.poll()
    assert not modem._admission_closing


@pytest.mark.parametrize('role', ['initiator', None])
def test_live_or_refused_disconnect_cannot_admit_beacon(rig, role, monkeypatch):
    host, replies, modem, sent = rig
    modem.station.state = VaraState.CONNECTED
    modem.station.role = role
    monkeypatch.setattr(modem.station, 'disconnect', lambda **kw: None)
    host._dispatch('DISCONNECT')
    modem.poll()
    assert modem.station.state == VaraState.CONNECTED
    host._dispatch(f'BEACONFORCE {BEACON}')
    assert replies[-1] == 'WRONG' and not sent


def test_rejection_log_has_guard_reason_without_payload(rig):
    host, replies, modem, sent = rig
    logged = []
    host._log = lambda *entry: logged.append(entry)
    modem._admission_closing = True
    host._dispatch(f'BEACONFORCE {BEACON}')
    assert replies == ['WRONG']
    logged = [entry for entry in logged if entry[0] == 'rejected']
    assert logged == [('rejected', 'BEACONFORCE: VARA contact is already active: '
                       'contact close admission pending')]
    assert BEACON not in str(logged)
