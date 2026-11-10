# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Kestrel-only one-frame overrides, using fake IO and no DSP or RF."""
import numpy as np
import pytest
from hfmodem.kestrel.host import vara_modem as VM
from hfmodem.kestrel.vara.vara_arq import VaraState
from .test_cq import session_mock

BEACON = '20AEN63BCW9SSJ'
COMMANDS = ('CQFRAMEFORCE W9SSJ 500', f'BEACONFORCE {BEACON}')

@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(VM, 'synth_frame', lambda _: np.zeros(8))
    monkeypatch.setattr(VM, 'synth_beacon', lambda _: np.zeros(8))
    session, replies, modem = session_mock()
    sent = []
    modem._transport.tx = sent.append
    return session, replies, modem, sent

@pytest.mark.parametrize('command', COMMANDS)
def test_force_is_admitted_once_and_does_not_change_busy_or_next_frame(rig, command):
    session, replies, modem, sent = rig
    modem._channel_busy = True
    session._dispatch(command)
    assert replies == ['OK'] and not sent
    modem.poll()
    assert len(sent) == 1 and modem._channel_busy is True
    assert not modem._ptt_active and modem._keyed_clean is None
    for ordinary in ('CQFRAME W9SSJ 500', None):
        if ordinary:
            session._dispatch(ordinary)
        else:
            modem.send_beacon(BEACON)
        modem.poll()
        assert len(sent) == 1  # neither later kind inherits the override

@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('guard', ['linked', 'connecting', 'closing', 'pending', 'intent', 'ptt', 'keying'])
def test_force_preserves_admission_guards(rig, command, guard):
    session, replies, modem, sent = rig
    if guard in ('linked', 'connecting'):
        modem.station.state = VaraState.CONNECTED if guard == 'linked' else VaraState.CONNECTING
    elif guard == 'pending':
        modem._incoming_pending_at = 1
    elif guard == 'intent':
        modem._intent_active = True
    elif guard == 'ptt':
        modem.station.io.report_ptt(True)
    elif guard == 'keying':
        modem._keyed_clean = object()
    else:
        modem._admission_closing = True
    session._dispatch(command)
    assert replies[-1] == 'WRONG'
    assert modem._events.empty() and not sent

@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('guard', ['linked', 'pending', 'ptt', 'abort', 'disconnect', 'stop'])
def test_force_rechecks_after_enqueue_and_cancels_stale_frames(rig, command, guard):
    session, replies, modem, sent = rig
    session._dispatch(command)
    if guard == 'linked':
        modem.station.state = VaraState.CONNECTED
    elif guard == 'pending':
        modem._incoming_pending_at = 1
    elif guard == 'ptt':
        modem.station.io.report_ptt(True)
    elif guard == 'abort':
        modem.abort()
    elif guard == 'disconnect':
        modem.disconnect()
    else:
        modem.stop()
    modem.poll()
    assert not sent

@pytest.mark.parametrize('command', [
    'BEACONFORCE', 'BEACONFORCE arbitrary text', 'BEACONFORCE 20AEN63BCW9SSJ extra',
    'BEACONFORCE 20AZZ99ZZW9SSJ', 'BEACONFORCE 20AEN63BCééé',
    'CQFRAMEFORCE W9SSJ 0', 'CQFRAMEFORCE W9SSJ 501',
    'CQFRAMEFORCE W9SSJ', 'CQFRAMEFORCE W9SSJ 500 extra'])
def test_invalid_force_payload_never_enters_owner_queue(rig, command):
    session, replies, modem, sent = rig
    session._dispatch(command)
    assert replies[-1] == 'WRONG'
    assert modem._events.empty() and not modem._cq_pending and not sent

@pytest.mark.parametrize('command', COMMANDS)
def test_force_does_not_replace_transport_or_hardware_refusal_path(rig, command):
    session, replies, modem, sent = rig
    attempted = []
    modem._transport.tx = lambda x: attempted.append(x)
    modem._transport.tx_went_out = lambda: False
    modem._channel_busy = True
    session._dispatch(command)
    modem.poll()
    assert len(attempted) == 1 and not modem.station.io.tx_went_out()
    assert replies[-3:] == ['WRONG', 'PTT OFF', 'CANCELPENDING']
    assert not modem._ptt_active

@pytest.mark.parametrize('command', COMMANDS)
def test_nonimplementing_modem_rejects_without_fallback(rig, command):
    session, replies, modem, sent = rig
    modem.supports_frame_force = False
    session._dispatch(command)
    assert replies == ['WRONG'] and modem._events.empty() and not sent

@pytest.mark.parametrize('refused', [False, True])
def test_real_client_crosses_host_queue_for_busy_forced_advanced_beacon(rig, refused):
    """Command/replies cross both production layers; only sockets/audio are fake."""
    from hfmodem.contact.varac import Client, Unsupported
    from hfmodem.tests.contact.test_varac import Socket
    session, replies, modem, sent = rig
    command = Socket(lambda raw: session._dispatch(raw.decode().rstrip('\r')))
    data = Socket()
    clock = [0.]
    def reply(line):
        replies.append(line)
        command.incoming.append((line + '\r').encode())
    session._send_cmd = reply
    def ready(sockets, timeout):
        clock[0] += timeout
        # Admission OK is delivered before the owner services the queued frame.
        if not command.incoming:
            modem.poll()
        return [s for s in sockets if s.incoming]
    client = Client(command, data, clock=lambda: clock[0], ready=ready)
    client.mycall, client.bandwidth = 'W9SSJ', 500
    client.busy = modem._channel_busy = True
    modem._transport.tx_went_out = lambda: not refused
    steps = client.forced_beacon_steps(BEACON)
    def finish():
        for _ in range(30):
            try:
                next(steps)
            except StopIteration:
                return
            client.poll()
        pytest.fail('forced beacon did not finish within bounded fake-owner steps')
    if refused:
        with pytest.raises(Unsupported, match='unavailable or rejected'):
            finish()
        assert replies.index('WRONG') < replies.index('CANCELPENDING')
    else:
        finish()
        assert 'WRONG' not in replies
    assert command.sent == [f'BEACONFORCE {BEACON}\r'.encode()]
    assert len(sent) == 1  # attempted waveform; refusal is not transmitted evidence
    assert not client.pending and not client.ptt and not client.calling
    assert client.busy and modem._channel_busy  # no persistent occupancy bypass
