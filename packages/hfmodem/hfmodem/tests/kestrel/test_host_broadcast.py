# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Host broadcast admission/completion using fake transport, no DSP or RF."""
from types import SimpleNamespace
import numpy as np
import pytest
from hfmodem.contact.broadcast import Broadcast, encode_ui
from hfmodem.kestrel.host.vara_modem import _HostIO
from hfmodem.kestrel.host.server import VaraServer
from hfmodem.kestrel.vara.vara_arq import VaraState
from .test_cq import session_mock


@pytest.fixture
def rig(monkeypatch):
    from hfmodem.kestrel.tx import broadcast
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda _: np.zeros(8))
    session, replies, modem = session_mock()
    session._dispatch('MYCALL W9SSJ')
    modem.poll()
    replies.clear()
    sent = []
    modem._transport.tx = sent.append
    return session, replies, modem, sent


def command(token='one', kind=1, source='W9SSJ', info=b'hello'):
    return f'BROADCAST {token} {kind} ' + encode_ui(source, 'ALL', info, command=kind).hex()


def outcomes(replies):
    return [r for r in replies if r.startswith('BROADCAST')]


@pytest.mark.parametrize('kind', [0, 1])
def test_admission_precedes_owner_completion_and_unkey(rig, kind):
    session, replies, modem, sent = rig
    session._dispatch('BROADCAST CAPS')
    assert replies == ['BROADCAST CAPS 0 1']
    replies.clear()
    session._dispatch(command(kind=kind))
    assert replies == ['BROADCAST one QUEUED'] and not sent
    modem.poll()
    assert len(sent) == 1 and not modem._ptt_active
    assert outcomes(replies) == ['BROADCAST one QUEUED', 'BROADCAST one SENT']
    assert replies.index('PTT OFF') < replies.index('BROADCAST one SENT')
    modem.poll()
    assert len(outcomes(replies)) == 2


@pytest.mark.parametrize('guard', ['busy', 'linked', 'incoming', 'keyed', 'abort', 'disconnect', 'stop', 'identity'])
def test_owner_rechecks_and_reports_failure_once(rig, guard):
    session, replies, modem, sent = rig
    session._dispatch(command())
    if guard == 'busy':
        modem._channel_busy = True
    elif guard == 'linked':
        modem.station.state = VaraState.CONNECTED
    elif guard == 'incoming':
        modem._incoming_pending_at = 1
    elif guard == 'keyed':
        modem._ptt_active = True
    elif guard == 'abort':
        modem.abort()
    elif guard == 'disconnect':
        modem.disconnect()
    elif guard == 'stop':
        modem.stop()
    else:
        modem.set_mycall(['K1ABC'])
    modem.poll()
    assert not sent
    assert outcomes(replies)[0] == 'BROADCAST one QUEUED'
    assert len(outcomes(replies)) == 2
    assert outcomes(replies)[1].startswith('BROADCAST one FAILED ')
    assert modem._broadcast_pending is None


def test_transport_refusal_is_not_sent(rig):
    session, replies, modem, sent = rig
    modem._transport.tx_went_out = lambda: False
    session._dispatch(command())
    modem.poll()
    assert outcomes(replies) == ['BROADCAST one QUEUED', 'BROADCAST one FAILED transport refused']
    assert not modem._ptt_active


@pytest.mark.parametrize('bad', ['BROADCAST', 'BROADCAST x 2 00', 'BROADCAST x 1 xyz',
                                 'BROADCAST x 1 00', command(source='K1ABC'),
                                 'BROADCAST x 1 ' + '00' * 257])
def test_malformed_or_other_source_never_queues(rig, bad):
    session, replies, modem, sent = rig
    session._dispatch(bad)
    assert replies == ['WRONG']
    assert modem._events.empty() and not sent


def test_single_pending_and_no_token_reuse(rig):
    session, replies, modem, sent = rig
    session._dispatch(command())
    session._dispatch(command('two'))
    assert replies[-1] == 'WRONG'
    modem.poll()
    session._dispatch(command())
    assert replies[-1] == 'WRONG' and len(sent) == 1


def test_packet_fit_is_validated_before_queued(rig):
    session, replies, modem, sent = rig
    # Host envelope encoder accepts this but the PHY cannot preserve hyphenated identity.
    session._dispatch('MYCALL X-W9SSJ')
    modem.poll()
    replies.clear()
    session._dispatch(command(source='X-W9SSJ'))
    assert replies == ['WRONG'] and modem._events.empty() and not sent


def test_raw_kiss_routes_same_queue_without_fabricated_admission(rig):
    session, replies, modem, sent = rig
    server = object.__new__(VaraServer)
    server._session = session
    server._kiss_frame(0, encode_ui('W9SSJ', 'ALL', b'hello', command=0))
    assert replies == []
    modem.poll()
    assert len(sent) == 1 and outcomes(replies) == []


def test_received_ui_never_enters_contact_data_stream(rig):
    session, replies, modem, sent = rig
    delivered = []
    session._kiss = SimpleNamespace(deliver_frame=lambda *args: delivered.append(args))
    message = Broadcast(1, 'K1ABC', 'ALL', b'hello')
    _HostIO(modem, modem._transport).cq(message)
    assert delivered == [(1, message.encode())] and not sent and not replies


def test_owner_completion_during_admission_is_still_after_queued(rig, monkeypatch):
    session, replies, modem, sent = rig
    original = modem.send_broadcast
    def immediate(*args, **kwargs):
        original(*args, **kwargs)
        modem.poll()
    monkeypatch.setattr(modem, 'send_broadcast', immediate)
    session._dispatch(command())
    assert replies[0] == 'BROADCAST one QUEUED'
    assert outcomes(replies) == ['BROADCAST one QUEUED', 'BROADCAST one SENT']


def test_transmit_exception_unkeys_and_fails_once(rig):
    session, replies, modem, sent = rig
    def fail(audio):
        raise OSError('fake playback failure')
    modem._transport.tx = fail
    session._dispatch(command())
    with pytest.raises(OSError, match='fake playback'):
        modem.poll()
    assert not modem._ptt_active
    assert outcomes(replies) == ['BROADCAST one QUEUED',
                                 'BROADCAST one FAILED transmit or unkey failed']
    modem.poll()
    assert len(outcomes(replies)) == 2


def test_kiss_delivery_preserves_command_and_escapes_binary():
    import threading
    from hfmodem.ax25 import kiss
    from hfmodem.kestrel.host.kiss import KissPort
    sent = []
    class Client:
        def settimeout(self, seconds):
            pass
        def sendall(self, data):
            sent.append(data)
    port = object.__new__(KissPort)
    port._clients = {Client(): None}
    port._lock = threading.Lock()
    port.deliver_frame(1, b'\xc0\xdbhello')
    assert list(kiss.Parser().feed(sent[0])) == [(0, 1, b'\xc0\xdbhello')]


def test_successful_tx_but_failed_unkey_never_reports_sent(rig):
    session, replies, modem, sent = rig
    modem._transport.unkey_confirmed = lambda: False
    session._dispatch(command())
    with pytest.raises(RuntimeError, match='unkey was not confirmed'):
        modem.poll()
    assert len(sent) == 1
    assert outcomes(replies) == ['BROADCAST one QUEUED',
                                 'BROADCAST one FAILED transmit or unkey failed']
    assert modem._broadcast_pending is None


@pytest.mark.parametrize('info', [b'hello', b'x' * 70])
def test_broadcast_metadata_uses_actual_500hz_phy_not_configured_session(rig, info):
    session, replies, modem, sent = rig
    modem.set_bandwidth('2300')
    modem.poll()
    notes = []
    modem._transport.tx_metadata = lambda **fields: notes.append(fields)
    session._dispatch(command(info=info))
    modem.poll()
    assert len(notes) == 1
    assert notes[0]['bandwidth_hz'] == 500
    assert notes[0]['source'] == 'W9SSJ' and notes[0]['destination'] == 'ALL'
    level = notes[0]['host_bitrate_level']
    assert notes[0]['speed_level'] == level and level in (1, 2, 4)
    assert notes[0]['modulation'] == ('DBPSK' if level == 4 else 'index modulation')
    assert outcomes(replies)[-1] == 'BROADCAST one SENT'


def test_audio_unkey_confirmation_does_not_infer_from_key_call_return(tools):
    from kestrel_connect import AudioVaraIO
    audio = object.__new__(AudioVaraIO)
    audio.rig = SimpleNamespace(retired=False, keyed=False, line_state=lambda: False)
    audio._timing_key_active = False
    assert audio.unkey_confirmed()
    audio._timing_key_active = True
    assert not audio.unkey_confirmed()
    audio._timing_key_active = False
    audio.rig.line_state = lambda: True
    assert not audio.unkey_confirmed()
    audio.rig.line_state = lambda: None
    assert not audio.unkey_confirmed()
