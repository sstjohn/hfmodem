# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KISS transmit admission and keying with fake transport, no DSP or RF."""
from types import SimpleNamespace

import pytest

from hfmodem.kestrel.broadcast import Broadcast, encode_ui
from hfmodem.kestrel.host.kiss_tx import KissTxDisposition as Outcome, KissTxQueue
from hfmodem.kestrel.host.server import VaraServer
from hfmodem.kestrel.host.vara_modem import _HostIO
from hfmodem.kestrel.vara.vara_arq import VaraState
from .test_cq import session_mock
from .test_kiss_hostless import Clock


@pytest.fixture
def rig(monkeypatch):
    from hfmodem.kestrel.tx import broadcast
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda message: message)
    session, replies, modem = session_mock()
    clock, logs, sent = Clock(), [], []
    server = object.__new__(VaraServer)
    server._session = session
    server.kiss_tx_queue = KissTxQueue(clock=clock, log=lambda *args: logs.append(args))
    modem.kiss_tx_queue = server.kiss_tx_queue
    modem._clock = clock
    modem._transport.tx = sent.append
    session._dispatch('MYCALL W9SSJ')
    modem.poll()
    replies.clear()
    return SimpleNamespace(session=session, replies=replies, modem=modem, sent=sent,
                           server=server, clock=clock, logs=logs)


def submit(rig, *, kind=1, source='W9SSJ', info=b'hello'):
    rig.server._kiss_frame(kind, encode_ui(source, 'ALL', info, command=kind))


def bracket(message):
    from hfmodem.kestrel.tx.ui import packetize
    level, frames = packetize(message)
    bps = {1: 18, 2: 41, 4: 88}[level]
    return ['PENDING', 'PTT ON', 'BUSY ON',
            *([f'BITRATE ({level})  {bps} bps TX'] * len(frames)),
            'PTT OFF', 'CANCELPENDING', 'BUSY OFF']


@pytest.mark.parametrize('kind', [0, 1])
def test_admission_precedes_owner_transmission_and_unkey(rig, kind):
    submit(rig, kind=kind)
    assert not rig.replies and not rig.sent
    assert len(rig.server.kiss_tx_queue) == 1
    assert rig.modem.service_kiss_queue() is None
    rig.clock.now = 2
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert len(rig.sent) == 1 and not rig.modem._ptt_active
    assert rig.replies == bracket(rig.sent[0])
    assert rig.modem.service_kiss_queue() is None
    assert len(rig.sent) == 1


@pytest.mark.parametrize('guard', ['busy', 'linked', 'incoming', 'keyed', 'stop'])
def test_owner_rechecks_gates_and_keeps_queue(rig, guard):
    submit(rig)
    if guard == 'busy':
        rig.modem._channel_busy = True
    elif guard == 'linked':
        rig.modem.station.state = VaraState.CONNECTED
    elif guard == 'incoming':
        rig.modem._incoming_pending_at = 1
    elif guard == 'keyed':
        rig.modem._ptt_active = True
    else:
        rig.modem.stop()
    rig.clock.now = 2
    assert rig.modem.service_kiss_queue() is None
    assert not rig.sent and len(rig.server.kiss_tx_queue) == 1
    if guard == 'stop':
        rig.modem.poll()
        assert rig.modem.observer is None
        assert len(rig.server.kiss_tx_queue) == 1
    else:
        rig.modem.station.state = VaraState.DISCONNECTED
        rig.modem._channel_busy = rig.modem._ptt_active = False
        rig.modem._incoming_pending_at = None
        rig.modem.poll()
        rig.clock.now = 2
        assert rig.modem.service_kiss_queue() is Outcome.SENT
        assert len(rig.sent) == 1 and not rig.server.kiss_tx_queue


@pytest.mark.parametrize('command', ['abort', 'disconnect'])
def test_eligible_kiss_frame_survives_owner_command(rig, command):
    submit(rig)
    rig.clock.now = 2
    getattr(rig.modem, command)()
    if command == 'disconnect':
        assert rig.modem.service_kiss_queue() is None
        assert not rig.sent and len(rig.server.kiss_tx_queue) == 1
    rig.modem.poll()
    assert rig.modem._events.empty()
    assert not rig.sent and len(rig.server.kiss_tx_queue) == 1
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert [message.info for message in rig.sent] == [b'hello']
    assert not rig.server.kiss_tx_queue


def test_transport_refusal_is_not_sent(rig):
    rig.modem._transport.tx_went_out = lambda: False
    submit(rig)
    rig.clock.now = 2
    with pytest.raises(OSError, match='failed after keying'):
        rig.modem.service_kiss_queue()
    assert not rig.modem._ptt_active and not rig.server.kiss_tx_queue
    assert sum('PARTIAL' in message for _, message in rig.logs) == 1
    assert rig.modem.service_kiss_queue() is None


@pytest.mark.parametrize('command', ['BROADCAST', 'BROADCAST CAPS',
    'BROADCAST one 0 ' + encode_ui('W9SSJ', 'ALL', b'hello', command=0).hex(),
    'BROADCAST two 1 ' + encode_ui('K1ABC', 'ALL', b'hello', command=1).hex(),
    'broadcast caps'])
def test_retired_commands_match_unknown_command(rig, command):
    rig.session._dispatch('UNKNOWN')
    assert rig.replies == ['WRONG']
    rig.replies.clear()
    rig.session._dispatch(command)
    assert rig.replies == ['WRONG']
    assert rig.modem._events.empty() and not rig.server.kiss_tx_queue and not rig.sent


def test_multiple_pending_frames_drain_in_order(rig):
    submit(rig, info=b'first')
    submit(rig, kind=0, info=b'second')
    assert len(rig.server.kiss_tx_queue) == 2
    rig.clock.now = 2
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert rig.modem.service_kiss_queue() is None
    rig.clock.now = 4
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert [message.info for message in rig.sent] == [b'first', b'second']
    assert not rig.server.kiss_tx_queue


def test_packet_fit_is_validated_before_queued(rig):
    submit(rig, source='X-W9SSJ')
    assert not rig.replies and not rig.server.kiss_tx_queue and not rig.sent
    assert any('ignored' in message for _, message in rig.logs)
    submit(rig)
    assert len(rig.server.kiss_tx_queue) == 1


def test_raw_kiss_routes_same_queue_without_fabricated_admission(rig):
    submit(rig, kind=0)
    assert rig.replies == []
    rig.clock.now = 2
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert len(rig.sent) == 1 and rig.replies == bracket(rig.sent[0])


def test_received_ui_never_enters_contact_data_stream(rig):
    delivered = []
    rig.session._kiss = SimpleNamespace(deliver_frame=lambda *args: delivered.append(args))
    message = Broadcast(1, 'K1ABC', 'ALL', b'hello')
    _HostIO(rig.modem, rig.modem._transport).cq(message)
    assert delivered == [(1, message.encode())] and not rig.sent and not rig.replies
    assert rig.session._data.output == []


def test_transmit_exception_unkeys_and_logs_once(rig):
    def fail(audio):
        raise OSError('fake playback failure')
    rig.modem._transport.tx = fail
    submit(rig)
    rig.clock.now = 2
    with pytest.raises(OSError, match='fake playback'):
        rig.modem.service_kiss_queue()
    assert not rig.modem._ptt_active and not rig.server.kiss_tx_queue
    assert sum('PARTIAL' in message for _, message in rig.logs) == 1
    assert rig.modem.service_kiss_queue() is None
    assert sum('PARTIAL' in message for _, message in rig.logs) == 1


def test_kiss_delivery_preserves_command_and_escapes_binary():
    import threading
    from hfmodem.ax25 import kiss
    from hfmodem.kestrel.host.kiss import KissPort, _Client
    client = _Client()
    port = object.__new__(KissPort)
    port._clients = {object(): client}
    port._clock = lambda: 0.0
    port._lock = threading.Lock()
    port.deliver_frame(1, b'\xc0\xdbhello')
    assert list(kiss.Parser().feed(b''.join(client.output))) == [(0, 1, b'\xc0\xdbhello')]


def test_successful_tx_but_failed_unkey_drops_head(rig):
    rig.modem._transport.unkey_confirmed = lambda: False
    submit(rig)
    rig.clock.now = 2
    with pytest.raises(RuntimeError, match='unkey was not confirmed'):
        rig.modem.service_kiss_queue()
    assert len(rig.sent) == 1 and not rig.server.kiss_tx_queue
    assert sum('UNKEY_UNCONFIRMED' in message for _, message in rig.logs) == 1
    assert rig.modem._events.empty()


@pytest.mark.parametrize('info', [b'hello', b'x' * 70])
def test_broadcast_metadata_uses_actual_500hz_phy_not_configured_session(rig, info):
    rig.modem.set_bandwidth('2300')
    rig.modem.poll()
    notes = []
    rig.modem._transport.tx_metadata = lambda **fields: notes.append(fields)
    submit(rig, info=info)
    rig.clock.now = 2
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert len(notes) == 1
    assert notes[0]['bandwidth_hz'] == 500
    assert notes[0]['source'] == 'W9SSJ' and notes[0]['destination'] == 'ALL'
    level = notes[0]['host_bitrate_level']
    assert notes[0]['speed_level'] == level and level in (1, 2, 4)
    assert notes[0]['modulation'] == ('DBPSK' if level == 4 else 'index modulation')


def test_audio_unkey_confirmation_does_not_infer_from_key_call_return(tools):
    from kestrel_connect import AudioVaraIO
    audio = object.__new__(AudioVaraIO)
    audio.rig = SimpleNamespace(retired=False, keyed=False, line_held=True,
                               released_low=None, line_state=lambda: False)
    audio._timing_key_active = False
    assert audio.unkey_confirmed()
    audio._timing_key_active = True
    assert not audio.unkey_confirmed()
    audio._timing_key_active = False
    audio.rig.line_state = lambda: True
    assert not audio.unkey_confirmed()
    audio.rig.line_state = lambda: None
    assert not audio.unkey_confirmed()
    audio.rig.line_held, audio.rig.released_low = False, False
    assert not audio.unkey_confirmed()
    audio.rig.released_low = True
    assert audio.unkey_confirmed()


@pytest.mark.parametrize('source', ['K1ABC', 'X', '123456', 'AAR5ZZ', 'NNN0ABC-15'])
def test_other_source_and_changed_mycall_still_key(rig, source):
    submit(rig, source=source)
    rig.session._dispatch('MYCALL W9XYZ')
    rig.modem.poll()
    rig.clock.now = 2
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert len(rig.sent) == 1 and rig.sent[0].source == source


def test_failure_drops_only_head_and_preserves_later_request(rig):
    tx = rig.modem._transport.tx
    def boom(audio):
        raise OSError('transmit failed')
    rig.modem._transport.tx = boom
    submit(rig, info=b'first')
    submit(rig, info=b'second')
    rig.clock.now = 2
    with pytest.raises(OSError, match='transmit failed'):
        rig.modem.service_kiss_queue()
    assert len(rig.server.kiss_tx_queue) == 1
    assert sum('PARTIAL' in message for _, message in rig.logs) == 1
    rig.modem._transport.tx = tx
    assert rig.modem.service_kiss_queue() is None
    rig.clock.now = 4
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert [message.info for message in rig.sent] == [b'second']
    assert not rig.server.kiss_tx_queue


@pytest.mark.parametrize('command', [0, 1])
def test_transmit_preserves_path_general_fields_and_long_info(rig, command):
    from hfmodem.kestrel.tx.ui import packetize
    from hfmodem.kestrel.ui import UIAddress, UIFrame

    message = UIFrame(command, (UIAddress('APRS'), UIAddress('123456'),
                               UIAddress('WIDE2', 2, 0xE1)), 0x13, 0xCF, b'x' * 400)
    notes = []
    rig.modem._transport.tx_metadata = lambda **fields: notes.append(fields)
    rig.server._kiss_frame(command, message.ax25())
    rig.modem.set_mycall(['W9XYZ'])
    rig.modem.poll()
    rig.clock.now = 2
    assert rig.modem.service_kiss_queue() is Outcome.SENT
    assert rig.sent == [message]
    assert notes[0]['frame_count'] == len(packetize(message)[1])


@pytest.mark.parametrize('guard', ['linked', 'incoming', 'intent', 'pending', 'keyed', 'stop'])
def test_command2_preserves_beacon_admission_guards(rig, guard):
    if guard == 'linked':
        rig.modem.station.state = VaraState.CONNECTED
    elif guard == 'incoming':
        rig.modem._incoming_pending_at = 0
    elif guard == 'intent':
        rig.modem._intent_active = True
    elif guard == 'pending':
        rig.modem._cq_pending = True
    elif guard == 'keyed':
        rig.modem._ptt_active = True
    else:
        rig.modem.stop()
    with pytest.raises(ValueError):
        rig.server._kiss_text('20OEN63BCW9SSJ')
    assert not rig.server.kiss_tx_queue and not rig.sent and not rig.replies


@pytest.mark.parametrize('busy_at_write', [False, True])
def test_command2_is_dropped_if_busy_at_owner_poll(rig, busy_at_write):
    rig.modem._channel_busy = busy_at_write
    rig.server._kiss_text('20OEN63BCW9SSJ')
    assert rig.modem._cq_pending and not rig.server.kiss_tx_queue
    rig.modem._channel_busy = True
    rig.modem.poll()
    assert not rig.modem._cq_pending and not rig.replies
    rig.modem._channel_busy = False
    for now in (2, 20, 600):
        rig.clock.now = now
        rig.modem.poll()
        rig.modem.service_kiss_queue()
    assert not rig.sent and not rig.server.kiss_tx_queue


def test_command2_keys_once_without_ui_queue_holdoff(rig, monkeypatch):
    from hfmodem.kestrel.host import vara_modem
    monkeypatch.setattr(vara_modem, 'synth_beacon', lambda text: text)
    rig.server._kiss_text('20OEN63BCW9SSJ')
    assert not rig.server.kiss_tx_queue
    rig.modem.poll()
    assert rig.sent == ['20OEN63BCW9SSJ'] and rig.clock.now == 0
    assert rig.replies == ['PENDING', 'PTT ON', 'BITRATE (2)  41 bps TX',
                           'PTT OFF', 'CANCELPENDING']


def test_command2_without_an_application_retains_release_admission(rig):
    rig.server._session = None
    with pytest.raises(ValueError, match='no application attached'):
        rig.server._kiss_text('20OEN63BCW9SSJ')
    assert not rig.server.kiss_tx_queue and not rig.sent
