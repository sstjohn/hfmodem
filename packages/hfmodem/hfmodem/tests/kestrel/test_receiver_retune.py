# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fake owner and transport; no synthesis, audio devices or rig calls."""
from types import SimpleNamespace as NS
import pytest
import threading
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.host.vara_modem import _HostIO
from hfmodem.kestrel.vara.vara_arq import VaraState, VaraStationHandshake


def setup():
    events = []
    station = NS(state=VaraState.CONNECTED, payload_ledger=object(), _txq=[b'AWQ', b'hello'],
                 _answer_owed='release', retuned=lambda: events.append('discontinuity'))
    modem = NS(station=station, observer=object(), _stopping=False, _intent_active=True, _admission_closing=False, _lock=threading.RLock(), _ptt_active=False,
               received_cursor=16, received_ack_cursor=0)
    transport = NS(retuned=lambda: events.append('epoch'))
    runtime = AudioHostRuntime(transport)
    runtime.modem = modem
    request = runtime.request_retune(lambda: events.append('tune'), modem=modem,
        station=station, receive_cursor=16, unkeyed=lambda: True, valid=lambda: True)
    return runtime, request, modem, events


def test_ack_handback_and_unkey_precede_owner_tune_preserving_queued_text():
    runtime, request, modem, events = setup()
    ledger, queue = modem.station.payload_ledger, modem.station._txq
    assert not runtime._service_retune() and events == []
    modem.received_ack_cursor = 16
    assert not runtime._service_retune() and events == []
    # Peer handback completes the turn; our own ACK alone did not.
    modem.station._answer_owed = None
    modem._ptt_active = True
    assert not runtime._service_retune() and events == []
    modem._ptt_active = False
    request.unkeyed = lambda: False
    assert not runtime._service_retune() and events == []
    request.unkeyed = lambda: True
    assert runtime._service_retune()
    assert events == ['tune', 'epoch', 'discontinuity'] and request.done
    assert modem.station.payload_ledger is ledger and modem.station._txq is queue
    assert queue == [b'AWQ', b'hello'] and modem.station._answer_owed is None


@pytest.mark.parametrize('change', ['modem', 'station', 'link', 'disconnected', 'stopping', 'closing'])
def test_stale_transaction_does_not_tune_or_kill_owner(change):
    runtime, request, modem, events = setup()
    modem.received_ack_cursor = 16
    if change == 'modem':
        runtime.modem = None
    elif change == 'station':
        modem.station = NS()
    elif change == 'link':
        request.valid = lambda: False
    elif change == 'disconnected':
        modem.station.state = VaraState.DISCONNECTED
    elif change == 'closing':
        modem._admission_closing = True
    else:
        modem._stopping = True
    assert not runtime._service_retune()
    assert request.done and request.error and not events


def test_cancel_before_owner_consumes_is_final():
    runtime, request, modem, events = setup()
    assert request.cancel()
    modem.received_ack_cursor = 16
    assert not runtime._service_retune()
    assert not events and request.cancelled


def test_tune_failure_invalidates_old_rx_and_stops_owner():
    runtime, request, modem, events = setup()
    modem.received_ack_cursor = 16
    modem.station._answer_owed = None
    def fail():
        events.append('partial tune')
        raise OSError('CAT failure')
    request.callback = fail
    with pytest.raises(OSError, match='CAT failure'):
        runtime._service_retune()
    assert request.done and isinstance(request.error, OSError)
    assert events == ['partial tune', 'epoch', 'discontinuity']


def test_final_ack_hook_only_after_successful_unkey_and_on_retry(monkeypatch):
    events = []
    io = NS(received_over_acknowledged=lambda: events.append('ack fence'), log=lambda text: None)
    station = object.__new__(VaraStationHandshake)
    station.io = io
    station.caller = "KQ4QWH"
    station._over_answer = ('control', 1)
    station._reacks = 0
    station._progressed = lambda: None
    def response(frame):
        events.extend(['ACK', 'unkey'])
        return True
    station._tx_control_burst = response
    station._reanswer_over()
    assert events == ['ACK', 'unkey', 'ack fence']
    events.clear()
    station._tx_control_burst = lambda frame: False
    station._reanswer_over()
    assert events == []


def test_host_payload_cursor_precedes_notification_and_ack_cursor_follows():
    events = []
    modem = NS(received_cursor=0, received_ack_cursor=0)
    modem.observer = NS(modem_data_received=lambda payload: events.append(modem.received_cursor))
    io = _HostIO(modem, NS())
    io.data(b'Q:225')
    assert events == [5] and modem.received_ack_cursor == 0
    io.received_over_acknowledged()
    assert modem.received_ack_cursor == 5


def test_request_during_rx_is_executed_only_after_ack_callback_returns():
    events = []
    station = NS(state=VaraState.CONNECTED,
                 retuned=lambda: events.append('discontinuity'))
    modem = NS(station=station, observer=object(), _stopping=False, _intent_active=True, _admission_closing=False, _lock=threading.RLock(),
               _ptt_active=False, received_cursor=9, received_ack_cursor=0,
               poll=lambda: None)
    transport = NS(retuned=lambda: events.append('epoch'))
    runtime = AudioHostRuntime(transport)
    runtime.modem = modem
    def receive(timeout, hs):
        runtime.request_retune(lambda: events.append('tune'), modem=modem,
            station=station, receive_cursor=9, unkeyed=lambda: True, valid=lambda: True)
        events.extend(['DATA delivered', 'ACK keyed', 'ACK played', 'unkey'])
        modem.received_ack_cursor = 9
        events.append('RX callback returned')
        return None
    transport.next_rx_burst = receive
    runtime.step()
    assert events == ['DATA delivered', 'ACK keyed', 'ACK played', 'unkey',
                      'RX callback returned', 'tune', 'epoch', 'discontinuity']


def test_audio_retune_retires_only_old_frequency_samples(tools):
    import threading
    from kestrel_connect import AudioVaraIO
    events = []
    transport = object.__new__(AudioVaraIO)
    transport._lock = threading.Lock()
    transport._buf = [b'old']
    transport._consumed = 12
    transport._rx_generation = 3
    transport._seg = NS(restart=lambda: events.append('segment reset'))
    transport._band = NS(restart=lambda: events.append('busy reset'))
    transport._bursts = [b'old burst']
    transport._gap = b'old gap'
    transport._gap_from = 4
    transport._gap_lead = 6
    transport._rx_inflight = True
    transport.progress = NS(inflight_done=lambda direction: events.append(direction))
    transport.retuned()
    assert transport._rx_generation == 4 and transport._buf == []
    assert transport._bursts == [] and transport._consumed == 0
    assert transport._gap is None and transport._gap_from is None
    assert not transport._rx_inflight
    assert events == ['segment reset', 'busy reset', 'rx']


def test_initial_final_ack_cursor_excludes_refused_and_nonfinal_responses():
    events = []
    station = object.__new__(VaraStationHandshake)
    station.io = NS(received_over_acknowledged=lambda: events.append('fence'), log=lambda text: None)
    station._owe_nothing = lambda: None
    station._chat_profile = True
    station._txq = [b'queued']
    station._tx_control_burst = lambda frame: False
    station._tx_over_response(True)
    assert events == []
    station._tx_control_burst = lambda frame: True
    station._tx_over_response(True)
    assert events == ['fence']


def test_protocol_rx_reset_preserves_ledger_queue_and_recovery():
    import numpy as np
    from hfmodem.tests.kestrel.test_compression import station as new_station
    station, _ = new_station()
    ledger = station.payload_ledger
    station._txq = [b'AWQ', b'hello']
    queue = station._txq
    station._answer_owed = 'release'
    station._reacks = 2
    station._reack_frame = 'remembered'
    station._over_answer = ('control', 1)
    station._ans_buf = station._grant_buf = station._lv_buf = np.ones(16)
    station.retuned()
    assert station.payload_ledger is ledger and station._txq is queue
    assert station._answer_owed == 'release' and station._reacks == 2
    assert station._reack_frame == 'remembered' and station._over_answer == ('control', 1)
    assert len(station._ans_buf) == len(station._grant_buf) == len(station._lv_buf) == 0


def test_launcher_adapter_uses_owner_and_cancels_on_generator_close(tools):
    from varac_chat import received_retune_adapter
    runtime, first, modem, events = setup()
    first.cancel()
    runtime._service_retune()
    client = NS(link_generation=7, connected=True, closed=False)
    steps = received_retune_adapter(runtime, client, lambda hz: events.append(hz), lambda: True)(14107250, 7)
    assert next(steps) == .05 and not events
    steps.close()
    modem.received_ack_cursor = 16
    assert not runtime._service_retune() and not events


def test_new_payload_requires_its_ack_before_tune():
    runtime, request, modem, events = setup()
    modem.station._answer_owed = None
    modem.received_cursor = 20
    modem.received_ack_cursor = 16
    assert not runtime._service_retune() and not events
    modem.received_ack_cursor = 20
    assert runtime._service_retune()


@pytest.mark.parametrize('bw', ['500', '2300'])
def test_successful_final_ack_recovery_ladder_updates_fence(bw):
    events = []
    io = NS(received_over_acknowledged=lambda: events.append('fence'), log=lambda text: None)
    station = VaraStationHandshake(['W9SSJ'], io, bw=bw, chat=True)
    station.called = 'KQ4QWH'
    station._txq = []
    station._tx_pending = None
    station._reacks = 0
    station._tx_control_burst = lambda: False
    assert not station._reack_release() and events == []
    station._tx_control_burst = lambda: True
    assert station._reack_release() and events == ['fence']


def test_repeated_final_ack_does_not_substitute_for_peer_handback():
    runtime, request, modem, events = setup()
    io = _HostIO(modem, NS())
    for _ in range(3):
        io.received_over_acknowledged()
        assert modem.received_ack_cursor == modem.received_cursor
        assert not runtime._service_retune()
        assert not request.done and events == []
    modem.station._answer_owed = None
    assert runtime._service_retune() and request.done
    assert events == ['tune', 'epoch', 'discontinuity']
